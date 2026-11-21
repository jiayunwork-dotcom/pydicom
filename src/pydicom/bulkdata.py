# Copyright 2008-2024 pydicom authors. See LICENSE file for details.
"""Lazy (deferred) retrieval of bulk data values referenced by a BulkDataURI.

This module implements the machinery used by
:meth:`Dataset.from_json(..., lazy_bulk_data=True)<pydicom.dataset.Dataset.from_json>`
to defer the retrieval of bulk data values referenced by a
:dcm:`BulkDataURI<part18/sect_F.2.6.html>` in the :dcm:`DICOM JSON Model
<part18/chapter_F.html>`.

With lazy bulk data, the JSON representation of a dataset is parsed without
retrieving the (potentially very large) bulk data values. Instead, each
affected :class:`~pydicom.dataelem.DataElement` holds a
:class:`BulkDataReference` that records where the value can be retrieved from.
The value is only retrieved when it is explicitly accessed (for example via
:attr:`~pydicom.dataelem.DataElement.value`) or when the dataset is written
to a DICOM file using :func:`~pydicom.filewriter.dcmwrite` or
:meth:`~pydicom.dataset.Dataset.save_as`, in which case the value is streamed
directly to the output without being loaded into memory.
"""

from collections.abc import Callable, Iterator
from typing import Any, BinaryIO, TypeAlias

from pydicom.tag import Tag

#: The maximum number of bytes requested in a single ``read()`` call when
#: consuming a bulk data stream (1 MiB).
MAX_BULK_DATA_CHUNK_SIZE = 1024 * 1024

#: The VRs that support lazily retrieved bulk data values.
LAZY_BULK_DATA_VRS = ("OB", "OW", "UN")


class BulkDataError(Exception):
    """Raised when retrieving or writing a lazy bulk data value fails.

    The exception message identifies the element tag and the BulkDataURI of
    the value that failed; when writing nested datasets, the position of the
    containing sequence item(s) is appended by the writer.
    """


#: The retrieval handler used with lazy bulk data: called with the element
#: tag (as an uppercase hex string such as "7FE00010"), the element VR and
#: the "BulkDataURI" value; must return a ``(stream, length)`` tuple, where
#: `stream` is a newly opened readable binary stream for the value and
#: `length` is the number of payload bytes (without padding) as a
#: non-negative :class:`int`.
BulkDataStreamHandlerType: TypeAlias = Callable[[str, str, str], "tuple[BinaryIO, int]"]


def _close_quietly(stream: Any) -> None:
    """Close `stream`, ignoring any errors."""
    try:
        stream.close()
    except Exception:
        pass


class BulkDataReference:
    """A reference to a bulk data value that has not been retrieved yet.

    An instance of this class is used as the value of a
    :class:`~pydicom.dataelem.DataElement` (with a VR of **OB**, **OW** or
    **UN**) created by :meth:`Dataset.from_json(..., lazy_bulk_data=True)
    <pydicom.dataset.Dataset.from_json>` for an element that has a
    "BulkDataURI" instead of an inline value. It records everything needed to
    retrieve the value later, without retrieving it.

    .. versionadded:: 3.1

    Attributes
    ----------
    tag : str
        The tag of the element the value belongs to, as an uppercase hex
        string such as ``"7FE00010"`` (the same form that is passed to the
        retrieval handler).
    vr : str
        The value representation of the element, one of ``"OB"``, ``"OW"``
        or ``"UN"``.
    uri : str
        The "BulkDataURI" the value can be retrieved from.
    handler : callable
        The retrieval handler, called as ``handler(tag, vr, uri)`` and
        returning a ``(stream, length)`` tuple. See
        :meth:`Dataset.from_json<pydicom.dataset.Dataset.from_json>` for the
        full handler contract.
    """

    def __init__(
        self,
        tag: str,
        vr: str,
        uri: str,
        handler: BulkDataStreamHandlerType,
    ) -> None:
        if vr not in LAZY_BULK_DATA_VRS:
            raise ValueError(
                f"A bulk data reference requires a VR of OB, OW or UN, got '{vr}'"
            )

        self.tag = tag
        self.vr = vr
        self.uri = uri
        self.handler = handler

    def _describe(self) -> str:
        """Return a description of the reference for use in error messages."""
        try:
            tag = Tag(int(self.tag, 16)) if isinstance(self.tag, str) else Tag(self.tag)
            tag_descr = str(tag)
        except Exception:
            tag_descr = repr(self.tag)

        return f"bulk data element {tag_descr} with URI '{self.uri}'"

    def open(self) -> "tuple[BinaryIO, int]":
        """Call the retrieval handler and return a validated ``(stream, length)``.

        The handler is called as ``handler(tag, vr, uri)``. The returned
        `length` is validated: it must be a non-negative :class:`int`
        (:class:`bool` is not accepted) and, for a VR of **OW**, must be even.
        If validation fails the returned stream is closed before raising.

        Returns
        -------
        tuple[BinaryIO, int]
            A newly opened readable binary stream and the number of payload
            bytes (without padding) to read from it. The caller is
            responsible for closing the stream.

        Raises
        ------
        BulkDataError
            If the handler raises, doesn't return a ``(stream, length)``
            tuple, or returns an invalid length.
        """
        try:
            result = self.handler(self.tag, self.vr, self.uri)
        except BulkDataError:
            raise
        except Exception as exc:
            raise BulkDataError(
                f"The bulk data handler raised an exception for "
                f"{self._describe()}: {exc}"
            ) from exc

        try:
            stream, length = result
        except (TypeError, ValueError) as exc:
            raise BulkDataError(
                f"The bulk data handler for {self._describe()} must return a "
                f"'(stream, length)' tuple, got {result!r}"
            ) from exc

        if not hasattr(stream, "read"):
            _close_quietly(stream)
            raise BulkDataError(
                f"The bulk data handler for {self._describe()} returned an "
                f"invalid stream of type {type(stream).__name__!r} - a "
                "readable binary stream is required"
            )

        if isinstance(length, bool) or not isinstance(length, int) or length < 0:
            _close_quietly(stream)
            raise BulkDataError(
                f"Invalid length {length!r} for {self._describe()}: the "
                "length must be a non-negative integer"
            )

        if self.vr == "OW" and length % 2:
            _close_quietly(stream)
            raise BulkDataError(
                f"Invalid odd length {length} for {self._describe()}: a "
                "value with a VR of OW must have an even length"
            )

        return stream, length

    def chunks(
        self,
        stream: BinaryIO,
        length: int,
        chunk_size: int = MAX_BULK_DATA_CHUNK_SIZE,
    ) -> Iterator[bytes]:
        """Yield the payload of `stream` in chunks, verifying the `length`.

        Reads exactly `length` bytes from `stream`, requesting at most
        `chunk_size` bytes per ``read()`` call. Short reads are retried; an
        empty read is treated as the end of the stream. After `length` bytes
        have been read, the stream is checked for unexpected trailing data.

        Parameters
        ----------
        stream : BinaryIO
            The stream to read from, as returned by :meth:`open`. The stream
            is *not* closed by this method.
        length : int
            The number of payload bytes to read, as returned by :meth:`open`.
        chunk_size : int, optional
            The maximum number of bytes to request in a single ``read()``
            call (default 1 MiB).

        Yields
        ------
        bytes
            The payload chunks; the concatenation of all chunks is exactly
            `length` bytes long.

        Raises
        ------
        BulkDataError
            If the stream ends early, contains more than `length` bytes, or
            its ``read()`` method raises.
        """
        remaining = length
        while remaining > 0:
            try:
                chunk = stream.read(min(chunk_size, remaining))
            except BulkDataError:
                raise
            except Exception as exc:
                raise BulkDataError(f"Error reading {self._describe()}: {exc}") from exc

            if not chunk:
                raise BulkDataError(
                    f"The stream for {self._describe()} ended after "
                    f"{length - remaining} of {length} declared bytes"
                )

            if len(chunk) > remaining:
                raise BulkDataError(
                    f"The stream for {self._describe()} returned more than "
                    f"the declared length of {length} bytes"
                )

            yield bytes(chunk)
            remaining -= len(chunk)

        # The stream must not contain more than the declared length
        try:
            extra = stream.read(1)
        except BulkDataError:
            raise
        except Exception as exc:
            raise BulkDataError(f"Error reading {self._describe()}: {exc}") from exc

        if extra:
            raise BulkDataError(
                f"The stream for {self._describe()} contains more than the "
                f"declared length of {length} bytes"
            )

    def read_all(self) -> bytes:
        """Retrieve and return the complete payload as :class:`bytes`.

        Opens a new stream via :meth:`open`, reads and verifies the whole
        payload and closes the stream (also on failure).

        Returns
        -------
        bytes
            The bulk data value, exactly `length` bytes long (without the
            padding byte used when writing odd-length **OB** or **UN**
            values).

        Raises
        ------
        BulkDataError
            If retrieval or verification fails. The reference remains
            usable, so a failed attempt can be retried.
        """
        stream, length = self.open()
        try:
            return b"".join(self.chunks(stream, length))
        finally:
            stream.close()

    def __eq__(self, other: Any) -> Any:
        if isinstance(other, BulkDataReference):
            return (self.tag, self.vr, self.uri) == (other.tag, other.vr, other.uri)

        return NotImplemented

    def __ne__(self, other: Any) -> Any:
        return not (self == other)

    def __repr__(self) -> str:
        return (
            f"BulkDataReference(tag={self.tag!r}, vr={self.vr!r}, " f"uri={self.uri!r})"
        )
