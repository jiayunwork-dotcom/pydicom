# Copyright 2008-2024 pydicom authors. See LICENSE file for details.
"""Replayable data sources for reading DICOM datasets.

This module provides the machinery used to read datasets encoded using the
*Deflated Explicit VR Little Endian* transfer syntax
(``1.2.840.10008.1.2.1.99``).

A deflated dataset is a single raw DEFLATE stream (RFC 1951) that starts
immediately after the *File Meta Information* and expands to a normal
*Explicit VR Little Endian* dataset. Because raw DEFLATE streams are not
seekable, and because datasets may be very large, the stream is expanded
incrementally into a spool: an in-memory buffer for small datasets that rolls
over to a temporary file once it exceeds :data:`SPOOL_MEMORY_LIMIT`. Every
consumer reads the spool through its own independently positioned view.

The source is used by:

* :func:`pydicom.filereader.dcmread` (and therefore deferred element reads),
  and
* the path/file-like entry points of :func:`pydicom.pixels.pixel_array` and
  :func:`pydicom.pixels.iter_pixels`.

Expanding into a spool means:

* the compressed input is only ever read in chunks of at most
  :data:`MAX_INPUT_READ_SIZE` bytes,
* sequential consumers read each compressed byte exactly once, so the cost is
  linear in the size of the compressed stream,
* random access into the decompressed dataset first expands the spool up to
  the requested offset (so the first access to a late frame is linear in the
  bytes preceding it) and afterwards costs only the spool read,
* deferred values remain retrievable for as long as the source is open, and
* peak memory is bounded by the chunk sizes and the spool's in-memory limit,
  not by the size of the decompressed dataset.

The trade-off is temporary disk usage (up to the size of the decompressed
dataset) for datasets larger than :data:`SPOOL_MEMORY_LIMIT`. Replaying the
compressed input for every consumer instead would avoid the temporary file,
but would make every frame access re-read and re-expand the compressed prefix,
so the total work would grow quadratically with the number of frames. Loading
the compressed input or the decompressed dataset into memory is never done.

Lifetime
--------
A :class:`DeflateDataSource` is owned by whichever object created it:

* when reading a path with :func:`~pydicom.filereader.dcmread`, the returned
  :class:`~pydicom.dataset.FileDataset` owns the source and keeps it alive so
  that deferred values can still be retrieved. Call
  :meth:`Dataset.close()<pydicom.dataset.Dataset.close>` (or use the dataset
  as a context manager) to release it.
* when reading a caller-provided stream, that stream is *never* closed by the
  source; a file handle the source opens itself (only used to re-read a path
  whose original handle has been closed) is closed by the source.
* :func:`~pydicom.pixels.pixel_array` and
  :func:`~pydicom.pixels.iter_pixels` release their source before returning
  or when their iterator is closed/exhausted.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from typing import Any, BinaryIO
import zlib

from pydicom.errors import DataSourceClosedError, DeflateError

#: The maximum number of bytes requested from the compressed input in a
#: single ``read()`` call (1 MiB).
MAX_INPUT_READ_SIZE = 1024 * 1024

#: The maximum number of bytes of decompressed data produced by a single
#: ``zlib`` call (1 MiB). Together with :data:`MAX_INPUT_READ_SIZE` this
#: bounds the transient memory used while expanding the stream.
MAX_OUTPUT_CHUNK_SIZE = 1024 * 1024

#: The number of decompressed bytes a spool may hold in memory before it
#: rolls over to a temporary file (1 MiB).
SPOOL_MEMORY_LIMIT = 1024 * 1024


def _close_quietly(obj: Any) -> None:
    """Close `obj`, ignoring any errors (used during cleanup)."""
    try:
        obj.close()
    except Exception:
        pass


class DeflateDataSource:
    """A seekable, replayable view of a raw DEFLATE encoded dataset.

    The compressed input is expanded lazily into a spool, using at most
    :data:`MAX_INPUT_READ_SIZE` bytes of input and
    :data:`MAX_OUTPUT_CHUNK_SIZE` bytes of output at a time. Expansion is
    incremental: asking for a position expands only as far as that position
    (and, for reads that reach the end, far enough to know the decompressed
    length), so sequential consumers never expand a prefix twice.

    Parameters
    ----------
    stream : BinaryIO
        A seekable binary stream containing the raw DEFLATE data. The stream
        is not closed by the source unless it was opened by the source itself
        (see :attr:`owns_stream`).
    start : int, optional
        The offset in `stream` at which the DEFLATE data starts. Default
        ``0``.
    owns_stream : bool, optional
        If ``True`` then `stream` is owned by the source and will be closed
        when the source is closed. Default ``False`` (the caller keeps
        ownership, as is the case for streams passed to
        :func:`~pydicom.filereader.dcmread` and the pixel functions).
    name : str, optional
        The name to associate with the source and its readers.

    Attributes
    ----------
    stats : dict[str, int]
        Instrumentation for the compressed input: ``"read_calls"`` (number of
        ``read()`` requests), ``"bytes_read"`` (total bytes returned),
        ``"duplicate_bytes"`` (bytes read that had already been read before)
        and ``"random_seeks"`` (number of times a read had to seek backwards
        to a previously read position).
    """

    def __init__(
        self,
        stream: BinaryIO,
        start: int = 0,
        *,
        owns_stream: bool = False,
        name: str | None = None,
        reopen_path: str | None = None,
        reopen_factory: Callable[[str], BinaryIO] | None = None,
    ) -> None:
        self._stream: BinaryIO | None = stream
        self._start = start
        self._input_pos = start
        self._owns_stream = owns_stream
        self.name = name if name is not None else getattr(stream, "name", None)
        # Only set when `start` is an offset into `reopen_path`. Used to
        #   retrieve deferred values after the original stream has been
        #   closed by its owner.
        self.reopen_path = reopen_path
        # Called with `reopen_path` to re-open the input (defaults to open())
        self.reopen_factory = reopen_factory

        # The decompressor and spool
        self._decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
        self._spool = tempfile.SpooledTemporaryFile(
            max_size=SPOOL_MEMORY_LIMIT, mode="w+b"
        )
        self._size = 0
        self._stream_end = False  # decompressor reached the final block
        self._input_end = False  # compressed input returned no more bytes
        self._closed = False
        self._max_read_end = start
        # Input read from the compressed stream that has not been fed to the
        #   decompressor yet (kept when an expansion target is reached)
        self._pending = b""
        self.stats = {
            "read_calls": 0,
            "bytes_read": 0,
            "duplicate_bytes": 0,
            "random_seeks": 0,
        }

    @property
    def closed(self) -> bool:
        """Return ``True`` if the source has been closed."""
        return self._closed

    @property
    def size(self) -> int:
        """Return the number of decompressed bytes currently available.

        This does not expand the stream, so the value may be less than the
        final decompressed length.
        """
        return self._size

    @property
    def spool_rolled(self) -> bool:
        """Return ``True`` if the spool has rolled over to a temporary file."""
        return bool(getattr(self._spool, "_rolled", False))

    @property
    def spool_disk_bytes(self) -> int:
        """Return the number of bytes held by the temporary file (0 if none)."""
        if not self.spool_rolled or self._closed:
            return 0

        try:
            self._spool.flush()
            return os.fstat(self._spool.fileno()).st_size
        except (OSError, ValueError):
            return 0

    def open_reader(self) -> SpoolReader:
        """Return a new independently positioned readable view of the spool.

        The reader starts at position 0 and has its own ``read()``,
        ``seek()`` and ``tell()`` state, so multiple readers (for example two
        frame iterators) may be interleaved without affecting each other.
        Closing the reader does not close the source.
        """
        self._check_open()
        return SpoolReader(self)

    def take_ownership(self, stream: BinaryIO | None = None) -> None:
        """Take ownership of :attr:`stream` (or `stream` if given).

        Used by :func:`~pydicom.filereader.dcmread` when it opened the file
        for a dataset whose source still needs the input: the source will
        close it on :meth:`close`. If the input has already been fully
        consumed and released, the handle is closed immediately.
        """
        target = self._stream if stream is None else stream
        if self._input_end and self._stream is None:
            _close_quietly(target)
            return

        if stream is not None:
            self._stream = stream
        self._owns_stream = True

        if self._input_end:
            # Fully expanded already, no need to keep the handle
            _close_quietly(self._stream)
            self._stream = None

    def close(self) -> None:
        """Close the source and release its temporary resources.

        This is idempotent. The caller-provided stream (if any) is not closed.
        Any value that has not been read yet can no longer be retrieved.
        """
        if self._closed:
            return

        self._closed = True
        _close_quietly(self._spool)
        if self._owns_stream and self._stream is not None:
            _close_quietly(self._stream)

        self._stream = None

    def __enter__(self) -> DeflateDataSource:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def __del__(self) -> None:
        # Best-effort release if the owning dataset was dropped without close()
        try:
            self.close()
        except Exception:
            pass

    def _release_input(self) -> None:
        """Release the compressed input once it can no longer be needed.

        Expansion is append-only and every consumer reads the spool, so after
        the whole compressed stream has been expanded the input handle is no
        longer required. If the source owns the stream it is closed here; a
        caller-provided stream is left open (the caller owns it) and only the
        source's reference is dropped.
        """
        if self._input_end and self._stream is not None:
            if not self._owns_stream:
                # Leave caller-owned streams open; they may still be needed by
                #   the caller, but the source no longer needs the reference
                self._stream = None
                return

            _close_quietly(self._stream)
            self._stream = None

    def _check_open(self) -> None:
        if self._closed:
            raise DataSourceClosedError(
                "The dataset's data source has been closed and its value can "
                "no longer be retrieved"
            )

    # Expansion -----------------------------------------------------------

    def _read_input(self) -> bytes:
        """Read at most :data:`MAX_INPUT_READ_SIZE` compressed bytes."""
        if self._stream is not None and not getattr(self._stream, "closed", False):
            stream = self._stream
        elif self.reopen_path is not None:
            # The original stream has been closed, so fall back to re-opening
            #   the path. The new handle is owned by the source.
            try:
                if self.reopen_factory is not None:
                    stream = self.reopen_factory(self.reopen_path)
                else:
                    stream = open(self.reopen_path, "rb")
            except OSError as exc:
                raise DataSourceClosedError(
                    f"Unable to re-open the dataset's data source "
                    f"{self.reopen_path!r}: {exc}"
                ) from exc
            self._stream = stream
            self._owns_stream = True
        else:
            raise DataSourceClosedError(
                "The dataset's data source has been closed and its value can "
                "no longer be retrieved"
            )

        stream.seek(self._input_pos)
        data = stream.read(MAX_INPUT_READ_SIZE)
        if not data:
            self._input_end = True
            return b""

        # Track compression-read statistics
        start = self._input_pos
        end = start + len(data)
        self.stats["read_calls"] += 1
        self.stats["bytes_read"] += len(data)
        if start < self._max_read_end:
            overlap = min(end, self._max_read_end) - start
            self.stats["duplicate_bytes"] += max(overlap, 0)
            self.stats["random_seeks"] += 1
        self._max_read_end = max(self._max_read_end, end)
        self._input_pos = end

        return data

    def _write_output(self, data: bytes | bytearray) -> None:
        if not data:
            return

        self._spool.seek(self._size)
        self._spool.write(data)
        self._size += len(data)

    def _feed(self, chunk: bytes, target: int) -> None:
        """Expand `chunk` until the spool holds at least `target` bytes.

        ``zlib`` may stop early because of ``max_length``, leaving input in
        ``unconsumed_tail`` or output buffered internally. This method keeps
        going until the target is reached, the chunk is exhausted or the
        DEFLATE stream ends. Any input that produced more output than needed
        is stored in ``self._pending`` so it is not lost or re-read.
        """
        decompressor = self._decompressor
        pending = chunk
        while True:
            try:
                output = decompressor.decompress(pending, MAX_OUTPUT_CHUNK_SIZE)
            except zlib.error as exc:
                raise DeflateError(
                    f"Unable to decompress the dataset's Deflated Explicit VR "
                    f"Little Endian data: {exc}"
                ) from exc

            self._write_output(output)
            if decompressor.eof:
                # The DEFLATE stream is complete. Any remaining bytes are
                #   ignored, as zlib.decompress() does; the DICOM Standard
                #   permits trailing zero padding after the stream.
                self._stream_end = True
                self._pending = b""
                return

            tail = decompressor.unconsumed_tail
            if self._size >= target:
                # Enough has been produced; keep the rest of this chunk for
                #   the next expansion instead of reading it again
                self._pending = tail
                return

            if tail:
                # The output limit stopped us before all the input was used
                pending = tail
                continue

            if len(output) == MAX_OUTPUT_CHUNK_SIZE:
                # Output was capped by max_length; more may be buffered inside
                #   the decompressor, which is drained by passing empty input
                pending = b""
                continue

            self._pending = b""
            return

    def _expand(self, target: int) -> None:
        """Expand the spool until it holds at least `target` bytes.

        Expansion stops early if the compressed input is exhausted, in which
        case `target` may not be reached. Whether the shortfall is acceptable
        is left to the caller; the current length can be checked against
        :attr:`size`. Truncation (a stream that ends without the final DEFLATE
        block) and decompression errors are always raised.
        """
        self._check_open()

        while self._size < target and not self._input_end:
            if self._pending:
                chunk = self._pending
                self._pending = b""
            else:
                chunk = self._read_input()

            if chunk:
                self._feed(chunk, target)

        self._maybe_finish_input()

    def _maybe_finish_input(self) -> None:
        """Check for a truncated stream once the input has been exhausted."""
        if not self._input_end or self._stream_end:
            if self._input_end:
                self._release_input()
            return

        raise DeflateError(
            "The Deflated Explicit VR Little Endian data ended before the end "
            "of the DEFLATE stream (the compressed data is truncated)"
        )

    def expand_all(self) -> int:
        """Expand the complete stream and return the decompressed length.

        Raises
        ------
        DeflateError
            If the stream is truncated or cannot be decompressed.
        """
        self._check_open()
        while not self._input_end and not self._stream_end:
            if self._pending:
                chunk = self._pending
                self._pending = b""
            else:
                chunk = self._read_input()

            if chunk:
                self._feed(chunk, self._size + MAX_OUTPUT_CHUNK_SIZE)

        if not self._stream_end:
            raise DeflateError(
                "The Deflated Explicit VR Little Endian data ended before the "
                "end of the DEFLATE stream (the compressed data is truncated)"
            )

        self._release_input()
        return self._size

    def verify_complete(self) -> None:
        """Verify the DEFLATE stream ends properly after the data consumed so far.

        Draining the remaining compressed input confirms that the data which
        has already been consumed did not come from a truncated stream. This
        is used when a consumer has finished reading all the frames it was
        asked for and must not report success for truncated or corrupt input.

        Raises
        ------
        DeflateError
            If the remaining compressed data cannot be decompressed or the
            stream ends without the final DEFLATE block.
        """
        self.expand_all()

    def clone(self) -> DeflateDataSource:
        """Return an independent copy of the decompressed data source.

        The copy has its own spool and therefore its own lifetime: closing
        either source does not affect the other. The decompressed data is
        copied in bounded chunks, so this uses at most
        :data:`MAX_OUTPUT_CHUNK_SIZE` bytes of working memory (plus the
        copy's own spool, which may be a temporary file).

        Used when a :class:`~pydicom.dataset.FileDataset` that was read from a
        deflated file is deep-copied.
        """
        self.expand_all()

        clone = DeflateDataSource.__new__(DeflateDataSource)
        clone._stream = None
        clone._start = 0
        clone._input_pos = 0
        clone._owns_stream = False
        clone.name = self.name
        clone.reopen_path = self.reopen_path
        clone.reopen_factory = self.reopen_factory
        clone._decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
        clone._spool = tempfile.SpooledTemporaryFile(
            max_size=SPOOL_MEMORY_LIMIT, mode="w+b"
        )
        clone._size = 0
        clone._stream_end = True
        clone._input_end = True
        clone._closed = False
        clone._max_read_end = 0
        clone._pending = b""
        clone.stats = {
            "read_calls": 0,
            "bytes_read": 0,
            "duplicate_bytes": 0,
            "random_seeks": 0,
        }

        # Copy the expanded data in bounded chunks
        offset = 0
        while offset < self._size:
            chunk = self.read_at(offset, MAX_OUTPUT_CHUNK_SIZE)
            if not chunk:
                break
            clone._write_output(chunk)
            offset += len(chunk)

        return clone

    def read_at(self, offset: int, length: int) -> bytes:
        """Return up to `length` decompressed bytes starting at `offset`."""
        self._check_open()
        if length <= 0:
            return b""

        self._expand(offset + length)
        available = max(self._size - offset, 0)
        length = min(length, available)
        if length <= 0:
            return b""

        self._spool.seek(offset)
        return self._spool.read(length)

    # Internal helpers used by the tests/benchmarks -----------------------

    def expanded_length(self) -> int:
        """Expand the complete stream and return its length (see :meth:`expand_all`)."""
        return self.expand_all()


class SpoolReader:
    """An independently positioned readable view of a :class:`DeflateDataSource`.

    Instances are returned by :meth:`DeflateDataSource.open_reader` and behave
    like a read-only binary file object. Closing a reader only ends that
    reader; it never closes the underlying source or another reader.
    """

    def __init__(self, source: DeflateDataSource) -> None:
        self._source = source
        self.name = source.name
        self._pos = 0
        self._closed = False

    @property
    def closed(self) -> bool:
        """Return ``True`` if the reader or its source has been closed."""
        return self._closed or self._source.closed

    def readable(self) -> bool:
        """Return ``True``, the reader is readable."""
        return True

    def seekable(self) -> bool:
        """Return ``True``, the reader is seekable."""
        return True

    def tell(self) -> int:
        """Return the current position within the decompressed dataset."""
        return self._pos

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        """Set the position within the decompressed dataset."""
        if self.closed:
            raise DataSourceClosedError(
                "The dataset's data source has been closed and its value can "
                "no longer be retrieved"
            )

        if whence == os.SEEK_SET:
            pos = offset
        elif whence == os.SEEK_CUR:
            pos = self._pos + offset
        elif whence == os.SEEK_END:
            pos = self._source.expand_all() + offset
        else:
            raise ValueError(f"Invalid value for 'whence': {whence}")

        if pos < 0:
            raise ValueError("Negative seek position")

        self._pos = pos
        return self._pos

    def read(self, size: int = -1) -> bytes:
        """Read up to `size` decompressed bytes from the current position."""
        if self.closed:
            raise DataSourceClosedError(
                "The dataset's data source has been closed and its value can "
                "no longer be retrieved"
            )

        if size is None or size < 0:
            self._source.expand_all()
            data = self._source.read_at(self._pos, self._source.size - self._pos)
        else:
            data = self._source.read_at(self._pos, size)

        self._pos += len(data)
        return data

    def close(self) -> None:
        """Close this reader (does not close the source)."""
        self._closed = True

    def __getstate__(self) -> dict[str, Any]:
        # Readers aren't picklable (they wrap a temporary file), match the
        #   behaviour of a closed stream for copies
        raise TypeError("cannot pickle 'SpoolReader' instances")

    def __enter__(self) -> SpoolReader:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
