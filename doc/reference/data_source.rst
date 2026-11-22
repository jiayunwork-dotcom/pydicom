.. _api_data_source:

Replayable Data Sources (:mod:`pydicom.data_source`)
====================================================

.. currentmodule:: pydicom.data_source

Support for reading datasets encoded using the *Deflated Explicit VR Little
Endian* transfer syntax (``1.2.840.10008.1.2.1.99``).

A deflated dataset is a single raw DEFLATE stream (RFC 1951) that starts
immediately after the *File Meta Information* and expands to a normal
*Explicit VR Little Endian* dataset. The compressed stream is expanded
incrementally into a spool - an in-memory buffer for small datasets that rolls
over to a temporary file - and every consumer reads the spool through its own
independently positioned view.

This means:

* the compressed input is only requested in chunks of at most
  :data:`MAX_INPUT_READ_SIZE` bytes, and expansion produces at most
  :data:`MAX_OUTPUT_CHUNK_SIZE` bytes at a time,
* each compressed byte is read once for a sequentially consumed dataset, so
  the work is linear in the size of the compressed stream,
* deferred element values remain retrievable for as long as the dataset's data
  source is open,
* peak memory is bounded by the chunk sizes and :data:`SPOOL_MEMORY_LIMIT`
  rather than by the size of the decompressed dataset, at the cost of
  temporary disk space for datasets larger than the in-memory spool limit.

Source lifetime
---------------
When :func:`~pydicom.filereader.dcmread` reads a deflated dataset from a path
the returned :class:`~pydicom.dataset.FileDataset` owns the data source. Call
:meth:`Dataset.close()<pydicom.dataset.Dataset.close>` (or use the dataset as a
context manager) to release its temporary resources. Values that have already
been read are unaffected; accessing a deferred element that has not been read
yet raises :class:`~pydicom.errors.DataSourceClosedError`.

When reading from a caller-provided stream that stream is never closed by
pydicom.

:func:`~pydicom.pixels.pixel_array` and
:func:`~pydicom.pixels.iter_pixels` release their data source before returning
(or when their iterator is closed or exhausted).

.. autosummary::
   :toctree: generated/

   DeflateDataSource
   SpoolReader

Constants
---------

.. data:: MAX_INPUT_READ_SIZE

   The maximum number of bytes requested from the compressed input in a single
   ``read()`` call (1 MiB).

.. data:: MAX_OUTPUT_CHUNK_SIZE

   The maximum number of decompressed bytes produced by a single ``zlib``
   call, which bounds the transient memory used while expanding the stream
   (1 MiB).

.. data:: SPOOL_MEMORY_LIMIT

   The number of decompressed bytes the spool may hold in memory before it
   rolls over to a temporary file (1 MiB).
