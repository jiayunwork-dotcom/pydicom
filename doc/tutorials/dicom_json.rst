============================
Introduction to JSON support
============================

Starting in *pydicom* version 1.3, some support for converting DICOM data to
and from JSON format has been added. This support is considered to be in
beta state, and the API is still subject to change.

Support for the JSON format has been added to the DICOM Standard in
Part 18 as the :dcm:`DICOM JSON Model<part18/chapter_F.html>`. The standard
describes how different DICOM value representations can be encoded in JSON.

Converting a dataset into JSON format
=====================================

*pydicom* supports the conversion of a DICOM dataset both into a JSON string
and into a deserialized JSON dictionary:

  >>> import pydicom
  >>> ds = pydicom.examples.ct
  >>> ds.to_json()
  '{"00080005": {"Value": ["ISO_IR 100"], "vr": "CS"}, "00080008": {"Value":...
  >>> ds.to_json_dict()
  {"00080005": {"Value": ["ISO_IR 100"], "vr": "CS"}, "00080008": {"Value":...

Which of these methods you need depends on your use case. The JSON string
format created by :func:`~pydicom.dataset.Dataset.to_json` can be used in
low-level APIs to serialize the dataset.
Higher-level Python APIs (like Django) often work directly with Python
dictionaries deserialized from a JSON string instead, so
:func:`~pydicom.dataset.Dataset.to_json_dict` can be more convenient here.

Creating a dataset from JSON
============================

Similar, a dataset can be created both from a JSON string and from a JSON
dictionary. There is only a single function to handle both cases:

  >>> from pydicom.dataset import Dataset
  >>> Dataset.from_json('{"00080005": {"Value": ["ISO_IR 100"], "vr": "CS"}}')
  (0008, 0005) Specific Character Set              CS: u'ISO_IR 100'
  >>> Dataset.from_json({"00080005": {"Value": ["ISO_IR 100"], "vr": "CS"}})
  (0008, 0005) Specific Character Set              CS: u'ISO_IR 100'

The conversion in both directions is symmetric:

  >>> import pydicom
  >>> ds = pydicom.examples.ct
  >>> ds_json = ds.to_json()
  >>> ds1 = pydicom.dataset.Dataset.from_json(ds_json)
  >>> assert ds == ds1


Working with large binary data
==============================

Large binary data can be handled in two ways. It can be encoded
:dcm:`inline<part18/sect_F.2.7.html>` as a base64-encoded string, or it can
be accessed via a :dcm:`BulkDataURI<part18/sect_F.2.6.html>` provided in the
JSON data, that provides the possibility to retrieve the data using the
`DICOMweb WADO-RS <https://www.dicomstandard.org/dicomweb/retrieve-wado-rs-and-wado-uri/>`_
standard.

If you don't provide additional arguments to the encoding functions, the
data is encoded inline. If you want to save or retrieve data using DICOMweb
WADO-RS, you have to provide a bulk data handler.

On writing JSON data, the bulk data handler is responsible to store the data
so it can be retrieved via the ``BulkDataURI`` saved in the JSON dataset.
Note that only data greater than ``bulk_data_threshold`` (by default set to
1024) is handled by the bulk data handler - smaller data is encoded inline.

  >>> import pydicom
  >>> def bulk_data_handler(data_element):
  >>>     uri = store_data_and_return_uri(data_element)
  >>>     return uri
  >>>
  >>> ds = pydicom.examples.ct
  >>> ds_json = ds.to_json(bulk_data_element_handler=bulk_data_handler)

On reading JSON data, the handler must be able to retrieve the data using
the stored ``BulkDataURI``:

  >>> def bulk_data_reader(bulk_data_uri):
  >>>     return data_retrieved_via_uri(bulk_data_uri)
  >>>
  >>> json_data = {
  >>>     "00091002": {"vr": "OB", "BulkDataURI": "https://my.wado.org/123"}
  >>> }
  >>> ds = Dataset.from_json(json_data, bulk_data_uri_handler=bulk_data_reader)

or, if you need to also know the tag and the vr, in addition to the stored
``BulkDataURI``:

  >>> def bulk_data_reader(tag, vr, bulk_data_uri):
  >>>     return data_retrieved_for_tag_and_vr_via_uri(tag, vr, bulk_data_uri)
  >>>
  >>> json_data = {
  >>>     "00091002": {"vr": "OB", "BulkDataURI": "https://my.wado.org/123"}
  >>> }
  >>> ds = Dataset.from_json(json_data, bulk_data_uri_handler=bulk_data_reader)

Deferring bulk data retrieval
=============================

.. versionadded:: 3.1

When a dataset is archived through DICOMweb, its metadata is available as
DICOM JSON while the bulk data values (such as *Pixel Data*) are only
available through their ``BulkDataURI``. Downloading those values can be
expensive, so it may be desirable to defer the retrieval until the value is
actually needed. Passing ``lazy_bulk_data=True`` to
:meth:`~pydicom.dataset.Dataset.from_json` keeps this order: the JSON is
parsed, ordinary attributes can be inspected and modified, and the bulk data
is only retrieved when a value is explicitly accessed or the dataset is
written to a DICOM file:

  >>> json_data = {
  >>>     "00100010": {"vr": "PN", "Value": [{"Alphabetic": "CITIZEN^Joan"}]},
  >>>     "7FE00010": {"vr": "OW", "BulkDataURI": "https://my.wado.org/123"},
  >>> }
  >>> ds = Dataset.from_json(
  >>>     json_data, bulk_data_uri_handler=bulk_data_streamer,
  >>>     lazy_bulk_data=True,
  >>> )

In lazy mode the ``bulk_data_uri_handler`` is **not** called while the JSON
is parsed (this also applies to elements nested in sequences). It must accept
the three arguments ``(tag, vr, uri)`` - with `tag` being the element tag as
an uppercase hex string such as ``"7FE00010"`` - and return a
``(stream, length)`` tuple for each call:

  >>> def bulk_data_streamer(tag, vr, uri):
  >>>     response = http_get(uri)  # open a new stream for this call
  >>>     return response.raw_stream, response.content_length

* ``stream`` is a newly opened readable binary stream for the value. It
  doesn't have to be seekable and may return fewer bytes than requested from
  ``read(n)``; an empty read signals the end of the stream. The stream is
  closed by *pydicom* once it has been consumed or when an error occurs.
* ``length`` is the number of payload bytes (without padding) as a
  non-negative :class:`int`. The payload shall be the native (uncompressed)
  value of the element in little endian byte order. For a VR of **OW** the
  length must be even.

Only elements with a VR of **OB**, **OW** or **UN** can be deferred; a
``BulkDataURI`` for any other VR raises a :class:`ValueError` when the JSON
is parsed, as does using ``lazy_bulk_data=True`` without a handler.

Elements with an unretrieved value are real
:class:`~pydicom.dataelem.DataElement` instances whose value is a
:class:`~pydicom.bulkdata.BulkDataReference`; use
:attr:`~pydicom.dataelem.DataElement.is_lazy_bulk` and
:attr:`~pydicom.dataelem.DataElement.bulk_data_uri` to inspect them. Reading
other elements, iterating the dataset, testing for element presence and
printing the dataset do not retrieve the bulk data, and neither do
:meth:`~pydicom.dataset.Dataset.to_json` and
:meth:`~pydicom.dataset.Dataset.to_json_dict` - an unretrieved value is
serialized with its original ``BulkDataURI`` regardless of
``bulk_data_threshold``. Replacing or deleting such an element discards the
reference without retrieving it.

The value is retrieved on explicit access only:

  >>> ds.PatientName  # ordinary attribute - no retrieval
  'CITIZEN^Joan'
  >>> pixel_data = ds[0x7FE00010].value  # retrieves and returns bytes
  >>> pixel_data = ds.PixelData  # keyword access also retrieves

Accessing the value retrieves it once and stores it as :class:`bytes` on that
element only; the other references are unaffected.

When the dataset is written with :func:`~pydicom.filewriter.dcmwrite` or
:meth:`~pydicom.dataset.Dataset.save_as`, unretrieved values are streamed
directly to the output in chunks of at most 1 MiB without being loaded into
memory, and the elements keep their unretrieved references afterwards (a
later write retrieves them again). This requires a file path or a seekable
binary file object as the output and the *Explicit VR Little Endian* or
*Implicit VR Little Endian* transfer syntax; other outputs and transfer
syntaxes are rejected with a :class:`ValueError` before any URI is accessed.
**OB** and **UN** values with an odd length are padded with a zero byte as
required by the DICOM Standard.

All retrieval failures - a handler that raises, an invalid length, a stream
that ends early, contains extra bytes or fails during ``read()`` - raise
:class:`~pydicom.bulkdata.BulkDataError`. The error message identifies the
URI, the element tag and, for nested elements, the containing sequence
items. The element keeps its unretrieved reference, so the operation can be
retried (for example with a new stream) once the problem is fixed. A failed
write may leave an incomplete output file and is never reported as
successful; a file object provided by the caller is not closed by *pydicom*.
