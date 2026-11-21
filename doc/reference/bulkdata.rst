.. _api_bulkdata:

Lazy Bulk Data (:mod:`pydicom.bulkdata`)
========================================

.. module:: pydicom.bulkdata

Lazy (deferred) retrieval of bulk data values referenced by a
:dcm:`BulkDataURI<part18/sect_F.2.6.html>` in the DICOM JSON Model, as used
by :meth:`Dataset.from_json(..., lazy_bulk_data=True)
<pydicom.dataset.Dataset.from_json>`.

.. autosummary::
   :toctree: generated/

   BulkDataReference
   BulkDataError

Constants
---------

.. data:: MAX_BULK_DATA_CHUNK_SIZE

   The maximum number of bytes requested in a single ``read()`` call when
   consuming a bulk data stream (1 MiB).

.. data:: LAZY_BULK_DATA_VRS

   The VRs that support lazily retrieved bulk data values: ``OB``, ``OW``
   and ``UN``.
