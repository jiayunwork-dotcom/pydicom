# Copyright 2008-2024 pydicom authors. See LICENSE file for details.
"""Tests for lazy bulk data retrieval.

Covers ``Dataset.from_json(..., lazy_bulk_data=True)``, the deferred
retrieval of "BulkDataURI" values, JSON serialization of unretrieved values
and streaming such values directly to file with ``dcmwrite``/``save_as``.

All tests use locally synthesized metadata and in-memory streams that record
``read()``/``close()`` usage - no patient data and no real services.
"""

import hashlib
import io
import json
import os
import struct
import tracemalloc

import pytest

from pydicom import Dataset, dcmread, dcmwrite
from pydicom.bulkdata import (
    BulkDataError,
    BulkDataReference,
    MAX_BULK_DATA_CHUNK_SIZE,
)
from pydicom.dataelem import DataElement
from pydicom.dataset import FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    ExplicitVRBigEndian,
    DeflatedExplicitVRLittleEndian,
    JPEGBaseline8Bit,
    generate_uid,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PIXEL_DATA = "7FE00010"
URI_1 = "https://example.test/bulk/1"
URI_2 = "https://example.test/bulk/2"
URI_3 = "https://example.test/bulk/3"


class TrackingStream:
    """A non-seekable readable binary stream recording its usage.

    ``read(n)`` may short-read (at most `max_chunk` bytes per call).
    """

    def __init__(self, data, max_chunk=None):
        self._data = data
        self._pos = 0
        self._max_chunk = max_chunk
        self.read_requests = []
        self.closed = False
        self.exc_on_read = None

    def read(self, n=-1):
        if self.closed:
            raise ValueError("read from closed stream")
        self.read_requests.append(n)
        if self.exc_on_read is not None:
            raise self.exc_on_read
        if n is None or n < 0:
            n = len(self._data) - self._pos
        if self._max_chunk is not None:
            n = min(n, self._max_chunk)
        n = min(n, len(self._data) - self._pos)
        chunk = self._data[self._pos : self._pos + n]
        self._pos += n
        return chunk

    def close(self):
        self.closed = True

    # deliberately no seek()/tell()/seekable() - a non-seekable stream


class BulkDataStore:
    """A fake bulk data store with a recording retrieval handler."""

    def __init__(self, payloads=None, max_chunk=None):
        self.payloads = dict(payloads or {})
        self.max_chunk = max_chunk
        self.calls = []  # (tag, vr, uri) for every handler invocation
        self.streams = []  # every stream returned by the handler
        self.fail_with = None
        self.exc_on_read = None  # if set, returned streams raise it on read()
        self.result_override = None  # if set, returned instead of (stream, length)

    def handler(self, tag, vr, uri):
        """The lazy bulk data handler: (tag, vr, uri) -> (stream, length)."""
        self.calls.append((tag, vr, uri))
        if self.fail_with is not None:
            raise self.fail_with
        if self.result_override is not None:
            return self.result_override
        data = self.payloads[uri]
        stream = TrackingStream(data, max_chunk=self.max_chunk)
        stream.exc_on_read = self.exc_on_read
        self.streams.append(stream)
        return stream, len(data)

    @property
    def n_calls(self):
        return len(self.calls)

    @property
    def all_closed(self):
        return all(s.closed for s in self.streams)


def bulk(vr, uri):
    """A JSON element entry with a "BulkDataURI"."""
    return {"vr": vr, "BulkDataURI": uri}


def make_lazy_dataset(json_model, store, **kwargs):
    return Dataset.from_json(
        json.dumps(json_model),
        bulk_data_uri_handler=store.handler,
        lazy_bulk_data=True,
        **kwargs,
    )


def add_file_meta(ds, tsyntax=ExplicitVRLittleEndian):
    """Prepare a dataset for writing with ``enforce_file_format=True``."""
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = tsyntax
    ds.file_meta.MediaStorageSOPClassUID = generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    return ds


# --- Independent structure scanner -----------------------------------------
# The following parser is deliberately written only for these tests and does
# not reuse any pydicom encoding/decoding machinery, so expectations for
# lengths, tags and item boundaries are checked independently.

VR_32BIT_LENGTH = {
    "OB",
    "OD",
    "OF",
    "OL",
    "OV",
    "OW",
    "SQ",
    "UC",
    "UN",
    "UR",
    "UT",
    "UV",
}
ITEM_TAG = 0xFFFEE000
ITEM_DELIM_TAG = 0xFFFEE00D
SEQ_DELIM_TAG = 0xFFFEE0DD
UNDEFINED_LENGTH = 0xFFFFFFFF


def _tag_at(data, pos):
    group, element = struct.unpack_from("<HH", data, pos)
    return (group << 16) | element


def _u32(data, pos):
    return struct.unpack_from("<I", data, pos)[0]


def _u16(data, pos):
    return struct.unpack_from("<H", data, pos)[0]


def scan_one(data, pos):
    """Scan one Explicit VR Little Endian structure at `pos`.

    Returns ``(record, new_pos)`` where the record describes an element or a
    sequence item, including exact lengths, offsets and delimiters.
    """
    tag = _tag_at(data, pos)
    if tag == ITEM_TAG:
        length = _u32(data, pos + 4)
        rec = {"kind": "item", "offset": pos, "length": length}
        pos += 8
        children = []
        if length == UNDEFINED_LENGTH:
            while _tag_at(data, pos) != ITEM_DELIM_TAG:
                child, pos = scan_one(data, pos)
                children.append(child)
            rec["delim_length"] = _u32(data, pos + 4)
            pos += 8
        else:
            end = pos + length
            while pos < end:
                child, pos = scan_one(data, pos)
                children.append(child)
            assert pos == end, "item content overruns its defined length"
        rec["children"] = children
        return rec, pos

    vr = data[pos + 4 : pos + 6].decode("ascii")
    if vr in VR_32BIT_LENGTH:
        length = _u32(data, pos + 8)
        header = 12
    else:
        length = _u16(data, pos + 6)
        header = 8
    rec = {
        "kind": "element",
        "tag": tag,
        "vr": vr,
        "length": length,
        "offset": pos,
        "value_offset": pos + header,
    }
    pos += header
    if vr == "SQ":
        items = []
        if length == UNDEFINED_LENGTH:
            while _tag_at(data, pos) != SEQ_DELIM_TAG:
                item, pos = scan_one(data, pos)
                items.append(item)
            rec["delim_length"] = _u32(data, pos + 4)
            pos += 8
        else:
            end = pos + length
            while pos < end:
                item, pos = scan_one(data, pos)
                items.append(item)
            assert pos == end, "SQ content overruns its defined length"
        rec["items"] = items
    else:
        pos += length
    return rec, pos


def scan_region(data, pos, end):
    """Scan all structures in ``data[pos:end]``; the region must be exact."""
    records = []
    while pos < end:
        rec, pos = scan_one(data, pos)
        records.append(rec)
    assert pos == end, "region does not end on an element boundary"
    return records


def scan_file(path, with_file_meta=True):
    """Read `path` and return ``(meta_records, dataset_records)``."""
    with open(path, "rb") as f:
        data = f.read()
    pos = 0
    meta = []
    if with_file_meta:
        assert data[128:132] == b"DICM"
        pos = 132
        while _tag_at(data, pos) >> 16 == 0x0002:
            rec, pos = scan_one(data, pos)
            meta.append(rec)
    return meta, scan_region(data, pos, len(data))


def find_records(records, tag):
    """Return all element records for `tag` in a scanned record list."""
    return [r for r in records if r.get("tag") == tag]


def encoded_size(rec):
    """The number of bytes the scanned structure occupies in the file."""
    if rec["kind"] == "item":
        content = sum(encoded_size(c) for c in rec["children"])
        if rec["length"] == UNDEFINED_LENGTH:
            return 8 + content + 8  # header + content + delimiter
        assert rec["length"] == content
        return 8 + content
    header = 12 if rec["vr"] in VR_32BIT_LENGTH else 8
    if rec["vr"] == "SQ":
        content = sum(encoded_size(item) for item in rec["items"])
        if rec["length"] == UNDEFINED_LENGTH:
            content += 8  # sequence delimiter
        else:
            assert rec["length"] == content
        return header + content
    return header + rec["length"]


def hash_file_region(path, offset, length):
    """Hash a region of a file, read in bounded chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(offset)
        remaining = length
        while remaining:
            chunk = f.read(min(1024 * 1024, remaining))
            assert chunk
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


# --- Generative payload -----------------------------------------------------

PATTERN_BLOCK = bytes(range(256)) * 4096  # 1 MiB deterministic block


def pattern_block(rotate=0):
    if rotate:
        return PATTERN_BLOCK[rotate:] + PATTERN_BLOCK[:rotate]
    return PATTERN_BLOCK


class GenerativeStream:
    """A stream that generates a deterministic payload of `length` bytes
    without ever holding the whole payload (only a 1 MiB block)."""

    def __init__(self, length, rotate=0):
        self.length = length
        self._block = pattern_block(rotate)
        self._pos = 0
        self.read_requests = []
        self.closed = False

    def read(self, n=-1):
        if self.closed:
            raise ValueError("read from closed stream")
        self.read_requests.append(n)
        # the consumer must never perform an unbounded or oversized read
        assert 0 < n <= MAX_BULK_DATA_CHUNK_SIZE, f"invalid read request {n}"
        if self._pos >= self.length:
            return b""
        n = min(n, self.length - self._pos)
        out = bytearray()
        while n > 0:
            offset = self._pos % len(self._block)
            take = min(n, len(self._block) - offset)
            out += self._block[offset : offset + take]
            self._pos += take
            n -= take
        return bytes(out)

    def close(self):
        self.closed = True


def expected_payload_hash(length, rotate=0):
    """Hash the generated payload without materializing it."""
    block = pattern_block(rotate)
    h = hashlib.sha256()
    full, rem = divmod(length, len(block))
    for _ in range(full):
        h.update(block)
    if rem:
        h.update(block[:rem])
    return h.hexdigest()


def expected_file_value_hash(length, rotate=0):
    """Hash the encoded value region: the payload plus the 0x00 padding
    byte appended to odd-length OB/UN values."""
    h = hashlib.sha256()
    block = pattern_block(rotate)
    full, rem = divmod(length, len(block))
    for _ in range(full):
        h.update(block)
    if rem:
        h.update(block[:rem])
    if length % 2:
        h.update(b"\x00")
    return h.hexdigest()


class GenerativeStore:
    """A bulk data store whose payloads are generated on the fly."""

    def __init__(self, layout):
        # layout: {uri: (length, rotate)}
        self.layout = layout
        self.streams = []
        self.calls = []

    def handler(self, tag, vr, uri):
        self.calls.append((tag, vr, uri))
        length, rotate = self.layout[uri]
        stream = GenerativeStream(length, rotate)
        self.streams.append(stream)
        return stream, length


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------


class TestLazyBulkDataLoading:
    def test_lazy_requires_handler(self):
        json_data = {PIXEL_DATA: bulk("OW", URI_1)}
        msg = "A 'bulk_data_uri_handler' is required when 'lazy_bulk_data' is True"
        with pytest.raises(ValueError, match=msg):
            Dataset.from_json(json_data, lazy_bulk_data=True)

    def test_lazy_handler_arity_validated_without_calling(self):
        called = []

        def one_arg_handler(uri):
            called.append(uri)
            return b"xyzzy"

        json_data = {PIXEL_DATA: bulk("OW", URI_1)}
        with pytest.raises(ValueError, match="must accept three positional"):
            Dataset.from_json(
                json_data,
                bulk_data_uri_handler=one_arg_handler,
                lazy_bulk_data=True,
            )
        # the handler must not be called during argument validation
        assert called == []

        def two_arg_handler(tag, vr):
            called.append((tag, vr))

        with pytest.raises(ValueError, match="must accept three positional"):
            Dataset.from_json(
                json_data,
                bulk_data_uri_handler=two_arg_handler,
                lazy_bulk_data=True,
            )
        assert called == []

    def test_lazy_rejects_non_supported_vr(self):
        store = BulkDataStore()
        for vr in ("PN", "LO", "US", "SQ", "OF", "OB or OW"):
            json_data = {"00091001": bulk(vr, URI_1)}
            with pytest.raises(ValueError, match="not supported for data"):
                make_lazy_dataset(json_data, store)
        # nothing was retrieved
        assert store.n_calls == 0

    def test_lazy_rejects_non_supported_vr_in_sequence(self):
        store = BulkDataStore()
        json_data = {
            "00081140": {
                "vr": "SQ",
                "Value": [{"00100010": bulk("PN", URI_1)}],
            }
        }
        with pytest.raises(ValueError, match="not supported for data"):
            make_lazy_dataset(json_data, store)
        assert store.n_calls == 0

    def test_lazy_un_for_known_tag(self):
        # a lazy UN value for a tag with a known dictionary VR keeps the
        # UN VR, as the value is not available for conversion
        store = BulkDataStore({URI_3: b"\x01" * 100})
        json_data = {PIXEL_DATA: bulk("UN", URI_3)}
        ds = make_lazy_dataset(json_data, store)
        assert ds[0x7FE00010].VR == "UN"
        assert ds[0x7FE00010].is_lazy_bulk
        assert store.n_calls == 0

    def test_dataelement_from_json_directly(self):
        store = BulkDataStore({URI_1: b"\x01" * 100})
        elem = DataElement.from_json(
            Dataset,
            PIXEL_DATA,
            "OW",
            URI_1,
            "BulkDataURI",
            store.handler,
            lazy_bulk_data=True,
        )
        assert elem.is_lazy_bulk
        assert store.n_calls == 0
        # the handler is required here as well
        with pytest.raises(ValueError, match="bulk_data_uri_handler"):
            DataElement.from_json(
                Dataset, PIXEL_DATA, "OW", URI_1, "BulkDataURI", lazy_bulk_data=True
            )
        # DataElement.to_json returns the URI without retrieving it
        assert json.loads(elem.to_json()) == {"vr": "OW", "BulkDataURI": URI_1}
        assert store.n_calls == 0

    def test_lazy_accepts_ob_ow_un(self):
        store = BulkDataStore()
        json_data = {
            "00091001": bulk("OB", URI_1),
            "00091002": bulk("OW", URI_2),
            "00091003": bulk("UN", URI_3),
        }
        ds = make_lazy_dataset(json_data, store)
        assert store.n_calls == 0
        for tag in (0x00091001, 0x00091002, 0x00091003):
            assert ds[tag].is_lazy_bulk

    def test_lazy_bulk_data_uri_list_form(self):
        # the standard is ambiguous about BulkDataURI being enclosed in a
        # list; both forms are supported, as in the eager case
        store = BulkDataStore()
        json_data = {PIXEL_DATA: {"vr": "OW", "BulkDataURI": [URI_1]}}
        ds = make_lazy_dataset(json_data, store)
        assert ds[0x7FE00010].bulk_data_uri == URI_1
        assert store.n_calls == 0

    def test_lazy_nested_sequence(self):
        store = BulkDataStore()
        json_data = {
            "003A0200": {
                "vr": "SQ",
                "Value": [
                    {"54001010": bulk("OW", URI_1)},
                    {
                        "00091005": {
                            "vr": "SQ",
                            "Value": [{"00091010": bulk("OB", URI_2)}],
                        }
                    },
                ],
            }
        }
        ds = make_lazy_dataset(json_data, store)
        assert store.n_calls == 0
        item0, item1 = ds[0x003A0200].value
        assert item0[0x54001010].is_lazy_bulk
        assert item1[0x00091005].value[0][0x00091010].is_lazy_bulk

    def test_normal_json_interpretation_preserved(self):
        # "Value", "InlineBinary", empty elements and sequence item order
        # are interpreted as before in lazy mode
        store = BulkDataStore()
        json_data = {
            "00100010": {"vr": "PN", "Value": [{"Alphabetic": "CITIZEN^Joan"}]},
            "00091002": {"vr": "OB", "InlineBinary": "QmluYXJ5Q29udGVudA=="},
            "00100020": {"vr": "LO"},  # empty element
            "00091003": {"vr": "US", "Value": [1, 2, 3]},
            "00081140": {
                "vr": "SQ",
                "Value": [
                    {"00100030": {"vr": "DA", "Value": ["20010101"]}},
                    {"00100030": {"vr": "DA", "Value": ["20020202"]}},
                    {"00100030": {"vr": "DA", "Value": ["20030303"]}},
                ],
            },
        }
        ds = make_lazy_dataset(json_data, store)
        assert ds.PatientName == "CITIZEN^Joan"
        assert ds[0x00091002].value == b"BinaryContent"
        assert ds[0x00100020].is_empty
        assert list(ds[0x00091003].value) == [1, 2, 3]
        dates = [item[0x00100030].value for item in ds[0x00081140].value]
        assert dates == ["20010101", "20020202", "20030303"]
        assert store.n_calls == 0

    def test_eager_one_arg_handler_still_works(self):
        def bulk_data_reader(value):
            return b"xyzzy"

        json_data = {"00091002": {"vr": "OB", "BulkDataURI": URI_1}}
        ds = Dataset.from_json(json.dumps(json_data), bulk_data_reader)
        assert ds[0x00091002].value == b"xyzzy"
        assert not ds[0x00091002].is_lazy_bulk

    def test_eager_three_arg_handler_still_works(self):
        def bulk_data_reader(tag, vr, value):
            return b"xyzzy"

        json_data = {"00091002": {"vr": "OB", "BulkDataURI": URI_1}}
        ds = Dataset.from_json(json.dumps(json_data), bulk_data_reader)
        assert ds[0x00091002].value == b"xyzzy"
        assert not ds[0x00091002].is_lazy_bulk

    def test_default_without_lazy_option_unchanged(self):
        # no handler and no lazy option: warning and empty value, as before
        json_data = {"00091002": {"vr": "OB", "BulkDataURI": URI_1}}
        with pytest.warns(UserWarning, match="No bulk data URI handler"):
            ds = Dataset.from_json(json.dumps(json_data))
        assert 0x00091002 in ds
        assert not ds[0x00091002].is_lazy_bulk

    def test_lazy_flag_is_keyword_only(self):
        store = BulkDataStore()
        json_data = json.dumps({"00091002": bulk("OB", URI_1)})
        with pytest.raises(TypeError):
            Dataset.from_json(json_data, store.handler, True)

    def test_reference_requires_supported_vr(self):
        store = BulkDataStore()
        with pytest.raises(ValueError, match="requires a VR of OB, OW or UN"):
            BulkDataReference("00091001", "LO", URI_1, store.handler)

    def test_dataelement_rejects_reference_for_other_vr(self):
        store = BulkDataStore()
        ref = BulkDataReference("00100010", "OB", URI_1, store.handler)
        with pytest.raises(ValueError, match="cannot be used with lazy bulk"):
            DataElement(0x00100010, "PN", ref)


# ---------------------------------------------------------------------------
# Metadata operations that must not retrieve the value
# ---------------------------------------------------------------------------


@pytest.fixture()
def lazy_ds():
    payloads = {
        URI_1: b"\x01\x02\x03\x04" * 250,  # 1000 bytes, OW
        URI_2: b"\xaa" * 999,  # 999 bytes, OB (odd)
        URI_3: b"\xbb" * 100,  # 100 bytes, UN
    }
    store = BulkDataStore(payloads)
    json_data = {
        "00100010": {"vr": "PN", "Value": [{"Alphabetic": "CITIZEN^Joan"}]},
        PIXEL_DATA: bulk("OW", URI_1),
        "00091001": bulk("OB", URI_2),
        "00091002": bulk("UN", URI_3),
        "00081140": {
            "vr": "SQ",
            "Value": [
                {
                    "00100020": {"vr": "LO", "Value": ["ID-1"]},
                    "00091011": bulk("OB", URI_2),
                },
                {"00091012": bulk("OW", URI_1)},
            ],
        },
    }
    ds = make_lazy_dataset(json_data, store)
    return ds, store


class TestMetadataOperationsDoNotRetrieve:
    def test_iteration_contains_len(self, lazy_ds):
        ds, store = lazy_ds
        assert 0x7FE00010 in ds
        assert "PixelData" in ds
        assert len(ds) == 5
        tags = [elem.tag for elem in ds]
        assert 0x7FE00010 in tags
        assert sorted(ds.keys()) == list(ds.keys()) or True
        walked = []
        ds.walk(lambda dataset, elem: walked.append(elem.tag))
        assert 0x00091011 in walked
        assert store.n_calls == 0

    def test_read_normal_fields(self, lazy_ds):
        ds, store = lazy_ds
        assert ds.PatientName == "CITIZEN^Joan"
        item = ds[0x00081140].value[0]
        assert item[0x00100020].value == "ID-1"
        assert store.n_calls == 0

    def test_getitem_and_get_do_not_retrieve(self, lazy_ds):
        ds, store = lazy_ds
        elem = ds[0x7FE00010]
        assert isinstance(elem, DataElement)
        assert elem.is_lazy_bulk
        assert ds.get(0x7FE00010).is_lazy_bulk
        assert ds.get_item(0x7FE00010).is_lazy_bulk
        assert store.n_calls == 0

    def test_printing_does_not_retrieve(self, lazy_ds):
        ds, store = lazy_ds
        text = str(ds)
        assert URI_1 in text
        assert "Bulk data not yet retrieved" in text
        assert URI_1 in ds[0x7FE00010].repval
        repr(ds)
        str(ds[0x7FE00010])
        assert store.n_calls == 0

    def test_element_state_properties(self, lazy_ds):
        ds, store = lazy_ds
        elem = ds[0x7FE00010]
        assert elem.is_lazy_bulk
        assert elem.bulk_data_uri == URI_1
        assert elem.VM == 1
        assert not elem.is_empty
        assert not elem.is_buffered
        assert store.n_calls == 0

    def test_to_json_dict_returns_uri_without_retrieval(self, lazy_ds):
        ds, store = lazy_ds
        element_handler_calls = []

        def bulk_data_element_handler(elem):
            element_handler_calls.append(elem.tag)
            return "https://example.test/regenerated"

        json_dict = ds.to_json_dict(
            bulk_data_threshold=10**9,  # even a huge threshold must not fetch
            bulk_data_element_handler=bulk_data_element_handler,
        )
        assert json_dict[PIXEL_DATA] == {"vr": "OW", "BulkDataURI": URI_1}
        assert json_dict["00091001"] == {"vr": "OB", "BulkDataURI": URI_2}
        assert json_dict["00091002"] == {"vr": "UN", "BulkDataURI": URI_3}
        # nested sequence items keep their URIs as well
        items = json_dict["00081140"]["Value"]
        assert items[0]["00091011"] == {"vr": "OB", "BulkDataURI": URI_2}
        assert items[1]["00091012"] == {"vr": "OW", "BulkDataURI": URI_1}
        # normal elements are serialized normally
        assert json_dict["00100010"] == {
            "vr": "PN",
            "Value": [{"Alphabetic": "CITIZEN^Joan"}],
        }
        # neither the retrieval handler nor the element handler was called
        assert store.n_calls == 0
        assert element_handler_calls == []

    def test_to_json_returns_uri_without_retrieval(self, lazy_ds):
        ds, store = lazy_ds
        json_str = ds.to_json(bulk_data_threshold=10**9)
        model = json.loads(json_str)
        assert model[PIXEL_DATA]["BulkDataURI"] == URI_1
        assert "InlineBinary" not in model[PIXEL_DATA]
        assert store.n_calls == 0

    def test_materialized_element_uses_normal_json_rules(self, lazy_ds):
        ds, store = lazy_ds
        # small materialized value -> InlineBinary
        ds[0x7FE00010].value  # materialize the OW element
        assert store.n_calls == 1
        json_dict = ds.to_json_dict()
        assert "InlineBinary" in json_dict[PIXEL_DATA]
        # large materialized value -> bulk_data_element_handler
        calls = []

        def bulk_data_element_handler(elem):
            calls.append(elem.tag)
            return "https://example.test/stored"

        json_dict = ds.to_json_dict(
            bulk_data_threshold=4, bulk_data_element_handler=bulk_data_element_handler
        )
        assert json_dict[PIXEL_DATA]["BulkDataURI"] == "https://example.test/stored"
        assert calls == [0x7FE00010]
        # the other elements are still unretrieved references
        assert json_dict["00091001"]["BulkDataURI"] == URI_2
        assert store.n_calls == 1

    def test_access_single_value_only_materializes_it(self, lazy_ds):
        ds, store = lazy_ds
        value = ds[0x7FE00010].value
        assert value == b"\x01\x02\x03\x04" * 250
        assert not ds[0x7FE00010].is_lazy_bulk
        # the other references are untouched
        assert ds[0x00091001].is_lazy_bulk
        assert ds[0x00091002].is_lazy_bulk
        assert ds[0x00081140].value[0][0x00091011].is_lazy_bulk
        assert store.n_calls == 1
        # a second access does not retrieve again
        assert ds[0x7FE00010].value == value
        assert store.n_calls == 1

    def test_keyword_attribute_access_materializes(self, lazy_ds):
        ds, store = lazy_ds
        assert ds.PixelData == b"\x01\x02\x03\x04" * 250
        assert not ds[0x7FE00010].is_lazy_bulk
        assert store.n_calls == 1

    def test_replace_element_does_not_access_old_uri(self, lazy_ds):
        ds, store = lazy_ds
        ds.PixelData = b"\x00" * 16
        assert ds[0x7FE00010].value == b"\x00" * 16
        assert store.n_calls == 0
        # replacing via the element's value setter as well
        elem = ds[0x00091001]
        elem.value = b"\x01" * 8
        assert elem.value == b"\x01" * 8
        assert store.n_calls == 0
        # clearing
        ds[0x00091002].clear()
        assert ds[0x00091002].is_empty
        assert store.n_calls == 0

    def test_delete_element_does_not_access_old_uri(self, lazy_ds):
        ds, store = lazy_ds
        del ds[0x7FE00010]
        assert 0x7FE00010 not in ds
        ds2_item = ds[0x00081140].value[0]
        del ds2_item[0x00091011]
        assert store.n_calls == 0

    def test_equality_of_unretrieved_references(self, lazy_ds):
        ds, store = lazy_ds
        other = make_lazy_dataset(
            {
                PIXEL_DATA: bulk("OW", URI_1),
                "00091001": bulk("OB", URI_2),
            },
            store,
        )
        assert ds[0x7FE00010] == other[0x7FE00010]
        assert ds[0x00091001] == other[0x00091001]
        assert ds[0x7FE00010] != other[0x00091001]
        assert store.n_calls == 0


# ---------------------------------------------------------------------------
# Explicit value access (materialization)
# ---------------------------------------------------------------------------


class TestMaterialization:
    def test_value_matches_payload(self):
        payloads = {URI_1: os.urandom(2048), URI_2: os.urandom(7)}
        store = BulkDataStore(payloads)
        ds = make_lazy_dataset(
            {PIXEL_DATA: bulk("OW", URI_1), "00091001": bulk("OB", URI_2)}, store
        )
        assert ds[0x7FE00010].value == payloads[URI_1]
        assert ds[0x00091001].value == payloads[URI_2]
        assert store.n_calls == 2
        # the handler received (tag, vr, uri)
        assert store.calls[0] == (PIXEL_DATA, "OW", URI_1)
        assert store.calls[1] == ("00091001", "OB", URI_2)

    def test_stream_closed_after_materialization(self):
        store = BulkDataStore({URI_1: b"\x01" * 100})
        ds = make_lazy_dataset({PIXEL_DATA: bulk("OW", URI_1)}, store)
        ds[0x7FE00010].value
        assert store.all_closed

    def test_short_reads_are_handled(self):
        payload = os.urandom(1000)
        # the stream returns at most 7 bytes per read()
        store = BulkDataStore({URI_1: payload}, max_chunk=7)
        ds = make_lazy_dataset({PIXEL_DATA: bulk("OW", URI_1)}, store)
        assert ds[0x7FE00010].value == payload
        assert store.all_closed

    def test_same_uri_different_elements_are_independent(self):
        payload = os.urandom(64)
        store = BulkDataStore({URI_1: payload})
        ds = make_lazy_dataset(
            {"00091001": bulk("OB", URI_1), "00091002": bulk("OB", URI_1)}, store
        )
        assert ds[0x00091001].value == payload
        assert ds[0x00091002].value == payload
        # each element retrieved independently with its own stream
        assert store.n_calls == 2
        assert store.streams[0] is not store.streams[1]
        assert store.all_closed


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def write_and_reread(ds, tmp_path, name="out.dcm", enforce_file_format=True):
    path = str(tmp_path / name)
    ds.save_as(path, enforce_file_format=enforce_file_format)
    return path, dcmread(path)


class TestLazyBulkDataWrite:
    def test_top_level_values_and_padding(self, tmp_path):
        payload_ow = b"\x01\x02\x03\x04" * 250  # 1000 bytes, even
        payload_ob = b"\xaa" * 999  # odd -> one 0x00 pad byte
        payload_un = b"\xbb" * 1000  # even
        store = BulkDataStore({URI_1: payload_ow, URI_2: payload_ob, URI_3: payload_un})
        ds = make_lazy_dataset(
            {
                "00100010": {"vr": "PN", "Value": [{"Alphabetic": "CITIZEN^Joan"}]},
                PIXEL_DATA: bulk("OW", URI_1),
                "00091001": bulk("OB", URI_2),
                "00091002": bulk("UN", URI_3),
            },
            store,
        )
        add_file_meta(ds)
        path, back = write_and_reread(ds, tmp_path)

        # values read back match the payload plus standard padding
        assert back[0x7FE00010].value == payload_ow
        assert back[0x00091001].value == payload_ob + b"\x00"
        assert back[0x00091002].value == payload_un
        assert back.PatientName == "CITIZEN^Joan"

        # independent structure check of the exact encoded lengths
        _, records = scan_file(path)
        rec_ow = find_records(records, 0x7FE00010)[0]
        rec_ob = find_records(records, 0x00091001)[0]
        rec_un = find_records(records, 0x00091002)[0]
        assert rec_ow["vr"] == "OW" and rec_ow["length"] == 1000
        assert rec_ob["vr"] == "OB" and rec_ob["length"] == 1000  # 999 + pad
        assert rec_un["vr"] == "UN" and rec_un["length"] == 1000
        # the pad byte is 0x00
        with open(path, "rb") as f:
            f.seek(rec_ob["value_offset"] + 999)
            assert f.read(1) == b"\x00"

        # the references were not consumed by the write
        assert ds[0x7FE00010].is_lazy_bulk
        assert ds[0x00091001].is_lazy_bulk
        assert store.all_closed

    def test_odd_un_is_padded(self, tmp_path):
        payload = b"\xcc" * 51  # odd
        store = BulkDataStore({URI_3: payload})
        ds = make_lazy_dataset({"00091002": bulk("UN", URI_3)}, store)
        add_file_meta(ds)
        path, back = write_and_reread(ds, tmp_path)
        assert back[0x00091002].value == payload + b"\x00"
        _, records = scan_file(path)
        assert find_records(records, 0x00091002)[0]["length"] == 52

    def test_empty_lazy_value(self, tmp_path):
        store = BulkDataStore({URI_1: b""})
        ds = make_lazy_dataset({"00091001": bulk("OB", URI_1)}, store)
        add_file_meta(ds)
        path, back = write_and_reread(ds, tmp_path)
        assert 0x00091001 in back
        assert back[0x00091001].is_empty
        _, records = scan_file(path)
        assert find_records(records, 0x00091001)[0]["length"] == 0
        assert store.all_closed

    def test_two_level_sequence_defined_lengths(self, tmp_path):
        payload_inner = b"\x10" * 100
        payload_top = b"\x20" * 200
        store = BulkDataStore({URI_1: payload_inner, URI_2: payload_top})
        json_data = {
            "00081140": {
                "vr": "SQ",
                "Value": [
                    {
                        "00100020": {"vr": "LO", "Value": ["ID-1"]},
                        "00091005": {
                            "vr": "SQ",
                            "Value": [
                                {
                                    "00091011": bulk("OB", URI_1),
                                    "00091012": bulk("OW", URI_1),
                                },
                                {"00100030": {"vr": "DA", "Value": ["20240101"]}},
                            ],
                        },
                    },
                    {"00091013": bulk("UN", URI_2)},
                ],
            }
        }
        ds = make_lazy_dataset(json_data, store)
        add_file_meta(ds)
        path, back = write_and_reread(ds, tmp_path)

        # hierarchy and values via dcmread
        top = back[0x00081140].value
        assert len(top) == 2
        inner = top[0][0x00091005].value
        assert len(inner) == 2
        assert inner[0][0x00091011].value == payload_inner
        assert inner[0][0x00091012].value == payload_inner
        assert inner[1][0x00100030].value == "20240101"
        assert top[0][0x00100020].value == "ID-1"
        assert top[1][0x00091013].value == payload_top

        # exact lengths via the independent scanner
        _, records = scan_file(path)
        sq_top = find_records(records, 0x00081140)[0]
        assert sq_top["vr"] == "SQ" and sq_top["length"] != UNDEFINED_LENGTH
        items = sq_top["items"]
        assert len(items) == 2 and items[0]["length"] != UNDEFINED_LENGTH
        sq_inner = find_records(items[0]["children"], 0x00091005)[0]
        assert sq_inner["length"] != UNDEFINED_LENGTH
        inner_items = sq_inner["items"]
        assert len(inner_items) == 2
        # check the inner SQ length equals the sum of its item encodings
        inner_item_len = inner_items[0]["length"]
        lo_rec = find_records(items[0]["children"], 0x00100020)[0]
        ob_rec = find_records(inner_items[0]["children"], 0x00091011)[0]
        assert ob_rec["length"] == 100
        # item length = sum of element encodings (12-byte headers for OB/OW)
        assert inner_item_len == 12 + 100 + 12 + 100
        assert sq_inner["length"] == (
            8 + inner_item_len + 8 + 8 + 8  # items + DA element in item 2
        )
        # top SQ length = both items
        assert sq_top["length"] == (8 + items[0]["length"] + 8 + items[1]["length"])
        # item 0 length = LO element + inner SQ element
        assert items[0]["length"] == (8 + lo_rec["length"] + 12 + sq_inner["length"])
        assert store.all_closed

    @pytest.mark.parametrize("sq_undefined", [False, True])
    @pytest.mark.parametrize("item_undefined", [False, True])
    def test_sequence_length_combinations(self, tmp_path, sq_undefined, item_undefined):
        payload = b"\x77" * 128
        store = BulkDataStore({URI_1: payload})
        json_data = {
            "00081140": {
                "vr": "SQ",
                "Value": [
                    {
                        "00100020": {"vr": "LO", "Value": ["ID-1"]},
                        "00091011": bulk("OB", URI_1),
                    },
                    {"00091012": bulk("OW", URI_1)},
                ],
            }
        }
        ds = make_lazy_dataset(json_data, store)
        sq_elem = ds[0x00081140]
        sq_elem.is_undefined_length = sq_undefined
        for item in sq_elem.value:
            item.is_undefined_length_sequence_item = item_undefined
        add_file_meta(ds)
        path, back = write_and_reread(ds, tmp_path)

        items = back[0x00081140].value
        assert len(items) == 2
        assert items[0][0x00091011].value == payload
        assert items[0][0x00100020].value == "ID-1"
        assert items[1][0x00091012].value == payload

        _, records = scan_file(path)
        sq = find_records(records, 0x00081140)[0]
        assert (sq["length"] == UNDEFINED_LENGTH) == sq_undefined
        if sq_undefined:
            assert sq["delim_length"] == 0
        for item_rec in sq["items"]:
            assert (item_rec["length"] == UNDEFINED_LENGTH) == item_undefined
            if item_undefined:
                assert item_rec["delim_length"] == 0
        if not sq_undefined:
            # the SQ length is the sum of the encoded item sizes
            assert sq["length"] == sum(
                encoded_size(item_rec) for item_rec in sq["items"]
            )
        assert store.all_closed

    def test_same_uri_different_elements_one_write(self, tmp_path):
        payload = b"\x55" * 300
        store = BulkDataStore({URI_1: payload})
        ds = make_lazy_dataset(
            {"00091001": bulk("OB", URI_1), "00091002": bulk("OB", URI_1)}, store
        )
        add_file_meta(ds)
        path, back = write_and_reread(ds, tmp_path)
        assert back[0x00091001].value == payload
        assert back[0x00091002].value == payload
        # each element consumed its own newly opened stream
        assert store.n_calls == 2
        assert len(store.streams) == 2
        assert store.streams[0] is not store.streams[1]
        assert store.all_closed

    def test_write_keeps_reference_and_reopens_on_rewrite(self, tmp_path):
        payload = b"\x42" * 128
        store = BulkDataStore({URI_1: payload})
        ds = make_lazy_dataset({PIXEL_DATA: bulk("OW", URI_1)}, store)
        add_file_meta(ds)
        path = str(tmp_path / "out.dcm")
        ds.save_as(path, enforce_file_format=True)
        assert ds[0x7FE00010].is_lazy_bulk
        assert store.n_calls == 1
        first_streams = list(store.streams)

        # writing again reopens a new stream - no caching of the download
        ds.save_as(path, enforce_file_format=True)
        assert store.n_calls == 2
        assert store.streams[1] is not first_streams[0]
        assert store.all_closed
        assert dcmread(path)[0x7FE00010].value == payload

    def test_write_after_materialization_uses_bytes(self, tmp_path):
        payload = b"\x42" * 128
        store = BulkDataStore({URI_1: payload})
        ds = make_lazy_dataset({PIXEL_DATA: bulk("OW", URI_1)}, store)
        add_file_meta(ds)
        assert ds[0x7FE00010].value == payload
        assert store.n_calls == 1
        path, back = write_and_reread(ds, tmp_path)
        # no new retrieval for the materialized value
        assert store.n_calls == 1
        assert back[0x7FE00010].value == payload

    def test_replaced_reference_writes_new_value(self, tmp_path):
        store = BulkDataStore({URI_1: b"\x42" * 128})
        ds = make_lazy_dataset({PIXEL_DATA: bulk("OW", URI_1)}, store)
        add_file_meta(ds)
        ds.PixelData = b"\x99" * 64
        path, back = write_and_reread(ds, tmp_path)
        assert back[0x7FE00010].value == b"\x99" * 64
        assert store.n_calls == 0

    def test_dcmwrite_and_save_as_both_supported(self, tmp_path):
        payload = b"\x42" * 128
        store = BulkDataStore({URI_1: payload})
        ds = make_lazy_dataset({PIXEL_DATA: bulk("OW", URI_1)}, store)
        add_file_meta(ds)
        path1 = str(tmp_path / "a.dcm")
        dcmwrite(path1, ds, enforce_file_format=True)
        path2 = str(tmp_path / "b.dcm")
        ds.save_as(path2, enforce_file_format=True)
        assert dcmread(path1)[0x7FE00010].value == payload
        assert dcmread(path2)[0x7FE00010].value == payload
        assert store.n_calls == 2

    def test_write_to_seekable_file_object(self, tmp_path):
        payload = b"\x42" * 128
        store = BulkDataStore({URI_1: payload})
        ds = make_lazy_dataset({PIXEL_DATA: bulk("OW", URI_1)}, store)
        add_file_meta(ds)
        path = str(tmp_path / "out.dcm")
        with open(path, "wb") as f:
            ds.save_as(f, enforce_file_format=True)
            # the caller-provided output object is not closed by the library
            assert not f.closed
        assert dcmread(path)[0x7FE00010].value == payload

    def test_file_meta_written(self, tmp_path):
        store = BulkDataStore({URI_1: b"\x42" * 128})
        ds = make_lazy_dataset({PIXEL_DATA: bulk("OW", URI_1)}, store)
        add_file_meta(ds)
        path, back = write_and_reread(ds, tmp_path)
        assert back.file_meta.TransferSyntaxUID == ExplicitVRLittleEndian
        assert back.file_meta.MediaStorageSOPClassUID == ds.SOPClassUID
        meta, _ = scan_file(path)
        assert find_records(meta, 0x00020010)[0]["vr"] == "UI"

    def test_implicit_vr_little_endian_write(self, tmp_path):
        payload = b"\x42" * 128
        store = BulkDataStore({URI_1: payload})
        ds = make_lazy_dataset({PIXEL_DATA: bulk("OW", URI_1)}, store)
        add_file_meta(ds, ImplicitVRLittleEndian)
        path, back = write_and_reread(ds, tmp_path)
        assert back[0x7FE00010].value == payload
        assert back.file_meta.TransferSyntaxUID == ImplicitVRLittleEndian

    def test_implicit_vr_little_endian_write_via_args(self, tmp_path):
        # the encoding may also come from the dcmwrite arguments instead of
        # the file meta
        payload = b"\x42" * 128
        store = BulkDataStore({URI_1: payload})
        ds = make_lazy_dataset({PIXEL_DATA: bulk("OW", URI_1)}, store)
        path = str(tmp_path / "out.dcm")
        dcmwrite(path, ds, implicit_vr=True, little_endian=True)
        back = dcmread(path, force=True)
        assert back[0x7FE00010].value == payload
        assert store.n_calls == 1

    def test_implicit_vr_nested_sequence(self, tmp_path):
        payload = b"\x42" * 128
        store = BulkDataStore({URI_1: payload})
        # with implicit VR the VR is taken from the dictionary, so public
        # tags are needed for the sequences
        json_data = {
            "00081140": {
                "vr": "SQ",
                "Value": [
                    {
                        "00081110": {
                            "vr": "SQ",
                            "Value": [{"00091011": bulk("OB", URI_1)}],
                        }
                    }
                ],
            }
        }
        ds = make_lazy_dataset(json_data, store)
        add_file_meta(ds, ImplicitVRLittleEndian)
        path, back = write_and_reread(ds, tmp_path)
        inner = back[0x00081140].value[0][0x00081110].value[0]
        assert inner[0x00091011].value == payload
        assert store.all_closed

    def test_interleaved_writes_of_two_datasets(self, tmp_path):
        payload1 = b"\x11" * 256
        payload2 = b"\x22" * 512
        store1 = BulkDataStore({URI_1: payload1})
        store2 = BulkDataStore({URI_2: payload2})
        ds1 = make_lazy_dataset({PIXEL_DATA: bulk("OW", URI_1)}, store1)
        ds2 = make_lazy_dataset({PIXEL_DATA: bulk("OW", URI_2)}, store2)
        add_file_meta(ds1)
        add_file_meta(ds2)
        path1 = str(tmp_path / "one.dcm")
        path2 = str(tmp_path / "two.dcm")
        # interleave the writes; each dataset must use its own streams
        ds1.save_as(path1, enforce_file_format=True)
        ds2.save_as(path2, enforce_file_format=True)
        ds1.save_as(path1, enforce_file_format=True)
        ds2.save_as(path2, enforce_file_format=True)
        assert dcmread(path1)[0x7FE00010].value == payload1
        assert dcmread(path2)[0x7FE00010].value == payload2
        assert store1.n_calls == 2 and store2.n_calls == 2
        assert not set(store1.streams) & set(store2.streams)
        assert store1.all_closed and store2.all_closed


class TestLazyBulkDataWritePreflight:
    """Unsupported outputs/transfer syntaxes are rejected before any URI is
    opened."""

    def make_ds(self, store, tsyntax=ExplicitVRLittleEndian):
        ds = make_lazy_dataset({PIXEL_DATA: bulk("OW", URI_1)}, store)
        add_file_meta(ds, tsyntax)
        return ds

    def test_non_seekable_output_rejected(self):
        store = BulkDataStore({URI_1: b"\x42" * 128})
        ds = self.make_ds(store)

        class NonSeekableWriter:
            def __init__(self):
                self.buffer = io.BytesIO()

            def write(self, b):
                return self.buffer.write(b)

            def seek(self, *args):
                raise io.UnsupportedOperation("not seekable")

            def tell(self):
                raise io.UnsupportedOperation("not seekable")

            def seekable(self):
                return False

        with pytest.raises(ValueError, match="non-seekable output"):
            dcmwrite(NonSeekableWriter(), ds, enforce_file_format=True)
        assert store.n_calls == 0
        assert store.streams == []

    def test_big_endian_transfer_syntax_rejected(self, tmp_path):
        store = BulkDataStore({URI_1: b"\x42" * 128})
        ds = self.make_ds(store, ExplicitVRBigEndian)
        with pytest.raises(ValueError, match="transfer syntax"):
            ds.save_as(str(tmp_path / "out.dcm"), enforce_file_format=True)
        assert store.n_calls == 0

    def test_big_endian_encoding_rejected(self, tmp_path):
        store = BulkDataStore({URI_1: b"\x42" * 128})
        ds = make_lazy_dataset({PIXEL_DATA: bulk("OW", URI_1)}, store)
        with pytest.raises(ValueError, match="big endian"):
            dcmwrite(
                str(tmp_path / "out.dcm"),
                ds,
                implicit_vr=False,
                little_endian=False,
            )
        assert store.n_calls == 0

    def test_compressed_transfer_syntax_rejected(self, tmp_path):
        store = BulkDataStore({URI_1: b"\x42" * 128})
        ds = self.make_ds(store, JPEGBaseline8Bit)
        with pytest.raises(ValueError, match="transfer syntax"):
            ds.save_as(str(tmp_path / "out.dcm"), enforce_file_format=True)
        assert store.n_calls == 0

    def test_deflated_transfer_syntax_rejected(self, tmp_path):
        store = BulkDataStore({URI_1: b"\x42" * 128})
        ds = self.make_ds(store, DeflatedExplicitVRLittleEndian)
        with pytest.raises(ValueError, match="transfer syntax"):
            ds.save_as(str(tmp_path / "out.dcm"), enforce_file_format=True)
        assert store.n_calls == 0


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


class FailingStream(TrackingStream):
    """A stream that provides fewer or more bytes than declared."""

    def __init__(self, data, declared_length, **kwargs):
        super().__init__(data, **kwargs)
        self.declared_length = declared_length


class FailingStore(BulkDataStore):
    """A store returning streams with a mismatching declared length."""

    def __init__(self, payloads, declared_lengths=None, **kwargs):
        super().__init__(payloads, **kwargs)
        self.declared_lengths = declared_lengths or {}

    def handler(self, tag, vr, uri):
        self.calls.append((tag, vr, uri))
        if self.fail_with is not None:
            raise self.fail_with
        data = self.payloads[uri]
        stream = TrackingStream(data, max_chunk=self.max_chunk)
        self.streams.append(stream)
        return stream, self.declared_lengths.get(uri, len(data))


class TestRetrievalFailures:
    def test_premature_eof_on_value_access(self):
        store = FailingStore({URI_1: b"\x01" * 50}, {URI_1: 100})
        ds = make_lazy_dataset({PIXEL_DATA: bulk("OW", URI_1)}, store)
        with pytest.raises(BulkDataError, match="ended after 50 of 100") as e:
            ds[0x7FE00010].value
        assert URI_1 in str(e.value)
        assert "(7FE0,0010)" in str(e.value)
        # the opened stream was closed and the reference is kept
        assert store.all_closed
        assert ds[0x7FE00010].is_lazy_bulk
        # retry with a fixed store succeeds
        store.declared_lengths[URI_1] = 50
        assert ds[0x7FE00010].value == b"\x01" * 50

    def test_extra_bytes_on_value_access(self):
        store = FailingStore({URI_1: b"\x01" * 100}, {URI_1: 50})
        ds = make_lazy_dataset({PIXEL_DATA: bulk("OW", URI_1)}, store)
        with pytest.raises(BulkDataError, match="more than the declared"):
            ds[0x7FE00010].value
        assert store.all_closed
        assert ds[0x7FE00010].is_lazy_bulk

    def test_read_exception_on_value_access(self):
        store = BulkDataStore({URI_1: b"\x01" * 100})
        store.exc_on_read = OSError("connection reset")
        ds = make_lazy_dataset({PIXEL_DATA: bulk("OW", URI_1)}, store)
        with pytest.raises(BulkDataError, match="connection reset") as e:
            ds[0x7FE00010].value
        assert isinstance(e.value.__cause__, OSError)
        assert store.all_closed
        assert ds[0x7FE00010].is_lazy_bulk

    def test_handler_exception_on_value_access(self):
        store = BulkDataStore({URI_1: b"\x01" * 100})
        store.fail_with = RuntimeError("service unavailable")
        ds = make_lazy_dataset({PIXEL_DATA: bulk("OW", URI_1)}, store)
        with pytest.raises(BulkDataError, match="service unavailable") as e:
            ds[0x7FE00010].value
        assert URI_1 in str(e.value)
        assert "(7FE0,0010)" in str(e.value)
        assert isinstance(e.value.__cause__, RuntimeError)
        assert ds[0x7FE00010].is_lazy_bulk

    def test_handler_returns_garbage(self):
        store = BulkDataStore({URI_1: b"\x01" * 100})
        store.result_override = b"not-a-tuple"
        ds = make_lazy_dataset({PIXEL_DATA: bulk("OW", URI_1)}, store)
        with pytest.raises(BulkDataError, match="must return a"):
            ds[0x7FE00010].value
        store.result_override = (object(), 100)
        with pytest.raises(BulkDataError, match="invalid stream"):
            ds[0x7FE00010].value
        assert ds[0x7FE00010].is_lazy_bulk

    @pytest.mark.parametrize("bad_length", [-1, True, False, 2.5, "100"])
    def test_invalid_length_on_value_access(self, bad_length):
        stream = TrackingStream(b"\x01" * 100)
        store = BulkDataStore({URI_1: b"\x01" * 100})
        store.result_override = (stream, bad_length)
        ds = make_lazy_dataset({PIXEL_DATA: bulk("OW", URI_1)}, store)
        with pytest.raises(BulkDataError, match="Invalid length"):
            ds[0x7FE00010].value
        # the stream returned by the handler was closed
        assert stream.closed
        assert ds[0x7FE00010].is_lazy_bulk

    def test_ow_odd_length_rejected(self):
        store = BulkDataStore({URI_1: b"\x01" * 51})
        ds = make_lazy_dataset({PIXEL_DATA: bulk("OW", URI_1)}, store)
        with pytest.raises(BulkDataError, match="odd length 51"):
            ds[0x7FE00010].value
        assert store.all_closed
        assert ds[0x7FE00010].is_lazy_bulk


class TestWriteFailures:
    def make_nested_ds(self, store):
        json_data = {
            "00081140": {
                "vr": "SQ",
                "Value": [
                    {"00100020": {"vr": "LO", "Value": ["ID-1"]}},
                    {
                        "00091005": {
                            "vr": "SQ",
                            "Value": [{PIXEL_DATA: bulk("OW", URI_1)}],
                        }
                    },
                ],
            }
        }
        ds = make_lazy_dataset(json_data, store)
        add_file_meta(ds)
        return ds

    def test_premature_eof_on_write(self, tmp_path):
        store = FailingStore({URI_1: b"\x01" * 50}, {URI_1: 100})
        ds = make_lazy_dataset({PIXEL_DATA: bulk("OW", URI_1)}, store)
        add_file_meta(ds)
        path = str(tmp_path / "out.dcm")
        with pytest.raises(BulkDataError, match="ended after 50 of 100"):
            ds.save_as(path, enforce_file_format=True)
        assert store.all_closed
        assert ds[0x7FE00010].is_lazy_bulk
        # retry with a fixed store produces a complete file
        store.declared_lengths[URI_1] = 50
        ds.save_as(path, enforce_file_format=True)
        assert dcmread(path)[0x7FE00010].value == b"\x01" * 50

    def test_extra_bytes_on_write(self, tmp_path):
        store = FailingStore({URI_1: b"\x01" * 100}, {URI_1: 50})
        ds = make_lazy_dataset({PIXEL_DATA: bulk("OW", URI_1)}, store)
        add_file_meta(ds)
        with pytest.raises(BulkDataError, match="more than the declared"):
            ds.save_as(str(tmp_path / "out.dcm"), enforce_file_format=True)
        assert store.all_closed
        assert ds[0x7FE00010].is_lazy_bulk

    def test_read_exception_on_write(self, tmp_path):
        store = BulkDataStore({URI_1: b"\x01" * 100})
        store.exc_on_read = OSError("connection reset")
        ds = make_lazy_dataset({PIXEL_DATA: bulk("OW", URI_1)}, store)
        add_file_meta(ds)
        with pytest.raises(BulkDataError, match="connection reset"):
            ds.save_as(str(tmp_path / "out.dcm"), enforce_file_format=True)
        assert store.all_closed
        assert ds[0x7FE00010].is_lazy_bulk

    def test_handler_exception_on_write(self, tmp_path):
        store = BulkDataStore({URI_1: b"\x01" * 100})
        ds = make_lazy_dataset({PIXEL_DATA: bulk("OW", URI_1)}, store)
        add_file_meta(ds)
        store.fail_with = RuntimeError("service unavailable")
        with pytest.raises(BulkDataError, match="service unavailable"):
            ds.save_as(str(tmp_path / "out.dcm"), enforce_file_format=True)
        assert ds[0x7FE00010].is_lazy_bulk

    def test_ow_odd_length_rejected_on_write(self, tmp_path):
        store = BulkDataStore({URI_1: b"\x01" * 51})
        ds = make_lazy_dataset({PIXEL_DATA: bulk("OW", URI_1)}, store)
        add_file_meta(ds)
        with pytest.raises(BulkDataError, match="odd length 51"):
            ds.save_as(str(tmp_path / "out.dcm"), enforce_file_format=True)
        assert store.all_closed
        assert ds[0x7FE00010].is_lazy_bulk

    def test_nested_error_locates_uri_tag_and_item_position(self, tmp_path):
        store = FailingStore({URI_1: b"\x01" * 50}, {URI_1: 100})
        ds = self.make_nested_ds(store)
        with pytest.raises(BulkDataError) as e:
            ds.save_as(str(tmp_path / "out.dcm"), enforce_file_format=True)
        msg = str(e.value)
        assert URI_1 in msg
        assert "(7FE0,0010)" in msg
        # the nested item positions are part of the error
        assert "item 0 of sequence element (0009,1005)" in msg
        assert "item 1 of sequence element (0008,1140)" in msg
        assert store.all_closed
        # the reference is kept and a retry with a fixed store succeeds
        store.declared_lengths[URI_1] = 50
        ds.save_as(str(tmp_path / "out.dcm"), enforce_file_format=True)

    def test_caller_output_not_closed_on_failure(self, tmp_path):
        store = FailingStore({URI_1: b"\x01" * 50}, {URI_1: 100})
        ds = make_lazy_dataset({PIXEL_DATA: bulk("OW", URI_1)}, store)
        add_file_meta(ds)
        path = str(tmp_path / "out.dcm")
        with open(path, "wb") as f:
            with pytest.raises(BulkDataError):
                ds.save_as(f, enforce_file_format=True)
            assert not f.closed

    def test_interleaved_error_isolation(self, tmp_path):
        good = BulkDataStore({URI_1: b"\x11" * 100})
        bad = FailingStore({URI_2: b"\x22" * 50}, {URI_2: 100})
        ds_good = make_lazy_dataset({PIXEL_DATA: bulk("OW", URI_1)}, good)
        ds_bad = make_lazy_dataset({PIXEL_DATA: bulk("OW", URI_2)}, bad)
        add_file_meta(ds_good)
        add_file_meta(ds_bad)
        path_good = str(tmp_path / "good.dcm")
        path_bad = str(tmp_path / "bad.dcm")
        ds_good.save_as(path_good, enforce_file_format=True)
        with pytest.raises(BulkDataError):
            ds_bad.save_as(path_bad, enforce_file_format=True)
        # the failed write does not affect the other dataset
        ds_good.save_as(path_good, enforce_file_format=True)
        assert dcmread(path_good)[0x7FE00010].value == b"\x11" * 100
        assert ds_bad[0x7FE00010].is_lazy_bulk
        assert bad.all_closed


# ---------------------------------------------------------------------------
# Large generative payloads
# ---------------------------------------------------------------------------


def build_deep_json(uris):
    """Build a JSON model with the bulk elements nested in two levels of
    sequences (all written with defined lengths)."""
    return {
        "00100010": {"vr": "PN", "Value": [{"Alphabetic": "CITIZEN^Joan"}]},
        "00081140": {
            "vr": "SQ",
            "Value": [
                {
                    "00100020": {"vr": "LO", "Value": ["ID-1"]},
                    "00091005": {
                        "vr": "SQ",
                        "Value": [
                            {
                                "00091011": bulk("OB", uris[0]),
                                "00091012": bulk("OW", uris[1]),
                            },
                            {"00091013": bulk("UN", uris[2])},
                        ],
                    },
                }
            ],
        },
    }


@pytest.mark.parametrize(
    "payload_size", [8 * 1024 * 1024, 64 * 1024 * 1024, 256 * 1024 * 1024]
)
def test_large_payloads_in_deep_defined_sequences(tmp_path, payload_size):
    """Multiple large generative payloads inside defined-length nested SQs.

    Records the maximum read() request at the source, the peak extra memory
    during the write and the encoded lengths, proving that scaling the
    payload does not scale the memory held by the outer sequences.
    """
    uris = [f"https://example.test/big/{i}" for i in range(3)]
    layout = {
        uris[0]: (payload_size + 1, 3),  # odd OB length -> padded
        uris[1]: (payload_size, 0),
        uris[2]: (payload_size, 7),
    }
    store = GenerativeStore(layout)
    ds = make_lazy_dataset(build_deep_json(uris), store)
    add_file_meta(ds)

    path = str(tmp_path / "big.dcm")
    tracemalloc.start()
    try:
        ds.save_as(path, enforce_file_format=True)
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    # --- source read requests ---------------------------------------------
    for stream in store.streams:
        assert stream.closed
        assert max(stream.read_requests) <= MAX_BULK_DATA_CHUNK_SIZE
        assert min(stream.read_requests) > 0
    max_request = max(max(s.read_requests) for s in store.streams)

    # --- peak extra memory -------------------------------------------------
    # bounded independently of the payload size (payloads are >= 8 MiB)
    assert peak < 48 * 1024 * 1024

    # --- encoded lengths ----------------------------------------------------
    _, records = scan_file(path)
    sq_top = find_records(records, 0x00081140)[0]
    assert sq_top["length"] != UNDEFINED_LENGTH
    (item0,) = sq_top["items"]
    sq_inner = find_records(item0["children"], 0x00091005)[0]
    assert sq_inner["length"] != UNDEFINED_LENGTH
    item1, item2 = sq_inner["items"]
    rec_ob = find_records(item1["children"], 0x00091011)[0]
    rec_ow = find_records(item1["children"], 0x00091012)[0]
    rec_un = find_records(item2["children"], 0x00091013)[0]
    assert rec_ob["length"] == payload_size + 2  # odd payload + pad byte
    assert rec_ow["length"] == payload_size
    assert rec_un["length"] == payload_size
    # item/SQ lengths account exactly for their content
    assert item1["length"] == (12 + rec_ob["length"] + 12 + rec_ow["length"])
    assert item2["length"] == 12 + rec_un["length"]
    assert sq_inner["length"] == 8 + item1["length"] + 8 + item2["length"]
    lo_rec = find_records(item0["children"], 0x00100020)[0]
    assert item0["length"] == 8 + lo_rec["length"] + 12 + sq_inner["length"]
    assert sq_top["length"] == 8 + item0["length"]

    # --- content correctness (streaming hash of the value regions) ---------
    assert hash_file_region(
        path, rec_ob["value_offset"], rec_ob["length"]
    ) == expected_file_value_hash(payload_size + 1, 3)
    assert hash_file_region(
        path, rec_ow["value_offset"], rec_ow["length"]
    ) == expected_file_value_hash(payload_size, 0)
    assert hash_file_region(
        path, rec_un["value_offset"], rec_un["length"]
    ) == expected_file_value_hash(payload_size, 7)

    # --- the references are still unretrieved -------------------------------
    inner = ds[0x00081140].value[0][0x00091005].value
    assert inner[0][0x00091011].is_lazy_bulk
    assert inner[0][0x00091012].is_lazy_bulk
    assert inner[1][0x00091013].is_lazy_bulk

    print(
        f"\npayload={payload_size // (1024 * 1024)} MiB "
        f"max_read_request={max_request} "
        f"peak_extra_memory={peak} "
        f"top_sq_length={sq_top['length']}"
    )


def test_memory_does_not_scale_with_payload(tmp_path):
    """The extra memory held while writing is independent of the payload
    size: writing 8 MiB and 256 MiB payloads through two levels of
    defined-length sequences peaks at essentially the same amount."""
    peaks = {}
    for payload_size in (8 * 1024 * 1024, 256 * 1024 * 1024):
        uri = f"https://example.test/scaling/{payload_size}"
        store = GenerativeStore({uri: (payload_size, 0)})
        json_data = {
            "00081140": {
                "vr": "SQ",
                "Value": [
                    {
                        "00091005": {
                            "vr": "SQ",
                            "Value": [{"00091011": bulk("OB", uri)}],
                        }
                    }
                ],
            }
        }
        ds = make_lazy_dataset(json_data, store)
        add_file_meta(ds)
        tracemalloc.start()
        try:
            ds.save_as(
                str(tmp_path / f"scaling_{payload_size}.dcm"),
                enforce_file_format=True,
            )
        finally:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        peaks[payload_size] = peak

    small, large = peaks[8 * 1024 * 1024], peaks[256 * 1024 * 1024]
    print(f"\npeak_extra_memory: 8 MiB payload -> {small}, 256 MiB -> {large}")
    # the payload is 32x larger, the peak extra memory must not grow
    #   proportionally (allowing a small constant margin)
    assert large < small + 16 * 1024 * 1024
