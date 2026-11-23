# Copyright 2024 pydicom authors. See LICENSE file for details.
"""Tests for the per-read ``private_dictionary`` overlay of ``dcmread``.

All input data in this module is hand encoded with :mod:`struct` (no patient
data), including one file that is written without any pydicom writer.
"""

import copy
import io
import pickle
import struct
import threading

import pytest

from pydicom import dcmread, dcmwrite, Dataset, DataElement
from pydicom._private_dict import private_dictionaries
from pydicom.datadict import (
    PrivateDictionaryBinding,
    make_private_dictionary_binding,
    get_active_private_binding,
)
from pydicom.dataelem import RawDataElement
from pydicom.tag import BaseTag, Tag
from .test_util import save_private_dict


CREATOR = "ACME TEST CREATOR"
OTHER_CREATOR = "ZENITH TEST CREATOR"
GROUP = 0x0041
NESTED_GROUP = 0x0043
TAG_VALUE = Tag(GROUP, 0x1001)  # block 0x10, offset 0x01
TAG_NESTED = Tag(NESTED_GROUP, 0x2002)  # block 0x20, offset 0x02
TAG_SQ = Tag(GROUP, 0x1010)
# payload that decodes differently under UL (one 32-bit int) and US (two 16s)
PAYLOAD = struct.pack("<I", 0x00010002)
UL_VALUE = 0x00010002
US_VALUES = [2, 1]


# ---------------------------------------------------------------------------
# Hand encoders: implicit and explicit VR little endian, no pydicom writer
# ---------------------------------------------------------------------------
def _pad(data: bytes) -> bytes:
    return data + b"\x00" if len(data) % 2 else data


def _implicit_element(group: int, element: int, value: bytes) -> bytes:
    return struct.pack("<HHI", group, element, len(value)) + value


def _explicit_element(group: int, element: int, vr: str, value: bytes, reserved: bool = True) -> bytes:
    head = struct.pack("<HH", group, element) + vr.encode("ascii")
    if vr in {"OB", "OW", "OF", "SQ", "UN", "UT"}:
        return head + b"\x00\x00" + struct.pack("<I", len(value)) + value
    return head + struct.pack("<H", len(value)) + value


def _item(data: bytes, undefined: bool = False) -> bytes:
    length = 0xFFFFFFFF if undefined else len(data)
    return struct.pack("<HHI", 0xFFFE, 0xE000, length) + data


def _sequence_delimiter() -> bytes:
    return struct.pack("<HHI", 0xFFFE, 0xE0DD, 0)


def _item_delimiter() -> bytes:
    return struct.pack("<HHI", 0xFFFE, 0xE00D, 0)


def _creator_element(group: int, block: int, name: str) -> bytes:
    return _implicit_element(group, block, _pad(name.encode("ascii")))


def _explicit_creator(group: int, block: int, name: str) -> bytes:
    return _explicit_element(group, block, "LO", _pad(name.encode("ascii")))


def _nested_dataset(undefined_item: bool, undefined_sq: bool) -> bytes:
    """A dataset containing one nested SQ whose item holds a private value."""
    inner_creator = _creator_element(NESTED_GROUP, 0x20, CREATOR)
    inner_value = _implicit_element(NESTED_GROUP, 0x2002, PAYLOAD)
    inner = inner_creator + inner_value

    sq_item = _item(inner, undefined=undefined_item)
    if undefined_item:
        sq_item += _item_delimiter()

    sq = sq_item
    if undefined_sq:
        sq += _sequence_delimiter()
    return _implicit_element(GROUP, 0x1010, sq)


def build_implicit_stream(
    *,
    with_sq: bool = False,
    undefined_item: bool = False,
    undefined_sq: bool = False,
    second_creator_block: bool = False,
    unknown_private: bool = False,
    empty_private: bool = False,
) -> io.BytesIO:
    """Build a complete implicit VR little endian byte stream by hand."""
    out = _creator_element(GROUP, 0x10, CREATOR)
    out += _implicit_element(GROUP, 0x1001, PAYLOAD)
    if second_creator_block:
        # same creator name registered at a different block of another group
        out += _creator_element(NESTED_GROUP, 0x20, OTHER_CREATOR)
        out += _implicit_element(NESTED_GROUP, 0x2002, struct.pack("<H", 7))
    if unknown_private:
        # no entry in any overlay -> old "UN" behaviour
        out += _implicit_element(GROUP, 0x1077, b"\x01\x02")
    if empty_private:
        out += _implicit_element(GROUP, 0x1003, b"")
    if with_sq:
        out += _nested_dataset(undefined_item, undefined_sq)
    return io.BytesIO(out)


def build_explicit_stream(vr: str = "US") -> io.BytesIO:
    """Explicit VR LE stream where the file pins the private element VR."""
    out = _explicit_creator(GROUP, 0x10, CREATOR)
    out += _explicit_element(GROUP, 0x1001, vr, PAYLOAD)
    return io.BytesIO(out)


def overlay(vr: str, description: str) -> dict:
    return {
        CREATOR: {
            (GROUP, 0x01): (vr, "1" if vr == "UL" else "2", description, ""),
        }
    }


OVERLAY_UL = overlay("UL", "Overlay Unsigned Long")
OVERLAY_US = overlay("US", "Overlay Unsigned Short")
NESTED_OVERLAY = {
    CREATOR: {
        (NESTED_GROUP, 0x02): ("UL", "1", "Nested Long", ""),
    },
    OTHER_CREATOR: {
        (NESTED_GROUP, 0x02): ("US", "1", "Other Short", ""),
    },
}
SQ_OVERLAY = {CREATOR: {(GROUP, 0x10): ("SQ", "1", "Private Sequence", "")}}


# ---------------------------------------------------------------------------
# Validation: everything invalid is a ValueError raised before any read
# ---------------------------------------------------------------------------
class TestValidation:
    def test_none_and_empty_are_ignored(self):
        assert make_private_dictionary_binding(None) is None
        assert make_private_dictionary_binding({}) is None

    @pytest.mark.parametrize(
        "creator",
        ["", 123, True, None, b"ACME"],
    )
    def test_invalid_creator(self, creator):
        with pytest.raises(ValueError):
            make_private_dictionary_binding({creator: {(GROUP, 1): ("UL", "1", "d", "")}})

    @pytest.mark.parametrize("group", [0x0000, 0x0002, 0x0010, 0x10000, -1, True])
    def test_invalid_group(self, group):
        with pytest.raises(ValueError):
            make_private_dictionary_binding({CREATOR: {(group, 1): ("UL", "1", "d", "")}})

    @pytest.mark.parametrize("offset", [-1, 256, 1000, True, False])
    def test_invalid_offset(self, offset):
        with pytest.raises(ValueError):
            make_private_dictionary_binding({CREATOR: {(GROUP, offset): ("UL", "1", "d", "")}})

    def test_key_must_be_tuple(self):
        with pytest.raises(ValueError):
            make_private_dictionary_binding({CREATOR: {GROUP * 0x10000 + 1: ("UL", "1", "d", "")}})

    @pytest.mark.parametrize("vr", ["US or SS", "XX", "ul", "", None, 2])
    def test_invalid_vr(self, vr):
        with pytest.raises(ValueError):
            make_private_dictionary_binding({CREATOR: {(GROUP, 1): (vr, "1", "d", "")}})

    @pytest.mark.parametrize("vm", ["", "one", "1..3", "n", "0", True, 3])
    def test_invalid_vm(self, vm):
        with pytest.raises(ValueError):
            make_private_dictionary_binding({CREATOR: {(GROUP, 1): ("UL", vm, "d", "")}})

    @pytest.mark.parametrize("vm", ["1", "1-n", "2-2n", "3-3n", "1-32", "16"])
    def test_valid_vm_notation(self, vm):
        binding = make_private_dictionary_binding({CREATOR: {(GROUP, 1): ("UL", vm, "d", "")}})
        assert binding is not None

    @pytest.mark.parametrize("description", [1, None, b"d", True])
    def test_invalid_description(self, description):
        with pytest.raises(ValueError):
            make_private_dictionary_binding({CREATOR: {(GROUP, 1): ("UL", "1", description, "")}})

    @pytest.mark.parametrize("retired", [1, None, b"", False])
    def test_invalid_retired(self, retired):
        with pytest.raises(ValueError):
            make_private_dictionary_binding({CREATOR: {(GROUP, 1): ("UL", "1", "d", retired)}})

    def test_entry_must_be_four_tuple(self):
        with pytest.raises(ValueError):
            make_private_dictionary_binding({CREATOR: {(GROUP, 1): ["UL", "1", "d", ""]}})
        with pytest.raises(ValueError):
            make_private_dictionary_binding({CREATOR: {(GROUP, 1): ("UL", "1", "d")}})

    def test_callables_rejected_everywhere(self):
        with pytest.raises(ValueError):
            make_private_dictionary_binding({CREATOR: {(GROUP, 1): (lambda: "UL", "1", "d", "")}})

        class Weird(dict):
            def items(self):
                raise RuntimeError("boom from caller container")

        with pytest.raises(ValueError):
            make_private_dictionary_binding({CREATOR: Weird()})

    def test_not_a_mapping(self):
        with pytest.raises(ValueError):
            make_private_dictionary_binding([("a", {})])

    def test_invalid_config_raises_before_first_byte(self):
        # A broken file object that fails if touched at all
        class ExplodingIO(io.RawIOBase):
            def readable(self):
                return True

            def read(self, *args, **kwargs):
                raise AssertionError("input must not be read on bad config")

            def seek(self, *args):
                raise AssertionError("input must not be seeked on bad config")

            def tell(self):
                return 0

        with pytest.raises(ValueError):
            dcmread(
                ExplodingIO(),
                force=True,
                private_dictionary={CREATOR: {(GROUP, 256): ("UL", "1", "d", "")}},
            )

    def test_binding_is_frozen_copy(self):
        caller = {CREATOR: {(GROUP, 1): ("UL", "1", "orig", "")}}
        binding = make_private_dictionary_binding(caller)
        caller[CREATOR][(GROUP, 1)] = ("US", "1", "changed", "")
        caller["NEW"] = {}
        assert binding.overlay[CREATOR][(GROUP, 1)] == ("UL", "1", "orig", "")
        assert "NEW" not in binding.overlay


# ---------------------------------------------------------------------------
# Interpretation: UL vs US, names, standard tags, fallback, unknown/empty
# ---------------------------------------------------------------------------
class TestInterpretation:
    def test_same_payload_decoded_per_overlay(self):
        ds_a = dcmread(build_implicit_stream(), force=True, private_dictionary=OVERLAY_UL)
        ds_b = dcmread(build_implicit_stream(), force=True, private_dictionary=OVERLAY_US)
        elem_a = ds_a[TAG_VALUE]
        elem_b = ds_b[TAG_VALUE]

        assert elem_a.VR == "UL"
        assert elem_a.value == UL_VALUE
        assert elem_a.name == "[Overlay Unsigned Long]"
        assert elem_b.VR == "US"
        assert list(elem_b.value) == US_VALUES
        assert elem_b.name == "[Overlay Unsigned Short]"

    def test_no_overlay_keeps_old_behaviour(self):
        ds = dcmread(build_implicit_stream(), force=True)
        elem = ds[TAG_VALUE]
        assert elem.VR == "UN"
        assert elem.value == PAYLOAD
        assert elem.name == "Private tag data"

    def test_overlay_takes_precedence_over_global(self):
        with save_private_dict():
            private_dictionaries[CREATOR] = {f"{GROUP:04X}xx01": ("US", "2", "Global Short", "")}
            ds = dcmread(build_implicit_stream(), force=True, private_dictionary=OVERLAY_UL)
            elem = ds[TAG_VALUE]
            assert elem.VR == "UL"
            assert elem.value == UL_VALUE
            assert elem.name == "[Overlay Unsigned Long]"

    def test_unmatched_falls_back_to_global(self):
        with save_private_dict():
            private_dictionaries[CREATOR] = {f"{GROUP:04X}xx03": ("LO", "1", "Global Fallback", "")}
            ds = dcmread(build_implicit_stream(empty_private=True), force=True, private_dictionary=OVERLAY_UL)
            # 0x1003 (offset 3) is only in the global dictionary
            elem = ds[Tag(GROUP, 0x1003)]
            assert elem.VR == "LO"
            assert elem.name == "[Global Fallback]"
            # the overlay entry is still used
            assert ds[TAG_VALUE].VR == "UL"

    def test_unknown_private_keeps_un_behaviour(self):
        ds = dcmread(build_implicit_stream(unknown_private=True), force=True, private_dictionary=OVERLAY_UL)
        unknown = ds[Tag(GROUP, 0x1077)]
        assert unknown.VR == "UN"
        assert unknown.value == b"\x01\x02"
        assert unknown.name == "Private tag data"

    def test_empty_private_value(self):
        ds = dcmread(build_implicit_stream(empty_private=True), force=True, private_dictionary={CREATOR: {(GROUP, 3): ("UL", "1", "Empty Long", "")}})
        elem = ds[Tag(GROUP, 0x1003)]
        assert elem.VR == "UL"
        assert elem.value is None or elem.value == ""
        assert elem.name == "[Empty Long]"

    def test_standard_tags_unaffected(self):
        stream = io.BytesIO(_implicit_element(0x0010, 0x0010, _pad(b"DOE^JOHN")) + _creator_element(GROUP, 0x10, CREATOR) + _implicit_element(GROUP, 0x1001, PAYLOAD))
        ds = dcmread(stream, force=True, private_dictionary=OVERLAY_UL)
        assert ds.PatientName == "DOE^JOHN"
        assert ds[TAG_VALUE].VR == "UL"

    def test_block_agnostic_lookup(self):
        # same offset in another block of the same group must resolve too;
        # each block has its own private creator element (same creator name)
        stream = io.BytesIO(_creator_element(GROUP, 0x10, CREATOR) + _implicit_element(GROUP, 0x1001, PAYLOAD) + _creator_element(GROUP, 0x11, CREATOR) + _implicit_element(GROUP, 0x1101, PAYLOAD))
        ds = dcmread(stream, force=True, private_dictionary=OVERLAY_UL)
        assert ds[Tag(GROUP, 0x1101)].VR == "UL"
        assert ds[Tag(GROUP, 0x1101)].value == UL_VALUE

    def test_distinct_creators_distinct_slots(self):
        ds = dcmread(
            build_implicit_stream(second_creator_block=True),
            force=True,
            private_dictionary=NESTED_OVERLAY,
        )
        # group 0x0043 block 0x20 belongs to OTHER_CREATOR
        elem = ds[TAG_NESTED]
        assert elem.VR == "US"
        assert elem.value == 7
        assert elem.private_creator == OTHER_CREATOR

    def test_global_registration_after_read_does_not_change_name(self):
        with save_private_dict():
            ds = dcmread(build_implicit_stream(), force=True, private_dictionary=OVERLAY_UL)
            elem = ds[TAG_VALUE]
            assert elem.name == "[Overlay Unsigned Long]"
            # register a conflicting global entry after the read
            private_dictionaries[CREATOR] = {f"{GROUP:04X}xx01": ("US", "2", "Late Global Name", "")}
            assert elem.name == "[Overlay Unsigned Long]"
            assert elem.VR == "UL"

    def test_standalone_element_name_is_frozen(self):
        with save_private_dict():
            ds = dcmread(build_implicit_stream(), force=True, private_dictionary=OVERLAY_UL)
            elem = ds[TAG_VALUE]
            detached = copy.copy(elem)
            private_dictionaries[CREATOR] = {f"{GROUP:04X}xx01": ("US", "2", "Late Global Name", "")}
            assert detached.name == "[Overlay Unsigned Long]"
            assert detached.VR == "UL"

    def test_overlay_not_written_to_global(self):
        before = copy.deepcopy(private_dictionaries)
        dcmread(build_implicit_stream(), force=True, private_dictionary=OVERLAY_UL)
        assert private_dictionaries == before


# ---------------------------------------------------------------------------
# Explicit VR: the file's VR wins; the overlay only adds the description
# ---------------------------------------------------------------------------
class TestExplicitVR:
    def test_file_us_not_overridden_by_overlay_ul(self):
        ds = dcmread(build_explicit_stream("US"), force=True, private_dictionary=OVERLAY_UL)
        elem = ds[TAG_VALUE]
        assert elem.VR == "US"
        assert list(elem.value) == US_VALUES
        # description still comes from the overlay
        assert elem.name == "[Overlay Unsigned Long]"
        assert "2" in elem.repval or "1" in elem.repval

    def test_file_ul_respected(self):
        ds = dcmread(build_explicit_stream("UL"), force=True, private_dictionary=OVERLAY_US)
        elem = ds[TAG_VALUE]
        assert elem.VR == "UL"
        assert elem.value == UL_VALUE
        assert elem.name == "[Overlay Unsigned Short]"

    def test_explicit_un_resolved_through_binding(self):
        name = _pad(CREATOR.encode("ascii"))
        stream = io.BytesIO(_explicit_element(GROUP, 0x10, "LO", name) + _explicit_element(GROUP, 0x1001, "UN", PAYLOAD))
        ds = dcmread(stream, force=True, private_dictionary=OVERLAY_UL)
        elem = ds[TAG_VALUE]
        assert elem.VR == "UL"
        assert elem.value == UL_VALUE
        assert elem.name == "[Overlay Unsigned Long]"


# ---------------------------------------------------------------------------
# Nested private sequences, defined and undefined length, two levels deep
# ---------------------------------------------------------------------------
class TestNestedSequences:
    @pytest.mark.parametrize("undefined_item", [False, True])
    @pytest.mark.parametrize("undefined_sq", [False, True])
    def test_private_sq_two_levels(self, undefined_item, undefined_sq):
        stream = build_implicit_stream(
            with_sq=True,
            undefined_item=undefined_item,
            undefined_sq=undefined_sq,
        )
        ds = dcmread(
            stream,
            force=True,
            private_dictionary={
                CREATOR: {
                    (GROUP, 0x01): ("UL", "1", "Outer Long", ""),
                    (GROUP, 0x10): ("SQ", "1", "Private Sequence", ""),
                    (NESTED_GROUP, 0x02): ("UL", "1", "Nested Long", ""),
                }
            },
        )
        outer = ds[TAG_VALUE]
        assert outer.VR == "UL" and outer.value == UL_VALUE

        sq_elem = ds[TAG_SQ]
        assert sq_elem.VR == "SQ"
        item = sq_elem.value[0]
        nested = item[TAG_NESTED]
        assert nested.VR == "UL"
        assert nested.value == UL_VALUE
        assert nested.name == "[Nested Long]"
        # the very same binding object is shared, no per-item copy
        assert item._private_binding is ds._private_binding

    def test_nested_sq_has_own_creators(self):
        # inner dataset registers its own creator at block 0x20
        ds = dcmread(
            build_implicit_stream(with_sq=True),
            force=True,
            private_dictionary={
                CREATOR: {
                    (GROUP, 0x01): ("UL", "1", "Outer", ""),
                    (GROUP, 0x10): ("SQ", "1", "Seq", ""),
                    (NESTED_GROUP, 0x02): ("US", "2", "Nested Shorts", ""),
                }
            },
        )
        nested = ds[TAG_SQ].value[0][TAG_NESTED]
        assert nested.VR == "US"
        assert list(nested.value) == US_VALUES

    def test_undefined_length_sq_identified_by_binding(self):
        # The binding (not the item look-ahead alone) must mark an implicit
        # undefined-length private element as SQ, and carry its creator.
        name = _pad(CREATOR.encode("ascii"))
        inner = _implicit_element(NESTED_GROUP, 0x20, name) + _implicit_element(NESTED_GROUP, 0x2002, PAYLOAD)
        sq_body = _item(inner, undefined=True) + _item_delimiter()
        sq_body += _sequence_delimiter()
        stream = io.BytesIO(_implicit_element(GROUP, 0x10, name) + struct.pack("<HHI", GROUP, 0x1010, 0xFFFFFFFF) + sq_body)
        ds = dcmread(
            stream,
            force=True,
            private_dictionary={
                CREATOR: {
                    (GROUP, 0x10): ("SQ", "1", "Seq", ""),
                    (NESTED_GROUP, 0x02): ("UL", "1", "Nested", ""),
                }
            },
        )
        sq_elem = ds[TAG_SQ]
        assert sq_elem.VR == "SQ"
        assert sq_elem.private_creator == CREATOR
        nested = sq_elem.value[0][TAG_NESTED]
        assert nested.VR == "UL"
        assert nested.value == UL_VALUE


# ---------------------------------------------------------------------------
# specific_tags: selected private tags work without the creator being returned
# ---------------------------------------------------------------------------
class TestSpecificTags:
    def test_private_tag_interpretable_without_creator_in_result(self):
        ds = dcmread(
            build_implicit_stream(),
            force=True,
            specific_tags=[TAG_VALUE],
            private_dictionary=OVERLAY_UL,
        )
        assert TAG_VALUE in ds
        # the private creator must not sneak into the returned tag set
        assert Tag(GROUP, 0x10) not in ds
        elem = ds[TAG_VALUE]
        assert elem.VR == "UL"
        assert elem.value == UL_VALUE
        assert elem.name == "[Overlay Unsigned Long]"

    def test_specific_tags_inside_sequence(self):
        ds = dcmread(
            build_implicit_stream(with_sq=True),
            force=True,
            specific_tags=[TAG_SQ],
            private_dictionary={
                CREATOR: {
                    (GROUP, 0x10): ("SQ", "1", "Seq", ""),
                    (NESTED_GROUP, 0x02): ("UL", "1", "Nested", ""),
                }
            },
        )
        assert TAG_SQ in ds
        assert Tag(GROUP, 0x10) not in ds
        nested = ds[TAG_SQ].value[0][TAG_NESTED]
        assert nested.VR == "UL"
        assert nested.value == UL_VALUE


# ---------------------------------------------------------------------------
# Deferred reads keep the frozen interpretation
# ---------------------------------------------------------------------------
class TestDeferred:
    def test_deferred_value_uses_binding(self, tmp_path):
        path = tmp_path / "deferred.dcm"
        path.write_bytes(build_implicit_stream().getvalue())
        ds = dcmread(str(path), defer_size=2, force=True, private_dictionary=OVERLAY_UL)
        raw = ds.get_item(TAG_VALUE, keep_deferred=True)
        assert isinstance(raw, RawDataElement)
        assert raw.value is None
        elem = ds[TAG_VALUE]  # triggers the deferred re-read
        assert elem.VR == "UL"
        assert elem.value == UL_VALUE
        assert elem.name == "[Overlay Unsigned Long]"

    def test_deferred_with_conflicting_global_change(self, tmp_path):
        with save_private_dict():
            path = tmp_path / "deferred.dcm"
            path.write_bytes(build_implicit_stream().getvalue())
            ds = dcmread(str(path), defer_size=2, force=True, private_dictionary=OVERLAY_UL)
            private_dictionaries[CREATOR] = {f"{GROUP:04X}xx01": ("US", "2", "Global", "")}
            elem = ds[TAG_VALUE]
            assert elem.VR == "UL"
            assert elem.value == UL_VALUE

    def test_deferred_creator_itself_is_not_deferred(self, tmp_path):
        path = tmp_path / "creator.dcm"
        path.write_bytes(build_implicit_stream().getvalue())
        ds = dcmread(str(path), defer_size=1, force=True, private_dictionary=OVERLAY_UL)
        # the value element is deferred, but the creator must stay loaded
        value_raw = ds.get_item(TAG_VALUE, keep_deferred=True)
        assert isinstance(value_raw, RawDataElement) and value_raw.value is None
        creator_raw = ds.get_item(Tag(GROUP, 0x10), keep_deferred=True)
        assert isinstance(creator_raw, RawDataElement)
        assert creator_raw.value is not None
        assert ds[TAG_VALUE].VR == "UL"

    def test_deferred_nested_private_element(self, tmp_path):
        path = tmp_path / "nested.dcm"
        path.write_bytes(build_implicit_stream(with_sq=True).getvalue())
        ds = dcmread(
            str(path),
            defer_size=2,
            force=True,
            private_dictionary={
                CREATOR: {
                    (GROUP, 1): ("UL", "1", "Outer", ""),
                    (GROUP, 0x10): ("SQ", "1", "Seq", ""),
                    (NESTED_GROUP, 2): ("UL", "1", "Nested", ""),
                }
            },
        )
        nested = ds[TAG_SQ].value[0][TAG_NESTED]
        assert nested.VR == "UL"
        assert nested.value == UL_VALUE
        assert nested.name == "[Nested]"


# ---------------------------------------------------------------------------
# Copy / deepcopy / pickle retain the binding semantics
# ---------------------------------------------------------------------------
class TestCopyAndPickle:
    def test_shallow_copy_retains_interpretation(self):
        ds = dcmread(build_implicit_stream(), force=True, private_dictionary=OVERLAY_UL)
        shallow = ds.copy()
        assert shallow._private_binding is ds._private_binding
        assert shallow[TAG_VALUE].VR == "UL"
        assert shallow[TAG_VALUE].name == "[Overlay Unsigned Long]"

    def test_shallow_copy_mutation_independence(self):
        ds = dcmread(
            build_implicit_stream(empty_private=True),
            force=True,
            private_dictionary={
                CREATOR: {
                    (GROUP, 1): ("UL", "1", "V1", ""),
                    (GROUP, 3): ("UL", "1", "V3", ""),
                }
            },
        )
        shallow = ds.copy()
        # the shared mutable value container follows the old shallow semantics:
        # deleting an element on the copy deletes it from the shared _dict,
        # but per-dataset state such as the binding reference is independent
        del shallow[TAG_VALUE]
        assert TAG_VALUE not in shallow
        # give the copy its own binding state; the original must be untouched
        shallow._private_binding = None
        shallow._private_creator_map = {}
        assert ds._private_binding is not None
        assert ds[Tag(GROUP, 0x1003)].VR == "UL"
        # mutable values that were already converted remain shared, as before
        assert shallow._private_binding is None
        assert ds._private_binding is not None

    def test_deepcopy_retains_interpretation(self):
        ds = dcmread(
            build_implicit_stream(with_sq=True),
            force=True,
            private_dictionary={
                CREATOR: {
                    (GROUP, 1): ("UL", "1", "Outer", ""),
                    (GROUP, 0x10): ("SQ", "1", "Seq", ""),
                    (NESTED_GROUP, 2): ("UL", "1", "Nested", ""),
                }
            },
        )
        duplicate = copy.deepcopy(ds)
        assert duplicate[TAG_VALUE].VR == "UL"
        nested = duplicate[TAG_SQ].value[0][TAG_NESTED]
        assert nested.VR == "UL"
        assert nested.value == UL_VALUE
        assert duplicate._private_binding is not None

    def test_pickle_roundtrip_in_new_process_namespace(self, tmp_path):
        with save_private_dict():
            ds = dcmread(build_implicit_stream(), force=True, private_dictionary=OVERLAY_UL)
            data = pickle.dumps(ds)
            restored = pickle.loads(data)
            # simulate a different process: clear the global registry
            private_dictionaries.clear()
            elem = restored[TAG_VALUE]
            assert elem.VR == "UL"
            assert elem.value == UL_VALUE
            assert elem.name == "[Overlay Unsigned Long]"

    def test_pickle_subprocess(self, tmp_path):
        """The pickled dataset must be usable by a fresh interpreter."""
        import subprocess
        import sys
        import pydicom

        ds = dcmread(build_implicit_stream(), force=True, private_dictionary=OVERLAY_UL)
        pkl = tmp_path / "ds.pkl"
        pkl.write_bytes(pickle.dumps(ds))
        # -I isolates the interpreter, so make the source tree importable
        # explicitly inside the script
        src_path = list(pydicom.__path__)[0].rsplit("pydicom", 1)[0]
        script = (
            "import sys; sys.path.insert(0, %r); "
            "import pickle; "
            "ds = pickle.loads(open(sys.argv[1], 'rb').read()); "
            "e = ds[%d]; "
            "assert e.VR == 'UL' and e.value == %d and "
            "e.name == '[Overlay Unsigned Long]'; "
            "print('subprocess ok')"
        ) % (src_path, int(TAG_VALUE), UL_VALUE)
        result = subprocess.run(
            [sys.executable, "-I", "-c", script, str(pkl)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "subprocess ok" in result.stdout


# ---------------------------------------------------------------------------
# Round trip: write explicit VR LE and re-read with the standard reader
# ---------------------------------------------------------------------------
class TestWriteRoundTrip:
    def test_explicit_output_read_back_without_overlay(self, tmp_path):
        ds = dcmread(build_implicit_stream(), force=True, private_dictionary=OVERLAY_UL)
        # write explicit VR little endian without enforcing file meta
        # (the test stream itself has no file meta)
        path = tmp_path / "roundtrip.dcm"
        dcmwrite(path, ds, implicit_vr=False, little_endian=True)

        reread = dcmread(path, force=True)  # no private_dictionary argument
        elem = reread[TAG_VALUE]
        # explicit VR was written from the interpreted element
        assert elem.VR == "UL"
        assert elem.value == UL_VALUE
        assert isinstance(reread, Dataset)

    def test_sq_roundtrip_preserves_hierarchy(self, tmp_path):
        overlay_dict = {
            CREATOR: {
                (GROUP, 1): ("UL", "1", "Outer", ""),
                (GROUP, 0x10): ("SQ", "1", "Seq", ""),
                (NESTED_GROUP, 2): ("UL", "1", "Nested", ""),
            }
        }
        ds = dcmread(build_implicit_stream(with_sq=True), force=True, private_dictionary=overlay_dict)
        path = tmp_path / "sq.dcm"
        dcmwrite(path, ds, implicit_vr=False, little_endian=True)

        reread = dcmread(path, force=True)
        assert reread[TAG_SQ].VR == "SQ"
        nested = reread[TAG_SQ].value[0][TAG_NESTED]
        assert nested.VR == "UL"
        assert nested.value == UL_VALUE


# ---------------------------------------------------------------------------
# Threads: different bindings concurrently, events coordinate the overlap
# ---------------------------------------------------------------------------
class TestThreads:
    def test_concurrent_bindings_do_not_cross_contaminate(self):
        # Both threads parse many interleaved streams behind a barrier so the
        # reads genuinely overlap in time, while a conflicting global entry is
        # present.
        barrier = threading.Barrier(2)
        errors = []

        with save_private_dict():
            private_dictionaries[CREATOR] = {f"{GROUP:04X}xx01": ("LO", "1", "Global", "")}

            def worker(overlay, expected_vr, expected_value):
                try:
                    barrier.wait()
                    for _ in range(100):
                        ds = dcmread(build_implicit_stream(), force=True, private_dictionary=overlay)
                        elem = ds[TAG_VALUE]
                        assert elem.VR == expected_vr
                        if expected_vr == "UL":
                            assert elem.value == expected_value
                        else:
                            assert list(elem.value) == expected_value
                except BaseException as exc:  # pragma: no cover - reported
                    errors.append(exc)

            t_ul = threading.Thread(target=worker, args=(OVERLAY_UL, "UL", UL_VALUE))
            t_us = threading.Thread(target=worker, args=(OVERLAY_US, "US", US_VALUES))
            t_ul.start()
            t_us.start()
            t_ul.join(20)
            t_us.join(20)
            assert not t_ul.is_alive() and not t_us.is_alive()
            assert errors == []

    def test_concurrent_lazy_conversion_uses_thread_local_binding(self):
        # Keep the elements raw, then convert from two threads behind a barrier
        # so the conversions overlap; each must see its own binding.
        from pydicom.datadict import get_active_private_binding

        ds_a = dcmread(build_implicit_stream(), force=True, private_dictionary=OVERLAY_UL)
        ds_b = dcmread(build_implicit_stream(), force=True, private_dictionary=OVERLAY_US)
        barrier = threading.Barrier(2)
        results = {}
        errors = []

        def worker(name, ds, expected_vr):
            try:
                barrier.wait()
                elem = ds[TAG_VALUE]
                # the binding is active only for the duration of conversion
                results[name] = (elem.VR, get_active_private_binding() is None)
                assert elem.VR == expected_vr
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

        t1 = threading.Thread(target=worker, args=("A", ds_a, "UL"))
        t2 = threading.Thread(target=worker, args=("B", ds_b, "US"))
        t1.start()
        t2.start()
        t1.join(10)
        t2.join(10)
        assert errors == []
        assert results == {"A": ("UL", True), "B": ("US", True)}

    def test_failed_read_does_not_pollute_later_reads(self):
        with pytest.raises(ValueError):
            dcmread(
                build_implicit_stream(),
                force=True,
                private_dictionary={CREATOR: {(GROUP, 999): ("UL", "1", "x", "")}},
            )
        assert get_active_private_binding() is None
        ds = dcmread(build_implicit_stream(), force=True, private_dictionary=OVERLAY_UL)
        assert ds[TAG_VALUE].VR == "UL"


# ---------------------------------------------------------------------------
# Memory: one shared binding for N datasets; growth must be O(N), not O(K*N)
# ---------------------------------------------------------------------------
class TestMemoryFootprint:
    def _deep_stream(self, items: int) -> io.BytesIO:
        inner = _creator_element(NESTED_GROUP, 0x20, CREATOR)
        inner += _implicit_element(NESTED_GROUP, 0x2002, PAYLOAD)
        seq = b"".join(_item(inner) for _ in range(items))
        out = _creator_element(GROUP, 0x10, CREATOR)
        out += _implicit_element(GROUP, 0x1001, PAYLOAD)
        out += _implicit_element(GROUP, 0x1010, seq)
        return io.BytesIO(out)

    def test_binding_shared_across_all_datasets(self):
        big_overlay = {
            CREATOR: {
                (GROUP, 1): ("UL", "1", "Outer", ""),
                (GROUP, 0x10): ("SQ", "1", "Seq", ""),
                (NESTED_GROUP, 2): ("UL", "1", "Nested", ""),
            }
        }
        # pad the overlay to a large K (private groups must stay odd)
        extra_groups = range(0x0045, 0x0045 + 2 * 200, 2)
        for group in extra_groups:
            big_overlay[CREATOR][(group, 1)] = ("UL", "1", f"d{group}", "")

        n_items = 50
        ds = dcmread(self._deep_stream(n_items), force=True, private_dictionary=big_overlay)
        item_datasets = ds[TAG_SQ].value
        assert len(item_datasets) == n_items
        root_binding = ds._private_binding
        assert all(item._private_binding is root_binding for item in item_datasets)

    def test_footprint_grows_linearly_with_item_count(self):
        import sys

        overlay_dict = {
            CREATOR: {
                (GROUP, 1): ("UL", "1", "Outer", ""),
                (GROUP, 0x10): ("SQ", "1", "Seq", ""),
                (NESTED_GROUP, 2): ("UL", "1", "Nested", ""),
            }
        }
        # K is intentionally large so an O(K*N) regression would be obvious
        extra_groups = range(0x0045, 0x0045 + 2 * 500, 2)
        for group in extra_groups:
            overlay_dict[CREATOR][(group, 1)] = ("UL", "1", f"d{group}", "")

        def measure(items: int) -> int:
            ds = dcmread(self._deep_stream(items), force=True, private_dictionary=overlay_dict)
            seen = set()
            total = 0
            stack = [ds]
            while stack:
                current = stack.pop()
                if id(current) in seen:
                    continue
                seen.add(id(current))
                total += sys.getsizeof(current)
                for elem in current.elements():
                    if elem.VR == "SQ":
                        stack.extend(elem.value)
            return total

        small = measure(10)
        large = measure(100)
        # 10x the items must not approach 10x the total footprint (each extra
        # item is a small dataset sharing one single binding object)
        assert large < small * 5
        # and the delta per item is far below the cost of one overlay copy
        per_item = (large - small) / 90
        overlay_cost = sys.getsizeof(overlay_dict)
        assert per_item < overlay_cost
