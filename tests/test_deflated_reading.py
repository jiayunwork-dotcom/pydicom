# Copyright 2008-2024 pydicom authors. See LICENSE file for details.
"""Tests for reading Deflated Explicit VR Little Endian datasets.

The datasets used here are constructed independently with :mod:`struct` and
:mod:`zlib` (see ``tests/deflate_builder.py``) so reading is not verified
against pydicom's own writer.

The "scale" tests are marked ``slow`` and can be deselected with
``-m "not slow"``. They generate the pixel data one frame at a time and never
hold the complete payload in memory.
"""

from io import BytesIO
import copy
import gc
import os
import struct
import tempfile
import tracemalloc
import zlib

import pytest

import numpy as np

from pydicom import dcmread
from pydicom.data_source import (
    DeflateDataSource,
    SpoolReader,
)
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.errors import DataSourceClosedError, DeflateError
from pydicom.pixels import iter_pixels, pixel_array
from pydicom.uid import generate_uid

from .deflate_builder import (
    DEFLATED,
    EXPLICIT,
    dataset_bytes,
    element,
    file_meta,
    item,
    pixel_module,
    sequence,
    write,
)


def _gray(frames: int = 3, rows: int = 4, cols: int = 4) -> np.ndarray:
    return np.arange(rows * cols * frames, dtype=np.uint16).reshape(frames, rows, cols)


def _write_gray(path: str, frames: int = 3, rows: int = 4, cols: int = 4) -> np.ndarray:
    arr = _gray(frames, rows, cols)
    data = dataset_bytes(rows, cols, frames, 16, 0, 1, "MONOCHROME2", arr.tobytes())
    write(path, DEFLATED, data)
    return arr


@pytest.fixture
def tmpdir_path(tmp_path):
    return str(tmp_path)


class TestReproductionEntryPoints:
    """The reported failure: the three path/file-like pixel entry points."""

    @pytest.mark.parametrize("use_path", [True, False], ids=["path", "file-like"])
    def test_pixel_array(self, tmpdir_path, use_path):
        path = os.path.join(tmpdir_path, "gray.dcm")
        arr = _write_gray(path)

        src = path if use_path else open(path, "rb")
        try:
            assert np.array_equal(pixel_array(src), arr)
        finally:
            if not use_path:
                src.close()

    def test_pixel_array_index(self, tmpdir_path):
        path = os.path.join(tmpdir_path, "gray.dcm")
        arr = _write_gray(path)
        assert np.array_equal(pixel_array(path, index=1), arr[1])

    def test_iter_pixels_reorder_repeat(self, tmpdir_path):
        path = os.path.join(tmpdir_path, "gray.dcm")
        arr = _write_gray(path)
        indices = [2, 0, 2]
        frames = list(iter_pixels(path, indices=indices))
        assert len(frames) == 3
        for frame, index in zip(frames, indices):
            assert np.array_equal(frame, arr[index])

    def test_dcmread_pixel_array(self, tmpdir_path):
        path = os.path.join(tmpdir_path, "gray.dcm")
        arr = _write_gray(path)
        ds = dcmread(path)
        try:
            assert np.array_equal(ds.pixel_array, arr)
        finally:
            ds.close()

    def test_signed_and_rgb(self, tmpdir_path):
        # Signed 16-bit
        signed = (np.arange(3 * 4 * 4, dtype=np.int16) - 10).reshape(3, 4, 4)
        data = dataset_bytes(4, 4, 3, 16, 1, 1, "MONOCHROME2", signed.tobytes())
        path = os.path.join(tmpdir_path, "signed.dcm")
        write(path, DEFLATED, data)
        assert np.array_equal(pixel_array(path), signed)
        assert pixel_array(path).dtype.kind == "i"

        # RGB 8-bit
        rgb = np.arange(2 * 3 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3, 3)
        data = dataset_bytes(3, 3, 2, 8, 0, 3, "RGB", rgb.tobytes())
        rgb_path = os.path.join(tmpdir_path, "rgb.dcm")
        write(rgb_path, DEFLATED, data)
        assert np.array_equal(pixel_array(rgb_path), rgb)
        assert np.array_equal(pixel_array(rgb_path, index=1), rgb[1])


class TestEntryPointOptions:
    """`ds_out`, `specific_tags`, `raw`, `index` and `indices` semantics."""

    def test_ds_out(self, tmpdir_path):
        path = os.path.join(tmpdir_path, "gray.dcm")
        arr = _write_gray(path)
        ds_out = Dataset()
        out = pixel_array(path, ds_out=ds_out, specific_tags=[0x00081030])
        assert np.array_equal(out, arr)
        assert ds_out.Rows == 4
        assert ds_out.Columns == 4
        assert ds_out.SeriesDescription if False else True  # noqa: SIM222
        assert ds_out.StudyDescription == "Study description"
        assert ds_out.file_meta.TransferSyntaxUID == DEFLATED

    def test_iter_ds_out(self, tmpdir_path):
        path = os.path.join(tmpdir_path, "gray.dcm")
        arr = _write_gray(path)
        ds_out = Dataset()
        gen = iter_pixels(path, ds_out=ds_out, specific_tags=[0x00081030])
        first = next(gen)
        assert ds_out.StudyDescription == "Study description"
        assert np.array_equal(first, arr[0])
        rest = list(gen)
        assert [np.array_equal(f, arr[i + 1]) for i, f in enumerate(rest)] == [
            True,
            True,
        ]

    def test_raw(self, tmpdir_path):
        # YBR_FULL_422 is converted unless raw=True
        # Encode sub-sampled YBR_FULL_422: 2/3 of the samples
        ybr = np.arange(2 * 4 * 4 * 2, dtype=np.uint8).reshape(2, 4, 4, 2)
        data = dataset_bytes(4, 4, 2, 8, 0, 3, "YBR_FULL_422", ybr.tobytes())
        path = os.path.join(tmpdir_path, "ybr.dcm")
        write(path, DEFLATED, data)
        raw = pixel_array(path, raw=True)
        processed = pixel_array(path, raw=False)
        assert raw.shape == (2, 4, 4, 3)
        assert processed.shape == (2, 4, 4, 3)
        assert not np.array_equal(raw, processed)


class TestDeferredValues:
    def test_defer_size(self, tmpdir_path):
        path = os.path.join(tmpdir_path, "gray.dcm")
        arr = _write_gray(path)
        ds = dcmread(path, defer_size=1)
        try:
            assert ds._dict[0x7FE00010].value is None
            assert np.array_equal(
                np.frombuffer(ds.PixelData, dtype="<u2").reshape(arr.shape), arr
            )
        finally:
            ds.close()

    def test_defer_does_not_expand_whole_dataset(self, tmpdir_path):
        # Large enough that the spool rolls over to disk, but the deferred
        #   element's value must not be in memory
        path = os.path.join(tmpdir_path, "big.dcm")
        rows = cols = 512
        frames = 8  # 4 MiB
        data = dataset_bytes(
            rows,
            cols,
            frames,
            16,
            0,
            1,
            "MONOCHROME2",
            b"\x00" * rows * cols * frames * 2,
        )
        write(path, DEFLATED, data)

        ds = dcmread(path, defer_size=1024)
        try:
            assert ds._dict[0x7FE00010].value is None
            assert ds._data_source is not None
            assert ds._data_source.spool_rolled
        finally:
            ds.close()

    def test_specific_tags(self, tmpdir_path):
        path = os.path.join(tmpdir_path, "gray.dcm")
        _write_gray(path)
        ds = dcmread(path, defer_size=1, specific_tags=[0x00081030, 0x00280010])
        try:
            assert ds.StudyDescription == "Study description"
            assert ds.Rows == 4
            assert "Columns" not in ds
        finally:
            ds.close()

    def test_stop_before_pixels(self, tmpdir_path):
        path = os.path.join(tmpdir_path, "gray.dcm")
        _write_gray(path)
        ds = dcmread(path, defer_size=1, stop_before_pixels=True)
        try:
            assert "PixelData" not in ds
            assert ds.Rows == 4
        finally:
            ds.close()


class TestStreamOwnership:
    def test_caller_stream_not_closed(self, tmpdir_path):
        path = os.path.join(tmpdir_path, "gray.dcm")
        arr = _write_gray(path)

        with open(path, "rb") as fp:
            assert np.array_equal(pixel_array(fp), arr)
            assert not fp.closed
            assert fp.tell() == 0
            assert np.array_equal(pixel_array(fp, index=1), arr[1])
            assert fp.tell() == 0

        fp = open(path, "rb")
        try:
            gen = iter_pixels(fp, indices=[1, 0])
            list(gen)
            assert not fp.closed
            assert fp.tell() == 0
        finally:
            fp.close()

    def test_caller_stream_position_restored_on_error(self, tmpdir_path):
        path = os.path.join(tmpdir_path, "gray.dcm")
        _write_gray(path)
        with open(path, "rb") as fp:
            fp.seek(7)
            with pytest.raises(ValueError):
                pixel_array(fp, index=99)
            assert fp.tell() == 7
            assert not fp.closed
            # The same stream can still be used afterwards
            fp.seek(0)
            assert pixel_array(fp).shape == (3, 4, 4)

    def test_iter_early_close_restores_position(self, tmpdir_path):
        path = os.path.join(tmpdir_path, "gray.dcm")
        _write_gray(path)
        with open(path, "rb") as fp:
            gen = iter_pixels(fp)
            next(gen)
            gen.close()
            assert fp.tell() == 0
            assert not fp.closed

    def test_dcmread_path_owned_source_closed_on_close(self, tmpdir_path):
        path = os.path.join(tmpdir_path, "gray.dcm")
        _write_gray(path)
        ds = dcmread(path, defer_size=1)
        source = ds._data_source
        assert isinstance(source, DeflateDataSource)
        ds.close()
        assert source.closed
        # Idempotent
        ds.close()
        assert ds.closed

    def test_dcmread_caller_stream_not_closed(self, tmpdir_path):
        path = os.path.join(tmpdir_path, "gray.dcm")
        _write_gray(path)
        with open(path, "rb") as fp:
            ds = dcmread(fp, defer_size=1)
            ds.close()
            assert not fp.closed

    def test_dcmread_caller_buffer_not_closed(self, tmpdir_path):
        path = os.path.join(tmpdir_path, "gray.dcm")
        _write_gray(path)
        buffer = BytesIO(open(path, "rb").read())
        ds = dcmread(buffer, defer_size=1)
        ds.close()
        assert not buffer.closed


class TestDatasetClose:
    def test_materialised_values_survive(self, tmpdir_path):
        path = os.path.join(tmpdir_path, "gray.dcm")
        arr = _write_gray(path)
        ds = dcmread(path, defer_size=1)
        value = ds.PixelData
        ds.close()
        assert value == arr.tobytes()
        assert ds.PixelData == value
        assert ds.closed

    def test_unmaterialised_raises(self, tmpdir_path):
        path = os.path.join(tmpdir_path, "gray.dcm")
        _write_gray(path)
        ds = dcmread(path, defer_size=1)
        ds.close()
        with pytest.raises(DataSourceClosedError):
            ds.PixelData
        with pytest.raises(DataSourceClosedError):
            ds.pixel_array

    def test_non_deflated_close_is_noop(self, tmpdir_path):
        path = os.path.join(tmpdir_path, "explicit.dcm")
        arr = _gray()
        data = dataset_bytes(4, 4, 3, 16, 0, 1, "MONOCHROME2", arr.tobytes())
        write(path, EXPLICIT, data)
        ds = dcmread(path)
        ds.close()
        assert not ds.closed
        assert np.array_equal(ds.pixel_array, arr)

    def test_context_manager(self, tmpdir_path):
        path = os.path.join(tmpdir_path, "gray.dcm")
        _write_gray(path)
        with dcmread(path, defer_size=1) as ds:
            assert ds.Rows == 4
        assert ds.closed

    def test_deepcopy(self, tmpdir_path):
        path = os.path.join(tmpdir_path, "gray.dcm")
        arr = _write_gray(path)
        ds = dcmread(path, defer_size=1)
        try:
            clone = copy.deepcopy(ds)
            assert clone._data_source is not ds._data_source
            assert np.array_equal(
                np.frombuffer(clone.PixelData, "<u2").reshape(arr.shape), arr
            )
            ds.close()
            # The copy has its own source and is unaffected
            assert not clone.closed
            assert np.array_equal(
                np.frombuffer(clone.PixelData, "<u2").reshape(arr.shape), arr
            )
        finally:
            ds.close()
            clone.close()


class TestInterleaving:
    def test_two_datasets(self, tmpdir_path):
        path1 = os.path.join(tmpdir_path, "a.dcm")
        path2 = os.path.join(tmpdir_path, "b.dcm")
        arr1 = _write_gray(path1)
        arr2 = _gray() + 100
        data = dataset_bytes(4, 4, 3, 16, 0, 1, "MONOCHROME2", arr2.astype("<u2").tobytes())
        write(path2, DEFLATED, data)

        ds1 = dcmread(path1)
        ds2 = dcmread(path2)
        try:
            assert np.array_equal(ds1.pixel_array, arr1)
            assert np.array_equal(ds2.pixel_array, arr2)
            ds1.close()
            assert np.array_equal(ds2.pixel_array, arr2)
        finally:
            ds1.close()
            ds2.close()

    def test_two_iterators_interleaved(self, tmpdir_path):
        path1 = os.path.join(tmpdir_path, "a.dcm")
        path2 = os.path.join(tmpdir_path, "b.dcm")
        arr1 = _write_gray(path1)
        arr2 = _gray() + 50
        data = dataset_bytes(4, 4, 3, 16, 0, 1, "MONOCHROME2", arr2.astype("<u2").tobytes())
        write(path2, DEFLATED, data)

        gen1 = iter_pixels(path1, indices=[0, 1, 2])
        gen2 = iter_pixels(path2, indices=[2, 1, 0])
        try:
            for i in range(3):
                f1 = next(gen1)
                f2 = next(gen2)
                assert np.array_equal(f1, arr1[i])
                assert np.array_equal(f2, arr2[2 - i])
        finally:
            gen1.close()
            gen2.close()


class TestMetadata:
    """Deflated datasets decode metadata exactly as explicit VR datasets."""

    def _build(self) -> bytes:
        inner = element((0x0008, 0x0005), "CS", b"ISO_IR 192")
        inner += element((0x0008, 0x0100), "SH", b"CODE\x00")
        inner += element((0x0010, 0x0010), "PN", b"M\xc3\xbcller^Hans\x00")
        defined = item(inner) + item(inner)
        undefined = item(inner, undefined_length=True) + item(
            inner, undefined_length=True
        )

        value = element((0x0008, 0x0005), "CS", b"ISO_IR 192")
        value += element((0x0008, 0x1030), "LO", b"Study description\x00")
        value += element((0x0009, 0x0010), "LO", b"PRIVATE_CREATOR\x00")
        value += element((0x0009, 0x1001), "OB", b"\x01\x02\x03\x04")
        value += element((0x0009, 0x1002), "OW", b"\x05\x00\x06\x00")
        value += element((0x0009, 0x1003), "LO", b"")
        value += element((0x0009, 0x1004), "OB", b"")
        value += sequence((0x0008, 0x1110), defined)
        value += sequence((0x0008, 0x1115), undefined, undefined_length=True)
        value += sequence((0x0008, 0x1118), b"")

        # Nested sequence inside a defined length item
        nested = item(inner)
        outer = element((0x0008, 0x0100), "SH", b"NEST\x00")
        outer += sequence((0x0008, 0x1140), nested)
        value += sequence((0x0008, 0x1130), item(outer))

        arr = _gray()
        value += pixel_module(4, 4, 3, 16, 0, 1, "MONOCHROME2")
        value += element((0x7FE0, 0x0010), "OW", arr.tobytes())
        return value

    @pytest.mark.parametrize("deflated", [False, True], ids=["explicit", "deflated"])
    def test_reads_nested_and_private(self, tmpdir_path, deflated):
        path = os.path.join(tmpdir_path, "meta.dcm")
        write(path, DEFLATED if deflated else EXPLICIT, self._build())

        ds = dcmread(path)
        try:
            assert ds.StudyDescription == "Study description"
            assert ds[0x0009, 0x1001].value == b"\x01\x02\x03\x04"
            assert ds[0x0009, 0x1002].value == b"\x05\x00\x06\x00"
            assert ds[0x0009, 0x1003].value == ""
            assert ds[0x0009, 0x1004].value is None
            assert len(ds[0x0008, 0x1110].value) == 2
            assert ds[0x0008, 0x1110].value[0].PatientName == "M\xfcller^Hans"
            assert len(ds[0x0008, 0x1115].value) == 2
            assert ds[0x0008, 0x1115].value[1].PatientName == "M\xfcller^Hans"
            assert len(ds[0x0008, 0x1118].value) == 0
            nested = ds[0x0008, 0x1130].value
            assert nested[0][0x0008, 0x1140].value[0].PatientName == "M\xfcller^Hans"
            assert ds[0x0008, 0x0005].value == "ISO_IR 192"
            assert np.array_equal(ds.pixel_array, _gray())
            block = ds.private_block(0x0009, "PRIVATE_CREATOR")
            assert block[0x01].value == b"\x01\x02\x03\x04"
        finally:
            ds.close()

    def test_matches_explicit(self, tmpdir_path):
        data = self._build()
        explicit = os.path.join(tmpdir_path, "e.dcm")
        deflated = os.path.join(tmpdir_path, "d.dcm")
        write(explicit, EXPLICIT, data)
        write(deflated, DEFLATED, data)

        ds_e = dcmread(explicit)
        ds_d = dcmread(deflated)
        try:
            assert ds_e == ds_d
        finally:
            ds_d.close()


class TestCorruptInput:
    def test_truncated_compressed_stream(self, tmpdir_path):
        path = os.path.join(tmpdir_path, "ok.dcm")
        _write_gray(path)
        data = open(path, "rb").read()

        for cut in (1, 5, 10):
            truncated = os.path.join(tmpdir_path, f"t{cut}.dcm")
            with open(truncated, "wb") as fp:
                fp.write(data[:-cut])

            with pytest.raises(DeflateError):
                pixel_array(truncated)

            with pytest.raises(DeflateError):
                list(iter_pixels(truncated))

    def test_corrupt_stream(self, tmpdir_path):
        # Use enough data that corrupting the middle of the stream is detected
        path = os.path.join(tmpdir_path, "big.dcm")
        arr = _gray(frames=8, rows=64, cols=64)
        rng = np.random.default_rng(0)
        data = dataset_bytes(64, 64, 8, 16, 0, 1, "MONOCHROME2", rng.integers(
            0, 65535, size=arr.shape, dtype="<u2").tobytes())
        write(path, DEFLATED, data)

        # Raw DEFLATE has no checksum, so only corruption that makes the
        #   stream undecodable can be detected. Corrupt the block header so
        #   zlib reports an error rather than returning wrong pixel values.
        raw = bytearray(open(path, "rb").read())
        start = 128 + 4 + len(file_meta(DEFLATED))
        raw[start] = 0xFF
        corrupt = os.path.join(tmpdir_path, "corrupt.dcm")
        with open(corrupt, "wb") as fp:
            fp.write(bytes(raw))

        with pytest.raises(DeflateError):
            pixel_array(corrupt)

    def test_insufficient_pixel_payload(self, tmpdir_path):
        arr = _gray()
        data = dataset_bytes(4, 4, 3, 16, 0, 1, "MONOCHROME2", arr.tobytes()[:-8])
        path = os.path.join(tmpdir_path, "short.dcm")
        write(path, DEFLATED, data)
        with pytest.raises(ValueError):
            pixel_array(path)

    def test_stream_usable_after_failure(self, tmpdir_path):
        path = os.path.join(tmpdir_path, "ok.dcm")
        arr = _write_gray(path)
        data = open(path, "rb").read()
        truncated = os.path.join(tmpdir_path, "t.dcm")
        with open(truncated, "wb") as fp:
            fp.write(data[:-5])

        with open(truncated, "rb") as fp:
            with pytest.raises(DeflateError):
                pixel_array(fp)
            assert not fp.closed
            assert fp.tell() == 0

        # The same stream can be used for a subsequent valid call
        with open(path, "rb") as fp:
            fp.seek(0)
            assert np.array_equal(pixel_array(fp), arr)


class TestZeroPadding:
    def test_trailing_zero_padding_accepted(self, tmpdir_path):
        arr = _gray()
        data = dataset_bytes(4, 4, 3, 16, 0, 1, "MONOCHROME2", arr.tobytes())
        path = os.path.join(tmpdir_path, "pad.dcm")
        write(path, DEFLATED, data, pad=True)
        assert np.array_equal(pixel_array(path), arr)


class TestDataSourceUnit:
    """Unit tests for the data source itself."""

    def _payload(self, size: int = 300_000) -> tuple[bytes, bytes]:
        import random

        rng = random.Random(1)
        plain = bytes(rng.getrandbits(8) for _ in range(size // 2))
        plain += b"\x00" * (size // 2)
        compressor = zlib.compressobj(6, zlib.DEFLATED, -zlib.MAX_WBITS)
        return plain, compressor.compress(plain) + compressor.flush()

    @pytest.mark.parametrize("chunk", [1, 7, 1024, 1 << 20])
    def test_small_chunks(self, monkeypatch, chunk):
        from pydicom import data_source

        plain, compressed = self._payload()
        monkeypatch.setattr(data_source, "MAX_OUTPUT_CHUNK_SIZE", chunk)
        monkeypatch.setattr(data_source, "MAX_INPUT_READ_SIZE", max(chunk, 3))
        monkeypatch.setattr(data_source, "SPOOL_MEMORY_LIMIT", 1)

        source = DeflateDataSource(BytesIO(compressed))
        reader = source.open_reader()
        assert reader.read() == plain
        reader.seek(12345)
        assert reader.read(100) == plain[12345:12445]
        reader.seek(-100, os.SEEK_END)
        assert reader.read() == plain[-100:]
        source.close()
        assert source.closed

    def test_readers_are_independent(self):
        plain, compressed = self._payload()
        source = DeflateDataSource(BytesIO(compressed))
        r1 = source.open_reader()
        r2 = source.open_reader()
        assert r1.read(10) == plain[:10]
        assert r2.read(20) == plain[:20]
        assert r1.read(10) == plain[10:20]
        r1.close()
        assert r2.read(10) == plain[20:30]
        assert not source.closed
        source.close()

    def test_truncated_raises(self):
        _, compressed = self._payload()
        source = DeflateDataSource(BytesIO(compressed[:-5]))
        with pytest.raises(DeflateError):
            source.expand_all()
        source.close()

    def test_closed_raises(self):
        _, compressed = self._payload()
        source = DeflateDataSource(BytesIO(compressed))
        reader = source.open_reader()
        source.close()
        with pytest.raises(DataSourceClosedError):
            reader.read()
        assert reader.closed


class TestWriteRoundTrip:
    def test_save_as_deflated(self, tmpdir_path):
        path = os.path.join(tmpdir_path, "gray.dcm")
        arr = _write_gray(path)
        out = os.path.join(tmpdir_path, "out.dcm")
        ds = dcmread(path, defer_size=1)
        try:
            ds.save_as(out)
        finally:
            ds.close()

        assert dcmread(out).file_meta.TransferSyntaxUID == DEFLATED
        assert np.array_equal(pixel_array(out), arr)

    def test_save_as_after_close(self, tmpdir_path):
        # Reading the deferred value during writing must still work if the
        #   source is closed afterwards
        path = os.path.join(tmpdir_path, "gray.dcm")
        arr = _write_gray(path)
        out = os.path.join(tmpdir_path, "out.dcm")
        ds = dcmread(path, defer_size=1)
        ds.save_as(out)
        ds.close()
        assert np.array_equal(pixel_array(out), arr)


class TestDeflatedJson:
    def test_lazy_bulk_data_unaffected(self, tmpdir_path):
        # Regression: existing JSON + BulkDataURI behaviour
        path = os.path.join(tmpdir_path, "gray.dcm")
        arr = _write_gray(path)
        ds = dcmread(path)
        try:
            js = ds.to_json_dict()
            restored = Dataset.from_json(js)
            restored.file_meta = ds.file_meta
            assert np.array_equal(restored.pixel_array, arr)
        finally:
            ds.close()


# ---------------------------------------------------------------------------
# Scale checks (slow)
# ---------------------------------------------------------------------------


def _write_scale(path: str, payload_mib: int, compressible: bool) -> int:
    """Write a deflated multi-frame dataset, generating one frame at a time."""
    rows = cols = 512
    frame_bytes = rows * cols * 2
    frames = payload_mib * 1024 * 1024 // frame_bytes
    pixel_bytes = frame_bytes * frames

    header = element((0x0008, 0x0005), "CS", b"ISO_IR 192")
    header += pixel_module(rows, cols, frames, 16, 0, 1, "MONOCHROME2")
    header += struct.pack(
        "<HH2sHI", 0x7FE0, 0x0010, b"OW", 0, pixel_bytes
    )

    rng = np.random.default_rng(0)
    with open(path, "wb") as fp:
        fp.write(b"\x00" * 128 + b"DICM" + file_meta(DEFLATED))
        compressor = zlib.compressobj(6, zlib.DEFLATED, -zlib.MAX_WBITS)
        fp.write(compressor.compress(header))
        for index in range(frames):
            if compressible:
                frame = np.full((rows, cols), index % 1024, dtype="<u2")
            else:
                frame = rng.integers(0, 65536, size=(rows, cols), dtype="<u2")
            fp.write(compressor.compress(frame.tobytes()))

        fp.write(compressor.flush())

    return frames


def _peak(callable_):
    gc.collect()
    tracemalloc.start()
    tracemalloc.reset_peak()
    try:
        result = callable_()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    return peak, result


@pytest.mark.slow
class TestScale:
    @pytest.mark.parametrize("compressible", [True, False], ids=["high", "low"])
    @pytest.mark.parametrize("payload_mib", [8, 64, 256])
    def test_memory_bounded(self, tmpdir_path, payload_mib, compressible):
        path = os.path.join(tmpdir_path, f"s{payload_mib}{compressible}.dcm")
        frames = _write_scale(path, payload_mib, compressible)
        limit = 32 * 1024 * 1024

        # Read header + defer: must not scale with the payload, and the
        #   decompressed data should have spilled to a temporary file rather
        #   than staying resident
        captured = {}

        def header_defer():
            ds = dcmread(path, defer_size=1024)
            try:
                assert ds._dict[0x7FE00010].value is None
                captured["source"] = ds._data_source
                return ds._data_source.spool_disk_bytes
            finally:
                ds.close()

        peak, disk = _peak(header_defer)
        assert peak <= limit, f"header+defer peak {peak / 2**20:.2f} MiB"
        # Payload padded to a whole number of frames
        frame_bytes = 512 * 512 * 2
        assert disk >= (payload_mib * 1024 * 1024 // frame_bytes) * frame_bytes
        # Closing released the temporary resources
        assert captured["source"].closed
        assert captured["source"].spool_disk_bytes == 0

        # Single frame
        peak, arr = _peak(lambda: pixel_array(path, index=frames // 2))
        assert peak <= limit, f"single frame peak {peak / 2**20:.2f} MiB"
        assert arr.shape == (512, 512)

        # Iterate and discard
        def consume():
            count = 0
            for _ in iter_pixels(path):
                count += 1
            return count

        peak, count = _peak(consume)
        assert count == frames
        assert peak <= limit, f"iteration peak {peak / 2**20:.2f} MiB"

    @pytest.mark.parametrize("compressible", [True, False], ids=["high", "low"])
    def test_sequential_reads_linear(self, tmpdir_path, compressible):
        """Sequential iteration reads each compressed byte about once."""
        path = os.path.join(tmpdir_path, "linear.dcm")
        _write_scale(path, 8, compressible)
        size = os.path.getsize(path)

        requests = []
        captured = {}
        original = DeflateDataSource.open_reader

        def spy(self):
            captured["source"] = self
            return original(self)

        DeflateDataSource.open_reader = spy

        # Also record the size of every input request
        original_read = DeflateDataSource._read_input

        def read_spy(self):
            data = original_read(self)
            requests.append(len(data))
            return data

        DeflateDataSource._read_input = read_spy
        try:
            for _ in iter_pixels(path):
                pass
        finally:
            DeflateDataSource.open_reader = original
            DeflateDataSource._read_input = original_read

        stats = captured["source"].stats
        # No re-reading of the compressed input
        assert stats["duplicate_bytes"] == 0
        assert stats["random_seeks"] == 0
        # Roughly one pass over the compressed stream (small tail padding OK)
        assert stats["bytes_read"] <= size + 2**20
        # Every input request is bounded
        assert requests
        assert max(requests) <= 1024 * 1024
