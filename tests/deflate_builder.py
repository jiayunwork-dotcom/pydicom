# Copyright 2008-2024 pydicom authors. See LICENSE file for details.
"""Independently constructed datasets for Deflated Explicit VR Little Endian tests.

The builders here encode datasets using only :mod:`struct` and
:mod:`zlib`; they deliberately do not use pydicom's writer so that reading is
verified against data constructed independently of the code under test.
"""

import struct
import zlib

DEFLATED = "1.2.840.10008.1.2.1.99"
EXPLICIT = "1.2.840.10008.1.2.1"

_LONG_VRS = {
    "OB",
    "OW",
    "OF",
    "OD",
    "OL",
    "OV",
    "SQ",
    "UN",
    "UT",
    "UC",
    "UR",
    "SV",
    "UV",
}


def element(tag: tuple[int, int], vr: str, value: bytes | int) -> bytes:
    """Return an Explicit VR Little Endian encoded element."""
    group, elem = tag
    if vr in _LONG_VRS:
        payload = b"" if isinstance(value, int) else value
        length = value if isinstance(value, int) else len(payload)
        return struct.pack("<HH2sHI", group, elem, vr.encode(), 0, length) + payload

    payload = value if isinstance(value, bytes) else struct.pack("<H", value)
    return struct.pack("<HH2sH", group, elem, vr.encode(), len(payload)) + payload


def sequence(
    tag: tuple[int, int], items: bytes, *, undefined_length: bool = False
) -> bytes:
    """Return an Explicit VR Little Endian encoded sequence element."""
    group, elem = tag
    length = 0xFFFFFFFF if undefined_length else len(items)
    out = struct.pack("<HH2sHI", group, elem, b"SQ", 0, length) + items
    if undefined_length:
        out += struct.pack("<HHI", 0xFFFE, 0xE0DD, 0)

    return out


def item(content: bytes, *, undefined_length: bool = False) -> bytes:
    """Return an encoded sequence item."""
    if undefined_length:
        return (
            struct.pack("<HHI", 0xFFFE, 0xE000, 0xFFFFFFFF)
            + content
            + struct.pack("<HHI", 0xFFFE, 0xE00D, 0)
        )

    return struct.pack("<HHI", 0xFFFE, 0xE000, len(content)) + content


def file_meta(tsyntax: str) -> bytes:
    """Return the *File Meta Information* elements for `tsyntax`."""
    meta = b""
    meta += element((0x0002, 0x0001), "OB", b"\x00\x01")
    meta += element((0x0002, 0x0002), "UI", b"1.2.840.10008.5.1.4.1.1.4\x00")
    meta += element((0x0002, 0x0003), "UI", b"1.2.3.4.5.6.7.8.9\x00")
    meta += element((0x0002, 0x0010), "UI", tsyntax.encode() + b"\x00")
    meta += element((0x0002, 0x0012), "UI", b"1.2.3.4\x00")
    meta += element((0x0002, 0x0013), "SH", b"TEST\x00")

    return element((0x0002, 0x0000), "UL", struct.pack("<I", len(meta))) + meta


def pixel_module(
    rows: int,
    cols: int,
    frames: int | None,
    bits_allocated: int,
    pixel_representation: int,
    samples_per_pixel: int,
    photometric_interpretation: str,
) -> bytes:
    """Return the *Image Pixel* module elements."""
    value = b""
    value += element((0x0028, 0x0002), "US", samples_per_pixel)
    value += element(
        (0x0028, 0x0004), "CS", photometric_interpretation.encode() + b"\x00"
    )
    if samples_per_pixel > 1:
        value += element((0x0028, 0x0006), "US", 0)
    if frames is not None:
        value += element((0x0028, 0x0008), "IS", str(frames).encode())
    value += element((0x0028, 0x0010), "US", rows)
    value += element((0x0028, 0x0011), "US", cols)
    value += element((0x0028, 0x0100), "US", bits_allocated)
    value += element((0x0028, 0x0101), "US", bits_allocated)
    value += element((0x0028, 0x0102), "US", bits_allocated - 1)
    value += element((0x0028, 0x0103), "US", pixel_representation)
    return value


def dataset_bytes(
    rows: int,
    cols: int,
    frames: int | None,
    bits_allocated: int,
    pixel_representation: int,
    samples_per_pixel: int,
    photometric_interpretation: str,
    pixel_data: bytes,
    extra: bytes = b"",
) -> bytes:
    """Return an Explicit VR Little Endian encoded dataset."""
    value = b""
    value += element((0x0008, 0x0005), "CS", b"ISO_IR 192")
    value += element((0x0008, 0x1030), "LO", b"Study description\x00")
    value += extra
    value += pixel_module(
        rows,
        cols,
        frames,
        bits_allocated,
        pixel_representation,
        samples_per_pixel,
        photometric_interpretation,
    )
    vr = "OB" if bits_allocated <= 8 else "OW"
    value += element((0x7FE0, 0x0010), vr, pixel_data)
    return value


def write(
    path: str, tsyntax: str, dataset: bytes, *, pad: bool = False, level: int = 6
) -> None:
    """Write a DICOM file from ``dataset`` (the non-File-Meta dataset bytes)."""
    with open(path, "wb") as fp:
        fp.write(b"\x00" * 128)
        fp.write(b"DICM")
        fp.write(file_meta(tsyntax))
        if tsyntax == DEFLATED:
            compressor = zlib.compressobj(level, zlib.DEFLATED, -zlib.MAX_WBITS)
            compressed = compressor.compress(dataset) + compressor.flush()
            if pad and len(compressed) % 2:
                compressed += b"\x00"

            fp.write(compressed)
        else:
            fp.write(dataset)
