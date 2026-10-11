import numpy as np
import pytest

from spt.dsp.convert import components
from spt.protocol.builder import build_data, build_descriptor, build_marker, to_wire
from spt.protocol.format import SAMPLE_TYPES, ChannelInfo, Descriptor, Layout
from spt.protocol.parser import (
    DataFrame,
    DescriptorFrame,
    FrameError,
    MarkerFrame,
    decode_frames,
    samples_view,
)

# docs/SPTF_Protokol_v1.md §12 test vektörü
DOC_VECTOR = bytes.fromhex(
    "53505446010201003C00000001000100"
    "01000000000000004042 0F0000000000".replace(" ", "")
    + "00000000000000000200000000000000"
    "6400FFFFC800FEFFA8401E68"
)


def _desc(type_name: str, channels: int = 2, layout: Layout = Layout.INTERLEAVED) -> Descriptor:
    stype = next(t for t in SAMPLE_TYPES.values() if t.name == type_name)
    infos = tuple(ChannelInfo(f"ch{i}", "V", 0.5, 1.0) for i in range(channels))
    return Descriptor(1, 1, 1000.0, stype, layout, "test", infos)


def test_doc_vector_matches_builder():
    desc = _desc("int16")
    frame = build_data(desc, seq=1, timestamp_ns=1_000_000, first_sample=0,
                       samples=np.array([[100, -1], [200, -2]], dtype="<i2"), crc=True)
    assert frame == DOC_VECTOR


def test_doc_vector_decodes():
    (frame,) = decode_frames(DOC_VECTOR)
    assert isinstance(frame, DataFrame)
    assert frame.first_sample == 0 and frame.sample_count == 2
    values = np.frombuffer(frame.payload, dtype="<i2").reshape(2, 2)
    assert values.tolist() == [[100, -1], [200, -2]]


def _reference_values(type_name: str, n: int, channels: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    stype = next(t for t in SAMPLE_TYPES.values() if t.name == type_name)
    if stype.is_complex:
        lim = 1000
        return rng.integers(-lim, lim, (n, channels)) + 1j * rng.integers(-lim, lim, (n, channels))
    if stype.is_int24:
        lo, hi = (-(1 << 23), (1 << 23) - 1) if stype.signed else (0, (1 << 24) - 1)
        return rng.integers(lo, hi, (n, channels))
    if stype.is_float:
        return rng.standard_normal((n, channels)).astype(stype.dtype)
    info = np.iinfo(stype.dtype)
    return rng.integers(info.min, info.max, (n, channels), dtype=stype.dtype, endpoint=True)


@pytest.mark.parametrize("type_name", [t.name for t in SAMPLE_TYPES.values()])
@pytest.mark.parametrize("layout", [Layout.INTERLEAVED, Layout.PLANAR])
def test_roundtrip_all_types(type_name, layout):
    desc = _desc(type_name, channels=3, layout=layout)
    values = _reference_values(type_name, 17, 3)
    wire = to_wire(values, desc.sample_type)
    datagram = build_descriptor(desc) + build_data(desc, 5, 0, 1234, wire, crc=True)
    frames = decode_frames(datagram)
    assert isinstance(frames[0], DescriptorFrame) and frames[0].descriptor == desc
    data = frames[1]
    assert data.first_sample == 1234 and data.sample_count == 17
    decoded = samples_view(data.payload, 0, 17, desc)
    for ch in range(3):
        parts = components(decoded[:, ch], desc.sample_type)
        got = parts[0] + 1j * parts[1] if len(parts) == 2 else parts[0]
        np.testing.assert_array_equal(got, values[:, ch])


def test_marker_roundtrip():
    (frame,) = decode_frames(build_marker(3, 2, 9, 77, 123456, 42, "ISR overrun ğüş", crc=True))
    assert isinstance(frame, MarkerFrame)
    assert (frame.sample_index, frame.code, frame.text) == (123456, 42, "ISR overrun ğüş")
    assert frame.header.stream_id == 3 and frame.header.desc_rev == 2


@pytest.mark.parametrize("mutate, message", [
    (lambda b: b[:20], "başlığı"),
    (lambda b: b"XXXX" + b[4:], "magic"),
    (lambda b: b[:4] + b"\x02" + b[5:], "sürüm"),
    (lambda b: b[:-1] + bytes([b[-1] ^ 1]), "CRC"),
    (lambda b: b[:8] + (500).to_bytes(4, "little") + b[12:], "frame_len"),
])
def test_corrupt_frames_rejected(mutate, message):
    with pytest.raises(FrameError, match=message):
        decode_frames(mutate(DOC_VECTOR))


def test_descriptor_size_must_match_channel_count():
    frame = bytearray(build_descriptor(_desc("int16", channels=2)))
    frame[32 + 8] = 3  # channel_count = 3, ama 2 kanal bilgisi var
    with pytest.raises(FrameError, match="DESCRIPTOR boyutu"):
        decode_frames(bytes(frame))
