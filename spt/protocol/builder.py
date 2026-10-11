"""SPTF çerçevesi oluşturma: simülatör, örnek kayıt üreteci ve testler kullanır."""
from __future__ import annotations

import zlib

import numpy as np

from spt.protocol.format import (
    CHANNEL,
    CRC,
    DATA,
    DESCRIPTOR,
    FLAG_CRC32,
    HEADER,
    MAGIC,
    MARKER,
    VERSION,
    Descriptor,
    FrameType,
    Layout,
    SampleType,
)


def _pad(text: str, size: int) -> bytes:
    raw = text.encode("utf-8")[:size]
    return raw + b"\0" * (size - len(raw))


def build_frame(frame_type: FrameType, stream_id: int, desc_rev: int, seq: int, timestamp_ns: int,
                payload: bytes | memoryview, crc: bool = False) -> bytes:
    flags = FLAG_CRC32 if crc else 0
    frame_len = HEADER.size + len(payload) + (CRC.size if crc else 0)
    head = HEADER.pack(MAGIC, VERSION, frame_type, flags, frame_len, stream_id, desc_rev,
                       seq & 0xFFFFFFFF, 0, timestamp_ns)
    body = head + bytes(payload)
    if crc:
        body += CRC.pack(zlib.crc32(body))
    return body


def build_descriptor(desc: Descriptor, seq: int = 0, timestamp_ns: int = 0, crc: bool = False) -> bytes:
    payload = bytearray(DESCRIPTOR.pack(desc.sample_rate, desc.channel_count, desc.sample_type.code,
                                        desc.layout, 0, _pad(desc.name, 32)))
    for ch in desc.channels:
        payload += CHANNEL.pack(_pad(ch.name, 32), _pad(ch.unit, 16), ch.scale, ch.offset)
    return build_frame(FrameType.DESCRIPTOR, desc.stream_id, desc.desc_rev, seq, timestamp_ns, payload, crc)


def to_wire(values: np.ndarray, sample_type: SampleType) -> np.ndarray:
    """Ham değerleri (tamsayı veya karmaşık) kablodaki numpy tipine çevirir.

    - int24/uint24: tamsayı dizisi → 3 baytlık paketli değerler
    - cint16/cint32: karmaşık dizi → (re, im) yapısı
    - diğerleri: doğrudan tip dönüşümü (tamsayı tiplerde yuvarlama ve kırpma)
    """
    values = np.asarray(values)
    if sample_type.is_int24:
        ints = np.asarray(values, dtype=np.int64)
        lo, hi = (-(1 << 23), (1 << 23) - 1) if sample_type.signed else (0, (1 << 24) - 1)
        ints = np.clip(ints, lo, hi).astype("<i4")
        packed = ints.view(np.uint8).reshape(*ints.shape, 4)[..., :3]
        return np.ascontiguousarray(packed).view("V3").reshape(ints.shape)
    if sample_type.is_complex and sample_type.dtype.names:
        base = sample_type.dtype["re"]
        info = np.iinfo(base)
        out = np.empty(values.shape, dtype=sample_type.dtype)
        out["re"] = np.clip(np.rint(values.real), info.min, info.max)
        out["im"] = np.clip(np.rint(values.imag), info.min, info.max)
        return out
    if np.issubdtype(sample_type.dtype, np.integer) and not np.issubdtype(values.dtype, np.integer):
        info = np.iinfo(sample_type.dtype)
        # 64 bit tiplerde float64 sınırı tam temsil edemez (2^63 − 1 → 2^63 taşar): bir alt komşuya kırp
        hi = float(info.max) if info.bits < 64 else np.nextafter(float(info.max), -np.inf)
        values = np.clip(np.rint(values), float(info.min), hi)
    return values.astype(sample_type.dtype)


def encode_samples(desc: Descriptor, samples: np.ndarray) -> bytes:
    """(N, kanal) biçimindeki kablo tipli örnekleri descriptor'daki yerleşime göre baytlara çevirir."""
    if samples.dtype != desc.sample_type.dtype:
        samples = to_wire(samples, desc.sample_type)
    if desc.layout == Layout.PLANAR:
        samples = samples.T
    return np.ascontiguousarray(samples).tobytes()


def build_data(desc: Descriptor, seq: int, timestamp_ns: int, first_sample: int, samples: np.ndarray,
               crc: bool = False) -> bytes:
    payload = DATA.pack(first_sample, samples.shape[0], 0) + encode_samples(desc, samples)
    return build_frame(FrameType.DATA, desc.stream_id, desc.desc_rev, seq, timestamp_ns, payload, crc)


def build_marker(stream_id: int, desc_rev: int, seq: int, timestamp_ns: int, sample_index: int,
                 code: int, text: str, crc: bool = False) -> bytes:
    raw = text.encode("utf-8")[:0xFFFF]
    payload = MARKER.pack(sample_index, code, len(raw), 0) + raw
    return build_frame(FrameType.MARKER, stream_id, desc_rev, seq, timestamp_ns, payload, crc)


class DataFrameWriter:
    """Simülatörün sıcak yolu: önceden ayrılmış tampona DATA çerçevesi yazar (bellek ayırmadan)."""

    def __init__(self, desc: Descriptor, samples_per_frame: int, crc: bool = False):
        self.desc = desc
        self.crc = crc
        self.payload_size = samples_per_frame * desc.bytes_per_sample
        self.size = HEADER.size + DATA.size + self.payload_size + (CRC.size if crc else 0)
        self._buf = bytearray(self.size)
        self._mv = memoryview(self._buf)
        self._flags = FLAG_CRC32 if crc else 0

    def write(self, seq: int, timestamp_ns: int, first_sample: int, count: int, sample_bytes) -> memoryview:
        nbytes = count * self.desc.bytes_per_sample
        frame_len = HEADER.size + DATA.size + nbytes + (CRC.size if self.crc else 0)
        HEADER.pack_into(self._buf, 0, MAGIC, VERSION, FrameType.DATA, self._flags, frame_len,
                         self.desc.stream_id, self.desc.desc_rev, seq & 0xFFFFFFFF, 0, timestamp_ns)
        DATA.pack_into(self._buf, HEADER.size, first_sample, count, 0)
        start = HEADER.size + DATA.size
        self._mv[start:start + nbytes] = sample_bytes
        if self.crc:
            end = start + nbytes
            CRC.pack_into(self._buf, end, zlib.crc32(self._mv[:end]))
        return self._mv[:frame_len]
