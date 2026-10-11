"""SPTF çerçevelerini doğrulama ve çözme.

Alıcının sıcak yolu (saniyede on binlerce çerçeve) `check_frame` ve `unpack_*` gibi düşük seviyeli
fonksiyonları kullanır; testler ve araçlar için `decode_frames` okunur nesneler döndürür.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass

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
    MAX_CHANNELS,
    MAX_SAMPLE_RATE,
    MIN_SAMPLE_RATE,
    SAMPLE_TYPES,
    VERSION,
    ChannelInfo,
    Descriptor,
    FrameType,
    Layout,
)


class FrameError(ValueError):
    """Çerçeve biçimsel olarak hatalı."""


def _text(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("utf-8", "replace")


def check_frame(buf, off: int, end: int) -> tuple[int, int, int, int, int, int, int]:
    """`buf[off:end]` başındaki çerçeveyi doğrular.

    Dönüş: (frame_type, flags, frame_len, stream_id, desc_rev, seq, timestamp_ns)
    """
    if end - off < HEADER.size:
        raise FrameError(f"çerçeve başlığı için {end - off} bayt var, 32 gerekli")
    magic, version, ftype, flags, flen, sid, rev, seq, _res, ts = HEADER.unpack_from(buf, off)
    if magic != MAGIC:
        raise FrameError(f"magic hatalı: {bytes(magic)!r}")
    if version != VERSION:
        raise FrameError(f"desteklenmeyen sürüm: {version}")
    crc_size = CRC.size if flags & FLAG_CRC32 else 0
    if flen < HEADER.size + crc_size or off + flen > end:
        raise FrameError(f"frame_len geçersiz: {flen} (kalan {end - off} bayt)")
    if crc_size:
        (expected,) = CRC.unpack_from(buf, off + flen - CRC.size)
        actual = zlib.crc32(buf[off:off + flen - CRC.size])
        if actual != expected:
            raise FrameError(f"CRC hatası: hesaplanan 0x{actual:08X}, çerçevede 0x{expected:08X}")
    return ftype, flags, flen, sid, rev, seq, ts


def payload_bounds(off: int, flags: int, flen: int) -> tuple[int, int]:
    return off + HEADER.size, off + flen - (CRC.size if flags & FLAG_CRC32 else 0)


def unpack_descriptor(buf, start: int, end: int, stream_id: int, desc_rev: int) -> Descriptor:
    if end - start < DESCRIPTOR.size:
        raise FrameError("DESCRIPTOR payload'ı çok kısa")
    rate, count, type_code, layout, _res, name = DESCRIPTOR.unpack_from(buf, start)
    if not (np.isfinite(rate) and MIN_SAMPLE_RATE <= rate <= MAX_SAMPLE_RATE):
        raise FrameError(f"geçersiz örnekleme hızı: {rate} (izin verilen {MIN_SAMPLE_RATE:g} … {MAX_SAMPLE_RATE:g} Hz)")
    if not 1 <= count <= MAX_CHANNELS:
        raise FrameError(f"geçersiz kanal sayısı: {count}")
    sample_type = SAMPLE_TYPES.get(type_code)
    if sample_type is None:
        raise FrameError(f"bilinmeyen sample_type: 0x{type_code:02X}")
    if layout not in (Layout.INTERLEAVED, Layout.PLANAR):
        raise FrameError(f"geçersiz layout: {layout}")
    expected = DESCRIPTOR.size + count * CHANNEL.size
    if end - start != expected:
        raise FrameError(f"DESCRIPTOR boyutu {end - start} bayt, {count} kanal için {expected} olmalı")
    channels = []
    pos = start + DESCRIPTOR.size
    for _ in range(count):
        ch_name, unit, scale, offset = CHANNEL.unpack_from(buf, pos)
        channels.append(ChannelInfo(_text(ch_name), _text(unit), scale, offset))
        pos += CHANNEL.size
    return Descriptor(stream_id, desc_rev, rate, sample_type, Layout(layout), _text(name), tuple(channels))


def unpack_data_header(buf, start: int, end: int) -> tuple[int, int, int]:
    """Dönüş: (first_sample, sample_count, örnek baytları)"""
    if end - start < DATA.size:
        raise FrameError("DATA payload'ı çok kısa")
    first, count, _res = DATA.unpack_from(buf, start)
    return first, count, end - start - DATA.size


def unpack_marker(buf, start: int, end: int) -> tuple[int, int, str]:
    if end - start < MARKER.size:
        raise FrameError("MARKER payload'ı çok kısa")
    sample_index, code, text_len, _res = MARKER.unpack_from(buf, start)
    if MARKER.size + text_len > end - start:
        raise FrameError("MARKER metni payload'dan uzun")
    text = bytes(buf[start + MARKER.size:start + MARKER.size + text_len]).decode("utf-8", "replace")
    return sample_index, code, text


def samples_view(buf, start: int, count: int, desc: Descriptor) -> np.ndarray:
    """Örnek baytlarını kopyalamadan (count, kanal) biçiminde numpy görünümü olarak döndürür."""
    ch = desc.channel_count
    arr = np.frombuffer(buf, dtype=desc.sample_type.dtype, count=count * ch, offset=start)
    if desc.layout == Layout.PLANAR:
        return arr.reshape(ch, count).T
    return arr.reshape(count, ch)


# ---------------------------------------------------------------- okunur çözümleme (test / araç)
@dataclass(frozen=True)
class Header:
    frame_type: FrameType
    flags: int
    frame_len: int
    stream_id: int
    desc_rev: int
    seq: int
    timestamp_ns: int


@dataclass(frozen=True)
class DataFrame:
    header: Header
    first_sample: int
    sample_count: int
    payload: bytes   # ham örnek baytları; çözmek için descriptor gerekir


@dataclass(frozen=True)
class MarkerFrame:
    header: Header
    sample_index: int
    code: int
    text: str


@dataclass(frozen=True)
class DescriptorFrame:
    header: Header
    descriptor: Descriptor


def decode_frames(buf) -> list[DescriptorFrame | DataFrame | MarkerFrame]:
    """Bir datagram (veya bayt dizisi) içindeki tüm çerçeveleri çözer."""
    mv = memoryview(buf)
    frames = []
    off, end = 0, len(mv)
    while off < end:
        ftype, flags, flen, sid, rev, seq, ts = check_frame(mv, off, end)
        try:
            kind = FrameType(ftype)
        except ValueError:
            raise FrameError(f"bilinmeyen frame_type: {ftype}") from None
        header = Header(kind, flags, flen, sid, rev, seq, ts)
        start, stop = payload_bounds(off, flags, flen)
        if kind == FrameType.DESCRIPTOR:
            frames.append(DescriptorFrame(header, unpack_descriptor(mv, start, stop, sid, rev)))
        elif kind == FrameType.DATA:
            first, count, nbytes = unpack_data_header(mv, start, stop)
            frames.append(DataFrame(header, first, count, bytes(mv[start + DATA.size:stop])))
        else:
            frames.append(MarkerFrame(header, *unpack_marker(mv, start, stop)))
        off += flen
    return frames
