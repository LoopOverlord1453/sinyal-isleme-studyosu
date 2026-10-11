"""SPTF v1 sabitleri, yapı tanımları ve veri tipi tablosu. Bkz. docs/SPTF_Protokol_v1.md."""
from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

MAGIC = b"SPTF"
VERSION = 1
UDP_PORT = 50505
STANDARD_DATAGRAM = 1472   # MTU 1500 − IP/UDP başlıkları
JUMBO_DATAGRAM = 8972      # MTU 9000 (jumbo frame)
MAX_DATAGRAM = 65507       # UDP üst sınırı (IP parçalanmasıyla)
MAX_CHANNELS = 256
FLAG_CRC32 = 0x0001
MIN_SAMPLE_RATE = 1e-3      # Hz — tanımlayıcıda bunun dışındaki hız bozuk çerçeve sayılır
MAX_SAMPLE_RATE = 1e12
MAX_FIRST_SAMPLE = 1 << 62  # DATA.first_sample üst sınırı (64 bit sayaçta taşma payı)
MAX_SAMPLE_JUMP = 1 << 40   # tek çerçevede kabul edilen en büyük ileri sıçrama (örnek)

HEADER = struct.Struct("<4sBBHIHHIIQ")   # 32 B, tüm çerçevelerde ortak
DESCRIPTOR = struct.Struct("<dHBBI32s")  # 48 B
CHANNEL = struct.Struct("<32s16sdd")     # 64 B
DATA = struct.Struct("<QII")             # 16 B
MARKER = struct.Struct("<QIHH")          # 16 B
CRC = struct.Struct("<I")

DATA_OVERHEAD = HEADER.size + DATA.size  # örnekler dışındaki DATA baytları (CRC hariç)


class FrameType(IntEnum):
    DESCRIPTOR = 1
    DATA = 2
    MARKER = 3


class Layout(IntEnum):
    INTERLEAVED = 0
    PLANAR = 1


@dataclass(frozen=True)
class SampleType:
    code: int
    name: str
    size: int              # örnek başına bayt
    dtype: np.dtype        # kablodaki ve tampondaki numpy tipi
    signed: bool = True
    is_complex: bool = False
    is_int24: bool = False
    is_float: bool = False

    @property
    def bits(self) -> int:
        return self.size * 8 // (2 if self.is_complex else 1)


def _complex_int(base: str) -> np.dtype:
    return np.dtype([("re", base), ("im", base)])


SAMPLE_TYPES: dict[int, SampleType] = {t.code: t for t in (
    SampleType(0x01, "int8", 1, np.dtype("<i1")),
    SampleType(0x02, "uint8", 1, np.dtype("<u1"), signed=False),
    SampleType(0x03, "int16", 2, np.dtype("<i2")),
    SampleType(0x04, "uint16", 2, np.dtype("<u2"), signed=False),
    SampleType(0x05, "int24", 3, np.dtype("V3"), is_int24=True),
    SampleType(0x06, "uint24", 3, np.dtype("V3"), signed=False, is_int24=True),
    SampleType(0x07, "int32", 4, np.dtype("<i4")),
    SampleType(0x08, "uint32", 4, np.dtype("<u4"), signed=False),
    SampleType(0x09, "int64", 8, np.dtype("<i8")),
    SampleType(0x0A, "uint64", 8, np.dtype("<u8"), signed=False),
    SampleType(0x10, "float16", 2, np.dtype("<f2"), is_float=True),
    SampleType(0x11, "float32", 4, np.dtype("<f4"), is_float=True),
    SampleType(0x12, "float64", 8, np.dtype("<f8"), is_float=True),
    SampleType(0x20, "cint16", 4, _complex_int("<i2"), is_complex=True),
    SampleType(0x21, "cint32", 8, _complex_int("<i4"), is_complex=True),
    SampleType(0x22, "cfloat32", 8, np.dtype("<c8"), is_complex=True, is_float=True),
    SampleType(0x23, "cfloat64", 16, np.dtype("<c16"), is_complex=True, is_float=True),
)}
SAMPLE_TYPES_BY_NAME: dict[str, SampleType] = {t.name: t for t in SAMPLE_TYPES.values()}


@dataclass(frozen=True)
class ChannelInfo:
    name: str
    unit: str = ""
    scale: float = 1.0
    offset: float = 0.0


@dataclass(frozen=True)
class Descriptor:
    stream_id: int
    desc_rev: int
    sample_rate: float
    sample_type: SampleType
    layout: Layout
    name: str
    channels: tuple[ChannelInfo, ...]

    @property
    def channel_count(self) -> int:
        return len(self.channels)

    @property
    def bytes_per_sample(self) -> int:
        """Bir örnek anındaki tüm kanalların toplam bayt sayısı."""
        return self.sample_type.size * len(self.channels)

    def format_text(self) -> str:
        return f"{self.channel_count} × {self.sample_type.name} · {format_rate(self.sample_rate)}"


def format_rate(hz: float) -> str:
    """Örnekleme hızını okunur biçimde yazar: 500 Hz, 10 kHz, 1 MS/s."""
    if hz >= 1e6:
        return f"{hz / 1e6:g} MS/s"
    if hz >= 1e3:
        return f"{hz / 1e3:g} kHz"
    return f"{hz:g} Hz"


def samples_per_frame(desc: Descriptor, datagram: int = STANDARD_DATAGRAM, crc: bool = False) -> int:
    """Verilen datagram boyutuna sığan, kanal başına örnek sayısı."""
    room = datagram - DATA_OVERHEAD - (CRC.size if crc else 0)
    return max(1, room // desc.bytes_per_sample)
