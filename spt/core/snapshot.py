"""Anlık görüntüler (Dondur) ve çevrimdışı filtreleme.

Dondur, akışın paylaşımlı tampondaki verisini kopyalar; canlı akış devam ederken kopya değişmez. Anlık görüntü
mutlak örnek indekslerini ve zaman eksenini korur, böylece filtrelenmiş sonucu ham veriyle aynı eksende üst
üste görmek mümkündür. Kayıp bölgeler (boşluklar) filtreleme sırasında atlanır: her kayıpsız bölge ayrı
filtrelenir ve kayıp bölgeler sonuçta NaN (çizgi kesik) kalır.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime

import numpy as np

from spt.core.handle import RingHandle
from spt.core.local_ring import StaticRing
from spt.dsp.chain import ProcessingChain
from spt.dsp.convert import physical_column
from spt.dsp.filters import FilterError, apply_offline, finite_runs, format_hz
from spt.protocol.format import SAMPLE_TYPES_BY_NAME, ChannelInfo, Descriptor


@dataclass(eq=False)
class Snapshot:
    id: int
    name: str
    source: str
    desc: Descriptor
    t0: float
    ring: StaticRing
    created: datetime = field(default_factory=datetime.now)
    parent: int | None = None
    note: str = ""
    _handle: object = field(default=None, repr=False)

    @property
    def fs(self) -> float:
        return self.desc.sample_rate

    @property
    def samples(self) -> int:
        return self.ring.capacity

    @property
    def duration(self) -> float:
        return self.samples / self.fs

    @property
    def start_time(self) -> float:
        return self.t0 + self.ring.base / self.fs

    def details(self) -> str:
        lost = sum(e - s for s, e in self.ring.gaps)
        size = self.ring.nbytes
        size_text = f"{size / 1e6:.3g} MB" if size >= 1e6 else f"{size / 1e3:.3g} kB"
        text = f"{self.desc.channel_count} kanal · {self.duration:.3g} s · {format_hz(self.fs)} · {size_text}"
        return text + (f" · {lost} örnek kayıp" if lost else "")

    def handle(self) -> SnapshotHandle:
        """Tek tutamaç (görünümler aynı nesneyi paylaşır; kimliği değişmez)."""
        if self._handle is None:
            self._handle = SnapshotHandle(self)
        return self._handle


class SnapshotHandle(RingHandle):
    """Anlık görüntüyü canlı akış gibi okunabilir kılar (grafik kodu aynı)."""

    def __init__(self, snap: Snapshot):
        self.snapshot = snap
        self.key = f"ws{snap.id}"
        self.source = snap.source
        self.desc = snap.desc
        self.t0 = snap.t0
        self.ring = snap.ring


def take_snapshot(handle: RingHandle, end: int, snap_id: int, name: str) -> Snapshot | None:
    """Akışın tamponunda `end`'e kadar duran tüm veriyi kopyalar. Veri yoksa None."""
    ring = handle.ring
    end = min(end, ring.write_count)
    first, data = ring.read(ring.valid_from(), end)
    if len(data) == 0:
        return None
    last = first + len(data)
    gaps = [(max(s, first), min(e, last)) for s, e in ring.gaps_between(first, last)]
    return Snapshot(snap_id, name, handle.source, handle.desc, handle.t0, StaticRing(data, first, gaps))


def filter_snapshot(snap: Snapshot, chain: ProcessingChain, zero_phase: bool, snap_id: int,
                    name: str | None = None) -> Snapshot:
    """Zinciri anlık görüntünün tüm kanallarına uygular; sonuç fiziksel birimde yeni bir anlık görüntü.

    Hesap float64'tür; sonuç, girdi ≤ 24 bit hassasiyetteyse (int8/16/24, float16/32, cint16, cfloat32) float32'de
    saklanır (24 bit mantis girdinin çözünürlüğünü korur, bellek yarıya iner), aksi halde float64."""
    designs, errors = chain.designs(snap.fs)
    if errors:
        raise FilterError("\n".join(errors))
    if not designs:
        raise FilterError("İşlem zincirinde etkin aşama yok")
    desc = snap.desc
    st = desc.sample_type
    complex_ = st.is_complex
    single = st.name in ("float16", "float32", "cfloat32") or (not st.is_float and st.bits <= 24)
    dtype = (np.complex64 if single else np.complex128) if complex_ else (np.float32 if single else np.float64)
    out = np.full((snap.samples, desc.channel_count), np.nan, dtype=dtype)
    ring = snap.ring
    for ch, info in enumerate(desc.channels):
        for start, end in ring.clean_ranges():
            x = physical_column(ring.data[start - ring.base:end - ring.base, ch], desc.sample_type, info)
            for a, b, finite in finite_runs(x):  # NaN örnekler (ör. filtrelenmiş anlık görüntüde) kayıp gibi
                if finite:
                    out[start - ring.base + a:start - ring.base + b, ch] = apply_offline(designs, x[a:b], zero_phase)
    channels = tuple(ChannelInfo(f"{c.name or f'ch{i}'} · filtreli", c.unit, 1.0, 0.0)
                     for i, c in enumerate(desc.channels))
    stype = SAMPLE_TYPES_BY_NAME[("cfloat32" if single else "cfloat64") if complex_ else
                                 ("float32" if single else "float64")]
    new_desc = replace(desc, sample_type=stype, channels=channels, name=f"{desc.name or snap.name} (filtreli)")
    mode = "sıfır fazlı" if zero_phase else "nedensel"
    note = f"{chain.describe()} · {mode}"
    return Snapshot(snap_id, name or f"{snap.name} · filtreli ({mode})", snap.source, new_desc, snap.t0,
                    StaticRing(out, ring.base, ring.gaps), parent=snap.id, note=note)
