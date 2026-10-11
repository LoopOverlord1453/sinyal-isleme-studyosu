"""Arayüz tarafında bir akışın okuma arayüzü: tanımlayıcı + tampon (canlı paylaşımlı tampon ya da anlık görüntü)."""
from __future__ import annotations

import numpy as np

from spt.dsp.convert import components
from spt.protocol.format import Descriptor


class RingHandle:
    key: str
    source: str
    desc: Descriptor
    t0: float  # mutlak 0. örneğin zamanı (s)
    ring: object

    @property
    def fs(self) -> float:
        return self.desc.sample_rate

    def latest_index(self) -> int:
        return self.ring.write_count

    def time_of(self, index: float) -> float:
        return self.t0 + index / self.fs

    def column_segments(self, channel: int, start: int, end: int, part: int = 0) -> list[tuple[int, np.ndarray]]:
        """Kanalın [start, end) aralığı için sayısal bileşen görünümleri (part: 0=I/gerçel, 1=Q)."""
        st = self.desc.sample_type
        return [(s, components(view[:, channel], st)[part]) for s, view in self.ring.segments(start, end)]

    def close(self) -> None:
        self.ring.close()
