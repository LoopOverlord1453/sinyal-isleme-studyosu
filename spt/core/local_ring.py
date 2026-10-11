"""Süreç içi halka tampon ve sabit (kopyalanmış) veri için SharedRing ile aynı okuma arayüzü.

- `LocalRing`: tek kanallı, süreç içi halka tampon (canlı filtre çıktıları). Kayıp bölgeler NaN'dır.
- `StaticRing`: anlık görüntü (dondurulmuş veri) için; görünümler, boşluk kaydı ve mutlak örnek indeksleri
  canlı tampondaki gibi çalışır. Böylece grafikler canlı akışla aynı kodla çizilir.
"""
from __future__ import annotations

import numpy as np

from spt.core.ring import RingQueries


class LocalRing:
    def __init__(self, capacity: int, dtype, origin: int = 0):
        self.capacity = int(capacity)
        self.data = np.full(self.capacity, np.nan, dtype=dtype)
        self.origin = origin
        self.write_count = origin

    def valid_from(self) -> int:
        return max(self.origin, self.write_count - self.capacity)

    def write(self, values: np.ndarray) -> None:
        n = len(values)
        if n == 0:
            return
        cap = self.capacity
        start = self.write_count
        if n > cap:
            values = values[n - cap:]
            start += n - cap
        pos = start % cap
        first = min(len(values), cap - pos)
        self.data[pos:pos + first] = values[:first]
        if first < len(values):
            self.data[:len(values) - first] = values[first:]
        self.write_count += n

    def skip(self, n: int) -> None:
        """`n` örneklik bölge için çıktı yok (kayıp / atlandı): NaN yazılır, grafikte çizgi kesilir."""
        if n <= 0:
            return
        fill = min(n, self.capacity)
        cap = self.capacity
        pos = (self.write_count + n - fill) % cap
        first = min(fill, cap - pos)
        self.data[pos:pos + first] = np.nan
        if first < fill:
            self.data[:fill - first] = np.nan
        self.write_count += n

    def segments(self, start: int, end: int) -> list[tuple[int, np.ndarray]]:
        end = min(end, self.write_count)
        start = max(start, self.valid_from())
        if start >= end:
            return []
        cap = self.capacity
        a, b = start % cap, end % cap or cap
        if start // cap == (end - 1) // cap:
            return [(start, self.data[a:b])]
        return [(start, self.data[a:]), (start + cap - a, self.data[:b])]

    def read(self, start: int, end: int) -> tuple[int, np.ndarray]:
        segs = self.segments(start, end)
        if not segs:
            return end, self.data[:0].copy()
        out = np.concatenate([s for _, s in segs]) if len(segs) > 1 else segs[0][1].copy()
        return segs[0][0], out


class StaticRing(RingQueries):
    """Kopyalanmış (n, kanal) veriyi, ilk örneği `base` mutlak indeksinde olan bir tampon gibi sunar."""

    def __init__(self, data: np.ndarray, base: int, gaps: list[tuple[int, int]] | None = None):
        self.data = data
        self.base = int(base)
        self.capacity = len(data)
        self.channels = data.shape[1] if data.ndim > 1 else 1
        self.dtype = data.dtype
        self._gaps = sorted((int(s), int(e)) for s, e in (gaps or []) if e > s)

    @property
    def write_count(self) -> int:
        return self.base + len(self.data)

    @property
    def nbytes(self) -> int:
        return self.data.nbytes

    def valid_from(self) -> int:
        return self.base

    def segments(self, start: int, end: int) -> list[tuple[int, np.ndarray]]:
        start, end = max(start, self.base), min(end, self.write_count)
        if start >= end:
            return []
        return [(start, self.data[start - self.base:end - self.base])]

    def read(self, start: int, end: int) -> tuple[int, np.ndarray]:
        segs = self.segments(start, end)
        if not segs:
            return end, self.data[:0].copy()
        return segs[0][0], segs[0][1].copy()

    def gaps_between(self, start: int, end: int) -> list[tuple[int, int]]:
        return [(s, e) for s, e in self._gaps if s < end and e > start]

    def clean_ranges(self) -> list[tuple[int, int]]:
        """Kayıp içermeyen bitişik bölgeler (mutlak indeks)."""
        out, pos = [], self.base
        for s, e in self._gaps:
            s, e = max(s, self.base), min(e, self.write_count)
            if s > pos:
                out.append((pos, s))
            pos = max(pos, e)
        if pos < self.write_count:
            out.append((pos, self.write_count))
        return out

    @property
    def gaps(self) -> list[tuple[int, int]]:
        return list(self._gaps)

    def close(self) -> None:
        pass
