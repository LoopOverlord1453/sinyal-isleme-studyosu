"""Süreçler arası paylaşımlı halka tampon (ring buffer).

Tek yazıcı (alıcı / oynatıcı süreci) örnekleri ham kablo tipinde yazar; arayüz süreci aynı belleği
kopyalamadan okur. Kilit kullanılmaz: yazıcı önce veriyi, sonra 64 bitlik `write_count` sayacını
günceller. Okuyucu, okuma sırasında üzerine yazılmış olabilecek en eski bölgeyi sayaçtan anlar.

Bellek düzeni:
    [0:64)            int64 başlık: write_count, capacity, channels, itemsize, gap_count
    [64:64+16·GAPS)   boşluk kaydı: (başlangıç, uzunluk) çiftleri, dairesel
    [DATA_OFF: ...)   örnekler: (capacity, channels) ham tip
"""
from __future__ import annotations

from multiprocessing import shared_memory

import numpy as np

GAP_SLOTS = 4096  # tamponda hatırlanan en çok kayıp aralığı; aşılırsa en eskiden öncesi geçersiz sayılır
_HEADER_BYTES = 64
_GAPS_OFF = _HEADER_BYTES
_DATA_OFF = _GAPS_OFF + GAP_SLOTS * 16
_WC, _CAP, _CH, _ITEM, _GAPN = range(5)


class RingQueries:
    """`write_count`, `valid_from` ve `gaps_between` üzerinden tanımlı ortak aralık sorguları."""

    def latest_clean_range(self, n: int, end: int | None = None) -> tuple[int, int] | None:
        """`end`'den (varsayılan: en son örnek) geriye doğru, içinde kayıp olmayan en yeni `n` örneklik aralık.

        Kayıp bölgeler sıfırla doldurulduğu için FFT'ye girerse genliği düşürür; bu yüzden spektrum,
        kayıp içermeyen bir pencereden hesaplanmalıdır. Uygun aralık yoksa None.
        """
        end = self.write_count if end is None else min(end, self.write_count)
        lowest = self.valid_from()
        while end - n >= lowest:
            gaps = self.gaps_between(end - n, end)
            if not gaps:
                return end - n, end
            end = max(gs for gs, _ in gaps)  # en son kaybın başından önceye kay
        return None


class SharedRing(RingQueries):
    def __init__(self, shm: shared_memory.SharedMemory, dtype: np.dtype, owner: bool):
        self._shm = shm
        self._owner = owner
        self.dtype = np.dtype(dtype)
        self._hdr = np.ndarray((8,), dtype=np.int64, buffer=shm.buf, offset=0)
        self.capacity = int(self._hdr[_CAP])
        self.channels = int(self._hdr[_CH])
        if int(self._hdr[_ITEM]) != self.dtype.itemsize:
            raise ValueError("tampon örnek boyutu tiple uyuşmuyor")
        self._gaps = np.ndarray((GAP_SLOTS, 2), dtype=np.int64, buffer=shm.buf, offset=_GAPS_OFF)
        self.data = np.ndarray((self.capacity, self.channels), dtype=self.dtype, buffer=shm.buf, offset=_DATA_OFF)
        self._row_bytes = self.channels * self.dtype.itemsize
        self._bytes = np.ndarray((self.capacity * self._row_bytes,), dtype=np.uint8, buffer=shm.buf,
                                 offset=_DATA_OFF)
        # Yazıcının sıcak yolu için numpy'siz erişim: ham bayt görünümü ve 64 bit sayaç görünümü.
        self._raw = shm.buf[_DATA_OFF:_DATA_OFF + self.capacity * self._row_bytes]
        self._counter = shm.buf[:8].cast("q")
        self._wc = int(self._hdr[_WC])

    # ---------------------------------------------------------------- oluşturma
    @classmethod
    def create(cls, capacity: int, channels: int, dtype: np.dtype) -> SharedRing:
        dtype = np.dtype(dtype)
        size = _DATA_OFF + capacity * channels * dtype.itemsize
        shm = shared_memory.SharedMemory(create=True, size=size)
        hdr = np.ndarray((8,), dtype=np.int64, buffer=shm.buf, offset=0)
        hdr[:] = 0
        hdr[_CAP], hdr[_CH], hdr[_ITEM] = capacity, channels, dtype.itemsize
        del hdr
        return cls(shm, dtype, owner=True)

    @classmethod
    def attach(cls, name: str, dtype: np.dtype) -> SharedRing:
        shm = shared_memory.SharedMemory(name=name, track=False)
        return cls(shm, dtype, owner=False)

    @property
    def name(self) -> str:
        return self._shm.name

    @property
    def write_count(self) -> int:
        """Şimdiye kadar yazılan (boşluklar dahil) toplam örnek sayısı."""
        return self._counter[0]

    # ---------------------------------------------------------------- yazma (tek yazıcı)
    def write_bytes(self, src: memoryview, n: int) -> None:
        """Satır sıralı (interleaved) `n` örneği bayt olarak kopyalar — alıcının sıcak yolu."""
        cap = self.capacity
        if n >= cap:
            self.write(np.frombuffer(src, dtype=self.dtype).reshape(n, self.channels))
            return
        wc = self._wc
        rb = self._row_bytes
        pos = wc % cap
        end = pos + n
        if end <= cap:
            self._raw[pos * rb:end * rb] = src
        else:
            split = (cap - pos) * rb
            self._raw[pos * rb:] = src[:split]
            self._raw[:(end - cap) * rb] = src[split:]
        self._wc = wc + n
        self._counter[0] = wc + n

    def write(self, block: np.ndarray) -> None:
        n = block.shape[0]
        if n == 0:
            return
        wc = self._wc
        cap = self.capacity
        if n > cap:
            block = block[n - cap:]
            start = wc + n - cap
        else:
            start = wc
        pos = start % cap
        first = min(block.shape[0], cap - pos)
        self.data[pos:pos + first] = block[:first]
        if first < block.shape[0]:
            self.data[:block.shape[0] - first] = block[first:]
        self._wc = wc + n
        self._counter[0] = wc + n

    def skip(self, n: int) -> None:
        """`n` örneklik kayıp: bölgeyi sıfırlar ve boşluk kaydına ekler."""
        if n <= 0:
            return
        wc = self._wc
        slot = int(self._hdr[_GAPN]) % GAP_SLOTS
        self._gaps[slot] = (wc, n)
        self._hdr[_GAPN] += 1
        fill = min(n, self.capacity)
        pos = (wc + n - fill) % self.capacity
        first = min(fill, self.capacity - pos)
        rb = self._row_bytes
        self._bytes[pos * rb:(pos + first) * rb] = 0
        if first < fill:
            self._bytes[:(fill - first) * rb] = 0
        self._wc = wc + n
        self._counter[0] = wc + n

    # ---------------------------------------------------------------- okuma
    def segments(self, start: int, end: int) -> list[tuple[int, np.ndarray]]:
        """[start, end) mutlak örnek aralığını en fazla iki bitişik görünüm olarak döndürür.

        Tamponda artık bulunmayan kısım baştan kırpılır. Görünümler kopya değildir.
        """
        end = min(end, self.write_count)
        start = max(start, end - self.capacity, 0)
        if start >= end:
            return []
        cap = self.capacity
        a, b = start % cap, end % cap or cap
        if start // cap == (end - 1) // cap:
            return [(start, self.data[a:b])]
        return [(start, self.data[a:]), (start + cap - a, self.data[:b])]

    def read(self, start: int, end: int) -> tuple[int, np.ndarray]:
        """Aralığın kopyasını döndürür; kopyalama sırasında üzerine yazılan baş kısım atılır."""
        segs = self.segments(start, end)
        if not segs:
            return end, self.data[:0].copy()
        first_index = segs[0][0]
        out = np.concatenate([s for _, s in segs]) if len(segs) > 1 else segs[0][1].copy()
        valid_from = self.write_count - self.capacity
        if valid_from > first_index:
            cut = valid_from - first_index
            return first_index + cut, out[cut:]
        return first_index, out

    def valid_from(self) -> int:
        """Güvenilir en eski örneğin mutlak indeksi: üzerine yazılmamış olmalı ve kayıp kaydı taşmışsa (çok sayıda
        kayıp) hatırlanan en eski kaybın başından sonra olmalı — daha öncesindeki kayıplar artık bilinmediği için
        sıfırla doldurulmuş bölgeler veri sanılmasın."""
        lowest = max(0, self.write_count - self.capacity)
        count = int(self._hdr[_GAPN])
        if count > GAP_SLOTS:
            lowest = max(lowest, int(self._gaps[count % GAP_SLOTS, 0]))
        return lowest

    def gaps_between(self, start: int, end: int) -> list[tuple[int, int]]:
        """[start, end) ile kesişen kayıp aralıkları (başlangıç, bitiş), başlangıca göre sıralı."""
        count = int(self._hdr[_GAPN])
        if count == 0:
            return []
        rows = self._gaps[:min(count, GAP_SLOTS)]
        gs, ge = rows[:, 0], rows[:, 0] + rows[:, 1]
        hit = (gs < end) & (ge > start)
        if not hit.any():
            return []
        order = np.argsort(gs[hit], kind="stable")
        return [(int(a), int(b)) for a, b in zip(gs[hit][order], ge[hit][order])]

    def close(self) -> None:
        self.data = self._bytes = self._hdr = self._gaps = None
        for view in (self._raw, self._counter):
            view.release()
        try:
            self._shm.close()
            if self._owner:
                self._shm.unlink()
        except (BufferError, FileNotFoundError):
            pass
