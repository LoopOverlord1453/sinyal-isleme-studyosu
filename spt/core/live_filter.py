"""Canlı filtre: işlem zincirini akışlara blok blok, durum taşıyarak uygular.

Her (akış, kanal) için bir iz tutulur: zincirin çalıştırıcısı (zi / geçmiş örnekler) ve filtre çıktılarının
yazıldığı yerel halka tampon. Her güncellemede yalnızca yeni gelen örnekler filtrelenir; çıktı mutlak örnek
indeksleriyle ham veriye hizalıdır, böylece ham ve filtrelenmiş sinyal aynı eksende üst üste çizilir.

Kayıp bölgelerde çıktı NaN olur ve filtre kayıptan sonraki ilk örnekle yeniden başlar (kararlı hâl başlangıcı,
sıçrama yok). Filtre gelen veriye yetişemezse (tampon üzerine yazıldıysa) geride kalan kısım atlanır.

Arayüz donmasın diye: yeni bir iz (zincir değişti, geri al / yinele, akış eklendi) tüm tamponu değil yalnızca
istenen geçmişi (ör. görünen pencere) filtreleyerek başlar ve tek bir güncellemede en fazla `budget` örnek işlenir;
kalan kısım sonraki karelerde yetişilir.
"""
from __future__ import annotations

import numpy as np

from spt.core.local_ring import LocalRing
from spt.dsp.chain import ProcessingChain
from spt.dsp.convert import physical_column
from spt.dsp.filters import ChainRunner, finite_runs

MAX_TRACK_SAMPLES = 1 << 22  # kanal başına tutulan filtre çıktısı (float64'te 32 MB)
CHUNK = 1 << 20              # tek seferde filtrelenen en fazla örnek (geçici bellek sınırı)
UPDATE_BUDGET = 1 << 19      # bir güncellemede (kare) en fazla işlenen örnek
LIVE_MEDIAN_MAX = 4097       # canlıda medyan penceresi sınırı (her karede pencere boyu kadar geçmiş yeniden sıralanır)


class _Track:
    def __init__(self, handle, channel: int, runner: ChainRunner, out: LocalRing):
        self.handle = handle
        self.channel = channel
        self.runner = runner
        self.out = out


class LiveFilter:
    def __init__(self, max_samples: int = MAX_TRACK_SAMPLES):
        self.chain = ProcessingChain()
        self.enabled = False
        self.max_samples = max_samples
        self._tracks: dict[tuple[str, int], _Track] = {}
        self.errors: dict[float, list[str]] = {}  # fs → bu hızda tasarlanamayan aşamalar
        self.skipped = 0                          # yetişemediği için atlanan bölge sayısı

    @property
    def active(self) -> bool:
        return self.enabled and self.chain.enabled_count > 0

    def set_chain(self, chain: ProcessingChain) -> None:
        self.chain = chain
        self._tracks.clear()
        self.errors.clear()

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        if not enabled:
            self._tracks.clear()

    def forget(self, key: str) -> None:
        for k in [k for k in self._tracks if k[0] == key]:
            del self._tracks[k]

    def prune_errors(self, rates) -> None:
        """Artık hiçbir akışın kullanmadığı örnekleme hızlarının tasarım hatalarını unut."""
        keep = set(rates)
        for fs in [fs for fs in self.errors if fs not in keep]:
            del self.errors[fs]

    def clear(self) -> None:
        self._tracks.clear()

    def has(self, key: str, channel: int) -> bool:
        return (key, channel) in self._tracks

    # ---------------------------------------------------------------- işleme
    def update(self, handle, channel: int, end: int | None = None, history: int | None = None,
               budget: int | None = UPDATE_BUDGET) -> None:
        """Kanalın [işlenmiş, end) aralığını filtreler (end: varsayılan en son örnek).

        history: yeni iz en fazla bu kadar geriden başlar (None: tamponun tamamı). budget: bu çağrıda işlenecek en
        fazla örnek (None: sınırsız) — geride kalınırsa sonraki çağrılar yetişir."""
        if not self.active:
            return
        ring = handle.ring
        end = ring.write_count if end is None else min(end, ring.write_count)
        key = (handle.key, channel)
        track = self._tracks.get(key)
        if track is None or track.handle is not handle:
            designs, errors = self.chain.designs(handle.fs)
            long_median = [d for d in designs if d.kind == "median" and d.median_length > LIVE_MEDIAN_MAX]
            if long_median:  # kare başına yüzlerce ms sürer: canlıda uygulanmaz, Dondur + çevrimdışı filtre önerilir
                errors = errors + [f"medyan penceresi {d.median_length} örnek — canlıda en fazla {LIVE_MEDIAN_MAX}; "
                                   "uzun pencere için Dondur + çevrimdışı filtre kullanın" for d in long_median]
                designs = [d for d in designs if d not in long_median]
            self.errors[handle.fs] = errors
            if not designs:
                self._tracks.pop(key, None)
                return
            capacity = min(self.max_samples, ring.capacity)
            start = max(ring.valid_from(), end - capacity, end - history if history else 0)
            dtype = np.complex128 if handle.desc.sample_type.is_complex else np.float64
            track = _Track(handle, channel, ChainRunner(designs), LocalRing(capacity, dtype, origin=start))
            self._tracks[key] = track
        out = track.out
        pos = out.write_count
        lowest = max(ring.valid_from(), end - out.capacity)
        if pos < lowest:  # işlenmeden üzerine yazıldı ya da çok geride: en yeni kısımdan devam
            out.skip(lowest - pos)
            track.runner.reset()
            pos = lowest
            self.skipped += 1
        if pos >= end:
            return
        if budget is not None:
            end = min(end, pos + budget)
        for gs, ge in sorted(ring.gaps_between(pos, end)):
            gs, ge = max(gs, pos), min(ge, end)
            if ge <= pos:
                continue
            if gs > pos:
                self._run(track, pos, gs)
            out.skip(ge - gs)
            track.runner.reset()
            pos = ge
        if pos < end:
            try:
                self._run(track, pos, end)
            except Exception as exc:  # noqa: BLE001 — arayüzün yenilemesi asla durmamalı
                self.errors.setdefault(handle.fs, []).append(f"uygulama hatası: {exc}")
                self._tracks.pop(key, None)

    def _run(self, track: _Track, start: int, end: int) -> None:
        handle, ch = track.handle, track.channel
        desc = handle.desc
        pos = start
        while pos < end:
            stop = min(end, pos + CHUNK)
            first, raw = handle.ring.read(pos, stop)
            if first > pos:  # okunurken üzerine yazıldı
                track.out.skip(first - pos)
                track.runner.reset()
            if len(raw):
                x = physical_column(raw[:, ch], desc.sample_type, desc.channels[ch])
                for a, b, finite in finite_runs(x):  # NaN / ±inf örnek kayıp gibi: çıktı NaN, filtre yeniden başlar
                    if finite:
                        track.out.write(track.runner.process(x[a:b]))
                    else:
                        track.out.skip(b - a)
                        track.runner.reset()
            pos = stop

    # ---------------------------------------------------------------- okuma
    def segments(self, key: str, channel: int, start: int, end: int, part: int = 0) -> list[tuple[int, np.ndarray]]:
        track = self._tracks.get((key, channel))
        if track is None:
            return []
        segs = track.out.segments(start, end)
        if np.iscomplexobj(track.out.data):
            return [(s, v.imag if part else v.real) for s, v in segs]
        return segs if part == 0 else []

    def read_clean(self, key: str, channel: int, start: int, end: int) -> np.ndarray | None:
        """[start, end) aralığının filtre çıktısı; tamamı mevcut ve kayıpsız değilse None."""
        track = self._tracks.get((key, channel))
        if track is None:
            return None
        first, y = track.out.read(start, end)
        if first != start or len(y) != end - start or not np.all(np.isfinite(y)):
            return None
        return y
