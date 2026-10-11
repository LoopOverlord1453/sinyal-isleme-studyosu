"""Sensör füzyonu motoru: senaryonun algoritmalarını ayrı süreçlerde, paralel ve arayüzü bloke etmeden çalıştırır.

Senaryolar (`spt.dsp.fusion_generic`): IMU / MARG yönelimi, barometre + ivme irtifası, GNSS + jiroskop + odometri
2B konumu, çok sensörlü doğrusal Kalman. Her algoritma kendi sürecindedir (Python'da tek çekirdek sınırını aşar;
UKF / parçacık filtresi gibi pahalı algoritmalar diğerlerini ve arayüzü yavaşlatmaz). Arayüz tarafı
(`FusionEngine`, Qt'siz) tampondan yeni örnekleri parçalar halinde gönderir, sonuçları mutlak örnek indeksine hizalı
yerel halka tamponlara yazar. Canlı akış, kayıt oynatma ve dondurulmuş anlık görüntü aynı yoldan işlenir. Kayıp
bölgelerde çıktı NaN olur ve algoritmalar kayıptan sonra yeniden başlatılır.

Motor ayrıca referans (gerçek) değerleri ve "ham" (füzyonsuz, tek sensörden) kestirimi gönderim anında kendi
tamponuna kopyalar: akışın halka tamponu kısa olsa da (ör. 10 s) sonuçlar, referans ve ham değerler dakikalarca
aynı eksende gösterilebilir. Doğruluk metrikleri (RMSE, en büyük hata) gelen her sonuçla birikir — yalnız görünen
aralığı değil, işlenen tüm veriyi kapsar (her (yeniden) başlangıçtan sonraki ilk 2 s yakınsama hariç).

Kapatma engellemesizdir: durdurma bayrağı kurulur, kuyruklar besleyici iş parçacıkları beklenmeden kapatılır ve
süreçler arka planda toplanır (gerekirse sonlandırılır); işçiler ana süreç ölürse kendiliğinden çıkar.
"""
from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field

import numpy as np

from spt.core.local_ring import LocalRing
from spt.dsp.convert import physical_column
from spt.dsp.fusion_generic import (
    SCENARIOS,
    FusionError,
    Mapping,
    auto_map,
    errors,
    gather_inputs,
    gather_reference,
    make_estimator,
)

CHUNK = 2000             # bir seferde gönderilen en fazla örnek
FIRST_CHUNK = 200        # ilk parça küçük: maliyet ölçülür, sonraki parçalar ~CHUNK_SECONDS sürecek boyda
CHUNK_SECONDS = 0.2      # parça başına hedef işlem süresi (durdurma ve yeniden başlatma bu kadar gecikir)
MAX_IN_FLIGHT = 3        # algoritma başına yanıt bekleyen en fazla parça (geri basınç)
MAX_TRACK = 1 << 20      # seri başına tutulan en fazla sonuç
HISTORY_SECONDS = 600.0  # sonuç / referans geçmişi (halka tampondan bağımsız)
SETTLE_SECONDS = 2.0     # metriklerde her (yeniden) başlangıç sonrası dışarıda bırakılan yakınsama süresi

__all__ = ["FusionEngine", "Mapping", "auto_map", "run_worker"]


# ============================================================================ işçi süreç
def run_worker(scenario: str, key: str, scenario_values: dict, params: dict, commands, results,
               stop_event=None) -> None:
    """Bir algoritmanın süreci: ("data", başlangıç, girişler, dt) → ("result", …).

    Durdurma bayrağı her parçadan önce denetlenir; ana süreç ölmüşse (sert kapanma) işçi de çıkar."""
    parent = mp.parent_process()
    try:
        est = make_estimator(scenario, key, dict(scenario_values), dict(params))
        while True:
            try:
                cmd = commands.get(timeout=0.5)
            except queue.Empty:
                if parent is not None and not parent.is_alive():
                    break
                continue
            if cmd[0] == "stop" or (stop_event is not None and stop_event.is_set()):
                break
            if cmd[0] == "reset":
                est = make_estimator(scenario, key, dict(scenario_values), dict(params))
                continue
            _, start, inputs, dt = cmd
            y = est.run(inputs, dt)
            results.put(("result", key, start, np.asarray(y, dtype=np.float32), est.diagnostics(), est.cost_us))
    except Exception:  # noqa: BLE001
        results.put(("error", key, traceback.format_exc()))
    finally:
        results.put(("stopped", key))


# ============================================================================ arayüz tarafı motor
@dataclass
class _Algo:
    key: str
    params: dict
    proc: object
    commands: object
    out: list[LocalRing]
    sent: int                  # gönderilen son örneğin sonrası (mutlak)
    in_flight: int = 0
    diagnostics: dict = field(default_factory=dict)
    cost_us: float = 0.0
    error: str | None = None
    overrun: int = 0           # yetişemediği için işlenmeden atlanan örnek (tampon üzerine yazıldı)
    settle_until: int = 0      # bu indeksten önceki sonuçlar metriklere katılmaz (yakınsama)
    sq: np.ndarray | None = None      # Σ hata² (iç birim²), çıktı başına
    count: np.ndarray | None = None   # metrikteki örnek sayısı
    worst: np.ndarray | None = None   # en büyük |hata|


class FusionEngine:
    """Bir akış / anlık görüntü üzerinde senaryonun seçili algoritmalarını çalıştırır. `pump()` düzenli çağrılmalı."""

    def __init__(self, handle, scenario: str, mapping: Mapping, scenario_values: dict, algorithms: dict[str, dict],
                 start: int | None = None):
        self.handle = handle
        self.scenario = SCENARIOS[scenario]
        self.mapping = mapping
        self.values = dict(scenario_values)
        self.scenario.check(mapping.inputs, self.values)  # eksik giriş / geçersiz parametre → FusionError
        self.fs = float(handle.fs)
        self.k = len(self.scenario.outputs)
        ring = handle.ring
        self.start = ring.valid_from() if start is None else max(start, ring.valid_from())
        self._channels = sorted({i for idx in mapping.inputs.values() for i in idx} | set(mapping.reference.values()))
        self._raw_values = dict(self.values)  # raw() durum anahtarları (_p0, _origin) parça parça korunur
        self._ctx = mp.get_context("spawn")
        self._results = self._ctx.Queue()
        self._stop = self._ctx.Event()
        capacity = int(min(MAX_TRACK, max(ring.capacity, self.fs * HISTORY_SECONDS, 1024)))
        self.capacity = capacity
        self._ref = [LocalRing(capacity, np.float32, origin=self.start) for _ in range(self.k)]
        self._raw = [LocalRing(capacity, np.float32, origin=self.start) for _ in range(self.k)]
        self.algos: dict[str, _Algo] = {}
        settle = self.start + int(SETTLE_SECONDS * self.fs)
        for key, params in algorithms.items():
            if key not in self.scenario.algorithms:
                continue
            commands = self._ctx.Queue()
            proc = self._ctx.Process(target=run_worker,
                                     args=(self.scenario.key, key, self.values, dict(params), commands, self._results,
                                           self._stop),
                                     name=f"spt-fusion-{key}", daemon=True)
            proc.start()
            out = [LocalRing(capacity, np.float32, origin=self.start) for _ in range(self.k)]
            self.algos[key] = _Algo(key, dict(params), proc, commands, out, self.start, settle_until=settle,
                                    sq=np.zeros(self.k), count=np.zeros(self.k), worst=np.zeros(self.k))
        self.started = time.perf_counter()
        self.stopped = False

    # ---------------------------------------------------------------- veri okuma
    def _read(self, start: int, end: int) -> tuple[int, np.ndarray | None]:
        """[start, end) için (ilk indeks, (n, kanal) fiziksel değerler); eşlenmemiş kanallar NaN."""
        h = self.handle
        first, raw = h.ring.read(start, end)
        if len(raw) == 0:
            return first, None
        cols = np.full((len(raw), len(h.desc.channels)), np.nan)
        for i in self._channels:
            cols[:, i] = physical_column(raw[:, i], h.desc.sample_type, h.desc.channels[i]).real
        return first, cols

    def _chunk(self, algo: _Algo) -> int:
        """Parça boyu: ölçülen maliyete göre ~CHUNK_SECONDS sürecek kadar (durdurma gecikmesi sınırlı kalır)."""
        if algo.cost_us <= 0:
            return FIRST_CHUNK
        return int(min(CHUNK, max(50, CHUNK_SECONDS * 1e6 / algo.cost_us)))

    def _mirror(self, end: int) -> None:
        """Referans ve ham (füzyonsuz) değerleri [kopyalanan, end) için motorun kendi tamponlarına yazar."""
        ring = self.handle.ring
        pos = self._ref[0].write_count
        lowest = ring.valid_from()
        if pos < lowest:
            for r in (*self._ref, *self._raw):
                r.skip(lowest - pos)
            pos = lowest
        while pos < end:
            stop = min(end, pos + (1 << 16))
            first, cols = self._read(pos, stop)
            if cols is None:
                break
            if first > pos:
                for r in (*self._ref, *self._raw):
                    r.skip(first - pos)
            n = len(cols)
            ref = gather_reference(self.scenario, self.mapping, self.handle.desc, cols)
            try:
                raw = np.asarray(self.scenario.raw(gather_inputs(self.scenario, self.mapping, cols), self._raw_values),
                                 dtype=np.float64)
            except (FusionError, ValueError):
                raw = np.full((n, self.k), np.nan)
            for gs, ge in ring.gaps_between(first, first + n):
                a, b = max(gs, first) - first, min(ge, first + n) - first
                raw[a:b] = np.nan
                ref[a:b] = np.nan
            for i in range(self.k):
                self._raw[i].write(raw[:, i].astype(np.float32))
                self._ref[i].write(ref[:, i].astype(np.float32))
            pos = first + n

    # ---------------------------------------------------------------- veri gönderme / sonuç alma
    def pump(self, end: int | None = None, budget_s: float = 0.02) -> None:
        """Sonuçları topla ve bekleyen yeni örnekleri işçilere gönder."""
        if self.stopped:
            return
        deadline = time.perf_counter() + budget_s
        while time.perf_counter() < deadline:
            try:
                msg = self._results.get_nowait()
            except (queue.Empty, OSError, ValueError):
                break
            self._handle(msg)
        ring = self.handle.ring
        end = ring.write_count if end is None else min(end, ring.write_count)
        self._mirror(end)
        for algo in self.algos.values():
            if algo.error:
                continue
            if algo.in_flight == 0 and algo.out[0].write_count < algo.sent:
                for r in algo.out:  # sondaki kayıp / atlanan bölge: sonraki sonuç gelmeyecek, NaN ile kapat
                    r.skip(algo.sent - r.write_count)
            while algo.in_flight < MAX_IN_FLIGHT and algo.sent < end:
                if algo.sent < ring.valid_from():  # gönderilmeden üzerine yazıldı: en eskiden yeniden başla
                    algo.overrun += ring.valid_from() - algo.sent
                    self._skip(algo, ring.valid_from() - algo.sent)
                stop = min(end, algo.sent + self._chunk(algo))
                gaps = ring.gaps_between(algo.sent, stop)
                if gaps:
                    gs, ge = min(gaps)
                    if gs <= algo.sent:  # kayıp bölge: çıktıyı NaN bırak, algoritmayı yeniden başlat
                        self._skip(algo, min(ge, end) - algo.sent)
                        continue
                    stop = gs
                first, cols = self._read(algo.sent, stop)
                if cols is None:
                    break
                if first > algo.sent:
                    self._skip(algo, first - algo.sent)
                inputs = gather_inputs(self.scenario, self.mapping, cols)
                algo.commands.put(("data", first, inputs, 1.0 / self.fs))
                algo.sent = first + len(cols)
                algo.in_flight += 1

    def _skip(self, algo: _Algo, n: int) -> None:
        if n <= 0:
            return
        algo.commands.put(("reset",))  # sonuç halkası bir sonraki sonuçta boşluğu NaN ile doldurur
        algo.sent += n
        algo.settle_until = algo.sent + int(SETTLE_SECONDS * self.fs)  # yeniden başlangıç: yakınsama yine hariç

    def _handle(self, msg: tuple) -> None:
        kind, key = msg[0], msg[1]
        algo = self.algos.get(key)
        if algo is None:
            return
        if kind == "result":
            _, _, start, values, diagnostics, cost = msg
            algo.in_flight = max(0, algo.in_flight - 1)
            ring = algo.out[0]
            if start > ring.write_count:
                for r in algo.out:
                    r.skip(start - r.write_count)
            for i, r in enumerate(algo.out):
                r.write(values[:, i])
            algo.diagnostics, algo.cost_us = diagnostics, cost
            self._accumulate(algo, start, values)
        elif kind == "error":
            algo.error = msg[2]
            algo.in_flight = 0

    def _accumulate(self, algo: _Algo, start: int, values: np.ndarray) -> None:
        """Gelen sonucun referansa göre hatasını metriklere ekler (yakınsama süresi hariç)."""
        if not self.mapping.reference:
            return
        lo = max(start, algo.settle_until)
        hi = start + len(values)
        if lo >= hi:
            return
        ref = self._series(self._ref, lo, hi)
        err = np.abs(errors(self.scenario, values[lo - start:].astype(np.float64), ref))
        ok = np.isfinite(err)
        algo.sq += (np.where(ok, err, 0.0) ** 2).sum(axis=0)
        algo.count += ok.sum(axis=0)
        if ok.any():
            algo.worst = np.maximum(algo.worst, np.where(ok, err, 0.0).max(axis=0))

    def metrics(self, key: str) -> tuple[np.ndarray, np.ndarray, float]:
        """(RMSE (k) iç birimde — örnek yoksa NaN, en büyük |hata| (k), kapsanan süre s)."""
        a = self.algos[key]
        with np.errstate(invalid="ignore", divide="ignore"):
            rmse = np.where(a.count > 0, np.sqrt(a.sq / np.maximum(a.count, 1)), np.nan)
            worst = np.where(a.count > 0, a.worst, np.nan)
        return rmse, worst, float(a.count.max()) / self.fs if len(a.count) else 0.0

    # ---------------------------------------------------------------- durum
    def processed(self, key: str) -> int:
        return self.algos[key].out[0].write_count

    def progress(self, end: int | None = None) -> float:
        end = self.handle.ring.write_count if end is None else end
        total = max(end - self.start, 1)
        done = min((self.processed(k) for k in self.algos), default=end)
        return min(1.0, (done - self.start) / total)

    @property
    def busy(self) -> bool:
        return any(a.in_flight for a in self.algos.values())

    def _series(self, rings: list[LocalRing], start: int, end: int) -> np.ndarray:
        out = np.full((max(end - start, 0), len(rings)), np.nan)
        for i, r in enumerate(rings):
            for s0, v in r.segments(start, end):
                out[s0 - start:s0 - start + len(v), i] = v
        return out

    def read(self, key: str, start: int, end: int) -> np.ndarray:
        """Algoritmanın (n, k) kestirimi iç birimde (açılar rad); hesaplanmamış / kayıp örnekler NaN."""
        return self._series(self.algos[key].out, start, end)

    def reference(self, start: int, end: int) -> np.ndarray | None:
        """Referans (gerçek) değerler iç birimde; eşlenmemiş çıktı / kopyalanmamış örnek NaN."""
        if not self.mapping.reference:
            return None
        return self._series(self._ref, start, end)

    def raw_estimate(self, start: int, end: int) -> np.ndarray:
        """Füzyonsuz (tek sensörden) kestirim, iç birimde; uygulanamayan çıktı NaN."""
        return self._series(self._raw, start, end)

    @property
    def history_start(self) -> int:
        """Motor tamponlarında tutulan en eski örnek."""
        return max(self.start, self._ref[0].write_count - self.capacity)

    @property
    def mirrored(self) -> int:
        return self._ref[0].write_count

    # ---------------------------------------------------------------- kapatma
    def stop(self, wait: bool = False) -> None:
        """Engellemesiz durdurma: bayrak + "stop" komutu, kuyruklar besleyici iş parçacığı beklenmeden kapatılır,
        süreçler arka plan iş parçacığında toplanır (3 s'de çıkmayan sonlandırılır). Sonuçlar `algos` içinde
        okunmaya devam eder (görünüm son durumu gösterebilir). `wait=True`: toplama bitene kadar bekle."""
        if self.stopped:
            return
        self.stopped = True
        self._stop.set()
        procs = []
        for algo in self.algos.values():
            try:
                algo.commands.put(("stop",))
            except (OSError, ValueError):
                pass
            # Okuyan kalmayacak: kuyruğa yazılmış ama gönderilemeyen parçalar çıkışta süreci kilitlemesin
            algo.commands.cancel_join_thread()
            algo.commands.close()
            procs.append(algo.proc)
        self._results.cancel_join_thread()
        reaper = threading.Thread(target=_reap, args=(procs, self._results), name="spt-fusion-reaper", daemon=True)
        reaper.start()
        if wait:
            reaper.join()


def _reap(procs: list, results) -> None:
    """Durdurulan işçileri toplar: sonuç kuyruğunu boşaltır (işçi yazarken tıkanmasın), 3 s sonra sonlandırır."""
    deadline = time.monotonic() + 3.0
    while any(p.is_alive() for p in procs) and time.monotonic() < deadline:
        try:
            while True:
                results.get_nowait()
        except (queue.Empty, OSError, ValueError, EOFError):
            pass
        for p in procs:
            p.join(timeout=0.02)
    for p in procs:
        if p.is_alive():
            p.terminate()
            p.join(timeout=1.0)
    try:
        results.close()
    except (OSError, ValueError):
        pass
