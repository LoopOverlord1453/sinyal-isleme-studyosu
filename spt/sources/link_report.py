"""Pasif link testi: canlı akışın istatistik olaylarından belirli bir süre için bağlantı raporu.

Alıcı (ya da oynatıcı) her ≈ 0.25 s'de akış başına birikimli `StreamStats` gönderir; test başlangıç ve bitiş
anlarının farkı pencerenin değerlerini verir (paket arası süre ortalaması / varyansı birikimli Welford
değerlerinden tam, yüzdelikler histogram farkından ≈ %12 kutu çözünürlüğüyle). En yüksek veri hızı ve en büyük
boşluk ara anlık değerlerin en büyüğüdür. Sonuç gerçek zaman eşiklerine göre değerlendirilir; Markdown / CSV olarak
dışa aktarılabilir. Bu modül Qt kullanmaz.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from spt.core.stream_store import TIMING_EDGES_US, StreamStats, hist_percentile


@dataclass(frozen=True)
class Thresholds:
    """Gerçek zaman uygunluk eşikleri (kullanıcı ayarlar). NaN / 0 olmayan her eşik değerlendirilir."""
    max_gap_ms: float = 20.0          # paketler arası en büyük boşluk
    loss_pct: float = 0.01            # kayıp örnek oranı
    jitter_p99_us: float = 5000.0     # paket arası sürenin p99'u − beklenen aralık
    rate_ppm: float = 500.0           # etkin örnekleme hızının nominalden sapması (mutlak)
    min_mbps: float = 0.0             # en az ortalama veri hızı (0: değerlendirilmez)


@dataclass
class Check:
    name: str
    value: str
    limit: str
    ok: bool | None                    # None: ölçülemedi


@dataclass
class PassiveReport:
    stream: str
    source: str
    format: str
    duration_s: float
    packets: int
    bytes: int
    samples: int
    mean_mbps: float
    peak_mbps: float
    mean_pps: float
    peak_pps: float
    lost_samples: int
    loss_pct: float
    loss_events: int
    max_loss_burst: int
    duplicates: int
    late_packets: int
    restarts: int
    expected_us: float
    mean_us: float
    std_us: float
    p50_us: float
    p95_us: float
    p99_us: float
    max_us: float
    hist: np.ndarray                   # pencere histogramı (TIMING_EDGES_US kutuları)
    rate_hz: float
    nominal_hz: float
    rate_ppm: float
    rate_ppm_err: float
    pdv_us: float
    sender_ppm: float
    collisions: int = 0                # test süresince başka kaynak portundan gelen datagram
    conflicts: tuple[str, ...] = ()    # görülen çakışma türleri ("port", "sıra") ve portlar (metin)
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool | None:
        known = [c.ok for c in self.checks if c.ok is not None]
        return all(known) if known else None

    # ------------------------------------------------------------------ dışa aktarma
    def rows(self) -> list[tuple[str, str]]:
        """(ölçü, değer) satırları: rapor tablosu, Markdown ve CSV ortak kaynağı."""
        return [
            ("Akış", self.stream), ("Kaynak", self.source), ("Format", self.format),
            ("Süre", f"{self.duration_s:.2f} s"),
            ("Paket", f"{self.packets}"), ("Veri", f"{self.bytes / 1e6:.3f} MB"),
            ("Ortalama veri hızı", f"{self.mean_mbps:.3f} Mbit/s"), ("En yüksek veri hızı", f"{self.peak_mbps:.3f} Mbit/s"),
            ("Ortalama paket hızı", f"{self.mean_pps:.1f} /s"), ("En yüksek paket hızı", f"{self.peak_pps:.1f} /s"),
            ("Kayıp örnek", f"{self.lost_samples} ({self.loss_pct:.4f} %)"),
            ("Kayıp olayı", f"{self.loss_events} (en uzun {self.max_loss_burst} örnek)"),
            ("Yinelenen çerçeve", f"{self.duplicates}"), ("Sıra dışı (geç) çerçeve", f"{self.late_packets}"),
            ("Yeniden başlama", f"{self.restarts}"),
            ("Kaynak çakışması", "; ".join(self.conflicts) if self.conflicts else "yok"),
            ("Beklenen paket aralığı", fmt_us(self.expected_us)),
            ("Paket aralığı ortalama", fmt_us(self.mean_us)), ("Titreşim (std)", fmt_us(self.std_us)),
            ("Paket aralığı p50 / p95 / p99", f"{fmt_us(self.p50_us)} / {fmt_us(self.p95_us)} / {fmt_us(self.p99_us)}"),
            ("En büyük boşluk", fmt_us(self.max_us)),
            ("Etkin örnekleme hızı", f"{self.rate_hz:.6g} Hz" if math.isfinite(self.rate_hz) else "—"),
            ("Nominal örnekleme hızı", f"{self.nominal_hz:.6g} Hz"),
            ("Hız sapması", fmt_ppm(self.rate_ppm, self.rate_ppm_err)),
            ("Gecikme değişimi (PDV, p99 − en küçük)", fmt_us(self.pdv_us)),
            ("Gönderici saati − PC saati", fmt_ppm(self.sender_ppm)),
        ]

    def to_markdown(self) -> str:
        verdict = {True: "**UYGUN**", False: "**UYGUN DEĞİL**", None: "değerlendirilemedi"}[self.passed]
        lines = [f"# UDP link testi (pasif) — {self.stream}", "", f"Gerçek zaman uygunluğu: {verdict}", "",
                 "| Ölçü | Değer |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in self.rows()]
        lines += ["", "## Eşikler", "", "| Kontrol | Ölçülen | Sınır | Sonuç |", "|---|---|---|---|"]
        lines += [f"| {c.name} | {c.value} | {c.limit} | {_verdict(c.ok)} |" for c in self.checks]
        lines += ["", "## Paket arası süre histogramı", "", "| Alt sınır (µs) | Üst sınır (µs) | Paket |", "|---|---|---|"]
        lines += [f"| {lo:.3g} | {hi:.3g} | {int(n)} |" for lo, hi, n in histogram_rows(self.hist)]
        return "\n".join(lines) + "\n"

    def to_csv(self) -> str:
        out = ["ölçü;değer"] + [f"{k};{v}" for k, v in self.rows()]
        out += [f"kontrol: {c.name};{c.value} (sınır {c.limit}) {_verdict(c.ok)}" for c in self.checks]
        out += ["histogram alt (µs);üst (µs);paket"] + [f"{lo:.6g};{hi:.6g};{int(n)}"
                                                       for lo, hi, n in histogram_rows(self.hist)]
        return "\n".join(out) + "\n"


def _verdict(ok: bool | None) -> str:
    return {True: "uygun", False: "AŞILDI", None: "ölçülemedi"}[ok]


def fmt_us(us: float) -> str:
    if us is None or not math.isfinite(us):
        return "—"
    if us >= 1e6:
        return f"{us / 1e6:.3g} s"
    if us >= 1e3:
        return f"{us / 1e3:.3g} ms"
    return f"{us:.3g} µs"


def fmt_ppm(ppm: float, err: float | None = None) -> str:
    if ppm is None or not math.isfinite(ppm):
        return "—"
    tail = f" ± {err:.0f}" if err is not None and math.isfinite(err) and err >= 0.5 else ""
    return f"{ppm:+.1f}{tail} ppm"


def histogram_rows(hist: np.ndarray) -> list[tuple[float, float, float]]:
    """Boş olmayan histogram kutuları: (alt, üst, sayı), µs. İlk kutu: 0 … 1 µs, son kutu: 10 s … ∞."""
    edges = np.concatenate(([0.0], TIMING_EDGES_US, [math.inf]))
    return [(edges[i], edges[i + 1], float(n)) for i, n in enumerate(hist) if n]


class PassiveTest:
    """Bir akışın istatistik olaylarını biriktirir; `report` pencerenin sonuçlarını verir."""

    def __init__(self, key: str, thresholds: Thresholds | None = None):
        self.key = key
        self.thresholds = thresholds or Thresholds()
        self.first: tuple[float, StreamStats] | None = None
        self.last: tuple[float, StreamStats] | None = None
        self.peak_mbps = 0.0
        self.peak_pps = 0.0
        self.max_gap_us = 0.0
        self.pdv_us = math.nan
        self.series: list[tuple[float, float, float]] = []   # (t − başlangıç, Mbit/s, paket/s)
        self.conflicts: dict[str, str] = {}                     # tür → açıklama (test boyunca görülen)

    @property
    def elapsed_s(self) -> float:
        return self.last[0] - self.first[0] if self.first and self.last else 0.0

    def feed(self, t: float, st: StreamStats) -> None:
        """`t`: olayın saati (s, olayı üreten süreç); `st`: akışın birikimli istatistiği."""
        if st.key != self.key:
            return
        kind = getattr(st, "conflict", "")
        if kind == "port":
            self.conflicts["port"] = "iki kaynak portu: " + ", ".join(f"{st.source}:{p}" for p in st.source_ports)
        elif kind == "sıra":
            self.conflicts["sıra"] = "aynı porttan karışık first_sample dizisi"
        if self.first is None:
            self.first = self.last = (t, st)
            return
        t_prev, prev = self.last
        dt = t - t_prev
        if dt > 1e-3:
            mbps = (st.bytes - prev.bytes) * 8 / dt / 1e6
            pps = (st.packets - prev.packets) / dt
            self.peak_mbps = max(self.peak_mbps, mbps)
            self.peak_pps = max(self.peak_pps, pps)
            self.series.append((t - self.first[0], mbps, pps))
        tm = st.timing
        if tm is not None:
            if math.isfinite(tm.recent_max_us):
                self.max_gap_us = max(self.max_gap_us, tm.recent_max_us)
            if math.isfinite(tm.pdv_us):
                self.pdv_us = tm.pdv_us if not math.isfinite(self.pdv_us) else max(self.pdv_us, tm.pdv_us)
        self.last = (t, st)

    def report(self, thresholds: Thresholds | None = None) -> PassiveReport | None:
        if self.first is None or self.last is None or self.last is self.first:
            return None
        th = thresholds or self.thresholds
        (t0, a), (t1, b) = self.first, self.last
        dur = max(t1 - t0, 1e-9)
        packets, nbytes, samples = b.packets - a.packets, b.bytes - a.bytes, b.samples - a.samples
        lost = b.lost_samples - a.lost_samples
        loss_pct = 100.0 * lost / (samples + lost) if samples + lost else math.nan
        ta, tb = a.timing, b.timing
        hist = np.zeros(len(TIMING_EDGES_US) + 1)
        mean = std = math.nan
        expected = rate = ppm = err = sender = math.nan
        if tb is not None:
            hist = np.asarray(tb.hist, dtype=np.float64) - (np.asarray(ta.hist, dtype=np.float64) if ta else 0.0)
            hist = np.maximum(hist, 0)
            mean, std = _window_moments(ta, tb)
            expected, rate, ppm, err, sender = tb.expected_us, tb.rate_hz, tb.rate_ppm, tb.rate_ppm_err, tb.sender_ppm
        rep = PassiveReport(
            b.name or b.key, f"{b.source}:{b.port}" if b.port else b.source, b.format, dur, packets, nbytes, samples,
            nbytes * 8 / dur / 1e6, max(self.peak_mbps, nbytes * 8 / dur / 1e6), packets / dur,
            max(self.peak_pps, packets / dur), lost, loss_pct, b.loss_events - a.loss_events,
            b.max_loss_burst if b.loss_events > a.loss_events else 0,
            b.duplicates - a.duplicates, b.late_packets - a.late_packets, b.restarts - a.restarts, expected, mean, std,
            hist_percentile(hist, 50), hist_percentile(hist, 95), hist_percentile(hist, 99),
            self.max_gap_us if self.max_gap_us else math.nan, hist, rate, b.sample_rate, ppm, err, self.pdv_us,
            sender, getattr(b, "collisions", 0) - getattr(a, "collisions", 0), tuple(self.conflicts.values()))
        rep.checks = evaluate(rep, th)
        return rep


def _window_moments(a, b) -> tuple[float, float]:
    """Birikimli (n, ortalama, std) değerlerinden iki an arasındaki paketlerin ortalaması ve std'si."""
    n1, m1 = b.intervals, b.mean_us
    s1 = b.std_us if math.isfinite(b.std_us) else 0.0
    n0, m0, s0 = (a.intervals, a.mean_us, a.std_us) if a is not None else (0, 0.0, 0.0)
    if not math.isfinite(m0):
        m0 = 0.0
    if not math.isfinite(s0):
        s0 = 0.0
    nw = n1 - n0
    if nw <= 0:
        return math.nan, math.nan
    sum_w = n1 * m1 - n0 * m0
    sq1 = s1 * s1 * max(n1 - 1, 0) + n1 * m1 * m1                # Σx² = M2 + n·ortalama²
    sq0 = s0 * s0 * max(n0 - 1, 0) + n0 * m0 * m0
    mean = sum_w / nw
    var = (sq1 - sq0 - nw * mean * mean) / (nw - 1) if nw > 1 else math.nan
    return mean, math.sqrt(var) if math.isfinite(var) and var > 0 else (0.0 if nw > 1 else math.nan)


def evaluate(rep: PassiveReport, th: Thresholds) -> list[Check]:
    checks = [Check("Kaynak çakışması (aynı stream_id iki gönderici)",
                    "; ".join(rep.conflicts) if rep.conflicts else "yok", "yok",
                    not rep.conflicts and not rep.collisions)]

    def add(name: str, value: float, limit: float, text: str, limit_text: str, ok_when_below: bool = True):
        if not limit:
            return
        ok = None if not math.isfinite(value) else (value <= limit if ok_when_below else value >= limit)
        checks.append(Check(name, text, limit_text, ok))

    add("En büyük boşluk", rep.max_us / 1e3, th.max_gap_ms, fmt_us(rep.max_us), f"≤ {th.max_gap_ms:g} ms")
    add("Kayıp oranı", rep.loss_pct, th.loss_pct,
        f"{rep.loss_pct:.4f} %" if math.isfinite(rep.loss_pct) else "—", f"≤ {th.loss_pct:g} %")
    jitter = rep.p99_us - rep.expected_us if math.isfinite(rep.expected_us) else rep.p99_us
    add("Titreşim (p99 − beklenen aralık)", jitter, th.jitter_p99_us, fmt_us(jitter), f"≤ {fmt_us(th.jitter_p99_us)}")
    add("Örnekleme hızı sapması", abs(rep.rate_ppm), th.rate_ppm, fmt_ppm(rep.rate_ppm, rep.rate_ppm_err),
        f"≤ ±{th.rate_ppm:g} ppm")
    add("Ortalama veri hızı", rep.mean_mbps, th.min_mbps, f"{rep.mean_mbps:.3f} Mbit/s", f"≥ {th.min_mbps:g} Mbit/s",
        ok_when_below=False)
    return checks
