"""Kontrol cevap grafikleri: yeniden kullanılabilir kart bileşenleri (Analiz, Modeller ve Kontrolcü Tasarımı sekmeleri).

Her grafik kendi kartındadır (`Card`); Bode'de genlik ve faz ayrı kartlardır. Hesaplar `spt.control` kütüphanesinden
yapılır (adım / darbe / lsim, bode, nyquist, nichols, sigma, rlocus, pole / zero, damp, margin, stepinfo); bu modül
yalnız çizer. Kütüphane fonksiyonlarına erişim tek yerde (`calc_*` uyarlayıcıları) toplanmıştır.

Renk: her seri (sistem) tek renk taşır; renkleri çağıran verir (`PALETTE`'ten, görünüm içinde benzersiz). Kılavuzlar
(−1 noktası, birim çember, sanal eksen, Nyquist frekansı) seri renklerinden ayrışan nötr gri ile çizilir.

PALETTE ölçümü (OKLab, #121110 kart yüzeyinde): 8 rengin her çifti normal görüşte ΔE ≥ 19.1, döteranopi /
protanopi simülasyonunda (Machado 2009) ΔE ≥ 9.8; yüzeye kontrast ≥ 5.1:1; kılavuz grisinden (#8a837c) ΔE ≥ 13.
İlk altısı Sensör Füzyonu paletidir; son ikisi bu ölçütle aranarak eklendi.

Veri ipuçları (tıkla): adımda t / y; Bode'de ω (rad/s ve Hz), genlik (dB ve mutlak), faz; Nyquist'te ω ve L(jω);
kök yer eğrisinde kazanç k, kutup, sönüm oranı, doğal frekans ve aşım; kutup-sıfırda değer, ζ, ωn.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QSplitter, QVBoxLayout, QWidget

from spt.control.errors import ControlError
from spt.gui.theme import ThemeManager
from spt.gui.widgets.common import Card, FlowLayout, label, legend_chip
from spt.gui.widgets.inspector import DataTips, PlotAnalysis, Trace, curve_stats, fmt, visible_slice
from spt.gui.widgets.plot import make_plot, pen, set_si_prefix

log = logging.getLogger("spt.gui")

PALETTE = ("#f54748", "#488bfb", "#45f27f", "#dfa11a", "#e03fc7", "#4cd1ee", "#03a27f", "#e3fafe")
GUIDE = "#8a837c"          # kılavuz çizgileri (seri değildir)
CRITICAL = "#fffdfb"       # −1 noktası / (−180°, 0 dB) işareti: metin rengi, seri renklerinden ayrı
CHANNEL_STYLES = (Qt.PenStyle.SolidLine, Qt.PenStyle.DashLine, Qt.PenStyle.DotLine, Qt.PenStyle.DashDotLine)
MAX_SERIES = len(PALETTE)
CARD_MIN_HEIGHT = 170


@dataclass
class Series:
    """Çizilecek bir sistem: ad (lejant), LTI model, renk."""
    name: str
    sys: object
    color: str
    key: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.key:
            self.key = self.name


# ============================================================================ hesap uyarlayıcıları
def _timeresp():
    from spt.control import timeresp
    return timeresp


def _freqresp():
    from spt.control import freqresp
    return freqresp


def _analysis():
    from spt.control import analysis
    return analysis


def calc_step(sys, tfinal: float | None = None, kind: str = "step") -> tuple[np.ndarray, np.ndarray]:
    """(t (nt,), y (nt, ny, nu)) — adım ya da darbe cevabı."""
    fn = getattr(_timeresp(), "impulse" if kind == "impulse" else "step")
    y, t = fn(sys, tfinal) if tfinal is not None else fn(sys)
    return np.asarray(t, float), np.asarray(y, float).reshape(len(t), sys.noutputs, sys.ninputs)


def calc_lsim(sys, u: np.ndarray, t: np.ndarray) -> np.ndarray:
    """lsim çıkışı (nt, ny)."""
    y, _t = _timeresp().lsim(sys, u, t)[:2]
    return np.asarray(y, float).reshape(len(t), -1)


def calc_stepinfo(sys) -> dict:
    return _timeresp().stepinfo(sys)


def calc_bode(sys, w=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(w, mag (ny, nu, nw) mutlak, phase (ny, nu, nw) derece)."""
    mag, phase, w = _freqresp().bode(sys) if w is None else _freqresp().bode(sys, w)
    return np.asarray(w, float), np.asarray(mag, float), np.asarray(phase, float)


def calc_freqresp(sys, w) -> np.ndarray:
    """Karmaşık (ny, nu, nw)."""
    return np.asarray(_freqresp().freqresp(sys, np.atleast_1d(np.asarray(w, float))), complex)


def calc_nyquist(sys) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(w, re, im) — re / im (ny, nu, nw)."""
    re, im, w = _freqresp().nyquist(sys)
    return np.asarray(w, float), np.asarray(re, float), np.asarray(im, float)


def calc_nichols(sys) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(w, mag mutlak, phase derece) — (ny, nu, nw)."""
    mag, phase, w = _freqresp().nichols(sys)
    return np.asarray(w, float), np.asarray(mag, float), np.asarray(phase, float)


def calc_sigma(sys) -> tuple[np.ndarray, np.ndarray]:
    """(w, sv (k, nw) mutlak)."""
    sv, w = _freqresp().sigma(sys)
    return np.asarray(w, float), np.asarray(sv, float).reshape(-1, len(np.atleast_1d(w)))


def calc_margin(sys) -> tuple[float, float, float, float]:
    """(Gm mutlak, Pm derece, Wcg, Wcp) — payı olmayan için inf / nan (MATLAB margin)."""
    gm, pm, wcg, wcp = _freqresp().margin(sys)[:4]
    return float(gm), float(pm), float(wcg), float(wcp)


def calc_rlocus(sys) -> tuple[np.ndarray, np.ndarray]:
    """(r (nk, n) karmaşık kökler — her sütun bir kol, k (nk,)). Kütüphane MATLAB düzeninde (n, nk) döndürür."""
    r, k = _analysis().rlocus(sys)
    return np.asarray(r, complex).T, np.asarray(k, float)


def calc_poles(sys) -> np.ndarray:
    return np.asarray(_analysis().pole(sys), complex).ravel()


def calc_zeros(sys) -> np.ndarray:
    an = _analysis()
    fn = getattr(an, "tzero", None) or an.zero
    out = fn(sys)
    if isinstance(out, tuple):
        out = out[0]
    return np.asarray(out, complex).ravel()


def calc_damp(sys_or_p, Ts: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(wn, zeta, p) — MATLAB damp (ωn artan; ayrıkta s = log(z)/Ts üzerinden). Model ya da kök dizisi + Ts."""
    wn, zeta, p = _analysis().damp(sys_or_p, Ts)[:3] if Ts is not None else _analysis().damp(sys_or_p)[:3]
    return np.asarray(wn, float).ravel(), np.asarray(zeta, float).ravel(), np.asarray(p, complex).ravel()


def damp_of_root(root: complex, Ts: float) -> tuple[float, float]:
    """Tek kökün (ωn, ζ) değeri, kütüphanenin damp tanımıyla."""
    wn, zeta, _ = calc_damp(np.array([complex(root)]), Ts)
    return float(wn[0]), float(zeta[0])


def overshoot_of(zeta: float) -> float:
    """İkinci derece sistemin aşımı (%): 100·exp(−ζπ/√(1−ζ²)), 0 < ζ < 1 (MATLAB kök yer ipucu)."""
    if not 0.0 <= zeta < 1.0:
        return 0.0
    return 100.0 * math.exp(-zeta * math.pi / math.sqrt(1.0 - zeta * zeta))


def describe_root(root: complex, Ts: float) -> list[str]:
    """Kutup / kök ipucu satırları: değer, ζ, ωn, aşım (ayrıkta |z| ve eşdeğer s)."""
    var = "z" if Ts > 0 else "s"
    lines = [f"{var} = {root.real:+.6g} {'+' if root.imag >= 0 else '−'} j{abs(root.imag):.6g}"]
    if Ts > 0:
        lines.append(f"|z| = {abs(root):.6g}")
    try:
        wn, zeta = damp_of_root(root, Ts)
    except (ControlError, ImportError, ValueError, IndexError):
        return lines
    lines.append(f"sönüm ζ = {zeta:.4g}")
    lines.append(f"doğal frekans ωn = {wn:.5g} rad/s")
    if 0 < zeta < 1:
        lines.append(f"aşım ≈ %{overshoot_of(zeta):.3g}")
    return lines


def io_name(sys, i: int, j: int) -> str:
    """MIMO kanal etiketi (u → y); SISO'da boş."""
    if sys.ninputs == 1 and sys.noutputs == 1:
        return ""
    u = sys.input_names[j] if sys.input_names[j] else f"u{j + 1}"
    y = sys.output_names[i] if sys.output_names[i] else f"y{i + 1}"
    return f"{u} → {y}"


def _db(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return 20.0 * np.log10(np.abs(x))


def _finite(*arrays) -> np.ndarray:
    ok = np.ones(len(arrays[0]), bool)
    for a in arrays:
        ok &= np.isfinite(a)
    return ok


# ============================================================================ temel kart
class SeriesLegend(QWidget):
    """Seri lejantı (renk çizgisi + ad); dar alanda satır kaydırır."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._lay = FlowLayout(self, spacing=12)

    def set_series(self, items: list[tuple[str, str]]) -> None:
        while self._lay.count():
            w = self._lay.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        for color, name in items:
            self._lay.addWidget(legend_chip(color, name))
        self.setVisible(bool(items))
        self.updateGeometry()


class PlotCard(Card):
    """Tek grafikli kart: çizim öğelerini ve veri ipuçlarını yönetir. Alt sınıf `_draw(series)` uygular."""

    drawn = pyqtSignal()

    def __init__(self, theme: ThemeManager, title: str, x_label: str, y_label: str, x_units: str | None = None,
                 y_units: str | None = None, log_x: bool = False, si_prefix: bool = False,
                 parent: QWidget | None = None):
        super().__init__(title, parent)
        self._theme = theme
        self.setMinimumHeight(CARD_MIN_HEIGHT)
        self.plot = make_plot(theme, x_label, y_label, x_units, y_units, si_prefix=si_prefix)
        # Nyquist / Nichols / kök yer eğrisinde x tekdüze değil: tepe seyreltmesi ve görünüme kırpma x'in sıralı
        # olduğunu varsayar ve eğriyi bozar. Cevap eğrileri birkaç bin noktadır; ikisi de kapalı.
        self.plot.setDownsampling(auto=False, ds=1)
        self.plot.setClipToView(False)
        pi = self.plot.getPlotItem()
        if log_x:
            with np.errstate(over="ignore"):
                pi.setXRange(0.0, 1.0, padding=0)
                pi.setLogMode(x=True, y=False)
        for name in ("left", "bottom"):
            set_si_prefix(pi.getAxis(name), si_prefix)
        self.body.addWidget(self.plot, 1)
        self.note = label("", "Muted")
        self.note.setWordWrap(True)
        self.note.hide()
        self.body.addWidget(self.note)
        self._items: list = []
        self._traces: list[Trace] = []
        self.series: list[Series] = []
        self.errors: list[str] = []
        self.tips = DataTips(self.plot, theme, lambda: list(self._traces))

    # ---------------------------------------------------------------- öğeler
    @property
    def tokens(self):
        return self._theme.tokens

    def clear(self) -> None:
        self.tips.clear()
        for item in self._items:
            self.plot.removeItem(item)
        self._items.clear()
        self._traces.clear()
        self.series = []
        self.errors = []
        self.note.hide()

    def add_item(self, item, trace: Trace | None = None, bounds: bool = True):
        self.plot.addItem(item, ignoreBounds=not bounds)
        self._items.append(item)
        if trace is not None:
            self._traces.append(trace)
        return item

    def curve(self, x, y, color: str, name: str, describe=None, exact=None, style=Qt.PenStyle.SolidLine,
              width: float = 2.0, bounds: bool = True) -> pg.PlotDataItem:
        p = pen(color, width)
        p.setStyle(style)
        item = pg.PlotDataItem(np.asarray(x, float), np.asarray(y, float), pen=p, connect="finite")
        trace = Trace(item, name, color, describe, exact) if describe is not None else None
        return self.add_item(item, trace, bounds)

    def points(self, x, y, color: str, name: str, describe=None, symbol: str = "o", size: int = 10,
               filled: bool = True, bounds: bool = True) -> pg.PlotDataItem:
        """İşaret noktaları (çizgisiz PlotDataItem: logaritmik eksende de veri koordinatında kalır)."""
        brush = pg.mkBrush(color) if filled else pg.mkBrush(self.tokens.surface)
        item = pg.PlotDataItem(np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float)), pen=None,
                               symbol=symbol, symbolSize=size, symbolPen=pg.mkPen(color, width=2), symbolBrush=brush)
        item.setZValue(20)
        trace = Trace(item, name, color, describe, extra={"points": True}) if describe is not None else None
        return self.add_item(item, trace, bounds)

    def guide_line(self, pos: float, angle: int, color: str = GUIDE, style=Qt.PenStyle.DashLine,
                   text: str = "") -> pg.InfiniteLine:
        p = pg.mkPen(color, width=1, style=style)
        line = pg.InfiniteLine(pos, angle=angle, pen=p, movable=False)
        if text:
            line.label = pg.InfLineLabel(line, text, position=0.92, color=color, anchors=[(0, 1), (0, 1)])
        return self.add_item(line, bounds=False)

    # ---------------------------------------------------------------- çizim
    def set_series(self, series: list[Series], **opts) -> list[str]:
        """Serileri çizer; hesaplanamayanların iletilerini döndürür (kartın altında da gösterilir)."""
        self.clear()
        self.series = list(series)
        for s in self.series:
            try:
                self._draw(s, **opts)
            except ControlError as exc:
                self.errors.append(f"{s.name}: {exc}")
            except ImportError:
                self.errors.append(f"{s.name}: hesap modülü henüz kullanılamıyor")
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                log.warning("%s · %s çizilemedi: %s", self.title.text(), s.name, exc)
                self.errors.append(f"{s.name}: hesaplanamadı ({exc})")
        self._finish(**opts)
        self.note.setText("<br>".join(self.errors))
        self.note.setVisible(bool(self.errors))
        if not self.series:
            self.note.setText("Gösterilecek sistem yok")
            self.note.show()
        self.drawn.emit()
        return list(self.errors)

    def _draw(self, s: Series, **opts) -> None:  # pragma: no cover - soyut
        raise NotImplementedError

    def _finish(self, **opts) -> None:
        self.plot.getPlotItem().enableAutoRange()

    def trace_names(self) -> list[str]:
        return [t.name for t in self._traces]

    def curve_data(self, name: str) -> tuple[np.ndarray, np.ndarray] | None:
        """Adı verilen izin (x, y) verisi, gerçek koordinatta (testler ve tasarım sekmesi için)."""
        for t in self._traces:
            if t.name == name:
                return np.asarray(t.item.xData, float), np.asarray(t.item.yData, float)
        return None

    def line_traces(self) -> list[Trace]:
        """İşaret noktaları dışındaki eğriler."""
        return [t for t in self._traces if not t.extra.get("points")]


# ============================================================================ zaman cevabı
class StepPlot(PlotCard):
    """Adım / darbe cevabı (MIMO: her kanal aynı renkte, farklı çizgi stiliyle). Ortak bitiş zamanı: sistemlerin
    kendi önerdiği sürelerin en büyüğü. Karakteristik işaretleri (SISO adım): tepe (aşım), yükselme, yerleşme ve
    kalıcı durum (stepinfo, MATLAB tanımı)."""

    def __init__(self, theme: ThemeManager, title: str = "Adım Cevabı", kind: str = "step",
                 parent: QWidget | None = None):
        super().__init__(theme, title, "Zaman", "Genlik", x_units="s", si_prefix=False, parent=parent)
        self.kind = kind
        self.markers = {"peak", "settling", "rise", "steady"}
        self.analysis = PlotAnalysis(self, self.plot, theme, self._stats)
        self.info: dict[str, dict] = {}

    def set_series(self, series: list[Series], tfinal: float | None = None, **opts) -> list[str]:
        if tfinal is None:
            tfinal = self._common_tfinal(series)
        return super().set_series(series, tfinal=tfinal, **opts)

    def _common_tfinal(self, series: list[Series]) -> float | None:
        ends = []
        for s in series:
            try:
                t, _ = calc_step(s.sys, None, self.kind)
            except (ControlError, ImportError, ValueError, np.linalg.LinAlgError):
                continue
            if len(t):
                ends.append(float(t[-1]))
        return max(ends) if ends else None

    def _draw(self, s: Series, tfinal: float | None = None, **opts) -> None:
        sys = s.sys
        t, y = calc_step(sys, tfinal, self.kind)
        discrete = sys.Ts > 0
        for i in range(y.shape[1]):
            for j in range(y.shape[2]):
                io = io_name(sys, i, j)
                name = f"{s.name} ({io})" if io else s.name
                style = CHANNEL_STYLES[(i * y.shape[2] + j) % len(CHANNEL_STYLES)]
                yy = y[:, i, j]
                self._time_curve(t, yy, s.color, name, style, discrete, sys.Ts)
        if self.kind == "step" and sys.ninputs == 1 and sys.noutputs == 1:
            self._characteristics(s, t, y[:, 0, 0])

    def _time_curve(self, t, y, color, name, style, discrete: bool, Ts: float) -> None:
        def describe(x, v, discrete=discrete, Ts=Ts):
            lines = [f"t = {fmt(x, 's', 6)}"]
            if discrete:
                lines[0] += f"  (k = {int(round(x / Ts))})"
            lines.append(f"y = {v:.6g}")
            return lines

        def exact(x, t=t, y=y):  # basamak çiziminde köşe yerine örneğin kendisi
            k = int(np.clip(np.searchsorted(t, x), 0, len(t) - 1))
            if k > 0 and abs(t[k - 1] - x) < abs(t[k] - x):
                k -= 1
            return float(t[k]), float(y[k])

        if discrete and len(t) > 1:  # MATLAB gibi basamak (sıfır dereceli tutma) çizimi
            xs = np.repeat(t, 2)[1:]
            ys = np.repeat(y, 2)[:-1]
            self.curve(xs, ys, color, name, describe, exact, style)
        else:
            self.curve(t, y, color, name, describe, exact, style)

    def _characteristics(self, s: Series, t: np.ndarray, y: np.ndarray) -> None:
        """stepinfo değerlerinin işaretleri; değerler kütüphaneden (MATLAB tanımı), konumlar çizilen eğriden."""
        try:
            info = calc_stepinfo(s.sys)
        except (ControlError, ImportError, ValueError):
            return
        self.info[s.key] = info
        c = s.color
        ts = float(info.get("SettlingTime", np.nan))
        settles = np.isfinite(ts)
        try:
            final = float(np.ravel(_freqresp().dcgain(s.sys))[0]) if settles else np.nan
        except (ControlError, ImportError, ValueError, AttributeError):
            final = np.nan
        if "steady" in self.markers and np.isfinite(final):
            self.guide_line(final, 0, c, Qt.PenStyle.DotLine)
            self.points([t[-1]], [final], c, f"{s.name} · kalıcı durum",
                        lambda x, v: [f"kalıcı durum (DC kazancı) = {v:.6g}"], symbol="t1", size=9)
        discrete = s.sys.Ts > 0

        def y_at(x: float) -> float:
            """Çizilen eğrinin x anındaki değeri: ayrıkta örnek (basamak), süreklide örnekler arası doğrusal."""
            if discrete:
                return float(y[int(np.argmin(np.abs(t - x)))])
            return float(np.interp(x, t, y))

        peak_t, peak = float(info.get("PeakTime", np.nan)), float(info.get("Peak", np.nan))
        if "peak" in self.markers and np.isfinite(peak_t) and np.isfinite(peak) and 0 < peak_t <= t[-1] and peak > 0:
            os_ = float(info.get("Overshoot", np.nan))

            def describe_peak(x, v, os_=os_, pk=peak, pt=peak_t):
                return [f"tepe = {pk:.6g}", f"tepe zamanı = {fmt(pt, 's', 6)}", f"aşım = %{os_:.4g}"]

            # Peak = max|y − yinit| (yinit = 0, MATLAB): işaret cevabın o andaki işaretiyle tepe değerinde
            self.points([peak_t], [math.copysign(peak, y_at(peak_t))], c, f"{s.name} · tepe", describe_peak,
                        symbol="o", size=10)
        if "settling" in self.markers and settles and 0 < ts <= t[-1]:
            self.points([ts], [y_at(ts)], c, f"{s.name} · yerleşme",
                        lambda x, v, ts=ts: [f"yerleşme süresi = {fmt(ts, 's', 6)}", f"y = {v:.6g}"], symbol="s",
                        size=9)
        tr = float(info.get("RiseTime", np.nan))
        if "rise" in self.markers and np.isfinite(tr) and tr > 0 and np.isfinite(final) and final != 0:
            # yükselme süresi %10 → %90 aralığıdır (stepinfo, yinit = 0); işaret %90'a ilk ulaşılan örnekte
            k = np.flatnonzero(y / final >= 0.9)
            if len(k):
                self.points([t[k[0]]], [y[k[0]]], c, f"{s.name} · yükselme",
                            lambda x, v, tr=tr: [f"yükselme süresi (%10→%90) = {fmt(tr, 's', 6)}",
                                                 f"%90'a ulaşma t = {fmt(x, 's', 6)}"], symbol="d", size=10)

    def _stats(self, x0: float, x1: float) -> list:
        out = []
        for tr in self.line_traces():
            x, y = tr.item.xData, tr.item.yData
            if x is None or not len(x):
                continue
            sl = visible_slice(x, x0, x1)
            out.append((tr.name, tr.color, curve_stats(x[sl], y[sl], "s", "")))
        return out


class LsimPlot(StepPlot):
    """Gerçek ya da üretilmiş giriş sinyaliyle benzetim (lsim); giriş gri, ölçülen çıkış (varsa) ayrı seri."""

    def __init__(self, theme: ThemeManager, title: str = "Benzetim (lsim)", parent: QWidget | None = None):
        super().__init__(theme, title, kind="lsim", parent=parent)
        self.t = np.zeros(0)
        self.u = np.zeros(0)
        self.measured: tuple[str, str, np.ndarray] | None = None   # (ad, renk, y)
        self.input_name = "giriş u"

    def set_input(self, t: np.ndarray, u: np.ndarray, name: str = "giriş u",
                  measured: tuple[str, str, np.ndarray] | None = None) -> None:
        self.t = np.asarray(t, float)
        self.u = np.asarray(u, float)
        self.input_name = name
        self.measured = measured

    def set_series(self, series: list[Series], **opts) -> list[str]:
        return PlotCard.set_series(self, series, **opts)

    def _draw(self, s: Series, **opts) -> None:
        if len(self.t) < 2:
            raise ControlError("Benzetim için giriş sinyali yok")
        sys = s.sys
        if sys.ninputs != 1:
            raise ControlError("Benzetim tek girişli sistem için (MIMO'da kanal seçin)")
        t, u = self.t, self.u
        if sys.Ts > 0:  # ayrık sistem: giriş Ts aralığında örneklenmeli
            dt = np.diff(t)
            if len(dt) and np.max(np.abs(dt - sys.Ts)) > 1e-6 * sys.Ts:
                raise ControlError(f"Giriş örnekleme aralığı ({dt[0]:.6g} s) ayrık sistemin Ts değerine "
                                   f"({sys.Ts:g} s) eşit değil")
        y = calc_lsim(sys, u, t)
        for i in range(y.shape[1]):
            io = sys.output_names[i] if y.shape[1] > 1 else ""
            self._time_curve(t, y[:, i], s.color, f"{s.name} ({io})" if io else s.name,
                             CHANNEL_STYLES[i % len(CHANNEL_STYLES)], sys.Ts > 0, sys.Ts)

    def _finish(self, **opts) -> None:
        if len(self.t) > 1:
            def describe(x, v):
                return [f"t = {fmt(x, 's', 6)}", f"u = {v:.6g}"]
            self.curve(self.t, self.u, GUIDE, self.input_name, describe, width=1.5)
            if self.measured is not None:
                name, color, y = self.measured
                self.curve(self.t, y, color, name, lambda x, v: [f"t = {fmt(x, 's', 6)}", f"y = {v:.6g}"],
                           style=Qt.PenStyle.SolidLine, width=1.5)
        super()._finish(**opts)


# ============================================================================ frekans cevabı
class _LogFreqCard(PlotCard):
    def __init__(self, theme, title, y_label, y_units=None, parent=None):
        super().__init__(theme, title, "Frekans", y_label, x_units="rad/s", y_units=y_units, log_x=True,
                         parent=parent)


class BodePlot(QWidget):
    """Bode diyagramı: genlik (dB) ve faz (°) ayrı kartlarda, frekans eksenleri bağlı (rad/s, logaritmik).
    Ayrık sistemlerde Nyquist frekansı (π/Ts) kılavuz çizgisiyle; `margins` verilen serilerde kazanç ve faz payı
    işaretleri (MATLAB margin grafiği gibi)."""

    drawn = pyqtSignal()

    def __init__(self, theme: ThemeManager, parent: QWidget | None = None):
        super().__init__(parent)
        self._theme = theme
        self.mag_card = _LogFreqCard(theme, "Bode · Genlik", "Genlik", "dB")
        self.phase_card = _LogFreqCard(theme, "Bode · Faz", "Faz", "°")
        self.phase_card.plot.setXLink(self.mag_card.plot)
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(self.mag_card)
        self.splitter.addWidget(self.phase_card)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.splitter)
        self.errors: list[str] = []
        self._data: dict[str, tuple] = {}

    @property
    def cards(self) -> tuple[Card, Card]:
        return self.mag_card, self.phase_card

    def release_cards(self) -> tuple[Card, Card]:
        """Kartları kendi bölücüsünden çıkarır: çağıran ayrı ayrı yerleştirir (ör. grafik ızgarasında iki hücre).
        Bu bileşen hesap ve çizimi yönetmeye devam eder; kendisi gösterilmez."""
        for card in self.cards:
            card.setParent(None)
        self.hide()
        return self.cards

    def clear(self) -> None:
        self.mag_card.clear()
        self.phase_card.clear()
        self._data.clear()

    def set_series(self, series: list[Series], margins: tuple[str, ...] = ()) -> list[str]:
        self.clear()
        mc, pc = self.mag_card, self.phase_card
        mc.series = pc.series = list(series)
        errors = []
        nyq = set()
        for s in series:
            try:
                self._draw(s, s.key in margins)
                if s.sys.Ts > 0:
                    nyq.add(math.pi / s.sys.Ts)
            except ControlError as exc:
                errors.append(f"{s.name}: {exc}")
            except ImportError:
                errors.append(f"{s.name}: hesap modülü henüz kullanılamıyor")
            except (ValueError, np.linalg.LinAlgError) as exc:
                errors.append(f"{s.name}: hesaplanamadı ({exc})")
        for w in sorted(nyq):
            for card in (mc, pc):
                card.guide_line(math.log10(w), 90, GUIDE, Qt.PenStyle.DotLine, "π/Ts" if card is mc else "")
        for card in (mc, pc):
            card.errors = errors
            card.note.setText("<br>".join(errors))
            card.note.setVisible(bool(errors) and card is mc)
            card.plot.getPlotItem().enableAutoRange()
        self.errors = errors
        self.drawn.emit()
        return errors

    def _draw(self, s: Series, with_margins: bool) -> None:
        sys = s.sys
        w, mag, phase = calc_bode(sys)
        self._data[s.key] = (w, mag, phase)
        for i in range(mag.shape[0]):
            for j in range(mag.shape[1]):
                io = io_name(sys, i, j)
                name = f"{s.name} ({io})" if io else s.name
                style = CHANNEL_STYLES[(i * mag.shape[1] + j) % len(CHANNEL_STYLES)]
                db = _db(mag[i, j])

                def exact_resp(x, sys=sys, i=i, j=j):
                    if not (x > 0 and np.isfinite(x)):
                        return None
                    h = calc_freqresp(sys, [x])[i, j, 0]
                    return h

                def describe(x, _v, i=i, j=j, exact_resp=exact_resp):
                    lines = [f"ω = {x:.6g} rad/s  (f = {fmt(x / (2 * math.pi), 'Hz', 6)})"]
                    try:
                        h = exact_resp(x)
                    except (ControlError, ValueError, ImportError):
                        h = None
                    if h is not None:
                        lines.append(f"genlik = {20 * math.log10(max(abs(h), 1e-300)):.4f} dB  (|G| = {abs(h):.6g})")
                        lines.append(f"faz = {math.degrees(np.angle(h)):.3f}° (ana değer)")
                    return lines

                def exact_mag(x, exact_resp=exact_resp):
                    try:
                        h = exact_resp(x)
                    except (ControlError, ValueError, ImportError):
                        return None
                    return None if h is None or abs(h) == 0 else (x, 20 * math.log10(abs(h)))

                def phase_at(x, w=w, ph=phase[i, j]):  # çizilen (sarılmamış) fazla tutarlı: ızgarada aradeğer
                    return (x, float(np.interp(math.log10(x), np.log10(w), ph))) if x > 0 else None

                def describe_phase(x, v, describe=describe):
                    lines = describe(x, v)
                    lines.append(f"çizilen faz = {v:.3f}°")
                    return lines

                ok = _finite(w, db) & (w > 0)
                self.mag_card.curve(w[ok], db[ok], s.color, name, describe, exact_mag, style)
                okp = _finite(w, phase[i, j]) & (w > 0)
                self.phase_card.curve(w[okp], phase[i, j][okp], s.color, name, describe_phase, phase_at, style)
        if with_margins and sys.ninputs == 1 and sys.noutputs == 1:
            self._margins(s, w, phase[0, 0])

    def _margins(self, s: Series, w: np.ndarray, phase: np.ndarray) -> None:
        try:
            gm, pm, wcg, wcp = calc_margin(s.sys)
        except (ControlError, ImportError, ValueError):
            return
        c = s.color
        mc, pc = self.mag_card, self.phase_card
        if np.isfinite(gm) and np.isfinite(wcg) and wcg > 0 and gm > 0:
            gdb = 20 * math.log10(gm)
            x = math.log10(wcg)
            mc.curve([wcg, wcg], [0.0, -gdb], c, f"{s.name} · kazanç payı",
                     lambda *_a, gdb=gdb, wcg=wcg: [f"kazanç payı = {gdb:.4f} dB ({gm:.6g})",
                                                    f"faz geçiş frekansı = {wcg:.6g} rad/s"],
                     style=Qt.PenStyle.DashLine, width=1.5, bounds=False)
            mc.points([wcg], [-gdb], c, f"{s.name} · Gm", None, symbol="o", size=8)
            ph180 = float(np.interp(x, np.log10(w), phase)) if len(w) else -180.0
            pc.points([wcg], [ph180], c, f"{s.name} · −180° geçişi",
                      lambda *_a: [f"faz geçişi ω = {wcg:.6g} rad/s"], symbol="o", size=8)
        if np.isfinite(pm) and np.isfinite(wcp) and wcp > 0:
            ph = float(np.interp(math.log10(wcp), np.log10(w), phase)) if len(w) else pm - 180.0
            base = ph - pm   # aynı 360° dalındaki −180° (+k·360°) çizgisi
            pc.curve([wcp, wcp], [base, ph], c, f"{s.name} · faz payı",
                     lambda *_a, pm=pm, wcp=wcp: [f"faz payı = {pm:.4f}°", f"kazanç geçiş frekansı = {wcp:.6g} rad/s"],
                     style=Qt.PenStyle.DashLine, width=1.5, bounds=False)
            pc.guide_line(base, 0, GUIDE, Qt.PenStyle.DotLine)
            mc.points([wcp], [0.0], c, f"{s.name} · 0 dB geçişi",
                      lambda *_a: [f"kazanç geçişi ω = {wcp:.6g} rad/s"], symbol="o", size=8)
        mc.guide_line(0.0, 0, GUIDE, Qt.PenStyle.DotLine)

    def data(self, key: str):
        return self._data.get(key)


class NyquistPlot(PlotCard):
    """Nyquist diyagramı: pozitif frekanslar düz, negatif frekanslar (ayna) kesikli; −1 noktası (+) ve birim çember.
    Görünüm, sonsuza giden kollar (integratör) yüzünden ezilmesin diye −1 çevresine göre sınırlanır; çift tık tümü."""

    def __init__(self, theme: ThemeManager, parent: QWidget | None = None):
        super().__init__(theme, "Nyquist", "Gerçel", "Sanal", parent=parent)
        self._extent = 1.5

    def clear(self) -> None:
        super().clear()
        self._extent = 1.5

    def _draw(self, s: Series, **opts) -> None:
        sys = s.sys
        w, re, im = calc_nyquist(sys)
        for i in range(re.shape[0]):
            for j in range(re.shape[1]):
                io = io_name(sys, i, j)
                name = f"{s.name} ({io})" if io else s.name
                x, y = re[i, j], im[i, j]
                ok = _finite(x, y)
                ww, xs, ys = w[ok], x[ok], y[ok]

                def describe(px, py, ww=ww, xs=xs, ys=ys, sign=1.0):
                    k = int(np.argmin(np.hypot(xs - px, sign * ys - py))) if len(xs) else -1
                    lines = [f"ω = {sign * ww[k]:.6g} rad/s"] if k >= 0 else []
                    return lines + [f"L = {px:+.6g} {'+' if py >= 0 else '−'} j{abs(py):.6g}",
                                    f"|L| = {math.hypot(px, py):.6g}  ∠ {math.degrees(math.atan2(py, px)):.3f}°",
                                    f"−1 noktasına uzaklık = {math.hypot(px + 1, py):.6g}"]

                def describe_neg(px, py, describe=describe):
                    return describe(px, py, sign=-1.0)

                self.curve(xs, ys, s.color, name, describe, None, CHANNEL_STYLES[0])
                self.curve(xs, -ys, s.color, f"{name} (ω < 0)", describe_neg, None, Qt.PenStyle.DashLine,
                           width=1.5)
                r = np.hypot(xs, ys)
                if len(r):  # sonsuza giden kol (integratör) görünümü ezmesin: genliklerin ortancası çevresi
                    self._extent = max(self._extent, min(float(np.percentile(r, 60)) * 1.5, 1e6))

    def _finish(self, **opts) -> None:
        theta = np.linspace(0, 2 * np.pi, 361)
        p = pg.mkPen(GUIDE, width=1, style=Qt.PenStyle.DashLine)
        circle = pg.PlotDataItem(np.cos(theta), np.sin(theta), pen=p)
        self.add_item(circle, bounds=False)
        self.guide_line(0.0, 0)
        self.guide_line(0.0, 90)
        self.points([-1.0], [0.0], CRITICAL, "−1 noktası", lambda *_: ["kritik nokta −1 + j0"], symbol="+", size=16)
        e = self._extent
        self.plot.getPlotItem().setRange(xRange=(-e - 0.2, e + 0.2), yRange=(-e, e), padding=0.05)


class NicholsPlot(PlotCard):
    """Nichols diyagramı: açık çevrim faz (°) – genlik (dB); kritik nokta (−180°, 0 dB)."""

    def __init__(self, theme: ThemeManager, parent: QWidget | None = None):
        super().__init__(theme, "Nichols", "Açık çevrim fazı", "Açık çevrim genliği", x_units="°", y_units="dB",
                         parent=parent)

    def _draw(self, s: Series, **opts) -> None:
        sys = s.sys
        w, mag, phase = calc_nichols(sys)
        for i in range(mag.shape[0]):
            for j in range(mag.shape[1]):
                io = io_name(sys, i, j)
                name = f"{s.name} ({io})" if io else s.name
                db = _db(mag[i, j])
                ph = phase[i, j]
                ok = _finite(db, ph)

                def describe(px, py, ww=w[ok], xs=ph[ok], ys=db[ok]):
                    k = int(np.argmin(np.hypot(xs - px, ys - py))) if len(xs) else 0
                    return [f"ω = {ww[k]:.6g} rad/s" if len(ww) else "", f"faz = {px:.3f}°", f"genlik = {py:.4f} dB"]

                self.curve(ph[ok], db[ok], s.color, name, describe, None,
                           CHANNEL_STYLES[(i * mag.shape[1] + j) % len(CHANNEL_STYLES)])

    def _finish(self, **opts) -> None:
        self.guide_line(0.0, 0, GUIDE, Qt.PenStyle.DotLine)
        self.guide_line(-180.0, 90, GUIDE, Qt.PenStyle.DotLine)
        self.points([-180.0], [0.0], CRITICAL, "kritik nokta", lambda *_: ["kritik nokta (−180°, 0 dB)"],
                    symbol="+", size=16, bounds=False)
        super()._finish(**opts)


class SigmaPlot(_LogFreqCard):
    """Tekil değer diyagramı (MATLAB sigma): σ_i(G(jω)) dB; SISO'da |G| ile aynıdır."""

    def __init__(self, theme: ThemeManager, parent: QWidget | None = None):
        super().__init__(theme, "Tekil Değerler", "Tekil değer", "dB", parent=parent)

    def _draw(self, s: Series, **opts) -> None:
        w, sv = calc_sigma(s.sys)
        for k in range(sv.shape[0]):
            db = _db(sv[k])
            ok = _finite(w, db) & (w > 0)
            name = f"{s.name} (σ{k + 1})" if sv.shape[0] > 1 else s.name

            def describe(x, v):
                return [f"ω = {x:.6g} rad/s  (f = {fmt(x / (2 * math.pi), 'Hz', 6)})", f"σ = {v:.4f} dB "
                        f"({10 ** (v / 20):.6g})"]

            self.curve(w[ok], db[ok], s.color, name, describe, None, CHANNEL_STYLES[k % len(CHANNEL_STYLES)])


# ============================================================================ s / z düzlemi
class _PlanePlot(PlotCard):
    def __init__(self, theme, title, parent=None):
        super().__init__(theme, title, "Gerçel", "Sanal", parent=parent)
        self.plot.setAspectLocked(True)
        self._discrete = False
        self._pts: list[complex] = []

    def clear(self) -> None:
        super().clear()
        self._discrete = False
        self._pts = []

    def _pole_zero(self, s: Series, poles: np.ndarray, zeros: np.ndarray, size: int = 12) -> None:
        Ts = s.sys.Ts
        self._discrete |= Ts > 0
        if len(poles):
            self.points(poles.real, poles.imag, s.color, f"{s.name} · kutup",
                        lambda x, y, Ts=Ts: ["kutup"] + describe_root(complex(x, y), Ts), symbol="x", size=size)
        if len(zeros):
            self.points(zeros.real, zeros.imag, s.color, f"{s.name} · sıfır",
                        lambda x, y, Ts=Ts: ["sıfır"] + describe_root(complex(x, y), Ts), symbol="o", size=size - 1,
                        filled=False)
        self._pts += list(poles) + list(zeros)

    def _finish(self, **opts) -> None:
        if self._discrete:
            theta = np.linspace(0, 2 * np.pi, 361)
            circle = pg.PlotDataItem(np.cos(theta), np.sin(theta), pen=pg.mkPen(GUIDE, width=1,
                                                                                 style=Qt.PenStyle.DashLine))
            self.add_item(circle, bounds=False)
        self.guide_line(0.0, 0)
        self.guide_line(0.0, 90)
        pi = self.plot.getPlotItem()
        pi.enableAutoRange()
        if not self.line_traces():  # yalnız noktalar: simetrik, birim çemberi (ayrıkta) içeren görünüm
            pts = np.asarray([p for p in self._pts if np.isfinite(p)], complex)
            e = 1.2 if self._discrete or not len(pts) else 0.0
            if len(pts):
                e = max(e, float(np.max(np.abs(pts))) * 1.15, 1e-3)
            pi.setRange(xRange=(-e, e), yRange=(-e, e), padding=0.05)


class PZMapPlot(_PlanePlot):
    """Kutup-sıfır haritası (MATLAB pzmap): kutup ×, sıfır ○; ayrık sistemde birim çember."""

    def __init__(self, theme: ThemeManager, parent: QWidget | None = None):
        super().__init__(theme, "Kutup-Sıfır", parent)

    def _draw(self, s: Series, **opts) -> None:
        self._pole_zero(s, calc_poles(s.sys), calc_zeros(s.sys))


class RootLocusPlot(_PlanePlot):
    """Kök yer eğrisi (MATLAB rlocus): kollar seri renginde, açık çevrim kutupları × ve sıfırları ○; veri ipucu
    kolun o noktasındaki kazancı (k), kutbu, sönüm oranını, doğal frekansı ve aşımı gösterir. `gain_at` tasarım
    sekmesinin sürükleme etkileşimi için en yakın kol noktasının kazancını verir."""

    def __init__(self, theme: ThemeManager, parent: QWidget | None = None):
        super().__init__(theme, "Kök Yer Eğrisi", parent)
        self._loci: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
        self._view = 0.0

    def clear(self) -> None:
        super().clear()
        self._loci = {}
        self._view = 0.0

    def _finish(self, **opts) -> None:
        super()._finish(**opts)
        if self._view > 0:
            e = self._view
            self.plot.getPlotItem().setRange(xRange=(-e, e), yRange=(-e, e), padding=0.05)

    def _draw(self, s: Series, **opts) -> None:
        sys = s.sys
        if sys.ninputs != 1 or sys.noutputs != 1:
            raise ControlError("Kök yer eğrisi tek girişli, tek çıkışlı açık çevrim için tanımlı")
        r, k = calc_rlocus(sys)
        self._loci[s.key] = (r, k, sys.Ts)
        Ts = sys.Ts
        for b in range(r.shape[1]):
            branch = r[:, b]
            ok = _finite(branch.real, branch.imag)

            def describe(x, y, key=s.key, Ts=Ts):
                gk, root = self.gain_at(x, y, key)
                return [f"kazanç k = {gk:.6g}"] + describe_root(root, Ts)

            self.curve(branch.real[ok], branch.imag[ok], s.color, s.name if b == 0 else f"{s.name} · kol {b + 1}",
                       describe, None, Qt.PenStyle.SolidLine, width=2)
        poles, zeros = calc_poles(sys), calc_zeros(sys)
        self._pole_zero(s, poles, zeros)
        # görünüm: açık çevrim kutup / sıfırlarının çevresi (kollar k → ∞'da sonsuza gider)
        pz = np.concatenate([poles, zeros])
        pz = pz[np.isfinite(pz)]
        scale = max(float(np.max(np.abs(pz))) if len(pz) else 1.0, 0.5)
        near = r[np.isfinite(r) & (np.abs(r) <= 4 * scale)]
        self._view = max(getattr(self, "_view", 0.0), 2.5 * scale,
                         float(np.max(np.abs(near))) * 1.1 if len(near) else 0.0)

    def gain_at(self, x: float, y: float, key: str | None = None) -> tuple[float, complex]:
        """(x, y) noktasına en yakın kol noktasının (k, kök) değeri (kollar arası doğrusal aradeğer yok: ızgara
        noktası). Seri verilmezse tüm serilerde aranır."""
        best = (math.inf, 0.0, complex(x, y))
        target = complex(x, y)
        for skey, (r, k, _Ts) in self._loci.items():
            if key is not None and skey != key:
                continue
            d = np.abs(r - target)
            d[~np.isfinite(d)] = np.inf
            idx = np.unravel_index(int(np.argmin(d)), d.shape)
            if d[idx] < best[0]:
                best = (float(d[idx]), float(k[idx[0]]), complex(r[idx]))
        return best[1], best[2]

    def loci(self, key: str):
        return self._loci.get(key)


PLOT_TYPES = {
    "step": ("Adım", StepPlot),
    "impulse": ("Darbe", StepPlot),
    "bode": ("Bode", BodePlot),
    "nyquist": ("Nyquist", NyquistPlot),
    "nichols": ("Nichols", NicholsPlot),
    "rlocus": ("Kök yer eğrisi", RootLocusPlot),
    "pzmap": ("Kutup-sıfır", PZMapPlot),
    "sigma": ("Tekil değer", SigmaPlot),
    "lsim": ("Benzetim (lsim)", LsimPlot),
}


def make_response_plot(kind: str, theme: ThemeManager) -> QWidget:
    """Grafik türü anahtarından bileşen: step / impulse / bode / nyquist / nichols / rlocus / pzmap / sigma / lsim."""
    if kind == "step":
        return StepPlot(theme)
    if kind == "impulse":
        return StepPlot(theme, "Darbe Cevabı", kind="impulse")
    title, cls = PLOT_TYPES[kind]
    return cls(theme)
