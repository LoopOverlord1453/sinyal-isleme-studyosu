"""Kontrolcü Tasarımı sekmesinin etkileşimli grafik bileşenleri ve arka plan hesap yürütücüsü.

- `BackgroundCall`: uzun hesapları (kök yer eğrisi, MPC benzetimi, otomatik PID) iş parçacığında çalıştırır; anahtar
  başına tek iş koşar, yeni istek gelirse sıraya yalnız en yenisi girer; eskiyen sonuç yok sayılır (nesil sayacı).
- `install_drag`: pyqtgraph görünüm kutusuna "tutamaçtan sürükle" davranışı ekler (tutamaç yoksa normal kaydırma).
- `RootLocusEditor`: kök yer eğrisi (hesap arka planda, kütüphanenin `rlocus`'u) + kapalı çevrim kutupları (sürükle →
  kazanç, `rlocfind`) + kompansatör kutup / sıfırları (sürükle → konum).
- `BodeEditor`: açık çevrim Bode (genlik eğrisini dikey sürükle → kazanç), kazanç / faz payı işaretleri.
- `CurvesCard`: hazır zaman dizilerini (MPC benzetimi, kestirim hatası) veri ipuçlarıyla çizen kart.
"""
from __future__ import annotations

import logging
import math
import threading
import time
import traceback

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDoubleSpinBox,
    QHeaderView,
    QLabel,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from spt.control.errors import ControlError
from spt.gui.control import control_plots as cp
from spt.gui.theme import ThemeManager
from spt.gui.widgets.common import label
from spt.gui.widgets.inspector import Trace, fmt

log = logging.getLogger("spt.gui")
PICK_PX = 14


# ============================================================================ arka plan hesapları
class BackgroundCall(QObject):
    """Anahtarlı arka plan işleri. `done(anahtar, sonuç)` / `failed(anahtar, ileti)` yalnız en yeni istek için."""

    done = pyqtSignal(str, object)
    failed = pyqtSignal(str, str)
    _relay = pyqtSignal(str, int, bool, object)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._gen: dict[str, int] = {}
        self._running: set[str] = set()
        self._queued: dict[str, tuple] = {}
        self._closed = False
        self._relay.connect(self._finish)

    @property
    def busy(self) -> bool:
        return bool(self._running or self._queued)

    def is_busy(self, key: str) -> bool:
        return key in self._running or key in self._queued

    def submit(self, key: str, fn, *args) -> None:
        if self._closed:
            return
        gen = self._gen.get(key, 0) + 1
        self._gen[key] = gen
        if key in self._running:
            self._queued[key] = (gen, fn, args)
            return
        self._start(key, gen, fn, args)

    def cancel(self, key: str) -> None:
        """Sonucu yok say (iş arka planda biter)."""
        self._gen[key] = self._gen.get(key, 0) + 1
        self._queued.pop(key, None)

    def _start(self, key, gen, fn, args) -> None:
        self._running.add(key)
        threading.Thread(target=self._run, args=(key, gen, fn, args), daemon=True,
                         name=f"spt-design-{key}").start()

    def _run(self, key, gen, fn, args) -> None:
        try:
            res, ok = fn(*args), True
        except ControlError as exc:
            res, ok = str(exc), False
        except Exception as exc:  # noqa: BLE001 - arka plan hatası arayüze ileti olarak gelir
            log.warning("Tasarım hesabı (%s) başarısız:\n%s", key, traceback.format_exc())
            res, ok = f"hesaplanamadı ({type(exc).__name__}: {exc})", False
        if not self._closed:
            self._relay.emit(key, gen, ok, res)

    def _finish(self, key: str, gen: int, ok: bool, res) -> None:
        self._running.discard(key)
        nxt = self._queued.pop(key, None)
        if nxt is not None and not self._closed:
            self._start(key, *nxt)
        if self._closed or gen != self._gen.get(key):
            return
        if ok:
            self.done.emit(key, res)
        else:
            self.failed.emit(key, str(res))

    def wait(self, timeout: float = 30.0) -> bool:
        """İşler bitene kadar olay döngüsünü döndürür (testler / kapanış için)."""
        app = QApplication.instance()
        end = time.perf_counter() + timeout
        while self.busy and time.perf_counter() < end:
            if app is not None:
                app.processEvents()
            time.sleep(0.005)
        if app is not None:
            app.processEvents()
        return not self.busy

    def shutdown(self) -> None:
        self._closed = True
        self._queued.clear()


# ============================================================================ sürükleme
def install_drag(plot: pg.PlotWidget, pick, move, finish) -> None:
    """Sol tuşla tutamaçtan sürükleme. pick(x, y, px) → tutamaç ya da None (x, y görünüm koordinatı, px: (x, y)
    yönünde bir pikselin görünüm birimi); move(tutamaç, x, y) ve finish(tutamaç, x, y). Tutamaç yoksa görünüm kutusunun
    kendi sürüklemesi (kaydırma) çalışır."""
    vb = plot.getPlotItem().getViewBox()
    original = vb.mouseDragEvent
    state = {"h": None}

    def handler(ev, axis=None):
        if ev.button() != Qt.MouseButton.LeftButton:
            return original(ev, axis)
        if ev.isStart():
            p = vb.mapSceneToView(ev.buttonDownScenePos())
            px = vb.viewPixelSize()
            h = pick(p.x(), p.y(), px)
            state["h"] = h
            if h is None:
                return original(ev, axis)
            ev.accept()
            return None
        if state["h"] is None:
            return original(ev, axis)
        ev.accept()
        p = vb.mapSceneToView(ev.scenePos())
        if ev.isFinish():
            h, state["h"] = state["h"], None
            finish(h, p.x(), p.y())
        else:
            move(state["h"], p.x(), p.y())
        return None

    vb.mouseDragEvent = handler


def _px_dist(dx: float, dy: float, px) -> float:
    sx, sy = (abs(px[0]) or 1e-300), (abs(px[1]) or 1e-300)
    return math.hypot(dx / sx, dy / sy)


# ============================================================================ kök yer eğrisi düzenleyicisi
class RootLocusEditor(cp.RootLocusPlot):
    """Kök yer eğrisi + sürüklenebilir kapalı çevrim kutupları ve kompansatör kökleri.

    `set_locus(sys, r, k, renk)`: arka planda hesaplanmış eğriyi çizer (kütüphanenin rlocus çıktısı).
    `set_markers(kapalı_kutuplar, kompansatör_kökleri, …)`: işaretleri günceller (eğriden bağımsız, anında).
    Sürükleme geri çağrıları: `on_gain(nokta, bitti)` ve `on_root(tutamaç, nokta, bitti)`."""

    def __init__(self, theme: ThemeManager, parent=None):
        super().__init__(theme, parent)
        self.title.setText("Kök Yer Eğrisi · düzenleyici")
        self.busy_label = label("", "Readout")
        self.add_right(self.busy_label, optional=True)
        self._pre: dict[str, tuple] = {}
        self.cl_poles = np.zeros(0, complex)
        self.comp_roots: list[dict] = []     # {"elem", "which", "j", "value", "movable"}
        self._cl_color = cp.PALETTE[0]
        self._comp_color = cp.PALETTE[4]
        self._Ts = 0.0
        self.on_gain = None
        self.on_root = None
        self._cl_item = pg.PlotDataItem([], [], pen=None, symbol="s", symbolSize=11,
                                        symbolPen=pg.mkPen(self._cl_color, width=2), symbolBrush=pg.mkBrush(
                                            self._cl_color))
        self._cz_item = pg.PlotDataItem([], [], pen=None, symbol="o", symbolSize=13,
                                        symbolPen=pg.mkPen(self._comp_color, width=2.5),
                                        symbolBrush=pg.mkBrush(theme.tokens.surface))
        self._cp_item = pg.PlotDataItem([], [], pen=None, symbol="x", symbolSize=15,
                                        symbolPen=pg.mkPen(self._comp_color, width=3),
                                        symbolBrush=pg.mkBrush(self._comp_color))
        for it in (self._cl_item, self._cz_item, self._cp_item):
            it.setZValue(40)
            self.plot.addItem(it, ignoreBounds=True)
        self._overlay_traces = [
            Trace(self._cl_item, "kapalı çevrim kutupları", self._cl_color,
                  lambda x, y: ["kapalı çevrim kutbu (sürükle → kazanç)"] + cp.describe_root(complex(x, y), self._Ts),
                  extra={"points": True}),
            Trace(self._cp_item, "kompansatör kutbu", self._comp_color,
                  lambda x, y: ["kompansatör kutbu (sürükle)"] + cp.describe_root(complex(x, y), self._Ts),
                  extra={"points": True}),
            Trace(self._cz_item, "kompansatör sıfırı", self._comp_color,
                  lambda x, y: ["kompansatör sıfırı (sürükle)"] + cp.describe_root(complex(x, y), self._Ts),
                  extra={"points": True}),
        ]
        self._traces += self._overlay_traces
        install_drag(self.plot, self._pick, self._move, self._finish_drag)
        self.plot.setToolTip("Kapalı çevrim kutbunu (■) sürükle: kazanç · kompansatör kutup (×) / sıfırını (○) "
                             "sürükle: konum · tıkla: veri ipucu")

    # ---------------------------------------------------------------- eğri
    def set_locus(self, sys, r: np.ndarray, k: np.ndarray, color: str, name: str = "L0 = C0·G") -> list[str]:
        self._pre = {"L0": (sys, np.asarray(r, complex).T, np.asarray(k, float))}
        return self.set_series([cp.Series(name, sys, color, "L0")])

    def clear(self) -> None:
        super().clear()
        if hasattr(self, "_overlay_traces"):
            self._traces += self._overlay_traces

    def _draw(self, s, **opts) -> None:
        """RootLocusPlot._draw'ın önceden hesaplanmış (arka plan) eğriyle sürümü."""
        sys = s.sys
        pre = self._pre.get(s.key)
        if pre is not None and pre[0] is sys:
            r, k = pre[1], pre[2]
        else:
            r, k = cp.calc_rlocus(sys)
        self._loci[s.key] = (r, k, sys.Ts)
        Ts = sys.Ts
        for b in range(r.shape[1]):
            branch = r[:, b]
            ok = np.isfinite(branch.real) & np.isfinite(branch.imag)

            def describe(x, y, key=s.key, Ts=Ts):
                gk, root = self.gain_at(x, y, key)
                return [f"kazanç |K| = {gk:.6g}"] + cp.describe_root(root, Ts)

            self.curve(branch.real[ok], branch.imag[ok], s.color, s.name if b == 0 else f"{s.name} · kol {b + 1}",
                       describe, None, Qt.PenStyle.SolidLine, width=2)
        poles, zeros = cp.calc_poles(sys), cp.calc_zeros(sys)
        self._pole_zero(s, poles, zeros)
        pz = np.concatenate([poles, zeros])
        pz = pz[np.isfinite(pz)]
        if Ts > 0:   # z düzlemi: birim çember ve çevresi (uzaktaki sıfırlar için görünüm kaydırılabilir)
            inner = np.abs(pz[np.abs(pz) <= 3.0])
            self._view = max(getattr(self, "_view", 0.0), 1.3, 1.15 * float(np.max(inner)) if inner.size else 0.0)
            return
        scale = max(float(np.max(np.abs(pz))) if len(pz) else 1.0, 0.5)
        near = r[np.isfinite(r) & (np.abs(r) <= 4 * scale)]
        self._view = max(getattr(self, "_view", 0.0), 2.5 * scale,
                         float(np.max(np.abs(near))) * 1.1 if len(near) else 0.0)

    # ---------------------------------------------------------------- işaretler
    def set_markers(self, cl_poles, comp_roots: list[dict], Ts: float, cl_color: str, comp_color: str) -> None:
        self._Ts = Ts
        self._discrete = Ts > 0
        self.cl_poles = np.asarray(cl_poles, complex).ravel()
        self.comp_roots = list(comp_roots)
        if cl_color != self._cl_color:
            self._cl_color = cl_color
            self._cl_item.setSymbolPen(pg.mkPen(cl_color, width=2))
            self._cl_item.setSymbolBrush(pg.mkBrush(cl_color))
            self._overlay_traces[0].color = cl_color
        if comp_color != self._comp_color:
            self._comp_color = comp_color
            self._cz_item.setSymbolPen(pg.mkPen(comp_color, width=2.5))
            self._cp_item.setSymbolPen(pg.mkPen(comp_color, width=3))
            self._cp_item.setSymbolBrush(pg.mkBrush(comp_color))
            for t in self._overlay_traces[1:]:
                t.color = comp_color
        p = self.cl_poles[np.isfinite(self.cl_poles)]
        self._cl_item.setData(p.real, p.imag)
        zs = [r["value"] for r in comp_roots if r["which"] == "z"]
        ps = [r["value"] for r in comp_roots if r["which"] == "p"]
        zs += [complex(v).conjugate() for v in zs if abs(complex(v).imag) > 0]
        ps += [complex(v).conjugate() for v in ps if abs(complex(v).imag) > 0]
        self._cz_item.setData(np.real(zs), np.imag(zs))
        self._cp_item.setData(np.real(ps), np.imag(ps))

    # ---------------------------------------------------------------- sürükleme
    def _pick(self, x: float, y: float, px):
        best = None
        for r in self.comp_roots:
            if not r["movable"]:
                continue
            v = complex(r["value"])
            for cand in (v, v.conjugate()):
                d = _px_dist(cand.real - x, cand.imag - y, px)
                if d <= PICK_PX and (best is None or d < best[0]):
                    best = (d, {"kind": "root", **r, "mirror": cand.imag < 0 < v.imag or False})
        for p in self.cl_poles:      # en yakın tutamaç kazanır (eşitlikte kompansatör kökü)
            if not np.isfinite(p):
                continue
            d = _px_dist(p.real - x, p.imag - y, px)
            if d <= PICK_PX and (best is None or d < best[0]):
                best = (d, {"kind": "gain"})
        return None if best is None else best[1]

    def _point(self, h: dict, x: float, y: float) -> complex:
        if h["kind"] == "root" and h.get("mirror"):
            y = -y
        return complex(x, y)

    def _move(self, h, x, y) -> None:
        self._dispatch(h, self._point(h, x, y), False)

    def _finish_drag(self, h, x, y) -> None:
        self._dispatch(h, self._point(h, x, y), True)

    def _dispatch(self, h: dict, point: complex, finished: bool) -> None:
        if h["kind"] == "gain":
            if self.on_gain is not None:
                self.on_gain(point, finished)
        elif self.on_root is not None:
            self.on_root(h, point, finished)

    def drag_gain_to(self, point: complex) -> None:
        """Kapalı çevrim kutbunu noktaya sürükle (testler ve klavyesiz kullanım)."""
        self._dispatch({"kind": "gain"}, complex(point), True)


# ============================================================================ Bode düzenleyicisi
class BodeEditor(cp.BodePlot):
    """Açık çevrim Bode diyagramı: genlik kartında L eğrisini dikey sürükle → kazanç (Δ dB = 20·log10(K'/K)).
    Paylar tasarım çekirdeğinin `loop_margins` değerleriyle işaretlenir (pay tablosuyla aynı değerler)."""

    def __init__(self, theme: ThemeManager, parent=None):
        super().__init__(theme, parent)
        self.mag_card.title.setText("Açık Çevrim Bode · Genlik (sürükle → kazanç)")
        self.phase_card.title.setText("Açık Çevrim Bode · Faz")
        self.on_gain = None            # on_gain(çarpan, bitti)
        self._key = None
        self._drag_preview = None
        install_drag(self.mag_card.plot, self._pick, self._move, self._finish_drag)

    def show_loop(self, series: list[cp.Series], key: str, margins: dict | None) -> list[str]:
        """Serileri çizer; `key` sürüklenebilir açık çevrim serisi, `margins` (Gm, Wcg, Pm, Wcp) işaretleri."""
        self._key = key
        errs = self.set_series(series, margins=())
        s = next((x for x in series if x.key == key), None)
        if s is not None and margins:
            self._mark(s, margins)
        return errs

    def _mark(self, s: cp.Series, m: dict) -> None:
        data = self.data(s.key)
        if data is None:
            return
        w, _mag, phase = data
        ph = phase[0, 0]
        c = s.color
        mc, pc = self.mag_card, self.phase_card
        gm, wcg, pm, wcp = m.get("Gm"), m.get("Wcg"), m.get("Pm"), m.get("Wcp")
        if gm is not None and np.isfinite(gm) and np.isfinite(wcg) and wcg > 0 and gm > 0:
            gdb = 20 * math.log10(gm)
            mc.curve([wcg, wcg], [0.0, -gdb], c, f"{s.name} · kazanç payı",
                     lambda *_a, gdb=gdb, wcg=wcg: [f"kazanç payı = {gdb:.4f} dB", f"faz geçişi ω = {wcg:.6g} rad/s"],
                     style=Qt.PenStyle.DashLine, width=1.5, bounds=False)
            mc.points([wcg], [-gdb], c, f"{s.name} · Gm", None, symbol="o", size=8)
        if pm is not None and np.isfinite(pm) and np.isfinite(wcp) and wcp > 0 and len(w):
            p_at = float(np.interp(math.log10(wcp), np.log10(w), ph))
            base = p_at - pm
            pc.curve([wcp, wcp], [base, p_at], c, f"{s.name} · faz payı",
                     lambda *_a, pm=pm, wcp=wcp: [f"faz payı = {pm:.4f}°", f"kazanç geçişi ω = {wcp:.6g} rad/s"],
                     style=Qt.PenStyle.DashLine, width=1.5, bounds=False)
            pc.guide_line(base, 0, cp.guide(), Qt.PenStyle.DotLine)
            mc.points([wcp], [0.0], c, f"{s.name} · 0 dB geçişi",
                      lambda *_a, wcp=wcp: [f"kazanç geçişi ω = {wcp:.6g} rad/s"], symbol="o", size=8)
        mc.guide_line(0.0, 0, cp.guide(), Qt.PenStyle.DotLine)

    def _curve_db_at(self, xv: float) -> float | None:
        data = self.data(self._key) if self._key else None
        if data is None:
            return None
        w, mag, _ = data
        m = np.abs(mag[0, 0])
        ok = (w > 0) & (m > 0) & np.isfinite(m)
        if not np.any(ok):
            return None
        lw = np.log10(w[ok])
        if not lw[0] <= xv <= lw[-1]:
            return None
        return float(np.interp(xv, lw, 20 * np.log10(m[ok])))

    def _pick(self, x, y, px):
        yc = self._curve_db_at(x)
        if yc is None or abs(yc - y) / (abs(px[1]) or 1e-300) > PICK_PX:
            return None
        return {"y0": y}

    def _move(self, h, x, y) -> None:
        if self.on_gain is not None:
            self.on_gain(10 ** ((y - h["y0"]) / 20.0), False, h)

    def _finish_drag(self, h, x, y) -> None:
        if self.on_gain is not None:
            self.on_gain(10 ** ((y - h["y0"]) / 20.0), True, h)


# ============================================================================ uzunluk sınırlı adım cevabı
MAX_DISCRETE_SAMPLES = 20000


def auto_tfinal(sys) -> float | None:
    """Kütüphanenin otomatik benzetim süresi (timeresp._auto_tfinal; yoksa adım cevabının son zamanı)."""
    try:
        from spt.control.timeresp import _auto_tfinal
        return float(_auto_tfinal(sys))
    except ImportError:
        pass
    except (ControlError, ValueError):
        return None
    try:
        t, _y = cp.calc_step(sys)
        return float(t[-1]) if len(t) else None
    except (ControlError, ValueError):
        return None


def capped_tfinal(systems) -> tuple[float | None, bool]:
    """Ortak bitiş zamanı: sistemlerin önerdiği sürelerin en büyüğü; ayrık sistemde en çok MAX_DISCRETE_SAMPLES
    örnek (çok yavaş bir kip arayüzü dakikalarca dondurmasın). Dönüş (süre, kesildi mi)."""
    systems = [s for s in systems if s is not None]
    ends = [e for e in (auto_tfinal(s) for s in systems) if e]
    if not ends:
        return None, False
    tf = max(ends)
    caps = [MAX_DISCRETE_SAMPLES * s.Ts for s in systems if s.Ts > 0]
    if caps and tf > min(caps):
        return min(caps), True
    return tf, False


def response_data(sys, tfinal: float | None) -> dict:
    """Adım cevabı verisi (arka planda hesaplanır): {"t", "y" (nt, ny, nu), "info" (SISO stepinfo), "final" (DC
    kazancı, yerleşmiyorsa nan), "cut" (yavaş kip: ölçütler pencereden), "error"}. Ölçütler kütüphanenin
    `stepinfo(sys)`'idir; ayrık sistemin kendi benzetim süresi MAX_DISCRETE_SAMPLES örneği aşıyorsa çizilen pencereden
    `stepinfo(y, t, yfinal = DC kazancı)` (MATLAB'in veri biçimi)."""
    from spt.control.freqresp import dcgain
    from spt.control.timeresp import stepinfo
    try:
        t, y = cp.calc_step(sys, tfinal)
    except (ControlError, ValueError, np.linalg.LinAlgError) as exc:
        return {"error": str(exc)}
    out = {"t": t, "y": y, "info": None, "final": math.nan, "cut": False}
    if sys.ninputs != 1 or sys.noutputs != 1:
        return out
    try:
        own = auto_tfinal(sys)
        if sys.Ts > 0 and (own is None or own / sys.Ts > MAX_DISCRETE_SAMPLES):
            yf = float(np.ravel(dcgain(sys))[0])
            out["info"] = stepinfo(y[:, 0, 0], t, yfinal=yf)
            out["cut"] = True
        else:
            out["info"] = stepinfo(sys)
        if np.isfinite(float(out["info"].get("SettlingTime", math.nan))):
            out["final"] = float(np.ravel(dcgain(sys))[0])
    except (ControlError, ValueError, np.linalg.LinAlgError) as exc:
        out["info_error"] = str(exc)
    return out


def responses(items: dict, tfinal: float | None) -> dict:
    """Birden çok sistemin adım cevabı verisi (anahtar → response_data). Kararsız tasarımın üstel büyüyen cevabındaki
    taşma uyarıları bastırılır (eğri sonlu kısmıyla çizilir)."""
    with np.errstate(over="ignore", invalid="ignore"):
        return {k: response_data(s, tfinal) for k, s in items.items() if s is not None}


class DesignStepPlot(cp.StepPlot):
    """StepPlot; eğriler ve stepinfo ölçütleri arka planda hesaplanmış veriden (`response_data`) çizilir — yavaş ya da
    çok hızlı kipli bir tasarım arayüzü dondurmaz. Veri verilmezse aynı hesap eş zamanlı yapılır."""

    def set_series(self, series, tfinal: float | None = None, data: dict | None = None, **opts) -> list[str]:
        cut = False
        if tfinal is None and not data:
            tfinal, cut = capped_tfinal([s.sys for s in series])
        self._data = dict(data or {})
        self._cut_notes: list[str] = []
        self.info = {}
        errs = cp.PlotCard.set_series(self, series, tfinal=tfinal, **opts)
        notes = list(self._cut_notes)
        if cut and not notes:
            notes.append(f"cevap {tfinal:.4g} s'de kesildi (yavaş kip)")
        if notes:
            text = "<br>".join(([self.note.text()] if self.note.isVisible() and self.note.text() else []) + notes)
            self.note.setText(text)
            self.note.show()
        return errs

    def _draw(self, s, tfinal: float | None = None, **opts) -> None:
        d = self._data.get(s.key)
        if d is None:
            d = response_data(s.sys, tfinal)
        if d.get("error"):
            raise ControlError(d["error"])
        sys = s.sys
        t, y = d["t"], d["y"]
        for i in range(y.shape[1]):
            for j in range(y.shape[2]):
                io = cp.io_name(sys, i, j)
                name = f"{s.name} ({io})" if io else s.name
                style = cp.CHANNEL_STYLES[(i * y.shape[2] + j) % len(cp.CHANNEL_STYLES)]
                self._time_curve(t, y[:, i, j], s.color, name, style, sys.Ts > 0, sys.Ts)
        if d.get("info") is not None:
            self.info[s.key] = d["info"]
            if d.get("cut"):
                self._cut_notes.append(f"{s.name}: yavaş kip — ölçütler {t[-1]:.4g} s'lik pencereden")
            else:
                self._marks(s, t, y[:, 0, 0], d["info"], d["final"])

    def _marks(self, s, t, y, info: dict, final: float) -> None:
        """Karakteristik işaretleri (StepPlot'unkiyle aynı tanımlar; değerler kütüphanenin stepinfo'sundan)."""
        c = s.color
        ts = float(info.get("SettlingTime", np.nan))
        if "steady" in self.markers and np.isfinite(final):
            self.guide_line(final, 0, c, Qt.PenStyle.DotLine)
            self.points([t[-1]], [final], c, f"{s.name} · kalıcı durum",
                        lambda x, v: [f"kalıcı durum (DC kazancı) = {v:.6g}"], symbol="t1", size=9)
        peak_t, peak = float(info.get("PeakTime", np.nan)), float(info.get("Peak", np.nan))
        if "peak" in self.markers and np.isfinite(peak_t) and np.isfinite(peak) and peak_t > 0:
            k = int(np.argmin(np.abs(t - peak_t)))
            os_ = float(info.get("Overshoot", np.nan))
            self.points([t[k]], [y[k]], c, f"{s.name} · tepe",
                        lambda x, v, os_=os_, pk=peak, pt=peak_t: [f"tepe = {pk:.6g}",
                                                                  f"tepe zamanı = {fmt(pt, 's', 6)}",
                                                                  f"aşım = %{os_:.4g}"], symbol="o", size=10)
        if "settling" in self.markers and np.isfinite(ts) and ts > 0:
            k = int(np.argmin(np.abs(t - ts)))
            self.points([t[k]], [y[k]], c, f"{s.name} · yerleşme",
                        lambda x, v, ts=ts: [f"yerleşme süresi = {fmt(ts, 's', 6)}", f"y = {v:.6g}"], symbol="s",
                        size=9)
        tr = float(info.get("RiseTime", np.nan))
        if "rise" in self.markers and np.isfinite(tr) and tr > 0 and np.isfinite(final) and final != y[0]:
            frac = (y - y[0]) / (final - y[0])
            k = np.flatnonzero(frac >= 0.9)
            if len(k):
                self.points([t[k[0]]], [y[k[0]]], c, f"{s.name} · yükselme",
                            lambda x, v, tr=tr: [f"yükselme süresi (%10→%90) = {fmt(tr, 's', 6)}",
                                                 f"%90'a ulaşma t = {fmt(x, 's', 6)}"], symbol="d", size=10)


# ============================================================================ hazır zaman dizileri
class CurvesCard(cp.PlotCard):
    """Hazır (t, y) dizilerini çizer. `show(curves, guides)`: curves = [(ad, renk, t, y, stil, ayrık)],
    guides = [(değer, açı, metin)] (gri kılavuz çizgileri: kısıt sınırları, referans)."""

    def __init__(self, theme: ThemeManager, title: str, y_label: str = "Genlik", y_units: str | None = None,
                 parent=None):
        super().__init__(theme, title, "Zaman", y_label, x_units="s", y_units=y_units, parent=parent)

    def show(self, curves: list[tuple], guides: list[tuple] = (), note: str = "") -> None:
        self.clear()
        for name, color, t, y, style, discrete in curves:
            t = np.asarray(t, float)
            y = np.asarray(y, float)

            def describe(x, v, Ts=(t[1] - t[0]) if (discrete and len(t) > 1) else 0.0):
                lines = [f"t = {fmt(x, 's', 6)}"]
                if Ts > 0:
                    lines[0] += f"  (k = {int(round(x / Ts))})"
                return lines + [f"değer = {v:.6g}"]

            if discrete and len(t) > 1:
                xs = np.repeat(t, 2)[1:]
                ys = np.repeat(y, 2)[:-1]
                self.curve(xs, ys, color, name, describe, None, style)
            else:
                self.curve(t, y, color, name, describe, None, style)
        for value, angle, text in guides:
            if np.isfinite(value):
                self.guide_line(float(value), angle, cp.guide(), Qt.PenStyle.DashLine, text)
        self.note.setText(note)
        self.note.setVisible(bool(note))
        self._finish()
        self.drawn.emit()


# ============================================================================ küçük bileşenler
def spin_box(lo, hi, value, decimals=4, suffix="", tip="") -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(decimals)
    w.setValue(value)
    w.setSuffix(suffix)
    w.setKeyboardTracking(False)
    w.setToolTip(tip)
    w.setStepType(QDoubleSpinBox.StepType.AdaptiveDecimalStepType)
    return w


def mono_label(text: str = "") -> QLabel:
    w = label(text, "Readout")
    f = QFont("Consolas")
    f.setStyleHint(QFont.StyleHint.Monospace)
    w.setFont(f)
    w.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    w.setWordWrap(False)
    return w


def scroll_area(widget: QWidget, horizontal: bool = False) -> QScrollArea:
    s = QScrollArea()
    s.setWidgetResizable(True)
    s.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded if horizontal
                                   else Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    s.setFrameShape(QScrollArea.Shape.NoFrame)
    s.setWidget(widget)
    return s


# ============================================================================ tablo yardımcısı
def make_table(columns) -> QTableWidget:
    tbl = QTableWidget(0, len(columns))
    tbl.setHorizontalHeaderLabels([c for c, _ in columns])
    for i, (_c, tip) in enumerate(columns):
        tbl.horizontalHeaderItem(i).setToolTip(tip)
    tbl.verticalHeader().setVisible(False)
    tbl.setAlternatingRowColors(True)
    tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    tbl.setSelectionMode(QAbstractItemView.SelectionMode.ContiguousSelection)
    tbl.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft)
    tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    tbl.horizontalHeader().setStretchLastSection(True)
    tbl.verticalHeader().setDefaultSectionSize(24)
    tbl.setMinimumHeight(64)
    return tbl


def fill_table(tbl: QTableWidget, rows: list[list[str]], colors: list[str | None] | None = None,
               state_col: int | None = None, tokens=None) -> None:
    tbl.setRowCount(len(rows))
    for r, cells in enumerate(rows):
        for c, text in enumerate(cells):
            item = QTableWidgetItem(text)
            if c == 0 and colors and colors[r]:
                item.setForeground(QColor(colors[r]))
            if state_col is not None and c == state_col and tokens is not None and text in ("kararlı", "kararsız"):
                item.setForeground(QColor(tokens.good if text == "kararlı" else tokens.critical))
            tbl.setItem(r, c, item)


def num(x, digits: int = 5) -> str:
    if x is None:
        return "—"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(x):
        return "—"
    if math.isinf(x):
        return "∞" if x > 0 else "−∞"
    return f"{x:.{digits}g}"
