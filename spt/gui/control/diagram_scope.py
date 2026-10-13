"""Benzetim sonuçları: her Skop bloğu ve her kaydedilen sinyal (telde "Sinyali kaydet"; Simulink'teki To Workspace /
veri denetçisi) ayrı kartta, ölçülerek seçilmiş zıt renklerle, veri ipuçlarıyla.

Renkler: ilk altı renk Sensör Füzyonu görünümünde ölçülen paletten (kırmızı, sarı, yeşil, camgöbeği, mavi, eflatun;
açık temada sarı yerine turuncu; normal görüşte her çift OKLab ΔE ≥ 19), sonra metin rengine yakın nötr (koyu temada
#f5f5f5, açık temada #1d1a16); daha fazla seri gerekirse renkler OKLab'da en uzak nokta seçimiyle (kart yüzeyinden ve
önceki tüm renklerden en uzak; açıklık koyu temada 0.62–0.92, açık temada 0.30–0.62 bandında, ikisi de yüzeye ≥ 3:1)
eklenir. Aynı görünümde (tüm skop kartları birlikte) hiçbir renk iki seride kullanılmaz; renk seriye (skop + kanal)
bağlıdır.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QScrollArea, QVBoxLayout, QWidget

from spt.gui.theme import ThemedColors, active, themed
from spt.gui.widgets.common import Card, LegendBar, label
from spt.gui.widgets.inspector import DataTips, PlotAnalysis, Trace, curve_stats, fmt, visible_slice
from spt.gui.widgets.plot import make_plot, pen

BASE_COLORS = ThemedColors(
    dark=("#f54748", "#dfa11a", "#45f27f", "#4cd1ee", "#488bfb", "#e03fc7", "#f5f5f5"),
    light=("#b31319", "#ea6a09", "#17a664", "#5391cc", "#1a4eda", "#af1da0", "#1d1a16"),
)
_LIGHTNESS_BAND = {"dark": (0.62, 0.92), "light": (0.30, 0.62)}  # genişletme renklerinin OKLab açıklık bandı


def surface() -> str:
    """Kart yüzeyi (etkin tema): genişletme renkleri bundan uzak seçilir."""
    return themed("#121110", "#fffdfa")
PLOT_MIN_HEIGHT = 170   # zaman ekseni ve etiketleri kesilmeden okunur


def _srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (0–1, (..., 3)) → OKLab (Björn Ottosson tanımı)."""
    c = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                   [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
    m2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                   [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
    lms = c @ m1.T
    return np.cbrt(lms) @ m2.T


def _hex_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def delta_e(a: str, b: str) -> float:
    """İki rengin OKLab uzaklığı (×100, ΔE ölçeği)."""
    la, lb = _srgb_to_oklab(_hex_rgb(a)), _srgb_to_oklab(_hex_rgb(b))
    return float(np.linalg.norm(la - lb) * 100.0)


def distinct_colors(n: int) -> tuple[str, ...]:
    """n farklı seri rengi (etkin tema): önce ölçülmüş taban palet, sonra OKLab'da en uzak nokta seçimiyle
    genişletme."""
    return _distinct_colors(n, active().name)


@lru_cache(maxsize=16)
def _distinct_colors(n: int, theme: str) -> tuple[str, ...]:
    base = BASE_COLORS.dark if theme == "dark" else BASE_COLORS.light
    out = list(base[:n])
    if n <= len(out):
        return tuple(out)
    lo, hi = _LIGHTNESS_BAND[theme]
    steps = np.linspace(0.0, 1.0, 18)
    grid = np.stack(np.meshgrid(steps, steps, steps, indexing="ij"), axis=-1).reshape(-1, 3)
    lab = _srgb_to_oklab(grid)
    ok = (lab[:, 0] >= lo) & (lab[:, 0] <= hi) & (np.hypot(lab[:, 1], lab[:, 2]) >= 0.06)
    grid, lab = grid[ok], lab[ok]
    ref = [_srgb_to_oklab(_hex_rgb(c)) for c in out + ["#121110" if theme == "dark" else "#fffdfa"]]
    dmin = np.min(np.stack([np.linalg.norm(lab - r, axis=1) for r in ref]), axis=0)
    while len(out) < n:
        k = int(np.argmax(dmin))
        rgb = grid[k]
        out.append("#" + "".join(f"{int(round(v * 255)):02x}" for v in rgb))
        dmin = np.minimum(dmin, np.linalg.norm(lab - lab[k], axis=1))
        dmin[k] = -1.0
    return tuple(out)


class ScopePanel(QWidget):
    """Skop sonuç kartları. `set_result(sonuç, sıra)`: sıra = kartların gösterim sırası (diyagramdaki skoplar)."""

    def __init__(self, theme, parent: QWidget | None = None):
        super().__init__(parent)
        self._theme = theme
        self.cards: dict[str, Card] = {}               # Skop blokları (ad → kart)
        self.signal_cards: dict[str, Card] = {}        # kaydedilen sinyaller (sinyal adı → kart)
        self.plots: dict[str, pg.PlotWidget] = {}
        self.traces: dict[str, list[Trace]] = {}
        self._tips: list[DataTips] = []
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(self.scroll)
        self.clear()

    def clear(self, text: str = "Benzetim sonuçları burada: diyagrama Skop bloğu ekleyin (ya da bir teli sağ tıklayıp "
                                "\"Sinyali kaydet\" seçin) ve Çalıştır'a basın.") -> None:
        self.cards.clear()
        self.signal_cards.clear()
        self.plots.clear()
        self.traces.clear()
        self._tips.clear()
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        empty = label(text, "EmptyState")
        empty.setWordWrap(True)
        empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(empty)
        lay.addStretch(1)
        self.scroll.setWidget(w)

    def series_colors(self) -> list[str]:
        return [tr.color for trs in self.traces.values() for tr in trs]

    @property
    def count(self) -> int:
        """Gösterilen sonuç kartı sayısı (skop + kaydedilen sinyal)."""
        return len(self.cards) + len(self.signal_cards)

    def set_result(self, result, order: list[str] | None = None) -> None:
        scopes = dict(result.scopes)
        names = [n for n in (order or []) if n in scopes] + [n for n in scopes if n not in (order or [])]
        logged = [n for n in getattr(result, "signals", {}) if n not in scopes]
        if not names and not logged:
            self.clear("Sonuç yok: 'Gözlem' kategorisinden Skop ekleyin ya da bir teli sağ tıklayıp \"Sinyali kaydet\" "
                       "seçin, sonra yeniden çalıştırın.")
            return
        self.cards.clear()
        self.signal_cards.clear()
        self.plots.clear()
        self.traces.clear()
        self._tips.clear()
        series: list[tuple[str, str, np.ndarray, np.ndarray]] = []   # (kart anahtarı, ad, t, y)
        for name in names:
            sd = scopes[name]
            t = np.asarray(sd.t, dtype=np.float64)
            for data, lab in zip(sd.data, sd.labels):
                arr = np.asarray(data, dtype=np.float64)
                if arr.ndim == 1:
                    arr = arr[:, None]
                for j in range(arr.shape[1]):
                    series.append((name, lab if arr.shape[1] == 1 else f"{lab}[{j + 1}]", t, arr[:, j]))
        t_all = np.asarray(result.t, dtype=np.float64)
        for sig in logged:
            arr = np.asarray(result.signals[sig], dtype=np.float64)
            if arr.ndim == 1:
                arr = arr[:, None]
            for j in range(arr.shape[1]):
                series.append(("sinyal:" + sig, sig if arr.shape[1] == 1 else f"{sig}[{j + 1}]", t_all, arr[:, j]))
        colors = distinct_colors(len(series))
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 4, 0)
        lay.setSpacing(12)
        k = 0
        groups = [(name, name, False) for name in names] + [("sinyal:" + sig, sig, True) for sig in logged]
        for key, title, is_signal in groups:
            if is_signal and key == groups[len(names)][0] and names:
                lay.addWidget(label("Kaydedilen sinyaller", "CardTitle"))
            mine = [s for s in series if s[0] == key]
            card = Card(title if not is_signal else f"{title}  (kayıt)")
            plot = make_plot(self._theme, "Zaman", "", x_units="s")
            plot.setMinimumHeight(PLOT_MIN_HEIGHT)
            card.body.addWidget(plot, 1)
            traces: list[Trace] = []
            legend = LegendBar(lambda p=plot: [p])
            for _scope, sname, t, y in mine:
                color = colors[k]
                k += 1
                item = plot.plot(t, y, pen=pen(color), name=sname)
                traces.append(Trace(item, sname, color, self._describe(sname)))
            legend.set_items([(tr.color, tr.name) for tr in traces])
            card.add_left(legend, optional=True)
            if result.cancelled:
                card.add_right(label("İptal edildi: kısmi sonuç", "Muted"))
            self._tips.append(DataTips(plot, self._theme, lambda trs=traces: trs))
            PlotAnalysis(card, plot, self._theme, lambda x0, x1, trs=traces: self._stats(trs, x0, x1))
            card.setMinimumHeight(PLOT_MIN_HEIGHT + 70)
            lay.addWidget(card, 1)                 # kartlar alanı paylaşır; sığmazsa kaydırılır
            (self.signal_cards if is_signal else self.cards)[title] = card
            self.plots[key] = plot
            self.traces[key] = traces
        if len(groups) > 1:
            for plot in list(self.plots.values())[1:]:
                plot.setXLink(next(iter(self.plots.values())))    # tüm kartlar aynı zaman ekseni (MATLAB gibi)
        self.scroll.setWidget(w)

    @staticmethod
    def _describe(name: str):
        def f(x: float, y: float) -> list[str]:
            return [f"t = {fmt(x, 's')}", f"{name} = {y:.6g}"]
        return f

    @staticmethod
    def _stats(traces: list[Trace], x0: float, x1: float):
        out = []
        for tr in traces:
            x, y = tr.item.xData, tr.item.yData
            if x is None or y is None:
                continue
            sl = visible_slice(np.asarray(x), x0, x1)
            out.append((tr.name, tr.color, curve_stats(np.asarray(x)[sl], np.asarray(y)[sl], "s", "")))
        return out
