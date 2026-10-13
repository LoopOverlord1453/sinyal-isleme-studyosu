"""Tema ile uyumlu pyqtgraph grafik fabrikası ve imleç (crosshair) katmanı."""
from __future__ import annotations

import math
from collections.abc import Callable

import pyqtgraph as pg
from PyQt6.QtCore import QEvent, QObject, QPointF, Qt
from PyQt6.QtGui import QFontMetricsF

from spt.gui.theme import ThemeManager, ui_font

LINE_WIDTH = 2
AXIS_FONT_PX = 12   # eksen sayıları ve başlıkları: açık zeminde 11 px soluk gri silik kalıyordu


def _middle_button_pans() -> None:
    """Orta tuşla basılı tutup sürüklemek her grafikte kaydırır (sağa / sola / yukarı / aşağı), fare modundan bağımsız.

    pyqtgraph dikdörtgen yakınlaştırma modunda (RectMode) orta tuşu da sol tuş gibi yakınlaştırma kutusu sayar; Canlı
    İzleme (büyüteç aracı) ve Sensör Füzyonu bu modda olduğundan orta tuşla kaydırılamıyordu. Orta tuşlu sürükleme
    süresince mod geçici olarak kaydırmaya alınır; sol tuş davranışı değişmez. Tüm ViewBox'lara bir kez uygulanır."""
    if getattr(pg.ViewBox, "_spt_middle_pan", False):
        return
    original = pg.ViewBox.mouseDragEvent

    def mouse_drag(self, ev, axis=None):
        if ev.button() == Qt.MouseButton.MiddleButton and self.state["mouseMode"] == pg.ViewBox.RectMode:
            self.state["mouseMode"] = pg.ViewBox.PanMode
            try:
                original(self, ev, axis)
            finally:
                self.state["mouseMode"] = pg.ViewBox.RectMode
            return
        original(self, ev, axis)

    pg.ViewBox.mouseDragEvent = mouse_drag
    pg.ViewBox._spt_middle_pan = True


_middle_button_pans()


class ReadableAxis(pg.AxisItem):
    """Dikey eksen: ana çentik aralığı yazı yüksekliğine göre seçilir.

    pyqtgraph ana çentik sayılarını sıkışık olsa da her zaman çizer ve aralığı yalnız eksen uzunluğundan seçer; alçak
    kartlarda (≈ 70 px) 0.2 / 0.4 / 0.6 / 0.8 üst üste biniyordu. Aralık 1-2-5 dizisinde, ardışık iki sayı arasında
    `MIN_LINES` satır kalacak kadar büyütülür; bu aralıkta ikiden az sayı okunabiliyorsa (eksen ucundaki sayıyı
    pyqtgraph kesileceği için çizmez) üst üste binmeden en çok sayı gösteren aralık seçilir. Alt çentiklerden yalnız
    yeni aralığı tam bölenler kalır."""

    MIN_LINES = 1.6     # rahat aralık (satır yüksekliği cinsinden)
    MIN_TOUCH = 1.05    # sayıların üst üste binmediği en küçük aralık

    def tickSpacing(self, minVal, maxVal, size):  # noqa: N802 (pyqtgraph adı)
        levels = super().tickSpacing(minVal, maxVal, size)
        span = maxVal - minVal
        if not levels or self.logMode or span <= 0 or size <= 0:
            return levels
        font = self.style["tickFont"]
        line = QFontMetricsF(font).height() if font is not None else 14.0
        px = size / span   # veri birimi başına piksel
        major = levels[0][0]
        if major * px >= line * self.MIN_LINES:
            return levels

        edge = 0.5 * line / px   # ucuna yarım satırdan yakın sayı kesilir, çizilmez

        def readable(step: float) -> int:
            lo, hi = minVal + edge, maxVal - edge
            return max(0, math.floor(hi / step + 1e-9) - math.ceil(lo / step - 1e-9) + 1)

        decade = 10.0 ** math.floor(math.log10(major))
        steps = [m * decade * 10 ** k for k in range(4) for m in (1, 2, 5)]
        steps = [s for s in steps if s >= major * (1 - 1e-9)]
        step = next((s for s in steps if s * px >= line * self.MIN_LINES), steps[-1])
        if readable(step) < 2:
            dense = [s for s in steps if s * px >= line * self.MIN_TOUCH]
            best = max(dense, key=lambda s: (readable(s), s), default=step)
            if readable(best) > readable(step):
                step = best
        minor = [lvl for lvl in levels[1:]
                 if lvl[0] < step and abs(step / lvl[0] - round(step / lvl[0])) < 1e-6]
        return [(step, 0)] + minor


def make_plot(theme: ThemeManager, x_label: str, y_label: str, x_units: str | None = None,
              y_units: str | None = None, si_prefix: bool = True) -> pg.PlotWidget:
    """Kart yüzeyiyle kaynaşan, ızgarası geri planda kalan grafik. Eksen yazıları ikincil metin renginde (yüzeye
    ≥ 7:1), eksen çizgisi `axis` tonunda (yüzeye ≥ 2.4:1) — veri eğrilerinden geride ama okunur."""
    t = theme.tokens
    pw = pg.PlotWidget(background=t.surface, axisItems={"left": ReadableAxis("left")})
    pi = pw.getPlotItem()
    pi.hideButtons()
    pi.setContentsMargins(0, 4, 4, 0)
    tick_font = ui_font(AXIS_FONT_PX)
    for name in ("left", "bottom"):
        ax = pi.getAxis(name)
        ax.setPen(pg.mkPen(t.axis))
        ax.setTextPen(pg.mkPen(t.text_secondary))
        ax.setTickFont(tick_font)
        ax.setStyle(tickTextOffset=6)
        if not si_prefix:
            set_si_prefix(ax, False)
    pi.getAxis("left").setWidth(60)
    pi.showGrid(x=True, y=True, alpha=0.35)
    style = {"color": t.text_secondary, "font-size": f"{AXIS_FONT_PX}px"}
    pi.setLabel("bottom", x_label, units=x_units, **style)
    pi.setLabel("left", y_label, units=y_units, **style)
    pw.setDownsampling(auto=True, mode="peak")
    pw.setClipToView(True)
    return pw


def plot_font(family: str | None, px: int):
    """Grafik yazı tipi: aile verilmezse arayüz yazı tipi."""
    font = ui_font(px)
    if family:
        font.setFamilies([family])
    return font


def set_plot_font(plot: pg.PlotWidget, family: str | None, px: int) -> None:
    """Eksen sayıları ve başlıklarının yazı tipi / boyutu. Başlık setLabel ile değil stil sözlüğüyle güncellenir:
    setLabel SI ön eki ayarını sıfırlar (bkz. set_si_prefix). Sol eksen genişliği yazı boyuyla büyür (sayılar
    kesilmesin)."""
    font = plot_font(family, px)
    pi = plot.getPlotItem()
    for name in ("left", "bottom"):
        ax = pi.getAxis(name)
        ax.setTickFont(font)
        style = dict(ax.labelStyle)
        style["font-size"] = f"{px}px"
        if family:
            style["font-family"] = f"'{family}'"
        else:
            style.pop("font-family", None)
        ax.labelStyle = style
        ax._updateLabel()  # noqa: SLF001 (pyqtgraph'ın etiket yenileme yolu; setLabel SI ayarını bozar)
    pi.getAxis("left").setWidth(max(60, round(60 * px / AXIS_FONT_PX)))


def set_si_prefix(axis: pg.AxisItem, enabled: bool) -> None:
    """Eksen birimine otomatik SI ön eki (k, M, m…). pyqtgraph 0.14'te enableAutoSIPrefix(False) o anki ön eki
    dondurur (ör. log eksende "kHz" kalır); kapatırken ölçek burada 1'e sabitlenir. setLabel çağrısı bu ayarı
    sıfırladığından etiketten sonra çağrılmalıdır."""
    axis.setSIPrefixEnableRanges(None if enabled else ((0.0, 0.0),))
    axis.enableAutoSIPrefix(enabled)


def pen(color: str, width: float = LINE_WIDTH) -> pg.QtGui.QPen:
    p = pg.mkPen(color, width=width)
    p.setCosmetic(True)
    return p


class Crosshair(QObject):
    """Fare grafiğin üzerindeyken dikey çizgi gösterir ve `on_move(x, y)` çağırır.

    Fare grafikten çıkınca `on_move(None, None)` çağrılır.
    """

    def __init__(self, plot: pg.PlotWidget, theme: ThemeManager,
                 on_move: Callable[[float | None, float | None], None]):
        super().__init__(plot)
        self._plot = plot
        self._on_move = on_move
        line_pen = pg.mkPen(theme.tokens.text_muted, width=1, style=Qt.PenStyle.DashLine)
        self._vline = pg.InfiniteLine(angle=90, movable=False, pen=line_pen)
        self._vline.setVisible(False)
        plot.addItem(self._vline, ignoreBounds=True)
        plot.scene().sigMouseMoved.connect(self._moved)
        plot.viewport().installEventFilter(self)

    def _moved(self, pos: QPointF) -> None:
        vb = self._plot.getPlotItem().getViewBox()
        if not vb.sceneBoundingRect().contains(pos):
            self._hide()
            return
        p = vb.mapSceneToView(pos)
        self._vline.setPos(p.x())
        self._vline.setVisible(True)
        self._on_move(p.x(), p.y())

    def _hide(self) -> None:
        self._vline.setVisible(False)
        self._on_move(None, None)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.Leave:
            self._hide()
        return False


def series_color(item) -> str | None:
    """Çizim öğesinin seri rengi (#rrggbb): çizgi kalemi, çizgisizse işaretin kalemi / dolgusu. Bilinmeyen öğe: None."""
    if isinstance(item, pg.PlotDataItem):
        candidates = (item.opts.get("pen"), item.opts.get("symbolPen"), item.opts.get("symbolBrush"))
    elif isinstance(item, pg.PlotCurveItem):
        candidates = (item.opts.get("pen"),)
    elif isinstance(item, pg.ScatterPlotItem):
        candidates = (item.opts.get("pen"), item.opts.get("brush"))
    elif isinstance(item, pg.InfiniteLine):
        candidates = (item.pen,)
    else:
        return None
    for c in candidates:
        if c is None:
            continue
        color = c.color() if hasattr(c, "color") else pg.mkColor(c)
        if color.alpha() > 0 and not (hasattr(c, "style") and c.style() == Qt.PenStyle.NoPen):
            return color.name()
    return None


def apply_legend(plots, hidden: set[str]) -> None:
    """Lejantta kapatılan renklerdeki çizim öğelerini gizler; yeniden açılınca yalnız lejantın gizlediklerini gösterir
    (görünümün kendi gizlediği öğelere, ör. filtre kapalıyken filtreli eğriye, dokunulmaz)."""
    for plot in plots:
        for item in list(plot.getPlotItem().items):
            color = series_color(item)
            if color is None:
                continue
            if color in hidden:
                if item.isVisible():
                    item.setVisible(False)
                    item._spt_legend_hidden = True
            elif getattr(item, "_spt_legend_hidden", False):
                item._spt_legend_hidden = False
                item.setVisible(True)
