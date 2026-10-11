"""Tema ile uyumlu pyqtgraph grafik fabrikası ve imleç (crosshair) katmanı."""
from __future__ import annotations

from collections.abc import Callable

import pyqtgraph as pg
from PyQt6.QtCore import QEvent, QObject, QPointF, Qt

from spt.gui.theme import ThemeManager, ui_font

LINE_WIDTH = 2


def make_plot(theme: ThemeManager, x_label: str, y_label: str, x_units: str | None = None,
              y_units: str | None = None, si_prefix: bool = True) -> pg.PlotWidget:
    """Kart yüzeyiyle kaynaşan, ızgarası geri planda kalan grafik."""
    t = theme.tokens
    pw = pg.PlotWidget(background=t.surface)
    pi = pw.getPlotItem()
    pi.hideButtons()
    pi.setContentsMargins(0, 4, 4, 0)
    tick_font = ui_font(11)
    for name in ("left", "bottom"):
        ax = pi.getAxis(name)
        ax.setPen(pg.mkPen(t.axis))
        ax.setTextPen(pg.mkPen(t.text_muted))
        ax.setTickFont(tick_font)
        ax.setStyle(tickTextOffset=6)
        if not si_prefix:
            set_si_prefix(ax, False)
    pi.getAxis("left").setWidth(56)
    pi.showGrid(x=True, y=True, alpha=0.35)
    style = {"color": t.text_muted, "font-size": "11px"}
    pi.setLabel("bottom", x_label, units=x_units, **style)
    pi.setLabel("left", y_label, units=y_units, **style)
    pw.setDownsampling(auto=True, mode="peak")
    pw.setClipToView(True)
    return pw


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
