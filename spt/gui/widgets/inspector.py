"""Grafik inceleme: veri ipuçları (seçilen noktanın değerleri) ve görünen aralığın ayrıntılı analizi.

Veri ipuçları (MATLAB "data tip" benzeri):
  - Grafiğe tıklayınca en yakın eğrinin (piksel mesafesiyle) en yakın noktasına işaret konur, değerleri yazılır.
  - Eğri ekran için seyreltilmiş olsa bile, eğri bir `exact` geri çağrısı sağlıyorsa değer tampondaki gerçek
    örnekten okunur (örnek numarasıyla birlikte).
  - İşarete tekrar tıklamak onu kaldırır; sağ tık menüsündeki "Veri ipuçlarını temizle" hepsini kaldırır.
  - Çift tık grafiğin kendi işine (görünümü sıfırlama) ayrılmıştır: tek tık, çift tık süresi kadar beklenir.
  - Yön (MATLAB "Location"): etiket kutusu işarete göre sağ üst / sol üst / sağ alt / sol alt çeyrekte durur.
    Etiket sürüklenince imlecin işarete göre bulunduğu çeyreğe oturur; ipucuna sağ tıklayınca "Konum" menüsü.
    Seçilen yönde grafiğe sığmayan etiket o an sığan yöne çevrilir (tercih saklanır). Etiket işaretten piksel
    olarak sabit uzaklıktadır (yakınlaştırma / kaydırmada kaymaz). Son seçilen yön yeni ipuçlarının varsayılanıdır.

Analiz tablosu: kartın altında açılan tablo; görünen x aralığı için eğri başına istatistikler (yakınlaştırınca
güncellenir). İstatistikleri her grafik kendi türüne göre sağlar (`time_stats`, `spectrum_stats`, `curve_stats`).
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pyqtgraph as pg
from scipy import fft as sp_fft
from PyQt6.QtCore import QObject, QPointF, Qt, QTimer
from PyQt6.QtGui import QActionGroup, QColor, QKeySequence
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QHeaderView,
    QMenu,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
)

from spt.gui.settings import app_settings
from spt.gui.theme import ThemeManager

PICK_RADIUS_PX = 28
TIP_HIT_PX = 12
# Etiket yönü: kutunun işarete göre bulunduğu çeyrek (n/s: üst/alt, e/w: sağ/sol)
ORIENTATIONS = {"ne": "Sağ üst", "nw": "Sol üst", "se": "Sağ alt", "sw": "Sol alt"}
DEFAULT_ORIENTATION = "ne"
ORIENTATION_KEY = "display/datatip_orientation"
TIP_GAP_PX = 7                  # etiket köşesinin işaret merkezinden yatay ve dikey uzaklığı
MAX_STATS_SAMPLES = 1_000_000   # analiz tablosunda en fazla bu kadar örnek (en yeni kısım)
MAX_DOMINANT_FFT = 1 << 18      # baskın frekans FFT'si (hızlı uzunluk, en yeni kısım)


@dataclass
class Trace:
    """İncelenebilir bir eğri ve onu anlatan biçimlendiriciler."""
    item: object                                   # PlotDataItem veya ScatterPlotItem
    name: str
    color: str
    describe: Callable[[float, float], list[str]]  # (x, y) gerçek koordinat → etiket satırları
    exact: Callable[[float], tuple[float, float] | None] | None = None  # x → tampondaki gerçek örnek (x, y)
    extra: dict = field(default_factory=dict)


def _arrays(item) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(item, pg.PlotDataItem):
        x, y = item.xData, item.yData
    else:
        x, y = item.getData()
    if x is None or y is None:
        return np.zeros(0), np.zeros(0)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


# SI ön eki (m, µ, k…) yalnızca ön eksiz temel birimlere eklenir; "mV", "µT", "°/s" gibi birimler olduğu gibi kalır.
SI_UNITS = {"V", "A", "Hz", "s", "W", "T", "Pa", "N", "g", "m", "m/s", "m/s²", "rad", "rad/s", "Ω", "F", "H"}


def group(n: int) -> str:
    """Tam sayıyı ince boşlukla gruplar: 12 345 (nokta, ondalık noktayla karışıyordu)."""
    return f"{int(n):,}".replace(",", "\u202f")


def fmt(value: float, unit: str = "", digits: int = 5) -> str:
    """Sayıyı birimiyle yazar; temel SI birimlerinde ön ek seçilir (ör. 0.00123 V → 1.23 mV)."""
    if value is None or not np.isfinite(value):
        return "—"
    if unit in SI_UNITS:
        return pg.siFormat(value, precision=digits, suffix=unit, allowUnicode=True)
    text = f"{value:.{digits}g}"
    if unit == "°":
        return f"{text}°"
    return f"{text} {unit}".strip()


def saved_orientation() -> str:
    """Yeni ipuçlarının yönü: kullanıcının en son seçtiği (ayarlarda), yoksa sağ üst."""
    value = str(app_settings().value(ORIENTATION_KEY, DEFAULT_ORIENTATION))
    return value if value in ORIENTATIONS else DEFAULT_ORIENTATION


@dataclass(eq=False)
class _Tip:
    point: QPointF               # işaretin görünüm koordinatı (log x'te log10 x)
    marker: pg.ScatterPlotItem
    label: pg.TextItem
    orientation: str             # kullanıcının tercihi
    shown: str = ""              # gösterilen yön (tercih grafiğe sığmıyorsa çevrilmiş)


class _TipMarker(pg.ScatterPlotItem):
    """İpucu işareti: sağ tık ipucu menüsünü açar (sol tık DataTips'in kaldırma yolundan geçer)."""

    def __init__(self, owner: DataTips, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._owner = owner

    def mouseClickEvent(self, ev) -> None:
        if ev.button() == Qt.MouseButton.RightButton and len(self.pointsAt(ev.pos())):
            ev.accept()
            self._owner._popup_menu(self, ev.screenPos())
            return
        super().mouseClickEvent(ev)


class _TipLabel(pg.TextItem):
    """İpucu etiketi: sürüklenince yönü değişir (grafik kaydırılmaz), sağ tık ipucu menüsünü açar."""

    def __init__(self, owner: DataTips, **kwargs):
        super().__init__(**kwargs)
        self._owner = owner
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setToolTip("Sürükle: etiketin yönü · sağ tık: konum / kaldır")

    def hoverEvent(self, ev) -> None:
        if not ev.isExit():   # sürükleme ve sağ tık bu etikete gider (ViewBox'a değil)
            ev.acceptDrags(Qt.MouseButton.LeftButton)
            ev.acceptClicks(Qt.MouseButton.RightButton)

    def mouseDragEvent(self, ev) -> None:
        if ev.button() != Qt.MouseButton.LeftButton:
            ev.ignore()
            return
        ev.accept()
        if ev.isStart():
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        self._owner._drag(self, ev.scenePos(), ev.isFinish())
        if ev.isFinish():
            self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mouseClickEvent(self, ev) -> None:
        if ev.button() == Qt.MouseButton.RightButton:
            ev.accept()
            self._owner._popup_menu(self, ev.screenPos())


class DataTips(QObject):
    """Bir grafiğe veri ipucu davranışı ekler."""

    def __init__(self, plot: pg.PlotWidget, theme: ThemeManager, traces: Callable[[], list[Trace]]):
        super().__init__(plot)
        self._plot = plot
        self._theme = theme
        self._traces = traces
        self._tips: list[_Tip] = []
        self._menu: QMenu | None = None
        self._pending: QPointF | None = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(QApplication.doubleClickInterval() if QApplication.instance() else 250)
        self._timer.timeout.connect(self._commit_click)
        plot.scene().sigMouseClicked.connect(self._clicked)
        vb = plot.getPlotItem().getViewBox()
        if vb.menu is not None:
            vb.menu.addSeparator()
            vb.menu.addAction("Veri ipuçlarını temizle", self.clear)
        # etiketler işaretten piksel olarak sabit uzaklıkta: görünüm değişince yeniden yerleştir
        vb.sigRangeChanged.connect(self._relayout)
        vb.sigResized.connect(self._relayout)
        plot.setToolTip("Tıkla: noktanın değeri · işarete tekrar tıkla: kaldır · etiketi sürükle: yönü · "
                        "ipucuna sağ tık: konum / kaldır")

    # ---------------------------------------------------------------- olaylar
    def _clicked(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if event.double():  # çift tık: görünümü sıfırlama (grafiğin kendi işi); bekleyen tıklamayı iptal et
            self._timer.stop()
            self._pending = None
            return
        vb = self._plot.getPlotItem().getViewBox()
        if not vb.sceneBoundingRect().contains(event.scenePos()):
            return
        self._pending = QPointF(event.scenePos())
        self._timer.start()

    def _commit_click(self) -> None:
        pos, self._pending = self._pending, None
        if pos is None:
            return
        if any(tip.label.isVisible() and tip.label.sceneBoundingRect().contains(pos) for tip in self._tips):
            return  # etikete tıklamak ipucu eklemez / kaldırmaz (etiket sürüklenir; kaldırma: işaret ya da menü)
        if self._remove_near(pos):
            return
        self.add_at_scene(pos)

    # ---------------------------------------------------------------- ipucu
    def _log_x(self) -> bool:
        return bool(self._plot.getPlotItem().ctrl.logXCheck.isChecked())

    def _px_scale(self) -> tuple[float, float]:
        vb = self._plot.getPlotItem().getViewBox()
        (x0, x1), (y0, y1) = vb.viewRange()
        rect = vb.sceneBoundingRect()
        return rect.width() / max(x1 - x0, 1e-300), rect.height() / max(y1 - y0, 1e-300)

    def nearest(self, view_x: float, view_y: float) -> tuple[Trace, float, float] | None:
        """Görünüm koordinatındaki noktaya piksel olarak en yakın eğri noktası: (eğri, x, y) gerçek koordinat."""
        sx, sy = self._px_scale()
        log_x = self._log_x()
        best = None
        for trace in self._traces():
            item = trace.item
            if not item.isVisible():
                continue
            x, y = _arrays(item)
            if not len(x):
                continue
            xv = np.log10(np.where(x > 0, x, np.nan)) if log_x else x
            ok = np.isfinite(xv) & np.isfinite(y)
            if isinstance(item, pg.PlotDataItem) and len(xv) > 64 and np.all(np.diff(xv[ok][:256]) >= 0):
                i = int(np.searchsorted(xv, view_x))
                lo, hi = max(0, i - 2048), min(len(xv), i + 2048)
                idx = np.arange(lo, hi)
            else:
                idx = np.arange(len(xv))
            idx = idx[ok[idx]]
            if not len(idx):
                continue
            d2 = ((xv[idx] - view_x) * sx) ** 2 + ((y[idx] - view_y) * sy) ** 2
            j = int(np.argmin(d2))
            if best is None or d2[j] < best[0]:
                best = (float(d2[j]), trace, float(x[idx[j]]), float(y[idx[j]]))
        if best is None or best[0] > PICK_RADIUS_PX ** 2:
            return None
        _, trace, x, y = best
        if trace.exact is not None:
            refined = trace.exact(x)
            if refined is not None:
                x, y = refined
        return trace, x, y

    def add_at_scene(self, pos: QPointF) -> bool:
        p = self._plot.getPlotItem().getViewBox().mapSceneToView(pos)
        hit = self.nearest(p.x(), p.y())
        if hit is None:
            return False
        trace, x, y = hit
        self.add(trace, x, y)
        return True

    def add(self, trace: Trace, x: float, y: float, orientation: str | None = None) -> None:
        t = self._theme.tokens
        xv = math.log10(x) if self._log_x() and x > 0 else x
        marker = _TipMarker(self, [xv], [y], symbol="o", size=11, pen=pg.mkPen(t.text, width=2),
                            brush=pg.mkBrush(trace.color))
        lines = trace.describe(x, y)
        html = (f'<div style="color:{t.text}; font-size:11px;">'
                f'<span style="color:{trace.color}">●</span> <b>{trace.name}</b><br>' + "<br>".join(lines) + "</div>")
        fill = QColor(t.elevated)
        fill.setAlpha(235)
        label = _TipLabel(self, html=html, fill=pg.mkBrush(fill), border=pg.mkPen(t.border_strong))
        for item in (marker, label):
            item.setZValue(50)
            self._plot.addItem(item, ignoreBounds=True)
        tip = _Tip(QPointF(xv, y), marker, label, orientation if orientation in ORIENTATIONS else saved_orientation())
        self._tips.append(tip)
        self._place(tip)

    # ---------------------------------------------------------------- yön
    def _place(self, tip: _Tip) -> None:
        """Etiketi tercih edilen çeyreğe, köşesi işaretten TIP_GAP_PX uzakta yerleştirir. O yönde grafiğe sığmıyor ve
        karşı yönde sığıyorsa (kenara yakın nokta) o eksende çevrilir."""
        vb = self._plot.getPlotItem().getViewBox()
        m = vb.mapViewToScene(tip.point)
        area = vb.sceneBoundingRect()
        box = tip.label.textItem.boundingRect()   # etiket dönüşümü ölçeği iptal eder: piksel
        w, h, gap = box.width(), box.height(), TIP_GAP_PX
        north, east = tip.orientation[0] == "n", tip.orientation[1] == "e"
        fits_e, fits_w = m.x() + gap + w <= area.right(), m.x() - gap - w >= area.left()
        fits_n, fits_s = m.y() - gap - h >= area.top(), m.y() + gap + h <= area.bottom()
        if not (fits_e if east else fits_w) and (fits_w if east else fits_e):
            east = not east
        if not (fits_n if north else fits_s) and (fits_s if north else fits_n):
            north = not north
        tip.shown = ("n" if north else "s") + ("e" if east else "w")
        corner = QPointF(m.x() + (gap if east else -gap), m.y() + (-gap if north else gap))
        tip.label.setAnchor((0.0 if east else 1.0, 1.0 if north else 0.0))
        tip.label.setPos(vb.mapSceneToView(corner))

    def _relayout(self, *_) -> None:
        for tip in self._tips:
            self._place(tip)

    def set_orientation(self, index: int, orientation: str) -> None:
        """index. ipucunun yönü (ORIENTATIONS anahtarı); yeni ipuçlarının varsayılanı da olur."""
        self._set_orientation(self._tips[index], orientation)

    def _set_orientation(self, tip: _Tip, orientation: str, remember: bool = True) -> None:
        if orientation not in ORIENTATIONS:
            raise ValueError(f"Bilinmeyen yön: {orientation!r}")
        if orientation != tip.orientation:
            tip.orientation = orientation
            self._place(tip)
        if remember:
            app_settings().setValue(ORIENTATION_KEY, orientation)

    def orientations(self) -> list[str]:
        """İpuçlarının tercih edilen yönleri (eklenme sırasıyla)."""
        return [tip.orientation for tip in self._tips]

    def shown_orientations(self) -> list[str]:
        """Gösterilen yönler: tercih grafiğe sığmıyorsa çevrilmiş hâli."""
        return [tip.shown for tip in self._tips]

    def _drag(self, label: pg.TextItem, scene_pos: QPointF, finish: bool) -> None:
        """Etiket sürükleniyor: imlecin işarete göre bulunduğu çeyrek yeni yön olur (MATLAB'daki gibi yapışır)."""
        tip = next((t for t in self._tips if t.label is label), None)
        if tip is None:
            return
        m = self._plot.getPlotItem().getViewBox().mapViewToScene(tip.point)
        orientation = ("n" if scene_pos.y() < m.y() else "s") + ("e" if scene_pos.x() >= m.x() else "w")
        self._set_orientation(tip, orientation, remember=finish)

    # ---------------------------------------------------------------- bağlam menüsü
    def menu_for(self, index: int) -> QMenu:
        """index. ipucunun sağ tık menüsü: Konum (dört yön), bu ipucunu kaldır, tümünü temizle."""
        tip = self._tips[index]
        menu = QMenu(self._plot)
        where = menu.addMenu("Konum")
        group = QActionGroup(menu)
        for key, text in ORIENTATIONS.items():
            act = where.addAction(text)
            act.setCheckable(True)
            act.setChecked(key == tip.orientation)
            act.setData(key)
            group.addAction(act)
            act.triggered.connect(lambda _=False, tip=tip, key=key: self._set_orientation(tip, key))
        menu.addSeparator()
        menu.addAction("Bu ipucunu kaldır", lambda tip=tip: self._remove(tip))
        menu.addAction("Tüm ipuçlarını temizle", self.clear)
        return menu

    def _popup_menu(self, item, screen_pos) -> None:
        tip = next((t for t in self._tips if item in (t.marker, t.label)), None)
        if tip is None:
            return
        self._timer.stop()      # sağ tıktan önceki bekleyen sol tık işlenmesin
        self._pending = None
        self._menu = self.menu_for(self._tips.index(tip))
        self._menu.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)   # her sağ tıkta yeni menü: birikmesin
        self._menu.popup(screen_pos.toPoint())

    # ---------------------------------------------------------------- kaldırma
    def _remove(self, tip: _Tip) -> None:
        if tip in self._tips:
            self._plot.removeItem(tip.marker)
            self._plot.removeItem(tip.label)
            self._tips.remove(tip)

    def _remove_near(self, scene_pos: QPointF) -> bool:
        vb = self._plot.getPlotItem().getViewBox()
        for tip in self._tips:
            if (vb.mapViewToScene(tip.point) - scene_pos).manhattanLength() <= TIP_HIT_PX * 1.5:
                self._remove(tip)
                return True
        return False

    def clear(self) -> None:
        for tip in self._tips:
            self._plot.removeItem(tip.marker)
            self._plot.removeItem(tip.label)
        self._tips.clear()

    @property
    def count(self) -> int:
        return len(self._tips)

    def texts(self) -> list[str]:
        return [tip.label.textItem.toPlainText() for tip in self._tips]


# ============================================================================ analiz tablosu
class AnalysisTable(QTableWidget):
    """Görünen aralık istatistikleri: satırlar ölçüm, sütunlar eğri."""

    def __init__(self, theme: ThemeManager):
        super().__init__()
        self._theme = theme
        self.setObjectName("AnalysisTable")
        self.verticalHeader().setVisible(False)
        self.setShowGrid(False)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ContiguousSelection)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft)
        self.verticalHeader().setDefaultSectionSize(24)
        self.setMinimumWidth(240)
        self.setToolTip("Görünen aralığın istatistikleri — yakınlaştırınca / kaydırınca güncellenir. "
                        "Hücreler seçilip Ctrl+C ile kopyalanabilir.")

    def show_columns(self, columns: list[tuple[str, str, list[tuple[str, str]]]]) -> None:
        """columns: (başlık, renk, [(ölçüm adı, değer metni)…])."""
        rows: list[str] = []
        for _, _, items in columns:
            for name, _ in items:
                if name not in rows:
                    rows.append(name)
        self.setRowCount(len(rows))
        self.setColumnCount(len(columns) + 1)
        self.setHorizontalHeaderLabels(["Ölçüm"] + [title for title, _, _ in columns])
        muted = QColor(self._theme.tokens.text_muted)
        for r, name in enumerate(rows):
            item = QTableWidgetItem(name)
            item.setForeground(muted)
            self.setItem(r, 0, item)
        for c, (title, color, items) in enumerate(columns, 1):
            values = dict(items)
            for r, name in enumerate(rows):
                item = QTableWidgetItem(values.get(name, ""))
                item.setToolTip(f"{title} · {name}: {values.get(name, '')}")
                self.setItem(r, c, item)
            header = self.horizontalHeaderItem(c)
            if header is not None:
                header.setForeground(QColor(color))
                header.setToolTip(title)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.horizontalHeader().setStretchLastSection(True)

    def keyPressEvent(self, event) -> None:
        if event.matches(QKeySequence.StandardKey.Copy):  # seçili hücreleri sekmeyle ayrılmış metin olarak kopyala
            ranges = self.selectedRanges()
            if ranges:
                r = ranges[0]
                lines = ["\t".join((self.item(i, j).text() if self.item(i, j) else "")
                                   for j in range(r.leftColumn(), r.rightColumn() + 1))
                         for i in range(r.topRow(), r.bottomRow() + 1)]
                QApplication.clipboard().setText("\n".join(lines))
            return
        super().keyPressEvent(event)


class PlotAnalysis(QObject):
    """Kart başlığına "Analiz" düğmesi ve gövdesine açılır istatistik tablosu ekler.

    `provider(x0, x1)` görünen gerçek x aralığı için sütunları döndürür. Görünüm değiştikçe (yakınlaştırma,
    kaydırma, canlı akış) en fazla saniyede 4 kez güncellenir.
    """

    def __init__(self, card, plot: pg.PlotWidget, theme: ThemeManager,
                 provider: Callable[[float, float], list[tuple[str, str, list[tuple[str, str]]]]],
                 content=None):
        """`content`: kartta tablonun yanına alınacak widget (varsayılan: grafiğin kendisi; birden çok grafik
        içeren bir kap da olabilir — görünen aralık yine `plot`tan okunur)."""
        super().__init__(plot)
        from spt.gui.widgets.common import icon_button

        self._plot = plot
        self._provider = provider
        self.table = AnalysisTable(theme)
        self.table.hide()
        content = plot if content is None else content
        body = card.body
        index = body.indexOf(content)
        body.removeWidget(content)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(content)
        self.splitter.addWidget(self.table)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 1)
        body.insertWidget(max(index, 0), self.splitter, 1)
        self.button = icon_button("ph.table", "Ayrıntılı analiz: görünen aralığın istatistikleri", object_name="FlatIcon",
                                  size=16)
        self.button.setCheckable(True)
        self.button.toggled.connect(self._toggle)
        card.add_right(self.button)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self.update)
        self._last = 0.0       # son güncellemenin zamanı (monotonic)
        self._cost = 0.0       # son güncellemenin süresi (s)
        plot.getPlotItem().getViewBox().sigRangeChanged.connect(lambda *_: self.schedule())

    @property
    def active(self) -> bool:
        return self.button.isChecked()

    def _toggle(self, on: bool) -> None:
        self.table.setVisible(on)
        if on:
            width = self.splitter.width()
            table = int(min(560, max(260, 0.4 * width)))
            self.splitter.setSizes([max(width - table, 200), table])
            self.update()

    def schedule(self) -> None:
        """Görünüm değişince güncelle: elle yakınlaştırmada 250 ms sonra; canlı akışta (sürekli değişen aralık)
        en çok saniyede bir; pahalı tablolarda maliyetin 10 katından seyrek."""
        if not self.active or self._timer.isActive():
            return
        since = time.monotonic() - self._last
        delay = 0.25 if since >= 1.0 else max(0.25, 1.0 - since)
        self._timer.start(int(max(delay, 10 * self._cost) * 1000))

    def visible_range(self) -> tuple[float, float]:
        pi = self._plot.getPlotItem()
        x0, x1 = pi.getViewBox().viewRange()[0]
        if pi.ctrl.logXCheck.isChecked():
            x0, x1 = 10 ** x0, 10 ** x1
        return x0, x1

    def update(self) -> None:
        if not self.active:
            return
        start = time.monotonic()
        self.table.show_columns(self._provider(*self.visible_range()))
        self._last = time.monotonic()
        self._cost = self._last - start


# ============================================================================ istatistikler
def _interp_peak(mag: np.ndarray, i: int) -> tuple[float, float]:
    """Parabolik interpolasyon: (bin altı konum, tepe değeri)."""
    if 0 < i < len(mag) - 1:
        a, b, c = mag[i - 1], mag[i], mag[i + 1]
        den = a - 2 * b + c
        if den != 0 and np.isfinite(den):
            p = 0.5 * (a - c) / den
            return i + p, b - 0.25 * (a - c) * p
    return float(i), float(mag[i])


def dominant_frequency(y: np.ndarray, fs: float) -> float | None:
    """Hann pencereli FFT'nin (DC hariç) en büyük tepesinin parabolik interpolasyonlu frekansı.

    Pencerede en az 2 periyot yoksa None (kısa pencerede sonuç yanıltıcı hassasiyette olurdu). Uzun aralıkta en
    yeni MAX_DOMINANT_FFT örnek, FFT için hızlı bir uzunluğa sıfırla doldurularak kullanılır."""
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)][-MAX_DOMINANT_FFT:]
    n = len(y)
    if n < 16:
        return None
    nfft = sp_fft.next_fast_len(n, real=True)
    spec = np.abs(sp_fft.rfft((y - y.mean()) * np.hanning(n), nfft))
    if len(spec) < 3:
        return None
    i = int(np.argmax(spec[1:]) + 1)
    k, _ = _interp_peak(20 * np.log10(np.maximum(spec, 1e-300)), i)
    f = k * fs / nfft
    return f if f * n / fs >= 2.0 else None


def time_stats(y: np.ndarray, t0: float, fs: float, unit: str) -> list[tuple[str, str]]:
    """Zaman domeni istatistikleri. y: görünen aralığın örnekleri (NaN = kayıp), t0: ilk örneğin zamanı."""
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(y)
    n = int(finite.sum())
    if n == 0:
        return [("Örnek", "0")]
    v = y[finite]
    i_min, i_max = int(np.nanargmin(y)), int(np.nanargmax(y))
    mean = float(v.mean())
    rms = float(np.sqrt(np.mean(v ** 2)))
    std = float(v.std())
    peak = float(np.max(np.abs(v)))
    out = [
        ("Örnek", group(n) + ("" if n == len(y) else f" ({group(len(y) - n)} kayıp)")),
        ("Süre", fmt(len(y) / fs, "s")),
        ("En küçük", fmt(float(y[i_min]), unit)),
        ("En küçük zamanı", fmt(t0 + i_min / fs, "s", 7)),
        ("En büyük", fmt(float(y[i_max]), unit)),
        ("En büyük zamanı", fmt(t0 + i_max / fs, "s", 7)),
        ("Tepe-tepe", fmt(float(v.max() - v.min()), unit)),
        ("Ortalama (DC)", fmt(mean, unit)),
        ("RMS", fmt(rms, unit)),
        ("AC RMS (std)", fmt(std, unit)),
        ("Crest faktörü", f"{peak / rms:.4g}" if rms > 0 else "—"),
    ]
    f = dominant_frequency(y if n == len(y) else v, fs)
    out.append(("Baskın frekans", fmt(f, "Hz") if f else "— (pencerede < 2 periyot)"))
    if f:
        out.append(("Periyot", fmt(1 / f, "s")))
    return out


def spectrum_stats(freqs: np.ndarray, mag_db: np.ndarray) -> list[tuple[str, str]]:
    f = np.asarray(freqs, dtype=np.float64)
    m = np.asarray(mag_db, dtype=np.float64)
    ok = np.isfinite(m)
    if ok.sum() < 3:
        return [("Nokta", str(int(ok.sum())))]
    f, m = f[ok], m[ok]
    df = f[1] - f[0] if len(f) > 1 else 0.0
    away_from_dc = np.abs(f) > 3 * abs(df)  # DC ve pencerenin ana lobu (DC sızıntısı) tepe sayılmaz
    search = np.where(away_from_dc)[0] if away_from_dc.sum() >= 3 else np.arange(len(f))
    i = int(search[np.argmax(m[search])])
    k, level = _interp_peak(m, i)
    f_peak = f[0] + k * df if np.allclose(np.diff(f[:min(len(f), 64)]), df) else f[i]
    floor = float(np.median(m))
    # İkinci tepe: ilk tepenin ±%2 komşuluğu dışındaki en büyük yerel tepe
    width = max(3, len(f) // 50)
    masked = np.where(away_from_dc, m, -np.inf) if away_from_dc.sum() >= 3 else m.copy()
    masked[max(0, i - width):i + width + 1] = -np.inf
    j = int(np.argmax(masked))
    out = [
        ("Tepe frekansı (DC hariç)", fmt(float(f_peak), "Hz", 7)),
        ("Tepe seviyesi", f"{level:.2f} dB"),
        ("Tepe (doğrusal)", f"{10 ** (level / 20):.5g}"),
        ("Gürültü tabanı (medyan)", f"{floor:.1f} dB"),
        ("Tepe / taban", f"{level - floor:.1f} dB"),
    ]
    if np.isfinite(masked[j]):
        out.append(("2. tepe frekansı", fmt(float(f[j]), "Hz", 6)))
        out.append(("2. tepe seviyesi", f"{m[j]:.2f} dB"))
    out.append(("Çözünürlük (bin)", fmt(float(df), "Hz") if df else "—"))
    return out


def curve_stats(x: np.ndarray, y: np.ndarray, x_unit: str, y_unit: str) -> list[tuple[str, str]]:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.any():
        return [("Nokta", "0")]
    x, y = x[ok], y[ok]
    i_min, i_max = int(np.argmin(y)), int(np.argmax(y))
    return [
        ("Nokta", group(len(x))),
        ("x aralığı", f"{fmt(float(x.min()), x_unit)} … {fmt(float(x.max()), x_unit)}"),
        ("En küçük", fmt(float(y[i_min]), y_unit)),
        ("En küçük konumu", fmt(float(x[i_min]), x_unit)),
        ("En büyük", fmt(float(y[i_max]), y_unit)),
        ("En büyük konumu", fmt(float(x[i_max]), x_unit)),
        ("Ortalama", fmt(float(y.mean()), y_unit)),
    ]


def visible_slice(x: np.ndarray, x0: float, x1: float) -> slice:
    """Sıralı x dizisinde [x0, x1] aralığının dilimi."""
    return slice(int(np.searchsorted(x, x0, "left")), int(np.searchsorted(x, x1, "right")))
