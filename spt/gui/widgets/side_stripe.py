"""Panelleri kenara gizleme: pencerenin sol / sağ (ve alt) kenarındaki panel şeritleri.

Her panelin (QDockWidget) şeritte bir düğmesi vardır: simge + ad (yan şeritlerde dikey yazı). Düğme panelin
görünürlük eylemine (`toggleViewAction`) bağlıdır: işaretliyse panel açıktır, tıklayınca panel kenara gizlenir ya da
geri açılır. Panel başka bir kenara taşınınca düğmesi de o kenarın şeridine geçer. Böylece kapatılan panel kaybolmaz,
kenarda her zaman bir tıkla geri gelir.
"""
from __future__ import annotations

from PyQt6.QtCore import QRect, QSize, Qt
from PyQt6.QtGui import QAction, QColor, QIcon, QPainter
from PyQt6.QtWidgets import (
    QDockWidget,
    QMainWindow,
    QStyle,
    QStyleOptionToolButton,
    QStylePainter,
    QTabBar,
    QToolBar,
    QToolButton,
    QWidget,
)

from spt.gui.theme import tokens

SIDES = {
    Qt.DockWidgetArea.LeftDockWidgetArea: "left",
    Qt.DockWidgetArea.RightDockWidgetArea: "right",
    Qt.DockWidgetArea.BottomDockWidgetArea: "bottom",
    Qt.DockWidgetArea.TopDockWidgetArea: "bottom",   # üst bölgede panel yok; olursa alt şeritte gösterilir
}
_AREAS = {"left": Qt.ToolBarArea.LeftToolBarArea, "right": Qt.ToolBarArea.RightToolBarArea,
          "bottom": Qt.ToolBarArea.BottomToolBarArea}
ICON_PX = 16


def behind_tab(dock: QDockWidget) -> bool:
    """Panel açık ama sekme grubunda öndeki sekme değil mi? (Qt arkadaki sekmeyi de isVisible sayar; ana pencerenin
    sekme çubuğundan anlaşılır.)"""
    window = dock.parentWidget()
    if window is None or dock.isFloating():
        return False
    for bar in window.findChildren(QTabBar, options=Qt.FindChildOption.FindDirectChildrenOnly):
        for i in range(bar.count()):
            if bar.tabText(i) == dock.windowTitle():
                return bar.isVisible() and bar.currentIndex() != i
    return False


class StripeButton(QToolButton):
    """Kenar şeridi düğmesi. Yan şeritlerde metin dikey çizilir: solda alttan yukarı, sağda yukarıdan aşağı okunur
    (Qt düğmesi metni döndüremediği için arka plan stilden, simge ve metin elle çizilir)."""

    def __init__(self, action: QAction, side: str, dock: QDockWidget | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("StripeButton")
        self.setDefaultAction(action)
        self.side = side
        self._dock = dock
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setIconSize(QSize(ICON_PX, ICON_PX))
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.setAutoRaise(True)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        """Panel açık ama sekme grubunda arkadaysa (ör. Günlük, İşlem Zinciri'nin arkasında) tıklama onu öne getirir;
        gizlemek için ikinci tıklama. Öndeyse ya da kapalıysa normal aç / kapa."""
        dock = self._dock
        if (dock is not None and event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.pos())
                and not dock.isHidden() and behind_tab(dock)):
            self.setDown(False)
            dock.raise_()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def set_side(self, side: str) -> None:
        self.side = side
        self.updateGeometry()
        self.update()

    def _text(self) -> str:
        return self.defaultAction().text().replace("&", "")

    def sizeHint(self) -> QSize:  # noqa: N802
        fm = self.fontMetrics()
        length = ICON_PX + 8 + fm.horizontalAdvance(self._text()) + 20
        thick = max(fm.height(), ICON_PX) + 14
        return QSize(length, thick) if self.side == "bottom" else QSize(thick, length)

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return self.sizeHint()

    def paintEvent(self, event) -> None:  # noqa: N802
        if self.side == "bottom":
            super().paintEvent(event)
            return
        painter = QStylePainter(self)
        opt = QStyleOptionToolButton()
        self.initStyleOption(opt)
        opt.text = ""
        opt.icon = QIcon()
        painter.drawComplexControl(QStyle.ComplexControl.CC_ToolButton, opt)   # zemin / hover / işaretli (QSS)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        w, h = self.width(), self.height()
        if self.side == "left":
            painter.translate(0, h)
            painter.rotate(-90)
        else:
            painter.translate(w, 0)
            painter.rotate(90)
        length, thick = h, w                                    # döndürülmüş koordinatlarda
        t = tokens()
        on = self.isChecked()
        state = QIcon.State.On if on else QIcon.State.Off
        pix = self.icon().pixmap(QSize(ICON_PX, ICON_PX), QIcon.Mode.Normal, state)
        painter.drawPixmap(10, (thick - ICON_PX) // 2, pix)
        painter.setPen(QColor(t.text if on else t.text_secondary))
        painter.setFont(self.font())
        painter.drawText(QRect(10 + ICON_PX + 8, 0, length - ICON_PX - 18, thick),
                         Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self._text())


class SideStripes:
    """Ana pencerenin kenar şeritleri: panelleri bulundukları kenara göre ilgili şeritte gösterir."""

    def __init__(self, window: QMainWindow):
        self._window = window
        self.bars: dict[str, QToolBar] = {}
        self.buttons: dict[QDockWidget, StripeButton] = {}
        for side, area in _AREAS.items():
            bar = QToolBar(f"Panel şeridi ({side})", window)
            bar.setObjectName(f"SideStripe_{side}")   # saveState/restoreState için benzersiz ad
            bar.setProperty("stripe", True)          # stil: QToolBar[stripe="true"] (ad saveState için benzersiz)
            bar.setProperty("side", side)            # seçili göstergenin yönü (sağ şeritte sağda)
            bar.setMovable(False)
            bar.setFloatable(False)
            bar.setOrientation(Qt.Orientation.Horizontal if side == "bottom" else Qt.Orientation.Vertical)
            bar.toggleViewAction().setEnabled(False)
            window.addToolBar(area, bar)
            bar.hide()
            self.bars[side] = bar

    def add(self, dock: QDockWidget, on_side_changed=None) -> None:
        """Paneli bulunduğu kenarın şeridine ekler; panel taşınınca düğmesi de taşınır."""
        side = SIDES.get(self._window.dockWidgetArea(dock), "left")
        button = StripeButton(dock.toggleViewAction(), side, dock)
        button.setToolTip(f"{dock.windowTitle()} — gizle / göster (panel bu kenara gizlenir; sekmede arkadaysa "
                          "öne gelir)")
        self.buttons[dock] = button
        self.bars[side].addWidget(button)
        self._sync_bars()

        def moved(area, dock=dock):
            if area == Qt.DockWidgetArea.NoDockWidgetArea:   # yüzen panel: düğmesi bulunduğu şeritte kalır
                return
            self.move(dock, SIDES.get(area, "left"))
            if on_side_changed is not None:
                on_side_changed(SIDES.get(area, "left"))

        dock.dockLocationChanged.connect(moved)
        if on_side_changed is not None:
            on_side_changed(side)

    def side_of(self, dock: QDockWidget) -> str:
        return self.buttons[dock].side

    def move(self, dock: QDockWidget, side: str) -> None:
        button = self.buttons[dock]
        if button.side == side:
            return
        old = self.bars[button.side]
        for act in old.actions():
            if old.widgetForAction(act) is button:
                old.removeAction(act)
                break
        replacement = StripeButton(dock.toggleViewAction(), side, dock)   # QToolBar widget'ı yeniden ekleyemez
        replacement.setToolTip(button.toolTip())
        button.deleteLater()
        self.buttons[dock] = replacement
        self.bars[side].addWidget(replacement)
        self._sync_bars()

    def _sync_bars(self) -> None:
        for side, bar in self.bars.items():
            used = any(b.side == side for b in self.buttons.values())
            bar.setVisible(used)
