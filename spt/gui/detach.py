"""Grafik kartlarını ve panelleri bağımsız pencerelerde açma (ör. başka bir ekranda izlemek için).

Ayrılan içerik, gerçek bir üst düzey pencereye taşınır: görev çubuğunda görünür, başka ekrana sürüklenebilir,
büyütülebilir, F11 ile tam ekran olur. Pencere kapatılınca içerik ana penceredeki yerine geri döner.
Hangi öğelerin ayrık olduğu ve pencere konumları QSettings'te saklanır; uygulama yeniden açılınca geri yüklenir.
"""
from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import QObject, QSettings, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QCloseEvent, QKeySequence, QShortcut, QShowEvent
from PyQt6.QtWidgets import QDockWidget, QFrame, QHBoxLayout, QPushButton, QSplitter, QToolButton, QVBoxLayout, QWidget

from spt import APP_NAME
from spt.gui.icons import app_icon, icon, primary_icon
from spt.gui.theme import Tokens, style_native_titlebar
from spt.gui.widgets.common import Card, icon_button, label

PLACEHOLDER_HEIGHT = 56


class DetachedWindow(QWidget):
    closing = pyqtSignal()

    def __init__(self, title: str, content: QWidget, tokens: Tokens, actions: list[QAction]):
        super().__init__(None, Qt.WindowType.Window)
        self.setObjectName("DetachedWindow")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setWindowTitle(f"{title} — {APP_NAME}")
        self.setWindowIcon(app_icon())
        self._tokens = tokens
        self._styled = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.addWidget(content)
        content.show()
        # Ana penceredeki kısayollar (F5, F7, Ctrl+R…) bu pencerede de çalışsın
        self.addActions(actions)
        QShortcut(QKeySequence("F11"), self, activated=self.toggle_fullscreen)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, activated=self._leave_fullscreen)

    def toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _leave_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()

    def take_content(self) -> QWidget | None:
        item = self.layout().takeAt(0)
        return item.widget() if item else None

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        if not self._styled:
            style_native_titlebar(self, self._tokens)
            self._styled = True

    def closeEvent(self, event: QCloseEvent) -> None:
        self.closing.emit()  # yönetici içeriği ana penceredeki yerine geri taşır
        super().closeEvent(event)


class _Placeholder(QFrame):
    """Ayrılan öğenin yerinde duran ince şerit: durumu gösterir, geri alma / pencereyi öne getirme sağlar."""

    def __init__(self, title: str, on_attach: Callable[[], None], on_show: Callable[[], None]):
        super().__init__()
        self.setObjectName("Card")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 8, 10, 8)
        lay.setSpacing(10)
        glyph = label("")
        glyph.setPixmap(icon("ph.app-window").pixmap(16, 16))
        lay.addWidget(glyph)
        self._title = label(f"<b>{title}</b>&nbsp;&nbsp;ayrı pencerede açık", "Muted")
        self._title.setWordWrap(True)
        lay.addWidget(self._title, 1)
        show = QPushButton(icon("ph.eye"), "  Pencereyi göster")
        show.clicked.connect(on_show)
        back = QPushButton(primary_icon("ph.arrow-square-in"), "  Buraya geri al")
        back.setObjectName("Primary")
        back.clicked.connect(on_attach)
        self._buttons = ((show, "  Pencereyi göster"), (back, "  Buraya geri al"))
        for b, text in self._buttons:
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setToolTip(text.strip())
            lay.addWidget(b)

    def set_compact(self, on: bool) -> None:
        """Yan yana bölücüde (dar sütun) düğmeler yalnız ikon: yer tutucu kartın genişliğini işgal etmesin."""
        for b, text in self._buttons:
            b.setText("" if on else text)


class _CardSlot:
    """Bir bölücü (QSplitter) veya yerleşim (layout) içindeki kartı yer tutucuyla değiştirir."""

    def __init__(self, card: QWidget, placeholder: QWidget):
        self.card = card
        self.placeholder = placeholder
        self._splitter: QSplitter | None = None
        self._sizes: list[int] = []
        self._container: QWidget | None = None

    def detach(self) -> QWidget:
        parent = self.card.parentWidget()
        if isinstance(parent, QSplitter):
            self._splitter = parent
            self._sizes = parent.sizes()
            index = parent.indexOf(self.card)
            horizontal = parent.orientation() == Qt.Orientation.Horizontal
            if hasattr(self.placeholder, "set_compact"):
                self.placeholder.set_compact(horizontal)
            parent.replaceWidget(index, self.placeholder)
            sizes = parent.sizes()
            # bölücü yönündeki boy: dikeyde ince şerit, yatayda yer tutucunun en dar hâli (boşalan alan komşuya)
            extent = max(self.placeholder.minimumSizeHint().width(), 160) if horizontal else PLACEHOLDER_HEIGHT
            freed = max(0, sizes[index] - extent)
            sizes[index] = extent
            other = 1 if index == 0 else 0
            if other < len(sizes):
                sizes[other] += freed
            parent.setSizes(sizes)
        else:
            self._splitter = None
            self._container = parent
            parent.layout().replaceWidget(self.card, self.placeholder)
        self.placeholder.show()
        return self.card

    def attach(self, widget: QWidget) -> None:
        if self._splitter is not None:
            self._splitter.replaceWidget(self._splitter.indexOf(self.placeholder), widget)
            self._splitter.setSizes(self._sizes)
        else:
            self._container.layout().replaceWidget(self.placeholder, widget)
        self.placeholder.hide()
        widget.show()


class _DockSlot:
    """Panelin gövdesini pencereye taşır; panel gizlenir (yeniden gösterilirse yer tutucuyu gösterir)."""

    def __init__(self, dock: QDockWidget, placeholder: QWidget):
        self.dock = dock
        self.placeholder = placeholder

    def detach(self) -> QWidget:
        body = self.dock.widget()
        self.dock.setWidget(self.placeholder)
        self.dock.hide()
        return body

    def attach(self, widget: QWidget) -> None:
        self.dock.setWidget(widget)
        self.dock.show()


class _Item:
    def __init__(self, key: str, title: str, slot, action: QAction, button: QToolButton | None):
        self.key = key
        self.title = title
        self.slot = slot
        self.kind = "card" if isinstance(slot, _CardSlot) else "dock"
        self.action = action
        self.button = button
        self.window: DetachedWindow | None = None


class DetachManager(QObject):
    """Ayrılabilir öğelerin kaydı ve pencere yaşam döngüsü."""

    changed = pyqtSignal(str, bool)  # anahtar, ayrık mı

    def __init__(self, tokens: Tokens, settings: Callable[[], QSettings], shortcuts: Callable[[], list[QAction]],
                 parent: QObject | None = None):
        super().__init__(parent)
        self._tokens = tokens
        self._settings = settings
        self._shortcuts = shortcuts
        self._items: dict[str, _Item] = {}

    # ---------------------------------------------------------------- kayıt
    def register_card(self, key: str, title: str, card: Card) -> QAction:
        button = icon_button("ph.arrow-square-out", "Ayrı pencerede aç", object_name="FlatIcon", size=16)
        button.clicked.connect(lambda: self.toggle(key))
        card.add_right(button)
        placeholder = self._placeholder(key, title)
        return self._register(key, title, _CardSlot(card, placeholder), button)

    def register_dock(self, key: str, dock: QDockWidget) -> QAction:
        title = dock.windowTitle()
        return self._register(key, title, _DockSlot(dock, self._placeholder(key, title)), None)

    def _placeholder(self, key: str, title: str) -> QWidget:
        ph = _Placeholder(title, lambda: self.attach(key), lambda: self.show_window(key))
        ph.hide()
        return ph

    def _register(self, key: str, title: str, slot, button) -> QAction:
        action = QAction(icon("ph.app-window"), title, self)
        action.setCheckable(True)
        action.triggered.connect(lambda checked: self.detach(key) if checked else self.attach(key))
        self._items[key] = _Item(key, title, slot, action, button)
        return action

    def actions(self, kind: str | None = None) -> list[QAction]:
        """Menü komutları; `kind`: "card" (grafikler) veya "dock" (paneller)."""
        return [item.action for item in self._items.values() if kind is None or item.kind == kind]

    # ---------------------------------------------------------------- durum
    def is_detached(self, key: str) -> bool:
        item = self._items.get(key)
        return item is not None and item.window is not None

    def detached_keys(self) -> list[str]:
        return [k for k, item in self._items.items() if item.window is not None]

    def toggle(self, key: str) -> None:
        if self.is_detached(key):
            self.attach(key)
        else:
            self.detach(key)

    # ---------------------------------------------------------------- ayır / geri al
    def detach(self, key: str) -> None:
        item = self._items[key]
        if item.window is not None:
            self.show_window(key)
            return
        size = item.slot.card.size() if isinstance(item.slot, _CardSlot) else item.slot.dock.size()
        content = item.slot.detach()
        window = DetachedWindow(item.title, content, self._tokens,
                                [a for a in self._shortcuts() if a.shortcut() != QKeySequence("F11")])
        window.closing.connect(lambda: self._window_closing(key))
        geometry = self._settings().value(f"detached/{key}/geometry")
        if geometry is None or not window.restoreGeometry(geometry):
            window.resize(max(size.width(), 640), max(size.height(), 420) + 20)
        item.window = window
        self._set_state(item, True)
        window.show()
        window.raise_()
        window.activateWindow()

    def attach(self, key: str) -> None:
        item = self._items[key]
        if item.window is not None:
            item.window.close()  # kapanış sırasında _window_closing içeriği geri taşır

    def show_window(self, key: str) -> None:
        window = self._items[key].window
        if window is not None:
            if window.isMinimized():
                window.showNormal()
            window.raise_()
            window.activateWindow()

    def _window_closing(self, key: str) -> None:
        item = self._items[key]
        window = item.window
        if window is None:
            return
        self._settings().setValue(f"detached/{key}/geometry", window.saveGeometry())
        content = window.take_content()
        item.window = None
        if content is not None:
            item.slot.attach(content)
        window.deleteLater()
        self._set_state(item, False)

    def _set_state(self, item: _Item, detached: bool) -> None:
        item.action.blockSignals(True)
        item.action.setChecked(detached)
        item.action.blockSignals(False)
        if item.button is not None:
            item.button.setIcon(icon("ph.arrow-square-in" if detached else "ph.arrow-square-out"))
            item.button.setToolTip("Ana pencereye geri al" if detached else "Ayrı pencerede aç")
        self.changed.emit(item.key, detached)

    def attach_all(self) -> None:
        for key in self.detached_keys():
            self.attach(key)

    # ---------------------------------------------------------------- kalıcılık
    def save_open(self) -> None:
        self._settings().setValue("detached/open", self.detached_keys())

    def restore_open(self) -> None:
        keys = self._settings().value("detached/open", [], type=list)
        for key in keys:
            if key in self._items:
                self.detach(key)
