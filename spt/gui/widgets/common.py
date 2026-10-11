"""Tema ile uyumlu, tekrar kullanılan küçük arayüz bileşenleri."""
from __future__ import annotations

from collections.abc import Iterable

from PyQt6.QtCore import QPoint, QRect, QSize, Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QSizePolicy,
    QStyledItemDelegate,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from spt.gui.icons import icon
from spt.gui.theme import TERRAFLUX_DARK


def label(text: str, object_name: str | None = None) -> QLabel:
    w = QLabel(text)
    if object_name:
        w.setObjectName(object_name)
    return w


def chip(text: str) -> QLabel:
    return label(text, "Chip")


def demo_chip() -> QLabel:
    w = label("DEMO", "DemoChip")
    w.setToolTip("Örnek (sentetik) veri: bu bölüm henüz canlı akışa bağlı değil — Aşama 2'de bağlanacak.")
    return w


def legend_chip(color: str, name: str) -> QLabel:
    """Seri lejantı: renk çizgisi + metin rengiyle ad (kimlik yalnızca renge bağlı kalmaz)."""
    return label(f'<span style="color:{color}">▬</span>&nbsp;{name}', "Readout")


def icon_button(icon_name: str | None = None, tip: str = "", action: QAction | None = None,
                object_name: str = "IconButton", size: int = 18) -> QToolButton:
    btn = QToolButton()
    btn.setObjectName(object_name)
    btn.setIconSize(QSize(size, size))
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    if action is not None:
        btn.setDefaultAction(action)
    else:
        btn.setIcon(icon(icon_name))
        btn.setToolTip(tip)
    return btn


class _Combo(QComboBox):
    """Yer varken içeriği kadar geniş; dar panelde içeriğinin altına daralabilir (metin kırpılırsa tamamı ipucunda).

    Varsayılan QComboBox en küçük genişliği en uzun öğe kadar tutar: dar ayar panelinde satır görünür alandan taşar
    ve sağı kesilir."""
    MIN_CHARS = 8
    _auto_tip = ""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.currentIndexChanged.connect(self._sync_tip)

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        hint = super().minimumSizeHint()
        floor = self.fontMetrics().horizontalAdvance("M" * self.MIN_CHARS) + 36  # ok ve dolgu payı
        return QSize(min(hint.width(), floor), hint.height())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._sync_tip()

    def _sync_tip(self, *_) -> None:
        auto = self.currentText() if self.width() < self.sizeHint().width() else ""
        if self.toolTip() in ("", self._auto_tip):  # kodun verdiği açıklayıcı ipucuna dokunulmaz
            self.setToolTip(auto)
        self._auto_tip = auto


def combo(items: Iterable[str], current: int = 0) -> QComboBox:
    """QSS'teki öğe dolgusunun uygulanması için QStyledItemDelegate kullanan açılır liste."""
    w = _Combo()
    w.setItemDelegate(QStyledItemDelegate(w))
    w.addItems(list(items))
    w.setCurrentIndex(current)
    return w


def form_layout() -> QFormLayout:
    form = QFormLayout()
    form.setContentsMargins(4, 4, 4, 8)
    form.setHorizontalSpacing(12)
    form.setVerticalSpacing(8)
    form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)  # dar panelde alan etiketin altına iner, kesilmez
    return form


def add_form_row(form: QFormLayout, text: str, widget: QWidget) -> None:
    form.addRow(label(text, "FormLabel"), widget)


class Card(QFrame):
    """Başlık satırı ve gövdesi olan yuvarlatılmış yüzey.

    Başlık hiçbir zaman kesilmez: kart daraldığında `optional=True` ile eklenen öğeler (okuma, lejant gibi bilgi
    satırları) sondan başlayarak gizlenir, genişleyince geri gelir. Düğmeler / seçim kutuları her zaman kalır."""

    def __init__(self, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Card")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 12, 16, 12)
        outer.setSpacing(8)

        self._header = QHBoxLayout()
        self._header.setSpacing(10)
        self.title = label(title, "CardTitle")
        self.title.setMinimumWidth(self.title.sizeHint().width())
        self._header.addWidget(self.title)
        self._header.addStretch(1)
        outer.addLayout(self._header)
        self._optional: list[QWidget] = []     # eklenme sırasıyla
        self._auto_hidden: list[QWidget] = []  # yer kalmadığı için gizlenenler (son gizlenen sonda)

        self.body = QVBoxLayout()
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(8)
        outer.addLayout(self.body, 1)

    def add_left(self, widget: QWidget, optional: bool = False) -> None:
        """Başlığın hemen sağına (sol grupta) ekler."""
        self._header.insertWidget(self._header.count() - 1, widget)
        if optional:
            self._make_optional(widget)

    def add_right(self, widget: QWidget, optional: bool = False) -> None:
        self._header.addWidget(widget)
        if optional:
            self._make_optional(widget)

    def _make_optional(self, widget: QWidget) -> None:
        # En küçük genişliğe katılmasın: kartın daralmasını engellemez, geri besleme (yeniden boyut döngüsü) olmaz
        widget.setMinimumWidth(1)
        self._optional.append(widget)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._optional:
            self._fit_header()

    def _fit_header(self) -> None:
        """Sığmayan isteğe bağlı öğeleri sondan başlayarak gizler; yer açılınca geri getirir. Görünürlük yalnız
        karar değişirse değiştirilir (karar, öğelerin sizeHint'lerinden görünürlüğe dokunmadan hesaplanır)."""
        lay = self._header
        items = [lay.itemAt(i).widget() for i in range(lay.count()) if lay.itemAt(i).widget() is not None]
        shown = [w for w in items if not w.isHidden() or w in self._auto_hidden]  # uygulamanın gizledikleri hariç
        m = self.layout().contentsMargins()

        def need(hidden) -> int:
            widths = [w.sizeHint().width() for w in shown if w not in hidden]
            return sum(widths) + lay.spacing() * max(len(widths) - 1, 0) + m.left() + m.right()

        hidden: list[QWidget] = []
        for widget in reversed(self._optional):
            if need(hidden) <= self.width():
                break
            if widget in shown:
                hidden.append(widget)
        for widget in self._optional:
            if widget in hidden and not widget.isHidden():
                widget.setVisible(False)
                self._auto_hidden.append(widget)
            elif widget not in hidden and widget in self._auto_hidden:
                self._auto_hidden.remove(widget)
                widget.setVisible(True)


class PageHeader(QWidget):
    """Sayfa başlığı + açıklama, sağda kontroller (TerraFlux 'Project Sites' satırı)."""

    def __init__(self, title: str, subtitle: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 4, 0, 4)
        lay.setSpacing(8)
        col = QVBoxLayout()
        col.setSpacing(2)
        col.addWidget(label(title, "PageTitle"))
        self.subtitle = label(subtitle, "Muted")
        col.addWidget(self.subtitle)
        lay.addLayout(col)
        lay.addStretch(1)
        self._lay = lay

    def add_right(self, widget: QWidget) -> None:
        self._lay.addWidget(widget, 0, Qt.AlignmentFlag.AlignVCenter)


class StatusBadge(QLabel):
    """Sol kenarında durum çizgisi ve noktası olan rozet (ör. ● Aktif)."""

    COLORS = {
        "good": TERRAFLUX_DARK.good,
        "warning": TERRAFLUX_DARK.warning,
        "critical": TERRAFLUX_DARK.critical,
        "idle": TERRAFLUX_DARK.text_muted,
    }

    def __init__(self, text: str, state: str = "idle", parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Badge")
        self.set_state(text, state)

    def set_state(self, text: str, state: str) -> None:
        if (text, state) == getattr(self, "_current", None):
            return
        self._current = (text, state)
        self.setProperty("state", state)
        color = self.COLORS.get(state, TERRAFLUX_DARK.text_muted)
        self.setText(f'<span style="color:{color}">●</span>&nbsp;&nbsp;{text}')
        self.style().unpolish(self)
        self.style().polish(self)


def num_chip(text: str) -> QLabel:
    w = label(text, "NumChip")
    w.setFixedSize(22, 22)
    w.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return w


def hbox(*widgets: QWidget, spacing: int = 8, margins: tuple[int, int, int, int] = (0, 0, 0, 0),
         stretch: bool = False) -> QWidget:
    """Widget'ları yatay sırada tutan kap; `stretch` ile sola yaslanır."""
    host = QWidget()
    lay = QHBoxLayout(host)
    lay.setContentsMargins(*margins)
    lay.setSpacing(spacing)
    for w in widgets:
        lay.addWidget(w)
    if stretch:
        lay.addStretch(1)
    return host


def cell(*widgets: QWidget, spacing: int = 8) -> QWidget:
    """Tablo hücresine sola hizalı widget grubu yerleştirmek için kap."""
    return hbox(*widgets, spacing=spacing, margins=(8, 0, 8, 0), stretch=True)


class DockTitleBar(QFrame):
    """Kart görünümlü panel başlığı: ikon, başlık, ek rozetler, ayrı pencerede aç / kapat düğmeleri."""

    def __init__(self, dock: QDockWidget, icon_name: str, on_detach=None):
        super().__init__(dock)
        self.setObjectName("DockTitle")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 10, 8, 6)
        lay.setSpacing(8)

        ico = QLabel()
        ico.setPixmap(icon(icon_name, TERRAFLUX_DARK.text_muted).pixmap(16, 16))
        lay.addWidget(ico)
        lay.addWidget(label(dock.windowTitle(), "DockTitleLabel"))
        lay.addStretch(1)
        self._extras = lay

        if on_detach is not None:
            float_btn = icon_button("ph.arrow-square-out", "Ayrı pencerede aç (başka ekranda izlemek için)",
                                    object_name="FlatIcon", size=14)
            float_btn.clicked.connect(on_detach)
        else:
            float_btn = icon_button("ph.arrow-square-out", "Paneli ayır / yerleştir", object_name="FlatIcon", size=14)
            float_btn.clicked.connect(lambda: dock.setFloating(not dock.isFloating()))
        close_btn = icon_button("ph.x", "Paneli kapat", object_name="FlatIcon", size=14)
        close_btn.clicked.connect(dock.close)
        lay.addWidget(float_btn)
        lay.addWidget(close_btn)

    def add_widget(self, widget: QWidget) -> None:
        self._extras.insertWidget(self._extras.count() - 3, widget)


class _SectionHeader(QToolButton):
    """Bölüm başlığı: dar panelde metin “…” ile kısalır (tamamı ipucunda); uzun başlık paneli genişletmez."""

    def __init__(self, title: str):
        super().__init__()
        self.title = title
        self.setText(title)
        self.setMinimumWidth(1)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        room = self.width() - self.iconSize().width() - 22  # simge, dolgu ve aradaki boşluk
        text = self.fontMetrics().elidedText(self.title, Qt.TextElideMode.ElideRight, max(room, 24))
        if text != self.text():
            self.setText(text)
            self.setToolTip(self.title if text != self.title else "")


class CollapsibleSection(QWidget):
    """Başlığına tıklanınca açılıp kapanan form bölümü."""

    def __init__(self, title: str, expanded: bool = True, parent: QWidget | None = None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)

        self.header = _SectionHeader(title)
        self.header.setObjectName("SectionHeader")
        self.header.setCheckable(True)
        self.header.setChecked(expanded)
        self.header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.header.setIconSize(QSize(12, 12))
        self.header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.header.setCursor(Qt.CursorShape.PointingHandCursor)
        self.header.toggled.connect(self._toggle)
        lay.addWidget(self.header)

        self.content = QWidget()
        self.form = form_layout()
        self.content.setLayout(self.form)
        lay.addWidget(self.content)
        self._toggle(expanded)

    def _toggle(self, expanded: bool) -> None:
        self.content.setVisible(expanded)
        self.header.setIcon(icon("ph.caret-down" if expanded else "ph.caret-right", TERRAFLUX_DARK.text_muted))

    def add_row(self, text: str, widget: QWidget) -> None:
        add_form_row(self.form, text, widget)


class FlowLayout(QLayout):
    """Öğeleri satıra sığdığı kadar dizip taşanı alt satıra kaydıran yerleşim (lejantlar için: dar pencerede metin
    kesilmez, satır sayısı artar)."""

    def __init__(self, parent: QWidget | None = None, spacing: int = 8, uniform: bool = False):
        super().__init__(parent)
        self._items: list = []
        self._spacing = spacing
        self._uniform = uniform  # tüm hücreler en genişi kadar: sığdığı kadar sütun, hizalı
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item) -> None:  # noqa: N802 (Qt adı)
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):  # noqa: N802
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):  # noqa: N802
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):  # noqa: N802
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._arrange(width, apply=False)

    def setGeometry(self, rect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._arrange(rect.width(), apply=True, origin=(rect.x(), rect.y()))

    def sizeHint(self) -> QSize:  # noqa: N802
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        return size

    def _arrange(self, width: int, apply: bool, origin: tuple[int, int] = (0, 0)) -> int:
        x0, y0 = origin
        x, y, line = x0, y0, 0
        shown = [i for i in self._items if i.widget() is None or not i.widget().isHidden()]
        cell = max((i.sizeHint().width() for i in shown), default=0) if self._uniform else 0
        for item in shown:
            hint = item.sizeHint()
            if cell:
                hint = QSize(min(cell, width), hint.height())
            if x > x0 and x + hint.width() > x0 + width:
                x, y, line = x0, y + line + self._spacing, 0
            if apply:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width() + self._spacing
            line = max(line, hint.height())
        return y + line - y0
