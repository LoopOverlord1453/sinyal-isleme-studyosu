"""Blok diyagram düzenleyicisinin blok kütüphanesi (Simulink Library Browser karşılığı).

Bloklar motorun blok kaydından (`block_catalog`) okunur ve kategorilere göre gruplanır; arama kutusu Türkçe ad,
kalıcı tür adı, açıklama ve Simulink karşılığı içinde arar (Türkçe karakter ve büyük / küçük harf duyarsız: "integ"
→ "İntegratör"). Bir blok tuvale sürüklenerek ya da çift tıklanarak (görünümün ortasına) eklenir.
"""
from __future__ import annotations

from PyQt6.QtCore import QMimeData, Qt, pyqtSignal
from PyQt6.QtGui import QDrag
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QLineEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from spt.gui.icons import icon
from spt.gui.widgets.common import label

MIME_BLOCK = "application/x-spt-diagram-block"
_TYPE_ROLE = Qt.ItemDataRole.UserRole
_FOLD = str.maketrans({"İ": "i", "I": "i", "ı": "i", "ö": "o", "Ö": "o", "ü": "u", "Ü": "u", "ş": "s", "Ş": "s",
                       "ç": "c", "Ç": "c", "ğ": "g", "Ğ": "g", "â": "a", "î": "i", "û": "u"})


def fold(text: str) -> str:
    """Aramada karşılaştırma için Türkçe karakterleri ASCII'ye indirip küçültür."""
    return text.translate(_FOLD).lower()


def catalog_entries() -> list[dict]:
    """Kütüphanedeki bloklar (kategori sırasıyla, kategori içinde Türkçe ada göre)."""
    from spt.control.diagram.block import CATEGORIES, block_catalog

    order = {k: i for i, k in enumerate(CATEGORIES)}
    entries = block_catalog()
    return sorted(entries, key=lambda e: (order.get(e["category"], len(order)), e["category_label"],
                                          fold(e["display_name"])))


class _Tree(QTreeWidget):
    """Sürükleme verisi blok türü olan ağaç."""

    def startDrag(self, actions) -> None:  # noqa: N802 (Qt adı)
        item = self.currentItem()
        type_name = item.data(0, _TYPE_ROLE) if item is not None else None
        if not type_name:
            return
        mime = QMimeData()
        mime.setData(MIME_BLOCK, type_name.encode("utf-8"))
        mime.setText(type_name)
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.CopyAction)


class BlockPalette(QWidget):
    """Arama kutulu kategori ağacı. `block_activated(tür)`: çift tık / Enter ile ekleme isteği."""

    block_activated = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        box = QFrame()
        box.setObjectName("SearchBox")
        bl = QHBoxLayout(box)
        bl.setContentsMargins(8, 0, 6, 0)
        bl.setSpacing(4)
        glyph = label("")
        glyph.setPixmap(icon("ph.magnifying-glass").pixmap(14, 14))
        bl.addWidget(glyph)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Blok ara (ör. PID, integratör, doyum)")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self.filter)
        self.search.returnPressed.connect(self._activate_first)
        bl.addWidget(self.search, 1)
        lay.addWidget(box)

        self.tree = _Tree()
        self.tree.setHeaderHidden(True)
        self.tree.setDragEnabled(True)
        self.tree.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.setIndentation(14)
        self.tree.itemActivated.connect(self._activated)
        lay.addWidget(self.tree, 1)

        self.hint = label("Sürükleyip tuvale bırakın ya da çift tıklayın.", "Muted")
        self.hint.setWordWrap(True)
        lay.addWidget(self.hint)

        self._items: list[tuple[QTreeWidgetItem, str]] = []   # (öğe, aranacak metin)
        self._groups: dict[str, QTreeWidgetItem] = {}
        self.reload()

    # ---------------------------------------------------------------- içerik
    def reload(self) -> None:
        """Kütüphaneyi blok kaydından yeniden okur (eklenti blokları sonradan kaydolduysa)."""
        self.tree.clear()
        self._items.clear()
        self._groups.clear()
        for e in catalog_entries():
            cat = e["category"]
            group = self._groups.get(cat)
            if group is None:
                group = QTreeWidgetItem([e["category_label"]])
                group.setFlags(Qt.ItemFlag.ItemIsEnabled)
                f = group.font(0)
                f.setBold(True)
                group.setFont(0, f)
                self.tree.addTopLevelItem(group)
                self._groups[cat] = group
            item = QTreeWidgetItem([e["display_name"]])
            item.setData(0, _TYPE_ROLE, e["type"])
            tip = e["description"] or e["display_name"]
            if e.get("simulink"):
                tip += f"\nSimulink karşılığı: {e['simulink']}"
            if e.get("lazy"):
                tip += "\n(eklenti bloğu: ilk kullanımda yüklenir)"
            item.setToolTip(0, tip)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsDragEnabled)
            group.addChild(item)
            haystack = " ".join([e["display_name"], e["type"], e["description"], e.get("simulink", ""),
                                 e["category_label"]])
            self._items.append((item, fold(haystack)))
        self.tree.expandAll()
        self.filter(self.search.text())

    def types(self) -> list[str]:
        return [it.data(0, _TYPE_ROLE) for it, _ in self._items]

    def visible_types(self) -> list[str]:
        return [it.data(0, _TYPE_ROLE) for it, _ in self._items if not it.isHidden()]

    def filter(self, text: str) -> None:
        words = fold(text).split()
        for item, hay in self._items:
            item.setHidden(bool(words) and not all(w in hay for w in words))
        for group in self._groups.values():
            shown = any(not group.child(i).isHidden() for i in range(group.childCount()))
            group.setHidden(not shown)
            if words and shown:
                group.setExpanded(True)

    # ---------------------------------------------------------------- olaylar
    def _activated(self, item: QTreeWidgetItem, _column: int = 0) -> None:
        type_name = item.data(0, _TYPE_ROLE)
        if type_name:
            self.block_activated.emit(type_name)

    def _activate_first(self) -> None:
        for item, _ in self._items:
            if not item.isHidden():
                self._activated(item)
                return
