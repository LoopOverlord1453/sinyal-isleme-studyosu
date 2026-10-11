"""Çalışma alanı: dondurulmuş anlık görüntüler ve bunlardan üretilen (filtrelenmiş) sinyaller.

Dondur (F6) canlı akışın tampondaki verisini buraya kopyalar; canlı akış devam ederken kopya değişmez.
Seçilen anlık görüntüler İnceleme görünümünde (Ctrl+2) gösterilir; birden çok seçimde (ör. ham + filtreli)
aynı zaman ekseninde üst üste çizilir. İşlem zinciri seçilenlere nedensel ya da sıfır fazlı uygulanabilir.
"""
from __future__ import annotations

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QStackedLayout,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from spt.core.snapshot import Snapshot
from spt.gui.icons import icon
from spt.gui.widgets.common import hbox, label

_ID_ROLE = Qt.ItemDataRole.UserRole


class WorkspacePanel(QWidget):
    selection_changed = pyqtSignal(list)    # list[Snapshot]
    review_requested = pyqtSignal()
    filter_requested = pyqtSignal(bool)     # sıfır fazlı mı
    removed = pyqtSignal(list)              # silinen kimlikler

    def __init__(self, freeze_action: QAction | None = None):
        super().__init__()
        self.snapshots: dict[int, Snapshot] = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        holder = QWidget()
        self._stack = QStackedLayout(holder)
        empty = label("Henüz sinyal yok.\n\nCanlı akışı Dondur (F6) ile buraya alın; dondurulan veri canlı akış "
                      "devam ederken değişmeden incelenir ve filtrelenebilir.", "EmptyState")
        empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty.setWordWrap(True)
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list.setIconSize(QSize(16, 16))
        self.list.setWordWrap(True)
        self.list.itemSelectionChanged.connect(self._on_selection)
        self.list.itemDoubleClicked.connect(lambda *_: self.review_requested.emit())
        self._stack.addWidget(empty)
        self._stack.addWidget(self.list)
        lay.addWidget(holder, 1)

        self.review = QPushButton(icon("ph.magnifying-glass"), " İncele")
        self.review.setToolTip("Seçilenleri İnceleme görünümünde aç (Ctrl+2)")
        self.review.clicked.connect(self.review_requested.emit)
        self.filter = QToolButton()
        self.filter.setObjectName("MenuButton")
        self.filter.setText("Filtrele")
        self.filter.setIcon(icon("ph.funnel"))
        self.filter.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.filter.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.filter.setToolTip("İşlem zincirini seçilen anlık görüntülere uygula; sonuç yeni bir öğe olarak eklenir")
        menu = QMenu(self)
        causal = menu.addAction("Nedensel (canlı filtreyle aynı)")
        causal.setToolTip("sosfilt / FIR konvolüsyonu; gecikme ve faz kayması canlıdaki gibi")
        causal.triggered.connect(lambda: self.filter_requested.emit(False))
        zero = menu.addAction("Sıfır fazlı (ileri + geri)")
        zero.setToolTip("sosfiltfilt: gecikme ve faz kayması yok; genlik cevabının karesi uygulanır")
        zero.triggered.connect(lambda: self.filter_requested.emit(True))
        self.filter.setMenu(menu)
        self.delete = QPushButton(icon("ph.trash"), "")
        self.delete.setToolTip("Seçilenleri çalışma alanından sil")
        self.delete.clicked.connect(self.remove_selected)
        buttons = [self.review, self.filter, self.delete]
        if freeze_action is not None:
            freeze = QToolButton()
            freeze.setObjectName("IconButton")
            freeze.setDefaultAction(freeze_action)
            buttons.insert(0, freeze)
        for b in buttons:
            b.setCursor(Qt.CursorShape.PointingHandCursor)
        lay.addWidget(hbox(*buttons, spacing=6, stretch=True))
        self._update()

    # ---------------------------------------------------------------- öğeler
    def add(self, snap: Snapshot, select: bool = False) -> None:
        self.snapshots[snap.id] = snap
        item = QListWidgetItem(icon("ph.funnel" if snap.parent is not None else "ph.snowflake"),
                               f"{snap.name}\n{snap.details()}")
        item.setData(_ID_ROLE, snap.id)
        tip = [snap.name, snap.details(), f"Başlangıç: t = {snap.start_time:.6f} s",
               f"Alındı: {snap.created:%H:%M:%S}"]
        if snap.note:
            tip.append(snap.note)
        item.setToolTip("\n".join(tip))
        self.list.addItem(item)
        if select:
            item.setSelected(True)
        self._update()

    def select_ids(self, ids: list[int]) -> None:
        self.list.blockSignals(True)
        self.list.clearSelection()
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.data(_ID_ROLE) in ids:
                item.setSelected(True)
                self.list.scrollToItem(item)
        self.list.blockSignals(False)
        self._on_selection()

    def selected(self) -> list[Snapshot]:
        ids = [item.data(_ID_ROLE) for item in self.list.selectedItems()]
        return [self.snapshots[i] for i in sorted(ids, key=self._row_of) if i in self.snapshots]

    def _row_of(self, snap_id: int) -> int:
        for i in range(self.list.count()):
            if self.list.item(i).data(_ID_ROLE) == snap_id:
                return i
        return -1

    def remove_selected(self) -> None:
        ids = [item.data(_ID_ROLE) for item in self.list.selectedItems()]
        if not ids:
            return
        self.list.blockSignals(True)
        for item in self.list.selectedItems():
            self.list.takeItem(self.list.row(item))
        self.list.blockSignals(False)
        for i in ids:
            self.snapshots.pop(i, None)
        self.removed.emit(ids)
        self._on_selection()
        self._update()

    def _on_selection(self) -> None:
        self._update()
        self.selection_changed.emit(self.selected())

    def _update(self) -> None:
        has_items = self.list.count() > 0
        self._stack.setCurrentIndex(1 if has_items else 0)
        selected = bool(self.list.selectedItems())
        self.review.setEnabled(selected)
        self.filter.setEnabled(selected)
        self.delete.setEnabled(selected)
