"""İşlem Zinciri paneli: sıralı filtre aşamaları, canlı uygulama ve zinciri kaydetme / açma.

Aşamalar sürükle-bırak ya da ok düğmeleriyle sıralanır, onay kutusuyla açılıp kapatılır. Çift tık (ya da
Düzenle) aşamayı Filtre Tasarımcısı'na yükler. Zincir .sptchain (JSON) olarak kaydedilip başka oturumlarda ve
akışlarda kullanılabilir; her akışa kendi örnekleme hızıyla yeniden tasarlanarak uygulanır.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from spt.dsp.chain import ChainStage, ProcessingChain
from spt.dsp.filters import FilterError
from spt.gui.settings import app_settings
from spt.gui.theme import Tokens
from spt.gui.widgets.common import combo, hbox, icon_button, label

DISPLAY_MODES = [("Ham + filtreli", "both"), ("Yalnız filtreli", "filtered")]
_STAGE_ROLE = Qt.ItemDataRole.UserRole


class ChainPanel(QWidget):
    chain_changed = pyqtSignal(object)          # ProcessingChain (kullanıcı düzenlemesi)
    live_toggled = pyqtSignal(bool)
    display_changed = pyqtSignal(str)           # "both" | "filtered"
    edit_requested = pyqtSignal(int, object)    # aşama indeksi, FilterSpec

    def __init__(self, tokens: Tokens):
        super().__init__()
        self._tokens = tokens
        self.chain = ProcessingChain()
        self._loading = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        self.live = QCheckBox("Canlı akışa uygula")
        self.live.setToolTip("Zinciri tüm canlı / oynatılan akışların kanallarına blok blok uygular (nedensel)")
        self.live.toggled.connect(self.live_toggled.emit)
        self.display = combo([text for text, _ in DISPLAY_MODES])
        self.display.setToolTip("Ham + filtreli: ham sinyal nötr gri, filtrelenmiş sinyal kanalın renginde çizilir")
        self.display.currentIndexChanged.connect(lambda i: self.display_changed.emit(DISPLAY_MODES[i][1]))
        lay.addWidget(hbox(self.live, self.display, spacing=8))

        self.list = QListWidget()
        self.list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list.itemChanged.connect(self._on_item_changed)
        self.list.itemDoubleClicked.connect(lambda item: self._edit(self.list.row(item)))
        self.list.model().rowsMoved.connect(lambda *_: self._from_items())
        self.list.currentRowChanged.connect(lambda *_: self._update_buttons())
        empty = label("Zincir boş.\n\nFiltre Tasarımcısı'nda (Ctrl+4) bir filtre tasarlayıp “Sinyale Uygula” deyin "
                      "ya da kayıtlı bir zinciri açın.", "EmptyState")
        empty.setWordWrap(True)
        empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        holder = QWidget()
        self._stack = QStackedLayout(holder)
        self._stack.addWidget(empty)
        self._stack.addWidget(self.list)
        lay.addWidget(holder, 1)

        self.up = icon_button("ph.arrow-up", "Yukarı taşı", object_name="FlatIcon", size=16)
        self.down = icon_button("ph.arrow-down", "Aşağı taşı", object_name="FlatIcon", size=16)
        self.edit = icon_button("ph.pencil-simple", "Tasarımcıda düzenle (çift tık)", object_name="FlatIcon", size=16)
        self.remove = icon_button("ph.trash", "Aşamayı sil", object_name="FlatIcon", size=16)
        self.open = icon_button("ph.folder-open", "Zincir aç (.sptchain)", object_name="FlatIcon", size=16)
        self.save = icon_button("ph.floppy-disk", "Zinciri kaydet (.sptchain)", object_name="FlatIcon", size=16)
        self.up.clicked.connect(lambda: self._move(-1))
        self.down.clicked.connect(lambda: self._move(1))
        self.edit.clicked.connect(lambda: self._edit(self.list.currentRow()))
        self.remove.clicked.connect(self._remove)
        self.open.clicked.connect(self._open)
        self.save.clicked.connect(self._save)
        row = hbox(self.up, self.down, self.edit, self.remove, spacing=2)
        row.layout().addStretch(1)
        row.layout().addWidget(self.open)
        row.layout().addWidget(self.save)
        lay.addWidget(row)

        self.status = label("", "Readout")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(self.status)
        self._rebuild()

    # ---------------------------------------------------------------- durum
    @property
    def display_mode(self) -> str:
        return DISPLAY_MODES[self.display.currentIndex()][1]

    def set_chain(self, chain: ProcessingChain, select: int | None = None) -> None:
        """Zinciri programla ayarlar (sinyal yaymaz)."""
        self.chain = chain
        self._rebuild(select)

    def set_live(self, on: bool) -> None:
        if self.live.isChecked() != on:
            self.live.setChecked(on)

    def set_status(self, html: str) -> None:
        self.status.setText(html)

    def _commit(self, chain: ProcessingChain, select: int | None = None) -> None:
        self.chain = chain
        self._rebuild(select)
        self.chain_changed.emit(chain)

    def _rebuild(self, select: int | None = None) -> None:
        self._loading = True
        current = self.list.currentRow() if select is None else select
        self.list.clear()
        muted = QColor(self._tokens.text_muted)
        for i, stage in enumerate(self.chain.stages):
            item = QListWidgetItem(f"{i + 1}.  {stage.spec.label()}")
            item.setData(_STAGE_ROLE, stage)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsDragEnabled)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsDropEnabled)
            item.setCheckState(Qt.CheckState.Checked if stage.enabled else Qt.CheckState.Unchecked)
            item.setToolTip(stage.spec.label() + ("" if stage.enabled else " (kapalı)"))
            if not stage.enabled:
                item.setForeground(muted)
            self.list.addItem(item)
        if self.list.count():
            self.list.setCurrentRow(min(max(current, 0), self.list.count() - 1))
        self._stack.setCurrentIndex(1 if self.list.count() else 0)
        self._loading = False
        self._update_buttons()

    def _update_buttons(self) -> None:
        row, n = self.list.currentRow(), self.list.count()
        self.up.setEnabled(row > 0)
        self.down.setEnabled(0 <= row < n - 1)
        self.edit.setEnabled(row >= 0)
        self.remove.setEnabled(row >= 0)
        self.save.setEnabled(n > 0)

    # ---------------------------------------------------------------- düzenleme
    def _on_item_changed(self, item: QListWidgetItem) -> None:
        if self._loading:
            return
        row = self.list.row(item)
        enabled = item.checkState() == Qt.CheckState.Checked
        if 0 <= row < len(self.chain) and self.chain.stages[row].enabled != enabled:
            self._commit(self.chain.toggled(row, enabled), row)

    def _from_items(self) -> None:
        """Sürükle-bırak sonrası sıra: öğelerdeki aşamalardan zinciri yeniden kurar."""
        if self._loading:
            return
        stages = [self.list.item(i).data(_STAGE_ROLE) for i in range(self.list.count())]
        if tuple(stages) != self.chain.stages:
            self._commit(ProcessingChain(stages), self.list.currentRow())

    def _move(self, delta: int) -> None:
        row = self.list.currentRow()
        to = row + delta
        if 0 <= row < len(self.chain) and 0 <= to < len(self.chain):
            self._commit(self.chain.moved(row, to), to)

    def _remove(self) -> None:
        row = self.list.currentRow()
        if 0 <= row < len(self.chain):
            self._commit(self.chain.removed(row), row)

    def _edit(self, row: int) -> None:
        if 0 <= row < len(self.chain):
            self.edit_requested.emit(row, self.chain.stages[row].spec)

    def add_stage(self, stage: ChainStage) -> None:
        self._commit(self.chain.added(stage), len(self.chain))

    def replace_stage(self, index: int, stage: ChainStage) -> None:
        self._commit(self.chain.replaced(index, stage), index)

    # ---------------------------------------------------------------- dosya
    def _open(self) -> None:
        s = app_settings()
        path, _ = QFileDialog.getOpenFileName(self, "İşlem zinciri aç", str(s.value("chain/dir", str(Path.home()))),
                                              "İşlem zinciri (*.sptchain);;Tüm dosyalar (*)")
        if not path:
            return
        s.setValue("chain/dir", str(Path(path).parent))
        try:
            chain = ProcessingChain.load(Path(path))
        except FilterError as exc:
            self.set_status(f'<span style="color:{self._tokens.critical}">●</span> {exc}')
            return
        self._commit(chain, 0)

    def _save(self) -> None:
        s = app_settings()
        default = Path(str(s.value("chain/dir", str(Path.home())))) / "zincir.sptchain"
        path, _ = QFileDialog.getSaveFileName(self, "İşlem zincirini kaydet", str(default),
                                              "İşlem zinciri (*.sptchain)")
        if not path:
            return
        s.setValue("chain/dir", str(Path(path).parent))
        self.chain.save(Path(path))
        self.set_status(f"Kaydedildi: {Path(path).name}")
