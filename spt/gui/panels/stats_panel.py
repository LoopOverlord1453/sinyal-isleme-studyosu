"""Paket istatistikleri: akış başına alım hızı, kayıp ve durum."""
from __future__ import annotations

import time

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QStackedLayout,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from spt.gui.widgets.common import StatusBadge, cell, label, num_chip
from spt.gui.widgets.status_bar import format_bytes_rate

COLUMNS = ["Akış", "Format", "Paket/s", "Veri hızı", "Kayıp", "Durum"]
NAME_COLUMN_WIDTH = 140
LOSS_HOLD_S = 2.0   # kayıp görüldükten sonra "Kayıplı" rozetinin kalma süresi
IDLE_AFTER_S = 1.0


class StatsPanel(QWidget):
    def __init__(self):
        super().__init__()
        self._stack = QStackedLayout(self)
        empty = label("Akış yok.\n\nUDP dinlemeyi başlatın ya da bir kayıt açın.", "EmptyState")
        empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stack.addWidget(empty)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(40)
        self.table.setShowGrid(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        header = self.table.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        # Hücre widget'ları içerik genişliğine sayılmadığından ad sütunu elle boyutlanır.
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        header.resizeSection(0, NAME_COLUMN_WIDTH)
        header.setStretchLastSection(True)
        self._stack.addWidget(self.table)

        self._rows: dict[str, tuple[int, StatusBadge]] = {}

    def clear(self) -> None:
        self.table.setRowCount(0)
        self._rows.clear()
        self._stack.setCurrentIndex(0)

    def remove(self, key: str) -> None:
        if key not in self._rows:
            return
        row, _ = self._rows.pop(key)
        self.table.removeRow(row)
        self._rows = {k: (r - 1 if r > row else r, b) for k, (r, b) in self._rows.items()}
        self._stack.setCurrentIndex(1 if self._rows else 0)

    def update_rates(self, rates: dict) -> None:
        now = time.monotonic()
        for key, rate in rates.items():
            st = rate.stats
            if st is None:
                continue
            if key not in self._rows:
                self._add_row(key, st)
            row, badge = self._rows[key]
            values = (st.format, f"{rate.packets_per_s:,.0f}".replace(",", "\u202f"),
                      format_bytes_rate(rate.bytes_per_s), f"{st.lost_samples:,}".replace(",", "\u202f"))
            for col, value in enumerate(values, start=1):
                item = self.table.item(row, col)
                if item.text() != value:
                    item.setText(value)
            self.table.item(row, 4).setToolTip(
                f"Kayıp örnek: {st.lost_samples}\nYinelenen çerçeve: {st.duplicates}\n"
                f"Yeniden başlama (reset): {st.restarts}")
            conflict = getattr(st, "conflict", "")
            badge.setToolTip("Aynı akış kimliği iki kaynaktan geliyor (ayrıntı: Bağlantı paneli / Günlük)"
                             if conflict else "")
            if conflict:
                badge.set_state("Çakışma", "critical")
            elif st.idle_s >= IDLE_AFTER_S:
                badge.set_state("Bekliyor", "idle")
            elif now - rate.last_loss <= LOSS_HOLD_S:
                badge.set_state("Kayıplı", "warning")
            else:
                badge.set_state("Aktif", "good")

    def _add_row(self, key: str, st) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        name = cell(num_chip(f"{st.stream_id:02d}"), label(st.name or f"akış {st.stream_id}", "LinkText"))
        name.setToolTip(f"Kaynak: {st.source}")
        self.table.setCellWidget(row, 0, name)
        for col in range(1, len(COLUMNS) - 1):
            self.table.setItem(row, col, QTableWidgetItem(""))
        badge = StatusBadge("Aktif", "good")
        self.table.setCellWidget(row, len(COLUMNS) - 1, cell(badge))
        self._rows[key] = (row, badge)
        self._stack.setCurrentIndex(1)
