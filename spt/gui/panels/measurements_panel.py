"""Ölçümler paneli: kanal başına temel istatistikler."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QAbstractItemView, QHeaderView, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from spt.gui import demo


class MeasurementsPanel(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget()
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.table.verticalHeader().setDefaultSectionSize(34)
        lay.addWidget(self.table)
        self._fill_demo()

    def _fill_demo(self) -> None:
        _, channels = demo.time_signals()
        rows = demo.measurements(channels)
        self.table.setColumnCount(1 + len(channels))
        self.table.setRowCount(len(rows))
        self.table.setHorizontalHeaderItem(0, QTableWidgetItem("Ölçüm"))
        right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        for c, ch in enumerate(channels, start=1):
            head = QTableWidgetItem(ch.name)
            head.setTextAlignment(right)
            self.table.setHorizontalHeaderItem(c, head)
        for r, (name, values) in enumerate(rows.items()):
            name_item = QTableWidgetItem(name)
            name_item.setForeground(self.palette().placeholderText())
            self.table.setItem(r, 0, name_item)
            for c, value in enumerate(values, start=1):
                item = QTableWidgetItem(value)
                item.setTextAlignment(right)
                self.table.setItem(r, c, item)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in range(1, self.table.columnCount()):
            header.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
