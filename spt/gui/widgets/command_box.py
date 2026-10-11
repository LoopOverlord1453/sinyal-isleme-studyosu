"""Üst çubuktaki arama kutusu: komut adını yazıp Enter ile çalıştırma (komut paleti)."""
from __future__ import annotations

from PyQt6.QtCore import QSize, QStringListModel, Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QCompleter, QFrame, QHBoxLayout, QLabel, QLineEdit

from spt.gui.icons import icon
from spt.gui.theme import TERRAFLUX_DARK
from spt.gui.widgets.common import label


class CommandBox(QFrame):
    def __init__(self, actions: list[QAction]):
        super().__init__()
        self.setObjectName("SearchBox")
        self.setFixedWidth(300)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 0, 6, 0)
        lay.setSpacing(6)

        glyph = QLabel()
        glyph.setPixmap(icon("ph.magnifying-glass", TERRAFLUX_DARK.text_muted).pixmap(QSize(16, 16)))
        lay.addWidget(glyph)

        self.edit = QLineEdit()
        self.edit.setPlaceholderText("Komut ara…")
        lay.addWidget(self.edit, 1)
        lay.addWidget(label("Ctrl K", "Kbd"))

        self._by_text = {a.text().replace("&", ""): a for a in actions if a.text()}
        completer = QCompleter(QStringListModel(sorted(self._by_text)), self)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        completer.popup().setObjectName("CompleterPopup")
        completer.activated.connect(self._run)
        self.edit.setCompleter(completer)
        self.edit.returnPressed.connect(lambda: self._run(self.edit.text()))

    def focus(self) -> None:
        self.edit.setFocus(Qt.FocusReason.ShortcutFocusReason)
        self.edit.selectAll()

    def _run(self, text: str) -> None:
        action = self._by_text.get(text)
        if action is None:
            return
        self.edit.clear()
        self.edit.clearFocus()
        action.trigger()
