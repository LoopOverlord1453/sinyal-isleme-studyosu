"""Günlük paneli: `spt` logger'ına yazılan iletileri renkli gösterir.

İletiler sinyal üzerinden aktarıldığı için alıcı thread'lerinden de güvenle yazılabilir.
"""
from __future__ import annotations

import html
import logging
import time

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QPlainTextEdit, QVBoxLayout, QWidget

from spt.gui.theme import Tokens, mono_font

_LEVELS = {
    logging.DEBUG: ("DEBUG", "text_muted"),
    logging.INFO: ("BİLGİ", "text_secondary"),
    logging.WARNING: ("UYARI", "warning"),
    logging.ERROR: ("HATA", "critical"),
    logging.CRITICAL: ("KRİTİK", "critical"),
}


class _Bridge(QObject):
    record = pyqtSignal(float, int, str)


class QtLogHandler(logging.Handler):
    def __init__(self, bridge: _Bridge):
        super().__init__()
        self._bridge = bridge

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._bridge.record.emit(record.created, record.levelno, record.getMessage())
        except RuntimeError:  # panel kapatılmış
            pass


class LogPanel(QWidget):
    def __init__(self, tokens: Tokens):
        super().__init__()
        self._t = tokens
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.view = QPlainTextEdit()
        self.view.setObjectName("Log")
        self.view.setReadOnly(True)
        self.view.setMaximumBlockCount(2000)
        self.view.setFont(mono_font(12))
        lay.addWidget(self.view)

        self._bridge = _Bridge(self)
        self._bridge.record.connect(self._append)
        self.handler = QtLogHandler(self._bridge)
        logging.getLogger("spt").addHandler(self.handler)

    def detach(self) -> None:
        logging.getLogger("spt").removeHandler(self.handler)

    def _append(self, created: float, levelno: int, message: str) -> None:
        level_text, role = _LEVELS.get(levelno, ("BİLGİ", "text_secondary"))
        stamp = time.strftime("%H:%M:%S", time.localtime(created))
        t = self._t
        self.view.appendHtml(
            f'<span style="color:{t.text_muted}">{stamp}</span>&nbsp;&nbsp;'
            f'<span style="color:{getattr(t, role)}">{level_text:<6}</span>&nbsp;&nbsp;'
            f'<span style="color:{t.text}">{html.escape(message)}</span>'
        )
