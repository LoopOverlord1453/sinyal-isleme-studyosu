"""Geri al / yinele: durum anlık görüntülerinden oluşan yığın.

Filtre Tasarımcısı ve işlem zinciri durumu (JSON'a çevrilebilir sözlük) her kalıcı değişiklikte yığına eklenir.
Geri alma bir önceki durumu döndürür; yeni bir değişiklik yinelenebilecek durumları siler.
"""
from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal

MAX_STATES = 200


class History(QObject):
    changed = pyqtSignal()

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._states: list[dict] = []
        self._index = -1

    def reset(self, state: dict) -> None:
        self._states, self._index = [state], 0
        self.changed.emit()

    @property
    def current(self) -> dict | None:
        return self._states[self._index] if self._index >= 0 else None

    def push(self, state: dict) -> bool:
        if state == self.current:
            return False
        del self._states[self._index + 1:]
        self._states.append(state)
        if len(self._states) > MAX_STATES:
            del self._states[0]
        self._index = len(self._states) - 1
        self.changed.emit()
        return True

    @property
    def can_undo(self) -> bool:
        return self._index > 0

    @property
    def can_redo(self) -> bool:
        return 0 <= self._index < len(self._states) - 1

    def undo(self) -> dict | None:
        if not self.can_undo:
            return None
        self._index -= 1
        self.changed.emit()
        return self._states[self._index]

    def redo(self) -> dict | None:
        if not self.can_redo:
            return None
        self._index += 1
        self.changed.emit()
        return self._states[self._index]
