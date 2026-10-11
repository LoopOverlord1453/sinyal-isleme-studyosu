"""Alt durum çubuğu: bağlantı durumu, veri hızı, kayıp ve kayıt durumu."""
from __future__ import annotations

from PyQt6.QtWidgets import QStatusBar

from spt import __version__
from spt.gui.theme import Tokens
from spt.gui.widgets.common import label

_CONNECTION_STATES = {
    "idle": ("Bağlı değil", "text_muted"),
    "starting": ("Başlatılıyor", "text_secondary"),
    "listening": ("Dinleniyor", "accent"),
    "receiving": ("Veri alınıyor", "good"),
    "replay": ("Kayıt oynatılıyor", "accent"),
    "replay_end": ("Kayıt bitti", "text_muted"),
    "error": ("Hata", "critical"),
}


def format_bytes_rate(bps: float) -> str:
    if bps >= 1e6:
        return f"{bps / 1e6:.2f} MB/s"
    if bps >= 1e3:
        return f"{bps / 1e3:.1f} kB/s"
    return f"{bps:.0f} B/s"


class StatusBar(QStatusBar):
    def __init__(self, tokens: Tokens):
        super().__init__()
        self._t = tokens
        self.setSizeGripEnabled(False)

        self.connection = label("", "StatusChip")
        self.endpoint = label("")
        self.packets = label("")
        self.rate = label("")
        self.loss = label("")
        for w, width in ((self.packets, 120), (self.rate, 90), (self.loss, 150)):
            w.setMinimumWidth(width)
        for w in (self.connection, self.endpoint, self.packets, self.rate, self.loss):
            self.addWidget(w)

        self.recording = label("")
        self.addPermanentWidget(self.recording)
        self.addPermanentWidget(label(f"v{__version__}"))

        self.set_connection("idle", "")
        self.set_traffic(0.0, 0.0, 0)
        self.set_recording(False)

    def set_connection(self, state: str, detail: str) -> None:
        text, role = _CONNECTION_STATES.get(state, _CONNECTION_STATES["idle"])
        color = getattr(self._t, role)
        self.connection.setText(f'<span style="color:{color}">●</span>&nbsp;&nbsp;{text}')
        self.connection.setToolTip(detail)
        self.endpoint.setText(detail if state != "error" else "")

    def set_traffic(self, packets_per_s: float, bytes_per_s: float, lost: int) -> None:
        self.packets.setText(f"{packets_per_s:,.0f} paket/s".replace(",", "\u202f"))
        self.rate.setText(format_bytes_rate(bytes_per_s))
        color = self._t.warning if lost else self._t.text_muted
        self.loss.setText(f'<span style="color:{color}">Kayıp {lost:,} örnek</span>'.replace(",", "\u202f"))

    def set_recording(self, active: bool, size_bytes: int = 0) -> None:
        color = self._t.critical if active else self._t.text_muted
        text = f"Kayıt açık · {size_bytes / 1e6:.1f} MB" if active else "Kayıt kapalı"
        self.recording.setText(f'<span style="color:{color}">●</span>&nbsp;{text}')
