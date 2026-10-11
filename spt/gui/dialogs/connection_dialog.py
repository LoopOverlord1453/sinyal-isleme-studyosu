"""Bağlantı ayarları: adres, port, soket ve akış tamponu boyutları."""
from __future__ import annotations

from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QDoubleSpinBox, QLineEdit, QSpinBox, QVBoxLayout, QWidget

from spt.gui.theme import Tokens, style_native_titlebar
from spt.gui.widgets.common import add_form_row, form_layout, label
from spt.sources.receiver import ReceiverConfig


class ConnectionDialog(QDialog):
    def __init__(self, cfg: ReceiverConfig, tokens: Tokens, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Bağlantı Ayarları")
        self.setMinimumWidth(420)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 18, 20, 16)
        lay.setSpacing(12)

        self.address = QLineEdit(cfg.address)
        self.port = QSpinBox()
        self.port.setRange(1, 65535)
        self.port.setValue(cfg.port)
        self.socket_mb = QSpinBox()
        self.socket_mb.setRange(1, 1024)
        self.socket_mb.setSuffix(" MB")
        self.socket_mb.setValue(cfg.socket_buffer_mb)
        self.ring_seconds = QDoubleSpinBox()
        self.ring_seconds.setRange(1.0, 600.0)
        self.ring_seconds.setDecimals(0)
        self.ring_seconds.setSuffix(" s")
        self.ring_seconds.setValue(cfg.ring_seconds)
        self.max_ring_mb = QSpinBox()
        self.max_ring_mb.setRange(16, 8192)
        self.max_ring_mb.setSuffix(" MB")
        self.max_ring_mb.setValue(cfg.max_ring_mb)

        form = form_layout()
        add_form_row(form, "Dinlenecek adres", self.address)
        add_form_row(form, "UDP portu", self.port)
        add_form_row(form, "Soket tamponu", self.socket_mb)
        add_form_row(form, "Akış geçmişi", self.ring_seconds)
        add_form_row(form, "Akış başına en fazla bellek", self.max_ring_mb)
        lay.addLayout(form)

        note = label("Soket tamponu, arayüz meşgulken gelen paketleri işletim sisteminde bekletir; "
                     "gigabit hızlarında 64 MB veya üstü önerilir. Akış geçmişi, her akış için bellekte "
                     "tutulan süredir; yüksek hızlı akışlarda bellek sınırı önce dolar.<br><br>"
                     "Değişiklikler bir sonraki dinleme veya oynatmada geçerli olur.", "Muted")
        note.setWordWrap(True)
        lay.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Kaydet")
        buttons.button(QDialogButtonBox.StandardButton.Ok).setObjectName("Primary")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Vazgeç")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        style_native_titlebar(self, tokens)

    def config(self) -> ReceiverConfig:
        return ReceiverConfig(self.address.text().strip() or "0.0.0.0", self.port.value(), self.socket_mb.value(),
                              self.ring_seconds.value(), self.max_ring_mb.value())
