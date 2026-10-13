"""Karta Gönder: Sinyal Üreteci'ndeki tabloyu çalışan karta Ethernet (UDP) üzerinden gönderir; kart tabloyu
RAM'e alıp DAC'tan çalar (derleme ve yükleme gerekmez).

Kartta `stm32f4_examples/001_signal_generator` yazılımı çalışmalıdır. Protokol ve istemci: `spt.protocol.sptg`.
Ağ işlemi ayrı iş parçacığında yapılır; pencere açık kalır, sinyal değiştirilip yeniden gönderilebilir.
"""
from __future__ import annotations

import threading
from collections.abc import Callable

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QLineEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget

from spt.dsp.filters import format_hz
from spt.gui.settings import app_settings
from spt.gui.theme import Tokens, style_native_titlebar
from spt.gui.widgets.common import add_form_row, combo, form_layout, label
from spt.protocol import sptg

DEFAULT_HOST = "192.168.172.98"   # 001_signal_generator projesinin sabit IP adresi
SETTINGS_KEY = "siggen_board"


class _Bridge(QObject):
    """İş parçacığındaki ağ işleminin sonucunu arayüz iş parçacığına taşır."""
    done = pyqtSignal(int, object)   # nesil, PlayResult | BoardInfo | Exception


class BoardSendDialog(QDialog):
    def __init__(self, tokens: Tokens, table_provider: Callable[[], object | None], parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Sinyal Üreteci — Karta Gönder")
        self.setMinimumWidth(460)
        self._tokens = tokens
        self._table_provider = table_provider
        self._generation = 0
        self._bridge = _Bridge(self)
        self._bridge.done.connect(self._finished)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 18, 20, 16)
        lay.setSpacing(12)

        s = app_settings()
        self.host = QLineEdit(str(s.value(f"{SETTINGS_KEY}/host", DEFAULT_HOST)))
        self.host.setToolTip("Kartın IP adresi (001_signal_generator: 192.168.172.98)")
        self.port = QSpinBox()
        self.port.setRange(1, 65535)
        self.port.setValue(int(s.value(f"{SETTINGS_KEY}/port", sptg.UDP_PORT)))
        self.output = combo([])
        for output in sptg.Output:
            self.output.addItem(sptg.OUTPUT_LABELS[output], int(output))
        self.output.setCurrentIndex(max(self.output.findData(int(s.value(f"{SETTINGS_KEY}/output", 1))), 0))
        self.output.setToolTip("Sinyalin verileceği DAC çıkışı; “DAC1 + DAC2” aynı sinyali iki çıkışa birden verir")

        form = form_layout()
        add_form_row(form, "Kart IP adresi", self.host)
        add_form_row(form, "UDP portu", self.port)
        add_form_row(form, "Çıkış", self.output)
        lay.addLayout(form)

        note = label("Ekrandaki tablo karta gönderilir; kart onu RAM'e alır ve hemen çalar. Derleme ve yükleme "
                     "gerekmez; kart yeniden başlatılınca derlenmiş sinyale döner.<br><br>"
                     "Zamanlayıcı ayarı (PSC / ARR) kartın bildirdiği zamanlayıcı saatinden hesaplanır.", "Muted")
        note.setWordWrap(True)
        note.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(note)

        self.status = label("", "Readout")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(self.status)

        row = QHBoxLayout()
        row.addStretch(1)
        self.send_button = QPushButton("Gönder")
        self.send_button.setObjectName("Primary")
        self.send_button.setDefault(True)
        self.send_button.clicked.connect(self.send)
        self.stop_button = QPushButton("Durdur")
        self.stop_button.setToolTip("Karttaki DAC çıkışını kapatır")
        self.stop_button.clicked.connect(self.stop)
        close = QPushButton("Kapat")
        close.clicked.connect(self.close)
        for button in (self.send_button, self.stop_button, close):
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            row.addWidget(button)
        lay.addLayout(row)
        style_native_titlebar(self, tokens)

    # ---------------------------------------------------------------- durum
    def set_status(self, text: str, state: str = "") -> None:
        t = self._tokens
        color = {"good": t.good, "warning": t.warning, "critical": t.critical}.get(state)
        self.status.setText((f'<span style="color:{color}">●</span> ' if color else "") + text)

    def _set_busy(self, busy: bool) -> None:
        for widget in (self.send_button, self.stop_button, self.host, self.port, self.output):
            widget.setEnabled(not busy)

    def _save_settings(self) -> None:
        s = app_settings()
        s.setValue(f"{SETTINGS_KEY}/host", self.host.text().strip())
        s.setValue(f"{SETTINGS_KEY}/port", self.port.value())
        s.setValue(f"{SETTINGS_KEY}/output", int(self.output.currentData()))

    # ---------------------------------------------------------------- eylemler
    def send(self) -> None:
        table = self._table_provider()
        if table is None:
            self.set_status("Gönderilecek tablo yok: önce tasarımdaki hatayı giderin.", "warning")
            return
        try:
            payload = sptg.payload_from_table(table)
        except sptg.SptgError as exc:
            self.set_status(str(exc), "warning")
            return
        output = sptg.Output(int(self.output.currentData()))
        self._run(lambda client: client.play(payload, output), "Gönderiliyor…")

    def stop(self) -> None:
        self._run(lambda client: client.stop(), "Durduruluyor…")

    def _run(self, action: Callable[[sptg.SptgClient], object], text: str) -> None:
        host, port = self.host.text().strip(), self.port.value()
        if not host:
            self.set_status("Kartın IP adresini yazın.", "warning")
            return
        self._save_settings()
        self._generation += 1
        generation = self._generation
        self._set_busy(True)
        self.set_status(text)

        def work() -> None:
            try:
                with sptg.SptgClient(host, port) as client:
                    result: object = action(client)
            except Exception as exc:  # noqa: BLE001 — hata arayüzde gösterilir
                result = exc
            try:
                self._bridge.done.emit(generation, result)
            except RuntimeError:      # pencere bu arada kapatılıp silinmiş
                pass

        threading.Thread(target=work, name="spt-board-send", daemon=True).start()

    def _finished(self, generation: int, result: object) -> None:
        if generation != self._generation:
            return
        self._set_busy(False)
        if isinstance(result, sptg.PlayResult):
            parts = [f"Çalıyor: <b>{sptg.OUTPUT_LABELS[result.output]}</b>", f"fs = {format_hz(result.fs)}"]
            if result.f > 0:
                parts.append(f"f = {format_hz(result.f)}")
            parts.append(f"N = {result.sample_count}")
            parts.append(f"PSC {result.timer_prescaler} / ARR {result.timer_autoreload}")
            if abs(result.fs_error_ppm) > 1.0:
                parts.append(f"fs hatası {result.fs_error_ppm:+.1f} ppm")
            self.set_status(" · ".join(parts), "good")
        elif isinstance(result, sptg.BoardInfo):
            self.set_status(f"Kart: {result.state_text}.", "good")
        elif isinstance(result, sptg.SptgError):
            self.set_status(str(result), "critical")
        else:
            self.set_status(f"{type(result).__name__}: {result}", "critical")
