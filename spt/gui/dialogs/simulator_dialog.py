"""Cihaz simülatörü penceresi: `spt.sources.simulator` komutunu ayrı süreçte çalıştırır."""
from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtCore import QProcess, QProcessEnvironment, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from spt.console import CHILD_ENV
from spt.gui.icons import icon
from spt.gui.theme import Tokens, style_native_titlebar
from spt.gui.widgets.common import CollapsibleSection, combo, label
from spt.protocol.format import MAX_DATAGRAM, SAMPLE_TYPES, UDP_PORT

ROOT = Path(__file__).resolve().parents[3]
SAMPLES_DIR = ROOT / "samples_signal_data"
SIGNALS = [("Karışık", "mix"), ("Sinüs", "sine"), ("Kare", "square"), ("Üçgen", "triangle"),
           ("Testere", "sawtooth"), ("Chirp", "chirp"), ("Gürültü", "noise")]
DATAGRAMS = [("1472 B · standart MTU", 1472), ("8972 B · jumbo frame", 8972), ("65507 B · en büyük", MAX_DATAGRAM)]


def _console_python() -> str:
    """Çıktısı okunabilsin diye pythonw yerine python.exe (Qt konsol penceresi açmaz)."""
    exe = Path(sys.executable)
    candidate = exe.with_name("python.exe")
    return str(candidate if candidate.exists() else exe)


class SimulatorDialog(QDialog):
    started = pyqtSignal(str, int)  # hedef adres, port

    def __init__(self, tokens: Tokens, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Cihaz Simülatörü")
        self.setMinimumWidth(460)
        self._tokens = tokens
        self._proc: QProcess | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(8)

        intro = label("Gerçek cihaz yerine bu bilgisayardan SPTF formatında UDP verisi gönderir.", "Muted")
        intro.setWordWrap(True)
        lay.addWidget(intro)

        target = CollapsibleSection("Hedef")
        self.host = QLineEdit("127.0.0.1")
        self.port = QSpinBox()
        self.port.setRange(1, 65535)
        self.port.setValue(UDP_PORT)
        self.auto_listen = QCheckBox("Arayüzde dinlemeyi de başlat")
        self.auto_listen.setChecked(True)
        target.add_row("Adres", self.host)
        target.add_row("Port", self.port)
        target.form.addRow(self.auto_listen)
        lay.addWidget(target)

        source = CollapsibleSection("Sinyal")
        self.signal = combo([s for s, _ in SIGNALS])
        self.freq = self._dspin(0.001, 1e9, 50.0, " Hz")
        self.amp = self._dspin(1e-6, 1e6, 1.0, "")
        self.fs = self._dspin(1.0, 1e9, 10_000.0, " Hz")
        self.channels = QSpinBox()
        self.channels.setRange(1, 64)
        self.channels.setValue(2)
        self.type = combo([t.name for t in SAMPLE_TYPES.values()], 2)
        self.layout_combo = combo(["Interleaved", "Planar"])
        file_row = QWidget()
        fr = QHBoxLayout(file_row)
        fr.setContentsMargins(0, 0, 0, 0)
        fr.setSpacing(6)
        self.file = QLineEdit()
        self.file.setPlaceholderText("boş: üreteç kullanılır")
        browse = QPushButton("…")
        browse.setFixedWidth(36)
        browse.clicked.connect(self._browse)
        fr.addWidget(self.file, 1)
        fr.addWidget(browse)
        self.loop = QCheckBox("Dosya bitince başa dön")
        self.loop.setChecked(True)
        source.add_row("Sinyal", self.signal)
        source.add_row("Frekans", self.freq)
        source.add_row("Genlik", self.amp)
        source.add_row("Örnekleme", self.fs)
        source.add_row("Kanal", self.channels)
        source.add_row("Veri tipi", self.type)
        source.add_row("Yerleşim", self.layout_combo)
        source.add_row("Dosyadan", file_row)
        source.form.addRow(self.loop)
        lay.addWidget(source)

        transport = CollapsibleSection("İletim ve test", expanded=False)
        self.datagram = combo([d for d, _ in DATAGRAMS])
        self.crc = QCheckBox("CRC32 ekle")
        self.drop = self._dspin(0.0, 50.0, 0.0, " %")
        self.marker = self._dspin(0.0, 3600.0, 0.0, " s")
        self.marker.setToolTip("0 = kapalı")
        self.max_rate = QCheckBox("Hız testi (gerçek zamanı bekleme)")
        transport.add_row("Datagram", self.datagram)
        transport.add_row("Yapay kayıp", self.drop)
        transport.add_row("MARKER aralığı", self.marker)
        transport.form.addRow(self.crc)
        transport.form.addRow(self.max_rate)
        lay.addWidget(transport)

        self.status = label("Hazır.", "Readout")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)

        row = QHBoxLayout()
        row.addStretch(1)
        self.start_btn = QPushButton(icon("ph.play"), "  Başlat")
        self.start_btn.setObjectName("Primary")
        self.start_btn.clicked.connect(self.toggle)
        close = QPushButton("Kapat")
        close.clicked.connect(self.close)
        for b in (self.start_btn, close):
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            row.addWidget(b)
        lay.addLayout(row)
        style_native_titlebar(self, tokens)

    @staticmethod
    def _dspin(lo: float, hi: float, value: float, suffix: str) -> QDoubleSpinBox:
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setDecimals(3)
        w.setValue(value)
        w.setSuffix(suffix)
        return w

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Gönderilecek dosya", str(SAMPLES_DIR),
                                              "Sinyal dosyaları (*.sptr *.wav *.csv *.txt *.npy);;Tümü (*)")
        if path:
            self.file.setText(path)

    def arguments(self) -> list[str]:
        args = ["-u", "-m", "spt.sources.simulator", "--host", self.host.text().strip(),
                "--port", str(self.port.value()), "--datagram", str(DATAGRAMS[self.datagram.currentIndex()][1]),
                "--drop", str(self.drop.value() / 100), "--marker-every", str(self.marker.value())]
        if self.file.text().strip():
            args += ["--file", self.file.text().strip(), "--fs", str(self.fs.value())]
            if self.loop.isChecked():
                args.append("--loop")
        else:
            args += ["--signal", SIGNALS[self.signal.currentIndex()][1], "--freq", str(self.freq.value()),
                     "--amp", str(self.amp.value()), "--fs", str(self.fs.value()),
                     "--channels", str(self.channels.value()), "--type", self.type.currentText(),
                     "--layout", self.layout_combo.currentText().lower()]
        if self.crc.isChecked():
            args.append("--crc")
        if self.max_rate.isChecked():
            args.append("--max-rate")
        return args

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.state() != QProcess.ProcessState.NotRunning

    def toggle(self) -> None:
        if self.running:
            self.stop()
            return
        self._proc = QProcess(self)
        self._proc.setWorkingDirectory(str(ROOT))
        self._proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        for name, value in CHILD_ENV.items():
            env.insert(name, value)
        self._proc.setProcessEnvironment(env)
        self._proc.readyReadStandardOutput.connect(self._read_output)
        self._proc.finished.connect(self._finished)
        self._proc.start(_console_python(), self.arguments())
        self.start_btn.setText("  Durdur")
        self.start_btn.setIcon(icon("ph.stop"))
        self.status.setText("Başlatılıyor…")
        if self.auto_listen.isChecked():
            self.started.emit(self.host.text().strip(), self.port.value())

    def stop(self) -> None:
        if self.running:
            self._proc.kill()
            self._proc.waitForFinished(2000)

    def _read_output(self) -> None:
        text = bytes(self._proc.readAllStandardOutput()).decode("utf-8", "replace").strip()
        if text:
            self.status.setText(text.splitlines()[-1])

    def _finished(self, *_) -> None:
        self.start_btn.setText("  Başlat")
        self.start_btn.setIcon(icon("ph.play"))
        if not self.status.text().startswith(("BİTTİ", "Hata")):
            self.status.setText("Durduruldu.")
