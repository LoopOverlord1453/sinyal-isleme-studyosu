"""Özellikler paneli: görüntü, spektrum ve tetikleme ayarları (Aşama 1–2'de bağlanacak)."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QScrollArea, QSpinBox, QVBoxLayout, QWidget

from spt.gui.widgets.common import CollapsibleSection, combo

FFT_SIZES = [str(2 ** n) for n in range(10, 17)]
WINDOWS = ["Hann", "Hamming", "Blackman", "Blackman-Harris", "Flat-top", "Kaiser", "Dikdörtgen"]
SIDES = ["Tek taraflı (0 … fs/2)", "Çift taraflı (−fs/2 … +fs/2)"]  # sıra: spectrum.ONE_SIDED, TWO_SIDED


MIN_FIELD_WIDTH = 80  # geniş aralıklı kutular (±1e6) paneli gereksiz genişletmesin


def _spin(lo: int, hi: int, value: int, suffix: str = "") -> QSpinBox:
    w = QSpinBox()
    w.setMinimumWidth(MIN_FIELD_WIDTH)
    w.setRange(lo, hi)
    w.setValue(value)
    w.setSuffix(suffix)
    return w


def _dspin(lo: float, hi: float, value: float, decimals: int = 3, suffix: str = "") -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setMinimumWidth(MIN_FIELD_WIDTH)
    w.setRange(lo, hi)
    w.setDecimals(decimals)
    w.setValue(value)
    w.setSuffix(suffix)
    return w


class PropertiesPanel(QWidget):
    def __init__(self):
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(scroll)

        host = QWidget()
        lay = QVBoxLayout(host)
        lay.setContentsMargins(0, 0, 6, 0)
        lay.setSpacing(6)
        scroll.setWidget(host)

        display = CollapsibleSection("Görüntü")
        self.time_window = combo(["10 ms", "50 ms", "100 ms", "500 ms", "1 s", "5 s", "10 s"], 2)
        self.y_mode = combo(["Otomatik", "Sabit"])
        self.y_min = _dspin(-1e6, 1e6, -2.0, suffix=" V")
        self.y_max = _dspin(-1e6, 1e6, 2.0, suffix=" V")
        self.grid = QCheckBox("Izgara")
        self.grid.setChecked(True)
        self.peak_decimation = QCheckBox("Tepe korumalı seyreltme")
        self.peak_decimation.setChecked(True)
        display.add_row("Zaman penceresi", self.time_window)
        display.add_row("Y ekseni", self.y_mode)
        display.add_row("Y min", self.y_min)
        display.add_row("Y max", self.y_max)
        display.form.addRow(self.grid)
        display.form.addRow(self.peak_decimation)
        self.y_mode.currentIndexChanged.connect(self._update_y_fields)
        self._update_y_fields()

        spectrum = CollapsibleSection("Spektrum (FFT)")
        self.fft_size = combo(FFT_SIZES, FFT_SIZES.index("8192"))
        self.window = combo(WINDOWS)
        self.sides = combo(SIDES)
        self.sides.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.sides.setMinimumContentsLength(8)
        self.sides.setToolTip("Tek taraflı: 0 … fs/2, gerçel sinüs A genliğinde.\n"
                              "Çift taraflı: −fs/2 … +fs/2, gerçel sinüs ±f'de A/2 (−6 dB) olarak iki lobda.")
        self.kaiser_beta = _dspin(0.0, 40.0, 8.6, decimals=1)
        self.averages = _spin(1, 64, 4)
        self.unit = combo(["dBV", "dBFS", "Lineer (V)"])
        self.log_freq = QCheckBox("Logaritmik frekans ekseni")
        spectrum.add_row("FFT boyutu", self.fft_size)
        spectrum.add_row("Pencere", self.window)
        spectrum.add_row("Gösterim", self.sides)
        spectrum.add_row("Kaiser β", self.kaiser_beta)
        spectrum.add_row("Ortalama", self.averages)
        spectrum.add_row("Birim", self.unit)
        spectrum.form.addRow(self.log_freq)
        self.window.currentTextChanged.connect(self._update_kaiser)
        self._update_kaiser()

        trigger = CollapsibleSection("Tetikleme", expanded=False)
        self.trig_mode = combo(["Serbest", "Otomatik", "Normal", "Tek"])
        self.trig_source = combo(["ch0", "ch1"])
        self.trig_level = _dspin(-1e6, 1e6, 0.0, suffix=" V")
        self.trig_edge = combo(["Yükselen", "Düşen"])
        self.trig_pre = _spin(0, 100, 20, suffix=" %")
        trigger.add_row("Mod", self.trig_mode)
        trigger.add_row("Kaynak", self.trig_source)
        trigger.add_row("Seviye", self.trig_level)
        trigger.add_row("Kenar", self.trig_edge)
        trigger.add_row("Ön tetik", self.trig_pre)

        for section in (display, spectrum, trigger):
            lay.addWidget(section)
        lay.addStretch(1)

        # Henüz bağlı olmayan ayarlar: görünür ama devre dışı, ipucu nedenini söyler (yanıltıcı olmasın)
        later = "Bu sürümde yok — Aşama 2'de (spektral analiz ve ölçüm) planlandı"
        for w in (self.averages, self.unit, self.log_freq, self.trig_mode, self.trig_source, self.trig_level,
                  self.trig_edge, self.trig_pre):
            w.setEnabled(False)
            w.setToolTip(later)
        trigger.header.setToolTip(later)

    def set_unit(self, unit: str) -> None:
        """Y ekseni sınırlarının eki görünen kanalların birimi olsun (karışıksa birimsiz)."""
        for w in (self.y_min, self.y_max, self.trig_level):
            w.setSuffix(f" {unit}" if unit else "")

    def _update_y_fields(self) -> None:
        fixed = self.y_mode.currentText() == "Sabit"
        self.y_min.setEnabled(fixed)
        self.y_max.setEnabled(fixed)

    def _update_kaiser(self) -> None:
        self.kaiser_beta.setEnabled(self.window.currentText() == "Kaiser")
