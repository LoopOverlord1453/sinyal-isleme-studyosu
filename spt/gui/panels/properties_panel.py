"""Özellikler paneli: görüntü, spektrum ve tetikleme ayarları (Aşama 1–2'de bağlanacak)."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFontDatabase
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from spt.gui.theme import active
from spt.gui.widgets.common import CollapsibleSection, ColorButton, combo, label

FFT_SIZES = [str(2 ** n) for n in range(10, 17)]
WINDOWS = ["Hann", "Hamming", "Blackman", "Blackman-Harris", "Flat-top", "Kaiser", "Dikdörtgen"]
SIDES = ["Tek taraflı (0 … fs/2)", "Çift taraflı (−fs/2 … +fs/2)"]  # sıra: spectrum.ONE_SIDED, TWO_SIDED


MIN_FIELD_WIDTH = 80  # geniş aralıklı kutular (±1e6) paneli gereksiz genişletmesin
FONT_PX_RANGE = (8, 24)
DEFAULT_FONT = "Varsayılan (arayüz yazı tipi)"


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
    spectrum_colors_changed = pyqtSignal(object)   # {"spectrum" | "max" | "min": renk ya da None}
    plot_font_changed = pyqtSignal(object, int)    # aile (None: arayüz yazı tipi), piksel boyu
    reset_requested = pyqtSignal()                 # "Varsayılana dön": kanal renkleri de temaya döner

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

        look = self._build_appearance()

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

        for section in (display, spectrum, look, trigger):
            lay.addWidget(section)
        lay.addStretch(1)

        # Henüz bağlı olmayan ayarlar: görünür ama devre dışı, ipucu nedenini söyler (yanıltıcı olmasın)
        later = "Bu sürümde yok — Aşama 2'de (spektral analiz ve ölçüm) planlandı"
        for w in (self.averages, self.unit, self.log_freq, self.trig_mode, self.trig_source, self.trig_level,
                  self.trig_edge, self.trig_pre):
            w.setEnabled(False)
            w.setToolTip(later)
        trigger.header.setToolTip(later)

    def _build_appearance(self) -> CollapsibleSection:
        """Zaman ve spektrum grafiklerinin renkleri ve yazı tipi. Kanal renkleri kanala özgüdür: Kaynaklar panelinde
        kanal satırına ya da zaman lejantına sağ tıklanarak değiştirilir."""
        look = CollapsibleSection("Grafik görünümü", expanded=False)
        self.color_buttons = {
            # varsayılanlar etkin temanın spektrum vurgusu ve iz renkleri (kayıt yüklenince set_appearance yeniler)
            "spectrum": ColorButton(active().tokens.accent, "Spektrum rengi"),
            "max": ColorButton(active().hold[0], "Max-hold rengi"),
            "min": ColorButton(active().hold[1], "Min-hold rengi"),
        }
        look.add_row("Spektrum", self.color_buttons["spectrum"])
        look.add_row("Max-hold", self.color_buttons["max"])
        look.add_row("Min-hold", self.color_buttons["min"])
        for button in self.color_buttons.values():
            button.color_changed.connect(lambda _c: self.spectrum_colors_changed.emit(self.spectrum_colors()))
        # Düz liste (QFontComboBox'a "varsayılan" öğesi eklenince seçilen aile yanlış okunuyordu)
        self.font_family = combo([DEFAULT_FONT, *QFontDatabase.families()])
        self.font_family.setMinimumWidth(MIN_FIELD_WIDTH)
        self.font_family.setToolTip("Eksen sayıları ve başlıkları, tepe / max / min etiketleri")
        self.font_px = _spin(*FONT_PX_RANGE, 12, suffix=" px")
        look.add_row("Yazı tipi", self.font_family)
        look.add_row("Yazı boyutu", self.font_px)
        self.font_family.currentIndexChanged.connect(lambda _i: self._emit_font())
        self.font_px.valueChanged.connect(lambda _v: self._emit_font())
        hint = label("Kanal rengi: Kaynaklar panelinde kanala ya da zaman lejantına sağ tıklayın.", "Muted")
        hint.setWordWrap(True)
        look.form.addRow(hint)
        reset = QPushButton("Varsayılana dön")
        reset.setToolTip("Spektrum / iz renklerini ve yazı tipini tema varsayılanına döndürür (kanal renkleri dahil)")
        reset.clicked.connect(self.reset_appearance)
        look.form.addRow(reset)
        self.reset_button = reset
        return look

    def spectrum_colors(self) -> dict[str, str | None]:
        return {key: button.color for key, button in self.color_buttons.items()}

    def set_appearance(self, colors: dict, defaults: dict, family: str | None, px: int) -> None:
        """Kayıttan / görünümden gelen değerleri sinyal yaymadan gösterir."""
        for key, button in self.color_buttons.items():
            button.set_default(defaults[key])
            button.set_color(colors.get(key))
        self.font_family.blockSignals(True)
        self.font_px.blockSignals(True)
        index = self.font_family.findText(family) if family else 0
        if index < 0:                       # kayıtlı aile bu makinede yok: listede yine de seçilebilsin
            self.font_family.addItem(family)
            index = self.font_family.count() - 1
        self.font_family.setCurrentIndex(index)
        self.font_px.setValue(int(px))
        self.font_family.blockSignals(False)
        self.font_px.blockSignals(False)

    def plot_font(self) -> tuple[str | None, int]:
        family = None if self.font_family.currentIndex() <= 0 else self.font_family.currentText()
        return family, self.font_px.value()

    def _emit_font(self) -> None:
        self.plot_font_changed.emit(*self.plot_font())

    def reset_appearance(self) -> None:
        for button in self.color_buttons.values():
            button.set_color(None)
        self.font_family.blockSignals(True)
        self.font_family.setCurrentIndex(0)
        self.font_family.blockSignals(False)
        self.font_px.blockSignals(True)
        self.font_px.setValue(12)
        self.font_px.blockSignals(False)
        self.spectrum_colors_changed.emit(self.spectrum_colors())
        self._emit_font()
        self.reset_requested.emit()

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
