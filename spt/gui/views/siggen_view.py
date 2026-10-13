"""Sinyal Üreteci: DAC ile üretilecek dalga şekillerini tasarlar, DAC'ın gerçekte vereceği (nicemlenmiş) çıkışı
ve spektrumunu gösterir, MCU için C kodu üretir.

- Hazır dalga türleri (sinüs, kare, üçgen, testere, darbe, chirp, gürültü, AM / FM, EKG …) ve özel tasarım:
  formül (güvenli ifade), sürüklenebilir noktalarla eğri, harmonikler (Fourier serisi), CSV örnek dizisi.
- DAC modeli: çözünürlük, referans / çıkış aralığı, kodlama, hizalama, kelime genişliği (hazır ayarlar: STM32
  dahili DAC, STM32'ye bağlanan harici SPI / I2C DAC ve I2S kodek). Kırpılan örnekler her zaman sayılır ve
  gösterilir.
- Tablo: tam periyot (DMA dairesel; frekans hatası en küçük tablo boyu) ya da DDS (faz akümülatörü).
- Ölçümler: THD, SFDR, SNR, ENOB, nicemleme SNR'si, en büyük eğim, görüntü frekansı ve önerilen yeniden
  yapılandırma (alçak geçiren) filtresi, bellek, STM32 zamanlayıcı PSC / ARR.
- "Simülatörle Gönder": nicemlenmiş çıkışı simülatör üzerinden UDP akışı olarak gönderir; Canlı İzleme'de
  diğer sinyaller gibi incelenir / filtrelenir.
- "Karta Gönder…": tabloyu çalışan karta Ethernet üzerinden gönderir; kart derleme ve yükleme olmadan DAC'tan
  çalar (`spt.protocol.sptg`, kart yazılımı: stm32f4_examples/001_signal_generator).

Hesaplar `spt.dsp.siggen`, C kodu `spt.codegen.dac_c` içindedir (Qt'siz, testli).
"""
from __future__ import annotations

import html
import json
import logging
import math
import re
import tempfile
import threading
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHeaderView,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from spt.codegen import dac_c
from spt.dsp import siggen as sg
from spt.dsp.filters import format_hz
from spt.gui.dialogs.board_send_dialog import BoardSendDialog
from spt.gui.dialogs.code_dialog import CodeDialog
from spt.gui.settings import app_settings
from spt.gui.theme import ThemeManager, mono_font
from spt.gui.widgets.common import (
    Card,
    CollapsibleSection,
    LegendBar,
    PageHeader,
    chip,
    combo,
    hbox,
    icon_button,
    label,
)
from spt.gui.widgets.inspector import DataTips, PlotAnalysis, Trace, fmt, visible_slice
from spt.gui.widgets.plot import Crosshair, make_plot, pen

log = logging.getLogger("spt.gui")

RECOMPUTE_MS = 150
PLOT_SAMPLES = 4000           # zaman grafiğinde gösterilen en fazla örnek
SIM_SECONDS = 1.0             # simülatöre gönderilen dosyanın süresi (döngüyle tekrarlanır)
NUMERIC = {  # alan → (en küçük, en büyük, ondalık, adım)
    "freq": (1e-3, 1e9, 4, 10.0), "amplitude": (-1e6, 1e6, 6, 0.1), "offset": (-1e6, 1e6, 6, 0.1),
    "phase_deg": (-360.0, 360.0, 3, 15.0), "duty": (0.0, 1.0, 4, 0.05), "symmetry": (0.0, 1.0, 4, 0.05),
    "width": (1e-12, 1e3, 9, 1e-5), "rise": (0.0, 1.0, 4, 0.05), "high": (0.0, 1.0, 4, 0.05),
    "fall": (0.0, 1.0, 4, 0.05), "low": (0.0, 1.0, 4, 0.05), "f1": (1e-3, 1e9, 4, 100.0),
    "sweep_time": (1e-9, 1e4, 9, 1e-3), "mod_freq": (1e-3, 1e9, 4, 10.0), "depth": (0.0, 1.0, 4, 0.05),
    "deviation": (0.0, 1e9, 4, 10.0), "tau": (1e-12, 1e4, 9, 1e-4), "repeat_period": (1e-9, 1e4, 9, 1e-3),
    "bpm": (1.0, 600.0, 2, 5.0), "duration": (1e-6, 60.0, 6, 0.01),
}
INTEGER = {"steps": (2, 4096), "seed": (0, 2_147_483_647)}
EDITOR_FIELDS = ("expr", "points", "tones", "harmonics", "samples")


# Kod boyutu ölçümünün çekirdeği (STM32F4 / G4 / L4 sınıfı)
MEASURE_PROFILE = "cortex-m4f"


class _SizeBridge(QObject):
    """Arka plan ölçümünün sonucunu arayüz iş parçacığına taşır."""
    done = pyqtSignal(int, object)  # nesil, [(derleyici adı, Sizes | None, açıklama)]


def _measure_sizes(bridge: _SizeBridge, generation: int, files, source: str, profile_key: str) -> None:
    """Üretilen <ad>.c dosyasını makinedeki ARM derleyicileriyle derleyip gerçek boyutu ölçer (daemon iş parçacığı:
    beklenmez, kapanışı geciktirmez)."""
    from spt.codegen import mcu, toolchain
    rows = []
    prof = mcu.MCU_PROFILES[profile_key]
    for compiler in ("gcc", "clang"):
        try:
            build = toolchain.compile_cross(files, [source], prof, compiler, "-Os", warnings=("-Wall",))
            if build.ok:
                tool = build.tool
                rows.append((f"{mcu.COMPILERS[compiler]} ({tool.version if tool else ''}) -Os, {prof.name}",
                             build.sizes(source), ""))
            else:
                rows.append((mcu.COMPILERS[compiler], None, build.log.splitlines()[0] if build.log else "derlenemedi"))
        except Exception as exc:  # noqa: BLE001 — ölçüm başarısız olsa da kod üretilmiş durumda
            rows.append((mcu.COMPILERS[compiler], None, f"{type(exc).__name__}: {exc}"))
    bridge.done.emit(generation, rows)


def _short(field: str) -> str:
    text = sg.FIELD_INFO[field][0]
    return text.split(" (")[0]


class SignalGeneratorView(QWidget):
    simulate_requested = pyqtSignal(str, float)  # .npy dosya yolu, örnekleme hızı

    def __init__(self, theme: ThemeManager):
        super().__init__()
        self._theme = theme
        self.table: sg.Table | None = None
        self.metrics: sg.Metrics | None = None
        self.error: str | None = None
        self._loading = False
        self._samples: tuple[float, ...] = ()
        self._spec_fields: dict[str, QWidget] = {}
        self._code_dialog: CodeDialog | None = None
        self._board_dialog: BoardSendDialog | None = None
        self._time_x = np.empty(0)
        self._time_ideal = np.empty(0)
        self._time_out = np.empty(0)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        self.header = PageHeader("Sinyal Üreteci",
                                 "DAC ile üretilecek dalga şekilleri · özel tasarım · nicemlenmiş çıkış · C kodu")
        self.summary_chip = chip("")
        self.header.add_right(self.summary_chip)
        lay.addWidget(self.header)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        lay.addWidget(split, 1)
        split.addWidget(self._build_setup())
        split.addWidget(self._build_plots())
        split.setStretchFactor(1, 1)
        split.setSizes([380, 900])

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(RECOMPUTE_MS)
        self._timer.timeout.connect(self.recompute)
        self._load_settings()
        self._rebuild_wave_fields()
        self.recompute()

    # ================================================================ kurulum paneli
    def _build_setup(self) -> Card:
        card = Card("Tasarım")
        card.setMinimumWidth(300)
        card.setMaximumWidth(480)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        host = QWidget()
        col = QVBoxLayout(host)
        col.setContentsMargins(0, 0, 6, 0)
        col.setSpacing(6)
        scroll.setWidget(host)
        card.body.addWidget(scroll, 1)

        # ---- dalga şekli
        self.wave_section = CollapsibleSection("Dalga şekli")
        self.kind = combo([])
        for key, text in sg.WAVEFORMS.items():
            self.kind.addItem(text, key)
        self.kind.currentIndexChanged.connect(self._on_kind)
        self.wave_section.add_row("Tür", self._compact(self.kind))
        self.amplitude = self._spin("amplitude")
        self.offset = self._spin("offset")
        self.wave_section.add_row("Genlik (tepe)", self.amplitude)
        self.wave_section.add_row("Ofset / taban", self.offset)
        col.addWidget(self.wave_section)
        self._field_rows: list[QWidget] = []

        # ---- özel tasarım editörleri
        self.custom_section = CollapsibleSection("Özel tasarım")
        self._build_editors()
        col.addWidget(self.custom_section)

        # ---- DAC
        dac = CollapsibleSection("DAC")
        self.preset = combo(["Özel"] + list(sg.DAC_PRESETS))
        self.preset.setToolTip("Hazır DAC tanımları; değiştirince alanlar doldurulur")
        self.preset.currentIndexChanged.connect(self._on_preset)
        dac.add_row("Hazır ayar", self._compact(self.preset))
        self.bits = QSpinBox()
        self.bits.setRange(6, 24)
        self.bits.setValue(12)
        self.vref = self._plain_spin(0.001, 100.0, 4, 3.3, " V")
        self.out_min = self._plain_spin(-100.0, 100.0, 4, 0.0, " V")
        self.out_max = self._plain_spin(-100.0, 100.0, 4, 3.3, " V")
        self.coding = combo([])
        for key, text in sg.DAC_CODINGS.items():
            self.coding.addItem(text, key)
        self.align = combo([])
        for key, text in sg.DAC_ALIGNS.items():
            self.align.addItem(text, key)
        self.word_bits = combo(["8", "16", "32"], 1)
        self._compact(self.coding)
        self._compact(self.align)
        for name, widget in (("Çözünürlük (bit)", self.bits), ("Referans Vref", self.vref),
                             ("Çıkış en küçük", self.out_min), ("Çıkış en büyük", self.out_max),
                             ("Kodlama", self.coding), ("Hizalama", self.align), ("Kelime (bit)", self.word_bits)):
            dac.add_row(name, widget)
        fit = QPushButton("DAC aralığına sığdır")
        fit.setToolTip("Genlik ve ofseti, dalga DAC'ın ulaşabildiği aralığı (%2 pay ile) dolduracak şekilde ayarlar")
        fit.clicked.connect(self._fit_to_dac)
        dac.form.addRow(fit)
        col.addWidget(dac)

        # ---- örnekleme / tablo
        sampling = CollapsibleSection("Örnekleme ve tablo")
        self.fs = self._plain_spin(1.0, 1e9, 2, 48_000.0, " Hz")
        self.fs.setToolTip("DAC güncelleme hızı (zamanlayıcı tetik frekansı)")
        self.mode = combo([])
        for key, text in sg.TABLE_MODES.items():
            self.mode.addItem(text, key)
        self.max_len = QSpinBox()
        self.max_len.setRange(4, sg.MAX_TABLE_LEN)
        self.max_len.setValue(4096)
        self.max_len.setToolTip("Tam periyot kipinde en uzun tablo; DDS'te tablo boyu (2'nin kuvvetine yuvarlanır)")
        self.timer_clock = self._plain_spin(0.001, 1000.0, 3, 84.0, " MHz")
        self.timer_clock.setToolTip("Zamanlayıcının giriş saati (ör. STM32F4 APB1 zamanlayıcıları 84 MHz)")
        self.timer_bits = combo(["16 bit", "32 bit"])
        self._compact(self.mode)
        for name, widget in (("Örnekleme hızı fs", self.fs), ("Tablo kipi", self.mode), ("En uzun tablo", self.max_len),
                             ("Zamanlayıcı saati", self.timer_clock), ("Zamanlayıcı", self.timer_bits)):
            sampling.add_row(name, widget)
        col.addWidget(sampling)

        # ---- C kodu hedefi
        target = CollapsibleSection("C kodu hedefi")
        self.target = combo([])
        for key, text in dac_c.TARGETS.items():
            self.target.addItem(text, key)
        self.code_name = QLineEdit("wave")
        self.code_name.setToolTip("C tanımlayıcılarının öneki ve dosya adı")
        self.cpu_clock = self._plain_spin(1.0, 2000.0, 1, 168.0, " MHz")
        self.dac_channel = QSpinBox()
        self.dac_channel.setRange(1, 2)
        self.interpolate = QCheckBox("DDS doğrusal ara değerleme")
        self._compact(self.target)
        for name, widget in (("Platform", self.target), ("Ad", self.code_name), ("CPU saati", self.cpu_clock),
                             ("DAC kanalı", self.dac_channel)):
            target.add_row(name, widget)
        target.form.addRow(self.interpolate)
        col.addWidget(target)
        col.addStretch(1)

        self.status = label("", "Readout")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.RichText)
        card.body.addWidget(self.status)
        self.code_button = QPushButton("C Kodu Üret…")
        self.code_button.setObjectName("Primary")
        self.code_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.code_button.clicked.connect(self.show_code)
        self.sim_button = QPushButton("Simülatörle Gönder")
        self.sim_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.sim_button.setToolTip("Nicemlenmiş DAC çıkışını simülatör üzerinden UDP akışı olarak gönderir "
                                   "(Canlı İzleme'de incelemek / filtrelemek için)")
        self.sim_button.clicked.connect(self._simulate)
        card.body.addWidget(hbox(self.code_button, self.sim_button, spacing=8))
        self.board_button = QPushButton("Karta Gönder…")
        self.board_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.board_button.setToolTip("Tabloyu çalışan karta Ethernet (UDP) ile gönderir; kart derleme ve yükleme "
                                     "olmadan DAC'tan çalar (kartta 001_signal_generator yazılımı olmalı)")
        self.board_button.clicked.connect(self.show_board_dialog)
        card.body.addWidget(self.board_button)

        for w in (self.amplitude, self.offset, self.bits, self.vref, self.out_min, self.out_max, self.fs,
                  self.max_len, self.timer_clock, self.cpu_clock, self.dac_channel):
            w.valueChanged.connect(self._changed)
        for w in (self.coding, self.align, self.word_bits, self.mode, self.timer_bits, self.target):
            w.currentIndexChanged.connect(self._changed)
        for w in (self.bits, self.vref, self.out_min, self.out_max, self.coding, self.align, self.word_bits):
            (w.valueChanged if hasattr(w, "valueChanged") else w.currentIndexChanged).connect(self._dac_edited)
        self.code_name.textChanged.connect(self._changed)
        self.interpolate.toggled.connect(self._changed)
        return card

    def _spin(self, field: str) -> QDoubleSpinBox:
        lo, hi, dec, step = NUMERIC[field]
        unit = sg.FIELD_INFO[field][1]
        spin = self._plain_spin(lo, hi, dec, getattr(sg.WaveSpec(), field), f" {unit}" if unit in ("Hz", "V", "°", "s")
                                else "")
        spin.setSingleStep(step)
        spin.setToolTip(sg.FIELD_INFO[field][0] + (f" [{unit}]" if unit else ""))
        return spin

    @staticmethod
    def _compact(box: QComboBox, chars: int = 16) -> QComboBox:
        box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        box.setMinimumContentsLength(chars)
        return box

    @staticmethod
    def _plain_spin(lo: float, hi: float, dec: int, value: float, suffix: str = "") -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setDecimals(dec)
        spin.setValue(value)
        spin.setSuffix(suffix)
        spin.setKeyboardTracking(False)
        spin.setMinimumWidth(110)
        return spin

    def _build_editors(self) -> None:
        sec = self.custom_section
        # formül
        self.expr = QLineEdit(sg.WaveSpec().expr)
        self.expr.setFont(mono_font(12))
        variables = "<br>".join(f"<b>{k}</b>: {v}" for k, v in sg.EXPR_VARIABLES.items())
        functions = "<br>".join(f"<b>{k}</b>: {v}" for k, v in sg.EXPR_FUNCTIONS.items())
        self.expr.setToolTip(f"<b>Değişkenler</b><br>{variables}<br><br><b>Fonksiyonlar</b><br>{functions}")
        self.expr.textChanged.connect(self._changed)
        self.expr_error = label("", "Readout")
        self.expr_error.setWordWrap(True)
        self.expr_row = QWidget()
        lay = QVBoxLayout(self.expr_row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(label("Formül — ör. <code>0.6*sin(2*pi*ph) + 0.3*square(ph, 0.2)</code>", "Muted"))
        lay.addWidget(self.expr)
        lay.addWidget(self.expr_error)
        sec.form.addRow(self.expr_row)

        # noktalar / eğri editörü
        self.points_plot = make_plot(self._theme, "Faz (periyot oranı)", "Değer", si_prefix=False)
        self.points_plot.setMinimumHeight(190)
        self.points_plot.setXRange(0, 1, padding=0.02)
        self.points_plot.getPlotItem().getViewBox().setMouseEnabled(x=False, y=False)
        self._points_curve = self.points_plot.plot(pen=pen(self._theme.tokens.text_muted, 1))
        self._roi = None
        self.points_row = QWidget()
        lay = QVBoxLayout(self.points_row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(label("Noktaları sürükleyin · çizgiye tıklayınca nokta eklenir · noktaya sağ tık: sil", "Muted"))
        lay.addWidget(self.points_plot)
        sec.form.addRow(self.points_row)

        # ton / harmonik tablosu
        self.rows_table = QTableWidget(0, 3)
        self.rows_table.verticalHeader().setVisible(False)
        self.rows_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.rows_table.setMinimumHeight(150)
        self.rows_table.itemChanged.connect(lambda *_: self._changed())
        add = icon_button("ph.plus", "Satır ekle", object_name="FlatIcon", size=16)
        remove = icon_button("ph.minus", "Seçili satırı sil", object_name="FlatIcon", size=16)
        add.clicked.connect(self._add_row)
        remove.clicked.connect(self._remove_row)
        self.rows_row = QWidget()
        lay = QVBoxLayout(self.rows_row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.rows_table)
        lay.addWidget(hbox(add, remove, stretch=True))
        sec.form.addRow(self.rows_row)

        # CSV örnekleri
        self.samples_info = label("Örnek yok", "Readout")
        open_csv = QPushButton("CSV / metin dosyası aç…")
        open_csv.clicked.connect(self._open_samples)
        self.samples_row = QWidget()
        lay = QVBoxLayout(self.samples_row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(open_csv)
        lay.addWidget(self.samples_info)
        sec.form.addRow(self.samples_row)

    # ---------------------------------------------------------------- dinamik alanlar
    def _on_kind(self, *_) -> None:
        if not self._loading:
            self._rebuild_wave_fields()
            self._changed()

    def _rebuild_wave_fields(self) -> None:
        form = self.wave_section.form
        for widget in self._field_rows:
            form.removeRow(widget)
        self._field_rows.clear()
        self._spec_fields.clear()
        kind = self.kind.currentData()
        current = self._spec_cache if hasattr(self, "_spec_cache") else sg.WaveSpec()
        for field in sg.PARAMS[kind]:
            if field in EDITOR_FIELDS:
                continue
            if field in sg.CHOICES:
                widget = self._compact(combo([]), 12)
                for key, text in sg.CHOICES[field].items():
                    widget.addItem(text, key)
                widget.setCurrentIndex(max(widget.findData(getattr(current, field)), 0))
                widget.currentIndexChanged.connect(self._changed)
            elif field in INTEGER:
                widget = QSpinBox()
                widget.setRange(*INTEGER[field])
                widget.setValue(int(getattr(current, field)))
                widget.valueChanged.connect(self._changed)
            else:
                widget = self._spin(field)
                widget.setValue(float(getattr(current, field)))
                widget.valueChanged.connect(self._changed)
            widget.setToolTip(sg.FIELD_INFO[field][0])
            form.addRow(_short(field), widget)
            self._field_rows.append(widget)
            self._spec_fields[field] = widget
        self.amplitude.setEnabled(kind != "dc")
        # özel editörler
        fields = set(sg.PARAMS[kind])
        self.expr_row.setVisible("expr" in fields)
        self.points_row.setVisible("points" in fields)
        self.rows_row.setVisible(bool({"tones", "harmonics"} & fields))
        self.samples_row.setVisible("samples" in fields)
        self.custom_section.setVisible(bool(set(EDITOR_FIELDS) & fields))
        if {"tones", "harmonics"} & fields:
            self._fill_rows(current.tones if "tones" in fields else current.harmonics,
                            ("f (Hz)", "Genlik", "Faz (°)") if "tones" in fields else ("n", "Genlik", "Faz (°)"))
        if "points" in fields:
            self._set_points(current.points)

    def _fill_rows(self, rows, headers) -> None:
        self.rows_table.blockSignals(True)
        self.rows_table.setHorizontalHeaderLabels(list(headers))
        self.rows_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, v in enumerate(row):
                self.rows_table.setItem(r, c, QTableWidgetItem(f"{v:g}"))
        self.rows_table.blockSignals(False)

    def _rows(self) -> tuple[tuple[float, float, float], ...]:
        """Tablonun geçerli satırları; sayı olmayan hücre kırmızıyla işaretlenir ve `_bad_rows`'a yazılır."""
        out, bad = [], []
        bad_color, ok_color = pg.mkColor(self._theme.tokens.critical), pg.mkColor(0, 0, 0, 0)
        self.rows_table.blockSignals(True)
        for r in range(self.rows_table.rowCount()):
            values, valid = [], True
            for c in range(3):
                item = self.rows_table.item(r, c)
                try:
                    values.append(float(item.text().replace(",", ".")))
                    item.setBackground(ok_color)
                except (AttributeError, ValueError):
                    valid = False
                    if item is not None:
                        item.setBackground(bad_color)
            if valid:
                out.append(tuple(values))
            else:
                bad.append(r + 1)
        self.rows_table.blockSignals(False)
        self._bad_rows = bad
        return tuple(out)

    def _add_row(self) -> None:
        rows = list(self._rows())
        last = rows[-1] if rows else (1.0, 0.5, 0.0)
        tones = "tones" in sg.PARAMS[self.kind.currentData()]
        rows.append((last[0] * 2 if tones else last[0] + 2, last[1] / 2, 0.0))
        self._fill_rows(rows, [self.rows_table.horizontalHeaderItem(i).text() for i in range(3)])
        self._changed()

    def _remove_row(self) -> None:
        row = self.rows_table.currentRow()
        if row >= 0:
            self.rows_table.removeRow(row)
            self._changed()

    def _set_points(self, points) -> None:
        plot = self.points_plot
        if self._roi is not None:
            plot.removeItem(self._roi)
        self._roi = pg.PolyLineROI([list(p) for p in points], closed=False, movable=False,
                                   pen=pg.mkPen(self._theme.series_color(0), width=2))
        self._roi.sigRegionChangeFinished.connect(self._points_moved)
        plot.addItem(self._roi)
        values = [p[1] for p in points] or [0.0]
        lo, hi = min(values), max(values)
        span = max(hi - lo, 1.0)
        plot.setYRange(lo - 0.2 * span, hi + 0.2 * span, padding=0)

    def _points(self) -> tuple[tuple[float, float], ...]:
        if self._roi is None:
            return sg.WaveSpec().points
        origin = self._roi.pos()
        pts = sorted((min(max(p.x() + origin.x(), 0.0), 1.0), p.y() + origin.y())
                     for p in self._roi.getState()["points"])
        out = []
        for x, y in pts:  # aynı fazda iki nokta olmasın
            if out and abs(x - out[-1][0]) < 1e-6:
                continue
            out.append((round(x, 6), round(y, 6)))
        return tuple(out)

    def _points_moved(self, *_) -> None:
        pts = self._points()
        state = [(p.x() + self._roi.pos().x(), p.y() + self._roi.pos().y()) for p in self._roi.getState()["points"]]
        if [tuple(map(lambda v: round(v, 6), s)) for s in state] != list(pts):  # faz aralığına kırp / sırala
            QTimer.singleShot(0, lambda: self._set_points(pts))
        self._changed()

    def _open_samples(self) -> None:
        s = app_settings()
        path, _ = QFileDialog.getOpenFileName(self, "Örnek dizisi aç", str(s.value("siggen/dir", str(Path.home()))),
                                              "CSV / metin (*.csv *.txt);;Tüm dosyalar (*)")
        if not path:
            return
        s.setValue("siggen/dir", str(Path(path).parent))
        try:
            samples = sg.load_samples_csv(path)
        except (sg.SignalError, OSError, ValueError) as exc:
            self._set_status(f"Dosya okunamadı: {exc}", "critical")
            return
        self._samples = tuple(float(v) for v in samples)
        self.samples_info.setText(f"{Path(path).name} · {len(samples)} örnek · "
                                  f"{float(np.min(samples)):.4g} … {float(np.max(samples)):.4g}")
        self._changed()

    # ---------------------------------------------------------------- spesifikasyon
    def wave_spec(self) -> sg.WaveSpec:
        kind = self.kind.currentData()
        values = {"kind": kind, "amplitude": self.amplitude.value(), "offset": self.offset.value()}
        for field, widget in self._spec_fields.items():
            if isinstance(widget, QComboBox):
                values[field] = widget.currentData()
            else:
                values[field] = widget.value()
        fields = sg.PARAMS[kind]
        if "expr" in fields:
            values["expr"] = self.expr.text()
        if "points" in fields:
            values["points"] = self._points()
        if "tones" in fields:
            values["tones"] = self._rows()
        if "harmonics" in fields:
            values["harmonics"] = self._rows()
        if "samples" in fields:
            values["samples"] = self._samples
        base = self._spec_cache if hasattr(self, "_spec_cache") else sg.WaveSpec()
        merged = {**base.to_dict(), **values}
        return sg.WaveSpec.from_dict(merged)

    def dac_spec(self) -> sg.DacSpec:
        return sg.DacSpec(bits=self.bits.value(), vref=self.vref.value(), out_min=self.out_min.value(),
                          out_max=self.out_max.value(), coding=self.coding.currentData(),
                          align=self.align.currentData(), word_bits=int(self.word_bits.currentText()), channels=1,
                          name=self.preset.currentText() if self.preset.currentIndex() > 0 else "Özel DAC")

    def _on_preset(self, index: int) -> None:
        if index <= 0:
            return
        d = sg.DAC_PRESETS[self.preset.currentText()]
        self._loading = True
        try:
            self.bits.setValue(d.bits)
            self.vref.setValue(d.vref)
            self.out_min.setValue(d.out_min)
            self.out_max.setValue(d.out_max if d.out_max is not None else d.vref)
            self.coding.setCurrentIndex(self.coding.findData(d.coding))
            self.align.setCurrentIndex(self.align.findData(d.align))
            self.word_bits.setCurrentIndex(self.word_bits.findText(str(d.word_bits)))
        finally:
            self._loading = False
        self._changed()

    def _dac_edited(self, *_) -> None:
        if self._loading or self.preset.currentIndex() <= 0:
            return
        d = sg.DAC_PRESETS[self.preset.currentText()]
        now = self.dac_spec()
        if (now.bits, now.vref, now.out_min, now.out_max, now.coding, now.align, now.word_bits) != (
                d.bits, d.vref, d.out_min, d.out_max if d.out_max is not None else d.vref, d.coding, d.align,
                d.word_bits):
            self.preset.blockSignals(True)
            self.preset.setCurrentIndex(0)  # elle değiştirildi: "Özel"
            self.preset.blockSignals(False)

    def _fit_to_dac(self) -> None:
        try:
            spec = sg.fit_to_dac(self.wave_spec(), self.dac_spec(), margin=0.02)
        except sg.SignalError as exc:
            self._set_status(str(exc), "warning")
            return
        self._loading = True
        try:
            self.amplitude.setValue(spec.amplitude)
            self.offset.setValue(spec.offset)
        finally:
            self._loading = False
        self._changed()

    def _changed(self, *_) -> None:
        if not self._loading:
            self._timer.start()

    # ================================================================ grafikler
    def _build_plots(self) -> QWidget:
        outer = QSplitter(Qt.Orientation.Horizontal)
        outer.setChildrenCollapsible(False)
        split = QSplitter(Qt.Orientation.Vertical)
        split.setChildrenCollapsible(False)
        t = self._theme.tokens
        self.ideal_color, self.out_color = t.text_muted, self._theme.series_color(0)

        self.time_card = Card("DAC Çıkışı (zaman)")
        self.time_legend = LegendBar(lambda: [self.time_plot])
        self.time_legend.set_items([(self.out_color, "DAC çıkışı (nicemlenmiş)"),
                                    (self.ideal_color, "ideal", Qt.PenStyle.DashLine)])
        self.time_card.add_left(self.time_legend, optional=True)
        self.time_readout = label("", "Readout")
        self.time_card.add_right(self.time_readout, optional=True)
        self.time_plot = make_plot(self._theme, "Zaman", "Çıkış", x_units="s", y_units="V")
        ideal_pen = pen(self.ideal_color, 1)
        ideal_pen.setStyle(Qt.PenStyle.DashLine)
        self._ideal_curve = self.time_plot.plot(pen=ideal_pen)
        self._out_curve = self.time_plot.plot(pen=pen(self.out_color, 1.5))  # basamak (ZOH) olarak çizilir
        self.time_card.body.addWidget(self.time_plot)
        Crosshair(self.time_plot, self._theme, self._on_time_hover)
        self.time_tips = DataTips(self.time_plot, self._theme, self._time_traces)
        PlotAnalysis(self.time_card, self.time_plot, self._theme, self._time_stats)
        split.addWidget(self.time_card)

        self.spec_card = Card("Spektrum (DAC çıkışı)")
        self.spec_readout = label("", "Readout")
        self.spec_card.add_right(self.spec_readout, optional=True)
        self.spec_plot = make_plot(self._theme, "Frekans", "Seviye (dBFS)", x_units="Hz")
        self._spec_curve = self.spec_plot.plot(pen=pen(self.out_color, 1.2))
        # Tam periyot tablosunun spektrumu çizgi spektrumudur (yalnız f0'ın katlarında enerji): kutular çizgiyle
        # birleştirilirse aradaki boşlukta enerji varmış gibi üçgenler görünür; bu yüzden çubuk (stem) çizilir.
        self._stems = self.spec_plot.plot(pen=pen(self.out_color, 1.4), connect="pairs")
        self._harm = pg.ScatterPlotItem(size=9, pen=pg.mkPen(t.warning, width=1.5), brush=None, symbol="o")
        self.spec_plot.addItem(self._harm)
        self.spec_card.body.addWidget(self.spec_plot)
        Crosshair(self.spec_plot, self._theme, self._on_spec_hover)
        self.spec_tips = DataTips(self.spec_plot, self._theme, self._spec_traces)
        split.addWidget(self.spec_card)
        split.setSizes([460, 380])
        outer.addWidget(split)

        self.metrics_card = Card("Ölçümler")
        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setObjectName("AnalysisTable")
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setHorizontalHeaderLabels(["Ölçüt", "Değer"])
        self.metrics_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        self.metrics_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.metrics_table.setAlternatingRowColors(True)
        self.metrics_table.verticalHeader().setDefaultSectionSize(22)
        self.metrics_card.setMinimumWidth(260)
        self.metrics_card.body.addWidget(self.metrics_table)
        outer.addWidget(self.metrics_card)
        outer.setStretchFactor(0, 1)
        outer.setSizes([620, 440])
        return outer

    # ================================================================ hesap
    def recompute(self) -> None:
        self._timer.stop()
        self._spec_cache = spec = self.wave_spec() if not self._loading else self._spec_cache
        dac = self.dac_spec()
        fs = self.fs.value()
        self.expr_error.setText("")
        try:
            sg.validate(spec, fs)
            table = sg.make_table(spec, dac, fs, max_len=self.max_len.value(), mode=self.mode.currentData())
            m = sg.metrics(table, dds_interpolate=self.interpolate.isChecked())
        except sg.SignalError as exc:
            need = re.search(r"en az ≈ (\d+) örnek gerekir", str(exc))
            if need and int(need.group(1)) <= sg.MAX_TABLE_LEN and int(need.group(1)) > self.max_len.value():
                self.max_len.setValue(int(need.group(1)))  # tek periyot için gereken boy
                log.info("Sinyal üreteci: en uzun tablo %s örneğe çıkarıldı (bir periyot için gerekli)", need.group(1))
                self._timer.stop()
                return self.recompute()
            self.table, self.metrics, self.error = None, None, str(exc)
            if spec.kind == "expr" and exc.position:
                self.expr_error.setText(f'<span style="color:{self._theme.tokens.critical}">'
                                        f'{exc.position}. karakter: {html.escape(str(exc))}</span>')
            self._clear_plots()
            self._set_status(html.escape(str(exc)), "critical")
            self.code_button.setEnabled(False)
            self.sim_button.setEnabled(False)
            self.board_button.setEnabled(False)
            self.summary_chip.setText("Geçersiz tasarım")
            return
        self.table, self.metrics, self.error = table, m, None
        self.code_button.setEnabled(True)
        self.sim_button.setEnabled(True)
        self.board_button.setEnabled(True)
        self._draw_time(table)
        self._draw_spectrum(table, m)
        self._fill_metrics(table, m)
        warn = list(table.warnings)
        if {"tones", "harmonics"} & set(sg.PARAMS[spec.kind]) and getattr(self, "_bad_rows", None):
            warn.insert(0, f"Tablodaki {', '.join(map(str, self._bad_rows))}. satır sayı değil: hesaba katılmadı")
        if table.clipped:
            warn.insert(0, f"{table.clipped} örnek DAC aralığı dışında: kırpıldı (Genliği DAC aralığına sığdır)")
        state = "warning" if warn else "good"
        text = html.escape(sg.describe(spec)) + " · " + (f"f = {format_hz(table.f_real)}" if table.periodic
                                                          else "periyodik değil")
        self._set_status(text + ("<br>" + "<br>".join(f"⚠ {html.escape(w)}" for w in warn) if warn else ""), state)
        self.summary_chip.setText(f"{format_hz(table.f_real) if table.periodic else 'tek atım'} · N = {table.N} · "
                                  f"{table.memory_bytes} B")
        self._save_settings()

    def _output_samples(self, table: sg.Table, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Gösterim için (ideal, nicemlenmiş) çıkış örnekleri. DDS'te C koduyla birebir benzetim."""
        if table.mode == "dds":
            codes = sg.dds_sequence(table, n, interpolate=self.interpolate.isChecked())
            out = sg.codes_to_volts(codes, table.dac)
            ideal = sg.synthesize(table.spec, table.fs, n) if table.spec is not None else out
            return ideal, out
        reps = int(math.ceil(n / table.N))
        out = np.tile(table.volts_out.reshape(table.N, -1)[:, 0], reps)[:n]
        ideal = np.tile(np.asarray(table.values).reshape(table.N, -1)[:, 0], reps)[:n]
        return ideal, out

    def _draw_time(self, table: sg.Table) -> None:
        self.time_tips.clear()
        if table.periodic:
            per = table.samples_per_period
            n = int(min(PLOT_SAMPLES, max(2 * per, 64), max(table.N * 2, 64)))
        else:
            n = int(min(PLOT_SAMPLES, table.N))
        ideal, out = self._output_samples(table, n)
        x = np.arange(n) / table.fs
        self._time_x, self._time_ideal, self._time_out = x, ideal, out
        self._ideal_curve.setData(x, ideal)
        if n:  # sıfır dereceli tutma: her kod bir örnek aralığı boyunca sabit (DAC'ın gerçek çıkışı)
            edges = np.arange(n + 1) / table.fs
            xs = np.repeat(edges, 2)[1:-1]
            xs[1::2] -= 1e-6 / table.fs  # her basamağın sağ ucu kendi örneğine düşer (ipucu sonraki örneği almasın)
            self._out_curve.setData(xs, np.repeat(out, 2))
        else:
            self._out_curve.setData([], [])
        self.time_plot.enableAutoRange()

    def _spectrum(self, table: sg.Table) -> tuple[np.ndarray, np.ndarray]:
        full = (table.dac.v_hi - table.dac.v_lo) / 2  # tam ölçekli sinüsün genliği = 0 dBFS
        if table.mode == "exact" and table.periodic:
            y = table.volts_out.reshape(table.N, -1)[:, 0]
            spec = np.abs(np.fft.rfft(y - y.mean())) / len(y) * 2
        else:
            n = 32768
            _, y = self._output_samples(table, n)
            from scipy.signal import windows
            w = windows.flattop(n, sym=False)  # tepe genliği kutu konumundan bağımsız (dBFS doğru okunur)
            spec = np.abs(np.fft.rfft((y - y.mean()) * w)) / w.sum() * 2
        freqs = np.fft.rfftfreq(len(y), 1 / table.fs)
        return freqs, 20 * np.log10(np.maximum(spec / max(full, 1e-300), 1e-12))

    def _draw_spectrum(self, table: sg.Table, m: sg.Metrics) -> None:
        self.spec_tips.clear()
        f, db = self._spectrum(table)
        floor = max(float(np.min(db[1:])) if len(db) > 1 else -160, -200.0)
        if table.mode == "exact" and table.periodic:  # çizgi spektrumu: çubuk + uçta nokta (ipucu noktaya oturur)
            self._spec_curve.setData(f[1:], db[1:], pen=None, symbol="o", symbolSize=5, symbolPen=None,
                                     symbolBrush=self.out_color)
            self._stems.setData(np.repeat(f[1:], 2), np.column_stack([np.full(len(f) - 1, floor), db[1:]]).ravel())
        else:
            self._spec_curve.setData(f[1:], db[1:], pen=pen(self.out_color, 1.2), symbol=None)
            self._stems.setData([], [])
        if m.harmonics:
            hf = np.array([h[1] for h in m.harmonics])
            hd = np.interp(hf, f, db)
            self._harm.setData(hf, hd)
        else:
            self._harm.setData([], [])
        self.spec_plot.setYRange(floor, 5.0, padding=0)
        self.spec_plot.setXRange(0, table.fs / 2, padding=0.01)

    def _clear_plots(self) -> None:
        for c in (self._ideal_curve, self._out_curve, self._spec_curve, self._stems):
            c.setData([], [])
        self._harm.setData([], [])
        self.metrics_table.setRowCount(0)

    def _fill_metrics(self, table: sg.Table, m: sg.Metrics) -> None:
        def num(v, unit="", digits=4):
            return "—" if v is None or (isinstance(v, float) and not math.isfinite(v)) else fmt(float(v), unit, digits)

        rows = [("— Tablo —", ""),
                ("Kip", sg.TABLE_MODES[table.mode]),
                ("Tablo boyu N", f"{table.N} örnek" + (f" · {table.periods} periyot" if table.periodic else "")),
                ("İstenen frekans", format_hz(table.f_target) if table.f_target and table.periodic else "—"),
                ("Gerçekleşen frekans", format_hz(table.f_real) if table.periodic else "—"),
                ("Frekans hatası", f"{table.freq_error_hz:+.6g} Hz ({table.freq_error_ppm:+.3g} ppm)"
                 if table.periodic and table.f_target else "—"),
                ("Bellek (tablo)", f"{table.memory_bytes} B")]
        if table.mode == "dds":
            rows += [("DDS ayar kelimesi", f"{table.tuning_word}"),
                     ("DDS frekans çözünürlüğü", format_hz(table.freq_resolution))]
        try:
            bits = 32 if self.timer_bits.currentIndex() else 16
            tc = sg.timer_config(table.fs, self.timer_clock.value() * 1e6, bits)
            rows += [("— Zamanlayıcı —", ""), ("PSC / ARR", f"{tc.psc} / {tc.arr}"),
                     ("Gerçekleşen fs", f"{format_hz(tc.fs_real)} ({tc.error_ppm:+.3g} ppm)")]
        except sg.SignalError as exc:
            rows += [("— Zamanlayıcı —", ""), ("PSC / ARR", f"ulaşılamaz: {exc}")]
        rows += [("— Seviyeler —", ""),
                 ("En küçük / en büyük", f"{num(m.vmin, 'V')} / {num(m.vmax, 'V')}"),
                 ("Tepe-tepe", num(m.vpp, "V")), ("Ortalama (DC)", num(m.mean, "V")), ("RMS", num(m.rms, "V")),
                 ("AC RMS", num(m.ac_rms, "V")), ("Crest faktörü", num(m.crest_factor)),
                 ("LSB", num(m.lsb_v, "V")), ("Kırpılan örnek", str(m.clipped)),
                 ("— Kalite —", ""),
                 ("THD", "—" if not math.isfinite(m.thd_pct) else f"%{m.thd_pct:.4g} ({m.thd_db:.2f} dB)"),
                 ("SFDR", "—" if not math.isfinite(m.sfdr_dbc) else f"{m.sfdr_dbc:.2f} dBc"),
                 ("SNR", "—" if not math.isfinite(m.snr_db) else f"{m.snr_db:.2f} dB"),
                 ("SINAD", "—" if not math.isfinite(m.sinad_db) else f"{m.sinad_db:.2f} dB"),
                 ("ENOB", "—" if not math.isfinite(m.enob) else f"{m.enob:.2f} bit"),
                 ("Nicemleme SNR'si", "—" if not math.isfinite(m.quant_snr_db) else f"{m.quant_snr_db:.2f} dB"),
                 ("— Zaman / filtre —", ""),
                 ("Güncelleme aralığı", num(m.update_interval, "s")),
                 ("En büyük eğim", num(m.max_slew_v_per_s, "V/s")),
                 ("İlk görüntü frekansı", format_hz(m.first_image_hz) if math.isfinite(m.first_image_hz) else "—"),
                 ("Önerilen alçak geçiren kesim", format_hz(m.filter_cutoff_hz) if math.isfinite(m.filter_cutoff_hz)
                  else "—"),
                 ("Filtre derecesi (Butterworth)", "—" if m.filter_order is None else
                  f"{m.filter_order} (görüntüde −{m.image_atten_db:.0f} dB)"),
                 ("ZOH düşüşü (en yüksek bileşen)", "—" if not math.isfinite(m.zoh_droop_db)
                  else f"{m.zoh_droop_db:.3f} dB")]
        try:
            mem = dac_c.memory_report(table, table.dac, self.target.currentData(),
                                      interpolate=self.interpolate.isChecked(), cpu_clock_hz=self.cpu_clock.value() * 1e6)
            code = mem.get("code_size_estimate_bytes")
            rows += [("— MCU maliyeti —", ""),
                     ("Flash (tablo)", f"{mem['table_flash_bytes']} B"),
                     ("RAM", f"{mem['ram_bytes']} B"),
                     ("Kod boyutu", (f"~{code} B tahmin · " if code else "") + "kesin değer C Kodu penceresinde ölçülür"),
                     ("CPU yükü (tahmin)", f"%{mem['cpu_load_pct_estimate']:.3g}")]
        except (sg.SignalError, ValueError, KeyError, TypeError) as exc:
            rows.append(("MCU maliyeti", f"hesaplanamadı: {exc}"))
        t = self._theme.tokens
        self.metrics_table.setRowCount(len(rows))
        for r, (name, value) in enumerate(rows):
            a, b = QTableWidgetItem(name), QTableWidgetItem(value)
            b.setToolTip(value)  # dar sütunda kesilen değerin tamamı
            if name.startswith("—"):
                a.setForeground(pg.mkColor(t.accent_text))
                a.setText(name.strip("— ").upper())
            self.metrics_table.setItem(r, 0, a)
            self.metrics_table.setItem(r, 1, b)

    # ================================================================ inceleme
    def _time_traces(self) -> list[Trace]:
        if not len(self._time_x):
            return []
        fs = self.table.fs if self.table else 1.0

        def exact(x, arr):
            i = int(np.clip(math.floor(x * fs + 1e-9), 0, len(arr) - 1))
            return float(i / fs), float(arr[i])

        def describe(x, y, kind):
            n = int(round(x * fs))
            lines = [f"t = {fmt(x, 's', 6)}  (n = {n})", f"{kind} = {fmt(y, 'V', 6)}"]
            if self.table is not None and kind == "çıkış":
                lsb = self.table.dac.lsb
                level = int(round((y - self.table.dac.out_min) / lsb))
                word = int(sg.levels_to_codes(np.array([level]), self.table.dac)[0])
                lines.append(f"seviye = {level}  (LSB {fmt(lsb, 'V', 4)})")
                lines.append(f"C dizisindeki kelime = {word}  (0x{word & 0xFFFFFFFF:X})")
            return lines

        return [Trace(self._out_curve, "DAC çıkışı", self.out_color, lambda x, y: describe(x, y, "çıkış"),
                      lambda x: exact(x, self._time_out)),
                Trace(self._ideal_curve, "İdeal", self.ideal_color, lambda x, y: describe(x, y, "ideal"),
                      lambda x: exact(x, self._time_ideal))]

    def _spec_traces(self) -> list[Trace]:
        def describe(x, y):
            return [f"f = {fmt(x, 'Hz', 7)}", f"{y:.2f} dBFS"]
        return [Trace(self._spec_curve, "Spektrum", self.out_color, describe)]

    def _time_stats(self, x0: float, x1: float) -> list:
        if not len(self._time_x):
            return []
        from spt.gui.widgets.inspector import time_stats
        sl = visible_slice(self._time_x, x0, x1)
        fs = self.table.fs if self.table else 1.0
        t0 = float(self._time_x[sl][0]) if len(self._time_x[sl]) else 0.0
        return [("DAC çıkışı", self.out_color, time_stats(self._time_out[sl], t0, fs, "V")),
                ("İdeal", self.ideal_color, time_stats(self._time_ideal[sl], t0, fs, "V"))]

    def _on_time_hover(self, x: float | None, _y: float | None) -> None:
        if x is None or not len(self._time_x) or self.table is None:
            self.time_readout.setText("")
            return
        i = int(np.clip(math.floor(x * self.table.fs + 1e-9), 0, len(self._time_out) - 1))
        self.time_readout.setText(f"t = {fmt(x, 's', 5)}   çıkış {fmt(float(self._time_out[i]), 'V', 5)}   "
                                  f"ideal {fmt(float(self._time_ideal[i]), 'V', 5)}")

    def _on_spec_hover(self, x: float | None, _y: float | None) -> None:
        xs, ys = self._spec_curve.xData, self._spec_curve.yData
        if x is None or xs is None or not len(xs):
            self.spec_readout.setText("")
            return
        i = int(np.clip(np.searchsorted(xs, x), 0, len(xs) - 1))
        self.spec_readout.setText(f"f = {format_hz(float(xs[i]))}   {float(ys[i]):.2f} dBFS")

    # ================================================================ C kodu ve simülatör
    def show_code(self) -> None:
        if self.table is None:
            return
        if self._code_dialog is None:
            self._code_dialog = CodeDialog(self._theme.tokens, "Sinyal Üreteci — C Kodu", "siggen_code", self)
        self._render_code()
        self._code_dialog.show()
        self._code_dialog.raise_()

    def show_board_dialog(self) -> None:
        if self._board_dialog is None:
            self._board_dialog = BoardSendDialog(self._theme.tokens, lambda: self.table, self)
        self._board_dialog.show()
        self._board_dialog.raise_()

    def _render_code(self) -> None:
        dlg, table = self._code_dialog, self.table
        target = self.target.currentData()
        try:
            files = dac_c.generate(self.wave_spec(), table.dac, table, target, self.code_name.text() or "wave",
                                   fs=table.fs, timer_clock_hz=self.timer_clock.value() * 1e6,
                                   timer_bits=32 if self.timer_bits.currentIndex() else 16,
                                   cpu_clock_hz=self.cpu_clock.value() * 1e6, dac_channel=self.dac_channel.value(),
                                   interpolate=self.interpolate.isChecked())
        except (sg.SignalError, ValueError) as exc:
            dlg.set_files([])
            dlg.set_summary(f"<p>Kod üretilemedi: {exc}</p>")
            dlg.set_status(str(exc), "critical")
            return
        dlg.set_files(files)
        self._measure_code(files)
        rows = "".join(f"<tr><td>{self.metrics_table.item(r, 0).text()}</td>"
                       f"<td>{self.metrics_table.item(r, 1).text()}</td></tr>"
                       for r in range(self.metrics_table.rowCount()))
        self._summary_base = (f"<h3>{html.escape(sg.describe(table.spec))}</h3><p>Platform: "
                              f"<b>{dac_c.TARGETS[target]}</b> · "
                              f"DAC: <b>{table.dac.name}</b></p><table cellpadding='3'>{rows}</table>"
                              "<p>Kullanım adımları <b>KULLANIM.md</b> sekmesinde. CPU yükü tahmindir (gerçek "
                              "donanımda DWT->CYCCNT ile doğrulayın).</p>")
        dlg.set_summary(self._summary_base + "<p><i>Kod boyutu derlenerek ölçülüyor…</i></p>")
        dlg.set_status(f"{len(files)} dosya üretildi · kod boyutu ölçülüyor…")

    def shutdown(self) -> None:
        """Ölçüm daemon iş parçacığında: beklenmez; kapanıştan sonra gelen sonuç yok sayılır."""
        self._size_generation = getattr(self, "_size_generation", 0) + 1

    def _measure_code(self, files) -> None:
        source = next((f.name for f in files if f.name.endswith(".c") and not f.name.endswith("_ornek.c")), None)
        if source is None:
            return
        if getattr(self, "_size_bridge", None) is None:
            self._size_bridge = _SizeBridge(self)
            self._size_bridge.done.connect(self._sizes_measured)
        self.shutdown()  # önceki ölçümün sonucu gelirse yok sayılır (arayüz beklemez)
        files_map = {f.name: f.text for f in files if not f.name.endswith(".md")}
        threading.Thread(target=_measure_sizes, args=(self._size_bridge, self._size_generation, files_map, source,
                                                      MEASURE_PROFILE),
                         name="spt-dac-size", daemon=True).start()

    def _sizes_measured(self, generation: int, rows) -> None:
        dlg = self._code_dialog
        if dlg is None or generation != self._size_generation:
            return
        body = []
        for name, sizes, note in rows:
            if sizes is None:
                body.append(f"<tr><td>{name}</td><td colspan='4'>{note}</td></tr>")
            else:
                body.append(f"<tr><td>{name}</td><td align='right'>{sizes.text}</td><td align='right'>{sizes.rodata}</td>"
                            f"<td align='right'>{sizes.data + sizes.bss}</td><td align='right'>{sizes.flash}</td></tr>")
        table = ("<h4>Kod boyutu — derlenerek ölçüldü (bayt)</h4><table border='1' cellspacing='0' cellpadding='4'>"
                 "<tr><th>Derleyici / çekirdek</th><th>Kod (.text)</th><th>Sabitler + tablo (.rodata)</th>"
                 "<th>RAM (.data + .bss)</th><th>Flash toplam</th></tr>" + "".join(body) + "</table>"
                 "<p>Ölçüm yalnız üretilen <code>.c</code> dosyasının nesnesidir: HAL sürücü kodu ve "
                 "<code>USE_HAL_DRIVER</code> ile korunan başlat / durdur yardımcıları "
                 "(ör. <code>_start</code>, <code>_stop</code>) dahil değildir; "
                 "IAR için GCC değerinin ~%90'ı beklenebilir (tahmin).</p>")
        dlg.set_summary(getattr(self, "_summary_base", "") + table)
        measured = [r for r in rows if r[1] is not None]
        dlg.set_status(f"{len(dlg.files)} dosya üretildi · kod boyutu {'ölçüldü' if measured else 'ölçülemedi'}.",
                       "good" if measured else "warning")

    def _simulate(self) -> None:
        if self.table is None:
            return
        table = self.table
        n = int(max(table.N, min(table.fs * SIM_SECONDS, 2_000_000)))
        if table.mode == "dds":
            # Faz akümülatörü 2³² / gcd(W, 2³²) örnekte başa döner: dosya bunun katıysa döngü kesintisiz
            period = (1 << 32) // math.gcd(int(table.tuning_word), 1 << 32)
            if period <= 4_000_000:
                n = period * max(1, n // period)
            else:
                log.warning("DDS dizisinin periyodu %d örnek: simülatör döngüsünde küçük bir faz sıçraması olur "
                            "(frekansı fs / 2^k'nin katına yaklaştırın).", period)
            _, out = self._output_samples(table, n)
        else:
            reps = max(1, n // table.N)  # tablo (tam periyotlu ya da tek atımlık) tam sayıda tekrar: kesintisiz
            out = np.tile(table.volts_out.reshape(table.N, -1)[:, 0], reps)
        path = Path(tempfile.gettempdir()) / "spt_sinyal_uretici.npy"
        np.save(path, out.astype(np.float64))  # 24 bit DAC seviyeleri float32 mantisine sığmaz
        log.info("Sinyal üreteci: %s simülatöre gönderiliyor (%d örnek, %s)", sg.describe(table.spec), len(out),
                 format_hz(table.fs))
        self.simulate_requested.emit(str(path), float(table.fs))

    # ================================================================ ayarlar
    def _save_settings(self) -> None:
        s = app_settings()
        try:
            data = self.wave_spec().to_dict()
            data["samples"] = []  # büyük olabilir: kaydedilmez
            s.setValue("siggen/spec", json.dumps(data))
        except sg.SignalError:
            pass
        s.setValue("siggen/dac", json.dumps({"preset": self.preset.currentText(), "bits": self.bits.value(),
                                             "vref": self.vref.value(), "out_min": self.out_min.value(),
                                             "out_max": self.out_max.value(), "coding": self.coding.currentData(),
                                             "align": self.align.currentData(),
                                             "word": self.word_bits.currentText()}))
        s.setValue("siggen/opts", json.dumps({"fs": self.fs.value(), "mode": self.mode.currentData(),
                                              "max_len": self.max_len.value(), "timer": self.timer_clock.value(),
                                              "timer_bits": self.timer_bits.currentIndex(),
                                              "target": self.target.currentData(), "name": self.code_name.text(),
                                              "cpu": self.cpu_clock.value()}))

    def _load_settings(self) -> None:
        s = app_settings()
        self._loading = True
        try:
            try:
                spec = sg.WaveSpec.from_dict(json.loads(str(s.value("siggen/spec", "{}"))))
            except (ValueError, TypeError, sg.SignalError):
                spec = sg.WaveSpec(kind="sine", freq=1000.0, amplitude=1.5, offset=1.65)
            if not s.contains("siggen/spec"):
                spec = sg.WaveSpec(kind="sine", freq=1000.0, amplitude=1.5, offset=1.65)  # 0–3.3 V DAC'a uygun
            self._spec_cache = spec
            self.kind.setCurrentIndex(max(self.kind.findData(spec.kind), 0))
            self.amplitude.setValue(spec.amplitude)
            self.offset.setValue(spec.offset)
            self.expr.setText(spec.expr)
            dac = json.loads(str(s.value("siggen/dac", "{}")) or "{}")
            preset = dac.get("preset", next(iter(sg.DAC_PRESETS)))
            index = self.preset.findText(preset)
            if index < 0:   # kaldırılmış hazır ayar (ör. eski ESP32 / Arduino): kayıtlı alanlar yerine ilk STM32 ayarı
                index, dac = 1, {}
            self.preset.setCurrentIndex(index)
            if dac:
                self.bits.setValue(int(dac.get("bits", 12)))
                self.vref.setValue(float(dac.get("vref", 3.3)))
                self.out_min.setValue(float(dac.get("out_min", 0.0)))
                self.out_max.setValue(float(dac.get("out_max", 3.3)))
                self.coding.setCurrentIndex(max(self.coding.findData(dac.get("coding")), 0))
                self.align.setCurrentIndex(max(self.align.findData(dac.get("align")), 0))
                self.word_bits.setCurrentIndex(max(self.word_bits.findText(str(dac.get("word", "16"))), 0))
            else:
                self._loading = False
                self._on_preset(self.preset.currentIndex())
                self._loading = True
            opts = json.loads(str(s.value("siggen/opts", "{}")) or "{}")
            self.fs.setValue(float(opts.get("fs", 48_000.0)))
            self.mode.setCurrentIndex(max(self.mode.findData(opts.get("mode", "exact")), 0))
            self.max_len.setValue(int(opts.get("max_len", 4096)))
            self.timer_clock.setValue(float(opts.get("timer", 84.0)))
            self.timer_bits.setCurrentIndex(int(opts.get("timer_bits", 0)))
            self.target.setCurrentIndex(max(self.target.findData(opts.get("target", "stm32_hal_dma")), 0))
            self.code_name.setText(str(opts.get("name", "wave")))
            self.cpu_clock.setValue(float(opts.get("cpu", 168.0)))
        except (ValueError, TypeError) as exc:
            log.warning("Sinyal üreteci ayarları okunamadı (%s); varsayılanlar kullanılıyor.", exc)
        finally:
            self._loading = False

    def _set_status(self, text: str, state: str = "") -> None:
        t = self._theme.tokens
        color = {"good": t.good, "warning": t.warning, "critical": t.critical}.get(state)
        self.status.setText((f'<span style="color:{color}">●</span> ' if color else "") + text)



