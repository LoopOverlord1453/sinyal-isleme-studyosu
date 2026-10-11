"""Filtre Tasarımcısı: tasarım formu ve cevap grafikleri.

Her parametre değişikliğinde filtre yeniden tasarlanır ve grafikler anında güncellenir:
genlik cevabı (+ tasarım maskesi ve isteğe bağlı sabit nokta kuantizasyonu), faz, grup gecikmesi, impuls ve
basamak cevabı, kutup-sıfır diyagramı. Yönteme göre ilgisiz alanlar gizlenir. "Sinyale Uygula" tasarımı işlem
zincirine ekler (ya da düzenlenen aşamayı günceller) ve canlı akışa uygular.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QMenu,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from spt.dsp import coefficients as coeffs
from spt.dsp.filters import (
    BANDS,
    COMB_TYPES,
    FAMILIES,
    KALMAN_MODELS,
    MAX_FIR_TAPS,
    MAX_IIR_ORDER,
    MAX_LENGTH,
    METHODS,
    TWO_EDGE,
    Design,
    FilterError,
    FilterSpec,
    analyze,
    check_mask,
    design,
    format_hz,
    impulse_response,
    mask_for,
    step_response,
    uses_atten,
    uses_ripple,
)
from spt.gui.actions import ActionRegistry
from spt.gui.theme import ThemeManager
from spt.gui.widgets.common import Card, PageHeader, chip, combo, form_layout, hbox, icon_button, label
from spt.gui.widgets.inspector import DataTips, PlotAnalysis, Trace, curve_stats, fmt, visible_slice
from spt.gui.widgets.plot import Crosshair, make_plot, pen, set_si_prefix

REDESIGN_MS = 60
QUANT_CHOICES = [("İdeal", None), ("float32", "float32"), ("Q31", "q31"), ("Q15", "q15")]
ANALYSIS_MODES = [("phase", "Faz"), ("gd", "Grup gecikmesi"), ("impulse", "İmpuls"), ("step", "Basamak")]
STEM_LIMIT = 160  # bundan kısa impuls/basamak cevapları gövde (stem) grafiği olarak çizilir
MIN_FIELD_WIDTH = 90


def _freq_spin(value: float) -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setMinimumWidth(MIN_FIELD_WIDTH)  # aralığın en büyük değeri (1e9) kutuyu gereksiz genişletmesin
    w.setRange(0.0, 1e9)
    w.setDecimals(3)
    w.setValue(value)
    w.setSuffix(" Hz")
    w.setKeyboardTracking(False)
    return w


def _dspin(lo: float, hi: float, value: float, decimals: int, suffix: str = "") -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setMinimumWidth(MIN_FIELD_WIDTH)
    w.setRange(lo, hi)
    w.setDecimals(decimals)
    w.setValue(value)
    w.setSuffix(suffix)
    w.setKeyboardTracking(False)
    return w


def _spin(lo: int, hi: int, value: int) -> QSpinBox:
    w = QSpinBox()
    w.setMinimumWidth(MIN_FIELD_WIDTH)
    w.setRange(lo, hi)
    w.setValue(value)
    w.setKeyboardTracking(False)
    return w


def _segment(text: str, tip: str = "") -> QToolButton:
    btn = QToolButton()
    btn.setObjectName("Segment")
    btn.setText(text)
    btn.setToolTip(tip)
    btn.setCheckable(True)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    return btn


def _hint(spec: FilterSpec) -> str:
    m = spec.method
    if spec.family == "special":
        return {
            "notch": "Çentik f₀'da ve harmoniklerinde (k·f₀) aynı bant genişliğiyle (f₀ / Q) açılır.",
            "dc_block": "1. derece yüksek geçiren: y[n] = g·(x[n] − x[n−1]) + r·y[n−1]; kesimde −3 dB.",
            "moving_avg": "Son N örneğin ortalaması. Gecikme (N−1)/2 örnek; ilk sıfır fs/N'de.",
            "median": "Doğrusal olmayan: darbe gürültüsünü (spike) temizler, kenarları korur. "
                      "Frekans cevabı ve katsayısı yoktur.",
            "savgol": "Her pencerede polinom uydurur; tepe biçimini korur. Gecikme (N−1)/2 örnek.",
            "comb": "fs / f₀ tam sayı olmalı. Çentik: f₀ ve tüm katlarını (DC dahil) bastırır — şebeke ve "
                    "harmoniklerini tek filtrede temizler. Tepe: yalnız f₀ ve katlarını geçirir.",
            "ema": "y[n] = y[n−1] + α·(x[n] − y[n−1]). α, kesim frekansında tam −3 dB olacak şekilde hesaplanır.",
            "kalman": "Sabit kazançlı (kararlı hâl) Kalman: kazanç Riccati denkleminden. σ_ölçüm büyüdükçe / σ_süreç "
                      "küçüldükçe daha çok yumuşatır. Sabit hız modeli rampayı gecikmesiz izler.",
            "alpha_beta": "Konum + hız izleyici: x̂ = x̂ₚ + α·r, v̂ = v̂ₚ + (β/T)·r. Kararlılık: 0 < α < 2, "
                          "0 < β < 4 − 2α.",
            "gaussian": "Normalize Gauss çekirdek (DC kazancı 1, aşma yok). Gecikme (N−1)/2 örnek; "
                        "−3 dB ≈ 0.1325·fs/σ.",
            "differentiator": "Parks-McClellan türev alıcı: geçirme bandında H(f) = j·2πf (çıkış birim/s); "
                              "f₁ + Δf üstü bastırılır (gürültü büyütülmez).",
            "integrator": "y[n] = λ·y[n−1] + T·x[n]: köşe frekansının üstünde integral (1/2πf), altında sınırlı "
                          "kazanç — sensör ofsetiyle sürüklenmez. Çıkış birimi × s.",
            "hilbert": "90° faz kaydırıcı (FIR, tip III): f₁ … fs/2 − f₁ bandında |H| ≈ 1. Analitik sinyal / zarf "
                       "için; gecikme N/2 örnek.",
            "allpass": "|H| = 1 (genlik değişmez), faz f₀'da −180°. Faz / grup gecikmesi düzeltmede kullanılır.",
        }[m]
    if spec.uses_band_edges:
        return {
            "lowpass": "f₁ geçirme bandı kenarı; durdurma bandı f₁ + Δf'de başlar.",
            "highpass": "f₁ geçirme bandı kenarı; durdurma bandı f₁ − Δf'de biter.",
            "bandpass": "f₁ … f₂ geçirme bandı; durdurma bantları f₁ − Δf altında ve f₂ + Δf üstünde.",
            "bandstop": "f₁ ve f₂ geçirme bandı kenarları; durdurma bandı f₁ + Δf … f₂ − Δf arası.",
        }[spec.band] + (" Derece, maskeyi karşılayan en küçük değer seçilir." if spec.auto_order else "")
    if spec.family == "iir":
        if m in ("butter", "bessel"):
            return "f₁ (ve f₂) −3 dB kesim frekansıdır."
        if m in ("cheby1", "ellip"):
            return "f₁ (ve f₂) geçirme bandı kenarıdır: kazanç burada −dalgalanma dB'ye iner."
        return "f₁ (ve f₂) durdurma bandının başladığı frekanstır: kazanç burada −bastırma dB'dir."
    return "f₁ (ve f₂) −6 dB kesim frekansıdır (geçiş bandının ortası). N. derece = N + 1 katsayı."


def _set_log_x(plot: pg.PlotWidget, on: bool) -> None:
    """Logaritmik frekans ekseni. Geçiş anında görünüm hâlâ doğrusal aralıktadır (ör. 0 … 5000); pyqtgraph bunu
    10^5000 olarak yorumlayıp taşar. Önce zararsız bir aralığa (10^0 … 10^1) alınır, sonra çizim kendi aralığını
    kurar."""
    pi = plot.getPlotItem()
    if on and not pi.ctrl.logXCheck.isChecked():
        pi.setXRange(0.0, 1.0, padding=0)
    with np.errstate(over="ignore"):
        pi.setLogMode(x=on, y=False)


def _multiplicities(points: np.ndarray) -> list[tuple[complex, int]]:
    """Çakışan kök gruplarını (yaklaşık eşitlik) ve sayılarını döndürür."""
    out: list[list] = []
    for p in points:
        for item in out:
            if abs(item[0] - p) < 1e-6 * max(1.0, abs(p)):
                item[1] += 1
                break
        else:
            out.append([p, 1])
    return [(complex(p), n) for p, n in out]


class FilterDesignerView(QWidget):
    spec_changed = pyqtSignal()                 # form değişti (geri al yığını için)
    design_changed = pyqtSignal(object)         # Design | None
    apply_requested = pyqtSignal(object, object)  # FilterSpec, düzenlenen aşama (None: yeni aşama)
    editing_cancelled = pyqtSignal()

    def __init__(self, theme: ThemeManager, actions: ActionRegistry):
        super().__init__()
        self._theme = theme
        self.design: Design | None = None
        self.error: str | None = None
        self.editing: int | None = None
        self.log_freq = False
        self.analysis_mode = "phase"
        self._loading = False
        self._mag_manual = False
        self._mask_items: list[pg.GraphicsObject] = []
        self._marker_items: list[pg.GraphicsObject] = []
        self._pz_items: list[pg.GraphicsObject] = []
        self._response = None
        self._response_of = None    # _response hangi tasarımın (görünüm işlemlerinde yeniden hesaplanmaz)
        self._designed_spec = None  # son tasarlanan spesifikasyon
        self._quant_design = None  # kuantize tasarım (imleç / ipucu tam değeri için)
        self._quant_cache = None   # (tasarım, biçim, Quantized, tasarım, cevap, rapor)
        self._time_response = np.zeros(0)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        self.header = PageHeader("Filtre Tasarımcısı", "FIR / IIR tasarım, cevap analizi ve katsayı dışa aktarma")
        self.summary_chip = chip("")
        self.header.add_right(hbox(icon_button(action=actions["edit.undo"]), icon_button(action=actions["edit.redo"]),
                                   spacing=4))
        self.header.add_right(self.summary_chip)
        lay.addWidget(self.header)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        lay.addWidget(split, 1)
        split.addWidget(self._build_form(actions))
        split.addWidget(self._build_plots())
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([340, 900])

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(REDESIGN_MS)
        self._timer.timeout.connect(self.redesign)
        self._update_fields()
        self.redesign()

    # ================================================================ form
    def _build_form(self, actions: ActionRegistry) -> Card:
        card = Card("Tasarım")
        card.setMinimumWidth(320)
        card.setMaximumWidth(440)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        host = QWidget()
        col = QVBoxLayout(host)
        col.setContentsMargins(0, 0, 6, 0)
        col.setSpacing(8)
        scroll.setWidget(host)
        card.body.addWidget(scroll, 1)

        self._family_buttons: dict[str, QToolButton] = {}
        group = QButtonGroup(self)
        for key, text in FAMILIES.items():
            btn = _segment(text, {"iir": "Sonsuz impuls cevaplı: düşük derecede keskin geçiş, doğrusal olmayan faz",
                                  "fir": "Sonlu impuls cevaplı: her zaman kararlı, doğrusal faz yapılabilir",
                                  "special": "Notch, DC engelleyici, hareketli ortalama, medyan, Savitzky-Golay"}[key])
            btn.clicked.connect(lambda _=False, k=key: self._set_family(k))
            group.addButton(btn)
            self._family_buttons[key] = btn
        self._family_buttons["iir"].setChecked(True)
        col.addWidget(hbox(*self._family_buttons.values(), spacing=2, stretch=True))

        form = form_layout()
        self._form = form
        self._rows: dict[str, tuple] = {}
        defaults = FilterSpec()

        self.method = self._combo([])
        self.band = self._combo([(text, key) for key, text in BANDS.items()])
        for box in (self.method, self.band):  # uzun yöntem adları formu genişletmesin (liste açılınca tam görünür)
            box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            box.setMinimumContentsLength(10)
        self.fs = _freq_spin(defaults.fs)
        self.fs.setMinimum(0.001)
        self.fs.setDecimals(1)
        self.fs_menu = QMenu(self)
        self.fs_button = QToolButton()
        self.fs_button.setObjectName("FlatIcon")
        self.fs_button.setText("Akış")
        self.fs_button.setToolTip("Örnekleme hızını bir akıştan al")
        self.fs_button.setMenu(self.fs_menu)
        self.fs_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.fs_button.setEnabled(False)
        self.order = _spin(1, MAX_FIR_TAPS - 1, defaults.order)
        self.auto_order = QCheckBox("Otomatik derece")
        self.auto_order.setToolTip("Dereceyi geçirme / durdurma bandı spesifikasyonundan (Δf, dalgalanma, bastırma) "
                                   "hesapla")
        self.f1 = _freq_spin(defaults.f1)
        self.f2 = _freq_spin(defaults.f2)
        self.transition = _freq_spin(defaults.transition)
        self.ripple = _dspin(0.001, 20.0, defaults.ripple_db, 3, " dB")
        self.atten = _dspin(1.0, 200.0, defaults.atten_db, 1, " dB")
        self.beta = _dspin(0.0, 40.0, defaults.kaiser_beta, 2)
        self.notch_f0 = _freq_spin(defaults.notch_f0)
        self.notch_q = _dspin(0.1, 10_000.0, defaults.notch_q, 1)
        self.harmonics = _spin(1, 50, defaults.harmonics)
        self.dc_cutoff = _freq_spin(defaults.dc_cutoff)
        self.length = _spin(1, MAX_LENGTH, defaults.length)
        self.polyorder = _spin(0, 20, defaults.polyorder)
        self.comb_type = self._combo([(text, key) for key, text in COMB_TYPES.items()])
        self.kalman_model = self._combo([(text, key) for key, text in KALMAN_MODELS.items()])
        self.kalman_q = _dspin(1e-6, 1e9, defaults.kalman_q, 6)
        self.kalman_r = _dspin(1e-9, 1e9, defaults.kalman_r, 6)
        self.ab_alpha = _dspin(0.0001, 1.9999, defaults.ab_alpha, 4)
        self.ab_beta = _dspin(0.0001, 3.9999, defaults.ab_beta, 4)
        self.gauss_sigma = _dspin(0.1, 10_000.0, defaults.gauss_sigma, 2, " örnek")
        for box in (self.comb_type, self.kalman_model):
            box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            box.setMinimumContentsLength(10)

        self._add_row("method", "Yöntem", self.method)
        self._add_row("band", "Bant tipi", self.band)
        self._add_row("fs", "Örnekleme", hbox(self.fs, self.fs_button, spacing=4))
        self._add_row("order", "Derece", self.order)
        self._add_row("auto", "", self.auto_order)
        self._add_row("f1", "f₁", self.f1)
        self._add_row("f2", "f₂", self.f2)
        self._add_row("transition", "Geçiş bandı Δf", self.transition)
        self._add_row("ripple", "Geçirme dalgalanması", self.ripple)
        self._add_row("atten", "Durdurma", self.atten)
        self._add_row("beta", "Kaiser β", self.beta)
        self._add_row("notch_f0", "Merkez f₀", self.notch_f0)
        self._add_row("notch_q", "Kalite (Q)", self.notch_q)
        self._add_row("harmonics", "Harmonik", self.harmonics)
        self._add_row("dc_cutoff", "Kesim", self.dc_cutoff)
        self._add_row("length", "Pencere boyu", self.length)
        self._add_row("polyorder", "Polinom derecesi", self.polyorder)
        self._add_row("comb_type", "Tür", self.comb_type)
        self._add_row("kalman_model", "Model", self.kalman_model)
        self._add_row("kalman_q", "σ süreç", self.kalman_q)
        self._add_row("kalman_r", "σ ölçüm", self.kalman_r)
        self._add_row("ab_alpha", "α", self.ab_alpha)
        self._add_row("ab_beta", "β", self.ab_beta)
        self._add_row("gauss_sigma", "σ", self.gauss_sigma)
        self.kalman_r.setToolTip("Ölçüm gürültüsünün standart sapması (sinyalin biriminde)")
        self.ripple.setToolTip("Geçirme bandında izin verilen tepe-tepe dalgalanma")
        self.atten.setToolTip("Durdurma bandında en az bu kadar bastırma")
        self.harmonics.setToolTip("1: yalnızca f₀. 3: f₀, 2f₀, 3f₀ (ör. şebeke ve harmonikleri)")
        col.addLayout(form)

        self.hint = label("", "Muted")
        self.hint.setWordWrap(True)
        col.addWidget(self.hint)

        info = QFrame()
        info.setObjectName("Inset")
        info_lay = QVBoxLayout(info)
        info_lay.setContentsMargins(10, 8, 10, 8)
        self.info = label("", "Readout")
        self.info.setWordWrap(True)
        self.info.setTextFormat(Qt.TextFormat.RichText)
        info_lay.addWidget(self.info)
        col.addWidget(info)
        col.addStretch(1)

        self.edit_label = label("", "Muted")
        self.edit_label.setWordWrap(True)
        self.edit_cancel = QPushButton("Düzenlemeyi bırak")
        self.edit_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        self.edit_cancel.clicked.connect(self.stop_editing)
        self.edit_box = QFrame()
        self.edit_box.setObjectName("Inset")
        edit_lay = QVBoxLayout(self.edit_box)
        edit_lay.setContentsMargins(10, 8, 10, 8)
        edit_lay.addWidget(self.edit_label)
        edit_lay.addWidget(self.edit_cancel)
        self.edit_box.hide()
        card.body.addWidget(self.edit_box)

        self.apply_button = QPushButton("Sinyale Uygula")
        self.apply_button.setObjectName("Primary")
        self.apply_button.setIcon(actions["flt.apply"].icon())
        self.apply_button.setToolTip("Tasarımı işlem zincirine ekler (düzenlenen aşama varsa onu günceller) ve "
                                     "canlı akışa uygular")
        self.apply_button.clicked.connect(actions["flt.apply"].trigger)
        export = QPushButton("Katsayıları Dışa Aktar…")
        export.setIcon(actions["flt.export"].icon())
        export.clicked.connect(actions["flt.export"].trigger)
        self.export_button = export
        for b in (self.apply_button, export):
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            card.body.addWidget(b)

        for w in (self.method, self.band, self.comb_type, self.kalman_model):
            w.currentIndexChanged.connect(self._on_changed)
        for w in (self.fs, self.order, self.f1, self.f2, self.transition, self.ripple, self.atten, self.beta,
                  self.notch_f0, self.notch_q, self.harmonics, self.dc_cutoff, self.length, self.polyorder,
                  self.kalman_q, self.kalman_r, self.ab_alpha, self.ab_beta, self.gauss_sigma):
            w.valueChanged.connect(self._on_changed)
        self.auto_order.toggled.connect(self._on_changed)
        self._fill_methods("iir")
        return card

    @staticmethod
    def _combo(items: list[tuple[str, str]]):
        w = combo([])
        for text, key in items:
            w.addItem(text, key)
        return w

    def _add_row(self, key: str, text: str, widget: QWidget) -> None:
        lbl = label(text, "FormLabel")
        self._form.addRow(lbl, widget)
        self._rows[key] = (lbl, widget)

    def _fill_methods(self, family: str, method: str | None = None) -> None:
        self.method.blockSignals(True)
        self.method.clear()
        for key, text in METHODS[family].items():
            self.method.addItem(text, key)
        index = self.method.findData(method) if method else 0
        self.method.setCurrentIndex(max(index, 0))
        self.method.blockSignals(False)

    @property
    def family(self) -> str:
        return next(k for k, b in self._family_buttons.items() if b.isChecked())

    def _set_family(self, family: str, method: str | None = None) -> None:
        self._family_buttons[family].setChecked(True)
        self._fill_methods(family, method)
        self._on_changed()

    def select(self, family: str, method: str | None = None) -> None:
        """Menüden (ör. Filtre ▸ IIR Filtre / Özel Filtreler ▸ Notch) aile ve yöntem seçimi."""
        if family == self.family and (method is None or method == self.method.currentData()):
            return
        self._set_family(family, method)

    def _on_changed(self, *_) -> None:
        if self._loading:
            return
        self._update_fields()
        self._timer.start()
        self.spec_changed.emit()

    # ---------------------------------------------------------------- form ↔ spesifikasyon
    def spec(self) -> FilterSpec:
        return FilterSpec(
            family=self.family,
            method=self.method.currentData() or next(iter(METHODS[self.family])),
            band=self.band.currentData(),
            fs=self.fs.value(),
            order=self.order.value(),
            auto_order=self.auto_order.isChecked(),
            f1=self.f1.value(),
            f2=self.f2.value(),
            transition=self.transition.value(),
            ripple_db=self.ripple.value(),
            atten_db=self.atten.value(),
            kaiser_beta=self.beta.value(),
            notch_f0=self.notch_f0.value(),
            notch_q=self.notch_q.value(),
            harmonics=self.harmonics.value(),
            dc_cutoff=self.dc_cutoff.value(),
            length=self.length.value(),
            polyorder=self.polyorder.value(),
            comb_type=self.comb_type.currentData(),
            kalman_model=self.kalman_model.currentData(),
            kalman_q=self.kalman_q.value(),
            kalman_r=self.kalman_r.value(),
            ab_alpha=self.ab_alpha.value(),
            ab_beta=self.ab_beta.value(),
            gauss_sigma=self.gauss_sigma.value(),
        )

    def set_spec(self, spec: FilterSpec, emit: bool = False) -> None:
        self._loading = True
        try:
            self._family_buttons[spec.family].setChecked(True)
            self._fill_methods(spec.family, spec.method)
            self.band.setCurrentIndex(max(self.band.findData(spec.band), 0))
            if spec.family == "iir":
                self.order.setMaximum(MAX_IIR_ORDER)
            else:
                self.order.setMaximum(MAX_FIR_TAPS - 1)
            for widget, value in ((self.fs, spec.fs), (self.order, spec.order), (self.f1, spec.f1),
                                  (self.f2, spec.f2), (self.transition, spec.transition),
                                  (self.ripple, spec.ripple_db), (self.atten, spec.atten_db),
                                  (self.beta, spec.kaiser_beta), (self.notch_f0, spec.notch_f0),
                                  (self.notch_q, spec.notch_q), (self.harmonics, spec.harmonics),
                                  (self.dc_cutoff, spec.dc_cutoff), (self.length, spec.length),
                                  (self.polyorder, spec.polyorder), (self.kalman_q, spec.kalman_q),
                                  (self.kalman_r, spec.kalman_r), (self.ab_alpha, spec.ab_alpha),
                                  (self.ab_beta, spec.ab_beta), (self.gauss_sigma, spec.gauss_sigma)):
                widget.setValue(value)
            self.comb_type.setCurrentIndex(max(self.comb_type.findData(spec.comb_type), 0))
            self.kalman_model.setCurrentIndex(max(self.kalman_model.findData(spec.kalman_model), 0))
            self.auto_order.setChecked(spec.auto_order)
        finally:
            self._loading = False
        self._update_fields()
        self.redesign()
        if emit:
            self.spec_changed.emit()

    def _update_fields(self) -> None:
        spec = self.spec()
        fam, m = spec.family, spec.method
        rows = {"method", "fs"}
        if fam in ("iir", "fir"):
            rows |= {"band", "order", "auto", "f1"}
            if spec.band in TWO_EDGE:
                rows.add("f2")
            if spec.uses_band_edges:
                rows.add("transition")
            if uses_ripple(spec):
                rows.add("ripple")
            if uses_atten(spec):
                rows.add("atten")
            if m == "kaiser" and not spec.auto_order:
                rows.add("beta")
        elif m == "notch":
            rows |= {"notch_f0", "notch_q", "harmonics"}
        elif m in ("comb", "allpass"):
            rows |= {"notch_f0", "notch_q"} | ({"comb_type"} if m == "comb" else set())
        elif m in ("dc_block", "ema", "integrator"):
            rows.add("dc_cutoff")
        elif m == "kalman":
            rows |= {"kalman_model", "kalman_q", "kalman_r"}
        elif m == "alpha_beta":
            rows |= {"ab_alpha", "ab_beta"}
        elif m in ("differentiator", "hilbert"):
            rows |= {"order", "f1"} | ({"transition"} if m == "differentiator" else set())
        else:
            rows.add("length")
            if m == "savgol":
                rows.add("polyorder")
            if m == "gaussian":
                rows.add("gauss_sigma")
        for key, (lbl, widget) in self._rows.items():
            self._form.setRowVisible(lbl, key in rows)
        self.order.setEnabled(not (spec.auto_order and fam in ("iir", "fir")))
        self.order.setMaximum(MAX_IIR_ORDER if fam == "iir" else MAX_FIR_TAPS - 1)
        order_label = self._rows["order"][0]
        order_label.setText("Derece (N)" if fam == "fir" else "Derece")
        self.order.setToolTip("FIR: N. derece = N + 1 katsayı" if fam == "fir" else
                              "Prototip derecesi; bant geçiren / durduranda gerçek derece 2 katıdır")
        edge = spec.uses_band_edges or m == "differentiator"
        self._rows["f1"][0].setText("Alt kenar f₁" if m == "hilbert" else "Kenar f₁" if edge else "Kesim f₁")
        self._rows["f2"][0].setText("Kenar f₂" if spec.uses_band_edges else "Kesim f₂")
        self._rows["dc_cutoff"][0].setText("Köşe frekansı" if m == "integrator" else "Kesim (−3 dB)")
        self._rows["notch_f0"][0].setText("Merkez f₀" if m in ("notch", "allpass") else "Temel f₀")
        # süreç gürültüsü yoğunluğu: sabit hızda beyaz ivme (birim/s² /√Hz = birim/s^1.5), sabit seviyede
        # beyaz hız (birim/s /√Hz = birim/√s)
        unit = "birim/s^1.5" if spec.kalman_model == "cv" else "birim/√s"
        self._rows["kalman_q"][0].setText(f"σ süreç ({unit})")
        self.hint.setText(_hint(spec))

    # ================================================================ akışlar / düzenleme
    def set_stream_rates(self, rates: list[tuple[str, float]]) -> None:
        """Örnekleme hızı menüsü: bağlı akışların hızları."""
        self.fs_menu.clear()
        for name, fs in rates:
            act = self.fs_menu.addAction(f"{name} · {format_hz(fs)}")
            act.triggered.connect(lambda _=False, v=fs: self.fs.setValue(v))
        self.fs_button.setEnabled(bool(rates))

    def start_editing(self, index: int, spec: FilterSpec) -> None:
        self.editing = index
        self.set_spec(spec)
        self.edit_label.setText(f"Zincirin <b>{index + 1}. adımı</b> düzenleniyor. “Adımı Güncelle” bu adımı "
                                "değiştirir.")
        self.edit_box.show()
        self.apply_button.setText("Adımı Güncelle")

    def stop_editing(self) -> None:
        if self.editing is None:
            return
        self.editing = None
        self.edit_box.hide()
        self.apply_button.setText("Sinyale Uygula")
        self.editing_cancelled.emit()

    def request_apply(self) -> None:
        if self._timer.isActive() or self._designed_spec != self.spec():  # tasarım güncelse yeniden tasarlama
            self.redesign()
        if self.design is None:
            return
        self.apply_requested.emit(self.spec(), self.editing)

    # ================================================================ grafikler
    def _build_plots(self) -> QWidget:
        t = self._theme.tokens
        split = QSplitter(Qt.Orientation.Vertical)
        split.setChildrenCollapsible(False)

        self.mag_card = Card("Genlik Cevabı")
        self.mag_plot = make_plot(self._theme, "Frekans", "Genlik (dB)", x_units="Hz")
        set_si_prefix(self.mag_plot.getPlotItem().getAxis("left"), False)
        self.quant = self._combo([(text, key) for text, key in QUANT_CHOICES])
        self.quant.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.quant.setMinimumContentsLength(8)  # dar kartta da seçili metin okunur
        self.quant.setToolTip("Katsayıları seçilen sayı biçimine yuvarlayıp cevabı kesikli çizgiyle üst üste çizer "
                              "(yalnızca katsayı kuantizasyonu; aritmetik yuvarlama modellenmez). İdeal: kuantizasyon yok.")
        self.quant.currentIndexChanged.connect(lambda *_: self._draw())
        self.log_button = _segment("Log f", "Logaritmik frekans ekseni")
        self.log_button.toggled.connect(self.set_log_freq)
        fit = icon_button("ph.arrows-out", "Görünümü sıfırla", object_name="FlatIcon", size=16)
        fit.clicked.connect(self._fit_mag)
        self.mag_legend = label("", "Readout")
        self.mag_readout = label("", "Readout")
        self.mag_card.add_left(self.mag_legend, optional=True)
        self.mag_card.add_right(self.mag_readout, optional=True)
        self.mag_card.add_right(self.quant)
        self.mag_card.add_right(self.log_button)
        self.mag_card.add_right(fit)
        self.mag_card.body.addWidget(self.mag_plot)
        self.quant_label = label("", "Readout")
        self.quant_label.setWordWrap(True)
        self.quant_label.hide()
        self.mag_card.body.addWidget(self.quant_label)
        self._mag_curve = self.mag_plot.plot(pen=pen(self._theme.series_color(0)))
        qpen = pen(self._theme.series_color(1), 1)  # tasarım eğrisini (2 px) örtmesin
        qpen.setStyle(Qt.PenStyle.DashLine)
        self._quant_curve = self.mag_plot.plot(pen=qpen)
        mag_vb = self.mag_plot.getPlotItem().getViewBox()
        mag_vb.sigRangeChangedManually.connect(lambda *_: setattr(self, "_mag_manual", True))
        self.mag_plot.scene().sigMouseClicked.connect(lambda ev: self._fit_mag() if ev.double() else None)
        Crosshair(self.mag_plot, self._theme, self._on_mag_hover)
        split.addWidget(self.mag_card)

        bottom = QSplitter(Qt.Orientation.Horizontal)
        bottom.setChildrenCollapsible(False)
        self.analysis_card = Card("Analiz")
        self.analysis_mode_combo = self._combo([(text, key) for key, text in ANALYSIS_MODES])
        self.analysis_mode_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.analysis_mode_combo.setMinimumContentsLength(8)  # dar kartta da seçili metin okunur
        self.analysis_mode_combo.setToolTip("Faz cevabı, grup gecikmesi, impuls ya da basamak cevabı")
        self.analysis_mode_combo.currentIndexChanged.connect(
            lambda *_: self.set_analysis_mode(self.analysis_mode_combo.currentData()))
        self.analysis_card.add_left(self.analysis_mode_combo)
        self.analysis_readout = label("", "Readout")
        self.analysis_card.add_right(self.analysis_readout, optional=True)
        self.analysis_plot = make_plot(self._theme, "Frekans", "Faz (°)", x_units="Hz")
        set_si_prefix(self.analysis_plot.getPlotItem().getAxis("left"), False)
        self.analysis_card.body.addWidget(self.analysis_plot)
        self._analysis_curve = self.analysis_plot.plot(pen=pen(self._theme.series_color(0)))
        Crosshair(self.analysis_plot, self._theme, self._on_analysis_hover)
        bottom.addWidget(self.analysis_card)

        self.pz_card = Card("Kutup-Sıfır")
        c0, c1 = self._theme.series_color(0), self._theme.series_color(1)
        self.pz_card.add_left(label(f'<span style="color:{c0}">✕</span>&nbsp;Kutup&nbsp;&nbsp;'
                                    f'<span style="color:{c1}">○</span>&nbsp;Sıfır', "Readout"), optional=True)
        self.pz_readout = label("", "Readout")
        self.pz_card.add_right(self.pz_readout, optional=True)
        self.pz_plot = make_plot(self._theme, "Reel", "Sanal")
        set_si_prefix(self.pz_plot.getPlotItem().getAxis("left"), False)
        set_si_prefix(self.pz_plot.getPlotItem().getAxis("bottom"), False)
        self.pz_plot.setAspectLocked(True)
        guide = pg.mkPen(t.axis, width=1)
        theta = np.linspace(0, 2 * np.pi, 361)
        self.pz_plot.plot(np.cos(theta), np.sin(theta), pen=guide)
        self.pz_plot.addItem(pg.InfiniteLine(0, angle=0, pen=guide))
        self.pz_plot.addItem(pg.InfiniteLine(0, angle=90, pen=guide))
        self._poles = pg.ScatterPlotItem(symbol="x", size=12, pen=pg.mkPen(c0, width=2), brush=None)
        self._zeros = pg.ScatterPlotItem(symbol="o", size=11, pen=pg.mkPen(c1, width=2), brush=pg.mkBrush(t.surface))
        self.pz_plot.addItem(self._zeros)
        self.pz_plot.addItem(self._poles)
        self.pz_card.body.addWidget(self.pz_plot)
        bottom.addWidget(self.pz_card)
        bottom.setStretchFactor(0, 3)
        bottom.setStretchFactor(1, 2)
        bottom.setSizes([560, 340])
        split.addWidget(bottom)
        split.setSizes([520, 400])

        # Seçilen noktanın değerleri ve görünen aralığın analizi
        self.mag_tips = DataTips(self.mag_plot, self._theme, self._mag_traces)
        self.analysis_tips = DataTips(self.analysis_plot, self._theme, self._analysis_traces)
        self.pz_tips = DataTips(self.pz_plot, self._theme, self._pz_traces)
        PlotAnalysis(self.mag_card, self.mag_plot, self._theme, self._mag_stats)
        PlotAnalysis(self.analysis_card, self.analysis_plot, self._theme, self._analysis_stats)
        PlotAnalysis(self.pz_card, self.pz_plot, self._theme, self._pz_stats)
        return split

    def set_quant(self, fmt: str | None) -> None:
        """Genlik grafiğinde kuantize cevabı göster (None: gizle)."""
        index = self.quant.findData(fmt) if fmt is not None else 0
        self.quant.setCurrentIndex(max(index, 0))

    def set_log_freq(self, on: bool) -> None:
        self.log_freq = on
        if self.log_button.isChecked() != on:
            self.log_button.setChecked(on)
        self._mag_manual = False
        self._draw()

    def set_analysis_mode(self, mode: str) -> None:
        self.analysis_mode = mode
        index = self.analysis_mode_combo.findData(mode)
        if self.analysis_mode_combo.currentIndex() != index:
            self.analysis_mode_combo.setCurrentIndex(index)
        self._draw_analysis()

    def _fit_mag(self) -> None:
        self._mag_manual = False
        self._draw()

    # ---------------------------------------------------------------- tasarım
    def redesign(self) -> None:
        self._timer.stop()
        spec = self.spec()
        try:
            self.design, self.error = design(spec), None
        except FilterError as exc:
            self.design, self.error = None, str(exc)
        self._designed_spec = spec
        d = self.design
        if d is not None and spec.auto_order and spec.family in ("iir", "fir"):
            shown = d.order // 2 if (spec.family == "iir" and spec.band in TWO_EDGE) else d.order
            self.order.blockSignals(True)
            self.order.setValue(shown)
            self.order.blockSignals(False)
        self._draw()
        self._update_info()
        self.design_changed.emit(d)

    def _update_info(self) -> None:
        t = self._theme.tokens
        d = self.design
        if d is None:
            self.info.setText(f'<span style="color:{t.critical}">●</span> <b>Tasarım hatası</b><br>{self.error}')
            self.summary_chip.setText("Geçersiz tasarım")
            self.apply_button.setEnabled(False)
            self.export_button.setEnabled(False)
            return
        self.apply_button.setEnabled(True)
        self.export_button.setEnabled(d.is_linear)
        lines = [f"<b>{d.describe()}</b>", d.structure_text()]
        if d.kind == "iir":
            color, text = (t.good, "Kararlı") if d.stable else (t.critical, "Kararsız")
            lines.append(f'<span style="color:{color}">●</span> {text} · en büyük |p| = {d.max_pole_radius:.6f}')
        elif d.kind == "fir":
            lines.append(f'<span style="color:{t.good}">●</span> Her zaman kararlı (FIR)')
        delay = d.delay_samples
        if delay is not None:
            lines.append(f"Gecikme: {delay:g} örnek ({delay / d.fs * 1e3:.4g} ms)")
        mask = mask_for(d.spec)
        if mask is not None and d.is_linear:
            ok, excess, stop_peak = check_mask(d, mask)
            if ok:
                lines.append(f'<span style="color:{t.good}">●</span> Maske karşılanıyor · durdurma ≤ '
                             f'{stop_peak:.1f} dB')
            else:
                problems = []
                if excess > 0.01:
                    problems.append(f"geçirme bandında {excess:.2f} dB aşım")
                if stop_peak > mask.stop_db + 0.01:
                    problems.append(f"durdurma {stop_peak:.1f} dB (hedef {mask.stop_db:.0f} dB)")
                lines.append(f'<span style="color:{t.warning}">●</span> Maske ihlali: ' + ", ".join(problems))
        for w in d.warnings:
            lines.append(f'<span style="color:{t.warning}">⚠</span> {w}')
        self.info.setText("<br>".join(lines))
        self.summary_chip.setText(d.describe())

    # ---------------------------------------------------------------- çizim
    def _clear_items(self, plot: pg.PlotWidget, items: list) -> None:
        for item in items:
            plot.removeItem(item)
        items.clear()

    def _clear_tips(self) -> None:
        for tips in (getattr(self, "mag_tips", None), getattr(self, "analysis_tips", None),
                     getattr(self, "pz_tips", None)):
            if tips is not None:
                tips.clear()

    def _draw(self) -> None:
        self._clear_tips()
        self._clear_items(self.mag_plot, self._mask_items)
        self._clear_items(self.mag_plot, self._marker_items)
        _set_log_x(self.mag_plot, self.log_freq)
        set_si_prefix(self.mag_plot.getPlotItem().getAxis("bottom"), not self.log_freq)
        d = self.design
        if d is None or not d.is_linear:
            self._mag_curve.setData([], [])
            self._quant_curve.setData([], [])
            self.quant_label.hide()
            self.mag_legend.setText("Medyan filtre doğrusal değildir: frekans cevabı yoktur" if d is not None
                                    else "Tasarım hatası — ayrıntı Tasarım panelinde")
            self._draw_analysis()
            self._draw_pz()
            return
        if self._response_of is not d:  # log f / sığdır / kuantizasyon gibi görünüm işlemleri cevabı yeniden hesaplamaz
            self._response, self._response_of = analyze(d), d
        r = self._response
        keep = r.freqs > 0 if self.log_freq else slice(None)
        f, mag = r.freqs[keep], r.magnitude_db[keep]
        self._mag_curve.setData(f, mag)
        self._draw_mask(d, f)
        self._draw_markers(d)
        self._draw_quant(d, r, keep)
        if not self._mag_manual:
            floor = -max(100.0, d.spec.atten_db + 40.0) if uses_atten(d.spec) else -100.0
            floor = max(floor, float(np.floor(mag.min() / 10) * 10) - 10)
            self.mag_plot.setYRange(floor, max(5.0, float(mag.max()) + 3.0), padding=0)
            lo = np.log10(f[0]) if self.log_freq else 0.0
            hi = np.log10(f[-1]) if self.log_freq else f[-1]
            self.mag_plot.setXRange(lo, hi, padding=0.01)
        self._draw_analysis()
        self._draw_pz()

    def _draw_mask(self, d: Design, f: np.ndarray) -> None:
        mask = mask_for(d.spec)
        if mask is None:
            self.mag_legend.setText(f'<span style="color:{self._theme.series_color(0)}">▬</span>&nbsp;Tasarım')
            return
        t = self._theme.tokens
        fill = QColor(t.critical)
        fill.setAlpha(56)  # yüzeyde seçilebilir (ΔE ~10), üstündeki eğriler okunur kalır (≥ 3.9:1)
        line = pg.mkPen(t.text_muted, width=1, style=Qt.PenStyle.DashLine)
        fmin = f[0] if len(f) else 0.0

        def x(v: float) -> float:
            return max(v, fmin) if self.log_freq else v

        for lo, hi in mask.passbands:
            below = pg.PlotDataItem([x(lo), x(hi)], [mask.pass_lo_db] * 2, pen=line, fillLevel=-400,
                                    brush=pg.mkBrush(fill))
            above = pg.PlotDataItem([x(lo), x(hi)], [mask.pass_hi_db] * 2, pen=line)
            self._mask_items += [below, above]
        for lo, hi in mask.stopbands:
            self._mask_items.append(pg.PlotDataItem([x(lo), x(hi)], [mask.stop_db] * 2, pen=line, fillLevel=400,
                                                    brush=pg.mkBrush(fill)))
        for item in self._mask_items:
            item.setZValue(-5)
            self.mag_plot.addItem(item)
        self.mag_legend.setText(f'<span style="color:{self._theme.series_color(0)}">▬</span>&nbsp;Tasarım&nbsp;&nbsp;'
                                f'<span style="color:{t.text_muted}">┅</span>&nbsp;Maske (taralı: yasak bölge)')

    def _draw_markers(self, d: Design) -> None:
        s = d.spec
        if mask_for(s) is not None:
            return
        if s.family == "special":
            m = s.method
            if m == "notch":
                freqs = [k * s.notch_f0 for k in range(1, s.harmonics + 1) if k * s.notch_f0 < s.nyquist]
            elif m in ("dc_block", "ema", "integrator"):
                freqs = [s.dc_cutoff]
            elif m == "allpass":
                freqs = [s.notch_f0]
            elif m == "differentiator":
                freqs = [s.f1, s.f1 + s.transition]
            elif m == "hilbert":
                freqs = [s.f1, s.nyquist - s.f1]
            else:
                freqs = []
        else:
            freqs = [s.f1, s.f2] if s.band in TWO_EDGE else [s.f1]
        line = pg.mkPen(self._theme.tokens.text_muted, width=1, style=Qt.PenStyle.DashLine)
        for fc in freqs:
            item = pg.InfiniteLine(np.log10(fc) if self.log_freq else fc, angle=90, pen=line)
            item.setZValue(-5)
            self.mag_plot.addItem(item)
            self._marker_items.append(item)

    def _draw_quant(self, d: Design, r, keep) -> None:
        fmt = self.quant.currentData()
        if fmt is None:
            self._quant_design = None
            self._quant_curve.setData([], [])
            self.quant_label.hide()
            return
        t = self._theme.tokens
        cache = self._quant_cache
        if cache is None or cache[0] is not d or cache[1] != fmt:  # görünüm işlemlerinde yeniden kuantize etme
            q = coeffs.quantize(d, fmt)
            qd = q.as_design(d)
            cache = self._quant_cache = (d, fmt, q, qd, analyze(qd, None if d.kind == "fir" else r.freqs),
                                         coeffs.quantization_report(d, q))
        _, _, q, self._quant_design, rq, rep = cache
        self._quant_curve.setData(rq.freqs[keep], rq.magnitude_db[keep])
        name = dict((k, v) for v, k in QUANT_CHOICES)[fmt]
        parts = [f'<span style="color:{self._theme.series_color(1)}">┅</span>&nbsp;<b>{name}</b>',
                 f"geçirme bandı sapması {rep.pass_dev_db:.4g} dB",
                 f"en büyük sapma {rep.dev_db:.4g} dB"]
        if rep.stop_peak_db is not None:
            parts.append(f"durdurma tepesi {rep.stop_peak_db:.1f} dB (ideal {rep.ideal_stop_peak_db:.1f})")
        if q.post_shift:
            parts.append(f"postShift {q.post_shift}")
        if q.scaled:
            parts.append("L∞ bölüm ölçekleme")
        if d.kind == "iir":
            color, text = (t.good, "kararlı") if rep.stable else (t.critical, "KARARSIZ")
            parts.append(f'<span style="color:{color}">{text}</span> (|p| = {rep.max_pole_radius:.6f})')
        if rep.clipped:
            parts.append(f'<span style="color:{t.warning}">{rep.clipped} katsayı kırpıldı</span>')
        self.quant_label.setText(" · ".join(parts))
        self.quant_label.show()

    def _draw_analysis(self) -> None:
        if getattr(self, "analysis_tips", None) is not None:
            self.analysis_tips.clear()
        plot, curve = self.analysis_plot, self._analysis_curve
        pi = plot.getPlotItem()
        d = self.design
        mode = self.analysis_mode
        frequency = mode in ("phase", "gd")
        _set_log_x(plot, self.log_freq and frequency)
        curve.setSymbol(None)
        if frequency:
            pi.setLabel("bottom", "Frekans", units="Hz")
            set_si_prefix(pi.getAxis("bottom"), not self.log_freq)
            pi.setLabel("left", "Faz (°)" if mode == "phase" else "Gecikme (örnek)")
        else:
            pi.setLabel("bottom", "Örnek (n)", units=None)
            set_si_prefix(pi.getAxis("bottom"), False)
            pi.setLabel("left", "Genlik")
        set_si_prefix(pi.getAxis("left"), False)
        if d is None or not d.is_linear:
            curve.setData([], [])
            self.analysis_readout.setText("")
            return
        if frequency:
            r = self._response
            keep = r.freqs > 0 if self.log_freq else slice(None)
            y = r.phase_deg if mode == "phase" else r.group_delay
            curve.setData(r.freqs[keep], y[keep], connect="finite")
        else:
            y = impulse_response(d) if mode == "impulse" else step_response(d)
            self._time_response = y
            n = np.arange(len(y), dtype=np.float64)
            if len(y) <= STEM_LIMIT:  # gövde grafiği: her örnek için 0'dan değere dikey çizgi + nokta
                xs = np.repeat(n, 2)
                ys = np.column_stack((np.zeros_like(y), y)).ravel()
                curve.setData(xs, ys, connect="pairs")
                curve.setSymbol("o")
                curve.setSymbolSize(5)
                curve.setSymbolPen(None)
                curve.setSymbolBrush(pg.mkBrush(self._theme.series_color(0)))
            else:
                curve.setData(n, y, connect="all")
        pi.enableAutoRange()

    def _draw_pz(self) -> None:
        self._clear_items(self.pz_plot, self._pz_items)
        d = self.design
        if d is None or not d.is_linear:
            self._poles.setData([], [])
            self._zeros.setData([], [])
            self.pz_readout.setText("")
            return
        zeros, poles, _ = d.zpk
        self._poles.setData(poles.real, poles.imag)
        if zeros is None:
            self._zeros.setData([], [])
            self.pz_readout.setText(f"{len(d.b)} katsayı: sıfırlar gösterilmiyor")
            zeros = np.zeros(0)
        else:
            self._zeros.setData(zeros.real, zeros.imag)
            self.pz_readout.setText(f"{len(poles)} kutup · {len(zeros)} sıfır")
        color = self._theme.tokens.text_secondary
        for points in (poles, zeros):
            for p, count in _multiplicities(points):
                if count > 1:
                    text = pg.TextItem(f"{count}", color=color, anchor=(-0.3, 1.1))
                    text.setPos(p.real, p.imag)
                    self.pz_plot.addItem(text)
                    self._pz_items.append(text)
        extent = max([1.15] + [abs(v) * 1.1 for v in np.concatenate((poles, zeros))])
        extent = min(extent, 4.0)
        self.pz_plot.setRange(xRange=(-extent, extent), yRange=(-extent, extent), padding=0)

    # ---------------------------------------------------------------- veri ipuçları ve analiz
    def _response_at_index(self, f: float):
        r = self._response
        if r is None:
            return None
        return int(np.clip(np.searchsorted(r.freqs, f), 0, len(r.freqs) - 1))

    def _mag_traces(self) -> list[Trace]:
        d = self.design
        if d is None or not d.is_linear:
            return []

        def describe(f, db):
            lines = [f"f = {fmt(f, 'Hz', 7)}", f"{db:.3f} dB  (|H| = {10 ** (db / 20):.6g})"]
            r = self._exact(f)
            if r is not None:
                lines.append(f"faz {r.phase_deg[0]:.2f}°")
                gd = r.group_delay[0]
                if np.isfinite(gd):
                    lines.append(f"grup gecikmesi {gd:.3f} örnek ({gd / d.fs * 1e3:.4g} ms)")
            return lines

        def exact(f, quantized=False):
            r = self._exact(f, quantized)
            return None if r is None else (f, float(r.magnitude_db[0]))

        traces = [Trace(self._mag_curve, "Tasarım", self._theme.series_color(0), describe, exact)]
        if self.quant.currentData() is not None:
            name = self.quant.currentText()
            traces.append(Trace(self._quant_curve, f"Kuantize ({name})", self._theme.series_color(1),
                                lambda f, db: [f"f = {fmt(f, 'Hz', 7)}", f"{db:.3f} dB"],
                                lambda f: exact(f, True)))
        return traces

    def _analysis_traces(self) -> list[Trace]:
        d = self.design
        if d is None or not d.is_linear:
            return []
        mode = self.analysis_mode

        def describe(x, y):
            if mode == "phase":
                return [f"f = {fmt(x, 'Hz', 7)}", f"faz {y:.3f}°"]
            if mode == "gd":
                return [f"f = {fmt(x, 'Hz', 7)}", f"{y:.4f} örnek", f"{y / d.fs * 1e3:.5g} ms"]
            n = int(round(x))
            return [f"n = {n}  (t = {fmt(n / d.fs, 's')})", f"değer = {y:+.6g}"]

        def snap_to_sample(x):  # gövde grafiğinde noktalar çift; değer doğrudan cevaptan okunur
            y = self._time_response
            n = int(np.clip(round(x), 0, len(y) - 1))
            return float(n), float(y[n])

        def at_frequency(f):
            r = self._exact(f)
            if r is None:
                return None
            value = float(r.phase_deg[0] if mode == "phase" else r.group_delay[0])
            return (f, value) if np.isfinite(value) else None

        name = dict(ANALYSIS_MODES)[mode]
        exact = snap_to_sample if mode in ("impulse", "step") else at_frequency
        return [Trace(self._analysis_curve, name, self._theme.series_color(0), describe, exact)]

    def _pz_traces(self) -> list[Trace]:
        d = self.design
        if d is None or not d.is_linear:
            return []
        fs = d.fs

        def describe(x, y):
            z = complex(x, y)
            r, ang = abs(z), np.angle(z)
            lines = [f"z = {x:+.6f} {'+' if y >= 0 else '−'} j{abs(y):.6f}", f"|z| = {r:.6f}  ∠ {np.degrees(ang):.3f}°",
                     f"f = {fmt(abs(ang) / (2 * np.pi) * fs, 'Hz', 6)}"]
            if 0 < r < 1:
                lines.append(f"τ = {fmt(-1 / (fs * np.log(r)), 's')}")
            return lines

        return [Trace(self._poles, "Kutup", self._theme.series_color(0), describe),
                Trace(self._zeros, "Sıfır", self._theme.series_color(1), describe)]

    def _mag_stats(self, x0: float, x1: float) -> list:
        d = self.design
        if d is None or not d.is_linear or self._response is None:
            return []
        r = self._response
        sl = visible_slice(r.freqs, x0, x1)
        f, m = r.freqs[sl], r.magnitude_db[sl]
        items = curve_stats(f, m, "Hz", "dB")
        top = r.magnitude_db.max()
        level = top - 3.0103
        above = r.magnitude_db > level
        cross = np.flatnonzero(np.diff(above.astype(np.int8)))  # her −3 dB geçişi (doğrusal ara değerlemeyle)
        edges = []
        for i in cross[:8]:
            f0, f1, m0, m1 = r.freqs[i], r.freqs[i + 1], r.magnitude_db[i], r.magnitude_db[i + 1]
            edges.append(f0 + (level - m0) * (f1 - f0) / (m1 - m0) if m1 != m0 else f0)
        if edges:
            label_ = "−3 dB noktası" if len(edges) == 1 else "−3 dB noktaları"
            items.append((label_, ", ".join(fmt(float(e), "Hz", 6) for e in edges)
                          + (" …" if len(cross) > 8 else "")))
        items.append(("Geçirme bandı tepesi", f"{top:.4f} dB"))
        columns = [("Tasarım", self._theme.series_color(0), items)]
        qx, qy = self._quant_curve.xData, self._quant_curve.yData
        if self.quant.currentData() is not None and qx is not None and len(qx):
            qs = visible_slice(np.asarray(qx), x0, x1)
            columns.append((f"Kuantize ({self.quant.currentText()})", self._theme.series_color(1),
                            curve_stats(qx[qs], qy[qs], "Hz", "dB")))
        return columns

    def _analysis_stats(self, x0: float, x1: float) -> list:
        x, y = self._analysis_curve.xData, self._analysis_curve.yData
        if x is None or not len(x):
            return []
        mode = self.analysis_mode
        if mode in ("impulse", "step"):
            y = self._time_response
            x = np.arange(len(y), dtype=np.float64)
        sl = visible_slice(np.asarray(x), x0, x1)
        unit = {"phase": "°", "gd": "örnek"}.get(mode, "")
        items = curve_stats(x[sl], y[sl], "Hz" if mode in ("phase", "gd") else "", unit)
        if mode == "step" and len(y) > 2:
            final = float(y[-1])
            peak = float(np.max(np.abs(y)))
            if abs(final) <= 1e-3 * peak:  # sıfır DC kazancı: aşma / yükselme / oturma tanımsız
                items.append(("Son değer", f"{final:.3g} (≈ 0: DC kazancı sıfır)"))
                items += [("Aşma", "—"), ("Yükselme (%10–90)", "—"), ("Oturma (±%2)", "—")]
            else:
                over = (float(np.max(y)) - final) / abs(final) * 100
                rise = np.flatnonzero(y >= 0.9 * final)
                start = np.flatnonzero(y >= 0.1 * final)
                items.append(("Son değer", f"{final:.6g}"))
                items.append(("Aşma", f"%{max(over, 0):.3g}"))
                if len(rise) and len(start):
                    n = int(rise[0] - start[0])
                    items.append(("Yükselme (%10–90)", f"{n} örnek ({fmt(n / self.design.fs, 's')})"))
                settle = np.flatnonzero(np.abs(y - final) > 0.02 * abs(final))
                if len(settle):
                    n = int(settle[-1] + 1)
                    items.append(("Oturma (±%2)", f"{n} örnek ({fmt(n / self.design.fs, 's')})"))
        if mode == "impulse":
            items.append(("Enerji Σh²", f"{float(np.sum(self._time_response ** 2)):.6g}"))
        return [(dict(ANALYSIS_MODES)[mode], self._theme.series_color(0), items)]

    def _pz_stats(self, _x0: float, _x1: float) -> list:
        d = self.design
        if d is None or not d.is_linear:
            return []
        zeros, poles, _ = d.zpk
        fs = d.fs
        items = [("Kutup", str(len(poles))), ("Sıfır", "gösterilmiyor" if zeros is None else str(len(zeros)))]
        moving = poles[np.abs(poles) > 1e-12]
        if len(moving):
            p = moving[np.argmax(np.abs(moving))]
            r = abs(p)
            items += [("En büyük |p|", f"{r:.6f}"), ("Kararlılık payı 1 − |p|", f"{1 - r:.3g}"),
                      ("Baskın kutup f", fmt(abs(np.angle(p)) / (2 * np.pi) * fs, "Hz", 6))]
            if 0 < r < 1:
                s_plane = np.log(p) * fs  # z = e^{sT}
                wn = abs(s_plane)
                zeta = -s_plane.real / wn if wn > 0 else 1.0
                items += [("Baskın kutup ζ", f"{zeta:.4g}"),
                          ("Baskın kutup Q", f"{1 / (2 * zeta):.4g}" if zeta > 0 else "∞"),
                          ("Zaman sabiti τ", fmt(-1 / (fs * np.log(r)), "s"))]
        else:
            items.append(("Kararlılık", "FIR — kutuplar orijinde"))
        return [("Kutup-sıfır", self._theme.series_color(0), items)]

    # ---------------------------------------------------------------- imleç okuması
    def _exact(self, f: float, quantized: bool = False):
        """f'de tam cevap (ızgara / ekran seyreltmesinden bağımsız): Response (tek nokta) ya da None."""
        d = self._quant_design if quantized else self.design
        if d is None or not d.is_linear or not 0 <= f <= d.fs / 2:
            return None
        r = analyze(d, np.array([float(f)]))
        if self._response is not None and len(self._response.freqs):  # faz dalını çizilen sarılmamış fazla eşle
            near = float(np.interp(f, self._response.freqs, self._response.phase_deg))
            r.phase_deg[0] += 360.0 * round((near - r.phase_deg[0]) / 360.0)
        return r

    def _on_mag_hover(self, x: float | None, _y: float | None) -> None:
        if x is None or self.design is None or not self.design.is_linear:
            self.mag_readout.setText("")
            return
        f = 10 ** x if self.log_freq else x  # log eksende görünüm koordinatı log10(f)
        r = self._exact(f)
        if r is None:
            self.mag_readout.setText("")
            return
        text = f"f = {format_hz(f)}   {r.magnitude_db[0]:.2f} dB"
        if self.quant.currentData() is not None:
            rq = self._exact(f, quantized=True)
            if rq is not None:
                text += f"   (kuantize {rq.magnitude_db[0]:.2f} dB)"
        self.mag_readout.setText(text)

    def _on_analysis_hover(self, x: float | None, _y: float | None) -> None:
        if x is None or self.design is None or not self.design.is_linear:
            self.analysis_readout.setText("")
            return
        xs, ys = self._analysis_curve.getData()
        if xs is None or not len(xs):
            return
        mode = self.analysis_mode
        if mode in ("phase", "gd"):
            f = 10 ** x if self.log_freq else x
            r = self._exact(f)
            if r is None:
                self.analysis_readout.setText("")
                return
            if mode == "phase":
                self.analysis_readout.setText(f"f = {format_hz(f)}   {r.phase_deg[0]:.2f}°")
            else:
                value = r.group_delay[0]
                ms = value / self.design.fs * 1e3 if np.isfinite(value) else float("nan")
                self.analysis_readout.setText(f"f = {format_hz(f)}   {value:.2f} örnek ({ms:.4g} ms)")
        else:
            n = int(round(x))
            y = self._time_response
            if 0 <= n < len(y):
                self.analysis_readout.setText(f"n = {n} ({n / self.design.fs * 1e3:.4g} ms)   {y[n]:+.5g}")
