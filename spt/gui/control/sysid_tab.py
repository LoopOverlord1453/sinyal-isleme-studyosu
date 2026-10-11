"""Kontrol Tasarımı → "Sistem Tanımlama" sekmesi (MATLAB System Identification uygulamasının karşılığı).

İş akışı (soldan sağa): veri kaynağı (canlı akış / kayıt / Çalışma Alanı anlık görüntüsü — `set_sources`; CSV /
TXT / MAT dosyası) → giriş ve çıkış kanalları → Ts (akıştan / zaman sütunundan otomatik ya da elle) → zaman aralığı
(veri grafiğinde sürüklenen bölge) → ön işleme (eksikleri doldur, trend, süzgeç, yeniden örnekleme, tahmin /
doğrulama bölmesi) → yöntem ve dereceler → "Tahmin et" (ayrı süreçte; iptal edilebilir, arayüz donmaz).

Sonuçlar: veri (u ve y ayrı kartlarda, doğrulama bölgesi gölgeli), karşılaştırma (doğrulama verisinde ölçülen +
model benzetimleri, uyum % lejantta), artık analizi (özilinti + girişle çapraz ilinti, güven sınırları), frekans
cevabı (modeller + çalışma verisinden spa) ve model tablosu (yöntem, derece, uyum % tahmin / doğrulama, FPE, AIC)
→ "Modellere ekle" (Modeller sekmesindeki çalışma alanına).

Tüm hesaplar `spt.control.sysid` API'sindendir (bkz. `sysid_jobs`). Uyum %: `compare` (benzetim, k = ∞,
başlangıç durumu tahmin edilir — MATLAB compare varsayılanı); FPE / AIC: modelin tahmin raporu (MATLAB
`Report.Fit` tanımları).

Renkler ölçülerek seçildi (OKLab ΔE, #121110 yüzeyinde; protan / deutan benzetimi Machado 2009, şiddet 1.0):
8 model rengi + 7 kanal rengi + spa rengi — 16 rengin her çifti normal görüşte ΔE ≥ 10.0, renk körlüğünde ≥ 6.4,
yüzeye kontrast ≥ 4.9:1. Aynı sekmede hiçbir renk iki seride kullanılmaz: kanal rengi kanala, model rengi modele
bağlıdır (model silinince renk serbest kalır); bu yüzden en çok 8 model ve 7 kanal (giriş + çıkış) tutulur.
"""
from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass, field

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRegularExpression, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QRegularExpressionValidator
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from spt.control.errors import ControlError
from spt.gui.control import sysid_jobs as J
from spt.gui.control.sysid_runner import JobRunner
from spt.gui.icons import line_swatch
from spt.gui.theme import ThemeManager
from spt.gui.widgets.common import Card, CollapsibleSection, combo, form_layout, hbox, label
from spt.gui.widgets.inspector import DataTips, Trace, fmt
from spt.gui.widgets.plot import make_plot, pen

log = logging.getLogger("spt.gui")

# Ölçülmüş palet (bkz. modül belgesi)
MODEL_COLORS = ("#f54748", "#488bfb", "#dfa11a", "#45f27f", "#e03fc7", "#4cd1ee", "#c396a8", "#8fa1e4")
CHANNEL_COLORS = ("#f5f5f5", "#ced51f", "#9becc1", "#bebebe", "#baff3f", "#94c080", "#219a2c")
SPA_COLOR = "#5c8c79"
MEASURED_WIDTH = 2.0
MODEL_WIDTH = 1.6
PLOT_MIN_HEIGHT = 100
MAX_LAG = 25

INT_LIST = QRegularExpression(r"^\s*\d+([\s,;]+\d+)*\s*$")

# yöntem → [(anahtar, etiket, tür, varsayılan, ipucu)]; tür: int | vec | float | bool | choice
_NA = ("na", "na (A derecesi)", "int", 2, "Çıkış geçmişi sayısı: A(q) polinomunun derecesi")
_NB = ("nb", "nb (B derecesi)", "vec", "2", "Giriş geçmişi sayısı. Tek değer tüm girişlere uygulanır; giriş başına "
       "değer için boşlukla ayırın (ör. 2 3)")
_NC = ("nc", "nc (C derecesi)", "int", 2, "Gürültü modelinin pay derecesi")
_ND = ("nd", "nd (D derecesi)", "int", 2, "Gürültü modelinin payda derecesi")
_NF = ("nf", "nf (F derecesi)", "vec", "2", "Ölçüm modelinin payda derecesi (giriş başına ya da tek değer)")
_NK = ("nk", "nk (giriş gecikmesi)", "vec", "1", "Girişin çıkışı etkilemesinden önceki örnek sayısı (giriş başına "
       "ya da tek değer); doğrudan geçişte 0")
METHOD_FIELDS: dict[str, list[tuple]] = {
    "arx": [_NA, _NB, _NK],
    "armax": [_NA, _NB, _NC, _NK],
    "oe": [_NB, _NF, _NK],
    "bj": [_NB, _NC, _ND, _NF, _NK],
    "tfest": [("np", "Kutup sayısı", "vec", "2", "Transfer fonksiyonunun kutup sayısı (giriş başına ya da tek)"),
              ("nz", "Sıfır sayısı", "vec", "1", "Sıfır sayısı (kutup sayısını aşamaz)"),
              ("delay", "Giriş gecikmesi (s)", "float", 0.0, "Bilinen giriş–çıkış ölü zamanı (0: yok)"),
              ("estimate_delay", "Gecikmeyi tahmin et", "bool", False, "Ölü zaman da parametre olarak tahmin "
               "edilir (yukarıdaki değer yok sayılır)"),
              ("discrete", "Ayrık zamanlı model", "bool", False, "İşaretliyse z düzleminde (verinin Ts'iyle) tahmin")],
    "procest": [],  # özel form: kutup sayısı + D / Z / U / I
    "n4sid": [("nx", "Durum sayısı", "int", 2, "Modelin derecesi (durum sayısı)")],
    "ssest": [("nx", "Durum sayısı", "int", 2, "Modelin derecesi (durum sayısı)"),
              ("discrete", "Ayrık zamanlı model", "bool", False, "İşaretsizse sürekli zamanlı model (MATLAB "
               "varsayılanı); işaretliyse verinin Ts'iyle ayrık")],
    "impulseest": [("n", "Darbe cevabı uzunluğu", "int", 0, "FIR katsayı sayısı; 0: otomatik (MATLAB kuralı, "
                    "en çok 70)"),
                   ("nk", "İlk gecikme (nk)", "int", 0, "Dürtü cevabının başladığı örnek")],
    "spa": [("M", "Pencere uzunluğu", "int", 0, "Hann gecikme penceresi uzunluğu; 0: otomatik (MATLAB "
             "varsayılanı min(N/10, 30))")],
    "etfe": [("M", "Düzleştirme penceresi", "int", 0, "Frekans düzleştirme penceresi; 0: düzleştirme yok")],
    "nlarx": [_NA, _NB, _NK,
              ("nonlinearity", "Doğrusal olmayan kısım", "choice",
               [("wavenet", "Dalgacık ağı"), ("sigmoidnet", "Sigmoid ağı"), ("linear", "Doğrusal")],
               "Regresörlerden çıkışa eşleme (MATLAB varsayılanı dalgacık ağı)"),
              ("units", "Birim sayısı", "int", 0, "Ağdaki birim sayısı; 0: otomatik (MATLAB varsayılanı)")],
    "nlhw": [_NB, _NF, _NK,
             ("input_nl", "Giriş doğrusalsızlığı", "choice",
              [("pwlinear", "Parçalı doğrusal"), ("saturation", "Doyum"), ("deadzone", "Ölü bölge"),
               ("poly1d", "Polinom"), ("unitgain", "Yok (birim kazanç)")], "Doğrusal bloktan önceki statik eşleme"),
             ("output_nl", "Çıkış doğrusalsızlığı", "choice",
              [("pwlinear", "Parçalı doğrusal"), ("saturation", "Doyum"), ("deadzone", "Ölü bölge"),
               ("poly1d", "Polinom"), ("unitgain", "Yok (birim kazanç)")], "Doğrusal bloktan sonraki statik eşleme")],
}
INPUT_HINT = re.compile(r"(^u\d*$|^u_|giri|input|cmd|komut|pwm|motor|throttle|gaz|^ref|setpoint|uyar)", re.I)
OUTPUT_HINT = re.compile(r"(^y\d*$|^y_|çıkı|cikis|output|meas|ölç|olc|açı|aci|angle|rate|hız|hiz|pos|konum)", re.I)


class _Legend(QLabel):
    """Seri lejantı: renk çizgisi + ad (zengin metin). Başlık satırında tek satır, grafiğin altında kaydırmalı."""

    def __init__(self, card: Card, wrap: bool = False):
        super().__init__()
        self.setObjectName("Readout")
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setWordWrap(wrap)
        self._card = card
        self._items: list[tuple[str, str]] = []

    def set_items(self, items) -> None:
        self._items = list(items)
        self.setText("&nbsp;&nbsp;&nbsp; ".join(f'<span style="color:{c}">▬</span>&nbsp;{html.escape(t)}'
                                                for c, t in self._items))
        fit = getattr(self._card, "_fit_header", None)
        if not self.wordWrap() and callable(fit):
            fit()  # başlıkta: yeni genişliğe göre sığmayan isteğe bağlı öğeleri yeniden seç

    def add(self, color: str, text: str) -> None:
        self.set_items(self._items + [(color, text)])

    def texts(self) -> list[str]:
        return [t for _, t in self._items]


@dataclass
class ModelEntry:
    """Tablodaki bir tanımlanmış model."""
    name: str
    method: str
    opts: dict
    model: object
    color: str
    visible: bool = True
    ev: J.Evaluation | None = None
    data_text: str = ""          # tahmin edildiği veri (ör. "Akış · imu, 2 giriş / 1 çıkış, Ts 0.01 s")

    @property
    def addable(self) -> bool:
        return self.method not in J.NONPARAMETRIC and self.method not in J.NONLINEAR


@dataclass
class _Context:
    """Son değerlendirmenin verisi (karşılaştırma ve artık grafikleri bunu kullanır)."""
    work: object
    est: object
    val: object
    spa: tuple | None = None
    spa_error: str = ""
    w: np.ndarray | None = None
    names: dict = field(default_factory=dict)   # {"u": [...], "y": [...]}


def _fmt_num(v) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:.4g}"


def _fmt_fit(a) -> str:
    if a is None:
        return "—"
    a = np.atleast_1d(a)
    return " / ".join("—" if not np.isfinite(x) else f"{x:.2f}" for x in a)


class SysIdTab(QWidget):
    """Sistem Tanımlama sekmesi. `workspace`: Modeller sekmesinin çalışma alanı (yoksa yalnız `models_ready`)."""

    models_ready = pyqtSignal(list)      # [(ad, LTI modeli, tanımlanmış model)] — "Modellere ekle"
    busy_changed = pyqtSignal(bool)

    def __init__(self, theme: ThemeManager, workspace=None, parent: QWidget | None = None):
        super().__init__(parent)
        self._theme = theme
        self.workspace = workspace
        self._sources: list[tuple[str, str, object]] = []
        self._files: dict[str, J.SourceTable] = {}      # açılan dosyalar (kimlik → tablo)
        self.table: J.SourceTable | None = None
        self.entries: list[ModelEntry] = []
        self.context: _Context | None = None
        self._pending: dict | None = None               # süren işin ek bilgisi
        self._chan_colors: dict[tuple[str, int], str] = {}
        self._tips: list[DataTips] = []
        self._data_curves: list[Trace] = []
        self._cmp_traces: list[Trace] = []
        self._res_traces: dict[str, list[Trace]] = {"auto": [], "cross": []}
        self._freq_traces: dict[str, list[Trace]] = {"mag": [], "phase": []}
        self._region_lock = False
        self._warmed = False
        self.status_text = ""            # son durum iletisi (düz metin) ve türü
        self.status_state = ""

        self.runner = JobRunner(self)
        self.runner.progress.connect(lambda t: self._set_status(t, "busy"))
        self.runner.finished.connect(self._on_finished)
        self.runner.failed.connect(self._on_failed)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        lay.addWidget(split, 1)
        split.addWidget(self._build_setup())
        split.addWidget(self._build_results())
        split.setStretchFactor(1, 1)
        split.setSizes([350, 1000])
        self._split = split

        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(60)
        self._redraw_timer.timeout.connect(self._draw_data)
        self._set_method("arx")
        self._refresh_sources()
        self._update_buttons()

    # ================================================================ kurulum paneli
    def _build_setup(self) -> Card:
        card = Card("Veri ve Yöntem")
        card.setMinimumWidth(300)
        card.setMaximumWidth(440)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        host = QWidget()
        col = QVBoxLayout(host)
        col.setContentsMargins(0, 0, 6, 0)
        col.setSpacing(6)
        scroll.setWidget(host)
        card.body.addWidget(scroll, 1)

        # --- kaynak
        src = CollapsibleSection("Veri kaynağı")
        self.source = combo([])
        self.source.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.source.setMinimumContentsLength(10)
        self.source.currentIndexChanged.connect(self._on_source)
        src.add_row("Kaynak", self.source)
        self.open_button = QPushButton("Dosya aç…")
        self.open_button.setToolTip("CSV / TXT (sütunlar kanal; başlık ve zaman sütunu isteğe bağlı) ya da MATLAB "
                                    ".mat (v5 / v7) dosyası")
        self.open_button.clicked.connect(self._open_dialog)
        self.reload_button = QPushButton("Yenile")
        self.reload_button.setToolTip("Canlı akışın tamponundaki en güncel veriyi yeniden al")
        self.reload_button.clicked.connect(self._reload)
        src.form.addRow(hbox(self.open_button, self.reload_button, spacing=8))
        self.source_info = label("", "Muted")
        self.source_info.setWordWrap(True)
        src.form.addRow(self.source_info)
        col.addWidget(src)

        # --- kanallar
        ch = CollapsibleSection("Kanallar")
        self.inputs_list = self._channel_list("Giriş kanalları (u): sistemi süren sinyaller")
        self.outputs_list = self._channel_list("Çıkış kanalları (y): ölçülen cevaplar")
        ch.add_row("Girişler (u)", self.inputs_list)
        ch.add_row("Çıkışlar (y)", self.outputs_list)
        col.addWidget(ch)

        # --- Ts ve aralık
        tr = CollapsibleSection("Örnekleme ve aralık")
        self.ts_edit = QLineEdit()
        self.ts_edit.setPlaceholderText("ör. 0.01")
        self.ts_edit.setToolTip("Örnekleme aralığı (saniye)")
        self.ts_edit.editingFinished.connect(self._on_ts_changed)
        self.ts_auto = QCheckBox("Otomatik")
        self.ts_auto.setToolTip("Akışın örnekleme hızından ya da dosyanın zaman sütunundan / Ts değişkeninden")
        self.ts_auto.toggled.connect(self._on_ts_auto)
        tr.add_row("Ts (s)", hbox(self.ts_edit, self.ts_auto, spacing=8))
        self.t0_spin = self._time_spin()
        self.t1_spin = self._time_spin()
        self.t0_spin.valueChanged.connect(self._on_range_spin)
        self.t1_spin.valueChanged.connect(self._on_range_spin)
        tr.add_row("Başlangıç", self.t0_spin)
        tr.add_row("Bitiş", self.t1_spin)
        self.full_button = QPushButton("Tüm veri")
        self.full_button.setToolTip("Zaman aralığını verinin tamamına ayarla (aralık veri grafiğinde sürüklenebilir)")
        self.full_button.clicked.connect(self._full_range)
        tr.form.addRow(self.full_button)
        col.addWidget(tr)

        # --- ön işleme
        pp = CollapsibleSection("Ön işleme")
        self.missing_combo = combo(["Hata ver", "Doğrusal ara değer", "Modelle (ARX)"])
        self.missing_combo.setToolTip("Kayıp (NaN) örnekler: tahmin eksiksiz veri ister. Eksikleri doldur: doğrusal "
                                      "ara değer ya da ARX modeliyle (misdata benzeri)")
        pp.add_row("Eksikleri doldur", self.missing_combo)
        self.detrend_combo = combo(["Yok", "Ortalama (sabit)", "Doğrusal"], 1)
        self.detrend_combo.setToolTip("Çalışma noktasını çıkar (MATLAB detrend). Doğrusal modeller sapmasız veri ister")
        pp.add_row("Trend çıkar", self.detrend_combo)
        self.filter_combo = combo(["Yok"] + list(J.FILTER_KINDS.values()))
        self.filter_combo.setToolTip("Butterworth süzgeç (MATLAB idfilt): tüm giriş ve çıkış kanalları aynı süzgeçten")
        self.filter_combo.currentIndexChanged.connect(self._on_filter_kind)
        pp.add_row("Süzgeç", self.filter_combo)
        self.filter_order = QSpinBox()
        self.filter_order.setRange(1, 10)
        self.filter_order.setValue(4)
        pp.add_row("Süzgeç derecesi", self.filter_order)
        self.filter_f1 = self._freq_spin()
        self.filter_f2 = self._freq_spin()
        pp.add_row("Kesim / alt kenar", self.filter_f1)
        pp.add_row("Üst kenar", self.filter_f2)
        self.zero_phase = QCheckBox("Sıfır fazlı (ileri-geri)")
        self.zero_phase.setToolTip("Faz kaymasız süzme (filtfilt). İşaretsizse nedensel süzme")
        pp.form.addRow(self.zero_phase)
        self.res_p = QSpinBox()
        self.res_q = QSpinBox()
        for s in (self.res_p, self.res_q):
            s.setRange(1, 1000)
            s.setValue(1)
        self.res_p.setToolTip("Yeni örnekleme hızı = hız · P / Q (MATLAB resample, kenar süzgeçli)")
        self.res_q.setToolTip("Seyreltme için Q > P (ör. P=1, Q=4: hız dörtte bir)")
        pp.add_row("Yeniden örnekle P / Q", hbox(self.res_p, label("/", "Muted"), self.res_q, spacing=6))
        self.fraction_spin = QSpinBox()
        self.fraction_spin.setRange(10, 100)
        self.fraction_spin.setValue(50)
        self.fraction_spin.setSuffix(" %")
        self.fraction_spin.setToolTip("Aralığın ilk bölümü tahmin, kalanı doğrulama verisi. %100: doğrulama tahmin "
                                      "verisiyle yapılır")
        self.fraction_spin.valueChanged.connect(lambda *_: self._update_split_region())
        pp.add_row("Tahmin oranı", self.fraction_spin)
        col.addWidget(pp)
        self._on_filter_kind()

        # --- yöntem
        me = CollapsibleSection("Yöntem ve dereceler")
        self.method_combo = combo([])
        for key, (text, _) in J.METHODS.items():
            self.method_combo.addItem(text, key)
        self.method_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.method_combo.setMinimumContentsLength(12)
        self.method_combo.currentIndexChanged.connect(lambda *_: self._set_method(self.method_combo.currentData()))
        me.add_row("Yöntem", self.method_combo)
        self.method_stack = QStackedWidget()
        self._fields: dict[str, dict[str, QWidget]] = {}
        self._pages: dict[str, int] = {}
        for key in J.METHODS:
            page, widgets = self._method_page(key)
            self._pages[key] = self.method_stack.addWidget(page)
            self._fields[key] = widgets
        me.form.addRow(self.method_stack)
        self.method_note = label("", "Muted")
        self.method_note.setWordWrap(True)
        me.form.addRow(self.method_note)
        self.suggest_button = QPushButton("Derece öner")
        self.suggest_button.setToolTip("ARX dereceleri na, nb = 1…10, nk = 1…5 taranır; doğrulama verisinde en iyi "
                                       "(AIC) derece seçilir (MATLAB arxstruc / selstruc)")
        self.suggest_button.clicked.connect(self.suggest_orders)
        me.form.addRow(self.suggest_button)
        col.addWidget(me)
        col.addStretch(1)

        self.status = label("", "Readout")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.RichText)
        card.body.addWidget(self.status)
        self.estimate_button = QPushButton("Tahmin et")
        self.estimate_button.setObjectName("Primary")
        self.estimate_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.estimate_button.setToolTip("Seçili yöntemle modeli arka planda tahmin et ve tüm modelleri doğrula")
        self.estimate_button.clicked.connect(self.estimate)
        self.cancel_button = QPushButton("İptal")
        self.cancel_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cancel_button.clicked.connect(self.cancel)
        card.body.addWidget(hbox(self.estimate_button, self.cancel_button, spacing=8))
        return card

    def _channel_list(self, tip: str) -> QListWidget:
        w = QListWidget()
        w.setToolTip(tip)
        w.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        w.setFixedHeight(40)
        w.itemChanged.connect(self._on_channel_item)
        return w

    @staticmethod
    def _fit_list(lst: QListWidget) -> None:
        """Liste yüksekliği kanal sayısına göre (en çok 6 satır; fazlası kaydırılır)."""
        rows = max(min(lst.count(), 6), 1)
        row = lst.sizeHintForRow(0) if lst.count() else 28
        lst.setFixedHeight(rows * max(row, 24) + 2 * lst.frameWidth() + 4)

    @staticmethod
    def _time_spin() -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setDecimals(4)
        s.setRange(0.0, 0.0)
        s.setSuffix(" s")
        s.setKeyboardTracking(False)
        return s

    @staticmethod
    def _freq_spin() -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setDecimals(4)
        s.setRange(0.0, 1e9)
        s.setSuffix(" Hz")
        s.setKeyboardTracking(False)
        return s

    def _method_page(self, key: str) -> tuple[QWidget, dict]:
        page = QWidget()
        form = form_layout()
        form.setContentsMargins(0, 0, 0, 0)
        page.setLayout(form)
        widgets: dict[str, QWidget] = {}
        if key == "procest":
            poles = combo(["0", "1", "2", "3"], 1)
            poles.setToolTip("Kutup sayısı (P0…P3)")
            flags = {}
            for flag, text, tip in (("D", "Gecikme (D)", "Ölü zaman e^(−Td·s)"),
                                    ("Z", "Sıfır (Z)", "Payda (1 + Tz·s)"),
                                    ("U", "Salınımlı kutuplar (U)", "Karmaşık eşlenik kutup çifti (en az 2 kutup)"),
                                    ("I", "İntegratör (I)", "Ek 1/s")):
                cb = QCheckBox(text)
                cb.setToolTip(tip)
                flags[flag] = cb
            flags["D"].setChecked(True)
            kind = label("", "Readout")
            form.addRow(label("Kutup sayısı", "FormLabel"), poles)
            for cb in flags.values():
                form.addRow(cb)
                cb.toggled.connect(lambda *_: self._sync_proc_type())
            form.addRow(label("Tür", "FormLabel"), kind)
            poles.currentIndexChanged.connect(lambda *_: self._sync_proc_type())
            widgets.update({"poles": poles, "flags": flags, "kind": kind})
            return page, widgets
        for fkey, text, ftype, default, tip in METHOD_FIELDS[key]:
            if ftype == "int":
                w = QSpinBox()
                w.setRange(0, 400 if fkey in ("n", "M", "units") else 30)
                w.setValue(int(default))
            elif ftype == "vec":
                w = QLineEdit(str(default))
                w.setValidator(QRegularExpressionValidator(INT_LIST, w))
            elif ftype == "float":
                w = QDoubleSpinBox()
                w.setDecimals(4)
                w.setRange(0.0, 1e6)
                w.setValue(float(default))
                w.setSuffix(" s")
            elif ftype == "bool":
                w = QCheckBox(text)
                w.setChecked(bool(default))
            else:
                w = combo([])
                for value, t in default:
                    w.addItem(t, value)
            if tip:
                w.setToolTip(tip)
            widgets[fkey] = w
            if ftype == "bool":
                form.addRow(w)
            else:
                form.addRow(label(text, "FormLabel"), w)
        return page, widgets

    def _sync_proc_type(self) -> None:
        w = self._fields.get("procest")
        if not w:
            return
        w["kind"].setText(self._proc_type())

    def _proc_type(self) -> str:
        w = self._fields["procest"]
        return "P" + w["poles"].currentText() + "".join(f for f, cb in w["flags"].items() if cb.isChecked())

    # ================================================================ sonuç alanı
    def _build_results(self) -> QWidget:
        vsplit = QSplitter(Qt.Orientation.Vertical)
        vsplit.setChildrenCollapsible(False)
        self.pages = QTabWidget()
        self.pages.setDocumentMode(True)
        self.pages.addTab(self._page_data(), "Veri")
        self.pages.addTab(self._page_compare(), "Karşılaştırma")
        self.pages.addTab(self._page_resid(), "Artık analizi")
        self.pages.addTab(self._page_freq(), "Frekans cevabı")
        vsplit.addWidget(self.pages)
        vsplit.addWidget(self._build_table())
        vsplit.setStretchFactor(0, 3)
        vsplit.setStretchFactor(1, 1)
        vsplit.setSizes([620, 230])
        self._vsplit = vsplit
        return vsplit

    @staticmethod
    def _page(*cards: QWidget) -> QWidget:
        """Sayfa: kartlar alanı eşit paylaşır (kaydırma yok: lejant akış yerleşimi kaydırma alanında grafiği
        tercih edilen boyuna (480 px) uzatıyordu; en küçük grafik yüksekliği ezilmeyi önler)."""
        host = QWidget()
        col = QVBoxLayout(host)
        col.setContentsMargins(0, 6, 0, 0)
        col.setSpacing(10)
        for c in cards:
            col.addWidget(c, 1)
        return host

    def _plot_card(self, title: str, x_label: str, y_label: str, x_units=None, y_units=None,
                   si_prefix: bool = True, legend_below: bool = False) -> tuple[Card, pg.PlotWidget, _Legend]:
        """Grafik kartı. Lejant başlık satırında (iki kartlı sayfalarda dikey yer kazanmak için; dar kartta
        sığmazsa gizlenir, renkler tablodaki / listedeki adlarla da eşleşir) ya da `legend_below` ile grafiğin
        altında (satır kaydırmalı)."""
        card = Card(title)
        plot = make_plot(self._theme, x_label, y_label, x_units, y_units, si_prefix)
        plot.setMinimumHeight(PLOT_MIN_HEIGHT)
        card.body.addWidget(plot, 1)
        legend = _Legend(card, wrap=legend_below)
        if legend_below:
            card.body.addWidget(legend)
        else:
            card.add_left(legend, optional=True)
        return card, plot, legend

    def _page_data(self) -> QWidget:
        self.u_card, self.u_plot, self.u_legend = self._plot_card("Giriş (u)", "Zaman", "u", "s")
        self.y_card, self.y_plot, self.y_legend = self._plot_card("Çıkış (y)", "Zaman", "y", "s")
        self.u_plot.setXLink(self.y_plot)
        for c in (self.u_card, self.y_card):
            c.title.setToolTip("Seçili zaman aralığı (turuncu kenarlı bölge) sürüklenerek değiştirilir; gölgeli "
                               "kısım doğrulama verisi, öncesi tahmin verisi")
        t = self._theme.tokens
        self._regions = []
        for plot in (self.u_plot, self.y_plot):
            region = pg.LinearRegionItem(values=(0, 1), movable=True, brush=pg.mkBrush(255, 255, 255, 12),
                                         pen=pg.mkPen(t.accent, width=2))
            region.setZValue(-10)
            region.sigRegionChanged.connect(self._on_region)
            plot.addItem(region, ignoreBounds=True)
            val = pg.LinearRegionItem(values=(0, 1), movable=False, brush=pg.mkBrush(QColor(t.accent).red(),
                                                                                   QColor(t.accent).green(),
                                                                                   QColor(t.accent).blue(), 34),
                                      pen=pg.mkPen(None))
            val.setZValue(-11)
            for ln in val.lines:
                ln.setPen(pg.mkPen(None))
            plot.addItem(val, ignoreBounds=True)
            region.setVisible(False)
            val.setVisible(False)
            self._regions.append((region, val))
        self._tips.append(DataTips(self.u_plot, self._theme, lambda: self._traces_for(self.u_plot)))
        self._tips.append(DataTips(self.y_plot, self._theme, lambda: self._traces_for(self.y_plot)))
        return self._page(self.u_card, self.y_card)

    def _page_compare(self) -> QWidget:
        self.cmp_card, self.cmp_plot, self.cmp_legend = self._plot_card(
            "Ölçülen ve benzetilen çıkış (doğrulama verisi)", "Zaman", "y", "s", legend_below=True)
        self.cmp_output = combo([])
        self.cmp_output.setToolTip("Gösterilen çıkış kanalı")
        self.cmp_output.currentIndexChanged.connect(lambda *_: self._draw_compare())
        self.cmp_card.add_right(self.cmp_output)
        self.cmp_note = label("", "Muted")
        self.cmp_note.setWordWrap(True)
        self.cmp_card.body.addWidget(self.cmp_note)
        self._tips.append(DataTips(self.cmp_plot, self._theme, lambda: self._cmp_traces))
        return self._page(self.cmp_card)

    def _page_resid(self) -> QWidget:
        self.auto_card, self.auto_plot, self.auto_legend = self._plot_card(
            "Artıkların özilintisi", "Gecikme (örnek)", "r_e", si_prefix=False)
        self.cross_card, self.cross_plot, self.cross_legend = self._plot_card(
            "Giriş ile artıkların çapraz ilintisi", "Gecikme (örnek)", "r_ue", si_prefix=False)
        self.res_output = combo([])
        self.res_output.setToolTip("Artıkları gösterilen çıkış")
        self.res_input = combo([])
        self.res_input.setToolTip("Çapraz ilintideki giriş")
        self.res_output.currentIndexChanged.connect(lambda *_: self._draw_resid())
        self.res_input.currentIndexChanged.connect(lambda *_: self._draw_resid())
        self.auto_card.add_right(self.res_output)
        self.cross_card.add_right(self.res_input)
        resid_tip = ("Doğrulama verisinde bir adım öngörü hataları (MATLAB resid). Beyaz artık: özilinti 0 dışındaki "
                     "gecikmelerde kesikli sınırlar içinde. Artık girişten bağımsız: çapraz ilinti sınırlar içinde "
                     "(negatif gecikmede sınır dışı → geri besleme olabilir). Sınırlar %99, Ljung'un tanımıyla; "
                     "çapraz ilinti sınırı model başına (model renginde kesikli).")
        self.auto_card.title.setToolTip(resid_tip)
        self.cross_card.title.setToolTip(resid_tip)
        self._tips.append(DataTips(self.auto_plot, self._theme, lambda: self._res_traces["auto"]))
        self._tips.append(DataTips(self.cross_plot, self._theme, lambda: self._res_traces["cross"]))
        return self._page(self.auto_card, self.cross_card)

    def _page_freq(self) -> QWidget:
        self.mag_card, self.mag_plot, self.mag_legend = self._plot_card("Genlik", "Frekans", "Genlik (dB)",
                                                                         si_prefix=False)
        self.phase_card, self.phase_plot, self.phase_legend = self._plot_card("Faz", "Frekans", "Faz (°)",
                                                                               si_prefix=False)
        for p in (self.mag_plot, self.phase_plot):
            p.setLogMode(x=True, y=False)
        self.phase_plot.setXLink(self.mag_plot)
        self.freq_output = combo([])
        self.freq_input = combo([])
        self.freq_output.setToolTip("Çıkış kanalı")
        self.freq_input.setToolTip("Giriş kanalı")
        self.freq_unit = combo(["rad/s", "Hz"])
        self.freq_unit.setToolTip("Frekans ekseni birimi")
        for w in (self.freq_output, self.freq_input, self.freq_unit):
            w.currentIndexChanged.connect(lambda *_: self._draw_freq())
        self.mag_card.add_right(self.freq_input)
        self.mag_card.add_right(self.freq_output)
        self.mag_card.add_right(self.freq_unit)
        self._freq_tip = ("Ölçüm modelinin frekans cevabı (Nyquist frekansına kadar); parametrik olmayan referans: "
                          "çalışma verisinden spektral analiz (spa). Doğrusal olmayan modellerin frekans cevabı "
                          "yoktur.")
        self.mag_card.title.setToolTip(self._freq_tip)
        self.phase_card.title.setToolTip(self._freq_tip)
        self._tips.append(DataTips(self.mag_plot, self._theme, lambda: self._freq_traces["mag"]))
        self._tips.append(DataTips(self.phase_plot, self._theme, lambda: self._freq_traces["phase"]))
        return self._page(self.mag_card, self.phase_card)

    def _build_table(self) -> Card:
        card = Card("Modeller")
        self.add_button = QPushButton("Modellere ekle")
        self.add_button.setObjectName("Primary")
        self.add_button.setToolTip("Seçili modelleri (yoksa işaretli tümünü) Modeller sekmesindeki çalışma alanına "
                                   "ekle (transfer fonksiyonu / durum uzayı)")
        self.add_button.clicked.connect(lambda: self.add_to_workspace())
        self.validate_button = QPushButton("Geçerli veriyle doğrula")
        self.validate_button.setToolTip("Tüm modelleri şu anki kanal, aralık ve ön işleme ayarlarının doğrulama "
                                        "verisiyle yeniden karşılaştır")
        self.validate_button.clicked.connect(self.validate)
        self.delete_button = QPushButton("Sil")
        self.delete_button.setToolTip("Seçili modelleri sil (renkleri serbest kalır)")
        self.delete_button.clicked.connect(self.delete_selected)
        for b in (self.validate_button, self.delete_button, self.add_button):
            card.add_right(b)
        self.model_table = QTableWidget(0, 9)
        self.model_table.setHorizontalHeaderLabels(["Göster", "Ad", "Yöntem", "Derece", "Ts",
                                                    "Uyum % (tahmin)", "Uyum % (doğrulama)", "FPE", "AIC"])
        tips = ["Grafiklerde göster", "Model adı (renk kimliği)", "Tahmin yöntemi", "Dereceler",
                "Örnekleme aralığı (0: sürekli zaman)",
                "Tahmin verisinde benzetim uyumu: 100·(1 − ‖y − ŷ‖/‖y − ȳ‖), çıkış başına (compare, k = ∞)",
                "Doğrulama verisinde benzetim uyumu (compare, k = ∞, başlangıç durumu tahmin edilir)",
                "Son öngörü hatası (tahmin raporu, MATLAB tanımı)", "Akaike ölçütü (tahmin raporu, MATLAB tanımı)"]
        for i, tip in enumerate(tips):
            self.model_table.horizontalHeaderItem(i).setToolTip(tip)
        self.model_table.verticalHeader().setVisible(False)
        self.model_table.setShowGrid(False)
        self.model_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.model_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.model_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.model_table.verticalHeader().setDefaultSectionSize(26)
        hh = self.model_table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hh.setStretchLastSection(True)
        hh.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft)
        self.model_table.itemChanged.connect(self._on_table_item)
        self.model_table.itemSelectionChanged.connect(self._update_buttons)
        self.model_table.setMinimumHeight(90)
        card.body.addWidget(self.model_table, 1)
        return card

    # ================================================================ kaynaklar
    def set_sources(self, sources: list[tuple[str, str, object]]) -> None:
        """sources: (kimlik, etiket, tutamaç) — canlı / oynatılan akışlar ve Çalışma Alanı anlık görüntüleri.

        Seçili kaynak listede kalırsa korunur (veri yeniden okunmaz; canlı akışta "Yenile")."""
        self._sources = list(sources)
        self._refresh_sources()

    def _refresh_sources(self) -> None:
        current = self.source.currentData()
        self.source.blockSignals(True)
        self.source.clear()
        for sid, text, _ in self._sources:
            self.source.addItem(text, sid)
        for key, tab in self._files.items():
            self.source.addItem(tab.label, key)
        index = self.source.findData(current) if current is not None else -1
        self.source.setCurrentIndex(index if index >= 0 else (0 if self.source.count() else -1))
        self.source.blockSignals(False)
        if self.source.currentData() != current or (self.table is None and self.source.count()):
            self._on_source()
        elif self.source.count() == 0:
            self._set_table(None)

    def _handle(self, sid):
        return next((h for s, _, h in self._sources if s == sid), None)

    def _on_source(self, *_) -> None:
        sid = self.source.currentData()
        if sid is None:
            self._set_table(None)
            return
        if sid in self._files:
            self._set_table(self._files[sid])
            return
        handle = self._handle(sid)
        if handle is None:
            self._set_table(None)
            return
        try:
            tab = J.table_from_handle(sid, self.source.currentText(), handle)
        except ControlError as exc:
            self._set_table(None)
            self._set_status(str(exc), "critical")
            return
        self._set_table(tab)

    def _reload(self) -> None:
        sid = self.source.currentData()
        if sid is None or sid in self._files:
            self._on_source()
            return
        keep = (self._checked(self.inputs_list), self._checked(self.outputs_list), self.t0_spin.value(),
                self.t1_spin.value())
        self._on_source()
        if self.table is not None:
            self._apply_channels(*keep[:2])
            self._set_status("Akışın güncel verisi alındı.", "good")

    def open_file(self, path) -> bool:
        """CSV / TXT / MAT dosyasını kaynak olarak ekler ve seçer. Hata Türkçe durum iletisiyle bildirilir."""
        try:
            tab = J.table_from_file(path)
        except (ControlError, OSError, UnicodeDecodeError) as exc:
            self._set_status(f"Dosya açılamadı: {exc}", "critical")
            return False
        self._files[tab.key] = tab
        self._refresh_sources()
        self.source.setCurrentIndex(self.source.findData(tab.key))
        if self.table is not tab:
            self._set_table(tab)
        return True

    def _open_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Tanımlama verisi aç", "",
                                              "Veri dosyaları (*.csv *.txt *.dat *.mat);;Tüm dosyalar (*)")
        if path:
            self.open_file(path)

    def _set_table(self, tab: J.SourceTable | None) -> None:
        self.table = tab
        for lst in (self.inputs_list, self.outputs_list):
            lst.blockSignals(True)
            lst.clear()
        if tab is None:
            for lst in (self.inputs_list, self.outputs_list):
                lst.blockSignals(False)
            self.source_info.setText("Kaynak yok: akış başlatın, Çalışma Alanı'na anlık görüntü alın ya da dosya açın.")
            self.reload_button.setEnabled(False)
            self._draw_data()
            self._update_buttons()
            return
        for i in tab.channels():
            for lst in (self.inputs_list, self.outputs_list):
                it = QListWidgetItem(tab.channel_text(i))
                it.setData(Qt.ItemDataRole.UserRole, i)
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(Qt.CheckState.Unchecked)
                lst.addItem(it)
        for lst in (self.inputs_list, self.outputs_list):
            lst.blockSignals(False)
            self._fit_list(lst)
        self.reload_button.setEnabled(tab.kind == "stream")
        dur = tab.N * tab.Ts if tab.Ts else None
        info = f"{tab.N} örnek · {len(tab.channels())} kanal"
        if tab.Ts:
            info += f" · Ts {fmt(tab.Ts, 's')} ({fmt(1 / tab.Ts, 'Hz')}) · {fmt(dur, 's')}"
        else:
            info += " · zaman bilgisi yok (Ts'i girin)"
        missing = int(np.isnan(tab.X[:, tab.channels()]).any(axis=1).sum()) if tab.N else 0
        if missing:
            info += f" · {missing} eksik örnek"
        if tab.note:
            info += f"<br>{tab.note}"
        self.source_info.setText(info)
        self.ts_auto.blockSignals(True)
        self.ts_auto.setEnabled(tab.Ts is not None)
        self.ts_auto.setChecked(tab.Ts is not None)
        self.ts_auto.blockSignals(False)
        self.ts_edit.setEnabled(tab.Ts is None)
        self.ts_edit.setText(f"{tab.Ts:.10g}" if tab.Ts else "")
        self._guess_channels()
        self._update_time_limits(full=True)
        self._draw_data()
        self._update_buttons()

    def _guess_channels(self) -> None:
        tab = self.table
        chans = tab.channels()
        names = {i: tab.names[i] for i in chans}
        ins = [i for i in chans if INPUT_HINT.search(names[i])]
        outs = [i for i in chans if OUTPUT_HINT.search(names[i]) and i not in ins]
        if not ins or not outs:
            ins, outs = chans[:1], chans[1:2]
        self._apply_channels(ins[:3], outs[:3])

    def _apply_channels(self, inputs: list[int], outputs: list[int]) -> None:
        for lst, sel in ((self.inputs_list, inputs), (self.outputs_list, outputs)):
            lst.blockSignals(True)
            for r in range(lst.count()):
                it = lst.item(r)
                it.setCheckState(Qt.CheckState.Checked if it.data(Qt.ItemDataRole.UserRole) in sel
                                 else Qt.CheckState.Unchecked)
            lst.blockSignals(False)
        self._sync_channel_colors()
        self._schedule_data()

    def select_channels(self, inputs=(), outputs=()) -> None:
        """Kanalları adla ya da indisle seçer (kaynak tablosundaki sırayla)."""
        if self.table is None:
            return

        def idx(sel):
            out = []
            for s in sel:
                if isinstance(s, str):
                    if s not in self.table.names:
                        raise ControlError(f"'{s}' adlı kanal yok")
                    out.append(self.table.names.index(s))
                else:
                    out.append(int(s))
            return out
        self._apply_channels(idx(inputs), idx(outputs))

    @staticmethod
    def _checked(lst: QListWidget) -> list[int]:
        return [lst.item(r).data(Qt.ItemDataRole.UserRole) for r in range(lst.count())
                if lst.item(r).checkState() == Qt.CheckState.Checked]

    @property
    def inputs(self) -> list[int]:
        return self._checked(self.inputs_list)

    @property
    def outputs(self) -> list[int]:
        return self._checked(self.outputs_list)

    def _on_channel_item(self, item: QListWidgetItem) -> None:
        if item.checkState() == Qt.CheckState.Checked:  # bir kanal hem giriş hem çıkış olmasın
            other = self.outputs_list if item.listWidget() is self.inputs_list else self.inputs_list
            ch = item.data(Qt.ItemDataRole.UserRole)
            other.blockSignals(True)
            for r in range(other.count()):
                if other.item(r).data(Qt.ItemDataRole.UserRole) == ch:
                    other.item(r).setCheckState(Qt.CheckState.Unchecked)
            other.blockSignals(False)
        self._sync_channel_colors()
        self._schedule_data()
        self._update_buttons()

    def _sync_channel_colors(self) -> None:
        """Kanal renkleri: önce çıkışlar, sonra girişler (seçim sırasıyla); 7 kanala kadar."""
        colors = {}
        k = 0
        for kind, chans in (("y", self.outputs), ("u", self.inputs)):
            for i in chans:
                if k < len(CHANNEL_COLORS):
                    colors[(kind, i)] = CHANNEL_COLORS[k]
                k += 1
        self._chan_colors = colors
        for lst, kind in ((self.inputs_list, "u"), (self.outputs_list, "y")):
            lst.blockSignals(True)
            for r in range(lst.count()):
                it = lst.item(r)
                c = colors.get((kind, it.data(Qt.ItemDataRole.UserRole)))
                it.setIcon(line_swatch(c) if c else line_swatch("#00000000"))
            lst.blockSignals(False)

    # ================================================================ Ts ve aralık
    def _on_ts_auto(self, on: bool) -> None:
        self.ts_edit.setEnabled(not on)
        if on and self.table is not None and self.table.Ts:
            self.ts_edit.setText(f"{self.table.Ts:.10g}")
        self._update_time_limits(full=True)
        self._schedule_data()

    def _on_ts_changed(self) -> None:
        self._update_time_limits(full=True)
        self._schedule_data()

    def current_Ts(self) -> float | None:
        if self.table is not None and self.ts_auto.isChecked() and self.table.Ts:
            return self.table.Ts
        text = self.ts_edit.text().strip().replace(",", ".")
        try:
            v = float(text)
        except ValueError:
            return None
        return v if np.isfinite(v) and v > 0 else None

    def set_Ts(self, Ts: float | None) -> None:
        """Ts'i elle verir (None: otomatik)."""
        if Ts is None:
            self.ts_auto.setChecked(True)
            return
        self.ts_auto.setChecked(False)
        self.ts_edit.setText(f"{Ts:.10g}")
        self._on_ts_changed()

    def _time_axis(self) -> np.ndarray | None:
        Ts = self.current_Ts()
        if self.table is None or Ts is None:
            return None
        return self.table.tstart + Ts * np.arange(self.table.N)

    def _update_time_limits(self, full: bool = False) -> None:
        t = self._time_axis()
        for s in (self.t0_spin, self.t1_spin):
            s.blockSignals(True)
        if t is None or len(t) == 0:
            for s in (self.t0_spin, self.t1_spin):
                s.setRange(0.0, 0.0)
        else:
            lo, hi = float(t[0]), float(t[-1])
            old = (self.t0_spin.value(), self.t1_spin.value())
            for s in (self.t0_spin, self.t1_spin):
                s.setRange(lo, hi)
                s.setSingleStep(max((hi - lo) / 100, 1e-4))
            if full:
                self.t0_spin.setValue(lo)
                self.t1_spin.setValue(hi)
            else:
                self.t0_spin.setValue(min(max(old[0], lo), hi))
                self.t1_spin.setValue(min(max(old[1], lo), hi))
        for s in (self.t0_spin, self.t1_spin):
            s.blockSignals(False)
        self._sync_regions()

    def set_range(self, t0: float, t1: float) -> None:
        self.t0_spin.setValue(t0)
        self.t1_spin.setValue(t1)

    def _full_range(self) -> None:
        self._update_time_limits(full=True)

    def _on_range_spin(self, *_) -> None:
        if self.t1_spin.value() < self.t0_spin.value():
            sender = self.sender()
            other = self.t1_spin if sender is self.t0_spin else self.t0_spin
            other.blockSignals(True)
            other.setValue(sender.value())
            other.blockSignals(False)
        self._sync_regions()

    def _sync_regions(self) -> None:
        if self._region_lock:
            return
        self._region_lock = True
        try:
            a, b = self.t0_spin.value(), self.t1_spin.value()
            for region, _ in self._regions:
                region.setRegion((a, b))
        finally:
            self._region_lock = False
        self._update_split_region()

    def _on_region(self, region) -> None:
        if self._region_lock:
            return
        a, b = region.getRegion()
        self._region_lock = True
        try:
            for other, _ in self._regions:
                if other is not region:
                    other.setRegion((a, b))
            for s, v in ((self.t0_spin, a), (self.t1_spin, b)):
                s.blockSignals(True)
                s.setValue(v)
                s.blockSignals(False)
        finally:
            self._region_lock = False
        self._update_split_region()

    def _split_time(self) -> float | None:
        """Tahmin / doğrulama sınırının zamanı (split ile aynı kural: ilk round(n·oran) örnek tahmin)."""
        Ts = self.current_Ts()
        if self.table is None or Ts is None:
            return None
        f = self.fraction_spin.value() / 100.0
        if f >= 1.0:
            return None
        k0, k1 = J.sample_range(self.table, Ts, self.t0_spin.value(), self.t1_spin.value())
        n = k1 - k0
        if n < 2:
            return None
        return self.table.tstart + Ts * (k0 + int(round(n * f)))

    def _update_split_region(self) -> None:
        ts = self._split_time()
        for _, val in self._regions:
            if ts is None:
                val.setVisible(False)
            else:
                val.setRegion((ts, max(self.t1_spin.value(), ts)))
                val.setVisible(self.table is not None)

    # ================================================================ ön işleme / yöntem ayarları
    def _on_filter_kind(self, *_) -> None:
        kind = self._filter_kind()
        on = kind is not None
        band = kind in ("bandpass", "bandstop")
        for w in (self.filter_order, self.filter_f1, self.zero_phase):
            w.setEnabled(on)
        self.filter_f2.setEnabled(band)

    def _filter_kind(self) -> str | None:
        i = self.filter_combo.currentIndex()
        return None if i <= 0 else list(J.FILTER_KINDS)[i - 1]

    def prep(self) -> J.Prep:
        return J.Prep(
            t0=self.t0_spin.value(), t1=self.t1_spin.value(),
            missing=("error", "linear", "model")[self.missing_combo.currentIndex()],
            detrend=(None, 0, 1)[self.detrend_combo.currentIndex()],
            filter_kind=self._filter_kind(), filter_order=self.filter_order.value(),
            filter_f1=self.filter_f1.value(), filter_f2=self.filter_f2.value(), zero_phase=self.zero_phase.isChecked(),
            P=self.res_p.value(), Q=self.res_q.value(), fraction=self.fraction_spin.value() / 100.0)

    def set_prep(self, **kw) -> None:
        """Ön işleme ayarları (testler / betikler): detrend (None/0/1), missing, filter_kind, filter_order,
        filter_f1, filter_f2, zero_phase, P, Q, fraction (0–1)."""
        if "detrend" in kw:
            self.detrend_combo.setCurrentIndex({None: 0, 0: 1, 1: 2}[kw["detrend"]])
        if "missing" in kw:
            self.missing_combo.setCurrentIndex(("error", "linear", "model").index(kw["missing"]))
        if "filter_kind" in kw:
            k = kw["filter_kind"]
            self.filter_combo.setCurrentIndex(0 if k is None else list(J.FILTER_KINDS).index(k) + 1)
        for key, w in (("filter_order", self.filter_order), ("filter_f1", self.filter_f1),
                       ("filter_f2", self.filter_f2), ("P", self.res_p), ("Q", self.res_q)):
            if key in kw:
                w.setValue(kw[key])
        if "zero_phase" in kw:
            self.zero_phase.setChecked(bool(kw["zero_phase"]))
        if "fraction" in kw:
            self.fraction_spin.setValue(int(round(kw["fraction"] * 100)))

    def _set_method(self, key: str) -> None:
        if key is None:
            return
        i = self.method_combo.findData(key)
        if i != self.method_combo.currentIndex():
            self.method_combo.blockSignals(True)
            self.method_combo.setCurrentIndex(i)
            self.method_combo.blockSignals(False)
        self.method_stack.setCurrentIndex(self._pages[key])
        self._sync_proc_type()
        self.suggest_button.setVisible(key == "arx")
        notes = {
            "arx": "A(q) y = B(q) u + e — doğrusal en küçük kareler (hızlı; çok çıkışlı veride de).",
            "armax": "A y = B u + C e — öngörü hatası yöntemi (tek çıkış).",
            "oe": "y = B/F u + e — gürültü modeli yok; benzetim uyumu için iyi (tek çıkış).",
            "bj": "y = B/F u + C/D e — ölçüm ve gürültü ayrı modellenir (tek çıkış).",
            "tfest": "Sürekli (ya da ayrık) zamanlı transfer fonksiyonu (tek çıkış).",
            "procest": "Kazanç, zaman sabitleri, ölü zaman: Kp·(1+Tz·s)/((1+Tp1·s)…)·e^(−Td·s) (tek çıkış).",
            "n4sid": "Altuzay yöntemiyle durum uzayı (çok giriş / çok çıkış).",
            "ssest": "Öngörü hatası yöntemiyle durum uzayı (çok giriş / çok çıkış).",
            "impulseest": "Parametrik olmayan FIR darbe cevabı.",
            "spa": "Parametrik olmayan frekans cevabı (Blackman–Tukey); yalnız frekans grafiğinde.",
            "etfe": "Ampirik transfer fonksiyonu (çıkış / giriş Fourier oranı); yalnız frekans grafiğinde.",
            "nlarx": "Doğrusal olmayan ARX (regresörler üzerinde doğrusal olmayan eşleme).",
            "nlhw": "Hammerstein–Wiener: statik giriş doğrusalsızlığı → doğrusal blok → çıkış doğrusalsızlığı.",
        }
        text = notes.get(key, "")
        if not J.method_available(key):
            text += " <b>Bu yöntem bu sürümde henüz kullanılamıyor.</b>"
        self.method_note.setText(text)
        self._update_buttons()

    def method(self) -> str:
        return self.method_combo.currentData()

    def set_method(self, key: str, **opts) -> None:
        """Yöntemi ve derecelerini ayarlar (ör. `set_method("arx", na=2, nb=2, nk=1)`)."""
        if key not in J.METHODS:
            raise ControlError("Bilinmeyen yöntem")
        self._set_method(key)
        w = self._fields[key]
        if key == "procest" and "type" in opts:
            t = str(opts["type"]).upper()
            m = re.match(r"^P(\d)([DZUI]*)$", t)
            if not m:
                raise ControlError(f"Geçersiz süreç modeli türü: {t}")
            w["poles"].setCurrentIndex(int(m.group(1)))
            for f, cb in w["flags"].items():
                cb.setChecked(f in m.group(2))
            return
        for k, v in opts.items():
            if k not in w:
                raise ControlError(f"'{k}' bu yöntemin ayarı değil")
            box = w[k]
            if isinstance(box, QLineEdit):
                box.setText(" ".join(str(int(x)) for x in np.atleast_1d(v)))
            elif isinstance(box, QCheckBox):
                box.setChecked(bool(v))
            elif isinstance(box, QComboBox):
                box.setCurrentIndex(max(box.findData(v), 0))
            else:
                box.setValue(v)

    def method_opts(self, key: str | None = None) -> dict:
        key = key or self.method()
        w = self._fields[key]
        if key == "procest":
            return {"type": self._proc_type()}
        out = {}
        for fkey, text, ftype, _, _ in METHOD_FIELDS[key]:
            box = w[fkey]
            if ftype == "vec":
                s = box.text().strip()
                if not s:
                    raise ControlError(f"'{text}' boş olamaz")
                out[fkey] = [int(x) for x in re.split(r"[\s,;]+", s) if x]
            elif ftype == "bool":
                out[fkey] = box.isChecked()
            elif ftype == "choice":
                out[fkey] = box.currentData()
            else:
                out[fkey] = box.value()
        return out

    # ================================================================ tahmin işi
    @property
    def busy(self) -> bool:
        return self.runner.busy

    def _prepare(self):
        """Seçimden (çalışma, tahmin, doğrulama) verisi; geçersiz seçimde ControlError."""
        if self.table is None:
            raise ControlError("Önce bir veri kaynağı seçin ya da dosya açın")
        ins, outs = self.inputs, self.outputs
        if len(ins) + len(outs) > len(CHANNEL_COLORS):
            raise ControlError(f"En çok {len(CHANNEL_COLORS)} kanal (giriş + çıkış) seçilebilir")
        Ts = self.current_Ts()
        if Ts is None:
            raise ControlError("Örnekleme aralığı Ts girin (pozitif sayı, saniye)")
        return J.build_data(self.table, ins, outs, Ts, self.prep())

    def _unique_name(self, base: str) -> str:
        names = {e.name for e in self.entries}
        if base not in names:
            return base
        k = 2
        while f"{base}_{k}" in names:
            k += 1
        return f"{base}_{k}"

    def _free_color(self) -> str | None:
        used = {e.color for e in self.entries}
        return next((c for c in MODEL_COLORS if c not in used), None)

    def estimate(self) -> bool:
        """Seçili yöntemle tahmini arka planda başlatır. Geçersiz seçimde Türkçe durum iletisi, False."""
        if self.busy:
            return False
        try:
            method = self.method()
            if not J.method_available(method):
                raise ControlError(f"{J.METHODS[method][0]} yöntemi bu sürümde henüz kullanılamıyor")
            if self._free_color() is None:
                raise ControlError(f"En çok {len(MODEL_COLORS)} model tutulabilir (her birinin ayrı rengi var); "
                                   "yer açmak için model silin")
            opts = self.method_opts(method)
            work, est, val = self._prepare()
            J.plan(method, opts, est.nu, est.ny, est.Ts)  # geçersiz derece / yapı: iş başlamadan ileti
        except ControlError as exc:
            self._set_status(str(exc), "critical")
            return False
        name = self._unique_name(J.model_name(method, opts, est.nu))
        job = {"est": est, "val": val, "work": work, "new": {"method": method, "opts": opts, "name": name},
               "models": [(e.name, e.model) for e in self.entries], "max_lag": MAX_LAG, "spa": True}
        self._pending = {"method": method, "opts": opts, "name": name, "work": work, "est": est, "val": val,
                         "data_text": self._data_text(work)}
        self._start("estimate", J.run_job, job)
        self._set_status(f"{name} tahmin ediliyor…", "busy")
        return True

    def validate(self) -> bool:
        """Tüm modelleri geçerli ayarların doğrulama verisiyle yeniden değerlendirir (arka planda)."""
        if self.busy:
            return False
        if not self.entries:
            self._set_status("Doğrulanacak model yok", "warning")
            return False
        try:
            work, est, val = self._prepare()
        except ControlError as exc:
            self._set_status(str(exc), "critical")
            return False
        job = {"est": est, "val": val, "work": work, "new": None,
               "models": [(e.name, e.model) for e in self.entries], "max_lag": MAX_LAG, "spa": True}
        self._pending = {"work": work, "est": est, "val": val}
        self._start("validate", J.run_job, job)
        self._set_status("Modeller doğrulanıyor…", "busy")
        return True

    def suggest_orders(self) -> bool:
        """ARX derece önerisi (arxstruc / selstruc) arka planda; sonuç na / nb / nk alanlarına yazılır."""
        if self.busy:
            return False
        try:
            if not (J.method_available("arxstruc") and J.method_available("selstruc")):
                raise ControlError("Derece önerisi (arxstruc / selstruc) bu sürümde henüz kullanılamıyor")
            work, est, val = self._prepare()
            if est.ny != 1 or est.nu != 1:
                raise ControlError("Derece önerisi tek giriş / tek çıkışlı veride yapılır")
        except ControlError as exc:
            self._set_status(str(exc), "critical")
            return False
        self._pending = {}
        self._start("suggest", J.suggest_arx_orders, est, val)
        self._set_status("ARX dereceleri taranıyor…", "busy")
        return True

    def _start(self, kind: str, func, *args) -> None:
        self.runner.start(kind, func, *args)
        self._update_buttons()
        self.busy_changed.emit(True)

    def cancel(self) -> bool:
        """Süren hesabı durdurur; tablo ve grafikler önceki hâlinde kalır."""
        if not self.runner.cancel():
            return False
        self._pending = None
        self._set_status("Hesap iptal edildi.", "warning")
        self._update_buttons()
        self.busy_changed.emit(False)
        return True

    def _on_failed(self, kind: str, message: str) -> None:
        self._pending = None
        self._set_status(message, "critical")
        self._update_buttons()
        self.busy_changed.emit(False)

    def _on_finished(self, kind: str, result) -> None:
        pending, self._pending = self._pending, None
        self._update_buttons()
        self.busy_changed.emit(False)
        if pending is None:
            return
        if kind == "suggest":
            na, nb, nk = result
            self.set_method("arx", na=na, nb=nb, nk=nk)
            self._set_status(f"Önerilen ARX derecesi: na={na}, nb={nb}, nk={nk} (doğrulama verisinde AIC).", "good")
            return
        self.apply_result(result, pending)

    def apply_result(self, result: dict, pending: dict) -> None:
        """İş sonucunu tabloya ve grafiklere uygular (`run_job` çıktısı)."""
        new = result.get("new")
        if new is not None:
            name, model = new
            self.entries.append(ModelEntry(name, pending["method"], pending["opts"], model, self._free_color(),
                                           data_text=pending.get("data_text", "")))
        for e in self.entries:
            if e.name in result["evals"]:
                e.ev = result["evals"][e.name]
        work, val = pending["work"], pending["val"]
        self.context = _Context(work, pending["est"], val, result.get("spa"), result.get("spa_error", ""),
                                result.get("w"), {"u": list(val.input_names), "y": list(val.output_names)})
        self._fill_io_combos()
        self._refresh_table()
        self.redraw_results()
        if new is not None:
            ev = self.entries[-1].ev
            errs = "; ".join(ev.errors) if ev and ev.errors else ""
            fit = _fmt_fit(ev.fit_val) if ev else "—"
            self._set_status(f"{new[0]} tahmin edildi · doğrulama uyumu %{fit}" + (f" · {errs}" if errs else ""),
                             "warning" if errs else "good")
            if self.pages.currentIndex() == 0:
                self.pages.setCurrentIndex(1 if new[0] and pending["method"] not in J.NONPARAMETRIC else 3)
        else:
            self._set_status("Modeller geçerli veriyle doğrulandı.", "good")

    @staticmethod
    def _data_text(d) -> str:
        return f"{d.name}, {d.nu} giriş / {d.ny} çıkış, {sum(d.lengths)} örnek, Ts {d.Ts:g} s"

    # ================================================================ model tablosu
    def _refresh_table(self) -> None:
        t = self.model_table
        t.blockSignals(True)
        t.setRowCount(len(self.entries))
        for r, e in enumerate(self.entries):
            show = QTableWidgetItem()
            show.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            show.setCheckState(Qt.CheckState.Checked if e.visible else Qt.CheckState.Unchecked)
            show.setIcon(line_swatch(e.color))
            ev = e.ev
            Ts = getattr(e.model, "Ts", None)
            cells = [show, QTableWidgetItem(e.name), QTableWidgetItem(J.METHODS[e.method][0]),
                     QTableWidgetItem(J.orders_text(e.method, e.opts)),
                     QTableWidgetItem("sürekli" if Ts == 0 else ("—" if Ts is None else f"{Ts:g} s")),
                     QTableWidgetItem(_fmt_fit(ev.fit_est if ev else None)),
                     QTableWidgetItem(_fmt_fit(ev.fit_val if ev else None)),
                     QTableWidgetItem(_fmt_num(ev.fpe if ev else None)),
                     QTableWidgetItem(_fmt_num(ev.aic if ev else None))]
            cells[1].setForeground(QColor(e.color))
            tip = f"{e.name}: {J.METHODS[e.method][0]} · veri: {e.data_text}"
            if ev and ev.errors:
                tip += "\n" + "\n".join(ev.errors)
            for c, it in enumerate(cells):
                if c > 0:
                    it.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                it.setToolTip(tip)
                t.setItem(r, c, it)
        t.blockSignals(False)
        self._update_buttons()

    def _on_table_item(self, item: QTableWidgetItem) -> None:
        if item.column() != 0:
            return
        e = self.entries[item.row()]
        e.visible = item.checkState() == Qt.CheckState.Checked
        self.redraw_results()

    def selected_entries(self) -> list[ModelEntry]:
        rows = sorted({i.row() for i in self.model_table.selectedIndexes()})
        return [self.entries[r] for r in rows if r < len(self.entries)]

    def entry(self, name: str) -> ModelEntry | None:
        return next((e for e in self.entries if e.name == name), None)

    def delete_selected(self) -> None:
        sel = self.selected_entries()
        if not sel:
            self._set_status("Silinecek modeli tabloda seçin", "warning")
            return
        self.remove_models([e.name for e in sel])

    def remove_models(self, names) -> None:
        names = set(names)
        self.entries = [e for e in self.entries if e.name not in names]
        self._refresh_table()
        self.redraw_results()

    def add_to_workspace(self, names=None) -> list[str]:
        """Modelleri Modeller sekmesindeki çalışma alanına ekler (tanımlanmış model olarak; analizde ölçüm modeli
        kullanılır). Döner: çalışma alanındaki son adlar (ad doluysa arx2212 gibi benzersiz ad seçilir).

        Parametrik olmayan (spa / etfe) ve doğrusal olmayan (nlarx / nlhw) modeller eklenemez (Türkçe ileti).
        Çalışma alanı yoksa yalnız `models_ready` sinyali yayılır."""
        if names is None:
            sel = self.selected_entries() or [e for e in self.entries if e.visible]
        else:
            sel = [e for e in self.entries if e.name in set(names)]
        if not sel:
            self._set_status("Eklenecek model yok: önce bir model tahmin edin", "warning")
            return []
        out, skipped = [], []
        for e in sel:
            if not e.addable:
                kind = "parametrik olmayan" if e.method in J.NONPARAMETRIC else "doğrusal olmayan"
                skipped.append(f"{e.name} ({kind} model; transfer fonksiyonu / durum uzayı değil)")
                continue
            try:
                sys = e.model.to_lti()
            except (ControlError, NotImplementedError, AttributeError) as exc:
                skipped.append(f"{e.name} ({exc})")
                continue
            out.append((e, sys))
        added, ready = [], []
        for e, sys in out:
            name = e.name
            if self.workspace is not None:
                try:
                    name = self._workspace_add(e)
                except (ControlError, ValueError, TypeError) as exc:
                    skipped.append(f"{e.name} ({exc})")
                    continue
            added.append(name)
            ready.append((name, sys, e.model))
        if ready:
            self.models_ready.emit(ready)
        msg = f"Modellere eklendi: {', '.join(added)}" if added else "Hiçbir model eklenmedi"
        if skipped:
            msg += f" · eklenemeyen: {'; '.join(skipped)}"
        self._set_status(msg, "good" if added and not skipped else ("warning" if added else "critical"))
        return added

    def _workspace_add(self, e: ModelEntry) -> str:
        """ModelWorkspace.add(ad, model, source, note) — son adı döndürür."""
        try:
            from spt.gui.control.workspace import SOURCE_SYSID
        except ImportError:  # pragma: no cover
            SOURCE_SYSID = "sistem tanımlama"
        ev = e.ev
        note = f"{J.METHODS[e.method][0]} {J.orders_text(e.method, e.opts)}"
        if ev is not None and ev.fit_val is not None:
            note += f", doğrulama uyumu %{_fmt_fit(ev.fit_val)}"
        name = self.workspace.add(e.name, e.model, source=SOURCE_SYSID, note=note)
        return name if isinstance(name, str) else e.name

    def detachable_cards(self) -> list[tuple[str, str, Card]]:
        """Ayrı pencereye alınabilen sonuç kartları (anahtar, başlık, kart)."""
        return [("u", "Sistem Tanımlama · Giriş (u)", self.u_card),
                ("y", "Sistem Tanımlama · Çıkış (y)", self.y_card),
                ("compare", "Sistem Tanımlama · Karşılaştırma", self.cmp_card),
                ("auto", "Sistem Tanımlama · Artık özilintisi", self.auto_card),
                ("cross", "Sistem Tanımlama · Çapraz ilinti", self.cross_card),
                ("mag", "Sistem Tanımlama · Genlik", self.mag_card),
                ("phase", "Sistem Tanımlama · Faz", self.phase_card)]

    # ================================================================ çizim: veri
    def _schedule_data(self) -> None:
        self._redraw_timer.start()

    def _traces_for(self, plot) -> list[Trace]:
        return [tr for tr in self._data_curves if tr.extra.get("plot") is plot]

    def _draw_data(self) -> None:
        for plot, legend in ((self.u_plot, self.u_legend), (self.y_plot, self.y_legend)):
            for item in list(plot.listDataItems()):
                plot.removeItem(item)
            self._clear_legend(legend)
        self._data_curves = []
        t = self._time_axis()
        visible = self.table is not None and t is not None
        for region, val in self._regions:
            region.setVisible(visible)
        if not visible:
            self._update_split_region()
            return
        tab = self.table
        for kind, plot, legend, chans in (("u", self.u_plot, self.u_legend, self.inputs),
                                          ("y", self.y_plot, self.y_legend, self.outputs)):
            for i in chans:
                color = self._chan_colors.get((kind, i))
                if color is None:
                    continue
                item = plot.plot(t, tab.X[:, i], pen=pen(color, 1.4), connect="finite")
                unit = tab.units[i]
                name = tab.names[i]
                self._data_curves.append(Trace(item, name, color, self._describe_time(unit), extra={"plot": plot}))
                self._add_legend(legend, color, tab.channel_text(i))
            plot.getPlotItem().setLabel("left", "u" if kind == "u" else "y")
        self._update_split_region()

    @staticmethod
    def _describe_time(unit: str):
        return lambda x, y: [f"t = {fmt(x, 's')}", f"değer = {fmt(y, unit)}"]

    @staticmethod
    def _clear_legend(legend: _Legend) -> None:
        legend.set_items([])

    @staticmethod
    def _add_legend(legend: _Legend, color: str, text: str) -> None:
        legend.add(color, text)

    def legend_texts(self, which: str) -> list[str]:
        """Lejant metinleri (testler için): 'u', 'y', 'compare', 'auto', 'cross', 'mag', 'phase'."""
        return {"u": self.u_legend, "y": self.y_legend, "compare": self.cmp_legend, "auto": self.auto_legend,
                "cross": self.cross_legend, "mag": self.mag_legend, "phase": self.phase_legend}[which].texts()

    # ================================================================ çizim: sonuçlar
    def _fill_io_combos(self) -> None:
        ctx = self.context
        for box, names in ((self.cmp_output, ctx.names["y"]), (self.res_output, ctx.names["y"]),
                           (self.res_input, ctx.names["u"]), (self.freq_output, ctx.names["y"]),
                           (self.freq_input, ctx.names["u"])):
            cur = box.currentText()
            box.blockSignals(True)
            box.clear()
            box.addItems(names)
            i = box.findText(cur)
            box.setCurrentIndex(i if i >= 0 else 0)
            box.setVisible(len(names) > 1)
            box.blockSignals(False)

    def redraw_results(self) -> None:
        self._draw_compare()
        self._draw_resid()
        self._draw_freq()

    def _shown(self) -> list[ModelEntry]:
        return [e for e in self.entries if e.visible and e.ev is not None]

    def _output_color(self, name: str) -> str:
        """Ölçülen çıkışın kanal rengi (veri grafiğindeki ile aynı seri)."""
        if self.table is not None and name in self.table.names:
            c = self._chan_colors.get(("y", self.table.names.index(name)))
            if c:
                return c
        return CHANNEL_COLORS[0]

    def _draw_compare(self) -> None:
        plot = self.cmp_plot
        plot.clear()
        self._clear_legend(self.cmp_legend)
        self._cmp_traces = []
        ctx = self.context
        if ctx is None:
            self.cmp_note.setText("Model tahmin edildiğinde doğrulama verisi ve model benzetimleri burada görünür.")
            return
        i = max(self.cmp_output.currentIndex(), 0)
        val = ctx.val
        t = val.time
        yname = ctx.names["y"][i]
        color = self._output_color(yname)
        item = plot.plot(t, val.y[:, i], pen=pen(color, MEASURED_WIDTH))
        unit = val.output_units[i] if i < len(val.output_units) else ""
        self._cmp_traces.append(Trace(item, f"Ölçülen {yname}", color, self._describe_time(unit)))
        self._add_legend(self.cmp_legend, color, f"Ölçülen {yname}")
        skipped = []
        for e in self._shown():
            ev = e.ev
            if ev.yhat_val is None or len(ev.yhat_val) != len(t):
                if e.method not in J.NONPARAMETRIC:
                    skipped.append(e.name)
                continue
            item = plot.plot(t, ev.yhat_val[:, i], pen=pen(e.color, MODEL_WIDTH))
            text = f"{e.name}: %{ev.fit_val[i]:.2f}"
            self._cmp_traces.append(Trace(item, text, e.color, self._describe_time(unit)))
            self._add_legend(self.cmp_legend, e.color, text)
        note = (f"Doğrulama verisi: {sum(val.lengths)} örnek, Ts {val.Ts:g} s. Uyum % = 100·(1 − ‖y − ŷ‖/‖y − ȳ‖), "
                "benzetim (k = ∞), başlangıç durumu en iyi uyacak şekilde tahmin edilir.")
        if skipped:
            note += f" Gösterilemeyen: {', '.join(skipped)} (bu veriyle değerlendirilemedi; tablodaki ipucuna bakın)."
        self.cmp_note.setText(note)
        plot.getPlotItem().setLabel("left", yname, units=unit or None)

    def _draw_resid(self) -> None:
        for plot, legend in ((self.auto_plot, self.auto_legend), (self.cross_plot, self.cross_legend)):
            plot.clear()
            self._clear_legend(legend)
        self._res_traces = {"auto": [], "cross": []}
        ctx = self.context
        if ctx is None:
            return
        i = max(self.res_output.currentIndex(), 0)
        j = max(self.res_input.currentIndex(), 0)
        muted = self._theme.tokens.text_muted
        conf_drawn = False
        for e in self._shown():
            r = e.ev.resid
            if r is None:
                continue
            item = self.auto_plot.plot(r.lags_auto, r.auto[:, i], pen=pen(e.color, MODEL_WIDTH), symbol="o",
                                       symbolSize=5, symbolBrush=e.color, symbolPen=None)
            self._res_traces["auto"].append(Trace(item, e.name, e.color, self._describe_lag("r_e")))
            self._add_legend(self.auto_legend, e.color, e.name)
            if not conf_drawn:
                c = float(r.conf_auto[i])
                for v in (c, -c):
                    self.auto_plot.addItem(pg.InfiniteLine(v, angle=0, pen=pg.mkPen(muted, width=1,
                                                                                    style=Qt.PenStyle.DashLine)))
                conf_drawn = True
            if r.cross.shape[2] > j:
                item = self.cross_plot.plot(r.lags_cross, r.cross[:, i, j], pen=pen(e.color, MODEL_WIDTH),
                                            symbol="o", symbolSize=5, symbolBrush=e.color, symbolPen=None)
                self._res_traces["cross"].append(Trace(item, e.name, e.color, self._describe_lag("r_ue")))
                c = float(r.conf_cross[i, j])
                for v in (c, -c):
                    self.cross_plot.addItem(pg.InfiniteLine(v, angle=0, pen=pg.mkPen(e.color, width=1,
                                                                                     style=Qt.PenStyle.DashLine)))
                self._add_legend(self.cross_legend, e.color, e.name)
        if conf_drawn:
            self._add_legend(self.auto_legend, muted, "%99 güven sınırı (kesikli)")

    @staticmethod
    def _describe_lag(what: str):
        return lambda x, y: [f"gecikme = {int(round(x))}", f"{what} = {y:.4g}"]

    def _draw_freq(self) -> None:
        for plot, legend in ((self.mag_plot, self.mag_legend), (self.phase_plot, self.phase_legend)):
            plot.clear()
            self._clear_legend(legend)
        self._freq_traces = {"mag": [], "phase": []}
        ctx = self.context
        hz = self.freq_unit.currentIndex() == 1
        unit = "Hz" if hz else "rad/s"
        for p in (self.mag_plot, self.phase_plot):
            p.getPlotItem().setLabel("bottom", f"Frekans ({unit})")
        if ctx is None:
            return
        i = max(self.freq_output.currentIndex(), 0)
        j = max(self.freq_input.currentIndex(), 0)
        series = []
        if ctx.spa is not None:
            series.append(("spa (çalışma verisi)", SPA_COLOR, ctx.spa[0], ctx.spa[1], MEASURED_WIDTH))
        for e in self._shown():
            if e.ev.H is not None and e.ev.w is not None:
                series.append((e.name, e.color, e.ev.w, e.ev.H, MODEL_WIDTH))
        for name, color, w, H, width in series:
            if H.shape[0] <= i or H.shape[1] <= j:
                continue
            h = H[i, j]
            ok = (w > 0) & np.isfinite(h)
            x = w[ok] / (2 * np.pi) if hz else w[ok]
            mag = 20 * np.log10(np.maximum(np.abs(h[ok]), 1e-300))
            ph = np.degrees(np.unwrap(np.angle(h[ok])))
            it1 = self.mag_plot.plot(x, mag, pen=pen(color, width))
            it2 = self.phase_plot.plot(x, ph, pen=pen(color, width))
            self._freq_traces["mag"].append(Trace(it1, name, color, self._describe_freq(unit, "dB")))
            self._freq_traces["phase"].append(Trace(it2, name, color, self._describe_freq(unit, "°")))
            self._add_legend(self.mag_legend, color, name)
            self._add_legend(self.phase_legend, color, name)
        tip = self._freq_tip + (f"\nspa: {ctx.spa_error}" if ctx.spa_error else "")
        self.mag_card.title.setToolTip(tip)

    @staticmethod
    def _describe_freq(unit: str, what: str):
        return lambda x, y: [f"f = {x:.5g} {unit}", f"{y:.4g} {what}"]

    # ================================================================ durum
    def _update_buttons(self) -> None:
        busy = self.busy
        self.estimate_button.setEnabled(not busy and self.table is not None)
        self.cancel_button.setEnabled(busy)
        self.suggest_button.setEnabled(not busy and self.table is not None)
        has = bool(self.entries)
        self.validate_button.setEnabled(not busy and has and self.table is not None)
        self.delete_button.setEnabled(not busy and has)
        self.add_button.setEnabled(has)

    def _set_status(self, text: str, state: str = "") -> None:
        t = self._theme.tokens
        color = {"good": t.good, "warning": t.warning, "critical": t.critical, "busy": t.accent}.get(state,
                                                                                                    t.text_muted)
        safe = text.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        self.status.setText(f'<span style="color:{color}">●</span>&nbsp;{safe}' if text else "")
        self.status_text = text
        self.status_state = state

    def shutdown(self) -> None:
        """Yardımcı hesap sürecini kapatır (görünüm kapanırken)."""
        self.runner.shutdown()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if not self._warmed:  # yardımcı süreci önceden aç: ilk tahminde içe aktarma beklenmesin
            self._warmed = True
            QTimer.singleShot(0, self.runner.warm)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.shutdown()
        super().closeEvent(event)
