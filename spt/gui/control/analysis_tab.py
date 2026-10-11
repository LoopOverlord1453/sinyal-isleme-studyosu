"""Analiz sekmesi (MATLAB Linear System Analyzer / Control System Designer'ın analiz bölümü).

Çevrim yapısı (tek serbestlik dereceli, ön filtreli; MATLAB'in standart yapısı):

    r ─► F ─►(+)─► C ─► G ─┬─► y          u = C·(F·r − H·y),   y = G·u
               ▲ −          │
               └──── H ◄────┘

    L  = H·G·C                      açık çevrim (çevrim kazancı; margin, Nyquist, Nichols, kök yer eğrisi)
    T  = (I + G·C·H)⁻¹·G·C·F        r → y (kapalı çevrim)                       = feedback(G·C, H)·F
    S  = (I + L)⁻¹                  duyarlılık (çıkış bozucusu → hata)          = feedback(I, L)
    KS = C·S                        r → u (F = 1; kontrol çabası)
    GS = (I + G·C·H)⁻¹·G            giriş bozucusu → y                          = feedback(G, C·H)

Seçilmeyen C, H, F birim kazançtır. Zaman alanı: modellerin kendi zamanı, ayrık (her bileşen c2d / d2d ile aynı Ts'ye)
ya da sürekli (d2c). Bileşimler kütüphanenin `feedback`'iyle yapılır; hiçbir kontrol hesabı burada yazılmaz.

Grafik ızgarası seçilebilir (adım, darbe, Bode, Nyquist, Nichols, kök yer eğrisi, kutup-sıfır, tekil değer, lsim);
her grafik ayrı kartta, birden çok sistem üst üste ayrı renklerle (aynı görünümde bir renk yalnız bir seride). Açık
çevrim grafikleri (Nyquist, Nichols, kök yer eğrisi) L'yi ve üst üste eklenen modelleri (açık çevrim olarak) çizer.
Pay tablosu (margin, MATLAB tanımı) ve stepinfo tablosu altta.
"""
from __future__ import annotations

import logging
import math

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDoubleSpinBox,
    QGridLayout,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from spt.control.errors import ControlError
from spt.gui.control import control_plots as cp
from spt.gui.control.models_tab import C2D_METHODS, D2C_METHODS, compact_columns
from spt.gui.control.workspace import ModelWorkspace
from spt.gui.icons import line_swatch
from spt.gui.theme import ThemeManager
from spt.gui.widgets.common import Card, CollapsibleSection, FlowLayout, combo, label

log = logging.getLogger("spt.gui")

NONE = "— (birim kazanç)"
LOOP_SERIES = (  # anahtar, kısa ad, açıklama
    ("L", "L = H·G·C", "açık çevrim (frekans, Nyquist, Nichols, kök yer eğrisi, kutup-sıfır grafiklerinde)"),
    ("T", "T: r → y", "kapalı çevrim"),
    ("S", "S = 1/(1+L)", "duyarlılık"),
    ("KS", "KS = C·S", "r → u, kontrol çabası"),
    ("GS", "GS = G·S", "giriş bozucusu → y"),
)
OPEN_LOOP_PLOTS = ("nyquist", "nichols", "rlocus")
TIME_PLOTS = ("step", "impulse")
PLOT_ORDER = ("step", "impulse", "bode", "nyquist", "nichols", "rlocus", "pzmap", "sigma", "lsim")
DEFAULT_PLOTS = ("step", "bode", "nyquist", "pzmap")
DOMAINS = (("own", "Modellerin kendi zamanı"), ("discrete", "Ayrık (c2d)"), ("continuous", "Sürekli (d2c)"))
LSIM_SIGNALS = (("square", "Kare dalga"), ("sine", "Sinüs"), ("step", "Adım"), ("pulse", "Darbe (tek)"))
MARKERS = (("peak", "Tepe (aşım)"), ("rise", "Yükselme"), ("settling", "Yerleşme"), ("steady", "Kalıcı durum"))
MARGIN_COLUMNS = (("Sistem", "Açık çevrim (L ya da üst üste eklenen model)"),
                  ("Gm\n(dB)", "Kazanç payı: fazın −180° geçtiği frekansta 1/|L| (MATLAB margin; mutlak değer "
                               "hücre ipucunda)"),
                  ("ω Gm\n(rad/s)", "Faz geçiş frekansı (−180°)"),
                  ("Pm\n(°)", "Faz payı: |L| = 1 frekansında 180° + ∠L"),
                  ("ω Pm\n(rad/s)", "Kazanç geçiş frekansı (|L| = 1)"),
                  ("Kapalı\nçevrim", "Birim negatif geri beslemeli kapalı çevrimin kararlılığı (allmargin Stable)"))
STEP_COLUMNS = (("Sistem", "Kapalı çevrim cevabı ya da model"),
                ("tr\n(s)", "Yükselme süresi tr: son değerin %10'undan %90'ına (yinit = 0)"),
                ("ts\n(s)", "Yerleşme süresi ts: cevabın son değerin ±%2 bandında kaldığı an"),
                ("Aşım\n(%)", "Aşım: tepe değerin son değeri yüzde olarak aşması"),
                ("Tepe", "En büyük mutlak değer"),
                ("tp\n(s)", "Tepe zamanı tp (tekdüze cevapta ∞: tepe sonsuzdaki sınır)"),
                ("Kalıcı\ndurum", "Kalıcı durum değeri (DC kazancı)"),
                ("BG\n(rad/s)", "Bant genişliği: kazancın DC değerinin 3 dB altına indiği ilk frekans"))
STEP_DIGITS = 4   # adım tablosunda anlamlı basamak (1366 genişlikte sekiz sütun sığsın); ipucunda tam değer
UNDEFINED_STEPINFO = ("Son değer 0 (yfinal = yinit): yükselme, yerleşme ve aşım son değere göre tanımsız "
                      "(MATLAB stepinfo bu durumda aşımı ∞, yerleşmeyi NaN verir).")


def _num(x: float, digits: int = 5) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if math.isinf(x):
        return "∞" if x > 0 else "−∞"
    return f"{x:.{digits}g}"


class AnalysisTab(QWidget):
    computed = pyqtSignal()

    def __init__(self, theme: ThemeManager, workspace: ModelWorkspace, parent: QWidget | None = None):
        super().__init__(parent)
        self._theme = theme
        self.workspace = workspace
        self._sources: list[tuple[str, str, object]] = []
        self.systems: dict[str, object] = {}     # anahtar → LTI (son hesap)
        self.series: list[cp.Series] = []
        self.messages: list[str] = []
        self.margin_values: dict[str, tuple[float, float, float, float]] = {}   # anahtar → MATLAB margin çıktısı
        self.step_values: dict[str, dict] = {}                                  # anahtar → stepinfo
        self._measured_color: str | None = None     # lsim ölçülen çıkışının rengi (görünümde benzersiz)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self.recompute)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        split.addWidget(self._build_setup())
        split.addWidget(self._build_right())
        split.setStretchFactor(1, 1)
        split.setSizes([340, 1000])
        lay.addWidget(split)
        for sig in (workspace.added, workspace.removed, workspace.changed):
            sig.connect(self._on_workspace)
        workspace.renamed.connect(self._on_renamed)
        self._fill_model_choices()
        self._layout_plots()

    # ================================================================ kurulum paneli
    def _build_setup(self) -> Card:
        card = Card("Analiz Kurulumu")
        card.setMinimumWidth(300)
        card.setMaximumWidth(440)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        host = QWidget()
        col = QVBoxLayout(host)
        col.setContentsMargins(0, 0, 6, 0)
        col.setSpacing(6)
        scroll.setWidget(host)
        card.body.addWidget(scroll, 1)

        loop = CollapsibleSection("Çevrim yapısı")
        diagram = label("r → F → (+) → C → G → y<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
                        "&nbsp;&nbsp;− ↑&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;H ←┘", "Muted")
        diagram.setToolTip("u = C·(F·r − H·y), y = G·u — seçilmeyen C, H, F birim kazançtır")
        loop.form.addRow(diagram)
        self.boxes: dict[str, object] = {}
        for key, text, tip in (("G", "Bitki G", "Kontrol edilen sistem"),
                               ("C", "Kontrolcü C", "Seri kompansatör / kontrolcü"),
                               ("H", "Algılayıcı H", "Geri besleme yolu (ölçüm dinamiği)"),
                               ("F", "Ön filtre F", "Referans ön filtresi")):
            box = combo([])
            box.setToolTip(tip)
            box.currentIndexChanged.connect(self.schedule)
            self.boxes[key] = box
            loop.add_row(text, box)
        col.addWidget(loop)

        shown = CollapsibleSection("Gösterilecek cevaplar")
        self.loop_checks: dict[str, QCheckBox] = {}
        holder = QWidget()
        grid = FlowLayout(holder, spacing=6, uniform=True)
        for key, text, tip in LOOP_SERIES:
            box = QCheckBox(text)
            box.setChecked(key in ("L", "T"))
            box.setToolTip(f"{text} — {tip}")
            box.toggled.connect(self._sync_swatches)
            box.toggled.connect(self.schedule)
            self.loop_checks[key] = box
            grid.addWidget(box)
        self._sync_swatches()
        shown.form.addRow(holder)
        shown.form.addRow(label("Üst üste eklenecek modeller", "FormLabel"))
        self.extra_list = QListWidget()
        self.extra_list.setToolTip("İşaretlenen modeller de çizilir (açık çevrim grafiklerinde açık çevrim olarak)")
        self.extra_list.setMaximumHeight(130)
        self.extra_list.itemChanged.connect(self.schedule)
        shown.form.addRow(self.extra_list)
        col.addWidget(shown)

        dom = CollapsibleSection("Zaman alanı")
        self.domain = combo([])
        for key, text in DOMAINS:
            self.domain.addItem(text, key)
        self.domain.setToolTip("Bileşenler birleştirilmeden önce bu zamana dönüştürülür (farklı Ts'li modeller "
                               "doğrudan birleştirilemez)")
        self.domain.currentIndexChanged.connect(self._sync_domain)
        dom.add_row("Görünüm", self.domain)
        self.ts = QDoubleSpinBox()
        self.ts.setRange(1e-6, 1e4)
        self.ts.setDecimals(6)
        self.ts.setValue(0.01)
        self.ts.setSuffix(" s")
        self.ts.setKeyboardTracking(False)
        self.ts.valueChanged.connect(self.schedule)
        dom.add_row("Ts", self.ts)
        self.method = combo([])
        for key, text in C2D_METHODS:
            self.method.addItem(text, key)
        self.method.currentIndexChanged.connect(self.schedule)
        dom.add_row("Yöntem", self.method)
        col.addWidget(dom)

        plots = CollapsibleSection("Grafikler")
        holder = QWidget()
        grid = FlowLayout(holder, spacing=6, uniform=True)
        self.plot_checks: dict[str, QCheckBox] = {}
        for key in PLOT_ORDER:
            box = QCheckBox(cp.PLOT_TYPES[key][0])
            box.setChecked(key in DEFAULT_PLOTS)
            box.toggled.connect(self._on_plot_toggled)
            self.plot_checks[key] = box
            grid.addWidget(box)
        plots.form.addRow(holder)
        self.columns = combo(["Otomatik", "1", "2", "3"])
        self.columns.setToolTip("Grafik ızgarasının sütun sayısı")
        self.columns.currentIndexChanged.connect(lambda *_: self._layout_plots())
        plots.add_row("Sütun", self.columns)
        col.addWidget(plots)

        marks = CollapsibleSection("Karakteristik işaretleri (adım)", expanded=False)
        holder = QWidget()
        grid = FlowLayout(holder, spacing=6, uniform=True)
        self.marker_checks: dict[str, QCheckBox] = {}
        for key, text in MARKERS:
            box = QCheckBox(text)
            box.setChecked(True)
            box.toggled.connect(self.schedule)
            self.marker_checks[key] = box
            grid.addWidget(box)
        marks.form.addRow(holder)
        col.addWidget(marks)

        lsim = CollapsibleSection("Benzetim (lsim) girişi", expanded=False)
        self.lsim_source = combo([])
        self.lsim_source.currentIndexChanged.connect(self._on_lsim_source)
        lsim.add_row("Giriş", self.lsim_source)
        self.lsim_channel = combo([])
        self.lsim_channel.setToolTip("Giriş olarak kullanılacak kanal (fiziksel birimde)")
        self.lsim_channel.currentIndexChanged.connect(self.schedule)
        lsim.add_row("Kanal", self.lsim_channel)
        self.lsim_measured = combo([])
        self.lsim_measured.setToolTip("Karşılaştırma için ölçülen çıkış kanalı (isteğe bağlı)")
        self.lsim_measured.currentIndexChanged.connect(self.schedule)
        lsim.add_row("Ölçülen çıkış", self.lsim_measured)
        self.lsim_period = self._spin(0.001, 1e5, 4.0, " s", "Kare dalga / sinüs periyodu")
        lsim.add_row("Periyot", self.lsim_period)
        self.lsim_duration = self._spin(0.001, 1e6, 10.0, " s", "Benzetim süresi")
        lsim.add_row("Süre", self.lsim_duration)
        self.lsim_amp = self._spin(-1e6, 1e6, 1.0, "", "Genlik")
        lsim.add_row("Genlik", self.lsim_amp)
        col.addWidget(lsim)
        self._lsim_rows = (self.lsim_channel, self.lsim_measured, self.lsim_period, self.lsim_duration,
                           self.lsim_amp)
        col.addStretch(1)

        self.status = label("", "Readout")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.RichText)
        card.body.addWidget(self.status)
        self._fill_lsim_sources()
        self._sync_domain()
        return card

    def _sync_swatches(self, *_) -> None:
        """Renk örneği yalnız seçili çevrim cevabında: seçilmeyenin rengi üst üste eklenen bir modele verilebilir,
        görünümde aynı renk iki şeyi göstermesin."""
        for i, (key, _text, _tip) in enumerate(LOOP_SERIES):
            box = self.loop_checks[key]
            box.setIcon(line_swatch(cp.PALETTE[i] if box.isChecked() else "#00000000", Qt.PenStyle.SolidLine, 3))

    def _spin(self, lo, hi, value, suffix, tip) -> QDoubleSpinBox:
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setDecimals(4)
        w.setValue(value)
        w.setSuffix(suffix)
        w.setKeyboardTracking(False)
        w.setToolTip(tip)
        w.valueChanged.connect(self.schedule)
        return w

    def _sync_domain(self, *_) -> None:
        mode = self.domain.currentData()
        self.ts.setEnabled(mode == "discrete")
        self.method.setEnabled(mode != "own")
        self.schedule()

    # ================================================================ sağ: grafik ızgarası ve tablolar
    def _build_right(self) -> QWidget:
        t = self._theme
        vsplit = QSplitter(Qt.Orientation.Vertical)
        vsplit.setChildrenCollapsible(False)
        top = QWidget()
        tl = QVBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(6)
        self.legend = cp.SeriesLegend()
        tl.addWidget(self.legend)
        self._plots_scroll = QScrollArea()
        self._plots_scroll.setWidgetResizable(True)
        self._plots_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._plots_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._grid_host = QWidget()
        self._grid = QGridLayout(self._grid_host)
        self._grid.setContentsMargins(0, 0, 4, 0)
        self._grid.setSpacing(12)
        self._plots_scroll.setWidget(self._grid_host)
        tl.addWidget(self._plots_scroll, 1)
        vsplit.addWidget(top)

        self.plots: dict[str, QWidget] = {}
        self._cells: dict[str, QWidget] = {}   # ızgara hücreleri; Bode iki hücre (genlik, faz)

        def cell_of(card: QWidget) -> QWidget:
            cell = QWidget()  # kart ayrık pencereye alınınca yer tutucu bu hücrede kalır; ızgara hücreyi taşır
            cl = QVBoxLayout(cell)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.addWidget(card)
            cell.setMinimumHeight(cp.CARD_MIN_HEIGHT)
            return cell

        for key in PLOT_ORDER:
            widget = cp.make_response_plot(key, t)
            self.plots[key] = widget
            if isinstance(widget, cp.BodePlot):
                mag, phase = widget.release_cards()
                widget.setParent(top)
                self._cells["bode_mag"] = cell_of(mag)
                self._cells["bode_phase"] = cell_of(phase)
            else:
                self._cells[key] = cell_of(widget)
        bottom = QSplitter(Qt.Orientation.Horizontal)
        bottom.setChildrenCollapsible(False)
        self.margin_card = Card("Kararlılık Payları")
        self.margin_card.add_right(label("margin · açık çevrim", "Readout"), optional=True)
        self.margin_table = self._table(MARGIN_COLUMNS)
        self.margin_card.body.addWidget(self.margin_table, 1)
        self.step_card = Card("Adım Cevabı Ölçütleri")
        self.step_card.add_right(label("stepinfo · %2 yerleşme, %10–%90 yükselme", "Readout"), optional=True)
        self.step_table = self._table(STEP_COLUMNS)
        self.step_card.body.addWidget(self.step_table, 1)
        bottom.addWidget(self.margin_card)
        bottom.addWidget(self.step_card)
        bottom.setSizes([430, 570])
        vsplit.addWidget(bottom)
        vsplit.setStretchFactor(0, 1)
        vsplit.setSizes([600, 190])
        return vsplit

    def _table(self, columns) -> QTableWidget:
        tbl = QTableWidget(0, len(columns))
        tbl.setHorizontalHeaderLabels([c for c, _ in columns])
        for i, (_c, tip) in enumerate(columns):
            tbl.horizontalHeaderItem(i).setToolTip(tip)
        tbl.verticalHeader().setVisible(False)
        tbl.setShowGrid(True)
        tbl.setAlternatingRowColors(True)
        tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        tbl.setSelectionMode(QAbstractItemView.SelectionMode.ContiguousSelection)
        compact_columns(tbl)   # iki satırlı kısa başlıklarla 1366 genişlikte de yatay kaydırma ve kısaltma yok
        tbl.verticalHeader().setDefaultSectionSize(24)
        tbl.setMinimumHeight(96)   # iki satırlık başlık + en az iki sistem satırı
        return tbl

    def detachable_cards(self) -> list[tuple[str, str, Card]]:
        out = []
        for key in PLOT_ORDER:
            w = self.plots[key]
            name = cp.PLOT_TYPES[key][0]
            if isinstance(w, cp.BodePlot):
                out += [("ctl_an_bode_mag", "Analiz · Bode Genlik", w.mag_card),
                        ("ctl_an_bode_phase", "Analiz · Bode Faz", w.phase_card)]
            else:
                out.append((f"ctl_an_{key}", f"Analiz · {name}", w))
        out += [("ctl_an_margin", "Analiz · Kararlılık Payları", self.margin_card),
                ("ctl_an_stepinfo", "Analiz · Adım Cevabı Ölçütleri", self.step_card)]
        return out

    def selected_plots(self) -> list[str]:
        return [k for k in PLOT_ORDER if self.plot_checks[k].isChecked()]

    def _on_plot_toggled(self, *_) -> None:
        self._layout_plots()
        self.schedule()

    def _cell_keys(self) -> list[str]:
        out = []
        for key in self.selected_plots():
            out += ["bode_mag", "bode_phase"] if key == "bode" else [key]
        return out

    def _layout_plots(self) -> None:
        """Seçili grafik hücrelerini ızgaraya dizer (Bode genlik ve faz yan yana iki hücre). Otomatik sütun: hücre
        sayısına göre 1–3; satırlar en az CARD_MIN_HEIGHT, sığmazsa alan kaydırılır."""
        keys = self._cell_keys()
        for i in reversed(range(self._grid.count())):
            self._grid.takeAt(i)
        text = self.columns.currentText()
        n = len(keys)
        ncol = int(text) if text.isdigit() else (1 if n <= 1 else 2 if n in (2, 4) else 3)
        r = c = 0
        for key, cell in self._cells.items():
            if key not in keys:
                cell.hide()
        for key in keys:
            cell = self._cells[key]
            self._grid.addWidget(cell, r, c)
            cell.show()
            c += 1
            if c >= ncol:
                r, c = r + 1, 0
        for i in range(4):
            self._grid.setColumnStretch(i, 1 if i < ncol else 0)

    # ================================================================ model seçimleri
    def _fill_model_choices(self) -> None:
        names = self.workspace.lti_names()
        for key, box in self.boxes.items():
            cur = box.currentData()
            box.blockSignals(True)
            box.clear()
            if key != "G":
                box.addItem(NONE, None)
            else:
                box.addItem("— seçilmedi —", None)
            for n in names:
                box.addItem(n, n)
            i = box.findData(cur)
            if i < 0 and key == "G" and names:
                i = 1  # ilk model bitki olarak önerilir
            box.setCurrentIndex(max(i, 0))
            box.blockSignals(False)
        checked = self.extra_models()
        self.extra_list.blockSignals(True)
        self.extra_list.clear()
        for n in names:
            item = QListWidgetItem(n)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if n in checked else Qt.CheckState.Unchecked)
            self.extra_list.addItem(item)
        self.extra_list.blockSignals(False)

    def extra_models(self) -> list[str]:
        return [self.extra_list.item(i).text() for i in range(self.extra_list.count())
                if self.extra_list.item(i).checkState() == Qt.CheckState.Checked]

    def set_extra_models(self, names: list[str]) -> None:
        self.extra_list.blockSignals(True)
        for i in range(self.extra_list.count()):
            item = self.extra_list.item(i)
            item.setCheckState(Qt.CheckState.Checked if item.text() in names else Qt.CheckState.Unchecked)
        self.extra_list.blockSignals(False)
        self.schedule()

    def set_loop(self, G: str | None = None, C: str | None = None, H: str | None = None,
                 F: str | None = None) -> None:
        """Çevrim bileşenlerini adlarıyla seç (None: değiştirme; "" : birim kazanç / seçilmedi)."""
        for key, name in (("G", G), ("C", C), ("H", H), ("F", F)):
            if name is None:
                continue
            box = self.boxes[key]
            i = box.findData(name or None)
            if i >= 0:
                box.setCurrentIndex(i)
        self.schedule()

    def set_shown(self, keys) -> None:
        for key, box in self.loop_checks.items():
            box.setChecked(key in keys)

    def set_plots(self, keys) -> None:
        for key, box in self.plot_checks.items():
            box.blockSignals(True)
            box.setChecked(key in keys)
            box.blockSignals(False)
        self._layout_plots()
        self.schedule()

    def _on_workspace(self, *_) -> None:
        self._fill_model_choices()
        self.schedule()

    def _on_renamed(self, old: str, new: str) -> None:
        for box in self.boxes.values():
            if box.currentData() == old:
                box.blockSignals(True)
                box.setItemData(box.currentIndex(), new)
                box.blockSignals(False)
        for i in range(self.extra_list.count()):
            if self.extra_list.item(i).text() == old:
                self.extra_list.item(i).setText(new)
        self._fill_model_choices()
        self.schedule()

    # ================================================================ lsim girişi
    def set_sources(self, sources: list[tuple[str, str, object]]) -> None:
        """Gerçek veri kaynakları (kimlik, etiket, tutamaç): akışlar ve Çalışma Alanı anlık görüntüleri."""
        self._sources = list(sources)
        self._fill_lsim_sources()

    def _fill_lsim_sources(self) -> None:
        cur = self.lsim_source.currentData()
        box = self.lsim_source
        box.blockSignals(True)
        box.clear()
        for key, text in LSIM_SIGNALS:
            box.addItem(text, key)
        for sid, text, _h in self._sources:
            box.addItem(text, f"src:{sid}")
        i = box.findData(cur)
        box.setCurrentIndex(max(i, 0))
        box.blockSignals(False)
        self._on_lsim_source()

    def _handle(self):
        data = self.lsim_source.currentData() or ""
        if not str(data).startswith("src:"):
            return None
        sid = data[4:]
        for key, _t, h in self._sources:
            if key == sid:
                return h
        return None

    def _on_lsim_source(self, *_) -> None:
        h = self._handle()
        for box in (self.lsim_channel, self.lsim_measured):
            cur = box.currentData()
            box.blockSignals(True)
            box.clear()
            if box is self.lsim_measured:
                box.addItem("— yok —", None)
            if h is not None:
                for i, ch in enumerate(h.desc.channels):
                    box.addItem(f"{ch.name or f'kanal {i}'}" + (f" [{ch.unit}]" if ch.unit else ""), i)
            i = box.findData(cur)
            box.setCurrentIndex(max(i, 0))
            box.blockSignals(False)
        data = h is not None
        self.lsim_channel.setEnabled(data)
        self.lsim_measured.setEnabled(data)
        kind = self.lsim_source.currentData()
        self.lsim_period.setEnabled(not data and kind in ("square", "sine"))
        self.lsim_amp.setEnabled(not data)
        self.lsim_duration.setEnabled(not data)
        self.schedule()

    def lsim_input(self, dt_hint: float | None) -> tuple[np.ndarray, np.ndarray, str, tuple | None]:
        """(t, u, ad, ölçülen) — üretilen sinyal ya da kaynaktaki kanal (fiziksel birim)."""
        h = self._handle()
        if h is not None:
            from spt.control.sysid.data import iddata_from_handle
            ch = self.lsim_channel.currentData()
            meas = self.lsim_measured.currentData()
            if ch is None:
                raise ControlError("Giriş kanalı seçin")
            if meas is not None and meas != ch:
                d = iddata_from_handle(h, inputs=[ch], outputs=[meas])
                y = np.asarray(d.y, float)[:, 0]
                u = np.asarray(d.u, float)[:, 0]
            else:
                d = iddata_from_handle(h, outputs=[ch])
                u = np.asarray(d.y, float)[:, 0]
                y = None
            n = len(u)
            if n < 2:
                raise ControlError("Kaynakta yeterli veri yok")
            if not np.all(np.isfinite(u)):
                raise ControlError("Giriş kanalında kayıp (eksik) örnekler var; eksiksiz bir anlık görüntü seçin")
            t = np.arange(n) * d.Ts
            name = f"giriş: {self.lsim_channel.currentText()}"
            measured = (f"ölçülen: {self.lsim_measured.currentText()}", y) if y is not None else None
            return t, u, name, measured
        kind = self.lsim_source.currentData()
        T, P, A = self.lsim_duration.value(), self.lsim_period.value(), self.lsim_amp.value()
        dt = dt_hint if dt_hint else min(P / 200.0, T / 2000.0)
        n = int(math.floor(T / dt + 1e-9)) + 1
        t = np.arange(n) * dt
        if kind == "square":  # MATLAB gensig('square'): ilk yarı periyot 1, ikinci yarı 0
            u = A * (np.mod(t, P) < P / 2).astype(float)
        elif kind == "sine":
            u = A * np.sin(2 * np.pi * t / P)
        elif kind == "pulse":
            u = np.zeros(n)
            u[0] = A
        else:
            u = np.full(n, A)
        return t, u, f"giriş: {self.lsim_source.currentText()}", None

    # ================================================================ hesap
    def schedule(self, *_) -> None:
        self._timer.start()

    def _component(self, key: str):
        name = self.boxes[key].currentData()
        if name is None:
            return None
        return self.workspace.lti(name)

    def _to_domain(self, sys):
        if sys is None:
            return None
        mode = self.domain.currentData()
        method = self.method.currentData()
        if mode == "discrete":
            from spt.control.discretize import c2d, d2d
            Ts = self.ts.value()
            if sys.Ts == 0:
                return c2d(sys, Ts, method=method)
            if abs(sys.Ts - Ts) > 1e-12 * Ts:
                return d2d(sys, Ts, method="tustin" if method == "tustin" else "zoh")
            return sys
        if mode == "continuous" and sys.Ts > 0:
            from spt.control.discretize import d2c
            return d2c(sys, method=method if method in D2C_METHODS else "zoh")
        return sys

    def loop_systems(self) -> dict[str, object]:
        """Seçili çevrimin sistemleri (L, T, S, KS, GS); bitki seçilmemişse boş."""
        from spt.control import lti
        G = self._to_domain(self._component("G"))
        if G is None:
            return {}
        C = self._to_domain(self._component("C"))
        H = self._to_domain(self._component("H"))
        F = self._to_domain(self._component("F"))
        ny, nu = G.noutputs, G.ninputs
        kind = getattr(type(G), "_kind", "ss")
        gain = getattr(lti, kind if kind in ("tf", "zpk", "ss") else "ss")  # birim kazanç bitkiyle aynı türde

        def eye(n):
            return gain(np.eye(n), Ts=G.Ts)

        if C is None:
            if ny != nu:
                raise ControlError("Kontrolcü seçilmediğinde bitki kare olmalı (giriş sayısı = çıkış sayısı); "
                                   "MIMO bitki için bir kontrolcü seçin.")
            C = eye(ny)
        H = eye(ny) if H is None else H
        F = eye(ny) if F is None else F
        try:
            from spt.control.connect import feedback
        except ImportError:
            feedback = None
        GC = G * C
        L = H * GC
        if feedback is not None:
            T = feedback(GC, H) * F
            S = feedback(eye(L.noutputs), L)
            GS = feedback(G, C * H)
        else:  # bağlantı modülü yoksa model aritmetiğiyle (sadeleştirmesiz, aynı transfer)
            S = lti.inv(eye(L.noutputs) + L)
            T = lti.inv(eye(ny) + GC * H) * GC * F
            GS = lti.inv(eye(ny) + GC * H) * G
        KS = C * S
        return {"L": _like(L, G), "T": _like(T, G), "S": _like(S, G), "KS": _like(KS, G), "GS": _like(GS, G)}

    def recompute(self) -> None:
        """Seçimlere göre sistemleri kurar, grafikleri ve tabloları günceller."""
        self._timer.stop()
        msgs: list[str] = []
        systems: dict[str, object] = {}
        series: list[cp.Series] = []
        try:
            loop = self.loop_systems()
        except ControlError as exc:
            loop = {}
            msgs.append(f"Çevrim kurulamadı: {exc}")
        except ImportError:
            loop = {}
            msgs.append("Çevrim hesabı için gereken modüller hazırlanıyor.")
        used: set[str] = set()
        for i, (key, text, _tip) in enumerate(LOOP_SERIES):
            if key in loop and self.loop_checks[key].isChecked():
                systems[key] = loop[key]
                series.append(cp.Series(text, loop[key], cp.PALETTE[i], key))
                used.add(cp.PALETTE[i])
        free = [c for c in cp.PALETTE if c not in used and c not in cp.PALETTE[:len(LOOP_SERIES)]]
        free += [c for c in cp.PALETTE[:len(LOOP_SERIES)] if c not in used]
        # lsim'deki ölçülen çıkış da bir seridir: rengi görünümdeki hiçbir sistemle paylaşılmasın diye önce ayrılır
        self._measured_color = None
        measured_name = ""
        if "lsim" in self.selected_plots() and self._measured_selected() and free:
            self._measured_color = free.pop(0)
            measured_name = f"ölçülen: {self.lsim_measured.currentText()}"
        for name in self.extra_models():
            if not free:
                msgs.append(f"{name}: en fazla {cp.MAX_SERIES} seri üst üste çizilir (her biri ayrı renkte)")
                continue
            try:
                sys = self._to_domain(self.workspace.lti(name))
            except ControlError as exc:
                msgs.append(f"{name}: {exc}")
                continue
            systems[f"m:{name}"] = sys
            series.append(cp.Series(name, sys, free.pop(0), f"m:{name}"))
        self.systems = systems
        self.series = series
        legend = [(s.color, s.name) for s in series]
        if self._measured_color is not None:
            legend.append((self._measured_color, measured_name))
        self.legend.set_series(legend)
        open_loop = [s for s in series if s.key == "L" or s.key.startswith("m:")]
        # zaman cevaplarında açık çevrim L yok (integratörlü L'nin sınırsız adımı kapalı çevrim eğrilerini ezer;
        # MATLAB Control System Designer'da da zaman grafikleri kapalı çevrimdir)
        timed = [s for s in series if s.key != "L"]
        markers = {k for k, b in self.marker_checks.items() if b.isChecked()}
        for key in self.selected_plots():
            widget = self.plots[key]
            if key in OPEN_LOOP_PLOTS:
                errs = widget.set_series(open_loop)
            elif key == "bode":
                errs = widget.set_series(series, margins=("L",))
            elif key == "lsim":
                errs = self._draw_lsim(widget, timed)
            elif key in TIME_PLOTS:
                widget.markers = markers
                errs = widget.set_series(timed)
            else:
                errs = widget.set_series(series)
            title = cp.PLOT_TYPES[key][0]
            for e in errs:
                text = f"{title} · {e}"
                if text not in msgs:
                    msgs.append(text)
        self._fill_margin_table(open_loop)
        self._fill_step_table(timed)
        self.messages = msgs
        tok = self._theme.tokens
        if not series:
            text = "Bir bitki (G) seçin ya da üst üste çizilecek bir model işaretleyin."
            if not self.workspace.lti_names():
                text = "Çalışma alanında model yok: Modeller sekmesinden ifadeyle model ekleyin."
            self.status.setText(f'<span style="color:{tok.text_muted}">●</span> {text}')
        elif msgs:
            self.status.setText("<br>".join(f'<span style="color:{tok.warning}">⚠</span> {m}' for m in msgs[:6]))
        else:
            self.status.setText(f'<span style="color:{tok.good}">●</span> {len(series)} sistem çizildi')
        self.computed.emit()

    def _draw_lsim(self, widget: cp.LsimPlot, series: list[cp.Series]) -> list[str]:
        dts = [s.sys.Ts for s in series if s.sys.Ts > 0]
        try:
            t, u, name, measured = self.lsim_input(dts[0] if dts else None)
        except ControlError as exc:
            widget.set_input(np.zeros(0), np.zeros(0))
            widget.set_series([])
            return [f"Benzetim girişi: {exc}"]
        meas = None
        if measured is not None and self._measured_color is not None:
            meas = (measured[0], self._measured_color, measured[1])
        widget.set_input(t, u, name, meas)
        return widget.set_series(series)

    def _measured_selected(self) -> bool:
        """lsim girişi gerçek veri ve girişten farklı bir ölçülen çıkış kanalı seçili mi?"""
        meas = self.lsim_measured.currentData()
        return self._handle() is not None and meas is not None and meas != self.lsim_channel.currentData()

    def _fill_margin_table(self, open_loop: list[cp.Series]) -> None:
        tbl = self.margin_table
        tbl.setRowCount(len(open_loop))
        tok = self._theme.tokens
        self.margin_values: dict[str, tuple[float, float, float, float]] = {}
        for r, s in enumerate(open_loop):
            cells = [s.name] + ["—"] * 5
            state = None
            gm_tip = ""
            if s.sys.ninputs == 1 and s.sys.noutputs == 1:
                try:
                    gm, pm, wcg, wcp = cp.calc_margin(s.sys)
                    self.margin_values[s.key] = (gm, pm, wcg, wcp)
                    gdb = 20 * math.log10(gm) if gm > 0 else math.nan   # MATLAB: Gm mutlak; tabloda dB
                    cells[1:5] = [_num(gdb), _num(wcg), _num(pm), _num(wcp)]
                    gm_tip = f"Gm = {_num(gm, 8)} (mutlak, MATLAB margin çıktısı) = {_num(gdb, 8)} dB"
                    state = self._closed_loop_stable(s.sys)
                except (ControlError, ImportError, ValueError) as exc:
                    cells[1] = f"hesaplanamadı: {exc}" if not isinstance(exc, ImportError) else "hazırlanıyor"
            else:
                cells[1] = "yalnız tek girişli, tek çıkışlı"
            if state is not None:
                cells[5] = "kararlı" if state else "kararsız"
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c == 0:
                    item.setForeground(QColor(s.color))
                if c == 1 and gm_tip:
                    item.setToolTip(gm_tip)
                if c == 5 and state is not None:
                    item.setForeground(QColor(tok.good if state else tok.critical))
                if not item.toolTip() and len(text) > 12:   # dar sütunda kısalan uzun ileti tam okunsun
                    item.setToolTip(text)
                tbl.setItem(r, c, item)

    @staticmethod
    def _closed_loop_stable(sys) -> bool | None:
        """Birim negatif geri beslemeyle kapalı çevrimin kararlılığı (MATLAB margin / allmargin 'Stable')."""
        try:
            from spt.control.freqresp import allmargin
            info = allmargin(sys)
            stable = info.get("Stable") if isinstance(info, dict) else None
            if stable is not None:
                return bool(stable)
        except (ImportError, ControlError, ValueError, AttributeError):
            pass
        try:
            from spt.control.analysis import isstable
            from spt.control.connect import feedback
            return bool(isstable(feedback(sys, 1.0)))
        except (ImportError, ControlError, ValueError):
            return None

    def _fill_step_table(self, series: list[cp.Series]) -> None:
        rows = [s for s in series if s.sys.ninputs == 1 and s.sys.noutputs == 1]
        tbl = self.step_table
        tbl.setRowCount(len(rows))
        self.step_values: dict[str, dict] = {}
        for r, s in enumerate(rows):
            cells = [s.name] + ["—"] * 7
            raw: dict[int, float] = {}   # sütun → tam değer (ipucunda tam duyarlıkla)
            tip = ""
            dc = math.nan
            try:
                from spt.control.freqresp import bandwidth, dcgain
                dc = float(np.ravel(dcgain(s.sys))[0])
                raw[6] = dc
                raw[7] = float(np.ravel(bandwidth(s.sys))[0])
            except (ControlError, ImportError, ValueError):
                pass
            try:
                info = cp.calc_stepinfo(s.sys)
                self.step_values[s.key] = dict(info)
                if dc == 0:
                    # yfinal = yinit = 0 (ör. integral etkili çevrimde S, GS): ölçütler son değere göre tanımsız
                    tip = UNDEFINED_STEPINFO
                # Peak = max|y − yinit| (yinit = 0): sıfır olmayan cevapta 0 olamaz; böyle bir sonuç tutarsızdır ve
                # gösterilmez (yanlış sayı yerine "—")
                if not (float(info.get("Peak", math.nan)) == 0 and self._nonzero_step(s.sys)):
                    for c, k in enumerate(("RiseTime", "SettlingTime", "Overshoot", "Peak", "PeakTime"), 1):
                        raw[c] = float(info.get(k, math.nan))
            except (ControlError, ImportError, ValueError) as exc:
                cells[1] = "hazırlanıyor" if isinstance(exc, ImportError) else f"hesaplanamadı: {exc}"
            for c, v in raw.items():
                cells[c] = _num(v, STEP_DIGITS)
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c == 0:
                    item.setForeground(QColor(s.color))
                if tip and 1 <= c <= 5:
                    item.setToolTip(tip)
                elif c in raw and math.isfinite(raw[c]):
                    item.setToolTip(f"{raw[c]:.12g}")
                if not item.toolTip() and len(text) > 12:
                    item.setToolTip(text)
                tbl.setItem(r, c, item)

    @staticmethod
    def _nonzero_step(sys) -> bool:
        """Adım cevabı özdeş sıfır değil mi (sıfır sistemde ölçütler gerçekten 0'dır)?"""
        try:
            _t, y = cp.calc_step(sys)
        except (ControlError, ImportError, ValueError, np.linalg.LinAlgError):
            return True
        return bool(np.any(np.abs(y) > 0))

    def margin_rows(self) -> dict[str, list[str]]:
        """Pay tablosunun metinleri: sistem adı → hücreler (testler için)."""
        out = {}
        for r in range(self.margin_table.rowCount()):
            cells = [self.margin_table.item(r, c).text() for c in range(self.margin_table.columnCount())]
            out[cells[0]] = cells
        return out

    # ================================================================ ayarlar
    def save_state(self, s) -> None:
        s.setValue("control/analysis_plots", ",".join(self.selected_plots()))
        s.setValue("control/analysis_shown", ",".join(k for k, b in self.loop_checks.items() if b.isChecked()))
        s.setValue("control/analysis_domain", self.domain.currentData())
        s.setValue("control/analysis_ts", self.ts.value())
        s.setValue("control/analysis_method", self.method.currentData())
        s.setValue("control/analysis_columns", self.columns.currentIndex())
        s.setValue("control/analysis_loop", ",".join(b.currentData() or "" for b in self.boxes.values()))

    def restore_state(self, s) -> None:
        plots = str(s.value("control/analysis_plots", ""))
        if plots:
            self.set_plots([p for p in plots.split(",") if p in self.plot_checks])
        shown = s.value("control/analysis_shown", None)
        if shown is not None:
            self.set_shown(str(shown).split(","))
        for box, key in ((self.domain, "control/analysis_domain"), (self.method, "control/analysis_method")):
            i = box.findData(str(s.value(key, "")))
            if i >= 0:
                box.setCurrentIndex(i)
        try:
            self.ts.setValue(float(s.value("control/analysis_ts", self.ts.value())))
            self.columns.setCurrentIndex(int(s.value("control/analysis_columns", 0)))
        except (TypeError, ValueError):
            pass
        loop = str(s.value("control/analysis_loop", ""))
        if loop:
            names = (loop.split(",") + ["", "", "", ""])[:4]
            self.set_loop(*names)
        self.schedule()


def _like(sys, proto):
    """Bileşim sonucu, bitkiyle aynı biçimde (tf / zpk / ss) gösterilsin — durum uzayında hesaplanır."""
    from spt.control import lti
    kind = getattr(type(proto), "_kind", "ss")
    if kind in ("tf", "zpk") and getattr(type(sys), "_kind", "") != kind:
        try:
            return getattr(lti, kind)(sys)
        except ControlError:
            return sys
    return sys
