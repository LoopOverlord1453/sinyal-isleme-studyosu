"""Sensör Füzyonu: senaryo tabanlı kestirim ve algoritmaların karşılaştırması.

Senaryolar (`spt.dsp.fusion_generic`): IMU / MARG yönelimi, barometre + ivmeölçer irtifası, GNSS + jiroskop +
odometri 2B konumu, çok sensörlü doğrusal Kalman. Kaynak canlı bir akış (UDP / kayıt oynatma) ya da Çalışma
Alanı'ndaki dondurulmuş bir anlık görüntü olabilir; kanal eşleme kanal adlarından ve birimlerinden otomatik önerilir.
Seçilen algoritmalar ayrı süreçlerde paralel çalışır (`FusionEngine`); grafikler ve doğruluk tablosu ilerledikçe
güncellenir. Referans (gerçek) değer varsa her algoritmanın hatası işlenen tüm veri üzerinden hesaplanır.

Görünüm:
  - Her çıktı (ör. roll / pitch / yaw ya da irtifa / dikey hız) kendi kartında, kendi başlığı ve birimiyle; kartlar
    ortak zaman ekseniyle hizalıdır ve küçük ekranda kaydırılır (ezilmez).
  - Renkler her tema için ölçülerek seçildi: her algoritma ayrı bir renk ailesinden (kırmızı, sarı, yeşil,
    camgöbeği, mavi, eflatun; açık temada sarı yerine turuncu); referans metin renginde kesikli (koyu temada beyaz,
    açık temada siyah), ham (füzyonsuz) gri — iki temada da normal görüşte her çift ΔE ≥ 19, renk körlüğünde
    algoritmalar arası ΔE ≥ 9.8 (OKLab). Aynı ekranda hiçbir renk iki seride kullanılmaz.
  - Yönelimde 3B görünüm (kuaterniyon), yapay ufuk ve pusula (Euler); 2B konumda x–y yörüngesi.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QEvent, Qt, QTimer
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHeaderView,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from spt.core.fusion_engine import FusionEngine
from spt.dsp import fusion_generic as fg
from spt.gui.icons import line_swatch
from spt.gui.theme import ThemedColors, ThemedMap, ThemeManager, themed
from spt.gui.widgets.common import Card, CollapsibleSection, FlowLayout, PageHeader, chip, combo, hbox, icon_button, label
from spt.gui.widgets.inspector import DataTips, PlotAnalysis, Trace
from spt.gui.widgets.orientation3d import Body, Orientation3D, quat_from_euler_deg
from spt.gui.widgets.plot import Crosshair, make_plot, pen

log = logging.getLogger("spt.gui")

# Ölçülerek seçilen palet (bkz. modül belgesi): renk algoritmaya bağlıdır, sıraya değil
PALETTE = ThemedColors(
    dark=("#f54748", "#dfa11a", "#45f27f", "#4cd1ee", "#488bfb", "#e03fc7"),
    light=("#b31319", "#ea6a09", "#17a664", "#5391cc", "#1a4eda", "#af1da0"),
)
_ALGO_SLOTS = {"mahony": 0, "madgwick": 4, "kalman": 1, "kf": 1, "ekf": 2, "ukf": 5, "pf": 3, "complementary": 0,
               "dead_reckoning": 0, "weighted_mean": 4}
ALGO_COLORS = ThemedMap(lambda: {key: PALETTE[slot] for key, slot in _ALGO_SLOTS.items()})


def ref_color() -> str:
    """Referans (gerçek) değer: metin rengine yakın nötr, kesikli çizilir."""
    return themed("#f5f5f5", "#1d1a16")


def raw_color() -> str:
    """Ham (füzyonsuz) kestirim: kılavuz grisi."""
    return themed("#8a837c", "#7c7670")
ALGO_WIDTH = 2.0
WINDOWS = [("10 s", 10.0), ("30 s", 30.0), ("60 s", 60.0), ("Tümü", None)]
PUMP_MS = 60
DRAW_EVERY = 3          # her 3 pompalamada bir çizim (~5 Hz)
PLOT_POINTS = 6000      # eksen başına çizilen en fazla nokta
CARD_MIN_HEIGHT = 150   # çıktı kartı: küçük ekranda kartlar ezilmez, alan kaydırılır
# Geniş alanda (genişlik/yükseklik ≥ 1.3; ör. 1366×768, 1600×960) 3B kartı ve doğruluk tablosu grafiklerin altına
# değil sağına, üst üste dizilir: grafikler tam boy kalır. Dar-uzun alanda alt alta. Eşik histerezisli (gidip gelmesin).
SIDE_BY_SIDE_ASPECT = (1.3, 1.2)   # yan yanaya geçiş / alt altaya dönüş oranı
SIDE_BY_SIDE_MIN_WIDTH = 780
NONE = "— yok —"
QUAT = (("w", "q_w"), ("x", "q_x"), ("y", "q_y"), ("z", "q_z"))
PLAY_SPEEDS = [("0.5×", 0.5), ("1×", 1.0), ("2×", 2.0), ("5×", 5.0), ("20×", 20.0)]


def _break_wraps(y: np.ndarray, limit: float = 180.0) -> np.ndarray:
    """±180° sarılmasında dikey çizgi olmasın: sıçrama noktalarına NaN koy."""
    y = np.asarray(y, dtype=np.float64).copy()
    if len(y) > 1:
        jumps = np.flatnonzero(np.abs(np.diff(y)) > limit)
        y[jumps + 1] = np.nan
    return y


def _dispose(widget: QWidget) -> None:
    """Kaldırılan öğeyi hemen gizle ve ayır (deleteLater olay döngüsüne dönünce siler; arada üst üste binmesin)."""
    widget.hide()
    widget.setParent(None)
    widget.deleteLater()


def _quaternions(euler: np.ndarray) -> np.ndarray:
    """(n, 3) roll / pitch / yaw (rad) → (n, 4) kuaterniyon; işaret sürekliliği korunur (q ile −q aynı yönelim)."""
    r, p, y = (euler[:, i] / 2 for i in range(3))
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    q = np.column_stack((cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                         cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy))
    ok = np.isfinite(q).all(axis=1)
    idx = np.flatnonzero(ok)
    if len(idx) > 1:
        dots = np.einsum("ij,ij->i", q[idx[1:]], q[idx[:-1]])
        flips = np.cumsum(dots < 0) % 2  # işaret değişimlerini biriktir
        q[idx[1:]] *= np.where(flips[:, None] == 1, -1.0, 1.0)
    return q


class FusionView(QWidget):
    def __init__(self, theme: ThemeManager):
        super().__init__()
        self._theme = theme
        self.engine: FusionEngine | None = None
        self._sources: list[tuple[str, str, object]] = []  # (kimlik, etiket, tutamaç)
        self._source_id: str | None = None
        self._mode = "est"           # "est" | "error" | "quat"
        self._follow = True
        self._tick = 0
        self._hidden: set[str] = set()                 # lejanttan gizlenen seriler
        self._pending_restart: str | None = None       # kaynak geçici olarak kayboldu (ör. döngülü oynatma başa sardı)
        self._saved: tuple | None = None               # yeniden başlatmada kullanıcının eşlemesi ve ayarları
        self._curves: dict[tuple[str, int], pg.PlotDataItem] = {}
        self._plots: list[pg.PlotWidget] = []
        self._cards: list[Card] = []
        self._tips: list[DataTips] = []
        self._hover_n: int | None = None
        self._frozen: int | None = None                  # "canlı işle" kapalıyken işlenecek son örnek
        self._maps: dict[str, fg.Mapping] = {}          # kaynak → kullanıcının son eşlemesi (geri gelince korunur)
        self._mapped_names: dict[str, tuple] = {}       # kaynak → eşleme yapılırkenki kanal adları
        self.scenario = fg.SCENARIOS["orientation"]

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        self.header = PageHeader("Sensör Füzyonu", "")
        self.progress_chip = chip("")
        self.progress_chip.setMinimumWidth(self.progress_chip.fontMetrics().horizontalAdvance("Canlı · gecikme 999 ms")
                                           + 28)  # metin her tikte değişir: başlık zıplamasın, metin kesilmesin
        self.progress_chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_chip.hide()
        self.mode_combo = combo([])
        self.mode_combo.setMaximumWidth(250)
        self.mode_combo.setToolTip("Kestirimleri, referansa göre hatayı ya da (yönelimde) kuaterniyon bileşenlerini göster")
        self.mode_combo.currentIndexChanged.connect(self._on_mode)
        self.window_combo = combo([w for w, _ in WINDOWS], 1)
        self.window_combo.setMaximumWidth(100)
        self.window_combo.setToolTip("Gösterilen son süre (motor sonuçları 10 dakikaya kadar saklar)")
        self.window_combo.currentIndexChanged.connect(lambda *_: self._return_to_follow())
        fit = icon_button("ph.arrows-out", "Tümünü göster / takibe dön", object_name="IconButton", size=16)
        fit.clicked.connect(self._return_to_follow)
        self.code_button = QPushButton("  C Kodu Üret…")
        self.code_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.code_button.setToolTip("Seçili algoritmalar için MCU'ya özel .h / .c kodu, kullanım kılavuzu, bellek ve "
                                    "işlem maliyeti")
        self.code_button.clicked.connect(self.show_code)
        for w in (self.progress_chip, self.mode_combo, self.window_combo, fit, self.code_button):
            self.header.add_right(w)
        lay.addWidget(self.header)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        lay.addWidget(split, 1)
        split.addWidget(self._build_setup())
        split.addWidget(self._build_right())
        split.setStretchFactor(1, 1)
        split.setSizes([360, 1000])

        self._timer = QTimer(self)
        self._timer.setInterval(PUMP_MS)
        self._timer.timeout.connect(self._pump)
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(40)
        self._play_timer.timeout.connect(self._play_step)
        self._set_scenario("orientation")

    # ================================================================ kurulum paneli
    def _build_setup(self) -> Card:
        card = Card("Girdi ve Algoritmalar")
        card.setMinimumWidth(300)
        card.setMaximumWidth(460)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        host = QWidget()
        self._setup_col = col = QVBoxLayout(host)
        col.setContentsMargins(0, 0, 6, 0)
        col.setSpacing(6)
        scroll.setWidget(host)
        card.body.addWidget(scroll, 1)

        problem = CollapsibleSection("Problem")
        self.scenario_combo = combo([])
        for key, scen in fg.SCENARIOS.items():
            self.scenario_combo.addItem(scen.name, key)
        self.scenario_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.scenario_combo.setMinimumContentsLength(18)
        self.scenario_combo.currentIndexChanged.connect(lambda *_: self._set_scenario(self.scenario_combo.currentData()))
        problem.add_row("Senaryo", self.scenario_combo)
        self.scenario_desc = label("", "Muted")
        self.scenario_desc.setWordWrap(True)
        problem.form.addRow(self.scenario_desc)
        col.addWidget(problem)

        self.source = self._combo()
        self.source.currentIndexChanged.connect(self._on_source)
        src = CollapsibleSection("Kaynak")
        src.add_row("Veri", self.source)
        self.auto_button = QPushButton("Otomatik eşle")
        self.auto_button.setToolTip("Kanal adlarından ve birimlerinden eşle (ivme_x, gyro_z, basinc, gps_x, ref_* …)")
        self.auto_button.clicked.connect(self._auto_map)
        src.form.addRow(self.auto_button)
        col.addWidget(src)

        self.mapping_section = CollapsibleSection("Kanal eşleme")
        col.addWidget(self.mapping_section)
        self.params_section = CollapsibleSection("Senaryo ayarları (sensör gürültüleri, model)")
        col.addWidget(self.params_section)
        self.algos_section = CollapsibleSection("Algoritmalar")
        col.addWidget(self.algos_section)
        self._algo_param_host = QWidget()
        self._algo_param_lay = QVBoxLayout(self._algo_param_host)
        self._algo_param_lay.setContentsMargins(0, 0, 0, 0)
        self._algo_param_lay.setSpacing(6)
        col.addWidget(self._algo_param_host)
        col.addStretch(1)

        self.status = label("", "Readout")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.RichText)
        card.body.addWidget(self.status)
        self.live = QCheckBox("Canlı akışta yeni örnekleri işle")
        self.live.setChecked(True)
        self.live.toggled.connect(self._on_live_toggled)
        card.body.addWidget(self.live)
        self.run_button = QPushButton("Çalıştır")
        self.run_button.setObjectName("Primary")
        self.run_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.run_button.clicked.connect(self.run)
        self.stop_button = QPushButton("Durdur")
        self.stop_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.setEnabled(False)
        card.body.addWidget(hbox(self.run_button, self.stop_button, spacing=8))
        return card

    @staticmethod
    def _combo() -> QComboBox:
        w = combo([])
        w.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        w.setMinimumContentsLength(8)
        return w

    @staticmethod
    def _clear_form(section: CollapsibleSection) -> None:
        form = section.form
        while form.rowCount():
            form.removeRow(0)

    def _param_widget(self, p) -> QWidget:
        if isinstance(p, fg.Choice):
            w = combo([])
            for value, text in p.options:
                w.addItem(text, value)
            w.setCurrentIndex(max(w.findData(p.default), 0))
        else:
            w = QDoubleSpinBox()
            w.setRange(p.lo, p.hi)
            w.setDecimals(p.decimals)
            w.setValue(p.default)
            w.setKeyboardTracking(False)
            w.setMinimumWidth(90)
        w.setToolTip(p.tip)
        return w

    @staticmethod
    def _param_value(widget):
        return widget.currentData() if isinstance(widget, QComboBox) else widget.value()

    def algo_color(self, key: str) -> str:
        """Algoritmanın rengi (senaryo içinde benzersiz): bilinen anahtar sabit renkte, bilinmeyen boş renkte."""
        if key in ALGO_COLORS:
            return ALGO_COLORS[key]
        used = {ALGO_COLORS.get(k) for k in self.scenario.algorithms if k in ALGO_COLORS}
        free = [c for c in PALETTE if c not in used]
        unknown = [k for k in self.scenario.algorithms if k not in ALGO_COLORS]
        return free[unknown.index(key) % len(free)] if free else PALETTE[0]

    # ---------------------------------------------------------------- senaryo değişimi
    def _set_scenario(self, key: str) -> None:
        if self.engine is not None:
            self.stop()
        self.scenario = scen = fg.SCENARIOS[key]
        if self.scenario_combo.currentData() != key:
            self.scenario_combo.blockSignals(True)
            self.scenario_combo.setCurrentIndex(self.scenario_combo.findData(key))
            self.scenario_combo.blockSignals(False)
        self.header.subtitle.setText(f"{scen.name} · algoritma karşılaştırması")
        self.scenario_desc.setText(scen.description)
        # kanal eşleme satırları
        self._clear_form(self.mapping_section)
        self._map_inputs: dict[str, tuple[list[QComboBox], QComboBox]] = {}
        for inp in scen.inputs:
            boxes = [self._combo() for _ in inp.axes]
            unit = combo(list(inp.units))
            unit.setToolTip("Kanalın birimi (SI'ya çevrilir)")
            short = inp.label.split(" (")[0]
            optional = "" if inp.required else " — isteğe bağlı (yoksa “— yok —”)"
            for box, axis in zip(boxes, inp.axes):
                box.setToolTip(f"{inp.label}{' ' + axis if axis else ''}{optional}" + (f"<br>{inp.tip}" if inp.tip else ""))
                text = f"{short} {axis}".strip() if len(boxes) > 1 else short
                self.mapping_section.add_row(text if inp.required else f"{text} ◦", box)
            self.mapping_section.add_row(f"{short} birimi", unit)
            self._map_inputs[inp.key] = (boxes, unit)
        self._map_refs: dict[str, QComboBox] = {}
        for out in scen.outputs:
            box = self._combo()
            box.setToolTip(f"{out.label} için referans (gerçek değer) kanalı — hata ve RMSE için")
            self.mapping_section.add_row(f"Referans: {out.label.split(' (')[0]}", box)
            self._map_refs[out.key] = box
        # senaryo ayarları
        self._clear_form(self.params_section)
        self._scen_widgets: dict[str, QWidget] = {}
        for p in scen.params:
            w = self._param_widget(p)
            self._scen_widgets[p.key] = w
            self.params_section.add_row(p.label, w)
        self.params_section.setVisible(bool(scen.params))
        # algoritmalar
        self._clear_form(self.algos_section)
        holder = QWidget()
        grid = FlowLayout(holder, spacing=6, uniform=True)  # sığarsa iki sütun, dar panelde tek sütun
        self._enabled: dict[str, QCheckBox] = {}
        for akey, cls in scen.algorithms.items():
            box = QCheckBox(cls.name)
            box.setChecked(True)
            box.setIcon(line_swatch(self.algo_color(akey), Qt.PenStyle.SolidLine, ALGO_WIDTH + 1))
            box.setToolTip(f"{cls.name}: grafiklerde bu renkle çizilir")
            box.toggled.connect(lambda *_: self._rebuild_legend())
            self._enabled[akey] = box
            grid.addWidget(box)
        self.algos_section.form.addRow(holder)
        while self._algo_param_lay.count():
            w = self._algo_param_lay.takeAt(0).widget()
            if w is not None:
                _dispose(w)
        self._params: dict[str, dict[str, QWidget]] = {}
        for akey, cls in scen.algorithms.items():
            if not cls.params:
                continue
            section = CollapsibleSection(f"{cls.name} parametreleri", expanded=False)
            widgets = {}
            for p in cls.params:
                w = self._param_widget(p)
                widgets[p.key] = w
                section.add_row(p.label, w)
            self._params[akey] = widgets
            self._algo_param_lay.addWidget(section)
        # görünüm
        self.mode_combo.blockSignals(True)
        self.mode_combo.clear()
        self.mode_combo.addItems(["Kestirim", "Hata (kestirim − referans)"]
                                 + (["Kuaterniyon bileşenleri"] if key == "orientation" else []))
        self.mode_combo.blockSignals(False)
        self._mode = "est"
        self.side_card.setVisible(key in ("orientation", "position2d"))
        self.orient_box.setVisible(key == "orientation")
        self.traj_box.setVisible(key == "position2d")
        self.side_card.title.setText("3B Yönelim" if key == "orientation" else "Yörünge (x–y)")
        self._saved = None
        self._maps.clear()
        self._mapped_names.clear()
        self._build_output_cards()
        self._rebuild_legend()
        self._fill_mapping_sources()
        self._auto_map()

    # ================================================================ sağ taraf: grafikler, yan kart, tablo
    def _build_right(self) -> QWidget:
        self._arrange_timer = QTimer(self)  # düzen değişimi yeniden boyutlama olayının dışında (görünüme bağlı)
        self._arrange_timer.setSingleShot(True)
        self._arrange_timer.timeout.connect(lambda: self._set_side_by_side(self._want_side))
        self._want_side = False
        self._vsplit = vsplit = QSplitter(Qt.Orientation.Vertical)
        vsplit.setChildrenCollapsible(False)
        vsplit.installEventFilter(self)  # boyut değişince düzen (alt alta / yan yana) seçilir
        self._bottom = bottom = QSplitter(Qt.Orientation.Horizontal)
        bottom.setChildrenCollapsible(False)

        self.angles_card = Card("Kestirimler")
        legend_host = QWidget()
        self._legend_lay = FlowLayout(legend_host, spacing=10)
        self.angles_card.body.addWidget(legend_host)
        self._cards_scroll = QScrollArea()
        self._cards_scroll.setWidgetResizable(True)
        self._cards_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._cards_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._cards_host = QWidget()
        self._cards_lay = QVBoxLayout(self._cards_host)
        self._cards_lay.setContentsMargins(0, 0, 4, 0)
        self._cards_lay.setSpacing(12)  # kartlar arasında belirgin boşluk
        self._cards_scroll.setWidget(self._cards_host)
        self.angles_card.body.addWidget(self._cards_scroll, 1)
        vsplit.addWidget(self.angles_card)

        self.side_card = Card("3B Yönelim")  # en küçük genişlik içerikten (dar ekranda satırlar alta kayar)
        self.orient_box = self._build_orientation_box()
        self.traj_box = self._build_trajectory_box()
        self.side_card.body.addWidget(self.orient_box, 1)
        self.side_card.body.addWidget(self.traj_box, 1)

        self.metrics_card = Card("Doğruluk ve performans")
        self.metrics_note = label("", "Muted")
        self.metrics_note.setWordWrap(True)
        self.metrics_card.body.addWidget(self.metrics_note)
        self.table = QTableWidget()
        self.table.setObjectName("FusionTable")
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(True)
        self.table.setAlternatingRowColors(True)  # satırlar birbirinden belirgin ayrılır
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ContiguousSelection)
        self.table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft)
        self.table.verticalHeader().setDefaultSectionSize(26)
        self.table.setMinimumHeight(110)
        self.table.setToolTip("Hücreler seçilip Ctrl+C ile kopyalanabilir")
        t = self._theme.tokens
        self.table.setStyleSheet(f"QTableWidget#FusionTable {{ gridline-color: {t.border_strong}; "
                                 f"alternate-background-color: {t.surface_alt}; }}"
                                 f"QTableWidget#FusionTable QHeaderView::section {{ font-weight: 600; "
                                 f"border-bottom: 2px solid {t.border_strong}; }}")
        self.metrics_card.body.addWidget(self.table, 1)
        bottom.addWidget(self.metrics_card)
        bottom.addWidget(self.side_card)
        bottom.setStretchFactor(0, 3)
        bottom.setStretchFactor(1, 2)
        bottom.setSizes([620, 420])
        vsplit.addWidget(bottom)
        vsplit.setStretchFactor(0, 2)
        vsplit.setStretchFactor(1, 1)
        vsplit.setSizes([620, 320])
        return vsplit

    # ---------------------------------------------------------------- uyarlanır düzen
    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if obj is self._vsplit and event.type() == QEvent.Type.Resize:
            self._arrange()
        return super().eventFilter(obj, event)

    @property
    def side_by_side(self) -> bool:
        return self._vsplit.orientation() == Qt.Orientation.Horizontal

    def _arrange(self) -> None:
        v = self._vsplit
        side = self.side_by_side
        enter, leave = SIDE_BY_SIDE_ASPECT
        want = (v.width() >= (leave if side else enter) * v.height()
                and v.width() >= SIDE_BY_SIDE_MIN_WIDTH - (40 if side else 0))
        if want != side:
            self._want_side = want
            self._arrange_timer.start(0)

    def _set_side_by_side(self, on: bool) -> None:
        """Yan yana: [grafikler | 3B üstte, tablo altta]; alt alta: [grafikler / tablo | 3B]. Kartların yerinde
        ayrık pencere yer tutucusu olabilir: kartlar değil, bölücüdeki yuvalar yer değiştirir."""
        if on == self.side_by_side:
            return
        v, b = self._vsplit, self._bottom
        v.setOrientation(Qt.Orientation.Horizontal if on else Qt.Orientation.Vertical)
        b.setOrientation(Qt.Orientation.Vertical if on else Qt.Orientation.Horizontal)
        b.insertWidget(0, b.widget(1))  # yan yanada 3B (yer tutucusu) üstte, alt altada tablo solda
        total = v.width() if on else v.height()
        first = int(total * (0.64 if on else 0.62))
        v.setSizes([first, total - first])
        span = b.height() if on else b.width()
        b.setSizes([int(span * 0.58), span - int(span * 0.58)] if on else [int(span * 0.6), span - int(span * 0.6)])

    def _build_orientation_box(self) -> QWidget:
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        t = self._theme.tokens
        self.view3d_algo = combo([])
        self.view3d_algo.setToolTip("Dolu gövdeyle gösterilen algoritma")
        self.view3d_algo.currentIndexChanged.connect(lambda *_: self._update_3d())
        self.ghost_ref = QCheckBox("Referans")
        self.ghost_ref.setChecked(True)
        self.ghost_ref.setToolTip("Referans yönelimi beyaz kesikli tel kafes (hayalet) olarak göster")
        self.ghost_others = QCheckBox("Diğerleri")
        self.ghost_others.setToolTip("Diğer algoritmaları kendi renklerinde tel kafes olarak göster")
        for w in (self.ghost_ref, self.ghost_others):
            w.toggled.connect(lambda *_: self._update_3d())
        top = QWidget()
        flow = FlowLayout(top, spacing=8)  # dar kartta onay kutuları alt satıra iner, kesilmez
        for w in (self.view3d_algo, self.ghost_ref, self.ghost_others):
            flow.addWidget(w)
        lay.addWidget(top)
        self.view3d = Orientation3D(t)  # yapay ufuk (roll / pitch) ve pusula (yaw) alt köşelerinde
        self.view3d.setMinimumSize(200, 140)
        lay.addWidget(self.view3d, 1)
        self.play_button = icon_button("ph.play", "Yönelim geçmişini oynat / duraklat", object_name="IconButton", size=16)
        self.play_button.setCheckable(True)
        self.play_button.toggled.connect(self._toggle_play)
        self.play_speed = combo([s for s, _ in PLAY_SPEEDS], 1)
        self.play_speed.setMaximumWidth(70)
        self.follow_latest = QCheckBox("En son")
        self.follow_latest.setChecked(True)
        self.follow_latest.setToolTip("Her zaman en son işlenen örneği göster (grafiklerde gezinirken imlecin zamanı)")
        self.follow_latest.toggled.connect(lambda *_: self._update_3d())
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setMinimumWidth(50)
        self.time_slider.setRange(0, 0)
        self.time_slider.valueChanged.connect(self._on_slider)
        lay.addWidget(hbox(self.play_button, self.play_speed, self.time_slider, self.follow_latest, spacing=6))
        return box

    def _build_trajectory_box(self) -> QWidget:
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        self.traj_plot = make_plot(self._theme, "x (doğu)", "y (kuzey)", x_units="m", y_units="m")
        self.traj_plot.setAspectLocked(True)
        self._traj_curves: dict[str, pg.PlotDataItem] = {}
        lay.addWidget(self.traj_plot, 1)
        lay.addWidget(label("Tıkla: noktanın değerleri · GNSS ölçümleri gri noktalar", "Muted"))
        self.traj_tips = DataTips(self.traj_plot, self._theme, self._traj_traces)
        return box

    # ---------------------------------------------------------------- çıktı kartları
    def _series_defs(self) -> list[tuple[str, str, float, bool]]:
        """Çizilen her kartın (başlık, birim, gösterim çarpanı, açı mı)."""
        if self._mode == "quat":
            return [(f"Kuaterniyon {name}", "", 1.0, False) for name, _ in QUAT]
        return [(o.label, o.unit, o.display_scale, o.angle) for o in self.scenario.outputs]

    def _build_output_cards(self) -> None:
        for tips in self._tips:
            tips.clear()
        while self._cards_lay.count():
            w = self._cards_lay.takeAt(0).widget()
            if w is not None:
                _dispose(w)
        self._plots, self._cards, self._tips, self._curves = [], [], [], {}
        defs = self._series_defs()
        error = self._mode == "error"
        for i, (title, unit, _scale, _angle) in enumerate(defs):
            card = Card(f"{title}{' hatası' if error else ''}" + (f"  [{unit}]" if unit else ""))
            card.setObjectName("Card")
            card.setMinimumHeight(CARD_MIN_HEIGHT)
            plot = make_plot(self._theme, "Zaman" if i == len(defs) - 1 else "", unit or "", x_units="s",
                             si_prefix=False)
            plot.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.RectMode)
            plot.getPlotItem().getViewBox().sigRangeChangedManually.connect(self._on_manual_range)
            plot.scene().sigMouseClicked.connect(lambda ev: self._return_to_follow() if ev.double() else None)
            if i:
                plot.setXLink(self._plots[0])
            ref_pen = pg.mkPen(ref_color(), width=1.6, style=Qt.PenStyle.DashLine)
            ref_pen.setCosmetic(True)
            ref = plot.plot(pen=ref_pen, connect="finite")
            raw = plot.plot(pen=pen(raw_color(), 1), connect="finite")
            ref.setZValue(5)
            raw.setZValue(-1)
            self._curves[("ref", i)] = ref
            self._curves[("raw", i)] = raw
            for akey in self.scenario.algorithms:
                self._curves[(akey, i)] = plot.plot(pen=pen(self.algo_color(akey), ALGO_WIDTH), connect="finite")
            Crosshair(plot, self._theme, self._on_hover)
            self._tips.append(DataTips(plot, self._theme, lambda i=i: self._traces(i)))
            card.body.addWidget(plot, 1)
            PlotAnalysis(card, plot, self._theme, lambda x0, x1, i=i: self._window_stats(i, x0, x1))
            self._cards_lay.addWidget(card)
            self._plots.append(plot)
            self._cards.append(card)
        self._apply_visibility()

    def _rebuild_legend(self) -> None:
        """Lejant öğeleri düğmedir: tıklayınca o seri tüm kartlarda gizlenir / gösterilir. Satıra sığmayan alta geçer."""
        lay = self._legend_lay
        while lay.count():
            item = lay.takeAt(0)
            if item.widget() is not None:
                _dispose(item.widget())
        items = [(k, self.scenario.algorithms[k].name.split(" (")[0], self.algo_color(k), Qt.PenStyle.SolidLine)
                 for k, box in self._enabled.items() if box.isChecked()]
        items += [("ref", "referans", ref_color(), Qt.PenStyle.DashLine),
                  ("raw", "ham (füzyonsuz)", raw_color(), Qt.PenStyle.SolidLine)]
        for key, name, color, style in items:
            btn = QToolButton()
            btn.setObjectName("LegendToggle")
            btn.setText(name)
            btn.setIcon(line_swatch(color, style, 3))
            btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            btn.setCheckable(True)
            btn.setChecked(key not in self._hidden)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setToolTip("Göster / gizle")
            btn.toggled.connect(lambda on, k=key: self._set_trace_visible(k, on))
            lay.addWidget(btn)
        self._apply_visibility()

    def _set_trace_visible(self, key: str, on: bool) -> None:
        (self._hidden.discard if on else self._hidden.add)(key)
        self._apply_visibility()
        curve = self._traj_curves.get(key)
        if curve is not None:
            curve.setVisible(on)
        self._update_3d()

    def _apply_visibility(self) -> None:
        running = set(self.engine.algos) if self.engine is not None else None
        for (key, _), curve in self._curves.items():
            if key in ("ref", "raw"):
                visible = key not in self._hidden and not (key == "raw" and self._mode == "quat")
            else:
                visible = (key not in self._hidden and self._enabled.get(key) is not None
                           and self._enabled[key].isChecked() and (running is None or key in running))
            curve.setVisible(visible)

    # ================================================================ kaynaklar ve eşleme
    def set_sources(self, sources: list[tuple[str, str, object]]) -> None:
        """sources: (kimlik, etiket, tutamaç). Kimlik akış anahtarı ya da 'ws<n>'.

        Çalışan füzyonun kaynağı geçici olarak kaybolursa (akış kapandı, döngülü oynatma başa sardı) eşleme kutuları
        olduğu gibi korunur ve başka kaynağa geçilmez; aynı akış gelince kullanıcının eşlemesiyle yeniden başlar."""
        current = self._source_id
        running = self.engine is not None and not self.engine.stopped
        self._sources = list(sources)
        available = {sid for sid, _, _ in self._sources}
        self.source.blockSignals(True)
        self.source.clear()
        for sid, text, _ in self._sources:
            self.source.addItem(text, sid)
        wanted = self._pending_restart or current
        other_streams = any(not str(sid).startswith("ws") for sid in available)
        transient = wanted is not None and not str(wanted).startswith("ws") and not other_streams
        if wanted is not None and wanted not in available and (running or self._pending_restart or transient):
            # kaynak şu an yok: başka akışa atlama, kutuları koru (geri gelince aynı eşlemeyle devam)
            self.source.setCurrentIndex(-1)
            self.source.blockSignals(False)
            if running:
                self._pending_restart = current
                self.stop()
                self._set_status("Kaynak şu an yok (akış kapandı ya da kayıt başa sarıyor); aynı akış gelince "
                                 "füzyon aynı ayarlarla kendiliğinden yeniden başlar.", "warning")
            return
        target = self._pending_restart or current
        index = self.source.findData(target) if target else -1
        self.source.setCurrentIndex(index if index >= 0 else (0 if self._sources else -1))
        self.source.blockSignals(False)
        if self._pending_restart and self.source.currentData() == self._pending_restart:
            self._source_id = self._pending_restart
            self._pending_restart = None
            self._fill_mapping_sources(keep=True)
            if self._saved is not None:
                self._apply_map(self._saved[1])  # kutular da kullanıcının eşlemesini göstersin
            self.run(restore=True)
            return
        if self.source.currentData() != current:
            self._on_source()
        elif running and self._handle() is not self.engine.handle:  # aynı akış yeniden kuruldu: yeniden başlat
            self.run(restore=True)

    def _handle(self):
        sid = self.source.currentData()
        return next((h for s, _, h in self._sources if s == sid), None)

    def _channel_names(self, handle) -> tuple:
        return tuple(f"{c.name}|{c.unit}" for c in handle.desc.channels) if handle is not None else ()

    def _remember_mapping(self) -> None:
        """Geçerli kaynağın kutulardaki eşlemesini sakla (kaynak geri gelince otomatik eşleme üzerine yazmasın)."""
        sid = self._source_id
        if sid is not None and sid in self._mapped_names:
            self._maps[sid] = self._selection()

    def _on_source(self, *_) -> None:
        self._remember_mapping()
        self._source_id = sid = self.source.currentData()
        handle = self._handle()
        self.live.setVisible(handle is not None and not hasattr(handle, "snapshot"))  # anlık görüntüde anlamsız
        if handle is None:
            return  # kaynak yok: kutulara dokunma
        names = self._channel_names(handle)
        self._fill_mapping_sources()
        saved = self._maps.get(sid)
        if saved is not None and self._mapped_names.get(sid) == names:
            self._apply_map(saved)  # aynı akış geri geldi (ör. döngülü oynatma başa sardı): kullanıcı eşlemesi
            self._set_status("Bu kaynağın önceki eşlemesi korundu (yeniden önermek için “Otomatik eşle”).", "good")
        else:
            self._auto_map()
        self._mapped_names[sid] = names

    def _fill_mapping_sources(self, keep: bool = False) -> None:
        handle = self._handle()
        names = [] if handle is None else [f"{i}: {c.name or f'ch{i}'} [{c.unit}]"
                                           for i, c in enumerate(handle.desc.channels)]
        boxes = [(box, not inp.required) for inp in self.scenario.inputs for box in self._map_inputs[inp.key][0]]
        boxes += [(box, True) for box in self._map_refs.values()]
        for box, optional in boxes:
            previous = box.currentData() if keep else None
            box.blockSignals(True)
            box.clear()
            if optional:
                box.addItem(NONE, None)
            for i, text in enumerate(names):
                box.addItem(text, i)
            if keep and previous is not None:
                box.setCurrentIndex(max(box.findData(previous), 0))
            box.blockSignals(False)

    def _auto_map(self) -> None:
        handle = self._handle()
        if handle is not None:
            self._maps.pop(self._source_id, None)  # kullanıcı yeniden öneri istedi
        if handle is None:
            self._set_status("Kaynak seçin: canlı / oynatılan akış ya da Çalışma Alanı'ndaki anlık görüntü. "
                             "Örnekler: <i>10_imu…</i> (yönelim), <i>11_irtifa…</i>, <i>12_arac_gnss…</i>, "
                             "<i>13_coklu_mesafe…</i>")
            return
        m = fg.auto_map(self.scenario, handle.desc)
        if m is None:
            self._set_status(f"Kanal adlarından “{self.scenario.name}” eşlemesi çıkarılamadı; gerekli kanalları elle "
                             "seçin ya da başka bir senaryo deneyin.", "warning")
            return
        self._apply_map(m)
        found = [self.scenario.input(k).label for k in m.inputs]
        ref = f"referans ✓ ({len(m.reference)} çıktı)" if m.reference else "referans yok (hata hesaplanmaz)"
        self._set_status(f"Eşleme önerildi: {', '.join(found)} · {ref}", "good")

    def _apply_map(self, m: fg.Mapping) -> None:
        for inp in self.scenario.inputs:
            boxes, unit = self._map_inputs[inp.key]
            idx = m.inputs.get(inp.key)
            for a, box in enumerate(boxes):
                box.setCurrentIndex(max(box.findData(None if idx is None else idx[a]), 0))
            if inp.key in m.units:
                unit.setCurrentIndex(max(unit.findText(m.units[inp.key]), 0))
        for out in self.scenario.outputs:
            box = self._map_refs[out.key]
            box.setCurrentIndex(max(box.findData(m.reference.get(out.key)), 0))
        for key, value in (m.values or {}).items():  # önerilen senaryo değerleri (ör. sensör türleri)
            w = self._scen_widgets.get(key)
            if isinstance(w, QComboBox):
                w.setCurrentIndex(max(w.findData(value), 0))
            elif w is not None:
                w.setValue(float(value))

    def _selection(self) -> fg.Mapping:
        """Kutulardaki eşleme (zorunlu giriş eksik olsa da): yalnız tamamı seçili girişler."""
        inputs, units, reference = {}, {}, {}
        for inp in self.scenario.inputs:
            boxes, unit = self._map_inputs[inp.key]
            idx = [b.currentData() for b in boxes]
            if all(i is not None for i in idx):
                inputs[inp.key] = tuple(idx)
                units[inp.key] = unit.currentText()
        for out in self.scenario.outputs:
            ch = self._map_refs[out.key].currentData()
            if ch is not None:
                reference[out.key] = ch
        return fg.Mapping(inputs, units, reference)

    def mapping(self) -> fg.Mapping | None:
        inputs, units, reference = {}, {}, {}
        for inp in self.scenario.inputs:
            boxes, unit = self._map_inputs[inp.key]
            idx = [b.currentData() for b in boxes]
            if any(i is None for i in idx):
                if inp.required:
                    return None
                continue
            inputs[inp.key] = tuple(idx)
            units[inp.key] = unit.currentText()
        for out in self.scenario.outputs:
            ch = self._map_refs[out.key].currentData()
            if ch is not None:
                reference[out.key] = ch
        return fg.Mapping(inputs, units, reference)

    def scenario_values(self) -> dict:
        return {k: self._param_value(w) for k, w in self._scen_widgets.items()}

    def algorithms(self) -> dict[str, dict]:
        return {k: {p: self._param_value(w) for p, w in self._params.get(k, {}).items()}
                for k, box in self._enabled.items() if box.isChecked()}

    # ================================================================ çalıştırma
    def run(self, restore: bool = False) -> bool:
        self._pending_restart = None
        handle = self._handle()
        if restore and self._saved is not None and self._saved[0] == self.scenario.key:
            _, mapping, values, algos = self._saved
        else:
            mapping, values, algos = self.mapping(), self.scenario_values(), self.algorithms()
        if handle is None:
            self._set_status("Önce bir kaynak seçin (UDP dinleyin, bir kayıt açın ya da Dondur ile anlık görüntü alın).",
                             "warning")
            return False
        if mapping is None:
            missing = [inp.label for inp in self.scenario.inputs
                       if inp.required and any(b.currentData() is None for b in self._map_inputs[inp.key][0])]
            self._set_status("Zorunlu girişler eşlenmeli: " + ", ".join(missing), "warning")
            return False
        if not algos:
            self._set_status("En az bir algoritma seçin.", "warning")
            return False
        warning = self._unit_sanity(handle, mapping)
        if self.engine is not None:
            self.stop()
        try:
            self.engine = FusionEngine(handle, self.scenario.key, mapping, values, algos)
        except (fg.FusionError, ValueError) as exc:
            self.engine = None
            self._set_status(str(exc), "critical")
            return False
        self._saved = (self.scenario.key, mapping, values, algos)
        self._remember_mapping()
        live = not hasattr(handle, "snapshot")
        self._frozen = handle.latest_index() if live and not self.live.isChecked() else None
        for curve in self._traj_curves.values():  # önceki çalıştırmanın yörüngeleri kalmasın
            self.traj_plot.removeItem(curve)
        self._traj_curves.clear()
        self._tick = 0
        for tips in self._tips:
            tips.clear()
        self.traj_tips.clear()
        for curve in self._curves.values():
            curve.setData([], [])
        self._apply_visibility()
        self._rebuild_legend()
        self._update_3d_choices()
        self._follow = True
        self.run_button.setText("Yeniden çalıştır")
        self.stop_button.setEnabled(True)
        self.progress_chip.show()
        self._set_status(f"{len(self.engine.algos)} algoritma ayrı süreçlerde başlatıldı…"
                         + (f"<br>⚠ {warning}" if warning else ""), "warning" if warning else "")
        log.info("Sensör füzyonu (%s) başladı: %s · %s", self.scenario.name,
                 ", ".join(self.scenario.algorithms[k].name for k in self.engine.algos), self.source.currentText())
        self._timer.start()
        return True

    def _unit_sanity(self, handle, mapping: fg.Mapping) -> str:
        """Yönelimde jiroskop birimi makul mü (°/s veri rad/s seçilirse ×57 hata — denetim bulgusu)."""
        if self.scenario.key != "orientation" or "gyro" not in mapping.inputs:
            return ""
        ring = handle.ring
        end = ring.write_count
        first, raw = ring.read(max(ring.valid_from(), end - 2000), end)
        if not len(raw):
            return ""
        from spt.dsp.convert import physical_column
        cols = [physical_column(raw[:, i], handle.desc.sample_type, handle.desc.channels[i]).real
                for i in mapping.inputs["gyro"]]
        unit = mapping.units.get("gyro", "rad/s")
        scale = self.scenario.input("gyro").units.get(unit, 1.0)
        rate = float(np.nanmedian(np.linalg.norm(np.column_stack(cols), axis=1))) * scale
        if rate > 20.0:  # rad/s; insan / araç hareketinde medyan ~1 rad/s'den küçük
            return (f"Jiroskop medyan hızı {rate:.1f} rad/s — birim büyük olasılıkla yanlış (veri °/s ise birimi "
                    "“°/s” seçin).")
        return ""

    def stop(self) -> None:
        self._timer.stop()
        self._play_timer.stop()
        if self.engine is not None:
            self.engine.stop()  # engellemesiz; son sonuçlar görünümde kalır
        self.engine = None
        self.stop_button.setEnabled(False)
        self.run_button.setText("Çalıştır")
        self.progress_chip.hide()

    def shutdown(self) -> None:
        self.stop()
        dlg = getattr(self, "_code_dialog", None)
        if dlg is not None:
            dlg.shutdown()

    def _on_live_toggled(self, on: bool) -> None:
        """Dondurulmuş aralık bitip zamanlayıcı durduysa, canlı işleme yeniden açılınca kaldığı yerden sürer."""
        e = self.engine
        if on and e is not None and not e.stopped and not hasattr(e.handle, "snapshot") and not self._timer.isActive():
            self._timer.start()
            self.stop_button.setEnabled(True)
            self._set_status("Canlı işleme sürüyor: yeni örnekler işleniyor.", "good")

    def _pump(self) -> None:
        e = self.engine
        if e is None:
            return
        live = not hasattr(e.handle, "snapshot")
        if not live or self.live.isChecked():
            end = e.handle.latest_index()
            self._frozen = None
        else:  # canlı işleme kapalı: kapatıldığı (ya da Çalıştır'a basıldığı) andaki son örneğe kadar işle
            if self._frozen is None:
                self._frozen = e.handle.latest_index()
            end = self._frozen
        e.pump(end)
        errors = [f"{self.scenario.algorithms[k].name}: {a.error.strip().splitlines()[-1]}"
                  for k, a in e.algos.items() if a.error]
        if errors:
            self._set_status("<br>".join(errors), "critical")
        progress = e.progress(end)
        lag = (end - min((e.processed(k) for k in e.algos), default=end)) / e.fs
        if live and self.live.isChecked() and lag < 1.0 and e.processed(next(iter(e.algos))) > e.start:
            self.progress_chip.setText(f"Canlı · gecikme {lag * 1000:.0f} ms")
        else:
            self.progress_chip.setText(f"İşlenen %{progress * 100:.0f}" if progress < 1 else
                                       ("Canlı" if live and self.live.isChecked() else "Tamamlandı"))
        self._tick += 1
        if self._tick % DRAW_EVERY == 1:
            self.redraw()
        if self._tick % (DRAW_EVERY * 4) == 1:
            self._update_metrics()
        overrun = {k: a.overrun for k, a in e.algos.items() if a.overrun}
        if overrun and self._tick % (DRAW_EVERY * 4) == 1:
            names = ", ".join(f"{self.scenario.algorithms[k].name.split(' (')[0]} {n / e.fs:.1f} s"
                              for k, n in overrun.items())
            self._set_status(f"Yetişemeyen algoritmalar veri atladı: {names}. Oynatma hızını düşürün, parçacık "
                             "sayısını azaltın ya da yavaş algoritmayı kapatın.", "warning")
        if progress >= 1 and not e.busy and not (live and self.live.isChecked()):
            self.redraw()
            self._update_metrics()
            self._timer.stop()
            self.progress_chip.setText("Tamamlandı")
            self.stop_button.setEnabled(False)
            if not errors and not overrun:
                self._set_status("Tamamlandı. Grafiklere tıklayarak noktaların değerlerini, kart başlığındaki tablo "
                                 "düğmesiyle görünen aralığın ayrıntılı analizini görebilirsiniz.", "good")

    # ================================================================ çizim
    def _range(self) -> tuple[int, int]:
        e = self.engine
        h = e.handle
        end = max((e.processed(k) for k in e.algos), default=e.start)
        start = e.history_start
        seconds = WINDOWS[self.window_combo.currentIndex()][1]
        if self._follow and seconds is not None:
            start = max(start, end - int(seconds * h.fs))
        if not self._follow and self._plots:
            x0, x1 = self._plots[0].getPlotItem().getViewBox().viewRange()[0]
            start = max(start, int(math.floor((x0 - h.t0) * h.fs)) - 1)
            end = min(end, int(math.ceil((x1 - h.t0) * h.fs)) + 2)
        return start, max(start, end)

    def _display(self, values: np.ndarray) -> np.ndarray:
        """İç birimden gösterime (açılar °); kuaterniyon kipinde (n, 4)."""
        if self._mode == "quat":
            return _quaternions(values)
        return values * np.array([o.display_scale for o in self.scenario.outputs])

    def redraw(self) -> None:
        e = self.engine
        if e is None or not self._plots:
            return
        start, end = self._range()
        if end <= start:
            return
        h = e.handle
        step = max(1, (end - start) // PLOT_POINTS)
        idx = np.arange(start, end, step)
        t = h.t0 + idx / h.fs
        defs = self._series_defs()
        error = self._mode == "error"
        ref_all = e.reference(start, end)
        raw_all = e.raw_estimate(start, end)
        ref = None if ref_all is None else ref_all[::step]
        raw = raw_all[::step]
        angle = [d[3] for d in defs]

        def show(curve, y, i, sparse_ok=False):
            y = np.asarray(y, dtype=np.float64)
            if angle[i]:  # ±180° sarılması dikey çizgi olmasın
                y = _break_wraps(y)
            finite = np.isfinite(y)
            sparse = sparse_ok and finite.any() and finite.mean() < 0.5
            if sparse:  # seyrek sensör (5 Hz GNSS, 25 Hz baro): noktalar
                curve.setData(t[finite], y[finite], pen=None, symbol="o", symbolSize=4,
                              symbolBrush=pg.mkBrush(raw_color()), symbolPen=None)
            else:
                curve.setData(t[:len(y)], y, connect="finite")

        ref_disp = None if ref is None else (self._display(ref) if not error else None)
        raw_disp = self._display(raw) if self._mode != "quat" else None
        for i in range(len(defs)):
            if error:
                self._curves[("ref", i)].setData([], [])
                if raw_disp is not None and ref is not None:
                    err = fg.errors(self.scenario, raw, ref) * np.array([o.display_scale for o in self.scenario.outputs])
                    show(self._curves[("raw", i)], err[:, i], i, sparse_ok=True)
                else:
                    self._curves[("raw", i)].setData([], [])
            else:
                if ref_disp is not None and np.isfinite(ref_disp[:, i]).any():
                    show(self._curves[("ref", i)], ref_disp[:, i], i)
                else:
                    self._curves[("ref", i)].setData([], [])
                if raw_disp is not None:
                    show(self._curves[("raw", i)], raw_disp[:, i], i, sparse_ok=True)
                else:
                    self._curves[("raw", i)].setData([], [])
        for key in e.algos:
            est = e.read(key, start, end)[::step]
            if error:
                if ref is None:
                    for i in range(len(defs)):
                        self._curves[(key, i)].setData([], [])
                    continue
                disp = fg.errors(self.scenario, est, ref) * np.array([o.display_scale for o in self.scenario.outputs])
            else:
                disp = self._display(est)
            for i in range(len(defs)):
                show(self._curves[(key, i)], disp[:, i], i)
        if self._follow:
            self._plots[0].setXRange(t[0], t[-1], padding=0)
            for p in self._plots:
                p.getPlotItem().getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis)
        if self.scenario.key == "position2d":
            self._draw_trajectory(start, end, step)
        if self.scenario.key == "orientation":
            self._sync_slider()
            self._update_3d()

    def _on_mode(self, index: int) -> None:
        self._mode = ("est", "error", "quat")[index] if index >= 0 else "est"
        self._build_output_cards()
        self.redraw()

    def _on_manual_range(self, *_) -> None:
        self._follow = False
        QTimer.singleShot(0, self.redraw)

    def _return_to_follow(self) -> None:
        self._follow = True
        self.redraw()

    # ---------------------------------------------------------------- 2B yörünge
    def _draw_trajectory(self, start: int, end: int, step: int) -> None:
        e = self.engine
        plot = self.traj_plot
        ix, iy = self.scenario.output_index("x"), self.scenario.output_index("y")
        series = [(k, self.algo_color(k), e.read(k, start, end)[::step]) for k in e.algos]
        ref = e.reference(start, end)
        if ref is not None:
            series.append(("ref", ref_color(), ref[::step]))
        series.append(("raw", raw_color(), e.raw_estimate(start, end)[::step]))
        for key, color, values in series:
            curve = self._traj_curves.get(key)
            if curve is None:
                if key == "raw":
                    curve = plot.plot(pen=None, symbol="o", symbolSize=4, symbolBrush=pg.mkBrush(color),
                                      symbolPen=None)
                else:
                    p = pen(color, 1.6 if key == "ref" else ALGO_WIDTH)
                    if key == "ref":
                        p.setStyle(Qt.PenStyle.DashLine)
                    curve = plot.plot(pen=p, connect="finite")
                self._traj_curves[key] = curve
            x, y = values[:, ix], values[:, iy]
            if key == "raw":
                ok = np.isfinite(x) & np.isfinite(y)
                x, y = x[ok], y[ok]
            curve.setData(x, y)
            curve.setVisible(key not in self._hidden)

    def _traj_traces(self) -> list[Trace]:
        out = []
        for key, curve in self._traj_curves.items():
            if not curve.isVisible():
                continue
            name = {"ref": "Referans", "raw": "GNSS (ham)"}.get(key) or self.scenario.algorithms[key].name
            color = {"ref": ref_color(), "raw": raw_color()}.get(key) or self.algo_color(key)
            out.append(Trace(curve, name, color, lambda x, y: [f"x = {x:.3f} m", f"y = {y:.3f} m"]))
        return out

    # ---------------------------------------------------------------- 3B yönelim
    def _update_3d_choices(self) -> None:
        current = self.view3d_algo.currentData()
        self.view3d_algo.blockSignals(True)
        self.view3d_algo.clear()
        keys = list(self.engine.algos) if self.engine is not None else []
        for k in keys:
            self.view3d_algo.addItem(line_swatch(self.algo_color(k), Qt.PenStyle.SolidLine, 3),
                                     self.scenario.algorithms[k].name, k)
        pick = current if current in keys else ("ekf" if "ekf" in keys else (keys[0] if keys else None))
        self.view3d_algo.setCurrentIndex(max(self.view3d_algo.findData(pick), 0))
        self.view3d_algo.blockSignals(False)

    def _sync_slider(self) -> None:
        e = self.engine
        if e is None:
            return
        end = max((e.processed(k) for k in e.algos), default=e.start)
        lo = e.history_start
        self.time_slider.blockSignals(True)
        self.time_slider.setRange(lo, max(lo, end - 1))
        if self.follow_latest.isChecked() and not self._play_timer.isActive():
            self.time_slider.setValue(max(lo, end - 1))
        self.time_slider.blockSignals(False)

    def _on_slider(self, value: int) -> None:
        if not self._play_timer.isActive():
            self.follow_latest.setChecked(False)
        self._update_3d()

    def _toggle_play(self, on: bool) -> None:
        if on:
            self.follow_latest.setChecked(False)
            if self.time_slider.value() >= self.time_slider.maximum():
                self.time_slider.setValue(self.time_slider.minimum())
            self._play_timer.start()
        else:
            self._play_timer.stop()

    def _play_step(self) -> None:
        e = self.engine
        if e is None:
            self.play_button.setChecked(False)
            return
        speed = PLAY_SPEEDS[self.play_speed.currentIndex()][1]
        n = self.time_slider.value() + max(1, int(round(e.fs * self._play_timer.interval() / 1000 * speed)))
        if n >= self.time_slider.maximum():
            n = self.time_slider.maximum()
            self.play_button.setChecked(False)
        self.time_slider.blockSignals(True)
        self.time_slider.setValue(n)
        self.time_slider.blockSignals(False)
        self._update_3d()

    def _on_hover(self, x: float | None, _y: float | None) -> None:
        e = self.engine
        if e is None or x is None:
            self._hover_n = None
        else:
            self._hover_n = int(round((x - e.handle.t0) * e.handle.fs))
        if self.scenario.key == "orientation" and not self._play_timer.isActive():
            self._update_3d()

    def _update_3d(self) -> None:
        if self.scenario.key != "orientation" or self.side_card.isHidden():
            return
        e = self.engine
        if e is None or not e.algos:
            self.view3d.set_scene([], ["Füzyonu çalıştırın: yönelim burada 3B gösterilir"])
            return
        end = max((e.processed(k) for k in e.algos), default=e.start)
        if end <= e.start:
            return
        if self._hover_n is not None:
            n = self._hover_n
        elif self.follow_latest.isChecked():
            n = end - 1
        else:
            n = self.time_slider.value()
        n = int(min(max(n, e.history_start), end - 1))
        selected = self.view3d_algo.currentData()
        bodies, lines = [], []
        ref = e.reference(n, n + 1)
        if ref is not None and self.ghost_ref.isChecked() and np.isfinite(ref).all():
            r = np.degrees(ref[0])
            bodies.append(Body("ref", "Referans", ref_color(), quat_from_euler_deg(*r), False))
        main = None
        for key in e.algos:
            values = e.read(key, n, n + 1)
            if not np.isfinite(values).all():
                continue
            q = _quaternions(values)[0]
            solid = key == selected
            if solid:
                main = (key, q, np.degrees(values[0]))
            elif self.ghost_others.isChecked() and key not in self._hidden:
                bodies.append(Body(key, self.scenario.algorithms[key].name, self.algo_color(key), q, False))
        if main is not None:
            key, q, deg = main
            bodies.append(Body(key, self.scenario.algorithms[key].name, self.algo_color(key), q, True))
            angle = 2 * math.degrees(math.acos(min(1.0, abs(float(q[0])))))
            s = math.sqrt(max(1e-12, 1 - float(q[0]) ** 2))
            axis = q[1:] / s * (1 if q[0] >= 0 else -1)
            t = e.handle.t0 + n / e.fs
            lines = [f"●{self.algo_color(key)} {self.scenario.algorithms[key].name}",
                     f"t {t:.3f} s · n {n}",
                     f"q w {q[0]:+.4f}  x {q[1]:+.4f}",
                     f"  y {q[2]:+.4f}  z {q[3]:+.4f}",
                     f"R {deg[0]:+.2f}°  P {deg[1]:+.2f}°  Y {deg[2]:+.2f}°",
                     f"eksen-açı {angle:.2f}° [{axis[0]:+.2f} {axis[1]:+.2f} {axis[2]:+.2f}]"]
            if ref is not None and np.isfinite(ref).all():
                err = np.degrees(fg.errors(self.scenario, np.array([np.radians(deg)]), ref)[0])
                lines.append(f"hata R {err[0]:+.2f}° P {err[1]:+.2f}° Y {err[2]:+.2f}°")
            gauges = (float(deg[0]), float(deg[1]), float(deg[2]))
            warning = ("Gimbal kilidine yakın (|pitch| > 85°): Euler açıları belirsizleşir, kuaterniyon sorunsuz"
                       if abs(deg[1]) > 85 else "")
        else:
            warning, gauges = "", None
        compact = [lines[0], lines[4], lines[-1]] if len(lines) > 5 else lines
        self.view3d.set_scene(bodies, lines, warning, gauges, compact)

    # ================================================================ doğruluk tablosu
    def _update_metrics(self) -> None:
        e = self.engine
        if e is None:
            return
        outs = self.scenario.outputs
        has_ref = bool(e.mapping.reference)
        diag_keys: list[str] = []
        for a in e.algos.values():
            for k in a.diagnostics:
                if k not in diag_keys:
                    diag_keys.append(k)
        columns = ["Algoritma"]
        if has_ref:
            columns += [f"{o.label.split(' (')[0]} RMSE ({o.unit})" for o in outs if o.key in e.mapping.reference]
            columns += ["En büyük |hata|"]
        columns += diag_keys + ["µs/örnek", "İşlenen", "Atlanan"]
        rows = []
        covered = 0.0
        for key, algo in e.algos.items():
            row = {"Algoritma": self.scenario.algorithms[key].name, "_key": key}
            if has_ref:
                rmse, worst, covered = e.metrics(key)
                worst_parts = []
                for j, o in enumerate(outs):
                    if o.key not in e.mapping.reference:
                        continue
                    name = f"{o.label.split(' (')[0]} RMSE ({o.unit})"
                    value = rmse[j] * o.display_scale
                    row[name] = f"{value:.4g}" if np.isfinite(value) else "—"
                    if np.isfinite(worst[j]):
                        worst_parts.append(f"{worst[j] * o.display_scale:.3g}")
                row["En büyük |hata|"] = " / ".join(worst_parts) or "—"
            row.update(algo.diagnostics)
            row["µs/örnek"] = f"{algo.cost_us:.0f}" if algo.cost_us else "—"
            row["İşlenen"] = f"{(e.processed(key) - e.start) / e.fs:.1f} s"
            row["Atlanan"] = f"{algo.overrun / e.fs:.1f} s" if algo.overrun else "0"
            if algo.error:
                row["İşlenen"] = "HATA"
            rows.append(row)
        t = self._theme.tokens
        self.table.setColumnCount(len(columns))
        self.table.setRowCount(len(rows))
        self.table.setHorizontalHeaderLabels(columns)
        for r, row in enumerate(rows):
            for c, col in enumerate(columns):
                item = QTableWidgetItem(row.get(col, ""))
                if c == 0:
                    item.setIcon(line_swatch(self.algo_color(row["_key"]), Qt.PenStyle.SolidLine, 3))
                    item.setForeground(pg.mkColor(self.algo_color(row["_key"])))
                if col == "Atlanan" and row.get(col, "0") != "0":
                    item.setForeground(pg.mkColor(t.warning))
                    item.setToolTip("Algoritma gelen veriye yetişemedi; bu kadar veri işlenmeden atlandı")
                self.table.setItem(r, c, item)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        if has_ref:
            note = (f"RMSE: referansa göre, işlenen tüm veri ({covered:.1f} s; her başlangıçtan sonraki ilk 2 s "
                    "yakınsama hariç). En büyük |hata| çıktı sırasıyla. Tanılar: algoritmanın kendi kestirimleri.")
        else:
            note = "Referans kanalı yok: yalnızca tanılar ve maliyet (hata hesaplanmaz)."
        self.metrics_note.setText(note)

    # ================================================================ inceleme
    def _traces(self, i: int) -> list[Trace]:
        e = self.engine
        if e is None:
            return []
        h = e.handle
        defs = self._series_defs()
        title, unit, _, _ = defs[i]
        out = []
        kind = "hata" if self._mode == "error" else "değer"
        for key in list(e.algos) + ["ref", "raw"]:
            curve = self._curves.get((key, i))
            if curve is None or not curve.isVisible() or curve.xData is None or not len(curve.xData):
                continue
            name = {"ref": "Referans", "raw": "Ham (füzyonsuz)"}.get(key) or self.scenario.algorithms[key].name
            color = {"ref": ref_color(), "raw": raw_color()}.get(key) or self.algo_color(key)

            def describe(x, y, h=h):
                n = int(round((x - h.t0) * h.fs))
                return [f"t = {x:.4f} s  (n = {n})", f"{kind} = {y:+.5g} {unit}".rstrip()]
            out.append(Trace(curve, f"{name} · {title.split(' (')[0]}", color, describe))
        return out

    def _window_stats(self, i: int, x0: float, x1: float) -> list:
        """Görünen aralıkta algoritma başına: referans varsa RMSE / en büyük hata, yoksa ortalama ve std."""
        e = self.engine
        if e is None:
            return []
        h = e.handle
        n0 = max(e.history_start, int(math.ceil((x0 - h.t0) * h.fs)))
        n1 = int(math.floor((x1 - h.t0) * h.fs)) + 1
        n1 = min(n1, max((e.processed(k) for k in e.algos), default=n0))
        if n1 <= n0:
            return []
        title, unit, scale, _ = self._series_defs()[i]
        ref = e.reference(n0, n1)
        columns = []
        for key in e.algos:
            est = e.read(key, n0, n1)
            items = [("Aralık", f"{(n1 - n0) / h.fs:.3g} s ({n1 - n0} örnek)")]
            if self._mode == "quat":
                comp = _quaternions(est)[:, i]
                ok = np.isfinite(comp)
                if ok.any():
                    items += [("ortalama", f"{comp[ok].mean():+.5f}"), ("std", f"{comp[ok].std():.5f}")]
            elif ref is not None and np.isfinite(ref[:, i]).any():
                err = fg.errors(self.scenario, est, ref)[:, i] * scale
                ok = np.isfinite(err)
                if ok.any():
                    items += [("RMSE", f"{np.sqrt(np.mean(err[ok] ** 2)):.4g} {unit}"),
                              ("En büyük |hata|", f"{np.max(np.abs(err[ok])):.4g} {unit}"),
                              ("Ortalama hata (sapma)", f"{np.mean(err[ok]):+.4g} {unit}")]
            else:
                v = est[:, i] * scale
                ok = np.isfinite(v)
                if ok.any():
                    items += [("ortalama", f"{v[ok].mean():.5g} {unit}"), ("std", f"{v[ok].std():.4g} {unit}"),
                              ("en küçük / en büyük", f"{v[ok].min():.5g} / {v[ok].max():.5g} {unit}")]
            columns.append((self.scenario.algorithms[key].name.split(" (")[0], self.algo_color(key), items))
        return columns

    # ================================================================ C kodu
    def show_code(self) -> None:
        from spt.gui.dialogs.fusion_code_dialog import open_fusion_code_dialog
        open_fusion_code_dialog(self)

    def _set_status(self, text: str, state: str = "") -> None:
        t = self._theme.tokens
        color = {"good": t.good, "warning": t.warning, "critical": t.critical}.get(state)
        dot = f'<span style="color:{color}">●</span> ' if color else ""
        self.status.setText(dot + text)


