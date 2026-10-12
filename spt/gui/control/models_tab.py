"""Modeller sekmesi (MATLAB çalışma alanının karşılığı).

- Yeni model ifadeyle: `G = 1/(s^2 + 2*s + 1)`, `tf([1 3], [1 2 5])`, `zpk([-1], [-2 -3], 5)`, `ss(A, B, C, D)`,
  ayrıkta `0.5/(z - 0.8)` (Ts ile); çalışma alanındaki modeller adlarıyla kullanılabilir (`G*C/(1 + G*C)`).
  Ayrıştırma `spt.control.parse.parse_model` ile yapılır.
- Liste: ad, tür (tf / zpk / ss / pid / tanımlanmış / mpc), Ts, G/Ç, derece, kararlılık rozeti.
- Seçili modelin MATLAB biçimli gösterimi, kutup / sıfır tablosu (damp: ζ, ωn, τ; ayrıkta |z|) ve hızlı grafikler
  (adım, kutup-sıfır, Bode genlik / faz — her biri ayrı kartta).
- Dönüşümler (sonuç yeni model olarak eklenir): c2d / d2c (zoh, foh, tustin [+ ön bükme], impulse, matched), tf / zpk /
  ss biçimi, minreal, balred. `.sptm` dosyasına kaydet / dosyadan aç.
"""
from __future__ import annotations

import logging
import math

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDoubleSpinBox,
    QFileDialog,
    QHeaderView,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from spt.control.errors import ControlError
from spt.gui.control import control_plots as cp
from spt.gui.control.workspace import SOURCE_CONVERT, SOURCE_EXPR, ModelWorkspace, as_lti, check_name
from spt.gui.theme import ThemeManager, mono_font
from spt.gui.widgets.common import Card, CollapsibleSection, combo, hbox, icon_button, label

log = logging.getLogger("spt.gui")

C2D_METHODS = [("zoh", "Sıfır dereceli tutucu (zoh)"), ("foh", "Birinci dereceli tutucu (foh)"),
               ("tustin", "Çift doğrusal (tustin)"), ("impulse", "Darbe değişmez (impulse)"),
               ("matched", "Kutup-sıfır eşleme (matched)")]
D2C_METHODS = ("zoh", "tustin", "matched")
COLUMNS = ("Ad", "Tür", "Ts", "G/Ç", "Derece", "Kararlılık")
EXAMPLES = ("Örnekler: 1/(s^2 + 2*s + 1) · tf([1 3], [1 2 5]) · zpk([-1], [-2 -3], 5) · "
            "ss([0 1; -2 -3], [0; 1], [1 0], 0) · 0.5/(z - 0.8) (Ts ile) · exp(-0.2*s)/(s + 1) · "
            "feedback(C*G, 1) · c2d(G, 0.01) (zoh) · d2c(Gd) · series(C, G) · parallel(G1, G2) · minreal(G) · "
            "pid(Kp, Ki, Kd, Tf)")


def _fmt_ts(Ts: float) -> str:
    return "sürekli" if Ts == 0 else f"{Ts:g} s"


def compact_columns(tbl: QTableWidget) -> None:
    """Sayı tablosu sütunları: içeriğe göre, son sütun esner, dar hücre dolgusu — 1366 genişlikte yatay kaydırma
    ve kısaltma (…) olmadan sığar. Analiz sekmesinin pay / stepinfo tabloları da kullanır."""
    head = tbl.horizontalHeader()
    head.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    head.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    head.setDefaultSectionSize(40)   # esneyen son sütun en az bu kadar (Qt varsayılanı 100 tabloyu taşırıyordu)
    head.setStretchLastSection(True)
    tbl.setStyleSheet("QTableView::item { padding: 0 5px; } QHeaderView::section { padding: 6px 5px; }")


def _relayout(card) -> None:
    """Kart başlığındaki okuma metni değişince başlık satırını hemen yeniden yerleştirir: görünüm gizliyken ertelenen
    yerleşim isteği kaybolunca etiket eski genişlikte (kırpık: "tf · süre") kalıyordu."""
    lay = card.layout() if card is not None else None
    if lay is not None:
        lay.invalidate()
        lay.activate()


def _g(x: float, digits: int = 6) -> str:
    """Sayı gösterimi: ∞ / −∞ / — (NaN) ya da `digits` anlamlı basamak."""
    if math.isnan(x):
        return "—"
    if math.isinf(x):
        return "∞" if x > 0 else "−∞"
    return f"{x:.{digits}g}"


def parse_expression(text: str, variables: dict, Ts: float | None):
    """İfadeden model (`spt.control.parse`, güvenli ayrıştırıcı; eval yok).

    Çalışma alanındaki modeller adlarıyla kullanılabilsin diye ayrıştırıcının ad çözümleyicisi genişletilir: bilinen
    ad (s, z, pi …) değilse çalışma alanındaki modelin LTI karşılığı döner. Sonucun modele çevrilmesi `parse_model`
    ile aynı kurallarla (sayı / matris → statik kazanç)."""
    try:
        from spt.control import parse as P
    except ImportError:
        raise ControlError("İfade ayrıştırıcı henüz kullanılamıyor (hazırlanıyor).") from None
    Ts = float(Ts or 0.0)
    base = getattr(P, "_Parser", None)
    if not variables or base is None:
        return P.parse_model(text, Ts=Ts)
    if not isinstance(text, str) or len(text) > P.MAX_LEN:
        raise ControlError(f"İfade çok uzun (en çok {P.MAX_LEN} karakter).")
    if not (Ts >= 0 and math.isfinite(Ts)):
        raise ControlError("Ts ≥ 0 olmalı.")

    class _WorkspaceParser(base):
        def name(self, t):
            model = variables.get(t.text)
            return model if model is not None else super().name(t)

    try:
        v = _WorkspaceParser(text, Ts).parse()
    except RecursionError:
        raise ControlError("İfade çok derin iç içe.") from None
    from spt.control.lti import LTI, tf
    if isinstance(v, LTI):
        return v
    if isinstance(v, (int, float, complex, np.number)):
        if isinstance(v, complex) and v.imag != 0:
            raise ControlError("Sonuç karmaşık bir sayı; model değil.")
        return tf(float(np.real(v)), Ts=Ts)
    if isinstance(v, np.ndarray) and v.size:
        return tf(P._mat(v, 1, "Sonuç"), Ts=Ts)
    raise ControlError("İfade bir model üretmedi.")


def model_order(sys) -> int | None:
    try:
        return len(cp.calc_poles(sys))
    except (ControlError, ImportError, ValueError, np.linalg.LinAlgError):
        return None


def stability(sys) -> tuple[str, str]:
    """(metin, durum) — durum: good / critical / idle. Kararlılık kütüphaneden (isstable, MATLAB tanımı)."""
    try:
        from spt.control.analysis import isstable
        ok = bool(isstable(sys))
    except (ImportError, ControlError, ValueError, np.linalg.LinAlgError):
        return "—", "idle"
    return ("kararlı", "good") if ok else ("kararsız", "critical")


def time_constant(zeta: float, wn: float) -> float:
    """MATLAB damp tablosundaki zaman sabiti τ = 1/(ζ·ωn) (ayrıkta s = log(z)/Ts üzerinden).

    MATLAB R2025b'den ölçülen gösterim: kararsız kökte negatif (ör. 0.5 ± 1.94j → −2), sanal eksendeki ve
    s = 0'daki kökte Inf, ayrıkta z = 0'da (ωn = Inf, ζ = 1) 0."""
    sigma = zeta * wn  # = −Re(s)
    if math.isnan(sigma):
        return math.nan
    if sigma == 0:
        return math.inf
    if math.isinf(sigma):
        return 0.0
    return 1.0 / sigma


def pole_zero_rows(sys) -> list[tuple[str, complex, float, float, float]]:
    """(tür, değer, ζ, ωn, τ) satırları: kutuplar ve sıfırlar, damp sırasıyla (ωn artan)."""
    rows = []
    for kind, values in (("kutup", cp.calc_poles(sys)), ("sıfır", cp.calc_zeros(sys))):
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        wn, zeta, p = cp.calc_damp(values, sys.Ts)
        for w, z, v in zip(wn, zeta, p):
            rows.append((kind, complex(v), float(z), float(w), time_constant(float(z), float(w))))
    return rows


class ModelsTab(QWidget):
    """Model listesi, ifade girişi, dönüşümler, gösterim, kutup / sıfır tablosu, hızlı grafikler."""

    model_selected = pyqtSignal(str)
    analyze_requested = pyqtSignal(str)     # "Analize gönder": Analiz sekmesinde bitki olarak seç

    def __init__(self, theme: ThemeManager, workspace: ModelWorkspace, parent: QWidget | None = None):
        super().__init__(parent)
        self._theme = theme
        self.workspace = workspace
        self._current: str | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        split.addWidget(self._build_left())
        split.addWidget(self._build_right())
        split.setStretchFactor(1, 1)
        split.setSizes([420, 960])   # model listesinin altı sütunu 1366 genişlikte de yatay kaydırmasız sığar
        lay.addWidget(split)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(30)
        self._refresh_timer.timeout.connect(self._show_current)
        ws = workspace
        ws.added.connect(self._on_added)
        ws.removed.connect(lambda *_: self._fill_table())
        ws.changed.connect(self._on_changed)
        ws.renamed.connect(self._on_renamed)
        self._fill_table()

    # ================================================================ sol: liste ve girişler
    def _build_left(self) -> Card:
        card = Card("Modeller")
        card.setMinimumWidth(370)
        card.setMaximumWidth(520)
        self.count_label = label("", "Readout")
        # metni değişen okumalar isteğe bağlı (dar kartta gizlenen) değil: gizliyken değişen metin, yeniden
        # gösterildiğinde başlık satırında eski genişlikte (kırpık) kalıyordu
        card.add_right(self.count_label)
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

        new = CollapsibleSection("Yeni model (ifade)")
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("G")
        self.name_edit.setMaximumWidth(90)
        self.name_edit.setToolTip("Model adı (harfle başlar; harf, rakam, _). Boşsa G, G2 … seçilir.")
        self.expr_edit = QLineEdit()
        self.expr_edit.setPlaceholderText("1/(s^2 + 2*s + 1)")
        self.expr_edit.setToolTip(EXAMPLES)
        self.expr_edit.returnPressed.connect(self.add_from_expression)
        new.form.addRow(hbox(self.name_edit, label("=", "Muted"), self.expr_edit, spacing=6))
        self.expr_ts = QDoubleSpinBox()
        self.expr_ts.setRange(0.0, 1e4)
        self.expr_ts.setDecimals(6)
        self.expr_ts.setSingleStep(0.001)
        self.expr_ts.setSpecialValueText("sürekli (0)")
        self.expr_ts.setSuffix(" s")
        self.expr_ts.setKeyboardTracking(False)
        self.expr_ts.setToolTip("z içeren ifadeler için örnekleme zamanı (0 = sürekli zaman, s değişkeni)")
        new.add_row("Ts", self.expr_ts)
        self.add_button = QPushButton("Ekle")
        self.add_button.setObjectName("Primary")
        self.add_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.add_button.clicked.connect(self.add_from_expression)
        new.form.addRow(self.add_button)
        self.expr_error = label("", "Readout")
        self.expr_error.setWordWrap(True)
        self.expr_error.hide()
        new.form.addRow(self.expr_error)
        hint = label(EXAMPLES, "Muted")
        hint.setWordWrap(True)
        new.form.addRow(hint)
        col.addWidget(new)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setObjectName("ModelTable")
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        head = self.table.horizontalHeader()
        head.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft)
        head.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        head.setStretchLastSection(True)
        self.table.verticalHeader().setDefaultSectionSize(26)
        self.table.setMinimumHeight(170)
        self.table.itemSelectionChanged.connect(self._on_selection)
        self.table.itemDoubleClicked.connect(lambda *_: self.rename_dialog())
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._context_menu)
        col.addWidget(self.table, 1)

        self.rename_button = icon_button("ph.pencil-simple", "Yeniden adlandır (çift tık)", object_name="IconButton",
                                         size=16)
        self.rename_button.clicked.connect(self.rename_dialog)
        self.copy_button = icon_button("ph.copy", "Çoğalt", object_name="IconButton", size=16)
        self.copy_button.clicked.connect(self.duplicate)
        self.delete_button = icon_button("ph.trash", "Sil", object_name="IconButton", size=16)
        self.delete_button.clicked.connect(self.delete_selected)
        self.analyze_button = QPushButton("Analize gönder")
        self.analyze_button.setToolTip("Analiz sekmesinde bitki (G) olarak seç")
        self.analyze_button.clicked.connect(lambda: self._current and self.analyze_requested.emit(self._current))
        col.addWidget(hbox(self.rename_button, self.copy_button, self.delete_button, self.analyze_button,
                           spacing=6, stretch=True))

        conv = CollapsibleSection("Dönüştür (sonuç yeni model)")
        self.c2d_ts = QDoubleSpinBox()
        self.c2d_ts.setRange(1e-6, 1e4)
        self.c2d_ts.setDecimals(6)
        self.c2d_ts.setValue(0.01)
        self.c2d_ts.setSuffix(" s")
        self.c2d_ts.setKeyboardTracking(False)
        self.c2d_ts.setToolTip("Ayrıklaştırmada örnekleme zamanı")
        conv.add_row("Ts", self.c2d_ts)
        self.c2d_method = combo([])
        for key, text in C2D_METHODS:
            self.c2d_method.addItem(text, key)
        self.c2d_method.currentIndexChanged.connect(self._sync_prewarp)
        conv.add_row("Yöntem", self.c2d_method)
        self.prewarp = QDoubleSpinBox()
        self.prewarp.setRange(0.0, 1e7)
        self.prewarp.setDecimals(4)
        self.prewarp.setSuffix(" rad/s")
        self.prewarp.setSpecialValueText("yok")
        self.prewarp.setToolTip("Tustin ön bükme (prewarp) frekansı: bu frekansta sürekli ve ayrık cevap eşleşir")
        conv.add_row("Ön bükme", self.prewarp)
        b_c2d = QPushButton("Ayrıklaştır (c2d)")
        b_c2d.clicked.connect(self.c2d_selected)
        b_d2c = QPushButton("Sürekliye (d2c)")
        b_d2c.clicked.connect(self.d2c_selected)
        conv.form.addRow(hbox(b_c2d, b_d2c, spacing=6))
        self.form_combo = combo(["tf", "zpk", "ss"])
        self.form_combo.setToolTip("Gösterim biçimi: transfer fonksiyonu, sıfır-kutup-kazanç, durum uzayı")
        b_form = QPushButton("Biçime çevir")
        b_form.clicked.connect(self.convert_selected)
        conv.add_row("Biçim", hbox(self.form_combo, b_form, spacing=6))
        b_min = QPushButton("Sadeleştir (minreal)")
        b_min.setToolTip("Sadeleşen kutup-sıfır çiftlerini / denetlenemez-gözlenemez durumları at")
        b_min.clicked.connect(self.minreal_selected)
        conv.form.addRow(b_min)
        self.red_order = QSpinBox()
        self.red_order.setRange(1, 200)
        self.red_order.setValue(2)
        self.red_method = combo(["MatchDC", "Truncate"])
        self.red_method.setToolTip("MatchDC: DC kazancı korunur (MATLAB varsayılanı); Truncate: doğrudan kesme")
        b_red = QPushButton("Derece indir (balred)")
        b_red.clicked.connect(self.balred_selected)
        conv.add_row("Derece", hbox(self.red_order, self.red_method, spacing=6))
        conv.form.addRow(b_red)
        self.conv_status = label("", "Readout")
        self.conv_status.setWordWrap(True)
        self.conv_status.hide()
        conv.form.addRow(self.conv_status)
        col.addWidget(conv)
        self._sync_prewarp()
        return card

    def _sync_prewarp(self, *_) -> None:
        self.prewarp.setEnabled(self.c2d_method.currentData() == "tustin")

    # ================================================================ sağ: gösterim, tablo, grafikler
    def _build_right(self) -> QWidget:
        t = self._theme
        vsplit = QSplitter(Qt.Orientation.Vertical)
        vsplit.setChildrenCollapsible(False)
        top = QSplitter(Qt.Orientation.Horizontal)
        top.setChildrenCollapsible(False)

        self.display_card = Card("Gösterim")
        self.display_kind = label("", "Readout")
        self.display_card.add_right(self.display_kind)
        self.display = QPlainTextEdit()
        self.display.setReadOnly(True)
        self.display.setFont(mono_font(12))
        self.display.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.display.setMinimumHeight(110)
        self.display_card.body.addWidget(self.display, 1)
        top.addWidget(self.display_card)

        self.pz_card = Card("Kutuplar ve Sıfırlar")
        self.pz_summary = label("", "Readout")
        self.pz_card.add_right(self.pz_summary)
        self.pz_table = QTableWidget(0, 6)
        self.pz_table.setObjectName("PoleTable")
        self.pz_table.verticalHeader().setVisible(False)
        self.pz_table.setShowGrid(True)
        self.pz_table.setAlternatingRowColors(True)
        self.pz_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.pz_table.setSelectionMode(QAbstractItemView.SelectionMode.ContiguousSelection)
        compact_columns(self.pz_table)
        self.pz_table.verticalHeader().setDefaultSectionSize(24)
        self.pz_card.body.addWidget(self.pz_table, 1)
        top.addWidget(self.pz_card)
        top.setSizes([470, 530])
        vsplit.addWidget(top)

        plots = QSplitter(Qt.Orientation.Vertical)
        plots.setChildrenCollapsible(False)
        row1 = QSplitter(Qt.Orientation.Horizontal)
        row1.setChildrenCollapsible(False)
        self.step_plot = cp.StepPlot(t, "Adım Cevabı")
        self.pzmap_plot = cp.PZMapPlot(t)
        row1.addWidget(self.step_plot)
        row1.addWidget(self.pzmap_plot)
        row1.setSizes([600, 400])
        self.bode_plot = cp.BodePlot(t)
        self.bode_plot.splitter.setOrientation(Qt.Orientation.Horizontal)
        plots.addWidget(row1)
        plots.addWidget(self.bode_plot)
        vsplit.addWidget(plots)
        vsplit.setStretchFactor(1, 2)
        vsplit.setSizes([260, 560])
        return vsplit

    def detachable_cards(self) -> list[tuple[str, str, Card]]:
        return [("ctl_models_step", "Modeller · Adım Cevabı", self.step_plot),
                ("ctl_models_pz", "Modeller · Kutup-Sıfır", self.pzmap_plot),
                ("ctl_models_bode_mag", "Modeller · Bode Genlik", self.bode_plot.mag_card),
                ("ctl_models_bode_phase", "Modeller · Bode Faz", self.bode_plot.phase_card)]

    # ================================================================ liste
    @property
    def current(self) -> str | None:
        return self._current

    def _fill_table(self) -> None:
        names = self.workspace.names()
        keep = self._current if self._current in names else (names[-1] if names else None)
        tbl = self.table
        tbl.blockSignals(True)
        tbl.setRowCount(len(names))
        tok = self._theme.tokens
        colors = {"good": tok.good, "critical": tok.critical, "idle": tok.text_muted}
        for r, name in enumerate(names):
            entry = self.workspace.entry(name)
            sys = as_lti(entry.model)
            Ts = getattr(entry.model, "Ts", getattr(sys, "Ts", 0.0))
            io = f"{sys.noutputs}×{sys.ninputs}" if sys is not None else "—"
            order = model_order(sys) if sys is not None else None
            stab, state = stability(sys) if sys is not None else ("—", "idle")
            values = (name, entry.kind, _fmt_ts(float(Ts or 0.0)), io, "—" if order is None else str(order),
                      f"● {stab}")
            for c, text in enumerate(values):
                item = QTableWidgetItem(text)
                if c == 0:
                    item.setData(Qt.ItemDataRole.UserRole, name)
                    tip = [name, entry.kind]
                    if entry.source:
                        tip.append(f"kaynak: {entry.source}")
                    if entry.note:
                        tip.append(entry.note)
                    if entry.expr:
                        tip.append(f"ifade: {entry.expr}")
                    item.setToolTip("\n".join(tip))
                if c == 5:
                    item.setForeground(QColor(colors[state]))
                    item.setToolTip("Asimptotik kararlılık (MATLAB isstable): sürekli zamanda tüm kutuplar Re(p) < 0, "
                                    "ayrıkta |p| < 1; sınırdaki kutup (ör. integratör) kararsız sayılır")
                tbl.setItem(r, c, item)
        tbl.blockSignals(False)
        n = len(names)
        self.count_label.setText(f"{n} model" if n else "")
        _relayout(self.count_label.parentWidget())
        if keep is not None:
            self.select(keep)
        else:
            self._current = None
            self._show_current()

    def select(self, name: str) -> None:
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == name:
                self.table.selectRow(r)
                if self._current != name:
                    self._current = name
                    self._refresh_timer.start()
                return

    def _on_selection(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        item = self.table.item(rows[0].row(), 0)
        name = item.data(Qt.ItemDataRole.UserRole) if item else None
        if name and name != self._current:
            self._current = name
            self.model_selected.emit(name)
            self._refresh_timer.start()

    def _on_added(self, name: str) -> None:
        self._current = name
        self._fill_table()
        self._refresh_timer.start()

    def _on_changed(self, name: str) -> None:
        self._fill_table()
        if name == self._current:
            self._refresh_timer.start()

    def _on_renamed(self, old: str, new: str) -> None:
        if self._current == old:
            self._current = new
        self._fill_table()
        self._refresh_timer.start()

    def _context_menu(self, pos) -> None:
        if self._current is None:
            return
        menu = QMenu(self)
        menu.addAction("Yeniden adlandır…", self.rename_dialog)
        menu.addAction("Çoğalt", self.duplicate)
        menu.addAction("Analize gönder", lambda: self.analyze_requested.emit(self._current))
        menu.addSeparator()
        menu.addAction("Sil", self.delete_selected)
        menu.exec(self.table.viewport().mapToGlobal(pos))

    # ================================================================ eylemler
    def _set_error(self, widget, text: str) -> None:
        tok = self._theme.tokens
        widget.setText(f'<span style="color:{tok.critical}">●</span> {text}' if text else "")
        widget.setVisible(bool(text))

    def _set_ok(self, widget, text: str) -> None:
        tok = self._theme.tokens
        widget.setText(f'<span style="color:{tok.good}">●</span> {text}' if text else "")
        widget.setVisible(bool(text))

    def add_from_expression(self) -> str | None:
        expr = self.expr_edit.text().strip()
        if not expr:
            self._set_error(self.expr_error, "İfade boş.")
            return None
        name = self.name_edit.text().strip()
        try:
            if name:
                check_name(name)
            Ts = self.expr_ts.value() or None
            model = parse_expression(expr, self.workspace.variables(), Ts)
            final = self.workspace.add(name or self.workspace.unique_name("G"), model, SOURCE_EXPR, expr=expr,
                                       replace=bool(name))
        except ControlError as exc:
            self._set_error(self.expr_error, str(exc))
            return None
        self._set_ok(self.expr_error, f"{final} eklendi.")
        self.name_edit.clear()
        return final

    def rename_dialog(self) -> None:
        from PyQt6.QtWidgets import QInputDialog

        old = self._current
        if old is None:
            return
        new, ok = QInputDialog.getText(self, "Yeniden adlandır", "Yeni ad:", text=old)
        if not ok or not new.strip() or new.strip() == old:
            return
        try:
            self.workspace.rename(old, new.strip())
        except ControlError as exc:
            self._set_error(self.conv_status, str(exc))

    def duplicate(self) -> None:
        if self._current is None:
            return
        import copy

        e = self.workspace.entry(self._current)
        try:
            model = copy.deepcopy(e.model)  # türü korunur (PID, tanımlanmış model …)
        except (TypeError, copy.Error):
            model = e.model
        self.workspace.add(self.workspace.unique_name(self._current), model, e.source, e.note, e.expr)

    def delete_selected(self) -> None:
        if self._current is not None:
            self.workspace.remove(self._current)

    def _selected_lti(self):
        if self._current is None:
            raise ControlError("Önce listeden bir model seçin.")
        return self._current, self.workspace.lti(self._current)

    def _transform(self, fn, suffix: str, what: str) -> str | None:
        try:
            name, sys = self._selected_lti()
            out = fn(sys)
            final = self.workspace.add(self.workspace.unique_name(f"{name}_{suffix}"), out, SOURCE_CONVERT,
                                       note=f"{name} → {what}")
        except ControlError as exc:
            self._set_error(self.conv_status, str(exc))
            return None
        except ImportError:
            self._set_error(self.conv_status, "Bu dönüşümün hesap modülü henüz kullanılamıyor.")
            return None
        self._set_ok(self.conv_status, f"{final} = {what}")
        return final

    def c2d_selected(self) -> str | None:
        Ts, method = self.c2d_ts.value(), self.c2d_method.currentData()
        w = self.prewarp.value() if method == "tustin" and self.prewarp.value() > 0 else None

        def fn(sys):
            from spt.control.discretize import c2d
            if sys.Ts > 0:
                raise ControlError("Model zaten ayrık; sürekli bir model seçin (ya da önce sürekliye çevirin).")
            return c2d(sys, Ts, method=method, prewarp=w) if w else c2d(sys, Ts, method=method)

        extra = f", ön bükme {w:g} rad/s" if w else ""
        return self._transform(fn, "d", f"ayrık ({method}, Ts = {Ts:g} s{extra})")

    def d2c_selected(self) -> str | None:
        method = self.c2d_method.currentData()
        if method not in D2C_METHODS:
            method = "zoh"
        w = self.prewarp.value() if method == "tustin" and self.prewarp.value() > 0 else None

        def fn(sys):
            from spt.control.discretize import d2c
            if sys.Ts == 0:
                raise ControlError("Model zaten sürekli; ayrık bir model seçin.")
            return d2c(sys, method=method, prewarp=w) if w else d2c(sys, method=method)

        return self._transform(fn, "c", f"sürekli ({method})")

    def convert_selected(self) -> str | None:
        form = self.form_combo.currentText()

        def fn(sys):
            from spt.control import lti
            return getattr(lti, form)(sys)

        return self._transform(fn, form, f"{form} biçimi")

    def minreal_selected(self) -> str | None:
        def fn(sys):
            from spt.control.convert import minreal
            return minreal(sys)

        return self._transform(fn, "min", "en küçük gerçekleme (minreal)")

    def balred_selected(self) -> str | None:
        order, method = self.red_order.value(), self.red_method.currentText()

        def fn(sys):
            from spt.control.convert import balred
            return balred(sys, order, method=method)

        return self._transform(fn, f"r{order}", f"{order}. dereceye indirgenmiş ({method})")

    # ---------------------------------------------------------------- dosya
    def _settings(self):
        from spt.gui.settings import app_settings
        return app_settings()

    def open_file(self) -> None:
        s = self._settings()
        path, _ = QFileDialog.getOpenFileName(self, "Model çalışma alanı aç", str(s.value("control/dir", "")),
                                              "Model çalışma alanı (*.sptm);;Tüm dosyalar (*)")
        if path:
            from pathlib import Path
            s.setValue("control/dir", str(Path(path).parent))
            self.load_path(path)

    def load_path(self, path) -> list[str]:
        try:
            added, failed = self.workspace.load(path)
        except ControlError as exc:
            self._set_error(self.conv_status, str(exc))
            return []
        msg = f"{len(added)} model eklendi" + (f"; okunamayan: {'; '.join(failed)}" if failed else "")
        (self._set_error if failed and not added else self._set_ok)(self.conv_status, msg)
        return added

    def save_file(self) -> None:
        if not len(self.workspace):
            self._set_error(self.conv_status, "Kaydedilecek model yok.")
            return
        s = self._settings()
        from pathlib import Path
        start = Path(str(s.value("control/dir", ""))) / "modeller.sptm"
        path, _ = QFileDialog.getSaveFileName(self, "Model çalışma alanını kaydet", str(start),
                                              "Model çalışma alanı (*.sptm)")
        if path:
            if not path.lower().endswith(".sptm"):
                path += ".sptm"
            s.setValue("control/dir", str(Path(path).parent))
            self.save_path(path)

    def save_path(self, path) -> list[str]:
        try:
            skipped = self.workspace.save(path)
        except (OSError, TypeError, ValueError) as exc:
            self._set_error(self.conv_status, f"Kaydedilemedi: {exc}")
            return []
        msg = f"{len(self.workspace) - len(skipped)} model kaydedildi"
        if skipped:
            msg += f"; dosyaya yazılamayan: {', '.join(skipped)}"
        self._set_ok(self.conv_status, msg)
        return skipped

    # ================================================================ seçili modelin gösterimi
    def _show_current(self) -> None:
        self._update_current()
        for card in (self.display_card, self.pz_card):
            _relayout(card)

    def _update_current(self) -> None:
        name = self._current
        if name is None or name not in self.workspace:
            self.display.setPlainText("Model yok. Soldaki alana bir ifade yazıp “Ekle”ye basın.\n\n" + EXAMPLES)
            self.display_kind.setText("")
            self.pz_table.setRowCount(0)
            self.pz_summary.setText("")
            for plot in (self.step_plot, self.pzmap_plot, self.bode_plot):
                plot.set_series([])
            return
        entry = self.workspace.entry(name)
        sys = as_lti(entry.model)
        text = str(entry.model)
        if sys is not None and sys is not entry.model:
            text += f"\n\nAnalizde kullanılan doğrusal karşılık:\n{sys}"
        lines = [text]
        if entry.source or entry.note or entry.expr:
            lines.append("")
            if entry.expr:
                lines.append(f"İfade: {entry.expr}")
            if entry.source:
                lines.append(f"Kaynak: {entry.source}")
            if entry.note:
                lines.append(f"Not: {entry.note}")
        self.display.setPlainText("\n".join(lines))
        self.display_kind.setText(f"{entry.kind} · {_fmt_ts(float(getattr(sys, 'Ts', 0.0) or 0.0))}")
        self._fill_pz(sys)
        series = [cp.Series(name, sys, cp.PALETTE[0])] if sys is not None else []
        for plot in (self.step_plot, self.pzmap_plot, self.bode_plot):
            plot.set_series(series)

    def _fill_pz(self, sys) -> None:
        tbl = self.pz_table
        tbl.setRowCount(0)
        if sys is None:
            self.pz_summary.setText("")
            return
        discrete = sys.Ts > 0
        # MATLAB damp tablosunun sütunları (ayrıkta "Magnitude" = |z| ek sütunu)
        heads = ["Tür", "Değer"] + (["|z|"] if discrete else []) + ["Sönüm ζ", "ωn (rad/s)", "τ (s)"]
        tips = ["kutup / sıfır", "kök (karmaşık)"] + (["büyüklük |z| (kararlılık: < 1)"] if discrete else []) + [
            "sönüm oranı ζ = −Re(s)/|s|" + (" (s = ln z / Ts)" if discrete else ""),
            "doğal frekans ωn = |s|" + (" (s = ln z / Ts)" if discrete else ""),
            "zaman sabiti τ = 1/(ζ·ωn); kararsız kökte negatif (MATLAB damp gösterimi)"]
        tbl.setColumnCount(len(heads))
        tbl.setHorizontalHeaderLabels(heads)
        for c, tip in enumerate(tips):
            tbl.horizontalHeaderItem(c).setToolTip(tip)
        try:
            rows = pole_zero_rows(sys)
        except (ControlError, ImportError, ValueError, np.linalg.LinAlgError) as exc:
            self.pz_summary.setText(f"hesaplanamadı: {exc}" if not isinstance(exc, ImportError) else
                                    "hesap modülü hazırlanıyor")
            return
        tbl.setRowCount(len(rows))
        muted = QColor(self._theme.tokens.text_muted)
        for r, (kind, v, zeta, wn, tau) in enumerate(rows):
            val = f"{v.real:.6g}" if v.imag == 0 else f"{v.real:.6g} {'+' if v.imag > 0 else '−'} {abs(v.imag):.6g}j"
            cells = [kind, val] + ([_g(abs(v))] if discrete else []) + [_g(zeta, 4), _g(wn), _g(tau)]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c == 0:
                    item.setForeground(muted)
                tbl.setItem(r, c, item)
        npole = sum(1 for r in rows if r[0] == "kutup")
        self.pz_summary.setText(f"{npole} kutup · {len(rows) - npole} sıfır")

    # ================================================================ ayarlar
    def save_state(self, s) -> None:
        s.setValue("control/c2d_ts", self.c2d_ts.value())
        s.setValue("control/c2d_method", self.c2d_method.currentData())
        s.setValue("control/models_current", self._current or "")

    def restore_state(self, s) -> None:
        try:
            self.c2d_ts.setValue(float(s.value("control/c2d_ts", self.c2d_ts.value())))
        except (TypeError, ValueError):
            pass
        i = self.c2d_method.findData(str(s.value("control/c2d_method", "zoh")))
        if i >= 0:
            self.c2d_method.setCurrentIndex(i)
        cur = str(s.value("control/models_current", ""))
        if cur in self.workspace:
            self.select(cur)
