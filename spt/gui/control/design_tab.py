"""Kontrolcü Tasarımı sekmesi (MATLAB Control System Designer + PID Tuner karşılığı).

Sol panel kipleri (üstteki sekme çubuğu):
  Kompansatör   mimari (tek çevrim / ön filtreli / ileri beslemeli / kademeli iç-dış çevrim), bitki seçimi, zaman
                alanı (sürekli ya da c2d ile ayrık; sürekli eşdeğer gösterilir), kompansatör düzenleyici (kazanç,
                gerçel / karmaşık kutup-sıfır, integratör, türevleyici, öncül / ardıl / çentik; tablo + grafikte
                sürükleme), otomatik öncül-ardıl ve sönüm için kazanç yardımcıları.
  PID           PID Tuner: tür, biçim, ayrıklaştırma formülleri, "cevap süresi" ve "geçici durum davranışı"
                kaydırıcıları → pidtune hedef ωc / faz payı; parametre ve performans tablosu; karşılaştırma tasarımı.
  Durum uzayı   kutup yerleştirme, LQR / LQI (Bryson), gözlemci, Kalman, LQG (`design_modern`).
  MPC           ufuklar, ağırlıklar, kısıtlar, referansla benzetim, C kodu (`design_mpc`).

Klasik görünüm (sağ): etkileşimli kök yer eğrisi (kapalı çevrim kutbunu sürükle → kazanç; kompansatör kutup / sıfırını
sürükle), açık çevrim Bode düzenleyicisi (genliği sürükle → kazanç; paylar işaretli), referans izleme (r → y), giriş
bozucusu reddi (d → y), kontrol çabası (r → u) ve pay / adım ölçütü / kapalı çevrim kutup tabloları. Her değişiklik
kısa bir gecikmeyle tüm görünümü günceller; kök yer eğrisi, PID ayarı ve MPC benzetimi arka planda hesaplanır.

Tüm hesaplar `spt.control` kütüphanesindendir (bkz. `design_core`); bu modül yalnız arayüzdür.
"""
from __future__ import annotations

import json
import logging
import math

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QMenu,
    QPushButton,
    QSlider,
    QSplitter,
    QStackedWidget,
    QTabBar,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from spt.control.errors import ControlError
from spt.gui.control import control_plots as cp
from spt.gui.control import design_core as dc
from spt.gui.control.design_widgets import (
    BackgroundCall,
    BodeEditor,
    DesignStepPlot,
    RootLocusEditor,
    capped_tfinal,
    fill_table,
    make_table,
    num,
    responses,
)
from spt.gui.control.design_widgets import mono_label as _mono
from spt.gui.control.design_widgets import scroll_area as _scroll
from spt.gui.control.design_widgets import spin_box as _spin
from spt.gui.control.models_tab import C2D_METHODS
from spt.gui.control.workspace import SOURCE_DESIGN, ModelWorkspace
from spt.gui.theme import ThemeManager
from spt.gui.widgets.common import Card, CollapsibleSection, combo, label

log = logging.getLogger("spt.gui")

MODES = (("comp", "Kompansatör"), ("pid", "PID"), ("modern", "Durum uzayı"), ("mpc", "MPC"))
INTEGRATOR = "__integrator__"
NONE = None
# Klasik görünümde her seri tek renk (aynı görünümde tekrar yok; PALETTE ölçümü control_plots'ta)
COLORS = {"T": cp.PALETTE[0], "L": cp.PALETTE[1], "GS": cp.PALETTE[2], "KS": cp.PALETTE[3], "comp": cp.PALETTE[4],
          "T_base": cp.PALETTE[5], "T2": cp.PALETTE[6], "T_cont": cp.PALETTE[7]}
NAMES = {"L": "L = C·G (açık çevrim)", "T": "T: r → y", "GS": "d → y (giriş bozucusu)", "KS": "r → u (kontrol)",
         "T_base": "T: karşılaştırma", "T2": "T2: iç çevrim ω_ref → ω", "T_cont": "T: sürekli eşdeğer"}
PLOTS = (("rlocus", "Kök yer eğrisi"), ("bode", "Bode"), ("step", "Referans izleme"), ("dist", "Bozucu reddi"),
         ("effort", "Kontrol çabası"))
MARGIN_COLUMNS = (("Çevrim", "Düzenlenen kompansatörün açık çevrimi"),
                  ("Gm (dB)", "Kazanç payı (MATLAB margin): faz −180° geçişinde 1/|L|"),
                  ("ω Gm", "Faz geçiş frekansı (rad/s)"),
                  ("Pm (°)", "Faz payı: |L| = 1 frekansında 180° + ∠L (en küçük)"),
                  ("ω Pm", "Kazanç geçiş frekansı (rad/s)"),
                  ("Kapalı çevrim", "Tüm kapalı çevrimin kararlılığı (kutuplarla)"))
STEP_COLUMNS = (("Cevap", "Kapalı çevrim cevabı"), ("Yükselme (s)", "%10 → %90 (stepinfo)"),
                ("Yerleşme (s)", "±%2 bandı (stepinfo)"), ("Aşım %", "Aşım (stepinfo)"), ("Tepe", "Tepe değeri"),
                ("t tepe (s)", "Tepe zamanı"), ("Kalıcı", "Kalıcı durum (DC kazancı)"))
POLE_COLUMNS = (("Kapalı çevrim kutbu", "1 + L = 0 kökleri (düzenlenen çevrim)"), ("ζ", "Sönüm oranı (damp)"),
                ("ωn (rad/s)", "Doğal frekans (damp)"))


# ============================================================================ kompansatör paneli
class CompensatorPanel(QWidget):
    """Kompansatör düzenleyici: kazanç, öğe tablosu (kök başına satır; konum düzenlenebilir), öğe ekleme, öncül /
    ardıl / çentik ve otomatik yardımcılar. Değişiklikleri sekmeye `changed` ile bildirir."""

    changed = pyqtSignal()

    def __init__(self, tab: "DesignTab"):
        super().__init__()
        self.tab = tab
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        sec = CollapsibleSection("Kompansatör C")
        self.which = label("", "Muted")
        sec.form.addRow(self.which)
        self.text = _mono()
        sec.form.addRow(self.text)
        self.cont_text = _mono()
        self.cont_text.setToolTip("Ayrık kompansatörün sürekli eşdeğeri (d2c, seçilen yöntem; foh / impulse için "
                                  "tustin)")
        sec.form.addRow(self.cont_text)
        self.gain = _spin(-1e12, 1e12, 1.0, 6, "", "ZPK kazancı K: C = K·Π(x − z)/Π(x − p). Kök yer eğrisi "
                                                    "parametresi.")
        self.gain.valueChanged.connect(self._gain_edited)
        sec.add_row("Kazanç K", self.gain)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Öğe", "Konum", "ζ", "ωn (rad/s)"])
        self.table.horizontalHeaderItem(1).setToolTip("Kökün konumu (sürekli: s düzlemi, ayrık: z düzlemi); çift "
                                                      "tıklayıp düzenleyin (ör. −2, −1+2j). Karmaşık çiftin eşleniği "
                                                      "kendiliğinden eklenir.")
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.setMinimumHeight(120)
        self.table.itemChanged.connect(self._cell_edited)
        sec.form.addRow(self.table)
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        add = QToolButton()
        add.setText("Öğe ekle ▾")
        add.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(add)
        for kind in ("real_pole", "real_zero", "cplx_pole", "cplx_zero", "integrator", "differentiator"):
            menu.addAction(dc.ELEMENT_KINDS[kind][0], lambda k=kind: self.add_element(k))
        add.setMenu(menu)
        rl.addWidget(add)
        rm = QPushButton("Sil")
        rm.setToolTip("Seçili satırın öğesini sil")
        rm.clicked.connect(self.remove_selected)
        rl.addWidget(rm)
        reset = QPushButton("Sıfırla")
        reset.setToolTip("C = 1")
        reset.clicked.connect(self.reset)
        rl.addWidget(reset)
        rl.addStretch(1)
        sec.form.addRow(row)
        lay.addWidget(sec)

        ll = CollapsibleSection("Öncül / ardıl / çentik", expanded=False)
        self.ll_phase = _spin(-89.0, 89.0, 45.0, 2, "°", "Öncülde en büyük faz katkısı (0…90°), ardılda (−90…0°)")
        ll.add_row("Faz φ", self.ll_phase)
        self.ll_w = _spin(1e-6, 1e9, 1.0, 5, " rad/s", "Faz katkısının en büyük olduğu frekans ωm")
        ll.add_row("ωm", self.ll_w)
        b = QWidget()
        bl = QHBoxLayout(b)
        bl.setContentsMargins(0, 0, 0, 0)
        for text, kind in (("Öncül ekle", "lead"), ("Ardıl ekle", "lag")):
            btn = QPushButton(text)
            btn.clicked.connect(lambda _=False, k=kind: self.add_leadlag(k))
            bl.addWidget(btn)
        ll.form.addRow(b)
        self.n_w0 = _spin(1e-6, 1e9, 10.0, 5, " rad/s", "Çentik merkez frekansı ω0")
        ll.add_row("Çentik ω0", self.n_w0)
        self.n_depth = _spin(-200.0, -0.1, -20.0, 2, " dB", "ω0'daki kazanç (dB, negatif)")
        ll.add_row("Derinlik", self.n_depth)
        self.n_width = _spin(1e-6, 1e9, 2.0, 5, " rad/s", "Zayıflamanın derinliğin yarısını aştığı bant genişliği")
        ll.add_row("Genişlik", self.n_width)
        nb = QPushButton("Çentik ekle")
        nb.clicked.connect(self.add_notch)
        ll.form.addRow(nb)
        lay.addWidget(ll)

        auto = CollapsibleSection("Otomatik yardımcılar", expanded=False)
        self.a_pm = _spin(1.0, 89.0, 50.0, 2, "°", "Hedef faz payı")
        auto.add_row("Hedef Pm", self.a_pm)
        self.a_wc = _spin(1e-6, 1e9, 1.0, 5, " rad/s", "Hedef kazanç geçiş frekansı")
        auto.add_row("Hedef ωc", self.a_wc)
        ab = QPushButton("Öncül / ardıl tasarla (leadlag)")
        ab.setToolTip("C'yi tek öncül / ardıl kompansatörle değiştirir: |L(jωc)| = 1, ∠L = −180° + Pm tam")
        ab.clicked.connect(self.auto_leadlag)
        auto.form.addRow(ab)
        self.a_zeta = _spin(0.0, 0.999, 0.7, 3, "", "İstenen sönüm oranı ζ (kök yer eğrisinin sabit-ζ ışınıyla "
                                                     "kesişimi)")
        auto.add_row("Hedef ζ", self.a_zeta)
        zb = QPushButton("Sönüm için kazanç")
        zb.setToolTip("K'yi, kapalı çevrim kutup çiftinin sönümü ζ olacak şekilde seçer (gain_for_damping)")
        zb.clicked.connect(self.gain_for_damping)
        auto.form.addRow(zb)
        lay.addWidget(auto)

        out = QWidget()
        ol = QHBoxLayout(out)
        ol.setContentsMargins(0, 4, 0, 0)
        self.add_btn = QToolButton()
        self.add_btn.setText("Modellere ekle ▾")
        self.add_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        m = QMenu(self.add_btn)
        m.addAction("Kompansatör(ler)", lambda: tab.add_to_workspace("C"))
        m.addAction("Kapalı çevrim T (r → y)", lambda: tab.add_to_workspace("T"))
        m.addAction("Açık çevrim L", lambda: tab.add_to_workspace("L"))
        self.add_btn.setMenu(m)
        ol.addWidget(self.add_btn)
        code = QPushButton("C Kodu Üret…")
        code.setToolTip("Düzenlenen kompansatör için MCU C kodu (ayrık tasarım gerekir)")
        code.clicked.connect(tab.generate_code)
        ol.addWidget(code)
        ol.addStretch(1)
        lay.addWidget(out)
        self._rows: list[tuple[int, str, int]] = []
        self._filling = False

    # ---------------------------------------------------------------- görünüm
    def refresh(self) -> None:
        tab = self.tab
        comp = tab.comp
        key = tab.edit_key()
        names = {"C": "Kompansatör C", "C1": "Dış çevrim kompansatörü C1 (açı)", "C2": "İç çevrim kompansatörü C2 "
                 "(açısal hız)"}
        pid = " · PID yapısında (PID ayarlayıcısından)" if comp.pid is not None else ""
        self.which.setText(names.get(key, key) + pid)
        try:
            self.text.setText(str(comp.zpk()).strip("\n"))
        except ControlError as exc:
            self.text.setText(str(exc))
        if comp.discrete:
            try:
                cc, m = dc.cont_equivalent(comp.zpk(), tab.method())
                self.cont_text.setText(f"Sürekli eşdeğer (d2c, {m}):\n" + str(cc).strip("\n"))
            except ControlError as exc:
                self.cont_text.setText(f"Sürekli eşdeğer hesaplanamadı: {exc}")
            self.cont_text.show()
        else:
            self.cont_text.hide()
        self.gain.blockSignals(True)
        self.gain.setValue(comp.K)
        self.gain.blockSignals(False)
        self._fill_table(comp)

    def _fill_table(self, comp: dc.Compensator) -> None:
        self._filling = True
        rows = []
        for i, e in enumerate(comp.elements):
            for which, roots in (("z", e.zeros), ("p", e.poles)):
                for j, v in enumerate(roots):
                    rows.append((i, which, j, e, v))
        self.table.setRowCount(len(rows))
        self._rows = [(i, which, j) for i, which, j, _e, _v in rows]
        damp = {}
        try:
            vals = [complex(v) for *_x, v in rows]
            for (p, z, w) in dc.damp_rows(vals, comp.Ts):
                damp[complex(p)] = (z, w)
        except ControlError:
            pass
        for r, (i, which, j, e, v) in enumerate(rows):
            part = "" if e.kind not in ("lead", "lag", "notch") else (" · sıfır" if which == "z" else " · kutup")
            cells = [e.label + part, dc.fmt_complex(v, 6)]
            zw = damp.get(complex(v)) or damp.get(complex(v).conjugate())
            cells += [num(zw[0], 4), num(zw[1], 5)] if zw else ["—", "—"]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c != 1 or not e.movable:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(r, c, item)
        self._filling = False

    # ---------------------------------------------------------------- düzenleme
    def _gain_edited(self, value: float) -> None:
        self.tab.set_gain(value)

    def _cell_edited(self, item: QTableWidgetItem) -> None:
        if self._filling or item.column() != 1:
            return
        r = item.row()
        if r >= len(self._rows):
            return
        i, which, j = self._rows[r]
        try:
            vals = dc.parse_roots(item.text(), "Konum")
            v = complex(vals[0])
            self.tab.move_root(i, which, j, v, finished=True)
        except (ControlError, ValueError, IndexError) as exc:
            self.tab.show_message(f"Konum: {exc}")
            self.refresh()

    def add_element(self, kind: str) -> None:
        self.tab.edit_comp(lambda c: c.add(kind))

    def remove_selected(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            self.tab.show_message("Silmek için tablodan bir satır seçin.")
            return
        r = rows[0].row()
        if r < len(self._rows):
            i = self._rows[r][0]
            self.tab.edit_comp(lambda c: c.remove(i))

    def reset(self) -> None:
        def fn(c):
            c.elements.clear()
            c.K = 1.0
            c.touch()
        self.tab.edit_comp(fn)

    def add_leadlag(self, kind: str) -> None:
        from spt.control import classical
        phi, w = self.ll_phase.value(), self.ll_w.value()
        Ts = self.tab.Ts

        def fn(c):
            sys = classical.lead(phi, w, 1.0, Ts) if kind == "lead" else classical.lag(phi, w, 1.0, Ts)
            c.add_system(kind, sys, {"φ (°)": phi, "ωm (rad/s)": w})
        self.tab.edit_comp(fn)

    def add_notch(self) -> None:
        from spt.control import classical
        w0, depth, width = self.n_w0.value(), self.n_depth.value(), self.n_width.value()
        Ts = self.tab.Ts
        self.tab.edit_comp(lambda c: c.add_system("notch", classical.notch(w0, depth, width, Ts),
                                                  {"ω0": w0, "derinlik (dB)": depth, "genişlik": width}))

    def auto_leadlag(self) -> None:
        self.tab.auto_leadlag(self.a_pm.value(), self.a_wc.value())

    def gain_for_damping(self) -> None:
        self.tab.gain_for_damping(self.a_zeta.value())


# ============================================================================ PID ayarlayıcı paneli
class PidPanel(QWidget):
    """MATLAB PID Tuner karşılığı. Kontroller değişince (kısa gecikmeyle) `pidtune` arka planda çalışır; sonuç
    düzenlenen çevrimin kompansatörü olur (PID yapısı korunur)."""

    def __init__(self, tab: "DesignTab"):
        super().__init__()
        self.tab = tab
        self._sync = False
        self._tau_ref = 1.0
        self.result = None         # (C, info)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        sec = CollapsibleSection("PID ayarlayıcı")
        self.type = combo(list(dc.PID_TYPES))
        self.type.setCurrentText("PIDF")
        self.type.setToolTip("Denetleyici türü (F: türev filtresi)")
        sec.add_row("Tür", self.type)
        self.form = combo(["Paralel (Kp, Ki, Kd, Tf)", "Standart (Kp, Ti, Td, N)"])
        sec.add_row("Biçim", self.form)
        self.focus = combo([t for _k, t in dc.DESIGN_FOCUS])
        self.focus.setToolTip("PID / PIDF'de Ti = α·Td: dengeli α = 4, referans izleme 8, bozucu reddi 2")
        sec.add_row("Tasarım odağı", self.focus)
        self.iform = combo([t for _k, t in dc.FORMULAS])
        self.iform.setToolTip("Ayrık integratör formülü (IFormula)")
        sec.add_row("IFormula", self.iform)
        self.dform = combo(["Otomatik"] + [t for _k, t in dc.FORMULAS])
        self.dform.setToolTip("Ayrık türev formülü (DFormula); Otomatik: filtreli türlerde ileri Euler, filtresiz "
                              "PD / PID'de geri Euler")
        sec.add_row("DFormula", self.dform)
        lay.addWidget(sec)

        tune = CollapsibleSection("Hedefler")
        self.tau_slider = QSlider(Qt.Orientation.Horizontal)
        self.tau_slider.setRange(0, 400)
        self.tau_slider.setValue(200)
        self.tau_slider.setToolTip("Cevap süresi τ (sola: hızlı, sağa: yavaş); hedef geçiş frekansı ωc = 2/τ")
        tune.add_row("Cevap süresi", self.tau_slider)
        self.tau_label = label("", "Readout")
        tune.form.addRow(self.tau_label)
        self.pm_slider = QSlider(Qt.Orientation.Horizontal)
        self.pm_slider.setRange(0, 100)
        self.pm_slider.setValue(60)
        self.pm_slider.setToolTip("Geçici durum davranışı (sola: agresif, sağa: dayanıklı); faz payı 30° + 50°·θ")
        tune.add_row("Geçici durum", self.pm_slider)
        self.wc = _spin(1e-6, 1e9, 1.0, 5, " rad/s", "Hedef kazanç geçiş frekansı ωc")
        tune.add_row("ωc", self.wc)
        self.pm = _spin(1.0, 89.0, 60.0, 2, "°", "Hedef faz payı")
        tune.add_row("Faz payı", self.pm)
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        auto = QPushButton("Otomatik ayarla")
        auto.setToolTip("ωc'yi kütüphanenin kuralıyla seç (pidtune, ωc verilmeden)")
        auto.clicked.connect(lambda: self.request(auto=True))
        rl.addWidget(auto)
        base = QPushButton("Karşılaştırma olarak sakla")
        base.setToolTip("Geçerli kapalı çevrimi karşılaştırma eğrisi olarak sakla (MATLAB PID Tuner 'Baseline')")
        base.clicked.connect(tab.store_baseline)
        rl.addWidget(base)
        rl.addStretch(1)
        tune.form.addRow(row)
        lay.addWidget(tune)

        res = CollapsibleSection("Parametreler ve performans")
        self.params = QTableWidget(0, 2)
        self.params.setHorizontalHeaderLabels(["Büyüklük", "Değer"])
        self.params.verticalHeader().setVisible(False)
        self.params.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.params.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.params.horizontalHeader().setStretchLastSection(True)
        self.params.verticalHeader().setDefaultSectionSize(22)
        self.params.setMinimumHeight(220)
        res.form.addRow(self.params)
        self.state = label("", "Muted")
        self.state.setWordWrap(True)
        res.form.addRow(self.state)
        out = QWidget()
        ol = QHBoxLayout(out)
        ol.setContentsMargins(0, 0, 0, 0)
        add = QPushButton("Modellere ekle")
        add.setToolTip("Ayarlanan PID'i çalışma alanına ekle")
        add.clicked.connect(lambda: tab.add_to_workspace("C"))
        ol.addWidget(add)
        code = QPushButton("C Kodu Üret…")
        code.clicked.connect(tab.generate_code)
        ol.addWidget(code)
        ol.addStretch(1)
        res.form.addRow(out)
        lay.addWidget(res)

        for w in (self.type, self.form, self.focus, self.iform, self.dform):
            w.currentIndexChanged.connect(lambda *_: self.request())
        self.tau_slider.valueChanged.connect(self._tau_moved)
        self.pm_slider.valueChanged.connect(self._pm_moved)
        self.wc.valueChanged.connect(self._wc_edited)
        self.pm.valueChanged.connect(self._pm_edited)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(80)
        self._timer.timeout.connect(lambda: tab.tune_pid(auto=False))
        self._sync_labels()

    # ---------------------------------------------------------------- kaydırıcı eşlemesi
    def _tau_of(self, v: int) -> float:
        return self._tau_ref * 10 ** ((v - 200) / 100.0)

    def _slider_of(self, tau: float) -> int:
        return int(round(200 + 100 * math.log10(tau / self._tau_ref)))

    def set_targets(self, wc: float, pm: float) -> None:
        """Kaydırıcıları ve kutuları hedeflere getirir (sinyal üretmeden); τ ölçeği gerekirse ωc'ye ortalanır."""
        self._sync = True
        tau = dc.wc_to_response_time(wc)
        v = self._slider_of(tau)
        if not 0 <= v <= 400:
            self._tau_ref = tau
            v = 200
        self.tau_slider.setValue(v)
        self.wc.setValue(wc)
        self.pm.setValue(pm)
        self.pm_slider.setValue(int(round(100 * dc.pm_to_transient(pm))))
        self._sync = False
        self._sync_labels()

    def _sync_labels(self) -> None:
        wc = self.wc.value()
        self.tau_label.setText(f"τ = {dc.wc_to_response_time(wc):.4g} s  ·  ωc = {wc:.5g} rad/s  ·  "
                               f"Pm = {self.pm.value():.4g}°")

    def _tau_moved(self, v: int) -> None:
        if self._sync:
            return
        self._sync = True
        self.wc.setValue(dc.response_time_to_wc(self._tau_of(v)))
        self._sync = False
        self._sync_labels()
        self.request()

    def _pm_moved(self, v: int) -> None:
        if self._sync:
            return
        self._sync = True
        self.pm.setValue(dc.transient_to_pm(v / 100.0))
        self._sync = False
        self._sync_labels()
        self.request()

    def _wc_edited(self, wc: float) -> None:
        if self._sync:
            return
        self.set_targets(wc, self.pm.value())
        self.request()

    def _pm_edited(self, pm: float) -> None:
        if self._sync:
            return
        self.set_targets(self.wc.value(), pm)
        self.request()

    # ---------------------------------------------------------------- ayar
    def options(self) -> dict:
        focus = dc.DESIGN_FOCUS[self.focus.currentIndex()][0]
        iform = dc.FORMULAS[self.iform.currentIndex()][0]
        d = self.dform.currentIndex()
        dform = None if d == 0 else dc.FORMULAS[d - 1][0]
        return {"type": self.type.currentText(), "form": "standard" if self.form.currentIndex() else "parallel",
                "DesignFocus": focus, "IFormula": iform, "DFormula": dform, "wc": self.wc.value(),
                "PhaseMargin": self.pm.value()}

    def request(self, auto: bool = False) -> None:
        if auto:
            self._timer.stop()
            self.tab.tune_pid(auto=True)
        else:
            self._timer.start()

    def set_discrete(self, discrete: bool) -> None:
        self.iform.setEnabled(discrete)
        self.dform.setEnabled(discrete)

    def show_result(self, C, info: dict, perf: dict | None) -> None:
        """Parametre ve performans tablosu: kazançlar kütüphanenin PID nesnesinden, ölçütler kapalı çevrimden."""
        self.result = (C, info)
        rows = []
        Kp, Ki, Kd, Tf = C.parallel()
        if getattr(C, "_form", "parallel") == "standard":
            Kps, Ti, Td, N = C.standard()
            rows += [("Kp", Kps), ("Ti (s)", Ti), ("Td (s)", Td), ("N", N)]
        else:
            rows += [("Kp", Kp), ("Ki", Ki), ("Kd", Kd), ("Tf (s)", Tf)]
        if C.Ts > 0:
            rows.append(("Ts (s)", C.Ts))
        rows += [("Hedef ωc (rad/s)", info.get("CrossoverFrequency")), ("Tasarım Pm (°)", info.get("PhaseMargin"))]
        if perf:
            rows += [("Ölçülen Pm (°)", perf.get("Pm")), ("Ölçülen ωc (rad/s)", perf.get("Wcp")),
                     ("Kazanç payı (dB)", perf.get("GmdB")), ("Yükselme (s)", perf.get("RiseTime")),
                     ("Yerleşme (s)", perf.get("SettlingTime")), ("Aşım %", perf.get("Overshoot")),
                     ("Tepe", perf.get("Peak"))]
        self.params.setRowCount(len(rows))
        for r, (name, v) in enumerate(rows):
            self.params.setItem(r, 0, QTableWidgetItem(name))
            self.params.setItem(r, 1, QTableWidgetItem(num(v, 6) if not isinstance(v, str) else v))
        tok = self.tab._theme.tokens
        stable = info.get("Stable")
        color = tok.good if stable else tok.critical
        self.state.setText(f'<span style="color:{color}">●</span> ' + (
            "Kapalı çevrim kararlı" if stable else "Kapalı çevrim KARARSIZ: ωc'yi ya da türü değiştirin"))


# ============================================================================ klasik görünüm (sağ)
class ClassicalView(QWidget):
    def __init__(self, theme: ThemeManager):
        super().__init__()
        self._theme = theme
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        vsplit = QSplitter(Qt.Orientation.Vertical)
        vsplit.setChildrenCollapsible(False)
        top = QWidget()
        tl = QVBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(4)
        bar = QWidget()
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(0, 0, 0, 0)
        plots_btn = QToolButton()
        plots_btn.setText("Grafikler ▾")
        plots_btn.setToolTip("Gösterilecek grafikler")
        plots_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(plots_btn)
        self.plot_checks: dict[str, object] = {}
        for key, text in PLOTS:
            act = menu.addAction(text)
            act.setCheckable(True)
            act.setChecked(key != "effort")      # 1366×768'de okunur varsayılan: kontrol çabası menüden açılır
            act.toggled.connect(self.layout_plots)
            self.plot_checks[key] = act
        plots_btn.setMenu(menu)
        bl.addWidget(plots_btn, 0, Qt.AlignmentFlag.AlignTop)
        self.legend = cp.SeriesLegend()
        bl.addWidget(self.legend, 1)
        tl.addWidget(bar)
        host = QWidget()
        self.grid = QGridLayout(host)
        self.grid.setContentsMargins(0, 0, 4, 0)
        self.grid.setSpacing(10)
        tl.addWidget(_scroll(host), 1)
        vsplit.addWidget(top)
        self.rlocus = RootLocusEditor(theme)
        self.bode = BodeEditor(theme)
        mag, phase = self.bode.release_cards()
        self.bode.setParent(top)
        self.step = DesignStepPlot(theme, "Referans İzleme · r → y")
        self.dist = DesignStepPlot(theme, "Giriş Bozucusu Reddi · d → y")
        self.dist.markers = {"peak", "settling"}
        self.effort = DesignStepPlot(theme, "Kontrol Çabası · r → u")
        self.effort.markers = {"peak"}
        self.cells: dict[str, list[QWidget]] = {}

        def cell(card):
            c = QWidget()
            cl = QVBoxLayout(c)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.addWidget(card)
            c.setMinimumHeight(cp.CARD_MIN_HEIGHT)
            return c
        self.cells = {"rlocus": [cell(self.rlocus)], "bode": [cell(mag), cell(phase)], "step": [cell(self.step)],
                      "dist": [cell(self.dist)], "effort": [cell(self.effort)]}
        bottom = QSplitter(Qt.Orientation.Horizontal)
        bottom.setChildrenCollapsible(False)
        self.margin_card = Card("Kararlılık Payları")
        self.margin_table = make_table(MARGIN_COLUMNS)
        self.margin_card.body.addWidget(self.margin_table, 1)
        self.step_card = Card("Adım Cevabı Ölçütleri")
        self.step_card.add_right(label("stepinfo · %2 yerleşme", "Readout"), optional=True)
        self.step_table = make_table(STEP_COLUMNS)
        self.step_card.body.addWidget(self.step_table, 1)
        self.pole_card = Card("Kapalı Çevrim Kutupları")
        self.pole_table = make_table(POLE_COLUMNS)
        self.pole_card.body.addWidget(self.pole_table, 1)
        for c in (self.margin_card, self.step_card, self.pole_card):
            bottom.addWidget(c)
        bottom.setSizes([380, 520, 300])
        vsplit.addWidget(bottom)
        vsplit.setStretchFactor(0, 1)
        vsplit.setSizes([660, 140])
        lay.addWidget(vsplit)
        self.layout_plots()

    def layout_plots(self, *_) -> None:
        """Kök yer eğrisi (eşit eksenli) solda iki satır boyunca; diğer kartlar sütun sütun iki satırda (Bode genlik /
        faz alt alta). Kök yer eğrisi kapalıysa kartlar 2–3 sütunlu ızgaraya dizilir."""
        keys = [k for k, _t in PLOTS if self.plot_checks[k].isChecked()]
        for i in reversed(range(self.grid.count())):
            self.grid.takeAt(i)
        for k, lst in self.cells.items():
            for c in lst:
                c.setVisible(k in keys)
        others = [c for k in keys if k != "rlocus" for c in self.cells[k]]
        col0 = 0
        if "rlocus" in keys:
            rows = 2 if len(others) >= 2 else 1
            self.grid.addWidget(self.cells["rlocus"][0], 0, 0, rows, 1)
            col0 = 1
            for i, c in enumerate(others):
                self.grid.addWidget(c, i % 2, col0 + i // 2)
            ncol = col0 + (len(others) + 1) // 2
        else:
            n = len(others)
            ncol = 1 if n <= 1 else 2 if n <= 4 else 3
            for i, c in enumerate(others):
                self.grid.addWidget(c, i // ncol, i % ncol)
        for i in range(5):
            self.grid.setColumnStretch(i, 1 if i < ncol else 0)
        for i in range(3):
            self.grid.setRowStretch(i, 1)

    def detachable_cards(self) -> list[tuple[str, str, Card]]:
        return [("ctl_ds_rlocus", "Tasarım · Kök Yer Eğrisi", self.rlocus),
                ("ctl_ds_bode_mag", "Tasarım · Bode Genlik", self.bode.mag_card),
                ("ctl_ds_bode_phase", "Tasarım · Bode Faz", self.bode.phase_card),
                ("ctl_ds_step", "Tasarım · Referans İzleme", self.step),
                ("ctl_ds_dist", "Tasarım · Bozucu Reddi", self.dist),
                ("ctl_ds_effort", "Tasarım · Kontrol Çabası", self.effort),
                ("ctl_ds_margins", "Tasarım · Kararlılık Payları", self.margin_card),
                ("ctl_ds_stepinfo", "Tasarım · Adım Ölçütleri", self.step_card),
                ("ctl_ds_poles", "Tasarım · Kapalı Çevrim Kutupları", self.pole_card)]


# ============================================================================ sekme
class DesignTab(QWidget):
    """Kontrolcü Tasarımı sekmesi. Kabuk `(theme, workspace=…)` ile kurar; `set_sources`, `shutdown`,
    `detachable_cards`, `save_state` / `restore_state` yöntemleri vardır."""

    computed = pyqtSignal()

    def __init__(self, theme: ThemeManager, workspace: ModelWorkspace | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self._theme = theme
        self.workspace: ModelWorkspace | None = None
        self._sources: list = []
        self.jobs = BackgroundCall(self)
        self.jobs.done.connect(self._job_done)
        self.jobs.failed.connect(self._job_failed)
        self.Ts = 0.0
        self.comps = {"C": dc.Compensator(1.0), "C1": dc.Compensator(1.0), "C2": dc.Compensator(1.0)}
        self.loops: dc.LoopSet | None = None
        self.loops_cont: dc.LoopSet | None = None
        self.plants: dict = {}
        self.baseline = None
        self.messages: list[str] = []
        self.margin_values: dict = {}
        self.step_values: dict = {}
        self.cl_poles = np.zeros(0, complex)
        self._locus_key = None
        self._locus_drawn = None
        self._locus_sys = None
        self._drag_preview = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self.recompute)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        split.addWidget(self._build_left())
        self.views = QStackedWidget()
        self.classical = ClassicalView(theme)
        self.views.addWidget(self.classical)
        from spt.gui.control.design_modern import ModernPanel, ModernView
        from spt.gui.control.design_mpc import MpcPanel, MpcView
        self.modern_view = ModernView(theme)
        self.mpc_view = MpcView(theme)
        self.views.addWidget(self.modern_view)
        self.views.addWidget(self.mpc_view)
        self.modern = ModernPanel(self, self.modern_view)
        self.mpc = MpcPanel(self, self.mpc_view)
        self.pages.addWidget(self.modern)
        self.pages.addWidget(self.mpc)
        split.addWidget(self.views)
        split.setStretchFactor(1, 1)
        split.setSizes([360, 1000])
        lay.addWidget(split)
        rl = self.classical.rlocus
        rl.on_gain = self._drag_gain_locus
        rl.on_root = self._drag_root
        self.classical.bode.on_gain = self._drag_gain_bode
        self._sync_arch()
        self._sync_domain_widgets()
        if workspace is not None:
            self.set_workspace(workspace)

    # ================================================================ sol panel
    def _build_left(self) -> Card:
        card = Card("Tasarım")
        card.setMinimumWidth(320)
        card.setMaximumWidth(470)
        self.mode_bar = QTabBar()
        self.mode_bar.setExpanding(True)
        self.mode_bar.setDocumentMode(True)
        for _k, text in MODES:
            self.mode_bar.addTab(text)
        self.mode_bar.currentChanged.connect(self._on_mode)
        card.body.addWidget(self.mode_bar)
        host = QWidget()
        col = QVBoxLayout(host)
        col.setContentsMargins(0, 0, 6, 0)
        col.setSpacing(6)
        card.body.addWidget(_scroll(host), 1)

        self.loop_sec = CollapsibleSection("Çevrim yapısı")
        self.arch = combo([t for _k, t, _f in dc.ARCHS])
        for i, (_k, _t, f) in enumerate(dc.ARCHS):
            self.arch.setItemData(i, f, Qt.ItemDataRole.ToolTipRole)
        self.arch.currentIndexChanged.connect(self._sync_arch)
        self.loop_sec.add_row("Mimari", self.arch)
        self.diagram = _mono()
        self.loop_sec.form.addRow(self.diagram)
        self.boxes: dict[str, object] = {}
        self._box_labels: dict[str, object] = {}
        for key, text, tip in (("G", "Bitki G", "Kontrol edilen sistem (SISO)"),
                               ("G1", "Dış bitki G1", "Kademeli yapıda dış bitki (ω → θ); drone açısı için 1/s"),
                               ("H", "Algılayıcı H", "Geri besleme yolundaki algılayıcı dinamiği (yoksa 1)"),
                               ("F", "Ön filtre F", "Referans ön filtresi / ileri besleme")):
            box = combo([])
            box.setToolTip(tip)
            box.currentIndexChanged.connect(self._on_plant_changed)
            self.boxes[key] = box
            lab = label(text, "FormLabel")
            self._box_labels[key] = lab
            self.loop_sec.form.addRow(lab, box)
        self.edit = combo(["Dış çevrim C1 (açı)", "İç çevrim C2 (açısal hız)"])
        self.edit.setToolTip("Kademeli yapıda düzenlenen kompansatör. İç çevrim, dış çevrim açıkken (L = C2·G2); dış "
                             "çevrim, iç çevrim kapalıyken (L = C1·T2·G1) tasarlanır.")
        self.edit.currentIndexChanged.connect(self._on_edit_loop)
        self._edit_label = label("Düzenlenen", "FormLabel")
        self.loop_sec.form.addRow(self._edit_label, self.edit)
        col.addWidget(self.loop_sec)

        self.dom_sec = CollapsibleSection("Zaman alanı")
        self.domain = combo(["Sürekli (s)", "Ayrık (z) — c2d"])
        self.domain.setToolTip("Ayrık tasarımda bitki seçilen yöntemle ayrıklaştırılır, kompansatör z düzleminde "
                               "düzenlenir; sürekli eşdeğeri (d2c) ve onunla kapalı çevrim de gösterilir")
        self.domain.currentIndexChanged.connect(self._on_domain)
        self.dom_sec.add_row("Tasarım", self.domain)
        self.ts = _spin(1e-6, 1e4, 0.01, 6, " s", "Örnekleme zamanı Ts")
        self.ts.setStepType(QDoubleSpinBox.StepType.DefaultStepType)
        self.ts.setSingleStep(0.001)
        self.ts.valueChanged.connect(self._on_domain)
        self.dom_sec.add_row("Ts", self.ts)
        self.method_box = combo([])
        for key, text in C2D_METHODS:
            self.method_box.addItem(text, key)
        self.method_box.currentIndexChanged.connect(self._on_domain)
        self.dom_sec.add_row("Yöntem", self.method_box)
        col.addWidget(self.dom_sec)

        self.pages = QStackedWidget()
        self.comp_panel = CompensatorPanel(self)
        self.pid_panel = PidPanel(self)
        self.pages.addWidget(self.comp_panel)
        self.pages.addWidget(self.pid_panel)
        col.addWidget(self.pages)
        col.addStretch(1)
        self.status = label("", "Readout")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.RichText)
        card.body.addWidget(self.status)
        return card

    # ================================================================ durum sorguları
    @property
    def mode(self) -> str:
        return MODES[self.mode_bar.currentIndex()][0]

    def set_mode(self, mode: str) -> None:
        self.mode_bar.setCurrentIndex([k for k, _t in MODES].index(mode))

    @property
    def arch_key(self) -> str:
        return dc.ARCHS[self.arch.currentIndex()][0]

    def set_arch(self, key: str) -> None:
        self.arch.setCurrentIndex([k for k, *_r in dc.ARCHS].index(key))

    def edit_key(self) -> str:
        if self.arch_key == "cascade":
            return "C2" if self.edit.currentIndex() == 1 else "C1"
        return "C"

    @property
    def comp(self) -> dc.Compensator:
        return self.comps[self.edit_key()]

    def method(self) -> str:
        return self.method_box.currentData() or "zoh"

    # ================================================================ çalışma alanı
    def set_workspace(self, workspace: ModelWorkspace) -> None:
        self.workspace = workspace
        for sig in (workspace.added, workspace.removed, workspace.changed):
            sig.connect(self._on_workspace)
        workspace.renamed.connect(lambda *_: self._on_workspace())
        self._fill_boxes()
        self.modern.fill_plants()
        self.mpc.fill_plants()
        self.schedule()

    def _on_workspace(self, *_) -> None:
        self._fill_boxes()
        self.modern.fill_plants()
        self.mpc.fill_plants()
        self.schedule()

    def _fill_boxes(self) -> None:
        names = self.workspace.lti_names() if self.workspace is not None else []
        for key, box in self.boxes.items():
            cur = box.currentData()
            had_models = box.count() > 1
            box.blockSignals(True)
            box.clear()
            if key in ("G",):
                box.addItem("— seçilmedi —", NONE)
            elif key == "G1":
                box.addItem("1/s (integratör)", INTEGRATOR)
            else:
                box.addItem("— yok (1) —", NONE)
            for n in names:
                box.addItem(n, n)
            i = box.findData(cur)
            if key == "G" and names and (i < 0 or (cur is None and not had_models)):
                i = 1   # ilk model bitki olarak önerilir
            box.setCurrentIndex(max(i, 0))
            box.blockSignals(False)

    def set_plant(self, G: str | None = None, G1: str | None = None, H: str | None = None,
                  F: str | None = None) -> None:
        """Bileşenleri adlarıyla seç (None: değiştirme; "": yok / integratör)."""
        for key, name in (("G", G), ("G1", G1), ("H", H), ("F", F)):
            if name is None:
                continue
            box = self.boxes[key]
            data = (INTEGRATOR if key == "G1" else NONE) if name == "" else name
            i = box.findData(data)
            if i >= 0:
                box.setCurrentIndex(i)
        self.schedule()

    def _model(self, key: str):
        data = self.boxes[key].currentData()
        if data is None:
            return None
        if data == INTEGRATOR:
            return dc.lti_module().tf([1.0], [1.0, 0.0])
        return self.workspace.lti(data)

    def plants_in(self, Ts: float) -> dict:
        """Seçili bileşenler verilen zamanda (c2d / d2c / d2d, seçili yöntemle)."""
        m = self.method()
        out = {}
        keys = ("G", "G1") if self.arch_key == "cascade" else ("G", "H", "F")
        for key in keys:
            sys = self._model(key)
            if key == "F" and self.arch_key == "single":
                sys = None
            if sys is not None:
                what = {"G": "İç bitki G2" if self.arch_key == "cascade" else "Bitki G", "G1": "Dış bitki G1",
                        "H": "Algılayıcı H", "F": "F"}[key]
                dc.require_siso(sys, what)
                sys = dc.to_domain(sys, Ts, m)
            out[key] = sys
        return out

    # ================================================================ olaylar
    def _on_mode(self, index: int) -> None:
        mode = self.mode
        self.pages.setCurrentIndex(index)
        self.loop_sec.setVisible(mode in ("comp", "pid"))
        self.dom_sec.setVisible(mode in ("comp", "pid", "modern"))
        self.views.setCurrentIndex({"comp": 0, "pid": 0, "modern": 1, "mpc": 2}[mode])
        if mode == "pid" and self.pid_panel.result is None:
            self.tune_pid(auto=True)
        self.schedule()

    def _sync_arch(self, *_) -> None:
        a = self.arch_key
        cascade = a == "cascade"
        self.diagram.setText(dc.ARCH_DIAGRAMS[a])
        self._box_labels["G"].setText("İç bitki G2 (u → ω)" if cascade else "Bitki G")
        for key, show in (("G1", cascade), ("H", not cascade), ("F", a in ("prefilter", "feedforward"))):
            self.boxes[key].setVisible(show)
            self._box_labels[key].setVisible(show)
        self._box_labels["F"].setText("İleri besleme F" if a == "feedforward" else "Ön filtre F")
        self.edit.setVisible(cascade)
        self._edit_label.setVisible(cascade)
        self._locus_drawn = None
        self.schedule()

    def _on_edit_loop(self, *_) -> None:
        self._locus_drawn = None
        self.pid_panel.result = None
        if self.mode == "pid":
            self.tune_pid(auto=True)
        self.schedule()

    def _on_plant_changed(self, *_) -> None:
        self._locus_drawn = None
        if self.mode == "pid":
            self.pid_panel.request()
        self.schedule()

    def _sync_domain_widgets(self) -> None:
        discrete = self.domain.currentIndex() == 1
        self.ts.setEnabled(discrete)
        self.pid_panel.set_discrete(discrete)

    def design_Ts(self) -> float:
        return float(self.ts.value()) if self.domain.currentIndex() == 1 else 0.0

    def set_domain(self, Ts: float, method: str | None = None) -> None:
        """Tasarım zamanı: Ts = 0 sürekli, > 0 ayrık (c2d yöntemiyle)."""
        self.domain.blockSignals(True)
        self.ts.blockSignals(True)
        self.method_box.blockSignals(True)
        if method is not None:
            i = self.method_box.findData(method)
            if i >= 0:
                self.method_box.setCurrentIndex(i)
        if Ts > 0:
            self.ts.setValue(Ts)
        self.domain.setCurrentIndex(1 if Ts > 0 else 0)
        self.domain.blockSignals(False)
        self.ts.blockSignals(False)
        self.method_box.blockSignals(False)
        self._on_domain()

    def _on_domain(self, *_) -> None:
        self._sync_domain_widgets()
        Ts = self.design_Ts()
        notes = []
        if abs(Ts - self.Ts) > 1e-15:
            for key, c in list(self.comps.items()):
                try:
                    new, note = c.to_domain(Ts, self.method())
                except ControlError as exc:
                    new, note = dc.Compensator(1.0, Ts=Ts), f"{key} dönüştürülemedi ({exc}); C = 1"
                self.comps[key] = new
                if note and (c.elements or c.pid is not None):
                    notes.append(f"{key}: {note}")
            if self.baseline is not None:
                notes.append("karşılaştırma eğrisi önceki zaman alanında kaldı")
            self.Ts = Ts
            self.pid_panel.result = None
        self._locus_drawn = None
        self.modern.on_domain()
        for n in notes:
            self.show_message(n)
        if self.mode == "pid":
            self.pid_panel.request()
        self.schedule()

    # ================================================================ kompansatör düzenleme
    def edit_comp(self, fn) -> None:
        c = self.comp.copy()
        try:
            fn(c)
        except ControlError as exc:
            self.show_message(str(exc))
            return
        self.comps[self.edit_key()] = c
        self._locus_drawn = None
        self.recompute()

    def set_gain(self, K: float, finished: bool = True) -> None:
        c = self.comp
        try:
            self._scale_gain(c, K)
        except ControlError as exc:
            self.show_message(str(exc))
            return
        if finished:
            self.recompute()
        else:
            self._quick_markers()
            self.schedule(140)

    @staticmethod
    def _scale_gain(c: dc.Compensator, K: float) -> None:
        """K değişimi: PID yapısındaysa tüm kazançlar aynı oranla ölçeklenir (C·α tam olarak yine PID'dir)."""
        pid = c.pid
        if pid is not None and c.K != 0 and math.isfinite(K) and K != 0:
            from spt.control import classical
            a = K / c.K
            if getattr(pid, "_form", "parallel") == "standard":
                Kp, Ti, Td, N = pid.standard()
                new = classical.pidstd(Kp * a, Ti, Td, N, pid.Ts, IFormula=pid.IFormula, DFormula=pid.DFormula)
            else:
                Kp, Ki, Kd, Tf = pid.parallel()
                new = classical.pid(Kp * a, Ki * a, Kd * a, Tf, pid.Ts, IFormula=pid.IFormula,
                                    DFormula=pid.DFormula)
            c.K = float(K)
            c.pid = new
            return
        c.set_gain(K)

    def move_root(self, index: int, which: str, j: int, value: complex, finished: bool = True) -> None:
        try:
            self.comp.move(index, which, j, value)
        except (ControlError, IndexError) as exc:
            self.show_message(str(exc))
            return
        self._locus_drawn = None
        if finished:
            self.recompute()
        else:
            self._quick_markers()
            self.schedule(140)

    def _drag_gain_locus(self, point: complex, finished: bool) -> None:
        """Kapalı çevrim kutbu sürüklendi: K = işaret·rlocfind(L0, nokta) (sürükleme sırasında eğri ızgarasının en
        yakın noktası; bırakınca kütüphanenin rlocfind'i)."""
        if self.loops is None:
            return
        comp = self.comp
        sign, L0 = dc.locus_system(comp, self.loops.Gloop)
        rl = self.classical.rlocus
        if rl.loci("L0") is not None:
            self.set_gain(sign * float(rl.gain_at(point.real, point.imag, "L0")[0]), finished=False)
        if finished:
            key = self.edit_key()

            def job(sys=L0, p=complex(point), sign=sign, key=key):
                from spt.control.classical import rlocfind
                k, _p = rlocfind(sys, [p])
                return key, sign * float(k[0])
            self.jobs.submit("rlocfind", job)

    def _rlocfind_done(self, res) -> None:
        key, K = res
        if key == self.edit_key():
            self.set_gain(K, True)

    def _drag_root(self, h: dict, point: complex, finished: bool) -> None:
        self.move_root(h["elem"], h["which"], h["j"], point, finished)

    def _drag_gain_bode(self, factor: float, finished: bool, h: dict) -> None:
        if "K0" not in h:
            h["K0"] = self.comp.K
        self.set_gain(h["K0"] * factor, finished)

    def auto_leadlag(self, pm: float, wc: float) -> None:
        from spt.control.classical import leadlag
        if self.loops is None:
            self.show_message("Önce bitkiyi seçin.")
            return
        try:
            C, info = leadlag(self.loops.Gloop, pm, wc)
            new = dc.Compensator.from_lti(C)
        except ControlError as exc:
            self.show_message(f"Öncül / ardıl: {exc}")
            return
        kind = "lead" if info.get("Type") == "lead" else "lag"
        # tek gerçel sıfır + tek gerçel kutup → tek öncül / ardıl öğesi
        z = [e for e in new.elements if e.kind == "real_zero"]
        p = [e for e in new.elements if e.kind == "real_pole"]
        if len(z) == 1 and len(p) == 1 and len(new.elements) == 2:
            new.elements = [dc.Element(kind, list(z[0].zeros), list(p[0].poles), {"Pm": pm, "ωc": wc})]
        self.comps[self.edit_key()] = new
        self._locus_drawn = None
        self.show_message(f"{'Öncül' if kind == 'lead' else 'Ardıl'} tasarlandı: ωc = {info['CrossoverFrequency']:.5g}"
                          f" rad/s, Pm = {info['PhaseMargin']:.4g}°" + ("" if info.get("Stable") else " — KARARSIZ"))
        self.recompute()

    def gain_for_damping(self, zeta: float) -> None:
        from spt.control.classical import gain_for_damping
        if self.loops is None:
            self.show_message("Önce bitkiyi seçin.")
            return
        sign, L0 = dc.locus_system(self.comp, self.loops.Gloop)
        try:
            res = gain_for_damping(L0, zeta=zeta)
        except ControlError as exc:
            self.show_message(f"Sönüm için kazanç: {exc}")
            return
        if not res:
            self.show_message(f"Kök yer eğrisi ζ = {zeta:g} ışınını kesmiyor.")
            return
        self.set_gain(sign * float(res[0]["K"]))
        self.show_message(f"ζ = {zeta:g} için K = {sign * res[0]['K']:.6g} (kutup {dc.fmt_complex(res[0]['pole'])})")

    # ================================================================ PID
    def tune_pid(self, auto: bool = False) -> None:
        try:
            plants = self.plants_in(self.Ts)
            loops = self._build(plants, self._comp_lti())
        except ControlError as exc:
            self.pid_panel.state.setText(str(exc))
            return
        opts = self.pid_panel.options()
        Gloop = loops.Gloop
        wc = None if auto else opts["wc"]
        key = self.edit_key()

        def job(G=Gloop, o=opts, wc=wc, key=key):
            from spt.control.classical import pidtune
            C, info = pidtune(G, o["type"], wc, PhaseMargin=o["PhaseMargin"], DesignFocus=o["DesignFocus"],
                              IFormula=o["IFormula"], DFormula=o["DFormula"], form=o["form"])
            return C, info, wc is None, key
        self.jobs.submit("pid", job)

    def _pid_done(self, res) -> None:
        C, info, auto, key = res
        self.comps[key] = dc.Compensator.from_pid(C)
        if auto:
            self.pid_panel.set_targets(info["CrossoverFrequency"], self.pid_panel.pm.value())
        self.pid_panel.result = (C, info)
        self._locus_drawn = None
        self.recompute()

    def store_baseline(self) -> None:
        if self.loops is None or self.loops.T is None:
            self.show_message("Saklanacak kapalı çevrim yok.")
            return
        self.baseline = self.loops.T
        self.show_message("Geçerli kapalı çevrim karşılaştırma eğrisi olarak saklandı.")
        self.recompute()

    # ================================================================ hesap
    def schedule(self, ms: int | None = None) -> None:
        if ms is not None:
            self._timer.setInterval(ms)
        else:
            self._timer.setInterval(50)
        self._timer.start()

    def _comp_lti(self) -> dict:
        return {k: c.zpk() for k, c in self.comps.items()}

    def _build(self, plants: dict, comps: dict) -> dc.LoopSet:
        if plants.get("G") is None:
            raise ControlError("Bitki seçin (Modeller sekmesinde ifadeyle model ekleyebilirsiniz).")
        return dc.build_loops(self.arch_key, comps, plants["G"], plants.get("G1"), plants.get("H"),
                              plants.get("F"), self.edit_key())

    def recompute(self) -> None:
        self._timer.stop()
        mode = self.mode
        if mode == "modern":
            self.modern.recompute()
        elif mode == "mpc":
            self.mpc.refresh()
        else:
            self._recompute_classical()
        self.computed.emit()

    def _recompute_classical(self) -> None:
        msgs: list[str] = []
        view = self.classical
        self.comp_panel.refresh()
        try:
            self.plants = self.plants_in(self.Ts)
            self.loops = self._build(self.plants, self._comp_lti())
        except ControlError as exc:
            self.loops = None
            self._clear_classical(str(exc))
            return
        loops = self.loops
        comp = self.comp
        keys = ("C1", "C2") if self.arch_key == "cascade" else ("C",)
        for k in keys:
            c = self.comps[k]
            if c.discrete and not c.is_proper():
                msgs.append(f"{k} nedensel değil: ayrık kompansatörde sıfır sayısı kutup sayısını aşamaz (ör. z = 0'a "
                            "kutup ekleyin)")
        # kapalı çevrim kutupları ve kompansatör işaretleri (anında)
        try:
            self.cl_poles = dc.closed_loop_poles(loops.L)
        except ControlError as exc:
            self.cl_poles = np.zeros(0, complex)
            msgs.append(f"Kapalı çevrim kutupları: {exc}")
        self._quick_markers()
        self._request_locus()
        # Bode (paylar tablo ile aynı değerler)
        sL = cp.Series(NAMES["L"], loops.L, COLORS["L"], "L")
        try:
            margins = dc.loop_margins(loops.L, loops.Gloop, comp.zpk())
            if margins.get("note"):
                log.info("Tasarım payı: %s", margins["note"])
        except ControlError as exc:
            margins = None
            msgs.append(f"Paylar: {exc}")
        self.margin_values = {"L": margins} if margins else {}
        for e in view.bode.show_loop([sL], "L", margins):
            msgs.append(f"Bode · {e}")
        # zaman cevapları
        for key, err in loops.errors.items():
            msgs.append(f"{NAMES.get(key, key)}: {err}")
        series_T = [cp.Series(NAMES["T"], loops.T, COLORS["T"], "T")] if loops.T is not None else []
        if loops.T2 is not None and self.edit_key() == "C2":
            series_T.append(cp.Series(NAMES["T2"], loops.T2, COLORS["T2"], "T2"))
        if self.baseline is not None:
            series_T.append(cp.Series(NAMES["T_base"], self.baseline, COLORS["T_base"], "T_base"))
        self.loops_cont = None
        if self.Ts > 0:
            try:
                comps_c = {k: dc.cont_equivalent(c.zpk(), self.method())[0] for k, c in self.comps.items()}
                self.loops_cont = self._build(self.plants_in(0.0), comps_c)
                if self.loops_cont.T is not None:
                    series_T.append(cp.Series(NAMES["T_cont"], self.loops_cont.T, COLORS["T_cont"], "T_cont"))
            except ControlError as exc:
                msgs.append(f"Sürekli eşdeğer: {exc}")
        tfinal = None
        if loops.T is not None:
            tfinal, _cut = capped_tfinal([loops.T])
        other = {k: [cp.Series(NAMES[k], getattr(loops, k), COLORS[k], k)] if getattr(loops, k) is not None else []
                 for k in ("GS", "KS")}
        legend = ([(COLORS["L"], NAMES["L"])] + [(s.color, s.name) for s in series_T] +
                  [(s.color, s.name) for k in ("GS", "KS") for s in other[k]] +
                  [(COLORS["comp"], "kompansatör kutup / sıfırları")])
        if legend != getattr(self, "_legend_items", None):    # aynıysa yeniden kurma (sürüklerken titremesin)
            self._legend_items = legend
            view.legend.set_series(legend)
        self._fill_margin_table(margins)
        # adım cevapları ve stepinfo arka planda (yavaş / çok hızlı kipler arayüzü dondurmasın)
        cards = ((view.step, series_T), (view.dist, other["GS"]), (view.effort, other["KS"]))
        items = {s.key: s.sys for _c, ser in cards for s in ser}
        self._resp_pending = (cards, tfinal, margins, msgs)
        view.step_card.title.setText("Adım Cevabı Ölçütleri · hesaplanıyor…")
        self.jobs.submit("resp", lambda items=items, tf=tfinal: (items, responses(items, tf)))
        self._set_status(msgs)

    def _resp_done(self, res) -> None:
        items, data = res
        pend = getattr(self, "_resp_pending", None)
        if pend is None:
            return
        cards, tfinal, margins, msgs = pend
        if any(items.get(s.key) is not s.sys for _c, ser in cards for s in ser):
            return     # daha yeni bir hesap bekliyor
        self._resp_pending = None
        msgs = list(msgs)
        for card, series in cards:
            for e in card.set_series(series, tfinal=tfinal, data=data):
                msgs.append(f"{card.title.text()} · {e}")
        self.classical.step_card.title.setText("Adım Cevabı Ölçütleri")
        self._fill_step_table([s for _c, ser in cards for s in ser], data)
        if self.mode == "pid" and self.pid_panel.result is not None:
            self._pid_perf(margins)
        self._set_status(msgs)

    def _quick_markers(self) -> None:
        """Kök yer eğrisi işaretleri: kapalı çevrim kutupları ve kompansatör kökleri (eğriyi beklemeden)."""
        comp = self.comp
        if self.loops is not None:
            try:
                L = comp.zpk() * self.loops.Gloop
                self.cl_poles = dc.closed_loop_poles(L)
            except ControlError:
                pass
        roots = []
        for i, e in enumerate(comp.elements):
            for which, lst in (("z", e.zeros), ("p", e.poles)):
                for j, v in enumerate(lst):
                    roots.append({"elem": i, "which": which, "j": j, "value": complex(v), "movable": e.movable})
        self.classical.rlocus.set_markers(self.cl_poles, roots, self.Ts, COLORS["T"], COLORS["comp"])

    def _request_locus(self) -> None:
        if self.loops is None:
            return
        sign, L0 = dc.locus_system(self.comp, self.loops.Gloop)
        z, p, _k = dc.lti_module().zpk(L0).zpkdata(squeeze=True)
        key = (tuple(np.round(np.sort_complex(np.asarray(z, complex)), 12)),
               tuple(np.round(np.sort_complex(np.asarray(p, complex)), 12)), round(float(_k), 12), L0.Ts, sign,
               self.edit_key())
        if key == self._locus_drawn or key == self._locus_key:
            return
        self._locus_key = key
        self.classical.rlocus.busy_label.setText("eğri hesaplanıyor…")

        def job(sys=L0, key=key):
            from spt.control.analysis import rlocus
            r, k = rlocus(sys)
            return key, sys, r, k
        self.jobs.submit("rlocus", job)

    def _locus_done(self, res) -> None:
        key, sys, r, k = res
        rl = self.classical.rlocus
        rl.busy_label.setText("")
        if key != self._locus_key:
            return
        self._locus_drawn = key
        self._locus_key = None
        self._locus_sys = sys
        name = "L0 = C0·G" if key[4] > 0 else "L0 = −C0·G (K < 0)"
        rl.set_locus(sys, r, k, COLORS["L"], name)
        self._quick_markers()

    def _job_done(self, key: str, res) -> None:
        if key == "rlocus":
            self._locus_done(res)
        elif key == "pid":
            self._pid_done(res)
        elif key == "mpc":
            self.mpc.sim_done(res)
        elif key == "rlocfind":
            self._rlocfind_done(res)
        elif key == "resp":
            self._resp_done(res)
        elif key == "mresp":
            self.modern.resp_done(res)

    def _job_failed(self, key: str, msg: str) -> None:
        if key == "rlocus":
            self.classical.rlocus.busy_label.setText("")
            self._locus_key = None
            self.classical.rlocus.clear()
            self.classical.rlocus.note.setText(f"Kök yer eğrisi: {msg}")
            self.classical.rlocus.note.show()
        elif key == "pid":
            self.pid_panel.state.setText(f"PID ayarlanamadı: {msg}")
            self.show_message(f"PID: {msg}")
        elif key == "mpc":
            self.mpc.sim_failed(msg)
        else:
            self.show_message(f"{key}: {msg}")

    def _clear_classical(self, msg: str) -> None:
        view = self.classical
        self._resp_pending = None
        self.jobs.cancel("resp")
        for card in (view.step, view.dist, view.effort):
            card.set_series([])
        view.bode.set_series([])
        view.rlocus.set_series([])
        view.rlocus.set_markers([], [], self.Ts, COLORS["T"], COLORS["comp"])
        view.legend.set_series([])
        self._legend_items = None
        for t in (view.margin_table, view.step_table, view.pole_table):
            t.setRowCount(0)
        self.margin_values, self.step_values = {}, {}
        self._set_status([msg])

    def _fill_margin_table(self, margins: dict | None) -> None:
        view = self.classical
        tok = self._theme.tokens
        st = dc.stable(self.loops.T) if self.loops.T is not None else None
        if margins:
            gm = margins["Gm"]
            gdb = 20 * math.log10(gm) if gm and gm > 0 and math.isfinite(gm) else (math.inf if gm == math.inf
                                                                                 else math.nan)
            rows = [[NAMES["L"], num(gdb), num(margins["Wcg"]), num(margins["Pm"]), num(margins["Wcp"]),
                     "—" if st is None else ("kararlı" if st else "kararsız")]]
        else:
            rows = [[NAMES["L"], "—", "—", "—", "—", "—"]]
        fill_table(view.margin_table, rows, [COLORS["L"]], 5, tok)
        prow = []
        for p, z, w in dc.damp_rows(self.cl_poles, self.Ts):
            prow.append([dc.fmt_complex(p, 6), num(z, 4), num(w, 5)])
        fill_table(view.pole_table, prow)

    def _fill_step_table(self, all_series: list[cp.Series], data: dict) -> None:
        """stepinfo tablosu: değerler arka planda hesaplanan kütüphane stepinfo'sundan, kalıcı değer DC kazancı."""
        from spt.control.freqresp import dcgain
        rows, colors = [], []
        self.step_values = {}
        for s in all_series:
            d = data.get(s.key) or {}
            info = d.get("info")
            cells = [s.name] + ["—"] * 6
            if info:
                self.step_values[s.key] = dict(info)
                cells[1:6] = [num(info.get(k)) for k in ("RiseTime", "SettlingTime", "Overshoot", "Peak",
                                                         "PeakTime")]
            elif d.get("error"):
                cells[1] = d["error"]
            try:
                cells[6] = num(float(np.ravel(dcgain(s.sys))[0]))
            except (ControlError, ValueError):
                pass
            rows.append(cells)
            colors.append(s.color)
        fill_table(self.classical.step_table, rows, colors)

    def _pid_perf(self, margins: dict | None) -> None:
        C, info = self.pid_panel.result
        if self.comp.pid is None:
            self.pid_panel.state.setText("Kompansatör elle değiştirildi: PID yapısında değil (yeniden ayarlayın).")
            return
        C = self.comp.pid
        perf = {}
        if margins:
            gm = margins["Gm"]
            perf.update(Pm=margins["Pm"], Wcp=margins["Wcp"],
                        GmdB=20 * math.log10(gm) if gm and 0 < gm < math.inf else gm)
        perf.update(self.step_values.get("T", {}))
        self.pid_panel.show_result(C, info, perf)

    def _set_status(self, msgs: list[str]) -> None:
        tok = self._theme.tokens
        self.messages = list(msgs)
        if msgs:
            self.status.setText("<br>".join(f'<span style="color:{tok.warning}">⚠</span> {m}' for m in msgs[:5]))
        else:
            d = f"ayrık, Ts = {self.Ts:g} s" if self.Ts > 0 else "sürekli"
            self.status.setText(f'<span style="color:{tok.good}">●</span> Tasarım güncel ({d})')

    def show_message(self, text: str) -> None:
        tok = self._theme.tokens
        self.status.setText(f'<span style="color:{tok.accent}">●</span> {text}')
        log.info("Kontrolcü Tasarımı: %s", text)

    # ================================================================ çıktılar
    def add_to_workspace(self, what: str) -> list[str]:
        """Tasarımı çalışma alanına ekler: "C" (kompansatör(ler), PID yapısı korunur), "T" ya da "L"."""
        if self.workspace is None:
            return []
        added = []
        d = f"Ts = {self.Ts:g} s" if self.Ts > 0 else "sürekli"
        try:
            if what == "C":
                keys = ["C1", "C2"] if self.arch_key == "cascade" else ["C"]
                for k in keys:
                    c = self.comps[k]
                    obj = c.controller()
                    kind = "PID" if c.pid is not None else "kompansatör"
                    added.append(self.workspace.add(self.workspace.unique_name(k), obj, SOURCE_DESIGN,
                                                    f"{kind} ({d})"))
            elif self.loops is not None:
                sys = self.loops.T if what == "T" else self.loops.L
                note = "kapalı çevrim r → y" if what == "T" else f"açık çevrim ({self.edit_key()})"
                added.append(self.workspace.add(self.workspace.unique_name(what), sys, SOURCE_DESIGN,
                                                f"{note} ({d})"))
        except ControlError as exc:
            self.show_message(str(exc))
            return added
        if added:
            self.show_message("Çalışma alanına eklendi: " + ", ".join(added))
        return added

    def generate_code(self):
        comp = self.comp
        if not comp.discrete:
            self.show_message("C kodu için ayrık tasarım gerekir: Zaman alanı → Ayrık (Ts) seçin.")
            return None
        from spt.gui.dialogs.control_code_dialog import open_control_code_dialog
        try:
            return open_control_code_dialog(self, comp.controller(), self.edit_key())
        except ControlError as exc:
            self.show_message(f"C kodu: {exc}")
            return None

    # ================================================================ kabuk arayüzü
    def set_sources(self, sources: list) -> None:
        self._sources = list(sources)

    def shutdown(self) -> None:
        self._timer.stop()
        self.jobs.shutdown()

    def wait_idle(self, timeout: float = 30.0) -> bool:
        """Bekleyen gecikmeli hesapları ve arka plan işlerini bitirir (testler için)."""
        from PyQt6.QtWidgets import QApplication
        for _ in range(12):
            pending = False
            if self.pid_panel._timer.isActive():
                self.pid_panel._timer.stop()
                self.tune_pid(auto=False)
                pending = True
            if self.mpc.pending():
                self.mpc.run_now()
                pending = True
            if self.modern._timer.isActive():
                self.modern._timer.stop()
                self.schedule()
                pending = True
            if self._timer.isActive():
                self.recompute()
                pending = True
            self.jobs.wait(timeout)
            QApplication.processEvents()
            if not pending and not self.jobs.busy and not self._timer.isActive():
                break
        return not self.jobs.busy

    def detachable_cards(self) -> list[tuple[str, str, Card]]:
        return self.classical.detachable_cards() + self.modern_view.detachable_cards() + \
            self.mpc_view.detachable_cards()

    def save_state(self, s) -> None:
        s.setValue("control/design_mode", self.mode)
        s.setValue("control/design_arch", self.arch_key)
        s.setValue("control/design_plants", json.dumps({k: b.currentData() for k, b in self.boxes.items()}))
        s.setValue("control/design_edit", self.edit.currentIndex())
        s.setValue("control/design_domain", json.dumps({"Ts": self.design_Ts(), "ts_value": self.ts.value(),
                                                        "method": self.method()}))
        s.setValue("control/design_comps", json.dumps({k: c.to_dict() for k, c in self.comps.items()}))
        s.setValue("control/design_pid", json.dumps({"type": self.pid_panel.type.currentText(),
                                                     "form": self.pid_panel.form.currentIndex(),
                                                     "focus": self.pid_panel.focus.currentIndex(),
                                                     "wc": self.pid_panel.wc.value(),
                                                     "pm": self.pid_panel.pm.value()}))
        s.setValue("control/design_plots", ",".join(k for k, b in self.classical.plot_checks.items()
                                                    if b.isChecked()))
        self.modern.save_state(s)
        self.mpc.save_state(s)

    def restore_state(self, s) -> None:
        try:
            arch = str(s.value("control/design_arch", "single"))
            if arch in [k for k, *_r in dc.ARCHS]:
                self.set_arch(arch)
            plants = json.loads(str(s.value("control/design_plants", "{}")) or "{}")
            for key, name in plants.items():
                if key in self.boxes:
                    i = self.boxes[key].findData(name)
                    if i >= 0:
                        self.boxes[key].setCurrentIndex(i)
            self.edit.setCurrentIndex(int(s.value("control/design_edit", 0)))
            dom = json.loads(str(s.value("control/design_domain", "{}")) or "{}")
            if dom:
                self.ts.blockSignals(True)
                self.ts.setValue(float(dom.get("ts_value", self.ts.value())))
                self.ts.blockSignals(False)
                comps = json.loads(str(s.value("control/design_comps", "{}")) or "{}")
                self.set_domain(float(dom.get("Ts", 0.0)), dom.get("method"))
                for k, d in comps.items():
                    if k in self.comps:
                        c = dc.Compensator.from_dict(d)
                        if abs(c.Ts - self.Ts) <= 1e-12 * max(1.0, self.Ts):
                            self.comps[k] = c
            pid = json.loads(str(s.value("control/design_pid", "{}")) or "{}")
            if pid:
                pp = self.pid_panel
                for w in (pp.type, pp.form, pp.focus):
                    w.blockSignals(True)
                pp.type.setCurrentText(pid.get("type", "PIDF"))
                pp.form.setCurrentIndex(int(pid.get("form", 0)))
                pp.focus.setCurrentIndex(int(pid.get("focus", 0)))
                for w in (pp.type, pp.form, pp.focus):
                    w.blockSignals(False)
                pp.set_targets(float(pid.get("wc", 1.0)), float(pid.get("pm", 60.0)))
            plots = str(s.value("control/design_plots", ""))
            if plots:
                for k, b in self.classical.plot_checks.items():
                    b.setChecked(k in plots.split(","))
            self.modern.restore_state(s)
            self.mpc.restore_state(s)
            mode = str(s.value("control/design_mode", "comp"))
            if mode in [k for k, _t in MODES]:
                self.set_mode(mode)
        except (TypeError, ValueError, ControlError, json.JSONDecodeError) as exc:
            log.warning("Kontrolcü Tasarımı ayarları okunamadı: %s", exc)
        self.schedule()
