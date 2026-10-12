"""Blok Diyagram sekmesi: Simulink benzeri düzenleyici (kütüphane · tuval · parametreler · skop sonuçları).

Bu modül yalnız arayüzdür; model, benzetim ve doğrusallaştırma `spt.control.diagram` motorundadır:
- Tuvaldeki diyagram `to_model()` ile motorun `Model`'ine çevrilir (aynı `Block` nesneleri; konum, yön, tel kıvrımları
  ve doğrusallaştırma noktaları `meta` alanlarında). `.sptd` kaydet / aç `Model.save` / `Model.load` ile yapılır.
- Çalıştır: modelin kopyası arka plan iş parçacığında `simulate` ile benzetilir (ilerleme, iptal); arayüz donmaz.
  Hatalar (bağlanmamış port, cebirsel döngü, örnekleme zamanı çatışması …) motorun Türkçe iletisiyle gösterilir ve
  iletide adı geçen bloklar tuvalde kırmızı çerçeveyle işaretlenir.
- Doğrusallaştır: tellerde (sinyalin kaynak portu) seçilen noktalarla `linearize`; sonuç durum uzayı modeli
  `model_created(model, ad)` sinyaliyle yayınlanır (Modeller çalışma alanına eklenir).

Şablonlar: `register_template(ad, kurucu, açıklama, kategori)` ile eklenir (kurucu `Model` döndürür); ayrıca
`TEMPLATE_MODULES`'teki modüller (ör. İHA otopilot şablonları) varsa `TEMPLATES` listesi ya da `templates()`
fonksiyonuyla aynı sözlükleri ("name", "build", "description", "category") sağlayabilir.
"""
from __future__ import annotations

import importlib
import importlib.util
import math
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QObject, QPointF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence, QUndoStack
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QLineEdit,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from spt.control.errors import ControlError
from spt.gui.control.diagram_canvas import GRID, BlockItem, DiagramScene, DiagramView, WireItem
from spt.gui.control.diagram_palette import BlockPalette
from spt.gui.control.diagram_params import ParamPanel, fmt_number
from spt.gui.control.diagram_scope import ScopePanel
from spt.gui.icons import icon
from spt.gui.widgets.common import Card, StatusBadge, add_form_row, combo, form_layout, hbox, icon_button, label

TEMPLATE_MODULES: tuple[str, ...] = ("spt.control.uav.diagram_templates",)


# ============================================================================ şablonlar
@dataclass
class Template:
    name: str
    build: Callable[[], Any]
    description: str = ""
    category: str = "Genel"


_TEMPLATES: dict[str, Template] = {}


def register_template(name: str, build: Callable[[], Any], description: str = "", category: str = "Genel") -> None:
    """Şablon kaydeder: `build()` motorun `Model`'ini döndürür (bloklarda `meta['pos']` yoksa otomatik yerleşim)."""
    _TEMPLATES[name] = Template(name, build, description, category)


def templates() -> list[Template]:
    """Kayıtlı şablonlar (eklenti modülleri dahil), kategoriye göre."""
    for mod_name in TEMPLATE_MODULES:
        try:
            if importlib.util.find_spec(mod_name) is None:
                continue
        except (ImportError, ValueError):
            continue
        try:
            mod = importlib.import_module(mod_name)
        except Exception:  # eklenti hatası düzenleyiciyi bozmamalı
            continue
        items = getattr(mod, "TEMPLATES", None)
        if items is None and callable(getattr(mod, "templates", None)):
            try:
                items = mod.templates()
            except Exception:
                items = None
        for d in items or []:
            if isinstance(d, Template):
                _TEMPLATES.setdefault(d.name, d)
            elif isinstance(d, dict) and "name" in d and "build" in d:
                _TEMPLATES.setdefault(d["name"], Template(d["name"], d["build"], d.get("description", ""),
                                                          d.get("category", "Otopilot")))
    return sorted(_TEMPLATES.values(), key=lambda t: (t.category != "Genel", t.category, t.name))


def _pos(m, name: str, x: float, y: float) -> None:
    m.blocks[name].meta["pos"] = [float(x), float(y)]


def _layout_loop(m, pid: str) -> None:
    """Tek çevrim şablonunun yerleşimi: portlar aynı satırda (düz teller); referans skopa üstten dolaşır."""
    for n, x, y in (("Referans", 0, 40), ("Hata", 140, 40), (pid, 240, 40), ("Bitki", 400, 30), ("Çıkış", 600, 30)):
        _pos(m, n, x, y)
    for c in m.connections:
        if c.src == "Referans" and c.dst == "Çıkış":       # Referans çıkışı (60, 60) → Çıkış u1 (600, 40)
            c.meta = {"vertices": [[100.0, 60.0], [100.0, 0.0], [560.0, 0.0], [560.0, 40.0]], "orient": [True, True]}


def _tpl_pid_continuous():
    from spt.control.diagram import Model

    m = Model("PID geri besleme")
    m.add("Step", "Referans", step_time=0.0)
    m.add("Sum", "Hata", signs="+-")
    m.add("PIDController", "PID", P=2.0, I=1.0, D=0.2, N=50.0)
    m.add("TransferFcn", "Bitki", num=[1.0], den=[1.0, 2.0, 1.0])
    m.add("Scope", "Çıkış", inputs=2, labels="r, y")
    m.connect("Referans", "Hata")
    m.connect("Hata", "PID")
    m.connect("PID", "Bitki")
    m.connect("Bitki", "Hata", dst_port=1)
    m.connect("Referans", "Çıkış")
    m.connect("Bitki", "Çıkış", dst_port=1)
    _layout_loop(m, "PID")
    m.settings.update(solver="ode45", stop_time=10.0)
    return m


def _tpl_pid_discrete():
    from spt.control.diagram import Model

    m = Model("Ayrık PID + sürekli bitki")
    m.add("Step", "Referans", step_time=0.0)
    m.add("Sum", "Hata", signs="+-")
    m.add("PIDController", "PID (ayrık)", P=2.0, I=1.0, D=0.2, N=50.0, Ts=0.01,
          integrator_method="forward_euler", filter_method="forward_euler")
    m.add("TransferFcn", "Bitki", num=[1.0], den=[1.0, 2.0, 1.0])
    m.add("Scope", "Çıkış", inputs=2, labels="r, y")
    m.connect("Referans", "Hata")
    m.connect("Hata", "PID (ayrık)")
    m.connect("PID (ayrık)", "Bitki")
    m.connect("Bitki", "Hata", dst_port=1)
    m.connect("Referans", "Çıkış")
    m.connect("Bitki", "Çıkış", dst_port=1)
    _layout_loop(m, "PID (ayrık)")
    m.settings.update(solver="ode4", step=0.001, stop_time=10.0)
    return m


register_template("PID ile geri besleme (sürekli)", _tpl_pid_continuous,
                  "Basamak referans, sürekli PID ve ikinci dereceden bitki; r ve y skopta.")
register_template("Ayrık PID + sürekli bitki (çok hızlı)", _tpl_pid_discrete,
                  "Ts = 10 ms ayrık PID, sürekli bitki (örneklenmiş veri sistemi), ode4 1 ms.")


# ============================================================================ arka plan benzetimi
class _Bridge(QObject):
    progress = pyqtSignal(float, float)
    done = pyqtSignal(object, object)          # (SimResult | None, hata iletisi | None)


SOLVER_TEXT = {"ode1": "ode1 · Euler (sabit)", "ode4": "ode4 · RK4 (sabit)", "ode45": "ode45 · Dormand–Prince"}
_QUOTED = re.compile(r"'([^'\n]+)'")


class DiagramEditor(QWidget):
    """Blok diyagram düzenleyicisi.

    Sinyaller: `model_created(model, ad)` — Doğrusallaştır sonucu (LTI durum uzayı); `run_finished(sonuç | None)`;
    `status_changed(metin, durum)`."""

    model_created = pyqtSignal(object, str)
    run_finished = pyqtSignal(object)
    status_changed = pyqtSignal(str, str)

    def __init__(self, theme, workspace=None, parent: QWidget | None = None):
        super().__init__(parent)
        self._theme = theme
        self.undo_stack = QUndoStack(self)
        self.scene = DiagramScene(theme, self.undo_stack, self)
        self.view = DiagramView(self.scene)
        self.path: Path | None = None
        self.model_name = "model"
        self._signals: dict[tuple[str, int], dict] = {}
        self._model_meta: dict = {}
        self._sources: list[tuple[str, str, object]] = []
        self._thread: threading.Thread | None = None
        self._cancel: threading.Event | None = None
        self._bridge = _Bridge(self)
        self._bridge.progress.connect(self._on_progress)
        self._bridge.done.connect(self._on_done)
        self.last_result = None
        self.last_error: str | None = None
        self._lin_count = 0
        self._workspace = None
        self._build()
        self.scene.selectionChanged.connect(self._selection_changed)
        self.scene.message.connect(self._set_status)
        self.scene.context_requested.connect(self._context_menu)
        self.scene.block_double_clicked.connect(self._block_double_clicked)
        self.scene.changed_doc.connect(self._doc_changed)
        self.scene.block_renamed.connect(self._block_renamed)
        self.view.block_dropped.connect(lambda t, p: self.add_block(t, pos=p, center=True))
        self.palette.block_activated.connect(self._add_at_center)
        self.params.status.connect(self._set_status)
        self._set_status("Kütüphaneden blok sürükleyin ya da Şablonlar'dan başlayın. Bağlamak için çıkış portundan "
                         "giriş portuna sürükleyin.", "idle")
        if workspace is not None:
            self.set_workspace(workspace)

    # ================================================================ kurulum
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)
        root.addWidget(self._build_toolbar())

        self.palette = BlockPalette()
        pal_card = Card("Kütüphane")
        pal_card.body.addWidget(self.palette, 1)
        pal_card.setMinimumWidth(200)

        canvas_card = Card("Diyagram")
        self.title_label = label("adsız", "Muted")
        canvas_card.add_left(self.title_label, optional=True)
        for name, tip, fn in (("ph.magnifying-glass-plus", "Yakınlaştır", lambda: self.view.zoom_by(1.25)),
                              ("ph.minus", "Uzaklaştır", lambda: self.view.zoom_by(0.8)),
                              ("ph.corners-out", "Sığdır (Ctrl+0)", self.view.fit_all)):
            b = icon_button(name, tip, object_name="FlatIcon", size=16)
            b.clicked.connect(fn)
            canvas_card.add_right(b)
        canvas_card.body.addWidget(self.view, 1)
        # durum satırı tuvalin ALTINDA ayrı satırda (tuvalin üstüne binip blokları örtmez)
        self.status_badge = StatusBadge("Hazır", "idle")
        self.status_badge.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        self.progress.setFixedWidth(140)
        self.progress.hide()
        self.status_overlay = hbox(self.status_badge, self.progress, spacing=10)   # (ad geriye uyum için)
        self.status_overlay.layout().setStretch(0, 1)

        self.scopes = ScopePanel(self._theme)
        scope_card = Card("Benzetim sonuçları")
        scope_card.body.addWidget(self.scopes, 1)
        self.scope_card = scope_card

        # Diyagram ve sonuçlar ayrı sekmelerde: tuval ve grafikler küçük ekranda da tam boy. Yan yana görmek için
        # sonuç kartı başlığındaki düğmeyle ayrı pencereye alınır (Simulink'teki Scope penceresi gibi).
        def page(card: Card) -> QWidget:
            host = QWidget()
            lay = QVBoxLayout(host)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(card)
            return host

        self.center_tabs = QTabWidget()
        self.center_tabs.setDocumentMode(True)
        self.center_tabs.addTab(page(canvas_card), icon("ph.flow-arrow"), "Diyagram")
        self.center_tabs.addTab(page(scope_card), icon("ph.chart-line"), "Sonuçlar")
        self.center_tabs.currentChanged.connect(self._center_tab_changed)
        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(6)
        cl.addWidget(self.center_tabs, 1)
        cl.addWidget(self.status_overlay)
        self.vsplit = center                            # (ad geriye uyum için: orta sütun)
        canvas_card.setMinimumHeight(220)

        self.params = ParamPanel(self._theme)
        par_card = Card("Blok parametreleri")
        par_card.body.addWidget(self.params, 1)
        par_card.setMinimumWidth(240)

        self.hsplit = QSplitter(Qt.Orientation.Horizontal)
        self.hsplit.setChildrenCollapsible(False)
        self.hsplit.addWidget(pal_card)
        self.hsplit.addWidget(center)
        self.hsplit.addWidget(par_card)
        self.hsplit.setStretchFactor(0, 0)
        self.hsplit.setStretchFactor(1, 1)
        self.hsplit.setStretchFactor(2, 0)
        self.hsplit.setSizes([230, 880, 280])
        root.addWidget(self.hsplit, 1)
        self._build_actions()

    def _center_tab_changed(self, index: int) -> None:
        if index == 0:
            QTimer.singleShot(0, self._fit_if_needed)

    def _fit_if_needed(self) -> None:
        """Diyagram sekmesine dönünce bloklar görünür alanın dışındaysa sığdır."""
        if not self.scene.blocks:
            return
        vis = self.view.mapToScene(self.view.viewport().rect()).boundingRect()
        if not vis.contains(self.scene.itemsBoundingRect()):
            self.view.fit_all()

    def show_results(self) -> None:
        """Sonuçlar sekmesine geç (sonuç kartı ayrı penceredeyse o pencere öne gelir)."""
        page_index = 1
        if self.center_tabs.widget(page_index).isAncestorOf(self.scope_card):
            self.center_tabs.setCurrentIndex(page_index)
        else:
            win = self.scope_card.window()
            win.raise_()
            win.activateWindow()

    def _place_overlay(self) -> None:  # durum satırı artık yerleşimde (eski API)
        return

    def _tool(self, icon_name: str, tip: str, fn, text: str = "") -> QToolButton:
        b = icon_button(icon_name, tip, size=16)
        if text:
            b.setText(text)
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        b.clicked.connect(lambda *_: fn())
        return b

    def _build_toolbar(self) -> QWidget:
        self.btn_new = self._tool("ph.file-plus", "Yeni diyagram (Ctrl+N)", self.new_diagram)
        self.btn_open = self._tool("ph.folder-open", "Aç… (.sptd, Ctrl+O)", self.open_dialog)
        self.btn_save = self._tool("ph.floppy-disk", "Kaydet (Ctrl+S)", self.save_dialog)
        self.btn_templates = QToolButton()
        self.btn_templates.setObjectName("MenuButton")
        self.btn_templates.setText("Şablonlar")
        self.btn_templates.setToolTip("Hazır diyagramlar (otopilot şablonları İHA modülüyle eklenir)")
        self.btn_templates.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.template_menu = QMenu(self.btn_templates)
        self.template_menu.aboutToShow.connect(self._fill_templates)
        self.btn_templates.setMenu(self.template_menu)
        self.btn_undo = self._tool("ph.arrow-counter-clockwise", "Geri al (Ctrl+Z)", self.undo_stack.undo)
        self.btn_redo = self._tool("ph.arrow-clockwise", "Yinele (Ctrl+Y)", self.undo_stack.redo)
        self.undo_stack.canUndoChanged.connect(self.btn_undo.setEnabled)
        self.undo_stack.canRedoChanged.connect(self.btn_redo.setEnabled)
        self.btn_undo.setEnabled(False)
        self.btn_redo.setEnabled(False)

        self.btn_run = QPushButton(icon("ph.play"), "Çalıştır")
        self.btn_run.setObjectName("Primary")
        self.btn_run.setToolTip("Benzetimi arka planda çalıştır (Ctrl+T)")
        self.btn_run.clicked.connect(lambda *_: self.run())
        self.btn_stop = QPushButton(icon("ph.stop"), "Durdur")
        self.btn_stop.setToolTip("Benzetimi durdur (o ana kadarki sonuç gösterilir)")
        self.btn_stop.clicked.connect(lambda *_: self.stop())
        self.btn_stop.setEnabled(False)

        from spt.control.diagram.engine import SOLVERS

        self.solver = combo([SOLVER_TEXT.get(k, k) for k in SOLVERS], list(SOLVERS).index("ode4"))
        for i, k in enumerate(SOLVERS):
            self.solver.setItemData(i, k)
            self.solver.setItemData(i, f"{k}: {SOLVERS[k]}", Qt.ItemDataRole.ToolTipRole)
        self.solver.setToolTip("Çözücü: ode1 / ode4 sabit adımlı, ode45 değişken adımlı (MATLAB ode45 adım denetimi)")
        self.solver.currentIndexChanged.connect(self._solver_changed)
        self.stop_edit = QLineEdit("10")
        self.stop_edit.setFixedWidth(64)
        self.stop_edit.setToolTip("Bitiş zamanı (s)")
        self.step_edit = QLineEdit("")
        self.step_edit.setPlaceholderText("otomatik")
        self.step_edit.setFixedWidth(80)
        self.step_edit.setToolTip("Sabit adım (s); boş: otomatik (ayrık hızların ortak böleni ya da süre / 1000)")
        self.btn_solver = QToolButton()
        self.btn_solver.setObjectName("IconButton")
        self.btn_solver.setIcon(icon("ph.sliders-horizontal"))
        self.btn_solver.setToolTip("Çözücü ayarları: toleranslar, en büyük adım, sıfır geçiş")
        self.btn_solver.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.btn_solver.setMenu(self._solver_menu())

        self.btn_lin = QToolButton()
        self.btn_lin.setObjectName("MenuButton")
        self.btn_lin.setText("Doğrusallaştır")
        self.btn_lin.setToolTip("Tellerde seçilen doğrusallaştırma noktaları (sağ tık → Doğrusallaştırma noktası) "
                                "arasında doğrusal model çıkar ve Modeller'e ekle")
        self.btn_lin.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        lin_menu = QMenu(self.btn_lin)
        lin_menu.addAction("Başlangıç koşullarında", lambda: self.linearize("initial"))
        lin_menu.addAction("Son benzetimin bitiş anında", lambda: self.linearize("last"))
        self.btn_lin.setMenu(lin_menu)
        self.btn_lin.clicked.connect(lambda *_: self.linearize("initial"))

        bar = hbox(self.btn_new, self.btn_open, self.btn_save, self.btn_templates, self.btn_undo, self.btn_redo,
                   spacing=6)
        lay = bar.layout()
        lay.addSpacing(12)
        lay.addWidget(self.btn_run)
        lay.addWidget(self.btn_stop)
        lay.addSpacing(12)
        lay.addWidget(label("Çözücü", "FormLabel"))
        lay.addWidget(self.solver)
        lay.addWidget(label("Bitiş (s)", "FormLabel"))
        lay.addWidget(self.stop_edit)
        lay.addWidget(label("Adım (s)", "FormLabel"))
        lay.addWidget(self.step_edit)
        lay.addWidget(self.btn_solver)
        lay.addStretch(1)
        lay.addWidget(self.btn_lin)
        return bar

    def _solver_menu(self) -> QMenu:
        menu = QMenu(self)
        box = QWidget()
        form = form_layout()
        self.rtol_edit = QLineEdit("1e-3")
        self.rtol_edit.setToolTip("Bağıl tolerans (ode45)")
        self.atol_edit = QLineEdit("auto")
        self.atol_edit.setToolTip("Mutlak tolerans (ode45; 'auto': Simulink gibi durum büyüklüğünden)")
        self.maxstep_edit = QLineEdit("")
        self.maxstep_edit.setPlaceholderText("otomatik")
        self.maxstep_edit.setToolTip("En büyük adım (ode45; boş: otomatik)")
        self.zc_check = QCheckBox("Sıfır geçiş tespiti (ode45)")
        self.zc_check.setChecked(True)
        add_form_row(form, "Bağıl tolerans", self.rtol_edit)
        add_form_row(form, "Mutlak tolerans", self.atol_edit)
        add_form_row(form, "En büyük adım (s)", self.maxstep_edit)
        form.addRow(self.zc_check)
        box.setLayout(form)
        box.setMinimumWidth(300)
        act = QWidgetAction(menu)
        act.setDefaultWidget(box)
        menu.addAction(act)
        return menu

    def _build_actions(self) -> None:
        ctx = Qt.ShortcutContext.WidgetWithChildrenShortcut

        def act(text: str, keys, fn) -> QAction:
            a = QAction(text, self)
            a.setShortcuts([QKeySequence(k) for k in keys])
            a.setShortcutContext(ctx)
            a.triggered.connect(lambda *_: fn())
            self.addAction(a)
            return a

        self.act_delete = act("Sil", [QKeySequence.StandardKey.Delete, "Backspace"], self._delete_if_canvas)
        self.act_copy = act("Kopyala", [QKeySequence.StandardKey.Copy], self._copy_if_canvas)
        self.act_cut = act("Kes", [QKeySequence.StandardKey.Cut], self._cut_if_canvas)
        self.act_paste = act("Yapıştır", [QKeySequence.StandardKey.Paste], self._paste_if_canvas)
        self.act_undo = act("Geri al", [QKeySequence.StandardKey.Undo], self.undo_stack.undo)
        self.act_redo = act("Yinele", [QKeySequence.StandardKey.Redo, "Ctrl+Y"], self.undo_stack.redo)
        self.act_all = act("Tümünü seç", [QKeySequence.StandardKey.SelectAll], self._select_all_if_canvas)
        self.act_flip = act("Çevir", ["Ctrl+I"], lambda: self.scene.flip(self.scene.selected_blocks()))
        self.act_save = act("Kaydet", [QKeySequence.StandardKey.Save], self.save_dialog)
        self.act_open = act("Aç", [QKeySequence.StandardKey.Open], self.open_dialog)
        self.act_new = act("Yeni", [QKeySequence.StandardKey.New], self.new_diagram)
        self.act_run = act("Çalıştır", ["Ctrl+T"], self.run)
        self.act_fit = act("Sığdır", ["Ctrl+0"], self.view.fit_all)

    def _canvas_focus(self) -> bool:
        w = QApplication.focusWidget()
        return w is None or w is self.view or self.view.isAncestorOf(w)

    def _delete_if_canvas(self) -> None:
        if self._canvas_focus():
            self.scene.delete_selection()

    def _copy_if_canvas(self) -> None:
        if self._canvas_focus():
            self.scene.copy_selection()

    def _cut_if_canvas(self) -> None:
        if self._canvas_focus() and self.scene.copy_selection():
            self.scene.remove_items(self.scene.selected_blocks(), self.scene.selected_wires(), "Kes")

    def _paste_if_canvas(self) -> None:
        if self._canvas_focus():
            self.scene.paste()

    def _select_all_if_canvas(self) -> None:
        if self._canvas_focus():
            for it in list(self.scene.blocks.values()) + list(self.scene.wires):
                it.setSelected(True)

    # ================================================================ durum
    def _set_status(self, text: str, state: str = "idle") -> None:
        self._status = (text, state)
        self.status_badge.set_state(text, state)
        self.status_badge.setToolTip(text)
        pass
        self.status_changed.emit(text, state)

    @property
    def status_text(self) -> str:
        return self._status[0]

    @property
    def status_state(self) -> str:
        return self._status[1]

    def _doc_changed(self) -> None:
        name = self.path.name if self.path else "adsız"
        self.title_label.setText(name + ("" if self.undo_stack.isClean() else " •"))
        b = self.params.block
        if b is not None:
            item = self.scene.blocks.get(b.name)
            if item is None or item.block is not b:
                self.params.clear()
            else:
                self.params.refresh()

    def _block_renamed(self, old: str, new: str) -> None:
        """Adlandırılmış sinyaller (dosyadan gelen) blok adıyla anahtarlanır: yeni ada taşınır."""
        self._signals = {((new if b == old else b), p): v for (b, p), v in self._signals.items()}

    def _solver_changed(self, *_args) -> None:
        var = self.solver.currentData() == "ode45"
        self.step_edit.setPlaceholderText("—" if var else "otomatik")
        self.step_edit.setEnabled(not var)

    # ================================================================ bloklar
    def add_block(self, type_name: str, name: str | None = None, pos: QPointF | tuple | None = None,
                  center: bool = False, **params: Any) -> str:
        """Blok ekler (geri alınabilir) ve adını döndürür. Hata (bilinmeyen tür, geçersiz parametre) `ControlError`;
        tuvalden gelen isteklerde durum satırında gösterilir."""
        from spt.control.diagram.block import block_class, create_block

        try:
            cls = block_class(type_name)
            disp = cls.display_name.split("/")[0].strip() or cls.display_name   # "Skop / kayıt" → "Skop"
            base = name or disp[:1].upper() + disp[1:]
            block = create_block(type_name, self.scene.unique_name(base) if name is None else name, **params)
        except ControlError as exc:
            self._set_status(str(exc), "critical")
            raise
        if pos is None:
            c = self.view.mapToScene(self.view.viewport().rect().center())
            pos = QPointF(c.x(), c.y())
            center = True
        p = pos if isinstance(pos, QPointF) else QPointF(float(pos[0]), float(pos[1]))
        if center:
            probe = BlockItem(block, self._theme.tokens)
            r = probe.body_rect()
            p = QPointF(p.x() - r.width() / 2, p.y() - r.height() / 2)
            block.meta.pop("pos", None)
        item = self.scene.add_block(block, p)
        self.scene.clearSelection()
        item.setSelected(True)
        return item.name

    def _add_at_center(self, type_name: str) -> None:
        try:
            self.add_block(type_name)
        except ControlError:
            pass

    def connect(self, src: str, dst: str, src_port: int = 0, dst_port: int = 0) -> WireItem:
        """(src, çıkış portu) → (dst, giriş portu) teli (0 tabanlı, geri alınabilir)."""
        return self.scene.connect(src, src_port, dst, dst_port)

    def branch(self, wire: WireItem, dst: str, dst_port: int = 0, anchor: float = 0.5) -> WireItem:
        """Var olan telin `anchor` kesrindeki noktasından dal çeker (aynı kaynak sinyal)."""
        return self.scene.connect(wire.src, wire.src_port, dst, dst_port, parent=wire, anchor=anchor)

    def remove(self, *names: str) -> None:
        self.scene.remove_items([self.scene.blocks[n] for n in names], [])

    def set_params(self, name: str, new_name: str | None = None, **changes: Any) -> None:
        """Blok parametrelerini doğrulayıp uygular (geri alınabilir); hatada `ControlError`, blok değişmez."""
        self.scene.set_params(self.scene.blocks[name], new_name, changes)

    def block(self, name: str):
        return self.scene.blocks[name].block

    def set_lin_point(self, block: str, port: int = 0, kind: str | None = "input") -> None:
        """Sinyal (blok çıkış portu) üzerinde doğrusallaştırma noktası; kind None → kaldır."""
        from spt.control.diagram.linearize import POINT_TYPES

        item = self.scene.blocks[block]
        if kind is not None and kind not in POINT_TYPES:
            raise ControlError(f"Bilinmeyen doğrusallaştırma noktası türü '{kind}'.")
        item.port("out", port)
        cur = {str(k): v for k, v in (item.block.meta.get("lin_io") or {}).items()}
        if kind is None:
            cur.pop(str(port), None)
        else:
            cur[str(port)] = kind
        text = "Doğrusallaştırma noktası" if kind else "Doğrusallaştırma noktasını kaldır"
        self.scene.set_meta(item, "lin_io", cur or None, text)

    def lin_points(self) -> list[tuple[str, int, str]]:
        out = []
        for name, item in self.scene.blocks.items():
            for port, kind in sorted(item.lin_io().items()):
                if 0 <= port < item.n_out():
                    out.append((name, port, kind))
        return out

    def set_sources(self, sources: list[tuple[str, str, object]]) -> None:
        """Veri kaynağı bloğunun bağlanabileceği kaynaklar: (kimlik, etiket, tutamaç) — akış / kayıt / anlık görüntü."""
        self._sources = list(sources)
        self.params.set_sources(self._sources)

    def set_workspace(self, workspace) -> None:
        """Doğrusallaştırma sonuçlarını bu çalışma alanına (ModelWorkspace) ekler."""
        self._workspace = workspace
        self.model_created.connect(lambda m, n: workspace.add(n, m, source="Blok diyagram",
                                                              note="Blok diyagramdan doğrusallaştırıldı"))

    # ================================================================ seçim / bağlam menüsü
    def _selection_changed(self) -> None:
        try:
            blocks = self.scene.selected_blocks()
        except RuntimeError:          # sahne kapanırken
            return
        if len(blocks) == 1:
            item = blocks[0]
            self.params.set_target(item.block, lambda n, ch, it=item: self.scene.set_params(it, n, ch))
        elif self.params.block is not None:
            self.params.clear()

    def _block_double_clicked(self, name: str) -> None:
        item = self.scene.blocks.get(name)
        if item is not None and item.block.type_name == "Scope" and name in self.scopes.cards:
            self.scopes.scroll.ensureWidgetVisible(self.scopes.cards[name])
        self.params.setFocus()

    def _context_menu(self, item, screen_pos, scene_pos) -> None:
        menu = QMenu(self)
        if isinstance(item, BlockItem):
            menu.addAction("Çevir (Ctrl+I)", lambda: self.scene.flip(self.scene.selected_blocks()))
            menu.addAction("Kopyala (Ctrl+C)", self.scene.copy_selection)
            menu.addSeparator()
            menu.addAction("Sil", self.scene.delete_selection)
        elif isinstance(item, WireItem):
            from spt.control.diagram.linearize import POINT_TYPES

            sub = menu.addMenu("Doğrusallaştırma noktası")
            cur = item.src.lin_io().get(item.src_port)
            none = sub.addAction("Yok", lambda w=item: self.set_lin_point(w.src.name, w.src_port, None))
            none.setCheckable(True)
            none.setChecked(cur is None)
            for key, text in POINT_TYPES.items():
                a = sub.addAction(text, lambda w=item, k=key: self.set_lin_point(w.src.name, w.src_port, k))
                a.setCheckable(True)
                a.setChecked(cur == key)
            entry = self._signals.get((item.src.name, item.src_port), {})
            log = menu.addAction("Sinyali kaydet (Sonuçlar'da göster)",
                                 lambda w=item, on=not entry.get("log"): self.set_signal_log(w.src.name, w.src_port, on))
            log.setCheckable(True)
            log.setChecked(bool(entry.get("log")))
            menu.addSeparator()
            hint = menu.addAction("Dal çekmek için: Ctrl + telden sürükle")
            hint.setEnabled(False)
            menu.addAction("Sil", self.scene.delete_selection)
        else:
            menu.addAction("Yapıştır (Ctrl+V)", lambda: self.scene.paste())
            menu.addAction("Sığdır (Ctrl+0)", self.view.fit_all)
        menu.popup(screen_pos)

    def set_signal_log(self, block: str, port: int, on: bool = True, name: str | None = None) -> None:
        """Telin taşıdığı sinyali benzetim sonucuna kaydet / kaydetme (Simulink "Log signal" karşılığı); sonuç
        Sonuçlar sekmesinde kendi kartında görünür. name: isteğe bağlı sinyal adı (yoksa "Blok" ya da "Blok:2")."""
        entry = self._signals.setdefault((block, int(port)), {"name": None, "log": False})
        entry["log"] = bool(on)
        if name:
            entry["name"] = str(name)
        if not entry.get("log") and not entry.get("name"):
            self._signals.pop((block, int(port)), None)
        self._doc_changed()
        self._set_status(f"'{block}' çıkışı {'kaydedilecek' if on else 'artık kaydedilmeyecek'}; sonuç için "
                         "yeniden çalıştırın.", "good" if on else "idle")

    # ================================================================ model ↔ tuval
    def solver_settings(self) -> dict:
        """Araç çubuğundaki çözücü ayarları (doğrulanmış). Hatalıysa `ControlError`."""
        def num(edit: QLineEdit, what: str, allow_empty: bool = True, pos: bool = True) -> float | None:
            text = edit.text().strip()
            if not text:
                if allow_empty:
                    return None
                raise ControlError(f"{what} boş olamaz.")
            try:
                v = float(text)
            except ValueError:
                raise ControlError(f"{what} sayı olmalı (verilen '{text}').") from None
            if not math.isfinite(v) or (pos and v <= 0):
                raise ControlError(f"{what} pozitif ve sonlu olmalı.")
            return v

        st = {"solver": self.solver.currentData(), "stop_time": num(self.stop_edit, "Bitiş zamanı", False),
              "step": num(self.step_edit, "Adım"), "rtol": num(self.rtol_edit, "Bağıl tolerans", False),
              "max_step": num(self.maxstep_edit, "En büyük adım"), "zero_crossing": self.zc_check.isChecked()}
        at = self.atol_edit.text().strip().lower()
        st["atol"] = "auto" if at in ("", "auto", "otomatik") else num(self.atol_edit, "Mutlak tolerans")
        return st

    def _apply_settings(self, st: dict) -> None:
        i = self.solver.findData(st.get("solver", "ode4"))
        self.solver.setCurrentIndex(i if i >= 0 else self.solver.findData("ode4"))
        self.stop_edit.setText(fmt_number(st.get("stop_time", 10.0)))
        step = st.get("step")
        self.step_edit.setText("" if step in (None, "") else fmt_number(step))
        self.rtol_edit.setText(fmt_number(st.get("rtol", 1e-3)))
        atol = st.get("atol", "auto")
        self.atol_edit.setText("auto" if isinstance(atol, str) else fmt_number(atol))
        ms = st.get("max_step")
        self.maxstep_edit.setText("" if ms in (None, "") else fmt_number(ms))
        self.zc_check.setChecked(bool(st.get("zero_crossing", True)))

    def to_model(self, settings: bool = True):
        """Tuvaldeki diyagramın motor modeli (aynı Block nesneleri; kopya için `.copy()`)."""
        from spt.control.diagram.model import DEFAULT_SETTINGS, Model

        m = Model(self.model_name)
        for name, item in self.scene.blocks.items():
            m.add(item.block, name)
        index = {w: i for i, w in enumerate(self.scene.wires)}
        for w in self.scene.wires:
            meta: dict[str, Any] = {}
            if w.vertices:
                meta["vertices"] = [[float(v.x()), float(v.y())] for v in w.vertices]
                meta["orient"] = [bool(w._h_first), bool(w._h_last)]
            if w.parent_wire is not None and w.parent_wire in index:
                meta["parent"] = index[w.parent_wire]
                meta["anchor"] = float(w.anchor)
            m.connect(w.src.name, w.dst.name, w.src_port, w.dst_port, meta=meta)
        for (b, p), entry in self._signals.items():
            if b in m.blocks and p < m.blocks[b].n_outputs():
                m.signals[(b, p)] = dict(entry)
        m.meta = dict(self._model_meta)
        if settings:
            try:
                m.settings.update(self.solver_settings())
            except ControlError:
                m.settings = dict(DEFAULT_SETTINGS)
        return m

    def load_model(self, model, path: Path | None = None) -> None:
        """Motor modelini tuvale yükler (geri al geçmişi temizlenir)."""
        self.stop()
        self.scene.clear_all()
        self.params.clear()
        self.scopes.clear()
        self.center_tabs.setTabText(1, "Sonuçlar")
        self.last_result = None
        self.model_name = model.name
        self._signals = {k: dict(v) for k, v in model.signals.items()}
        self._model_meta = dict(model.meta)
        if any("pos" not in b.meta for b in model.blocks.values()):
            _auto_layout(model)
        for name, b in model.blocks.items():
            b.name = name
            pos = b.meta.get("pos", [0.0, 0.0])
            self.scene.add_block(b, QPointF(float(pos[0]), float(pos[1])))
        wires: list[WireItem] = []
        for c in model.connections:
            meta = c.meta or {}
            par = meta.get("parent")
            parent = wires[par] if isinstance(par, int) and 0 <= par < len(wires) else None
            verts = [QPointF(float(x), float(y)) for x, y in meta.get("vertices") or []] or None
            try:
                w = self.scene.connect(c.src, c.src_port, c.dst, c.dst_port, vertices=verts, parent=parent,
                                       anchor=float(meta.get("anchor", 0.5)))
            except (ControlError, KeyError) as exc:
                self._set_status(f"Bağlantı yüklenemedi ({c.src} → {c.dst}): {exc}", "warning")
                continue
            orient = meta.get("orient")
            if verts and isinstance(orient, list) and len(orient) == 2:
                w._h_first, w._h_last = bool(orient[0]), bool(orient[1])
                w._last_ends = None
                w.update_path()
            wires.append(w)
        self._apply_settings(model.settings)
        self.undo_stack.clear()
        self.undo_stack.setClean()
        self.path = path
        self.scene.refresh_all()
        self._doc_changed()
        self.view.fit_all()

    def new_diagram(self, confirm: bool = True) -> bool:
        if confirm and not self._confirm_discard():
            return False
        from spt.control.diagram.model import DEFAULT_SETTINGS, Model

        m = Model("model")
        m.settings = dict(DEFAULT_SETTINGS)
        self.load_model(m)
        self._set_status("Yeni diyagram.", "idle")
        return True

    def _confirm_discard(self) -> bool:
        if self.undo_stack.isClean() or not self.scene.blocks:
            return True
        ans = QMessageBox.question(self, "Kaydedilmemiş değişiklikler",
                                   "Diyagramda kaydedilmemiş değişiklikler var. Yine de devam edilsin mi?")
        return ans == QMessageBox.StandardButton.Yes

    def save(self, path: str | Path) -> Path:
        """`.sptd` olarak kaydeder (motorun Model.save'i); kaydedilen yolu döndürür."""
        m = self.to_model()
        p = Path(path)
        if self.model_name in ("model", "") or (self.path is None):
            m.name = p.stem
        saved = m.save(p)
        self.model_name = m.name
        self.path = saved
        self.undo_stack.setClean()
        self._doc_changed()
        self._set_status(f"Kaydedildi: {saved.name}", "good")
        return saved

    def open(self, path: str | Path) -> None:
        from spt.control.diagram.model import Model

        m = Model.load(path)
        self.load_model(m, Path(path))
        self._set_status(f"Açıldı: {Path(path).name} · {len(m.blocks)} blok, {len(m.connections)} bağlantı", "good")

    def save_dialog(self) -> None:
        start = str(self.path) if self.path else f"{self.model_name}.sptd"
        path, _ = QFileDialog.getSaveFileName(self, "Blok diyagramı kaydet", start, "Blok diyagram (*.sptd)")
        if path:
            try:
                self.save(path)
            except (ControlError, OSError) as exc:
                self._set_status(f"Kaydedilemedi: {exc}", "critical")

    def open_dialog(self) -> None:
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Blok diyagramı aç", "", "Blok diyagram (*.sptd);;Tüm dosyalar (*)")
        if path:
            try:
                self.open(path)
            except ControlError as exc:
                self._set_status(str(exc), "critical")

    # ================================================================ şablonlar
    def _fill_templates(self) -> None:
        menu = self.template_menu
        menu.clear()
        items = templates()
        cats: dict[str, list[Template]] = {}
        for t in items:
            cats.setdefault(t.category, []).append(t)
        for cat, lst in cats.items():
            sec = menu.addAction(cat)
            sec.setEnabled(False)
            for t in lst:
                a = menu.addAction(t.name, lambda tt=t: self.load_template(tt.name))
                a.setToolTip(t.description)
            menu.addSeparator()
        if not any(t.category == "Otopilot" for t in items):
            a = menu.addAction("Otopilot şablonları (İHA modülüyle eklenecek)")
            a.setEnabled(False)

    def load_template(self, name: str) -> None:
        t = {tt.name: tt for tt in templates()}.get(name)
        if t is None:
            raise ControlError(f"'{name}' adlı şablon yok.")
        if not self._confirm_discard():
            return
        m = t.build()
        self.load_model(m)
        self.path = None
        self.undo_stack.resetClean()
        self._doc_changed()
        self._set_status(f"Şablon yüklendi: {t.name}. {t.description}", "good")

    # ================================================================ benzetim
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _check_structure(self, model) -> bool:
        rep = model.validate(compile=False)
        if not rep.ok:
            errs = rep.errors
            self._error([i.message for i in errs], {i.block for i in errs if i.block})
            return False
        return True

    def _error(self, messages: list[str], blocks: set[str] | None = None) -> None:
        text = messages[0] + (f" (+{len(messages) - 1} hata daha)" if len(messages) > 1 else "")
        names = set(blocks or set())
        for msg in messages:
            names |= {n for n in _QUOTED.findall(msg) if n in self.scene.blocks}
        self.scene.highlight(names)
        self.last_error = "\n".join(messages)
        self._set_status(text, "critical")

    def run(self, wait: bool = False) -> bool:
        """Benzetimi başlatır (arka plan). `wait=True`: bitene kadar bekler (olay döngüsü işlenir; betik / test).
        Başlatılamazsa (hata) False."""
        if self.running:
            return False
        self.scene.highlight(set())
        self.last_error = None
        try:
            st = self.solver_settings()
        except ControlError as exc:
            self._error([str(exc)])
            return False
        if not self.scene.blocks:
            self._error(["Diyagram boş: kütüphaneden blok ekleyin."])
            return False
        try:
            model = self.to_model(settings=False)
        except ControlError as exc:              # (ör. geçersiz blok adı) istisna yerine durum satırında Türkçe
            self._error([str(exc)])
            return False
        if not self._check_structure(model):
            return False
        model = model.copy()             # benzetim, arayüzdeki blok nesnelerinden bağımsız çalışır
        model.settings.update(st)
        self._run_order = self._scope_order()
        self._cancel = threading.Event()
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setValue(0)
        self.progress.show()
        self._set_status(f"Benzetim çalışıyor… ({st['solver']}, bitiş {fmt_number(st['stop_time'])} s)", "warning")
        self._t_start = time.perf_counter()

        def work(m=model, cancel=self._cancel, bridge=self._bridge):
            from spt.control.diagram.engine import simulate

            try:
                res = simulate(m, cancel=cancel, progress=lambda f, t: bridge.progress.emit(float(f), float(t)))
            except ControlError as exc:
                bridge.done.emit(None, str(exc))
                return
            except Exception as exc:  # beklenmeyen motor hatası arayüzü düşürmesin
                bridge.done.emit(None, f"Benzetim hatası: {type(exc).__name__}: {exc}")
                return
            bridge.done.emit(res, None)

        self._thread = threading.Thread(target=work, name="spt-diagram-sim", daemon=True)
        self._thread.start()
        if wait:
            self.wait()
        return True

    def wait(self, timeout: float = 120.0) -> bool:
        """Çalışan benzetimin bitmesini bekler (olay döngüsünü işleyerek)."""
        end = time.perf_counter() + timeout
        while self._thread is not None and time.perf_counter() < end:
            QApplication.processEvents()        # bitiş sinyali (kuyruklu) burada işlenir ve _thread None olur
            time.sleep(0.005)
        return self._thread is None

    def stop(self) -> None:
        if self._cancel is not None:
            self._cancel.set()
        if self.running:
            self._set_status("Durduruluyor…", "warning")

    def _on_progress(self, frac: float, t: float) -> None:
        self.progress.setValue(int(max(0.0, min(1.0, frac)) * 1000))
        self.status_badge.set_state(f"Benzetim çalışıyor… t = {t:.4g} s ({100 * frac:.0f} %)", "warning")

    def _on_done(self, result, error) -> None:
        self._thread = None
        self._cancel = None
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress.hide()
        if error is not None:
            self._error([error])
            self.run_finished.emit(None)
            return
        self.last_result = result
        self.scopes.set_result(result, getattr(self, "_run_order", None))
        n = self.scopes.count
        self.center_tabs.setTabText(1, f"Sonuçlar ({n})" if n else "Sonuçlar")
        if n:
            self.show_results()
        st = result.stats
        wall = st.get("wall_time", time.perf_counter() - self._t_start)
        if result.cancelled:
            msg = f"Benzetim durduruldu: t = {result.t[-1]:.4g} s'ye kadar sonuç ({st.get('steps', 0)} adım)."
            state = "warning"
        else:
            msg = (f"Benzetim bitti: {st.get('steps', 0)} adım, {st.get('solver', '')}, "
                   f"{wall:.2f} s" + (f", {st['rejected']} reddedilen adım" if st.get("rejected") else ""))
            state = "good"
        self._set_status(msg, state)
        self.run_finished.emit(result)

    # ================================================================ görünüm sözleşmesi (Kontrol Tasarımı sekmesi)
    def shutdown(self) -> None:
        """Uygulama kapanırken çalışan benzetimi durdurur."""
        self.stop()
        self.wait(5.0)

    def detachable_cards(self) -> list[tuple[str, str, Card]]:
        return [("scopes", "Blok diyagram · benzetim sonuçları", self.scope_card)]

    def save_state(self, s) -> None:
        """Ayarlar: bölücü boyları, çözücü ayarları ve (1 MB'tan küçükse) açık diyagram."""
        s.setValue("control/diagram/hsplit", self.hsplit.sizes())
        s.setValue("control/diagram/tab", self.center_tabs.currentIndex())
        s.setValue("control/diagram/path", str(self.path) if self.path else "")
        try:
            import json as _json

            text = _json.dumps(self.to_model().to_dict(), ensure_ascii=False, allow_nan=False)
        except (ControlError, ValueError, TypeError):
            text = ""
        s.setValue("control/diagram/model", text if len(text) < 1_000_000 else "")

    def restore_state(self, s) -> None:
        import json as _json

        for key, split in (("hsplit", self.hsplit),):
            sizes = s.value(f"control/diagram/{key}")
            try:
                sizes = [int(v) for v in sizes] if sizes else None
            except (TypeError, ValueError):
                sizes = None
            if sizes and len(sizes) == split.count() and all(v > 0 for v in sizes):
                split.setSizes(sizes)
        text = s.value("control/diagram/model", "")
        if not text:
            return
        from spt.control.diagram.model import Model

        try:
            m = Model.from_dict(_json.loads(str(text)))
        except (ControlError, ValueError, TypeError, KeyError):
            return
        path = s.value("control/diagram/path", "")
        self.load_model(m, Path(path) if path else None)
        if m.blocks:
            self.undo_stack.resetClean()          # dosyayla aynı olduğu bilinmiyor: kapatmadan önce sorulsun
            self._doc_changed()

    def _scope_order(self) -> list[str]:
        items = [it for it in self.scene.blocks.values() if it.block.type_name == "Scope"]
        items.sort(key=lambda it: (it.pos().y(), it.pos().x()))
        return [it.name for it in items]

    # ================================================================ doğrusallaştırma
    def linearize(self, at: str = "initial", name: str | None = None):
        """Doğrusallaştırma noktaları arasında doğrusal model; `model_created` ile yayınlar ve (model, ad) döndürür.
        at: "initial" (başlangıç koşulları) | "last" (son benzetimin bitiş anı). Hata durum satırında gösterilir."""
        from spt.control.diagram.linearize import linearize, linio

        pts = self.lin_points()
        try:
            if not pts:
                raise ControlError("Doğrusallaştırma noktası yok: bir tele sağ tıklayıp 'Doğrusallaştırma noktası' "
                                   "menüsünden giriş ve çıkış noktası seçin.")
            model = self.to_model(settings=False).copy()
            op = None
            if at == "last":
                if self.last_result is None:
                    raise ControlError("Henüz benzetim sonucu yok: önce Çalıştır'a basın ya da başlangıç koşullarında "
                                       "doğrusallaştırın.")
                op = self.last_result.final
            self.scene.highlight(set())
            lsys = linearize(model, [linio(b, p, k) for b, p, k in pts], op)
            try:
                sys = lsys.to_lti()
            except ControlError:
                sys = lsys
        except ControlError as exc:
            self._error([str(exc)])
            return None
        if name is None:
            name = self._next_lin_name()
        if hasattr(sys, "name"):
            try:
                sys.name = name
            except AttributeError:
                pass
        ny, nu = lsys.D.shape
        kind = "ayrık" if lsys.Ts > 0 else "sürekli"
        self._set_status(f"Doğrusallaştırıldı: {name} ({kind}, {lsys.A.shape[0]} durum, {nu} giriş, {ny} çıkış) → "
                         "Modeller'e eklendi.", "good")
        self.model_created.emit(sys, name)
        return sys, name

    def _next_lin_name(self) -> str:
        if self._workspace is not None and hasattr(self._workspace, "unique_name"):
            return self._workspace.unique_name("linsys")
        self._lin_count += 1
        return f"linsys{self._lin_count}"


# ============================================================================ yerleşim
def _auto_layout(model) -> None:
    """Konumu olmayan bloklar için soldan sağa katmanlı yerleşim: DFS ile geri besleme kenarları ayrılır, kalan
    döngüsüz grafikte en uzun yol katmanı sütunu verir."""
    names = list(model.blocks)
    succ: dict[str, list[str]] = {n: [] for n in names}
    for c in model.connections:
        if c.src in succ and c.dst in succ:
            succ[c.src].append(c.dst)
    indeg = {n: 0 for n in names}
    for n in names:
        for m in succ[n]:
            indeg[m] += 1
    color: dict[str, int] = {}
    back: set[tuple[str, str]] = set()
    for root in sorted(names, key=lambda k: (indeg[k] > 0, names.index(k))):
        if color.get(root):
            continue
        stack = [(root, iter(succ[root]))]
        color[root] = 1
        while stack:
            node, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                color[node] = 2
                stack.pop()
            elif color.get(nxt) == 1:
                back.add((node, nxt))
            elif not color.get(nxt):
                color[nxt] = 1
                stack.append((nxt, iter(succ[nxt])))
    dag = {n: [m for m in succ[n] if (n, m) not in back] for n in names}
    deg = {n: 0 for n in names}
    for n in names:
        for m in dag[n]:
            deg[m] += 1
    layer = {n: 0 for n in names}
    queue = [n for n in names if deg[n] == 0]
    while queue:
        n = queue.pop(0)
        for m in dag[n]:
            layer[m] = max(layer[m], layer[n] + 1)
            deg[m] -= 1
            if deg[m] == 0:
                queue.append(m)
    # katman içi sıra: ağırlık merkezi (barycenter) süpürmeleri — kesişen tel sayısını azaltır (Sugiyama yerleşimi)
    nlayers = max(layer.values(), default=0) + 1
    layers: list[list[str]] = [[] for _ in range(nlayers)]
    for n in names:
        layers[layer[n]].append(n)
    pred: dict[str, list[str]] = {n: [] for n in names}
    for n in names:
        for m in succ[n]:
            pred[m].append(n)
    rank = {n: float(i) for lay in layers for i, n in enumerate(lay)}

    def sweep(order_range, neighbours) -> None:
        for L in order_range:
            lay = layers[L]
            keyed = []
            for i, n in enumerate(lay):
                nb = [rank[m] for m in neighbours[n] if layer[m] != L]
                keyed.append((sum(nb) / len(nb) if nb else rank[n], i, n))
            keyed.sort()
            layers[L] = [n for _k, _i, n in keyed]
            for i, n in enumerate(layers[L]):
                rank[n] = float(i)

    for _ in range(4):
        sweep(range(1, nlayers), pred)
        sweep(range(nlayers - 2, -1, -1), succ)
    tallest = max((len(lay) for lay in layers), default=1)
    row_h, col_w = 8 * GRID, 18 * GRID
    for L, lay in enumerate(layers):
        top = (tallest - len(lay)) * row_h / 2.0         # katmanlar dikeyde ortalanır (tek uzun sütun oluşmaz)
        for r, n in enumerate(lay):
            b = model.blocks[n]
            if "pos" in b.meta:
                continue
            b.meta["pos"] = [float(L * col_w), float(round((top + r * row_h) / GRID) * GRID)]


__all__ = ["DiagramEditor", "register_template", "templates", "Template", "TEMPLATE_MODULES"]
