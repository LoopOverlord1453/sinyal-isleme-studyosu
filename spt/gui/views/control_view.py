"""Kontrol Tasarımı görünümü (Aşama 4): MATLAB Control System / System Identification / MPC araç kutuları ve
Simulink'in karşılığı.

Sekmeler (hepsi aynı model çalışma alanını paylaşır, `spt.gui.control.workspace.ModelWorkspace`):
  1. Modeller            — ifadeyle model, liste, MATLAB biçimli gösterim, kutup / sıfır, dönüşümler, .sptm
  2. Sistem Tanımlama    — `spt.gui.control.sysid_tab.SysIdTab`
  3. Analiz              — G / C / H / F çevrimi; L, T, S, KS, GS; seçilebilir grafik ızgarası; pay ve stepinfo
  4. Kontrolcü Tasarımı  — `spt.gui.control.design_tab.DesignTab`
  5. Blok Diyagram       — `spt.gui.control.diagram_editor.DiagramEditor`

2, 4 ve 5. sekmeler kendi modüllerinden tembel olarak yüklenir: modül yoksa (hazırlanıyor) ya da yüklenemezse sekmede
nedenini anlatan bir yer tutucu gösterilir. Sekme sınıfları `workspace` parametresi varsa `(theme, workspace=…)`, yoksa
`(theme)` + `set_workspace(…)` ile kurulur; varsa `set_sources`, `shutdown`, `detachable_cards`, `save_state` /
`restore_state` yöntemleri çağrılır.
"""
from __future__ import annotations

import importlib
import inspect
import json
import logging

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QPushButton, QTabWidget, QVBoxLayout, QWidget

from spt.control.errors import ControlError
from spt.gui.control.analysis_tab import AnalysisTab
from spt.gui.control.models_tab import ModelsTab
from spt.gui.control.workspace import ModelWorkspace
from spt.gui.theme import ThemeManager
from spt.gui.widgets.common import Card, PageHeader, chip, label

log = logging.getLogger("spt.gui")

# (anahtar, sekme başlığı, modül, sınıf) — None: bu modülde
TABS = (
    ("models", "Modeller", None, None),
    ("sysid", "Sistem Tanımlama", "spt.gui.control.sysid_tab", "SysIdTab"),
    ("analysis", "Analiz", None, None),
    ("design", "Kontrolcü Tasarımı", "spt.gui.control.design_tab", "DesignTab"),
    ("diagram", "Blok Diyagram", "spt.gui.control.diagram_editor", "DiagramEditor"),
)
SUBTITLES = {
    "models": "Modeller · MATLAB çalışma alanı: tf / zpk / ss, dönüşümler, kutup-sıfır",
    "sysid": "Sistem Tanımlama · veriden doğrusal / doğrusal olmayan model",
    "analysis": "Analiz · açık / kapalı çevrim cevapları, kararlılık payları",
    "design": "Kontrolcü Tasarımı · PID, kompansatör, durum geri beslemesi, MPC",
    "diagram": "Blok Diyagram · benzetim ve doğrusallaştırma",
}


class _Placeholder(QWidget):
    """Henüz yüklenemeyen sekmenin yerine: ne olduğunu ve nedenini söyler."""

    def __init__(self, title: str, reason: str):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        card = Card(title)
        text = label(reason, "Muted")
        text.setWordWrap(True)
        text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card.body.addStretch(1)
        card.body.addWidget(text)
        card.body.addStretch(1)
        lay.addWidget(card)
        self.reason = reason


class ControlView(QWidget):
    def __init__(self, theme: ThemeManager, parent: QWidget | None = None):
        super().__init__(parent)
        self._theme = theme
        self.workspace = ModelWorkspace(self)
        self._sources: list = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        self.header = PageHeader("Kontrol Tasarımı", SUBTITLES["models"])
        self.model_chip = chip("")
        self.model_chip.setToolTip("Çalışma alanındaki model sayısı (tüm sekmeler paylaşır)")
        self.header.add_right(self.model_chip)
        open_b = QPushButton("  Modelleri Aç…")
        open_b.setCursor(Qt.CursorShape.PointingHandCursor)
        open_b.setToolTip("Model çalışma alanı dosyasından (.sptm) modelleri ekle")
        save_b = QPushButton("  Modelleri Kaydet…")
        save_b.setCursor(Qt.CursorShape.PointingHandCursor)
        save_b.setToolTip("Tüm modelleri .sptm dosyasına kaydet")
        self.header.add_right(open_b)
        self.header.add_right(save_b)
        lay.addWidget(self.header)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.models_tab = ModelsTab(theme, self.workspace)
        self.analysis_tab = AnalysisTab(theme, self.workspace)
        self.pages: dict[str, QWidget] = {}
        for key, title, module, cls in TABS:
            if key == "models":
                page = self.models_tab
            elif key == "analysis":
                page = self.analysis_tab
            else:
                page = self._load_tab(title, module, cls)
            self.pages[key] = page
            self.tabs.addTab(page, title)
        lay.addWidget(self.tabs, 1)
        open_b.clicked.connect(self.models_tab.open_file)
        save_b.clicked.connect(self.models_tab.save_file)
        self.tabs.currentChanged.connect(self._on_tab)
        self.models_tab.analyze_requested.connect(self.analyze_model)
        for sig in (self.workspace.added, self.workspace.removed, self.workspace.renamed):
            sig.connect(self._update_chip)
        self._update_chip()

    # ---------------------------------------------------------------- sekmeler
    def _load_tab(self, title: str, module: str, cls_name: str) -> QWidget:
        try:
            mod = importlib.import_module(module)
        except ModuleNotFoundError as exc:
            if exc.name == module:
                return _Placeholder(title, f"{title} hazırlanıyor — bu sekme bir sonraki güncellemeyle gelecek.")
            log.exception("%s sekmesi yüklenemedi", title)
            return _Placeholder(title, f"{title} yüklenemedi: eksik bileşen ({exc.name}). Ayrıntı Günlük panelinde.")
        except Exception as exc:  # noqa: BLE001 - başka sekmenin hatası görünümü düşürmesin
            log.exception("%s sekmesi yüklenemedi", title)
            return _Placeholder(title, f"{title} yüklenemedi: {exc}. Ayrıntı Günlük panelinde.")
        cls = getattr(mod, cls_name, None)
        if cls is None:
            return _Placeholder(title, f"{title} hazırlanıyor.")
        try:
            if "workspace" in inspect.signature(cls).parameters:
                return cls(self._theme, workspace=self.workspace)
            page = cls(self._theme)
            if callable(getattr(page, "set_workspace", None)):
                page.set_workspace(self.workspace)
            return page
        except Exception as exc:  # noqa: BLE001
            log.exception("%s sekmesi kurulamadı", title)
            return _Placeholder(title, f"{title} kurulamadı: {exc}. Ayrıntı Günlük panelinde.")

    def tab_keys(self) -> list[str]:
        return [k for k, *_ in TABS]

    def show_tab(self, key: str) -> None:
        self.tabs.setCurrentIndex(self.tab_keys().index(key))

    def current_tab(self) -> str:
        return self.tab_keys()[self.tabs.currentIndex()]

    def is_placeholder(self, key: str) -> bool:
        return isinstance(self.pages[key], _Placeholder)

    def _on_tab(self, index: int) -> None:
        key = self.tab_keys()[index]
        self.header.subtitle.setText(SUBTITLES[key])
        if key == "analysis":
            self.analysis_tab.schedule()

    def _update_chip(self, *_) -> None:
        n = len(self.workspace)
        self.model_chip.setText(f"{n} model")
        self.model_chip.setVisible(n > 0)

    def analyze_model(self, name: str) -> None:
        """Modeli Analiz sekmesinde bitki olarak seç ve sekmeye geç."""
        self.analysis_tab.set_loop(G=name)
        self.show_tab("analysis")

    # ---------------------------------------------------------------- ana pencere arayüzü
    def set_sources(self, sources: list[tuple[str, str, object]]) -> None:
        """Veri kaynakları (kimlik, etiket, tutamaç): Sistem Tanımlama'ya ve Analiz'deki benzetim girişine iletilir."""
        self._sources = list(sources)
        self.analysis_tab.set_sources(self._sources)
        for key in ("sysid", "design", "diagram"):
            fn = getattr(self.pages[key], "set_sources", None)
            if callable(fn):
                try:
                    fn(self._sources)
                except Exception:  # noqa: BLE001
                    log.exception("%s sekmesine kaynaklar iletilemedi", key)

    def detachable_cards(self) -> list[tuple[str, str, Card]]:
        """Ayrı pencereye alınabilen kartlar (anahtar, başlık, kart): tüm sekmelerden."""
        out = self.models_tab.detachable_cards() + self.analysis_tab.detachable_cards()
        for key in ("sysid", "design", "diagram"):
            fn = getattr(self.pages[key], "detachable_cards", None)
            if callable(fn):
                try:
                    out += [(k if k.startswith("ctl_") else f"ctl_{key}_{k}", t, c) for k, t, c in fn()]
                except Exception:  # noqa: BLE001
                    log.exception("%s sekmesinin kartları alınamadı", key)
        return out

    def shutdown(self) -> None:
        for key in ("sysid", "design", "diagram"):
            fn = getattr(self.pages[key], "shutdown", None)
            if callable(fn):
                try:
                    fn()
                except Exception:  # noqa: BLE001
                    log.exception("%s sekmesi kapatılamadı", key)

    def save_state(self, s) -> None:
        """Ayarlar: geçerli sekme, sekme tercihleri ve çalışma alanındaki modeller (bir sonraki oturuma)."""
        s.setValue("control/tab", self.tabs.currentIndex())
        data, _skipped = self.workspace.to_dict()
        s.setValue("control/workspace", json.dumps(data, ensure_ascii=False))
        self.models_tab.save_state(s)
        self.analysis_tab.save_state(s)
        for key in ("sysid", "design", "diagram"):
            fn = getattr(self.pages[key], "save_state", None)
            if callable(fn):
                try:
                    fn(s)
                except Exception:  # noqa: BLE001
                    log.exception("%s sekmesinin ayarları kaydedilemedi", key)

    def restore_state(self, s) -> None:
        raw = s.value("control/workspace", "")
        if raw:
            try:
                _added, failed = self.workspace.load_dict(json.loads(str(raw)), replace=True)
                for f in failed:
                    log.warning("Kayıtlı model geri yüklenemedi: %s", f)
            except (ControlError, ValueError, TypeError) as exc:
                log.warning("Kayıtlı model çalışma alanı okunamadı (%s).", exc)
        self.models_tab.restore_state(s)
        self.analysis_tab.restore_state(s)
        for key in ("sysid", "design", "diagram"):
            fn = getattr(self.pages[key], "restore_state", None)
            if callable(fn):
                try:
                    fn(s)
                except Exception:  # noqa: BLE001
                    log.exception("%s sekmesinin ayarları okunamadı", key)
        try:
            index = int(s.value("control/tab", 0))
        except (TypeError, ValueError):
            index = 0
        if 0 <= index < self.tabs.count():
            self.tabs.setCurrentIndex(index)
