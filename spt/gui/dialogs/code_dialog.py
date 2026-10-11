"""Üretilen C kodunu gösteren ortak pencere: dosya sekmeleri, KULLANIM.md (biçimlendirilmiş), özet / maliyet
tabloları ve klasöre kaydetme. Sinyal üreteci (DAC) ve sensör füzyonu kod üreticileri kullanır.

Pencerenin sol tarafına isteğe bağlı bir ayar paneli (ör. MCU özellikleri) konabilir; ayar değişince sahibi
`set_files` / `set_summary` ile içeriği yeniler.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from spt.gui.settings import app_settings
from spt.gui.theme import Tokens, mono_font, style_native_titlebar
from spt.gui.widgets.common import label


class CodeDialog(QDialog):
    def __init__(self, tokens: Tokens, title: str, settings_key: str, parent: QWidget | None = None,
                 side: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        screen = QGuiApplication.primaryScreen()  # küçük ekranda alt düğmeler taşmasın
        area = screen.availableGeometry() if screen is not None else None
        self.resize(min(1180, int(area.width() * 0.92)) if area else 1180,
                    min(760, int(area.height() * 0.88)) if area else 760)
        self._tokens = tokens
        self._settings_key = settings_key
        self.files: list = []  # (name, text) alanlı nesneler

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(10)
        self.status = label("", "Readout")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.RichText)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        if side is not None:  # ayar paneli kaydırılabilir: kısa ekranda (768 px) alt düğmeler pencere dışına taşmaz
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setWidget(side)
            scroll.setMinimumWidth(max(300, side.minimumSizeHint().width() + scroll.verticalScrollBar().sizeHint().width()))
            split.addWidget(scroll)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.summary = QTextBrowser()
        self.summary.setOpenExternalLinks(False)
        self.tabs.addTab(self.summary, "Özet ve maliyet")
        split.addWidget(self.tabs)
        split.setStretchFactor(split.count() - 1, 1)
        if side is not None:
            split.setSizes([340, 840])
        lay.addWidget(split, 1)
        lay.addWidget(self.status)

        row = QHBoxLayout()
        row.addStretch(1)
        self.copy_button = QPushButton("Bu sekmeyi kopyala")
        self.copy_button.clicked.connect(self._copy)
        self.save_button = QPushButton("Klasöre Kaydet…")
        self.save_button.setObjectName("Primary")
        self.save_button.clicked.connect(self._save)
        close = QPushButton("Kapat")
        close.clicked.connect(self.close)
        for b in (self.copy_button, self.save_button, close):
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            row.addWidget(b)
        lay.addLayout(row)
        style_native_titlebar(self, tokens)

    # ---------------------------------------------------------------- içerik
    def set_summary(self, html: str) -> None:
        self.summary.setHtml(html)

    def set_files(self, files: list) -> None:
        """files: `name` ve `text` alanlı nesneler (GeneratedFile). .md dosyaları biçimlendirilmiş gösterilir."""
        current = self.tabs.tabText(self.tabs.currentIndex())
        while self.tabs.count() > 1:
            widget = self.tabs.widget(1)
            self.tabs.removeTab(1)
            widget.deleteLater()
        self.files = list(files)
        for f in self.files:
            if f.name.lower().endswith(".md"):
                view = QTextBrowser()
                view.setMarkdown(f.text)
            else:
                view = QPlainTextEdit(f.text)
                view.setReadOnly(True)
                view.setFont(mono_font(12))
                view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
            self.tabs.addTab(view, f.name)
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == current:
                self.tabs.setCurrentIndex(i)
        self.save_button.setEnabled(bool(self.files))

    def set_status(self, text: str, state: str = "") -> None:
        t = self._tokens
        color = {"good": t.good, "warning": t.warning, "critical": t.critical}.get(state)
        self.status.setText((f'<span style="color:{color}">●</span> ' if color else "") + text)

    # ---------------------------------------------------------------- eylemler
    def _copy(self) -> None:
        index = self.tabs.currentIndex()
        if index == 0:
            text = self.summary.toPlainText()
        else:
            text = self.files[index - 1].text
        QGuiApplication.clipboard().setText(text)
        self.set_status(f"“{self.tabs.tabText(index)}” panoya kopyalandı.", "good")

    def _save(self) -> None:
        if not self.files:
            return
        s = app_settings()
        start = str(s.value(f"{self._settings_key}/dir", str(Path.home())))
        folder = QFileDialog.getExistingDirectory(self, "Kodun kaydedileceği klasör", start)
        if not folder:
            return
        s.setValue(f"{self._settings_key}/dir", folder)
        self.save_to(Path(folder))

    def save_to(self, folder: Path, ask: bool = True) -> list[Path]:
        existing = [f.name for f in self.files if (folder / f.name).exists()]
        if existing and ask:
            answer = QMessageBox.question(self, "Dosyalar var",
                                          "Şu dosyalar zaten var ve üzerine yazılacak:\n" + "\n".join(existing),
                                          QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
            if answer != QMessageBox.StandardButton.Yes:
                return []
        written = []
        for f in self.files:
            path = folder / f.name
            path.write_text(f.text, encoding="utf-8", newline="\n")
            written.append(path)
        self.set_status(f"{len(written)} dosya kaydedildi: {folder}", "good")
        return written
