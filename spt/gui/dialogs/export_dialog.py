"""Katsayıları dışa aktar: CSV veya C başlık dosyası (float / Q15 / Q31, CMSIS-DSP veya düz C).

Pencere açıkken tasarımcıdaki değişiklikler önizlemeye anında yansır. Seçilen sayı biçimi tasarımcının genlik
grafiğine kuantize cevap olarak çizilir; böylece yuvarlamanın frekans cevabına etkisi aynı anda görülür.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from spt.dsp import coefficients as C
from spt.dsp.filters import Design, FilterError
from spt.gui.settings import app_settings
from spt.gui.theme import Tokens, mono_font, style_native_titlebar
from spt.gui.widgets.common import add_form_row, combo, form_layout, label

FILE_KINDS = [("C başlık dosyası (.h)", "c"), ("CSV", "csv")]
LAYOUTS = [("CMSIS-DSP (a ters işaretli)", "cmsis"), ("Düz C (standart işaret)", "plain")]


def _combo(items):
    w = combo([])
    for text, key in items:
        w.addItem(text, key)
    return w


class ExportDialog(QDialog):
    format_changed = pyqtSignal(object)  # sayı biçimi (tasarımcıda kuantize cevabı çizmek için)
    closed = pyqtSignal()                # pencere kapandı (tasarımcıdaki kuantize eğri kaldırılır)

    def __init__(self, tokens: Tokens, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Katsayıları Dışa Aktar")
        self.resize(760, 640)
        self._tokens = tokens
        self.design: Design | None = None
        self.text = ""
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 18, 20, 16)
        lay.setSpacing(10)

        self.kind = _combo(FILE_KINDS)
        self.number = _combo([(text, key) for key, text in C.NUMBER_FORMATS.items()])
        self.number.setCurrentIndex(self.number.findData("float32"))
        self.layout_combo = _combo(LAYOUTS)
        self.structure = _combo([(text, key) for key, text in C.IIR_STRUCTURES.items()])
        self.scale = QCheckBox("L∞ bölüm ölçekleme")
        self.scale.setToolTip("Her bölümün çıkışında |H| ≤ 1 olacak şekilde pay katsayılarını ölçekler (toplam kazanç "
                              "değişmez). Sabit noktada ara düğümlerde taşmayı önler; Q15/Q31'de varsayılan açıktır.")
        self.name = QLineEdit("flt")
        self.name.setToolTip("C dizilerinin ve makroların ön eki")

        form = form_layout()
        self._form = form
        add_form_row(form, "Dosya", self.kind)
        add_form_row(form, "Sayı biçimi", self.number)
        add_form_row(form, "Düzen", self.layout_combo)
        add_form_row(form, "IIR yapısı", self.structure)
        add_form_row(form, "Ölçekleme", self.scale)
        add_form_row(form, "Ad", self.name)
        lay.addLayout(form)

        self.report = label("", "Readout")
        self.report.setWordWrap(True)
        self.report.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(self.report)

        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setFont(mono_font(12))
        self.preview.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        lay.addWidget(self.preview, 1)

        row = QHBoxLayout()
        copy = QPushButton("Panoya Kopyala")
        copy.clicked.connect(self._copy)
        self.save_button = QPushButton("Kaydet…")
        self.save_button.setObjectName("Primary")
        self.save_button.clicked.connect(self._save)
        close = QPushButton("Kapat")
        close.clicked.connect(self.close)
        for b in (copy, self.save_button, close):
            b.setCursor(Qt.CursorShape.PointingHandCursor)
        row.addStretch(1)
        row.addWidget(copy)
        row.addWidget(self.save_button)
        row.addWidget(close)
        lay.addLayout(row)

        self._scale_touched = False
        self.scale.clicked.connect(lambda: setattr(self, "_scale_touched", True))
        for w in (self.kind, self.number, self.layout_combo, self.structure):
            w.currentIndexChanged.connect(self._render)
        self.number.currentIndexChanged.connect(lambda *_: self.format_changed.emit(self.number.currentData()))
        self.scale.toggled.connect(self._render)
        self.name.textChanged.connect(self._render)
        style_native_titlebar(self, tokens)

    def set_design(self, d: Design | None) -> None:
        self.design = d
        if self.isVisible():  # kapalıyken her tasarım değişikliğinde C / CSV üretip kuantize etme
            self._render()
        else:
            self._dirty = True

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if getattr(self, "_dirty", True):
            self._dirty = False
            self._render()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self.closed.emit()

    def _render(self, *_) -> None:
        d = self.design
        fixed = self.number.currentData() in C.FIXED_BITS
        if not self._scale_touched:
            self.scale.blockSignals(True)
            self.scale.setChecked(fixed)
            self.scale.blockSignals(False)
        iir = d is not None and d.kind == "iir"
        c_file = self.kind.currentData() == "c"
        cmsis = self.layout_combo.currentData() == "cmsis"
        self._form.setRowVisible(self.layout_combo, c_file)
        self._form.setRowVisible(self.structure, c_file and cmsis and iir and self.number.currentData() == "float32")
        self._form.setRowVisible(self.scale, iir)
        self._form.setRowVisible(self.name, c_file)
        if d is None or not d.is_linear:
            self.text = ""
            self.preview.setPlainText("")
            self.report.setText("Dışa aktarılacak doğrusal bir tasarım yok (tasarım hatası ya da medyan filtre).")
            self.save_button.setEnabled(False)
            return
        fmt = self.number.currentData()
        scale = self.scale.isChecked() if iir else None
        try:
            if c_file:
                self.text = C.to_c(d, fmt, cmsis=cmsis, structure=self.structure.currentData(),
                                   name=self.name.text() or "flt", scale=scale)
            else:
                self.text = C.to_csv(d, fmt, scale=scale)
            rep = C.quantization_report(d, C.quantize(d, fmt, scale))
        except FilterError as exc:
            self.text = ""
            self.preview.setPlainText("")
            self.report.setText(str(exc))
            self.save_button.setEnabled(False)
            return
        t = self._tokens
        lines = [f"<b>{d.describe()}</b> · {C.NUMBER_FORMATS[fmt]}",
                 f"Kuantizasyon: geçirme bandı sapması {rep.pass_dev_db:.4g} dB · en büyük sapma {rep.dev_db:.4g} dB"]
        if rep.stop_peak_db is not None:
            lines[-1] += f" · durdurma tepesi {rep.stop_peak_db:.1f} dB (ideal {rep.ideal_stop_peak_db:.1f} dB)"
        if d.kind == "iir":
            color, text = (t.good, "kararlı") if rep.stable else (t.critical, "KARARSIZ")
            lines.append(f'<span style="color:{color}">●</span> Kuantize filtre {text} (en büyük |p| = '
                         f"{rep.max_pole_radius:.6f})")
        if rep.dev_db > 1.0 or not rep.stable:
            lines.append(f'<span style="color:{t.warning}">⚠</span> Bu sayı biçimi cevabı belirgin biçimde bozuyor; '
                         "daha yüksek çözünürlük (Q31 / float32) ya da daha düşük derece deneyin.")
        self.report.setText("<br>".join(lines))
        self.preview.setPlainText(self.text)
        self.save_button.setEnabled(True)

    def _copy(self) -> None:
        if self.text:
            QGuiApplication.clipboard().setText(self.text)

    def _save(self) -> None:
        if not self.text:
            return
        s = app_settings()
        start_dir = Path(str(s.value("export/dir", str(Path.home()))))
        c_file = self.kind.currentData() == "c"
        default = start_dir / f"{self.name.text() or 'flt'}{'.h' if c_file else '.csv'}"
        filters = "C başlık dosyası (*.h);;Tüm dosyalar (*)" if c_file else "CSV (*.csv);;Tüm dosyalar (*)"
        path, _ = QFileDialog.getSaveFileName(self, "Katsayıları kaydet", str(default), filters)
        if not path:
            return
        Path(path).write_text(self.text, encoding="utf-8")
        s.setValue("export/dir", str(Path(path).parent))
        self.report.setText(self.report.text() + f"<br>Kaydedildi: {path}")
