"""Blok parametre paneli (Simulink "Block Parameters" penceresinin karşılığı) ve Veri kaynağı bağlama.

Form, bloğun parametre tanımlarından (`Block.PARAMS`: tür, Türkçe etiket, birim, ipucu, grup, seçenekler) kendiliğinden
kurulur. Sayısal alanlar MATLAB yazımıyla girilir (`[1 2 1]`, `[1 2; 3 4]`, `2*pi`, `inf`); değerler en kısa tam
gösterimle yazılır (ör. 0.1 → "0.1"), böylece değiştirilmeyen alan yuvarlanmadan aynen kalır — "Uygula" yalnız
metni değişen alanları bloğa gönderir. Her alan düzenlenince tek başına doğrulanır; "Uygula"da blok tüm değerleri
birlikte doğrular (parametreler arası tutarlılık) ve hata olursa hiçbiri değişmez. İletiler Türkçedir.

Veri kaynağı bloğu için "Veriye bağla" bölümü: canlı akış / kayıt / Çalışma Alanı anlık görüntüsü (tamponun kayıpsız
en yeni kısmı; zaman ilk örnekten başlar) ya da CSV / TXT dosyası (zaman sütunu ya da Ts) seçilen kanallarla bloğun
zaman ve değer parametrelerine yazılır.
"""
from __future__ import annotations

import csv
import io
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from spt.control.errors import ControlError
from spt.gui.widgets.common import add_form_row, chip, combo, form_layout, label

MAX_BIND_SAMPLES = 200_000      # bağlanan verinin en fazla örnek sayısı (en yeni kısım)
MAX_EDIT_ELEMENTS = 400         # bundan büyük diziler formda salt okunur özetle gösterilir


# ============================================================================ değer ↔ metin
def fmt_number(v: float) -> str:
    """En kısa tam (gidiş-dönüşte aynı float) gösterim: 1 → "1", 0.1 → "0.1", inf → "inf"."""
    v = float(v)
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    if v.is_integer() and abs(v) < 1e15:
        return str(int(v))
    return repr(v)


def _fmt_complex(z: complex) -> str:
    re_, im_ = float(z.real), float(z.imag)
    if im_ == 0.0:
        return fmt_number(re_)
    sign = "+" if im_ >= 0 else "-"
    return f"{fmt_number(re_)}{sign}{fmt_number(abs(im_))}i"


def format_value(kind: str, value: Any) -> str:
    """Parametre değerinin düzenlenebilir metni (Param.validate bu metni aynı değere çözer)."""
    if kind in ("str", "expr", "signs", "choice"):
        return str(value)
    if kind == "bool":
        return "açık" if value else "kapalı"
    if kind == "int":
        return str(int(value))
    if kind == "float":
        return fmt_number(value)
    if kind == "sample_time":
        if value.offset:
            return f"[{fmt_number(value.period)} {fmt_number(value.offset)}]"
        return fmt_number(value.period)
    if kind == "cvector":
        arr = np.asarray(value, dtype=np.complex128).ravel()
        return "[" + " ".join(_fmt_complex(z) for z in arr) + "]"
    arr = np.asarray(value, dtype=np.float64)
    if kind == "vector" or arr.ndim <= 1:
        flat = arr.ravel()
        if flat.size == 1:
            return fmt_number(flat[0])
        return "[" + " ".join(fmt_number(x) for x in flat) + "]"
    if arr.size == 1:
        return fmt_number(arr.ravel()[0])
    return "[" + "; ".join(" ".join(fmt_number(x) for x in row) for row in arr) + "]"


# ============================================================================ veri bağlama (Qt'siz)
def data_from_handle(handle, channels: list[int], max_samples: int = MAX_BIND_SAMPLES) \
        -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Akış / kayıt / anlık görüntü tutamacından (zaman, değerler, kanal adları).

    Tamponun kayıpsız en yeni kısmı alınır (en fazla `max_samples` örnek); zaman ilk alınan örnekten başlar
    (t = k / fs). Değerler fiziksel birimdedir (ölçek / ofset uygulanmış)."""
    from spt.dsp.convert import physical_column

    desc = handle.desc
    chans = list(desc.channels)
    if not channels:
        raise ControlError("En az bir kanal seçilmeli.")
    for c in channels:
        if not 0 <= c < len(chans):
            raise ControlError(f"Kanal {c + 1} bu kaynakta yok.")
    if desc.sample_type.is_complex:
        raise ControlError("Karmaşık (I/Q) örnekli akış Veri kaynağı bloğuna bağlanamaz.")
    ring = handle.ring
    hi = int(handle.latest_index())
    lo = max(int(ring.valid_from()), hi - int(max_samples))
    gaps = ring.gaps_between(lo, hi) if hasattr(ring, "gaps_between") else []
    if gaps:
        lo = max(lo, max(int(ge) for _, ge in gaps))       # son kaybın sonrasındaki kayıpsız kısım
    if hi - lo < 2:
        raise ControlError("Kaynakta bağlanacak kadar (en az 2 örnek) kayıpsız veri yok.")
    first, raw = ring.read(lo, hi)
    raw = np.asarray(raw)
    if raw.ndim == 1:
        raw = raw[:, None]
    vals = np.column_stack([np.asarray(physical_column(raw[:, c], desc.sample_type, chans[c]), dtype=np.float64)
                            for c in channels])
    if not np.isfinite(vals).all():
        raise ControlError("Seçilen kanallarda sonlu olmayan değer var.")
    t = np.arange(len(vals), dtype=np.float64) / float(handle.fs)
    names = [chans[c].name or f"Kanal {c + 1}" for c in channels]
    return t, vals, names


def read_table(path: str | Path) -> tuple[list[str], np.ndarray]:
    """CSV / TXT tablosu: (sütun adları, sayısal matris). Ayırıcı kendiliğinden bulunur; ilk satır sayı değilse başlık
    sayılır. Boş / 'nan' hücreler NaN olur."""
    try:
        text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    except OSError as exc:
        raise ControlError(f"Dosya okunamadı: {exc}") from None
    sample = text[:20000]
    try:
        delim = csv.Sniffer().sniff(sample, delimiters=",;\t ").delimiter
    except csv.Error:
        delim = "," if "," in sample else (";" if ";" in sample else (("\t" if "\t" in sample else " ")))
    rows = list(csv.reader(io.StringIO(text), delimiter=delim, skipinitialspace=True))
    if delim == " ":
        rows = [[c for c in r if c != ""] for r in rows]
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        raise ControlError("Dosya boş.")

    def num(c: str) -> float:
        c = c.strip()
        if c == "" or c.lower() in ("nan", "na", "null"):
            return math.nan
        return float(c)

    header: list[str] | None = None
    try:
        [num(c) for c in rows[0]]
    except ValueError:
        header = [c.strip() for c in rows[0]]
        rows = rows[1:]
    if not rows:
        raise ControlError("Dosyada sayısal satır yok.")
    width = max(len(r) for r in rows)
    try:
        M = np.array([[num(c) for c in r] + [math.nan] * (width - len(r)) for r in rows], dtype=np.float64)
    except ValueError as exc:
        raise ControlError(f"Sayıya çevrilemeyen hücre: {exc}") from None
    names = [(header[i] if header and i < len(header) and header[i] else f"Sütun {i + 1}") for i in range(width)]
    return names, M


def table_binding(names: list[str], M: np.ndarray, columns: list[int], time_column: int | None,
                  Ts: float | None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Tablodan (zaman, değerler, adlar). Zaman sütunu yoksa t = k·Ts (k = 0, 1, …)."""
    if not columns:
        raise ControlError("En az bir değer sütunu seçilmeli.")
    if time_column is not None and time_column in columns:
        raise ControlError("Zaman sütunu değer olarak da seçilemez.")
    vals = M[:, columns]
    bad = ~np.isfinite(vals).all(axis=1)
    if bad.any():
        row = int(np.argmax(bad)) + 1
        raise ControlError(f"Seçilen sütunlarda boş ya da sayı olmayan hücre var (veri satırı {row}).")
    if time_column is not None:
        t = M[:, time_column]
        if not np.isfinite(t).all():
            raise ControlError("Zaman sütununda boş hücre var.")
        if np.any(np.diff(t) < 0):
            raise ControlError("Zaman sütunu azalmayan sırada olmalı.")
    else:
        if Ts is None or not (Ts > 0) or not math.isfinite(Ts):
            raise ControlError("Zaman sütunu yoksa örnekleme aralığı Ts > 0 verilmeli.")
        t = np.arange(len(vals), dtype=np.float64) * float(Ts)
    return np.asarray(t, dtype=np.float64), np.asarray(vals, dtype=np.float64), [names[c] for c in columns]


# ============================================================================ panel
_CSV = "__csv__"


class ParamPanel(QWidget):
    """Seçili bloğun parametre formu.

    `set_target(block, apply_fn)`: `apply_fn(yeni_ad | None, {ad: değer}) ` bloğa uygular (geri al yığını üzerinden);
    `ControlError` yükseltirse ileti panelde gösterilir. `status(metin, durum)` sinyali düzenleyicinin durum
    satırına gider."""

    status = pyqtSignal(str, str)

    def __init__(self, theme, parent: QWidget | None = None):
        super().__init__(parent)
        self._theme = theme
        self._block = None
        self._apply_fn: Callable[[str | None, dict], None] | None = None
        self._fields: dict[str, tuple[Any, QWidget, str]] = {}   # ad → (Param, alan, ilk metin)
        self._sources: list[tuple[str, str, object]] = []
        self._table: tuple[list[str], np.ndarray, str] | None = None   # CSV: (adlar, matris, dosya)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(self.scroll, 1)
        self.error = label("", "Readout")
        self.error.setWordWrap(True)
        self.error.setStyleSheet(f"color: {theme.tokens.critical};")
        self.error.hide()
        outer.addWidget(self.error)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.apply_button = QPushButton("Uygula")
        self.apply_button.setObjectName("Primary")
        self.apply_button.setToolTip("Değişen parametreleri bloğa uygula (Enter); geri alınabilir.")
        self.apply_button.clicked.connect(self.apply)
        self.revert_button = QPushButton("Sıfırla")
        self.revert_button.setToolTip("Formu bloğun şu anki değerlerine döndür")
        self.revert_button.clicked.connect(self.refresh)
        row.addWidget(self.apply_button)
        row.addWidget(self.revert_button)
        row.addStretch(1)
        outer.addLayout(row)
        self.clear()

    # ---------------------------------------------------------------- hedef
    @property
    def block(self):
        return self._block

    def clear(self) -> None:
        self._block = None
        self._apply_fn = None
        self._fields.clear()
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)
        empty = label("Parametrelerini düzenlemek için tuvalde bir blok seçin.", "EmptyState")
        empty.setWordWrap(True)
        lay.addWidget(empty)
        lay.addStretch(1)
        self.scroll.setWidget(w)
        self.apply_button.setEnabled(False)
        self.revert_button.setEnabled(False)
        self._show_error("")

    def set_target(self, block, apply_fn: Callable[[str | None, dict], None]) -> None:
        self._block = block
        self._apply_fn = apply_fn
        self.refresh()

    def set_sources(self, sources: list[tuple[str, str, object]]) -> None:
        """Veri kaynağı bağlama seçenekleri: (kimlik, etiket, tutamaç)."""
        self._sources = list(sources)
        if hasattr(self, "_src_combo") and self._block is not None and self._block.type_name == "DataSource":
            self._fill_sources()

    def refresh(self) -> None:
        """Formu bloğun şu anki değerlerinden yeniden kurar."""
        b = self._block
        if b is None:
            self.clear()
            return
        self._fields.clear()
        self._table = None
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(2, 2, 6, 2)
        lay.setSpacing(6)
        title = label(b.display_name, "CardTitle")
        lay.addWidget(title)
        if b.description:
            d = label(b.description, "Muted")
            d.setWordWrap(True)
            lay.addWidget(d)
        if b.simulink:
            lay.addWidget(chip(f"Simulink: {b.simulink}"), 0, Qt.AlignmentFlag.AlignLeft)
        form = form_layout()
        self.name_edit = QLineEdit(b.name)
        self.name_edit.setToolTip("Blok adı (diyagramda tek olmalı; Skop sonuç kartı bu adla görünür)")
        self.name_edit.returnPressed.connect(self.apply)
        add_form_row(form, "Blok adı", self.name_edit)
        lay.addLayout(form)

        groups: dict[str, list] = {}
        for p in b.PARAMS:
            groups.setdefault(p.group or "", []).append(p)
        for gname, params in groups.items():
            if len(groups) > 1 or gname:
                lay.addWidget(label((gname or "Genel").upper(), "SectionLabel"))
            form = form_layout()
            form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)    # dar panelde alanlar tam genişlik
            for p in params:
                self._add_field(form, p, b.params[p.name])
            lay.addLayout(form)
        if b.type_name == "DataSource":
            self._build_binding(lay)
        lay.addStretch(1)
        self.scroll.setWidget(w)
        self.apply_button.setEnabled(True)
        self.revert_button.setEnabled(True)
        self._show_error("")

    def _add_field(self, form: QFormLayout, p, value) -> None:
        text_label = p.label + (f" ({p.unit})" if p.unit else "")
        tip = p.hint or ""
        if p.kind == "bool":
            w = QCheckBox()
            w.setChecked(bool(value))
            init = "1" if value else "0"
            w.toggled.connect(lambda *_: self._show_error(""))
        elif p.kind == "choice":
            keys = list(p.choices or {})
            w = combo([p.choices[k] for k in keys], keys.index(value) if value in keys else 0)
            for i, k in enumerate(keys):
                w.setItemData(i, k)
            init = str(value)
        elif p.kind in ("vector", "matrix", "cvector") and np.asarray(value).size > MAX_EDIT_ELEMENTS:
            shape = " × ".join(str(s) for s in np.shape(value))
            w = QLineEdit(f"({shape} değer)")
            w.setReadOnly(True)                    # büyük veri metin olarak düzenlenmez (bağlama / dosya ile gelir)
            tip = (tip + "\n" if tip else "") + "Büyük dizi: metin olarak düzenlenemez."
            init = w.text()
        else:
            w = QLineEdit(format_value(p.kind, value))
            init = w.text()
            w.editingFinished.connect(lambda name=p.name: self._check_field(name))
            w.returnPressed.connect(self.apply)
        if tip:
            w.setToolTip(tip)
        add_form_row(form, text_label, w)
        self._fields[p.name] = (p, w, init)

    # ---------------------------------------------------------------- değerler
    def field_text(self, name: str) -> str:
        p, w, _ = self._fields[name]
        if isinstance(w, QCheckBox):
            return "1" if w.isChecked() else "0"
        if isinstance(w, QComboBox):
            return str(w.currentData())
        return w.text()

    def set_field(self, name: str, text: Any) -> None:
        """Alanı metinle / değerle doldurur (test ve betik için)."""
        if name == "__name__":
            self.name_edit.setText(str(text))
            return
        p, w, _ = self._fields[name]
        if isinstance(w, QCheckBox):
            w.setChecked(bool(text) if not isinstance(text, str) else text.strip().lower() in ("1", "true", "açık"))
        elif isinstance(w, QComboBox):
            i = w.findData(text)
            w.setCurrentIndex(i if i >= 0 else w.findText(str(text)))
        else:
            w.setText(text if isinstance(text, str) else format_value(p.kind, text))

    def _value(self, name: str) -> Any:
        p, w, _ = self._fields[name]
        if isinstance(w, QCheckBox):
            return w.isChecked()
        if isinstance(w, QComboBox):
            return w.currentData()
        return w.text()

    def changes(self) -> dict[str, Any]:
        """Metni ilk hâlinden farklı olan alanlar (değiştirilmeyen alan aynen kalır)."""
        out = {}
        for name, (p, w, init) in self._fields.items():
            if self.field_text(name) != init:
                out[name] = self._value(name)
        return out

    def _check_field(self, name: str) -> None:
        if name not in self._fields or self._block is None:
            return
        p, w, init = self._fields[name]
        try:
            p.validate(self._value(name), self._block.name)
        except ControlError as exc:
            self._mark(w, True)
            self._show_error(str(exc))
            return
        self._mark(w, False)
        self._show_error("")

    def _mark(self, w: QWidget, bad: bool) -> None:
        w.setStyleSheet(f"border-color: {self._theme.tokens.critical};" if bad else "")

    def _show_error(self, text: str) -> None:
        self.error.setText(text)
        self.error.setVisible(bool(text))

    @property
    def error_text(self) -> str:
        return self.error.text() if self.error.isVisible() else ""

    def apply(self) -> bool:
        """Değişiklikleri bloğa uygular; başarılıysa True. Hata Türkçe olarak panelde gösterilir."""
        if self._block is None or self._apply_fn is None:
            return False
        changes = self.changes()
        new_name = self.name_edit.text().strip()
        if new_name == self._block.name:
            new_name = None
        if not changes and new_name is None:
            self._show_error("")
            return True
        for name in changes:                       # alan bazlı ön doğrulama: hangi alanın hatalı olduğu görünsün
            p, w, _ = self._fields[name]
            try:
                p.validate(changes[name], self._block.name)
                self._mark(w, False)
            except ControlError as exc:
                self._mark(w, True)
                self._show_error(str(exc))
                return False
        try:
            self._apply_fn(new_name, changes)
        except ControlError as exc:
            self._show_error(str(exc))
            return False
        self.refresh()
        return True

    # ---------------------------------------------------------------- Veri kaynağı bağlama
    def _build_binding(self, lay: QVBoxLayout) -> None:
        lay.addWidget(label("VERİYE BAĞLA", "SectionLabel"))
        info = (self._block.meta or {}).get("binding")
        if info:
            cur = label(f"Bağlı: {info.get('label', '')} · {', '.join(info.get('channels', []))}", "Readout")
            cur.setWordWrap(True)
            lay.addWidget(cur)
        form = form_layout()
        self._src_combo = QComboBox()
        self._src_combo.currentIndexChanged.connect(self._source_changed)
        add_form_row(form, "Kaynak", self._src_combo)
        self._time_combo = QComboBox()
        self._time_combo.setToolTip("CSV: zaman sütunu (yoksa Ts ile eşit aralık)")
        add_form_row(form, "Zaman sütunu", self._time_combo)
        self._ts_edit = QLineEdit("")
        self._ts_edit.setPlaceholderText("örn. 0.01")
        self._ts_edit.setToolTip("Zaman sütunu yoksa örnekleme aralığı (s)")
        add_form_row(form, "Ts (s)", self._ts_edit)
        lay.addLayout(form)
        self._chan_list = QListWidget()
        self._chan_list.setMaximumHeight(120)
        self._chan_list.setToolTip("Bloğun çıkış kanalları (sırayla)")
        lay.addWidget(self._chan_list)
        self._bind_button = QPushButton("Bağla")
        self._bind_button.setToolTip("Seçilen kanalları bloğun zaman / değer parametrelerine yaz (geri alınabilir)")
        self._bind_button.clicked.connect(self.bind)
        lay.addWidget(self._bind_button, 0, Qt.AlignmentFlag.AlignLeft)
        self._fill_sources()

    def _fill_sources(self) -> None:
        cb = self._src_combo
        cb.blockSignals(True)
        cb.clear()
        for sid, text, _h in self._sources:
            cb.addItem(text, sid)
        cb.addItem("CSV / TXT dosyası…", _CSV)
        cb.setCurrentIndex(0)
        cb.blockSignals(False)
        self._source_changed()

    def _handle(self):
        sid = self._src_combo.currentData()
        for s, _t, h in self._sources:
            if s == sid:
                return h
        return None

    def _source_changed(self, *_args) -> None:
        self._chan_list.clear()
        self._time_combo.clear()
        is_csv = self._src_combo.currentData() == _CSV
        self._time_combo.setEnabled(is_csv)
        self._ts_edit.setEnabled(is_csv)
        if is_csv:
            if self._table is None:
                return
            names = self._table[0]
            self._time_combo.addItem("Yok (Ts ile)", None)
            for i, n in enumerate(names):
                self._time_combo.addItem(n, i)
            guess = next((i for i, n in enumerate(names) if n.strip().lower() in ("t", "time", "zaman", "t (s)",
                                                                                  "time (s)", "zaman (s)")), None)
            if guess is not None:
                self._time_combo.setCurrentIndex(guess + 1)
            for i, n in enumerate(names):
                self._add_chan(n, i, i != guess)
            return
        h = self._handle()
        if h is None:
            return
        for i, ch in enumerate(h.desc.channels):
            self._add_chan(ch.name or f"Kanal {i + 1}", i, i == 0)

    def _add_chan(self, text: str, index: int, checked: bool) -> None:
        item = QListWidgetItem(text)
        item.setData(Qt.ItemDataRole.UserRole, index)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self._chan_list.addItem(item)

    def checked_channels(self) -> list[int]:
        return [self._chan_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self._chan_list.count())
                if self._chan_list.item(i).checkState() == Qt.CheckState.Checked]

    def load_table(self, path: str | Path) -> None:
        """CSV / TXT dosyasını bağlama için yükler (kaynak kutusu "CSV" olur)."""
        names, M = read_table(path)
        self._table = (names, M, str(path))
        i = self._src_combo.findData(_CSV)
        self._src_combo.blockSignals(True)
        self._src_combo.setCurrentIndex(i)
        self._src_combo.blockSignals(False)
        self._source_changed()

    def bind(self, *_args) -> bool:
        """Seçili kaynağı bloğa bağlar (zaman ve değer parametreleri). Başarılıysa True."""
        if self._block is None or self._apply_fn is None:
            return False
        try:
            if self._src_combo.currentData() == _CSV:
                if self._table is None:
                    path, _ = QFileDialog.getOpenFileName(self, "Veri dosyası aç", "",
                                                          "Tablo (*.csv *.txt *.tsv *.dat);;Tüm dosyalar (*)")
                    if not path:
                        return False
                    self.load_table(path)
                    return False      # kullanıcı sütunları seçsin, sonra yeniden "Bağla"
                names, M, path = self._table
                ts_text = self._ts_edit.text().strip()
                Ts = float(ts_text) if ts_text else None
                t, vals, chosen = table_binding(names, M, self.checked_channels(), self._time_combo.currentData(), Ts)
                info = {"kind": "csv", "label": Path(path).name, "channels": chosen}
            else:
                h = self._handle()
                if h is None:
                    raise ControlError("Bağlanacak kaynak yok (akış, kayıt ya da Çalışma Alanı anlık görüntüsü).")
                t, vals, chosen = data_from_handle(h, self.checked_channels())
                info = {"kind": "stream", "label": self._src_combo.currentText(), "channels": chosen}
        except ValueError as exc:            # ControlError da ValueError'dır
            self._show_error(str(exc) if isinstance(exc, ControlError) else f"Ts sayı olmalı: {exc}")
            return False
        try:
            self._apply_fn(None, {"time": t, "values": vals, "__meta_binding__": info})
        except ControlError as exc:
            self._show_error(str(exc))
            return False
        self.status.emit(f"Veri kaynağı bağlandı: {info['label']} · {len(t)} örnek · {', '.join(chosen)}", "good")
        self.refresh()
        return True
