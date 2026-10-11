"""Model çalışma alanı: Kontrol Tasarımı sekmelerinin paylaştığı adlandırılmış modeller (MATLAB çalışma alanı).

Modeller sekmesi ifadeyle model ekler; Sistem Tanımlama sonuçları, Blok Diyagram doğrusallaştırması ve tasarlanan
kontrolcüler de buraya eklenir; Analiz ve Kontrolcü Tasarımı sekmeleri modelleri buradan seçer. Değişiklikler Qt
sinyalleriyle duyurulur (`added`, `removed`, `changed`, `renamed`).

Model türleri: LTI (tf / zpk / ss, `spt.control.lti`), PID nesneleri, tanımlanmış modeller (`spt.control.sysid`;
analizde ölçüm modeli `to_lti()` kullanılır) ve MPC nesneleri (analizde doğrusal karşılığı varsa).

`.sptm` dosyası (JSON, sürüm alanlı):
    {"format": "sptm", "version": 1, "models": [{"name", "source", "note", "expr", "kind", "class", "model"}, …]}
`model` alanı nesnenin `to_dict()` çıktısıdır; geri yükleme `class` alanındaki (yalnız `spt.control.` altındaki)
sınıfın `from_dict`'iyle, yoksa `LTI.from_dict` ile yapılır. `to_dict`'i olmayan tanımlanmış model ölçüm modeli (LTI)
olarak yazılır (gürültü modeli ve kovaryans dosyaya girmez); yazılamayan nesneler (ör. MPC) atlanır ve bildirilir.
"""
from __future__ import annotations

import importlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal

from spt.control.errors import ControlError

log = logging.getLogger("spt.gui")

SPTM_FORMAT = "sptm"
SPTM_VERSION = 1
NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,62}$")
# İfadelerde değişken / fonksiyon adı olarak kullanılan, model adı olamayacak sözcükler
RESERVED = frozenset({
    "s", "z", "q", "p", "pi", "e", "j", "i", "inf", "Inf", "nan", "NaN", "Ts", "tf", "zpk", "ss", "pid", "pidstd",
    "c2d", "d2c", "minreal", "feedback", "series", "parallel", "exp", "sqrt", "eye", "zeros", "ones", "inv",
})

# Kaynak adları (listede gösterilir)
SOURCE_EXPR = "ifade"
SOURCE_CONVERT = "dönüşüm"
SOURCE_FILE = "dosya"
SOURCE_SYSID = "sistem tanımlama"
SOURCE_DIAGRAM = "blok diyagram"
SOURCE_DESIGN = "tasarım"


@dataclass
class ModelEntry:
    name: str
    model: object
    source: str = ""
    note: str = ""      # kısa açıklama (ör. "ARX [2 2 1], fit %92")
    expr: str = ""      # ifadeyle oluşturulduysa ifade

    @property
    def kind(self) -> str:
        return model_kind(self.model)


# ============================================================================ tür ve LTI dönüşümü
def _lti_module():
    try:
        from spt.control import lti
    except ImportError:  # pragma: no cover - çekirdek henüz yok
        return None
    return lti


def is_lti(model) -> bool:
    lti = _lti_module()
    return lti is not None and isinstance(model, lti.LTI)


def model_kind(model) -> str:
    """Listede gösterilen tür: tf / zpk / ss / pid / pidstd / pid2 / idpoly / idss / idtf / idproc / mpc / …"""
    name = type(model).__name__.lower()
    if "pid" in name:  # PID nesneleri transfer fonksiyonundan türer; türleri adlarıyla gösterilir (MATLAB gibi)
        return name
    kind = getattr(type(model), "_kind", None)
    if isinstance(kind, str) and kind != "lti":
        return kind
    if name.startswith("id"):
        return name
    if "mpc" in name:
        return "mpc"
    return name


def as_lti(model):
    """Analizde kullanılacak LTI karşılığı; yoksa None.

    LTI (PID nesneleri dahil, LTI'den türüyorsa) olduğu gibi; tanımlanmış model `to_lti()` (ölçüm modeli); başka
    nesnelerde `tf(nesne)` denenir (ör. LTI'den türemeyen PID)."""
    if model is None:
        return None
    if is_lti(model):
        return model
    to_lti = getattr(model, "to_lti", None)
    if callable(to_lti):
        try:
            out = to_lti()
        except ControlError:
            return None
        return out if is_lti(out) else None
    lti = _lti_module()
    if lti is not None and "pid" in type(model).__name__.lower():
        try:
            return lti.tf(model)
        except (ControlError, TypeError, ValueError, AttributeError):
            return None
    return None


def check_name(name: str) -> str:
    """Geçerli model adı ya da ControlError (Türkçe neden)."""
    name = (name or "").strip()
    if not name:
        raise ControlError("Model adı boş olamaz.")
    if not NAME_RE.match(name):
        raise ControlError(f"“{name}” geçerli bir ad değil: harfle başlamalı; yalnız İngilizce harf, rakam ve _ "
                           "içermeli (ifadelerde değişken olarak kullanılır).")
    if name in RESERVED:
        raise ControlError(f"“{name}” ifadelerde ayrılmış bir sözcük (değişken / fonksiyon adı); başka ad seçin.")
    return name


# ============================================================================ çalışma alanı
class ModelWorkspace(QObject):
    """Adlandırılmış modeller; ekleme sırası korunur."""

    added = pyqtSignal(str)
    removed = pyqtSignal(str)
    changed = pyqtSignal(str)
    renamed = pyqtSignal(str, str)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._entries: dict[str, ModelEntry] = {}

    # ---------------------------------------------------------------- sorgular
    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def names(self) -> list[str]:
        return list(self._entries)

    def entries(self) -> list[ModelEntry]:
        return list(self._entries.values())

    def entry(self, name: str) -> ModelEntry:
        try:
            return self._entries[name]
        except KeyError:
            raise ControlError(f"Çalışma alanında “{name}” adlı model yok.") from None

    def get(self, name: str):
        return self.entry(name).model

    def lti(self, name: str):
        """Modelin LTI karşılığı (analiz için); yoksa ControlError."""
        entry = self.entry(name)
        out = as_lti(entry.model)
        if out is None:
            raise ControlError(f"“{name}” ({entry.kind}) doğrusal model olarak analiz edilemiyor.")
        return out

    def lti_names(self) -> list[str]:
        """Analizde seçilebilen (LTI karşılığı olan) modeller."""
        return [n for n, e in self._entries.items() if as_lti(e.model) is not None]

    def variables(self) -> dict[str, object]:
        """İfade ayrıştırıcısına verilecek değişkenler: ad → LTI karşılığı (karşılığı olmayanlar hariç)."""
        out = {}
        for name, entry in self._entries.items():
            sys = as_lti(entry.model)
            if sys is not None:
                out[name] = sys
        return out

    def unique_name(self, base: str) -> str:
        """`base` boşsa olduğu gibi, değilse base2, base3 … (geçersiz karakterler _ olur)."""
        base = re.sub(r"[^A-Za-z0-9_]", "_", (base or "").strip()) or "G"
        if not base[0].isalpha():
            base = "G" + base
        base = base[:56]
        if base in RESERVED:
            base += "1"
        if base not in self._entries:
            return base
        k = 2
        while f"{base}{k}" in self._entries:
            k += 1
        return f"{base}{k}"

    # ---------------------------------------------------------------- değişiklik
    def add(self, name: str, model, source: str = "", note: str = "", expr: str = "",
            replace: bool = False) -> str:
        """Modeli ekler ve son adını döndürür. Ad doluysa `replace=True` ile değiştirilir (changed), yoksa benzersiz
        bir ad seçilir (G2, G3 …)."""
        if model is None:
            raise ControlError("Boş model eklenemez.")
        name = check_name(name) if name else self.unique_name("G")
        if name in self._entries and not replace:
            name = self.unique_name(name)
        model = _named(model, name)
        existed = name in self._entries
        self._entries[name] = ModelEntry(name, model, source, note, expr)
        (self.changed if existed else self.added).emit(name)
        return name

    def set_model(self, name: str, model, note: str | None = None) -> None:
        entry = self.entry(name)
        entry.model = model
        if note is not None:
            entry.note = note
        self.changed.emit(name)

    def remove(self, name: str) -> None:
        if name in self._entries:
            del self._entries[name]
            self.removed.emit(name)

    def rename(self, old: str, new: str) -> str:
        entry = self.entry(old)
        new = check_name(new)
        if new == old:
            return new
        if new in self._entries:
            raise ControlError(f"“{new}” adlı bir model zaten var.")
        items = list(self._entries.items())
        self._entries = {(new if k == old else k): v for k, v in items}
        entry.name = new
        entry.model = _named(entry.model, new)
        self.renamed.emit(old, new)
        return new

    def clear(self) -> None:
        for name in list(self._entries):
            self.remove(name)

    # ---------------------------------------------------------------- dosya
    def to_dict(self, names: list[str] | None = None) -> tuple[dict, list[str]]:
        """(.sptm sözlüğü, yazılamayan modellerin adları)."""
        models, skipped = [], []
        for name in names if names is not None else self.names():
            entry = self.entry(name)
            item = _serialize(entry)
            if item is None:
                skipped.append(name)
            else:
                models.append(item)
        return {"format": SPTM_FORMAT, "version": SPTM_VERSION, "models": models}, skipped

    def load_dict(self, data: dict, replace: bool = False) -> tuple[list[str], list[str]]:
        """Sözlükteki modelleri ekler: (eklenen adlar, okunamayanlar "ad: neden"). `replace=False` iken aynı adlı
        model varsa yeni model benzersiz adla eklenir."""
        if not isinstance(data, dict) or data.get("format") != SPTM_FORMAT:
            raise ControlError("Bu dosya bir model çalışma alanı (.sptm) değil.")
        version = data.get("version", 1)
        if not isinstance(version, int) or version > SPTM_VERSION:
            raise ControlError(f"Dosya daha yeni bir sürümle kaydedilmiş (sürüm {version}); uygulamayı güncelleyin.")
        added, failed = [], []
        for item in data.get("models", []):
            name = str(item.get("name", "")) if isinstance(item, dict) else ""
            try:
                model = _deserialize(item)
                final = self.add(name or self.unique_name("G"), model, item.get("source", SOURCE_FILE),
                                 item.get("note", ""), item.get("expr", ""), replace=replace)
            except (ControlError, KeyError, TypeError, ValueError) as exc:
                failed.append(f"{name or '?'}: {exc}")
                continue
            added.append(final)
        return added, failed

    def save(self, path: str | Path, names: list[str] | None = None) -> list[str]:
        """`.sptm` dosyasına yazar; yazılamayan modellerin adlarını döndürür."""
        data, skipped = self.to_dict(names)
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        return skipped

    def load(self, path: str | Path, replace: bool = False) -> tuple[list[str], list[str]]:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError) as exc:
            raise ControlError(f"Dosya okunamadı: {exc}") from None
        except json.JSONDecodeError as exc:
            raise ControlError(f"Dosya bozuk (JSON hatası, satır {exc.lineno}).") from None
        return self.load_dict(data, replace)


def _named(model, name: str):
    """Gösterimdeki "ad =" satırı için modelin adı. Düz tf / zpk / ss kopyalanır (çağıranın nesnesi değişmez);
    türetilmiş sınıflar (PID …) kopyada türünü kaybedeceğinden yerinde adlandırılır."""
    if not is_lti(model) or getattr(model, "name", name) == name:
        return model
    lti = _lti_module()
    if type(model) in (lti.TransferFunction, lti.ZPK, lti.StateSpace):
        model = model.copy()
    try:
        model.name = name
    except (AttributeError, ControlError, TypeError):
        pass
    return model


def _class_path(obj) -> str:
    cls = type(obj)
    return f"{cls.__module__}:{cls.__qualname__}"


def _serialize(entry: ModelEntry) -> dict | None:
    model = entry.model
    item = {"name": entry.name, "source": entry.source, "note": entry.note, "expr": entry.expr,
            "kind": entry.kind}
    to_dict = getattr(model, "to_dict", None)
    if callable(to_dict):
        try:
            item["model"] = _jsonable(to_dict())
            item["class"] = _class_path(model)
            return item
        except (ControlError, TypeError, ValueError, NotImplementedError) as exc:
            log.warning("“%s” dosyaya yazılamadı: %s", entry.name, exc)
            return None
    sys = as_lti(model)
    if sys is not None:  # tanımlanmış model: ölçüm modeli LTI olarak
        item["model"] = _jsonable(sys.to_dict())
        item["class"] = _class_path(sys)
        note = "ölçüm modeli (LTI) olarak kaydedildi"
        item["note"] = f"{entry.note} · {note}" if entry.note else note
        return item
    return None


def _jsonable(d):
    """numpy sayı / dizi kalıntılarını JSON türlerine çevirir; JSON'a yazılamıyorsa TypeError."""
    import numpy as np

    if isinstance(d, dict):
        return {str(k): _jsonable(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_jsonable(v) for v in d]
    if isinstance(d, np.ndarray):
        return _jsonable(d.tolist())
    if isinstance(d, np.generic):
        return d.item()
    if d is None or isinstance(d, (bool, int, float, str)):
        return d
    raise TypeError(f"JSON'a yazılamayan değer: {type(d).__name__}")


def _deserialize(item: dict):
    if not isinstance(item, dict) or "model" not in item:
        raise ControlError("model kaydı eksik")
    data = item["model"]
    path = item.get("class", "")
    if isinstance(path, str) and ":" in path:
        module, _, qual = path.partition(":")
        if module.startswith("spt.control.") and "." not in qual:
            try:
                cls = getattr(importlib.import_module(module), qual)
            except (ImportError, AttributeError):
                cls = None
            from_dict = getattr(cls, "from_dict", None) if cls is not None else None
            if callable(from_dict):
                return from_dict(data)
    lti = _lti_module()
    if lti is None:
        raise ControlError("LTI modelleri kullanılamıyor")
    return lti.LTI.from_dict(data)
