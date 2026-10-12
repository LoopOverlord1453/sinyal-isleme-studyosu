"""Sistem Tanımlama sekmesinin Qt'siz iş katmanı: veri tablosu, ön işleme zinciri, tahmin / değerlendirme işi.

Hesapların tamamı `spt.control.sysid` paketinden gelir (bu modül yalnız arayüz ayarlarını API çağrılarına çevirir;
tahmin matematiği yazmaz). İş, arayüz donmasın ve iptal edilebilsin diye ayrı süreçte çalışır (`child_main`):
Qt içe aktarmaz, sonuçları seçilebilir (picklable) sözlüklerle döndürür.

Akış (MATLAB System Identification uygulamasının sırası):
  kaynak tablosu → kanal / Ts / zaman aralığı → eksikleri doldur → trend çıkar → süz → yeniden örnekle →
  tahmin / doğrulama bölmesi → tahmin → compare (benzetim, k = ∞, başlangıç 'e'), resid (doğrulama verisi),
  frekans cevabı (modeller + çalışma verisinden spa).
"""
from __future__ import annotations

import csv
import inspect
import io
import re
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from spt.control.errors import ControlError

# ------------------------------------------------------------------------------------------- yöntemler
# anahtar → (arayüz adı, model adı öneki)
METHODS: dict[str, tuple[str, str]] = {
    "arx": ("ARX", "arx"),
    "armax": ("ARMAX", "amx"),
    "oe": ("OE (çıkış hatası)", "oe"),
    "bj": ("Box–Jenkins (BJ)", "bj"),
    "tfest": ("Transfer fonksiyonu (tfest)", "tf"),
    "procest": ("Süreç modeli (procest)", "P"),
    "n4sid": ("Durum uzayı — altuzay (n4sid)", "n4s"),
    "ssest": ("Durum uzayı — PEM (ssest)", "ss"),
    "impulseest": ("Darbe cevabı / FIR (impulseest)", "imp"),
    "spa": ("Spektral analiz (spa)", "spa"),
    "etfe": ("Ampirik transfer fonksiyonu (etfe)", "etfe"),
    "nlarx": ("Doğrusal olmayan ARX (nlarx)", "nlarx"),
    "nlhw": ("Hammerstein–Wiener (nlhw)", "nlhw"),
}
NL_NAMES = {"wavenet": "dalgacık ağı", "sigmoidnet": "sigmoid ağı", "linear": "doğrusal",
            "pwlinear": "parçalı doğrusal", "saturation": "doyum", "deadzone": "ölü bölge", "poly1d": "polinom",
            "unitgain": "yok"}
NONPARAMETRIC = ("spa", "etfe")
NONLINEAR = ("nlarx", "nlhw")

FILTER_KINDS = {"low": "Alçak geçiren", "high": "Yüksek geçiren", "bandpass": "Bant geçiren",
                "bandstop": "Bant söndüren"}
TIME_NAMES = re.compile(r"^(t|time|zaman|sure|süre|timestamp|t_?s|t\s*\[s\]|time\s*\[s\]|time_s|zaman_s)$", re.I)
MIN_SAMPLES = 8


def sysid_api(name: str):
    """`spt.control.sysid` fonksiyonu; henüz yoksa None (birinci dalga modülleri tamamlandıkça dolar)."""
    try:
        import spt.control.sysid as pkg
    except ImportError:  # pragma: no cover
        return None
    f = getattr(pkg, name, None)
    if f is not None:
        return f
    import importlib
    import pkgutil
    for mod in pkgutil.iter_modules(pkg.__path__):
        try:
            m = importlib.import_module(f"{pkg.__name__}.{mod.name}")
        except Exception:  # noqa: BLE001 - yazılmakta olan modül içe aktarılamayabilir
            continue
        f = getattr(m, name, None)
        if callable(f):
            return f
    return None


def method_available(key: str) -> bool:
    return sysid_api(key) is not None


def _call(func, *args, **kw):
    """Fonksiyonun imzasında olmayan anahtar sözcükleri atar (seçenek adları modül sürümüyle değişebilir)."""
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):  # pragma: no cover
        return func(*args, **kw)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return func(*args, **kw)
    return func(*args, **{k: v for k, v in kw.items() if k in params and v is not None})


# ------------------------------------------------------------------------------------------- kaynak tablosu
@dataclass
class SourceTable:
    """Seçilebilir kanalları olan ham veri: (N, m) fiziksel değerler (NaN = eksik örnek)."""
    key: str
    label: str
    names: list[str]
    units: list[str]
    X: np.ndarray
    Ts: float | None          # akıştan / zaman sütunundan / dosyadaki Ts'ten (None: elle verilmeli)
    tstart: float = 0.0
    time_column: int | None = None
    kind: str = "file"        # "stream" | "snapshot" | "file"
    note: str = ""            # kullanıcıya bilgi (ör. zaman sütunu eşit aralıklı değil)

    @property
    def N(self) -> int:
        return int(self.X.shape[0])

    def channels(self) -> list[int]:
        """Seçilebilir kanal indisleri (zaman sütunu hariç)."""
        return [i for i in range(len(self.names)) if i != self.time_column]

    def channel_text(self, i: int) -> str:
        u = self.units[i] if i < len(self.units) else ""
        return f"{self.names[i]} [{u}]" if u else self.names[i]


def table_from_handle(key: str, label: str, handle) -> SourceTable:
    """Canlı akış / kayıt / anlık görüntü tutamacının tampondaki tüm verisi (kayıp bölgeler NaN)."""
    from spt.control.sysid import iddata_from_handle

    n = len(handle.desc.channels)
    d = iddata_from_handle(handle, outputs=list(range(n)), name=label)
    kind = "snapshot" if hasattr(handle, "snapshot") else "stream"
    return SourceTable(key, label, list(d.output_names), list(d.output_units), np.asarray(d.y, float), d.Ts,
                       float(d.tstart), None, kind)


def _csv_layout(path) -> tuple[int, str, str]:
    """(sütun sayısı, sütun ayırıcı, ondalık ayırıcı) — seçim kutularını doldurmak için hafif okuma.

    Noktalı virgül / sekme her satırda varsa ayırıcıdır (Türkçe Excel CSV'si: ';' ve ondalık virgül); yoksa virgül,
    o da yoksa boşluk. Ayırıcı virgül değilse ve hücrelerde ondalık virgül varsa ondalık ayırıcı ','."""
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    lines = [ln for ln in text[:20000].splitlines() if ln.strip()]
    if len(lines) > 1:
        lines = lines[:-1]  # son satır örnek sınırında kesilmiş olabilir
    if not lines:
        raise ControlError("Dosya boş")
    delim = next((d for d in (";", "\t", ",") if all(d in ln for ln in lines)), " ")
    rows = list(csv.reader(io.StringIO("\n".join(lines)), delimiter=delim, skipinitialspace=True))
    if delim == " ":
        rows = [[c for c in r if c != ""] for r in rows]
    width = max(len(r) for r in rows)
    body = rows[1:] if len(rows) > 1 else rows
    decimal = "," if delim != "," and any(re.fullmatch(r"\s*[-+]?\d+,\d*([eE][-+]?\d+)?\s*", c)
                                          for r in body for c in r) else "."
    return width, delim, decimal


def table_from_csv(path) -> SourceTable:
    """CSV / TXT dosyası: tüm sütunlar okunur; adı zaman olan (t, time, zaman …) eşit aralıklı sütun Ts'i verir."""
    from spt.control.sysid import read_csv

    width, delim, decimal = _csv_layout(path)
    d = read_csv(path, outputs=list(range(width)), Ts=1.0, delimiter=delim, decimal=decimal)
    names = list(d.output_names)
    X = np.asarray(d.y, float)
    tcol = next((i for i, n in enumerate(names) if TIME_NAMES.match(n.strip())), None)
    Ts, tstart, note = None, 0.0, ""
    if tcol is not None:
        t = X[:, tcol]
        dt = np.diff(t)
        if len(dt) and np.all(np.isfinite(t)) and np.all(dt > 0):
            ts = float(np.median(dt))
            if np.max(np.abs(dt - ts)) <= 1e-6 * ts + 1e-12 * np.max(np.abs(t)):
                Ts, tstart = ts, float(t[0])
            else:
                note = f"'{names[tcol]}' sütunu eşit aralıklı değil; Ts'i elle girin."
        else:
            note = f"'{names[tcol]}' sütunu artan değil; Ts'i elle girin."
    return SourceTable(f"file:{Path(path).resolve()}", f"Dosya · {Path(path).name}", names, [""] * len(names), X,
                       Ts, tstart if Ts else 0.0, tcol, "file", note)


def table_from_mat(path) -> SourceTable:
    """`.mat` (v5 / v7): aynı uzunluktaki sayısal vektör / matris değişkenleri sütun olur; `Ts` değişkeni ya da
    zaman adlı (t, time, zaman …) eşit aralıklı vektör örnekleme aralığını verir."""
    from spt.control.sysid import mat_variables, read_mat

    shapes = mat_variables(path)
    if not shapes:
        raise ControlError("Dosyada sayısal değişken yok")

    def length(s):
        return s[0] if (len(s) == 2 and s[0] > 1) else (s[1] if len(s) == 2 and s[0] == 1 else 0)

    lens = [length(s) for s in shapes.values() if length(s) > 1]
    if not lens:
        raise ControlError("Dosyada vektör ya da matris değişkeni yok")
    N = max(set(lens), key=lens.count)
    names, cols = [], []
    for nm, s in shapes.items():
        if length(s) != N:
            continue
        v = np.asarray(read_mat(path, y=nm, Ts=1.0).y, float)
        if v.shape[0] != N:
            continue
        for j in range(v.shape[1]):
            names.append(nm if v.shape[1] == 1 else f"{nm}({j + 1})")
            cols.append(v[:, j])
    X = np.column_stack(cols)
    Ts, tstart, note = None, 0.0, ""
    if "Ts" in shapes and int(np.prod(shapes["Ts"])) == 1:
        Ts = float(np.asarray(read_mat(path, y="Ts", Ts=1.0).y).ravel()[0])
        Ts = Ts if np.isfinite(Ts) and Ts > 0 else None
    tcol = next((i for i, n in enumerate(names) if TIME_NAMES.match(n)), None)
    if tcol is not None:
        t = X[:, tcol]
        dt = np.diff(t)
        ts = float(np.median(dt)) if len(dt) else 0.0
        if ts > 0 and np.all(dt > 0) and np.max(np.abs(dt - ts)) <= 1e-6 * ts + 1e-12 * np.max(np.abs(t)):
            if Ts is not None and abs(Ts - ts) > 1e-6 * Ts:
                note = f"Dosyadaki Ts ({Ts:g} s) '{names[tcol]}' ile ({ts:g} s) uyuşmuyor; zaman değişkeni kullanıldı."
            Ts, tstart = ts, float(t[0])
        else:
            note = f"'{names[tcol]}' eşit aralıklı ve artan değil; Ts'i elle girin."
    return SourceTable(f"file:{Path(path).resolve()}", f"Dosya · {Path(path).name}", names, [""] * len(names), X,
                       Ts, tstart if Ts else 0.0, tcol, "file", note)


def table_from_file(path) -> SourceTable:
    if str(path).lower().endswith(".mat"):
        return table_from_mat(path)
    return table_from_csv(path)


# ------------------------------------------------------------------------------------------- ön işleme
@dataclass
class Prep:
    """Ön işleme ayarları (MATLAB System Identification uygulamasının "Preprocess" menüsü sırasıyla)."""
    t0: float | None = None             # zaman aralığı (s); None: verinin başı / sonu
    t1: float | None = None
    missing: str = "error"              # "error" | "linear" | "model"
    detrend: int | None = 0             # None: yok, 0: ortalama (sabit), 1: doğrusal
    filter_kind: str | None = None      # None | "low" | "high" | "bandpass" | "bandstop"
    filter_order: int = 4
    filter_f1: float = 0.0              # Hz (alçak / yüksek: kesim; bant: alt kenar)
    filter_f2: float = 0.0              # Hz (bant: üst kenar)
    zero_phase: bool = False
    P: int = 1                          # yeniden örnekleme: yeni hız = hız · P / Q
    Q: int = 1
    fraction: float = 0.5               # tahmin verisi oranı; 1.0: doğrulama = tahmin verisi


def sample_range(table: SourceTable, Ts: float, t0, t1) -> tuple[int, int]:
    """Zaman aralığının [k0, k1) örnek indisleri (uçlar dahil: t0 ≤ tstart + k·Ts ≤ t1)."""
    N = table.N
    k0 = 0 if t0 is None else int(np.ceil((t0 - table.tstart) / Ts - 1e-9))
    k1 = N if t1 is None else int(np.floor((t1 - table.tstart) / Ts + 1e-9)) + 1
    return max(0, min(k0, N)), max(0, min(k1, N))


def build_data(table: SourceTable, inputs: list[int], outputs: list[int], Ts: float, prep: Prep):
    """(çalışma verisi, tahmin verisi, doğrulama verisi) — hepsi IDData. Geçersiz seçimde Türkçe ControlError."""
    from spt.control import sysid as si

    if not outputs:
        raise ControlError("En az bir çıkış kanalı seçin")
    if not inputs:
        raise ControlError("En az bir giriş kanalı seçin (zaman serisi modelleri bu sekmede desteklenmiyor)")
    if set(inputs) & set(outputs):
        raise ControlError("Bir kanal hem giriş hem çıkış olamaz")
    if Ts is None or not np.isfinite(Ts) or Ts <= 0:
        raise ControlError("Örnekleme aralığı Ts pozitif bir sayı olmalı")
    k0, k1 = sample_range(table, Ts, prep.t0, prep.t1)
    if k1 - k0 < MIN_SAMPLES:
        raise ControlError(f"Seçilen zaman aralığında {max(k1 - k0, 0)} örnek var; en az {MIN_SAMPLES} gerekli")
    X = table.X[k0:k1]
    data = si.iddata(X[:, outputs], X[:, inputs], Ts, tstart=table.tstart + k0 * Ts,
                     output_names=[table.names[i] for i in outputs], input_names=[table.names[i] for i in inputs],
                     output_units=[table.units[i] for i in outputs], input_units=[table.units[i] for i in inputs],
                     name=table.label)
    if data.has_missing:
        if prep.missing == "error":
            from spt.control.sysid.data import check_complete
            check_complete(data, "Seçilen aralığın")
        data = si.fill_missing(data, prep.missing)
    if prep.detrend is not None:
        data = si.detrend(data, int(prep.detrend))
    if prep.filter_kind:
        nyq = 0.5 / Ts
        if prep.filter_kind in ("bandpass", "bandstop"):
            if not 0 < prep.filter_f1 < prep.filter_f2:
                raise ControlError("Bant süzgecinde alt kenar frekansı üst kenardan küçük ve pozitif olmalı")
            band = [prep.filter_f1 / nyq, prep.filter_f2 / nyq]
        else:
            band = prep.filter_f1 / nyq
        if np.any(np.asarray(band) <= 0) or np.any(np.asarray(band) >= 1):
            raise ControlError(f"Süzgeç frekansları 0 ile Nyquist frekansı ({nyq:g} Hz) arasında olmalı")
        data = si.idfilt(data, order=int(prep.filter_order), band=band, kind=prep.filter_kind,
                         causal=not prep.zero_phase)
    if (prep.P, prep.Q) != (1, 1):
        data = si.resample(data, int(prep.P), int(prep.Q))
    if prep.fraction >= 1.0:
        est, val = data, data
    else:
        est, val = si.split(data, prep.fraction)
    for part, what in ((est, "tahmin"), (val, "doğrulama")):
        if min(part.lengths) < MIN_SAMPLES:
            raise ControlError(f"Bölme sonrası {what} verisinde {min(part.lengths)} örnek kalıyor; en az {MIN_SAMPLES} "
                               "gerekli (aralığı genişletin ya da bölme oranını değiştirin)")
    const = si.excitation_warnings(data) if hasattr(si, "excitation_warnings") else []
    if const:
        import warnings
        from spt.control.errors import ControlWarning
        for msg in const:
            warnings.warn(msg, ControlWarning, stacklevel=2)
    return data, est, val


# ------------------------------------------------------------------------------------------- tahmin
def _vec(v, n: int, what: str) -> list[int]:
    a = [int(x) for x in np.atleast_1d(v)]
    if len(a) == 1:
        a = a * n
    if len(a) != n:
        raise ControlError(f"{what} için {n} değer (giriş başına) ya da tek değer verin")
    if any(x < 0 for x in a):
        raise ControlError(f"{what} negatif olamaz")
    return a


def model_name(method: str, opts: dict, nu: int) -> str:
    """MATLAB System Identification uygulamasındaki gibi kısa ad: arx221, oe221, tf21, P1D, n4s3 …"""
    p = METHODS[method][1]
    j = lambda v: "".join(str(x) for x in np.atleast_1d(v))  # noqa: E731
    if method == "arx":
        return f"{p}{opts['na']}{j(opts['nb'])}{j(opts['nk'])}"
    if method == "armax":
        return f"{p}{opts['na']}{j(opts['nb'])}{opts['nc']}{j(opts['nk'])}"
    if method == "oe":
        return f"{p}{j(opts['nb'])}{j(opts['nf'])}{j(opts['nk'])}"
    if method == "bj":
        return f"{p}{j(opts['nb'])}{opts['nc']}{opts['nd']}{j(opts['nf'])}{j(opts['nk'])}"
    if method == "tfest":
        return f"{p}{j(opts['np'])}{j(opts['nz'])}" + ("d" if opts.get("discrete") else "")
    if method == "procest":
        return str(opts["type"]).upper()
    if method in ("n4sid", "ssest"):
        return f"{p}{opts['nx']}"
    if method == "impulseest":
        return f"{p}{opts['n']}" if opts.get("n") else p
    if method in NONLINEAR:
        return f"{p}{opts['na']}{j(opts['nb'])}{j(opts['nk'])}" if method == "nlarx" else \
            f"{p}{j(opts['nb'])}{j(opts['nf'])}{j(opts['nk'])}"
    return p


def orders_text(method: str, opts: dict) -> str:
    keys = {"arx": ("na", "nb", "nk"), "armax": ("na", "nb", "nc", "nk"), "oe": ("nb", "nf", "nk"),
            "bj": ("nb", "nc", "nd", "nf", "nk"), "tfest": ("np", "nz"), "n4sid": ("nx",), "ssest": ("nx",),
            "impulseest": ("n", "nk"), "nlarx": ("na", "nb", "nk"), "nlhw": ("nb", "nf", "nk")}.get(method, ())
    parts = []
    for k in keys:
        v = opts.get(k)
        parts.append(f"{k}={' '.join(str(x) for x in np.atleast_1d(v))}")
    if method == "procest":
        parts.append(str(opts["type"]).upper())
    if method == "tfest":
        parts.append("ayrık" if opts.get("discrete") else "sürekli")
        if opts.get("estimate_delay"):
            parts.append("gecikme tahmin")
        elif opts.get("delay"):
            parts.append(f"gecikme={opts['delay']:g} s")
    if method == "impulseest" and opts.get("n") == 0:
        parts = ["n=otomatik"]
    if method in ("spa", "etfe") and opts.get("M"):
        parts.append(f"M={opts['M']}")
    if method == "ssest":
        parts.append("ayrık" if opts.get("discrete") else "sürekli")
    if method == "nlarx":
        parts.append(NL_NAMES.get(opts.get("nonlinearity", ""), ""))
    if method == "nlhw":
        parts.append(f"{NL_NAMES.get(opts.get('input_nl', ''), '')} / {NL_NAMES.get(opts.get('output_nl', ''), '')}")
    return "  ".join(p for p in parts if p)


SINGLE_OUTPUT = ("armax", "oe", "bj", "tfest", "procest", "impulseest", "nlhw")


def plan(method: str, opts: dict, nu: int, ny: int, Ts: float = 0.0) -> tuple[tuple, dict]:
    """Arayüz ayarlarını API çağrısının (veriden sonraki) argümanlarına çevirir ve doğrular — veri gerekmez, bu
    yüzden arayüz geçersiz seçimi iş başlamadan Türkçe iletiyle bildirir. Döner (args, kwargs)."""
    if method not in METHODS:
        raise ControlError("Bilinmeyen yöntem")
    if ny != 1 and method in SINGLE_OUTPUT:
        raise ControlError(f"{METHODS[method][0]} tek çıkışlı veri ister; çok çıkış için ARX, n4sid ya da ssest "
                           "seçin veya tek çıkış kanalı seçin")
    if method == "arx":
        return ([int(opts["na"])] + _vec(opts["nb"], nu, "nb") + _vec(opts["nk"], nu, "nk"),), {}
    if method == "armax":
        return ([int(opts["na"])] + _vec(opts["nb"], nu, "nb") + [int(opts["nc"])]
                + _vec(opts["nk"], nu, "nk"),), {}
    if method == "oe":
        return (_vec(opts["nb"], nu, "nb") + _vec(opts["nf"], nu, "nf") + _vec(opts["nk"], nu, "nk"),), {}
    if method == "bj":
        return (_vec(opts["nb"], nu, "nb") + [int(opts["nc"]), int(opts["nd"])] + _vec(opts["nf"], nu, "nf")
                + _vec(opts["nk"], nu, "nk"),), {}
    if method == "tfest":
        np_ = _vec(opts["np"], nu, "Kutup sayısı")
        nz = _vec(opts["nz"], nu, "Sıfır sayısı")
        discrete = bool(opts.get("discrete"))
        if not discrete and any(z > p for z, p in zip(nz, np_)):
            raise ControlError("Sürekli zamanda sıfır sayısı kutup sayısını aşamaz")
        if discrete and opts.get("estimate_delay"):
            raise ControlError("Ayrık zamanlı transfer fonksiyonunda gecikme tahmin edilemez: gecikmeyi tamsayı örnek "
                               "olarak girin ya da sürekli zamanlı modeli seçin")
        delay = np.nan if opts.get("estimate_delay") else float(opts.get("delay") or 0.0)
        return ((np_ if nu > 1 else np_[0]), (nz if nu > 1 else nz[0]), delay), {"Ts": Ts if discrete else 0.0}
    if method == "procest":
        from spt.control.sysid.models import parse_proc_type
        parse_proc_type(opts["type"])  # geçersiz tür → Türkçe ileti
        return ((opts["type"] if nu == 1 else [opts["type"]] * nu),), {}
    if method in ("n4sid", "ssest"):
        nx = int(opts["nx"])
        if nx < 1:
            raise ControlError("Durum sayısı en az 1 olmalı")
        if method == "ssest":  # MATLAB gibi varsayılan sürekli zaman; işaretliyse verinin Ts'iyle ayrık
            return (nx,), {"Ts": Ts if opts.get("discrete") else 0.0}
        return (nx,), {}
    if method == "impulseest":
        n = int(opts["n"])
        return ((n if n > 0 else None), int(opts.get("nk", 0))), {}
    if method in NONPARAMETRIC:
        return (), {"M": opts.get("M") or None, "w": opts.get("w")}
    if method == "nlarx":
        orders = [int(opts["na"])] + _vec(opts["nb"], nu, "nb") + _vec(opts["nk"], nu, "nk")
        kind = opts.get("nonlinearity") or "wavenet"
        if kind not in ("wavenet", "sigmoidnet", "linear"):
            raise ControlError("Doğrusal olmayan kısım dalgacık ağı, sigmoid ağı ya da doğrusal olmalı")
        units = int(opts.get("units") or 0)
        if units > 0 and kind != "linear":  # birim sayısı verildiyse nesneyle (0: MATLAB varsayılanı)
            cls = sysid_api("SigmoidNet" if kind == "sigmoidnet" else "WaveletNet")
            if cls is not None:
                kind = cls(units)
        return (orders, kind), {}
    if method == "nlhw":
        orders = _vec(opts["nb"], nu, "nb") + _vec(opts["nf"], nu, "nf") + _vec(opts["nk"], nu, "nk")
        nls = ("pwlinear", "saturation", "deadzone", "poly1d", "unitgain")
        inl, onl = opts.get("input_nl") or "pwlinear", opts.get("output_nl") or "pwlinear"
        if inl not in nls or onl not in nls:
            raise ControlError("Geçersiz doğrusalsızlık türü")
        return (orders, inl, onl), {}
    raise ControlError("Bilinmeyen yöntem")  # pragma: no cover


def estimate(method: str, est, opts: dict):
    """Ayarları sysid API çağrısına çevirir ve modeli döndürür (hesap tamamen API'de)."""
    if method not in METHODS:
        raise ControlError("Bilinmeyen yöntem")
    func = sysid_api(method)
    if func is None:
        raise ControlError(f"{METHODS[method][0]} yöntemi bu sürümde henüz kullanılamıyor")
    args, kw = plan(method, opts, est.nu, est.ny, est.Ts)
    return _call(func, est, *args, **kw)


def freq_grid(data) -> np.ndarray:
    """Frekans karşılaştırma ızgarası (rad/s): verinin çözünürlüğünden (2π/(N·Ts)) Nyquist'e (π/Ts) 400 log
    nokta (bizim seçimimiz; MATLAB'in varsayılan ızgarası yayımlanmamış)."""
    N = sum(data.lengths)
    w_hi = np.pi / data.Ts
    w_lo = min(2 * np.pi / (N * data.Ts), w_hi / 1000)
    return np.logspace(np.log10(w_lo), np.log10(w_hi), 400)


@dataclass
class Evaluation:
    """Bir modelin doğrulama sonuçları (tamamı sysid API çıktısı)."""
    fit_est: np.ndarray | None = None       # compare(tahmin verisi) uyum %, çıkış başına
    fit_val: np.ndarray | None = None       # compare(doğrulama verisi) uyum %, çıkış başına
    fpe: float | None = None                # Report.Fit.FPE (tahmin ölçütü, MATLAB tanımı)
    aic: float | None = None                # Report.Fit.AIC
    yhat_val: np.ndarray | None = None      # compare çıktısı (N_val, ny)
    resid: object | None = None             # ResidResult (doğrulama verisi)
    w: np.ndarray | None = None             # frekans cevabı ızgarası (rad/s)
    H: np.ndarray | None = None             # (ny, nu, nw) karmaşık
    errors: list[str] = field(default_factory=list)


def is_frd(model) -> bool:
    return hasattr(model, "response") and hasattr(model, "frequency")


def evaluate(model, est, val, w: np.ndarray, max_lag: int = 25) -> Evaluation:
    """compare (tahmin + doğrulama), resid (doğrulama), frekans cevabı — her biri ayrı denenir; hata metni saklanır."""
    from spt.control import sysid as si

    ev = Evaluation()
    if is_frd(model):
        ev.w, ev.H = np.asarray(model.frequency, float), np.asarray(model.response, complex)
        return ev
    if getattr(model, "ny", None) != val.ny or getattr(model, "nu", None) != val.nu:
        ev.errors.append(f"Model {model.ny} çıkış / {model.nu} giriş; geçerli veri {val.ny} çıkış / {val.nu} giriş")
        return ev
    if getattr(model, "Ts", 0.0) > 0 and abs(model.Ts - val.Ts) > 1e-9 * val.Ts:
        ev.errors.append(f"Modelin Ts'i ({model.Ts:g} s) geçerli veriyle ({val.Ts:g} s) aynı değil")
        return ev
    try:
        r = si.compare(est, model)
        ev.fit_est = np.asarray(r.fit[0], float)
        r = si.compare(val, model)
        ev.fit_val = np.asarray(r.fit[0], float)
        ev.yhat_val = np.asarray(r.yhat[0].y, float)
    except ControlError as exc:
        ev.errors.append(f"Karşılaştırma: {exc}")
    rep = getattr(model, "report", None)
    fit = getattr(rep, "fit", None)
    if fit is not None:
        ev.fpe, ev.aic = float(fit.FPE), float(fit.AIC)
    # uyarılar durum sınıflandırmasına girer (yeşil "iyi" gösterilmez): yetersiz uyarım, kısa veri, negatif uyum
    for msg in list(getattr(rep, "warnings", []) or []):
        if msg not in ev.errors:
            ev.errors.append(msg)
    if ev.fit_val is not None and np.any(np.asarray(ev.fit_val, float) < 0):
        ev.errors.append(f"Doğrulama uyumu negatif (%{float(np.min(ev.fit_val)):.1f}): model, çıkışın ortalamasından "
                         "daha kötü öngörüyor")
    try:
        lag = int(min(max_lag, max(sum(val.lengths) // 4, 1)))
        ev.resid = si.resid(val, model, max_lag=lag)
    except ControlError as exc:
        ev.errors.append(f"Artık analizi: {exc}")
    if hasattr(model, "freqresp"):
        try:
            ev.w = np.asarray(w, float)
            ev.H = np.asarray(model.freqresp(w), complex)
        except (ControlError, NotImplementedError, AttributeError) as exc:
            ev.w = None
            ev.errors.append(f"Frekans cevabı: {exc}")
    return ev


def run_job(job: dict, progress=None) -> dict:
    """Tahmin + değerlendirme işi.

    job: `est`, `val`, `work` (IDData), `new` (None ya da {"method", "opts", "name"}), `models` [(ad, model)],
    `max_lag`, `spa` (bool: çalışma verisinden spa referansı). Döner: {"new": (ad, model) | None, "evals":
    {ad: Evaluation}, "spa": (w, H) | None, "spa_error": str, "w": ızgara}."""
    say = progress or (lambda _t: None)
    est, val, work = job["est"], job["val"], job["work"]
    w = freq_grid(work)
    out = {"new": None, "evals": {}, "spa": None, "spa_error": "", "w": w}
    models = list(job.get("models", []))
    new = job.get("new")
    if new is not None:
        say(f"{new['name']} tahmin ediliyor ({METHODS[new['method']][0]})…")
        opts = dict(new["opts"])
        if new["method"] in NONPARAMETRIC:
            opts["w"] = w
        model = estimate(new["method"], est, opts)
        try:
            model.name = new["name"]
        except AttributeError:  # pragma: no cover
            pass
        out["new"] = (new["name"], model)
        models.append((new["name"], model))
    for name, model in models:
        say(f"{name} doğrulanıyor…")
        out["evals"][name] = evaluate(model, est, val, w, job.get("max_lag", 25))
    if job.get("spa", True):
        spa = sysid_api("spa")
        if spa is None:
            out["spa_error"] = "Spektral analiz (spa) bu sürümde henüz kullanılamıyor"
        else:
            say("Spektral analiz (spa)…")
            try:
                g = _call(spa, work, w=w)
                out["spa"] = (np.asarray(g.frequency, float), np.asarray(g.response, complex))
            except (ControlError, np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
                out["spa_error"] = error_text(exc)  # spa hatası işi düşürmez; tahmin edilen model korunur
    return out


def suggest_arx_orders(est, val, na_max: int = 10, nb_max: int = 10, nk_range=(1, 5), criterion: str = "aic"):
    """ARX derece önerisi (MATLAB `selstruc(arxstruc(ze, zv, struc(1:na, 1:nb, nk)), crit)`): tek giriş / tek
    çıkış. Döner (na, nb, nk)."""
    if est.ny != 1 or est.nu != 1:
        raise ControlError("Derece önerisi tek giriş / tek çıkışlı veride yapılır")
    arxstruc, selstruc, struc = sysid_api("arxstruc"), sysid_api("selstruc"), sysid_api("struc")
    if arxstruc is None or selstruc is None:
        raise ControlError("Derece önerisi (arxstruc / selstruc) bu sürümde henüz kullanılamıyor")
    if struc is not None:
        NN = struc(range(1, na_max + 1), range(1, nb_max + 1), range(nk_range[0], nk_range[1] + 1))
    else:
        NN = np.array([[a, b, k] for a in range(1, na_max + 1) for b in range(1, nb_max + 1)
                       for k in range(nk_range[0], nk_range[1] + 1)])
    V = arxstruc(est, val, NN)
    nn = selstruc(V, criterion)
    nn = np.asarray(nn[0] if isinstance(nn, tuple) else nn).ravel().astype(int)
    return int(nn[0]), int(nn[1]), int(nn[2])


# ------------------------------------------------------------------------------------------- süreç girişi
def call_job(func, args, progress):
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        params = {}
    return func(*args, progress=progress) if "progress" in params else func(*args)


def error_text(exc: BaseException) -> str:
    """Kullanıcıya gösterilecek Türkçe ileti (iç ayrıntı günlüğe gider)."""
    if isinstance(exc, ControlError):
        return str(exc)
    if isinstance(exc, np.linalg.LinAlgError):
        return ("Sayısal hata (tekil ya da kötü koşullu problem). Dereceyi düşürmeyi, veriyi ön işlemeyi ya da "
                "daha uzun bir aralık seçmeyi deneyin.")
    if isinstance(exc, MemoryError):
        return "Bellek yetmedi: daha kısa bir zaman aralığı ya da daha düşük derece seçin"
    return f"Beklenmeyen hata ({type(exc).__name__}): {exc}"


def worker_main(conn) -> None:
    """Yardımcı süreç döngüsü: (iş no, fonksiyon, argümanlar) seçilmiş baytlarını alır, `func(*args,
    progress=…)` çalıştırır. İletiler: ("progress", no, metin), ("ok", no, sonuç), ("error", no, Türkçe ileti,
    iz (günlük için)). Boru kapanınca çıkar."""
    import pickle

    import importlib

    try:  # önceden içe aktar: ilk iş beklemesin (süreç arayüz açılınca boşta başlatılır)
        importlib.import_module("spt.control.sysid")
    except ImportError:  # pragma: no cover
        pass
    while True:
        try:
            job_id, func, args = pickle.loads(conn.recv_bytes())
        except (EOFError, OSError):
            return

        def progress(text, job_id=job_id):
            try:
                conn.send(("progress", job_id, str(text)))
            except (OSError, ValueError):  # pragma: no cover
                pass
        try:
            msg = ("ok", job_id, call_job(func, args, progress))
        except Exception as exc:  # noqa: BLE001 - her hata iletiye çevrilir
            msg = ("error", job_id, error_text(exc), traceback.format_exc(limit=8))
        try:
            conn.send(msg)
        except (OSError, ValueError):
            return
        except Exception as exc:  # noqa: BLE001 - sonuç seçilemedi (pickle)
            conn.send(("error", job_id, error_text(exc), traceback.format_exc(limit=8)))
