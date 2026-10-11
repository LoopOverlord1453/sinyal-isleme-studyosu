"""Genel (senaryo tabanlı) sensör füzyonunun yapı taşları.

Bu modül `spt.dsp.fusion_generic` cephesinin kullandığı tanımları içerir; uygulama kodu doğrudan cepheyi
(`fusion_generic`) kullanmalıdır. Burada tanımlananlar:

  - `Choice`, `Input`, `Output`  : arayüzün form / kanal eşleme / sonuç tablosu için okuduğu tanımlar
                                   (sayısal parametreler `spt.dsp.fusion.Param` ile aynı sınıftır)
  - `Simulation`, `Scenario`     : senaryo tanımı ve analitik (gerçeği tam bilinen) simülasyon çıktısı
  - `Estimator`                  : tüm genel kestiricilerin taban sınıfı (blok blok çalışır, durum taşınır)
  - `FusionError`                : Türkçe, kullanıcıya gösterilebilir doğrulama hatası
  - Birim tabloları, ad / birim normalleştirme ve parçacık filtresi yardımcıları

Ortak kurallar (tüm senaryolar):
  - Girişler SI birimindedir (rad/s, m/s², Pa, m, m/s, rad, T …); dönüşümü arayüz `Input.units` ile yapar.
  - Girişte NaN = "bu örnekte ölçüm yok" (çok hızlı akışta seyrek sensör: 100 Hz akışta 5 Hz GNSS gibi).
  - Açısal çıktılar içeride radyandır; `Output.angle` olan sütunların hatası ±π aralığına sarılır, arayüz
    derece (°) gösterir.
  - `Estimator.run` blok blok çağrılabilir; durum bloklar arasında taşınır ve sonuç, veriyi tek seferde vermekle
    birebir aynıdır (parçacık filtrelerinde de: rastgele sayılar örnek başına aynı sırayla çekilir).
"""
from __future__ import annotations

import math
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
from scipy.linalg.lapack import dpotrf as _dpotrf

from spt.dsp.fusion import GRAVITY, Param

D2R = math.pi / 180.0
R2D = 180.0 / math.pi
TWO_PI = 2.0 * math.pi
NAN = float("nan")

__all__ = [
    "Choice", "D2R", "Estimator", "FusionError", "GRAVITY", "Input", "Output", "Param", "R2D", "Scenario",
    "Simulation", "UNITS_ACC", "UNITS_ANGLE", "UNITS_GENERIC", "UNITS_LENGTH", "UNITS_MAG", "UNITS_PRESSURE",
    "UNITS_RATE", "UNITS_SPEED", "chol_psd", "liu_west_resample", "name_tokens", "normalize_unit",
    "normalized_weights", "resolve_values", "wrap", "wrap_scalar",
]


class FusionError(ValueError):
    """Geçersiz senaryo / algoritma parametresi, eksik giriş ya da hatalı veri şekli.

    Mesaj Türkçedir ve doğrudan kullanıcıya gösterilebilir (ör. durum çubuğunda veya iletişim kutusunda).
    """


# ============================================================================ birim tabloları (birim → SI çarpanı)
UNITS_RATE: dict[str, float] = {"rad/s": 1.0, "°/s": D2R, "mdps": D2R / 1000.0}
UNITS_ACC: dict[str, float] = {"m/s²": 1.0, "g": GRAVITY, "mg": GRAVITY / 1000.0}
UNITS_MAG: dict[str, float] = {"µT": 1e-6, "nT": 1e-9, "G": 1e-4, "mG": 1e-7, "T": 1.0}
UNITS_PRESSURE: dict[str, float] = {"Pa": 1.0, "hPa": 100.0, "kPa": 1000.0, "mbar": 100.0, "bar": 1e5,
                                    "psi": 6894.757293168}
UNITS_LENGTH: dict[str, float] = {"m": 1.0, "cm": 0.01, "mm": 0.001, "km": 1000.0, "ft": 0.3048}
UNITS_SPEED: dict[str, float] = {"m/s": 1.0, "km/h": 1.0 / 3.6, "cm/s": 0.01, "mm/s": 0.001, "mph": 0.44704,
                                 "kn": 1852.0 / 3600.0}
UNITS_ANGLE: dict[str, float] = {"°": D2R, "rad": 1.0}
# Genel (birimden bağımsız) yuvalar: "ham" = olduğu gibi. Diğerleri yalnızca ölçek uyumu içindir.
UNITS_GENERIC: dict[str, float] = {"ham": 1.0, "m": 1.0, "cm": 0.01, "mm": 0.001, "m/s": 1.0, "cm/s": 0.01,
                                   "mm/s": 0.001, "km/h": 1.0 / 3.6, "m/s²": 1.0, "g": GRAVITY,
                                   "mg": GRAVITY / 1000.0, "°C": 1.0}


# ============================================================================ tanım sınıfları
@dataclass(frozen=True)
class Choice:
    """Seçim parametresi (arayüzde açılır liste). `options`: (anahtar, Türkçe etiket) çiftleri."""
    key: str
    label: str
    default: str
    options: tuple[tuple[str, str], ...]
    tip: str = ""

    def __post_init__(self) -> None:
        if self.default not in dict(self.options):
            raise ValueError(f"Choice {self.key}: varsayılan {self.default!r} seçeneklerde yok")


@dataclass(frozen=True)
class Input:
    """Kullanıcının bir (skaler) ya da birkaç (vektör) kanal eşleyeceği giriş yuvası.

    axes   : ("",) skaler; ("x", "y", "z") / ("x", "y") / ("lat", "lon") vektör (eksen başına bir kanal)
    units  : birim etiketi → SI çarpanı; ilk anahtar varsayılandır. Arayüz değeri `çarpan` ile çarparak SI'ya çevirir.
    words  : otomatik eşleme anahtar sözcükleri (küçük harf, ASCII'ye katlanmış; "acc z" gibi çok sözcüklü olabilir)
    """
    key: str
    label: str
    axes: tuple[str, ...]
    units: dict[str, float]
    required: bool
    words: tuple[str, ...]
    tip: str = ""

    @property
    def width(self) -> int:
        return len(self.axes)

    @property
    def scalar(self) -> bool:
        return self.axes == ("",)


@dataclass(frozen=True)
class Output:
    """Kestirilen büyüklük. `angle`: içeride radyan, arayüzde "°"; hatası ±180°'ye sarılır."""
    key: str
    label: str
    unit: str
    angle: bool = False
    words: tuple[str, ...] = ()   # bu çıktının referans (gerçek değer) kanalını bulmak için anahtar sözcükler

    @property
    def display_scale(self) -> float:
        """İç değer → gösterim birimi çarpanı (açılarda rad → °)."""
        return R2D if self.angle else 1.0


@dataclass
class Simulation:
    """Analitik senaryo simülasyonu: girişler SI'da, gerçek değerler (truth) çıktılarla aynı sırada.

    values: simülasyondaki sensörlerle uyumlu senaryo parametreleri (ör. gürültü σ'ları, yuva türleri); arayüz
            "simülasyonla dene" için ya da testler için doğrudan `make_estimator`'a verebilir.
    """
    fs: float
    inputs: dict[str, np.ndarray]
    truth: np.ndarray            # (n, len(outputs)); açılar radyan
    notes: str
    values: dict = field(default_factory=dict)

    @property
    def dt(self) -> float:
        return 1.0 / self.fs

    @property
    def n(self) -> int:
        return len(self.truth)


def resolve_values(params: Iterable[Param | Choice], values: dict | None, what: str) -> dict:
    """Parametre değerlerini doğrular ve varsayılanlarla tamamlar; hatada Türkçe `FusionError`.

    Sayısal: sonlu ve [lo, hi] içinde olmalı (decimals == 0 ise tamsayıya yuvarlanır). Seçim: seçeneklerden biri.
    Bilinmeyen anahtar da hatadır (yazım yanlışı sessizce varsayılana düşmesin). "_" ile başlayan anahtarlar
    parametre değil, çalışma durumudur (ör. `Scenario.raw`'ın sakladığı `_p0`, `_origin`) ve yok sayılır.
    """
    params = tuple(params)
    values = {k: v for k, v in dict(values or {}).items() if not str(k).startswith("_")}
    known = {p.key: p for p in params}
    unknown = sorted(set(values) - set(known))
    if unknown:
        raise FusionError(f"{what}: bilinmeyen parametre: {', '.join(unknown)}")
    out: dict = {}
    for p in params:
        v = values.get(p.key, p.default)
        if isinstance(p, Choice):
            options = dict(p.options)
            if str(v) not in options:
                raise FusionError(f"{what}: '{p.label}' için geçersiz seçim {v!r} "
                                  f"(seçenekler: {', '.join(options)})")
            out[p.key] = str(v)
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise FusionError(f"{what}: '{p.label}' sayı olmalı (verilen: {v!r})") from None
        if not math.isfinite(f) or f < p.lo or f > p.hi:
            raise FusionError(f"{what}: '{p.label}' = {f:g} geçersiz; izin verilen aralık [{p.lo:g}, {p.hi:g}]")
        out[p.key] = float(round(f)) if p.decimals == 0 else f
    return out


def check_input_keys(inputs: tuple[Input, ...], one_of: tuple[tuple[str, ...], ...], keys: Iterable[str]) -> None:
    """Eşlenen giriş anahtarlarını denetler: bilinmeyen, eksik zorunlu ve "en az biri" grupları."""
    keys = set(keys)
    spec = {i.key: i for i in inputs}
    unknown = sorted(keys - set(spec))
    if unknown:
        raise FusionError(f"Bilinmeyen giriş: {', '.join(unknown)} (geçerli: {', '.join(spec)})")
    missing = [i.label for i in inputs if i.required and i.key not in keys]
    if missing:
        raise FusionError(f"Zorunlu giriş eşlenmemiş: {', '.join(missing)}")
    for group in one_of:
        if not keys & set(group):
            labels = " ya da ".join(spec[k].label for k in group)
            raise FusionError(f"Şu girişlerden en az biri eşlenmeli: {labels}")


def input_arrays(inputs: tuple[Input, ...], data) -> tuple[dict[str, np.ndarray], int]:
    """Giriş sözlüğünü doğrular ve float64 dizilere çevirir: skaler (n,), vektör (n, eksen); ortak n döner."""
    if not isinstance(data, dict):
        raise FusionError("Girişler {anahtar: dizi} sözlüğü olmalı")
    spec = {i.key: i for i in inputs}
    unknown = sorted(set(data) - set(spec))
    if unknown:
        raise FusionError(f"Bilinmeyen giriş: {', '.join(unknown)} (geçerli: {', '.join(spec)})")
    arrays: dict[str, np.ndarray] = {}
    n: int | None = None
    for key, value in data.items():
        inp = spec[key]
        try:
            a = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            raise FusionError(f"'{inp.label}' girişi sayısal bir dizi olmalı") from None
        if inp.scalar:
            if a.ndim == 2 and a.shape[1] == 1:
                a = a[:, 0]
            if a.ndim != 1:
                raise FusionError(f"'{inp.label}' skaler giriştir: (n,) dizisi bekleniyor, gelen {a.shape}")
        elif a.ndim != 2 or a.shape[1] != inp.width:
            raise FusionError(f"'{inp.label}' girişi (n, {inp.width}) olmalı, gelen {a.shape}")
        if n is None:
            n = len(a)
        elif len(a) != n:
            raise FusionError("Tüm girişler aynı uzunlukta olmalı")
        arrays[key] = np.ascontiguousarray(a)
    return arrays, n or 0


@dataclass(frozen=True)
class Scenario:
    """Bir füzyon senaryosu: girişler, çıktılar, ortak parametreler, algoritmalar ve analitik simülatör.

    algorithms: sıralı; ilk algoritma en basit taban çizgisidir (karşılaştırma için).
    one_of    : her grup için en az bir girişin eşlenmesi gerekir (ör. basınç YA DA baro irtifa).
    """
    key: str
    name: str
    description: str
    inputs: tuple[Input, ...]
    outputs: tuple[Output, ...]
    params: tuple[Param | Choice, ...]
    algorithms: dict[str, type[Estimator]]
    simulator: Callable[..., Simulation] | None = field(default=None, repr=False, compare=False)
    one_of: tuple[tuple[str, ...], ...] = ()
    default_fs: float = 100.0
    suggest: Callable[..., dict] | None = field(default=None, repr=False, compare=False)
    raw_fn: Callable[..., np.ndarray] | None = field(default=None, repr=False, compare=False)

    def raw(self, inputs: dict[str, np.ndarray], values: dict | None = None) -> np.ndarray:
        """Füzyonsuz ("ham") kestirim: her çıktı tek sensörden doğrudan, (n, len(outputs)); uygun değilse NaN.

        Arayüzdeki gri "ham (füzyonsuz)" eğri içindir. Girişler `Estimator.run` ile aynı (SI, NaN = ölçüm yok);
        eksik girişler hata değildir (ilgili sütun NaN olur). Zamanda filtreleme yoktur, her örnek bağımsızdır.

        Parça parça çağrı: hesap durumsuzdur; tek istisna bir referans noktası gereken dönüşümlerdir (irtifada
        p0 = 0 "otomatik" iken ilk basınç örneği, 2B konumda enlem/boylamın açıldığı ilk GNSS düzeltmesi). Bunlar
        ilk bulunduklarında `values` sözlüğüne "_" önekli anahtarla yazılır (`_p0` Pa, `_origin` = (enlem, boylam)
        rad) ve sonraki çağrılarda oradan okunur. Aynı sözlüğü akış boyunca yeniden verin: parça parça sonuç tek
        seferlik çağrıyla birebir aynıdır ve kestiricilerin (akışın ilk örneğinden aldıkları) referansıyla örtüşür.
        "_" önekli anahtarlar parametre doğrulamasında yok sayılır; sözlük `make_estimator`'a da verilebilir.
        """
        if values is None:
            values = {}
        sv = self.resolve(values)
        arrays, n = input_arrays(self.inputs, inputs)
        if n == 0 or self.raw_fn is None:
            return np.full((n, len(self.outputs)), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.asarray(self.raw_fn(arrays, n, sv, values), dtype=np.float64)
        out[~np.isfinite(out)] = np.nan
        return out

    def simulate(self, duration: float = 60.0, fs: float | None = None, seed: int = 1, **kw) -> Simulation:
        """Gerçeği tam bilinen analitik veri üretir (girişler SI, NaN = o örnekte ölçüm yok)."""
        if self.simulator is None:
            raise FusionError(f"{self.name}: simülatör tanımlı değil")
        fs = self.default_fs if fs is None else float(fs)
        if not (duration > 0 and fs > 0):
            raise FusionError("Simülasyon süresi ve örnekleme hızı pozitif olmalı")
        return self.simulator(float(duration), fs, int(seed), **kw)

    def input(self, key: str) -> Input:
        for i in self.inputs:
            if i.key == key:
                return i
        raise FusionError(f"{self.name}: bilinmeyen giriş {key!r}")

    def output_index(self, key: str) -> int:
        for i, o in enumerate(self.outputs):
            if o.key == key:
                return i
        raise FusionError(f"{self.name}: bilinmeyen çıktı {key!r}")

    def resolve(self, values: dict | None) -> dict:
        """Senaryo parametrelerini doğrular, varsayılanlarla tamamlar."""
        return resolve_values(self.params, values, self.name)

    def check(self, inputs: Iterable[str], values: dict | None = None) -> None:
        """Başlatmadan önce doğrulama: eşlenen giriş anahtarları + senaryo değerleri (+ aralarındaki tutarlılık).

        Bir algoritmanın oluşturulması / ilk `run` çağrısı da aynı denetimi yapar; arayüz hatayı erkenden
        göstermek için bunu çağırabilir.
        """
        keys = set(inputs)
        check_input_keys(self.inputs, self.one_of, keys)
        first = next(iter(self.algorithms.values()))
        est = first(values or {})
        est._check_inputs(frozenset(keys))


# ============================================================================ kestirici taban sınıfı
class Estimator:
    """Genel kestirici. Alt sınıf `_setup` (durumu sıfırla) ve `_run` (örnek döngüsü) yazar.

    Kullanım:
        est = make_estimator("altitude", "kf", {"sigma_p": 4.0}, {})
        y = est.run({"baro_p": p, "acc_z": a}, dt)       # (n, len(outputs)); bloklar halinde tekrar çağrılabilir
        est.diagnostics()                                 # {"İvme sapması (m/s²)": "+0.118", …}

    `run`'a verilen giriş anahtarları ilk çağrıda sabitlenir (eşleme değişirse yeni kestirici oluşturun).
    Eşlenmeyen isteğe bağlı girişler sözlükte hiç bulunmamalıdır (tümü NaN dizi de kabul edilir).
    """
    key: str = ""
    name: str = ""
    params: tuple[Param | Choice, ...] = ()
    # Senaryo tanımı: senaryo modülündeki ortak taban sınıf doldurur
    scenario_params: tuple[Param | Choice, ...] = ()
    inputs: tuple[Input, ...] = ()
    outputs: tuple[Output, ...] = ()
    one_of: tuple[tuple[str, ...], ...] = ()

    def __init__(self, scenario_values: dict | None = None, **values) -> None:
        self.scenario_values: dict = resolve_values(self.scenario_params, scenario_values, "Senaryo")
        self.values: dict = resolve_values(self.params, values, self.name or self.key)
        self.sv = self.scenario_values            # kısa ad (alt sınıflar için)
        self.seconds = 0.0                         # toplam hesap süresi (örnek başına maliyet için)
        self.samples = 0
        self._keys: frozenset[str] | None = None
        self.reset()

    # ---------------------------------------------------------------- alt sınıf kancaları
    def _setup(self) -> None:
        """Durumu başlangıca döndür."""

    def _check_inputs(self, keys: frozenset[str]) -> None:
        """Eşlenen girişlere bağlı ek doğrulama (ör. yuva türü ↔ model derecesi); hatada `FusionError`."""

    def _run(self, x: dict[str, np.ndarray], n: int, dt: float) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    # ---------------------------------------------------------------- genel arayüz
    def reset(self) -> None:
        """Kestiriciyi ilk haline döndürür (parametreler korunur; giriş anahtarları yeniden sabitlenir)."""
        self._keys = None
        self._setup()

    def run(self, inputs: dict[str, np.ndarray], dt: float) -> np.ndarray:
        """Bir blok işler; (n, len(outputs)) float64 döndürür. Başlatılmamış örneklerde çıktı NaN'dır."""
        try:
            dt = float(dt)
        except (TypeError, ValueError):
            raise FusionError(f"Örnekleme aralığı sayı olmalı (verilen: {dt!r})") from None
        if not (math.isfinite(dt) and dt > 0):
            raise FusionError(f"Örnekleme aralığı pozitif ve sonlu olmalı (verilen: {dt!r})")
        arrays, n = self._prepare(inputs)
        if n == 0:
            return np.empty((0, len(self.outputs)))
        start = time.perf_counter()
        out = self._run(arrays, n, dt)
        self.seconds += time.perf_counter() - start
        self.samples += n
        return out

    def diagnostics(self) -> dict[str, str]:
        """Sonuç tablosu için (etiket → metin) ek bilgiler: sapma kestirimleri, ESS …"""
        return {}

    @property
    def cost_us(self) -> float:
        """Örnek başına ortalama hesap süresi (µs)."""
        return self.seconds / self.samples * 1e6 if self.samples else 0.0

    # ---------------------------------------------------------------- yardımcılar
    def _prepare(self, inputs) -> tuple[dict[str, np.ndarray], int]:
        if not isinstance(inputs, dict):
            raise FusionError("Girişler {anahtar: dizi} sözlüğü olmalı")
        keys = frozenset(inputs)
        if self._keys is None:
            check_input_keys(self.inputs, self.one_of, keys)
            self._check_inputs(keys)
            self._keys = keys
        elif keys != self._keys:
            raise FusionError("Eşlenen girişler çağrılar arasında değişemez "
                              f"(ilk: {', '.join(sorted(self._keys))}; şimdi: {', '.join(sorted(keys))})")
        return input_arrays(self.inputs, inputs)


# ============================================================================ sayısal yardımcılar
def wrap(angle):
    """Açıyı [−π, π) aralığına sarar (dizi; NaN korunur)."""
    with np.errstate(invalid="ignore"):
        return (np.asarray(angle, dtype=np.float64) + math.pi) % TWO_PI - math.pi


def wrap_scalar(a: float) -> float:
    return (a + math.pi) % TWO_PI - math.pi


def chol_psd(a: np.ndarray) -> np.ndarray:
    """A = L·Lᵀ olan L (alt üçgen Cholesky — LAPACK dpotrf, numpy'den ~2× hızlı küçük matrislerde;
    yarı-tanımlı / bozuk matriste özdeğer ayrışımıyla, negatif özdeğerler sıfırlanarak)."""
    c, info = _dpotrf(a, lower=1, clean=1)
    if info == 0:
        return c
    vals, vecs = np.linalg.eigh(0.5 * (a + a.T))
    return vecs * np.sqrt(np.clip(vals, 0.0, None))


def ut_weights(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Kokusuz dönüşüm ağırlıkları, α = 1, β = 2, κ = 0 (λ = 0): tüm ortalama ağırlıkları ≥ 0, yayılım √(n·P).

    `spt.dsp.fusion.UKF` ile aynı seçim; 2n + 1 sigma noktası, merkez noktanın ortalama ağırlığı 0.
    """
    wm = np.r_[0.0, np.full(2 * n, 1.0 / (2 * n))]
    wc = np.r_[2.0, np.full(2 * n, 1.0 / (2 * n))]
    return wm, wc


def normalized_weights(logw: np.ndarray) -> np.ndarray:
    """Log-ağırlıklardan normalize ağırlıklar; tümü geçersizse (NaN / −∞) eşit ağırlık."""
    logw = np.where(np.isnan(logw), -np.inf, logw)
    m = logw.max()
    if not math.isfinite(m):
        return np.full(len(logw), 1.0 / len(logw))
    w = np.exp(logw - m)
    return w / w.sum()


def systematic_indices(rng: np.random.Generator, w: np.ndarray) -> np.ndarray:
    n = len(w)
    c = np.cumsum(w)
    c[-1] = 1.0
    return np.searchsorted(c, (rng.random() + np.arange(n)) / n)


LIU_WEST_A = 0.98       # çekirdek daraltma katsayısı a
LIU_WEST_INFLATE = 0.02  # yeniden örnekleme başına net varyans şişirmesi (çökmeye karşı)


def liu_west_resample(rng: np.random.Generator, X: np.ndarray, w: np.ndarray,
                      angle_rows: tuple[int, ...] = (), a: float = LIU_WEST_A,
                      inflate: float = LIU_WEST_INFLATE) -> np.ndarray:
    """Sistematik yeniden örnekleme + Liu–West çekirdek yumuşatma (düzenlileştirilmiş parçacık filtresi).

    X: (d, N) parçacıklar (satır = durum bileşeni), w: normalize ağırlıklar. Seçilen parçacıklar ağırlıklı
    ortalamaya a oranında çekilir ve N(0, (1 − a² + inflate)·Σ) gürültü eklenir: ağırlıklı ortalama korunur,
    kovaryans `inflate` (%2) kadar büyür. Tam momen koruyan çekirdek (inflate = 0) süreç gürültüsü küçük
    sistemlerde (ör. baro-ataletsel) Monte Carlo hatasıyla varyansın sıfıra yürümesine ve izin kaybına yol
    açar; küçük net şişirme bunu önler. Sabit parametreler (sapma, ölçek) de çeşitliliğini korur.
    `angle_rows` satırları sarılı (dairesel ortalama) işlenir.
    """
    d, n = X.shape
    mean = X @ w
    dev = X - mean[:, None]
    for r in angle_rows:
        mean[r] = math.atan2(float(w @ np.sin(X[r])), float(w @ np.cos(X[r])))
        dev[r] = wrap(X[r] - mean[r])
    cov = (dev * w) @ dev.T
    idx = systematic_indices(rng, w)
    new = mean[:, None] + a * dev[:, idx]
    new += chol_psd((1.0 - a * a + inflate) * cov) @ rng.standard_normal((d, n))
    for r in angle_rows:
        new[r] = wrap(new[r])
    return new


# ============================================================================ ad / birim normalleştirme
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_SPLIT = re.compile(r"[^a-z0-9]+")


def fold(text: str) -> str:
    """NFKD + birleşik işaretleri at + Türkçe ı/İ → i: "İvme X" → "Ivme X", "Basınç" → "Basinc"."""
    text = (text or "").replace("ı", "i").replace("İ", "I")
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def name_tokens(name: str) -> list[str]:
    """Kanal adını küçük harfli ASCII parçalara böler: "GyroX" → [gyro, x]; "İvme_Z" → [ivme, z]."""
    text = _CAMEL.sub(" ", fold(name))
    return [t for t in _SPLIT.split(text.lower()) if t]


_UNIT_ALIASES = {
    "dps": "°/s", "°/sec": "°/s", "°ps": "°/s", "rad/sec": "rad/s", "rps": "rad/s",
    "mps": "m/s", "kmh": "km/h", "kph": "km/h", "km/sa": "km/h", "knot": "kn", "kt": "kn",
    "m/s/s": "m/s2", "ms-2": "m/s2", "m.s-2": "m/s2", "mps2": "m/s2", "mpss": "m/s2",
    "gauss": "g", "mgauss": "mg", "millibar": "mbar", "mb": "mbar", "pascal": "pa",
    "ut": "ut", "microtesla": "ut", "metre": "m", "meter": "m", "derece": "°", "radyan": "rad", "°c": "°c",
}


def normalize_unit(unit: str) -> str:
    """Birim metnini karşılaştırma için normalleştirir: "m/s²" → "m/s2", "deg/s" → "°/s", "µT" → "ut"."""
    u = (unit or "").strip().replace("º", "°").replace("˚", "°")
    u = unicodedata.normalize("NFKC", u).lower()
    u = u.replace(" ", "").replace("^", "").replace("μ", "u").replace("µ", "u")
    u = re.sub(r"degrees?|deg", "°", u)
    return _UNIT_ALIASES.get(u, u)


def unit_key(units: dict[str, float], unit: str) -> str | None:
    """Kanal biriminin `units` içindeki anahtarı (eşleşme yoksa None)."""
    target = normalize_unit(unit)
    if not target:
        return None
    for key in units:
        if normalize_unit(key) == target:
            return key
    return None


def fmt_vec(values: Iterable[float], digits: int = 3) -> str:
    return "[" + ", ".join(f"{v:+.{digits}f}" for v in values) + "]"
