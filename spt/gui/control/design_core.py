"""Kontrolcü Tasarımı sekmesinin Qt'siz çekirdeği: kompansatör gösterimi, çevrim mimarileri, zaman alanı dönüşümü,
metin ayrıştırıcıları ve PID ayarlayıcı kaydırıcı eşlemeleri.

Hiçbir kontrol hesabı burada yeniden yazılmaz: bileşimler `spt.control.connect.feedback` ve LTI aritmetiğiyle,
ayrıklaştırma `c2d` / `d2c` / `d2d` ile, kutuplar `pole` ile, paylar `margin` / `loop_margins` ile, öncül / ardıl /
çentik öğeleri `spt.control.classical.lead / lag / notch` ile kurulur.

Kompansatör (MATLAB Control System Designer'ın "zero/pole/gain" biçimi):

    C(x) = K · Π(x − z_i) / Π(x − p_j)          x = s (sürekli) ya da z (ayrık; kutup / sıfırlar z düzleminde)

Öğeler: gerçel kutup / sıfır, karmaşık kutup / sıfır çifti (üst yarı düzlemdeki eleman saklanır, eşleniği
kendiliğinden eklenir), integratör (s = 0 / z = 1'de kutup), türevleyici (s = 0 / z = 1'de sıfır), öncül / ardıl
(bir gerçel sıfır + bir gerçel kutup), çentik (karmaşık sıfır çifti + karmaşık kutup çifti). Kök yer eğrisi
parametresi K'dir: L = K · C0 · G (C0 birim kazançlı kompansatör); K < 0 ise eğri −C0·G için |K| ile çizilir.

Çevrim mimarileri (MATLAB'in standart yapıları; d: bitki girişindeki bozucu):

    single       u = C·(r − H·y)                       T = feedback(G·C, H)          L = C·H·G
    prefilter    u = C·(F·r − H·y)                     T = feedback(G·C, H)·F
    feedforward  u = C·(r − H·y) + F·r                 T = feedback(G, C·H)·(C + F)
    cascade      ω_ref = C1·(r − θ), u = C2·(ω_ref − ω), ω = G2·u, θ = G1·ω   (drone: iç açısal hız, dış açı)
                 T2 = feedback(G2·C2, 1)  (iç kapalı çevrim),  T = feedback(C1·T2·G1, 1)
                 iç çevrim düzenlenirken dış çevrim açık: L = C2·G2; dış çevrimde iç çevrim kapalı: L = C1·T2·G1

Bozucu (d → y) ve kontrol çabası (r → u) cevapları da aynı bileşimlerle (`GS`, `KS`) kurulur.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from spt.control.errors import ControlError

ARCHS = (
    ("single", "Tek çevrim", "u = C·(r − H·y)"),
    ("prefilter", "Ön filtreli", "u = C·(F·r − H·y)"),
    ("feedforward", "İleri beslemeli", "u = C·(r − H·y) + F·r"),
    ("cascade", "Kademeli (iç / dış çevrim)", "ω_ref = C1·(r − θ), u = C2·(ω_ref − ω)"),
)
ARCH_DIAGRAMS = {
    "single": "\n".join(["r ─►(+)─► C ─► G ─┬─► y",
                         "     ▲ −          │",
                         "     └──── H ◄────┘"]),
    "prefilter": "\n".join(["r ─► F ─►(+)─► C ─► G ─┬─► y",
                            "          ▲ −          │",
                            "          └──── H ◄────┘"]),
    "feedforward": "\n".join(["r ─┬──► F ──────┐",
                              "   └►(+)─► C ─►(+)─► G ─┬─► y",
                              "      ▲ −               │",
                              "      └────── H ◄───────┘"]),
    "cascade": "\n".join(["r ─►(+)─► C1 ─►(+)─► C2 ─► G2 ─┬─► G1 ─┬─► θ",
                          "     ▲ −        ▲ −            │ ω     │",
                          "     │          └──────────────┘       │",
                          "     └─────────────────────────────────┘"]),
}
ELEMENT_KINDS = {  # anahtar: (görünen ad, sıfır sayısı, kutup sayısı, karmaşık mı, sürüklenebilir mi)
    "real_pole": ("Gerçel kutup", 0, 1, False, True),
    "real_zero": ("Gerçel sıfır", 1, 0, False, True),
    "cplx_pole": ("Karmaşık kutup çifti", 0, 1, True, True),
    "cplx_zero": ("Karmaşık sıfır çifti", 1, 0, True, True),
    "integrator": ("İntegratör", 0, 1, False, False),
    "differentiator": ("Türevleyici", 1, 0, False, False),
    "lead": ("Öncül (lead)", 1, 1, False, True),
    "lag": ("Ardıl (lag)", 1, 1, False, True),
    "notch": ("Çentik (notch)", 1, 1, True, True),
}
PID_TYPES = ("P", "I", "PI", "PD", "PDF", "PID", "PIDF")
DESIGN_FOCUS = (("balanced", "Dengeli"), ("reference-tracking", "Referans izleme"),
                ("disturbance-rejection", "Bozucu reddi"))
FORMULAS = (("ForwardEuler", "İleri Euler"), ("BackwardEuler", "Geri Euler"), ("Trapezoidal", "Yamuk (Tustin)"))
D2C_OK = ("zoh", "tustin", "matched")
_TOL = 1e-12


# ============================================================================ PID ayarlayıcı eşlemeleri
def response_time_to_wc(tau: float) -> float:
    """Cevap süresi τ (s) → hedef geçiş frekansı ωc = 2/τ (rad/s). Tanım bizimdir (MATLAB PID Tuner'ın eşlemesi
    yayımlanmamış); ölçülen yükselme / yerleşme süreleri tabloda ayrıca gösterilir."""
    if not tau > 0:
        raise ControlError("Cevap süresi pozitif olmalı.")
    return 2.0 / tau


def wc_to_response_time(wc: float) -> float:
    return 2.0 / wc


def transient_to_pm(theta: float) -> float:
    """Geçici durum davranışı θ ∈ [0, 1] (0 agresif, 1 dayanıklı) → hedef faz payı 30° + 50°·θ (θ = 0.6 → 60°)."""
    return 30.0 + 50.0 * min(max(float(theta), 0.0), 1.0)


def pm_to_transient(pm: float) -> float:
    return min(max((float(pm) - 30.0) / 50.0, 0.0), 1.0)


# ============================================================================ metin ayrıştırıcıları
_NUM = r"[-+−]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def _clean(text: str) -> str:
    return (text or "").replace("−", "-").replace("·", "*").strip()


def parse_number(text: str, what: str = "Değer") -> float:
    t = _clean(text).lower()
    if t in ("inf", "+inf", "∞"):
        return math.inf
    if t in ("-inf", "-∞"):
        return -math.inf
    try:
        return float(t)
    except ValueError:
        raise ControlError(f"{what}: “{text}” bir sayı değil.") from None


def parse_vector(text: str, what: str = "Vektör", allow_empty: bool = True) -> np.ndarray | None:
    """“1, 2, 3”, “1 2 3”, “[1 2 3]” → dizi; boş → None (allow_empty) ya da hata. inf / -inf kabul edilir."""
    t = _clean(text).strip("[]() ")
    if not t:
        if allow_empty:
            return None
        raise ControlError(f"{what} boş olamaz.")
    parts = [p for p in re.split(r"[,\s;]+", t) if p]
    return np.array([parse_number(p, what) for p in parts], dtype=float)


def parse_matrix(text: str, n: int, what: str = "Matris") -> np.ndarray:
    """Ağırlık matrisi: “1, 10” (köşegen), “diag(1, 10)”, “[1 0; 0 10]” (tam) ya da tek sayı (sayı·I)."""
    t = _clean(text)
    if not t:
        raise ControlError(f"{what} boş olamaz.")
    m = re.fullmatch(r"diag\s*\((.*)\)", t, flags=re.IGNORECASE)
    if m:
        t = m.group(1)
    elif ";" in t:
        rows = [r for r in t.strip("[] ").split(";") if r.strip()]
        M = np.array([[parse_number(v, what) for v in re.split(r"[,\s]+", r.strip()) if v] for r in rows],
                     dtype=float)
        if M.shape != (n, n):
            raise ControlError(f"{what} {n}×{n} olmalı ({M.shape[0]}×{M.shape[1] if M.ndim > 1 else 1} verildi).")
        return M
    v = parse_vector(t, what, allow_empty=False)
    if v.size == 1:
        return float(v[0]) * np.eye(n)
    if v.size != n:
        raise ControlError(f"{what}: {n} köşegen elemanı bekleniyordu ({v.size} verildi).")
    return np.diag(v)


def parse_roots(text: str, what: str = "Kutuplar") -> np.ndarray:
    """“-2±3j, -5”, “-1+2j -1-2j”, “0.9±0.1j” → karmaşık dizi (± eşlenik çifti açar)."""
    t = _clean(text).replace("+-", "±").replace("+/-", "±")
    if not t.strip("[] "):
        raise ControlError(f"{what} boş olamaz.")
    # “−1.5 + 2j” gibi boşluklu yazımı tek sayıya birleştir
    t = re.sub(r"\s*([+\-±])\s*(?=(?:\d+\.?\d*|\.\d+)(?:[eE][+\-]?\d+)?\*?[ij])", r"\1", t)
    out: list[complex] = []
    for tok in [p for p in re.split(r"[,;\s]+", t.strip("[] ")) if p]:
        tok = tok.replace("i", "j")
        if "±" in tok:
            a, b = tok.split("±", 1)
            re_ = parse_number(a or "0", what)
            b = b.replace("*", "").replace("j", "")
            im = parse_number(b or "1", what)
            out += [complex(re_, abs(im)), complex(re_, -abs(im))]
            continue
        try:
            out.append(complex(tok.replace("*", "")))
        except ValueError:
            raise ControlError(f"{what}: “{tok}” bir sayı değil.") from None
    return np.array(out, dtype=complex)


def fmt_complex(x: complex, digits: int = 6) -> str:
    x = complex(x)
    if abs(x.imag) <= _TOL * max(1.0, abs(x)):
        return f"{x.real:.{digits}g}"
    return f"{x.real:.{digits}g} {'+' if x.imag >= 0 else '−'} {abs(x.imag):.{digits}g}j"


def fmt_root(x: complex, digits: int = 10) -> str:
    """Boşluksuz kök yazımı (metin kutusuna geri ayrıştırılabilir): −1.5+2j, 0.8."""
    x = complex(x)
    if abs(x.imag) <= _TOL * max(1.0, abs(x)):
        return f"{x.real:.{digits}g}"
    return f"{x.real:.{digits}g}{'+' if x.imag >= 0 else '-'}{abs(x.imag):.{digits}g}j"


def fmt_matrix(M, digits: int = 6) -> str:
    M = np.atleast_2d(np.asarray(M, float))
    return "\n".join("  ".join(f"{v: .{digits}g}" for v in row) for row in M)


# ============================================================================ zaman alanı
def lti_module():
    from spt.control import lti
    return lti


def to_domain(sys, Ts: float, method: str = "zoh"):
    """Modeli tasarım zamanına taşır: Ts = 0 → sürekli (ayrıksa d2c), Ts > 0 → ayrık (sürekliyse c2d; farklı Ts'li
    ayrıksa d2d). d2c / d2d yalnız zoh / tustin / matched destekler; diğer yöntemlerde zoh kullanılır."""
    if sys is None:
        return None
    from spt.control.discretize import c2d, d2c, d2d
    if Ts <= 0:
        if sys.Ts > 0:
            return d2c(sys, method if method in D2C_OK else "zoh")
        return sys
    if sys.Ts == 0:
        return c2d(sys, Ts, method=method)
    if abs(sys.Ts - Ts) > 1e-12 * Ts:
        return d2d(sys, Ts, method="tustin" if method == "tustin" else "zoh")
    return sys


def cont_equivalent(sys, method: str = "zoh"):
    """Ayrık modelin sürekli eşdeğeri: d2c (seçili yöntem; foh / impulse ya da uygulanamazsa — ör. zoh'ta z = 0
    yakınında kutup — sırayla tustin, matched). Dönüş (model, kullanılan yöntem)."""
    from spt.control.discretize import d2c
    tried = [method] if method in D2C_OK else []
    tried += [m for m in ("tustin", "matched") if m not in tried]
    err = None
    for m in tried:
        try:
            return d2c(sys, m), m
        except ControlError as exc:
            err = exc
    raise err


def require_siso(sys, what: str):
    if sys is None:
        raise ControlError(f"{what} seçilmedi.")
    if sys.ninputs != 1 or sys.noutputs != 1:
        raise ControlError(f"{what} tek girişli, tek çıkışlı (SISO) olmalı: klasik tasarım SISO çevrim içindir "
                           "(çok girişli bitki için Durum Uzayı ya da MPC panelini kullanın).")
    return sys


# ============================================================================ kompansatör
@dataclass
class Element:
    kind: str
    zeros: list = field(default_factory=list)   # temsilci kökler (karmaşık türlerde üst yarı düzlem elemanı)
    poles: list = field(default_factory=list)
    params: dict = field(default_factory=dict)  # bilgi amaçlı (öncül: φmax, ωm; çentik: ω0, derinlik, genişlik)

    @property
    def label(self) -> str:
        return ELEMENT_KINDS[self.kind][0]

    @property
    def complex_pair(self) -> bool:
        return ELEMENT_KINDS[self.kind][3]

    @property
    def movable(self) -> bool:
        return ELEMENT_KINDS[self.kind][4]

    def expand(self, roots) -> list[complex]:
        out = []
        for r in roots:
            r = complex(r)
            if self.complex_pair:
                out += [complex(r.real, abs(r.imag)), complex(r.real, -abs(r.imag))]
            else:
                out.append(complex(r.real, 0.0))
        return out

    def copy(self) -> "Element":
        return Element(self.kind, list(self.zeros), list(self.poles), dict(self.params))

    def to_dict(self) -> dict:
        return {"kind": self.kind, "zeros": [[complex(z).real, complex(z).imag] for z in self.zeros],
                "poles": [[complex(p).real, complex(p).imag] for p in self.poles], "params": dict(self.params)}

    @staticmethod
    def from_dict(d: dict) -> "Element":
        if d.get("kind") not in ELEMENT_KINDS:
            raise ControlError("Bilinmeyen kompansatör öğesi.")
        return Element(d["kind"], [complex(a, b) for a, b in d.get("zeros", [])],
                       [complex(a, b) for a, b in d.get("poles", [])], dict(d.get("params", {})))


class Compensator:
    """C = K·Π(x − z)/Π(x − p) (öğe listesi). `pid`: kompansatör PID ayarlayıcısından geldiyse (ve sonra elle
    değiştirilmediyse) PID nesnesi — C kodu ve Modellere ekleme PID yapısını korur."""

    def __init__(self, K: float = 1.0, elements: list[Element] | None = None, Ts: float = 0.0, pid=None):
        self.K = float(K)
        self.elements: list[Element] = list(elements or [])
        self.Ts = float(Ts)
        self.pid = pid

    # ---------------------------------------------------------------- sorgular
    @property
    def discrete(self) -> bool:
        return self.Ts > 0

    @property
    def origin(self) -> float:
        """İntegratör / türevleyici konumu: s = 0 ya da z = 1."""
        return 1.0 if self.discrete else 0.0

    def zeros(self) -> np.ndarray:
        return np.array([z for e in self.elements for z in e.expand(e.zeros)], dtype=complex)

    def poles(self) -> np.ndarray:
        return np.array([p for e in self.elements for p in e.expand(e.poles)], dtype=complex)

    def zpk(self):
        """Kompansatör (spt.control.lti.ZPK)."""
        return lti_module().zpk(self.zeros(), self.poles(), self.K, self.Ts)

    def normalized(self, sign: float = 1.0):
        """C0 = sign·Π(x − z)/Π(x − p) (kök yer eğrisi için birim kazançlı biçim)."""
        return lti_module().zpk(self.zeros(), self.poles(), sign, self.Ts)

    def controller(self):
        """Modellere eklenecek / C kodu üretilecek nesne: PID yapısı korunuyorsa PID, değilse ZPK."""
        return self.pid if self.pid is not None else self.zpk()

    def dc_gain(self) -> float:
        """C(0) (sürekli) / C(1) (ayrık); integratör varsa ±inf, türevleyici varsa 0."""
        x = self.origin
        z, p = self.zeros(), self.poles()
        nz = int(np.sum(np.abs(z - x) <= _TOL))
        npl = int(np.sum(np.abs(p - x) <= _TOL))
        if npl > nz:
            return math.copysign(math.inf, self.K) if self.K else 0.0
        if nz > npl:
            return 0.0
        zz = z[np.abs(z - x) > _TOL]
        pp = p[np.abs(p - x) > _TOL]
        return float(np.real(self.K * np.prod(x - zz) / np.prod(x - pp)))

    def copy(self) -> "Compensator":
        return Compensator(self.K, [e.copy() for e in self.elements], self.Ts, self.pid)

    def is_proper(self) -> bool:
        return self.zeros().size <= self.poles().size

    # ---------------------------------------------------------------- değişiklikler (PID yapısı bozulur)
    def touch(self) -> None:
        self.pid = None

    def set_gain(self, K: float) -> None:
        K = float(K)
        if not math.isfinite(K):
            raise ControlError("Kazanç sonlu olmalı.")
        self.K = K
        self.touch()

    def add(self, kind: str, value: complex | None = None) -> Element:
        """Temel öğe ekler (integratör / türevleyici konumu sabittir)."""
        if kind not in ELEMENT_KINDS or kind in ("lead", "lag", "notch"):
            raise ControlError("Bu öğe kendi tasarım yardımcısıyla eklenir (öncül / ardıl / çentik).")
        _label, nz, np_, cplx, _mov = ELEMENT_KINDS[kind]
        if kind in ("integrator", "differentiator"):
            v = complex(self.origin)
        else:
            v = complex(value if value is not None else self._default_root(cplx))
            if not cplx:
                v = complex(v.real, 0.0)
            elif v.imag < 0:
                v = v.conjugate()
        e = Element(kind, [v] * nz, [v] * np_)
        self.elements.append(e)
        self.touch()
        return e

    def _default_root(self, cplx: bool) -> complex:
        if self.discrete:
            return complex(0.5, 0.3) if cplx else complex(0.5, 0.0)
        return complex(-1.0, 1.0) if cplx else complex(-1.0, 0.0)

    def add_system(self, kind: str, sys, params: dict | None = None) -> Element:
        """Kütüphanenin kurduğu öğeyi (lead / lag / notch) seri ekler: C ← C·sys (sys'in ZPK kazancı K'ye katılır)."""
        if abs(sys.Ts - self.Ts) > 1e-12 * max(1.0, self.Ts):
            raise ControlError("Öğenin örnekleme zamanı kompansatörünkiyle aynı olmalı.")
        part = Compensator.from_lti(sys)
        kinds = sorted(e.kind for e in part.elements)
        if kind in ("lead", "lag") and kinds == ["real_pole", "real_zero"]:
            zero = next(e for e in part.elements if e.kind == "real_zero").zeros
            pole = next(e for e in part.elements if e.kind == "real_pole").poles
            new = [Element(kind, list(zero), list(pole), dict(params or {}))]
        elif kind == "notch" and kinds == ["cplx_pole", "cplx_zero"]:
            zero = next(e for e in part.elements if e.kind == "cplx_zero").zeros
            pole = next(e for e in part.elements if e.kind == "cplx_pole").poles
            new = [Element("notch", list(zero), list(pole), dict(params or {}))]
        else:   # ör. sönümü 1'i aşan çentik kutupları gerçel çıkar: ayrı öğeler olarak eklenir
            new = part.elements
        self.elements += new
        self.K *= part.K
        self.touch()
        return new[0]

    def remove(self, index: int) -> None:
        del self.elements[index]
        self.touch()

    def move(self, index: int, which: str, j: int, value: complex) -> complex:
        """Öğenin köküne yeni konum (sürükleme / tablo): gerçel öğeler gerçel eksende kalır, karmaşık çiftlerde üst
        yarı düzlem elemanı saklanır. Dönen değer uygulanan konumdur."""
        e = self.elements[index]
        if not e.movable:
            raise ControlError(f"{e.label} konumu sabittir (s = 0 / z = 1).")
        v = complex(value)
        if not (math.isfinite(v.real) and math.isfinite(v.imag)):
            raise ControlError("Konum sonlu olmalı.")
        v = complex(v.real, abs(v.imag)) if e.complex_pair else complex(v.real, 0.0)
        roots = e.zeros if which == "z" else e.poles
        roots[j] = v
        self.touch()
        return v

    # ---------------------------------------------------------------- kurucular
    @classmethod
    def from_lti(cls, sys, keep_pid: bool = False) -> "Compensator":
        """SISO modeli öğelere ayırır: x = 0 / 1'deki kökler integratör / türevleyici, gerçel kökler gerçel öğe,
        eşlenik çiftler karmaşık öğe. Kazanç ZPK kazancıdır."""
        lti = lti_module()
        require_siso(sys, "Kompansatör")
        if sys.has_delay():
            raise ControlError("Gecikmeli kompansatör desteklenmez.")
        z, p, k = lti.zpk(sys).zpkdata(squeeze=True)
        out = cls(float(k), [], sys.Ts, sys if keep_pid else None)
        x0 = out.origin
        for roots, which in ((np.asarray(z, complex), "z"), (np.asarray(p, complex), "p")):
            scale = max([1.0] + [abs(r) for r in roots])
            for r in roots:
                if abs(r - x0) <= 1e-10 * scale:
                    out.elements.append(Element("differentiator" if which == "z" else "integrator",
                                                [complex(x0)] if which == "z" else [],
                                                [complex(x0)] if which == "p" else []))
                elif abs(r.imag) <= 1e-10 * max(1.0, abs(r)):
                    kind = "real_zero" if which == "z" else "real_pole"
                    v = complex(r.real, 0.0)
                    out.elements.append(Element(kind, [v] if which == "z" else [], [v] if which == "p" else []))
                elif r.imag > 0:
                    kind = "cplx_zero" if which == "z" else "cplx_pole"
                    out.elements.append(Element(kind, [r] if which == "z" else [], [r] if which == "p" else []))
        return out

    @classmethod
    def from_pid(cls, C) -> "Compensator":
        """PID nesnesinden (PID yapısı korunur)."""
        lti = lti_module()
        out = cls.from_lti(lti.tf(C), keep_pid=False)
        out.pid = C
        return out

    def to_domain(self, Ts: float, method: str = "zoh") -> tuple["Compensator", str]:
        """Kompansatörü yeni tasarım zamanına taşır (c2d / d2c / d2d, seçilen yöntemle; uygun değilse tustin).
        Dönüş (yeni kompansatör, not)."""
        if abs(Ts - self.Ts) <= 1e-12 * max(Ts, self.Ts, 1e-300):
            return self.copy(), ""
        note = ""
        sys = self.zpk()
        try:
            new = to_domain(sys, Ts, method)
        except ControlError:
            new = to_domain(sys, Ts, "tustin")
            note = f"kompansatör '{method}' ile dönüştürülemedi; 'tustin' kullanıldı"
        if new.has_delay():
            new = to_domain(sys, Ts, "tustin")
            note = "kompansatör dönüşümü gecikme üretti; 'tustin' kullanıldı"
        out = Compensator.from_lti(new)
        if self.pid is not None:
            note = (note + "; " if note else "") + ("PID yapısı yeni zamanda genel kompansatöre dönüştü "
                                                    "(yeniden ayarlayın)")
        return out, note

    def to_dict(self) -> dict:
        return {"K": self.K, "Ts": self.Ts, "elements": [e.to_dict() for e in self.elements]}

    @classmethod
    def from_dict(cls, d: dict) -> "Compensator":
        return cls(float(d.get("K", 1.0)), [Element.from_dict(e) for e in d.get("elements", [])],
                   float(d.get("Ts", 0.0)))


# ============================================================================ çevrimler
@dataclass
class LoopSet:
    """Kurulan çevrim sistemleri (hepsi tasarım zamanında, LTI)."""
    L: object                 # düzenlenen kompansatörün açık çevrimi (kazanç dahil)
    Gloop: object             # düzenlenen kompansatörün "gördüğü" bitki: L = C·Gloop
    T: object                 # r → y
    GS: object                # d (bitki girişi) → y
    KS: object                # r → u (bitki girişi)
    T2: object | None = None  # kademeli: iç kapalı çevrim ω_ref → ω
    errors: dict = field(default_factory=dict)   # kurulamayan bileşim → neden (ör. uygun olmayan r → u)


def _fb(a, b=1, sign: int = -1):
    """feedback; uygun olmayan (improper) SISO bileşimde durum uzayı yolu yerine tf formülüyle (MATLAB tf
    feedback'i gibi: num = n1·d2, den = d1·d2 + n1·n2)."""
    from spt.control.connect import feedback
    try:
        return feedback(a, b, sign)
    except ControlError:
        lti = lti_module()
        if not (isinstance(a, lti.LTI) and a.is_siso) or a.has_delay():
            raise
        bb = lti.tf(b) if isinstance(b, lti.LTI) else b
        if isinstance(bb, lti.LTI) and (not bb.is_siso or bb.has_delay()):
            raise
        return feedback(lti.tf(a), bb, sign)


def build_loops(arch: str, comps: dict, G, G1=None, H=None, F=None, edit: str = "C") -> LoopSet:
    """Mimariye göre çevrim sistemleri (`feedback` ve LTI aritmetiğiyle).

    comps: {"C": …} (tek çevrim türleri) ya da {"C1": dış, "C2": iç} (kademeli) — LTI kompansatörler. G: bitki
    (kademelide iç bitki G2), G1: kademelide dış bitki, H: algılayıcı (None = 1), F: ön filtre / ileri besleme.
    Açık çevrim L kurulamazsa ControlError; T / GS / KS kurulamazsa None ve `errors`'ta nedeni."""
    errors: dict = {}

    def safe(key, fn):
        try:
            return fn()
        except ControlError as exc:
            errors[key] = str(exc)
            return None

    if arch == "cascade":
        C1, C2 = comps["C1"], comps["C2"]
        G2 = G
        if G1 is None:
            raise ControlError("Kademeli yapıda dış bitki (G1) seçilmeli.")
        T2 = _fb(G2 * C2, 1)
        if edit == "C2":
            Gloop = G2
            L = C2 * G2
        else:
            Gloop = T2 * G1
            L = C1 * Gloop
        T = safe("T", lambda: _fb(C1 * T2 * G1, 1))
        GS = safe("GS", lambda: _fb(G1 * _fb(G2, C2), C1 * C2))
        KS = safe("KS", lambda: C1 * _fb(C2, G2 * (1 + C1 * G1)))
        return LoopSet(L, Gloop, T, GS, KS, T2, errors)
    C = comps["C"]
    Hs = 1 if H is None else H
    Gloop = G if H is None else H * G
    L = C * Gloop
    if arch == "feedforward":
        Ff = 0 if F is None else F
        T = safe("T", lambda: _fb(G, C * Hs) * (C + Ff))
        GS = safe("GS", lambda: _fb(G, C * Hs))
        KS = safe("KS", lambda: _fb(1, C * Hs * G) * (C + Ff))
    else:
        Fp = F if (arch == "prefilter" and F is not None) else 1
        T = safe("T", lambda: _fb(G * C, Hs) * Fp)
        GS = safe("GS", lambda: _fb(G, C * Hs))
        KS = safe("KS", lambda: _fb(C, G * Hs) * Fp)
    return LoopSet(L, Gloop, T, GS, KS, None, errors)


def closed_loop_poles(L) -> np.ndarray:
    """1 + L = 0 kökleri: pole(feedback(L, 1))."""
    from spt.control.analysis import pole
    return np.asarray(pole(_fb(L, 1)), complex)


def locus_system(comp: Compensator, Gloop) -> tuple[float, object]:
    """(işaret, L0): kök yer eğrisi L0 = işaret·C0·Gloop için |K| ile çizilir (K < 0 ise işaret −1)."""
    sign = -1.0 if comp.K < 0 else 1.0
    return sign, comp.normalized(sign) * Gloop


def loop_margins(L, Gloop=None, C=None) -> dict:
    """Çevrim payları: Gm, Wcg MATLAB `margin` tanımıyla (`spt.control.freqresp.margin`); faz payı ve kazanç geçiş
    frekansı `spt.control.classical.loop_margins` (ızgara + kök inceltme, allmargin'in en küçük faz payı) ile —
    iki tanım aynıdır; ayrık çevrimde margin'in polinom kökleri c = cos ωTs ≈ 1 çevresinde kümelendiğinde kaçan
    geçişler böylece kaybolmaz. `margin` geçiş bulduysa iki sonuç tutarlılık için karşılaştırılır."""
    from spt.control import classical
    from spt.control.freqresp import margin
    out = {"Gm": math.nan, "Wcg": math.nan, "Pm": math.nan, "Wcp": math.nan, "note": ""}
    gm, pm, wcg, wcp = (float(v) for v in margin(L)[:4])
    out.update(Gm=gm, Wcg=wcg, Pm=pm, Wcp=wcp)
    try:
        if Gloop is not None and C is not None:
            lm = classical.loop_margins(Gloop, C)
        else:
            lm = classical.loop_margins(L, ([1.0], [1.0], L.Ts))
    except (ControlError, ValueError, TypeError) as exc:
        out["note"] = f"faz payı doğrulanamadı: {exc}"
        return out
    # iki yöntemin bulduğu geçişlerin birleşiminde en küçük faz payı (ikisi de |L| = 1 noktalarında ölçer)
    cands = [(float(p), float(w)) for p, w in zip(lm["PhaseMargins"], lm["Crossovers"])]
    if math.isfinite(pm) and math.isfinite(wcp):
        cands.append((pm, wcp))
    if cands:
        pm2, wcp2 = min(cands)
        if not (math.isfinite(pm) and abs(pm - pm2) <= 1e-6 * max(1.0, abs(pm2))
                and abs(wcp - wcp2) <= 1e-6 * max(1.0, wcp2)):
            out["note"] = (f"margin: Pm = {pm:.6g}° ({wcp:.6g} rad/s); ızgara + kök inceltmeyle bulunan geçiş "
                           f"{pm2:.6g}° ({wcp2:.6g} rad/s) gösteriliyor")
            out.update(Pm=pm2, Wcp=wcp2)
    return out


def stable(sys) -> bool | None:
    from spt.control.analysis import isstable
    try:
        return bool(isstable(sys))
    except ControlError:
        return None


def damp_rows(poles, Ts: float) -> list[tuple[complex, float, float]]:
    """(kök, ζ, ωn) satırları (kütüphanenin damp tanımıyla)."""
    from spt.control.analysis import damp
    p = np.asarray(poles, complex).ravel()
    if p.size == 0:
        return []
    wn, zeta, pp = damp(p, Ts) if Ts > 0 else damp(p)
    return [(complex(a), float(b), float(c)) for a, b, c in zip(np.ravel(pp), np.ravel(zeta), np.ravel(wn))]
