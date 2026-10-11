"""Klasik kontrolcü tasarımı — MATLAB Control System Toolbox karşılıkları.

- PID nesneleri: `pid(Kp, Ki, Kd, Tf, Ts)` (paralel), `pidstd(Kp, Ti, Td, N, Ts)` (standart), 2 serbestlik dereceli
  `pid2` / `pidstd2` (b, c ağırlıkları); `IFormula` / `DFormula` ('ForwardEuler' Ts/(z−1), 'BackwardEuler'
  Ts·z/(z−1), 'Trapezoidal' (Ts/2)(z+1)/(z−1)). Nesneler spt.control.lti.TransferFunction'dır: tüm analiz
  fonksiyonları (step, bode, feedback …) doğrudan kullanır; `tf(C)` MATLAB `tf(pid(...))` ile birebir aynı katsayılar.
  Paralel ↔ standart dönüşüm: `pid(C_std)`, `pidstd(C_par)`.
- `pidtune(G, tür, wc, PhaseMargin=60, DesignFocus=...)` → `(C, info)`. Yöntem (MATLAB'in yayımlanmamış
  algoritması yerine açık tanım): hedef geçiş frekansı wc'de |L(jwc)| = 1 ve ∠L(jwc) = −180° + PM **tam** tutturulur
  (kapalı biçimli çözüm). Ek serbestlikler: PID / PIDF'de Ti = α·Td (α: 'balanced' 4, 'reference-tracking' 8,
  'disturbance-rejection' 2); PDF / PIDF'de türev filtresi kutbu β·wc'de (β = 10; ayrık ileri Euler'de
  Tf ≥ Ts). wc verilmezse `_default_wc` kuralı (belgede). Kapalı çevrim kararlılığı kutuplarla (sürekli zamanda
  gecikme varsa Nyquist sarım sayısıyla) denetlenir.
- Öncül / ardıl (`leadlag`): istenen faz payı + geçiş frekansı ve / veya kalıcı durum hatası; tam çözüm (Phillips).
  `lead`, `lag` el ile; çentik filtresi (`notch`: derinlik ve genişlik); kök yer eğrisinde nokta için kazanç
  (`rlocfind`), istenen sönüm / doğal frekans için kazanç (`gain_for_damping`).
"""
from __future__ import annotations

import cmath
import math
import warnings
from typing import Any

import numpy as np
from scipy.optimize import brentq

from spt.control.errors import ControlError, ControlWarning

try:  # PID nesneleri LTI transfer fonksiyonudur (çekirdek yoksa yalın taban)
    from spt.control.lti import TransferFunction as _TFBase
    _HAVE_LTI = True
except ImportError:  # pragma: no cover - LTI çekirdeği olmadan da içe aktarılabilsin
    _HAVE_LTI = False

    class _TFBase:  # type: ignore[no-redef]
        """LTI çekirdeği yokken en küçük transfer fonksiyonu tabanı."""

        def __init__(self, num, den, Ts=0.0, **_kw):
            self._num = [[np.asarray(p, float) for p in row] for row in num]
            self._den = [[np.asarray(p, float) for p in row] for row in den]
            self.Ts = float(Ts)

        @property
        def num(self):
            return [[p.copy() for p in row] for row in self._num]

        @property
        def den(self):
            return [[p.copy() for p in row] for row in self._den]

        @property
        def is_discrete(self):
            return self.Ts > 0

__all__ = ["PID", "PIDStd", "PID2", "PIDStd2", "pid", "pidstd", "pid2", "pidstd2", "pidtune", "lead", "lag",
           "leadlag", "notch", "rlocfind", "gain_for_damping", "loop_margins"]

_FORMULAS = ("ForwardEuler", "BackwardEuler", "Trapezoidal")
_TYPES = ("P", "I", "PI", "PD", "PDF", "PID", "PIDF")
_FOCUS_ALPHA = {"balanced": 4.0, "reference-tracking": 8.0, "disturbance-rejection": 2.0}
_FILTER_BETA = 10.0


# ---------------------------------------------------------------------------------------------------------------
# Polinom yardımcıları
# ---------------------------------------------------------------------------------------------------------------
def _trim(p) -> np.ndarray:
    p = np.atleast_1d(np.asarray(p, dtype=float)).ravel()
    nz = np.flatnonzero(p != 0)
    return p[nz[0]:] if nz.size else np.zeros(1)


def _padd(a, b) -> np.ndarray:
    n = max(a.size, b.size)
    return _trim(np.concatenate([np.zeros(n - a.size), a]) + np.concatenate([np.zeros(n - b.size), b]))


def _pmul(a, b) -> np.ndarray:
    return _trim(np.convolve(a, b))


def _formula(name: str, what: str) -> str:
    for f in _FORMULAS:
        if str(name).lower() == f.lower():
            return f
    raise ControlError(f"{what} 'ForwardEuler', 'BackwardEuler' ya da 'Trapezoidal' olmalı.")


def _int_poly(Ts: float, formula: str) -> tuple[np.ndarray, np.ndarray]:
    """Integratör IF(s|z) = n/d."""
    if Ts == 0:
        return np.array([1.0]), np.array([1.0, 0.0])
    if formula == "ForwardEuler":
        return np.array([Ts]), np.array([1.0, -1.0])
    if formula == "BackwardEuler":
        return np.array([Ts, 0.0]), np.array([1.0, -1.0])
    return np.array([Ts / 2, Ts / 2]), np.array([1.0, -1.0])


def _int_eval(x: complex | np.ndarray, Ts: float, formula: str):
    """IF değeri: sürekli x = s, ayrık x = z."""
    if Ts == 0:
        return 1.0 / x
    if formula == "ForwardEuler":
        return Ts / (x - 1.0)
    if formula == "BackwardEuler":
        return Ts * x / (x - 1.0)
    return (Ts / 2) * (x + 1.0) / (x - 1.0)


# ---------------------------------------------------------------------------------------------------------------
# PID nesneleri
# ---------------------------------------------------------------------------------------------------------------
def _real(x, what: str) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ControlError(f"{what} gerçel sayı olmalı.") from None
    if math.isnan(v):
        raise ControlError(f"{what} sayı olmalı (NaN değil).")
    return v


def _check_pid(Kp, Ki, Kd, Tf, Ts, IF, DF) -> None:
    for v, w in ((Kp, "Kp"), (Ki, "Ki"), (Kd, "Kd")):
        if not math.isfinite(v):
            raise ControlError(f"{w} sonlu olmalı.")
    if not math.isfinite(Tf) or Tf < 0:
        raise ControlError("Türev filtresi zaman sabiti Tf sonlu ve negatif olmayan bir sayı olmalı.")
    if not math.isfinite(Ts) or Ts < 0:
        raise ControlError("Örnekleme zamanı Ts ≥ 0 olmalı (0 = sürekli).")
    if Ts > 0 and Kd != 0:
        if DF == "ForwardEuler" and 0 < Tf <= Ts / 2:
            raise ControlError("İleri Euler türev formülünde türev filtresinin kararlı olması için Ts < 2·Tf olmalı "
                               f"(Tf = {Tf:g}, Ts = {Ts:g}). Tf'yi büyütün ya da başka bir formül seçin.")
        if DF == "Trapezoidal" and Tf == 0:
            raise ControlError("Yamuk (Trapezoidal) türev formülü Tf = 0 ile kullanılamaz (z = −1'de kutup); Tf > 0 "
                               "verin ya da geri Euler seçin.")


class _PIDBase(_TFBase):
    """PID nesnelerinin ortak tabanı: parametreler paralel biçimde (Kp, Ki, Kd, Tf, b, c) saklanır; transfer
    fonksiyonu her değişiklikte yeniden kurulur."""

    _two_dof = False
    _form = "parallel"
    _kind = "tf"

    def __init__(self, Kp, Ki, Kd, Tf, Ts, IFormula, DFormula, b=1.0, c=1.0, name=""):
        object.__setattr__(self, "_ready", False)
        self._p = dict(Kp=_real(Kp, "Kp"), Ki=_real(Ki, "Ki"), Kd=_real(Kd, "Kd"), Tf=_real(Tf, "Tf"),
                       b=_real(b, "b"), c=_real(c, "c"))
        self._IF = _formula(IFormula, "IFormula")
        self._DF = _formula(DFormula, "DFormula")
        Ts = _real(Ts, "Ts")
        _check_pid(self._p["Kp"], self._p["Ki"], self._p["Kd"], self._p["Tf"], Ts, self._IF, self._DF)
        num, den = self._polys(Ts)
        nu = 2 if self._two_dof else 1
        names = ("r", "y") if self._two_dof else None
        if _HAVE_LTI:
            super().__init__(num, den, Ts, input_names=names, output_names=("u",), name=name)
        else:
            super().__init__(num, den, Ts)
        self._nu_pid = nu
        object.__setattr__(self, "_ready", True)

    # --- parametreler ---------------------------------------------------------------------------------------
    def _set(self, **kw) -> None:
        p = dict(self._p)
        p.update({k: _real(v, k) for k, v in kw.items()})
        _check_pid(p["Kp"], p["Ki"], p["Kd"], p["Tf"], self.Ts, self._IF, self._DF)
        self._p = p
        self._num, self._den = self._polys(self.Ts)

    @property
    def IFormula(self) -> str:
        return self._IF

    @IFormula.setter
    def IFormula(self, v) -> None:
        old = self._IF
        self._IF = _formula(v, "IFormula")
        try:
            self._set()
        except ControlError:
            self._IF = old
            raise

    @property
    def DFormula(self) -> str:
        return self._DF

    @DFormula.setter
    def DFormula(self, v) -> None:
        old = self._DF
        self._DF = _formula(v, "DFormula")
        try:
            self._set()
        except ControlError:
            self._DF = old
            raise

    def parallel(self) -> tuple[float, float, float, float]:
        """(Kp, Ki, Kd, Tf) — paralel biçim."""
        p = self._p
        return p["Kp"], p["Ki"], p["Kd"], p["Tf"]

    def standard(self) -> tuple[float, float, float, float]:
        """(Kp, Ti, Td, N) — standart biçim (Ti = Kp/Ki, Td = Kd/Kp, N = Td/Tf). Kp = 0 iken (I / D terimi
        varsa) standart biçim yoktur → ControlError."""
        return _par2std(*self.parallel())

    @property
    def Ts(self) -> float:  # noqa: D401 - LTI özniteliği
        return self.__dict__.get("_Ts", 0.0)

    @Ts.setter
    def Ts(self, v) -> None:
        if self.__dict__.get("_ready"):
            raise ControlError("PID nesnesinin örnekleme zamanı sonradan değiştirilemez; yeni nesne oluşturun.")
        self.__dict__["_Ts"] = float(v)

    # --- transfer fonksiyonu --------------------------------------------------------------------------------
    def _branch(self, Ts: float, wp: float, wd: float) -> tuple[np.ndarray, np.ndarray]:
        """wp·Kp + Ki·IF + wd·Kd/(Tf + DF) → (pay, payda); payda baş katsayısı 1 (MATLAB tf(pid) biçimi)."""
        p = self._p
        num, den = np.array([wp * p["Kp"]]), np.array([1.0])
        if p["Ki"] != 0:
            n_i, d_i = _int_poly(Ts, self._IF)
            num, den = _padd(_pmul(num, d_i), _pmul(den, p["Ki"] * n_i)), _pmul(den, d_i)
        if p["Kd"] != 0:
            n_d, d_d = _int_poly(Ts, self._DF)       # DF = n_d/d_d → Kd/(Tf + DF) = Kd·d_d/(Tf·d_d + n_d)
            dd = _padd(p["Tf"] * d_d, n_d)
            num, den = _padd(_pmul(num, dd), _pmul(den, wd * p["Kd"] * d_d)), _pmul(den, dd)
        num, den = _trim(num), _trim(den)
        lead = den[0]
        num, den = num / lead, den / lead
        if np.all(num == 0):
            return np.zeros(1), np.ones(1)
        return num, den

    def _polys(self, Ts: float):
        p = self._p
        if self._two_dof:
            nr, dr = self._branch(Ts, p["b"], p["c"])
            ny_, dy = self._branch(Ts, 1.0, 1.0)
            return [[nr, -ny_ if np.any(ny_) else ny_]], [[dr, dy]]
        n, d = self._branch(Ts, 1.0, 1.0)
        return [[n]], [[d]]

    def evalfr(self, x) -> np.ndarray:
        """C(x) doğrudan formülden (sürekli x = s, ayrık x = z); 2-DOF'ta (r, −y) dalları sütunlarda."""
        x = np.asarray(x, dtype=complex)
        p = self._p
        Ts = self.Ts

        def br(wp, wd):
            v = wp * p["Kp"] + p["Ki"] * _int_eval(x, Ts, self._IF) if p["Ki"] != 0 else wp * p["Kp"] + 0 * x
            if p["Kd"] != 0:
                v = v + wd * p["Kd"] / (p["Tf"] + _int_eval(x, Ts, self._DF))
            return v
        if self._two_dof:
            return np.stack([br(p["b"], p["c"]), -br(1.0, 1.0)], axis=-1)
        return br(1.0, 1.0)

    def to_tf(self):
        """Transfer fonksiyonu (spt.control.lti.TransferFunction kopyası)."""
        if _HAVE_LTI:
            from spt.control.lti import TransferFunction
            return TransferFunction(self.num, self.den, self.Ts, input_names=self.input_names,
                                    output_names=self.output_names)
        return self

    def to_ss(self):
        """Durum uzayı gerçeklemesi (uygun olmayan PID — Tf = 0 ve Kd ≠ 0 — için ControlError)."""
        if any(n.size > d.size for rn, rd in zip(self._num, self._den) for n, d in zip(rn, rd)):
            raise ControlError("Türev filtresi olmayan (Tf = 0) PID uygun değil (pay derecesi > payda derecesi); durum "
                               "uzayına çevrilemez. Tf > 0 verin.")
        from spt.control import lti
        return lti.ss(self.to_tf())

    def copy(self):
        if _HAVE_LTI:
            return self.to_tf()
        return self

    if _HAVE_LTI:
        # Aritmetik: PID önce düz transfer fonksiyonuna çevrilir (MATLAB'de de pid ∘ tf → tf)
        def __add__(self, o):
            return self.to_tf() + _plain(o)

        def __radd__(self, o):
            return _plain(o) + self.to_tf()

        def __sub__(self, o):
            return self.to_tf() - _plain(o)

        def __rsub__(self, o):
            return _plain(o) - self.to_tf()

        def __mul__(self, o):
            return self.to_tf() * _plain(o)

        def __rmul__(self, o):
            return _plain(o) * self.to_tf()

        __matmul__ = __mul__
        __rmatmul__ = __rmul__

        def __truediv__(self, o):
            return self.to_tf() / _plain(o)

        def __rtruediv__(self, o):
            return _plain(o) / self.to_tf()

        def __neg__(self):
            return -self.to_tf()

        def __pow__(self, k):
            return self.to_tf() ** k

        def __getitem__(self, key):
            return self.to_tf()[key]

    def _pid_str(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        Ts = self.Ts
        lines = [self._pid_str()]
        if Ts > 0:
            lines.append(f"IFormula: {self._IF}, DFormula: {self._DF}; örnekleme zamanı {Ts:g} s")
            lines.append("Ayrık zamanlı PID denetleyici" + (" (2 serbestlik dereceli)." if self._two_dof else "."))
        else:
            lines.append("Sürekli zamanlı PID denetleyici" + (" (2 serbestlik dereceli)." if self._two_dof else "."))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()


def _plain(o):
    return o.to_tf() if isinstance(o, _PIDBase) else o


def _par2std(Kp, Ki, Kd, Tf):
    if Kp == 0:
        if Ki == 0 and Kd == 0:
            return 0.0, math.inf, 0.0, math.inf
        raise ControlError("Kp = 0 iken (integral ya da türev terimi varken) standart biçim tanımsız.")
    Ti = math.inf if Ki == 0 else Kp / Ki
    Td = Kd / Kp
    N = math.inf if (Kd == 0 or Tf == 0) else Td / Tf
    if Ti < 0 or Td < 0 or N < 0:
        raise ControlError("Kp, Ki ve Kd aynı işaretli olmalı; aksi hâlde standart biçimde Ti ve Td negatif çıkar "
                           f"(Ti = {Ti:g}, Td = {Td:g}).")
    return Kp, Ti, Td, N


def _std2par(Kp, Ti, Td, N):
    for v, w in ((Kp, "Kp"),):
        if not math.isfinite(v):
            raise ControlError(f"{w} sonlu olmalı.")
    if not Ti > 0:
        raise ControlError("Ti pozitif olmalı (integral yoksa Ti = inf).")
    if not (Td >= 0 and math.isfinite(Td)):
        raise ControlError("Td sonlu ve negatif olmayan bir sayı olmalı.")
    if not N > 0:
        raise ControlError("N pozitif olmalı (filtre yoksa N = inf).")
    Ki = 0.0 if math.isinf(Ti) else Kp / Ti
    Kd = Kp * Td
    Tf = 0.0 if (math.isinf(N) or Td == 0) else Td / N
    return Kp, Ki, Kd, Tf


class PID(_PIDBase):
    """Paralel PID (MATLAB `pid`): C = Kp + Ki·IF + Kd/(Tf + DF); sürekli: Kp + Ki/s + Kd·s/(Tf·s + 1)."""

    Kp = property(lambda self: self._p["Kp"], lambda self, v: self._set(Kp=v))
    Ki = property(lambda self: self._p["Ki"], lambda self, v: self._set(Ki=v))
    Kd = property(lambda self: self._p["Kd"], lambda self, v: self._set(Kd=v))
    Tf = property(lambda self: self._p["Tf"], lambda self, v: self._set(Tf=v))

    def _pid_str(self) -> str:
        p = self._p
        if self.Ts > 0:
            return (f"C = Kp + Ki·IF(z) + Kd / (Tf + DF(z)):  Kp = {p['Kp']:.6g}, Ki = {p['Ki']:.6g}, "
                    f"Kd = {p['Kd']:.6g}, Tf = {p['Tf']:.6g}")
        return (f"C = Kp + Ki/s + Kd·s/(Tf·s + 1):  Kp = {p['Kp']:.6g}, Ki = {p['Ki']:.6g}, Kd = {p['Kd']:.6g}, "
                f"Tf = {p['Tf']:.6g}")


class PIDStd(_PIDBase):
    """Standart PID (MATLAB `pidstd`): C = Kp(1 + IF/Ti + Td/(Td/N + DF)); sürekli:
    Kp(1 + 1/(Ti·s) + Td·s/((Td/N)·s + 1))."""

    _form = "standard"

    def _sstd(self, **kw):
        Kp, Ti, Td, N = self.standard()
        d = dict(Kp=Kp, Ti=Ti, Td=Td, N=N)
        d.update({k: _real(v, k) for k, v in kw.items()})
        Kp, Ki, Kd, Tf = _std2par(d["Kp"], d["Ti"], d["Td"], d["N"])
        self._std = (d["Kp"], d["Ti"], d["Td"], d["N"])
        self._set(Kp=Kp, Ki=Ki, Kd=Kd, Tf=Tf)

    def standard(self):
        return getattr(self, "_std", None) or _par2std(*self.parallel())

    Kp = property(lambda self: self.standard()[0], lambda self, v: self._sstd(Kp=v))
    Ti = property(lambda self: self.standard()[1], lambda self, v: self._sstd(Ti=v))
    Td = property(lambda self: self.standard()[2], lambda self, v: self._sstd(Td=v))
    N = property(lambda self: self.standard()[3], lambda self, v: self._sstd(N=v))

    def _pid_str(self) -> str:
        Kp, Ti, Td, N = self.standard()
        if self.Ts > 0:
            return (f"C = Kp·(1 + IF(z)/Ti + Td / (Td/N + DF(z))):  Kp = {Kp:.6g}, Ti = {Ti:.6g}, Td = {Td:.6g}, "
                    f"N = {N:.6g}")
        return f"C = Kp·(1 + 1/(Ti·s) + Td·s/((Td/N)·s + 1)):  Kp = {Kp:.6g}, Ti = {Ti:.6g}, Td = {Td:.6g}, N = {N:.6g}"


class PID2(PID):
    """2 serbestlik dereceli paralel PID (MATLAB `pid2`):
    u = Kp(b·r − y) + Ki·IF·(r − y) + Kd/(Tf + DF)·(c·r − y). Transfer fonksiyonu 1×2: [Cr, −Cy] (girişler r, y)."""

    _two_dof = True
    b = property(lambda self: self._p["b"], lambda self, v: self._set(b=v))
    c = property(lambda self: self._p["c"], lambda self, v: self._set(c=v))

    def _pid_str(self) -> str:
        return super()._pid_str() + f", b = {self._p['b']:.6g}, c = {self._p['c']:.6g}"


class PIDStd2(PIDStd):
    """2 serbestlik dereceli standart PID (MATLAB `pidstd2`): u = Kp[(b·r − y) + IF/Ti·(r − y) +
    Td/(Td/N + DF)·(c·r − y)]."""

    _two_dof = True
    b = property(lambda self: self._p["b"], lambda self, v: self._set(b=v))
    c = property(lambda self: self._p["c"], lambda self, v: self._set(c=v))

    def _pid_str(self) -> str:
        return super()._pid_str() + f", b = {self._p['b']:.6g}, c = {self._p['c']:.6g}"


def _opts(Ts, IFormula, DFormula, kw):
    IF = kw.pop("IFormula", IFormula)
    DF = kw.pop("DFormula", DFormula)
    if kw:
        raise ControlError(f"Bilinmeyen seçenek: {', '.join(kw)}")
    return _real(Ts, "Ts"), IF, DF


def pid(Kp: Any = 0.0, Ki: float = 0.0, Kd: float = 0.0, Tf: float = 0.0, Ts: float = 0.0,
        IFormula: str = "ForwardEuler", DFormula: str = "ForwardEuler", **kw) -> PID:
    """Paralel PID (MATLAB `pid`). `pid(C)`: başka bir PID nesnesini paralel biçime çevirir (aynı Ts ve
    formüllerle; 2-DOF nesnede b, c atılır — MATLAB'de bu dönüşüm `getComponents`'tir)."""
    if isinstance(Kp, _PIDBase):
        C = Kp
        return PID(*C.parallel(), C.Ts, C.IFormula, C.DFormula)
    Ts, IF, DF = _opts(Ts, IFormula, DFormula, kw)
    return PID(Kp, Ki, Kd, Tf, Ts, IF, DF)


def pidstd(Kp: Any = 1.0, Ti: float = math.inf, Td: float = 0.0, N: float = math.inf, Ts: float = 0.0,
           IFormula: str = "ForwardEuler", DFormula: str = "ForwardEuler", **kw) -> PIDStd:
    """Standart PID (MATLAB `pidstd`). `pidstd(C)`: paralel PID'i standart biçime çevirir (Ti = Kp/Ki,
    Td = Kd/Kp, N = Kd/(Kp·Tf)); Kp = 0 ya da karışık işaretli kazançlarda ControlError."""
    if isinstance(Kp, _PIDBase):
        C = Kp
        std = _par2std(*C.parallel())
        out = PIDStd(*_std2par(*std), C.Ts, C.IFormula, C.DFormula)
        out._std = std
        return out
    Ts, IF, DF = _opts(Ts, IFormula, DFormula, kw)
    Kp, Ti, Td, N = (_real(v, w) for v, w in ((Kp, "Kp"), (Ti, "Ti"), (Td, "Td"), (N, "N")))
    out = PIDStd(*_std2par(Kp, Ti, Td, N), Ts, IF, DF)
    out._std = (Kp, Ti, Td, N)
    return out


def pid2(Kp: Any = 0.0, Ki: float = 0.0, Kd: float = 0.0, Tf: float = 0.0, b: float = 1.0, c: float = 1.0,
         Ts: float = 0.0, IFormula: str = "ForwardEuler", DFormula: str = "ForwardEuler", **kw) -> PID2:
    """2 serbestlik dereceli paralel PID (MATLAB `pid2(Kp, Ki, Kd, Tf, b, c, Ts)`)."""
    if isinstance(Kp, _PIDBase):
        C = Kp
        p = C._p
        return PID2(*C.parallel(), C.Ts, C.IFormula, C.DFormula, p["b"], p["c"])
    Ts, IF, DF = _opts(Ts, IFormula, DFormula, kw)
    return PID2(Kp, Ki, Kd, Tf, Ts, IF, DF, b, c)


def pidstd2(Kp: Any = 1.0, Ti: float = math.inf, Td: float = 0.0, N: float = math.inf, b: float = 1.0,
            c: float = 1.0, Ts: float = 0.0, IFormula: str = "ForwardEuler", DFormula: str = "ForwardEuler",
            **kw) -> PIDStd2:
    """2 serbestlik dereceli standart PID (MATLAB `pidstd2(Kp, Ti, Td, N, b, c, Ts)`)."""
    if isinstance(Kp, _PIDBase):
        C = Kp
        p = C._p
        std = _par2std(*C.parallel())
        out = PIDStd2(*_std2par(*std), C.Ts, C.IFormula, C.DFormula, p["b"], p["c"])
        out._std = std
        return out
    Ts, IF, DF = _opts(Ts, IFormula, DFormula, kw)
    Kp, Ti, Td, N = (_real(v, w) for v, w in ((Kp, "Kp"), (Ti, "Ti"), (Td, "Td"), (N, "N")))
    out = PIDStd2(*_std2par(Kp, Ti, Td, N), Ts, IF, DF, b, c)
    out._std = (Kp, Ti, Td, N)
    return out


# ---------------------------------------------------------------------------------------------------------------
# SISO bitki verisi (ördek tipleme)
# ---------------------------------------------------------------------------------------------------------------
class _Siso:
    """SISO bitki: pay / payda polinomları (kutuplar, karakteristik polinom için), frekans cevabı, gecikme."""

    def __init__(self, sys: Any, what: str = "Bitki"):
        self.ss = None
        if isinstance(sys, tuple) and len(sys) in (2, 3) and not hasattr(sys, "A"):
            num, den = sys[0], sys[1]
            Ts = float(sys[2]) if len(sys) == 3 else 0.0
            delay = 0.0
        elif isinstance(sys, tuple) and len(sys) in (4, 5):
            from spt.control.modern import ss_data
            A, B, C, D, Ts = ss_data(sys, what)
            num, den, delay = None, None, 0.0
            self.ss = (A, B, C, D)
        else:
            Ts = getattr(sys, "Ts", None)
            Ts = 0.0 if Ts in (None, False) else float(Ts)
            tot = getattr(sys, "total_delay", None)
            if callable(tot):
                td = np.asarray(tot(), dtype=float)
                delay = float(td.ravel()[0]) if td.size == 1 else 0.0
            else:
                delay = 0.0
                for a in ("input_delay", "output_delay", "io_delay"):
                    v = getattr(sys, a, None)
                    if v is not None and np.size(v):
                        delay += float(np.asarray(v, dtype=float).ravel()[0])
            if all(hasattr(sys, a) for a in ("A", "B", "C", "D")):
                A = np.atleast_2d(np.asarray(sys.A, float))
                n = A.shape[0] if A.size else 0
                B = np.asarray(sys.B, float).reshape(n, -1) if n else np.zeros((0, 1))
                C = np.asarray(sys.C, float).reshape(-1, n) if n else np.zeros((1, 0))
                D = np.atleast_2d(np.asarray(sys.D, float))
                self.ss = (A if n else np.zeros((0, 0)), B, C, D)
                num = den = None
            elif hasattr(sys, "num") and hasattr(sys, "den"):
                num, den = sys.num, sys.den
                while isinstance(num, (list, tuple)):
                    if len(num) != 1:
                        raise ControlError(f"{what} tek girişli tek çıkışlı (SISO) olmalı.")
                    num = num[0]
                while isinstance(den, (list, tuple)):
                    den = den[0]
            else:
                try:
                    from spt.control import lti
                    t = lti.tf(sys)
                    num, den = t.num[0][0], t.den[0][0]
                    if t.shape != (1, 1):
                        raise ControlError(f"{what} tek girişli tek çıkışlı (SISO) olmalı.")
                except ControlError:
                    raise
                except Exception:  # noqa: BLE001
                    raise ControlError(f"{what} tanınmadı: transfer fonksiyonu ya da durum uzayı "
                                       "bekleniyordu.") from None
        if self.ss is not None:
            A, B, C, D = self.ss
            if D.shape != (1, 1) or B.shape[1] != 1 or C.shape[0] != 1:
                raise ControlError(f"{what} tek girişli tek çıkışlı (SISO) olmalı.")
            from scipy.signal import ss2tf
            if A.shape[0]:
                n_, d_ = ss2tf(A, B, C, D)
                num, den = n_[0], d_
            else:
                num, den = D[0], np.ones(1)
        self.num = _trim(num)
        self.den = _trim(den)
        if np.all(self.den == 0):
            raise ControlError(f"{what}: payda sıfır olamaz.")
        if Ts < 0:
            raise ControlError("Belirtilmemiş örnekleme zamanlı (Ts = −1) model desteklenmez.")
        self.Ts = Ts
        if Ts > 0 and abs(delay - round(delay)) > 1e-9:
            raise ControlError("Ayrık modelde gecikme tamsayı örnek olmalı.")
        self.delay = float(round(delay)) if Ts > 0 else float(delay)
        if np.all(self.num == 0):
            raise ControlError(f"{what} sıfır transfer fonksiyonu; tasarım yapılamaz.")

    @property
    def discrete(self) -> bool:
        return self.Ts > 0

    def point(self, w):
        """Frekans noktası: sürekli jw, ayrık e^{jwTs}."""
        w = np.asarray(w, dtype=float)
        return np.exp(1j * w * self.Ts) if self.discrete else 1j * w

    def at(self, x) -> np.ndarray:
        """G(x) (x: s ya da z; gecikme dahil)."""
        x = np.asarray(x, dtype=complex)
        if self.ss is not None and self.ss[0].shape[0]:
            A, B, C, D = self.ss
            n = A.shape[0]
            flat = x.ravel()
            try:
                M = flat[:, None, None] * np.eye(n)[None] - A[None]
                X = np.linalg.solve(M, np.broadcast_to(B.astype(complex), (flat.size, n, 1)))
                out = (C[None] @ X)[:, 0, 0] + D[0, 0]
            except np.linalg.LinAlgError:        # tekil nokta (kutup üzerinde): tek tek
                out = np.empty(flat.shape, dtype=complex)
                for i, xi in enumerate(flat):
                    try:
                        out[i] = (C @ np.linalg.solve(xi * np.eye(n) - A, B))[0, 0] + D[0, 0]
                    except np.linalg.LinAlgError:
                        out[i] = np.inf
            g = out.reshape(x.shape)
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                g = np.polyval(self.num, x) / np.polyval(self.den, x)
        if self.delay:
            g = g * (x ** (-self.delay) if self.discrete else np.exp(-x * self.delay))
        return g

    def freq(self, w) -> np.ndarray:
        return self.at(self.point(w))

    def pz_scale(self) -> float:
        """Kutup / sıfırların (sıfır olmayan) büyüklüklerinin geometrik ortası (sürekli eşdeğerde, rad/s)."""
        r = np.concatenate([np.roots(self.num) if self.num.size > 1 else np.zeros(0),
                            np.roots(self.den) if self.den.size > 1 else np.zeros(0)])
        if self.discrete:
            r = r[np.abs(r) > 1e-12]
            r = np.log(r.astype(complex)) / self.Ts
        mags = np.abs(r)
        mags = mags[mags > 1e-9]
        if mags.size == 0:
            return 1.0 if not self.discrete else 0.1 * math.pi / self.Ts
        return float(np.exp(np.mean(np.log(mags))))


# ---------------------------------------------------------------------------------------------------------------
# Kapalı çevrim kararlılığı
# ---------------------------------------------------------------------------------------------------------------
def _cl_stable(G: _Siso, nC: np.ndarray, dC: np.ndarray) -> bool:
    """1 + C·G = 0 kapalı çevrimi kararlı mı? Gecikmesiz ya da ayrık: karakteristik polinomun kökleri (iç
    kararlılık: sadeleşen kipler dahil). Sürekli + gecikme: Nyquist sarım sayısı."""
    nG, dG = G.num, G.den
    if G.discrete or G.delay == 0:
        if G.discrete and G.delay:
            dG = np.concatenate([dG, np.zeros(int(G.delay))])
        char = _padd(_pmul(dC, dG), _pmul(nC, nG))
        if char.size <= 1:
            return char[0] != 0
        r = np.roots(char)
        if G.discrete:
            return bool(np.all(np.abs(r) < 1.0 - 1e-10))
        return bool(np.all(r.real < -1e-10 * np.maximum(1.0, np.abs(r))))
    return _nyquist_stable(G, nC, dC)


def _nyquist_stable(G: _Siso, nC, dC) -> bool:
    """Sürekli zaman, gecikmeli çevrim: Z = P − N (N: −1'in saat yönü tersine sarım sayısı, D-çevresi sağdan
    girintili). Rasyonel kısım uygun değilse (ileri tip) ya da çift uygun ve |L(∞)| ≥ 1 ise (nötr tip, sağ yarı
    düzlemde sonsuz kök) kararsızdır; |L(∞)| < 1 ise sonsuzdaki yay ve kesilen eksen parçası sarıma katkı
    vermez (|L| < 1 bölgesinde 1 + L sağ yarı düzlemde kalır)."""
    num = _pmul(nC, G.num)
    den = _pmul(dC, G.den)
    if num.size > den.size:
        return False
    Linf = num[0] / den[0] if num.size == den.size else 0.0
    if abs(Linf) >= 1.0:
        return False
    rho = 0.5 * (1.0 + abs(Linf))
    ol = np.roots(den) if den.size > 1 else np.zeros(0)
    zs = np.roots(num) if num.size > 1 else np.zeros(0)
    scale = max([1.0] + [abs(v) for v in np.concatenate([ol, zs])])
    tol = 1e-8 * scale
    imag_poles = sorted({round(float(abs(p.imag)), 12) for p in ol if abs(p.real) <= tol})
    P = int(np.sum(ol.real > tol))
    for p in ol[ol.real > tol]:      # sağ yarı düzlemde sadeleşen kip: iç kararsızlık
        if zs.size and np.min(np.abs(zs - p)) <= 1e-8 * max(1.0, abs(p)):
            return False

    def Lf(s):
        s = np.asarray(s, dtype=complex)
        return np.polyval(num, s) / np.polyval(den, s) * np.exp(-s * G.delay)

    # üst sınır: |L| < ρ < 1 kalıcı olarak (rasyonel kısım |L(∞)|'a yakınsar)
    W = 10.0 * scale
    for _ in range(60):
        ww = np.geomspace(W, 100 * W, 200)
        if np.max(np.abs(np.polyval(num, 1j * ww) / np.polyval(den, 1j * ww))) < rho:
            break
        W *= 10
    eps = 1e-6 * min([scale] + [0.1 * abs(a - b) for a in imag_poles for b in imag_poles if a != b])
    # yol: (0'daki girinti θ: 0→π/2) + pozitif eksen (girintilerle)
    pts = []
    if imag_poles and imag_poles[0] == 0.0:
        th = np.linspace(0.0, math.pi / 2, 200)
        pts.append(eps * np.exp(1j * th))
        start = eps
    else:
        start = 0.0
    marks = [w for w in imag_poles if w > 0]
    base = np.unique(np.concatenate([np.geomspace(max(start, 1e-9 * scale), W, 4000),
                                     np.linspace(start, W, min(200000, int(W * G.delay * 8 / math.pi) + 2000))]))
    for p in ol:   # sönümü düşük kutupların çevresinde sık ızgara
        if abs(p.real) > tol and 0 < abs(p.imag) < W:
            wd = abs(p.imag)
            base = np.concatenate([base, wd + np.linspace(-20, 20, 81) * max(abs(p.real), 1e-6 * wd)])
    base = np.unique(base[(base >= start) & (base <= W)])
    seg_start = start
    for wk in marks + [None]:
        if wk is None:
            seg = base[base >= seg_start]
            pts.append(1j * seg)
            break
        seg = base[(base >= seg_start) & (base <= wk - eps)]
        pts.append(1j * np.concatenate([seg, [wk - eps]]))
        th = np.linspace(-math.pi / 2, math.pi / 2, 200)
        pts.append(1j * wk + eps * np.exp(1j * th))
        seg_start = wk + eps
    path = np.concatenate(pts)
    f = 1.0 + Lf(path)
    total = 0.0
    for _ in range(25):   # uyarlamalı inceltme: ardışık açı farkı < 0.5 rad
        d = np.angle(f[1:] / f[:-1])
        bad = np.flatnonzero(np.abs(d) > 0.5)
        if bad.size == 0:
            total = float(np.sum(d))
            break
        mid = 0.5 * (path[bad] + path[bad + 1])
        path = np.insert(path, bad + 1, mid)
        f = np.insert(f, bad + 1, 1.0 + Lf(mid))
    else:
        total = float(np.sum(np.angle(f[1:] / f[:-1])))
    N_ccw = int(round(2 * total / (2 * math.pi)))
    Z = P - N_ccw
    return Z == 0


# ---------------------------------------------------------------------------------------------------------------
# Çevrim payları (tasarım doğrulaması için)
# ---------------------------------------------------------------------------------------------------------------
def _wrap_deg(a):
    return (np.asarray(a, dtype=float) + 180.0) % 360.0 - 180.0


def loop_margins(G, C) -> dict:
    """L = C·G açık çevriminin kazanç geçiş frekansları ve faz payları (tasarım doğrulaması).

    Dönüş {'Crossovers': [...] (rad/s), 'PhaseMargins': [...] (derece), 'PhaseMargin': en küçük,
    'CrossoverFrequency': en küçük payın frekansı}. Izgara + kök inceltme; MATLAB `allmargin`ın faz payı tanımı."""
    Gs = G if isinstance(G, _Siso) else _Siso(G)
    Cs = C if isinstance(C, _Siso) else _Siso(C, "Denetleyici")
    if Gs.Ts != Cs.Ts:
        raise ControlError("Bitki ve denetleyicinin örnekleme zamanları aynı olmalı.")
    wmax = math.pi / Gs.Ts if Gs.discrete else 1e4 * max(Gs.pz_scale(), Cs.pz_scale())
    wmin = 1e-6 * min(Gs.pz_scale(), Cs.pz_scale())
    w = np.geomspace(wmin, wmax * (1 - 1e-9), 20000)

    def lm(x):
        return np.log(np.abs(Gs.freq(x) * Cs.freq(x)))
    v = lm(w)
    wc, pm = [], []
    for i in np.flatnonzero(np.sign(v[:-1]) * np.sign(v[1:]) < 0):
        r = brentq(lambda x: float(lm(np.array([x]))[0]), w[i], w[i + 1], xtol=1e-14 * w[i], rtol=1e-15)
        Lr = complex((Gs.freq(np.array([r])) * Cs.freq(np.array([r])))[0])
        wc.append(r)
        pm.append(float(_wrap_deg(math.degrees(cmath.phase(Lr)) + 180.0)))
    if not wc:
        return {"Crossovers": [], "PhaseMargins": [], "PhaseMargin": math.inf, "CrossoverFrequency": math.nan}
    k = int(np.argmin(pm))
    return {"Crossovers": wc, "PhaseMargins": pm, "PhaseMargin": pm[k], "CrossoverFrequency": wc[k]}


# ---------------------------------------------------------------------------------------------------------------
# pidtune
# ---------------------------------------------------------------------------------------------------------------
def _ctrl_polys(Kp, Ki, Kd, Tf, Ts, IF, DF) -> tuple[np.ndarray, np.ndarray]:
    C = PID.__new__(PID)
    C._p = dict(Kp=Kp, Ki=Ki, Kd=Kd, Tf=Tf, b=1.0, c=1.0)
    C._IF, C._DF = IF, DF
    return C._branch(Ts, 1.0, 1.0)


def _design_at(G: _Siso, typ: str, w: float, PM: float, alpha: float, IF: str, DF: str, Tf_rule=None):
    """wc = w'de tam hedefli tasarım adayları: [(Kp, Ki, Kd, Tf), ...] (boş: olurlu değil)."""
    x = complex(G.point(np.array([w]))[0])
    g = complex(G.at(np.array([x]))[0])
    if not np.isfinite(g) or g == 0:
        return []
    c = cmath.exp(1j * (-math.pi + math.radians(PM))) / g
    Ts = G.Ts
    Iv = complex(_int_eval(x, Ts, IF))
    Dv = complex(_int_eval(x, Ts, DF))
    Tf = 0.0
    if typ in ("PDF", "PIDF"):
        Tf = 1.0 / (_FILTER_BETA * w) if Tf_rule is None else Tf_rule
        if Ts > 0 and DF == "ForwardEuler":
            Tf = max(Tf, Ts)
    out = []
    if typ == "P":
        out.append((c.real, 0.0, 0.0, 0.0))
    elif typ == "I":
        out.append((0.0, (c / Iv).real, 0.0, 0.0))
    elif typ == "PI":
        if abs(Iv.imag) > 0:
            Ki = c.imag / Iv.imag
            Kp = c.real - Ki * Iv.real
            if Kp * Ki > 0:
                out.append((Kp, Ki, 0.0, 0.0))
    elif typ in ("PD", "PDF"):
        Dd = 1.0 / (Tf + Dv)
        if abs(Dd.imag) > 0:
            Kd = c.imag / Dd.imag
            Kp = c.real - Kd * Dd.real
            if Kp * Kd > 0:
                out.append((Kp, 0.0, Kd, Tf))
    else:  # PID / PIDF: Kp·(1 + IF/(α·Td) + Td·Dd), Td kökü
        Dd = 1.0 / (Tf + Dv)
        a, b = Iv / alpha, Dd
        cc = c.conjugate()
        coef = [(b * cc).imag, cc.imag, (a * cc).imag]
        roots = np.roots(coef) if coef[0] != 0 else (np.array([-coef[2] / coef[1]]) if coef[1] != 0 else [])
        for T in roots:
            if abs(T.imag) > 1e-12 * max(1.0, abs(T)) or T.real <= 0:
                continue
            T = float(T.real)
            S = 1 + a / T + b * T
            Kp = (c * S.conjugate()).real / abs(S) ** 2
            if Kp == 0:
                continue
            out.append((Kp, Kp / (alpha * T), Kp * T, Tf))
    return out


def _check_targets(G: _Siso, gains, w, PM, IF, DF) -> float:
    Kp, Ki, Kd, Tf = gains
    x = G.point(np.array([w]))
    C = Kp + Ki * _int_eval(x, G.Ts, IF) + (Kd / (Tf + _int_eval(x, G.Ts, DF)) if Kd else 0)
    L = complex((C * G.at(x))[0])
    return max(abs(abs(L) - 1.0), abs(math.radians(float(_wrap_deg(math.degrees(cmath.phase(L)) + 180 - PM)))))


def _phase_eff(phi_deg: float) -> float:
    return phi_deg if abs(phi_deg) <= 90 else float(_wrap_deg(phi_deg - 180.0))


_WINDOW = {"PI": (-85.0, -5.0), "PD": (5.0, 85.0), "PDF": (5.0, 85.0), "PID": (-85.0, 85.0), "PIDF": (-85.0, 85.0)}


def _admissible(G, typ, w, PM, alpha, IF, DF):
    """wc = w'de pencere içinde ve kapalı çevrimi kararlı tasarım (yoksa None)."""
    g = complex(G.freq(np.array([w]))[0])
    if not np.isfinite(g) or g == 0:
        return None
    phi = float(_wrap_deg(-180.0 + PM - math.degrees(cmath.phase(g))))
    lo, hi = _WINDOW[typ]
    if not lo <= _phase_eff(phi) <= hi:
        return None
    for gains in _design_at(G, typ, w, PM, alpha, IF, DF):
        nC, dC = _ctrl_polys(*gains, G.Ts, IF, DF)
        if _cl_stable(G, nC, dC):
            return gains
    return None


def _wref(G: _Siso) -> float:
    """Bitkinin karakteristik frekansı: sonlu sıfırdan farklı DC kazançta −3 dB bant genişliği; aksi hâlde bitki
    kazancının 1'i kestiği ilk frekans; o da yoksa kutup / sıfır büyüklüklerinin geometrik ortası."""
    scale = G.pz_scale()
    wmax = 0.9 * math.pi / G.Ts if G.discrete else 1e4 * scale
    w = np.geomspace(1e-5 * scale, wmax, 4000)
    mag = np.abs(G.freq(w))
    x0 = 1.0 if G.discrete else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        g0 = complex(np.polyval(G.num, x0) / np.polyval(G.den, x0)) if np.polyval(G.den, x0) != 0 else np.inf
    if np.isfinite(g0) and abs(g0) > 0:
        target = abs(g0) / math.sqrt(2)
    else:
        target = 1.0
    s = np.sign(mag - target)
    idx = np.flatnonzero(s[:-1] * s[1:] < 0)
    if idx.size:
        i = idx[0]
        return float(brentq(lambda x: float(np.abs(G.freq(np.array([x])))[0] - target), w[i], w[i + 1]))
    return min(scale, wmax)


def _default_wc(G: _Siso, typ: str, PM: float, alpha: float, IF: str, DF: str):
    """wc verilmediğinde: ω_ref = `_wref(G)` başlangıç; ω_ref'ten başlayarak logaritmik ızgarada (adım 10^(1/40))
    önce aşağı, sonra yukarı dönüşümlü taranır; pencere içinde (PI: kontrolcü fazı −85…−5°, PD / PDF: 5…85°,
    PID / PIDF: −85…85°) ve kapalı çevrimi kararlı ilk tasarım seçilir. P ve I türünde iki hedef birden
    yalnız ∠(G·şekil) = −180° + PM olan frekansta tutar: bu koşulun en düşük (kararlı) kökü."""
    wr = _wref(G)
    wmax = 0.95 * math.pi / G.Ts if G.discrete else 1e6 * G.pz_scale()
    wmin = 1e-6 * G.pz_scale()
    if typ in ("P", "I"):
        def fv(w):
            w = np.asarray(w, dtype=float)
            g = G.freq(w)
            if typ == "I":
                g = g * _int_eval(G.point(w), G.Ts, IF)
            return np.sin(math.radians(-180.0 + PM) - np.angle(g))

        def f(w):
            return float(fv(np.array([w]))[0])
        grid = np.geomspace(wmin, wmax, 6000)
        vals = fv(grid)
        sgn = math.copysign(1.0, _plant_type_gain(G, 0)[1])
        for i in np.flatnonzero(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0):
            r = brentq(f, grid[i], grid[i + 1], xtol=1e-15 * grid[i], rtol=1e-15)
            for gains in _design_at(G, typ, r, PM, alpha, IF, DF):
                if abs(_check_targets(G, gains, r, PM, IF, DF)) > 1e-6 or (gains[0] + gains[1]) * sgn <= 0:
                    continue
                nC, dC = _ctrl_polys(*gains, G.Ts, IF, DF)
                if _cl_stable(G, nC, dC):
                    return r, gains
        raise ControlError(f"{typ} denetleyici bu bitkide hedef faz payını ({PM:g}°) kararlı bir çevrimle "
                           "sağlayamıyor; başka bir denetleyici türü seçin ya da geçiş frekansı verin.")
    step = 10 ** (1 / 40)
    for k in range(0, 481):
        for sgn in ((0,) if k == 0 else (-1, 1)):
            w = wr * step ** (sgn * k)
            if not wmin <= w <= wmax:
                continue
            gains = _admissible(G, typ, w, PM, alpha, IF, DF)
            if gains is not None:
                return w, gains
    raise ControlError(f"{typ} denetleyiciyle hedef faz payını ({PM:g}°) sağlayan kararlı bir tasarım bulunamadı; "
                       "başka bir tür (ör. PIDF) ya da daha küçük faz payı deneyin.")


def pidtune(G, type: Any = "PI", wc: float | None = None, *, PhaseMargin: float = 60.0,
            DesignFocus: str = "balanced", IFormula: str = "ForwardEuler", DFormula: str | None = None,
            form: str = "parallel"):
    """PID ayarlama (MATLAB `pidtune(G, tür, wc, pidtuneOptions(PhaseMargin=…, DesignFocus=…))`).

    Tanım: L = C·G için hedef geçiş frekansı wc'de |L(jwc)| = 1 ve ∠L(jwc) = −180° + PhaseMargin **tam**
    sağlanır (ayrık bitkide z = e^{jwcTs}; gecikmeler dahil). Serbestlikler:
    - P, I: tek parametre; wc verilirse yalnız |L| = 1 (faz payı sonuçtur), verilmezse iki hedefi birden sağlayan
      frekans aranır. PI, PD: iki parametre, iki denklem (tek çözüm).
    - PDF / PIDF: türev filtresi Tf = 1/(10·wc) (ayrık ileri Euler'de en az Ts).
    - PID / PIDF: Ti = α·Td, α = 4 ('balanced'), 8 ('reference-tracking'), 2 ('disturbance-rejection').
    wc verilmezse `_default_wc` kuralı. Kapalı çevrim kararlılığı kutuplarla (sürekli + gecikmede Nyquist)
    denetlenir; wc verilip çevrim kararsızsa ControlWarning ve info['Stable'] = False.

    `type`: 'P', 'I', 'PI', 'PD', 'PDF', 'PID', 'PIDF' ya da şablon PID nesnesi (biçim / formüller ondan alınır).
    Dönüş `(C, info)`: C paralel (`form='standard'` ile standart) PID; info = {'Stable', 'CrossoverFrequency',
    'PhaseMargin'} (sonuç denetleyiciyle yeniden ölçülmüş değerler)."""
    if isinstance(type, _PIDBase):
        tmpl = type
        IFormula, DFormula = tmpl.IFormula, tmpl.DFormula
        p = tmpl._p
        typ = ("P" if p["Kp"] else "") + ("I" if p["Ki"] else "") + ("D" if p["Kd"] else "")
        typ = typ + ("F" if p["Kd"] and p["Tf"] else "")
        form = tmpl._form
        if not typ:
            raise ControlError("Şablon PID nesnesinin en az bir kazancı sıfırdan farklı olmalı.")
    else:
        typ = str(type).upper()
    if typ not in _TYPES:
        raise ControlError(f"Denetleyici türü {', '.join(_TYPES)} türlerinden biri olmalı.")
    if DesignFocus not in _FOCUS_ALPHA:
        raise ControlError("Tasarım odağı 'balanced', 'reference-tracking' ya da 'disturbance-rejection' olmalı.")
    PM = _real(PhaseMargin, "Faz payı")
    if not 0 < PM < 90:
        raise ControlError("Hedef faz payı 0° ile 90° arasında olmalı.")
    Gs = G if isinstance(G, _Siso) else _Siso(G)
    IF = _formula(IFormula, "IFormula")
    if DFormula is None:
        # Tf = 0'lı ayrık türevde ileri Euler nedensel değildir (MATLAB yine de onu döndürür), yamuk z = −1'de
        # kutup verir: varsayılan olarak geri Euler; filtreli türlerde MATLAB varsayılanı ileri Euler
        DFormula = "BackwardEuler" if (Gs.discrete and typ in ("PD", "PID")) else "ForwardEuler"
    DF = _formula(DFormula, "DFormula")
    if Gs.discrete and typ in ("PD", "PID") and DF == "Trapezoidal":
        raise ControlError("Yamuk (Trapezoidal) türev formülü filtresiz (Tf = 0) PID / PD ile kullanılamaz.")
    alpha = _FOCUS_ALPHA[DesignFocus]
    if wc is None:
        w, gains = _default_wc(Gs, typ, PM, alpha, IF, DF)
    else:
        w = _real(wc, "Geçiş frekansı")
        if not w > 0:
            raise ControlError("Geçiş frekansı pozitif olmalı.")
        if Gs.discrete and w >= math.pi / Gs.Ts:
            raise ControlError(f"Geçiş frekansı Nyquist frekansından (π/Ts = {math.pi / Gs.Ts:.4g} rad/s) küçük "
                               "olmalı.")
        if typ in ("P", "I"):
            x = Gs.point(np.array([w]))
            g = complex(Gs.at(x)[0]) * (complex(_int_eval(x[0], Gs.Ts, IF)) if typ == "I" else 1.0)
            if not np.isfinite(g) or g == 0:
                raise ControlError("Bitkinin bu frekanstaki kazancı sıfır ya da sonsuz.")
            # |L| = 1; işaret: bitkinin (integratörler / türevleyiciler ayrılmış) statik kazancının işareti
            k = math.copysign(1.0 / abs(g), _plant_type_gain(Gs, 0)[1])
            gains = (k, 0.0, 0.0, 0.0) if typ == "P" else (0.0, k, 0.0, 0.0)
        else:
            cands = _design_at(Gs, typ, w, PM, alpha, IF, DF)
            if not cands:
                raise ControlError(f"{typ} denetleyici {w:.4g} rad/s'de {PM:g}° faz payını sağlayamaz (gereken faz "
                                   "katkısı bu türün yapabileceği aralıkta değil ya da kazançlar zıt işaretli çıkıyor);"
                                   " başka bir geçiş frekansı ya da tür seçin.")
            gains = cands[0]
            for gg in cands:
                nC, dC = _ctrl_polys(*gg, Gs.Ts, IF, DF)
                if _cl_stable(Gs, nC, dC):
                    gains = gg
                    break
    Kp, Ki, Kd, Tf = gains
    C = PID(Kp, Ki, Kd, Tf, Gs.Ts, IF, DF)
    nC, dC = C._num[0][0], C._den[0][0]
    stable = _cl_stable(Gs, nC, dC)
    x = Gs.point(np.array([w]))
    L = complex((C.evalfr(x) * Gs.at(x))[0])
    info = {"Stable": bool(stable), "CrossoverFrequency": float(w),
            "PhaseMargin": float(_wrap_deg(math.degrees(cmath.phase(L)) + 180.0)), "LoopGainAtCrossover": abs(L)}
    if not stable:
        warnings.warn("Ayarlanan denetleyiciyle kapalı çevrim kararsız; geçiş frekansını ya da türü değiştirin.",
                      ControlWarning, stacklevel=2)
    if form == "standard":
        C = pidstd(C)
    return C, info


# ---------------------------------------------------------------------------------------------------------------
# Öncül / ardıl, çentik
# ---------------------------------------------------------------------------------------------------------------
def _make_tf(num, den, Ts):
    if _HAVE_LTI:
        from spt.control.lti import TransferFunction
        return TransferFunction(np.asarray(num, float), np.asarray(den, float), Ts)
    return _TFBase([[np.asarray(num, float)]], [[np.asarray(den, float)]], Ts)


def _first_order_tf(K: float, zc: float, pc: float, Ts: float):
    """K(1 + p/zc)/(1 + p/pc); ayrıkta p = (2/Ts)(z − 1)/(z + 1) (çift doğrusal; DC kazancı korunur)."""
    if Ts == 0:
        num = K * np.array([1.0 / zc, 1.0])
        den = np.array([1.0 / pc, 1.0])
    else:
        c1, c2 = 2.0 / (Ts * zc), 2.0 / (Ts * pc)
        num = K * np.array([1.0 + c1, 1.0 - c1])
        den = np.array([1.0 + c2, 1.0 - c2])
    return num / den[0], den / den[0]


def lead(phase_max: float, wm: float, K: float = 1.0, Ts: float = 0.0):
    """Öncül kompansatör: en büyük faz katkısı `phase_max` (derece, 0…90) `wm` (rad/s) frekansında.
    C = K·(1 + s/z)/(1 + s/p), α = (1 − sin φ)/(1 + sin φ), z = wm·√α, p = wm/√α; |C(jwm)| = K/√α.
    Ayrıkta çift doğrusal (Tustin) eşlemeyle, wm frekansı ön bükmeli (z = e^{jwmTs}'de aynı faz)."""
    phi = _real(phase_max, "Faz katkısı")
    if not 0 < phi < 90:
        raise ControlError("Öncül kompansatörün faz katkısı 0° ile 90° arasında olmalı.")
    return _sym_comp(phi, _real(wm, "Frekans"), K, Ts)


def lag(phase_min: float, wm: float, K: float = 1.0, Ts: float = 0.0):
    """Ardıl kompansatör: `wm` frekansında `phase_min` (derece, −90…0) faz katkısı (simetrik ardıl)."""
    phi = _real(phase_min, "Faz katkısı")
    if not -90 < phi < 0:
        raise ControlError("Ardıl kompansatörün faz katkısı −90° ile 0° arasında olmalı.")
    return _sym_comp(phi, _real(wm, "Frekans"), K, Ts)


def _nu(w: float, Ts: float) -> float:
    return w if Ts == 0 else (2.0 / Ts) * math.tan(w * Ts / 2.0)


def _sym_comp(phi: float, wm: float, K: float, Ts: float):
    if not wm > 0:
        raise ControlError("Frekans pozitif olmalı.")
    if Ts > 0 and wm >= math.pi / Ts:
        raise ControlError("Frekans Nyquist frekansından küçük olmalı.")
    s = math.sin(math.radians(phi))
    alpha = (1 - s) / (1 + s)
    nu = _nu(wm, Ts)
    num, den = _first_order_tf(K, nu * math.sqrt(alpha), nu / math.sqrt(alpha), Ts)
    return _make_tf(num, den, Ts)


def _plant_type_gain(G: _Siso, k: int):
    """(n0, g0): kökendeki (ayrıkta z = 1'deki) kutup sayısı ve lim p^k·G (k = n0 iken hata sabiti)."""
    x0 = 1.0 if G.discrete else 0.0
    den = G.den.copy()
    n0 = 0
    fac = np.array([1.0, -x0])
    while den.size > 1:
        q, r = np.polydiv(den, fac)
        if np.max(np.abs(r)) > 1e-10 * max(1.0, np.max(np.abs(den))):
            break
        den = _trim(q)
        n0 += 1
    num = G.num.copy()
    m0 = 0
    while num.size > 1:
        q, r = np.polydiv(num, fac)
        if np.max(np.abs(r)) > 1e-10 * max(1.0, np.max(np.abs(num))):
            break
        num = _trim(q)
        m0 += 1
    n0 -= m0
    g0 = float(np.polyval(num, x0) / np.polyval(den, x0))
    if G.discrete:
        g0 = g0 / G.Ts ** n0 if n0 > 0 else g0 * G.Ts ** (-n0)
    return n0, g0


def leadlag(G, PhaseMargin: float = 60.0, wc: float | None = None, *, ess: float | None = None,
            input: str = "step", K: float | None = None):
    """Öncül / ardıl kompansatör tasarımı: C = K·(1 + p/z)/(1 + p/q) (p = s; ayrıkta çift doğrusal w düzlemi,
    DC kazancı C(0) = K).

    - Kalıcı durum hatası: `ess` ve `input` ('step', 'ramp', 'parabola') verilirse K hata sabitinden belirlenir
      (bitki türüne göre; hatanın zaten sıfır olduğu durumda K serbest kalır). Doğrudan `K` de verilebilir.
    - wc ve K belli: hedefler (|L(jwc)| = 1, ∠L = −180° + PM) Phillips formülüyle tam çözülür (tek çözüm).
    - wc belli, K serbest: simetrik kompansatör (en büyük faz katkısı wc'de).
    - K belli, wc serbest: simetrik kompansatör; |L| = 1 koşulunu sağlayan en düşük wc aranır (klasik yöntemin tam
      çözümü).
    Dönüş `(C, info)`: info = {'Type' ('lead' / 'lag'), 'K', 'Zero', 'Pole' (p düzleminde), 'CrossoverFrequency',
    'PhaseMargin', 'Stable'}."""
    Gs = G if isinstance(G, _Siso) else _Siso(G)
    PM = _real(PhaseMargin, "Faz payı")
    if not 0 < PM < 90:
        raise ControlError("Hedef faz payı 0° ile 90° arasında olmalı.")
    if ess is not None:
        if K is not None:
            raise ControlError("Kalıcı durum hatası ve K birlikte verilemez.")
        kk = {"step": 0, "ramp": 1, "parabola": 2}.get(input)
        if kk is None:
            raise ControlError("Giriş türü 'step', 'ramp' ya da 'parabola' olmalı.")
        e = _real(ess, "Kalıcı durum hatası")
        if not e > 0:
            raise ControlError("Kalıcı durum hatası pozitif olmalı (sıfır hata için integral etkisi gerekir: "
                               "PI / PID).")
        n0, g0 = _plant_type_gain(Gs, kk)
        if n0 < kk:
            raise ControlError("Bu bitkiyle (yeterli integratör yok) bu girişte kalıcı durum hatası sonsuzdur; "
                               "integral etkili denetleyici kullanın.")
        if n0 == kk:
            Kc = (1.0 / e - 1.0) if kk == 0 else 1.0 / e
            if kk == 0 and Kc <= -1:
                raise ControlError("Basamak hatası için ess < 1 olmalı.")
            K = Kc / g0
    if K is not None:
        K = _real(K, "K")
        if K == 0:
            raise ControlError("K sıfır olamaz.")

    def need(w):
        g = complex(Gs.freq(np.array([w]))[0])
        c = cmath.exp(1j * (-math.pi + math.radians(PM))) / g
        return abs(c), cmath.phase(c)

    def build(Kc, zc, pc):
        num, den = _first_order_tf(Kc, zc, pc, Gs.Ts)
        return num, den

    if wc is not None:
        w = _real(wc, "Geçiş frekansı")
        if not w > 0 or (Gs.discrete and w >= math.pi / Gs.Ts):
            raise ControlError("Geçiş frekansı pozitif ve (ayrıkta) Nyquist frekansından küçük olmalı.")
        M, phi = need(w)
        nu = _nu(w, Gs.Ts)
        if K is None:
            if not -math.radians(89.9) < phi < math.radians(89.9) or abs(phi) < 1e-12:
                raise ControlError(f"{w:.4g} rad/s'de gereken faz katkısı ({math.degrees(phi):.1f}°) tek bir öncül / "
                                   "ardıl ile sağlanamaz (−90°…90°, sıfır değil).")
            s = math.sin(phi)
            alpha = (1 - s) / (1 + s)
            Kc = M * math.sqrt(alpha)
            zc, pc = nu * math.sqrt(alpha), nu / math.sqrt(alpha)
        else:
            r = M / abs(K)
            ph = phi if K > 0 else phi - math.pi
            ph = (ph + math.pi) % (2 * math.pi) - math.pi
            if abs(math.sin(ph)) < 1e-12:
                raise ControlError("Gereken faz katkısı sıfır; kompansatör gerekmez (yalnız kazanç).")
            a = (r - math.cos(ph)) / math.sin(ph)
            b = (r * math.cos(ph) - 1) / (r * math.sin(ph))
            if not (a > 0 and b > 0):
                raise ControlError("Verilen kazanç (kalıcı durum şartı) ve geçiş frekansıyla hedef faz payı tek bir "
                                   "öncül / ardıl ile sağlanamaz (sıfır / kutup sağ yarı düzleme düşüyor); geçiş "
                                   "frekansını değiştirin.")
            Kc, zc, pc = K, nu / a, nu / b
    else:
        if K is None:
            raise ControlError("Geçiş frekansı ya da kalıcı durum şartı (ess / K) verin.")
        scale = Gs.pz_scale()
        wmax = 0.95 * math.pi / Gs.Ts if Gs.discrete else 1e4 * scale
        grid = np.geomspace(1e-4 * scale, wmax, 4000)

        def hv(w):
            c = np.exp(1j * (-math.pi + math.radians(PM))) / Gs.freq(np.asarray(w, dtype=float))
            ph = np.angle(c) if K > 0 else np.angle(c) - math.pi
            ph = (ph + math.pi) % (2 * math.pi) - math.pi
            s = np.sin(ph)
            with np.errstate(divide="ignore", invalid="ignore"):
                v = np.log(np.abs(c) * np.sqrt((1 - s) / (1 + s)) / abs(K))
            return np.where(np.abs(ph) < math.radians(89.0), v, np.nan)

        def h(w):
            return float(hv(np.array([w]))[0])
        vals = hv(grid)
        found = None
        for i in np.flatnonzero(np.isfinite(vals[:-1]) & np.isfinite(vals[1:]) & (vals[:-1] * vals[1:] < 0)):
            w = brentq(h, grid[i], grid[i + 1], xtol=1e-14 * grid[i], rtol=1e-15)
            M, phi = need(w)
            ph = phi if K > 0 else phi - math.pi
            ph = (ph + math.pi) % (2 * math.pi) - math.pi
            s = math.sin(ph)
            alpha = (1 - s) / (1 + s)
            nu = _nu(w, Gs.Ts)
            cand = (K, nu * math.sqrt(alpha), nu / math.sqrt(alpha))
            num, den = build(*cand)
            if _cl_stable(Gs, num, den):
                found = (w, cand)
                break
        if found is None:
            raise ControlError("Bu kazançla hedef faz payını sağlayan kararlı bir öncül / ardıl bulunamadı.")
        w, (Kc, zc, pc) = found
    num, den = build(Kc, zc, pc)
    C = _make_tf(num, den, Gs.Ts)
    stable = _cl_stable(Gs, num, den)
    x = Gs.point(np.array([w]))
    with np.errstate(divide="ignore", invalid="ignore"):
        L = complex((np.polyval(num, x) / np.polyval(den, x) * Gs.at(x))[0])
    info = {"Type": "lead" if zc < pc else "lag", "K": float(Kc), "Zero": -float(zc), "Pole": -float(pc),
            "CrossoverFrequency": float(w), "PhaseMargin": float(_wrap_deg(math.degrees(cmath.phase(L)) + 180.0)),
            "Stable": bool(stable)}
    if not stable:
        warnings.warn("Tasarlanan kompansatörle kapalı çevrim kararsız.", ControlWarning, stacklevel=2)
    return C, info


def notch(w0: float, depth_db: float, width: float, Ts: float = 0.0):
    """Çentik filtresi N(s) = (s² + 2ζz·w0·s + w0²)/(s² + 2ζp·w0·s + w0²).

    `depth_db` (< 0): w0'daki kazanç (dB) → g = 10^(depth/20) = ζz/ζp. `width` (rad/s): zayıflamanın derinliğin
    yarısını (dB) aştığı bandın genişliği — tam: ζp = width/(2·w0·√g), ζz = g·ζp. Sonsuz derinlikte (−inf)
    genişlik −3 dB bandıdır (ζz = 0, ζp = width/(2·w0)). Ayrıkta w0'a ön bükmeli Tustin (merkez frekans ve
    derinlik tam korunur; genişlik frekans bükülmesi kadar değişir)."""
    w0 = _real(w0, "Çentik frekansı")
    width = _real(width, "Çentik genişliği")
    depth_db = float(depth_db)
    Ts = _real(Ts, "Ts")
    if not w0 > 0 or not width > 0:
        raise ControlError("Çentik frekansı ve genişliği pozitif olmalı.")
    if not depth_db < 0:
        raise ControlError("Çentik derinliği negatif dB olmalı (ör. −20 dB).")
    if Ts > 0 and w0 >= math.pi / Ts:
        raise ControlError("Çentik frekansı Nyquist frekansından (π/Ts) küçük olmalı.")
    if math.isinf(depth_db):
        zp, zz = width / (2 * w0), 0.0
    else:
        g = 10 ** (depth_db / 20)
        zp = width / (2 * w0 * math.sqrt(g))
        zz = g * zp
    if Ts == 0:
        num = np.array([1.0, 2 * zz * w0, w0 ** 2])
        den = np.array([1.0, 2 * zp * w0, w0 ** 2])
    else:
        k = w0 / math.tan(w0 * Ts / 2)

        def bil(z):
            return (k * k * np.array([1.0, -2.0, 1.0]) + 2 * z * w0 * k * np.array([1.0, 0.0, -1.0])
                    + w0 * w0 * np.array([1.0, 2.0, 1.0]))
        num, den = bil(zz), bil(zp)
        num, den = num / den[0], den / den[0]
    return _make_tf(num, den, Ts)


# ---------------------------------------------------------------------------------------------------------------
# Kök yer eğrisi kazançları
# ---------------------------------------------------------------------------------------------------------------
def _cl_poles(G: _Siso, K: float) -> np.ndarray:
    dG = G.den
    if G.discrete and G.delay:
        dG = np.concatenate([dG, np.zeros(int(G.delay))])
    elif G.delay:
        raise ControlError("Sürekli zamanda gecikmeli modelin kök yer eğrisi için önce Padé yaklaşımı uygulayın.")
    char = _padd(dG, K * G.num)
    return np.roots(char) if char.size > 1 else np.zeros(0, dtype=complex)


def _nearest_gain(Gs: _Siso, x: complex) -> float:
    """Kök yer eğrisinin (K ≥ 0) x noktasına en yakın geçtiği kazanç: x eğri üzerindeyse K = −1/G(x) (tam);
    değilse min_K min_i |kutup_i(K) − x| (logaritmik ızgara + altın oran inceltmesi)."""
    g = complex(Gs.at(np.array([x]))[0])
    if g != 0 and np.isfinite(g):
        k = -1.0 / g
        if k.real > 0 and abs(k.imag) <= 1e-9 * abs(k):
            return float(k.real)

    def dist(K):
        return float(np.min(np.abs(_cl_poles(Gs, K) - x)))
    k0 = abs(-1.0 / g) if (g != 0 and np.isfinite(g)) else 1.0
    grid = np.concatenate([[0.0], k0 * np.geomspace(1e-8, 1e8, 1601)])
    dv = np.array([dist(K) for K in grid])
    i = int(np.argmin(dv))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, grid.size - 1)]
    gr = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = dist(c), dist(d)
    for _ in range(200):
        if abs(b - a) <= 1e-13 * max(1.0, abs(b)):
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = dist(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = dist(d)
    K = 0.5 * (a + b)
    return float(K if dist(K) <= dv[i] else grid[i])


def rlocfind(G, p):
    """Kök yer eğrisinde verilen noktalar için kazanç (MATLAB `rlocfind(sys, p)`: kapalı çevrim köklerinden biri
    p'ye yakın olan kazanç). Nokta eğri üzerindeyse K = −1/G(p) (tam); değilse eğrinin p'ye en yakın geçtiği
    K ≥ 0 (MATLAB kendi kazanç ızgarasındaki en yakın noktayı seçer — bizimki en az onun kadar yakındır).

    Dönüş `(K, poles)`: K (len(p),), poles (n, len(p)) (sütun başına kapalı çevrim kutupları)."""
    Gs = G if isinstance(G, _Siso) else _Siso(G)
    if Gs.delay and not Gs.discrete:
        raise ControlError("Sürekli zamanda gecikmeli modelin kök yer eğrisi için önce Padé yaklaşımı uygulayın.")
    pts = np.atleast_1d(np.asarray(p, dtype=complex)).ravel()
    Ks = [_nearest_gain(Gs, complex(x)) for x in pts]
    cols = [_cl_poles(Gs, K) for K in Ks]
    n = max((len(c) for c in cols), default=0)
    P = np.full((n, len(cols)), np.nan, dtype=complex)
    for j, c in enumerate(cols):
        P[:len(c), j] = c
    return np.array(Ks), P


def gain_for_damping(G, zeta: float | None = None, wn: float | None = None):
    """İstenen sönüm oranı ζ ve / veya doğal frekans wn için kök yer eğrisi kazançları (K > 0).

    - ζ verilirse (0 ≤ ζ < 1): kök yer eğrisinin sabit-ζ ışınıyla (ayrıkta z = e^{sTs} eğrisiyle) kesişimleri;
      ζ = 0 sürekli zamanda sanal eksen geçişidir (kararlılık sınırı kazancı).
    - yalnız wn: |s| = wn yarım çemberiyle kesişimler.
    - ikisi birden: tek nokta s = wn(−ζ + j√(1−ζ²)) için `rlocfind` (gerçel kısım).
    Dönüş: [{'K', 'pole', 'zeta', 'wn', 'poles'}] kazanca göre artan."""
    Gs = G if isinstance(G, _Siso) else _Siso(G)
    if Gs.delay and not Gs.discrete:
        raise ControlError("Sürekli zamanda gecikmeli modelin kök yer eğrisi için önce Padé yaklaşımı uygulayın.")
    if zeta is None and wn is None:
        raise ControlError("Sönüm oranı ya da doğal frekans verin.")
    if zeta is not None:
        zeta = _real(zeta, "Sönüm oranı")
        if not 0 <= zeta < 1:
            raise ControlError("Sönüm oranı 0 ≤ ζ < 1 olmalı.")
    if wn is not None:
        wn = _real(wn, "Doğal frekans")
        if not wn > 0:
            raise ControlError("Doğal frekans pozitif olmalı.")
    Ts = Gs.Ts

    def to_x(s):
        return np.exp(s * Ts) if Gs.discrete else s

    def rec(s):
        x = complex(to_x(s))
        g = complex(Gs.at(np.array([x]))[0])
        K = float((-1.0 / g).real)
        return {"K": K, "pole": x, "zeta": None if zeta is None else zeta, "wn": abs(s), "poles": _cl_poles(Gs, K)}

    if zeta is not None and wn is not None:
        return [rec(wn * complex(-zeta, math.sqrt(1 - zeta * zeta)))]
    scale = Gs.pz_scale()
    if zeta is not None:
        q = complex(-zeta, math.sqrt(1 - zeta * zeta))
        hi = (math.pi / Ts) / q.imag * 0.999999 if Gs.discrete else 1e4 * scale
        grid = np.geomspace(1e-5 * scale if not Gs.discrete else 1e-6 / Ts, hi, 20000)

        def curve(t):
            return t * q
    else:
        if Gs.discrete and wn >= math.pi / Ts:
            raise ControlError("Doğal frekans Nyquist frekansından küçük olmalı.")
        grid = np.linspace(math.pi / 2, math.pi, 20001)[:-1]
        grid = grid[1:]

        def curve(t):
            return wn * np.exp(1j * t)

    def fv(t):
        g = Gs.at(to_x(curve(np.asarray(t, dtype=float))))
        with np.errstate(divide="ignore", invalid="ignore"):
            v = g.imag / np.abs(g)
        return np.where(np.isfinite(g) & (g != 0), v, np.nan)

    def f(t):
        return float(fv(np.array([t]))[0])
    vals = fv(grid)
    out = []
    for i in np.flatnonzero(np.isfinite(vals[:-1]) & np.isfinite(vals[1:]) & (vals[:-1] * vals[1:] < 0)):
        t = brentq(f, grid[i], grid[i + 1], xtol=1e-15 * abs(grid[i]), rtol=1e-15)
        s = curve(t)
        g = complex(Gs.at(np.array([to_x(s)]))[0])
        if g.real < 0 and abs(g.imag) < 1e-6 * abs(g):
            r = rec(s)
            if r["K"] > 0:
                out.append(r)
    out.sort(key=lambda r: r["K"])
    return out
