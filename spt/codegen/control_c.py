"""Kontrolcüler için C99 kod üreteci (MCU), ayak izi (flash / RAM / yığın) ve hesap maliyeti.

Kapsam (her üretimde tek kontrolcü; dosyalar `<önek>_<tür>.h/.c`, `<önek>_<tür>_ornek.c`, `KULLANIM_<önek>.md`):
  - `PidSpec`  → "pid": ayrık PID, MATLAB `pid` / `pidstd` / `pid2` nesnesi ve Simulink "Discrete PID Controller
    (2DOF)" bloğu anlamıyla: C(z) = Kp + Ki·IF(z) + Kd / (Tf + DF(z)); IF / DF ∈ {ForwardEuler, BackwardEuler,
    Trapezoidal}; 2-DOF ağırlıkları b (oransal) ve c (türev); çıkış doyumu; anti-windup (geri hesaplama Kb ya da
    sıkıştırma — Simulink tanımı), izleme (tracking, Kt) modu; çalışma anında kazanç / sınır değişimi.
  - `LtiSpec`  → "lti": genel ayrık LTI kontrolcü. SISO'da ikinci derece bölümler (SOS, devrik doğrudan biçim II);
    MIMO'da durum uzayı (2'nin kuvvetleriyle tam dengelenmiş benzerlik dönüşümü: giriş / çıkış davranışı değişmez).
    Statik kazanç (durum yok) da bu türdür.
  - `ObserverSpec` → "obs": durum geri beslemesi + gözlemci: 'delayed' (MATLAB `reg` / `estim`, `lqgreg` ile
    'delayed' kalman) ve 'current' (MATLAB `lqgreg` / `lqgtrack` ile 'current' kalman: u = −K·x̂[n|n]);
    u = −K·x̂ (+ Nbar·r ya da LQI integral durumu xi[n+1] = xi[n] + Ts·(r[n] − y[n]), MATLAB `lqi` / `lqgtrack`).
    'state' (`ObserverSpec.state_feedback`): tüm durumlar ölçülür, gözlemci yok: u = −K·x (+ Nbar·r ya da LQI).
  - `KalmanSpec` → "kf": kararlı durum Kalman süzgeci (MATLAB `kalman` 'current' / 'delayed'); çıkış [ŷ; x̂];
    gürültü ölçüme doğrudan giriyorsa (H ≠ 0) MATLAB'in My kazancıyla ŷ[n|n].
  MPC'nin C kodu bu sürümde yoktur (MPC modülü tamamlanınca eklenecek).

Ortak C kuralları: SPT_REAL (float / double, `<P>_USE_DOUBLE`), sabitler `<P>_C()` ile (float'ta istemsiz double
yükseltmesi yok; -Wdouble-promotion temiz), malloc yok, global değişken yok, yeniden girişli (tüm durum çağıranın
yapısında), örnekleme zamanı Ts derleme anında sabittir. Sonlu olmayan (NaN / ±sonsuz) giriş: durum DEĞİŞMEZ ve
son çıkış tutulur (fonksiyon 0 döner); Kalman'da yalnız ölçüm eksikse o örnekte ölçüm güncellemesi atlanır.

Doğruluk: üretilen C, bu modüldeki Python başvurusuyla (`simulate`) aynı işlem sırasını kullanır (double'da ana
makinede bit düzeyinde aynı); MATLAB ile karşılaştırma tests/test_codegen_control.py içindedir (pid / pidstd / pid2
lsim, Simulink PID bloğu, tf / ss lsim, reg / estim / lqgreg / lqgtrack, dlqr + Nbar / lqi kapalı çevrimi,
kalman current / delayed ve My).

Ayak izi (`footprint`): yerel derleyici varsa ÖLÇÜLÜR (nesne bölümleri, sizeof, newlib-nano ile bağlanmış imaj farkı,
disassembly'den en kötü yığın); IAR / Xtensa / RISC-V ölçülmüş bir değerden TAHMİN edilir. Maliyet (`cost`): adım
başına işlem sayıları ana makinede C++ sayaç türüyle ÖLÇÜLÜR, çevrimler MCU profil tablosundan TAHMİN edilir
(`spt.codegen.fusion_c` ile aynı düzenek).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import warnings
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from spt.codegen import toolchain as tc
from spt.codegen.fusion_c import (CostRow, FootprintRow, GeneratedFile, _c_float, _count_prelude, _fmt_int,
                                  _op_weights, write_files)
from spt.codegen.mcu import (COMPILERS, IAR_LIBM_FACTOR, IAR_SIZE_FACTOR, MCU_PROFILES, OPS, OPT_LEVELS, McuProfile,
                             profile)
from spt.control.errors import ControlError, ControlWarning

__all__ = ["ControlCodegenOptions", "CostRow", "FootprintRow", "FORMULAS", "GeneratedFile", "KalmanSpec", "LtiSpec",
           "OBS_FORMS", "ObserverSpec", "PidSpec", "build_check", "check_prefix", "controller_spec", "cost",
           "describe", "footprint", "generate", "host_program", "kind_title", "nbar", "run_generated", "simulate",
           "state_size", "usage_markdown", "write_files"]

FORMULAS = ("ForwardEuler", "BackwardEuler", "Trapezoidal")
_ALPHA = {"ForwardEuler": 0.0, "BackwardEuler": 1.0, "Trapezoidal": 0.5}   # IF / DF(z) = Ts·(α·z + 1 − α)/(z − 1)
_FORMULA_TR = {"ForwardEuler": "ileri Euler", "BackwardEuler": "geri Euler", "Trapezoidal": "yamuk (Tustin)"}
ANTI_WINDUP = ("none", "back-calculation", "clamping")
_AW_TR = {"none": "yok", "back-calculation": "geri hesaplama", "clamping": "sıkıştırma (clamping)"}
KIND_TITLES = {"pid": "Ayrık PID", "lti": "Ayrık LTI kontrolcü", "obs": "Durum geri beslemesi + gözlemci",
               "kf": "Kararlı durum Kalman süzgeci"}
OBS_FORMS = ("delayed", "current", "state")
COST_SAMPLES = 2000
COST_SEED = 5
PREFIX_MAX = 12          # en uzun dış ad (<önek>_pid_set_standard) C99'un garanti ettiği 31 karakteri aşmasın
WARN_EXTRA = ("-Wshadow", "-Wfloat-conversion", "-Wundef")


# ============================================================================ seçenekler
@dataclass(frozen=True)
class ControlCodegenOptions:
    real: str = "float"                  # "float" | "double"
    mcu: str = "cortex-m4f"
    compiler: str = "gcc"
    opt: str = "-Os"
    clock_mhz: float = 168.0
    fs_hz: float | None = None           # çağrı hızı; None → 1/Ts (kontrolcü bu hız için tasarlanmıştır)
    flash_kb: float = 1024.0
    ram_kb: float = 192.0
    prefix: str = "ctl"
    name: str = ""                       # kullanıcıya görünen kontrolcü adı (belgede)


# ============================================================================ kontrolcü tanımları
def _finite(v: float) -> bool:
    return isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v))


@dataclass(frozen=True)
class PidSpec:
    """Ayrık PID (paralel eşdeğer kazançlarla saklanır).

    C(z) = Kp·(b·r − y) + Ki·IF(z)·(r − y) + Kd/(Tf + DF(z))·(c·r − y) (MATLAB pid / pid2; 1-DOF'ta b = c = 1).
    Doyum + anti-windup Simulink "Discrete PID Controller" tanımıyla (R2025b'den ölçüldü): a = P + D + x_I
    (integratörün durumuyla toplam) için geri hesaplama integratör girişine Kb·(sat(a) − a), izleme modu
    Kt·(tr − sat(a)) ekler; sıkıştırma, a doyumdayken aşma yönü ile integratör girişi (Ki·e + izleme terimi) aynı
    işaretliyse girişi sıfırlar. Çıkış u = sat(a + β·Ts·w) (geri Euler / yamukta integratörün doğrudan geçişi).
    Kb ve Kt paralel eşdeğerdir (Simulink ideal biçimindeki Kb / Kt, Kp ile çarpılmış hâlidir, bkz. `standard`)."""
    Kp: float = 1.0
    Ki: float = 0.0
    Kd: float = 0.0
    Tf: float = 0.0
    Ts: float = 0.01
    i_formula: str = "ForwardEuler"
    d_formula: str = "ForwardEuler"
    b: float = 1.0
    c: float = 1.0
    u_min: float = -math.inf
    u_max: float = math.inf
    anti_windup: str = "none"
    Kb: float = 1.0
    Kt: float = 1.0
    tracking: bool = False               # True: örnek kod ve sürücü izleme girişini (tr) kullanır
    form: str = "parallel"               # "parallel" | "standard" (gösterim; C hep paralel eşdeğerle hesaplar)
    two_dof: bool = False                # gösterim: r ve y ayrı (False: tek giriş e, C'de r = e, y = 0)

    kind = "pid"

    @staticmethod
    def standard(Kp: float, Ti: float = math.inf, Td: float = 0.0, N: float = math.inf, **kw) -> PidSpec:
        """MATLAB pidstd: Kp·(1 + 1/(Ti·IF) + Td/(Td/N + DF)) → Ki = Kp/Ti, Kd = Kp·Td, Tf = Td/N.

        Kb / Kt Simulink ideal biçimindeki gibi verilir (integratör girişinde, Kp'den önce) ve Kp ile çarpılarak
        paralel eşdeğere çevrilir."""
        if not (_finite(Kp) and (Ti == math.inf or (_finite(Ti) and Ti > 0)) and _finite(Td) and Td >= 0
                and (N == math.inf or (_finite(N) and N > 0))):
            raise ControlError("Standart PID: Kp sonlu, Ti > 0 (ya da sonsuz), Td ≥ 0, N > 0 (ya da sonsuz) olmalı")
        ki = 0.0 if Ti == math.inf else Kp / Ti
        tf = 0.0 if (N == math.inf or Td == 0) else Td / N
        for key in ("Kb", "Kt"):
            if key in kw:
                kw[key] = Kp * kw[key]
        return PidSpec(Kp=Kp, Ki=ki, Kd=Kp * Td, Tf=tf, form="standard", **kw)

    @property
    def has_limits(self) -> bool:
        return self.u_min != -math.inf or self.u_max != math.inf

    def validate(self) -> None:
        for key in ("Kp", "Ki", "Kd", "Tf", "Ts", "b", "c", "Kb", "Kt"):
            if not _finite(getattr(self, key)):
                raise ControlError(f"PID: {key} sonlu bir sayı olmalı")
        if self.Ts <= 0:
            raise ControlError("PID ayrık olmalı: örnekleme zamanı Ts > 0 (sürekli PID'yi önce ayrıklaştırın)")
        if self.Tf < 0:
            raise ControlError("PID: türev süzgeci zaman sabiti Tf ≥ 0 olmalı")
        for key in ("i_formula", "d_formula"):
            if getattr(self, key) not in FORMULAS:
                raise ControlError(f"PID: ayrıklaştırma formülü {', '.join(FORMULAS)} içinden olmalı")
        if self.Kd != 0:
            den = self.Tf + _ALPHA[self.d_formula] * self.Ts
            if not den > 0.5 * self.Ts:
                why = {"ForwardEuler": "ileri Euler türev formülünde Tf > Ts/2 olmalı (Tf = 0 nedensel değil, "
                                       "Tf ≤ Ts/2 kararsız süzgeç)",
                       "Trapezoidal": "yamuk türev formülünde Tf > 0 olmalı (Tf = 0'da z = −1'de kutup)",
                       "BackwardEuler": "Tf ≥ 0 olmalı"}[self.d_formula]
                raise ControlError(f"PID türev terimi gerçeklenemez: {why}")
        if self.anti_windup not in ANTI_WINDUP:
            raise ControlError("PID: anti-windup 'none', 'back-calculation' ya da 'clamping' olmalı")
        if any(math.isnan(v) for v in (self.u_min, self.u_max)) or not self.u_min < self.u_max:
            raise ControlError("PID: çıkış alt sınırı üst sınırdan küçük olmalı")
        if self.anti_windup != "none" and not self.has_limits:
            raise ControlError("Anti-windup için çıkış doyumu (alt / üst sınır) gerekli")
        if self.Kb < 0 or self.Kt < 0:
            raise ControlError("PID: geri hesaplama (Kb) ve izleme (Kt) kazançları negatif olamaz")
        if self.form not in ("parallel", "standard"):
            raise ControlError("PID biçimi 'parallel' ya da 'standard' olmalı")

    @property
    def cd(self) -> float:
        """Türev katsayısı 1/(Tf + α_D·Ts) (Kd = 0 ise 0)."""
        return 0.0 if self.Kd == 0 else 1.0 / (self.Tf + _ALPHA[self.d_formula] * self.Ts)


def _mat(M: Any, name: str, rows: int | None = None, cols: int | None = None) -> np.ndarray:
    A = np.atleast_2d(np.asarray(M, dtype=float))
    if A.ndim != 2:
        raise ControlError(f"{name} bir matris olmalı")
    if rows is not None and A.shape[0] != rows or cols is not None and A.shape[1] != cols:
        raise ControlError(f"{name} boyutu {A.shape[0]}×{A.shape[1]}; beklenen {rows}×{cols}")
    if not np.all(np.isfinite(A)):
        raise ControlError(f"{name} sonlu olmayan değer içeriyor")
    return A


@dataclass(frozen=True, eq=False)
class LtiSpec:
    """Genel ayrık LTI kontrolcü: durum uzayı (A, B, C, D) ya da SISO tf (num, den; azalan z kuvvetleri)."""
    Ts: float
    A: np.ndarray | None = None
    B: np.ndarray | None = None
    C: np.ndarray | None = None
    D: np.ndarray | None = None
    num: np.ndarray | None = None
    den: np.ndarray | None = None
    realization: str = "auto"            # "auto" (SISO → sos, MIMO → ss) | "sos" | "ss"

    kind = "lti"

    @staticmethod
    def from_tf(num, den, Ts: float, realization: str = "auto") -> LtiSpec:
        return LtiSpec(Ts=float(Ts), num=np.atleast_1d(np.asarray(num, float)),
                       den=np.atleast_1d(np.asarray(den, float)), realization=realization)

    @staticmethod
    def from_ss(A, B, C, D, Ts: float, realization: str = "auto") -> LtiSpec:
        D = np.atleast_2d(np.asarray(D, float))
        nx = np.asarray(A, float).size and np.atleast_2d(np.asarray(A, float)).shape[0]
        A = np.zeros((0, 0)) if nx == 0 else np.atleast_2d(np.asarray(A, float))
        B = np.zeros((0, D.shape[1])) if nx == 0 else np.atleast_2d(np.asarray(B, float))
        C = np.zeros((D.shape[0], 0)) if nx == 0 else np.atleast_2d(np.asarray(C, float))
        return LtiSpec(Ts=float(Ts), A=A, B=B, C=C, D=D, realization=realization)

    @staticmethod
    def gain(K, Ts: float) -> LtiSpec:
        """Statik kazanç matrisi y = K·u (ör. durum geri beslemesi u = −K·x için K'nin negatifi verilir)."""
        K = np.atleast_2d(np.asarray(K, float))
        return LtiSpec.from_ss(np.zeros((0, 0)), np.zeros((0, K.shape[1])), np.zeros((K.shape[0], 0)), K, Ts, "ss")

    @property
    def is_tf(self) -> bool:
        return self.num is not None

    @property
    def shape(self) -> tuple[int, int]:
        return (1, 1) if self.is_tf else np.atleast_2d(self.D).shape

    def validate(self) -> None:
        if not (_finite(self.Ts) and self.Ts > 0):
            raise ControlError("Kontrolcü ayrık olmalı: örnekleme zamanı Ts > 0 (sürekli modeli önce ayrıklaştırın)")
        if self.realization not in ("auto", "sos", "ss"):
            raise ControlError("Gerçekleme 'auto', 'sos' ya da 'ss' olmalı")
        if self.is_tf:
            num, den = np.trim_zeros(self.num, "f"), np.trim_zeros(self.den, "f")
            if den.size == 0 or not (np.all(np.isfinite(num)) and np.all(np.isfinite(den))):
                raise ControlError("Aktarım fonksiyonu sonlu katsayılı olmalı ve paydası sıfır olamaz")
            if num.size > den.size:
                raise ControlError("Aktarım fonksiyonu nedensel değil (pay derecesi paydadan büyük): ayrık kontrolcü "
                                   "olarak gerçeklenemez")
        else:
            D = _mat(self.D, "D")
            nx = 0 if self.A is None or np.size(self.A) == 0 else _mat(self.A, "A").shape[0]
            if nx:
                _mat(self.A, "A", nx, nx)
                _mat(self.B, "B", nx, D.shape[1])
                _mat(self.C, "C", D.shape[0], nx)
        if self.realization == "sos" and self.shape != (1, 1):
            raise ControlError("İkinci derece bölümler (SOS) yalnız tek giriş-tek çıkışlı kontrolcüde kullanılabilir")

    @property
    def nx(self) -> int:
        return 0 if self.A is None or np.size(self.A) == 0 else np.atleast_2d(self.A).shape[0]

    def use_sos(self) -> bool:
        return self.shape == (1, 1) and self.realization in ("auto", "sos")


@dataclass(frozen=True, eq=False)
class ObserverSpec:
    """Gözlemcili durum geri beslemesi (bitki ayrık: x[n+1] = A x + B u, y = C x + D u).

    delayed: u[n] = −K·x̂[n|n−1]; current: x̂[n|n] = x̂[n|n−1] + M·(y − C·x̂[n|n−1]), u[n] = −K·x̂[n|n].
    Tahmin: x̂[n+1|n] = A·x̂[n|n−1] + B·u + L·(y − C·x̂[n|n−1] − D·u). Kalman tasarımında L ve M (MATLAB kalman'ın
    L, Mx çıktıları) birlikte verilir; M verilmezse ders kitabı "current estimator" bağıntısı L = A·M ile
    M = A⁻¹·L alınır (MATLAB R2025b estim / reg 'current' biçimi sunmaz).
    reference: "none" | "nbar" (u += Nbar·r; Nbar verilmezse kapalı çevrim DC kazancından) |
    "lqi" (K = [Kx Ki], xi[n+1] = xi[n] + Ts·(r − y), u = −Kx·x̂ − Ki·xi).
    form "state" (`state_feedback`): gözlemci yok, ölçüm girişi durumun kendisi x (L kullanılmaz); LQI'da
    y = C·x + D·u (uygulanan u) modelden hesaplanır."""
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    K: np.ndarray
    L: np.ndarray | None
    Ts: float
    form: str = "delayed"
    M: np.ndarray | None = None
    reference: str = "none"
    Nbar: np.ndarray | None = None
    u_min: np.ndarray | None = None
    u_max: np.ndarray | None = None

    kind = "obs"

    @staticmethod
    def state_feedback(A, B, C, D, K, Ts: float, reference: str = "none", Nbar=None, u_min=None,
                       u_max=None) -> ObserverSpec:
        """Tam durum geri beslemesi (tüm durumlar ölçülür, gözlemci yok): u = −K·x (+ Nbar·r ya da LQI:
        K = [Kx Ki], u = −Kx·x − Ki·xi, xi[n+1] = xi[n] + Ts·(r − C·x − D·u)). MATLAB dlqr / lqi kazançlarıyla."""
        return ObserverSpec(A, B, C, D, K, None, Ts, form="state", reference=reference, Nbar=Nbar, u_min=u_min,
                            u_max=u_max)

    @property
    def is_state(self) -> bool:
        return self.form == "state"

    @property
    def dims(self) -> tuple[int, int, int]:
        A, B, C = np.atleast_2d(self.A), np.atleast_2d(self.B), np.atleast_2d(self.C)
        return A.shape[0], B.shape[1], C.shape[0]

    @property
    def n_meas(self) -> int:
        """Ölçüm girişi sayısı: 'state' biçiminde nx (durum), aksi ny."""
        nx, _, ny = self.dims
        return nx if self.is_state else ny

    @property
    def n_ref(self) -> int:
        return self.dims[2] if self.reference != "none" else 0

    def matrices(self) -> dict[str, np.ndarray]:
        """Doğrulanmış matrisler: A, B, C, D, Kx, L (gözlemcili), M (current), Ki (lqi), Nbar (nbar), umin, umax."""
        if not (_finite(self.Ts) and self.Ts > 0):
            raise ControlError("Durum geri beslemeli kontrolcü ayrık olmalı: Ts > 0")
        if self.form not in OBS_FORMS:
            raise ControlError("Kestirici biçimi 'delayed', 'current' ya da 'state' (gözlemcisiz) olmalı")
        if self.reference not in ("none", "nbar", "lqi"):
            raise ControlError("Referans biçimi 'none', 'nbar' ya da 'lqi' olmalı")
        A = _mat(self.A, "A")
        nx = A.shape[0]
        _mat(A, "A", nx, nx)
        B = _mat(self.B, "B", nx)
        nu = B.shape[1]
        C = _mat(self.C, "C", None, nx)
        ny = C.shape[0]
        D = _mat(self.D, "D", ny, nu) if self.D is not None and np.size(self.D) else np.zeros((ny, nu))
        Kall = _mat(self.K, "K", nu)
        out = {"A": A, "B": B, "C": C, "D": D}
        if not self.is_state:
            if self.L is None:
                raise ControlError("Gözlemcili kontrolcü için kestirici kazancı L gerekli (tüm durumlar ölçülüyorsa "
                                   "'state' biçimini kullanın)")
            out["L"] = _mat(self.L, "L", nx, ny)
        L = out.get("L")
        if self.reference == "lqi":
            if Kall.shape[1] != nx + ny:
                raise ControlError(f"LQI kazancı {nu}×{nx + ny} olmalı (K = [Kx Ki]); verilen {Kall.shape}")
            out["Kx"], out["Ki"] = Kall[:, :nx], Kall[:, nx:]
        else:
            if Kall.shape[1] != nx:
                raise ControlError(f"Durum geri besleme kazancı K {nu}×{nx} olmalı; verilen {Kall.shape}")
            out["Kx"] = Kall
        if self.form == "current":
            if np.any(D != 0):
                raise ControlError("'current' kestiricili kontrolcüde bitkinin doğrudan geçişi (D) sıfır olmalı "
                                   "(u, aynı örnekteki ölçüme cebirsel olarak bağlanır); 'delayed' biçimini kullanın")
            if self.M is not None:
                out["M"] = _mat(self.M, "M", nx, ny)
            else:
                try:
                    out["M"] = np.linalg.solve(A, L)
                except np.linalg.LinAlgError:
                    raise ControlError("'current' biçimi için A tekil: yenilik kazancı M'yi açıkça verin") from None
        if self.reference == "nbar":
            if self.Nbar is not None:
                out["Nbar"] = _mat(self.Nbar, "Nbar", nu, ny)
            else:
                out["Nbar"] = nbar(A, B, C, D, out["Kx"])
        for key, default in (("u_min", -np.inf), ("u_max", np.inf)):
            v = getattr(self, key)
            arr = np.full(nu, default) if v is None else np.broadcast_to(np.asarray(v, float), (nu,)).copy()
            if np.any(np.isnan(arr)):
                raise ControlError("Çıkış sınırları NaN olamaz")
            out[key] = arr
        if np.any(out["u_min"] >= out["u_max"]):
            raise ControlError("Her kanalda çıkış alt sınırı üst sınırdan küçük olmalı")
        return out

    def validate(self) -> None:
        self.matrices()


def nbar(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Birim DC kazancı için referans ölçeği: Nbar = [(C − D·K)(I − A + B·K)⁻¹·B + D]⁻¹ (ayrık)."""
    nx = A.shape[0]
    try:
        G = (C - D @ K) @ np.linalg.solve(np.eye(nx) - A + B @ K, B) + D
    except np.linalg.LinAlgError:
        raise ControlError("Kapalı çevrimin z = 1'de kutbu var: Nbar hesaplanamaz (LQI kullanın)") from None
    if G.shape[0] != G.shape[1]:
        warnings.warn("Nbar: çıkış ve giriş sayısı farklı; en küçük kareler (sözde ters) kullanıldı", ControlWarning,
                      stacklevel=2)
        return np.linalg.pinv(G)
    if np.linalg.cond(G) > 1e12:
        raise ControlError("Kapalı çevrim DC kazancı tekil: Nbar hesaplanamaz (LQI kullanın)")
    return np.linalg.inv(G)


@dataclass(frozen=True, eq=False)
class KalmanSpec:
    """Kararlı durum Kalman süzgeci / gözlemci (MATLAB kalman / estim): girişler [u; y], çıkışlar [ŷ; x̂].

    current: x̂[n|n] = x̂[n|n−1] + M·ν, ν = y − C x̂[n|n−1] − D u; ŷ = C x̂[n|n] + D u — ya da My verilirse
    (MATLAB kalman'ın 6. çıktısı; ölçüm gürültüsü süreç gürültüsüyle ilişkili / H ≠ 0 iken My ≠ C·M)
    ŷ = C x̂[n|n−1] + D u + My·ν;
    delayed: çıktı x̂[n|n−1], ŷ = C x̂[n|n−1] + D u; tahmin: x̂[n+1|n] = A x̂[n|n−1] + B u + L·ν."""
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    L: np.ndarray
    Ts: float
    M: np.ndarray | None = None
    form: str = "current"
    My: np.ndarray | None = None

    kind = "kf"

    def matrices(self) -> dict[str, np.ndarray]:
        if not (_finite(self.Ts) and self.Ts > 0):
            raise ControlError("Kalman süzgeci ayrık olmalı: Ts > 0")
        if self.form not in ("delayed", "current"):
            raise ControlError("Kestirici biçimi 'delayed' ya da 'current' olmalı")
        A = _mat(self.A, "A")
        nx = A.shape[0]
        _mat(A, "A", nx, nx)
        C = _mat(self.C, "C", None, nx)
        ny = C.shape[0]
        B = np.zeros((nx, 0)) if self.B is None or np.size(self.B) == 0 else _mat(self.B, "B", nx)
        nu = B.shape[1]
        D = np.zeros((ny, nu)) if self.D is None or np.size(self.D) == 0 else _mat(self.D, "D", ny, nu)
        out = {"A": A, "B": B, "C": C, "D": D, "L": _mat(self.L, "L", nx, ny)}
        if self.form == "current":
            if self.M is not None:
                out["M"] = _mat(self.M, "M", nx, ny)
            else:
                try:
                    out["M"] = np.linalg.solve(A, out["L"])
                except np.linalg.LinAlgError:
                    raise ControlError("'current' biçimi için A tekil: yenilik kazancı M'yi açıkça verin") from None
            if self.My is not None:
                out["My"] = _mat(self.My, "My", ny, ny)
        return out

    @property
    def dims(self) -> tuple[int, int, int]:
        nx = np.atleast_2d(self.A).shape[0]
        nu = 0 if self.B is None or np.size(self.B) == 0 else np.atleast_2d(self.B).shape[1]
        return nx, nu, np.atleast_2d(self.C).shape[0]

    def validate(self) -> None:
        self.matrices()


Spec = PidSpec | LtiSpec | ObserverSpec | KalmanSpec


# ============================================================================ nesneden tanıma (pid / ss / tf / kazanç)
def _attr(obj, *names, default=None):
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            return v() if callable(v) and not isinstance(v, (np.ndarray, float, int)) else v
    return default


def _delay_total(obj) -> int:
    """Ayrık modelin toplam G/Ç gecikmesi (örnek; SISO). Tamsayı değilse hata."""
    total = 0.0
    for name in ("input_delay", "output_delay", "io_delay", "InputDelay", "OutputDelay", "IODelay"):
        v = getattr(obj, name, None)
        if v is None:
            continue
        arr = np.asarray(v, float).ravel()
        if arr.size:
            total += float(arr.max())
    if total != int(total):
        raise ControlError("Ayrık kontrolcünün gecikmesi tamsayı örnek olmalı")
    return int(total)


def controller_spec(obj: Any, Ts: float | None = None, **pid_options) -> Spec:
    """Kontrolcü nesnesini (pid / pidstd / pid2, ss, tf, zpk, kazanç dizisi ya da bu modülün *Spec türleri) C kodu
    tanımına çevirir. pid_options: PID'ye eklenecek uygulama seçenekleri (u_min, u_max, anti_windup, Kb, Kt, …)."""
    if isinstance(obj, (PidSpec, LtiSpec, ObserverSpec, KalmanSpec)):
        if isinstance(obj, PidSpec) and pid_options:
            obj = replace(obj, **pid_options)
        obj.validate()
        return obj
    kest = getattr(obj, "kest", None)        # MATLAB kalman / kalmd dönüşü (kest, L, P, Mx, Z): kestirici modeli
    if kest is not None and all(hasattr(kest, n) for n in ("A", "B", "C", "D")):
        obj = kest
    if isinstance(obj, (int, float, np.ndarray, list, tuple)) and not hasattr(obj, "Ts"):
        if Ts is None or not Ts > 0:
            raise ControlError("Statik kazanç için örnekleme zamanı (çağrı periyodu) verilmeli")
        spec = LtiSpec.gain(obj, Ts)
        spec.validate()
        return spec
    ts = _attr(obj, "Ts", "dt")
    ts = float(ts) if ts is not None else (Ts if Ts is not None else 0.0)
    if not ts > 0:
        raise ControlError("Kontrolcü sürekli zamanlı: C kodu için önce ayrıklaştırın (c2d) ya da ayrık tasarlayın")
    if hasattr(obj, "Kp") and (hasattr(obj, "Ki") or hasattr(obj, "Ti")):
        f_i = str(_attr(obj, "IFormula", "i_formula", "iformula", default="ForwardEuler"))
        f_d = str(_attr(obj, "DFormula", "d_formula", "dformula", default="ForwardEuler"))
        b = float(_attr(obj, "b", default=1.0))
        c = float(_attr(obj, "c", default=1.0))
        two = hasattr(obj, "b") or hasattr(obj, "c")
        if hasattr(obj, "Ti"):
            spec = PidSpec.standard(float(obj.Kp), float(obj.Ti), float(_attr(obj, "Td", default=0.0)),
                                    float(_attr(obj, "N", default=math.inf)), Ts=ts, i_formula=f_i, d_formula=f_d,
                                    b=b, c=c, two_dof=two, **pid_options)
        else:
            kw = dict(Kp=float(obj.Kp), Ki=float(_attr(obj, "Ki", default=0.0)),
                      Kd=float(_attr(obj, "Kd", default=0.0)), Tf=float(_attr(obj, "Tf", default=0.0)), Ts=ts,
                      i_formula=f_i, d_formula=f_d, b=b, c=c, two_dof=two)
            kw.update(pid_options)
            spec = PidSpec(**kw)
        spec.validate()
        return spec
    if all(hasattr(obj, n) for n in ("A", "B", "C", "D")):
        if _delay_total(obj):
            raise ControlError("Durum uzayı kontrolcüsünde gecikme var: önce durum olarak ekleyin (absorb_delay)")
        spec = LtiSpec.from_ss(obj.A, obj.B, obj.C, obj.D, ts)
    elif hasattr(obj, "num") and hasattr(obj, "den"):
        num, den = obj.num, obj.den
        if isinstance(num, (list, tuple)) and num and isinstance(num[0], (list, tuple)):
            if len(num) != 1 or len(num[0]) != 1:
                raise ControlError("Çok girişli / çok çıkışlı aktarım fonksiyonu: önce durum uzayına çevirin (ss)")
            num, den = num[0][0], den[0][0]
        num = np.atleast_1d(np.asarray(num, float))
        den = np.atleast_1d(np.asarray(den, float))
        if num.ndim > 1:
            if num.shape[:2] != (1, 1) and num.size != max(num.shape):
                raise ControlError("Çok girişli / çok çıkışlı aktarım fonksiyonu: önce durum uzayına çevirin (ss)")
            num, den = num.ravel(), den.ravel()
        d = _delay_total(obj)
        if d:
            den = np.concatenate([den, np.zeros(d)])
        spec = LtiSpec.from_tf(num, den, ts)
    elif all(hasattr(obj, n) for n in ("z", "p", "k")):
        z, p, k = obj.z, obj.p, obj.k
        if isinstance(z, (list, tuple)):
            if len(z) != 1 or len(z[0]) != 1:
                raise ControlError("Çok girişli / çok çıkışlı ZPK: önce durum uzayına çevirin (ss)")
            z, p = z[0][0], p[0][0]
        kk = float(np.asarray(k, float).ravel()[0])
        num = np.real_if_close(np.poly(np.asarray(z, complex))) * kk
        den = np.real_if_close(np.poly(np.asarray(p, complex)))
        d = _delay_total(obj)
        if d:
            den = np.concatenate([den, np.zeros(d)])
        spec = LtiSpec.from_tf(np.real(num), np.real(den), ts)
    else:
        raise ControlError("Bu nesneden C kodu üretilemez: PID, durum uzayı, aktarım fonksiyonu ya da kazanç "
                           "bekleniyor")
    spec.validate()
    return spec


# ============================================================================ SOS (SISO) ve dengeleme
def _sos_from_tf(num: np.ndarray, den: np.ndarray) -> tuple[np.ndarray, float]:
    """(sos [nsec × 5: b0 b1 b2 a1 a2], doğrudan kazanç). nsec = 0 → yalnız kazanç (statik)."""
    from scipy import signal

    num = np.trim_zeros(np.asarray(num, float), "f")
    den = np.trim_zeros(np.asarray(den, float), "f")
    if num.size == 0:
        return np.zeros((0, 5)), 0.0
    num, den = num / den[0], den / den[0]
    nd, nn = den.size - 1, num.size - 1
    if nd == 0:
        return np.zeros((0, 5)), float(num[0])
    if nd > 12:
        warnings.warn(f"{nd}. derece aktarım fonksiyonunun kökleri kötü koşullu olabilir; durum uzayı "
                      "gerçeklemesi daha güvenilirdir", ControlWarning, stacklevel=3)
    rel = nd - nn
    z = np.concatenate([np.roots(num), np.zeros(rel)])     # z^rel·H(z): eşit sayıda sıfır / kutup
    p = np.roots(den)
    sos = signal.zpk2sos(z, p, float(num[0]), pairing="nearest")
    for _ in range(rel):                                    # H = z^-rel · (z^rel·H): rel gecikme bölümlere dağıtılır
        idx = next((i for i in range(sos.shape[0]) if sos[i, 2] == 0.0), None)
        if idx is None:
            raise ControlError("İkinci derece bölümlere ayırma başarısız (gecikme yerleştirilemedi)")
        sos[idx, :3] = [0.0, sos[idx, 0], sos[idx, 1]]
    out = np.column_stack([sos[:, 0:3] / sos[:, [3]], sos[:, 4:6] / sos[:, [3]]])
    # doğrulama: birim çember üzerinde frekans cevabı tf ile aynı mı (kötü koşullu köklere karşı)
    w = np.exp(1j * np.linspace(0.01, np.pi - 0.01, 37))
    h_tf = np.polyval(num, w) / np.polyval(den, w)
    h = np.ones_like(w)
    for b0, b1, b2, a1, a2 in out:
        h *= (b0 * w * w + b1 * w + b2) / (w * w + a1 * w + a2)
    err = np.max(np.abs(h - h_tf)) / max(np.max(np.abs(h_tf)), 1e-300)
    if not err < 1e-7:
        raise ControlError(f"İkinci derece bölümler aktarım fonksiyonunu yeterli doğrulukla temsil etmiyor "
                           f"(göreli fark {err:.1e}); durum uzayı gerçeklemesini seçin")
    return out, 0.0


def _balanced(A, B, C):
    """2'nin kuvvetleriyle köşegen benzerlik (tam aritmetik: G/Ç davranışı değişmez)."""
    from scipy.linalg import matrix_balance

    if A.shape[0] == 0:
        return A, B, C
    _, (scale, _) = matrix_balance(A, permute=False, separate=True)
    return (A * scale[None, :]) / scale[:, None], B / scale[:, None], C * scale[None, :]


def _lti_realization(spec: LtiSpec) -> dict:
    if spec.use_sos():
        if spec.is_tf:
            num, den = spec.num, spec.den
        else:
            from scipy import signal
            if spec.nx == 0:
                num, den = np.atleast_1d(float(np.atleast_2d(spec.D)[0, 0])), np.array([1.0])
            else:
                num, den = signal.ss2tf(np.atleast_2d(spec.A), np.atleast_2d(spec.B), np.atleast_2d(spec.C),
                                        np.atleast_2d(spec.D))
                num = num[0]
        sos, _ = _sos_from_tf(num, den)
        if sos.shape[0] == 0:
            nn, dd = np.trim_zeros(np.asarray(num, float), "f"), np.trim_zeros(np.asarray(den, float), "f")
            k = float(nn[0] / dd[0]) if nn.size else 0.0
            return {"mode": "ss", "A": np.zeros((0, 0)), "B": np.zeros((0, 1)), "C": np.zeros((1, 0)),
                    "D": np.array([[k]])}
        return {"mode": "sos", "sos": sos}
    if spec.is_tf:
        from scipy import signal
        num = np.trim_zeros(np.asarray(spec.num, float), "f")
        den = np.trim_zeros(np.asarray(spec.den, float), "f")
        if den.size == 1:
            A, B, C, D = np.zeros((0, 0)), np.zeros((0, 1)), np.zeros((1, 0)), np.array([[num[0] / den[0]
                                                                                          if num.size else 0.0]])
        else:
            A, B, C, D = signal.tf2ss(num, den)
    else:
        A, B, C, D = (np.atleast_2d(np.asarray(m, float)) for m in (spec.A, spec.B, spec.C, spec.D))
        if spec.nx == 0:
            A, B, C = np.zeros((0, 0)), np.zeros((0, D.shape[1])), np.zeros((D.shape[0], 0))
    A, B, C = _balanced(A, B, C)
    return {"mode": "ss", "A": A, "B": B, "C": C, "D": np.atleast_2d(D)}


# ============================================================================ Python başvurusu (C ile aynı sıra)
def _simulate_pid(sp: PidSpec, inputs: np.ndarray) -> np.ndarray:
    """inputs: (N, 2) [r, y] ya da (N, 3) [r, y, tr] (izleme açık); çıkış (N, 1). C ile aynı işlem sırası.

    Simulink "Discrete PID Controller" yapısı (R2025b'den durum kaydıyla ölçüldü): anti-windup / izleme sinyalleri
    integratörün DURUMUYLA kurulan toplamdan (a = P + D + x_I) hesaplanır; çıkış ise integratörün tam çıkışını
    (x_I + β·Ts·w; geri Euler / yamukta doğrudan geçişli) kullanır. Böylece cebirsel döngü oluşmaz."""
    ts = sp.Ts
    bi = _ALPHA[sp.i_formula] * ts
    cd = sp.cd
    track = inputs.shape[1] >= 3 and sp.tracking
    lim = sp.has_limits
    hi, lo = sp.u_max, sp.u_min
    xi = xd = uprev = 0.0
    out = np.empty((inputs.shape[0], 1))
    for k, row in enumerate(inputs):
        r, y = float(row[0]), float(row[1])
        tr = float(row[2]) if track else 0.0
        if not (math.isfinite(r) and math.isfinite(y) and (not track or math.isfinite(tr))):
            out[k, 0] = uprev
            continue
        d = cd * (sp.Kd * (sp.c * r - y) - xd)
        vi = sp.Ki * (r - y)
        a = sp.Kp * (sp.b * r - y) + d + xi
        sa = (hi if a > hi else (lo if a < lo else a)) if lim else a
        w = vi
        if sp.anti_windup == "back-calculation":
            w += sp.Kb * (sa - a)
        if track:
            w += sp.Kt * (tr - sa)
        if sp.anti_windup == "clamping" and ((a > sa and w > 0.0) or (a < sa and w < 0.0)):
            w = 0.0
        u = a + bi * w if bi != 0.0 else a
        if lim:
            u = hi if u > hi else (lo if u < lo else u)
        xd += ts * d
        xi += ts * w
        uprev = u
        out[k, 0] = u
    return out


def _matvec_acc(M: np.ndarray, x: np.ndarray, acc: np.ndarray | None = None, sign: float = 1.0) -> np.ndarray:
    """C döngüleriyle aynı sırada (soldan sağa) birikimli matris-vektör çarpımı."""
    rows = M.shape[0]
    out = np.zeros(rows) if acc is None else acc.copy()
    for i in range(rows):
        s = out[i]
        for j in range(M.shape[1]):
            if sign > 0:
                s += M[i, j] * x[j]
            else:
                s -= M[i, j] * x[j]
        out[i] = s
    return out


def _simulate_lti(spec: LtiSpec, inputs: np.ndarray) -> np.ndarray:
    real = _lti_realization(spec)
    n = inputs.shape[0]
    if real["mode"] == "sos":
        sos = real["sos"]
        w = np.zeros((sos.shape[0], 2))
        out = np.empty((n, 1))
        prev = 0.0
        for k in range(n):
            v = float(inputs[k, 0])
            if not math.isfinite(v):
                out[k, 0] = prev
                continue
            for i, (b0, b1, b2, a1, a2) in enumerate(sos):
                o = b0 * v + w[i, 0]
                w[i, 0] = b1 * v - a1 * o + w[i, 1]
                w[i, 1] = b2 * v - a2 * o
                v = o
            prev = v
            out[k, 0] = v
        return out
    A, B, C, D = real["A"], real["B"], real["C"], real["D"]
    x = np.zeros(A.shape[0])
    out = np.empty((n, D.shape[0]))
    prev = np.zeros(D.shape[0])
    for k in range(n):
        u = inputs[k, :D.shape[1]].astype(float)
        if not np.all(np.isfinite(u)):
            out[k] = prev
            continue
        y = _matvec_acc(D, u, _matvec_acc(C, x))
        x = _matvec_acc(B, u, _matvec_acc(A, x))
        prev = y
        out[k] = y
    return out


def _sat_vec(u: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.where(u > hi, hi, np.where(u < lo, lo, u))


def _simulate_obs(spec: ObserverSpec, inputs: np.ndarray) -> np.ndarray:
    """inputs: (N, nm [+ ny]) [y (ya da 'state' biçiminde x), r]; çıkış (N, nu)."""
    m = spec.matrices()
    nx, nu, ny = spec.dims
    nm, nr = spec.n_meas, spec.n_ref
    xp, xi = np.zeros(nx), np.zeros(ny)
    uprev = np.zeros(nu)
    out = np.empty((inputs.shape[0], nu))
    lo, hi = m["u_min"], m["u_max"]
    for k in range(inputs.shape[0]):
        y = inputs[k, :nm].astype(float)
        r = inputs[k, nm:nm + nr].astype(float)
        if not (np.all(np.isfinite(y)) and np.all(np.isfinite(r))):
            out[k] = uprev
            continue
        if spec.is_state:
            xh = y                                   # ölçülen durum
        elif spec.form == "current":
            inn = _matvec_acc(m["C"], xp, y.copy(), sign=-1.0)
            xh = _matvec_acc(m["M"], inn, xp.copy())
        else:
            xh = xp
        u = _matvec_acc(m["Kx"], xh, np.zeros(nu), sign=-1.0)
        if spec.reference == "nbar":
            u = _matvec_acc(m["Nbar"], r, u)
        elif spec.reference == "lqi":
            u = _matvec_acc(m["Ki"], xi, u, sign=-1.0)
        u = _sat_vec(u, lo, hi)
        if spec.is_state:
            if spec.reference == "lqi":
                yv = _matvec_acc(m["D"], u, _matvec_acc(m["C"], y))      # y = C·x + D·u (model)
                xi = np.array([xi[i] + spec.Ts * (r[i] - yv[i]) for i in range(ny)])
        else:
            inn = _matvec_acc(m["D"], u, _matvec_acc(m["C"], xp, y.copy(), sign=-1.0), sign=-1.0)
            xp = _matvec_acc(m["L"], inn, _matvec_acc(m["B"], u, _matvec_acc(m["A"], xp)))
            if spec.reference == "lqi":
                xi = np.array([xi[i] + spec.Ts * (r[i] - y[i]) for i in range(ny)])
        uprev = u
        out[k] = u
    return out


def _simulate_kf(spec: KalmanSpec, inputs: np.ndarray) -> np.ndarray:
    """inputs: (N, nu + ny) [u, y]; çıkış (N, ny + nx) [ŷ, x̂] (MATLAB kalman çıkış sırası)."""
    m = spec.matrices()
    nx, nu, ny = spec.dims
    xp = np.zeros(nx)
    out = np.empty((inputs.shape[0], ny + nx))
    last = np.zeros(ny + nx)
    for k in range(inputs.shape[0]):
        u = inputs[k, :nu].astype(float)
        y = inputs[k, nu:nu + ny].astype(float)
        if not np.all(np.isfinite(u)):
            out[k] = last
            continue
        if np.all(np.isfinite(y)):
            inn = _matvec_acc(m["D"], u, _matvec_acc(m["C"], xp, y.copy(), sign=-1.0), sign=-1.0)
        else:
            inn = np.zeros(ny)
        xh = _matvec_acc(m["M"], inn, xp.copy()) if spec.form == "current" else xp.copy()
        if "My" in m:                    # ŷ[n|n] = C·x̂[n|n−1] + D·u + My·ν (MATLAB kalman, H ≠ 0)
            yh = _matvec_acc(m["My"], inn, _matvec_acc(m["D"], u, _matvec_acc(m["C"], xp)))
        else:
            yh = _matvec_acc(m["D"], u, _matvec_acc(m["C"], xh))
        xp = _matvec_acc(m["L"], inn, _matvec_acc(m["B"], u, _matvec_acc(m["A"], xp)))
        last = np.concatenate([yh, xh])
        out[k] = last
    return out


def simulate(spec: Spec, inputs: np.ndarray) -> np.ndarray:
    """Üretilen C ile aynı algoritmanın Python başvurusu. inputs (N, girişler) → (N, çıkışlar).

    Girişler: pid [r, y(, tr)], lti [u…], obs [y…(, r…)], kf [u…, y…]; çıkışlar: pid [u], lti [y…], obs [u…],
    kf [ŷ…, x̂…]."""
    inputs = np.atleast_2d(np.asarray(inputs, float))
    spec.validate()
    if isinstance(spec, PidSpec):
        if inputs.shape[1] == 1:
            inputs = np.column_stack([inputs[:, 0], np.zeros(inputs.shape[0])])
        return _simulate_pid(spec, inputs)
    if isinstance(spec, LtiSpec):
        return _simulate_lti(spec, inputs)
    if isinstance(spec, ObserverSpec):
        return _simulate_obs(spec, inputs)
    return _simulate_kf(spec, inputs)


# ============================================================================ seçenek denetimi
_RESERVED_MACRO = re.compile(r"^(E[0-9A-Z]|LC_|SIG|PRI|SCN|U?INT|FLT|DBL|LDBL|TIME_|ATOMIC_|MATH_|FP_)")
_RESERVED_FUNC = re.compile(r"^(is|to|str|mem|wcs)[a-z]")


def check_prefix(prefix: str) -> None:
    """Önek C'de saklı / standart kütüphaneyle çakışan ad üretmemeli."""
    if not re.fullmatch(r"[A-Za-z](?:_?[A-Za-z0-9])*", prefix or ""):
        raise ControlError("Önek harfle başlamalı; harf, rakam ve tek alt çizgi içerebilir (başta / sonda ya da çift "
                           "alt çizgi C'de saklı adlar üretir: _x, a__b)")
    if len(prefix) > PREFIX_MAX:
        raise ControlError(f"Önek en çok {PREFIX_MAX} karakter olmalı (dış adlar 31 karakteri aşmasın)")
    if _RESERVED_MACRO.match(prefix.upper()):
        raise ControlError(f"'{prefix}' öneki standart C başlıklarına ayrılmış makro adları üretir (E…, INT…, SIG…, "
                           "PRI…, LC_…, FLT…): başka bir önek seçin")
    if _RESERVED_FUNC.match(prefix.lower()):
        raise ControlError(f"'{prefix}' öneki C kütüphanesine ayrılmış fonksiyon adları üretir (is…, to…, str…, "
                           "mem…): başka bir önek seçin")


def check_options(opts: ControlCodegenOptions) -> None:
    if opts.real not in ("float", "double"):
        raise ControlError("Sayı türü 'float' ya da 'double' olmalı")
    profile(opts.mcu)
    if opts.compiler not in COMPILERS:
        raise ControlError(f"Bilinmeyen derleyici: {opts.compiler!r}")
    if opts.opt not in OPT_LEVELS:
        raise ControlError(f"Optimizasyon düzeyi {', '.join(OPT_LEVELS)} içinden olmalı")
    check_prefix(opts.prefix)
    if not opts.clock_mhz > 0 or (opts.fs_hz is not None and not opts.fs_hz > 0):
        raise ControlError("Saat frekansı ve çağrı hızı pozitif olmalı")


def _rate(spec: Spec, opts: ControlCodegenOptions) -> float:
    return float(opts.fs_hz) if opts.fs_hz else 1.0 / spec.Ts


# ============================================================================ C yazımı yardımcıları
def _lit(v: float) -> str:
    return f"SPT_C({_c_float(float(v))})"


def _fill(text: str, **kw: str) -> str:
    for k, v in kw.items():
        text = text.replace("{{" + k + "}}", v)
    left = re.findall(r"\{\{[A-Z_0-9]+\}\}", text)
    assert not left, left
    return text


def _prefixed(text: str, prefix: str) -> str:
    if prefix != "spt":
        text = re.sub(r"\bspt_", prefix.lower() + "_", text)
        text = re.sub(r"\bSPT_", prefix.upper() + "_", text)
    return text


def _wrap_c_line(line: str, width: int) -> list[str]:
    """Uzun C satırını böler: yorumda boşluktan, kodda ", " / " || " / " && " noktalarından (açık parantez hizalı)."""
    out: list[str] = []
    while len(line) > width:
        body = line.lstrip()
        lead = line[:len(line) - len(body)]
        if body.startswith(("*", "/*")):                     # yorum satırı
            cut = line.rfind(" ", len(lead) + 3, width + 1)
            if cut < 0:
                break
            out.append(line[:cut].rstrip())
            line = (lead + " *   " if body.startswith("*") else lead + "   ") + line[cut + 1:].lstrip()
            continue
        stack: list[int] = []
        best = None
        for i, ch in enumerate(line[:width]):
            if ch == "(":
                stack.append(i)
            elif ch == ")" and stack:
                stack.pop()
            elif line.startswith(", ", i):
                best = (i + 1, i + 2, stack[-1] + 1 if stack else len(lead) + 8)
            elif (line.startswith(" || ", i) or line.startswith(" && ", i)) and stack:
                best = (i, i + 1, stack[-1] + 1)
        if best is None:
            cmt = line.find("/*")                            # kod + sondaki yorum: yorumu alt satıra al
            if cmt > len(lead):
                out.append(line[:cmt].rstrip())
                line = lead + line[cmt:]
                continue
            break
        head, start, indent = best
        out.append(line[:head].rstrip())
        line = " " * indent + line[start:].lstrip()
    out.append(line)
    return out


def _realign(lines: list[str]) -> list[str]:
    """Şablondaki çok satırlı imzaların devam satırlarını (önek uzunluğu değişince) açık parantezin hizasına getirir."""
    out: list[str] = []
    for line in lines:
        prev = out[-1] if out else ""
        if prev.rstrip().endswith(",") and line.startswith(" ") and not line.lstrip().startswith(("*", "/*")):
            stack = []
            for i, ch in enumerate(prev):
                if ch == "(":
                    stack.append(i)
                elif ch == ")" and stack:
                    stack.pop()
            if stack:
                line = " " * (stack[-1] + 1) + line.lstrip()
        out.append(line)
    return out


def _wrap_c(text: str, width: int = 120) -> str:
    lines = _realign(text.split("\n"))
    return "\n".join(part for line in lines for part in _wrap_c_line(line, width))


def _wrap_items(items: list[str], indent: int, width: int = 118) -> list[str]:
    lines, cur = [], ""
    for i, it in enumerate(items):
        piece = it + ("," if i < len(items) - 1 else "")
        if cur and len(cur) + 1 + len(piece) + indent > width:
            lines.append(" " * indent + cur)
            cur = piece
        else:
            cur = f"{cur} {piece}" if cur else piece
    if cur:
        lines.append(" " * indent + cur)
    return lines


def _c_matrix(name: str, M: np.ndarray, doc: str) -> str:
    M = np.atleast_2d(M)
    rows = []
    for i, row in enumerate(M):
        body = _wrap_items([_lit(v) for v in row], 8)
        rows.append("    {\n" + "\n".join(body) + "\n    }" + ("," if i < M.shape[0] - 1 else ""))
    return f"/** {doc} */\nstatic const SPT_REAL {name}[{M.shape[0]}][{M.shape[1]}] = {{\n" + "\n".join(rows) + "\n};\n"


def _c_vector(name: str, v: np.ndarray, doc: str) -> str:
    body = _wrap_items([_lit(x) for x in np.ravel(v)], 4)
    return f"/** {doc} */\nstatic const SPT_REAL {name}[{np.size(v)}] = {{\n" + "\n".join(body) + "\n};\n"


_BANNER = "Signal Processing Toolbox tarafından üretildi (Kontrol Tasarımı). Ayrıntılı kullanım: KULLANIM_@PFX@.md"

_TYPES = r"""
/* ---- sayı türü ve ortak yardımcılar (aynı önekli kontrolcü başlıklarıyla paylaşılır) --------------------------- */
#ifndef SPT_CTRL_TYPES_H
#define SPT_CTRL_TYPES_H
#include <stddef.h>   /* NULL: kullanılmayan giriş işaretçileri (referans / bilinen giriş yoksa) */
#ifndef SPT_USE_DOUBLE
#define SPT_USE_DOUBLE {{USE_DOUBLE}}
#endif
#if defined(SPT_CUSTOM_REAL)
/* SPT_REAL, SPT_C ve SPT_REAL_MAX bu dosyadan önce tanımlanır (ör. ana makinedeki işlem sayacı testi). */
#elif SPT_USE_DOUBLE
#include <float.h>
/** Hesap türü: double (çift hassasiyet FPU'su olmayan çekirdeklerde yazılım kayan noktadır, yavaştır). */
typedef double SPT_REAL;
#define SPT_C(x) (x)
#define SPT_REAL_MAX DBL_MAX
#else
#include <float.h>
/** Hesap türü: float. Sabitler SPT_C() ile float sabiti olur (istemsiz double yükseltmesi yok). */
typedef float SPT_REAL;
#define SPT_C(x) (x##f)
#define SPT_REAL_MAX FLT_MAX
#endif

#if defined(__cplusplus) && !defined(SPT_CUSTOM_REAL)
#define SPT_CTRL_BEGIN_DECLS extern "C" {
#define SPT_CTRL_END_DECLS }
#else
#define SPT_CTRL_BEGIN_DECLS
#define SPT_CTRL_END_DECLS
#endif

/** Sonlu mu (NaN / ±sonsuz değil). -ffast-math / -ffinite-math-only bu denetimi kaldırır: kullanmayın. */
static inline int spt_ctrl_finite(SPT_REAL v)
{
    return v == v && v <= SPT_REAL_MAX && v >= -SPT_REAL_MAX;
}
#endif /* SPT_CTRL_TYPES_H */
"""


# ============================================================================ PID şablonu
_PID_H = r"""/**
 * @file spt_pid.h
 * @brief {{TITLE}} — ayrık PID (MATLAB pid / pid2 ve Simulink Discrete PID Controller anlamıyla).
 *
 * {{BANNER}}
 *
 * Kontrol yasası (paralel eşdeğer; {{FORM}}):
 *   u = Kp·(b·r − y) + Ki·IF(z)·(r − y) + Kd / (Tf + DF(z))·(c·r − y)
 *   IF(z) = {{IF_TEXT}}, DF(z) = {{DF_TEXT}}, Ts = {{TS_TEXT}} s (sabit; kontrolcü bu periyotla çağrılmalı).
 * Doyum / anti-windup (Simulink tanımı): {{AW_TEXT}}.
 * İzleme (tracking) modu: spt_pid_step_track(); integratör girişine Kt·(tr − sat(a)) eklenir (yumuşak geçiş);
 * a = Kp·(b·r − y) + D + x_I, integratörün durumuyla doyum öncesi toplam (Simulink yapısı).
 * Tek serbestlik dereceli kullanım: spt_pid_step(&s, e, 0) (b = c = 1 iken u = C(z)·e).
 * Sonlu olmayan giriş (NaN / ±sonsuz): durum değişmez, son çıkış döner.
 */
#ifndef SPT_PID_H
#define SPT_PID_H
{{TYPES}}
/* ---- derleme anı yapılandırması (yapıyı değiştirir: yeniden üretin) ------------------------------------------- */
#define SPT_PID_TS SPT_C({{TS}})        /**< örnekleme zamanı (s) */
#define SPT_PID_IF {{IF}}                    /**< integratör formülü: 0 ileri Euler, 1 geri Euler, 2 yamuk */
#define SPT_PID_DF {{DF}}                    /**< türev süzgeci formülü: 0 ileri Euler, 1 geri Euler, 2 yamuk */
#define SPT_PID_AW {{AW}}                    /**< anti-windup: 0 yok, 1 geri hesaplama, 2 sıkıştırma */
#define SPT_PID_LIMIT {{LIMIT}}                 /**< 1: çıkış doyumu etkin */

/* ---- varsayılan parametreler (#ifndef korumalı: derleyici komut satırından ezilebilir) --------------------------- */
{{DEFAULTS}}

/** PID durumu ve çalışma anı parametreleri (malloc yok; her kontrolcü için bir örnek). */
typedef struct {
    SPT_REAL kp;      /**< oransal kazanç (paralel) */
    SPT_REAL ki;      /**< integral kazancı (paralel, 1/s) */
    SPT_REAL kd;      /**< türev kazancı (paralel, s) */
    SPT_REAL tf;      /**< türev süzgeci zaman sabiti (s); değiştirmek için spt_pid_set_gains() */
    SPT_REAL b;       /**< oransal terim referans ağırlığı (2-DOF) */
    SPT_REAL c;       /**< türev terimi referans ağırlığı (2-DOF) */
    SPT_REAL u_min;   /**< çıkış alt sınırı (SPT_PID_LIMIT 1 iken) */
    SPT_REAL u_max;   /**< çıkış üst sınırı */
    SPT_REAL kb;      /**< geri hesaplama kazancı (SPT_PID_AW 1) */
    SPT_REAL kt;      /**< izleme kazancı (spt_pid_step_track) */
    SPT_REAL cd;      /**< türetilmiş: 1/(tf + α·Ts) — spt_pid_set_gains() hesaplar */
    SPT_REAL xi;      /**< integratör durumu (çıkış biriminde) */
    SPT_REAL xd;      /**< türev süzgeci durumu */
    SPT_REAL u;       /**< son (doyumlu) çıkış */
} spt_pid_t;

SPT_CTRL_BEGIN_DECLS
/** Parametreleri derleme anı varsayılanlarına getirir ve durumu sıfırlar. */
void spt_pid_init(spt_pid_t *s);
/** Durumu sıfırlar (integratör, süzgeç, son çıkış); parametreler korunur. */
void spt_pid_reset(spt_pid_t *s);
/** Paralel kazançları değiştirir (çalışma anında). Geçersizse (sonlu değil, tf < 0, kararsız / nedensel olmayan
 *  türev süzgeci) hiçbir şey değişmez ve 0 döner. Integratör durumu çıkış birimindedir: Ki değişimi sarsıntısızdır. */
int spt_pid_set_gains(spt_pid_t *s, SPT_REAL kp, SPT_REAL ki, SPT_REAL kd, SPT_REAL tf);
/** Standart biçim (MATLAB pidstd): Kp·(1 + 1/(Ti·IF) + Td/(Td/N + DF)). ti ≤ 0 → integral yok, n ≤ 0 → süzgeç yok. */
int spt_pid_set_standard(spt_pid_t *s, SPT_REAL kp, SPT_REAL ti, SPT_REAL td, SPT_REAL n);
/** Çıkış sınırları (u_min < u_max olmalı; değilse 0 döner). SPT_PID_LIMIT 0 ise etkisizdir. */
int spt_pid_set_limits(spt_pid_t *s, SPT_REAL u_min, SPT_REAL u_max);
/** Integratör durumunu ayarlar (ör. elle → otomatik geçişte son çıkışa göre ön yükleme). */
void spt_pid_set_integrator(spt_pid_t *s, SPT_REAL xi);
/** Bir örnek: r referans, y ölçüm → doyumlu çıkış u. Sabit Ts periyoduyla çağırın. */
SPT_REAL spt_pid_step(spt_pid_t *s, SPT_REAL r, SPT_REAL y);
/** İzleme modlu örnek: tr = uygulanan gerçek çıkış (elle kontrol / başka kontrolcü); integratör ona yaklaşır. */
SPT_REAL spt_pid_step_track(spt_pid_t *s, SPT_REAL r, SPT_REAL y, SPT_REAL tr);
SPT_CTRL_END_DECLS

#endif /* SPT_PID_H */
"""

_PID_C = r"""/**
 * @file spt_pid.c
 * @brief {{TITLE}} — ayrık PID uygulaması.
 *
 * {{BANNER}}
 */
#include "spt_pid.h"

#define SPT_PID_BI SPT_C({{BI}})     /* integratör doğrudan geçişi β·Ts (ileri 0, geri Ts, yamuk Ts/2) */
#define SPT_PID_AD SPT_C({{AD}})     /* türev süzgeci α·Ts */
#define SPT_PID_HALF_TS SPT_C({{HALF_TS}})

void spt_pid_reset(spt_pid_t *s)
{
    s->xi = SPT_C(0.0);
    s->xd = SPT_C(0.0);
    s->u = SPT_C(0.0);
}

int spt_pid_set_gains(spt_pid_t *s, SPT_REAL kp, SPT_REAL ki, SPT_REAL kd, SPT_REAL tf)
{
    SPT_REAL den = tf + SPT_PID_AD;
    if (!spt_ctrl_finite(kp) || !spt_ctrl_finite(ki) || !spt_ctrl_finite(kd) || !spt_ctrl_finite(tf)
        || tf < SPT_C(0.0)) {
        return 0;
    }
    if (kd != SPT_C(0.0) && !(den > SPT_PID_HALF_TS)) {
        return 0;   /* türev süzgeci kutbu 1 − Ts/(tf + α·Ts) birim çemberin içinde olmalı */
    }
    s->kp = kp;
    s->ki = ki;
    s->kd = kd;
    s->tf = tf;
    s->cd = (kd != SPT_C(0.0)) ? SPT_C(1.0) / den : SPT_C(0.0);
    return 1;
}

int spt_pid_set_standard(spt_pid_t *s, SPT_REAL kp, SPT_REAL ti, SPT_REAL td, SPT_REAL n)
{
    SPT_REAL ki = (ti > SPT_C(0.0)) ? kp / ti : SPT_C(0.0);
    SPT_REAL tf = (n > SPT_C(0.0)) ? td / n : SPT_C(0.0);
    if (ti != ti || n != n || !spt_ctrl_finite(td) || td < SPT_C(0.0)) {
        return 0;           /* NaN Ti / N, sonlu olmayan ya da negatif Td: değişiklik yok */
    }
    if (!spt_ctrl_finite(ti) && ti > SPT_C(0.0)) {
        ki = SPT_C(0.0);    /* Ti = +sonsuz: integral yok */
    }
    if (!spt_ctrl_finite(n) && n > SPT_C(0.0)) {
        tf = SPT_C(0.0);    /* N = +sonsuz: süzgeç yok */
    }
    return spt_pid_set_gains(s, kp, ki, kp * td, tf);
}

int spt_pid_set_limits(spt_pid_t *s, SPT_REAL u_min, SPT_REAL u_max)
{
    if (!(u_min < u_max) || u_min != u_min || u_max != u_max) {
        return 0;
    }
    s->u_min = u_min;
    s->u_max = u_max;
    return 1;
}

void spt_pid_set_integrator(spt_pid_t *s, SPT_REAL xi)
{
    if (spt_ctrl_finite(xi)) {
        s->xi = xi;
    }
}

void spt_pid_init(spt_pid_t *s)
{
    s->b = SPT_PID_B;
    s->c = SPT_PID_C;
    s->u_min = SPT_PID_U_MIN;
    s->u_max = SPT_PID_U_MAX;
    s->kb = SPT_PID_KB;
    s->kt = SPT_PID_KT;
    s->kp = s->ki = s->kd = s->tf = s->cd = SPT_C(0.0);
    (void)spt_pid_set_gains(s, SPT_PID_KP, SPT_PID_KI, SPT_PID_KD, SPT_PID_TF);
    spt_pid_reset(s);
}

/* Simulink "Discrete PID Controller" yapısı: anti-windup ve izleme sinyalleri integratörün DURUMUYLA kurulan
 * toplamdan (a = P + D + x_I) hesaplanır; çıkış integratörün tam çıkışını (x_I + β·Ts·w) kullanır. Geri Euler /
 * yamuk integratörde de cebirsel döngü yoktur. */
static SPT_REAL spt_pid_core(spt_pid_t *s, SPT_REAL r, SPT_REAL y, SPT_REAL tr, int track)
{
    SPT_REAL d, vi, a, sa, w, u;
    if (!spt_ctrl_finite(r) || !spt_ctrl_finite(y) || (track && !spt_ctrl_finite(tr))) {
        return s->u;    /* geçersiz giriş: durum değişmez, son çıkış tutulur */
    }
    d = s->cd * (s->kd * (s->c * r - y) - s->xd);
    vi = s->ki * (r - y);
    a = s->kp * (s->b * r - y) + d + s->xi;
{{BODY}}
    s->xd += SPT_PID_TS * d;
    s->xi += SPT_PID_TS * w;
    s->u = u;
    return u;
}

SPT_REAL spt_pid_step(spt_pid_t *s, SPT_REAL r, SPT_REAL y)
{
    return spt_pid_core(s, r, y, SPT_C(0.0), 0);
}

SPT_REAL spt_pid_step_track(spt_pid_t *s, SPT_REAL r, SPT_REAL y, SPT_REAL tr)
{
    return spt_pid_core(s, r, y, tr, 1);
}
"""

def _pid_body(sp: PidSpec, implicit: bool) -> str:
    """spt_pid_core gövdesinin yapılandırmaya göre özelleşmiş kısmı (Python _simulate_pid ile aynı sıra)."""
    lines = []
    if sp.has_limits:
        lines.append("    sa = (a > s->u_max) ? s->u_max : ((a < s->u_min) ? s->u_min : a);")
    else:
        lines.append("    sa = a;")
    lines.append("    w = vi;")
    if sp.anti_windup == "back-calculation":
        lines.append("    w += s->kb * (sa - a);         /* geri hesaplama */")
    lines += ["    if (track) {", "        w += s->kt * (tr - sa);    /* izleme */", "    }"]
    if sp.anti_windup == "clamping":
        lines += ["    /* sıkıştırma: doyumda aşma yönü integratör girişiyle aynı işaretliyse integrasyon durur */",
                  "    if ((a > sa && w > SPT_C(0.0)) || (a < sa && w < SPT_C(0.0))) {",
                  "        w = SPT_C(0.0);", "    }"]
    if implicit:
        lines.append("    u = a + SPT_PID_BI * w;        /* integratörün doğrudan geçişi β·Ts·w */")
        if sp.has_limits:
            lines.append("    u = (u > s->u_max) ? s->u_max : ((u < s->u_min) ? s->u_min : u);")
    else:
        lines.append("    u = sa;")
    return "\n".join(lines)


def _pid_defaults(sp: PidSpec) -> str:
    def lim(v: float, neg: bool) -> str:
        if math.isinf(v):
            return "(-SPT_REAL_MAX)" if neg else "SPT_REAL_MAX"
        return _lit(v)

    rows = [("KP", _lit(sp.Kp), "oransal kazanç Kp"), ("KI", _lit(sp.Ki), "integral kazancı Ki (1/s)"),
            ("KD", _lit(sp.Kd), "türev kazancı Kd (s)"), ("TF", _lit(sp.Tf), "türev süzgeci zaman sabiti Tf (s)"),
            ("B", _lit(sp.b), "2-DOF oransal referans ağırlığı b"),
            ("C", _lit(sp.c), "2-DOF türev referans ağırlığı c"),
            ("U_MIN", lim(sp.u_min, True), "çıkış alt sınırı"), ("U_MAX", lim(sp.u_max, False), "çıkış üst sınırı"),
            ("KB", _lit(sp.Kb), "geri hesaplama kazancı Kb (paralel eşdeğer)"),
            ("KT", _lit(sp.Kt), "izleme kazancı Kt (paralel eşdeğer)")]
    return "\n".join(f"/** {doc} */\n#ifndef SPT_PID_{key}\n#define SPT_PID_{key} {val}\n#endif"
                     for key, val, doc in rows)


def _pid_files(sp: PidSpec, title: str, real: str) -> list[GeneratedFile]:
    ts = sp.Ts
    bi = _ALPHA[sp.i_formula] * ts
    ad = _ALPHA[sp.d_formula] * ts
    body = _pid_body(sp, bi != 0.0)
    form = "standart (ideal) biçimden çevrildi" if sp.form == "standard" else "paralel biçim"
    if sp.has_limits:
        aw_text = (f"çıkış [{_c_float(sp.u_min)}, {_c_float(sp.u_max)}] aralığında doyurulur, anti-windup "
                   f"{_AW_TR[sp.anti_windup]}")
    else:
        aw_text = "doyum yok"

    def ftext(f: str) -> str:
        return {"ForwardEuler": "Ts/(z − 1)", "BackwardEuler": "Ts·z/(z − 1)",
                "Trapezoidal": "(Ts/2)·(z + 1)/(z − 1)"}[f]

    types = _fill(_TYPES, USE_DOUBLE="1" if real == "double" else "0")
    header = _fill(_PID_H, TITLE=title, BANNER=_BANNER, FORM=form, IF_TEXT=ftext(sp.i_formula),
                   DF_TEXT=ftext(sp.d_formula), TS_TEXT=f"{ts:g}", AW_TEXT=aw_text, TYPES=types, TS=_c_float(ts),
                   IF=str(FORMULAS.index(sp.i_formula)), DF=str(FORMULAS.index(sp.d_formula)),
                   AW=str(ANTI_WINDUP.index(sp.anti_windup)), LIMIT="1" if sp.has_limits else "0",
                   DEFAULTS=_pid_defaults(sp))
    source = _fill(_PID_C, TITLE=title, BANNER=_BANNER, BI=_c_float(bi), AD=_c_float(ad), HALF_TS=_c_float(0.5 * ts),
                   BODY=body)
    return [GeneratedFile("spt_pid.h", header), GeneratedFile("spt_pid.c", source)]


# ============================================================================ LTI şablonu
_LTI_H = r"""/**
 * @file spt_lti.h
 * @brief {{TITLE}} — ayrık LTI kontrolcü ({{REAL_TEXT}}).
 *
 * {{BANNER}}
 *
 * Ts = {{TS_TEXT}} s (sabit; kontrolcü bu periyotla çağrılmalı). Katsayılar .c dosyasında flash'ta (const) durur;
 * değiştirmek için yeniden üretin. Sonlu olmayan giriş: durum değişmez, son çıkış döner (0).
 */
#ifndef SPT_LTI_H
#define SPT_LTI_H
{{TYPES}}
#define SPT_LTI_TS SPT_C({{TS}})     /**< örnekleme zamanı (s) */
#define SPT_LTI_NU {{NU}}             /**< giriş sayısı */
#define SPT_LTI_NY {{NY}}             /**< çıkış sayısı */
{{EXTRA_DEFS}}
/** Kontrolcü durumu (malloc yok). */
typedef struct {
{{FIELDS}}
} spt_lti_t;

SPT_CTRL_BEGIN_DECLS
/** Durumu sıfırlar. */
void spt_lti_init(spt_lti_t *s);
{{API}}
SPT_CTRL_END_DECLS

#endif /* SPT_LTI_H */
"""

_LTI_SOS_C = r"""/**
 * @file spt_lti.c
 * @brief {{TITLE}} — ikinci derece bölümler (devrik doğrudan biçim II), kazanç ilk bölümde.
 *
 * {{BANNER}}
 */
#include "spt_lti.h"

{{SOS}}
void spt_lti_init(spt_lti_t *s)
{
    int i;
    for (i = 0; i < SPT_LTI_NSEC; ++i) {
        s->w[i][0] = SPT_C(0.0);
        s->w[i][1] = SPT_C(0.0);
    }
    s->y = SPT_C(0.0);
}

SPT_REAL spt_lti_step(spt_lti_t *s, SPT_REAL u)
{
    SPT_REAL v = u, o;
    int i;
    if (!spt_ctrl_finite(u)) {
        return s->y;
    }
    for (i = 0; i < SPT_LTI_NSEC; ++i) {
        const SPT_REAL *c = spt_lti_sos[i];
        o = c[0] * v + s->w[i][0];
        s->w[i][0] = c[1] * v - c[3] * o + s->w[i][1];
        s->w[i][1] = c[2] * v - c[4] * o;
        v = o;
    }
    s->y = v;
    return v;
}
"""


def _loop_matvec(dst: str, mat: str, vec: str, rows: str, cols: str, op: str, init: str | None,
                 ind: str = "    ") -> str:
    """dst[i] = init[i] (op) Σ mat[i][j]·vec[j] — C döngüsü (Python _matvec_acc ile aynı sıra)."""
    start = f"{init}" if init is not None else "SPT_C(0.0)"
    return (f"{ind}for (i = 0; i < {rows}; ++i) {{\n"
            f"{ind}    SPT_REAL acc = {start};\n"
            f"{ind}    for (j = 0; j < {cols}; ++j) {{\n"
            f"{ind}        acc {op}= {mat}[i][j] * {vec}[j];\n"
            f"{ind}    }}\n"
            f"{ind}    {dst}[i] = acc;\n"
            f"{ind}}}\n")


def _lti_files(spec: LtiSpec, title: str, real: str) -> list[GeneratedFile]:
    types = _fill(_TYPES, USE_DOUBLE="1" if real == "double" else "0")
    rz = _lti_realization(spec)
    ts = spec.Ts
    if rz["mode"] == "sos":
        sos = rz["sos"]
        sos_text = _c_matrix("spt_lti_sos", sos, "Bölüm katsayıları [b0, b1, b2, a1, a2]: H_i(z) = (b0 + b1 z⁻¹ + "
                             "b2 z⁻²) / (1 + a1 z⁻¹ + a2 z⁻²)")
        header = _fill(_LTI_H, TITLE=title, REAL_TEXT=f"{len(sos)} ikinci derece bölüm, SISO", BANNER=_BANNER,
                       TS_TEXT=f"{ts:g}", TYPES=types, TS=_c_float(ts), NU="1", NY="1",
                       EXTRA_DEFS=f"#define SPT_LTI_NSEC {len(sos)}            /**< ikinci derece bölüm sayısı */\n",
                       FIELDS="    SPT_REAL w[SPT_LTI_NSEC][2];   /**< bölüm durumları */\n"
                              "    SPT_REAL y;                    /**< son çıkış */",
                       API="/** Bir örnek: giriş u (ör. hata e = r − y) → çıkış. Sabit Ts periyoduyla çağırın. */\n"
                           "SPT_REAL spt_lti_step(spt_lti_t *s, SPT_REAL u);")
        source = _fill(_LTI_SOS_C, TITLE=title, BANNER=_BANNER, SOS=sos_text)
        return [GeneratedFile("spt_lti.h", header), GeneratedFile("spt_lti.c", source)]
    A, B, C, D = rz["A"], rz["B"], rz["C"], rz["D"]
    nx, nu, ny = A.shape[0], D.shape[1], D.shape[0]
    fields = ""
    if nx:
        fields += "    SPT_REAL x[SPT_LTI_NX];   /**< durum */\n"
    fields += "    SPT_REAL y[SPT_LTI_NY];   /**< son çıkış */"
    header = _fill(_LTI_H, TITLE=title, REAL_TEXT=f"durum uzayı, {nx} durum, {nu} giriş, {ny} çıkış", BANNER=_BANNER,
                   TS_TEXT=f"{ts:g}", TYPES=types, TS=_c_float(ts), NU=str(nu), NY=str(ny),
                   EXTRA_DEFS=f"#define SPT_LTI_NX {nx}             /**< durum sayısı */\n",
                   FIELDS=fields,
                   API="/** Bir örnek: u[SPT_LTI_NU] → y[SPT_LTI_NY]. Sonlu olmayan girişte durum değişmez, son çıkış\n"
                       " *  yazılır ve 0 döner; aksi 1. Sabit Ts periyoduyla çağırın. */\n"
                       "int spt_lti_step(spt_lti_t *s, const SPT_REAL u[SPT_LTI_NU], SPT_REAL y[SPT_LTI_NY]);")
    consts = ""
    if nx:
        consts += _c_matrix("spt_lti_a", A, "A (dengelenmiş)") + _c_matrix("spt_lti_b", B, "B") + \
            _c_matrix("spt_lti_c", C, "C")
    consts += _c_matrix("spt_lti_d", D, "D")
    body = "int spt_lti_step(spt_lti_t *s, const SPT_REAL u[SPT_LTI_NU], SPT_REAL y[SPT_LTI_NY])\n{\n"
    if nx:
        body += "    SPT_REAL xn[SPT_LTI_NX];\n"
    body += "    int i, j;\n"
    body += ("    for (i = 0; i < SPT_LTI_NU; ++i) {\n        if (!spt_ctrl_finite(u[i])) {\n"
             "            for (j = 0; j < SPT_LTI_NY; ++j) {\n                y[j] = s->y[j];\n            }\n"
             "            return 0;\n        }\n    }\n")
    if nx:
        body += _loop_matvec("y", "spt_lti_c", "s->x", "SPT_LTI_NY", "SPT_LTI_NX", "+", None)
        body += _loop_matvec("y", "spt_lti_d", "u", "SPT_LTI_NY", "SPT_LTI_NU", "+", "y[i]")
        body += _loop_matvec("xn", "spt_lti_a", "s->x", "SPT_LTI_NX", "SPT_LTI_NX", "+", None)
        body += _loop_matvec("xn", "spt_lti_b", "u", "SPT_LTI_NX", "SPT_LTI_NU", "+", "xn[i]")
        body += "    for (i = 0; i < SPT_LTI_NX; ++i) {\n        s->x[i] = xn[i];\n    }\n"
    else:
        body += _loop_matvec("y", "spt_lti_d", "u", "SPT_LTI_NY", "SPT_LTI_NU", "+", None)
    body += "    for (i = 0; i < SPT_LTI_NY; ++i) {\n        s->y[i] = y[i];\n    }\n    return 1;\n}\n"
    init = "void spt_lti_init(spt_lti_t *s)\n{\n    int i;\n"
    if nx:
        init += "    for (i = 0; i < SPT_LTI_NX; ++i) {\n        s->x[i] = SPT_C(0.0);\n    }\n"
    init += "    for (i = 0; i < SPT_LTI_NY; ++i) {\n        s->y[i] = SPT_C(0.0);\n    }\n}\n"
    source = (f"/**\n * @file spt_lti.c\n * @brief {title} — durum uzayı gerçeklemesi.\n *\n * {_BANNER}\n */\n"
              f"#include \"spt_lti.h\"\n\n{consts}\n{init}\n{body}")
    return [GeneratedFile("spt_lti.h", header), GeneratedFile("spt_lti.c", source)]


# ============================================================================ gözlemci / Kalman şablonu
_EST_H = r"""/**
 * @file spt_{{K}}.h
 * @brief {{TITLE}} — {{BRIEF}}
 *
 * {{BANNER}}
 *
{{DOC}}
 * Ts = {{TS_TEXT}} s (sabit). Matrisler .c dosyasında flash'ta (const); değiştirmek için yeniden üretin.
 */
#ifndef SPT_{{KU}}_H
#define SPT_{{KU}}_H
{{TYPES}}
#define SPT_{{KU}}_TS SPT_C({{TS}})   /**< örnekleme zamanı (s) */
#define SPT_{{KU}}_NX {{NX}}           /**< durum sayısı */
#define SPT_{{KU}}_NU {{NU}}           /**< {{NU_DOC}} */
#define SPT_{{KU}}_NY {{NY}}           /**< ölçüm sayısı */
{{EXTRA_DEFS}}
/** Durum (malloc yok). */
typedef struct {
{{FIELDS}}
} spt_{{K}}_t;

SPT_CTRL_BEGIN_DECLS
{{API}}
SPT_CTRL_END_DECLS

#endif /* SPT_{{KU}}_H */
"""


def _finite_check(arr: str, n: str, ind: str = "    ") -> str:
    return (f"{ind}for (i = 0; i < {n}; ++i) {{\n{ind}    if (!spt_ctrl_finite({arr}[i])) {{\n"
            f"{ind}        return 0;\n{ind}    }}\n{ind}}}\n")


def _obs_files(spec: ObserverSpec, title: str, real: str) -> list[GeneratedFile]:
    m = spec.matrices()
    nx, nu, ny = spec.dims
    ref = spec.reference
    nr = spec.n_ref
    cur = spec.form == "current"
    st = spec.is_state                       # gözlemcisiz: ölçüm girişi durumun kendisi
    mv, nmeas = ("x", "SPT_OBS_NX") if st else ("y", "SPT_OBS_NY")
    lim = bool(np.any(np.isfinite(m["u_min"])) or np.any(np.isfinite(m["u_max"])))
    types = _fill(_TYPES, USE_DOUBLE="1" if real == "double" else "0")
    doc = [" * Bitki modeli: x[n+1] = A·x[n] + B·u[n], y[n] = C·x[n] + D·u[n] (ayrık)."]
    if st:
        doc.append(" * Tüm durumlar ölçülür (gözlemci yok): girişler x[SPT_OBS_NX]" + (" ve r." if nr else "."))
        law = " * Kontrol yasası: u = −K·x"
    else:
        doc += [" * Kestirici (MATLAB estim / reg, '" + spec.form + "'):",
                " *   yenilik  ν = y − C·x̂[n|n−1] − D·u",
                " *   tahmin   x̂[n+1|n] = A·x̂[n|n−1] + B·u + L·ν"]
        if cur:
            doc.append(" *   güncel   x̂[n|n] = x̂[n|n−1] + M·ν   (u bu kestirimden hesaplanır; D = 0)")
        law = " * Kontrol yasası: u = −K·x̂" + ("[n|n]" if cur else "[n|n−1]")
    if ref == "nbar":
        law += " + Nbar·r   (Nbar: birim DC kazancı)"
    elif ref == "lqi":
        law += (" − Ki·xi,   xi[n+1] = xi[n] + Ts·(r − y)   (MATLAB lqi, ileri Euler"
                + ("; y = C·x + D·u modelden)" if st else ")"))
    doc.append(law)
    if lim:
        doc.append(" * Çıkış kanal başına [u_min, u_max] aralığında doyurulur; "
                   + ("LQI integratörü" if st else "kestirici") + " UYGULANAN u ile ilerler.")
    doc.append(" * Kullanım: her örnekte spt_obs_step() ya da (eyleyici doyumu / dış sınırlama varsa) "
               "spt_obs_output() → uygula → spt_obs_update(uygulanan u).")
    fields = [] if st else ["    SPT_REAL xp[SPT_OBS_NX];   /**< x̂[n|n−1] (tahmin) */"]
    if cur:
        fields.append("    SPT_REAL xc[SPT_OBS_NX];   /**< x̂[n|n] (son güncel kestirim) */")
    if ref == "lqi":
        fields.append("    SPT_REAL xi[SPT_OBS_NY];   /**< LQI integral durumu */")
    fields.append("    SPT_REAL u[SPT_OBS_NU];    /**< son çıkış */")
    if lim:
        fields += ["    SPT_REAL u_min[SPT_OBS_NU]; /**< çıkış alt sınırı (çalışma anında değiştirilebilir) */",
                   "    SPT_REAL u_max[SPT_OBS_NU]; /**< çıkış üst sınırı */"]
    rdecl = "const SPT_REAL r[SPT_OBS_NR]" if nr else "const SPT_REAL *r"
    mdecl = f"const SPT_REAL {mv}[{nmeas}]"
    zero = [] if st else ["x̂ = 0"]
    if ref == "lqi":
        zero.append("xi = 0")
    zero.append("son çıkış = 0")
    api = ["/** Durumu sıfırlar (" + ", ".join(zero) + ")" + (" ve sınırları varsayılana getirir" if lim else "")
           + ". */",
           "void spt_obs_init(spt_obs_t *s);",
           "/** Kontrol çıkışı (durum ilerlemez). r: " + ("referans [SPT_OBS_NR]" if nr else "kullanılmaz (NULL)")
           + f". Sonlu olmayan {mv} / r: son çıkış yazılır, 0 döner. */",
           f"int spt_obs_output(spt_obs_t *s, {mdecl}, {rdecl}, SPT_REAL u[SPT_OBS_NU]);",
           "/** " + ("LQI integratörünü" if st else "Kestiriciyi") + " bir örnek ilerletir; ua = bu örnekte gerçekten "
           "uygulanan çıkış. Geçersiz girişte durum değişmez, 0. */",
           f"int spt_obs_update(spt_obs_t *s, {mdecl}, {rdecl}, const SPT_REAL ua[SPT_OBS_NU]);",
           "/** output + update (u uygulandığı varsayılır). Sabit Ts periyoduyla çağırın. */",
           f"int spt_obs_step(spt_obs_t *s, {mdecl}, {rdecl}, SPT_REAL u[SPT_OBS_NU]);"]
    if not st:
        api += ["/** Kontrol yasasının kullandığı kestirim x̂ (" + ("x̂[n|n]" if cur else "x̂[n|n−1]") + "). */",
                "void spt_obs_state(const spt_obs_t *s, SPT_REAL xhat[SPT_OBS_NX]);"]
    extra = f"#define SPT_OBS_NR {nr}           /**< referans sayısı (0: referans yok) */\n"
    extra += f"#define SPT_OBS_LIMIT {1 if lim else 0}        /**< 1: çıkış doyumu */\n"
    extra += f"#define SPT_OBS_OBSERVER {0 if st else 1}     /**< 0: tüm durumlar ölçülür (gözlemci yok) */\n"
    brief = "tam durum geri beslemesi (gözlemcisiz)." if st else "gözlemcili durum geri beslemesi."
    header = _fill(_EST_H, K="obs", KU="OBS", TITLE=title, BRIEF=brief, BANNER=_BANNER,
                   DOC="\n".join(doc), TS_TEXT=f"{spec.Ts:g}", TYPES=types, TS=_c_float(spec.Ts), NX=str(nx),
                   NU=str(nu), NU_DOC="kontrol çıkışı sayısı", NY=str(ny), EXTRA_DEFS=extra,
                   FIELDS="\n".join(fields), API="\n".join(api))
    has_d = bool(np.any(m["D"] != 0))
    consts = ""
    if not st:
        consts += _c_matrix("spt_obs_a", m["A"], "A") + _c_matrix("spt_obs_b", m["B"], "B")
    if not st or ref == "lqi":
        consts += _c_matrix("spt_obs_c", m["C"], "C")
        if has_d:
            consts += _c_matrix("spt_obs_d", m["D"], "D")
    consts += _c_matrix("spt_obs_k", m["Kx"], "K (durum geri besleme)")
    if not st:
        consts += _c_matrix("spt_obs_l", m["L"], "L (tahmin kazancı)")
    if cur:
        consts += _c_matrix("spt_obs_m", m["M"], "M (yenilik kazancı)")
    if ref == "nbar":
        consts += _c_matrix("spt_obs_nbar", m["Nbar"], "Nbar (referans ölçeği)")
    if ref == "lqi":
        consts += _c_matrix("spt_obs_ki", m["Ki"], "Ki (LQI integral kazancı)")
    if lim:
        def lim_lit(v: float, neg: bool) -> str:
            return ("(-SPT_REAL_MAX)" if neg else "SPT_REAL_MAX") if math.isinf(v) else _lit(v)
        consts += ("/** Varsayılan çıkış sınırları */\nstatic const SPT_REAL spt_obs_umin[SPT_OBS_NU] = {"
                   + ", ".join(lim_lit(v, True) for v in m["u_min"]) + "};\n"
                   "static const SPT_REAL spt_obs_umax[SPT_OBS_NU] = {"
                   + ", ".join(lim_lit(v, False) for v in m["u_max"]) + "};\n")
    src = [f"/**\n * @file spt_obs.c\n * @brief {title} — {brief[:-1]}\n *\n * {_BANNER}\n */",
           '#include "spt_obs.h"', "", consts]
    init = "void spt_obs_init(spt_obs_t *s)\n{\n    int i;\n"
    if not st:
        init += "    for (i = 0; i < SPT_OBS_NX; ++i) {\n        s->xp[i] = SPT_C(0.0);\n"
        if cur:
            init += "        s->xc[i] = SPT_C(0.0);\n"
        init += "    }\n"
    init += "    for (i = 0; i < SPT_OBS_NU; ++i) {\n        s->u[i] = SPT_C(0.0);\n"
    if lim:
        init += "        s->u_min[i] = spt_obs_umin[i];\n        s->u_max[i] = spt_obs_umax[i];\n"
    init += "    }\n"
    if ref == "lqi":
        init += "    for (i = 0; i < SPT_OBS_NY; ++i) {\n        s->xi[i] = SPT_C(0.0);\n    }\n"
    init += "}\n"
    src.append(init)
    out = [f"int spt_obs_output(spt_obs_t *s, {mdecl}, {rdecl}, SPT_REAL u[SPT_OBS_NU])", "{"]
    out.append("    int i, j;")
    if cur:
        out.append("    SPT_REAL nu_[SPT_OBS_NY];")
    out.append(f"    for (i = 0; i < {nmeas}; ++i) {{\n        if (!spt_ctrl_finite({mv}[i])) {{\n"
               "            for (j = 0; j < SPT_OBS_NU; ++j) {\n                u[j] = s->u[j];\n            }\n"
               "            return 0;\n        }\n    }")
    if nr:
        out.append("    for (i = 0; i < SPT_OBS_NR; ++i) {\n        if (!spt_ctrl_finite(r[i])) {\n"
                   "            for (j = 0; j < SPT_OBS_NU; ++j) {\n                u[j] = s->u[j];\n            }\n"
                   "            return 0;\n        }\n    }")
    else:
        out.append("    (void)r;")
    xh = "x" if st else ("s->xc" if cur else "s->xp")
    body = ""
    if cur:
        body += _loop_matvec("nu_", "spt_obs_c", "s->xp", "SPT_OBS_NY", "SPT_OBS_NX", "-", "y[i]")
        body += _loop_matvec("s->xc", "spt_obs_m", "nu_", "SPT_OBS_NX", "SPT_OBS_NY", "+", "s->xp[i]")
    body += _loop_matvec("u", "spt_obs_k", xh, "SPT_OBS_NU", "SPT_OBS_NX", "-", None)
    if ref == "nbar":
        body += _loop_matvec("u", "spt_obs_nbar", "r", "SPT_OBS_NU", "SPT_OBS_NR", "+", "u[i]")
    elif ref == "lqi":
        body += _loop_matvec("u", "spt_obs_ki", "s->xi", "SPT_OBS_NU", "SPT_OBS_NY", "-", "u[i]")
    if lim:
        body += ("    for (i = 0; i < SPT_OBS_NU; ++i) {\n"
                 "        u[i] = (u[i] > s->u_max[i]) ? s->u_max[i] : ((u[i] < s->u_min[i]) ? s->u_min[i] : u[i]);\n"
                 "    }\n")
    body += "    for (i = 0; i < SPT_OBS_NU; ++i) {\n        s->u[i] = u[i];\n    }\n    return 1;\n}\n"
    src.append("\n".join(out) + "\n" + body)
    upd = [f"int spt_obs_update(spt_obs_t *s, {mdecl}, {rdecl}, const SPT_REAL ua[SPT_OBS_NU])", "{"]
    if not st:
        upd.append("    SPT_REAL nu_[SPT_OBS_NY], xn[SPT_OBS_NX];")
    elif ref == "lqi":
        upd.append("    SPT_REAL yv[SPT_OBS_NY];")
    upd.append("    int i, j;")
    checks = _finite_check(mv, nmeas) + _finite_check("ua", "SPT_OBS_NU")
    if nr:
        checks += _finite_check("r", "SPT_OBS_NR")
    upd.append(checks.rstrip("\n"))
    ub = ""
    if st:
        if ref == "lqi":           # y = C·x + D·u (uygulanan u), xi += Ts·(r − y)
            ub += _loop_matvec("yv", "spt_obs_c", "x", "SPT_OBS_NY", "SPT_OBS_NX", "+", None)
            if has_d:
                ub += _loop_matvec("yv", "spt_obs_d", "ua", "SPT_OBS_NY", "SPT_OBS_NU", "+", "yv[i]")
            ub += "    for (i = 0; i < SPT_OBS_NY; ++i) {\n        s->xi[i] += SPT_OBS_TS * (r[i] - yv[i]);\n    }\n"
        else:
            ub += "    (void)r;\n"
    else:
        ub += _loop_matvec("nu_", "spt_obs_c", "s->xp", "SPT_OBS_NY", "SPT_OBS_NX", "-", "y[i]")
        if has_d:
            ub += _loop_matvec("nu_", "spt_obs_d", "ua", "SPT_OBS_NY", "SPT_OBS_NU", "-", "nu_[i]")
        ub += _loop_matvec("xn", "spt_obs_a", "s->xp", "SPT_OBS_NX", "SPT_OBS_NX", "+", None)
        ub += _loop_matvec("xn", "spt_obs_b", "ua", "SPT_OBS_NX", "SPT_OBS_NU", "+", "xn[i]")
        ub += _loop_matvec("xn", "spt_obs_l", "nu_", "SPT_OBS_NX", "SPT_OBS_NY", "+", "xn[i]")
        if ref == "lqi":
            ub += "    for (i = 0; i < SPT_OBS_NY; ++i) {\n        s->xi[i] += SPT_OBS_TS * (r[i] - y[i]);\n    }\n"
        else:
            ub += "    (void)r;\n"
        ub += "    for (i = 0; i < SPT_OBS_NX; ++i) {\n        s->xp[i] = xn[i];\n    }\n"
    ub += "    for (i = 0; i < SPT_OBS_NU; ++i) {\n        s->u[i] = ua[i];\n    }\n    (void)j;\n    return 1;\n}\n"
    src.append("\n".join(upd) + "\n" + ub)
    src.append(f"int spt_obs_step(spt_obs_t *s, {mdecl}, {rdecl}, SPT_REAL u[SPT_OBS_NU])\n"
               f"{{\n    if (!spt_obs_output(s, {mv}, r, u)) {{\n        return 0;\n    }}\n"
               f"    return spt_obs_update(s, {mv}, r, u);\n}}\n")
    if not st:
        src.append("void spt_obs_state(const spt_obs_t *s, SPT_REAL xhat[SPT_OBS_NX])\n{\n    int i;\n"
                   f"    for (i = 0; i < SPT_OBS_NX; ++i) {{\n        xhat[i] = {xh}[i];\n    }}\n}}\n")
    return [GeneratedFile("spt_obs.h", header), GeneratedFile("spt_obs.c", "\n".join(src))]


def _kf_files(spec: KalmanSpec, title: str, real: str) -> list[GeneratedFile]:
    m = spec.matrices()
    nx, nu, ny = spec.dims
    cur = spec.form == "current"
    types = _fill(_TYPES, USE_DOUBLE="1" if real == "double" else "0")
    doc = [" * Model: x[n+1] = A·x[n] + B·u[n] + G·w[n], y[n] = C·x[n] + D·u[n] + v[n] (kararlı durum kazançları).",
           " * MATLAB kalman / estim ('" + spec.form + "'):",
           " *   yenilik  ν = y − C·x̂[n|n−1] − D·u",
           " *   tahmin   x̂[n+1|n] = A·x̂[n|n−1] + B·u + L·ν"]
    has_my = cur and "My" in m
    if has_my:
        doc.append(" *   güncel   x̂[n|n] = x̂[n|n−1] + M·ν,  ŷ = C·x̂[n|n−1] + D·u + My·ν   (çıktı; MATLAB My)")
    elif cur:
        doc.append(" *   güncel   x̂[n|n] = x̂[n|n−1] + M·ν,  ŷ = C·x̂[n|n] + D·u   (çıktı)")
    else:
        doc.append(" *   çıktı    x̂[n|n−1],  ŷ = C·x̂[n|n−1] + D·u")
    doc.append(" * Ölçümde sonlu olmayan değer (sensör yok / hatalı): o örnekte ölçüm güncellemesi atlanır (ν = 0),")
    doc.append(" * yalnız tahmin yapılır ve 0 döner. Sonlu olmayan u: durum değişmez, son kestirim yazılır, 0 döner.")
    udecl = "const SPT_REAL u[SPT_KF_NU]" if nu else "const SPT_REAL *u"
    api = ["/** Durumu sıfırlar (x̂ = 0). */", "void spt_kf_init(spt_kf_t *s);",
           "/** Bir örnek: u (bilinen girişler" + ("" if nu else "; yok → NULL") + "), y (ölçümler) → ŷ, x̂. "
           "Ölçüm kullanıldıysa 1. */",
           f"int spt_kf_step(spt_kf_t *s, {udecl}, const SPT_REAL y[SPT_KF_NY], SPT_REAL yhat[SPT_KF_NY], "
           "SPT_REAL xhat[SPT_KF_NX]);"]
    fields = ["    SPT_REAL xp[SPT_KF_NX];   /**< x̂[n|n−1] */", "    SPT_REAL xo[SPT_KF_NX];   /**< son çıktı x̂ */",
              "    SPT_REAL yo[SPT_KF_NY];   /**< son çıktı ŷ */"]
    header = _fill(_EST_H, K="kf", KU="KF", TITLE=title, BRIEF="kararlı durum Kalman süzgeci.", BANNER=_BANNER,
                   DOC="\n".join(doc), TS_TEXT=f"{spec.Ts:g}", TYPES=types, TS=_c_float(spec.Ts), NX=str(nx),
                   NU=str(nu), NU_DOC="bilinen giriş sayısı", NY=str(ny), EXTRA_DEFS="", FIELDS="\n".join(fields),
                   API="\n".join(api))
    has_b = nu > 0 and bool(np.any(m["B"] != 0))
    has_d = nu > 0 and bool(np.any(m["D"] != 0))
    consts = _c_matrix("spt_kf_a", m["A"], "A") + _c_matrix("spt_kf_c", m["C"], "C")
    if has_b:
        consts += _c_matrix("spt_kf_b", m["B"], "B (bilinen girişler)")
    if has_d:
        consts += _c_matrix("spt_kf_d", m["D"], "D (bilinen girişler)")
    consts += _c_matrix("spt_kf_l", m["L"], "L (tahmin kazancı)")
    if cur:
        consts += _c_matrix("spt_kf_m", m["M"], "M (yenilik kazancı)")
    if has_my:
        consts += _c_matrix("spt_kf_my", m["My"], "My (çıkış yenilik kazancı)")
    src = [f"/**\n * @file spt_kf.c\n * @brief {title} — kararlı durum Kalman süzgeci.\n *\n * {_BANNER}\n */",
           '#include "spt_kf.h"', "", consts]
    src.append("void spt_kf_init(spt_kf_t *s)\n{\n    int i;\n    for (i = 0; i < SPT_KF_NX; ++i) {\n"
               "        s->xp[i] = SPT_C(0.0);\n        s->xo[i] = SPT_C(0.0);\n    }\n"
               "    for (i = 0; i < SPT_KF_NY; ++i) {\n        s->yo[i] = SPT_C(0.0);\n    }\n}\n")
    f = [f"int spt_kf_step(spt_kf_t *s, {udecl}, const SPT_REAL y[SPT_KF_NY], SPT_REAL yhat[SPT_KF_NY], "
         "SPT_REAL xhat[SPT_KF_NX])", "{", "    SPT_REAL nu_[SPT_KF_NY], xn[SPT_KF_NX];", "    int i, j, meas = 1;"]
    body = ""
    if nu:
        body += ("    for (i = 0; i < SPT_KF_NU; ++i) {\n        if (!spt_ctrl_finite(u[i])) {\n"
                 "            for (j = 0; j < SPT_KF_NX; ++j) {\n                xhat[j] = s->xo[j];\n            }\n"
                 "            for (j = 0; j < SPT_KF_NY; ++j) {\n                yhat[j] = s->yo[j];\n            }\n"
                 "            return 0;\n        }\n    }\n")
    else:
        body += "    (void)u;\n"
    body += ("    for (i = 0; i < SPT_KF_NY; ++i) {\n        if (!spt_ctrl_finite(y[i])) {\n            meas = 0;\n"
             "        }\n    }\n    if (meas) {\n")
    inner = _loop_matvec("nu_", "spt_kf_c", "s->xp", "SPT_KF_NY", "SPT_KF_NX", "-", "y[i]", "        ")
    if has_d:
        inner += _loop_matvec("nu_", "spt_kf_d", "u", "SPT_KF_NY", "SPT_KF_NU", "-", "nu_[i]", "        ")
    body += inner + ("    } else {\n        for (i = 0; i < SPT_KF_NY; ++i) {\n            nu_[i] = SPT_C(0.0);\n"
                     "        }\n    }\n")
    if cur:
        body += _loop_matvec("s->xo", "spt_kf_m", "nu_", "SPT_KF_NX", "SPT_KF_NY", "+", "s->xp[i]")
    else:
        body += "    for (i = 0; i < SPT_KF_NX; ++i) {\n        s->xo[i] = s->xp[i];\n    }\n"
    body += _loop_matvec("s->yo", "spt_kf_c", "s->xp" if has_my else "s->xo", "SPT_KF_NY", "SPT_KF_NX", "+", None)
    if has_d:
        body += _loop_matvec("s->yo", "spt_kf_d", "u", "SPT_KF_NY", "SPT_KF_NU", "+", "s->yo[i]")
    if has_my:
        body += _loop_matvec("s->yo", "spt_kf_my", "nu_", "SPT_KF_NY", "SPT_KF_NY", "+", "s->yo[i]")
    body += _loop_matvec("xn", "spt_kf_a", "s->xp", "SPT_KF_NX", "SPT_KF_NX", "+", None)
    if has_b:
        body += _loop_matvec("xn", "spt_kf_b", "u", "SPT_KF_NX", "SPT_KF_NU", "+", "xn[i]")
    body += _loop_matvec("xn", "spt_kf_l", "nu_", "SPT_KF_NX", "SPT_KF_NY", "+", "xn[i]")
    body += ("    for (i = 0; i < SPT_KF_NX; ++i) {\n        s->xp[i] = xn[i];\n        xhat[i] = s->xo[i];\n    }\n"
             "    for (i = 0; i < SPT_KF_NY; ++i) {\n        yhat[i] = s->yo[i];\n    }\n    (void)j;\n"
             "    return meas;\n}\n")
    src.append("\n".join(f) + "\n" + body)
    return [GeneratedFile("spt_kf.h", header), GeneratedFile("spt_kf.c", "\n".join(src))]


# ============================================================================ üretim
def describe(spec: Spec) -> str:
    """Kısa Türkçe tanım (arayüz ve belgeler)."""
    if isinstance(spec, PidSpec):
        parts = [f"Kp = {spec.Kp:.6g}", f"Ki = {spec.Ki:.6g}", f"Kd = {spec.Kd:.6g}", f"Tf = {spec.Tf:.6g}"]
        if spec.b != 1 or spec.c != 1 or spec.two_dof:
            parts += [f"b = {spec.b:.6g}", f"c = {spec.c:.6g}"]
        txt = (f"Ayrık PID ({'standart biçimden' if spec.form == 'standard' else 'paralel'}), " + ", ".join(parts)
               + f"; IF {_FORMULA_TR[spec.i_formula]}, DF {_FORMULA_TR[spec.d_formula]}; Ts = {spec.Ts:g} s")
        if spec.has_limits:
            txt += f"; doyum [{spec.u_min:g}, {spec.u_max:g}], anti-windup {_AW_TR[spec.anti_windup]}"
        return txt
    if isinstance(spec, LtiSpec):
        ny, nu = spec.shape
        rz = _lti_realization(spec)
        if rz["mode"] == "sos":
            return f"Ayrık SISO kontrolcü, {len(rz['sos'])} ikinci derece bölüm; Ts = {spec.Ts:g} s"
        nx = rz["A"].shape[0]
        if nx == 0:
            return f"Statik kazanç {ny}×{nu}; çağrı periyodu {spec.Ts:g} s"
        return f"Ayrık durum uzayı kontrolcü: {nx} durum, {nu} giriş, {ny} çıkış; Ts = {spec.Ts:g} s"
    if isinstance(spec, ObserverSpec):
        nx, nu, ny = spec.dims
        ref = {"none": "referanssız (regülatör)", "nbar": "Nbar referans ölçekli", "lqi": "LQI integral eylemli"}
        if spec.is_state:
            return (f"Tam durum geri beslemesi (gözlemcisiz, u = −K·x), {nx} durum, {nu} çıkış, "
                    f"{ref[spec.reference]}; Ts = {spec.Ts:g} s")
        return (f"Gözlemcili durum geri beslemesi ({spec.form}), {nx} durum, {nu} çıkış, {ny} ölçüm, "
                f"{ref[spec.reference]}; Ts = {spec.Ts:g} s")
    nx, nu, ny = spec.dims
    return f"Kararlı durum Kalman süzgeci ({spec.form}), {nx} durum, {nu} bilinen giriş, {ny} ölçüm; Ts = {spec.Ts:g} s"


def kind_title(spec: Spec) -> str:
    """Kontrolcü türünün kısa başlığı (dosya başlığı, tablo satırları)."""
    if isinstance(spec, ObserverSpec) and spec.is_state:
        return "Tam durum geri beslemesi"
    return KIND_TITLES[spec.kind]


def _title(spec: Spec, opts: ControlCodegenOptions) -> str:
    return (opts.name or kind_title(spec)).replace("*/", "* /").replace("\n", " ")


def _sources(spec: Spec, real: str, title: str = "") -> list[GeneratedFile]:
    """C dosyaları (önek uygulanmamış: spt_*)."""
    title = title or kind_title(spec)
    if isinstance(spec, PidSpec):
        return _pid_files(spec, title, real)
    if isinstance(spec, LtiSpec):
        return _lti_files(spec, title, real)
    if isinstance(spec, ObserverSpec):
        return _obs_files(spec, title, real)
    return _kf_files(spec, title, real)


def _example(spec: Spec, opts: ControlCodegenOptions) -> str:
    k = spec.kind
    fs = _rate(spec, opts)
    head = (f"/**\n * @file spt_{k}_ornek.c\n * @brief Örnek: kontrolcüyü {fs:g} Hz'de çağıran görev (sensör okuma ve "
            "eyleyici yazma kullanıcıya ait).\n *\n * " + _BANNER + "\n */\n" + f'#include "spt_{k}.h"\n\n')
    if isinstance(spec, PidSpec):
        track = ("\n/* Elle kontrolden otomatiğe sarsıntısız geçiş: elle kipte uygulanan çıkışı izlet. */\n"
                 "SPT_REAL ornek_elle(SPT_REAL r, SPT_REAL y, SPT_REAL elle_u)\n{\n"
                 "    (void)spt_pid_step_track(&ornek_pid, r, y, elle_u);\n    return elle_u;\n}\n")
        return head + ("static spt_pid_t ornek_pid;\n\nvoid ornek_baslat(void)\n{\n    spt_pid_init(&ornek_pid);\n}\n\n"
                       f"/* Zamanlayıcı kesmesinden ya da RTOS görevinden tam {spec.Ts:g} s periyotla çağırın. */\n"
                       "SPT_REAL ornek_adim(SPT_REAL r, SPT_REAL y)\n{\n"
                       "    return spt_pid_step(&ornek_pid, r, y);   /* doyumlu eyleyici komutu */\n}\n" + track)
    if isinstance(spec, LtiSpec):
        init = "static spt_lti_t ornek_lti;\n\nvoid ornek_baslat(void)\n{\n    spt_lti_init(&ornek_lti);\n}\n\n"
        if _lti_realization(spec)["mode"] == "sos":
            return head + (init + f"/* Tam {spec.Ts:g} s periyotla: giriş genellikle hata e = r − y. */\n"
                           "SPT_REAL ornek_adim(SPT_REAL e)\n{\n    return spt_lti_step(&ornek_lti, e);\n}\n")
        return head + (init + f"/* Tam {spec.Ts:g} s periyotla. */\n"
                       "int ornek_adim(const SPT_REAL giris[SPT_LTI_NU], SPT_REAL cikis[SPT_LTI_NY])\n{\n"
                       "    return spt_lti_step(&ornek_lti, giris, cikis);\n}\n")
    if isinstance(spec, ObserverSpec):
        r = "r" if spec.reference != "none" else "NULL"
        rparam = ", const SPT_REAL r[SPT_OBS_NR]" if spec.reference != "none" else ""
        mv, nmeas = ("x", "SPT_OBS_NX") if spec.is_state else ("y", "SPT_OBS_NY")
        what = "integratörü" if spec.is_state else "kestiriciyi"
        return head + ("static spt_obs_t ornek_obs;\n\nvoid ornek_baslat(void)\n{\n    spt_obs_init(&ornek_obs);\n}\n\n"
                       f"/* Tam {spec.Ts:g} s periyotla: ölç → hesapla → uygula → {what} uygulanan u ile "
                       "ilerlet. */\n"
                       f"void ornek_adim(const SPT_REAL {mv}[{nmeas}]{rparam}, SPT_REAL u[SPT_OBS_NU])\n{{\n"
                       f"    if (spt_obs_output(&ornek_obs, {mv}, {r}, u)) {{\n"
                       "        /* eyleyici_yaz(u); — eyleyici ayrıca sınırlıyorsa gerçekten uygulanan değeri "
                       "verin */\n"
                       f"        (void)spt_obs_update(&ornek_obs, {mv}, {r}, u);\n    }}\n}}\n")
    nu = spec.dims[1]
    u = "u" if nu else "NULL"
    uparam = "const SPT_REAL u[SPT_KF_NU], " if nu else ""
    return head + ("static spt_kf_t ornek_kf;\n\nvoid ornek_baslat(void)\n{\n    spt_kf_init(&ornek_kf);\n}\n\n"
                   f"/* Tam {spec.Ts:g} s periyotla; ölçüm yoksa ilgili y[i] = NAN verin (yalnız tahmin). */\n"
                   f"void ornek_adim({uparam}const SPT_REAL y[SPT_KF_NY], SPT_REAL yhat[SPT_KF_NY], "
                   "SPT_REAL xhat[SPT_KF_NX])\n{\n"
                   f"    (void)spt_kf_step(&ornek_kf, {u}, y, yhat, xhat);\n}}\n")


def generate(spec: Any, opts: ControlCodegenOptions, footprint_rows: list[FootprintRow] | None = None,
             cost_rows: list[CostRow] | None = None) -> list[GeneratedFile]:
    """<önek>_<tür>.h / .c, <önek>_<tür>_ornek.c ve KULLANIM_<önek>.md (önek uygulanmış)."""
    spec = controller_spec(spec)
    check_options(opts)
    title = _title(spec, opts)
    files = _sources(spec, opts.real, title) + [GeneratedFile(f"spt_{spec.kind}_ornek.c", _example(spec, opts))]
    files = [GeneratedFile(_prefixed(f.name, opts.prefix),
                           _wrap_c(_prefixed(f.text.replace("@PFX@", opts.prefix), opts.prefix))) for f in files]
    files.append(GeneratedFile(f"KULLANIM_{opts.prefix}.md", usage_markdown(spec, opts, footprint_rows, cost_rows)))
    return files


# ============================================================================ boyut (sizeof, analitik)
def state_size(spec: Spec, real: str) -> int:
    """sizeof(durum yapısı) — ILP32 ve x86-64'te aynı (yalnız SPT_REAL / int alanları)."""
    r = 8 if real == "double" else 4
    if isinstance(spec, PidSpec):
        return 14 * r
    if isinstance(spec, LtiSpec):
        rz = _lti_realization(spec)
        if rz["mode"] == "sos":
            return (2 * len(rz["sos"]) + 1) * r
        return (rz["A"].shape[0] + rz["D"].shape[0]) * r
    if isinstance(spec, ObserverSpec):
        m = spec.matrices()
        nx, nu, ny = spec.dims
        lim = bool(np.any(np.isfinite(m["u_min"])) or np.any(np.isfinite(m["u_max"])))
        n = nx * {"current": 2, "delayed": 1, "state": 0}[spec.form] + (ny if spec.reference == "lqi" else 0) \
            + nu * (3 if lim else 1)
        return n * r
    nx, nu, ny = spec.dims
    return (2 * nx + ny) * r


# ============================================================================ ana makine sürücüsü
_DRIVER = r"""/* Ana makine doğrulama / maliyet sürücüsü (MCU'ya eklenmez). Giriş: [N, nin] + N×nin double;
   çıkış: N×nout double (+ sayaç modunda N×SPT_NOPS işlem sayısı). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "spt_{{K}}.h"

#if defined(SPT_CUSTOM_REAL)
unsigned long long spt_ops[SPT_NOPS];
#define SPT_DRV_REAL(x) spt_cnt(x)
#define SPT_DRV_DBL(r) ((r).v)
#define SPT_DRV_NOPS SPT_NOPS
#else
#define SPT_DRV_REAL(x) ((SPT_REAL)(x))
#define SPT_DRV_DBL(r) ((double)(r))
#define SPT_DRV_NOPS 0
#endif

int main(int argc, char **argv)
{
    FILE *f;
    double hdr[2];
    double *in, *outv, *ops;
    long n, k;
    int nin, i;
    static spt_{{K}}_t st;
    unsigned long long before[SPT_DRV_NOPS + 1];
    SPT_REAL vin[{{NIN_MAX}}], vout[{{NOUT_MAX}}];
    if (argc < 3) {
        return 2;
    }
    f = fopen(argv[1], "rb");
    if (!f || fread(hdr, sizeof(double), 2, f) != 2) {
        return 3;
    }
    n = (long)hdr[0];
    nin = (int)hdr[1];
    in = (double *)malloc(sizeof(double) * (size_t)(n * nin + 1));
    outv = (double *)malloc(sizeof(double) * (size_t)(n * {{NOUT}} + 1));
    ops = (double *)malloc(sizeof(double) * (size_t)(n * (SPT_DRV_NOPS + 1)));
    if (!in || !outv || !ops || fread(in, sizeof(double), (size_t)(n * nin), f) != (size_t)(n * nin)) {
        return 4;
    }
    fclose(f);
    (void)before;
    for (i = 0; i < {{NIN_MAX}}; ++i) {
        vin[i] = SPT_DRV_REAL(0.0);
    }
    {{INIT}}
    for (k = 0; k < n; ++k) {
        for (i = 0; i < nin && i < {{NIN_MAX}}; ++i) {
            vin[i] = SPT_DRV_REAL(in[k * nin + i]);
        }
#if defined(SPT_CUSTOM_REAL)
        memcpy(before, spt_ops, sizeof(spt_ops));
#endif
        {{CALL}}
#if defined(SPT_CUSTOM_REAL)
        for (i = 0; i < SPT_NOPS; ++i) {
            ops[k * SPT_NOPS + i] = (double)(spt_ops[i] - before[i]);
        }
#endif
        for (i = 0; i < {{NOUT}}; ++i) {
            outv[k * {{NOUT}} + i] = SPT_DRV_DBL(vout[i]);
        }
    }
    f = fopen(argv[2], "wb");
    if (!f) {
        return 6;
    }
    fwrite(outv, sizeof(double), (size_t)(n * {{NOUT}}), f);
    if (SPT_DRV_NOPS > 0) {
        fwrite(ops, sizeof(double), (size_t)(SPT_DRV_NOPS * n), f);
    }
    fclose(f);
    return 0;
}
"""


def _io_dims(spec: Spec) -> tuple[int, int]:
    """(giriş sütunu, çıkış sütunu) sayısı — simulate() ile aynı düzen."""
    if isinstance(spec, PidSpec):
        return 3, 1
    if isinstance(spec, LtiSpec):
        ny, nu = spec.shape
        return nu, ny
    nx, nu, ny = spec.dims
    if isinstance(spec, ObserverSpec):
        return spec.n_meas + spec.n_ref, nu
    return nu + ny, ny + nx


def _driver_source(spec: Spec, track: bool) -> str:
    k = spec.kind
    nin, nout = _io_dims(spec)
    if isinstance(spec, PidSpec):
        call = ("vout[0] = spt_pid_step_track(&st, vin[0], vin[1], vin[2]);" if track
                else "vout[0] = spt_pid_step(&st, vin[0], vin[1]);")
    elif isinstance(spec, LtiSpec):
        if _lti_realization(spec)["mode"] == "sos":
            call = "vout[0] = spt_lti_step(&st, vin[0]);"
        else:
            call = "(void)spt_lti_step(&st, vin, vout);"
    elif isinstance(spec, ObserverSpec):
        r = f"vin + {spec.n_meas}" if spec.reference != "none" else "NULL"
        call = f"(void)spt_obs_step(&st, vin, {r}, vout);"
    else:
        nx, nu, ny = spec.dims
        u = "vin" if nu else "NULL"
        call = f"(void)spt_kf_step(&st, {u}, vin + {nu}, vout, vout + {ny});"
    return _fill(_DRIVER, K=k, NIN_MAX=str(max(nin, 1)), NOUT_MAX=str(max(nout, 1)), NOUT=str(nout),
                 INIT=f"spt_{k}_init(&st);", CALL=call)


def host_program(spec: Any, *, real: str = "double", count_ops: bool = False, track: bool = False) -> tc.Build:
    """Üretilen C'yi ana makinede derler (doğrulama sürücüsüyle). count_ops: C++ olarak, SPT_REAL = sayaç türü."""
    spec = controller_spec(spec)
    files = {f.name: f.text for f in _sources(spec, real)}
    files["spt_host_driver.c"] = _driver_source(spec, track)
    sources = [f"spt_{spec.kind}.c", "spt_host_driver.c"]
    if count_ops:
        files["spt_count_prelude.hpp"] = _count_prelude()
        return tc.host_build(files, sources, cxx=True, extra=["-std=c++17", "-include", "spt_count_prelude.hpp"],
                             opt="-O2", output="spt_count")
    return tc.host_build(files, sources, defines=[f"SPT_USE_DOUBLE={1 if real == 'double' else 0}"],
                         extra=["-std=c99", "-Wall", "-Wextra", "-pedantic", "-Werror"], opt="-O2",
                         output=f"spt_host_{real}")


@dataclass(frozen=True)
class HostRun:
    out: np.ndarray                     # (N, çıkış)
    ops: np.ndarray | None = None       # (N, len(OPS))


def run_generated(spec: Any, inputs: np.ndarray, *, real: str = "double", count_ops: bool = False) -> HostRun:
    """Üretilen C'yi ana makinede verilen girişlerle çalıştırır (simulate() ile aynı giriş / çıkış düzeni)."""
    spec = controller_spec(spec)
    x = np.atleast_2d(np.asarray(inputs, float))
    nin, nout = _io_dims(spec)
    track = False
    if isinstance(spec, PidSpec):
        track = x.shape[1] >= 3 and spec.tracking
        if x.shape[1] < 3:
            x = np.column_stack([x, np.zeros((x.shape[0], 3 - x.shape[1]))])
    if x.shape[1] != nin:
        raise ControlError(f"Giriş sütun sayısı {nin} olmalı (verilen {x.shape[1]})")
    build = host_program(spec, real=real, count_ops=count_ops, track=track)
    if not build.ok:
        raise RuntimeError("Ana makine derlemesi başarısız:\n" + build.log)
    n = x.shape[0]
    blob = np.r_[float(n), float(nin), x.ravel()].astype("<f8").tobytes()
    tag = hashlib.sha256(blob).hexdigest()[:16]
    runs = build.workdir / "runs"
    runs.mkdir(exist_ok=True)
    inp, outp = runs / f"in_{tag}.bin", runs / f"out_{tag}.bin"
    if not outp.exists():
        inp.write_bytes(blob)
        res = tc.run_program(build, [str(inp), str(outp)], timeout=600)
        if res.returncode != 0 or not outp.exists():
            raise RuntimeError(f"Sürücü hatası ({res.returncode}): {res.stderr}")
        inp.unlink(missing_ok=True)
    raw = np.fromfile(outp, dtype="<f8")
    out = raw[:n * nout].reshape(n, nout)
    ops = raw[n * nout:].reshape(n, len(OPS)) if count_ops else None
    return HostRun(out, ops)


# ============================================================================ çapraz derleme denetimi ve ayak izi
def build_check(spec: Any, prof: McuProfile, compiler: str, real: str, opt: str) -> tc.Build:
    """-std=c99 -Wall -Wextra -pedantic -Werror (+ float'ta -Wdouble-promotion) ile nesneye derler."""
    spec = controller_spec(spec)
    files = {f.name: f.text for f in _sources(spec, real)}
    files[f"spt_{spec.kind}_ornek.c"] = _example(spec, ControlCodegenOptions(real=real))
    warn = list(tc.WARN_FLAGS) + list(WARN_EXTRA) + (["-Wdouble-promotion"] if real == "float" else [])
    return tc.compile_cross(files, [f"spt_{spec.kind}.c", f"spt_{spec.kind}_ornek.c"], prof, compiler, opt,
                            defines=[f"SPT_USE_DOUBLE={1 if real == 'double' else 0}"], warnings=warn)


_FP_CACHE: dict[str, dict] = {}
_FP_VERSION = 1


def _harness(spec: Spec, baseline: bool) -> str:
    """Ayak izi koşumu: tüm genel API çağrılır, giriş / çıkış volatile (derleyici kodu atamaz)."""
    k = spec.kind
    nin, nout = _io_dims(spec)
    nin = max(nin, 1)
    if baseline:
        calls = "        for (i = 0; i < NOUT; ++i) {\n            vout[i] = vin[i % NIN];\n        }\n"
        state = ""
        init = ""
    else:
        state = f"static spt_{k}_t st;\n"
        init = f"    spt_{k}_init(&st);\n"
        if isinstance(spec, PidSpec):
            calls = ("        vout[0] = spt_pid_step(&st, vin[0], vin[1]);\n"
                     "        vout[0] += spt_pid_step_track(&st, vin[0], vin[1], vin[2]);\n"
                     "        (void)spt_pid_set_gains(&st, vin[0], vin[1], vin[2], vin[0]);\n"
                     "        (void)spt_pid_set_standard(&st, vin[0], vin[1], vin[2], vin[0]);\n"
                     "        (void)spt_pid_set_limits(&st, vin[1], vin[2]);\n"
                     "        spt_pid_set_integrator(&st, vin[2]);\n")
        elif isinstance(spec, LtiSpec):
            if _lti_realization(spec)["mode"] == "sos":
                calls = "        vout[0] = spt_lti_step(&st, vin[0]);\n"
            else:
                calls = "        (void)spt_lti_step(&st, vin, vout);\n"
        elif isinstance(spec, ObserverSpec):
            r = f"vin + {spec.n_meas}" if spec.reference != "none" else "NULL"
            calls = (f"        (void)spt_obs_step(&st, vin, {r}, vout);\n"
                     f"        (void)spt_obs_output(&st, vin, {r}, vout);\n"
                     f"        (void)spt_obs_update(&st, vin, {r}, vout);\n")
            if not spec.is_state:
                calls += ("        {\n            SPT_REAL xh[SPT_OBS_NX];\n            spt_obs_state(&st, xh);\n"
                          "            vout[0] += xh[0];\n        }\n")
        else:
            nx, nu, ny = spec.dims
            u = "vin" if nu else "NULL"
            calls = f"        (void)spt_kf_step(&st, {u}, vin + {nu}, vout, vout + {ny});\n"
    return (f"/* Ayak izi ölçüm koşumu ({'temel' if baseline else k}). */\n#include \"spt_{k}.h\"\n"
            f"#define NIN {max(nin, 3)}\n#define NOUT {nout}\n{state}"
            f"volatile SPT_REAL spt_in[NIN];\nvolatile SPT_REAL spt_out[NOUT];\nint main(void);\nint main(void)\n{{\n"
            f"    SPT_REAL vin[NIN], vout[NOUT];\n    int i;\n"
            f"    for (i = 0; i < NIN; ++i) {{\n        vin[i] = spt_in[i];\n    }}\n"
            f"    for (i = 0; i < NOUT; ++i) {{\n        vout[i] = vin[i % NIN];\n    }}\n{init}    for (;;) {{\n"
            f"        for (i = 0; i < NIN; ++i) {{\n            vin[i] = spt_in[i];\n        }}\n{calls}"
            f"        for (i = 0; i < NOUT; ++i) {{\n            spt_out[i] = vout[i];\n        }}\n    }}\n}}\n")


def _measure(spec: Spec, opts: ControlCodegenOptions, prof: McuProfile, compiler: str) -> dict | None:
    tool = tc.cross_tool(compiler)
    if tool is None or tc.cross_flags(prof, compiler) is None:
        return None
    k = spec.kind
    files = {f.name: f.text for f in _sources(spec, opts.real)}
    real_def = f"SPT_USE_DOUBLE={1 if opts.real == 'double' else 0}"
    files["spt_sizeof_probe.c"] = f'#include "spt_{k}.h"\nunsigned char spt_sizeof_{k}[sizeof(spt_{k}_t)];\n'
    gcc = tc.find_arm_gcc()
    key_blob = json.dumps([_FP_VERSION, tc.STACK_ANALYSIS_VERSION, sorted(files.items()), prof.key, compiler,
                           opts.opt, real_def, tool.path, tool.version, gcc.version if gcc else None])
    key = hashlib.sha256(key_blob.encode()).hexdigest()[:24]
    cache_file = tc.WORK_ROOT / "footprint_control" / f"{key}.json"
    if key in _FP_CACHE:
        return _FP_CACHE[key]
    if cache_file.exists():
        try:
            _FP_CACHE[key] = json.loads(cache_file.read_text(encoding="utf-8"))
            return _FP_CACHE[key]
        except (OSError, ValueError):
            pass
    build = tc.compile_cross(files, [f"spt_{k}.c", "spt_sizeof_probe.c"], prof, compiler, opts.opt,
                             defines=[real_def], warnings=())
    if not build.ok:
        raise RuntimeError(f"{compiler} {prof.key} derleme hatası:\n{build.log}")
    syms = tc.elf_symbols(build.objects["spt_sizeof_probe.c"])
    sz = build.sizes(f"spt_{k}.c")
    row = {"flash_code": sz.flash, "ram_state": syms[f"spt_sizeof_{k}"].size, "flash_with_libm": None,
           "stack": None, "stack_own": None, "stack_path": [], "stack_exact": None}
    if gcc is not None and prof.gcc_flags is not None:
        bfiles = {f"spt_{k}.h": files[f"spt_{k}.h"], "spt_fp_base.c": _harness(spec, True)}
        base = tc.link_cross(bfiles, ["spt_fp_base.c"], prof, compiler, opts.opt, defines=[real_def])
        hfiles = {f"spt_{k}.h": files[f"spt_{k}.h"], f"spt_{k}.c": files[f"spt_{k}.c"],
                  "spt_fp_harness.c": _harness(spec, False)}
        img = tc.link_cross(hfiles, ["spt_fp_harness.c", f"spt_{k}.c"], prof, compiler, opts.opt, defines=[real_def])
        if base.ok and img.ok:
            row["flash_with_libm"] = tc.elf_sizes(img.output).flash - tc.elf_sizes(base.output).flash
            entry = {"pid": "spt_pid_step_track", "lti": "spt_lti_step", "obs": "spt_obs_step",
                     "kf": "spt_kf_step"}[k]
            st = tc.stack_analysis(img.output, [entry])
            own = tc.stack_analysis(img.output, [entry], include=lambda name: name.startswith("spt_"))
            if entry in st:
                row["stack"] = st[entry].bytes
                row["stack_path"] = list(st[entry].path)
                row["stack_exact"] = st[entry].exact
                row["stack_own"] = own[entry].bytes if entry in own else None
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(row), encoding="utf-8")
    _FP_CACHE[key] = row
    return row


_REF_FLASH = {"pid": (520, 560, 40), "lti": (220, 240, 24), "obs": (900, 950, 80), "kf": (800, 850, 80)}
_REF_CORE = {"none": 1.6, "sp": 1.0, "dp": 1.0}
_REF_OPT = {"-O0": 2.2, "-Os": 1.0, "-O2": 1.25, "-O3": 1.6}


def footprint(spec: Any, opts: ControlCodegenOptions, compilers: tuple[str, ...] | None = None) -> list[FootprintRow]:
    """Derleyici başına flash / RAM / yığın. Yerel derleyici varsa ölçülür, yoksa tahmin edilir."""
    spec = controller_spec(spec)
    check_options(opts)
    prof = profile(opts.mcu)
    rows: list[FootprintRow] = []
    title = kind_title(spec)
    for comp in tuple(compilers or (opts.compiler,)):
        if comp not in COMPILERS:
            raise ControlError(f"Bilinmeyen derleyici: {comp!r}")
        if comp == "iar" and prof.arch == "xtensa":
            continue
        base_prof = prof if prof.measurable else MCU_PROFILES[prof.estimate_base]
        src_comp = "gcc" if comp == "iar" else comp
        d = _measure(spec, opts, base_prof, src_comp) if base_prof.measurable else None
        if d is None:
            f, fl, st = _REF_FLASH[spec.kind]
            k = _REF_CORE[prof.fpu] * _REF_OPT[opts.opt] * prof.density * (IAR_SIZE_FACTOR if comp == "iar" else 1.0)
            rows.append(FootprintRow(title, comp, int(round(f * k)), int(round(fl * k)), state_size(spec, opts.real),
                                     int(round(st * prof.density)), False,
                                     f"tahmin: {COMPILERS[src_comp]} bulunamadı; kaba referanstan ölçeklendi "
                                     f"(×{k:.2f}); RAM durum = sizeof (kesin)"))
            continue
        tool = tc.cross_tool(src_comp)
        label = tool.label if tool else src_comp
        stack_note = ""
        if d.get("stack") is not None:
            libs = [f for f in d.get("stack_path", []) if not f.startswith("spt_")]
            stack_note = (f"; yığın: objdump statik analizi, en kötü yol {' → '.join(d['stack_path'][:4])}"
                          + (f" (kütüphane: {' → '.join(libs)})" if libs else ""))
            if d.get("stack_exact") is False:
                stack_note += " — çözülemeyen yığın ayarı var, alt sınır"

        def sc(v, kf: float):
            return None if v is None else int(round(v * kf))

        if comp == "iar":
            kf = IAR_SIZE_FACTOR * prof.density
            rows.append(FootprintRow(title, comp, sc(d["flash_code"], kf), sc(d["flash_with_libm"],
                                                                              IAR_LIBM_FACTOR * prof.density),
                                     d["ram_state"], sc(d["stack"], prof.density), False,
                                     f"tahmin: IAR yok — {label} {opts.opt} ({base_prof.name}) ölçümü × "
                                     f"{IAR_SIZE_FACTOR}; RAM durum = sizeof (kesin)",
                                     sc(d["stack_own"], prof.density)))
        elif prof.measurable:
            rows.append(FootprintRow(title, comp, d["flash_code"], d["flash_with_libm"], d["ram_state"], d["stack"],
                                     True, f"{label} {opts.opt}, {prof.name}, {opts.real}: ölçüldü (nesne "
                                     f".text+.rodata+.data; kütüphane dahil: newlib-nano ile bağlanmış imaj − boş "
                                     f"temel; RAM: sizeof){stack_note}", d["stack_own"]))
        else:
            kf = prof.density
            rows.append(FootprintRow(title, comp, sc(d["flash_code"], kf), sc(d["flash_with_libm"], kf),
                                     d["ram_state"], sc(d["stack"], kf), False,
                                     f"tahmin: {prof.arch} derleyicisi yok — {label} {opts.opt} ({base_prof.name}) "
                                     f"ölçümü × {kf} kod yoğunluğu; RAM durum = sizeof (kesin)",
                                     sc(d["stack_own"], kf)))
    return rows


# ============================================================================ maliyet
def _cost_inputs(spec: Spec, n: int) -> np.ndarray:
    """Maliyet ölçümü için belirlenimci girişler (doyuma giren ve çıkan PID dahil)."""
    rng = np.random.default_rng(COST_SEED)
    nin, _ = _io_dims(spec)
    t = np.arange(n)
    x = 0.5 * np.sin(2 * np.pi * t[:, None] * (0.003 + 0.002 * np.arange(nin))[None, :]) + \
        0.05 * rng.standard_normal((n, nin))
    if isinstance(spec, PidSpec):
        x[:, 0] = np.where((t // 300) % 2 == 0, 1.0, -1.0) * 2.0      # büyük basamaklar: doyum
    return x


def cost(spec: Any, opts: ControlCodegenOptions) -> list[CostRow]:
    """Adım başına işlem sayısı (ölçülür) → çevrim / süre / CPU yükü (MCU tablosundan tahmin)."""
    spec = controller_spec(spec)
    check_options(opts)
    prof = profile(opts.mcu)
    fs = _rate(spec, opts)
    weights = _op_weights(prof, opts.real)
    x = _cost_inputs(spec, COST_SAMPLES)
    rows = []
    variants = [(False, "normal")]
    if isinstance(spec, PidSpec):
        variants.append((True, "izleme modu"))
    for track, label in variants:
        run_spec = replace(spec, tracking=track) if isinstance(spec, PidSpec) else spec
        run = run_generated(run_spec, x, count_ops=True)
        ops = run.ops[1:]
        cyc = ops @ weights
        cycles = float(cyc.mean())
        us = cycles / opts.clock_mhz
        rows.append(CostRow(f"{kind_title(spec)} ({label})", {op: float(v) for op, v in zip(OPS, ops.mean(0))},
                            cycles, us, us * fs / 1e4, 1e6 / us if us > 0 else math.inf,
                            f"işlem sayısı ölçüldü (C++ sayaç, {len(ops)} adım), çevrim çekirdek tablosundan tahmin "
                            f"({prof.name}, {opts.real}"
                            + (", double yazılım kayan nokta" if opts.real == "double" and prof.double_is_soft()
                               else "") + ")", False, float(cyc.max()), float(cyc.max()) / opts.clock_mhz))
    return rows


# ============================================================================ kullanım kılavuzu
def _fp_table(opts: ControlCodegenOptions, rows: list[FootprintRow]) -> str:
    lines = ["| Derleyici | Flash (kod) | Flash (kütüphane dahil) | RAM (durum) | Yığın (kütüphane dahil / bu kod) | "
             f"Flash % ({opts.flash_kb:g} kB) | RAM % ({opts.ram_kb:g} kB) | Ölçüm |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        flash = r.flash_with_libm if r.flash_with_libm is not None else r.flash_code
        ram = r.ram_state + (r.stack or 0)
        lines.append(f"| {COMPILERS[r.compiler]} | {_fmt_int(r.flash_code)} B | {_fmt_int(r.flash_with_libm)}"
                     f"{' B' if r.flash_with_libm is not None else ''} | {_fmt_int(r.ram_state)} B | "
                     f"{_fmt_int(r.stack)} / {_fmt_int(r.stack_own)} B | {100 * flash / (opts.flash_kb * 1024):.2f} | "
                     f"{100 * ram / (opts.ram_kb * 1024):.2f} | {'ölçüldü' if r.measured else 'tahmin'} |")
    methods = sorted({r.method for r in rows})
    return "\n".join(lines) + "\n\n" + "\n".join(f"- {m}" for m in methods)


def _cost_table(opts: ControlCodegenOptions, fs: float, rows: list[CostRow]) -> str:
    lines = [f"| Kip | Çevrim / adım (ort.) | En kötü adım | µs / adım @ {opts.clock_mhz:g} MHz | CPU yükü @ {fs:g} Hz "
             "| En yüksek hız |", "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r.algorithm} | {_fmt_int(r.cycles)} | {_fmt_int(r.cycles_max)} | {r.us_per_update:.2f} | "
                     f"%{r.cpu_load_percent:.3f} | {_fmt_int(r.max_rate_hz)} Hz |")
    used = [o for o in OPS if any(r.ops[o] > 0 for r in rows)]
    ops = ["| Kip | " + " | ".join(used) + " |", "|---|" + "---:|" * len(used)]
    for r in rows:
        ops.append(f"| {r.algorithm} | " + " | ".join(f"{r.ops[o]:.1f}" for o in used) + " |")
    return ("\n".join(lines) + "\n\n" + (rows[0].method if rows else "")
            + "\n\nAdım başına ortalama işlem sayıları (ölçülen):\n\n" + "\n".join(ops))


def _api_md(spec: Spec, p: str, P: str) -> str:
    if isinstance(spec, PidSpec):
        return f"""| Fonksiyon | Açıklama |
|---|---|
| `{p}_pid_init(&s)` | Varsayılan parametreler (`{P}_PID_KP` …) + sıfır durum |
| `{p}_pid_step(&s, r, y)` | Bir örnek → doyumlu çıkış. 1-DOF: `{p}_pid_step(&s, e, 0)` |
| `{p}_pid_step_track(&s, r, y, tr)` | İzleme modu: integratör `tr`'ye yaklaşır (elle → otomatik sarsıntısız) |
| `{p}_pid_set_gains(&s, kp, ki, kd, tf)` | Çalışma anında paralel kazançlar (geçersizse 0 döner, değişmez) |
| `{p}_pid_set_standard(&s, kp, ti, td, n)` | Standart biçim (pidstd): `ti ≤ 0` integral yok, `n ≤ 0` süzgeç yok |
| `{p}_pid_set_limits(&s, u_min, u_max)` | Çıkış sınırları (`{P}_PID_LIMIT 1` iken etkili) |
| `{p}_pid_set_integrator(&s, xi)` | Integratör durumunu ön yükleme |
| `{p}_pid_reset(&s)` | Durumu sıfırlar, parametreler korunur |

Doğrudan yazılabilen alanlar: `s.b`, `s.c`, `s.kb`, `s.kt` (ve `kp`, `ki`). `tf` ve `kd` değişimi için
`{p}_pid_set_gains` kullanın (türetilmiş katsayı `cd` yeniden hesaplanır ve kararlılık denetlenir)."""
    if isinstance(spec, LtiSpec):
        if _lti_realization(spec)["mode"] == "sos":
            return f"""| Fonksiyon | Açıklama |
|---|---|
| `{p}_lti_init(&s)` | Durumu sıfırlar |
| `{p}_lti_step(&s, u)` | Bir örnek: giriş (genellikle e = r − y) → çıkış |"""
        return f"""| Fonksiyon | Açıklama |
|---|---|
| `{p}_lti_init(&s)` | Durumu sıfırlar |
| `{p}_lti_step(&s, u, y)` | Bir örnek: `u[{P}_LTI_NU]` → `y[{P}_LTI_NY]`; geçersiz girişte 0 |"""
    if isinstance(spec, ObserverSpec) and spec.is_state:
        return f"""| Fonksiyon | Açıklama |
|---|---|
| `{p}_obs_init(&s)` | LQI'da xi = 0, son çıkış 0, sınırlar varsayılan |
| `{p}_obs_output(&s, x, r, u)` | Ölçülen durum `x[{P}_OBS_NX]` → kontrol çıkışı (durum ilerlemez); `r` yoksa NULL |
| `{p}_obs_update(&s, x, r, ua)` | LQI integratörünü **uygulanan** çıkış `ua` ile ilerletir (LQI yoksa çıkışı saklar) |
| `{p}_obs_step(&s, x, r, u)` | `output` + `update(u)` |

Eyleyici çıkışı kendisi sınırlıyorsa `output` → uygula → `update(gerçek değer)` sırasını kullanın."""
    if isinstance(spec, ObserverSpec):
        return f"""| Fonksiyon | Açıklama |
|---|---|
| `{p}_obs_init(&s)` | x̂ = 0 (LQI'da xi = 0), sınırlar varsayılan |
| `{p}_obs_output(&s, y, r, u)` | Kontrol çıkışı (durum ilerlemez); referans yoksa `r = NULL` |
| `{p}_obs_update(&s, y, r, ua)` | Kestiriciyi **uygulanan** çıkış `ua` ile ilerletir |
| `{p}_obs_step(&s, y, r, u)` | `output` + `update(u)` |
| `{p}_obs_state(&s, xhat)` | Kontrol yasasının kullandığı kestirim |

Eyleyici çıkışı kendisi sınırlıyorsa (PWM doygunluğu, motor sürücüsü) `output` → uygula → `update(gerçek değer)`
sırasını kullanın: gözlemci bitkiye giren gerçek girişle ilerlemezse kestirim sapar."""
    return f"""| Fonksiyon | Açıklama |
|---|---|
| `{p}_kf_init(&s)` | x̂ = 0 |
| `{p}_kf_step(&s, u, y, yhat, xhat)` | Bir örnek; ölçüm kullanıldıysa 1 (y'de NaN → yalnız tahmin, 0) |"""


def _semantics_md(spec: Spec) -> str:
    if isinstance(spec, PidSpec):
        return """\
- Doğrusal kısım MATLAB `pid` / `pidstd` / `pid2` nesnesiyle aynıdır: `C(z) = Kp + Ki·IF(z) + Kd/(Tf + DF(z))`,
  `IF / DF`: ileri Euler `Ts/(z−1)`, geri Euler `Ts·z/(z−1)`, yamuk `(Ts/2)(z+1)/(z−1)`. 2-DOF:
  `u = Kp(b·r − y) + Ki·IF·(r − y) + Kd/(Tf + DF)·(c·r − y)`.
- Uygulama Simulink "Discrete PID Controller (2DOF)" bloğunun yapısıdır: integratör `Ki·e`'yi biriktirir (durum çıkış
  birimindedir), türev süzgeci `d = (Kd·(c·r − y) − x_d)/(Tf + α·Ts)`, `x_d += Ts·d`.
- Anti-windup sinyalleri, integratörün **durumuyla** kurulan doyum öncesi toplamdan `a = P + D + x_I` hesaplanır
  (Simulink R2025b'nin iç durum kaydından ölçüldü; geri Euler / yamukta da cebirsel döngü yoktur):
  **geri hesaplama** integratör girişi `Ki·e + Kb·(sat(a) − a)`; **izleme** girişe `Kt·(tr − sat(a))` eklenir;
  **sıkıştırma** `a` doyumdayken aşma yönü ile integratör girişi (`Ki·e`, izlemede `+ Kt·(tr − sat(a))`) aynı
  işaretliyse integratör girişi 0. Çıkış
  `u = sat(a + β·Ts·w)`, β = 0 (ileri Euler), 1 (geri Euler), ½ (yamuk) — integratörün doğrudan geçişi.
- Standart (ideal) biçimden gelen kazançlar paralel eşdeğere çevrilir (`Ki = Kp/Ti`, `Kd = Kp·Td`, `Tf = Td/N`);
  Simulink ideal biçimindeki `Kb`, `Kt` (Kp'den önce) paralel eşdeğerde `Kp·Kb`, `Kp·Kt` olur."""
    if isinstance(spec, LtiSpec):
        return """\
- SISO kontrolcü, kutup / sıfırlarından **ikinci derece bölümlere** (devrik doğrudan biçim II) ayrılır: yüksek
  dereceli tek polinom biçimine göre katsayı yuvarlamasına çok daha az duyarlıdır. Bölümler, kutbu birim çembere en
  yakın olan sona gelecek biçimde sıralanır ve eşleştirilir (en yakın sıfır); kazanç ilk bölümdedir. Nedensel
  gecikmeler (pay derecesi < payda derecesi) bölümlerin payına z⁻¹ olarak dağıtılır. Üretimde bölümlerin çarpımının
  birim çember üzerinde aktarım fonksiyonuyla aynı olduğu (göreli < 1e-7) doğrulanır.
- MIMO kontrolcü durum uzayıdır; A, 2'nin kuvvetleriyle köşegen benzerlikle dengelenir (yuvarlamasız: giriş / çıkış
  davranışı bit düzeyinde değişmez, ara büyüklükler dengelenir)."""
    if isinstance(spec, ObserverSpec) and spec.is_state:
        return """\
- Tüm durumlar ölçülür (gözlemci yok): `u = −K·x` — MATLAB `dlqr` / `lqr` / `place` kazancıyla; `Nbar` ile
  `u = −K·x + Nbar·r` (kapalı çevrimin DC kazancı birim; modele duyarlı).
- LQI (`lqi`, `K = [Kx Ki]`): `u = −Kx·x − Ki·xi`, `xi[n+1] = xi[n] + Ts·(r − y)` (MATLAB'in ayrık tanımı), `y`
  ölçülen durumdan modelle `y = C·x + D·u` (uygulanan u) hesaplanır. Integratörün doyuma karşı anti-windup'ı yoktur
  (MATLAB tanımı); büyük referans basamaklarında referansı yumuşatın.
- Ölçülen durum, tasarımda kullanılan modelin durum koordinatlarında olmalıdır (ör. ss2ss / ölçekleme sonrası
  dönüşümü uygulayın)."""
    if isinstance(spec, ObserverSpec):
        return """\
- 'delayed': MATLAB `reg(sys, K, L)` / `estim` ve 'delayed' Kalman ile `lqgreg`; 'current': 'current' Kalman
  (`kalman(..., 'current')`) ile `lqgreg` / `lqgtrack` — aynı denklemler. Kalman tasarımından gelen `L` ve `Mx`
  birlikte verilir; kutup yerleştirmeyle tasarlanan 'current' gözlemcide `M` verilmezse `L = A·M` (ders kitabı
  "current estimator") bağıntısıyla `M = A⁻¹·L` alınır.
- `Nbar`: kapalı çevrimin DC kazancını birim yapar; model hatasına duyarlıdır. Kalıcı durum hatası istenmiyorsa LQI
  (`lqi`, integral durumu `xi[n+1] = xi[n] + Ts·(r − y)` — MATLAB'in ayrık tanımı) kullanın.
- LQI integratörünün doyuma karşı anti-windup'ı yoktur (MATLAB tanımı); büyük referans basamaklarında çıkış sınırları
  ile birlikte kullanıyorsanız referansı yumuşatın."""
    return """\
- MATLAB `kalman(sys, Qn, Rn, Nn, 'current' | 'delayed')` sonucuyla aynı denklemler: `L` (tahmin) ve `M` (`Mx`,
  yenilik) kazançları kararlı durum değerleridir (Riccati çözümü üretimde yapılmıştır; MCU'da matris tersi yok).
- 'current' çıktısı `ŷ[n|n] = C·x̂[n|n−1] + D·u + My·ν`; `My` verilmezse `My = C·Mx` (süreç gürültüsü ölçüme
  doğrudan girmiyorsa — `H = 0` — MATLAB'in My'si budur) yani `ŷ = C·x̂[n|n] + D·u`.
- Çıkış düzeni MATLAB `kalmf` ile aynı: `[ŷ; x̂]`."""


def usage_markdown(spec: Any, opts: ControlCodegenOptions, footprint_rows: list[FootprintRow] | None = None,
                   cost_rows: list[CostRow] | None = None) -> str:
    """Türkçe kullanım kılavuzu (KULLANIM_<önek>.md)."""
    spec = controller_spec(spec)
    check_options(opts)
    p, P = opts.prefix.lower(), opts.prefix.upper()
    k = spec.kind
    prof = profile(opts.mcu)
    fs = _rate(spec, opts)
    rate_warn = ""
    if abs(fs * spec.Ts - 1.0) > 1e-9:
        rate_warn = (f"\n> **Uyarı:** kontrolcü Ts = {spec.Ts:g} s ({1 / spec.Ts:g} Hz) için ayrıklaştırılmıştır; "
                     f"seçilen çağrı hızı {fs:g} Hz. Farklı hızda çağırmak dinamiği değiştirir — tasarımı bu hız için "
                     "yenileyin.\n")
    real_note = ""
    if opts.real == "double" and prof.double_is_soft():
        real_note = (f"\n> **Uyarı:** `double` seçildi ama {prof.name} çekirdeğinde çift hassasiyet donanımı yok; tüm "
                     "hesap yazılım kayan noktayla yapılır (10–100× yavaş).\n")
    fpu = {"none": "FPU yok: tüm kayan nokta yazılımda (float bile onlarca çevrim)",
           "sp": "tek hassasiyet FPU: `float` önerilir; `double` yazılım kayan noktadır",
           "dp": "çift hassasiyet FPU: `double` de donanımda (yine de `float` daha hızlı ve küçük)"}[prof.fpu]
    flags = " ".join(prof.gcc_flags) if prof.gcc_flags else "(çekirdek bayrakları için üreticinin SDK'sına bakın)"
    fp_md = _fp_table(opts, footprint_rows) if footprint_rows else \
        "_Bu dosya ölçümsüz üretildi; ayak izi tablosu için ölçümle yeniden üretin._"
    cost_md = _cost_table(opts, fs, cost_rows) if cost_rows else \
        "_Bu dosya ölçümsüz üretildi; maliyet tablosu için ölçümle yeniden üretin._"
    nan_md = {"pid": "sonlu olmayan `r`, `y` (izlemede `tr`): durum değişmez, son çıkış döner",
              "lti": "sonlu olmayan giriş: durum değişmez, son çıkış döner (MIMO'da 0 dönüş değeri)",
              "obs": f"sonlu olmayan `{'x' if k == 'obs' and spec.is_state else 'y'}` / `r`: `output` son çıkışı "
                     "yazar ve 0 döner; `update` durumu değiştirmez",
              "kf": "`y`'de sonlu olmayan değer: o örnekte ölçüm güncellemesi atlanır (yalnız tahmin, 0 döner); "
                    "sonlu olmayan `u`: durum değişmez"}[k]
    return f"""# {_title(spec, opts)} — C kodu kullanım kılavuzu

Signal Processing Toolbox (Kontrol Tasarımı) tarafından üretildi.

**Kontrolcü:** {describe(spec)}

**Seçimler:** sayı türü `{opts.real}` · MCU {prof.name} ({prof.examples}) · derleyici {COMPILERS[opts.compiler]}
`{opts.opt}` · saat {opts.clock_mhz:g} MHz · çağrı hızı {fs:g} Hz · önek `{p}`
{rate_warn}{real_note}
## Dosyalar

| Dosya | İçerik |
|---|---|
| `{p}_{k}.h` | Sayı türü (`{P}_REAL`), yapılandırma makroları, durum yapısı `{p}_{k}_t`, API |
| `{p}_{k}.c` | Uygulama (katsayılar `static const`: flash'ta) |
| `{p}_{k}_ornek.c` | Sabit periyotla çağıran örnek (sensör / eyleyici erişimi kullanıcıya ait) |

`malloc` yok, global değişken yok, fonksiyonlar yeniden girişlidir: tüm durum çağıranın yapısındadır. Aynı kodla
birden çok kontrolcü (ör. roll / pitch / yaw) ayrı yapılarla çalışır; farklı katsayılı kontrolcüler için **farklı
önekle** üretin (dosya ve ad çakışması olmaz).

## API

{_api_md(spec, p, P)}

## Anlam (MATLAB / Simulink karşılığı)

{_semantics_md(spec)}

## Sabit örnekleme zamanı

- Katsayılar Ts = {spec.Ts:g} s için hesaplanmıştır: kontrolcüyü **tam bu periyotla** çağırın (donanım zamanlayıcı
  kesmesi ya da `vTaskDelayUntil`). Kesmede uzun hesap istemiyorsanız kesmede yalnız bayrak kurup hesabı yüksek
  öncelikli görevde yapın; ölçüm → hesap → eyleyici gecikmesini sabit tutun.
- Değişken dt desteklenmez (ayrık tasarım sabit Ts varsayar). Hesap süresini `DWT->CYCCNT` ile ölçün; aşağıdaki
  maliyet tablosu çekirdek tablosundan **tahmindir**.

## Geçersiz (NaN / sonsuz) giriş

{nan_md[0].upper() + nan_md[1:]}. Denetim `{p}_ctrl_finite()` (`v == v` ve sınır karşılaştırması) ile yapılır:
**`-ffast-math` / `-ffinite-math-only` ile derlemeyin** (denetim kaldırılır, NaN durumu kalıcı bozar).

## Projeye ekleme

### STM32CubeIDE (GCC)
1. `.h` dosyasını `Core/Inc`, `.c` dosyasını `Core/Src` altına kopyalayın (örnek dosya isteğe bağlı).
2. *Properties → C/C++ Build → Settings → MCU Settings*: FPU ve ABI. Bu çekirdek için bayraklar: `{flags}`.
3. İsteğe bağlı: *Preprocessor* altında `{P}_USE_DOUBLE=0/1` ve parametre makroları (`#ifndef` korumalı).
4. Kod yalnız temel aritmetik kullanır (libm gerekmez).

### Keil MDK (Arm Compiler 6)
Dosyaları bir *Group*'a ekleyin; include yolunu *Options for Target → C/C++ (AC6)* altına yazın. *Floating Point
Hardware*: M4F / M33F "Single Precision", M7 "Double Precision". Dil C99 ya da üstü; *Fast math* kapalı. AC6 clang
tabanlıdır: boyutlar "Clang / Arm Compiler 6" satırlarında yerelde ölçülmüştür.

### IAR EWARM
Dosyaları projeye ekleyin; *General Options → Target → FPU* ayarını çekirdeğe göre seçin; dil C99. IAR bu makinede
olmadığından IAR boyutları **tahmindir** (GCC ölçümü × {IAR_SIZE_FACTOR}).

## Sayı türü ve FPU

{prof.name}: {fpu}. `{P}_USE_DOUBLE` ile seçilir. Sabitler `{P}_C()` ile doğru türde yazılır (`-Wdouble-promotion`
temiz). Kod -std=c99 -Wall -Wextra -pedantic -Werror (+ float'ta -Wdouble-promotion) ile tüm Cortex-M profillerinde
GCC ve Clang'da uyarısız derlenecek biçimde doğrulanmıştır.

## Ölçülen ayak izi

{fp_md}

## Hesap maliyeti

{cost_md}

Maliyet modeli: çevrim = Σ temel işlem × (çekirdek çevrimi + bellek / dal ek yükü). İşlem sayıları **ölçülmüştür**
(üretilen C ana makinede C++ sayaç türüyle çalıştırılır); çekirdek çevrim tabloları **tahmindir**
(`spt/codegen/mcu.py`). CPU yükü = µs/adım × çağrı hızı / 10⁴ (%).

## Doğrulama

Üretilen C, ana makinede derlenip MATLAB R2025b sonuçlarıyla karşılaştırılmıştır (double'da en büyük fark ~1e-12
düzeyinde): `pid` / `pidstd` / `pid2` nesnelerinin `lsim` çıktısı, Simulink "Discrete PID Controller (2DOF)"
bloğunun doyum + anti-windup + izleme modlu çıktısı, `tf` / `ss` kontrolcülerin `lsim` çıktısı, `reg` / `estim`
(delayed / current), `lqi` / `lqgtrack`, gözlemcisiz `dlqr` + Nbar ve `lqi` kapalı çevrimi, `kalman` (current /
delayed; gürültü ölçüme doğrudan girerken My ile). Ayrıntı: `tests/test_codegen_control.py`.
"""
