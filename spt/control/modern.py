"""Modern (durum uzayı) kontrolcü tasarımı — MATLAB Control System Toolbox karşılıkları.

Fonksiyonlar MATLAB adlarını ve dönüş sıralarını korur:

- `care(A, B, Q, R, S, E)` / `dare(...)` → `(X, K, L)` (MATLAB `icare` / `idare`: çözüm, kazanç, kapalı çevrim
  özdeğerleri).
- `lqr(sys | A, B, Q, R, N)` → `(K, S, P)`; `dlqr`; `lqrd` (Van Loan ile ayrık eşdeğer maliyet); `lqi` (integral
  eklemeli); `lqry` (çıkış ağırlıklı).
- `place(A, B, p)` (dayanıklı yerleştirme: scipy `place_poles`, Yang–Tits; MATLAB'in KNV yöntemiyle aynı amaç —
  çok girişlide kazanç tek değildir, kapalı çevrim özdeğerleri aynıdır), `acker(A, b, p)` (Ackermann, tek giriş).
- `kalman(sys, Qn, Rn, Nn, type=)` → `(kest, L, P, Mx, Z)` (+ `My` özniteliği); `kalmd`; `lqg`; `reg`; `estim`.
- `nbar` (kapalı çevrim kalıcı durum kazanç düzeltmesi), `bryson` (Bryson kuralıyla Q, R).

Girdi modeli ördek tiplemeyle alınır: A, B, C, D (ve Ts) öznitelikli nesne (spt.control.lti.StateSpace), num / den
öznitelikli transfer fonksiyonu (spt.control.lti ile durum uzayına çevrilir) ya da `(A, B, C, D[, Ts])` demeti.
Çıkan modeller spt.control.lti.StateSpace'tir (modül yoksa aynı öznitelikli `SSData`).

Riccati denklemleri scipy'nin Schur tabanlı çözücüleriyle (`solve_continuous_are`, `solve_discrete_are`) çözülür;
çözüm, kalıntı ve kapalı çevrim kararlılığı ile doğrulanır. Kararlılaştırılamayan / tespit edilemeyen sistemler,
pozitif tanımlı olmayan ağırlıklar ve çözümsüz durumlar Türkçe `ControlError` verir.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.linalg as sla
from scipy.signal import place_poles

from spt.control.errors import ControlError, ControlWarning

__all__ = ["SSData", "KalmanResult", "care", "dare", "lqr", "dlqr", "lqrd", "lqi", "lqry", "place", "acker",
           "kalman", "kalmd", "lqg", "reg", "estim", "nbar", "bryson", "ss_data", "make_ss"]

_EPS = float(np.finfo(float).eps)


# ---------------------------------------------------------------------------------------------------------------
# Model girdi / çıktı yardımcıları
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class SSData:
    """spt.control.lti yokken dönen durum uzayı verisi (aynı öznitelik adları)."""
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    Ts: float = 0.0
    input_names: tuple = ()
    output_names: tuple = ()
    state_names: tuple = ()
    name: str = ""

    @property
    def nx(self) -> int:
        return self.A.shape[0]

    @property
    def ninputs(self) -> int:
        return self.D.shape[1]

    @property
    def noutputs(self) -> int:
        return self.D.shape[0]

    @property
    def is_discrete(self) -> bool:
        return self.Ts > 0


def make_ss(A, B, C, D, Ts: float = 0.0, *, input_names=None, output_names=None, state_names=None, name=""):
    """Durum uzayı modeli: spt.control.lti varsa StateSpace, yoksa SSData."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    D = np.asarray(D, dtype=float)
    try:
        from spt.control import lti  # tembel: LTI çekirdeği ayrı modülde
    except ImportError:
        return SSData(A, B, C, D, float(Ts), tuple(input_names or ()), tuple(output_names or ()),
                      tuple(state_names or ()), name)
    return lti.StateSpace(A, B, C, D, float(Ts), input_names=input_names, output_names=output_names,
                          state_names=state_names, name=name)


def _is_model(x: Any) -> bool:
    if isinstance(x, tuple) and len(x) in (4, 5):
        return True
    return all(hasattr(x, a) for a in ("A", "B", "C", "D")) or (hasattr(x, "num") and hasattr(x, "den"))


def ss_data(sys: Any, what: str = "Model") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """(A, B, C, D, Ts) — ördek tiplemeyle. Ayrık modeldeki gecikmeler durum olarak katılır (tam); sürekli
    zamanda gecikmeli model hata verir (önce Padé yaklaşımı)."""
    if isinstance(sys, tuple) and len(sys) in (4, 5):
        A, B, C, D = sys[:4]
        Ts = float(sys[4]) if len(sys) == 5 else 0.0
    else:
        if not all(hasattr(sys, a) for a in ("A", "B", "C", "D")):
            if not (hasattr(sys, "num") and hasattr(sys, "den")):
                raise ControlError(f"{what} tanınmadı: durum uzayı (A, B, C, D) ya da transfer fonksiyonu "
                                   "bekleniyordu.")
            try:
                from spt.control import lti
            except ImportError:
                raise ControlError(f"{what} durum uzayına çevrilemedi (LTI çekirdeği yok).") from None
            sys = lti.ss(sys)
        has_delay = getattr(sys, "has_delay", None)
        if callable(has_delay) and has_delay():
            Ts0 = float(getattr(sys, "Ts", 0.0) or 0.0)
            if Ts0 > 0:
                from spt.control import lti
                sys = lti.ss_absorb(sys)
            else:
                raise ControlError(f"{what} gecikme içeriyor; sürekli zamanda durum uzayı tasarımı için önce gecikmeyi "
                                   "Padé yaklaşımıyla rasyonel hale getirin ya da modeli ayrıklaştırın.")
        A, B, C, D = sys.A, sys.B, sys.C, sys.D
        Ts = getattr(sys, "Ts", None)
        if Ts is None:
            Ts = getattr(sys, "dt", 0.0)
        Ts = 0.0 if Ts in (None, False) else float(Ts)
    if Ts < 0:
        raise ControlError("Belirtilmemiş örnekleme zamanlı (Ts = −1) model desteklenmez.")
    A = np.atleast_2d(np.asarray(A, dtype=float))
    D = np.atleast_2d(np.asarray(D, dtype=float))
    n = A.shape[0] if A.size else 0
    B = np.asarray(B, dtype=float).reshape(n, -1) if n else np.zeros((0, D.shape[1]))
    C = np.asarray(C, dtype=float).reshape(-1, n) if n else np.zeros((D.shape[0], 0))
    A = A if n else np.zeros((0, 0))
    if A.shape != (n, n) or B.shape[0] != n or C.shape[1] != n or D.shape != (C.shape[0], B.shape[1]):
        raise ControlError(f"{what}: durum uzayı matris boyutları uyumsuz.")
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(B)) and np.all(np.isfinite(C)) and np.all(np.isfinite(D))):
        raise ControlError(f"{what}: matrislerde sonlu olmayan (NaN / sonsuz) değer var.")
    return A, B, C, D, Ts


def _mat(x: Any, what: str, shape: tuple[int, int] | None = None) -> np.ndarray:
    try:
        M = np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        raise ControlError(f"{what} gerçel sayısal matris olmalı.") from None
    if M.ndim == 0:
        M = M.reshape(1, 1)
    elif M.ndim == 1:
        M = M.reshape(1, -1) if shape is None or shape[0] == 1 else M.reshape(-1, 1)
    if M.ndim != 2:
        raise ControlError(f"{what} 2-B matris olmalı.")
    if shape is not None and M.shape != shape:
        if M.size == 1 and shape[0] == shape[1] and shape[0] > 0:
            M = M[0, 0] * np.eye(shape[0])          # skaler ağırlık → skaler·I (MATLAB'de boyut hatası)
        else:
            raise ControlError(f"{what} boyutu {M.shape[0]}×{M.shape[1]}; {shape[0]}×{shape[1]} bekleniyordu.")
    if not np.all(np.isfinite(M)):
        raise ControlError(f"{what} sonlu olmayan değer içeriyor.")
    return M


def _bmat(B: Any, n: int) -> np.ndarray:
    """Giriş matrisi (n×m); 1-B vektör tek giriş sütunu sayılır."""
    if np.ndim(B) == 1 and np.size(B) == n:
        B = np.reshape(np.asarray(B, dtype=float), (n, 1))
    B = _mat(B, "B matrisi")
    if B.shape[0] != n:
        raise ControlError(f"B matrisinin satır sayısı ({B.shape[0]}) durum sayısıyla ({n}) aynı olmalı.")
    return B


def _sym(M: np.ndarray, what: str) -> np.ndarray:
    scale = max(1.0, float(np.max(np.abs(M)))) if M.size else 1.0
    if M.size and float(np.max(np.abs(M - M.T))) > 1e-10 * scale:
        raise ControlError(f"{what} simetrik olmalı.")
    return 0.5 * (M + M.T)


def _check_pd(R: np.ndarray, what: str) -> None:
    try:
        np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        raise ControlError(f"{what} pozitif tanımlı olmalı.") from None


def _check_psd(W: np.ndarray, what: str) -> None:
    if not W.size:
        return
    ev = np.linalg.eigvalsh(W)
    if ev[0] < -1e-10 * max(1.0, float(np.max(np.abs(ev)))):
        raise ControlError(f"{what} pozitif yarı tanımlı olmalı (en küçük özdeğer {ev[0]:.3g}).")


def _stable(ev: np.ndarray, discrete: bool) -> bool:
    if ev.size == 0:
        return True
    return bool(np.all(np.abs(ev) < 1.0)) if discrete else bool(np.all(ev.real < 0.0))


def _pbh_fail(A: np.ndarray, B: np.ndarray, discrete: bool, only_unstable: bool = True) -> complex | None:
    """Kontrol edilemeyen (only_unstable: kararsız ya da sınırda) özdeğer varsa onu döndürür (PBH testi)."""
    n = A.shape[0]
    if n == 0:
        return None
    scale = max(1.0, float(np.linalg.norm(np.hstack([A, B]), 2)))
    for lam in np.linalg.eigvals(A):
        if only_unstable:
            if discrete and abs(lam) < 1.0 - 1e-9:
                continue
            if not discrete and lam.real < -1e-9 * max(1.0, abs(lam)):
                continue
        M = np.hstack([A - lam * np.eye(n), B.astype(complex)])
        smin = np.linalg.svd(M, compute_uv=False)[-1]
        if smin <= 1e-10 * scale:
            return complex(lam)
    return None


def _fmt_c(z: complex) -> str:
    z = complex(z)
    if abs(z.imag) < 1e-12 * max(1.0, abs(z)):
        return f"{z.real:.4g}"
    return f"{z.real:.4g}{'+' if z.imag >= 0 else '−'}{abs(z.imag):.4g}j"


# ---------------------------------------------------------------------------------------------------------------
# Riccati denklemleri (MATLAB icare / idare)
# ---------------------------------------------------------------------------------------------------------------
def _riccati_inputs(A, B, Q, R, S, E):
    A = _mat(A, "A matrisi")
    n = A.shape[0]
    if A.shape != (n, n):
        raise ControlError("A matrisi kare olmalı.")
    B = _bmat(B, n)
    m = B.shape[1]
    Q = _sym(_mat(Q, "Q matrisi", (n, n)), "Q matrisi")
    R = np.eye(m) if R is None or (np.size(R) == 0) else _sym(_mat(R, "R matrisi", (m, m)), "R matrisi")
    S = np.zeros((n, m)) if S is None or np.size(S) == 0 else _mat(S, "Çapraz ağırlık (N / S) matrisi", (n, m))
    E = None if E is None or np.size(E) == 0 else _mat(E, "E matrisi", (n, n))
    return A, B, Q, R, S, E


def _residual(A, B, Q, R, S, E, X, discrete: bool) -> float:
    Em = np.eye(A.shape[0]) if E is None else E
    if discrete:
        G = B.T @ X @ B + R
        F = A.T @ X @ B + S
        res = A.T @ X @ A - Em.T @ X @ Em - F @ np.linalg.solve(G, F.T) + Q
        scale = (np.linalg.norm(A.T @ X @ A) + np.linalg.norm(Em.T @ X @ Em) + np.linalg.norm(Q)
                 + np.linalg.norm(F @ np.linalg.solve(G, F.T)))
    else:
        F = Em.T @ X @ B + S
        res = A.T @ X @ Em + Em.T @ X @ A - F @ np.linalg.solve(R, F.T) + Q
        scale = 2 * np.linalg.norm(A.T @ X @ Em) + np.linalg.norm(Q) + np.linalg.norm(F @ np.linalg.solve(R, F.T))
    return float(np.linalg.norm(res) / max(scale, 1e-300))


def _solve_are(A, B, Q, R, S, E, discrete: bool, what: str):
    """(X, K, L) — kararlılaştırıcı çözüm; başarısızlıkta Türkçe ControlError."""
    n, m = B.shape
    if n == 0:
        return np.zeros((0, 0)), np.zeros((m, 0)), np.zeros(0, dtype=complex)
    if not discrete:
        _check_pd(R, "R ağırlık matrisi")
    if E is not None and np.linalg.matrix_rank(E) < n:
        raise ControlError("E matrisi tekil olmamalı.")
    solver = sla.solve_discrete_are if discrete else sla.solve_continuous_are
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X = solver(A, B, Q, R, e=E, s=S)
    except (np.linalg.LinAlgError, ValueError) as exc:
        raise ControlError(f"{what}: Riccati denkleminin kararlılaştırıcı çözümü bulunamadı (sistem "
                           "kararlılaştırılamıyor ya da birim çember / sanal eksen üzerinde gözlenemeyen kip var). "
                           f"Ayrıntı: {exc}") from None
    if not np.all(np.isfinite(X)):
        raise ControlError(f"{what}: Riccati çözümü sonlu değil (sistem kararlılaştırılamıyor).")
    X = 0.5 * (X + X.T)
    Em = np.eye(n) if E is None else E
    if discrete:
        Gm = B.T @ X @ B + R
        try:
            K = np.linalg.solve(Gm, B.T @ X @ A + S.T)
        except np.linalg.LinAlgError:
            raise ControlError(f"{what}: B'XB + R tekil; ağırlıkları gözden geçirin.") from None
    else:
        K = np.linalg.solve(R, B.T @ X @ Em + S.T)
    L = sla.eigvals(A - B @ K, Em) if E is not None else np.linalg.eigvals(A - B @ K)
    if not _stable(L, discrete):
        raise ControlError(f"{what}: Riccati çözümü kapalı çevrimi kararlı yapmıyor (kararlılaştırıcı çözüm yok: "
                           "kararsız bir kip kontrol edilemiyor ya da gözlenemeyen kip sınırda).")
    rr = _residual(A, B, Q, R, S, E, X, discrete)
    if rr > 1e-6:
        raise ControlError(f"{what}: Riccati çözümü doğrulanamadı (göreli kalıntı {rr:.2g}); problem kötü koşullu.")
    if rr > 1e-10:
        warnings.warn(f"{what}: Riccati çözümünün göreli kalıntısı {rr:.2g} (kötü koşullu problem).", ControlWarning,
                      stacklevel=3)
    return X, K, L


def care(A, B, Q, R=None, S=None, E=None):
    """Sürekli cebirsel Riccati denklemi (MATLAB `icare`): A'XE + E'XA − (E'XB + S)R⁻¹(B'XE + S') + Q = 0.

    Dönüş `(X, K, L)`: kararlılaştırıcı çözüm X, kazanç K = R⁻¹(B'XE + S'), kapalı çevrim özdeğerleri
    L = eig(A − BK, E). R verilmezse birim matris. Kararlılaştırıcı çözüm yoksa ControlError."""
    A, B, Q, R, S, E = _riccati_inputs(A, B, Q, R, S, E)
    return _solve_are(A, B, Q, R, S, E, False, "Sürekli Riccati")


def dare(A, B, Q, R=None, S=None, E=None):
    """Ayrık cebirsel Riccati denklemi (MATLAB `idare`):
    A'XA − E'XE − (A'XB + S)(B'XB + R)⁻¹(A'XB + S)' + Q = 0.

    Dönüş `(X, K, L)`: K = (B'XB + R)⁻¹(B'XA + S'), L = eig(A − BK, E)."""
    A, B, Q, R, S, E = _riccati_inputs(A, B, Q, R, S, E)
    return _solve_are(A, B, Q, R, S, E, True, "Ayrık Riccati")


# ---------------------------------------------------------------------------------------------------------------
# LQR ailesi
# ---------------------------------------------------------------------------------------------------------------
def _lq_core(A, B, Q, R, N, discrete: bool, what: str = "LQR"):
    A = _mat(A, "A matrisi")
    n = A.shape[0]
    if A.shape != (n, n):
        raise ControlError("A matrisi kare olmalı.")
    B = _bmat(B, n)
    m = B.shape[1]
    Q = _sym(_mat(Q, "Q ağırlık matrisi", (n, n)), "Q ağırlık matrisi")
    R = _sym(_mat(R, "R ağırlık matrisi", (m, m)), "R ağırlık matrisi")
    N = np.zeros((n, m)) if N is None or np.size(N) == 0 else _mat(N, "N çapraz ağırlık matrisi", (n, m))
    _check_pd(R, "R ağırlık matrisi")
    _check_psd(np.block([[Q, N], [N.T, R]]), "[Q N; N' R] ağırlık matrisi")
    lam = _pbh_fail(A, B, discrete)
    if lam is not None:
        raise ControlError(f"{what}: sistem kararlılaştırılamaz — kararsız / sınırda kip (özdeğer {_fmt_c(lam)}) "
                           "girişlerden kontrol edilemiyor.")
    X, K, L = _solve_are(A, B, Q, R, N, None, discrete, what)
    return K, X, L


def _split_sys_args(args, name: str):
    """(sys, Q, R[, N]) ya da (A, B, Q, R[, N]) → (A, B, Q, R, N, Ts, sys_given)."""
    if not args:
        raise ControlError(f"{name}: argüman eksik.")
    if _is_model(args[0]):
        A, B, _, _, Ts = ss_data(args[0])
        rest = list(args[1:])
        if len(rest) not in (2, 3):
            raise ControlError(f"{name}(sistem, Q, R[, N]) biçiminde çağırın.")
        return A, B, rest[0], rest[1], (rest[2] if len(rest) == 3 else None), Ts, True
    if len(args) not in (4, 5):
        raise ControlError(f"{name}(A, B, Q, R[, N]) biçiminde çağırın.")
    return args[0], args[1], args[2], args[3], (args[4] if len(args) == 5 else None), 0.0, False


def lqr(*args):
    """Doğrusal karesel düzenleyici (MATLAB `lqr`): `lqr(sys, Q, R, N=None)` ya da `lqr(A, B, Q, R, N=None)`.

    Sürekli: u = −Kx, J = ∫(x'Qx + u'Ru + 2x'Nu)dt. Ayrık model verilirse (Ts > 0) MATLAB gibi ayrık LQR (`dlqr`).
    Dönüş `(K, S, P)`: kazanç, Riccati çözümü, kapalı çevrim kutupları eig(A − BK)."""
    A, B, Q, R, N, Ts, _ = _split_sys_args(args, "lqr")
    return _lq_core(A, B, Q, R, N, Ts > 0, "LQR")


def dlqr(A, B, Q, R, N=None):
    """Ayrık LQR (MATLAB `dlqr`): x[k+1] = Ax + Bu, J = Σ(x'Qx + u'Ru + 2x'Nu). Dönüş `(K, S, e)`."""
    return _lq_core(A, B, Q, R, N, True, "Ayrık LQR")


def _vanloan(A: np.ndarray, B: np.ndarray, W: np.ndarray, Ts: float):
    """ZOH ayrıklaştırması ve ∫₀^Ts F(t)' W F(t) dt, F(t) = expm([[A, B], [0, 0]] t) (Van Loan, 1978).

    Dönüş (Ad, Bd, Wd)."""
    n, m = B.shape
    k = n + m
    Ah = np.zeros((k, k))
    Ah[:n, :n] = A
    Ah[:n, n:] = B
    M = np.zeros((2 * k, 2 * k))
    M[:k, :k] = -Ah.T
    M[:k, k:] = W
    M[k:, k:] = Ah
    F = sla.expm(M * Ts)
    F22 = F[k:, k:]
    Wd = F22.T @ F[:k, k:]
    Wd = 0.5 * (Wd + Wd.T)
    return F22[:n, :n], F22[:n, n:], Wd


def lqrd(*args):
    """Sürekli maliyetin ayrık eşdeğeriyle ayrık LQR (MATLAB `lqrd`).

    `lqrd(A, B, Q, R, Ts)`, `lqrd(A, B, Q, R, N, Ts)` ya da `lqrd(sys, Q, R[, N], Ts)` (sys sürekli). Bitki ZOH ile
    ayrıklaştırılır; maliyet [Qd Nd; Nd' Rd] = ∫₀^Ts [Φ Γ; 0 I]'[Q N; N' R][Φ Γ; 0 I]dτ (Van Loan matris üsteli) ile
    tam hesaplanır, ardından `dlqr`. Dönüş `(Kd, S, e)`."""
    if args and _is_model(args[0]):
        A, B, _, _, Ts0 = ss_data(args[0])
        if Ts0 > 0:
            raise ControlError("lqrd sürekli zamanlı bitki ister (ayrık bitki için dlqr / lqr kullanın).")
        rest = list(args[1:])
        if len(rest) == 3:
            Q, R, Ts = rest
            N = None
        elif len(rest) == 4:
            Q, R, N, Ts = rest
        else:
            raise ControlError("lqrd(sistem, Q, R[, N], Ts) biçiminde çağırın.")
    else:
        if len(args) == 5:
            A, B, Q, R, Ts = args
            N = None
        elif len(args) == 6:
            A, B, Q, R, N, Ts = args
        else:
            raise ControlError("lqrd(A, B, Q, R[, N], Ts) biçiminde çağırın.")
    Ts = float(Ts)
    if not Ts > 0:
        raise ControlError("Örnekleme zamanı Ts > 0 olmalı.")
    A = _mat(A, "A matrisi")
    n = A.shape[0]
    B = _bmat(B, n)
    m = B.shape[1]
    Q = _sym(_mat(Q, "Q ağırlık matrisi", (n, n)), "Q ağırlık matrisi")
    R = _sym(_mat(R, "R ağırlık matrisi", (m, m)), "R ağırlık matrisi")
    N = np.zeros((n, m)) if N is None or np.size(N) == 0 else _mat(N, "N çapraz ağırlık matrisi", (n, m))
    _check_pd(R, "R ağırlık matrisi")
    _check_psd(np.block([[Q, N], [N.T, R]]), "[Q N; N' R] ağırlık matrisi")
    Ad, Bd, Wd = _vanloan(A, B, np.block([[Q, N], [N.T, R]]), Ts)
    return _lq_core(Ad, Bd, Wd[:n, :n], Wd[n:, n:], Wd[:n, n:], True, "Ayrık eşdeğer LQR")


def _lqi_aug(A, B, C, D, Ts):
    n, m = B.shape
    p = C.shape[0]
    Aa = np.zeros((n + p, n + p))
    Ba = np.zeros((n + p, m))
    Aa[:n, :n] = A
    Ba[:n] = B
    if Ts > 0:
        Aa[n:, :n] = -Ts * C
        Aa[n:, n:] = np.eye(p)
        Ba[n:] = -Ts * D
    else:
        Aa[n:, :n] = -C
        Ba[n:] = -D
    return Aa, Ba


def lqi(sys, Q, R, N=None):
    """Integral eklemeli LQR (MATLAB `lqi`): u = −K[x; xi], xi = ∫(r − y) (ayrıkta MATLAB gibi ileri Euler:
    xi[k+1] = xi[k] + Ts·(r[k] − y[k])). Q (nx+ny)×(nx+ny), R nu×nu, N (nx+ny)×nu. Dönüş `(K, S, e)`."""
    A, B, C, D, Ts = ss_data(sys, "Bitki")
    Aa, Ba = _lqi_aug(A, B, C, D, Ts)
    return _lq_core(Aa, Ba, Q, R, N, Ts > 0, "LQI")


def lqry(sys, Q, R, N=None):
    """Çıkış ağırlıklı LQR (MATLAB `lqry`): J = ∫(y'Qy + u'Ru + 2y'Nu). Eşdeğer durum ağırlıkları
    Qx = C'QC, Rx = R + D'QD + D'N + N'D, Nx = C'QD + C'N. Dönüş `(K, S, e)`."""
    A, B, C, D, Ts = ss_data(sys, "Bitki")
    p, m = D.shape
    Q = _sym(_mat(Q, "Q (çıkış) ağırlık matrisi", (p, p)), "Q (çıkış) ağırlık matrisi")
    R = _sym(_mat(R, "R ağırlık matrisi", (m, m)), "R ağırlık matrisi")
    N = np.zeros((p, m)) if N is None or np.size(N) == 0 else _mat(N, "N çapraz ağırlık matrisi", (p, m))
    Qx = C.T @ Q @ C
    Rx = R + D.T @ Q @ D + D.T @ N + N.T @ D
    Nx = C.T @ Q @ D + C.T @ N
    return _lq_core(A, B, 0.5 * (Qx + Qx.T), 0.5 * (Rx + Rx.T), Nx, Ts > 0, "LQRY")


# ---------------------------------------------------------------------------------------------------------------
# Kutup yerleştirme
# ---------------------------------------------------------------------------------------------------------------
def _poles(p: Any, n: int) -> np.ndarray:
    p = np.atleast_1d(np.asarray(p, dtype=complex)).ravel()
    if p.size != n:
        raise ControlError(f"{n} kutup verilmeli ({p.size} verildi).")
    if not np.all(np.isfinite(p)):
        raise ControlError("Kutuplar sonlu olmalı.")
    cx = p[np.abs(p.imag) > 1e-12 * np.maximum(1.0, np.abs(p))]
    for z in cx:
        if np.min(np.abs(p - np.conj(z))) > 1e-9 * max(1.0, abs(z)):
            raise ControlError("Karmaşık kutuplar eşlenik çiftler halinde verilmeli.")
    return p


def _match_err(ev: np.ndarray, p: np.ndarray) -> float:
    """Kapalı çevrim özdeğerlerinin istenen kutuplara en büyük göreli uzaklığı (eşleştirmeli)."""
    from scipy.optimize import linear_sum_assignment
    Dm = np.abs(ev[:, None] - p[None, :])
    r, c = linear_sum_assignment(Dm)
    return float(np.max(Dm[r, c] / np.maximum(1.0, np.abs(p[c])))) if p.size else 0.0


def place(A, B, p):
    """Durum geri beslemesiyle kutup yerleştirme (MATLAB `place`): eig(A − BK) = p.

    Dayanıklı (özvektör koşulunu iyileştiren) yöntem: scipy `place_poles` (Yang–Tits; MATLAB KNV ile aynı amaç).
    Tek girişte K tektir (MATLAB ile aynı); çok girişte kazanç tek değildir, kapalı çevrim özdeğerleri aynıdır.
    Bir kutbun katlılığı rank(B)'yi aşamaz; kontrol edilemeyen kip varsa ControlError."""
    if _is_model(A):
        raise ControlError("place(A, B, p) biçiminde çağırın (sistem için A ve B matrislerini verin).")
    A = _mat(A, "A matrisi")
    n = A.shape[0]
    if A.shape != (n, n):
        raise ControlError("A matrisi kare olmalı.")
    B = _bmat(B, n)
    p = _poles(p, n)
    rb = np.linalg.matrix_rank(B)
    if rb == 0:
        raise ControlError("B matrisi sıfır: kutuplar yerleştirilemez.")
    for z in p:
        mult = int(np.sum(np.abs(p - z) <= 1e-9 * max(1.0, abs(z))))
        if mult > rb:
            raise ControlError(f"Kutup {_fmt_c(z)} {mult} kez istendi; bir kutbun katlılığı giriş matrisinin rankını "
                               f"({rb}) aşamaz.")
    lam = _pbh_fail(A, B, False, only_unstable=False)
    if lam is not None:
        raise ControlError(f"Sistem kontrol edilemez (özdeğer {_fmt_c(lam)} girişlerden etkilenmiyor); kutuplar "
                           "yerleştirilemez.")
    if B.shape[1] > rb:
        raise ControlError("B matrisinin sütunları doğrusal bağımlı; bağımsız giriş sütunlarıyla çağırın.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = place_poles(A, B, p, method="YT", rtol=1e-6, maxiter=200)
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise ControlError(f"Kutuplar yerleştirilemedi: {exc}") from None
    K = np.asarray(res.gain_matrix, dtype=float)
    err = _match_err(np.linalg.eigvals(A - B @ K), p)
    if err > 0.1:
        raise ControlError(f"Kutuplar yerleştirilemedi (göreli hata {err:.2g}); sistem kontrol edilemezliğe çok yakın "
                           "ya da problem kötü koşullu.")
    if err > 1e-6:
        warnings.warn(f"Kutup yerleştirme hatası göreli {err:.2g} (kötü koşullu problem).", ControlWarning,
                      stacklevel=2)
    return K


def acker(A, b, p):
    """Ackermann formülüyle tek girişli kutup yerleştirme (MATLAB `acker`): K = [0 … 0 1]·Co⁻¹·φ(A).

    Sayısal olarak güvenilir değildir (yüksek derecede / kötü koşullu kontrol edilebilirlik matrisinde); MATLAB gibi
    yalnız tek giriş içindir. Kontrol edilemez sistemde ControlError."""
    A = _mat(A, "A matrisi")
    n = A.shape[0]
    if A.shape != (n, n):
        raise ControlError("A matrisi kare olmalı.")
    b = np.asarray(b, dtype=float)
    if b.size != n:
        raise ControlError("acker yalnız tek girişli sistem içindir (b, n×1 vektör olmalı).")
    b = b.reshape(n, 1)
    p = _poles(p, n)
    Co = np.empty((n, n))
    v = b[:, 0]
    for k in range(n):
        Co[:, k] = v
        v = A @ v
    s = np.linalg.svd(Co, compute_uv=False)
    if n and (s[-1] <= n * _EPS * s[0] or s[0] == 0):
        raise ControlError("Sistem kontrol edilemez (kontrol edilebilirlik matrisi tekil); kutuplar yerleştirilemez.")
    phi = np.real(np.poly(p))
    phiA = np.zeros((n, n))
    for c in phi:
        phiA = phiA @ A + c * np.eye(n)
    en = np.zeros(n)
    en[-1] = 1.0
    K = np.linalg.solve(Co.T, en)[None, :] @ phiA
    err = _match_err(np.linalg.eigvals(A - b @ K), p)
    if err > 0.1:
        warnings.warn(f"Kutup konumları %{100 * err:.0f}'tan fazla hatalı (kötü koşullu problem; place kullanın).",
                      ControlWarning, stacklevel=2)
    return K


# ---------------------------------------------------------------------------------------------------------------
# Kestiriciler: kalman, kalmd, estim, reg, lqg
# ---------------------------------------------------------------------------------------------------------------
class KalmanResult(tuple):
    """`(kest, L, P, Mx, Z)` demeti (MATLAB `kalman` dönüş sırası); `My` ayrıca öznitelik.

    Sürekli zamanda Mx, Z, My = None."""

    def __new__(cls, kest, L, P, Mx=None, Z=None, My=None):
        obj = super().__new__(cls, (kest, L, P, Mx, Z))
        obj._My = My
        return obj

    kest = property(lambda self: self[0])
    L = property(lambda self: self[1])
    P = property(lambda self: self[2])
    Mx = property(lambda self: self[3])
    Z = property(lambda self: self[4])
    My = property(lambda self: self._My)


def _index_list(idx, n: int, what: str) -> list[int]:
    out = [int(i) for i in np.atleast_1d(np.asarray(idx, dtype=int)).ravel()]
    for i in out:
        if not 0 <= i < n:
            raise ControlError(f"{what} indisi {i} geçersiz (0…{n - 1}).")
    if len(set(out)) != len(out):
        raise ControlError(f"{what} indisleri tekrarlanmamalı.")
    return out


def _kalman_core(A, Bu, G, C, Du, H, Qn, Rn, Nn, discrete: bool):
    """Kalman kazancı. Dönüş (L, P, Mx, Z, My, Rb)."""
    n = A.shape[0]
    nw = G.shape[1]
    ny = C.shape[0]
    Qn = _sym(_mat(Qn, "Süreç gürültüsü kovaryansı Qn", (nw, nw)), "Süreç gürültüsü kovaryansı Qn")
    Rn = _sym(_mat(Rn, "Ölçüm gürültüsü kovaryansı Rn", (ny, ny)), "Ölçüm gürültüsü kovaryansı Rn")
    Nn = np.zeros((nw, ny)) if Nn is None or np.size(Nn) == 0 else _mat(Nn, "Çapraz kovaryans Nn", (nw, ny))
    _check_psd(np.block([[Qn, Nn], [Nn.T, Rn]]), "Gürültü kovaryansı [Qn Nn; Nn' Rn]")
    Qb = G @ Qn @ G.T
    Rb = Rn + H @ Nn + Nn.T @ H.T + H @ Qn @ H.T
    Nb = G @ (Qn @ H.T + Nn)
    Qb, Rb = 0.5 * (Qb + Qb.T), 0.5 * (Rb + Rb.T)
    _check_pd(Rb, "Ölçüm gürültüsü kovaryansı (Rn + HNn + Nn'H' + HQnH')")
    lam = _pbh_fail(A.T, C.T, discrete)
    if lam is not None:
        raise ControlError(f"Kalman: sistem tespit edilemez — kararsız / sınırda kip (özdeğer {_fmt_c(lam)}) "
                           "ölçümlerde görünmüyor.")
    if n == 0:
        z = np.zeros((0, ny))
        return z, np.zeros((0, 0)), z, np.zeros((0, 0)), np.zeros((ny, ny)), Rb
    P, K, _ = _solve_are(A.T, C.T, Qb, Rb, Nb, None, discrete, "Kalman")
    L = K.T
    if not discrete:
        return L, P, None, None, None, Rb
    Sinv = np.linalg.inv(C @ P @ C.T + Rb)
    Mx = P @ C.T @ Sinv
    Z = (np.eye(n) - Mx @ C) @ P
    Z = 0.5 * (Z + Z.T)
    My = (C @ P @ C.T + H @ (Qn @ H.T + Nn)) @ Sinv
    return L, P, Mx, Z, My, Rb


def _noise_split(B, D, nw: int, known, sensors):
    nu_tot = B.shape[1]
    ny_tot = D.shape[0]
    if known is None:
        if nw > nu_tot:
            raise ControlError(f"Süreç gürültüsü kovaryansı {nw}×{nw}, ama modelin yalnız {nu_tot} girişi var "
                               "(gürültü girişleri modelin son girişleri olmalı).")
        known = list(range(nu_tot - nw))
    else:
        known = _index_list(known, nu_tot, "Bilinen giriş")
    wi = [j for j in range(nu_tot) if j not in known]
    if len(wi) != nw:
        raise ControlError(f"Gürültü girişi sayısı ({len(wi)}) Qn boyutuyla ({nw}) uyuşmuyor.")
    sensors = list(range(ny_tot)) if sensors is None else _index_list(sensors, ny_tot, "Ölçülen çıkış")
    return known, wi, sensors


def kalman(sys, Qn, Rn, Nn=None, *, sensors=None, known=None, type: str = "current") -> KalmanResult:
    """Kalman kestiricisi (MATLAB `kalman`).

    Bitki: ẋ = Ax + Bu + Gw, y = Cx + Du + Hw + v (ayrıkta x[n+1] …). `sys` girişleri [u; w]: gürültü girişleri
    son nw = boyut(Qn) giriştir (ya da `known` ile bilinen girişler verilir); `sensors` ölçülen çıkışlar.
    E{ww'} = Qn, E{vv'} = Rn, E{wv'} = Nn.

    Dönüş `(kest, L, P, Mx, Z)` (+ `.My`):
    - sürekli: x̂' = Ax̂ + Bu + L(y − Cx̂ − Du); kest girişleri [u; y], çıkışları [ŷ; x̂]; P hata kovaryansı.
    - ayrık: L = (APC' + N̄)(CPC' + R̄)⁻¹ öngörü kazancı, Mx = PC'(CPC' + R̄)⁻¹ yenileme kazancı, P = x[n|n−1]'in,
      Z = x[n|n]'in hata kovaryansı; `type='current'` (varsayılan) kest çıkışları [ŷ[n|n]; x̂[n|n]],
      `'delayed'` [ŷ[n|n−1]; x̂[n|n−1]]."""
    A, B, C, D, Ts = ss_data(sys, "Bitki")
    Qm = _mat(Qn, "Süreç gürültüsü kovaryansı Qn")
    nw = Qm.shape[0]
    known, wi, sensors = _noise_split(B, D, nw, known, sensors)
    if type not in ("current", "delayed"):
        raise ControlError("Kestirici türü 'current' ya da 'delayed' olmalı.")
    Bu, G = B[:, known], B[:, wi]
    Cs = C[sensors]
    Du, H = D[np.ix_(sensors, known)], D[np.ix_(sensors, wi)]
    discrete = Ts > 0
    L, P, Mx, Z, My, _ = _kalman_core(A, Bu, G, Cs, Du, H, Qm, Rn, Nn, discrete)
    kest = _estimator_ss(A, Bu, Cs, Du, L, Mx, My, Ts, "current" if discrete and type == "current" else "delayed")
    return KalmanResult(kest, L, P, Mx, Z, My)


def _estimator_ss(A, Bu, C, Du, L, Mx, My, Ts, kind: str):
    """Kestirici modeli: girişler [u; y], çıkışlar [ŷ; x̂]."""
    n, m = Bu.shape
    ny = C.shape[0]
    Ae = A - L @ C
    Be = np.hstack([Bu - L @ Du, L])
    if kind == "current":
        Ce = np.vstack([C - My @ C, np.eye(n) - Mx @ C])
        De = np.block([[Du - My @ Du, My], [-Mx @ Du, Mx]])
    else:
        Ce = np.vstack([C, np.eye(n)])
        De = np.block([[Du, np.zeros((ny, ny))], [np.zeros((n, m)), np.zeros((n, ny))]])
    return make_ss(Ae, Be, Ce, De, Ts)


def kalmd(sys, Qn, Rn, Ts):
    """Sürekli bitki için ayrık Kalman kestiricisi (MATLAB `kalmd`).

    Bitki ẋ = Ax + Bu + Gw, y = Cx + Du + v (sys girişleri [u; w]). ZOH ayrıklaştırması; süreç gürültüsü
    Qd = ∫₀^Ts e^{Aτ}GQnG'e^{A'τ}dτ (Van Loan), ölçüm gürültüsü Rd = Rn/Ts. Dönüş `(kest, L, P, M, Z)`."""
    A, B, C, D, Ts0 = ss_data(sys, "Bitki")
    if Ts0 > 0:
        raise ControlError("kalmd sürekli zamanlı bitki ister (ayrık bitki için kalman kullanın).")
    Ts = float(Ts)
    if not Ts > 0:
        raise ControlError("Örnekleme zamanı Ts > 0 olmalı.")
    Qm = _mat(Qn, "Süreç gürültüsü kovaryansı Qn")
    nw = Qm.shape[0]
    known, wi, _ = _noise_split(B, D, nw, None, None)
    n = A.shape[0]
    Bu, G = B[:, known], B[:, wi]
    Qm = _sym(_mat(Qm, "Süreç gürültüsü kovaryansı Qn", (nw, nw)), "Süreç gürültüsü kovaryansı Qn")
    _check_psd(Qm, "Süreç gürültüsü kovaryansı Qn")
    Ad, Bd, _ = _vanloan(A, Bu, np.zeros((n + Bu.shape[1],) * 2), Ts)
    # Qd = ∫ e^{Aτ} G Qn G' e^{A'τ} dτ : Van Loan [[−A, GQG'], [0, A']]
    M = np.zeros((2 * n, 2 * n))
    M[:n, :n] = -A
    M[:n, n:] = G @ Qm @ G.T
    M[n:, n:] = A.T
    F = sla.expm(M * Ts)
    Qd = F[n:, n:].T @ F[:n, n:]
    Qd = 0.5 * (Qd + Qd.T)
    ny = C.shape[0]
    Rd = _mat(Rn, "Ölçüm gürültüsü kovaryansı Rn", (ny, ny)) / Ts
    Du = D[:, known]
    L, P, Mx, Z, My, _ = _kalman_core(Ad, Bd, np.eye(n), C, Du, np.zeros((ny, n)), Qd, Rd, None, True)
    kest = _estimator_ss(Ad, Bd, C, Du, L, Mx, My, Ts, "current")
    return KalmanResult(kest, L, P, Mx, Z, My)


def estim(sys, L, sensors=None, known=None):
    """Durum / çıkış kestiricisi (MATLAB `estim(sys, L, sensors, known)`): verilen kestirici kazancı L ile.

    `known` verilmezse tüm girişler stokastik (gürültü) kabul edilir ve kestiricinin girişi yalnız ölçümlerdir;
    verilirse girişler [u_bilinen; y]. Çıkışlar [ŷ; x̂]. Sürekli: x̂' = Ax̂ + Bu + L(y − Cx̂ − Du); ayrık (MATLAB
    gibi yalnız öngörü biçimi): x̂[n+1|n] = Ax̂ + Bu + L(y − Cx̂ − Du), çıkışlar ŷ[n|n−1], x̂[n|n−1].
    İndisler 0 tabanlıdır."""
    A, B, C, D, Ts = ss_data(sys, "Bitki")
    n = A.shape[0]
    sensors = list(range(C.shape[0])) if sensors is None else _index_list(sensors, C.shape[0], "Ölçülen çıkış")
    known = [] if known is None else _index_list(known, B.shape[1], "Bilinen giriş")
    Cs = C[sensors]
    ny = Cs.shape[0]
    L = _mat(L, "Kestirici kazancı L", (n, ny))
    return _estimator_ss(A, B[:, known], Cs, D[np.ix_(sensors, known)], L, None, None, Ts, "delayed")


def reg(sys, K, L):
    """Gözlemci tabanlı düzenleyici (MATLAB `reg`): girişi y, çıkışı u = −Kx̂.

    x̂' = [A − LC − (B − LD)K]x̂ + Ly (ayrıkta öngörü kestiricisi x̂[n+1|n]). Kapalı çevrim pozitif geri beslemeyle
    kurulur (MATLAB: feedback(sys, rsys, +1))."""
    A, B, C, D, Ts = ss_data(sys, "Bitki")
    n, m = B.shape
    ny = C.shape[0]
    K = _mat(K, "Durum geri besleme kazancı K", (m, n))
    L = _mat(L, "Kestirici kazancı L", (n, ny))
    Ar = A - L @ C - (B - L @ D) @ K
    return make_ss(Ar, L, -K, np.zeros((m, ny)), Ts)


def lqg(sys, QXU, QWV, QI=None, dof: str = "2dof", estimator: str = "delayed"):
    """LQG düzenleyici / izleyici (MATLAB `lqg`).

    Bitki ẋ = Ax + Bu + w, y = Cx + Du + v; maliyet J = ∫[x; u]'·QXU·[x; u]dt, gürültü kovaryansı
    E{[w; v][w; v]'} = QWV. QI verilmezse düzenleyici (girişi y, çıkışı u; kapalı çevrim pozitif geri besleme:
    feedback(sys, K, +1)). QI verilirse integral etkili izleyici (u = −Kx·x̂ − Ki·xi, xi = ∫(r − y); ayrıkta
    xi[k+1] = xi[k] + Ts(r − y)): '2dof' girişler [r; y], '1dof' girişi e = r − y. Durumlar [x̂; xi].
    Ayrık zamanda MATLAB gibi varsayılan kestirim x̂[n|n−1] (`estimator='delayed'`); 'current' x̂[n|n]."""
    A, B, C, D, Ts = ss_data(sys, "Bitki")
    n, m = B.shape
    ny = C.shape[0]
    discrete = Ts > 0
    if dof not in ("1dof", "2dof"):
        raise ControlError("Serbestlik derecesi '1dof' ya da '2dof' olmalı.")
    if estimator not in ("delayed", "current"):
        raise ControlError("Kestirici türü 'delayed' ya da 'current' olmalı.")
    QXU = _sym(_mat(QXU, "QXU ağırlık matrisi", (n + m, n + m)), "QXU ağırlık matrisi")
    QWV = _sym(_mat(QWV, "QWV kovaryans matrisi", (n + ny, n + ny)), "QWV kovaryans matrisi")
    Qx, Nxu, Ru = QXU[:n, :n], QXU[:n, n:], QXU[n:, n:]
    Qn, Nn, Rn = QWV[:n, :n], QWV[:n, n:], QWV[n:, n:]
    L, _, Mx, _, _, _ = _kalman_core(A, B, np.eye(n), C, D, np.zeros((ny, n)), Qn, Rn, Nn, discrete)
    if not (discrete and estimator == "current"):
        Mx = None
    if QI is None:
        K, _, _ = _lq_core(A, B, Qx, Ru, Nxu, discrete, "LQG (LQR kısmı)")
        Kx, Ki = K, np.zeros((m, 0))
        Aa = None
    else:
        QI = _sym(_mat(QI, "QI ağırlık matrisi", (ny, ny)), "QI ağırlık matrisi")
        Qa = np.zeros((n + ny, n + ny))
        Qa[:n, :n] = Qx
        Qa[n:, n:] = QI
        Na = np.vstack([Nxu, np.zeros((ny, m))])
        Aa, Ba = _lqi_aug(A, B, C, D, Ts)
        Ka, _, _ = _lq_core(Aa, Ba, Qa, Ru, Na, discrete, "LQG (LQI kısmı)")
        Kx, Ki = Ka[:, :n], Ka[:, n:]
    # u = Cx·x̂ + Ci·xi + Dy·y  (güncel kestirimde x̂[n|n] = x̂ + Mx(y − Cx̂ − Du), cebirsel döngü çözülür)
    if Mx is not None:
        try:
            W = np.linalg.inv(np.eye(m) + Kx @ Mx @ (-D))
        except np.linalg.LinAlgError:
            raise ControlError("Güncel kestirimli LQG'de cebirsel döngü çözülemiyor (I − K·Mx·D tekil).") from None
        Cx = -W @ Kx @ (np.eye(n) - Mx @ C)
        Ci = -W @ Ki
        Dy = -W @ Kx @ Mx
    else:
        Cx, Ci, Dy = -Kx, -Ki, np.zeros((m, ny))
    BL = B - L @ D
    Axx = A - L @ C + BL @ Cx
    Bxy = L + BL @ Dy
    if QI is None:
        return make_ss(Axx, Bxy, Cx, Dy, Ts)
    q = Ts if discrete else 1.0
    # xi satırı: ayrık xi[k+1] = xi + Ts(r − y); sürekli xi' = r − y (y ölçülen, x̂'den bağımsız)
    Ar = np.block([[Axx, BL @ Ci], [np.zeros((ny, n)), np.eye(ny) if discrete else np.zeros((ny, ny))]])
    Br = np.vstack([np.zeros((n, ny)), q * np.eye(ny)])
    By = np.vstack([Bxy, -q * np.eye(ny)])
    Cr = np.hstack([Cx, Ci])
    if dof == "2dof":
        return make_ss(Ar, np.hstack([Br, By]), Cr, np.hstack([np.zeros((m, ny)), Dy]), Ts)
    return make_ss(Ar, -By, Cr, -Dy, Ts)     # 1dof: r = 0, y = −e


# ---------------------------------------------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------------------------------------------
def nbar(sys, K):
    """Kapalı çevrim kalıcı durum kazanç düzeltmesi: u = −Kx + N̄r ile y → r (kalıcı durumda).

    Sürekli: N̄ = [−(C − DK)(A − BK)⁻¹B + D]⁻¹; ayrık: N̄ = [(C − DK)(I − A + BK)⁻¹B + D]⁻¹. Kare olmayan
    durumda en küçük kareler (sözde ters) — uyarıyla. Kapalı çevrim DC kazancı tekilse ControlError."""
    A, B, C, D, Ts = ss_data(sys, "Bitki")
    n, m = B.shape
    K = _mat(K, "Durum geri besleme kazancı K", (m, n))
    Acl = A - B @ K
    Ccl = C - D @ K
    try:
        if Ts > 0:
            G0 = Ccl @ np.linalg.solve(np.eye(n) - Acl, B) + D
        else:
            G0 = -Ccl @ np.linalg.solve(Acl, B) + D
    except np.linalg.LinAlgError:
        raise ControlError("Kapalı çevrimin DC kazancı tanımsız (kapalı çevrimde sıfırda / z = 1'de kutup "
                           "var).") from None
    if G0.shape[0] == G0.shape[1]:
        if np.linalg.cond(G0) > 1e12:
            raise ControlError("Kapalı çevrim DC kazanç matrisi tekil (sistemin sıfırda iletim sıfırı var); referans "
                               "kazancı hesaplanamaz.")
        return np.linalg.inv(G0)
    warnings.warn("Giriş ve çıkış sayıları farklı; N̄ en küçük kareler anlamında (sözde ters).", ControlWarning,
                  stacklevel=2)
    return np.linalg.pinv(G0)


def bryson(xmax, umax):
    """Bryson kuralı: Q = diag(1/xmax²), R = diag(1/umax²) (her durum / giriş kabul edilebilir en büyük değerine
    göre ölçeklenir). Dönüş (Q, R)."""
    x = np.atleast_1d(np.asarray(xmax, dtype=float)).ravel()
    u = np.atleast_1d(np.asarray(umax, dtype=float)).ravel()
    if np.any(~np.isfinite(u)) or np.any(u <= 0) or np.any(x <= 0):
        raise ControlError("Bryson kuralı için en büyük değerler pozitif olmalı (durum için sonsuz = ağırlık yok).")
    with np.errstate(divide="ignore"):
        q = np.where(np.isinf(x), 0.0, 1.0 / x ** 2)
    return np.diag(q), np.diag(1.0 / u ** 2)

