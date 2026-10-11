"""Blok diyagram için küçük, bağımsız LTI yardımcıları (LTI çekirdeği `spt.control.lti`'ye bağımlı değildir).

- `tf2ss`: denetlenebilir kanonik gerçekleme (Simulink Transfer Fcn bloğuyla aynı biçim: x1 en yüksek türev).
- `zpk2tf`: sıfır / kutup / kazançtan gerçel katsayılı polinomlar (eşlenik çiftler denetlenir).
- `c2d`: 'zoh' (genişletilmiş matrisin üsteli; MATLAB c2d ile aynı tanım) ve 'tustin' (isteğe bağlı ön bükme).
- `d2c_zoh`, `resample_zoh`: ayrık modelin hız dönüşümü (MATLAB d2c / d2d 'zoh'): tam sayı katlarında matris kuvveti
  (kesin), aksi hâlde matris logaritması.
- `pade`: e^(−sT) için Padé yaklaşımı (MATLAB pade ile aynı rasyonel fonksiyon).
- `lti_data`: ördek tiplemeyle LTI nesnesinden (A, B, C, D, Ts) ya da (num, den, Ts) alır.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import scipy.linalg as sla

from spt.control.errors import ControlError


def _trim_leading(p: np.ndarray) -> np.ndarray:
    p = np.atleast_1d(np.asarray(p, dtype=np.float64))
    nz = np.flatnonzero(p != 0)
    return p[nz[0]:] if nz.size else np.zeros(1)


def tf2ss(num: Any, den: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Transfer fonksiyonu → denetlenebilir kanonik durum uzayı.

    num: 1-B (SISO) ya da 2-B (her satır bir çıkış; tek giriş). den: 1-B. deg(num) ≤ deg(den) olmalı.
    A = [[-a1 … -an]; [I 0]], B = e1, C = b − b0·a (a, b baş katsayısı 1'e normalize), D = b0."""
    den = _trim_leading(den)
    if den.size == 0 or den[0] == 0:
        raise ControlError("Payda polinomu sıfır olamaz.")
    numm = np.atleast_2d(np.asarray(num, dtype=np.float64))
    if numm.size == 0:
        numm = np.zeros((1, 1))
    n = den.size - 1
    rows = []
    for r in numm:
        r = _trim_leading(r)
        if r.size - 1 > n:
            raise ControlError(f"Transfer fonksiyonu uygun değil: pay derecesi ({r.size - 1}) payda derecesinden "
                               f"({n}) büyük.")
        rows.append(np.concatenate([np.zeros(n + 1 - r.size), r]))
    a = den / den[0]
    b = np.array(rows) / den[0]
    D = b[:, :1].copy()
    if n == 0:
        return np.zeros((0, 0)), np.zeros((0, 1)), np.zeros((b.shape[0], 0)), D
    A = np.zeros((n, n))
    A[0, :] = -a[1:]
    if n > 1:
        A[1:, :-1] = np.eye(n - 1)
    B = np.zeros((n, 1))
    B[0, 0] = 1.0
    C = b[:, 1:] - b[:, :1] * a[1:]
    return A, B, C, D


def zpk2tf(z: Any, p: Any, k: Any) -> tuple[np.ndarray, np.ndarray]:
    """Sıfır / kutup / kazanç → gerçel katsayılı (num, den). Karmaşık kökler eşlenik çiftler hâlinde olmalı."""
    z = np.atleast_1d(np.asarray(z, dtype=np.complex128)).ravel()
    p = np.atleast_1d(np.asarray(p, dtype=np.complex128)).ravel()
    for arr, what in ((z, "Sıfırlar"), (p, "Kutuplar")):
        if arr.size and not _conj_closed(arr):
            raise ControlError(f"{what} karmaşık eşlenik çiftler hâlinde verilmeli (gerçel katsayılı sistem).")
    num = np.real(np.poly(z)) * float(k) if z.size else np.array([float(k)])
    den = np.real(np.poly(p)) if p.size else np.array([1.0])
    return np.atleast_1d(num).astype(np.float64), np.atleast_1d(den).astype(np.float64)


def _conj_closed(r: np.ndarray, tol: float = 1e-9) -> bool:
    cplx = r[np.abs(r.imag) > tol * np.maximum(1.0, np.abs(r))]
    if cplx.size == 0:
        return True
    rem = list(cplx)
    while rem:
        c = rem.pop()
        j = int(np.argmin([abs(c.conjugate() - q) for q in rem])) if rem else -1
        if j < 0 or abs(c.conjugate() - rem[j]) > 1e-8 * max(1.0, abs(c)):
            return False
        rem.pop(j)
    return True


def c2d(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, Ts: float, method: str = "zoh",
        prewarp: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sürekli durum uzayını ayrıklaştırır. method: 'zoh' | 'tustin' (prewarp > 0 ise ön bükmeli)."""
    A = np.atleast_2d(np.asarray(A, dtype=np.float64))
    B = np.asarray(B, dtype=np.float64).reshape(A.shape[0], -1) if A.size else np.zeros((0, np.shape(D)[1]))
    n, m = B.shape
    if method == "zoh":
        if n == 0:
            return A.copy(), B.copy(), np.asarray(C, dtype=np.float64).copy(), np.asarray(D, dtype=np.float64).copy()
        M = np.zeros((n + m, n + m))
        M[:n, :n] = A * Ts
        M[:n, n:] = B * Ts
        E = sla.expm(M)
        return E[:n, :n], E[:n, n:], np.asarray(C, dtype=np.float64).copy(), np.asarray(D, dtype=np.float64).copy()
    if method == "tustin":
        if prewarp and prewarp > 0:
            if prewarp * Ts / 2 >= math.pi / 2:
                raise ControlError("Ön bükme frekansı Nyquist frekansından küçük olmalı.")
            alpha = math.tan(prewarp * Ts / 2) / prewarp
        else:
            alpha = Ts / 2
        C = np.asarray(C, dtype=np.float64)
        D = np.asarray(D, dtype=np.float64)
        if n == 0:
            return A.copy(), B.copy(), C.copy(), D.copy()
        I = np.eye(n)
        M = I - alpha * A
        try:
            Mi = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            raise ControlError("Tustin dönüşümü tanımsız: sistemin 2/Ts'de kutbu var.") from None
        Ad = Mi @ (I + alpha * A)
        Bd = Mi @ B
        Cd = 2 * alpha * C @ Mi
        Dd = D + alpha * C @ Mi @ B
        return Ad, Bd, Cd, Dd
    raise ControlError(f"Bilinmeyen ayrıklaştırma yöntemi '{method}'.")


def d2c_zoh(Ad: np.ndarray, Bd: np.ndarray, Ts: float) -> tuple[np.ndarray, np.ndarray]:
    """ZOH ayrık modelden sürekli (A, B) (MATLAB d2c 'zoh'): logm([[Ad Bd]; [0 I]]) / Ts."""
    n, m = Bd.shape
    if n == 0:
        return Ad.copy(), Bd.copy()
    M = np.zeros((n + m, n + m))
    M[:n, :n] = Ad
    M[:n, n:] = Bd
    M[n:, n:] = np.eye(m)
    ev = np.linalg.eigvals(Ad)
    if np.any(np.abs(ev) < 1e-12) or np.any((np.abs(ev.imag) < 1e-12) & (ev.real < 0)):
        raise ControlError("Ayrık blok sürekli zamana çevrilemiyor: z = 0 ya da negatif gerçel eksende kutup var "
                           "(ZOH tersinin logaritması tanımsız).")
    L = sla.logm(M)
    if np.iscomplexobj(L):
        if np.max(np.abs(L.imag)) > 1e-8 * max(1.0, np.max(np.abs(L.real))):
            raise ControlError("Ayrık blok sürekli zamana çevrilemiyor (karmaşık matris logaritması).")
        L = L.real
    L = L / Ts
    return L[:n, :n], L[:n, n:]


def resample_zoh(Ad: np.ndarray, Bd: np.ndarray, Ts_old: float, Ts_new: float) -> tuple[np.ndarray, np.ndarray]:
    """Ayrık modelin örnekleme zamanını ZOH varsayımıyla değiştirir (MATLAB d2d 'zoh'). Ts_new / Ts_old tam sayı
    ise [[Ad Bd]; [0 I]]^m (kesin), değilse d2c + c2d."""
    n, m = Bd.shape
    ratio = Ts_new / Ts_old
    r = round(ratio)
    if r >= 1 and abs(ratio - r) <= 1e-9 * ratio:
        if n == 0:
            return Ad.copy(), Bd.copy()
        M = np.zeros((n + m, n + m))
        M[:n, :n] = Ad
        M[:n, n:] = Bd
        M[n:, n:] = np.eye(m)
        P = np.linalg.matrix_power(M, int(r))
        return P[:n, :n], P[:n, n:]
    A, B = d2c_zoh(Ad, Bd, Ts_old)
    Ad2, Bd2, _, _ = c2d(A, B, np.zeros((0, n)), np.zeros((0, m)), Ts_new, "zoh")
    return Ad2, Bd2


def pade(T: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """e^(−sT) için n. dereceden Padé yaklaşımı (num, den; azalan kuvvetler, payda baş katsayısı 1)."""
    if n < 0:
        raise ControlError("Padé derecesi negatif olamaz.")
    if n == 0 or T == 0:
        return np.array([1.0]), np.array([1.0])
    c = [math.factorial(2 * n - k) * math.factorial(n) / (math.factorial(2 * n) * math.factorial(k)
                                                            * math.factorial(n - k)) for k in range(n + 1)]
    num = np.array([c[k] * (-T) ** k for k in range(n, -1, -1)])
    den = np.array([c[k] * T ** k for k in range(n, -1, -1)])
    return num / den[0], den / den[0]


def lti_data(sys: Any) -> dict[str, Any]:
    """Ördek tiplemeyle LTI nesnesinden veri: {'A','B','C','D','Ts'} (durum uzayı) ya da {'num','den','Ts'} (SISO
    transfer fonksiyonu; num/den iç içe listelerse [0][0] alınır). `spt.control.lti` sınıfları, scipy.signal
    `StateSpace` / `TransferFunction` ve benzerleri desteklenir."""
    Ts = getattr(sys, "Ts", None)
    if Ts is None:
        Ts = getattr(sys, "dt", None)
    Ts = 0.0 if Ts in (None, False) else float(Ts)
    if Ts < 0:
        raise ControlError("Belirtilmemiş örnekleme zamanlı (Ts = −1) model desteklenmez.")
    if all(hasattr(sys, a) for a in ("A", "B", "C", "D")):
        A = np.atleast_2d(np.asarray(sys.A, dtype=np.float64))
        B = np.asarray(sys.B, dtype=np.float64)
        C = np.asarray(sys.C, dtype=np.float64)
        D = np.atleast_2d(np.asarray(sys.D, dtype=np.float64))
        n = A.shape[0] if A.size else 0
        B = B.reshape(n, -1) if n else np.zeros((0, D.shape[1]))
        C = C.reshape(-1, n) if n else np.zeros((D.shape[0], 0))
        return {"A": A if n else np.zeros((0, 0)), "B": B, "C": C, "D": D, "Ts": Ts}
    if hasattr(sys, "num") and hasattr(sys, "den"):
        num, den = sys.num, sys.den
        while isinstance(num, (list, tuple)) or (isinstance(num, np.ndarray) and num.dtype == object):
            if len(num) != 1:
                raise ControlError("Çok girişli / çok çıkışlı transfer fonksiyonu için Durum Uzayı bloğunu kullanın.")
            num = num[0]
        while isinstance(den, (list, tuple)) or (isinstance(den, np.ndarray) and den.dtype == object):
            den = den[0]
        num = np.atleast_1d(np.asarray(num, dtype=np.float64))
        den = np.atleast_1d(np.asarray(den, dtype=np.float64))
        if num.ndim > 1:
            if num.shape[0] != 1:
                raise ControlError("Çok çıkışlı transfer fonksiyonu için Durum Uzayı bloğunu kullanın.")
            num = num[0]
        return {"num": num, "den": den.ravel(), "Ts": Ts}
    raise ControlError("LTI model tanınmadı: A, B, C, D ya da num, den öznitelikleri bekleniyordu.")


def freqresp(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, s: complex) -> np.ndarray:
    """H(s) = C (sI − A)^−1 B + D (doğrulama için)."""
    n = A.shape[0]
    if n == 0:
        return np.asarray(D, dtype=np.complex128)
    return C @ np.linalg.solve(s * np.eye(n) - A, B) + D
