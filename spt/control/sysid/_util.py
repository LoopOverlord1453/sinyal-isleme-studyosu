"""Sistem tanımlamanın iç yardımcıları: q⁻¹ polinomları, Levenberg–Marquardt çözücü, sayısal türev.

Polinomlar MATLAB System Identification Toolbox'taki gibi **q⁻¹'in artan kuvvetleri**dir:
`[1, a1, a2]` = 1 + a1·q⁻¹ + a2·q⁻² (z cinsinden z² + a1·z + a2 / z²; `np.roots` doğrudan z-kutuplarını verir).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import linalg, signal

from spt.control.errors import ControlError


# ----------------------------------------------------------------------------------------------- polinomlar
def pmul(*polys) -> np.ndarray:
    out = np.array([1.0])
    for p in polys:
        out = np.convolve(out, np.asarray(p, dtype=float))
    return out


def padd(a, b) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(len(a), len(b))
    return np.pad(a, (0, n - len(a))) + np.pad(b, (0, n - len(b)))


def trim(p, keep: int = 1) -> np.ndarray:
    """Sondaki (yüksek gecikmeli) sıfırları atar; en az `keep` katsayı kalır."""
    p = np.asarray(p, float)
    nz = np.flatnonzero(p)
    n = max(keep, (nz[-1] + 1) if len(nz) else 0)
    return p[:n].copy()


def is_stable(p, threshold: float = 1.0) -> bool:
    """q⁻¹ polinomunun tüm kökleri |z| < threshold mi (1/p filtresi kararlı mı)."""
    p = trim(p)
    if len(p) <= 1:
        return True
    if not np.all(np.isfinite(p)):
        return False
    return bool(np.max(np.abs(np.roots(p))) < threshold)


def stabilize(p) -> np.ndarray:
    """Birim çemberin dışındaki kökleri içeri yansıtır (z → 1/z̄); genlik spektrumu ölçek dışında korunur."""
    p = np.asarray(p, float)
    if len(trim(p)) <= 1:
        return p.copy()
    r = np.roots(trim(p))
    out = np.where(np.abs(r) > 1.0, 1.0 / np.conj(r), r)
    out = np.where(np.abs(out) > 0.999, out / np.abs(out) * 0.999, out) if np.any(np.abs(out) >= 0.9999) else out
    q = np.real(np.poly(out)) * p[0]
    return np.pad(q, (0, len(p) - len(q)))


def filt(b, a, x, zi=None):
    """`scipy.signal.lfilter` (sıfır başlangıç durumlu, eksen 0)."""
    b = np.atleast_1d(np.asarray(b, float))
    a = np.atleast_1d(np.asarray(a, float))
    if zi is None:
        return signal.lfilter(b, a, x, axis=0)
    return signal.lfilter(b, a, x, axis=0, zi=zi)[0]


def shift(x, k: int) -> np.ndarray:
    """x(t − k) (k ≥ 0; başlangıç öncesi sıfır). x: (N,) ya da (N, m)."""
    x = np.asarray(x, float)
    if k == 0:
        return x.copy()
    out = np.zeros_like(x)
    if k < len(x):
        out[k:] = x[:len(x) - k]
    return out


# ----------------------------------------------------------------------------------------------- LM çözücü
@dataclass
class LMResult:
    x: np.ndarray
    cost: float
    iterations: int
    why: str
    jac: np.ndarray | None = None
    history: list = field(default_factory=list)


def fd_jacobian(fun, x, r0=None, *, central: bool = True, rel: float | None = None) -> np.ndarray:
    """Sonlu fark Jacobian'ı (merkezi fark varsayılan; adım ≈ eps^(1/3)·max(|x|, 1))."""
    x = np.asarray(x, float)
    if r0 is None:
        r0 = fun(x)
    m = len(r0)
    J = np.empty((m, len(x)))
    if rel is None:
        rel = np.finfo(float).eps ** (1 / 3) if central else np.sqrt(np.finfo(float).eps)
    for k in range(len(x)):
        h = rel * max(abs(x[k]), 1.0)
        xp = x.copy()
        xp[k] += h
        if central:
            xm = x.copy()
            xm[k] -= h
            J[:, k] = (fun(xp) - fun(xm)) / (2 * h)
        else:
            J[:, k] = (fun(xp) - r0) / h
    return J


def _factor(Js, r):
    """Ölçekli Jacobian'ın (s, Vᵀ, Uᵀr) çarpanları; adım δ = −V (s/(s²+λ) · Uᵀr)."""
    if Js.shape[1] > 60:
        # çok parametreli (ağ eğitimi): normal denklemlerin özayrışımı (s² = özdeğerler), her λ için ucuz
        ev, V = np.linalg.eigh(Js.T @ Js)
        s = np.sqrt(np.maximum(ev, 0.0))
        Vt = V.T
        with np.errstate(divide="ignore", invalid="ignore"):
            Ur = np.where(s > 0, (Vt @ (Js.T @ r)) / np.where(s > 0, s, 1.0), 0.0)
        return s, Vt, Ur
    if Js.shape[0] > Js.shape[1]:
        # J = Q R, R = U_r S Vᵀ → Uᵀr = U_rᵀ(Qᵀr): büyük SVD yerine QR + küçük SVD (aynı adım, daha hızlı)
        Q, Rr = np.linalg.qr(Js)
        Ur_, s, Vt = linalg.svd(Rr, check_finite=False)
        return s, Vt, Ur_.T @ (Q.T @ r)
    U, s, Vt = linalg.svd(Js, full_matrices=False, check_finite=False)
    return s, Vt, U.T @ r


def levenberg_marquardt(fun, x0, jac=None, *, accept=None, max_iter: int = 200, ftol: float = 1e-13,
                        xtol: float = 1e-11, lam0: float = 1e-4, central: bool = True, project=None) -> LMResult:
    """min ‖r(x)‖² için Marquardt ölçeklemeli Levenberg–Marquardt (eksik ranklı Jacobian'a dayanıklı).

    `accept(x)` False dönerse (ör. kararsız filtre) adım reddedilir ve sönüm artırılır. `project(x)`: sınır
    kısıtlarına izdüşüm; adım sınıra takılan bileşenler o adımda sabitlenerek (etkin küme) yeniden çözülür.
    Durma: göreli maliyet azalması < ftol ve göreli adım < √xtol, ya da göreli adım < xtol, ya da iyileşme yok."""
    x = np.asarray(x0, float).copy()
    if project is not None:
        x = project(x)
    r = fun(x)
    if not np.all(np.isfinite(r)):
        raise ControlError("Başlangıç tahmininde artıklar sonlu değil (kararsız başlangıç modeli)")
    cost = float(r @ r)
    lam = lam0
    why = "en çok yineleme"
    it = 0
    J = None
    hist = [cost]
    n = len(x)
    for it in range(1, max_iter + 1):
        J = jac(x) if jac is not None else fd_jacobian(fun, x, r, central=central)
        d = np.sqrt(np.sum(J * J, axis=0))
        dmax = d.max() if len(d) and d.max() > 0 else 1.0
        # ölçütü (neredeyse) etkilemeyen parametreler (ör. sıfır girişte B) adımdan çıkarılır: küçük normla
        # ölçekleme sayısal gürültüyü büyütüp anlamsız büyük değerler üretirdi
        live = d > 1e-10 * dmax
        if not live.any():
            why = "ölçüt parametrelere duyarsız"
            break
        d = np.where(live, d, 1.0)
        Js = J / d
        fac = {}

        def solve(mask, lam):
            key = tuple(np.flatnonzero(~mask))
            if key not in fac:
                fac[key] = _factor(Js[:, mask], r)
            s_, Vt_, Ur_ = fac[key]
            st = np.zeros(n)
            st[mask] = -(Vt_.T @ (s_ / (s_ * s_ + lam) * Ur_)) / d[mask]
            return st

        improved = False
        while lam < 1e12:
            step = solve(live, lam)
            cand = x + step
            if project is not None:
                pc = project(cand)
                clipped = np.abs(pc - cand) > 0
                if clipped.any() and (live & ~clipped).any():
                    step = solve(live & ~clipped, lam)
                    pc = project(x + step)
                cand = pc
                step = cand - x
            if accept is not None and not accept(cand):
                lam *= 10.0
                continue
            rc = fun(cand)
            if not np.all(np.isfinite(rc)):
                lam *= 10.0
                continue
            with np.errstate(over="ignore", invalid="ignore"):
                cc = float(rc @ rc)
            if np.isfinite(cc) and cc <= cost:
                rel_dec = (cost - cc) / max(cost, 1e-300)
                rel_step = np.linalg.norm(step * d) / max(np.linalg.norm(x * d), 1e-300)
                x, r, cost = cand, rc, cc
                lam = max(lam / 10.0, 1e-12)
                improved = True
                hist.append(cost)
                if rel_dec < ftol and rel_step < np.sqrt(xtol):
                    why = "yakınsadı"
                    return LMResult(x, cost, it, why, J, hist)
                if rel_step < xtol:
                    why = "adım çok küçük"
                    return LMResult(x, cost, it, why, J, hist)
                break
            lam *= 10.0
        if not improved:
            why = "iyileşme yok (yerel en küçük)"
            break
    return LMResult(x, cost, it, why, J, hist)
