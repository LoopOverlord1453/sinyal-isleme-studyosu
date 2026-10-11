"""Model analizi: pole, zero, tzero, damp, pzmap, rlocus (+ kırılma noktaları, asimptotlar), ctrb, obsv, gram,
covar, isstable, dsort, esort (MATLAB Control System Toolbox karşılığı).

MATLAB'den ölçülen tanımlar:
- MIMO tf / zpk'nin kutupları ve sıfırları MATLAB'deki gibi modelin durum uzayı gerçeklemesinden (aynı derece) alınır;
  ör. [(s+1)/(s+1)²; 1/(s+1)] için 3 kutup ve gerçeklemenin ayrıştırma sıfırı −1 (MATLAB sonucu).
- damp: ω_n'ye göre artan sıra; sürekli ζ = −Re(p)/|p| (p = 0 → ω_n = 0, ζ = −1); ayrık s = ln(z)/Ts (z = 0 →
  ω_n = ∞, ζ = 1). Kutuplar özgün (z) değerleriyle döner.
- rlocus: 1 + k·G(s) = 0 (negatif geri besleme); kollar süreklilikle (Macar eşleme) sıralı; r şekli (n, nk).
"""
from __future__ import annotations

import math

import numpy as np
import scipy.linalg as sla
from scipy.optimize import linear_sum_assignment

from spt.control.errors import ControlError
from spt.control.lti import LTI, StateSpace, TransferFunction, ZPK, cplxpair, poly_from_roots, roots
from spt.control import convert


# ---------------------------------------------------------------------------------------------------------------
# Kutuplar ve sıfırlar
# ---------------------------------------------------------------------------------------------------------------
def pole(sys: LTI) -> np.ndarray:
    """Kutuplar (MATLAB `pole`): ss → özdeğerler; SISO tf → payda kökleri; SISO zpk → p; MIMO tf / zpk → durum
    uzayı gerçeklemesinin özdeğerleri (MATLAB gibi)."""
    if isinstance(sys, StateSpace):
        return cplxpair(np.linalg.eigvals(sys.A), tol=1e-9) if sys.nx else np.zeros(0, complex)
    if sys.is_siso:
        if isinstance(sys, TransferFunction):
            return roots(sys._den[0][0]) if np.any(sys._num[0][0]) else np.zeros(0, complex)
        return sys._p[0][0].copy()
    s = convert.to_ss(_nodelay(sys))
    return cplxpair(np.linalg.eigvals(s.A), tol=1e-9) if s.nx else np.zeros(0, complex)


def _nodelay(sys: LTI) -> LTI:
    out = sys.copy()
    out.input_delay = np.zeros(sys.ninputs)
    out.output_delay = np.zeros(sys.noutputs)
    if hasattr(out, "io_delay"):
        out.io_delay = np.zeros(sys.shape)
    return out


def tzero(sys: LTI) -> np.ndarray:
    """Değişmez (iletim + ayrıştırma) sıfırlar (MATLAB `tzero`): Rosenbrock kalemi [[A − λI, B], [C, D]]'nin sonlu
    genelleştirilmiş özdeğerleri (Emami-Naeini & Van Dooren indirgemesi). tf / zpk girişi MATLAB gibi durum uzayı
    gerçeklemesi üzerinden."""
    s = convert.to_ss(_nodelay(sys)) if not isinstance(sys, StateSpace) else sys
    z, _ = convert.invariant_zeros(s.A, s.B, s.C, s.D)
    return z


def zero(sys: LTI) -> np.ndarray:
    """Sıfırlar (MATLAB `zero`): SISO tf → pay kökleri; SISO zpk → z; diğerleri tzero."""
    if sys.is_siso and isinstance(sys, TransferFunction):
        return roots(sys._num[0][0]) if np.any(sys._num[0][0]) else np.zeros(0, complex)
    if sys.is_siso and isinstance(sys, ZPK):
        return sys._z[0][0].copy()
    return tzero(sys)


def pzmap(sys: LTI):
    """Kutup-sıfır verisi (MATLAB `[p, z] = pzmap(sys)`)."""
    return pole(sys), zero(sys)


def damp(sys_or_p, Ts: float | None = None):
    """Doğal frekans, sönüm oranı ve kutuplar (MATLAB `[wn, zeta, p] = damp(sys)`), ω_n artan sırada."""
    if isinstance(sys_or_p, LTI):
        p = pole(sys_or_p)
        Ts = sys_or_p.Ts
    else:
        p = np.atleast_1d(np.asarray(sys_or_p, complex))
        Ts = 0.0 if Ts is None else float(Ts)
    if Ts > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(p == 0, -np.inf + 0j, np.log(p.astype(complex)) / Ts)
    else:
        s = p.astype(complex)
    wn = np.abs(s)
    with np.errstate(invalid="ignore", divide="ignore"):
        zeta = np.where(wn == 0, -1.0, -s.real / np.where(wn == 0, 1.0, wn))
    zeta = np.where(np.isinf(s.real) & (s.real < 0), 1.0, zeta)
    order = np.argsort(wn, kind="stable")
    return wn[order], zeta[order], p[order]


def isstable(sys: LTI) -> bool:
    """Asimptotik kararlılık (MATLAB `isstable`): sürekli Re(p) < 0, ayrık |p| < 1 (sınırdaki kutup kararsız)."""
    p = pole(sys)
    if p.size == 0:
        return True
    if sys.is_discrete:
        return bool(np.all(np.abs(p) < 1.0 - 1e-12))
    return bool(np.all(p.real < -1e-12 * np.maximum(1.0, np.abs(p))))


def dsort(p):
    """Ayrık kutupları genliğe göre azalan sıralar (MATLAB `[s, ndx] = dsort(p)`)."""
    p = np.atleast_1d(np.asarray(p, complex))
    idx = np.argsort(-np.abs(p), kind="stable")
    return p[idx], idx


def esort(p):
    """Sürekli kutupları gerçel kısma göre azalan sıralar (MATLAB `[s, ndx] = esort(p)`)."""
    p = np.atleast_1d(np.asarray(p, complex))
    idx = np.argsort(-p.real, kind="stable")
    return p[idx], idx


# ---------------------------------------------------------------------------------------------------------------
# Denetlenebilirlik / gözlenebilirlik / Gramyenler
# ---------------------------------------------------------------------------------------------------------------
def ctrb(A, B=None) -> np.ndarray:
    """Denetlenebilirlik matrisi [B, AB, …, A^{n−1}B] (MATLAB `ctrb(A, B)` ya da `ctrb(sys)`)."""
    if isinstance(A, LTI):
        s = convert.to_ss(A)
        A, B = s.A, s.B
    A = np.atleast_2d(np.asarray(A, float))
    B = np.asarray(B, float).reshape(A.shape[0], -1)
    n = A.shape[0]
    blocks = [B]
    for _ in range(1, n):
        blocks.append(A @ blocks[-1])
    return np.hstack(blocks) if n else np.zeros((0, B.shape[1] * 0))


def obsv(A, C=None) -> np.ndarray:
    """Gözlenebilirlik matrisi [C; CA; …; CA^{n−1}] (MATLAB `obsv(A, C)` ya da `obsv(sys)`)."""
    if isinstance(A, LTI):
        s = convert.to_ss(A)
        A, C = s.A, s.C
    A = np.atleast_2d(np.asarray(A, float))
    C = np.asarray(C, float).reshape(-1, A.shape[0])
    n = A.shape[0]
    blocks = [C]
    for _ in range(1, n):
        blocks.append(blocks[-1] @ A)
    return np.vstack(blocks) if n else np.zeros((0, 0))


def _require_stable(s: StateSpace, what: str):
    if s.nx == 0:
        return
    ev = np.linalg.eigvals(s.A)
    bad = np.any(np.abs(ev) >= 1.0 - 1e-12) if s.is_discrete else np.any(ev.real >= -1e-12 * np.maximum(1, abs(ev)))
    if bad:
        raise ControlError(f"{what} yalnız kararlı model için tanımlıdır (kararsız ya da sınırda kutup var).")


def gram(sys: LTI, kind: str = "c") -> np.ndarray:
    """Gramyen (MATLAB `gram`): 'c' denetlenebilirlik (A W + W Aᵀ + B Bᵀ = 0; ayrık A W Aᵀ − W + B Bᵀ = 0),
    'o' gözlenebilirlik. Kararlı model gerekir."""
    s = convert.to_ss(sys)
    _require_stable(s, "Gramyen")
    kind = str(kind).lower()
    if kind not in ("c", "o"):
        raise ControlError("Gramyen türü 'c' ya da 'o' olmalı.")
    if s.nx == 0:
        return np.zeros((0, 0))
    if kind == "c":
        W = sla.solve_discrete_lyapunov(s.A, s.B @ s.B.T) if s.is_discrete else \
            sla.solve_continuous_lyapunov(s.A, -s.B @ s.B.T)
    else:
        W = sla.solve_discrete_lyapunov(s.A.T, s.C.T @ s.C) if s.is_discrete else \
            sla.solve_continuous_lyapunov(s.A.T, -s.C.T @ s.C)
    return 0.5 * (W + W.T)


def covar(sys: LTI, W):
    """Beyaz gürültü girişli çıkış ve durum kovaryansı (MATLAB `[P, Q] = covar(sys, W)`); W gürültü şiddeti.
    Sürekli zamanda D ≠ 0 ise P sonsuzdur. Kararlı model gerekir."""
    s = convert.to_ss(sys)
    _require_stable(s, "Kovaryans")
    W = np.atleast_2d(np.asarray(W, float))
    if W.shape != (s.ninputs, s.ninputs):
        raise ControlError(f"W {s.ninputs}×{s.ninputs} olmalı.")
    n = s.nx
    if n:
        M = s.B @ W @ s.B.T
        Q = sla.solve_discrete_lyapunov(s.A, M) if s.is_discrete else sla.solve_continuous_lyapunov(s.A, -M)
        Q = 0.5 * (Q + Q.T)
    else:
        Q = np.zeros((0, 0))
    if not s.is_discrete:
        if np.any(s.D @ W @ s.D.T != 0):
            return np.full((s.noutputs, s.noutputs), np.inf), Q
        P = s.C @ Q @ s.C.T
    else:
        P = s.C @ Q @ s.C.T + s.D @ W @ s.D.T
    return 0.5 * (P + P.T), Q


# ---------------------------------------------------------------------------------------------------------------
# Kök yer eğrisi
# ---------------------------------------------------------------------------------------------------------------
def _siso_rational(sys: LTI):
    if not sys.is_siso:
        raise ControlError("Kök yer eğrisi SISO model içindir.")
    if sys.has_delay():
        if not sys.is_discrete:
            raise ControlError("Sürekli zamanda gecikmeli modelin kök yer eğrisi sonsuz dallıdır; önce Padé "
                               "yaklaşımı (pade) uygulayın.")
        from spt.control.discretize import absorb_delay
        sys = absorb_delay(sys)
    z = convert.to_zpk(sys)
    return z._z[0][0], z._p[0][0], float(z._k[0, 0]), sys


def _cl_roots(s: StateSpace, k: float) -> np.ndarray:
    d = 1.0 + k * float(s.D[0, 0])
    if abs(d) <= 1e-14:
        return np.full(s.nx, np.nan + 0j)
    Acl = s.A - k * np.outer(s.B[:, 0], s.C[0]) / d
    return np.linalg.eigvals(Acl)


def _match(prev: np.ndarray, new: np.ndarray) -> np.ndarray:
    ok_p = np.isfinite(prev)
    ok_n = np.isfinite(new)
    if not np.all(ok_p) or not np.all(ok_n):
        return new
    cost = np.abs(prev[:, None] - new[None, :])
    _, col = linear_sum_assignment(cost)
    return new[col]


def rlocus(sys: LTI, k=None):
    """Kök yer eğrisi (MATLAB `[r, k] = rlocus(sys, k)`): 1 + k·G = 0 kökleri, r (n, nk), kollar süreklilikle
    sıralı (ilk sütun açık çevrim kutupları). k verilmezse uyarlamalı kazanç ızgarası (bizim seçimimiz)."""
    zz, pp, kk, sys2 = _siso_rational(sys)
    s = convert.to_ss(sys2)
    n = s.nx
    if n == 0:
        raise ControlError("Statik kazancın kök yer eğrisi yok.")
    if k is None:
        kv = _auto_gains(s, zz, pp, kk)
    else:
        kv = np.atleast_1d(np.asarray(k, float)).ravel()
    R = np.zeros((n, kv.size), complex)
    prev = None
    for i, kval in enumerate(kv):
        r = _cl_roots(s, float(kval))
        if prev is None:
            if kval == 0:
                r = _order_like(r, pole(sys2))
        else:
            r = _match(prev, r)
        R[:, i] = r
        prev = r
    return R, kv


def _order_like(r: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if ref.size != r.size:
        return r
    cost = np.abs(ref[:, None] - r[None, :])
    _, col = linear_sum_assignment(cost)
    return r[col]


def _auto_gains(s: StateSpace, zz, pp, kk) -> np.ndarray:
    pts = np.concatenate([zz, pp]) if (zz.size or pp.size) else np.ones(1)
    scale = max(1.0, float(np.max(np.abs(pts))))
    s0 = 2.0 * scale * 1j
    num = kk * np.prod(s0 - zz)
    den = np.prod(s0 - pp)
    kref = abs(den / num) if num != 0 else 1.0
    kv = np.concatenate([[0.0], np.logspace(math.log10(kref) - 4, math.log10(kref) + 4, 200)])
    bk = [b for b in breakaway(_tfsys(zz, pp, kk, s.Ts))[1] if b > 0]
    kv = np.unique(np.concatenate([kv, bk]))
    # uyarlamalı sıklaştırma: ardışık kökler arası hareket ölçeğin %2'sinden büyükse ara kazanç ekle
    for _ in range(6):
        Rs = [np.sort_complex(_cl_roots(s, float(v))) for v in kv]
        add = []
        for i in range(kv.size - 1):
            a, b = Rs[i], Rs[i + 1]
            if np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
                mv = np.max(np.min(np.abs(a[:, None] - b[None, :]), axis=1))
                if mv > 0.02 * scale:
                    add.append(math.sqrt(kv[i] * kv[i + 1]) if kv[i] > 0 else 0.5 * kv[i + 1])
        if not add or kv.size > 2000:
            break
        kv = np.unique(np.concatenate([kv, add]))
    return kv


def _tfsys(zz, pp, kk, Ts):
    return ZPK(zz, pp, kk, Ts)


def breakaway(sys: LTI):
    """Kök yer eğrisinin gerçel eksen kırılma / birleşme noktaları ve kazançları: dK/ds = 0, K = −D/N gerçel ve
    pozitif olanlar. Dönüş: (noktalar, kazançlar)."""
    zz, pp, kk, _ = _siso_rational(sys)
    if kk == 0:
        return np.zeros(0, complex), np.zeros(0)
    N = kk * poly_from_roots(zz)
    D = poly_from_roots(pp)
    P = np.polysub(np.polymul(np.polyder(N), D), np.polymul(N, np.polyder(D)))
    cand = roots(P) if np.any(P) else np.zeros(0, complex)
    pts, gains = [], []
    for s0 in cand:
        nv = np.polyval(N, s0)
        if nv == 0:
            continue
        K = -np.polyval(D, s0) / nv
        if abs(K.imag) <= 1e-8 * max(1.0, abs(K)) and K.real > 0 and abs(s0.imag) <= 1e-8 * max(1.0, abs(s0)):
            pts.append(complex(s0.real, 0.0))
            gains.append(float(K.real))
    return np.asarray(pts, complex), np.asarray(gains)


def asymptotes(sys: LTI):
    """Kök yer eğrisi asimptotları: (merkez σ, açılar radyan). n − m asimptot; σ = (Σp − Σz)/(n − m),
    θ_q = (2q + 1)π/(n − m). Uygun olmayan / eşit dereceli modelde asimptot yoktur."""
    zz, pp, _, _ = _siso_rational(sys)
    d = pp.size - zz.size
    if d <= 0:
        return math.nan, np.zeros(0)
    sig = float(np.real(np.sum(pp) - np.sum(zz))) / d
    return sig, np.array([(2 * q + 1) * math.pi / d for q in range(d)])


__all__ = ["pole", "zero", "tzero", "pzmap", "damp", "isstable", "dsort", "esort", "ctrb", "obsv", "gram", "covar",
           "rlocus", "breakaway", "asymptotes"]
