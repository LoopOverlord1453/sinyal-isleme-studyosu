"""Model dönüşümleri ve indirgeme: tf ↔ zpk ↔ ss, minreal, sminreal, balreal, balred, modred, ss2ss, canon, hsvd.

MATLAB karşılıkları ve bilinen farklar:
- `ss(tf)`: SISO'da denetlenebilir yoldaş (companion) biçimi + köşegen dengeleme (MATLAB'in matrisleriyle aynı);
  MIMO'da MATLAB'in yapısı ölçülerek izlendi: sütun sütun (ya da daha küçük derece veriyorsa satır satır) gerçekleme;
  bir sütundaki bütün paydalar özdeşse ortak payda, değilse her eleman ayrı (blok köşegen). Gerçekleme MATLAB'inkiyle
  aynı dereceyi verir; matrisler aynı olmak zorunda değildir (karşılaştırma değişmezlerle: transfer, kutup, Markov).
- `ss(zpk)`: birinci / ikinci derece bölümlerin kaskadı (yüksek derecede polinom katsayısına göre çok daha sağlam).
- `tf(ss)` / `zpk(ss)`: her G/Ç çifti için yapısal indirgeme (sminreal), kutuplar = özdeğerler, sıfırlar =
  Rosenbrock kalemi üzerinden Emami-Naeini & Van Dooren indirgemesi, kazanç = bir değerlendirme noktasında oran.
- `minreal`: tf / zpk'de |z − p| ≤ tol·max(|z|, |p|) olan sıfır-kutup çiftleri sadeleşir (MATLAB'den ölçüldü,
  tol = √eps); ss'de dik merdiven (staircase) ile denetlenemez / gözlenemez durumlar atılır.
- `balreal` / `balred` / `hsvd`: kararsız kısım (sürekli Re ≥ 0, ayrık |z| ≥ 1) ayrılır ve korunur (HSV = ∞, MATLAB
  gibi); kararlı kısım karekök yöntemiyle dengelenir. `balred` varsayılanı MATLAB gibi 'MatchDC'.
- `canon(sys, 'modal')`: karmaşık çift [[σ, ω], [−ω, σ]] (ω > 0, MATLAB gibi); bloklar |λ| artan sırada. FARK: MATLAB'in
  blok sırası kendi Schur ayrıştırmasının sırasına bağlıdır (ölçülen 6 durumun 5'inde |λ| artan, birinde değil).
  `canon(sys, 'companion')`: ilk girişe göre Krylov tabanı (MATLAB gibi).
"""
from __future__ import annotations

import warnings

import numpy as np
import scipy.linalg as sla

from spt.control.errors import ControlError, ControlWarning
from spt.control.lti import (
    DELAY_INTERNAL_MSG, EPS, LTI, StateSpace, TransferFunction, ZPK, _blkdiag, _pzero, _strip, cplxpair,
    factor_delays, poly_from_roots, roots)

SQRT_EPS = float(np.sqrt(EPS))


# ---------------------------------------------------------------------------------------------------------------
# Rasyonel eleman yardımcıları
# ---------------------------------------------------------------------------------------------------------------
def _tf_entry_to_zpk(num: np.ndarray, den: np.ndarray):
    if _pzero(num):
        return np.zeros(0, complex), np.zeros(0, complex), 0.0
    return roots(num), roots(den), float(num[0] / den[0])


def _zpk_entry_to_tf(z: np.ndarray, p: np.ndarray, k: float):
    if k == 0.0:
        return np.zeros(1), np.ones(1)
    return k * poly_from_roots(z), poly_from_roots(p)


def _monic(p: np.ndarray) -> np.ndarray:
    return p / p[0]


def to_zpk(sys: LTI) -> ZPK:
    """Herhangi bir modeli zpk'ye çevirir (gecikmeler korunur)."""
    if isinstance(sys, ZPK):
        return sys
    meta = dict(input_names=sys.input_names, output_names=sys.output_names, name=sys.name,
                input_delay=sys.input_delay.copy(), output_delay=sys.output_delay.copy())
    ny, nu = sys.shape
    Z = [[None] * nu for _ in range(ny)]
    P = [[None] * nu for _ in range(ny)]
    K = np.zeros((ny, nu))
    if isinstance(sys, TransferFunction):
        for i in range(ny):
            for j in range(nu):
                Z[i][j], P[i][j], K[i, j] = _tf_entry_to_zpk(sys._num[i][j], sys._den[i][j])
        return ZPK(Z, P, K, sys.Ts, io_delay=sys.io_delay.copy(), **meta)
    if isinstance(sys, StateSpace):
        for i in range(ny):
            for j in range(nu):
                Z[i][j], P[i][j], K[i, j] = ss_entry_zpk(sys.A, sys.B[:, [j]], sys.C[[i], :], sys.D[i, j],
                                                         sys.is_discrete)
        return ZPK(Z, P, K, sys.Ts, **meta)
    raise ControlError("Bilinmeyen model türü.")


HIGH_ORDER = 12      # bu dereceden büyük polinom gösterimi sayısal olarak kötü koşullu sayılır (uyarı)


def _warn_order(n: int, what: str):
    if n > HIGH_ORDER:
        warnings.warn(f"{n}. derece {what}: polinom katsayıları sayısal olarak kötü koşullu olabilir; mümkünse zpk ya "
                      "da ss gösterimiyle çalışın.", ControlWarning, stacklevel=3)


def to_tf(sys: LTI) -> TransferFunction:
    """Herhangi bir modeli tf'ye çevirir (gecikmeler korunur). 12. dereceden yüksek elemanda uyarı verir."""
    if isinstance(sys, TransferFunction):
        return sys
    z = to_zpk(sys)
    ny, nu = z.shape
    N = [[None] * nu for _ in range(ny)]
    D = [[None] * nu for _ in range(ny)]
    for i in range(ny):
        for j in range(nu):
            N[i][j], D[i][j] = _zpk_entry_to_tf(z._z[i][j], z._p[i][j], float(z._k[i, j]))
            _warn_order(D[i][j].size - 1, "transfer fonksiyonu")
    return TransferFunction(N, D, z.Ts, io_delay=z.io_delay.copy(), input_names=z.input_names,
                            output_names=z.output_names, name=z.name, input_delay=z.input_delay.copy(),
                            output_delay=z.output_delay.copy())


# ---------------------------------------------------------------------------------------------------------------
# Gerçekleme (tf / zpk → ss)
# ---------------------------------------------------------------------------------------------------------------
def _balance(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    """Köşegen (2'nin kuvvetleri) benzerlik dengelemesi; transferi değiştirmez."""
    if A.shape[0] <= 1:
        return A, B, C
    _, (scale, _perm) = sla.matrix_balance(A, permute=False, separate=True)
    s = scale.astype(float) / float(scale[0])
    return A * (s[None, :] / s[:, None]), B / s[:, None], C * s[None, :]


def _companion_simo(nums: list[np.ndarray], den: np.ndarray):
    """Ortak paydalı tek girişli çok çıkışlı gerçekleme (denetlenebilir yoldaş biçimi)."""
    den = _strip(np.asarray(den, float))
    a0 = den[0]
    den = den / a0
    n = den.size - 1
    p = len(nums)
    C = np.zeros((p, n))
    D = np.zeros((p, 1))
    for i, num in enumerate(nums):
        num = _strip(np.asarray(num, float)) / a0
        if num.size > n + 1:
            raise ControlError("Uygun olmayan (pay derecesi paydadan büyük) transfer fonksiyonu durum uzayına "
                               "çevrilemez.")
        b = np.zeros(n + 1)
        b[n + 1 - num.size:] = num
        D[i, 0] = b[0]
        C[i] = b[1:] - b[0] * den[1:]
    A = np.zeros((n, n))
    if n:
        A[0] = -den[1:]
        A[1:, :-1] = np.eye(n - 1)
    B = np.zeros((n, 1))
    if n:
        B[0, 0] = 1.0
    A, B, C = _balance(A, B, C)
    return A, B, C, D


def _sections(r: np.ndarray) -> list[np.ndarray]:
    """Kökleri gerçel ikinci derece (önce) ve en çok bir birinci derece çarpana ayırır."""
    r = cplxpair(r, tol=1e-9)
    quad: list[np.ndarray] = []
    reals: list[float] = []
    i = 0
    while i < r.size:
        if r[i].imag != 0.0:
            quad.append(np.array([1.0, -2.0 * r[i].real, r[i].real ** 2 + r[i].imag ** 2]))
            i += 2
        else:
            reals.append(r[i].real)
            i += 1
    reals.sort()
    while len(reals) >= 2:
        a, b = reals.pop(0), reals.pop(0)
        quad.append(np.array([1.0, -(a + b), a * b]))
    lin = [np.array([1.0, -reals[0]])] if reals else []
    return quad + lin


def _zpk_siso_ss(z: np.ndarray, p: np.ndarray, k: float):
    """SISO zpk → ss: birinci / ikinci derece bölümlerin kaskadı."""
    if k == 0.0:
        return np.zeros((0, 0)), np.zeros((0, 1)), np.zeros((1, 0)), np.zeros((1, 1))
    if z.size > p.size:
        raise ControlError("Uygun olmayan (sıfır sayısı kutup sayısından fazla) model durum uzayına çevrilemez.")
    ps = _sections(p)
    zs = _sections(z)
    A = np.zeros((0, 0))
    B = np.zeros((0, 1))
    C = np.zeros((1, 0))
    D = np.ones((1, 1))
    for idx, den in enumerate(ps):
        num = zs[idx] if idx < len(zs) else np.ones(1)
        a, b, c, d = _companion_simo([num], den)
        # kaskad: önceki çıkış → bu bölümün girişi
        n0, n1 = A.shape[0], a.shape[0]
        An = np.zeros((n0 + n1, n0 + n1))
        An[:n0, :n0] = A
        An[n0:, :n0] = b @ C
        An[n0:, n0:] = a
        B = np.vstack([B, b @ D])
        C = np.hstack([d @ C, c])
        D = d @ D
        A = An
    C = k * C
    D = k * D
    return _balance(A, B, C) + (D,)


def _entry_dyn(sys: LTI, i: int, j: int) -> bool:
    if isinstance(sys, TransferFunction):
        return sys._den[i][j].size > 1 and not _pzero(sys._num[i][j])
    return sys._p[i][j].size > 0 and sys._k[i, j] != 0.0


def _entry_den_key(sys: LTI, i: int, j: int):
    if isinstance(sys, TransferFunction):
        return tuple(_monic(sys._den[i][j]).tolist())
    p = np.sort_complex(sys._p[i][j])
    return tuple((float(v.real), float(v.imag)) for v in p)


def _realize_line(sys: LTI, idx: list[tuple[int, int]]):
    """Tek giriş (sütun) gerçeklemesi: idx = o sütundaki (i, j) elemanları (çıkış sırasıyla)."""
    dyn = [e for e in idx if _entry_dyn(sys, *e)]
    keys = {_entry_den_key(sys, *e) for e in dyn}
    p = len(idx)
    if len(keys) == 1:
        if isinstance(sys, TransferFunction):
            i0, j0 = dyn[0]
            den = sys._den[i0][j0]
            nums = []
            for (i, j) in idx:
                n, d = sys._num[i][j], sys._den[i][j]
                if _entry_dyn(sys, i, j):
                    nums.append(n * (den[0] / d[0]))      # aynı (monik) payda; ölçek farkını paya taşı
                else:
                    nums.append(np.polymul(n / d[0], den))
            return _companion_simo(nums, den)
        i0, j0 = dyn[0]
        if sys._p[i0][j0].size <= 12:
            den = poly_from_roots(sys._p[i0][j0])
            nums = []
            for (i, j) in idx:
                k = float(sys._k[i, j])
                if _entry_dyn(sys, i, j):
                    nums.append(k * poly_from_roots(sys._z[i][j]))
                else:
                    nums.append(np.polymul([k], den))
            return _companion_simo(nums, den)
    # her eleman ayrı (blok köşegen, ortak giriş)
    As, Bs, Cs = [], [], []
    D = np.zeros((p, 1))
    for r, (i, j) in enumerate(idx):
        if isinstance(sys, TransferFunction):
            a, b, c, d = _companion_simo([sys._num[i][j]], sys._den[i][j])
        else:
            a, b, c, d = _zpk_siso_ss(sys._z[i][j], sys._p[i][j], float(sys._k[i, j]))
        cc = np.zeros((p, c.shape[1]))
        cc[r] = c[0]
        As.append(a)
        Bs.append(b)
        Cs.append(cc)
        D[r, 0] = d[0, 0]
    n = sum(a.shape[0] for a in As)
    A = _blkdiag(*As) if As else np.zeros((0, 0))
    B = np.vstack(Bs) if Bs else np.zeros((n, 1))
    C = np.hstack(Cs) if Cs else np.zeros((p, 0))
    return A, B.reshape(n, 1), C.reshape(p, n), D


def _realize_columns(sys: LTI, transpose: bool):
    ny, nu = sys.shape
    lines = range(ny) if transpose else range(nu)
    parts = []
    for L in lines:
        idx = [(L, j) for j in range(nu)] if transpose else [(i, L) for i in range(ny)]
        parts.append(_realize_line(sys, idx))
    A = _blkdiag(*[pt[0] for pt in parts]) if parts else np.zeros((0, 0))
    n = A.shape[0]
    q = len(parts)
    other = nu if transpose else ny
    B = np.zeros((n, q))
    C = np.zeros((other, n))
    D = np.zeros((other, q))
    off = 0
    for c, (a, b, cc, d) in enumerate(parts):
        k = a.shape[0]
        B[off:off + k, c] = b[:, 0]
        C[:, off:off + k] = cc
        D[:, c] = d[:, 0]
        off += k
    if transpose:
        return A.T.copy(), C.T.copy(), B.T.copy(), D.T.copy()
    return A, B, C, D


def _realize(sys: LTI):
    """tf / zpk → (A, B, C, D) (gecikmesiz). MATLAB'in derecesini veren sütun / satır gerçeklemesi."""
    if not sys.is_proper():
        raise ControlError("Uygun olmayan (pay derecesi paydadan büyük) model durum uzayına çevrilemez.")
    if isinstance(sys, TransferFunction):
        _warn_order(max(d.size - 1 for row in sys._den for d in row), "transfer fonksiyonundan gerçekleme")
    ny, nu = sys.shape
    cols = _realize_columns(sys, transpose=False)
    if ny == 1 and nu == 1:
        return cols
    rows = _realize_columns(sys, transpose=True)
    return rows if rows[0].shape[0] < cols[0].shape[0] else cols


def to_ss(sys: LTI) -> StateSpace:
    """Herhangi bir modeli ss'ye çevirir. tf / zpk'nin G/Ç gecikmeleri giriş / çıkış gecikmesine ayrıştırılır;
    ayrıştırılamıyorsa ayrık zamanda kalan gecikme z⁻¹ durumu olarak katılır, sürekli zamanda hata verilir."""
    if isinstance(sys, StateSpace):
        return sys
    ny, nu = sys.shape
    ind, outd = sys.input_delay.copy(), sys.output_delay.copy()
    work = sys
    io = sys.io_delay
    if np.any(io):
        mask = sys._structure()
        res = factor_delays(io, mask)
        if res is not None:
            ind = ind + res[0]
            outd = outd + res[1]
            work = sys.copy()
            work.io_delay = np.zeros_like(io)
        elif sys.is_discrete:
            # ayrıştırılamayan kısım: kalan gecikmeyi elemana kutup (z = 0) olarak kat
            a_in = np.array([min([io[i, j] for i in range(ny) if mask[i, j]] or [0.0]) for j in range(nu)])
            work = sys.copy()
            for i in range(ny):
                for j in range(nu):
                    r = int(round(io[i, j] - a_in[j])) if mask[i, j] else 0
                    if isinstance(work, TransferFunction):
                        if r > 0:
                            work._den[i][j] = np.concatenate([work._den[i][j], np.zeros(r)])
                    elif r > 0:
                        work._p[i][j] = np.concatenate([work._p[i][j], np.zeros(r, complex)])
            work.io_delay = np.zeros_like(io)
            ind = ind + a_in
        else:
            raise ControlError("G/Ç gecikmeleri giriş / çıkış gecikmesi olarak ayrıştırılamıyor; durum uzayında "
                               "iç gecikme gerekir. " + DELAY_INTERNAL_MSG)
    A, B, C, D = _realize(work)
    return StateSpace(A, B, C, D, sys.Ts, input_delay=ind, output_delay=outd, input_names=sys.input_names,
                      output_names=sys.output_names, name=sys.name)


# ---------------------------------------------------------------------------------------------------------------
# Sıfırlar (Emami-Naeini & Van Dooren 1982) ve ss → zpk
# ---------------------------------------------------------------------------------------------------------------
def _zero_tol(A, B, C, D) -> float:
    M = np.block([[A, B], [C, D]]) if A.size else D
    nrm = np.linalg.norm(M, "fro") if M.size else 0.0
    return 10.0 * max(M.shape + (1,)) * EPS * max(nrm, 1.0)


def _reduce(A, B, C, D, tol):
    """Değişmez sıfırları koruyarak D'yi tam satır ranklı yapar (sistem kalemi indirgemesi)."""
    while True:
        n, m = B.shape
        p = C.shape[0]
        if p == 0:
            return A, B, C, D
        if D.size and m:
            U, s, _ = np.linalg.svd(D, full_matrices=True)
            rho = int(np.sum(s > tol))
        else:
            U, rho = np.eye(p), 0
        Ur = np.hstack([U[:, rho:], U[:, :rho]])
        C = Ur.T @ C
        D = Ur.T @ D
        sigma = p - rho
        if sigma == 0:
            return A, B, C, D
        if n == 0:
            return A, B, C[sigma:], D[sigma:]
        C1 = C[:sigma]
        _, s1, Vh1 = np.linalg.svd(C1, full_matrices=True)
        tau = int(np.sum(s1 > tol))
        if tau == 0:
            C, D = C[sigma:], D[sigma:]
            continue
        W = Vh1.T[:, ::-1]
        A = W.T @ A @ W
        B = W.T @ B
        C = C @ W
        k = n - tau
        A, B, C, D = (A[:k, :k], B[:k], np.vstack([A[k:, :k], C[sigma:, :k]]), np.vstack([B[k:], D[sigma:]]))


def invariant_zeros(A, B, C, D):
    """Değişmez sıfırlar (sonlu) ve normal rank: Rosenbrock kalemi [[A − λI, B], [C, D]]."""
    A, B, C, D = (np.asarray(M, float) for M in (A, B, C, D))
    n = A.shape[0]
    tol = _zero_tol(A, B, C, D)
    A1, B1, C1, D1 = _reduce(A, B, C, D, tol)
    A2, C2, B2, D2 = _reduce(A1.T, C1.T, B1.T, D1.T, tol)
    A2, B2, C2, D2 = A2.T, B2.T, C2.T, D2.T
    nr = A2.shape[0]
    rank = D2.shape[0]
    if n == 0 or nr == 0 or rank == 0:
        return np.zeros(0, complex), rank
    CD = np.hstack([C2, D2])
    _, _, Vh = np.linalg.svd(CD, full_matrices=True)
    V = np.hstack([Vh[rank:].T, Vh[:rank].T])
    AB = np.hstack([A2, B2]) @ V
    IE = np.hstack([np.eye(nr), np.zeros((nr, B2.shape[1]))]) @ V
    Af, Bf = AB[:, :nr], IE[:, :nr]
    alpha, beta = sla.eig(Af, Bf, right=False, homogeneous_eigvals=True)
    fin = np.abs(beta) > 1e3 * EPS * np.maximum(np.abs(alpha), 1.0)
    z = alpha[fin] / beta[fin]
    return cplxpair(z, tol=1e-7), rank


def _sminreal_arrays(A, B, C):
    """Yapısal olarak ulaşılamayan / gözlenemeyen durumların indisleri dışındakiler (keep maskesi)."""
    n = A.shape[0]
    if n == 0:
        return np.zeros(0, bool)
    adj = A != 0.0
    reach = np.any(B != 0.0, axis=1)
    for _ in range(n):
        new = reach | np.any(adj[:, reach], axis=1)
        if np.array_equal(new, reach):
            break
        reach = new
    obs = np.any(C != 0.0, axis=0)
    for _ in range(n):
        new = obs | np.any(adj[obs, :], axis=0)
        if np.array_equal(new, obs):
            break
        obs = new
    return reach & obs


def _gain_point(p: np.ndarray, z: np.ndarray, discrete: bool) -> complex:
    pts = np.concatenate([p, z]) if (p.size or z.size) else np.zeros(1)
    scale = max(1.0, float(np.max(np.abs(pts)))) * (1.6 if not discrete else 1.6)
    best, best_d = None, -1.0
    for ang in np.linspace(0.13, 2.9, 9):
        s0 = scale * np.exp(1j * ang)
        d = float(np.min(np.abs(pts - s0))) if pts.size else scale
        if d > best_d:
            best, best_d = s0, d
    return complex(best)


def ss_entry_zpk(A, B, C, d, discrete: bool = False):
    """SISO durum uzayı (b: n×1, c: 1×n) → (z, p, k); yapısal olarak bağlı olmayan durumlar atılır."""
    keep = _sminreal_arrays(A, B, C)
    a, b, c = A[np.ix_(keep, keep)], B[keep], C[:, keep]
    d = float(np.asarray(d).ravel()[0])
    if a.shape[0] == 0:
        return np.zeros(0, complex), np.zeros(0, complex), d
    z, rank = invariant_zeros(a, b, c, np.array([[d]]))
    if rank == 0:
        return np.zeros(0, complex), np.zeros(0, complex), 0.0
    p = cplxpair(np.linalg.eigvals(a), tol=1e-9)
    s0 = _gain_point(p, z, discrete)
    g = complex((c @ np.linalg.solve(s0 * np.eye(a.shape[0]) - a, b))[0, 0] + d)
    k = g * np.prod(s0 - p) / np.prod(s0 - z)
    return z, p, float(k.real)


# ---------------------------------------------------------------------------------------------------------------
# minreal / sminreal
# ---------------------------------------------------------------------------------------------------------------
def _ctrb_staircase(A, B, C, tol):
    """Dik merdiven biçimi: denetlenebilir kısım ilk nc durum."""
    n = A.shape[0]
    A, B, C = A.copy(), B.copy(), C.copy()
    Z = np.eye(n)
    nc = 0
    blk = B
    while nc < n and blk.size:
        U, s, _ = np.linalg.svd(blk, full_matrices=True)
        r = int(np.sum(s > tol))
        if r == 0:
            break
        T = np.eye(n)
        T[nc:, nc:] = U
        A = T.T @ A @ T
        B = T.T @ B
        C = C @ T
        Z = Z @ T
        blk = A[nc + r:, nc:nc + r]
        nc += r
    return A, B, C, Z, nc


def sminreal(sys: LTI) -> LTI:
    """Yapısal en küçük gerçekleme (MATLAB `sminreal`): girişlerden ulaşılamayan ya da çıkışlara bağlı olmayan
    durumları (A, B, C'nin sıfır deseninden) atar. tf / zpk değişmeden döner."""
    if not isinstance(sys, StateSpace):
        return sys
    keep = _sminreal_arrays(sys.A, sys.B, sys.C)
    names = [s for s, k in zip(sys.state_names, keep) if k]
    return StateSpace(sys.A[np.ix_(keep, keep)], sys.B[keep], sys.C[:, keep], sys.D, sys.Ts, state_names=names,
                      input_delay=sys.input_delay, output_delay=sys.output_delay, input_names=sys.input_names,
                      output_names=sys.output_names, name=sys.name)


def _cancel(z: np.ndarray, p: np.ndarray, tol: float):
    z = list(z)
    p = list(p)
    zi = 0
    while zi < len(z):
        best, bj = None, -1
        for j, pv in enumerate(p):
            dist = abs(z[zi] - pv)
            if dist <= tol * max(abs(z[zi]), abs(pv)) and (best is None or dist < best):
                best, bj = dist, j
        if bj >= 0:
            p.pop(bj)
            z.pop(zi)
        else:
            zi += 1
    return cplxpair(np.asarray(z, complex), tol=1e-7), cplxpair(np.asarray(p, complex), tol=1e-7)


def minreal(sys: LTI, tol: float | None = None) -> LTI:
    """En küçük gerçekleme / sıfır-kutup sadeleştirmesi (MATLAB `minreal`, varsayılan tol = √eps).

    tf / zpk: her elemanda |z − p| ≤ tol·max(|z|, |p|) olan çiftler sadeleşir (tf sonucu monik paydalıdır).
    ss: dik merdiven yöntemiyle denetlenemez ve gözlenemez durumlar atılır (rank eşiği tol·max(1, ‖[A B; C D]‖))."""
    tol = SQRT_EPS if tol is None else float(tol)
    if isinstance(sys, (TransferFunction, ZPK)):
        z = to_zpk(sys).copy()
        ny, nu = z.shape
        for i in range(ny):
            for j in range(nu):
                z._z[i][j], z._p[i][j] = _cancel(z._z[i][j], z._p[i][j], tol)
        return to_tf(z) if isinstance(sys, TransferFunction) else z
    A, B, C, D = sys.A, sys.B, sys.C, sys.D
    n = A.shape[0]
    if n == 0:
        return sys.copy()
    nrm = np.linalg.norm(np.block([[A, B], [C, D]]), 2)
    t = tol * max(1.0, nrm)
    A1, B1, C1, _, nc = _ctrb_staircase(A, B, C, t)
    A1, B1, C1 = A1[:nc, :nc], B1[:nc], C1[:, :nc]
    At, Ct, Bt, _, no = _ctrb_staircase(A1.T, C1.T, B1.T, t)
    A2, B2, C2 = At[:no, :no].T, Bt[:, :no].T, Ct[:no].T
    return StateSpace(A2, B2, C2, D, sys.Ts, input_delay=sys.input_delay, output_delay=sys.output_delay,
                      input_names=sys.input_names, output_names=sys.output_names, name=sys.name)


# ---------------------------------------------------------------------------------------------------------------
# Kararlı / kararsız ayrıştırma, Gramyenler, dengeli gerçekleme
# ---------------------------------------------------------------------------------------------------------------
def _unstable_mask(ev: np.ndarray, discrete: bool) -> np.ndarray:
    if discrete:
        return np.abs(ev) >= 1.0 - 1e-12
    return ev.real >= -1e-12 * np.maximum(1.0, np.abs(ev))


def stabsep(sys: StateSpace, return_transform: bool = False):
    """Kararsız (önce) / kararlı ayrıştırma: G = G_u + G_s. Dönüş: (A_u, B_u, C_u), (A_s, B_s, C_s, D)
    [, W, Wi] — x = W·[x_u; x_s], [x_u; x_s] = Wi·x. Sınırdaki (sürekli Re = 0, ayrık |z| = 1) kutuplar kararsız
    sayılır (MATLAB gibi)."""
    A, B, C = sys.A, sys.B, sys.C
    n = A.shape[0]
    m, p = B.shape[1], C.shape[0]
    if n == 0:
        z = (np.zeros((0, 0)), np.zeros((0, m)), np.zeros((p, 0)))
        res = (z, z + (sys.D,))
        return res + (np.zeros((0, 0)), np.zeros((0, 0))) if return_transform else res
    if sys.is_discrete:
        T, Z, nu = sla.schur(A, output="real", sort=lambda x, y: x * x + y * y >= 1.0 - 2e-12)
    else:
        T, Z, nu = sla.schur(A, output="real", sort=lambda x, y: x >= -1e-12 * max(1.0, abs(complex(x, y))))
    S11, S12, S22 = T[:nu, :nu], T[:nu, nu:], T[nu:, nu:]
    if 0 < nu < n:
        X = sla.solve_sylvester(S11, -S22, -S12)
    else:
        X = np.zeros((nu, n - nu))
    # x = Z·[[I, X], [0, I]]·x̃  →  Ã = blkdiag(S11, S22)
    E = np.eye(n)
    E[:nu, nu:] = X
    Ei = np.eye(n)
    Ei[:nu, nu:] = -X
    W, Wi = Z @ E, Ei @ Z.T
    Bt, Ct = Wi @ B, C @ W
    res = ((S11, Bt[:nu], Ct[:, :nu]), (S22, Bt[nu:], Ct[:, nu:], sys.D))
    return res + (W, Wi) if return_transform else res


def _gramians(A, B, C, discrete):
    if discrete:
        P = sla.solve_discrete_lyapunov(A, B @ B.T)
        Q = sla.solve_discrete_lyapunov(A.T, C.T @ C)
    else:
        P = sla.solve_continuous_lyapunov(A, -B @ B.T)
        Q = sla.solve_continuous_lyapunov(A.T, -C.T @ C)
    return 0.5 * (P + P.T), 0.5 * (Q + Q.T)


def _psd_factor(P: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(P)
    w = np.clip(w, 0.0, None)
    return V * np.sqrt(w)[None, :]


def _balance_stable(A, B, C, discrete):
    """Kararlı kısmın dengeli gerçeklemesi: (Ab, Bb, Cb, g, T, Ti) ile Ab = T A Ti."""
    n = A.shape[0]
    if n == 0:
        return A, B, C, np.zeros(0), np.zeros((0, 0)), np.zeros((0, 0))
    P, Q = _gramians(A, B, C, discrete)
    Lc, Lo = _psd_factor(P), _psd_factor(Q)
    U, g, Vh = np.linalg.svd(Lo.T @ Lc)
    if g[-1] <= 1e-14 * max(g[0], 1e-300):
        warnings.warn("Model en küçük gerçekleme değil (çok küçük Hankel tekil değerleri); dengeli gerçekleme kötü "
                      "koşullu. Önce minreal uygulayın.", ControlWarning, stacklevel=3)
    gs = np.maximum(g, 1e-300)
    si = 1.0 / np.sqrt(gs)
    T = (si[:, None] * U.T) @ Lo.T
    Ti = Lc @ Vh.T * si[None, :]
    return T @ A @ Ti, T @ B, C @ Ti, g, T, Ti


def hsvd(sys: LTI) -> np.ndarray:
    """Hankel tekil değerleri (MATLAB `hsvd`): kararsız kısım için ∞ (önce), sonra azalan sırada."""
    s = to_ss(sys)
    (Au, _, _), (As, Bs, Cs, _) = stabsep(s)
    _, _, _, g, _, _ = _balance_stable(As, Bs, Cs, s.is_discrete)
    return np.concatenate([np.full(Au.shape[0], np.inf), g])


def balreal(sys: LTI):
    """Dengeli gerçekleme (MATLAB `[sysb, g, T, Ti] = balreal(sys)`): kararlı kısmın kontrol edilebilirlik ve
    gözlenebilirlik Gramyenleri eşit ve köşegen (= diag(g)); kararsız kısım (g = ∞) önce ve korunur.

    Dönüş: (sysb, g, T, Ti) — xb = T·x, x = Ti·xb."""
    s = to_ss(sys)
    (Au, Bu, Cu), (As, Bs, Cs, D), W, Wi = stabsep(s, return_transform=True)
    Ab, Bb, Cb, g, T, Ti = _balance_stable(As, Bs, Cs, s.is_discrete)
    nu = Au.shape[0]
    A = _blkdiag(Au, Ab)
    B = np.vstack([Bu, Bb])
    C = np.hstack([Cu, Cb])
    gg = np.concatenate([np.full(nu, np.inf), g])
    out = StateSpace(A, B, C, D, s.Ts, input_delay=s.input_delay, output_delay=s.output_delay,
                     input_names=s.input_names, output_names=s.output_names, name=s.name)
    Tfull = _blkdiag(np.eye(nu), T) @ Wi
    Tinv = W @ _blkdiag(np.eye(nu), Ti)
    return out, gg, Tfull, Tinv


def _residualize(A, B, C, D, r, method, discrete):
    """İlk r durumu tut, gerisini at ('Truncate') ya da DC kazancını koruyarak yok et ('MatchDC')."""
    A11, A12, A21, A22 = A[:r, :r], A[:r, r:], A[r:, :r], A[r:, r:]
    B1, B2, C1, C2 = B[:r], B[r:], C[:, :r], C[:, r:]
    if method == "truncate" or A22.size == 0:
        return A11, B1, C1, D.copy()
    M = (np.eye(A22.shape[0]) - A22) if discrete else -A22
    try:
        X = np.linalg.solve(M, np.hstack([A21, B2]))
    except np.linalg.LinAlgError:
        raise ControlError("Atılan durumların alt matrisi tekil; 'MatchDC' uygulanamıyor ('Truncate' deneyin).") \
            from None
    XA, XB = X[:, :r], X[:, r:]
    return A11 + A12 @ XA, B1 + A12 @ XB, C1 + C2 @ XA, D + C2 @ XB


def _method(m: str) -> str:
    key = str(m).lower().replace("_", "")
    if key in ("matchdc", "dc"):
        return "matchdc"
    if key in ("truncate", "trunc"):
        return "truncate"
    raise ControlError("Yöntem 'MatchDC' ya da 'Truncate' olmalı.")


def balred(sys: LTI, order: int, method: str = "MatchDC") -> StateSpace:
    """Dengeli kesmeyle model indirgeme (MATLAB `balred`, varsayılan 'MatchDC' = tekil pertürbasyon, DC kazancı
    korunur; 'Truncate' = doğrudan kesme). Kararsız kutuplar korunur; `order` en az kararsız kutup sayısı olmalı."""
    method = _method(method)
    s = to_ss(sys)
    order = int(order)
    (Au, Bu, Cu), (As, Bs, Cs, D) = stabsep(s)
    nu = Au.shape[0]
    if order < nu:
        raise ControlError(f"Hedef derece ({order}) kararsız kutup sayısından ({nu}) küçük olamaz.")
    if order > s.nx:
        raise ControlError(f"Hedef derece ({order}) model derecesinden ({s.nx}) büyük olamaz.")
    Ab, Bb, Cb, g, _, _ = _balance_stable(As, Bs, Cs, s.is_discrete)
    Ar, Br, Cr, Dr = _residualize(Ab, Bb, Cb, D, order - nu, method, s.is_discrete)
    return StateSpace(_blkdiag(Au, Ar), np.vstack([Bu, Br]), np.hstack([Cu, Cr]), Dr, s.Ts,
                      input_delay=s.input_delay, output_delay=s.output_delay, input_names=s.input_names,
                      output_names=s.output_names, name=s.name)


def modred(sys: LTI, elim, method: str = "MatchDC") -> StateSpace:
    """Belirtilen durumları atar (MATLAB `modred`, varsayılan 'MatchDC'). `elim`: indis listesi ya da bool maske.

    'MatchDC' atılan blok A22 (ayrıkta I − A22) tersinir olmalıdır; tekilse hata verilir (MATLAB bu durumda tanımlayıcı
    (descriptor) modele geçer — burada E matrisi yoktur). Dengeli gerçeklemede (balreal) A22 tersinirdir."""
    method = _method(method)
    s = to_ss(sys)
    n = s.nx
    e = np.asarray(elim)
    mask = e.astype(bool) if (e.dtype == bool and e.size == n) else np.isin(np.arange(n), e.astype(int))
    keep = np.flatnonzero(~mask)
    drop = np.flatnonzero(mask)
    perm = np.concatenate([keep, drop])
    A = s.A[np.ix_(perm, perm)]
    Ar, Br, Cr, Dr = _residualize(A, s.B[perm], s.C[:, perm], s.D, keep.size, method, s.is_discrete)
    return StateSpace(Ar, Br, Cr, Dr, s.Ts, state_names=[s.state_names[i] for i in keep],
                      input_delay=s.input_delay, output_delay=s.output_delay, input_names=s.input_names,
                      output_names=s.output_names, name=s.name)


def ss2ss(sys: LTI, T) -> StateSpace:
    """Benzerlik dönüşümü x̄ = T x (MATLAB `ss2ss`): Ā = T A T⁻¹, B̄ = T B, C̄ = C T⁻¹."""
    s = to_ss(sys)
    T = np.asarray(T, float)
    if T.shape != (s.nx, s.nx):
        raise ControlError(f"Dönüşüm matrisi {s.nx}×{s.nx} olmalı.")
    try:
        Ti = np.linalg.inv(T)
    except np.linalg.LinAlgError:
        raise ControlError("Dönüşüm matrisi tekil.") from None
    return StateSpace(T @ s.A @ Ti, T @ s.B, s.C @ Ti, s.D, s.Ts, input_delay=s.input_delay,
                      output_delay=s.output_delay, input_names=s.input_names, output_names=s.output_names,
                      name=s.name)


def canon(sys: LTI, form: str = "modal"):
    """Kanonik biçim (MATLAB `[csys, T] = canon(sys, type)`), xc = T x.

    'modal': blok köşegen; özdeğerler |λ| artan sırada, karmaşık çift σ ± jω için [[σ, ω], [−ω, σ]].
    'companion': ilk girişin Krylov tabanı; A son sütunu karakteristik polinom katsayıları (−a_n … −a_1),
    alt köşegen birler, B'nin ilk sütunu e1."""
    s = to_ss(sys)
    n = s.nx
    form = str(form).lower()
    if n == 0:
        return s.copy(), np.zeros((0, 0))
    if form == "companion":
        b = s.B[:, 0]
        K = np.column_stack([np.linalg.matrix_power(s.A, k) @ b for k in range(n)])
        if np.linalg.matrix_rank(K) < n or np.linalg.cond(K) > 1e14:
            raise ControlError("Model ilk girişten denetlenebilir değil (ya da çok kötü koşullu); yoldaş biçimi "
                               "kurulamıyor.")
        T = np.linalg.inv(K)
        return ss2ss(s, T), T
    if form != "modal":
        raise ControlError("Biçim 'modal' ya da 'companion' olmalı.")
    ev, V = np.linalg.eig(s.A)
    order = np.argsort(np.abs(ev), kind="stable")
    ev, V = ev[order], V[:, order]
    cols, used = [], np.zeros(n, bool)
    for i in range(n):
        if used[i]:
            continue
        used[i] = True
        if abs(ev[i].imag) <= 1e-12 * max(1.0, abs(ev[i])):
            v = V[:, i].real if np.linalg.norm(V[:, i].real) >= np.linalg.norm(V[:, i].imag) else V[:, i].imag
            cols.append(v / np.linalg.norm(v))
            continue
        j = next((k for k in range(i + 1, n) if not used[k] and abs(ev[k] - np.conj(ev[i])) <=
                  1e-8 * max(1.0, abs(ev[i]))), None)
        if j is not None:
            used[j] = True
        v = V[:, i] if ev[i].imag > 0 else V[:, j if j is not None else i]
        cols.extend([v.real, v.imag])
    Ti = np.column_stack(cols)
    if np.linalg.cond(Ti) > 1e10:
        raise ControlError("Özvektörler neredeyse bağımlı (tekrarlı / kusurlu özdeğerler); modal biçim sayısal "
                           "olarak güvenilir kurulamıyor.")
    T = np.linalg.inv(Ti)
    out = ss2ss(s, T)
    # blok yapısını tam yap (yuvarlama kalıntılarını temizle)
    A = out.A
    mask = np.zeros_like(A, bool)
    i = 0
    while i < n:
        if i + 1 < n and abs(A[i + 1, i]) > 1e-12 * max(1.0, np.abs(A).max()) and \
                abs(A[i, i + 1]) > 1e-12 * max(1.0, np.abs(A).max()):
            mask[i:i + 2, i:i + 2] = True
            i += 2
        else:
            mask[i, i] = True
            i += 1
    out.A = np.where(mask, A, 0.0)
    return out, T
