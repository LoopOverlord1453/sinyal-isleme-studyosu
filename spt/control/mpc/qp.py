"""Yoğun (dense) konveks QP çözücüsü — Goldfarb–Idnani çift (dual) aktif küme yöntemi.

Problem (iç biçim; MATLAB'in MPC çekirdeği ``qpkwik`` ile aynı)::

    min  ½ xᵀ H x + fᵀ x
    s.t. Ac x ≥ bc        (m eşitsizlik)
         Aeq x = beq      (q eşitlik)

H simetrik pozitif tanımlı olmalıdır (kesin konveks QP → çözüm tektir). Kullanıcı biçimi MATLAB
``mpcActiveSetSolver`` / ``quadprog`` gibi ``A x ≤ b``'dir (:func:`qp_solve`); MPC iç biçimi ``Ac x ≥ bc``'dir
(:func:`qp_solve_ge`, ``Ac = −A``, ``bc = −b``).

MATLAB karşılığı ve ilişkisi
----------------------------
MATLAB MPC Toolbox'ın varsayılan 'active-set' çözücüsü KWIK (Schmid & Biegler, 1994) Goldfarb–Idnani (GI, 1983) çift
yönteminin türevidir; kaynak kodu yayımlanmamıştır (p-kod). Burada GI'nin yayımlanmış, sayısal kararlı biçimi
(Cholesky + Givens tabanlı QR güncellemeleri) uygulanır. Kesin konveks QP'nin çözümü tek olduğundan sonuç (x, aktif
küme, çarpanlar) KWIK ve quadprog ile aynıdır (testler: ~1e-14). Yineleme sayısı MATLAB'in sayımıyla aynı kuralla
verilir: ``1 + (ekleme + çıkarma adımları)``; eşitlikler ve sıcak başlangıç eklemeleri sayılmaz, sıcak başlangıçta
negatif çarpanlı kısıt çıkarmaları sayılır (MATLAB'den ölçüldü). İhlal edilen kısıt seçimi "en çok ihlal edilen"dir;
KWIK'in seçim kuralı yayımlanmadığından yol (dolayısıyla yineleme sayısı) bazen farklıdır — çözüm aynıdır.

Durum kodları (``status``, MATLAB ``mpcActiveSetSolver`` exitflag anlamıyla):

====== =====================================================================================
 > 0   en iyi çözüm bulundu; değer yineleme sayısıdır
   0   en çok yineleme sınırına ulaşıldı; x olursuz (MPC bu durumda önceki MV'yi tutar)
  −1   problem olursuz (çözümsüz): kısıtlar birlikte sağlanamaz
  −2   onarılamaz sayısal hata (H pozitif tanımlı değil / bozulma)
====== =====================================================================================

Algoritma (GI çift yöntem)
--------------------------
Notasyon: H = L Lᵀ (Cholesky), aktif küme 𝒜 (q kısıt, normalleri N = [n_i]), J = L⁻ᵀ Q ve L⁻¹N = Q [R; 0]
(R q×q üst üçgen). J = [J₁ J₂] (J₁ ilk q sütun). Böylece JᵀHJ = I, JJᵀ = H⁻¹ ve JᵀN = [R; 0].

0. x = −H⁻¹f (kısıtsız en iyi), 𝒜 = ∅, u = []. Eşitlik kısıtları tam adımla eklenir (1b). Sıcak başlangıç:
   önceki aktif kümedeki eşitsizlikler de tam adımla eklenir; ardından negatif çarpanlı olanlar (en negatifi önce)
   çıkarılır ve x, u kapalı biçimle yeniden hesaplanır (x = J₁R⁻ᵀb𝒜 − J₂J₂ᵀf, u = R⁻¹(R⁻ᵀb𝒜 + J₁ᵀf)) — böylece
   (x, 𝒜) bir "S-çifti" olur (𝒜 üzerinde en iyi, çarpanlar ≥ 0).
1. Aktif olmayan eşitsizliklerden en çok ihlal edileni seç: s_i = n_iᵀx − b_i, p = argmin s_i.
   s_p ≥ −tol ise dur: x en iyidir (KKT: Hx + f = N u, u ≥ 0, s ≥ −tol).
2. d = Jᵀn_p, z = J₂ d₂ (birincil yön), r = R⁻¹ d₁ (çift yön); z ᵀn_p = ‖d₂‖².
   t₁ = min{u_j / r_j : j eşitsizlik, r_j > DEP_TOL·max|r|} (çift olurluk sınırı, j = l; yuvarlama düzeyindeki
   r_j sayılmaz), t₂ = −s_p / ‖d₂‖² (‖d₂‖ ≤ DEP_TOL·‖d‖ ise ∞).
   t = min(t₁, t₂). t = ∞ → olursuz (−1). t₂ = ∞ → yalnız çift adım: u ← u − t r, u_p += t, l'yi çıkar, 2'ye dön.
   Aksi hâlde x ← x + t z, u ← u − t r, u_p += t; t = t₂ ise p'yi ekle ve 1'e dön, değilse l'yi çıkar ve 2'ye dön.
   Değişmez: her an Hx + f = N u (N aktif normaller), eşitsizlik çarpanları ≥ 0.
Ekleme: d₂'yi Givens yansımalarıyla (alttan yukarı) e₁ doğrultusuna döndür, J'nin q.. sütunlarına uygula;
R'nin yeni sütunu d[0..q]. Çıkarma: R'den l. sütun silinir, oluşan alt-köşegen Givens ile sıfırlanır, aynı dönüşüm
J'nin sütunlarına uygulanır. Doğrusal bağımlılık: ‖d₂‖ ≤ DEP_TOL·‖d‖.

C'ye taşıma (malloc'suz)
------------------------
Tüm bellek :class:`QPWorkspace` içindedir; boyutlar derleme anında sabit (N = en çok değişken, M = en çok kısıt):
J[N][N], R[N][N] (yalnız üst üçgen kullanılır), x, d, z, r, u [N], act [N] (int), s [M], iact [M] (bayt).
Toplam ≈ (2N² + 5N + M) double + N int + M bayt. Döngüler yalnız sabit üst sınırlı (``max_iter``). Her yineleme
O(N² + M·N) işlem: Jᵀn_p (N²), geri yerine koyma (q²/2), z (N(N−q)), ekleme/çıkarma Givens (≤ 2N²), ihlal taraması
(M·N). Kayan nokta: çift duyarlık önerilir; tek duyarlıkta DEP_TOL ≈ 1e6·eps_float alınmalıdır.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot, inf, sqrt

import numpy as np
from scipy.linalg import cholesky, solve_triangular

from spt.control.errors import ControlError

__all__ = ["QPResult", "QPWorkspace", "qp_solve", "qp_solve_ge", "mpcActiveSetSolver", "mpcActiveSetOptions",
           "hessian_factor", "STATUS_MAXITER", "STATUS_INFEASIBLE", "STATUS_NUMERICAL", "DEP_TOL"]

STATUS_MAXITER = 0
STATUS_INFEASIBLE = -1
STATUS_NUMERICAL = -2

#: Doğrusal bağımlılık eşiği (göreli): ‖J₂ᵀn_p‖ ≤ DEP_TOL·‖Jᵀn_p‖ ise n_p aktif normallerin açıklığında sayılır.
DEP_TOL = 1e6 * float(np.finfo(float).eps)


@dataclass
class QPResult:
    """QP sonucu.

    ``lam``: eşitsizlik çarpanları (m,), ≥ 0; :func:`qp_solve` (A x ≤ b) için H x + f + Aᵀλ + Aeqᵀμ = 0
    (quadprog işareti), :func:`qp_solve_ge` (Ac x ≥ bc) için H x + f = Acᵀλ + Aeqᵀν (KWIK işareti).
    ``lam_eq``: eşitlik çarpanları (q,) (sırasıyla μ ya da ν). ``active``: (m,) aktif eşitsizlikler (MATLAB iA).
    """
    x: np.ndarray
    status: int
    iterations: int
    active: np.ndarray
    lam: np.ndarray
    lam_eq: np.ndarray
    fval: float

    @property
    def exitflag(self) -> int:
        return self.status


@dataclass
class QPWorkspace:
    """Sabit boyutlu çalışma alanı (C'deki statik diziler). Aynı nesne tekrar tekrar kullanılabilir."""
    n_max: int
    m_max: int
    J: np.ndarray = field(init=False)
    R: np.ndarray = field(init=False)
    x: np.ndarray = field(init=False)
    d: np.ndarray = field(init=False)
    z: np.ndarray = field(init=False)
    r: np.ndarray = field(init=False)
    u: np.ndarray = field(init=False)
    act: np.ndarray = field(init=False)
    s: np.ndarray = field(init=False)
    iact: np.ndarray = field(init=False)

    def __post_init__(self):
        n, m = int(self.n_max), max(int(self.m_max), 1)
        self.J = np.zeros((n, n))
        self.R = np.zeros((n, n))
        self.x, self.d, self.z, self.r, self.u = (np.zeros(n) for _ in range(5))
        self.act = np.zeros(n, dtype=np.int64)
        self.s = np.zeros(m)
        self.iact = np.zeros(m, dtype=bool)

    def fits(self, n: int, m: int) -> bool:
        return n <= self.n_max and m <= max(self.m_max, 1)


def hessian_factor(H) -> np.ndarray:
    """Linv = L⁻¹ (H = L Lᵀ). H pozitif tanımlı değilse ControlError (MATLAB: 'Hessian is not positive definite')."""
    H = np.asarray(H, dtype=float)
    try:
        L = cholesky(0.5 * (H + H.T), lower=True, check_finite=True)
    except (np.linalg.LinAlgError, ValueError):
        raise ControlError("QP Hessian matrisi pozitif tanımlı değil.") from None
    if not np.all(np.diag(L) > 0.0):
        raise ControlError("QP Hessian matrisi pozitif tanımlı değil.")
    return solve_triangular(L, np.eye(H.shape[0]), lower=True)


def _gi_core(Linv, f, C, bvec, m, iA0, max_iter, tol, ws: QPWorkspace):
    """GI çekirdeği. C: (m+q)×n, ilk m satır eşitsizlik (C x ≥ b), son q satır eşitlik (C x = b).

    Döndürür: (x, status, iterations, iact (m+q,), lam (m+q,)) — lam KWIK işaretiyle (Hx + f = Cᵀlam)."""
    n = f.shape[0]
    mt = C.shape[0]
    J = ws.J[:n, :n]
    R = ws.R[:n, :n]
    x = ws.x[:n]
    d = ws.d[:n]
    z = ws.z[:n]
    r = ws.r[:n]
    u = ws.u[:n]
    act = ws.act[:n]
    iact = ws.iact[:mt] if mt else ws.iact[:0]
    iact[:] = False
    J[:, :] = Linv.T                     # J = L⁻ᵀ (Q = I, 𝒜 = ∅)
    R[:, :] = 0.0
    x[:] = -(J @ (J.T @ f))              # kısıtsız en iyi
    q = 0

    def back_r(qq):                      # r = R⁻¹ d₁ (üst üçgen geri yerine koyma)
        if qq:
            r[:qq] = solve_triangular(R[:qq, :qq], d[:qq], lower=False, check_finite=False)

    def add_column(qq):
        """d₂'yi Givens yansımalarıyla sıfırla (J'nin qq.. sütunlarına uygula); R'nin qq. sütunu = d[0..qq]."""
        for j in range(n - 1, qq, -1):
            a, b_ = d[j - 1], d[j]
            if b_ == 0.0:
                continue
            h = hypot(a, b_)
            c, s_ = a / h, b_ / h
            d[j - 1], d[j] = h, 0.0
            cj1 = J[:, j - 1].copy()
            cj = J[:, j]
            J[:, j - 1] = c * cj1 + s_ * cj
            J[:, j] = s_ * cj1 - c * cj
        R[:qq + 1, qq] = d[:qq + 1]

    def drop_column(l, qq):
        """Aktif kümeden l. elemanı çıkar (qq: çıkarmadan önceki boyut); yeni boyutu döndürür."""
        for i in range(l, qq - 1):
            R[:i + 2, i] = R[:i + 2, i + 1]
            u[i] = u[i + 1]
            act[i] = act[i + 1]
        R[:, qq - 1] = 0.0
        qn = qq - 1
        for j in range(l, qn):
            a, b_ = R[j, j], R[j + 1, j]
            if b_ == 0.0:
                continue
            h = hypot(a, b_)
            c, s_ = a / h, b_ / h
            R[j, j], R[j + 1, j] = h, 0.0
            if j + 1 < qn:
                t1 = R[j, j + 1:qn].copy()
                t2 = R[j + 1, j + 1:qn].copy()
                R[j, j + 1:qn] = c * t1 + s_ * t2
                R[j + 1, j + 1:qn] = s_ * t1 - c * t2
            cj = J[:, j].copy()
            cj1 = J[:, j + 1].copy()
            J[:, j] = c * cj + s_ * cj1
            J[:, j + 1] = s_ * cj - c * cj1
        return qn

    def force_add(i):
        """Tam adımla ekle (eşitlik / sıcak başlangıç). Bağımlıysa False."""
        nonlocal q
        if q >= n:
            return False
        npv = C[i]
        d[:] = J.T @ npv
        nd = sqrt(float(d @ d))
        nd2 = sqrt(float(d[q:] @ d[q:]))
        if nd == 0.0 or nd2 <= DEP_TOL * nd:
            return False
        back_r(q)
        z[:] = J[:, q:] @ d[q:]
        t = (bvec[i] - float(npv @ x)) / (nd2 * nd2)
        x[:] += t * z
        if q:
            u[:q] -= t * r[:q]
        add_column(q)
        u[q] = t
        act[q] = i
        iact[i] = True
        q += 1
        return True

    # --- eşitlik kısıtları (her zaman aktif, işaret serbest)
    for i in range(m, mt):
        if not force_add(i):
            if abs(float(C[i] @ x) - bvec[i]) > tol * (1.0 + abs(bvec[i])):
                return x.copy(), STATUS_INFEASIBLE, 0, iact.copy(), _lam_out(u, act, q, mt)
    # --- sıcak başlangıç: önceki aktif eşitsizlikler (negatif çarpanlı çıkarmalar yineleme sayılır — MATLAB gibi)
    iters = 1
    if iA0 is not None and m:
        added = False
        for i in np.flatnonzero(np.asarray(iA0[:m], dtype=bool)):
            added |= force_add(int(i))
        if added:
            while True:
                _recompute(J, R, x, u, act, q, f, bvec)
                worst, wpos = 0.0, -1
                for j in range(q):
                    if act[j] < m and u[j] < worst:
                        worst, wpos = u[j], j
                if wpos < 0:
                    break
                iact[act[wpos]] = False
                q = drop_column(wpos, q)
                iters += 1

    status = None
    s = ws.s[:m]
    while status is None:
        # 1. en çok ihlal edilen pasif eşitsizlik
        p = -1
        if m:
            s[:] = C[:m] @ x - bvec[:m]
            s[iact[:m]] = 0.0
            p = int(np.argmin(s))
            if s[p] >= -tol:
                p = -1
        if p < 0:
            status = iters
            break
        up = 0.0
        npv = C[p]
        # 2. adım döngüsü (aynı p için)
        while True:
            if not np.all(np.isfinite(x)):
                status = STATUS_NUMERICAL
                break
            d[:] = J.T @ npv
            back_r(q)
            nd = sqrt(float(d @ d))
            nd2 = sqrt(float(d[q:] @ d[q:])) if q < n else 0.0
            t1, l = inf, -1
            rtol = DEP_TOL * float(np.abs(r[:q]).max()) if q else 0.0   # yuvarlama düzeyindeki r_j > 0 sayılmaz
            for j in range(q):
                if act[j] < m and r[j] > rtol:
                    ratio = u[j] / r[j]
                    if ratio < t1:
                        t1, l = ratio, j
            if nd2 <= DEP_TOL * nd:
                t2 = inf
            else:
                sp = float(npv @ x) - bvec[p]
                t2 = -sp / (nd2 * nd2)
            if t1 == inf and t2 == inf:
                status = STATUS_INFEASIBLE
                break
            if iters >= max_iter:
                status = STATUS_MAXITER
                break
            iters += 1
            if t2 == inf:                     # yalnız çift adım
                if q:
                    u[:q] -= t1 * r[:q]
                up += t1
                iact[act[l]] = False
                q = drop_column(l, q)
                continue
            t = t2 if t2 <= t1 else t1
            z[:] = J[:, q:] @ d[q:]
            x[:] += t * z
            if q:
                u[:q] -= t * r[:q]
            up += t
            if t2 <= t1:                      # tam adım: p eklenir
                add_column(q)
                u[q] = up
                act[q] = p
                iact[p] = True
                q += 1
                break
            iact[act[l]] = False              # kısmi adım: l çıkarılır
            q = drop_column(l, q)
    return x.copy(), int(status), iters, iact.copy(), _lam_out(u, act, q, mt)


def _recompute(J, R, x, u, act, q, f, bvec):
    """𝒜 üzerinde kapalı biçim: y₁ = R⁻ᵀ b𝒜, x = J₁y₁ − J₂J₂ᵀf, u = R⁻¹(y₁ + J₁ᵀf)."""
    if q == 0:
        x[:] = -(J @ (J.T @ f))
        return
    bA = bvec[act[:q]]
    y1 = solve_triangular(R[:q, :q], bA, trans="T", lower=False, check_finite=False)
    J1, J2 = J[:, :q], J[:, q:]
    x[:] = J1 @ y1 - J2 @ (J2.T @ f)
    u[:q] = solve_triangular(R[:q, :q], y1 + J1.T @ f, lower=False, check_finite=False)


def _lam_out(u, act, q, mt):
    lam = np.zeros(mt)
    for j in range(q):
        lam[act[j]] = u[j]
    return lam


def _as2d(A, n, name):
    if A is None:
        return np.zeros((0, n))
    A = np.asarray(A, dtype=float)
    if A.size == 0:
        return np.zeros((0, n))
    A = np.atleast_2d(A)
    if A.shape[1] != n:
        raise ControlError(f"{name} matrisinin sütun sayısı {n} olmalı.")
    return A


def qp_solve_ge(H, f, Ac=None, bc=None, Aeq=None, beq=None, *, iA0=None, max_iter=200, tol=1e-6, Linv=None,
                workspace: QPWorkspace | None = None) -> QPResult:
    """min ½xᵀHx + fᵀx, Ac x ≥ bc, Aeq x = beq (MPC iç biçimi; MATLAB qpkwik). Bkz. modül belgesi.

    ``Linv`` verilirse H yerine kullanılır (L⁻¹, H = LLᵀ; MPC'de bir kez hesaplanır). ``iA0``: (m,) bool sıcak
    başlangıç. ``max_iter``: MATLAB sayımıyla yineleme sınırı. ``tol``: kısıt ihlali toleransı (mutlak)."""
    f = np.asarray(f, dtype=float).ravel()
    n = f.size
    if n == 0:
        raise ControlError("QP en az bir karar değişkeni içermeli.")
    if Linv is None:
        Linv = hessian_factor(H)
    else:
        Linv = np.asarray(Linv, dtype=float)
    if H is None:                         # H = L Lᵀ, L = Linv⁻¹
        Lf = solve_triangular(Linv, np.eye(n), lower=True)
        Hm = Lf @ Lf.T
    else:
        Hm = np.asarray(H, dtype=float)
    Ac = _as2d(Ac, n, "Eşitsizlik")
    m = Ac.shape[0]
    bc = np.zeros(0) if m == 0 else np.asarray(bc, dtype=float).ravel()
    Aeq = _as2d(Aeq, n, "Eşitlik")
    q = Aeq.shape[0]
    beq = np.zeros(0) if q == 0 else np.asarray(beq, dtype=float).ravel()
    if bc.size != m or beq.size != q:
        raise ControlError("Kısıt sağ taraf vektörlerinin boyu kısıt sayısıyla uyuşmuyor.")
    C = np.vstack([Ac, Aeq]) if q else Ac
    bvec = np.concatenate([bc, beq]) if q else bc
    ws = workspace if (workspace is not None and workspace.fits(n, m + q)) else QPWorkspace(n, m + q)
    if iA0 is not None:
        iA0 = np.asarray(iA0, dtype=bool).ravel()
        if iA0.size != m:
            raise ControlError("Sıcak başlangıç aktif küme vektörünün boyu eşitsizlik sayısıyla uyuşmuyor.")
    x, status, iters, iact, lam = _gi_core(Linv, f, C, bvec, m, iA0, int(max_iter), float(tol), ws)
    fval = float(0.5 * x @ Hm @ x + f @ x)
    return QPResult(x=x, status=status, iterations=iters, active=iact[:m].copy(), lam=lam[:m].copy(),
                    lam_eq=lam[m:].copy(), fval=fval)


def qp_solve(H, f, A=None, b=None, Aeq=None, beq=None, *, iA0=None, max_iter=200, tol=1e-6, Linv=None,
             workspace: QPWorkspace | None = None) -> QPResult:
    """min ½xᵀHx + fᵀx, A x ≤ b, Aeq x = beq (MATLAB quadprog / mpcActiveSetSolver biçimi).

    Çarpanlar quadprog işaretiyle: H x + f + Aᵀλ + Aeqᵀμ = 0, λ ≥ 0."""
    f = np.asarray(f, dtype=float).ravel()
    n = f.size
    A = _as2d(A, n, "Eşitsizlik")
    b = np.zeros(0) if A.shape[0] == 0 else np.asarray(b, dtype=float).ravel()
    res = qp_solve_ge(H, f, -A, -b, Aeq, beq, iA0=iA0, max_iter=max_iter, tol=tol, Linv=Linv, workspace=workspace)
    res.lam_eq = -res.lam_eq
    return res


def mpcActiveSetOptions(**kw) -> dict:
    """MATLAB ``mpcActiveSetOptions`` varsayılanları: MaxIterations 200, ConstraintTolerance 1e-6,
    UseHessianAsInput True (False ise ilk girdi Linv = L⁻¹)."""
    opts = {"MaxIterations": 200, "ConstraintTolerance": 1e-6, "UseHessianAsInput": True, "IntegrityChecks": True}
    for k, v in kw.items():
        if k not in opts:
            raise ControlError(f"Bilinmeyen çözücü seçeneği: {k}")
        opts[k] = v
    return opts


def _rank_matlab(M) -> int:
    """MATLAB ``rank``: tekil değer > max(size)·eps(max σ)."""
    if M.size == 0:
        return 0
    s = np.linalg.svd(M, compute_uv=False)
    return int(np.sum(s > max(M.shape) * np.spacing(s.max())))


def mpcActiveSetSolver(H, f, A, b, Aeq, beq, iA0, options: dict | None = None):
    """MATLAB ``[x, exitflag, iA, lambda] = mpcActiveSetSolver(H, f, A, b, Aeq, beq, iA0, options)`` karşılığı.

    A x ≤ b, Aeq x = beq. ``lambda`` sözlüğü: ``ineqlin`` (≥ 0), ``eqlin`` (KWIK işareti, quadprog'unkinin tersi —
    MATLAB'den ölçüldü). ``IntegrityChecks`` (varsayılan açık) MATLAB'in girdi denetimleriyle aynıdır: eşitlik sayısı
    n'den fazla (OverSpecEquaCons), rank(Aeq) < q (AeqRank) ve iA0'daki aktif sayısı + q > n (InvalidWarmStart) açık
    hatadır."""
    opts = mpcActiveSetOptions() if options is None else {**mpcActiveSetOptions(), **options}
    f = np.asarray(f, dtype=float).ravel()
    n = f.size
    if opts["IntegrityChecks"]:
        Ae = _as2d(Aeq, n, "Eşitlik")
        q = Ae.shape[0]
        if q > n:
            raise ControlError(f"Eşitlik kısıtı sayısı ({q}) değişken sayısından ({n}) fazla: QP çözümsüz.")
        if q and _rank_matlab(Ae) < q:
            raise ControlError(f"Eşitlik kısıtları doğrusal bağımsız değil: rank(Aeq) = {_rank_matlab(Ae)} < {q}.")
        nact = int(np.count_nonzero(np.asarray(iA0, dtype=bool))) if iA0 is not None else 0
        if nact + q > n:
            raise ControlError(f"Sıcak başlangıç aktif kümesinde {nact} eşitsizlik + {q} eşitlik = {nact + q} aktif "
                               f"kısıt değişken sayısını ({n}) aşıyor: QP çözümsüz.")
    if opts["UseHessianAsInput"]:
        res = qp_solve(H, f, A, b, Aeq, beq, iA0=iA0, max_iter=opts["MaxIterations"],
                       tol=opts["ConstraintTolerance"])
    else:
        Linv = np.asarray(H, dtype=float)
        if not (np.allclose(Linv, np.tril(Linv)) and np.all(np.diag(Linv) > 0)):
            raise ControlError("QP Hessian matrisi pozitif tanımlı değil.")
        res = qp_solve(None, f, A, b, Aeq, beq, iA0=iA0, max_iter=opts["MaxIterations"],
                       tol=opts["ConstraintTolerance"], Linv=Linv)
    m = _as2d(A, n, "Eşitsizlik").shape[0]
    iA = res.active.copy() if m else np.zeros(0, dtype=bool)
    lam = {"ineqlin": res.lam.copy(), "eqlin": -res.lam_eq}
    return res.x, int(res.status), iA, lam
