"""Parametrik doğrusal polinom modeller: `arx`, `armax`, `oe`, `bj`, `polyest`, `ar` (MATLAB anlamıyla).

**ARX** (en küçük kareler; MATLAB ile ~1e-10): R2025b'de ölçülen kural — regresyon, model durum boyutu
nx = max(na, max_j(nb_j + nk_j − 1 − d_j)) (d_j = max(nk_j − 1, 0) saf giriş gecikmesi) kadar ilk örneği atar
(bu örneklerin öngörü hataları tahmin edilen başlangıç durumuyla tam sıfırlanır: `InitialCondition='estimate'`);
saf gecikme tamponu sıfırla başlar. LossFcn = Σe²/N (N tüm örnekler), d = parametre + nx.
MATLAB `arx` 'zero' seçeneğini yok sayar (ölçüldü: aynı sonuç); bizde `initial_condition='zero'` gerçekten sıfır
başlangıçlı (sıfır dolgulu) regresyondur.

**ARMAX / OE / BJ / polyest** (öngörü hatası yöntemi, PEM): ölçüt V(θ) = Σ e(t,θ)², e = (D/C)(A y − Σ B_j/F_j u_j).
Analitik Jacobian (süzülmüş regresörler) ile Levenberg–Marquardt, sıkı yakınsamaya kadar. Başlangıç koşulu:
'zero' (geçmiş veri sıfır), 'estimate' (başlangıç durumu ek parametre: öngörücünün serbest cevabı
(1/(C·ΠF))·p, deg p < nx — modelin yenilik biçimindeki durum boyutu) ya da 'auto' (MATLAB AutoInitThreshold
kuralı: sıfır ve tahmin edilen başlangıçla öngörü hatası normlarının oranı > 1.05 ise 'estimate').
Başlangıç tahminleri: ARMAX — yüksek dereceli ARX + Hannan–Rissanen; OE — Steiglitz–McBride; BJ — OE + ARMA.
MATLAB'in varsayılan durdurma ölçütü (Tolerance 0.01, en çok 20 yineleme) optimumdan önce durabilir; biz optimuma
kadar yineleriz (MATLAB'le karşılaştırma testleri sıkı toleranslı MATLAB seçenekleriyle yapılır).
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from spt.control.errors import ControlError
from spt.control.sysid._util import fd_jacobian, is_stable, levenberg_marquardt, pmul, shift, stabilize
from spt.control.sysid.data import IDData, check_complete
from spt.control.sysid.models import IDPoly, Report, make_fit

__all__ = ["arx", "armax", "oe", "bj", "polyest", "ar"]

_IC = ("auto", "zero", "estimate", "backcast")


# ============================================================================================ yardımcılar
def _int_mat(x, shape, what):
    a = np.asarray(x)
    if a.size == 1:
        a = np.full(shape, int(a.ravel()[0]))
    a = np.asarray(a, dtype=float)
    if a.size != int(np.prod(shape)):
        raise ControlError(f"{what} {shape} boyutlu olmalı")
    if np.any(a < 0) or np.any(a != np.round(a)):
        raise ControlError(f"{what} negatif olmayan tamsayı olmalı")
    return a.astype(int).reshape(shape)


def _arx_orders(orders, ny, nu):
    """[na nb nk] → na (ny,ny), nb (ny,nu), nk (ny,nu). MATLAB biçimleri: tek çıkışta [na, nb..., nk...];
    çok çıkışta ([na], [nb], [nk]) matrisleri (demet / liste)."""
    if isinstance(orders, (tuple, list)) and len(orders) == 3 and any(np.ndim(o) >= 1 for o in orders):
        def mat(x, shape, what):
            a = np.asarray(x, float)
            if a.ndim == 1 and shape[0] > 1 and a.size == shape[1]:
                a = np.tile(a, (shape[0], 1))  # giriş başına vektör → her çıkış için aynı
            return _int_mat(a, shape, what)
        return mat(orders[0], (ny, ny), "na"), mat(orders[1], (ny, nu), "nb"), mat(orders[2], (ny, nu), "nk")
    o = np.atleast_1d(np.asarray(orders, float)).ravel()
    if ny > 1:
        if len(o) == 3:
            return (_int_mat(o[0], (ny, ny), "na"), _int_mat(o[1], (ny, nu), "nb"), _int_mat(o[2], (ny, nu), "nk"))
        if len(o) != 1 + 2 * nu:
            raise ControlError("Çok çıkışlı ARX için derece (na, nb, nk) matrisleri ya da skalerler olmalı")
        na = _int_mat(o[0], (ny, ny), "na")
        nb = _int_mat(np.tile(o[1:1 + nu], (ny, 1)), (ny, nu), "nb")
        nk = _int_mat(np.tile(o[1 + nu:], (ny, 1)), (ny, nu), "nk")
        return na, nb, nk
    if len(o) != 1 + 2 * nu:
        raise ControlError(f"ARX derecesi [na, nb(×{nu}), nk(×{nu})] olmalı ({1 + 2 * nu} sayı)")
    return (_int_mat(o[0], (1, 1), "na"), _int_mat(o[1:1 + nu], (1, nu), "nb"),
            _int_mat(o[1 + nu:], (1, nu), "nk"))


def _delays_and_n(na, nb, nk):
    """Saf giriş gecikmeleri d_j ve durum ufku n (atılan satır sayısı)."""
    ny, nu = nb.shape
    d = np.maximum(nk.min(axis=0) - 1, 0) if nu else np.zeros(0, int)
    degs = [int(na.max())] if na.size else [0]
    for i in range(ny):
        for j in range(nu):
            if nb[i, j] > 0:
                degs.append(int(nk[i, j] + nb[i, j] - 1 - d[j]))
    return d, max(degs)


def _arx_regression(data: IDData, i: int, na, nb, nk, start: str = "estimate"):
    """i. çıkış için ARX regresyonu (deneyler alt alta). start: 'estimate' (ilk n satırı at) ya da 'zero'."""
    ny, nu = data.ny, data.nu
    _, n = _delays_and_n(na, nb, nk)
    rows, Ys = [], []
    for y, u in zip(data.ys, data.us):
        N = len(y)
        t0 = n if start == "estimate" else 0
        if t0 >= N:
            raise ControlError("Veri, model derecesine göre çok kısa")
        cols = []
        for j in range(ny):
            for k in range(1, na[i, j] + 1):
                cols.append(-shift(y[:, j], k)[t0:])
        for j in range(nu):
            for k in range(nb[i, j]):
                cols.append(shift(u[:, j], nk[i, j] + k)[t0:])
        rows.append(np.column_stack(cols) if cols else np.zeros((N - t0, 0)))
        Ys.append(y[t0:, i])
    return np.vstack(rows), np.concatenate(Ys)


def _prep(data, what="Tahmin"):
    if not isinstance(data, IDData):
        raise ControlError("Veri IDData olmalı")
    check_complete(data, what)
    return data


def _names_kw(data):
    return dict(input_names=data.input_names, output_names=data.output_names)


# ============================================================================================ ARX
def arx(data: IDData, orders, *, initial_condition: str = "estimate", regularization=None) -> IDPoly:
    """ARX modeli (MATLAB `arx(data, [na nb nk])`) — doğrusal en küçük kareler.

    Tek çıkış: `orders = [na, nb1..nbnu, nk1..nknu]`; zaman serisinde `[na]` (AR). Çok çıkış: `(na, nb, nk)`
    matrisleri ya da skaler dereceler (tam A matrisi). `initial_condition`: 'estimate' (varsayılan; MATLAB) ya da
    'zero'. `regularization=(lam, R)`: θ = argmin ‖Y − Φθ‖² + lam·θᵀRθ (MATLAB Regularization.Lambda / R)."""
    data = _prep(data)
    if initial_condition not in ("estimate", "zero", "auto"):
        raise ControlError("Başlangıç koşulu 'estimate' ya da 'zero' olmalı")
    start = "zero" if initial_condition == "zero" else "estimate"
    ny, nu = data.ny, data.nu
    if nu == 0 and ny == 1 and np.size(orders) == 1:
        orders = [int(np.ravel(orders)[0])]
        na = _int_mat(orders[0], (1, 1), "na")
        nb = np.zeros((1, 0), int)
        nk = np.zeros((1, 0), int)
    else:
        na, nb, nk = _arx_orders(orders, ny, nu)
    d, n = _delays_and_n(na, nb, nk)
    thetas, E_rows = [], []
    Phis = []
    for i in range(ny):
        Phi, Y = _arx_regression(data, i, na, nb, nk, start)
        if Phi.shape[1] == 0:
            th = np.zeros(0)
        elif regularization is not None:
            lam, R = regularization
            R = np.eye(Phi.shape[1]) if R is None else np.asarray(R, float)
            th = np.linalg.solve(Phi.T @ Phi + lam * R, Phi.T @ Y)
        else:
            th, *_ = np.linalg.lstsq(Phi, Y, rcond=None)
        thetas.append(th)
        E_rows.append(Y - Phi @ th)
        Phis.append(Phi)
    # Artıklar: atılan satırlar (başlangıç durumuyla) sıfır hatalıdır
    t0 = n if start == "estimate" else 0
    e_list, y_list = [], []
    k = 0
    for y in data.ys:
        N = len(y)
        e = np.zeros((N, ny))
        for i in range(ny):
            e[t0:, i] = E_rows[i][k:k + N - t0]
        k += N - t0
        e_list.append(e)
        y_list.append(y)
    npar = sum(len(t) for t in thetas)
    nx0 = (n * ny if start == "estimate" else 0) * data.ne
    Ntot = sum(data.lengths)
    fit = make_fit(y_list, e_list, npar + nx0, loss="trace")
    E = np.vstack(e_list)
    lam = E.T @ E / (Ntot - npar - nx0)  # MATLAB: d = tüm parametreler + başlangıç durumları (çok çıkışta da)
    # Model
    A = [[None] * ny for _ in range(ny)]
    B = [[None] * nu for _ in range(ny)]
    for i in range(ny):
        th = thetas[i]
        p = 0
        for j in range(ny):
            m = na[i, j]
            A[i][j] = np.r_[1.0 if i == j else 0.0, th[p:p + m]]
            p += m
        for j in range(nu):
            m = nb[i, j]
            B[i][j] = np.r_[np.zeros(nk[i, j]), th[p:p + m]]
            p += m
    kw = _names_kw(data)
    if ny == 1:
        model = IDPoly(A[0][0], B[0] if nu > 1 else (B[0][0] if nu else None), Ts=data.Ts, nk=nk,
                       noise_variance=lam, **kw)
    else:
        model = IDPoly(A, B, Ts=data.Ts, nk=nk, noise_variance=lam, **kw)
    # Kovaryans: blok köşegen λ_ii (ΦᵢᵀΦᵢ)⁻¹ (çıkışlar arası çapraz terimler λ_ij (ΦᵢᵀΦᵢ)⁻¹ΦᵢᵀΦⱼ(ΦⱼᵀΦⱼ)⁻¹)
    blocks = []
    for i in range(ny):
        P = Phis[i]
        if regularization is not None:
            lamr, R = regularization
            R = np.eye(P.shape[1]) if R is None else np.asarray(R, float)
            M = np.linalg.inv(P.T @ P + lamr * R)
            blocks.append(M @ (P.T @ P) @ M)
        else:
            blocks.append(np.linalg.pinv(P.T @ P) if P.shape[1] else np.zeros((0, 0)))
    # MATLAB: kovaryans ölçeği EᵀE/(N − np − n·ne) (n: atılan satır; çok çıkışta NoiseVariance'tan farklı olarak n·ny
    # değil n — R2025b'de ölçüldü)
    lamc = E.T @ E / (Ntot - npar - (n if start == "estimate" else 0) * data.ne)
    if ny == 1:
        cov = lamc[0, 0] * blocks[0]
    else:
        sizes = [b.shape[0] for b in blocks]
        cov = np.zeros((sum(sizes), sum(sizes)))
        oi = np.cumsum([0] + sizes)
        for i in range(ny):
            for j in range(ny):
                Pi, Pj = Phis[i], Phis[j]
                if len(Pi) == len(Pj):
                    cov[oi[i]:oi[i + 1], oi[j]:oi[j + 1]] = lamc[i, j] * blocks[i] @ (Pi.T @ Pj) @ blocks[j]
    model.covariance = cov
    model.report = Report("ARX", fit, "estimate" if start == "estimate" else "zero", 0, "Doğrudan çözüm",
                          npar, nx0, None, data.name, Ntot, data.Ts,
                          {"orders": (na.tolist(), nb.tolist(), nk.tolist())},
                          f"ARX ile tahmin edildi (en küçük kareler), uyum %{fit.FitPercent.round(2).tolist()}")
    return model


# ============================================================================================ PEM motoru
class _PolyStructure:
    """Tek çıkışlı genel polinom yapısı: na, nb[j], nc, nd, nf[j], nk[j]."""

    def __init__(self, na, nb, nc, nd, nf, nk):
        self.na, self.nc, self.nd = int(na), int(nc), int(nd)
        self.nb = np.asarray(nb, int).ravel()
        self.nf = np.asarray(nf, int).ravel()
        self.nk = np.asarray(nk, int).ravel()
        self.nu = len(self.nb)
        if len(self.nf) != self.nu or len(self.nk) != self.nu:
            raise ControlError("nb, nf ve nk giriş sayısı kadar olmalı")
        if np.any(np.r_[self.na, self.nb, self.nc, self.nd, self.nf, self.nk] < 0):
            raise ControlError("Dereceler negatif olamaz")
        self.np = self.na + int(self.nb.sum()) + self.nc + self.nd + int(self.nf.sum())

    def unpack(self, th):
        k = 0
        A = np.r_[1.0, th[k:k + self.na]]
        k += self.na
        B = []
        for j in range(self.nu):
            B.append(np.r_[np.zeros(self.nk[j]), th[k:k + self.nb[j]]])
            k += self.nb[j]
        C = np.r_[1.0, th[k:k + self.nc]]
        k += self.nc
        D = np.r_[1.0, th[k:k + self.nd]]
        k += self.nd
        F = []
        for j in range(self.nu):
            F.append(np.r_[1.0, th[k:k + self.nf[j]]])
            k += self.nf[j]
        return A, B, C, D, F

    def pack(self, A, B, C, D, F):
        p = list(np.asarray(A, float)[1:1 + self.na])
        for j in range(self.nu):
            b = np.asarray(B[j], float)
            p.extend(np.pad(b, (0, max(0, self.nk[j] + self.nb[j] - len(b))))[self.nk[j]:self.nk[j] + self.nb[j]])
        p.extend(np.pad(np.asarray(C, float), (0, self.nc + 1))[1:1 + self.nc])
        p.extend(np.pad(np.asarray(D, float), (0, self.nd + 1))[1:1 + self.nd])
        for j in range(self.nu):
            p.extend(np.pad(np.asarray(F[j], float), (0, self.nf[j] + 1))[1:1 + self.nf[j]])
        return np.array(p, float)

    def nx(self, th=None, split: bool = True) -> int:
        """Yenilik biçimi durum boyutu. `split=True`: nk−1 kadar giriş gecikmesi saf gecikme (durum değil; MATLAB
        idpoly → idss), `split=False`: gecikmeler de durum (MATLAB polyest 'estimate' kestirimi böyle çalışır)."""
        d = (np.maximum(self.nk - 1, 0) if split else np.zeros(self.nu, int)) if self.nu else np.zeros(0, int)
        degs = [self.na + self.nd + int(self.nf.sum()), self.nc + int(self.nf.sum())]
        for j in range(self.nu):
            if self.nb[j] > 0:
                degs.append(int(self.nk[j] + self.nb[j] - 1 - d[j]) + self.nd + int(self.nf.sum()) - self.nf[j])
        return max(degs)

    def to_model(self, th, Ts, nv, names) -> IDPoly:
        A, B, C, D, F = self.unpack(th)
        Bm = B if self.nu > 1 else (B[0] if self.nu else None)
        Fm = F if self.nu > 1 else (F[0] if self.nu else None)
        return IDPoly(A, Bm, C, D, Fm, Ts, nk=self.nk.reshape(1, -1), noise_variance=nv, **names)


class _PEM:
    """Tek çıkışlı polinom modeller için öngörü hatası ölçütü, analitik Jacobian ve başlangıç durumları."""

    def __init__(self, st: _PolyStructure, exps, ic: str, split: bool = False):
        self.st = st
        self.exps = exps  # [(y (N,), U (N, nu))]
        self.ic = ic
        self.nx = st.nx(None, split)
        self.nsx = self.nx if ic == "estimate" else 0

    def split(self, z):
        th = z[:self.st.np]
        s = z[self.st.np:].reshape(len(self.exps), self.nsx) if self.nsx else None
        return th, s

    def accept(self, z) -> bool:
        A, B, C, D, F = self.st.unpack(z[:self.st.np])
        return is_stable(C) and all(is_stable(f) for f in F)

    def _parts(self, th, s, k):
        y, U = self.exps[k]
        A, B, C, D, F = self.st.unpack(th)
        w = [signal.lfilter(B[j], F[j], U[:, j]) for j in range(self.st.nu)]
        v = signal.lfilter(A, [1.0], y) - (np.sum(w, axis=0) if w else 0.0)
        e = signal.lfilter(D, C, v)
        es = None
        if s is not None:
            Nc = pmul(C, *F) if F else C
            imp = np.zeros(len(y))
            imp[:self.nx] = s[k]
            es = signal.lfilter([1.0], Nc, imp)
            e = e + es
        return y, U, A, B, C, D, F, w, v, e, es

    def resid(self, z):
        th, s = self.split(z)
        return np.concatenate([self._parts(th, s, k)[9] for k in range(len(self.exps))])

    def jac(self, z):
        th, s = self.split(z)
        st = self.st
        blocks = []
        for k in range(len(self.exps)):
            y, U, A, B, C, D, F, w, v, e, es = self._parts(th, s, k)
            N = len(y)
            cols = []
            if st.na:
                yf = signal.lfilter(D, C, y)
                cols += [shift(yf, i) for i in range(1, st.na + 1)]
            for j in range(st.nu):
                if st.nb[j]:
                    uf = signal.lfilter(D, pmul(C, F[j]), U[:, j])
                    cols += [-shift(uf, st.nk[j] + i) for i in range(st.nb[j])]
            if st.nc:
                ef = signal.lfilter([1.0], C, e)
                cols += [-shift(ef, i) for i in range(1, st.nc + 1)]
            if st.nd:
                vf = signal.lfilter([1.0], C, v)
                cols += [shift(vf, i) for i in range(1, st.nd + 1)]
            for j in range(st.nu):
                if st.nf[j]:
                    g = signal.lfilter(D, C, w[j])
                    if es is not None:
                        g = g - es
                    g = signal.lfilter([1.0], F[j], g)
                    cols += [shift(g, i) for i in range(1, st.nf[j] + 1)]
            J = np.column_stack(cols) if cols else np.zeros((N, 0))
            if s is not None:
                Nc = pmul(C, *F) if F else C
                h = signal.lfilter([1.0], Nc, np.r_[1.0, np.zeros(N - 1)])
                Js = np.zeros((N, self.nsx * len(self.exps)))
                for i in range(self.nsx):
                    Js[:, k * self.nsx + i] = shift(h, i)
                J = np.hstack([J, Js])
            blocks.append(J)
        return np.vstack(blocks)

    def best_s(self, th):
        """Sabit θ için başlangıç durumlarının en küçük kareler çözümü."""
        if not self.nsx:
            return np.zeros(0)
        z0 = np.r_[th, np.zeros(self.nsx * len(self.exps))]
        r0 = self.resid(z0)
        J = self.jac(z0)[:, self.st.np:]
        sol, *_ = np.linalg.lstsq(J, -r0, rcond=None)
        return sol

    def fit(self, th0, max_iter=200):
        z0 = np.r_[th0, self.best_s(th0)] if self.nsx else np.asarray(th0, float)
        res = levenberg_marquardt(self.resid, z0, self.jac, accept=self.accept, max_iter=max_iter)
        return res


class _PEMBackcast:
    """'backcast' başlangıç koşulu (MATLAB polyest): birleşik polinomlarla w = Ā y − Σ B̄_j u_j (t ≥ tstart),
    1/C̄ süzgecinin başlangıç durumu w'nin geriye süzülmesinden; e = (1/C̄) w bu başlangıçla. Ā = A·ΠF·D,
    B̄_j = B_j·D·Π_{k≠j}F_k, C̄ = C·ΠF; tstart = 1 + max(deg Ā, nb̄_j − [nk_j = 0] − 1) (MATLAB tanımı).
    Ölçüt Σe²/N_toplam; Jacobian merkezi farklarla."""

    def __init__(self, st: _PolyStructure, exps):
        self.st = st
        self.exps = exps
        self.ic = "backcast"
        self.nsx = 0
        self.nx = st.nx(None, False)

    def split(self, z):
        return z[:self.st.np], None

    def accept(self, z) -> bool:
        A, B, C, D, F = self.st.unpack(z[:self.st.np])
        return is_stable(C) and all(is_stable(f) for f in F)

    def _combined(self, th):
        st = self.st
        A, B, C, D, F = st.unpack(th)
        if int(st.nf.sum()) == 0 and st.nd == 0:
            return A, B, C
        ff = pmul(*F) if F else np.array([1.0])
        afd = pmul(A, ff, D)
        cf = pmul(ff, C)
        bfd = [pmul(B[j], D, *[F[k] for k in range(st.nu) if k != j]) for j in range(st.nu)]
        return afd, bfd, cf

    def resid(self, z):
        th = z[:self.st.np]
        a, b, c = self._combined(th)
        st = self.st
        na = len(a) - 1
        nbc = [int(st.nb[j] + st.nd + int(st.nf.sum()) - st.nf[j]) for j in range(st.nu)]
        tstart = 1 + max([na] + [nbc[j] - (1 if st.nk[j] == 0 else 0) - 1 for j in range(st.nu)])
        nc = len(c) - 1
        out = []
        for y, U in self.exps:
            w = signal.lfilter(a, [1.0], y)
            for j in range(st.nu):
                w = w - signal.lfilter(b[j], [1.0], U[:, j])
            w = w[tstart - 1:]
            L = len(w)
            if nc == 0:
                out.append(w / c[0])
                continue
            if L < nc:
                raise ControlError("Veri, model derecesine göre çok kısa")
            ecb = signal.lfilter([1.0], c, w[::-1])
            tail = ecb[np.arange(L - 1, L - nc - 1, -1)]
            crev = c[:0:-1]
            wc = signal.lfilter(crev, [1.0], tail)
            wc = signal.lfilter(c, [1.0], np.zeros(nc), zi=wc[::-1])[0]
            ei = signal.lfilter([1.0], c, wc[::-1])
            e1 = signal.lfilter(crev, [1.0], ei[::-1])
            e = signal.lfilter([1.0], c, w, zi=-e1[::-1])[0]
            out.append(e)
        return np.concatenate(out)

    def jac(self, z):
        return fd_jacobian(self.resid, z, central=True)

    def fit(self, th0, max_iter=200):
        return levenberg_marquardt(self.resid, np.asarray(th0, float), self.jac, accept=self.accept,
                                   max_iter=max_iter)


def _auto_ic(st, exps, th0, threshold=1.05) -> str:
    """MATLAB polyest 'auto' kuralı (R2025b): çok deneyde 'backcast'; tek deneyde başlangıç modelinin sıfır ve
    tahmin edilen başlangıç durumlu öngörü hatası normlarının oranı > AutoInitThreshold (1.05) ise 'backcast',
    değilse 'zero'."""
    if len(exps) > 1:
        return "backcast"
    pz = _PEM(st, exps, "zero")
    pe_ = _PEM(st, exps, "estimate", split=True)
    ez = pz.resid(np.asarray(th0, float))
    s = pe_.best_s(th0)
    ee = pe_.resid(np.r_[th0, s])
    nz, ne = np.linalg.norm(ez), np.linalg.norm(ee)
    return "backcast" if ne > 0 and nz / ne > threshold else "zero"


def _finish(st: _PolyStructure, pem: _PEM, res, data: IDData, method: str, ic: str, opts: dict) -> IDPoly:
    th, s = pem.split(res.x)
    exps = pem.exps
    y_list = [y.reshape(-1, 1) for y, _ in exps]
    Ntot = sum(len(y) for y, _ in exps)
    r = pem.resid(res.x)
    V_est = float(r @ r) / Ntot  # MATLAB LossFcn: kestirim ölçütü / toplam örnek sayısı
    e_list = []
    if len(r) == Ntot:
        off = 0
        for y, _ in exps:
            e_list.append(r[off:off + len(y)].reshape(-1, 1))
            off += len(y)
    npar = st.np
    nx0 = pem.nsx * len(exps)
    if pem.ic == "backcast" or (pem.nsx and st.nx(None, True) != pem.nx):
        # MATLAB: kestirim ölçütü (LossFcn) gecikme durumları dahil başlangıçla; diğer ölçütler ve NoiseVariance,
        # gecikmesi ayrılmış gerçeklemede yeniden kestirilen başlangıç durumuyla (R2025b oe [2 2 2]'de ölçüldü).
        rep = _PEM(st, exps, "estimate", split=True)
        s_split = rep.best_s(th)
        r2 = rep.resid(np.r_[th, s_split])
        e_list = [r2[o:o + len(y)].reshape(-1, 1) for o, (y, _) in zip(np.cumsum([0] + [len(y) for y, _ in exps]),
                                                                          exps)]
        nx0 = rep.nsx * len(exps)
        s = s_split.reshape(len(exps), rep.nsx)
    fit = make_fit(y_list, e_list, npar + nx0)
    E = np.vstack(e_list)
    lam = float(E[:, 0] @ E[:, 0]) / (Ntot - npar - nx0)
    fit.LossFcn = V_est
    model = st.to_model(th, data.Ts, lam, _names_kw(data))
    # Kovaryans: λ·(J_θᵀJ_θ)⁻¹, J_θ = ∂e/∂θ (tahmin edilen başlangıç durumu sabit) — MATLAB ARMAX / BJ ile 1e-8.
    # Gürültü modelsiz yapılarda (OE) MATLAB, artıklara n4sid ile 5. dereceden zaman serisi modeli uydurup
    # gradyanı onunla süzerek renkli artık düzeltmesi yapar (yayımlanmamış n4sid ayrıntısına bağlı); burada
    # yapılmaz → OE kovaryansı MATLAB'den ~%1 farklı olabilir.
    J = pem.jac(res.x)[:, :npar]
    try:
        model.covariance = lam * np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        model.covariance = None
    x0 = [s[k].copy() for k in range(len(exps))] if s is not None else None
    model.report = Report(method, fit, ic, res.iterations, res.why, npar, nx0, x0, data.name, Ntot, data.Ts, opts,
                          f"{method} ile tahmin edildi (öngörü hatası yöntemi), uyum %"
                          f"{fit.FitPercent.round(2).tolist()}")
    return model


def _exps(data):
    return [(y[:, 0].copy(), u.copy()) for y, u in zip(data.ys, data.us)]


def _run_pem(data, st, inits, ic, method, max_iter, opts):
    if ic not in _IC:
        raise ControlError("Başlangıç koşulu 'auto', 'zero', 'estimate' ya da 'backcast' olmalı")
    exps = _exps(data)
    N = sum(len(y) for y, _ in exps)
    if N <= st.np + 1:
        raise ControlError("Veri, parametre sayısına göre çok kısa")
    inits = [np.asarray(t, float) for t in inits if t is not None and np.all(np.isfinite(t))]
    if not inits:
        raise ControlError("Başlangıç modeli kurulamadı")
    used_ic = ic
    if ic == "auto":
        used_ic = _auto_ic(st, exps, inits[0])
    if used_ic == "estimate" and len(exps) > 1:
        used_ic = "backcast"  # MATLAB: çok deneyde 'estimate' yerine 'backcast' (uyarıyla)
    if used_ic == "backcast" and len(exps) == 1 and st.nc + int(st.nf.sum()) == 0:
        used_ic = "estimate"  # MATLAB: C ve F yoksa geri süzme tanımsız → 'estimate'
    best = None
    for th0 in inits:
        pem = _PEMBackcast(st, exps) if used_ic == "backcast" else _PEM(st, exps, used_ic)
        if not pem.accept(np.r_[th0, np.zeros(pem.nsx * len(exps))]):
            continue
        try:
            res = pem.fit(th0, max_iter)
        except ControlError:
            continue
        if best is None or res.cost < best[1].cost:
            best = (pem, res)
    if best is None:
        raise ControlError("Kestirim başarısız: kararlı bir başlangıç modeli bulunamadı")
    return _finish(st, best[0], best[1], data, method, used_ic, opts)


# ---- başlangıç tahminleri
def _ls(Phi, Y):
    th, *_ = np.linalg.lstsq(Phi, Y, rcond=None)
    return th


def _hr_armax(exps, na, nb, nc, nk, nu):
    """Hannan–Rissanen: yüksek dereceli ARX artıklarını regresör yaparak A, B, C."""
    nh = max(4 * (na + nc), 10) if nc else 0
    rows, Ys = [], []
    for y, U in exps:
        N = len(y)
        if nc:
            cols = [-shift(y, k) for k in range(1, nh + 1)] + [shift(U[:, j], nk[j] + k)
                                                              for j in range(nu) for k in range(max(nb[j], nh))]
            P = np.column_stack(cols)
            t0 = nh + int(max(nk + np.maximum(nb, nh))) if nu else nh
            t0 = min(t0, N // 2)
            th = _ls(P[t0:], y[t0:])
            ehat = y - P @ th
            ehat[:t0] = 0.0
        else:
            ehat = np.zeros(N)
        cols = [-shift(y, k) for k in range(1, na + 1)]
        cols += [shift(U[:, j], nk[j] + k) for j in range(nu) for k in range(nb[j])]
        cols += [shift(ehat, k) for k in range(1, nc + 1)]
        t0 = max([na, nc] + [nk[j] + nb[j] for j in range(nu)])
        if nc:
            t0 = max(t0, min(nh + 1, N // 2))
        rows.append(np.column_stack(cols)[t0:] if cols else np.zeros((N - t0, 0)))
        Ys.append(y[t0:])
    th = _ls(np.vstack(rows), np.concatenate(Ys))
    A = np.r_[1.0, th[:na]]
    p = na
    B = []
    for j in range(nu):
        B.append(np.r_[np.zeros(nk[j]), th[p:p + nb[j]]])
        p += nb[j]
    C = np.r_[1.0, th[p:p + nc]]
    if not is_stable(C):
        C = stabilize(C)
    return A, B, C


def _sm_oe(exps, nb, nf, nk, nu, iters=30):
    """Steiglitz–McBride (çok girişte ortak payda ile ARX → giriş başına ayrıştırma yerine giriş başına
    dönüşümlü güncelleme). Tek girişte klasik SM."""
    B = [np.r_[np.zeros(nk[j]), np.zeros(nb[j])] for j in range(nu)]
    F = [np.r_[1.0, np.zeros(nf[j])] for j in range(nu)]
    for _ in range(iters):
        changed = 0.0
        for j in range(nu):
            rows, Ys = [], []
            for y, U in exps:
                others = sum(signal.lfilter(B[k], F[k], U[:, k]) for k in range(nu) if k != j) if nu > 1 else 0.0
                yt = y - others
                yf = signal.lfilter([1.0], F[j], yt)
                uf = signal.lfilter([1.0], F[j], U[:, j])
                cols = [-shift(yf, k) for k in range(1, nf[j] + 1)] + [shift(uf, nk[j] + k) for k in range(nb[j])]
                t0 = 0
                rows.append(np.column_stack(cols)[t0:])
                Ys.append(yf[t0:])
            th = _ls(np.vstack(rows), np.concatenate(Ys))
            Fn = np.r_[1.0, th[:nf[j]]]
            if not is_stable(Fn):
                Fn = stabilize(Fn)
            Bn = np.r_[np.zeros(nk[j]), th[nf[j]:]]
            changed = max(changed, np.max(np.abs(Fn - F[j])) if nf[j] else 0.0)
            F[j], B[j] = Fn, Bn
        if changed < 1e-10:
            break
    return B, F


def _arx_oe_init(exps, nb, nf, nk, nu):
    """ARX(nf, nb, nk) ile tek girişli OE başlangıcı (F ≈ A)."""
    B, F = [], []
    for j in range(nu):
        rows, Ys = [], []
        for y, U in exps:
            cols = [-shift(y, k) for k in range(1, nf[j] + 1)] + [shift(U[:, j], nk[j] + k) for k in range(nb[j])]
            t0 = max(nf[j], nk[j] + nb[j])
            rows.append(np.column_stack(cols)[t0:])
            Ys.append(y[t0:])
        th = _ls(np.vstack(rows), np.concatenate(Ys))
        Fj = np.r_[1.0, th[:nf[j]]]
        F.append(stabilize(Fj) if not is_stable(Fj) else Fj)
        B.append(np.r_[np.zeros(nk[j]), th[nf[j]:]])
    return B, F


def _arma_init(v_list, nc, nd):
    """Gürültü dizisine ARMA(nd, nc) Hannan–Rissanen: D v = C e."""
    exps = [(v, np.zeros((len(v), 0))) for v in v_list]
    A, _, C = _hr_armax(exps, nd, np.zeros(0, int), nc, np.zeros(0, int), 0)
    if not is_stable(C):
        C = stabilize(C)
    return C, A


# ============================================================================================ arayüzler
def _siso_orders(orders, nu, names):
    o = np.atleast_1d(np.asarray(orders, float)).ravel()
    return o


def armax(data: IDData, orders, *, initial_condition: str = "auto", max_iter: int = 200) -> IDPoly:
    """ARMAX modeli (MATLAB `armax(data, [na nb nc nk])`): A y = B u + C e. Zaman serisinde `[na nc]` (ARMA).
    Çok girişte `[na, nb(×nu), nc, nk(×nu)]`. Yalnız tek çıkış."""
    data = _prep(data)
    if data.ny != 1:
        raise ControlError("ARMAX tek çıkışlı veri ister (çok çıkış için n4sid / ssest ya da çıkış seçin)")
    nu = data.nu
    o = np.atleast_1d(np.asarray(orders, float)).ravel()
    if nu == 0:
        if len(o) != 2:
            raise ControlError("Zaman serisi ARMA için derece [na nc] olmalı")
        na, nc = int(o[0]), int(o[1])
        nb = np.zeros(0, int)
        nk = np.zeros(0, int)
    else:
        if len(o) != 2 + 2 * nu:
            raise ControlError(f"ARMAX derecesi [na, nb(×{nu}), nc, nk(×{nu})] olmalı")
        na = int(o[0])
        nb = o[1:1 + nu].astype(int)
        nc = int(o[1 + nu])
        nk = o[2 + nu:].astype(int)
    st = _PolyStructure(na, nb, nc, 0, np.zeros(nu, int), nk)
    exps = _exps(data)
    A, B, C = _hr_armax(exps, na, nb, nc, nk, nu)
    th0 = st.pack(A, B, C, [1.0], [[1.0]] * nu)
    inits = [th0, st.pack(A, B, np.r_[1.0, np.zeros(nc)], [1.0], [[1.0]] * nu)]
    return _run_pem(data, st, inits, initial_condition, "ARMAX", max_iter,
                    {"orders": o.tolist(), "initial_condition": initial_condition})


def oe(data: IDData, orders, *, initial_condition: str = "auto", max_iter: int = 200) -> IDPoly:
    """Çıkış hatası modeli (MATLAB `oe(data, [nb nf nk])`): y = Σ B_j/F_j u_j + e. Çok girişte
    `[nb(×nu), nf(×nu), nk(×nu)]`. Yalnız tek çıkış."""
    data = _prep(data)
    if data.ny != 1:
        raise ControlError("OE tek çıkışlı veri ister")
    nu = data.nu
    if nu == 0:
        raise ControlError("OE modeli giriş verisi ister")
    o = np.atleast_1d(np.asarray(orders, float)).ravel()
    if len(o) != 3 * nu:
        raise ControlError(f"OE derecesi [nb(×{nu}), nf(×{nu}), nk(×{nu})] olmalı")
    nb, nf, nk = o[:nu].astype(int), o[nu:2 * nu].astype(int), o[2 * nu:].astype(int)
    st = _PolyStructure(0, nb, 0, 0, nf, nk)
    exps = _exps(data)
    B1, F1 = _sm_oe(exps, nb, nf, nk, nu)
    B2, F2 = _arx_oe_init(exps, nb, nf, nk, nu)
    inits = [st.pack([1.0], B1, [1.0], [1.0], F1), st.pack([1.0], B2, [1.0], [1.0], F2)]
    return _run_pem(data, st, inits, initial_condition, "OE", max_iter,
                    {"orders": o.tolist(), "initial_condition": initial_condition})


def bj(data: IDData, orders, *, initial_condition: str = "auto", max_iter: int = 200) -> IDPoly:
    """Box–Jenkins modeli (MATLAB `bj(data, [nb nc nd nf nk])`): y = Σ B_j/F_j u_j + C/D e.
    Çok girişte `[nb(×nu), nc, nd, nf(×nu), nk(×nu)]`. Yalnız tek çıkış."""
    data = _prep(data)
    if data.ny != 1:
        raise ControlError("BJ tek çıkışlı veri ister")
    nu = data.nu
    if nu == 0:
        raise ControlError("BJ modeli giriş verisi ister (zaman serisi için armax [na nc] kullanın)")
    o = np.atleast_1d(np.asarray(orders, float)).ravel()
    if len(o) != 2 + 3 * nu:
        raise ControlError(f"BJ derecesi [nb(×{nu}), nc, nd, nf(×{nu}), nk(×{nu})] olmalı")
    nb = o[:nu].astype(int)
    nc, nd = int(o[nu]), int(o[nu + 1])
    nf = o[nu + 2:2 * nu + 2].astype(int)
    nk = o[2 * nu + 2:].astype(int)
    st = _PolyStructure(0, nb, nc, nd, nf, nk)
    exps = _exps(data)
    st_oe = _PolyStructure(0, nb, 0, 0, nf, nk)
    inits = []
    for B, F in (_sm_oe(exps, nb, nf, nk, nu), _arx_oe_init(exps, nb, nf, nk, nu)):
        try:
            pem = _PEM(st_oe, exps, "zero")
            r = pem.fit(st_oe.pack([1.0], B, [1.0], [1.0], F), 50)
            _, Bo, _, _, Fo = st_oe.unpack(r.x)
        except ControlError:
            Bo, Fo = B, F
        v_list = [y - sum(signal.lfilter(Bo[j], Fo[j], U[:, j]) for j in range(nu)) for y, U in exps]
        C, D = _arma_init(v_list, nc, nd)
        if not is_stable(D):
            D = stabilize(D)
        inits.append(st.pack([1.0], Bo, C, D, Fo))
    return _run_pem(data, st, inits, initial_condition, "BJ", max_iter,
                    {"orders": o.tolist(), "initial_condition": initial_condition})


def polyest(data: IDData, na: int = 0, nb=(), nc: int = 0, nd: int = 0, nf=(), nk=(), *,
            initial_condition: str = "auto", max_iter: int = 200) -> IDPoly:
    """Genel polinom model (MATLAB `polyest(data, [na nb nc nd nf nk])`): A y = Σ B_j/F_j u_j + C/D e."""
    data = _prep(data)
    if data.ny != 1:
        raise ControlError("Genel polinom model tek çıkışlı veri ister")
    nu = data.nu
    nb = np.atleast_1d(np.asarray(nb, int)) if nu else np.zeros(0, int)
    nf = np.atleast_1d(np.asarray(nf, int)) if nu else np.zeros(0, int)
    nk = np.atleast_1d(np.asarray(nk, int)) if nu else np.zeros(0, int)
    if nu and len(nf) == 0:
        nf = np.zeros(nu, int)
    if nu and len(nk) == 0:
        nk = np.ones(nu, int)
    st = _PolyStructure(na, nb, nc, nd, nf, nk)
    exps = _exps(data)
    A, B, C = _hr_armax(exps, na, nb, nc, nk, nu)
    inits = [st.pack(A, B, C, np.r_[1.0, np.zeros(nd)], [np.r_[1.0, np.zeros(f)] for f in nf])]
    return _run_pem(data, st, inits, initial_condition, "POLYEST", max_iter,
                    {"orders": [na, nb.tolist(), nc, nd, nf.tolist(), nk.tolist()]})


# ============================================================================================ AR
def ar(data: IDData, n: int, approach: str = "ls") -> IDPoly:
    """AR zaman serisi modeli A(q) y = e (MATLAB `ar(y, n, approach)`).

    - 'ls': ileri öngörü en küçük kareleri (pencereleme yok; ilk n örnek atılır) — `arx(y, n)` ile aynı.
    - 'fb': ileri-geri en küçük kareler (MATLAB `ar` varsayılanı): ileri ve geri öngörü hatalarının toplamı.
    - 'yw': Yule–Walker (önyargılı özilinti, ortalama çıkarılmaz).
    - 'burg': Burg yöntemi (harmonik ortalamalı yansıma katsayıları)."""
    data = _prep(data)
    if data.nu != 0 or data.ny != 1:
        raise ControlError("AR modeli tek çıkışlı zaman serisi ister")
    n = int(n)
    if n < 1:
        raise ControlError("AR derecesi en az 1 olmalı")
    if approach == "ls":
        return arx(data, [n])
    ys = [y[:, 0] for y in data.ys]
    if approach == "fb":
        rows, Ys = [], []
        for y in ys:
            N = len(y)
            Pf = np.column_stack([-y[n - k:N - k] for k in range(1, n + 1)])
            Pb = np.column_stack([-y[k:N - n + k] for k in range(1, n + 1)])
            rows += [Pf, Pb]
            Ys += [y[n:], y[:N - n]]
        a = _ls(np.vstack(rows), np.concatenate(Ys))
    elif approach == "yw":
        y = np.concatenate(ys)
        N = len(y)
        r = np.array([y[k:] @ y[:N - k] / N for k in range(n + 1)])
        from scipy.linalg import solve_toeplitz
        a = solve_toeplitz(r[:n], -r[1:n + 1])
    elif approach == "burg":
        if len(ys) > 1:
            raise ControlError("Burg yöntemi tek deneyli veri ister")
        y = ys[0]
        f, b = y.copy(), y.copy()
        a = np.zeros(0)
        for m in range(n):
            ff, bb = f[m + 1:], b[m:-1]
            k = -2 * (ff @ bb) / (ff @ ff + bb @ bb)
            a = np.r_[a + k * a[::-1], k]
            f_new = ff + k * bb
            b_new = bb + k * ff
            f = np.r_[np.zeros(m + 1), f_new]
            b = np.r_[np.zeros(m + 1), b_new]
    else:
        raise ControlError("AR yöntemi 'ls', 'fb', 'yw' ya da 'burg' olmalı")
    A = np.r_[1.0, a]
    e_list = [signal.lfilter(A, [1.0], y).reshape(-1, 1) for y in ys]
    for e in e_list:
        e[:n] = 0.0  # ilk n örnek: başlangıç durumu
    Ntot = sum(len(y) for y in ys)
    fit = make_fit([y.reshape(-1, 1) for y in ys], e_list, n + n * len(ys))
    E = np.vstack(e_list)
    lam = float(E[:, 0] @ E[:, 0]) / (Ntot - n - n * len(ys))
    model = IDPoly(A, None, Ts=data.Ts, noise_variance=lam, output_names=data.output_names)
    model.report = Report("AR", fit, "estimate", 0, "Doğrudan çözüm", n, n * len(ys), None, data.name, Ntot,
                          data.Ts, {"approach": approach}, f"AR ({approach}) ile tahmin edildi")
    return model
