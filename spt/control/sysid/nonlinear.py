"""Doğrusal olmayan modeller: `nlarx` (doğrusal olmayan ARX) ve `nlhw` (Hammerstein–Wiener) — MATLAB anlamıyla.

**nlarx**: ŷ(t) = F(φ(t)), φ(t) regresörler: y_i(t−1…t−na_ij), u_j(t−nk_ij…t−nk_ij−nb_ij+1) (MATLAB
linearRegressor sırası), ayrıca `PolyRegressor` (MATLAB polynomialRegressor) ve `CustomRegressor`
(MATLAB customRegressor). F (çıkış başına):
- `Linear()` (idLinear): F = Lᵀφ + d — en küçük kareler (MATLAB ile ~1e-10).
- `SigmoidNet(n)` (idSigmoidNetwork): F = Lᵀx̃ + d + Σ a_k σ(b_kᵀx̃ + c_k), σ(z) = 1/(1+e^{−z}).
- `WaveletNet(n='auto')` (idWaveletNetwork, MATLAB varsayılanı): F = Lᵀx̃ + d + Σ a_s κ(b_s(Qx̃ − c_s)) +
  Σ a_w ψ(b_w(Qx̃ − c_w)), κ(v) = e^{−½|v|²}, ψ(v) = (dim − |v|²)e^{−½|v|²}.
x̃: tahmin verisinden ortalama/standart sapmayla normalize regresörler (MATLAB Normalize = true), Q: temel
bileşen dönüşümü. Eğitim ölçütü bir adım öngörü hatası (MATLAB Focus 'prediction'); sigmoid ağ: tohumlu çok
başlangıçlı Levenberg–Marquardt (analitik Jacobian); dalgacık ağı: ikili (dyadic) ızgaradan ortogonal en küçük
karelerle birim seçimi, birim sayısı GCV ile ('auto'), ardından tüm parametrelerin LM ile arıtılması. MATLAB'in ağ
eğitim ayrıntıları yayımlanmamıştır: aynı model sınıfı ve ölçüt kullanılır, sonuç uyum yüzdesiyle karşılaştırılır.

Benzetim (k = ∞): geçmiş çıkışlar yerine model çıktıları geri beslenir; k adım öngörü: ölçülen çıkış t−k'ya kadar,
sonrası öngörülerle. Başlangıç koşulu: 'z' (veri öncesi sıfır) ya da 'e' — doğrusal olmayan modellerde ilk
max-gecikme örneğinde ölçülen çıkışlar kullanılır (MATLAB 'e' başlangıç durumunu doğrusal olmayan en küçük
kareyle arar; fark yalnız ilk birkaç örneği etkiler).

**nlhw**: w_j = f_j(u_j), x = Σ B_j/F_j w_j (q⁻¹, nk), y = h(x) + e. Giriş / çıkış doğrusal olmayanlıkları:
`Saturation`, `DeadZone`, `PiecewiseLinear`, `Poly1D`, `UnitGain`. Ölçüt benzetim hatası (MATLAB nlhw), sıfır
başlangıç koşulu (nlhwOptions varsayılanı); tüm parametreler (B, F, doğrusal olmayanlık) birlikte LM ile.
Doyum / ölü bölge sınırlarında ölçüt türevsizdir; çok başlangıçlı LM kullanılır.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal
from scipy.special import expit

from spt.control.errors import ControlError
from spt.control.sysid._util import fd_jacobian, is_stable, levenberg_marquardt, shift, stabilize
from spt.control.sysid.data import IDData, check_complete
from spt.control.sysid.models import Report, make_fit

__all__ = ["nlarx", "nlhw", "IDNLARX", "IDNLHW", "Linear", "SigmoidNet", "WaveletNet", "PolyRegressor",
           "CustomRegressor", "Saturation", "DeadZone", "PiecewiseLinear", "Poly1D", "UnitGain"]


# ============================================================================================ regresörler
@dataclass
class PolyRegressor:
    """MATLAB `polynomialRegressor`: değişkenlerin gecikmeli değerlerinin kuvvetleri.

    `variables`: sinyal adları (ör. ['y1', 'u1']), `lags`: değişken başına gecikme listesi, `order`: kuvvet (≥2).
    `allow_variable_mix` / `allow_lag_mix`: farklı değişken / gecikmelerin çarpımları (MATLAB AllowVariableMix,
    AllowLagMix; varsayılan False → yalnız tek terimin kuvveti)."""
    variables: list
    lags: list
    order: int = 2
    allow_variable_mix: bool = False
    allow_lag_mix: bool = False

    def terms(self):
        """[(adlar ve gecikmeler: [(var, lag)], üsler)] listesi."""
        from itertools import combinations_with_replacement

        base = [(v, int(lag)) for v, lg in zip(self.variables, self.lags) for lag in np.atleast_1d(lg)]
        out = []
        for combo in combinations_with_replacement(range(len(base)), self.order):
            items = [base[i] for i in combo]
            vs = {v for v, _ in items}
            ls = {lag for _, lag in items}
            if len(vs) > 1 and not self.allow_variable_mix:
                continue
            if len(ls) > 1 and not self.allow_lag_mix:
                continue
            out.append(items)
        return out


@dataclass
class CustomRegressor:
    """MATLAB `customRegressor`: fn(x1(t−l1), x2(t−l2), …). `variables`, `lags` eşit uzunlukta; `fn` numpy
    vektörleştirilmiş fonksiyon. `name`: görüntülenecek ad."""
    variables: list
    lags: list
    fn: object
    name: str = ""


def _regressor_spec(ny, nu, na, nb, nk, extra, ynames, unames):
    """Çıkış başına regresör listesi: her biri ('lin', sinyal türü, indis, gecikme) ya da ('poly', terimler)
    ya da ('custom', CustomRegressor)."""
    specs = []
    for i in range(ny):
        s = []
        for j in range(ny):
            for k in range(1, na[i, j] + 1):
                s.append(("lin", "y", j, k, f"{ynames[j]}(t-{k})"))
        for j in range(nu):
            for k in range(nb[i, j]):
                lag = nk[i, j] + k
                s.append(("lin", "u", j, lag, f"{unames[j]}(t-{lag})" if lag else f"{unames[j]}(t)"))
        for r in extra:
            if isinstance(r, PolyRegressor):
                for items in r.terms():
                    parts = []
                    for v, lag in items:
                        parts.append(_sig_index(v, ynames, unames) + (lag,))
                    name = "*".join(f"{v}(t-{lag})" for v, lag in items)
                    s.append(("poly", parts, name))
            elif isinstance(r, CustomRegressor):
                parts = [_sig_index(v, ynames, unames) + (int(lag),) for v, lag in zip(r.variables, r.lags)]
                s.append(("custom", parts, r.fn, r.name or "custom"))
            else:
                raise ControlError("Ek regresörler PolyRegressor ya da CustomRegressor olmalı")
        specs.append(s)
    return specs


def _sig_index(v, ynames, unames):
    if v in ynames:
        return ("y", ynames.index(v))
    if v in unames:
        return ("u", unames.index(v))
    raise ControlError(f"'{v}' adlı sinyal yok")


def _max_lag(specs):
    m = 0
    for s in specs:
        for r in s:
            if r[0] == "lin":
                m = max(m, r[3] if r[1] == "y" else r[3])
            else:
                m = max(m, max(p[2] for p in r[1]))
    return m


def _reg_matrix(spec, Y, U, t_idx, ylag=None):
    """Regresör matrisi (len(t_idx), nr); t − lag < 0 → 0. `ylag(lag)`: o gecikme için çıkış kaynağı (k adım
    öngörüde gecikmeye göre ölçülen ya da öngörülen dizi)."""
    cols = []

    def get(kind, j, lag):
        if kind == "y":
            src = (ylag(lag) if ylag is not None else Y)[:, j]
        else:
            src = U[:, j]
        out = np.zeros(len(t_idx))
        tt = t_idx - lag
        ok = tt >= 0
        out[ok] = src[tt[ok]]
        return out

    for r in spec:
        if r[0] == "lin":
            cols.append(get(r[1], r[2], r[3]))
        elif r[0] == "poly":
            v = np.ones(len(t_idx))
            for kind, j, lag in r[1]:
                v = v * get(kind, j, lag)
            cols.append(v)
        else:
            args = [get(kind, j, lag) for kind, j, lag in r[1]]
            cols.append(np.asarray(r[2](*args), float) * np.ones(len(t_idx)))
    return np.column_stack(cols) if cols else np.zeros((len(t_idx), 0))


# ============================================================================================ eşlemeler
class _Map:
    """Normalize regresörlerden skaler çıkışa eşleme tabanı."""

    def setup(self, X):
        self.mu = X.mean(axis=0)
        sd = X.std(axis=0)
        self.sd = np.where(sd > 0, sd, 1.0)

    def norm(self, X):
        return (X - self.mu) / self.sd


class Linear(_Map):
    """MATLAB `idLinear`: F(φ) = Lᵀφ + d (en küçük kareler)."""

    def fit(self, X, y, nl_idx, rng):
        A = np.column_stack([X, np.ones(len(X))])
        th, *_ = np.linalg.lstsq(A, y, rcond=None)
        self.L, self.d = th[:-1], th[-1]
        return self

    def __call__(self, X):
        return X @ self.L + self.d

    @property
    def n_params(self):
        return len(self.L) + 1


class SigmoidNet(_Map):
    """MATLAB `idSigmoidNetwork(n)`: doğrusal terim + n sigmoid birimi (normalize regresörlerde)."""

    def __init__(self, n_units: int = 10, linear_term: bool = True, restarts: int = 3, max_iter: int = 100):
        self.n_units = int(n_units)
        self.linear_term = linear_term
        self.restarts = restarts
        self.max_iter = max_iter

    def _unpack(self, p, nr, nn):
        K = self.n_units
        k = 0
        L = p[k:k + nr] if self.linear_term else np.zeros(nr)
        k += nr if self.linear_term else 0
        d = p[k]
        k += 1
        a = p[k:k + K]
        k += K
        B = p[k:k + K * nn].reshape(K, nn)
        k += K * nn
        c = p[k:k + K]
        return L, d, a, B, c

    def _eval(self, p, Xn, nl_idx):
        nr = Xn.shape[1]
        L, d, a, B, c = self._unpack(p, nr, len(nl_idx))
        Z = Xn[:, nl_idx] @ B.T + c
        S = expit(Z)
        return Xn @ L + d + S @ a, S

    def _jac(self, p, Xn, nl_idx):
        nr = Xn.shape[1]
        L, d, a, B, c = self._unpack(p, nr, len(nl_idx))
        Xs = Xn[:, nl_idx]
        S = expit(Xs @ B.T + c)
        dS = S * (1 - S) * a
        cols = []
        if self.linear_term:
            cols.append(Xn)
        cols.append(np.ones((len(Xn), 1)))
        cols.append(S)
        cols.append((dS[:, :, None] * Xs[:, None, :]).reshape(len(Xn), -1))
        cols.append(dS)
        return np.hstack(cols)

    def fit(self, X, y, nl_idx, rng):
        self.setup(X)
        Xn = self.norm(X)
        self.nl_idx = np.asarray(nl_idx, int)
        nr = Xn.shape[1]
        nn = len(self.nl_idx)
        K = self.n_units
        best = None
        for _ in range(max(self.restarts, 1)):
            B0 = rng.normal(0, 1.0 / np.sqrt(max(nn, 1)), (K, nn)) * 2
            c0 = rng.uniform(-1, 1, K)
            S = expit(Xn[:, self.nl_idx] @ B0.T + c0)
            A = np.hstack(([Xn] if self.linear_term else []) + [np.ones((len(Xn), 1)), S])
            lin, *_ = np.linalg.lstsq(A, y, rcond=None)
            k = nr if self.linear_term else 0
            p0 = np.r_[lin[:k], lin[k], lin[k + 1:], B0.ravel(), c0]
            res = levenberg_marquardt(lambda p: y - self._eval(p, Xn, self.nl_idx)[0], p0,
                                      lambda p: -self._jac(p, Xn, self.nl_idx), max_iter=self.max_iter,
                                      ftol=1e-10, xtol=1e-9)
            if best is None or res.cost < best.cost:
                best = res
        self.p = best.x
        self.train_cost = best.cost
        return self

    def __call__(self, X):
        return self._eval(self.p, self.norm(X), self.nl_idx)[0]

    @property
    def n_params(self):
        return len(self.p)


class WaveletNet(_Map):
    """MATLAB `idWaveletNetwork(n)` (nlarx varsayılanı): doğrusal terim + ölçek fonksiyonları ve dalgacıklar.
    `n_units='auto'`: birim sayısı genelleştirilmiş çapraz doğrulama (GCV) ile."""

    def __init__(self, n_units="auto", max_units: int = 40, refine: bool = True, max_iter: int = 20):
        self.n_units = n_units
        self.max_units = max_units
        self.refine = refine
        self.max_iter = max_iter  # MATLAB SearchOptions.MaxIterations varsayılanı 20

    @staticmethod
    def _unit(V, kind):
        r2 = np.sum(V * V, axis=1)
        e = np.exp(-0.5 * r2)
        return e if kind == 0 else (V.shape[1] - r2) * e

    def _design(self, Z, units):
        cols = []
        for kind, b, c in units:
            cols.append(self._unit(b * (Z - c), kind))
        return np.column_stack(cols) if cols else np.zeros((len(Z), 0))

    def fit(self, X, y, nl_idx, rng):
        self.setup(X)
        Xn = self.norm(X)
        self.nl_idx = np.asarray(nl_idx, int)
        Xs = Xn[:, self.nl_idx]
        # temel bileşenler (Q): birim varyanslı döndürme
        cov = np.cov(Xs.T) if Xs.shape[1] > 1 else np.atleast_2d(np.var(Xs))
        ev, V = np.linalg.eigh(np.atleast_2d(cov))
        ev = np.maximum(ev, 1e-12)
        self.Q = V / np.sqrt(ev)
        Z = Xs @ self.Q
        N, dim = Z.shape
        # aday birimler: ikili ızgara, veri desteğinde
        cands = []
        lo, hi = Z.min(axis=0), Z.max(axis=0)
        span = max(float(np.max(hi - lo)), 1e-9)
        for level in range(0, 3):
            b = 2.0 ** level / span * 2.0
            step = 1.0 / b
            grids = [np.arange(lo[i], hi[i] + step / 2, step) for i in range(dim)]
            if np.prod([len(g) for g in grids]) > 300:
                # yüksek boyutta veri noktalarından örnekle (tohumlu)
                pts = Z[rng.choice(N, size=min(N, 150), replace=False)]
            else:
                mesh = np.meshgrid(*grids, indexing="ij")
                pts = np.column_stack([m.ravel() for m in mesh])
            for c in pts:
                dist2 = np.sum(((Z - c) * b) ** 2, axis=1)
                if np.sum(dist2 < 2.25) >= max(3, dim + 1):
                    for kind in (0, 1):
                        cands.append((kind, b, c.copy()))
        lin = np.column_stack([Xn, np.ones(N)])
        # ortogonal en küçük kareler ile ileri seçim (adaylar artımlı Gram–Schmidt ile güncel tutulur)
        Qb, _ = np.linalg.qr(lin)
        r = y - Qb @ (Qb.T @ y)
        chosen = []
        maxu = self.max_units if self.n_units == "auto" else int(self.n_units)
        maxu = min(maxu, max(1, N // 10), len(cands))
        P = self._design(Z, cands) if cands else np.zeros((N, 0))
        P = P - Qb @ (Qb.T @ P)
        gcv_hist = []
        avail = np.ones(P.shape[1], bool)
        p_lin = lin.shape[1]
        gcv_hist.append(N * float(r @ r) / (N - p_lin) ** 2)
        nrm0 = np.sum(P * P, axis=0)
        for _ in range(maxu):
            if not avail.any():
                break
            nrm = np.sum(P * P, axis=0)
            ok = avail & (nrm > 1e-8 * np.maximum(nrm0, 1e-300))
            score = np.where(ok, (r @ P) ** 2 / np.maximum(nrm, 1e-300), -1.0)
            j = int(np.argmax(score))
            if score[j] <= 0:
                break
            q = P[:, j] / np.sqrt(nrm[j])
            r = r - q * (q @ r)
            P = P - np.outer(q, q @ P)
            avail[j] = False
            chosen.append(j)
            pk = p_lin + len(chosen)
            if N - pk <= 0:
                break
            gcv_hist.append(N * float(r @ r) / (N - pk) ** 2)
        if self.n_units == "auto":
            k = int(np.argmin(gcv_hist))
        else:
            k = len(chosen)
        self.units = [cands[j] for j in chosen[:k]]
        self._solve_linear(Xn, Z, y)
        if self.refine and self.units:
            self._refine(Xn, Z, y)
        self.gcv = gcv_hist
        return self

    def _solve_linear(self, Xn, Z, y):
        A = np.column_stack([Xn, np.ones(len(Xn)), self._design(Z, self.units)])
        th, *_ = np.linalg.lstsq(A, y, rcond=None)
        nr = Xn.shape[1]
        self.L, self.d, self.a = th[:nr], th[nr], th[nr + 1:]

    def _pack(self):
        return np.r_[self.L, self.d, self.a, [u[1] for u in self.units], np.concatenate([u[2] for u in self.units])]

    def _apply(self, p, Xn, Z):
        nr = Xn.shape[1]
        K = len(self.units)
        dim = Z.shape[1]
        L, d, a = p[:nr], p[nr], p[nr + 1:nr + 1 + K]
        b = p[nr + 1 + K:nr + 1 + 2 * K]
        c = p[nr + 1 + 2 * K:].reshape(K, dim)
        out = Xn @ L + d
        for k, (kind, _, _) in enumerate(self.units):
            out = out + a[k] * self._unit(b[k] * (Z - c[k]), kind)
        return out

    def _jac(self, p, Xn, Z):
        """∂F/∂p (analitik): κ' = −vκ, ψ' = −v e^{−|v|²/2}(2 + m − |v|²), v = b(z − c)."""
        nr = Xn.shape[1]
        K = len(self.units)
        N, dim = Z.shape
        a = p[nr + 1:nr + 1 + K]
        b = p[nr + 1 + K:nr + 1 + 2 * K]
        c = p[nr + 1 + 2 * K:].reshape(K, dim)
        Ja = np.zeros((N, K))
        Jb = np.zeros((N, K))
        Jc = np.zeros((N, K, dim))
        for k, (kind, _, _) in enumerate(self.units):
            D = Z - c[k]
            V = b[k] * D
            r2 = np.sum(V * V, axis=1)
            e = np.exp(-0.5 * r2)
            if kind == 0:
                Ja[:, k] = e
                g = -e  # dκ/dv = g·v
            else:
                Ja[:, k] = (dim - r2) * e
                g = -e * (2 + dim - r2)
            Jb[:, k] = a[k] * g * np.sum(V * D, axis=1)
            Jc[:, k, :] = -a[k] * b[k] * g[:, None] * V
        return np.hstack([Xn, np.ones((N, 1)), Ja, Jb, Jc.reshape(N, K * dim)])

    def _refine(self, Xn, Z, y):
        p0 = self._pack()
        f = lambda p: y - self._apply(p, Xn, Z)  # noqa: E731
        res = levenberg_marquardt(f, p0, lambda p: -self._jac(p, Xn, Z), max_iter=self.max_iter,
                                  ftol=1e-10, xtol=1e-9)
        nr = Xn.shape[1]
        K = len(self.units)
        dim = Z.shape[1]
        p = res.x
        self.L, self.d, self.a = p[:nr], p[nr], p[nr + 1:nr + 1 + K]
        b = p[nr + 1 + K:nr + 1 + 2 * K]
        c = p[nr + 1 + 2 * K:].reshape(K, dim)
        self.units = [(u[0], b[k], c[k]) for k, u in enumerate(self.units)]

    def __call__(self, X):
        Xn = self.norm(X)
        Z = Xn[:, self.nl_idx] @ self.Q
        return self._apply(self._pack(), Xn, Z)

    @property
    def n_params(self):
        return len(self._pack())


# ============================================================================================ nlarx modeli
class IDNLARX:
    """Doğrusal olmayan ARX modeli (MATLAB `idnlarx`)."""

    def __init__(self, specs, maps, Ts, ny, nu, input_names, output_names, noise_variance):
        self.specs = specs
        self.maps = maps
        self.Ts = Ts
        self.ny, self.nu = ny, nu
        self.input_names, self.output_names = tuple(input_names), tuple(output_names)
        self.noise_variance = np.atleast_2d(noise_variance)
        self.max_lag = _max_lag(specs)
        self.report: Report | None = None
        self.name = ""

    @property
    def regressor_names(self):
        return [[r[-1] if r[0] != "custom" else r[3] for r in s] for s in self.specs]

    def _step_outputs(self, Yfull, U, t_idx):
        return np.column_stack([self.maps[i](_reg_matrix(self.specs[i], Yfull, U, t_idx)) for i in range(self.ny)])

    def _simulate_data(self, data: IDData, initial_condition="zero", y_init=None):
        out = []
        for y, U in zip(data.ys, data.us):
            N = len(U)
            Yh = np.zeros((N, self.ny))
            n0 = 0
            if not (isinstance(initial_condition, str) and initial_condition in ("z", "zero")):
                n0 = min(self.max_lag, N)
                Yh[:n0] = y[:n0]
            for t in range(n0, N):
                Yh[t] = self._step_outputs(Yh, U, np.array([t]))[0]
            out.append(Yh)
        return out

    def _predict_data(self, data: IDData, k, ic):
        ys, xs = [], []
        for y, U in zip(data.ys, data.us):
            d1 = IDData(y, U if U.shape[1] else None, data.Ts)
            if k is None:
                Yh = self._simulate_data(d1, ic)[0]
            else:
                # P_s(t): t−s'ye kadarki ölçümlerle y(t) öngörüsü; P_0 = y. P_s'nin regresöründe gecikme i < s
                # için P_{s−i}(t−i), i ≥ s için ölçülen y kullanılır.
                N = len(y)
                t_all = np.arange(N)
                P = [y]
                for s in range(1, k + 1):
                    src = lambda lag, s=s: P[max(s - lag, 0)]  # noqa: E731
                    P.append(np.column_stack([self.maps[i](_reg_matrix(self.specs[i], y, U, t_all, src))
                                              for i in range(self.ny)]))
                Yh = P[k].copy()
                if not (isinstance(ic, str) and ic in ("z", "zero")):
                    n0 = min(self.max_lag, N)
                    Yh[:n0] = y[:n0]
            ys.append(Yh)
            xs.append(None)
        return ys, xs

    def __repr__(self):
        return f"<IDNLARX ny={self.ny} nu={self.nu} {[type(m).__name__ for m in self.maps]}>"


def _resolve_map(nl):
    if isinstance(nl, str):
        key = nl.lower()
        if key in ("wavenet", "wavelet", "idwaveletnetwork"):
            return WaveletNet()
        if key in ("sigmoidnet", "sigmoid", "idsigmoidnetwork"):
            return SigmoidNet()
        if key in ("linear", "idlinear"):
            return Linear()
        raise ControlError(f"Bilinmeyen doğrusal olmayanlık: {nl}")
    if isinstance(nl, (Linear, SigmoidNet, WaveletNet)):
        import copy
        return copy.deepcopy(nl)
    raise ControlError("Doğrusal olmayanlık 'wavenet', 'sigmoidnet', 'linear' ya da eşleme nesnesi olmalı")


def nlarx(data: IDData, orders, nonlinearity="wavenet", *, custom_regressors=(), nonlinear_regressors="all",
          seed: int = 0) -> IDNLARX:
    """Doğrusal olmayan ARX modeli (MATLAB `nlarx(data, [na nb nk], nonlinearity)`).

    `orders`: [na, nb(×nu), nk(×nu)] (çok çıkışta (na, nb, nk) matrisleri). `nonlinearity`: 'wavenet' (varsayılan),
    'sigmoidnet', 'linear' ya da `WaveletNet(...)`, `SigmoidNet(n)`, `Linear()` (çok çıkışta liste olabilir).
    `custom_regressors`: `PolyRegressor` / `CustomRegressor` listesi. `nonlinear_regressors`: doğrusal olmayan
    bloğa giren regresörlerin indisleri ya da 'all' (MATLAB RegressorUsage). `seed`: başlangıç rastgeleliği."""
    from spt.control.sysid.linear import _arx_orders

    if not isinstance(data, IDData):
        raise ControlError("Veri IDData olmalı")
    check_complete(data)
    ny, nu = data.ny, data.nu
    if nu == 0:
        o = np.atleast_1d(np.asarray(orders, int)).ravel()
        na = np.full((ny, ny), int(o[0]))
        nb = np.zeros((ny, 0), int)
        nk = np.zeros((ny, 0), int)
    else:
        na, nb, nk = _arx_orders(orders, ny, nu)
    specs = _regressor_spec(ny, nu, na, nb, nk, list(custom_regressors), data.output_names, data.input_names)
    nls = nonlinearity if isinstance(nonlinearity, (list, tuple)) else [nonlinearity] * ny
    if len(nls) != ny:
        raise ControlError("Doğrusal olmayanlık sayısı çıkış sayısına eşit olmalı")
    rng = np.random.default_rng(seed)
    nmax = _max_lag(specs)
    maps = []
    Xs, Ts_ = [], []
    for i in range(ny):
        rows, tg = [], []
        for y, U in zip(data.ys, data.us):
            t_idx = np.arange(nmax, len(y))
            rows.append(_reg_matrix(specs[i], y, U, t_idx))
            tg.append(y[t_idx, i])
        X = np.vstack(rows)
        Y = np.concatenate(tg)
        nr = X.shape[1]
        if nr == 0:
            raise ControlError("Regresör yok")
        nl_idx = np.arange(nr) if (isinstance(nonlinear_regressors, str) and nonlinear_regressors == "all") \
            else np.asarray(nonlinear_regressors, int)
        m = _resolve_map(nls[i])
        m.fit(X, Y, nl_idx, rng)
        maps.append(m)
        Xs.append(X)
        Ts_.append(Y)
    E = np.column_stack([Ts_[i] - maps[i](Xs[i]) for i in range(ny)])
    lam = E.T @ E / len(E)
    model = IDNLARX(specs, maps, data.Ts, ny, nu, data.input_names, data.output_names, lam)
    # rapor: bir adım öngörü uyumu (ilk nmax örnek ölçülen çıkışla)
    from spt.control.sysid.evaluate import _model_predict
    yh, _ = _model_predict(model, data, 1, "e")
    e_list = [y - h for y, h in zip(data.ys, yh)]
    npar = sum(getattr(m, "n_params", 0) for m in maps)
    fit = make_fit(data.ys, e_list, npar)
    model.report = Report("NLARX", fit, "estimate", 0, "", npar, 0, None, data.name, sum(data.lengths), data.Ts,
                          {"orders": orders, "nonlinearity": [type(m).__name__ for m in maps]},
                          f"NLARX ({', '.join(type(m).__name__ for m in maps)}) ile tahmin edildi")
    return model


# ============================================================================================ 1-B doğrusal olmayanlıklar
class _NL1:
    def params(self):
        return np.zeros(0)

    def set_params(self, p):
        pass

    def init_from(self, x):
        pass


class UnitGain(_NL1):
    """MATLAB `idUnitGain`: f(x) = x."""

    def __call__(self, x):
        return np.asarray(x, float)


class Saturation(_NL1):
    """MATLAB `idSaturation`: f(x) = min(max(x, alt), üst); tek taraflı için ±inf."""

    def __init__(self, lower=None, upper=None):
        self.lower, self.upper = lower, upper

    def params(self):
        return np.array([v for v in (self.lower, self.upper) if v is not None and np.isfinite(v)], float)

    def set_params(self, p):
        k = 0
        if self.lower is not None and np.isfinite(self.lower):
            self.lower = float(p[k])
            k += 1
        if self.upper is not None and np.isfinite(self.upper):
            self.upper = float(p[k])

    def init_from(self, x, frac=0.25):
        lo, hi = np.quantile(x, [0.02, 0.98])
        span = hi - lo
        if self.lower is None:
            self.lower = lo + frac * span
        if self.upper is None:
            self.upper = hi - frac * span

    def __call__(self, x):
        lo = -np.inf if self.lower is None else self.lower
        hi = np.inf if self.upper is None else self.upper
        return np.clip(x, min(lo, hi), max(lo, hi))

    @property
    def LinearInterval(self):
        return (self.lower, self.upper)


class DeadZone(_NL1):
    """MATLAB `idDeadZone`: [alt, üst] aralığında 0, dışında x − üst / x − alt."""

    def __init__(self, lower=None, upper=None):
        self.lower, self.upper = lower, upper

    def params(self):
        return np.array([self.lower, self.upper], float)

    def set_params(self, p):
        self.lower, self.upper = float(p[0]), float(p[1])

    def init_from(self, x, frac=0.1):
        lo, hi = np.quantile(x, [0.02, 0.98])
        span = hi - lo
        mid = np.median(x)
        if self.lower is None:
            self.lower = mid - frac * span
        if self.upper is None:
            self.upper = mid + frac * span

    def __call__(self, x):
        lo, hi = min(self.lower, self.upper), max(self.lower, self.upper)
        x = np.asarray(x, float)
        return np.where(x > hi, x - hi, np.where(x < lo, x - lo, 0.0))

    @property
    def ZeroInterval(self):
        return (self.lower, self.upper)


class PiecewiseLinear(_NL1):
    """MATLAB `idPiecewiseLinear`: kırılma noktaları (x_k, y_k) arasında doğrusal, dışında uç doğrularla."""

    def __init__(self, n_breakpoints: int = 10, breakpoints=None):
        self.n = int(n_breakpoints)
        self.bx = None if breakpoints is None else np.asarray(breakpoints[0], float)
        self.by = None if breakpoints is None else np.asarray(breakpoints[1], float)

    def params(self):
        return np.r_[self.bx, self.by]

    def set_params(self, p):
        n = len(p) // 2
        self.bx, self.by = np.asarray(p[:n], float), np.asarray(p[n:], float)

    def init_from(self, x):
        if self.bx is None:
            lo, hi = np.quantile(x, [0.01, 0.99])
            self.bx = np.linspace(lo, hi, self.n)
            self.by = self.bx.copy()

    def __call__(self, x):
        x = np.asarray(x, float)
        o = np.argsort(self.bx)
        bx, by = self.bx[o], self.by[o]
        if len(bx) == 1:
            return x - bx[0] + by[0]
        y = np.interp(x, bx, by)
        s0 = (by[1] - by[0]) / (bx[1] - bx[0]) if bx[1] != bx[0] else 0.0
        s1 = (by[-1] - by[-2]) / (bx[-1] - bx[-2]) if bx[-1] != bx[-2] else 0.0
        y = np.where(x < bx[0], by[0] + s0 * (x - bx[0]), y)
        return np.where(x > bx[-1], by[-1] + s1 * (x - bx[-1]), y)

    @property
    def BreakPoints(self):
        return np.vstack([self.bx, self.by])


class Poly1D(_NL1):
    """MATLAB `idPolynomial1D(derece)`: f(x) = Σ c_k x^k (katsayılar polyval sırasında, en yüksek önce)."""

    def __init__(self, degree: int = 2, coefficients=None):
        self.degree = int(degree)
        self.coef = None if coefficients is None else np.asarray(coefficients, float)

    def params(self):
        return self.coef.copy()

    def set_params(self, p):
        self.coef = np.asarray(p, float)

    def init_from(self, x):
        if self.coef is None:
            self.coef = np.zeros(self.degree + 1)
            self.coef[-2] = 1.0  # birim eğim

    def __call__(self, x):
        return np.polyval(self.coef, x)

    @property
    def Coefficients(self):
        return self.coef


def _resolve_nl1(nl):
    import copy
    if nl is None:
        return UnitGain()
    if isinstance(nl, str):
        k = nl.lower()
        return {"saturation": Saturation(), "idsaturation": Saturation(), "deadzone": DeadZone(),
                "iddeadzone": DeadZone(), "pwlinear": PiecewiseLinear(), "piecewiselinear": PiecewiseLinear(),
                "idpiecewiselinear": PiecewiseLinear(), "poly1d": Poly1D(), "idpolynomial1d": Poly1D(),
                "unitgain": UnitGain(), "idunitgain": UnitGain(), "": UnitGain()}.get(k) or _bad(nl)
    if isinstance(nl, _NL1):
        return copy.deepcopy(nl)
    raise ControlError("Doğrusal olmayanlık nesnesi tanınmadı")


def _bad(nl):
    raise ControlError(f"Bilinmeyen doğrusal olmayanlık: {nl}")


# ============================================================================================ nlhw modeli
class IDNLHW:
    """Hammerstein–Wiener modeli (MATLAB `idnlhw`)."""

    def __init__(self, B, F, nk, input_nl, output_nl, Ts, input_names, output_names, noise_variance):
        self.B, self.F = [np.asarray(b, float) for b in B], [np.asarray(f, float) for f in F]
        self.nk = np.asarray(nk, int)
        self.input_nl, self.output_nl = input_nl, output_nl
        self.Ts = Ts
        self.ny, self.nu = 1, len(self.B)
        self.input_names, self.output_names = tuple(input_names), tuple(output_names)
        self.noise_variance = np.atleast_2d(noise_variance)
        self.report: Report | None = None
        self.name = ""

    def _linear_out(self, U):
        x = np.zeros(len(U))
        for j in range(self.nu):
            x += signal.lfilter(self.B[j], self.F[j], self.input_nl[j](U[:, j]))
        return x

    def _simulate_data(self, data: IDData, initial_condition="zero"):
        return [self.output_nl(self._linear_out(U)).reshape(-1, 1) for U in data.us]

    def _predict_data(self, data: IDData, k, ic):
        ys = self._simulate_data(data)
        if isinstance(ic, str) and ic in ("e", "estimate"):
            ys = [self._est_x0(y[:, 0], U) for y, U in zip(data.ys, data.us)]
        return ys, [None] * len(ys)

    def _est_x0(self, y, U):
        """Doğrusal bloğun başlangıç durumunu çıktı hatasını en küçükleyecek şekilde (doğrusal olmayan EKK)."""
        from scipy.signal import lfilter
        den = np.array([1.0])
        for f in self.F:
            den = np.convolve(den, f)
        n = len(den) - 1
        if n == 0:
            return self.output_nl(self._linear_out(U)).reshape(-1, 1)
        x_zero = self._linear_out(U)
        N = len(y)
        basis = np.column_stack([shift(lfilter([1.0], den, np.r_[1.0, np.zeros(N - 1)]), i) for i in range(n)])
        f = lambda s: y - self.output_nl(x_zero + basis @ s)  # noqa: E731
        res = levenberg_marquardt(f, np.zeros(n), lambda s: fd_jacobian(f, s, central=True), max_iter=100)
        return self.output_nl(x_zero + basis @ res.x).reshape(-1, 1)

    def __repr__(self):
        return f"<IDNLHW nu={self.nu} giriş={[type(n).__name__ for n in self.input_nl]} çıkış={type(self.output_nl).__name__}>"


def nlhw(data: IDData, orders, input_nl="pwlinear", output_nl="pwlinear", *, max_iter: int = 300,
         seed: int = 0) -> IDNLHW:
    """Hammerstein–Wiener modeli (MATLAB `nlhw(data, [nb nf nk], InputNL, OutputNL)`), tek çıkış.

    `input_nl`: giriş başına (liste) ya da tek: 'saturation', 'deadzone', 'pwlinear', 'poly1d', 'unitgain' ya da
    nesne (`Saturation(alt, üst)` …; verilen değerler başlangıç). `output_nl`: aynı. Ölçüt benzetim hatası."""
    if not isinstance(data, IDData):
        raise ControlError("Veri IDData olmalı")
    check_complete(data)
    if data.ny != 1:
        raise ControlError("Hammerstein–Wiener modeli tek çıkışlı veri ister")
    nu = data.nu
    if nu == 0:
        raise ControlError("Hammerstein–Wiener modeli giriş verisi ister")
    o = np.atleast_1d(np.asarray(orders, int)).ravel()
    if len(o) != 3 * nu:
        raise ControlError(f"Derece [nb(×{nu}), nf(×{nu}), nk(×{nu})] olmalı")
    nb, nf, nk = o[:nu], o[nu:2 * nu], o[2 * nu:]
    in_nls = input_nl if isinstance(input_nl, (list, tuple)) else [input_nl] * nu
    if len(in_nls) != nu:
        raise ControlError("Giriş doğrusal olmayanlığı sayısı giriş sayısına eşit olmalı")
    exps = [(y[:, 0].copy(), u.copy()) for y, u in zip(data.ys, data.us)]
    rng = np.random.default_rng(seed)
    U_all = np.vstack([u for _, u in exps])
    from spt.control.sysid.linear import oe as _oe

    lin = _oe(data, list(nb) + list(nf) + list(nk), initial_condition="zero")
    Bl = lin.B if nu > 1 else [lin.B]
    Fl = lin.F if nu > 1 else [lin.F]

    def make_nls(variant):
        """Başlangıç değişkesi: doyum / ölü bölge sınırları farklı oranlarla (verilenler korunur)."""
        ins = [_resolve_nl1(n) for n in in_nls]
        out = _resolve_nl1(output_nl)
        fr_sat = (0.25, 0.1, 0.4, 0.25)[variant]
        fr_dz = (0.1, 0.05, 0.2, 0.02)[variant]
        for j, n in enumerate(ins):
            if isinstance(n, Saturation):
                n.init_from(U_all[:, j], frac=fr_sat)
            elif isinstance(n, DeadZone):
                n.init_from(U_all[:, j], frac=fr_dz)
            else:
                n.init_from(U_all[:, j])
        x_all = np.concatenate([sum(signal.lfilter(Bl[j], Fl[j], ins[j](u[:, j])) for j in range(nu))
                                for _, u in exps])
        if isinstance(out, Saturation):
            out.init_from(x_all, frac=fr_sat)
        elif isinstance(out, DeadZone):
            out.init_from(x_all, frac=fr_dz)
        else:
            out.init_from(x_all)
        return ins, out

    def pack(B, F, ins, out):
        p = []
        for j in range(nu):
            p += list(B[j][nk[j]:nk[j] + nb[j]])
        for j in range(nu):
            p += list(F[j][1:1 + nf[j]])
        for n in ins:
            p += list(n.params())
        p += list(out.params())
        return np.array(p, float)

    def unpack(p, ins, out):
        k = 0
        B, F = [], []
        for j in range(nu):
            B.append(np.r_[np.zeros(nk[j]), p[k:k + nb[j]]])
            k += nb[j]
        for j in range(nu):
            F.append(np.r_[1.0, p[k:k + nf[j]]])
            k += nf[j]
        for n in ins:
            m = len(n.params())
            n.set_params(p[k:k + m])
            k += m
        m = len(out.params())
        out.set_params(p[k:k + m])
        return B, F

    best = None
    def ls_input_init(ins):
        """Polinom / parçalı doğrusal giriş doğrusallığı: doğrusal blok (OE) sabitken f'nin katsayıları doğrusaldır →
        y ≈ Σ_k c_k·(B/F)[φ_k(u)] en küçük karelerle (çıkış birim kabulüyle) başlangıç."""
        cols, idx = [], []
        for j, n in enumerate(ins):
            if isinstance(n, Poly1D):
                for k in range(len(n.coef)):
                    e = np.zeros(len(n.coef))
                    e[k] = 1.0
                    cols.append(np.concatenate([signal.lfilter(Bl[j], Fl[j], np.polyval(e, u[:, j]))
                                                for _, u in exps]))
                    idx.append((j, "c", k))
            elif isinstance(n, PiecewiseLinear):
                by0 = n.by.copy()
                for k in range(len(by0)):
                    n.by = np.eye(len(by0))[k]
                    cols.append(np.concatenate([signal.lfilter(Bl[j], Fl[j], n(u[:, j])) for _, u in exps]))
                    idx.append((j, "y", k))
                n.by = by0
            else:
                cols.append(np.concatenate([signal.lfilter(Bl[j], Fl[j], n(u[:, j])) for _, u in exps]))
                idx.append((j, "fix", 0))
        if not any(t[1] in ("c", "y") for t in idx):
            return False
        Y = np.concatenate([y for y, _ in exps])
        th, *_ = np.linalg.lstsq(np.column_stack(cols), Y, rcond=None)
        for (j, kind, k), v in zip(idx, th):
            if kind == "c":
                ins[j].coef[k] = v
            elif kind == "y":
                ins[j].by[k] = v
        return True

    for variant in range(5):
        ins, out = make_nls(min(variant, 3))
        if variant == 4 and not (isinstance(out, UnitGain) and ls_input_init(ins)):
            continue
        p0 = pack(Bl, Fl, ins, out)

        def resid(p, ins=ins, out=out):
            B, F = unpack(p, ins, out)
            r = []
            for y, U in exps:
                x = sum(signal.lfilter(B[j], F[j], ins[j](U[:, j])) for j in range(nu))
                r.append(y - out(x))
            return np.concatenate(r)

        def accept(p, ins=ins, out=out):
            k = int(nb.sum())
            for j in range(nu):
                if not is_stable(np.r_[1.0, p[k:k + nf[j]]]):
                    return False
                k += nf[j]
            return True

        if not accept(p0):
            for j in range(nu):
                Fl[j] = stabilize(Fl[j])
            p0 = pack(Bl, Fl, ins, out)
        try:
            res = levenberg_marquardt(resid, p0, lambda p: fd_jacobian(resid, p, central=True), accept=accept,
                                      max_iter=max_iter, ftol=1e-12, xtol=1e-10)
        except ControlError:
            continue
        if best is None or res.cost < best[0].cost:
            unpack(res.x, ins, out)
            best = (res, ins, out, unpack(res.x, ins, out))
    if best is None:
        raise ControlError("Hammerstein–Wiener kestirimi başarısız")
    res, ins, out, (B, F) = best
    r_all = []
    model = IDNLHW(B, F, nk, ins, out, data.Ts, data.input_names, data.output_names, 1.0)
    sims = model._simulate_data(data)
    r_all = [y - s for y, s in zip(data.ys, sims)]
    E = np.vstack(r_all)
    model.noise_variance = np.atleast_2d(float(E[:, 0] @ E[:, 0]) / len(E))
    npar = len(res.x)
    model.report = Report("NLHW", make_fit(data.ys, r_all, npar), "zero", res.iterations, res.why, npar, 0, None,
                          data.name, len(E), data.Ts, {"orders": o.tolist()},
                          f"NLHW ({[type(n).__name__ for n in ins]}, {type(out).__name__}) ile tahmin edildi")
    _ = rng
    return model
