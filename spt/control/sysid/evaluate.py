"""Model değerlendirme: `sim`, `predict`, `compare`, `pe`, `resid` (MATLAB anlamıyla).

Tüm doğrusal modeller ayrık yenilik biçiminde (`models.DSS`) değerlendirilir:

- benzetim (k = ∞): ŷ = G u (gürültü modeli kullanılmaz);
- k adım öngörü: ŷ(t|t−k) = C x̂(t|t−k) + D w(t); x̂(·|·−1) Kalman öngörücüsüyle (x̂⁺ = A x̂ + B w + K(y − C x̂ − D w)),
  ardından k−1 adım açık çevrim; veri öncesi (t − k < 0) bölgede öngörü saf benzetimdir.
- Başlangıç koşulu 'e' (MATLAB compare / predict / pe / resid varsayılanı): x0, ilgili öngörücünün (k adım ya da
  benzetim) hatasının kareleri toplamını en küçükleyen değerdir (çıktı x0'da afin olduğundan doğrusal en küçük
  kareler). 'z': x0 = 0. Ayrıca açık x0 vektörü verilebilir.
- Uyum (fit): 100·(1 − ‖y − ŷ‖/‖y − ȳ‖) çıkış başına.

Doğrusal olmayan modeller (nlarx / nlhw) kendi `_simulate` / `_predict` yöntemleriyle aynı arayüzden geçer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spt.control.errors import ControlError
from spt.control.sysid.data import IDData, check_complete
from spt.control.sysid.models import DSS, IDModel, fit_percent

__all__ = ["sim", "predict", "compare", "pe", "resid", "ResidResult", "CompareResult"]


# ============================================================================================ çekirdek
def _delayed(U: np.ndarray, d) -> np.ndarray:
    W = np.zeros_like(U)
    for j in range(U.shape[1]):
        dj = int(d[j])
        if dj == 0:
            W[:, j] = U[:, j]
        elif dj < len(U):
            W[dj:, j] = U[:len(U) - dj, j]
    return W


def _run(A, B, C, D, W, x0=None, K=None, Y=None):
    """Durum yinelemesi; K ve Y verilirse bir adım öngörücü. Döner: (çıktı (N, ny), durumlar (N, nx))."""
    N = W.shape[0]
    nx = A.shape[0]
    ny = C.shape[0]
    X = np.zeros((N, nx))
    out = np.zeros((N, ny))
    x = np.zeros(nx) if x0 is None else np.asarray(x0, float).copy()
    BW = W @ B.T if B.size else np.zeros((N, nx))
    DW = W @ D.T if D.size else np.zeros((N, ny))
    if K is None:
        for t in range(N):
            X[t] = x
            x = A @ x + BW[t]
        out = X @ C.T + DW
        return out, X
    Ab = A - K @ C
    KY = Y @ K.T
    BKW = BW - DW @ K.T
    for t in range(N):
        X[t] = x
        x = Ab @ x + BKW[t] + KY[t]
    out = X @ C.T + DW
    return out, X


def _free(M: np.ndarray, Cm: np.ndarray, N: int, pre: np.ndarray | None = None, start: int = 0):
    """Serbest cevap matrisi: satır t → Cm·pre·M^(t−start) (t ≥ start), t < start için ayrı doldurulur.
    Döner (N, ny, nx)."""
    nx = M.shape[0]
    ny = Cm.shape[0]
    out = np.zeros((N, ny, nx))
    P = Cm @ (pre if pre is not None else np.eye(nx))
    scale = max(np.abs(P).max(), 1e-300)
    for t in range(start, N):
        out[t] = P
        P = P @ M
        if np.abs(P).max() < 1e-17 * scale:
            break
    return out


def _linear_predict(s: DSS, Y: np.ndarray, U: np.ndarray, k, x0_mode):
    """Bir deney için k adım öngörü (k=None → benzetim). Döner: (ŷ, x0)."""
    N = len(Y)
    ny = Y.shape[1]
    W = _delayed(U, s.delay) if U.shape[1] else np.zeros((N, 0))
    nx = s.nx
    if k is None:
        yd, _ = _run(s.A, s.B, s.C, s.D, W)
        Psi = _free(s.A, s.C, N) if nx else np.zeros((N, ny, 0))
    elif k == 1:
        yd, _ = _run(s.A, s.B, s.C, s.D, W, K=s.K, Y=Y)
        Psi = _free(s.A - s.K @ s.C, s.C, N) if nx else np.zeros((N, ny, 0))
    else:
        _, X1 = _run(s.A, s.B, s.C, s.D, W, K=s.K, Y=Y)
        Ak = np.linalg.matrix_power(s.A, k - 1)
        Xk = np.zeros_like(X1)
        if k - 1 < N:
            Xk[k - 1:] = X1[:N - k + 1] @ Ak.T
        P = np.eye(nx)
        for j in range(k - 1, 0, -1):  # j = k−1 … 1: A^{k−1−j} B w(t−k+j)
            M = P @ s.B
            lag = k - j
            if lag < N and M.size:
                Xk[lag:] += W[:N - lag] @ M.T
            P = s.A @ P
        yd = Xk @ s.C.T + (W @ s.D.T if s.D.size else 0.0)
        Psi = np.zeros((N, ny, nx))
        if nx:
            Pw = s.C.copy()
            for t in range(min(k - 1, N)):  # t < k−1: A^t x0
                Psi[t] = Pw
                Pw = Pw @ s.A
            if k - 1 < N:
                Psi[k - 1:] = _free(s.A - s.K @ s.C, s.C @ Ak, N - k + 1)
    if isinstance(x0_mode, str):
        if x0_mode in ("z", "zero"):
            x0 = np.zeros(nx)
        elif x0_mode in ("e", "estimate"):
            if nx:
                M = Psi.transpose(1, 0, 2).reshape(N * ny, nx, order="C")
                r = (Y - yd).T.reshape(-1)
                x0, *_ = np.linalg.lstsq(M, r, rcond=None)
            else:
                x0 = np.zeros(0)
        else:
            raise ControlError("Başlangıç koşulu 'e' (tahmin), 'z' (sıfır) ya da durum vektörü olmalı")
    else:
        x0 = np.asarray(x0_mode, float).ravel()
        if len(x0) != nx:
            raise ControlError(f"Başlangıç durumu {nx} elemanlı olmalı")
    yhat = yd + (np.einsum("tij,j->ti", Psi, x0) if nx else 0.0)
    return yhat, x0


def _is_linear(model) -> bool:
    return isinstance(model, IDModel)


def _model_predict(model, data: IDData, k, ic):
    """Tüm deneyler için öngörü listesi ve x0 listesi."""
    if model.ny != data.ny or model.nu != data.nu:
        raise ControlError(f"Model {model.ny} çıkış / {model.nu} giriş, veri {data.ny} çıkış / {data.nu} giriş")
    check_complete(data, "Doğrulama")
    if not _is_linear(model):
        return model._predict_data(data, k, ic)
    s = model.dss(data.Ts, data.intersample)
    out, xs = [], []
    ic_list = ic if (isinstance(ic, list) and data.ne > 1) else [ic] * data.ne
    for (Y, U), ic_e in zip(zip(data.ys, data.us), ic_list):
        yh, x0 = _linear_predict(s, Y, U, k, ic_e)
        out.append(yh)
        xs.append(x0)
    return out, xs


def _k(k):
    if k is None or (isinstance(k, float) and np.isinf(k)):
        return None
    k = int(k)
    if k < 1:
        raise ControlError("Öngörü ufku k ≥ 1 olmalı (benzetim için inf)")
    return k


# ============================================================================================ arayüzler
def sim(model, data, *, initial_condition="zero", add_noise: bool = False, seed=None):
    """Benzetim (MATLAB `sim(sys, data)`): ölçülen girişlerle model çıktısı.

    `data`: IDData (girişler kullanılır) ya da (N, nu) giriş dizisi (ayrık modelde Ts modelinki). Başlangıç:
    'zero' (MATLAB varsayılanı) ya da durum vektörü. `add_noise=True`: H e eklenir, e ~ N(0, NoiseVariance)
    (`seed` ile tekrarlanabilir). Döner: IDData verildiyse IDData, aksi halde (N, ny) dizi."""
    if isinstance(data, IDData):
        d = data
    else:
        U = np.asarray(data, float)
        U = U[:, None] if U.ndim == 1 else U
        if model.Ts <= 0:
            raise ControlError("Sürekli zamanlı modelin benzetimi için örnekleme aralıklı IDData verin")
        d = IDData(np.zeros((len(U), model.ny)), U, model.Ts)
    if d.nu != model.nu:
        raise ControlError(f"Model {model.nu} giriş bekliyor, veride {d.nu} var")
    for u in d.us:
        if np.isnan(u).any():
            raise ControlError("Giriş verisinde eksik (NaN) örnek var")
    ys = []
    rng = np.random.default_rng(seed)
    if _is_linear(model):
        s = model.dss(d.Ts, d.intersample)
        for U in d.us:
            W = _delayed(U, s.delay) if U.shape[1] else np.zeros((len(U), 0))
            x0 = None if (isinstance(initial_condition, str) and initial_condition in ("zero", "z")) else \
                np.asarray(initial_condition, float).ravel()
            if x0 is not None and len(x0) != s.nx:
                raise ControlError(f"Başlangıç durumu {s.nx} elemanlı olmalı")
            y, _ = _run(s.A, s.B, s.C, s.D, W, x0)
            if add_noise:
                L = np.linalg.cholesky(model.noise_variance)
                E = rng.standard_normal((len(U), model.ny)) @ L.T
                yn, _ = _run(s.A, s.K, s.C, np.eye(model.ny), E)
                y = y + yn
            ys.append(y)
    else:
        ys = model._simulate_data(d, initial_condition)
        if add_noise:
            L = np.linalg.cholesky(model.noise_variance)
            ys = [y + rng.standard_normal(y.shape) @ L.T for y in ys]
    if isinstance(data, IDData):
        return IDData(ys, d.us, d.Ts, tstart=d.tstarts, output_names=model.output_names,
                      input_names=d.input_names, name=f"{d.name} benzetim", intersample=d.intersample)
    return ys[0]


def predict(model, data: IDData, k=1, *, initial_condition="e", return_x0: bool = False):
    """k adım öngörü (MATLAB `predict(sys, data, k)`; k = inf benzetim). Varsayılan başlangıç 'e' (tahmin)."""
    kk = _k(k)
    ys, xs = _model_predict(model, data, kk, initial_condition)
    out = IDData(ys, data.us, data.Ts, tstart=data.tstarts, output_names=data.output_names,
                 input_names=data.input_names, name=f"{data.name} öngörü", intersample=data.intersample)
    return (out, xs if data.ne > 1 else xs[0]) if return_x0 else out


def pe(model, data: IDData, *, initial_condition="e"):
    """Bir adım öngörü hataları e = y − ŷ(t|t−1) (MATLAB `pe`). Döner IDData (çıkışlar = hatalar)."""
    ys, _ = _model_predict(model, data, 1, initial_condition)
    es = [y - yh for y, yh in zip(data.ys, ys)]
    return IDData(es, data.us, data.Ts, tstart=data.tstarts, output_names=data.output_names,
                  input_names=data.input_names, name=f"{data.name} öngörü hatası", intersample=data.intersample)


@dataclass
class CompareResult:
    """`compare` sonucu: `yhat[m]` model başına IDData, `fit[m]` (deney başına, çıkış başına) uyum %, `x0[m]`."""
    yhat: list
    fit: list
    x0: list
    k: object

    def __iter__(self):
        return iter((self.yhat, self.fit, self.x0))


def compare(data: IDData, *models, k=np.inf, initial_condition="e") -> CompareResult:
    """Modelleri veriyle karşılaştırır (MATLAB `compare(data, sys1, sys2, …, k)`).

    Uyum: 100·(1 − ‖y − ŷ‖/‖y − ȳ‖), çıkış başına (çok deneyde deney başına). Varsayılan k = inf (benzetim) ve
    başlangıç koşulu 'e' (MATLAB compareOptions varsayılanı: başlangıç durumu en iyi uyacak şekilde tahmin)."""
    if not models:
        raise ControlError("En az bir model verilmeli")
    kk = _k(k)
    yh_all, fit_all, x0_all = [], [], []
    for m in models:
        ys, xs = _model_predict(m, data, kk, initial_condition)
        fits = [fit_percent([y], [yh]) for y, yh in zip(data.ys, ys)]
        yh_all.append(IDData(ys, data.us, data.Ts, tstart=data.tstarts, output_names=data.output_names,
                             input_names=data.input_names, name=getattr(m, "name", "") or type(m).__name__,
                             intersample=data.intersample))
        fit_all.append(fits[0] if data.ne == 1 else np.array(fits))
        x0_all.append(xs[0] if data.ne == 1 else xs)
    return CompareResult(yh_all, fit_all, x0_all, k)


@dataclass
class ResidResult:
    """`resid` sonucu.

    - `e`: artıklar (IDData), `R`: MATLAB düzeninde kovaryanslar (M+1, ny+nu, ny+nu),
      R[τ, i, j] = (1/N) Σ z_i(t) z_j(t+τ), z = [e, u] (R2025b'de ölçüldü).
    - `lags_auto` (0…M), `auto` (M+1, ny): normalize özilinti r_e(τ) = R_ee(τ)/R_ee(0).
    - `lags_cross` (−M…M), `cross` (2M+1, ny, nu): normalize çapraz ilinti r_ue(τ) = E[u(t) e(t+τ)]/√(R_u(0)R_e(0)).
    - `conf_auto` (ny,), `conf_cross` (ny, nu): güven sınırları (±)."""
    e: IDData
    R: np.ndarray
    lags_auto: np.ndarray
    auto: np.ndarray
    lags_cross: np.ndarray
    cross: np.ndarray
    conf_auto: np.ndarray
    conf_cross: np.ndarray
    num_sd: float


def resid(data: IDData, model, *, max_lag: int = 25, initial_condition="e", num_sd: float = 2.58,
          ) -> ResidResult:
    """Artık analizi (MATLAB `resid(data, sys)`): bir adım öngörü hatalarının özilintisi ve girişle çapraz
    ilintisi, güven sınırlarıyla (MaxLag 25 — MATLAB residOptions; %99 bölge = 2.58 standart sapma).

    Güven sınırları MATLAB R2025b residCorr tanımıyla (Ljung, §16.6): normalize özilinti için ±num_sd/√N; artık
    girişten bağımsızken çapraz ilinti için ±num_sd·√(1 + 2Σ_{k=1..M} r_e(k) r_u(k))/√N. R matrisi MATLAB'den
    ölçülerek doğrulandı."""
    E = pe(model, data, initial_condition=initial_condition)
    M = int(max_lag)
    e_all = np.vstack(E.ys)
    u_all = np.vstack(data.us) if data.nu else np.zeros((len(e_all), 0))
    z = np.hstack([e_all, u_all])
    N, nz = z.shape
    if M >= N:
        raise ControlError("En büyük gecikme veri uzunluğundan küçük olmalı")
    R = np.zeros((M + 1, nz, nz))
    for tau in range(M + 1):
        R[tau] = z[:N - tau].T @ z[tau:] / N
    ny, nu = data.ny, data.nu
    q = float(num_sd)
    auto = np.stack([R[:, i, i] / R[0, i, i] for i in range(ny)], axis=1)
    lags_cross = np.arange(-M, M + 1)
    cross = np.zeros((2 * M + 1, ny, nu))
    conf_cross = np.zeros((ny, nu))
    for i in range(ny):
        for j in range(nu):
            uj = ny + j
            norm = np.sqrt(R[0, i, i] * R[0, uj, uj])
            # τ ≥ 0: E[u(t) e(t+τ)] = R[τ, u, e]; τ < 0: E[u(t) e(t−|τ|)] = E[e(t) u(t+|τ|)] = R[|τ|, e, u]
            pos = R[:, uj, i]
            neg = R[1:, i, uj][::-1]
            cross[:, i, j] = np.r_[neg, pos] / norm
            re = R[:, i, i] / R[0, i, i]
            ru = R[:, uj, uj] / R[0, uj, uj]
            P = re[0] * ru[0] + 2 * np.sum(re[1:] * ru[1:])
            conf_cross[i, j] = q * np.sqrt(max(P, 0.0) / N)
    conf_auto = np.full(ny, q / np.sqrt(N))
    return ResidResult(E, R, np.arange(M + 1), auto, lags_cross, cross, conf_auto, conf_cross, q)
