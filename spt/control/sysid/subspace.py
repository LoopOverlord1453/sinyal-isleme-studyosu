"""Durum uzayı kestirimi: `n4sid` (alt uzay) ve `ssest` (n4sid başlangıcı + öngörü hatası yöntemi).

**n4sid** — yayımlanmış alt uzay algoritmaları (Van Overschee & De Moor 1996; Verhaegen 1994 MOESP;
Larimore 1990 CVA). Geçmiş vektörü w_p(t) = [y(t−sy…t−1); u(t−su…t−1)], gelecek U_f, Y_f (ufuk r):
eğik izdüşüm O = Y_f /_{U_f} W_p; ağırlık MOESP: SVD(O Π⊥_{U_f}), CVA: SVD((Y_f Π⊥ Y_fᵀ)^{-1/2} O Π⊥).
Genişletilmiş gözlenebilirlik matrisinden kaydırma değişmezliğiyle A, C; durum dizisi kestiriminin artık
kovaryanslarından (DARE) K; A, C, K sabitken B, D, x0 öngörücü hatasında doğrusal en küçük karelerle.
MATLAB'in 'auto' ağırlığı CVA'dır, 'auto' ufku r = ⌈1.5·nx⌉ ve geçmiş ufku AIC ile seçilen ARX derecesidir
(R2025b kaynak davranışı); MATLAB'in K ve B/D hesap ayrıntıları yayımlanmamıştır → MATLAB'le frekans cevabı ve
uyum yüzdesi üzerinden karşılaştırılır (gürültüsüz veride tam sistem kurtarılır).

**ssest** — n4sid başlangıcından yenilik biçimli modelin öngörü hatası ölçütüyle (çok çıkışta det(EᵀE/N),
MATLAB OutputWeight 'noise') en küçüklenmesi. Tek çıkışta nx dereceli yenilik biçimi, na = nc = nx, nb = nx
(nk = 1; doğrudan geçişte nb = nx+1, nk = 0) ARMAX ile aynı model sınıfıdır (gözlemci kanonik gerçekleme) →
analitik Jacobian'lı polinom PEM kullanılır. Çok çıkışta serbest parametreleme + Levenberg–Marquardt (SVD ile
tanımsız yönler sönümlenir). `Ts=0` (MATLAB ssest varsayılanı): ayrık optimum ZOH ile sürekli zamana çevrilir
(d2c; ayrık yenilik kazancı K_d = ∫₀^Ts e^{Aσ}dσ·K kuralıyla) — örneklenmiş öngörücü aynıdır.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg, signal

from spt.control.errors import ControlError
from spt.control.sysid._util import fd_jacobian, levenberg_marquardt
from spt.control.sysid.data import IDData, check_complete
from spt.control.sysid.models import IDSS, Report, attach_warnings, make_fit, noise_variance

__all__ = ["n4sid", "ssest", "d2c_ss"]


# ============================================================================================ yardımcılar
def _exps(data):
    check_complete(data)
    return [(y.copy(), u.copy()) for y, u in zip(data.ys, data.us)]


def _arx_aic_order(Y, U, nmax):
    """ARX(n, n, 1) çok değişkenli en küçük kareler; AIC ile derece (1…nmax)."""
    Z = np.hstack([Y, U])
    nz = Z.shape[1]
    ny = Y.shape[1]
    best, bn = None, 1
    for n in range(1, nmax + 1):
        Phi = np.hstack([np.roll(Z, k, axis=0) for k in range(1, n + 1)])[nmax:]
        T = Y[nmax:]
        th, *_ = np.linalg.lstsq(Phi, T, rcond=None)
        E = T - Phi @ th
        V = np.linalg.det(E.T @ E / len(E)) if ny > 1 else float(E[:, 0] @ E[:, 0] / len(E))
        if V <= 0:
            continue
        aic = np.log(V) + 2 * n * nz * ny / len(E)
        if best is None or aic < best:
            best, bn = aic, n
    return bn


def _auto_horizon(exps, nx):
    Y, U = exps[0]
    N, ny = Y.shape
    nu = U.shape[1]
    r = int(np.ceil(1.5 * nx))
    maxo = int(np.ceil(min(4 * nx, (N - (ny + nu)) / 2, max(np.floor(N / (ny + nu + 1)) - 1 + nx / ny, 1))))
    p = _arx_aic_order(Y, U, max(maxo, 1))
    return max(r, nx + 1), max(p, 1), max(p, 1)


def _hankels(exps, r, sy, su):
    Wp, Uf, Yf = [], [], []
    for Y, U in exps:
        N, ny = Y.shape
        nu = U.shape[1]
        p = max(sy, su)
        J = N - p - r + 1
        if J <= 0:
            raise ControlError("Veri, alt uzay ufuklarına göre çok kısa")
        t = np.arange(p, p + J)
        wp = [Y[t - k] for k in range(sy, 0, -1)] + [U[t - k] for k in range(su, 0, -1)]
        Wp.append(np.hstack(wp).T if wp else np.zeros((0, J)))
        Uf.append(np.hstack([U[t + k] for k in range(r)]).T if nu else np.zeros((0, J)))
        Yf.append(np.hstack([Y[t + k] for k in range(r)]).T)
    return np.hstack(Wp), np.hstack(Uf), np.hstack(Yf)


def _proj_out(X, U):
    """X Π⊥_U (U'nun satır uzayına dik bileşen)."""
    if U.shape[0] == 0:
        return X
    coef, *_ = np.linalg.lstsq(U.T, X.T, rcond=None)
    return X - (U.T @ coef).T


def _predictor_lsq(exps, A, C, K, nu, feedthrough, est_x0=True):
    """A, C, K sabitken B, D, x0: e = y − ŷ öngörücü hatasında doğrusal en küçük kareler."""
    nx = A.shape[0]
    ny = C.shape[0]
    Ab = A - K @ C
    cols_all, rhs = [], []
    nexp = len(exps)
    npar_b = nx * nu
    npar_d = ny * nu if feedthrough else 0
    for k, (Y, U) in enumerate(exps):
        N = len(Y)
        # ŷ = C x̂ + D u, x̂⁺ = Ab x̂ + (B − K D) u + K y
        # y'ye bağlı kısım (parametresiz): x_y⁺ = Ab x_y + K y
        xy = np.zeros(nx)
        Xy = np.zeros((N, nx))
        for t in range(N):
            Xy[t] = xy
            xy = Ab @ xy + K @ Y[t]
        base = Y - Xy @ C.T  # e = base − (B, D, x0 katkıları)
        # B[i, j] katkısı: x_b⁺ = Ab x_b + e_i u_j
        cols = []
        # birim girdilere cevap: her j için durum yanıtı Sj (N, nx): Sj⁺ = Ab Sj + e_i u_j → tüm i birlikte
        for j in range(nu):
            Xs = np.zeros((N, nx, nx))  # [t, durum, i]
            xs = np.zeros((nx, nx))
            for t in range(N):
                Xs[t] = xs
                xs = Ab @ xs + np.eye(nx) * U[t, j]
            resp = np.einsum("yn,tni->tyi", C, Xs)  # (N, ny, nx): ∂ŷ/∂B[i, j]
            for i in range(nx):
                cols.append(resp[:, :, i])
        if feedthrough:
            for j in range(nu):
                # D[q, j] katkısı: doğrudan e_q u_j ve −K e_q u_j üzerinden durum
                xs = np.zeros((nx, ny))
                Xs = np.zeros((N, nx, ny))
                for t in range(N):
                    Xs[t] = xs
                    xs = Ab @ xs - K * U[t, j]
                resp = np.einsum("yn,tnq->tyq", C, Xs)
                for q in range(ny):
                    r_ = resp[:, :, q].copy()
                    r_[:, q] += U[:, j]
                    cols.append(r_)
        x0cols = []
        if est_x0:
            P = C.copy()
            Fr = np.zeros((N, ny, nx))
            for t in range(N):
                Fr[t] = P
                P = P @ Ab
            for i in range(nx):
                x0cols.append(Fr[:, :, i])
        cols_all.append((cols, x0cols))
        rhs.append(base)
    # birleştir: parametreler [B (sütun sıralı: j, i), D, x0 (deney başına)]
    nrow = sum(len(Y) * Y.shape[1] for Y, _ in exps)
    nx0 = nx * nexp if est_x0 else 0
    M = np.zeros((nrow, npar_b + npar_d + nx0))
    b = np.zeros(nrow)
    off = 0
    for k, ((cols, x0cols), base) in enumerate(zip(cols_all, rhs)):
        N = base.shape[0]
        sl = slice(off, off + N * ny)
        for c, col in enumerate(cols):
            M[sl, c] = col.reshape(-1)
        for i, col in enumerate(x0cols):
            M[sl, npar_b + npar_d + k * nx + i] = col.reshape(-1)
        b[sl] = base.reshape(-1)
        off += N * ny
    sol, *_ = np.linalg.lstsq(M, b, rcond=None)
    B = np.zeros((nx, nu))
    c = 0
    for j in range(nu):
        for i in range(nx):
            B[i, j] = sol[c]
            c += 1
    D = np.zeros((ny, nu))
    if feedthrough:
        for j in range(nu):
            for q in range(ny):
                D[q, j] = sol[c]
                c += 1
    x0 = [sol[npar_b + npar_d + k * nx: npar_b + npar_d + (k + 1) * nx] for k in range(nexp)] if est_x0 else \
        [np.zeros(nx) for _ in range(nexp)]
    return B, D, x0


def _pred_errors(A, B, C, D, K, exps, x0s):
    """Bir adım öngörü hataları e = y − ŷ (yenilik biçimi öngörücüsü). Küçük derecede transfer fonksiyonu +
    lfilter (hızlı), aksi halde durum döngüsü."""
    nx = A.shape[0]
    if 0 < nx <= 12:
        return _pred_errors_tf(A, B, C, D, K, exps, x0s)
    return _pred_errors_loop(A, B, C, D, K, exps, x0s)


def _pred_errors_tf(A, B, C, D, K, exps, x0s):
    Ab = A - K @ C
    Bb = B - K @ D
    ny = C.shape[0]
    nu = B.shape[1]
    Bin = np.hstack([Bb, K])
    den = np.real(np.poly(Ab))
    nums = []
    for c in range(nu + ny):
        num, _ = signal.ss2tf(Ab, Bin[:, [c]], C, np.zeros((ny, 1)))
        nums.append(np.atleast_2d(np.real(num)))
    out = []
    for (Y, U), x0 in zip(exps, x0s):
        N = len(Y)
        W = np.hstack([U, Y])
        Yh = U @ D.T if nu else np.zeros_like(Y)
        Yh = Yh.copy()
        for c in range(nu + ny):
            for i in range(ny):
                Yh[:, i] += signal.lfilter(nums[c][i], den, W[:, c])
        x0 = np.asarray(x0, float)
        if np.any(x0):
            num, _ = signal.ss2tf(Ab, (Ab @ x0)[:, None], C, (C @ x0)[:, None])
            imp = np.r_[1.0, np.zeros(N - 1)]
            for i in range(ny):
                Yh[:, i] += signal.lfilter(np.real(np.atleast_2d(num)[i]), den, imp)
        out.append(Y - Yh)
    return out


def _pred_errors_loop(A, B, C, D, K, exps, x0s):
    out = []
    Ab = A - K @ C
    for (Y, U), x0 in zip(exps, x0s):
        N = len(Y)
        x = np.asarray(x0, float).copy()
        E = np.zeros_like(Y)
        BU = U @ (B - K @ D).T if U.shape[1] else np.zeros((N, A.shape[0]))
        DU = U @ D.T if U.shape[1] else np.zeros_like(Y)
        KY = Y @ K.T
        for t in range(N):
            E[t] = Y[t] - C @ x - DU[t]
            x = Ab @ x + BU[t] + KY[t]
        out.append(E)
    return out


def _stabilize_A(A, thresh=1.0):
    ev, V = np.linalg.eig(A)
    if np.all(np.abs(ev) < thresh):
        return A
    ev2 = np.where(np.abs(ev) >= thresh, ev / np.abs(ev) ** 2, ev)
    ev2 = np.where(np.abs(ev2) >= thresh, ev2 / np.abs(ev2) * 0.999, ev2)
    return np.real(V @ np.diag(ev2) @ np.linalg.inv(V))


# ============================================================================================ n4sid
def _n4sid_core(exps, nx, r, sy, su, weighting, feedthrough, estimate_k):
    ny = exps[0][0].shape[1]
    nu = exps[0][1].shape[1]
    Wp, Uf, Yf = _hankels(exps, r, sy, su)
    J = Wp.shape[1]
    # eğik izdüşüm: Y_f = Lw W_p + Lu U_f
    reg = np.vstack([Wp, Uf])
    coef, *_ = np.linalg.lstsq(reg.T, Yf.T, rcond=None)
    Lw = coef[:Wp.shape[0]].T
    Wpp = _proj_out(Wp, Uf)
    Yfp = _proj_out(Yf, Uf)
    Q, Rw = np.linalg.qr(Wpp.T / np.sqrt(J), mode="reduced")
    M = Lw @ Rw.T  # SVD(O Π⊥) ile aynı sol tekil vektörler
    if weighting == "cva":
        Syy = Yfp @ Yfp.T / J
        ev, V = np.linalg.eigh((Syy + Syy.T) / 2)
        ev = np.maximum(ev, 1e-12 * ev.max())
        W1 = V @ np.diag(ev ** -0.5) @ V.T
        W1i = V @ np.diag(ev ** 0.5) @ V.T
        Us, S, _ = np.linalg.svd(W1 @ M, full_matrices=False)
        G = W1i @ Us[:, :nx] @ np.diag(np.sqrt(S[:nx]))
    elif weighting == "moesp":
        Us, S, _ = np.linalg.svd(M, full_matrices=False)
        G = Us[:, :nx] @ np.diag(np.sqrt(S[:nx]))
    else:
        raise ControlError("Ağırlık 'moesp', 'cva' ya da 'auto' olmalı")
    if np.linalg.matrix_rank(G) < nx:
        raise ControlError("Veri bu derece için yeterince uyarıcı değil (alt uzay boyutu yetersiz)")
    C = G[:ny]
    A = np.linalg.lstsq(G[:-ny], G[ny:], rcond=None)[0]
    A = _stabilize_A(A)
    K = np.zeros((nx, ny))
    if estimate_k:
        # durum dizisi: X̂ = Γ⁺ O, ardışık sütunlar → artık kovaryansları → DARE
        O = Lw @ Wp
        X = np.linalg.pinv(G) @ O
        # deney sınırlarında ardışıklık bozulur: yalnız aynı deneydeki çiftler
        idx, off = [], 0
        for Y, _ in exps:
            Jk = len(Y) - max(sy, su) - r + 1
            idx.extend(range(off, off + Jk - 1))
            off += Jk
        idx = np.array(idx)
        t_cols = []
        for Y, U in exps:
            p = max(sy, su)
            Jk = len(Y) - p - r + 1
            t_cols.append(np.arange(p, p + Jk))
        Ycur = np.vstack([Y[t] for (Y, _), t in zip(exps, t_cols)])
        Ucur = np.vstack([U[t] for (_, U), t in zip(exps, t_cols)]) if nu else np.zeros((len(Ycur), 0))
        Xc, Xn = X[:, idx], X[:, idx + 1]
        Yc, Ucc = Ycur[idx], Ucur[idx]
        Phi = np.vstack([Xc, Ucc.T]) if nu else Xc
        Th = np.linalg.lstsq(Phi.T, np.vstack([Xn, Yc.T]).T, rcond=None)[0].T
        Res = np.vstack([Xn, Yc.T]) - Th @ Phi
        S_ = Res @ Res.T / Res.shape[1]
        Qn, Sn, Rn = S_[:nx, :nx], S_[:nx, nx:], S_[nx:, nx:]
        try:
            P = linalg.solve_discrete_are(A.T, C.T, Qn, Rn, s=Sn)
            K = (A @ P @ C.T + Sn) @ np.linalg.inv(C @ P @ C.T + Rn)
        except (linalg.LinAlgError, ValueError, np.linalg.LinAlgError):
            K = np.zeros((nx, ny))
        if np.max(np.abs(np.linalg.eigvals(A - K @ C))) >= 1:
            K = np.zeros((nx, ny))
    B, D, x0 = _predictor_lsq(exps, A, C, K, nu, feedthrough)
    return A, B, C, D, K, x0, S


def n4sid(data: IDData, nx: int, *, weighting: str = "auto", horizon=None, feedthrough: bool = False,
          disturbance: bool = True, initial_state: str = "estimate", Ts: float | None = None) -> IDSS:
    """Alt uzay yöntemiyle durum uzayı modeli (MATLAB `n4sid(data, nx)`), ayrık (Ts = veri Ts'si).

    `weighting`: 'auto' (= 'cva', MATLAB), 'cva', 'moesp'. `horizon`: (r, sy, su) ya da None (otomatik).
    `feedthrough`: D tahmin edilsin mi (MATLAB varsayılanı hayır). `disturbance`: K tahmin edilsin mi.
    `Ts=0`: sonuç sürekli zamana çevrilir (d2c, ZOH)."""
    exps = _exps(data)
    nx = int(nx)
    if nx < 1:
        raise ControlError("Durum sayısı en az 1 olmalı")
    if data.nu == 0 and not disturbance:
        raise ControlError("Zaman serisi için gürültü modeli (K) tahmin edilmeli")
    w = weighting.lower()
    if w == "auto":
        w = "cva"
    ny_, nu_ = data.ny, data.nu

    def enough(r, sy, su):
        # eğik izdüşüm regresyonu [W_p; U_f] belirli olmalı: sütun sayısı satır sayısından büyük
        rows = sy * ny_ + su * nu_ + r * nu_
        return all(len(y) - max(sy, su) - r + 1 > rows + 1 for y, _ in exps)

    if horizon is None:
        r, sy, su = _auto_horizon(exps, nx)
        if data.nu == 0:
            su = 0
        while not enough(r, sy, su) and max(sy, su) > 1:  # kısa veride geçmiş ufku küçültülür
            sy, su = max(sy - 1, 1), (max(su - 1, 1) if data.nu else 0)
        while not enough(r, sy, su) and r > nx + 1:
            r -= 1
    else:
        r, sy, su = (int(h) for h in horizon)
        if r <= nx:
            raise ControlError("Gelecek ufku r durum sayısından büyük olmalı")
        if data.nu == 0:
            su = 0
    if not enough(r, sy, su):
        need = max(sy, su) + r + sy * ny_ + su * nu_ + r * nu_ + 1
        raise ControlError(f"Veri, {nx}. dereceden alt uzay kestirimi için çok kısa (deney başına en az ~{need} örnek "
                           "gerekli); dereceyi düşürün ya da daha uzun veri kullanın")
    A, B, C, D, K, x0, S = _n4sid_core(exps, nx, r, sy, su, w, feedthrough, disturbance)
    if initial_state == "zero":
        x0 = [np.zeros(nx) for _ in exps]
    E = _pred_errors(A, B, C, D, K, exps, x0)
    Ntot = sum(len(e) for e in E)
    Em = np.vstack(E)
    lam = Em.T @ Em / Ntot
    npar = nx * (data.ny + data.nu) + (data.ny * data.nu if feedthrough else 0) + (nx * data.ny if disturbance else 0)
    nx0 = nx * len(exps) if initial_state != "zero" else 0
    fit = make_fit([y for y, _ in exps], E, npar + nx0)
    model = IDSS(A, B, C, D, K, data.Ts, x0=x0[0], noise_variance=lam, input_names=data.input_names,
                 output_names=data.output_names)
    model.report = Report("N4SID", fit, "estimate" if initial_state != "zero" else "zero", 0, "", npar, nx0, x0,
                          data.name, Ntot, data.Ts, {"weighting": w, "horizon": (r, sy, su)},
                          f"N4SID ({w.upper()}, ufuk {r},{sy},{su}) ile tahmin edildi")
    model.singular_values = S
    attach_warnings(model, data)
    if Ts is not None and Ts == 0:
        return d2c_ss(model)
    return model


# ============================================================================================ d2c
def d2c_ss(model: IDSS) -> IDSS:
    """Ayrık IDSS → sürekli (ZOH tersi, matris logaritması); K: K_d = ∫₀^Ts e^{Aσ}dσ·K kuralının tersi."""
    Ts = model.Ts
    if Ts <= 0:
        return model.copy()
    n, nu, ny = model.nx, model.nu, model.ny
    ev = np.linalg.eigvals(model.A)
    if np.any(np.abs(ev) < 1e-12) or np.any((np.abs(ev.imag) < 1e-12) & (ev.real < 0)):
        raise ControlError("Ayrık modelin sıfırda ya da negatif gerçel eksende kutbu var; sürekli zamana "
                           "çevrilemez (ayrık model kullanın)")
    Mbig = np.zeros((n + nu + ny, n + nu + ny))
    Mbig[:n, :n] = model.A
    Mbig[:n, n:n + nu] = model.B
    Mbig[:n, n + nu:] = model.K
    Mbig[n:, n:] = np.eye(nu + ny)
    L = linalg.logm(Mbig) / Ts
    L = np.real(L)
    Ac, Bc, Kc = L[:n, :n], L[:n, n:n + nu], L[:n, n + nu:]
    out = IDSS(Ac, Bc, model.C.copy(), model.D.copy(), Kc, 0.0, x0=model.x0.copy(),
               input_delay=model.input_delay * Ts, noise_variance=model.noise_variance,
               input_names=model.input_names, output_names=model.output_names)
    out.report = model.report
    return out


# ============================================================================================ ssest
def _ss_from_armax(m, nx, nu, feedthrough, data):
    s = m.dss(data.Ts)
    return s.A, s.B, s.C, s.D, s.K


def ssest(data: IDData, nx: int, *, Ts: float | None = 0.0, feedthrough: bool = False, disturbance: bool = True,
          initial_state: str = "auto", max_iter: int = 200, init_sys: IDSS | None = None) -> IDSS:
    """Durum uzayı modeli, öngörü hatası yöntemi (MATLAB `ssest(data, nx)`).

    `Ts`: 0 (varsayılan, MATLAB gibi sürekli zaman), veri Ts'si (ayrık) ya da None (= veri Ts'si).
    `feedthrough`, `disturbance` (K), `initial_state` ('auto' | 'zero' | 'estimate'). Başlangıç: n4sid (CVA)
    ya da `init_sys`."""
    exps = _exps(data)
    nx = int(nx)
    ny = data.ny
    disc_Ts = data.Ts
    if init_sys is None:
        try:
            init = n4sid(data, nx, feedthrough=feedthrough, disturbance=disturbance)
        except ControlError:
            init = n4sid(data, nx, weighting="moesp", feedthrough=feedthrough, disturbance=disturbance)
    else:
        init = init_sys if init_sys.Ts > 0 else _c2d_idss(init_sys, disc_Ts, data)
    if ny == 1 and disturbance:
        model = _ssest_siso(data, nx, init, feedthrough, initial_state, max_iter)
    else:
        model = _ssest_mimo(data, exps, nx, init, feedthrough, disturbance, initial_state, max_iter)
    attach_warnings(model, data)
    if Ts is not None and Ts == 0:
        if any(h == "foh" for h in data.intersample):
            # FOH girişli veride ZOH tersi yanlış olur: sürekli parametreler FOH ayrıklaştırmasıyla (değerlendirmeyle
            # aynı) öngörü hatası ölçütünde arıtılır; başlangıç ayrık optimumun FOH tersi
            init_ct = _foh_inverse(model, data.Ts, feedthrough)
            out = _ssest_mimo(data, exps, nx, init_ct, feedthrough, disturbance, initial_state, max_iter,
                              ct_holds=data.intersample)
            attach_warnings(out, data)
            return out
        return d2c_ss(model)
    return model


def _foh_inverse(md: IDSS, Ts: float, feedthrough: bool) -> IDSS:
    """Ayrık modeli FOH ayrıklaştırmasının tersiyle sürekliye çevirir: A = log(A_d)/Ts, B = W(A)⁻¹B_d,
    W = (e^{ATs} − I)·S2/Ts + S1 (S1 = ∫₀^Ts e^{Aσ}dσ, S2 = ∫₀^Ts e^{Aσ}(Ts−σ)dσ); K ZOH kuralıyla."""
    mc = d2c_ss(md)
    A = mc.A
    n = A.shape[0]
    M = np.zeros((3 * n, 3 * n))
    M[:n, :n] = A * Ts
    M[:n, n:2 * n] = np.eye(n) * Ts
    M[n:2 * n, 2 * n:] = np.eye(n)
    E = linalg.expm(M)
    Phi, S1, S2 = E[:n, :n], E[:n, n:2 * n], E[:n, 2 * n:] * Ts
    W = (Phi - np.eye(n)) @ S2 / Ts + S1
    B = np.linalg.solve(W, md.B)
    D = md.D - md.C @ S2 @ B / Ts if feedthrough else np.zeros_like(md.D)
    return IDSS(A, B, md.C.copy(), D, mc.K, 0.0, noise_variance=md.noise_variance)


def _c2d_idss(m, Ts, data):
    s = m.dss(Ts, data.intersample)
    return IDSS(s.A, s.B, s.C, s.D, s.K, Ts, noise_variance=m.noise_variance)


def _ssest_siso(data, nx, init: IDSS, feedthrough, initial_state, max_iter):
    from spt.control.sysid.linear import _PolyStructure, _run_pem

    nu = data.nu
    nk = np.zeros(nu, int) if feedthrough else np.ones(nu, int)
    nb = np.full(nu, nx + 1 if feedthrough else nx)
    st = _PolyStructure(nx, nb, nx, 0, np.zeros(nu, int), nk)
    # başlangıç: n4sid modelinin polinom biçimi
    A, B, C, D, K = init.A, init.B, init.C, init.D, init.K
    den = np.real(np.poly(A))
    nc_ = np.real(np.poly(A - K @ C))
    Bs = []
    for j in range(nu):
        num, _ = signal.ss2tf(A, B[:, [j]], C, D[:, [j]])
        num = np.real(num[0])
        Bs.append(num if feedthrough else np.r_[0.0, num[1:]])
    th0 = st.pack(den, Bs, nc_, [1.0], [[1.0]] * nu)
    inits = [th0]
    m = _run_pem(data, st, inits, initial_state, "SSEST", max_iter, {"nx": nx})
    s = m.dss(data.Ts)
    out = IDSS(s.A, s.B, s.C, s.D, s.K, data.Ts, noise_variance=m.noise_variance, input_names=data.input_names,
               output_names=data.output_names)
    rep = m.report
    rep.method = "SSEST"
    x0 = rep.x0
    if x0 is not None:
        # polinom serbest cevabından durum: aynı gerçeklemede e_free = −C Ãᵗ x0 → en küçük kareler
        Ab = s.A - s.K @ s.C
        Nn = max(4 * nx, 20)
        Psi = np.zeros((Nn, s.nx))
        P = s.C.copy()
        for t in range(Nn):
            Psi[t] = -P[0]
            P = P @ Ab
        Nc = np.real(np.poly(Ab))
        xs = []
        for sv in x0:
            imp = np.zeros(Nn)
            imp[:len(sv)] = sv
            fr = signal.lfilter([1.0], Nc, imp)
            xs.append(np.linalg.lstsq(Psi, fr, rcond=None)[0])
        rep.x0 = xs
        out.x0 = xs[0]
    out.report = rep
    out.covariance = None
    return out


def _ssest_mimo(data, exps, nx, init: IDSS, feedthrough, disturbance, initial_state, max_iter, ct_holds=None):
    """Serbest parametreli PEM. `ct_holds` verilirse parametreler sürekli (A, B, C, D, K) ve öngörücü verinin
    örnekler arası davranışıyla (ZOH/FOH) tam ayrıklaştırılmış modelden kurulur."""
    ny, nu = data.ny, data.nu
    A0, B0, C0, D0, K0 = (np.array(M, float) for M in (init.A, init.B, init.C, init.D, init.K))
    if not disturbance:
        K0 = np.zeros((nx, ny))
    shapes = [(nx, nx), (nx, nu), (ny, nx)] + ([(ny, nu)] if feedthrough else []) + ([(nx, ny)] if disturbance else [])

    def unpack(th):
        mats, k = [], 0
        for s in shapes:
            n = s[0] * s[1]
            mats.append(th[k:k + n].reshape(s, order="F"))
            k += n
        A, B, C = mats[0], mats[1], mats[2]
        i = 3
        D = mats[i] if feedthrough else np.zeros((ny, nu))
        i += 1 if feedthrough else 0
        K = mats[i] if disturbance else np.zeros((nx, ny))
        if ct_holds is not None:
            sd = IDSS(A, B, C, D, K, 0.0).dss(data.Ts, ct_holds)
            return sd.A, sd.B, sd.C, sd.D, sd.K
        return A, B, C, D, K

    def unpack_raw(th):
        mats, k = [], 0
        for s_ in shapes:
            n_ = s_[0] * s_[1]
            mats.append(th[k:k + n_].reshape(s_, order="F"))
            k += n_
        D = mats[3] if feedthrough else np.zeros((ny, nu))
        K = mats[3 + (1 if feedthrough else 0)] if disturbance else np.zeros((nx, ny))
        return mats[0], mats[1], mats[2], D, K

    th0 = np.concatenate([M.ravel(order="F") for M in ([A0, B0, C0] + ([D0] if feedthrough else [])
                                                       + ([K0] if disturbance else []))])
    nth = len(th0)
    if ct_holds is not None:
        A0, B0, C0, D0, K0 = unpack(th0)
    ic = initial_state
    if ic == "auto":
        Ez = _pred_errors(A0, B0, C0, D0, K0, exps, [np.zeros(nx)] * len(exps))
        x0e = list(init.report.x0) if (init.report and init.report.x0) else [np.zeros(nx)] * len(exps)
        Ee = _pred_errors(A0, B0, C0, D0, K0, exps, x0e)
        nz = np.linalg.norm(np.vstack(Ez))
        ne = np.linalg.norm(np.vstack(Ee))
        ic = "estimate" if ne > 0 and nz / ne > 1.05 else "zero"
    nx0 = nx * len(exps) if ic == "estimate" else 0
    W = np.eye(ny)

    def resid_fn(z, Wh):
        A, B, C, D, K = unpack(z[:nth])
        x0s = [z[nth + k * nx: nth + (k + 1) * nx] for k in range(len(exps))] if nx0 else [np.zeros(nx)] * len(exps)
        E = np.vstack(_pred_errors(A, B, C, D, K, exps, x0s))
        return (E @ Wh).ravel()

    def accept(z):
        A, B, C, D, K = unpack(z[:nth])
        try:
            return bool(np.max(np.abs(np.linalg.eigvals(A - K @ C))) < 1.0)
        except np.linalg.LinAlgError:
            return False

    x0init = (list(init.report.x0) if (init.report and init.report.x0) else [np.zeros(nx)] * len(exps))
    z = np.r_[th0, np.concatenate(x0init)] if nx0 else th0.copy()
    if not accept(z):
        A0s = A0.copy()
        z[:nth] = np.concatenate([M.ravel(order="F") for M in ([A0s, B0, C0] + ([D0] if feedthrough else [])
                                                              + ([np.zeros((nx, ny))] if disturbance else []))])
    res = None
    for _ in range(20):  # det ölçütü: ağırlık güncellemeli yinelemeler
        Wh = linalg.sqrtm(np.linalg.inv(W)).real if ny > 1 else np.eye(1)
        f = lambda zz: resid_fn(zz, Wh)  # noqa: E731
        res = levenberg_marquardt(f, z, lambda zz: fd_jacobian(f, zz, central=True), accept=accept,
                                  max_iter=max_iter)
        z = res.x
        A, B, C, D, K = unpack(z[:nth])
        x0s = [z[nth + k * nx: nth + (k + 1) * nx] for k in range(len(exps))] if nx0 else [np.zeros(nx)] * len(exps)
        E = np.vstack(_pred_errors(A, B, C, D, K, exps, x0s))
        Wn = E.T @ E / len(E)
        if np.max(np.abs(Wn - W)) <= 1e-10 * np.max(np.abs(W)):
            W = Wn
            break
        W = Wn
    A, B, C, D, K = unpack(z[:nth])
    x0s = [z[nth + k * nx: nth + (k + 1) * nx] for k in range(len(exps))] if nx0 else [np.zeros(nx)] * len(exps)
    Es = _pred_errors(A, B, C, D, K, exps, x0s)
    out_Ts = data.Ts
    if ct_holds is not None:  # rapor ayrık öngörücüden; model sürekli parametrelerle
        A, B, C, D, K = unpack_raw(z[:nth])
        out_Ts = 0.0
    npar = nx * (ny + nu) + (ny * nu if feedthrough else 0) + (nx * ny if disturbance else 0)
    fit = make_fit([y for y, _ in exps], Es, npar + nx0)
    Ntot = sum(len(e) for e in Es)
    Em = np.vstack(Es)
    lam, short_msg = noise_variance(Em.T @ Em, Ntot, npar + nx0)
    out = IDSS(A, B, C, D, K, out_Ts, x0=x0s[0], noise_variance=lam, input_names=data.input_names,
               output_names=data.output_names)
    out.report = Report("SSEST", fit, ic, res.iterations if res else 0, res.why if res else "", npar, nx0, x0s,
                        data.name, Ntot, data.Ts, {"nx": nx}, "SSEST ile tahmin edildi (öngörü hatası yöntemi)")
    if short_msg:
        out.report.warnings.append(short_msg)
    return out
