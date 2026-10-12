"""Parametrik olmayan ve yarı parametrik kestirimler: `covf`, `spa`, `etfe`, `impulseest`, `delayest`,
yapı seçimi `arxstruc` / `selstruc` (MATLAB R2025b tanımlarıyla; MATLAB ile sayısal olarak karşılaştırılmıştır).

- **spa** (Blackman–Tukey): R(τ) = (1/N) Σ z(t) z(t+τ)ᵀ, τ = 0…M−1; Hann gecikme penceresi
  W(τ) = ½(1 + cos(πτ/(M−1))); M varsayılanı min(30, N/10); frekanslar varsayılan (1:128)/128·π/Ts.
  G = Φ_yu Φ_u⁻¹, gürültü spektrumu Ts·(Φ_y − Φ_yu Φ_u⁻¹ Φ_uy) (MATLAB SpectrumData ölçeği).
- **etfe**: Y(ω)/U(ω) (DTFT oranı). Yumuşatmalı (M verilirse): nfft = 2·⌈N/Nω⌉·Nω noktalı FFT'de çapraz ve öz
  periodogramlar, genişliği ⌊nfft/(M/2)⌋ olan normalize Hamming penceresiyle dairesel olarak süzülür ve Nω
  frekansta örneklenir (MATLAB ile aynı tanım).
- **impulseest** (düzenlileştirilmiş FIR; Chen–Ohlsson–Ljung 2012): θ ~ N(0, P(η)), P çekirdeği TC
  (P_kj = c·β^max(k,j)), DC (c·β^((k+j)/2)·ρ^|k−j|), DI, SS, SE, CS, HF; σ² en küçük kareler FIR artığından;
  η negatif log marjinal olabilirliğin en küçüklenmesiyle (MATLAB'in ızgara başlangıcı ve sınırlarıyla).
  `ar_order` > 0 (MATLAB varsayılanı 5): girişe ek olarak −y regresörleriyle düzenlileştirilmiş ARX kestirilir ve
  B/A'nın dürtü cevabının ilk n örneği döndürülür.
- **delayest**: ARX[na nb nk] kayıplarının nk ∈ [nkmin, nkmax] üzerinde en küçüğü (arxstruc + selstruc(…, 0)).
- **arxstruc**: tüm yapılar için ortak regresyon başlangıcı (en büyük gecikme), kayıp = Σe²/N_toplam.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg, optimize, signal

from spt.control.errors import ControlError
from spt.control.sysid._util import shift
from spt.control.sysid.data import IDData, check_complete
from spt.control.sysid.models import IDFRD, IDPoly, Report, make_fit

__all__ = ["covf", "spa", "etfe", "impulseest", "delayest", "arxstruc", "selstruc", "struc"]


def _z(data: IDData):
    check_complete(data, "Analiz")
    return [np.hstack([y, u]) for y, u in zip(data.ys, data.us)]


def covf(data: IDData, M: int) -> np.ndarray:
    """Kovaryans fonksiyonu (MATLAB `covf`): R[τ, i, j] = (1/N) Σ_t z_i(t) z_j(t+τ), z = [y, u], τ = 0…M−1
    (deneyler toplanır, N toplam örnek sayısı)."""
    zs = _z(data)
    nz = zs[0].shape[1]
    N = sum(len(z) for z in zs)
    R = np.zeros((M, nz, nz))
    for z in zs:
        n = len(z)
        for tau in range(min(M, n)):
            R[tau] += z[:n - tau].T @ z[tau:]
    return R / N


def _default_w(Ts, n=128):
    return np.arange(1, n + 1) / n * np.pi / Ts


def spa(data: IDData, M: int | None = None, w=None) -> IDFRD:
    """Spektral analiz (Blackman–Tukey, Hann penceresi) — MATLAB `spa(data, M, w)`. `w` rad/s."""
    zs = _z(data)
    Ts = data.Ts
    Nmin = min(len(z) for z in zs)
    if M is None:
        M = int(min(30, np.floor(sum(len(z) for z in zs) / 10)))
    M = int(M)
    if M >= Nmin:
        M = Nmin - 1
    if M < 2:
        raise ControlError("Pencere uzunluğu M en az 2 olmalı")
    w = _default_w(Ts) if w is None else np.unique(np.atleast_1d(np.asarray(w, float)))
    from spt.control.sysid.data import excitation_warnings

    const = excitation_warnings(data)
    if const:
        raise ControlError("Spektral analiz yapılamaz: " + "; ".join(const))
    R = covf(data, M)  # (M, nz, nz), τ = 0…M−1
    ny, nu = data.ny, data.nu
    win = np.r_[1.0, 0.5 + 0.5 * np.cos(np.pi * np.arange(1, M) / (M - 1))]
    tau = np.arange(M)
    E = np.exp(-1j * np.outer(w * Ts, tau))  # (nw, M)
    Rw = R * win[:, None, None]
    # Φ_ij(ω) = Σ_τ W(τ)[R_ji(τ) e^{-iωτ} + R_ij(τ) e^{iωτ}] − W(0)R_ij(0)   (R_ij(τ) = E z_i(t) z_j(t+τ))
    pos = np.einsum("wt,tij->wji", E, Rw)  # Σ R_ji(τ) e^{-iωτ}  → eleman (i, j)
    neg = np.einsum("wt,tij->wij", np.conj(E), Rw)
    Phi = pos + neg - Rw[0][None, :, :]
    nw = len(w)
    resp = np.zeros((ny, nu, nw), complex)
    spect = np.zeros((ny, ny, nw))
    for k in range(nw):
        F = Phi[k]
        if nu:
            Fu = F[ny:, ny:]
            Fyu = F[:ny, ny:]
            try:
                G = np.linalg.solve(Fu.T, Fyu.T).T
            except np.linalg.LinAlgError:
                raise ControlError("Spektral analiz yapılamaz: giriş spektrumu tekil (girişler doğrusal bağımlı ya da "
                                   "yetersiz uyarım)") from None
            S = F[:ny, :ny] - G @ np.conj(Fyu.T)
            resp[:, :, k] = G
        else:
            S = F[:ny, :ny]
        spect[:, :, k] = np.abs(S.real) if ny == 1 else S.real
    spect *= Ts
    g = IDFRD(resp, w, Ts, spectrum=spect, method="SPA", input_names=data.input_names,
              output_names=data.output_names)
    g.window_size = M
    return g


def etfe(data: IDData, M: int | None = None, N: int = 128, *, w=None) -> IDFRD:
    """Deneysel transfer fonksiyonu kestirimi (MATLAB `etfe(data, M, N)`); zaman serisinde periodogram.

    `w` (rad/s; bizim eklentimiz — MATLAB zaman verisinde frekans vektörü almaz): yumuşatmasızsa tanım gereği
    tam DTFT oranı Σ Y·Ū / Σ |U|² bu frekanslarda; yumuşatmalıysa N = 1024 eşit ızgaradan doğrusal ara değer."""
    if w is not None:
        w = np.atleast_1d(np.asarray(w, float))
        if M is None and data.nu > 0:
            zs = _z(data)
            ny, nu = data.ny, data.nu
            num = np.zeros((len(w), ny, nu), complex)
            den = np.zeros((len(w), nu))
            for z in zs:
                E = np.exp(-1j * np.outer(w * data.Ts, np.arange(len(z))))
                Yf, Uf = E @ z[:, :ny], E @ z[:, ny:]
                num += Yf[:, :, None] * np.conj(Uf[:, None, :])
                den += np.abs(Uf) ** 2
            resp = np.transpose(num / den[:, None, :], (1, 2, 0))
            return IDFRD(resp, w, data.Ts, method="ETFE", input_names=data.input_names,
                         output_names=data.output_names)
        g = etfe(data, M, max(int(N), 1024))
        fr = g.frequency
        if data.nu == 0:
            sp = np.stack([[np.interp(w, fr, g.spectrum[i, j]) for j in range(data.ny)] for i in range(data.ny)])
            return IDFRD(np.zeros((data.ny, 0, len(w))), w, data.Ts, spectrum=sp, method="ETFE",
                         output_names=data.output_names)
        R = g.response
        resp = np.stack([[np.interp(w, fr, R[i, j].real) + 1j * np.interp(w, fr, R[i, j].imag)
                          for j in range(R.shape[1])] for i in range(R.shape[0])])
        return IDFRD(resp, w, data.Ts, method="ETFE", input_names=data.input_names, output_names=data.output_names)
    zs = _z(data)
    Ts = data.Ts
    ny, nu = data.ny, data.nu
    N = int(N)
    Mh = None if M is None else max(1.0, M / 2.0)
    Yd = np.zeros((N, ny, max(nu, 1)), complex) if nu else np.zeros((N, ny))
    Ud = np.zeros((N, nu))
    for z in zs:
        Ncap = len(z)
        nfft = 2 * int(np.ceil(Ncap / N)) * N
        L = nfft
        M1 = int(np.trunc(L / (L if Mh is None else Mh)))
        sc = L // (2 * N)
        y, u = z[:, :ny], z[:, ny:]
        if M1 > 1:
            ha = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(M1 + 1) / M1)
            ha = ha / (ha @ ha)
        idx = (M1 + M1 // 2 + sc) + sc * np.arange(N) - 1  # MATLAB 1 tabanlı → 0 tabanlı
        if nu == 0:
            Yp = np.abs(np.fft.fft(y, nfft, axis=0)) ** 2 * Ts
            if M1 > 1:
                Yp = np.vstack([Yp[L - M1 + 1:], Yp])
                Yp = signal.lfilter(ha, [1.0], Yp, axis=0)
            Yd += Yp[idx]
        else:
            Yf = np.fft.fft(y, nfft, axis=0)
            Uf = np.fft.fft(u, nfft, axis=0)
            if M1 > 1:
                Yf = np.vstack([Yf[L - M1 + 1:], Yf])
                Uf = np.vstack([Uf[L - M1 + 1:], Uf])
            C = Yf[:, :, None] * np.conj(Uf[:, None, :])
            P = np.abs(Uf) ** 2
            if M1 > 1:
                C = signal.lfilter(ha, [1.0], C, axis=0)
                P = signal.lfilter(ha, [1.0], P, axis=0)
            Yd = Yd + C[idx]
            Ud += P[idx]
    freqs = np.arange(1, N + 1) * np.pi / N / Ts
    Ntot = sum(len(z) for z in zs)
    if nu == 0:
        spect = np.zeros((ny, ny, N))
        for i in range(ny):
            spect[i, i] = Yd[:, i] / Ntot
        return IDFRD(np.zeros((ny, 0, N)), freqs, Ts, spectrum=spect, method="ETFE",
                     output_names=data.output_names)
    resp = np.zeros((ny, nu, N), complex)
    for i in range(ny):
        for j in range(nu):
            d = Ud[:, j].copy()
            zero = d == 0
            d[zero] = 1.0
            resp[i, j] = Yd[:, i, j] / d
            resp[i, j, zero] = np.inf
    g = IDFRD(resp, freqs, Ts, method="ETFE", input_names=data.input_names, output_names=data.output_names)
    g.window_size = M
    return g


# ============================================================================================ impulseest
def _kernel(name: str, h, n: int):
    """Normalize çekirdek matrisi K (P = c·K), indisler 1…n (MATLAB regul_fir tanımı)."""
    k = np.arange(1, n + 1)
    if name == "TC":
        t = abs(h[0]) ** k
        return np.minimum.outer(t, t)
    if name == "DC":
        return abs(h[0]) ** ((k[:, None] + k[None, :]) / 2) * h[1] ** np.abs(k[:, None] - k[None, :])
    if name == "DI":
        return np.diag(abs(h[0]) ** k)
    if name == "SS":
        t = abs(h[0]) ** k
        mn = np.minimum.outer(t, t)
        return np.outer(t, t) * mn / 2 - mn ** 3 / 6
    if name == "SE":
        return np.exp(-np.subtract.outer(k, k) ** 2 / (2 * h[0] ** 2))
    if name == "CS":
        mn = np.minimum.outer(k, k).astype(float)
        return np.outer(k, k) * mn / 2 - mn ** 3 / 6
    if name == "HF":
        t = abs(h[0]) ** k
        sgn = np.where((k[:, None] + k[None, :]) % 2 == 0, 1.0, -1.0)
        return sgn * np.minimum.outer(t, t)
    raise ControlError(f"Bilinmeyen düzenlileştirme çekirdeği: {name}")


_KBOUNDS = {"TC": ([0.72], [1 - np.sqrt(np.finfo(float).eps)]), "DI": ([0.72], [1 - np.sqrt(np.finfo(float).eps)]),
            "HF": ([0.72], [1 - np.sqrt(np.finfo(float).eps)]), "DC": ([0.74, -0.98], [1 - np.sqrt(np.finfo(float).eps), 0.98]),
            "SS": ([0.9], [0.999]), "SE": ([np.sqrt(np.finfo(float).eps)], [2.5]), "CS": ([], [])}


def _nglk(hp, Rt, Ney, kernel, sigma, blocks):
    """Negatif log marjinal olabilirlik (sabitler hariç) — MATLAB regul_fir `nglglklhd` ile aynı tanım.
    blocks: [(başlangıç, uzunluk, hiperparametre dilimi)]."""
    Nth = Rt.shape[1] - 1
    Pis = np.zeros((Nth, Nth))
    lgdet = 0.0
    for (s0, n, hs) in blocks:
        h = hp[hs]
        K = _kernel(kernel, h, n)
        try:
            U = linalg.cholesky(K, lower=False)
        except linalg.LinAlgError:
            return 1.0 / np.finfo(float).eps
        if not np.all(np.isfinite(U)) or np.any(np.diag(U) <= 0):
            return 1.0 / np.finfo(float).eps
        Pis[s0:s0 + n, s0:s0 + n] = np.sqrt(sigma / np.exp(h[-1])) * linalg.solve_triangular(
            U, np.eye(n), lower=False)
        lgdet += n * h[-1] + 2 * np.sum(np.log(np.abs(np.diag(U))))
    M = np.vstack([Rt, np.hstack([Pis.T, np.zeros((Nth, 1))])])
    R1 = linalg.qr(M, mode="r")[0][:Nth + 1]
    d = np.abs(np.diag(R1[:Nth, :Nth]))
    if np.any(d == 0):
        return 1.0 / np.finfo(float).eps
    return 0.5 * (R1[Nth, Nth] ** 2 / sigma + (Ney - Nth) * np.log(sigma) + 2 * np.sum(np.log(d)) + lgdet), R1, Pis


def _obj(hp, *a):
    r = _nglk(hp, *a)
    return r[0] if isinstance(r, tuple) else r


def _regul_fir(cols_blocks, Y, kernel, Ney):
    """cols_blocks: [Φ_b] (Ney × n_b) blok regresörleri. Döner (θ, hiperparametreler, σ²)."""
    Phi = np.hstack(cols_blocks)
    Nth = Phi.shape[1]
    Rt = linalg.qr(np.column_stack([Phi, Y]), mode="r")[0][:Nth + 1]
    X = np.linalg.pinv(Rt[:, :Nth]) @ Rt[:, -1]
    sigma = float(np.sum((Rt[:, -1] - Rt[:, :Nth] @ X) ** 2) / max(Ney - Nth, 1))
    K = kernel.upper()
    lbk, ubk = _KBOUNDS[K]
    betas = [1.0] if K == "CS" else [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    hpr = [-6, -4, -2, 0, 1, 2, 3, 4, 5]
    blocks = []
    hp0 = []
    lb, ub = [], []
    s0 = 0
    for b, Pb in enumerate(cols_blocks):
        n = Pb.shape[1]
        nh = (0 if K == "CS" else 1) + (1 if K == "DC" else 0) + 1
        sl = slice(len(hp0), len(hp0) + nh)
        lbi = list(lbk) + [-1e16]
        ubi = list(ubk) + [1e16 if K == "DC" else np.inf]
        # blok başına ızgara başlangıcı (yalnız bu bloğun regresörleriyle)
        Rti = linalg.qr(np.column_stack([Pb, Y]), mode="r")[0][:n + 1]
        best = None
        for be in betas:
            for c in hpr:
                h = ([be] if K != "CS" else []) + ([0.99] if K == "DC" else []) + [c]
                h = np.clip(h, lbi, ubi)
                v = _obj(np.asarray(h, float), Rti, Ney, K, sigma, [(0, n, slice(0, nh))])
                if best is None or v < best[0]:
                    best = (v, np.asarray(h, float))
        hstart = best[1]
        if len(cols_blocks) > 1:
            r = optimize.minimize(_obj, hstart, args=(Rti, Ney, K, sigma, [(0, n, slice(0, nh))]),
                                  method="L-BFGS-B", bounds=list(zip(lbi, ubi)),
                                  options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 2000})
            hstart = r.x
        hp0 += list(hstart)
        lb += lbi
        ub += ubi
        blocks.append((s0, n, sl))
        s0 += n
    hp0 = np.asarray(hp0, float)
    args = (Rt, Ney, K, sigma, blocks)
    r = optimize.minimize(_obj, hp0, args=args, method="L-BFGS-B", bounds=list(zip(lb, ub)),
                          options={"ftol": 1e-15, "gtol": 1e-11, "maxiter": 5000})
    hp = r.x
    # yerel arıtma: sınırlar içinde Nelder–Mead ile son doğrulama (eğim gürültüsüne karşı)
    r2 = optimize.minimize(lambda h: _obj(np.clip(h, lb, ub), *args), hp, method="Nelder-Mead",
                           options={"xatol": 1e-10, "fatol": 1e-13, "maxiter": 4000})
    if r2.fun < r.fun:
        hp = np.clip(r2.x, lb, ub)
    _, R1, _ = _nglk(hp, *args)
    th = linalg.solve_triangular(R1[:Nth, :Nth], R1[:Nth, Nth], lower=False)
    return th, hp, sigma


def impulseest(data: IDData, n: int | None = None, nk: int = 0, *, kernel: str = "TC", ar_order: int = 5,
               ) -> IDPoly:
    """Düzenlileştirilmiş FIR (dürtü cevabı) kestirimi (MATLAB `impulseest(data, n, nk)`).

    `n`: FIR uzunluğu (varsayılan MATLAB kuralı: ⌊min(N/max(nu,3) − nk − ar_order, 70)⌋), `nk`: ilk gecikme
    (varsayılan 0). `kernel`: 'TC' (varsayılan), 'DC', 'DI', 'SS', 'SE', 'CS', 'HF' ya da 'none' (düzenlileştirmesiz
    LS; MATLAB bu durumda girişe önceden beyazlatma uygular — burada uygulanmaz). `ar_order`: MATLAB
    Advanced.AROrder (varsayılan 5). Döner: FIR IDPoly (A = 1, B = dürtü cevabı; MATLAB idtf döndürür)."""
    if data.nu == 0:
        raise ControlError("Dürtü cevabı kestirimi giriş verisi ister")
    if data.ny != 1:
        raise ControlError("Dürtü cevabı kestirimi tek çıkış için (çıkış seçin)")
    zs = _z(data)
    nu = data.nu
    nk = int(nk)
    if nk < 0:
        raise ControlError("Nedensel olmayan (negatif nk) dürtü cevabı desteklenmiyor")
    Ncap = min(len(z) for z in zs)
    K = kernel.upper()
    if n is None:
        n = int(np.floor(min(Ncap / max(nu, 3) - nk - (ar_order if K != "NONE" else 0), 70)))
        if n < 1:
            raise ControlError("Veri dürtü cevabı kestirimi için çok kısa")
    n = int(n)
    nb = max(n, 3) if K != "NONE" else n
    fo = int(ar_order) if K != "NONE" else 0
    # regresörler: [−y gecikmeleri 1…fo] + giriş başına gecikmeler nk…nk+nb−1; satırlar t ≥ nmax
    lags_max = max(nk + nb - 1, fo)
    nmax = max(lags_max, 1)
    blocks, Ys = [[] for _ in range((1 if fo else 0) + nu)], []
    for z in zs:
        y = z[:, 0]
        U = z[:, 1:]
        b = 0
        if fo:
            blocks[0].append(np.column_stack([-shift(y, k) for k in range(1, fo + 1)])[nmax:])
            b = 1
        for j in range(nu):
            blocks[b + j].append(np.column_stack([shift(U[:, j], nk + k) for k in range(nb)])[nmax:])
        Ys.append(y[nmax:])
    Pbs = [np.vstack(bl) for bl in blocks]
    Y = np.concatenate(Ys)
    Ney = len(Y)
    if Ney <= sum(P.shape[1] for P in Pbs):
        raise ControlError("Veri dürtü cevabı kestirimi için çok kısa")
    if K == "NONE":
        th = np.linalg.lstsq(np.hstack(Pbs), Y, rcond=None)[0]
        hp, sigma = np.zeros(0), None
    else:
        th, hp, sigma = _regul_fir(Pbs, Y, K, Ney)
    if fo:
        a = np.r_[1.0, th[:fo]]
        off = fo
    else:
        a = np.array([1.0])
        off = 0
    Bs = []
    for j in range(nu):
        bj = th[off + j * nb: off + (j + 1) * nb]
        Bfull = np.r_[np.zeros(nk), bj]
        if fo:
            imp = signal.lfilter(Bfull, a, np.r_[1.0, np.zeros(nk + n - 1)])
            g = imp[nk:nk + n]
        else:
            g = bj[:n]
        Bs.append(np.r_[np.zeros(nk), g])
    # rapor: benzetim artığı, ilk nx örnek ölçülen çıkışla değiştirilir (MATLAB)
    nx = nk + n - 1
    e_list, y_list = [], []
    for z in zs:
        y = z[:, 0]
        yp = sum(signal.lfilter(Bs[j], [1.0], z[:, 1 + j]) for j in range(nu))
        yp[:nx] = y[:nx]
        e_list.append((y - yp).reshape(-1, 1))
        y_list.append(y.reshape(-1, 1))
    npar = n * nu
    fit = make_fit(y_list, e_list, npar)
    E = np.vstack(e_list)
    lam = float(E[:, 0] @ E[:, 0]) / (len(E) - npar)
    model = IDPoly([1.0], Bs if nu > 1 else Bs[0], Ts=data.Ts, nk=np.full((1, nu), nk), noise_variance=lam,
                   input_names=data.input_names, output_names=data.output_names)
    model.report = Report("IMPULSEEST", fit, "zero", 0, "", npar, 0, None, data.name, len(E), data.Ts,
                          {"n": n, "nk": nk, "kernel": K, "ar_order": fo})
    model.hyperparameters = hp
    model.reg_noise_variance = sigma
    return model


# ============================================================================================ yapı seçimi
def arxstruc(ze: IDData, zv: IDData, NN) -> np.ndarray:
    """ARX yapıları için kayıp fonksiyonları (MATLAB `arxstruc(ze, zv, NN)`), tek çıkış.

    NN satırları [na, nb(×nu), nk(×nu)]. Döner V: ilk satır kayıplar (Σe²/N_v), sonraki satırlar yapılar; son
    sütun [N_v, ẏ'y/N, …]. Tüm yapılar ortak başlangıç indeksini (en büyük gecikme) kullanır."""
    NN = np.atleast_2d(np.asarray(NN, float)).astype(int)
    for d in (ze, zv):
        if d.ny != 1:
            raise ControlError("arxstruc tek çıkışlı veri ister")
    nu = ze.nu
    if zv.nu != nu:
        raise ControlError("Tahmin ve doğrulama verilerinin giriş sayıları farklı")
    if NN.shape[1] != 1 + 2 * nu:
        raise ControlError(f"NN satırları [na, nb(×{nu}), nk(×{nu})] olmalı")
    na = NN[:, 0]
    nb = NN[:, 1:1 + nu] if nu else np.zeros((len(NN), 0), int)
    nk = NN[:, 1 + nu:] if nu else np.zeros((len(NN), 0), int)
    if nu:
        nbord = nb - (nk == 0)
        nmax = max(int(na.max()) + 1, int(nbord.max()))
    else:
        nmax = int(na.max()) + 1
    same = (ze.ne == zv.ne and all(a.shape == b.shape and np.array_equal(a, b)
                                   for a, b in zip(_z(ze), _z(zv))))

    def regress(d, row):
        rows, Ys = [], []
        for y, U in zip(d.ys, d.us):
            y = y[:, 0]
            cols = [-shift(y, k) for k in range(1, na[row] + 1)]
            for j in range(nu):
                cols += [shift(U[:, j], nk[row, j] + k) for k in range(nb[row, j])]
            P = np.column_stack(cols)[nmax:] if cols else np.zeros((len(y) - nmax, 0))
            rows.append(P)
            Ys.append(y[nmax:])
        return np.vstack(rows), np.concatenate(Ys)

    Nv = sum(zv.lengths)
    V = np.zeros((NN.shape[1] + 1, len(NN) + 1))
    v1 = float(sum(np.sum(y[nmax:, 0] ** 2) for y in zv.ys))
    jj = 0
    for r in range(len(NN)):
        if na[r] + nb[r].sum() == 0:
            continue
        P, Y = regress(ze, r)
        th = np.linalg.lstsq(P, Y, rcond=None)[0] if np.any(P.T @ Y) else np.zeros(P.shape[1])
        if same:
            loss = (float(Y @ Y) - float((P.T @ Y) @ th)) / sum(ze.lengths)
        else:
            Pv, Yv = regress(zv, r)
            loss = float(np.sum((Yv - Pv @ th) ** 2)) / Nv
        V[0, jj] = max(loss, np.finfo(float).eps)
        V[1:, jj] = NN[r]
        jj += 1
    V[0, jj] = Nv
    V[1, jj] = v1 / Nv
    return V[:, :jj + 1]


def struc(*ranges) -> np.ndarray:
    """Yapı matrisi (MATLAB `struc(NA, NB, NK)`): tüm bileşimler, son argüman en hızlı değişir."""
    from itertools import product

    if not ranges:
        raise ControlError("En az bir derece aralığı verilmeli")
    return np.array(list(product(*[np.atleast_1d(np.asarray(list(r) if not np.isscalar(r) else [r], int))
                                   for r in ranges])), int)


def selstruc(V: np.ndarray, c=0):
    """En iyi yapıyı seçer (MATLAB `selstruc(V, c)`): ölçüt V·(1 + α·d/N), d = na + Σnb;
    c = 'aic' → α = 2, 'mdl' → α = log N, sayı → α = c (0: en küçük kayıp). Döner (nn, Vmod)
    (Vmod ilk satırı log ölçüt)."""
    V = np.asarray(V, float)
    nl, nm = V.shape
    nu = (nl - 2) // 2
    Nc = V[0, nm - 1]
    if isinstance(c, str):
        cl = c.lower()
        if cl.startswith("a"):
            alpha = 2.0
        elif cl.startswith("m"):
            alpha = np.log(Nc)
        else:
            raise ControlError("Ölçüt 'aic', 'mdl' ya da sayı olmalı")
    else:
        alpha = float(c)
    Vmod = V[:, :nm - 1].copy()
    for k in range(nm - 1):
        d = V[1:nu + 2, k].sum()
        Vmod[0, k] = V[0, k] * (1 + alpha * d / Nc)
    sel = int(np.argmin(Vmod[0]))
    nn = V[1:2 + 2 * nu, sel].astype(int)
    Vmod[0] = np.log(Vmod[0])
    return nn, Vmod


def delayest(data: IDData, na: int = 2, nb=2, nkmin=0, nkmax=None, maxtest: int = 10000):
    """Gecikme (nk) kestirimi (MATLAB `delayest(data, na, nb, nkmin, nkmax)`); varsayılan nkmax = nkmin + 40.
    Çok girişte tüm gecikme bileşimleri denenir. Tek girişte int, çok girişte dizi döner."""
    if data.ny != 1:
        raise ControlError("Gecikme kestirimi tek çıkışlı veri ister")
    nu = data.nu
    if nu == 0:
        raise ControlError("Gecikme kestirimi giriş verisi ister")
    nb = np.atleast_1d(np.asarray(nb, int)) * np.ones(nu, int)
    nkmin = np.atleast_1d(np.asarray(nkmin, int)) * np.ones(nu, int)
    nkmax = nkmin + 40 if nkmax is None else np.atleast_1d(np.asarray(nkmax, int)) * np.ones(nu, int)
    if np.any(nkmin < 0):
        raise ControlError("nkmin negatif olamaz")
    if np.prod(nkmax - nkmin) > maxtest:
        raise ControlError("Denenecek gecikme bileşimi çok fazla")
    grids = np.meshgrid(*[np.arange(nkmin[j], nkmax[j] + 1) for j in range(nu)], indexing="ij")
    combos = np.column_stack([g.ravel(order="F") for g in grids])
    NN = np.column_stack([np.full(len(combos), na), np.tile(nb, (len(combos), 1)), combos])
    V = arxstruc(data, data, NN)
    nn, _ = selstruc(V, 0)
    nk = nn[1 + nu:]
    return int(nk[0]) if nu == 1 else nk
