"""Ayrıklaştırma ve gecikme işlemleri: c2d, d2c, d2d, pade, absorb_delay.

MATLAB karşılıkları (R2025b'den ölçülerek doğrulandı):
- `c2d(sys, Ts, method)`: 'zoh' (sıfırıncı derece tutucu), 'foh' (üçgen / birinci derece tutucu, nedensel olmayan
  biçimin durum dönüşümüyle nedensel gerçeklemesi), 'impulse' (dürtü değişmez: h_d[k] = Ts·h(kTs); MATLAB gibi D'nin
  Dirac bileşeni atılır), 'tustin' (iki doğrusal, isteğe bağlı `prewarp` rad/s), 'matched' (kutup-sıfır eşleme:
  z = e^{sTs}, sonsuzdaki fazla sıfırların (göreli derece − 1) tanesi z = −1'e, kazanç DC'de — integratör / türev
  varsa (z−1)/Ts ↔ s limitinde — eşlenir).
- Gecikmeler: 'zoh', 'foh' ve 'impulse' kesirli gecikmeyi de TAM ayrıklaştırır (modifiye z-dönüşümü; tamsayı kısım
  gecikme olarak kalır, kesirli kısım ek durum / modifiye C, D olarak modele girer — MATLAB'in sonucuyla aynı transfer).
  'tustin' ve 'matched'de gecikme en yakın örneğe yuvarlanır ve uyarı verilir (MATLAB'in varsayılanı gibi).
- `d2c(sys, method)`: 'zoh' (matris logaritması; negatif gerçel kutuplar MATLAB gibi derece artırılarak karmaşık
  çifte eşlenir; z = 0 civarındaki kutuplar hata verir), 'tustin', 'matched'.
- `d2d(sys, Ts_new, method)`: d2c + c2d ('zoh' ya da 'tustin').
- `pade(T, N)` → (num, den); `pade(sys, N)` sürekli modelin bütün gecikmelerini Padé yaklaşımıyla değiştirir.
- `absorb_delay(sys)`: ayrık modelin gecikmelerini z = 0'daki kutuplar / z⁻¹ durumları olarak modele katar.

Bilinen farklar: MATLAB kesirli giriş gecikmesini ss'de "iç gecikme" (InternalDelay) ile gösterir; burada aynı transfer
ek durumlarla gösterilir (durum sayısı farklı olabilir). 'tustin' / 'matched' için Thiran kesirli gecikme yaklaşımı
(FractDelayApproxOrder) yoktur.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import scipy.linalg as sla

from spt.control.errors import ControlError, ControlWarning
from spt.control.lti import (
    LTI, StateSpace, TransferFunction, ZPK, _blkdiag, cplxpair, ss_absorb)
from spt.control import convert

METHODS_C2D = ("zoh", "foh", "impulse", "tustin", "matched")
METHODS_D2C = ("zoh", "tustin", "matched")


# ---------------------------------------------------------------------------------------------------------------
# Tam integraller
# ---------------------------------------------------------------------------------------------------------------
def _integrals(A: np.ndarray, B: np.ndarray, h: float):
    """Φ(h) = e^{Ah}, Γ0(h) = ∫₀ʰ e^{Aσ}dσ·B, J(h) = ∫₀ʰ e^{Aσ}σ dσ·B (blok matris üsteliyle, tam)."""
    n, m = B.shape
    if h == 0.0:
        return np.eye(n), np.zeros((n, m)), np.zeros((n, m))
    M = np.zeros((n + 2 * m, n + 2 * m))
    M[:n, :n] = A * h
    M[:n, n:n + m] = B * h
    M[n:n + m, n + m:] = np.eye(m)
    E = sla.expm(M)
    Phi = E[:n, :n]
    G0 = E[:n, n:n + m]
    G1 = E[:n, n + m:]                      # = (1/h)∫₀ʰ e^{Aσ}(h−σ)dσ·B
    J = h * (G0 - G1)
    return Phi, G0, J


def _split_delay(tau: float, T: float):
    """τ = (l − μ)·T, l = ⌈τ/T⌉, μ ∈ [0, 1)."""
    r = tau / T
    l = math.ceil(r - 1e-9)
    mu = l - r
    if abs(mu) < 1e-9:
        mu = 0.0
    if mu < 0:
        mu = 0.0
    return int(l), float(mu)


def delay_taps(A: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, T: float, tau: float, method: str):
    """Tek girişli (b: n×1, d: p×1) sürekli sistemin, τ gecikmeli girişle tam ayrıklaştırma katsayıları.

    x[k+1] = Φ x[k] + Σ_o G[o]·u[k+o],  y[k] = C x[k] + Σ_o H[o]·u[k+o]  (o ≤ 1 tamsayı ofsetler).
    'zoh': u parça parça sabit; 'foh': u örnekler arasında doğrusal (MATLAB'in üçgen tutucusu)."""
    l, mu = _split_delay(tau, T)
    G: dict[int, np.ndarray] = {}
    H: dict[int, np.ndarray] = {}

    def add(D, o, v):
        if np.any(v):
            D[o] = D.get(o, 0.0) + v

    if method == "zoh":
        Pm, G0m, _ = _integrals(A, b, mu * T)
        _, G0r, _ = _integrals(A, b, (1.0 - mu) * T)
        add(G, -l + 1, G0m[:, 0])
        add(G, -l, (Pm @ G0r)[:, 0])
        add(H, -l, d[:, 0])
    elif method == "foh":
        _, G0m, Jm = _integrals(A, b, mu * T)
        _, G0T, JT = _integrals(A, b, T)
        I0a, I1a = G0T - G0m, JT - Jm               # σ ∈ (μT, T]
        I0b, I1b = G0m, Jm                          # σ ∈ (0, μT]
        wa = ((1.0 + mu) * T * I0a - I1a) / T
        wb = (mu * T * I0b - I1b) / T
        add(G, -l, (I0a - wa)[:, 0])
        add(G, -l + 1, (wa + I0b - wb)[:, 0])
        add(G, -l + 2, wb[:, 0])
        add(H, -l, (1.0 - mu) * d[:, 0])
        add(H, -l + 1, mu * d[:, 0])
    else:
        raise ControlError(f"Bilinmeyen tutucu: {method}")
    return G, H


def realize_taps(Phi: np.ndarray, C: np.ndarray, taps: list[tuple[dict, dict]]):
    """Ofsetli katsayılardan nedensel ayrık durum uzayı. Dönüş: (A, B, C, D, input_delays, Xmap).

    Her giriş j için tamsayı giriş gecikmesi d_j seçilir (ofsetler ≤ 1 / ≤ 0 olacak biçimde en büyük), +1 ofset
    durum dönüşümüyle (ξ = x − G[+1]·v) kaldırılır, negatif ofsetler z⁻¹ zinciriyle gerçeklenir.
    Xmap: (n, m) — özgün durum x = ξ + Xmap·v[k] (ilk n durum)."""
    n = Phi.shape[0]
    p = C.shape[0]
    m = len(taps)
    dly = np.zeros(m)
    shifted = []
    for j, (G, H) in enumerate(taps):
        og = max(G) if G else None
        oh = max(H) if H else None
        lo = min(list(G) + list(H)) if (G or H) else 0
        # A: bütün ofsetler ≤ 0 (dönüşümsüz); B: G'de +1 ofset (ξ dönüşümü) — daha az durum veren seçilir
        dA = min([-o for o in (og, oh) if o is not None], default=0)
        dB = min([v for v in ((1 - og) if og is not None else None, (-oh) if oh is not None else None)
                  if v is not None], default=0)
        qA = max(0, -(lo + dA))
        qB = max(0, -(lo + dB))
        if dA >= 0 and (qA <= qB or dB < 0):
            dj = dA
        else:
            dj = dB
        if dj < 0:
            raise ControlError("Nedensel olmayan ayrıklaştırma (iç hata).")
        dly[j] = dj
        shifted.append(({o + dj: v for o, v in G.items()}, {o + dj: v for o, v in H.items()}))
    Xmap = np.zeros((n, m))
    g0 = np.zeros((n, m))
    h0 = np.zeros((p, m))
    chains = []
    for j, (G, H) in enumerate(shifted):
        if 1 in G:
            Xmap[:, j] = G[1]
            g0[:, j] += Phi @ G[1]
            h0[:, j] += C @ G[1]
        g0[:, j] += G.get(0, 0.0)
        h0[:, j] += H.get(0, 0.0)
        q = -min([o for o in list(G) + list(H) if o < 0], default=0)
        chains.append((q, G, H))
    nq = sum(q for q, _, _ in chains)
    N = n + nq
    A = np.zeros((N, N))
    A[:n, :n] = Phi
    B = np.zeros((N, m))
    B[:n] = g0
    Cn = np.zeros((p, N))
    Cn[:, :n] = C
    off = n
    for j, (q, G, H) in enumerate(chains):
        if q == 0:
            continue
        B[off, j] = 1.0
        for i in range(1, q + 1):
            col = off + i - 1
            if i > 1:
                A[col, col - 1] = 1.0
            if -i in G:
                A[:n, col] += G[-i]
            if -i in H:
                Cn[:, col] += H[-i]
        off += q
    return A, B, Cn, h0, dly, Xmap


# ---------------------------------------------------------------------------------------------------------------
# c2d
# ---------------------------------------------------------------------------------------------------------------
def _round_half_up(x: np.ndarray) -> np.ndarray:
    return np.floor(np.asarray(x, float) + 0.5)


def _rounded_delays(sys: LTI, Ts: float, warn: bool = True):
    """'tustin' / 'matched': gecikmeleri en yakın örneğe yuvarla (MATLAB gibi), gerekirse uyar."""
    ind = sys.input_delay / Ts
    outd = sys.output_delay / Ts
    io = sys._io() / Ts
    r = [_round_half_up(v) for v in (ind, outd, io)]
    if warn and any(np.any(np.abs(a - b) > 1e-9) for a, b in zip((ind, outd, io), r)):
        warnings.warn("Gecikmeler en yakın örnekleme periyodu katına yuvarlandı. Zaman bölgesinde tam sonuç için "
                      "'zoh' ya da 'foh' yöntemini kullanın.", ControlWarning, stacklevel=3)
    return r


def _ss_c2d_core(sys: StateSpace, T: float, method: str) -> StateSpace:
    """Gecikmeli (giriş / çıkış, kesirli dahil) sürekli ss → ayrık ss: zoh / foh / impulse."""
    A, B, C, D = sys.A, sys.B, sys.C, sys.D
    n, m, p = A.shape[0], B.shape[1], C.shape[0]
    outd = sys.output_delay
    out_l = np.array([_split_delay(t, T)[0] for t in outd])
    out_mu = np.array([_split_delay(t, T)[1] for t in outd])
    in_frac = any(_split_delay(t, T)[1] > 0 for t in sys.input_delay)
    if np.any(out_mu > 0) and (method != "zoh" or in_frac):
        return _rowwise_c2d(sys, T, method)
    if method == "impulse":
        return _impulse_c2d(sys, T)
    Phi = sla.expm(A * T) if n else np.zeros((0, 0))
    taps = [delay_taps(A, B[:, [j]], C, D[:, [j]], T, float(sys.input_delay[j]), method) for j in range(m)]
    Ad, Bd, Cd, Dd, dly, _ = realize_taps(Phi, C, taps)
    od = outd / T
    if np.any(out_mu > 0):
        # zoh, tamsayı giriş gecikmeleri: kesirli çıkış gecikmesi C ve D'ye katılır (ek durum yok, MATLAB gibi)
        nn = Ad.shape[0]
        for i in range(p):
            if out_mu[i] == 0:
                continue
            Pm, G0m, _ = _integrals(A, B, out_mu[i] * T)
            Ci = np.zeros(nn)
            Ci[:n] = C[i] @ Pm
            Ci[n:] = Cd[i, n:]
            Cd[i] = Ci
            Dd[i] = Dd[i] + C[i] @ G0m
        od = out_l.astype(float)
    names = list(sys.state_names) + [""] * (Ad.shape[0] - n)
    return StateSpace(Ad, Bd, Cd, Dd, T, state_names=names, input_delay=dly, output_delay=np.round(od),
                      input_names=sys.input_names, output_names=sys.output_names, name=sys.name)


def _rowwise_c2d(sys: StateSpace, T: float, method: str) -> StateSpace:
    """Kesirli çıkış gecikmeleri: her çıkış satırını, çıkış gecikmesi girişlere eklenmiş olarak ayrı ayrıklaştır."""
    parts = []
    for i in range(sys.noutputs):
        row = StateSpace(sys.A, sys.B, sys.C[[i]], sys.D[[i]], 0.0, input_delay=sys.input_delay + sys.output_delay[i])
        parts.append(_ss_c2d_core(row, T, method))
    # ortak giriş gecikmesini dışarı al, kalanları duruma kat
    dmin = np.min(np.vstack([pt.input_delay for pt in parts]), axis=0)
    blocks = []
    for pt in parts:
        extra = pt.input_delay - dmin
        if np.any(extra):
            pt = ss_absorb(StateSpace(pt.A, pt.B, pt.C, pt.D, T, input_delay=extra), inputs=True, outputs=False)
        blocks.append(pt)
    A = _blkdiag(*[b.A for b in blocks])
    B = np.vstack([b.B for b in blocks])
    C = _blkdiag(*[b.C for b in blocks])
    D = np.vstack([b.D for b in blocks])
    return StateSpace(A, B, C, D, T, input_delay=dmin, input_names=sys.input_names, output_names=sys.output_names,
                      name=sys.name)


def _impulse_c2d(sys: StateSpace, T: float) -> StateSpace:
    """Dürtü değişmez: h_d[k] = T·C e^{A(kT − τ)} B (kT ≥ τ); D (Dirac) atılır (MATLAB gibi)."""
    A, B, C = sys.A, sys.B, sys.C
    n, m, p = A.shape[0], B.shape[1], C.shape[0]
    if n == 0:
        return StateSpace(np.zeros((0, 0)), np.zeros((0, m)), np.zeros((p, 0)), np.zeros((p, m)), T,
                          input_names=sys.input_names, output_names=sys.output_names, name=sys.name)
    Phi = sla.expm(A * T)
    splits = [_split_delay(float(t), T) for t in sys.input_delay]
    out_splits = [_split_delay(float(t), T) for t in sys.output_delay]
    if any(mu > 0 for _, mu in out_splits):
        raise ControlError("İç hata: kesirli çıkış gecikmesi satır satır işlenmeli.")
    groups: dict[float, list[int]] = {}
    for j, (_, mu) in enumerate(splits):
        groups.setdefault(mu, []).append(j)
    As, Bs, Cs, Ds = [], [], [], []
    for mu, cols in groups.items():
        Em = sla.expm(A * mu * T)
        Bg = np.zeros((n, m))
        Dg = np.zeros((p, m))
        Bg[:, cols] = T * Phi @ B[:, cols]
        Dg[:, cols] = T * C @ Em @ B[:, cols]
        As.append(Phi)
        Bs.append(Bg)
        Cs.append(C @ Em)
        Ds.append(Dg)
    Ad = _blkdiag(*As)
    Bd = np.vstack(Bs)
    Cd = np.hstack(Cs)
    Dd = sum(Ds)
    dly = np.array([l for l, _ in splits], float)
    return StateSpace(Ad, Bd, Cd, Dd, T, input_delay=dly, output_delay=[l for l, _ in out_splits],
                      input_names=sys.input_names, output_names=sys.output_names, name=sys.name)


def _tustin_ss(A, B, C, D, c):
    n = A.shape[0]
    if n == 0:
        return A, B, C, D
    M = np.linalg.solve(c * np.eye(n) - A, np.eye(n))
    Ad = M @ (c * np.eye(n) + A)
    r = math.sqrt(2.0 * c)
    return Ad, r * M @ B, r * C @ M, D + C @ M @ B


def _tustin_c(Ts: float, prewarp) -> float:
    if prewarp is None or prewarp == 0:
        return 2.0 / Ts
    w = float(prewarp)
    if not (0 < w < math.pi / Ts):
        raise ControlError("Ön bükme (prewarp) frekansı 0 < ω < π/Ts aralığında olmalı.")
    return w / math.tan(w * Ts / 2.0)


def _matched_entry(z, p, k, T):
    """Sürekli SISO (z, p, k) → ayrık, kutup-sıfır eşleme (MATLAB 'matched')."""
    if k == 0.0:
        return np.zeros(0, complex), np.zeros(0, complex), 0.0
    if z.size > p.size:
        raise ControlError("'matched' yöntemi uygun olmayan (sıfır sayısı > kutup sayısı) modele uygulanamaz.")
    zd = np.exp(np.asarray(z) * T)
    pd = np.exp(np.asarray(p) * T)
    extra = p.size - z.size - 1
    if extra > 0:
        zd = np.concatenate([zd, -np.ones(extra)])
    tol = 1e-12
    z0 = np.abs(z) <= tol
    p0 = np.abs(p) <= tol
    q = int(p0.sum() - z0.sum())
    dc = k * np.prod(-z[~z0]) / np.prod(-p[~p0])
    zz = np.abs(zd - 1.0) <= 1e-12
    pp = np.abs(pd - 1.0) <= 1e-12
    kd = dc * T ** q * np.prod(1.0 - pd[~pp]) / np.prod(1.0 - zd[~zz])
    return cplxpair(zd, tol=1e-9), cplxpair(pd, tol=1e-9), float(np.real(kd))


def c2d(sys: LTI, Ts: float, method: str = "zoh", prewarp: float | None = None) -> LTI:
    """Sürekli modeli ayrıklaştırır (MATLAB `c2d`); sonuç aynı türdedir (tf / zpk / ss).

    method: 'zoh' | 'foh' | 'impulse' | 'tustin' | 'matched'; prewarp (rad/s) yalnız 'tustin' için."""
    if not isinstance(sys, LTI):
        raise ControlError("c2d bir model bekler.")
    if sys.is_discrete:
        raise ControlError("Model zaten ayrık; yeniden örnekleme için d2d kullanın.")
    Ts = float(Ts)
    if not (Ts > 0 and math.isfinite(Ts)):
        raise ControlError("Örnekleme zamanı Ts > 0 olmalı.")
    method = str(method).lower()
    if method not in METHODS_C2D:
        raise ControlError(f"Bilinmeyen yöntem: {method!r} ({', '.join(METHODS_C2D)}).")
    if prewarp is not None and method != "tustin":
        raise ControlError("Ön bükme (prewarp) yalnız 'tustin' yönteminde kullanılır.")
    if method in ("zoh", "foh", "impulse"):
        if isinstance(sys, StateSpace):
            return _ss_c2d_core(sys, Ts, method)
        return _entrywise(sys, lambda e: _ss_c2d_core(e, Ts, method), Ts)
    rin, rout, rio = _rounded_delays(sys, Ts)
    if method == "tustin":
        c = _tustin_c(Ts, prewarp)
        s = convert.to_ss(_strip_delays(sys))
        Ad, Bd, Cd, Dd = _tustin_ss(s.A, s.B, s.C, s.D, c)
        out = StateSpace(Ad, Bd, Cd, Dd, Ts, input_names=sys.input_names, output_names=sys.output_names,
                         name=sys.name)
        out = _as_type(out, sys)
    else:   # matched
        if not sys.is_siso:
            raise ControlError("'matched' yöntemi yalnız SISO modeli destekler (MATLAB gibi).")
        z = convert.to_zpk(_strip_delays(sys))
        ny, nu = z.shape
        Z = [[None] * nu for _ in range(ny)]
        P = [[None] * nu for _ in range(ny)]
        K = np.zeros((ny, nu))
        for i in range(ny):
            for j in range(nu):
                Z[i][j], P[i][j], K[i, j] = _matched_entry(z._z[i][j], z._p[i][j], float(z._k[i, j]), Ts)
        out = _as_type(ZPK(Z, P, K, Ts, input_names=sys.input_names, output_names=sys.output_names,
                           name=sys.name), sys)
    out.input_delay = rin
    out.output_delay = rout
    if isinstance(out, (TransferFunction, ZPK)):
        out.io_delay = rio
    return out


def _strip_delays(sys: LTI) -> LTI:
    out = sys.copy()
    out.input_delay = np.zeros(sys.ninputs)
    out.output_delay = np.zeros(sys.noutputs)
    if isinstance(out, (TransferFunction, ZPK)):
        out.io_delay = np.zeros(sys.shape)
    return out


def _as_type(s: LTI, like: LTI) -> LTI:
    if isinstance(like, TransferFunction):
        return convert.to_tf(s)
    if isinstance(like, ZPK):
        return convert.to_zpk(s)
    return convert.to_ss(s)


def _entrywise(sys: LTI, fn, Ts: float) -> LTI:
    """tf / zpk: her eleman (toplam gecikmesiyle) ayrı ayrıklaştırılır; sonuç aynı türde, io gecikmeli."""
    ny, nu = sys.shape
    tot = sys.total_delay()
    is_tf = isinstance(sys, TransferFunction)
    N = [[None] * nu for _ in range(ny)]
    Dn = [[None] * nu for _ in range(ny)]
    K = np.zeros((ny, nu))
    io = np.zeros((ny, nu))
    for i in range(ny):
        for j in range(nu):
            e = _strip_delays(sys[i, j])
            se = convert.to_ss(e)
            se.input_delay = np.array([tot[i, j]])
            d = fn(se)
            dz = convert.to_zpk(StateSpace(d.A, d.B, d.C, d.D, Ts))
            io[i, j] = float(d.input_delay[0] + d.output_delay[0]) if dz._k[0, 0] != 0 else 0.0
            if is_tf:
                t = convert.to_tf(dz)
                N[i][j], Dn[i][j] = t._num[0][0], t._den[0][0]
            else:
                N[i][j], Dn[i][j], K[i, j] = dz._z[0][0], dz._p[0][0], dz._k[0, 0]
    meta = dict(input_names=sys.input_names, output_names=sys.output_names, name=sys.name)
    if is_tf:
        return TransferFunction(N, Dn, Ts, io_delay=io, **meta)
    return ZPK(N, Dn, K, Ts, io_delay=io, **meta)


# ---------------------------------------------------------------------------------------------------------------
# d2c / d2d
# ---------------------------------------------------------------------------------------------------------------
def _real_log(Ad: np.ndarray, T: float):
    """Gerçel matris logaritması; negatif gerçel özdeğerler için durum ikizlenir (MATLAB d2c 'zoh' gibi).

    Dönüş: (L, W, Wi, nneg) — x̃ = Wi·x dönüşümünden sonra ilk nneg durum negatif kutuplu blok; L, ikizlenmiş
    (n + nneg) boyutlu dönüşmüş Ad'nin gerçel logaritması."""
    n = Ad.shape[0]
    ev = np.linalg.eigvals(Ad)
    scale = max(1.0, float(np.max(np.abs(ev)))) if n else 1.0
    if n and np.min(np.abs(ev)) <= 1e3 * np.finfo(float).eps * scale:
        raise ControlError("'zoh' yöntemi z = 0 yakınında kutbu olan ayrık modele uygulanamaz (MATLAB gibi); "
                           "'tustin' ya da 'matched' deneyin.")
    neg = (np.abs(ev.imag) <= 1e-10 * np.abs(ev)) & (ev.real < 0)
    if not np.any(neg):
        L = sla.logm(Ad)
        L = np.real_if_close(L, tol=1e6)
        if np.iscomplexobj(L):
            raise ControlError("Matris logaritması gerçel değil; sürekli eşdeğer bulunamadı.")
        return np.asarray(L, float), np.eye(n), np.eye(n), 0
    S, Z, k = sla.schur(Ad, output="real", sort=lambda x, y: (abs(y) <= 1e-10 * abs(complex(x, y))) and x < 0)
    S11, S12, S22 = S[:k, :k], S[:k, k:], S[k:, k:]
    X = sla.solve_sylvester(S11, -S22, -S12) if 0 < k < n else np.zeros((k, n - k))
    E = np.eye(n)
    E[:k, k:] = X
    Ei = np.eye(n)
    Ei[:k, k:] = -X
    W, Wi = Z @ E, Ei @ Z.T
    LP = np.real_if_close(sla.logm(-S11), tol=1e6)
    L22 = np.real_if_close(sla.logm(S22), tol=1e6) if n - k else np.zeros((0, 0))
    if np.iscomplexobj(LP) or np.iscomplexobj(L22):
        raise ControlError("Matris logaritması gerçel değil; sürekli eşdeğer bulunamadı.")
    Ik = np.eye(k)
    Lneg = np.block([[LP, math.pi * Ik], [-math.pi * Ik, LP]])
    L = _blkdiag(Lneg, np.asarray(L22, float))
    return np.asarray(L, float), W, Wi, k


def _d2c_zoh_ss(s: StateSpace) -> StateSpace:
    T = s.Ts
    n = s.nx
    if n == 0:
        return StateSpace(s.A, s.B, s.C, s.D, 0.0)
    L, W, Wi, k = _real_log(s.A, T)
    Bt = Wi @ s.B
    Ct = s.C @ W
    if k:
        Bt = np.vstack([Bt[:k], np.zeros((k, Bt.shape[1])), Bt[k:]])
        Ct = np.hstack([Ct[:, :k], np.zeros((Ct.shape[0], k)), Ct[:, k:]])
    A = L / T
    N = A.shape[0]
    _, G0, _ = _integrals(A, np.eye(N), T)
    B = np.linalg.solve(G0, Bt)
    return StateSpace(A, B, Ct, s.D, 0.0)


def _d2c_tustin_ss(A, B, C, D, c):
    n = A.shape[0]
    if n == 0:
        return A, B, C, D
    I = np.eye(n)
    if np.min(np.abs(np.linalg.eigvals(A) + 1.0)) <= 1e-12:
        raise ControlError("'tustin' ters dönüşümü z = −1'de kutbu olan modele uygulanamaz.")
    N = np.linalg.solve(I + A, I)
    Ac = c * N @ (A - I)
    r = math.sqrt(2.0 * c)
    return Ac, r * N @ B, r * C @ N, D - C @ N @ B


def _unmatched_entry(z, p, k, T):
    if k == 0.0:
        return np.zeros(0, complex), np.zeros(0, complex), 0.0
    tol = 1e-12

    def lg(r, what):
        r = np.asarray(r, complex)
        bad = (np.abs(r) <= tol) | ((np.abs(r.imag) <= 1e-12 * np.abs(r)) & (r.real < 0))
        if np.any(bad):
            raise ControlError(f"'matched' ters dönüşümü z = 0 ya da negatif gerçel eksendeki {what} için "
                               "tanımsız.")
        return np.log(r) / T

    z = np.asarray(z, complex)
    p = np.asarray(p, complex)
    zm1 = np.abs(z + 1.0) <= 1e-9
    zc = lg(z[~zm1], "sıfır")
    pc = lg(p, "kutup")
    z1 = np.abs(z - 1.0) <= 1e-12
    p1 = np.abs(p - 1.0) <= 1e-12
    q = int(p1.sum() - z1.sum())
    dcd = k * np.prod(1.0 - z[~z1]) / np.prod(1.0 - p[~p1])
    z0 = np.abs(zc) <= tol
    p0 = np.abs(pc) <= tol
    kc = dcd / T ** q * np.prod(-pc[~p0]) / np.prod(-zc[~z0])
    return cplxpair(zc, tol=1e-9), cplxpair(pc, tol=1e-9), float(np.real(kc))


def d2c(sys: LTI, method: str = "zoh", prewarp: float | None = None) -> LTI:
    """Ayrık modeli sürekli zamana çevirir (MATLAB `d2c`): 'zoh' | 'tustin' | 'matched'. Gecikmeler n·Ts saniye."""
    if not isinstance(sys, LTI):
        raise ControlError("d2c bir model bekler.")
    if not sys.is_discrete:
        raise ControlError("Model zaten sürekli zamanlı.")
    method = str(method).lower()
    if method not in METHODS_D2C:
        raise ControlError(f"Bilinmeyen yöntem: {method!r} ({', '.join(METHODS_D2C)}).")
    T = sys.Ts
    base = _strip_delays(sys)
    if method == "zoh":
        out = _as_type(_d2c_zoh_ss(convert.to_ss(base)), sys)
    elif method == "tustin":
        s = convert.to_ss(base)
        A, B, C, D = _d2c_tustin_ss(s.A, s.B, s.C, s.D, _tustin_c(T, prewarp))
        out = _as_type(StateSpace(A, B, C, D, 0.0), sys)
    else:
        if not sys.is_siso:
            raise ControlError("'matched' yöntemi yalnız SISO modeli destekler (MATLAB gibi).")
        z = convert.to_zpk(base)
        ny, nu = z.shape
        Z = [[None] * nu for _ in range(ny)]
        P = [[None] * nu for _ in range(ny)]
        K = np.zeros((ny, nu))
        for i in range(ny):
            for j in range(nu):
                Z[i][j], P[i][j], K[i, j] = _unmatched_entry(z._z[i][j], z._p[i][j], float(z._k[i, j]), T)
        out = _as_type(ZPK(Z, P, K, 0.0), sys)
    out.input_names, out.output_names, out.name = sys.input_names, sys.output_names, sys.name
    out.input_delay = sys.input_delay * T
    out.output_delay = sys.output_delay * T
    if isinstance(out, (TransferFunction, ZPK)):
        out.io_delay = sys._io() * T
    return out


def d2d(sys: LTI, Ts: float, method: str = "zoh") -> LTI:
    """Ayrık modeli yeni örnekleme zamanına taşır (MATLAB `d2d`): d2c + c2d ('zoh' ya da 'tustin')."""
    method = str(method).lower()
    if method not in ("zoh", "tustin"):
        raise ControlError("d2d yöntemi 'zoh' ya da 'tustin' olmalı.")
    return c2d(d2c(sys, method), Ts, method)


# ---------------------------------------------------------------------------------------------------------------
# Padé ve gecikme emme
# ---------------------------------------------------------------------------------------------------------------
def _pade_coeffs(T: float, N: int):
    """MATLAB `[num, den] = pade(T, N)`: e^{-sT} ≈ num/den (payda monik)."""
    N = int(N)
    if N < 0:
        raise ControlError("Padé derecesi N ≥ 0 olmalı.")
    if T < 0:
        raise ControlError("Gecikme negatif olamaz.")
    if T == 0 or N == 0:
        return np.ones(1), np.ones(1)
    a = np.zeros(N + 1)
    b = np.zeros(N + 1)
    a[0] = b[0] = 1.0
    for k in range(1, N + 1):
        fact = T * (N - k + 1) / (2 * N - k + 1) / k
        a[k] = -fact * a[k - 1]
        b[k] = fact * b[k - 1]
    a = a[::-1] / b[N]
    b = b[::-1] / b[N]
    return a, b


def pade(T, N: int = 1):
    """Padé yaklaşımı (MATLAB `pade`).

    - `pade(T, N)` (T sayı) → (num, den): e^{−sT} ≈ num(s)/den(s), N. derece.
    - `pade(sys, N)` → sürekli modelin bütün gecikmeleri (giriş, çıkış, io) N. derece Padé ile değiştirilmiş model
      (aynı tür). Ayrık modelde gecikme zaten tamdır: absorb_delay kullanın."""
    if not isinstance(T, LTI):
        return _pade_coeffs(float(T), N)
    sys = T
    if sys.is_discrete:
        raise ControlError("Padé yaklaşımı sürekli zaman içindir; ayrık modelde absorb_delay kullanın.")
    if not sys.has_delay():
        return sys.copy()
    ny, nu = sys.shape
    base = _strip_delays(sys)
    if isinstance(sys, StateSpace):
        out = base
        for j in range(nu):
            if sys.input_delay[j] > 0:
                n_, d_ = _pade_coeffs(float(sys.input_delay[j]), N)
                blk = [convert.to_ss(TransferFunction(n_, d_)) if k == j else 1.0 for k in range(nu)]
                out = out * _append_ss(blk)
        for i in range(ny):
            if sys.output_delay[i] > 0:
                n_, d_ = _pade_coeffs(float(sys.output_delay[i]), N)
                blk = [convert.to_ss(TransferFunction(n_, d_)) if k == i else 1.0 for k in range(ny)]
                out = _append_ss(blk) * out
        out.input_names, out.output_names, out.name = sys.input_names, sys.output_names, sys.name
        return out
    is_tf = isinstance(sys, TransferFunction)
    work = convert.to_tf(base) if is_tf else convert.to_zpk(base)
    for (vec, side) in ((sys.input_delay, "in"), (sys.output_delay, "out")):
        for idx, t in enumerate(vec):
            if t <= 0:
                continue
            n_, d_ = _pade_coeffs(float(t), N)
            P = TransferFunction(n_, d_)
            P = P if is_tf else convert.to_zpk(P)
            if side == "in":
                for i in range(ny):
                    work = _set_entry(work, i, idx, work[i, idx] * P)
            else:
                for j in range(nu):
                    work = _set_entry(work, idx, j, P * work[idx, j])
    for i in range(ny):
        for j in range(nu):
            t = float(sys._io()[i, j])
            if t > 0:
                n_, d_ = _pade_coeffs(t, N)
                P = TransferFunction(n_, d_)
                P = P if is_tf else convert.to_zpk(P)
                work = _set_entry(work, i, j, work[i, j] * P)
    work.input_names, work.output_names, work.name = sys.input_names, sys.output_names, sys.name
    return work


def _append_ss(blocks) -> StateSpace:
    ss_blocks = [b if isinstance(b, StateSpace) else StateSpace(np.zeros((0, 0)), None, None, np.array([[b]]))
                 for b in blocks]
    A = _blkdiag(*[b.A for b in ss_blocks])
    B = _blkdiag(*[b.B for b in ss_blocks])
    C = _blkdiag(*[b.C for b in ss_blocks])
    D = _blkdiag(*[b.D for b in ss_blocks])
    return StateSpace(A, B, C, D, 0.0)


def _set_entry(sys, i, j, e):
    out = sys.copy()
    if isinstance(out, TransferFunction):
        out._num[i][j], out._den[i][j] = e._num[0][0], e._den[0][0]
    else:
        out._z[i][j], out._p[i][j], out._k[i, j] = e._z[0][0], e._p[0][0], e._k[0, 0]
    return out


def absorb_delay(sys: LTI) -> LTI:
    """Ayrık modelin bütün gecikmelerini modele katar (MATLAB `absorbDelay`): tf / zpk'de z = 0'daki kutuplar,
    ss'de z⁻¹ durumları. Sürekli modelde gecikme rasyonel değildir: pade kullanın."""
    if not sys.is_discrete:
        raise ControlError("Sürekli zamanda gecikme modele katılamaz (rasyonel değil); Padé yaklaşımı (pade) "
                           "kullanın.")
    if isinstance(sys, StateSpace):
        return ss_absorb(sys)
    tot = sys.total_delay()
    out = _strip_delays(sys)
    ny, nu = sys.shape
    for i in range(ny):
        for j in range(nu):
            d = int(round(tot[i, j]))
            if d <= 0:
                continue
            if isinstance(out, TransferFunction):
                if np.any(out._num[i][j]):
                    out._den[i][j] = np.concatenate([out._den[i][j], np.zeros(d)])
            elif out._k[i, j] != 0:
                out._p[i][j] = np.concatenate([out._p[i][j], np.zeros(d, complex)])
    return out


__all__ = ["c2d", "d2c", "d2d", "pade", "absorb_delay", "delay_taps", "realize_taps"]
