"""Zaman cevabı: step, impulse, initial, lsim, stepinfo, lsiminfo (MATLAB Control System Toolbox karşılığı).

Dönüş şekilleri (sözleşme): step / impulse → y (nt, ny, nu), t (nt,), isteğe bağlı x (nt, nx, nu);
initial / lsim → y (nt, ny), t, isteğe bağlı x (nt, nx).

Doğruluk: sürekli sistemler ızgara üzerinde TAM hesaplanır (blok matris üsteliyle tam ayrıklaştırma; adım ve dürtü
girişi için yaklaşıklık yoktur; lsim'de giriş 'zoh' (örnekler arası sabit) ya da 'foh' (örnekler arası doğrusal)
kabulüyle tamdır). Kesirli dahil bütün giriş / çıkış / G/Ç gecikmeleri tamdır.

MATLAB'den ölçülen tanımlar:
- `impulse`: sürekli zamanda D'nin Dirac bileşeni gösterilmez (y = C e^{At} B); ayrık zamanda birim alanlı darbe
  (u[0] = 1/Ts) cevabı (MATLAB gibi: 1/(z − 0.5), Ts = 0.1 → 0, 10, 5, 2.5 …).
- `stepinfo(y, t)` (dizi): yinit varsayılanı 0, yfinal varsayılanı y'nin son değeri; RiseTime eşik geçişleri örnekler
  arası doğrusal aradeğerle; SettlingTime / TransientTime son bant dışı örnekten sonraki bant geçişinin aradeğeri;
  SettlingTime bandı eşik·|yfinal − yinit|, TransientTime bandı eşik·max|y − yfinal|; SettlingMin / Max yükselme
  (üst eşik) örneğinden itibaren; Peak = max|y − yinit|; `t` verilmezse t = 1, 2, …, n (MATLAB gibi).
- `stepinfo(sys)`: MATLAB kendi ızgarasında aradeğerle hesaplar (ör. ζ = 0.3 için aşım %37.141, kesin değer
  %37.233). Burada sürekli sistem için TAM hesaplanır: eşik geçişleri ve uç değerler kesin cevap üzerinde kök bulma /
  yerel en iyileme ile inceltilir. Aşımsız (tekdüze) cevapta tepe, sonlu bir anda değil sonsuzda erişilen
  |yfinal − yinit| sınırıdır: Peak bu değer, PeakTime = ∞ (MATLAB ızgara sonunu verir). Ayrık sistemde değerler
  örnek anlarındadır (aradeğer yok, MATLAB gibi). Kararsız / sönümsüz / integratörlü sistemde MATLAB gibi NaN ve
  Peak = PeakTime = ∞.
- Varsayılan zaman ızgarası bizim seçimimizdir (MATLAB'inki yayımlanmamış); karşılaştırmada açık ızgara verin.
"""
from __future__ import annotations

import math

import numpy as np
import scipy.linalg as sla
from scipy.optimize import brentq, minimize_scalar

from spt.control.errors import ControlError
from spt.control.lti import LTI, StateSpace
from spt.control import convert
from spt.control.discretize import delay_taps, _integrals

MAX_POINTS = 200_000


# ---------------------------------------------------------------------------------------------------------------
# Zaman ızgarası
# ---------------------------------------------------------------------------------------------------------------
def _poles_of(sys: LTI) -> np.ndarray:
    s = convert.to_ss(sys) if not isinstance(sys, StateSpace) else sys
    s = convert.sminreal(_nodelay(s))
    return np.linalg.eigvals(s.A) if s.nx else np.zeros(0, complex)


def _nodelay(sys: LTI) -> LTI:
    out = sys.copy()
    out.input_delay = np.zeros(sys.ninputs)
    out.output_delay = np.zeros(sys.noutputs)
    if hasattr(out, "io_delay"):
        out.io_delay = np.zeros(sys.shape)
    return out


def _auto_tfinal(sys: LTI) -> float:
    """Sürekli eşdeğer kutuplardan yeterli benzetim süresi (bizim seçimimiz)."""
    p = _poles_of(sys)
    if sys.is_discrete and p.size:
        with np.errstate(divide="ignore", invalid="ignore"):  # z = 0 kutbu: log 0 = −∞ (aşağıda atılır)
            p = np.log(p.astype(complex)) / sys.Ts
        p = p[np.isfinite(p)]
    tmax = float(np.max(sys.total_delay())) if sys.has_delay() else 0.0
    cands = []
    if p.size:
        sig = -p.real
        stable = sig > 1e-9 * np.maximum(1.0, np.abs(p))
        if np.any(stable):
            cands.append(8.0 / float(np.min(sig[stable])))
        unst = ~stable
        if np.any(unst):
            mags = np.abs(p[unst])
            grow = p[unst].real
            if np.any(grow > 1e-9):
                cands.append(5.0 / float(np.max(grow[grow > 1e-9])))
            osc = mags[mags > 1e-9]
            if osc.size:
                cands.append(6.0 * 2 * math.pi / float(np.min(osc)))
            if np.any(mags <= 1e-9):
                fast = np.abs(p[np.abs(p) > 1e-9])
                cands.append(10.0 / float(np.min(fast)) if fast.size else 10.0)
    tf = max(cands) if cands else 1.0
    if sys.is_discrete:
        tf = max(tf, 20 * sys.Ts)
    return min(tf, 1e6) + tmax


def _auto_dt(sys: LTI, tfinal: float) -> float:
    p = _poles_of(sys)
    mags = np.abs(p[np.abs(p) > 1e-12]) if p.size else np.zeros(0)
    dt = tfinal / 500.0
    if mags.size:
        dt = min(dt, 0.1 / float(np.max(mags)) * 2 * math.pi / 2)
    return max(dt, tfinal / 20000.0)


def _time_vector(sys: LTI, t) -> np.ndarray:
    if t is None or np.ndim(t) == 0:
        tfinal = _auto_tfinal(sys) if t is None else float(t)
        if not tfinal > 0:
            raise ControlError("Bitiş zamanı pozitif olmalı.")
        if sys.is_discrete:
            n = int(math.floor(tfinal / sys.Ts + 1e-9))
            return np.arange(n + 1) * sys.Ts
        dt = _auto_dt(sys, tfinal)
        n = max(int(math.ceil(tfinal / dt)), 2)
        return np.linspace(0.0, tfinal, n + 1)
    t = np.asarray(t, float).ravel()
    if t.size == 0 or not np.all(np.isfinite(t)) or np.any(t < 0):
        raise ControlError("Zaman vektörü negatif olmayan sonlu değerlerden oluşmalı.")
    if np.any(np.diff(t) <= 0):
        raise ControlError("Zaman vektörü kesin artan olmalı.")
    if sys.is_discrete:
        k = t / sys.Ts
        if np.any(np.abs(k - np.round(k)) > 1e-6):
            raise ControlError("Ayrık modelde zaman noktaları Ts'nin katları olmalı.")
    return t


def _uniform(t: np.ndarray) -> float | None:
    if t.size < 2:
        return None
    d = np.diff(t)
    dt = (t[-1] - t[0]) / (t.size - 1)
    if np.all(np.abs(d - dt) <= 1e-9 * max(dt, abs(t[-1]))):
        return float(dt)
    return None


# ---------------------------------------------------------------------------------------------------------------
# Tam sürekli cevaplar
# ---------------------------------------------------------------------------------------------------------------
def _states_at(A: np.ndarray, b: np.ndarray, times: np.ndarray, kind: str) -> np.ndarray:
    """x(t) tam: kind='step' → Γ0(t)·b, kind='impulse' / 'free' → e^{At}·b. times ≥ 0 artan. (len, n)."""
    n = A.shape[0]
    out = np.zeros((times.size, n))
    if n == 0 or times.size == 0:
        return out
    bb = b.reshape(n, 1)

    def point(tt):
        Phi, G0, _ = _integrals(A, bb, float(tt))
        return (G0 if kind == "step" else Phi @ bb)[:, 0]

    dt = _uniform(times)
    if dt is None:
        for k, tt in enumerate(times):
            out[k] = point(tt)
        return out
    Phi, G0, _ = _integrals(A, bb, dt)
    x = point(times[0])
    out[0] = x
    g = G0[:, 0] if kind == "step" else 0.0
    for k in range(1, times.size):
        x = Phi @ x + g
        out[k] = x
    return out


def _shift_groups(sys: StateSpace):
    """(giriş j, toplam kayma, çıkış satırları) grupları."""
    for j in range(sys.ninputs):
        vals = {}
        for i in range(sys.noutputs):
            key = float(sys.input_delay[j] + sys.output_delay[i])
            vals.setdefault(key, []).append(i)
        for sh, rows in vals.items():
            yield j, sh, rows


def _cont_resp(sys: StateSpace, t: np.ndarray, kind: str):
    """Sürekli ss'nin adım / dürtü cevabı (gecikmeler tam). y (nt, ny, nu), x (nt, nx, nu)."""
    A, B, C, D = sys.A, sys.B, sys.C, sys.D
    nt, ny, nu, n = t.size, sys.noutputs, sys.ninputs, sys.nx
    y = np.zeros((nt, ny, nu))
    x = np.zeros((nt, n, nu))
    for j, sh, rows in _shift_groups(sys):
        tt = t - sh
        ok = tt >= -1e-12 * max(1.0, abs(sh))
        tt = np.maximum(tt[ok], 0.0)
        X = _states_at(A, B[:, j], tt, kind)
        yy = X @ C[rows].T
        if kind == "step":
            yy = yy + D[rows, j][None, :]
        y[np.ix_(ok, rows, [j])] = yy[:, :, None]
    for j in range(nu):
        tt = t - sys.input_delay[j]
        ok = tt >= -1e-12 * max(1.0, abs(sys.input_delay[j]))
        x[ok, :, j] = _states_at(A, B[:, j], np.maximum(tt[ok], 0.0), kind)
    return y, x


def _disc_seq(A, B, C, D, N, u0_only: bool):
    """Ayrık: u0_only → x[0]=0, u[0]=1, sonra 0 (dürtü); değilse u ≡ 1 (adım). Gecikmesiz ŷ (N, ny, nu), x."""
    n, m = B.shape
    p = C.shape[0]
    y = np.zeros((N, p, m))
    xs = np.zeros((N, n, m))
    for j in range(m):
        x = np.zeros(n)
        for k in range(N):
            u = 1.0 if (not u0_only or k == 0) else 0.0
            xs[k, :, j] = x
            y[k, :, j] = C @ x + D[:, j] * u
            x = A @ x + B[:, j] * u
    return y, xs


def _disc_resp(sys: StateSpace, t: np.ndarray, kind: str):
    k = np.round(t / sys.Ts).astype(int)
    N = int(k.max()) + 1
    yh, xh = _disc_seq(sys.A, sys.B, sys.C, sys.D, N, kind == "impulse")
    if kind == "impulse":
        yh = yh / sys.Ts
        xh = xh / sys.Ts
    ny, nu, n = sys.noutputs, sys.ninputs, sys.nx
    y = np.zeros((t.size, ny, nu))
    x = np.zeros((t.size, n, nu))
    for i in range(ny):
        for j in range(nu):
            d = int(round(sys.input_delay[j] + sys.output_delay[i]))
            idx = k - d
            ok = idx >= 0
            y[ok, i, j] = yh[idx[ok], i, j]
    for j in range(nu):
        idx = k - int(round(sys.input_delay[j]))
        ok = idx >= 0
        x[ok, :, j] = xh[idx[ok], :, j]
    return y, x


def _response(sys: LTI, t, kind: str, return_x: bool):
    if not isinstance(sys, LTI):
        raise ControlError("Bir model bekleniyor.")
    t = _time_vector(sys, t)
    if isinstance(sys, StateSpace):
        y, x = (_disc_resp if sys.is_discrete else _cont_resp)(sys, t, kind)
        return (y, t, x) if return_x else (y, t)
    ny, nu = sys.shape
    y = np.zeros((t.size, ny, nu))
    tot = sys.total_delay()
    for i in range(ny):
        for j in range(nu):
            e = convert.to_ss(_nodelay(sys[i, j]))
            e.input_delay = np.array([tot[i, j]])
            yy, _ = (_disc_resp if sys.is_discrete else _cont_resp)(e, t, kind)
            y[:, i, j] = yy[:, 0, 0]
    return (y, t, None) if return_x else (y, t)


def step(sys: LTI, t=None, *, return_x: bool = False):
    """Birim adım cevabı (MATLAB `[y, t, x] = step(sys, t)`). `t`: None (otomatik), bitiş zamanı ya da zaman
    vektörü (ayrıkta Ts katları). y (nt, ny, nu); x (nt, nx, nu) yalnız ss için."""
    return _response(sys, t, "step", return_x)


def impulse(sys: LTI, t=None, *, return_x: bool = False):
    """Dürtü cevabı (MATLAB `impulse`): sürekli zamanda C e^{At} B (D'nin Dirac'ı gösterilmez), ayrıkta birim alanlı
    darbe (u[0] = 1/Ts) cevabı. y (nt, ny, nu)."""
    return _response(sys, t, "impulse", return_x)


def initial(sys: LTI, x0, t=None, *, return_x: bool = False):
    """Başlangıç koşulu cevabı (MATLAB `initial`, yalnız ss): giriş sıfır, x(0) = x0. y (nt, ny)."""
    if not isinstance(sys, StateSpace):
        raise ControlError("Başlangıç koşulu cevabı yalnız durum uzayı modelinde tanımlıdır (ss(sys) ile çevirin).")
    x0 = np.asarray(x0, float).ravel()
    if x0.size != sys.nx:
        raise ControlError(f"x0 {sys.nx} elemanlı olmalı.")
    t = _time_vector(sys, t)
    y, x = _free(sys, x0, t)
    return (y, t, x) if return_x else (y, t)


def _free(sys: StateSpace, x0: np.ndarray, t: np.ndarray):
    """Serbest cevap (çıkış gecikmeleri tam): y (nt, ny), x (nt, nx)."""
    nt, ny, n = t.size, sys.noutputs, sys.nx
    y = np.zeros((nt, ny))
    if sys.is_discrete:
        k = np.round(t / sys.Ts).astype(int)
        N = int(k.max()) + 1
        xs = np.zeros((N, n))
        x = x0.copy()
        for i in range(N):
            xs[i] = x
            x = sys.A @ x
        X = xs[k]
        for i in range(ny):
            idx = k - int(round(sys.output_delay[i]))
            ok = idx >= 0
            y[ok, i] = xs[idx[ok]] @ sys.C[i]
        return y, X
    X = _states_at(sys.A, x0, t, "free")
    for od in set(float(v) for v in sys.output_delay):
        rows = [i for i in range(ny) if float(sys.output_delay[i]) == od]
        tt = t - od
        ok = tt >= -1e-12 * max(1.0, od)
        xx = _states_at(sys.A, x0, np.maximum(tt[ok], 0.0), "free") if od else X[ok]
        y[np.ix_(ok, rows)] = xx @ sys.C[rows].T
    return y, X


# ---------------------------------------------------------------------------------------------------------------
# lsim
# ---------------------------------------------------------------------------------------------------------------
def _shifted(u: np.ndarray, o: int) -> np.ndarray:
    """v[k] = u[k + o] (sınır dışı sıfır)."""
    out = np.zeros_like(u)
    n = u.size
    if o >= 0:
        if o < n:
            out[:n - o] = u[o:]
    else:
        if -o < n:
            out[-o:] = u[:n + o]
    return out


def _lsim_cont(sys: StateSpace, U: np.ndarray, t: np.ndarray, x0: np.ndarray, method: str):
    dt = _uniform(t)
    if dt is None:
        raise ControlError("Sürekli modelde lsim için zaman vektörü eşit aralıklı olmalı.")
    A, B, C, D = sys.A, sys.B, sys.C, sys.D
    n, nt = A.shape[0], t.size
    od = sys.output_delay
    frac_out = [abs(v / dt - round(v / dt)) > 1e-9 for v in od]
    if any(frac_out):
        # kesirli çıkış gecikmesi: satır satır (çıkış gecikmesi girişlere eklenerek), serbest cevap ayrı
        y = np.zeros((nt, sys.noutputs))
        for i in range(sys.noutputs):
            row = StateSpace(A, B, C[[i]], D[[i]], 0.0, input_delay=sys.input_delay + od[i])
            yi, _ = _lsim_cont(row, U, t, np.zeros(n), method)
            y[:, i] = yi[:, 0]
        if np.any(x0):
            yf, _ = _free(sys, x0, t - t[0])
            y += yf
        _, X = _lsim_cont(StateSpace(A, B, C, D, 0.0, input_delay=sys.input_delay), U, t, x0, method)
        return y, X
    Phi = sla.expm(A * dt) if n else np.zeros((0, 0))
    W = np.zeros((nt, n))
    Y = np.zeros((nt, sys.noutputs))
    for j in range(sys.ninputs):
        G, H = delay_taps(A, B[:, [j]], C, D[:, [j]], dt, float(sys.input_delay[j]), method)
        for o, g in G.items():
            W += np.outer(_shifted(U[:, j], o), g)
        for o, h in H.items():
            Y += np.outer(_shifted(U[:, j], o), h)
    X = np.zeros((nt, n))
    x = x0.copy()
    for k in range(nt):
        X[k] = x
        if k + 1 < nt:
            x = Phi @ x + W[k]
    Y += X @ C.T
    # tamsayı çıkış gecikmeleri
    for i in range(sys.noutputs):
        d = int(round(od[i] / dt))
        if d:
            Y[:, i] = _shifted(Y[:, i], -d)
    return Y, X


def _lsim_disc(sys: StateSpace, U: np.ndarray, x0: np.ndarray):
    A, B, C, D = sys.A, sys.B, sys.C, sys.D
    nt = U.shape[0]
    Ud = np.column_stack([_shifted(U[:, j], -int(round(sys.input_delay[j]))) for j in range(sys.ninputs)]) \
        if sys.ninputs else U
    X = np.zeros((nt, sys.nx))
    x = x0.copy()
    for k in range(nt):
        X[k] = x
        x = A @ x + B @ Ud[k]
    Y = X @ C.T + Ud @ D.T
    for i in range(sys.noutputs):
        d = int(round(sys.output_delay[i]))
        if d:
            Y[:, i] = _shifted(Y[:, i], -d)
    return Y, X


def lsim(sys: LTI, u, t=None, x0=None, method: str = "zoh", *, return_x: bool = False):
    """Keyfi giriş cevabı (MATLAB `[y, t, x] = lsim(sys, u, t, x0, method)`).

    u: (nt,) ya da (nt, nu). t: eşit aralıklı zaman (sürekli) / Ts katları (ayrık; verilmezse 0, Ts, 2Ts …).
    method: 'zoh' (örnekler arası sabit) ya da 'foh' (doğrusal); sürekli sistem bu kabulle TAM ayrıklaştırılır
    (kesirli gecikmeler dahil). MATLAB'in varsayılanı girişe göre otomatik seçimdir — karşılaştırmada yöntemi açık
    verin. x0 yalnız ss için. Dönüş y (nt, ny), t [, x (nt, nx)]."""
    if not isinstance(sys, LTI):
        raise ControlError("Bir model bekleniyor.")
    method = str(method).lower()
    if method not in ("zoh", "foh"):
        raise ControlError("lsim yöntemi 'zoh' ya da 'foh' olmalı.")
    U = np.asarray(u, float)
    if U.ndim == 1:
        U = U[:, None]
    if U.ndim != 2 or U.shape[1] != sys.ninputs:
        raise ControlError(f"Giriş u (nt, {sys.ninputs}) boyutlu olmalı.")
    if not np.all(np.isfinite(U)):
        raise ControlError("Giriş sonlu değerlerden oluşmalı.")
    nt = U.shape[0]
    if t is None:
        if not sys.is_discrete:
            raise ControlError("Sürekli modelde lsim için zaman vektörü gerekli.")
        t = np.arange(nt) * sys.Ts
    t = np.asarray(t, float).ravel()
    if t.size != nt:
        raise ControlError("Zaman vektörü ile giriş aynı uzunlukta olmalı.")
    if np.any(np.diff(t) <= 0):
        raise ControlError("Zaman vektörü kesin artan olmalı.")
    if sys.is_discrete:
        d = np.diff(t) / sys.Ts
        if np.any(np.abs(d - 1.0) > 1e-6):
            raise ControlError("Ayrık modelde zaman adımı Ts olmalı.")
    if x0 is not None and not isinstance(sys, StateSpace):
        raise ControlError("Başlangıç durumu (x0) yalnız durum uzayı modelinde verilebilir.")
    if isinstance(sys, StateSpace):
        x0v = np.zeros(sys.nx) if x0 is None else np.asarray(x0, float).ravel()
        if x0v.size != sys.nx:
            raise ControlError(f"x0 {sys.nx} elemanlı olmalı.")
        if sys.is_discrete:
            y, x = _lsim_disc(sys, U, x0v)
        else:
            y, x = _lsim_cont(sys, U, t, x0v, method)
        return (y, t, x) if return_x else (y, t)
    ny, nu = sys.shape
    tot = sys.total_delay()
    y = np.zeros((nt, ny))
    for i in range(ny):
        for j in range(nu):
            e = convert.to_ss(_nodelay(sys[i, j]))
            e.input_delay = np.array([tot[i, j]])
            if sys.is_discrete:
                yy, _ = _lsim_disc(e, U[:, [j]], np.zeros(e.nx))
            else:
                yy, _ = _lsim_cont(e, U[:, [j]], t, np.zeros(e.nx), method)
            y[:, i] += yy[:, 0]
    return (y, t, None) if return_x else (y, t)


# ---------------------------------------------------------------------------------------------------------------
# stepinfo / lsiminfo (dizi)
# ---------------------------------------------------------------------------------------------------------------
_NAN_INFO = dict(RiseTime=math.nan, TransientTime=math.nan, SettlingTime=math.nan, SettlingMin=math.nan,
                 SettlingMax=math.nan, Overshoot=math.nan, Undershoot=math.nan, Peak=math.inf, PeakTime=math.inf)


def _first_cross(e: np.ndarray, t: np.ndarray, level: float, interp: bool) -> tuple[float, int]:
    idx = np.flatnonzero(e >= level)
    if idx.size == 0:
        return math.nan, -1
    k = int(idx[0])
    if k == 0 or not interp:
        return float(t[k]), k
    e0, e1 = e[k - 1], e[k]
    return float(t[k - 1] + (level - e0) * (t[k] - t[k - 1]) / (e1 - e0)), k


def _settle(y: np.ndarray, t: np.ndarray, yf: float, band: float, interp: bool) -> float:
    err = np.abs(y - yf)
    out = np.flatnonzero(err > band)
    if out.size == 0:
        return float(t[0])
    k = int(out[-1])
    if k == y.size - 1:
        return math.nan
    if not interp:
        return float(t[k + 1])
    # örnekler arası doğrusal y ile bant kenarı geçişi
    y0, y1 = y[k], y[k + 1]
    edge = yf + band if y0 > yf else yf - band
    if y1 == y0:
        return float(t[k + 1])
    return float(t[k] + (edge - y0) * (t[k + 1] - t[k]) / (y1 - y0))


def _info_array(y, t, yfinal, yinit, thr, lims, interp=True) -> dict:
    y = np.asarray(y, float).ravel()
    t = np.asarray(t, float).ravel()
    yf = float(y[-1]) if yfinal is None else float(yfinal)
    y0 = 0.0 if yinit is None else float(yinit)
    dy = yf - y0
    if not np.isfinite(yf):
        return dict(_NAN_INFO)
    if dy == 0 and np.all(y == y0):
        return dict(RiseTime=0.0, TransientTime=0.0, SettlingTime=0.0, SettlingMin=y0, SettlingMax=y0,
                    Overshoot=0.0, Undershoot=0.0, Peak=0.0, PeakTime=float(t[0]))
    if dy == 0:
        return _info_zero_change(y, t, yf, y0, thr, interp)
    e = (y - y0) / dy
    t_lo, _ = _first_cross(e, t, lims[0], interp)
    t_hi, k_hi = _first_cross(e, t, lims[1], interp)
    rise = t_hi - t_lo if k_hi >= 0 else math.nan
    st = _settle(y, t, yf, thr * abs(dy), interp)
    tr = _settle(y, t, yf, thr * float(np.max(np.abs(y - yf))), interp)
    if k_hi >= 0:
        smin, smax = float(np.min(y[k_hi:])), float(np.max(y[k_hi:]))
    else:
        smin = smax = math.nan
    ov = 100.0 * max(0.0, float(np.max(e)) - 1.0)
    un = 100.0 * max(0.0, -float(np.min(e)))
    a = np.abs(y - y0)
    kp = int(np.argmax(a))
    return dict(RiseTime=rise, TransientTime=tr, SettlingTime=st, SettlingMin=smin, SettlingMax=smax,
                Overshoot=ov, Undershoot=un, Peak=float(a[kp]), PeakTime=float(t[kp]))


def _info_zero_change(y, t, yf, y0, thr, interp=True) -> dict:
    """Son değer = başlangıç değeri ama cevap sıfır değil (ör. S = feedback(1, L)): yüzdelik ölçütler tanımsız.

    MATLAB R2025b'den ölçüldü: RiseTime 0, SettlingTime NaN, Overshoot = Undershoot = ∞; TransientTime, SettlingMin /
    SettlingMax (tüm cevap), Peak = max|y − yinit| ve PeakTime olağan tanımla. (MATLAB ayrık modelde yfinal'i ~1e-17
    yuvarlama kalıntısı olarak bulup 1e15 mertebesinde sonlu aşım verir; tanımın limiti ∞'dur.)"""
    tr = _settle(y, t, yf, thr * float(np.max(np.abs(y - yf))), interp)
    a = np.abs(y - y0)
    kp = int(np.argmax(a))
    return dict(RiseTime=0.0, TransientTime=tr, SettlingTime=math.nan, SettlingMin=float(np.min(y)),
                SettlingMax=float(np.max(y)), Overshoot=math.inf, Undershoot=math.inf, Peak=float(a[kp]),
                PeakTime=float(t[kp]))


def stepinfo(sys_or_y, t=None, yfinal=None, yinit=None, SettlingTimeThreshold: float = 0.02,
             RiseTimeLimits=(0.1, 0.9)):
    """Adım cevabı ölçütleri (MATLAB `stepinfo`, R2025b tanımları).

    - Dizi girişi: `stepinfo(y, t, yfinal, yinit)` → sözlük (y (nt,)) ya da sözlük listeleri (y (nt, ny[, nu])).
    - Model girişi: `stepinfo(sys)` → SISO'da sözlük, MIMO'da [i][j] sözlük listesi (yfinal = DC kazancı, yinit = 0).
    Alanlar: RiseTime, TransientTime, SettlingTime, SettlingMin, SettlingMax, Overshoot (%), Undershoot (%), Peak,
    PeakTime."""
    thr = float(SettlingTimeThreshold)
    lims = (float(RiseTimeLimits[0]), float(RiseTimeLimits[1]))
    if not (0 < thr < 1) or not (0 <= lims[0] < lims[1] <= 1):
        raise ControlError("Eşikler geçersiz: 0 < SettlingTimeThreshold < 1, 0 ≤ alt < üst ≤ 1.")
    if isinstance(sys_or_y, LTI):
        return _stepinfo_sys(sys_or_y, thr, lims, yinit)
    y = np.asarray(sys_or_y, float)
    if y.ndim == 0 or y.shape[0] < 2:
        raise ControlError("Cevap en az iki örnek içermeli.")
    tt = np.arange(1, y.shape[0] + 1, dtype=float) if t is None else np.asarray(t, float).ravel()
    if tt.size != y.shape[0]:
        raise ControlError("y ve t aynı uzunlukta olmalı.")
    if y.ndim == 1:
        return _info_array(y, tt, yfinal, yinit, thr, lims)
    yy = y.reshape(y.shape[0], y.shape[1], -1)
    yf = None if yfinal is None else np.broadcast_to(np.asarray(yfinal, float), yy.shape[1:]) \
        if np.ndim(yfinal) else np.full(yy.shape[1:], float(yfinal))
    y0 = None if yinit is None else np.broadcast_to(np.asarray(yinit, float), yy.shape[1:]) \
        if np.ndim(yinit) else np.full(yy.shape[1:], float(yinit))
    res = [[_info_array(yy[:, i, j], tt, None if yf is None else yf[i, j], None if y0 is None else y0[i, j], thr,
                        lims) for j in range(yy.shape[2])] for i in range(yy.shape[1])]
    return res if y.ndim == 3 else [r[0] for r in res]


def lsiminfo(y, t=None, yfinal=None, yinit=None, SettlingTimeThreshold: float = 0.02):
    """Genel cevap ölçütleri (MATLAB `lsiminfo`, R2025b): TransientTime, SettlingTime, Peak, PeakTime, Min, MinTime,
    Max, MaxTime.

    yfinal varsayılanı son değer, yinit varsayılanı 0 (MATLAB'den ölçüldü); SettlingTime bandı eşik·|yfinal − yinit|,
    TransientTime bandı eşik·max|y − yfinal| (stepinfo ile aynı aradeğer); Peak = max|y − yinit|."""
    y = np.asarray(y, float).ravel()
    tt = np.arange(1, y.size + 1, dtype=float) if t is None else np.asarray(t, float).ravel()
    if tt.size != y.size or y.size < 2:
        raise ControlError("y ve t aynı uzunlukta ve en az iki örnek olmalı.")
    thr = float(SettlingTimeThreshold)
    yf = float(y[-1]) if yfinal is None else float(yfinal)
    y0 = 0.0 if yinit is None else float(yinit)
    st = _settle(y, tt, yf, thr * abs(yf - y0), True)
    tr = _settle(y, tt, yf, thr * float(np.max(np.abs(y - yf))), True)
    a = np.abs(y - y0)
    kp, kmin, kmax = int(np.argmax(a)), int(np.argmin(y)), int(np.argmax(y))
    return dict(TransientTime=tr, SettlingTime=st, Peak=float(a[kp]), PeakTime=float(tt[kp]), Min=float(y[kmin]),
                MinTime=float(tt[kmin]), Max=float(y[kmax]), MaxTime=float(tt[kmax]))


# ---------------------------------------------------------------------------------------------------------------
# stepinfo (model): sürekli zamanda tam
# ---------------------------------------------------------------------------------------------------------------
def _stepinfo_sys(sys: LTI, thr, lims, yinit):
    from spt.control.freqresp import dcgain
    ny, nu = sys.shape
    g = np.atleast_2d(dcgain(sys))
    tot = sys.total_delay()
    res = []
    for i in range(ny):
        row = []
        for j in range(nu):
            e = convert.to_ss(_nodelay(sys[i, j]))
            e = convert.sminreal(e)
            y0 = 0.0 if yinit is None else float(yinit)
            row.append(_stepinfo_siso(e, float(g[i, j]), float(tot[i, j]), thr, lims, y0))
        res.append(row)
    return res[0][0] if (ny, nu) == (1, 1) else res


def _stepinfo_siso(e: StateSpace, yf: float, tau: float, thr, lims, y0) -> dict:
    p = np.linalg.eigvals(e.A) if e.nx else np.zeros(0)
    if e.is_discrete:
        stable = np.all(np.abs(p) < 1 - 1e-12)
    else:
        stable = np.all(p.real < -1e-12 * np.maximum(1.0, np.abs(p)))
    if not stable or not np.isfinite(yf):
        return dict(_NAN_INFO)
    dy = yf - y0
    if dy == 0 and e.nx and np.any(e.C) and np.any(e.B) or dy == 0 and np.any(e.D):
        # son değer = başlangıç ama geçici cevap var: sık ızgarada MATLAB'in sıfır değişim anlamları
        tf_ = _auto_tfinal(e)
        if e.is_discrete:
            n = int(round(tf_ / e.Ts))
            tt = np.arange(n + 1) * e.Ts
        else:
            tt = np.linspace(0.0, tf_, 20001)
        yy = step(e, tt)[0][:, 0, 0]
        if not np.any(yy != y0):
            return dict(RiseTime=0.0, TransientTime=0.0, SettlingTime=0.0, SettlingMin=yf, SettlingMax=yf,
                        Overshoot=0.0, Undershoot=0.0, Peak=0.0, PeakTime=0.0)
        if tau:
            tt, yy = np.concatenate([[0.0], tt + tau]), np.concatenate([[y0], yy])
        info = _info_zero_change(yy, tt, yf, y0, thr, interp=not e.is_discrete)
        # son değer (sınır) da cevabın değerler kümesindedir: alt / üst sınır pencereden bağımsız
        info["SettlingMin"] = min(info["SettlingMin"], yf)
        info["SettlingMax"] = max(info["SettlingMax"], yf)
        return info
    if e.nx == 0 or dy == 0:
        return dict(RiseTime=0.0, TransientTime=0.0, SettlingTime=0.0, SettlingMin=yf, SettlingMax=yf,
                    Overshoot=0.0, Undershoot=0.0, Peak=abs(dy), PeakTime=tau if dy else 0.0)
    if e.is_discrete:
        return _stepinfo_disc(e, yf, tau, thr, lims, y0, p)
    return _stepinfo_cont(e, yf, tau, thr, lims, y0, p)


def _stepinfo_disc(e, yf, tau, thr, lims, y0, p):
    T = e.Ts
    rmax = float(np.max(np.abs(p))) if p.size else 0.0
    nsteps = 50 if rmax <= 1e-12 else int(min(MAX_POINTS, max(50, math.ceil(math.log(1e-12) / math.log(rmax)))))
    d = int(round(tau))
    N = nsteps + d + 1
    yh, _ = _disc_seq(e.A, e.B, e.C, e.D, N, False)
    y = np.concatenate([np.zeros(d), yh[:N - d, 0, 0]])
    t = np.arange(N) * T
    info = _info_array(y, t, yf, y0, thr, lims, interp=False)
    _limit_peak(info, y, y0, yf, discrete=True)
    return info


def _limit_peak(info: dict, y: np.ndarray, y0: float, yf: float, discrete: bool):
    """Tekdüze yaklaşımda tepe sonsuzdaki sınırdır: Peak = |yf − y0|, PeakTime = ∞. Tepe sonlu bir örnekte de
    erişilmişse (eşitlik dahil) o ilk örnek kalır."""
    a = np.abs(y - y0)
    if np.max(a) < abs(yf - y0) * (1 - 1e-14):          # sınır hiçbir örnekte erişilmiyor
        info["Peak"] = abs(yf - y0)
        info["PeakTime"] = math.inf


def _stepinfo_cont(e: StateSpace, yf, tau, thr, lims, y0, p):
    A, b, c, d = e.A, e.B[:, 0], e.C[0], float(e.D[0, 0])
    n = A.shape[0]
    sig = float(np.min(-p.real))
    tend = 30.0 / sig
    wmax = float(np.max(np.abs(p)))
    dt = min(tend / 4000.0, 2 * math.pi / max(wmax, 1e-300) / 40.0)
    npts = int(min(MAX_POINTS, math.ceil(tend / dt))) + 1
    tg = np.linspace(0.0, tend, npts)
    yg = _states_at(A, b, tg, "step") @ c + d
    bb = b.reshape(n, 1)
    eps = np.finfo(float).eps

    def f(tt):
        _, G0, _ = _integrals(A, bb, float(tt))
        return float(c @ G0[:, 0] + d)

    def h(tt):
        return float(c @ (sla.expm(A * float(tt)) @ b))

    dy = yf - y0
    ef = (yg - y0) / dy

    def cross(level):
        idx = np.flatnonzero(ef >= level)
        if idx.size == 0:
            return math.nan, -1
        k = int(idx[0])
        if k == 0:
            return 0.0, 0
        return brentq(lambda s_: (f(s_) - y0) / dy - level, tg[k - 1], tg[k], xtol=1e-15, rtol=4 * eps,
                      maxiter=200), k

    t_lo, _ = cross(lims[0])
    t_hi, k_hi = cross(lims[1])
    sgn = np.sign(np.diff(yg))
    kmax = np.flatnonzero((sgn[:-1] > 0) & (sgn[1:] <= 0)) + 1
    kmin = np.flatnonzero((sgn[:-1] < 0) & (sgn[1:] >= 0)) + 1
    span = float(np.ptp(yg)) + abs(yf) + 1e-300

    def best(ks, sign):
        """Izgaradaki en iyi yerel uca yakın adayları inceltip en iyisini döndür (zaman, değer) ya da None."""
        if ks.size == 0:
            return None
        v = sign * yg[ks]
        cand = ks[v >= v.max() - 1e-6 * span]
        out = None
        for k in cand:
            a_, b_ = tg[k - 1], tg[k + 1]
            ha, hb = h(a_), h(b_)
            if ha * hb < 0:                   # tepe anı: türev (= dürtü cevabı) sıfırı, makine doğruluğunda
                tt = brentq(h, a_, b_, xtol=1e-15, rtol=4 * eps, maxiter=200)
            else:
                tt = float(minimize_scalar(lambda s_: -sign * f(s_), bounds=(a_, b_), method="bounded",
                                           options={"xatol": 1e-14 * max(1.0, tg[k])}).x)
            val = (tt, f(tt))
            if out is None or sign * val[1] > sign * out[1]:
                out = val
        return out

    pts = [(0.0, yg[0])]
    if tau > 0:
        pts.append((-tau, 0.0))              # gecikme süresince çıkış sıfır
    gmax, gmin = best(kmax, +1), best(kmin, -1)
    pts += [v for v in (gmax, gmin) if v is not None]
    vals = [v for _, v in pts] + [yf]
    ymax, ymin = max(vals), min(vals)
    ov = 100.0 * max(0.0, ((ymax if dy > 0 else ymin) - y0) / dy - 1.0)
    un = 100.0 * max(0.0, -((ymin if dy > 0 else ymax) - y0) / dy)
    cand = [(tt, abs(v - y0)) for tt, v in pts]
    pk = float(max(c[1] for c in cand))
    pk_t = min(tt for tt, v in cand if v >= pk * (1 - 1e-13))      # eşit tepelerde ilk an (MATLAB gibi)
    if abs(dy) > pk * (1 + 1e-12):                                   # tepe yalnız sonsuzda erişilen sınır
        pk_t, pk = math.inf, abs(dy)
    if k_hi >= 0:
        after = [f(t_hi), yf]
        for ks, sign in ((kmax, +1), (kmin, -1)):
            r = best(ks[ks > k_hi], sign)
            if r is not None and r[0] >= t_hi:
                after.append(r[1])
        smin, smax = min(after), max(after)
    else:
        smin = smax = math.nan

    def settle(band):
        out = np.flatnonzero(np.abs(yg - yf) > band)
        if out.size == 0:
            return -tau if tau > 0 and abs(yf) > band else 0.0
        k = int(out[-1])
        if k == npts - 1:
            return math.nan
        return brentq(lambda s_: abs(f(s_) - yf) - band, tg[k], tg[k + 1], xtol=1e-15, rtol=4 * eps, maxiter=200)

    emax = max(ymax - yf, yf - ymin)
    st = settle(thr * abs(dy))
    tr = settle(thr * emax)
    return dict(RiseTime=t_hi - t_lo, TransientTime=tr + tau, SettlingTime=st + tau, SettlingMin=smin,
                SettlingMax=smax, Overshoot=ov, Undershoot=un, Peak=pk, PeakTime=pk_t + tau)
