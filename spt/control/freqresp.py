"""Frekans cevabı ve kararlılık payları: freqresp, evalfr, bode, nyquist, nichols, sigma, margin, allmargin,
bandwidth, dcgain, norm (H2 / H∞), getPeakGain.

Sözleşme: freqresp → karmaşık (ny, nu, nw); bode / nichols → mag (mutlak, dB değil), phase (derece) (ny, nu, nw);
nyquist → (re, im) aynı şekil; sigma → (min(ny, nu), nw). w rad/s; ayrık zamanda z = e^{jωTs}, ω ∈ [0, π/Ts].

MATLAB'den ölçülen tanımlar ve farklar:
- bode fazı: her sıfır / kutup çarpanının açısı [−90°, 270°) aralığında alınıp frekans boyunca AYRI AYRI açılır
  (unwrap) ve toplanır, k < 0 ise +180° — ör. 1/s³ → −270°, −1/(s+1) → +180°, sağ yarı düzlem karmaşık sıfır çifti
  düşük frekansta +360° katkı; gecikme fazı (−ωτ) ayrıca ve tam eklenir (MATLAB'den ölçüldü).
- allmargin / margin: kazanç geçişleri |G| = 1, faz geçişleri ∠G = −180° (mod 360°); ω = 0 ve ω = ∞ (faz sınırı
  −180° ise) dahil (MATLAB gibi). Faz payı [−180°, 180°) aralığına sarılır (MATLAB: 1/(s+1) → −180° @ ω = 0).
  Gecikme payı: (Pm mod 360°)/ω (en küçük pozitif gecikme); sürekli zamanda kapalı çevrim kararsızsa Pm/ω (negatif
  olabilir; ayrıkta MATLAB yine mod 360 kullanır);
  ω = 0'da ∞ (Pm = 0 ise 0); sürekli zamanda |G(∞)| ≥ 1 ise ω = ∞'da 0 eklenir; ayrık zamanda bir örneklik ek
  gecikme kapalı çevrimi kararsız yapıyorsa ω = π/Ts'de 1 eklenir (MATLAB R2025b'den ölçüldü). Ayrık zamanda gecikme
  payı örnek sayısıdır.
  margin: GainMargin'den |log Gm| en küçük, PhaseMargin'den |Pm| en küçük olan; yoksa ∞ ve frekans NaN.
  FARK: MATLAB'in geçiş frekansları yaklaşıktır (ör. 2/(s−1) için Wcp = 1.7320003, kesin √3 = 1.7320508; göreli hata
  ~3e-5; tf / zpk / ss aynı). Burada geçişler kesin hesaplanır: rasyonel kısımda polinom kökleri (sürekli ω², ayrık
  cos θ değişkeninde) + çarpanlardan analitik türevle Newton inceltmesi; gecikmede yoğun tarama + Brent.
  Sürekli gecikmede sonsuz sayıda faz geçişi vardır: ω ≤ 200·max(|p|, |z|, 1/τ, ω_c) aralığındakiler (en çok 100)
  listelenir (MATLAB'in aralığı yayımlanmamış ve düzensiz). Gecikmeli sürekli kapalı çevrimin kararlılığı
  (Stable) argüman ilkesiyle (D(s) + kN(s)e^{−sτ}'nin sağ yarı düzlem sıfırları) belirlenir.
- norm(sys, inf) ve getPeakGain: kararsız modelde de frekans cevabının tepe değeri (MATLAB R2025b gibi; sanal eksende
  kutup varsa ∞). Bruinsma–Steinbuch Hamiltonyen yinelemesi + yerel inceltme; tolerans varsayılanı 1e-10 (MATLAB'in
  varsayılanı 1e-2'dir). norm(sys, 2): kararsızsa ya da sürekli zamanda D ≠ 0 ise ∞; ayrık zamanda Ts ile
  ölçeklenmez (√Σ|h[k]|², MATLAB gibi).
- bandwidth: |G(jω)| ilk kez |G(0)|·10^(dbdrop/20) altına düştüğü frekans (kesin); DC kazancı sonsuzsa NaN, hiç
  düşmüyorsa ∞.
"""
from __future__ import annotations

import math

import numpy as np
import scipy.linalg as sla
from scipy.optimize import brentq, minimize_scalar

from spt.control.errors import ControlError
from spt.control.lti import LTI, StateSpace, TransferFunction, ZPK, _padd, cplxpair, poly_from_roots
from spt.control import convert

_DEG = 180.0 / math.pi


# ---------------------------------------------------------------------------------------------------------------
# Frekans cevabı
# ---------------------------------------------------------------------------------------------------------------
def _points(sys: LTI, w: np.ndarray) -> np.ndarray:
    return np.exp(1j * w * sys.Ts) if sys.is_discrete else 1j * w


def _delay_factor(sys: LTI, x: np.ndarray) -> np.ndarray:
    """(ny, nu, nx) gecikme çarpanı: e^{−xτ} (sürekli) / x^{−d} (ayrık)."""
    tot = sys.total_delay()
    if not np.any(tot):
        return np.ones(tot.shape + (x.size,), complex)
    if sys.is_discrete:
        return x[None, None, :] ** (-tot[:, :, None])
    return np.exp(-x[None, None, :] * tot[:, :, None])


def _eval_nodelay(sys: LTI, x: np.ndarray) -> np.ndarray:
    """Gecikmesiz kısmın x noktalarındaki değeri (ny, nu, nx)."""
    ny, nu = sys.shape
    H = np.zeros((ny, nu, x.size), complex)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if isinstance(sys, TransferFunction):
            for i in range(ny):
                for j in range(nu):
                    H[i, j] = np.polyval(sys._num[i][j], x) / np.polyval(sys._den[i][j], x)
            return H
        if isinstance(sys, ZPK):
            for i in range(ny):
                for j in range(nu):
                    k = sys._k[i, j]
                    if k == 0:
                        continue
                    v = np.full(x.size, k, complex)
                    for zz in sys._z[i][j]:
                        v = v * (x - zz)
                    for pp in sys._p[i][j]:
                        v = v / (x - pp)
                    H[i, j] = v
            return H
        A, B, C, D = sys.A, sys.B, sys.C, sys.D
        n = A.shape[0]
        if n == 0:
            return np.repeat(D[:, :, None].astype(complex), x.size, axis=2)
        Hh, Q = sla.hessenberg(A, calc_q=True)
        Bq, Cq = Q.T @ B, C @ Q
        I = np.eye(n)
        chunk = max(1, 4_000_000 // (n * n + 1))
        for s0 in range(0, x.size, chunk):
            xs = x[s0:s0 + chunk]
            M = xs[:, None, None] * I[None] - Hh[None]
            try:
                X = np.linalg.solve(M, np.broadcast_to(Bq, (xs.size,) + Bq.shape))
            except np.linalg.LinAlgError:
                X = np.empty((xs.size, n, B.shape[1]), complex)
                for k, xv in enumerate(xs):
                    try:
                        X[k] = np.linalg.solve(xv * I - Hh, Bq)
                    except np.linalg.LinAlgError:
                        X[k] = np.inf
            H[:, :, s0:s0 + chunk] = np.transpose(Cq[None] @ X, (1, 2, 0)) + D[:, :, None]
    return H


def _wvec(w) -> np.ndarray:
    w = np.atleast_1d(np.asarray(w, float)).ravel()
    if w.size == 0 or not np.all(np.isfinite(w)):
        raise ControlError("Frekans vektörü sonlu değerlerden oluşmalı.")
    return w


def freqresp(sys: LTI, w) -> np.ndarray:
    """Frekans cevabı H(jω) (ayrık: H(e^{jωTs})), karmaşık (ny, nu, nw), gecikmeler dahil (MATLAB `freqresp`)."""
    w = _wvec(w)
    x = _points(sys, w)
    return _eval_nodelay(sys, x) * _delay_factor(sys, x)


def evalfr(sys: LTI, x) -> np.ndarray | complex:
    """Modelin karmaşık bir s (ya da z) noktasındaki değeri, gecikmeler dahil (MATLAB `evalfr`)."""
    xv = np.atleast_1d(np.asarray(x, complex)).ravel()
    H = _eval_nodelay(sys, xv) * _delay_factor(sys, xv)
    H = H[:, :, 0]
    return complex(H[0, 0]) if sys.is_siso else H


def _auto_w(sys: LTI) -> np.ndarray:
    """Varsayılan logaritmik frekans ızgarası (bizim seçimimiz; MATLAB'inki yayımlanmamış)."""
    z = convert.to_zpk(sys)
    roots = []
    for i in range(z.noutputs):
        for j in range(z.ninputs):
            roots.extend(z._p[i][j].tolist())
            roots.extend(z._z[i][j].tolist())
    r = np.asarray(roots, complex)
    if sys.is_discrete and r.size:
        with np.errstate(divide="ignore"):
            r = np.log(r[np.abs(r) > 0]) / sys.Ts
    m = np.abs(r[np.isfinite(r)])
    m = m[m > 1e-12]
    lo = float(np.min(m)) / 100 if m.size else 0.1
    hi = float(np.max(m)) * 100 if m.size else 10.0
    if sys.has_delay() and not sys.is_discrete:
        hi = max(hi, 30.0 / float(np.max(sys.total_delay())))
    if sys.is_discrete:
        hi = math.pi / sys.Ts
        lo = min(lo, hi / 1000)
    dec = math.log10(hi / lo)
    n = int(min(2000, max(200, 60 * dec)))
    w = np.logspace(math.log10(lo), math.log10(hi), n)
    extra = []
    for p in r[np.isfinite(r)]:
        if abs(p) > 0 and abs(p.real) < 0.2 * abs(p):
            wn = abs(p)
            extra.extend(wn * (1 + np.linspace(-3, 3, 41) * max(abs(p.real) / wn, 1e-4)))
    if extra:
        e = np.asarray(extra)
        w = np.unique(np.concatenate([w, e[(e > lo) & (e < hi)]]))
    return w


def _fangle(v: np.ndarray) -> np.ndarray:
    """Çarpan açısı [−90°, 270°) aralığında (MATLAB'den ölçüldü: sağ yarı düzlem karmaşık kökün açısı ω = ∞'daki
    +90°'ye sürekli dalda; sol yarı düzlem ve sanal eksen kökleri esas değerde), sonra frekans boyunca açılır."""
    a = np.angle(v)
    return np.where(a < -0.5 * math.pi - 1e-15, a + 2 * math.pi, a)


def _factor_phase(sys: LTI, x: np.ndarray) -> np.ndarray:
    """MATLAB'in bode fazı (derece, gecikmesiz): her sıfır / kutup çarpanının açısı ayrı ayrı (bkz. _fangle) frekans
    boyunca açılır (unwrap) ve toplanır; k < 0 ise +180°. (ny, nu, nw)."""
    z = convert.to_zpk(sys)
    out = np.zeros(sys.shape + (x.size,))
    for i in range(z.noutputs):
        for j in range(z.ninputs):
            k = z._k[i, j]
            if k == 0:
                continue
            a = np.full(x.size, math.pi if k < 0 else 0.0)
            for zz in z._z[i][j]:
                a += np.unwrap(_fangle(x - zz))
            for pp in z._p[i][j]:
                a -= np.unwrap(_fangle(x - pp))
            out[i, j] = a * _DEG
    return out


def bode(sys: LTI, w=None):
    """Bode verisi (MATLAB `[mag, phase, w] = bode(sys, w)`): mag mutlak, phase derece, şekil (ny, nu, nw)."""
    w = _auto_w(sys) if w is None else _wvec(w)
    x = _points(sys, w)
    mag = np.abs(_eval_nodelay(sys, x))
    phase = _factor_phase(sys, x)
    tot = sys.total_delay()
    if np.any(tot):
        T = sys.Ts if sys.is_discrete else 1.0
        phase = phase - (w[None, None, :] * T * tot[:, :, None]) * _DEG
    return mag, phase, w


def nichols(sys: LTI, w=None):
    """Nichols verisi (MATLAB `nichols`): bode ile aynı (mag mutlak, phase derece)."""
    return bode(sys, w)


def nyquist(sys: LTI, w=None):
    """Nyquist verisi (MATLAB `[re, im, w] = nyquist(sys, w)`), (ny, nu, nw)."""
    w = _auto_w(sys) if w is None else _wvec(w)
    H = freqresp(sys, w)
    return H.real, H.imag, w


def sigma(sys: LTI, w=None):
    """Tekil değerler (MATLAB `[sv, w] = sigma(sys, w)`): (min(ny, nu), nw), azalan sırada, mutlak."""
    w = _auto_w(sys) if w is None else _wvec(w)
    H = freqresp(sys, w)
    sv = np.linalg.svd(np.transpose(H, (2, 0, 1)), compute_uv=False)
    return sv.T, w


# ---------------------------------------------------------------------------------------------------------------
# DC kazancı
# ---------------------------------------------------------------------------------------------------------------
def _dc_entry(z: np.ndarray, p: np.ndarray, k: float, discrete: bool) -> float:
    if k == 0.0:
        return 0.0
    c = 1.0 if discrete else 0.0
    tol = 1e-12
    zi = np.abs(z - c) <= tol * max(1.0, abs(c))
    pi_ = np.abs(p - c) <= tol * max(1.0, abs(c))
    q = int(pi_.sum() - zi.sum())
    val = k * np.prod(c - z[~zi]) / np.prod(c - p[~pi_])
    val = float(np.real(val))
    if q > 0:
        return math.inf                      # MATLAB: işaretten bağımsız +Inf
    if q < 0:
        return 0.0
    return val


def dcgain(sys: LTI):
    """DC kazancı (MATLAB `dcgain`): sürekli G(0), ayrık G(1); integratörlü kanal ±∞. SISO'da sayı, MIMO'da (ny, nu)."""
    if isinstance(sys, StateSpace) and sys.nx:
        n = sys.nx
        M = sys.A if not sys.is_discrete else np.eye(n) - sys.A
        if np.linalg.cond(M) < 1e10:
            g = sys.D + (-1.0 if not sys.is_discrete else 1.0) * sys.C @ np.linalg.solve(M, sys.B)
            return float(g[0, 0]) if sys.is_siso else g
    z = convert.to_zpk(sys)
    g = np.zeros(sys.shape)
    for i in range(z.noutputs):
        for j in range(z.ninputs):
            g[i, j] = _dc_entry(z._z[i][j], z._p[i][j], float(z._k[i, j]), sys.is_discrete)
    return float(g[0, 0]) if sys.is_siso else g


# ---------------------------------------------------------------------------------------------------------------
# Rasyonel SISO yardımcıları (çarpanlardan analitik değer / türev)
# ---------------------------------------------------------------------------------------------------------------
def _quiet(fn):
    """Kökün tam üzerindeki değerlendirmelerde (ω = 0'da integratör gibi) numpy uyarılarını bastırır."""
    def wrap(*a, **k):
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return fn(*a, **k)
    wrap.__name__ = fn.__name__
    wrap.__doc__ = fn.__doc__
    return wrap


class _Rat:
    """SISO rasyonel kısım (z, p, k) + gecikme τ (sürekli s) / d (ayrık örnek)."""

    def __init__(self, sys: LTI):
        if not sys.is_siso:
            raise ControlError("Bu hesap SISO model içindir.")
        zz = convert.to_zpk(sys)
        self.z = np.asarray(zz._z[0][0], complex)
        self.p = np.asarray(zz._p[0][0], complex)
        self.k = float(zz._k[0, 0])
        self.Ts = sys.Ts
        self.disc = sys.is_discrete
        self.tau = float(sys.total_delay()[0, 0])

    def x(self, w):
        return np.exp(1j * w * self.Ts) if self.disc else 1j * w

    @_quiet
    def logmag(self, w):
        x = self.x(w)
        return math.log(abs(self.k)) + sum(math.log(abs(x - z)) for z in self.z) - \
            sum(math.log(abs(x - p)) for p in self.p)

    @_quiet
    def dlog(self, w):
        """d/dω (log|G| + j·∠G) (rasyonel kısım)."""
        x = self.x(w)
        dx = 1j * self.Ts * x if self.disc else 1j
        return sum(dx / (x - z) for z in self.z) - sum(dx / (x - p) for p in self.p)

    @_quiet
    def phase(self, w):
        """Toplam faz (radyan; çarpanların esas açıları + gecikme), sürekli (sarılmamış) fonksiyon değildir."""
        x = self.x(w)
        a = (math.pi if self.k < 0 else 0.0) + sum(np.angle(x - z) for z in self.z) - \
            sum(np.angle(x - p) for p in self.p)
        return float(a) - w * self.tau * (self.Ts if self.disc else 1.0)

    @_quiet
    def value(self, w):
        x = self.x(w)
        v = self.k * np.prod(x - self.z) / np.prod(x - self.p)
        if self.tau:
            v = v * (x ** (-self.tau) if self.disc else np.exp(-x * self.tau))
        return complex(v)


def _mag2_poly_cont(roots: np.ndarray) -> np.ndarray:
    """Π|jω − r|² polinomu W = ω² cinsinden (azalan kuvvet)."""
    p = np.ones(1)
    r = cplxpair(roots, tol=1e-9) if roots.size else roots
    i = 0
    while i < r.size:
        a, b = r[i].real, r[i].imag
        if b == 0.0:
            p = np.convolve(p, [1.0, a * a])
            i += 1
        else:
            m2 = a * a + b * b
            p = np.convolve(p, [1.0, 2.0 * (a * a - b * b), m2 * m2])
            i += 2
    return p


def _mag2_poly_disc(roots: np.ndarray) -> np.ndarray:
    """Π|e^{jθ} − r|² polinomu c = cos θ cinsinden."""
    p = np.ones(1)
    r = cplxpair(roots, tol=1e-9) if roots.size else roots
    i = 0
    while i < r.size:
        a, b = r[i].real, r[i].imag
        if b == 0.0:
            p = np.convolve(p, [-2.0 * a, 1.0 + a * a])
            i += 1
        else:
            m = 1.0 + a * a + b * b
            # (m − 2ac)² − 4b²(1 − c²) = (4a² + 4b²)c² − 4am·c + m² − 4b²
            p = np.convolve(p, [4.0 * (a * a + b * b), -4.0 * a * m, m * m - 4.0 * b * b])
            i += 2
    return p


def _real_roots(poly: np.ndarray, lo: float, hi: float) -> list[float]:
    poly = np.trim_zeros(np.asarray(poly, float), "f")
    if poly.size <= 1:
        return []
    r = np.roots(poly)
    scale = max(1.0, float(np.max(np.abs(r)))) if r.size else 1.0
    out = []
    for v in r:
        if abs(v.imag) <= 1e-6 * max(1.0, abs(v)) and lo - 1e-9 * scale <= v.real <= hi + 1e-9 * scale:
            out.append(float(min(max(v.real, lo), hi)))
    return sorted(out)


def _newton(f, df, x0: float, lo: float, hi: float, tol: float = 1e-15) -> float:
    x = x0
    for _ in range(60):
        fx, d = f(x), df(x)
        if not np.isfinite(fx) or d == 0 or not np.isfinite(d):
            break
        step = fx / d
        xn = min(max(x - step, lo), hi)
        if abs(xn - x) <= tol * max(1.0, abs(x)):
            x = xn
            break
        x = xn
    return x


# ---------------------------------------------------------------------------------------------------------------
# Geçiş frekansları
# ---------------------------------------------------------------------------------------------------------------
def _wmax(R: _Rat) -> float:
    return math.pi / R.Ts if R.disc else math.inf


def _deflate(poly: np.ndarray, x0: float) -> tuple[np.ndarray, bool]:
    """x0 (neredeyse) kök ise polinomu (x − x0) ile böl; (yeni polinom, kök müydü)."""
    poly = np.trim_zeros(np.asarray(poly, float), "f")
    if poly.size <= 1:
        return poly, False
    scale = float(np.sum(np.abs(poly) * max(1.0, abs(x0)) ** np.arange(poly.size - 1, -1, -1)))
    if abs(np.polyval(poly, x0)) <= 1e-12 * max(scale, 1e-300):
        q, _ = np.polydiv(poly, [1.0, -x0])
        return q, True
    return poly, False


def _min_w(R: _Rat) -> float:
    """Sistemin en yavaş (sıfır olmayan) dinamiğinin frekansı (sürekli eşdeğer)."""
    r = np.concatenate([R.z, R.p])
    r = r[np.abs(r) > 0]
    if R.disc:
        r = r[np.abs(r - 1.0) > 0]
        r = np.log(r) / R.Ts if r.size else r
    m = np.abs(r)
    m = m[m > 0]
    return float(np.min(m)) if m.size else 1.0


def _scale_w(R: _Rat) -> float:
    r = np.concatenate([R.z, R.p])
    if R.disc:
        return math.pi / R.Ts
    m = np.abs(r[np.abs(r) > 0])
    return max(1.0, float(np.max(m))) if m.size else 1.0


def _gain_crossovers(R: _Rat) -> list[float]:
    wmax = _wmax(R)
    if R.k == 0:
        return []
    if R.z.size == 0 and R.p.size == 0:
        return [0.0] if abs(abs(R.k) - 1.0) <= 1e-12 else []
    if R.disc:
        F = _padd(R.k * R.k * _mag2_poly_disc(R.z), -_mag2_poly_disc(R.p))
        if not np.any(F):
            return [0.0]
        for e in (1.0, -1.0):
            hit = True
            while hit:
                F, hit = _deflate(F, e)
        cs = [c for c in _real_roots(F, -1.0, 1.0) if abs(abs(c) - 1.0) > 1e-10]
        cand = [math.acos(c) / R.Ts for c in cs]
        ends = [0.0, math.pi / R.Ts]
    else:
        F = _padd(R.k * R.k * _mag2_poly_cont(R.z), -_mag2_poly_cont(R.p))
        if not np.any(F):
            return [0.0]
        wlo = 1e-6 * _min_w(R)
        cand = [math.sqrt(W) for W in _real_roots(F, 0.0, math.inf) if W > 0 and math.sqrt(W) > wlo]
        ends = [0.0]
    out = []
    for w0 in cand:
        w = _newton(lambda v: R.logmag(v), lambda v: float(np.real(R.dlog(v))), w0, 0.0,
                    wmax if math.isfinite(wmax) else 1e300)
        if abs(R.logmag(w)) <= 1e-8:
            out.append(w)
    for we in ends:
        try:
            if abs(R.logmag(we)) <= 1e-10:
                out.append(we)
        except (ValueError, ZeroDivisionError):
            pass
    return _uniq(out, _scale_w(R))


def _uniq(ws: list[float], scale: float = 1.0) -> list[float]:
    ws = sorted(ws)
    out: list[float] = []
    for w in ws:
        new_inf = math.isinf(w) and not math.isinf(out[-1]) if out else True
        if not out or new_inf or (math.isfinite(w) and abs(w - out[-1]) > 1e-9 * max(scale, abs(w))):
            out.append(w)
    return out


def _phase_limit0(R: _Rat) -> tuple[float, float]:
    """ω → 0⁺ limitinde (faz derece mod 360, |G|)."""
    c = 1.0 if R.disc else 0.0
    tol = 1e-12
    a = 180.0 if R.k < 0 else 0.0
    q = 0
    rest = R.k
    for zz in R.z:
        if abs(zz - c) <= tol:
            a += 90.0
            q -= 1
        else:
            a += math.degrees(np.angle(c - zz))
            rest *= (c - zz)
    for pp in R.p:
        if abs(pp - c) <= tol:
            a -= 90.0
            q += 1
        else:
            a -= math.degrees(np.angle(c - pp))
            rest /= (c - pp)
    mag = math.inf if q > 0 else (0.0 if q < 0 else abs(complex(rest)))
    return a % 360.0, mag


def _phase_crossovers(R: _Rat) -> list[tuple[float, float]]:
    """∠G = −180° (mod 360°) frekansları ve |G| değerleri; ω = 0 ve ω = ∞ dahil."""
    out: list[tuple[float, float]] = []
    if R.k == 0:
        return out
    a0, m0 = _phase_limit0(R)
    if abs(a0 - 180.0) <= 1e-9:
        out.append((0.0, m0))
    if R.disc:
        out += _phase_cross_poly(R)
        # ω = π/Ts
        v = R.value(math.pi / R.Ts)
        if abs(v.imag) <= 1e-12 * max(abs(v), 1e-300) and v.real < 0:
            out.append((math.pi / R.Ts, abs(v)))
    elif R.tau == 0:
        out += _phase_cross_poly(R)
        rel = R.p.size - R.z.size
        a_inf = ((180.0 if R.k < 0 else 0.0) + 90.0 * (R.z.size - R.p.size)) % 360.0
        if abs(a_inf - 180.0) <= 1e-9:
            out.append((math.inf, 0.0 if rel > 0 else abs(R.k)))
    else:
        out += _phase_cross_delay(R)
    res: list[tuple[float, float]] = []
    sc = _scale_w(R)
    for w, m in sorted(out, key=lambda z: z[0]):
        new_inf = math.isinf(w) and not math.isinf(res[-1][0]) if res else True
        if not res or new_inf or (math.isfinite(w) and abs(w - res[-1][0]) > 1e-9 * max(sc, abs(w))):
            res.append((w, m))
    return res


def _poly_eval_jw(poly: np.ndarray):
    """Gerçel polinom a(s) için s = jω'da gerçel ve sanal kısım polinomları (ω'da)."""
    n = poly.size - 1
    re = np.zeros(n + 1)
    im = np.zeros(n + 1)
    for idx, c in enumerate(poly):
        k = n - idx
        f = [1, 1j, -1, -1j][k % 4]
        if f.real:
            re[idx] = c * f.real
        else:
            im[idx] = c * f.imag
    return re, im


def _phase_cross_poly(R: _Rat) -> list[tuple[float, float]]:
    """Gecikmesiz (sürekli) ya da ayrık rasyonel kısımda Im G = 0, Re G < 0 kesin çözümü."""
    num = R.k * poly_from_roots(R.z)
    den = poly_from_roots(R.p)
    cand: list[float] = []
    if R.disc:
        d = int(round(R.tau))
        den = np.concatenate([den, np.zeros(d)])
        # N(z)D(1/z) = Σ c_k z^k ; Im = Σ_{k>0} (c_k − c_{−k}) sin kθ = sin θ · Q(cos θ)
        Nc = num[::-1]
        Dc = den[::-1]
        conv = np.convolve(Nc, Dc[::-1])          # katsayı indisi k + len(D) − 1
        off = Dc.size - 1
        K = max(conv.size - 1 - off, off)
        Q = np.zeros(1)
        U = [np.array([1.0]), np.array([2.0, 0.0])]          # Chebyshev U_0, U_1 (azalan kuvvet)
        for k in range(2, K + 1):
            U.append(_padd(np.convolve([2.0, 0.0], U[-1]), -U[-2]))
        for k in range(1, K + 1):
            ck = conv[k + off] if 0 <= k + off < conv.size else 0.0
            cmk = conv[-k + off] if 0 <= -k + off < conv.size else 0.0
            if ck - cmk != 0.0:
                Q = _padd(Q, (ck - cmk) * U[k - 1])
        if np.any(Q):
            for e in (1.0, -1.0):
                hit = True
                while hit:
                    Q, hit = _deflate(Q, e)
        cs = [c for c in _real_roots(Q, -1.0, 1.0) if abs(abs(c) - 1.0) > 1e-10] if np.any(Q) else []
        cand = [math.acos(c) / R.Ts for c in cs]
        lo, hi = 0.0, math.pi / R.Ts
    else:
        nr, ni = _poly_eval_jw(num)
        dr, di = _poly_eval_jw(den)
        Im = _padd(np.convolve(ni, dr), -np.convolve(nr, di))
        Im = np.trim_zeros(Im, "f")
        # tek polinom: ω ile bölüp ω² cinsine indir
        if Im.size > 1 and np.any(Im):
            Im = np.trim_zeros(Im, "b") if Im[-1] == 0 else Im
            W = Im[::-1][1::2][::-1] if (Im.size - 1) % 2 == 1 else Im[::-1][0::2][::-1]
            Ws = _real_roots(W, 0.0, math.inf)
            cand = [math.sqrt(v) for v in Ws if v > 0]
        lo, hi = 0.0, 1e300
    out = []
    for w0 in cand:
        if w0 <= 0 or w0 >= hi * (1 - 1e-12):
            continue
        ref = R.phase(w0)
        target = math.pi * (2 * round((ref + math.pi) / (2 * math.pi)) - 1)

        def f(v, ref=ref, target=target):
            return _branch(R, v, ref) - target

        w = _newton(f, lambda v: float(np.imag(R.dlog(v))) - R.tau * (R.Ts if R.disc else 1.0), w0, lo, hi)
        v = R.value(w)
        if v.real < 0 and abs(v.imag) <= 1e-7 * abs(v):
            out.append((w, abs(v)))
    return out


def _phase_grid(R: _Rat, wmax: float) -> np.ndarray:
    pts = [np.linspace(0.0, wmax, int(min(400_000, max(4000, 16 * R.tau * wmax / math.pi))) + 1)]
    for r in np.concatenate([R.z, R.p]):
        a, b = abs(r.real), abs(r.imag)
        if b > 0 and a < 0.3 * b:
            pts.append(b + np.linspace(-30, 30, 241) * max(a, 1e-9 * b))
    for r in np.concatenate([R.z, R.p]):
        m = abs(r)
        if m > 0:
            pts.append(np.logspace(math.log10(m) - 3, math.log10(m) + 1, 200))
    w = np.concatenate(pts)
    return np.unique(w[(w >= 0) & (w <= wmax)])


def _unwrapped_phase(R: _Rat, w: np.ndarray) -> np.ndarray:
    x = R.x(w)
    a = np.full(w.size, math.pi if R.k < 0 else 0.0)
    for z in R.z:
        a += np.unwrap(np.angle(x - z))
    for p in R.p:
        a -= np.unwrap(np.angle(x - p))
    return a - w * R.tau


def _phase_cross_delay(R: _Rat) -> list[tuple[float, float]]:
    """Sürekli + gecikme: ∠G(ω) − ωτ = −π − 2πm çözümleri (yoğun tarama + Brent), ω ≤ wmax."""
    gcs = _gain_crossovers(R)
    refs = [abs(r) for r in np.concatenate([R.z, R.p]) if abs(r) > 0] + [1.0 / R.tau] + [g for g in gcs if g > 0]
    wmax = 200.0 * max(refs)
    w = _phase_grid(R, wmax)
    ph = _unwrapped_phase(R, w)
    q = np.floor((ph + math.pi) / (2 * math.pi))
    idx = np.flatnonzero(np.diff(q) != 0)
    out = []
    for k in idx:
        lev = max(q[k], q[k + 1])
        target = -math.pi + 2 * math.pi * lev

        def f(v, k0=k, target=target):
            # örnekler arası doğrusal açılmış faza en yakın dal (aralık küçük, faz sürekli)
            r = (v - w[k0]) / (w[k0 + 1] - w[k0])
            return _branch(R, v, ph[k0] + r * (ph[k0 + 1] - ph[k0])) - target

        try:
            wc = brentq(f, w[k], w[k + 1], xtol=1e-15, rtol=4 * np.finfo(float).eps, maxiter=200)
        except ValueError:
            continue
        if wc > 0:
            v = R.value(wc)
            out.append((wc, abs(v)))
        if len(out) >= 100:
            break
    return out


def _branch(R: _Rat, v: float, ref: float) -> float:
    a = R.phase(v)
    return a + 2 * math.pi * round((ref - a) / (2 * math.pi))


# ---------------------------------------------------------------------------------------------------------------
# Kapalı çevrim kararlılığı
# ---------------------------------------------------------------------------------------------------------------
def _cl_stable(R: _Rat, sys: LTI) -> bool:
    if R.disc or R.tau == 0:
        from spt.control.lti import ss_absorb
        s = convert.to_ss(sys)
        if s.is_discrete and s.has_delay():
            s = ss_absorb(s)
        elif s.has_delay():
            s = StateSpace(s.A, s.B, s.C, s.D, s.Ts)
        if s.nx == 0:
            return abs(1.0 + float(s.D[0, 0])) > 0
        den = 1.0 + float(s.D[0, 0])
        if abs(den) <= 1e-14:
            return False
        Acl = s.A - np.outer(s.B[:, 0], s.C[0]) / den
        ev = np.linalg.eigvals(Acl)
        if s.is_discrete:
            return bool(np.all(np.abs(ev) < 1 - 1e-10))
        return bool(np.all(ev.real < -1e-10 * np.maximum(1.0, np.abs(ev))))
    return _cl_stable_delay(R)


def _cl_stable_delay(R: _Rat) -> bool:
    """f(s) = D(s) + k N(s) e^{−sτ}'nin sağ yarı düzlemde sıfırı yoksa kararlı (argüman ilkesi)."""
    num = R.k * poly_from_roots(R.z)
    den = poly_from_roots(R.p)
    if num.size >= den.size:
        g_inf = abs(num[0] / den[0]) if num.size == den.size else math.inf
        if g_inf >= 1.0:
            return False                      # nötr tip: sonsuz sayıda sağ yarı düzlem kökü
    # |G(s)| < 1/2 olan yarıçap (sağ yarı düzlemde) kaba sınırla
    mz = np.abs(R.z) if R.z.size else np.zeros(0)
    mp = np.abs(R.p) if R.p.size else np.zeros(0)
    Rr = max([1.0] + list(mp) + list(mz)) * 2.0
    for _ in range(200):
        bound = abs(R.k) * np.prod(Rr + mz) / np.prod(Rr - mp) if mp.size else abs(R.k) * np.prod(Rr + mz)
        if mp.size and np.all(Rr > mp) and bound < 0.5:
            break
        Rr *= 1.5
    tau = R.tau

    def f(s):
        return np.polyval(den, s) + np.polyval(num, s) * np.exp(-s * tau)

    # kapalı eğri: sanal eksen (ω: Rr → −Rr, aşağı) + sağ yarı çember (−π/2 → π/2)
    def track(param, a, b):
        ts = np.linspace(a, b, 2001)
        vals = f(param(ts))
        total = 0.0
        stack = [(ts[i], ts[i + 1], vals[i], vals[i + 1]) for i in range(ts.size - 1)]
        while stack:
            t0, t1, v0, v1 = stack.pop()
            d = np.angle(v1 / v0) if v0 != 0 and v1 != 0 else math.nan
            if not np.isfinite(d) or abs(d) > math.pi / 4:
                if abs(t1 - t0) <= 1e-13 * max(1.0, abs(t0)):
                    return math.nan
                tm = 0.5 * (t0 + t1)
                vm = f(param(np.array([tm])))[0]
                if vm == 0:
                    return math.nan
                stack.append((tm, t1, vm, v1))
                stack.append((t0, tm, v0, vm))
                continue
            total += d
        return total

    a1 = track(lambda t: 1j * t, Rr, -Rr)
    a2 = track(lambda t: Rr * np.exp(1j * t), -math.pi / 2, math.pi / 2)
    if not (np.isfinite(a1) and np.isfinite(a2)):
        return False                          # eksen üzerinde kök: sınırda kararlı → kararsız say
    wn = (a1 + a2) / (2 * math.pi)
    return round(wn) == 0


# ---------------------------------------------------------------------------------------------------------------
# allmargin / margin
# ---------------------------------------------------------------------------------------------------------------
def _wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _allmargin_siso(sys: LTI) -> dict:
    R = _Rat(sys)
    stable = _cl_stable(R, sys)
    pcs = _phase_crossovers(R)
    gm = [(1.0 / m if m > 0 else math.inf) if math.isfinite(m) else 0.0 for _, m in pcs]
    gmf = [w for w, _ in pcs]
    gcs = _gain_crossovers(R)
    pm, pmf = [], []
    for w in gcs:
        if w == 0:
            a0 = _phase_anchor_rat(R)
            pm.append(_wrap180(a0 + 180.0))
        else:
            pm.append(_wrap180(math.degrees(R.phase(w)) + 180.0))
        pmf.append(w)
    dm, dmf = [], []
    T = R.Ts if R.disc else 1.0
    for P, w in zip(pm, pmf):
        if w == 0:
            dm.append(0.0 if P == 0 else math.inf)
        else:
            a = (P % 360.0) if (stable or R.disc) else P
            dm.append(math.radians(a) / (w * T))
        dmf.append(w)
    if not R.disc and R.k != 0 and R.z.size == R.p.size and abs(R.k) >= 1.0 - 1e-12:
        dm.append(0.0)
        dmf.append(math.inf)
    if R.disc and stable and not _cl_stable(R, _extra_delay(sys)):
        dm.append(1.0)
        dmf.append(math.pi / R.Ts)
    return dict(GainMargin=np.asarray(gm), GMFrequency=np.asarray(gmf), PhaseMargin=np.asarray(pm),
                PMFrequency=np.asarray(pmf), DelayMargin=np.asarray(dm), DMFrequency=np.asarray(dmf),
                Stable=bool(stable))


def _extra_delay(sys: LTI) -> LTI:
    out = sys.copy()
    out.input_delay = out.input_delay + 1.0
    return out


def _phase_anchor_rat(R: _Rat) -> float:
    a0, _ = _phase_limit0(R)
    return a0


def allmargin(sys: LTI):
    """Bütün kararlılık payları (MATLAB `allmargin`): sözlük (SISO) ya da [i][j] sözlük listesi (MIMO).

    Alanlar: GainMargin, GMFrequency, PhaseMargin (derece), PMFrequency, DelayMargin (sürekli s / ayrık örnek),
    DMFrequency, Stable (birim negatif geri beslemeli kapalı çevrim)."""
    if sys.is_siso:
        return _allmargin_siso(sys)
    return [[_allmargin_siso(sys[i, j]) for j in range(sys.ninputs)] for i in range(sys.noutputs)]


def margin(sys: LTI):
    """Kazanç ve faz payı (MATLAB `[Gm, Pm, Wcg, Wcp] = margin(sys)`): Gm mutlak, Pm derece; pay yoksa ∞ (frekans
    NaN). Birden çok geçişte |log Gm| ve |Pm| en küçük olanlar."""
    if not sys.is_siso:
        return [[margin(sys[i, j]) for j in range(sys.ninputs)] for i in range(sys.noutputs)]
    a = _allmargin_siso(sys)
    if a["GainMargin"].size:
        with np.errstate(divide="ignore"):
            k = int(np.argmin(np.abs(np.log(a["GainMargin"]))))
        Gm, Wcg = float(a["GainMargin"][k]), float(a["GMFrequency"][k])
    else:
        Gm, Wcg = math.inf, math.nan
    if a["PhaseMargin"].size:
        k = int(np.argmin(np.abs(a["PhaseMargin"])))
        Pm, Wcp = float(a["PhaseMargin"][k]), float(a["PMFrequency"][k])
    else:
        Pm, Wcp = math.inf, math.nan
    return Gm, Pm, Wcg, Wcp


# ---------------------------------------------------------------------------------------------------------------
# Bant genişliği
# ---------------------------------------------------------------------------------------------------------------
def bandwidth(sys: LTI, dbdrop: float = -3.0):
    """Bant genişliği (MATLAB `bandwidth`): |G| ilk kez |G(0)|·10^(dbdrop/20) altına düştüğü frekans (rad/s).
    DC kazancı sonsuzsa NaN, hiç düşmüyorsa ∞. MIMO'da her eleman için (ny, nu)."""
    if not sys.is_siso:
        return np.array([[bandwidth(sys[i, j], dbdrop) for j in range(sys.ninputs)] for i in range(sys.noutputs)])
    if dbdrop >= 0:
        raise ControlError("dbdrop negatif olmalı (ör. −3).")
    g0 = abs(dcgain(_strip(sys)))
    if not math.isfinite(g0):
        return math.nan
    R = _Rat(sys)
    level = g0 * 10.0 ** (dbdrop / 20.0)
    if level == 0:
        return math.inf
    if R.disc:
        F = _padd(R.k * R.k * _mag2_poly_disc(R.z), -level * level * _mag2_poly_disc(R.p))
        cand = [math.acos(c) / R.Ts for c in _real_roots(F, -1.0, 1.0)] if np.any(F) else []
        hi = math.pi / R.Ts
    else:
        F = _padd(R.k * R.k * _mag2_poly_cont(R.z), -level * level * _mag2_poly_cont(R.p))
        cand = [math.sqrt(W) for W in _real_roots(F, 0.0, math.inf)] if np.any(F) else []
        hi = 1e300
    lg = math.log(level)
    best = math.inf
    for w0 in sorted(cand):
        if w0 <= 0:
            continue
        w = _newton(lambda v: R.logmag(v) - lg, lambda v: float(np.real(R.dlog(v))), w0, 0.0, hi)
        if abs(R.logmag(w) - lg) <= 1e-8 and float(np.real(R.dlog(w))) < 0:
            best = min(best, w)
    return best


def _strip(sys: LTI) -> LTI:
    out = sys.copy()
    out.input_delay = np.zeros(sys.ninputs)
    out.output_delay = np.zeros(sys.noutputs)
    if hasattr(out, "io_delay"):
        out.io_delay = np.zeros(sys.shape)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Normlar
# ---------------------------------------------------------------------------------------------------------------
def _smax(sys: StateSpace, w: float) -> float:
    H = _eval_nodelay(sys, _points(sys, np.array([w])))[:, :, 0]
    if not np.all(np.isfinite(H)):
        return math.inf
    return float(np.linalg.svd(H, compute_uv=False)[0]) if H.size else 0.0


def _hamiltonian_freqs(A, B, C, D, g):
    n = A.shape[0]
    m = B.shape[1]
    p = C.shape[0]
    R = g * g * np.eye(m) - D.T @ D
    Ri = np.linalg.inv(R)
    A1 = A + B @ Ri @ D.T @ C
    H = np.block([[A1, B @ Ri @ B.T], [-C.T @ (np.eye(p) + D @ Ri @ D.T) @ C, -A1.T]])
    ev = np.linalg.eigvals(H)
    scale = max(1.0, float(np.max(np.abs(ev)))) if n else 1.0
    im = ev[np.abs(ev.real) <= 1e-8 * scale]
    return np.sort(np.unique(np.round(np.abs(im.imag), 14)))


def _peak_cont(s: StateSpace, tol: float):
    A, B, C, D = s.A, s.B, s.C, s.D
    p = np.linalg.eigvals(A) if s.nx else np.zeros(0)
    cand = [0.0] + [abs(v) for v in p] + [abs(v.imag) for v in p]
    gl, wl = -1.0, 0.0
    for w in cand:
        v = _smax(s, w)
        if v > gl:
            gl, wl = v, w
    gD = float(np.linalg.svd(D, compute_uv=False)[0]) if D.size else 0.0
    if gD > gl:
        gl, wl = gD, math.inf
    for _ in range(100):
        g = (1.0 + 2.0 * tol) * gl
        ws = _hamiltonian_freqs(A, B, C, D, g)
        if ws.size == 0:
            break
        mids = 0.5 * (ws[:-1] + ws[1:]) if ws.size > 1 else ws
        improved = False
        for w in np.concatenate([mids, ws]):
            v = _smax(s, float(w))
            if v > gl:
                gl, wl, improved = v, float(w), True
        if not improved:
            break
    return gl, wl


def _refine_peak(f, w0: float, lo: float, hi: float) -> tuple[float, float]:
    if not math.isfinite(w0):
        return f(w0) if math.isfinite(w0) else math.nan, w0
    span = max(1e-6 * max(w0, 1e-12), 1e-3 * max(w0, 1e-6))
    a, b = max(lo, w0 - span), min(hi, w0 + span)
    best_w, best_v = w0, f(w0)
    if b > a:
        r = minimize_scalar(lambda v: -f(v), bounds=(a, b), method="bounded",
                            options={"xatol": 1e-15 * max(1.0, w0)})
        if -r.fun > best_v:
            best_w, best_v = float(r.x), float(-r.fun)
    return best_v, best_w


def getPeakGain(sys: LTI, tol: float = 1e-10, fband=None):
    """Frekans cevabının en büyük tekil değeri ve frekansı (MATLAB `[gpeak, fpeak] = getPeakGain(sys, tol)`);
    kararsız modelde de tanımlıdır (L∞). Sanal eksende (ayrık: birim çemberde) kutup varsa (∞, kutup frekansı)."""
    if fband is not None:
        lo_b, hi_b = float(fband[0]), float(fband[1])
    else:
        lo_b, hi_b = 0.0, (math.pi / sys.Ts if sys.is_discrete else math.inf)
    s = convert.sminreal(convert.to_ss(_strip(sys)))
    io_dep = isinstance(sys, (TransferFunction, ZPK)) and np.any(sys._io()) and not sys.is_siso
    p = np.linalg.eigvals(s.A) if s.nx else np.zeros(0)
    if sys.is_discrete:
        on = p[np.abs(np.abs(p) - 1.0) <= 1e-10]
        if on.size:
            ww = np.abs(np.angle(on)) / sys.Ts
            return math.inf, float(np.min(ww))
    else:
        on = p[np.abs(p.real) <= 1e-12 * np.maximum(1.0, np.abs(p))]
        if on.size:
            return math.inf, float(np.min(np.abs(on.imag)))
    if io_dep or fband is not None:
        return _peak_grid(sys, lo_b, hi_b)
    if sys.is_discrete:
        n = s.nx
        if n and np.min(np.abs(np.linalg.eigvals(s.A) + 1.0)) > 1e-10:
            I = np.eye(n)
            M = np.linalg.inv(s.A + I)
            Ac = M @ (s.A - I)
            Bc = math.sqrt(2.0) * M @ s.B
            Cc = math.sqrt(2.0) * s.C @ M
            Dc = s.D - s.C @ M @ s.B
            g, wc = _peak_cont(StateSpace(Ac, Bc, Cc, Dc), tol)
            w0 = (2.0 * math.atan(wc) if math.isfinite(wc) else math.pi) / sys.Ts
        else:
            g, w0 = _peak_grid(sys, 0.0, math.pi / sys.Ts)
        for we in (0.0, math.pi / sys.Ts):
            v = _smax(s, we)
            if v > g:
                g, w0 = v, we
        v, w = _refine_peak(lambda x: _smax(s, x), w0, 0.0, math.pi / sys.Ts)
        return _snap_ends(s, *_siso_polish(sys, max(v, g), w, math.pi / sys.Ts), math.pi / sys.Ts)
    g, w0 = _peak_cont(s, tol)
    if not math.isfinite(w0):
        return g, math.inf
    v, w = _refine_peak(lambda x: _smax(s, x), w0, 0.0, math.inf)
    return _snap_ends(s, *_siso_polish(sys, max(v, g), w, math.inf), math.inf)


def _snap_ends(s: StateSpace, g: float, w: float, hi: float):
    """Tepe ω = 0'a (ya da ayrıkta π/Ts'ye) inceltme çözünürlüğü kadar yakınsa uç noktanın kendisidir."""
    for we in (0.0, hi):
        if math.isfinite(we) and abs(w - we) <= 1e-7 * max(1.0, abs(we)):
            v = _smax(s, we)
            if v >= g * (1 - 1e-12):
                return max(v, g), we
    return g, w


def _siso_polish(sys: LTI, g: float, w: float, hi: float):
    """SISO tepe frekansını d log|G|/dω = 0 kökü olarak makine doğruluğunda inceltir."""
    if not sys.is_siso or not (0.0 < w < hi):
        return g, w
    try:
        R = _Rat(_strip(sys))
    except ControlError:
        return g, w

    def d(v):
        return float(np.real(R.dlog(v)))

    span = 1e-4 * w
    for _ in range(20):
        a, b = max(0.0, w - span), min(hi, w + span) if math.isfinite(hi) else w + span
        da, db = d(a), d(b)
        if da > 0 > db:
            wr = brentq(d, a, b, xtol=1e-16 * w, rtol=4 * np.finfo(float).eps, maxiter=200)
            vr = abs(R.value(wr))
            return (vr, wr) if vr >= g * (1 - 1e-12) else (g, w)
        span *= 2
        if span > w:
            break
    return g, w


def _peak_grid(sys: LTI, lo: float, hi: float):
    """Gecikmenin tekil değerleri değiştirdiği MIMO / bant sınırlı durum: yoğun ızgara + yerel inceltme."""
    w = _auto_w(sys)
    if math.isfinite(hi):
        w = np.concatenate([w[(w >= lo) & (w <= hi)], [lo, hi]])
    else:
        w = np.concatenate([w[w >= lo], [lo]])
    w = np.unique(w)

    def f(x):
        H = freqresp(sys, np.array([x]))[:, :, 0]
        return float(np.linalg.svd(H, compute_uv=False)[0])

    vals = np.array([f(x) for x in w])
    k = int(np.argmax(vals))
    return _refine_peak(f, float(w[k]), lo, hi if math.isfinite(hi) else 1e300)


def norm(sys: LTI, p=2, tol: float = 1e-10):
    """Sistem normu (MATLAB `norm`): p = 2 → H2 (sayı); p = inf → (H∞ / L∞ değeri, tepe frekansı)."""
    if p in (2, "2"):
        s = convert.to_ss(_strip(sys))
        n = s.nx
        if n:
            ev = np.linalg.eigvals(s.A)
            unstable = np.any(np.abs(ev) >= 1 - 1e-12) if s.is_discrete else \
                np.any(ev.real >= -1e-12 * np.maximum(1.0, np.abs(ev)))
            if unstable:
                return math.inf
        if not s.is_discrete:
            if np.any(s.D != 0):
                return math.inf
            if n == 0:
                return 0.0
            P = sla.solve_continuous_lyapunov(s.A, -s.B @ s.B.T)
            return float(math.sqrt(max(0.0, np.trace(s.C @ P @ s.C.T))))
        val = np.trace(s.D @ s.D.T)
        if n:
            P = sla.solve_discrete_lyapunov(s.A, s.B @ s.B.T)
            val += np.trace(s.C @ P @ s.C.T)
        return float(math.sqrt(max(0.0, val)))
    if p in (math.inf, "inf", np.inf):
        return getPeakGain(sys, tol)
    raise ControlError("Norm türü 2 ya da inf olmalı.")

