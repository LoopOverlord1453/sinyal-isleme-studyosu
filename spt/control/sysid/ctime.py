"""Sürekli zamanlı modeller: `tfest` (transfer fonksiyonu) ve `procest` (süreç modelleri) — MATLAB anlamıyla.

Ölçüt, MATLAB'in bu iki komuttaki varsayılan gürültü yapısıyla aynı **çıkış hatasıdır** (OE): verinin örnekler arası
davranışıyla (varsayılan ZOH) tam ayrıklaştırılmış modelin benzetim hatasının kareleri toplamı

    V(θ) = Σ_t (y(t) − G_θ(q) u(t))²,  G_θ: G(s)·e^{−τs}'nin kesirli gecikme dahil tam ZOH (ya da FOH) eşdeğeri.

Başlangıç koşulu 'zero', 'estimate' (ayrık gerçeklemenin başlangıç durumu ek parametre) ya da 'auto' (MATLAB
AutoInitThreshold = 1.05 kuralı). En küçükleme Levenberg–Marquardt (merkezi farklı sayısal Jacobian; başlangıç
durumu sütunları analitik), optimuma kadar.

- `tfest` başlangıcı: SRIVC (Young'ın basitleştirilmiş yinelemeli araç değişkeni yöntemi; süzülmüş türevler sürekli
  süzgeçlerin ZOH eşdeğeriyle) — ilk payda ayrık ARX'in d2c dönüşümünden.
- `procest` başlangıcı: tfest tabanlı (kutuplardan zaman sabitleri), gecikme için ızgara taraması.
- Gecikme (`iodelay` / Td) tahmin edilecekse ızgara + sürekli parametre olarak LM.
MATLAB'in kendi başlatma yöntemi ayrıntılı yayımlanmamıştır; aynı optimuma ulaşıldığı test edilir.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg, signal

from spt.control.errors import ControlError
from spt.control.sysid._util import levenberg_marquardt, shift
from spt.control.sysid.data import IDData, check_complete
from spt.control.sysid.models import IDTF, IDProc, Report, c2d_exact, make_fit, parse_proc_type, tf_to_ss

__all__ = ["tfest", "procest"]


# ============================================================================================ ayrıklaştırma
def _ct_to_dtf(num, den, delay, Ts, hold="zoh"):
    """Sürekli SISO G(s)e^{−τs} → (pay_q, payda_q, d): y = q^{-d}·pay/payda·u (q⁻¹ artan kuvvetler)."""
    if not np.isfinite(delay) or delay < 0:
        raise ControlError("Gecikme negatif olamaz")
    A, B, C, D = tf_to_ss(num, den)
    s = c2d_exact(A, B, C, D, Ts, [delay], (hold,))
    n = s.nx
    if n == 0:
        return np.array([float(s.D[0, 0])]), np.array([1.0]), int(s.delay[0])
    b, a = signal.ss2tf(s.A, s.B, s.C, s.D)
    b = np.atleast_1d(np.real(b[0]))
    a = np.real(a)
    return b, a, int(s.delay[0])


def _core(a):
    """Ayrık paydadan kesirli gecikme durumunun z = 0 kökünü (sondaki sıfır katsayıyı) atar."""
    a = np.asarray(a, float)
    n = len(a)
    tol = 1e-10 * np.max(np.abs(a))
    while n > 1 and abs(a[n - 1]) <= tol:
        n -= 1
    return a[:n]


class _CTOE:
    """Sürekli zamanlı modellerin OE ölçütü. `build(θ)` → giriş başına (pay_q, payda_q, d)."""

    def __init__(self, exps, build, ntheta, ic, accept_theta=None, bounds=None):
        self.exps = exps
        self.build = build
        self.nth = ntheta
        self.ic = ic
        self.accept_theta = accept_theta
        self.lo, self.hi = (np.full(ntheta, -np.inf), np.full(ntheta, np.inf)) if bounds is None else             (np.asarray(bounds[0], float), np.asarray(bounds[1], float))
        self._nx = None

    def project(self, z):
        z = np.array(z, float)
        z[:self.nth] = np.clip(z[:self.nth], self.lo, self.hi)
        return z

    def nx_of(self, th):
        parts = self.build(th)
        return sum(len(_core(a)) - 1 for _, a, _ in parts)

    def setup(self, th0):
        self._nx = self.nx_of(th0) if self.ic == "estimate" else 0
        return self

    def split(self, z):
        th = z[:self.nth]
        s = z[self.nth:].reshape(len(self.exps), self._nx) if self._nx else None
        return th, s

    def accept(self, z):
        th = z[:self.nth]
        if self.accept_theta is not None and not self.accept_theta(th):
            return False
        try:
            parts = self.build(th)
        except (ControlError, linalg.LinAlgError, ValueError):
            return False
        nx = sum(len(_core(a)) - 1 for _, a, _ in parts)
        if self._nx and nx != self._nx:
            return False
        return all(np.all(np.isfinite(a)) and np.all(np.isfinite(b)) for b, a, _ in parts)

    def _sim(self, th, U):
        parts = self.build(th)
        y = np.zeros(len(U))
        for j, (b, a, d) in enumerate(parts):
            y += signal.lfilter(b, a, shift(U[:, j], d))
        return y, parts

    def _free_basis(self, parts, N):
        """Serbest cevap tabanı: yalnız modelin kendi durumları (kesirli gecikmenin ek durumu — önceki giriş
        örneği — sıfır geçmişli; MATLAB'in x0 boyutu böyle: tfest / procest FPE'sinden ölçüldü)."""
        den = np.array([1.0])
        for _, a, _ in parts:
            den = np.convolve(den, _core(a))
        h = signal.lfilter([1.0], den, np.r_[1.0, np.zeros(N - 1)])
        return np.column_stack([shift(h, i) for i in range(self._nx)])

    def resid(self, z):
        th, s = self.split(z)
        out = []
        for k, (y, U) in enumerate(self.exps):
            yh, parts = self._sim(th, U)
            if s is not None:
                yh = yh + self._free_basis(parts, len(y)) @ s[k]
            out.append(y - yh)
        return np.concatenate(out)

    def jac(self, z):
        th, s = self.split(z)

        def r_th(t):
            return self.resid(np.r_[t, z[self.nth:]])

        # merkezi fark; sınırdaki parametrede (ör. gecikme = 0) sınır dışına çıkmadan tek yönlü fark
        r0 = r_th(th)
        J = np.empty((len(r0), self.nth))
        eps3 = np.finfo(float).eps ** (1 / 3)
        for k in range(self.nth):
            h = eps3 * max(abs(th[k]), 1.0)
            tp, tm = th.copy(), th.copy()
            tp[k] += h
            tm[k] -= h
            up, dn = tp[k] <= self.hi[k], tm[k] >= self.lo[k]
            if up and dn:
                J[:, k] = (r_th(tp) - r_th(tm)) / (2 * h)
            elif up:
                J[:, k] = (r_th(tp) - r0) / h
            else:
                J[:, k] = (r0 - r_th(tm)) / h
        if not self._nx:
            return J
        cols = np.zeros((J.shape[0], self._nx * len(self.exps)))
        parts = self.build(th)
        off = 0
        for k, (y, U) in enumerate(self.exps):
            N = len(y)
            cols[off:off + N, k * self._nx:(k + 1) * self._nx] = -self._free_basis(parts, N)
            off += N
        return np.hstack([J, cols])

    def best_s(self, th):
        """Sabit θ için başlangıç durumlarının en küçük kareler çözümü (serbest cevap sütunları analitik)."""
        if not self._nx:
            return np.zeros(0)
        z0 = np.r_[th, np.zeros(self._nx * len(self.exps))]
        r0 = self.resid(z0)
        parts = self.build(th)
        Js = np.zeros((len(r0), self._nx * len(self.exps)))
        off = 0
        for k, (y, _) in enumerate(self.exps):
            Js[off:off + len(y), k * self._nx:(k + 1) * self._nx] = -self._free_basis(parts, len(y))
            off += len(y)
        sol, *_ = np.linalg.lstsq(Js, -r0, rcond=None)
        return sol

    def cost(self, th):
        z = np.r_[th, self.best_s(th)] if self._nx else th
        r = self.resid(z)
        return float(r @ r)


def _auto_ic(exps, build, th0, nth, accept_theta=None, threshold=1.05):
    ez = _CTOE(exps, build, nth, "zero", accept_theta).setup(th0)
    ee = _CTOE(exps, build, nth, "estimate", accept_theta).setup(th0)
    r0 = ez.resid(th0)
    r1 = ee.resid(np.r_[th0, ee.best_s(th0)])
    return "estimate" if np.linalg.norm(r1) > 0 and np.linalg.norm(r0) / np.linalg.norm(r1) > threshold else "zero"


def _optimize(exps, build, inits, nth, ic, accept_theta, max_iter, bounds=None):
    used = ic
    if ic == "auto":
        used = _auto_ic(exps, build, inits[0], nth, accept_theta)
    best = None
    for th0 in inits:
        pr = _CTOE(exps, build, nth, used, accept_theta, bounds)
        try:
            pr.setup(th0)
            th0 = np.clip(np.asarray(th0, float), pr.lo, pr.hi)
            z0 = np.r_[th0, pr.best_s(th0)] if pr._nx else th0
            if not pr.accept(z0):
                continue
            res = levenberg_marquardt(pr.resid, z0, pr.jac, accept=pr.accept, max_iter=max_iter,
                                      project=pr.project)
        except (ControlError, linalg.LinAlgError, ValueError):
            continue
        if best is None or res.cost < best[1].cost:
            best = (pr, res)
    if best is None:
        raise ControlError("Kestirim başarısız: uygun başlangıç modeli bulunamadı")
    return best, used


def _report_and_cov(pr, res, data, npar, method, ic, opts):
    r = pr.resid(res.x)
    e_list, y_list = [], []
    off = 0
    for y, _ in pr.exps:
        N = len(y)
        e_list.append(r[off:off + N].reshape(-1, 1))
        y_list.append(y.reshape(-1, 1))
        off += N
    nx0 = pr._nx * len(pr.exps)
    fit = make_fit(y_list, e_list, npar + nx0)
    V = float(r @ r) / off
    lam_d = V * off / (off - npar - nx0)
    lam = lam_d * data.Ts  # MATLAB: sürekli modelde NoiseVariance = Ts·λ (gürültü yoğunluğu)
    J = pr.jac(res.x)[:, :npar]
    try:  # λ·(J_θᵀJ_θ)⁻¹ (başlangıç durumu sabit; MATLAB'in OE renkli artık düzeltmesi yapılmaz — bkz. linear)
        cov = lam_d * np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        cov = None
    _, s = pr.split(res.x)
    x0 = [s[k].copy() for k in range(len(pr.exps))] if s is not None else None
    rep = Report(method, fit, ic, res.iterations, res.why, npar, nx0, x0, data.name, off, data.Ts, opts,
                 f"{method} ile tahmin edildi (çıkış hatası, sürekli zaman), uyum %{fit.FitPercent.round(2).tolist()}")
    return lam, cov, rep


def _prep(data):
    if not isinstance(data, IDData):
        raise ControlError("Veri IDData olmalı")
    check_complete(data)
    if data.ny != 1:
        raise ControlError("Bu kestirici tek çıkışlı veri ister (çıkış seçin)")
    if data.nu == 0:
        raise ControlError("Bu kestirici giriş verisi ister")
    return [(y[:, 0].copy(), u.copy()) for y, u in zip(data.ys, data.us)]


# ============================================================================================ SRIVC
def _filt_ct(num, den, x, Ts):
    """Sürekli süzgeç num/den'i (uygun) örneklenmiş x'e uygular (ZOH eşdeğeri)."""
    b, a, d = _ct_to_dtf(num, den, 0.0, Ts)
    return signal.lfilter(b, a, x)


def _srivc(exps, Ts, npo, nz, delay, den0, iters=20):
    """Tek giriş için SRIVC; döner (num, den) azalan s kuvvetleri, payda mönik."""
    den = np.asarray(den0, float)
    num = None
    for _ in range(iters):
        rows, rhs, inst = [], [], []
        for y, U in exps:
            u = U[:, 0]
            ud = _delay_signal(u, delay, Ts)
            if num is None:
                xh = y
            else:
                xh = _filt_ct(num, den, ud, Ts)
            yf = [_filt_ct(_spow(i), den, y, Ts) for i in range(npo + 1)]
            uf = [_filt_ct(_spow(i), den, ud, Ts) for i in range(nz + 1)]
            xf = [_filt_ct(_spow(i), den, xh, Ts) for i in range(npo)]
            Phi = np.column_stack([-yf[npo - 1 - i] for i in range(npo)] + [uf[nz - i] for i in range(nz + 1)])
            Z = np.column_stack([-xf[npo - 1 - i] for i in range(npo)] + [uf[nz - i] for i in range(nz + 1)])
            t0 = min(len(y) // 10, 20)
            rows.append(Phi[t0:])
            inst.append(Z[t0:])
            rhs.append(yf[npo][t0:])
        Phi, Z, Y = np.vstack(rows), np.vstack(inst), np.concatenate(rhs)
        try:
            th = np.linalg.lstsq(Z.T @ Phi, Z.T @ Y, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        new_den = np.r_[1.0, th[:npo]]
        new_num = th[npo:]
        if not np.all(np.isfinite(new_den)) or np.any(np.real(np.roots(new_den)) >= 0):
            break
        change = np.max(np.abs(new_den - den)) / max(np.max(np.abs(den)), 1e-12)
        den, num = new_den, new_num
        if change < 1e-9:
            break
    return num, den


def _spow(i):
    return np.r_[1.0, np.zeros(i)]


def _delay_signal(u, tau, Ts):
    """ZOH sinyalinin τ kadar gecikmesi (örnek noktalarında): kesirli kısım için sonraki örneğe yuvarlar (yaklaşık;
    yalnız başlangıçta kullanılır)."""
    d = int(np.ceil(tau / Ts - 1e-9))
    return shift(u, d)


def _d2c_den(a_q, Ts, npo):
    """Ayrık payda (q⁻¹) → sürekli payda tahmini (kutuplar s = log(z)/Ts); uygun değilse None."""
    try:
        p = np.roots(a_q)
    except (ValueError, np.linalg.LinAlgError):
        return None
    if len(p) != npo or np.any(np.abs(p) >= 1) or np.any(np.abs(p) < 1e-8):
        return None
    sp = np.log(p.astype(complex)) / Ts
    den = np.real(np.poly(sp))
    if not np.all(np.isfinite(den)) or np.any(np.real(np.roots(den)) >= 0):
        return None
    return den


# ============================================================================================ tfest
def tfest(data: IDData, np_, nz=None, iodelay=None, *, Ts: float = 0.0, initial_condition: str = "auto",
          max_iter: int = 200, init_sys: IDTF | None = None) -> IDTF:
    """Transfer fonksiyonu kestirimi (MATLAB `tfest(data, np, nz, iodelay)`).

    - Sürekli zaman (varsayılan, Ts=0): G(s) = (b0 s^nz + … + b_nz)/(s^np + a1 s^(np−1) + … + a_np)·e^{−τs};
      `nz` varsayılanı np − 1 (MATLAB). Çok girişte np, nz, iodelay giriş başına listeler olabilir.
    - `iodelay`: sabit gecikme (s; varsayılan 0) ya da NaN (tahmin edilir).
    - `Ts > 0` (MATLAB 'Ts' seçeneği): ayrık zamanlı model (q⁻¹), OE yapısıyla (bkz. `oe`).
    Ölçüt: çıkış hatası (ayrıntı modül belgesinde). `init_sys`: başlangıç modeli (verilirse SRIVC atlanır)."""
    exps = _prep(data)
    nu = data.nu
    npl = np.atleast_1d(np.asarray(np_, int)) * np.ones(nu, int)
    nzl = (np.maximum(npl - 1, 0) if nz is None else np.atleast_1d(np.asarray(nz, int)) * np.ones(nu, int))
    if iodelay is None:
        dl = np.zeros(nu)
    else:
        dl = np.atleast_1d(np.asarray(iodelay, float)) * np.ones(nu)
    if np.any(npl < 0) or np.any(nzl < 0):
        raise ControlError("Kutup ve sıfır sayıları negatif olamaz")
    if Ts and Ts > 0:
        return _tfest_discrete(data, npl, nzl, dl, initial_condition, max_iter)
    if np.any(nzl > npl):
        raise ControlError("Sürekli zamanda sıfır sayısı kutup sayısını aşamaz")
    est_delay = np.isnan(dl)
    Tsd = data.Ts
    hold = data.intersample

    # ---- parametre düzeni: giriş başına [pay (nz+1), payda (np)], ardından tahmin edilen gecikmeler
    sizes = [int(nzl[j] + 1 + npl[j]) for j in range(nu)]
    nth = int(sum(sizes) + est_delay.sum())

    def unpack(th):
        k = 0
        out = []
        for j in range(nu):
            num = th[k:k + nzl[j] + 1]
            k += nzl[j] + 1
            den = np.r_[1.0, th[k:k + npl[j]]]
            k += npl[j]
            out.append([num, den, dl[j]])
        for j in range(nu):
            if est_delay[j]:
                out[j][2] = th[k]
                k += 1
        return out

    def build(th):
        return [_ct_to_dtf(n, d, tau, Tsd, hold[j]) for j, (n, d, tau) in enumerate(unpack(th))]

    def accept_theta(th):
        for n, d, tau in unpack(th):
            if tau < 0 or not np.all(np.isfinite(d)):
                return False
            if len(d) > 1 and np.max(np.abs(d)) > 1e12:
                return False
        return True

    # ---- başlangıç
    inits = []
    if init_sys is not None:
        th = []
        for j in range(nu):
            n = np.asarray(init_sys.num[j], float)
            d = np.asarray(init_sys.den[j], float)
            n, d = n / d[0], d / d[0]
            th += list(np.pad(n, (nzl[j] + 1 - len(n), 0))[-(nzl[j] + 1):]) + list(d[1:1 + npl[j]])
        th += [init_sys.io_delay[j] for j in range(nu) if est_delay[j]]
        inits.append(np.array(th))
    else:
        delays0 = _initial_delays(exps, data, dl, est_delay)
        for cand in delays0:
            th = []
            for j in range(nu):
                ej = [(y - _others(exps_k, j, nu), U[:, [j]]) for (y, U), exps_k in zip(exps, exps)]
                num, den = _tf_init_single(ej, Tsd, int(npl[j]), int(nzl[j]), cand[j])
                th += list(num) + list(den[1:])
            th += [cand[j] for j in range(nu) if est_delay[j]]
            inits.append(np.array(th, float))
    lo = np.full(nth, -np.inf)
    lo[sum(sizes):] = 0.0  # tahmin edilen gecikmeler ≥ 0
    (pr, res), used = _optimize(exps, build, inits, nth, initial_condition, accept_theta, max_iter,
                                (lo, np.full(nth, np.inf)))
    th = res.x[:nth]
    parts = unpack(th)
    npar = nth
    lam, cov, rep = _report_and_cov(pr, res, data, npar, "TFEST", used,
                                    {"np": npl.tolist(), "nz": nzl.tolist(), "iodelay": dl.tolist(),
                                     "initial_condition": initial_condition})
    model = IDTF([p[0] for p in parts], [p[1] for p in parts], 0.0, io_delay=[p[2] for p in parts],
                 noise_variance=lam, input_names=data.input_names, output_names=data.output_names)
    model.covariance = cov
    model.report = rep
    return model


def _others(exp, j, nu):
    return 0.0  # MISO başlangıcında diğer girişlerin katkısı ihmal edilir (PEM düzeltir)


def _initial_delays(exps, data, dl, est_delay):
    """Tahmin edilecek gecikmeler için aday kümeleri (ızgara)."""
    nu = data.nu
    if not est_delay.any():
        return [dl.copy()]
    from spt.control.sysid.nonparam import delayest
    cands = []
    base = dl.copy()
    for j in range(nu):
        if est_delay[j]:
            try:
                nk = delayest(data.select(inputs=[j]), 2, 2, 0, 20)
            except ControlError:
                nk = 1
            base[j] = max(nk - 1, 0) * data.Ts
    for f in (0.0, 0.5, -0.5, 1.0):
        c = base.copy()
        for j in range(nu):
            if est_delay[j]:
                c[j] = max(base[j] + f * data.Ts, 0.0)
        cands.append(c)
    return cands


def _tf_init_single(exps, Ts, npo, nz, delay):
    """Tek giriş için başlangıç: ayrık ARX → d2c payda → SRIVC."""
    d = int(np.ceil(delay / Ts - 1e-9)) + 1
    rows, Ys = [], []
    for y, U in exps:
        u = U[:, 0]
        cols = [-shift(y, k) for k in range(1, npo + 1)] + [shift(u, d + k) for k in range(max(npo, nz + 1))]
        t0 = npo + d + max(npo, nz + 1)
        rows.append(np.column_stack(cols)[t0:])
        Ys.append(y[t0:])
    th = np.linalg.lstsq(np.vstack(rows), np.concatenate(Ys), rcond=None)[0] if npo else np.zeros(0)
    den0 = _d2c_den(np.r_[1.0, th[:npo]], Ts, npo) if npo else np.array([1.0])
    if den0 is None:
        bw = 0.5 / Ts * 0.2
        den0 = np.real(np.poly(-bw * np.ones(npo)))
    if npo == 0:
        rows, Ys = [], []
        for y, U in exps:
            ud = _delay_signal(U[:, 0], delay, Ts)
            rows.append(ud[:, None])
            Ys.append(y)
        g = np.linalg.lstsq(np.vstack(rows), np.concatenate(Ys), rcond=None)[0]
        return np.r_[np.zeros(nz), g], np.array([1.0])
    num, den = _srivc(exps, Ts, npo, nz, delay, den0)
    if num is None:
        num = np.zeros(nz + 1)
        num[-1] = den[-1]
    return num, den


def _tfest_discrete(data, npl, nzl, dl, initial_condition, max_iter):
    from spt.control.sysid.linear import oe

    if np.any(np.isnan(dl)):
        raise ControlError("Ayrık zamanlı tfest'te gecikme tamsayı örnek olarak verilmeli")
    nu = data.nu
    # MATLAB ayrık tfest: pay [0, b1 … b_nz] (bir örnek örtük gecikme) · z^(−iodelay) (R2025b'de ölçüldü)
    nb = np.maximum(nzl, 1)
    iod = np.round(dl).astype(int)
    nk = iod + 1
    m = oe(data, list(nb) + list(npl) + list(nk), initial_condition=initial_condition, max_iter=max_iter)
    B = m.B if nu > 1 else [m.B]
    F = m.F if nu > 1 else [m.F]
    num = [np.asarray(B[j], float)[iod[j]:] for j in range(nu)]
    tfm = IDTF(num, [np.asarray(F[j], float) for j in range(nu)], data.Ts, io_delay=iod,
               noise_variance=m.noise_variance, input_names=data.input_names, output_names=data.output_names)
    tfm.covariance = m.covariance
    tfm.report = m.report
    tfm.report.method = "TFEST"
    return tfm


# ============================================================================================ procest
_PBOUNDS = {"Kp": (-np.inf, np.inf), "Tp1": (0.0, np.inf), "Tp2": (0.0, np.inf), "Tp3": (0.0, np.inf),
            "Tw": (0.0, np.inf), "Zeta": (0.0, np.inf), "Td": (0.0, np.inf), "Tz": (-np.inf, np.inf)}


def procest(data: IDData, type_="P1D", *, initial_condition: str = "auto", init=None, max_iter: int = 200,
            td_max: float | None = None) -> IDProc:
    """Süreç modeli kestirimi (MATLAB `procest(data, 'P1D')`), sürekli zaman, çıkış hatası ölçütü.

    `type_`: 'P1', 'P1D', 'P2', 'P2D', 'P2U', 'P2DU', 'P2DZ', 'P2DZU', 'P1I', 'P1DI', 'P3', 'P3D', 'P3U' …
    (çok girişte giriş başına liste). Sınırlar MATLAB varsayılanları gibi: Tp*, Tw, Zeta, Td ≥ 0; `td_max`
    (varsayılan veri süresinin yarısı). `init`: başlangıç parametre sözlükleri (giriş başına).
    Gürültü modeli: yok (DisturbanceModel 'none'); MATLAB'in varsayılanı 'estimate' — fark belgede.
    Giriş kaydırması (InputOffset) tahmin edilmez (MATLAB 'auto' yalnız integratörlü modellerde kaydırma ekleyebilir)."""
    exps = _prep(data)
    nu = data.nu
    types = [type_] * nu if isinstance(type_, str) else list(type_)
    if len(types) != nu:
        raise ControlError("Süreç türü sayısı giriş sayısına eşit olmalı")
    structs = [parse_proc_type(t) for t in types]
    names = [IDProc.active_names(st) for st in structs]
    Tsd = data.Ts
    hold = data.intersample
    if td_max is None:
        td_max = 0.5 * min(len(y) for y, _ in exps) * Tsd

    def unpack(th):
        out, k = [], 0
        for j in range(nu):
            p = {}
            for nm in names[j]:
                p[nm] = th[k]
                k += 1
            out.append(p)
        return out

    def model_of(th):
        return IDProc(types, unpack(th))

    def build(th):
        m = model_of(th)
        out = []
        for j in range(nu):
            n, d, td = m.tf_coeffs(j)
            dd = d / d[0] if d[0] != 0 else d
            out.append(_ct_to_dtf(n / d[0], dd, td, Tsd, hold[j]))
        return out

    def accept_theta(th):
        for j, p in enumerate(unpack(th)):
            for nm, v in p.items():
                lo, hi = _PBOUNDS[nm]
                if nm == "Td":
                    hi = td_max
                if not lo <= v <= hi or not np.isfinite(v):
                    return False
            n, d, _ = model_of(th).tf_coeffs(j)
            if d[0] == 0:  # baş katsayı sıfır: derece düşer
                return False
        return True

    nth = sum(len(n) for n in names)
    if init is not None:
        init = [init] if isinstance(init, dict) else list(init)
        th0 = np.array([float(init[j].get(nm, 1.0)) for j in range(nu) for nm in names[j]])
        inits = [th0]
    else:
        inits = _proc_inits(data, exps, structs, names, td_max)
    lo = np.array([_PBOUNDS[nm][0] for j in range(nu) for nm in names[j]])
    hi = np.array([td_max if nm == "Td" else _PBOUNDS[nm][1] for j in range(nu) for nm in names[j]])
    (pr, res), used = _optimize(exps, build, inits, nth, initial_condition, accept_theta, max_iter, (lo, hi))
    th = res.x[:nth]
    lam, cov, rep = _report_and_cov(pr, res, data, nth, "PROCEST", used,
                                    {"type": types, "initial_condition": initial_condition})
    model = IDProc(types, unpack(th), noise_variance=lam, input_names=data.input_names,
                   output_names=data.output_names)
    model.covariance = cov
    model.report = rep
    return model


def _proc_from_tf(st, num, den, td):
    """Sürekli TF (azalan s) → süreç parametreleri (yaklaşık; başlangıç için)."""
    den = np.asarray(den, float) / den[-1] if den[-1] != 0 else np.asarray(den, float)
    num = np.asarray(num, float)
    K = num[-1] / 1.0 if len(num) else 1.0
    p = {"Kp": float(K)}
    r = np.roots(den) if len(den) > 1 else np.zeros(0)
    taus = sorted([1.0 / max(abs(x.real), 1e-6) for x in r], reverse=True) if len(r) else []
    if st["under"]:
        if len(den) >= 3:
            a2, a1 = den[-3], den[-2]
            Tw = np.sqrt(abs(a2)) if a2 != 0 else 1.0
            p["Tw"] = float(Tw)
            p["Zeta"] = float(max(a1 / (2 * Tw), 0.05))
        else:
            p["Tw"], p["Zeta"] = 1.0, 0.7
        if st["np"] == 3:
            p["Tp3"] = float(taus[-1]) if taus else 0.1
    else:
        for i in range(st["np"]):
            p[f"Tp{i + 1}"] = float(taus[i]) if i < len(taus) else 0.1 * (i + 1)
    if st["delay"]:
        p["Td"] = float(max(td, 0.0))
    if st["zero"]:
        p["Tz"] = float(num[0] / num[-1]) if len(num) > 1 and num[-1] != 0 else 0.0
    return p


def _proc_inits(data, exps, structs, names, td_max):
    nu = data.nu
    Ts = data.Ts
    # gecikme adayları: ayrık gecikme kestiriminden ızgara
    from spt.control.sysid.nonparam import delayest
    td_cands = []
    for j in range(nu):
        if structs[j]["delay"]:
            try:
                nk = delayest(data.select(inputs=[j]), 2, 2, 0, min(30, max(2, int(td_max / Ts))))
            except ControlError:
                nk = 1
            c = max(nk - 1, 0) * Ts
            td_cands.append([max(c + f * Ts, 0.0) for f in (0.5, 0.0, 1.0, -0.5)])
        else:
            td_cands.append([0.0])
    inits = []
    for ci in range(4):
        th = []
        for j in range(nu):
            st = structs[j]
            td = td_cands[j][min(ci, len(td_cands[j]) - 1)]
            npo = st["np"] + (1 if st["integ"] else 0)
            nz = 1 if st["zero"] else 0
            ej = [(y, U[:, [j]]) for y, U in exps]
            if st["integ"]:
                # integratörlü: girişin integrali üzerinden integratörsüz kısmı başlat
                ej = [(y, (np.cumsum(U[:, j]) * Ts)[:, None]) for y, U in exps]
                npo -= 1
            try:
                num, den = _tf_init_single(ej, Ts, npo, nz, td)
                p = _proc_from_tf(st, num, den, td)
            except (ControlError, np.linalg.LinAlgError, ValueError):
                p = {}
            th += [float(p.get(nm, {"Kp": 1.0, "Zeta": 0.7}.get(nm, 1.0))) for nm in names[j]]
        th = np.array(th)
        if not any(np.allclose(th, t) for t in inits):
            inits.append(th)
    return inits
