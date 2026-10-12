"""Tanımlanmış model sınıfları (MATLAB `idpoly`, `idss`, `idtf`, `idproc`, `idfrd`) ve ortak hesaplar.

Her model, değerlendirme (sim / predict / compare / resid) için **ayrık yenilik (innovations) biçimine** çevrilir:

    x(t+1) = A x(t) + B w(t) + K e(t)
    y(t)   = C x(t) + D w(t) + e(t),        w_j(t) = u_j(t − d_j)  (d_j: tamsayı giriş gecikmesi)

`E[e eᵀ] = NoiseVariance`. Giriş gecikmesi durum değil saf gecikmedir (başlangıç öncesi giriş sıfır) — MATLAB'in
idpoly → ss dönüşümüyle aynı: `arx [2 3 2]` modelinde başlangıç durumu 3 boyutludur (R2025b'de ölçüldü; FPE'deki
parametre sayısı ve atılan satır sayısı). Sürekli zamanlı modeller verinin örneklemesiyle (ZOH / FOH, kesirli
gecikme dahil tam) ayrıklaştırılır.

Polinomlar q⁻¹'in artan kuvvetleridir (MATLAB idpoly): `B = [0, 0, b1, b2]` → nk = 2.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import linalg

from spt.control.errors import ControlError
from spt.control.sysid._util import pmul, trim

__all__ = ["IDModel", "IDPoly", "IDSS", "IDTF", "IDProc", "IDFRD", "Fit", "Report", "DSS", "idpoly", "idss",
           "idtf", "idproc"]


# ============================================================================================ rapor
@dataclass
class Fit:
    """MATLAB `Report.Fit`: FitPercent (çıkış başına), LossFcn, MSE, FPE, AIC, AICc, nAIC, BIC.

    R2025b'de ölçülen tanımlar (N: tüm örnek sayısı, d: serbest parametre + tahmin edilen başlangıç durumu sayısı,
    E: tahmin verisindeki artıklar (odak 'prediction': bir adım öngörü hatası)):
    LossFcn = det(EᵀE/N) (tek çıkışta ΣE²/N), MSE = ΣE²/(N·ny), FPE = LossFcn·(1+d/N)/(1−d/N),
    AIC = N·log(LossFcn) + 2d + N·ny·(log 2π + 1), AICc = AIC + 2d(d+1)/(N−d−1), nAIC = log(LossFcn) + 2d/N,
    BIC = N·log(LossFcn) + N·ny·(log 2π + 1) + d·log N, FitPercent = 100·(1 − ‖y−ŷ‖/‖y−ȳ‖)."""
    FitPercent: np.ndarray
    LossFcn: float
    MSE: float
    FPE: float
    AIC: float
    AICc: float
    nAIC: float
    BIC: float

    def as_dict(self) -> dict:
        return {"FitPercent": self.FitPercent, "LossFcn": self.LossFcn, "MSE": self.MSE, "FPE": self.FPE,
                "AIC": self.AIC, "AICc": self.AICc, "nAIC": self.nAIC, "BIC": self.BIC}


@dataclass
class Report:
    """Tahmin raporu (MATLAB `sys.Report`)."""
    method: str
    fit: Fit | None = None
    initial_condition: str = "zero"
    iterations: int = 0
    why_stop: str = ""
    n_free: int = 0
    n_x0: int = 0
    x0: list | None = None
    data_name: str = ""
    data_length: int = 0
    Ts: float = 0.0
    options: dict = field(default_factory=dict)
    status: str = ""
    warnings: list = field(default_factory=list)

    @property
    def Fit(self) -> Fit | None:  # MATLAB adıyla erişim
        return self.fit


def make_fit(y_list, e_list, npar: int, loss: str = "det") -> Fit:
    """Artıklardan MATLAB ölçütleri (bkz. `Fit`). R2025b'de ölçülen çok çıkışlı tanımlar (ARX): LossFcn =
    trace(EᵀE/N) (`loss='trace'`) ya da det(EᵀE/N); MSE = trace(EᵀE)/N; FPE, nAIC, AIC, BIC det(EᵀE/N) ile;
    AIC/BIC sabiti N·(ny·log 2π + 1). Çok deneyde FitPercent ve MSE deney başına."""
    E = np.vstack(e_list)
    N, ny = E.shape
    S = E.T @ E / N
    V = float(np.linalg.det(S)) if ny > 1 else float(S[0, 0])
    d = int(npar)
    logV = np.log(V) if V > 0 else -np.inf
    c = N * (ny * np.log(2 * np.pi) + 1)
    fpe = V * (1 + d / N) / (1 - d / N) if N > d else np.nan  # MATLAB: N ≤ d ise hesaplanamaz
    aic = N * logV + 2 * d + c
    aicc = aic + 2 * d * (d + 1) / (N - d - 1) if N - d - 1 > 0 else np.nan
    if len(y_list) == 1:
        fitp = fit_percent(y_list, [y_list[0] - e_list[0]])
        mse = float(np.sum(E * E) / N)
    else:  # MATLAB: çok deneyde FitPercent ve MSE deney başına (R2025b'de ölçüldü)
        fitp = np.array([fit_percent([y], [y - e]) for y, e in zip(y_list, e_list)])
        mse = np.array([float(np.sum(np.asarray(e) ** 2) / len(e)) for e in e_list])
    lossv = float(np.trace(S)) if (loss == "trace" and ny > 1) else V
    return Fit(fitp, lossv, mse, fpe, aic, aicc, logV + 2 * d / N, N * logV + c + d * np.log(N))


SHORT_DATA_MSG = ("Gürültü varyansı ve FPE hesaplanamadı: örnek sayısı, parametre ve başlangıç durumu sayısını aşmıyor "
                  "(MATLAB da hesaplamaz); daha uzun veri ya da daha düşük derece kullanın")


def noise_variance(EtE, N: int, d: int):
    """λ = EᵀE/(N − d); N − d ≤ 0 ise NaN (MATLAB gibi) ve uyarı iletisi döner: (λ, ileti | None)."""
    EtE = np.atleast_2d(np.asarray(EtE, float))
    if N - d <= 0:
        return np.full_like(EtE, np.nan), SHORT_DATA_MSG
    return EtE / (N - d), None


def attach_warnings(model, data, extra=()) -> None:
    """Uyarım ve kısa veri uyarılarını rapora ekler ve ControlWarning olarak yayar."""
    import warnings as _w

    from spt.control.errors import ControlWarning
    from spt.control.sysid.data import excitation_warnings

    msgs = list(excitation_warnings(data)) + [m for m in extra if m]
    rep = getattr(model, "report", None)
    if rep is not None:
        rep.warnings = list(dict.fromkeys(list(rep.warnings) + msgs))
    for m in msgs:
        _w.warn(m, ControlWarning, stacklevel=3)


def fit_percent(y_list, yhat_list) -> np.ndarray:
    """NRMSE uyum yüzdesi, çıkış başına: 100·(1 − ‖y − ŷ‖ / ‖y − ȳ‖) (deneyler birleştirilerek)."""
    Y = np.vstack(y_list)
    H = np.vstack(yhat_list)
    num = np.linalg.norm(Y - H, axis=0)
    den = np.linalg.norm(Y - Y.mean(axis=0), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100.0 * (1.0 - num / den)


# ============================================================================================ ayrık biçim
@dataclass
class DSS:
    """Ayrık yenilik biçimi (bkz. modül belgesi); `delay` tamsayı giriş gecikmeleri (nu,)."""
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    K: np.ndarray
    delay: np.ndarray

    @property
    def nx(self) -> int:
        return self.A.shape[0]


def lmfd_to_dss(den_terms, num_terms, noise_terms, ny: int, nu: int, delay=None) -> DSS:
    """Sol matris kesir biçimi y = Den⁻¹(N w + Nc e) → blok gözlemci kanonik yenilik biçimi.

    `den_terms[k]` (ny×ny, k=0..n, Den_0 = I), `num_terms[k]` (ny×nu), `noise_terms[k]` (ny×ny, Nc_0 = I).
    x_k(t+1) = −Den_k y(t) + N_k w(t) + Nc_k e(t) + x_{k+1}(t);  y(t) = x_1(t) + N_0 w(t) + e(t)."""
    n = max(len(den_terms), len(num_terms), len(noise_terms)) - 1

    def get(terms, k, shape):
        return terms[k] if k < len(terms) else np.zeros(shape)

    N0 = get(num_terms, 0, (ny, nu))
    nx = n * ny
    A = np.zeros((nx, nx))
    B = np.zeros((nx, nu))
    K = np.zeros((nx, ny))
    for k in range(1, n + 1):
        r = slice((k - 1) * ny, k * ny)
        Dk = get(den_terms, k, (ny, ny))
        A[r, 0:ny] = -Dk
        if k < n:
            A[r, k * ny:(k + 1) * ny] = np.eye(ny)
        B[r] = get(num_terms, k, (ny, nu)) - Dk @ N0
        K[r] = get(noise_terms, k, (ny, ny)) - Dk
    C = np.zeros((ny, nx))
    if nx:
        C[:, 0:ny] = np.eye(ny)
    d = np.zeros(nu, int) if delay is None else np.asarray(delay, int)
    return DSS(A, B, C, N0.copy(), K, d)


def _expm_blocks(A, B, h):
    """(e^{Ah}, ∫₀ʰ e^{Aσ}dσ B)."""
    n, m = B.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A * h
    M[:n, n:] = B * h
    E = linalg.expm(M)
    return E[:n, :n], E[:n, n:]


def c2d_exact(A, B, C, D, Ts: float, delays=None, hold=None) -> DSS:
    """Sürekli (A, B, C, D) + giriş gecikmeleri (s) → tam ayrık model (ZOH: kesirli gecikme dahil; FOH).

    Gecikme τ_j = d_j·Ts + f_j: tamsayı kısmı saf gecikme (DSS.delay), kesirli kısım ZOH'ta bir ek durumla tam
    modellenir (aralığın ilk f_j kısmında önceki örnek, kalanında yeni örnek etkindir)."""
    A, B, C, D = (np.atleast_2d(np.asarray(M, float)) for M in (A, B, C, D))
    n, nu = B.shape
    ny = C.shape[0]
    delays = np.zeros(nu) if delays is None else np.asarray(delays, float)
    hold = ("zoh",) * nu if hold is None else tuple(hold)
    dint = np.floor(delays / Ts + 1e-9).astype(int)
    frac = delays - dint * Ts
    frac[np.abs(frac) < 1e-9 * Ts] = 0.0
    Phi, Gam = _expm_blocks(A, B, Ts) if n else (np.zeros((0, 0)), np.zeros((0, nu)))
    Ad, Bd, Cd, Dd = Phi, np.zeros((n, nu)), C.copy(), np.zeros((ny, nu))
    extra_A, extra_B, extra_C = [], [], []  # ek durumlar (kesirli gecikme: önceki giriş örneği)
    for j in range(nu):
        bj = B[:, [j]]
        if hold[j] == "foh":
            if frac[j] > 0:
                raise ControlError("FOH örnekler arası davranışta kesirli (örnek katı olmayan) gecikme desteklenmiyor; "
                                   "yeniden örneklemeyi kapatın ya da gecikmeyi örnek aralığının katı yapın")
            M = np.zeros((n + 2, n + 2))
            M[:n, :n] = A * Ts
            M[:n, n:n + 1] = bj * Ts
            M[n, n + 1] = 1.0
            E = linalg.expm(M)
            G0 = E[:n, n:n + 1]
            Gs = E[:n, n + 1:n + 2] * Ts  # ∫₀^Ts e^{Aσ}(Ts−σ)dσ b
            Bd[:, [j]] = Phi @ Gs / Ts + G0 - Gs / Ts
            Dd[:, [j]] = D[:, [j]] + C @ Gs / Ts
            continue
        if frac[j] == 0.0:
            Bd[:, [j]] = Gam[:, [j]]
            Dd[:, [j]] = D[:, [j]]
            continue
        f = frac[j]
        _, G0 = _expm_blocks(A, bj, Ts - f)
        Pf, Gf = _expm_blocks(A, bj, f)
        Pr, _ = _expm_blocks(A, bj, Ts - f)
        G1 = Pr @ Gf
        Bd[:, [j]] = G0
        extra_A.append(G1)
        extra_B.append(j)
        extra_C.append(D[:, [j]])
    m = len(extra_A)
    if m:
        Ad2 = np.zeros((n + m, n + m))
        Ad2[:n, :n] = Ad
        Bd2 = np.zeros((n + m, nu))
        Bd2[:n] = Bd
        Cd2 = np.zeros((ny, n + m))
        Cd2[:, :n] = Cd
        for i, (G1, j, Dj) in enumerate(zip(extra_A, extra_B, extra_C)):
            Ad2[:n, n + i] = G1[:, 0]
            Bd2[n + i, j] = 1.0
            Cd2[:, n + i] = Dj[:, 0]  # y(k) = C x(k) + D u(k − d − 1)
            Dd[:, j] = 0.0
        Ad, Bd, Cd = Ad2, Bd2, Cd2
    return DSS(Ad, Bd, Cd, Dd, np.zeros((Ad.shape[0], ny)), dint)


def tf_to_ss(num, den):
    """Sürekli SISO transfer fonksiyonu (azalan kuvvetler) → gözlemci kanonik (A, B, C, D)."""
    num = np.atleast_1d(np.asarray(num, float))
    den = trim_leading(np.atleast_1d(np.asarray(den, float)))
    if den[0] == 0:
        raise ControlError("Payda sıfır olamaz")
    num, den = num / den[0], den / den[0]
    num = trim_leading(num)
    n = len(den) - 1
    if len(num) > len(den):
        raise ControlError("Uygun olmayan (pay derecesi > payda derecesi) transfer fonksiyonu")
    num = np.pad(num, (len(den) - len(num), 0))
    d0 = num[0]
    rest = num[1:] - d0 * den[1:]
    A = np.zeros((n, n))
    if n:
        A[:, 0] = -den[1:]
        A[:-1, 1:] = np.eye(n - 1)
    B = rest.reshape(n, 1)
    C = np.zeros((1, n))
    if n:
        C[0, 0] = 1.0
    return A, B, C, np.array([[d0]])


def trim_leading(p):
    p = np.atleast_1d(np.asarray(p, float))
    nz = np.flatnonzero(p)
    return p[nz[0]:].copy() if len(nz) else np.zeros(1)


def _lti():
    try:
        from spt.control import lti  # tembel: LTI çekirdeği ayrı modül
    except ImportError as exc:  # pragma: no cover
        raise ControlError("LTI modelleri (spt.control.lti) henüz kullanılamıyor") from exc
    return lti


# ============================================================================================ temel sınıf
class IDModel:
    """Tanımlanmış doğrusal modellerin ortak tabanı."""

    Ts: float
    ny: int
    nu: int

    def _init_meta(self, ny, nu, Ts, noise_variance, input_names, output_names, name):
        self.Ts = float(Ts)
        if self.Ts < 0:
            raise ControlError("Ts negatif olamaz (0 = sürekli zaman)")
        self.ny, self.nu = int(ny), int(nu)
        nv = np.eye(ny) if noise_variance is None else np.atleast_2d(np.asarray(noise_variance, float))
        if nv.shape != (ny, ny):
            raise ControlError("Gürültü varyansı ny×ny olmalı")
        self.noise_variance = nv
        self.input_names = tuple(input_names) if input_names is not None else tuple(f"u{i + 1}" for i in range(nu))
        self.output_names = tuple(output_names) if output_names is not None else tuple(
            f"y{i + 1}" for i in range(ny))
        self.name = name
        self.report: Report | None = None
        self.covariance: np.ndarray | None = None

    @property
    def NoiseVariance(self):
        return float(self.noise_variance[0, 0]) if self.ny == 1 else self.noise_variance

    @property
    def is_discrete(self) -> bool:
        return self.Ts > 0

    @property
    def has_noise_model(self) -> bool:
        return True

    # --- alt sınıfların sağladıkları
    def dss(self, Ts: float | None = None, hold=None) -> DSS:  # pragma: no cover - soyut
        raise NotImplementedError

    def freqresp(self, w) -> np.ndarray:  # pragma: no cover - soyut
        """Ölçüm modelinin frekans cevabı G(e^{jωTs}) ya da G(jω): karmaşık (ny, nu, nw); w rad/s."""
        raise NotImplementedError

    def getpvec(self) -> np.ndarray:  # pragma: no cover - soyut
        raise NotImplementedError

    def getcov(self):
        """Serbest parametrelerin kovaryansı (MATLAB `getcov`), yoksa None."""
        return self.covariance

    def std(self) -> np.ndarray | None:
        """Serbest parametrelerin standart sapmaları (getpvec sırasıyla)."""
        if self.covariance is None:
            return None
        return np.sqrt(np.maximum(np.diag(self.covariance), 0.0))

    @property
    def nparams(self) -> int:
        return len(self.getpvec())

    def _check_Ts(self, Ts):
        if Ts is None:
            if self.Ts <= 0:
                raise ControlError("Sürekli zamanlı model için örnekleme aralığı gerekli")
            return self.Ts
        Ts = float(Ts)
        if self.Ts > 0 and abs(Ts - self.Ts) > 1e-9 * self.Ts:
            raise ControlError(f"Modelin örnekleme aralığı ({self.Ts:g} s) veriyle ({Ts:g} s) aynı olmalı")
        return Ts

    def _dss_freqresp(self, w) -> np.ndarray:
        w = np.atleast_1d(np.asarray(w, float))
        s = self.dss(self.Ts) if self.Ts > 0 else None
        out = np.zeros((self.ny, self.nu, len(w)), complex)
        for k, wk in enumerate(w):
            z = np.exp(1j * wk * self.Ts)
            n = s.nx
            G = s.C @ np.linalg.solve(z * np.eye(n) - s.A, s.B) + s.D if n else s.D.astype(complex)
            out[:, :, k] = G * z ** (-s.delay.astype(float))[None, :]
        return out

    # --- LTI dönüşümü (tembel)
    def to_lti(self):  # pragma: no cover - soyut
        raise NotImplementedError

    def noise_lti(self, normalize: bool = False):
        """Gürültü modeli H (ny×ny; H(∞) = I). `normalize=True`: H·L (L = chol(NoiseVariance), MATLAB
        `noise2meas(sys, 'normalize')`), giriş birim varyanslı beyaz gürültü olur."""
        s = self.dss(self.Ts) if self.Ts > 0 else None
        if s is None:
            raise ControlError("Sürekli zamanlı modelin gürültü modeli ayrık örneklemeyle tanımlıdır; dss(Ts) kullanın")
        L = np.linalg.cholesky(self.noise_variance) if normalize else np.eye(self.ny)
        lti = _lti()
        return lti.ss(s.A, s.K @ L, s.C, L, self.Ts)


# ============================================================================================ idpoly
def _as_poly(p, monic=False, what=""):
    if p is None:
        return np.array([1.0])
    a = np.atleast_1d(np.asarray(p, float)).copy()
    if a.ndim != 1 or len(a) == 0:
        raise ControlError(f"{what} polinomu 1-B dizi olmalı")
    if monic and a[0] != 1.0:
        if a[0] == 0:
            raise ControlError(f"{what} polinomunun ilk katsayısı 1 olmalı")
        raise ControlError(f"{what} polinomu mönik olmalı (ilk katsayı 1)")
    return a


class IDPoly(IDModel):
    """Polinom model (MATLAB `idpoly`): A(q) y = Σ_j B_j(q)/F_j(q) u_j + C(q)/D(q) e.

    Tek çıkış: `A`, `C`, `D` 1-B; `B`, `F` giriş başına liste (tek girişte 1-B). Çok çıkış (yalnız ARX):
    `A` ny×ny, `B` ny×nu iç içe liste (A[i][i] mönik, A[i][j] (i≠j) ilk katsayısı 0). `nk` = B'nin baştaki
    sıfır sayısı (ny, nu)."""

    def __init__(self, A=None, B=None, C=None, D=None, F=None, Ts: float = 1.0, *, nk=None,
                 noise_variance=None, input_names=None, output_names=None, name: str = ""):
        multi = isinstance(A, (list, tuple)) and len(A) > 0 and isinstance(A[0], (list, tuple))
        if multi:
            ny = len(A)
            self._A = [[_as_poly(A[i][j]) for j in range(ny)] for i in range(ny)]
            for i in range(ny):
                if self._A[i][i][0] != 1.0:
                    raise ControlError("A[i][i] mönik olmalı")
                for j in range(ny):
                    if i != j and self._A[i][j][0] != 0.0:
                        raise ControlError("A[i][j] (i≠j) ilk katsayısı 0 olmalı")
            Bl = B if B is not None else [[] for _ in range(ny)]
            nu = len(Bl[0]) if ny and len(Bl) else 0
            self._B = [[_as_poly(Bl[i][j]) for j in range(nu)] for i in range(ny)]
            if C is not None or D is not None or F is not None:
                raise ControlError("Çok çıkışlı polinom model yalnız ARX yapısında (C = D = F = 1) desteklenir")
            self._C = [np.array([1.0]) for _ in range(ny)]
            self._D = [np.array([1.0]) for _ in range(ny)]
            self._F = [[np.array([1.0]) for _ in range(nu)] for _ in range(ny)]
        else:
            ny = 1
            self._A = [[_as_poly(A, True, "A")]]
            if B is None:
                Bl = []
            elif isinstance(B, (list, tuple)) and len(B) and np.ndim(B[0]) >= 1:
                Bl = [_as_poly(b) for b in B]
            else:
                Bl = [_as_poly(B)] if B is not None and np.size(B) else []
            nu = len(Bl)
            self._B = [Bl]
            self._C = [_as_poly(C, True, "C")]
            self._D = [_as_poly(D, True, "D")]
            if F is None:
                Fl = [np.array([1.0]) for _ in range(nu)]
            elif isinstance(F, (list, tuple)) and len(F) and np.ndim(F[0]) >= 1:
                Fl = [_as_poly(f, True, "F") for f in F]
            else:
                Fl = [_as_poly(F, True, "F")]
            if len(Fl) != nu:
                raise ControlError("F polinomu sayısı giriş sayısına eşit olmalı")
            self._F = [Fl]
        self._init_meta(ny, nu, Ts, noise_variance, input_names, output_names, name)
        if self.Ts <= 0:
            raise ControlError("Polinom model ayrık zamanlı olmalı (Ts > 0)")
        if nk is None:
            nk = np.array([[int(np.flatnonzero(b)[0]) if np.any(b) else len(b) for b in row] for row in self._B],
                          int).reshape(ny, nu)
        self.nk = np.asarray(nk, int).reshape(ny, nu)
        for i in range(ny):
            for j in range(nu):
                b = self._B[i][j]
                if len(b) < self.nk[i, j] or np.any(b[:self.nk[i, j]] != 0):
                    raise ControlError("B polinomunun ilk nk katsayısı 0 olmalı")

    # --- MATLAB biçimli erişim
    @property
    def A(self):
        return self._A[0][0] if self.ny == 1 else self._A

    @property
    def B(self):
        if self.ny == 1:
            return self._B[0][0] if self.nu == 1 else list(self._B[0])
        return self._B

    @property
    def C(self):
        return self._C[0] if self.ny == 1 else list(self._C)

    @property
    def D(self):
        return self._D[0] if self.ny == 1 else list(self._D)

    @property
    def F(self):
        if self.ny == 1:
            return self._F[0][0] if self.nu == 1 else list(self._F[0])
        return self._F

    def orders(self) -> dict:
        """na, nb, nc, nd, nf, nk (MATLAB adlarıyla)."""
        ny, nu = self.ny, self.nu
        na = np.array([[len(self._A[i][j]) - 1 for j in range(ny)] for i in range(ny)])
        nb = np.array([[len(self._B[i][j]) - self.nk[i, j] for j in range(nu)] for i in range(ny)]).reshape(ny, nu)
        nf = np.array([[len(self._F[i][j]) - 1 for j in range(nu)] for i in range(ny)]).reshape(ny, nu)
        nc = np.array([len(c) - 1 for c in self._C])
        nd = np.array([len(d) - 1 for d in self._D])
        return {"na": na, "nb": nb, "nc": nc, "nd": nd, "nf": nf, "nk": self.nk.copy()}

    @property
    def has_noise_model(self) -> bool:
        return any(len(trim(c)) > 1 for c in self._C) or any(len(trim(d)) > 1 for d in self._D) or True

    def getpvec(self) -> np.ndarray:
        """Serbest parametreler MATLAB sırasıyla: a, b (nk sonrası), c, d, f; çok çıkışta çıkış başına [a, b]."""
        ny, nu = self.ny, self.nu
        p = []
        if ny > 1:  # çok çıkışlı ARX: MATLAB sırası çıkış başına [A satırı, B satırı] (R2025b getcov ile ölçüldü)
            for i in range(ny):
                for j in range(ny):
                    p.extend(self._A[i][j][1:])
                for j in range(nu):
                    p.extend(self._B[i][j][self.nk[i, j]:])
            return np.array(p, float)
        for i in range(ny):
            for j in range(ny):
                p.extend(self._A[i][j][1:])
        for i in range(ny):
            for j in range(nu):
                p.extend(self._B[i][j][self.nk[i, j]:])
        for i in range(ny):
            p.extend(self._C[i][1:])
        for i in range(ny):
            p.extend(self._D[i][1:])
        for i in range(ny):
            for j in range(nu):
                p.extend(self._F[i][j][1:])
        return np.array(p, float)

    def setpvec(self, p) -> IDPoly:
        """Aynı yapıda, parametreleri `p` olan kopya."""
        p = np.asarray(p, float)
        k = 0
        new = self.copy()
        if self.ny > 1:
            for i in range(self.ny):
                for j in range(self.ny):
                    n = len(new._A[i][j]) - 1
                    new._A[i][j] = np.r_[new._A[i][j][0], p[k:k + n]]
                    k += n
                for j in range(self.nu):
                    nk = self.nk[i, j]
                    n = len(new._B[i][j]) - nk
                    new._B[i][j] = np.r_[np.zeros(nk), p[k:k + n]]
                    k += n
            if k != len(p):
                raise ControlError("Parametre vektörünün uzunluğu yapıya uymuyor")
            return new
        for i in range(self.ny):
            for j in range(self.ny):
                n = len(new._A[i][j]) - 1
                new._A[i][j] = np.r_[new._A[i][j][0], p[k:k + n]]
                k += n
        for i in range(self.ny):
            for j in range(self.nu):
                nk = self.nk[i, j]
                n = len(new._B[i][j]) - nk
                new._B[i][j] = np.r_[np.zeros(nk), p[k:k + n]]
                k += n
        for i in range(self.ny):
            n = len(new._C[i]) - 1
            new._C[i] = np.r_[1.0, p[k:k + n]]
            k += n
        for i in range(self.ny):
            n = len(new._D[i]) - 1
            new._D[i] = np.r_[1.0, p[k:k + n]]
            k += n
        for i in range(self.ny):
            for j in range(self.nu):
                n = len(new._F[i][j]) - 1
                new._F[i][j] = np.r_[1.0, p[k:k + n]]
                k += n
        if k != len(p):
            raise ControlError("Parametre vektörünün uzunluğu yapıya uymuyor")
        return new

    def copy(self) -> IDPoly:
        new = object.__new__(IDPoly)
        new.__dict__.update(self.__dict__)
        new._A = [[a.copy() for a in row] for row in self._A]
        new._B = [[b.copy() for b in row] for row in self._B]
        new._C = [c.copy() for c in self._C]
        new._D = [d.copy() for d in self._D]
        new._F = [[f.copy() for f in row] for row in self._F]
        new.nk = self.nk.copy()
        new.noise_variance = self.noise_variance.copy()
        return new

    def polydata_std(self) -> dict | None:
        """Polinom katsayılarının standart sapmaları (MATLAB `polydata` dA, dB, …); sabit katsayılar 0."""
        sd = self.std()
        if sd is None:
            return None
        tmp = self.setpvec(sd)
        z = lambda p: np.r_[0.0, p[1:]]  # noqa: E731
        out = {"A": [[z(a) for a in row] for row in tmp._A], "B": [[b.copy() for b in row] for row in tmp._B],
               "C": [z(c) for c in tmp._C], "D": [z(d) for d in tmp._D],
               "F": [[z(f) for f in row] for row in tmp._F]}
        if self.ny == 1:
            out = {"A": out["A"][0][0], "B": out["B"][0][0] if self.nu == 1 else out["B"][0], "C": out["C"][0],
                   "D": out["D"][0], "F": out["F"][0][0] if self.nu == 1 else out["F"][0]}
        return out

    # --- ayrık biçim
    def _input_delays(self) -> np.ndarray:
        """Saf gecikme olarak ayrılan giriş gecikmesi: max(min_i nk_ij − 1, 0)."""
        if self.nu == 0:
            return np.zeros(0, int)
        return np.maximum(self.nk.min(axis=0) - 1, 0)

    def dss(self, Ts: float | None = None, hold=None) -> DSS:
        self._check_Ts(Ts)
        ny, nu = self.ny, self.nu
        dly = self._input_delays()
        Bt = [[self._B[i][j][dly[j]:] for j in range(nu)] for i in range(ny)]
        if ny == 1:
            A, C, D = self._A[0][0], self._C[0], self._D[0]
            F = self._F[0]
            PF = pmul(*F) if nu else np.array([1.0])
            den = pmul(A, D, PF)
            nums = [pmul(Bt[0][j], D, *[F[k] for k in range(nu) if k != j]) for j in range(nu)]
            nc = pmul(C, PF)
            n = max([len(den), len(nc)] + [len(x) for x in nums]) - 1
            den_t = [np.array([[den[k] if k < len(den) else 0.0]]) for k in range(n + 1)]
            num_t = [np.array([[x[k] if k < len(x) else 0.0 for x in nums]]).reshape(1, nu) for k in range(n + 1)]
            nc_t = [np.array([[nc[k] if k < len(nc) else 0.0]]) for k in range(n + 1)]
            return lmfd_to_dss(den_t, num_t, nc_t, 1, nu, dly)
        # çok çıkışlı ARX
        n = max([len(self._A[i][j]) for i in range(ny) for j in range(ny)]
                + [len(Bt[i][j]) for i in range(ny) for j in range(nu)]) - 1
        den_t = [np.array([[self._A[i][j][k] if k < len(self._A[i][j]) else 0.0 for j in range(ny)]
                           for i in range(ny)]) for k in range(n + 1)]
        num_t = [np.array([[Bt[i][j][k] if k < len(Bt[i][j]) else 0.0 for j in range(nu)]
                           for i in range(ny)]).reshape(ny, nu) for k in range(n + 1)]
        nc_t = [np.eye(ny)] + [np.zeros((ny, ny))] * n
        return lmfd_to_dss(den_t, num_t, nc_t, ny, nu, dly)

    def freqresp(self, w) -> np.ndarray:
        w = np.atleast_1d(np.asarray(w, float))
        if self.ny > 1:
            return self._dss_freqresp(w)
        zi = np.exp(-1j * np.outer(np.arange(64), w * self.Ts))  # yalnız uzunluk kadar kullanılır

        def ev(p):
            p = np.asarray(p, float)
            if len(p) > zi.shape[0]:
                return np.polyval(p[::-1], np.exp(-1j * w * self.Ts))
            return p @ zi[:len(p)]

        A = ev(self._A[0][0])
        out = np.zeros((1, self.nu, len(w)), complex)
        for j in range(self.nu):
            out[0, j] = ev(self._B[0][j]) / (A * ev(self._F[0][j]))
        return out

    def noise_freqresp(self, w) -> np.ndarray:
        """Gürültü modeli H = C/(A·D) frekans cevabı (tek çıkış): (1, 1, nw)."""
        w = np.atleast_1d(np.asarray(w, float))
        z = np.exp(-1j * w * self.Ts)
        ev = lambda p: np.polyval(np.asarray(p, float)[::-1], z)  # noqa: E731
        if self.ny > 1:
            raise ControlError("Çok çıkışlı modelde gürültü cevabı için dss kullanın")
        return (ev(self._C[0]) / (ev(self._A[0][0]) * ev(self._D[0])))[None, None, :]

    def to_lti(self):
        """Ölçüm modeli G = B/(A·F) ayrık transfer fonksiyonu (z'nin azalan kuvvetleri; gecikmeler z⁻ᵏ olarak)."""
        lti = _lti()
        if self.ny > 1:
            s = self.dss()
            sys = lti.ss(s.A, s.B, s.C, s.D, self.Ts)
            if np.any(s.delay):
                sys = sys * 1  # gecikmeleri aşağıda giriş gecikmesi olarak ekle
                try:
                    sys.input_delay = np.asarray(s.delay, float)
                except AttributeError:  # pragma: no cover
                    pass
            return sys
        nums, dens = [], []
        for j in range(self.nu):
            num = trim(self._B[0][j])
            den = trim(pmul(self._A[0][0], self._F[0][j]))
            n = max(len(num), len(den))
            nums.append(np.pad(num, (0, n - len(num))))
            dens.append(np.pad(den, (0, n - len(den))))
        if self.nu == 1:
            return lti.tf(nums[0], dens[0], self.Ts)
        return lti.tf([nums], [dens], self.Ts)

    def __repr__(self) -> str:
        o = self.orders()
        return (f"<IDPoly ny={self.ny} nu={self.nu} Ts={self.Ts:g} na={o['na'].ravel().tolist()} "
                f"nb={o['nb'].ravel().tolist()} nc={o['nc'].tolist()} nd={o['nd'].tolist()} "
                f"nf={o['nf'].ravel().tolist()} nk={self.nk.ravel().tolist()}>")


def idpoly(A=None, B=None, C=None, D=None, F=None, Ts: float = 1.0, **kw) -> IDPoly:
    """MATLAB `idpoly(A, B, C, D, F, NoiseVariance, Ts)` (anahtar sözcüklerle)."""
    return IDPoly(A, B, C, D, F, Ts, **kw)


# ============================================================================================ idss
class IDSS(IDModel):
    """Durum uzayı modeli (MATLAB `idss`), yenilik biçiminde: x' = Ax + Bu + Ke, y = Cx + Du + e.

    Ts = 0: sürekli zaman (K, gürültünün ZOH ile örneklendiği kabulüyle ayrıklaştırılır — bizim kuralımız;
    ölçüm modeli G bundan etkilenmez). `input_delay`: ayrıkta örnek, sürekli zamanda saniye."""

    def __init__(self, A, B, C, D=None, K=None, Ts: float = 1.0, *, x0=None, input_delay=None,
                 noise_variance=None, input_names=None, output_names=None, name: str = ""):
        A = np.atleast_2d(np.asarray(A, float))
        C = np.atleast_2d(np.asarray(C, float))
        n = A.shape[0]
        if A.shape != (n, n):
            raise ControlError("A kare olmalı")
        ny = C.shape[0]
        B = np.asarray(B, float).reshape(n, -1) if np.size(B) else np.zeros((n, 0))
        nu = B.shape[1]
        if C.shape[1] != n:
            raise ControlError("C'nin sütun sayısı durum sayısına eşit olmalı")
        D = np.zeros((ny, nu)) if D is None else np.asarray(D, float).reshape(ny, nu)
        K = np.zeros((n, ny)) if K is None else np.asarray(K, float).reshape(n, ny)
        self.A, self.B, self.C, self.D, self.K = A, B, C, D, K
        self._init_meta(ny, nu, Ts, noise_variance, input_names, output_names, name)
        self.x0 = np.zeros(n) if x0 is None else np.asarray(x0, float).ravel()
        self.input_delay = np.zeros(nu) if input_delay is None else np.asarray(input_delay, float).ravel()
        if self.Ts > 0 and np.any(np.abs(self.input_delay - np.round(self.input_delay)) > 1e-12):
            raise ControlError("Ayrık modelde giriş gecikmesi tamsayı örnek olmalı")

    @property
    def nx(self) -> int:
        return self.A.shape[0]

    def getpvec(self) -> np.ndarray:
        """Serbest parametreler (MATLAB serbest ss sırası): A, B, C, D, K sütun sıralı."""
        return np.concatenate([M.ravel(order="F") for M in (self.A, self.B, self.C, self.D, self.K)])

    def copy(self) -> IDSS:
        new = object.__new__(IDSS)
        new.__dict__.update(self.__dict__)
        for k in ("A", "B", "C", "D", "K", "x0", "input_delay", "noise_variance"):
            setattr(new, k, getattr(self, k).copy())
        return new

    def dss(self, Ts: float | None = None, hold=None) -> DSS:
        Ts = self._check_Ts(Ts)
        if self.Ts > 0:
            return DSS(self.A, self.B, self.C, self.D, self.K, np.round(self.input_delay).astype(int))
        s = c2d_exact(self.A, self.B, self.C, self.D, Ts, self.input_delay, hold)
        Kd = np.zeros((s.nx, self.ny))
        if self.nx:
            _, G = _expm_blocks(self.A, self.K, Ts)
            Kd[:self.nx] = G
        s.K = Kd
        return s

    def freqresp(self, w) -> np.ndarray:
        w = np.atleast_1d(np.asarray(w, float))
        if self.Ts > 0:
            return self._dss_freqresp(w)
        out = np.zeros((self.ny, self.nu, len(w)), complex)
        I = np.eye(self.nx)
        for k, wk in enumerate(w):
            G = self.C @ np.linalg.solve(1j * wk * I - self.A, self.B) + self.D if self.nx else self.D + 0j
            out[:, :, k] = G * np.exp(-1j * wk * self.input_delay)[None, :]
        return out

    def to_lti(self):
        lti = _lti()
        sys = lti.ss(self.A, self.B, self.C, self.D, self.Ts)
        if np.any(self.input_delay):
            try:
                sys.input_delay = self.input_delay.copy()
            except AttributeError:  # pragma: no cover
                raise ControlError("LTI modeli giriş gecikmesini desteklemiyor") from None
        return sys

    def __repr__(self) -> str:
        kind = "sürekli" if self.Ts == 0 else f"Ts={self.Ts:g}"
        return f"<IDSS nx={self.nx} ny={self.ny} nu={self.nu} {kind}>"


def idss(A, B, C, D=None, K=None, Ts: float = 1.0, **kw) -> IDSS:
    return IDSS(A, B, C, D, K, Ts, **kw)


# ============================================================================================ idtf
class IDTF(IDModel):
    """Transfer fonksiyonu modeli (MATLAB `idtf`), tek çıkış: y = Σ_j num_j/den_j u_j (+ e).

    Sürekli zamanda (Ts = 0) azalan s kuvvetleri ve `io_delay` saniye; ayrıkta q⁻¹ artan kuvvetleri (MATLAB
    idtf 'z^-1' gösterimi) ve `io_delay` örnek. Gürültü modeli yoktur (çıkış hatası: H = 1)."""

    def __init__(self, num, den, Ts: float = 0.0, *, io_delay=None, noise_variance=None, input_names=None,
                 output_names=None, name: str = ""):
        if isinstance(num, (list, tuple)) and len(num) and np.ndim(num[0]) >= 1:
            nums = [np.atleast_1d(np.asarray(n, float)) for n in num]
            dens = [np.atleast_1d(np.asarray(d, float)) for d in den]
        else:
            nums = [np.atleast_1d(np.asarray(num, float))]
            dens = [np.atleast_1d(np.asarray(den, float))]
        if len(nums) != len(dens):
            raise ControlError("Pay ve payda sayıları eşit olmalı")
        self.num, self.den = nums, dens
        self._init_meta(1, len(nums), Ts, noise_variance, input_names, output_names, name)
        self.io_delay = np.zeros(self.nu) if io_delay is None else np.asarray(io_delay, float).ravel() * np.ones(
            self.nu)
        for d in self.den:
            if d[0] == 0:
                raise ControlError("Paydanın ilk katsayısı sıfır olamaz")

    def getpvec(self) -> np.ndarray:
        """Serbest parametreler: her giriş için pay katsayıları, sonra payda (baş katsayı hariç) — MATLAB sırası."""
        p = []
        for n in self.num:
            p.extend(n)
        for d in self.den:
            p.extend(d[1:])
        return np.array(p, float)

    def copy(self) -> IDTF:
        new = object.__new__(IDTF)
        new.__dict__.update(self.__dict__)
        new.num = [n.copy() for n in self.num]
        new.den = [d.copy() for d in self.den]
        new.io_delay = self.io_delay.copy()
        new.noise_variance = self.noise_variance.copy()
        return new

    def _ct_ss(self):
        """Sürekli zamanlı gerçekleme: girişler paralel (blok köşegen)."""
        As, Bs, Cs, Ds = [], [], [], []
        for n, d in zip(self.num, self.den):
            A, B, C, D = tf_to_ss(n, d)
            As.append(A)
            Bs.append(B)
            Cs.append(C)
            Ds.append(D)
        A = linalg.block_diag(*As) if As else np.zeros((0, 0))
        nx = A.shape[0]
        B = np.zeros((nx, self.nu))
        C = np.zeros((1, nx))
        D = np.zeros((1, self.nu))
        k = 0
        for j in range(self.nu):
            m = As[j].shape[0]
            B[k:k + m, j] = Bs[j][:, 0]
            C[0, k:k + m] = Cs[j][0]
            D[0, j] = Ds[j][0, 0]
            k += m
        return A, B, C, D

    def dss(self, Ts: float | None = None, hold=None) -> DSS:
        Ts = self._check_Ts(Ts)
        if self.Ts > 0:
            ny, nu = 1, self.nu
            dl = np.round(self.io_delay).astype(int)
            PF = pmul(*self.den)
            nums = [pmul(self.num[j], *[self.den[k] for k in range(nu) if k != j]) for j in range(nu)]
            den = PF / PF[0]
            nums = [x / PF[0] for x in nums]
            n = max([len(den)] + [len(x) for x in nums]) - 1
            den_t = [np.array([[den[k] if k < len(den) else 0.0]]) for k in range(n + 1)]
            num_t = [np.array([[x[k] if k < len(x) else 0.0 for x in nums]]).reshape(ny, nu) for k in range(n + 1)]
            nc_t = [np.array([[den[k] if k < len(den) else 0.0]]) for k in range(n + 1)]  # H = 1 → K = 0
            return lmfd_to_dss(den_t, num_t, nc_t, 1, nu, dl)
        A, B, C, D = self._ct_ss()
        return c2d_exact(A, B, C, D, Ts, self.io_delay, hold)

    def freqresp(self, w) -> np.ndarray:
        w = np.atleast_1d(np.asarray(w, float))
        out = np.zeros((1, self.nu, len(w)), complex)
        for j, (n, d) in enumerate(zip(self.num, self.den)):
            if self.Ts > 0:
                z = np.exp(-1j * w * self.Ts)
                out[0, j] = (np.polyval(n[::-1], z) / np.polyval(d[::-1], z)) * z ** self.io_delay[j]
            else:
                s = 1j * w
                out[0, j] = np.polyval(n, s) / np.polyval(d, s) * np.exp(-s * self.io_delay[j])
        return out

    def to_lti(self):
        lti = _lti()
        if self.Ts > 0:
            nums, dens = [], []
            for j in range(self.nu):
                num = np.r_[np.zeros(int(round(self.io_delay[j]))), self.num[j]]
                den = self.den[j]
                n = max(len(num), len(den))
                nums.append(np.pad(num, (0, n - len(num))))
                dens.append(np.pad(den, (0, n - len(den))))
            return lti.tf(nums[0], dens[0], self.Ts) if self.nu == 1 else lti.tf([nums], [dens], self.Ts)
        sys = lti.tf(self.num[0], self.den[0], 0.0) if self.nu == 1 else lti.tf([self.num], [self.den], 0.0)
        if np.any(self.io_delay):
            try:
                sys.io_delay = self.io_delay.reshape(1, self.nu).copy()
            except AttributeError:  # pragma: no cover
                raise ControlError("LTI modeli gecikmeyi desteklemiyor") from None
        return sys

    def __repr__(self) -> str:
        np_ = [len(d) - 1 for d in self.den]
        nz = [len(trim_leading(n)) - 1 for n in self.num]
        kind = "sürekli" if self.Ts == 0 else f"Ts={self.Ts:g}"
        return f"<IDTF nu={self.nu} np={np_} nz={nz} io_delay={self.io_delay.tolist()} {kind}>"


def idtf(num, den, Ts: float = 0.0, **kw) -> IDTF:
    return IDTF(num, den, Ts, **kw)


# ============================================================================================ idproc
_PROC_LETTERS = "PDZUI"


def parse_proc_type(t: str) -> dict:
    """'P2DZU' → {'np': 2, 'delay': True, 'zero': True, 'under': True, 'integ': False} (MATLAB idproc türü)."""
    s = t.upper().strip()
    if not s.startswith("P") or len(s) < 2 or not s[1].isdigit():
        raise ControlError(f"Geçersiz süreç modeli türü: {t} (ör. P1D, P2U, P1DI)")
    n = int(s[1])
    if n not in (0, 1, 2, 3):
        raise ControlError("Kutup sayısı 0–3 olmalı")
    rest = s[2:]
    if any(c not in "DZUI" for c in rest) or len(set(rest)) != len(rest):
        raise ControlError(f"Geçersiz süreç modeli türü: {t}")
    under = "U" in rest
    if under and n < 2:
        raise ControlError("Sönümsüz/salınımlı (U) yapı en az 2 kutup ister")
    return {"np": n, "delay": "D" in rest, "zero": "Z" in rest, "under": under, "integ": "I" in rest}


_PROC_PARAMS = ("Kp", "Tp1", "Tp2", "Tp3", "Tw", "Zeta", "Td", "Tz")


class IDProc(IDModel):
    """Süreç modeli (MATLAB `idproc`), sürekli zaman, tek çıkış:

        G(s) = Kp · (1 + Tz s) / [(1 + Tp1 s)(1 + Tp2 s)(1 + Tp3 s)] · e^{−Td s}   (I: ayrıca 1/s)
        U yapısında iki kutup: 1 / (1 + 2 Zeta Tw s + (Tw s)²).

    Parametreler `params[j]` sözlüğünde (giriş başına; Kp, Tp1, Tp2, Tp3, Tw, Zeta, Td, Tz)."""

    def __init__(self, type_="P1D", params=None, *, noise_variance=None, input_names=None, output_names=None,
                 name: str = ""):
        types = [type_] if isinstance(type_, str) else list(type_)
        self.types = [t.upper() for t in types]
        self.structs = [parse_proc_type(t) for t in self.types]
        nu = len(types)
        if params is None:
            params = [{} for _ in range(nu)]
        elif isinstance(params, dict):
            params = [params]
        if len(params) != nu:
            raise ControlError("Parametre sözlüğü sayısı giriş sayısına eşit olmalı")
        self.params = []
        for st, p in zip(self.structs, params):
            q = {k: float(p.get(k, 0.0)) for k in self.active_names(st)}
            self.params.append(q)
        self._init_meta(1, nu, 0.0, noise_variance, input_names, output_names, name)

    @staticmethod
    def active_names(st: dict) -> list[str]:
        names = ["Kp"]
        if st["under"]:
            names += ["Tw", "Zeta"]
            if st["np"] == 3:
                names += ["Tp3"]
        else:
            names += [f"Tp{i + 1}" for i in range(st["np"])]
        if st["delay"]:
            names.append("Td")
        if st["zero"]:
            names.append("Tz")
        return names

    @property
    def type(self):
        return self.types[0] if self.nu == 1 else self.types

    def __getattr__(self, item):
        if item in _PROC_PARAMS and "params" in self.__dict__:
            vals = [p.get(item, np.nan) for p in self.__dict__["params"]]
            return vals[0] if len(vals) == 1 else np.array(vals)
        raise AttributeError(item)

    def tf_coeffs(self, j: int = 0):
        """(num, den, Td) — azalan s kuvvetleri."""
        st, p = self.structs[j], self.params[j]
        num = np.array([p["Kp"]])
        if st["zero"]:
            num = np.array([p["Kp"] * p["Tz"], p["Kp"]])
        den = np.array([1.0])
        if st["under"]:
            Tw, z = p["Tw"], p["Zeta"]
            den = np.polymul(den, [Tw * Tw, 2 * z * Tw, 1.0])
            if st["np"] == 3:
                den = np.polymul(den, [p["Tp3"], 1.0])
        else:
            for i in range(st["np"]):
                den = np.polymul(den, [p[f"Tp{i + 1}"], 1.0])
        if st["integ"]:
            den = np.polymul(den, [1.0, 0.0])
        return num, den, (p["Td"] if st["delay"] else 0.0)

    def to_idtf(self) -> IDTF:
        nums, dens, dl = [], [], []
        for j in range(self.nu):
            n, d, td = self.tf_coeffs(j)
            dd = trim_leading(d)
            nums.append(n / dd[0])
            dens.append(dd / dd[0])
            dl.append(td)
        return IDTF(nums, dens, 0.0, io_delay=dl, noise_variance=self.noise_variance,
                    input_names=self.input_names, output_names=self.output_names)

    def getpvec(self) -> np.ndarray:
        return np.array([p[k] for st, p in zip(self.structs, self.params) for k in self.active_names(st)], float)

    def copy(self) -> IDProc:
        new = object.__new__(IDProc)
        new.__dict__.update(self.__dict__)
        new.params = [dict(p) for p in self.params]
        new.noise_variance = self.noise_variance.copy()
        return new

    def dss(self, Ts: float | None = None, hold=None) -> DSS:
        return self.to_idtf().dss(Ts, hold)

    def freqresp(self, w) -> np.ndarray:
        return self.to_idtf().freqresp(w)

    def to_lti(self):
        return self.to_idtf().to_lti()

    def __repr__(self) -> str:
        ps = "; ".join(", ".join(f"{k}={v:.6g}" for k, v in p.items()) for p in self.params)
        return f"<IDProc {self.types} {ps}>"


def idproc(type_="P1D", params=None, **kw) -> IDProc:
    return IDProc(type_, params, **kw)


# ============================================================================================ idfrd
class IDFRD:
    """Frekans cevabı verisi (MATLAB `idfrd`): `frequency` (rad/s), `response` (ny, nu, nw) karmaşık,
    `spectrum` (ny, ny, nw) gerçel (gürültü spektrumu; MATLAB SpectrumData ölçeği: Ts·Φ_v), `Ts`."""

    def __init__(self, response, frequency, Ts: float, *, spectrum=None, method: str = "",
                 input_names=None, output_names=None):
        self.response = np.asarray(response, complex)
        self.frequency = np.asarray(frequency, float)
        self.Ts = float(Ts)
        self.spectrum = None if spectrum is None else np.asarray(spectrum, float)
        self.method = method
        ny, nu = self.response.shape[:2]
        self.input_names = tuple(input_names) if input_names else tuple(f"u{i + 1}" for i in range(nu))
        self.output_names = tuple(output_names) if output_names else tuple(f"y{i + 1}" for i in range(ny))

    @property
    def ResponseData(self):
        return self.response

    @property
    def magnitude(self):
        return np.abs(self.response)

    @property
    def phase_deg(self):
        return np.degrees(np.unwrap(np.angle(self.response), axis=-1))

    def __repr__(self) -> str:
        return f"<IDFRD {self.method} {self.response.shape[0]}×{self.response.shape[1]}, {len(self.frequency)} frekans>"
