"""Doğrusal MPC — MATLAB Model Predictive Control Toolbox ``mpc`` nesnesi anlamlarıyla.

Ayrıntılı tasarım notları (QP formülasyonu, ölçekleme, ECR, kestirici):
``docs/SPT-TSR-001_MPC_Tasarim_Notlari.docx``.

Özet (MATLAB R2025b kaynağından ölçülen / doğrulanan tanımlar):

- Bitki: ayrık ya da sürekli (denetleyici Ts ile ZOH) durum uzayı. Girişler MV / MD / UD, çıkışlar MO / UO
  (:func:`setmpcsignals`; indisler 0 tabanlı). MV'den çıkışa doğrudan geçiş yasaktır (MATLAB gibi).
- İç model **boyutsuzdur**: u_s = u / ScaleFactor, y_s = y / ScaleFactor; durumlar ölçeklenmez. Nominal değerler
  (Model.Nominal: X, U, Y, DX) çıkarılır; DX bitkiye ek bir "ofset" ölçülen bozucusu (değeri 1) olarak girer.
- Varsayılan giriş bozucu modeli (UD kanalları): gözlenebilirliği bozmuyorsa her kanala Ts kazançlı tümlev
  (x⁺ = x + Ts·w, d = x), bozuyorsa birim statik kazanç. Varsayılan çıkış bozucu modeli: ölçülen çıkışlara, çıkış
  ağırlığı büyükten küçüğe sırayla, z = 1'de PBH gözlenebilirlik testi geçerse Ts kazançlı tümlev. Ölçüm gürültüsü:
  ölçülen her çıkışta birim beyaz gürültü (boyutsuz).
- Kestirici: artırılmış model üzerinde kararlı hâl Kalman süzgeci; bilinmeyen girişler MV, MD, bozucu ve gürültü
  beyaz gürültüleri (birim varyans; DX ofseti hariç): Q = GGᵀ, R = HHᵀ, N = GHᵀ. M = PCᵀ(CPCᵀ + R)⁻¹,
  L = (APCᵀ + N)(CPCᵀ + R)⁻¹ (MATLAB ``kalman``, 'current').
- QP: karar değişkenleri bloklanmış Δu hamleleri (+ kısıt varsa tek gevşek değişken ε);
  J = Σ‖W_y(y − r)‖² + Σ‖W_u(u − u_hedef)‖² + Σ‖W_Δu Δu‖² + ρ_ε ε², QP ½zᵀHz + fᵀz (J/2), Ac z ≥ b(x, u₋₁, v).
  MV / MV oranı kısıtları varsayılan sert (ECR 0), çıkış kısıtları yumuşak (ECR 1). ρ_ε = Weights.ECR
  (varsayılan 1e5·max(tüm ağırlıklar), kurulum anında).
- QP çözümsüzse ya da yineleme sınırı aşılırsa MV önceki değerinde tutulur (MATLAB gibi).

Bilinçli farklar: gecikmeli bitkiler, d2d yeniden örnekleme, köşegen dışı (cell) ağırlıklar, karma (custom)
kısıtlar, ayrık (tamsayı) MV'ler, uyarlamalı / LTV kipleri ve özel kestirici desteklenmez (açık hata). Yineleme
sayıları KWIK'inkinden farklı olabilir (çözüm aynıdır; bkz. qp.py).
"""
from __future__ import annotations

import copy
import pickle
import warnings
from collections import namedtuple
from dataclasses import dataclass
from math import sqrt

import numpy as np
from scipy.linalg import LinAlgError, cholesky, expm, solve_discrete_are, solve_triangular

from spt.control.errors import ControlError, ControlWarning
from spt.control.mpc.qp import QPWorkspace, qp_solve_ge

__all__ = ["MPC", "mpc", "MPCState", "mpcstate", "mpcmove", "sim", "mpcsimopt", "getEstimator", "cloffset", "ss",
           "setmpcsignals", "setoutdist", "MPCPlant", "MPCData", "SimResult", "MPCLinearController", "Struct"]

_EPS = float(np.finfo(float).eps)
_DEFAULT_P = 10
_DEFAULT_M = 2


# ---------------------------------------------------------------------------------------------------------------
# MATLAB struct benzeri kaplar
# ---------------------------------------------------------------------------------------------------------------
class Struct(dict):
    """MATLAB struct benzeri sözlük: ``s.Alan`` ve ``s['Alan']`` ikisi de çalışır; alan adları büyük / küçük harf
    duyarsızdır (MATLAB ``CaseInsensitiveProperties`` gibi). Tanımlı alan listesi varsa bilinmeyen ad hatadır."""
    _fields: tuple = ()
    _aliases: dict = {}

    def __init__(self, *args, **kw):
        super().__init__()
        for k, v in dict(*args, **kw).items():
            self[k] = v

    def _key(self, k):
        if not isinstance(k, str) or not self._fields:
            return k
        kl = k.lower()
        for f in self._fields:
            if f.lower() == kl:
                return f
        if kl in self._aliases:
            return self._aliases[kl]
        raise ControlError(f"Geçersiz alan adı: '{k}' (geçerli: {', '.join(self._fields)})")

    def __getattr__(self, k):
        if k.startswith("_"):
            raise AttributeError(k)
        try:
            return dict.__getitem__(self, self._key(k))
        except KeyError:
            raise AttributeError(k) from None

    def __setattr__(self, k, v):
        if k.startswith("_"):
            object.__setattr__(self, k, v)
        else:
            dict.__setitem__(self, self._key(k), v)

    def __getitem__(self, k):
        return dict.__getitem__(self, self._key(k))

    def __setitem__(self, k, v):
        dict.__setitem__(self, self._key(k), v)

    def __contains__(self, k):
        try:
            return dict.__contains__(self, self._key(k))
        except ControlError:
            return False

    def get(self, k, default=None):
        try:
            return dict.get(self, self._key(k), default)
        except ControlError:
            return default

    def __reduce__(self):
        return (self.__class__, (dict(self),))

    def __deepcopy__(self, memo):
        return self.__class__({k: copy.deepcopy(v, memo) for k, v in dict.items(self)})


class _MVSpec(Struct):
    _fields = ("Min", "Max", "MinECR", "MaxECR", "RateMin", "RateMax", "RateMinECR", "RateMaxECR", "Target",
               "Name", "Units", "ScaleFactor", "Type")


class _OVSpec(Struct):
    _fields = ("Min", "Max", "MinECR", "MaxECR", "Name", "Units", "ScaleFactor")


class _DVSpec(Struct):
    _fields = ("Name", "Units", "ScaleFactor")


class _Weights(Struct):
    _fields = ("ManipulatedVariables", "ManipulatedVariablesRate", "OutputVariables", "ECR")
    _aliases = {"mv": "ManipulatedVariables", "manipulated": "ManipulatedVariables", "input": "ManipulatedVariables",
                "ov": "OutputVariables", "controlled": "OutputVariables", "output": "OutputVariables",
                "mvrate": "ManipulatedVariablesRate", "manipulatedrate": "ManipulatedVariablesRate",
                "inputrate": "ManipulatedVariablesRate"}


class _SolverOptions:
    """MATLAB R2025b yazımı ``Optimizer.SolverOptions.{MaxIterations, ConstraintTolerance, UseWarmStart}`` —
    üst Optimizer yapısının aynı adlı alanlarına bağlı görünüm (iki yazım da aynı değeri okur / yazar)."""
    _FIELDS = ("MaxIterations", "ConstraintTolerance", "UseWarmStart")

    def __init__(self, parent):
        object.__setattr__(self, "_parent", parent)

    def _k(self, k):
        for f in self._FIELDS:
            if isinstance(k, str) and f.lower() == k.lower():
                return f
        raise ControlError(f"Geçersiz alan adı: 'SolverOptions.{k}' (geçerli: {', '.join(self._FIELDS)})")

    def __getattr__(self, k):
        if k.startswith("_"):
            raise AttributeError(k)
        return self._parent[self._k(k)]

    def __setattr__(self, k, v):
        self._parent[self._k(k)] = v

    def __getitem__(self, k):
        return self._parent[self._k(k)]

    def __setitem__(self, k, v):
        self._parent[self._k(k)] = v

    def items(self):
        return [(f, self._parent[f]) for f in self._FIELDS]

    def keys(self):
        return list(self._FIELDS)

    def __repr__(self):
        return repr(dict(self.items()))


class _Optimizer(Struct):
    _fields = ("MaxIterations", "ConstraintTolerance", "UseWarmStart", "UseSuboptimalSolution", "MinOutputECR")

    @staticmethod
    def _is_so(k):
        return isinstance(k, str) and k.lower() == "solveroptions"

    def _set_so(self, v):
        so = _SolverOptions(self)
        for k, x in (v.items() if hasattr(v, "items") else dict(v).items()):
            so[k] = x

    def __getattr__(self, k):
        if self._is_so(k):
            return _SolverOptions(self)
        return super().__getattr__(k)

    def __setattr__(self, k, v):
        if self._is_so(k):
            self._set_so(v)
        else:
            super().__setattr__(k, v)

    def __getitem__(self, k):
        return _SolverOptions(self) if self._is_so(k) else super().__getitem__(k)

    def __setitem__(self, k, v):
        if self._is_so(k):
            self._set_so(v)
        else:
            super().__setitem__(k, v)


class _Nominal(Struct):
    _fields = ("X", "U", "Y", "DX")


class _Model(Struct):
    _fields = ("Plant", "Disturbance", "Noise", "Nominal")


# ---------------------------------------------------------------------------------------------------------------
# Model alımı (ördek tipleme)
# ---------------------------------------------------------------------------------------------------------------
class MPCPlant:
    """Sinyal türleri işaretlenmiş durum uzayı bitki (MATLAB ``setmpcsignals`` sonucu). İndisler 0 tabanlı."""

    def __init__(self, A, B, C, D, Ts=0.0, *, input_names=None, output_names=None, mpc_signals=None):
        self.A, self.B, self.C, self.D, self.Ts = A, B, C, D, float(Ts)
        self.input_names = tuple(input_names) if input_names else ()
        self.output_names = tuple(output_names) if output_names else ()
        self.mpc_signals = dict(mpc_signals or {})

    @property
    def ninputs(self):
        return np.atleast_2d(np.asarray(self.D, float)).shape[1]

    @property
    def noutputs(self):
        return np.atleast_2d(np.asarray(self.D, float)).shape[0]


def _ss_data(sys, what="Bitki"):
    """(A, B, C, D, Ts) — A, B, C, D, Ts öznitelikli nesne, (A, B, C, D[, Ts]) dörtlüsü / beşlisi ya da statik kazanç
    (sayı / matris; Ts = None: denetleyicininkini alır). Diğer LTI türleri spt.control.lti.ss ile çevrilir."""
    if isinstance(sys, tuple):
        if len(sys) not in (4, 5):
            raise ControlError(f"{what} modeli (A, B, C, D) ya da (A, B, C, D, Ts) biçiminde olmalı.")
        A, B, C, D = sys[:4]
        Ts = float(sys[4]) if len(sys) == 5 else 0.0
    elif isinstance(sys, (int, float, list, np.ndarray, np.number)):
        Dm = np.atleast_2d(np.asarray(sys, dtype=float))
        return np.zeros((0, 0)), np.zeros((0, Dm.shape[1])), np.zeros((Dm.shape[0], 0)), Dm, None
    else:
        if not all(hasattr(sys, a) for a in ("A", "B", "C", "D")):
            try:
                from spt.control import lti  # tembel: LTI çekirdeği ayrı modülde
                sys = lti.ss(sys)
            except Exception as e:  # noqa: BLE001
                raise ControlError(f"{what} modeli durum uzayına çevrilemedi: {e}") from None
        for attr in ("input_delay", "output_delay", "io_delay", "internal_delay"):
            val = getattr(sys, attr, None)
            if val is not None and np.size(val) and np.any(np.asarray(val, dtype=float) != 0):
                raise ControlError(f"{what} modelinde gecikme var; MPC şu an gecikmeli modelleri desteklemiyor "
                                   "(gecikmeyi ayrık durum olarak modele katın).")
        A, B, C, D = sys.A, sys.B, sys.C, sys.D
        Ts = float(getattr(sys, "Ts", 0.0) or 0.0)
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    D = np.asarray(D, dtype=float)
    nx = A.shape[0] if A.ndim == 2 else (1 if A.size == 1 else 0)
    A = A.reshape(nx, nx)
    if B.ndim == 2:
        nu = B.shape[1]
    elif D.ndim == 2:
        nu = D.shape[1]
    else:
        nu = B.size // max(nx, 1) if nx else max(D.size, 1)
    if C.ndim == 2:
        ny = C.shape[0]
    elif D.ndim == 2:
        ny = D.shape[0]
    else:
        ny = C.size // max(nx, 1) if nx else max(D.size, 1)
    B = B.reshape(nx, nu)
    C = C.reshape(ny, nx)
    if D.size == 1 and ny * nu != 1:
        D = np.full((ny, nu), float(D.ravel()[0]))
    D = D.reshape(ny, nu)
    if Ts is not None and Ts < 0:
        raise ControlError("Örnekleme zamanı Ts negatif olamaz (belirtilmemiş Ts desteklenmez).")
    return A, B, C, D, Ts


def setmpcsignals(plant, MV=None, MD=None, UD=None, MO=None, UO=None):
    """MATLAB ``setmpcsignals``: giriş / çıkış sinyal türlerini işaretler (0 tabanlı indisler).

    Belirtilmeyen girişler MV, belirtilmeyen çıkışlar MO sayılır (MATLAB gibi)."""
    A, B, C, D, Ts = _ss_data(plant)
    sig = {k: [int(i) for i in np.atleast_1d(v)] for k, v in
           (("MV", MV), ("MD", MD), ("UD", UD), ("MO", MO), ("UO", UO)) if v is not None}
    old = getattr(plant, "mpc_signals", None) or {}
    merged = {**old, **sig}
    return MPCPlant(A, B, C, D, 0.0 if Ts is None else Ts,
                    input_names=getattr(plant, "input_names", None), output_names=getattr(plant, "output_names", None),
                    mpc_signals=merged)


def _c2d_zoh(A, B, Ts):
    n, m = B.shape
    Mx = np.zeros((n + m, n + m))
    Mx[:n, :n] = A * Ts
    Mx[:n, n:] = B * Ts
    E = expm(Mx)
    return E[:n, :n], E[:n, n:]


def _rank(M):
    """MATLAB ``rank``: tekil değer > max(size)·eps(max σ)."""
    if M.size == 0:
        return 0
    s = np.linalg.svd(M, compute_uv=False)
    tol = max(M.shape) * np.spacing(s.max()) if s.size else 0.0
    return int(np.sum(s > tol))


def _blkdiag(*mats):
    rows = sum(m.shape[0] for m in mats)
    cols = sum(m.shape[1] for m in mats)
    out = np.zeros((rows, cols))
    r = c = 0
    for m in mats:
        out[r:r + m.shape[0], c:c + m.shape[1]] = m
        r += m.shape[0]
        c += m.shape[1]
    return out


def _dist_model(sys, nout, Ts, scale, kind):
    """Özel bozucu / gürültü modeli: ayrık, boyutsuz (C, D satırları 1/scale) (A, B, C, D)."""
    A, B, C, D, Tsm = _ss_data(sys, kind)
    if C.shape[0] != nout:
        raise ControlError(f"{kind} modelinin çıkış sayısı {nout} olmalı.")
    if Tsm is not None:
        if Tsm == 0.0:
            if A.shape[0]:
                A, B = _c2d_zoh(A, B, Ts)
        elif abs(Tsm - Ts) > 1e-10:
            raise ControlError(f"{kind} modelinin örnekleme zamanı denetleyicininkiyle aynı olmalı.")
    sc = 1.0 / np.asarray(scale, dtype=float)
    return A, B, C * sc[:, None], D * sc[:, None]


def _dcgain_d(A, B, C, D):
    """Ayrık sistemin DC kazancı; z = 1'de yapısal olarak kontrol / gözlenemeyen durumlar atılır (MATLAB sminreal)."""
    n = A.shape[0]
    if n:
        keep = _sminreal_mask(A, B, C)
        A, B, C = A[np.ix_(keep, keep)], B[keep], C[:, keep]
        n = A.shape[0]
    if n == 0:
        return D.copy()
    Mi = np.eye(n) - A
    if np.linalg.cond(Mi) > 1.0 / _EPS:
        return np.full(D.shape, np.inf)
    return C @ np.linalg.solve(Mi, B) + D


def _sminreal_mask(A, B, C):
    n = A.shape[0]
    adj = A != 0
    reach = np.any(B != 0, axis=1)
    while True:
        new = reach | np.any(adj[:, reach], axis=1)
        if np.array_equal(new, reach):
            break
        reach = new
    obs = np.any(C != 0, axis=0)
    while True:
        new = obs | np.any(adj[obs, :], axis=0)
        if np.array_equal(new, obs):
            break
        obs = new
    return np.flatnonzero(reach & obs) if n else np.zeros(0, int)


# ---------------------------------------------------------------------------------------------------------------
# İç veri (MATLAB @mpcdata karşılığı)
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class MPCData:
    """MPC'nin derlenmiş iç verisi (MATLAB ``getCodeGenerationData`` alan adlarıyla). Tümü **boyutsuz** (ölçekli);
    indisler 0 tabanlı. C kodu üretimi bu yapıyı kullanır (bkz. SPT-TSR-001 "Çevrimiçi hesap")."""
    Ts: float
    p: int
    moves: np.ndarray
    mvindex: np.ndarray
    mdindex: np.ndarray
    udindex: np.ndarray
    myindex: np.ndarray
    uyindex: np.ndarray
    Uscale: np.ndarray
    Yscale: np.ndarray
    uoff: np.ndarray
    yoff: np.ndarray
    xpoff: np.ndarray
    dxpoff: np.ndarray
    nxp: int
    nxindist: int
    nxoutdist: int
    nxnoise: int
    nuindist: int
    nuoutdist: int
    nunoise: int
    indistchannels: np.ndarray
    outdistchannels: np.ndarray
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    Bu: np.ndarray
    Bv: np.ndarray
    Cm: np.ndarray
    Dv: np.ndarray
    Dvm: np.ndarray
    L: np.ndarray
    M: np.ndarray
    Pcov: np.ndarray
    isQP: bool
    H: np.ndarray
    Hinv: np.ndarray
    Linv: np.ndarray
    BadH: int
    Kx: np.ndarray
    Kr: np.ndarray
    Ku1: np.ndarray
    Kv: np.ndarray
    Kut: np.ndarray
    Ac: np.ndarray
    Mlim: np.ndarray
    Mx: np.ndarray
    Mu1: np.ndarray
    Mv: np.ndarray
    Mrows: np.ndarray
    Jm: np.ndarray
    I1: np.ndarray
    I2Jm: np.ndarray
    SuJm: np.ndarray
    Su1: np.ndarray
    Sx: np.ndarray
    Hv: np.ndarray
    Wy: np.ndarray
    Wu: np.ndarray
    Wdu: np.ndarray
    Wecr: float
    utarget: np.ndarray
    max_iter: int
    tol: float
    use_warm_start: bool
    use_suboptimal: bool
    Ap: np.ndarray
    Bup: np.ndarray
    Bvp: np.ndarray
    Bdp: np.ndarray
    Cp: np.ndarray
    Dvp: np.ndarray
    Ddp: np.ndarray

    @property
    def nmv(self):
        return int(self.mvindex.size)

    @property
    def nmd(self):
        return int(self.mdindex.size)

    @property
    def nud(self):
        return int(self.udindex.size)

    @property
    def nv(self):
        return self.nmd + 1

    @property
    def ny(self):
        return int(self.myindex.size + self.uyindex.size)

    @property
    def nmy(self):
        return int(self.myindex.size)

    @property
    def nxdist(self):
        return self.nxindist + self.nxoutdist

    @property
    def nxQP(self):
        return self.nxp + self.nxdist

    @property
    def nx(self):
        return self.nxQP + self.nxnoise

    @property
    def degrees(self):
        return int(self.Jm.shape[1] + (1 if self.isQP else 0))

    @property
    def nfree(self):
        return int(self.Jm.shape[1])

    @property
    def mvoff(self):
        return self.uoff[self.mvindex]

    @property
    def mdoff(self):
        return self.uoff[self.mdindex]

    @property
    def myoff(self):
        return self.yoff[self.myindex]

    @property
    def xoff(self):
        return np.concatenate([self.xpoff, np.zeros(self.nxdist + self.nxnoise)])


def _augment(val, p, what):
    a = np.atleast_1d(np.asarray(val, dtype=float)).ravel()
    if a.size == 0:
        raise ControlError(f"{what} boş olamaz.")
    if a.size > p:
        a = a[:p]
    return np.concatenate([a, np.full(p - a.size, a[-1])])


def _weights_matrix(w, n, p, what):
    """Ağırlık → (p, n) dizisi (tekdüze tek satır tekrarlanır, eksik satırlar son satırla doldurulur)."""
    if isinstance(w, (list, tuple)) and len(w) == 1 and isinstance(w[0], np.ndarray) and w[0].ndim == 2 \
            and w[0].shape == (n, n) and n > 1:
        raise ControlError("Köşegen dışı (tam matris) ağırlıklar desteklenmiyor.")
    a = np.asarray(w, dtype=float)
    if a.ndim <= 1:
        a = np.atleast_1d(a).reshape(1, -1)
        if a.shape[1] != n:
            raise ControlError(f"{what} ağırlığının uzunluğu {n} olmalı.")
    if a.shape[1] != n:
        raise ControlError(f"{what} ağırlığının sütun sayısı {n} olmalı.")
    if np.any(~np.isfinite(a)) or np.any(a < 0):
        raise ControlError(f"{what} ağırlıkları sonlu ve negatif olmayan sayılar olmalı.")
    if a.shape[0] > p:
        a = a[:p]
    if 1 < a.shape[0] < p:
        a = np.vstack([a, np.repeat(a[-1:], p - a.shape[0], axis=0)])
    return a


def _moves_vector(m, p):
    if np.ndim(m) == 0:
        mm = int(m)
        return np.array([1] * (mm - 1) + [p - mm + 1], dtype=int)
    return np.asarray(m, dtype=int).ravel()


def _chk_m(m, p):
    """MATLAB mpc_chkm."""
    if m is None:
        return min(_DEFAULT_M, p)
    arr = np.atleast_1d(np.asarray(m, dtype=float)).ravel()
    if arr.size == 0:
        return min(_DEFAULT_M, p)
    if np.any(arr < 1) or np.any(arr != np.round(arr)):
        raise ControlError("Kontrol ufku pozitif tamsayı(lar) olmalı.")
    if arr.size == 1:
        mm = int(arr[0])
        if mm > p:
            warnings.warn("Kontrol ufku tahmin ufkundan büyük; tahmin ufkuna eşitlendi.", ControlWarning, stacklevel=3)
            mm = p
        return mm
    moves = [int(v) for v in arr]
    if sum(moves) > p:
        cs = np.cumsum(moves)
        nb = int(np.flatnonzero(cs > p)[0])
        moves = moves[:nb] + [p - sum(moves[:nb])]
        if nb == 0:
            moves = [1] if p == 1 else [p]
    elif sum(moves) < p:
        moves.append(p - sum(moves))
    if moves[-1] == 0:
        moves.pop()
    return moves


def _check_hessian(H):
    """MATLAB mpc_checkhessian: (L, BadH). BadH 0 iyi, 1 köşegen artırıldı, 2 çok büyük, 3 tekil."""
    small = 10.0 * sqrt(_EPS)

    def chol_ok(M):
        try:
            Lc = cholesky(M, lower=True)
        except LinAlgError:
            return None
        return Lc if np.min(np.diag(Lc)) > small else None

    Lc = chol_ok(H)
    if Lc is not None:
        return Lc, 0
    if np.linalg.norm(H, np.inf) >= 1e10:
        return None, 2
    Hc = H.copy()
    for tries in range(5):
        Hc = Hc + (10.0 ** tries) * small * np.eye(H.shape[0])
        Lc = chol_ok(Hc)
        if Lc is not None:
            return Lc, 1
    return None, 3


# ---------------------------------------------------------------------------------------------------------------
# Durum ve benzetim seçenekleri
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class MPCState:
    """MATLAB ``mpcstate``: Plant (bitki durumları, ofset dahil), Disturbance, Noise, LastMove (mühendislik birimi),
    Covariance; ``iA`` sıcak başlangıç aktif kümesi. ``mpcmove`` nesneyi yerinde günceller (MATLAB tutamaç nesnesi)."""
    Plant: np.ndarray
    Disturbance: np.ndarray
    Noise: np.ndarray
    LastMove: np.ndarray
    Covariance: np.ndarray | None = None
    iA: np.ndarray | None = None

    def copy(self) -> "MPCState":
        return copy.deepcopy(self)


class mpcsimopt(Struct):
    """MATLAB ``mpcsimopt``: PlantInitialState, ControllerInitialState (MPCState), UnmeasuredDisturbance (T×nud),
    InputNoise (T×nmv), OutputNoise (T×nmy), RefLookAhead / MDLookAhead ('on'/'off'), Constraints ('on'/'off'),
    Model (model uyumsuzluğu için gerçek bitki ya da {'Plant', 'Nominal'}), OpenLoop ('on'/'off'), MVSignal."""
    _fields = ("PlantInitialState", "ControllerInitialState", "UnmeasuredDisturbance", "InputNoise", "OutputNoise",
               "RefLookAhead", "MDLookAhead", "Constraints", "Model", "OpenLoop", "MVSignal")

    def __init__(self, *args, **kw):
        super().__init__()
        for f in self._fields:
            dict.__setitem__(self, f, None)
        for k, v in dict(*args, **kw).items():
            self[k] = v


SimResult = namedtuple("SimResult", ["y", "t", "u", "xp", "xmpc", "status"])


@dataclass
class MPCLinearController:
    """Kısıtsız MPC'nin eşdeğer doğrusal kontrolcüsü (MATLAB ``ss(mpcobj)``); mühendislik birimlerinde."""
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    Ts: float
    input_names: tuple
    output_names: tuple
    state_names: tuple


# ---------------------------------------------------------------------------------------------------------------
# MPC nesnesi
# ---------------------------------------------------------------------------------------------------------------
class MPC:
    """MATLAB ``mpc(plant, Ts, p, m, W, MV, OV, DV)`` karşılığı.

    ``plant``: durum uzayı (A, B, C, D, Ts öznitelikli nesne, (A, B, C, D[, Ts]) ya da :func:`setmpcsignals` sonucu)
    veya MATLAB Model yapısı gibi sözlük ``{'Plant', 'Disturbance', 'Noise', 'Nominal'}``. Sürekli bitki ``Ts`` ile
    ZOH ayrıklaştırılır. ``signals``: ``{'MV': [...], 'MD': [...], 'UD': [...], 'MO': [...], 'UO': [...]}``
    (0 tabanlı; plant.mpc_signals'ı geçersiz kılar).

    Özellikler MATLAB adlarıyla: ``Ts``, ``PredictionHorizon`` (p, varsayılan 10), ``ControlHorizon`` (m, varsayılan
    2; tamsayı ya da blok vektörü), ``Model``, ``ManipulatedVariables`` / ``MV`` (liste: Min, Max, MinECR, MaxECR,
    RateMin, RateMax, RateMinECR, RateMaxECR, Target, Name, Units, ScaleFactor), ``OutputVariables`` / ``OV``
    (Min, Max, MinECR, MaxECR, Name, Units, ScaleFactor), ``DisturbanceVariables`` / ``DV`` (MD'ler sonra UD'ler),
    ``Weights`` (ManipulatedVariables, ManipulatedVariablesRate, OutputVariables, ECR), ``Optimizer``
    (MaxIterations 'default' = 4·max(30, n_z + n_kısıt), ConstraintTolerance 1e-6, UseWarmStart, UseSuboptimalSolution,
    MinOutputECR). Değişiklikler bir sonraki hesapta iç veriye otomatik yansır.
    """

    def __init__(self, plant, Ts=None, p=None, m=None, W=None, MV=None, OV=None, DV=None, *, signals=None):
        if isinstance(plant, MPC):
            src = copy.deepcopy(plant)
            self.__dict__.update(src.__dict__)
            return
        if isinstance(plant, dict):
            model = _Model({k: v for k, v in dict(plant).items()})
        else:
            model = _Model(Plant=plant)
        for f in _Model._fields:
            if f not in model:
                model[f] = None
        A, B, C, D, Tsp = _ss_data(model.Plant)
        if Tsp is None:
            raise ControlError("Bitki modeli dinamik bir sistem olmalı.")
        ny, nu = D.shape
        sig = dict(getattr(model.Plant, "mpc_signals", None) or {})
        if signals:
            sig.update({k.upper(): v for k, v in signals.items()})
        self._signals = {k: sorted(int(i) for i in np.atleast_1d(v)) for k, v in sig.items()}
        mv, md, ud, mo, uo = self._indices(nu, ny)
        if Ts is None:
            if Tsp == 0.0:
                raise ControlError("Sürekli zamanlı bitki için denetleyici örnekleme zamanı Ts verilmeli.")
            Ts = Tsp
        if not (np.isscalar(Ts) and np.isfinite(Ts) and Ts > 0):
            raise ControlError("Denetleyici örnekleme zamanı Ts pozitif bir sayı olmalı.")
        if Tsp > 0 and abs(Tsp - Ts) > 1e-10:
            raise ControlError("Ayrık bitkinin örnekleme zamanı denetleyicininkinden farklı; yeniden örnekleme (d2d) "
                               "desteklenmiyor (bitkiyi denetleyici Ts'iyle ayrıklaştırın).")
        self.Ts = float(Ts)
        object.__setattr__(self, "_p", self._chk_p(p))
        object.__setattr__(self, "_m", _chk_m(m, self._p))
        X, DX, U, Y = _nominal_of(model.Nominal, A.shape[0], nu, ny)
        if np.any(U[ud] != 0):
            raise ControlError("Ölçülmeyen bozucu girişlerinin nominal değeri sıfır olmalı.")
        model.Nominal = _Nominal(X=X, U=U, Y=Y, DX=DX)
        if np.any(D[:, mv] != 0):
            raise ControlError("Bitkide MV'den çıkışa doğrudan geçiş (D ≠ 0) var; MPC bunu desteklemez.")
        self.Model = model
        in_names = list(getattr(model.Plant, "input_names", ()) or ())
        out_names = list(getattr(model.Plant, "output_names", ()) or ())
        self._in_names = [in_names[i] if i < len(in_names) and in_names[i] else _sig_name(i, mv, md, ud)
                          for i in range(nu)]
        self._out_names = [out_names[i] if i < len(out_names) and out_names[i] else _sig_name_y(i, mo, uo)
                           for i in range(ny)]
        self.Optimizer = None
        self.Weights = W
        self.ManipulatedVariables = MV
        self.OutputVariables = OV
        self.DisturbanceVariables = DV
        self._outdist = None
        self._cache_key = None
        self._cache = None
        self._ws = None

    # --- MATLAB özellik adları -------------------------------------------------------------------------------
    @staticmethod
    def _chk_p(p):
        if p is None:
            return _DEFAULT_P
        if not np.isscalar(p) or not np.isfinite(p) or round(p) < 1:
            raise ControlError("Tahmin ufku pozitif, sonlu bir tamsayı olmalı.")
        return int(round(p))

    @property
    def PredictionHorizon(self):
        return self._p

    @PredictionHorizon.setter
    def PredictionHorizon(self, p):
        self._p = self._chk_p(p)
        self._m = _chk_m(self._m, self._p)

    @property
    def ControlHorizon(self):
        return self._m

    @ControlHorizon.setter
    def ControlHorizon(self, m):
        self._m = _chk_m(m, self._p)

    @property
    def ManipulatedVariables(self):
        return self._MV

    @ManipulatedVariables.setter
    def ManipulatedVariables(self, specs):
        nu, ny = self._nu_ny()
        mv = self._indices(nu, ny)[0]
        self._MV = _make_specs(specs, len(mv), _MVSpec, [self._in_names[i] for i in mv], "MV")

    MV = ManipulatedVariables

    @property
    def OutputVariables(self):
        return self._OV

    @OutputVariables.setter
    def OutputVariables(self, specs):
        nu, ny = self._nu_ny()
        self._OV = _make_specs(specs, ny, _OVSpec, self._out_names, "OV")

    OV = OutputVariables

    @property
    def DisturbanceVariables(self):
        return self._DV

    @DisturbanceVariables.setter
    def DisturbanceVariables(self, specs):
        nu, ny = self._nu_ny()
        _, md, ud, _, _ = self._indices(nu, ny)
        idx = list(md) + list(ud)
        self._DV = _make_specs(specs, len(idx), _DVSpec, [self._in_names[i] for i in idx], "DV")

    DV = DisturbanceVariables

    @property
    def Model(self):
        return self._Model

    @Model.setter
    def Model(self, model):
        """Model yapısı (Plant, Disturbance, Noise, Nominal); düz sözlük de kabul edilir."""
        mdl = _Model({k: v for k, v in dict(model).items()})
        for f in _Model._fields:
            if f not in mdl:
                mdl[f] = None
        self._Model = mdl

    @property
    def Optimizer(self):
        return self._Opt

    @Optimizer.setter
    def Optimizer(self, opt):
        o = _Optimizer(MaxIterations="default", ConstraintTolerance=1e-6, UseWarmStart=True,
                       UseSuboptimalSolution=False, MinOutputECR=0.0)
        for k, v in dict(opt or {}).items():
            o[k] = v
        self._Opt = o

    @property
    def Weights(self):
        return self._W

    @Weights.setter
    def Weights(self, W):
        nu, ny = self._nu_ny()
        mv, _, _, mo, uo = self._indices(nu, ny)
        self._W = _chk_weights(W, self._p, len(mv), ny, mo, uo)

    # --- yardımcılar -------------------------------------------------------------------------------------------
    def _nu_ny(self):
        D = np.atleast_2d(np.asarray(_ss_data(self.Model.Plant)[3]))
        return D.shape[1], D.shape[0]

    def _indices(self, nu, ny):
        sig = self._signals
        for k in sig:
            if k not in ("MV", "MD", "UD", "MO", "UO"):
                raise ControlError(f"Geçersiz sinyal türü: {k} (MV, MD, UD, MO, UO)")
        md = np.array(sig.get("MD", []), dtype=int)
        ud = np.array(sig.get("UD", []), dtype=int)
        uo = np.array(sig.get("UO", []), dtype=int)
        mv = np.array(sorted(set(sig.get("MV", [])) | (set(range(nu)) - set(sig.get("MV", [])) - set(md) - set(ud))),
                      dtype=int)
        mo = np.array(sorted(set(sig.get("MO", [])) | (set(range(ny)) - set(sig.get("MO", [])) - set(uo))), dtype=int)
        for name, idx, n in (("MV", mv, nu), ("MD", md, nu), ("UD", ud, nu), ("MO", mo, ny), ("UO", uo, ny)):
            if idx.size and (idx.min() < 0 or idx.max() >= n):
                raise ControlError(f"{name} indisleri 0..{n - 1} aralığında olmalı.")
        if set(mv) & set(md) or set(mv) & set(ud) or set(md) & set(ud):
            raise ControlError("Bir giriş birden çok sinyal türüne atanamaz (MV / MD / UD).")
        if set(mo) & set(uo):
            raise ControlError("Bir çıkış hem ölçülen hem ölçülmeyen olamaz.")
        if mv.size == 0:
            raise ControlError("Bitkinin en az bir MV girişi olmalı.")
        if mo.size == 0:
            raise ControlError("Bitkinin en az bir ölçülen çıkışı (MO) olmalı.")
        return mv, md, ud, mo, uo

    def setoutdist(self, method="integrators", model=None):
        """MATLAB ``setoutdist``: 'integrators' (varsayılan kural) ya da 'model' (özel çıkış bozucu modeli; ny
        çıkışlı, birim beyaz gürültüyle sürülen LTI; DC kazancı 0 ise bozucu modeli eklenmez)."""
        method = str(method).lower()
        if method == "integrators":
            self._outdist = None
        elif method == "model":
            self._outdist = None if model is None else model
        else:
            raise ControlError("setoutdist yöntemi 'integrators' ya da 'model' olmalı.")

    def _key(self):
        def norm(v):
            if isinstance(v, np.ndarray):
                return ("nd", v.shape, v.tobytes())
            if isinstance(v, dict):
                return tuple((k, norm(x)) for k, x in v.items())
            if isinstance(v, (list, tuple)):
                return tuple(norm(x) for x in v)
            if isinstance(v, (int, float, str, bool, type(None), np.number)):
                return v
            try:
                return ("ss",) + tuple(norm(np.asarray(a)) if a is not None else None for a in _ss_data(v))
            except ControlError:
                return ("id", id(v))
        parts = (self.Ts, self._p, norm(self._m), norm(dict(self.Model)), norm(self._signals),
                 norm([dict(s) for s in self._MV]), norm([dict(s) for s in self._OV]),
                 norm([dict(s) for s in self._DV]), norm(dict(self._W)), norm(dict(self.Optimizer)),
                 norm(self._outdist))
        return pickle.dumps(parts)

    @property
    def data(self) -> MPCData:
        """Güncel iç veri (gerekirse yeniden derlenir)."""
        key = self._key()
        if self._cache is None or key != self._cache_key:
            self._cache = _build(self)
            self._cache_key = key
            self._ws = None
        return self._cache

    def _workspace(self, d: MPCData):
        if d.isQP and (self._ws is None or not self._ws.fits(d.degrees, d.Ac.shape[0])):
            self._ws = QPWorkspace(d.degrees, d.Ac.shape[0])
        return self._ws

    def __repr__(self):
        nu, ny = self._nu_ny()
        mv, md, ud, mo, uo = self._indices(nu, ny)
        return (f"MPC(Ts={self.Ts:g}, p={self._p}, m={self._m}, MV={mv.size}, MD={md.size}, UD={ud.size}, "
                f"MO={mo.size}, UO={uo.size})")

    # --- MATLAB işlevleri (yöntem olarak) ------------------------------------------------------------------------
    def mpcstate(self, plant=None, disturbance=None, noise=None, lastu=None, covariance=None) -> MPCState:
        return mpcstate(self, plant, disturbance, noise, lastu, covariance)

    def mpcmove(self, state, ym=None, r=None, v=None):
        return mpcmove(self, state, ym, r, v)

    def sim(self, T=None, r=None, v=None, options=None):
        return sim(self, T, r, v, options)

    def getEstimator(self):
        return getEstimator(self)

    def cloffset(self):
        return cloffset(self)

    def ss(self, signals="", ref_preview=False, md_preview=False):
        return _mpc_ss(self, signals, ref_preview, md_preview)


def mpc(plant, Ts=None, p=None, m=None, W=None, MV=None, OV=None, DV=None, *, signals=None) -> MPC:
    """MATLAB ``mpc(plant, Ts, p, m, W, MV, OV, DV)`` (bkz. :class:`MPC`)."""
    return MPC(plant, Ts, p, m, W, MV, OV, DV, signals=signals)


def _nominal_of(nom, nx, nu, ny):
    """Model.Nominal → (X, DX, U, Y); eksik alanlar sıfır (MATLAB mpc_chkoffsets)."""
    src = {str(k).upper(): v for k, v in dict(nom or {}).items()}
    for k in src:
        if k not in ("X", "DX", "U", "Y"):
            raise ControlError(f"Model.Nominal alanı geçersiz: {k} (X, U, Y, DX)")

    def get(key, n):
        v = src.get(key)
        return np.zeros(n) if v is None or np.size(v) == 0 else _vec(v, n, f"Nominal.{key}")
    return get("X", nx), get("DX", nx), get("U", nu), get("Y", ny)


def _finite_vec(v, n, what):
    """_vec + sonluluk: MATLAB mpcmove / mpcstate NaN / ±∞ değerleri reddeder ("must be real finite values");
    NaN kabul edilseydi kestirici durumu kalıcı olarak NaN olurdu."""
    a = _vec(v, n, what)
    if not np.all(np.isfinite(a)):
        raise ControlError(f"{what} sonlu gerçel değerler olmalı (NaN / sonsuz içeremez).")
    return a


def _vec(v, n, what):
    a = np.atleast_1d(np.asarray(v, dtype=float)).ravel()
    if a.size != n:
        raise ControlError(f"{what} {n} elemanlı olmalı.")
    return a


def _sig_name(i, mv, md, ud):
    for name, idx in (("MV", mv), ("MD", md), ("UD", ud)):
        hits = np.flatnonzero(idx == i)
        if hits.size:
            return f"{name}{hits[0] + 1}"
    return f"u{i + 1}"


def _sig_name_y(i, mo, uo):
    for name, idx in (("MO", mo), ("UO", uo)):
        hits = np.flatnonzero(idx == i)
        if hits.size:
            return f"{name}{hits[0] + 1}"
    return f"y{i + 1}"


def _make_specs(specs, n, cls, names, what):
    if specs is None:
        lst = []
    elif isinstance(specs, dict):
        lst = [specs]
    else:
        lst = list(specs)
    if len(lst) > n:
        raise ControlError(f"{what} tanımı en çok {n} elemanlı olabilir.")
    out = []
    for i in range(n):
        s = cls()
        src = lst[i] if i < len(lst) else {}
        for k, v in (src or {}).items():
            s[k] = v
        if cls is _MVSpec:
            defaults = dict(Min=-np.inf, Max=np.inf, MinECR=0.0, MaxECR=0.0, RateMin=-np.inf, RateMax=np.inf,
                            RateMinECR=0.0, RateMaxECR=0.0, Target="nominal", Units="", ScaleFactor=1.0,
                            Type="continuous")
        elif cls is _OVSpec:
            defaults = dict(Min=-np.inf, Max=np.inf, MinECR=1.0, MaxECR=1.0, Units="", ScaleFactor=1.0)
        else:
            defaults = dict(Units="", ScaleFactor=1.0)
        for k, v in defaults.items():
            if s.get(k) is None:
                s[k] = v
        if not s.get("Name"):
            s["Name"] = names[i] if i < len(names) else f"{what}{i + 1}"
        out.append(s)
    return out


def _chk_weights(W, p, nmv, ny, mo, uo):
    """MATLAB mpc_chkweights (varsayılanlar ve nmv < ny kuralı; ECR = 1e5·max(ağırlıklar) yalnız verilmemişse)."""
    w = _Weights()
    if W is not None:
        for k, v in dict(W).items():
            w[k] = v
    if w.get("ManipulatedVariables") is None:
        w.ManipulatedVariables = np.zeros(nmv)
    if w.get("ManipulatedVariablesRate") is None:
        w.ManipulatedVariablesRate = 0.1 * np.ones(nmv)
    if w.get("OutputVariables") is None:
        wo = np.ones(ny)
        if nmv < ny:
            order = list(mo) + list(uo)
            wo[order[nmv:]] = 0.0
        w.OutputVariables = wo
    for k, n in (("ManipulatedVariables", nmv), ("ManipulatedVariablesRate", nmv), ("OutputVariables", ny)):
        a = np.asarray(w[k], dtype=float)
        a = a.reshape(1, -1) if a.ndim <= 1 else a
        if a.shape[1] != n or np.any(~np.isfinite(a)) or np.any(a < 0):
            raise ControlError(f"Weights.{k}: {n} sütunlu, sonlu, negatif olmayan değerler olmalı.")
        w[k] = np.atleast_1d(np.asarray(w[k], dtype=float))
    if w.get("ECR") is None:
        allw = np.concatenate([np.ravel(w.ManipulatedVariables), np.ravel(w.ManipulatedVariablesRate),
                               np.ravel(w.OutputVariables)])
        w.ECR = 1e5 * float(allw.max()) if allw.size else 1e5
    ecr = float(np.asarray(w.ECR, dtype=float).ravel()[0])
    if not np.isfinite(ecr) or ecr <= 0:
        raise ControlError("Weights.ECR pozitif ve sonlu olmalı.")
    w.ECR = ecr
    return w


# ---------------------------------------------------------------------------------------------------------------
# Derleme (MATLAB @mpcdata/update.m karşılığı)
# ---------------------------------------------------------------------------------------------------------------
def _prep_plant(A, B, C, D, Tsp, Ts, X, DX, U, Y, Uscale, Yscale):
    """MATLAB mpc_chkmodel: DX ofset girişini ekle, ZOH ile ayrıklaştır, boyutsuzlaştır."""
    nx = A.shape[0]
    ny = C.shape[0]
    if nx == 0:
        raise ControlError("Bitki modelinin en az bir durumu olmalı.")
    Bx = np.hstack([B, DX.reshape(-1, 1)])
    Dx = np.hstack([D, np.zeros((ny, 1))])
    if Tsp == 0.0:
        Ad, Bd = _c2d_zoh(A, Bx, Ts)
        dxoff = DX * Ts
    elif abs(Tsp - Ts) <= 1e-10:
        Ad, Bd = A.copy(), Bx
        dxoff = DX.copy()
    else:
        raise ControlError("Bitki örnekleme zamanı denetleyicininkinden farklı; ayrık bitkiyi yeniden örnekleme "
                           "desteklenmiyor (bitkiyi denetleyici Ts'iyle ayrıklaştırın).")
    us1 = np.concatenate([Uscale, [1.0]])
    Bs = Bd * us1[None, :]
    Cs = C / Yscale[:, None]
    Ds = (Dx / Yscale[:, None]) * us1[None, :]
    return Ad, Bs, Cs, Ds, X.copy(), dxoff, U / Uscale, Y / Yscale


def _detect_id_channels(A, B, C, D, ud, mo):
    A_ = A.copy()
    C_ = C.copy()
    j = 0
    ch = []
    for ct, col in enumerate(ud):
        nx = A_.shape[0]
        B1 = np.concatenate([B[:, col], np.zeros(j)])
        A1 = np.zeros((nx + 1, nx + 1))
        A1[:nx, :nx] = A_
        A1[:nx, nx] = B1
        A1[nx, nx] = 1.0
        C1 = np.hstack([C_, D[:, [col]]])
        if _rank(np.vstack([np.eye(nx + 1) - A1, C1[mo, :]])) == nx + 1:
            A_, C_ = A1, C1
            j += 1
            ch.append(ct)
    return ch


def _build(obj: MPC) -> MPCData:
    Ts = float(obj.Ts)
    p = int(obj.PredictionHorizon)
    moves = _moves_vector(obj.ControlHorizon, p)
    A0, B0, C0, D0, Tsp = _ss_data(obj.Model.Plant)
    ny, nu = D0.shape
    nxp0 = A0.shape[0]
    mv, md, ud, mo, uo = obj._indices(nu, ny)
    nmv, nmd, nud, nmy = mv.size, md.size, ud.size, mo.size
    nv = nmd + 1
    if len(obj.MV) != nmv or len(obj.OV) != ny or len(obj.DV) != nmd + nud:
        raise ControlError("MV / OV / DV tanımlarının sayısı bitkinin sinyal sayılarıyla uyuşmuyor "
                           "(bitki değiştiyse MV, OV, DV yeniden atanmalı).")
    # ölçek çarpanları
    Uscale = np.ones(nu)
    Yscale = np.ones(ny)
    for i, k in enumerate(mv):
        Uscale[k] = _scale(obj.MV[i], "MV")
    for i, k in enumerate(list(md) + list(ud)):
        Uscale[k] = _scale(obj.DV[i], "DV")
    for i in range(ny):
        Yscale[i] = _scale(obj.OV[i], "OV")
    X, DX, U, Y = _nominal_of(obj.Model.Nominal, nxp0, nu, ny)
    if np.any(U[ud] != 0):
        raise ControlError("Ölçülmeyen bozucu girişlerinin nominal değeri sıfır olmalı.")
    if np.any(D0[:, mv] != 0):
        raise ControlError("Bitkide MV'den çıkışa doğrudan geçiş (D ≠ 0) var; MPC bunu desteklemez.")
    Ap, Bp, Cp, Dp, xpoff, dxpoff, uoff, yoff = _prep_plant(A0, B0, C0, D0, Tsp, Ts, X, DX, U, Y, Uscale, Yscale)
    nxp = Ap.shape[0]
    off_col = nu
    mdDX = list(md) + [off_col]
    # --- giriş bozucu modeli (mpc_augmentID)
    if nud == 0:
        if obj.Model.Disturbance is not None:
            raise ControlError("Giriş bozucu modeli verilmiş ama bitkide ölçülmeyen bozucu (UD) girişi yok.")
        iseq = list(mv) + mdDX
        A1, B1, C1, D1 = Ap.copy(), Bp[:, iseq], Cp.copy(), Dp[:, iseq]
        nxid = nuid = 0
        idch = []
    else:
        if obj.Model.Disturbance is None:
            idch = _detect_id_channels(Ap, Bp, Cp, Dp, ud, mo)
            nxid_ = len(idch)
            Abar = np.eye(nxid_)
            Bbar = np.zeros((nxid_, nud))
            Cbar = np.zeros((nud, nxid_))
            Dbar = np.eye(nud)
            for ct, j in enumerate(idch):
                Bbar[ct, j] = Ts
                Cbar[j, ct] = 1.0
                Dbar[j, j] = 0.0
        else:
            idch = []
            Abar, Bbar, Cbar, Dbar = _dist_model(obj.Model.Disturbance, nud, Ts, Uscale[ud], "Giriş bozucu")
        nxid, nuid = Bbar.shape
        Bd_, Dd_ = Bp[:, ud], Dp[:, ud]
        newb = np.vstack([Bp, np.zeros((nxid, nu + 1))])
        newd = Dp.copy()
        udidx = list(ud)
        B2_ = np.vstack([Bd_ @ Dbar, Bbar])
        D2_ = Dd_ @ Dbar
        if nud == nuid:
            newb[:, ud] = B2_
            newd[:, ud] = D2_
        elif nud > nuid:
            keep = list(ud[:nuid])
            newb[:, keep] = B2_
            newd[:, keep] = D2_
            udidx = keep
        else:
            newb[:, ud] = B2_[:, :nud]
            newb = np.hstack([newb, B2_[:, nud:]])
            newd[:, ud] = D2_[:, :nud]
            newd = np.hstack([newd, D2_[:, nud:]])
            udidx = list(ud) + list(range(nu + 1, nu + 1 + nuid - nud))
        A1 = np.block([[Ap, Bd_ @ Cbar], [np.zeros((nxid, nxp)), Abar]])
        C1 = np.hstack([Cp, Dd_ @ Cbar])
        iseq = list(mv) + mdDX + udidx
        B1, D1 = newb[:, iseq], newd[:, iseq]
    # --- çıkış bozucu modeli (mpc_augmentOD)
    ywt = _weights_matrix(obj.Weights.OutputVariables, ny, p, "Çıkış")
    nx1 = A1.shape[0]
    nin1 = B1.shape[1]
    if obj._outdist is None:
        order = np.argsort(-np.sum(np.abs(np.vstack([ywt, np.zeros((1, ny))])), axis=0), kind="stable")
        A2, C2 = A1, C1
        odch = []
        for ct in order:
            if ct in mo:
                n_ = A2.shape[0]
                A3 = np.zeros((n_ + 1, n_ + 1))
                A3[:n_, :n_] = A2
                A3[n_, n_] = 1.0
                C3 = np.hstack([C2, np.zeros((ny, 1))])
                C3[ct, n_] = 1.0
                if _rank(np.vstack([np.eye(n_ + 1) - A3, C3[mo, :]])) == n_ + 1:
                    A2, C2 = A3, C3
                    odch.append(int(ct))
        nod = len(odch)
        B2 = np.block([[B1, np.zeros((nx1, nod))], [np.zeros((nod, nin1)), Ts * np.eye(nod)]])
        D2 = np.hstack([D1, np.zeros((ny, nod))])
        nxod, nuod = nod, nod
    else:
        odch = []
        Ao, Bo, Co, Do = _dist_model(obj._outdist, ny, Ts, Yscale, "Çıkış bozucu")
        if np.linalg.norm(_dcgain_d(Ao, Bo, Co, Do), np.inf) == 0:
            A2, B2, C2, D2 = A1, B1, C1, D1
            nxod = nuod = 0
        else:
            nxod, nuod = Bo.shape
            A2 = _blkdiag(A1, Ao)
            B2 = np.block([[B1, np.zeros((nx1, nuod))], [np.zeros((nxod, nin1)), Bo]])
            C2 = np.hstack([C1, Co])
            D2 = np.hstack([D1, Do])
    nxQP = A2.shape[0]
    # --- kısıtlar (mpc_constraints), boyutsuz ve ofsetsiz
    umin, umax, Vumin, Vumax = (np.zeros((p, nmv)) for _ in range(4))
    dumin, dumax, Vdumin, Vdumax, utarget = (np.zeros((p, nmv)) for _ in range(5))
    mvoff = uoff[mv]
    for i in range(nmv):
        s = obj.MV[i]
        sf = _scale(s, "MV")
        mn, mx = _augment(s.Min, p, "MV.Min") / sf, _augment(s.Max, p, "MV.Max") / sf
        rmn, rmx = _augment(s.RateMin, p, "MV.RateMin") / sf, _augment(s.RateMax, p, "MV.RateMax") / sf
        if np.any(mx < mn) or np.any(rmx < rmn):
            raise ControlError(f"MV{i + 1}: Max ≥ Min ve RateMax ≥ RateMin olmalı.")
        if np.any(rmn > 0) or np.any(rmx < 0):
            raise ControlError(f"MV{i + 1}: RateMin ≤ 0 ≤ RateMax olmalı.")
        umin[:, i], umax[:, i] = mn - mvoff[i], mx - mvoff[i]
        dumin[:, i], dumax[:, i] = rmn, rmx
        for arr, key in ((Vumin, "MinECR"), (Vumax, "MaxECR"), (Vdumin, "RateMinECR"), (Vdumax, "RateMaxECR")):
            vv = _augment(s[key], p, f"MV.{key}")
            if np.any(vv < 0) or np.any(~np.isfinite(vv)):
                raise ControlError(f"MV{i + 1}.{key} negatif olmayan sonlu değer(ler) olmalı.")
            arr[:, i] = vv
        tg = s.Target
        if isinstance(tg, str):
            if tg.lower() != "nominal":
                raise ControlError("MV.Target 'nominal' ya da sayı olmalı.")
            utarget[:, i] = 0.0
        else:
            utarget[:, i] = _augment(tg, p, "MV.Target") / sf - mvoff[i]
        typ = s.get("Type")
        if typ is not None and not (isinstance(typ, str) and typ.lower() == "continuous"):
            raise ControlError("Ayrık (binary / integer / sonlu küme) MV'ler desteklenmiyor.")
    ymin, ymax, Vymin, Vymax = (np.zeros((p, ny)) for _ in range(4))
    minecr = float(obj.Optimizer.MinOutputECR or 0.0)
    for i in range(ny):
        s = obj.OV[i]
        sf = _scale(s, "OV")
        mn, mx = _augment(s.Min, p, "OV.Min") / sf, _augment(s.Max, p, "OV.Max") / sf
        if np.any(mx < mn):
            raise ControlError(f"OV{i + 1}: Max ≥ Min olmalı.")
        ymin[:, i], ymax[:, i] = mn - yoff[i], mx - yoff[i]
        for arr, key in ((Vymin, "MinECR"), (Vymax, "MaxECR")):
            vv = _augment(s[key], p, f"OV.{key}")
            if np.any(~np.isfinite(vv)):
                raise ControlError(f"OV{i + 1}.{key} sonlu olmalı.")
            arr[:, i] = np.maximum(vv, minecr)
    isQP = bool(np.any(np.isfinite(umin)) or np.any(np.isfinite(umax)) or np.any(np.isfinite(dumin))
                or np.any(np.isfinite(dumax)) or np.any(np.isfinite(ymin)) or np.any(np.isfinite(ymax)))
    # --- tahmin modeli katsayıları (mpc_constraintcoef)
    Bu, Bv = B2[:, :nmv], B2[:, nmv:nmv + nv]
    Dv = D2[:, nmv:nmv + nv]
    nb = moves.size
    nfree = nb * nmv
    Jm = np.zeros((p * nmv, nfree))
    j = 0
    for i, mi in enumerate(moves):
        Jm[j * nmv:(j + 1) * nmv, i * nmv:(i + 1) * nmv] = np.eye(nmv)
        j += int(mi)
    SuJm, Sx, Su1, Hv, I2Jm = _prediction(A2, Bu, Bv, C2, Dv, Jm, p)
    I1 = np.kron(np.ones((p, 1)), np.eye(nmv))
    # --- ağırlıklar (kareleri, ufuk boyunca açılmış köşegenler)
    uwt = _weights_matrix(obj.Weights.ManipulatedVariables, nmv, p, "MV")
    duwt = _weights_matrix(obj.Weights.ManipulatedVariablesRate, nmv, p, "MV oranı")
    Wy = _expand_w(ywt, p)
    Wu = _expand_w(uwt, p)
    Wdu = _expand_w(duwt, p)
    WySuJm = Wy[:, None] * SuJm
    WuI2Jm = Wu[:, None] * I2Jm
    H = SuJm.T @ WySuJm + Jm.T @ (Wdu[:, None] * Jm) + I2Jm.T @ WuI2Jm
    Ku1 = Su1.T @ WySuJm + I1.T @ WuI2Jm
    Kut = -WuI2Jm
    Kx = Sx.T @ WySuJm
    Kv = Hv.T @ WySuJm
    Kr = -WySuJm
    Wecr = float(obj.Weights.ECR)
    # --- kısıt matrisleri (mpc_buildmat)
    if isQP:
        I3 = np.eye(nfree)
        Mu = np.vstack([SuJm, -SuJm, I2Jm, -I2Jm, I3, -I3])
        aux = 2 * (nmv * p + nfree)
        Mx = np.vstack([-Sx, Sx, np.zeros((aux, nxQP))])
        Mu1 = np.vstack([-Su1, Su1, -I1, I1, np.zeros((2 * nfree, nmv))])
        Mv = np.vstack([-Hv, Hv, np.zeros((aux, (p + 1) * nv))])
        Mrows = np.arange(Mu.shape[0])
        index = np.zeros(p * nmv, dtype=bool)
        j = 0
        for mi in moves:
            index[j * nmv:(j + 1) * nmv] = True
            j += int(mi)
        index2 = np.flatnonzero(~index)
        dumin_v, dumax_v = dumin.ravel()[index], dumax.ravel()[index]
        Vdumin_v, Vdumax_v = Vdumin.ravel()[index], Vdumax.ravel()[index]
        k = 0
        for mi in moves:
            blk = slice(k, k + int(mi))
            umin[k] = umin[blk].max(axis=0)
            umax[k] = umax[blk].min(axis=0)
            Vumin[k] = Vumin[blk].min(axis=0)
            Vumax[k] = Vumax[blk].min(axis=0)
            k += int(mi)
        umin_v, umax_v = umin.ravel()[index], umax.ravel()[index]
        Vumin_v, Vumax_v = Vumin.ravel()[index], Vumax.ravel()[index]
        iU = 2 * p * ny + np.concatenate([index2, index2 + p * nmv])
        keep = np.ones(Mu.shape[0], dtype=bool)
        keep[iU] = False
        Mu, Mx, Mu1, Mv, Mrows = Mu[keep], Mx[keep], Mu1[keep], Mv[keep], Mrows[keep]
        Mlim = np.concatenate([ymax.ravel(), -ymin.ravel(), umax_v, -umin_v, dumax_v, -dumin_v])
        fin = np.isfinite(Mlim)
        Mlim, Mu, Mx, Mu1, Mv, Mrows = Mlim[fin], Mu[fin], Mx[fin], Mu1[fin], Mv[fin], Mrows[fin]
        Me = -np.concatenate([Vymax.ravel(), Vymin.ravel(), Vumax_v, Vumin_v, Vdumax_v, Vdumin_v])[fin]
        Mu = np.hstack([Mu, Me[:, None]])
        H = _blkdiag(H, np.array([[Wecr]]))
        Ac = -Mu
    else:
        Ac = np.zeros((0, nfree))
        Mlim = np.zeros(0)
        Mx = np.zeros((0, nxQP))
        Mu1 = np.zeros((0, nmv))
        Mv = np.zeros((0, (p + 1) * nv))
        Mrows = np.zeros(0, dtype=int)
    Lh, BadH = _check_hessian(H)
    if BadH == 1:
        warnings.warn("MPC Hessian'ı tekile yakın; köşegen hafifçe artırıldı (MATLAB ile aynı düzeltme).",
                      ControlWarning, stacklevel=3)
    elif BadH == 2:
        raise ControlError(f"MPC Hessian'ının normu çok büyük ({np.linalg.norm(H, np.inf):.2e}); bitki muhtemelen "
                           "kararsız ve tahmin ufku uzun — ufku kısaltın ya da ölçek çarpanlarını düzeltin.")
    elif BadH == 3:
        raise ControlError("MPC Hessian'ı tekil ya da pozitif tanımlı değil (ağırlıkları kontrol edin: en az bir MV "
                           "oranı / MV ağırlığı ya da çıkış ağırlığı sıfırdan büyük olmalı).")
    Linv = solve_triangular(Lh, np.eye(H.shape[0]), lower=True)
    Hinv = Linv.T @ Linv
    # --- ölçüm gürültüsü modeli (mpc_chkmnmodel) ve Kalman kestiricisi
    if obj.Model.Noise is None:
        An, Bn, Cn, Dn = np.zeros((0, 0)), np.zeros((0, nmy)), np.zeros((nmy, 0)), np.eye(nmy)
    else:
        An, Bn, Cn, Dn = _dist_model(obj.Model.Noise, nmy, Ts, Yscale[mo], "Ölçüm gürültüsü")
    if np.min(np.linalg.eigvalsh(Dn @ Dn.T)) < 1e-6:
        Dn = Dn + 1e-3 * np.eye(*Dn.shape)
    nxn, nun = Bn.shape
    A3 = _blkdiag(A2, An)
    C3 = np.hstack([C2, np.zeros((ny, nxn))])
    C3[np.ix_(mo, np.arange(nxQP, nxQP + nxn))] = Cn
    B3 = np.block([[B2, np.zeros((nxQP, nun))], [np.zeros((nxn, B2.shape[1])), Bn]])
    D3 = np.hstack([D2, np.zeros((ny, nun))])
    D3[np.ix_(mo, np.arange(D3.shape[1] - nun, D3.shape[1]))] = Dn
    Cm = C3[mo, :]
    unk = [c for c in range(B3.shape[1]) if c != nmv + nv - 1]
    G = B3[:, unk]
    Hk = D3[mo][:, unk]
    L, M, P = _kalman(A3, Cm, G, Hk)
    nC = Ac.shape[0]
    mi_ = obj.Optimizer.MaxIterations
    if isinstance(mi_, str):
        if mi_.lower() != "default":
            raise ControlError("Optimizer.MaxIterations 'default' ya da pozitif tamsayı olmalı.")
        max_iter = 4 * max(30, (Ac.shape[1] + nC) if isQP else 0)
    else:
        max_iter = int(mi_)
        if max_iter < 0:
            raise ControlError("Optimizer.MaxIterations negatif olamaz.")
    return MPCData(
        Ts=Ts, p=p, moves=moves, mvindex=mv, mdindex=md, udindex=ud, myindex=mo, uyindex=uo, Uscale=Uscale,
        Yscale=Yscale, uoff=uoff, yoff=yoff, xpoff=xpoff, dxpoff=dxpoff, nxp=nxp, nxindist=nxid, nxoutdist=nxod,
        nxnoise=nxn, nuindist=nuid, nuoutdist=nuod, nunoise=nun, indistchannels=np.array(idch, dtype=int),
        outdistchannels=np.array(odch, dtype=int), A=A3, B=B3, C=C3, D=D3, Bu=B3[:, :nmv], Bv=B3[:, nmv:nmv + nv],
        Cm=Cm, Dv=D3[:, nmv:nmv + nv], Dvm=D3[mo][:, nmv:nmv + nv], L=L, M=M, Pcov=P, isQP=isQP, H=H, Hinv=Hinv,
        Linv=Linv, BadH=BadH, Kx=Kx, Kr=Kr, Ku1=Ku1, Kv=Kv, Kut=Kut, Ac=Ac, Mlim=Mlim, Mx=Mx, Mu1=Mu1, Mv=Mv,
        Mrows=Mrows, Jm=Jm, I1=I1, I2Jm=I2Jm, SuJm=SuJm, Su1=Su1, Sx=Sx, Hv=Hv, Wy=Wy, Wu=Wu, Wdu=Wdu, Wecr=Wecr,
        utarget=utarget.ravel(), max_iter=max_iter, tol=float(obj.Optimizer.ConstraintTolerance),
        use_warm_start=bool(obj.Optimizer.UseWarmStart), use_suboptimal=bool(obj.Optimizer.UseSuboptimalSolution),
        Ap=Ap, Bup=Bp[:, mv], Bvp=Bp[:, mdDX], Bdp=Bp[:, ud], Cp=Cp, Dvp=Dp[:, mdDX], Ddp=Dp[:, ud])


def _scale(spec, what):
    sf = float(np.asarray(spec.ScaleFactor, dtype=float).ravel()[0])
    if not np.isfinite(sf) or sf <= 0:
        raise ControlError(f"{what}.ScaleFactor pozitif ve sonlu olmalı.")
    return sf


def _expand_w(w2, p):
    if w2.shape[0] == 1:
        return np.tile(w2[0] ** 2, p)
    return (w2 ** 2).ravel()


def _prediction(A, Bu, Bv, C, Dv, Jm, p):
    """MATLAB mpc_constraintcoef: y(k+1..k+p) = Su·Δu + Hv·v + Sx·x + Su1·u(k−1)."""
    ny, nx = C.shape
    nu = Bu.shape[1]
    nv = Bv.shape[1]
    CA = C @ A
    Sum = C @ Bu
    Hv = np.zeros((p * ny, (p + 1) * nv))
    Hv[:ny, :nv] = C @ Bv
    Hv[:ny, nv:2 * nv] = Dv
    Sx = np.zeros((p * ny, nx))
    Sx[:ny] = CA
    Su1 = np.zeros((p * ny, nu))
    Su1[:ny] = Sum
    Su = np.zeros((p * ny, p * nu))
    Su[:ny, :nu] = Sum
    for i in range(1, p):
        rows = slice(i * ny, (i + 1) * ny)
        prev = slice((i - 1) * ny, i * ny)
        Sum = Sum + CA @ Bu
        Su1[rows] = Sum
        Su[rows] = np.hstack([Sum, Su[prev, :(p - 1) * nu]])
        Hv[rows] = np.hstack([CA @ Bv, Hv[prev, :p * nv]])
        CA = CA @ A
        Sx[rows] = CA
    SuJm = Su @ Jm
    I2Jm = np.kron(np.tril(np.ones((p, p))), np.eye(nu)) @ Jm
    return SuJm, Sx, Su1, Hv, I2Jm


def _kalman(A, Cm, G, H):
    """Kararlı hâl Kalman: Q = GGᵀ, R = HHᵀ, N = GHᵀ (MATLAB mpc_estimator + kalman 'current')."""
    Q = G @ G.T
    R = H @ H.T
    N = G @ H.T
    Q = 0.5 * (Q + Q.T)
    R = 0.5 * (R + R.T)
    try:
        P = solve_discrete_are(A.T, Cm.T, Q, R, s=N)
        if not np.all(np.isfinite(P)):
            raise LinAlgError("sonlu değil")
    except (LinAlgError, ValueError) as e:
        raise ControlError("Varsayılan Kalman kestiricisi tasarlanamadı: artırılmış model (bitki + bozucu modelleri) "
                           "ölçülen çıkışlardan sezilebilir değil. Bitkinin kararsız / sınırda kararlı gözlenemeyen "
                           f"kipi olabilir ya da bozucu modeli gözlenebilirliği bozuyor ({e}).") from None
    P = 0.5 * (P + P.T)
    S = Cm @ P @ Cm.T + R
    M = np.linalg.solve(S.T, (P @ Cm.T).T).T
    L = np.linalg.solve(S.T, (A @ P @ Cm.T + N).T).T
    return L, M, P


# ---------------------------------------------------------------------------------------------------------------
# mpcstate / mpcmove
# ---------------------------------------------------------------------------------------------------------------
def mpcstate(obj: MPC, plant=None, disturbance=None, noise=None, lastu=None, covariance=None) -> MPCState:
    """MATLAB ``mpcstate(mpcobj, plant, disturbance, noise, lastu, covariance)``. Varsayılanlar: bitki durumu
    Nominal.X, bozucu / gürültü durumları 0, LastMove nominal MV, Covariance kestiricinin P'si."""
    d = obj.data
    xp = d.xpoff.copy() if plant is None else _finite_vec(plant, d.nxp, "Plant durumu")
    xd = np.zeros(d.nxdist) if disturbance is None else _finite_vec(disturbance, d.nxdist, "Disturbance durumu")
    xn = np.zeros(d.nxnoise) if noise is None else _finite_vec(noise, d.nxnoise, "Noise durumu")
    u1 = d.mvoff * d.Uscale[d.mvindex] if lastu is None else _finite_vec(lastu, d.nmv, "LastMove")
    cov = d.Pcov.copy() if covariance is None else np.asarray(covariance, dtype=float)
    return MPCState(Plant=xp, Disturbance=xd, Noise=xn, LastMove=u1, Covariance=cov,
                    iA=np.zeros(d.Ac.shape[0], dtype=bool))


def _ref_seq(r, d: MPCData, p):
    """Referans (boyutsuz, ofsetsiz) p·ny dizisi; r: None, (ny,) ya da (n, ny) önizleme."""
    ny = d.ny
    if r is None:
        rr = np.zeros((1, ny))
    else:
        rr = np.asarray(r, dtype=float)
        if rr.ndim <= 1:
            rr = np.atleast_1d(rr)
            if rr.size != ny:
                raise ControlError(f"Referans {ny} elemanlı olmalı.")
            rr = rr.reshape(1, ny)
        elif rr.shape[1] != ny:
            raise ControlError(f"Referans önizlemesi {ny} sütunlu olmalı.")
        if not np.all(np.isfinite(rr)):
            raise ControlError("Referans sonlu olmalı.")
        rr = rr / d.Yscale[None, :] - d.yoff[None, :]
    if rr.shape[0] >= p:
        rr = rr[:p]
    else:
        rr = np.vstack([rr, np.repeat(rr[-1:], p - rr.shape[0], axis=0)])
    return rr.ravel()


def _md_seq(v, d: MPCData, p):
    """Ölçülen bozucu (boyutsuz, ofsetsiz, sonuna 1 eklenmiş) (p+1)·nv dizisi."""
    nmd = d.nmd
    if v is None or nmd == 0:
        if v is not None and np.size(v):
            raise ControlError("Bitkide ölçülen bozucu (MD) girişi yok.")
        vv = np.zeros((1, nmd))
    else:
        vv = np.asarray(v, dtype=float)
        if vv.ndim <= 1:
            vv = np.atleast_1d(vv)
            if vv.size != nmd:
                raise ControlError(f"Ölçülen bozucu {nmd} elemanlı olmalı.")
            vv = vv.reshape(1, nmd)
        elif vv.shape[1] != nmd:
            raise ControlError(f"Ölçülen bozucu önizlemesi {nmd} sütunlu olmalı.")
        if not np.all(np.isfinite(vv)):
            raise ControlError("Ölçülen bozucu sonlu olmalı.")
        vv = vv / d.Uscale[d.mdindex][None, :] - d.mdoff[None, :]
    if vv.shape[0] >= p + 1:
        vv = vv[:p + 1]
    else:
        vv = np.vstack([vv, np.repeat(vv[-1:], p + 1 - vv.shape[0], axis=0)])
    return np.hstack([vv, np.ones((p + 1, 1))]).ravel()


def _solve(d: MPCData, xQP, old_u, rseq, vseq, iA, ws, isQP=None):
    """MPC QP'si (mpc_solveQP): (zopt, f, status, iA)."""
    isQP = d.isQP if isQP is None else isQP
    nfree = d.nfree
    degrees = nfree + (1 if isQP else 0)
    f = np.zeros(degrees)
    f[:nfree] = d.Kx.T @ xQP + d.Kr.T @ rseq + d.Ku1.T @ old_u + d.Kv.T @ vseq + d.Kut.T @ d.utarget
    if not isQP or d.max_iter == 0:
        return -d.Hinv[:degrees, :degrees] @ f, f, 1, iA
    bc = -(d.Mlim + d.Mx @ xQP + d.Mu1 @ old_u + d.Mv @ vseq)
    res = qp_solve_ge(d.H, f, d.Ac, bc, iA0=iA if d.use_warm_start else None, max_iter=d.max_iter, tol=d.tol,
                      Linv=d.Linv, workspace=ws)
    z = res.x
    status = res.status
    if status < 0 or (status == 0 and not d.use_suboptimal):
        z = np.zeros(degrees)
    return z, f, status, res.active


def _clip(u, u_prev, d: MPCData):
    """MATLAB mpc_clipping (alt-en iyi çözüm / 0 yineleme): ilk adım MV ve oran sınırlarına kırp."""
    nmv = d.nmv
    p, ny = d.p, d.ny
    nylim = 2 * p * ny
    pnu = p * nmv
    nfree = d.nfree
    umax = np.full(nmv, np.inf)
    umin = -umax.copy()
    dumax = umax.copy()
    dumin = umin.copy()
    for i, ix in enumerate(d.Mrows):
        if nylim <= ix < nylim + nmv:
            umax[ix - nylim] = d.Mlim[i] + d.mvoff[ix - nylim]
        elif nylim + pnu <= ix < nylim + pnu + nmv:
            umin[ix - nylim - pnu] = -d.Mlim[i] + d.mvoff[ix - nylim - pnu]
        elif nylim + 2 * pnu <= ix < nylim + 2 * pnu + nmv:
            dumax[ix - nylim - 2 * pnu] = d.Mlim[i]
        elif nylim + 2 * pnu + nfree <= ix < nylim + 2 * pnu + nfree + nmv:
            dumin[ix - nylim - 2 * pnu - nfree] = -d.Mlim[i]
    out = u.copy()
    for i in range(nmv):
        du = out[i] - u_prev[i]
        if du > dumax[i]:
            out[i] = u_prev[i] + dumax[i]
        elif du < dumin[i]:
            out[i] = u_prev[i] + dumin[i]
        out[i] = min(max(out[i], umin[i]), umax[i])
    return out


def mpcmove(obj: MPC, state: MPCState, ym=None, r=None, v=None):
    """MATLAB ``[u, info] = mpcmove(mpcobj, state, ym, r, v)``.

    ``ym`` (nmy,) ölçülen çıkışlar; ``r`` (ny,) ya da (n, ny) referans önizlemesi (satır i → y(k+i)); ``v`` (nmd,)
    ya da (n, nmd) ölçülen bozucu önizlemesi (satır 0 → v(k)). Mühendislik birimleri. ``state`` yerinde güncellenir
    (x(k+1|k), LastMove, iA). ``info``: Uopt ((p+1)×nmv), Yopt ((p+1)×ny), Xopt ((p+1)×nxQP), Topt, Slack,
    Iterations, QPCode ('feasible' / 'infeasible' / 'unreliable'), Cost (J, boyutsuz)."""
    d = obj.data
    p, nmv, nv, nxp, nxdist = d.p, d.nmv, d.nv, d.nxp, d.nxdist
    MVscale = d.Uscale[d.mvindex]
    if ym is None:
        y = np.zeros(d.nmy)
    else:
        y = _finite_vec(ym, d.nmy, "Ölçülen çıkış (ym)") / d.Yscale[d.myindex] - d.myoff
    rseq = _ref_seq(r, d, p)
    vseq = _md_seq(v, d, p)
    xp = _finite_vec(state.Plant, nxp, "Durum (Plant)") - d.xpoff
    xd = _finite_vec(state.Disturbance, nxdist, "Durum (Disturbance)") if nxdist else np.zeros(0)
    xn = _finite_vec(state.Noise, d.nxnoise, "Durum (Noise)") if d.nxnoise else np.zeros(0)
    uold = _finite_vec(state.LastMove, nmv, "Durum (LastMove)") / MVscale - d.mvoff
    xk = np.concatenate([xp, xd, xn])
    iA = state.iA
    if iA is None or np.size(iA) != d.Ac.shape[0] or not d.use_warm_start:
        iA = np.zeros(d.Ac.shape[0], dtype=bool)
    v0 = vseq[:nv]
    e = y - d.Cm @ xk - d.Dvm @ v0
    x = xk + d.M @ e
    xQP = x[:d.nxQP]
    z, f, status, iAout = _solve(d, xQP, uold, rseq, vseq, iA, obj._workspace(d))
    slack = float(z[-1]) if d.isQP else 0.0
    u = uold + z[:nmv] + d.mvoff
    suboptimal = status == 0 and d.use_suboptimal
    if d.isQP and (suboptimal or d.max_iter == 0):
        u = _clip(u, uold + d.mvoff, d)
    if d.isQP:
        how = "infeasible" if status <= -1 else ("unreliable" if status == 0 else "feasible")
    else:
        how, status = "feasible", 1
    cost = 0.0
    if status > 0 or suboptimal:
        aux2 = d.Sx @ xQP + d.Su1 @ uold + d.Hv @ vseq - rseq
        aux3 = d.I1 @ uold - d.utarget
        cost = float(aux3 @ (d.Wu * aux3) + aux2 @ (d.Wy * aux2) + z @ (d.H @ z + 2.0 * f))
    useq = np.zeros((p + 1, nmv))
    useq[:p] = (d.I2Jm @ z[:d.nfree] + d.I1 @ (uold + d.mvoff)).reshape(p, nmv)
    useq[p] = useq[p - 1]
    yopt, xopt = _sequence(d, xQP, useq, vseq)
    info = {"Uopt": useq * MVscale[None, :], "Yopt": yopt * d.Yscale[None, :], "Xopt": xopt,
            "Topt": np.arange(p + 1) * d.Ts, "Slack": slack, "Iterations": int(status), "QPCode": how,
            "Cost": cost}
    xnext = d.A @ xk + d.Bu @ (u - d.mvoff) + d.Bv @ v0 + d.L @ e
    xnext = xnext + d.xoff
    state.Plant = xnext[:nxp]
    state.Disturbance = xnext[nxp:nxp + nxdist]
    state.Noise = xnext[nxp + nxdist:]
    state.LastMove = u * MVscale
    state.iA = np.asarray(iAout, dtype=bool) if d.isQP else np.zeros(0, dtype=bool)
    return u * MVscale, info


def _sequence(d: MPCData, xQP, useq, vseq):
    """MATLAB mpc_computeSequence: tahmin edilen y ve x dizileri (boyutsuz, ofset dahil)."""
    p, nv, nxQP = d.p, d.nv, d.nxQP
    A, Bu, Bv = d.A[:nxQP, :nxQP], d.Bu[:nxQP], d.Bv[:nxQP]
    C, Dv = d.C[:, :nxQP], d.Dv
    ud = useq[:p] - d.mvoff[None, :]
    yopt = np.zeros((p + 1, d.ny))
    xopt = np.zeros((p + 1, nxQP))
    x = xQP.copy()
    for k in range(p + 1):
        xopt[k] = x
        vk = vseq[k * nv:(k + 1) * nv]
        yopt[k] = C @ x + Dv @ vk
        if k < p:
            x = A @ x + Bu @ ud[k] + Bv @ vk
    return yopt + d.yoff[None, :], xopt + d.xoff[None, :nxQP]


# ---------------------------------------------------------------------------------------------------------------
# sim (MATLAB mpcloop_engine)
# ---------------------------------------------------------------------------------------------------------------
def _sim_plant(obj: MPC, d: MPCData, model):
    """Benzetim bitkisi (model uyumsuzluğu): MATLAB mpcsimoptchk."""
    if model is None:
        return dict(Ap=d.Ap, Bup=d.Bup, Bvp=d.Bvp, Bdp=d.Bdp, Cp=d.Cp, Dvp=d.Dvp, Ddp=d.Ddp, xpoff=d.xpoff,
                    ypoff=d.yoff, upoff=d.mvoff, vpoff=d.mdoff, ndp=d.nud, dscale=d.Uscale[d.udindex])
    if isinstance(model, dict) and not isinstance(model, MPCState):
        sm = {str(k).lower(): v for k, v in model.items()}
        splant = sm.get("plant") if sm.get("plant") is not None else obj.Model.Plant
        snom = sm.get("nominal")
    else:
        splant, snom = model, None
    A, B, C, D, Tsp = _ss_data(splant, "Benzetim bitkisi")
    if Tsp is None:
        raise ControlError("Benzetim bitkisi dinamik bir sistem olmalı.")
    ny, nu = D.shape
    sig = dict(getattr(splant, "mpc_signals", None) or {})
    tmp = object.__new__(MPC)
    tmp._signals = {k: sorted(int(i) for i in np.atleast_1d(v)) for k, v in sig.items()}
    smv, smd, sud, smo, suo = tmp._indices(nu, ny)
    if smv.size != d.nmv or ny != d.ny or smo.size != d.nmy or smd.size != d.nmd:
        raise ControlError("Benzetim bitkisinin MV / MD / çıkış / ölçülen çıkış sayıları denetleyici modeliyle "
                           "aynı olmalı.")
    if not (np.array_equal(smo, d.myindex) and np.array_equal(smv, d.mvindex) and np.array_equal(smd, d.mdindex)):
        raise ControlError("Benzetim bitkisinin sinyal türleri (MV, MD, MO indisleri) denetleyici modeliyle aynı olmalı.")
    Ac_, _, _, Dc_, _ = _ss_data(obj.Model.Plant)
    cX, cDX, cU, cY = _nominal_of(obj.Model.Nominal, Ac_.shape[0], Dc_.shape[1], Dc_.shape[0])
    cUs = np.zeros(nu)                       # denetleyicinin nominal MV / MD'si benzetim bitkisinin girişlerine
    cUs[smv] = cU[d.mvindex]
    cUs[smd] = cU[d.mdindex]
    nom = {} if snom is None else {str(k).upper(): v for k, v in dict(snom).items()}
    inherit_state = "X" not in nom or "DX" not in nom
    X = nom.get("X", cX)
    DX = nom.get("DX", cDX)
    U = nom.get("U", cUs)
    Y = nom.get("Y", cY)
    if inherit_state and cX.size != A.shape[0]:
        X, DX = np.zeros(A.shape[0]), np.zeros(A.shape[0])
    Us1 = np.ones(nu)
    Us1[smv] = d.Uscale[d.mvindex]
    Us1[smd] = d.Uscale[d.mdindex]
    snd, nud = sud.size, d.nud
    if snd == nud:
        Us1[sud] = d.Uscale[d.udindex]
    elif snd < nud:
        Us1[sud] = d.Uscale[d.udindex[:snd]]
    else:
        Us1[sud[:nud]] = d.Uscale[d.udindex]
    Ap, Bp, Cp, Dp, xoff, _, uoff, yoff = _prep_plant(A, B, C, D, Tsp, d.Ts, _vec(X, A.shape[0], "Nominal.X"),
                                                      _vec(DX, A.shape[0], "Nominal.DX"), _vec(U, nu, "Nominal.U"),
                                                      _vec(Y, ny, "Nominal.Y"), Us1, d.Yscale)
    mdDX = list(smd) + [nu]
    return dict(Ap=Ap, Bup=Bp[:, smv], Bvp=Bp[:, mdDX], Bdp=Bp[:, sud], Cp=Cp, Dvp=Dp[:, mdDX], Ddp=Dp[:, sud],
                xpoff=xoff, ypoff=yoff, upoff=uoff[smv], vpoff=uoff[smd], ndp=snd, dscale=Us1[sud])


def _signal(sig, ncol, scale, what):
    if sig is None:
        return None
    a = np.asarray(sig, dtype=float)
    if a.ndim == 0 or (a.ndim == 1 and ncol == 1):
        a = a.reshape(-1, 1)
    elif a.ndim == 1:
        if a.size != ncol:                   # MATLAB: size(r, 2) ~= ny → hata (skaler yayılmaz)
            raise ControlError(f"{what} {ncol} sütunlu olmalı (her satır bir zaman adımı); {a.size} elemanlı "
                               f"vektör verildi.")
        a = a.reshape(1, ncol)
    if a.ndim != 2 or a.shape[1] != ncol:
        raise ControlError(f"{what} {ncol} sütunlu olmalı (her satır bir zaman adımı); boyut {a.shape}.")
    if not np.all(np.isfinite(a)):
        raise ControlError(f"{what} sonlu olmalı.")
    return a / np.asarray(scale, dtype=float)[None, :]


def sim(obj: MPC, T=None, r=None, v=None, options=None) -> SimResult:
    """MATLAB ``[y, t, u, xp, xmpc, SimOptions, status] = sim(mpcobj, T, r, v, SimOptions)``.

    ``r`` (ny,) ya da (n, ny); ``v`` (nmd,) ya da (n, nmd) (eksik satırlar sonuncuyla doldurulur). ``v`` yerine
    ``options`` (:class:`mpcsimopt` ya da sözlük) verilebilir (MATLAB gibi). Varsayılan: önizleme kapalı, kısıtlar
    açık, bitki = denetleyici modeli, başlangıç nominal. Dönüş :class:`SimResult` (y (T×ny), t, u (T×nmv),
    xp (T×nxp), xmpc (sözlük: Plant, Disturbance, Noise, LastMove dizileri), status (T,))."""
    if isinstance(v, (mpcsimopt, dict)) and options is None and not isinstance(v, np.ndarray):
        v, options = None, v
    opt = mpcsimopt() if options is None else (options if isinstance(options, mpcsimopt) else mpcsimopt(options))
    d = obj.data
    p, nmv, ny, nmy = d.p, d.nmv, d.ny, d.nmy
    MVscale = d.Uscale[d.mvindex]
    onoff = {"on": True, "off": False, True: True, False: False, None: None}

    def flag(val, default):
        if val is None:
            return default
        if isinstance(val, str):
            val = val.lower()
        if val not in onoff:
            raise ControlError("Benzetim seçeneği 'on' ya da 'off' olmalı.")
        return onoff[val]

    ref_prev = flag(opt.RefLookAhead, False)
    md_prev = flag(opt.MDLookAhead, False)
    use_con = flag(opt.Constraints, True)
    openloop = flag(opt.OpenLoop, False)
    sp = _sim_plant(obj, d, opt.Model)
    # sinyaller (boyutsuz)
    if r is None:
        rr = d.yoff.reshape(1, ny)
    else:
        rr = _signal(r, ny, d.Yscale, "Referans")
    if v is None or d.nmd == 0:
        vv = d.mdoff.reshape(1, d.nmd)
    else:
        vv = _signal(v, d.nmd, d.Uscale[d.mdindex], "Ölçülen bozucu")
    dd = _signal(opt.UnmeasuredDisturbance, sp["ndp"], sp["dscale"], "Ölçülmeyen bozucu")
    yn = _signal(opt.OutputNoise, nmy, d.Yscale[d.myindex], "Çıkış gürültüsü")
    un = _signal(opt.InputNoise, nmv, MVscale, "Giriş gürültüsü")
    mvs = _signal(opt.MVSignal, nmv, MVscale, "MV sinyali")
    if T is None:
        T = max([a.shape[0] for a in (rr, vv, dd, yn, un, mvs) if a is not None] + [p, 10])
    if not (np.isscalar(T) and np.isfinite(T) and T >= 1):
        raise ControlError("Benzetim adım sayısı T ≥ 1 olmalı.")
    T = int(round(T))
    rr = rr - d.yoff[None, :]
    vv = vv - d.mdoff[None, :]

    def pad(a, n):
        if a is None:
            return np.zeros((T, n))
        if a.shape[0] < T:
            a = np.vstack([a, np.repeat(a[-1:], T - a.shape[0], axis=0)])
        return a

    rr, vv = pad(rr, ny), pad(vv, d.nmd)
    dd, yn, un, mvs = pad(dd, sp["ndp"]), pad(yn, nmy), pad(un, nmv), pad(mvs, nmv)
    # başlangıç
    cs = opt.ControllerInitialState
    if cs is None:
        cs = mpcstate(obj)
    xmpc = np.concatenate([_vec(cs.Plant, d.nxp, "Plant durumu") - d.xpoff,
                           np.asarray(cs.Disturbance, dtype=float).ravel(), np.asarray(cs.Noise, dtype=float).ravel()])
    if xmpc.size != d.nx:
        raise ControlError("Denetleyici başlangıç durumunun boyutu uyuşmuyor.")
    old_u = _vec(cs.LastMove, nmv, "LastMove") / MVscale - d.mvoff
    nxp_s = sp["Ap"].shape[0]
    x0p = sp["xpoff"] if opt.PlantInitialState is None else _vec(opt.PlantInitialState, nxp_s, "Bitki başlangıç durumu")
    xp = x0p - sp["xpoff"]
    dvoff = sp["vpoff"] - d.mdoff
    dyoff = sp["ypoff"] - d.yoff
    isQP = d.isQP and use_con
    ws = obj._workspace(d)
    iA = np.zeros(d.Ac.shape[0], dtype=bool)
    YY = np.zeros((T, ny))
    UU = np.zeros((T, nmv))
    XP = np.zeros((T, nxp_s))
    XM = np.zeros((T, d.nx))
    ST = np.zeros(T)
    LM = np.zeros((T, nmv))
    one = np.ones(1)

    def rget(sig, k):
        return sig[min(k, sig.shape[0] - 1)]

    e = np.zeros(nmy)
    for t in range(T):
        XP[t] = xp + sp["xpoff"]
        XM[t] = xmpc + d.xoff
        LM[t] = (old_u + d.mvoff) * MVscale
        vt = rget(vv, t)
        dt = rget(dd, t)
        vp = vt + dvoff
        vfull = np.concatenate([vt, one])
        y = sp["Cp"] @ xp + sp["Dvp"] @ np.concatenate([vp, one]) + sp["Ddp"] @ dt
        YY[t] = y + sp["ypoff"]
        if openloop:
            u = rget(mvs, t)
            status = 1
        else:
            ymeas = y[d.myindex] + dyoff[d.myindex] + rget(yn, t)
            e = ymeas - d.Cm @ xmpc - d.Dvm @ vfull
            xkk = xmpc + d.M @ e
            xQP = xkk[:d.nxQP]
            if ref_prev:
                rseq = np.concatenate([rget(rr, t + i) for i in range(1, p + 1)])
            else:
                rseq = np.tile(rget(rr, t), p)
            if md_prev:
                vseq = np.concatenate([np.concatenate([rget(vv, t + i), one]) for i in range(p + 1)])
            else:
                vseq = np.tile(vfull, p + 1)
            if not d.use_warm_start:
                iA[:] = False
            z, _, status, iA_new = _solve(d, xQP, old_u, rseq, vseq, iA, ws, isQP=isQP)
            if isQP:
                iA = np.asarray(iA_new, dtype=bool)
            u = old_u + z[:nmv] + d.mvoff
            if isQP and ((status == 0 and d.use_suboptimal) or d.max_iter == 0):
                u = _clip(u, old_u + d.mvoff, d)
            old_u = u - d.mvoff
        up = u + rget(un, t)
        UU[t] = up
        ST[t] = status
        xmpc = d.A @ xmpc + d.Bu @ (u - d.mvoff) + d.Bv @ vfull + d.L @ e
        xp = sp["Ap"] @ xp + sp["Bup"] @ (up - sp["upoff"]) + sp["Bvp"] @ np.concatenate([vp, one]) + \
            sp["Bdp"] @ dt
    y = YY * d.Yscale[None, :]
    u = UU * MVscale[None, :]
    t = np.arange(T) * d.Ts
    xm = None if openloop else {"Plant": XM[:, :d.nxp], "Disturbance": XM[:, d.nxp:d.nxQP],
                                "Noise": XM[:, d.nxQP:], "LastMove": LM}
    return SimResult(y=y, t=t, u=u, xp=XP, xmpc=xm, status=ST)


# ---------------------------------------------------------------------------------------------------------------
# Analiz: getEstimator, ss (eşdeğer doğrusal kontrolcü), cloffset
# ---------------------------------------------------------------------------------------------------------------
def getEstimator(obj: MPC):
    """MATLAB ``[L, M, A, Cm, Bu, Bv, Dvm] = getEstimator(mpcobj)`` (mühendislik birimleri; durumlar: bitki,
    bozucu, gürültü). x(k|k) = x(k|k−1) + M(ym − ŷm), x(k+1|k) = A x(k|k−1) + Bu u + Bv [v; 1] + L(ym − ŷm)."""
    d = obj.data
    ys = d.Yscale[d.myindex]
    M = d.M / ys[None, :]
    L = d.L / ys[None, :]
    Cm = ys[:, None] * d.Cm
    Bu = d.Bu / d.Uscale[d.mvindex][None, :]
    Bv = d.Bv.copy()
    Dvm = d.Dvm.copy()
    if d.nmd:
        mds = d.Uscale[d.mdindex]
        Bv[:, :-1] = Bv[:, :-1] / mds[None, :]
        Dvm = ys[:, None] * Dvm
        Dvm[:, :-1] = Dvm[:, :-1] / mds[None, :]
    return L, M, d.A.copy(), Cm, Bu, Bv, Dvm


def _mpc_ss(obj: MPC, signals="", ref_preview=False, md_preview=False):
    """MATLAB ``ss(mpcobj, signals, ref_preview, md_preview)``: kısıtsız MPC'nin doğrusal eşdeğeri.

    Girişler: ym (ölçülen çıkışlar), [r (signals'ta 'r'), v ('v'), ofset 1 ('o'), MV hedefi ('t')]; çıkış u.
    Durumlar: x(k|k−1) (nx) ve u(k−1) (nmv). Sapma (nominal çıkarılmış) değişkenlerde."""
    d = obj.data
    signals = (signals or "").lower()
    if any(ch not in "rvot" for ch in signals):
        raise ControlError("ss sinyal seçeneği yalnız 'r', 'v', 'o', 't' harflerini içerebilir.")
    p, nmv, nx, nv, nxQP = d.p, d.nmv, d.nx, d.nv, d.nxQP
    nmoves = d.moves.size
    nfm = nmoves * nmv
    A, Cm, Bu, Bv0, Dvm0, L, M = d.A, d.Cm, d.Bu, d.Bv, d.Dvm, d.L, d.M
    Kmat = -np.hstack([np.eye(nmv), np.zeros((nmv, (nmoves - 1) * nmv))]) @ d.Hinv[:nfm, :nfm]
    KmatQP = Kmat @ d.Kx.T @ np.hstack([np.eye(nxQP), np.zeros((nxQP, nx - nxQP))])
    Fx = KmatQP @ (np.eye(nx) - M @ Cm)
    Dym = KmatQP @ M
    IIoff = np.arange(1, p + 2) * nv - 1
    IIv = np.setdiff1d(np.arange(nv * (p + 1)), IIoff)
    Ivaux = np.hstack([np.eye(nv - 1), np.zeros((nv - 1, p * (nv - 1)))])
    Dv = -KmatQP @ M @ Dvm0[:, :nv - 1] @ Ivaux + Kmat @ d.Kv[IIv, :].T
    Doff = -KmatQP @ M @ Dvm0[:, nv - 1] + Kmat @ d.Kv[IIoff, :].T @ np.ones(p + 1)
    Fu = Kmat @ d.Ku1.T
    Dut = Kmat @ d.Kut.T
    Dr = Kmat @ d.Kr.T
    C = np.hstack([Fx, Fu + np.eye(nmv)])
    AA = np.block([[A - L @ Cm + Bu @ Fx, Bu + Bu @ Fu], [C]])
    BB = np.vstack([L + Bu @ Dym, Dym])
    DD = Dym
    in_scale = list(d.Yscale[d.myindex])
    in_names = [obj.OV[i].Name for i in d.myindex]
    if "r" in signals:
        Br = np.vstack([Bu @ Dr, Dr])
        Drr = Dr
        if not ref_preview:
            aux = np.kron(np.ones((p, 1)), np.eye(d.ny))
            Br, Drr = Br @ aux, Drr @ aux
            in_names += [f"ref.{obj.OV[i].Name}" for i in range(d.ny)]
            in_scale += list(d.Yscale)
        else:
            in_names += [f"ref.{obj.OV[i].Name}({j + 1})" for j in range(p) for i in range(d.ny)]
            in_scale += list(np.tile(d.Yscale, p))
        BB, DD = np.hstack([BB, Br]), np.hstack([DD, Drr])
    if "v" in signals and d.nmd:
        Bvv = np.vstack([(Bv0[:, :nv - 1] - L @ Dvm0[:, :nv - 1]) @ Ivaux + Bu @ Dv, Dv])
        Dvv = Dv
        mds = d.Uscale[d.mdindex]
        if not md_preview:
            aux = np.kron(np.ones((p + 1, 1)), np.eye(nv - 1))
            Bvv, Dvv = Bvv @ aux, Dvv @ aux
            in_names += [obj.DV[i].Name for i in range(d.nmd)]
            in_scale += list(mds)
        else:
            in_names += [f"{obj.DV[i].Name}({j})" for j in range(p + 1) for i in range(d.nmd)]
            in_scale += list(np.tile(mds, p + 1))
        BB, DD = np.hstack([BB, Bvv]), np.hstack([DD, Dvv])
    if "o" in signals:
        Boff = np.concatenate([Bv0[:, nv - 1] + Bu @ Doff - L @ Dvm0[:, nv - 1], Doff])
        BB, DD = np.hstack([BB, Boff[:, None]]), np.hstack([DD, Doff[:, None]])
        in_names.append("1 [offset]")
        in_scale.append(1.0)
    if "t" in signals:
        But = np.vstack([Bu @ Dut, Dut])
        aux = np.kron(np.ones((p, 1)), np.eye(nmv))
        BB, DD = np.hstack([BB, But @ aux]), np.hstack([DD, Dut @ aux])
        in_names += [f"{obj.MV[i].Name}.target" for i in range(nmv)]
        in_scale += list(d.Uscale[d.mvindex])
    in_scale = np.asarray(in_scale, dtype=float)
    nxc = AA.shape[0]
    mvs = d.Uscale[d.mvindex]
    xs = np.concatenate([np.ones(nxc - nmv), mvs])
    AA = xs[:, None] * AA / xs[None, :]
    BB = xs[:, None] * BB / in_scale[None, :]
    C = mvs[:, None] * C / xs[None, :]
    DD = mvs[:, None] * DD / in_scale[None, :]
    st_names = tuple([f"xMPC{i + 1}" if nx > 1 else "xMPC" for i in range(nx)] +
                     [f"prev.{obj.MV[i].Name}" for i in range(nmv)])
    out = MPCLinearController(A=AA, B=BB, C=C, D=DD, Ts=d.Ts, input_names=tuple(in_names),
                              output_names=tuple(obj.MV[i].Name for i in range(nmv)), state_names=st_names)
    return _to_lti(out)


def _to_lti(c: MPCLinearController):
    """spt.control.lti hazırsa StateSpace'e çevir (tembel); değilse aynı öznitelikli veri sınıfı döner."""
    try:
        from spt.control import lti
    except Exception:  # noqa: BLE001
        return c
    try:
        return lti.StateSpace(c.A, c.B, c.C, c.D, c.Ts, state_names=list(c.state_names),
                              input_names=list(c.input_names), output_names=list(c.output_names))
    except Exception:  # noqa: BLE001
        try:
            return lti.StateSpace(c.A, c.B, c.C, c.D, c.Ts)
        except Exception:  # noqa: BLE001
            return c


def ss(*args, **kw):
    """``ss(mpcobj, signals, ref_preview, md_preview)`` → eşdeğer doğrusal kontrolcü (MATLAB). İlk argüman MPC
    değilse spt.control.lti.ss'e devreder."""
    if args and isinstance(args[0], MPC):
        return _mpc_ss(*args, **kw)
    from spt.control import lti
    return lti.ss(*args, **kw)


def cloffset(obj: MPC):
    """MATLAB ``cloffset``: kısıtsız kapalı çevrimde ölçülen çıkışlara eklenen birim basamak çıkış bozucusunun
    kararlı hâl etkisi (nmy×nmy; 0 = ofsetsiz izleme). Mühendislik birimlerinde tutarlı hesap (ölçek çarpanları 1
    iken MATLAB ile birebir)."""
    d = obj.data
    K = _mpc_ss(obj)
    Ac, Bc, Cc, Dc = (np.asarray(getattr(K, a), dtype=float) for a in ("A", "B", "C", "D"))
    Cm = d.Yscale[d.myindex][:, None] * d.Cm
    Bu = d.Bu / d.Uscale[d.mvindex][None, :]
    bigA = np.block([[d.A + Bu @ Dc @ Cm, Bu @ Cc], [Bc @ Cm, Ac]])
    bigB = np.vstack([Bu @ Dc, Bc])
    bigC = np.hstack([Cm, np.zeros((d.nmy, Ac.shape[0]))])
    return _dcgain_d(bigA, bigB, bigC, np.eye(d.nmy))


def setoutdist(obj: MPC, method="integrators", model=None):
    """MATLAB ``setoutdist(mpcobj, 'integrators' | 'model', sys)``."""
    obj.setoutdist(method, model)
