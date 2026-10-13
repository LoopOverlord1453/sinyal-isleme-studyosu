"""İHA modelleri için ortak araçlar: döndürmeler, kuaterniyonlar, sayısal Jacobian, benzetim, doğrusal model kabı,
motor / pervane yardımcıları, Dryden türbülansı ve algılayıcı modelleri.

Koordinat ve birim kuralları (docs/SPT-DEV-001_Gelistirme_Asamalari.docx, Aşama 4 "Koordinatlar (İHA)"):
  - Dünya çerçevesi NED: x kuzey, y doğu, z aşağı. Gövde çerçevesi FRD: x ileri, y sağ, z aşağı.
  - Euler açıları ZYX (yaw ψ, pitch θ, roll φ): R = Rz(ψ)·Ry(θ)·Rx(φ), v_NED = R·v_gövde (gövde → NED).
  - Kuaterniyon q = [w, x, y, z] (Hamilton), gövde → NED döndürmesi: R(q) aynı matristir; kinematik
    q̇ = ½·q ⊗ [0, ω_gövde]. spt/dsp/fusion.py ile aynı sıra ve formüller (orada dünya z yukarı; formüller
    çerçeveden bağımsızdır).
  - Açılar radyan, açısal hız rad/s, uzunluk m, kuvvet N, moment N·m, kütle kg. Yerçekimi NED'de [0, 0, +g].
  - Durağan ve düz duran bir ivmeölçer (özgül kuvvet f = a − g) gövdede [0, 0, −g] okur (z aşağı olduğundan).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from spt.control.errors import ControlError

GRAVITY = 9.80665
E3 = np.array([0.0, 0.0, 1.0])


# ============================================================================ temel döndürmeler
def skew(v) -> np.ndarray:
    """[v×]: skew(v) @ a = v × a."""
    v = np.asarray(v, dtype=np.float64)
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def rot_x(a: float) -> np.ndarray:
    """Etkin (aktif) x döndürmesi."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotmat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """R = Rz(ψ)·Ry(θ)·Rx(φ): gövde → NED. (MATLAB: eul2rotm([ψ θ φ], 'ZYX') = R; angle2dcm(ψ, θ, φ) = Rᵀ.)"""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def euler_from_rotmat(R: np.ndarray) -> tuple[float, float, float]:
    """(roll, pitch, yaw); pitch ∈ [−π/2, π/2]."""
    R = np.asarray(R, dtype=np.float64)
    roll = math.atan2(R[2, 1], R[2, 2])
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def euler_rate_matrix(roll: float, pitch: float) -> np.ndarray:
    """W(φ, θ): [φ̇, θ̇, ψ̇] = W·[p, q, r]. θ = ±90°'de tekil."""
    sr, cr = math.sin(roll), math.cos(roll)
    cp, tp = math.cos(pitch), math.tan(pitch)
    return np.array([[1.0, sr * tp, cr * tp], [0.0, cr, -sr], [0.0, sr / cp, cr / cp]])


def wrap_angle(a):
    """(−π, π] aralığına sarar."""
    a = np.asarray(a, dtype=np.float64)
    out = np.mod(a + np.pi, 2 * np.pi) - np.pi
    out = np.where(out == -np.pi, np.pi, out)
    return float(out) if out.ndim == 0 else out


# ============================================================================ kuaterniyonlar ([w, x, y, z])
def quat_mul(a, b) -> np.ndarray:
    """Hamilton çarpımı a ⊗ b."""
    aw, ax, ay, az = (float(v) for v in a)
    bw, bx, by, bz = (float(v) for v in b)
    return np.array([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw])


def quat_conj(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = math.sqrt(float(q @ q))
    if n == 0.0 or not math.isfinite(n):
        raise ControlError("Kuaterniyon sıfır ya da sonlu değil; yönelim tanımsız.")
    return q / n


def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX Euler → birim kuaterniyon (w ≥ 0 olması garanti değil; fusion.quat_from_euler ile aynı formül)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([cr * cp * cy + sr * sp * sy,
                     sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy,
                     cr * cp * sy - sr * sp * cy])


def euler_from_quat(q) -> tuple[float, float, float]:
    """Kuaterniyon (normlanır) → (roll, pitch, yaw) radyan, ZYX."""
    w, x, y, z = quat_normalize(q)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def euler_from_quat_array(q: np.ndarray) -> np.ndarray:
    """(n, 4) kuaterniyon dizisi → (n, 3) [roll, pitch, yaw]."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack((np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
                     np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0)),
                     np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))), axis=-1)


def rotmat_from_quat(q) -> np.ndarray:
    """R(q̂), q̂ = q/|q|: gövde → NED. Homojen biçim R = [(w²−|v|²)I + 2vvᵀ + 2w[v×]] / |q|² — normsuz q için de
    tam bir döndürme matrisidir (dinamik normdan bağımsız kalır)."""
    q = np.asarray(q, dtype=np.float64)
    w, v = q[0], q[1:]
    n2 = float(q @ q)
    return ((w * w - v @ v) * np.eye(3) + 2.0 * np.outer(v, v) + 2.0 * w * skew(v)) / n2


def quat_omega_matrix(omega) -> np.ndarray:
    """Ω(ω): q ⊗ [0, ω] = Ω(ω)·q."""
    p, q, r = (float(v) for v in omega)
    return np.array([[0.0, -p, -q, -r], [p, 0.0, r, -q], [q, -r, 0.0, p], [r, q, -p, 0.0]])


def quat_xi_matrix(q) -> np.ndarray:
    """Ξ(q): q ⊗ [0, ω] = Ξ(q)·ω."""
    w, x, y, z = (float(v) for v in q)
    return np.array([[-x, -y, -z], [w, -z, y], [z, w, -x], [-y, x, w]])


def d_rotmat_quat_times(q, a) -> np.ndarray:
    """∂(R(q̂)·a)/∂q (3×4), R homojen biçimiyle (rotmat_from_quat). a sabit vektör."""
    q = np.asarray(q, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    w, v = q[0], q[1:]
    n2 = float(q @ q)
    ra = (w * w - v @ v) * a + 2.0 * (v @ a) * v + 2.0 * w * np.cross(v, a)   # |q|²·R(q̂)a
    d = np.empty((3, 4))
    d[:, 0] = 2.0 * w * a + 2.0 * np.cross(v, a)
    d[:, 1:] = -2.0 * np.outer(a, v) + 2.0 * np.outer(v, a) + 2.0 * (v @ a) * np.eye(3) - 2.0 * w * skew(a)
    return d / n2 - 2.0 * np.outer(ra, q) / (n2 * n2)


def d_rotmat_T_quat_times(q, a) -> np.ndarray:
    """∂(R(q̂)ᵀ·a)/∂q (3×4). R(q)ᵀ = R(q*) olduğundan zincir kuralı."""
    qc = quat_conj(q)
    return d_rotmat_quat_times(qc, a) * np.array([1.0, -1.0, -1.0, -1.0])


def d_euler_d_quat(q) -> np.ndarray:
    """∂[φ, θ, ψ]/∂q (3×4), Euler açıları q̂ = q/|q| üzerinden (norm yönünde türev sıfır)."""
    q = np.asarray(q, dtype=np.float64)
    nq = math.sqrt(float(q @ q))
    w, x, y, z = q / nq
    # φ = atan2(N1, D1), θ = asin(S), ψ = atan2(N3, D3) — birim q̂'ya göre türevler
    N1, D1 = 2 * (w * x + y * z), 1 - 2 * (x * x + y * y)
    S = 2 * (w * y - z * x)
    N3, D3 = 2 * (w * z + x * y), 1 - 2 * (y * y + z * z)
    dN1 = np.array([2 * x, 2 * w, 2 * z, 2 * y])
    dD1 = np.array([0.0, -4 * x, -4 * y, 0.0])
    dS = np.array([2 * y, -2 * z, 2 * w, -2 * x])
    dN3 = np.array([2 * z, 2 * y, 2 * x, 2 * w])
    dD3 = np.array([0.0, 0.0, -4 * y, -4 * z])
    g = np.empty((3, 4))
    g[0] = (D1 * dN1 - N1 * dD1) / (N1 * N1 + D1 * D1)
    g[1] = dS / math.sqrt(max(1e-300, 1.0 - S * S))
    g[2] = (D3 * dN3 - N3 * dD3) / (N3 * N3 + D3 * D3)
    qh = np.array([w, x, y, z])
    return g @ ((np.eye(4) - np.outer(qh, qh)) / nq)


# ============================================================================ sayısal Jacobian
def numjac(f: Callable[[np.ndarray], np.ndarray], x0, *, rel_step: float = 1e-3,
           abs_step: float = 1e-3) -> np.ndarray:
    """∂f/∂x merkezi fark + Richardson ekstrapolasyonu (4. mertebe; tipik bağıl doğruluk ~1e-10).

    Adım h_i = max(abs_step, rel_step·|x_i|); D(h) ve D(h/2) birleşimi: (4·D(h/2) − D(h)) / 3."""
    x0 = np.asarray(x0, dtype=np.float64).copy()
    f0 = np.asarray(f(x0), dtype=np.float64)
    J = np.empty((f0.size, x0.size))
    for i in range(x0.size):
        h = max(abs_step, rel_step * abs(x0[i]))
        cols = []
        for hh in (h, h / 2):
            xp, xm = x0.copy(), x0.copy()
            xp[i] += hh
            xm[i] -= hh
            cols.append((np.asarray(f(xp)) - np.asarray(f(xm))) / (2 * hh))
        J[:, i] = (4 * cols[1] - cols[0]) / 3
    return J


# ============================================================================ doğrusal model kabı
@dataclass
class LinearModel:
    """Çalışma noktasında doğrusal model: δẋ = A δx + B δu, δy = C δx + D δu (sürekli zaman).

    `x0`, `u0`, `y0`: çalışma noktası (bu modelin koordinatlarında); `xdot0`: f(x0, u0) (dengede ≈ 0).
    `to_ss()` isteğe bağlı olarak spt.control.lti.ss nesnesine çevirir (tembel içe aktarma)."""
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    state_names: tuple[str, ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    x0: np.ndarray | None = None
    u0: np.ndarray | None = None
    y0: np.ndarray | None = None
    xdot0: np.ndarray | None = None
    info: dict = field(default_factory=dict)

    @property
    def nx(self) -> int:
        return self.A.shape[0]

    def matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.A, self.B, self.C, self.D

    def subsystem(self, states: Sequence[str] | None = None, inputs: Sequence[str] | None = None,
                  outputs: Sequence[str] | None = None) -> "LinearModel":
        """Adlarla alt blok (seçilmeyen durumlarla bağlaşım atılır — boylamsal / yanal ayrıştırma gibi)."""
        si = _indices(self.state_names, states)
        ii = _indices(self.input_names, inputs)
        if outputs is None:
            sel = [self.state_names[k] for k in si]      # çıkışlar durumların kendisiyse onları izle
            outputs = sel if set(sel) <= set(self.output_names) else self.output_names
        oi = _indices(self.output_names, outputs)
        pick = lambda v, idx: None if v is None else np.asarray(v)[idx]  # noqa: E731
        return LinearModel(self.A[np.ix_(si, si)], self.B[np.ix_(si, ii)], self.C[np.ix_(oi, si)],
                           self.D[np.ix_(oi, ii)], tuple(self.state_names[k] for k in si),
                           tuple(self.input_names[k] for k in ii), tuple(self.output_names[k] for k in oi),
                           pick(self.x0, si), pick(self.u0, ii), pick(self.y0, oi), pick(self.xdot0, si),
                           dict(self.info))

    def freqresp(self, s: complex) -> np.ndarray:
        """G(s) = C (sI − A)⁻¹ B + D (tek karmaşık frekans; doğrulama kolaylığı)."""
        n = self.A.shape[0]
        return self.C @ np.linalg.solve(s * np.eye(n) - self.A, self.B) + self.D

    def to_ss(self):
        """spt.control.lti StateSpace nesnesi (LTI çekirdeği kuruluysa)."""
        try:
            from spt.control.lti import ss  # tembel: LTI çekirdeği ayrı modül
        except ImportError as exc:  # pragma: no cover - çekirdek henüz yoksa
            raise ControlError("Durum uzayı nesnesi için LTI çekirdeği bulunamadı.") from exc
        return ss(self.A, self.B, self.C, self.D, state_names=self.state_names, input_names=self.input_names,
                  output_names=self.output_names)


def _indices(names: Sequence[str], pick: Sequence[str] | None) -> list[int]:
    if pick is None:
        return list(range(len(names)))
    out = []
    for p in pick:
        if p not in names:
            raise ControlError(f"'{p}' adlı sinyal modelde yok (mevcut: {', '.join(names)}).")
        out.append(list(names).index(p))
    return out


# ============================================================================ benzetim
@dataclass
class SimResult:
    """Benzetim çıktısı: t (nt,), x (nt, nx), u (nt, nu); ada göre erişim `res["pd"]`."""
    t: np.ndarray
    x: np.ndarray
    u: np.ndarray
    state_names: tuple[str, ...]
    input_names: tuple[str, ...]
    info: dict = field(default_factory=dict)

    def __getitem__(self, name: str) -> np.ndarray:
        if name in self.state_names:
            return self.x[:, self.state_names.index(name)]
        if name in self.input_names:
            return self.u[:, self.input_names.index(name)]
        raise KeyError(name)


def simulate(model, x0, u, t, *, method: str = "RK45", rtol: float = 1e-8, atol: float = 1e-10,
             max_step: float = np.inf, substeps: int = 1, hold: str = "stage") -> SimResult:
    """Doğrusal olmayan modeli benzetir.

    model : `derivatives(x, u, t)`, `nx`, `nu`, `state_names`, `input_names` (isteğe bağlı `project(x)`).
    u     : sabit giriş (nu,), `u(t, x)` çağrılabilir, ya da t ızgarasında (nt, nu) dizi — dizi her aralıkta sabit
            tutulur (sıfırıncı derece tutucu; aralıklar ayrı ayrı tümlenir, süreksizlik çözücüyü bozmaz).
    method: scipy `solve_ivp` yöntemi ("RK45" = Dormand–Prince / ode45, "DOP853", "Radau", "LSODA" …) ya da
            "rk4" (sabit adımlı klasik Runge–Kutta, her t aralığında `substeps` alt adım; her adım sonunda
            `model.project` — ör. kuaterniyon normlama — uygulanır).
    hold  : yalnız "rk4" + çağrılabilir u: "stage" (u her RK aşamasında yeniden hesaplanır, ODE ile tutarlı) ya da
            "zoh" (u adım başında bir kez hesaplanır — ayrık denetleyici benzetimi).
    """
    from scipy.integrate import solve_ivp

    t = np.asarray(t, dtype=np.float64).ravel()
    if t.size < 2 or np.any(np.diff(t) <= 0):
        raise ControlError("Zaman ızgarası en az iki noktalı ve kesin artan olmalı.")
    x0 = np.asarray(x0, dtype=np.float64).ravel()
    if x0.size != model.nx:
        raise ControlError(f"Başlangıç durumu {model.nx} elemanlı olmalı ({x0.size} verildi).")
    nu = model.nu
    if callable(u):
        u_fn = lambda tt, xx: np.asarray(u(tt, xx), dtype=np.float64).ravel()  # noqa: E731
        u_tab = None
    else:
        ua = np.asarray(u, dtype=np.float64)
        if ua.ndim == 1:
            if ua.size != nu:
                raise ControlError(f"Giriş {nu} elemanlı olmalı.")
            u_tab = np.tile(ua, (t.size, 1))
        elif ua.shape == (t.size, nu):
            u_tab = ua
        else:
            raise ControlError(f"Giriş dizisi ({t.size}, {nu}) biçiminde olmalı.")
        u_fn = None
    project = getattr(model, "project", None)
    X = np.empty((t.size, model.nx))
    U = np.empty((t.size, nu))
    X[0] = x0
    nfev = 0
    if method.lower() == "rk4":
        x = x0.copy()
        for k in range(t.size - 1):
            h = (t[k + 1] - t[k]) / substeps
            tk = t[k]
            uk_hold = u_tab[k] if u_tab is not None else u_fn(tk, x)
            if k == 0:
                U[0] = uk_hold
            for _ in range(substeps):
                if u_tab is not None or hold == "zoh":
                    ufun = lambda tt, xx: uk_hold  # noqa: E731
                else:
                    ufun = u_fn
                k1 = model.derivatives(x, ufun(tk, x), tk)
                k2 = model.derivatives(x + 0.5 * h * k1, ufun(tk + 0.5 * h, x + 0.5 * h * k1), tk + 0.5 * h)
                k3 = model.derivatives(x + 0.5 * h * k2, ufun(tk + 0.5 * h, x + 0.5 * h * k2), tk + 0.5 * h)
                k4 = model.derivatives(x + h * k3, ufun(tk + h, x + h * k3), tk + h)
                x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                nfev += 4
                if project is not None:
                    x = project(x)
                tk += h
                if hold == "zoh" and u_tab is None:
                    uk_hold = u_fn(tk, x)
            X[k + 1] = x
            U[k + 1] = u_tab[k + 1] if u_tab is not None else u_fn(t[k + 1], x)
        return SimResult(t, X, U, tuple(model.state_names), tuple(model.input_names), {"nfev": nfev})

    if u_tab is not None:
        x = x0.copy()
        U[:] = u_tab
        for k in range(t.size - 1):
            uk = u_tab[k]
            sol = solve_ivp(lambda tt, xx: model.derivatives(xx, uk, tt), (t[k], t[k + 1]), x, method=method,
                            rtol=rtol, atol=atol, max_step=max_step)
            if not sol.success:
                raise ControlError(f"Tümleme başarısız: {sol.message}")
            nfev += sol.nfev
            x = sol.y[:, -1]
            X[k + 1] = x
        return SimResult(t, X, U, tuple(model.state_names), tuple(model.input_names), {"nfev": nfev})

    sol = solve_ivp(lambda tt, xx: model.derivatives(xx, u_fn(tt, xx), tt), (t[0], t[-1]), x0, method=method,
                    t_eval=t, rtol=rtol, atol=atol, max_step=max_step)
    if not sol.success:
        raise ControlError(f"Tümleme başarısız: {sol.message}")
    X = sol.y.T.copy()
    for k in range(t.size):
        U[k] = u_fn(t[k], X[k])
    return SimResult(t, X, U, tuple(model.state_names), tuple(model.input_names), {"nfev": sol.nfev})


# ============================================================================ motor / pervane yardımcıları
def prop_coefficients(CT: float, CQ: float, diameter: float, rho: float = 1.225) -> tuple[float, float]:
    """Boyutsuz pervane katsayılarından (T = CT·ρ·n²·D⁴, Q = CQ·ρ·n²·D⁵, n devir/s) açısal hız katsayıları:
    T = kT·Ω², Q = kQ·Ω² (Ω rad/s) → kT = CT·ρ·D⁴/(4π²), kQ = CQ·ρ·D⁵/(4π²)."""
    c = rho / (4 * math.pi ** 2)
    return CT * c * diameter ** 4, CQ * c * diameter ** 5


def hover_speed(mass: float, kT: float, n_motors: int = 4, g: float = GRAVITY) -> float:
    """Simetrik askıda kalmada motor hızı Ω = √(m·g / (n·kT)) (rad/s)."""
    return math.sqrt(mass * g / (n_motors * kT))


def throttle_to_speed(u, cr: float, wb: float, omega_min: float = 0.0, omega_max: float = np.inf):
    """Normalize gaz (0–1) → kararlı motor hızı: Ω_ss = C_R·u + ϖ_b, [omega_min, omega_max] aralığına kırpılır
    (Quan 2017, "Introduction to Multicopter Design and Control", motor modeli)."""
    return np.clip(cr * np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0) + wb, omega_min, omega_max)


def speed_to_throttle(omega, cr: float, wb: float):
    """throttle_to_speed'in tersi (kırpmasız)."""
    return (np.asarray(omega, dtype=np.float64) - wb) / cr


# ============================================================================ Dryden türbülansı
DRYDEN_PRESETS = {
    # Beard & McLain, Small Unmanned Aircraft (2012), Böl. 4 Dryden tablosu — (σu, σv, σw) m/s, (Lu, Lv, Lw) m
    "low_light": ((1.06, 1.06, 0.7), (200.0, 200.0, 50.0)),
    "low_moderate": ((2.12, 2.12, 1.4), (200.0, 200.0, 50.0)),
    "medium_light": ((1.5, 1.5, 1.5), (533.0, 533.0, 533.0)),
    "medium_moderate": ((3.0, 3.0, 3.0), (533.0, 533.0, 533.0)),
}


class DrydenGust:
    """Dryden türbülansı (gövde ekseninde u_wg, v_wg, w_wg), Beard & McLain biçimi:

        H_u(s) = σu·√(2Va/Lu) / (s + Va/Lu)
        H_v(s) = σv·√(3Va/Lv)·(s + Va/(√3·Lv)) / (s + Va/Lv)²      (H_w aynı biçim, w indisleriyle)

    birim yoğunluklu sürekli beyaz gürültüyle (E[n(t)n(t+τ)] = δ(τ)) sürülür; bu ölçekle durağan varyans tam
    σ²'dir (analitik: ∫|H|² dω/2π = σ²). Ayrıklaştırma tam: x[k+1] = e^{A·dt}x[k] + w[k],
    w ~ N(0, ∫₀^dt e^{Aτ}BBᵀe^{Aᵀτ}dτ) (Van Loan). Başlangıç durumu durağan dağılımdan çekilir.
    Filtre sabit (nominal) Va ile kurulur — yaygın uygulama; Va değişimi süzgece yansıtılmaz."""

    def __init__(self, Va: float, sigma=(1.06, 1.06, 0.7), length=(200.0, 200.0, 50.0), *, preset: str | None = None,
                 seed: int | None = None):
        if preset is not None:
            if preset not in DRYDEN_PRESETS:
                raise ControlError(f"Bilinmeyen Dryden ön ayarı: {preset}")
            sigma, length = DRYDEN_PRESETS[preset]
        if Va <= 0:
            raise ControlError("Dryden süzgeci için hava hızı pozitif olmalı.")
        self.Va = float(Va)
        self.sigma = tuple(float(s) for s in sigma)
        self.length = tuple(float(v) for v in length)
        self.seed = seed
        self.filters = [self._first_order(self.sigma[0], self.length[0]),
                        self._second_order(self.sigma[1], self.length[1]),
                        self._second_order(self.sigma[2], self.length[2])]

    def _first_order(self, s, L):
        a = self.Va / L
        return np.array([[-a]]), np.array([[1.0]]), np.array([[s * math.sqrt(2 * self.Va / L)]])

    def _second_order(self, s, L):
        a = self.Va / L
        k = s * math.sqrt(3 * self.Va / L)
        # (s + a/√3)/(s + a)²: denetlenebilir kanonik biçim
        A = np.array([[0.0, 1.0], [-a * a, -2 * a]])
        B = np.array([[0.0], [1.0]])
        C = np.array([[k * a / math.sqrt(3), k]])
        return A, B, C

    def stationary_variance(self) -> np.ndarray:
        """Her bileşenin analitik durağan varyansı C·P∞·Cᵀ (P∞: A P + P Aᵀ + B Bᵀ = 0) — σ² olmalı."""
        from scipy.linalg import solve_continuous_lyapunov
        out = []
        for A, B, C in self.filters:
            P = solve_continuous_lyapunov(A, -B @ B.T)
            out.append((C @ P @ C.T).item())
        return np.array(out)

    def sample(self, t_end: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """(t (nt,), gust (nt, 3)) eşit aralıklı örnekler. Her bileşen ayrı (tohumdan türetilmiş) rastgele akış
        kullanır: aynı tohumla daha kısa bir örnekleme, uzun olanın ön ekidir."""
        from scipy.linalg import expm, solve_continuous_lyapunov
        rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(self.seed).spawn(3)]
        n = int(math.floor(t_end / dt + 1e-9)) + 1
        t = np.arange(n) * dt
        out = np.empty((n, 3))
        for j, (A, B, C) in enumerate(self.filters):
            rng = rngs[j]
            m = A.shape[0]
            M = np.zeros((2 * m, 2 * m))
            M[:m, :m] = -A
            M[:m, m:] = B @ B.T
            M[m:, m:] = A.T
            E = expm(M * dt)
            Ad = E[m:, m:].T
            Qd = Ad @ E[:m, m:]
            Qd = 0.5 * (Qd + Qd.T)
            Lq = np.linalg.cholesky(Qd + 1e-300 * np.eye(m)) if m > 1 else np.sqrt(np.maximum(Qd, 0.0))
            P = solve_continuous_lyapunov(A, -B @ B.T)
            x = np.linalg.cholesky(P) @ rng.standard_normal(m)
            for k in range(n):
                out[k, j] = (C @ x).item()
                x = Ad @ x + Lq @ rng.standard_normal(m)
        return t, out

    def as_function(self, t_end: float, dt: float = 0.01) -> Callable[[float], np.ndarray]:
        """Zamanın fonksiyonu olarak rüzgar sapması (doğrusal ara değerleme; t_end sonrası son değer)."""
        t, g = self.sample(t_end, dt)

        def gust(tt: float) -> np.ndarray:
            return np.array([np.interp(tt, t, g[:, i]) for i in range(3)])
        return gust


# ============================================================================ algılayıcı modelleri
def _uniform_dt(t: np.ndarray) -> float:
    t = np.asarray(t, dtype=np.float64)
    d = np.diff(t)
    if d.size == 0 or np.any(d <= 0) or np.ptp(d) > 1e-9 * max(1.0, d.mean()):
        raise ControlError("Algılayıcı modeli eşit aralıklı, artan bir zaman ızgarası ister.")
    return float(d.mean())


@dataclass
class ImuModel:
    """IMU: jiroskop ω_ölç = ω + b_g + η_g, ivmeölçer f_ölç = f + b_a + η_a (gövde, FRD).

    Gürültü yoğunlukları sürekli zamanda (rad/s/√Hz, m/s²/√Hz): örnek başı σ = yoğunluk/√dt. Sapma rastgele
    yürüyüşü (rad/s/√s, m/s²/√s): b[k+1] = b[k] + yürüyüş·√dt·n. Tohumlu (`seed`) — aynı tohum aynı gürültü."""
    gyro_noise_density: float = 0.0
    accel_noise_density: float = 0.0
    gyro_bias: Sequence[float] = (0.0, 0.0, 0.0)
    accel_bias: Sequence[float] = (0.0, 0.0, 0.0)
    gyro_bias_walk: float = 0.0
    accel_bias_walk: float = 0.0
    seed: int | None = None

    def measure(self, t, omega_body, specific_force_body) -> tuple[np.ndarray, np.ndarray]:
        dt = _uniform_dt(t)
        rng = np.random.default_rng(self.seed)
        w = np.asarray(omega_body, dtype=np.float64)
        f = np.asarray(specific_force_body, dtype=np.float64)
        n = w.shape[0]

        def channel(true, density, bias0, walk):
            steps = rng.standard_normal((n, 3)) * walk * math.sqrt(dt)
            bias = np.asarray(bias0, dtype=np.float64) + np.vstack((np.zeros((1, 3)), np.cumsum(steps[:-1], axis=0)))
            return true + bias + rng.standard_normal((n, 3)) * density / math.sqrt(dt)
        return (channel(w, self.gyro_noise_density, self.gyro_bias, self.gyro_bias_walk),
                channel(f, self.accel_noise_density, self.accel_bias, self.accel_bias_walk))


def barometer(t, altitude, *, sigma: float = 0.0, bias: float = 0.0, bias_walk: float = 0.0,
              seed: int | None = None) -> np.ndarray:
    """Barometrik irtifa (m, yukarı pozitif): h + b + η; b rastgele yürüyüş (m/√s)."""
    dt = _uniform_dt(t)
    rng = np.random.default_rng(seed)
    h = np.asarray(altitude, dtype=np.float64)
    walk = np.concatenate(([0.0], np.cumsum(rng.standard_normal(h.size - 1) * bias_walk * math.sqrt(dt))))
    return h + bias + walk + rng.standard_normal(h.size) * sigma


def gps(t, position_ned, velocity_ned, *, rate: float = 1.0, delay: float = 0.0,
        sigma_pos=(0.21, 0.21, 0.40), k_gps: float = 1.0 / 1100.0, sigma_vel: float = 0.05,
        seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GPS: konum hatası Gauss–Markov (Beard & McLain 2012, Böl. 7, GPS modeli): ν[n+1] = e^{−k·Ts}ν[n] + η[n],
    η ~ N(0, σ²); hız beyaz gürültülü. Ölçüm `rate` Hz ile örneklenir ve `delay` s gecikmeli gerçek durumu yansıtır
    (t_k anındaki ölçüm = gerçek(t_k − delay) + hata). Dönüş: (t_gps, konum (m, 3), hız (m, 3))."""
    t = np.asarray(t, dtype=np.float64)
    pos = np.asarray(position_ned, dtype=np.float64)
    vel = np.asarray(velocity_ned, dtype=np.float64)
    rng = np.random.default_rng(seed)
    Ts = 1.0 / rate
    tg = np.arange(t[0] + delay, t[-1] + 1e-12, Ts)
    if tg.size == 0:
        return tg, np.empty((0, 3)), np.empty((0, 3))
    src = tg - delay
    p = np.column_stack([np.interp(src, t, pos[:, i]) for i in range(3)])
    v = np.column_stack([np.interp(src, t, vel[:, i]) for i in range(3)])
    s = np.asarray(sigma_pos, dtype=np.float64)
    a = math.exp(-k_gps * Ts)
    nu = np.zeros(3)
    err = np.empty((tg.size, 3))
    for k in range(tg.size):
        err[k] = nu
        nu = a * nu + rng.standard_normal(3) * s
    return tg, p + err, v + rng.standard_normal(v.shape) * sigma_vel


def magnetometer(quat, field_ned=(0.22, 0.02, 0.43), *, sigma: float = 0.0, seed: int | None = None) -> np.ndarray:
    """Manyetometre (gövde): m_b = R(q)ᵀ·m_NED + η. Varsayılan alan gauss biriminde örnek bir orta enlem değeri."""
    rng = np.random.default_rng(seed)
    q = np.atleast_2d(np.asarray(quat, dtype=np.float64))
    m = np.asarray(field_ned, dtype=np.float64)
    out = np.array([rotmat_from_quat(qk).T @ m for qk in q])
    return out + rng.standard_normal(out.shape) * sigma
