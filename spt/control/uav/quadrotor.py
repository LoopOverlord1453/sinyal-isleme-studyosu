"""Çok rotorlu (varsayılan dörtrotor) 6 serbestlik dereceli katı cisim modeli — NED / FRD, kuaterniyonlu.

Durum (varsayılan, motor dinamiği açık; n motor):
    x = [pn, pe, pd,  vn, ve, vd,  qw, qx, qy, qz,  p, q, r,  Omega1 … Omegan]       (13 + n)
    - konum NED (m), hız **NED dünya çerçevesinde** (m/s) — gövde hızı için `body_velocity(x)`,
    - q: gövde → NED birim kuaterniyon [w, x, y, z]; Euler (ZYX) çıktısı `euler(x)`,
    - p, q, r: gövde açısal hızları (rad/s), Omega_i: motor açısal hız büyüklükleri (rad/s, ≥ 0).
    `motor_dynamics=False` ile Omega durumları kalkar (Ω = Ω_ref anında; 13 durum).

Girişler (`input_mode`):
    "omega"   : Ω_ref,i (rad/s), [omega_min, omega_max] aralığına kırpılır.
    "pwm"     : normalize gaz u_i ∈ [0, 1] → Ω_ref,i = C_R·u_i + ϖ_b (kırpılır) — Quan (2017) motor modeli.
    "virtual" : sanal girişler [T (N), Mx, My, Mz (N·m)] → karıştırıcı (doygunluk gidermeli, `desaturation`)
                → motor itkileri → Ω_ref,i = √(T_i / kT).

Denklemler:
    ṙ = v
    v̇ = g·e3 + (1/m)·R(q)·( −T·e3 − D_t·R(q)ᵀ(v − w_rüzgar) ),             T = Σ kT·Ω_i²
    q̇ = ½·q ⊗ [0, ω] + k_s·(1 − |q|²)·q                                      (k_s: norm kararlılaştırma)
    J·ω̇ = M − D_r·ω − ω × (J·ω + h_r) − ḣ_r
          M = Σ r_i × (−T_i·e3) + Σ s_i·kQ·Ω_i²·e3 = [Σ −y_i·T_i,  Σ x_i·T_i,  Σ s_i·kQ·Ω_i²]
          h_r = −J_r·(Σ s_i·Ω_i)·e3   (yalnız `gyro=True`: rotor açısal momentumu, jiroskopik etki ve
                                        ḣ_r = −J_r·(Σ s_i·Ω̇_i)·e3 rotor ivmelenme tepkisi; motor dinamiği yoksa ḣ_r = 0)
    Ω̇_i = (Ω_ref,i − Ω_i) / τ_m                                               (1. derece motor dinamiği)
R(q) normlanmış q̂ ile kurulur (homojen biçim), dolayısıyla norm sapması fiziği değiştirmez.

Dönüş yönü s_i: +1 = yukarıdan bakışla saat yönünün tersi (CCW), −1 = saat yönü (CW). CCW pervane gövdeye NED'de
pozitif (burun sağa) yaw momenti verir — PX4 CA_ROTORn_KM işaretiyle aynı (CCW için pozitif).

Varsayılan parametreler (`QuadrotorParams`): DJI F450 sınıfı, 10 inç pervane — Quan Quan, "Introduction to
Multicopter Design and Control", Springer 2017, F450 örnek parametreleri (RflySim varsayılan dörtrotor modeliyle
aynı):
m = 1.4 kg, kol 0.225 m, J = diag(0.0211, 0.0219, 0.0366) kg·m², c_T = 1.105e-5 N/(rad/s)²,
c_M = 1.779e-7 N·m/(rad/s)², T_m = 0.0136 s, C_R = 1148 rad/s, ϖ_b = −141.4 rad/s, J_RP = 1.287e-4 kg·m².
Askıda Ω ≈ 557 rad/s (≈ 5320 rpm), itki/ağırlık ≈ 3.3.

Karıştırıcı (mixer) ve doygunluk giderme önceliği: bkz. `Mixer`.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np

from spt.control.errors import ControlError, ControlWarning
from spt.control.uav.common import (
    E3, GRAVITY, LinearModel, SimResult, d_euler_d_quat, d_rotmat_quat_times, d_rotmat_T_quat_times,
    euler_from_quat, euler_from_quat_array, euler_rate_matrix, numjac, quat_from_euler, quat_normalize,
    quat_omega_matrix, quat_xi_matrix, rot_x, rot_y, rot_z, rotmat_from_euler, rotmat_from_quat, simulate, skew,
    throttle_to_speed,
)

E1, E2 = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])


# ============================================================================ parametreler
@dataclass
class QuadrotorParams:
    """Dörtrotor fiziksel parametreleri (SI). Ayrıntı ve kaynak: modül belgesi."""
    mass: float = 1.4
    inertia: np.ndarray = field(default_factory=lambda: np.diag([0.0211, 0.0219, 0.0366]))
    arm_length: float = 0.225
    config: str = "x"                       # "x" | "+" (rotor_positions verilirse yok sayılır)
    kT: float = 1.105e-5                    # itki katsayısı, N/(rad/s)²
    kQ: float = 1.779e-7                    # tepki momenti katsayısı, N·m/(rad/s)²
    tau_m: float = 0.0136                   # motor zaman sabiti, s
    cr: float = 1148.0                      # gaz → hız eğimi, rad/s
    wb: float = -141.4                      # gaz → hız kayması, rad/s
    omega_min: float = 0.0                  # rad/s
    omega_max: float = 1006.6               # rad/s (= cr + wb)
    Jr: float = 1.287e-4                    # rotor + pervane ataleti, kg·m²
    gyro: bool = False                      # rotor jiroskopik etkisi + ivmelenme tepkisi
    drag: tuple = (0.0, 0.0, 0.0)           # doğrusal öteleme sürtünmesi (gövde eksenleri), N/(m/s)
    rot_drag: tuple = (0.0, 0.0, 0.0)       # doğrusal dönme sürtünmesi, N·m/(rad/s)
    g: float = GRAVITY
    rotor_positions: np.ndarray | None = None   # (n, 2) ya da (n, 3) gövde [x, y(, z)], m
    rotor_spins: np.ndarray | None = None       # (n,) +1 CCW / −1 CW (yukarıdan)
    quat_stab: float = 1.0                  # kuaterniyon norm kararlılaştırma kazancı k_s, 1/s

    def geometry(self) -> tuple[np.ndarray, np.ndarray]:
        """(konumlar (n, 3), dönüş yönleri (n,)). Varsayılan sıralama PX4 ile aynı:
        X: 1 ön-sağ CCW, 2 arka-sol CCW, 3 ön-sol CW, 4 arka-sağ CW;  +: 1 sağ CCW, 2 sol CCW, 3 ön CW, 4 arka CW."""
        if self.rotor_positions is not None:
            pos = np.asarray(self.rotor_positions, dtype=np.float64)
            if pos.ndim != 2 or pos.shape[1] not in (2, 3) or pos.shape[0] < 4:
                raise ControlError("Rotor konumları (n, 2) ya da (n, 3) biçiminde ve en az 4 rotor olmalı.")
            if pos.shape[1] == 2:
                pos = np.column_stack((pos, np.zeros(pos.shape[0])))
            if self.rotor_spins is None:
                raise ControlError("Özel rotor konumlarıyla birlikte dönüş yönleri de verilmeli.")
            spins = np.asarray(self.rotor_spins, dtype=np.float64).ravel()
            if spins.size != pos.shape[0] or not np.all(np.abs(spins) == 1.0):
                raise ControlError("Dönüş yönleri her rotor için +1 (CCW) ya da −1 (CW) olmalı.")
            return pos, spins
        L = float(self.arm_length)
        if self.config.lower() == "x":
            a = L / math.sqrt(2.0)
            pos = np.array([[a, a, 0.0], [-a, -a, 0.0], [a, -a, 0.0], [-a, a, 0.0]])
        elif self.config == "+":
            pos = np.array([[0.0, L, 0.0], [0.0, -L, 0.0], [L, 0.0, 0.0], [-L, 0.0, 0.0]])
        else:
            raise ControlError(f"Bilinmeyen yapılandırma: {self.config!r} ('x' ya da '+').")
        spins = np.array([1.0, 1.0, -1.0, -1.0]) if self.rotor_spins is None else \
            np.asarray(self.rotor_spins, dtype=np.float64).ravel()
        return pos, spins


# ============================================================================ karıştırıcı
class Mixer:
    """Sanal girişler v = [T, Mx, My, Mz] ↔ motor itkileri T_i (N).

    Etkinlik (ters karıştırıcı, motorlar → sanal):  v = E·t,  E = [1 …; −y_i …; x_i …; s_i·kQ/kT …]  (4 × n).
    Karıştırıcı (sanal → motorlar): t = P·v, P = E⁺ (n = 4'te tam ters; n > 4'te en küçük normlu çözüm), sonra
    [T_min, T_max] sınırları için doygunluk giderme (`allocate(..., method=)`):

      "px4"         PX4 ControlAllocationSequentialDesaturation, airmode kapalı (PX4 varsayılanı):
                    (1) roll + pitch + itki karıştırılır; doygunluk yalnız itkiyi AZALTARAK giderilir (artırılmaz),
                    (2) gerekirse roll, sonra pitch kazancı düşürülür, (3) yaw ayrıca eklenir ve doygunluk önce
                    YAW kesilerek giderilir (üst sınıra %15 yaw payı tanınır), (4) son olarak itki yalnız azaltılır.
                    Öncelik: roll/pitch > itki > yaw — yani evet, PX4'te önce yaw kesilir.
      "px4_airmode" PX4 airmode roll/pitch: (1)'de itki her iki yönde kaydırılabilir; yaw yine en düşük öncelik.
      "scale"       Betaflight benzeri (airmode): üç moment birlikte ölçeklenir (motor aralığı sığmazsa),
                    ardından itki kaydırılır — eksenler arası öncelik yok, itki en düşük öncelik. (Betaflight ayrıca
                    yaw PID toplamını daha düşük sınırla — pidsum_limit_yaw — kısar; bu, denetleyicinin işidir.)
      "clip"        E⁺·v sonra her motor ayrı kırpılır (öncelik yok; moment yönü bozulabilir).
      "none"        sınırsız E⁺·v (yalnız çözümleme için).

    Not: PX4 algoritması normalize [0, 1] aktüatör uzayında çalışır; burada aynı adımlar N biriminde uygulanır
    (kazanç hesapları ölçekten bağımsızdır). PX4'ün yeni sürümlerindeki "zayıf etkinlikli aktüatörü atla" eşiği
    (normalize 0.2) yerine sayısal sıfır eşiği kullanılır — simetrik çerçevelerde sonuç aynıdır."""

    YAW_MARGIN = 0.15

    def __init__(self, effectiveness: np.ndarray, t_min, t_max):
        self.E = np.asarray(effectiveness, dtype=np.float64)
        if self.E.shape[0] != 4 or self.E.shape[1] < 4:
            raise ControlError("Etkinlik matrisi 4 × n (n ≥ 4) olmalı.")
        if np.linalg.matrix_rank(self.E) < 4:
            raise ControlError("Rotor yerleşimi dört sanal girişi bağımsız üretemiyor (etkinlik matrisi tekil).")
        n = self.E.shape[1]
        self.P = np.linalg.pinv(self.E)
        self.t_min = np.broadcast_to(np.asarray(t_min, dtype=np.float64), (n,)).copy()
        self.t_max = np.broadcast_to(np.asarray(t_max, dtype=np.float64), (n,)).copy()

    @property
    def n(self) -> int:
        return self.E.shape[1]

    def virtual(self, thrusts) -> np.ndarray:
        """Ters karıştırıcı: motor itkileri (N) → [T, Mx, My, Mz]."""
        return self.E @ np.asarray(thrusts, dtype=np.float64)

    # --- PX4 sıralı doygunluk giderme yardımcıları
    @staticmethod
    def _gain(d, sp, lo, hi) -> float:
        """PX4 computeDesaturationGain: doygun her aktüatör için sınıra getiren k; k_min (≤ 0) + k_max (≥ 0)."""
        ad = np.abs(d)
        ok = ad > 1e-12 * float(ad.max()) if d.size else ad > 0
        below = ok & (sp < lo)
        above = ok & (sp > hi)
        if not (below.any() or above.any()):
            return 0.0
        ks = np.concatenate(((lo - sp)[below] / d[below], (hi - sp)[above] / d[above]))
        return min(0.0, float(ks.min())) + max(0.0, float(ks.max()))

    def _desat(self, sp, d, lo, hi, reduce_only=False):
        k = self._gain(d, sp, lo, hi)
        if reduce_only and k > 0.0:      # d itki sütunu (pozitif): k > 0 itkiyi artırırdı
            return sp
        sp = sp + k * d
        return sp + 0.5 * self._gain(d, sp, lo, hi) * d

    def allocate(self, v, method: str = "px4") -> tuple[np.ndarray, dict]:
        """Sanal giriş → motor itkileri (N). Dönüş: (t (n,), bilgi: {'saturated', 'achieved'})."""
        v = np.asarray(v, dtype=np.float64).ravel()
        if v.size != 4:
            raise ControlError("Sanal giriş [T, Mx, My, Mz] dört elemanlı olmalı.")
        P, lo, hi = self.P, self.t_min, self.t_max
        raw = P @ v
        if method == "none":
            t = raw
        elif method == "clip":
            t = np.clip(raw, lo, hi)
        elif method in ("px4", "px4_airmode"):
            sp = P[:, 0] * v[0] + P[:, 1] * v[1] + P[:, 2] * v[2]
            if method == "px4":
                sp = self._desat(sp, P[:, 0], lo, hi, reduce_only=True)
                sp = self._desat(sp, P[:, 1], lo, hi)
                sp = self._desat(sp, P[:, 2], lo, hi)
            else:
                sp = self._desat(sp, P[:, 0], lo, hi)
            sp = sp + P[:, 3] * v[3]
            sp = self._desat(sp, P[:, 3], lo, hi + (hi - lo) * self.YAW_MARGIN)
            sp = self._desat(sp, P[:, 0], lo, hi, reduce_only=True)
            t = np.clip(sp, lo, hi)
        elif method == "scale":
            m = P[:, 1:] @ v[1:]
            span = float(np.max(m) - np.min(m))
            avail = float(np.min(hi - lo))
            if span > avail > 0:
                m = m * (avail / span)
            sp = self._desat(m + P[:, 0] * v[0], P[:, 0], lo, hi)
            t = np.clip(sp, lo, hi)
        else:
            raise ControlError(f"Bilinmeyen doygunluk giderme yöntemi: {method!r}.")
        sat = bool(np.any(raw < lo - 1e-12) or np.any(raw > hi + 1e-12))
        return t, {"saturated": sat, "achieved": self.E @ t, "unconstrained": raw}


# ============================================================================ model
class Quadrotor:
    """Dörtrotor (n rotorlu) 6-DOF modeli. Ayrıntı: modül belgesi."""

    def __init__(self, params: QuadrotorParams | None = None, *, input_mode: str = "omega",
                 motor_dynamics: bool = True, desaturation: str = "px4", wind=None):
        self.params = params if params is not None else QuadrotorParams()
        if input_mode not in ("omega", "pwm", "virtual"):
            raise ControlError("Giriş kipi 'omega', 'pwm' ya da 'virtual' olmalı.")
        self.input_mode = input_mode
        self.motor_dynamics = bool(motor_dynamics)
        self.desaturation = desaturation
        self.wind = wind                       # None, sabit NED vektörü (3,) ya da çağrılabilir w(t) → (3,)
        p = self.params
        J = np.asarray(p.inertia, dtype=np.float64)
        if J.shape != (3, 3) or not np.allclose(J, J.T) or np.any(np.linalg.eigvalsh(J) <= 0):
            raise ControlError("Eylemsizlik matrisi 3×3 simetrik ve pozitif tanımlı olmalı.")
        if p.mass <= 0 or p.kT <= 0 or p.kQ < 0 or p.tau_m <= 0:
            raise ControlError("Kütle, itki katsayısı ve motor zaman sabiti pozitif olmalı.")
        self.J = J
        self.Jinv = np.linalg.inv(J)
        self.pos, self.spins = p.geometry()
        self.n = self.pos.shape[0]
        self.Dt = np.diag(np.asarray(p.drag, dtype=np.float64))
        self.Dr = np.diag(np.asarray(p.rot_drag, dtype=np.float64))
        # hızlı skaler yol için önbellek
        self._drag = tuple(float(v) for v in np.asarray(p.drag, dtype=np.float64))
        self._drot = tuple(float(v) for v in np.asarray(p.rot_drag, dtype=np.float64))
        self._has_drag = any(v != 0.0 for v in self._drag)
        self._Jl = J.tolist()
        self._Jinvl = self.Jinv.tolist()
        E = np.vstack((np.ones(self.n), -self.pos[:, 1], self.pos[:, 0], self.spins * p.kQ / p.kT))
        self.mixer = Mixer(E, p.kT * p.omega_min ** 2, p.kT * p.omega_max ** 2)
        nm = [f"Omega{i + 1}" for i in range(self.n)]
        self.state_names = ("pn", "pe", "pd", "vn", "ve", "vd", "qw", "qx", "qy", "qz", "p", "q", "r") + \
            (tuple(nm) if self.motor_dynamics else ())
        if input_mode == "omega":
            self.input_names = tuple(f"{s}_ref" for s in nm)
        elif input_mode == "pwm":
            self.input_names = tuple(f"u{i + 1}" for i in range(self.n))
        else:
            self.input_names = ("T", "Mx", "My", "Mz")
        self.euler_state_names = ("pn", "pe", "pd", "vn", "ve", "vd", "phi", "theta", "psi", "p", "q", "r") + \
            (tuple(nm) if self.motor_dynamics else ())

    # ---------------------------------------------------------------- boyutlar / yardımcılar
    @property
    def nx(self) -> int:
        return 13 + (self.n if self.motor_dynamics else 0)

    @property
    def nu(self) -> int:
        return len(self.input_names)

    def _wind(self, t: float) -> np.ndarray:
        if self.wind is None:
            return np.zeros(3)
        if callable(self.wind):
            return np.asarray(self.wind(t), dtype=np.float64)
        return np.asarray(self.wind, dtype=np.float64)

    def motor_reference(self, u) -> np.ndarray:
        """Girişten motor hız referansı Ω_ref (rad/s), kırpılmış."""
        p = self.params
        u = np.asarray(u, dtype=np.float64).ravel()
        if u.size != self.nu:
            raise ControlError(f"Giriş {self.nu} elemanlı olmalı ({u.size} verildi).")
        if self.input_mode == "omega":
            return np.clip(u, p.omega_min, p.omega_max)
        if self.input_mode == "pwm":
            return throttle_to_speed(u, p.cr, p.wb, p.omega_min, p.omega_max)
        t, _ = self.mixer.allocate(u, self.desaturation)
        return np.sqrt(np.maximum(t, 0.0) / p.kT)

    def _dref_du(self, u) -> np.ndarray:
        """∂Ω_ref/∂u (n × nu), doygunluk dışında."""
        p = self.params
        u = np.asarray(u, dtype=np.float64).ravel()
        if self.input_mode == "omega":
            inside = (u > p.omega_min) & (u < p.omega_max)
            if not np.all(inside):
                warnings.warn("Doğrusallaştırma noktasında motor komutu sınırda; türev sıfır alındı.", ControlWarning)
            return np.diag(inside.astype(float))
        if self.input_mode == "pwm":
            w = p.cr * u + p.wb
            inside = (u > 0) & (u < 1) & (w > p.omega_min) & (w < p.omega_max)
            if not np.all(inside):
                warnings.warn("Doğrusallaştırma noktasında gaz komutu sınırda; türev sıfır alındı.", ControlWarning)
            return np.diag(p.cr * inside.astype(float))
        t, info = self.mixer.allocate(u, self.desaturation)
        if info["saturated"]:
            warnings.warn("Doğrusallaştırma noktasında karıştırıcı doygun; doygunluk giderme ihmal edildi.",
                          ControlWarning)
        om = np.sqrt(np.maximum(info["unconstrained"], 0.0) / p.kT)
        if np.any(om <= 0):
            raise ControlError("Sıfır motor hızında sanal giriş doğrusallaştırması tanımsız (√ türevi sonsuz).")
        return np.diag(1.0 / (2.0 * p.kT * om)) @ self.mixer.P

    def split(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size != self.nx:
            raise ControlError(f"Durum {self.nx} elemanlı olmalı ({x.size} verildi).")
        return x[0:3], x[3:6], x[6:10], x[10:13], (x[13:] if self.motor_dynamics else None)

    def project(self, x) -> np.ndarray:
        """Kuaterniyonu normlar (sabit adımlı tümlemede her adım sonunda)."""
        x = np.array(x, dtype=np.float64)
        x[6:10] = quat_normalize(x[6:10])
        return x

    def initial_state(self, position=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0), euler=(0.0, 0.0, 0.0),
                      omega=(0.0, 0.0, 0.0), motor_speeds=None) -> np.ndarray:
        """Durum vektörü kurar (euler = (roll, pitch, yaw) rad; motor hızları verilmezse askı hızı)."""
        x = np.concatenate((np.asarray(position, float), np.asarray(velocity, float), quat_from_euler(*euler),
                            np.asarray(omega, float)))
        if self.motor_dynamics:
            ms = self.hover_motor_speeds() if motor_speeds is None else \
                np.broadcast_to(np.asarray(motor_speeds, float), (self.n,))
            x = np.concatenate((x, ms))
        return x

    # ---------------------------------------------------------------- dinamik
    def _rotor_terms(self, Om, Om_dot):
        p = self.params
        Ti = p.kT * Om * Om
        M = np.array([-self.pos[:, 1] @ Ti, self.pos[:, 0] @ Ti, p.kQ * (self.spins @ (Om * Om))])
        if p.gyro:
            h = -p.Jr * (self.spins @ Om) * E3
            hdot = -p.Jr * (self.spins @ Om_dot) * E3
        else:
            h = hdot = np.zeros(3)
        return Ti.sum(), M, h, hdot

    def derivatives_reference(self, x, u, t: float = 0.0) -> np.ndarray:
        """ẋ = f(x, u, t) — matris biçimli başvuru uygulaması (`derivatives` hızlı skaler yolla aynıdır; test edilir)."""
        p = self.params
        _, v, q, w, Om = self.split(x)
        ref = self.motor_reference(u)
        if self.motor_dynamics:
            Om_dot = (ref - Om) / p.tau_m
        else:
            Om, Om_dot = ref, np.zeros(self.n)
        T, M, h, hdot = self._rotor_terms(Om, Om_dot)
        R = rotmat_from_quat(q)
        vrel = v - self._wind(t)
        F_b = -T * E3 - self.Dt @ (R.T @ vrel)
        vdot = p.g * E3 + R @ F_b / p.mass
        qdot = 0.5 * quat_omega_matrix(w) @ q + p.quat_stab * (1.0 - q @ q) * q
        wdot = self.Jinv @ (M - self.Dr @ w - np.cross(w, self.J @ w + h) - hdot)
        out = np.concatenate((v, vdot, qdot, wdot))
        return np.concatenate((out, Om_dot)) if self.motor_dynamics else out

    def derivatives(self, x, u, t: float = 0.0) -> np.ndarray:
        """ẋ = f(x, u, t) (skaler hızlı yol; denklemler modül belgesinde)."""
        p = self.params
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1 or x.size != self.nx:
            raise ControlError(f"Durum {self.nx} elemanlı olmalı ({x.size} verildi).")
        ref = self.motor_reference(u)
        if self.motor_dynamics:
            Om = x[13:]
            Om_dot = (ref - Om) / p.tau_m
        else:
            Om, Om_dot = ref, None
        Om2 = Om * Om
        Ti = p.kT * Om2
        T = float(Ti.sum())
        Mx, My = -float(self.pos[:, 1] @ Ti), float(self.pos[:, 0] @ Ti)
        Mz = p.kQ * float(self.spins @ Om2)
        if p.gyro:
            hz = -p.Jr * float(self.spins @ Om)
            hdz = -p.Jr * float(self.spins @ Om_dot) if Om_dot is not None else 0.0
        else:
            hz = hdz = 0.0
        _, _, _, vn, ve, vd, qw, qx, qy, qz, wp, wq, wr = x[:13].tolist()
        n2 = qw * qw + qx * qx + qy * qy + qz * qz
        s = 1.0 / n2
        r00 = (qw * qw + qx * qx - qy * qy - qz * qz) * s
        r01 = 2.0 * (qx * qy - qw * qz) * s
        r02 = 2.0 * (qx * qz + qw * qy) * s
        r10 = 2.0 * (qx * qy + qw * qz) * s
        r11 = (qw * qw - qx * qx + qy * qy - qz * qz) * s
        r12 = 2.0 * (qy * qz - qw * qx) * s
        r20 = 2.0 * (qx * qz - qw * qy) * s
        r21 = 2.0 * (qy * qz + qw * qx) * s
        r22 = (qw * qw - qx * qx - qy * qy + qz * qz) * s
        fx = fy = 0.0
        fz = -T
        if self._has_drag:
            wn, we, wd = self._wind(t).tolist()
            a, b, c = vn - wn, ve - we, vd - wd
            dx, dy, dz = self._drag
            fx -= dx * (r00 * a + r10 * b + r20 * c)
            fy -= dy * (r01 * a + r11 * b + r21 * c)
            fz -= dz * (r02 * a + r12 * b + r22 * c)
        im = 1.0 / p.mass
        an = (r00 * fx + r01 * fy + r02 * fz) * im
        ae = (r10 * fx + r11 * fy + r12 * fz) * im
        ad = (r20 * fx + r21 * fy + r22 * fz) * im + p.g
        ks = p.quat_stab * (1.0 - n2)
        dqw = 0.5 * (-wp * qx - wq * qy - wr * qz) + ks * qw
        dqx = 0.5 * (wp * qw + wr * qy - wq * qz) + ks * qx
        dqy = 0.5 * (wq * qw - wr * qx + wp * qz) + ks * qy
        dqz = 0.5 * (wr * qw + wq * qx - wp * qy) + ks * qz
        (j00, j01, j02), (j10, j11, j12), (j20, j21, j22) = self._Jl
        ax_ = j00 * wp + j01 * wq + j02 * wr
        ay_ = j10 * wp + j11 * wq + j12 * wr
        az_ = j20 * wp + j21 * wq + j22 * wr + hz
        dp, dq, dr = self._drot
        m0 = Mx - dp * wp - (wq * az_ - wr * ay_)
        m1 = My - dq * wq - (wr * ax_ - wp * az_)
        m2 = Mz - dr * wr - (wp * ay_ - wq * ax_) - hdz
        (i00, i01, i02), (i10, i11, i12), (i20, i21, i22) = self._Jinvl
        head = [vn, ve, vd, an, ae, ad, dqw, dqx, dqy, dqz,
                i00 * m0 + i01 * m1 + i02 * m2, i10 * m0 + i11 * m1 + i12 * m2, i20 * m0 + i21 * m1 + i22 * m2]
        if Om_dot is None:
            return np.array(head)
        out = np.empty(self.nx)
        out[:13] = head
        out[13:] = Om_dot
        return out

    def specific_force(self, x, u, t: float = 0.0) -> np.ndarray:
        """İvmeölçerin ölçtüğü özgül kuvvet (gövde, m/s²): f = Rᵀ(v̇ − g·e3); askıda ≈ [0, 0, −g]."""
        p = self.params
        _, v, q, _, Om = self.split(x)
        if not self.motor_dynamics:
            Om = self.motor_reference(u)
        R = rotmat_from_quat(q)
        T = p.kT * float(Om @ Om)
        return (-T * E3 - self.Dt @ (R.T @ (v - self._wind(t)))) / p.mass

    def euler(self, x) -> tuple[float, float, float]:
        return euler_from_quat(np.asarray(x, float)[6:10])

    def body_velocity(self, x) -> np.ndarray:
        x = np.asarray(x, float)
        return rotmat_from_quat(x[6:10]).T @ x[3:6]

    def kinetic_energy(self, x) -> float:
        """Öteleme + dönme kinetik enerjisi (rotor bağıl dönüşü hariç)."""
        x = np.asarray(x, float)
        w = x[10:13]
        return 0.5 * self.params.mass * float(x[3:6] @ x[3:6]) + 0.5 * float(w @ self.J @ w)

    def angular_momentum_world(self, x) -> np.ndarray:
        x = np.asarray(x, float)
        return rotmat_from_quat(x[6:10]) @ (self.J @ x[10:13])

    # ---------------------------------------------------------------- denge (trim)
    def hover_thrusts(self) -> np.ndarray:
        p = self.params
        t = self.mixer.P @ np.array([p.mass * p.g, 0.0, 0.0, 0.0])
        if np.any(t < self.mixer.t_min - 1e-12) or np.any(t > self.mixer.t_max + 1e-12):
            raise ControlError("Askıda kalma motor sınırları içinde mümkün değil (itki yetersiz ya da dengesiz).")
        return t

    def hover_motor_speeds(self) -> np.ndarray:
        return np.sqrt(self.hover_thrusts() / self.params.kT)

    def trim_hover(self, position=(0.0, 0.0, 0.0), yaw: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Askıda kalma dengesi (x0, u0): her motor (simetrik çerçevede) m·g/n itki, Ω = √(m·g/(n·kT)), ẋ = 0."""
        p = self.params
        om = self.hover_motor_speeds()
        x0 = np.concatenate((np.asarray(position, float), np.zeros(3), quat_from_euler(0.0, 0.0, yaw), np.zeros(3)))
        if self.motor_dynamics:
            x0 = np.concatenate((x0, om))
        if self.input_mode == "omega":
            u0 = om.copy()
        elif self.input_mode == "pwm":
            u0 = (om - p.wb) / p.cr
        else:
            u0 = np.array([p.mass * p.g, 0.0, 0.0, 0.0])
        return x0, u0

    # ---------------------------------------------------------------- doğrusallaştırma
    def _euler_to_native(self, xe):
        xe = np.asarray(xe, dtype=np.float64)
        q = quat_from_euler(xe[6], xe[7], xe[8])
        return np.concatenate((xe[0:6], q, xe[9:]))

    def derivatives_euler(self, xe, u, t: float = 0.0) -> np.ndarray:
        """Euler açılı eşdeğer model: xe = [r, v, φ, θ, ψ, ω, (Ω)], η̇ = W(φ, θ)·ω."""
        xe = np.asarray(xe, dtype=np.float64)
        d = self.derivatives(self._euler_to_native(xe), u, t)
        etadot = euler_rate_matrix(xe[6], xe[7]) @ xe[9:12]
        return np.concatenate((d[0:6], etadot, d[10:]))

    def _outputs(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return np.concatenate((x[0:6], np.array(euler_from_quat(x[6:10])), x[10:]))

    @property
    def output_names(self) -> tuple[str, ...]:
        return self.euler_state_names

    def output(self, x, u=None, t: float = 0.0) -> np.ndarray:
        """Çıkış vektörü (`output_names`): [pn pe pd vn ve vd phi theta psi p q r (Omega_i)] — Euler q̂'dan."""
        return self._outputs(x)

    def linearize(self, x0, u0, *, attitude: str = "euler", t: float = 0.0) -> LinearModel:
        """(x0, u0) noktasında **analitik** Jacobian.

        attitude="euler"      : durumlar [pn pe pd vn ve vd phi theta psi p q r (Omega_i)], C = I.
        attitude="quaternion" : modelin kendi durumları (kuaterniyonlu); çıkışlar Euler'li durumlar
                                [pn … vd phi theta psi p q r (Omega_i)] (C, Euler(q̂) türeviyle). Kuaterniyon
                                normu yönünde −2·k_s özdeğerli, çıkışlardan gözlenemeyen bir kip bulunur.
        x0 her iki durumda da modelin kendi (kuaterniyonlu) durum vektörüdür. Girişler `input_mode`'a göre.
        Doygunluk / kırpma etkin değilken geçerlidir (sınırdaysa uyarı)."""
        p = self.params
        _, v, q, w, Om = self.split(x0)
        q = np.asarray(q, float)
        u0 = np.asarray(u0, dtype=np.float64).ravel()
        ref = self.motor_reference(u0)
        dref = self._dref_du(u0)
        md = self.motor_dynamics
        if not md:
            Om = ref
        Om_dot = (ref - Om) / p.tau_m if md else np.zeros(self.n)
        T, M, h, _ = self._rotor_terms(Om, Om_dot)
        vrel = v - self._wind(t)
        n = self.n
        euler = attitude == "euler"
        if attitude not in ("euler", "quaternion"):
            raise ControlError("Yönelim gösterimi 'euler' ya da 'quaternion' olmalı.")
        na = 3 if euler else 4
        iv, ia, iw = slice(3, 6), slice(6, 6 + na), slice(6 + na, 9 + na)
        io = slice(9 + na, 9 + na + n)
        nx = 9 + na + (n if md else 0)
        A = np.zeros((nx, nx))
        Bref = np.zeros((nx, n))           # ∂f/∂Ω_ref
        A[0:3, iv] = np.eye(3)
        # öteleme
        if euler:
            phi, th, psi = euler_from_quat(q)
            Rx, Ry, Rz = rot_x(phi), rot_y(th), rot_z(psi)
            R = Rz @ Ry @ Rx
            dR = (R @ skew(E1), Rz @ Ry @ skew(E2) @ Rx, skew(E3) @ R)
            a = self.Dt @ (R.T @ vrel)
            for k in range(3):
                A[iv, 6 + k] = (dR[k] @ (-T * E3) - dR[k] @ a - R @ self.Dt @ (dR[k].T @ vrel)) / p.mass
        else:
            R = rotmat_from_quat(q)
            a = self.Dt @ (R.T @ vrel)
            A[iv, ia] = (d_rotmat_quat_times(q, -T * E3) - d_rotmat_quat_times(q, a)
                         - R @ self.Dt @ d_rotmat_T_quat_times(q, vrel)) / p.mass
        A[iv, iv] = -R @ self.Dt @ R.T / p.mass
        dv_dOm = -np.outer(R @ E3, 2.0 * p.kT * Om) / p.mass
        # yönelim
        if euler:
            sp, cp = math.sin(phi), math.cos(phi)
            ct, tt = math.cos(th), math.tan(th)
            pw, qw_, rw = w
            A[ia, ia] = np.array([
                [(qw_ * cp - rw * sp) * tt, (qw_ * sp + rw * cp) / ct ** 2, 0.0],
                [-qw_ * sp - rw * cp, 0.0, 0.0],
                [(qw_ * cp - rw * sp) / ct, (qw_ * sp + rw * cp) * math.sin(th) / ct ** 2, 0.0]])
            A[ia, iw] = euler_rate_matrix(phi, th)
        else:
            A[ia, ia] = 0.5 * quat_omega_matrix(w) + p.quat_stab * ((1.0 - q @ q) * np.eye(4) - 2.0 * np.outer(q, q))
            A[ia, iw] = 0.5 * quat_xi_matrix(q)
        # dönme
        A[iw, iw] = self.Jinv @ (-self.Dr - skew(w) @ self.J + skew(self.J @ w + h))
        dM = np.vstack((-self.pos[:, 1] * 2.0 * p.kT * Om, self.pos[:, 0] * 2.0 * p.kT * Om,
                        self.spins * 2.0 * p.kQ * Om))
        dh = -p.Jr * np.outer(E3, self.spins) if p.gyro else np.zeros((3, n))
        dw_dOm = self.Jinv @ (dM - skew(w) @ dh)
        if md:
            if p.gyro:
                dhdot = p.Jr * np.outer(E3, self.spins) / p.tau_m      # ∂ḣ_r/∂Ω  (∂ḣ_r/∂Ω_ref = −bu)
                dw_dOm = dw_dOm - self.Jinv @ dhdot
                Bref[iw, :] = self.Jinv @ dhdot
            A[iv, io] = dv_dOm
            A[iw, io] = dw_dOm
            A[io, io] = -np.eye(n) / p.tau_m
            Bref[io, :] = np.eye(n) / p.tau_m
        else:
            Bref[iv, :] = dv_dOm
            Bref[iw, :] = dw_dOm
        B = Bref @ dref
        if euler:
            names = self.euler_state_names
            xe0 = np.concatenate((np.asarray(x0, float)[0:6], [phi, th, psi], np.asarray(x0, float)[10:]))
            C = np.eye(nx)
            y0 = xe0.copy()
            xdot0 = self.derivatives_euler(xe0, u0, t)
            x_op = xe0
        else:
            names = self.state_names
            C = np.zeros((nx - 1, nx))
            C[0:6, 0:6] = np.eye(6)
            C[6:9, 6:10] = d_euler_d_quat(q)
            C[9:, 10:] = np.eye(nx - 10)
            x_op = np.asarray(x0, float).copy()
            y0 = self._outputs(x_op)
            xdot0 = self.derivatives(x_op, u0, t)
        D = np.zeros((C.shape[0], B.shape[1]))
        return LinearModel(A, B, C, D, tuple(names), tuple(self.input_names), tuple(self.euler_state_names),
                           x_op, u0.copy(), y0, xdot0, {"attitude": attitude, "method": "analitik"})

    def linearize_numeric(self, x0, u0, *, attitude: str = "euler", t: float = 0.0) -> LinearModel:
        """Aynı modelin merkezi fark + Richardson sayısal Jacobian'ı (analitik sonucun bağımsız denetimi)."""
        x0 = np.asarray(x0, dtype=np.float64)
        u0 = np.asarray(u0, dtype=np.float64).ravel()
        if attitude == "euler":
            xe0 = np.concatenate((x0[0:6], euler_from_quat(x0[6:10]), x0[10:]))
            A = numjac(lambda xx: self.derivatives_euler(xx, u0, t), xe0)
            B = numjac(lambda uu: self.derivatives_euler(xe0, uu, t), u0)
            C = np.eye(xe0.size)
            names, x_op, y0 = self.euler_state_names, xe0, xe0
        elif attitude == "quaternion":
            A = numjac(lambda xx: self.derivatives(xx, u0, t), x0)
            B = numjac(lambda uu: self.derivatives(x0, uu, t), u0)
            C = numjac(self._outputs, x0)
            names, x_op, y0 = self.state_names, x0, self._outputs(x0)
        else:
            raise ControlError("Yönelim gösterimi 'euler' ya da 'quaternion' olmalı.")
        return LinearModel(A, B, C, np.zeros((C.shape[0], B.shape[1])), tuple(names), tuple(self.input_names),
                           tuple(self.euler_state_names), x_op, u0, y0, None,
                           {"attitude": attitude, "method": "sayısal"})

    # ---------------------------------------------------------------- benzetim
    def simulate(self, x0, u, t, **kwargs) -> SimResult:
        """common.simulate ile benzetim; sonuç `info['euler']` (nt, 3) [roll, pitch, yaw] içerir."""
        res = simulate(self, x0, u, t, **kwargs)
        res.info["euler"] = euler_from_quat_array(res.x[:, 6:10])
        return res


# ============================================================================ küçük açı (standart) doğrusal model
def hover_small_angle_model(params: QuadrotorParams | None = None, *, yaw: float = 0.0,
                            motor_dynamics: bool = True) -> LinearModel:
    """Askıda kalma çevresinde standart küçük açı modeli (kapalı biçimde türetilmiş).

    Durumlar: [pn pe pd vn ve vd phi theta psi p q r] + (motor dinamiği varsa) gerçekleşen sanal kuvvetler
    [T Mx My Mz]; girişler komut [T_c Mx_c My_c Mz_c]; tümü askı noktasından sapma (T için x0'da m·g).

        v̇n = −g(cosψ0·θ + sinψ0·φ),  v̇e = −g(sinψ0·θ − cosψ0·φ),  v̇d = −δT/m      (+ −R0·D_t·R0ᵀ·v/m)
        φ̇ = p, θ̇ = q, ψ̇ = r,   ω̇ = J⁻¹([Mx, My, Mz] − D_r·ω + [h_r0×]ω)
        Ḟ = (F_c − F)/τ_m                                       (motor gecikmesi sanal koordinatlarda tam)
        gyro=True ise ek: ω̇ += J⁻¹·e3·J_r·(sᵀ·G⁻¹)(F_c − F)/τ_m   (G = ∂F/∂Ω = E·diag(2·kT·Ω0))
    Eksen başına (sürtünmesiz, gyro yok): φ/Mx_c = 1/(Jx·s²(τs+1)), θ/My_c = 1/(Jy·s²(τs+1)),
    ψ/Mz_c = 1/(Jz·s²(τs+1)), −pd/T_c = 1/(m·s²(τs+1)), x/θ = −g/s², y/φ = g/s² (ψ0 = 0).
    n = 4 rotorda tam Jacobian'ın benzerlik dönüşümüyle aynıdır (test edilir)."""
    p = params if params is not None else QuadrotorParams()
    quad = Quadrotor(p, input_mode="virtual", motor_dynamics=True)
    om0 = quad.hover_motor_speeds()
    m, g, tau = p.mass, p.g, p.tau_m
    cps, sps = math.cos(yaw), math.sin(yaw)
    nx = 16 if motor_dynamics else 12
    A = np.zeros((nx, nx))
    B = np.zeros((nx, 4))
    A[0:3, 3:6] = np.eye(3)
    Rz = rot_z(yaw)
    A[3:6, 3:6] = -Rz @ quad.Dt @ Rz.T / m
    A[3, 6], A[3, 7] = -g * sps, -g * cps
    A[4, 6], A[4, 7] = g * cps, -g * sps
    A[6:9, 9:12] = np.eye(3)
    h0 = -p.Jr * float(quad.spins @ om0) * E3 if p.gyro else np.zeros(3)
    A[9:12, 9:12] = quad.Jinv @ (-quad.Dr + skew(h0))
    if motor_dynamics:
        A[5, 12] = -1.0 / m
        A[9:12, 13:16] = quad.Jinv
        A[12:16, 12:16] = -np.eye(4) / tau
        B[12:16, :] = np.eye(4) / tau
        if p.gyro:
            if quad.n != 4:
                raise ControlError("Jiroskopik terimli küçük açı modeli yalnız 4 rotor için tanımlı.")
            G = quad.mixer.E @ np.diag(2.0 * p.kT * om0)
            row = p.Jr * (quad.spins @ np.linalg.inv(G)) / tau
            gyro = np.outer(quad.Jinv @ E3, row)
            A[9:12, 12:16] -= gyro
            B[9:12, :] += gyro
        names = quad.euler_state_names[:12] + ("T", "Mx", "My", "Mz")
        x0 = np.concatenate((np.zeros(8), [yaw], np.zeros(3), [m * g, 0.0, 0.0, 0.0]))
    else:
        B[5, 0] = -1.0 / m
        B[9:12, 1:4] = quad.Jinv
        names = quad.euler_state_names[:12]
        x0 = np.concatenate((np.zeros(8), [yaw], np.zeros(3)))
    return LinearModel(A, B, np.eye(nx), np.zeros((nx, 4)), names, ("T_c", "Mx_c", "My_c", "Mz_c"), names,
                       x0, np.array([m * g, 0.0, 0.0, 0.0]), x0.copy(), np.zeros(nx),
                       {"hover_speeds": om0, "kind": "küçük açı"})


def axis_transfer_functions(params: QuadrotorParams | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Eksen başına standart transfer fonksiyonları (pay, payda; azalan kuvvetler), askıda, ψ0 = 0, sürtünmesiz,
    jiroskopik etkisiz, çapraz eylemsizlik terimleri J⁻¹ köşegeniyle:
      'roll' φ/Mx_c, 'pitch' θ/My_c, 'yaw' ψ/Mz_c, 'altitude' h/T_c (h = −pd), 'x_from_pitch' x/θ, 'y_from_roll' y/φ."""
    p = params if params is not None else QuadrotorParams()
    Jinv = np.linalg.inv(np.asarray(p.inertia, dtype=np.float64))
    tau = p.tau_m
    den3 = np.array([tau, 1.0, 0.0, 0.0])
    return {
        "roll": (np.array([Jinv[0, 0]]), den3.copy()),
        "pitch": (np.array([Jinv[1, 1]]), den3.copy()),
        "yaw": (np.array([Jinv[2, 2]]), den3.copy()),
        "altitude": (np.array([1.0 / p.mass]), den3.copy()),
        "x_from_pitch": (np.array([-p.g]), np.array([1.0, 0.0, 0.0])),
        "y_from_roll": (np.array([p.g]), np.array([1.0, 0.0, 0.0])),
    }


def euler_rotmat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Kolaylık: gövde → NED döndürme matrisi."""
    return rotmat_from_euler(roll, pitch, yaw)
