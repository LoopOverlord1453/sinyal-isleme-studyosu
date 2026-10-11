"""Sabit kanatlı İHA 6-DOF modeli — Beard & McLain, "Small Unmanned Aircraft: Theory and Practice" (Princeton, 2012)
denklemleri; Aerosonde parametreleri kitabın resmî eşlik deposundan.

Durum (12, kitabın 3. bölümü, Euler açılı):
    x = [pn, pe, pd,  u, v, w,  phi, theta, psi,  p, q, r]
    - konum NED (m; irtifa h = −pd), u v w: **yere göre (eylemsiz) hızın gövde bileşenleri** (m/s),
    - φ θ ψ: ZYX Euler (rad; θ = ±90°'de tekil), p q r: gövde açısal hızları (rad/s).
Giriş (4): u = [delta_e, delta_a, delta_r, delta_t] — dümenler rad, gaz δt ∈ [0, 1] (kırpılmaz; trim sınır denetler).

Kinematik / dinamik (kitap Böl. 3):
    [ṗn ṗe ṗd]ᵀ = R(φ,θ,ψ)·[u v w]ᵀ
    u̇ = r·v − q·w + fx/m,   v̇ = p·w − r·u + fy/m,   ẇ = q·u − p·v + fz/m
    [φ̇ θ̇ ψ̇]ᵀ = W(φ,θ)·[p q r]ᵀ
    ṗ = Γ1·p·q − Γ2·q·r + Γ3·l + Γ4·n,   q̇ = Γ5·p·r − Γ6·(p² − r²) + m/Jy,   ṙ = Γ7·p·q − Γ1·q·r + Γ4·l + Γ8·n
    (J = [[Jx, 0, −Jxz], [0, Jy, 0], [−Jxz, 0, Jz]], Γ katsayıları kitaptaki gibi.)

Kuvvet / moment (Böl. 4):
    Rüzgar: w_gövde = R(φ,θ,ψ)ᵀ·w_NED(sabit) + [u_wg, v_wg, w_wg] (gövde ekseninde sapma, ör. Dryden);
    [ur vr wr] = [u v w] − w_gövde, Va = |·|, α = atan2(wr, ur), β = asin(vr / Va).
    C_L(α) = (1 − σ)(C_L0 + C_Lα·α) + σ·2·sign(α)·sin²α·cosα,
    σ(α) = (1 + e^{−M(α−α0)} + e^{M(α+α0)}) / ((1 + e^{−M(α−α0)})(1 + e^{M(α+α0)}))   (sayısal kararlı biçimde
           1 − expit(−M(α−α0))·expit(M(α+α0)) olarak hesaplanır — cebirsel olarak özdeş),
    C_D(α) = C_Dp + (C_L0 + C_Lα·α)² / (π·e·AR)   (drag_model="quadratic", kitap Böl. 4; varsayılan) ya da
    C_D(α) = C_D0 + C_Dα·α                         (drag_model="linear"),
    C_m(α) = C_m0 + C_mα·α.
    F_lift = ½ρVa²S·[C_L(α) + C_Lq·c·q/(2Va) + C_Lδe·δe],  F_drag = ½ρVa²S·[C_D(α) + C_Dq·c·q/(2Va) + C_Dδe·δe]
    fx = −m·g·sinθ − cosα·F_drag + sinα·F_lift + T_p
    fy =  m·g·cosθ·sinφ + ½ρVa²S·[C_Y0 + C_Yβ·β + C_Yp·b·p/(2Va) + C_Yr·b·r/(2Va) + C_Yδa·δa + C_Yδr·δr]
    fz =  m·g·cosθ·cosφ − sinα·F_drag − cosα·F_lift
    l  = ½ρVa²S·b·[C_l0 + C_lβ·β + C_lp·b·p/(2Va) + C_lr·b·r/(2Va) + C_lδa·δa + C_lδr·δr] − Q_p
    m  = ½ρVa²S·c·[C_m(α) + C_mq·c·q/(2Va) + C_mδe·δe]
    n  = ½ρVa²S·b·[C_n0 + C_nβ·β + C_np·b·p/(2Va) + C_nr·b·r/(2Va) + C_nδa·δa + C_nδr·δr]
    (−Q_p: pervane arkadan bakışla saat yönünde döner; gövdeye tepki momenti negatif roll.)

Pervane modeli (`prop_model`):
    "motor"  (varsayılan) — kitabın güncellenmiş 4. bölümündeki (McLain) motor + pervane modeli: V_in = V_max·δt,
             a·Ω² + b·Ω + c = 0, a = ρD⁵C_Q0/(4π²), b = ρD⁴C_Q1·Va/(2π) + K_Q·K_V/R, c = ρD³C_Q2·Va² − K_Q·V_in/R
             + K_Q·i0, Ω = (−b + √(b² − 4ac))/(2a);  T_p = ρD⁴C_T0Ω²/(4π²) + ρD³C_T1·Va·Ω/(2π) + ρD²C_T2·Va²,
             Q_p = ρD⁵C_Q0Ω²/(4π²) + ρD⁴C_Q1·Va·Ω/(2π) + ρD³C_Q2·Va²  (= ρn²D⁴C_T(J), ρn²D⁵C_Q(J), J = 2πVa/(ΩD)).
    "simple" — kitabın 1. baskı (2012) modeli (Böl. 4): T_p = ½ρS_prop·C_prop·[(k_motor·δt)² − Va²],
             Q_p = k_Tp·(k_Ω·δt)².

Varsayılan parametreler `FixedWingParams.aerosonde()`: byu-magicc/mavsim_public deposu (kitabın yazarlarının
resmî eşlik kodu), mavsim_python/parameters/aerosonde_parameters.py — m = 11 kg, Jx = 0.8244, Jy = 1.135,
Jz = 1.759, Jxz = 0.1204 kg·m², S = 0.55 m², b = 2.8956 m, c = 0.18994 m, ρ = 1.2682 kg/m³, g = 9.81 m/s²,
katsayılar sınıf alanlarında. Basit pervane sabitleri (k_motor = 80, C_prop = 1, k_Tp = k_Ω = 0) aynı deponun
mavsim_matlab/parameters/aerosonde_parameters.m dosyasından.

Trim (Böl. 5 yöntemi, genelleştirilmiş): verilen Va*, γ*, R* (yatay izdüşüm yarıçapı; sonsuz = düz uçuş; R > 0
sağa dönüş) ve β* (varsayılan 0 — koordineli) için bilinmeyenler (α, φ, θ): u v w = Va·(cosα·cosβ, sinβ,
sinα·cosβ); ψ̇* = Va·cosγ/R; [p q r] = ψ̇*·(−sinθ, sinφ·cosθ, cosφ·cosθ) (φ̇ = θ̇ = 0, ψ̇ = ψ̇* kendiliğinden);
δe q̇ = 0'dan, δt u̇ = 0'dan, (δa, δr) ṗ = ṙ = 0'dan kapalı biçimde (δt motor modelinde tek değişkenli kök);
kalan v̇ = 0, ẇ = 0, ḣ = Va·sinγ denklemleri Newton türevli kök bulucuyla (scipy root) çözülür. Kitaptaki θ* = α* + γ*
varsayımı yalnız kanat düz ve β = 0 iken doğrudur; burada θ bilinmeyendir, böylece tırmanan dönüşte de ḣ = Va·sinγ
tam sağlanır. Kitaptaki ψ̇* = Va*/R* yerine yatay yarıçap tanımı (ψ̇* = Va·cosγ/R) kullanılır (γ = 0'da aynı).
Trim rüzgarsız yapılır.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace

import numpy as np
from scipy.special import expit

from spt.control.errors import ControlError
from spt.control.uav.common import LinearModel, SimResult, euler_rate_matrix, numjac, rotmat_from_euler, simulate


# ============================================================================ parametreler
@dataclass
class FixedWingParams:
    # fiziksel
    mass: float = 11.0
    Jx: float = 0.8244
    Jy: float = 1.135
    Jz: float = 1.759
    Jxz: float = 0.1204
    S_wing: float = 0.55
    b: float = 2.8956
    c: float = 0.18994
    rho: float = 1.2682
    e: float = 0.9
    gravity: float = 9.81
    # boylamsal
    C_L_0: float = 0.23
    C_D_0: float = 0.0424
    C_m_0: float = 0.0135
    C_L_alpha: float = 5.61
    C_D_alpha: float = 0.132
    C_m_alpha: float = -2.74
    C_L_q: float = 7.95
    C_D_q: float = 0.0
    C_m_q: float = -38.21
    C_L_delta_e: float = 0.13
    C_D_delta_e: float = 0.0135
    C_m_delta_e: float = -0.99
    M: float = 50.0
    alpha0: float = 0.47
    epsilon: float = 0.16
    C_D_p: float = 0.043
    # yanal
    C_Y_0: float = 0.0
    C_ell_0: float = 0.0
    C_n_0: float = 0.0
    C_Y_beta: float = -0.98
    C_ell_beta: float = -0.13
    C_n_beta: float = 0.073
    C_Y_p: float = 0.0
    C_ell_p: float = -0.51
    C_n_p: float = 0.069
    C_Y_r: float = 0.0
    C_ell_r: float = 0.25
    C_n_r: float = -0.095
    C_Y_delta_a: float = 0.075
    C_ell_delta_a: float = 0.17
    C_n_delta_a: float = -0.011
    C_Y_delta_r: float = 0.19
    C_ell_delta_r: float = 0.0024
    C_n_delta_r: float = -0.069
    # pervane: motor modeli (güncellenmiş Böl. 4)
    D_prop: float = 20 * 0.0254
    KV: float = 60.0 / (2.0 * math.pi * 145.0)     # V·s/rad (145 rpm/V)
    KQ: float = 60.0 / (2.0 * math.pi * 145.0)     # N·m/A
    R_motor: float = 0.042
    i0: float = 1.5
    V_max: float = 3.7 * 12
    C_Q2: float = -0.01664
    C_Q1: float = 0.004970
    C_Q0: float = 0.005230
    C_T2: float = -0.1079
    C_T1: float = -0.06044
    C_T0: float = 0.09357
    # pervane: basit model (1. baskı)
    S_prop: float = 0.2027
    C_prop: float = 1.0
    k_motor: float = 80.0
    k_T_p: float = 0.0
    k_Omega: float = 0.0
    # model seçimleri
    prop_model: str = "motor"           # "motor" | "simple"
    drag_model: str = "quadratic"       # "quadratic" | "linear"

    @classmethod
    def aerosonde(cls, **overrides) -> "FixedWingParams":
        """Aerosonde (mavsim_public). Anahtar sözcüklerle alan değiştirilebilir."""
        return replace(cls(), **overrides)

    @property
    def AR(self) -> float:
        return self.b ** 2 / self.S_wing

    def gammas(self) -> tuple[float, ...]:
        """(Γ, Γ1 … Γ8) — Beard & McLain Böl. 3."""
        Jx, Jy, Jz, Jxz = self.Jx, self.Jy, self.Jz, self.Jxz
        G = Jx * Jz - Jxz ** 2
        return (G, Jxz * (Jx - Jy + Jz) / G, (Jz * (Jz - Jy) + Jxz ** 2) / G, Jz / G, Jxz / G, (Jz - Jx) / Jy,
                Jxz / Jy, ((Jx - Jy) * Jx + Jxz ** 2) / G, Jx / G)

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


STATE_NAMES = ("pn", "pe", "pd", "u", "v", "w", "phi", "theta", "psi", "p", "q", "r")
INPUT_NAMES = ("delta_e", "delta_a", "delta_r", "delta_t")
LON_STATES = ("u", "w", "q", "theta", "h")
LAT_STATES = ("v", "p", "r", "phi", "psi")


@dataclass
class TrimResult:
    x: np.ndarray            # (12,) trim durumu
    u: np.ndarray            # (4,) trim girişi [δe, δa, δr, δt]
    Va: float
    gamma: float
    R: float
    alpha: float
    beta: float
    phi: float
    theta: float
    psidot: float
    xdot: np.ndarray          # f(x*, u*)
    residual: np.ndarray      # istenen türevlerden fark (u̇ … ṙ, φ̇, θ̇, ψ̇ − ψ̇*, ḣ − Va·sinγ) — ~0
    success: bool = True


# ============================================================================ model
class FixedWing:
    """Beard & McLain 6-DOF sabit kanat modeli. `wind`: None, sabit NED (3,) ya da (6,) [w_n w_e w_d u_g v_g w_g],
    veya çağrılabilir w(t) → (3,) / (6,)."""

    state_names = STATE_NAMES
    input_names = INPUT_NAMES
    nx = 12
    nu = 4

    def __init__(self, params: FixedWingParams | None = None, *, wind=None):
        self.params = params if params is not None else FixedWingParams.aerosonde()
        if self.params.prop_model not in ("motor", "simple"):
            raise ControlError("Pervane modeli 'motor' ya da 'simple' olmalı.")
        if self.params.drag_model not in ("quadratic", "linear"):
            raise ControlError("Sürükleme modeli 'quadratic' ya da 'linear' olmalı.")
        self.wind = wind

    # ---------------------------------------------------------------- rüzgar / hava verisi
    def _wind6(self, t: float) -> np.ndarray:
        w = self.wind
        if w is None:
            return np.zeros(6)
        w = np.asarray(w(t) if callable(w) else w, dtype=np.float64).ravel()
        if w.size == 3:
            return np.concatenate((w, np.zeros(3)))
        if w.size == 6:
            return w
        raise ControlError("Rüzgar 3 (NED sabit) ya da 6 (NED sabit + gövde sapması) elemanlı olmalı.")

    def air_data(self, x, t: float = 0.0) -> dict:
        """Va, α, β, hava-göreli hız (gövde), yer hızı Vg, uçuş yolu açısı γ, rota χ, irtifa h."""
        x = np.asarray(x, dtype=np.float64)
        R = rotmat_from_euler(x[6], x[7], x[8])
        w6 = self._wind6(t)
        vb = x[3:6]
        vr = vb - (R.T @ w6[:3] + w6[3:])
        Va = float(np.linalg.norm(vr))
        if Va <= 0:
            raise ControlError("Hava hızı sıfır: α ve β tanımsız.")
        vg = R @ vb
        Vg = float(np.linalg.norm(vg))
        return {"Va": Va, "alpha": math.atan2(vr[2], vr[0]), "beta": math.asin(max(-1.0, min(1.0, vr[1] / Va))),
                "v_air_body": vr, "Vg": Vg, "gamma": math.asin(max(-1.0, min(1.0, -vg[2] / Vg))) if Vg > 0 else 0.0,
                "chi": math.atan2(vg[1], vg[0]), "h": -float(x[2])}

    # ---------------------------------------------------------------- aerodinamik katsayılar
    def sigma(self, alpha: float) -> float:
        p = self.params
        return 1.0 - float(expit(-p.M * (alpha - p.alpha0)) * expit(p.M * (alpha + p.alpha0)))

    def CL(self, alpha: float) -> float:
        p = self.params
        s = self.sigma(alpha)
        return (1 - s) * (p.C_L_0 + p.C_L_alpha * alpha) + s * (2 * math.copysign(1.0, alpha) * (alpha != 0)
                                                                 * math.sin(alpha) ** 2 * math.cos(alpha))

    def CD(self, alpha: float) -> float:
        p = self.params
        if p.drag_model == "linear":
            return p.C_D_0 + p.C_D_alpha * alpha
        return p.C_D_p + (p.C_L_0 + p.C_L_alpha * alpha) ** 2 / (math.pi * p.e * p.AR)

    def propeller(self, Va: float, delta_t: float) -> tuple[float, float, float]:
        """(T_p itki N, Q_p moment N·m, Ω rad/s) — seçili pervane modeli."""
        p = self.params
        if p.prop_model == "simple":
            T = 0.5 * p.rho * p.S_prop * p.C_prop * ((p.k_motor * delta_t) ** 2 - Va ** 2)
            om = p.k_Omega * delta_t
            return T, p.k_T_p * om ** 2, om
        rho, D = p.rho, p.D_prop
        V_in = p.V_max * delta_t
        a = rho * D ** 5 * p.C_Q0 / (4 * math.pi ** 2)
        b = rho * D ** 4 * p.C_Q1 * Va / (2 * math.pi) + p.KQ * p.KV / p.R_motor
        c = rho * D ** 3 * p.C_Q2 * Va ** 2 - p.KQ * V_in / p.R_motor + p.KQ * p.i0
        disc = b * b - 4 * a * c
        if disc < 0:
            raise ControlError("Motor / pervane denkleminin gerçel çözümü yok (gaz ya da hava hızı aralık dışı).")
        om = (-b + math.sqrt(disc)) / (2 * a)
        T = rho * D ** 4 * p.C_T0 * om ** 2 / (4 * math.pi ** 2) + rho * D ** 3 * p.C_T1 * Va * om / (2 * math.pi) \
            + rho * D ** 2 * p.C_T2 * Va ** 2
        Q = rho * D ** 5 * p.C_Q0 * om ** 2 / (4 * math.pi ** 2) + rho * D ** 4 * p.C_Q1 * Va * om / (2 * math.pi) \
            + rho * D ** 3 * p.C_Q2 * Va ** 2
        return T, Q, om

    def propeller_partials(self, Va: float, delta_t: float) -> tuple[float, float]:
        """(∂T_p/∂Va, ∂T_p/∂δt) analitik (motor modelinde örtük fonksiyon teoremiyle)."""
        p = self.params
        if p.prop_model == "simple":
            return -p.rho * p.S_prop * p.C_prop * Va, p.rho * p.S_prop * p.C_prop * p.k_motor ** 2 * delta_t
        rho, D = p.rho, p.D_prop
        _, _, om = self.propeller(Va, delta_t)
        a = rho * D ** 5 * p.C_Q0 / (4 * math.pi ** 2)
        b = rho * D ** 4 * p.C_Q1 * Va / (2 * math.pi) + p.KQ * p.KV / p.R_motor
        dF_dom = 2 * a * om + b
        dF_dVa = rho * D ** 4 * p.C_Q1 * om / (2 * math.pi) + 2 * rho * D ** 3 * p.C_Q2 * Va
        dF_ddt = -p.KQ * p.V_max / p.R_motor
        dT_dom = 2 * rho * D ** 4 * p.C_T0 * om / (4 * math.pi ** 2) + rho * D ** 3 * p.C_T1 * Va / (2 * math.pi)
        dT_dVa = rho * D ** 3 * p.C_T1 * om / (2 * math.pi) + 2 * rho * D ** 2 * p.C_T2 * Va
        return dT_dVa + dT_dom * (-dF_dVa / dF_dom), dT_dom * (-dF_ddt / dF_dom)

    # ---------------------------------------------------------------- kuvvetler / dinamik
    def forces_moments(self, x, u, t: float = 0.0) -> tuple[np.ndarray, np.ndarray, dict]:
        """(f gövde (3,) N, [l, m, n] (3,) N·m, hava verisi sözlüğü)."""
        p = self.params
        x = np.asarray(x, dtype=np.float64)
        de, da, dr, dt = (float(v) for v in np.asarray(u, dtype=np.float64).ravel())
        ad = self.air_data(x, t)
        Va, al, be = ad["Va"], ad["alpha"], ad["beta"]
        phi, th = x[6], x[7]
        P, Q, Rr = x[9], x[10], x[11]
        qbar = 0.5 * p.rho * Va ** 2
        ca, sa = math.cos(al), math.sin(al)
        lift = qbar * p.S_wing * (self.CL(al) + p.C_L_q * p.c * Q / (2 * Va) + p.C_L_delta_e * de)
        drag = qbar * p.S_wing * (self.CD(al) + p.C_D_q * p.c * Q / (2 * Va) + p.C_D_delta_e * de)
        Tp, Qp, _ = self.propeller(Va, dt)
        mg = p.mass * p.gravity
        fx = -mg * math.sin(th) - ca * drag + sa * lift + Tp
        fy = mg * math.cos(th) * math.sin(phi) + qbar * p.S_wing * (
            p.C_Y_0 + p.C_Y_beta * be + p.C_Y_p * p.b * P / (2 * Va) + p.C_Y_r * p.b * Rr / (2 * Va)
            + p.C_Y_delta_a * da + p.C_Y_delta_r * dr)
        fz = mg * math.cos(th) * math.cos(phi) - sa * drag - ca * lift
        l_ = qbar * p.S_wing * p.b * (
            p.C_ell_0 + p.C_ell_beta * be + p.C_ell_p * p.b * P / (2 * Va) + p.C_ell_r * p.b * Rr / (2 * Va)
            + p.C_ell_delta_a * da + p.C_ell_delta_r * dr) - Qp
        m_ = qbar * p.S_wing * p.c * (p.C_m_0 + p.C_m_alpha * al + p.C_m_q * p.c * Q / (2 * Va) + p.C_m_delta_e * de)
        n_ = qbar * p.S_wing * p.b * (
            p.C_n_0 + p.C_n_beta * be + p.C_n_p * p.b * P / (2 * Va) + p.C_n_r * p.b * Rr / (2 * Va)
            + p.C_n_delta_a * da + p.C_n_delta_r * dr)
        ad.update({"thrust": Tp, "prop_torque": Qp, "lift": lift, "drag": drag})
        return np.array([fx, fy, fz]), np.array([l_, m_, n_]), ad

    # ---------------------------------------------------------------- hızlı skaler yol
    def _air_fast(self, xl, t):
        """(R satırları, ur, vr, wr, Va) — skaler; xl: durum listesi."""
        phi, th, psi = xl[6], xl[7], xl[8]
        cph, sph, cth, sth = math.cos(phi), math.sin(phi), math.cos(th), math.sin(th)
        cps, sps = math.cos(psi), math.sin(psi)
        R = ((cth * cps, sph * sth * cps - cph * sps, cph * sth * cps + sph * sps),
             (cth * sps, sph * sth * sps + cph * cps, cph * sth * sps - sph * cps),
             (-sth, sph * cth, cph * cth))
        uu, vv, ww = xl[3], xl[4], xl[5]
        if self.wind is None:
            ur, vr, wr = uu, vv, ww
        else:
            wn, we, wd, gu, gv, gw = self._wind6(t).tolist()
            ur = uu - (R[0][0] * wn + R[1][0] * we + R[2][0] * wd + gu)
            vr = vv - (R[0][1] * wn + R[1][1] * we + R[2][1] * wd + gv)
            wr = ww - (R[0][2] * wn + R[1][2] * we + R[2][2] * wd + gw)
        Va = math.sqrt(ur * ur + vr * vr + wr * wr)
        if Va <= 0:
            raise ControlError("Hava hızı sıfır: α ve β tanımsız.")
        return R, ur, vr, wr, Va

    def output(self, x, u=None, t: float = 0.0) -> np.ndarray:
        """Çıkış vektörü (`output_names` sırasıyla): 12 durum + Va, α, β, γ, χ, Vg."""
        x = np.asarray(x, dtype=np.float64)
        xl = x.tolist()
        R, ur, vr, wr, Va = self._air_fast(xl, t)
        uu, vv, ww = xl[3], xl[4], xl[5]
        gn = R[0][0] * uu + R[0][1] * vv + R[0][2] * ww
        ge = R[1][0] * uu + R[1][1] * vv + R[1][2] * ww
        gd = R[2][0] * uu + R[2][1] * vv + R[2][2] * ww
        Vg = math.sqrt(gn * gn + ge * ge + gd * gd)
        gam = math.asin(max(-1.0, min(1.0, -gd / Vg))) if Vg > 0 else 0.0
        out = np.empty(18)
        out[:12] = x
        out[12:] = (Va, math.atan2(wr, ur), math.asin(max(-1.0, min(1.0, vr / Va))), gam, math.atan2(ge, gn), Vg)
        return out

    def derivatives(self, x, u, t: float = 0.0) -> np.ndarray:
        """ẋ = f(x, u, t) — skaler hızlı yol (`derivatives_reference` ile aynı denklemler; test edilir)."""
        p = self.params
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1 or x.size != 12:
            raise ControlError("Sabit kanat durumu 12 elemanlı olmalı.")
        uin = np.asarray(u, dtype=np.float64).ravel()
        if uin.size != 4:
            raise ControlError("Sabit kanat girişi [δe δa δr δt] 4 elemanlı olmalı.")
        de, da, dr, dt = uin.tolist()
        xl = x.tolist()
        R, ur, vr, wr, Va = self._air_fast(xl, t)
        uu, vv, ww, phi, th = xl[3], xl[4], xl[5], xl[6], xl[7]
        P, Q, Rr = xl[9], xl[10], xl[11]
        al = math.atan2(wr, ur)
        be = math.asin(max(-1.0, min(1.0, vr / Va)))
        qS = 0.5 * p.rho * Va * Va * p.S_wing
        ca, sa = math.cos(al), math.sin(al)
        c2V, b2V = p.c / (2 * Va), p.b / (2 * Va)
        lift = qS * (self.CL(al) + p.C_L_q * c2V * Q + p.C_L_delta_e * de)
        drag = qS * (self.CD(al) + p.C_D_q * c2V * Q + p.C_D_delta_e * de)
        Tp, Qp, _ = self.propeller(Va, dt)
        mg = p.mass * p.gravity
        fx = R[2][0] * mg - ca * drag + sa * lift + Tp          # R[2][0]·mg = −mg·sinθ
        fy = R[2][1] * mg + qS * (p.C_Y_0 + p.C_Y_beta * be + p.C_Y_p * b2V * P + p.C_Y_r * b2V * Rr
                                  + p.C_Y_delta_a * da + p.C_Y_delta_r * dr)
        fz = R[2][2] * mg - sa * drag - ca * lift
        l_ = qS * p.b * (p.C_ell_0 + p.C_ell_beta * be + p.C_ell_p * b2V * P + p.C_ell_r * b2V * Rr
                         + p.C_ell_delta_a * da + p.C_ell_delta_r * dr) - Qp
        m_ = qS * p.c * (p.C_m_0 + p.C_m_alpha * al + p.C_m_q * c2V * Q + p.C_m_delta_e * de)
        n_ = qS * p.b * (p.C_n_0 + p.C_n_beta * be + p.C_n_p * b2V * P + p.C_n_r * b2V * Rr
                         + p.C_n_delta_a * da + p.C_n_delta_r * dr)
        G, G1, G2, G3, G4, G5, G6, G7, G8 = p.gammas()
        m = p.mass
        sph, cph = math.sin(phi), math.cos(phi)
        cth, tth = math.cos(th), math.tan(th)
        qs = Q * sph + Rr * cph
        return np.array([
            R[0][0] * uu + R[0][1] * vv + R[0][2] * ww,
            R[1][0] * uu + R[1][1] * vv + R[1][2] * ww,
            R[2][0] * uu + R[2][1] * vv + R[2][2] * ww,
            Rr * vv - Q * ww + fx / m, P * ww - Rr * uu + fy / m, Q * uu - P * vv + fz / m,
            P + qs * tth, Q * cph - Rr * sph, qs / cth,
            G1 * P * Q - G2 * Q * Rr + G3 * l_ + G4 * n_,
            G5 * P * Rr - G6 * (P * P - Rr * Rr) + m_ / p.Jy,
            G7 * P * Q - G1 * Q * Rr + G4 * l_ + G8 * n_])

    def derivatives_reference(self, x, u, t: float = 0.0) -> np.ndarray:
        """Matris biçimli başvuru uygulaması (forces_moments + döndürme matrisleri)."""
        p = self.params
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size != 12:
            raise ControlError("Sabit kanat durumu 12 elemanlı olmalı.")
        f, mom, _ = self.forces_moments(x, u, t)
        uu, vv, ww = x[3:6]
        phi, th, psi = x[6:9]
        P, Q, Rr = x[9:12]
        G, G1, G2, G3, G4, G5, G6, G7, G8 = p.gammas()
        pos_dot = rotmat_from_euler(phi, th, psi) @ x[3:6]
        m = p.mass
        vel_dot = np.array([Rr * vv - Q * ww + f[0] / m, P * ww - Rr * uu + f[1] / m, Q * uu - P * vv + f[2] / m])
        eul_dot = euler_rate_matrix(phi, th) @ x[9:12]
        l_, m_, n_ = mom
        rate_dot = np.array([G1 * P * Q - G2 * Q * Rr + G3 * l_ + G4 * n_,
                             G5 * P * Rr - G6 * (P * P - Rr * Rr) + m_ / p.Jy,
                             G7 * P * Q - G1 * Q * Rr + G4 * l_ + G8 * n_])
        return np.concatenate((pos_dot, vel_dot, eul_dot, rate_dot))

    def specific_force(self, x, u, t: float = 0.0) -> np.ndarray:
        """İvmeölçerin ölçtüğü özgül kuvvet (gövde, m/s²): (aerodinamik + itki) / m = (F − m·g·Rᵀe3) / m."""
        p = self.params
        x = np.asarray(x, dtype=np.float64)
        f, _, _ = self.forces_moments(x, u, t)
        g_b = rotmat_from_euler(x[6], x[7], x[8]).T @ np.array([0.0, 0.0, p.mass * p.gravity])
        return (f - g_b) / p.mass

    output_names = STATE_NAMES + ("Va", "alpha", "beta", "gamma", "chi", "Vg")

    def output_reference(self, x, u=None, t: float = 0.0) -> np.ndarray:
        """`output` ile aynı (air_data sözlüğü üzerinden; başvuru)."""
        x = np.asarray(x, dtype=np.float64)
        ad = self.air_data(x, t)
        return np.concatenate((x, [ad["Va"], ad["alpha"], ad["beta"], ad["gamma"], ad["chi"], ad["Vg"]]))

    def simulate(self, x0, u, t, **kwargs) -> SimResult:
        return simulate(self, x0, u, t, **kwargs)

    # ---------------------------------------------------------------- trim
    def _trim_completion(self, Va, alpha, beta, phi, theta, psidot):
        """(α, β, φ, θ) ve ψ̇*'dan durum ve kapalı biçimli girişler (δe, δt, δa, δr)."""
        p = self.params
        G, G1, G2, G3, G4, G5, G6, G7, G8 = p.gammas()
        P = -psidot * math.sin(theta)
        Q = psidot * math.sin(phi) * math.cos(theta)
        Rr = psidot * math.cos(phi) * math.cos(theta)
        uu = Va * math.cos(alpha) * math.cos(beta)
        vv = Va * math.sin(beta)
        ww = Va * math.sin(alpha) * math.cos(beta)
        qbar = 0.5 * p.rho * Va ** 2
        # δe: q̇ = 0
        m_req = -p.Jy * (G5 * P * Rr - G6 * (P * P - Rr * Rr))
        de = (m_req / (qbar * p.S_wing * p.c) - p.C_m_0 - p.C_m_alpha * alpha - p.C_m_q * p.c * Q / (2 * Va)) \
            / p.C_m_delta_e
        # δt: u̇ = 0
        ca, sa = math.cos(alpha), math.sin(alpha)
        lift = qbar * p.S_wing * (self.CL(alpha) + p.C_L_q * p.c * Q / (2 * Va) + p.C_L_delta_e * de)
        drag = qbar * p.S_wing * (self.CD(alpha) + p.C_D_q * p.c * Q / (2 * Va) + p.C_D_delta_e * de)
        T_req = p.mass * (Q * ww - Rr * vv) + p.mass * p.gravity * math.sin(theta) + ca * drag - sa * lift
        dt = self._throttle_for_thrust(Va, T_req)
        # δa, δr: ṗ = ṙ = 0  → [Γ3 Γ4; Γ4 Γ8]·[l; n] = [−Γ1pq + Γ2qr; −Γ7pq + Γ1qr]
        ln = np.linalg.solve(np.array([[G3, G4], [G4, G8]]),
                             np.array([-G1 * P * Q + G2 * Q * Rr, -G7 * P * Q + G1 * Q * Rr]))
        _, Qp, _ = self.propeller(Va, dt)
        k_l, k_n = qbar * p.S_wing * p.b, qbar * p.S_wing * p.b
        rest_l = p.C_ell_0 + p.C_ell_beta * beta + p.C_ell_p * p.b * P / (2 * Va) + p.C_ell_r * p.b * Rr / (2 * Va)
        rest_n = p.C_n_0 + p.C_n_beta * beta + p.C_n_p * p.b * P / (2 * Va) + p.C_n_r * p.b * Rr / (2 * Va)
        rhs = np.array([(ln[0] + Qp) / k_l - rest_l, ln[1] / k_n - rest_n])
        da, dr = np.linalg.solve(np.array([[p.C_ell_delta_a, p.C_ell_delta_r], [p.C_n_delta_a, p.C_n_delta_r]]), rhs)
        x = np.array([0.0, 0.0, 0.0, uu, vv, ww, phi, theta, 0.0, P, Q, Rr])
        return x, np.array([de, da, dr, dt])

    def _throttle_for_thrust(self, Va: float, T_req: float) -> float:
        p = self.params
        if p.prop_model == "simple":
            arg = 2 * T_req / (p.rho * p.S_prop * p.C_prop) + Va ** 2
            if arg < 0:
                return 0.0 if arg > -1e-12 else -math.sqrt(-arg) / p.k_motor   # süzülmeden daha dik iniş: ters
            return math.sqrt(arg) / p.k_motor
        from scipy.optimize import brentq
        f = lambda d: self.propeller(Va, d)[0] - T_req  # noqa: E731
        lo, hi = 0.0, 1.0
        while f(hi) < 0 and hi < 8.0:
            hi *= 1.5
        while f(lo) > 0 and lo > -2.0:
            lo -= 0.25
        if f(lo) * f(hi) > 0:
            raise ControlError("Trim için gerekli itki pervane modelinin erişebileceği aralıkta değil.")
        return brentq(f, lo, hi, xtol=1e-15, rtol=4 * np.finfo(float).eps, maxiter=200)

    def trim(self, Va: float, gamma: float = 0.0, R: float = math.inf, *, beta: float = 0.0, altitude: float = 100.0,
             psi: float = 0.0, x_guess=None) -> TrimResult:
        """Kararlı uçuş trimi (Va m/s, γ rad, R m: yatay dönüş yarıçapı, +sağa / −sola, inf = düz)."""
        if Va <= 0:
            raise ControlError("Trim hava hızı pozitif olmalı.")
        if R == 0 or not (math.isinf(R) or abs(R) > 0):
            raise ControlError("Dönüş yarıçapı sıfır olamaz.")
        psidot = 0.0 if math.isinf(R) else Va * math.cos(gamma) / R
        g = self.params.gravity
        from scipy.optimize import root

        def resid(z):
            al, ph, th = z
            x, u = self._trim_completion(Va, al, beta, ph, th, psidot)
            d = self.derivatives(x, u)
            return np.array([d[4], d[5], -d[2] - Va * math.sin(gamma)])

        phi0 = math.atan(Va ** 2 * math.cos(gamma) / (g * R)) if not math.isinf(R) else 0.0
        guesses = [np.array(x_guess, float)] if x_guess is not None else []
        guesses += [np.array([a0, phi0, a0 + gamma]) for a0 in (0.05, 0.0, 0.1, -0.05, 0.2)]
        best = None
        for z0 in guesses:
            try:
                sol = root(resid, z0, method="hybr", tol=1e-14)
            except (ControlError, ValueError, ZeroDivisionError):
                continue
            r = np.max(np.abs(resid(sol.x))) if np.all(np.isfinite(sol.x)) else np.inf
            if best is None or r < best[1]:
                best = (sol.x, r)
            if r < 1e-10:
                break
        if best is None or not np.isfinite(best[1]):
            raise ControlError("Trim çözümü bulunamadı.")
        al, ph, th = best[0]
        x, u = self._trim_completion(Va, al, beta, ph, th, psidot)
        x[2] = -altitude
        x[8] = psi
        xdot = self.derivatives(x, u)
        residual = np.array([xdot[3], xdot[4], xdot[5], xdot[9], xdot[10], xdot[11], xdot[6], xdot[7],
                             xdot[8] - psidot, -xdot[2] - Va * math.sin(gamma)])
        ok = bool(np.max(np.abs(residual)) < 1e-8 and 0.0 <= u[3] <= 1.0)
        return TrimResult(x, u, float(Va), float(gamma), float(R), float(al), float(beta), float(ph), float(th),
                          float(psidot), xdot, residual, ok)

    # ---------------------------------------------------------------- doğrusallaştırma
    OUTPUTS = ("Va", "alpha", "beta", "gamma", "chi", "h", "Vg")

    def linearize(self, x0, u0, *, outputs=None, t: float = 0.0) -> LinearModel:
        """12 durumlu tam doğrusal model (Richardson'lı merkezi fark; bağıl doğruluk ~1e-10).
        outputs: None → C = I (çıkış = durumlar); ya da ad listesi (durum adları ve 'Va','alpha','beta','gamma',
        'chi','h','Vg')."""
        x0 = np.asarray(x0, dtype=np.float64).ravel()
        u0 = np.asarray(u0, dtype=np.float64).ravel()
        A = numjac(lambda xx: self.derivatives(xx, u0, t), x0, rel_step=1e-3, abs_step=1e-4)
        B = numjac(lambda uu: self.derivatives(x0, uu, t), u0, rel_step=1e-3, abs_step=1e-4)
        if outputs is None:
            outputs = STATE_NAMES
        outputs = tuple(outputs)

        def yfun(xx, uu):
            ad = self.air_data(xx, t)
            return np.array([xx[STATE_NAMES.index(o)] if o in STATE_NAMES else ad[o] for o in outputs])
        for o in outputs:
            if o not in STATE_NAMES and o not in self.OUTPUTS:
                raise ControlError(f"Bilinmeyen çıkış: {o}")
        C = numjac(lambda xx: yfun(xx, u0), x0, rel_step=1e-3, abs_step=1e-4)
        D = numjac(lambda uu: yfun(x0, uu), u0, rel_step=1e-3, abs_step=1e-4)
        return LinearModel(A, B, C, D, STATE_NAMES, INPUT_NAMES, outputs, x0, u0, yfun(x0, u0),
                           self.derivatives(x0, u0, t), {"method": "sayısal (Richardson)"})


# ============================================================================ boylamsal / yanal ayrıştırma
def _reduced(lin: LinearModel, names, inputs) -> LinearModel:
    idx = {"u": 3, "v": 4, "w": 5, "phi": 6, "theta": 7, "psi": 8, "p": 9, "q": 10, "r": 11, "h": 2}
    sgn = np.array([-1.0 if n == "h" else 1.0 for n in names])
    si = [idx[n] for n in names]
    ii = [INPUT_NAMES.index(n) for n in inputs]
    A = lin.A[np.ix_(si, si)] * np.outer(sgn, sgn)
    B = lin.B[np.ix_(si, ii)] * sgn[:, None]
    x0 = None if lin.x0 is None else lin.x0[si] * sgn
    other = [k for k in range(12) if k not in si and k not in (0, 1)]
    coupling = float(np.max(np.abs(lin.A[np.ix_(si, other)]))) if other else 0.0
    return LinearModel(A, B, np.eye(len(si)), np.zeros((len(si), len(ii))), tuple(names), tuple(inputs),
                       tuple(names), x0, None if lin.u0 is None else lin.u0[ii], x0, None,
                       {"coupling_max": coupling})


def longitudinal(lin: LinearModel) -> LinearModel:
    """Boylamsal model: durumlar [u, w, q, theta, h] (h = −pd), girişler [delta_e, delta_t].
    Tam Jacobian'ın alt bloğu (kitap Böl. 5'teki boylamsal model; yanal durumlarla bağlaşım atılır — kanat düz, β = 0
    trimde tam sıfırdır; info['coupling_max'] atılan en büyük bağlaşım terimi)."""
    return _reduced(lin, LON_STATES, ("delta_e", "delta_t"))


def lateral(lin: LinearModel) -> LinearModel:
    """Yanal model: durumlar [v, p, r, phi, psi], girişler [delta_a, delta_r] (kitap Böl. 5'teki yanal model)."""
    return _reduced(lin, LAT_STATES, ("delta_a", "delta_r"))


# ============================================================================ kitabın transfer fonksiyonu yaklaşımları
def transfer_function_coefficients(model: FixedWing, trim: TrimResult) -> dict:
    """Beard & McLain Böl. 5 transfer fonksiyonu katsayıları (trim noktasında):
      a_phi1 = −½ρVa²Sb·C_pp·b/(2Va),  a_phi2 = ½ρVa²Sb·C_pδa         φ/δa = a_phi2 / (s(s + a_phi1))
      a_beta1 = −ρVaS·C_Yβ/(2m),       a_beta2 = ρVaS·C_Yδr/(2m)        β/δr = a_beta2 / (s + a_beta1)
      a_theta1 = −ρVa²cS·C_mq·c/(2Jy·2Va), a_theta2 = −ρVa²cS·C_mα/(2Jy), a_theta3 = ρVa²cS·C_mδe/(2Jy)
                                                                        θ/δe = a_theta3 / (s² + a_theta1·s + a_theta2)
      a_V1 = ρVaS/m·[C_D(α*) + C_Dδe·δe*] − (1/m)·∂T_p/∂Va,  a_V2 = (1/m)·∂T_p/∂δt,  a_V3 = g·cos(θ* − α*)
                                                                        Va = (a_V2·δt − a_V3·θ) / (s + a_V1)
      χ/φ = (g/Vg)/s,  h/θ = Va/s.
    a_V1 kitapta doğrusal sürükleme ve basit pervaneyle yazılmıştır; burada seçili sürükleme / pervane modelinin
    tam türeviyle genelleştirilir (doğrusal + basit seçildiğinde kitap formülüyle birebir aynıdır). q* = 0 (düz uçuş)
    varsayımı kitaptaki gibidir; dönüş trimlerinde yaklaşık kalır."""
    p = model.params
    Va, al = trim.Va, trim.alpha
    G, G1, G2, G3, G4, G5, G6, G7, G8 = p.gammas()
    C_pp = G3 * p.C_ell_p + G4 * p.C_n_p
    C_pda = G3 * p.C_ell_delta_a + G4 * p.C_n_delta_a
    qS = 0.5 * p.rho * Va ** 2 * p.S_wing
    dT_dVa, dT_ddt = model.propeller_partials(Va, trim.u[3])
    out = {
        "a_phi1": -qS * p.b * C_pp * p.b / (2 * Va),
        "a_phi2": qS * p.b * C_pda,
        "a_beta1": -p.rho * Va * p.S_wing * p.C_Y_beta / (2 * p.mass),
        "a_beta2": p.rho * Va * p.S_wing * p.C_Y_delta_r / (2 * p.mass),
        "a_theta1": -p.rho * Va ** 2 * p.c * p.S_wing * p.C_m_q * p.c / (2 * p.Jy * 2 * Va),
        "a_theta2": -p.rho * Va ** 2 * p.c * p.S_wing * p.C_m_alpha / (2 * p.Jy),
        "a_theta3": p.rho * Va ** 2 * p.c * p.S_wing * p.C_m_delta_e / (2 * p.Jy),
        "a_V1": p.rho * Va * p.S_wing / p.mass * (model.CD(al) + p.C_D_delta_e * trim.u[0]) - dT_dVa / p.mass,
        "a_V2": dT_ddt / p.mass,
        "a_V3": p.gravity * math.cos(trim.theta - al),
        "Va": Va, "Vg": Va, "g": p.gravity,
    }
    return out


def book_transfer_functions(coef: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(pay, payda) azalan kuvvetler."""
    return {
        "phi_delta_a": (np.array([coef["a_phi2"]]), np.array([1.0, coef["a_phi1"], 0.0])),
        "chi_phi": (np.array([coef["g"] / coef["Vg"]]), np.array([1.0, 0.0])),
        "beta_delta_r": (np.array([coef["a_beta2"]]), np.array([1.0, coef["a_beta1"]])),
        "theta_delta_e": (np.array([coef["a_theta3"]]), np.array([1.0, coef["a_theta1"], coef["a_theta2"]])),
        "h_theta": (np.array([coef["Va"]]), np.array([1.0, 0.0])),
        "Va_delta_t": (np.array([coef["a_V2"]]), np.array([1.0, coef["a_V1"]])),
        "Va_theta": (np.array([-coef["a_V3"]]), np.array([1.0, coef["a_V1"]])),
    }


# ============================================================================ mod çözümlemesi
def _describe(lam: np.ndarray) -> dict:
    lam = np.asarray(lam, dtype=complex)
    l0 = lam[np.argmax(lam.imag)] if lam.size > 1 else lam[0]
    wn = float(abs(l0))
    re = float(l0.real)
    d = {"eigenvalues": lam, "wn": wn, "zeta": (-re / wn) if wn > 0 else float("nan")}
    if abs(l0.imag) > 0:
        d["period"] = 2 * math.pi / abs(l0.imag)
    else:
        d["time_constant"] = (-1.0 / re) if re < 0 else float("inf")
    if re < 0:
        d["time_to_half"] = math.log(2) / -re
    elif re > 0:
        d["time_to_double"] = math.log(2) / re
    return d


def mode_analysis(lon: LinearModel, lat: LinearModel) -> dict:
    """Boylamsal (kısa periyot, fugoid) ve yanal (yuvarlanma, spiral, Hollanda yuvarlanması) kipleri.
    Sıfır özdeğerler (h ve ψ tümleyicileri) atılır. Boylamsal: büyüklüğe göre büyük çift kısa periyot, küçük çift
    fugoid. Yanal: karmaşık çift Hollanda yuvarlanması; gerçellerden büyüğü yuvarlanma, küçüğü spiral.
    Her kip: özdeğerler, ωn (rad/s), ζ, periyot ya da zaman sabiti, yarılanma / ikiye katlanma süresi."""
    def drop_zero(A):
        ev = np.linalg.eigvals(A)
        k = int(np.argmin(np.abs(ev)))
        return np.delete(ev, k)

    ev = drop_zero(lon.A)
    ev = ev[np.argsort(-np.abs(ev))]
    out = {"short_period": _describe(ev[:2]), "phugoid": _describe(ev[2:])}
    ev = drop_zero(lat.A)
    cplx = ev[np.abs(ev.imag) > 1e-9 * np.max(np.abs(ev))]
    real = np.sort(ev[np.abs(ev.imag) <= 1e-9 * np.max(np.abs(ev))].real)
    if cplx.size == 2 and real.size == 2:
        rs = real[np.argsort(-np.abs(real))]
        out.update(dutch_roll=_describe(cplx), roll=_describe(rs[:1]), spiral=_describe(rs[1:]))
    else:
        evs = ev[np.argsort(-np.abs(ev))]
        out.update(roll=_describe(evs[:1]), dutch_roll=_describe(evs[1:3]), spiral=_describe(evs[3:]))
    return out


def wind_function(steady_ned=(0.0, 0.0, 0.0), gust=None):
    """Sabit NED rüzgarı ve isteğe bağlı gövde sapması (ör. DrydenGust.as_function) birleştirir: t → (6,)."""
    s = np.asarray(steady_ned, dtype=np.float64)

    def w(t):
        g = np.zeros(3) if gust is None else np.asarray(gust(t), dtype=np.float64)
        return np.concatenate((s, g))
    return w
