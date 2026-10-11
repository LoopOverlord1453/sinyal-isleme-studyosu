"""Otopilot tasarım yardımcıları: kazançlar doğrusallaştırılmış modellerden kütüphane fonksiyonlarıyla hesaplanır.

Dörtrotor kademeli (PX4 tarzı) otopilot — `design_quad_cascade`
-----------------------------------------------------------------
Her çevrim, içteki kapalı çevrimi içeren **sürekli** modelde `spt.control.classical.pidtune` ile (hedef geçiş
frekansı wc ve faz payı tam sağlanır) tasarlanır, sonra ayrık PID bloklarıyla (ileri Euler) uygulanır. Örnekleme
hızları geçiş frekanslarının ≥ 50 katı olduğundan ayrıklaştırma etkisi küçüktür; şablon benzetimle doğrulanır.

  1. Açısal hız (roll / pitch / yaw): G = 1/(J·s·(τm·s + 1))  (Mx_c → p; motor gecikmesi askıda sanal koordinatlarda
     tam — bkz. hover_small_angle_model). roll/pitch PIDF, yaw PI.
  2. Açı: G = T_hız(s)/s (φ̇ ≈ p, küçük açı) → P.
  3. Yatay hız: ivme komutu → (İvme → tutum bloğu: θ ≈ −a_n/g) → açı kapalı çevrimi → a → v: G = T_açı(s)/s → PI.
  4. Konum: G = T_hız(s)/s → P.
  5. İrtifa (pd): dikey ivme komutu a_d → T = m·(g − a_d) (yerçekimi ileri beslemesi) → motor gecikmesi.
     Kademeli (PX4 gibi): dikey hız G = 1/(s·(τm·s + 1)) → PI; irtifa G = T_vz(s)/s → P. Tek çevrim seçeneği
     (LQR şablonunda): G = 1/(s²·(τm·s + 1)) → PIDF (Ti = 4·Td kuralı çift yavaş sıfır verir: referans adımında
     aşım ya da yavaş kuyruk — bu yüzden kademeli yapı önerilir).

LQR tutum — `design_quad_lqr_attitude`: askı küçük açı modelinin tutum alt sistemi (motor gecikmesi sanal koordinatta)
[φ θ ψ p q r Mx My Mz], girişler [Mx_c My_c Mz_c]; ZOH ile ayrıklaştırılıp `dlqr` (Bryson ağırlıkları).

Sabit kanat ardışık döngü kapama — `design_fixedwing_slc`: Beard & McLain Böl. 6 formülleri, trim çevresindeki
transfer fonksiyonu katsayılarından (a_φ1,2, a_β1,2, a_θ1..3, a_V1,2): roll ve pitch kazançları en büyük dümen
açısı / en büyük hata oranıyla, dış çevrimler bant genişliği ayrımı W ile.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from spt.control.errors import ControlError
from spt.control.uav.common import GRAVITY
from spt.control.uav.quadrotor import QuadrotorParams


# ============================================================================ yardımcılar
def _tf(num, den):
    from spt.control.lti import tf
    return tf(list(map(float, num)), list(map(float, den)))


def _pid_gains(C) -> dict:
    """Kütüphane PID nesnesi → blok parametreleri (paralel biçim: P + I/s + D·N·s/(s + N), N = 1/Tf)."""
    Kp, Ki, Kd, Tf = float(C.Kp), float(C.Ki), float(C.Kd), float(C.Tf)
    out = {"P": Kp, "I": Ki, "D": Kd, "N": (1.0 / Tf) if (Kd and Tf > 0) else 100.0}
    out["controller"] = ("P" if Kp else "") + ("I" if Ki else "") + ("D" if Kd else "")
    return out


def _tune(G, typ: str, wc: float, pm: float):
    from spt.control.classical import pidtune
    C, info = pidtune(G, typ, wc, PhaseMargin=pm)
    if not info.get("Stable", True):
        raise ControlError(f"{typ} tasarımı kararsız kapalı çevrim verdi (wc = {wc:g} rad/s).")
    return C, info


def _closed(C, G):
    from spt.control.connect import feedback
    return feedback(C * G, 1.0)


# ============================================================================ dörtrotor kademeli otopilot
@dataclass
class QuadCascadeDesign:
    """Kademeli otopilot kazançları (blok parametreleri) ve tasarım bilgileri."""
    rate: dict                 # {"roll": {...}, "pitch": {...}, "yaw": {...}} → P, I, D, N, controller
    angle: dict                # {"roll": Kp, "pitch": Kp, "yaw": Kp}
    velocity: dict             # PI: P, I (yatay; N ve E aynı)
    position: float            # P
    altitude: dict             # tek çevrim PIDF (pd hatası → a_d): P, I, D, N
    wc: dict                   # hedef geçiş frekansları (rad/s)
    info: dict = field(default_factory=dict)
    alt_position: float = 0.0  # kademeli irtifa: pd P → v_d referansı
    alt_velocity: dict = field(default_factory=dict)   # dikey hız PI → a_d
    Ts: dict = field(default_factory=lambda: {"rate": 0.002, "angle": 0.004, "outer": 0.02})
    limits: dict = field(default_factory=dict)


DEFAULT_WC = {"rate_rp": 30.0, "rate_yaw": 12.0, "angle_rp": 8.0, "angle_yaw": 3.0, "velocity": 2.0,
              "position": 0.7, "altitude": 3.0, "alt_velocity": 4.0, "alt_position": 1.2}
DEFAULT_PM = {"rate": 60.0, "angle": 60.0, "velocity": 70.0, "position": 60.0, "altitude": 60.0, "alt_velocity": 70.0}


def design_quad_cascade(params: QuadrotorParams | None = None, *, wc: dict | None = None,
                        phase_margin: dict | None = None) -> QuadCascadeDesign:
    """Kademeli dörtrotor otopilotu kazançları (modül belgesindeki sıra). wc: DEFAULT_WC, phase_margin: DEFAULT_PM
    anahtarlarıyla (P denetleyicilerde yalnız wc tutturulur; faz payı sonuçtur). Kazançlar PID bloklarına yazılır."""
    p = params if params is not None else QuadrotorParams()
    w = dict(DEFAULT_WC)
    w.update(wc or {})
    pm = dict(DEFAULT_PM)
    pm.update(phase_margin or {})
    J = np.diag(np.asarray(p.inertia, dtype=float)) if np.ndim(p.inertia) == 2 else np.asarray(p.inertia)
    tau = p.tau_m
    integ = _tf([1.0], [1.0, 0.0])
    rate, angle, info, T_ang = {}, {}, {}, {}
    for k, (axis, Jii) in enumerate((("roll", J[0]), ("pitch", J[1]), ("yaw", J[2]))):
        G = _tf([1.0 / Jii], [tau, 1.0, 0.0])
        typ, wcr, wca = ("PIDF", w["rate_rp"], w["angle_rp"]) if axis != "yaw" else ("PI", w["rate_yaw"],
                                                                                      w["angle_yaw"])
        Cr, ir = _tune(G, typ, wcr, pm["rate"])
        rate[axis] = _pid_gains(Cr)
        Tr = _closed(Cr, G)
        Ga = Tr * integ
        Ca, ia = _tune(Ga, "P", wca, pm["angle"])
        angle[axis] = float(Ca.Kp)
        T_ang[axis] = _closed(Ca, Ga)
        info[axis] = {"rate": ir, "angle": ia}
    Gv = T_ang["pitch"] * integ                       # θ ekseni (Jy ≥ Jx: daha yavaş eksen) üzerinden
    Cv, iv = _tune(Gv, "PI", w["velocity"], pm["velocity"])
    Gp = _closed(Cv, Gv) * integ
    Cp, ip = _tune(Gp, "P", w["position"], pm["position"])
    Gh = _tf([1.0], [tau, 1.0, 0.0, 0.0])
    Ch, ih = _tune(Gh, "PIDF", w["altitude"], pm["altitude"])
    Gvz = _tf([1.0], [tau, 1.0, 0.0])                # a_d → v_d (motor gecikmeli)
    Cvz, ivz = _tune(Gvz, "PI", w["alt_velocity"], pm["alt_velocity"])
    Gpz = _closed(Cvz, Gvz) * integ
    Cpz, ipz = _tune(Gpz, "P", w["alt_position"], pm["position"])
    info.update(velocity=iv, position=ip, altitude=ih, alt_velocity=ivz, alt_position=ipz,
                phase_margin_targets=pm)
    tmax = 4 * p.kT * p.omega_max ** 2
    limits = {"rate_cmd_rp": 4.0, "rate_cmd_yaw": 2.0, "moment_rp": 1.5, "moment_yaw": 0.15, "accel_h": 5.0,
              "accel_v": 5.0, "velocity": 3.0, "max_tilt": math.radians(35.0), "T_max": tmax}
    return QuadCascadeDesign(rate, angle, {"P": float(Cv.Kp), "I": float(Cv.Ki)}, float(Cp.Kp), _pid_gains(Ch), w,
                             info, alt_position=float(Cpz.Kp), alt_velocity={"P": float(Cvz.Kp), "I": float(Cvz.Ki)},
                             limits=limits)


# ============================================================================ LQR tutum
def quad_attitude_model(params: QuadrotorParams | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Askı tutum alt sistemi (sürekli): x = [φ θ ψ p q r Mx My Mz], u = [Mx_c My_c Mz_c];
    η̇ = ω, ω̇ = J⁻¹·M, Ṁ = (M_c − M)/τm (jiroskopik etkisiz; hover_small_angle_model'in alt bloğu)."""
    p = params if params is not None else QuadrotorParams()
    J = np.asarray(p.inertia, dtype=float)
    J = np.diag(J) if J.ndim == 1 else J
    A = np.zeros((9, 9))
    A[0:3, 3:6] = np.eye(3)
    A[3:6, 6:9] = np.linalg.inv(J)
    A[6:9, 6:9] = -np.eye(3) / p.tau_m
    B = np.zeros((9, 3))
    B[6:9, :] = np.eye(3) / p.tau_m
    return A, B


def design_quad_lqr_attitude(params: QuadrotorParams | None = None, *, Ts: float = 0.004,
                             angle_max=(0.1, 0.1, 0.2), rate_max=(1.0, 1.0, 1.0), moment_max=(1.0, 1.0, 0.1),
                             command_max=(0.5, 0.5, 0.08)) -> dict:
    """Ayrık LQR tutum denetleyicisi (Bryson kuralı: Q = diag(1/x_max²), R = diag(1/u_max²)).
    Dönüş: K (3×9), Nr = K[:, :3] (u = Nr·η_ref − K·x: açı referansında kalıcı hatasız), Ad, Bd, A, B, Q, R,
    kapalı çevrim kutupları eig(Ad − Bd·K)."""
    from spt.control.discretize import c2d
    from spt.control.lti import ss
    from spt.control.modern import dlqr
    A, B = quad_attitude_model(params)
    sysd = c2d(ss(A, B, np.eye(9), np.zeros((9, 3))), Ts, "zoh")
    Ad, Bd = np.asarray(sysd.A, float), np.asarray(sysd.B, float)
    Q = np.diag(1.0 / np.square(np.concatenate((angle_max, rate_max, moment_max))))
    R = np.diag(1.0 / np.square(np.asarray(command_max, float)))
    K, S, e = dlqr(Ad, Bd, Q, R)
    K = np.asarray(K, float)
    return {"K": K, "Nr": K[:, :3].copy(), "Ad": Ad, "Bd": Bd, "A": A, "B": B, "Q": Q, "R": R, "Ts": Ts,
            "poles": np.asarray(e), "S": np.asarray(S)}


# ============================================================================ MPC tahmin modeli
def quad_mpc_model(tau_h: float, tau_v: float, *, measure_velocity: bool = True)         -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """MPC tahmin modeli (sürekli): eksen başına [p, v, a], ṗ = v, v̇ = a, ȧ = (a_c − a)/τ; yatay eksenlerde τ_h
    (iç tutum çevriminin eşdeğer zaman sabiti), dikeyde τ_v (motor gecikmesi). Durumlar [pn vn an pe ve ae pd vd ad],
    girişler [a_n a_e a_d] (m/s²), çıkışlar [pn pe pd] (+ measure_velocity ise [vn ve vd])."""
    A = np.zeros((9, 9))
    B = np.zeros((9, 3))
    ny = 6 if measure_velocity else 3
    C = np.zeros((ny, 9))
    for k, tau in enumerate((tau_h, tau_h, tau_v)):
        i = 3 * k
        A[i, i + 1] = 1.0
        A[i + 1, i + 2] = 1.0
        A[i + 2, i + 2] = -1.0 / tau
        B[i + 2, k] = 1.0 / tau
        C[k, i] = 1.0
        if measure_velocity:
            C[3 + k, i + 1] = 1.0
    return A, B, C, np.zeros((ny, 3))


# ============================================================================ sabit kanat ardışık döngü kapama
DEFAULT_FW_SPECS = {
    "roll": {"delta_max": math.radians(45.0), "e_max": math.radians(15.0), "zeta": 0.707},
    "course": {"W": 15.0, "zeta": 0.9},
    "sideslip": {"delta_max": math.radians(45.0), "e_max": math.radians(15.0), "zeta": 0.707},
    "pitch": {"delta_max": math.radians(45.0), "e_max": math.radians(10.0), "zeta": 0.707},
    "altitude": {"W": 15.0, "zeta": 1.0},
    "airspeed": {"wn": 1.5, "zeta": 0.9},
}


def design_fixedwing_slc(model, trim, specs: dict | None = None) -> dict:
    """Beard & McLain Böl. 6 kazançları:
      roll:     kp_φ = δa_max/e_φ_max·sign(a_φ2), ωn_φ = √(|a_φ2|·|kp_φ|), kd_φ = (2ζ_φωn_φ − a_φ1)/a_φ2
                δa = kp_φ(φ_c − φ) − kd_φ·p
      course:   ωn_χ = ωn_φ/W_χ, kp_χ = 2ζ_χωn_χ·Vg/g, ki_χ = ωn_χ²·Vg/g           φ_c = kp_χ·e_χ + ki_χ∫e_χ
      sideslip: kp_β = δr_max/e_β_max·sign(a_β2), ki_β = ((a_β1 + a_β2·kp_β)/(2ζ_β))²/a_β2   δr = −kp_β·β − ki_β∫β
      pitch:    kp_θ = δe_max/e_θ_max·sign(a_θ3), ωn_θ = √(a_θ2 + kp_θ·a_θ3), kd_θ = (2ζ_θωn_θ − a_θ1)/a_θ3,
                K_θDC = kp_θ·a_θ3/(a_θ2 + kp_θ·a_θ3)                                 δe = kp_θ(θ_c − θ) − kd_θ·q
      altitude: ωn_h = ωn_θ/W_h, ki_h = ωn_h²/(K_θDC·Va), kp_h = 2ζ_hωn_h/(K_θDC·Va)  θ_c = kp_h·e_h + ki_h∫e_h
      airspeed: ki_V = ωn_V²/a_V2, kp_V = (2ζ_Vωn_V − a_V1)/a_V2                     δt = δt* + kp_V·e_V + ki_V∫e_V
    Dönüş: kazançlar, katsayılar, kipler (ωn) ve kullanılan özellikler."""
    from spt.control.uav.fixedwing import transfer_function_coefficients
    s = {k: dict(v) for k, v in DEFAULT_FW_SPECS.items()}
    for k, v in (specs or {}).items():
        s.setdefault(k, {}).update(v)
    c = transfer_function_coefficients(model, trim)
    g, Va, Vg = c["g"], c["Va"], c["Vg"]
    r = s["roll"]
    kp_phi = r["delta_max"] / r["e_max"] * math.copysign(1.0, c["a_phi2"])
    wn_phi = math.sqrt(abs(c["a_phi2"]) * abs(kp_phi))
    kd_phi = (2 * r["zeta"] * wn_phi - c["a_phi1"]) / c["a_phi2"]
    wn_chi = wn_phi / s["course"]["W"]
    kp_chi = 2 * s["course"]["zeta"] * wn_chi * Vg / g
    ki_chi = wn_chi ** 2 * Vg / g
    b = s["sideslip"]
    kp_beta = b["delta_max"] / b["e_max"] * math.copysign(1.0, c["a_beta2"])
    ki_beta = ((c["a_beta1"] + c["a_beta2"] * kp_beta) / (2 * b["zeta"])) ** 2 / c["a_beta2"]
    pt = s["pitch"]
    kp_th = pt["delta_max"] / pt["e_max"] * math.copysign(1.0, c["a_theta3"])
    wn2 = c["a_theta2"] + kp_th * c["a_theta3"]
    if wn2 <= 0:
        raise ControlError("Pitch çevrimi için seçilen kazanç kararlı bir doğal frekans vermiyor.")
    wn_th = math.sqrt(wn2)
    kd_th = (2 * pt["zeta"] * wn_th - c["a_theta1"]) / c["a_theta3"]
    K_dc = kp_th * c["a_theta3"] / wn2
    wn_h = wn_th / s["altitude"]["W"]
    ki_h = wn_h ** 2 / (K_dc * Va)
    kp_h = 2 * s["altitude"]["zeta"] * wn_h / (K_dc * Va)
    a = s["airspeed"]
    ki_V = a["wn"] ** 2 / c["a_V2"]
    kp_V = (2 * a["zeta"] * a["wn"] - c["a_V1"]) / c["a_V2"]
    return {
        "kp_phi": kp_phi, "kd_phi": kd_phi, "kp_chi": kp_chi, "ki_chi": ki_chi, "kp_beta": kp_beta,
        "ki_beta": ki_beta, "kp_theta": kp_th, "kd_theta": kd_th, "K_theta_DC": K_dc, "kp_h": kp_h, "ki_h": ki_h,
        "kp_V": float(kp_V), "ki_V": float(ki_V),
        "wn": {"roll": wn_phi, "course": wn_chi, "pitch": wn_th, "altitude": wn_h, "airspeed": a["wn"],
               "sideslip": math.sqrt(c["a_beta2"] * ki_beta)},
        "coef": c, "specs": s,
    }


# ============================================================================ başarım ölçütleri
def step_metrics(t, y, *, t_step: float, y0: float, y1: float, band: float = 0.02) -> dict:
    """Basamak cevabı ölçütleri (adım anından itibaren): aşım (%; y0 → y1 yönünde), yükselme (%10 → %90),
    yerleşme (|y − y1| ≤ band·|y1 − y0| bandına son giriş), son hata (son örnek)."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    k = t >= t_step - 1e-12
    tt, yy = t[k] - t_step, y[k]
    amp = y1 - y0
    if amp == 0:
        raise ControlError("Basamak genliği sıfır.")
    z = (yy - y0) / amp
    over = max(0.0, float(z.max()) - 1.0) * 100.0
    def first(level):
        idx = np.nonzero(z >= level)[0]
        return float(tt[idx[0]]) if idx.size else math.inf
    rise = first(0.9) - first(0.1)
    outside = np.nonzero(np.abs(yy - y1) > band * abs(amp))[0]
    if outside.size == 0:
        settle = 0.0
    elif outside[-1] == yy.size - 1:
        settle = math.inf
    else:
        settle = float(tt[outside[-1] + 1])
    return {"overshoot": over, "rise_time": rise, "settling_time": settle, "final_error": float(yy[-1] - y1),
            "peak": float(yy[np.argmax(z)])}


__all__ = ["QuadCascadeDesign", "DEFAULT_WC", "DEFAULT_PM", "design_quad_cascade", "quad_attitude_model", "design_quad_lqr_attitude",
           "quad_mpc_model", "DEFAULT_FW_SPECS", "design_fixedwing_slc", "step_metrics", "GRAVITY"]
