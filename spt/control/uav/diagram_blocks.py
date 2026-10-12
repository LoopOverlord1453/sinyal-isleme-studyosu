"""İHA blok diyagram blokları (eklenti): dörtrotor ve sabit kanat bitkileri, karıştırıcı / ters karıştırıcı, ivme →
tutum + itki dönüşümü, doğrusal MPC ve algılayıcılar (IMU, barometre, GPS).

`spt.control.diagram.blocks` bu modülü kendiliğinden yükler (PLUGIN_MODULES); bloklar `@register_block` ile kaydolur.
Bitkiler sürekli zamanlıdır; doğrusallaştırma için analitik (dörtrotor) ya da yüksek doğruluklu Richardson (sabit
kanat) Jacobian'ı `linearize` ile verilir. Koordinatlar NED / FRD, açılar radyan (bkz. spt/control/uav/common.py).
"""
from __future__ import annotations

import math

import numpy as np

from spt.control.diagram.block import Block, Param, p_ts, register_block, CONTINUOUS_TS
from spt.control.diagram.blocks.plants import DiscreteController
from spt.control.errors import ControlError
from spt.control.uav.common import DRYDEN_PRESETS, DrydenGust, numjac
from spt.control.uav.fixedwing import FixedWing, FixedWingParams
from spt.control.uav.quadrotor import Mixer, Quadrotor, QuadrotorParams

_GROUP_BODY, _GROUP_ROTOR, _GROUP_ENV, _GROUP_IN, _GROUP_INIT = "Gövde", "Rotor / motor", "Sürtünme / rüzgar", \
    "Giriş", "Başlangıç"


def _quad_params_from(p: dict) -> QuadrotorParams:
    J = np.asarray(p["inertia"], dtype=float).ravel()
    if J.size == 3:
        J = np.diag(J)
    elif J.size == 9:
        J = J.reshape(3, 3)
    else:
        raise ControlError("Eylemsizlik 3 (köşegen) ya da 9 (3×3 satır sıralı) elemanlı olmalı.")
    return QuadrotorParams(mass=p["mass"], inertia=J, arm_length=p["arm_length"], config=p["config"], kT=p["kT"],
                           kQ=p["kQ"], tau_m=p["tau_m"], cr=p["cr"], wb=p["wb"], omega_min=p["omega_min"],
                           omega_max=p["omega_max"], Jr=p["Jr"], gyro=p["gyro"], drag=tuple(p["drag"]),
                           rot_drag=tuple(p["rot_drag"]))


def _geometry_params(group: str = _GROUP_BODY) -> tuple[Param, ...]:
    d = QuadrotorParams()
    return (
        Param("config", "choice", "x", "Yapılandırma", choices={"x": "X (PX4 sırası)", "+": "+ (PX4 sırası)"},
              hint="X: 1 ön-sağ CCW, 2 arka-sol CCW, 3 ön-sol CW, 4 arka-sağ CW. +: 1 sağ, 2 sol, 3 ön, 4 arka.",
              group=group),
        Param("arm_length", "float", d.arm_length, "Kol uzunluğu", unit="m", min=0.0, strict_min=True, group=group),
        Param("kT", "float", d.kT, "İtki katsayısı kT", unit="N/(rad/s)²", min=0.0, strict_min=True,
              hint="T = kT·Ω²", group=_GROUP_ROTOR),
        Param("kQ", "float", d.kQ, "Moment katsayısı kQ", unit="N·m/(rad/s)²", min=0.0, hint="Q = kQ·Ω²",
              group=_GROUP_ROTOR),
        Param("omega_min", "float", d.omega_min, "En küçük motor hızı", unit="rad/s", min=0.0, group=_GROUP_ROTOR),
        Param("omega_max", "float", d.omega_max, "En büyük motor hızı", unit="rad/s", min=0.0, strict_min=True,
              group=_GROUP_ROTOR),
        Param("cr", "float", d.cr, "Gaz → hız eğimi C_R", unit="rad/s", hint="Ω_ss = C_R·u + ϖ_b (u: 0–1 gaz)",
              group=_GROUP_ROTOR),
        Param("wb", "float", d.wb, "Gaz → hız kayması ϖ_b", unit="rad/s", group=_GROUP_ROTOR),
    )


# ============================================================================ dörtrotor bitkisi
@register_block
class QuadrotorPlant(Block):
    type_name = "UAVQuadrotor"
    display_name = "Dörtrotor (6-DOF)"
    category = "plants"
    description = ("Dörtrotor katı cisim modeli (NED, kuaterniyonlu), 1. derece motor dinamiği, isteğe bağlı sürtünme / "
                   "jiroskopik etki. Giriş: motor hız komutları (rad/s), gaz (0–1) ya da sanal [T, Mx, My, Mz]. "
                   "Çıkışlar: konum (NED), hız (NED), Euler [φ θ ψ], gövde açısal hızları [p q r], motor hızları, "
                   "ivmeölçer özgül kuvveti (gövde).")
    simulink = "UAV Toolbox / Aerospace Blockset 6DOF (Quaternion) + motor modeli"
    PARAMS = (
        Param("mass", "float", 1.4, "Kütle", unit="kg", min=0.0, strict_min=True, group=_GROUP_BODY),
        Param("inertia", "vector", [0.0211, 0.0219, 0.0366], "Eylemsizlik (Jx Jy Jz ya da 3×3)", unit="kg·m²",
              hint="3 eleman: köşegen; 9 eleman: 3×3 matris satır sıralı.", group=_GROUP_BODY),
        *_geometry_params(),
        Param("tau_m", "float", 0.0136, "Motor zaman sabiti", unit="s", min=0.0, strict_min=True, group=_GROUP_ROTOR),
        Param("motor_dynamics", "bool", True, "Motor dinamiği (1. derece)", group=_GROUP_ROTOR),
        Param("Jr", "float", 1.287e-4, "Rotor + pervane ataleti", unit="kg·m²", min=0.0, group=_GROUP_ROTOR),
        Param("gyro", "bool", False, "Rotor jiroskopik etkisi", group=_GROUP_ROTOR),
        Param("drag", "vector", [0.0, 0.0, 0.0], "Öteleme sürtünmesi (gövde x y z)", unit="N/(m/s)", min=0.0,
              group=_GROUP_ENV),
        Param("rot_drag", "vector", [0.0, 0.0, 0.0], "Dönme sürtünmesi (p q r)", unit="N·m/(rad/s)", min=0.0,
              group=_GROUP_ENV),
        Param("wind", "vector", [0.0, 0.0, 0.0], "Sabit rüzgar (NED)", unit="m/s", group=_GROUP_ENV),
        Param("input_mode", "choice", "omega", "Giriş türü",
              choices={"omega": "Motor hız komutu Ω_ref (rad/s)", "pwm": "Gaz 0–1", "virtual": "Sanal [T Mx My Mz]"},
              group=_GROUP_IN),
        Param("desaturation", "choice", "px4", "Doygunluk giderme (sanal girişte)",
              choices={"px4": "PX4 (yaw önce kesilir)", "px4_airmode": "PX4 airmode", "scale": "Ölçekle (Betaflight)",
                       "clip": "Kırp"}, group=_GROUP_IN),
        Param("position0", "vector", [0.0, 0.0, -10.0], "Başlangıç konumu (NED)", unit="m", group=_GROUP_INIT),
        Param("velocity0", "vector", [0.0, 0.0, 0.0], "Başlangıç hızı (NED)", unit="m/s", group=_GROUP_INIT),
        Param("euler0", "vector", [0.0, 0.0, 0.0], "Başlangıç Euler [φ θ ψ]", unit="rad", group=_GROUP_INIT),
        Param("rates0", "vector", [0.0, 0.0, 0.0], "Başlangıç açısal hızı [p q r]", unit="rad/s", group=_GROUP_INIT),
        Param("motors0", "vector", [], "Başlangıç motor hızları (boş: askı hızı)", unit="rad/s", group=_GROUP_INIT),
    )

    def check_params(self):
        p = self.params
        for k in ("drag", "rot_drag", "wind", "position0", "velocity0", "euler0", "rates0"):
            if np.asarray(p[k]).size != 3:
                self.error(f"'{self._param_defs[k].label}' 3 elemanlı olmalı.")
        try:
            self._quad = Quadrotor(_quad_params_from(p), input_mode=p["input_mode"],
                                   motor_dynamics=p["motor_dynamics"], desaturation=p["desaturation"],
                                   wind=tuple(np.asarray(p["wind"], float)))
            self._quad.hover_motor_speeds()
        except ControlError as exc:
            self.error(str(exc))
        m0 = np.asarray(p["motors0"], float)
        if m0.size not in (0, self._quad.n):
            self.error(f"başlangıç motor hızları boş ya da {self._quad.n} elemanlı olmalı.")

    @property
    def model(self) -> Quadrotor:
        return self._quad

    def sample_time(self):
        return CONTINUOUS_TS

    def allowed_sample_times(self):
        return "continuous"

    def n_inputs(self):
        return 1

    def n_outputs(self):
        return 6

    def input_labels(self):
        return [{"omega": "Ω_ref", "pwm": "gaz", "virtual": "[T Mx My Mz]"}[self.params["input_mode"]]]

    def output_labels(self):
        return ["konum", "hız", "Euler", "pqr", "Ω", "f_ivme"]

    def feedthrough(self):
        return [not self.params["motor_dynamics"]]

    def propagate_widths(self, in_widths, guess=False):
        return [3, 3, 3, 3, self._quad.n, 3]

    def setup(self, in_widths, out_widths):
        if in_widths[0] != self._quad.nu:
            self.error(f"giriş genişliği {self._quad.nu} olmalı ({in_widths[0]} bağlandı).")
        self.nx = self._quad.nx

    def x0(self):
        p = self.params
        m0 = np.asarray(p["motors0"], float)
        return self._quad.initial_state(p["position0"], p["velocity0"], p["euler0"], p["rates0"],
                                        m0 if m0.size else None)

    def output(self, t, x, xd, u):
        y = self._quad.output(x)
        f = self._quad.specific_force(x, u[0], t)
        n = self._quad.n
        return [y[0:3], y[3:6], y[6:9], y[9:12], y[12:12 + n] if self.params["motor_dynamics"] else
                self._quad.motor_reference(u[0]), f]

    def derivatives(self, t, x, xd, u):
        return self._quad.derivatives(x, u[0], t)

    def project(self, x):
        nq = float(np.linalg.norm(x[6:10]))
        return None if abs(nq - 1.0) < 1e-9 else self._quad.project(x)

    def linearize(self, t, x, xd, u):
        q = self._quad
        lin = q.linearize(x, u[0], attitude="quaternion", t=t)
        C, D = lin.C, lin.D
        if not self.params["motor_dynamics"]:          # Ω çıkışı = Ω_ref(u): D'ye taşınır
            C = np.vstack((C, np.zeros((q.n, C.shape[1]))))
            D = np.vstack((D, q._dref_du(u[0])))
        Cf = numjac(lambda xx: q.specific_force(xx, u[0], t), x)
        Df = numjac(lambda uu: q.specific_force(x, uu, t), np.asarray(u[0], float))
        return lin.A, lin.B, np.vstack((C, Cf)), np.vstack((D, Df))


# ============================================================================ sabit kanat bitkisi
@register_block
class FixedWingPlant(Block):
    type_name = "UAVFixedWing"
    display_name = "Sabit kanat (6-DOF)"
    category = "plants"
    description = ("Beard & McLain sabit kanat modeli (Aerosonde). Giriş [δe δa δr δt] (rad, gaz 0–1). Çıkışlar: konum "
                   "(NED), hız (gövde, yere göre), Euler, [p q r], hava verisi [Va α β γ χ Vg]. Başlangıç trimden.")
    simulink = "Aerospace Blockset 6DOF (Euler Angles) + aerodinamik"
    PARAMS = (
        Param("prop_model", "choice", "motor", "Pervane modeli",
              choices={"motor": "Motor + pervane (güncel Böl. 4)", "simple": "Basit (1. baskı)"}, group="Model"),
        Param("drag_model", "choice", "quadratic", "Sürükleme modeli",
              choices={"quadratic": "C_Dp + C_L²/(πeAR)", "linear": "C_D0 + C_Dα·α"}, group="Model"),
        Param("trim_init", "bool", True, "Trimden başlat", group=_GROUP_INIT),
        Param("Va0", "float", 25.0, "Trim hava hızı", unit="m/s", min=0.0, strict_min=True, group=_GROUP_INIT),
        Param("gamma0", "float", 0.0, "Trim uçuş yolu açısı", unit="rad", group=_GROUP_INIT),
        Param("h0", "float", 100.0, "Başlangıç irtifası", unit="m", group=_GROUP_INIT),
        Param("psi0", "float", 0.0, "Başlangıç yaw", unit="rad", group=_GROUP_INIT),
        Param("x0", "vector", [], "Başlangıç durumu (trim kapalıysa, 12)", group=_GROUP_INIT),
        Param("wind", "vector", [0.0, 0.0, 0.0], "Sabit rüzgar (NED)", unit="m/s", group=_GROUP_ENV),
        Param("gust", "choice", "none", "Türbülans (Dryden)",
              choices={"none": "Yok", **{k: k for k in DRYDEN_PRESETS}}, group=_GROUP_ENV),
        Param("gust_seed", "int", 1, "Türbülans tohumu", group=_GROUP_ENV),
        Param("gust_duration", "float", 120.0, "Türbülans süresi", unit="s", min=0.0, strict_min=True,
              hint="Bu süreden sonra son değer tutulur.", group=_GROUP_ENV),
    )

    def check_params(self):
        p = self.params
        if np.asarray(p["wind"]).size != 3:
            self.error("rüzgar 3 elemanlı olmalı.")
        self._fw = FixedWing(FixedWingParams.aerosonde(prop_model=p["prop_model"], drag_model=p["drag_model"]))
        x0 = np.asarray(p["x0"], float)
        if not p["trim_init"] and x0.size != 12:
            self.error("trim kapalıyken başlangıç durumu 12 elemanlı olmalı.")
        self._gust_fn = None

    @property
    def model(self) -> FixedWing:
        return self._fw

    def trim(self):
        p = self.params
        return self._fw.trim(p["Va0"], p["gamma0"], altitude=p["h0"], psi=p["psi0"])

    def sample_time(self):
        return CONTINUOUS_TS

    def allowed_sample_times(self):
        return "continuous"

    def n_outputs(self):
        return 5

    def input_labels(self):
        return ["[δe δa δr δt]"]

    def output_labels(self):
        return ["konum", "uvw", "Euler", "pqr", "hava"]

    def feedthrough(self):
        return [False]              # çıkışlar yalnız duruma bağlı

    def propagate_widths(self, in_widths, guess=False):
        return [3, 3, 3, 3, 6]

    def setup(self, in_widths, out_widths):
        if in_widths[0] != 4:
            self.error(f"giriş genişliği 4 olmalı ({in_widths[0]} bağlandı).")
        self.nx = 12

    def reset(self):
        p = self.params
        steady = np.asarray(p["wind"], float)
        gust = None
        if p["gust"] != "none":
            Va = p["Va0"] if p["trim_init"] else max(1.0, float(np.linalg.norm(np.asarray(p["x0"])[3:6])))
            gust = DrydenGust(Va, preset=p["gust"], seed=p["gust_seed"]).as_function(p["gust_duration"], 0.01)
        if gust is None and not np.any(steady):
            self._fw.wind = None
        else:
            self._fw.wind = (lambda t, s=steady, g=gust: np.concatenate((s, np.zeros(3) if g is None else g(t))))

    def x0(self):
        p = self.params
        if p["trim_init"]:
            return self.trim().x.copy()
        return np.asarray(p["x0"], float).copy()

    def output(self, t, x, xd, u):
        y = self._fw.output(x, None, t)
        return [y[0:3], y[3:6], y[6:9], y[9:12], y[12:18]]

    def derivatives(self, t, x, xd, u):
        return self._fw.derivatives(x, u[0], t)

    def linearize(self, t, x, xd, u):
        lin = self._fw.linearize(x, u[0], outputs=self._fw.output_names, t=t)
        return lin.A, lin.B, lin.C, lin.D


# ============================================================================ karıştırıcı
def _mixer_from(p: dict) -> tuple[Mixer, QuadrotorParams]:
    qp = QuadrotorParams(arm_length=p["arm_length"], config=p["config"], kT=p["kT"], kQ=p["kQ"],
                         omega_min=p["omega_min"], omega_max=p["omega_max"], cr=p["cr"], wb=p["wb"])
    pos, spins = qp.geometry()
    E = np.vstack((np.ones(pos.shape[0]), -pos[:, 1], pos[:, 0], spins * qp.kQ / qp.kT))
    return Mixer(E, qp.kT * qp.omega_min ** 2, qp.kT * qp.omega_max ** 2), qp


@register_block
class MixerBlock(Block):
    type_name = "UAVMixer"
    display_name = "Karıştırıcı (mixer)"
    category = "controllers"
    description = ("Sanal girişler [T Mx My Mz] → motor komutları; motor sınırlarında doygunluk giderme önceliğiyle "
                   "(PX4: roll/pitch > itki > yaw; Betaflight benzeri ölçekleme; kırpma).")
    simulink = "UAV Toolbox / PX4 control allocation"
    PARAMS = (
        *_geometry_params(),
        Param("method", "choice", "px4", "Doygunluk giderme",
              choices={"px4": "PX4 (airmode kapalı; yaw önce kesilir)", "px4_airmode": "PX4 airmode",
                       "scale": "Ölçekle (Betaflight benzeri)", "clip": "Kırp"}, group="Karıştırıcı"),
        Param("output", "choice", "omega", "Çıkış",
              choices={"omega": "Motor hızı Ω_ref (rad/s)", "pwm": "Gaz 0–1", "thrust": "Motor itkisi (N)"},
              group="Karıştırıcı"),
        p_ts(-1.0),
    )

    def check_params(self):
        try:
            self._mix, self._qp = _mixer_from(self.params)
        except ControlError as exc:
            self.error(str(exc))

    def input_labels(self):
        return ["[T Mx My Mz]"]

    def propagate_widths(self, in_widths, guess=False):
        return [self._mix.n]

    def setup(self, in_widths, out_widths):
        if in_widths[0] != 4:
            self.error(f"giriş genişliği 4 olmalı ({in_widths[0]} bağlandı).")

    def _convert(self, t_motor):
        qp = self._qp
        om = np.sqrt(np.maximum(t_motor, 0.0) / qp.kT)
        out = self.params["output"]
        if out == "omega":
            return om
        if out == "pwm":
            return np.clip((om - qp.wb) / qp.cr, 0.0, 1.0)
        return t_motor

    def output(self, t, x, xd, u):
        tm, _ = self._mix.allocate(u[0], self.params["method"])
        return [self._convert(tm)]

    def linearize(self, t, x, xd, u):
        tm, info = self._mix.allocate(u[0], self.params["method"])
        if info["saturated"] or np.any(tm <= 0):
            return None                    # doygunlukta sayısal merkezi fark
        qp = self._qp
        P = self._mix.P
        om = np.sqrt(tm / qp.kT)
        dom = np.diag(1.0 / (2.0 * qp.kT * om)) @ P
        D = {"omega": dom, "pwm": dom / qp.cr, "thrust": P}[self.params["output"]]
        n = self._mix.n
        return np.zeros((0, 0)), np.zeros((0, 4)), np.zeros((n, 0)), D


@register_block
class InverseMixerBlock(Block):
    type_name = "UAVMotorsToVirtual"
    display_name = "Ters karıştırıcı (motor → sanal)"
    category = "controllers"
    description = "Motor hızları Ω (rad/s) → üretilen [T Mx My Mz] (T_i = kT·Ω_i², etkinlik matrisiyle)."
    simulink = "Gain (etkinlik matrisi)"
    PARAMS = (*_geometry_params(), p_ts(-1.0))

    def check_params(self):
        self._mix, self._qp = _mixer_from(self.params)

    def input_labels(self):
        return ["Ω"]

    def propagate_widths(self, in_widths, guess=False):
        return [4]

    def setup(self, in_widths, out_widths):
        if in_widths[0] != self._mix.n:
            self.error(f"giriş genişliği {self._mix.n} olmalı.")

    def output(self, t, x, xd, u):
        return [self._mix.virtual(self._qp.kT * np.asarray(u[0], float) ** 2)]

    def linearize(self, t, x, xd, u):
        D = self._mix.E @ np.diag(2.0 * self._qp.kT * np.asarray(u[0], float))
        return np.zeros((0, 0)), np.zeros((0, self._mix.n)), np.zeros((4, 0)), D


# ============================================================================ ivme → tutum + itki
@register_block
class AccelToAttitude(Block):
    type_name = "UAVAccelToAttitude"
    display_name = "İvme → tutum + itki"
    category = "controllers"
    description = ("İstenen NED ivmesi [a_n a_e a_d] ve yaw'dan tutum komutu [φ θ] ve toplam itki T (tam geometrik "
                   "ters çözüm, yerçekimi ileri beslemeli): R·e3 ∥ g·e3 − a, T = m·|g·e3 − a|. Eğim sınırı yatay "
                   "ivmeyi kısar (dikey öncelikli).")
    simulink = "PX4 thrust vector → attitude"
    PARAMS = (
        Param("mass", "float", 1.4, "Kütle", unit="kg", min=0.0, strict_min=True),
        Param("g", "float", 9.80665, "Yerçekimi", unit="m/s²", min=0.0, strict_min=True),
        Param("max_tilt", "float", math.radians(35.0), "En büyük eğim", unit="rad", min=0.0, max=math.radians(80)),
        Param("T_min", "float", 0.0, "En küçük itki", unit="N", min=0.0),
        Param("T_max", "float", 44.0, "En büyük itki", unit="N", min=0.0, strict_min=True),
        p_ts(-1.0),
    )

    def n_inputs(self):
        return 2

    def n_outputs(self):
        return 2

    def input_labels(self):
        return ["a_NED", "ψ_ref"]

    def output_labels(self):
        return ["[φ θ]_ref", "T"]

    def propagate_widths(self, in_widths, guess=False):
        return [2, 1]

    def setup(self, in_widths, out_widths):
        if in_widths[0] != 3 or in_widths[1] != 1:
            self.error("girişler [a_n a_e a_d] (3) ve ψ_ref (1) olmalı.")

    def compute(self, a, psi):
        p = self.params
        c, s = math.cos(psi), math.sin(psi)
        w1 = -(c * a[0] + s * a[1])
        w2 = s * a[0] - c * a[1]
        w3 = max(p["g"] - a[2], 1e-3 * p["g"])
        h = math.hypot(w1, w2)
        hmax = w3 * math.tan(p["max_tilt"])
        if h > hmax:
            w1, w2 = w1 * hmax / h, w2 * hmax / h
        theta = math.atan2(w1, w3)
        phi = math.atan2(-w2, math.hypot(w1, w3))
        T = min(max(p["mass"] * math.sqrt(w1 * w1 + w2 * w2 + w3 * w3), p["T_min"]), p["T_max"])
        return np.array([phi, theta]), np.array([T])

    def output(self, t, x, xd, u):
        a = np.asarray(u[0], float)
        return list(self.compute(a, float(np.asarray(u[1]).ravel()[0])))


# ============================================================================ doğrusal MPC (ayrık)
@register_block
class LinearMPCBlock(DiscreteController):
    type_name = "UAVLinearMPC"
    display_name = "Doğrusal MPC (ayrık)"
    category = "controllers"
    description = ("spt.control.mpc (MATLAB mpc / mpcmove anlamları) ile ayrık doğrusal MPC: girişler ölçülen çıkışlar "
                   "ym ve referans r; çıkış MV. Varsayılan Kalman kestiricisi ve çıkış bozucu modeli (MATLAB gibi).")
    simulink = "MPC Toolbox / MPC Controller"
    PARAMS = (
        Param("A", "matrix", [[1.0]], "Tahmin modeli A", group="Model"),
        Param("B", "matrix", [[1.0]], "Tahmin modeli B", group="Model"),
        Param("C", "matrix", [[1.0]], "Tahmin modeli C", group="Model"),
        Param("D", "matrix", [[0.0]], "Tahmin modeli D", group="Model"),
        Param("model_Ts", "float", 0.0, "Model örnekleme zamanı (0: sürekli, ZOH ile ayrıklaştırılır)", unit="s",
              min=0.0, group="Model"),
        Param("p", "int", 20, "Tahmin ufku p", min=1, group="Ufuk"),
        Param("m", "int", 3, "Kontrol ufku m", min=1, group="Ufuk"),
        Param("w_y", "vector", [1.0], "Çıkış ağırlıkları", min=0.0, group="Ağırlık"),
        Param("w_u", "vector", [0.0], "MV ağırlıkları", min=0.0, group="Ağırlık"),
        Param("w_du", "vector", [0.1], "MV değişim ağırlıkları", min=0.0, group="Ağırlık"),
        Param("u_min", "vector", [-math.inf], "MV alt sınırı", allow_inf=True, group="Kısıt"),
        Param("u_max", "vector", [math.inf], "MV üst sınırı", allow_inf=True, group="Kısıt"),
        Param("xp0", "vector", [], "Kestirici başlangıç bitki durumu (boş: nominal)", group="Başlangıç",
              hint="MATLAB mpcstate(mpcobj, plant) karşılığı; ölçülen çıkışlarla tutarlı başlatmak için."),
        p_ts(0.05, hint="Denetleyici örnekleme zamanı (s)."),
    )

    def check_params(self):
        p = self.params
        A, B, C, D = p["A"], p["B"], p["C"], p["D"]
        n, nu, ny = A.shape[0], B.shape[1], C.shape[0]
        if A.shape != (n, n) or B.shape[0] != n or C.shape[1] != n or D.shape != (ny, nu):
            self.error("A (n×n), B (n×m), C (p×n), D (p×m) boyutları uyumsuz.")
        if not p["Ts"].is_discrete:
            self.error("MPC örnekleme zamanı pozitif olmalı.")
        self._n, self._nu, self._ny = n, nu, ny

        def vec(name, k):
            v = np.asarray(p[name], float).ravel()
            if v.size == 1:
                v = np.full(k, v[0])
            if v.size != k:
                self.error(f"'{self._param_defs[name].label}' {k} elemanlı olmalı.")
            return v
        self._wy, self._wu, self._wdu = vec("w_y", ny), vec("w_u", nu), vec("w_du", nu)
        self._umin, self._umax = vec("u_min", nu), vec("u_max", nu)
        xp0 = np.asarray(p["xp0"], float).ravel()
        if xp0.size not in (0, n):
            self.error(f"başlangıç bitki durumu boş ya da {n} elemanlı olmalı.")
        self._xp0 = xp0 if xp0.size else None
        self._obj = None

    def _mpc(self):
        if self._obj is None:
            from spt.control.mpc import mpc
            p = self.params
            Ts = p["Ts"].period
            plant = (p["A"], p["B"], p["C"], p["D"], p["model_Ts"])
            if p["model_Ts"] > 0 and abs(p["model_Ts"] - Ts) > 1e-12:
                self.error("ayrık tahmin modelinin örnekleme zamanı denetleyicininkine eşit olmalı.")
            obj = mpc(plant, Ts, p["p"], p["m"], dict(OutputVariables=self._wy, ManipulatedVariables=self._wu,
                                                     ManipulatedVariablesRate=self._wdu))
            for i in range(self._nu):
                if math.isfinite(self._umin[i]):
                    obj.MV[i].Min = float(self._umin[i])
                if math.isfinite(self._umax[i]):
                    obj.MV[i].Max = float(self._umax[i])
            self._obj = obj
        return self._obj

    @property
    def mpc_object(self):
        return self._mpc()

    def input_labels(self):
        return ["ym", "r"]

    def port_widths_in(self):
        return [self._ny, self._ny]

    def output_width(self):
        return self._nu

    def initial_memory(self):
        from spt.control.mpc import mpcstate
        return mpcstate(self._mpc(), plant=self._xp0)

    def compute(self, t, u, memory):
        from spt.control.mpc import mpcmove
        st = memory.copy()
        ym, r = np.asarray(u[0], float), np.asarray(u[1], float)
        try:
            mv, _ = mpcmove(self._mpc(), st, ym, r)
        except ControlError as exc:              # ör. sonlu olmayan ölçüm (NaN) — blok adıyla bildir
            self.error(f"t = {t:.6g} s: {exc}")
        return np.asarray(mv, float).ravel(), st

    def linearize(self, t, x, xd, u):
        """MPC'nin belleği (kestirici durumu, son hamle) motorun göremediği bir Python nesnesidir; sayısal geri dönüş
        dinamikleri yok sayıp yanlış (statik) bir model verirdi. Bu yüzden açık hata: kısıtsız eşdeğer doğrusal
        denetleyici `spt.control.mpc.ss(blok.mpc_object, 'r')` ile alınıp diyagramda Ayrık durum uzayı bloğu olarak
        kullanılabilir (MATLAB ss(mpcobj, 'r') karşılığı)."""
        raise ControlError(f"'{self.name}' (MPC) bloğu doğrudan doğrusallaştırılamaz: kısıtsız eşdeğer denetleyiciyi "
                           "mpc.ss(mpcobj, 'r') ile alıp Ayrık durum uzayı bloğu olarak kullanın.")


# ============================================================================ algılayıcılar
def _rng(seed: int, k: int, stream: int) -> np.random.Generator:
    """Sayaç tabanlı (durumsuz) rastgele akış: aynı (tohum, örnek, akış) aynı gürültü."""
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, int(k), int(stream)])


class _SensorBase(DiscreteController):
    category = "sinks"

    def _k(self, t: float) -> int:
        return int(round(t / self.params["Ts"].period))


@register_block
class IMUSensor(_SensorBase):
    type_name = "UAVIMU"
    display_name = "IMU (jiroskop + ivmeölçer)"
    description = ("Gövde açısal hızı ve özgül kuvvetin örneklenmiş, gürültülü (yoğunluk/√Ts) ve sapmalı (rastgele "
                   "yürüyüş) ölçümü; tohumlu. Girişler: pqr (3), f_ivme (3); çıkış [jiroskop(3) ivmeölçer(3)].")
    simulink = "UAV Toolbox / IMU"
    PARAMS = (
        Param("gyro_noise", "float", 0.0, "Jiroskop gürültü yoğunluğu", unit="rad/s/√Hz", min=0.0),
        Param("accel_noise", "float", 0.0, "İvmeölçer gürültü yoğunluğu", unit="m/s²/√Hz", min=0.0),
        Param("gyro_bias", "vector", [0.0, 0.0, 0.0], "Jiroskop sapması", unit="rad/s"),
        Param("accel_bias", "vector", [0.0, 0.0, 0.0], "İvmeölçer sapması", unit="m/s²"),
        Param("gyro_walk", "float", 0.0, "Jiroskop sapma yürüyüşü", unit="rad/s/√s", min=0.0),
        Param("accel_walk", "float", 0.0, "İvmeölçer sapma yürüyüşü", unit="m/s²/√s", min=0.0),
        Param("seed", "int", 1, "Tohum"),
        p_ts(0.002),
    )

    def input_labels(self):
        return ["pqr", "f_ivme"]

    def output_labels(self):
        return ["[ω_ölç f_ölç]"]

    def port_widths_in(self):
        return [3, 3]

    def output_width(self):
        return 6

    def initial_memory(self):
        return np.concatenate((np.asarray(self.params["gyro_bias"], float), np.asarray(self.params["accel_bias"],
                                                                                       float)))

    def compute(self, t, u, bias):
        p = self.params
        Ts = p["Ts"].period
        k = self._k(t)
        noise = _rng(p["seed"], k, 0).standard_normal(6) * np.repeat([p["gyro_noise"], p["accel_noise"]], 3) \
            / math.sqrt(Ts)
        y = np.concatenate((np.asarray(u[0], float), np.asarray(u[1], float))) + bias + noise
        walk = _rng(p["seed"], k, 1).standard_normal(6) * np.repeat([p["gyro_walk"], p["accel_walk"]], 3) \
            * math.sqrt(Ts)
        return y, bias + walk

    def linearize(self, t, x, xd, u):
        return np.zeros((0, 0)), np.zeros((0, 6)), np.zeros((6, 0)), np.eye(6)


@register_block
class BarometerSensor(_SensorBase):
    type_name = "UAVBarometer"
    display_name = "Barometre (irtifa)"
    description = "Barometrik irtifa h = −pd ölçümü: sapma + beyaz gürültü (σ), tohumlu. Giriş konum (NED, 3)."
    simulink = "UAV Toolbox / Barometer"
    PARAMS = (
        Param("sigma", "float", 0.0, "Gürültü σ", unit="m", min=0.0),
        Param("bias", "float", 0.0, "Sapma", unit="m"),
        Param("seed", "int", 2, "Tohum"),
        p_ts(0.02),
    )

    def input_labels(self):
        return ["konum"]

    def output_labels(self):
        return ["h"]

    def port_widths_in(self):
        return [3]

    def output_width(self):
        return 1

    def compute(self, t, u, memory):
        p = self.params
        n = _rng(p["seed"], self._k(t), 0).standard_normal() * p["sigma"]
        return np.array([-float(np.asarray(u[0])[2]) + p["bias"] + n]), memory

    def linearize(self, t, x, xd, u):
        return np.zeros((0, 0)), np.zeros((0, 3)), np.zeros((1, 0)), np.array([[0.0, 0.0, -1.0]])


@register_block
class GPSSensor(_SensorBase):
    type_name = "UAVGPS"
    display_name = "GPS (gecikmeli, gürültülü)"
    description = ("Konum + hız (NED) ölçümü; konum hatası Gauss–Markov (Beard & McLain), hız beyaz gürültülü, "
                   "'delay' örnek gecikmeli; tohumlu. Girişler konum (3), hız (3); çıkış [konum(3) hız(3)].")
    simulink = "UAV Toolbox / GPS"
    PARAMS = (
        Param("sigma_pos", "vector", [0.21, 0.21, 0.40], "Konum Gauss–Markov σ (N E D)", unit="m", min=0.0),
        Param("k_gps", "float", 1.0 / 1100.0, "Gauss–Markov ters zaman sabiti", unit="1/s", min=0.0),
        Param("sigma_vel", "float", 0.05, "Hız gürültüsü σ", unit="m/s", min=0.0),
        Param("delay", "int", 0, "Gecikme (örnek)", min=0),
        Param("seed", "int", 3, "Tohum"),
        p_ts(0.2),
    )

    def input_labels(self):
        return ["konum", "hız"]

    def output_labels(self):
        return ["[konum hız]_ölç"]

    def port_widths_in(self):
        return [3, 3]

    def output_width(self):
        return 6

    def initial_memory(self):
        return (np.zeros(3), ())                       # (Gauss–Markov hatası, geçmiş ölçümler)

    def compute(self, t, u, memory):
        p = self.params
        nu, hist = memory
        k = self._k(t)
        rng = _rng(p["seed"], k, 0)
        true = np.concatenate((np.asarray(u[0], float), np.asarray(u[1], float)))
        hist = (hist + (true,))[-(p["delay"] + 1):]
        y = hist[0] + np.concatenate((nu, rng.standard_normal(3) * p["sigma_vel"]))
        a = math.exp(-p["k_gps"] * p["Ts"].period)
        nu_new = a * nu + rng.standard_normal(3) * np.asarray(p["sigma_pos"], float)
        return y, (nu_new, hist)

    def linearize(self, t, x, xd, u):
        return np.zeros((0, 0)), np.zeros((0, 6)), np.zeros((6, 0)), np.eye(6)


__all__ = ["QuadrotorPlant", "FixedWingPlant", "MixerBlock", "InverseMixerBlock", "AccelToAttitude", "LinearMPCBlock",
           "IMUSensor", "BarometerSensor", "GPSSensor"]
