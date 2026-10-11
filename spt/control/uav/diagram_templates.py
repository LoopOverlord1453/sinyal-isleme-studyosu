"""İHA otopilot şablonları (blok diyagram düzenleyicisinin "Şablonlar" menüsü; `TEMPLATES` listesi).

Her şablonun `build()` fonksiyonu motorun `Model`'ini döndürür; kazançlar `spt.control.uav.autopilot` ile
doğrusallaştırılmış modellerden hesaplanır (bkz. o modülün belgesi). Ölçümler idealdir (bitki çıkışları); algılayıcı
blokları (IMU / barometre / GPS) kütüphanededir ve araya eklenebilir.

1. Dörtrotor kademeli otopilot (PX4 tarzı): konum P (50 Hz) → hız PI (50 Hz) → ivme → tutum + itki (yerçekimi ileri
   beslemesi) → açı P (250 Hz) → açısal hız PID (500 Hz) → karıştırıcı (PX4 doygunluk giderme) → motorlar; irtifa
   kademesi pd P → dikey hız PI (50 Hz); yaw açı P + hız PI. Doyumlar + sıkıştırma (clamping) anti-windup.
2. Dörtrotor LQR tutum + irtifa (250 Hz): askı doğrusallaştırmasından dlqr; tutum + motor momenti geri beslemesi
   (ters karıştırıcıyla ölçülen motor hızlarından), irtifa PIDF.
3. Dörtrotor MPC konum / irtifa (20 Hz): doğrusal MPC (spt.control.mpc; MATLAB mpcmove anlamları) ivme komutu üretir;
   iç tutum çevrimleri 1. şablondaki gibi.
4. Sabit kanat ardışık döngü kapama (Beard & McLain Böl. 6, 100 Hz): roll (P + hız sönümü) ← rota PI; yan kayma PI
   (dümen); pitch (P + hız sönümü) ← irtifa PI; hava hızı PI (gaz). Trim değerleri ileri besleme / integratör
   başlangıcı olarak.
5. Öğretici: dörtrotor roll ekseni (motor gecikmesi + 1/(Jx·s) + 1/s) açı P → hız PID kademesi (500 Hz, tek hız).
"""
from __future__ import annotations

import math

import numpy as np

from spt.control.diagram import Model
from spt.control.uav.autopilot import (
    design_fixedwing_slc, design_quad_cascade, design_quad_lqr_attitude, quad_mpc_model,
)
from spt.control.uav.fixedwing import FixedWing, FixedWingParams
from spt.control.uav.quadrotor import QuadrotorParams

CATEGORY = "Otopilot"


# ============================================================================ küçük yardımcılar
def _pid(m: Model, name: str, gains: dict, Ts: float, *, limit: tuple[float, float] | None = None,
         controller: str | None = None, xi0: float = 0.0):
    kw = dict(controller=controller or gains.get("controller", "PID"), form="parallel", P=gains.get("P", 0.0),
              I=gains.get("I", 0.0), D=gains.get("D", 0.0), N=gains.get("N", 100.0), Ts=Ts, xi0=xi0)
    if limit is not None:
        kw.update(limit_output=True, lower=limit[0], upper=limit[1])
        if "I" in kw["controller"]:
            kw["anti_windup"] = "clamping"
    return m.add("PIDController", name, **kw)


def _pid2(m: Model, name: str, gains: dict, Ts: float, ref, meas, *, b: float, c: float,
          limit: tuple[float, float], ref_port: int = 0, meas_port: int = 0):
    """2-DOF PID (ayar noktası ağırlıklı: e_p = b·r − y, e_i = r − y, e_d = c·r − y) — c = 0 türev vuruşunu,
    b < 1 PID sıfırının yol açtığı aşımı azaltır; kapalı çevrim kutupları (kararlılık) 1-DOF ile aynıdır."""
    blk = m.add("PIDController2DOF", name, controller=gains.get("controller", "PID"), form="parallel",
                P=gains["P"], I=gains.get("I", 0.0), D=gains.get("D", 0.0), N=gains.get("N", 100.0), b=b, c=c, Ts=Ts,
                limit_output=True, lower=limit[0], upper=limit[1], anti_windup="clamping")
    m.connect(ref, blk, src_port=ref_port, dst_port=0)
    m.connect(meas, blk, src_port=meas_port, dst_port=1)
    return blk


def _gain_block(m: Model, name: str, k: float, Ts: float):
    """Ayrık oransal kazanç (P denetleyici olarak — örnekleme hızında tutulur)."""
    return m.add("PIDController", name, controller="P", P=float(k), Ts=Ts)


def _sel(m: Model, name: str, src, port: int, idx: int):
    s = m.add("Selector", name, indices=[idx])
    m.connect(src, s, src_port=port)
    return s


def _err(m: Model, name: str, ref, meas, ref_port: int = 0, meas_port: int = 0):
    s = m.add("Sum", name, signs="+-")
    m.connect(ref, s, src_port=ref_port, dst_port=0)
    m.connect(meas, s, src_port=meas_port, dst_port=1)
    return s


def _quad_plant(m: Model, p: QuadrotorParams, *, input_mode: str = "omega", position0=(0.0, 0.0, -10.0)):
    return m.add("UAVQuadrotor", "Dörtrotor", mass=p.mass, inertia=list(np.diag(np.asarray(p.inertia))),
                 arm_length=p.arm_length, config=p.config, kT=p.kT, kQ=p.kQ, tau_m=p.tau_m, cr=p.cr, wb=p.wb,
                 omega_min=p.omega_min, omega_max=p.omega_max, Jr=p.Jr, gyro=p.gyro, input_mode=input_mode,
                 position0=list(position0))


def _mixer(m: Model, p: QuadrotorParams, Ts: float, method: str = "px4"):
    return m.add("UAVMixer", "Karıştırıcı", config=p.config, arm_length=p.arm_length, kT=p.kT, kQ=p.kQ,
                 omega_min=p.omega_min, omega_max=p.omega_max, cr=p.cr, wb=p.wb, method=method, output="omega", Ts=Ts)


def _attitude_core(m: Model, d, p: QuadrotorParams, plant, att_ref, yaw_ref, thrust, *, att_port=0, yaw_port=0,
                   thrust_port=0, yaw_idx: int | None = None):
    """Açı P (250 Hz) → açısal hız PID (500 Hz) → karıştırıcı → bitki. att_ref: [φ_ref θ_ref] (2) sinyali."""
    Ta, Tr = d.Ts["angle"], d.Ts["rate"]
    L = d.limits
    rates = []
    for k, axis in enumerate(("roll", "pitch", "yaw")):
        sym = "φθψ"[k]
        meas = _sel(m, f"{sym} ölçüm", plant, 2, k + 1)
        if axis == "yaw":
            ref, rport = (yaw_ref, yaw_port) if yaw_idx is None else (_sel(m, "ψ ref", yaw_ref, yaw_port, yaw_idx), 0)
        else:
            ref, rport = _sel(m, f"{sym} ref", att_ref, att_port, k + 1), 0
        e = _err(m, f"{sym} hatası", ref, meas, ref_port=rport)
        lim = L["rate_cmd_yaw"] if axis == "yaw" else L["rate_cmd_rp"]
        a = _pid(m, f"Açı P ({axis})", {"P": d.angle[axis]}, Ta, limit=(-lim, lim), controller="P")
        m.connect(e, a)
        w = "pqr"[k]
        wm = _sel(m, f"{w} ölçüm", plant, 3, k + 1)
        ew = _err(m, f"{w} hatası", a, wm)
        lim = L["moment_yaw"] if axis == "yaw" else L["moment_rp"]
        r = _pid(m, f"Hız PID ({axis})", d.rate[axis], Tr, limit=(-lim, lim))
        m.connect(ew, r)
        rates.append(r)
    mux = m.add("Mux", "Sanal giriş [T Mx My Mz]", inputs=4)
    m.connect(thrust, mux, src_port=thrust_port, dst_port=0)
    for k, r in enumerate(rates):
        m.connect(r, mux, dst_port=k + 1)
    mix = _mixer(m, p, Tr)
    m.connect(mux, mix, name="sanal komut", log=True)
    m.connect(mix, plant, name="motor komutu", log=True)
    m.log_signal(plant, 0, "konum")
    m.log_signal(plant, 1, "hız")
    m.log_signal(plant, 2, "Euler")
    m.log_signal(plant, 3, "pqr")
    m.log_signal(plant, 4, "motor hızı")
    m.connect(plant, m.add("Terminator", "f_ivme (kullanılmıyor)"), src_port=5)
    return mix


# ============================================================================ 1. kademeli otopilot
ALT_WEIGHTS = (1.0, 0.0)          # tek çevrim irtifa 2-DOF PIDF (LQR şablonu): b = 1, c = 0 (türev yalnız ölçümde)
# Not: b ≠ 1 mutlak koordinatlarda (pd = −10 m) dengede sabit bir P ofseti yaratır (b·r − y ≠ 0).



def build_quad_cascade(params: QuadrotorParams | None = None, *, step=(2.0, 1.0, -1.0, 0.5), t_step: float = 1.0,
                       stop_time: float = 8.0, design=None) -> Model:
    """Dörtrotor kademeli otopilot. Senaryo: t_step anında konum adımı [Δpn Δpe Δpd] ve yaw adımı (rad); başlangıç
    askıda pd = −10 m. İrtifa kademeli: pd P → dikey hız PI → a_d (yerçekimi ileri beslemesi ivme → itki bloğunda)."""
    p = params if params is not None else QuadrotorParams()
    d = design if design is not None else design_quad_cascade(p)
    To = d.Ts["outer"]
    L = d.limits
    m = Model("Dörtrotor kademeli otopilot")
    m.meta["design"] = {"wc": d.wc, "Ts": d.Ts}
    plant = _quad_plant(m, p)
    ref = m.add("Step", "Referans [pn pe pd ψ]", step_time=t_step, initial=[0.0, 0.0, -10.0, 0.0],
                final=[step[0], step[1], -10.0 + step[2], step[3]])
    m.log_signal(ref, 0, "referans")
    acc = []
    for k, (ax, vname) in enumerate((("N", "vn"), ("E", "ve"))):
        pr = _sel(m, f"p{ax.lower()} ref", ref, 0, k + 1)
        pm = _sel(m, f"p{ax.lower()} ölçüm", plant, 0, k + 1)
        ep = _err(m, f"Konum hatası {ax}", pr, pm)
        cp = _pid(m, f"Konum P ({ax})", {"P": d.position}, To, limit=(-L["velocity"], L["velocity"]), controller="P")
        m.connect(ep, cp)
        vm = _sel(m, f"{vname} ölçüm", plant, 1, k + 1)
        ev = _err(m, f"Hız hatası {ax}", cp, vm)
        cv = _pid(m, f"Hız PI ({ax})", {**d.velocity, "controller": "PI"}, To, limit=(-L["accel_h"], L["accel_h"]))
        m.connect(ev, cv)
        acc.append(cv)
    hr = _sel(m, "pd ref", ref, 0, 3)
    hm = _sel(m, "pd ölçüm", plant, 0, 3)
    eh = _err(m, "İrtifa hatası (pd)", hr, hm)
    cz = _pid(m, "İrtifa P", {"P": d.alt_position}, To, limit=(-L["velocity"], L["velocity"]), controller="P")
    m.connect(eh, cz)
    vzm = _sel(m, "vd ölçüm", plant, 1, 3)
    evz = _err(m, "Dikey hız hatası", cz, vzm)
    ch = _pid(m, "Dikey hız PI", {**d.alt_velocity, "controller": "PI"}, To, limit=(-L["accel_v"], L["accel_v"]))
    m.connect(evz, ch)
    amux = m.add("Mux", "İvme komutu [an ae ad]", inputs=3)
    for k, b in enumerate(acc + [ch]):
        m.connect(b, amux, dst_port=k)
    a2a = m.add("UAVAccelToAttitude", "İvme → tutum + itki", mass=p.mass, g=p.g, max_tilt=L["max_tilt"],
                T_max=L["T_max"], Ts=To)
    m.connect(amux, a2a, dst_port=0, name="ivme komutu", log=True)
    yaw = _sel(m, "ψ ref (dış)", ref, 0, 4)
    m.connect(yaw, a2a, dst_port=1)
    m.name_signal(a2a, 0, "tutum ref", log=True)
    m.name_signal(a2a, 1, "itki komutu", log=True)
    _attitude_core(m, d, p, plant, a2a, yaw, a2a, att_port=0, thrust_port=1)
    m.settings.update(solver="ode4", step=d.Ts["rate"], stop_time=stop_time)
    return m


# ============================================================================ 2. LQR tutum + irtifa
def build_quad_lqr(params: QuadrotorParams | None = None, *, step=(0.3, -0.2, 0.5), t_step: float = 0.5,
                   stop_time: float = 4.0, Ts: float = 0.004) -> Model:
    """Dörtrotor LQR tutum (dlqr) + irtifa PIDF; tüm denetleyiciler tek hızda (Ts) — doğrusallaştırma tam örneklenmiş
    veri modelidir. Senaryo: t_step anında [φ θ ψ] referans adımı (rad), irtifa sabit."""
    p = params if params is not None else QuadrotorParams()
    lq = design_quad_lqr_attitude(p, Ts=Ts)
    d = design_quad_cascade(p)
    m = Model("Dörtrotor LQR tutum + irtifa")
    m.meta["design"] = {"K": lq["K"].tolist(), "poles": [[float(z.real), float(z.imag)] for z in lq["poles"]]}
    plant = _quad_plant(m, p)
    ref = m.add("Step", "Tutum referansı [φ θ ψ]", step_time=t_step, initial=[0.0, 0.0, 0.0], final=list(step))
    m.log_signal(ref, 0, "referans")
    inv = m.add("UAVMotorsToVirtual", "Ters karıştırıcı", config=p.config, arm_length=p.arm_length, kT=p.kT, kQ=p.kQ,
                omega_min=p.omega_min, omega_max=p.omega_max, cr=p.cr, wb=p.wb)
    m.connect(plant, inv, src_port=4)
    msel = m.add("Selector", "Momentler [Mx My Mz]", indices=[2, 3, 4])
    m.connect(inv, msel)
    xm = m.add("Mux", "Durum [η ω M]", inputs=3)
    m.connect(plant, xm, src_port=2, dst_port=0)
    m.connect(plant, xm, src_port=3, dst_port=1)
    m.connect(msel, xm, dst_port=2)
    fb = m.add("StateFeedback", "LQR (dlqr)", K=lq["K"], use_reference=True, Nr=lq["Nr"], Ts=Ts)
    m.connect(xm, fb, dst_port=0, name="durum", log=True)
    m.connect(ref, fb, dst_port=1)
    sat = m.add("Saturation", "Moment sınırı", upper=[1.5, 1.5, 0.15], lower=[-1.5, -1.5, -0.15])
    m.connect(fb, sat, name="moment komutu", log=True)
    # irtifa: PIDF (aynı hızda) → a_d → T = m·(g − a_d)
    hr = m.add("Constant", "pd ref", value=[-10.0])
    hm = _sel(m, "pd ölçüm", plant, 0, 3)
    ch = _pid2(m, "İrtifa PIDF (2-DOF)", d.altitude, Ts, hr, hm, b=ALT_WEIGHTS[0], c=ALT_WEIGHTS[1],
               limit=(-d.limits["accel_v"], d.limits["accel_v"]))
    gm = m.add("Gain", "−m", K=[[-p.mass]])
    m.connect(ch, gm)
    mg = m.add("Constant", "m·g (askı itkisi)", value=[p.mass * p.g])
    tsum = m.add("Sum", "İtki", signs="++")
    m.connect(mg, tsum, dst_port=0)
    m.connect(gm, tsum, dst_port=1)
    mux = m.add("Mux", "Sanal giriş [T Mx My Mz]", inputs=2)
    m.connect(tsum, mux, dst_port=0)
    m.connect(sat, mux, dst_port=1)
    mix = _mixer(m, p, Ts)
    m.connect(mux, mix, name="sanal komut", log=True)
    m.connect(mix, plant, name="motor komutu", log=True)
    for k, nm in enumerate(("konum", "hız", "Euler", "pqr")):
        m.log_signal(plant, k, nm)
    m.name_signal(plant, 4, "motor hızı", log=True)
    m.connect(plant, m.add("Terminator", "hız (kullanılmıyor)"), src_port=1)
    m.connect(plant, m.add("Terminator", "f_ivme (kullanılmıyor)"), src_port=5)
    m.settings.update(solver="ode4", step=Ts / 2, stop_time=stop_time)
    return m


# ============================================================================ 3. MPC konum / irtifa
MPC_SETTINGS = {"Ts": 0.05, "p": 30, "m": 4, "w_y": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0], "w_u": [0.0, 0.0, 0.0],
                "w_du": [0.1, 0.1, 0.1], "u_min": [-3.5, -3.5, -5.0], "u_max": [3.5, 3.5, 5.0]}


def build_quad_mpc(params: QuadrotorParams | None = None, *, step=(2.0, 1.0, -1.0), t_step: float = 1.0,
                   stop_time: float = 8.0) -> Model:
    """MPC konum / irtifa: tahmin modeli quad_mpc_model(τ_h = 1/wc_açı, τ_v = τm); ölçülen çıkışlar konum (3) ve
    hız (3; ağırlık 0, referans 0) — MATLAB varsayılan kestiricisiyle. MV: ivme komutu [a_n a_e a_d] (sınırlı)."""
    p = params if params is not None else QuadrotorParams()
    d = design_quad_cascade(p)
    s = MPC_SETTINGS
    A, B, C, D = quad_mpc_model(1.0 / d.wc["angle_rp"], p.tau_m)
    m = Model("Dörtrotor MPC konum / irtifa")
    plant = _quad_plant(m, p)
    ref = m.add("Step", "Konum referansı [pn pe pd]", step_time=t_step, initial=[0.0, 0.0, -10.0],
                final=[step[0], step[1], -10.0 + step[2]])
    m.log_signal(ref, 0, "referans")
    mpc = m.add("UAVLinearMPC", "MPC (konum)", A=A, B=B, C=C, D=D, model_Ts=0.0, p=s["p"], m=s["m"], w_y=s["w_y"],
                w_u=s["w_u"], w_du=s["w_du"], u_min=s["u_min"], u_max=s["u_max"], Ts=s["Ts"],
                xp0=[0.0] * 6 + [-10.0, 0.0, 0.0])           # kestirici askıdaki ölçümle tutarlı başlar
    ym = m.add("Mux", "Ölçüm [konum hız]", inputs=2)
    m.connect(plant, ym, src_port=0, dst_port=0)
    m.connect(plant, ym, src_port=1, dst_port=1)
    m.connect(ym, mpc, dst_port=0)
    rv = m.add("Mux", "Referans [konum 0]", inputs=2)
    m.connect(ref, rv, dst_port=0)
    m.connect(m.add("Constant", "Hız referansı (0)", value=[0.0, 0.0, 0.0]), rv, dst_port=1)
    m.connect(rv, mpc, dst_port=1)
    a2a = m.add("UAVAccelToAttitude", "İvme → tutum + itki", mass=p.mass, g=p.g, max_tilt=d.limits["max_tilt"],
                T_max=d.limits["T_max"], Ts=s["Ts"])
    m.connect(mpc, a2a, dst_port=0, name="ivme komutu", log=True)
    yaw = m.add("Constant", "ψ ref", value=[0.0])
    m.connect(yaw, a2a, dst_port=1)
    m.name_signal(a2a, 0, "tutum ref", log=True)
    m.name_signal(a2a, 1, "itki komutu", log=True)
    _attitude_core(m, d, p, plant, a2a, yaw, a2a, att_port=0, thrust_port=1)
    m.settings.update(solver="ode4", step=d.Ts["rate"], stop_time=stop_time)
    return m


# ============================================================================ 4. sabit kanat ardışık döngü kapama
def build_fixedwing_slc(*, h_step: float = 15.0, chi_step: float = math.radians(60.0), Va_step: float = 3.0,
                        times=(2.0, 12.0, 25.0), stop_time: float = 45.0, Ts: float = 0.01) -> Model:
    """Beard & McLain ardışık döngü kapama; Aerosonde, trim Va = 25 m/s, h = 100 m. Senaryo: times[0]'da irtifa
    adımı, times[1]'de rota adımı, times[2]'de hava hızı adımı."""
    fw = FixedWing(FixedWingParams.aerosonde())
    tr = fw.trim(25.0, altitude=100.0)
    g = design_fixedwing_slc(fw, tr)
    s = g["specs"]
    de0, da0, dr0, dt0 = (float(v) for v in tr.u)
    m = Model("Sabit kanat ardışık döngü kapama")
    m.meta["design"] = {k: v for k, v in g.items() if k not in ("coef", "specs")}
    plant = m.add("UAVFixedWing", "Aerosonde", trim_init=True, Va0=25.0, h0=100.0)
    # ölçümler
    phi = _sel(m, "φ ölçüm", plant, 2, 1)
    theta = _sel(m, "θ ölçüm", plant, 2, 2)
    pr = _sel(m, "p ölçüm", plant, 3, 1)
    qr = _sel(m, "q ölçüm", plant, 3, 2)
    Va = _sel(m, "Va ölçüm", plant, 4, 1)
    beta = _sel(m, "β ölçüm", plant, 4, 3)
    chi = _sel(m, "χ ölçüm", plant, 4, 5)
    pd = _sel(m, "pd ölçüm", plant, 0, 3)
    h = m.add("Gain", "h = −pd", K=[[-1.0]])
    m.connect(pd, h)
    # referanslar
    hc = m.add("Step", "İrtifa komutu", step_time=times[0], initial=[100.0], final=[100.0 + h_step])
    cc = m.add("Step", "Rota komutu", step_time=times[1], initial=[0.0], final=[chi_step])
    vc = m.add("Step", "Hava hızı komutu", step_time=times[2], initial=[25.0], final=[25.0 + Va_step])
    # yanal: rota PI → φ_c; roll P + hız sönümü → δa; yan kayma PI → δr
    phimax = math.radians(45.0)
    ec = _err(m, "Rota hatası", cc, chi)
    cchi = _pid(m, "Rota PI", {"P": g["kp_chi"], "I": g["ki_chi"]}, Ts, limit=(-phimax, phimax), controller="PI",
                xi0=tr.phi)
    m.connect(ec, cchi, name="φ_c", log=True)
    ephi = _err(m, "Roll hatası", cchi, phi)
    kphi = _gain_block(m, "Roll P (kp_φ)", g["kp_phi"], Ts)
    m.connect(ephi, kphi)
    kdphi = _gain_block(m, "Roll sönümü (kd_φ)", g["kd_phi"], Ts)
    m.connect(pr, kdphi)
    da_sum = m.add("Sum", "δa", signs="++-")
    m.connect(m.add("Constant", "δa*", value=[da0]), da_sum, dst_port=0)
    m.connect(kphi, da_sum, dst_port=1)
    m.connect(kdphi, da_sum, dst_port=2)
    dmax_a = s["roll"]["delta_max"]
    da = m.add("Saturation", "δa sınırı", upper=[dmax_a], lower=[-dmax_a])
    m.connect(da_sum, da)
    dmax_r = s["sideslip"]["delta_max"]
    cbeta = _pid(m, "Yan kayma PI", {"P": -g["kp_beta"], "I": -g["ki_beta"]}, Ts, limit=(-dmax_r, dmax_r),
                 controller="PI", xi0=dr0)
    m.connect(beta, cbeta)
    # boylamsal: irtifa PI → θ_c; pitch P + hız sönümü → δe; hava hızı PI → δt
    thmax = math.radians(30.0)
    eh = _err(m, "İrtifa hatası", hc, h)
    ch = _pid(m, "İrtifa PI", {"P": g["kp_h"], "I": g["ki_h"]}, Ts, limit=(-thmax, thmax), controller="PI",
              xi0=tr.theta)
    m.connect(eh, ch, name="θ_c", log=True)
    eth = _err(m, "Pitch hatası", ch, theta)
    kth = _gain_block(m, "Pitch P (kp_θ)", g["kp_theta"], Ts)
    m.connect(eth, kth)
    kdth = _gain_block(m, "Pitch sönümü (kd_θ)", g["kd_theta"], Ts)
    m.connect(qr, kdth)
    de_sum = m.add("Sum", "δe", signs="++-")
    m.connect(m.add("Constant", "δe*", value=[de0]), de_sum, dst_port=0)
    m.connect(kth, de_sum, dst_port=1)
    m.connect(kdth, de_sum, dst_port=2)
    dmax_e = s["pitch"]["delta_max"]
    de = m.add("Saturation", "δe sınırı", upper=[dmax_e], lower=[-dmax_e])
    m.connect(de_sum, de)
    ev = _err(m, "Hava hızı hatası", vc, Va)
    cv = _pid(m, "Hava hızı PI (gaz)", {"P": g["kp_V"], "I": g["ki_V"]}, Ts, limit=(0.0, 1.0), controller="PI",
              xi0=dt0)
    m.connect(ev, cv)
    mux = m.add("Mux", "[δe δa δr δt]", inputs=4)
    for k, b in enumerate((de, da, cbeta, cv)):
        m.connect(b, mux, dst_port=k)
    m.connect(mux, plant, name="dümenler", log=True)
    m.connect(plant, m.add("Terminator", "uvw (kullanılmıyor)"), src_port=1)
    for k, nm in enumerate(("konum", "uvw", "Euler", "pqr", "hava verisi")):
        m.log_signal(plant, k, nm)
    m.log_signal(h, 0, "h")
    m.log_signal(hc, 0, "h_c")
    m.log_signal(cc, 0, "χ_c")
    m.log_signal(vc, 0, "Va_c")
    m.settings.update(solver="ode4", step=Ts, stop_time=stop_time)
    return m


# ============================================================================ 5. öğretici: roll ekseni
def build_roll_axis_tutorial(params: QuadrotorParams | None = None, *, step: float = 0.3, t_step: float = 0.1,
                             stop_time: float = 1.5, Ts: float = 0.002) -> Model:
    """Tek eksen: Mx_c → motor gecikmesi 1/(τm·s + 1) → 1/Jx → ∫ → p → ∫ → φ. Açı P → hız PID (aynı Ts), moment
    doyumu ±1.5 N·m + sıkıştırma anti-windup; kazançlar 1. şablonun roll ekseniyle aynı."""
    p = params if params is not None else QuadrotorParams()
    d = design_quad_cascade(p)
    Jx = float(np.diag(np.asarray(p.inertia))[0]) if np.ndim(p.inertia) == 2 else float(p.inertia[0])
    m = Model("Öğretici: roll ekseni kademeli PID")
    r = m.add("Step", "φ referansı", step_time=t_step, initial=[0.0], final=[step])
    ea = m.add("Sum", "φ hatası", signs="+-")
    ca = _pid(m, "Açı P", {"P": d.angle["roll"]}, Ts, limit=(-d.limits["rate_cmd_rp"], d.limits["rate_cmd_rp"]),
              controller="P")
    er = m.add("Sum", "p hatası", signs="+-")
    cr = _pid(m, "Hız PID", d.rate["roll"], Ts, limit=(-d.limits["moment_rp"], d.limits["moment_rp"]))
    motor = m.add("TransferFcn", "Motor gecikmesi", num=[[1.0]], den=[p.tau_m, 1.0])
    ij = m.add("Gain", "Jx⁻¹", K=[[1.0 / Jx]])
    ip = m.add("Integrator", "∫ → p")
    iphi = m.add("Integrator", "∫ → φ")
    m.connect(r, ea, dst_port=0)
    m.connect(ea, ca)
    m.connect(ca, er, dst_port=0, name="p_ref", log=True)
    m.connect(er, cr)
    m.connect(cr, motor, name="Mx_c", log=True)
    m.connect(motor, ij, name="Mx", log=True)
    m.connect(ij, ip)
    m.connect(ip, iphi, name="p", log=True)
    m.connect(ip, er, dst_port=1)
    m.connect(iphi, ea, dst_port=1, name="φ", log=True)
    m.log_signal(r, 0, "φ_ref")
    m.settings.update(solver="ode4", step=Ts, stop_time=stop_time)
    return m


TEMPLATES = [
    {"name": "Dörtrotor kademeli otopilot (PX4 tarzı)", "build": build_quad_cascade, "category": CATEGORY,
     "description": "Konum P → hız PI → tutum (ivme → tutum + itki, yerçekimi ileri beslemesi) → açı P → açısal hız "
                    "PID → karıştırıcı; irtifa PIDF, yaw. 50 / 250 / 500 Hz; pidtune ile tasarlandı."},
    {"name": "Dörtrotor LQR tutum + irtifa", "build": build_quad_lqr, "category": CATEGORY,
     "description": "Askı doğrusallaştırmasından dlqr (tutum + motor momenti geri beslemesi), irtifa PIDF; 250 Hz."},
    {"name": "Dörtrotor MPC konum / irtifa", "build": build_quad_mpc, "category": CATEGORY,
     "description": "Doğrusal MPC (20 Hz, kısıtlı ivme komutu) + iç tutum çevrimleri."},
    {"name": "Sabit kanat ardışık döngü kapama (Beard & McLain)", "build": build_fixedwing_slc, "category": CATEGORY,
     "description": "Rota → roll, irtifa → pitch, hava hızı → gaz, yan kayma → dümen; kitabın kazanç formülleri."},
    {"name": "Öğretici: dörtrotor roll ekseni (kademeli PID)", "build": build_roll_axis_tutorial,
     "category": CATEGORY,
     "description": "Çift tümleyici + motor gecikmesi; açı P → açısal hız PID, doyum + anti-windup."},
]


def templates() -> list[dict]:
    return list(TEMPLATES)


__all__ = ["TEMPLATES", "templates", "build_quad_cascade", "build_quad_lqr", "build_quad_mpc", "build_fixedwing_slc",
           "build_roll_axis_tutorial", "MPC_SETTINGS", "CATEGORY"]
