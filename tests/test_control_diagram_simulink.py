"""Blok diyagram motoru ↔ MATLAB R2025b Simulink karşılaştırmaları.

Her vaka bir Python `Model` kurar; aynı model `_to_simulink` ile Simulink betiğine (aynı blok parametreleri, aynı
çözücü / adım) çevrilir, MATLAB'de `sim` ile çalıştırılır ve kayıtlı referans (tests/data/matlab/*.json) ile
karşılaştırılır. Testler MATLAB'siz çalışır (betik değişirse referans yeniden üretilir).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spt.control.diagram import Model, simulate  # noqa: E402
from tools.matlab_oracle import OracleUnavailable, fixture  # noqa: E402


# ============================================================================ MATLAB değer biçimleri
def _num(x: float) -> str:
    x = float(x)
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    return format(x, ".17g")


def _mat(v: Any) -> str:
    a = np.asarray(v)
    if np.iscomplexobj(a):
        a = a.ravel()
        return "[" + " ".join(f"{_num(z.real)}{'+' if z.imag >= 0 else '-'}{_num(abs(z.imag))}i" for z in a) + "]"
    a = np.asarray(v, dtype=float)
    if a.ndim == 0 or a.size == 1:
        return _num(float(a.ravel()[0])) if a.size else "[]"
    if a.ndim == 1:
        return "[" + " ".join(_num(x) for x in a) + "]"
    return "[" + "; ".join(" ".join(_num(x) for x in row) for row in a) + "]"


def _ts(st) -> str:
    if st.offset:
        return f"[{_num(st.period)} {_num(st.offset)}]"
    return _num(st.period)


_DTI = {"forward_euler": "Integration: Forward Euler", "backward_euler": "Integration: Backward Euler",
        "trapezoidal": "Integration: Trapezoidal"}
_PIDM = {"forward_euler": "Forward Euler", "backward_euler": "Backward Euler", "trapezoidal": "Trapezoidal"}


def _pid(p: dict, two: bool) -> tuple[str, list[tuple[str, str]]]:
    lib = "simulink/Continuous/PID Controller (2DOF)" if two else "simulink/Continuous/PID Controller"
    st = p["Ts"]
    disc = st.is_discrete
    prm = [("Controller", p["controller"]), ("Form", "Parallel" if p["form"] == "parallel" else "Ideal"),
           ("TimeDomain", "Discrete-time" if disc else "Continuous-time"),
           ("P", _num(p["P"])), ("I", _num(p["I"])), ("D", _num(p["D"])), ("N", _num(p["N"])),
           ("UseFilter", "on" if p["use_filter"] else "off"),
           ("InitialConditionForIntegrator", _num(p["xi0"])), ("InitialConditionForFilter", _num(p["xf0"])),
           ("LimitOutput", "on" if p["limit_output"] else "off"),
           ("UpperSaturationLimit", _num(p["upper"])), ("LowerSaturationLimit", _num(p["lower"])),
           ("AntiWindupMode", {"none": "none", "back_calculation": "back-calculation",
                               "clamping": "clamping"}[p["anti_windup"]]),
           ("Kb", _num(p["Kb"])), ("TrackingMode", "on" if p["tracking"] else "off"), ("Kt", _num(p["Kt"]))]
    if disc:
        prm += [("SampleTime", _ts(st)), ("IntegratorMethod", _PIDM[p["integrator_method"]]),
                ("FilterMethod", _PIDM[p["filter_method"]])]
    if two:
        prm += [("b", _num(p["b"])), ("c", _num(p["c"]))]
    return lib, prm


def _block(b, data_vars: list[str], bname: str) -> tuple[str, list[tuple[str, str]]]:
    p = b.params
    t = b.type_name
    if t == "Step":
        return "simulink/Sources/Step", [("Time", _num(p["step_time"])), ("Before", _mat(p["initial"])),
                                         ("After", _mat(p["final"])), ("SampleTime", _ts(p["Ts"]))]
    if t == "Ramp":
        return "simulink/Sources/Ramp", [("slope", _mat(p["slope"])), ("start", _num(p["start_time"])),
                                         ("InitialOutput", _mat(p["initial"]))]
    if t == "SineWave":
        return "simulink/Sources/Sine Wave", [("SineType", "Time based"), ("Amplitude", _mat(p["amplitude"])),
                                              ("Bias", _mat(p["bias"])), ("Frequency", _mat(p["frequency"])),
                                              ("Phase", _mat(p["phase"])), ("SampleTime", _ts(p["Ts"]))]
    if t == "Constant":
        return "simulink/Sources/Constant", [("Value", _mat(p["value"])), ("SampleTime", _ts(p["Ts"]))]
    if t == "PulseGenerator":
        return "simulink/Sources/Pulse Generator", [("PulseType", "Time based"), ("Amplitude", _mat(p["amplitude"])),
                                                    ("Period", _num(p["period"])), ("PulseWidth", _num(p["width"])),
                                                    ("PhaseDelay", _num(p["delay"]))]
    if t == "Clock":
        return "simulink/Sources/Clock", []
    if t == "DataSource":
        var = f"ds_{bname}"
        data_vars.append(f"assignin('base', '{var}', [{_mat(p['time'])}.' {_mat(p['values'])}]);")
        return "simulink/Sources/From Workspace", [("VariableName", var),
                                                   ("Interpolate", "on" if p["interp"] == "linear" else "off"),
                                                   ("OutputAfterFinalValue", "Holding final value"),
                                                   ("SampleTime", _ts(p["Ts"]))]
    if t == "Inport":
        return "simulink/Sources/In1", [("Port", str(p["port"]))]
    if t == "Gain":
        mult = "Element-wise(K.*u)" if p["mode"] == "elementwise" else "Matrix(K*u)"
        return "simulink/Math Operations/Gain", [("Gain", _mat(p["K"])), ("Multiplication", mult),
                                                 ("SampleTime", _ts(p["Ts"]))]
    if t == "Sum":
        return "simulink/Math Operations/Sum", [("Inputs", p["signs"])]
    if t == "Product":
        return "simulink/Math Operations/Product", [("Inputs", p["ops"])]
    if t == "Integrator":
        lim = bool(np.any(np.isfinite(p["upper"])) or np.any(np.isfinite(p["lower"])))
        return "simulink/Continuous/Integrator", [("InitialCondition", _mat(p["x0"])),
                                                  ("LimitOutput", "on" if lim else "off"),
                                                  ("UpperSaturationLimit", _mat(p["upper"])),
                                                  ("LowerSaturationLimit", _mat(p["lower"]))]
    if t == "FilteredDerivative":
        N = p["N"]
        return "simulink/Continuous/Transfer Fcn", [("Numerator", f"[{_num(N)} 0]"), ("Denominator", f"[1 {_num(N)}]")]
    if t == "TransferFcn":
        return "simulink/Continuous/Transfer Fcn", [("Numerator", _mat(p["num"])), ("Denominator", _mat(p["den"]))]
    if t == "StateSpace":
        return "simulink/Continuous/State-Space", [("A", _mat(p["A"])), ("B", _mat(p["B"])), ("C", _mat(p["C"])),
                                                   ("D", _mat(p["D"])), ("InitialCondition", _mat(p["x0"]))]
    if t == "ZeroPole":
        return "simulink/Continuous/Zero-Pole", [("Zeros", _mat(p["zeros"]) if p["zeros"].size else "[]"),
                                                 ("Poles", _mat(p["poles"])), ("Gain", _num(p["gain"]))]
    if t == "TransportDelay":
        return "simulink/Continuous/Transport Delay", [("DelayTime", _num(p["delay"])),
                                                       ("InitialOutput", _mat(p["initial"])),
                                                       ("PadeOrder", str(p["pade_order"]))]
    if t == "DiscreteTransferFcn":
        return "simulink/Discrete/Discrete Transfer Fcn", [("Numerator", _mat(p["num"])),
                                                           ("Denominator", _mat(p["den"])),
                                                           ("SampleTime", _ts(p["Ts"]))]
    if t == "DiscreteFilter":
        return "simulink/Discrete/Discrete Filter", [("Numerator", _mat(p["num"])), ("Denominator", _mat(p["den"])),
                                                     ("SampleTime", _ts(p["Ts"]))]
    if t == "DiscreteStateSpace":
        return "simulink/Discrete/Discrete State-Space", [("A", _mat(p["A"])), ("B", _mat(p["B"])),
                                                          ("C", _mat(p["C"])), ("D", _mat(p["D"])),
                                                          ("InitialCondition", _mat(p["x0"])),
                                                          ("SampleTime", _ts(p["Ts"]))]
    if t == "UnitDelay":
        return "simulink/Discrete/Unit Delay", [("InitialCondition", _mat(p["x0"])), ("SampleTime", _ts(p["Ts"]))]
    if t == "DiscreteIntegrator":
        lim = bool(np.any(np.isfinite(p["upper"])) or np.any(np.isfinite(p["lower"])))
        return "simulink/Discrete/Discrete-Time Integrator", [
            ("IntegratorMethod", _DTI[p["method"]]), ("gainval", _num(p["K"])), ("InitialCondition", _mat(p["x0"])),
            ("LimitOutput", "on" if lim else "off"), ("UpperSaturationLimit", _mat(p["upper"])),
            ("LowerSaturationLimit", _mat(p["lower"])), ("SampleTime", _ts(p["Ts"]))]
    if t == "ZeroOrderHold":
        return "simulink/Discrete/Zero-Order Hold", [("SampleTime", _ts(p["Ts"]))]
    if t == "DiscreteDerivative":            # Simulink'te örnekleme zamanı girişten miras alınır
        assert p["Ts"].is_inherited
        return "simulink/Discrete/Discrete Derivative", [("gainval", _num(p["K"])),
                                                         ("ICPrevScaledInput", _mat(p["x0"]))]
    if t == "Memory":
        return "simulink/Discrete/Memory", [("InitialCondition", _mat(p["x0"]))]
    if t == "Saturation":
        return "simulink/Discontinuities/Saturation", [("UpperLimit", _mat(p["upper"])),
                                                       ("LowerLimit", _mat(p["lower"]))]
    if t == "DeadZone":
        return "simulink/Discontinuities/Dead Zone", [("LowerValue", _mat(p["start"])), ("UpperValue", _mat(p["end"]))]
    if t == "RateLimiter":
        mode = "continuous" if p["Ts"].is_continuous else "inherited"
        return "simulink/Discontinuities/Rate Limiter", [("RisingSlewLimit", _mat(p["rising"])),
                                                         ("FallingSlewLimit", _mat(p["falling"])),
                                                         ("InitialCondition", _mat(p["x0"])),
                                                         ("SampleTimeMode", mode)]
    if t == "Quantizer":
        return "simulink/Discontinuities/Quantizer", [("QuantizationInterval", _mat(p["interval"]))]
    if t == "Relay":
        return "simulink/Discontinuities/Relay", [("OnSwitchValue", _mat(p["on_point"])),
                                                  ("OffSwitchValue", _mat(p["off_point"])),
                                                  ("OnOutputValue", _mat(p["on_value"])),
                                                  ("OffOutputValue", _mat(p["off_value"]))]
    if t == "Abs":
        return "simulink/Math Operations/Abs", []
    if t == "Sign":
        return "simulink/Math Operations/Sign", []
    if t == "Lookup1D":
        im = {"linear": "Linear point-slope", "flat": "Flat", "nearest": "Nearest"}[p["interp"]]
        em = {"linear": "Linear", "clip": "Clip"}[p["extrap"]]
        return "simulink/Lookup Tables/1-D Lookup Table", [("Table", _mat(p["table"])),
                                                           ("BreakpointsForDimension1", _mat(p["breakpoints"])),
                                                           ("InterpMethod", im), ("ExtrapMethod", em)]
    if t == "Switch":
        cr = {"ge": "u2 >= Threshold", "gt": "u2 > Threshold", "ne0": "u2 ~= 0"}[p["criteria"]]
        return "simulink/Signal Routing/Switch", [("Criteria", cr), ("Threshold", _num(p["threshold"]))]
    if t == "MinMax":
        return "simulink/Math Operations/MinMax", [("Function", p["function"]), ("Inputs", str(p["inputs"]))]
    if t == "Fcn":
        return "simulink/User-Defined Functions/Fcn", [("Expr", p["expr"])]
    if t == "Mux":
        return "simulink/Signal Routing/Mux", [("Inputs", str(p["inputs"]))]
    if t == "Demux":
        return "simulink/Signal Routing/Demux", [("Outputs", _mat(p["widths"]) if p["widths"].size
                                                  else str(p["outputs"]))]
    if t == "Selector":
        return "simulink/Signal Routing/Selector", [("IndexOptions", "Index vector (dialog)"),
                                                    ("Indices", _mat(p["indices"])),
                                                    ("InputPortWidth", str(b.in_widths[0] if b.in_widths else -1))]
    if t == "Feedforward":
        st = p["Ts"]
        if p["num"].size == 1 and p["den"].size == 1:
            return "simulink/Math Operations/Gain", [("Gain", _num(p["num"][0] / p["den"][0]))]
        if st.is_discrete:
            return "simulink/Discrete/Discrete Transfer Fcn", [("Numerator", _mat(p["num"])),
                                                               ("Denominator", _mat(p["den"])),
                                                               ("SampleTime", _ts(st))]
        return "simulink/Continuous/Transfer Fcn", [("Numerator", _mat(p["num"])), ("Denominator", _mat(p["den"]))]
    if t == "PIDController":
        return _pid(p, False)
    if t == "PIDController2DOF":
        return _pid(p, True)
    if t in ("Scope", "Terminator"):
        return "simulink/Sinks/Terminator", []
    if t == "Outport":
        return "simulink/Sinks/Out1", [("Port", str(p["port"]))]
    raise KeyError(f"Simulink karşılığı tanımsız: {t}")


def build_lines(model: Model, mdl: str, probes: list[tuple[str, int]] = ()) -> tuple[list[str], dict[str, str]]:
    """Simulink modelini kuran MATLAB satırları ve blok adı eşlemesi (bizim ad → 'bN')."""
    from spt.control.diagram.engine import compile_model
    cm = compile_model(model)        # genişlikler (Selector) için
    names = list(model.blocks)
    sl = {n: f"b{k + 1}" for k, n in enumerate(names)}
    lines = [f"mdl = '{mdl}';", "new_system(mdl);"]
    data_vars: list[str] = []
    for n in names:
        b = cm.blocks[cm.index[n]] if n in cm.index else model.blocks[n]
        lib, prm = _block(b, data_vars, sl[n])
        if b.params.get("linearize_as_gain") is False:
            prm.append(("LinearizeAsGain", "off"))
        args = "".join(f", '{k}', '{v}'" for k, v in prm)
        lines.append(f"add_block('{lib}', [mdl '/{sl[n]}']{args});")
    for c in model.connections:
        lines.append(f"add_line(mdl, '{sl[c.src]}/{c.src_port + 1}', '{sl[c.dst]}/{c.dst_port + 1}');")
    for k, (bn, port) in enumerate(probes):
        lines.append(f"add_block('simulink/Sinks/Out1', [mdl '/probe{k + 1}'], 'Port', '{k + 1}');")
        lines.append(f"add_line(mdl, '{sl[bn]}/{port + 1}', 'probe{k + 1}/1');")
    lines += data_vars
    return lines, sl


def _to_simulink(model: Model, mdl: str, probes: list[tuple[str, int]], settings: dict[str, Any]) -> str:
    """Modeli Simulink betiğine çevirir; probes sinyalleri Out1 bloklarıyla yout sütunlarına alınır."""
    lines, _ = build_lines(model, mdl, probes)
    opts = {"StopTime": _num(settings["stop"]), "SaveOutput": "on", "SaveFormat": "Array", "SaveTime": "on",
            "ReturnWorkspaceOutputs": "on", "AlgebraicLoopMsg": "error"}
    if settings["solver"] == "ode45":
        opts.update({"Solver": "ode45", "RelTol": _num(settings["rtol"]), "AbsTol": _num(settings["atol"]),
                     "MaxStep": _num(settings.get("max_step", settings["stop"] / 50)),
                     "OutputOption": "SpecifiedOutputTimes", "OutputTimes": _mat(settings["times"]),
                     "ZeroCrossControl": "UseLocalSettings" if settings.get("zc", True) else "DisableAll"})
    else:
        opts.update({"Solver": settings["solver"], "FixedStep": _num(settings["step"])})
    lines.append("set_param(mdl" + "".join(f", '{k}', '{v}'" for k, v in opts.items()) + ");")
    lines.append("so = sim(mdl);")
    lines.append(f"out.{mdl}_t = so.tout; out.{mdl}_y = so.yout;")
    lines.append("close_system(mdl, 0);")
    body = "\n".join("  " + ln for ln in lines)
    return (f"try\n{body}\ncatch e\n  out.{mdl}_err = e.message;\n  try close_system('{mdl}', 0); catch; end\n"
            "end")


# ============================================================================ vaka kaydı
Case = tuple[Model, list[tuple[str, int]], dict[str, Any]]
CASES: dict[str, dict[str, Callable[[], Case]]] = {}


def case(group: str):
    def deco(fn):
        CASES.setdefault(group, {})[fn.__name__] = fn
        return fn
    return deco


def _fixed(stop, step, solver="ode4"):
    return {"solver": solver, "stop": stop, "step": step}


def _var(stop, dt, rtol=1e-10, atol=1e-12, zc=True, max_step=None):
    s = {"solver": "ode45", "stop": stop, "rtol": rtol, "atol": atol, "times": np.round(np.arange(0, stop + dt / 2, dt),
                                                                                        12), "zc": zc}
    if max_step:
        s["max_step"] = max_step
    return s


def _loop(m: Model, ref, ctrl, plant, *, sum_name="E") -> None:
    """ref → (+) → ctrl → plant → (−) geri besleme."""
    s = m.add("Sum", sum_name, signs="+-")
    m.connect(ref, s)
    m.connect(s, ctrl)
    m.connect(ctrl, plant)
    m.connect(plant, s, dst_port=1)


# ---------------------------------------------------------------- A: PID ve LTI (sabit adım)
@case("pid")
def pid_cont_parallel():
    m = Model()
    r = m.add("Step", "R", step_time=0.1)
    c = m.add("PIDController", "C", P=2.0, I=3.0, D=0.1, N=20.0)
    g = m.add("TransferFcn", "G", num=[1.0], den=[1.0, 2.0, 1.0])
    _loop(m, r, c, g)
    return m, [("G", 0), ("C", 0)], _fixed(5.0, 0.01)


@case("pid")
def pid_disc_fe():
    m = Model()
    r = m.add("Step", "R", step_time=0.0)
    c = m.add("PIDController", "C", P=1.5, I=2.0, D=0.2, N=15.0, Ts=0.05)
    g = m.add("TransferFcn", "G", num=[2.0], den=[1.0, 3.0, 2.0])
    _loop(m, r, c, g)
    return m, [("G", 0), ("C", 0)], _fixed(4.0, 0.01)


@case("pid")
def pid_disc_be_clamp():
    m = Model()
    r = m.add("Step", "R", step_time=0.0, final=2.0)
    c = m.add("PIDController", "C", P=3.0, I=4.0, D=0.3, N=30.0, Ts=0.02, integrator_method="backward_euler",
              filter_method="backward_euler", limit_output=True, upper=1.5, lower=-1.0, anti_windup="clamping")
    g = m.add("TransferFcn", "G", num=[1.0], den=[1.0, 1.0, 0.0])
    _loop(m, r, c, g)
    return m, [("G", 0), ("C", 0)], _fixed(8.0, 0.01)


@case("pid")
def pid_disc_trap_backcalc():
    m = Model()
    r = m.add("Step", "R", step_time=0.0, final=2.0)
    c = m.add("PIDController", "C", P=3.0, I=4.0, D=0.3, N=30.0, Ts=0.02, integrator_method="trapezoidal",
              filter_method="trapezoidal", limit_output=True, upper=1.5, lower=-1.0, anti_windup="back_calculation",
              Kb=2.0)
    g = m.add("TransferFcn", "G", num=[1.0], den=[1.0, 1.0, 0.0])
    _loop(m, r, c, g)
    return m, [("G", 0), ("C", 0)], _fixed(8.0, 0.01)


@case("pid")
def pid_disc_be_backcalc():
    m = Model()
    r = m.add("Step", "R", step_time=0.0, final=2.0)
    c = m.add("PIDController", "C", P=3.0, I=4.0, D=0.3, N=30.0, Ts=0.02, integrator_method="backward_euler",
              filter_method="forward_euler", limit_output=True, upper=1.5, lower=-1.0, anti_windup="back_calculation",
              Kb=2.0, xf0=0.2, xi0=0.1)
    g = m.add("TransferFcn", "G", num=[1.0], den=[1.0, 1.0, 0.0])
    _loop(m, r, c, g)
    return m, [("G", 0), ("C", 0)], _fixed(8.0, 0.01)


@case("pid")
def pid_disc_trap_clamp_ideal():
    m = Model()
    r = m.add("Step", "R", step_time=0.0, final=2.0)
    c = m.add("PIDController", "C", form="ideal", P=2.0, I=1.5, D=0.15, N=30.0, Ts=0.02,
              integrator_method="trapezoidal", filter_method="backward_euler", limit_output=True, upper=1.5,
              lower=-1.0, anti_windup="clamping")
    g = m.add("TransferFcn", "G", num=[1.0], den=[1.0, 1.0, 0.0])
    _loop(m, r, c, g)
    return m, [("G", 0), ("C", 0)], _fixed(8.0, 0.01)


@case("pid")
def pid_cont_ideal_backcalc():
    m = Model()
    r = m.add("Step", "R", step_time=0.0, final=2.0)
    c = m.add("PIDController", "C", form="ideal", P=2.0, I=1.5, D=0.15, N=30.0, limit_output=True, upper=1.5,
              lower=-1.0, anti_windup="back_calculation", Kb=1.5)
    g = m.add("TransferFcn", "G", num=[1.0], den=[1.0, 1.0, 0.0])
    _loop(m, r, c, g)
    return m, [("G", 0), ("C", 0)], _fixed(8.0, 0.005)


@case("pid")
def pid_cont_clamp():
    m = Model()
    r = m.add("Step", "R", step_time=0.0, final=2.0)
    c = m.add("PIDController", "C", P=3.0, I=4.0, D=0.3, N=30.0, limit_output=True, upper=1.5, lower=-1.0,
              anti_windup="clamping", xi0=0.3, xf0=-0.1)
    g = m.add("TransferFcn", "G", num=[1.0], den=[1.0, 1.0, 0.0])
    _loop(m, r, c, g)
    return m, [("G", 0), ("C", 0)], _fixed(8.0, 0.005)


@case("pid")
def pid_2dof_clamp():
    m = Model()
    r = m.add("Step", "R", step_time=0.2, final=1.0)
    c = m.add("PIDController2DOF", "C", P=3.0, I=4.0, D=0.3, N=30.0, b=0.6, c=0.0, limit_output=True, upper=2.0,
              lower=-2.0, anti_windup="clamping")
    g = m.add("TransferFcn", "G", num=[1.0], den=[1.0, 1.0, 0.0])
    m.connect(r, c, 0, 0)
    m.connect(c, g)
    m.connect(g, c, 0, 1)
    return m, [("G", 0), ("C", 0)], _fixed(8.0, 0.005)


@case("pid")
def pid_2dof_disc_ideal():
    m = Model()
    r = m.add("Step", "R", step_time=0.2, final=1.0)
    c = m.add("PIDController2DOF", "C", form="ideal", P=2.0, I=2.0, D=0.25, N=20.0, b=0.5, c=0.3, Ts=0.01,
              integrator_method="backward_euler", filter_method="trapezoidal")
    g = m.add("TransferFcn", "G", num=[1.0], den=[1.0, 1.0, 0.0])
    m.connect(r, c, 0, 0)
    m.connect(c, g)
    m.connect(g, c, 0, 1)
    return m, [("G", 0), ("C", 0)], _fixed(6.0, 0.005)


@case("pid")
def pid_tracking_disc_be():
    m = Model()
    r = m.add("Step", "R", step_time=0.0, final=1.0)
    tr = m.add("SineWave", "TR", amplitude=0.5, frequency=2.0, Ts=0.0)
    c = m.add("PIDController", "C", P=2.0, I=3.0, D=0.0, controller="PI", Ts=0.05, integrator_method="backward_euler",
              limit_output=True, upper=1.2, lower=-1.2, anti_windup="back_calculation", Kb=1.0, tracking=True,
              Kt=0.7)
    g = m.add("TransferFcn", "G", num=[1.0], den=[1.0, 1.0])
    s = m.add("Sum", "E", signs="+-")
    m.connect(r, s)
    m.connect(s, c, 0, 0)
    m.connect(tr, c, 0, 1)
    m.connect(c, g)
    m.connect(g, s, 0, 1)
    return m, [("G", 0), ("C", 0)], _fixed(6.0, 0.01)


@case("pid")
def pid_tracking_cont_clamp():
    m = Model()
    r = m.add("Step", "R", step_time=0.0, final=1.0)
    tr = m.add("SineWave", "TR", amplitude=0.5, frequency=2.0)
    c = m.add("PIDController", "C", P=2.0, I=3.0, D=0.1, N=40.0, limit_output=True, upper=1.2, lower=-1.2,
              anti_windup="clamping", tracking=True, Kt=0.7)
    g = m.add("TransferFcn", "G", num=[1.0], den=[1.0, 1.0])
    s = m.add("Sum", "E", signs="+-")
    m.connect(r, s)
    m.connect(s, c, 0, 0)
    m.connect(tr, c, 0, 1)
    m.connect(c, g)
    m.connect(g, s, 0, 1)
    return m, [("G", 0), ("C", 0)], _fixed(6.0, 0.005)


@case("pid")
def pid_variants_unfiltered():
    m = Model()
    r = m.add("SineWave", "R", amplitude=1.0, frequency=1.3)
    specs = [("PI", dict(controller="PI", P=1.0, I=2.0)), ("PD", dict(controller="PD", P=1.0, D=0.4, N=10.0)),
             ("P", dict(controller="P", P=1.7)), ("I", dict(controller="I", I=2.5)),
             ("DU", dict(controller="PID", P=1.0, I=1.0, D=0.05, use_filter=False, Ts=0.02, xf0=0.3, xi0=-0.2)),
             ("IDI", dict(controller="PID", form="ideal", P=1.4, I=0.8, D=0.1, N=25.0, Ts=0.02,
                          integrator_method="trapezoidal", filter_method="forward_euler", xi0=0.4, xf0=0.25))]
    probes = []
    for nm, kw in specs:
        c = m.add("PIDController", "C" + nm, **kw)
        g = m.add("TransferFcn", "G" + nm, num=[1.0], den=[1.0, 2.0, 2.0])
        _loop(m, r, c, g, sum_name="E" + nm)
        probes += [("G" + nm, 0), ("C" + nm, 0)]
    return m, probes, _fixed(5.0, 0.01)


@case("pid")
def ff_plus_fb_multirate():
    m = Model()
    r = m.add("SineWave", "R", amplitude=1.0, frequency=0.8)
    ff = m.add("Feedforward", "FF", num=[0.5], den=[1.0])
    c = m.add("PIDController", "C", P=4.0, I=2.0, D=0.0, controller="PI", Ts=0.1)
    zoh = m.add("ZeroOrderHold", "Z", Ts=0.1)
    u = m.add("Sum", "U", signs="++")
    g = m.add("ZeroPole", "G", zeros=[-3.0], poles=[-1.0 + 2.0j, -1.0 - 2.0j], gain=2.0)
    s = m.add("Sum", "E", signs="+-")
    m.connect(r, s)
    m.connect(s, zoh)
    m.connect(zoh, c)
    m.connect(r, ff)
    m.connect(c, u, 0, 0)
    m.connect(ff, u, 0, 1)
    m.connect(u, g)
    m.connect(g, s, 0, 1)
    return m, [("G", 0), ("U", 0)], _fixed(6.0, 0.02)


@case("pid")
def lti_blocks_mix():
    m = Model()
    r = m.add("Step", "R", step_time=0.3, initial=0.5, final=-1.0)
    ss = m.add("StateSpace", "SS", A=[[0.0, 1.0], [-4.0, -0.8]], B=[[0.0], [1.0]], C=[[1.0, 0.0], [0.0, 1.0]],
               D=[[0.0], [0.5]], x0=[0.2, -0.1])
    dm = m.add("Demux", "DM", outputs=2)
    tf = m.add("TransferFcn", "TF", num=[1.0, 2.0], den=[1.0, 3.0, 5.0])
    fd = m.add("FilteredDerivative", "FD", N=40.0)
    ig = m.add("Integrator", "I", x0=0.3)
    gn = m.add("Gain", "K", K=[[1.0, -2.0], [0.5, 3.0]], mode="matrix")
    mx = m.add("Mux", "MX", inputs=2)
    m.connect(r, ss)
    m.connect(ss, dm)
    m.connect(dm, tf, 0, 0)
    m.connect(dm, fd, 1, 0)
    m.connect(fd, ig)
    m.connect(tf, mx, 0, 0)
    m.connect(ig, mx, 0, 1)
    m.connect(mx, gn)
    m.add("Terminator", "T1")
    m.connect(gn, "T1")
    return m, [("K", 0), ("TF", 0), ("I", 0)], _fixed(4.0, 0.002)


# ---------------------------------------------------------------- B: doğrusal olmayanlar (sabit adım)
@case("nonlin")
def nl_saturation_deadzone_loop():
    m = Model()
    r = m.add("SineWave", "R", amplitude=2.0, frequency=1.0)
    c = m.add("Gain", "K", K=5.0)
    sat = m.add("Saturation", "SAT", upper=1.0, lower=-0.7)
    dz = m.add("DeadZone", "DZ", start=-0.1, end=0.2)
    g = m.add("TransferFcn", "G", num=[1.0], den=[1.0, 1.0])
    s = m.add("Sum", "E", signs="+-")
    m.connect(r, s)
    m.connect(s, c)
    m.connect(c, sat)
    m.connect(sat, dz)
    m.connect(dz, g)
    m.connect(g, s, 0, 1)
    return m, [("G", 0), ("SAT", 0), ("DZ", 0)], _fixed(10.0, 0.01)


@case("nonlin")
def nl_rate_limiter_quantizer():
    m = Model()
    r = m.add("PulseGenerator", "R", amplitude=1.5, period=2.0, width=50.0, delay=0.25)
    rl = m.add("RateLimiter", "RL", rising=0.8, falling=-1.6)
    q = m.add("Quantizer", "Q", interval=0.25)
    g = m.add("TransferFcn", "G", num=[3.0], den=[1.0, 2.0, 3.0])
    m.connect(r, rl)
    m.connect(rl, q)
    m.connect(q, g)
    return m, [("RL", 0), ("Q", 0), ("G", 0)], _fixed(6.0, 0.01)


@case("nonlin")
def nl_relay_loop():
    m = Model()
    r = m.add("Constant", "R", value=1.0)
    rel = m.add("Relay", "REL", on_point=0.05, off_point=-0.05, on_value=2.0, off_value=-2.0)
    g = m.add("TransferFcn", "G", num=[1.0], den=[1.0, 1.0, 0.0])
    _loop(m, r, rel, g)
    return m, [("G", 0), ("REL", 0)], _fixed(6.0, 0.01)


@case("nonlin")
def nl_transport_delay_loop():
    m = Model()
    r = m.add("Step", "R", step_time=0.0)
    c = m.add("PIDController", "C", P=0.8, I=0.5, D=0.0, controller="PI")
    g = m.add("TransferFcn", "G", num=[1.0], den=[2.0, 1.0])
    td = m.add("TransportDelay", "TD", delay=0.37, initial=0.0)
    s = m.add("Sum", "E", signs="+-")
    m.connect(r, s)
    m.connect(s, c)
    m.connect(c, td)
    m.connect(td, g)
    m.connect(g, s, 0, 1)
    return m, [("G", 0), ("TD", 0)], _fixed(15.0, 0.01)


@case("nonlin")
def nl_small_delay():
    m = Model()
    r = m.add("SineWave", "R", amplitude=1.0, frequency=2.0)
    td = m.add("TransportDelay", "TD", delay=0.03, initial=0.2)
    ig = m.add("Integrator", "I")
    m.connect(r, td)
    m.connect(td, ig)
    return m, [("TD", 0), ("I", 0)], _fixed(2.0, 0.1)


@case("nonlin")
def nl_math_blocks():
    m = Model()
    s1 = m.add("SineWave", "S1", amplitude=1.5, frequency=1.7)
    s2 = m.add("SineWave", "S2", amplitude=1.0, frequency=0.6, phase=0.4, bias=0.2)
    ab = m.add("Abs", "ABS")
    sg = m.add("Sign", "SGN")
    mm = m.add("MinMax", "MM", function="min", inputs=2)
    mx = m.add("MinMax", "MX", function="max", inputs=2)
    sw = m.add("Switch", "SW", criteria="gt", threshold=0.3)
    lut = m.add("Lookup1D", "LUT", breakpoints=[-1.0, 0.0, 0.5, 1.2], table=[2.0, 0.0, 1.0, -1.0])
    lutc = m.add("Lookup1D", "LUTC", breakpoints=[-1.0, 0.0, 0.5, 1.2], table=[2.0, 0.0, 1.0, -1.0], extrap="clip")
    pr = m.add("Product", "PR", ops="*/")
    c = m.add("Constant", "C", value=2.5)
    fc = m.add("Fcn", "F", expr="u(1)^2 + sin(u(2)) - 0.5*abs(u(1)*u(2))")
    mux = m.add("Mux", "MUX", inputs=2)
    ig = m.add("Integrator", "I", x0=0.0, upper=0.4, lower=-0.3)
    m.connect(s1, ab)
    m.connect(s1, sg)
    m.connect(s1, mm, 0, 0)
    m.connect(s2, mm, 0, 1)
    m.connect(s1, mx, 0, 0)
    m.connect(s2, mx, 0, 1)
    m.connect(s1, sw, 0, 0)
    m.connect(s2, sw, 0, 1)
    m.connect(c, sw, 0, 2)
    m.connect(s1, lut)
    m.connect(s1, lutc)
    m.connect(s1, pr, 0, 0)
    m.connect(c, pr, 0, 1)
    m.connect(s1, mux, 0, 0)
    m.connect(s2, mux, 0, 1)
    m.connect(mux, fc)
    m.connect(fc, ig)
    probes = [("ABS", 0), ("SGN", 0), ("MM", 0), ("MX", 0), ("SW", 0), ("LUT", 0), ("LUTC", 0), ("PR", 0), ("F", 0),
              ("I", 0)]
    return m, probes, _fixed(6.0, 0.01)


@case("nonlin")
def nl_ode1_loop():
    m = Model()
    r = m.add("Step", "R", step_time=0.05)
    sat = m.add("Saturation", "SAT", upper=0.8, lower=-0.8)
    c = m.add("PIDController", "C", P=4.0, I=3.0, D=0.1, N=20.0)
    g = m.add("TransferFcn", "G", num=[1.0], den=[1.0, 1.5, 1.0])
    s = m.add("Sum", "E", signs="+-")
    m.connect(r, s)
    m.connect(s, c)
    m.connect(c, sat)
    m.connect(sat, g)
    m.connect(g, s, 0, 1)
    return m, [("G", 0), ("SAT", 0)], _fixed(5.0, 0.002, solver="ode1")


@case("nonlin")
def nl_memory_loop_and_variants():
    m = Model()
    s = m.add("SineWave", "S", amplitude=1.0, frequency=2.5)
    e = m.add("Sum", "E", signs="+-")
    ig = m.add("Integrator", "I")
    mem = m.add("Memory", "MEM", x0=0.4)
    k = m.add("Gain", "K", K=3.0)
    m.connect(s, e)
    m.connect(e, ig)
    m.connect(ig, mem)
    m.connect(mem, k)
    m.connect(k, e, dst_port=1)
    lf = m.add("Lookup1D", "LF", breakpoints=[-1.0, -0.2, 0.3, 0.9], table=[1.0, -1.0, 0.5, 2.0], interp="flat")
    ln = m.add("Lookup1D", "LN", breakpoints=[-1.0, -0.2, 0.3, 0.9], table=[1.0, -1.0, 0.5, 2.0], interp="nearest")
    sw = m.add("Switch", "SW", criteria="ne0")
    q = m.add("Quantizer", "Q", interval=0.5)
    c = m.add("Constant", "C", value=-0.3)
    m.connect(s, lf)
    m.connect(s, ln)
    m.connect(s, q)
    m.connect(s, sw, dst_port=0)
    m.connect(q, sw, dst_port=1)
    m.connect(c, sw, dst_port=2)
    return m, [("I", 0), ("MEM", 0), ("LF", 0), ("LN", 0), ("SW", 0)], _fixed(4.0, 0.01)


# ---------------------------------------------------------------- C: ayrık ve çok hızlı
@case("discrete")
def disc_blocks_multirate():
    m = Model()
    s = m.add("SineWave", "S", amplitude=1.0, frequency=2.0)
    z1 = m.add("ZeroOrderHold", "Z1", Ts=0.1)
    dtf = m.add("DiscreteTransferFcn", "DTF", num=[0.2, 0.1], den=[1.0, -1.2, 0.5], Ts=0.1)
    dfl = m.add("DiscreteFilter", "DFL", num=[0.3, 0.2], den=[1.0, -0.4, 0.1], Ts=0.05)
    dss = m.add("DiscreteStateSpace", "DSS", A=[[0.9, 0.1], [-0.2, 0.7]], B=[[1.0], [0.5]], C=[[1.0, -1.0]],
                D=[[0.2]], x0=[0.1, -0.3], Ts=0.025)
    ud = m.add("UnitDelay", "UD", x0=-0.5, Ts=0.1)
    z2 = m.add("ZeroOrderHold", "Z2", Ts=0.05)
    dd = m.add("DiscreteDerivative", "DD", K=0.5, x0=0.2, Ts=-1.0)
    mem = m.add("Memory", "MEM", x0=0.7)
    m.connect(s, z1)
    m.connect(z1, dtf)
    m.connect(s, dfl)
    m.connect(s, dss)
    m.connect(dtf, ud)
    m.connect(s, z2)
    m.connect(z2, dd)
    m.connect(s, mem)
    probes = [("Z1", 0), ("DTF", 0), ("DFL", 0), ("DSS", 0), ("UD", 0), ("DD", 0), ("MEM", 0)]
    return m, probes, _fixed(2.0, 0.025)


@case("discrete")
def disc_integrators_limits():
    m = Model()
    s = m.add("SineWave", "S", amplitude=2.0, frequency=1.5)
    probes = []
    for k, meth in enumerate(["forward_euler", "backward_euler", "trapezoidal"]):
        d = m.add("DiscreteIntegrator", f"D{k}", method=meth, K=1.3, x0=0.2, upper=0.8, lower=-0.6, Ts=0.05)
        m.connect(s, d)
        probes.append((f"D{k}", 0))
    return m, probes, _fixed(6.0, 0.05, solver="ode4")


@case("discrete")
def disc_ctrl_cont_plant_two_rates():
    m = Model()
    r = m.add("Step", "R", step_time=0.0)
    c1 = m.add("DiscreteTransferFcn", "C1", num=[2.0, -1.6], den=[1.0, -1.0], Ts=0.1)
    c2 = m.add("DiscreteTransferFcn", "C2", num=[1.0, -0.5], den=[1.0, 0.2], Ts=0.02)
    g = m.add("TransferFcn", "G", num=[1.0], den=[1.0, 1.2, 1.0])
    s = m.add("Sum", "E", signs="+-")
    m.connect(r, s)
    m.connect(s, c1)
    m.connect(c1, c2)
    m.connect(c2, g)
    m.connect(g, s, 0, 1)
    return m, [("G", 0), ("C1", 0), ("C2", 0)], _fixed(8.0, 0.01)


@case("discrete")
def disc_rate_limiter_inherited():
    m = Model()
    s = m.add("SineWave", "S", amplitude=1.0, frequency=2.0)
    z = m.add("ZeroOrderHold", "Z", Ts=0.1)
    rl = m.add("RateLimiter", "RL", rising=1.0, falling=-2.0, x0=0.5, Ts=-1.0)
    m.connect(s, z)
    m.connect(z, rl)
    return m, [("RL", 0), ("Z", 0)], _fixed(3.0, 0.05)


@case("discrete")
def disc_data_source():
    m = Model()
    t = np.arange(0.0, 3.0001, 0.25)
    v = np.column_stack([np.sin(1.3 * t), np.cos(0.7 * t) ** 2])
    d1 = m.add("DataSource", "D1", time=t, values=v, interp="linear")
    d2 = m.add("DataSource", "D2", time=t, values=v[:, :1], interp="zoh")
    g = m.add("TransferFcn", "G", num=[1.0], den=[0.5, 1.0])
    sel = m.add("Selector", "SEL", indices=[2])
    m.connect(d1, sel)
    m.connect(sel, g)
    m.add("Terminator", "T")
    m.connect(d2, "T")
    return m, [("D1", 0), ("D2", 0), ("G", 0)], _fixed(3.0, 0.05)


# ---------------------------------------------------------------- D: değişken adım (ode45)
@case("ode45")
def v_pid_loop():
    m = Model()
    r = m.add("Step", "R", step_time=0.33)
    c = m.add("PIDController", "C", P=2.0, I=3.0, D=0.1, N=20.0)
    g = m.add("TransferFcn", "G", num=[1.0], den=[1.0, 2.0, 1.0])
    _loop(m, r, c, g)
    return m, [("G", 0), ("C", 0)], _var(5.0, 0.05)


@case("ode45")
def v_saturation_loop():
    m = Model()
    r = m.add("SineWave", "R", amplitude=2.0, frequency=1.0)
    k = m.add("Gain", "K", K=5.0)
    sat = m.add("Saturation", "SAT", upper=1.0, lower=-0.7)
    g = m.add("TransferFcn", "G", num=[1.0], den=[1.0, 1.0])
    s = m.add("Sum", "E", signs="+-")
    m.connect(r, s)
    m.connect(s, k)
    m.connect(k, sat)
    m.connect(sat, g)
    m.connect(g, s, 0, 1)
    return m, [("G", 0), ("SAT", 0)], _var(10.0, 0.05)


@case("ode45")
def v_relay_loop():
    m = Model()
    r = m.add("Constant", "R", value=1.0)
    rel = m.add("Relay", "REL", on_point=0.05, off_point=-0.05, on_value=2.0, off_value=-2.0)
    g = m.add("TransferFcn", "G", num=[1.0], den=[1.0, 1.0, 0.0])
    _loop(m, r, rel, g)
    return m, [("G", 0), ("REL", 0)], _var(6.0, 0.05)


@case("ode45")
def v_multirate_pid_clamp():
    m = Model()
    r = m.add("Step", "R", step_time=0.0, final=2.0)
    c = m.add("PIDController", "C", P=3.0, I=4.0, D=0.3, N=30.0, Ts=0.02, integrator_method="backward_euler",
              filter_method="backward_euler", limit_output=True, upper=1.5, lower=-1.0, anti_windup="clamping")
    g = m.add("TransferFcn", "G", num=[1.0], den=[1.0, 1.0, 0.0])
    _loop(m, r, c, g)
    return m, [("G", 0), ("C", 0)], _var(8.0, 0.04)


# Not: sürekli PID'de sıkıştırma (clamping) kararı Simulink'te bir ana adım gecikmeli Bellek bloğundan geçer;
# değişken adımda sonuç adım dizisine bağlıdır (Simulink'in kendisi ince adımlı çözümden ~1e-2 sapar). Bu yüzden
# sıkıştırma ode45'te karşılaştırılmaz (sabit adımda birebir: pid_cont_clamp, pid_2dof_clamp …); burada geri
# hesaplama kullanılır.
@case("ode45")
def v_cont_pid_sat_backcalc():
    m = Model()
    r = m.add("Step", "R", step_time=0.0, final=2.0)
    c = m.add("PIDController", "C", P=3.0, I=4.0, D=0.3, N=30.0, limit_output=True, upper=1.5, lower=-1.0,
              anti_windup="back_calculation", Kb=2.5)
    g = m.add("TransferFcn", "G", num=[1.0], den=[1.0, 1.0, 0.0])
    _loop(m, r, c, g)
    return m, [("G", 0), ("C", 0)], _var(8.0, 0.05)


@case("ode45")
def v_sources_pulse_ramp():
    m = Model()
    p = m.add("PulseGenerator", "P", amplitude=1.0, period=0.7, width=30.0, delay=0.15)
    rp = m.add("Ramp", "RP", slope=0.5, start_time=0.42, initial=-0.2)
    s = m.add("Sum", "S", signs="++")
    g = m.add("TransferFcn", "G", num=[4.0], den=[1.0, 0.8, 4.0])
    ig = m.add("Integrator", "I", upper=0.6, lower=-1.0)
    m.connect(p, s, 0, 0)
    m.connect(rp, s, 0, 1)
    m.connect(s, g)
    m.connect(g, ig)
    return m, [("G", 0), ("I", 0)], _var(5.0, 0.05)


@case("ode45")
def v_transport_delay():
    m = Model()
    r = m.add("Step", "R", step_time=0.0)
    c = m.add("PIDController", "C", P=0.8, I=0.5, D=0.0, controller="PI")
    g = m.add("TransferFcn", "G", num=[1.0], den=[2.0, 1.0])
    td = m.add("TransportDelay", "TD", delay=0.37, initial=0.0)
    s = m.add("Sum", "E", signs="+-")
    m.connect(r, s)
    m.connect(s, c)
    m.connect(c, td)
    m.connect(td, g)
    m.connect(g, s, 0, 1)
    return m, [("G", 0)], _var(15.0, 0.1)


@case("ode45")
def v_data_zoh_events():
    m = Model()
    t = np.array([0.0, 0.3, 0.3, 0.75, 1.2, 1.9, 2.5])
    v = np.array([0.0, 1.0, -0.5, 2.0, 0.3, -1.0, 0.0])
    d = m.add("DataSource", "D", time=t, values=v, interp="zoh")
    g = m.add("TransferFcn", "G", num=[5.0], den=[1.0, 2.0, 5.0])
    m.connect(d, g)
    return m, [("G", 0), ("D", 0)], _var(2.5, 0.05)


# ============================================================================ çalıştırma ve karşılaştırma
def _script(group: str) -> str:
    parts = ["warning('off', 'all');"]
    for name, fn in CASES[group].items():
        m, probes, st = fn()
        parts.append(_to_simulink(m, name, probes, st))
    return "\n".join(parts)


_REF: dict[str, dict] = {}


def _reference(group: str) -> dict:
    if group not in _REF:
        try:
            _REF[group] = fixture(f"control_diagram_sim_{group}", _script(group))
        except OracleUnavailable as exc:
            pytest.skip(f"MATLAB referansı yok: {exc}")
    return _REF[group]


def _ours(m: Model, probes, st):
    log = [(b, p) for b, p in probes]
    kw: dict[str, Any] = {"solver": st["solver"]}
    if st["solver"] == "ode45":
        kw.update(rtol=st["rtol"], atol=st["atol"], output_times=st["times"], zero_crossing=st.get("zc", True),
                  max_step=st.get("max_step", st["stop"] / 50))
    else:
        kw.update(step=st["step"])
    res = simulate(m, st["stop"], log=log, **kw)
    keys = list(res.signals)
    cols = [res.signals[k] for k in keys]
    return res.t, np.hstack(cols)


# Toleranslar: sabit adımda yalnız yuvarlama farkı (aynı ayrık denklemler); ode45'te her iki çözücü de
# RelTol 1e-10 / AbsTol 1e-12 ile çalışır, fark tolerans düzeyindedir (adım dizileri farklı).
_TOL = {"pid": 1e-10, "nonlin": 1e-10, "discrete": 1e-11, "ode45": 2e-7}
_SPECIAL = {"v_relay_loop": 1e-6, "v_transport_delay": 1e-6}

PARAMS = [(g, n) for g in CASES for n in CASES[g]]


@pytest.mark.parametrize("group,name", PARAMS, ids=[f"{g}-{n}" for g, n in PARAMS])
def test_simulink(group: str, name: str):
    ref = _reference(group)
    err = ref.get(f"{name}_err")
    assert err is None, f"Simulink hatası: {err}"
    t_ref = np.asarray(ref[f"{name}_t"], dtype=float).ravel()
    y_ref = np.asarray(ref[f"{name}_y"], dtype=float)
    if y_ref.ndim == 1:
        y_ref = y_ref.reshape(-1, 1)
    m, probes, st = CASES[group][name]()
    t, y = _ours(m, probes, st)
    assert t.shape == t_ref.shape, (t.shape, t_ref.shape)
    np.testing.assert_allclose(t, t_ref, rtol=0, atol=1e-12 * max(1.0, t_ref[-1]))
    assert y.shape == y_ref.shape, (y.shape, y_ref.shape)
    scale = max(1.0, float(np.max(np.abs(y_ref))))
    tol = _SPECIAL.get(name, _TOL[group]) * scale
    dev = float(np.max(np.abs(y - y_ref)))
    assert dev <= tol, f"{name}: en büyük fark {dev:.3e} > {tol:.1e}"
