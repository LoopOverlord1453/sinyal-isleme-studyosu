r"""Sabit kanat (Beard & McLain, Aerosonde) — MATLAB R2025b referanslı testler.

MATLAB tarafı (Python'dan bağımsız yazım): aynı denklemler fw_f.m içinde (kuvvetler [cosα −sinα; sinα cosα]·[−D; −L]
döndürmesiyle, yerçekimi Rᵀ·[0 0 mg] ile, pervane C_T(J) / C_Q(J) biçiminde, açısal dinamik J\(M − ω×Jω) ile);
trim fsolve ile 10 bilinmeyenli tam sistemden (α, φ, θ, p, q, r, δe, δa, δr, δt); Jacobian Symbolic Math Toolbox ile
sembolik türetilip noktada değerlendirilir; benzetim ode45 (RelTol = AbsTol = 1e-12).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from spt.control.errors import ControlError
from spt.control.uav.common import DRYDEN_PRESETS, DrydenGust, numjac
from spt.control.uav.fixedwing import (
    FixedWing, FixedWingParams, book_transfer_functions, lateral, longitudinal, mode_analysis,
    transfer_function_coefficients, wind_function,
)


def _mfile(name: str, code: str) -> str:
    lines = [ln.replace('"', '""') for ln in code.strip("\n").splitlines()]
    body = ";\n    ".join(f'"{ln}"' for ln in lines)
    return f"writelines([\n    {body}], fullfile(pwd, '{name}.m'));\n"


FW_F = r"""
function dx = fw_f(x, u, P, w6)
uu = x(4); vv = x(5); ww = x(6); ph = x(7); th = x(8); ps = x(9); p = x(10); q = x(11); r = x(12);
de = u(1); da = u(2); dr = u(3); dt = u(4);
Rx = [1 0 0; 0 cos(ph) -sin(ph); 0 sin(ph) cos(ph)];
Ry = [cos(th) 0 sin(th); 0 1 0; -sin(th) 0 cos(th)];
Rz = [cos(ps) -sin(ps) 0; sin(ps) cos(ps) 0; 0 0 1];
R = Rz*Ry*Rx;
wb = R.'*w6(1:3) + w6(4:6);
ur = uu - wb(1); vr = vv - wb(2); wr = ww - wb(3);
Va = sqrt(ur^2 + vr^2 + wr^2);
al = atan2(wr, ur);
be = asin(vr/Va);
qbar = 0.5*P.rho*Va^2;
e1 = exp(-P.M*(al - P.a0)); e2 = exp(P.M*(al + P.a0));
sig = (1 + e1 + e2)/((1 + e1)*(1 + e2));
CL = (1 - sig)*(P.CL0 + P.CLa*al) + sig*(2*sign(al)*sin(al)^2*cos(al));
if P.dragq
    CD = P.CDp + (P.CL0 + P.CLa*al)^2/(pi*P.e*P.b^2/P.S);
else
    CD = P.CD0 + P.CDa*al;
end
Fl = qbar*P.S*(CL + P.CLq*P.c*q/(2*Va) + P.CLde*de);
Fd = qbar*P.S*(CD + P.CDq*P.c*q/(2*Va) + P.CDde*de);
if P.prop
    Vin = P.Vmax*dt;
    a = P.rho*P.D^5*P.CQ0/(4*pi^2);
    b = P.rho*P.D^4*P.CQ1*Va/(2*pi) + P.KQ*P.KV/P.Rm;
    c = P.rho*P.D^3*P.CQ2*Va^2 - P.KQ*Vin/P.Rm + P.KQ*P.i0;
    Om = (-b + sqrt(b^2 - 4*a*c))/(2*a);
    J = 2*pi*Va/(Om*P.D);
    n = Om/(2*pi);
    Tp = P.rho*n^2*P.D^4*(P.CT2*J^2 + P.CT1*J + P.CT0);
    Qp = P.rho*n^2*P.D^5*(P.CQ2*J^2 + P.CQ1*J + P.CQ0);
else
    Tp = 0.5*P.rho*P.Sprop*P.Cprop*((P.kmotor*dt)^2 - Va^2);
    Qp = P.kTp*(P.kOm*dt)^2;
end
fxz = [cos(al) -sin(al); sin(al) cos(al)]*[-Fd; -Fl];
fy = qbar*P.S*(P.CY0 + P.CYb*be + P.CYp*P.b*p/(2*Va) + P.CYr*P.b*r/(2*Va) + P.CYda*da + P.CYdr*dr);
F = R.'*[0; 0; P.mass*P.g] + [fxz(1) + Tp; fy; fxz(2)];
L = qbar*P.S*P.b*(P.Cl0 + P.Clb*be + P.Clp*P.b*p/(2*Va) + P.Clr*P.b*r/(2*Va) + P.Clda*da + P.Cldr*dr) - Qp;
Mm = qbar*P.S*P.c*(P.Cm0 + P.Cma*al + P.Cmq*P.c*q/(2*Va) + P.Cmde*de);
N = qbar*P.S*P.b*(P.Cn0 + P.Cnb*be + P.Cnp*P.b*p/(2*Va) + P.Cnr*P.b*r/(2*Va) + P.Cnda*da + P.Cndr*dr);
Jm = [P.Jx 0 -P.Jxz; 0 P.Jy 0; -P.Jxz 0 P.Jz];
w = [p; q; r]; V = [uu; vv; ww];
W = [1 sin(ph)*tan(th) cos(ph)*tan(th); 0 cos(ph) -sin(ph); 0 sin(ph)/cos(th) cos(ph)/cos(th)];
dx = [R*V; -cross(w, V) + F/P.mass; W*w; Jm\([L; Mm; N] - cross(w, Jm*w))];
end
"""

FW_TRIM = r"""
function F = fw_trim_res(z, Va, gam, psid, P)
x = [0; 0; -100; Va*cos(z(1)); 0; Va*sin(z(1)); z(2); z(3); 0; z(4); z(5); z(6)];
d = fw_f(x, z(7:10), P, zeros(6,1));
F = [d(4:6); d(10:12); d(7); d(8); d(9) - psid; -d(3) - Va*sin(gam)];
end
"""

FW_SCRIPT = _mfile("fw_f", FW_F) + _mfile("fw_trim_res", FW_TRIM) + r"""
rehash;
P = struct('mass', 11.0, 'Jx', 0.8244, 'Jy', 1.135, 'Jz', 1.759, 'Jxz', 0.1204, 'S', 0.55, 'b', 2.8956, ...
    'c', 0.18994, 'rho', 1.2682, 'e', 0.9, 'g', 9.81, 'CL0', 0.23, 'CD0', 0.0424, 'Cm0', 0.0135, 'CLa', 5.61, ...
    'CDa', 0.132, 'Cma', -2.74, 'CLq', 7.95, 'CDq', 0.0, 'Cmq', -38.21, 'CLde', 0.13, 'CDde', 0.0135, ...
    'Cmde', -0.99, 'M', 50, 'a0', 0.47, 'CDp', 0.043, 'CY0', 0, 'Cl0', 0, 'Cn0', 0, 'CYb', -0.98, ...
    'Clb', -0.13, 'Cnb', 0.073, 'CYp', 0, 'Clp', -0.51, 'Cnp', 0.069, 'CYr', 0, 'Clr', 0.25, 'Cnr', -0.095, ...
    'CYda', 0.075, 'Clda', 0.17, 'Cnda', -0.011, 'CYdr', 0.19, 'Cldr', 0.0024, 'Cndr', -0.069, ...
    'D', 20*0.0254, 'KV', 60/(2*pi*145), 'KQ', 60/(2*pi*145), 'Rm', 0.042, 'i0', 1.5, 'Vmax', 3.7*12, ...
    'CQ2', -0.01664, 'CQ1', 0.004970, 'CQ0', 0.005230, 'CT2', -0.1079, 'CT1', -0.06044, 'CT0', 0.09357, ...
    'Sprop', 0.2027, 'Cprop', 1, 'kmotor', 80, 'kTp', 0, 'kOm', 0, 'prop', 1, 'dragq', 1);
P2 = P; P2.prop = 0; P2.dragq = 0;
opts = optimoptions('fsolve', 'Display', 'off', 'FunctionTolerance', 1e-16, 'StepTolerance', 1e-16, ...
    'OptimalityTolerance', 1e-16, 'MaxIterations', 2000, 'MaxFunctionEvaluations', 1e5, ...
    'FiniteDifferenceType', 'central');
cases = [25 0 Inf 1; 25 5 Inf 1; 30 0 150 1; 22 3 -200 1; 25 0 Inf 2; 30 0 150 2];
for k = 1:size(cases, 1)
    Va = cases(k,1); gam = cases(k,2)*pi/180; Rad = cases(k,3);
    if cases(k,4) == 1, PP = P; else, PP = P2; end
    psid = 0; phi0 = 0;
    if isfinite(Rad), psid = Va*cos(gam)/Rad; phi0 = atan(Va^2*cos(gam)/(PP.g*Rad)); end
    fun = @(z) fw_trim_res(z, Va, gam, psid, PP);
    z = fsolve(fun, [0.05; phi0; 0.05 + gam; 0; 0; 0; -0.1; 0; 0; 0.5], opts);
    for it = 1:4
        Jn = zeros(10);
        for j = 1:10
            dz = zeros(10,1); dz(j) = 1e-6;
            Jn(:,j) = (fun(z + dz) - fun(z - dz))/2e-6;
        end
        z = z - Jn\fun(z);
    end
    out.trim{k} = struct('z', z, 'F', fun(z), 'case', cases(k,:));
end
xs = sym('x', [12 1], 'real'); us = sym('u', [4 1], 'real');
fs = fw_f(xs, us, P, zeros(6,1));
JA = jacobian(fs, xs); JB = jacobian(fs, us);
X1 = [10; -5; -120; 24; 1.5; 2.0; 0.1; 0.08; 0.5; 0.05; -0.03; 0.08]; U1 = [-0.1; 0.02; -0.01; 0.75];
out.j1.A = double(subs(JA, [xs; us], [X1; U1])); out.j1.B = double(subs(JB, [xs; us], [X1; U1]));
out.j1.f = double(subs(fs, [xs; us], [X1; U1]));
z = out.trim{3}.z;
Xt = [0; 0; -100; 30*cos(z(1)); 0; 30*sin(z(1)); z(2); z(3); 0; z(4); z(5); z(6)];
out.jt.A = double(subs(JA, [xs; us], [Xt; z(7:10)])); out.jt.B = double(subs(JB, [xs; us], [Xt; z(7:10)]));
x0 = [0; 0; -100; 24.9; 0.3; 1.3; 0.05; 0.06; 0.2; 0.02; -0.01; 0.03];
uf = @(t) [-0.124 + 0.05*sin(1.5*t)*exp(-0.3*t); 0.02*sin(2.1*t); -0.01*cos(0.8*t); 0.77 + 0.1*sin(0.5*t)];
wf = @(t) [3; -2; 0.5; 0.8*sin(0.9*t); 0.5*cos(1.3*t); 0.3*sin(2.2*t)];
oo = odeset('RelTol', 1e-12, 'AbsTol', 1e-12);
[tt, X] = ode45(@(t, x) fw_f(x, uf(t), P, wf(t)), 0:0.25:8, x0, oo);
out.sim1.t = tt; out.sim1.x = X;
uf2 = @(t) [-0.124 + 0.03*sin(2*t); 0.01*sin(1.1*t); 0.005*sin(0.7*t); 0.34 + 0.05*cos(0.6*t)];
[tt, X] = ode45(@(t, x) fw_f(x, uf2(t), P2, zeros(6,1)), 0:0.25:6, x0, oo);
out.sim2.t = tt; out.sim2.x = X;
"""

CASES = [(25, 0, math.inf, "motor"), (25, 5, math.inf, "motor"), (30, 0, 150.0, "motor"), (22, 3, -200.0, "motor"),
         (25, 0, math.inf, "simple"), (30, 0, 150.0, "simple")]


@pytest.fixture(scope="module")
def ml():
    from tools.matlab_oracle import OracleUnavailable, fixture
    try:
        return fixture("uav_fixedwing", FW_SCRIPT)
    except OracleUnavailable as exc:  # pragma: no cover
        pytest.skip(str(exc))


def _model(kind: str, **kw) -> FixedWing:
    if kind == "motor":
        return FixedWing(FixedWingParams.aerosonde(), **kw)
    return FixedWing(FixedWingParams.aerosonde(prop_model="simple", drag_model="linear"), **kw)


def _rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b)) / max(1.0, float(np.max(np.abs(b)))))


# ============================================================================ MATLAB referansları
@pytest.mark.parametrize("k", range(len(CASES)))
def test_trim_matches_matlab_fsolve(ml, k):
    Va, gdeg, R, kind = CASES[k]
    ref = ml["trim"][k]
    assert np.max(np.abs(ref["F"])) < 1e-9                      # MATLAB çözümü gerçekten trim
    tr = _model(kind).trim(Va, math.radians(gdeg), R)
    z = np.array(ref["z"])
    mine = np.concatenate(([tr.alpha, tr.phi, tr.theta], tr.x[9:12], tr.u))
    np.testing.assert_allclose(mine, z, atol=1e-12, rtol=0)
    assert tr.success and np.max(np.abs(tr.residual)) < 1e-11


def test_jacobian_matches_matlab_symbolic(ml):
    fw = _model("motor")
    X1 = np.array([10, -5, -120, 24, 1.5, 2.0, 0.1, 0.08, 0.5, 0.05, -0.03, 0.08])
    U1 = np.array([-0.1, 0.02, -0.01, 0.75])
    lin = fw.linearize(X1, U1)
    assert _rel(lin.xdot0, ml["j1"]["f"]) < 1e-13
    assert _rel(lin.A, ml["j1"]["A"]) < 1e-10
    assert _rel(lin.B, ml["j1"]["B"]) < 1e-10
    tr = fw.trim(30, 0, 150.0)
    lt = fw.linearize(tr.x, tr.u)
    assert _rel(lt.A, ml["jt"]["A"]) < 1e-10
    assert _rel(lt.B, ml["jt"]["B"]) < 1e-10


@pytest.mark.parametrize("case", ["sim1", "sim2"])
def test_simulation_matches_matlab_ode45(ml, case):
    x0 = np.array([0, 0, -100, 24.9, 0.3, 1.3, 0.05, 0.06, 0.2, 0.02, -0.01, 0.03])
    if case == "sim1":
        fw = _model("motor", wind=lambda t: [3, -2, 0.5, 0.8 * math.sin(0.9 * t), 0.5 * math.cos(1.3 * t),
                                             0.3 * math.sin(2.2 * t)])
        uf = lambda t, x: [-0.124 + 0.05 * math.sin(1.5 * t) * math.exp(-0.3 * t), 0.02 * math.sin(2.1 * t),  # noqa
                           -0.01 * math.cos(0.8 * t), 0.77 + 0.1 * math.sin(0.5 * t)]
    else:
        fw = _model("simple")
        uf = lambda t, x: [-0.124 + 0.03 * math.sin(2 * t), 0.01 * math.sin(1.1 * t),  # noqa: E731
                           0.005 * math.sin(0.7 * t), 0.34 + 0.05 * math.cos(0.6 * t)]
    ref = ml[case]
    t = np.array(ref["t"])
    res = fw.simulate(x0, uf, t, method="DOP853", rtol=1e-12, atol=1e-12)
    X = np.array(ref["x"])
    scale = np.maximum(1.0, np.max(np.abs(X), axis=0))
    err = float(np.max(np.abs(res.x - X) / scale))
    assert err < 1e-10, err
    assert np.ptp(res["phi"]) > 0.05 and np.ptp(res["pn"]) > 100


# ============================================================================ trim özellikleri
@pytest.mark.parametrize("kind", ["motor", "simple"])
def test_trimmed_flight_stays_trimmed(kind):
    fw = _model(kind)
    for Va, gam, R in ((25, 0.0, math.inf), (25, math.radians(5), math.inf), (22, math.radians(3), -200.0)):
        tr = fw.trim(Va, gam, R)
        assert tr.success
        t = np.linspace(0, 4, 9)
        res = fw.simulate(tr.x, tr.u, t, method="DOP853", rtol=1e-11, atol=1e-11)
        for name in ("u", "v", "w", "phi", "theta", "p", "q", "r"):
            assert np.ptp(res[name]) < 1e-7, name
        np.testing.assert_allclose(-res["pd"], 100 + Va * math.sin(gam) * t, atol=1e-7)
        np.testing.assert_allclose(res["psi"], tr.psidot * t, atol=1e-8)
        ad = fw.air_data(tr.x)
        assert ad["Va"] == pytest.approx(Va, rel=1e-14) and ad["beta"] == pytest.approx(0.0, abs=1e-14)
        assert ad["gamma"] == pytest.approx(gam, abs=1e-12)
        if not math.isinf(R):                                   # yatay iz R yarıçaplı çember
            chi0 = ad["chi"]
            c = tr.x[:2] + R * np.array([-math.sin(chi0), math.cos(chi0)])
            d = np.hypot(res["pn"] - c[0], res["pe"] - c[1])
            np.testing.assert_allclose(d, abs(R), rtol=1e-9)


def test_straight_level_trim_book_structure():
    """Düz-düzey uçuş, β = 0: φ = 0, θ = α (kitaptaki θ* = α* + γ*), p = q = r = 0; basit pervanede δa = δr = 0."""
    tr = _model("simple").trim(25.0)
    assert tr.phi == pytest.approx(0.0, abs=1e-14) and tr.theta == pytest.approx(tr.alpha, abs=1e-14)
    np.testing.assert_allclose(tr.x[9:12], 0, atol=1e-15)
    np.testing.assert_allclose(tr.u[1:3], 0, atol=1e-14)
    trm = _model("motor").trim(25.0, math.radians(5))
    assert trm.theta == pytest.approx(trm.alpha + math.radians(5), abs=1e-5)   # küçük φ (pervane torku) farkı
    assert 0 < trm.u[3] < 1 and -0.3 < trm.u[0] < 0                           # makul gaz ve irtifa dümeni


def test_trim_input_validation():
    fw = _model("motor")
    with pytest.raises(ControlError):
        fw.trim(-5)
    with pytest.raises(ControlError):
        fw.trim(25, 0, 0.0)
    with pytest.raises(ControlError):
        fw.derivatives(np.zeros(5), np.zeros(4))
    with pytest.raises(ControlError):
        FixedWing(FixedWingParams.aerosonde(prop_model="jet"))


# ============================================================================ doğrusal modeller
@pytest.mark.parametrize("kind", ["motor", "simple"])
def test_book_coefficients_equal_jacobian_entries(kind):
    fw = _model(kind)
    tr = fw.trim(25.0)
    lin = fw.linearize(tr.x, tr.u)
    lo, la = longitudinal(lin), lateral(lin)
    c = transfer_function_coefficients(fw, tr)
    i = la.state_names.index
    assert c["a_phi1"] == pytest.approx(-la.A[i("p"), i("p")], rel=1e-9)
    assert c["a_phi2"] == pytest.approx(la.B[i("p"), 0], rel=1e-9)
    assert c["a_beta1"] == pytest.approx(-la.A[i("v"), i("v")], rel=1e-9)
    assert c["a_beta2"] == pytest.approx(la.B[i("v"), 1] / tr.Va, rel=1e-9)
    j = lo.state_names.index
    assert c["a_theta1"] == pytest.approx(-lo.A[j("q"), j("q")], rel=1e-9)
    assert c["a_theta3"] == pytest.approx(lo.B[j("q"), 0], rel=1e-9)
    assert c["a_theta2"] == pytest.approx(-lo.A[j("q"), j("w")] * tr.Va / math.cos(tr.alpha), rel=1e-9)
    assert c["a_V2"] == pytest.approx(lo.B[j("u"), 1], rel=1e-9)
    # h = −pd: ḣ = u·sinθ − w·cosθ (φ = 0 iken) → ∂ḣ/∂θ = Va (= h/θ = Va/s)
    assert lo.A[j("h"), j("theta")] == pytest.approx(tr.Va * math.cos(tr.theta - tr.alpha), rel=1e-9)
    tfs = book_transfer_functions(c)
    assert set(tfs) >= {"phi_delta_a", "theta_delta_e", "Va_delta_t", "beta_delta_r", "chi_phi", "h_theta"}


def test_a_V1_reduces_to_book_formula():
    """Doğrusal sürükleme + basit pervanede kitabın a_V1 = ρVaS/m·[C_D0 + C_Dα·α* + C_Dδe·δe*] + ρ·S_prop·C_prop·Va/m."""
    fw = _model("simple")
    p = fw.params
    tr = fw.trim(25.0)
    c = transfer_function_coefficients(fw, tr)
    book = p.rho * tr.Va * p.S_wing / p.mass * (p.C_D_0 + p.C_D_alpha * tr.alpha + p.C_D_delta_e * tr.u[0]) \
        + p.rho * p.S_prop * p.C_prop * tr.Va / p.mass
    assert c["a_V1"] == pytest.approx(book, rel=1e-14)
    assert c["a_V2"] == pytest.approx(p.rho * p.S_prop * p.C_prop * p.k_motor ** 2 * tr.u[3] / p.mass, rel=1e-14)
    assert c["a_V3"] == pytest.approx(p.gravity * math.cos(tr.theta - tr.alpha), rel=1e-15)


def test_decoupling_exact_for_symmetric_trim():
    """Kanat düz, β = 0, pervane torku yok (basit model): boylamsal ↔ yanal bağlaşım tam sıfır."""
    fw = _model("simple")
    tr = fw.trim(25.0)
    lin = fw.linearize(tr.x, tr.u)
    assert longitudinal(lin).info["coupling_max"] < 1e-9
    assert lateral(lin).info["coupling_max"] < 1e-9
    lin_m = _model("motor").linearize(*_trim_xu(_model("motor"), 25.0))
    assert lateral(lin_m).info["coupling_max"] > 1e-4            # pervane torku küçük bağlaşım yaratır (raporlanır)


def _trim_xu(fw, Va):
    tr = fw.trim(Va)
    return tr.x, tr.u


@pytest.mark.parametrize("kind", ["motor", "simple"])
def test_mode_analysis_physically_plausible(kind):
    fw = _model(kind)
    tr = fw.trim(25.0)
    lin = fw.linearize(tr.x, tr.u)
    lo, la = longitudinal(lin), lateral(lin)
    md = mode_analysis(lo, la)
    c = transfer_function_coefficients(fw, tr)
    sp, ph = md["short_period"], md["phugoid"]
    assert sp["wn"] > 5 * ph["wn"]
    assert 0.2 < sp["zeta"] < 1.0 and 0.0 < ph["zeta"] < 1.0
    if kind == "motor":
        # Lanchester yaklaşımı (itki hızdan bağımsız varsayımıyla) fugoid ωn ≈ √2·g/Va, hafif sönümlü.
        # Basit (1. baskı) pervane modeli ∂T/∂Va = −ρ·S_prop·C_prop·Va ile büyük hız sönümü verir; fugoid ζ ≈ 0.66
        # çıkar — bu modelin bilinen sonucudur, bu yüzden Lanchester denetimi yalnız motor modelinde yapılır.
        lanchester = math.sqrt(2) * fw.params.gravity / tr.Va
        assert ph["wn"] == pytest.approx(lanchester, rel=0.15) and ph["zeta"] < 0.5
    roll = md["roll"]["eigenvalues"][0].real
    assert roll == pytest.approx(-c["a_phi1"], rel=0.05)                  # yuvarlanma kipi ≈ −a_φ1
    dr = md["dutch_roll"]
    assert 1.0 < dr["wn"] < 10.0 and 0.0 < dr["zeta"] < 0.6
    assert abs(md["spiral"]["eigenvalues"][0]) < 0.2
    # kitabın kısa periyot yaklaşımı (a_θ1, a_θ2) aynı mertebede
    assert math.sqrt(c["a_theta2"]) == pytest.approx(sp["wn"], rel=0.25)
    # özdeğerlerin tamamı (sıfır hariç) mod listesinde
    ev = np.sort_complex(np.concatenate([md[k]["eigenvalues"] for k in ("short_period", "phugoid")]))
    full = np.linalg.eigvals(lo.A)
    full = np.sort_complex(np.delete(full, np.argmin(np.abs(full))))
    np.testing.assert_allclose(ev, full, atol=1e-12)


def test_linearize_outputs_and_reduced_models():
    fw = _model("motor")
    tr = fw.trim(25.0)
    lin = fw.linearize(tr.x, tr.u, outputs=("Va", "alpha", "beta", "h", "chi", "phi"))
    assert lin.C.shape == (6, 12) and lin.D.shape == (6, 4)
    np.testing.assert_allclose(lin.y0[:4], [25.0, tr.alpha, 0.0, 100.0], atol=1e-12)
    # ∂Va/∂u = cosα, ∂α/∂w = cosα/Va, ∂β/∂v = 1/Va (rüzgarsız, β = 0)
    k = list(lin.state_names).index
    assert lin.C[0, k("u")] == pytest.approx(math.cos(tr.alpha), rel=1e-9)
    assert lin.C[1, k("w")] == pytest.approx(math.cos(tr.alpha) / 25.0, rel=1e-9)
    assert lin.C[2, k("v")] == pytest.approx(1 / 25.0, rel=1e-9)
    assert lin.C[3, k("pd")] == pytest.approx(-1.0, rel=1e-12)
    lo = longitudinal(fw.linearize(tr.x, tr.u))
    assert lo.state_names == ("u", "w", "q", "theta", "h") and lo.input_names == ("delta_e", "delta_t")
    assert lo.x0[-1] == pytest.approx(100.0)
    with pytest.raises(ControlError):
        fw.linearize(tr.x, tr.u, outputs=("bilinmeyen",))


# ============================================================================ aerodinamik / pervane / rüzgar
def test_sigma_stable_form_equals_book_formula():
    fw = _model("motor")
    p = fw.params
    for a in np.linspace(-1.0, 1.0, 41):
        e1, e2 = math.exp(-p.M * (a - p.alpha0)), math.exp(p.M * (a + p.alpha0))
        book = (1 + e1 + e2) / ((1 + e1) * (1 + e2))
        assert fw.sigma(a) == pytest.approx(book, rel=1e-12, abs=1e-15)
    assert math.isfinite(fw.sigma(30.0)) and math.isfinite(fw.sigma(-30.0))
    assert fw.CL(0.05) == pytest.approx(p.C_L_0 + p.C_L_alpha * 0.05, rel=1e-9)      # stall öncesi doğrusal
    assert fw.CL(1.2) == pytest.approx(2 * math.sin(1.2) ** 2 * math.cos(1.2), rel=1e-6)   # düz levha


@pytest.mark.parametrize("kind", ["motor", "simple"])
def test_propeller_partials(kind):
    fw = _model(kind)
    for Va, dt in ((25.0, 0.77), (18.0, 0.5), (32.0, 0.95)):
        a = fw.propeller_partials(Va, dt)
        n = numjac(lambda z: np.array([fw.propeller(z[0], z[1])[0]]), np.array([Va, dt]), abs_step=1e-4)[0]
        np.testing.assert_allclose(a, n, rtol=1e-8)
    if kind == "motor":                                         # Ω denklemi sağlanıyor ve J ile itki tutarlı
        p = fw.params
        T, Q, om = fw.propeller(25.0, 0.77)
        Qm = p.KQ * ((p.V_max * 0.77 - p.KV * om) / p.R_motor - p.i0)
        assert Q == pytest.approx(Qm, rel=1e-10)
        J = 2 * math.pi * 25.0 / (om * p.D_prop)
        nrev = om / (2 * math.pi)
        assert T == pytest.approx(p.rho * nrev ** 2 * p.D_prop ** 4 * (p.C_T2 * J * J + p.C_T1 * J + p.C_T0),
                                  rel=1e-12)


def test_air_data_with_wind():
    x = np.array([0, 0, -50, 25.0, 0, 0, 0, 0, 0, 0, 0, 0])
    fw = _model("motor", wind=(5.0, 0.0, 0.0))
    ad = fw.air_data(x)
    assert ad["Va"] == pytest.approx(20.0) and ad["Vg"] == pytest.approx(25.0)
    fw.wind = wind_function((5.0, 0.0, 0.0), gust=lambda t: (0.0, 0.0, 1.0))
    ad = fw.air_data(x, 0.0)
    assert ad["alpha"] == pytest.approx(math.atan2(-1.0, 20.0), rel=1e-14)
    fw.wind = (0.0, 3.0, 0.0)                                  # doğudan esen yan rüzgar (doğuya doğru akış)
    ad = fw.air_data(x)
    assert ad["beta"] == pytest.approx(math.asin(-3.0 / math.hypot(25.0, 3.0)), rel=1e-14)
    fw.wind = (1.0, 2.0)
    with pytest.raises(ControlError):
        fw.air_data(x)


def test_dryden_variance_and_reproducibility():
    for name in DRYDEN_PRESETS:
        g = DrydenGust(25.0, preset=name, seed=3)
        np.testing.assert_allclose(g.stationary_variance(), np.square(g.sigma), rtol=1e-12)
    g = DrydenGust(25.0, preset="low_moderate", seed=7)
    t, s = g.sample(2000.0, 0.05)
    var = s.var(axis=0)
    np.testing.assert_allclose(var, np.square(g.sigma), rtol=0.12)
    t2, s2 = DrydenGust(25.0, preset="low_moderate", seed=7).sample(10.0, 0.05)
    np.testing.assert_array_equal(s2, s[:t2.size])
    f = g.as_function(5.0, 0.01)
    assert f(0.0).shape == (3,)
    with pytest.raises(ControlError):
        DrydenGust(25.0, preset="kasırga")
    with pytest.raises(ControlError):
        DrydenGust(0.0)


def test_simulation_with_dryden_gust_runs():
    fw = _model("motor")
    tr = fw.trim(25.0)
    gust = DrydenGust(25.0, preset="low_light", seed=1).as_function(5.0, 0.01)
    fw.wind = wind_function((2.0, -1.0, 0.0), gust)
    res = fw.simulate(tr.x, tr.u, np.linspace(0, 5, 51), method="RK45", rtol=1e-7, atol=1e-9, max_step=0.01)
    assert np.all(np.isfinite(res.x)) and np.ptp(res["phi"]) > 1e-4


def test_output_vector_and_specific_force():
    fw = _model("motor")
    tr = fw.trim(25.0)
    y = fw.output(tr.x)
    assert y.shape == (len(fw.output_names),) == (18,)
    np.testing.assert_allclose(y[:12], tr.x)
    np.testing.assert_allclose(y[12:15], [25.0, tr.alpha, 0.0], atol=1e-14)
    # dengede net kuvvet sıfır → ivmeölçer yalnız yerçekiminin tepkisini ölçer: −Rᵀ·g·e3
    g = fw.params.gravity
    f = fw.specific_force(tr.x, tr.u)
    th, ph = tr.theta, tr.phi
    np.testing.assert_allclose(f, [g * math.sin(th), -g * math.cos(th) * math.sin(ph), -g * math.cos(th) * math.cos(ph)],
                               atol=1e-10)


def test_steady_sideslip_trim():
    """β* ≠ 0 düz uçuş (çapraz kumanda): yan kuvvet yatışla dengelenir, tüm türevler sıfır, uçuş korunur."""
    fw = _model("motor")
    for beta in (0.05, -0.08):
        tr = fw.trim(25.0, beta=beta)
        assert tr.success and np.max(np.abs(tr.residual)) < 1e-11
        assert fw.air_data(tr.x)["beta"] == pytest.approx(beta, abs=1e-14)
        assert np.sign(tr.phi) == np.sign(beta)
        res = fw.simulate(tr.x, tr.u, np.linspace(0, 3, 7), method="DOP853", rtol=1e-11, atol=1e-11)
        for name in ("u", "v", "w", "phi", "theta", "psi", "p", "q", "r"):
            assert np.ptp(res[name]) < 1e-7, name


def test_fast_paths_equal_reference_implementation():
    rng = np.random.default_rng(1)
    for kind in ("motor", "simple"):
        for wind in (None, (3.0, -2.0, 0.5), lambda t: [3.0, -2.0, 0.5, 0.8 * math.sin(t), 0.3, 0.1]):
            fw = _model(kind, wind=wind)
            for _ in range(10):
                x = np.array([10, -5, -120, 24 + rng.normal(), rng.normal(), 2 + rng.normal(), 0.3 * rng.normal(),
                              0.2 * rng.normal(), rng.normal(), 0.2 * rng.normal(), 0.2 * rng.normal(),
                              0.2 * rng.normal()])
                u = np.array([-0.1, 0.02, -0.01, 0.75]) + 0.05 * rng.normal(size=4)
                assert _rel(fw.derivatives(x, u, 0.7), fw.derivatives_reference(x, u, 0.7)) < 1e-14
                np.testing.assert_allclose(fw.output(x, None, 0.7), fw.output_reference(x, None, 0.7), atol=1e-13)
