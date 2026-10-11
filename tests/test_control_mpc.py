"""MPC (Model Öngörülü Kontrol) — MATLAB MPC Toolbox (R2025b) referanslı testler.

Referans: `tests/data/matlab/mpc_cases.json`. Her durumda MATLAB'in iç QP / kestirici verisi (getCodeGenerationData),
getEstimator, ss(mpcobj), cloffset, sim ve mpcmove çıktıları karşılaştırılır.
"""
from __future__ import annotations

import numpy as np
import pytest

from spt.control.errors import ControlError
from spt.control.mpc import (MPC, cloffset, getEstimator, mpc, mpcmove, mpcsimopt, mpcstate, setmpcsignals,
                             setoutdist, sim)
from tools.matlab_oracle import OracleUnavailable, fixture

MPC_SCRIPT = r"""
mpcverbosity off;
A1 = [0 1; -2 -3]; B1 = [0; 1]; C1 = [1 0];
plant1 = ss(A1, B1, C1, 0);
obj1 = mpc(plant1, 0.1);
[y, ~, u] = sim(obj1, 30, 1);
out.c1.y = y; out.c1.u = u;
[L, M] = getEstimator(obj1); out.c1.L = L; out.c1.M = M;
K = ss(obj1, 'r'); out.c1.ssA = K.A; out.c1.ssB = K.B; out.c1.ssC = K.C; out.c1.ssD = K.D;
out.c1.cloffset = cloffset(obj1);
plant2 = ss([0.9 0.2; 0 0.7], [0; 0.5], [1 0.3], 0, 0.2);
W2 = struct('ManipulatedVariables', 0, 'ManipulatedVariablesRate', 0.3, 'OutputVariables', 2);
MV2 = struct('Min', -0.4, 'Max', 0.6, 'RateMin', -0.15, 'RateMax', 0.2);
obj2 = mpc(plant2, 0.2, 15, 3, W2, MV2);
r2 = [ones(20,1); -0.5*ones(20,1)];
[y, ~, u] = sim(obj2, 40, r2);
out.c2.y = y; out.c2.u = u;
st = mpcstate(obj2);
ym2 = [0 0.1 0.3 0.5 0.7 0.8 0.85 0.9];
for k = 1:numel(ym2)
    [uk, info] = mpcmove(obj2, st, ym2(k), 1);
    out.c2.mv_u(k) = uk; out.c2.mv_cost(k) = info.Cost; out.c2.mv_iter(k) = info.Iterations;
    out.c2.mv_slack(k) = info.Slack; out.c2.mv_code{k} = info.QPCode;
    out.c2.mv_Uopt{k} = info.Uopt; out.c2.mv_Yopt{k} = info.Yopt; out.c2.mv_Xopt{k} = info.Xopt;
    out.c2.mv_Topt{k} = info.Topt; out.c2.mv_xp{k} = st.Plant; out.c2.mv_xd{k} = st.Disturbance;
    out.c2.mv_last(k) = st.LastMove;
end
st2 = mpcstate(obj2); st2.LastMove = 2;
[ui, infoi] = mpcmove(obj2, st2, 0, 1);
out.c2.inf_u = ui; out.c2.inf_code = infoi.QPCode; out.c2.inf_iter = infoi.Iterations; out.c2.inf_cost = infoi.Cost;
A3 = [0.8 0.1 0 0; 0 0.7 0 0; 0 0 0.85 0.05; 0 0 0 0.6];
B3 = [0.5 0; 0.3 0.1; 0 0.4; 0.2 0.3];
C3 = [1 0 0.5 0; 0 0.3 1 0];
plant3 = ss(A3, B3, C3, zeros(2,2), 0.5);
obj3 = mpc(plant3, 0.5, 12, 3);
obj3.Weights.OutputVariables = [1 0.5];
obj3.Weights.ManipulatedVariables = [0.1 0];
obj3.Weights.ManipulatedVariablesRate = [0.2 0.1];
obj3.MV(1).Target = 0.3;
obj3.MV(1).Min = -1; obj3.MV(1).Max = 1; obj3.MV(2).Min = -2; obj3.MV(2).Max = 2;
obj3.OV(1).Max = 1.05; obj3.OV(2).Min = -0.2; obj3.OV(2).MinECR = 0.5;
obj3.OV(1).ScaleFactor = 2; obj3.MV(2).ScaleFactor = 4;
r3 = [ones(15,1)*[1 0.5]; ones(15,1)*[1.2 -0.5]];
[y, ~, u] = sim(obj3, 30, r3);
out.c3.y = y; out.c3.u = u;
opt3 = mpcsimopt; opt3.RefLookAhead = 'on';
[y, ~, u] = sim(obj3, 30, r3, opt3);
out.c3.yp = y; out.c3.up = u;
st3 = mpcstate(obj3);
rp = [0.5 0.1; 0.7 0.2; 0.9 0.3; 1 0.4; 1.1 0.5];
[u, info] = mpcmove(obj3, st3, [0.1; 0.2], rp);
out.c3.pv_u = u; out.c3.pv_Uopt = info.Uopt; out.c3.pv_Yopt = info.Yopt; out.c3.pv_cost = info.Cost;
out.c3.pv_slack = info.Slack;
[L, M] = getEstimator(obj3); out.c3.L = L; out.c3.M = M;
K = ss(obj3); out.c3.ssA = K.A; out.c3.ssB = K.B; out.c3.ssC = K.C; out.c3.ssD = K.D;
A4 = [0.9 0; 0.1 0.8]; B4 = [0.4 0.2; 0 0.3]; C4 = [0 1]; D4 = [0 0.1];
plant4 = setmpcsignals(ss(A4, B4, C4, D4, 0.1), 'MV', 1, 'MD', 2);
nom4 = struct('U', [1; 0.5], 'Y', 2, 'X', [0.5; 1], 'DX', [0.01; 0]);
obj4 = mpc(struct('Plant', plant4, 'Nominal', nom4), 0.1, 10, 2);
obj4.MV.Min = 0; obj4.MV.Max = 2.5;
v4 = [0.5*ones(10,1); 1.5*ones(20,1)];
[y, ~, u] = sim(obj4, 30, 2.5*ones(30,1), v4);
out.c4.y = y; out.c4.u = u;
[L, M] = getEstimator(obj4); out.c4.L = L; out.c4.M = M;
K = ss(obj4, 'rvo'); out.c4.ssA = K.A; out.c4.ssB = K.B; out.c4.ssC = K.C; out.c4.ssD = K.D;
out.c4.cloffset = cloffset(obj4);
st4 = mpcstate(obj4);
[u, info] = mpcmove(obj4, st4, 2.1, 2.5, 0.7);
out.c4.mv_u = u; out.c4.mv_Yopt = info.Yopt; out.c4.mv_Uopt = info.Uopt; out.c4.mv_cost = info.Cost;
out.c4.mv_xp = st4.Plant; out.c4.mv_xd = st4.Disturbance;
plant5 = setmpcsignals(ss([0 1; 0 0], [0; 1], eye(2), zeros(2,1)), 'MO', 1, 'UO', 2);
obj5 = mpc(plant5, 0.05, 20, [2 3 5]);
obj5.MV.Min = -3; obj5.MV.Max = [3; 2.8; 2.6]; obj5.MV.RateMin = -1; obj5.MV.RateMax = 1;
obj5.OV(2).Min = -1.5; obj5.OV(2).Max = 1.5;
[y, ~, u] = sim(obj5, 60, [2 0]);
out.c5.y = y; out.c5.u = u;
[L, M] = getEstimator(obj5); out.c5.L = L; out.c5.M = M;
out.c5.W = obj5.Weights.OutputVariables; out.c5.ECR = obj5.Weights.ECR;
plant6 = setmpcsignals(ss([0.85 0.1; 0 0.9], [0.3 0.2; 0.1 0.4], [1 0.2], [0 0], 1), 'MV', 1, 'UD', 2);
obj6 = mpc(plant6, 1, 8, 2);
obj6.MV.Min = -2; obj6.MV.Max = 2;
true6 = setmpcsignals(ss([0.88 0.12; 0 0.87], [0.33 0.25; 0.12 0.35], [1 0.25], [0 0], 1), 'MV', 1, 'UD', 2);
opt6 = mpcsimopt; opt6.Model = true6; opt6.UnmeasuredDisturbance = [zeros(15,1); 0.5*ones(25,1)];
[y, ~, u] = sim(obj6, 40, 1, opt6);
out.c6.y = y; out.c6.u = u;
[L, M] = getEstimator(obj6); out.c6.L = L; out.c6.M = M;
obj4b = obj4; obj4b.DV(1).ScaleFactor = 2; obj4b.OV(1).ScaleFactor = 3; obj4b.MV(1).ScaleFactor = 0.5;
opt4 = mpcsimopt; opt4.MDLookAhead = 'on';
[y, ~, u] = sim(obj4b, 30, 2.5*ones(30,1), v4, opt4);
out.c4b.y = y; out.c4b.u = u;
[L, M] = getEstimator(obj4b); out.c4b.L = L; out.c4b.M = M;
A7 = [0.9 0.1; 0 0.8]; B7 = [0.2 0.1; 0.3 0]; C7 = [1 0; 0 1];
plant7 = setmpcsignals(ss(A7, B7, C7, zeros(2,2), 0.1), 'MV', 1, 'UD', 2);
mdl7 = struct('Plant', plant7, 'Disturbance', ss(0.95, 0.1, 1, 0, 0.1), 'Noise', ss(0.5, 1, [1; 0.5], [0.2; 0.1], 0.1));
obj7 = mpc(mdl7, 0.1, 8, 3);
obj7.Weights.OutputVariables = [1 0.5; 2 0.5; 3 1];
obj7.MV.Min = -1; obj7.MV.Max = 1;
obj7.OV(1).Max = [1.2; 1.1; 1.0];
setoutdist(obj7, 'model', ss(0.8, 0.5, [1; 0], [0; 0], 0.1));
opt7 = mpcsimopt; opt7.UnmeasuredDisturbance = [zeros(10,1); 0.3*ones(20,1)];
opt7.OutputNoise = 0.01*sin((1:30)'*[0.5 0.9]);
[y, ~, u] = sim(obj7, 30, [1 0.5], opt7);
out.c7.y = y; out.c7.u = u;
[L, M] = getEstimator(obj7); out.c7.L = L; out.c7.M = M;
flds = {'H','Hinv','Linv','Kx','Kr','Ku1','Kv','Kut','Ac','Mlim','Mx','Mu1','Mv','A','B','C','D','Bu','Bv', ...
    'Cm','Dv','Dvm','L','M','Pcov','utarget','Mrows','Jm','uoff','yoff','xpoff','outdistchannels', ...
    'indistchannels','nxp','nx','nxQP','Uscale','Yscale','degrees'};
objs = {obj1, obj2, obj3, obj4, obj5, obj6, obj7, obj4b};
for i = 1:numel(objs)
    cdi = getCodeGenerationData(objs{i});
    s = struct();
    for k = 1:numel(flds)
        s.(flds{k}) = double(cdi.(flds{k}));
    end
    s.isQP = double(cdi.isQP);
    s.maxit = double(cdi.ASOptions.MaxIterations);
    s.ECR = objs{i}.Weights.ECR;
    out.(sprintf('cfg%d', i)) = s;
end
"""

# MATLAB ile yinelemeli QP'nin aynı optimumu: belge ~1e-6 ister; ölçülen fark ~1e-12 (bkz. rapor). Sıkı tutuyoruz.
TOL = 1e-9


@pytest.fixture(scope="module")
def ref():
    try:
        return fixture("mpc_cases", MPC_SCRIPT)
    except OracleUnavailable as e:  # pragma: no cover
        pytest.skip(str(e))


def _a(x, shape=None):
    a = np.asarray([] if x is None else x, dtype=float)
    return a.reshape(shape) if shape is not None else a


def _close(ours, theirs, tol=TOL, what=""):
    ours = np.asarray(ours, dtype=float)
    theirs = _a(theirs, ours.shape)
    err = np.abs(ours - theirs).max() if ours.size else 0.0
    scale = 1.0 + (np.abs(theirs).max() if theirs.size else 0.0)
    assert err <= tol * scale, f"{what}: fark {err:.3e}"


# --- MATLAB betiğindeki durumların Python karşılıkları --------------------------------------------------------
def case1():
    return MPC(([[0, 1], [-2, -3]], [[0], [1]], [[1, 0]], [[0]]), 0.1)


def case2():
    return MPC(([[0.9, 0.2], [0, 0.7]], [[0], [0.5]], [[1, 0.3]], [[0]], 0.2), 0.2, 15, 3,
               dict(ManipulatedVariables=0, ManipulatedVariablesRate=0.3, OutputVariables=2),
               dict(Min=-0.4, Max=0.6, RateMin=-0.15, RateMax=0.2))


def case3():
    A3 = [[0.8, 0.1, 0, 0], [0, 0.7, 0, 0], [0, 0, 0.85, 0.05], [0, 0, 0, 0.6]]
    B3 = [[0.5, 0], [0.3, 0.1], [0, 0.4], [0.2, 0.3]]
    C3 = [[1, 0, 0.5, 0], [0, 0.3, 1, 0]]
    o = MPC((A3, B3, C3, np.zeros((2, 2)), 0.5), 0.5, 12, 3)
    o.Weights.OutputVariables = [1, 0.5]
    o.Weights.ManipulatedVariables = [0.1, 0]
    o.Weights.ManipulatedVariablesRate = [0.2, 0.1]
    o.MV[0].Target = 0.3
    o.MV[0].Min, o.MV[0].Max, o.MV[1].Min, o.MV[1].Max = -1, 1, -2, 2
    o.OV[0].Max = 1.05
    o.OV[1].Min = -0.2
    o.OV[1].MinECR = 0.5
    o.OV[0].ScaleFactor = 2
    o.MV[1].ScaleFactor = 4
    return o


def case4():
    plant4 = setmpcsignals(([[0.9, 0], [0.1, 0.8]], [[0.4, 0.2], [0, 0.3]], [[0, 1]], [[0, 0.1]], 0.1), MV=0, MD=1)
    o = MPC({"Plant": plant4, "Nominal": {"U": [1, 0.5], "Y": 2, "X": [0.5, 1], "DX": [0.01, 0]}}, 0.1, 10, 2)
    o.MV[0].Min = 0
    o.MV[0].Max = 2.5
    return o


def case4b():
    o = case4()
    o.DV[0].ScaleFactor = 2
    o.OV[0].ScaleFactor = 3
    o.MV[0].ScaleFactor = 0.5
    return o


def case5():
    plant5 = setmpcsignals(([[0, 1], [0, 0]], [[0], [1]], np.eye(2), np.zeros((2, 1))), MO=0, UO=1)
    o = MPC(plant5, 0.05, 20, [2, 3, 5])
    o.MV[0].Min, o.MV[0].Max, o.MV[0].RateMin, o.MV[0].RateMax = -3, [3, 2.8, 2.6], -1, 1
    o.OV[1].Min, o.OV[1].Max = -1.5, 1.5
    return o


def case6():
    plant6 = setmpcsignals(([[0.85, 0.1], [0, 0.9]], [[0.3, 0.2], [0.1, 0.4]], [[1, 0.2]], [[0, 0]], 1), MV=0, UD=1)
    o = MPC(plant6, 1, 8, 2)
    o.MV[0].Min, o.MV[0].Max = -2, 2
    return o


TRUE6 = setmpcsignals(([[0.88, 0.12], [0, 0.87]], [[0.33, 0.25], [0.12, 0.35]], [[1, 0.25]], [[0, 0]], 1), MV=0, UD=1)


def case7():
    plant7 = setmpcsignals(([[0.9, 0.1], [0, 0.8]], [[0.2, 0.1], [0.3, 0]], np.eye(2), np.zeros((2, 2)), 0.1),
                           MV=0, UD=1)
    o = MPC({"Plant": plant7, "Disturbance": ([[0.95]], [[0.1]], [[1]], [[0]], 0.1),
             "Noise": ([[0.5]], [[1]], [[1], [0.5]], [[0.2], [0.1]], 0.1)}, 0.1, 8, 3)
    o.Weights.OutputVariables = [[1, 0.5], [2, 0.5], [3, 1]]
    o.MV[0].Min, o.MV[0].Max = -1, 1
    o.OV[0].Max = [1.2, 1.1, 1.0]
    setoutdist(o, "model", ([[0.8]], [[0.5]], [[1], [0]], [[0], [0]], 0.1))
    return o


CASES = {1: case1, 2: case2, 3: case3, 4: case4, 5: case5, 6: case6, 7: case7, 8: case4b}
CFG_FIELDS = ["H", "Hinv", "Linv", "Kx", "Kr", "Ku1", "Kv", "Kut", "Ac", "Mlim", "Mx", "Mu1", "Mv", "A", "B", "C", "D",
              "Bu", "Bv", "Cm", "Dv", "Dvm", "L", "M", "Pcov", "Jm", "uoff", "yoff", "xpoff", "utarget"]


@pytest.mark.parametrize("k", sorted(CASES))
def test_ic_qp_ve_kestirici_verisi_matlab(ref, k):
    """QP (H, Kx, Kr, Ku1, Kv, Kut, Ac, Mlim, Mx, Mu1, Mv) ve kestirici (artırılmış A, B, C, D, L, M, P) MATLAB'in
    getCodeGenerationData çıktısıyla — formülasyonun birebir aynı olduğunun kanıtı."""
    c = ref[f"cfg{k}"]
    d = CASES[k]().data
    assert d.isQP == bool(c["isQP"])
    assert d.max_iter == c["maxit"]
    assert d.Wecr == pytest.approx(c["ECR"], rel=1e-15)
    assert list(d.outdistchannels + 1) == list(np.atleast_1d(_a(c["outdistchannels"])).astype(int))
    assert list(d.indistchannels + 1) == list(np.atleast_1d(_a(c["indistchannels"])).astype(int))
    assert d.nx == c["nx"] and d.nxQP == c["nxQP"] and d.nxp == c["nxp"] and d.degrees == c["degrees"]
    for f in CFG_FIELDS:
        if not d.isQP and f in ("Ac", "Mlim", "Mx", "Mu1", "Mv"):
            continue
        _close(getattr(d, f), c[f], 1e-10, f"durum {k} {f}")
    _close(d.Uscale, c["Uscale"], 0, "Uscale")
    _close(d.Yscale, c["Yscale"], 0, "Yscale")
    if d.isQP:
        assert np.array_equal(d.Mrows + 1, _a(c["Mrows"]).ravel().astype(int))


def _sim_case(k, ref):
    o = CASES[k]()
    if k == 1:
        return sim(o, 30, 1), ref["c1"]
    if k == 2:
        return sim(o, 40, np.r_[np.ones(20), -0.5 * np.ones(20)]), ref["c2"]
    if k == 3:
        r3 = np.vstack([np.tile([1, 0.5], (15, 1)), np.tile([1.2, -0.5], (15, 1))])
        return sim(o, 30, r3), ref["c3"]
    if k == 4:
        return sim(o, 30, 2.5 * np.ones(30), np.r_[0.5 * np.ones(10), 1.5 * np.ones(20)]), ref["c4"]
    if k == 5:
        return sim(o, 60, [2, 0]), ref["c5"]
    if k == 6:
        opt = mpcsimopt(Model=TRUE6, UnmeasuredDisturbance=np.r_[np.zeros(15), 0.5 * np.ones(25)])
        return sim(o, 40, 1, opt), ref["c6"]
    if k == 7:
        t = np.arange(1, 31)[:, None]
        opt = mpcsimopt(UnmeasuredDisturbance=np.r_[np.zeros(10), 0.3 * np.ones(20)],
                        OutputNoise=0.01 * np.sin(t * np.array([[0.5, 0.9]])))
        return sim(o, 30, [1, 0.5], opt), ref["c7"]
    opt = mpcsimopt(MDLookAhead="on")
    return sim(o, 30, 2.5 * np.ones(30), np.r_[0.5 * np.ones(10), 1.5 * np.ones(20)], opt), ref["c4b"]


@pytest.mark.parametrize("k", sorted(CASES))
def test_sim_matlab(ref, k):
    """Kapalı çevrim benzetimi (u, y dizileri) MATLAB sim ile: SISO kısıtsız / kısıtlı, MIMO yumuşak OV, MD + nominal,
    çift tümlev (drone ekseni, UO hız kısıtı, blok hamle), model uyumsuzluğu + UD, özel bozucu / gürültü modelleri,
    ölçek çarpanları + MD önizlemesi."""
    res, r = _sim_case(k, ref)
    _close(res.u, r["u"], TOL, f"durum {k} u")
    _close(res.y, r["y"], TOL, f"durum {k} y")


@pytest.mark.parametrize("k,key", [(1, "c1"), (3, "c3"), (4, "c4"), (5, "c5"), (6, "c6"), (7, "c7"), (8, "c4b")])
def test_getEstimator_matlab(ref, k, key):
    L, M, *_ = getEstimator(CASES[k]())
    _close(L, ref[key]["L"], 1e-10, f"durum {k} L")
    _close(M, ref[key]["M"], 1e-10, f"durum {k} M")


def test_mpcmove_ayrintilari_matlab(ref):
    """mpcmove: u, Cost, Iterations (sıcak başlangıçlı), Uopt / Yopt / Xopt / Topt, durum güncellemesi."""
    r = ref["c2"]
    o = case2()
    st = mpcstate(o)
    for k, ym in enumerate([0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9]):
        u, info = mpcmove(o, st, ym, 1)
        assert u[0] == pytest.approx(r["mv_u"][k], abs=TOL)
        assert info["Cost"] == pytest.approx(r["mv_cost"][k], rel=1e-9)
        assert info["QPCode"] == r["mv_code"][k] == "feasible"
        assert info["Iterations"] == r["mv_iter"][k]
        assert info["Slack"] == pytest.approx(r["mv_slack"][k], abs=1e-12)
        _close(info["Uopt"], r["mv_Uopt"][k], TOL, "Uopt")
        _close(info["Yopt"], r["mv_Yopt"][k], TOL, "Yopt")
        _close(info["Xopt"], r["mv_Xopt"][k], TOL, "Xopt")
        _close(info["Topt"], r["mv_Topt"][k], 1e-14, "Topt")
        _close(st.Plant, r["mv_xp"][k], TOL, "x(k+1|k) bitki")
        _close(st.Disturbance, r["mv_xd"][k], TOL, "x(k+1|k) bozucu")
        assert st.LastMove[0] == pytest.approx(r["mv_last"][k], abs=TOL)


def test_mpcmove_cozumsuz_qp_mv_dondurulur(ref):
    """Çözümsüz QP (son MV sınır dışı, oran kısıtı yetişemez): MATLAB gibi MV önceki değerde tutulur."""
    r = ref["c2"]
    o = case2()
    st = mpcstate(o)
    st.LastMove = np.array([2.0])
    u, info = mpcmove(o, st, 0, 1)
    assert u[0] == pytest.approx(r["inf_u"]) == 2.0
    assert info["QPCode"] == r["inf_code"] == "infeasible"
    assert info["Iterations"] < 0 and r["inf_iter"] < 0   # MATLAB KWIK burada −2, biz −1 (ikisi de "olursuz")
    assert info["Cost"] == r["inf_cost"] == 0


def test_onizleme_matlab(ref):
    r = ref["c3"]
    o = case3()
    st = mpcstate(o)
    rp = np.array([[0.5, 0.1], [0.7, 0.2], [0.9, 0.3], [1, 0.4], [1.1, 0.5]])
    u, info = mpcmove(o, st, [0.1, 0.2], rp)
    _close(u, r["pv_u"], TOL, "u")
    _close(info["Uopt"], r["pv_Uopt"], TOL, "Uopt")
    _close(info["Yopt"], r["pv_Yopt"], TOL, "Yopt")
    assert info["Cost"] == pytest.approx(r["pv_cost"], rel=1e-9)
    assert info["Slack"] == pytest.approx(r["pv_slack"], rel=1e-6)   # yumuşak kısıt tam sınırda (ε ≈ 3e-6)
    res = sim(o, 30, np.vstack([np.tile([1, 0.5], (15, 1)), np.tile([1.2, -0.5], (15, 1))]),
              mpcsimopt(RefLookAhead="on"))
    _close(res.u, r["up"], TOL, "önizlemeli sim u")
    _close(res.y, r["yp"], TOL, "önizlemeli sim y")


def test_md_ve_nominal_mpcmove_matlab(ref):
    r = ref["c4"]
    o = case4()
    st = mpcstate(o)
    u, info = mpcmove(o, st, 2.1, 2.5, 0.7)
    _close(u, r["mv_u"], TOL, "u")
    _close(info["Yopt"], r["mv_Yopt"], TOL, "Yopt")
    _close(info["Uopt"], r["mv_Uopt"], TOL, "Uopt")
    assert info["Cost"] == pytest.approx(r["mv_cost"], rel=1e-9)
    _close(st.Plant, r["mv_xp"], TOL, "xp")
    _close(st.Disturbance, r["mv_xd"], TOL, "xd")


@pytest.mark.parametrize("k,key,signals", [(1, "c1", "r"), (3, "c3", ""), (4, "c4", "rvo")])
def test_esdeger_dogrusal_kontrolcu_matlab(ref, k, key, signals):
    K = CASES[k]().ss(signals)
    for a in "ABCD":
        _close(getattr(K, a), ref[key]["ss" + a], 1e-10, f"ss {a}")


@pytest.mark.parametrize("k,key", [(1, "c1"), (4, "c4")])
def test_cloffset_matlab(ref, k, key):
    off = cloffset(CASES[k]())
    _close(off, ref[key]["cloffset"], 1e-10, "cloffset")
    assert np.abs(off).max() < 1e-12     # varsayılan tümlevli bozucu modeli → ofsetsiz izleme


def test_varsayilan_kurallar(ref):
    """nmv < ny iken çıkış ağırlıkları (ölçülenler önce) ve ECR = 1e5·max(ağırlıklar); tümlevli bitkide çıkış
    tümlevi gözlenebilirliği bozduğu için eklenmez (PBH testi)."""
    o = case5()
    _close(o.Weights.OutputVariables, ref["c5"]["W"], 0)
    assert o.Weights.ECR == ref["c5"]["ECR"] == 1e5
    assert o.data.outdistchannels.size == 0
    assert case2().Weights.ECR == 2e5
    o1 = case1()
    assert o1.PredictionHorizon == 10 and o1.ControlHorizon == 2
    assert list(case5().data.moves) == [2, 3, 5, 10]


# --- MATLAB'siz iç tutarlılık testleri ---------------------------------------------------------------------------
def test_kisitsiz_mpcmove_esdeger_kontrolcuyle_ayni():
    """Kısıtsız MPC'de mpcmove dizisi = ss(mpcobj, 'r') doğrusal kontrolcüsünün çıktısı (bağımsız yol)."""
    o = case1()
    K = o.ss("r")
    st = mpcstate(o)
    xk = np.zeros(K.A.shape[0])
    rng = np.random.default_rng(0)
    for _ in range(15):
        ym = rng.standard_normal(1)
        r = rng.standard_normal(1)
        u, _ = mpcmove(o, st, ym, r)
        inp = np.concatenate([ym, r])
        uk = K.C @ xk + K.D @ inp
        xk = K.A @ xk + K.B @ inp
        assert np.allclose(u, uk, atol=1e-11)


def test_kisitlar_saglaniyor():
    """Sert MV / oran kısıtları benzetimde ihlal edilmez."""
    res = sim(case5(), 60, [2, 0])
    u = res.u[:, 0]
    assert u.max() <= 3 + 1e-9 and u.min() >= -3 - 1e-9
    du = np.diff(np.r_[0.0, u])
    assert du.max() <= 1 + 1e-9 and du.min() >= -1 - 1e-9
    assert u[0] <= 2.8 + 1e-9                     # ilk blokta zamanla değişen Max (min(3, 2.8))
    res2 = sim(case2(), 40, np.r_[np.ones(20), -0.5 * np.ones(20)])
    u2 = res2.u[:, 0]
    assert u2.max() <= 0.6 + 1e-9 and u2.min() >= -0.4 - 1e-9
    assert np.diff(np.r_[0.0, u2]).max() <= 0.2 + 1e-9


def test_struct_erisimi_ve_hatalar():
    o = case2()
    assert o.MV[0].max == 0.6 and o.MV[0]["MAX"] == 0.6     # büyük / küçük harf duyarsız (MATLAB gibi)
    o.Weights.MVRate = 0.5                                   # takma ad
    assert o.Weights.ManipulatedVariablesRate == 0.5
    with pytest.raises(ControlError):
        o.MV[0].Maks = 1
    with pytest.raises(ControlError):                        # MV'den doğrudan geçiş
        MPC(([[0.5]], [[1]], [[1]], [[0.2]], 0.1))
    with pytest.raises(ControlError):                        # sürekli bitki, Ts yok
        MPC(([[-1]], [[1]], [[1]], [[0]]))
    o2 = case2()
    o2.MV[0].RateMin = 0.1                                   # RateMin > 0
    with pytest.raises(ControlError):
        o2.data
    with pytest.raises(ControlError):                        # ölçülen çıkış yok
        MPC(setmpcsignals(([[0.5]], [[1]], [[1]], [[0]], 0.1), UO=0))


def test_degisiklik_yeniden_derlenir():
    o = case2()
    h1 = o.data.H.copy()
    o.Weights.OutputVariables = 5
    assert not np.allclose(o.data.H, h1)
    o.PredictionHorizon = 5
    assert o.data.p == 5 and o.data.Sx.shape[0] == 5
    assert mpc(o).data.p == 5                                # kopya kurucu


def test_yineleme_siniri_mv_dondurulur_ve_altoptimal():
    """MaxIterations aşılırsa (QP durum 0): MV tutulur, QPCode 'unreliable' (MATLAB). UseSuboptimalSolution açıksa
    alt-en iyi çözüm ilk adım sınırlarına kırpılarak uygulanır."""
    o = case2()
    o.Optimizer.MaxIterations = 1                  # bu adım MATLAB'de 5 yineleme ister
    st = mpcstate(o)
    u, info = mpcmove(o, st, 0, 1)
    assert info["Iterations"] == 0 and info["QPCode"] == "unreliable" and u[0] == 0.0 and info["Cost"] == 0
    o.Optimizer.UseSuboptimalSolution = True
    st = mpcstate(o)
    u, info = mpcmove(o, st, 0, 1)
    assert info["QPCode"] == "unreliable" and -0.15 - 1e-12 <= u[0] <= 0.2 + 1e-12 and u[0] != 0.0


def test_ordek_tipleme_bitki():
    """A, B, C, D, Ts öznitelikli herhangi bir nesne (LTI sınıflarına bağımlılık yok); adlar MV / OV'ye geçer."""
    from types import SimpleNamespace
    p = SimpleNamespace(A=[[0.9, 0.2], [0, 0.7]], B=[[0], [0.5]], C=[[1, 0.3]], D=[[0]], Ts=0.2,
                        input_names=("itki",), output_names=("irtifa",))
    o = MPC(p, None, 15, 3, dict(ManipulatedVariables=0, ManipulatedVariablesRate=0.3, OutputVariables=2),
            dict(Min=-0.4, Max=0.6, RateMin=-0.15, RateMax=0.2))
    assert o.Ts == 0.2 and o.MV[0].Name == "itki" and o.OV[0].Name == "irtifa"
    a = sim(o, 25, 1)
    b = sim(case2(), 25, 1)
    assert np.array_equal(a.u, b.u) and np.array_equal(a.y, b.y)


def test_kisitlar_kapali_benzetim_kisitsiz_ile_ayni():
    """Constraints='off' benzetimi, kısıtsız aynı nesneyle birebir aynı (MATLAB mpcsimopt.Constraints)."""
    a = sim(case2(), 20, 1, mpcsimopt(Constraints="off"))
    o2 = case2()
    o2.MV[0].Min, o2.MV[0].Max, o2.MV[0].RateMin, o2.MV[0].RateMax = -np.inf, np.inf, -np.inf, np.inf
    b = sim(o2, 20, 1)
    assert np.allclose(a.u, b.u, atol=1e-12) and np.allclose(a.y, b.y, atol=1e-12)
