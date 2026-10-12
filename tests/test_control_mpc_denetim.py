"""MPC denetim bulguları için MATLAB referanslı regresyon testleri (yineleme sınırı kuralı, girdi denetimi, MATLAB
alan adları, mpcActiveSetSolver bütünlük denetimleri).

Referans: `tests/data/matlab/mpc_denetim_cases.json` (MATLAB R2025b).
"""
from __future__ import annotations

DENETIM_SCRIPT = r"""
mpcverbosity off;
warning('off', 'all');
W2 = struct('ManipulatedVariables', 0, 'ManipulatedVariablesRate', 0.3, 'OutputVariables', 2);
MV2 = struct('Min', -0.4, 'Max', 0.6, 'RateMin', -0.15, 'RateMax', 0.2);
plant2 = ss([0.9 0.2; 0 0.7], [0; 0.5], [1 0.3], 0, 0.2);
for mi = 1:6
    o = mpc(plant2, 0.2, 15, 3, W2, MV2);
    o.Optimizer.SolverOptions.MaxIterations = mi;
    [u, info] = mpcmove(o, mpcstate(o), 0, 1);
    out.cold_u(mi) = u; out.cold_it(mi) = info.Iterations; out.cold_code{mi} = info.QPCode;
end
o = mpc(plant2, 0.2, 15, 3, W2, MV2);
st = mpcstate(o);
ym2 = [0 0.1 0.3 0.5 0.7 0.8 0.85 0.9];
for k = 1:numel(ym2)
    for mi = 1:4
        o2 = o; o2.Optimizer.SolverOptions.MaxIterations = mi;
        s2 = copy(st);
        [u, info] = mpcmove(o2, s2, ym2(k), 1);
        out.warm_u(k, mi) = u; out.warm_it(k, mi) = info.Iterations;
    end
    [~, info] = mpcmove(o, st, ym2(k), 1);
    out.warm_ref_it(k) = info.Iterations;
end
plant5 = setmpcsignals(ss([0 1; 0 0], [0; 1], eye(2), zeros(2,1)), 'MO', 1, 'UO', 2);
for mi = [2 3]
    for so = [0 1]
        o5 = mpc(plant5, 0.05, 20, [2 3 5]);
        o5.MV.Min = -3; o5.MV.Max = [3; 2.8; 2.6]; o5.MV.RateMin = -1; o5.MV.RateMax = 1;
        o5.OV(2).Min = -1.5; o5.OV(2).Max = 1.5;
        o5.Optimizer.SolverOptions.MaxIterations = mi;
        o5.Optimizer.UseSuboptimalSolution = logical(so);
        [y, ~, u, ~, ~, ~, status] = sim(o5, 60, [2 0]);
        out.(sprintf('dr%d_%d', mi, so)) = struct('y', y, 'u', u, 'st', status);
    end
end
o = mpc(plant2, 0.2, 15, 3, W2, MV2);
out.so_props = properties(o.Optimizer.SolverOptions);
out.so_maxit_default = o.Optimizer.SolverOptions.MaxIterations;
out.opt_fields = fieldnames(o.Optimizer);
try
    mpcmove(o, mpcstate(o), NaN, 1); out.e_nan = '';
catch e
    out.e_nan = e.identifier; out.e_nan_msg = e.message;
end
try
    mpcmove(o, mpcstate(o), Inf, 1); out.e_inf = '';
catch e
    out.e_inf = e.identifier;
end
try
    mpcmove(o, mpcstate(o), 0, NaN); out.e_rnan = '';
catch e
    out.e_rnan = e.identifier;
end
o6 = mpc(ss(0.9, 1, [1; 1], [0; 0], 0.1), 0.1, 10, 2);
try
    sim(o6, 5, 1); out.e_simr = '';
catch e
    out.e_simr = e.identifier; out.e_simr_msg = e.message;
end
kopt = mpcActiveSetOptions;
H = eye(3); f = [1; 1; 1];
try
    mpcActiveSetSolver(H, f, [1 0 0], 1, [1 1 0; 2 2 0], [1; 2], false, kopt); out.e_rank = '';
catch e
    out.e_rank = e.identifier;
end
try
    mpcActiveSetSolver(H, f, zeros(0,3), zeros(0,1), [eye(3); 1 1 1], ones(4,1), false(0,1), kopt); out.e_over = '';
catch e
    out.e_over = e.identifier;
end
try
    mpcActiveSetSolver(H, f, [eye(3); 1 1 1], ones(4,1), zeros(0,3), zeros(0,1), true(4,1), kopt); out.e_warm = '';
catch e
    out.e_warm = e.identifier;
end
try
    mpcActiveSetSolver(H, f, [eye(3); 1 1 1], ones(4,1), [1 1 1], 1, [true; true; true; false], kopt); out.e_warm2 = '';
catch e
    out.e_warm2 = e.identifier;
end
[x, ef] = mpcActiveSetSolver(H, f, [eye(3); 1 1 1], ones(4,1), [1 1 1], 1, [true; true; false; false], kopt);
out.ok_warm_x = x; out.ok_warm_ef = ef;
"""

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from spt.control.errors import ControlError  # noqa: E402
from spt.control.mpc import mpcActiveSetSolver, mpcmove, mpcstate, sim  # noqa: E402
from tests.test_control_mpc import case2, case5  # noqa: E402
from tools.matlab_oracle import OracleUnavailable, fixture  # noqa: E402


@pytest.fixture(scope="module")
def ref():
    try:
        return fixture("mpc_denetim_cases", DENETIM_SCRIPT)
    except OracleUnavailable as e:  # pragma: no cover
        pytest.skip(str(e))


def test_yineleme_siniri_kurali_matlab(ref):
    """Bulgu 1: QP adımı (ekleme / çıkarma) sayısı ≤ MaxIterations; bildirilen yineleme = 1 + adım. Soğuk
    başlangıçta 1 + 4 adımlık adım: MaxIterations 1..3 → durum 0 (MV tutulur), 4..6 → 5 (MATLAB ile birebir)."""
    for mi in range(1, 7):
        o = case2()
        o.Optimizer.SolverOptions.MaxIterations = mi            # MATLAB R2025b yazımı
        u, info = mpcmove(o, mpcstate(o), 0, 1)
        assert info["Iterations"] == ref["cold_it"][mi - 1], mi
        assert u[0] == pytest.approx(ref["cold_u"][mi - 1], abs=1e-12)
        assert info["QPCode"] == ref["cold_code"][mi - 1]


@pytest.mark.parametrize("so", [0, 1])
def test_drone_yineleme_siniri_3_matlab(ref, so):
    """Drone ekseni, MaxIterations = 3 (sıcak başlangıç çıkarmaları sınıra sayılır): u, y MATLAB ile ~1e-13."""
    o = case5()
    o.Optimizer.SolverOptions.MaxIterations = 3
    o.Optimizer.UseSuboptimalSolution = bool(so)
    s = sim(o, 60, [2, 0])
    r = ref[f"dr3_{so}"]
    assert np.abs(s.u - np.asarray(r["u"]).reshape(s.u.shape)).max() < 1e-9
    assert np.abs(s.y - np.asarray(r["y"]).reshape(s.y.shape)).max() < 1e-9
    assert np.sum(s.status == np.asarray(r["st"]).ravel()) >= 55   # sınırdaki adımlarda bildirilen sayı yol bağımlı
    assert s.status.max() <= 4 and np.asarray(r["st"]).max() <= 4


@pytest.mark.parametrize("so", [0, 1])
def test_drone_yineleme_siniri_2_kurali(ref, so):
    """MaxIterations = 2: bildirilen yineleme ≤ 3 (MATLAB'de de) ve sınır durumları (0) oluşur. Sınıra takılan
    adımlarda KWIK'in yolu farklı olabildiğinden u dizisinin birebir eşleşmesi beklenmez (TASARIM.md §7)."""
    o = case5()
    o.Optimizer.MaxIterations = 2
    o.Optimizer.UseSuboptimalSolution = bool(so)
    s = sim(o, 60, [2, 0])
    mst = np.asarray(ref[f"dr2_{so}"]["st"]).ravel()
    assert s.status.max() <= 3 and mst.max() == 3 and np.sum(s.status == 0) >= 3 and np.sum(mst == 0) >= 3
    assert np.array_equal(s.status[:3], mst[:3])


def test_gecersiz_olcum_ve_referans_hatasi(ref):
    """Bulgu 2 ve 9: NaN / ±∞ ölçüm MATLAB'de hata (kestirici NaN'a gitmez); ny = 2 iken skaler referans hata."""
    assert ref["e_nan"] == ref["e_inf"] == "MPC:computation:InvalidMeasurement"
    assert ref["e_rnan"] == "MPC:computation:InvalidSetpoint" and ref["e_simr"] == "MPC:computation:InvalidSetpointSize"
    o = case2()
    st = mpcstate(o)
    x0 = st.Plant.copy()
    for bad in (np.nan, np.inf):
        with pytest.raises(ControlError, match="sonlu"):
            mpcmove(o, st, bad, 1)
    assert np.array_equal(st.Plant, x0)                           # durum bozulmadı
    with pytest.raises(ControlError):
        mpcmove(o, st, 0, np.nan)
    from spt.control.mpc import MPC
    o6 = MPC(([[0.9]], [[1]], [[1], [1]], [[0], [0]], 0.1), 0.1, 10, 2)
    with pytest.raises(ControlError, match="2 sütunlu"):
        sim(o6, 5, 1)


def test_solveroptions_ve_activeset_butunluk(ref):
    """Bulgu 7 ve 8: MATLAB R2025b Optimizer.SolverOptions alanları; mpcActiveSetSolver bütünlük hataları."""
    assert ref["so_props"] == ["MaxIterations", "ConstraintTolerance", "UseWarmStart"]
    o = case2()
    assert o.Optimizer.SolverOptions.MaxIterations == ref["so_maxit_default"] == "default"
    o.Optimizer.SolverOptions = {"ConstraintTolerance": 1e-7, "UseWarmStart": False}
    assert o.Optimizer.ConstraintTolerance == 1e-7 and o.data.tol == 1e-7 and not o.data.use_warm_start
    H, f = np.eye(3), np.ones(3)
    assert ref["e_rank"] == "MPC:ActiveSetSolver:AeqRank"
    with pytest.raises(ControlError, match="rank"):
        mpcActiveSetSolver(H, f, [[1, 0, 0]], [1], [[1, 1, 0], [2, 2, 0]], [1, 2], [False])
    assert ref["e_over"] == "MPC:ActiveSetSolver:OverSpecEquaCons"
    with pytest.raises(ControlError, match="fazla"):
        mpcActiveSetSolver(H, f, None, None, np.vstack([np.eye(3), np.ones(3)]), np.ones(4), None)
    assert ref["e_warm"] == ref["e_warm2"] == "MPC:ActiveSetSolver:InvalidWarmStart"
    A4 = np.vstack([np.eye(3), np.ones(3)])
    with pytest.raises(ControlError, match="aşıyor"):
        mpcActiveSetSolver(H, f, A4, np.ones(4), None, None, np.ones(4, bool))
    with pytest.raises(ControlError, match="aşıyor"):
        mpcActiveSetSolver(H, f, A4, np.ones(4), [[1, 1, 1]], [1], [True, True, True, False])
    x, ef, _, _ = mpcActiveSetSolver(H, f, A4, np.ones(4), [[1, 1, 1]], [1], [True, True, False, False])
    assert ef > 0 and np.allclose(x, np.asarray(ref["ok_warm_x"]), atol=1e-12)
