"""MPC C kodu üreteci — üretilen C'nin Python MPC'si (spt.control.mpc) ve MATLAB R2025b ile karşılaştırılması.

Referanslar:
  - `tests/data/matlab/mpc_codegen_cases.json`: MATLAB'in KENDİ kod üretim yolu (`getCodeGenerationData` +
    `mpcmoveCodeGeneration`) kapalı çevrimde (bitki betikte benzetilir): SISO kısıtsız, MV + oran kısıtlı,
    MIMO yumuşak OV (ölçek + MV hedefi; önizlemeli / önizlemesiz), MD + nominal (önizlemeli / önizlemesiz), drone
    irtifa ekseni (çift tümlev, UO hız kısıtı, blok hamle), UD (giriş bozucu tümlevi) + bozucu enjeksiyonu, çözümsüz
    QP (MV tutulur).
  - `tests/data/matlab/mpc_cases.json` (MPC modülünün kahini): `sim` ve `mpcmove` ayrıntıları (özel bozucu /
    gürültü modelleri, model uyumsuzluğu dahil).
Çift duyarlıkta üretilen C ile Python arasındaki fark ~1e-13, MATLAB ile ~1e-12'dir (testler 1e-10 / 1e-9 ister);
yineleme sayıları ve durum kodları Python ile birebir aynıdır.
"""
from __future__ import annotations

import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from spt.codegen import mpc_c as mc
from spt.codegen import toolchain as tc
from spt.codegen.mcu import MCU_PROFILES, OPS
from spt.control.mpc import MPC, mpcmove, mpcsimopt, mpcstate, setmpcsignals, setoutdist, sim
from tools.matlab_oracle import OracleUnavailable, fixture

HOST_CC = tc.find_host_cc()
HOST_CXX = tc.find_host_cxx()
ARM_GCC = tc.find_arm_gcc()
CLANG = tc.find_clang()
needs_cc = pytest.mark.skipif(HOST_CC is None, reason="ana makine C derleyicisi yok")
needs_cxx = pytest.mark.skipif(HOST_CXX is None, reason="ana makine C++ derleyicisi yok")
needs_cross = pytest.mark.skipif(ARM_GCC is None and CLANG is None, reason="ARM çapraz derleyici yok")

DOUBLE = mc.MpcCodegenOptions(real="double")
FLOAT = mc.MpcCodegenOptions(real="float")
TOL_PY = 1e-10        # üretilen C (double) ↔ Python MPC (ölçülen ~1e-13)
TOL_ML = 1e-9         # üretilen C (double) ↔ MATLAB (ölçülen ~1e-12)

CODEGEN_SCRIPT = r"""
mpcverbosity off;
C = {};
P1 = ss([0 1; -2 -3], [0; 1], [1 0], 0);
o1 = mpc(P1, 0.1);
P1d = c2d(P1, 0.1);
C{end+1} = struct('obj', o1, 'A', P1d.A, 'B', P1d.B, 'C', P1d.C, 'D', P1d.D, 'X0', [0; 0], 'U0', 0, 'Y0', 0, ...
    'DX', [0; 0], 'mv', 1, 'md', [], 'ud', [], 'mo', 1, 'T', 30, 'r', [ones(15, 1); 0.5 * ones(15, 1)], ...
    'v', zeros(30, 0), 'd', zeros(30, 0), 'rp', 0, 'vp', 0);
P2 = ss([0.9 0.2; 0 0.7], [0; 0.5], [1 0.3], 0, 0.2);
o2 = mpc(P2, 0.2, 15, 3, struct('ManipulatedVariables', 0, 'ManipulatedVariablesRate', 0.3, ...
    'OutputVariables', 2), struct('Min', -0.4, 'Max', 0.6, 'RateMin', -0.15, 'RateMax', 0.2));
C{end+1} = struct('obj', o2, 'A', P2.A, 'B', P2.B, 'C', P2.C, 'D', P2.D, 'X0', [0; 0], 'U0', 0, 'Y0', 0, ...
    'DX', [0; 0], 'mv', 1, 'md', [], 'ud', [], 'mo', 1, 'T', 40, 'r', [ones(20, 1); -0.5 * ones(20, 1)], ...
    'v', zeros(40, 0), 'd', zeros(40, 0), 'rp', 0, 'vp', 0);
A3 = [0.8 0.1 0 0; 0 0.7 0 0; 0 0 0.85 0.05; 0 0 0 0.6];
B3 = [0.5 0; 0.3 0.1; 0 0.4; 0.2 0.3];
C3 = [1 0 0.5 0; 0 0.3 1 0];
o3 = mpc(ss(A3, B3, C3, zeros(2, 2), 0.5), 0.5, 12, 3);
o3.Weights.OutputVariables = [1 0.5];
o3.Weights.ManipulatedVariables = [0.1 0];
o3.Weights.ManipulatedVariablesRate = [0.2 0.1];
o3.MV(1).Target = 0.3;
o3.MV(1).Min = -1; o3.MV(1).Max = 1; o3.MV(2).Min = -2; o3.MV(2).Max = 2;
o3.OV(1).Max = 1.05; o3.OV(2).Min = -0.2; o3.OV(2).MinECR = 0.5;
o3.OV(1).ScaleFactor = 2; o3.MV(2).ScaleFactor = 4;
g3 = struct('obj', o3, 'A', A3, 'B', B3, 'C', C3, 'D', zeros(2, 2), 'X0', zeros(4, 1), 'U0', [0; 0], ...
    'Y0', [0; 0], 'DX', zeros(4, 1), 'mv', [1 2], 'md', [], 'ud', [], 'mo', [1 2], 'T', 30, ...
    'r', [ones(15, 1) * [1 0.5]; ones(15, 1) * [1.2 -0.5]], 'v', zeros(30, 0), 'd', zeros(30, 0), 'rp', 0, 'vp', 0);
C{end+1} = g3;
g3.rp = 1;
C{end+1} = g3;
A4 = [0.9 0; 0.1 0.8]; B4 = [0.4 0.2; 0 0.3]; C4 = [0 1]; D4 = [0 0.1];
plant4 = setmpcsignals(ss(A4, B4, C4, D4, 0.1), 'MV', 1, 'MD', 2);
o4 = mpc(struct('Plant', plant4, 'Nominal', struct('U', [1; 0.5], 'Y', 2, 'X', [0.5; 1], 'DX', [0.01; 0])), ...
    0.1, 10, 2);
o4.MV.Min = 0; o4.MV.Max = 2.5;
g5 = struct('obj', o4, 'A', A4, 'B', B4, 'C', C4, 'D', D4, 'X0', [0.5; 1], 'U0', [1; 0.5], 'Y0', 2, ...
    'DX', [0.01; 0], 'mv', 1, 'md', 2, 'ud', [], 'mo', 1, 'T', 30, 'r', 2.5 * ones(30, 1), ...
    'v', [0.5 * ones(10, 1); 1.5 * ones(20, 1)], 'd', zeros(30, 0), 'rp', 0, 'vp', 0);
C{end+1} = g5;
g5.vp = 1;
C{end+1} = g5;
Pz = ss([0 1; 0 0], [0; 1], eye(2), zeros(2, 1));
oz = mpc(setmpcsignals(Pz, 'MO', 1, 'UO', 2), 0.02, 30, [2 3 5 10 10]);
oz.MV.Min = -4; oz.MV.Max = 4; oz.MV.RateMin = -0.5; oz.MV.RateMax = 0.5; oz.MV.ScaleFactor = 4;
oz.OV(1).ScaleFactor = 5; oz.OV(2).Min = -2; oz.OV(2).Max = 2; oz.OV(2).ScaleFactor = 2;
oz.Weights.OutputVariables = [1 0]; oz.Weights.ManipulatedVariablesRate = 0.05;
Pzd = c2d(Pz, 0.02);
C{end+1} = struct('obj', oz, 'A', Pzd.A, 'B', Pzd.B, 'C', Pzd.C, 'D', Pzd.D, 'X0', [0; 0], 'U0', 0, ...
    'Y0', [0; 0], 'DX', [0; 0], 'mv', 1, 'md', [], 'ud', [], 'mo', 1, 'T', 200, ...
    'r', [3 * ones(100, 1) zeros(100, 1); 0.5 * ones(100, 1) zeros(100, 1)], 'v', zeros(200, 0), ...
    'd', zeros(200, 0), 'rp', 0, 'vp', 0);
plant6 = setmpcsignals(ss([0.85 0.1; 0 0.9], [0.3 0.2; 0.1 0.4], [1 0.2], [0 0], 1), 'MV', 1, 'UD', 2);
o6 = mpc(plant6, 1, 8, 2); o6.MV.Min = -2; o6.MV.Max = 2;
C{end+1} = struct('obj', o6, 'A', plant6.A, 'B', plant6.B, 'C', plant6.C, 'D', plant6.D, 'X0', [0; 0], ...
    'U0', [0; 0], 'Y0', 0, 'DX', [0; 0], 'mv', 1, 'md', [], 'ud', 2, 'mo', 1, 'T', 40, 'r', ones(40, 1), ...
    'v', zeros(40, 0), 'd', [zeros(15, 1); 0.5 * ones(25, 1)], 'rp', 0, 'vp', 0);
for ci = 1:numel(C)
    c = C{ci};
    s = struct();
    try
        [cfg, st, od] = getCodeGenerationData(c.obj);
        T = c.T; p = c.obj.PredictionHorizon; x = c.X0;
        s.u = zeros(T, numel(c.mv)); s.y = zeros(T, size(c.C, 1)); s.it = zeros(T, 1); s.slack = zeros(T, 1);
        s.xp = zeros(T, numel(st.Plant)); s.xd = zeros(T, numel(st.Disturbance));
        for t = 1:T
            w = c.U0;
            if ~isempty(c.md), w(c.md) = c.v(t, :)'; end
            if ~isempty(c.ud), w(c.ud) = c.d(t, :)'; end
            y = c.Y0 + c.C * (x - c.X0) + c.D * (w - c.U0);
            od.signals.ym = y(c.mo);
            if c.rp
                od.signals.ref = c.r(min(t + (1:p), T), :);
            else
                od.signals.ref = c.r(t, :);
            end
            if ~isempty(c.md)
                if c.vp
                    od.signals.md = c.v(min(t + (0:p), T), :);
                else
                    od.signals.md = c.v(t, :);
                end
            end
            [u, st, info] = mpcmoveCodeGeneration(cfg, st, od);
            w(c.mv) = u;
            x = c.X0 + c.A * (x - c.X0) + c.B * (w - c.U0) + c.DX;
            s.u(t, :) = u'; s.y(t, :) = y'; s.it(t) = info.Iterations; s.slack(t) = info.Slack;
            s.xp(t, :) = st.Plant'; s.xd(t, :) = st.Disturbance';
        end
    catch err
        s.err = err.message;
    end
    out.(sprintf('g%d', ci)) = s;
end
[cfg, st, od] = getCodeGenerationData(o2);
st.LastMove = 2; od.signals.ym = 0; od.signals.ref = 1;
[u, st, info] = mpcmoveCodeGeneration(cfg, st, od);
out.inf.u = u; out.inf.it = info.Iterations; out.inf.xp = st.Plant; out.inf.fields = fieldnames(info);
"""
MPC_CASES_JSON = Path(__file__).resolve().parent / "data" / "matlab" / "mpc_cases.json"


@pytest.fixture(scope="module")
def ref_cg():
    try:
        return fixture("mpc_codegen_cases", CODEGEN_SCRIPT)
    except OracleUnavailable as e:  # pragma: no cover
        pytest.skip(str(e))


@pytest.fixture(scope="module")
def ref_mpc():
    """MPC modülünün MATLAB kahini (sim / mpcmove); betik bu testteki durum tanımlarıyla aynı olmalı."""
    if not MPC_CASES_JSON.exists():  # pragma: no cover
        pytest.skip("mpc_cases.json yok")
    data = json.loads(MPC_CASES_JSON.read_text(encoding="utf-8"))
    script = data.get("script", "")
    for line in ("obj2 = mpc(plant2, 0.2, 15, 3, W2, MV2);", "obj5 = mpc(plant5, 0.05, 20, [2 3 5]);",
                 "setoutdist(obj7, 'model', ss(0.8, 0.5, [1; 0], [0; 0], 0.1));"):
        if line not in script:  # pragma: no cover
            pytest.skip("mpc_cases.json betiği değişmiş")
    return data["out"]


def _a(x, shape=None):
    a = np.asarray([] if x is None else x, dtype=float)
    return a.reshape(shape) if shape is not None else a


def _close(ours, theirs, tol, what=""):
    ours = np.asarray(ours, dtype=float)
    theirs = _a(theirs, ours.shape)
    err = np.abs(ours - theirs).max() if ours.size else 0.0
    scale = 1.0 + (np.abs(theirs).max() if theirs.size else 0.0)
    assert err <= tol * scale, f"{what}: fark {err:.3e}"


# ============================================================================ durumlar
def g_cases():
    """CODEGEN_SCRIPT durumlarının Python karşılıkları: (mpc, T, r, v, UD, ref önizleme, MD önizleme)."""
    out = {}
    o1 = MPC(([[0, 1], [-2, -3]], [[0], [1]], [[1, 0]], [[0]]), 0.1)
    out[1] = (o1, 30, np.r_[np.ones(15), 0.5 * np.ones(15)], None, None, False, False)
    out[2] = (case2(), 40, np.r_[np.ones(20), -0.5 * np.ones(20)], None, None, False, False)
    r3 = np.vstack([np.tile([1, 0.5], (15, 1)), np.tile([1.2, -0.5], (15, 1))])
    out[3] = (case3(), 30, r3, None, None, False, False)
    out[4] = (case3(), 30, r3, None, None, True, False)
    v4 = np.r_[0.5 * np.ones(10), 1.5 * np.ones(20)]
    out[5] = (case4(), 30, 2.5 * np.ones(30), v4, None, False, False)
    out[6] = (case4(), 30, 2.5 * np.ones(30), v4, None, False, True)
    rz = np.vstack([np.tile([3, 0], (100, 1)), np.tile([0.5, 0], (100, 1))])
    out[7] = (drone_altitude(), 200, rz, None, None, False, False)
    out[8] = (case6(), 40, np.ones(40), None, np.r_[np.zeros(15), 0.5 * np.ones(25)], False, False)
    return out


def drone_altitude():
    """Drone irtifa ekseni: z̈ = a (çift tümlev), Ts = 20 ms, p = 30, blok hamle [2 3 5 10 10]; ivme ±4 m/s²
    (ölçek 4), oran ±0.5 / adım, dikey hız (ölçülmeyen çıkış) ±2 m/s yumuşak kısıt."""
    pz = setmpcsignals(([[0, 1], [0, 0]], [[0], [1]], np.eye(2), np.zeros((2, 1))), MO=0, UO=1)
    o = MPC(pz, 0.02, 30, [2, 3, 5, 10, 10])
    o.MV[0].Min, o.MV[0].Max, o.MV[0].RateMin, o.MV[0].RateMax, o.MV[0].ScaleFactor = -4, 4, -0.5, 0.5, 4
    o.OV[0].ScaleFactor = 5
    o.OV[1].Min, o.OV[1].Max, o.OV[1].ScaleFactor = -2, 2, 2
    o.Weights.OutputVariables = [1, 0]
    o.Weights.ManipulatedVariablesRate = 0.05
    return o


# --- mpc_cases.json durumları (tests/test_control_mpc.py ile aynı tanımlar) ----------------------------------
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


def sim_case(k):
    """mpc_cases.json sim durumları: (mpc, T, r, v, seçenekler, kahin anahtarı)."""
    v4 = np.r_[0.5 * np.ones(10), 1.5 * np.ones(20)]
    t = np.arange(1, 31)[:, None]
    return {1: (case1(), 30, 1, None, None, "c1"),
            2: (case2(), 40, np.r_[np.ones(20), -0.5 * np.ones(20)], None, None, "c2"),
            3: (case3(), 30, np.vstack([np.tile([1, 0.5], (15, 1)), np.tile([1.2, -0.5], (15, 1))]), None, None,
                "c3"),
            4: (case4(), 30, 2.5 * np.ones(30), v4, None, "c4"),
            5: (case5(), 60, [2, 0], None, None, "c5"),
            6: (case6(), 40, 1, None, mpcsimopt(Model=TRUE6, UnmeasuredDisturbance=np.r_[np.zeros(15),
                                                                                          0.5 * np.ones(25)]), "c6"),
            7: (case7(), 30, [1, 0.5], None, mpcsimopt(UnmeasuredDisturbance=np.r_[np.zeros(10), 0.3 * np.ones(20)],
                                                       OutputNoise=0.01 * np.sin(t * np.array([[0.5, 0.9]]))), "c7"),
            8: (case4b(), 30, 2.5 * np.ones(30), v4, mpcsimopt(MDLookAhead="on"), "c4b")}[k]


def _sim_opts(dist, rp, vp):
    return mpcsimopt(UnmeasuredDisturbance=dist, RefLookAhead="on" if rp else None, MDLookAhead="on" if vp else None)


# ============================================================================ üretilen dosyalar
def test_dosyalar_onek_ve_api():
    o = case4()
    files = {f.name: f.text for f in mc.generate(o, mc.MpcCodegenOptions(prefix="irtifa"), name="İrtifa MPC")}
    assert set(files) == {"irtifa.h", "irtifa.c", "irtifa_ornek.c", "KULLANIM_irtifa.md"}
    for name, text in files.items():
        assert not re.search(r"\bspt_|\bSPT_|spt\.h|KULLANIM_spt", text), name    # önek her yerde uygulandı
        if name.endswith((".c", ".h")):
            long = [ln for ln in text.splitlines() if len(ln) > 120]
            assert not long, (name, long[:3])
            assert "malloc" not in text.replace("malloc yok", "")
    h = files["irtifa.h"]
    d = o.data
    for macro, val in (("NX", d.nx), ("NXP", d.nxp), ("NMV", d.nmv), ("NMD", d.nmd), ("NYM", d.nmy), ("NY", d.ny),
                       ("P", d.p), ("NFREE", d.nfree), ("NZ", d.degrees), ("NC", d.Ac.shape[0])):
        assert re.search(rf"#define IRTIFA_{macro}\s+{val}\b", h), macro
    assert "#define IRTIFA_MAX_ITER 120" in h and "IRTIFA_REF_PREVIEW 0" in h
    sig = "int irtifa_step("
    assert (sig + "irtifa_t *s, const IRTIFA_REAL *ym, const IRTIFA_REAL *ref, int ref_rows,\n" + " " * len(sig)
            + "const IRTIFA_REAL *md, int md_rows, IRTIFA_REAL u[IRTIFA_NMV]);") in h       # imza hizalı
    c = files["irtifa.c"]
    assert "static const IRTIFA_REAL irtifa_Linv[IRTIFA_NZ * (IRTIFA_NZ + 1) / 2]" in c
    assert "irtifa_Bv_off" in c                                      # Nominal.DX ofset sütunu
    assert c.count("static const IRTIFA_REAL") == len(mc.plan(o).arrays)
    # kısıtsız tasarımda QP kodu yok, kapalı biçim var
    cu = {f.name: f.text for f in mc.generate(case1())}
    assert "mpc_qp_solve" not in cu["mpc.c"] and "mpc_Kmpc" in cu["mpc.c"] and "MPC_NC    0" in cu["mpc.h"]
    assert mc.default_prefix("İrtifa MPC") == "irtifa_mpc"
    assert mc.default_prefix("1. eksen") == "mpc_1_eksen"
    assert mc.default_prefix("") == "mpc"


def test_gecersiz_secenekler():
    o = case2()
    for bad in ("_x", "a__b", "x_", "1abc", "", "a-b"):
        with pytest.raises(ValueError, match="Önek"):
            mc.generate(o, mc.MpcCodegenOptions(prefix=bad))
    for kw in (dict(real="half"), dict(mcu="z80"), dict(compiler="tcc"), dict(opt="-O9")):
        with pytest.raises(ValueError):
            mc.generate(o, mc.MpcCodegenOptions(**kw))
    o.Optimizer.MaxIterations = 0
    with pytest.raises(ValueError, match="MaxIterations"):
        mc.generate(o)


# ============================================================================ doğruluk: MATLAB ve Python
@needs_cc
@pytest.mark.parametrize("k", range(1, 9))
def test_double_matlab_kod_uretim_yolu_ile_ayni(ref_cg, k):
    """Kapalı çevrim: üretilen C (double) = MATLAB getCodeGenerationData + mpcmoveCodeGeneration (u, y, kestirici
    bitki durumu, gevşek değişken) ve = Python sim (durum kodları / yineleme sayıları birebir)."""
    o, T, r, v, dist, rp, vp = g_cases()[k]
    g = ref_cg[f"g{k}"]
    assert "err" not in g, g.get("err")
    opt = _sim_opts(dist, rp, vp)
    res = mc.run_closed_loop(o, mc.MpcCodegenOptions(real="double", ref_preview=rp, md_preview=vp), T, r, v, opt)
    _close(res.u, g["u"], TOL_ML, f"g{k} u")
    _close(res.y, g["y"], TOL_ML, f"g{k} y")
    xp = _a(g["xp"]).reshape(T, -1)
    _close(res.x[:, :xp.shape[1]], xp, TOL_ML, f"g{k} kestirici bitki durumu")
    _close(res.slack, g["slack"], TOL_ML, f"g{k} gevşek")
    py = sim(o, T, r, v, opt)
    _close(res.u, py.u, TOL_PY, f"g{k} u (Python)")
    _close(res.y, py.y, TOL_PY, f"g{k} y (Python)")
    assert np.array_equal(res.status, py.status.astype(int))
    assert np.all(res.status > 0)
    same = np.mean(res.status == _a(g["it"]).ravel())        # KWIK'in yolu yayımlanmamış: çoğunlukla aynı
    assert same >= 0.9, same


@needs_cc
@pytest.mark.parametrize("k", range(1, 9))
def test_double_python_ve_matlab_sim_ile_ayni(ref_mpc, k):
    """mpc_cases.json'daki tüm sim durumları (özel bozucu / gürültü modelleri, model uyumsuzluğu + UD, ölçek + MD
    önizlemesi dahil): C = Python sim (≤ 1e-10, durum kodları birebir) ve = MATLAB sim (≤ 1e-9)."""
    o, T, r, v, opt, key = sim_case(k)
    opts = mc.MpcCodegenOptions(real="double", md_preview=(k == 8))
    res = mc.run_closed_loop(o, opts, T, r, v, opt)
    py = sim(o, T, r, v, opt)
    _close(res.u, py.u, TOL_PY, f"durum {k} u")
    _close(res.y, py.y, TOL_PY, f"durum {k} y")
    assert np.array_equal(res.status, py.status.astype(int))
    _close(res.u, ref_mpc[key]["u"], TOL_ML, f"durum {k} u (MATLAB)")
    _close(res.y, ref_mpc[key]["y"], TOL_ML, f"durum {k} y (MATLAB)")
    # kestirici durumu Python'un iç durumu (sonraki adımın x̂(k|k−1)) ile
    _close(res.x[:-1], np.hstack([py.xmpc["Plant"], py.xmpc["Disturbance"], py.xmpc["Noise"]])[1:], TOL_PY,
           f"durum {k} x̂")


@needs_cc
def test_mpcmove_dizisi_ve_cozumsuz_qp(ref_mpc, ref_cg):
    """Açık çevrim mpcmove dizisi (MATLAB: u, gevşek, yineleme, x̂(k+1|k), son MV) ve çözümsüz QP: MV tutulur
    (C −1; MATLAB KWIK −2 — ikisi de 'infeasible', MV = 2)."""
    r = ref_mpc["c2"]
    o = case2()
    ym = np.array([0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9])[:, None]
    res = mc.run_steps(o, DOUBLE, ym, ref=np.ones((8, 1, 1)))
    _close(res.u[:, 0], r["mv_u"], TOL_ML, "u")
    _close(res.slack, r["mv_slack"], TOL_ML, "gevşek")
    assert list(res.status) == [int(i) for i in r["mv_iter"]]
    _close(res.x[:, :2], r["mv_xp"], TOL_ML, "x̂ bitki")
    _close(res.x[:, 2:], _a(r["mv_xd"]).reshape(8, -1), TOL_ML, "x̂ bozucu")
    st = mpcstate(o)
    for k in range(8):
        u, info = mpcmove(o, st, ym[k], 1)
        assert u[0] == pytest.approx(res.u[k, 0], abs=TOL_PY) and info["Iterations"] == res.status[k]
    # çözümsüz: son MV = 2 (üst sınır 0.6, oran 0.15 / adım → bir adımda sınıra dönülemez)
    inf = mc.run_steps(o, DOUBLE, [[0.0]], ref=[[[1.0]]], x0=np.zeros(3), u0=[2.0])
    assert inf.status[0] == -1 and inf.u[0, 0] == 2.0 == ref_cg["inf"]["u"] == r["inf_u"]
    assert ref_cg["inf"]["it"] < 0 and r["inf_code"] == "infeasible"
    _close(inf.x[0, :2], ref_cg["inf"]["xp"], TOL_ML, "çözümsüz adımda kestirici")
    st = mpcstate(o)
    st.LastMove = np.array([2.0])
    u, info = mpcmove(o, st, 0.0, 1.0)
    assert u[0] == 2.0 and info["Iterations"] == -1


@needs_cc
def test_onizleme_ve_md_mpcmove(ref_mpc):
    """Referans önizlemesi (MATLAB mpcmove, 5 satır < p: son satırla doldurulur; p'den fazla satır kesilir) ve
    MD + nominal mpcmove."""
    o = case3()
    rp = np.array([[0.5, 0.1], [0.7, 0.2], [0.9, 0.3], [1, 0.4], [1.1, 0.5]])
    opts = mc.MpcCodegenOptions(real="double", ref_preview=True)
    res = mc.run_steps(o, opts, [[0.1, 0.2]], ref=rp[None])
    _close(res.u[0], ref_mpc["c3"]["pv_u"], TOL_ML, "önizlemeli u")
    _close(res.slack, [ref_mpc["c3"]["pv_slack"]], TOL_ML, "önizlemeli gevşek")
    long_ref = np.vstack([rp, np.tile([2.0, -1.0], (o.data.p + 2 - len(rp), 1))])    # p + 2 satır
    res2 = mc.run_steps(o, opts, [[0.1, 0.2]], ref=long_ref[None])
    u2, _ = mpcmove(o, mpcstate(o), [0.1, 0.2], long_ref)
    _close(res2.u[0], u2, TOL_PY, "p'den uzun önizleme")
    r4 = ref_mpc["c4"]
    o4 = case4()
    res4 = mc.run_steps(o4, DOUBLE, [[2.1]], ref=[[[2.5]]], md=[[[0.7]]])
    _close(res4.u[0], [r4["mv_u"]], TOL_ML, "MD u")
    _close(res4.x[0, :2], r4["mv_xp"], TOL_ML, "MD x̂ bitki")
    _close(res4.x[0, 2:], r4["mv_xd"], TOL_ML, "MD x̂ bozucu")
    o4b = case4b()
    vprev = np.array([[0.7], [0.9], [1.4]])                   # 3 satır < p + 1: son satırla doldurulur
    res5 = mc.run_steps(o4b, mc.MpcCodegenOptions(real="double", md_preview=True), [[2.1]], ref=[[[2.5]]],
                        md=vprev[None])
    u5, _ = mpcmove(o4b, mpcstate(o4b), 2.1, 2.5, vprev)
    _close(res5.u[0], u5, TOL_PY, "MD önizlemesi")


@needs_cc
@pytest.mark.parametrize("k", [2, 3, 6, 7, 8])
def test_float_dogrulugu(ref_cg, k):
    """float (MCU'da önerilen tür): Python double ile u / y farkı ≤ 1e-3 × ölçek, durum kodları ≥ %95 aynı."""
    o, T, r, v, dist, rp, vp = g_cases()[k]
    opt = _sim_opts(dist, rp, vp)
    res = mc.run_closed_loop(o, mc.MpcCodegenOptions(real="float", ref_preview=rp, md_preview=vp), T, r, v, opt)
    py = sim(o, T, r, v, opt)
    d = o.data
    du = np.abs(res.u - py.u).max(axis=0) / d.Uscale[d.mvindex]
    dy = np.abs(res.y - py.y).max(axis=0) / d.Yscale
    assert du.max() < 1e-3 and dy.max() < 1e-3, (du, dy)
    assert np.mean(res.status == py.status) >= 0.95
    assert np.all(res.status > 0)


@needs_cc
@pytest.mark.parametrize("subopt", [False, True])
def test_yineleme_siniri_mv_tutulur_ya_da_kirpilir(subopt):
    """MaxIterations = 2: sınıra takılan adımda MV tutulur (UseSuboptimalSolution kapalı) ya da alt-en iyi hamle
    MV / oran sınırlarına kırpılır (açık) — Python sim ile birebir."""
    o = case5()
    o.Optimizer.MaxIterations = 2
    o.Optimizer.UseSuboptimalSolution = subopt
    res = mc.run_closed_loop(o, DOUBLE, 60, [2, 0])
    py = sim(o, 60, [2, 0])
    assert np.array_equal(res.status, py.status.astype(int)) and np.sum(res.status == 0) >= 3
    _close(res.u, py.u, TOL_PY, "u")
    _close(res.y, py.y, TOL_PY, "y")
    held = np.flatnonzero(res.status == 0)
    if not subopt:
        prev = np.vstack([[0.0], res.u[:-1]])
        assert np.array_equal(res.u[held], prev[held])          # MV önceki değerinde
    else:
        assert np.all(np.abs(res.u) <= 3 + 1e-12) and np.all(np.abs(np.diff(np.r_[0, res.u[:, 0]])) <= 1 + 1e-12)


@needs_cc
def test_sicak_baslangic_kapali_python_ile_ayni():
    o = drone_altitude()
    o.Optimizer.UseWarmStart = False
    rz = np.vstack([np.tile([3, 0], (60, 1)), np.tile([0.5, 0], (60, 1))])
    res = mc.run_closed_loop(o, DOUBLE, 120, rz)
    py = sim(o, 120, rz)
    assert np.array_equal(res.status, py.status.astype(int)) and res.status.max() > 3
    _close(res.u, py.u, TOL_PY, "u")


@needs_cc
def test_gecersiz_giris_adim_yapmaz():
    """NaN / ∞ ölçüm, geçersiz satır sayısı → −3: MV önceki değer, kestirici durumu değişmez; sonra kaldığı yerden
    Python mpcmove ile aynı devam eder."""
    o = case2()
    nan = float("nan")
    ym = np.array([[0.0], [nan], [0.1], [np.inf], [0.3]])
    res = mc.run_steps(o, DOUBLE, ym, ref=np.ones((5, 1, 1)))
    assert list(res.status[[1, 3]]) == [-3, -3]
    assert np.array_equal(res.u[1], res.u[0]) and np.array_equal(res.x[1], res.x[0])
    assert np.array_equal(res.u[3], res.u[2]) and np.array_equal(res.x[3], res.x[2])
    st = mpcstate(o)
    for k in (0, 2, 4):
        u, info = mpcmove(o, st, ym[k], 1)
        assert u[0] == pytest.approx(res.u[k, 0], abs=TOL_PY) and info["Iterations"] == res.status[k]
    bad_ref = mc.run_steps(o, DOUBLE, [[0.0]], ref=np.ones((1, 2, 1)))          # önizlemesiz üretimde 2 satır
    assert bad_ref.status[0] == -3 and bad_ref.u[0, 0] == 0.0
    prev = mc.MpcCodegenOptions(real="double", ref_preview=True)
    p = o.data.p
    rows = np.ones((1, p + 2, 1))
    rows[0, p:] = nan                                            # p'den sonraki satırlar kullanılmaz → okunmaz
    assert mc.run_steps(o, prev, [[0.0]], ref=rows).status[0] > 0
    rows[0, 3] = nan                                             # kullanılan aralıkta NaN → adım yok
    assert mc.run_steps(o, prev, [[0.0]], ref=rows).status[0] == -3


# ============================================================================ derleme
def _plans():
    """Derleme denetimi için tasarımlar: kısıtsız; MD + önizleme + alt-en iyi + Nominal.DX; drone (soğuk başlangıç)."""
    o8 = case4b()
    o8.Optimizer.UseSuboptimalSolution = True
    o5 = drone_altitude()
    o5.Optimizer.UseWarmStart = False
    return [(case1(), mc.MpcCodegenOptions(prefix="kisitsiz")),
            (o8, mc.MpcCodegenOptions(prefix="md_onizleme", ref_preview=True, md_preview=True)),
            (o5, mc.MpcCodegenOptions(prefix="irtifa"))]


@needs_cross
def test_capraz_derleme_uyarisiz_tum_cortex_m():
    """Tüm yerel Cortex-M profilleri × GCC / Clang × float / double × -O0 / -Os / -O2: -std=c99 -Wall -Wextra
    -pedantic -Werror (float'ta -Wdouble-promotion) uyarısız; örnek dosya dahil."""
    jobs = []
    for o, opts in _plans():
        files = {f.name: f.text for f in mc.generate(o, opts)}
        srcs = [n for n in files if n.endswith(".c")]
        P = opts.prefix.upper()
        for prof in MCU_PROFILES.values():
            for comp in ("gcc", "clang"):
                if tc.cross_flags(prof, comp) is None:
                    continue
                for real in ("0", "1"):
                    for opt in ("-O0", "-Os", "-O2"):
                        extra = ["-Wdouble-promotion"] if real == "0" else []
                        jobs.append((files, srcs, prof, comp, opt, [f"{P}_USE_DOUBLE={real}"], extra))
    if not jobs:
        pytest.skip("çapraz derleyici yok")

    def run(j):
        files, srcs, prof, comp, opt, defines, extra = j
        b = tc.compile_cross(files, srcs, prof, comp, opt, defines=defines, extra=extra, stack_usage=False)
        return f"{prof.key}/{comp}/{opt}/{defines[0]}", b

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run, jobs))
    failures = [f"{name}:\n{b.log}" for name, b in results if not b.ok or b.log.strip()]
    assert not failures, "\n\n".join(failures[:3])
    assert len(results) >= 3 * 4 * 6


@needs_cxx
@pytest.mark.parametrize("real", ["0", "1"])
def test_cpp_derleyicisiyle_uyarisiz(real):
    """C++ projelerinden (STM32CubeIDE C++ …) de derlenir: -x c++ -std=c++11 -Wall -Wextra -pedantic -Werror."""
    for o, opts in _plans():
        folder = tc.WORK_ROOT / "cxx_check_mpc" / f"{opts.prefix}_{real}"
        shutil.rmtree(folder, ignore_errors=True)
        mc.write_files(mc.generate(o, opts), folder)
        sources = sorted(p.name for p in folder.glob("*.c"))
        res = tc._run([HOST_CXX.path, "-x", "c++", "-std=c++11", *tc.WARN_FLAGS,
                       f"-D{opts.prefix.upper()}_USE_DOUBLE={real}", "-fsyntax-only", *sources], cwd=folder,
                      path_prefix=[HOST_CXX.bin_dir])
        assert res.returncode == 0 and not res.stderr.strip(), res.stderr


# ============================================================================ ayak izi
@needs_cross
@pytest.mark.parametrize("real", ["float", "double"])
def test_ayak_izi_olculur_ve_size_araciyla_ayni(real):
    """Cortex-M4F -Os: kod + sabitler nesneden ölçülür (= arm-none-eabi-size text + data), RAM'de global yok,
    RAM durum = sizeof = yapı düzeni formülü, yığın statik analizi (GCC -fstack-usage ile aynı çerçeve)."""
    o = drone_altitude()
    opts = mc.MpcCodegenOptions(real=real, mcu="cortex-m4f", opt="-Os")
    rows = {r.compiler: r for r in mc.footprint(o, opts, ("gcc", "clang", "iar"))}
    pl = mc.plan(o, opts)
    size_tool = ARM_GCC.sibling("arm-none-eabi-size") if ARM_GCC else None
    # küçük sabit dizileri (ör. sıfır ofsetler) derleyici sabit katlamayla kaldırabilir; Ac değişken indisle okunur
    ac_bytes = pl.array("Ac").size * (4 if real == "float" else 8)
    for comp, tool in (("gcc", ARM_GCC), ("clang", CLANG)):
        if tool is None:
            continue
        r = rows[comp]
        assert r.measured and "ölçüldü" in r.method
        assert r.ram_state == mc.state_size(pl, real)
        raw = mc._measure(o, opts, MCU_PROFILES["cortex-m4f"], comp)
        assert raw["bss"] == 0 and raw["data"] == 0                      # yalnız const veri + kod
        assert raw["rodata"] >= ac_bytes                                 # sabit matrisler flash'ta
        if size_tool:
            out = tc._run([size_tool, "-B", raw["object"]]).stdout.splitlines()[1].split()
            assert int(out[0]) + int(out[1]) == r.flash_code              # bağımsız ölçüm
        if ARM_GCC is not None:
            assert r.flash_with_libm >= r.flash_code
            assert r.stack >= r.stack_own > 0
            if comp == "gcc":
                assert raw["frame_step"] == raw["su_step"]
    if ARM_GCC is not None:
        assert not rows["iar"].measured and rows["iar"].flash_code == round(rows["gcc"].flash_code * 0.9)


def test_ayak_izi_derleyicisiz_tahmin(monkeypatch):
    monkeypatch.setattr(tc, "find_arm_gcc", lambda: None)
    monkeypatch.setattr(tc, "find_clang", lambda: None)
    o = case2()
    opts = mc.MpcCodegenOptions(real="double", mcu="cortex-m3")
    rows = mc.footprint(o, opts, ("gcc", "clang", "iar"))
    assert len(rows) == 3
    for r in rows:
        assert not r.measured and "tahmin" in r.method and r.flash_code > 0
        assert r.ram_state == mc.state_size(mc.plan(o, opts), "double")


# ============================================================================ maliyet
@needs_cxx
def test_maliyet_ve_en_kotu_durum_siniri():
    """İşlem sayıları ölçülür; analitik sınır (step_ops_bound) her ölçülen adımın HER işlem türünde üstündedir
    (yineleme sınırına takılan adımlar dahil); kısıtsız tasarımda sınır tam eşittir. Yük = µs × fs / 10⁴."""
    cases = [(drone_altitude(), None, 120), (case3(), None, 40), (case4(), None, 40), (case1(), None, 30)]
    o_lim = case5()
    o_lim.Optimizer.MaxIterations = 3
    cases.append((o_lim, None, 60))
    o_cold = case7()
    o_cold.Optimizer.UseWarmStart = False
    cases.append((o_cold, None, 30))
    for o, opt, T in cases:
        pl = mc.plan(o, DOUBLE)
        r, v = mc._cost_scenario(o, o.data, T)
        run = mc.run_closed_loop(o, DOUBLE, T, r, v, opt, count_ops=True)
        for t in range(T):
            st = int(run.status[t])
            bound = mc.step_ops_bound(pl, st if st > 0 else pl.max_iter)
            assert np.all(run.ops[t] <= bound), (pl.nc, t, st, dict(zip(OPS, run.ops[t] - bound)))
            if pl.nc == 0:
                assert np.array_equal(run.ops[t], bound)
    o = drone_altitude()
    opts = mc.MpcCodegenOptions(real="float", mcu="cortex-m4f", clock_mhz=168.0)
    c = mc.cost(o, opts)[0]
    fs = 1.0 / o.data.Ts
    assert 0 < c.cycles_mean <= c.cycles_max <= c.cycles_wc and c.cycles_iter > 0
    assert c.us_mean == pytest.approx(c.cycles_mean / 168.0) and c.load_wc == pytest.approx(c.us_wc * fs / 1e4)
    assert c.max_iter == o.data.max_iter and 1 <= c.iters_max <= c.max_iter
    assert c.iter_budget == int((o.data.Ts * 168e6 - mc.step_ops_bound(mc.plan(o, opts), 0) @
                                 mc._op_weights(MCU_PROFILES["cortex-m4f"], "float")) // c.cycles_iter)
    assert sum(c.status_counts.values()) == max(mc.COST_MIN_STEPS, 4 * o.data.p)
    soft = mc.cost(o, mc.MpcCodegenOptions(real="double", mcu="cortex-m4f", clock_mhz=168.0))[0]
    assert soft.cycles_mean > 10 * c.cycles_mean                     # SP-FPU'da double yazılım kayan nokta


def test_kullanim_kilavuzu():
    o = drone_altitude()
    opts = mc.MpcCodegenOptions(prefix="irtifa")
    md = mc.usage_markdown(o, opts, name="Drone irtifa")
    for s in ("# MPC denetleyicisi „Drone irtifa\"", "`irtifa.h`",
              "`irtifa_step(&s, ym, ref, ref_rows, md, md_rows, u)`", "tam Ts = 0.02 s", "50 Hz", "IRTIFA_MAX_ITER",
              "ScaleFactor", "Kısıtlar", "IRTIFA_STATUS_INFEASIBLE", "yumuşak", "darbesiz", "-ffast-math",
              "ölçümsüz üretildi"):
        assert s in md, s
    fp = [mc.FootprintRow("x", "gcc", 4000, 4300, 640, 220, True, "ölçüldü (deneme)", 200)]
    # tutarlı veri: Ts 0.02 s × 168 MHz = 3.36e6 çevrim; sabit kısım 3.035e6 sığıyor, 344 adımlık en kötü durum değil
    cost = [mc.MpcCostRow("senaryo", {op: 1.0 for op in OPS}, 5000, 20000, 11979000, 26000, 30, 120, 71304, 0.1,
                          0.6, 356.5, 1.5, 9, 344, 12, {"çözüldü": 120}, "yöntem")]
    md2 = mc.usage_markdown(o, opts, footprint_rows=fp, cost_rows=cost)
    assert "En kötü durum" in md2 and "**12** yineleme" in md2 and "Ts'yi aşıyor" in md2 and "ölçümsüz" not in md2
    assert "IRTIFA_MAX_ITER=12" in md2                       # bütçeye sığan adım sayısı önerilir
    assert mc.RAM_PERCENT_NOTE in md2


def test_butce_tavsiyesi_dogrudan_yuk_karsilastirmasi():
    """Denetim bulgusu 5: kısıtsız tasarımda yineleme yok — yük doğrudan karşılaştırılır; kısıtlıda sabit kısım
    sığmıyorsa MAX_ITER düşürmek önerilmez."""
    def row(wc, it, max_iter, budget, load_wc, iters_max=5):
        return mc.MpcCostRow("s", {op: 0.0 for op in OPS}, 1, 1, wc, it, 1, 1, 1, 1, 1, load_wc, 1, iters_max,
                             max_iter, budget, {}, "m")
    lvl, txt = mc.budget_advice(row(32000, 0.0, 0, 0, 1971.0), "X")
    assert lvl == "critical" and "Kısıtsız" in txt and "MAX_ITER" not in txt and "hızlı" in txt
    assert mc.budget_advice(row(1000, 0.0, 0, 0, 40.0), "X")[0] == "good"
    lvl, txt = mc.budget_advice(row(2.0e6, 1000, 120, 0, 200.0), "X")       # sabit kısım 1.88e6: yük %188 > 100
    assert lvl == "critical" and "-DX_MAX_ITER" not in txt and "çözüm değildir" in txt
    lvl, txt = mc.budget_advice(row(2.0e6, 15000, 120, 40, 200.0), "X")     # sabit kısım 2e5: %20
    assert lvl == "warning" and "-DX_MAX_ITER=40" in txt
    o = case1()                                                              # kısıtsız KULLANIM: QP makroları yok
    md = mc.usage_markdown(o, mc.MpcCodegenOptions(prefix="kz"))
    for s in ("KZ_MAX_ITER", "KZ_TOL", "KZ_DEP_TOL", "KZ_STATUS_MAXITER", "KZ_STATUS_INFEASIBLE"):
        assert s not in md, s
    assert "KZ_STATUS_BAD_INPUT" in md


def test_float_sabit_araligi_ve_onek_cakismasi():
    """Denetim bulguları 3 ve 11: |x| < FLT_MIN sabitler float'ta 0 (SPT_CT), sınırlarda |x| > FLT_MAX ±FLT_MAX
    (SPT_CH); matriste |x| > FLT_MAX float'ta açık hata; <stdint.h> ile çakışan önek reddedilir."""
    o = MPC(([[-3000.0, 0], [0, -1.0]], [[3000.0], [1.0]], [[1.0, 1.0]], [[0.0]]), 0.1, 10, 2)
    o.MV[0].Min, o.MV[0].Max = -1, 1
    c_txt = next(f.text for f in mc.generate(o, mc.MpcCodegenOptions(prefix="hz")) if f.name == "hz.c")
    assert "HZ_CT(" in c_txt and not re.search(r"HZ_C\(-?\d[\d.]*e-(?:[4-9]\d|\d{3})\)", c_txt)
    o2 = MPC(([[0.9]], [[1.0]], [[1.0]], [[0.0]], 0.1), 0.1)
    o2.MV[0].Max = 1e40
    c2 = next(f.text for f in mc.generate(o2, mc.MpcCodegenOptions(prefix="bk")) if f.name == "bk.c")
    assert "BK_CH(1e+40)" in c2
    assert mc._lit(-1e40, True) == "(-SPT_CH(1e+40))" and mc._lit(5e-131) == "(SPT_CT(5e-131))"
    assert mc._lit(-0.25) == "SPT_C(-0.25)"
    o3 = MPC(([[0.9]], [[1e39]], [[1.0]], [[0.0]], 0.1), 0.1)               # ScaleFactor 1e-39 → 1/ölçek > FLT_MAX
    o3.MV[0].ScaleFactor = 1e-39
    o3.MV[0].Max = 1
    with pytest.raises(ValueError, match="float aralığını"):
        mc.generate(o3, mc.MpcCodegenOptions(real="float"))
    c3 = next(f.text for f in mc.generate(o3, mc.MpcCodegenOptions(real="double", prefix="dv")) if f.name == "dv.c")
    assert "#error" in c3 and "DV_USE_DOUBLE" in c3
    for bad in ("uint8", "int16", "intmax", "size", "math", "stdint"):
        with pytest.raises(ValueError, match="çakışır"):
            mc.check_options(mc.MpcCodegenOptions(prefix=bad))
    assert mc.default_prefix("Time") == "time_mpc"


@needs_cross
def test_float_sabit_araligi_werror_derlenir():
    """Hızlı kutup (τ = 1/3000 s, Ts = 0.1 s → ~5e-131) ve MV.Max = 1e40: float -Werror derlemesi (gcc / clang)."""
    o = MPC(([[-3000.0, 0], [0, -1.0]], [[3000.0], [1.0]], [[1.0, 1.0]], [[0.0]]), 0.1, 10, 2)
    o.MV[0].Min, o.MV[0].Max = -1, 1e40
    files = {f.name: f.text for f in mc.generate(o, mc.MpcCodegenOptions(real="float", prefix="hizli"))}
    for comp in ("gcc", "clang"):
        if tc.cross_flags(MCU_PROFILES["cortex-m4f"], comp) is None:
            continue
        b = tc.compile_cross(files, ["hizli.c"], MCU_PROFILES["cortex-m4f"], comp, "-O2",
                             defines=["HIZLI_USE_DOUBLE=0"], extra=["-Wdouble-promotion"], stack_usage=False)
        assert b.ok and not b.log.strip(), (comp, b.log[:600])


@needs_cc
@pytest.mark.parametrize("so", [0, 1])
def test_c_yineleme_siniri_matlab_drone(so):
    """Denetim bulgusu 1 (C şablonu): MaxIterations = 3, drone ekseni — üretilen C (double) MATLAB sim ile ~1e-13;
    MaxIterations = 2'de bildirilen yineleme ≤ 3 (adım ≤ MaxIterations)."""
    from tests.test_control_mpc_denetim import DENETIM_SCRIPT
    try:
        ref = fixture("mpc_denetim_cases", DENETIM_SCRIPT)
    except OracleUnavailable as e:  # pragma: no cover
        pytest.skip(str(e))
    o = case5()
    o.Optimizer.MaxIterations = 3
    o.Optimizer.UseSuboptimalSolution = bool(so)
    res = mc.run_closed_loop(o, DOUBLE, 60, [2, 0])
    _close(res.u, ref[f"dr3_{so}"]["u"], TOL_ML, "u")
    _close(res.y, ref[f"dr3_{so}"]["y"], TOL_ML, "y")
    o.Optimizer.MaxIterations = 2
    res2 = mc.run_closed_loop(o, DOUBLE, 60, [2, 0])
    assert res2.status.max() <= 3 and np.array_equal(res2.status, sim(o, 60, [2, 0]).status.astype(int))


def test_kotu_kosullu_hessian_float_uyarisi():
    """Denetim bulgusu 4: özdeş iki MV, MV ve oran ağırlığı 0 (BadH = 1, koşul ~6e9): float'ta açık uyarı ve
    KULLANIM notu; iyi koşullu tasarımda uyarı yok."""
    o = MPC(([[0.9]], [[1, 1]], [[1]], [[0, 0]], 0.1), 0.1, 10, 3)
    o.Weights.ManipulatedVariables = [0, 0]
    o.Weights.ManipulatedVariablesRate = [0, 0]
    o.MV[0].Min, o.MV[0].Max, o.MV[1].Min, o.MV[1].Max = -1, 1, -1, 1
    with pytest.warns(Warning):
        o.data
    w = mc.numeric_warnings(o, FLOAT)
    assert len(w) == 2 and "BadH" in w[0] and "double" in w[0]
    with pytest.warns(Warning, match="koşul sayısı"):
        files = mc.generate(o, mc.MpcCodegenOptions(real="float", prefix="tk"))
    assert "Sayısal uyarı" in next(f.text for f in files if f.name.endswith(".md"))
    assert mc.numeric_warnings(drone_altitude(), FLOAT) == []


# ============================================================================ arayüz
@needs_cc
def test_mpc_kod_diyalogu():
    import os
    import time
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from spt.gui.dialogs.mpc_code_dialog import open_mpc_code_dialog
    app = QApplication.instance() or QApplication([])
    o = case2()
    dlg = open_mpc_code_dialog(None, o, "Sıcaklık MPC")
    assert {f.name for f in dlg.files} == {"sicaklik_mpc.h", "sicaklik_mpc.c", "sicaklik_mpc_ornek.c",
                                           "KULLANIM_sicaklik_mpc.md"}
    assert not dlg.md_preview.isEnabled() and "5 Hz" in dlg.rate.text()
    t0 = time.time()
    while not dlg.generate_button.isEnabled() and time.time() - t0 < 300:
        app.processEvents()
        time.sleep(0.05)
    assert dlg.generate_button.isEnabled()
    text = dlg.summary.toPlainText()
    assert "Bellek" in text and "CPU yükü" in text and "En kötü durum" in text, text[:400]
    kul = next(f.text for f in dlg.files if f.name.endswith(".md"))
    assert "Ölçülen ayak izi" in kul and "ölçümsüz" not in kul
    o.MV[0].Max = 0.5                                               # kopya alındı: ölçümü etkilemez
    dlg.ref_preview.setChecked(True)
    dlg.prefix.setText("__x")
    dlg.refresh()
    assert "Önek" in dlg.status.text() and not dlg.files
    dlg.prefix.setText("isi")
    dlg.refresh()
    assert "ISI_REF_PREVIEW 1" in next(f.text for f in dlg.files if f.name == "isi.h")
    dlg.shutdown()
    dlg.close()
