"""Klasik tasarım (spt.control.classical) — MATLAB R2025b Control System Toolbox referanslı testler.

Referans: `tests/data/matlab/control_design_classical.json`.
- pid / pidstd / pid2 / pidstd2 → tf: tüm IFormula / DFormula birleşimleri, hata durumları — **birebir** katsayılar.
- paralel ↔ standart dönüşüm ve hataları.
- pidtune: MATLAB'in algoritması yayımlanmamış. Verilen wc ve PM = 60° ile MATLAB'in de bizim de hedefleri
  (|L(jwc)| = 1, ∠L = −180° + PM) sağladığı ölçülür; parametre farkları raporlanır (tek çözümlü türlerde — P, I, PI,
  PD — ve MATLAB'in 'balanced' PID'inde (Ti = 4·Td) parametreler de eşleşir).
- rlocfind: eğri üzerindeki noktada tam kazanç; dışındaki noktada kutbumuz p'ye MATLAB'inkinden uzak değil.
- öncül / ardıl, çentik, sönüm için kazanç: tanımlar doğrudan frekans / kutup hesabıyla doğrulanır.
"""
from __future__ import annotations

import cmath
import math
import warnings

import numpy as np
import pytest

from spt.control import classical as cl
from spt.control import lti
from spt.control.errors import ControlError, ControlWarning
from tools.matlab_oracle import OracleUnavailable, fixture

SCRIPT = r"""
F = {'ForwardEuler', 'BackwardEuler', 'Trapezoidal'};
P = [1 2 3 0.5; 1 2 0 0.5; 1 0 3 0.5; 1 2 3 0; 0 2 0 0; 2 0 0 0; 0 0 3 0.2; -1.5 -0.7 -0.4 0.3; 0 0 0 0;
     1 1 0.5 0.04; 1 1 0.5 0.05; 0.3 0 0.2 0];
k = 0;
for i = 1:size(P, 1)
  for Ts = [0 0.1]
    for a = 1:3
      for b = 1:3
        if Ts == 0 && (a > 1 || b > 1), continue; end
        k = k + 1;
        r = struct('p', P(i, :), 'Ts', Ts, 'IF', F{a}, 'DF', F{b}, 'n', [], 'd', [], 'err', '');
        try
          if Ts == 0
            C = pid(P(i, 1), P(i, 2), P(i, 3), P(i, 4));
          else
            C = pid(P(i, 1), P(i, 2), P(i, 3), P(i, 4), Ts, 'IFormula', F{a}, 'DFormula', F{b});
          end
          [n, d] = tfdata(tf(C), 'v'); r.n = n; r.d = d;
        catch e
          r.err = e.message;
        end
        out.pid{k} = r;
      end
    end
  end
end
Q = [1 2 3 0.5; -1 -2 -3 0.5; 2 0 1 0.1; 2 1 0 0; 1 2 3 0; 0 2 0 0; -1 2 0 0; 1 -2 1 0.1; 0 0 0 0];
for i = 1:size(Q, 1)
  r = struct('p', Q(i, :), 'std', [], 'err', '');
  try
    S = pidstd(pid(Q(i, 1), Q(i, 2), Q(i, 3), Q(i, 4))); r.std = [S.Kp S.Ti S.Td S.N];
  catch e
    r.err = e.message;
  end
  out.p2s{i} = r;
end
S2 = [2 3 0.5 10; 2 Inf 0.5 10; 2 3 0 10; 2 3 0.5 Inf; -1 0.5 3 6; 1 Inf 0 Inf];
for i = 1:size(S2, 1)
  r = struct('s', S2(i, :), 'par', [], 'n', [], 'd', [], 'err', '');
  try
    C = pid(pidstd(S2(i, 1), S2(i, 2), S2(i, 3), S2(i, 4))); r.par = [C.Kp C.Ki C.Kd C.Tf];
    Cd = pidstd(S2(i, 1), S2(i, 2), S2(i, 3), S2(i, 4), 0.05, 'IFormula', 'Trapezoidal', 'DFormula', 'BackwardEuler');
    [n, d] = tfdata(tf(Cd), 'v'); r.n = n; r.d = d;
  catch e
    r.err = e.message;
  end
  out.s2p{i} = r;
end
C2 = pid2(1, 2, 3, 0.5, 0.6, 0.2); [n, d] = tfdata(tf(C2));
out.p2a = struct('n1', n{1}, 'd1', d{1}, 'n2', n{2}, 'd2', d{2});
C2 = pid2(1, 2, 3, 0.5, 0.6, 0.2, 0.1, 'IFormula', 'Trapezoidal', 'DFormula', 'BackwardEuler');
[n, d] = tfdata(tf(C2)); out.p2b = struct('n1', n{1}, 'd1', d{1}, 'n2', n{2}, 'd2', d{2});
C2 = pidstd2(2, 3, 0.5, 10, 0.7, 0.3); [n, d] = tfdata(tf(C2));
out.p2c = struct('n1', n{1}, 'd1', d{1}, 'n2', n{2}, 'd2', d{2});
C2 = pidstd2(2, 3, 0.5, 10, 0.7, 0.3, 0.05, 'IFormula', 'BackwardEuler', 'DFormula', 'Trapezoidal');
[n, d] = tfdata(tf(C2)); out.p2d = struct('n1', n{1}, 'd1', d{1}, 'n2', n{2}, 'd2', d{2});
G = {tf(1, [1 3 3 1]), tf(10, conv([1 0], conv([1 2], [1 5]))), tf(1, [2 1], 'InputDelay', 0.3), ...
     c2d(tf(1, [1 3 3 1]), 0.1), c2d(tf(2, [1 3 2]), 0.05)};
G{5}.InputDelay = 2;
T = {'P', 'I', 'PI', 'PD', 'PDF', 'PID', 'PIDF'};
opt = pidtuneOptions('PhaseMargin', 60);
for gi = 1:numel(G)
  [n, d] = tfdata(G{gi}, 'v');
  pl = struct('n', n, 'd', d, 'Ts', G{gi}.Ts, 'delay', G{gi}.InputDelay);
  for ti = 1:numel(T)
    r = struct('type', T{ti}, 'p', [], 'info', [], 'IF', '', 'DF', '', 'p60', [], 'info60', [], 'Pm60', [], ...
               'Wcp60', [], 'err', '', 'err60', '', 'errm', '');
    try
      [C, info] = pidtune(G{gi}, T{ti});
      r.p = [C.Kp C.Ki C.Kd C.Tf]; r.info = info; r.IF = C.IFormula; r.DF = C.DFormula;
      try
        [C2, info2] = pidtune(G{gi}, T{ti}, info.CrossoverFrequency, opt);
        r.p60 = [C2.Kp C2.Ki C2.Kd C2.Tf]; r.info60 = info2;
        try
          [~, Pm, ~, Wcp] = margin(C2 * G{gi}); r.Pm60 = Pm; r.Wcp60 = Wcp;
        catch e3
          r.errm = e3.message;
        end
      catch e2
        r.err60 = e2.message;
      end
    catch e
      r.err = e.message;
    end
    pl.cases{ti} = r;
  end
  out.tune{gi} = pl;
end
G = tf([1 3], [1 4 5 0]); p = [-1+2i, -0.5, -2+0.3i, -4];
[K, Pl] = rlocfind(G, p); out.rl.K = K; out.rl.p = spt_c(p);
Gd = c2d(G, 0.1); pd = [0.9+0.1i, 0.95, 0.7];
[K, Pl] = rlocfind(Gd, pd); out.rld.K = K; out.rld.p = spt_c(pd);
[n, d] = tfdata(Gd, 'v'); out.rld.n = n; out.rld.d = d;
"""


@pytest.fixture(scope="module")
def ref():
    try:
        return fixture("control_design_classical", SCRIPT)
    except OracleUnavailable as exc:  # pragma: no cover - MATLAB yok ve kayıt yok
        pytest.skip(str(exc))


# ---------------------------------------------------------------------------------------------------------------
# yardımcılar
# ---------------------------------------------------------------------------------------------------------------
def _v(x) -> np.ndarray:
    return np.atleast_1d(np.asarray(x if x is not None else [], dtype=float)).ravel()


def _strip(p) -> np.ndarray:
    p = _v(p)
    nz = np.flatnonzero(p != 0)
    return p[nz[0]:] if nz.size else np.zeros(1)


def _close(a, b, rtol=1e-12):
    a, b = _v(a), _v(b)
    assert a.shape == b.shape, (a, b)
    assert np.array_equal(np.isinf(a), np.isinf(b)) and np.array_equal(a[np.isinf(a)], b[np.isinf(b)]), (a, b)
    a, b = a[np.isfinite(b)], b[np.isfinite(b)]
    scale = max(1.0, float(np.max(np.abs(b)))) if b.size else 1.0
    assert not b.size or np.max(np.abs(a - b)) <= rtol * scale, (a, b)


def _cplx(x) -> np.ndarray:
    a = np.asarray(x, dtype=float).reshape(-1, 2)
    return a[:, 0] + 1j * a[:, 1]


def _plant(pl):
    return lti.tf(_v(pl["n"]), _v(pl["d"]), float(pl["Ts"]), input_delay=float(pl["delay"]))


def _loop(G, C, w):
    """L(jw) = C·G (gecikme dahil) — doğrudan frekans değerlendirmesiyle."""
    Gs = cl._Siso(G)
    x = Gs.point(np.array([w]))
    return complex((C.evalfr(x) * Gs.at(x))[0])


def _targets(L: complex, PM: float) -> tuple[float, float]:
    """(||L| − 1|, |faz payı − PM| derece)."""
    pm = (math.degrees(cmath.phase(L)) + 180.0 + 180.0) % 360.0 - 180.0
    return abs(abs(L) - 1.0), abs(pm - PM)


# ---------------------------------------------------------------------------------------------------------------
# PID nesneleri — MATLAB tf(pid(...)) ile birebir
# ---------------------------------------------------------------------------------------------------------------
def test_pid_tf_matches_matlab(ref):
    n_ok = n_err = 0
    for r in ref["pid"]:
        Kp, Ki, Kd, Tf = _v(r["p"])
        Ts = float(r["Ts"])
        kw = {} if Ts == 0 else dict(IFormula=r["IF"], DFormula=r["DF"])
        if r["err"]:
            with pytest.raises(ControlError):
                cl.pid(Kp, Ki, Kd, Tf, Ts, **kw)
            n_err += 1
            continue
        C = cl.pid(Kp, Ki, Kd, Tf, Ts, **kw)
        _close(C.num[0][0], _strip(r["n"]))
        _close(C.den[0][0], _strip(r["d"]))
        n_ok += 1
    assert n_ok >= 80 and n_err >= 4


def test_pidstd_conversions_match_matlab(ref):
    for r in ref["p2s"]:
        Kp, Ki, Kd, Tf = _v(r["p"])
        if r["err"]:
            with pytest.raises(ControlError):
                cl.pidstd(cl.pid(Kp, Ki, Kd, Tf))
            continue
        S = cl.pidstd(cl.pid(Kp, Ki, Kd, Tf))
        _close(S.standard(), _v(r["std"]))
    for r in ref["s2p"]:
        Kp, Ti, Td, N = _v(r["s"])
        P = cl.pid(cl.pidstd(Kp, Ti, Td, N))
        _close(P.parallel(), _v(r["par"]))
        Cd = cl.pidstd(Kp, Ti, Td, N, Ts=0.05, IFormula="Trapezoidal", DFormula="BackwardEuler")
        _close(Cd.num[0][0], _strip(r["n"]))
        _close(Cd.den[0][0], _strip(r["d"]))
        _close(Cd.standard(), [Kp, Ti, Td, N])


def test_pid2_matches_matlab(ref):
    cases = {"p2a": cl.pid2(1, 2, 3, 0.5, 0.6, 0.2),
             "p2b": cl.pid2(1, 2, 3, 0.5, 0.6, 0.2, 0.1, IFormula="Trapezoidal", DFormula="BackwardEuler"),
             "p2c": cl.pidstd2(2, 3, 0.5, 10, 0.7, 0.3),
             "p2d": cl.pidstd2(2, 3, 0.5, 10, 0.7, 0.3, 0.05, IFormula="BackwardEuler", DFormula="Trapezoidal")}
    for key, C in cases.items():
        m = ref[key]
        assert C.shape == (1, 2)
        _close(C.num[0][0], _strip(m["n1"]))
        _close(C.den[0][0], _strip(m["d1"]))
        _close(C.num[0][1], _strip(m["n2"]))
        _close(C.den[0][1], _strip(m["d2"]))


def test_pid_object_behaviour():
    C = cl.pid(1, 2, 3, 0.5)
    assert isinstance(C, lti.TransferFunction)
    s = np.array([0.3j, 2.0 + 1.0j])
    tfv = np.polyval(C.num[0][0], s) / np.polyval(C.den[0][0], s)
    _close(C.evalfr(s).real, tfv.real)
    _close(C.evalfr(s).imag, tfv.imag)
    C.Kp = 4.0
    assert C.parallel() == (4.0, 2.0, 3.0, 0.5)
    _close(C.num[0][0], cl.pid(4, 2, 3, 0.5).num[0][0])
    T = C * lti.tf([1], [1, 1])                      # LTI aritmetiği
    assert isinstance(T, lti.LTI) and not isinstance(T, cl.PID)
    T2 = lti.tf([1], [1, 1]) * C + cl.pidstd(2, 3) - 1
    s0 = 0.7j
    v = (np.polyval(C.num[0][0], s0) / np.polyval(C.den[0][0], s0)) / (s0 + 1) + 2 * (1 + 1 / (3 * s0)) - 1
    got = np.polyval(T2.num[0][0], s0) / np.polyval(T2.den[0][0], s0)
    assert abs(got - v) < 1e-12 * abs(v)
    S = cl.pidstd(2, 3, 0.5, 10)
    S.Ti = 6.0
    assert S.standard() == (2.0, 6.0, 0.5, 10.0)
    _close(S.parallel(), [2.0, 1 / 3, 1.0, 0.05])
    with pytest.raises(ControlError):
        cl.pid(1, 2, 3).to_ss()                      # Tf = 0: uygun değil
    sys = cl.pid(1, 2, 3, 0.5, 0.1).to_ss()
    assert sys.A.shape == (2, 2)
    with pytest.raises(ControlError):
        cl.pid(1, 1, 1, 0.05, 0.1)                   # ileri Euler: Ts < 2·Tf olmalı
    with pytest.raises(ControlError):
        cl.pid(1, 1, 1, 0, 0.1, DFormula="Trapezoidal")
    with pytest.raises(ControlError):
        cl.pid(1, 1, 1, -0.1)
    with pytest.raises(ControlError):
        cl.pidstd(1, -2)
    assert "Kp" in str(C) and "Sürekli" in str(C)


# ---------------------------------------------------------------------------------------------------------------
# pidtune
# ---------------------------------------------------------------------------------------------------------------
UNIQUE = ("P", "I", "PI", "PD")


def _tune_rows(ref):
    for gi, pl in enumerate(ref["tune"]):
        G = _plant(pl)
        cases = pl["cases"] if isinstance(pl["cases"], list) else [pl["cases"]]
        for r in cases:
            yield gi, G, r


def test_pidtune_given_wc_matlab_and_ours_hit_targets(ref):
    """wc = MATLAB'in seçtiği geçiş frekansı verildiğinde:
    1. MATLAB'in denetleyicisi wc'de |L| = 1 ve MATLAB'in bildirdiği faz payını (PM_M) gerçekten sağlar (bizim
       bağımsız frekans hesabımızla ölçülür). PI / PD / PDF'de PM_M = 60° tam; PID / PIDF'de MATLAB çoğu kez 60°'tan
       büyük pay bırakır (hedefi alt sınır olarak kullanır).
    2. Bizim tasarımımız hem PM = 60° hem PM = PM_M için hedefleri tam sağlar.
    3. Tek çözümlü türlerde (P, I, PI, PD) ve MATLAB'in PID'i Ti = 4·Td yapısındaysa parametreler MATLAB'inkiyle
       aynıdır (aynı denklemler → aynı çözüm); diğerlerinde fark raporlanır."""
    rows = []
    n_same = 0
    for gi, G, r in _tune_rows(ref):
        if r["err"] or r["err60"] or r["p60"] is None or len(_v(r["p60"])) == 0:
            continue
        typ = r["type"]
        wc = float(r["info"]["CrossoverFrequency"])
        pm_m = r["info60"]["PhaseMargin"]
        IF = r["IF"] or "ForwardEuler"
        DF = r["DF"] or "ForwardEuler"
        Ts = float(G.Ts)
        pmat = np.asarray(_v(r["p60"]))
        Cm = cl.pid(*pmat, Ts, IFormula=IF, DFormula=DF)
        Lm = _loop(G, Cm, wc)
        assert abs(abs(Lm) - 1) < 1e-6, (gi, typ, abs(Lm))
        if pm_m is not None and np.isfinite(pm_m):
            assert _targets(Lm, pm_m)[1] < 1e-6, (gi, typ)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ControlWarning)
            C, info = cl.pidtune(G, typ, wc, PhaseMargin=60.0, IFormula=IF, DFormula=DF)
        eo = _targets(_loop(G, C, wc), 60.0)
        assert eo[0] < 1e-9 and (typ in ("P", "I") or eo[1] < 1e-7), (gi, typ, eo)
        po = np.asarray(C.parallel())
        rel60 = float(np.max(np.abs(po - pmat)) / np.max(np.abs(pmat)))
        relm = math.nan
        if typ not in ("P", "I") and pm_m is not None and np.isfinite(pm_m):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ControlWarning)
                C2, info2 = cl.pidtune(G, typ, wc, PhaseMargin=pm_m, IFormula=IF, DFormula=DF)
            e2 = _targets(_loop(G, C2, wc), pm_m)
            assert e2[0] < 1e-9 and e2[1] < 1e-7
            relm = float(np.max(np.abs(np.asarray(C2.parallel()) - pmat)) / np.max(np.abs(pmat)))
            Kp, Ki, Kd = pmat[:3]
            ti_td = (Kp / Ki) / (Kd / Kp) if (Ki and Kd and Kp) else math.nan
            if typ in UNIQUE or (typ == "PID" and abs(ti_td - 4.0) < 1e-6):
                assert relm < 1e-6, (gi, typ, C2.parallel(), pmat)
                n_same += 1
            assert info2["Stable"] == bool(r["info60"]["Stable"]), (gi, typ)
        elif typ in ("P", "I"):
            assert rel60 < 1e-9, (gi, typ, po, pmat)       # |L| = 1 ve bitki işaretli kazanç: tek çözüm
            n_same += 1
        rows.append((gi, typ, wc, pm_m, rel60, relm, bool(info["Stable"]), bool(r["info60"]["Stable"])))
    assert len(rows) >= 30 and n_same >= 20
    print("\n plant tür   wc        PM(MATLAB)  fark(PM=60)  fark(PM=PM_M)  kararlı(biz, PM=60 / MATLAB)")
    for gi, typ, wc, pm_m, rel60, relm, so, sm in rows:
        print(f"  G{gi + 1}  {typ:5s} {wc:8.4f}  {pm_m if pm_m is not None else float('nan'):9.4f}  {rel60:10.2e}  "
              f"{relm:12.2e}   {so}/{sm}")


def test_pidtune_default_wc_hits_targets(ref):
    """wc verilmeden: bizim kuralımız hedefleri tam sağlar ve kapalı çevrim kararlıdır (MATLAB'in seçtiği wc ile
    karşılaştırma raporlanır)."""
    print("\n plant tür   wc(biz)   wc(MATLAB)  PM(MATLAB)")
    n = 0
    for gi, G, r in _tune_rows(ref):
        typ = r["type"]
        try:
            C, info = cl.pidtune(G, typ)
        except ControlError:
            assert r["err"] or typ in ("P", "I"), (gi, typ)
            continue
        assert info["Stable"]
        L = _loop(G, C, info["CrossoverFrequency"])
        e = _targets(L, 60.0)
        assert e[0] < 1e-9 and e[1] < 1e-7, (gi, typ, e)
        mi = r["info"] or {}
        print(f"  G{gi + 1}  {typ:5s} {info['CrossoverFrequency']:8.4f}  "
              f"{mi.get('CrossoverFrequency', float('nan')):9.4f}  {mi.get('PhaseMargin', float('nan')):8.3f}")
        n += 1
    assert n >= 25


def test_pidtune_options_and_errors():
    G = lti.tf([1], [1, 3, 3, 1])
    Cb, _ = cl.pidtune(G, "PID", 0.8, PhaseMargin=60)
    Cr, _ = cl.pidtune(G, "PID", 0.8, PhaseMargin=60, DesignFocus="reference-tracking")
    Cd, _ = cl.pidtune(G, "PID", 0.8, PhaseMargin=60, DesignFocus="disturbance-rejection")
    for C, alpha in ((Cb, 4.0), (Cr, 8.0), (Cd, 2.0)):
        Kp, Ti, Td, N = C.standard()
        assert abs(Ti / Td - alpha) < 1e-9
        assert _targets(_loop(G, C, 0.8), 60.0)[1] < 1e-7
    C, info = cl.pidtune(G, "PIDF", 0.8, PhaseMargin=45)
    assert abs(C.Tf - 1 / (10 * 0.8)) < 1e-12 and abs(info["PhaseMargin"] - 45) < 1e-7
    S, _ = cl.pidtune(G, "PI", 0.5, form="standard")
    assert isinstance(S, cl.PIDStd)
    with pytest.warns(ControlWarning):      # 3 rad/s'de PI ancak ters işaretli kazançla tutturur → kararsız
        _, info = cl.pidtune(G, "PI", 3.0)
    assert not info["Stable"]
    with pytest.raises(ControlError):
        cl.pidtune(G, "PI", 1.19)           # 1.19 rad/s'de +30° faz öncülü gerekir: PI kazançları zıt işaretli
    with pytest.raises(ControlError):
        cl.pidtune(G, "PIX")
    with pytest.raises(ControlError):
        cl.pidtune(G, "PI", PhaseMargin=95)
    Gd = lti.tf([0.1], [1, -0.9], 0.1)
    with pytest.raises(ControlError):
        cl.pidtune(Gd, "PI", 40.0)          # Nyquist frekansının üstü


def test_pidtune_delay_plant_stability_by_nyquist():
    """Gecikmeli sürekli bitkide kararlılık Nyquist sarım sayısıyla: P kazancının analitik sınırı."""
    from scipy.optimize import brentq
    G = lti.tf([1], [1, 1], input_delay=0.5)
    w = brentq(lambda x: -math.atan(x) - 0.5 * x + math.pi, 0.1, 10)
    Ku = math.sqrt(1 + w * w)
    Gs = cl._Siso(G)
    assert cl._cl_stable(Gs, np.array([0.99 * Ku]), np.array([1.0]))
    assert not cl._cl_stable(Gs, np.array([1.01 * Ku]), np.array([1.0]))
    C, info = cl.pidtune(G, "PI")
    assert info["Stable"] and abs(info["PhaseMargin"] - 60) < 1e-7


# ---------------------------------------------------------------------------------------------------------------
# kök yer eğrisi
# ---------------------------------------------------------------------------------------------------------------
def test_rlocfind(ref):
    plants = (("rl", lti.tf([1, 3], [1, 4, 5, 0])),
              ("rld", lti.tf(_v(ref["rld"]["n"]), _v(ref["rld"]["d"]), 0.1)))
    for key, G in plants:
        pts = _cplx(ref[key]["p"])
        Km = _v(ref[key]["K"])
        K, P = cl.rlocfind(G, pts)
        Gs = cl._Siso(G)
        for j, x in enumerate(pts):
            d_ours = np.min(np.abs(P[:, j] - x))
            d_m = np.min(np.abs(cl._cl_poles(Gs, Km[j]) - x))
            assert d_ours <= d_m + 1e-9, (key, x, K[j], Km[j])
    K, P = cl.rlocfind(lti.tf([1, 3], [1, 4, 5, 0]), -0.5)       # eğri üzerinde: K = −1/G(−0.5) = 0.65
    assert abs(K[0] - 0.65) < 1e-12 and np.min(np.abs(P[:, 0] + 0.5)) < 1e-12


def test_gain_for_damping():
    G = lti.tf([1, 3], [1, 4, 5, 0])
    res = cl.gain_for_damping(G, zeta=0.5)
    assert res
    for r in res:
        assert r["K"] > 0
        p = r["pole"]
        assert abs(-p.real / abs(p) - 0.5) < 1e-9
        assert np.min(np.abs(r["poles"] - p)) < 1e-8 * abs(p)
    G3 = lti.tf([1], [1, 3, 3, 1])
    r = cl.gain_for_damping(G3, zeta=0.0)            # sanal eksen geçişi: Routh sınırı K = 8
    assert abs(r[0]["K"] - 8.0) < 1e-9 and abs(r[0]["wn"] - math.sqrt(3)) < 1e-9
    r = cl.gain_for_damping(G3, wn=1.5)
    assert r
    for x in r:
        assert abs(abs(x["pole"]) - 1.5) < 1e-9 and np.min(np.abs(x["poles"] - x["pole"])) < 1e-8
    Gd = lti.tf([0.1, 0.05], [1, -1.6, 0.64], 0.1)
    r = cl.gain_for_damping(Gd, zeta=0.6)
    assert r
    for x in r:
        s = np.log(x["pole"]) / 0.1
        assert abs(-s.real / abs(s) - 0.6) < 1e-8
        assert np.min(np.abs(x["poles"] - x["pole"])) < 1e-8
    with pytest.raises(ControlError):
        cl.gain_for_damping(G, zeta=1.2)


# ---------------------------------------------------------------------------------------------------------------
# öncül / ardıl, çentik
# ---------------------------------------------------------------------------------------------------------------
def _lead_check(G, C, info, PM):
    w = info["CrossoverFrequency"]
    Gs = cl._Siso(G)
    Cs = cl._Siso(C)
    L = complex((Cs.freq(np.array([w])) * Gs.freq(np.array([w])))[0])
    e = _targets(L, PM)
    assert e[0] < 1e-9 and e[1] < 1e-7, e
    assert info["Stable"]
    assert abs(info["PhaseMargin"] - PM) < 1e-7


def test_leadlag_exact_targets():
    G = lti.tf([1], [1, 1, 0])                        # tip 1: rampa hatası 1/Kv
    C, info = cl.leadlag(G, 50, ess=0.1, input="ramp")
    _lead_check(G, C, info, 50)
    assert info["Type"] == "lead" and abs(info["K"] - 10.0) < 1e-12
    _close(np.polyval(C.num[0][0], 0) / np.polyval(C.den[0][0], 0), 10.0)    # Kv = K·1 = 10 → ess = 0.1
    C, info = cl.leadlag(G, 50, wc=3)
    _lead_check(G, C, info, 50)
    C, info = cl.leadlag(G, 45, wc=1.5, K=2.0)        # Phillips: K ve wc birlikte
    _lead_check(G, C, info, 45)
    assert abs(info["K"] - 2.0) < 1e-12
    with pytest.raises(ControlError):
        cl.leadlag(G, 45, wc=1.5, K=3.0)              # bu K ile wc'de tek öncül / ardıl yetmez
    G0 = lti.tf([4], [1, 3, 2])
    C, info = cl.leadlag(G0, 60, ess=0.05)            # basamak hatası: 1/(1 + K·2) = 0.05
    assert abs(1 / (1 + info["K"] * 2.0) - 0.05) < 1e-12
    _lead_check(G0, C, info, 60)
    Gd = lti.tf([0.0048, 0.0047], [1, -1.905, 0.905], 0.1)
    C, info = cl.leadlag(Gd, 50, wc=1.0)
    _lead_check(Gd, C, info, 50)
    assert C.Ts == 0.1
    with pytest.raises(ControlError):
        cl.leadlag(G, 50)
    with pytest.raises(ControlError):
        cl.leadlag(lti.tf([1], [1, 1]), 50, ess=0.1, input="ramp")   # tip 0: rampa hatası sonsuz


def test_lead_lag_shapes():
    C = cl.lead(40, 2.0, K=3.0)
    v = complex(np.polyval(C.num[0][0], 2j) / np.polyval(C.den[0][0], 2j))
    assert abs(math.degrees(cmath.phase(v)) - 40) < 1e-10
    w = np.geomspace(0.01, 100, 2001)
    ph = np.angle(np.polyval(C.num[0][0], 1j * w) / np.polyval(C.den[0][0], 1j * w))
    assert abs(w[np.argmax(ph)] - 2.0) / 2.0 < 0.01
    C = cl.lag(-30, 0.5, Ts=0.05)
    z = np.exp(0.5j * 0.05)
    v = complex(np.polyval(C.num[0][0], z) / np.polyval(C.den[0][0], z))
    assert abs(math.degrees(cmath.phase(v)) + 30) < 1e-9


def test_notch_definition():
    from scipy.optimize import brentq
    w0, depth, width = 10.0, -20.0, 2.0
    N = cl.notch(w0, depth, width)

    def H(w):
        return abs(np.polyval(N.num[0][0], 1j * w) / np.polyval(N.den[0][0], 1j * w))
    assert abs(20 * math.log10(H(w0)) - depth) < 1e-10
    w1 = brentq(lambda w: 20 * math.log10(H(w)) - depth / 2, 1.0, w0)
    w2 = brentq(lambda w: 20 * math.log10(H(w)) - depth / 2, w0, 100.0)
    assert abs((w2 - w1) - width) < 1e-9
    Nd = cl.notch(w0, depth, width, Ts=0.01)
    z0 = np.exp(1j * w0 * 0.01)
    Hd = abs(np.polyval(Nd.num[0][0], z0) / np.polyval(Nd.den[0][0], z0))
    assert abs(20 * math.log10(Hd) - depth) < 1e-9
    assert abs(np.polyval(Nd.num[0][0], 1.0) / np.polyval(Nd.den[0][0], 1.0) - 1) < 1e-12
    Ni = cl.notch(w0, -np.inf, width)
    assert abs(np.polyval(Ni.num[0][0], 1j * w0)) < 1e-9
    with pytest.raises(ControlError):
        cl.notch(w0, 5, width)
