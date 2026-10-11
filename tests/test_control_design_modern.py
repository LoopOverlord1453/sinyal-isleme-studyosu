"""Modern tasarım (spt.control.modern) — MATLAB R2025b Control System Toolbox referanslı testler.

Referans: `tests/data/matlab/control_design_modern.json`. Riccati / kazanç tabanlı her şey (icare, idare, lqr, dlqr,
lqrd, lqi, lqry, kalman, kalmd, lqg, reg, estim) MATLAB ile göreli 1e-8; kutup yerleştirme: tek girişte K, çok girişte
kapalı çevrim özdeğerleri; kontrol edilemez / tespit edilemez durumlarda ControlError.
"""
from __future__ import annotations

import numpy as np
import pytest

from spt.control import modern as md
from spt.control.errors import ControlError
from tools.matlab_oracle import OracleUnavailable, fixture

SCRIPT = r"""
A = [-0.5 1 0; 0 -1 2; 1 0 -3]; B = [1 0; 0 1; 1 1];
Q = [2 0.5 0; 0.5 1 0; 0 0 1]; R = [1 0.2; 0.2 2]; S = [0.1 0; 0 0.2; 0.1 0.1]; E = [1 0.1 0; 0 1 0; 0 0 1.2];
Ad = [0.9 0.2 0; 0 0.8 0.1; 0.1 0 1.05];
[X, K, L] = icare(A, B, Q, R, S, E); out.care = struct('X', X, 'K', K, 'L', spt_c(L));
try
  [X, K, L] = icare(A, B, Q, []); out.care0 = struct('X', X, 'K', K, 'L', spt_c(L));
catch e
  out.err_care0 = e.message;
end
[X, K, L] = idare(Ad, B, Q, R, S, E); out.dare = struct('X', X, 'K', K, 'L', spt_c(L));
[X, K, L] = idare(Ad, B, Q, R); out.dare0 = struct('X', X, 'K', K, 'L', spt_c(L));
Nx = [0.1 0; 0 0.2; 0.1 0.1];
[K, Sx, Pc] = lqr(A, B, Q, R, Nx); out.lqr = struct('K', K, 'S', Sx, 'P', spt_c(Pc));
[K, Sx, Pc] = lqr(ss(Ad, B, eye(3), 0, 0.1), Q, R, Nx); out.lqrsysd = struct('K', K, 'S', Sx, 'P', spt_c(Pc));
[K, Sx, Pc] = dlqr(Ad, B, Q, R, Nx); out.dlqr = struct('K', K, 'S', Sx, 'P', spt_c(Pc));
[K, Sx, Pc] = lqrd(A, B, Q, R, Nx, 0.1); out.lqrd = struct('K', K, 'S', Sx, 'P', spt_c(Pc));
[K, Sx, Pc] = lqrd(A, B, Q, R, 0.05); out.lqrd0 = struct('K', K, 'S', Sx, 'P', spt_c(Pc));
Cy = [1 0 0; 0 1 1]; Dy = [0 0; 0.1 0];
sysc = ss(A, B, Cy, Dy); sysd = c2d(sysc, 0.1);
out.sysd = struct('A', sysd.A, 'B', sysd.B, 'C', sysd.C, 'D', sysd.D);
[K, Sx, Pc] = lqi(sysc, blkdiag(Q, [1 0.1; 0.1 0.5]), R); out.lqic = struct('K', K, 'S', Sx, 'P', spt_c(Pc));
[K, Sx, Pc] = lqi(sysd, blkdiag(Q, [1 0.1; 0.1 0.5]), R, [Nx; 0.05 0; 0 0.05]);
out.lqid = struct('K', K, 'S', Sx, 'P', spt_c(Pc));
Qy = [2 0.3; 0.3 1]; Ny = [0.1 0; 0 0.2];
[K, Sx, Pc] = lqry(sysc, Qy, R, Ny); out.lqryc = struct('K', K, 'S', Sx, 'P', spt_c(Pc));
[K, Sx, Pc] = lqry(sysd, Qy, R, Ny); out.lqryd = struct('K', K, 'S', Sx, 'P', spt_c(Pc));
K = place(A, B(:, 1), [-1 -2 -3]); out.place1 = K;
K = place(A, B, [-1+1i -1-1i -4]); out.place2 = spt_c(eig(A - B * K));
K = place(Ad, B, [0.5 0.2+0.3i 0.2-0.3i]); out.place3 = spt_c(eig(Ad - B * K));
K = acker(A, B(:, 1), [-1+2i -1-2i -3]); out.acker = K;
Bk = [B, [0.5 0; 1 0.2; 0 1]]; Dk = [0 0 0.1 0; 0 0 0 0.2];
sysk = ss(A, Bk, Cy, Dk);
Qn = [1 0.2; 0.2 0.5]; Rn = [0.3 0.05; 0.05 0.2]; Nn = [0.05 0; 0 0.02];
[kest, L, Pk] = kalman(sysk, Qn, Rn, Nn);
out.kalc = struct('A', kest.A, 'B', kest.B, 'C', kest.C, 'D', kest.D, 'L', L, 'P', Pk);
syskd = c2d(sysk, 0.1);
out.syskd = struct('A', syskd.A, 'B', syskd.B, 'C', syskd.C, 'D', syskd.D);
[kest, L, Pk, Mx, Z, My] = kalman(syskd, Qn, Rn, Nn);
out.kald = struct('A', kest.A, 'B', kest.B, 'C', kest.C, 'D', kest.D, 'L', L, 'P', Pk, 'Mx', Mx, 'Z', Z, 'My', My);
[kest, L, Pk, Mx, Z] = kalman(syskd, Qn, Rn, Nn, 'delayed');
out.kaldd = struct('A', kest.A, 'B', kest.B, 'C', kest.C, 'D', kest.D);
try, [kest, L, Pk] = kalman(sysk, Qn, 0.2, [0.05; 0.02], 2, [1 2]); catch e, out.err_kalsk = e.message; end
if exist('kest', 'var'), out.kalsk = struct('A', kest.A, 'B', kest.B, 'C', kest.C, 'D', kest.D, 'L', L, 'P', Pk); end
[kest, L, Pk, M, Z] = kalmd(ss(A, [B(:, 1), [0.5; 1; 0]], [1 0 0], 0), 0.7, 0.05, 0.1);
out.kalmd = struct('A', kest.A, 'B', kest.B, 'C', kest.C, 'D', kest.D, 'L', L, 'P', Pk, 'M', M, 'Z', Z);
A1 = [0.9 0.1; -0.2 0.8]; B1 = [1; 0]; C1 = [1 0.3];
sd1 = ss(A1, B1, C1, 0, 0.1); sd2 = ss(A1, B1, C1, 0.3, 0.1);
QXU = blkdiag(eye(2), 1); QWV = blkdiag(0.5 * eye(2), 0.2);
rg = lqg(sd1, QXU, QWV); out.lqg1 = struct('A', rg.A, 'B', rg.B, 'C', rg.C, 'D', rg.D);
try
  rg = lqg(sd2, QXU, QWV, 'current'); out.lqg2 = struct('A', rg.A, 'B', rg.B, 'C', rg.C, 'D', rg.D);
catch e
  out.err_lqg2 = e.message;
end
try
  rg = lqg(sd2, QXU, QWV, 2, '2dof', 'current'); out.lqg3 = struct('A', rg.A, 'B', rg.B, 'C', rg.C, 'D', rg.D);
catch e
  out.err_lqg3 = e.message;
end
rg = lqg(sd2, QXU, QWV, 2, '1dof'); out.lqg4 = struct('A', rg.A, 'B', rg.B, 'C', rg.C, 'D', rg.D);
sc = ss([0 1; -2 -3], [0; 1], [1 0], 0.2);
rg = lqg(sc, QXU, QWV, 2); out.lqg5 = struct('A', rg.A, 'B', rg.B, 'C', rg.C, 'D', rg.D);
rg = lqg(sc, QXU, QWV); out.lqg6 = struct('A', rg.A, 'B', rg.B, 'C', rg.C, 'D', rg.D);
rg = reg(sd2, [0.5 0.2], [0.3; 0.1]); out.reg = struct('A', rg.A, 'B', rg.B, 'C', rg.C, 'D', rg.D);
try
  es = estim(ss(A, Bk, Cy, Dk), [0.3 0; 0.1 0.2; 0 0.4], [1 2], [1 2]);
  out.estim = struct('A', es.A, 'B', es.B, 'C', es.C, 'D', es.D);
catch e
  out.err_estim = e.message;
end
try, lqr([1 0; 0 2], [1; 0], eye(2), 1); catch e, out.err_lqr = e.message; end
try, place([1 0; 0 2], [1; 0], [-1 -2]); catch e, out.err_place = e.message; end
try, kalman(ss([1 0; 0 2], [1 0; 0 1], [1 0], 0), 1, 1); catch e, out.err_kalman = e.message; end
try, lqr(A, B, Q, -R); catch e, out.err_R = e.message; end
"""


@pytest.fixture(scope="module")
def ref():
    try:
        return fixture("control_design_modern", SCRIPT)
    except OracleUnavailable as exc:  # pragma: no cover
        pytest.skip(str(exc))


# ---------------------------------------------------------------------------------------------------------------
# veri ve yardımcılar (MATLAB betiğindeki matrislerin aynısı)
# ---------------------------------------------------------------------------------------------------------------
A = np.array([[-0.5, 1, 0], [0, -1, 2], [1, 0, -3.0]])
B = np.array([[1, 0], [0, 1], [1, 1.0]])
Q = np.array([[2, 0.5, 0], [0.5, 1, 0], [0, 0, 1.0]])
R = np.array([[1, 0.2], [0.2, 2.0]])
S = np.array([[0.1, 0], [0, 0.2], [0.1, 0.1]])
E = np.array([[1, 0.1, 0], [0, 1, 0], [0, 0, 1.2]])
AD = np.array([[0.9, 0.2, 0], [0, 0.8, 0.1], [0.1, 0, 1.05]])
NX = S.copy()
CY = np.array([[1, 0, 0], [0, 1, 1.0]])
DY = np.array([[0, 0], [0.1, 0]])
BK = np.hstack([B, [[0.5, 0], [1, 0.2], [0, 1]]])
DK = np.array([[0, 0, 0.1, 0], [0, 0, 0, 0.2]])
QN = np.array([[1, 0.2], [0.2, 0.5]])
RN = np.array([[0.3, 0.05], [0.05, 0.2]])
NN = np.array([[0.05, 0], [0, 0.02]])
A1 = np.array([[0.9, 0.1], [-0.2, 0.8]])
B1 = np.array([[1.0], [0]])
C1 = np.array([[1, 0.3]])
QXU = np.diag([1.0, 1, 1])
QWV = np.diag([0.5, 0.5, 0.2])


def M(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return a if a.ndim == 2 else (a.reshape(1, -1) if a.ndim == 1 else a.reshape(1, 1))


def cplx(x) -> np.ndarray:
    a = np.asarray(x, dtype=float).reshape(-1, 2)
    return a[:, 0] + 1j * a[:, 1]


def rel(a, b, tol=1e-8):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float).reshape(a.shape)
    scale = max(float(np.max(np.abs(b))), 1e-300) if b.size else 1.0
    err = float(np.max(np.abs(a - b))) / scale if b.size else 0.0
    assert err <= tol, f"göreli fark {err:.2e}"
    return err


def same_set(a, b, tol=1e-8):
    from scipy.optimize import linear_sum_assignment
    a = np.asarray(a, dtype=complex).ravel()
    b = np.asarray(b, dtype=complex).ravel()
    assert a.size == b.size
    D = np.abs(a[:, None] - b[None, :])
    r, c = linear_sum_assignment(D)
    err = float(np.max(D[r, c] / np.maximum(1.0, np.abs(b[c]))))
    assert err <= tol, f"özdeğer farkı {err:.2e}"


def ss_cmp(sys, m, tol=1e-8):
    for f in ("A", "B", "C", "D"):
        rel(getattr(sys, f), M(m[f]).reshape(np.shape(getattr(sys, f))), tol)


# ---------------------------------------------------------------------------------------------------------------
# Riccati ve LQR ailesi
# ---------------------------------------------------------------------------------------------------------------
def test_care_dare(ref):
    for key, fn, args in (("care", md.care, (A, B, Q, R, S, E)), ("care0", md.care, (A, B, Q)),
                          ("dare", md.dare, (AD, B, Q, R, S, E)), ("dare0", md.dare, (AD, B, Q, R))):
        X, K, L = fn(*args)
        rel(X, ref[key]["X"])
        rel(K, ref[key]["K"])
        same_set(L, cplx(ref[key]["L"]))


def test_lqr_dlqr_lqrd(ref):
    for key, out in (("lqr", md.lqr(A, B, Q, R, NX)),
                     ("lqrsysd", md.lqr((AD, B, np.eye(3), np.zeros((3, 2)), 0.1), Q, R, NX)),
                     ("dlqr", md.dlqr(AD, B, Q, R, NX)),
                     ("lqrd", md.lqrd(A, B, Q, R, NX, 0.1)),
                     ("lqrd0", md.lqrd(A, B, Q, R, 0.05))):
        K, X, P = out
        rel(K, ref[key]["K"])
        rel(X, ref[key]["S"])
        same_set(P, cplx(ref[key]["P"]))


def test_lqi_lqry(ref):
    sd = ref["sysd"]
    sysd = (M(sd["A"]), M(sd["B"]), M(sd["C"]), M(sd["D"]), 0.1)
    sysc = (A, B, CY, DY)
    Qi = np.block([[Q, np.zeros((3, 2))], [np.zeros((2, 3)), np.array([[1, 0.1], [0.1, 0.5]])]])
    Ni = np.vstack([NX, [[0.05, 0], [0, 0.05]]])
    QY = np.array([[2, 0.3], [0.3, 1]])
    NY = np.array([[0.1, 0], [0, 0.2]])
    for key, out in (("lqic", md.lqi(sysc, Qi, R)), ("lqid", md.lqi(sysd, Qi, R, Ni)),
                     ("lqryc", md.lqry(sysc, QY, R, NY)), ("lqryd", md.lqry(sysd, QY, R, NY))):
        K, X, P = out
        rel(K, ref[key]["K"])
        rel(X, ref[key]["S"])
        same_set(P, cplx(ref[key]["P"]))


def test_lqr_accepts_lti_and_checks_inputs():
    from spt.control import lti
    sys = lti.ss(A, B, CY, DY)
    K1, _, _ = md.lqr(sys, Q, R)
    K2, _, _ = md.lqr(A, B, Q, R)
    rel(K1, K2, 1e-14)
    with pytest.raises(ControlError):
        md.lqr(A, B, Q, -R)                         # R pozitif tanımlı değil
    with pytest.raises(ControlError):
        md.lqr(A, B, Q, R, 10 * NX)                 # [Q N; N' R] belirsiz
    with pytest.raises(ControlError):
        md.lqr(A, B, Q + np.triu(np.ones((3, 3)), 1), R)   # simetrik değil
    with pytest.raises(ControlError):
        md.lqr(np.diag([1.0, 2.0]), [[1.0], [0.0]], np.eye(2), 1)   # kararsız kip 2 kontrol edilemez
    with pytest.raises(ControlError):
        md.lqr(lti.tf([1], [1, 1], input_delay=0.2), 1, 1)          # sürekli gecikme


# ---------------------------------------------------------------------------------------------------------------
# Kutup yerleştirme
# ---------------------------------------------------------------------------------------------------------------
def test_place_acker(ref):
    K = md.place(A, B[:, :1], [-1, -2, -3])
    rel(K, ref["place1"], 1e-8)
    for key, Am, p in (("place2", A, [-1 + 1j, -1 - 1j, -4]), ("place3", AD, [0.5, 0.2 + 0.3j, 0.2 - 0.3j])):
        K = md.place(Am, B, p)
        ev = np.linalg.eigvals(Am - B @ K)
        same_set(ev, cplx(ref[key]), 1e-8)
        same_set(ev, p, 1e-8)
    K = md.acker(A, B[:, 0], [-1 + 2j, -1 - 2j, -3])
    rel(K, ref["acker"], 1e-8)
    with pytest.raises(ControlError):
        md.place(np.diag([1.0, 2.0]), [[1.0], [0.0]], [-1, -2])
    with pytest.raises(ControlError):
        md.place(A, B, [-1, -1, -1])                # katlılık 3 > rank(B) = 2
    with pytest.raises(ControlError):
        md.place(A, B, [-1 + 1j, -1 + 1j, -4])      # eşlenik değil
    with pytest.raises(ControlError):
        md.acker(np.diag([1.0, 2.0]), [1.0, 0.0], [-1, -2])


# ---------------------------------------------------------------------------------------------------------------
# Kestiriciler
# ---------------------------------------------------------------------------------------------------------------
def test_kalman_continuous_and_discrete(ref):
    r = md.kalman((A, BK, CY, DK), QN, RN, NN)
    ss_cmp(r.kest, ref["kalc"])
    rel(r.L, ref["kalc"]["L"])
    rel(r.P, ref["kalc"]["P"])
    assert r.Mx is None and r.Z is None
    sk = ref["syskd"]
    sysd = (M(sk["A"]), M(sk["B"]), M(sk["C"]), M(sk["D"]), 0.1)
    r = md.kalman(sysd, QN, RN, NN)
    m = ref["kald"]
    ss_cmp(r.kest, m)
    for f in ("L", "P", "Mx", "Z", "My"):
        rel(getattr(r, f), M(m[f]).reshape(np.shape(getattr(r, f))))
    kest, L, P, Mx, Z = md.kalman(sysd, QN, RN, NN, type="delayed")
    ss_cmp(kest, ref["kaldd"])
    r = md.kalman((A, BK, CY, DK), QN, 0.2, [[0.05], [0.02]], sensors=[1], known=[0, 1])
    ss_cmp(r.kest, ref["kalsk"])
    rel(r.L, M(ref["kalsk"]["L"]).reshape(r.L.shape))
    rel(r.P, ref["kalsk"]["P"])


def test_kalmd(ref):
    sysc = (A, np.hstack([B[:, :1], [[0.5], [1.0], [0.0]]]), np.array([[1.0, 0, 0]]), np.zeros((1, 2)))
    kest, L, P, Mx, Z = md.kalmd(sysc, 0.7, 0.05, 0.1)
    m = ref["kalmd"]
    ss_cmp(kest, m)
    rel(L, M(m["L"]).reshape(L.shape))
    rel(P, m["P"])
    rel(Mx, M(m["M"]).reshape(Mx.shape))
    rel(Z, m["Z"])


def test_lqg_reg_estim(ref):
    sd1 = (A1, B1, C1, np.zeros((1, 1)), 0.1)
    sd2 = (A1, B1, C1, np.array([[0.3]]), 0.1)
    sc = (np.array([[0, 1], [-2, -3.0]]), np.array([[0], [1.0]]), np.array([[1.0, 0]]), np.array([[0.2]]))
    cases = [("lqg1", md.lqg(sd1, QXU, QWV)),
             ("lqg2", md.lqg(sd2, QXU, QWV, estimator="current")),
             ("lqg3", md.lqg(sd2, QXU, QWV, 2, "2dof", estimator="current")),
             ("lqg4", md.lqg(sd2, QXU, QWV, 2, "1dof")),
             ("lqg5", md.lqg(sc, QXU, QWV, 2)),
             ("lqg6", md.lqg(sc, QXU, QWV)),
             ("reg", md.reg(sd2, [[0.5, 0.2]], [[0.3], [0.1]]))]
    for key, sys in cases:
        ss_cmp(sys, ref[key])
    es = md.estim((A, BK, CY, DK), [[0.3, 0], [0.1, 0.2], [0, 0.4]], sensors=[0, 1], known=[0, 1])
    ss_cmp(es, ref["estim"])


def test_errors_match_matlab(ref):
    assert "err_lqr" in ref and "err_place" in ref and "err_kalman" in ref and "err_R" in ref
    with pytest.raises(ControlError):
        md.lqr(np.diag([1.0, 2.0]), [[1.0], [0.0]], np.eye(2), 1)
    with pytest.raises(ControlError):
        md.place(np.diag([1.0, 2.0]), [[1.0], [0.0]], [-1, -2])
    with pytest.raises(ControlError):
        md.kalman((np.diag([1.0, 2.0]), np.eye(2), np.array([[1.0, 0]]), np.zeros((1, 2))), 1, 1)
    with pytest.raises(ControlError):
        md.lqr(A, B, Q, -R)


# ---------------------------------------------------------------------------------------------------------------
# Tanım testleri: N̄, Bryson, lqi'nin izleme özelliği
# ---------------------------------------------------------------------------------------------------------------
def test_nbar_unit_dc_gain():
    for Ts, Am in ((0.0, A), (0.1, AD)):
        sys = (Am, B, CY, DY, Ts)
        K, _, _ = md.lqr(sys, Q, R)
        Nb = md.nbar(sys, K)
        Acl, Bcl, Ccl, Dcl = Am - B @ K, B @ Nb, CY - DY @ K, DY @ Nb
        G0 = (Ccl @ np.linalg.solve(np.eye(3) - Acl, Bcl) + Dcl) if Ts else (-Ccl @ np.linalg.solve(Acl, Bcl) + Dcl)
        rel(G0, np.eye(2), 1e-12)


def test_bryson():
    Qb, Rb = md.bryson([0.5, 2.0, np.inf], [10.0])
    rel(Qb, np.diag([4.0, 0.25, 0.0]), 1e-15)
    rel(Rb, [[0.01]], 1e-15)
    with pytest.raises(ControlError):
        md.bryson([0.0], [1.0])


def test_outputs_are_lti_statespace():
    from spt.control import lti
    r = md.kalman((A1, np.hstack([B1, np.eye(2)[:, :1]]), C1, np.zeros((1, 2)), 0.1), 1.0, 0.1)
    assert isinstance(r.kest, lti.StateSpace) and r.kest.Ts == 0.1
    assert isinstance(md.lqg((A1, B1, C1, np.zeros((1, 1)), 0.1), QXU, QWV), lti.StateSpace)
