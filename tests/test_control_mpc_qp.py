"""QP çözücüsü (Goldfarb–Idnani çift aktif küme) — MATLAB quadprog / mpcActiveSetSolver ve KKT koşullarıyla.

Referanslar `tests/data/matlab/mpc_qp_cases.json` (MATLAB R2025b): problemler MATLAB'de rng(7) ile üretilir ve dosyaya
yazılır; testler aynı problemleri buradan okur.
"""
from __future__ import annotations

import numpy as np
import pytest

from spt.control.mpc.qp import (QPWorkspace, mpcActiveSetSolver, qp_solve, qp_solve_ge, STATUS_INFEASIBLE,
                                STATUS_MAXITER)
from tools.matlab_oracle import OracleUnavailable, fixture

QP_SCRIPT = r"""
rng(7, 'twister');
opts = optimoptions('quadprog', 'Algorithm', 'active-set', 'Display', 'off', ...
    'OptimalityTolerance', 1e-12, 'ConstraintTolerance', 1e-12, 'StepTolerance', 1e-14, 'MaxIterations', 2000);
kopt = mpcActiveSetOptions;
P = {};
for k = 1:6
    n = 2 + 3*k; m = 2*n + k;
    Mh = randn(n); H = Mh*Mh' + 0.5*eye(n); H = (H+H')/2;
    f = 5*randn(n,1); A = randn(m,n); b = rand(m,1);
    P{end+1} = struct('name', sprintf('rastgele_%d', k), 'H',H,'f',f,'A',A,'b',b,'Aeq',zeros(0,n),'beq',zeros(0,1));
end
for k = 1:2
    n = 8; m = 12; q = 2 + k;
    Mh = randn(n); H = Mh*Mh' + eye(n); H = (H+H')/2; f = 3*randn(n,1);
    x0 = randn(n,1); A = randn(m,n); b = A*x0 + rand(m,1);
    Aeq = randn(q,n); beq = Aeq*x0;
    P{end+1} = struct('name', sprintf('esitlikli_%d', k), 'H',H,'f',f,'A',A,'b',b,'Aeq',Aeq,'beq',beq);
end
P{end+1} = struct('name','dejenere_kose','H',eye(2),'f',[-2;-2],'A',[1 0; 0 1; 1 1],'b',[1;1;2],'Aeq',zeros(0,2),'beq',zeros(0,1));
n = 6; Mh = randn(n); H = Mh*Mh' + eye(n); H = (H+H')/2; f = 4*randn(n,1); A = randn(8,n); b = rand(8,1);
A = [A; A(1:3,:)]; b = [b; b(1:3)];
P{end+1} = struct('name','yinelenen_satir','H',H,'f',f,'A',A,'b',b,'Aeq',zeros(0,n),'beq',zeros(0,1));
n = 5; Mh = randn(n); H = Mh*Mh' + eye(n); H = (H+H')/2; f = randn(n,1); xs = -H\f;
A = randn(6,n); b = A*xs + 1 + rand(6,1);
P{end+1} = struct('name','hepsi_pasif','H',H,'f',f,'A',A,'b',b,'Aeq',zeros(0,n),'beq',zeros(0,1));
n = 6; [Q,~] = qr(randn(n)); H = Q*diag(logspace(0,6,n))*Q'; H = (H+H')/2; f = 1e3*randn(n,1);
P{end+1} = struct('name','kotu_kosullu_kutu','H',H,'f',f,'A',[eye(n); -eye(n)],'b',ones(2*n,1),'Aeq',zeros(0,n),'beq',zeros(0,1));
P{end+1} = struct('name','cozumsuz_esitsizlik','H',eye(3),'f',[1;1;1],'A',[1 0 0; -1 0 0; 0 1 0],'b',[0;-1;5],'Aeq',zeros(0,3),'beq',zeros(0,1));
P{end+1} = struct('name','cozumsuz_esitlik','H',eye(2),'f',[0;0],'A',[1 0; 0 1],'b',[1;1],'Aeq',[1 1],'beq',3);
P{end+1} = struct('name','zayif_aktif','H',diag([1 2 3 4]),'f',-[1;2;3;4],'A',[1 0 0 0; 0 1 1 0; -1 0 0 0; 0 0 0 1],'b',[1;3;0;2],'Aeq',zeros(0,4),'beq',zeros(0,1));
n = 30; m = 120; Mh = randn(n); H = Mh*Mh' + 0.1*eye(n); H = (H+H')/2; f = 10*randn(n,1);
A = randn(m,n); b = rand(m,1);
P{end+1} = struct('name','buyuk','H',H,'f',f,'A',A,'b',b,'Aeq',zeros(0,n),'beq',zeros(0,1));
R = {};
for k = 1:numel(P)
    p = P{k}; n = numel(p.f); m = size(p.A,1);
    [xq, fq, efq, ~, lq] = quadprog(p.H, p.f, p.A, p.b, p.Aeq, p.beq, [], [], zeros(n,1), opts);
    [xk, efk, iAk, lk] = mpcActiveSetSolver(p.H, p.f, p.A, p.b, p.Aeq, p.beq, false(m,1), kopt);
    if isstruct(lq), lqi = lq.ineqlin; lqe = lq.eqlin; else, lqi = []; lqe = []; end
    R{end+1} = struct('xq', xq, 'fq', fq, 'efq', efq, 'lq_ineq', lqi, 'lq_eq', lqe, ...
        'xk', xk, 'efk', efk, 'iAk', double(iAk), 'lk_ineq', lk.ineqlin, 'lk_eq', lk.eqlin);
end
out.P = P;
out.R = R;
p = P{4}; m = size(p.A,1);
[~, ~, iA4] = mpcActiveSetSolver(p.H, p.f, p.A, p.b, p.Aeq, p.beq, false(m,1), kopt);
f2 = p.f + 0.2*randn(size(p.f));
[xw, efw] = mpcActiveSetSolver(p.H, f2, p.A, p.b, p.Aeq, p.beq, iA4, kopt);
[xc, efc] = mpcActiveSetSolver(p.H, f2, p.A, p.b, p.Aeq, p.beq, false(m,1), kopt);
out.warm = struct('f2', f2, 'iA0', double(iA4), 'xw', xw, 'efw', efw, 'xc', xc, 'efc', efc);
"""


@pytest.fixture(scope="module")
def ref():
    try:
        return fixture("mpc_qp_cases", QP_SCRIPT)
    except OracleUnavailable as e:  # pragma: no cover
        pytest.skip(str(e))


def _arr(x, shape):
    a = np.asarray(x if x is not None else [], dtype=float)
    if a.size == 0:
        return np.zeros(shape)
    return a.reshape(shape)


def _problem(p):
    f = _arr(p["f"], (-1,))
    n = f.size
    A = _arr(p["A"], (-1, n))
    b = _arr(p["b"], (-1,))
    Aeq = _arr(p["Aeq"], (0, n)) if not p["Aeq"] else _arr(p["Aeq"], (-1, n))
    beq = _arr(p["beq"], (Aeq.shape[0],))
    return _arr(p["H"], (n, n)), f, A, b, Aeq, beq


def _kkt(H, f, A, b, Aeq, beq, x, lam, mu, tol):
    """KKT: durağanlık, birincil / çift olurluk, tümleyen gevşeklik (A x <= b, Aeq x = beq)."""
    g = H @ x + f + A.T @ lam + Aeq.T @ mu
    scale = 1.0 + np.abs(H).max() * (1 + np.abs(x).max()) + np.abs(f).max()
    assert np.abs(g).max() <= tol * scale, f"durağanlık {np.abs(g).max():.3e}"
    if A.size:
        s = b - A @ x
        assert s.min() >= -1e-9 * (1 + np.abs(b).max()), f"olurluk {s.min():.3e}"
        assert lam.min() >= -1e-12 * (1 + np.abs(lam).max())
        assert np.abs(lam * s).max() <= tol * (1 + np.abs(lam).max()) * (1 + np.abs(b).max())
    if Aeq.size:
        assert np.abs(Aeq @ x - beq).max() <= 1e-9 * (1 + np.abs(beq).max())


def test_qp_matlab_quadprog_ve_kwik(ref):
    n_ok = 0
    for p, r in zip(ref["P"], ref["R"]):
        H, f, A, b, Aeq, beq = _problem(p)
        res = qp_solve(H, f, A, b, Aeq, beq)
        name = p["name"]
        if name.startswith("cozumsuz"):
            assert res.status == STATUS_INFEASIBLE, name
            assert r["efk"] == -1 and r["efq"] == -2, name
            continue
        xq = _arr(r["xq"], (-1,))
        xk = _arr(r["xk"], (-1,))
        assert res.status > 0, (name, res.status)
        assert r["efk"] > 0 and r["efq"] == 1, name
        # quadprog (etkin küme, sıkı tolerans) ile ~1e-9 (göreli)
        sc = 1.0 + np.abs(xq).max()
        assert np.abs(res.x - xq).max() <= 1e-9 * sc * (1e3 if name == "kotu_kosullu_kutu" else 1), \
            (name, np.abs(res.x - xq).max())
        # MATLAB KWIK (mpcActiveSetSolver) ile de aynı optimum
        assert np.abs(res.x - xk).max() <= 1e-8 * sc * (1e3 if name == "kotu_kosullu_kutu" else 1), name
        # çarpanlar (dejenere köşede çarpanlar tek değil; orada yalnız KKT)
        if name not in ("dejenere_kose", "yinelenen_satir"):
            lq = _arr(r["lq_ineq"], (A.shape[0],))
            assert np.abs(res.lam - lq).max() <= 1e-7 * (1 + np.abs(lq).max()), name
            act_k = _arr(r["iAk"], (A.shape[0],)) > 0.5
            assert np.array_equal(res.active[:A.shape[0]], act_k), name
        _kkt(H, f, A, b, Aeq, beq, res.x, res.lam, res.lam_eq, 1e-9)
        n_ok += 1
    assert n_ok >= 12


def test_qp_sicak_baslangic(ref):
    p = ref["P"][3]
    H, f, A, b, Aeq, beq = _problem(p)
    w = ref["warm"]
    f2 = _arr(w["f2"], (-1,))
    iA0 = _arr(w["iA0"], (-1,)) > 0.5
    cold = qp_solve(H, f2, A, b)
    warm = qp_solve(H, f2, A, b, iA0=iA0)
    xw = _arr(w["xw"], (-1,))
    assert np.abs(warm.x - xw).max() <= 1e-9 * (1 + np.abs(xw).max())
    assert np.abs(cold.x - xw).max() <= 1e-9 * (1 + np.abs(xw).max())
    assert 0 < warm.iterations <= cold.iterations
    assert w["efw"] > 0 and w["efc"] > 0


def test_mpcActiveSetSolver_arayuzu(ref):
    p = ref["P"][6]  # eşitlikli
    H, f, A, b, Aeq, beq = _problem(p)
    r = ref["R"][6]
    x, exitflag, iA, lam = mpcActiveSetSolver(H, f, A, b, Aeq, beq, np.zeros(A.shape[0], bool))
    assert exitflag > 0
    assert np.abs(x - _arr(r["xk"], (-1,))).max() <= 1e-9 * (1 + np.abs(x).max())
    assert np.abs(lam["eqlin"] - _arr(r["lk_eq"], (-1,))).max() <= 1e-7 * (1 + np.abs(lam["eqlin"]).max())
    assert iA.dtype == bool and iA.shape == (A.shape[0],)


def test_rastgele_kkt():
    """MATLAB'siz: rastgele (bazıları dejenere) problemlerde KKT koşulları — konveks QP'de en iyiliğin kanıtı."""
    rng = np.random.default_rng(3)
    for trial in range(40):
        n = int(rng.integers(1, 12))
        m = int(rng.integers(0, 3 * n + 2))
        q = int(rng.integers(0, max(1, n // 2)))
        M = rng.standard_normal((n, n))
        H = M @ M.T + 0.05 * np.eye(n)
        f = 3 * rng.standard_normal(n)
        x0 = rng.standard_normal(n)
        A = rng.standard_normal((m, n))
        b = A @ x0 + rng.random(m) * (trial % 3)  # bazıları x0'da tam aktif (dejenere)
        Aeq = rng.standard_normal((q, n))
        beq = Aeq @ x0
        res = qp_solve(H, f, A, b, Aeq, beq)
        assert res.status > 0
        _kkt(H, f, A, b, Aeq, beq, res.x, res.lam, res.lam_eq, 1e-9)


def test_ge_bicimi_ve_iterasyon_siniri():
    # Ac x >= bc biçimi (MPC iç biçimi) — basit kutu
    H = np.diag([1.0, 4.0])
    f = np.array([-4.0, -8.0])
    Ac = np.array([[-1.0, 0.0], [0.0, -1.0]])
    bc = np.array([-1.0, -1.0])  # x <= 1
    res = qp_solve_ge(H, f, Ac, bc)
    assert res.status > 0 and np.allclose(res.x, [1.0, 1.0])
    assert np.allclose(res.lam, [3.0, 4.0])
    res = qp_solve_ge(H, f, Ac, bc, max_iter=1)
    assert res.status == STATUS_MAXITER


def test_linv_girdisi_ve_hata():
    """UseHessianAsInput=False (MATLAB: ilk girdi Linv = L⁻¹); pozitif tanımlı olmayan H açık hata."""
    from spt.control.errors import ControlError
    from spt.control.mpc.qp import hessian_factor
    H = np.array([[4.0, 1.0], [1.0, 3.0]])
    f = np.array([-1.0, -2.0])
    A = np.array([[1.0, 1.0]])
    b = np.array([0.2])
    x1, e1, _, _ = mpcActiveSetSolver(H, f, A, b, None, None, np.zeros(1, bool))
    opts = {"UseHessianAsInput": False}
    x2, e2, _, _ = mpcActiveSetSolver(hessian_factor(H), f, A, b, None, None, np.zeros(1, bool), opts)
    assert e1 > 0 and e2 > 0 and np.allclose(x1, x2, atol=1e-14)
    with pytest.raises(ControlError):
        qp_solve(np.array([[1.0, 2.0], [2.0, 1.0]]), f)


def test_calisma_alani_yeniden_kullanim():
    """Sabit boyutlu çalışma alanı: aynı nesne tekrar tekrar (MCU'daki statik bellek düzeni)."""
    ws = QPWorkspace(n_max=6, m_max=10)
    rng = np.random.default_rng(1)
    for _ in range(10):
        n = 5
        M = rng.standard_normal((n, n))
        H = M @ M.T + np.eye(n)
        f = rng.standard_normal(n)
        A = rng.standard_normal((8, n))
        b = rng.random(8)
        r1 = qp_solve(H, f, A, b, workspace=ws)
        r2 = qp_solve(H, f, A, b)
        assert np.allclose(r1.x, r2.x, atol=1e-12)
