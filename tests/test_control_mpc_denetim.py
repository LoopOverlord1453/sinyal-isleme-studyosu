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
