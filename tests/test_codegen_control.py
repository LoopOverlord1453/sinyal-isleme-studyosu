"""Kontrolcü C kod üreteci (spt.codegen.control_c): MATLAB ile birebirlik ve MCU derleme denetimleri.

MATLAB kahini (tests/data/matlab/codegen_control.json): pid / pidstd / pid2 nesnelerinin lsim çıktısı, Simulink
"Discrete PID Controller (2DOF)" bloğunun doyum + anti-windup (geri hesaplama / sıkıştırma) + izleme modlu ayrık
çıktısı, ayrık tf / ss lsim, reg / estim (delayed / current), lqi, kalman (current / delayed). Testler MATLAB'siz
çalışır; üretilen C ana makinede derlenip aynı girişlerle çalıştırılır.
"""
from __future__ import annotations

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from spt.codegen import control_c as cc
from spt.codegen import toolchain as tc
from spt.codegen.mcu import MCU_PROFILES
from spt.control.errors import ControlError, ControlWarning
from tools.matlab_oracle import OracleUnavailable, fixture


@pytest.fixture(scope="module")
def qapp_offscreen():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("SPT_SETTINGS_FILE", os.path.join(tempfile.gettempdir(), "spt_test_control_code.ini"))
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app

ORACLE = r"""
Ts = 0.01; N = 400; k = (0:N-1)'; t = k*Ts;
q = @(x) round(1024*x)/1024;
e  = q(sin(0.07*k) + 0.5*sign(sin(0.013*k + 0.3)) + 0.1*cos(0.9*k));
r  = q(1.0*(k >= 10) - 1.5*(k >= 200) + 0.2*sin(0.05*k));
y  = q(0.4*sin(0.03*k) + 0.1*cos(0.31*k));
tr = q(0.3*cos(0.02*k) - 0.2);
u  = q(0.5*sin(0.021*k) + 0.2*cos(0.17*k));
y2 = q(0.3*cos(0.027*k) - 0.1*sin(0.11*k));
out.Ts = Ts; out.e = e; out.r = r; out.y = y; out.tr = tr; out.u = u; out.y2 = y2;

% ---------------------------------------------------------------- A: pid nesnesi lsim (1-DOF, girdi e)
P = [2 1.5 0.3 0.05 1 1; 2 1.5 0.3 0.05 2 2; 2 1.5 0.3 0.05 3 3; 1 0.5 0.02 0 1 2; 0.8 2 0.1 0.02 3 1;
     1.5 0 0 0 1 1; 0 3 0 0 2 1; 1.2 0.7 0.04 0.03 2 3];
F = {'ForwardEuler','BackwardEuler','Trapezoidal'};
res = {};
for i = 1:size(P,1)
    C = pid(P(i,1), P(i,2), P(i,3), P(i,4), Ts, 'IFormula', F{P(i,5)}, 'DFormula', F{P(i,6)});
    s = struct('Kp', P(i,1), 'Ki', P(i,2), 'Kd', P(i,3), 'Tf', P(i,4), 'IF', F{P(i,5)}, 'DF', F{P(i,6)});
    s.out = lsim(C, e, t);
    res{end+1} = s;
end
out.pid = res;
Cs = pidstd(1.2, 0.8, 0.1, 8, Ts, 'IFormula', 'BackwardEuler', 'DFormula', 'Trapezoidal');
Cp = pid(Cs);
out.pidstd = struct('Kp', 1.2, 'Ti', 0.8, 'Td', 0.1, 'N', 8, 'IF', 'BackwardEuler', 'DF', 'Trapezoidal', ...
                    'par', [Cp.Kp Cp.Ki Cp.Kd Cp.Tf], 'out', lsim(Cs, e, t));
C2 = pid2(2, 1.5, 0.3, 0.05, 0.6, 0.2, Ts, 'IFormula', 'Trapezoidal', 'DFormula', 'BackwardEuler');
out.pid2 = struct('Kp', 2, 'Ki', 1.5, 'Kd', 0.3, 'Tf', 0.05, 'b', 0.6, 'c', 0.2, 'IF', 'Trapezoidal', ...
                  'DF', 'BackwardEuler', 'out', lsim(C2, [r y], t));
errs = {};
tests = {{1, 1, 0.1, 0, 'ForwardEuler'}, {1, 1, 0.1, 0, 'Trapezoidal'}, {1, 1, 0.1, 0.004, 'ForwardEuler'}};
for i = 1:numel(tests)
    a = tests{i};
    try
        C = pid(a{1}, a{2}, a{3}, a{4}, Ts, 'DFormula', a{5});
        errs{end+1} = sprintf('ok stable=%d', isstable(C));
    catch err
        errs{end+1} = err.message;
    end
end
out.pid_errors = errs;

% ---------------------------------------------------------------- C: Simulink Discrete PID Controller (2DOF)
% sütunlar: form(1 paralel, 2 ideal), IF, DF, aw(0 yok, 1 geri hesaplama, 2 sıkıştırma), izleme, doyum, P, I, D, N
S = [1 1 1 0 0 1 3 8 0.05 40; 1 1 1 1 0 1 3 8 0.05 40; 1 1 1 2 0 1 3 8 0.05 40; 1 2 2 1 0 1 3 8 0.05 40;
     1 3 3 1 0 1 3 8 0.05 40; 1 2 3 2 0 1 3 8 0.05 40; 1 3 1 2 0 1 3 8 0.05 40; 1 1 2 1 1 1 3 8 0.05 40;
     1 2 1 2 1 1 3 8 0.05 40; 2 1 1 1 0 1 3 2.5 0.02 40; 2 2 2 2 0 1 3 2.5 0.02 40; 1 2 3 0 0 0 3 8 0.05 40;
     1 3 2 0 1 1 3 8 0.05 40; 1 3 3 2 0 1 3 8 0.05 40];
FS = {'Forward Euler', 'Backward Euler', 'Trapezoidal'};
AW = {'none', 'back-calculation', 'clamping'};
Kb = 2; Kt = 1.5; UL = 1.5; LL = -1.2; b = 0.7; c = 0.3;
mdl = 'spt_pid_oracle';
sres = {};
for i = 1:size(S,1)
    s = struct('form', S(i,1), 'IF', F{S(i,2)}, 'DF', F{S(i,3)}, 'aw', S(i,4), 'tracking', S(i,5), ...
               'sat', S(i,6), 'P', S(i,7), 'I', S(i,8), 'D', S(i,9), 'N', S(i,10), 'b', b, 'c', c, 'Kb', Kb, ...
               'Kt', Kt, 'UL', UL, 'LL', LL, 'err', '', 'out', []);
    try
        if bdIsLoaded(mdl), close_system(mdl, 0); end
        new_system(mdl);
        add_block('simulink/Sources/In1', [mdl '/r'], 'SampleTime', num2str(Ts));
        add_block('simulink/Sources/In1', [mdl '/y'], 'SampleTime', num2str(Ts));
        blk = [mdl '/pid'];
        add_block('simulink/Discrete/Discrete PID Controller (2DOF)', blk);
        forms = {'Parallel', 'Ideal'};
        set_param(blk, 'Form', forms{S(i,1)}, 'Controller', 'PID', 'SampleTime', num2str(Ts), ...
                  'IntegratorMethod', FS{S(i,2)}, 'FilterMethod', FS{S(i,3)}, ...
                  'P', num2str(S(i,7), 17), 'I', num2str(S(i,8), 17), 'D', num2str(S(i,9), 17), ...
                  'N', num2str(S(i,10), 17), 'b', num2str(b, 17), 'c', num2str(c, 17), 'UseFilter', 'on');
        if S(i,6)
            set_param(blk, 'LimitOutput', 'on', 'UpperSaturationLimit', num2str(UL, 17), ...
                      'LowerSaturationLimit', num2str(LL, 17), 'AntiWindupMode', AW{S(i,4)+1}, ...
                      'Kb', num2str(Kb, 17));
        end
        add_block('simulink/Sinks/Out1', [mdl '/u']);
        add_line(mdl, 'r/1', 'pid/1');
        add_line(mdl, 'y/1', 'pid/2');
        add_line(mdl, 'pid/1', 'u/1');
        X = [t r y];
        if S(i,5)
            set_param(blk, 'TrackingMode', 'on', 'Kt', num2str(Kt, 17));
            add_block('simulink/Sources/In1', [mdl '/tr'], 'SampleTime', num2str(Ts));
            add_line(mdl, 'tr/1', 'pid/3');
            X = [t r y tr];
        end
        si = Simulink.SimulationInput(mdl);
        si = si.setExternalInput(X);
        si = si.setModelParameter('Solver', 'FixedStepDiscrete', 'FixedStep', num2str(Ts), ...
            'StopTime', num2str((N-1)*Ts, 17), 'SaveFormat', 'Array', 'SaveOutput', 'on', ...
            'OutputSaveName', 'yout', 'SaveTime', 'on', 'AlgebraicLoopMsg', 'none', 'ReturnWorkspaceOutputs', 'on');
        so = sim(si);
        s.out = so.yout;
        s.tout = so.tout;
        if ~isempty(so.ErrorMessage), s.err = so.ErrorMessage; end
    catch err
        s.err = getReport(err, 'basic', 'hyperlinks', 'off');
    end
    sres{end+1} = s;
end
if bdIsLoaded(mdl), close_system(mdl, 0); end
out.simulink = sres;

% ---------------------------------------------------------------- D: genel ayrık LTI (tf / ss) lsim
Cc = tf(10*conv([1 2], [1 0.2 100]), conv(conv(conv([1 20], [1 14 100]), [1/50 1]), [1 0]));
C1 = c2d(Cc, Ts, 'tustin');
[n1, d1] = tfdata(C1, 'v');
out.tf1 = struct('num', n1, 'den', d1, 'out', lsim(C1, e, t));
C2t = tf([0.1 0.05], [1 -1.6 0.7 -0.05], Ts);
[n2, d2] = tfdata(C2t, 'v');
out.tf2 = struct('num', n2, 'den', d2, 'out', lsim(C2t, e, t));
A = [0.9 0.1 0; -0.1 0.85 0.05; 0 0.02 0.7]; B = [1 0; 0 0.5; 0.3 -0.2]; C = [1 0 0.5; 0 1 -1];
D = [0.1 0; 0 0.2];
out.ss1 = struct('A', A, 'B', B, 'C', C, 'D', D, 'out', lsim(ss(A, B, C, D, Ts), [r y], t));

% ---------------------------------------------------------------- E: durum geri beslemesi + gözlemci, LQI
sysd = c2d(ss([0 1; 0 -10], [0; 20], [1 0], 0), Ts);
[Ad, Bd, Cd, Dd] = ssdata(sysd);
K = dlqr(Ad, Bd, diag([100 1]), 1);
L = place(Ad', Cd', [0.6 0.65])';
out.obs = struct('A', Ad, 'B', Bd, 'C', Cd, 'D', Dd, 'K', K, 'L', L);
out.obs.reg_delayed = lsim(reg(sysd, K, L), y, t);
sys2 = ss(Ad, Bd, Cd, 0.1, Ts);
out.obs.reg_delayed_D = lsim(reg(sys2, K, L), y, t);
[a1, b1, c1, d1x] = ssdata(estim(sysd, L, 1, 1));
out.obs.estim_delayed = struct('A', a1, 'B', b1, 'C', c1, 'D', d1x);
out.obs.Nbar = 1/dcgain(ss(Ad - Bd*K, Bd, Cd - Dd*K, Dd, Ts));
% R2025b'de estim / reg yalnız 'delayed'; 'current' biçimi kalman(..., 'current') + lqgreg / lqgtrack ile
sysw = ss(Ad, [Bd Bd], Cd, [Dd 0], Ts);
[kc, Lc, ~, Mc] = kalman(sysw, 0.1, 1e-3, 0, 'current');
[kd, Ld, ~, Md] = kalman(sysw, 0.1, 1e-3, 0, 'delayed');
out.obs.Lc = Lc; out.obs.Mc = Mc; out.obs.Ld = Ld;
out.obs.lqg_current = lsim(lqgreg(kc, K), y, t);
out.obs.lqg_delayed = lsim(lqgreg(kd, K), y, t);
Qi = diag([10 1 50]);
Kq = lqi(sysd, Qi, 1);
out.obs.Klqi = Kq;
out.obs.Klqi_aug = dlqr([Ad zeros(2,1); -Ts*Cd 1], [Bd; -Ts*Dd], Qi, 1);
out.obs.track_current = lsim(lqgtrack(kc, Kq), [r y], t);
out.obs.track_delayed = lsim(lqgtrack(kd, Kq), [r y], t);

% ---------------------------------------------------------------- F: kararlı durum Kalman (current / delayed)
A3 = [0.95 0.1 0; 0 0.9 0.1; 0 0 0.8]; B3 = [0; 0.1; 1]; G3 = [0.1; 0; 0.05];
C3 = [1 0 0; 0 1 0.5]; D3 = [0; 0.2];
sysk = ss(A3, [B3 G3], C3, [D3 zeros(2,1)], Ts);
Qn = 0.5; Rn = diag([0.01 0.04]);
[kc, Lc, Pc, Mc] = kalman(sysk, Qn, Rn, zeros(1,2), 'current');
[kd, Ld, Pd, Md] = kalman(sysk, Qn, Rn, zeros(1,2), 'delayed');
out.kf = struct('A', A3, 'B', B3, 'C', C3, 'D', D3, 'Lc', Lc, 'Mc', Mc, 'Ld', Ld, 'Md', Md, ...
                'out_current', lsim(kc, [u y y2], t), 'out_delayed', lsim(kd, [u y y2], t));
"""


# Sıkıştırma + izleme birlikteyken işaret denetiminin yalnız Ki·e ile mi (Simulink: "integratör girişi") yoksa
# izleme terimi eklenmiş girişle mi yapıldığını ayırt eden durumlar (doyumdayken tr uzak, Kt büyük).
ORACLE_TRACK = r"""
Ts = 0.01; N = 300; k = (0:N-1)'; t = k*Ts;
q = @(x) round(1024*x)/1024;
r  = q(1.0*(k >= 10) - 0.8*(k >= 150) + 0.1*sin(0.05*k));
y  = q(0.2*sin(0.03*k));
tr = q(-1 + 0.5*(k >= 100) + 0.1*cos(0.07*k));
out.r = r; out.y = y; out.tr = tr; out.Ts = Ts;
S = [1 2; 2 1; 3 2];       % IF, DF
FS = {'Forward Euler', 'Backward Euler', 'Trapezoidal'};
F = {'ForwardEuler','BackwardEuler','Trapezoidal'};
mdl = 'spt_pid_oracle_trk';
res = {};
for i = 1:size(S,1)
    if bdIsLoaded(mdl), close_system(mdl, 0); end
    new_system(mdl);
    add_block('simulink/Sources/In1', [mdl '/r'], 'SampleTime', num2str(Ts));
    add_block('simulink/Sources/In1', [mdl '/y'], 'SampleTime', num2str(Ts));
    add_block('simulink/Sources/In1', [mdl '/tr'], 'SampleTime', num2str(Ts));
    blk = [mdl '/pid'];
    add_block('simulink/Discrete/Discrete PID Controller (2DOF)', blk);
    set_param(blk, 'Form', 'Parallel', 'Controller', 'PID', 'SampleTime', num2str(Ts), ...
        'IntegratorMethod', FS{S(i,1)}, 'FilterMethod', FS{S(i,2)}, 'P', '2', 'I', '6', 'D', '0.04', ...
        'N', '30', 'b', '0.8', 'c', '0.5', 'UseFilter', 'on', 'LimitOutput', 'on', 'UpperSaturationLimit', '1', ...
        'LowerSaturationLimit', '-0.5', 'AntiWindupMode', 'clamping', 'TrackingMode', 'on', 'Kt', '40');
    add_block('simulink/Sinks/Out1', [mdl '/u']);
    add_line(mdl, 'r/1', 'pid/1'); add_line(mdl, 'y/1', 'pid/2'); add_line(mdl, 'tr/1', 'pid/3');
    add_line(mdl, 'pid/1', 'u/1');
    si = Simulink.SimulationInput(mdl);
    si = si.setExternalInput([t r y tr]);
    si = si.setModelParameter('Solver', 'FixedStepDiscrete', 'FixedStep', num2str(Ts), ...
        'StopTime', num2str((N-1)*Ts, 17), 'SaveFormat', 'Array', 'SaveOutput', 'on', 'OutputSaveName', 'yout', ...
        'ReturnWorkspaceOutputs', 'on');
    so = sim(si);
    res{end+1} = struct('IF', F{S(i,1)}, 'DF', F{S(i,2)}, 'P', 2, 'I', 6, 'D', 0.04, 'N', 30, 'b', 0.8, ...
                        'c', 0.5, 'UL', 1, 'LL', -0.5, 'Kt', 40, 'out', so.yout);
end
close_system(mdl, 0);
out.cases = res;
"""


# Tam durum geri beslemesi (gözlemcisiz): dlqr + Nbar ve lqi kapalı çevrimi (bitki D ≠ 0, connect ile kurulur);
# süreç gürültüsü ölçüme doğrudan giren (H ≠ 0, Nn ≠ 0) 'current' Kalman — ŷ[n|n] için MATLAB'in My kazancı.
ORACLE_EXTRA = r"""
Ts = 0.01; N = 300; k = (0:N-1)'; t = k*Ts;
q = @(x) round(1024*x)/1024;
r  = q(1.0*(k >= 10) - 0.6*(k >= 150) + 0.1*sin(0.05*k));
u  = q(0.5*sin(0.021*k) + 0.2*cos(0.17*k));
y1 = q(0.4*sin(0.03*k) + 0.1*cos(0.31*k));
y2 = q(0.3*cos(0.027*k) - 0.1*sin(0.11*k));
out.Ts = Ts; out.r = r; out.u = u; out.y1 = y1; out.y2 = y2;
A = [1 0.0095; 0 0.905]; B = [0.00094; 0.19]; C = [1 0]; D = 0.05;
K = dlqr(A, B, diag([100 1]), 1);
Nb = 1/dcgain(ss(A - B*K, B, C - D*K, D, Ts));
Y = lsim(ss(A - B*K, B*Nb, [eye(2); -K], [zeros(2,1); Nb], Ts), r, t);
out.sf = struct('A', A, 'B', B, 'C', C, 'D', D, 'K', K, 'Nbar', Nb, 'x', Y(:,1:2), 'u', Y(:,3));
Kq = lqi(ss(A, B, C, D, Ts), diag([10 1 50]), 1);
plant = ss(A, B, [eye(2); C], [zeros(2,1); D], Ts, 'InputName', 'u', 'OutputName', {'x1', 'x2', 'y'});
integ = ss(1, Ts, 1, 0, Ts, 'InputName', 'e', 'OutputName', 'xi');
ctrl = ss([], [], [], -Kq, Ts, 'InputName', {'x1', 'x2', 'xi'}, 'OutputName', 'u');
cl = connect(plant, integ, ctrl, sumblk('e = r - y'), 'r', {'x1', 'x2', 'u'});
Y2 = lsim(cl, r, t);
out.lqi = struct('K', Kq, 'x', Y2(:,1:2), 'u', Y2(:,3));
A3 = [0.95 0.1 0; 0 0.9 0.1; 0 0 0.8]; B3 = [0; 0.1; 1]; G3 = [0.1; 0; 0.05];
C3 = [1 0 0; 0 1 0.5]; D3 = [0; 0.2]; H3 = [0.3; -0.1];
sysk = ss(A3, [B3 G3], C3, [D3 H3], Ts);
[kc, Lc, ~, Mc] = kalman(sysk, 0.5, diag([0.01 0.04]), [0.02 0.01], 'current');
[~, ~, ~, dk] = ssdata(kc);
try
    [~, ~, ~, ~, ~, Myc] = kalman(sysk, 0.5, diag([0.01 0.04]), [0.02 0.01], 'current');
catch
    Myc = [];
end
out.kfh = struct('A', A3, 'B', B3, 'C', C3, 'D', D3, 'L', Lc, 'M', Mc, 'My', Myc, 'kest_D', dk, ...
                 'out', lsim(kc, [u y1 y2], t));
"""


# Yüksek dereceli SISO (Ts = 1e-4, kutuplar z = 1 yakınında): h1 6. derece (Tustin), h2 11. derece (integratör ×
# Butterworth-10, 200 Hz). h2'nin tf katsayıları kendisini temsil etmez: MATLAB'in kendi lsim(tf)'i zpk'den ~%25 sapar.
# (Değerler bağımsız denetimin R2025b çıktısının bu betikle aynı ilk Nh örneğidir; lsim nedensel.)
ORACLE_HIGH = r"""
q = @(x) round(1024*x)/1024;
Th = 1e-4; Nh = 6000; kh = (0:Nh-1)'; th = kh*Th;
eh = q(0.5*sin(2*pi*5*th) + 0.3*sign(sin(2*pi*2*th + 0.1)) + 0.05*sin(2*pi*400*th));
out.Th = Th; out.eh = eh;
s = tf('s');
Cc1 = 80*(s+5)*(s+20)*(s^2 + 2*0.05*2000*s + 2000^2) / ...
      (s*(s+1000)*(s^2 + 2*0.7*1500*s + 1500^2)*(s^2 + 2*0.5*2000*s + 2000^2));
Cd1 = c2d(Cc1, Th, 'tustin');
[n1, d1] = tfdata(Cd1, 'v');
out.h1 = struct('num', real(n1), 'den', real(d1), 'y_tf', real(lsim(Cd1, eh, th)), ...
                'y_zpk', real(lsim(zpk(Cd1), eh, th)));
wc = 2*pi*200; nb = 10; pb = wc*exp(1j*pi*(2*(1:nb/2)+nb-1)/(2*nb)); pb = [pb(:); conj(pb(:))];
Cd2 = c2d(zpk([-10], [0], 30) * zpk([], pb, wc^nb), Th, 'tustin');
[n2, d2] = tfdata(Cd2, 'v');
out.h2 = struct('num', real(n2), 'den', real(d2), 'y_tf', real(lsim(tf(Cd2), eh, th)), ...
                'y_zpk', real(lsim(Cd2, eh, th)));
"""


def oracle() -> dict:
    return fixture("codegen_control", ORACLE)


def oracle_track() -> dict:
    return fixture("codegen_control_track", ORACLE_TRACK)


def oracle_extra() -> dict:
    return fixture("codegen_control_extra", ORACLE_EXTRA)


def oracle_high() -> dict:
    return fixture("codegen_control_high", ORACLE_HIGH)


@pytest.fixture(scope="module")
def ref() -> dict:
    try:
        return oracle()
    except OracleUnavailable as exc:
        pytest.skip(f"MATLAB referansı yok: {exc}")


HOST_CC = tc.find_host_cc()
HOST_CXX = tc.find_host_cxx()
needs_host = pytest.mark.skipif(HOST_CC is None, reason="ana makine C derleyicisi yok")
needs_cxx = pytest.mark.skipif(HOST_CXX is None, reason="ana makine C++ derleyicisi yok")
needs_cross = pytest.mark.skipif(tc.find_arm_gcc() is None and tc.find_clang() is None,
                                 reason="ARM çapraz derleyici yok")


def _v(x) -> np.ndarray:
    return np.asarray(x, dtype=float).ravel()


def _m(x, rows: int, cols: int) -> np.ndarray:
    """MATLAB JSON'u (satır listesi, düz sütun / satır vektörü ya da sayı) → rows × cols."""
    return np.asarray(x, dtype=float).reshape(rows, cols)


def _close(got: np.ndarray, want: np.ndarray, rel: float = 1e-11) -> float:
    got, want = np.asarray(got, float), np.asarray(want, float).reshape(np.shape(got))
    err = float(np.max(np.abs(got - want)))
    scale = max(float(np.max(np.abs(want))), 1.0)
    assert err <= rel * scale, f"en büyük fark {err:.3e} (ölçek {scale:.3g}, izin {rel * scale:.1e})"
    return err


def _both(spec, x, want, rel=1e-11):
    """C (double, ana makine) ve Python başvurusu MATLAB'e karşı; C ile Python bit düzeyinde aynı."""
    py = cc.simulate(spec, x)
    _close(py, want, rel)
    if HOST_CC is not None:
        c_out = cc.run_generated(spec, x).out
        np.testing.assert_array_equal(c_out, py)
    return py


# ============================================================================ A: pid / pidstd / pid2 nesneleri
@needs_host
def test_pid_object_lsim(ref):
    e = _v(ref["e"])
    x = np.column_stack([e, np.zeros_like(e)])
    for case in ref["pid"]:
        spec = cc.PidSpec(Kp=case["Kp"], Ki=case["Ki"], Kd=case["Kd"], Tf=case["Tf"], Ts=ref["Ts"],
                          i_formula=case["IF"], d_formula=case["DF"])
        _both(spec, x, _v(case["out"]).reshape(-1, 1))


@needs_host
def test_pidstd_and_pid2(ref):
    s = ref["pidstd"]
    spec = cc.PidSpec.standard(s["Kp"], s["Ti"], s["Td"], s["N"], Ts=ref["Ts"], i_formula=s["IF"], d_formula=s["DF"])
    np.testing.assert_allclose([spec.Kp, spec.Ki, spec.Kd, spec.Tf], _v(s["par"]), rtol=1e-14)
    e = _v(ref["e"])
    _both(spec, np.column_stack([e, 0 * e]), _v(s["out"]).reshape(-1, 1))
    p = ref["pid2"]
    spec2 = cc.PidSpec(Kp=p["Kp"], Ki=p["Ki"], Kd=p["Kd"], Tf=p["Tf"], b=p["b"], c=p["c"], Ts=ref["Ts"],
                       i_formula=p["IF"], d_formula=p["DF"], two_dof=True)
    _both(spec2, np.column_stack([_v(ref["r"]), _v(ref["y"])]), _v(p["out"]).reshape(-1, 1))


def test_pid_unrealizable_derivative(ref):
    """Tf = 0 ileri Euler (nedensel değil), Tf = 0 yamuk (z = −1 kutbu), Tf ≤ Ts/2 ileri Euler (kararsız)."""
    msgs = ref["pid_errors"]
    for (tf, f), msg in zip(((0.0, "ForwardEuler"), (0.0, "Trapezoidal"), (0.004, "ForwardEuler")), msgs):
        with pytest.raises(ControlError):
            cc.PidSpec(Kp=1, Ki=1, Kd=0.1, Tf=tf, Ts=ref["Ts"], d_formula=f).validate()
        assert isinstance(msg, str)          # MATLAB'in tepkisi belgelenir (fixture'da)


# ============================================================================ C: Simulink Discrete PID Controller
def _simulink_spec(case: dict, Ts: float) -> cc.PidSpec:
    lim = bool(case["sat"])
    aw = ("none", "back-calculation", "clamping")[int(case["aw"])]
    kw = dict(Ts=Ts, i_formula=case["IF"], d_formula=case["DF"], b=case["b"], c=case["c"],
              u_min=case["LL"] if lim else -np.inf, u_max=case["UL"] if lim else np.inf,
              anti_windup=aw if lim else "none", Kb=case["Kb"], Kt=case["Kt"], tracking=bool(case["tracking"]),
              two_dof=True)
    if int(case["form"]) == 2:   # Simulink ideal: P·(1 + I·IF + D·N/(1 + N·DF)), filtre zaman sabiti 1/N
        P, I, D, N = case["P"], case["I"], case["D"], case["N"]
        kw["Kb"], kw["Kt"] = P * case["Kb"], P * case["Kt"]
        return cc.PidSpec(Kp=P, Ki=P * I, Kd=P * D, Tf=1.0 / N, form="standard", **kw)
    return cc.PidSpec(Kp=case["P"], Ki=case["I"], Kd=case["D"], Tf=1.0 / case["N"], **kw)


@needs_host
def test_simulink_pid_block(ref):
    x = np.column_stack([_v(ref["r"]), _v(ref["y"]), _v(ref["tr"])])
    checked = 0
    for case in ref["simulink"]:
        assert not case["err"], case["err"]
        spec = _simulink_spec(case, ref["Ts"])
        want = _v(case["out"]).reshape(-1, 1)
        xin = x if case["tracking"] else x[:, :2]
        _both(spec, xin, want, rel=1e-10)
        checked += 1
    assert checked == len(ref["simulink"]) >= 14


@needs_host
def test_simulink_clamping_with_tracking():
    """Sıkıştırmanın işaret denetimi integratör girişinin tamamıyla (Ki·e + Kt·(tr − sat(a))) yapılır: yalnız Ki·e
    ile yapılsaydı bu durumlarda fark 1.5 olurdu (ölçüldü)."""
    try:
        t = oracle_track()
    except OracleUnavailable as exc:
        pytest.skip(f"MATLAB referansı yok: {exc}")
    x = np.column_stack([_v(t["r"]), _v(t["y"]), _v(t["tr"])])
    for c in t["cases"]:
        spec = cc.PidSpec(Kp=c["P"], Ki=c["I"], Kd=c["D"], Tf=1.0 / c["N"], Ts=t["Ts"], i_formula=c["IF"],
                          d_formula=c["DF"], b=c["b"], c=c["c"], u_min=c["LL"], u_max=c["UL"], anti_windup="clamping",
                          Kt=c["Kt"], tracking=True, two_dof=True)
        _both(spec, x, _v(c["out"]).reshape(-1, 1), rel=1e-12)


# ============================================================================ D: genel LTI
@needs_host
def test_lti_tf_sos(ref):
    """tf1: 5. derece, z = 1'de integratör kutbu + birim çembere yakın karmaşık çift (Tustin). SOS kökten kurulur:
    katsayıyla tanımlı sisteme göre kök koşulu ~1e-9 fark verir (60 basamaklı başvuruyla ölçüldü: MATLAB lsim
    1.4e-11, SOS 1.4e-9) — sözleşmedeki 1e-8 sınırında. SISO "ss" de aynı bölümlerin kademesidir (eşlik biçimi
    değil: o, float'ta ~1000× kötüydü), bu yüzden aynı tolerans."""
    e = _v(ref["e"]).reshape(-1, 1)
    for key, rel in (("tf1", 1e-8), ("tf2", 1e-12)):
        t = ref[key]
        want = _v(t["out"]).reshape(-1, 1)
        spec = cc.LtiSpec.from_tf(_v(t["num"]), _v(t["den"]), ref["Ts"])
        assert cc._lti_realization(spec)["mode"] == "sos"
        _both(spec, e, want, rel=rel)
        ss_spec = cc.LtiSpec.from_tf(_v(t["num"]), _v(t["den"]), ref["Ts"], realization="ss")
        _close(cc.simulate(ss_spec, e), want, rel)
        if key == "tf2":     # SISO ss nesnesi (otomatik): kutuplar özdeğer, sıfırlar Rosenbrock → bölümler
            obj = cc.LtiSpec.from_ss(*cc._sos_to_ss(cc._lti_realization(spec)["sos"]), ref["Ts"])
            assert cc._lti_realization(obj)["mode"] == "sos"
            _both(obj, e, want, rel=1e-12)
        if key == "tf1":     # SISO "ss" = bölümlerin kademesi (eşlik biçimi değil): float'ta SOS kadar doğru
            err_sos = np.max(np.abs(cc.run_generated(spec, e, real="float").out - want))
            err_ss = np.max(np.abs(cc.run_generated(ss_spec, e, real="float").out - want))
            assert err_sos < 5e-4 and err_ss < 5e-4, (err_sos, err_ss)


@needs_host
def test_lti_mimo_ss(ref):
    s = ref["ss1"]
    spec = cc.LtiSpec.from_ss(_m(s["A"], 3, 3), _m(s["B"], 3, 2), _m(s["C"], 2, 3), _m(s["D"], 2, 2), ref["Ts"])
    x = np.column_stack([_v(ref["r"]), _v(ref["y"])])
    _both(spec, x, _m(s["out"], len(x), 2))


@pytest.fixture(scope="module")
def ref_high() -> dict:
    try:
        return oracle_high()
    except OracleUnavailable as exc:
        pytest.skip(f"MATLAB referansı yok: {exc}")


def _h2_zpk(Th: float):
    """h2'nin sıfır / kutup / kazancı: MATLAB c2d(zpk, 'tustin') = bilinear_zpk (denetimde 2.4e-14 ile doğrulandı)."""
    from scipy import signal

    wc, nb = 2 * np.pi * 200, 10
    pb = wc * np.exp(1j * np.pi * (2 * np.arange(1, nb // 2 + 1) + nb - 1) / (2 * nb))
    return signal.bilinear_zpk(np.array([-10.0]), np.r_[0.0, pb, np.conj(pb)], 30 * wc ** nb, 1 / Th)


@needs_host
def test_high_order_siso_realizations(ref_high):
    """SISO 'durum uzayı' eskiden eşlik biçimiydi (float'ta 6. derecede 11.7× hata, 11. derecede ıraksama); şimdi
    bölümlerin kademesi. tf katsayıları sistemi belirlemiyorsa (h2: MATLAB'in kendi lsim(tf)'i bile %25 sapıyor)
    üretim reddedilir; aynı kontrolcü zpk ya da SISO ss nesnesi olarak bölümlerden doğru üretilir (polinom yok)."""
    eh, Th = _v(ref_high["eh"]).reshape(-1, 1), ref_high["Th"]
    h1, h2 = ref_high["h1"], ref_high["h2"]

    def rel(y, want):
        return float(np.max(np.abs(np.ravel(y) - np.ravel(want))) / np.max(np.abs(want)))

    want1 = _v(h1["y_zpk"])
    for rz in ("sos", "ss"):
        spec = cc.LtiSpec.from_tf(_v(h1["num"]), _v(h1["den"]), Th, realization=rz)
        real = cc._lti_realization(spec)
        assert real["mode"] == rz
        if rz == "ss":       # blok alt üçgen: float32 yuvarlamasıyla da kararlı / sınırda (eşlik biçiminde değil)
            assert np.max(np.abs(np.linalg.eigvals(real["A"].astype(np.float32).astype(float)))) <= 1 + 1e-7
        assert rel(cc.run_generated(spec, eh).out, want1) < 1e-9
        assert rel(cc.run_generated(spec, eh, real="float").out, want1) < 2e-3
        assert cc.float_check(spec)[0] is None
    want2 = _v(h2["y_zpk"])
    assert rel(_v(h2["y_tf"]), want2) > 0.05      # tf katsayıları kontrolcüyü temsil etmiyor (MATLAB'de de: 0.6 s'de
    #                                               %6, 2 s'de %25 — kutup yarıçapı 1.000395, kararsız)
    for rz in ("auto", "sos", "ss"):
        with pytest.raises(ControlError, match="kötü koşullu"):
            cc.generate(cc.LtiSpec.from_tf(_v(h2["num"]), _v(h2["den"]), Th, realization=rz),
                        cc.ControlCodegenOptions(real="double"))
    zd, pd, kd = _h2_zpk(Th)
    zspec = cc.controller_spec(SimpleNamespace(z=zd, p=pd, k=kd, Ts=Th))
    assert zspec.is_zpk and cc._lti_realization(zspec)["mode"] == "sos"
    for rz in ("auto", "ss"):
        s = replace(zspec, realization=rz)
        assert rel(cc.run_generated(s, eh).out, want2) < 1e-12
        assert rel(cc.run_generated(s, eh, real="float").out, want2) < 1e-4
        assert cc.float_check(s)[0] is None
    # SISO ss nesnesi (11 durum): 10 katlı z = −1 sıfırı özdeğer olarak ~eps^(1/10) saçılır → bölümler 1e-7'de
    # doğrulanamaz → modelin kendi (blok üçgen) gerçeklemesi korunur; polinoma geçilmez
    A, B, C, D = cc._sos_to_ss(cc._lti_realization(zspec)["sos"])
    ss_spec = cc.LtiSpec.from_ss(A, B, C, D, Th)
    assert cc._lti_realization(ss_spec)["mode"] == "ss"
    assert rel(cc.run_generated(ss_spec, eh).out, want2) < 1e-11
    assert rel(cc.run_generated(ss_spec, eh, real="float").out, want2) < 1e-4


def test_tf_conditioning_and_float_policy():
    """tf'den bölümler: sapma katsayıların kendi koşulluluğuyla açıklanıyorsa (≤ 1e-5) görünür uyarıyla kabul, değilse
    red. float32 yuvarlaması kararlı kutbu birim çembere taşıyorsa float üretimi reddedilir (double serbest)."""
    from scipy import signal

    Th = 1e-4
    z = np.r_[-5, -20, np.roots([1, 2 * 0.05 * 2000, 2000 ** 2])]
    p = np.r_[0, -1000, np.roots([1, 2 * 0.7 * 1500, 1500 ** 2]), np.roots([1, 2 * 0.5 * 2000, 2000 ** 2])]
    zd, pd, kd = signal.bilinear_zpk(z, p, 80.0, 1 / Th)
    spec = cc.LtiSpec.from_tf(np.real(np.poly(zd)) * kd, np.real(np.poly(pd)), Th)
    with pytest.warns(ControlWarning, match="kötü koşullu"):
        files = cc.generate(spec, cc.ControlCodegenOptions(real="double", prefix="h6"))
    assert "kötü koşullu" in files[-1].text
    e = np.random.default_rng(2).standard_normal(3000)
    ref = signal.sosfilt(signal.zpk2sos(zd, pd, kd), e)
    assert np.max(np.abs(cc.simulate(spec, e[:, None])[:, 0] - ref)) < 1e-6 * np.max(np.abs(ref))
    wc, nb = 2 * np.pi * 200, 8                                 # 9. derece: katsayılar ~5e-3 sapma → red
    pb = wc * np.exp(1j * np.pi * (2 * np.arange(1, nb + 1) + nb - 1) / (2 * nb))
    zd, pd, kd = signal.bilinear_zpk(np.array([-10.0]), np.r_[0.0, pb], 30 * wc ** nb, 1 / Th)
    with pytest.raises(ControlError, match="kötü koşullu"):
        cc.generate(cc.LtiSpec.from_tf(np.real(np.poly(zd)) * kd, np.real(np.poly(pd)), Th),
                    cc.ControlCodegenOptions(real="double"))
    s = cc.LtiSpec.from_zpk([], [0.99999, 0.999995], 1e-6, 1e-3)   # float32'de kutup tam z = 1'e kayar
    with pytest.raises(ControlError, match="double seçin"):
        cc.generate(s, cc.ControlCodegenOptions(real="float"))
    assert cc.generate(s, cc.ControlCodegenOptions(real="double"))


def test_sos_relative_degree_and_gain():
    """Gecikmeli (pay derecesi < payda) tf: z⁻ʳ bölümlere dağıtılır; statik kazanç durum içermez."""
    spec = cc.LtiSpec.from_tf([0.5], [1, -0.5, 0, 0], 0.01)        # 0.5 z⁻³ / (1 − 0.5 z⁻¹)
    imp = cc.simulate(spec, np.r_[1.0, np.zeros(9)].reshape(-1, 1))[:, 0]
    np.testing.assert_allclose(imp, [0, 0, 0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125], atol=1e-15)
    g = cc.LtiSpec.gain([[2.0, -1.0]], 0.01)
    assert cc.state_size(g, "float") == 4
    np.testing.assert_allclose(cc.simulate(g, [[1.0, 3.0]]), [[-1.0]])
    with pytest.raises(ControlError):
        cc.LtiSpec.from_tf([1, 2, 3], [1, 0.5], 0.01).validate()          # nedensel değil


# ============================================================================ E: gözlemcili durum geri beslemesi
def _plant(o):
    return _m(o["A"], 2, 2), _m(o["B"], 2, 1), _m(o["C"], 1, 2), _m(o["D"], 1, 1)


@needs_host
def test_reg_delayed(ref):
    o = ref["obs"]
    A, B, C, D = _plant(o)
    K, L = _m(o["K"], 1, 2), _m(o["L"], 2, 1)
    y = _v(ref["y"]).reshape(-1, 1)
    _both(cc.ObserverSpec(A, B, C, D, K, L, ref["Ts"]), y, _v(o["reg_delayed"]).reshape(-1, 1))
    spec_d = cc.ObserverSpec(A, B, C, np.array([[0.1]]), K, L, ref["Ts"])
    _both(spec_d, y, _v(o["reg_delayed_D"]).reshape(-1, 1))
    est = o["estim_delayed"]                      # MATLAB estim matrisleri = kullandığımız denklemler
    np.testing.assert_allclose(_m(est["A"], 2, 2), A - L @ C, atol=1e-14)
    np.testing.assert_allclose(_m(est["B"], 2, 2), np.hstack([B - L @ D, L]), atol=1e-14)


@needs_host
def test_lqg_current_and_delayed(ref):
    o = ref["obs"]
    A, B, C, D = _plant(o)
    K = _m(o["K"], 1, 2)
    y = _v(ref["y"]).reshape(-1, 1)
    Lc, Mc, Ld = _m(o["Lc"], 2, 1), _m(o["Mc"], 2, 1), _m(o["Ld"], 2, 1)
    np.testing.assert_allclose(Lc, A @ Mc, rtol=1e-10)           # N = 0: L = A·Mx (M varsayılan kuralı)
    cur = cc.ObserverSpec(A, B, C, D, K, Lc, ref["Ts"], form="current", M=Mc)
    _both(cur, y, _v(o["lqg_current"]).reshape(-1, 1), rel=1e-10)
    _close(cc.simulate(cc.ObserverSpec(A, B, C, D, K, Lc, ref["Ts"], form="current"), y),
           _v(o["lqg_current"]).reshape(-1, 1), 1e-9)
    _both(cc.ObserverSpec(A, B, C, D, K, Ld, ref["Ts"]), y, _v(o["lqg_delayed"]).reshape(-1, 1), rel=1e-10)


@needs_host
def test_lqi_lqgtrack(ref):
    o = ref["obs"]
    A, B, C, D = _plant(o)
    Kq = _m(o["Klqi"], 1, 3)
    np.testing.assert_allclose(Kq, _m(o["Klqi_aug"], 1, 3), rtol=1e-9)   # xi[n+1] = xi[n] + Ts·(r − y)
    x = np.column_stack([_v(ref["y"]), _v(ref["r"])])                     # bizim düzen: [y, r]
    for form, L, M, key in (("current", o["Lc"], o["Mc"], "track_current"), ("delayed", o["Ld"], None,
                                                                             "track_delayed")):
        spec = cc.ObserverSpec(A, B, C, D, Kq, _m(L, 2, 1), ref["Ts"], form=form,
                               M=None if M is None else _m(M, 2, 1), reference="lqi")
        _both(spec, x, _v(o[key]).reshape(-1, 1), rel=1e-9)


def test_nbar(ref):
    o = ref["obs"]
    A, B, C, D = _plant(o)
    np.testing.assert_allclose(cc.nbar(A, B, C, D, _m(o["K"], 1, 2)), [[o["Nbar"]]], rtol=1e-10)


@pytest.fixture(scope="module")
def ref_extra() -> dict:
    try:
        return oracle_extra()
    except OracleUnavailable as exc:
        pytest.skip(f"MATLAB referansı yok: {exc}")


@needs_host
def test_state_feedback_nbar_lqi(ref_extra):
    """Gözlemcisiz durum geri beslemesi: MATLAB kapalı çevriminin durumları x ve r verilince C'nin çıkışı MATLAB'in
    u'su ile aynı (dlqr + Nbar; lqi integratörü y = C·x + D·u ile, D ≠ 0)."""
    s = ref_extra["sf"]
    A, B, C, D = _m(s["A"], 2, 2), _m(s["B"], 2, 1), _m(s["C"], 1, 2), _m(s["D"], 1, 1)
    r = _v(ref_extra["r"])
    n = len(r)
    x = np.column_stack([_m(s["x"], n, 2), r])
    for nb in (None, [[s["Nbar"]]]):                 # Nbar bizim nbar() ile ya da MATLAB'den
        spec = cc.ObserverSpec.state_feedback(A, B, C, D, _m(s["K"], 1, 2), ref_extra["Ts"], reference="nbar",
                                              Nbar=nb)
        _both(spec, x, _v(s["u"]).reshape(-1, 1), rel=1e-10)
    q = ref_extra["lqi"]
    spec = cc.ObserverSpec.state_feedback(A, B, C, D, _m(q["K"], 1, 3), ref_extra["Ts"], reference="lqi")
    _both(spec, np.column_stack([_m(q["x"], n, 2), r]), _v(q["u"]).reshape(-1, 1), rel=1e-10)
    assert cc.state_size(spec, "float") == 8 and cc._io_dims(spec) == (3, 1)


@needs_host
def test_kalman_noise_feedthrough_my(ref_extra):
    """H ≠ 0 (süreç gürültüsü ölçüme doğrudan giriyor), Nn ≠ 0: ŷ[n|n] = C·x̂[n|n−1] + D·u + My·ν (MATLAB)."""
    k = ref_extra["kfh"]
    A, B, C, D = _m(k["A"], 3, 3), _m(k["B"], 3, 1), _m(k["C"], 2, 3), _m(k["D"], 2, 1)
    M = _m(k["M"], 3, 2)
    My = _m(k["My"], 2, 2) if np.size(k["My"]) else _m(k["kest_D"], 5, 3)[:2, 1:]
    np.testing.assert_allclose(_m(k["kest_D"], 5, 3)[:2, 1:], My, rtol=1e-12, atol=1e-15)
    x = np.column_stack([_v(ref_extra["u"]), _v(ref_extra["y1"]), _v(ref_extra["y2"])])
    want = _m(k["out"], len(x), 5)
    spec = cc.KalmanSpec(A, B, C, D, _m(k["L"], 3, 2), ref_extra["Ts"], M=M, My=My)
    _both(spec, x, want, rel=1e-10)
    plain = cc.simulate(cc.KalmanSpec(A, B, C, D, _m(k["L"], 3, 2), ref_extra["Ts"], M=M), x)
    assert np.max(np.abs(plain[:, :2] - want[:, :2])) > 1e-4          # My = C·M varsayımı burada yanlış olurdu
    np.testing.assert_allclose(plain[:, 2:], want[:, 2:], rtol=1e-10, atol=1e-12)   # x̂[n|n] My'den bağımsız


# ============================================================================ F: Kalman
@needs_host
def test_kalman_current_delayed(ref):
    k = ref["kf"]
    A, B, C, D = _m(k["A"], 3, 3), _m(k["B"], 3, 1), _m(k["C"], 2, 3), _m(k["D"], 2, 1)
    x = np.column_stack([_v(ref["u"]), _v(ref["y"]), _v(ref["y2"])])
    n = len(x)
    cur = cc.KalmanSpec(A, B, C, D, _m(k["Lc"], 3, 2), ref["Ts"], M=_m(k["Mc"], 3, 2), form="current")
    _both(cur, x, _m(k["out_current"], n, 5), rel=1e-10)
    dly = cc.KalmanSpec(A, B, C, D, _m(k["Ld"], 3, 2), ref["Ts"], form="delayed")
    _both(dly, x, _m(k["out_delayed"], n, 5), rel=1e-10)


# ============================================================================ float, NaN, çalışma anı parametreleri
def _specs() -> dict:
    A = np.array([[1, 0.0095], [0, 0.905]])
    B = np.array([[0.00094], [0.19]])
    C = np.array([[1.0, 0.0]])
    return {
        "pid_bc": cc.PidSpec(Kp=3, Ki=8, Kd=0.05, Tf=0.025, Ts=0.01, b=0.7, c=0.3, u_min=-1.2, u_max=1.5,
                             anti_windup="back-calculation", Kb=2, tracking=True),
        "pid_clamp": cc.PidSpec(Kp=3, Ki=8, Kd=0.05, Tf=0.025, Ts=0.01, i_formula="BackwardEuler",
                                d_formula="Trapezoidal", u_min=-1.2, u_max=1.5, anti_windup="clamping"),
        "lti_sos": cc.LtiSpec.from_tf([0.1, 0.05], [1, -1.6, 0.7, -0.05], 0.01),
        "lti_ss": cc.LtiSpec.from_ss(np.diag([0.9, 0.8]), [[1, 0], [0, 0.5]], [[1, 0.5], [0, 1]],
                                     [[0.1, 0], [0, 0.2]], 0.01),
        "obs_nbar": cc.ObserverSpec(A, B, C, [[0.0]], [[10, 1]], [[0.6], [5]], 0.01, reference="nbar",
                                    u_min=[-2.0], u_max=[2.0]),
        "obs_lqi": cc.ObserverSpec(A, B, C, [[0.0]], [[10, 1, -3]], [[0.6], [5]], 0.01, form="current",
                                   reference="lqi"),
        "kf": cc.KalmanSpec(np.diag([0.9, 0.8]), [[0.1], [1]], np.eye(2), [[0], [0.2]], [[0.3, 0], [0, 0.2]], 0.01,
                            M=[[0.3, 0], [0, 0.25]]),
        "obs_state": cc.ObserverSpec.state_feedback(A, B, C, [[0.05]], [[10, 1, -3]], 0.01, reference="lqi",
                                                    u_min=[-2.0], u_max=[2.0]),
        "kf_my": cc.KalmanSpec(np.diag([0.9, 0.8]), None, np.eye(2), None, [[0.3, 0], [0, 0.2]], 0.01,
                               M=[[0.3, 0], [0, 0.25]], My=[[0.35, 0.02], [0.01, 0.3]]),
    }


@needs_host
@pytest.mark.parametrize("name", list(_specs()))
def test_float_and_nan(name):
    spec = _specs()[name]
    nin, _ = cc._io_dims(spec)
    rng = np.random.default_rng(3)
    x = rng.standard_normal((400, nin))
    py = cc.simulate(spec, x)
    np.testing.assert_array_equal(cc.run_generated(spec, x).out, py)
    f32 = cc.run_generated(spec, x, real="float").out
    assert np.max(np.abs(f32 - py)) <= 2e-5 * max(1.0, np.max(np.abs(py)))
    # NaN: durum değişmez, son çıkış tutulur → NaN örneği atlanmış akışla aynı devam
    xn = x.copy()
    xn[100, 0] = np.nan
    yn = cc.run_generated(spec, xn).out
    skip = cc.simulate(spec, np.delete(x, 100, axis=0))
    if isinstance(spec, cc.KalmanSpec):           # Kalman: u sonlu, y'de NaN → yalnız tahmin (atlanmaz)
        np.testing.assert_array_equal(yn, cc.simulate(spec, xn))
        assert np.all(np.isfinite(yn))
    else:
        np.testing.assert_array_equal(yn[100], yn[99])
        np.testing.assert_array_equal(np.delete(yn, 100, axis=0), skip)


_RUNTIME_MAIN = r"""
#include <stdio.h>
#include "spt_pid.h"
int main(void)
{
    spt_pid_t a, b;
    int k, ok1, ok2, ok3, ok4;
    double worst = 0.0;
    volatile double zero = 0.0;
    double nanv = zero / zero;
    spt_pid_init(&a);
    spt_pid_init(&b);
    ok1 = spt_pid_set_gains(&a, 1.0, 2.0, 0.1, 0.004);      /* ileri Euler: Tf <= Ts/2 → reddedilir */
    ok2 = spt_pid_set_gains(&a, 1.5, 0.5, 0.02, 0.03);
    ok3 = spt_pid_set_standard(&b, 1.5, 3.0, 0.02 / 1.5, (0.02 / 1.5) / 0.03);
    ok4 = spt_pid_set_limits(&a, 1.0, -1.0);
    /* NaN Ti / N ya da negatif Td: reddedilir (ok4'e eklenir: hepsi 0 olmalı) */
    ok4 += spt_pid_set_standard(&a, 1.0, nanv, 0.1, 10.0) + spt_pid_set_standard(&a, 1.0, 1.0, 0.1, nanv)
           + spt_pid_set_standard(&a, 1.0, 1.0, -0.1, 10.0);
    for (k = 0; k < 200; ++k) {
        double r = (k > 20) ? 1.0 : 0.0, y = 0.3 * (k % 7) / 7.0;
        double d = spt_pid_step(&a, r, y) - spt_pid_step(&b, r, y);
        if (d < 0) d = -d;
        if (d > worst) worst = d;
    }
    printf("%d %d %d %d %.3e %.17g %.17g\n", ok1, ok2, ok3, ok4, worst, a.ki, a.tf);
    return 0;
}
"""


@needs_host
def test_runtime_gain_change():
    spec = cc.PidSpec(Kp=2, Ki=1, Kd=0.05, Tf=0.02, Ts=0.01)
    files = {f.name: f.text for f in cc._sources(spec, "double")}
    files["main.c"] = _RUNTIME_MAIN
    b = tc.host_build(files, ["spt_pid.c", "main.c"], defines=["SPT_USE_DOUBLE=1"],
                      extra=["-std=c99", "-Wall", "-Wextra", "-pedantic", "-Werror"], output="rt")
    assert b.ok, b.log
    ok1, ok2, ok3, ok4, worst, ki, tf = tc.run_program(b, []).stdout.split()
    assert (ok1, ok2, ok3, ok4) == ("0", "1", "1", "0")
    assert float(worst) < 1e-12 and float(ki) == 0.5 and float(tf) == 0.03


# ============================================================================ çapraz derleme, ayak izi, maliyet
CORTEX = [k for k, p in MCU_PROFILES.items() if p.measurable]


@needs_cross
@pytest.mark.parametrize("name", list(_specs()))
def test_cross_compile_warning_free(name):
    """Tüm Cortex-M profilleri × GCC / Clang × float / double × -O0 / -Os / -O2: -Werror ile uyarısız."""
    spec = _specs()[name]
    jobs = [(p, comp, real, opt) for p in CORTEX for comp in ("gcc", "clang") for real in ("float", "double")
            for opt in ("-O0", "-Os", "-O2") if tc.cross_flags(MCU_PROFILES[p], comp) is not None]
    assert jobs

    def one(job):
        p, comp, real, opt = job
        return job, cc.build_check(spec, MCU_PROFILES[p], comp, real, opt)

    with ThreadPoolExecutor(max_workers=6) as ex:
        bad = [(job, b.log[:800]) for job, b in ex.map(one, jobs) if not b.ok]
    assert not bad, bad[:3]


@needs_cross
@needs_cxx
def test_footprint_and_cost():
    spec = _specs()["pid_bc"]
    opts = cc.ControlCodegenOptions(mcu="cortex-m4f", real="float", prefix="irtifa")
    rows = cc.footprint(spec, opts, ("gcc", "clang", "iar"))
    by = {r.compiler: r for r in rows}
    for comp in ("gcc", "clang"):
        if tc.cross_flags(MCU_PROFILES["cortex-m4f"], comp) is not None:
            assert by[comp].measured and 100 < by[comp].flash_code < 3000
            assert by[comp].ram_state == cc.state_size(spec, "float") == 56
            # kod sütunu iki derleyicide de .text+.rodata+.data; clang'ın .ARM.exidx'i ayrı belirtilir
            files = {f.name: f.text for f in cc._sources(spec, "float")}
            b = tc.compile_cross(files, ["spt_pid.c"], MCU_PROFILES["cortex-m4f"], comp, "-Os",
                                 defines=["SPT_USE_DOUBLE=0"], warnings=())
            ex = cc._exidx_bytes(b.objects["spt_pid.c"])
            assert by[comp].flash_code == b.sizes("spt_pid.c").flash - ex
            assert (".ARM.exidx" in by[comp].method) == (ex > 0)
    assert not by["iar"].measured
    costs = cc.cost(spec, opts)
    assert len(costs) == 2 and all(c.cycles > 20 and c.ops["mul"] > 5 and c.ops["div"] == 0 for c in costs)
    files = cc.generate(spec, opts, rows, costs)
    md = files[-1].text
    assert files[-1].name == "KULLANIM_irtifa.md" and "Ölçülen ayak izi" in md and "ölçüldü" in md


@needs_cxx
@pytest.mark.parametrize("name", ["lti_sos", "obs_lqi", "kf"])
def test_cost_other_kinds(name):
    spec = _specs()[name]
    rows = cc.cost(spec, cc.ControlCodegenOptions(mcu="cortex-m0plus", real="float"))
    assert rows and rows[0].cycles > 0 and rows[0].cpu_load_percent > 0


# ============================================================================ önek, dosyalar, nesneden tanıma
def test_prefix_rules_and_files():
    for bad in ("_x", "a__b", "x_", "1abc", "int8", "Elev", "sig", "isx", "strx", "toolongprefix123", "",
                "lc", "fp", "time", "atomic", "math"):          # LC_…, FP_…, TIME_…: <ÖNEK>_ ile sınanır
        with pytest.raises(ControlError):
            cc.check_prefix(bad)
    for good in ("roll", "Pitch2", "alt_hold", "ctl"):
        cc.check_prefix(good)
    for name, spec in _specs().items():
        files = cc.generate(spec, cc.ControlCodegenOptions(prefix="roll"))
        names = [f.name for f in files]
        assert names == [f"roll_{spec.kind}.h", f"roll_{spec.kind}.c", f"roll_{spec.kind}_ornek.c", "KULLANIM_roll.md"]
        for f in files[:3]:
            assert "spt_" not in f.text and "SPT_" not in f.text, (name, f.name)
            assert max(len(line) for line in f.text.splitlines()) <= 120, (name, f.name)


@needs_host
@pytest.mark.parametrize("name", list(_specs()))
def test_delivered_files_compile(name):
    """Kullanıcıya verilen dosyalar (uzun önek → satır kaydırma) olduğu gibi -Werror ile derlenir."""
    spec = _specs()[name]
    files = cc.generate(spec, cc.ControlCodegenOptions(prefix="rollpitchyaw", real="float"))
    for f in files[:3]:
        assert max(len(line) for line in f.text.splitlines()) <= 120, f.name
    src = {f.name: f.text for f in files[:3]}
    src["main.c"] = "int main(void) { return 0; }\n"
    b = tc.host_build(src, [files[1].name, files[2].name, "main.c"],
                      extra=["-std=c99", "-Wall", "-Wextra", "-pedantic", "-Werror", "-Wdouble-promotion",
                             "-Wshadow", "-Wfloat-conversion"], output="delivered")
    assert b.ok, b.log


def test_controller_spec_adapters():
    pid_obj = SimpleNamespace(Kp=2.0, Ki=1.0, Kd=0.1, Tf=0.02, Ts=0.01, IFormula="Trapezoidal",
                              DFormula="BackwardEuler")
    s = cc.controller_spec(pid_obj, u_min=-1, u_max=1, anti_windup="clamping")
    assert isinstance(s, cc.PidSpec) and s.i_formula == "Trapezoidal" and s.anti_windup == "clamping"
    std = cc.controller_spec(SimpleNamespace(Kp=2.0, Ti=0.5, Td=0.1, N=10.0, Ts=0.01))
    assert std.form == "standard" and np.isclose(std.Ki, 4.0) and np.isclose(std.Kd, 0.2) and np.isclose(std.Tf, 0.01)
    ss_obj = SimpleNamespace(A=[[0.5]], B=[[1.0]], C=[[1.0]], D=[[0.0]], Ts=0.1)
    assert isinstance(cc.controller_spec(ss_obj), cc.LtiSpec)
    tf_obj = SimpleNamespace(num=[[np.array([1.0, 0.5])]], den=[[np.array([1.0, -0.5])]], Ts=0.1, io_delay=[[2]])
    t = cc.controller_spec(tf_obj)
    imp = cc.simulate(t, np.r_[1.0, np.zeros(5)].reshape(-1, 1))[:, 0]
    np.testing.assert_allclose(imp, [0, 0, 1.0, 1.0, 0.5, 0.25], atol=1e-15)     # 2 örnek gecikme
    with pytest.raises(ControlError):
        cc.controller_spec(SimpleNamespace(Kp=1.0, Ki=1.0, Ts=0.0))                # sürekli
    with pytest.raises(ControlError):
        cc.controller_spec(np.array([[1.0, 2.0]]))                                 # kazançta Ts yok
    assert isinstance(cc.controller_spec(np.array([[1.0, 2.0]]), Ts=0.01), cc.LtiSpec)


def test_real_lti_classes_if_available():
    """Eş zamanlı yazılan spt.control.lti / classical modülleri varsa gerçek nesnelerle uçtan uca."""
    lti = pytest.importorskip("spt.control.lti")
    G = lti.tf([0.1, 0.05], [1, -1.6, 0.7, -0.05], 0.01)
    spec = cc.controller_spec(G)
    assert isinstance(spec, cc.LtiSpec)
    classical = pytest.importorskip("spt.control.classical")
    C = classical.pid(2.0, 1.0, 0.1, 0.02, 0.01)
    assert isinstance(cc.controller_spec(C), cc.PidSpec)
    rng = np.random.default_rng(4)
    e = rng.standard_normal((300, 1))
    # tasarım kütüphanesinin aktarım fonksiyonu ile bizim PID yapımız aynı doğrusal sistem
    for obj in (classical.pid(2.0, 1.0, 0.1, 0.02, 0.01, IFormula="Trapezoidal", DFormula="BackwardEuler"),
                classical.pidstd(1.2, 0.8, 0.1, 8.0, 0.02, IFormula="BackwardEuler", DFormula="Trapezoidal")):
        spec = cc.controller_spec(obj)
        tf_spec = cc.LtiSpec.from_tf(obj.num[0][0], obj.den[0][0], obj.Ts, realization="ss")
        _close(cc.simulate(spec, e), cc.simulate(tf_spec, e), 1e-10)
    C2 = classical.pid2(2.0, 1.5, 0.3, 0.05, 0.6, 0.2, 0.01, IFormula="Trapezoidal", DFormula="BackwardEuler")
    spec2 = cc.controller_spec(C2)
    assert spec2.two_dof and (spec2.b, spec2.c) == (0.6, 0.2)
    ry = rng.standard_normal((300, 2))
    want = sum(cc.simulate(cc.LtiSpec.from_tf(C2.num[0][j], C2.den[0][j], 0.01, realization="ss"), ry[:, [j]])
               for j in range(2))
    _close(cc.simulate(spec2, ry), want, 1e-10)


def test_modern_objects_adapter():
    """spt.control.modern sonuçları (reg, kalman demeti, lqg izleyici) → C tanımı; lsim ile aynı cevap."""
    lti = pytest.importorskip("spt.control.lti")
    modern = pytest.importorskip("spt.control.modern")
    timeresp = pytest.importorskip("spt.control.timeresp")
    A = np.array([[1, 0.0095], [0, 0.905]])
    B = np.array([[0.00094], [0.19]])
    C = np.array([[1.0, 0.0]])
    sysd = lti.ss(A, B, C, [[0.0]], 0.01)
    K = modern.dlqr(A, B, np.diag([100.0, 1.0]), 1.0)[0]
    L = np.array([[0.6], [5.0]])
    rng = np.random.default_rng(7)
    y = rng.standard_normal((200, 1))
    R = modern.reg(sysd, K, L)
    want = np.asarray(timeresp.lsim(R, y)[0], float).reshape(-1, 1)
    spec = cc.controller_spec(R)
    assert isinstance(spec, cc.LtiSpec)
    _close(cc.simulate(spec, y), want, 1e-9)
    _close(cc.simulate(cc.ObserverSpec(A, B, C, [[0.0]], K, L, 0.01), y), want, 1e-9)     # reg = 'delayed'
    kr = modern.kalman(lti.ss(A, np.hstack([B, B]), C, [[0.0, 0.0]], 0.01), 0.1, 1e-3)
    uy = rng.standard_normal((200, 2))
    want_k = np.asarray(timeresp.lsim(kr.kest, uy)[0], float).reshape(200, -1)
    _close(cc.simulate(cc.controller_spec(kr), uy), want_k, 1e-9)                     # demet → kestirici modeli
    _close(cc.simulate(cc.KalmanSpec(A, B, C, [[0.0]], kr.L, 0.01, M=kr.Mx, My=kr.My), uy), want_k, 1e-9)
    _close(cc.simulate(cc.KalmanSpec(A, B, C, [[0.0]], kr.L, 0.01, M=kr.Mx), uy), want_k, 1e-9)   # H = 0: My = C·Mx
    G = modern.lqg(sysd, np.diag([100.0, 1.0, 1.0]), np.diag([0.1, 0.1, 1e-3]), QI=[[50.0]])
    spec_g = cc.controller_spec(G)
    assert isinstance(spec_g, cc.LtiSpec) and spec_g.shape == (1, 2) and not spec_g.use_sos()
    ry = rng.standard_normal((200, 2))
    _close(cc.simulate(spec_g, ry), np.asarray(timeresp.lsim(G, ry)[0], float).reshape(-1, 1), 1e-9)


# ============================================================================ pencere (ekransız)
def test_dialog_offscreen(qapp_offscreen):
    from spt.gui.dialogs.control_code_dialog import ControlCodeDialog, open_control_code_dialog, prefix_from_name

    assert prefix_from_name("İrtifa PID") == "irtifa"
    assert prefix_from_name("Yalpa açısı kontrolcü") == "yalpa_acisi"
    assert prefix_from_name("123") == "ctl"
    assert prefix_from_name("Yükseklik süzgeci") == "yukseklik"
    assert prefix_from_name("Çokuzunbirkontrolcuadi") == "cokuzunbirko"
    dlg = ControlCodeDialog(None)
    assert dlg.set_controller(_specs()["pid_bc"], "Yunuslama")
    assert dlg.prefix.text() == "yunuslama" and dlg.limit.isChecked()
    dlg.aw.setCurrentIndex(2)
    dlg.refresh(measure=False)
    assert [f.name for f in dlg.files][-1] == "KULLANIM_yunuslama.md"
    assert dlg.current_spec().anti_windup == "clamping"
    assert dlg.set_controller(_specs()["lti_sos"], "Süzgeç")
    assert not dlg.limit.isVisibleTo(dlg) and dlg.realization.isVisibleTo(dlg)
    dlg.refresh(measure=False)
    assert dlg.files[0].name == "suzgec_lti.h"
    assert dlg.set_controller(_specs()["obs_state"], "Durum geri besleme")
    assert dlg.options().fs_hz is None                  # kutudaki hız = 1/Ts → "farklı hız" uyarısı yok
    dlg.refresh(measure=False)
    assert dlg.files[0].name == "durum_geri_obs.h" and "u = −K·x" in dlg.files[0].text
    assert not dlg.set_controller(SimpleNamespace(Kp=1.0, Ki=1.0, Ts=0.0), "sürekli")
    dlg.close()
    d2 = open_control_code_dialog(None, _specs()["kf"], "Kestirici")
    assert d2.files and d2.files[0].name == "kestirici_kf.h"
    d2.shutdown()
    d2.close()
