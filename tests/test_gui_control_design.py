"""Kontrolcü Tasarımı sekmesi (Aşama 4) arayüz testleri (ekransız), MATLAB R2025b referansıyla.

- Kök yer eğrisi: kazanç değişimi kapalı çevrim kutuplarını MATLAB `pole(feedback(K·C0·G, 1))` ile aynı yere taşır;
  kapalı çevrim kutbunu sürükleme (rlocfind) kazancı MATLAB `rlocfind` ile aynı verir; eğri verisi kütüphanenin
  `rlocus` çıktısıdır; fare sürüklemesi (görünüm kutusu olayı) aynı yoldan geçer.
- Bode düzenleyicisi: genliği +6.02 dB sürüklemek kazancı tam 2 kat yapar; paylar MATLAB `margin` ile aynı.
- PID ayarlayıcı: kazançlar kütüphanenin `pidtune` çıktısıyla birebir; kapalı çevrim faz payı / geçiş frekansı hedefi
  tutar ve MATLAB `margin` aynı kazançlarla aynı payı ölçer (sürekli ve ayrık); grafik verisi `step(feedback(C·G))`.
- Ayrık tasarım (c2d), kademeli (iç / dış çevrim) mimari, LQR / dlqr / lqrd / LQI / place / Kalman kazançları ve MPC
  benzetimi MATLAB ile karşılaştırılır; C kodu diyalogları açılır; çalışma alanına ekleme ve ayar kaydı.
"""
import math
import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPointF, Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

# Not: pidtune'un (MATLAB'in algoritması yayımlanmamış) kazançları aşağıya spt'nin kendi pidtune çıktısı olarak tam
# hassasiyetle yazıldı; MATLAB bu kazançlarla kapalı çevrimin payını kendisi ölçer (tanımın sağlandığının kanıtı).
MATLAB_SCRIPT = r"""
G = zpk([], [0 -2 -5], 10);
C0 = zpk(-1, -10, 1);
L = 20*C0*G;
T = feedback(L, 1);
out.cl20 = spt_c(pole(T));
[gm, pm, wcg, wcp] = margin(L);
out.gm = gm; out.pm = pm; out.wcg = wcg; out.wcp = wcp;
S = stepinfo(T);
out.rise = S.RiseTime; out.settling = S.SettlingTime; out.overshoot = S.Overshoot; out.peak = S.Peak;
out.cl50 = spt_c(pole(feedback(50*C0*G, 1)));
[k, p] = rlocfind(C0*G, -0.5);
out.rk_real = k; out.rp_real = spt_c(p);
[k, p] = rlocfind(C0*G, out.cl50(1, 1) + 1i*out.cl50(1, 2));
out.rk50 = k;
Cp = pid(2.2052871162583476, 1.0291251044667231, 1.1814140098290435, 0.050000000000000003);
[gm, pm, wcg, wcp] = margin(Cp*G);
out.pid_gm = gm; out.pid_pm = pm; out.pid_wcp = wcp; out.pid_wcg = wcg;
Sp = stepinfo(feedback(Cp*G, 1));
out.pid_rise = Sp.RiseTime; out.pid_settling = Sp.SettlingTime; out.pid_overshoot = Sp.Overshoot;
Ts = 0.05;
Gd = c2d(G, Ts, 'zoh');
Ld = zpk(0.9, 0.6, 8, Ts)*Gd;
out.dcl = spt_c(pole(feedback(Ld, 1)));
[gm, pm, wcg, wcp] = margin(Ld);
out.dgm = gm; out.dpm = pm; out.dwcg = wcg; out.dwcp = wcp;
Cpd = pid(1.7729858990442453, 0.90612186175236975, 0.86728924962986909, 0.066666666666666666, Ts);
[gm, pm, wcg, wcp] = margin(Cpd*Gd);
out.dpid_gm = gm; out.dpid_pm = pm; out.dpid_wcp = wcp;
G2 = tf(5, [0.05 1]); G1 = tf(1, [1 0]);
C2 = zpk([], [], 2); C1 = zpk(-0.5, [], 3);
T2 = feedback(G2*C2, 1);
Tc = feedback(C1*T2*G1, 1);
out.ccl = spt_c(pole(Tc));
[gm, pm, wcg, wcp] = margin(C1*T2*G1);
out.cpm = pm; out.cwcp = wcp;
GSc = feedback(G1*feedback(G2, C2), C1*C2);
out.cgs_dc = dcgain(GSc);
out.cgs_poles = spt_c(pole(GSc));
A = [0 1; -2 -3]; B = [0; 1]; Cm = [1 0];
sys = ss(A, B, Cm, 0);
K = lqr(sys, diag([10 1]), 0.5);
out.lqr = K;
Nb = 1/dcgain(ss(A - B*K, B, Cm, 0));
Si = stepinfo(ss(A - B*K, B*Nb, Cm, 0));
out.sf_settling = Si.SettlingTime; out.sf_overshoot = Si.Overshoot; out.sf_rise = Si.RiseTime;
sysd = c2d(sys, 0.05);
out.dlqr = lqr(sysd, diag([10 1]), 0.5);
out.lqrd = lqrd(A, B, diag([10 1]), 0.5, 0.05);
out.lqi = lqi(sys, diag([10 1 5]), 0.5);
out.place = place(A, B, [-3+2i, -3-2i]);
[kest, Lk, P] = kalman(ss(A, [B B], Cm, [0 0]), 1, 0.01);
out.kal = Lk;
[kest, Lkd, Pd, Mx] = kalman(ss(sysd.A, [sysd.B sysd.B], Cm, [0 0], 0.05), 1, 0.01);
out.dkal = Lkd; out.dkal_mx = Mx;
mobj = mpc(ss([0 1; 0 -1], [0; 1], [1 0], 0), 0.1, 10, 2);
mobj.MV.Min = -2; mobj.MV.Max = 2; mobj.MV.RateMin = -0.5; mobj.MV.RateMax = 0.5;
[y, t, u] = sim(mobj, 40, 1);
out.mpc_y = y; out.mpc_u = u;
"""
PID_C = (2.2052871162583476, 1.0291251044667231, 1.1814140098290435, 0.050000000000000003)
PID_D = (1.7729858990442453, 0.90612186175236975, 0.86728924962986909, 0.066666666666666666)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def theme(app):
    from spt.gui.theme import ThemeManager

    t = ThemeManager(app)
    t.apply()
    return t


@pytest.fixture(scope="module")
def ref():
    from tools import matlab_oracle as oracle
    try:
        return oracle.fixture("gui_control_design", MATLAB_SCRIPT)
    except oracle.OracleUnavailable as exc:
        pytest.skip(f"MATLAB referansı yok: {exc}")


@pytest.fixture
def tab(app, theme):
    from spt.gui.control.design_tab import DesignTab
    from spt.gui.control.workspace import ModelWorkspace

    ws = ModelWorkspace()
    t = DesignTab(theme, workspace=ws)
    t.resize(1366, 720)
    t.show()
    t.wait_idle()
    yield t
    t.shutdown()
    t.close()
    t.deleteLater()
    app.processEvents()


def pump(app, seconds):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        app.processEvents()
        time.sleep(0.005)


def cplx(v) -> np.ndarray:
    a = np.asarray(v, float).reshape(-1, 2)
    return a[:, 0] + 1j * a[:, 1]


def assert_roots(got, want, rel=1e-8):
    from scipy.optimize import linear_sum_assignment
    got, want = np.asarray(got, complex).ravel(), np.asarray(want, complex).ravel()
    assert got.size == want.size, (got, want)
    D = np.abs(got[:, None] - want[None, :])
    r, c = linear_sum_assignment(D)
    err = np.max(D[r, c] / np.maximum(1.0, np.abs(want[c])))
    assert err <= rel, (err, got, want)


def close(a, b, rel=1e-8):
    if math.isinf(b):
        return math.isinf(a) and (a > 0) == (b > 0)
    return abs(a - b) <= rel * max(1.0, abs(b))


def check_margins(L, m, ref, names=("gm", "pm", "wcg", "wcp"), rel_matlab=1e-3):
    """Pay tablosu değerleri (Gm, Pm, Wcg, Wcp):
    1) tanımı tam sağlar (1e-9): ∠L(jWcg) = −180° (mod 360), Gm = 1/|L(jWcg)|, |L(jWcp)| = 1, Pm = 180° + ∠L(jWcp);
    2) MATLAB R2025b margin ile, MATLAB'in kendi geçiş frekansı hassasiyeti içinde (projede ölçülmüş göreli hata
       ≤ 5e-4 — bkz. test_control_lti_freqresp — bu yüzden 1e-3; bu çevrimde ör. Gm 4.98668 (kesin, faz −180.000000°)
       iken MATLAB 4.98795)."""
    import cmath

    from spt.control.freqresp import freqresp
    gm, pm, wcg, wcp = m["Gm"], m["Pm"], m["Wcg"], m["Wcp"]
    if math.isfinite(wcg):
        hg = complex(freqresp(L, [wcg])[0, 0, 0])
        assert abs(abs(math.degrees(cmath.phase(hg))) - 180.0) < 1e-9
        assert close(gm, 1.0 / abs(hg), 1e-9)
    if math.isfinite(wcp):
        hc = complex(freqresp(L, [wcp])[0, 0, 0])
        assert close(abs(hc), 1.0, 1e-9)
        assert abs(((pm - (180.0 + math.degrees(cmath.phase(hc)))) + 180.0) % 360.0 - 180.0) < 1e-9
    for key, name in zip(("Gm", "Pm", "Wcg", "Wcp"), names):
        if name is not None:
            assert close(m[key], ref[name], rel_matlab), (key, m[key], ref[name])


# MATLAB stepinfo(sys) ölçütleri kendi benzetim ızgarasındaki örneklerden alır (tepe / %10–%90 geçişleri ızgaraya
# düşer); kütüphane aynı tanımı ızgarasız inceltir. Bu çevrimlerde ölçülen fark: yükselme ≤ 1.4e-3, aşım ≤ 2.8e-4,
# tepe / yerleşme ≤ 1.5e-5 (göreli). Arayüz değeri kütüphanenin stepinfo'suyla birebir olmalı.
STEPINFO_TOL = {"RiseTime": 2e-3, "SettlingTime": 1e-4, "Overshoot": 5e-4, "Peak": 1e-4}


def check_stepinfo(sys, info, ref, names):
    from spt.control.timeresp import stepinfo
    lib = stepinfo(sys)
    for key in ("RiseTime", "SettlingTime", "Overshoot", "Peak", "PeakTime"):
        assert close(info[key], lib[key], 1e-12), (key, info[key], lib[key])
    for key, name in names.items():
        assert close(info[key], ref[name], STEPINFO_TOL[key]), (key, info[key], ref[name])


def plant():
    from spt.control import lti
    return lti.zpk([], [0, -2, -5], 10)


def lead_comp(tab, K=20.0):
    def fn(c):
        c.add("real_zero", -1.0)
        c.add("real_pole", -10.0)
    tab.edit_comp(fn)
    tab.set_gain(K)
    tab.wait_idle()


def zcomp(tab, K):
    """Ayrık tasarımda z düzleminde öncül: sıfır 0.9, kutup 0.6."""
    def fn(c):
        c.add("real_zero", 0.9)
        c.add("real_pole", 0.6)
    tab.edit_comp(fn)
    tab.set_gain(K)
    tab.wait_idle()


class FakeDrag:
    """pyqtgraph sürükleme olayının en küçük taklidi (görünüm kutusu işleyicisini gerçek yoldan sınamak için)."""

    def __init__(self, start, pos, phase):
        self._start, self._pos, self._phase = start, pos, phase
        self.accepted = False

    def button(self):
        return Qt.MouseButton.LeftButton

    def isStart(self):
        return self._phase == "start"

    def isFinish(self):
        return self._phase == "finish"

    def buttonDownScenePos(self, btn=None):
        return self._start

    def scenePos(self):
        return self._pos

    def accept(self):
        self.accepted = True

    def ignore(self):
        pass


def drag(plot, x0, y0, x1, y1, steps=4):
    """Görünüm koordinatında (x0, y0) → (x1, y1) sürükleme; olaylar görünüm kutusunun mouseDragEvent'ine gider."""
    vb = plot.getPlotItem().getViewBox()
    a = vb.mapViewToScene(QPointF(x0, y0))
    ok = []
    for i in range(steps + 1):
        f = i / steps
        b = vb.mapViewToScene(QPointF(x0 + f * (x1 - x0), y0 + f * (y1 - y0)))
        phase = "start" if i == 0 else ("finish" if i == steps else "move")
        ev = FakeDrag(a, b, phase)
        vb.mouseDragEvent(ev)
        ok.append(ev.accepted)
    return ok


# ============================================================================ kabuk bağlantısı
def test_shell_loads_design_tab(app, theme):
    from spt.gui.control.design_tab import DesignTab
    from spt.gui.views.control_view import ControlView

    view = ControlView(theme)
    try:
        assert not view.is_placeholder("design")
        page = view.pages["design"]
        assert isinstance(page, DesignTab) and page.workspace is view.workspace
        keys = [k for k, _t, _c in view.detachable_cards()]
        assert "ctl_ds_rlocus" in keys and "ctl_ds_mpc_y" in keys and len(keys) == len(set(keys))
    finally:
        view.shutdown()
        view.deleteLater()
        app.processEvents()


# ============================================================================ kök yer eğrisi
def test_root_locus_gain_moves_closed_loop_poles(tab, ref):
    from spt.control import lti
    from spt.control.analysis import rlocus

    tab.workspace.add("G", plant())
    lead_comp(tab, 20.0)
    assert tab.messages == [], tab.messages
    assert_roots(tab.cl_poles, cplx(ref["cl20"]), 1e-9)
    # işaretler (■) kapalı çevrim kutuplarında
    x, y = tab.classical.rlocus._cl_item.getData()
    assert_roots(x + 1j * y, cplx(ref["cl20"]), 1e-9)
    # eğri: kütüphanenin rlocus çıktısı, L0 = C0·G
    sys = tab._locus_sys
    z, p, k = lti.zpk(sys).zpkdata(squeeze=True)
    assert_roots(z, [-1.0], 1e-12)
    assert_roots(p, [0, -2, -5, -10], 1e-12)
    assert k == pytest.approx(10.0, rel=1e-12)
    r_lib, k_lib = rlocus(sys)
    r_plot, k_plot, _Ts = tab.classical.rlocus.loci("L0")
    assert np.array_equal(k_plot, k_lib) and np.allclose(r_plot, r_lib.T, rtol=0, atol=1e-12)
    # paylar ve stepinfo MATLAB ile
    check_margins(tab.loops.L, tab.margin_values["L"], ref)
    check_stepinfo(tab.loops.T, tab.step_values["T"], ref, {"Overshoot": "overshoot", "Peak": "peak",
                                                            "SettlingTime": "settling", "RiseTime": "rise"})
    # kazanç 50: kutuplar MATLAB'in K = 50 kutuplarına taşınır
    tab.set_gain(50.0)
    tab.wait_idle()
    assert_roots(tab.cl_poles, cplx(ref["cl50"]), 1e-9)


def test_root_locus_drag_uses_rlocfind(tab, ref):
    tab.workspace.add("G", plant())
    lead_comp(tab, 20.0)
    rl = tab.classical.rlocus
    # eğri üzerindeki gerçel nokta: K = −1/(C0·G)(−0.5), MATLAB rlocfind ile aynı
    rl.drag_gain_to(-0.5)
    tab.wait_idle()
    assert tab.comp.K == pytest.approx(ref["rk_real"], rel=1e-9)
    assert_roots(tab.cl_poles, cplx(ref["rp_real"]), 1e-8)
    # MATLAB'in K = 50 kapalı çevrim kutbuna sürükle → K = 50
    target = cplx(ref["cl50"])
    target = target[np.argmax(target.imag)]
    rl.drag_gain_to(target)
    tab.wait_idle()
    assert tab.comp.K == pytest.approx(50.0, rel=1e-7)
    assert tab.comp.K == pytest.approx(ref["rk50"], rel=1e-7)
    assert_roots(tab.cl_poles, cplx(ref["cl50"]), 1e-7)


def test_mouse_drag_on_root_locus(app, tab):
    tab.workspace.add("G", plant())
    lead_comp(tab, 20.0)
    pump(app, 0.2)
    rl = tab.classical.rlocus
    # kapalı çevrim kutbu: gerçel kola (−1 … 0 arası) sürükle
    p0 = complex(tab.cl_poles[np.argmin(np.abs(tab.cl_poles - (-0.9)))])
    acc = drag(rl.plot, p0.real, p0.imag, -0.5, 0.0)
    assert all(acc)
    tab.wait_idle()
    assert np.min(np.abs(tab.cl_poles - (-0.5))) < 1e-6
    # kompansatör kutbunu (−10) −8'e sürükle
    acc = drag(rl.plot, -10.0, 0.0, -8.0, 0.0)
    assert all(acc)
    tab.wait_idle()
    pole = [e for e in tab.comp.elements if e.kind == "real_pole"][0]
    assert pole.poles[0] == pytest.approx(-8.0, abs=0.05)
    assert pole.poles[0].imag == 0.0           # gerçel öğe gerçel eksende kalır
    # tutamaç olmayan yerden (görünümün sağ üst köşesi) sürükleme görünümü kaydırır (olay kabul edilmez)
    (x0, x1), (y0, y1) = rl.plot.getPlotItem().getViewBox().viewRange()
    xa, ya = x1 - 0.05 * (x1 - x0), y1 - 0.05 * (y1 - y0)
    acc = drag(rl.plot, xa, ya, xa - 0.1 * (x1 - x0), ya)
    assert not acc[0]


def test_bode_drag_scales_gain(app, tab, ref):
    tab.workspace.add("G", plant())
    lead_comp(tab, 20.0)
    pump(app, 0.2)
    w, mag, _ph = tab.classical.bode.data("L")
    i = int(np.argmin(np.abs(w - 1.0)))
    x, y = math.log10(w[i]), 20 * math.log10(abs(mag[0, 0, i]))
    acc = drag(tab.classical.bode.mag_card.plot, x, y, x, y + 20 * math.log10(2.0))
    assert all(acc)
    tab.wait_idle()
    assert tab.comp.K == pytest.approx(40.0, rel=1e-9)
    # genlik eğrisi tam 6.0206 dB yukarı
    w2, mag2, _ = tab.classical.bode.data("L")
    assert np.allclose(np.abs(mag2[0, 0]), 2 * np.abs(mag[0, 0]), rtol=1e-12)


# ============================================================================ PID ayarlayıcı
def test_pid_tuner_hits_targets(tab, ref):
    from spt.control.classical import pidtune
    from spt.control.connect import feedback
    from spt.control.freqresp import freqresp
    from spt.control.timeresp import step

    tab.workspace.add("G", plant())
    tab.set_mode("pid")
    tab.wait_idle()
    pp = tab.pid_panel
    assert tab.comp.pid is not None
    pp.type.setCurrentText("PIDF")
    pp.pm.setValue(60.0)
    pp.wc.setValue(2.0)
    tab.wait_idle()
    C = tab.comp.pid
    lib, info = pidtune(plant(), "PIDF", 2.0, PhaseMargin=60.0)
    assert C.parallel() == pytest.approx(lib.parallel(), rel=1e-12)
    assert C.parallel() == pytest.approx(PID_C, rel=1e-9)
    m = tab.margin_values["L"]
    assert m["Pm"] == pytest.approx(60.0, abs=1e-9) and m["Wcp"] == pytest.approx(2.0, rel=1e-10)
    check_margins(tab.loops.L, m, ref, ("pid_gm", "pid_pm", "pid_wcg", "pid_wcp"))
    check_stepinfo(tab.loops.T, tab.step_values["T"], ref, {"Overshoot": "pid_overshoot", "RiseTime": "pid_rise",
                                                            "SettlingTime": "pid_settling"})
    # grafik verisi: kütüphanenin step(feedback(C·G, 1))'i
    T = tab.loops.T
    wv = np.logspace(-2, 2, 30)
    assert np.allclose(freqresp(T, wv), freqresp(feedback(C * plant(), 1), wv), rtol=1e-10, atol=1e-14)
    x, y = tab.classical.step.curve_data("T: r → y")
    yl, tl = step(T, x[-1])
    assert np.allclose(x, tl, rtol=0, atol=1e-12) and np.allclose(y, np.asarray(yl)[:, 0, 0], rtol=0, atol=1e-12)
    # tablo
    rows = {pp.params.item(r, 0).text(): pp.params.item(r, 1).text() for r in range(pp.params.rowCount())}
    assert rows["Kp"] == f"{lib.parallel()[0]:.6g}" and rows["Ölçülen Pm (°)"].startswith("60")
    # kaydırıcı: cevap süresi τ → ωc = 2/τ
    pp.tau_slider.setValue(pp.tau_slider.value() + 100)       # 10 kat yavaş
    tab.wait_idle()
    assert pp.wc.value() == pytest.approx(0.2, rel=1e-3)
    assert tab.margin_values["L"]["Wcp"] == pytest.approx(pp.wc.value(), rel=1e-7)
    assert tab.margin_values["L"]["Pm"] == pytest.approx(60.0, abs=1e-6)
    pp.pm_slider.setValue(30)          # geçici durum θ = 0.3 → Pm = 45°
    tab.wait_idle()
    assert tab.margin_values["L"]["Pm"] == pytest.approx(45.0, abs=1e-6)


def test_pid_discrete(tab, ref):
    from spt.control.classical import pidtune
    from spt.control.discretize import c2d

    tab.workspace.add("G", plant())
    tab.set_domain(0.05, "zoh")
    tab.set_mode("pid")
    tab.wait_idle()
    pp = tab.pid_panel
    pp.type.setCurrentText("PIDF")
    pp.pm.setValue(55.0)
    pp.wc.setValue(1.5)
    tab.wait_idle()
    C = tab.comp.pid
    assert C.Ts == 0.05 and C.IFormula == "ForwardEuler"
    lib, _info = pidtune(c2d(plant(), 0.05), "PIDF", 1.5, PhaseMargin=55.0)
    assert C.parallel() == pytest.approx(lib.parallel(), rel=1e-12)
    assert C.parallel() == pytest.approx(PID_D, rel=1e-9)
    m = tab.margin_values["L"]
    # hedef tam tutar; MATLAB margin aynı kazançlarla aynı payı ölçer (55.00002° @ 1.5000064 rad/s: MATLAB'in
    # geçiş hassasiyeti). Kütüphanenin freqresp.margin'i bu ayrık çevrimde geçişi kaçırır (c = cos ωTs ≈ 1
    # kümelenmesi) — tasarım sekmesi faz payını ayrıca classical.loop_margins ile bulur (design_core.loop_margins).
    assert m["Pm"] == pytest.approx(55.0, abs=1e-9) and m["Wcp"] == pytest.approx(1.5, rel=1e-10)
    check_margins(tab.loops.L, m, ref, ("dpid_gm", "dpid_pm", None, "dpid_wcp"))
    # sürekli eşdeğer gösterilir
    assert "T: sürekli eşdeğer" in tab.classical.step.trace_names()
    assert tab.comp_panel.cont_text.text().startswith("Sürekli eşdeğer")


# ============================================================================ ayrık ve kademeli
def test_discrete_compensator_matches_matlab(tab, ref):
    tab.workspace.add("G", plant())
    tab.set_domain(0.05, "zoh")

    def fn(c):
        c.add("real_zero", 0.9)
        c.add("real_pole", 0.6)
    tab.edit_comp(fn)
    tab.set_gain(8.0)
    tab.wait_idle()
    assert tab.Ts == 0.05 and tab.loops.L.Ts == 0.05
    assert_roots(tab.cl_poles, cplx(ref["dcl"]), 1e-8)
    check_margins(tab.loops.L, tab.margin_values["L"], ref, ("dgm", "dpm", "dwcg", "dwcp"))
    # z düzleminde birim çember ve sürekli eşdeğer eğri
    assert tab.classical.rlocus._discrete
    assert "T: sürekli eşdeğer" in tab.classical.step.trace_names()
    # sürekliye dönüş: kompansatör d2c ile taşınır, kutuplar s düzleminde
    tab.set_domain(0.0)
    tab.wait_idle()
    assert tab.Ts == 0.0 and all(e.zeros[0].real < 0 for e in tab.comp.elements if e.kind == "real_zero")


def test_cascade_inner_outer(tab, ref):
    from spt.control import lti

    tab.workspace.add("G2", lti.tf(5, [0.05, 1]))
    tab.set_arch("cascade")
    tab.set_plant(G="G2", G1="")          # dış bitki 1/s (integratör)
    tab.edit.setCurrentIndex(1)          # iç çevrim C2
    tab.set_gain(2.0)
    tab.edit.setCurrentIndex(0)          # dış çevrim C1
    tab.edit_comp(lambda c: c.add("real_zero", -0.5))
    tab.set_gain(3.0)
    tab.wait_idle()
    assert tab.edit_key() == "C1" and tab.comps["C2"].K == 2.0
    from spt.control.analysis import pole
    from spt.control.freqresp import dcgain
    assert_roots(pole(tab.loops.T), cplx(ref["ccl"]), 1e-9)
    check_margins(tab.loops.L, tab.margin_values["L"], ref, (None, "cpm", None, "cwcp"))
    assert float(np.ravel(dcgain(tab.loops.GS))[0]) == pytest.approx(ref["cgs_dc"], rel=1e-9)
    assert_roots(pole(tab.loops.GS), cplx(ref["cgs_poles"]), 1e-8)
    # iç çevrim düzenlenirken açık çevrim L = C2·G2 (dış çevrim açık) ve T2 eğrisi
    tab.edit.setCurrentIndex(1)
    tab.wait_idle()
    assert tab.loops.L.ninputs == 1 and "T2: iç çevrim ω_ref → ω" in tab.classical.step.trace_names()
    assert_roots(tab.cl_poles, pole(tab.loops.T2), 1e-9)


# ============================================================================ durum uzayı
def test_state_space_designs_match_matlab(tab, ref):
    from spt.control import lti

    A = np.array([[0.0, 1.0], [-2.0, -3.0]])
    B = np.array([[0.0], [1.0]])
    C = np.array([[1.0, 0.0]])
    tab.workspace.add("P", lti.ss(A, B, C, 0))
    tab.set_mode("modern")
    md = tab.modern
    md.set_plant("P")
    md.ctrl.setCurrentIndex(2)           # LQR
    md.Q.setText("10, 1")
    md.R.setText("0.5")
    md.recompute()
    tab.wait_idle()
    assert np.allclose(md.result["K"], np.atleast_2d(ref["lqr"]), rtol=1e-9)
    check_stepinfo(md.result["T"], tab.modern_view.step.info["T"], ref,
                   {"SettlingTime": "sf_settling", "RiseTime": "sf_rise", "Overshoot": "sf_overshoot"})
    md.ctrl.setCurrentIndex(3)           # LQI
    md.Q.setText("10, 1, 5")
    md.recompute()
    assert np.allclose(md.result["K"], np.atleast_2d(ref["lqi"]), rtol=1e-9)
    from spt.control.freqresp import dcgain
    assert float(np.ravel(dcgain(md.result["T"]))[0]) == pytest.approx(1.0, abs=1e-9)   # integral: ofsetsiz
    md.ctrl.setCurrentIndex(0)           # place
    md.poles.setText("-3±2j")
    md.recompute()
    assert np.allclose(md.result["K"], np.atleast_2d(ref["place"]), rtol=1e-9)
    md.est.setCurrentIndex(2)            # Kalman (G = B)
    md.Qn.setText("1")
    md.Rn.setText("0.01")
    md.recompute()
    assert np.allclose(md.result["L"].ravel(), np.ravel(ref["kal"]), rtol=1e-8)
    assert any("kestirici" in n for n in tab.modern_view.pz.trace_names())
    # ayrık: dlqr, lqrd, ayrık Kalman (current: L ve Mx)
    tab.set_domain(0.05, "zoh")
    md.est.setCurrentIndex(0)
    md.ctrl.setCurrentIndex(2)
    md.Q.setText("10, 1")
    md.recompute()
    assert np.allclose(md.result["K"], np.atleast_2d(ref["dlqr"]), rtol=1e-9)
    md.lqrd.setChecked(True)
    md.recompute()
    assert np.allclose(md.result["K"], np.atleast_2d(ref["lqrd"]), rtol=1e-8)
    md.lqrd.setChecked(False)
    md.est.setCurrentIndex(2)
    md.recompute()
    assert np.allclose(md.result["L"].ravel(), np.ravel(ref["dkal"]), rtol=1e-8)
    assert np.allclose(np.ravel(md.result["kf"].Mx), np.ravel(ref["dkal_mx"]), rtol=1e-8)
    # pole metni düzlem değişince taşınır (z = e^{sTs})
    md.ctrl.setCurrentIndex(0)
    tab.set_domain(0.0)
    md.poles.setText("-3±2j")
    tab.set_domain(0.05, "zoh")
    from spt.gui.control import design_core as dc
    zp = dc.parse_roots(md.poles.text())
    assert_roots(zp, np.exp(np.array([-3 + 2j, -3 - 2j]) * 0.05), 1e-10)


# ============================================================================ MPC
def test_mpc_simulation_matches_matlab(tab, ref):
    from spt.control import lti

    tab.workspace.add("M", lti.ss(np.array([[0.0, 1.0], [0.0, -1.0]]), np.array([[0.0], [1.0]]),
                                  np.array([[1.0, 0.0]]), 0))
    tab.set_mode("mpc")
    mp = tab.mpc
    mp.set_plant("M")
    mp.ts.setValue(0.1)
    mp.p.setValue(10)
    mp.m.setValue(2)
    for w, v in ((mp.umin, "-2"), (mp.umax, "2"), (mp.dumin, "-0.5"), (mp.dumax, "0.5"), (mp.ref, "1")):
        w.setText(v)
    mp.steps.setValue(40)
    mp.run_now()
    tab.wait_idle(60)
    res = mp.result
    assert res is not None, tab.messages
    y_ref = np.asarray(ref["mpc_y"], float).reshape(-1)
    u_ref = np.asarray(ref["mpc_u"], float).reshape(-1)
    assert np.allclose(res["y"][:, 0], y_ref, rtol=0, atol=1e-5)
    assert np.allclose(res["u"][:, 0], u_ref, rtol=0, atol=1e-5)
    assert np.all(res["u"] <= 2 + 1e-9) and np.all(np.abs(np.diff(np.r_[0.0, res["u"][:, 0]])) <= 0.5 + 1e-9)
    x, y = tab.mpc_view.y.curve_data("MO1") or tab.mpc_view.y.curve_data(tab.mpc_view.y.trace_names()[0])
    assert np.allclose(y[::2], res["y"][:, 0], atol=1e-12)


# ============================================================================ C kodu, çalışma alanı, ayarlar
def test_code_dialogs_open(app, tab):
    from spt.control import lti

    tab.workspace.add("G", plant())
    tab.set_domain(0.05, "zoh")
    zcomp(tab, 5.0)
    dlg = tab.generate_code()
    assert dlg is not None and dlg.isVisible()
    dlg.close()
    tab.set_mode("pid")
    tab.wait_idle()
    dlg = tab.generate_code()
    assert dlg is not None and dlg.isVisible()
    dlg.close()
    # durum uzayı: gözlemcili LQR ve yalnız Kalman
    tab.workspace.add("P", lti.ss(np.array([[0.0, 1.0], [-2.0, -3.0]]), np.array([[0.0], [1.0]]),
                                  np.array([[1.0, 0.0]]), 0))
    tab.set_mode("modern")
    md = tab.modern
    md.set_plant("P")
    md.ctrl.setCurrentIndex(2)
    md.est.setCurrentIndex(2)
    md.recompute()
    for what in ("ctrl", "kalman"):
        dlg = md.generate_code(what)
        assert dlg is not None and dlg.isVisible(), tab.status.text()
        dlg.close()
    tab.set_mode("mpc")
    tab.mpc.set_plant("P")
    dlg = tab.mpc.generate_code()
    assert dlg is not None and dlg.isVisible()
    dlg.close()
    app.processEvents()


def test_add_to_workspace_and_colors(tab):
    from spt.gui.control import control_plots as cp

    tab.workspace.add("G", plant())
    tab.set_domain(0.05, "zoh")
    tab.set_mode("pid")
    tab.wait_idle()
    tab.store_baseline()
    tab.wait_idle()
    names = tab.add_to_workspace("C")
    assert names and tab.workspace.entry(names[0]).kind == "pid"
    names = tab.add_to_workspace("T")
    assert names and tab.workspace.get(names[0]).Ts == 0.05
    # klasik görünümde her seri ayrı renkte (lejant)
    from PyQt6.QtCore import QCoreApplication, QEvent
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    colors = [lab.text().split("color:")[1][:7] for lab in tab.classical.legend.findChildren(type(tab.status))]
    assert len(colors) == len(set(colors)) >= 6
    assert set(colors) <= set(cp.PALETTE)


def test_save_restore_state(app, theme, tab, tmp_path):
    from PyQt6.QtCore import QSettings
    from spt.gui.control.design_tab import DesignTab

    tab.workspace.add("G", plant())
    tab.set_domain(0.05, "tustin")
    zcomp(tab, 3.0)
    s = QSettings(str(tmp_path / "a.ini"), QSettings.Format.IniFormat)
    tab.save_state(s)
    s.sync()
    t2 = DesignTab(theme, workspace=tab.workspace)
    try:
        t2.restore_state(QSettings(str(tmp_path / "a.ini"), QSettings.Format.IniFormat))
        t2.wait_idle()
        assert t2.Ts == 0.05 and t2.method() == "tustin"
        assert t2.comp.to_dict() == tab.comp.to_dict()
        assert_roots(t2.cl_poles, tab.cl_poles, 1e-12)
    finally:
        t2.shutdown()
        t2.deleteLater()
        app.processEvents()
