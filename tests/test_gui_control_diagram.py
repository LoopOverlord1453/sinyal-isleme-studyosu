"""Blok diyagram düzenleyicisi (Simulink benzeri arayüz) testleri — ekransız.

- Düzenleyici API'siyle PID + bitki geri besleme diyagramı kurulur, arka planda çalıştırılır; skop verisi motorun
  doğrudan `simulate` sonucuyla bit düzeyinde aynıdır ve MATLAB'in kapalı çevrim basamak cevabıyla (step) RK4 hata
  sınırında örtüşür.
- Doğrusallaştırma noktalarıyla çıkarılan kapalı çevrim (sürekli ve ayrık PID) MATLAB feedback(C·G, 1) ile 1e-8.
- .sptd kaydet / aç gidiş-dönüşü, geri al / yinele, kopyala / yapıştır, fareyle bağlama ve tel bölümü sürükleme.
- Bağlanmamış port ve cebirsel döngü hatalarının Türkçe gösterimi (ilgili blok işaretlenir).
- 1366×768 ekran görüntüsünde okunurluk (bileşen boyutları; görüntü geçici klasöre yazılır).
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPointF, Qt  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

MATLAB_SCRIPT = r"""
G = tf(1, [1 2 1]);
C = pid(2, 1, 0.2, 1/50);
T = feedback(C*G, 1);
t = (0:0.01:10)';
out.t = t;
out.y = step(T, t);
out.p = spt_c(esort(pole(T)));
out.dc = dcgain(T);
w = [0.1; 1; 10];
out.w = w;
out.H = spt_c(squeeze(freqresp(T, w)));
Cd = pid(2, 1, 0.2, 1/50, 0.01);
out.iformula = Cd.IFormula;
out.dformula = Cd.DFormula;
Td = feedback(Cd*c2d(G, 0.01, 'zoh'), 1);
out.pd = spt_c(dsort(pole(Td)));
out.dcd = dcgain(Td);
out.Hd = spt_c(squeeze(freqresp(Td, w)));
"""


def _oracle():
    from tools.matlab_oracle import OracleUnavailable, fixture

    try:
        return fixture("gui_control_diagram", MATLAB_SCRIPT)
    except OracleUnavailable as exc:  # pragma: no cover - MATLAB ve kayıtlı referans yoksa
        pytest.skip(str(exc))


def _c(pairs):
    a = np.asarray(pairs, dtype=float).reshape(-1, 2)
    return a[:, 0] + 1j * a[:, 1]


# ============================================================================ ortak
@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def theme(app):
    from spt.gui.theme import ThemeManager

    t = ThemeManager(app)
    t.apply()
    return t


@pytest.fixture
def ed(app, theme, tmp_path, monkeypatch):
    monkeypatch.setenv("SPT_SETTINGS_FILE", str(tmp_path / "settings.ini"))
    from spt.gui.control.diagram_editor import DiagramEditor

    e = DiagramEditor(theme)
    e.resize(1366, 720)
    e.show()
    pump(app, 0.05)
    yield e
    e.stop()
    e.wait(10)
    e.close()
    e.deleteLater()
    pump(app, 0.05)


def pump(app, seconds):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        app.processEvents()
        time.sleep(0.005)


def build_pid_loop(e, solver="ode4", step="0.01"):
    e.add_block("Step", "Referans", pos=(0, 40), step_time=0.0)
    e.add_block("Sum", "Hata", pos=(140, 40), signs="+-")
    e.add_block("PIDController", "PID", pos=(240, 40), P=2.0, I=1.0, D=0.2, N=50.0)
    e.add_block("TransferFcn", "Bitki", pos=(400, 40), num=[1.0], den=[1.0, 2.0, 1.0])
    e.add_block("Scope", "Skop", pos=(620, 40), inputs=2, labels="r, y")
    e.connect("Referans", "Hata")
    e.connect("Hata", "PID")
    e.connect("PID", "Bitki")
    fb = e.connect("Bitki", "Hata", 0, 1)
    e.connect("Referans", "Skop")
    e.branch(fb, "Skop", 1, anchor=0.1)                  # geri besleme telinden dal
    e.solver.setCurrentIndex(e.solver.findData(solver))
    e.stop_edit.setText("10")
    e.step_edit.setText(step)


def model_dict(e):
    d = e.to_model().to_dict()
    d.pop("name", None)
    return d


# ============================================================================ çalıştırma ve motor birebirliği
def test_pid_loop_runs_in_background_and_matches_engine_exactly(ed):
    from spt.control.diagram.engine import simulate

    build_pid_loop(ed)
    assert ed.run(wait=True), ed.status_text
    assert ed.last_error is None and ed.status_state == "good", ed.status_text
    res = ed.last_result
    direct = simulate(ed.to_model().copy())                    # aynı ayarlarla motorun doğrudan sonucu
    assert np.array_equal(res.t, direct.t)
    for a, b in zip(res.scopes["Skop"].data, direct.scopes["Skop"].data):
        assert np.array_equal(a, b)
    # kapsam paneli: tek kart, iki seri, farklı renkler; çizilen veri sonuçla aynı
    assert list(ed.scopes.cards) == ["Skop"]
    traces = ed.scopes.traces["Skop"]
    assert [t.name for t in traces] == ["r", "y"]
    assert len(set(ed.scopes.series_colors())) == len(ed.scopes.series_colors()) == 2
    assert np.array_equal(traces[1].item.yData, res.scopes["Skop"].data[1][:, 0])


def test_scope_data_matches_matlab_step_within_rk4_error(ed):
    out = _oracle()
    build_pid_loop(ed, "ode4", "0.001")
    assert ed.run(wait=True), ed.status_text
    res = ed.last_result
    t_ml = np.asarray(out["t"], dtype=float)
    y_ml = np.asarray(out["y"], dtype=float).ravel()
    idx = np.searchsorted(res.t, t_ml)
    assert np.allclose(res.t[idx], t_ml, rtol=0, atol=1e-12)
    y = res.scopes["Skop"].data[1][idx, 0]
    # RK4 (h = 1 ms) global hatası; PID türev filtresi kutbu −50 → h·50 = 0.05
    assert np.max(np.abs(y - y_ml)) < 1e-7


def test_linearize_closed_loop_matches_matlab(ed, theme):
    out = _oracle()
    build_pid_loop(ed)
    got = []
    ed.model_created.connect(lambda m, n: got.append((m, n)))
    ed.set_lin_point("Referans", 0, "input")
    ed.set_lin_point("Bitki", 0, "output")
    assert [p[2] for p in ed.lin_points()] == ["input", "output"]
    sys_, name = ed.linearize("initial")
    assert got and got[0][1] == name == "linsys1"
    A, B, C, D = (np.asarray(getattr(sys_, k), dtype=float) for k in "ABCD")
    p = np.linalg.eigvals(A)
    p_ml = _c(out["p"])
    assert len(p) == len(p_ml)
    assert np.allclose(np.sort_complex(p), np.sort_complex(p_ml), rtol=1e-8, atol=1e-10)
    for wk, Hk in zip(np.asarray(out["w"], dtype=float).ravel(), _c(out["H"])):
        h = (C @ np.linalg.solve(1j * wk * np.eye(A.shape[0]) - A, B) + D)[0, 0]
        assert abs(h - Hk) <= 1e-8 * abs(Hk)
    dc = (C @ np.linalg.solve(-A, B) + D)[0, 0]
    assert abs(dc - out["dc"]) < 1e-9


def test_linearize_discrete_pid_template_matches_matlab(ed):
    out = _oracle()
    assert out["iformula"] == "ForwardEuler" and out["dformula"] == "ForwardEuler"
    ed.load_template("Ayrık PID + sürekli bitki (çok hızlı)")
    ed.set_lin_point("Referans", 0, "input")
    ed.set_lin_point("Bitki", 0, "output")
    sys_, _ = ed.linearize()
    assert float(sys_.Ts) == pytest.approx(0.01, abs=0)
    A, B, C, D = (np.asarray(getattr(sys_, k), dtype=float) for k in "ABCD")
    p = np.linalg.eigvals(A)
    p_ml = _c(out["pd"])
    assert np.allclose(np.sort_complex(p), np.sort_complex(p_ml), rtol=1e-8, atol=1e-10)
    for wk, Hk in zip(np.asarray(out["w"], dtype=float).ravel(), _c(out["Hd"])):
        z = np.exp(1j * wk * 0.01)
        h = (C @ np.linalg.solve(z * np.eye(A.shape[0]) - A, B) + D)[0, 0]
        assert abs(h - Hk) <= 1e-8 * abs(Hk)


def test_stop_cancels_long_run_without_freezing(ed, app):
    build_pid_loop(ed, "ode4", "1e-3")
    ed.stop_edit.setText("500")                                # ~5·10⁵ adım: durdurulmazsa dakikalar sürer
    t0 = time.perf_counter()
    assert ed.run()                                            # hemen döner (arka plan)
    assert time.perf_counter() - t0 < 1.0 and ed.running
    pump(app, 0.3)
    assert ed.btn_stop.isEnabled() and not ed.btn_run.isEnabled()
    ed.stop()
    assert ed.wait(20)
    assert ed.last_result is not None and ed.last_result.cancelled
    assert "durduruldu" in ed.status_text and ed.btn_run.isEnabled()
    assert ed.last_result.t[-1] < 500


# ============================================================================ hatalar (Türkçe)
def test_unconnected_port_reported_in_turkish_and_block_marked(ed):
    ed.add_block("Step", "Referans", pos=(0, 0))
    ed.add_block("Sum", "Toplam", pos=(140, 0), signs="+-")
    ed.add_block("Scope", "Skop", pos=(260, 0))
    ed.connect("Referans", "Toplam")
    ed.connect("Toplam", "Skop")
    assert not ed.run()
    assert "'Toplam' bloğunun 2. girişi bağlanmamış" in ed.status_text
    assert ed.status_state == "critical"
    assert ed.scene.blocks["Toplam"].error and not ed.scene.blocks["Referans"].error
    ed.add_block("Constant", "Sabit", pos=(0, 80), value=0.5)     # düzelt → yeniden çalıştır → işaret kalkar
    ed.connect("Sabit", "Toplam", 0, 1)
    assert ed.run(wait=True) and ed.last_error is None
    assert not ed.scene.blocks["Toplam"].error


def test_algebraic_loop_reported_in_turkish(ed):
    ed.add_block("Step", "Referans", pos=(0, 0))
    ed.add_block("Sum", "Toplam", pos=(140, 0), signs="+-")
    ed.add_block("Gain", "Kazanç", pos=(240, 0), K=2.0)
    ed.add_block("Scope", "Skop", pos=(380, 0))
    ed.connect("Referans", "Toplam")
    ed.connect("Toplam", "Kazanç")
    ed.connect("Kazanç", "Toplam", 0, 1)
    ed.connect("Kazanç", "Skop")
    assert ed.run(wait=True)                                   # yapı geçerli; derleme arka planda hata verir
    assert ed.last_result is None
    assert "Cebirsel döngü" in ed.status_text and ed.status_state == "critical"
    assert ed.scene.blocks["Toplam"].error and ed.scene.blocks["Kazanç"].error
    assert not ed.scene.blocks["Referans"].error


def test_param_panel_validates_in_turkish_and_is_undoable(ed):
    ed.add_block("TransferFcn", "Bitki", pos=(0, 0))
    item = ed.scene.blocks["Bitki"]
    ed.scene.clearSelection()
    item.setSelected(True)
    panel = ed.params
    assert panel.block is item.block
    panel.set_field("den", "[1 0")                             # kapanmamış köşeli ayraç
    assert not panel.apply()
    assert "Payda" in panel.error_text and "'Bitki' bloğu" in panel.error_text
    assert np.array_equal(item.block.params["den"], [1.0, 1.0])
    panel.set_field("den", "[1 3 2]")
    panel.set_field("num", "[2 1]")
    assert panel.apply() and panel.error_text == ""
    assert np.array_equal(item.block.params["den"], [1.0, 3.0, 2.0])
    assert np.array_equal(item.block.params["num"], [[2.0, 1.0]])
    ed.undo_stack.undo()
    assert np.array_equal(item.block.params["den"], [1.0, 1.0])
    ed.undo_stack.redo()
    assert np.array_equal(item.block.params["den"], [1.0, 3.0, 2.0])
    # değiştirilmeyen alan yuvarlanmadan aynen kalır (0.1 gibi ondalıklar)
    ed.set_params("Bitki", num=[0.1], den=[1.0, 0.30000000000000004])
    panel.refresh()
    assert panel.changes() == {}
    assert panel.field_text("den") == "[1 0.30000000000000004]"
    # yeniden adlandırma çakışması
    ed.add_block("Gain", "K", pos=(0, 120))
    ed.scene.clearSelection()
    item.setSelected(True)
    panel.set_field("__name__", "K")
    assert not panel.apply() and "adlı başka bir blok var" in panel.error_text


def test_sum_signs_change_removes_wire_on_dropped_port_and_undo_restores(ed):
    ed.add_block("Step", "R", pos=(0, 0))
    ed.add_block("Constant", "C", pos=(0, 80))
    ed.add_block("Sum", "S", pos=(140, 0), signs="+-")
    ed.connect("R", "S")
    ed.connect("C", "S", 0, 1)
    ed.set_params("S", signs="+")
    assert len(ed.scene.wires) == 1 and ed.scene.blocks["S"].n_in() == 1
    ed.undo_stack.undo()
    assert len(ed.scene.wires) == 2 and ed.block("S").params["signs"] == "+-"


# ============================================================================ dosya
def test_save_open_roundtrip(ed, theme, tmp_path, app):
    from spt.control.diagram.engine import simulate
    from spt.gui.control.diagram_editor import DiagramEditor

    build_pid_loop(ed)
    ed.set_lin_point("Bitki", 0, "output")
    wire = ed.scene.wires[3]                                   # Bitki → Hata geri beslemesi (elle yol)
    pts = wire.points()
    pts[2].setY(pts[2].y() + 40)
    pts[3].setY(pts[3].y() + 40)
    wire.set_route(pts)
    ed.scene.flip([ed.scene.blocks["Referans"]])
    ed.scene.flip([ed.scene.blocks["Referans"]])
    path = ed.save(tmp_path / "pid")
    assert path.suffix == ".sptd" and ed.undo_stack.isClean()
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["format"] == "spt-diagram" and raw["settings"]["solver"] == "ode4"
    other = DiagramEditor(theme)
    try:
        other.open(path)
        assert model_dict(other) == model_dict(ed)
        w2 = other.scene.wires[3]
        assert [(p.x(), p.y()) for p in w2.points()] == [(p.x(), p.y()) for p in wire.points()]
        assert other.scene.wires[5].parent_wire is other.scene.wires[3]
        assert other.lin_points() == [("Bitki", 0, "output")]
        r1 = simulate(ed.to_model().copy())
        r2 = simulate(other.to_model().copy())
        assert np.array_equal(r1.scopes["Skop"].data[1], r2.scopes["Skop"].data[1])
    finally:
        other.deleteLater()
    bad = tmp_path / "bad.sptd"
    bad.write_text('{"format": "baska"}', encoding="utf-8")
    from spt.control.errors import ControlError

    with pytest.raises(ControlError, match="blok diyagram"):
        ed.open(bad)


# ============================================================================ düzenleme: geri al / yinele, pano
def test_undo_redo_full_history(ed):
    states = [model_dict(ed)]
    ed.add_block("Step", "R", pos=(0, 0))
    states.append(model_dict(ed))
    ed.add_block("Gain", "K", pos=(140, 0), K=3.0)
    states.append(model_dict(ed))
    ed.connect("R", "K")
    states.append(model_dict(ed))
    ed.set_params("K", new_name="Kazanç", K=5.0)
    states.append(model_dict(ed))
    ed.scene.flip([ed.scene.blocks["Kazanç"]])
    states.append(model_dict(ed))
    item = ed.scene.blocks["R"]
    from spt.gui.control.diagram_canvas import MoveCmd

    old = QPointF(item.pos())
    item.setPos(QPointF(30, 60))
    ed.undo_stack.push(MoveCmd(ed.scene, [(item, old, QPointF(item.pos()))]))
    states.append(model_dict(ed))
    ed.scene.clearSelection()
    ed.scene.blocks["R"].setSelected(True)
    ed.scene.delete_selection()
    states.append(model_dict(ed))
    assert len(ed.scene.wires) == 0
    for k in range(len(states) - 1, 0, -1):
        assert model_dict(ed) == states[k]
        ed.undo_stack.undo()
    assert model_dict(ed) == states[0] and not ed.scene.blocks
    for k in range(1, len(states)):
        ed.undo_stack.redo()
        assert model_dict(ed) == states[k]


def test_copy_paste_keeps_internal_wires_and_renames(ed):
    ed.add_block("Step", "R", pos=(0, 0))
    ed.add_block("Gain", "K", pos=(140, 0), K=3.0)
    ed.add_block("Scope", "Skop", pos=(260, 0))
    ed.connect("R", "K")
    ed.connect("K", "Skop")
    ed.scene.clearSelection()
    for n in ("R", "K"):
        ed.scene.blocks[n].setSelected(True)
    assert ed.scene.copy_selection()
    made = ed.scene.paste()
    assert sorted(i.name for i in made) == ["K 2", "R 2"]
    assert ed.block("K 2").params["K"][0, 0] == 3.0
    pairs = {(w.src.name, w.dst.name) for w in ed.scene.wires}
    assert ("R 2", "K 2") in pairs and len(ed.scene.wires) == 3   # yalnız iç tel kopyalanır
    ed.undo_stack.undo()                                          # yapıştırma tek adımda geri alınır
    assert set(ed.scene.blocks) == {"R", "K", "Skop"} and len(ed.scene.wires) == 2


# ============================================================================ fare etkileşimi
def _vp(ed, p):
    return ed.view.mapFromScene(p)


def _drag_to(vp, pos, mods=Qt.KeyboardModifier.NoModifier):
    """Sol tuş basılıyken fare hareketi (QTest.mouseMove düğme durumunu taşımaz)."""
    from PyQt6.QtGui import QMouseEvent

    p = QPointF(pos)
    ev = QMouseEvent(QMouseEvent.Type.MouseMove, p, QPointF(vp.mapToGlobal(pos)), Qt.MouseButton.NoButton,
                     Qt.MouseButton.LeftButton, mods)
    QApplication.sendEvent(vp, ev)


def test_mouse_wiring_branching_and_segment_drag(ed, app):
    ed.add_block("Step", "R", pos=(0, 0))
    ed.add_block("Gain", "K", pos=(200, 100))
    ed.add_block("Scope", "Skop", pos=(400, 0), inputs=2)
    ed.view.resetTransform()
    ed.view.centerOn(250, 60)
    pump(app, 0.05)
    vp = ed.view.viewport()
    src = ed.scene.blocks["R"].port_scene_pos("out", 0)
    dst = ed.scene.blocks["K"].port_scene_pos("in", 0)
    QTest.mousePress(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _vp(ed, src))
    _drag_to(vp, _vp(ed, (src + dst) / 2))
    QTest.mouseRelease(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _vp(ed, dst))
    assert len(ed.scene.wires) == 1
    w = ed.scene.wires[0]
    assert (w.src.name, w.dst.name) == ("R", "K")
    pts = w.points()
    assert all(abs(a.x() - b.x()) < 1e-9 or abs(a.y() - b.y()) < 1e-9 for a, b in zip(pts, pts[1:]))   # dik açılı
    # ters yönde: girişten çıkışa sürükleme
    s2 = ed.scene.blocks["Skop"].port_scene_pos("in", 1)
    k_out = ed.scene.blocks["K"].port_scene_pos("out", 0)
    QTest.mousePress(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _vp(ed, s2))
    QTest.mouseRelease(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _vp(ed, k_out))
    assert len(ed.scene.wires) == 2
    # ara (dikey) bölümü sürükle → yol değişir, geri al eski yola döner
    k = 1
    a, b = pts[k], pts[k + 1]
    mid = QPointF(a.x(), (a.y() + b.y()) / 2)
    QTest.mousePress(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _vp(ed, mid))
    _drag_to(vp, _vp(ed, mid + QPointF(40, 0)))
    QTest.mouseRelease(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _vp(ed, mid + QPointF(40, 0)))
    assert w.vertices is not None and abs(w.points()[1].x() - (a.x() + 40)) < 1e-9
    ed.undo_stack.undo()
    assert w.vertices is None and abs(w.points()[1].x() - a.x()) < 1e-9
    # dal: Ctrl + telden sürükle → aynı kaynaktan yeni tel, dallanma noktası üst telde
    on_wire = QPointF((pts[0].x() + pts[1].x()) / 2, pts[0].y())
    s1 = ed.scene.blocks["Skop"].port_scene_pos("in", 0)
    QTest.mousePress(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier, _vp(ed, on_wire))
    QTest.mouseRelease(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier, _vp(ed, s1))
    assert len(ed.scene.wires) == 3
    br = ed.scene.wires[2]
    assert br.parent_wire is w and br.src.name == "R" and br.dst.name == "Skop"
    assert abs(br.points()[0].y() - on_wire.y()) < 1e-6
    # dolu girişe ikinci tel bağlanamaz (Türkçe uyarı)
    from spt.control.errors import ControlError

    with pytest.raises(ControlError, match="zaten bağlı"):
        ed.connect("R", "K")
    # blok taşıma: tel uçları izler, geri alınabilir
    item = ed.scene.blocks["K"]
    center = item.mapToScene(item.body_rect().center())
    QTest.mousePress(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _vp(ed, center))
    _drag_to(vp, _vp(ed, center + QPointF(30, 20)))
    _drag_to(vp, _vp(ed, center + QPointF(60, 40)))
    QTest.mouseRelease(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _vp(ed, center + QPointF(60, 40)))
    assert item.pos() != QPointF(200, 100)
    assert w.points()[-1] == item.port_scene_pos("in", 0)
    ed.undo_stack.undo()
    assert item.pos() == QPointF(200, 100)


def test_palette_search_and_drop(ed):
    pal = ed.palette
    pal.search.setText("integ")
    vis = pal.visible_types()
    assert "Integrator" in vis and "DiscreteIntegrator" in vis and "Gain" not in vis
    pal.search.setText("pid")
    assert "PIDController" in pal.visible_types()
    pal.search.setText("")
    assert len(pal.visible_types()) == len(pal.types()) > 30
    ed.view.block_dropped.emit("Saturation", QPointF(100, 100))
    assert "Doyum" in ed.scene.blocks
    it = ed.scene.blocks["Doyum"]
    c = it.mapToScene(it.body_rect().center())
    assert abs(c.x() - 100) <= 10 and abs(c.y() - 100) <= 10


# ============================================================================ şablonlar ve veri kaynağı
def test_template_registry_and_plugin_hook(ed):
    from spt.control.diagram import Model
    from spt.gui.control.diagram_editor import register_template, templates

    def build():
        m = Model("deneme")
        m.add("Step", "R")
        m.add("Integrator", "I")
        m.add("Scope", "Skop")
        m.connect("R", "I")
        m.connect("I", "Skop")
        m.settings.update(solver="ode45", stop_time=10.0)    # Basamak anına tam basar
        return m

    register_template("Deneme şablonu", build, "integratör", category="Otopilot")
    names = [t.name for t in templates()]
    assert "Deneme şablonu" in names and "PID ile geri besleme (sürekli)" in names
    ed.load_template("Deneme şablonu")
    xs = [ed.scene.blocks[n].pos().x() for n in ("R", "I", "Skop")]
    assert xs == sorted(xs) and len(set(xs)) == 3                 # otomatik soldan sağa yerleşim
    assert ed.run(wait=True) and ed.last_error is None
    t = ed.last_result.scopes["Skop"].t
    assert np.allclose(ed.last_result.scopes["Skop"].data[0][:, 0], np.where(t >= 1.0, t - 1.0, 0.0), atol=1e-9)


def _snapshot_handle(data, fs, names):
    from spt.core.local_ring import StaticRing
    from spt.core.snapshot import Snapshot
    from spt.protocol.format import SAMPLE_TYPES, ChannelInfo, Descriptor, Layout

    desc = Descriptor(1, 1, fs, SAMPLE_TYPES[0x11], Layout.INTERLEAVED, "test",
                      tuple(ChannelInfo(n, "V", 2.0, 0.5) for n in names))
    return Snapshot(1, "anlık", "test", desc, 0.0, StaticRing(np.asarray(data, dtype=np.float32), 100)).handle()


def test_data_source_binds_to_snapshot_and_csv(ed, tmp_path):
    raw = np.column_stack([np.arange(50), np.arange(50) * 0.5]).astype(np.float32)
    h = _snapshot_handle(raw, 100.0, ["a", "b"])
    ed.set_sources([("ws1", "Çalışma alanı · anlık", h)])
    ed.add_block("DataSource", "Veri", pos=(0, 0))
    ed.scene.clearSelection()
    ed.scene.blocks["Veri"].setSelected(True)
    panel = ed.params
    assert panel._src_combo.count() == 2
    for i in range(panel._chan_list.count()):                       # iki kanalı da seç
        panel._chan_list.item(i).setCheckState(Qt.CheckState.Checked)
    assert panel.bind()
    b = ed.block("Veri")
    assert np.allclose(b.params["time"], np.arange(50) / 100.0, rtol=0, atol=0)
    assert np.allclose(b.params["values"][:, 1], raw[:, 1] * 2.0 + 0.5, rtol=0, atol=0)   # fiziksel birim
    assert b.meta["binding"]["channels"] == ["a", "b"]
    assert ed.scene.blocks["Veri"].n_out() == 1 and b.source_width() == 2
    ed.undo_stack.undo()
    assert ed.block("Veri").params["values"].shape == (2, 1)
    # CSV: başlıklı, zaman sütunu otomatik bulunur
    p = tmp_path / "olcum.csv"
    p.write_text("time;u;y\n0;1;2\n0.5;3;4\n1.5;5;6\n", encoding="utf-8")
    ed.scene.blocks["Veri"].setSelected(False)
    ed.scene.blocks["Veri"].setSelected(True)
    panel.load_table(p)
    assert panel._time_combo.currentData() == 0
    assert panel.bind()
    b = ed.block("Veri")
    assert np.array_equal(b.params["time"], [0.0, 0.5, 1.5])
    assert np.array_equal(b.params["values"], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    # bozuk veri → Türkçe hata, blok değişmez
    p2 = tmp_path / "bozuk.csv"
    p2.write_text("t,u\n0,1\n1,\n", encoding="utf-8")
    panel.load_table(p2)
    assert not panel.bind() and "boş ya da sayı olmayan hücre" in panel.error_text


# ============================================================================ görünüm
def test_layout_readable_at_1366x768(ed, app, tmp_path):
    ed.resize(1366, 540)            # 1366×768 pencerede menü, üst çubuk, sayfa başlığı ve sekmelerden kalan alan
    ed.load_template("PID ile geri besleme (sürekli)")
    assert ed.run(wait=True)
    pump(app, 0.2)
    shot = ed.grab()
    out = Path(os.environ.get("SPT_SHOT_DIR", tmp_path)) / "diagram_1366.png"
    shot.save(str(out))
    assert ed.palette.width() >= 180 and ed.params.width() >= 220
    assert ed.view.width() >= 600 and ed.view.height() >= 190
    assert ed.btn_lin.geometry().right() <= ed.width()                # araç çubuğu sığar
    assert ed.btn_run.isVisible() and ed.solver.isVisible() and ed.stop_edit.isVisible()
    assert list(ed.scopes.cards) == ["Çıkış"]
    assert ed.center_tabs.currentIndex() == 1                        # sonuç gelince Sonuçlar sekmesi
    plot = ed.scopes.plots["Çıkış"]
    vp = ed.scopes.scroll.viewport()
    from PyQt6.QtCore import QPoint, QRect

    assert plot.height() >= 170 and vp.rect().contains(QRect(plot.mapTo(vp, QPoint(0, 0)), plot.size()))  # kaydırmasız
    # durum satırı tuvalin dışında (blokları örtmez)
    assert ed.status_overlay.parentWidget() is not ed.view and "Benzetim bitti" in ed.status_text
    ed.center_tabs.setCurrentIndex(0)
    pump(app, 0.2)
    assert ed.view.height() >= 300                                    # tuval tam boy
    # tüm bloklar görünür alanda (Sığdır)
    vr = ed.view.mapToScene(ed.view.viewport().rect()).boundingRect()
    for it in ed.scene.blocks.values():
        assert vr.contains(it.sceneBoundingRect().center())


def test_distinct_colors_are_unique_and_far_apart():
    from spt.gui.control.diagram_scope import BASE_COLORS, SURFACE, delta_e, distinct_colors

    cols = distinct_colors(12)
    assert len(set(cols)) == 12 and cols[:len(BASE_COLORS)] == BASE_COLORS
    assert min(delta_e(a, b) for i, a in enumerate(cols) for b in cols[i + 1:]) > 10

    def lum(h):
        c = np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])
        c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        return float(c @ [0.2126, 0.7152, 0.0722])

    assert min((lum(c) + 0.05) / (lum(SURFACE) + 0.05) for c in cols) >= 3.0     # zemine kontrast ≥ 3:1


# ============================================================================ Kontrol Tasarımı görünümüyle entegrasyon
def test_control_view_hosts_editor_and_linearization_reaches_workspace(app, theme, tmp_path, monkeypatch):
    monkeypatch.setenv("SPT_SETTINGS_FILE", str(tmp_path / "settings.ini"))
    from PyQt6.QtCore import QSettings

    from spt.gui.control.diagram_editor import DiagramEditor
    from spt.gui.views.control_view import ControlView

    view = ControlView(theme)
    try:
        assert not view.is_placeholder("diagram")
        ed = view.pages["diagram"]
        assert isinstance(ed, DiagramEditor)
        ed.load_template("PID ile geri besleme (sürekli)")
        ed.set_lin_point("PID", 0, "looptransfer")                 # çevrim transferi (−C·G)
        sys_, name = ed.linearize()
        assert name in view.workspace and view.workspace.get(name) is sys_
        assert [k for k, _t, _c in view.detachable_cards() if k.startswith("ctl_diagram")] == ["ctl_diagram_scopes"]
        s = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
        ed.save_state(s)
        other = DiagramEditor(theme)
        try:
            other.restore_state(s)
            assert model_dict(other) == model_dict(ed) and not other.undo_stack.isClean()
        finally:
            other.deleteLater()
        view.shutdown()
    finally:
        view.deleteLater()


def test_subsystem_block_loads_runs_and_copies_with_contents(ed):
    from spt.control.diagram import Model

    inner = Model("iç")
    inner.add("Inport", "Giriş", port=1)
    inner.add("Gain", "K", K=2.0)
    inner.add("Outport", "Çıkış", port=1)
    inner.connect("Giriş", "K")
    inner.connect("K", "Çıkış")
    m = Model("dış")
    m.add("Step", "R", step_time=0.0)
    m.add_subsystem("Alt", inner)
    m.add("Scope", "Skop")
    m.connect("R", "Alt")
    m.connect("Alt", "Skop")
    m.settings.update(solver="ode4", step=0.1, stop_time=1.0)
    ed.load_model(m)
    item = ed.scene.blocks["Alt"]
    assert item.n_in() == 1 and item.n_out() == 1
    assert ed.run(wait=True) and ed.last_error is None
    assert np.array_equal(ed.last_result.scopes["Skop"].data[0][:, 0], np.full(11, 2.0))
    ed.scene.clearSelection()
    item.setSelected(True)
    assert ed.scene.copy_selection()
    made = ed.scene.paste()
    assert len(made) == 1 and made[0].name == "Alt 2"
    assert made[0].block.system.blocks["K"].params["K"][0, 0] == 2.0 and made[0].n_in() == 1


def test_logged_signals_and_templates_show_results(ed, app):
    """Kullanıcı bildirimi: "skop kayıt çalışmıyor" — İHA şablonları sinyalleri telde kaydediyordu (log), sonuç paneli
    yalnız Skop bloklarını gösterdiği için hiçbir şey görünmüyordu. Artık kaydedilen sinyaller ayrı kartlarda."""
    names = [t.name for t in __import__("spt.gui.control.diagram_editor", fromlist=["templates"]).templates()]
    tpl = next(n for n in names if "roll" in n)
    ed.load_template(tpl)
    assert ed.run(wait=True), ed.status_text
    pump(app, 0.2)
    res = ed.last_result
    assert res.signals and set(ed.scopes.signal_cards) == set(res.signals)
    for sig, arr in res.signals.items():
        tr = ed.scopes.traces["sinyal:" + sig]
        assert np.array_equal(tr[0].item.yData, np.asarray(arr).reshape(len(res.t), -1)[:, 0])
    assert "Sonuçlar (" in ed.center_tabs.tabText(1)
    cols = ed.scopes.series_colors()
    assert len(set(cols)) == len(cols)                                # aynı görünümde renk tekrarı yok
    # telde "Sinyali kaydet": Skop'suz diyagramda da sonuç gelir
    ed.new_diagram(confirm=False)
    ed.add_block("Step", "Basamak", (0, 0))
    ed.add_block("TransferFcn", "G", (200, 0))
    ed.add_block("Terminator", "Son", (400, 0))
    ed.connect("Basamak", "G")
    ed.connect("G", "Son")
    ed.set_signal_log("G", 0, True)
    assert ed.run(wait=True), ed.status_text
    pump(app, 0.2)
    assert list(ed.scopes.signal_cards) == ["G"] and not ed.scopes.cards
    y = ed.scopes.traces["sinyal:G"][0].item.yData
    assert y[-1] == pytest.approx(1 - np.exp(-9.0), rel=1e-5)        # 1/(s+1), basamak t = 1 s; RK4 h = 0.2 s hatası
    ed.set_signal_log("G", 0, False)
    assert ed.run(wait=True)
    pump(app, 0.2)
    assert ed.scopes.count == 0


def test_scope_added_from_palette_records(ed, app):
    """Kullanıcı bildirimi: paletten eklenen Skop'un varsayılan adı "Skop / kayıt" idi; '/' blok adında yasak
    (alt sistem yol ayıracı) olduğundan Çalıştır istisna fırlatıyordu. Artık ad "Skop" ve kayıt çalışır."""
    names = []
    for tn in ("Step", "TransferFcn", "Scope", "MinMax"):
        before = set(ed.scene.blocks)
        ed._add_at_center(tn)
        names.append(sorted(set(ed.scene.blocks) - before)[0])
    assert names[2] == "Skop" and names[3] == "En küçük"
    assert not any(c in n for n in names for c in "/\n\t")
    ed.remove_block(names[3]) if hasattr(ed, "remove_block") else ed.scene.blocks[names[3]].setSelected(True)
    if names[3] in ed.scene.blocks:
        ed.scene.delete_selection()
    ed.connect(names[0], names[1])
    ed.connect(names[1], names[2])
    assert ed.run(wait=True), ed.status_text
    pump(app, 0.2)
    assert list(ed.scopes.cards) == ["Skop"] and ed.last_error is None


def test_every_palette_block_adds_and_round_trips(ed, app, tmp_path):
    """Paletteki her blok türü: çift tıklamayla eklenir, seçilince parametre paneli açılır, modele dönüşür, kaydedilip
    açılır (geçersiz varsayılan ad ya da panel hatası kalmasın)."""
    from spt.control.diagram import block_types
    from spt.control.diagram.model import Model

    for tn in block_types():
        ed.new_diagram(confirm=False)
        ed._add_at_center(tn)
        (name,) = list(ed.scene.blocks)
        assert not any(c in name for c in "/\n\t"), (tn, name)
        ed.scene.blocks[name].setSelected(True)
        pump(app, 0.01)
        path = tmp_path / "x.sptd"
        ed.to_model(settings=False).save(path)
        assert name in Model.load(path).blocks, tn


def test_results_tab_title_resets_on_new_and_load(ed):
    """Denetim: yeni / açılan diyagramda 'Sonuçlar (n)' başlığı eski sayıyla kalmamalı."""
    build_pid_loop(ed)
    assert ed.run(wait=True), ed.status_text
    assert ed.center_tabs.tabText(1).startswith("Sonuçlar (")
    assert ed.new_diagram(confirm=False)
    assert ed.center_tabs.tabText(1) == "Sonuçlar"
    build_pid_loop(ed)
    assert ed.run(wait=True), ed.status_text
    ed.load_model(ed.to_model())
    assert ed.center_tabs.tabText(1) == "Sonuçlar"
