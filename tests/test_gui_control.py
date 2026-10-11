"""Kontrol Tasarımı görünümü (Aşama 4) arayüz testleri (ekransız): model çalışma alanı ve .sptm gidiş-dönüşü, ana
pencere entegrasyonu (görünüm, kısayol, odak düzeni, ayrık pencereler, ayarların korunması), Modeller sekmesi
(ifade, gösterim, kutup / sıfır, dönüşümler) ve Analiz sekmesi (çevrim, grafikler, pay ve stepinfo tabloları — MATLAB
R2025b referansıyla).

Ayarlar geçici bir INI dosyasında tutulur (SPT_SETTINGS_FILE); kullanıcının ayarlarına dokunulmaz.
"""
import json
import math
import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

# MATLAB referansı: G = 1/((s+1)(s+2)(s+3)), C = 10 + 5/s (PI), L = C·G, T = feedback(G·C, 1);
# ayrıkta her bileşen ZOH ile
MATLAB_SCRIPT = r"""
G = tf(1, [1 6 11 6]); C = tf([10 5], [1 0]);
L = C*G; T = feedback(G*C, 1);
[gm, pm, wcg, wcp] = margin(L);
out.gm = gm; out.pm = pm; out.wcg = wcg; out.wcp = wcp;
am = allmargin(L); out.stable = double(am.Stable);
S = stepinfo(T);
out.rise = S.RiseTime; out.settling = S.SettlingTime; out.overshoot = S.Overshoot;
out.peak = S.Peak; out.peaktime = S.PeakTime;
out.bw = bandwidth(T); out.dc = dcgain(T);
Ts = 0.1;
Ld = c2d(C, Ts, 'zoh') * c2d(G, Ts, 'zoh');
[gm, pm, wcg, wcp] = margin(Ld);
out.dgm = gm; out.dpm = pm; out.dwcg = wcg; out.dwcp = wcp;
Td = feedback(c2d(G, Ts, 'zoh') * c2d(C, Ts, 'zoh'), 1);
Sd = stepinfo(Td);
out.drise = Sd.RiseTime; out.dsettling = Sd.SettlingTime; out.dovershoot = Sd.Overshoot;
[wn, z, p] = damp(G);
out.wn = wn; out.zeta = z; out.p = spt_c(p);
S = feedback(1, L); KS = C*S;
k = stepinfo(KS);
out.ks_rise = k.RiseTime; out.ks_settling = k.SettlingTime; out.ks_overshoot = k.Overshoot;
out.ks_peak = k.Peak; out.ks_peaktime = k.PeakTime; out.ks_dc = dcgain(KS);
q = stepinfo(S);
out.s_rise = q.RiseTime; out.s_settling = q.SettlingTime; out.s_overshoot = q.Overshoot;
out.s_peak = q.Peak; out.s_peaktime = q.PeakTime; out.s_bw = bandwidth(S); out.s_dc = dcgain(S);
out.damp_u = evalc('damp(tf(1, [1 -1 4]))');
out.damp_i = evalc('damp(tf(1, [1 2 0]))');
out.damp_d = evalc('damp(tf([1 0.5], [1 -0.5 0 0], 0.1))');
"""


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def theme(app):
    from spt.gui.theme import ThemeManager

    t = ThemeManager(app)
    t.apply()
    return t


def pump(app, seconds):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        app.processEvents()
        time.sleep(0.005)


@pytest.fixture(scope="module")
def shared_win(app, theme, tmp_path_factory):
    """Modül boyunca tek ana pencere (pencere kurulumu yavaş); her test çalışma alanını boşaltarak başlar."""
    mp = pytest.MonkeyPatch()
    mp.setenv("SPT_SETTINGS_FILE", str(tmp_path_factory.mktemp("ayar") / "settings.ini"))
    from spt.gui.main_window import MainWindow

    w = MainWindow(theme)
    w.resize(1600, 960)
    w.show()
    pump(app, 0.2)
    yield w
    w.close()
    w.deleteLater()
    pump(app, 0.2)
    mp.undo()


@pytest.fixture
def win(app, shared_win):
    w = shared_win
    w.detacher.attach_all()
    w.control_view.workspace.clear()
    an = w.control_view.analysis_tab
    an.set_extra_models([])
    an.domain.setCurrentIndex(0)
    w.resize(1600, 960)
    pump(app, 0.1)
    return w


@pytest.fixture(scope="module")
def ref():
    oracle = pytest.importorskip("tools.matlab_oracle")
    try:
        return oracle.fixture("gui_control_analysis", MATLAB_SCRIPT)
    except oracle.OracleUnavailable as exc:
        pytest.skip(f"MATLAB referansı yok: {exc}")


def lti():
    return pytest.importorskip("spt.control.lti")


def need(*modules):
    for m in modules:
        pytest.importorskip(m)


def close(a, b, rel=1e-6):
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=1e-12)


def _margins_consistent(L, values, ref, prefix):
    """Pay tablosundaki (Gm, Pm, Wcg, Wcp):
    1) kütüphanenin margin(L) çıktısıyla birebir (arayüz değeri değiştirmez);
    2) tanımı sağlar: ∠L(jWcg) = −180° (mod 360), |L(jWcp)| = 1, Gm = 1/|L(jWcg)|, Pm = 180° + ∠L(jWcp) — 1e-9;
    3) MATLAB R2025b margin ile kendi hassasiyeti içinde: MATLAB'in bu çevrimde bulduğu Wcg'de faz 179.9991°
       (göreli hata 1.4e-5), dolayısıyla karşılaştırma göreli 1e-4 ile."""
    import cmath

    from spt.control.freqresp import freqresp, margin
    gm, pm, wcg, wcp = values
    assert (gm, pm, wcg, wcp) == tuple(float(v) for v in margin(L))
    hg = complex(freqresp(L, [wcg])[0, 0, 0])
    hc = complex(freqresp(L, [wcp])[0, 0, 0])
    assert abs(abs(math.degrees(cmath.phase(hg))) - 180.0) < 1e-9 and close(gm, 1.0 / abs(hg), 1e-9)
    assert close(abs(hc), 1.0, 1e-9)
    assert abs(((pm - (180.0 + math.degrees(cmath.phase(hc)))) + 180.0) % 360.0 - 180.0) < 1e-9
    for name, v in zip(("gm", "pm", "wcg", "wcp"), values):
        assert close(v, ref[prefix + name], 1e-4), name


# ============================================================================ çalışma alanı
def test_workspace_names_signals_and_sptm_roundtrip(app, tmp_path):
    L = lti()
    from spt.control.errors import ControlError
    from spt.gui.control.workspace import ModelWorkspace

    ws = ModelWorkspace()
    events = []
    ws.added.connect(lambda n: events.append(("+", n)))
    ws.removed.connect(lambda n: events.append(("-", n)))
    ws.renamed.connect(lambda a, b: events.append(("~", a, b)))
    ws.changed.connect(lambda n: events.append(("*", n)))
    G = L.tf([1, 3], [1, 2, 5])
    assert ws.add("G", G) == "G" and G.name != "G" or G.name == ""   # çağıranın nesnesi değişmez
    assert ws.add("G", L.zpk([-1], [-2, -3], 5.0)) == "G2"            # ad doluysa benzersiz ad
    ws.add("Gd", L.tf([0.5], [1, -0.8], 0.1), source="ifade", note="ayrık", expr="0.5/(z-0.8)")
    ws.add("P", L.ss([[0, 1], [-2, -3]], [[0], [1]], [[1, 0]], [[0]]))
    ws.add("G", L.tf([2], [1, 1]), replace=True)                       # aynı ad: değiştirme
    for bad in ("", "1G", "s", "z", "tf", "a-b", "çarpan"):
        with pytest.raises(ControlError):
            ws.add(bad or " ", G) if bad else ws.add("", None)
    ws.rename("G2", "Gz")
    with pytest.raises(ControlError):
        ws.rename("Gz", "Gd")
    assert ws.names() == ["G", "Gz", "Gd", "P"]
    assert ("~", "G2", "Gz") in events and ("*", "G") in events
    assert ws.get("Gz").name == "Gz" and str(ws.get("Gz")).startswith("Gz =")

    path = tmp_path / "m.sptm"
    assert ws.save(path) == []
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["format"] == "sptm" and data["version"] == 1 and len(data["models"]) == 4
    ws2 = ModelWorkspace()
    added, failed = ws2.load(path)
    assert failed == [] and added == ["G", "Gz", "Gd", "P"]
    for n in added:
        assert ws2.get(n).to_dict() == ws.get(n).to_dict()
        assert type(ws2.get(n)) is type(ws.get(n))
    e = ws2.entry("Gd")
    assert (e.source, e.note, e.expr) == ("ifade", "ayrık", "0.5/(z-0.8)") and ws2.get("Gd").Ts == 0.1
    # aynı adlar ikinci yüklemede benzersiz adla eklenir; replace=True ile değiştirilir
    added, _ = ws2.load(path)
    assert added == ["G2", "Gz2", "Gd2", "P2"]
    added, _ = ws2.load(path, replace=True)
    assert added == ["G", "Gz", "Gd", "P"] and len(ws2) == 8
    with pytest.raises(ControlError):
        ws2.load_dict({"format": "sptd", "models": []})
    with pytest.raises(ControlError):
        ws2.load_dict({"format": "sptm", "version": 99, "models": []})
    bad = tmp_path / "bad.sptm"
    bad.write_text("{", encoding="utf-8")
    with pytest.raises(ControlError):
        ws2.load(bad)
    ws.remove("P")
    assert ("-", "P") in events and "P" not in ws


# ============================================================================ ana pencere entegrasyonu
def test_control_view_is_registered_in_main_window(app, win):
    from spt.gui.main_window import FOCUS_DOCKS, NAV_ORDER, VIEW_KEYS, _ALL_PANELS

    # yeni görünüm sona eklendi: kayıtlı eski görünüm indeksleri bozulmaz
    assert VIEW_KEYS[:5] == ("view.live", "view.siggen", "view.filter", "view.review", "view.fusion")
    assert VIEW_KEYS[5] == "view.control" and NAV_ORDER[-1] == "view.control"
    assert set(FOCUS_DOCKS["view.control"]) == set(_ALL_PANELS)
    act = win.registry["view.control"]
    assert act.shortcut().toString() == "Ctrl+6" and act.isEnabled()
    shortcuts = [a.shortcut().toString() for _, a in win.registry.items() if not a.shortcut().isEmpty()]
    assert shortcuts.count("Ctrl+6") == 1
    act.trigger()
    pump(app, 0.3)
    cv = win.control_view
    assert win.views.currentWidget() is cv and win.views.currentIndex() == 5
    assert [cv.tabs.tabText(i) for i in range(cv.tabs.count())] == [
        "Modeller", "Sistem Tanımlama", "Analiz", "Kontrolcü Tasarımı", "Blok Diyagram"]
    assert not cv.is_placeholder("models") and not cv.is_placeholder("analysis")
    for key in cv.tab_keys():  # her sekme açılabilir (hazır değilse nedenini söyleyen yer tutucu)
        cv.show_tab(key)
        pump(app, 0.05)
        assert cv.current_tab() == key
    # dar pencerede tüm paneller gizlenir, başka görünüme geçince geri gelir
    assert not any(d.toggleViewAction().isChecked() for d in win.docks.values())
    win._show_view(0)
    pump(app, 0.2)
    assert win.docks["sources"].toggleViewAction().isChecked()
    # kartlar ayrı pencereye alınabilir ve geri gelir
    keys = [a.data() if a.data() else a.text() for a in win.detacher.actions("card")]
    assert any("Kontrol · Analiz · Adım" in a.text() for a in win.detacher.actions("card")), keys
    win._show_view(5)
    cv.show_tab("analysis")
    pump(app, 0.1)
    step = cv.analysis_tab.plots["step"]
    win.detacher.detach("ctl_an_step")
    pump(app, 0.1)
    assert win.detacher.is_detached("ctl_an_step") and step.window() is not win
    cv.analysis_tab.set_plots(["step", "bode", "nyquist"])   # ızgara yeniden dizilince ayrık kart çalınmaz
    pump(app, 0.1)
    assert step.window() is not win
    win.detacher.attach("ctl_an_step")
    pump(app, 0.1)
    assert step.window() is win and step.isVisible()
    win._show_view(0)


def test_control_state_and_models_persist_across_sessions(app, theme, tmp_path, monkeypatch):
    L = lti()
    monkeypatch.setenv("SPT_SETTINGS_FILE", str(tmp_path / "settings.ini"))
    from spt.gui.main_window import MainWindow

    w = MainWindow(theme)
    w.resize(1600, 960)
    w.show()
    pump(app, 0.2)
    cv = w.control_view
    cv.workspace.add("Gp", L.tf([1], [1, 6, 11, 6]))
    cv.workspace.add("Cpi", L.tf([10, 5], [1, 0]))
    w._show_view(5)
    cv.show_tab("analysis")
    an = cv.analysis_tab
    an.set_loop(G="Gp", C="Cpi")
    an.set_plots(["step", "rlocus"])
    an.domain.setCurrentIndex(an.domain.findData("discrete"))
    an.ts.setValue(0.05)
    pump(app, 0.2)
    w.close()
    pump(app, 0.2)
    w2 = MainWindow(theme)
    w2.show()
    pump(app, 0.3)
    try:
        cv2 = w2.control_view
        assert w2.views.currentIndex() == 5 and cv2.current_tab() == "analysis"
        assert cv2.workspace.names() == ["Gp", "Cpi"]
        assert cv2.workspace.get("Gp").to_dict() == cv.workspace.get("Gp").to_dict()
        an2 = cv2.analysis_tab
        assert an2.selected_plots() == ["step", "rlocus"]
        assert an2.domain.currentData() == "discrete" and an2.ts.value() == 0.05
        assert an2.boxes["G"].currentData() == "Gp" and an2.boxes["C"].currentData() == "Cpi"
    finally:
        w2.close()
        w2.deleteLater()
        w.deleteLater()
        pump(app, 0.2)


# ============================================================================ Modeller sekmesi
def test_models_tab_expression_display_and_pole_table(app, win, ref):
    need("spt.control.parse", "spt.control.analysis", "spt.control.timeresp", "spt.control.freqresp")
    cv = win.control_view
    win._show_view(5)
    cv.show_tab("models")
    mt = cv.models_tab
    mt.name_edit.setText("G")
    mt.expr_edit.setText("1/(s^3 + 6*s^2 + 11*s + 6)")
    assert mt.add_from_expression() == "G"
    pump(app, 0.3)
    assert mt.current == "G" and "G =" in mt.display.toPlainText()
    row = [mt.table.item(0, c).text() for c in range(mt.table.columnCount())]
    assert row[:5] == ["G", "tf", "sürekli", "1×1", "3"] and "kararlı" in row[5]
    # kutup tablosu: MATLAB damp(G) (kutuplar −1, −2, −3: ωn = 1, 2, 3, ζ = 1)
    cols = _pz_columns(mt)
    poles = [r for r in range(mt.pz_table.rowCount()) if mt.pz_table.item(r, 0).text() == "kutup"]
    assert len(poles) == 3
    for r, (wn, z) in enumerate(zip(ref["wn"], ref["zeta"])):
        assert math.isclose(float(mt.pz_table.item(poles[r], cols["ωn (rad/s)"]).text()), wn, rel_tol=1e-5)
        assert math.isclose(float(mt.pz_table.item(poles[r], cols["Sönüm ζ"]).text()), z, rel_tol=1e-4)
    # hızlı grafiklerde veri var
    assert mt.step_plot.line_traces() and mt.bode_plot.mag_card.line_traces()
    assert mt.pzmap_plot.trace_names() == ["G · kutup"]
    # geçersiz ifade: Türkçe hata, model eklenmez
    mt.name_edit.setText("H")
    mt.expr_edit.setText("1/(s^2 + ")
    assert mt.add_from_expression() is None and mt.expr_error.isVisible() and "H" not in cv.workspace


def test_models_tab_transforms_add_new_models(app, win):
    L = lti()
    need("spt.control.discretize", "spt.control.convert", "spt.control.analysis")
    cv = win.control_view
    mt = cv.models_tab
    cv.workspace.add("G", L.tf([1], [1, 6, 11, 6]))
    pump(app, 0.1)
    mt.select("G")
    mt.c2d_ts.setValue(0.1)
    mt.c2d_method.setCurrentIndex(mt.c2d_method.findData("zoh"))
    assert mt.c2d_selected() == "G_d"
    from spt.control.discretize import c2d
    expect = c2d(L.tf([1], [1, 6, 11, 6]), 0.1, method="zoh").to_dict()
    assert cv.workspace.get("G_d").to_dict()["num"] == expect["num"] and cv.workspace.get("G_d").Ts == 0.1
    mt.select("G_d")
    assert mt.d2c_selected() == "G_d_c" and cv.workspace.get("G_d_c").Ts == 0
    mt.select("G_d")
    assert mt.c2d_selected() is None and "zaten ayrık" in mt.conv_status.text()
    mt.select("G")
    mt.form_combo.setCurrentText("ss")
    assert mt.convert_selected() == "G_ss" and cv.workspace.entry("G_ss").kind == "ss"
    assert mt.minreal_selected() == "G_ss_min"
    mt.select("G")
    mt.red_order.setValue(2)
    assert mt.balred_selected() == "G_r2"
    assert len(cv.analysis_tab.extra_list.findItems("G_r2", Qt.MatchFlag.MatchExactly)) == 1   # Analiz seçimlerinde


def _pz_columns(mt) -> dict[str, int]:
    return {mt.pz_table.horizontalHeaderItem(c).text(): c for c in range(mt.pz_table.columnCount())}


def _parse_num(text: str) -> float:
    return {"∞": math.inf, "−∞": -math.inf, "—": math.nan}.get(text, None) or float(text.replace("−", "-"))


def _matlab_damp_tau(text: str) -> list[float]:
    """MATLAB `damp` çıktısının son sütunu (Time Constant), satır sırası damp sırası."""
    out = []
    for line in text.splitlines():
        toks = line.split()
        if toks:
            try:
                out.append(float(toks[-1]))
            except ValueError:
                continue
    return out


def test_models_tab_damp_time_constants_match_matlab(app, win, ref):
    """Kutup tablosunun τ sütunu MATLAB damp gösterimiyle: kararsız kökte negatif, s = 0'da ∞, ayrıkta z = 0'da 0;
    ayrıkta |z| sütunu var."""
    need("spt.control.analysis")
    L = lti()
    cv = win.control_view
    mt = cv.models_tab
    win._show_view(5)
    cv.show_tab("models")
    for name, sys, key in (("Gu", L.tf([1], [1, -1, 4]), "damp_u"), ("Gi", L.tf([1], [1, 2, 0]), "damp_i"),
                           ("Gz", L.tf([1, 0.5], [1, -0.5, 0, 0], 0.1), "damp_d")):
        cv.workspace.add(name, sys)
        mt.select(name)
        mt._show_current()
        cols = _pz_columns(mt)
        assert ("|z|" in cols) == (sys.Ts > 0)
        rows = [r for r in range(mt.pz_table.rowCount()) if mt.pz_table.item(r, 0).text() == "kutup"]
        mine = [_parse_num(mt.pz_table.item(r, cols["τ (s)"]).text()) for r in rows]
        theirs = _matlab_damp_tau(ref[key])
        assert len(mine) == len(theirs) > 0, (mine, theirs)
        for a, b in zip(mine, theirs):
            if math.isinf(b) or b == 0:
                assert a == b, (name, mine, theirs)
            else:
                assert math.isclose(a, b, rel_tol=6e-3), (name, mine, theirs)   # MATLAB 3 anlamlı basamak yazar
    row = [mt.table.item(0, c).text() for c in range(mt.table.columnCount())]
    assert row[0] == "Gu" and "kararsız" in row[5]


def test_models_tab_expression_uses_workspace_and_sptm_roundtrip(app, win, tmp_path):
    """İfadede çalışma alanı modelleri adlarıyla; sonuç kütüphanenin feedback'iyle aynı frekans cevabında. Modeller
    sekmesinin kaydet / aç yolu .sptm gidiş-dönüşü (aynı adlar benzersiz adla eklenir)."""
    need("spt.control.parse", "spt.control.freqresp", "spt.control.connect")
    import numpy as np

    L = lti()
    from spt.control.connect import feedback
    from spt.control.freqresp import freqresp

    cv = win.control_view
    mt = cv.models_tab
    cv.workspace.add("Gp", L.tf([1], [1, 6, 11, 6]))
    cv.workspace.add("Cpi", L.tf([10, 5], [1, 0]))
    mt.name_edit.setText("T")
    mt.expr_edit.setText("Gp*Cpi/(1 + Gp*Cpi)")
    assert mt.add_from_expression() == "T"
    w = np.logspace(-2, 2, 41)
    expect = freqresp(feedback(cv.workspace.get("Gp") * cv.workspace.get("Cpi"), 1), w)
    assert np.allclose(freqresp(cv.workspace.get("T"), w), expect, rtol=1e-9, atol=1e-14)
    mt.name_edit.setText("Gd")
    mt.expr_edit.setText("0.5/(z - 0.8)")
    mt.expr_ts.setValue(0.1)
    assert mt.add_from_expression() == "Gd" and cv.workspace.get("Gd").Ts == 0.1
    mt.expr_ts.setValue(0.0)
    path = tmp_path / "ws.sptm"
    assert mt.save_path(path) == [] and "4 model kaydedildi" in mt.conv_status.text()
    def data(name):
        d = dict(cv.workspace.get(name).to_dict())
        d.pop("name", None)   # gösterimdeki ad (yeni adla eklendi)
        return d

    before = {n: data(n) for n in cv.workspace.names()}
    assert mt.load_path(path) == ["Gp2", "Cpi2", "T2", "Gd2"]
    for n, d in before.items():
        assert data(f"{n}2") == d and type(cv.workspace.get(f"{n}2")) is type(cv.workspace.get(n))
    assert cv.workspace.entry("T2").expr == "Gp*Cpi/(1 + Gp*Cpi)"


# ============================================================================ Analiz sekmesi
def _setup_loop(app, win):
    L = lti()
    cv = win.control_view
    cv.workspace.add("Gp", L.tf([1], [1, 6, 11, 6]))
    cv.workspace.add("Cpi", L.tf([10, 5], [1, 0]))
    win._show_view(5)
    cv.show_tab("analysis")
    an = cv.analysis_tab
    an.set_plots(["step", "impulse", "bode", "nyquist", "nichols", "rlocus", "pzmap", "sigma", "lsim"])
    an.set_shown(["L", "T", "S", "KS", "GS"])
    an.set_loop(G="Gp", C="Cpi")
    an.recompute()
    pump(app, 0.1)
    return cv, an


def test_analysis_loop_margins_and_stepinfo_match_matlab(app, win, ref):
    need("spt.control.timeresp", "spt.control.freqresp", "spt.control.analysis", "spt.control.connect")
    cv, an = _setup_loop(app, win)
    assert an.messages == [], an.messages
    assert [s.key for s in an.series] == ["L", "T", "S", "KS", "GS"]
    colors = [s.color for s in an.series]
    assert len(set(colors)) == len(colors)                      # bir renk yalnız bir seride
    # pay tablosu: kütüphanenin margin(L)'si birebir; tanım sağlanıyor; MATLAB ile kendi hassasiyeti içinde
    gm, pm, wcg, wcp = an.margin_values["L"]
    _margins_consistent(an.systems["L"], (gm, pm, wcg, wcp), ref, "")
    cells = an.margin_rows()["L = H·G·C"]
    assert cells[1] == f"{20 * math.log10(gm):.5g}" and cells[3] == f"{pm:.5g}" and cells[4] == f"{wcp:.5g}"
    assert cells[5] == ("kararlı" if ref["stable"] else "kararsız")
    # stepinfo(T): MATLAB adım ızgarasında örnekler (sürekli), kütüphane kesin anı bulur → yükselme / tepe zamanı
    # MATLAB'den 1e-4–1e-3 farklı; aşım / tepe / yerleşme 2e-5 içinde; bant genişliği 1e-12
    info = an.step_values["T"]
    for key, mk, rel in (("RiseTime", "rise", 3e-4), ("SettlingTime", "settling", 3e-5),
                         ("Overshoot", "overshoot", 2e-5), ("Peak", "peak", 2e-6), ("PeakTime", "peaktime", 2e-3)):
        assert close(info[key], ref[mk], rel), key
    from spt.control.freqresp import bandwidth
    bw = bandwidth(an.systems["T"])
    assert close(bw, ref["bw"], 1e-12)
    assert an.step_table.item(0, 0).text() == "T: r → y"
    # tabloda 4 anlamlı basamak (1366 genişlikte sığsın), ipucunda tam değer
    assert an.step_table.item(0, 7).text() == f"{bw:.4g}" and close(an.step_table.item(0, 7).toolTip(), bw, 1e-11)
    # KS (D = 10: y(0) = 10 son değerin (6) %90'ını aşar → yükselme 0) MATLAB ile. MATLAB tepe zamanı adım
    # ızgarasının örneği (0.45758); kütüphaneninki kesin tepe: ince ızgaradaki en büyük değerle doğrulanır
    import numpy as np

    from spt.control.timeresp import step
    ks = an.step_values["KS"]
    assert ks["RiseTime"] == ref["ks_rise"] == 0
    for key, mk, rel in (("SettlingTime", "ks_settling", 3e-5), ("Overshoot", "ks_overshoot", 3e-5),
                         ("Peak", "ks_peak", 1e-5), ("PeakTime", "ks_peaktime", 1e-2)):
        assert close(ks[key], ref[mk], rel), key
    yk, _ = step(an.systems["KS"], np.linspace(0.0, 1.0, 100001))
    assert close(float(np.max(yk)), ks["Peak"], 1e-9)
    rows = {an.step_table.item(r, 0).text(): r for r in range(an.step_table.rowCount())}
    r = rows["KS = C·S"]
    assert an.step_table.item(r, 4).text() == f"{ks['Peak']:.4g}" and an.step_table.item(r, 6).text() == "6"
    # S: son değer 0 (yfinal = yinit). MATLAB: aşım ∞, yerleşme NaN, tepe 1 (t = 0), bant genişliği ∞, DC 0.
    # Tablo ya MATLAB'in değerlerini gösterir ya da (kütüphane tutarsızsa) "—"; asla yanlış bir sayı göstermez
    r = rows["S = 1/(1+L)"]
    cells = [an.step_table.item(r, c).text() for c in range(an.step_table.columnCount())]
    assert cells[6] == "0" and ref["s_dc"] == 0 and cells[7] == "∞" and math.isinf(ref["s_bw"])
    assert "Son değer 0" in an.step_table.item(r, 3).toolTip()
    if an.step_values["S"]["Peak"] != 0:
        assert cells[3] == "∞" and math.isinf(ref["s_overshoot"])
        assert cells[2] == "—" and math.isnan(ref["s_settling"])
        assert cells[4] == f"{ref['s_peak']:.4g}" and cells[5] == f"{ref['s_peaktime']:.4g}"
    else:
        assert cells[1:6] == ["—"] * 5, cells
    # grafiklerde veri: her seri her uygun grafikte çizildi
    p = an.plots
    steps = {t.name for t in p["step"].line_traces()}
    assert steps == {"T: r → y", "S = 1/(1+L)", "KS = C·S", "GS = G·S"}   # açık çevrim L zaman grafiğinde yok
    assert len(p["bode"].mag_card.line_traces()) >= 5 and len(p["sigma"].line_traces()) == 5
    assert {t.name for t in p["nyquist"].line_traces()} == {"L = H·G·C", "L = H·G·C (ω < 0)"}
    assert p["rlocus"].loci("L") is not None and p["lsim"].line_traces()
    # Bode'de pay işaretleri MATLAB değerleriyle
    names = p["bode"].mag_card.trace_names()
    assert "L = H·G·C · kazanç payı" in names
    x, y = p["bode"].mag_card.curve_data("L = H·G·C · kazanç payı")
    assert x[0] == wcg and close(-y[1], 20 * math.log10(gm), 1e-12)   # işaret tablodaki değerde
    # kök yer eğrisi ipucu: kollar k = Gm'de sanal ekseni ±jWcg'de keser (Routh): ızgara noktası yakınında
    k, root = p["rlocus"].gain_at(0.0, wcg, "L")
    assert abs(root.real) < 0.05 * wcg and math.isclose(k, gm, rel_tol=0.05)
    tips = p["rlocus"].tips
    tips.clear()
    trace, tx, ty = tips.nearest(root.real, root.imag) or (None, 0, 0)
    assert trace is not None
    tips.add(trace, tx, ty)
    assert "kazanç k" in tips.texts()[0] and "sönüm" in tips.texts()[0]
    # adım ipucu ve karakteristik işaretleri
    assert "T: r → y · tepe" in p["step"].trace_names() and "T: r → y · yerleşme" in p["step"].trace_names()


def test_analysis_discrete_domain_matches_matlab(app, win, ref):
    need("spt.control.timeresp", "spt.control.freqresp", "spt.control.analysis", "spt.control.discretize")
    cv, an = _setup_loop(app, win)
    an.domain.setCurrentIndex(an.domain.findData("discrete"))
    an.ts.setValue(0.1)
    an.method.setCurrentIndex(an.method.findData("zoh"))
    an.recompute()
    assert an.systems["L"].Ts == 0.1
    _margins_consistent(an.systems["L"], an.margin_values["L"], ref, "d")
    info = an.step_values["T"]   # ayrık stepinfo örneklerde tanımlı: MATLAB ile birebir
    assert close(info["RiseTime"], ref["drise"], 1e-9) and close(info["SettlingTime"], ref["dsettling"], 1e-9)
    assert close(info["Overshoot"], ref["dovershoot"], 1e-9)
    # ayrık grafiklerde birim çember ve Nyquist frekansı kılavuzu
    assert an.plots["pzmap"]._discrete


def test_analysis_plot_data_equals_library(app, win):
    """Çevrim sistemleri tanımla (L, T, S, KS, GS kapalı biçimleri, frekans cevabında 1e-10) ve her grafiğin
    verisi kütüphane API'sinin çıktısıyla birebir (arayüz veriyi değiştirmez); veri ipucu seçilen noktanın değeri."""
    need("spt.control.timeresp", "spt.control.freqresp", "spt.control.analysis", "spt.control.connect")
    import numpy as np

    from spt.control.analysis import pole, rlocus
    from spt.control.freqresp import bode, freqresp, nyquist, sigma
    from spt.control.timeresp import impulse, lsim, step

    cv, an = _setup_loop(app, win)
    w = np.logspace(-2, 2, 61)
    g = freqresp(cv.workspace.get("Gp"), w)
    c = freqresp(cv.workspace.get("Cpi"), w)
    lw = g * c
    closed = {"L": lw, "T": lw / (1 + lw), "S": 1 / (1 + lw), "KS": c / (1 + lw), "GS": g / (1 + lw)}
    for key, h in closed.items():
        assert np.allclose(freqresp(an.systems[key], w), h, rtol=1e-10, atol=1e-14), key
    p = an.plots
    T, Lsys = an.systems["T"], an.systems["L"]
    for plot, fn in ((p["step"], step), (p["impulse"], impulse)):
        x, y = plot.curve_data("T: r → y")
        yy, tt = fn(T, plot._common_tfinal(plot.series))   # ortak bitiş: sistemlerin önerdiklerinin en büyüğü
        assert np.array_equal(x, tt) and np.array_equal(y, yy[:, 0, 0])
    mag, phase, wb = bode(T)
    x, y = p["bode"].mag_card.curve_data("T: r → y")
    assert np.array_equal(x, wb) and np.allclose(y, 20 * np.log10(mag[0, 0]), rtol=0, atol=1e-12)
    x, y = p["bode"].phase_card.curve_data("T: r → y")
    assert np.array_equal(y, phase[0, 0])
    re, im, wn = nyquist(Lsys)
    ok = np.isfinite(re[0, 0]) & np.isfinite(im[0, 0])
    x, y = p["nyquist"].curve_data("L = H·G·C")
    assert np.array_equal(x, re[0, 0][ok]) and np.array_equal(y, im[0, 0][ok])
    x, y = p["nyquist"].curve_data("L = H·G·C (ω < 0)")
    assert np.array_equal(y, -im[0, 0][ok])
    mag, phase, _ = bode(Lsys)
    x, y = p["nichols"].curve_data("L = H·G·C")
    assert np.array_equal(x, phase[0, 0]) and np.allclose(y, 20 * np.log10(mag[0, 0]), rtol=0, atol=1e-12)
    sv, _ = sigma(T)
    x, y = p["sigma"].curve_data("T: r → y")
    assert np.allclose(y, 20 * np.log10(sv[0]), rtol=0, atol=1e-12)
    r, k = rlocus(Lsys)
    rr, kk = p["rlocus"].loci("L")[:2]
    assert np.array_equal(rr, np.asarray(r).T) and np.array_equal(kk, k)
    x, y = p["pzmap"].curve_data("T: r → y · kutup")
    assert np.array_equal(np.sort_complex(x + 1j * y), np.sort_complex(pole(T)))
    lp = p["lsim"]
    x, y = lp.curve_data("T: r → y")
    assert np.array_equal(y, lsim(T, lp.u, lp.t)[0][:, 0])
    # veri ipucu: Bode genliğinde bir ızgara noktası → ω ve |T(jω)| (dB) kütüphanenin frekans cevabıyla
    card = p["bode"].mag_card
    x, y = card.curve_data("T: r → y")
    k = int(np.argmin(np.abs(x - 1.0)))
    card.tips.clear()
    hit = card.tips.nearest(math.log10(x[k]), y[k])
    assert hit is not None and hit[0].name == "T: r → y"
    card.tips.add(*hit)
    hx = complex(freqresp(T, [hit[1]])[0, 0, 0])
    text = card.tips.texts()[0]
    assert f"genlik = {20 * math.log10(abs(hx)):.4f} dB" in text and f"ω = {hit[1]:.6g} rad/s" in text
    card.tips.clear()


def test_analysis_overlays_unique_colors_and_limits(app, win):
    need("spt.control.timeresp", "spt.control.freqresp", "spt.control.analysis")
    L = lti()
    cv, an = _setup_loop(app, win)
    for i in range(5):
        cv.workspace.add(f"M{i}", L.tf([1.0 + i], [1, 1.0 + i]))
    pump(app, 0.1)
    an.set_extra_models([f"M{i}" for i in range(5)])
    an.recompute()
    colors = [s.color for s in an.series]
    assert len(an.series) == 8 and len(set(colors)) == 8         # 5 çevrim + 3 model; 9. ve 10. reddedilir
    assert sum("en fazla" in m for m in an.messages) == 2
    an.set_shown(["T"])
    an.recompute()
    assert [s.key for s in an.series] == ["T", "m:M0", "m:M1", "m:M2", "m:M3", "m:M4"]
    assert len({s.color for s in an.series}) == 6


def test_analysis_screenshots_have_no_overlap(app, win, tmp_path):
    """1366×768 ve 1600×960'ta görünüm sığar: kurulum kartı ve tablolar kırpılmaz, ızgara kaydırılır."""
    cv, an = _setup_loop(app, win) if _has_calc() else (win.control_view, win.control_view.analysis_tab)
    for size in ((1366, 768), (1600, 960)):
        win.resize(*size)
        pump(app, 0.3)
        assert an.width() >= an.minimumSizeHint().width()
        img = win.grab()
        assert not img.isNull()
        img.save(str(tmp_path / f"analiz_{size[0]}.png"))


def _has_calc() -> bool:
    import importlib.util
    return all(importlib.util.find_spec(m) is not None for m in
               ("spt.control.freqresp", "spt.control.timeresp", "spt.control.analysis", "spt.control.connect"))


def test_lsim_uses_real_snapshot_data_and_measured_output(app, win):
    need("spt.control.timeresp", "spt.control.freqresp", "spt.control.analysis")
    import numpy as np

    L = lti()
    from spt.control.timeresp import lsim
    from spt.core.local_ring import StaticRing
    from spt.core.snapshot import Snapshot
    from spt.protocol.format import SAMPLE_TYPES_BY_NAME, ChannelInfo, Descriptor, Layout

    fs = 100.0
    t = np.arange(1000) / fs
    u = (np.mod(t, 4.0) < 2.0).astype(float)
    plant = L.tf([2.0], [1.0, 1.0])
    y = lsim(plant, u, t)[0][:, 0]
    desc = Descriptor(1, 1, fs, SAMPLE_TYPES_BY_NAME["float64"], Layout.INTERLEAVED, "deney",
                      (ChannelInfo("u", "V"), ChannelInfo("y", "V")))
    snap = Snapshot(1, "deney", "test", desc, 0.0, StaticRing(np.column_stack([u, y]), 0, []))
    cv = win.control_view
    cv.workspace.add("Gm", plant)
    win._show_view(5)
    cv.show_tab("analysis")
    an = cv.analysis_tab
    win.control_view.set_sources([("ws1", "Çalışma alanı · deney", snap.handle())])
    an.set_shown([])
    an.set_extra_models(["Gm"])
    an.set_plots(["lsim"])
    an.lsim_source.setCurrentIndex(an.lsim_source.findData("src:ws1"))
    an.lsim_channel.setCurrentIndex(an.lsim_channel.findData(0))
    an.lsim_measured.setCurrentIndex(an.lsim_measured.findData(1))
    an.recompute()
    plot = an.plots["lsim"]
    assert an.messages == [], an.messages
    x, ym = plot.curve_data("Gm")
    assert np.allclose(x, t) and np.allclose(ym, y, rtol=0, atol=1e-12)     # modelin kendisi: ölçümle aynı
    names = plot.trace_names()
    assert "giriş: u [V]" in names and "ölçülen: y [V]" in names
    colors = [tr.color for tr in plot._traces if tr.name in ("Gm", "ölçülen: y [V]", "giriş: u [V]")]
    assert len(set(colors)) == 3
    # ölçülen çıkış görünümdeki hiçbir sistemle renk paylaşmaz (çevrim cevapları ve açık çevrim L dahil); lejantta
    an.set_loop(G="Gm")
    an.set_shown(["L", "T", "S", "KS", "GS"])
    an.recompute()
    meas = next(tr.color for tr in plot._traces if tr.name == "ölçülen: y [V]")
    view_colors = [s.color for s in an.series] + [meas]
    assert len(view_colors) == 7 and len(set(view_colors)) == 7
    assert meas == an._measured_color
    an.set_shown([])
    # ayrık sistemde verinin örnekleme aralığı Ts'ye eşit olmalı: değilse açık hata
    cv.workspace.add("Gd", L.tf([0.1], [1, -0.9], 0.05))
    an.set_extra_models(["Gd"])
    an.recompute()
    assert any("örnekleme aralığı" in m for m in an.messages)
