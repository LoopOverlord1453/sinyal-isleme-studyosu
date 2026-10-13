"""Kontrolcü Tasarımı · Durum uzayı paneli: kutup yerleştirme (place / acker), LQR / LQI (Q, R, N; Bryson kuralı),
gözlemci (kutup yerleştirmeyle), Kalman (kalman / kalmd) ve LQG izleyici.

Tasarım bitkisi seçilen modelin durum uzayı gerçeklemesidir (`ss`); ayrık tasarımda c2d ile (seçili yöntem)
ayrıklaştırılır, ayrık gecikmeler duruma katılır (`absorb_delay`). Kazançlar yalnız `spt.control.modern`'den gelir.

Kapalı çevrimler (tam durum ölçümüyle):
    nbar (place / acker / LQR):  u = N̄·r − K·x                   N̄ = nbar(sys, K)
    LQI:                         u = −Kx·x − Ki·xi,  xi' = r − y   (ayrıkta xi[k+1] = xi[k] + Ts·(r − y), MATLAB lqi)
Denetleyici [r; y; x] girişli bir model olarak yazılır, bitki [y; x] çıkışlıdır; çevrim `feedback(…, +1)` ile kapanır.
Gözlemci / Kalman: hata dinamiği e' = (A − L·C)·e (öngörü kazancı L); ayrılma ilkesi gereği r → y aktarımı durum
geri beslemesininkiyle aynıdır, bu yüzden kestiricinin etkisi kestirim hatası cevabıyla (e(0) = 1) gösterilir.
LQG: `lqg(sys, blkdiag(Qx, R), blkdiag(G·Qn·Gᵀ, Rn), QI)` 2 serbestlik dereceli izleyici (girişler [r; y]).
"""
from __future__ import annotations

import logging
import math

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLineEdit,
    QMenu,
    QPushButton,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from spt.control.errors import ControlError
from spt.gui.control import control_plots as cp
from spt.gui.control import design_core as dc
from spt.gui.control.design_widgets import (
    CurvesCard,
    DesignStepPlot,
    capped_tfinal,
    mono_label,
    responses,
    scroll_area,
)
from spt.gui.control.design_widgets import spin_box as dc_spin
from spt.gui.control.workspace import SOURCE_DESIGN
from spt.gui.theme import ThemedMap, ThemeManager
from spt.gui.widgets.common import Card, CollapsibleSection, combo, label

log = logging.getLogger("spt.gui")

CTRL_METHODS = (("place", "Kutup yerleştirme (place)"), ("acker", "Ackermann (acker, tek giriş)"),
                ("lqr", "LQR"), ("lqi", "LQI (integral eylemli)"))
EST_METHODS = (("none", "Yok (tam durum ölçümü)"), ("observer", "Gözlemci (kutup yerleştirme)"),
               ("kalman", "Kalman süzgeci"))
COLORS = ThemedMap(lambda: {"T": cp.PALETTE[0], "G": cp.PALETTE[1], "obs": cp.PALETTE[2], "KS": cp.PALETTE[3],
                            "T_lqg": cp.PALETTE[4], "KS_lqg": cp.PALETTE[5]})


# ============================================================================ Qt'siz tasarım
def poles_from_spec(n: int, zeta: float, wn: float, Ts: float = 0.0) -> np.ndarray:
    """n kutup: baskın çift −ζωn ± jωn√(1−ζ²), kalanlar −ωn·(2 + 0.5k) (ayrık, birbirinden farklı); ayrıkta
    z = e^{sTs}."""
    if n <= 0:
        return np.zeros(0, complex)
    if not wn > 0:
        raise ControlError("ωn pozitif olmalı.")
    if n == 1:
        s = [complex(-wn)]
    elif zeta < 1:
        wd = wn * math.sqrt(1 - zeta * zeta)
        s = [complex(-zeta * wn, wd), complex(-zeta * wn, -wd)]
    else:
        s = [complex(-wn), complex(-1.1 * wn)]
    s += [complex(-wn * (2.0 + 0.5 * k)) for k in range(n - len(s))]
    s = np.array(s, complex)
    return np.exp(s * Ts) if Ts > 0 else s


def _design_plant(sys, Ts: float, method: str):
    """Tasarım bitkisi (ss, tasarım zamanında, gecikmeler duruma katılmış) ve sürekli asıl (lqrd / kalmd için)."""
    lti = dc.lti_module()
    if sys is None:
        raise ControlError("Bitki seçin.")
    P = dc.to_domain(sys, Ts, method)
    if P.has_delay():
        if P.is_discrete:
            from spt.control.discretize import absorb_delay
            P = absorb_delay(P)
        else:
            raise ControlError("Sürekli zamanda gecikmeli bitki: Padé yaklaşımı uygulayın ya da ayrık tasarlayın "
                               "(gecikme duruma katılır).")
    P = lti.ss(P)
    if P.nx == 0:
        raise ControlError("Bitkinin durumu yok (statik kazanç).")
    cont = None
    if Ts > 0 and sys.Ts == 0 and not sys.has_delay():
        cont = lti.ss(sys)
    return P, cont


def design_modern(sys, Ts: float, method: str, o: dict) -> dict:
    """Durum uzayı tasarımı. o: ctrl, poles, Q, R, N, lqrd, est, obs_poles, obs_factor, Qn, Rn, noise, kalmd,
    kf_form, lqg, QI (metinler ayrıştırılmış ya da ham). Dönüş: kazançlar, kapalı çevrimler, kutuplar."""
    from spt.control import modern
    from spt.control.analysis import pole
    from spt.control.connect import feedback
    lti = dc.lti_module()
    P, Pc = _design_plant(sys, Ts, method)
    A, B, C, D = P.A, P.B, P.C, P.D
    n, m, ny = A.shape[0], B.shape[1], C.shape[0]
    out = {"P": P, "n": n, "m": m, "ny": ny, "Ts": Ts, "notes": []}
    ctrl = o["ctrl"]
    if ctrl in ("place", "acker"):
        p = dc.parse_roots(o["poles"], "Denetleyici kutupları") if isinstance(o["poles"], str) else o["poles"]
        K = modern.place(A, B, p) if ctrl == "place" else modern.acker(A, B, p)
    else:
        nq = n + ny if ctrl == "lqi" else n
        Q = dc.parse_matrix(o["Q"], nq, "Q") if isinstance(o["Q"], str) else np.asarray(o["Q"], float)
        R = dc.parse_matrix(o["R"], m, "R") if isinstance(o["R"], str) else np.asarray(o["R"], float)
        Ntext = o.get("N") or ""
        N = None
        if isinstance(Ntext, str) and Ntext.strip():
            N = np.asarray(dc.parse_vector(Ntext, "N"), float).reshape(nq, m)
        elif not isinstance(Ntext, str):
            N = np.asarray(Ntext, float)
        out["Q"], out["R"] = Q, R
        if ctrl == "lqi":
            K, _S, _e = modern.lqi(P, Q, R, N)
        elif o.get("lqrd") and Pc is not None:
            K, _S, _e = modern.lqrd(Pc, Q, R, Ts) if N is None else modern.lqrd(Pc, Q, R, N, Ts)
            out["notes"].append("lqrd: sürekli maliyetin ayrık eşdeğeri (Van Loan)")
        else:
            K, _S, _e = modern.lqr(P, Q, R) if N is None else modern.lqr(P, Q, R, N)
    K = np.atleast_2d(np.asarray(K, float))
    out["K"] = K
    Kx = K[:, :n]
    Ki = K[:, n:] if ctrl == "lqi" else None
    out["Kx"], out["Ki"] = Kx, Ki
    # denetleyici: girişler [r; y; x] → u
    if ctrl == "lqi":
        q = Ts if Ts > 0 else 1.0
        Ac = np.eye(ny) if Ts > 0 else np.zeros((ny, ny))
        Bc = q * np.hstack([np.eye(ny), -np.eye(ny), np.zeros((ny, n))])
        Kc = lti.ss(Ac, Bc, -Ki, np.hstack([np.zeros((m, 2 * ny)), -Kx]), Ts)
        out["Nbar"] = None
    else:
        Nbar = modern.nbar(P, Kx)
        out["Nbar"] = np.atleast_2d(Nbar)
        Kc = lti.ss(np.hstack([out["Nbar"], np.zeros((m, ny)), -Kx]), Ts=Ts)
    out["Kc"] = Kc
    P2 = lti.ss(A, B, np.vstack([C, np.eye(n)]), np.vstack([D, np.zeros((n, m))]), Ts)
    fb = list(range(ny, 2 * ny + n))
    closed = feedback(P2 * Kc, np.eye(ny + n), +1, fb, list(range(ny + n)))
    T = closed[0:ny, 0:ny]
    KS = feedback(Kc, P2, +1, fb, list(range(m)))[:, 0:ny]
    out["T"], out["KS"] = T, KS
    out["cl_poles"] = np.asarray(pole(closed), complex)
    # kestirici
    est = o["est"]
    out["est"] = est
    L = None
    if est == "observer":
        txt = o.get("obs_poles") or ""
        if isinstance(txt, str) and txt.strip():
            po = dc.parse_roots(txt, "Gözlemci kutupları")
        else:
            f = float(o.get("obs_factor", 3.0))
            base = np.linalg.eigvals(A - B @ Kx)
            po = np.power(base.astype(complex), f) if Ts > 0 else base * f
            po = np.where(np.abs(po.imag) < 1e-12 * np.maximum(1.0, np.abs(po)), po.real, po)
        L = modern.place(A.T, C.T, po).T
        out["obs_poles_req"] = po
        out["form"] = "delayed"
    elif est == "kalman":
        noise = o.get("noise", "B")
        Gn = B if noise == "B" else np.eye(n)
        nw = Gn.shape[1]
        Qn = dc.parse_matrix(o["Qn"], nw, "Qn") if isinstance(o["Qn"], str) else np.asarray(o["Qn"], float)
        Rn = dc.parse_matrix(o["Rn"], ny, "Rn") if isinstance(o["Rn"], str) else np.asarray(o["Rn"], float)
        Hn = D if noise == "B" else np.zeros((ny, n))
        form = o.get("kf_form", "current")
        if Ts > 0 and o.get("kalmd") and Pc is not None:
            Gc = Pc.B if noise == "B" else np.eye(n)
            sysw = lti.ss(Pc.A, np.hstack([Pc.B, Gc]), Pc.C, np.hstack([Pc.D, np.zeros((ny, Gc.shape[1]))]))
            kf = modern.kalmd(sysw, Qn, Rn, Ts)
            out["notes"].append("kalmd: sürekli gürültünün ayrık eşdeğeri")
            form = "current"
        else:
            sysw = lti.ss(A, np.hstack([B, Gn]), C, np.hstack([D, Hn]), Ts)
            kf = modern.kalman(sysw, Qn, Rn, type=form if Ts > 0 else "current")
        L = np.asarray(kf.L, float)
        out["kf"] = kf
        out["Qn"], out["Rn"], out["Gn"] = Qn, Rn, Gn
        out["form"] = form if Ts > 0 else "delayed"
        if o.get("lqg"):
            if ctrl == "lqi":
                Qx, QI = out["Q"][:n, :n], out["Q"][n:, n:]
            elif ctrl == "lqr":
                Qx = out["Q"]
                QI = dc.parse_matrix(o.get("QI") or "1", ny, "QI")
            else:
                raise ControlError("LQG, LQR ya da LQI ağırlıklarıyla kurulur (Durum geri beslemesi: LQR / LQI).")
            R = out["R"]
            QXU = np.block([[Qx, np.zeros((n, m))], [np.zeros((m, n)), R]])
            Qnx = Gn @ Qn @ Gn.T
            QWV = np.block([[Qnx, np.zeros((n, ny))], [np.zeros((ny, n)), Rn]])
            Klqg = modern.lqg(P, QXU, QWV, QI)
            out["lqg"] = Klqg
            fbq = list(range(ny, 2 * ny))
            out["T_lqg"] = feedback(P * Klqg, np.eye(ny), +1, fbq, list(range(ny)))[:, 0:ny]
            out["KS_lqg"] = feedback(Klqg, P, +1, fbq, list(range(m)))[:, 0:ny]
    out["L"] = L
    if L is not None:
        Ae = A - L @ C
        out["obs_poles"] = np.linalg.eigvals(Ae)
        out["err"] = lti.ss(Ae, np.zeros((n, 1)), np.eye(n), np.zeros((n, 1)), Ts)
    return out


def controller_spec(res: dict):
    """C kodu tanımı (spt.codegen.control_c): tam durumlu → statik kazanç [N̄, −K] ya da LQI denetleyicisi;
    kestiricili → ObserverSpec (gözlemci: öngörü biçimi; Kalman: seçilen biçim, M = Mx)."""
    from spt.codegen import control_c as cc
    Ts = res["Ts"]
    if not Ts > 0:
        raise ControlError("C kodu için ayrık tasarım gerekir: Zaman alanı → Ayrık (Ts) seçin.")
    P = res["P"]
    lqi = res["Ki"] is not None
    if res["L"] is None:
        if lqi:
            return res["Kc"]
        return cc.LtiSpec.gain(np.hstack([res["Nbar"], -res["Kx"]]), Ts)
    M = None
    form = res.get("form", "delayed")
    if res["est"] == "kalman" and form == "current":
        M = np.asarray(res["kf"].Mx, float)
    return cc.ObserverSpec(P.A, P.B, P.C, P.D, res["K"], res["L"], Ts, form=form, M=M,
                           reference="lqi" if lqi else "nbar", Nbar=None if lqi else res["Nbar"])


def kalman_spec(res: dict):
    from spt.codegen import control_c as cc
    if res.get("kf") is None or not res["Ts"] > 0:
        raise ControlError("Kalman süzgeci kodu için ayrık Kalman tasarımı gerekir.")
    P = res["P"]
    kf = res["kf"]
    form = res.get("form", "current")
    return cc.KalmanSpec(P.A, P.B, P.C, P.D, np.asarray(kf.L, float), res["Ts"],
                         M=np.asarray(kf.Mx, float) if form == "current" else None, form=form,
                         My=np.asarray(kf.My, float) if form == "current" and kf.My is not None else None)


# ============================================================================ görünüm
class ModernView(QWidget):
    def __init__(self, theme: ThemeManager):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.legend = cp.SeriesLegend()
        lay.addWidget(self.legend)
        split = QSplitter(Qt.Orientation.Vertical)
        split.setChildrenCollapsible(False)
        top = QSplitter(Qt.Orientation.Horizontal)
        top.setChildrenCollapsible(False)
        bottom = QSplitter(Qt.Orientation.Horizontal)
        bottom.setChildrenCollapsible(False)
        self.step = DesignStepPlot(theme, "Adım Cevabı · r → y")
        self.effort = DesignStepPlot(theme, "Kontrol Sinyali · r → u")
        self.effort.markers = {"peak"}
        self.pz = cp.PZMapPlot(theme)
        self.pz.title.setText("Kutuplar · açık / kapalı çevrim / kestirici")
        self.err = CurvesCard(theme, "Kestirim Hatası · e(0) = 1")
        self.gains = Card("Kazançlar")
        self.text = mono_label()
        self.text.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.gains.body.addWidget(scroll_area(self.text, horizontal=True), 1)
        for w in (self.step, self.effort):
            top.addWidget(w)
        for w in (self.pz, self.err, self.gains):
            bottom.addWidget(w)
        bottom.setSizes([330, 330, 340])
        split.addWidget(top)
        split.addWidget(bottom)
        lay.addWidget(split, 1)

    def detachable_cards(self) -> list[tuple[str, str, Card]]:
        return [("ctl_ds_ss_step", "Durum Uzayı · Adım", self.step),
                ("ctl_ds_ss_effort", "Durum Uzayı · Kontrol Sinyali", self.effort),
                ("ctl_ds_ss_poles", "Durum Uzayı · Kutuplar", self.pz),
                ("ctl_ds_ss_err", "Durum Uzayı · Kestirim Hatası", self.err),
                ("ctl_ds_ss_gains", "Durum Uzayı · Kazançlar", self.gains)]


def _line(text: str, tip: str) -> QLineEdit:
    w = QLineEdit(text)
    w.setToolTip(tip)
    return w


class ModernPanel(QWidget):
    def __init__(self, tab, view: ModernView):
        super().__init__()
        self.tab = tab
        self.view = view
        self.result: dict | None = None
        self._Ts = 0.0
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        sec = CollapsibleSection("Bitki (durum uzayı)")
        self.plant = combo([])
        self.plant.setToolTip("Tasarım bitkisi (çok girişli / çok çıkışlı olabilir); ayrık tasarımda c2d ile")
        self.plant.currentIndexChanged.connect(self._plant_changed)
        sec.add_row("Bitki", self.plant)
        self.info = label("", "Muted")
        self.info.setWordWrap(True)
        sec.form.addRow(self.info)
        lay.addWidget(sec)

        fb = CollapsibleSection("Durum geri beslemesi")
        self.ctrl = combo([t for _k, t in CTRL_METHODS])
        self.ctrl.currentIndexChanged.connect(self._sync)
        fb.add_row("Yöntem", self.ctrl)
        self.poles = _line("", "İstenen kapalı çevrim kutupları (sürekli: s, ayrık: z düzlemi), ör. −2±2j, −5")
        self._poles_label = label("Kutuplar (s)", "FormLabel")
        fb.form.addRow(self._poles_label, self.poles)
        self.zeta = dc_spin(0.0, 0.999, 0.7, 3, "", "Baskın çiftin sönümü")
        fb.add_row("ζ", self.zeta)
        self.wn = dc_spin(1e-6, 1e9, 2.0, 4, " rad/s", "Baskın çiftin doğal frekansı")
        fb.add_row("ωn", self.wn)
        fill = QPushButton("Kutupları ζ, ωn'den doldur")
        fill.clicked.connect(self.fill_poles)
        fb.form.addRow(fill)
        self.Q = _line("1", "Durum ağırlığı Q: köşegen (1, 10), tam ([1 0; 0 2]) ya da sayı·I. LQI'da boyut nx + ny "
                            "(son ny eleman integral durumları)")
        fb.add_row("Q", self.Q)
        self.R = _line("1", "Giriş ağırlığı R (m×m)")
        fb.add_row("R", self.R)
        self.N = _line("", "Çapraz ağırlık N (isteğe bağlı; satır satır n×m eleman)")
        fb.add_row("N", self.N)
        self.xmax = _line("", "Bryson kuralı: durumların kabul edilebilir en büyük değerleri (inf = ağırlık yok)")
        fb.add_row("x en büyük", self.xmax)
        self.umax = _line("", "Bryson kuralı: girişlerin kabul edilebilir en büyük değerleri")
        fb.add_row("u en büyük", self.umax)
        br = QPushButton("Bryson kuralıyla Q, R doldur")
        br.setToolTip("Q = diag(1/x²), R = diag(1/u²) (modern.bryson)")
        br.clicked.connect(self.fill_bryson)
        fb.form.addRow(br)
        self.lqrd = QCheckBox("Ayrıkta lqrd (sürekli maliyetin ayrık eşdeğeri)")
        self.lqrd.setToolTip("Sürekli bitkide ayrık LQR'ı sürekli maliyetin tam ayrık eşdeğeriyle (lqrd) tasarla")
        fb.form.addRow(self.lqrd)
        lay.addWidget(fb)
        self._fb = fb

        es = CollapsibleSection("Kestirici")
        self.est = combo([t for _k, t in EST_METHODS])
        self.est.currentIndexChanged.connect(self._sync)
        es.add_row("Tür", self.est)
        self.obs_poles = _line("", "Gözlemci kutupları (boş: denetleyici kutuplarının k katı hızda)")
        es.add_row("Kutuplar", self.obs_poles)
        self.obs_factor = dc_spin(1.0, 50.0, 4.0, 2, "×", "Boş bırakılırsa gözlemci kutupları = k × denetleyici "
                                                          "kutupları (ayrıkta z^k)")
        es.add_row("Hız katı k", self.obs_factor)
        self.noise = combo(["Girişle aynı yoldan (G = B)", "Her duruma (G = I)"])
        self.noise.setToolTip("Süreç gürültüsünün bitkiye girdiği yol")
        es.add_row("Süreç gürültüsü", self.noise)
        self.Qn = _line("1", "Süreç gürültüsü kovaryansı Qn (G = B: m×m, G = I: n×n)")
        es.add_row("Qn", self.Qn)
        self.Rn = _line("0.01", "Ölçüm gürültüsü kovaryansı Rn (ny×ny)")
        es.add_row("Rn", self.Rn)
        self.kf_form = combo(["current (x̂[n|n])", "delayed (x̂[n|n−1])"])
        es.add_row("Ayrık biçim", self.kf_form)
        self.kalmd = QCheckBox("Ayrıkta kalmd (sürekli gürültünün ayrık eşdeğeri)")
        es.form.addRow(self.kalmd)
        self.lqg = QCheckBox("LQG izleyici (lqg, integral eylemli)")
        self.lqg.setToolTip("Kalman + LQR / LQI ağırlıklarıyla 2 serbestlik dereceli LQG izleyici")
        es.form.addRow(self.lqg)
        self.QI = _line("1", "LQG integral ağırlığı QI (LQR seçiliyken; LQI'da Q'nun son ny bloğu kullanılır)")
        es.add_row("QI", self.QI)
        lay.addWidget(es)
        self._es = es

        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 4, 0, 0)
        b = QPushButton("Tasarla")
        b.clicked.connect(self.recompute)
        rl.addWidget(b)
        add = QToolButton()
        add.setText("Modellere ekle ▾")
        add.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        m = QMenu(add)
        m.addAction("Kapalı çevrim T (r → y)", lambda: self.add_to_workspace("T"))
        m.addAction("Denetleyici", lambda: self.add_to_workspace("K"))
        m.addAction("Kestirici (Kalman / gözlemci)", lambda: self.add_to_workspace("est"))
        add.setMenu(m)
        rl.addWidget(add)
        code = QToolButton()
        code.setText("C Kodu Üret ▾")
        code.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        cm = QMenu(code)
        cm.addAction("Denetleyici (geri besleme + kestirici)", lambda: self.generate_code("ctrl"))
        cm.addAction("Yalnız Kalman süzgeci", lambda: self.generate_code("kalman"))
        code.setMenu(cm)
        rl.addWidget(code)
        rl.addStretch(1)
        lay.addWidget(row)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self.tab.schedule)
        for w in (self.poles, self.Q, self.R, self.N, self.obs_poles, self.Qn, self.Rn, self.QI):
            w.editingFinished.connect(self._timer.start)
        for w in (self.noise, self.kf_form):
            w.currentIndexChanged.connect(self._timer.start)
        for w in (self.lqrd, self.kalmd, self.lqg):
            w.toggled.connect(self._timer.start)
        self.obs_factor.valueChanged.connect(self._timer.start)
        self._sync()

    # ---------------------------------------------------------------- yardımcılar
    def fill_plants(self) -> None:
        ws = self.tab.workspace
        names = ws.lti_names() if ws is not None else []
        cur = self.plant.currentData()
        had_models = self.plant.count() > 1
        self.plant.blockSignals(True)
        self.plant.clear()
        self.plant.addItem("— seçilmedi —", None)
        for n in names:
            self.plant.addItem(n, n)
        i = self.plant.findData(cur)
        if names and (i < 0 or (cur is None and not had_models)):
            g = self.tab.boxes["G"].currentData()
            i = self.plant.findData(g) if g else 1
            i = i if i >= 0 else 1
        self.plant.setCurrentIndex(max(i, 0))
        self.plant.blockSignals(False)
        self._plant_info()

    def set_plant(self, name: str) -> None:
        i = self.plant.findData(name)
        if i >= 0:
            self.plant.setCurrentIndex(i)

    def _sys(self):
        name = self.plant.currentData()
        if name is None or self.tab.workspace is None:
            return None
        return self.tab.workspace.lti(name)

    def _plant_info(self) -> None:
        try:
            sys = self._sys()
        except ControlError as exc:
            self.info.setText(str(exc))
            return
        if sys is None:
            self.info.setText("Bitki seçin.")
            return
        try:
            P, _c = _design_plant(sys, self.tab.design_Ts(), self.tab.method())
            self.info.setText(f"nx = {P.nx}, girişler m = {P.ninputs}, çıkışlar ny = {P.noutputs}"
                              + (f", ayrık Ts = {P.Ts:g} s" if P.Ts > 0 else ", sürekli"))
        except ControlError as exc:
            self.info.setText(str(exc))

    def _plant_changed(self, *_) -> None:
        self._plant_info()
        if not self.poles.text().strip():
            self.fill_poles(schedule=False)
        self.tab.schedule()

    def _sync(self, *_) -> None:
        ctrl = CTRL_METHODS[self.ctrl.currentIndex()][0]
        est = EST_METHODS[self.est.currentIndex()][0]
        for w, show in ((self.poles, ctrl in ("place", "acker")), (self.zeta, ctrl in ("place", "acker")),
                        (self.wn, ctrl in ("place", "acker")), (self.Q, ctrl in ("lqr", "lqi")),
                        (self.R, ctrl in ("lqr", "lqi")), (self.N, ctrl in ("lqr", "lqi")),
                        (self.xmax, ctrl in ("lqr", "lqi")), (self.umax, ctrl in ("lqr", "lqi"))):
            w.setEnabled(show)
        self.lqrd.setEnabled(ctrl == "lqr" and self.tab.design_Ts() > 0)
        for w, on in ((self.obs_poles, est == "observer"), (self.obs_factor, est == "observer"),
                      (self.noise, est == "kalman"), (self.Qn, est == "kalman"), (self.Rn, est == "kalman"),
                      (self.kf_form, est == "kalman" and self.tab.design_Ts() > 0),
                      (self.kalmd, est == "kalman" and self.tab.design_Ts() > 0), (self.lqg, est == "kalman"),
                      (self.QI, est == "kalman" and ctrl == "lqr")):
            w.setEnabled(on)
        self._timer.start()

    def on_domain(self) -> None:
        """Tasarım zamanı değişti: kutup metinleri yeni düzleme taşınır (z = e^{sTs}, s = ln(z)/Ts)."""
        Ts = self.tab.design_Ts()
        old = self._Ts
        if abs(Ts - old) > 1e-15:
            for w in (self.poles, self.obs_poles):
                t = w.text().strip()
                if not t:
                    continue
                try:
                    p = dc.parse_roots(t)
                    s = p if old == 0 else np.log(p.astype(complex)) / old
                    q = s if Ts == 0 else np.exp(s * Ts)
                    w.setText(", ".join(dc.fmt_root(v, 12) for v in q))
                except (ControlError, ValueError, FloatingPointError):
                    w.setText("")
            self._Ts = Ts
        self._poles_label.setText("Kutuplar (z)" if Ts > 0 else "Kutuplar (s)")
        self._plant_info()
        self._sync()

    def fill_poles(self, *_, schedule: bool = True) -> None:
        try:
            sys = self._sys()
            if sys is None:
                return
            P, _c = _design_plant(sys, self.tab.design_Ts(), self.tab.method())
            p = poles_from_spec(P.nx, self.zeta.value(), self.wn.value(), self.tab.design_Ts())
            self.poles.setText(", ".join(dc.fmt_root(v, 12) for v in p))
        except ControlError as exc:
            self.tab.show_message(str(exc))
            return
        if schedule:
            self.tab.schedule()

    def fill_bryson(self) -> None:
        from spt.control.modern import bryson
        try:
            x = dc.parse_vector(self.xmax.text(), "x en büyük", allow_empty=False)
            u = dc.parse_vector(self.umax.text(), "u en büyük", allow_empty=False)
            Q, R = bryson(x, u)
        except ControlError as exc:
            self.tab.show_message(f"Bryson: {exc}")
            return
        self.Q.setText(", ".join(f"{v:.8g}" for v in np.diag(Q)))
        self.R.setText(", ".join(f"{v:.8g}" for v in np.diag(R)))
        self.tab.schedule()

    def options(self) -> dict:
        return {"ctrl": CTRL_METHODS[self.ctrl.currentIndex()][0], "poles": self.poles.text(),
                "Q": self.Q.text(), "R": self.R.text(), "N": self.N.text(), "lqrd": self.lqrd.isChecked(),
                "est": EST_METHODS[self.est.currentIndex()][0], "obs_poles": self.obs_poles.text(),
                "obs_factor": self.obs_factor.value(), "noise": "B" if self.noise.currentIndex() == 0 else "I",
                "Qn": self.Qn.text(), "Rn": self.Rn.text(), "kalmd": self.kalmd.isChecked(),
                "kf_form": "current" if self.kf_form.currentIndex() == 0 else "delayed",
                "lqg": self.lqg.isChecked(), "QI": self.QI.text()}

    # ---------------------------------------------------------------- hesap
    def recompute(self) -> None:
        self._timer.stop()
        v = self.view
        if CTRL_METHODS[self.ctrl.currentIndex()][0] in ("place", "acker") and not self.poles.text().strip():
            self.fill_poles(schedule=False)
        try:
            sys = self._sys()
            res = design_modern(sys, self.tab.design_Ts(), self.tab.method(), self.options())
        except (ControlError, np.linalg.LinAlgError, ValueError) as exc:
            self.result = None
            for card in (v.step, v.effort, v.pz):
                card.set_series([])
            v.err.show([])
            v.legend.set_series([])
            v.text.setText("")
            self.tab._set_status([f"Durum uzayı tasarımı: {exc}"])
            return
        self.result = res
        msgs = list(res["notes"])
        lti = dc.lti_module()
        steps = [cp.Series("T: r → y (durum geri beslemesi)", res["T"], COLORS["T"], "T")]
        efforts = [cp.Series("r → u (durum geri beslemesi)", res["KS"], COLORS["KS"], "KS")]
        if res.get("T_lqg") is not None:
            steps.append(cp.Series("T: r → y (LQG)", res["T_lqg"], COLORS["T_lqg"], "T_lqg"))
            efforts.append(cp.Series("r → u (LQG)", res["KS_lqg"], COLORS["KS_lqg"], "KS_lqg"))
        pz = [cp.Series("açık çevrim G", res["P"], COLORS["G"], "G"), steps[0]]
        if res.get("obs_poles") is not None:
            pz.append(cp.Series("kestirici A − L·C", lti.zpk([], res["obs_poles"], 1.0, res["Ts"]), COLORS["obs"],
                                "obs"))
        if res.get("T_lqg") is not None:
            pz.append(steps[1])
        v.pz.set_series(pz)
        legend = [(s.color, s.name) for s in steps + efforts] + [(COLORS["G"], "açık çevrim G")]
        if res.get("obs_poles") is not None:
            legend.append((COLORS["obs"], "kestirici (A − L·C)"))
        v.legend.set_series(legend)
        v.text.setText(self._summary(res))
        # adım cevapları ve kestirim hatası arka planda
        cards = ((v.step, steps), (v.effort, efforts))
        items = {s.key: s.sys for _c, ser in cards for s in ser}
        tfinal, _cut = capped_tfinal(list(items.values()))
        err, n = res.get("err"), res["n"]
        self._pending = (cards, tfinal, msgs, res)

        def job(items=items, tf=tfinal, err=err, n=n):
            from spt.control.timeresp import initial
            data = responses(items, tf)
            e = None
            if err is not None:
                try:
                    te, _c = capped_tfinal([err])
                    y, t = initial(err, np.ones(n), te)
                    e = (np.asarray(t, float), np.asarray(y, float).reshape(len(t), -1))
                except ControlError as exc:
                    e = str(exc)
            return items, data, e
        self.tab.jobs.submit("mresp", job)
        self.tab._set_status(msgs)

    def resp_done(self, out) -> None:
        items, data, e = out
        pend = getattr(self, "_pending", None)
        if pend is None:
            return
        cards, tfinal, msgs, res = pend
        if res is not self.result or any(items.get(s.key) is not s.sys for _c, ser in cards for s in ser):
            return
        self._pending = None
        v = self.view
        msgs = list(msgs)
        for card, ser in cards:
            for err in card.set_series(ser, tfinal=tfinal, data=data):
                msgs.append(f"{card.title.text()} · {err}")
        if e is None:
            v.err.show([], note="Kestirici seçilmedi (tam durum ölçümü varsayılıyor).")
        elif isinstance(e, str):
            v.err.show([], note=e)
        else:
            t, y = e
            curves = [(f"e{i + 1}" if i else "kestirim hatası e1", COLORS["obs"], t, y[:, i],
                       cp.CHANNEL_STYLES[i % len(cp.CHANNEL_STYLES)], res["Ts"] > 0) for i in range(y.shape[1])]
            v.err.show(curves, [(0.0, 0, "")])
        self.tab._set_status(msgs)

    @staticmethod
    def _summary(res: dict) -> str:
        lines = []
        lines.append(("K = [Kx Ki] (LQI)" if res["Ki"] is not None else "K (u = N̄·r − K·x)") + ":")
        lines.append(dc.fmt_matrix(res["K"]))
        if res.get("Nbar") is not None:
            lines += ["", "N̄:", dc.fmt_matrix(res["Nbar"])]
        lines += ["", "Kapalı çevrim kutupları (ζ, ωn):"]
        for p, z, w in dc.damp_rows(res["cl_poles"], res["Ts"]):
            lines.append(f"  {dc.fmt_complex(p, 6):>24}   ζ = {z:.4g}   ωn = {w:.5g}")
        if res.get("L") is not None:
            lines += ["", "Kestirici kazancı L (öngörü):", dc.fmt_matrix(res["L"])]
            if res.get("kf") is not None and res["kf"].Mx is not None:
                lines += ["", "Yenilik kazancı Mx:", dc.fmt_matrix(res["kf"].Mx)]
            lines += ["", "Kestirici kutupları eig(A − L·C):"]
            for p, z, w in dc.damp_rows(res["obs_poles"], res["Ts"]):
                lines.append(f"  {dc.fmt_complex(p, 6):>24}   ζ = {z:.4g}   ωn = {w:.5g}")
        return "\n".join(lines)

    # ---------------------------------------------------------------- çıktılar
    def add_to_workspace(self, what: str) -> list[str]:
        ws = self.tab.workspace
        res = self.result
        if ws is None or res is None:
            self.tab.show_message("Önce tasarlayın.")
            return []
        lti = dc.lti_module()
        items = []
        if what == "T":
            items.append(("T_ss", res["T"], "durum geri beslemeli kapalı çevrim r → y"))
            if res.get("T_lqg") is not None:
                items.append(("T_lqg", res["T_lqg"], "LQG kapalı çevrim r → y"))
        elif what == "K":
            if res.get("lqg") is not None:
                items.append(("K_lqg", res["lqg"], "LQG izleyici (girişler [r; y])"))
            elif res["L"] is not None:
                from spt.control.modern import reg
                items.append(("K_reg", lti.ss(reg(res["P"], res["Kx"], res["L"])), "gözlemcili düzenleyici y → u"))
            else:
                items.append(("K_sf", res["Kc"], "durum geri beslemesi (girişler [r; y; x])"))
        else:
            if res.get("kf") is not None:
                items.append(("kest", lti.ss(res["kf"].kest), "Kalman kestiricisi ([u; y] → [ŷ; x̂])"))
            elif res["L"] is not None:
                from spt.control.modern import estim
                items.append(("est", lti.ss(estim(res["P"], res["L"], known=list(range(res["m"])))),
                              "gözlemci ([u; y] → [ŷ; x̂])"))
            else:
                self.tab.show_message("Kestirici seçilmedi.")
                return []
        added = []
        try:
            for base, sys, note in items:
                added.append(ws.add(ws.unique_name(base), sys, SOURCE_DESIGN, note))
        except ControlError as exc:
            self.tab.show_message(str(exc))
        if added:
            self.tab.show_message("Çalışma alanına eklendi: " + ", ".join(added))
        return added

    def generate_code(self, what: str = "ctrl"):
        if self.result is None:
            self.tab.show_message("Önce tasarlayın.")
            return None
        from spt.gui.dialogs.control_code_dialog import open_control_code_dialog
        try:
            spec = kalman_spec(self.result) if what == "kalman" else controller_spec(self.result)
            return open_control_code_dialog(self.tab, spec, "kalman" if what == "kalman" else "durum_gb")
        except ControlError as exc:
            self.tab.show_message(f"C kodu: {exc}")
            return None

    # ---------------------------------------------------------------- ayarlar
    def save_state(self, s) -> None:
        import json
        o = self.options()
        o["plant"] = self.plant.currentData()
        o["ctrl_i"], o["est_i"] = self.ctrl.currentIndex(), self.est.currentIndex()
        s.setValue("control/design_modern", json.dumps(o))

    def restore_state(self, s) -> None:
        import json
        try:
            o = json.loads(str(s.value("control/design_modern", "{}")) or "{}")
        except json.JSONDecodeError:
            return
        if not o:
            return
        if o.get("plant"):
            self.set_plant(o["plant"])
        self.ctrl.setCurrentIndex(int(o.get("ctrl_i", 0)))
        self.est.setCurrentIndex(int(o.get("est_i", 0)))
        for w, k in ((self.poles, "poles"), (self.Q, "Q"), (self.R, "R"), (self.N, "N"), (self.obs_poles, "obs_poles"),
                     (self.Qn, "Qn"), (self.Rn, "Rn"), (self.QI, "QI")):
            if isinstance(o.get(k), str):
                w.setText(o[k])
        self._Ts = self.tab.design_Ts()
