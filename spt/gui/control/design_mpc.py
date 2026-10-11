"""Kontrolcü Tasarımı · MPC paneli (MATLAB MPC Designer'ın özü): ufuklar, ağırlıklar, MV / MV hızı / OV kısıtları,
referansla kapalı çevrim benzetimi ve C kodu.

MPC nesnesi `spt.control.mpc.mpc(plant, Ts, p, m, W, MV, OV)` ile kurulur (MATLAB anlamları: tüm girişler MV, tüm
çıkışlar ölçülen OV; varsayılan ağırlıklar ve bozucu / gürültü modelleri MATLAB'inki). Benzetim `sim(mpcobj, T, r)`
(bitki = denetleyicinin iç modeli, kısıtlar açık) arka planda çalışır; sonuçlar y / r, u ve Δu kartlarında, kısıt
sınırları gri kesikli çizgilerle gösterilir. Boş bırakılan ağırlık / kısıt MATLAB varsayılanıdır (kısıt yok).
"""
from __future__ import annotations

import json
import logging

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from spt.control.errors import ControlError
from spt.gui.control import control_plots as cp
from spt.gui.control import design_core as dc
from spt.gui.control.design_widgets import CurvesCard, mono_label, scroll_area, spin_box
from spt.gui.control.workspace import SOURCE_DESIGN
from spt.gui.theme import ThemeManager
from spt.gui.widgets.common import Card, CollapsibleSection, combo, label

log = logging.getLogger("spt.gui")
COLORS = {"y": cp.PALETTE[0], "u": cp.PALETTE[3], "du": cp.PALETTE[2]}


# ============================================================================ Qt'siz kurulum
def _per_channel(text: str, n: int, what: str) -> np.ndarray | None:
    v = dc.parse_vector(text, what)
    if v is None:
        return None
    if v.size == 1:
        return np.full(n, float(v[0]))
    if v.size != n:
        raise ControlError(f"{what}: {n} kanal için {n} değer ya da tek değer verin ({v.size} verildi).")
    return v


def build_mpc(sys, Ts: float, p: int, m: int, o: dict):
    """MPC nesnesi. o: metin alanları (wmv, wrate, wov, umin, umax, dumin, dumax, ymin, ymax)."""
    from spt.control.mpc import mpc
    lti = dc.lti_module()
    if sys is None:
        raise ControlError("Bitki seçin.")
    if sys.has_delay():
        if sys.is_discrete:
            from spt.control.discretize import absorb_delay
            sys = absorb_delay(sys)
        else:
            raise ControlError("MPC: sürekli zamanda gecikmeli bitki desteklenmez; Padé uygulayın ya da bitkiyi "
                               "denetleyici Ts'iyle ayrıklaştırın.")
    P = lti.ss(sys)
    nu, ny = P.ninputs, P.noutputs
    W = {}
    for key, field, n in (("wmv", "ManipulatedVariables", nu), ("wrate", "ManipulatedVariablesRate", nu),
                          ("wov", "OutputVariables", ny)):
        v = _per_channel(o.get(key, ""), n, "Ağırlık")
        if v is not None:
            W[field] = v
    MV = [{} for _ in range(nu)]
    for key, field in (("umin", "Min"), ("umax", "Max"), ("dumin", "RateMin"), ("dumax", "RateMax")):
        v = _per_channel(o.get(key, ""), nu, f"MV {field}")
        if v is not None:
            for i in range(nu):
                MV[i][field] = float(v[i])
    OV = [{} for _ in range(ny)]
    for key, field in (("ymin", "Min"), ("ymax", "Max")):
        v = _per_channel(o.get(key, ""), ny, f"OV {field}")
        if v is not None:
            for i in range(ny):
                OV[i][field] = float(v[i])
    return mpc(P, Ts, p, m, W or None, MV, OV)


def simulate(obj, T: int, r: np.ndarray, preview: bool) -> dict:
    from spt.control.mpc import cloffset, mpcsimopt
    res = obj.sim(T, r, options=mpcsimopt(RefLookAhead="on" if preview else "off"))
    out = {"y": np.asarray(res.y, float), "t": np.asarray(res.t, float), "u": np.asarray(res.u, float),
           "status": np.asarray(res.status, float), "r": np.asarray(r, float)}
    try:
        out["offset"] = np.asarray(cloffset(obj), float)
    except ControlError as exc:
        out["offset"] = None
        out["offset_err"] = str(exc)
    return out


# ============================================================================ görünüm
class MpcView(QWidget):
    def __init__(self, theme: ThemeManager):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.legend = cp.SeriesLegend()
        lay.addWidget(self.legend)
        split = QSplitter(Qt.Orientation.Vertical)
        split.setChildrenCollapsible(False)
        self.y = CurvesCard(theme, "Çıkışlar y · referans r (gri)")
        top = QSplitter(Qt.Orientation.Horizontal)
        top.setChildrenCollapsible(False)
        self.u = CurvesCard(theme, "Ayarlanan Değişkenler u (MV)")
        self.du = CurvesCard(theme, "MV Hızı Δu")
        self.info = Card("Benzetim Özeti")
        self.text = mono_label()
        self.text.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.info.body.addWidget(scroll_area(self.text, horizontal=True), 1)
        top.addWidget(self.y)
        top.addWidget(self.info)
        top.setSizes([700, 300])
        bottom = QSplitter(Qt.Orientation.Horizontal)
        bottom.setChildrenCollapsible(False)
        bottom.addWidget(self.u)
        bottom.addWidget(self.du)
        split.addWidget(top)
        split.addWidget(bottom)
        lay.addWidget(split, 1)

    def detachable_cards(self) -> list[tuple[str, str, Card]]:
        return [("ctl_ds_mpc_y", "MPC · Çıkışlar", self.y), ("ctl_ds_mpc_u", "MPC · MV", self.u),
                ("ctl_ds_mpc_du", "MPC · MV Hızı", self.du), ("ctl_ds_mpc_info", "MPC · Özet", self.info)]


def _line(text: str, tip: str) -> QLineEdit:
    w = QLineEdit(text)
    w.setToolTip(tip)
    return w


class MpcPanel(QWidget):
    def __init__(self, tab, view: MpcView):
        super().__init__()
        self.tab = tab
        self.view = view
        self.obj = None
        self.result: dict | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        sec = CollapsibleSection("Bitki ve ufuklar")
        self.plant = combo([])
        self.plant.setToolTip("Tüm girişler MV, tüm çıkışlar ölçülen OV (MATLAB varsayılanı). Sürekli bitki Ts ile "
                              "ZOH ayrıklaştırılır.")
        self.plant.currentIndexChanged.connect(self._changed)
        sec.add_row("Bitki", self.plant)
        self.info = label("", "Muted")
        self.info.setWordWrap(True)
        sec.form.addRow(self.info)
        self.ts = spin_box(1e-6, 1e4, 0.1, 6, " s", "Denetleyici örnekleme zamanı")
        self.ts.setStepType(QDoubleSpinBox.StepType.DefaultStepType)
        self.ts.setSingleStep(0.01)
        self.ts.valueChanged.connect(self._changed)
        sec.add_row("Ts", self.ts)
        self.p = QSpinBox()
        self.p.setRange(1, 500)
        self.p.setValue(10)
        self.p.setToolTip("Tahmin ufku p (PredictionHorizon)")
        self.p.valueChanged.connect(self._changed)
        sec.add_row("Tahmin ufku p", self.p)
        self.m = QSpinBox()
        self.m.setRange(1, 500)
        self.m.setValue(2)
        self.m.setToolTip("Kontrol ufku m (ControlHorizon, ≤ p)")
        self.m.valueChanged.connect(self._changed)
        sec.add_row("Kontrol ufku m", self.m)
        lay.addWidget(sec)

        w = CollapsibleSection("Ağırlıklar (boş: MATLAB varsayılanı)")
        self.wmv = _line("", "MV ağırlığı (kanal başına ya da tek değer; varsayılan 0)")
        w.add_row("MV", self.wmv)
        self.wrate = _line("", "MV hızı ağırlığı (varsayılan 0.1)")
        w.add_row("MV hızı", self.wrate)
        self.wov = _line("", "Çıkış ağırlığı (varsayılan 1)")
        w.add_row("OV", self.wov)
        lay.addWidget(w)

        c = CollapsibleSection("Kısıtlar (boş: yok)")
        self.umin = _line("", "MV alt sınırı (sert)")
        c.add_row("MV en küçük", self.umin)
        self.umax = _line("", "MV üst sınırı (sert)")
        c.add_row("MV en büyük", self.umax)
        self.dumin = _line("", "MV hızı alt sınırı (Δu, sert)")
        c.add_row("Δu en küçük", self.dumin)
        self.dumax = _line("", "MV hızı üst sınırı (Δu, sert)")
        c.add_row("Δu en büyük", self.dumax)
        self.ymin = _line("", "Çıkış alt sınırı (yumuşak, ECR = 1)")
        c.add_row("OV en küçük", self.ymin)
        self.ymax = _line("", "Çıkış üst sınırı (yumuşak)")
        c.add_row("OV en büyük", self.ymax)
        lay.addWidget(c)

        s = CollapsibleSection("Benzetim")
        self.steps = QSpinBox()
        self.steps.setRange(2, 100000)
        self.steps.setValue(60)
        self.steps.setToolTip("Benzetim adım sayısı T")
        self.steps.valueChanged.connect(self._changed)
        s.add_row("Adım sayısı", self.steps)
        self.ref = _line("1", "Referans (çıkış başına ya da tek değer), t = 0'dan itibaren sabit")
        s.add_row("Referans r", self.ref)
        self.preview = QCheckBox("Referans önizleme (RefLookAhead)")
        self.preview.toggled.connect(self._changed)
        s.form.addRow(self.preview)
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        run = QPushButton("Benzet")
        run.clicked.connect(self.run_now)
        rl.addWidget(run)
        add = QPushButton("Modellere ekle")
        add.clicked.connect(self.add_to_workspace)
        rl.addWidget(add)
        code = QPushButton("C Kodu Üret…")
        code.setToolTip("MPC için MCU C kodu (QP çözücüsüyle)")
        code.clicked.connect(self.generate_code)
        rl.addWidget(code)
        rl.addStretch(1)
        s.form.addRow(row)
        lay.addWidget(s)
        for e in (self.wmv, self.wrate, self.wov, self.umin, self.umax, self.dumin, self.dumax, self.ymin, self.ymax,
                  self.ref):
            e.editingFinished.connect(self._changed)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(300)
        self._timer.timeout.connect(self.run_now)

    # ---------------------------------------------------------------- bitki
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

    def set_plant(self, name: str) -> None:
        i = self.plant.findData(name)
        if i >= 0:
            self.plant.setCurrentIndex(i)

    def _sys(self):
        name = self.plant.currentData()
        if name is None or self.tab.workspace is None:
            return None
        return self.tab.workspace.lti(name)

    def options(self) -> dict:
        return {k: getattr(self, k).text() for k in ("wmv", "wrate", "wov", "umin", "umax", "dumin", "dumax",
                                                       "ymin", "ymax")}

    # ---------------------------------------------------------------- hesap
    def _changed(self, *_) -> None:
        if self.tab.mode == "mpc":
            self._timer.start()

    def pending(self) -> bool:
        return self._timer.isActive()

    def refresh(self) -> None:
        self._timer.start()

    def build(self):
        sys = self._sys()
        obj = build_mpc(sys, self.ts.value(), self.p.value(), min(self.m.value(), self.p.value()), self.options())
        d = obj.data   # derleme hatası burada çıksın
        self.info.setText(f"nx = {d.nxp}, MV = {d.nmv}, OV = {d.ny}; QP değişkeni {d.degrees}, kısıt "
                          f"{d.Ac.shape[0]}")
        return obj

    def run_now(self) -> None:
        self._timer.stop()
        try:
            obj = self.build()
            ny = obj.data.ny
            r = _per_channel(self.ref.text() or "1", ny, "Referans")
        except ControlError as exc:
            self.obj = None
            self.sim_failed(str(exc))
            return
        self.obj = obj
        T, preview = self.steps.value(), self.preview.isChecked()
        self.tab.jobs.submit("mpc", lambda o=obj, T=T, r=r, pv=preview: (o, simulate(o, T, r, pv)))

    def sim_failed(self, msg: str) -> None:
        self.result = None
        for card in (self.view.y, self.view.u, self.view.du):
            card.show([], note=msg)
        self.view.text.setText("")
        self.view.legend.set_series([])
        self.tab._set_status([f"MPC: {msg}"])

    def sim_done(self, res) -> None:
        obj, out = res
        if obj is not self.obj:
            return
        self.result = out
        v = self.view
        y, u, t = out["y"], out["u"], out["t"]
        d = obj.data
        styles = cp.CHANNEL_STYLES
        yname = [obj.OV[i].Name for i in range(y.shape[1])]
        uname = [obj.MV[i].Name for i in range(u.shape[1])]
        ycurves = [(yname[i], COLORS["y"], t, y[:, i], styles[i % len(styles)], True) for i in range(y.shape[1])]
        guides = [(float(out["r"][i]), 0, f"r {yname[i]}") for i in range(y.shape[1])]
        for i in range(y.shape[1]):
            for key, txt in (("Min", "min"), ("Max", "maks")):
                val = float(obj.OV[i][key])
                if np.isfinite(val):
                    guides.append((val, 0, f"{yname[i]} {txt}"))
        v.y.show(ycurves, guides)
        ucurves = [(uname[i], COLORS["u"], t, u[:, i], styles[i % len(styles)], True) for i in range(u.shape[1])]
        uguides = []
        du = np.diff(np.vstack([np.zeros((1, u.shape[1])), u]), axis=0)
        dcurves = [(f"Δ{uname[i]}", COLORS["du"], t, du[:, i], styles[i % len(styles)], True)
                   for i in range(u.shape[1])]
        dguides = []
        for i in range(u.shape[1]):
            for key, txt in (("Min", "min"), ("Max", "maks")):
                val = float(obj.MV[i][key])
                if np.isfinite(val):
                    uguides.append((val, 0, f"{uname[i]} {txt}"))
                val = float(obj.MV[i]["Rate" + key])
                if np.isfinite(val):
                    dguides.append((val, 0, f"Δ{uname[i]} {txt}"))
        v.u.show(ucurves, uguides)
        v.du.show(dcurves, dguides)
        v.legend.set_series([(COLORS["y"], "çıkış y"), (COLORS["u"], "MV u"), (COLORS["du"], "Δu"),
                             (cp.GUIDE, "referans / kısıt sınırları")])
        st = out["status"]
        lines = [f"Ts = {d.Ts:g} s, p = {d.p}, m = {obj.ControlHorizon}",
                 f"QP: {int(np.sum(st > 0))} adım optimal, {int(np.sum(st == 0))} yineleme sınırında, "
                 f"{int(np.sum(st < 0))} olursuz (MV tutuldu)",
                 f"son y = {', '.join(f'{x:.6g}' for x in y[-1])}",
                 f"en büyük |Δu| = {', '.join(f'{x:.6g}' for x in np.max(np.abs(du), axis=0))}"]
        if out.get("offset") is not None:
            lines.append("cloffset (kısıtsız kalıcı ofset):")
            lines.append(dc.fmt_matrix(out["offset"]))
        v.text.setText("\n".join(lines))
        self.tab._set_status([] if np.all(st >= 0) else ["MPC: bazı adımlarda QP olursuz (MV önceki değerde tutuldu)"])

    # ---------------------------------------------------------------- çıktılar
    def add_to_workspace(self) -> list[str]:
        ws = self.tab.workspace
        try:
            obj = self.obj or self.build()
        except ControlError as exc:
            self.tab.show_message(f"MPC: {exc}")
            return []
        if ws is None:
            return []
        name = ws.add(ws.unique_name("mpc"), obj, SOURCE_DESIGN,
                      f"MPC Ts = {obj.Ts:g} s, p = {obj.PredictionHorizon}, m = {obj.ControlHorizon}")
        self.tab.show_message(f"Çalışma alanına eklendi: {name}")
        return [name]

    def generate_code(self):
        try:
            obj = self.obj or self.build()
        except ControlError as exc:
            self.tab.show_message(f"MPC: {exc}")
            return None
        from spt.gui.dialogs.mpc_code_dialog import open_mpc_code_dialog
        return open_mpc_code_dialog(self.tab, obj, "mpc")

    # ---------------------------------------------------------------- ayarlar
    def save_state(self, s) -> None:
        o = self.options()
        o.update(plant=self.plant.currentData(), Ts=self.ts.value(), p=self.p.value(), m=self.m.value(),
                 steps=self.steps.value(), ref=self.ref.text(), preview=self.preview.isChecked())
        s.setValue("control/design_mpc", json.dumps(o))

    def restore_state(self, s) -> None:
        try:
            o = json.loads(str(s.value("control/design_mpc", "{}")) or "{}")
        except json.JSONDecodeError:
            return
        if not o:
            return
        widgets = [self.plant, self.ts, self.p, self.m, self.steps, self.preview]
        for w in widgets:
            w.blockSignals(True)
        try:
            if o.get("plant"):
                self.set_plant(o["plant"])
            self.ts.setValue(float(o.get("Ts", 0.1)))
            self.p.setValue(int(o.get("p", 10)))
            self.m.setValue(int(o.get("m", 2)))
            self.steps.setValue(int(o.get("steps", 60)))
            self.preview.setChecked(bool(o.get("preview", False)))
            for k in ("wmv", "wrate", "wov", "umin", "umax", "dumin", "dumax", "ymin", "ymax"):
                if isinstance(o.get(k), str):
                    getattr(self, k).setText(o[k])
            if isinstance(o.get("ref"), str):
                self.ref.setText(o["ref"])
        finally:
            for w in widgets:
                w.blockSignals(False)
