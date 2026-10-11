"""Kontrolcü (PID, LTI, gözlemcili durum geri beslemesi, Kalman) için MCU'ya özel C kodu üretme penceresi.

Solda MCU özellikleri (çekirdek / FPU, saat, derleyici, optimizasyon, sayı türü, çağrı hızı, flash / RAM — varsayılanlar
profilden) ve kontrolcünün uygulama seçenekleri (PID: çıkış doyumu, anti-windup, Kb / Kt; SISO LTI: gerçekleme)
sorulur. Kod `spt.codegen.control_c` ile üretilir; bellek (derleyiciye göre ölçülür) ve işlem maliyeti arka plan iş
parçacığında hesaplanır, arayüz donmaz (her ölçüm bir nesil: eskiyen sonuç yok sayılır).

Arayüzdeki sekmeler `open_control_code_dialog(parent, controller, name)` ile açar; `controller` bir pid / pidstd /
pid2, ss / tf / zpk nesnesi (ör. reg / lqg sonucu), kalman / kalmd demeti, kazanç dizisi ya da `control_c` tanımı
(PidSpec, LtiSpec, ObserverSpec — `ObserverSpec.state_feedback` gözlemcisiz —, KalmanSpec) olabilir.
"""
from __future__ import annotations

import html
import logging
import math
import re
import threading
from dataclasses import replace

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from spt.codegen import control_c as cc
from spt.codegen import mcu
from spt.control.errors import ControlError
from spt.gui.dialogs.code_dialog import CodeDialog
from spt.gui.theme import TERRAFLUX_DARK, Tokens
from spt.gui.widgets.common import CollapsibleSection, combo, label

log = logging.getLogger("spt.gui")

_AW_KEYS = ("none", "back-calculation", "clamping")
_AW_LABELS = ("Yok", "Geri hesaplama (Kb)", "Sıkıştırma (clamping)")
_REAL_KEYS = ("auto", "sos", "ss")
_REAL_LABELS = ("Otomatik (SISO: bölümler, MIMO: durum uzayı)", "İkinci derece bölümler (SISO)", "Durum uzayı")
_TR = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")


def _tokens(parent) -> Tokens:
    """Ebeveyn zincirindeki tema belirteçleri (bulunamazsa varsayılan koyu tema)."""
    w = parent
    while w is not None:
        theme = getattr(w, "_theme", None) or getattr(w, "theme", None)
        tokens = getattr(theme, "tokens", None)
        if isinstance(tokens, Tokens):
            return tokens
        w = w.parent() if hasattr(w, "parent") and callable(w.parent) else None
    return TERRAFLUX_DARK


def prefix_from_name(name: str) -> str:
    """Kontrolcü adından geçerli bir C öneki (ör. "Yükseklik PID" → "yukseklik")."""
    text = re.sub(r"[^a-z0-9]+", "_", (name or "").translate(_TR).lower()).strip("_")
    text = re.sub(r"_+", "_", text)
    text = re.sub(r"(?:_(?:pid|lti|obs|kf|kontrolcu|kontrol|denetleyici))+$", "", text)  # tür dosya adında zaten var
    text = re.sub(r"^[^a-z]+", "", text)
    if len(text) > cc.PREFIX_MAX:   # sözcük sınırında kısalt (ilk sözcük tek başına uzunsa onu kes)
        kept = ""
        for word in text.split("_"):
            cand = f"{kept}_{word}" if kept else word
            if len(cand) > cc.PREFIX_MAX:
                break
            kept = cand
        text = kept or text[:cc.PREFIX_MAX]
    text = text.rstrip("_")
    try:
        cc.check_prefix(text)
        return text
    except ControlError:
        return "ctl"


class _Bridge(QObject):
    """Arka plan iş parçacığından arayüze sonuç (sinyal ana iş parçacığında kuyruklanır)."""
    done = pyqtSignal(int, object, object, object, object, str)  # nesil, tanım, seçenekler, footprint, maliyet, hata


def _measure(bridge: _Bridge, generation: int, spec, opts, compilers) -> None:
    """Boyut / maliyet ölçümü (daemon iş parçacığı: hiçbir yerde beklenmez)."""
    try:
        fp = cc.footprint(spec, opts, compilers)
        cost = cc.cost(spec, opts)
        bridge.done.emit(generation, spec, opts, fp, cost, "")
    except Exception as exc:  # noqa: BLE001 — ölçüm başarısız olsa da kod üretilmiş olmalı
        log.exception("Kontrolcü C kodu ölçümü başarısız")
        bridge.done.emit(generation, spec, opts, None, None, f"{type(exc).__name__}: {exc}")


def _spin(lo: float, hi: float, dec: int, value: float, suffix: str = "") -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setValue(value)
    if suffix:
        s.setSuffix(suffix)
    s.setKeyboardTracking(False)
    return s


class ControlCodeDialog(CodeDialog):
    def __init__(self, parent=None):
        side = QWidget()
        lay = QVBoxLayout(side)
        lay.setContentsMargins(0, 0, 8, 0)
        lay.setSpacing(6)
        intro = label("Kontrolcünün uygulama seçeneklerini ve MCU özelliklerinizi seçin; kod, bellek ve maliyet "
                      "buna göre üretilir / ölçülür.", "Muted")
        intro.setWordWrap(True)
        lay.addWidget(intro)

        self.ctrl_box = CollapsibleSection("Kontrolcü")
        self.desc = label("", "Muted")
        self.desc.setWordWrap(True)
        self.ctrl_box.form.addRow(self.desc)
        self.limit = QCheckBox("Çıkış doyumu (sınırlar)")
        self.limit.setToolTip("Eyleyici sınırları: Simulink PID bloğundaki 'Limit output'")
        self.u_min = _spin(-1e9, 1e9, 6, -1.0)
        self.u_max = _spin(-1e9, 1e9, 6, 1.0)
        self.aw = combo(list(_AW_LABELS))
        self.aw.setToolTip("Doyumda integratör birikmesini önleme (Simulink tanımı): geri hesaplama integratör "
                           "girişine Kb·(doyumlu − doyumsuz toplam) ekler; sıkıştırma integrasyonu durdurur")
        self.kb = _spin(0.0, 1e6, 6, 1.0)
        self.kt = _spin(0.0, 1e6, 6, 1.0)
        self.kt.setToolTip("İzleme kazancı: …_pid_step_track() integratörü uygulanan çıkışa yaklaştırır (elle → "
                           "otomatik sarsıntısız geçiş)")
        self.realization = combo(list(_REAL_LABELS))
        self.ctrl_box.form.addRow(self.limit)
        for text, w in (("Alt sınır", self.u_min), ("Üst sınır", self.u_max), ("Anti-windup", self.aw),
                        ("Kb", self.kb), ("Kt (izleme)", self.kt), ("Gerçekleme", self.realization)):
            self.ctrl_box.add_row(text, w)
        self._pid_rows = (self.limit, self.u_min, self.u_max, self.aw, self.kb, self.kt)
        lay.addWidget(self.ctrl_box)

        target = CollapsibleSection("Hedef MCU")
        self.mcu = combo([])
        for key, prof in mcu.MCU_PROFILES.items():
            self.mcu.addItem(prof.name, key)
            self.mcu.setItemData(self.mcu.count() - 1, f"Örnek: {prof.examples}", 3)  # ToolTipRole
        self.mcu.setCurrentIndex(max(self.mcu.findData("cortex-m4f"), 0))
        self.mcu.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.mcu.setMinimumContentsLength(18)
        self.examples = label("", "Muted")
        self.examples.setWordWrap(True)
        self.clock = _spin(1.0, 2000.0, 1, 168.0, " MHz")
        self.compiler = combo([])
        for key, text in mcu.COMPILERS.items():
            self.compiler.addItem(text, key)
        self.opt = combo(list(mcu.OPT_LEVELS), 1)
        self.real = combo(["float", "double"])
        self.real.setToolTip("float: tek hassasiyet FPU'lu çekirdeklerde hızlı; double: FPU'suz / SP-FPU çekirdekte "
                             "yazılım kayan nokta (yavaş)")
        self.fs = _spin(0.001, 1e6, 3, 100.0, " Hz")
        self.fs.setToolTip("Kontrolcünün çağrılma hızı. Ayrık kontrolcü 1/Ts için tasarlanmıştır; farklı hız "
                           "dinamiği değiştirir (belgede uyarılır)")
        self.flash = _spin(1.0, 65536.0, 0, 1024.0, " KB")
        self.ram = _spin(1.0, 65536.0, 0, 192.0, " KB")
        self.prefix = QLineEdit("ctl")
        self.prefix.setToolTip("C tanımlayıcılarının ve dosyaların öneki (ör. irtifa → irtifa_pid_step, "
                               "KULLANIM_irtifa.md). Harfle başlar; harf, rakam, tek alt çizgi")
        for name, w in (("Çekirdek", self.mcu), ("", self.examples), ("Saat", self.clock), ("Derleyici", self.compiler),
                        ("Optimizasyon", self.opt), ("Sayı tipi", self.real), ("Çağrı hızı", self.fs),
                        ("Flash", self.flash), ("RAM", self.ram), ("Önek", self.prefix)):
            if name:
                target.add_row(name, w)
            else:
                target.form.addRow(w)
        lay.addWidget(target)
        self.generate_button = QPushButton("Kodu Üret ve Ölç")
        self.generate_button.setObjectName("Primary")
        self.generate_button.clicked.connect(self.refresh)
        lay.addWidget(self.generate_button)
        lay.addStretch(1)
        super().__init__(_tokens(parent), "Kontrolcü — C Kodu", "control_code", parent, side=side)
        self.mcu.currentIndexChanged.connect(self._on_mcu)
        self.limit.toggled.connect(self._sync_enabled)
        self.aw.currentIndexChanged.connect(self._sync_enabled)
        self._bridge = _Bridge(self)
        self._bridge.done.connect(self._measured)
        self._generation = 0
        self._base = None          # nesneden çevrilmiş tanım (uygulama seçenekleri eklenmeden)
        self._name = ""
        self._on_mcu()

    # ---------------------------------------------------------------- kontrolcü
    def set_controller(self, controller, name: str = "") -> bool:
        """Kontrolcüyü yükler; çevrilemezse hata durumda gösterilir ve False döner."""
        self._name = name or ""
        self.setWindowTitle(f"Kontrolcü — C Kodu · {name}" if name else "Kontrolcü — C Kodu")
        try:
            self._base = cc.controller_spec(controller)
        except ControlError as exc:
            self._base = None
            self.desc.setText(html.escape(str(exc)))
            self.set_files([])
            self.set_summary(f"<p>Bu kontrolcüden C kodu üretilemez: {html.escape(str(exc))}</p>")
            self.set_status(str(exc), "critical")
            self.generate_button.setEnabled(False)
            return False
        spec = self._base
        self.generate_button.setEnabled(True)
        self.desc.setText(html.escape(cc.describe(spec)))
        is_pid = isinstance(spec, cc.PidSpec)
        for w in self._pid_rows:
            self._set_row_visible(w, is_pid)
        is_siso_lti = isinstance(spec, cc.LtiSpec) and spec.shape == (1, 1)
        self._set_row_visible(self.realization, is_siso_lti)
        if is_pid:
            self.limit.setChecked(spec.has_limits)
            if spec.has_limits:
                self.u_min.setValue(spec.u_min if math.isfinite(spec.u_min) else -1e9)
                self.u_max.setValue(spec.u_max if math.isfinite(spec.u_max) else 1e9)
            self.aw.setCurrentIndex(_AW_KEYS.index(spec.anti_windup))
            self.kb.setValue(spec.Kb)
            self.kt.setValue(spec.Kt)
        if is_siso_lti:
            self.realization.setCurrentIndex(_REAL_KEYS.index(spec.realization))
        self.fs.setValue(1.0 / spec.Ts)
        self.prefix.setText(prefix_from_name(name) if name else self.prefix.text() or "ctl")
        self._sync_enabled()
        return True

    def _set_row_visible(self, widget: QWidget, visible: bool) -> None:
        form = self.ctrl_box.form
        widget.setVisible(visible)
        lab = form.labelForField(widget)
        if lab is not None:
            lab.setVisible(visible)

    def _sync_enabled(self, *_) -> None:
        on = self.limit.isChecked()
        for w in (self.u_min, self.u_max, self.aw):
            w.setEnabled(on)
        self.kb.setEnabled(on and self.aw.currentIndex() == 1)

    def _on_mcu(self, *_) -> None:
        prof = mcu.MCU_PROFILES[self.mcu.currentData()]
        self.clock.setValue(prof.default_clock_mhz)
        self.flash.setValue(prof.flash_kb)
        self.ram.setValue(prof.ram_kb)
        fpu = {"none": "FPU yok — kayan nokta yazılımla (yavaş)", "sp": "FPU tek hassasiyet: float önerilir",
               "dp": "FPU çift hassasiyet: double da donanımda"}[prof.fpu]
        self.examples.setText(f"{fpu}<br>Örnek: {html.escape(prof.examples)}")
        if prof.fpu in ("none", "sp"):
            self.real.setCurrentIndex(0)

    def current_spec(self):
        """Uygulama seçenekleri eklenmiş tanım (ControlError fırlatabilir)."""
        spec = self._base
        if spec is None:
            raise ControlError("Kontrolcü yüklenmedi")
        if isinstance(spec, cc.PidSpec):
            lim = self.limit.isChecked()
            spec = replace(spec, u_min=self.u_min.value() if lim else -math.inf,
                           u_max=self.u_max.value() if lim else math.inf,
                           anti_windup=_AW_KEYS[self.aw.currentIndex()] if lim else "none",
                           Kb=self.kb.value(), Kt=self.kt.value())
        elif isinstance(spec, cc.LtiSpec) and spec.shape == (1, 1):
            spec = replace(spec, realization=_REAL_KEYS[self.realization.currentIndex()])
        spec.validate()
        return spec

    def options(self) -> cc.ControlCodegenOptions:
        fs = self.fs.value()
        if self._base is not None and abs(fs - 1.0 / self._base.Ts) <= 0.5 * 10.0 ** -self.fs.decimals():
            fs = None              # kutudaki değer 1/Ts'nin yuvarlanmışı: tasarım hızı (yanlış hız uyarısı yok)
        return cc.ControlCodegenOptions(real=self.real.currentText(), mcu=self.mcu.currentData(),
                                        compiler=self.compiler.currentData(), opt=self.opt.currentText(),
                                        clock_mhz=self.clock.value(), fs_hz=fs,
                                        flash_kb=self.flash.value(), ram_kb=self.ram.value(),
                                        prefix=self.prefix.text().strip() or "ctl", name=self._name)

    # ---------------------------------------------------------------- üretim
    def shutdown(self) -> None:
        """Ölçüm daemon iş parçacığında: beklemeye gerek yok; sonucu gelirse yok sayılsın."""
        self._generation += 1

    def closeEvent(self, event) -> None:  # noqa: N802 — Qt adı
        self.shutdown()
        super().closeEvent(event)

    def refresh(self, measure: bool = True) -> None:
        try:
            spec = self.current_spec()
            opts = self.options()
            files = cc.generate(spec, opts)
        except ControlError as exc:
            self.set_files([])
            self.set_summary(f"<p>Kod üretilemedi: {html.escape(str(exc))}</p>")
            self.set_status(html.escape(str(exc)), "critical")
            return
        self.set_files(files)
        self._generation += 1
        if not measure:
            self.set_summary(self._summary_html(spec, opts, None, None))
            self.set_status(f"{len(files)} dosya üretildi.")
            return
        self.set_summary("<p>Kod üretildi. Bellek ve maliyet <b>ölçülüyor</b> (derleme birkaç saniye sürebilir)…</p>")
        self.set_status(f"{len(files)} dosya üretildi · ölçüm sürüyor…")
        self.generate_button.setEnabled(False)
        threading.Thread(target=_measure, args=(self._bridge, self._generation, spec, opts, tuple(mcu.COMPILERS)),
                         name="spt-control-codegen-measure", daemon=True).start()

    def _measured(self, generation: int, spec, opts, fp, cost, error: str) -> None:
        if generation != self._generation:
            return
        self.generate_button.setEnabled(True)
        if error:
            self.set_summary(self._summary_html(spec, opts, None, None)
                             + f"<p>Ölçüm yapılamadı: {html.escape(error)}</p>")
            self.set_status("Ölçüm başarısız (kod yine de kaydedilebilir).", "warning")
            return
        try:
            self.set_files(cc.generate(spec, opts, footprint_rows=fp, cost_rows=cost))
        except ControlError:
            pass
        self.set_summary(self._summary_html(spec, opts, fp, cost))
        worst = max((c.cpu_load_percent for c in cost), default=0.0)
        state = "critical" if worst > 100 else "warning" if worst > 50 else "good"
        self.set_status(f"Ölçüm tamamlandı · en yüksek CPU yükü %{worst:.3g} @ {cc._rate(spec, opts):g} Hz", state)

    def _summary_html(self, spec, opts, fp, cost) -> str:
        t = self._tokens
        prof = mcu.MCU_PROFILES[opts.mcu]
        fs = cc._rate(spec, opts)
        head = (f"<h3>{html.escape(opts.name or cc.kind_title(spec))}</h3>"
                f"<p>{html.escape(cc.describe(spec))}</p>"
                f"<p>{html.escape(prof.name)} · {opts.clock_mhz:g} MHz · {html.escape(opts.real)} · "
                f"{html.escape(opts.opt)} · çağrı hızı {fs:g} Hz · Flash {opts.flash_kb:g} KB · RAM {opts.ram_kb:g} KB"
                f"</p>")
        if abs(fs * spec.Ts - 1.0) > 1e-9:
            head += (f"<p><span style='color:{t.warning}'>⚠ Kontrolcü Ts = {spec.Ts:g} s ({1 / spec.Ts:g} Hz) için "
                     f"ayrıklaştırıldı; {fs:g} Hz'de çağırmak dinamiği değiştirir.</span></p>")
        if opts.real == "double" and prof.double_is_soft():
            head += (f"<p><span style='color:{t.warning}'>⚠ Bu çekirdekte double yazılım kayan noktadır "
                     "(10–100× yavaş).</span></p>")
        if fp is None or cost is None:
            return head
        methods: list[str] = []

        def note(text: str) -> str:
            if text not in methods:
                methods.append(text)
            return f"<sup>{methods.index(text) + 1}</sup>"

        flash_b, ram_b = opts.flash_kb * 1024, opts.ram_kb * 1024
        rows = []
        for r in fp:
            flash = r.flash_with_libm if r.flash_with_libm is not None else r.flash_code
            stack = "—" if r.stack is None else f"{r.stack_own if r.stack_own is not None else r.stack} / {r.stack}"
            measured = (f'<span style="color:{t.good}">ölçüldü</span>' if r.measured
                        else f'<span style="color:{t.warning}">tahmin</span>') + note(r.method)
            rows.append(f"<tr><td>{html.escape(mcu.COMPILERS.get(r.compiler, r.compiler))}</td>"
                        f"<td align='right'>{r.flash_code}</td><td align='right'>{r.flash_with_libm or '—'}</td>"
                        f"<td align='right'>{flash / flash_b * 100:.2g}%</td><td align='right'>{r.ram_state}</td>"
                        f"<td align='right'>{r.ram_state / ram_b * 100:.2g}%</td><td align='right'>{stack}</td>"
                        f"<td>{measured}</td></tr>")
        crow = []
        for c in cost:
            load = c.cpu_load_percent
            color = t.critical if load > 100 else t.warning if load > 50 else t.good
            crow.append(f"<tr><td>{html.escape(c.algorithm)}</td><td align='right'>{c.cycles:,.0f}</td>"
                        f"<td align='right'>{c.cycles_max:,.0f}</td><td align='right'>{c.us_per_update:.3g}</td>"
                        f"<td align='right'><span style='color:{color}'>%{load:.3g}</span></td>"
                        f"<td align='right'>{c.max_rate_hz:,.0f}</td><td>{note(c.method)}</td></tr>")
        fp_table = ("<h4>Bellek (derleyiciye göre)</h4><table border='1' cellspacing='0' cellpadding='4'>"
                    "<tr><th>Derleyici</th><th>Kod (B)</th><th>Kütüphane dahil (B)</th><th>Flash %</th>"
                    "<th>RAM durum (B)</th><th>RAM %</th><th>Yığın bu kod / kütüphane dahil (B)</th><th>Yöntem</th>"
                    "</tr>" + "".join(rows) + "</table>")
        cost_table = ("<h4>İşlem maliyeti (adım başına)</h4><table border='1' cellspacing='0' cellpadding='4'>"
                      "<tr><th>Kip</th><th>Çevrim (ort.)</th><th>Çevrim (en kötü)</th><th>µs</th><th>CPU yükü</th>"
                      "<th>En yüksek hız (Hz)</th><th>Not</th></tr>" + "".join(crow) + "</table>")
        notes = ("<p><b>Notlar:</b> Kod boyutu makinedeki araç zinciriyle gerçekten derlenerek ölçülür (GCC: "
                 "arm-none-eabi-gcc, Clang: Arm Compiler 6 / Keil MDK ile aynı altyapı); IAR ve yerel derleyicisi "
                 "olmayan çekirdekler <i>tahmin</i>dir. RAM = durum yapısının boyutu (malloc yok; katsayılar "
                 "flash'ta). Çevrim sayıları, ölçülen işlem sayıları × çekirdeğin komut çevrim tablosu — gerçek "
                 "donanımda DWT->CYCCNT ile doğrulayın. CPU yükü %100'ü aşarsa bu hızda gerçek zamanlı çalışamaz.</p>")
        details = "<ol>" + "".join(f"<li>{html.escape(m)}</li>" for m in methods) + "</ol>"
        return head + fp_table + cost_table + "<h4>Yöntem ayrıntıları</h4>" + details + notes


def open_control_code_dialog(parent, controller, name: str = "") -> ControlCodeDialog:
    """Kontrolcü C kodu penceresini açar (ebeveyn başına tek pencere; yeniden çağrıda kontrolcü değişir)."""
    dlg = getattr(parent, "_control_code_dialog", None) if parent is not None else None
    if dlg is None:
        dlg = ControlCodeDialog(parent)
        if parent is not None:
            try:
                parent._control_code_dialog = dlg
            except AttributeError:
                pass
    if dlg.set_controller(controller, name):
        dlg.refresh()
    dlg.show()
    dlg.raise_()
    return dlg
