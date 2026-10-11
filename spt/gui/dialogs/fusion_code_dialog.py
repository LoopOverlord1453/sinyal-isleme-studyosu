"""Sensör füzyonu için MCU'ya özel C kodu üretme penceresi.

Pencerenin solunda MCU özellikleri sorulur: çekirdek (FPU), saat frekansı, derleyici, optimizasyon, sayı tipi
(float / double), örnekleme hızı, flash / RAM kapasitesi ve (yönelimde) manyetometre. Seçimlere göre:
  - .h / .c dosyaları ve KULLANIM.md (entegrasyon adımları) üretilir,
  - kod boyutu derleyiciye göre ÖLÇÜLÜR (makinede arm-none-eabi-gcc / clang varsa; IAR ve araç zinciri olmayan
    çekirdeklerde açıkça "tahmin" yazar), RAM = durum yapısının boyutu, yığın = en kötü çağrı yolu,
  - maliyet: adım başına işlem sayıları (C++ sayaçla ölçülür), çekirdek çevrim tablosundan çevrim / µs / CPU yükü.
Derleme saniyeler sürebildiği için ölçüm arka plan iş parçacığında yapılır; arayüz donmaz.
"""
from __future__ import annotations

import html
import importlib
import logging
import threading

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

from spt.codegen import fusion_c, mcu
from spt.gui.dialogs.code_dialog import CodeDialog
from spt.gui.widgets.common import CollapsibleSection, combo, label

log = logging.getLogger("spt.gui")


def _generic_module():
    """IMU dışı senaryoların C üreticisi (varsa)."""
    try:
        return importlib.import_module("spt.codegen.fusion_generic_c")
    except Exception:  # noqa: BLE001 — modül yok ya da yüklenemedi: senaryo için kod üretimi kapalı
        return None


class _Bridge(QObject):
    """Arka plan iş parçacığından arayüze sonuç (sinyal ana iş parçacığında kuyruklanır)."""
    done = pyqtSignal(int, object, object, object, str)  # nesil, seçenekler, footprint, maliyet, hata


def _measure(bridge: _Bridge, generation: int, module, opts, compilers) -> None:
    """Boyut / maliyet ölçümü (daemon iş parçacığı: hiçbir yerde beklenmez, kapanışı geciktirmez)."""
    try:
        fp = module.footprint(opts, compilers)
        cost = module.cost(opts)
        bridge.done.emit(generation, opts, fp, cost, "")
    except Exception as exc:  # noqa: BLE001 — ölçüm başarısız olsa da kod üretilmiş olmalı
        log.exception("C kodu ölçümü başarısız")
        bridge.done.emit(generation, opts, None, None, f"{type(exc).__name__}: {exc}")


class FusionCodeDialog(CodeDialog):
    def __init__(self, view):
        self._view = view
        side = QWidget()
        lay = QVBoxLayout(side)
        lay.setContentsMargins(0, 0, 8, 0)
        lay.setSpacing(6)
        intro = label("MCU özelliklerinizi seçin; kod, bellek ve maliyet buna göre üretilir / ölçülür.", "Muted")
        intro.setWordWrap(True)
        lay.addWidget(intro)
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
        self.clock = self._spin(1.0, 2000.0, 1, 168.0, " MHz")
        self.compiler = combo([])
        for key, text in mcu.COMPILERS.items():
            self.compiler.addItem(text, key)
        self.opt = combo(list(mcu.OPT_LEVELS), 1)
        self.real = combo(["float", "double"])
        self.real.setToolTip("float: tek hassasiyet FPU'lu çekirdeklerde hızlı; double: FPU'suz / SP-FPU çekirdekte "
                             "yazılım kayan nokta (yavaş)")
        self.fs = self._spin(1.0, 1e6, 1, 200.0, " Hz")
        self.fs.setToolTip("Algoritmanın çağrılma hızı (CPU yükü buna göre hesaplanır)")
        self.flash = self._spin(1.0, 65536.0, 0, 1024.0, " KB")
        self.ram = self._spin(1.0, 65536.0, 0, 192.0, " KB")
        self.use_mag = QCheckBox("Manyetometre (MARG) kullan")
        self.use_mag.setChecked(True)
        self.prefix = QLineEdit("spt")
        self.prefix.setToolTip("C tanımlayıcılarının öneki (ör. spt_ekf_update)")
        for name, w in (("Çekirdek", self.mcu), ("", self.examples), ("Saat", self.clock), ("Derleyici", self.compiler),
                        ("Optimizasyon", self.opt), ("Sayı tipi", self.real), ("Çağrı hızı", self.fs),
                        ("Flash", self.flash), ("RAM", self.ram), ("Önek", self.prefix)):
            if name:
                target.add_row(name, w)
            else:
                target.form.addRow(w)
        target.form.addRow(self.use_mag)
        lay.addWidget(target)
        self.algos_box = CollapsibleSection("Algoritmalar")
        self._algo_checks: dict[str, QCheckBox] = {}
        lay.addWidget(self.algos_box)
        self.generate_button = QPushButton("Kodu Üret ve Ölç")
        self.generate_button.setObjectName("Primary")
        self.generate_button.clicked.connect(self.refresh)
        lay.addWidget(self.generate_button)
        lay.addStretch(1)
        super().__init__(view._theme.tokens, "Sensör Füzyonu — C Kodu", "fusion_code", view, side=side)
        self.mcu.currentIndexChanged.connect(self._on_mcu)
        self._bridge = _Bridge(self)
        self._bridge.done.connect(self._measured)
        self._generation = 0  # her ölçüm bir nesil: eski ölçümün sonucu gelirse yok sayılır
        self._on_mcu()

    @staticmethod
    def _spin(lo, hi, dec, value, suffix) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(dec)
        s.setValue(value)
        s.setSuffix(suffix)
        s.setKeyboardTracking(False)
        return s

    def _on_mcu(self, *_) -> None:
        prof = mcu.MCU_PROFILES[self.mcu.currentData()]
        self.clock.setValue(prof.default_clock_mhz)
        self.flash.setValue(prof.flash_kb)  # örnek parçanın tipik belleği (Flash % / RAM % buna göre)
        self.ram.setValue(prof.ram_kb)
        fpu = {"none": "FPU yok — kayan nokta yazılımla (yavaş)", "sp": "FPU tek hassasiyet: float önerilir",
               "dp": "FPU çift hassasiyet: double da donanımda"}[prof.fpu]
        self.examples.setText(f"{fpu}<br>Örnek: {prof.examples}")
        if prof.fpu == "none" or prof.fpu == "sp":
            self.real.setCurrentIndex(0)

    # ---------------------------------------------------------------- görünümden ayarları al
    def load_from_view(self) -> None:
        v = self._view
        scen = v.scenario
        self.setWindowTitle(f"Sensör Füzyonu — C Kodu · {scen.name}")
        self._clear_algos()
        supported = self._supported(scen.key)
        for key, cls in scen.algorithms.items():
            box = QCheckBox(cls.name)
            box.setChecked(v._enabled[key].isChecked() and key in supported)
            box.setEnabled(key in supported)
            if key not in supported:
                box.setToolTip("Bu algoritmanın C karşılığı henüz yok")
            self._algo_checks[key] = box
            self.algos_box.form.addRow(box)
        self.use_mag.setVisible(scen.key == "orientation")
        handle = v._handle()
        if v.engine is not None:
            self.fs.setValue(v.engine.fs)
        elif handle is not None:
            self.fs.setValue(float(handle.fs))
        mapping = v.mapping()
        if scen.key == "orientation" and mapping is not None:
            self.use_mag.setChecked("mag" in mapping.inputs)

    def _clear_algos(self) -> None:
        form = self.algos_box.form
        while form.rowCount():
            form.removeRow(0)
        self._algo_checks.clear()

    def _supported(self, scenario: str) -> tuple[str, ...]:
        if scenario == "orientation":
            return tuple(fusion_c.ALGORITHM_KEYS) if hasattr(fusion_c, "ALGORITHM_KEYS") else (
                "mahony", "madgwick", "kalman", "ekf", "ukf", "pf")
        module = _generic_module()
        return tuple(getattr(module, "SUPPORTED", {}).get(scenario, ())) if module is not None else ()

    # ---------------------------------------------------------------- üretim
    def _options(self):
        v = self._view
        scen = v.scenario
        algos = tuple(k for k, b in self._algo_checks.items() if b.isChecked())
        params = {k: {p: v._param_value(w) for p, w in v._params.get(k, {}).items()} for k in algos}
        common = dict(real=self.real.currentText(), mcu=self.mcu.currentData(), compiler=self.compiler.currentData(),
                      opt=self.opt.currentText(), clock_mhz=self.clock.value(), fs_hz=self.fs.value(),
                      flash_kb=self.flash.value(), ram_kb=self.ram.value(), prefix=self.prefix.text() or "spt")
        if scen.key == "orientation":
            return fusion_c, fusion_c.CodegenOptions(algorithms=algos, params=params,
                                                     use_mag=self.use_mag.isChecked(), **common)
        module = _generic_module()
        mapping = v.mapping()
        if mapping is not None and mapping.inputs:
            inputs = tuple(mapping.inputs)
        else:  # kaynak eşlenmemiş: zorunlu girişler + "birinden biri" gruplarının ilki
            keys = [i.key for i in scen.inputs if i.required]
            for group in getattr(scen, "one_of", ()):
                if not set(group) & set(keys):
                    keys.append(group[0])
            inputs = tuple(keys)
        return module, module.GenericCodegenOptions(scenario=scen.key, algorithms=algos,
                                                    scenario_values=v.scenario_values(), params=params,
                                                    inputs=inputs, **common)

    def shutdown(self) -> None:
        """Ölçüm daemon iş parçacığında: beklemeye gerek yok; sonucu gelirse yok sayılsın."""
        self._generation += 1

    def refresh(self) -> None:
        scen = self._view.scenario
        if scen.key != "orientation" and _generic_module() is None:
            self.set_files([])
            self.set_summary(f"<p><b>{html.escape(scen.name)}</b> için C kodu üreticisi bu sürümde henüz yok. "
                             "Yönelim (IMU / MARG) senaryosunda tüm algoritmaların C kodu üretilebilir.</p>")
            self.set_status("Bu senaryo için C kodu yok.", "warning")
            return
        if not any(b.isChecked() for b in self._algo_checks.values()):
            self.set_status("En az bir algoritma seçin.", "warning")
            return
        try:
            module, opts = self._options()
            files = module.generate(opts)
        except (ValueError, KeyError) as exc:
            self.set_files([])
            self.set_summary(f"<p>Kod üretilemedi: {html.escape(str(exc))}</p>")
            self.set_status(str(exc), "critical")
            return
        self.set_files(files)
        self.set_summary("<p>Kod üretildi. Bellek ve maliyet <b>ölçülüyor</b> (derleme birkaç saniye sürebilir)…</p>")
        self.set_status(f"{len(files)} dosya üretildi · ölçüm sürüyor…")
        self.generate_button.setEnabled(False)
        compilers = tuple(mcu.COMPILERS)
        self._generation += 1  # önceki ölçüm sürüyorsa sonucu yok sayılır (arayüz beklemez)
        threading.Thread(target=_measure, args=(self._bridge, self._generation, module, opts, compilers),
                         name="spt-codegen-measure", daemon=True).start()

    def _measured(self, generation: int, opts, fp, cost, error: str) -> None:
        if generation != self._generation:
            return  # eski seçeneklerin ölçümü
        self.generate_button.setEnabled(True)
        if error:
            self.set_summary(f"<p>Kod üretildi ancak ölçüm yapılamadı: {html.escape(error)}</p>")
            self.set_status("Ölçüm başarısız (kod yine de kaydedilebilir).", "warning")
            return
        module = fusion_c if isinstance(opts, fusion_c.CodegenOptions) else _generic_module()
        try:
            files = module.generate(opts, footprint_rows=fp, cost_rows=cost)  # KULLANIM.md'ye tablolar eklenir
            self.set_files(files)
        except (ValueError, KeyError):
            pass
        self.set_summary(self._summary_html(opts, fp, cost))
        worst = max((c.cpu_load_percent for c in cost), default=0.0)
        state = "critical" if worst > 100 else "warning" if worst > 50 else "good"
        self.set_status(f"Ölçüm tamamlandı · en yüksek CPU yükü %{worst:.3g} @ {opts.fs_hz:g} Hz", state)

    def _summary_html(self, opts, fp, cost) -> str:
        t = self._view._theme.tokens
        prof = mcu.MCU_PROFILES[opts.mcu]
        names = self._view.scenario.algorithms
        flash_b, ram_b = opts.flash_kb * 1024, opts.ram_kb * 1024

        def name(key):
            return html.escape(names[key].name if key in names else key)

        rows = []
        methods: list[str] = []  # ayrıntılı yöntem metinleri tablonun altında numaralı not olarak

        def note(text: str) -> str:
            if text not in methods:
                methods.append(text)
            return f"<sup>{methods.index(text) + 1}</sup>"

        for r in fp:
            flash = r.flash_with_libm if r.flash_with_libm is not None else r.flash_code
            pct = f"{flash / flash_b * 100:.2g}%" if flash_b else "—"
            stack = "—" if r.stack is None else f"{r.stack_own or r.stack} / {r.stack}"
            measured = (f'<span style="color:{t.good}">ölçüldü</span>' if r.measured
                        else f'<span style="color:{t.warning}">tahmin</span>') + note(r.method)
            rows.append(f"<tr><td>{name(r.algorithm)}</td><td>{html.escape(mcu.COMPILERS.get(r.compiler, r.compiler))}</td>"
                        f"<td align='right'>{r.flash_code}</td><td align='right'>{r.flash_with_libm or '—'}</td>"
                        f"<td align='right'>{pct}</td><td align='right'>{r.ram_state}</td>"
                        f"<td align='right'>{r.ram_state / ram_b * 100:.2g}%</td><td align='right'>{stack}</td>"
                        f"<td>{measured}</td></tr>")
        crow = []
        for c in cost:
            load = c.cpu_load_percent
            color = t.critical if load > 100 else t.warning if load > 50 else t.good
            mag = " (MARG)" if getattr(c, "mag", False) else ""
            crow.append(f"<tr><td>{name(c.algorithm)}{mag}</td><td align='right'>{c.cycles:,.0f}</td>"
                        f"<td align='right'>{c.us_per_update:.3g}</td>"
                        f"<td align='right'><span style='color:{color}'>%{load:.3g}</span></td>"
                        f"<td align='right'>{c.max_rate_hz:,.0f}</td>"
                        f"<td>{note(c.method)}</td></tr>")
        head = (f"<h3>{html.escape(prof.name)} · {opts.clock_mhz:g} MHz · {html.escape(opts.real)} · "
                f"{html.escape(opts.opt)} · çağrı hızı {opts.fs_hz:g} Hz</h3>"
                f"<p>Flash {opts.flash_kb:g} KB · RAM {opts.ram_kb:g} KB · örnek çipler: {html.escape(prof.examples)}</p>")
        if getattr(opts, "scenario", None) == "position2d" and opts.real == "float":
            head += (f"<p><span style='color:{t.warning}'>⚠ float'ta konum orijinden uzaklaştıkça doğruluğunu "
                     "yitirir:</span> 20 km'ye kadar double ile aynı; 70–90 km'de konum hatası ~2×, hız hatası ~2.4×. "
                     "Geniş alanda çalışan araçlar için <b>double</b> seçin (ayrıntı: kullanım kılavuzu).</p>")
        fp_table = ("<h4>Bellek (derleyiciye göre)</h4><table border='1' cellspacing='0' cellpadding='4'>"
                    "<tr><th>Algoritma</th><th>Derleyici</th><th>Kod (B)</th><th>libm dahil (B)</th><th>Flash %</th>"
                    "<th>RAM durum (B)</th><th>RAM %</th><th>Yığın kendi / libm dahil (B)</th><th>Yöntem</th></tr>"
                    + "".join(rows) + "</table>")
        cost_table = ("<h4>İşlem maliyeti (adım başına)</h4><table border='1' cellspacing='0' cellpadding='4'>"
                      "<tr><th>Algoritma</th><th>Çevrim</th><th>µs</th><th>CPU yükü</th><th>En yüksek hız (Hz)</th>"
                      "<th>Not</th></tr>" + "".join(crow) + "</table>")
        notes = ("<p><b>Notlar:</b> Kod boyutu, makinedeki araç zinciriyle gerçekten derlenerek ölçülür (GCC: "
                 "arm-none-eabi-gcc, Clang: Arm Compiler 6 / Keil MDK ile aynı altyapı); IAR ve yerel derleyicisi "
                 "olmayan çekirdekler <i>tahmin</i>dir. RAM = durum yapısının boyutu (malloc yok). Yığın: en kötü çağrı "
                 "yolu; libm dahil değer büyük açı indirgemesi nedeniyle muhafazakârdır. Çevrim sayıları, ölçülen işlem "
                 "sayıları × çekirdeğin komut çevrim tablosu (bellek / dallanma payıyla) — gerçek donanımda DWT->CYCCNT "
                 "ile doğrulayın. CPU yükü %100'ü aşan algoritma bu hızda gerçek zamanlı çalışamaz.</p>")
        details = "<ol>" + "".join(f"<li>{html.escape(m)}</li>" for m in methods) + "</ol>"
        return head + fp_table + cost_table + "<h4>Yöntem ayrıntıları</h4>" + details + notes


def open_fusion_code_dialog(view) -> FusionCodeDialog:
    dlg = getattr(view, "_code_dialog", None)
    if dlg is None:
        dlg = view._code_dialog = FusionCodeDialog(view)
    dlg.load_from_view()
    dlg.refresh()
    dlg.show()
    dlg.raise_()
    return dlg
