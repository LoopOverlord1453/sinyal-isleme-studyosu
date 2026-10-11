"""MPC denetleyicisi için MCU'ya özel C kodu üretme penceresi.

Solda MCU özellikleri (çekirdek / FPU, saat, derleyici, optimizasyon, sayı tipi, flash / RAM, önek) ve MPC kod
seçenekleri (referans / MD önizlemesi) sorulur. Çağrı hızı MPC'nin örnekleme zamanından gelir (fs = 1/Ts). Sonuç:
  - <önek>.h / <önek>.c / <önek>_ornek.c ve KULLANIM_<önek>.md,
  - kod / sabit veri boyutu derleyiciye göre ÖLÇÜLÜR (makinede arm-none-eabi-gcc / clang varsa; IAR ve araç zinciri
    olmayan çekirdeklerde açıkça "tahmin"), RAM = durum yapısının boyutu, yığın = en kötü çağrı yolu,
  - maliyet: kapalı çevrim senaryosunda adım başına işlem sayıları (C++ sayaçla ölçülür) + MaxIterations
    yinelemeli en kötü durum (analitik üst sınır) → çevrim / µs / CPU yükü ve Ts bütçesine sığan yineleme sayısı.
Ölçüm arka plan iş parçacığında yapılır (MPC nesnesinin kopyasıyla); arayüz donmaz.
"""
from __future__ import annotations

import copy
import html
import logging
import threading

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QPushButton, QVBoxLayout, QWidget

from spt.codegen import mcu, mpc_c
from spt.gui.dialogs.code_dialog import CodeDialog
from spt.gui.theme import TERRAFLUX_DARK
from spt.gui.widgets.common import CollapsibleSection, combo, label

log = logging.getLogger("spt.gui")


class _Bridge(QObject):
    """Arka plan iş parçacığından arayüze sonuç (sinyal ana iş parçacığında kuyruklanır)."""
    done = pyqtSignal(int, object, object, object, object, str)  # nesil, mpc, seçenekler, footprint, maliyet, hata


def _measure(bridge: _Bridge, generation: int, obj, opts, name: str, compilers) -> None:
    """Boyut / maliyet ölçümü (daemon iş parçacığı: hiçbir yerde beklenmez, kapanışı geciktirmez)."""
    try:
        fp = mpc_c.footprint(obj, opts, compilers, name)
        cost = mpc_c.cost(obj, opts)
        bridge.done.emit(generation, obj, opts, fp, cost, "")
    except Exception as exc:  # noqa: BLE001 — ölçüm başarısız olsa da kod üretilmiş olmalı
        log.exception("MPC C kodu ölçümü başarısız")
        bridge.done.emit(generation, obj, opts, None, None, f"{type(exc).__name__}: {exc}")


class MpcCodeDialog(CodeDialog):
    def __init__(self, parent=None, tokens=None):
        side = QWidget()
        lay = QVBoxLayout(side)
        lay.setContentsMargins(0, 0, 8, 0)
        lay.setSpacing(6)
        intro = label("MCU özelliklerinizi seçin; MPC kodu, bellek ve en kötü durum süresi buna göre üretilir / "
                      "ölçülür.", "Muted")
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
        self.real.setToolTip("float: tek duyarlık FPU'lu çekirdeklerde hızlı; double: FPU'suz / SP-FPU çekirdekte "
                             "yazılım kayan nokta (yavaş)")
        self.flash = self._spin(1.0, 65536.0, 0, 1024.0, " KB")
        self.ram = self._spin(1.0, 65536.0, 0, 192.0, " KB")
        self.prefix = QLineEdit("mpc")
        self.prefix.setToolTip("Dosya ve C tanımlayıcı öneki (ör. irtifa → irtifa.h, irtifa_step()). Birden çok MPC "
                               "için farklı önekler verin.")
        self.rate = label("", "Muted")
        self.rate.setWordWrap(True)
        for name, w in (("Çekirdek", self.mcu), ("", self.examples), ("Saat", self.clock), ("Derleyici", self.compiler),
                        ("Optimizasyon", self.opt), ("Sayı tipi", self.real), ("Flash", self.flash),
                        ("RAM", self.ram), ("Önek", self.prefix)):
            if name:
                target.add_row(name, w)
            else:
                target.form.addRow(w)
        lay.addWidget(target)
        opts_box = CollapsibleSection("MPC kodu")
        self.ref_preview = QCheckBox("Referans önizlemesi (p satır)")
        self.ref_preview.setToolTip("Açık: step() referansın gelecek p örneğini alır (yörünge izleme); kapalı: tek "
                                    "satır (sabit hedef, daha az flash)")
        self.md_preview = QCheckBox("Ölçülen bozucu önizlemesi (p + 1 satır)")
        opts_box.form.addRow(self.ref_preview)
        opts_box.form.addRow(self.md_preview)
        opts_box.form.addRow(self.rate)
        lay.addWidget(opts_box)
        self.generate_button = QPushButton("Kodu Üret ve Ölç")
        self.generate_button.setObjectName("Primary")
        self.generate_button.clicked.connect(self.refresh)
        lay.addWidget(self.generate_button)
        lay.addStretch(1)
        super().__init__(tokens or TERRAFLUX_DARK, "MPC — C Kodu", "mpc_code", parent, side=side)
        self.mcu.currentIndexChanged.connect(self._on_mcu)
        self._bridge = _Bridge(self)
        self._bridge.done.connect(self._measured)
        self._generation = 0  # her ölçüm bir nesil: eski ölçümün sonucu gelirse yok sayılır
        self._mpc = None
        self._name = "mpc"
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
        self.flash.setValue(prof.flash_kb)
        self.ram.setValue(prof.ram_kb)
        fpu = {"none": "FPU yok — kayan nokta yazılımla (MPC için yavaş)", "sp": "FPU tek duyarlık: float önerilir",
               "dp": "FPU çift duyarlık: double da donanımda"}[prof.fpu]
        self.examples.setText(f"{fpu}<br>Örnek: {prof.examples}")
        if prof.fpu in ("none", "sp"):
            self.real.setCurrentIndex(0)

    # ---------------------------------------------------------------- MPC nesnesi
    def set_mpc(self, mpcobj, name: str | None = None) -> None:
        """Kod üretilecek MPC (anlık kopyası alınır: arayüzde sonradan yapılan değişiklik ölçümü bozmaz)."""
        self._mpc = copy.deepcopy(mpcobj)
        self._name = str(name or "mpc")
        self.setWindowTitle(f"MPC — C Kodu · {self._name}")
        self.prefix.setText(mpc_c.default_prefix(self._name))
        try:
            d = self._mpc.data
        except Exception as exc:  # noqa: BLE001 — geçersiz tasarım: refresh() iletisi gösterir
            self.rate.setText(f"MPC verisi hesaplanamadı: {html.escape(str(exc))}")
            return
        self.md_preview.setEnabled(d.nmd > 0)
        if d.nmd == 0:
            self.md_preview.setChecked(False)
        self.rate.setText(f"Çağrı hızı = 1/Ts = {1.0 / d.Ts:g} Hz (Ts = {d.Ts:g} s) · p = {d.p} · MV {d.nmv}, "
                          f"MD {d.nmd}, MO {d.nmy} · QP {d.degrees if d.isQP else d.nfree} değişken, "
                          f"{d.Ac.shape[0] if d.isQP else 0} kısıt")

    def _options(self) -> mpc_c.MpcCodegenOptions:
        return mpc_c.MpcCodegenOptions(real=self.real.currentText(), mcu=self.mcu.currentData(),
                                       compiler=self.compiler.currentData(), opt=self.opt.currentText(),
                                       clock_mhz=self.clock.value(), flash_kb=self.flash.value(),
                                       ram_kb=self.ram.value(), prefix=self.prefix.text().strip() or "mpc",
                                       ref_preview=self.ref_preview.isChecked(),
                                       md_preview=self.md_preview.isChecked() and self.md_preview.isEnabled())

    # ---------------------------------------------------------------- üretim
    def shutdown(self) -> None:
        self._generation += 1

    def refresh(self) -> None:
        if self._mpc is None:
            self.set_status("Önce bir MPC tasarımı seçin.", "warning")
            return
        try:
            opts = self._options()
            files = mpc_c.generate(self._mpc, opts, self._name)
        except (ValueError, KeyError) as exc:
            self._generation += 1                       # süren eski ölçümün sonucu artık gösterilmez
            self.generate_button.setEnabled(True)
            self.set_files([])
            self.set_summary(f"<p>Kod üretilemedi: {html.escape(str(exc))}</p>")
            self.set_status(html.escape(str(exc)), "critical")
            return
        self.set_files(files)
        self.set_summary("<p>Kod üretildi. Bellek ve maliyet <b>ölçülüyor</b> (derleme ve kapalı çevrim benzetimi "
                         "birkaç saniye sürebilir)…</p>")
        self.set_status(f"{len(files)} dosya üretildi · ölçüm sürüyor…")
        self.generate_button.setEnabled(False)
        self._generation += 1
        threading.Thread(target=_measure, args=(self._bridge, self._generation, copy.deepcopy(self._mpc), opts,
                                                self._name, tuple(mcu.COMPILERS)),
                         name="spt-mpc-codegen-measure", daemon=True).start()

    def _measured(self, generation: int, obj, opts, fp, cost, error: str) -> None:
        if generation != self._generation:
            return
        self.generate_button.setEnabled(True)
        if error:
            self.set_summary(f"<p>Kod üretildi ancak ölçüm yapılamadı: {html.escape(error)}</p>")
            self.set_status("Ölçüm başarısız (kod yine de kaydedilebilir).", "warning")
            return
        try:
            self.set_files(mpc_c.generate(obj, opts, self._name, footprint_rows=fp, cost_rows=cost))
        except (ValueError, KeyError):
            pass
        self.set_summary(self._summary_html(obj, opts, fp, cost))
        c = cost[0]
        state = "critical" if c.load_max > 100 else "warning" if c.load_wc > 100 or c.load_max > 50 else "good"
        self.set_status(f"Ölçüm tamamlandı · CPU yükü ort. %{c.load_mean:.3g}, en pahalı adım %{c.load_max:.3g}, "
                        f"en kötü durum %{c.load_wc:.3g} @ {1.0 / obj.data.Ts:g} Hz", state)

    def _summary_html(self, obj, opts, fp, cost) -> str:
        t = self._tokens
        prof = mcu.MCU_PROFILES[opts.mcu]
        d = obj.data
        fs = 1.0 / d.Ts
        flash_b, ram_b = opts.flash_kb * 1024, opts.ram_kb * 1024
        methods: list[str] = []

        def note(text: str) -> str:
            if text not in methods:
                methods.append(text)
            return f"<sup>{methods.index(text) + 1}</sup>"

        rows = []
        for r in fp:
            flash = r.flash_with_libm if r.flash_with_libm is not None else r.flash_code
            stack = "—" if r.stack is None else f"{r.stack_own or r.stack} / {r.stack}"
            meas = (f'<span style="color:{t.good}">ölçüldü</span>' if r.measured
                    else f'<span style="color:{t.warning}">tahmin</span>') + note(r.method)
            rows.append(f"<tr><td>{html.escape(mcu.COMPILERS.get(r.compiler, r.compiler))}</td>"
                        f"<td align='right'>{r.flash_code}</td><td align='right'>{r.flash_with_libm or '—'}</td>"
                        f"<td align='right'>{flash / flash_b * 100:.2g}%</td><td align='right'>{r.ram_state}</td>"
                        f"<td align='right'>{r.ram_state / ram_b * 100:.2g}%</td><td align='right'>{stack}</td>"
                        f"<td>{meas}</td></tr>")
        c = cost[0]

        def load_cell(v):
            color = t.critical if v > 100 else t.warning if v > 50 else t.good
            return f"<span style='color:{color}'>%{v:.3g}</span>"

        crow = [f"<tr><td>{lbl}</td><td align='right'>{cyc:,.0f}</td><td align='right'>{us:.3g}</td>"
                f"<td align='right'>{load_cell(ld)}</td></tr>"
                for lbl, cyc, us, ld in (("Ortalama adım", c.cycles_mean, c.us_mean, c.load_mean),
                                         ("En pahalı adım (senaryo)", c.cycles_max, c.us_max, c.load_max),
                                         (f"En kötü durum ({c.max_iter} yineleme)", c.cycles_wc, c.us_wc, c.load_wc))]
        head = (f"<h3>{html.escape(self._name)} · {html.escape(prof.name)} · {opts.clock_mhz:g} MHz · "
                f"{html.escape(opts.real)} · {html.escape(opts.opt)} · çağrı hızı {fs:g} Hz (Ts = {d.Ts:g} s)</h3>"
                f"<p>Flash {opts.flash_kb:g} KB · RAM {opts.ram_kb:g} KB · örnek çipler: {html.escape(prof.examples)}"
                f"</p>")
        fp_table = ("<h4>Bellek (derleyiciye göre)</h4><table border='1' cellspacing='0' cellpadding='4'>"
                    "<tr><th>Derleyici</th><th>Kod + sabitler (B)</th><th>libm dahil (B)</th><th>Flash %</th>"
                    "<th>RAM durum (B)</th><th>RAM %</th><th>Yığın kendi / libm dahil (B)</th><th>Yöntem</th></tr>"
                    + "".join(rows) + "</table>")
        cost_table = ("<h4>İşlem maliyeti (adım başına)</h4><table border='1' cellspacing='0' cellpadding='4'>"
                      "<tr><th></th><th>Çevrim</th><th>µs</th><th>CPU yükü</th></tr>" + "".join(crow) + "</table>"
                      f"<p>Bir QP yinelemesi ≤ {c.cycles_iter:,.0f} çevrim · senaryoda yineleme ort. "
                      f"{c.iters_mean:.2f}, en çok {c.iters_max} · Ts bütçesine en çok <b>{c.iter_budget}</b> "
                      f"yineleme sığar (sınır {c.max_iter}){note(c.method)}</p>")
        warn = ""
        if c.iter_budget < c.max_iter:
            warn = (f"<p><span style='color:{t.warning}'>⚠ En kötü durum ({c.max_iter} yineleme) Ts'yi aşıyor.</span> "
                    f"<code>-D{opts.prefix.upper()}_MAX_ITER=n</code> ile sınırı ölçülen en çok yinelemenin üstünde, "
                    "bütçenin altında bir değere indirin (sınıra takılan adımda MV önceki değerinde tutulur) ya da "
                    "daha hızlı çekirdek / daha kısa ufuk seçin. Ayrıntı: KULLANIM.</p>")
        notes = ("<p><b>Notlar:</b> Kod boyutu makinedeki araç zinciriyle gerçekten derlenerek ölçülür; IAR ve yerel "
                 "derleyicisi olmayan çekirdekler <i>tahmin</i>dir. RAM = durum yapısı (QP çalışma alanı dahil; malloc "
                 "yok). İşlem sayıları üretilen C'nin kapalı çevrimde C++ sayaçla çalıştırılmasıyla ölçülür; en kötü "
                 "durum kodun yapısından türetilmiş üst sınırdır; çevrimler çekirdek tablosundan tahmindir — gerçek "
                 "donanımda DWT->CYCCNT ile doğrulayın.</p>")
        details = "<ol>" + "".join(f"<li>{html.escape(m)}</li>" for m in methods) + "</ol>"
        return head + warn + fp_table + cost_table + "<h4>Yöntem ayrıntıları</h4>" + details + notes


def open_mpc_code_dialog(parent, mpcobj, name: str | None = None) -> MpcCodeDialog:
    """MPC için C kodu penceresini açar (pencere üst nesnede saklanır ve yeniden kullanılır)."""
    dlg = getattr(parent, "_mpc_code_dialog", None) if parent is not None else None
    if dlg is None:
        tokens = getattr(getattr(parent, "_theme", None), "tokens", None)
        dlg = MpcCodeDialog(parent, tokens)
        if parent is not None:
            try:
                parent._mpc_code_dialog = dlg
            except AttributeError:
                pass
    dlg.set_mpc(mpcobj, name)
    dlg.refresh()
    dlg.show()
    dlg.raise_()
    return dlg
