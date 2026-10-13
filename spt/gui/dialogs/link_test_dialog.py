"""UDP Link Testi penceresi.

İki test (yalnız PC tarafı; kart yazılımında değişiklik gerekmez / yapılmadı):
  - Pasif: dinlenen bir SPTF akışını N saniye izler; veri / paket hızı, paket arası süre dağılımı (histogram,
    p50 / p95 / p99, en büyük boşluk), kayıp ve kayıp olayları, yineleme / sıra dışı, etkin örnekleme hızı (ppm) ve
    gönderici zaman damgasına göre gecikme değişimi. Sonuç, kullanıcının eşiklerine göre "gerçek zaman uygunluğu"
    olarak değerlendirilir ve Markdown / CSV olarak dışa aktarılabilir.
  - Yankı (RTT): hedefe sıra numaralı problar gönderir (spt.sources.echo_test); kartta UDP yankı sunucusu gerekir.
    Denemek için bu bilgisayarda yerel yankı sunucusu başlatılabilir.

Ayrı pencere (üst gezinme çubuğunda yeni görünüm yerine): ara sıra kullanılan bir tanı aracıdır ve Canlı İzleme
açıkken, grafikler izlenirken çalıştırılabilmelidir.
"""
from __future__ import annotations

import math
import multiprocessing as mp
import queue
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from spt.core.stream_store import TIMING_EDGES_US
from spt.gui.icons import icon, primary_icon
from spt.gui.theme import ThemeManager, style_native_titlebar, tokens
from spt.gui.widgets.common import FlowLayout, StatusBadge, combo, label
from spt.gui.widgets.plot import make_plot, pen
from spt.sources.echo_test import (
    DEFAULT_ECHO_PORT,
    MAX_PAYLOAD,
    MIN_PAYLOAD,
    EchoConfig,
    run_echo_process,
    run_echo_server,
)
from spt.sources.link_report import PassiveTest, Thresholds, fmt_us

UPDATE_MS = 250
POLL_MS = 100


def _table(columns: list[str]) -> QTableWidget:
    t = QTableWidget(0, len(columns))
    t.setHorizontalHeaderLabels(columns)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(26)
    t.setShowGrid(False)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    h = t.horizontalHeader()
    h.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    h.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    h.setStretchLastSection(True)
    return t


def _fill(table: QTableWidget, rows: list[tuple[str, ...]]) -> None:
    table.setRowCount(len(rows))
    for r, values in enumerate(rows):
        for c, value in enumerate(values):
            item = table.item(r, c)
            if item is None:
                table.setItem(r, c, QTableWidgetItem(value))
            elif item.text() != value:
                item.setText(value)


def _spin(lo: float, hi: float, value: float, suffix: str = "", decimals: int = 0, tip: str = ""):
    w = QDoubleSpinBox() if decimals else QSpinBox()
    if decimals:
        w.setDecimals(decimals)
    w.setRange(lo, hi)
    w.setValue(value)
    if suffix:
        w.setSuffix(suffix)
    w.setToolTip(tip)
    return w


def _save(parent: QWidget, title: str, default: str, text: str, kind: str) -> None:
    filt = "Markdown (*.md)" if kind == "md" else "CSV (*.csv)"
    path, _ = QFileDialog.getSaveFileName(parent, title, str(Path.home() / default), filt)
    if path:
        Path(path).write_text(text, encoding="utf-8-sig" if kind == "csv" else "utf-8")


class _LogTimeAxis(pg.AxisItem):
    """Logaritmik süre ekseni: değerler log10(µs); etiketler 1-2-5 dizisinde okunur süre (ör. 20 µs, 1 ms).
    pyqtgraph'ın log kipi dar aralıkta ara etiketleri üst üste bindirdiği için kullanılmaz."""

    def tickValues(self, minVal, maxVal, size):  # noqa: N802, N803 — pyqtgraph arayüzü
        lo, hi = math.floor(minVal), math.ceil(maxVal)
        decades = [float(d) for d in range(lo, hi + 1) if minVal <= d <= maxVal]
        steps = (2, 5) if maxVal - minVal <= 4 and size > 250 else ()
        minor = [d + math.log10(m) for d in range(lo, hi + 1) for m in steps if minVal <= d + math.log10(m) <= maxVal]
        return [(1.0, decades), (0.3, minor)]

    def tickStrings(self, values, scale, spacing):  # noqa: N802
        return [fmt_us(10 ** v) for v in values]


class _HistogramPlot:
    """Paket arası süre / RTT histogramı: logaritmik süre ekseni (x = log10 µs), çubuklar vurgu renginde."""

    def __init__(self, theme: ThemeManager, x_label: str):
        self.widget = make_plot(theme, x_label, "Paket")
        pi = self.widget.getPlotItem()
        old = pi.getAxis("bottom")
        axis = _LogTimeAxis("bottom")
        axis.setPen(old.pen())
        axis.setTextPen(old.textPen())
        axis.setTickFont(old.style.get("tickFont"))
        pi.setAxisItems({"bottom": axis})
        pi.setLabel("bottom", x_label)
        fill = pg.mkColor(theme.tokens.accent)
        fill.setAlpha(170)
        self._bars = pg.BarGraphItem(x0=[], x1=[], height=[], brush=fill, pen=pg.mkPen(theme.tokens.accent))
        pi.addItem(self._bars)
        self._marker = pg.InfiniteLine(angle=90, movable=False,
                                       pen=pg.mkPen(theme.tokens.text_secondary, width=1, style=Qt.PenStyle.DashLine))
        self._marker.hide()
        pi.addItem(self._marker)

    def set(self, edges_lo: np.ndarray, edges_hi: np.ndarray, counts: np.ndarray, mark: float | None = None) -> None:
        keep = counts > 0
        lo, hi = np.log10(np.maximum(edges_lo[keep], 1e-3)), np.log10(edges_hi[keep])
        self._bars.setOpts(x0=lo, x1=hi, height=counts[keep])
        if mark is not None and math.isfinite(mark) and mark > 0:
            self._marker.setValue(math.log10(mark))
            self._marker.show()
        else:
            self._marker.hide()


# ================================================================================================ pasif
class PassiveTab(QWidget):
    def __init__(self, theme: ThemeManager, sources):
        super().__init__()
        self._sources = sources
        self._test: PassiveTest | None = None
        self._report = None
        self._end_t: float | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)
        intro = label("Dinlenen (ya da oynatılan) bir SPTF akışını seçilen süre boyunca izler. Kart yazılımında "
                      "değişiklik gerekmez.", "Muted")
        intro.setWordWrap(True)
        lay.addWidget(intro)

        top = QHBoxLayout()
        top.setSpacing(8)
        self.stream = combo([])
        self.stream.setMinimumWidth(220)
        self.duration = _spin(1, 3600, 10, " s", tip="Test süresi")
        self.start_btn = QPushButton(primary_icon("ph.play"), "  Başlat")
        self.start_btn.setObjectName("Primary")
        self.start_btn.clicked.connect(self.toggle)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.progress.setMaximumHeight(6)
        top.addWidget(label("Akış", "FormLabel"))
        top.addWidget(self.stream, 1)
        top.addWidget(label("Süre", "FormLabel"))
        top.addWidget(self.duration)
        top.addWidget(self.start_btn)
        lay.addLayout(top)
        lay.addWidget(self.progress)

        # Eşikler satıra sığmazsa alt satıra kayar (Bağlantı sayfasında dar sütunda en az genişliği ≈ 1250 px'e
        # çıkarıp yan sütunu sıkıştırıyordu)
        thresholds = QWidget()
        th = FlowLayout(thresholds, spacing=12)
        self.max_gap = _spin(0, 10_000, 20, " ms", 2, "Paketler arası en büyük boşluk sınırı (0: değerlendirme)")
        self.loss = _spin(0, 100, 0.01, " %", 4, "Kayıp örnek oranı sınırı (0: değerlendirme)")
        self.jitter = _spin(0, 10_000_000, 5000, " µs", 0,
                            "Paket arası sürenin p99'u − beklenen aralık için sınır (0: değerlendirme)")
        self.ppm = _spin(0, 1_000_000, 500, " ppm", 0, "Etkin örnekleme hızının nominalden sapma sınırı (0: değerlendirme)")
        self.min_rate = _spin(0, 100_000, 0, " Mbit/s", 3, "En az ortalama veri hızı (0: değerlendirme)")
        for text, w in (("Boşluk ≤", self.max_gap), ("Kayıp ≤", self.loss), ("Titreşim p99 ≤", self.jitter),
                        ("Hız sapması ≤", self.ppm), ("Veri hızı ≥", self.min_rate)):
            pair = QWidget()
            row = QHBoxLayout(pair)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            row.addWidget(label(text, "FormLabel"))
            row.addWidget(w)
            th.addWidget(pair)
            w.valueChanged.connect(lambda *_: self._render())
        lay.addWidget(thresholds)

        verdict_row = QHBoxLayout()
        self.verdict = StatusBadge("Test çalıştırılmadı", "idle")
        verdict_row.addWidget(self.verdict)
        verdict_row.addStretch(1)
        self.export_md = QPushButton(icon("ph.export"), " Markdown…")
        self.export_csv = QPushButton(icon("ph.export"), " CSV…")
        for b, kind in ((self.export_md, "md"), (self.export_csv, "csv")):
            b.setEnabled(False)
            b.clicked.connect(lambda _=False, k=kind: self._export(k))
            verdict_row.addWidget(b)
        lay.addLayout(verdict_row)

        split = QSplitter(Qt.Orientation.Horizontal)
        left = QSplitter(Qt.Orientation.Vertical)
        self.results = _table(["Ölçü", "Değer"])
        self.checks = _table(["Kontrol", "Ölçülen", "Sınır", "Sonuç"])
        left.addWidget(self.results)
        left.addWidget(self.checks)
        left.setSizes([400, 160])
        right = QSplitter(Qt.Orientation.Vertical)
        self.hist = _HistogramPlot(theme, "Paket arası süre")
        self.rate_plot = make_plot(theme, "Süre", "Veri hızı", x_units="s", y_units="bit/s")
        self.rate_curve = self.rate_plot.plot(pen=pen(theme.tokens.accent, 1.5))
        right.addWidget(self.hist.widget)
        right.addWidget(self.rate_plot)
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([430, 470])
        lay.addWidget(split, 1)

        self._timer = QTimer(self)
        self._timer.setInterval(UPDATE_MS)
        self._timer.timeout.connect(self._tick)
        sources.stats_updated.connect(self._on_stats)
        self.refresh_streams()

    # ---------------------------------------------------------------- akışlar
    def refresh_streams(self, prefer: str = "") -> None:
        current = prefer or self.stream.currentData() or ""
        rates = self._sources.rates
        items = [(key, r.stats) for key, r in rates.items() if r.stats is not None]
        self.stream.blockSignals(True)
        self.stream.clear()
        for key, st in items:
            self.stream.addItem(f"{st.stream_id:02d} · {st.name or key} · {st.source}", key)
        keys = [k for k, _ in items]
        if current in keys:
            self.stream.setCurrentIndex(keys.index(current))
        self.stream.blockSignals(False)
        if not self.running:
            self.start_btn.setEnabled(bool(keys))
            if not keys and self._report is None:
                self.verdict.set_state("Akış yok: önce UDP dinlemeyi başlatın ya da bir kayıt açın", "idle")

    @property
    def running(self) -> bool:
        return self._test is not None and self._end_t is not None

    def toggle(self) -> None:
        if self.running:
            self.finish()
            return
        key = self.stream.currentData()
        if not key:
            return
        self._test = PassiveTest(key, self.thresholds())
        self._report = None
        self._end_t = math.inf          # çalışıyor; bitiş, ilk istatistik olayında olayın saatiyle belirlenir
        self.start_btn.setText("  Durdur")
        self.start_btn.setIcon(primary_icon("ph.stop"))
        self.stream.setEnabled(False)
        self.duration.setEnabled(False)
        self.verdict.set_state("Ölçülüyor…", "idle")
        self.progress.setValue(0)
        self._timer.start()

    def finish(self) -> None:
        self._end_t = None
        self._timer.stop()
        self.start_btn.setText("  Başlat")
        self.start_btn.setIcon(primary_icon("ph.play"))
        self.stream.setEnabled(True)
        self.duration.setEnabled(True)
        self.progress.setValue(1000)
        self._render(final=True)

    def thresholds(self) -> Thresholds:
        return Thresholds(self.max_gap.value(), self.loss.value(), self.jitter.value(), self.ppm.value(),
                          self.min_rate.value())

    def _on_stats(self, session: dict, rates: dict) -> None:
        if not self.running:
            known = {self.stream.itemData(i) for i in range(self.stream.count())}
            if known != {k for k, r in rates.items() if r.stats is not None}:
                self.refresh_streams()
            return
        rate = rates.get(self._test.key)
        if rate is None or rate.stats is None:
            return
        t = session.get("t")
        if t is None:
            return
        if self._test.first is None:
            self._end_t = t + self.duration.value()
        self._test.feed(t, rate.stats)
        if self._test.first is not None and self._end_t is not None and math.isfinite(self._end_t):
            span = self.duration.value()
            done = (t - self._test.first[0]) / span if span else 1.0
            self.progress.setValue(int(min(done, 1.0) * 1000))
            if t >= self._end_t:
                self.finish()

    def _tick(self) -> None:
        self._render()

    # ---------------------------------------------------------------- sonuç
    def _render(self, final: bool = False) -> None:
        if self._test is None:
            return
        rep = self._test.report(self.thresholds())
        if rep is None:
            return
        self._report = rep
        _fill(self.results, rep.rows())
        _fill(self.checks, [(c.name, c.value, c.limit, {True: "uygun", False: "AŞILDI", None: "ölçülemedi"}[c.ok])
                            for c in rep.checks])
        t = tokens()
        for r, c in enumerate(rep.checks):
            color = {True: t.good, False: t.critical}.get(c.ok, t.text_muted)
            self.checks.item(r, 3).setForeground(QColor(color))
        prefix = "" if final or not self.running else "Ara sonuç · "
        passed = rep.passed
        if passed is True:
            self.verdict.set_state(f"{prefix}Gerçek zamana uygun ({rep.duration_s:.1f} s)", "good")
        elif passed is False:
            failed = ", ".join(c.name for c in rep.checks if c.ok is False)
            self.verdict.set_state(f"{prefix}Uygun değil: {failed}", "critical")
        else:
            self.verdict.set_state(f"{prefix}Değerlendirilecek eşik yok", "idle")
        edges = np.concatenate(([0.0], TIMING_EDGES_US, [TIMING_EDGES_US[-1] * 10]))
        self.hist.set(edges[:-1], edges[1:], np.asarray(rep.hist, dtype=np.float64), rep.expected_us)
        series = self._test.series
        if series:
            arr = np.asarray(series)
            self.rate_curve.setData(arr[:, 0], arr[:, 1] * 1e6)
        for b in (self.export_md, self.export_csv):
            b.setEnabled(True)

    def _export(self, kind: str) -> None:
        if self._report is None:
            return
        name = f"link_testi_pasif.{kind}"
        _save(self, "Raporu kaydet", name, self._report.to_markdown() if kind == "md" else self._report.to_csv(), kind)

    def shutdown(self) -> None:
        self._timer.stop()
        self._end_t = None


# ================================================================================================ yankı
class EchoTab(QWidget):
    """Yankı testi ve isteğe bağlı yerel yankı sunucusu ayrı süreçlerde (bkz. spt.sources.echo_test)."""

    def __init__(self, theme: ThemeManager, sources):
        super().__init__()
        self._sources = sources
        self._ctx = mp.get_context("spawn")
        self._proc = self._server = None
        self._stop = self._server_stop = None
        self._events = self._server_events = None
        self._result = None
        self._saved_target = ("", DEFAULT_ECHO_PORT)
        self._poll = QTimer(self)
        self._poll.setInterval(POLL_MS)
        self._poll.timeout.connect(self._drain)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)
        intro = label("Hedefe sıra numaralı, zaman damgalı UDP probları gönderir ve geri dönenleri eşler: gidiş-dönüş "
                      "süresi (RTT), titreşim, kayıp ve yük altında verim. <b>Kartta UDP yankı sunucusu gerekir</b> "
                      "(ör. lwIP UDP echo, port 7); kart yazılımı bu sürümde değiştirilmedi. Denemek için "
                      "<i>Yerel yankı sunucusu</i> bu bilgisayarda bir yankıcı başlatır.", "Muted")
        intro.setWordWrap(True)
        lay.addWidget(intro)

        form = QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        target = QHBoxLayout()
        target.setSpacing(8)
        self.host = QLineEdit()
        self.host.setPlaceholderText("ör. 192.168.1.10")
        self.port = _spin(1, 65535, DEFAULT_ECHO_PORT, tip="Hedefin UDP yankı portu (RFC 862 echo: 7)")
        self.local = QCheckBox("Yerel yankı sunucusu")
        self.local.setToolTip("Bu bilgisayarda 127.0.0.1 üzerinde bir UDP yankıcı başlatır ve onu hedefler "
                              "(aracı denemek için; kartı ölçmez)")
        self.local.toggled.connect(self._on_local)
        target.addWidget(self.host, 1)
        target.addSpacing(8)
        target.addWidget(label("Port", "FormLabel"))
        target.addWidget(self.port)
        target.addSpacing(12)
        target.addWidget(self.local)
        form.addRow(label("Hedef", "FormLabel"), target)
        load_box = QWidget()   # dar sütunda alt satıra kayar (en az genişliği yan sütunu sıkıştırmasın)
        load = FlowLayout(load_box, spacing=12)
        self.payload = _spin(MIN_PAYLOAD, MAX_PAYLOAD, 64, " B", tip="UDP yük boyutu; 1472 B = standart MTU'da tek "
                                                                      "Ethernet çerçevesi")
        self.rate = _spin(0.1, 200_000, 100, " paket/s", 1, "Gönderim hızı. Yük altında verim için yük ve hızı artırın; "
                                                            "çok yüksek hızlarda PC zamanlaması yetişemeyebilir.")
        self.count = _spin(1, 10_000_000, 1000, " paket", tip="Gönderilecek prob sayısı")
        self.timeout = _spin(1, 60_000, 1000, " ms", tip="Yanıt için en uzun bekleme")
        for text, w in (("Yük", self.payload), ("Hız", self.rate), ("Sayı", self.count),
                        ("Zaman aşımı", self.timeout)):
            pair = QWidget()
            pair_lay = QHBoxLayout(pair)
            pair_lay.setContentsMargins(0, 0, 0, 0)
            pair_lay.setSpacing(6)
            pair_lay.addWidget(label(text, "FormLabel"))
            pair_lay.addWidget(w)
            load.addWidget(pair)
        form.addRow(label("Yük ve hız", "FormLabel"), load_box)
        lay.addLayout(form)

        row = QHBoxLayout()
        self.start_btn = QPushButton(primary_icon("ph.play"), "  Başlat")
        self.start_btn.setObjectName("Primary")
        self.start_btn.clicked.connect(self.toggle)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.progress.setMaximumHeight(6)
        self.verdict = StatusBadge("Test çalıştırılmadı", "idle")
        row.addWidget(self.start_btn)
        row.addWidget(self.verdict, 1)
        self.export_md = QPushButton(icon("ph.export"), " Markdown…")
        self.export_md.setEnabled(False)
        self.export_md.clicked.connect(self._export)
        row.addWidget(self.export_md)
        lay.addLayout(row)
        lay.addWidget(self.progress)
        self.diagnosis = label("", "Readout")
        self.diagnosis.setWordWrap(True)
        self.diagnosis.hide()
        lay.addWidget(self.diagnosis)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.results = _table(["Ölçü", "Değer"])
        right = QSplitter(Qt.Orientation.Vertical)
        self.rtt_plot = make_plot(theme, "Prob (sıra no)", "RTT", y_units="s")
        self.rtt_curve = self.rtt_plot.plot(pen=None, symbol="o", symbolSize=3, symbolPen=None,
                                            symbolBrush=pg.mkBrush(theme.tokens.accent))
        self.hist = _HistogramPlot(theme, "RTT")
        right.addWidget(self.rtt_plot)
        right.addWidget(self.hist.widget)
        split.addWidget(self.results)
        split.addWidget(right)
        split.setSizes([380, 520])
        lay.addWidget(split, 1)

    def suggest_host(self, host: str) -> None:
        if host and not self.host.text().strip() and not self.local.isChecked():
            self.host.setText(host)

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    # ---------------------------------------------------------------- yerel yankı sunucusu
    def _on_local(self, on: bool) -> None:
        if on:
            self._saved_target = (self.host.text(), self.port.value())
            self._server_stop = self._ctx.Event()
            self._server_events = self._ctx.Queue()
            self._server = self._ctx.Process(target=run_echo_server, args=("127.0.0.1", 0, self._server_stop,
                                                                          self._server_events),
                                             name="spt-echo-server", daemon=True)
            self._server.start()
            self.host.setText("127.0.0.1")
            self.start_btn.setEnabled(False)               # sunucu portunu bildirene kadar
            self.verdict.set_state("Yerel yankı sunucusu başlatılıyor…", "idle")
            self._poll.start()
        else:
            self._stop_server()
            host, port = self._saved_target
            self.host.setText(host)
            self.port.setValue(port)
            self.start_btn.setEnabled(True)
        self.host.setEnabled(not on)
        self.port.setEnabled(not on)

    def _stop_server(self) -> None:
        if self._server is None:
            return
        self._server_stop.set()
        self._server.join(timeout=2.0)
        if self._server.is_alive():
            self._server.terminate()
        self._server = self._server_stop = self._server_events = None

    # ---------------------------------------------------------------- test
    def config(self) -> EchoConfig:
        return EchoConfig(self.host.text().strip(), self.port.value(), self.payload.value(), self.rate.value(),
                          self.count.value(), self.timeout.value() / 1000.0)

    def toggle(self) -> None:
        if self.running:
            self._stop.set()
            return
        try:
            cfg = self.config()
            cfg.validate()
        except ValueError as exc:
            self.verdict.set_state(str(exc), "critical")
            return
        self._result = None
        self.diagnosis.hide()
        self.rtt_curve.setData([], [])
        self.hist.set(np.empty(0), np.empty(0), np.empty(0))
        _fill(self.results, [])
        self.progress.setValue(0)
        self.verdict.set_state(f"Gönderiliyor → {cfg.host}:{cfg.port}…", "idle")
        self.start_btn.setText("  Durdur")
        self.start_btn.setIcon(primary_icon("ph.stop"))
        self._stop = self._ctx.Event()
        self._events = self._ctx.Queue()
        self._proc = self._ctx.Process(target=run_echo_process, args=(cfg, self._stop, self._events,
                                                                      UPDATE_MS / 1000),
                                       name="spt-echo-test", daemon=True)
        self._proc.start()
        self._poll.start()

    def _drain(self) -> None:
        if self._server_events is not None:
            while True:
                try:
                    ev = self._server_events.get_nowait()
                except (queue.Empty, OSError, ValueError):
                    break
                if ev[0] == "ready":
                    self.port.setValue(ev[2])
                    self.start_btn.setEnabled(True)
                    self.verdict.set_state(f"Yerel yankı sunucusu hazır: {ev[1]}:{ev[2]}", "good")
                elif ev[0] == "error":
                    self.verdict.set_state(ev[1], "critical")
                    self.local.setChecked(False)
        if self._events is not None:
            last = None
            while True:
                try:
                    ev = self._events.get_nowait()
                except (queue.Empty, OSError, ValueError):
                    break
                last = ev[1]
            if last is not None:
                self._on_progress(last)
            elif self._proc is not None and not self._proc.is_alive() and (self._result is None
                                                                            or not self._result.finished):
                self._finished(None)                       # süreç sonuç bildirmeden kapandı
        if self._proc is None and self._server_events is None:
            self._poll.stop()

    def _on_progress(self, snap) -> None:
        self._result = snap
        cfg = snap.config
        self.progress.setValue(int(1000 * snap.sent / cfg.count))
        st = snap.stats
        payload_mbps, wire_mbps = snap.throughput
        rows = [
            ("Hedef", f"{cfg.host}:{cfg.port} · {cfg.payload} B · {cfg.rate_hz:g} paket/s"),
            ("Gönderilen / alınan", f"{snap.sent} / {snap.received}"),
            ("Kayıp", f"{snap.lost} ({snap.loss_pct:.3f} %)" if math.isfinite(snap.loss_pct) else "—"),
            ("Geç / yinelenen / bozuk", f"{snap.late} / {snap.duplicates} / {snap.invalid}"),
            ("ICMP port kapalı", f"{snap.unreachable}"),
            ("RTT en küçük / ortalama", f"{fmt_us(st['min'])} / {fmt_us(st['mean'])}"),
            ("RTT p50 / p95 / p99", f"{fmt_us(st['p50'])} / {fmt_us(st['p95'])} / {fmt_us(st['p99'])}"),
            ("RTT en büyük", fmt_us(st["max"])),
            ("Titreşim (std / ardışık fark)", f"{fmt_us(st['std'])} / {fmt_us(st['jitter'])}"),
            ("Verim (UDP yükü / hat)", f"{payload_mbps:.3f} / {wire_mbps:.3f} Mbit/s"
             if math.isfinite(payload_mbps) else "—"),
            ("Süre", f"{snap.elapsed_s:.2f} s"),
            ("Gönderim kayması", f"{snap.send_slips} (PC zamanlaması yetişemedi)" if snap.send_slips else "0"),
        ]
        _fill(self.results, rows)
        self.rtt_curve.setData(snap.plot_seq, snap.plot_rtt / 1e6)
        edges = np.concatenate(([0.0], TIMING_EDGES_US, [TIMING_EDGES_US[-1] * 10]))
        self.hist.set(edges[:-1], edges[1:], np.asarray(snap.hist, dtype=np.float64), st["p99"])
        if snap.finished:
            self._finished(snap)

    def _finished(self, snap) -> None:
        if self._proc is not None:
            self._proc.join(timeout=1.0)
        self._proc = self._stop = self._events = None
        self.start_btn.setText("  Başlat")
        self.start_btn.setIcon(primary_icon("ph.play"))
        if snap is None:
            self.verdict.set_state("Test süreci beklenmedik şekilde kapandı", "critical")
            return
        self.export_md.setEnabled(True)
        if snap.diagnosis:
            self.verdict.set_state("Yanıt yok" if not snap.error else "Hata", "critical")
            self.diagnosis.setText(snap.diagnosis)
            self.diagnosis.show()
        elif snap.lost:
            self.verdict.set_state(f"Bitti · {snap.loss_pct:.3f} % kayıp", "warning")
        else:
            state = "durduruldu" if snap.stopped else "bitti"
            self.verdict.set_state(f"Test {state} · kayıpsız · RTT p99 {fmt_us(snap.stats['p99'])}", "good")

    def _export(self) -> None:
        snap = self._result
        if snap is None:
            return
        lines = ["# UDP link testi (yankı / RTT)", "", "| Ölçü | Değer |", "|---|---|"]
        for r in range(self.results.rowCount()):
            lines.append(f"| {self.results.item(r, 0).text()} | {self.results.item(r, 1).text()} |")
        if snap.diagnosis:
            lines += ["", snap.diagnosis]
        _save(self, "Raporu kaydet", "link_testi_yanki.md", "\n".join(lines) + "\n", "md")

    def shutdown(self) -> None:
        self._poll.stop()
        if self._proc is not None:
            self._stop.set()
            self._proc.join(timeout=2.0)
            if self._proc.is_alive():
                self._proc.terminate()
            self._proc = None
        self._stop_server()


# ================================================================================================ pencere
class LinkTestDialog(QDialog):
    def __init__(self, theme: ThemeManager, sources, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("UDP Link Testi")
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.resize(980, 700)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        self.tabs = QTabWidget()
        self.passive = PassiveTab(theme, sources)
        self.echo = EchoTab(theme, sources)
        self.tabs.addTab(self.passive, icon("ph.activity"), "Pasif (akış)")
        self.tabs.addTab(self.echo, icon("ph.arrows-left-right"), "Yankı (RTT)")
        lay.addWidget(self.tabs, 1)
        style_native_titlebar(self, theme.tokens)

    def open_for(self, key: str = "") -> None:
        """Pencereyi gösterir; `key` verilirse pasif testte o akış seçilir, yankı hedefi akışın IP'si önerilir."""
        self.passive.refresh_streams(prefer=key)
        rate = self.passive._sources.rates.get(key) if key else None
        if rate is not None and rate.stats is not None and not rate.stats.source.startswith("127."):
            self.echo.suggest_host(rate.stats.source)
        self.show()
        self.raise_()
        self.activateWindow()

    def closeEvent(self, event) -> None:
        self.passive.shutdown()
        self.echo.shutdown()
        super().closeEvent(event)
