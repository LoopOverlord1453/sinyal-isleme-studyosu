"""Uygulamadaki tüm komutların (QAction) tek merkezden tanımı.

Menüler, üst çubuk, komut paleti ve kısayol listesi aynı kayıttan beslenir.
`phase`, komutun hangi geliştirme aşamasında işlevsel hale geleceğini belirtir;
henüz uygulanmamış komutlar tetiklendiğinde günlüğe bilgi yazılır.
"""
from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QObject
from PyQt6.QtGui import QAction, QKeySequence

from spt.gui.icons import icon


@dataclass(frozen=True)
class ActionSpec:
    key: str
    text: str
    icon: str | None = None
    shortcut: str | None = None
    tip: str | None = None
    checkable: bool = False
    phase: int | None = None


SPECS: tuple[ActionSpec, ...] = (
    # Dosya
    ActionSpec("file.new_project", "Yeni Proje", "ph.file-plus", "Ctrl+N", phase=5),
    ActionSpec("file.open_project", "Proje Aç…", "ph.folder-open", "Ctrl+O", phase=5),
    ActionSpec("file.save_project", "Projeyi Kaydet", "ph.floppy-disk", "Ctrl+S", phase=5),
    ActionSpec("import.sptr", "SPTF Kaydı (.sptr)…", "ph.file-arrow-down", "Ctrl+I", phase=1),
    ActionSpec("import.csv", "CSV / TXT…", "ph.table", phase=4),
    ActionSpec("import.wav", "WAV…", "ph.waves", phase=4),
    ActionSpec("import.npy", "NumPy (.npy)…", "ph.database", phase=4),
    ActionSpec("export.csv", "CSV…", "ph.table", phase=4),
    ActionSpec("export.wav", "WAV…", "ph.waves", phase=4),
    ActionSpec("export.png", "Grafik (PNG / SVG)…", "ph.image", phase=2),
    ActionSpec("app.quit", "Çıkış", "ph.sign-out", "Ctrl+Q"),
    # Düzen
    ActionSpec("edit.undo", "Geri Al", "ph.arrow-counter-clockwise", "Ctrl+Z",
               "Filtre tasarımı / işlem zinciri değişikliğini geri al"),
    ActionSpec("edit.redo", "Yinele", "ph.arrow-clockwise", "Ctrl+Y",
               "Geri alınan değişikliği yinele"),
    ActionSpec("edit.copy_plot", "Grafiği Kopyala", "ph.copy", "Ctrl+Shift+C", phase=2),
    ActionSpec("edit.preferences", "Tercihler…", "ph.gear-six", "Ctrl+,", phase=5),
    # Görünüm
    ActionSpec("view.live", "Canlı İzleme", "ph.chart-line", "Ctrl+1", checkable=True),
    ActionSpec("view.siggen", "Sinyal Üreteci", "mdi6.sine-wave", "Ctrl+3",
               "DAC için dalga şekli tasarla (hazır / formül / eğri / harmonik / CSV), nicemlenmiş çıkışı ve "
               "spektrumunu incele, MCU için C kodu üret", checkable=True),
    ActionSpec("view.filter", "Filtre Tasarımcısı", "ph.funnel", "Ctrl+4", checkable=True),
    ActionSpec("view.review", "İnceleme", "ph.snowflake", "Ctrl+2",
               "Çalışma Alanı'ndaki dondurulmuş sinyalleri incele", checkable=True),
    ActionSpec("view.fusion", "Sensör Füzyonu", "ph.compass", "Ctrl+5",
               "Senaryo tabanlı füzyon: IMU / MARG yönelimi, barometre + ivme irtifası, GNSS + odometri 2B "
               "konum, çok sensörlü doğrusal Kalman — Mahony, Madgwick, KF, EKF, UKF, parçacık filtresi",
               checkable=True),
    ActionSpec("view.control", "Kontrol Tasarımı", "ph.target", "Ctrl+6",
               "Model, sistem tanımlama, kararlılık analizi, PID / LQR / MPC tasarımı ve blok diyagram benzetimi "
               "(MATLAB Control System / System Identification / MPC araç kutuları ve Simulink karşılığı)",
               checkable=True),
    ActionSpec("view.command_palette", "Komut Paleti", "ph.magnifying-glass", "Ctrl+K"),
    ActionSpec("view.reset_layout", "Yerleşimi Sıfırla", "ph.layout"),
    ActionSpec("view.fullscreen", "Tam Ekran", "ph.corners-out", "F11", checkable=True),
    # Bağlantı ve akış
    ActionSpec("conn.listen", "UDP Dinlemeyi Başlat", "ph.broadcast", "F5",
               "UDP portunu dinlemeye başla / durdur", checkable=True, phase=1),
    ActionSpec("conn.settings", "Bağlantı Ayarları…", "ph.sliders-horizontal", phase=1),
    ActionSpec("stream.run", "Çalıştır / Durdur", "ph.play", "F7",
               "Canlı görüntü güncellemesini başlat / durdur", checkable=True, phase=1),
    ActionSpec("stream.freeze", "Dondur (Anlık Görüntü)", "ph.snowflake", "F6",
               "Tampondaki veriyi Çalışma Alanı'na kopyala (canlı akış devam ederken değişmeden incelenir)"),
    ActionSpec("stream.record", "Kaydı Başlat", "ph.record", "Ctrl+R",
               "Gelen çerçeveleri .sptr dosyasına kaydet", checkable=True, phase=1),
    ActionSpec("stream.replay", "Kaydı Oynat…", "ph.film-strip", phase=1),
    # Analiz
    ActionSpec("an.fft", "FFT Spektrum", "ph.chart-bar", phase=2),
    ActionSpec("an.psd", "Güç Spektral Yoğunluğu (Welch)", "ph.chart-line-up", phase=2),
    ActionSpec("an.measure", "Ölçümler", "ph.ruler", phase=2),
    ActionSpec("an.cursors", "İmleçler", "ph.crosshair", "Ctrl+M", checkable=True, phase=2),
    ActionSpec("an.corr", "Korelasyon", "ph.intersect", phase=4),
    ActionSpec("an.hilbert", "Zarf ve Anlık Frekans (Hilbert)", "ph.activity", phase=4),
    ActionSpec("an.resample", "Yeniden Örnekleme", "ph.arrows-out-line-horizontal", phase=4),
    ActionSpec("an.adc", "ADC Kalite Metrikleri (SNR, THD, ENOB)", "ph.gauge", phase=4),
    # Filtre
    ActionSpec("flt.fir", "FIR Filtre…", "ph.funnel-simple", tip="Filtre Tasarımcısı'nı FIR ailesiyle aç"),
    ActionSpec("flt.iir", "IIR Filtre…", "ph.funnel-simple", tip="Filtre Tasarımcısı'nı IIR ailesiyle aç"),
    ActionSpec("flt.notch", "Notch (50 / 60 Hz)"),
    ActionSpec("flt.comb", "Tarak (Comb) — Şebeke + Harmonikler"),
    ActionSpec("flt.dc_block", "DC Engelleyici"),
    ActionSpec("flt.ema", "Üstel Ortalama (EMA)"),
    ActionSpec("flt.kalman", "Kalman (Kararlı Hâl)"),
    ActionSpec("flt.alpha_beta", "Alfa-Beta İzleyici"),
    ActionSpec("flt.moving_avg", "Hareketli Ortalama"),
    ActionSpec("flt.gaussian", "Gauss"),
    ActionSpec("flt.median", "Medyan"),
    ActionSpec("flt.savgol", "Savitzky-Golay"),
    ActionSpec("flt.differentiator", "Türev Alıcı"),
    ActionSpec("flt.integrator", "Sızdıran İntegratör"),
    ActionSpec("flt.hilbert", "Hilbert Dönüştürücü"),
    ActionSpec("flt.allpass", "Tüm Geçiren (Faz Düzeltici)"),
    ActionSpec("flt.chain", "İşlem Zinciri…", "ph.flow-arrow", tip="İşlem Zinciri panelini göster"),
    ActionSpec("flt.design", "Tasarla", "ph.magic-wand", tip="Filtreyi şimdi yeniden tasarla"),
    ActionSpec("flt.apply", "Sinyale Uygula", "ph.lightning", "Ctrl+Return",
               "Tasarımı işlem zincirine ekle ve canlı akışa uygula"),
    ActionSpec("flt.export", "Katsayıları Dışa Aktar…", "ph.export", "Ctrl+E",
               "CSV / C başlık dosyası (float, Q15, Q31, CMSIS-DSP)"),
    # Araçlar
    ActionSpec("tools.generator", "Sinyal Üreteci…", "mdi6.sine-wave"),
    ActionSpec("tools.simulator", "Cihaz Simülatörü…", "ph.cpu", phase=1),
    ActionSpec("tools.packet_inspector", "Paket İnceleyici", "ph.brackets-curly", phase=2),
    # Yardım
    ActionSpec("help.docs", "Kullanım Kılavuzu", "ph.book-open", "F1", phase=5),
    ActionSpec("help.protocol", "SPTF Protokol Dokümanı", "ph.file-text"),
    ActionSpec("help.shortcuts", "Klavye Kısayolları", "ph.keyboard"),
    ActionSpec("help.about", "Hakkında", "ph.info"),
)


class ActionRegistry(QObject):
    def __init__(self, parent: QObject):
        super().__init__(parent)
        self._specs = {s.key: s for s in SPECS}
        self._actions: dict[str, QAction] = {}
        for spec in SPECS:
            act = QAction(spec.text, parent)
            if spec.icon:
                act.setIcon(icon(spec.icon))
            if spec.shortcut:
                act.setShortcut(QKeySequence(spec.shortcut))
            tip = spec.tip or spec.text
            if spec.shortcut:
                tip = f"{tip}  ({spec.shortcut})"
            act.setToolTip(tip)
            act.setStatusTip(spec.tip or spec.text)
            act.setCheckable(spec.checkable)
            self._actions[spec.key] = act

    def __getitem__(self, key: str) -> QAction:
        return self._actions[key]

    def spec(self, key: str) -> ActionSpec:
        return self._specs[key]

    def items(self):
        return self._actions.items()
