"""Bağlantı görünümü: UDP alıcı / akış istatistikleri, PC ağ arayüzü ve UDP link testi tek sayfada.

Üst gezinme çubuğundaki "Bağlantı" ile açılır. Solda canlı bağlantı istatistikleri (akış, zamanlama, oturum, ağ
arayüzü), sağda link testi sekmeleri: pasif (gelen akıştan hız, paket arası süre dağılımı, kayıp, gerçek zaman
uygunluğu) ve yankı (gidiş-dönüş süresi; kartta UDP yankı sunucusu gerekir).
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QSplitter, QTabWidget, QVBoxLayout, QWidget

from spt.gui.dialogs.link_test_dialog import EchoTab, PassiveTab
from spt.gui.icons import icon
from spt.gui.theme import ThemeManager
from spt.gui.widgets.common import PageHeader
from spt.gui.widgets.link_stats import LinkStatsPanel

STATS_MAX_WIDTH = 560   # tam sayfada istatistik sütunu yan paneldekinden geniş olabilir
STATS_WIDTH = 400       # ilk açılıştaki genişlik (sürüklenerek değiştirilebilir)


class ConnectionView(QWidget):
    def __init__(self, theme: ThemeManager, sources):
        super().__init__()
        self._sources = sources
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        self.header = PageHeader("Bağlantı", "Akış istatistikleri, PC ağ arayüzü ve UDP link testi")
        lay.addWidget(self.header)

        self.stats = LinkStatsPanel()
        self.stats.setMaximumWidth(STATS_MAX_WIDTH)
        self.stats.link_test_requested.connect(lambda key: self.open_test(key))
        self.tabs = QTabWidget()
        self.tabs.setObjectName("LinkTabs")
        self.passive = PassiveTab(theme, sources)
        self.echo = EchoTab(theme, sources)
        self.tabs.addTab(self.passive, icon("ph.activity"), "Pasif test (akış)")
        self.tabs.addTab(self.echo, icon("ph.arrows-left-right"), "Yankı testi (RTT)")

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        split.addWidget(self.stats)
        split.addWidget(self.tabs)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        self.split = split
        self._sized = False
        lay.addWidget(split, 1)

    def showEvent(self, event) -> None:
        """İlk gösterimde istatistik sütununa geniş yer. Yığındaki sayfa gizliyken boyutlanır, görünür olunca boyut
        olayı gelmez; kurulumda verilen boylar da yerleşimde sıfırlanıyordu (dar sütunda "35.4 ms · beklenen 35.5 ms"
        gibi satırlar kırılıyordu). Boy, gösterimden sonraki ilk olay döngüsünde verilir."""
        super().showEvent(event)
        if not self._sized:
            QTimer.singleShot(0, self._initial_split)

    def _initial_split(self) -> None:
        width = self.split.width()
        if not self._sized and width >= 2 * STATS_WIDTH:
            self._sized = True
            self.split.setSizes([STATS_WIDTH, width - STATS_WIDTH])

    def update_stats(self, session: dict, rates: dict) -> None:
        self.stats.update_stats(session, rates)

    def clear(self) -> None:
        self.stats.clear()

    def current_key(self) -> str:
        return self.stats.current_key()

    def open_test(self, key: str = "", echo: bool = False) -> None:
        """Test sekmesine geçer; `key` verilirse pasif testte o akış seçilir, yankı hedefi akışın IP'si önerilir."""
        key = key or self.current_key()
        self.passive.refresh_streams(prefer=key)
        rate = self._sources.rates.get(key) if key else None
        if rate is not None and rate.stats is not None and not rate.stats.source.startswith("127."):
            self.echo.suggest_host(rate.stats.source)
        self.tabs.setCurrentWidget(self.echo if echo else self.passive)

    def shutdown(self) -> None:
        """Yankı testi süreci ve yerel yankı sunucusu durur (uygulama kapanırken)."""
        self.passive.shutdown()
        self.echo.shutdown()
