"""Uygulama giriş noktası: QApplication, tema ve ana pencere."""
import logging
import os
import sys

# Qt bağlamasını, pyqtgraph ve qtawesome import edilmeden önce sabitle.
os.environ.setdefault("QT_API", "pyqt6")
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt6")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from spt import APP_NAME, ORG_NAME, __version__  # noqa: E402
from spt.gui.main_window import MainWindow  # noqa: E402
from spt.gui.theme import ThemeManager  # noqa: E402


def _set_windows_app_id() -> None:
    """Görev çubuğunda python.exe yerine uygulamanın kendi ikonunun görünmesini sağlar."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("spt.signal_processing_toolbox")
    except (AttributeError, OSError):
        pass


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("spt").setLevel(logging.INFO)
    _set_windows_app_id()

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)
    app.setApplicationVersion(__version__)

    theme = ThemeManager(app)
    theme.apply()

    window = MainWindow(theme)
    window.show()
    return app.exec()
