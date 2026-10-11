"""Signal Processing Toolbox başlatıcısı.

Çift tıklayarak, `py run.py` ile ya da `.venv\\Scripts\\python run.py` ile çalıştırılabilir.
Proje ortamı (.venv) dışından başlatılırsa kendini .venv içindeki Python ile yeniden başlatır;
bu yüzden bu dosya .venv kontrolünden önce yalnızca standart kütüphaneyi kullanır.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"


def _venv_python() -> Path | None:
    if os.name == "nt":
        # pythonw: arkada boş bir konsol penceresi açık kalmaz
        candidates = (VENV / "Scripts" / "pythonw.exe", VENV / "Scripts" / "python.exe")
    else:
        candidates = (VENV / "bin" / "python3", VENV / "bin" / "python")
    return next((p for p in candidates if p.exists()), None)


def _relaunch_in_venv() -> int:
    python = _venv_python()
    if python is None:
        print("Proje ortamı (.venv) bulunamadı. Kurmak için proje klasöründe şunları çalıştırın:\n"
              "  py -3.14 -m venv .venv\n"
              "  .venv\\Scripts\\python -m pip install -r requirements.txt")
        input("\nKapatmak için Enter'a basın...")
        return 1
    subprocess.Popen([str(python), str(ROOT / "run.py"), *sys.argv[1:]], cwd=ROOT)
    return 0


def _show_error(exc_type, exc, tb) -> None:
    """Konsolsuz (pythonw) çalışırken hataların kaybolmaması için mesaj kutusunda gösterir."""
    if sys.stderr is not None:
        sys.__excepthook__(exc_type, exc, tb)
    if os.name == "nt":
        import ctypes
        import traceback

        text = "".join(traceback.format_exception(exc_type, exc, tb))
        ctypes.windll.user32.MessageBoxW(None, text[-3000:], "Signal Processing Toolbox — hata", 0x10)


if __name__ == "__main__":
    if Path(sys.prefix).resolve() != VENV.resolve():
        sys.exit(_relaunch_in_venv())

    sys.excepthook = _show_error
    from spt.app import main

    sys.exit(main())
