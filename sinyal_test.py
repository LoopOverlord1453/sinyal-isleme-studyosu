"""Gerçek Zamanlı Sinyal Test Uygulaması başlatıcısı (Stüdyo'ya SPTF/UDP ile canlı sinyal gönderir).

Çift tıklayarak, `py sinyal_test.py` ile ya da `.venv\\Scripts\\python sinyal_test.py` ile çalıştırılabilir; komut
satırı seçenekleri `python -m spt.sender --help` ile aynıdır. Proje ortamı (.venv) dışından başlatılırsa kendini
.venv içindeki Python ile yeniden başlatır (run.py ile aynı yöntem).
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"


def _venv_python() -> Path | None:
    if os.name == "nt":
        candidates = (VENV / "Scripts" / "pythonw.exe", VENV / "Scripts" / "python.exe")
    else:
        candidates = (VENV / "bin" / "python3", VENV / "bin" / "python")
    return next((p for p in candidates if p.exists()), None)


if __name__ == "__main__":
    if Path(sys.prefix).resolve() != VENV.resolve():
        python = _venv_python()
        if python is None:
            print("Proje ortamı (.venv) bulunamadı; kurulum için README'ye bakın.")
            sys.exit(1)
        subprocess.Popen([str(python), str(ROOT / "sinyal_test.py"), *sys.argv[1:]], cwd=ROOT)
        sys.exit(0)
    sys.path.insert(0, str(ROOT))
    from spt.sender.__main__ import main

    sys.exit(main())
