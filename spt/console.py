"""Komut satırı araçları için konsol ayarı.

Windows'ta çıktı bir boruya / dosyaya yönlendirildiğinde Python yerel kod sayfasını (ör. cp1254) kullanır;
"→", "·", "≤" gibi karakterler bu kod sayfasında yoksa print() hata verir ve araç çöker. Araçlar çıktıyı her
zaman UTF-8 yazar (kodlanamayan karakter '?' olur); arayüz ve testler de UTF-8 okur.
"""
from __future__ import annotations

import sys

CHILD_ENV = {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}  # alt süreçlere verilecek ortam


def utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # zaten yazılmış / kapalı akış
                pass
