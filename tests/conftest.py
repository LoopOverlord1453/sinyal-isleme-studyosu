import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Ekransız (offscreen) Qt, sistem yazı tiplerini kendiliğinden bulamaz ve çok daha geniş bir yedek yazı tipine düşer:
# yerleşim ölçüleri (kart / panel genişlikleri) gerçek uygulamadakinden farklı çıkar. Windows'ta sistem yazı tiplerini
# gösteriyoruz; QApplication henüz oluşturulmadığı için toplama aşamasında ayarlamak yeterli.
_FONTS = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
if os.name == "nt" and _FONTS.is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", str(_FONTS))
