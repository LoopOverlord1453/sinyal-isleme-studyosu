import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Testler kullanıcının kayıtlı ayarlarını (tema, yerleşim) okumasın: ayar dosyası belirtilmemişse geçici bir INI kullanılır
# (ör. tema fikstürü kayıtlı tercihi okur; varsayılan açık tema). Testler kendi dosyalarını monkeypatch ile verir.
os.environ.setdefault("SPT_SETTINGS_FILE", str(Path(tempfile.mkdtemp(prefix="spt_test_")) / "settings.ini"))

# Ekransız (offscreen) Qt, sistem yazı tiplerini kendiliğinden bulamaz ve çok daha geniş bir yedek yazı tipine düşer:
# yerleşim ölçüleri (kart / panel genişlikleri) gerçek uygulamadakinden farklı çıkar. Windows'ta sistem yazı tiplerini
# gösteriyoruz; QApplication henüz oluşturulmadığı için toplama aşamasında ayarlamak yeterli.
_FONTS = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
if os.name == "nt" and _FONTS.is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", str(_FONTS))
