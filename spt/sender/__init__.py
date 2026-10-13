"""Gerçek Zamanlı Sinyal Test Uygulaması: SPTF akışlarını UDP ile Stüdyo'ya (ya da herhangi bir SPTF alıcısına)
örnekleme hızına göre zamanlanmış olarak gönderir.

- `spt.sender.signals`: kanal / akış / yapılandırma tanımları, sinyal üreteçleri, hazır senaryolar (Qt yok)
- `spt.sender.engine`: zamanlama ve gönderim motoru (ayrı süreç; Qt yok)
- `spt.sender.window`: PyQt6 arayüzü
- `python -m spt.sender`: arayüz ya da `--headless` ile yalnız motor
"""
