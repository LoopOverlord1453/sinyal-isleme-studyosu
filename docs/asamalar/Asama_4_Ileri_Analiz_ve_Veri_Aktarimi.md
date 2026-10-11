# Aşama 4 — İleri Analiz ve Veri İçe/Dışa Aktarma

**Durum:** ⏳ Bekliyor (Aşama 3'e bağlı)

## Amaç
Temel analizin ötesine geçen araçları eklemek ve SPTF dışındaki dosya formatlarıyla çalışabilmek.

## Adımlar

### 4.1 İçe / dışa aktarma
- [ ] CSV / TXT içe aktarma: sütun seçimi, ayırıcı, örnekleme frekansı girişi
- [ ] WAV ve NumPy (`.npy`) içe aktarma
- [ ] CSV ve WAV olarak dışa aktarma

### 4.2 Korelasyon
- [ ] Otokorelasyon ve çapraz korelasyon
- [ ] İki kanal arası gecikme tahmini (interpolasyonla örnek altı çözünürlük)

### 4.3 Hilbert dönüşümü
- [ ] Zarf, anlık faz, anlık frekans

### 4.4 Yeniden örnekleme
- [ ] Seyreltme (decimation), aradeğerleme, rasyonel oranlı (polyphase) yeniden örnekleme

### 4.5 Spektral ek araçlar
- [ ] Goertzel: tek frekansın genliği/fazı
- [ ] Koherans ve transfer fonksiyonu tahmini (giriş-çıkış kaydından sistem tanılama)

### 4.6 ADC kalite metrikleri
- [ ] SNR, THD, THD+N, SINAD, SFDR, ENOB
- [ ] Harmonik tablosu (seviye ve frekans)
- [ ] Kullanım: ADC'ye temiz sinüs verilip kaydedilir, metrikler otomatik hesaplanır

### 4.7 Ön işleme araçları
- [ ] Histogram
- [ ] Spike temizleme (Hampel, medyan)
- [ ] Doğrusal eğilim (trend) kaldırma, normalizasyon, kırpma

## Bu aşamada etkinleşecek menü komutları
İçe Aktar ▸ CSV / WAV / NumPy · Dışa Aktar ▸ CSV / WAV · Korelasyon ·
Zarf ve Anlık Frekans · Yeniden Örnekleme · ADC Kalite Metrikleri

## Kabul kriterleri
1. Bilinen gürültü ve harmonik içeren sentetik sinüste ADC metrikleri **±0.5 dB** içinde doğrudur.
2. Tamsayı örnek kadar kaydırılmış iki sinyalde gecikme tahmini tam doğrudur.
3. 1 milyon satırlık CSV dosyası arayüzü kilitlemeden içe aktarılır.

## Kapsam dışı
Wavelet analizi (Aşama 5)
