# Aşama 2 — Spektral Analiz ve Ölçüm Araçları

**Durum:** ⏳ Bekliyor (Aşama 1'e bağlı; 2.4'ün bir kısmı Aşama 3'te yapıldı)

## Amaç
Canlı akışı frekans domeninde incelemek, sinyal üzerinde ölçüm yapmak ve debug için
cihazdan gelen olay işaretlerini (MARKER) grafikte göstermek. Bu aşama sonunda araç,
temel bir **yazılım osiloskobu + spektrum analizörü** gibi çalışır.

## Adımlar

### 2.1 Canlı FFT spektrumu
- [ ] Özellikler ▸ Spektrum ayarlarının bağlanması: FFT boyutu, pencere, Kaiser β, ortalama sayısı
- [ ] Pencereler: Hann, Hamming, Blackman, Blackman-Harris, Flat-top, Kaiser, Dikdörtgen
- [ ] Genlik düzeltmesi (pencere kazancı) ve birimler: dBV, dBFS, lineer
- [ ] Logaritmik frekans ekseni, sıfır ekleme (zero-padding)

### 2.2 Güç spektral yoğunluğu (Welch)
- [ ] Analiz ▸ Güç Spektral Yoğunluğu: segment boyu, örtüşme, birim V²/Hz ve dB

### 2.3 Canlı spektrogram — ❌ kaldırıldı
Kullanıcı isteğiyle Spektrogram görünümü uygulamadan kaldırıldı (2026-10-11); yerini **Sinyal Üreteci** aldı
(Ctrl+3, bkz. [Aşama 3C](Asama_3C_Sinyal_Ureteci.md)). Zamana bağlı spektral analiz gerekirse Canlı İzleme'deki
spektrum ve Dondur + İnceleme kullanılır.

### 2.4 Dondur ve Çalışma Alanı
- [x] Dondur (F6): tampondaki veriyi anlık görüntü olarak Çalışma Alanı'na kopyalar *(Aşama 3'te yapıldı)*
- [x] Çalışma Alanı'ndan seçilen sinyal İnceleme görünümünde (Ctrl+2) çevrimdışı incelenebilir *(Aşama 3'te yapıldı)*
- [x] Araçlar ▸ Sinyal Üreteci *(Aşama 3C'de DAC odaklı tam bir görünüm olarak yapıldı; üretilen sinyal
      "Simülatörle Gönder" ile akış olarak incelenebilir)*

### 2.5 İmleçler
- [ ] İki dikey imleç: Δt, 1/Δt; spektrumda Δf, Δgenlik
- [ ] Yatay imleçler: ΔV

### 2.6 Ölçümler paneli (canlı)
- [ ] RMS, tepe-tepe, ortalama, min/max, standart sapma, crest faktörü
- [ ] Frekans: sıfır geçişi ve FFT tepe noktası (parabolik interpolasyonla bin altı çözünürlük)

### 2.7 Tetikleme
- [ ] Modlar: serbest, otomatik, normal, tek
- [ ] Kaynak kanal, seviye, kenar (yükselen/düşen), ön tetik yüzdesi, histerezis

### 2.8 MARKER gösterimi
- [ ] Cihazdan gelen işaretler zaman grafiğinde dikey çizgi + etiket olarak görünür
- [ ] İşaret listesi; tıklanınca o ana gidilir

### 2.9 Dışa aktarma ve paket inceleme
- [ ] Grafiği PNG / SVG olarak kaydetme, panoya kopyalama (Ctrl+Shift+C)
- [ ] Araçlar ▸ Paket İnceleyici: son gelen çerçevelerin hex dökümü ve alan alan çözümü
      (cihaz tarafındaki SPTF uygulamasını doğrulamak için)

## Bu aşamada etkinleşecek menü komutları
FFT Spektrum · Güç Spektral Yoğunluğu · Ölçümler · İmleçler · Dondur · Sinyal Üreteci ·
Grafik (PNG/SVG) · Grafiği Kopyala · Paket İnceleyici

## Kabul kriterleri
1. Bilinen genlikte sinüs, Flat-top pencereyle **±0.1 dB** doğrulukla okunur.
2. Frekans ölçümü, interpolasyonla bin aralığının 1/10'undan daha iyi doğruluktadır.
3. Normal tetikleme modunda periyodik sinyal ekranda sabit durur.
4. Simülatörün gönderdiği MARKER'lar doğru örnek indeksinde görünür.
5. Dondurulan anlık görüntü canlı akış devam ederken değişmeden incelenebilir.

## Kapsam dışı
Filtreleme (Aşama 3) · korelasyon, Hilbert, ADC metrikleri (Aşama 4)

## Not
MARKER gösterimi ilk planda Aşama 4'teydi; debug için erken lazım olacağından Aşama 2'ye alındı.
