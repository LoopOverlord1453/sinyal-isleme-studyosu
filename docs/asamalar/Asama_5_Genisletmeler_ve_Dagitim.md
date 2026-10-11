# Aşama 5 — Genişletmeler ve Dağıtım

**Durum:** ⏳ Bekliyor (Aşama 4'e bağlı)

## Amaç
Aracı günlük kullanıma hazır hale getirmek: TCP desteği, proje dosyaları, ayarlar,
ileri seviye analizler ve tek tıkla kurulabilir paket.

## Adımlar

### 5.1 TCP bağlantısı
- [ ] Arayüz TCP 50505'te sunucu olur; çerçeveler bayt akışında art arda gelir
- [ ] Senkron kaybında `"SPTF"` arama + uzunluk + CRC ile kurtarma
- [ ] Kaynaklar panelinde protokol seçimi (şu an TCP seçeneği kapalı)

### 5.2 Proje dosyası
- [ ] Yeni Proje / Proje Aç / Projeyi Kaydet
- [ ] İçerik: panel yerleşimi, bağlantı ayarları, filtreler, işlem zinciri, Çalışma Alanı sinyalleri

### 5.3 Tercihler penceresi
- [ ] Port ve adres, tampon süresi, ekran yenileme hızı, varsayılan birimler

### 5.4 Wavelet analizi (PyWavelets)
- [ ] CWT ile scalogram
- [ ] DWT ile gürültü giderme

### 5.5 Uyarlamalı filtreler
- [ ] LMS, NLMS, RLS (referans sinyalle gürültü giderme)
- [x] Kalman filtresi *(öne alındı: kararlı hâl Kalman özel filtre olarak Aşama 3'te; EKF / UKF / parçacık filtresi
      dahil sensör füzyonu Aşama 3B'de)*

### 5.6 Dağıtım ve dokümantasyon
- [ ] PyInstaller ile tek klasör / tek `.exe` paket (Python kurulu olmayan bilgisayarda çalışır)
- [ ] Inter yazı tipinin pakete eklenmesi
- [ ] Kullanım kılavuzu (Yardım ▸ Kullanım Kılavuzu, F1)

### 5.7 Opsiyonel fikirler
- [ ] Aynı anda birden fazla cihaz
- [ ] Blok diyagramlı (node) işlem zinciri editörü
- [ ] Python betik konsolu
- [ ] Açık tema

## Bu aşamada etkinleşecek menü komutları
Yeni Proje · Proje Aç · Projeyi Kaydet · Tercihler · Kullanım Kılavuzu

## Kabul kriterleri
1. TCP bağlantısı koptuğunda veya araya bozuk veri girdiğinde akış kendiliğinden toparlanır.
2. Kaydedilen proje açıldığında yerleşim, filtreler ve işlem zinciri aynen geri gelir.
3. Paketlenmiş `.exe`, Python kurulu olmayan temiz bir Windows'ta çalışır.
