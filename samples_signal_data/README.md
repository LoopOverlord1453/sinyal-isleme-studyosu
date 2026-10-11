# Örnek sinyal kayıtları

Bu klasördeki `.sptr` dosyaları, gerçek bir cihazın Ethernet üzerinden SPTF formatında göndereceği
çerçevelerin aynısını içerir (periyodik DESCRIPTOR, 1472 baytlık DATA çerçeveleri, MARKER olayları,
gerçekçi alınma zamanları). Ağ bağlantısı olmadan arayüzü denemek ve analiz yapmak için kullanılır.

## Nasıl açılır

- **Arayüzde:** Kaynaklar paneli ▸ **Kayıt Aç…** (veya Dosya ▸ İçe Aktar ▸ SPTF Kaydı, ya da Bağlantı ▸ Kaydı Oynat).
  Oynatma hızı (0.25× – 10×, Maksimum), döngü, duraklatma ve ileri/geri sarma Kaynaklar panelindedir.
- **Ağ üzerinden göndererek** (alıcıyı gerçek UDP yoluyla test etmek için):
  ```
  .venv\Scripts\python -m spt.sources.simulator --file samples_signal_data\01_sinus_50Hz_harmonik_2kanal_int16_10kHz.sptr --loop
  ```

## Kayıtlar

| Dosya | İçerik | Ne görmeli |
|---|---|---|
| `01_sinus_50Hz_harmonik_2kanal_int16_10kHz` | 2 kanal şebeke ölçümü: 50 Hz + 3. harmonik + 1.2 kHz / 3.1 kHz bileşen, 10 s | Spektrumda 50, 150, 1200, 3100 Hz tepeleri |
| `02_chirp_20Hz_8kHz_float32_20kHz` | 20 Hz → 8 kHz logaritmik tarama, 4 s'de bir tekrar, 8 s | Spektrumda yayılan enerji; tarama başında MARKER |
| `03_kare_ucgen_testere_3kanal_planar_int16_50kHz` | Kare 1 kHz, üçgen 500 Hz, testere 250 Hz; **planar** yerleşim, 4 s | Tek harmonikler (kare/üçgen), tüm harmonikler (testere) |
| `04_imu_6kanal_float32_1kHz` | İvme (g) + jiroskop (°/s), 8 s'de motor kalkışı, 22 s'de duruş, 30 s | 120/240 Hz titreşim; `motor_start` / `motor_stop` MARKER'ları |
| `05_ses_24bit_stereo_48kHz` | **int24** stereo, dört nota (440–880 Hz), sönümlü zarf, 6 s | Nota başlarında MARKER; 24 bit çözümleme |
| `06_iq_cint16_1MSps` | **Karmaşık (I/Q)** alıcı verisi, 1 MS/s: +125 kHz güçlü, −300 kHz zayıf ton, 1 s | İki taraflı spektrumda +125 kHz ve −300 kHz |
| `07_ekg_50Hz_parazitli_uint16_500Hz` | EKG benzeri sinyal, 50 Hz şebeke paraziti + taban kayması; **uint16 + ofset** ölçekleme, 20 s | Aşama 3'te notch ve yüksek geçiren filtre denemesi için |
| `08_kayipli_akis_ve_reset_int16_10kHz` | 100 Hz sinüs; %2 rastgele paket kaybı, 3.0 s'de 0.5 s toplu kayıp, 6 s'de cihaz reseti, 10 s | Paket İstatistikleri'nde kayıp sayısı, Günlük'te reset uyarısı |
| `09_coklu_akis_titresim_ve_sicaklik` | Aynı dosyada **iki akış**: 25.6 kHz rulman titreşimi (2 kanal) + 10 Hz sıcaklık, 10 s | İki akış birlikte; farklı örnekleme hızları |
| `10_imu_9eksen_yonelim_referansli_200Hz` | **Sensör füzyonu:** 9 eksen IMU — jiroskop (°/s, sabit sapmalı), ivmeölçer (m/s², doğrusal ivmeli), manyetometre (µT) + `ref_roll / ref_pitch / ref_yaw` gerçek yönelim (°), 60 s | Sensör Füzyonu görünümünde 6 algoritmanın kestirimi ve gerçeğe göre hata (RMSE) |
| `11_irtifa_baro_25Hz_ivme_200Hz_referansli` | **Genel füzyon — irtifa:** barometre basıncı (Pa, 25 Hz; aradaki örnekler NaN) + dikey ivme (m/s², 200 Hz, +g dahil, sabit sapmalı) + `ref_irtifa` / `ref_dikey_hiz`, 120 s | Sensör Füzyonu ▸ "İrtifa" senaryosu: tamamlayıcı / KF / EKF / UKF / parçacık filtresi; ham barometreye (≈0.34 m) göre ≈0.07 m RMSE |
| `12_arac_gnss_5Hz_jiro_odometri_100Hz_referansli` | **Genel füzyon — 2B konum:** GNSS `gps_x/gps_y` (yerel m, 5 Hz, σ 2.5 m, 60 s'de 3 s kesinti) + `jiro_z` (°/s, +0.4 °/s sapma) + `hiz` (odometri, %3 ölçek hatası) + `ref_x/ref_y/ref_baslik/ref_hiz`, 120 s | "2B konum" senaryosu: ölü hesabın kayması, EKF / UKF / PF ile < 1 m konum; kesintide ölü hesapla devam, jiroskop sapması ve odometri ölçeği kestirimi |
| `13_coklu_mesafe_sensoru_enkoder_100Hz_referansli` | **Genel füzyon — doğrusal çok sensörlü:** aynı mesafeyi ölçen lidar (σ 0.03 m, 100 Hz), ultrasonik (σ 0.10 m, 20 Hz), kızılötesi (σ 0.20 m, 50 Hz) + enkoder hızı (σ 0.02 m/s) + `ref_konum/ref_hiz`, 60 s | "Doğrusal çok sensörlü Kalman": σ'ları MARKER'dan girin; KF anlık ağırlıklı ortalamadan ≈10× daha doğru |

## Genel sensör füzyonu (11–13)

Sensör Füzyonu görünümü yalnızca IMU yönelimi değil, senaryo tabanlı genel füzyon da yapar: **İrtifa**
(barometre + ivmeölçer), **2B konum** (GNSS + jiroskop + odometri) ve **Doğrusal çok sensörlü Kalman**
(1–6 sensör: konum / hız / ivme). Kanallar adlarından ve birimlerinden otomatik eşlenir (`ivme_z`, `basinc`,
`gps_x`, `jiro_z`, `hiz`, `mesafe_*`, `ref_*` …). Seyrek sensörler (25 Hz barometre, 5 Hz GNSS) aynı akışta
ölçüm olmayan örneklerde NaN taşır. Her kaydın başındaki MARKER, senaryoyu ve simülasyondaki gürültü σ'larını
yazar. Yeniden üretmek için:

    .venv\Scripts\python tools\make_fusion_samples.py

## Yeniden üretme

Kayıtlar sabit tohumlu ve tekrarlanabilirdir:

```
.venv\Scripts\python tools\make_sample_recordings.py
```

Kendi kayıtlarınız: UDP dinlerken **Kaydı Başlat** (Ctrl+R) ile alınan `.sptr` dosyaları varsayılan olarak bu klasöre kaydedilir.
