# Aşama 3B — Sensör Füzyonu

**Durum:** ✅ Tamamlandı (2026-10-11) · Aşama 3'ün ardından kullanıcı isteğiyle eklendi; ikinci turda IMU dışı
sensörler, 3B görselleştirme ve MCU için C kodu üretimi eklendi.

## Amaç
Birden fazla sensörün verisini birleştirip (füzyon) tek sensörden daha doğru kestirim yapmak; algoritmaları aynı
veride yan yana çalıştırıp gerçek değere göre hatalarını ölçmek ve seçilen algoritmayı MCU'ya taşınabilir C kodu
olarak almak. Veri canlı akıştan (UDP / kayıt oynatma) ya da Çalışma Alanı'ndaki dondurulmuş anlık görüntüden gelir.

## Senaryolar — `spt/dsp/fusion_generic.py`, `spt/dsp/fusion_scenarios/`
| Senaryo | Girişler | Çıktılar | Algoritmalar |
|---|---|---|---|
| **Yönelim (IMU / MARG)** | jiroskop (rad/s, °/s, mdps), ivmeölçer, manyetometre (isteğe bağlı) | roll, pitch, yaw | Mahony, Madgwick, Kalman (eksen başına), EKF (çarpımsal), UKF, parçacık filtresi |
| **İrtifa: barometre + ivmeölçer** | basınç (Pa / hPa / kPa / mbar …) ya da barometrik irtifa, dikey ivme | irtifa, dikey hız | tamamlayıcı, KF (doğrusal), EKF (basınç ölçümlü, ISA), UKF, parçacık |
| **2B konum: GNSS + jiroskop + hız** | GNSS x/y (m) ya da enlem/boylam, sapma hızı, odometri hızı (isteğe bağlı) | x, y, başlık, hız | ölü hesap, EKF, UKF, parçacık |
| **Doğrusal çok sensörlü Kalman** | 1–6 sensör yuvası; her biri konum / hız / ivme türünde, kendi σ'sıyla | konum, hız, ivme | ağırlıklı ortalama, KF (sıralı skaler güncelleme), parçacık |

Ortak kurallar: girişler SI'ya çevrilir; NaN = "o örnekte ölçüm yok" (25 Hz barometre, 5 Hz GNSS aynı akışta
seyrek); blok blok işleme tek seferlik işlemle birebir aynı (parçacık filtreleri dahil, aynı tohumla); geçersiz
parametre / eksik giriş anlaşılır Türkçe hata. Her senaryonun gerçeği kesin bilinen analitik simülatörü vardır.

### Yönelim kuralları
- Gövde ve dünya çerçevesi sağ el, **z yukarı**; durağan ivmeölçer +g·ẑ_gövde ölçer.
- q = [w, x, y, z] **gövde → dünya**; Euler **ZYX** (R = Rz(yaw)·Ry(pitch)·Rx(roll)); yaw manyetik kuzeyden.
- Bu kuralda pozitif pitch **burun aşağı**dır (y ekseni sola); 3B görünüm ve yapay ufuk fiziksel duruşu çizer.
- EKF / UKF / parçacık filtresinde manyetometre yalnız başlığı düzeltir; başlık hatası dünya z ekseni etrafındaki
  dönme olarak ölçülür (EKF / UKF: h = R(q)[2, :]; parçacık: twist açısı) — pitch ±90°'ye yaklaşsa da çalışır.

## Motor — `spt/core/fusion_engine.py`
- Her algoritma **ayrı süreçte** (paralel). Parça boyu ölçülen maliyete göre ~0.2 s işlem sürecek şekilde
  ayarlanır; durdurma **engellemesizdir** (bayrak + kuyruklar beklenmeden kapanır, süreçler arka planda toplanır),
  işçiler ana süreç ölürse kendiliğinden çıkar.
- Referans ve ham (füzyonsuz) değerler gönderim anında motorun kendi tamponuna kopyalanır: halka tampon 10 s olsa da
  sonuçlar **10 dakikaya kadar** aynı eksende gösterilir (30 s / 60 s / Tümü pencereleri çalışır).
- Doğruluk metrikleri **işlenen tüm veriyi** kapsar (her (yeniden) başlangıçtan sonraki ilk 2 s yakınsama hariç).
- Yetişemeyen algoritmanın atladığı veri sayılır ve kullanıcıya gösterilir; kayıt bir kayıpla bitse de ilerleme
  %100'e ulaşır.

## Arayüz — Sensör Füzyonu (Ctrl+5)
- **Problem** seçimi (senaryo), kaynak, kanal eşleme (adlardan ve birimlerden otomatik: `ivme_z`, `basinc`, `gps_x`,
  `jiro_z`, `hiz`, `mesafe_*`, `ref_*`, "İvme X", "GyroX" …), senaryo ayarları (sensör gürültüleri, model), algoritma
  seçimi ve parametreleri.
- **Her çıktı kendi kartında** (başlık + birim), kartlar arası belirgin boşluk, ortak zaman ekseni; küçük ekranda
  kartlar ezilmez, alan kaydırılır. Lejant satıra sığmayanı alta kaydırır; tıklanınca seri tüm kartlarda gizlenir.
- **Renkler ölçülerek seçildi**: her algoritma ayrı bir renk ailesinden (kırmızı, sarı, yeşil, camgöbeği, mavi,
  eflatun), referans beyaz kesikli, ham gri — normal görüşte her çift ΔE ≥ 19, renk körlüğünde algoritmalar arası
  ΔE ≥ 10 (OKLab). Aynı ekranda hiçbir renk iki seride kullanılmaz; renk algoritmaya bağlıdır, sıraya değil.
- Gösterim: Kestirim · Hata (kestirim − referans) · (yönelimde) **Kuaterniyon bileşenleri** w, x, y, z.
- **3B Yönelim** (yönelim senaryosu): seçili algoritma dolu uçak gövdesiyle (gövde eksenleri x kırmızı, y yeşil,
  z mavi), referans beyaz kesikli hayalet, isteğe bağlı diğer algoritmalar; fareyle döndürme / yakınlaştırma.
  **Yapay ufuk** (roll, pitch) ve **pusula** (yaw) Euler açılarını gösterir; yanında q, Euler, eksen-açı ve referansa
  göre hata. Grafiklerde gezinirken imlecin zamanındaki yönelim gösterilir; zaman çubuğu ve oynat düğmesiyle
  geçmiş yeniden oynatılır. |pitch| > 85°'de gimbal uyarısı.
- **Yörünge (x–y)** (2B konum): algoritmaların yolları, referans ve GNSS ölçümleri (gri noktalar).
- **Doğruluk ve performans** tablosu: çıktı başına RMSE, en büyük |hata|, algoritmaların tanıları (jiroskop sapması,
  odometri ölçeği, σ'lar, ESS …), örnek başına süre, işlenen ve atlanan süre; satırlar ızgara ve dönüşümlü zeminle
  ayrılır, algoritma adı kendi renginde.
- Jiroskop birimi makullük denetimi: °/s veri rad/s seçilmişse uyarır.
- **Küçük ekran**: geniş alanda (genişlik/yükseklik ≥ 1.3, ör. 1366×768, 1600×960) 3B kartı ve tablo grafiklerin
  sağında üst üste durur, grafikler tam boy kalır; dar-uzun alanda altta. Pencere 1800 px'ten darsa füzyon
  görünümünde yan / alt paneller geçici gizlenir (durum çubuğunda söylenir, Görünüm menüsünden açılabilir;
  görünümden çıkınca geri gelir). Kaynak geçici kaybolursa (döngülü oynatma başa sardı) eşleme korunur.

## MCU için C kodu — "C Kodu Üret…"
Pencere MCU özelliklerini sorar: çekirdek (Cortex-M0+ / M3 / M4F / M7 / M33F, ESP32, RISC-V RV32IMAC / IMFC), saat,
derleyici (GCC, Clang / Arm Compiler 6 – Keil MDK, IAR), optimizasyon, float / double, çağrı hızı, flash / RAM,
(yönelimde) manyetometre. Üretilenler: ortak başlık, algoritma başına `.h` / `.c`, örnek ana döngü ve
**KULLANIM.md** (STM32CubeIDE / Keil / IAR'a ekleme, FPU bayrakları, sabit hızda çağırma, DWT ile dt, eksen
hizalama, kalibrasyon notları, iş parçacığı güvenliği).

- **Bellek derlenerek ölçülür**: makinedeki arm-none-eabi-gcc ve clang ile nesne boyutu (kod + sabitler), newlib-nano
  ile bağlanmış imajda libm dahil boyut, RAM = durum yapısının boyutu (malloc yok), yığın = en kötü çağrı yolu. IAR
  ve derleyicisi olmayan çekirdekler açıkça **tahmin** etiketlidir.
- **Maliyet**: adım başına işlem sayıları, üretilen C'nin C++ sayaç tipiyle derlenip çalıştırılmasıyla ölçülür;
  çevrim = işlem × çekirdek çevrim tablosu → µs / adım, CPU yükü (%), en yüksek çağrı hızı.
- **Yığın**: bağlanmış imajın disassembly'sinden çağrı grafiğiyle; her fonksiyon çerçevesi `-fstack-usage` ile aynıdır
  (Cortex-M0+'ın `ldr rN, [pc]` + `add sp, rN` büyük çerçeve kalıbı dahil; 5 çekirdek × 2 derleyici × 3 optimizasyon
  × 2 sayı türünde 10 000'i aşkın fonksiyonla karşılaştırıldı).
- **Sağlamlık**: `step()` dt ≤ 0 / NaN / sonsuz örneği atlar (0 döner, durum değişmez); parametreler Python'daki
  aralıklarla denetlenir; yönelim ve genel füzyon başlıkları aynı derleme biriminde hangi sırayla eklenirse eklensin
  derlenir, dosya adları çakışmaz (genel füzyonda `ornek_kullanim_<senaryo>.c`, `KULLANIM_<senaryo>.md`).
- Doğrulama (`tests/test_codegen_fusion.py`): double'da Python ile ~1e-14 rad, float'ta RMSE farkı < 0.001°;
  tüm Cortex-M profillerinde × gcc/clang × float/double uyarısız (`-Wall -Wextra -pedantic -Werror`).

Cortex-M4F, -Os, float (GCC kodu / libm dahil · RAM · 200 Hz'de CPU yükü): Mahony 1756 / 7476 B · 40 B · %0.10;
Madgwick 1870 / 7568 B · 24 B · %0.14; Kalman 1824 / 7872 B · 88 B · %0.20; EKF 3628 / 9704 B · 200 B · %0.53;
UKF 5326 / 11400 B · 1012 B · %3.0; parçacık (600) 3934 / 11208 B · 20 KB · %121 (bu hızda gerçek zamanlı değil:
parçacık sayısı azaltılmalı).

IMU dışı senaryoların C üreticisi `spt/codegen/fusion_generic_c.py` aynı altyapıyı (ölçüm, maliyet, kılavuz) kullanır;
NaN = "bu örnekte ölçüm yok" kuralı, otomatik p0 / ilk GNSS düzeltmesi ve çok hızlı seyrek sensör davranışı C'de de
aynıdır (`tests/test_codegen_fusion_generic.py`: double'da Python ile < 1e-6, eksik ölçümlü çalıştırmalar ve 10 Hz
GNSS dahil; parçacık filtreleri istatistiksel; tüm Cortex-M × gcc/clang uyarısız). 2B konumun başlatma geçmişi 160
düzeltme, en az 30 s / 159 aralıkla tutulur (her GNSS hızında 30 s; Python aynı kural). float'ta konum orijinden
20 km'ye kadar double kadar doğrudur, 70–90 km'de hatası ~2 katına çıkar: geniş alanda double önerilir (kılavuzda ve
kod penceresinde uyarı).

Cortex-M4F, -Os, float, GCC (kod · libm dahil · RAM · adım süresi @168 MHz · CPU yükü):

| Senaryo (çağrı hızı) | Algoritma | Kod (B) | libm dahil (B) | RAM (B) | µs / adım | CPU |
|---|---|---|---|---|---|---|
| İrtifa (200 Hz) | tamamlayıcı | 688 | 2900 | 92 | 0.70 | %0.014 |
| | KF | 1004 | 3216 | 104 | 1.48 | %0.030 |
| | EKF | 1102 | 3316 | 104 | 1.46 | %0.029 |
| | UKF | 1456 | 3728 | 104 | 2.04 | %0.041 |
| | parçacık (500) | 3012 | 6440 | 9200 | 1004 | %20 |
| 2B konum (100 Hz) | ölü hesap | 1180 | 6732 | 2712 | 1.87 | %0.019 |
| | EKF | 2184 | 7772 | 2864 | 6.14 | %0.061 |
| | UKF | 3828 | 9476 | 3616 | 55.0 | %0.55 |
| | parçacık | 4094 | 10956 | 18168 | 2072 | %21 |
| Çok sensörlü (100 Hz) | ağırlıklı ortalama | 552 | 744 | 124 | 1.12 | %0.011 |
| | KF | 1456 | 1648 | 160 | 2.28 | %0.023 |
| | parçacık | 3652 | 5236 | 7184 | 2458 | %25 |

2B konumda RAM'in çoğu başlatma için tutulan GNSS geçmişidir (`SPT_POS_FIX_HISTORY`, varsayılan 160 düzeltme).

## Örnek veri
| Dosya | Senaryo |
|---|---|
| `10_imu_9eksen_yonelim_referansli_200Hz.sptr` | Yönelim: jiroskop (°/s, sabit sapmalı), ivmeölçer, manyetometre + ref_roll / pitch / yaw |
| `11_irtifa_baro_25Hz_ivme_200Hz_referansli.sptr` | İrtifa: 25 Hz basınç (aradaki örnekler NaN) + 200 Hz dikey ivme + ref_irtifa / ref_dikey_hiz |
| `12_arac_gnss_5Hz_jiro_odometri_100Hz_referansli.sptr` | 2B konum: 5 Hz GNSS (60 s'de 3 s kesinti) + jiro_z + odometri + referanslar |
| `13_coklu_mesafe_sensoru_enkoder_100Hz_referansli.sptr` | Çok sensörlü: lidar / ultrasonik / kızılötesi mesafe + enkoder hızı + referanslar |

## Doğrulama
- `tests/test_fusion.py` (49 test): kurallar (Euler ↔ kuaterniyon, R, exp/log, eğim düzeltmeli başlık), Madgwick
  gradyanı, doğruluk, blok = tek sefer, motor (ayrı süreç = doğrudan çalıştırma, kayıpta NaN + temiz başlangıç),
  denetim regresyonları: manyetik bozulma roll / pitch'e sızmıyor (UKF / PF eskiden ~15°), pitch −89°'de başlık
  düzeltiliyor (eskiden 30°+), NaN / sıfır örnekte toparlanma, engellemesiz durdurma + süreç çıkışı, sonu kayıplı
  kayıtta %100 ilerleme, tüm çalışmayı kapsayan metrikler, maliyete göre parça boyu.
- `tests/test_fusion_generic.py` (68 test): her senaryo / algoritma için doğruluk sınırları ve teorinin öngördüğü
  sıralamalar (KF < tamamlayıcı, EKF/UKF/PF ≪ ölü hesap ve GNSS kesintisinde), EKF ↔ UKF uyumu, KF ↔ ders kitabı
  matris KF, blok = tek sefer, seyrek / eksik veri, birimler, otomatik eşleme, örnek kayıtlar.
- Simülatörde RMSE (yakınsama sonrası): irtifa KF 0.071 m (ham barometre 0.34 m); 2B konum EKF 0.65 m, GNSS
  kesintisinde 0.6 m (ölü hesap 37 m); çok sensörlü KF konum 0.0026 (ham ağırlıklı ortalama 0.030); yönelim EKF
  0.49° / 0.54° / 0.58°.
