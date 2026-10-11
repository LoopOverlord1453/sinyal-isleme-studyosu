# Aşama 3C — Sinyal Üreteci (DAC)

**Durum:** ✅ Tamamlandı (2026-10-11) · Kullanıcı isteğiyle eklendi; Spektrogram görünümünün yerini aldı (Ctrl+3).

## Amaç
MCU'nun DAC'ı ile üretilecek sinyali bilgisayarda tasarlamak, DAC'ın gerçekte vereceği **nicemlenmiş** çıkışı ve
spektrumunu görmek, mühendislik ölçümlerini (THD, SNR, ENOB, görüntü frekansı, yeniden yapılandırma filtresi …)
almak ve hedef platform için **C kodu + kullanım kılavuzu** üretmek.

## Dalga şekilleri — `spt/dsp/siggen.py`
- **Hazır**: sinüs, kare (doluluk), üçgen (simetri), testere (yön), darbe, yamuk, DC, chirp (doğrusal / logaritmik),
  beyaz / pembe gürültü (tohumlu), çoklu ton, AM, FM, sönümlü sinüs, EKG (sentetik kalp atımı), merdiven, yarım /
  tam dalga doğrultulmuş sinüs, sinc, Gauss darbesi, üstel sönüm.
- **Kendi tasarımınız**:
  - **Formül**: `t` (s), `ph` (periyot içi faz 0…1), `f`, `fs`, `n` ve `sin`, `cos`, `exp`, `square(ph, d)`, `saw`,
    `tri`, `pulse`, `where`, `clip` … ile ifade; `ast` beyaz listesiyle güvenli (öznitelik erişimi, lambda, büyük üs
    vb. reddedilir, hata konumu gösterilir).
  - **Noktalar / eğri**: grafikte sürüklenebilir noktalar (çizgiye tıklayınca nokta eklenir, sağ tık siler);
    basamak / doğrusal / kübik (periyodik spline) ara değerleme.
  - **Harmonikler**: Fourier serisi (n, genlik, faz) tablosu.
  - **CSV örnek dizisi**: Türkçe (`;`, ondalık virgül) ve İngilizce CSV; "fs'te çal" ya da "bir periyot kabul et".

## DAC modeli ve tablo
- Çözünürlük (6–24 bit), Vref, çıkış aralığı, kodlama (işaretsiz / ofsetli ikili / ikiye tümleyen), hizalama
  (sağ / sol, ör. STM32 DHR12L), kelime genişliği; hazır ayarlar: STM32 12/8 bit, ESP32, Arduino Due / Zero /
  Uno R4, MCP4725, I2S 16 / 24 bit. LSB = aralık / 2^b; aralık dışı örnekler kırpılır ve **her zaman sayılır**.
  "DAC aralığına sığdır" genlik ve ofseti otomatik ayarlar.
- **Tam periyot (DMA dairesel)**: frekans hatasını en küçük yapan tablo boyu N (ve tablodaki periyot sayısı);
  gerçekleşen frekans ve ppm hatası gösterilir. **DDS**: 2^k tablo + 32 bit faz akümülatörü, çalışırken frekans
  değişimi; frekans çözünürlüğü fs / 2³².
- **Zamanlayıcı**: STM32 PSC / ARR (16 / 32 bit), gerçekleşen fs ve hata.

## Ekran (Ctrl+3)
- **DAC Çıkışı (zaman)**: ideal eğri (gri kesikli) ve DAC'ın sıfır dereceli tutma (basamak) çıkışı; tıklanan
  noktanın zamanı, değeri ve **kod seviyesi**; ▦ ile görünen aralığın istatistikleri.
- **Spektrum**: nicemlenmiş çıkışın dBFS spektrumu, harmonikler işaretli. Tam periyot tablosunda pencere gerekmez
  ve spektrum çizgi spektrumudur (yalnız f0'ın katları): çubuk olarak çizilir; DDS / tek atımlıkta flattop
  pencereli sürekli spektrum.
- **Ölçümler**: tablo / zamanlayıcı / seviyeler / kalite (THD, SFDR, SNR, SINAD, ENOB, nicemleme SNR'si) / zaman ve
  filtre (en büyük eğim, ilk görüntü frekansı, önerilen alçak geçiren kesim ve Butterworth derecesi, ZOH düşüşü) /
  MCU maliyeti.
- **Simülatörle Gönder**: nicemlenmiş çıkış simülatör üzerinden UDP akışı olarak (döngülü) gönderilir; Canlı
  İzleme'de diğer sinyaller gibi incelenir ve filtrelenir.

## C kodu — `spt/codegen/dac_c.py`
Platformlar: **STM32 HAL** (DAC + TIM6 TRGO + DMA dairesel, doğru hizalama sabitiyle `HAL_DAC_Start_DMA`, PSC / ARR
makroları), **genel ISR** (`dac_write()` kullanıcıda), **DDS** (çalışırken frekans, isteğe bağlı doğrusal ara
değerleme), **ESP32** (ESP-IDF v5.1+ `dac_continuous`), **Arduino** (`analogWrite` + `micros()`). Dosyalar:
`<ad>.h`, `<ad>.c` (flash'ta `static const` tablo), örnek ve **KULLANIM.md** (CubeMX adımları, frekans değiştirme,
çıkış filtresi, tampon / op-amp, bellek, CPU maliyeti, sınırlamalar). Stereo tablo, ikili DAC yazmacına tek DMA
aktarımı için `uint32_t` üyeli birleşimle 4 bayt hizalıdır (her derleyici / optimizasyonda; öznitelik gerekmez). Ad C
tanımlayıcısına çevrilirken çift / sondaki alt çizgi üretilmez ("Sine 1k!" → `Sine_1k`, `SINE_1K_…`).

Kod boyutu, makinedeki arm-none-eabi-gcc ve clang ile **derlenerek ölçülür** (ör. 1 kHz sinüs, 48 kHz, 12 bit,
STM32: GCC -Os kod 132 B + tablo 100 B; Clang 284 B toplam); ESP32 (Xtensa) ve IAR tahmindir. CPU yükü tahmindir
(DMA'da ≈ %0).

## Doğrulama (`tests/test_siggen.py`, 117 test)
- Nicemleme teorisi: tam ölçekli sinüste SNR ≈ 6.02·b + 1.76 dB (8 / 12 / 16 bit, ±0.1 dB), ENOB ≈ b.
- Kare dalga THD %48.34 (tüm harmonikler) / %43.83 (ilk 10) — teoriyle aynı; pembe gürültü eğimi −3.03 dB/oktav.
- Tam periyot tablo boyu kaba kuvvetle en iyi olduğu kanıtlandı; 84 MHz → 48 kHz: PSC 0 / ARR 1749 (tam).
- Üretilen C (genel ISR, 5 DAC biçimi + stereo, DDS ± ara değerleme, frekans değişimi sonrası) derlenip
  çalıştırıldı: çıktı Python modeliyle **örnek örnek aynı**; tüm hedefler `-std=c99 -Wall -Wextra -pedantic -Werror`.
- 25 kötü niyetli / geçersiz formül reddedildi.
- Arayüz testleri: nicemlenmiş çıkış LSB ızgarasında, formül hatası, sığdırma (kırpma yok), kod penceresi ve
  simülatör dosyası.
