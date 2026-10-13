# wave — DAC sinyal üreteci kullanım kılavuzu

Signal Processing Toolbox tarafından üretildi. Hedef: **STM32 HAL — DAC + TIM6 TRGO + DMA dairesel**.

## Özet

| Özellik | Değer |
|---|---|
| Dalga | Çoklu ton, 1 kHz + 3 kHz + 6 kHz, genlik 1.80984 V, ofset 1.6496 V |
| Tablo | N = 250 örnek, M = 1 periyot |
| DAC güncelleme hızı | fs = 250000 Hz (her 4 µs bir örnek) |
| Gerçekleşen frekans | 1000 Hz (istenen 1000 Hz, hata +0 Hz / +0.000 ppm) |
| DAC | 12 bit, İşaretsiz (0 … 2^b−1), sağa hizalı, `uint16_t`; 0 … 3.3 V, 1 LSB = 0.8057 mV |
| Tablo belleği | 500 bayt (flash, `const`) |
| Kırpılan örnek | 0 |

Zamanlayıcı (STM32 tarzı, TIM_CLK = 90000000 Hz): **PSC = 0, ARR = 359** → fs = 250000 Hz (+0.000 ppm).

## Tablo nedir?

`wave_table`, DAC'a fs hızında sırayla ve döngüyle yazılacak **hazır DAC kelimeleridir** (kodlama ve sağa hizalama uygulanmış). Tabloda tam 1 periyot vardır; son örnekten ilk örneğe geçiş süreksizlik yaratmaz. M > 1 ise frekans M·fs/N ile istenen değere çok daha yakın tutturulur.

## Dosyalar

- `wave.h` — makrolar (N, fs, f, PSC/ARR, hizalama), tipler ve fonksiyon bildirimleri
- `wave.c` — tablo (`static const`, flash) ve hedefe özgü sürme kodu
- `wave_ornek.c` — kullanım örneği
- `KULLANIM.md` — bu kılavuz

Tabloyu RAM'e almak için (ör. DMA'nın flash'a erişemediği aileler) `-DWAVE_TABLE_IN_RAM` ile derleyin.

## Entegrasyon

### STM32CubeMX ayarları

1. **DAC**: OUT1 → "Connected to external pin only" (PA4 (OUT1)). Output Buffer: Enable. Trigger: **Timer 6 Trigger Out event**.
2. **TIM6**: Activated. Prescaler (PSC) = **0**, Counter Period (ARR) = **359**, auto-reload preload = Enable, Trigger Event Selection (TRGO) = **Update Event**. (TIM6 saati 90000000 Hz varsayıldı; Clock Configuration'daki "APB1 Timer clocks" farklıysa kodu doğru `timer_clock_hz` ile yeniden üretin.)
3. **DMA** (DAC → DMA Settings → Add, DAC_CH1): Direction = Memory To Peripheral, Mode = **Circular**, Increment Address = Memory, Data Width (Peripheral / Memory) = **Half Word / Half Word**.
4. Generate Code. `wave.h` ve `wave.c` dosyalarını Core/Inc ve Core/Src'ye ekleyin.

### Kod

`main.c` → USER CODE BEGIN 2: `wave_start(&hdac, &htim6);` (F4/F7: `hdac`, G4/L4/H7: `hdac1`).

- Hizalama sabiti: `DAC_ALIGN_12B_R` (12 bit, sağa hizalı; tablo elemanı `uint16_t`).
- `wave_start` PSC/ARR'yi `WAVE_TIM_PSC` / `WAVE_TIM_ARR` ile yazar, DMA'yı dairesel başlatır, TIM6'yı çalıştırır. Durdurmak için `wave_stop`.
- HAL başlığı varsayılan olarak `main.h`; değiştirmek için `-DWAVE_HAL_HEADER=\"stm32g4xx_hal.h\"`.

## Frekansı değiştirme

- **ARR ile (tablo aynı):** f_çıkış = M · TIM_CLK / ((PSC+1)(ARR+1) · N). `wave_arr_for_hz(hz)` uygun ARR'yi verir, `wave_set_arr(&htim6, arr)` uygular. ARR önyükleme (ARPE) açıkken değişiklik kesintisizdir; çözünürlük ARR adımıyla sınırlıdır (yüksek frekansta kaba).
- **Tabloyu yeniden üretmek:** farklı frekans için N/M yeniden seçilir (en hassas yol).
- **İnce ve hızlı ayar gerekiyorsa** `dds` hedefini kullanın (fs/2^32 çözünürlük, faz sürekli).

## Çıkış filtresi (yeniden yapılandırma)

DAC çıkışı basamaklıdır (sıfır dereceli tutma); fs'nin katları çevresinde görüntüler (imge) oluşur.

- Anlamlı içeriğin en yüksek frekansı: 6000 Hz (temel bileşene göre −40 dBc eşiği)
- İlk görüntü: 244000 Hz (temel bileşenin görüntüsü fs − f = 249000 Hz)
- Sıfır dereceli tutma düşüşü (içerik üst sınırında): -0.008 dB
- Önerilen alçak geçiren köşe frekansı: **3.826e+04 Hz** (içerik ile görüntünün geometrik ortası); ilk görüntüyü 40 dB bastırmak için **3. derece Butterworth** (tahmin).
- 1. derece RC örneği: R = 1 kΩ, C = 4.16 nF (en yakın standart değere yuvarlayın).
- 2. derece Sallen-Key (birim kazanç, Butterworth Q = 0.707): R1 = R2 = 10 kΩ, C2 (toprağa) = 0.2941 nF, C1 (geri besleme) = 0.5883 nF.

## Tamponlama / op-amp

- Ardışık örnekler arası en büyük sıçrama 91.04 mV → filtresiz çıkışta 0.02276 V/µs basamak eğimi; filtre sonrası gereken eğim hızı ≈ π · f_içerik · Vpp = 0.0449 V/µs. Op-amp eğim hızı (SR) bunun en az 2–3 katı, kazanç-bant genişliği çarpımı filtre köşesinin ≥ 10 katı olsun.
- DAC çıkışını doğrudan düşük empedanslı yüke bağlamayın; gerilim izleyici (buffer) kullanın.
- STM32 dahili DAC tamponu (Output Buffer = Enable) tipik olarak ≥ 5 kΩ / ≤ 50 pF yük sürer ve çıkış ≈ 0.2 V … VDDA − 0.2 V aralığında doğrusaldır (F4 veri sayfası; ailenize göre doğrulayın). Tam 0 V / VREF gerekiyorsa tamponu kapatıp harici rail-to-rail op-amp kullanın.

## Bellek ve CPU maliyeti

- Tablo: **500 bayt flash** (RAM'e alınırsa aynı miktar RAM).
- RAM: 0 bayt (durum 0 B).
- Kod boyutu: ≈ 160 bayt (**tahmin**; kesin değer için derlenmiş nesne dosyasının boyutuna bakın).
- CPU: DMA ile ≈ %0 (örnek başına yalnızca veri yolu çevrimleri).
- DMA dairesel: örnek başına CPU yükü ≈ 0 (yalnızca veri yolu çevrimleri). HAL DAC/TIM/DMA sürücüleri (birkaç kB) CubeMX projesinde zaten vardır, tahmine dahil değil.

## Ölçümler (nicemlenmiş çıkış)

| Ölçüm | Değer |
|---|---|
| Vmin / Vmax / Vpp | 0.4592 / 2.84 / 2.381 V |
| Ortalama (DC) / RMS / AC RMS | 1.65 / 1.805 / 0.7331 V |
| Tepe faktörü (AC) | 1.624 |
| THD (ilk 10 harmonik) | — (—) |
| SFDR | — |
| SNR / SINAD / ENOB | — / — / — |
| Nicemleme SNR'si | 69.6 dB |
| Analiz | tam periyot, pencere yok (kesin FFT) |

- Çok bileşenli sinyalde (multitone / AM / FM) tek ton ölçümleri (THD, SNR, SINAD, ENOB, SFDR) tanımsız; nicemleme kalitesi için quant_snr_db'ye bakın.

## Sınırlamalar

- Ölçümler ideal DAC modeline göredir (INL/DNL, kazanç/ofset hatası, yerleşme süresi ve glitch dahil değil). Gerçek çıkışı osiloskop / spektrum analizörüyle doğrulayın.
- Zamanlayıcı saati (TIM_CLK) ve kristal toleransı frekansı doğrudan etkiler (tipik ±20–50 ppm).
- STM32 DAC güncelleme hızı aileye bağlıdır (F4: ≈ 1 MS/s üst sınır, yerleşme ≈ 3–6 µs); veri sayfanıza bakın.
- Bazı ailelerde (ör. H7) DMA'nın erişebildiği bellek bölgeleri ve önbellek tutarlılığı dikkat ister; gerekirse `-DWAVE_TABLE_IN_RAM` ve uygun bölge.
