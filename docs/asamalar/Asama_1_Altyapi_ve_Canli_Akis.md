# Aşama 1 — Altyapı ve Canlı Veri Akışı

**Durum:** ✅ Tamamlandı (2026-10-10)

## Amaç
Ethernet üzerinden SPTF formatında gelen veriyi alıp arayüzde **canlı zaman grafiği** olarak göstermek;
ağ bağlantısı olmadan da önceden kaydedilmiş verinin aynı şekilde analiz edilebilmesi.

Cihaz henüz hazır değilken test için PC'de çalışan bir **simülatör** aynı formatta veri gönderir.

## Mimari karar: alıcı ayrı süreçte
Gigabit hızında (~115 MB/s, standart MTU'da saniyede ~80 bin datagram) Python'da arayüz ve alım aynı
süreçte olursa GIL paylaşımı yüzünden alım takılır. Bu yüzden:

```
 UDP / .sptr ──► Alıcı veya oynatıcı SÜRECİ ──► paylaşımlı bellek (akış başına halka tampon)
                 (StreamStore: doğrulama,          ▲
                  kayıp sayımı, kayıt)              │ kopyasız okuma, ~30 FPS
                        │ olay kuyruğu              │
                        └──────────────► Arayüz süreci (SourceManager → paneller, grafikler)
```

- Alıcı, sıcak yolda (tanımlı, interleaved, sıradaki DATA çerçevesi) numpy kullanmadan baytları doğrudan
  paylaşımlı belleğe kopyalar: paket başına ~4 µs.
- Kayıt oynatma aynı `StreamStore` katmanından geçer; canlı ve kayıttan analiz birebir aynıdır.
- Ekrana çizim, tampondan kopyasız min/max (tepe korumalı) seyreltmeyle piksel sayısı kadar noktaya indirilir.

## Adımlar

### 1.1 Arayüz iskeleti ve tema ✅
- [x] TerraFlux temalı ana pencere, 8 menü, üst çubuk, durum çubuğu
- [x] 3 görünüm: Canlı İzleme, Spektrogram, Filtre Tasarımcısı *(Spektrogram sonradan kullanıcı isteğiyle
      kaldırıldı; yerini Sinyal Üreteci aldı — Aşama 3C)*
- [x] 6 panel: Kaynaklar, Çalışma Alanı, Özellikler, Ölçümler, Paket İstatistikleri, Günlük
- [x] Merkezi komut kaydı, komut paleti (Ctrl+K), panel yerleşiminin hatırlanması
- [x] `run.py` başlatıcısı (çift tıklamayla açılır, kendini `.venv` ile yeniden başlatır)

### 1.2 SPTF protokol modülü ✅
- [x] `spt/protocol/format.py` — sabitler, çerçeve tipleri, 17 veri tipinin tablosu
- [x] `spt/protocol/parser.py` — doğrulama (magic, sürüm, uzunluk, CRC), DESCRIPTOR / DATA / MARKER çözme
- [x] `spt/protocol/builder.py` — çerçeve oluşturma, sıcak yol için bellek ayırmayan `DataFrameWriter`
- [x] `spt/protocol/recording.py` — `.sptr` okuma (mmap) / yazma
- [x] `tests/test_protocol.py` — doküman test vektörü bayt bayt eşleşiyor; 17 tip × 2 yerleşim gidiş-dönüş; bozuk çerçeveler

### 1.3 Akış modeli ve tampon ✅
- [x] `spt/core/ring.py` — süreçler arası paylaşımlı halka tampon (ham tipte, kilitsiz tek yazıcı)
- [x] `spt/core/stream_store.py` — DESCRIPTOR'dan akış, kayıp tespiti, `desc_rev` değişimi, istatistik
- [x] Kayıp: boşluk kaydı + grafikte çizgi kesilir; `first_sample == 0` → yeniden başlama; yinelenen çerçeve atılır
- [x] `spt/dsp/convert.py`, `spt/dsp/display.py` — int24 / I-Q çözme, fiziksel birim, tepe korumalı seyreltme

### 1.4 UDP alıcı ✅
- [x] `spt/sources/receiver.py` — ayrı süreç, 64 MB soket tamponu (ayarlanabilir), datagramda çoklu çerçeve
- [x] `spt/sources/manager.py` — süreç yönetimi, olayların Qt sinyallerine çevrilmesi
- [x] Port meşgul vb. hatalar Günlük paneline ve durum çubuğuna düşer

### 1.5 Simülatör ✅
- [x] `spt/sources/simulator.py` — komut satırı: `python -m spt.sources.simulator --help`
- [x] Sinyaller: sinüs, kare, üçgen, testere, chirp, gürültü, karışık; 17 veri tipinin hepsi, interleaved/planar
- [x] `.sptr` (kayıt zamanlamasıyla), WAV, CSV/TXT, NPY dosyalarını SPTF akışı olarak gönderme; döngü
- [x] Yapay paket kaybı (`--drop`), periyodik MARKER, CRC, datagram boyutu, hız testi modu
- [x] Araçlar ▸ Cihaz Simülatörü penceresi (arayüzden başlat/durdur, dinlemeyi otomatik açar)

### 1.6 Arayüze bağlama ✅
- [x] Kaynaklar paneli: Dinlemeyi Başlat/Durdur, gerçek akış ve kanal ağacı, kanal göster/gizle
- [x] Canlı İzleme: zaman grafiği (~30 FPS) + temel FFT spektrumu, imleç okuması, boş durum ekranı
- [x] Özellikler ▸ Görüntü: zaman penceresi, Y ekseni otomatik/sabit, ızgara, tepe korumalı seyreltme
- [x] Özellikler ▸ Spektrum: FFT boyutu, pencere, Kaiser β (ortalama, birim, log eksen → Aşama 2)
- [x] Çalıştır / Durdur (F7), Bağlantı Ayarları penceresi
- [x] Durum çubuğu ve Paket İstatistikleri gerçek değerlerle (Aktif / Kayıplı / Bekliyor)

### 1.7 Kayıt ve oynatma ✅
- [x] Kaydı Başlat (Ctrl+R): alıcı süreci gelen çerçeveleri `.sptr`'ye yazar
- [x] `spt/sources/replay.py` — oynatma süreci: 0.25×–10× / maksimum hız, duraklat, sar, döngü
- [x] Kayıt Aç… / Kaydı Oynat / Dosya ▸ İçe Aktar ▸ SPTF Kaydı
- [x] `samples_signal_data/` — 9 örnek kayıt (`tools/make_sample_recordings.py` ile tekrar üretilebilir)

### 1.8 Performans doğrulama ✅
- [x] `tools/udp_benchmark.py` — simülatör(ler) → alıcı, localhost
- [x] Çizim maliyeti ölçüldü; canlı grafikler 1 px yumuşatmasız çizgiye alındı (2 px yumuşatmalı çizgi yoğun veride ~100 kat yavaş)

## Ölçüm sonuçları (Windows 11, 8 çekirdek, Python 3.14, 8 s)

| Senaryo | Alım | Kayıp |
|---|---|---|
| 8 MB/s · 1472 B | 7.8 MB/s | 0 |
| 115 MB/s · 1472 B · 3 paralel gönderici (gigabit, standart MTU, ~77 bin datagram/s) | 113.3 MB/s | 0 / 115 M örnek |
| 115 MB/s · 8972 B (jumbo frame) | 113.5 MB/s | 0 |
| 115 MB/s · 65507 B (büyük datagram) | 113.5 MB/s | 0 |

Tek bir Python simülatörü standart MTU'da ~55 MB/s gönderebildiği için gigabit testinde üç gönderici paralel çalıştırıldı;
darboğaz göndericideydi, alıcı değil. Alıcının paket başına işleme maliyeti ~4 µs (soket hariç ~230 bin datagram/s).

Çizim (en kötü durum, 1 MS/s I/Q, tam yükseklikte zikzak): zaman grafiği ~25 ms, spektrum ~10 ms / kare.

## Kabul kriterleri

| # | Kriter | Sonuç |
|---|---|---|
| 1 | Simülatör → UDP → arayüz, 2 kanal int16 10 kHz kesintisiz | ✅ |
| 2 | Yapay kayıpta arayüzün kayıp sayısı simülatörünkiyle **tam** aynı | ✅ 3792 = 3792 örnek (%2 kayıp, 3 kanal int24 48 kHz) |
| 3 | Simülatör yeniden başlatılınca akış otomatik toparlanır | ✅ tek akış, 1 reset algılandı, kayıp 0 |
| 4 | Protokol testleri geçer, test vektörü bayt bayt eşleşir | ✅ 65 test |
| 5 | `.sptr` oynatma canlı akışla aynı görüntüyü verir | ✅ 59 808 örnek bayt bayt aynı |
| 6 | Localhost'ta ≥ 8 MB/s kayıpsız | ✅ 113 MB/s kayıpsız (gigabit) |

## Bu aşamada etkinleşen menü komutları
UDP Dinlemeyi Başlat · Bağlantı Ayarları · Çalıştır / Durdur · Kaydı Başlat · Kaydı Oynat ·
İçe Aktar ▸ SPTF Kaydı · Cihaz Simülatörü

## Notlar
- Simülatörün ilk ve son DATA çerçevesi yapay kayba uğramaz: alıcı, ilk çerçeveden önceki ve son çerçeveden sonraki
  kaybı protokol gereği göremez. Gerçek cihazda da bu iki uçtaki kayıp tespit edilemez.
- **Windows güvenlik duvarı:** Başka bir bilgisayardan / cihazdan veri alırken Python'a UDP 50505 gelen trafiği için
  izin verilmelidir (ilk dinlemede Windows sorar). Yalnızca bu bilgisayar içinde test için adres 127.0.0.1 seçilebilir.
- Ölçümler paneli, Spektrogram ve Filtre Tasarımcısı hâlâ örnek veri gösterir (DEMO etiketli) — Aşama 2 ve 3.

## Tamamlandıktan sonra eklenenler

### FFT doğrulaması
- `tools/verify_fft_samples.py` (testi: `tests/test_fft_samples.py`) — 9 örnek sinyaldeki bilinen 34 bileşen,
  arayüzle aynı yoldan (çerçeve → tampon → FFT) ölçülüp teoriyle karşılaştırılır. Sonuç: **34/34**,
  Flat-top ile genlik hatası ≤ 0.023 dB; notalar ve chirp frekansı yarım bin içinde.
- Kare/üçgen/testere için "doğru değer", tek periyodun ayrık Fourier dönüşümüyle (sızıntısız) hesaplanır.
- Hann'da iki bin arasına düşen tonlar en fazla 1.42 dB düşük okunur (pencere kaybı); genlik ölçümü için Flat-top.
- DC yalnızca 0. binden okunmalı: tek taraflı spektrumda DC'nin komşu binlere sızan kısmı ×2 ölçeklenir,
  Flat-top'ta 1–4. binler DC'yi ~6 dB yüksek gösterir (standart davranış).
- **Düzeltilen hata:** kayıp bölge (sıfırla dolu) FFT penceresine girerse genlik düşüyordu (08'de −2.7 dB).
  Spektrum artık kayıp içermeyen en son pencereden hesaplanır (`SharedRing.latest_clean_range`).
- **Düzeltilen hata:** test kare dalgasında `scipy.signal.square` kayan nokta yuvarlaması örneklerin %3.3'ünü
  ters işaretli üretiyordu (−40 dB sahte taban). Faz artık tamsayı aritmetiğiyle (`spt/dsp/waveforms.py`);
  taban −118 dB.

### MATLAB benzeri yakınlaştırma (zaman grafiği)
- Sol tıkla kutu çizerek yakınlaşma (araç çubuğunda büyüteç), tekerlekle iki eksende yakınlaşma,
  orta tuş veya el aracıyla kaydırma, çift tıkla sıfırlama ve canlı takibe dönüş.
- Yakınlaştırınca görünen aralığın verisi tampondan **tam çözünürlükte** yeniden okunur; yeterince yaklaşınca
  her örnek nokta olarak görünür (noktalar arası süre = 1/fs).
- Canlı akışta yakınlaştırma "inceleme" moduna geçirir (otomatik kaydırma durur, "Canlıya dön" düğmesi çıkar).
- Çalıştır/Durdur (F7) kapatıldığında görüntü o anki son örnekte dondurulur; yakınlaştırma bu veriyle çalışır.

### Tek / çift taraflı spektrum
- Spektrum kartındaki düğmeler veya Özellikler ▸ Spektrum ▸ Gösterim (ikisi senkron).
- Tek taraflı (0 … fs/2): gerçel sinüs A genliğinde. Çift taraflı (−fs/2 … +fs/2): gerçel sinüs ±f'de A/2
  (−6.02 dB) olarak iki lobda. Karmaşık (I/Q) sinyalde tek taraflı yalnızca pozitif yarıyı gösterir (katlama yapılmaz).
- Doğrulama: 40/40 kontrol (çift taraflı ±50 Hz −4.43 dB, beklenen −4.44 dB).

### Ayrı pencerede açma (çoklu ekran)
- Zaman, Spektrum, Spektrogram, dört filtre grafiği ve altı panelin hepsi bağımsız pencerede açılabilir:
  kart başlığındaki / panel başlığındaki ⧉ düğmesi veya Görünüm ▸ Ayrı Pencerede Aç.
- Pencere görev çubuğunda görünür, başka ekrana taşınabilir, büyütülebilir; F11 tam ekran, Esc çıkış.
  Ana penceredeki kısayollar (F5, F7, Ctrl+R…) ayrık pencerede de çalışır.
- Ana pencerede yerinde ince bir şerit kalır ("Pencereyi göster" / "Buraya geri al"); pencere kapatılınca içerik
  yerine döner. Canlı grafikler ayrık pencerede, ana pencerede başka görünüm açıkken de güncellenir.
- Hangi öğelerin ayrık olduğu ve pencere konumları hatırlanır; uygulama yeniden açılınca aynı ekranlara gelir.

## Sonraki aşamaya devreden işler
FFT ortalama / birim / logaritmik eksen, canlı spektrogram, Dondur ve Çalışma Alanı, imleçler, canlı ölçümler,
tetikleme, MARKER'ların grafikte gösterimi, Paket İnceleyici (Aşama 2).
