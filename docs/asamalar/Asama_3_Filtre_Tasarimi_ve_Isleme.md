# Aşama 3 — Filtre Tasarımı ve İşleme

**Durum:** ✅ Tamamlandı (2026-10-11)

## Amaç
FIR ve IIR filtre tasarlamak, cevaplarını incelemek, filtreyi canlı akışa ve dondurulmuş
sinyallere uygulamak, katsayıları dışa aktarmak. Filtre Tasarımcısı görünümü bu aşamada işlevsel hale gelir.

## Mimari
| Katman | Dosya | Görev |
|---|---|---|
| Tasarım motoru | `spt/dsp/filters.py` | Spesifikasyon (`FilterSpec`) → tasarım (`Design`: SOS / FIR / medyan), analiz, blok uygulama |
| İşlem zinciri | `spt/dsp/chain.py` | Sıralı aşamalar; her akışa kendi örnekleme hızıyla yeniden tasarlanır; `.sptchain` (JSON) |
| Katsayılar | `spt/dsp/coefficients.py` | float32 / Q31 / Q15 kuantizasyon, L∞ bölüm ölçekleme, CSV ve C (CMSIS-DSP) çıktısı |
| Canlı filtre | `spt/core/live_filter.py` | Paylaşımlı tampondaki yeni örnekleri durum taşıyarak filtreler; çıktı ham veriyle aynı indekste |
| Anlık görüntü | `spt/core/snapshot.py`, `spt/core/local_ring.py` | Dondur (kopya), çevrimdışı nedensel / sıfır fazlı filtre |
| Arayüz | `views/filter_designer_view.py`, `panels/chain_panel.py`, `panels/workspace_panel.py`, `dialogs/export_dialog.py` | Tasarımcı, İşlem Zinciri paneli, Çalışma Alanı, dışa aktarma |

## Adımlar

### 3.1 Tasarım motoru — `spt/dsp/filters.py` ✅
- [x] Filtre tanımı: aile, yöntem, bant tipi, fs, derece, kesim frekansları, Δf, ripple, bastırma
- [x] IIR: Butterworth, Chebyshev I, Chebyshev II, Eliptik, Bessel — SOS yapısında
- [x] FIR: pencere yöntemi (Kaiser, Hamming, Hann, Blackman, Dikdörtgen), Parks-McClellan, en küçük kareler
- [x] Bant tipleri: alçak geçiren, yüksek geçiren, bant geçiren, bant durduran
- [x] Spesifikasyondan otomatik derece: IIR'de `buttord / cheb1ord / cheb2ord / ellipord` (Bessel'de arama);
      FIR'de Kaiser tahmini + maskeyi karşılayan **en kısa** filtreyi bulan arama
- [x] Girdi doğrulama: Nyquist sınırı, f₁ < f₂, Δf'nin aralığa sığması, derece sınırı, kararsızlık / birim çembere
      çok yakın kutup uyarısı; hatalar Türkçe ve ne yapılacağını söyler

### 3.2 Filtre analizi ✅
- [x] Genlik cevabı + tasarım maskesi (yasak bölgeler taralı), faz cevabı, grup gecikmesi
- [x] Kutup-sıfır diyagramı (çakışan kökler sayıyla), kararlılık (en büyük |p|)
- [x] İmpuls ve basamak cevabı (kısa cevaplarda gövde/stem grafiği)
- [x] Dar özellikler için sıklaştırılmış frekans ızgarası (10 kHz'de 50 Hz notch da görünür), logaritmik frekans
      ekseni; FIR cevabı ve grup gecikmesi FFT ile (16 k katsayıda bile milisaniyeler)

### 3.3 Tasarımcı ekranının bağlanması ✅
- [x] Parametre değiştikçe grafikler 60 ms içinde yeniden çizilir
- [x] Yönteme göre ilgisiz alanlar gizlenir (ör. Butterworth'te ripple/bastırma, Kaiser dışında β);
      f₁/f₂'nin o yöntemdeki anlamı formun altında yazılır (−3 dB kesim, geçirme kenarı, durdurma kenarı…)
- [x] Bilgi kutusu: gerçek derece, bölüm sayısı, kararlılık, maske durumu, gecikme, uyarılar

### 3.4 Özel filtreler ✅
- [x] Notch (50 / 60 Hz şebeke, isteğe bağlı harmonikler — hepsi aynı bant genişliğinde), DC engelleyici
- [x] Hareketli ortalama, medyan (doğrusal olmayan), Savitzky-Golay

### 3.5 Filtreyi uygulama ✅
- [x] Çevrimdışı: Çalışma Alanı'ndaki dondurulmuş sinyale nedensel (`sosfilt`) veya sıfır fazlı (`sosfiltfilt`)
- [x] Canlı: durum (zi / geçmiş örnekler) taşıyan blok filtreleme; başlangıç durumu ilk örneğe göre kararlı hâl
      (DC'li sinyalde açılışta sıçrama yok); kayıpta çıktı kesilir, filtre kayıptan sonra temiz başlar
- [x] Ham ve filtrelenmiş sinyal üst üste (zaman ve spektrum), ya da yalnız filtrelenmiş
- [x] (Aşama 2.4'ten öne alındı) **Dondur (F6)** → Çalışma Alanı; **İnceleme** görünümü (Ctrl+4): anlık görüntüler
      canlı akış gibi çizilir, birden çok seçim aynı zaman ekseninde üst üste, spektrum görünen aralıktan

### 3.6 İşlem zinciri ✅
- [x] Sıralı bloklar (ör. DC kaldır → bant geçiren → notch); onay kutusuyla aç/kapa, sürükle-bırak veya oklarla sırala
- [x] Çift tık / Düzenle: aşama tasarımcıya yüklenir, "Adımı Güncelle" yerinde değiştirir
- [x] Zinciri kaydetme / açma (`.sptchain`) ve başka akışlara uygulama: aşamalar Hz cinsinden tutulur, her akışa
      kendi örnekleme hızıyla yeniden tasarlanır (Nyquist'i aşan aşama o akış için uyarıyla atlanır)

### 3.7 Katsayı dışa aktarma ✅
- [x] CSV ve C başlık dosyası: float64, float32, Q31, Q15
- [x] CMSIS-DSP düzeni: biquad `{b0, b1, b2, a1, a2}` a katsayıları **ters işaretli**, Q15'te `{b0, 0, b1, b2, a1, a2}`,
      ortak `postShift`; FIR katsayıları zamanda ters sırada, Q15 FIR çift sayıya tamamlanır; init/çağrı örneği
- [x] Sabit noktada L∞ bölüm ölçekleme (toplam kazanç korunur, ara düğümlerde taşma önlenir)
- [x] Kuantize cevap genlik grafiğinde kesikli çizgiyle; geçirme bandı sapması, en büyük sapma, durdurma tepesi
      ve kuantize filtrenin kararlılığı raporlanır

### 3.8 Geri al / yinele ✅
- [x] Tasarımcı ve işlem zinciri değişiklikleri için Ctrl+Z / Ctrl+Y (ardışık alan değişiklikleri tek adımda birleşir)

## Kabul kriterleri

| # | Kriter | Sonuç |
|---|---|---|
| 1 | Tasarımlar scipy referans sonuçlarıyla birebir aynıdır | ✅ 5 IIR × 4 bant, 5 pencere × 4 bant, Parks-McClellan ve en küçük kareler × 4 bant `array_equal`; otomatik derece `*ord` ile aynı ve maskeyi karşılıyor |
| 2 | Bloklar halinde filtreleme = tek seferde filtreleme | ✅ 9 filtre × gerçel/karmaşık, 1 … 4096 örneklik rastgele bloklarla en büyük fark < 1e-9 (bağıl); paylaşımlı tampon üzerinde ve arayüzde oynatılan kayıtta da aynı |
| 3 | 12. derece Eliptik SOS'ta kararlı | ✅ fs 48 kHz, kesim 100 Hz: SOS'ta en büyük \|p\| = 0.99994, 400 bin örnekte çıkış sonlu, kazanç hatası < %0.1. Aynı filtre tek polinomda \|p\| = 1.10 → çıkış NaN |
| 4 | Q15 katsayıların cevap sapması grafikte görülür | ✅ 8. derece Eliptik Q15: belirgin sapma (Q31'de < 1e-4 dB). 10 kHz'de 50 Hz notch Q15'te geçirme bandında ~54 dB bozuluyor — araç bunu kırmızı uyarıyla gösteriyor |

Uçtan uca (gerçek UDP): simülatör → alıcı → arayüz, 2 kHz ton 500 Hz'lik 6. derece Eliptik'ten canlı geçirildi,
durdurma bandında **> 70 dB** bastırıldı.

## Testler
`pytest` → **242 test** (Aşama 3 için 152 yeni): `test_filters.py` (129), `test_live_filter.py` (9),
`test_gui.py` (12, ekransız arayüz), `test_end_to_end.py` (2, gerçek UDP + kayıt/oynatma).

## Bu aşamada etkinleşen menü komutları
FIR Filtre · IIR Filtre · Özel Filtreler · İşlem Zinciri · Tasarla · Sinyale Uygula (Ctrl+Enter) ·
Katsayıları Dışa Aktar (Ctrl+E) · Geri Al · Yinele · Dondur (F6) · İnceleme (Ctrl+4)

## Notlar
- Sıfır fazlı uygulama genlik cevabının **karesini** uygular (dB cinsinden bastırma iki katı); yalnızca çevrimdışı.
- Parks-McClellan / en küçük kareler en fazla 4097 katsayı (süre ve bellek N² büyür); daha uzunu için Kaiser.
- Ayarlar `SPT_SETTINGS_FILE` ortam değişkeniyle ayrı bir INI dosyasına yönlendirilebilir (testler böyle çalışır).
- Kayıtlı panel yerleşiminin sürümü 2'ye çıktı (İşlem Zinciri paneli): ilk açılışta varsayılan yerleşim kurulur.

## Tamamlandıktan sonra eklenenler

### Özel (spesifik) filtreler
Filtre ▸ Özel Filtreler ve tasarımcının "Özel" ailesi — hepsi doğrusal olanlar tasarımcıda incelenir, zincire
eklenir ve katsayıları dışa aktarılır:

| Filtre | Kullanım | Doğrulama |
|---|---|---|
| Tarak (comb) | Şebeke + tüm harmonikleri tek filtrede (çentik) ya da yalnız harmonikleri geçirme (tepe) | 98 harmonikte < −100 dB, aralarda ±0.1 dB |
| Üstel ortalama (EMA) | En basit yumuşatma; α kesimden tam çözülür | Kesimde −3.0103 dB, y += α(x − y) ile birebir |
| Kalman (kararlı hâl) | Sabit hız / sabit seviye modeli; kazanç Riccati'den | Zamanla değişen Kalman ile oturduktan sonra birebir |
| Alfa-beta izleyici | Konum + hız izleme | Açık özyineleme ile birebir; kararsız α/β reddedilir |
| Gauss FIR | Aşmasız yumuşatma | DC kazancı 1, −3 dB = 0.1325·fs/σ |
| Türev alıcı | d/dt (birim/s), bant dışı gürültüyü büyütmez | Geçirme bandında 2πf ±%1, sinüste gerçek türev |
| Sızdıran integratör | ∫ (birim·s), ofsetle sürüklenmez | Köşenin üstünde 1/(2πf) ±%2 |
| Hilbert dönüştürücü | 90° faz, zarf / analitik sinyal | scipy analitik sinyalinin sanal kısmı ±%1 |
| Tüm geçiren | Faz / grup gecikmesi düzeltme | |H| = 1 (1e-9), f₀'da −180° |

### Ham / filtreli renk ayrımı
Filtre açıkken ham (filtresiz) sinyal **nötr açık gri (#b3aca5)**, filtrelenmiş sinyal **kanalın kendi renginde**
çizilir (zaman, spektrum, İnceleme; füzyonda ham ivmeölçer açısı da gri). Gri, 8 kanal renginin hepsinden normal
görüşte ve protan / deutan renk körlüğü simülasyonunda ayrışır (en kötü OKLab ΔE 9.1 ≥ 8); yüzeyde 8.4:1 kontrast.
Kanal renkleri filtre açılıp kapanınca değişmez; lejantta "· filtreli" ve "ham (filtresiz)" yazar.

### Her grafikte nokta inceleme ve ayrıntılı analiz
- **Tıkla:** en yakın eğrinin noktasına işaret + değerler. Zaman grafiğinde değer seyreltilmiş çizimden değil,
  tampondaki **gerçek örnekten** okunur (örnek numarasıyla). Spektrumda bin, frekans çözünürlüğü, dB ve lineer genlik;
  tasarımcıda genlik + faz + grup gecikmesi (imleç ve ipucu, ekran seyreltmesinden değil doğrudan tam cevaptan
  okunur; log f ekseninde de doğru); kutup-sıfırda |z|, açı, frekans, zaman sabiti.
  İşarete tekrar tıklamak kaldırır, sağ tık menüsü hepsini temizler.
- **▦ Analiz:** kartın yanında görünen aralığın istatistikleri (yakınlaştırınca güncellenir) — zaman: örnek, süre,
  min / max ve zamanı, tepe-tepe, DC, RMS, AC RMS, crest, baskın frekans (interpolasyonlu); spektrum: tepe frekansı
  (DC hariç, interpolasyonlu), seviye, gürültü tabanı, 2. tepe; tasarımcı: −3 dB noktası, basamak cevabında aşma /
  yükselme / oturma süresi, baskın kutbun ζ, Q, τ değerleri. Hücreler Ctrl+C ile kopyalanır.

### Sensör füzyonu
Ayrıntılar: [Aşama 3B — Sensör Füzyonu](Asama_3B_Sensor_Fuzyonu.md).

### Denetim ajan sistemi
Ayrıntılar: [docs/Ajan_Sistemi.md](../Ajan_Sistemi.md).
