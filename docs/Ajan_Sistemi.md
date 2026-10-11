# Denetim Ajan Sistemi

Arayüzün ve hesapların doğruluğunu birbirinden bağımsız uzman ajanlarla denetlemek için kurulan sistem.
Ajanlar **Claude Code** içinde çalışır; proje dosyalarını değiştirmezler, yalnızca okur, ölçer ve bulguları
**kanıtıyla** (çalıştırdıkları komut ve çıktısı) raporlarlar. Düzeltmeyi ana oturum yapar; ardından denetim
tekrarlanır. Bu, "yapan" ile "denetleyen"in ayrı olmasını sağlar.

## Ajanlar (`.claude/agents/`)

| Ajan | Sorumluluk |
|---|---|
| `dsp-dogrulayici` | Filtre tasarımı (IIR/FIR/özel), spektrum, işlem zinciri, canlı filtre, Q15/Q31 kuantizasyon ve CMSIS-DSP çıktısı — teoriye ve scipy'ye karşı bağımsız doğrulama |
| `fuzyon-dogrulayici` | Kuaterniyon/Euler kuralları, IMU simülasyonu, Mahony, Madgwick, Kalman, EKF, UKF, parçacık filtresi, genel senaryolar (irtifa, 2B konum, çok sensörlü KF), füzyon motoru, 3B görünüm |
| `arayuz-test-uzmani` | Her menü komutu, görünüm, panel, ayrı pencere, veri ipuçları, analiz tabloları — ekransız uçtan uca, değer kontrollü |
| `protokol-ag-denetcisi` | SPTF dokümanı ↔ kod birebirliği, UDP alıcı (gigabit, kayıp sayımı, bozuk çerçeve dayanıklılığı), kayıt / oynatma, simülatör |
| `ux-renk-denetcisi` | Renk ayrımı (renk körlüğü dahil ölçümle), farklı ekran boyutlarında düzen, Türkçe metin ve birim tutarlılığı |
| `kod-gozden-gecirici` | Hatalar, kenar durumları, süreç / paylaşımlı bellek sızıntıları, donmalar, performans |
| `c-kod-dogrulayici` | Üretilen MCU C kodu (füzyon, genel füzyon, DAC): derleme, Python ile sayısal eşdeğerlik, MCU güvenliği, bellek / maliyet raporları, KULLANIM.md |
| `sinyal-uretici-dogrulayici` | Sinyal Üreteci: dalga şekilleri, formül güvenliği, DAC nicemleme, tam periyot / DDS tabloları, THD / SNR / ENOB, PSC / ARR |

Ortak kurallar (her rol dosyasında yazılı): proje dosyaları değiştirilmez; betikler `%TEMP%\spt_denetim\<ajan>\`
altında; arayüz denemeleri `QT_QPA_PLATFORM=offscreen` ve geçici `SPT_SETTINGS_FILE` ile (kullanıcının kayıtlı
ayarlarına dokunulmaz); kullanıcının çalışan uygulamasına ve süreçlerine dokunulmaz; bulgu uydurulmaz.

## Çalıştırma

**Tam denetim (önerilen):** Claude Code'da
```
denetim iş akışını çalıştır
```
Kayıtlı iş akışı `.claude/workflows/denetim.js`:
1. **Denetim** — sekiz ajan paralel çalışır, her biri yapılandırılmış bulgu listesi döndürür.
2. **Doğrulama** — her bulgu iki bağımsız "şüpheci" ajan tarafından yeniden üretilmeye çalışılır.
3. **Rapor** — bulgular üç gruba ayrılır: *doğrulandı* (iki şüpheci de yeniden üretti), *tartışmalı* (biri),
   *reddedildi* (hiçbiri). Önem sırasıyla (KRİTİK → DÜŞÜK) döner.

Odaklı tur için: `denetim iş akışını "filtre tasarımcısı" odağıyla çalıştır` (args.focus).

**Tek ajan:** `dsp-dogrulayici ajanıyla türev alıcıyı denetle` gibi.

## İlk denetim turunun sonuçları

Her bulgu ana oturumda bağımsız olarak yeniden üretildi, düzeltildi ve bir regresyon testiyle kilitlendi.

| Alan | Doğrulanan bulgu | Düzeltme |
|---|---|---|
| DSP | Tarak filtresi büyük fs/f₀'da (N = 4096) dakikalarca tasarlanıyordu | Kökler kapalı biçimde, O(N log N); scipy ile fark ≤ 1.6e-10 |
| DSP | L∞ bölüm ölçeklemesi dar bantlı tasarımda tepeyi ızgarada kaçırıyordu | Kutup yarıçapına göre sık ızgara + yerel ince arama; her ara düğümde tepe tam 1 |
| DSP | Kalman (sabit hız) çok küçük / büyük mutlak gürültü değerlerinde kötü koşullu | Riccati λ²'ye normalize; kazanç yalnız σ oranına bağlı, aşırı oranda anlaşılır hata |
| DSP | EMA kesimi çok düşük fc/fs'de sayısal olarak sapıyordu | Kararlı α = √(e² + 2e) − e; her oranda tam −3.0103 dB |
| DSP | remez / scipy istisnaları kullanıcıya ham hata olarak çıkabiliyordu | Tümü anlaşılır `FilterError`; türev alıcı / Hilbert parametre denetimi |
| DSP | Türev alıcı, Hilbert, integratör, tüm geçiren "sıfır fazlı" uygulanabiliyordu (|H|² amacı bozar) | Sıfır fazlıda reddedilir; filtreden kısa bölüt NaN |
| DSP | float akıştaki tek NaN / inf örnek IIR durumunu kalıcı bozuyordu | Sonlu olmayan örnekler kayıp gibi: çıktı NaN, filtre sonra temiz yeniden başlar |
| DSP | Sabit noktalı FIR'de |b| ≥ 1 (ör. birim/s türev alıcı) katsayılar kırpılıyordu | IIR'deki gibi `postShift`; C başlığında `arm_shift_q15/q31` kullanım notu |
| DSP | Kayıp kaydı (256) taşınca eski kayıp bölgeleri veri sanılabiliyordu | 4096 kayıt; taşmada en eski hatırlanan kayıptan öncesi geçersiz |
| Protokol | Sıfırlama çerçevesi kaybolursa akış kilitleniyordu | 4 ardışık geri çerçeve = yeniden başlama |
| Protokol | Aşırı `first_sample` / örnekleme hızı belleği taşırabiliyordu | Sınırlar (dokümana da yazıldı), bozuk çerçeve sayılır |
| Protokol | Bilinmeyen çerçeve türü / sondaki sıfır dolgu datagramı bozuyordu | Atlanır, sayılır |
| Protokol | Kayıt ortadan başlarsa oynatmada tanımlayıcı yoktu | Kayıt başına son tanımlayıcılar yazılır; okuyucu bozuk kayıtta yeniden senkronlanır |
| Arayüz | Filtreli spektrum bazı yenilemelerde boş kalıyordu (uç indeks yarışı) | Yenileme başına sabitlenen uç, önceki eğri korunur |
| Arayüz | 8'den fazla kanalda renkler tekrar ediyordu | Tekrarda kesikli çizgi; ham gri, görünür renklere göre seçilir |
| Araçlar | Simülatör, çıktısı boruya / dosyaya yönlendirilince (arayüzdeki Simülatör penceresi dahil) "→" karakterinde çöküyordu | Araçlar çıktıyı UTF-8 yazar; pencere alt süreci UTF-8 ortamıyla başlatır |
| Araçlar | `--duration` tam çerçeveye aşağı yuvarlanıyordu (1 kHz, 1 s → 712 örnek) | Kısa son çerçeve; tam 1000 örnek |

### Füzyon, arayüz ve kod gözden geçirme denetçilerinin bulguları

| Alan | Doğrulanan bulgu | Düzeltme |
|---|---|---|
| Füzyon | UKF ve parçacık filtresi başlığı sigma / parçacık ZYX yaw'ıyla ölçüyordu: manyetik bozulma roll / pitch'e ~15° sızıyor, pitch −89°'de başlık düzeltilmiyordu (30°+) | Başlık dünya z etrafındaki dönme olarak (h = R(q)[2,:] / twist); bozulmada eğim 0.03° (UKF), 0.9° (PF); −89°'de başlık 0.04° |
| Füzyon | EKF başlık güncellemesinde küçük eğim sızıntısı | Kazanç h'ye izdüşürüldü, Joseph biçimi |
| Füzyon | Tek NaN jiroskop örneği tüm algoritmaları kalıcı NaN yapıyordu; Kalman sıfır ivme / mag'i ölçüm sanıyordu | Geçersiz jiroskop = kayıp (yeniden başlatma); geçersiz / sıfır ivme-mag o ölçümü atlar |
| Füzyon | Kalman'ın "jiro ofseti" aslında Euler hızı sapması | Tabloda "(Euler hızı)" olarak etiketlenir; Madgwick "— (kestirmez)" |
| Füzyon / Kod | Füzyon durdurulunca uygulama süreci kapanmıyordu (KRİTİK); durdurma arayüzü ~3 s donduruyordu; ana süreç ölünce işçiler yetim kalıyordu | Engellemesiz durdurma, kuyruklar beklemeden kapanır, maliyete göre ~0.2 s'lik parçalar, işçi ana süreci denetler |
| Füzyon | Kayıt kayıpla bitince ilerleme %75'te kalıyordu; RMSE yalnız son 10 s'yi kapsıyordu; 30 / 60 s pencereleri etkisizdi; ağır algoritmalar sessizce veri atlıyordu | Sondaki kayıp kapatılır; metrikler tüm çalışmayı kapsar; motor 10 dk geçmiş tutar; atlanan süre tabloda ve durumda |
| Kod | L∞ bölüm ölçekleme bölüm sayısının karesiyle büyüyordu (önceki turun gerilemesi) | Birikimli cevap O(K); çok bölümde parabolik tepe ara değerlemesi |
| Kod / Arayüz | Tarak filtresi tasarımcıyı dakikalarca donduruyordu (analiz, grup gecikmesi, Q15) | Tarakta kapalı biçim cevap ve grup gecikmesi (0.02 s), vektörel grup gecikmesi, görünüm işlemlerinde önbellek, kuantizasyon önbelleği |
| Kod | Zincir değişince 4M örneğe kadar tampon tek seferde filtreleniyordu | Yeni iz görünen pencere kadar geriden başlar, kare başına 2¹⁹ örnek bütçe |
| Kod | Düzenlenen adım indeksle tutuluyordu: başka adım silinince / geri alınınca yanlış adımın üzerine yazılıyordu | Adımlara oturum içi kimlik (uid); geri almada düzenleme bırakılır |
| Kod | Analiz tablosu büyük aralıkta saniyeler sürüyordu (asal uzunlukta FFT 4–8 s) ve canlıda saniyede 4 kez çalışıyordu | Hızlı FFT uzunluğu, 1M örnek sınırı, canlıda en çok saniyede bir |
| Kod | Maksimum hız + döngüde akış arayüzde hiç görünmüyordu | Döngüler arası en az 250 ms |
| Kod | Dondur / çevrimdışı filtrede bellek sınırsızdı | ≤ 24 bit girdide sonuç float32; bellek yetmezse anlaşılır mesaj |
| Kod | Eski veri ipuçları, kaldırılan akışın uyarıları, kapalı dışa aktarma penceresinin çalışması, canlıda dev medyan | Hepsi düzeltildi (ipuçları temizlenir, uyarılar budanır, pencere kapalıyken işlemez, medyan ≤ 4097) |
| Arayüz | Seyreltilmiş grafikte tıklanan tepe başka örneği gösteriyordu; imleç okumaları seyreltilmiş eğriden | Min / max gerçek örnek indeksinde; okumalar tampondaki gerçek örnekten |
| Arayüz | Log f modunda tasarımcı imleci her yerde −300 dB / −359.9° okuyordu; hareketli ortalamada −19.6 dB (gerçek −34.8) | Okuma / ipucu doğrudan tam frekans cevabından |
| Arayüz | Sıfır DC kazançlı filtrede basamak tablosu anlamsız (%1.6e8 aşma); −3 dB satırı tutarsız; kısa pencerede yanıltıcı baskın frekans | "—" ve açıklama; tüm −3 dB geçişleri; < 2 periyotta "—" |
| Arayüz | 1366×768'de kart başlıkları kesik, grafikler küçük; yatay bölücüde ayrılan kartın yer tutucusu kesik | Başlık hiç kesilmez (isteğe bağlı öğeler gizlenir), kısa ekranda alt paneller kapalı başlar, yer tutucu dar sütunda yalnız ikon |
| Arayüz | Zaman penceresi / döngü kaydedilmiyordu; uygulanmamış kısayollar geri bildirim vermiyordu | Kalıcı; kısayol durum çubuğunda neden yok olduğunu söyler |
| Araçlar | Simülatör, çıktısı boruya yönlendirilince çöküyordu (arayüzdeki Simülatör penceresi dahil); `--duration` eksik gönderiyordu | UTF-8 çıktı; kısa son çerçeve |

### Sinyal üreteci doğrulayıcısının bulguları (üçüncü tur)

| Doğrulanan bulgu | Düzeltme |
|---|---|
| Türkçe tek sütunlu CSV ("0,5") 5.0 okunuyor, işaret kayboluyordu | Her satırda tek virgül ve hiç nokta yoksa ondalık virgül; boşluk ayrılmış dosyalar da |
| Arduino hedefinde yeniden hesap çöküyor, ekran bayat kalıyordu | CPU yükü yoksa "donanımda ölçülmeli"; hata yakalanır |
| DDS'te THD / SINAD / ENOB Blackman-Harris yan loblarında ~86 dB'de takılıyordu | Kaiser β = 38: 16 bitte 95.8 dB (kesin 96.0) |
| "Simülatörle Gönder" DDS ve tek atımlık sinyallerde döngüde kesiliyordu | Faz akümülatörü periyodunun katı / tablonun tam tekrarı |
| "DAC aralığına sığdır" üstel sönüm ve Gauss darbesinde aralığı doldurmuyordu | Gerçek aralık sayısal örneklemeyle |
| Ölçüm sürerken pencere kapanınca süreç anormal çıkıyordu | Ölçüm daemon iş parçacığında, nesil numarasıyla: kapanış beklemez, geç gelen sonuç yok sayılır (füzyon kod penceresi de; bkz. arayüz 2. tur) |
| Küçük: "<" metni yutuyordu, DDS'te formül `fs`'i, ipucunda C kelimesi, DDS spektrum genliği, formül hata konumu, uzun toplamlar, float32 simülatör dosyası | Hepsi düzeltildi |

### Arayüz test uzmanının bulguları (ikinci tur: füzyon, sinyal üreteci, kod pencereleri)

Ajan gerçek fare olayları ve üç ekran boyutuyla (1366×768, 1600×960, 1920×1080) denedi. Her bulgu ana oturumda
yeniden üretildi, düzeltildi ve `tests/test_gui.py` / `tests/test_siggen.py` içinde regresyon testine bağlandı.

| Önem | Doğrulanan bulgu | Düzeltme |
|---|---|---|
| YÜKSEK | Döngülü oynatma başa sarınca çalışan füzyon yeniden başlıyor, ama eşleme kutuları hep "0: jiro_x" gösteriyordu: "Yeniden çalıştır" sessizce çöp sonuç veriyordu | Kaynak geçici kaybolunca kutulara dokunulmaz, başka akışa atlanmaz; geri gelince kullanıcının eşlemesi kutulara da uygulanır |
| YÜKSEK | 1366×768'de Füzyon'da çıktı kartlarına 64 px kalıyor, 3B kartı ve tablo kesiliyordu; Sinyal Üreteci'nde "Değer" sütunu görünmüyordu (doğrularken Filtre Tasarımcısı'nın da sıkıştığı görüldü: 814 px isteyip 656 px alıyordu) | Dar pencerede (< 1800 px) bu üç görünümde ilgisiz paneller geçici gizlenir (tasarımcıda İşlem Zinciri kalır) (durum çubuğunda söylenir; menüden açılan panel o görünümde bir daha gizlenmez, görünümden çıkınca hepsi geri gelir). Geniş alanda 3B + tablo grafiklerin sağına geçer (grafikler tam boy). En küçük genişlikler düşürüldü |
| ORTA | Ölçüm sürerken kod penceresinde MCU değiştirip yeniden üretmek arayüzü 7–12 s donduruyor, sonra kısa süre eski MCU'nun sonucunu gösteriyordu | Ölçüm daemon iş parçacığında; her ölçüme nesil numarası, eskisinin sonucu yok sayılır; hiçbir yerde beklenmez |
| ORTA | "Canlı akışta yeni örnekleri işle" kapalıyken Çalıştır hiçbir şey işlemiyordu (%0'da kalıyordu) | Bitiş, Çalıştır anındaki son örnekte dondurulur; kutu yeniden açılınca işleme kaldığı yerden sürer |
| ORTA | Füzyon durmuşken kayıt başa sarınca kullanıcının eşleme düzenlemeleri otomatik eşlemeyle eziliyordu | Kaynak başına eşleme belleği (kanal adları aynıysa korunur); "Otomatik eşle" yeniden öneri ister |
| ORTA | Yörüngede önceki çalıştırmanın eğrileri kalıyordu | Çalıştır'da yörünge eğrileri temizlenir |
| ORTA | Ayar panelleri yatayda kırpılıyordu (füzyon 345–363 px içerik / 286–316 px alan) | Açılır listeler dar yerde daralır (tam metin ipucunda), form satırları gerekirse alta kayar, uzun bölüm başlıkları "…" ile kısalır, algoritma onay kutuları akış düzeninde |
| ORTA | "DAC Çıkışı (zaman)" başlığının üstüne simgeler biniyordu; kod penceresi 768 px ekrana sığmıyordu, açıklama metni kesikti | Görünüm artık sığıyor (başlık sıkışmıyor); pencere boyu ekrana göre ve MCU ayar paneli kaydırılabilir (en küçük yükseklik ~800 px'ten 210 px'e indi); metin satır kaydırmalı |
| DÜŞÜK–ORTA | MCU değişince Flash / RAM 1024 / 192 KB'de kalıyordu (Flash % / RAM % 5–8 kat düşük) | Profilin tipik Flash / RAM değeri uygulanır |
| DÜŞÜK | Lejanttan gizlenen algoritma yörüngede görünmeye devam ediyordu | Lejant yörüngeye de uygulanır |
| DÜŞÜK | Noktalar editöründe çizgiyi sürüklemek tüm noktaları kaydırıyor, faz 1'i aşanlar kayboluyordu | Eğri bütün olarak ötelenemez; yalnız noktalar sürüklenir |
| DÜŞÜK | Harmonik tablosunda sayı olmayan hücre satırı sessizce siliyordu | Hücre kırmızı işaretlenir, durum satırında "n. satır sayı değil" uyarısı |
| DÜŞÜK | Basamağın sağ yarısına tıklanınca sonraki örnek gösteriliyordu; "İdeal" sütununun zamanları 0'dan sayılıyordu | Basamak sağ ucu kendi örneğinde; iki sütun aynı başlangıç zamanıyla |
| DÜŞÜK | Kullanıcıya iç API adları gösteriliyordu (`make_table`, `DacSpec`, `load_samples_csv`, `max_len`) | Tüm iletiler arayüzdeki alan / seçenek adlarıyla ("Tablo kipi", "En uzun tablo", hazır ayar adları) |

Bu turda ayrıca: EKG gibi uzun periyotlu dalgada "En uzun tablo" yetmezse bir periyodun gerektirdiği boya
kendiliğinden çıkarılır; tam periyot tablosunun spektrumu çizgi spektrumu olarak (çubuk) çizilir (kutuları çizgiyle
birleştirmek harmonikler arasında enerji varmış gibi üçgenler gösteriyordu). Ekransız testler artık Windows sistem yazı
tiplerini kullanır (`tests/conftest.py`): yerleşim ölçüleri gerçek uygulamadakiyle aynı.

### C kodu doğrulayıcısının bulguları (füzyon, genel füzyon, DAC üreticileri)

Ajan 1 900'ü aşkın derleme (`-std=c99 -Wall -Wextra -pedantic -Werror`, 5 Cortex-M × gcc/clang × float/double ×
-O0…-O3, C++11/17) yaptı: hepsi uyarısız. Python ile double eşdeğerliği (≤ 1.5e-14 rad yönelim, ≤ 2e-11 genel), 10⁶
adımlık uzun çalışma, flash / RAM / libm ölçümlerinin araç zinciriyle birebirliği ve DDS / PSC-ARR doğruluğu
doğrulandı. Bulunanlar ana oturumda yeniden üretildi, düzeltildi ve testlere bağlandı:

| Önem | Doğrulanan bulgu | Düzeltme |
|---|---|---|
| YÜKSEK | Cortex-M0+'ta yığın olduğundan çok düşük ve "kesin" raporlanıyordu (EKF update gcc -O0: 984 B yerine 0; clang -O0: 1832 B yerine 20). ARMv6-M büyük çerçeveyi `ldr rN, [pc]` + `add sp, rN` ile ayırıyor; ayrıca fonksiyon içi uzak dallanma `bl <kendi+ofset>` özyineleme sanılıyordu | Çözümleyici sabit havuzunu okur, yazmaç sabitlerini (`movs/lsls/negs/adds`) izler, bilinmeyen komutla ezilen yazmacı unutur; fonksiyon içi `bl` çağrı sayılmaz. Yönelimde 360, genel füzyonda 10 227 çerçeve `-fstack-usage` ile birebir, hepsi kesin. Ölçüm önbellekleri çözümleyici sürümüyle anahtarlandı |
| YÜKSEK | STM32 stereo DMA tablosu (12 bitte Word, 8 bitte Half Word okunur) gcc -Os / clang'da 2 / 1 bayt hizaya düşebiliyordu: kanallar kayar | Stereo tablo `uint32_t` üyeli C99 birleşimi içinde: her derleyicide 4 bayt hiza (gcc ve clang, -O0 / -Os / -O2'de `2**2` doğrulandı); API değişmedi |
| ORTA | `step()` dt'yi denetlemiyordu: tek bir dt = 0 / < 0 / NaN / sonsuz parçacık filtrelerini (NaN / sonsuzda tümünü) kalıcı NaN yapıyor, yine 1 dönüyordu | dt ≤ 0 ya da sonlu değilse örnek atlanır, 0 döner, durum değişmez: sonraki çıktı o örnek hiç gelmemiş gibi bit düzeyinde aynı (6 yönelim + 12 genel algoritma × 4 kötü dt). Korumasız kodda testin 24/24 durumda başarısız olduğu görüldü |
| ORTA | 2B konumun başlatma geçmişi 160 düzeltmeyle sınırlıydı (10 Hz GNSS'te 16 s): C 29.8 s'de, Python 19 s'de başlıyordu; belgedeki "fark yalnız çok yavaş başlangıçlarda" ifadesi yanlıştı | Python ve C aynı kural: düzeltmeler en az 30 s / 159 aralıkla saklanır (≤ 5.3 Hz'de hepsi, davranış değişmedi), en çok 160: her GNSS hızında 30 s. 10 Hz / 200 m örneği testlere eklendi: ikisi de 19.0 s'de başlıyor, fark 1e-13 |
| ORTA | float'ta 2B konum orijinden uzaklaştıkça bozuluyor (70–90 km'de konum RMSE ~2×, hız ~2.4×); belgedeki "Python'un %5'i içinde" bu mesafede tutmuyordu | Ölçülen sınır kılavuza (20 km'ye kadar double ile aynı) ve kod penceresine uyarı olarak eklendi; geniş alanda double önerilir. Orijin kaydırmalı float çözümü dört algoritmanın durumunu değiştirdiği için bu turda yapılmadı |
| DÜŞÜK | Yönelim üreticisi parametre aralıklarını denetlemiyordu (`sigma_heading = 0` → tamamı NaN; `particles = 30000`, `kp = −5` kabul ediliyordu) | Python'daki `[alt, üst]` aralıkları uygulanır, Türkçe hata |
| DÜŞÜK | Genel başlık önce eklenirse `spt_finite` yeniden tanımlanıyordu; iki üretici aynı `ornek_kullanim.c` / `KULLANIM.md` adlarını kullanıyordu | Ortak tanımlar tek koruma (`SPT_FUSION_SHARED_DEFS`) altında, genel başlığa eksik `SPT_ASIN` eklendi: iki sıra da `-Werror` ile derleniyor; genel füzyonda `ornek_kullanim_<senaryo>.c`, `KULLANIM_<senaryo>.md` |
| DÜŞÜK | Genel füzyonun çalışma klasörü `SPT_CODEGEN_DIR`'e uymuyordu | Derleme önbelleğiyle aynı köke taşındı |
| DÜŞÜK | Yönelim kılavuzu seçilmeyen alanlara başvuruyordu (`s.kp = 2.0f`, `s.ess` … yalnız Kalman seçiliyken bile) | Örnek, seçili algoritmanın gerçek alanı ve seçilen sayı türüyle; tanılama satırı yalnız PF / UKF seçiliyse |
| DÜŞÜK | Başlatma geçerliliği Python'da `\|a\|₁ > 0`, C'de `‖a‖₂ > 0` | Bilinçli (MCU'da 0/0 önlenir; gerçek sensör verisinde fark yok): "Python'dan bilinçli farklar" listesine yazıldı |
| DÜŞÜK | Saklı tanımlayıcılar üretilebiliyordu (önek `_x`; DAC adı "Sine 1k!" → `SINE_1K__…`) | Önek harfle başlar, çift / sondaki alt çizgi yok; DAC adında ardışık / baştaki / sondaki alt çizgiler atılır |

## Yapıcı ajanlar (ikinci tur)
Kullanıcının yeni istekleri için dört arka uç **bağımsız ajanlarca** yazıldı; her biri yalnız kendi dosyalarına yazdı,
kendi testleriyle doğrulandı ve ana oturum tarafından bağımsız olarak yeniden çalıştırılıp arayüze bağlandı:

| Ajan | Çıktı | Doğrulama |
|---|---|---|
| Füzyon C kodu üreteci | `spt/codegen/{mcu,toolchain,fusion_c}.py` | Üretilen C (double) Python ile ~1e-14 rad; tüm Cortex-M × gcc/clang uyarısız; bellek ölçülür |
| Sinyal üreteci (DAC) | `spt/dsp/siggen.py`, `spt/codegen/dac_c.py` | 109 test; C çıktısı Python ile örnek örnek aynı; nicemleme teorisi ±0.1 dB |
| Genel sensör füzyonu | `spt/dsp/fusion_generic.py`, `spt/dsp/fusion_scenarios/`, örnek kayıt 11–13 | 68 test; EKF ↔ UKF, KF ↔ ders kitabı KF, blok = tek sefer |
| Genel füzyon C kodu | `spt/codegen/fusion_generic_c.py` | Python ile karşılaştırmalı (double), çapraz derleme uyarısız |

## Kalıcı testlerle ilişkisi
Ajanların bulduğu ve düzeltilen her hata için `tests/` altına bir regresyon testi eklenir; böylece aynı hata
bir daha sessizce geri gelemez. Arayüz 2. tur ve C kodu doğrulayıcısı düzeltmelerinden sonra tüm paket **725 test**
(hepsi geçiyor; `pyflakes` temiz). Hızlı kontrol için:
```
.venv\Scripts\python -m pytest -q
```
