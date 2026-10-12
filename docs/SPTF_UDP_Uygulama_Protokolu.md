# SPTF-UDP Uygulama Katmanı Protokolü

| Alan | Değer |
|---|---|
| Doküman | SPT-PRT-002 — SPTF-UDP Uygulama Katmanı Protokolü |
| Sürüm | 1.0 (protokol sürümü: SPTF v1) |
| Tarih | 2026-10-12 |
| Durum | Yayımlandı — **normatif** |
| Kapsam | Gömülü cihazdan Signal Processing Toolbox arayüzüne UDP ile gerçek zamanlı sinyal aktarımı |
| İlgili | [SPTF_Protokol_v1.md](SPTF_Protokol_v1.md) (SPT-PRT-001, bayt düzeni) · [SPTF_UDP_Kullanim_Kilavuzu.md](SPTF_UDP_Kullanim_Kilavuzu.md) · referans gerçekleme [`embedded/sptf_udp`](../embedded/sptf_udp) |

---

## İçindekiler

1. [Kapsam](#1-kapsam)
2. [Normatif atıflar](#2-normatif-atıflar)
3. [Terimler ve kısaltmalar](#3-terimler-ve-kısaltmalar)
4. [Gösterim ve gereksinim dili](#4-gösterim-ve-gereksinim-dili)
5. [Mimari](#5-mimari)
6. [Adresleme](#6-adresleme)
7. [Datagram kuralları](#7-datagram-kuralları)
8. [Akış yaşam döngüsü](#8-akış-yaşam-döngüsü)
9. [Alan anlamları](#9-alan-anlamları)
10. [Zamanlama ve hız](#10-zamanlama-ve-hız)
11. [Hata yönetimi](#11-hata-yönetimi)
12. [Bütünlük](#12-bütünlük)
13. [Güvenlik hususları](#13-güvenlik-hususları)
14. [Uyumluluk](#14-uyumluluk)
15. [Port ve tanımlayıcı kayıtları](#15-port-ve-tanımlayıcı-kayıtları)
- [Ek A — Mesaj akışı örnekleri](#ek-a--mesaj-akışı-örnekleri-bilgilendirici)
- [Ek B — Boyut ve kapasite tabloları](#ek-b--boyut-ve-kapasite-tabloları-bilgilendirici)
- [Ek C — Test vektörleri](#ek-c--test-vektörleri-normatif)
- [Ek D — Revizyon geçmişi](#ek-d--revizyon-geçmişi)

---

## 1. Kapsam

Bu doküman, bir **gönderici** (mikrodenetleyici, FPGA yumuşak çekirdeği, gömülü Linux, PC simülatörü) ile
**alıcı** (Signal Processing Toolbox) arasındaki uygulama katmanı protokolünü tanımlar:

- hangi çerçevelerin hangi sırayla ve hangi sıklıkta gönderileceği,
- sayaç ve zaman damgası alanlarının anlamı,
- datagram boyutlandırma, kayıp ve hata durumlarında davranış,
- uyumluluk sınıfları ve doğrulama yöntemi.

Çerçevelerin bayt düzeni bu dokümanda yinelenmez; [SPT-PRT-001](SPTF_Protokol_v1.md) normatiftir. İki doküman
çelişirse bayt düzeni için SPT-PRT-001, davranış için bu doküman geçerlidir.

**Kapsam dışı:** TCP ve diğer akış taşımaları (ürün yalnız UDP destekler), IPv6, yayın / çoklu yayın, alıcıdan
göndericiye komut kanalı, şifreleme ve kimlik doğrulama (bkz. §13).

## 2. Normatif atıflar

| Atıf | Başlık | Kullanım |
|---|---|---|
| SPT-PRT-001 | [SPTF v1 çerçeve formatı](SPTF_Protokol_v1.md) | Bayt düzeni, alıcı kuralları |
| RFC 768 | User Datagram Protocol | Taşıma |
| RFC 791 | Internet Protocol (IPv4) | Ağ katmanı |
| RFC 1122 | Requirements for Internet Hosts | Sağlama toplamı, parçalanma |
| RFC 8085 (BCP 145) | UDP Usage Guidelines | Boyutlandırma, tıkanıklık, kontrollü ortam |
| RFC 2119, RFC 8174 (BCP 14) | Gereksinim anahtar sözcükleri | §4 |
| IEEE 754-2019 | Kayan nokta aritmetiği | float16/32/64, binary64 alanlar |
| ITU-T V.42 / ISO/IEC 13239 | CRC-32 (ISO-HDLC) | İsteğe bağlı çerçeve CRC'si |
| ISO/IEC 10646, RFC 3629 | UTF-8 | Metin alanları |

## 3. Terimler ve kısaltmalar

| Terim | Tanım |
|---|---|
| Gönderici | Örnekleri üreten ve SPTF çerçevelerini gönderen uç (cihaz). |
| Alıcı | UDP portunu dinleyip çerçeveleri çözen uç (Signal Processing Toolbox). |
| Akış | Aynı hız, tip, yerleşim ve kanal kümesine sahip örnek dizisi; (kaynak IPv4 adresi, `stream_id`) ile tanımlanır. |
| Örnek anı | Bir akışın tüm kanallarının aynı andaki örnekleri. `first_sample`, `sample_count` örnek anı sayar. |
| Çerçeve | DESCRIPTOR, DATA ya da MARKER; 32 baytlık ortak başlıkla başlar (SPT-PRT-001 §3). |
| Datagram | Bir UDP yükü; bir veya daha fazla tam çerçeve taşır. |
| Yapılandırma revizyonu | `desc_rev`; akışın DESCRIPTOR içeriği değiştiğinde artan 16 bit sayaç. |
| DT | DESCRIPTOR tekrar aralığı. |
| MTU | Maximum Transmission Unit; Ethernet'te 1500 B. |

## 4. Gösterim ve gereksinim dili

Bu dokümanda gereksinim sözcükleri BCP 14 (RFC 2119, RFC 8174) anlamındadır ve yalnızca BÜYÜK harfle
yazıldıklarında bu anlamı taşır:

| Bu doküman | BCP 14 | Anlam |
|---|---|---|
| **ZORUNLU** | MUST / REQUIRED | Uyumluluk için mutlak koşul |
| **YASAK** | MUST NOT | Mutlak yasak |
| **ÖNERİLİR** | SHOULD / RECOMMENDED | Geçerli bir gerekçe yoksa uyulmalı |
| **ÖNERİLMEZ** | SHOULD NOT | Geçerli bir gerekçe yoksa yapılmamalı |
| **SEÇİMLİK** | MAY / OPTIONAL | Serbest |

Gereksinimler `[G-nn]` (gönderici) ve `[A-nn]` (alıcı) biçiminde numaralanır; §14'teki uyumluluk tablosu bu
numaralara atıf yapar. "Bilgilendirici" işaretli bölümler gereksinim içermez.

## 5. Mimari

### 5.1 Katman modeli

```
 Gönderici (cihaz)                                       Alıcı (PC)
+-----------------------------+                         +------------------------------+
| Uygulama: ADC, IMU, çevrim  |                         | Canlı İzleme, Filtre, Füzyon |
+-----------------------------+                         +------------------------------+
| SPTF-UDP oturum kuralları   |  <-- bu doküman -->     | Akış durumu, kayıp tespiti   |
+-----------------------------+                         +------------------------------+
| SPTF v1 çerçeveleri         |  <-- SPT-PRT-001 -->    | Çerçeve doğrulama / çözme    |
+-----------------------------+                         +------------------------------+
| UDP (RFC 768)               |  ===== port 50505 ====> | UDP soketi (0.0.0.0:50505)   |
+-----------------------------+                         +------------------------------+
| IPv4 / Ethernet             |                         | IPv4 / Ethernet              |
+-----------------------------+                         +------------------------------+
```

### 5.2 İletişim modeli

- Protokol **tek yönlüdür** (gönderici → alıcı). v1'de alıcıdan göndericiye mesaj, onay (ACK) veya yeniden
  gönderim yoktur. Kayıplar sayılır ve görünür kılınır, telafi edilmez.
- Durum **göndericide** tutulur ve her çerçeve kendi başına yorumlanabilir: alıcı akışa herhangi bir anda
  katılabilir (en geç bir DT sonra veri çözmeye başlar).
- Bir gönderici birden fazla akış gönderebilir; bir alıcı birden fazla göndericiyi aynı portta dinleyebilir.

## 6. Adresleme

- **[G-01]** Gönderici, çerçeveleri alıcının IPv4 adresine **unicast** olarak göndermek ZORUNLUDUR. Yayın
  (broadcast) ve çoklu yayın (multicast) adreslerine gönderim v1'de tanımlı değildir ve ÖNERİLMEZ.
- **[G-02]** Hedef port yapılandırılabilir OLMALIDIR (ZORUNLU); varsayılan **50505**'tir.
- **[G-03]** Kaynak UDP portu SEÇİMLİKTİR (geçici port olabilir). Bir oturum boyunca sabit kalması ÖNERİLİR.
- **[G-04]** `stream_id`, aynı kaynak IPv4 adresinden gönderilen akışlar arasında benzersiz OLMALIDIR (ZORUNLU).
  Alıcı akışı (kaynak IPv4, `stream_id`) ikilisiyle ayırır; kaynak port anahtara dahil değildir.
- **[A-01]** Alıcı, yapılandırılan yerel adres ve portta (varsayılan `0.0.0.0:50505`) dinler ve birden fazla
  kaynaktan gelen akışları ayrı tutar.

## 7. Datagram kuralları

- **[G-05]** Bir datagram bir veya daha fazla **tam** çerçeve taşır. Bir çerçevenin iki datagrama bölünmesi
  YASAKTIR.
- **[G-06]** Datagram boyutu yol MTU'sunu aşmamalıdır (ÖNERİLİR, RFC 8085 §3.2): standart Ethernet'te
  **≤ 1472 bayt** (1500 − 20 IPv4 − 8 UDP). Yolun tamamı jumbo çerçeve destekliyorsa 8972 bayt SEÇİMLİKTİR.
  Hiçbir durumda **65507** bayt aşılamaz (YASAK).
- **[G-07]** Datagram sonuna sıfır dolgu eklenmesi SEÇİMLİKTİR; alıcı yok sayar.
- **[G-08]** Bir akışın DESCRIPTOR çerçevesi tek datagrama sığmalıdır (ZORUNLU). 1472 baytlık datagramda bu
  en fazla 21 kanal demektir; daha fazla kanal için jumbo çerçeve kullanın ya da kanalları birden fazla akışa
  bölün (bkz. Ek B).
- **[A-02]** Alıcı, datagramdaki çerçeveleri sırayla işler; bir çerçeve hatalıysa datagramın geri kalanını atar,
  bilinmeyen `frame_type`'ı `frame_len` kadar atlayıp devam eder (SPT-PRT-001 §8.6).

> Bilgilendirici: Referans gerçekleme her datagrama tek çerçeve koyar ve DATA çerçevelerini datagramı
> dolduracak en büyük örnek sayısıyla keser. Bu, alıcının hızlı yolu için en verimli biçimdir.

## 8. Akış yaşam döngüsü

### 8.1 Durumlar

```
             init / reconfigure                 DESCRIPTOR gönderildi
  [KAPALI] ----------------------> [TANIM BEKLİYOR] -------------------> [YAYINDA]
     ^                                   ^   |  gönderim başarısız             |   |
     |                                   |   +--(yeniden dene)                 |   |
     |  deinit / güç kesildi             |      reconfigure (desc_rev + 1)     |   | DT doldu:
     +-----------------------------------+-------------------------------------+   | DESCRIPTOR tekrar
                                                                                   +--(YAYINDA kalır)
```

| Durum | Gönderilebilecek çerçeveler |
|---|---|
| KAPALI | Yok |
| TANIM BEKLİYOR | Yalnız DESCRIPTOR |
| YAYINDA | DESCRIPTOR, DATA, MARKER |

### 8.2 Başlatma

- **[G-09]** Bir akışın ilk çerçevesi DESCRIPTOR OLMALIDIR (ZORUNLU). DESCRIPTOR başarıyla gönderilmeden o
  akışın DATA veya MARKER çerçevesi gönderilmesi ÖNERİLMEZ (alıcı tanımadığı akışın DATA'sını atar,
  MARKER'ını yok sayar).
- **[G-10]** Akış başında `first_sample = 0` OLMALIDIR (ZORUNLU). `desc_rev` başlangıç değeri SEÇİMLİKTİR;
  referans gerçekleme 1 kullanır.

### 8.3 DESCRIPTOR tekrarı

- **[G-11]** Akış YAYINDA iken DESCRIPTOR en az **1000 ms**'de bir tekrarlanmalıdır (ZORUNLU). Böylece sonradan
  bağlanan alıcı en geç 1 s içinde veri çözer.
- **[G-12]** Tekrar aralığının 500 ms veya daha kısa olması ÖNERİLİR (bir DESCRIPTOR kaybını tolere eder).
  Göndericinin saati yoksa aralık örnek sayısıyla ölçülebilir: her `⌊fs · DT⌋` örnekte bir (SEÇİMLİK).
- **[G-13]** Tekrarlanan DESCRIPTOR'ın içeriği ve `desc_rev`'i değişmemelidir (ZORUNLU); yalnız `seq` ve
  `timestamp_ns` farklıdır.
- **[A-03]** Alıcı, aynı içerikli DESCRIPTOR'ı tekrar olarak yok sayar.

### 8.4 Yeniden yapılandırma

- **[G-14]** DESCRIPTOR içeriğindeki herhangi bir alan (hız, kanal sayısı, tip, yerleşim, ad, birim, ölçek,
  ofset) değişirse `desc_rev` 1 artırılmalıdır (ZORUNLU, 65535'ten sonra 0) ve yeni DESCRIPTOR, yeni
  revizyonlu ilk DATA'dan önce gönderilmelidir (ZORUNLU).
- **[G-15]** Yeniden yapılandırmadan sonra `first_sample`'ın 0'dan başlaması ÖNERİLİR (yeni zaman ekseni).
- **[A-04]** Alıcı, revizyon değişimini yeni bir akış olarak ele alır; eski revizyonlu DATA çerçevelerini atar.

### 8.5 Yeniden başlama (cihaz reseti)

- **[G-16]** Gönderici yeniden başladığında akış §8.2'deki gibi baştan başlar (`first_sample = 0`, önce
  DESCRIPTOR).
- **[A-05]** Alıcı yeniden başlamayı `first_sample == 0` ile ya da (0 çerçevesi kaybolduysa) kendi içinde
  ardışık 4 geri çerçeveyle anlar; tamponu silmez, olayı Günlük'e yazar (SPT-PRT-001 §8.5).

### 8.6 Sonlandırma

v1'de açık bir kapanış mesajı yoktur. **[A-06]** Alıcı, son çerçeveden beri geçen süreyi akış istatistiklerinde
"boşta" olarak gösterir.

## 9. Alan anlamları

| Alan | Gereksinim |
|---|---|
| `seq` | **[G-17]** Akış başına, türünden bağımsız her çerçevede 1 artar; 2³²'de 0'a döner (ZORUNLU). Kodlanıp gönderilemeyen (sürücü / yığın hatası) çerçevenin de numara tüketmesi ÖNERİLİR; tampon alınamadığı için hiç üretilmeyen çerçeve numara tüketmez. Alıcı v1 bu alanı yalnız tanı amaçlı kullanır. |
| `first_sample` | **[G-18]** Bir akışın DATA çerçeveleri sürekli OLMALIDIR: `first(n+1) = first(n) + count(n)` (ZORUNLU). Gönderici örnek kaybederse (tampon yok, gönderim hatası) `first_sample`'ı kaybedilen örnek sayısı kadar **yine ilerletmelidir** (ZORUNLU); alıcı boşluğu tam uzunlukta NaN ile gösterir. Değer < 2⁶² OLMALIDIR (ZORUNLU); tek çerçevede 2⁴⁰'tan büyük ileri sıçrama YASAKTIR. |
| `sample_count` | **[G-19]** ≥ 1 olması ÖNERİLİR. Yük uzunluğu `sample_count × channel_count × tip_boyutu` OLMALIDIR (ZORUNLU). |
| `timestamp_ns` | **[G-20]** Göndericinin monoton saatine göre ns: DATA'da ilk örneğin, MARKER'da olayın, DESCRIPTOR'da gönderimin zamanı. Bilinmiyorsa 0 (SEÇİMLİK). Monoton olması ÖNERİLİR; PC saatine eşlenmesi gerekmez. Alıcı v1 örnek zamanlamasını `first_sample` ve hızdan hesaplar, bu alanı kullanmaz (bilgilendirici). |
| `flags` | **[G-21]** Yalnız bit0 (CRC-32) tanımlıdır; diğer bitler 0 OLMALIDIR (ZORUNLU). |
| `reserved` | **[G-22]** Tüm `reserved` alanları 0 yazılmalıdır (ZORUNLU); alıcı yok sayar. |
| Metin alanları | **[G-23]** UTF-8, sabit uzunluk, kalan baytlar `\0`. Alanı tam dolduran metinde sonlandırıcı gerekmez. Çok baytlı bir karakterin alan sınırında bölünmesi ÖNERİLMEZ. |
| `sample_index` (MARKER) | **[G-24]** Akışın `first_sample` eksenindeki örnek indeksidir; gelecekteki veya geçmişteki bir örneği gösterebilir (SEÇİMLİK). |
| `scale`, `offset` | **[G-25]** Sonlu IEEE 754 değerler OLMALIDIR (ZORUNLU). Fiziksel = ham · `scale` + `offset`. |

## 10. Zamanlama ve hız

- **[G-26]** Protokol **kontrollü ortamlar** için tasarlanmıştır (RFC 8085 §3.6: noktadan noktaya bağlantı ya da
  ayrılmış yerel ağ). İnternet üzerinden veya paylaşılan üretim ağlarında kullanılması YASAKTIR; v1'de
  tıkanıklık denetimi yoktur.
- **[G-27]** Gönderici, toplam hızın hattın ve alıcının kapasitesini aşmamasını sağlamalıdır (ZORUNLU).
  Örnekleri üretildikleri hızda, bloklar hâlinde göndermesi (patlama yerine düzenli akış) ÖNERİLİR.
- **[G-28]** Ağ yığını çağrılarının kesme (ISR) bağlamında yapılması ÖNERİLMEZ; ISR yalnız tamponu hazır
  olarak işaretlemelidir.
- Bilgilendirici: Ölçülen alıcı kapasitesi (localhost, 1472 B datagram) 113 MB/s, ~77 000 datagram/s'dir,
  kayıpsız (SPT-PRT-001 §10). Gigabit'e yakın hızlarda jumbo çerçeve, göndericinin datagram başına yükünü azaltır.

## 11. Hata yönetimi

### 11.1 Gönderici

| Durum | Gereken davranış |
|---|---|
| Tampon alınamadı (ör. lwIP pbuf havuzu boş) | **[G-29]** Etkilenen örnekler kayıp sayılır, `first_sample` ilerletilir ([G-18]); hata sayacı artırılır. Bloklanarak beklemek ÖNERİLMEZ (gerçek zamanlı çevrim). |
| Gönderim hatası / meşgul | **[G-30]** [G-29] ile aynı. Sonraki blok normal sürer. |
| DESCRIPTOR gönderilemedi | **[G-31]** Bir sonraki fırsatta yeniden denenir (ZORUNLU). İlk DESCRIPTOR gönderilemediyse o akışın DATA'sı gönderilmez ([G-09]). |
| Geçersiz yapılandırma | **[G-32]** Akış başlatılmamalıdır (ZORUNLU): hız ∉ [0.001, 10¹²] Hz, kanal ∉ [1, 256], bilinmeyen tip / yerleşim, sınırı aşan metin, sonlu olmayan ölçek / ofset, datagrama sığmayan DESCRIPTOR. |

### 11.2 Alıcı (SPT-PRT-001 §8 özeti)

- **[A-07]** Hiçbir bozuk çerçeve alıcıyı durdurmaz; hatalar sayılır, ilk 20'si Günlük'e yazılır.
- **[A-08]** Tanımlayıcısı bilinmeyen DATA atılır ve "tanımlayıcı yok" sayacına yazılır.
- **[A-09]** `first_sample` beklenenden büyükse fark kadar örnek kayıp sayılır ve araya NaN konur; küçükse
  çerçeve yinelenen kabul edilip atılır (yeniden başlama kuralı [A-05] hariç).
- **[A-10]** CRC hatalı çerçeve ve datagramın geri kalanı atılır.

## 12. Bütünlük

- **[G-33]** IPv4'te UDP sağlama toplamının göndericide etkin olması ÖNERİLİR (RFC 8085 §3.4; lwIP'de
  `CHECKSUM_GEN_UDP` ya da donanım sağlama toplamı boşaltması).
- **[G-34]** UDP sağlama toplamı kapalıysa ya da uçtan uca bütünlük gerekiyorsa çerçeve CRC-32'si (`flags` bit0)
  ÖNERİLİR. CRC, çerçevenin ilk baytından CRC'den önceki son bayta kadar hesaplanır (SPT-PRT-001 §9).

## 13. Güvenlik hususları

- Protokolde kimlik doğrulama, bütünlük imzası ve şifreleme yoktur. Ağa erişebilen herkes sahte akış
  gönderebilir veya trafiği okuyabilir. Yalnız güvenilen, yalıtılmış ağlarda kullanın ([G-26]).
- Alıcı tüm alanları doğrular (aralık, uzunluk, CRC) ve bozuk girdide durmaz ([A-07]); bellek kullanımı akış
  başına üst sınırlıdır.
- Protokol yanıt üretmediği için yansıtma / yükseltme saldırılarına aracı olmaz. v1 göndericisi gelen paket
  dinlemez.
- Windows'ta alıcının güvenlik duvarında yalnız gereken porta (UDP 50505) ve tercihen yalnız cihazın ağına
  izin verilmesi ÖNERİLİR.

## 14. Uyumluluk

### 14.1 Sınıflar

| Sınıf | Kapsam |
|---|---|
| **A — Temel gönderici** | DESCRIPTOR + DATA; [G-01]…[G-23], [G-25]…[G-33] |
| **B — Tam gönderici** | A + MARKER ([G-24]), CRC-32 ([G-34]), yeniden yapılandırma ([G-14]) |
| **Alıcı** | [A-01]…[A-10]; referans: Signal Processing Toolbox |

### 14.2 Referans gerçekleme

[`embedded/sptf_udp`](../embedded/sptf_udp) v1.0.0 **Sınıf B** göndericidir. Doğrulama
[`tests/test_embedded_sptf.py`](../tests/test_embedded_sptf.py) ile yapılır:

| Gereksinimler | Nasıl doğrulanır |
|---|---|
| Bayt düzeni, [G-05], [G-17], [G-18], [G-21], [G-22], [G-23] | 68 yapılandırmada (17 tip × 2 yerleşim × CRC açık / kapalı) C çıktısının Python referans kodlayıcısıyla **bayt bayt** aynı olması |
| [G-09], [G-11], [G-12], [G-13] | Aynı testte DESCRIPTOR sırası ve tekrar zamanlaması; C birim testinde saatli / saatsiz periyot |
| [G-14], [G-15] | C birim testi: `desc_rev` artışı, `first_sample = 0`, DESCRIPTOR önceliği |
| [G-18], [G-29], [G-30], [G-31] | C birim testi: tampon / gönderim hatasında kayıp sayımı ve sayaç ilerlemesi |
| [G-32] | C birim testi: geçersiz hız, NaN, kanal sınırları, metin uzunluğu, 22 kanal DESCRIPTOR |
| [A-01]…[A-10] ile birlikte uçtan uca | Gerçek UDP (127.0.0.1): 50 kHz × 4 kanal rampa, alıcıda kayıpsız ve değerler bit düzeyinde aynı; demo modunda iki akış + işaretler |
| Ek C vektörleri | C birim testi ve Python kodlayıcısı aynı baytları üretir |

### 14.3 Uyumluluk kontrol listesi (gönderici geliştiricisi için)

- [ ] Hedef IP ve port yapılandırılabilir; unicast ([G-01], [G-02])
- [ ] `stream_id` cihazda benzersiz ([G-04])
- [ ] Datagram ≤ 1472 B (ya da yolun MTU'su − 28); çerçeve bölünmüyor ([G-05], [G-06])
- [ ] İlk çerçeve DESCRIPTOR; ≤ 1 s'de bir tekrar ([G-09], [G-11])
- [ ] Yapılandırma değişince `desc_rev` + 1 ve önce DESCRIPTOR ([G-14])
- [ ] `first_sample` sürekli; kayıpta yine ilerliyor ([G-18])
- [ ] `seq` her çerçevede + 1 ([G-17]); `reserved` ve tanımsız `flags` bitleri 0 ([G-21], [G-22])
- [ ] Ağ çağrıları ISR dışında ([G-28]); yalnız kontrollü ağda ([G-26])

## 15. Port ve tanımlayıcı kayıtları

Bu protokol için IANA port kaydı yapılmamıştır. 50505 yapılandırılabilir bir varsayılandır; yerel ağda
çakışma olursa her iki uçta değiştirin. `frame_type` 4–255 ve `sample_type`'ın tanımsız kodları gelecek
sürümler için ayrılmıştır; v1 göndericisi bunları kullanmamalıdır (YASAK).

---

## Ek A — Mesaj akışı örnekleri (bilgilendirici)

**A.1 Normal akış** — 2 kanal uint16, 20 kHz, 256 örneklik bloklar, DT = 500 ms:

```
 t (ms)   Gönderici                                              Alıcı
    0     DESCRIPTOR seq=0  rev=1                       ---->    yeni akış "Motor akımları"
   12.8   DATA       seq=1  first=0      count=256      ---->    tampon açılır, çizim başlar
   25.6   DATA       seq=2  first=256    count=256      ---->
   ...
  499.2   DATA       seq=39 first=9728   count=256      ---->
  512.0   DESCRIPTOR seq=40 rev=1                       ---->    (tekrar, yok sayılır)
  512.0   DATA       seq=41 first=9984   count=256      ---->
```

**A.2 Göndericide tampon tükenmesi** — bir blok gönderilemedi:

```
  DATA seq=49 first=12032 count=256      ---->   normal
  (blok 12288..12543 için pbuf yok: çerçeve üretilmedi, seq tüketilmedi, first yine ilerledi)
  DATA seq=50 first=12544 count=256      ---->   256 örnek kayıp sayılır, araya NaN; zaman ekseni doğru
```

**A.3 Ortadan katılma** — alıcı 300. ms'de dinlemeye başlar:

```
  DATA seq=24 …                          ---->   atılır ("tanımlayıcı yok")
  …
  DESCRIPTOR seq=40 rev=1                ---->   akış tanındı
  DATA seq=41 first=9984                 ---->   çözülür (en geç 1 DT gecikme)
```

**A.4 Yeniden yapılandırma** — hız 20 kHz → 40 kHz:

```
  DATA seq=99  rev=1 first=…             ---->
  DESCRIPTOR seq=100 rev=2 (40 kHz)      ---->   eski akış kapanır, yeni akış açılır
  DATA seq=101 rev=2 first=0             ---->
```

## Ek B — Boyut ve kapasite tabloları (bilgilendirici)

**B.1 Bir DATA çerçevesine sığan örnek sayısı** (`⌊(datagram − 48 − crc) / (kanal × tip_boyutu)⌋`):

| Yapılandırma | Örnek anı başına | 1472 B, CRC yok | 1472 B, CRC | 8972 B, CRC yok |
|---|---:|---:|---:|---:|
| 1 × int16 | 2 B | 712 | 710 | 4462 |
| 2 × int16 / uint16 | 4 B | 356 | 355 | 2231 |
| 3 × int16 | 6 B | 237 | 236 | 1487 |
| 1 × int24 | 3 B | 474 | 473 | 2974 |
| 8 × int16, 4 × int32, 2 × cfloat32 | 16 B | 89 | 88 | 557 |
| 6 × float32 | 24 B | 59 | 59 | 371 |

**B.2 DESCRIPTOR sınırı** (`32 + 48 + 64 · kanal (+ 4)` ≤ datagram):

| Datagram | En fazla kanal |
|---|---:|
| 1472 B | 21 |
| 8972 B | 138 |
| 65507 B | 256 (protokol üst sınırı) |

**B.3 Örnek trafik** (yalnız UDP yükü; IP / Ethernet ek yükü hariç):

| Akış | Yük | Datagram / s (1472 B, CRC yok) |
|---|---:|---:|
| 2 × uint16 @ 20 kHz, 256'lık bloklar | 80 kB/s | 78 |
| 6 × float32 @ 1 kHz, 10'luk bloklar | 24 kB/s | 100 |
| 8 × int16 @ 1 MS/s | 16 MB/s | ≈ 11 236 |

## Ek C — Test vektörleri (normatif)

Uyumlu bir gönderici aşağıdaki girdilerden bu baytları üretmelidir. Üçü de referans gerçeklemenin birim
testinde ve Python referans kodlayıcısında doğrulanır.

**C.1 DATA** — SPT-PRT-001 §12'deki vektör (2 kanal int16, CRC açık).

**C.2 DESCRIPTOR** — `stream_id = 1`, `desc_rev = 1`, `seq = 0`, `timestamp_ns = 0`, 1000 Hz, 1 kanal int16
interleaved, akış adı `"ADC"`, kanal `"Ia"` / `"A"`, `scale = 0.001`, `offset = 0`, CRC açık (148 bayt):

```
0000: 53 50 54 46 01 01 01 00 94 00 00 00 01 00 01 00
0010: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
0020: 00 00 00 00 00 40 8F 40 01 00 03 00 00 00 00 00
0030: 41 44 43 00 00 00 00 00 00 00 00 00 00 00 00 00
0040: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
0050: 49 61 00 00 00 00 00 00 00 00 00 00 00 00 00 00
0060: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
0070: 41 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
0080: FC A9 F1 D2 4D 62 50 3F 00 00 00 00 00 00 00 00
0090: BA 27 3D FC
```

**C.3 MARKER** — `stream_id = 1`, `desc_rev = 1`, `seq = 5`, `timestamp_ns = 2 000 000`, `sample_index = 1000`,
`code = 42`, metin `"RUN"`, CRC açık (55 bayt):

```
0000: 53 50 54 46 01 03 01 00 37 00 00 00 01 00 01 00
0010: 05 00 00 00 00 00 00 00 80 84 1E 00 00 00 00 00
0020: E8 03 00 00 00 00 00 00 2A 00 00 00 03 00 00 00
0030: 52 55 4E 37 7D 55 30
```

**C.4 CRC-32 kontrol değeri:** `crc32("123456789") = 0xCBF43926`.

## Ek D — Revizyon geçmişi

| Sürüm | Tarih | Değişiklik |
|---|---|---|
| 1.0 | 2026-10-12 | İlk yayın. Taşıma yalnız UDP (TCP planı kaldırıldı); referans gerçekleme `embedded/sptf_udp` 1.0.0. |
