# SPTF — Signal Processing Toolbox Frame Formatı (v1)

Bu doküman, bir kaynağın (gömülü cihaz, PC simülatörü, kayıt dosyası) sinyal verisini
Signal Processing Toolbox arayüzüne göndermek için kullandığı ikili veri formatını (bayt düzenini) tanımlar.

- Gönderici ve alıcı davranışının normatif tanımı (oturum, zamanlama, hata yönetimi, uyumluluk):
  [SPTF_UDP_Uygulama_Protokolu.md](SPTF_UDP_Uygulama_Protokolu.md)
- Cihaz tarafı referans gerçekleme (C99, lwIP / BSD soket): [`embedded/sptf_udp`](../embedded/sptf_udp),
  kullanım kılavuzu [SPTF_UDP_Kullanim_Kilavuzu.md](SPTF_UDP_Kullanim_Kilavuzu.md)

## 1. Tasarım hedefleri

| Hedef | Nasıl sağlanıyor |
|---|---|
| Her tür sinyal | Kanal sayısı, örnekleme hızı, veri tipi, birim ve ölçek akışla birlikte gönderilir (kendini tanımlayan format) |
| Yüksek veri genişliği | 8–64 bit tamsayı, 24 bit paketli, float16/32/64, karmaşık (I/Q) tipler; 64 bit örnek sayacı ve ns zaman damgası |
| Yüksek bant genişliği | 1472 baytlık datagramda %96.5 verimlilik; başlık sabit ve küçük |
| Kayıp tespiti | 64 bit `first_sample` ile eksik örnek sayısı tam olarak bulunur |
| Tek taşıma | Canlı akış yalnız UDP ile taşınır; aynı çerçeve kayıt dosyasında (`.sptr`) değiştirilmeden saklanır |
| Ortaya katılabilme | Tanımlayıcı (DESCRIPTOR) periyodik tekrarlanır; alıcı akışın ortasında bağlanabilir |

## 2. Genel kurallar

- Tüm çok baytlı alanlar **little-endian**dir.
- Başlıklar 8 baytın katıdır; hizalama dolgusu yoktur, yapılar `packed` kabul edilir.
- Metin alanları UTF-8, sabit uzunluklu ve `\0` ile doldurulmuştur.
- `reserved` alanları gönderen tarafından 0 yazılır, alıcı tarafından yok sayılır.
- Bir cihaz birden fazla **akış** (`stream_id`) gönderebilir. Bir akıştaki tüm kanallar
  aynı veri tipine ve aynı örnekleme hızına sahiptir. Farklı hız/tip gerekiyorsa ayrı akış açılır.

## 3. Çerçeve başlığı (32 bayt, tüm çerçevelerde ortak)

| Ofset | Tip | Alan | Açıklama |
|---:|---|---|---|
| 0 | char[4] | `magic` | `"SPTF"` = `53 50 54 46` |
| 4 | u8 | `version` | `1` |
| 5 | u8 | `frame_type` | `1` = DESCRIPTOR, `2` = DATA, `3` = MARKER |
| 6 | u16 | `flags` | bit0: sonda CRC32 var. Diğer bitler 0 |
| 8 | u32 | `frame_len` | Başlık + payload + CRC dahil toplam bayt |
| 12 | u16 | `stream_id` | Akış numarası |
| 14 | u16 | `desc_rev` | Akış yapılandırma revizyonu (bkz. §8) |
| 16 | u32 | `seq` | Akış başına çerçeve sayacı; her çerçevede +1, taşınca 0 |
| 20 | u32 | `reserved` | 0 |
| 24 | u64 | `timestamp_ns` | Cihaz zamanı (ns). DATA için ilk örneğin zamanı. Bilinmiyorsa 0 |

## 4. DESCRIPTOR çerçevesi (`frame_type = 1`)

Akışın nasıl yorumlanacağını tanımlar.

| Ofset* | Tip | Alan | Açıklama |
|---:|---|---|---|
| 0 | f64 | `sample_rate_hz` | Kanal başına örnekleme hızı, **0.001 … 10¹² Hz** (dışındaki değerde tanımlayıcı reddedilir) |
| 8 | u16 | `channel_count` | 1..256 |
| 10 | u8 | `sample_type` | Bkz. §5 |
| 11 | u8 | `layout` | `0` = interleaved (örnek sıralı), `1` = planar (kanal sıralı) |
| 12 | u32 | `reserved` | 0 |
| 16 | char[32] | `stream_name` | Akış adı |
| 48 | ChannelInfo[`channel_count`] | `channels` | Kanal başına 64 bayt |

\* Ofsetler payload başlangıcına göredir (çerçeve ofseti = 32 + ofset).

**ChannelInfo (64 bayt)**

| Ofset | Tip | Alan | Açıklama |
|---:|---|---|---|
| 0 | char[32] | `name` | Kanal adı |
| 32 | char[16] | `unit` | Fiziksel birim (`V`, `A`, `g`, `°C`...) |
| 48 | f64 | `scale` | Ölçek |
| 56 | f64 | `offset` | Ofset |

Fiziksel değer: **`fiziksel = ham · scale + offset`**. Karmaşık tiplerde I ve Q bileşenlerine ayrı ayrı uygulanır.

Örnekler:
- 12 bit ADC, Vref = 3.3 V: `scale = 3.3 / 4096`, `offset = 0`
- Q15 sabit noktalı: `scale = 1 / 32768`
- Ham değer olduğu gibi: `scale = 1`, `offset = 0`

## 5. Örnek veri tipleri (`sample_type`)

| Kod | Tip | Bayt | Not |
|---:|---|---:|---|
| `0x01` | int8 | 1 | |
| `0x02` | uint8 | 1 | |
| `0x03` | int16 | 2 | |
| `0x04` | uint16 | 2 | |
| `0x05` | int24 | 3 | Paketli, little-endian, işaret uzatmalı |
| `0x06` | uint24 | 3 | Paketli, little-endian |
| `0x07` | int32 | 4 | |
| `0x08` | uint32 | 4 | |
| `0x09` | int64 | 8 | |
| `0x0A` | uint64 | 8 | |
| `0x10` | float16 | 2 | IEEE 754 half |
| `0x11` | float32 | 4 | |
| `0x12` | float64 | 8 | |
| `0x20` | cint16 | 4 | I (int16), Q (int16) |
| `0x21` | cint32 | 8 | I (int32), Q (int32) |
| `0x22` | cfloat32 | 8 | I (float32), Q (float32) |
| `0x23` | cfloat64 | 16 | I (float64), Q (float64) |

## 6. DATA çerçevesi (`frame_type = 2`)

| Ofset* | Tip | Alan | Açıklama |
|---:|---|---|---|
| 0 | u64 | `first_sample` | Bu çerçevedeki ilk örneğin akış içindeki indeksi (kanal başına sayılır, akış başında 0); **< 2⁶²** |
| 8 | u32 | `sample_count` | Kanal başına örnek sayısı (N) |
| 12 | u32 | `reserved` | 0 |
| 16 | — | `samples` | N × `channel_count` × tip boyutu |

**Yerleşim (`layout`)**, 3 kanal ve N örnek için:
- `0` interleaved: `s0c0 s0c1 s0c2  s1c0 s1c1 s1c2  …` (çok kanallı ADC + DMA tarama çıktısıyla aynı)
- `1` planar: `s0c0 s1c0 … sN-1c0  s0c1 s1c1 …`

Doğrulama: `frame_len == 32 + 16 + N · channel_count · tip_boyutu (+4 CRC varsa)`.

## 7. MARKER çerçevesi (`frame_type = 3`)

Debug sırasında olayları sinyal üzerinde işaretlemek için kullanılır (ör. "ISR overrun", "state = RUN").
Arayüzde grafik üzerinde dikey çizgi ve etiket olarak gösterilir.

| Ofset* | Tip | Alan | Açıklama |
|---:|---|---|---|
| 0 | u64 | `sample_index` | Olayın denk geldiği örnek indeksi |
| 8 | u32 | `code` | Kullanıcı tanımlı olay kodu |
| 12 | u16 | `text_len` | Metin uzunluğu (bayt) |
| 14 | u16 | `reserved` | 0 |
| 16 | char[`text_len`] | `text` | UTF-8, `\0` gerekmez |

## 8. Akış kuralları

1. Gönderen, akışa başlarken önce **DESCRIPTOR** gönderir.
2. DESCRIPTOR en az **saniyede bir** tekrarlanır. Alıcı akışa sonradan bağlansa da en geç 1 s içinde veriyi çözmeye başlar.
3. Yapılandırma değişirse (hız, kanal, tip, ölçek) `desc_rev` 1 artırılır ve hemen yeni DESCRIPTOR gönderilir.
4. Alıcı, `desc_rev` değeri bilinen DESCRIPTOR ile eşleşmeyen DATA çerçevelerini atar.
5. **Kayıp tespiti** (alıcı tarafında, akış başına `beklenen = önceki first_sample + önceki sample_count`):
   - `first_sample == beklenen` → normal
   - `first_sample > beklenen` → `first_sample − beklenen` örnek kayıp; araya `NaN` doldurulur
   - `first_sample == 0` → akış yeniden başladı (cihaz reset). Tampon sıfırlanmaz: yeni veri öncekinin
     arkasına eklenir, olay Günlük'e yazılır ve yeniden başlama sayacı artar
   - `first_sample < beklenen` → yinelenen / sırası bozuk çerçeve; atılır ve istatistiğe yazılır. **Ancak**
     kendi içinde ardışık (`first_sample` birbirini izleyen) **4 geri çerçeve** gelirse cihazın resetlendiği ve
     `first_sample = 0` çerçevesinin kaybolduğu anlaşılır: akış yeniden başlamış sayılır (ilk 3 çerçeve atılır).
     Aksi halde reset çerçevesinin kaybı, akışı eski konuma yetişene kadar kilitlerdi.
   - `first_sample − beklenen > 2⁴⁰` ya da `first_sample ≥ 2⁶²` → bozuk çerçeve (sayılır, atılır)
6. **Dayanıklılık:** Hiçbir bozuk çerçeve / datagram alıcıyı durdurmaz; hatalar sayılır ve ilk 20'si Günlük'e
   yazılır. Bilinmeyen `frame_type` değerleri (gelecek sürümler) `frame_len` kadar atlanır, datagramın geri kalanı
   işlenir. Datagram sonundaki sıfır dolgu yok sayılır.

## 9. CRC

- `flags` bit0 = 1 ise çerçevenin son 4 baytı CRC32'dir (u32, little-endian).
- Algoritma: **CRC-32/ISO-HDLC** (Ethernet/zlib CRC'si). Polinom `0x04C11DB7` (yansıtılmış), init `0xFFFFFFFF`, xorout `0xFFFFFFFF`.
- Kapsam: çerçevenin ilk baytından CRC'den önceki son bayta kadar.
- Kontrol değeri: `crc32("123456789") = 0xCBF43926`.
- İsteğe bağlıdır: UDP'nin kendi sağlama toplamı vardır. Uçtan uca bütünlük istendiğinde (ör. UDP sağlama
  toplamı göndericide kapalıysa) ve kayıt dosyasındaki bozulmayı yakalamak için açın.

## 10. Taşıma: UDP

Taşıma yalnızca UDP'dir (TCP desteklenmez). Gönderici davranışının normatif kuralları
[SPTF_UDP_Uygulama_Protokolu.md](SPTF_UDP_Uygulama_Protokolu.md)'dedir; bu bölüm özetidir.

- Arayüz, tüm arayüzlerde (`0.0.0.0`) **UDP 50505** portunu dinler (ayarlanabilir).
- Gönderen, PC'nin IP adresine unicast gönderir.
- Bir datagram **bir veya daha fazla tam çerçeve** taşır; bir çerçeve iki datagrama bölünmez.
- Önerilen datagram boyutu **≤ 1472 bayt** (MTU 1500 − IP/UDP başlıkları). Böylece IP parçalanması olmaz.
  Üst sınır 65507 bayttır.
- 1472 baytlık datagramda örnek verisi için 1420 bayt kalır (CRC ile). Örnek: 8 kanal int16 → 88 örnek/çerçeve.
- 1472 baytlık bir DESCRIPTOR en fazla **21 kanal** taşır. Daha fazla kanal IP parçalanması gerektirir;
  alternatif olarak kanallar birden fazla akışa bölünebilir.

**Bant genişliği (1472 B datagram, CRC dahil):**

| Hat | Örnek verisi (teorik) | Örnek kullanım |
|---|---|---|
| 100 Mbit/s | ~11.5 MB/s | 4 kanal × int16 × 1 MS/s = 8 MB/s |
| 1 Gbit/s | ~115 MB/s | 4 kanal × int16 × 14.4 MS/s |

Ölçülen alıcı kapasitesi (localhost): standart 1472 B datagramla 113 MB/s (~77 bin datagram/s), jumbo (8972 B)
ve büyük (65507 B) datagramla 113.5 MB/s — üçünde de kayıpsız. Ayrıntı: `docs/asamalar/Asama_1_Altyapi_ve_Canli_Akis.md`.
Gigabit'e yakın hızlarda cihaz tarafının yükünü azaltmak için, ağ destekliyorsa jumbo frame önerilir.
Ölçümler interleaved yerleşimle yapılmıştır; planar yerleşim de hızlı yoldan işlenir (kanal blokları tek
devrik kopyayla tampona yazılır) ancak en yüksek hızlarda **interleaved önerilir**.

> Windows: Güvenlik duvarında Python için UDP 50505 gelen trafiğine izin verilmelidir.

## 11. Kayıt dosyası (`.sptr`)

Alınan çerçeveler **değiştirilmeden** ve alındıkları zamanla birlikte dosyaya yazılır.
Dosya kendini tanımlar ve canlı akış gibi tekrar oynatılabilir.

**Dosya başlığı (32 bayt)**

| Ofset | Tip | Alan | Açıklama |
|---:|---|---|---|
| 0 | char[4] | `magic` | `"SPTR"` |
| 4 | u16 | `version` | `1` |
| 6 | u16 | `header_len` | `32` |
| 8 | u64 | `created_unix_ns` | Kayıt başlangıcı (Unix zamanı, ns) |
| 16 | u8[16] | `reserved` | 0 |

**Kayıtlar:** Başlıktan sonra dosya sonuna kadar tekrar eden `[u64 rx_time_ns][SPTF çerçevesi]` ikilileri.
`rx_time_ns`, çerçevenin PC'ye ulaştığı Unix zamanıdır (ns). Çerçeve uzunluğu `frame_len` alanından okunur.

- Kayıt akışın ortasında başlatılırsa, bilinen akışların son DESCRIPTOR'ları dosyanın başına yazılır; oynatma
  ilk örnekten itibaren çözülür.
- Okuyucu bozuk bir kayda rastlarsa bir sonraki geçerli kaydı (`"SPTF"` + makul `frame_len`) arayıp oradan devam
  eder; atlanan bayt ve bölge sayısı Günlük'e yazılır. Dosya sonundaki yarım kayıt atlanır.
- **Sınırlama (v1):** Kayıtta kaynak adresi saklanmaz. Aynı `stream_id`'yi kullanan iki cihaz aynı anda
  kaydedilirse oynatmada tek akış gibi görünür; çoklu cihazda farklı `stream_id` kullanın (v2'de kaynak alanı
  eklenecek).

## 12. Test vektörü

DATA çerçevesi: `stream_id = 1`, `desc_rev = 1`, `seq = 1`, `timestamp_ns = 1 000 000`,
`first_sample = 0`, `sample_count = 2`, 2 kanal int16 interleaved, değerler `[100, −1, 200, −2]`, CRC açık.

```
0000: 53 50 54 46 01 02 01 00 3C 00 00 00 01 00 01 00
0010: 01 00 00 00 00 00 00 00 40 42 0F 00 00 00 00 00
0020: 00 00 00 00 00 00 00 00 02 00 00 00 00 00 00 00
0030: 64 00 FF FF C8 00 FE FF A8 40 1E 68
```

## Ek A — C referans tanımları

Aşağıdaki `packed` yapılar yalnızca bayt düzenini göstermek içindir. Cihazda doğrudan
[`embedded/sptf_udp`](../embedded/sptf_udp) kütüphanesini kullanın: alanları bayt bayt küçük uçlu yazar, hizalama ve
bayt sırası varsayımı yapmaz ve bu dokümanın tüm kurallarını uygular.

```c
#include <stdint.h>

#define SPTF_MAGIC      0x46545053u   /* "SPTF" little-endian */
#define SPTF_VERSION    1u
#define SPTF_UDP_PORT   50505u

enum { SPTF_DESCRIPTOR = 1, SPTF_DATA = 2, SPTF_MARKER = 3 };
#define SPTF_FLAG_CRC32 0x0001u

typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint8_t  version;
    uint8_t  frame_type;
    uint16_t flags;
    uint32_t frame_len;
    uint16_t stream_id;
    uint16_t desc_rev;
    uint32_t seq;
    uint32_t reserved;
    uint64_t timestamp_ns;
} sptf_header_t;                       /* 32 B */

typedef struct __attribute__((packed)) {
    double   sample_rate_hz;
    uint16_t channel_count;
    uint8_t  sample_type;
    uint8_t  layout;
    uint32_t reserved;
    char     stream_name[32];
} sptf_descriptor_t;                   /* 48 B, ardından channel_count × sptf_channel_t */

typedef struct __attribute__((packed)) {
    char   name[32];
    char   unit[16];
    double scale;
    double offset;
} sptf_channel_t;                      /* 64 B */

typedef struct __attribute__((packed)) {
    uint64_t first_sample;
    uint32_t sample_count;
    uint32_t reserved;
} sptf_data_t;                         /* 16 B, ardından örnekler */

typedef struct __attribute__((packed)) {
    uint64_t sample_index;
    uint32_t code;
    uint16_t text_len;
    uint16_t reserved;
} sptf_marker_t;                       /* 16 B, ardından text_len bayt metin */
```
