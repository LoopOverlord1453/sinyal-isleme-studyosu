/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    sptf.h
 * @brief   SPTF-UDP gömülü gönderici kütüphanesi — genel API.
 *
 * Mikrodenetleyicideki örnekleri (ADC, IMU, kontrol çevrimi değişkenleri, I/Q …) SPTF v1 çerçeveleri hâlinde
 * UDP ile Signal Processing Toolbox arayüzüne gönderir.
 *
 * Katmanlar:
 *  - @ref sptf_types   — veri tipleri, durum kodları, yapılandırma yapıları
 *  - @ref sptf_frame   — taşımadan bağımsız çerçeve kodlayıcılar ve CRC-32 (dosyaya kayıt vb. için de)
 *  - @ref sptf_link    — UDP göndericisi: akış durumu, otomatik bölme, DESCRIPTOR tekrarı, işaretler
 *
 * Özellikler: C99, dinamik bellek yok, özyineleme yok, `packed` yapı yok (alanlar bayt bayt küçük uçlu yazılır),
 * hizalama ve bayt sırası varsayımı yok, yeniden girişli (durum yalnız çağıranın nesnelerinde).
 *
 * @note    Bir ::sptf_link_t ve ona bağlı akışlar aynı anda tek bir bağlamdan (görev / ana döngü) kullanılmalıdır.
 *          Kesme (ISR) içinden çağırmayın; ISR'de yalnızca tamponu hazır olarak işaretleyin.
 *
 * @version 1.0.0
 * @see     docs/SPTF_UDP_Uygulama_Protokolu.md (üst seviye protokol), docs/SPTF_Protokol_v1.md (çerçeve formatı)
 */
#ifndef SPTF_H
#define SPTF_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "sptf_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup sptf_types Tipler ve sabitler
 * @brief    Protokol sabitleri, veri tipleri ve yapılandırma yapıları.
 * @{
 */

/** @brief Kütüphane sürümü (anlamsal sürümleme: ana.alt.yama). */
#define SPTF_LIB_VERSION_MAJOR 1u
#define SPTF_LIB_VERSION_MINOR 0u     /**< @brief Alt sürüm. */
#define SPTF_LIB_VERSION_PATCH 0u     /**< @brief Yama sürümü. */

#define SPTF_PROTOCOL_VERSION  1u     /**< @brief Kablodaki `version` alanı (SPTF v1). */
#define SPTF_UDP_PORT          50505u /**< @brief Arayüzün varsayılan dinleme portu. */
#define SPTF_MAX_UDP_PAYLOAD   65507u /**< @brief IPv4 üzerinde UDP yükü üst sınırı (bayt). */

#define SPTF_HEADER_SIZE       32u    /**< @brief Ortak çerçeve başlığı (bayt). */
#define SPTF_DESC_BODY_SIZE    48u    /**< @brief DESCRIPTOR sabit gövdesi (bayt). */
#define SPTF_CHANNEL_SIZE      64u    /**< @brief Kanal başına ChannelInfo (bayt). */
#define SPTF_DATA_BODY_SIZE    16u    /**< @brief DATA alt başlığı (bayt). */
#define SPTF_MARKER_BODY_SIZE  16u    /**< @brief MARKER alt başlığı (bayt). */
#define SPTF_CRC_SIZE          4u     /**< @brief Sondaki CRC-32 (bayt). */

#define SPTF_NAME_SIZE         32u    /**< @brief Akış ve kanal adı alanı (bayt, UTF-8). */
#define SPTF_UNIT_SIZE         16u    /**< @brief Birim alanı (bayt, UTF-8). */
#define SPTF_MAX_CHANNELS      256u   /**< @brief Akış başına en fazla kanal. */
#define SPTF_MIN_SAMPLE_RATE   1.0e-3 /**< @brief En küçük örnekleme hızı (Hz). */
#define SPTF_MAX_SAMPLE_RATE   1.0e12 /**< @brief En büyük örnekleme hızı (Hz). */

/** @brief `timestamp_ns` bilinmiyor: alan 0 yazılır. */
#define SPTF_TS_NONE           0u
/** @brief `timestamp_ns` saatten türetilsin: bloğun ilk örneği = şimdi − N · Ts (blok yeni tamamlandı). */
#define SPTF_TS_AUTO           UINT64_MAX
/** @brief ::sptf_stream_marker için: işaret, akışın sıradaki örneğine konur. */
#define SPTF_MARKER_AT_NEXT    UINT64_MAX

/** @brief Fonksiyon dönüş kodları. Başarı 0, hatalar negatiftir. */
typedef enum
{
    SPTF_OK         = 0,   /**< Başarılı. */
    SPTF_ERR_ARG    = -1,  /**< NULL işaretçi ya da aralık dışı argüman. */
    SPTF_ERR_CONFIG = -2,  /**< Geçersiz akış / bağlantı yapılandırması. */
    SPTF_ERR_SIZE   = -3,  /**< Çerçeve datagrama ya da tampona sığmıyor. */
    SPTF_ERR_NOMEM  = -4,  /**< Taşıma katmanı tampon veremedi (ör. lwIP pbuf havuzu boş). */
    SPTF_ERR_IO     = -5,  /**< Taşıma katmanı gönderemedi (ağ / sürücü hatası). */
    SPTF_ERR_BUSY   = -6,  /**< Taşıma şu an meşgul, daha sonra yeniden denenebilir (EWOULDBLOCK). */
    SPTF_ERR_STATE  = -7   /**< Nesne başlatılmamış ya da bağlantıya ekli değil. */
} sptf_status_t;

/** @brief Çerçeve türleri (`frame_type`). */
typedef enum
{
    SPTF_FRAME_DESCRIPTOR = 1, /**< Akış tanımı: hız, tip, kanallar. */
    SPTF_FRAME_DATA       = 2, /**< Örnek verisi. */
    SPTF_FRAME_MARKER     = 3  /**< Olay işareti (grafikte dikey çizgi + etiket). */
} sptf_frame_type_t;

/** @brief Örnek veri tipleri (`sample_type`). Tüm çok baytlı tipler kabloda küçük uçludur. */
typedef enum
{
    SPTF_INT8     = 0x01, /**< 1 bayt, işaretli. */
    SPTF_UINT8    = 0x02, /**< 1 bayt, işaretsiz. */
    SPTF_INT16    = 0x03, /**< 2 bayt, işaretli. */
    SPTF_UINT16   = 0x04, /**< 2 bayt, işaretsiz. */
    SPTF_INT24    = 0x05, /**< 3 bayt paketli, küçük uçlu (çağıran paketler). */
    SPTF_UINT24   = 0x06, /**< 3 bayt paketli, küçük uçlu (çağıran paketler). */
    SPTF_INT32    = 0x07, /**< 4 bayt, işaretli. */
    SPTF_UINT32   = 0x08, /**< 4 bayt, işaretsiz. */
    SPTF_INT64    = 0x09, /**< 8 bayt, işaretli. */
    SPTF_UINT64   = 0x0A, /**< 8 bayt, işaretsiz. */
    SPTF_FLOAT16  = 0x10, /**< IEEE 754 binary16. */
    SPTF_FLOAT32  = 0x11, /**< IEEE 754 binary32. */
    SPTF_FLOAT64  = 0x12, /**< IEEE 754 binary64. */
    SPTF_CINT16   = 0x20, /**< Karmaşık: I (int16), Q (int16). */
    SPTF_CINT32   = 0x21, /**< Karmaşık: I (int32), Q (int32). */
    SPTF_CFLOAT32 = 0x22, /**< Karmaşık: I (float32), Q (float32). */
    SPTF_CFLOAT64 = 0x23  /**< Karmaşık: I (float64), Q (float64). */
} sptf_sample_type_t;

/** @brief Çok kanallı örneklerin bellekteki ve kablodaki yerleşimi. */
typedef enum
{
    SPTF_LAYOUT_INTERLEAVED = 0, /**< `s0c0 s0c1 … s1c0 s1c1 …` (ADC tarama + DMA çıktısı). Önerilen. */
    SPTF_LAYOUT_PLANAR      = 1  /**< `s0c0 s1c0 … s0c1 s1c1 …` (kanal blokları). */
} sptf_layout_t;

/**
 * @brief Kanal tanımı. Fiziksel değer = ham · @p scale + @p offset (karmaşıkta I ve Q'ya ayrı ayrı).
 *
 * Örnek: 12 bit ADC, Vref = 3.3 V → `scale = 3.3 / 4096.0`, `offset = 0.0`, `unit = "V"`.
 */
typedef struct
{
    const char *name;   /**< Kanal adı, UTF-8, en fazla ::SPTF_NAME_SIZE bayt (NULL → boş). */
    const char *unit;   /**< Fiziksel birim, UTF-8, en fazla ::SPTF_UNIT_SIZE bayt (NULL → boş). */
    double      scale;  /**< Ölçek (ham değer çarpanı). */
    double      offset; /**< Ofset (ölçekten sonra eklenir). */
} sptf_channel_t;

/**
 * @brief Akış yapılandırması.
 *
 * Bir akıştaki tüm kanallar aynı tipte ve aynı hızdadır; farklı hız / tip için ayrı akış açın.
 * @p channels dizisi kopyalanmaz: akış kullanıldığı sürece geçerli kalmalıdır (genelde `static const`).
 */
typedef struct
{
    uint16_t              stream_id;      /**< Akış numarası; aynı cihazda (IP adresinde) benzersiz olmalı. */
    const char           *name;           /**< Akış adı, UTF-8, en fazla ::SPTF_NAME_SIZE bayt (NULL → boş). */
    double                sample_rate_hz; /**< Kanal başına örnekleme hızı, [::SPTF_MIN_SAMPLE_RATE, ::SPTF_MAX_SAMPLE_RATE]. */
    sptf_sample_type_t    sample_type;    /**< Örnek tipi. */
    sptf_layout_t         layout;         /**< Bellek / kablo yerleşimi. */
    uint16_t              channel_count;  /**< Kanal sayısı, 1 … ::SPTF_MAX_CHANNELS. */
    const sptf_channel_t *channels;       /**< @p channel_count elemanlı kanal dizisi. */
    bool                  crc;            /**< true: her çerçevenin sonuna CRC-32 eklenir (::SPTF_CFG_CRC gerekir). */
} sptf_stream_config_t;

/** @} */

/**
 * @defgroup sptf_frame Çerçeve kodlayıcılar
 * @brief    Taşımadan bağımsız, durumsuz SPTF v1 çerçeve üreticileri.
 *
 * Bu fonksiyonlar ağ kullanmadan bir tampona tam çerçeve yazar. Kendi taşımanızı (UART köprüsü, SD kart
 * kaydı …) kullanıyorsanız doğrudan bunları çağırın; UDP için @ref sptf_link katmanı her şeyi otomatik yapar.
 * @{
 */

/** @brief Çerçeve başlığında akışa ait, her çerçevede değişen alanlar. */
typedef struct
{
    uint16_t stream_id;    /**< Akış numarası (yalnız MARKER'da; DESCRIPTOR / DATA'da `cfg->stream_id` yazılır). */
    uint16_t desc_rev;     /**< Yapılandırma revizyonu. */
    uint32_t seq;          /**< Akış başına çerçeve sayacı. */
    uint64_t timestamp_ns; /**< Cihaz zamanı (ns); DATA'da ilk örneğin zamanı; bilinmiyorsa 0. */
    bool     crc;          /**< true: sona CRC-32 eklenir. */
} sptf_frame_info_t;

/**
 * @brief Örnek tipinin bayt boyutu (karmaşıkta I+Q toplamı).
 * @param[in] type  Örnek tipi.
 * @return Bayt sayısı; tip geçersizse 0.
 */
uint32_t sptf_sample_size(sptf_sample_type_t type);

/**
 * @brief Akış yapılandırmasını protokol kurallarına göre doğrular.
 * @param[in] cfg  Doğrulanacak yapılandırma.
 * @retval SPTF_OK          Geçerli.
 * @retval SPTF_ERR_ARG     @p cfg NULL.
 * @retval SPTF_ERR_CONFIG  Hız, kanal sayısı, tip, yerleşim, ad uzunluğu ya da CRC seçimi geçersiz.
 */
sptf_status_t sptf_config_check(const sptf_stream_config_t *cfg);

/**
 * @brief DESCRIPTOR çerçevesinin bayt boyutu.
 * @param[in] channel_count  Kanal sayısı.
 * @param[in] crc            CRC eklenecek mi.
 * @return `32 + 48 + 64 · channel_count (+ 4)`.
 */
uint32_t sptf_descriptor_size(uint16_t channel_count, bool crc);

/**
 * @brief DESCRIPTOR çerçevesi yazar.
 * @param[out] buf      Hedef tampon.
 * @param[in]  cap      Tampon kapasitesi (bayt).
 * @param[in]  info     Başlık alanları; `crc` CRC'yi belirler (`stream_id` @p cfg'den alınır).
 * @param[in]  cfg      Akış yapılandırması (önceden ::sptf_config_check ile doğrulanmalı).
 * @param[out] out_len  Yazılan bayt sayısı.
 * @retval SPTF_OK, SPTF_ERR_ARG, SPTF_ERR_CONFIG, SPTF_ERR_SIZE
 */
sptf_status_t sptf_encode_descriptor(uint8_t *buf, size_t cap, const sptf_frame_info_t *info,
                                     const sptf_stream_config_t *cfg, size_t *out_len);

/**
 * @brief DATA çerçevesi yazar.
 *
 * @p samples, @p count örnek anının tüm kanallarını akışın yerleşiminde ve hedefin bellek düzeninde içerir:
 * interleaved için `count × channel_count` ardışık eleman, planar için kanal başına `count` elemanlık bloklar.
 *
 * @param[out] buf           Hedef tampon.
 * @param[in]  cap           Tampon kapasitesi (bayt).
 * @param[in]  info          Başlık alanları.
 * @param[in]  cfg           Akış yapılandırması.
 * @param[in]  first_sample  İlk örneğin akış içi indeksi (kanal başına, < 2^62).
 * @param[in]  samples       Örnekler.
 * @param[in]  count         Kanal başına örnek sayısı (≥ 1).
 * @param[out] out_len       Yazılan bayt sayısı.
 * @retval SPTF_OK, SPTF_ERR_ARG, SPTF_ERR_CONFIG, SPTF_ERR_SIZE
 */
sptf_status_t sptf_encode_data(uint8_t *buf, size_t cap, const sptf_frame_info_t *info,
                               const sptf_stream_config_t *cfg, uint64_t first_sample,
                               const void *samples, uint32_t count, size_t *out_len);

/**
 * @brief MARKER çerçevesi yazar.
 * @param[out] buf           Hedef tampon.
 * @param[in]  cap           Tampon kapasitesi (bayt).
 * @param[in]  info          Başlık alanları.
 * @param[in]  sample_index  İşaretin denk geldiği örnek indeksi.
 * @param[in]  code          Kullanıcı tanımlı olay kodu.
 * @param[in]  text          UTF-8 metin (NULL → boş), `\0` ile biten.
 * @param[out] out_len       Yazılan bayt sayısı.
 * @retval SPTF_OK, SPTF_ERR_ARG, SPTF_ERR_SIZE
 */
sptf_status_t sptf_encode_marker(uint8_t *buf, size_t cap, const sptf_frame_info_t *info,
                                 uint64_t sample_index, uint32_t code, const char *text, size_t *out_len);

#if (SPTF_CFG_CRC != 0) || defined(__DOXYGEN__)
/**
 * @brief CRC-32/ISO-HDLC'yi artımlı günceller.
 *
 * Başlangıç değeri 0'dır: `crc = sptf_crc32_update(0u, a, na); crc = sptf_crc32_update(crc, b, nb);`
 * Sonuç zlib `crc32()` ile aynıdır.
 * @param[in] crc   Önceki sonuç (ilk çağrıda 0).
 * @param[in] data  Veri.
 * @param[in] len   Bayt sayısı.
 * @return Güncellenmiş CRC.
 */
uint32_t sptf_crc32_update(uint32_t crc, const uint8_t *data, size_t len);
#endif

/** @} */

/**
 * @defgroup sptf_link UDP göndericisi
 * @brief    Bağlantı (taşıma + saat) ve akış nesneleri.
 *
 * Tipik kullanım:
 * @code
 * static sptf_link_t   link;
 * static sptf_stream_t adc;
 * static const sptf_channel_t ch[2] = { {"Ia", "A", 3.3 / 4096.0 / 0.1, 0.0}, {"Ib", "A", 3.3 / 4096.0 / 0.1, 0.0} };
 * static const sptf_stream_config_t adc_cfg = { 1u, "Motor akımları", 20000.0, SPTF_UINT16,
 *                                               SPTF_LAYOUT_INTERLEAVED, 2u, ch, false };
 *
 * sptf_link_config_t lc = { 0 };
 * sptf_lwip_raw_transport(&port, &lc.transport);   // ya da sptf_socket_transport()
 * lc.clock = board_time_ns;                          // isteğe bağlı
 * (void)sptf_link_init(&link, &lc);
 * (void)sptf_stream_init(&adc, &link, &adc_cfg);
 *
 * // DMA yarım / tam tamamlandığında (görev bağlamında):
 * (void)sptf_stream_write(&adc, half_buffer, 256u, SPTF_TS_AUTO);
 * @endcode
 * @{
 */

/**
 * @brief Taşıma katmanı arayüzü (port). Kütüphane ağ yığınını yalnız bu iki fonksiyonla kullanır.
 *
 * Her datagram için önce @p acquire, sonra tam olarak bir kez @p send çağrılır. @p acquire NULL dönerse
 * @p send çağrılmaz. Hazır portlar: `port/lwip_raw` (sıfır kopya, pbuf), `port/socket` (BSD / POSIX / Winsock /
 * lwIP socket).
 */
typedef struct
{
    /**
     * @brief En az @p len baytlık yazılabilir, bitişik tampon verir.
     * @param[in] ctx  Port bağlamı (::sptf_transport_t::ctx).
     * @param[in] len  Gereken bayt sayısı (≤ ::sptf_link_config_t::max_datagram).
     * @return Tampon; bellek yoksa NULL.
     */
    uint8_t *(*acquire)(void *ctx, size_t len);
    /**
     * @brief Tamponun ilk @p len baytını tek UDP datagramı olarak gönderir ve tamponu serbest bırakır.
     *
     * Gönderim başarısız olsa da tampon serbest bırakılmalıdır. @p len 0 ise hiçbir şey gönderilmez, tampon
     * yalnızca serbest bırakılır (kodlama hatasında kullanılır).
     * @param[in] ctx  Port bağlamı.
     * @param[in] buf  `acquire`'ın verdiği tampon.
     * @param[in] len  Datagram boyutu (0 → yalnız serbest bırak).
     * @return ::SPTF_OK, ::SPTF_ERR_IO ya da ::SPTF_ERR_BUSY.
     */
    sptf_status_t (*send)(void *ctx, uint8_t *buf, size_t len);
    void *ctx; /**< Port bağlamı; fonksiyonlara aynen geçirilir. */
} sptf_transport_t;

/**
 * @brief Monoton saat kancası.
 * @param[in] ctx  ::sptf_link_config_t::clock_ctx.
 * @return Cihaz zamanı (ns), monoton artan.
 */
typedef uint64_t (*sptf_clock_fn_t)(void *ctx);

/** @brief Bağlantı yapılandırması. Sıfırla doldurulmuş yapı geçerli varsayılanları seçer (taşıma hariç). */
typedef struct
{
    sptf_transport_t transport;             /**< Taşıma (zorunlu). */
    sptf_clock_fn_t  clock;                 /**< Saat (NULL: zaman damgası 0, DESCRIPTOR tekrarı örnek sayısına göre). */
    void            *clock_ctx;             /**< Saat bağlamı. */
    uint32_t         max_datagram;          /**< En büyük UDP yükü (bayt); 0 → ::SPTF_CFG_MAX_DATAGRAM; ≤ 65507. */
    uint32_t         descriptor_period_ms;  /**< DESCRIPTOR tekrar periyodu; 0 → ::SPTF_CFG_DESCRIPTOR_PERIOD_MS; ≤ 1000. */
} sptf_link_config_t;

/** @brief Bağlantı sayaçları (taşan sayaçlar sıfırdan devam eder). */
typedef struct
{
    uint32_t datagrams;   /**< Başarıyla gönderilen datagram. */
    uint64_t bytes;       /**< Başarıyla gönderilen bayt (UDP yükü). */
    uint32_t err_nomem;   /**< ::SPTF_ERR_NOMEM sayısı. */
    uint32_t err_io;      /**< ::SPTF_ERR_IO ve ::SPTF_ERR_BUSY sayısı. */
} sptf_link_stats_t;

/** @brief Akış sayaçları. */
typedef struct
{
    uint32_t data_frames;     /**< Gönderilen DATA çerçevesi. */
    uint32_t desc_frames;     /**< Gönderilen DESCRIPTOR çerçevesi. */
    uint32_t marker_frames;   /**< Gönderilen MARKER çerçevesi. */
    uint64_t samples_sent;    /**< Gönderilen örnek (kanal başına). */
    uint64_t samples_dropped; /**< Gönderilemeyen örnek (alıcıda NaN boşluğu olarak görünür). */
} sptf_stream_stats_t;

struct sptf_stream;

/** @brief Bağlantı nesnesi (bir hedef PC). Alanları doğrudan değiştirmeyin. */
typedef struct sptf_link
{
    sptf_link_config_t  cfg;      /**< Etkin yapılandırma (varsayılanlar çözülmüş). */
    struct sptf_stream *streams;  /**< Bağlı akışlar (tek yönlü liste). */
    sptf_link_stats_t   stats;    /**< Sayaçlar. */
    uint32_t            magic;    /**< Başlatılma işareti (iç kullanım). */
} sptf_link_t;

/** @brief Akış nesnesi. Alanları doğrudan değiştirmeyin; ::sptf_stream_init ile kurun. */
typedef struct sptf_stream
{
    sptf_stream_config_t cfg;            /**< Etkin yapılandırma (kanal dizisi işaretçi olarak). */
    sptf_link_t         *link;           /**< Bağlı olduğu bağlantı. */
    struct sptf_stream  *next;           /**< Listede sonraki akış. */
    uint64_t             next_sample;    /**< Sıradaki DATA çerçevesinin `first_sample` değeri. */
    uint64_t             desc_due;       /**< Sonraki DESCRIPTOR zamanı (ns) ya da örnek indeksi (saatsiz). */
    double               ns_per_sample;  /**< 1e9 / sample_rate_hz. */
    uint32_t             seq;            /**< Sıradaki çerçeve sayacı. */
    uint32_t             sample_bytes;   /**< Bir örnek anının tüm kanallarının bayt sayısı. */
    uint32_t             max_per_frame;  /**< Bir datagrama sığan en fazla örnek (kanal başına). */
    uint16_t             desc_rev;       /**< Yapılandırma revizyonu. */
    bool                 desc_pending;   /**< true: sıradaki işlemden önce DESCRIPTOR gönderilmeli. */
    sptf_stream_stats_t  stats;          /**< Sayaçlar. */
} sptf_stream_t;

/**
 * @brief Bağlantıyı başlatır.
 * @param[out] link  Bağlantı nesnesi (statik ya da kalıcı bellekte).
 * @param[in]  cfg   Yapılandırma (kopyalanır).
 * @retval SPTF_OK          Başarılı.
 * @retval SPTF_ERR_ARG     NULL argüman.
 * @retval SPTF_ERR_CONFIG  Taşıma fonksiyonları eksik, `max_datagram` < 128 ya da > 65507,
 *                          `descriptor_period_ms` > 1000.
 */
sptf_status_t sptf_link_init(sptf_link_t *link, const sptf_link_config_t *cfg);

/**
 * @brief Akışı doğrular, bağlantıya ekler ve ilk DESCRIPTOR'ı sıraya koyar.
 *
 * DESCRIPTOR, ilk ::sptf_stream_write ya da ::sptf_link_poll çağrısında gönderilir (ağ henüz hazır olmasa da
 * başlatma başarılı olur). `first_sample` 0'dan, `desc_rev` 1'den başlar.
 *
 * @param[out] stream  Akış nesnesi (statik ya da kalıcı bellekte).
 * @param[in]  link    Başlatılmış bağlantı.
 * @param[in]  cfg     Akış yapılandırması (kopyalanır; kanal dizisi işaretçi olarak tutulur).
 * @retval SPTF_OK          Başarılı.
 * @retval SPTF_ERR_ARG     NULL argüman ya da akış zaten bu bağlantıda.
 * @retval SPTF_ERR_STATE   Bağlantı başlatılmamış.
 * @retval SPTF_ERR_CONFIG  Yapılandırma geçersiz ya da aynı `stream_id` bağlantıda var.
 * @retval SPTF_ERR_SIZE    DESCRIPTOR ya da tek örnek anı `max_datagram`'a sığmıyor
 *                          (1472 B'de en fazla 21 kanal; daha fazlası için jumbo çerçeve ya da akışları bölün).
 */
sptf_status_t sptf_stream_init(sptf_stream_t *stream, sptf_link_t *link, const sptf_stream_config_t *cfg);

/**
 * @brief Örnek bloğunu gönderir; gerekirse datagram boyutuna göre birden fazla DATA çerçevesine böler.
 *
 * Önce DESCRIPTOR zamanı gelmişse (ilk çağrı, periyot doldu ya da yeniden yapılandırma) DESCRIPTOR gönderilir.
 * Gönderilemeyen DATA çerçevelerinin örnekleri kayıp sayılır; `first_sample` yine ilerletilir ki alıcı zaman
 * eksenini korusun ve boşluğu tam örnek sayısıyla göstersin.
 *
 * @param[in,out] stream        Akış.
 * @param[in]     samples       Akışın yerleşiminde örnekler (bkz. ::sptf_encode_data); hedefin bellek düzeninde.
 * @param[in]     count         Kanal başına örnek sayısı (≥ 1).
 * @param[in]     timestamp_ns  İlk örneğin zamanı (ns), ::SPTF_TS_NONE ya da ::SPTF_TS_AUTO.
 * @retval SPTF_OK                         Tüm çerçeveler gönderildi.
 * @retval SPTF_ERR_ARG, SPTF_ERR_STATE    Geçersiz çağrı (hiçbir şey gönderilmez, sayaç ilerlemez).
 * @retval SPTF_ERR_NOMEM, SPTF_ERR_IO, SPTF_ERR_BUSY  İlk başarısız gönderimin nedeni (kalan örnekler kayıp).
 */
sptf_status_t sptf_stream_write(sptf_stream_t *stream, const void *samples, uint32_t count, uint64_t timestamp_ns);

/**
 * @brief Olay işareti gönderir (ör. "ISR overrun", "state = RUN"). Arayüzde dikey çizgi + etiket olarak görünür.
 * @param[in,out] stream        Akış.
 * @param[in]     sample_index  Örnek indeksi ya da ::SPTF_MARKER_AT_NEXT (sıradaki örnek).
 * @param[in]     code          Kullanıcı tanımlı kod.
 * @param[in]     text          UTF-8 metin (NULL → boş).
 * @param[in]     timestamp_ns  Olay zamanı (ns), ::SPTF_TS_NONE ya da ::SPTF_TS_AUTO (şimdi).
 * @retval SPTF_OK, SPTF_ERR_ARG, SPTF_ERR_STATE, SPTF_ERR_SIZE, SPTF_ERR_NOMEM, SPTF_ERR_IO, SPTF_ERR_BUSY
 */
sptf_status_t sptf_stream_marker(sptf_stream_t *stream, uint64_t sample_index, uint32_t code, const char *text,
                                 uint64_t timestamp_ns);

/**
 * @brief Akışı yeni yapılandırmayla sürdürür: `desc_rev` artar, `first_sample` 0'a döner, DESCRIPTOR hemen
 *        sıraya girer. `stream_id` değiştirilemez.
 * @param[in,out] stream  Akış.
 * @param[in]     cfg     Yeni yapılandırma (aynı `stream_id`).
 * @retval SPTF_OK, SPTF_ERR_ARG, SPTF_ERR_STATE, SPTF_ERR_CONFIG, SPTF_ERR_SIZE
 */
sptf_status_t sptf_stream_reconfigure(sptf_stream_t *stream, const sptf_stream_config_t *cfg);

/**
 * @brief Akışı bağlantıdan çıkarır (nesne yeniden ::sptf_stream_init ile kullanılabilir).
 * @param[in,out] stream  Akış.
 * @retval SPTF_OK, SPTF_ERR_ARG, SPTF_ERR_STATE
 */
sptf_status_t sptf_stream_deinit(sptf_stream_t *stream);

/**
 * @brief Zamanı gelen DESCRIPTOR'ları gönderir. Veri akmadığı dönemlerde de akışın görünür kalması için
 *        ana döngüden ~100 ms'de bir çağırın (saat yoksa yalnız bekleyen ilk DESCRIPTOR'ları gönderir).
 * @param[in,out] link  Bağlantı.
 * @retval SPTF_OK                    Tümü gönderildi (ya da gönderilecek yok).
 * @retval SPTF_ERR_ARG, SPTF_ERR_STATE  Geçersiz çağrı.
 * @retval SPTF_ERR_NOMEM, SPTF_ERR_IO, SPTF_ERR_BUSY  İlk başarısız gönderimin nedeni (sonraki çağrıda yeniden denenir).
 */
sptf_status_t sptf_link_poll(sptf_link_t *link);

/**
 * @brief Bir datagrama sığan en fazla örnek sayısı (kanal başına). Blok boyunu bunun katı seçmek bölmeyi önler.
 * @param[in] stream  Başlatılmış akış.
 * @return Örnek sayısı; akış geçersizse 0.
 */
uint32_t sptf_stream_samples_per_frame(const sptf_stream_t *stream);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* SPTF_H */
