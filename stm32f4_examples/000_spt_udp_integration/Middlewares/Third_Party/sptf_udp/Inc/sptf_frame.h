/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    sptf_frame.h
 * @brief   SPTF v1 çerçeve biçimi: sabitler, akış tanımı ve çerçeve kodlayıcıları.
 *
 * SPTF (Signal Processing Toolbox Frame), gömülü bir cihazın örneklerini PC'deki
 * Signal Processing Toolbox arayüzüne taşıyan, kendini tanımlayan ikili çerçeve
 * biçimidir. Bayt düzeni docs/SPTF_Protokol_v1.md (SPT-PRT-001) ile, gönderici
 * kuralları docs/SPTF_UDP_Uygulama_Protokolu.md (SPT-PRT-002) ile tanımlıdır.
 *
 * Bu modül taşımadan bağımsızdır: çerçeveyi verilen tampona yazar, ağ yığını
 * kullanmaz. Çok baytlı alanlar küçük uçlu (little-endian) yazılır; tampon için
 * hizalama ve yapı dolgusu varsayılmaz. Fonksiyonlar dinamik bellek ve özyineleme
 * kullanmaz, durum tutmaz.
 *
 * Örnek verisi kablo düzeninde (küçük uçlu, IEEE 754) verilir ve olduğu gibi
 * kopyalanır. Cortex-M çekirdekli STM32 ailesinde belleğin düzeni kablo düzeniyle
 * aynıdır; ADC / DMA tamponları dönüştürülmeden gönderilebilir.
 *
 * Copyright (C) 2026, Kubilay Közleme <burhankozleme@gmail.com>
 */

#ifndef SPTF_FRAME_H
#define SPTF_FRAME_H

#include <stdbool.h>
#include <stdint.h>

/* ---------------------------------------------------------------- protokol */

/** Başlıktaki version alanı (SPTF v1). */
#define SPTF_PROTOCOL_VERSION           (1U)

/** Arayüzün varsayılan dinleme portu. */
#define SPTF_DEFAULT_UDP_PORT           (50505U)

/** IPv4 üzerinde UDP yükünün üst sınırı (bayt). */
#define SPTF_MAX_UDP_PAYLOAD            (65507U)

/** Standart Ethernet'te IP parçalanması olmadan gönderilebilen en büyük yük (bayt). */
#define SPTF_STANDARD_DATAGRAM_SIZE     (1472U)

/* ------------------------------------------------------------ çerçeve tipleri */

#define SPTF_FRAME_DESCRIPTOR           (1U)    /**< Akış tanımlayıcısı. */
#define SPTF_FRAME_DATA                 (2U)    /**< Örnek verisi. */
#define SPTF_FRAME_MARKER               (3U)    /**< Olay işareti. */

/** flags alanı bit 0: çerçevenin son 4 baytı CRC-32'dir. */
#define SPTF_FLAG_CRC32                 (0x0001U)

/* ---------------------------------------------------------- bölüm boyutları */

#define SPTF_HEADER_SIZE                (32U)   /**< Ortak çerçeve başlığı. */
#define SPTF_DESCRIPTOR_BODY_SIZE       (48U)   /**< DESCRIPTOR sabit gövdesi. */
#define SPTF_CHANNEL_RECORD_SIZE        (64U)   /**< Kanal başına ChannelInfo kaydı. */
#define SPTF_DATA_BODY_SIZE             (16U)   /**< DATA alt başlığı. */
#define SPTF_MARKER_BODY_SIZE           (16U)   /**< MARKER alt başlığı. */
#define SPTF_CRC_SIZE                   (4U)    /**< Çerçeve sonundaki CRC-32. */
#define SPTF_NAME_FIELD_SIZE            (32U)   /**< Akış ve kanal adı alanı. */
#define SPTF_UNIT_FIELD_SIZE            (16U)   /**< Birim alanı. */

/* ------------------------------------------------------------------ sınırlar */

/** Bir akıştaki en fazla kanal sayısı. */
#define SPTF_MAX_CHANNELS               (256U)

/** Örnekleme hızının alt sınırı (Hz). */
#define SPTF_MIN_SAMPLE_RATE_HZ         (1.0e-3)

/** Örnekleme hızının üst sınırı (Hz). */
#define SPTF_MAX_SAMPLE_RATE_HZ         (1.0e12)

/** first_sample alanı bu değerden küçük olmalıdır (2^62). */
#define SPTF_FIRST_SAMPLE_LIMIT         (0x4000000000000000ULL)

/** MARKER metninin en fazla bayt sayısı (text_len alanı 16 bittir). */
#define SPTF_MAX_MARKER_TEXT            (65535U)

/* ------------------------------------------------------------- örnek tipleri */

#define SPTF_SAMPLE_INT8                (0x01U) /**< int8, 1 bayt */
#define SPTF_SAMPLE_UINT8               (0x02U) /**< uint8, 1 bayt */
#define SPTF_SAMPLE_INT16               (0x03U) /**< int16, 2 bayt */
#define SPTF_SAMPLE_UINT16              (0x04U) /**< uint16, 2 bayt */
#define SPTF_SAMPLE_INT24               (0x05U) /**< int24 paketli, 3 bayt */
#define SPTF_SAMPLE_UINT24              (0x06U) /**< uint24 paketli, 3 bayt */
#define SPTF_SAMPLE_INT32               (0x07U) /**< int32, 4 bayt */
#define SPTF_SAMPLE_UINT32              (0x08U) /**< uint32, 4 bayt */
#define SPTF_SAMPLE_INT64               (0x09U) /**< int64, 8 bayt */
#define SPTF_SAMPLE_UINT64              (0x0AU) /**< uint64, 8 bayt */
#define SPTF_SAMPLE_FLOAT16             (0x10U) /**< IEEE 754 binary16, 2 bayt */
#define SPTF_SAMPLE_FLOAT32             (0x11U) /**< IEEE 754 binary32, 4 bayt */
#define SPTF_SAMPLE_FLOAT64             (0x12U) /**< IEEE 754 binary64, 8 bayt */
#define SPTF_SAMPLE_CINT16              (0x20U) /**< I, Q int16, 4 bayt */
#define SPTF_SAMPLE_CINT32              (0x21U) /**< I, Q int32, 8 bayt */
#define SPTF_SAMPLE_CFLOAT32            (0x22U) /**< I, Q float32, 8 bayt */
#define SPTF_SAMPLE_CFLOAT64            (0x23U) /**< I, Q float64, 16 bayt */

/* --------------------------------------------------------------- yerleşimler */

/** Örnek sıralı: s0c0 s0c1 ... s1c0 s1c1 ... (çok kanallı ADC + DMA çıktısı). */
#define SPTF_LAYOUT_INTERLEAVED         (0U)

/** Kanal sıralı: s0c0 s1c0 ... s0c1 s1c1 ... */
#define SPTF_LAYOUT_PLANAR              (1U)

/* ------------------------------------------------------------------- durumlar */

/** Kütüphane fonksiyonlarının dönüş değeri. */
typedef enum
{
    SPTF_STATUS_OK = 0,                 /**< İşlem tamamlandı. */
    SPTF_STATUS_NULL_POINTER,           /**< Zorunlu bir işaretçi NULL. */
    SPTF_STATUS_INVALID_ARGUMENT,       /**< Bağımsız değişken aralık dışında. */
    SPTF_STATUS_INVALID_RATE,           /**< Örnekleme hızı 0,001 Hz .. 10^12 Hz dışında. */
    SPTF_STATUS_INVALID_CHANNELS,       /**< Kanal sayısı 1 .. 256 dışında. */
    SPTF_STATUS_INVALID_SAMPLE_TYPE,    /**< Tanımsız örnek tipi. */
    SPTF_STATUS_INVALID_LAYOUT,         /**< Tanımsız yerleşim. */
    SPTF_STATUS_TEXT_TOO_LONG,          /**< Metin alanına sığmıyor. */
    SPTF_STATUS_NOT_FINITE,             /**< Ölçek ya da ofset sonlu değil (NaN, sonsuz). */
    SPTF_STATUS_BUFFER_TOO_SMALL,       /**< Çerçeve verilen tampona sığmıyor. */
    SPTF_STATUS_DATAGRAM_TOO_SMALL,     /**< DESCRIPTOR ya da tek örnek datagrama sığmıyor. */
    SPTF_STATUS_INVALID_ADDRESS,        /**< Hedef adres tekil yayın (unicast) değil. */
    SPTF_STATUS_DUPLICATE_STREAM_ID,    /**< stream_id başka bir açık akışta kullanılıyor. */
    SPTF_STATUS_NO_FREE_STREAM,         /**< Akış havuzu dolu. */
    SPTF_STATUS_NOT_OPEN,               /**< Akış açık değil. */
    SPTF_STATUS_NETWORK_ERROR,          /**< UDP denetim bloğu oluşturulamadı ya da bağlanamadı. */
    SPTF_STATUS_DESCRIPTOR_PENDING,     /**< İlk DESCRIPTOR henüz gönderilemedi. */
    SPTF_STATUS_NO_BUFFER,              /**< lwIP tampon (pbuf) veremedi. */
    SPTF_STATUS_SEND_FAILED,            /**< lwIP çerçeveyi gönderemedi. */
    SPTF_STATUS_COUNTER_LIMIT           /**< first_sample 2^62 sınırına ulaşacak. */
} sptf_status_t;

/* -------------------------------------------------------------------- tipler */

/**
 * 64 bitlik kayan noktalı tip (MISRA C:2012 Yönerge 4.6). DESCRIPTOR'daki f64 alanları
 * bu tiple verilir. Tipin IEEE 754 binary64 olduğu sptf_frame.c içinde derleme
 * zamanında denetlenir.
 */
typedef double sptf_float64_t;

/* ------------------------------------------------------------------ yapılar */

/**
 * Bir kanalın tanımı (DESCRIPTOR ChannelInfo kaydı).
 *
 * Fiziksel değer = ham * scale + offset. Karmaşık tiplerde I ve Q bileşenlerine
 * ayrı ayrı uygulanır. Metinler UTF-8'dir ve '\0' ile biter.
 */
typedef struct
{
    const char *name;               /**< Kanal adı, en fazla 32 bayt. */
    const char *unit;               /**< Fiziksel birim, en fazla 16 bayt ("" olabilir). */
    sptf_float64_t scale;           /**< Ölçek, sonlu. */
    sptf_float64_t offset;          /**< Ofset, sonlu. */
} sptf_channel_t;

/**
 * Bir akışın DESCRIPTOR içeriği.
 *
 * Bütün kanallar aynı tipte ve aynı hızdadır. Yapı işaretçileri kopyalanarak
 * saklanır; işaret edilen metinler ve kanal dizisi akış açık kaldığı sürece
 * geçerli kalmalıdır (tercihen const ve statik).
 */
typedef struct
{
    const char *name;               /**< Akış adı, en fazla 32 bayt. */
    const sptf_channel_t *channels; /**< channel_count elemanlı kanal dizisi. */
    sptf_float64_t sample_rate_hz;  /**< Kanal başına örnekleme hızı. */
    uint16_t channel_count;         /**< 1 .. 256. */
    uint8_t sample_type;            /**< SPTF_SAMPLE_xxx. */
    uint8_t layout;                 /**< SPTF_LAYOUT_xxx. */
} sptf_stream_desc_t;

/** Ortak başlığın çerçeveden çerçeveye değişen alanları. */
typedef struct
{
    uint64_t timestamp_ns;          /**< Cihaz zamanı (ns); bilinmiyorsa 0. */
    uint32_t seq;                   /**< Akış başına çerçeve sayacı. */
    uint16_t stream_id;             /**< Akış numarası. */
    uint16_t desc_rev;              /**< Yapılandırma revizyonu. */
    bool crc_enabled;               /**< true: çerçeveye CRC-32 eklenir. */
} sptf_frame_header_t;

/** Bir DATA çerçevesine konacak örnek aralığı. */
typedef struct
{
    const uint8_t *block;           /**< Örnek bloğu (kablo düzeninde). */
    uint64_t first_sample;          /**< İlk örneğin akış eksenindeki indeksi. */
    uint32_t block_length;          /**< Bloktaki örnek anı sayısı (planar adımı). */
    uint32_t offset;                /**< Çerçevenin ilk örneğinin blok içindeki indeksi. */
    uint32_t count;                 /**< Çerçevedeki örnek anı sayısı (en az 1). */
} sptf_data_slice_t;

/* ------------------------------------------------------------------ boyutlar */

/**
 * @brief  Bir örnek tipinin bayt boyutu.
 * @param  sample_type  SPTF_SAMPLE_xxx.
 * @return Bayt sayısı; tanımsız tip için 0.
 */
uint32_t sptf_sample_type_size(uint8_t sample_type);

/**
 * @brief  Bir örnek anındaki tüm kanalların toplam bayt sayısı.
 * @param  desc  Akış tanımı.
 * @return Bayt sayısı; desc NULL ya da tip / kanal sayısı geçersizse 0.
 */
uint32_t sptf_bytes_per_instant(const sptf_stream_desc_t *desc);

/**
 * @brief  DESCRIPTOR çerçevesinin bayt boyutu.
 * @param  channel_count  Kanal sayısı.
 * @param  crc_enabled    CRC-32 eklenecek mi.
 * @return 32 + 48 + 64 * channel_count (+ 4).
 */
uint32_t sptf_descriptor_frame_size(uint16_t channel_count, bool crc_enabled);

/**
 * @brief  DATA çerçevesinin bayt boyutu.
 * @param  bytes_per_instant  Örnek anı başına bayt (sptf_bytes_per_instant()).
 * @param  sample_count       Örnek anı sayısı.
 * @param  crc_enabled        CRC-32 eklenecek mi.
 * @return Bayt sayısı; 32 bite sığmıyorsa ya da bytes_per_instant 0 ise 0.
 */
uint32_t sptf_data_frame_size(uint32_t bytes_per_instant, uint32_t sample_count, bool crc_enabled);

/**
 * @brief  Verilen çerçeve boyutuna sığan en fazla örnek anı sayısı.
 * @param  bytes_per_instant  Örnek anı başına bayt.
 * @param  frame_size_limit   Çerçevenin üst sınırı (bayt), ör. datagram boyutu.
 * @param  crc_enabled        CRC-32 eklenecek mi.
 * @return Örnek anı sayısı; tek örnek bile sığmıyorsa 0.
 */
uint32_t sptf_max_samples_per_frame(uint32_t bytes_per_instant, uint32_t frame_size_limit, bool crc_enabled);

/* ------------------------------------------------------------------ doğrulama */

/**
 * @brief  Akış tanımını SPT-PRT-002 [G-32] kurallarına göre doğrular.
 *
 * Örnekleme hızı 0,001 Hz .. 10^12 Hz, kanal sayısı 1 .. 256, tip ve yerleşim tanımlı,
 * metinler alanlarına sığıyor, ölçek ve ofset sonlu olmalıdır.
 *
 * @param  desc  Akış tanımı.
 * @return SPTF_STATUS_OK ya da ilk bulunan hatanın kodu.
 */
sptf_status_t sptf_check_stream_desc(const sptf_stream_desc_t *desc);

/* ----------------------------------------------------------------- kodlayıcılar */

/**
 * @brief  DESCRIPTOR çerçevesi üretir.
 * @param  header       Başlık alanları.
 * @param  desc         Akış tanımı (sptf_check_stream_desc() ile doğrulanır).
 * @param  buffer       Çerçevenin yazılacağı tampon.
 * @param  buffer_size  Tamponun bayt boyutu.
 * @param  frame_size   Başarıda çerçevenin bayt boyutu yazılır.
 * @return SPTF_STATUS_OK ya da hata kodu; hata durumunda tampon değişmemiş olabilir.
 */
sptf_status_t sptf_encode_descriptor(const sptf_frame_header_t *header, const sptf_stream_desc_t *desc,
                                     uint8_t *buffer, uint32_t buffer_size, uint32_t *frame_size);

/**
 * @brief  DATA çerçevesi üretir.
 *
 * Örnekler slice->block içinden alınır: interleaved yerleşimde
 * [offset, offset + count) örnek anları ardışık kopyalanır; planar yerleşimde
 * her kanalın bloğundan (block_length örnek adımlı) aynı aralık alınır.
 *
 * @param  header       Başlık alanları.
 * @param  desc         Akış tanımı (tip, kanal sayısı ve yerleşim kullanılır).
 * @param  slice        Örnek aralığı.
 * @param  buffer       Çerçevenin yazılacağı tampon. slice->block ile örtüşmemelidir.
 * @param  buffer_size  Tamponun bayt boyutu.
 * @param  frame_size   Başarıda çerçevenin bayt boyutu yazılır.
 * @return SPTF_STATUS_OK ya da hata kodu.
 */
sptf_status_t sptf_encode_data(const sptf_frame_header_t *header, const sptf_stream_desc_t *desc,
                               const sptf_data_slice_t *slice, uint8_t *buffer, uint32_t buffer_size,
                               uint32_t *frame_size);

/**
 * @brief  MARKER çerçevesi üretir.
 * @param  header        Başlık alanları.
 * @param  sample_index  Olayın denk geldiği örnek indeksi.
 * @param  code          Kullanıcı tanımlı olay kodu.
 * @param  text          '\0' ile biten UTF-8 metin ("" olabilir).
 * @param  buffer        Çerçevenin yazılacağı tampon.
 * @param  buffer_size   Tamponun bayt boyutu.
 * @param  frame_size    Başarıda çerçevenin bayt boyutu yazılır.
 * @return SPTF_STATUS_OK ya da hata kodu.
 */
sptf_status_t sptf_encode_marker(const sptf_frame_header_t *header, uint64_t sample_index, uint32_t code,
                                 const char *text, uint8_t *buffer, uint32_t buffer_size, uint32_t *frame_size);

#endif /* SPTF_FRAME_H */
