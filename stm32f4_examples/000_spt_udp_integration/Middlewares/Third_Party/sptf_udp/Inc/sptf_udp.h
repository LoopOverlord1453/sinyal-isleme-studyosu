/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    sptf_udp.h
 * @brief   SPTF akışlarını lwIP raw API ile UDP üzerinden gönderen gönderici.
 *
 * Kütüphane SPT-PRT-002'deki Sınıf B gönderici kurallarını uygular:
 *
 *   - Akışın ilk çerçevesi DESCRIPTOR'dır; DESCRIPTOR gönderilmeden DATA ve MARKER
 *     gönderilmez ([G-09]).
 *   - DESCRIPTOR, ayarlanan aralıkla (varsayılan 500 ms) tekrarlanır ([G-11], [G-12]).
 *     Aralık örnek sayısıyla ölçülür; zaman damgası verilirse süreyle de ölçülür.
 *   - Örnek blokları datagrama sığacak DATA çerçevelerine bölünür; her datagram tek
 *     çerçeve taşır ([G-05], [G-06]).
 *   - seq her çerçevede, first_sample her örnekte ilerler. Tampon alınamaz ya da
 *     gönderim başarısız olursa örnekler kayıp sayılır ve first_sample yine ilerler
 *     ([G-17], [G-18], [G-29], [G-30]).
 *   - Yeniden yapılandırmada desc_rev artar, first_sample 0'dan başlar ([G-14], [G-15]).
 *
 * Çağrı bağlamı: Fonksiyonlar kesme (ISR) içinden çağrılmamalıdır ([G-28]).
 * FreeRTOS ile lwIP çağrıları LOCK_TCPIP_CORE altında yapılır (sptf_config.h).
 * Aynı akış nesnesi aynı anda tek bir görevden kullanılmalıdır; farklı akışlar farklı
 * görevlerden kullanılabilir. sptf_udp_open() ve sptf_udp_close() başlatma sırasında,
 * tek bir görevden çağrılmalıdır.
 *
 * Copyright (C) 2026, Kubilay Közleme <burhankozleme@gmail.com>
 */

#ifndef SPTF_UDP_H
#define SPTF_UDP_H

#include <stdbool.h>
#include <stdint.h>

#include "sptf_config.h"
#include "sptf_frame.h"

/** Varsayılan DESCRIPTOR tekrar aralığı (ms). */
#define SPTF_DEFAULT_DESCRIPTOR_INTERVAL_MS     (500U)

/** DESCRIPTOR tekrar aralığının üst sınırı (ms), SPT-PRT-002 [G-11]. */
#define SPTF_MAX_DESCRIPTOR_INTERVAL_MS         (1000U)

/** Akışın ilk desc_rev değeri (referans gerçekleme). */
#define SPTF_DEFAULT_DESC_REV                   (1U)

/** Kütüphane içindeki akış nesnesi; alanlarına yalnız kütüphane erişir. */
typedef struct sptf_udp_stream sptf_udp_stream_t;

/** Akış açma ayarları. Varsayılanlar sptf_udp_config_init() ile yüklenir. */
typedef struct
{
    sptf_stream_desc_t desc;            /**< DESCRIPTOR içeriği. */
    uint8_t remote_ip[4];               /**< Alıcının IPv4 adresi, ör. {192, 168, 1, 100}. */
    uint16_t remote_port;               /**< Alıcının portu (varsayılan 50505). */
    uint16_t local_port;                /**< Kaynak port; 0 ise lwIP geçici port verir. */
    uint16_t stream_id;                 /**< Cihazdaki akışlar arasında benzersiz numara. */
    uint16_t initial_desc_rev;          /**< İlk desc_rev (varsayılan 1). */
    uint16_t max_datagram_size;         /**< Datagram üst sınırı, <= SPTF_CFG_MAX_DATAGRAM_SIZE. */
    uint16_t descriptor_interval_ms;    /**< DESCRIPTOR tekrar aralığı, 1 .. 1000 ms. */
    bool crc_enabled;                   /**< true: her çerçeveye CRC-32 eklenir. */
} sptf_udp_config_t;

/** Akışın sayaçları ve anlık durumu. */
typedef struct
{
    uint64_t next_first_sample;         /**< Bir sonraki DATA'nın first_sample değeri. */
    uint64_t samples_sent;              /**< Gönderilen örnek anı sayısı. */
    uint64_t samples_lost;              /**< Gönderilemeyip kayıp sayılan örnek anı sayısı. */
    uint64_t bytes_sent;                /**< Gönderilen çerçeve baytı (UDP yükü). */
    uint32_t frames_sent;               /**< Gönderilen çerçeve sayısı (tüm tipler). */
    uint32_t descriptors_sent;          /**< Gönderilen DESCRIPTOR sayısı. */
    uint32_t markers_sent;              /**< Gönderilen MARKER sayısı. */
    uint32_t buffer_errors;             /**< lwIP'nin tampon veremediği çerçeve sayısı. */
    uint32_t send_errors;               /**< lwIP'nin gönderemediği çerçeve sayısı. */
    uint32_t next_seq;                  /**< Bir sonraki çerçevenin seq değeri. */
    uint16_t desc_rev;                  /**< Geçerli yapılandırma revizyonu. */
    bool described;                     /**< true: DESCRIPTOR en az bir kez gönderildi. */
} sptf_udp_stats_t;

/**
 * @brief  Ayarları varsayılanlarla doldurur.
 *
 * Varsayılanlar: port 50505, kaynak port 0, desc_rev 1, datagram
 * SPTF_CFG_MAX_DATAGRAM_SIZE, DESCRIPTOR aralığı 500 ms, CRC kapalı. Akış tanımı,
 * hedef adres ve stream_id uygulama tarafından doldurulur.
 *
 * @param  config  Doldurulacak ayar yapısı.
 * @return SPTF_STATUS_OK ya da SPTF_STATUS_NULL_POINTER.
 */
sptf_status_t sptf_udp_config_init(sptf_udp_config_t *config);

/**
 * @brief  Akış açar: ayarları doğrular, UDP denetim bloğunu oluşturup kaynak porta bağlar.
 *
 * İlk DESCRIPTOR, ilk sptf_udp_send_samples(), sptf_udp_send_marker() ya da
 * sptf_udp_send_descriptor() çağrısında gönderilir.
 *
 * @param  config  Akış ayarları; yapı kopyalanır, işaret ettiği metinler kopyalanmaz.
 * @param  stream  Başarıda akış nesnesinin adresi yazılır.
 * @return SPTF_STATUS_OK ya da hata kodu.
 */
sptf_status_t sptf_udp_open(const sptf_udp_config_t *config, sptf_udp_stream_t **stream);

/**
 * @brief  Akışı kapatır ve UDP denetim bloğunu bırakır.
 * @param  stream  Açık akış.
 * @return SPTF_STATUS_OK ya da hata kodu.
 */
sptf_status_t sptf_udp_close(sptf_udp_stream_t *stream);

/**
 * @brief  Bir örnek bloğu gönderir.
 *
 * Gerekirse önce DESCRIPTOR gönderilir. Blok, datagrama sığan DATA çerçevelerine
 * bölünür. timestamp_ns 0 değilse her çerçevenin zaman damgası, bloğun ilk örneğinin
 * zamanına çerçevenin blok içindeki konumu eklenerek hesaplanır.
 *
 * Blok kablo düzenindedir: interleaved'da sample_count * kanal örnek, planar'da her
 * kanal için sample_count örnekten oluşan ardışık kanal blokları. Örnek dizisi
 * gönderimden sonra yeniden kullanılabilir (veri lwIP tamponuna kopyalanır).
 *
 * @param  stream        Açık akış.
 * @param  samples       Örnek bloğu, ör. (const uint8_t *)adc_buffer.
 * @param  sample_count  Bloktaki örnek anı sayısı (kanal başına), en az 1.
 * @param  timestamp_ns  Bloğun ilk örneğinin cihaz zamanı (ns); bilinmiyorsa 0.
 * @return SPTF_STATUS_OK: bütün çerçeveler gönderildi.
 *         SPTF_STATUS_DESCRIPTOR_PENDING: ilk DESCRIPTOR gönderilemedi, blok kayıp sayıldı.
 *         SPTF_STATUS_NO_BUFFER / SPTF_STATUS_SEND_FAILED: en az bir çerçeve gönderilemedi;
 *         ilgili örnekler kayıp sayıldı, akış sürer.
 *         Diğer kodlar: bağımsız değişken hatası, hiçbir şey gönderilmedi.
 */
sptf_status_t sptf_udp_send_samples(sptf_udp_stream_t *stream, const uint8_t *samples, uint32_t sample_count,
                                    uint64_t timestamp_ns);

/**
 * @brief  MARKER (olay işareti) gönderir.
 * @param  stream        Açık akış.
 * @param  sample_index  Olayın denk geldiği örnek indeksi (ör. stats.next_first_sample).
 * @param  code          Kullanıcı tanımlı olay kodu.
 * @param  text          '\0' ile biten UTF-8 metin; datagrama sığmalıdır.
 * @param  timestamp_ns  Olayın cihaz zamanı (ns); bilinmiyorsa 0.
 * @return SPTF_STATUS_OK ya da hata kodu.
 */
sptf_status_t sptf_udp_send_marker(sptf_udp_stream_t *stream, uint64_t sample_index, uint32_t code,
                                   const char *text, uint64_t timestamp_ns);

/**
 * @brief  DESCRIPTOR'ı hemen gönderir ve tekrar sayacını yeniden başlatır.
 * @param  stream        Açık akış.
 * @param  timestamp_ns  Gönderim zamanı (ns); bilinmiyorsa 0.
 * @return SPTF_STATUS_OK ya da hata kodu.
 */
sptf_status_t sptf_udp_send_descriptor(sptf_udp_stream_t *stream, uint64_t timestamp_ns);

/**
 * @brief  Akışı yeni bir tanımla yeniden yapılandırır.
 *
 * desc_rev 1 artar (65535'ten sonra 0), first_sample 0'a döner ve yeni DESCRIPTOR
 * hemen gönderilir. Gönderim başarısızsa bir sonraki çağrıda yeniden denenir.
 *
 * @param  stream  Açık akış.
 * @param  desc    Yeni akış tanımı; yapı kopyalanır.
 * @return SPTF_STATUS_OK ya da hata kodu. Doğrulama hatasında akış değişmez.
 */
sptf_status_t sptf_udp_reconfigure(sptf_udp_stream_t *stream, const sptf_stream_desc_t *desc);

/**
 * @brief  Akışın sayaçlarını okur.
 * @param  stream  Açık akış.
 * @param  stats   Sayaçların yazılacağı yapı.
 * @return SPTF_STATUS_OK ya da hata kodu.
 */
sptf_status_t sptf_udp_get_stats(const sptf_udp_stream_t *stream, sptf_udp_stats_t *stats);

#endif /* SPTF_UDP_H */
