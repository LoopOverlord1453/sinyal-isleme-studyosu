/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    sptf_udp.c
 * @brief   SPTF akış göndericisi: lwIP raw API ile UDP.
 *
 * Her akış, kütüphane içindeki statik havuzdan bir nesne alır. Nesne kendi çerçeve
 * tamponunu taşır: çerçeve önce bu tampona kodlanır, sonra pbuf_take() ile lwIP
 * tamponuna kopyalanıp udp_sendto() ile gönderilir. lwIP tamponu gönderimden sonra
 * hemen bırakılır; sürücü sıfır kopyalı çalışsa bile akışın tamponu bir sonraki
 * çerçeve için güvenle yeniden kullanılır.
 *
 * Copyright (C) 2026, Kubilay Közleme <burhankozleme@gmail.com>
 */

#include "sptf_udp.h"

#include <stddef.h>
#include <string.h>

#include "lwip/opt.h"
#include "lwip/def.h"
#include "lwip/err.h"
#include "lwip/ip_addr.h"
#include "lwip/pbuf.h"
#include "lwip/udp.h"

/* lwIP çekirdek kilidi: ayarlanmadıysa lwIP yapılandırmasından seçilir. */
#ifndef SPTF_CFG_LWIP_CORE_LOCKING
#if (NO_SYS == 0) && (LWIP_TCPIP_CORE_LOCKING != 0)
#define SPTF_CFG_LWIP_CORE_LOCKING      (1U)
#else
#define SPTF_CFG_LWIP_CORE_LOCKING      (0U)
#endif
#endif

#if (SPTF_CFG_LWIP_CORE_LOCKING != 0U)
#if (NO_SYS != 0) || (LWIP_TCPIP_CORE_LOCKING == 0)
#error "SPTF_CFG_LWIP_CORE_LOCKING requires NO_SYS = 0 and LWIP_TCPIP_CORE_LOCKING = 1"
#endif
#include "lwip/tcpip.h"
#endif

#if (LWIP_UDP == 0)
#error "lwIP must be built with LWIP_UDP = 1"
#endif

/* ------------------------------------------------------------- akış durumları */

#define STATE_FREE                  (0U)    /* havuz yuvası boş */
#define STATE_DESCRIBING            (1U)    /* ilk DESCRIPTOR bekleniyor */
#define STATE_STREAMING             (2U)    /* DESCRIPTOR gönderildi, akış yayında */

/* ------------------------------------------------------------------ sabitler */

#define NANOSECONDS_PER_SECOND      (1.0e9)
#define NANOSECONDS_PER_MILLISECOND (1000000U)
#define MILLISECONDS_PER_SECOND     (1000.0)

/* Zaman damgası hesabında güvenle tamsayıya çevrilebilen en büyük gecikme (ns, < 2^63). */
#define MAX_FRAME_DELAY_NS          (9.0e18)

/** Akış nesnesi. Alanlarına yalnız bu dosya erişir. */
struct sptf_udp_stream
{
    sptf_stream_desc_t desc;                /* geçerli DESCRIPTOR içeriği */
    ip_addr_t remote_ip;                    /* alıcı adresi (ağ bayt sırası) */
    struct udp_pcb *pcb;                    /* lwIP UDP denetim bloğu */
    uint64_t first_sample;                  /* sonraki DATA'nın first_sample değeri */
    uint64_t next_descriptor_sample;        /* örnek sayısıyla bir sonraki DESCRIPTOR zamanı */
    uint64_t descriptor_period_samples;     /* DESCRIPTOR aralığının örnek karşılığı */
    uint64_t descriptor_interval_ns;        /* DESCRIPTOR aralığının ns karşılığı */
    uint64_t last_descriptor_ns;            /* son DESCRIPTOR'ın zaman damgası */
    uint64_t samples_sent;
    uint64_t samples_lost;
    uint64_t bytes_sent;
    sptf_float64_t ns_per_sample;           /* 1e9 / örnekleme hızı */
    uint32_t seq;                           /* sonraki çerçevenin seq değeri */
    uint32_t bytes_per_instant;             /* örnek anı başına bayt */
    uint32_t max_samples_per_frame;         /* bir DATA çerçevesine sığan örnek anı */
    uint32_t frames_sent;
    uint32_t descriptors_sent;
    uint32_t markers_sent;
    uint32_t buffer_errors;
    uint32_t send_errors;
    uint16_t stream_id;
    uint16_t desc_rev;
    uint16_t remote_port;
    uint16_t max_datagram_size;
    uint16_t descriptor_interval_ms;
    uint8_t state;
    bool crc_enabled;
    bool descriptor_retry;                  /* başarısız DESCRIPTOR ilk fırsatta yinelenir */
    bool described;                         /* DESCRIPTOR en az bir kez gönderildi */
    uint8_t frame[SPTF_CFG_MAX_DATAGRAM_SIZE];
};

/** Akış nesneleri havuzu (statik ömürlü, sıfırla başlar). */
static sptf_udp_stream_t stream_pool[SPTF_CFG_MAX_STREAMS];

/* ======================================================= havuz ve doğrulama */

/** İşaretçinin havuzdaki açık bir akışı gösterip göstermediğini sınar. */
static sptf_status_t check_stream(const sptf_udp_stream_t *stream)
{
    sptf_status_t status = SPTF_STATUS_NOT_OPEN;
    uint32_t index;

    if (stream == NULL)
    {
        status = SPTF_STATUS_NULL_POINTER;
    }
    else
    {
        for (index = 0U; index < SPTF_CFG_MAX_STREAMS; index++)
        {
            if ((stream == &stream_pool[index]) && (stream_pool[index].state != STATE_FREE))
            {
                status = SPTF_STATUS_OK;
            }
        }
    }

    return status;
}

static bool stream_id_in_use(uint16_t stream_id)
{
    bool in_use = false;
    uint32_t index;

    for (index = 0U; index < SPTF_CFG_MAX_STREAMS; index++)
    {
        if ((stream_pool[index].state != STATE_FREE) && (stream_pool[index].stream_id == stream_id))
        {
            in_use = true;
        }
    }

    return in_use;
}

static sptf_udp_stream_t *find_free_slot(void)
{
    sptf_udp_stream_t *slot = NULL;
    uint32_t index = 0U;

    while ((slot == NULL) && (index < SPTF_CFG_MAX_STREAMS))
    {
        if (stream_pool[index].state == STATE_FREE)
        {
            slot = &stream_pool[index];
        }

        index++;
    }

    return slot;
}

/**
 * Hedefin tekil yayın adresi olup olmadığını sınar ([G-01]): 0.0.0.0/8, çoklu yayın
 * (224.0.0.0/4) ve 240.0.0.0 üstü (sınırlı yayın dahil) reddedilir. Alt ağ yayın
 * adresi ağ maskesi bilinmeden ayırt edilemez; uygulama bunu kullanmamalıdır.
 */
static bool is_unicast(const uint8_t *address)
{
    bool unicast = true;

    if ((address[0] == 0U) || (address[0] >= 224U))
    {
        unicast = false;
    }

    return unicast;
}

/** DESCRIPTOR'ın ve en az bir örnek anının datagrama sığdığını sınar ([G-08]). */
static sptf_status_t check_fits(const sptf_stream_desc_t *desc, uint16_t max_datagram_size, bool crc_enabled)
{
    sptf_status_t status = SPTF_STATUS_OK;
    const uint32_t limit = (uint32_t)max_datagram_size;
    const uint32_t bytes_per_instant = sptf_bytes_per_instant(desc);

    if (sptf_descriptor_frame_size(desc->channel_count, crc_enabled) > limit)
    {
        status = SPTF_STATUS_DATAGRAM_TOO_SMALL;
    }
    else if (sptf_max_samples_per_frame(bytes_per_instant, limit, crc_enabled) == 0U)
    {
        status = SPTF_STATUS_DATAGRAM_TOO_SMALL;
    }
    else
    {
        status = SPTF_STATUS_OK;
    }

    return status;
}

static sptf_status_t check_config(const sptf_udp_config_t *config)
{
    sptf_status_t status = sptf_check_stream_desc(&config->desc);

    if (status == SPTF_STATUS_OK)
    {
        if (!is_unicast(config->remote_ip))
        {
            status = SPTF_STATUS_INVALID_ADDRESS;
        }
        else if (config->remote_port == 0U)
        {
            status = SPTF_STATUS_INVALID_ARGUMENT;
        }
        else if ((config->descriptor_interval_ms < 1U)
                 || (config->descriptor_interval_ms > SPTF_MAX_DESCRIPTOR_INTERVAL_MS))
        {
            status = SPTF_STATUS_INVALID_ARGUMENT;
        }
        else if ((uint32_t)config->max_datagram_size > SPTF_CFG_MAX_DATAGRAM_SIZE)
        {
            status = SPTF_STATUS_INVALID_ARGUMENT;
        }
        else
        {
            status = check_fits(&config->desc, config->max_datagram_size, config->crc_enabled);
        }
    }

    return status;
}

/** Tanımdan türetilen değerleri (çerçeve kapasitesi, DESCRIPTOR periyodu) hesaplar. */
static void update_derived_values(sptf_udp_stream_t *stream)
{
    const sptf_float64_t rate = stream->desc.sample_rate_hz;
    const sptf_float64_t period = (rate * (sptf_float64_t)stream->descriptor_interval_ms)
                                  / MILLISECONDS_PER_SECOND;

    stream->bytes_per_instant = sptf_bytes_per_instant(&stream->desc);
    stream->max_samples_per_frame = sptf_max_samples_per_frame(stream->bytes_per_instant,
                                                               (uint32_t)stream->max_datagram_size,
                                                               stream->crc_enabled);
    stream->ns_per_sample = NANOSECONDS_PER_SECOND / rate;
    stream->descriptor_interval_ns = (uint64_t)stream->descriptor_interval_ms * NANOSECONDS_PER_MILLISECOND;

    /* Hız 10^12 Hz ile sınırlı olduğu için periyot 10^12 örneği aşmaz; dönüşüm tanımlıdır. */
    if (period >= 1.0)
    {
        stream->descriptor_period_samples = (uint64_t)period;
    }
    else
    {
        stream->descriptor_period_samples = 1U;
    }
}

/* =============================================================== lwIP erişimi */

static sptf_status_t create_pcb(sptf_udp_stream_t *stream, uint16_t local_port)
{
    sptf_status_t status = SPTF_STATUS_OK;
    struct udp_pcb *pcb;

#if (SPTF_CFG_LWIP_CORE_LOCKING != 0U)
    LOCK_TCPIP_CORE();
#endif

    pcb = udp_new();

    if (pcb == NULL)
    {
        status = SPTF_STATUS_NETWORK_ERROR;
    }
    else
    {
        /* Kaynak port oturum boyunca sabit kalır ([G-03]). */
        const err_t result = udp_bind(pcb, IP_ADDR_ANY, local_port);

        if (result == (err_t)ERR_OK)
        {
            stream->pcb = pcb;
        }
        else
        {
            udp_remove(pcb);
            status = SPTF_STATUS_NETWORK_ERROR;
        }
    }

#if (SPTF_CFG_LWIP_CORE_LOCKING != 0U)
    UNLOCK_TCPIP_CORE();
#endif

    return status;
}

static void remove_pcb(struct udp_pcb *pcb)
{
#if (SPTF_CFG_LWIP_CORE_LOCKING != 0U)
    LOCK_TCPIP_CORE();
#endif

    udp_remove(pcb);

#if (SPTF_CFG_LWIP_CORE_LOCKING != 0U)
    UNLOCK_TCPIP_CORE();
#endif
}

/**
 * Akış tamponundaki çerçeveyi tek bir datagram olarak gönderir.
 * @return SPTF_STATUS_OK, SPTF_STATUS_NO_BUFFER ya da SPTF_STATUS_SEND_FAILED.
 */
static sptf_status_t transmit_frame(sptf_udp_stream_t *stream, uint32_t frame_size)
{
    sptf_status_t status = SPTF_STATUS_OK;
    struct pbuf *packet;

#if (SPTF_CFG_LWIP_CORE_LOCKING != 0U)
    LOCK_TCPIP_CORE();
#endif

    packet = pbuf_alloc(PBUF_TRANSPORT, (u16_t)frame_size, PBUF_RAM);

    if (packet == NULL)
    {
        status = SPTF_STATUS_NO_BUFFER;
    }
    else
    {
        err_t result = pbuf_take(packet, stream->frame, (u16_t)frame_size);

        if (result == (err_t)ERR_OK)
        {
            result = udp_sendto(stream->pcb, packet, &stream->remote_ip, stream->remote_port);
        }

        if (result != (err_t)ERR_OK)
        {
            status = SPTF_STATUS_SEND_FAILED;
        }

        (void)pbuf_free(packet);
    }

#if (SPTF_CFG_LWIP_CORE_LOCKING != 0U)
    UNLOCK_TCPIP_CORE();
#endif

    return status;
}

/* ============================================================ çerçeve gönderimi */

static void make_header(const sptf_udp_stream_t *stream, uint64_t timestamp_ns, sptf_frame_header_t *header)
{
    header->timestamp_ns = timestamp_ns;
    header->seq = stream->seq;
    header->stream_id = stream->stream_id;
    header->desc_rev = stream->desc_rev;
    header->crc_enabled = stream->crc_enabled;
}

/**
 * Kodlanmış çerçeveyi gönderir ve sayaçları günceller. Tampon alınamadığı için hiç
 * üretilmeyen çerçeve seq tüketmez; gönderimi başarısız olan çerçeve tüketir ([G-17]).
 */
static sptf_status_t send_frame(sptf_udp_stream_t *stream, uint32_t frame_size)
{
    const sptf_status_t status = transmit_frame(stream, frame_size);

    if (status == SPTF_STATUS_NO_BUFFER)
    {
        stream->buffer_errors++;
    }
    else
    {
        stream->seq++;

        if (status == SPTF_STATUS_OK)
        {
            stream->frames_sent++;
            stream->bytes_sent += (uint64_t)frame_size;
        }
        else
        {
            stream->send_errors++;
        }
    }

    return status;
}

static sptf_status_t send_descriptor_frame(sptf_udp_stream_t *stream, uint64_t timestamp_ns)
{
    sptf_frame_header_t header;
    uint32_t frame_size = 0U;
    sptf_status_t status;

    make_header(stream, timestamp_ns, &header);
    status = sptf_encode_descriptor(&header, &stream->desc, stream->frame, (uint32_t)stream->max_datagram_size,
                                    &frame_size);

    if (status == SPTF_STATUS_OK)
    {
        status = send_frame(stream, frame_size);
    }

    if (status == SPTF_STATUS_OK)
    {
        stream->state = STATE_STREAMING;
        stream->described = true;
        stream->descriptor_retry = false;
        stream->next_descriptor_sample = stream->first_sample + stream->descriptor_period_samples;
        stream->last_descriptor_ns = timestamp_ns;
        stream->descriptors_sent++;
    }
    else
    {
        /* Bir sonraki fırsatta yeniden denenir ([G-31]). */
        stream->descriptor_retry = true;
    }

    return status;
}

/**
 * DESCRIPTOR'ın gönderilmesi gerekip gerekmediğini belirler: akış henüz tanımlanmadıysa,
 * önceki deneme başarısızsa, periyodun örnek karşılığı dolduysa ya da (zaman damgaları
 * biliniyorsa) son DESCRIPTOR'dan beri aralık kadar süre geçtiyse.
 */
static bool descriptor_is_due(const sptf_udp_stream_t *stream, uint64_t timestamp_ns)
{
    bool due = (stream->state != STATE_STREAMING) || stream->descriptor_retry;

    if (!due)
    {
        due = (stream->first_sample >= stream->next_descriptor_sample);
    }

    if ((!due) && (stream->last_descriptor_ns != 0U) && (timestamp_ns > stream->last_descriptor_ns))
    {
        due = ((timestamp_ns - stream->last_descriptor_ns) >= stream->descriptor_interval_ns);
    }

    return due;
}

/**
 * Bloğun ilk örneğinin zamanından, blok içinde offset konumundaki örneğin zamanını
 * hesaplar (ns, en yakın tamsayıya yuvarlanır). Zaman bilinmiyorsa ya da sonuç 64 bite
 * sığmıyorsa 0 döner.
 */
static uint64_t frame_timestamp(const sptf_udp_stream_t *stream, uint64_t timestamp_ns, uint32_t offset)
{
    uint64_t result = 0U;

    if (timestamp_ns != 0U)
    {
        const sptf_float64_t delay_ns = ((sptf_float64_t)offset * stream->ns_per_sample) + 0.5;

        if (delay_ns < MAX_FRAME_DELAY_NS)
        {
            const uint64_t delay = (uint64_t)delay_ns;

            if (timestamp_ns <= (UINT64_MAX - delay))
            {
                result = timestamp_ns + delay;
            }
        }
    }

    return result;
}

static sptf_status_t send_data_frame(sptf_udp_stream_t *stream, const sptf_data_slice_t *slice,
                                     uint64_t timestamp_ns)
{
    sptf_frame_header_t header;
    uint32_t frame_size = 0U;
    sptf_status_t status;

    make_header(stream, timestamp_ns, &header);
    status = sptf_encode_data(&header, &stream->desc, slice, stream->frame, (uint32_t)stream->max_datagram_size,
                              &frame_size);

    if (status == SPTF_STATUS_OK)
    {
        status = send_frame(stream, frame_size);
    }

    /* Gönderilemeyen örnekler kayıp sayılır; first_sample her durumda ilerler ([G-18]). */
    if (status == SPTF_STATUS_OK)
    {
        stream->samples_sent += (uint64_t)slice->count;
    }
    else
    {
        stream->samples_lost += (uint64_t)slice->count;
    }

    stream->first_sample += (uint64_t)slice->count;

    return status;
}

static sptf_status_t send_block(sptf_udp_stream_t *stream, const uint8_t *samples, uint32_t sample_count,
                                uint64_t timestamp_ns)
{
    sptf_status_t result = SPTF_STATUS_OK;
    sptf_data_slice_t slice;
    uint32_t done = 0U;

    slice.block = samples;
    slice.block_length = sample_count;
    slice.first_sample = 0U;
    slice.offset = 0U;
    slice.count = 0U;

    while (done < sample_count)
    {
        const uint32_t remaining = sample_count - done;
        const uint64_t frame_time = frame_timestamp(stream, timestamp_ns, done);
        sptf_status_t status = SPTF_STATUS_OK;

        if (descriptor_is_due(stream, frame_time))
        {
            status = send_descriptor_frame(stream, frame_time);
        }

        if (stream->state != STATE_STREAMING)
        {
            /* İlk DESCRIPTOR gönderilemedi: bloğun kalanı kayıp sayılır ([G-09], [G-29]). */
            stream->samples_lost += (uint64_t)remaining;
            stream->first_sample += (uint64_t)remaining;
            status = SPTF_STATUS_DESCRIPTOR_PENDING;
            done = sample_count;
        }
        else
        {
            sptf_status_t data_status;

            slice.offset = done;
            slice.first_sample = stream->first_sample;

            if (remaining < stream->max_samples_per_frame)
            {
                slice.count = remaining;
            }
            else
            {
                slice.count = stream->max_samples_per_frame;
            }

            data_status = send_data_frame(stream, &slice, frame_time);

            if (data_status != SPTF_STATUS_OK)
            {
                status = data_status;
            }

            done += slice.count;
        }

        if (status != SPTF_STATUS_OK)
        {
            result = status;
        }
    }

    return result;
}

/* ================================================================= dış arayüz */

sptf_status_t sptf_udp_config_init(sptf_udp_config_t *config)
{
    sptf_status_t status = SPTF_STATUS_OK;

    if (config == NULL)
    {
        status = SPTF_STATUS_NULL_POINTER;
    }
    else
    {
        config->desc.name = NULL;
        config->desc.channels = NULL;
        config->desc.sample_rate_hz = 0.0;
        config->desc.channel_count = 0U;
        config->desc.sample_type = 0U;
        config->desc.layout = SPTF_LAYOUT_INTERLEAVED;
        config->remote_ip[0] = 0U;
        config->remote_ip[1] = 0U;
        config->remote_ip[2] = 0U;
        config->remote_ip[3] = 0U;
        config->remote_port = SPTF_DEFAULT_UDP_PORT;
        config->local_port = 0U;
        config->stream_id = 0U;
        config->initial_desc_rev = SPTF_DEFAULT_DESC_REV;
        config->max_datagram_size = SPTF_CFG_MAX_DATAGRAM_SIZE;
        config->descriptor_interval_ms = SPTF_DEFAULT_DESCRIPTOR_INTERVAL_MS;
        config->crc_enabled = false;
    }

    return status;
}

sptf_status_t sptf_udp_open(const sptf_udp_config_t *config, sptf_udp_stream_t **stream)
{
    sptf_status_t status = SPTF_STATUS_OK;
    sptf_udp_stream_t *slot = NULL;

    if ((config == NULL) || (stream == NULL))
    {
        status = SPTF_STATUS_NULL_POINTER;
    }
    else
    {
        status = check_config(config);
    }

    if (status == SPTF_STATUS_OK)
    {
        if (stream_id_in_use(config->stream_id))
        {
            status = SPTF_STATUS_DUPLICATE_STREAM_ID;
        }
        else
        {
            slot = find_free_slot();

            if (slot == NULL)
            {
                status = SPTF_STATUS_NO_FREE_STREAM;
            }
        }
    }

    if (status == SPTF_STATUS_OK)
    {
        (void)memset(slot, 0, sizeof(*slot));
        status = create_pcb(slot, config->local_port);
    }

    if (status == SPTF_STATUS_OK)
    {
        const uint32_t host_address = ((uint32_t)config->remote_ip[0] << 24U)
                                      | ((uint32_t)config->remote_ip[1] << 16U)
                                      | ((uint32_t)config->remote_ip[2] << 8U)
                                      | (uint32_t)config->remote_ip[3];

        /* Sayaçlar ve bayraklar memset ile sıfırlandı; seq ve first_sample 0'dan başlar ([G-10]). */
        ip_addr_set_ip4_u32_val(slot->remote_ip, lwip_htonl(host_address));
        slot->desc = config->desc;
        slot->stream_id = config->stream_id;
        slot->desc_rev = config->initial_desc_rev;
        slot->remote_port = config->remote_port;
        slot->max_datagram_size = config->max_datagram_size;
        slot->descriptor_interval_ms = config->descriptor_interval_ms;
        slot->crc_enabled = config->crc_enabled;
        update_derived_values(slot);
        slot->state = STATE_DESCRIBING;
        *stream = slot;
    }

    return status;
}

sptf_status_t sptf_udp_close(sptf_udp_stream_t *stream)
{
    const sptf_status_t status = check_stream(stream);

    if (status == SPTF_STATUS_OK)
    {
        /* memset durumu STATE_FREE (0) yapar; yuva yeniden kullanılabilir. */
        remove_pcb(stream->pcb);
        (void)memset(stream, 0, sizeof(*stream));
        stream->pcb = NULL;
    }

    return status;
}

sptf_status_t sptf_udp_send_samples(sptf_udp_stream_t *stream, const uint8_t *samples, uint32_t sample_count,
                                    uint64_t timestamp_ns)
{
    sptf_status_t status = check_stream(stream);

    if (status == SPTF_STATUS_OK)
    {
        if (samples == NULL)
        {
            status = SPTF_STATUS_NULL_POINTER;
        }
        else if (sample_count == 0U)
        {
            status = SPTF_STATUS_INVALID_ARGUMENT;
        }
        else if (sample_count > (UINT32_MAX / stream->bytes_per_instant))
        {
            status = SPTF_STATUS_INVALID_ARGUMENT;
        }
        else if ((uint64_t)sample_count > (SPTF_FIRST_SAMPLE_LIMIT - stream->first_sample))
        {
            status = SPTF_STATUS_COUNTER_LIMIT;
        }
        else
        {
            status = send_block(stream, samples, sample_count, timestamp_ns);
        }
    }

    return status;
}

sptf_status_t sptf_udp_send_marker(sptf_udp_stream_t *stream, uint64_t sample_index, uint32_t code,
                                   const char *text, uint64_t timestamp_ns)
{
    sptf_status_t status = check_stream(stream);
    sptf_status_t descriptor_status = SPTF_STATUS_OK;

    if ((status == SPTF_STATUS_OK) && (text == NULL))
    {
        status = SPTF_STATUS_NULL_POINTER;
    }

    if (status == SPTF_STATUS_OK)
    {
        if (descriptor_is_due(stream, timestamp_ns))
        {
            descriptor_status = send_descriptor_frame(stream, timestamp_ns);
        }

        if (stream->state != STATE_STREAMING)
        {
            /* Tanımlanmamış akışın MARKER'ı alıcıda yok sayılırdı ([G-09]). */
            status = SPTF_STATUS_DESCRIPTOR_PENDING;
        }
        else
        {
            sptf_frame_header_t header;
            uint32_t frame_size = 0U;

            make_header(stream, timestamp_ns, &header);
            status = sptf_encode_marker(&header, sample_index, code, text, stream->frame,
                                        (uint32_t)stream->max_datagram_size, &frame_size);

            if (status == SPTF_STATUS_OK)
            {
                status = send_frame(stream, frame_size);
            }

            if (status == SPTF_STATUS_OK)
            {
                stream->markers_sent++;
                status = descriptor_status;
            }
        }
    }

    return status;
}

sptf_status_t sptf_udp_send_descriptor(sptf_udp_stream_t *stream, uint64_t timestamp_ns)
{
    sptf_status_t status = check_stream(stream);

    if (status == SPTF_STATUS_OK)
    {
        status = send_descriptor_frame(stream, timestamp_ns);
    }

    return status;
}

sptf_status_t sptf_udp_reconfigure(sptf_udp_stream_t *stream, const sptf_stream_desc_t *desc)
{
    sptf_status_t status = check_stream(stream);

    if ((status == SPTF_STATUS_OK) && (desc == NULL))
    {
        status = SPTF_STATUS_NULL_POINTER;
    }

    if (status == SPTF_STATUS_OK)
    {
        status = sptf_check_stream_desc(desc);
    }

    if (status == SPTF_STATUS_OK)
    {
        status = check_fits(desc, stream->max_datagram_size, stream->crc_enabled);
    }

    if (status == SPTF_STATUS_OK)
    {
        /* Yeni revizyon yeni bir zaman ekseniyle başlar ([G-14], [G-15]). */
        stream->desc = *desc;
        stream->desc_rev = (uint16_t)(stream->desc_rev + 1U);
        stream->first_sample = 0U;
        stream->state = STATE_DESCRIBING;
        stream->descriptor_retry = false;
        update_derived_values(stream);
        status = send_descriptor_frame(stream, 0U);
    }

    return status;
}

sptf_status_t sptf_udp_get_stats(const sptf_udp_stream_t *stream, sptf_udp_stats_t *stats)
{
    sptf_status_t status = check_stream(stream);

    if ((status == SPTF_STATUS_OK) && (stats == NULL))
    {
        status = SPTF_STATUS_NULL_POINTER;
    }

    if (status == SPTF_STATUS_OK)
    {
        stats->next_first_sample = stream->first_sample;
        stats->samples_sent = stream->samples_sent;
        stats->samples_lost = stream->samples_lost;
        stats->bytes_sent = stream->bytes_sent;
        stats->frames_sent = stream->frames_sent;
        stats->descriptors_sent = stream->descriptors_sent;
        stats->markers_sent = stream->markers_sent;
        stats->buffer_errors = stream->buffer_errors;
        stats->send_errors = stream->send_errors;
        stats->next_seq = stream->seq;
        stats->desc_rev = stream->desc_rev;
        stats->described = stream->described;
    }

    return status;
}
