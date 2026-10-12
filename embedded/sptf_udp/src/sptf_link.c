/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    sptf_link.c
 * @brief   UDP göndericisi: akış durumu, datagram boyutuna göre bölme, DESCRIPTOR tekrarı, işaretler.
 *
 * Her datagram tek bir tam çerçeve taşır (protokol bir çerçevenin iki datagrama bölünmesine izin vermez).
 * Datagram tamponu taşıma katmanından alınır (::sptf_transport_t::acquire), çerçeve doğrudan oraya yazılır
 * (lwIP raw portunda sıfır kopya) ve ::sptf_transport_t::send ile gönderilir.
 *
 * @ingroup sptf_link
 */
#include <string.h>

#include "sptf_internal.h"

/** @brief ::sptf_link_t::magic başlatılma işareti ("SPTL"). */
#define LINK_MAGIC 0x4C545053u

/** @brief Bağlantı yapılandırmasında kabul edilen en küçük datagram (bayt). */
#define MIN_DATAGRAM 128u

/** @brief Protokolün izin verdiği en uzun DESCRIPTOR aralığı (ms). */
#define MAX_DESCRIPTOR_PERIOD_MS 1000u

/** @brief Bağlantı başlatılmış mı. */
static bool link_ok(const sptf_link_t *link)
{
    return (link != NULL) && (link->magic == LINK_MAGIC);
}

/** @brief Akış başlatılmış ve bir bağlantıya ekli mi. */
static bool stream_ok(const sptf_stream_t *s)
{
    return (s != NULL) && link_ok(s->link);
}

/** @brief Hata kodunu bağlantı sayaçlarına işler. */
static void count_error(sptf_link_t *link, sptf_status_t st)
{
    if (st == SPTF_ERR_NOMEM)
    {
        link->stats.err_nomem++;
    }
    else if ((st == SPTF_ERR_IO) || (st == SPTF_ERR_BUSY))
    {
        link->stats.err_io++;
    }
    else
    {
        /* diğer kodlar gönderim hatası değildir */
    }
}

/** @brief Taşımadan @p len baytlık tampon ister; alamazsa sayacı artırır. */
static uint8_t *tx_acquire(sptf_link_t *link, size_t len)
{
    uint8_t *buf = link->cfg.transport.acquire(link->cfg.transport.ctx, len);
    if (buf == NULL)
    {
        count_error(link, SPTF_ERR_NOMEM);
    }
    return buf;
}

/** @brief Tamponu gönderir (taşıma tamponu her durumda serbest bırakır) ve sayaçları günceller. */
static sptf_status_t tx_send(sptf_link_t *link, uint8_t *buf, size_t len)
{
    const sptf_status_t st = link->cfg.transport.send(link->cfg.transport.ctx, buf, len);
    if (st == SPTF_OK)
    {
        link->stats.datagrams++;
        link->stats.bytes += (uint64_t)len;
    }
    else
    {
        count_error(link, st);
    }
    return st;
}

/** @brief Kodlama başarısızsa (yapılandırma doğrulandığı için beklenmez) tamponu göndermeden bırakır. */
static sptf_status_t tx_release(sptf_link_t *link, uint8_t *buf, sptf_status_t why)
{
    (void)link->cfg.transport.send(link->cfg.transport.ctx, buf, 0u);
    SPTF_CFG_ASSERT(why == SPTF_OK);
    return why;
}

/** @brief Saat varsa şimdiki zaman (ns), yoksa 0. */
static uint64_t link_now(const sptf_link_t *link)
{
    return (link->cfg.clock != NULL) ? link->cfg.clock(link->cfg.clock_ctx) : 0u;
}

/** @brief Saatsiz modda DESCRIPTOR tekrar aralığı (örnek; en az 1). */
static uint64_t period_samples(const sptf_stream_t *s)
{
    const double n = (s->cfg.sample_rate_hz * (double)s->link->cfg.descriptor_period_ms) / 1000.0;
    return (n >= 1.0) ? (uint64_t)n : 1u;
}

/** @brief DESCRIPTOR zamanı geldi mi (bekleyen, periyot doldu). */
static bool descriptor_due(const sptf_stream_t *s, uint64_t now)
{
    bool due = s->desc_pending;
    if (!due)
    {
        if (s->link->cfg.clock != NULL)
        {
            due = (now >= s->desc_due);
        }
        else
        {
            due = (s->next_sample >= s->desc_due);
        }
    }
    return due;
}

/** @brief Akışın DESCRIPTOR'ını gönderir; başarılıysa bir sonraki tekrarı planlar. */
static sptf_status_t send_descriptor(sptf_stream_t *s, uint64_t now)
{
    sptf_link_t *link = s->link;
    const size_t len = (size_t)sptf_descriptor_size(s->cfg.channel_count, s->cfg.crc);
    sptf_status_t st;
    size_t n = 0u;
    sptf_frame_info_t info;
    uint8_t *buf = tx_acquire(link, len);
    if (buf == NULL)
    {
        return SPTF_ERR_NOMEM;
    }
    info.stream_id = s->cfg.stream_id;
    info.desc_rev = s->desc_rev;
    info.seq = s->seq;
    info.timestamp_ns = now;
    info.crc = s->cfg.crc;
    st = sptf_encode_descriptor(buf, len, &info, &s->cfg, &n);
    if (st != SPTF_OK)
    {
        return tx_release(link, buf, st);
    }
    s->seq++;
    st = tx_send(link, buf, n);
    if (st == SPTF_OK)
    {
        s->stats.desc_frames++;
        s->desc_pending = false;
        if (link->cfg.clock != NULL)
        {
            s->desc_due = now + ((uint64_t)link->cfg.descriptor_period_ms * 1000000u);
        }
        else
        {
            s->desc_due = s->next_sample + period_samples(s);
        }
    }
    return st;
}

/** @brief @p samples örnek süresi (ns), en yakın tamsayıya yuvarlanmış. */
static uint64_t offset_ns(uint32_t samples, double ns_per_sample)
{
    const double ns = ((double)samples * ns_per_sample) + 0.5;
    return (uint64_t)ns;
}

/**
 * @brief Bloğun `[offset, offset + k)` dilimini tek DATA çerçevesi olarak gönderir; zamanı geldiyse önce DESCRIPTOR.
 * @return ::SPTF_OK ya da ilk başarısız adımın kodu (dilim gönderilmemiştir).
 */
static sptf_status_t send_data(sptf_stream_t *s, const uint8_t *block, uint32_t count, uint32_t offset, uint32_t k,
                               uint64_t ts, uint64_t now)
{
    sptf_link_t *link = s->link;
    const size_t len = (size_t)SPTF_HEADER_SIZE + (size_t)SPTF_DATA_BODY_SIZE + ((size_t)k * (size_t)s->sample_bytes) +
                       (s->cfg.crc ? (size_t)SPTF_CRC_SIZE : 0u);
    sptf_status_t st = SPTF_OK;
    sptf_frame_info_t info;
    size_t n = 0u;
    uint8_t *buf;
    if (descriptor_due(s, now))
    {
        st = send_descriptor(s, now);
    }
    if (st != SPTF_OK)
    {
        return st;
    }
    buf = tx_acquire(link, len);
    if (buf == NULL)
    {
        return SPTF_ERR_NOMEM;
    }
    info.stream_id = s->cfg.stream_id;
    info.desc_rev = s->desc_rev;
    info.seq = s->seq;
    info.timestamp_ns = ts;
    info.crc = s->cfg.crc;
    st = sptf_priv_encode_data(buf, len, &info, &s->cfg, s->next_sample, block, count, offset, k, &n);
    if (st != SPTF_OK)
    {
        return tx_release(link, buf, st);
    }
    s->seq++;
    st = tx_send(link, buf, n);
    if (st == SPTF_OK)
    {
        s->stats.data_frames++;
        s->stats.samples_sent += (uint64_t)k;
        s->next_sample += (uint64_t)k;
    }
    return st;
}

/** @brief Yapılandırmadan türetilen boyutları hesaplar ve datagrama sığdığını denetler. */
static sptf_status_t derive(const sptf_link_t *link, const sptf_stream_config_t *cfg, uint32_t *sample_bytes,
                            uint32_t *max_per_frame)
{
    const uint32_t room_hdr = SPTF_HEADER_SIZE + SPTF_DATA_BODY_SIZE + (cfg->crc ? SPTF_CRC_SIZE : 0u);
    const uint32_t bytes = sptf_sample_size(cfg->sample_type) * (uint32_t)cfg->channel_count;
    if (sptf_descriptor_size(cfg->channel_count, cfg->crc) > link->cfg.max_datagram)
    {
        return SPTF_ERR_SIZE;
    }
    if ((bytes == 0u) || ((room_hdr + bytes) > link->cfg.max_datagram))
    {
        return SPTF_ERR_SIZE;
    }
    *sample_bytes = bytes;
    *max_per_frame = (link->cfg.max_datagram - room_hdr) / bytes;
    return SPTF_OK;
}

/** @brief Akışın durumunu yeni yapılandırmaya göre (yeniden) kurar. */
static void apply_config(sptf_stream_t *s, const sptf_stream_config_t *cfg, uint32_t sample_bytes,
                         uint32_t max_per_frame)
{
    s->cfg = *cfg;
    s->sample_bytes = sample_bytes;
    s->max_per_frame = max_per_frame;
    s->ns_per_sample = 1.0e9 / cfg->sample_rate_hz;
    s->next_sample = 0u;
    s->desc_due = 0u;
    s->desc_pending = true;
}

/* ------------------------------------------------------------------------------------------------- API */

sptf_status_t sptf_link_init(sptf_link_t *link, const sptf_link_config_t *cfg)
{
    uint32_t max_datagram;
    uint32_t period;
    if ((link == NULL) || (cfg == NULL))
    {
        return SPTF_ERR_ARG;
    }
    max_datagram = (cfg->max_datagram == 0u) ? (uint32_t)SPTF_CFG_MAX_DATAGRAM : cfg->max_datagram;
    period = (cfg->descriptor_period_ms == 0u) ? (uint32_t)SPTF_CFG_DESCRIPTOR_PERIOD_MS : cfg->descriptor_period_ms;
    if ((cfg->transport.acquire == NULL) || (cfg->transport.send == NULL) || (max_datagram < MIN_DATAGRAM) ||
        (max_datagram > SPTF_MAX_UDP_PAYLOAD) || (period > MAX_DESCRIPTOR_PERIOD_MS))
    {
        return SPTF_ERR_CONFIG;
    }
    (void)memset(link, 0, sizeof(*link));
    link->cfg = *cfg;
    link->cfg.max_datagram = max_datagram;
    link->cfg.descriptor_period_ms = period;
    link->streams = NULL;
    link->magic = LINK_MAGIC;
    return SPTF_OK;
}

sptf_status_t sptf_stream_init(sptf_stream_t *stream, sptf_link_t *link, const sptf_stream_config_t *cfg)
{
    sptf_status_t st;
    uint32_t sample_bytes = 0u;
    uint32_t max_per_frame = 0u;
    sptf_stream_t *it;
    sptf_stream_t *tail = NULL;
    if ((stream == NULL) || (cfg == NULL))
    {
        return SPTF_ERR_ARG;
    }
    if (!link_ok(link))
    {
        return SPTF_ERR_STATE;
    }
    st = sptf_config_check(cfg);
    if (st != SPTF_OK)
    {
        return st;
    }
    for (it = link->streams; it != NULL; it = it->next)
    {
        if (it == stream)
        {
            return SPTF_ERR_ARG;
        }
        if (it->cfg.stream_id == cfg->stream_id)
        {
            return SPTF_ERR_CONFIG;
        }
        tail = it;
    }
    st = derive(link, cfg, &sample_bytes, &max_per_frame);
    if (st != SPTF_OK)
    {
        return st;
    }
    (void)memset(stream, 0, sizeof(*stream));
    apply_config(stream, cfg, sample_bytes, max_per_frame);
    stream->link = link;
    stream->next = NULL;
    stream->seq = 0u;
    stream->desc_rev = 1u;
    if (tail == NULL)
    {
        link->streams = stream;
    }
    else
    {
        tail->next = stream;
    }
    return SPTF_OK;
}

sptf_status_t sptf_stream_write(sptf_stream_t *stream, const void *samples, uint32_t count, uint64_t timestamp_ns)
{
    sptf_link_t *link;
    sptf_status_t result = SPTF_OK;
    uint64_t now;
    uint64_t base_ts;
    uint32_t done = 0u;
    if ((stream == NULL) || (samples == NULL) || (count == 0u))
    {
        return SPTF_ERR_ARG;
    }
    if (!stream_ok(stream))
    {
        return SPTF_ERR_STATE;
    }
    link = stream->link;
    now = link_now(link);
    if (timestamp_ns == SPTF_TS_AUTO)
    {
        const uint64_t span = offset_ns(count, stream->ns_per_sample);
        base_ts = (now > span) ? (now - span) : 0u;
    }
    else
    {
        base_ts = timestamp_ns;
    }
    while ((done < count) && (result == SPTF_OK))
    {
        const uint32_t k = ((count - done) < stream->max_per_frame) ? (count - done) : stream->max_per_frame;
        const uint64_t ts = (base_ts != 0u) ? (base_ts + offset_ns(done, stream->ns_per_sample)) : 0u;
        result = send_data(stream, (const uint8_t *)samples, count, done, k, ts, now);
        if (result == SPTF_OK)
        {
            done += k;
        }
    }
    if (done < count)
    {
        /* Gönderilemeyen örnekler kayıptır; sayaç yine ilerler ki alıcı boşluğu doğru uzunlukta göstersin. */
        const uint64_t lost = (uint64_t)count - (uint64_t)done;
        stream->stats.samples_dropped += lost;
        stream->next_sample += lost;
    }
    return result;
}

sptf_status_t sptf_stream_marker(sptf_stream_t *stream, uint64_t sample_index, uint32_t code, const char *text,
                                 uint64_t timestamp_ns)
{
    sptf_link_t *link;
    sptf_status_t st;
    uint64_t now;
    size_t text_len;
    size_t len;
    size_t n = 0u;
    uint8_t *buf;
    sptf_frame_info_t info;
    if (stream == NULL)
    {
        return SPTF_ERR_ARG;
    }
    if (!stream_ok(stream))
    {
        return SPTF_ERR_STATE;
    }
    link = stream->link;
    text_len = sptf_priv_strnlen(text, (size_t)link->cfg.max_datagram);
    len = (size_t)SPTF_HEADER_SIZE + (size_t)SPTF_MARKER_BODY_SIZE + text_len +
          (stream->cfg.crc ? (size_t)SPTF_CRC_SIZE : 0u);
    if (len > (size_t)link->cfg.max_datagram)
    {
        return SPTF_ERR_SIZE;
    }
    now = link_now(link);
    if (descriptor_due(stream, now))
    {
        /* Alıcı tanımadığı akışın işaretini yok sayar: önce DESCRIPTOR. */
        st = send_descriptor(stream, now);
        if (st != SPTF_OK)
        {
            return st;
        }
    }
    buf = tx_acquire(link, len);
    if (buf == NULL)
    {
        return SPTF_ERR_NOMEM;
    }
    info.stream_id = stream->cfg.stream_id;
    info.desc_rev = stream->desc_rev;
    info.seq = stream->seq;
    info.timestamp_ns = (timestamp_ns == SPTF_TS_AUTO) ? now : timestamp_ns;
    info.crc = stream->cfg.crc;
    st = sptf_encode_marker(buf, len, &info, (sample_index == SPTF_MARKER_AT_NEXT) ? stream->next_sample : sample_index,
                            code, text, &n);
    if (st != SPTF_OK)
    {
        return tx_release(link, buf, st);
    }
    stream->seq++;
    st = tx_send(link, buf, n);
    if (st == SPTF_OK)
    {
        stream->stats.marker_frames++;
    }
    return st;
}

sptf_status_t sptf_stream_reconfigure(sptf_stream_t *stream, const sptf_stream_config_t *cfg)
{
    sptf_status_t st;
    uint32_t sample_bytes = 0u;
    uint32_t max_per_frame = 0u;
    if ((stream == NULL) || (cfg == NULL))
    {
        return SPTF_ERR_ARG;
    }
    if (!stream_ok(stream))
    {
        return SPTF_ERR_STATE;
    }
    if (cfg->stream_id != stream->cfg.stream_id)
    {
        return SPTF_ERR_CONFIG;
    }
    st = sptf_config_check(cfg);
    if (st == SPTF_OK)
    {
        st = derive(stream->link, cfg, &sample_bytes, &max_per_frame);
    }
    if (st == SPTF_OK)
    {
        apply_config(stream, cfg, sample_bytes, max_per_frame);
        stream->desc_rev++;   /* 65535 → 0 taşması protokolde geçerlidir: alıcı yalnız eşitliğe bakar */
    }
    return st;
}

sptf_status_t sptf_stream_deinit(sptf_stream_t *stream)
{
    sptf_link_t *link;
    sptf_stream_t *it;
    sptf_stream_t *prev = NULL;
    if (stream == NULL)
    {
        return SPTF_ERR_ARG;
    }
    if (!stream_ok(stream))
    {
        return SPTF_ERR_STATE;
    }
    link = stream->link;
    for (it = link->streams; it != NULL; it = it->next)
    {
        if (it == stream)
        {
            if (prev == NULL)
            {
                link->streams = it->next;
            }
            else
            {
                prev->next = it->next;
            }
            break;
        }
        prev = it;
    }
    stream->link = NULL;
    stream->next = NULL;
    return SPTF_OK;
}

sptf_status_t sptf_link_poll(sptf_link_t *link)
{
    sptf_status_t result = SPTF_OK;
    sptf_stream_t *it;
    uint64_t now;
    if (link == NULL)
    {
        return SPTF_ERR_ARG;
    }
    if (!link_ok(link))
    {
        return SPTF_ERR_STATE;
    }
    now = link_now(link);
    for (it = link->streams; it != NULL; it = it->next)
    {
        /* Saatsiz modda yalnız bekleyen DESCRIPTOR'lar: periyot örnek sayısıyla ölçülür, veri yoksa ilerlemez. */
        const bool due = (link->cfg.clock != NULL) ? descriptor_due(it, now) : it->desc_pending;
        if (due)
        {
            const sptf_status_t st = send_descriptor(it, now);
            if ((st != SPTF_OK) && (result == SPTF_OK))
            {
                result = st;
            }
        }
    }
    return result;
}

uint32_t sptf_stream_samples_per_frame(const sptf_stream_t *stream)
{
    return stream_ok(stream) ? stream->max_per_frame : 0u;
}
