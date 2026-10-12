/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    sptf_frame.c
 * @brief   SPTF v1 çerçeve kodlayıcıları ve CRC-32/ISO-HDLC.
 *
 * Tüm çok baytlı alanlar bayt bayt küçük uçlu yazılır: hedefin bayt sırasından, hizalamasından ve derleyicinin
 * yapı dolgusundan bağımsızdır. `double` alanlar IEEE 754 binary64 bit deseniyle kopyalanır.
 *
 * @ingroup sptf_frame
 */
#include <string.h>

#include "sptf_internal.h"

/* C99 derleme zamanı denetimi: double, 8 baytlık IEEE 754 binary64 olmalı (AVR'de -mdouble=64 gerekir). */
typedef char sptf_priv_double_is_64bit[(sizeof(double) == 8u) ? 1 : -1];

/* ------------------------------------------------------------------------------------------ bayt yazıcılar */

/** @brief 16 bit değeri küçük uçlu yazar. */
static void put_u16(uint8_t *p, uint16_t v)
{
    p[0] = (uint8_t)(v & 0xFFu);
    p[1] = (uint8_t)((uint32_t)v >> 8u);
}

/** @brief 32 bit değeri küçük uçlu yazar. */
static void put_u32(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)(v & 0xFFu);
    p[1] = (uint8_t)((v >> 8u) & 0xFFu);
    p[2] = (uint8_t)((v >> 16u) & 0xFFu);
    p[3] = (uint8_t)(v >> 24u);
}

/** @brief 64 bit değeri küçük uçlu yazar. */
static void put_u64(uint8_t *p, uint64_t v)
{
    put_u32(p, (uint32_t)(v & 0xFFFFFFFFu));
    put_u32(&p[4], (uint32_t)(v >> 32u));
}

/** @brief binary64 değeri küçük uçlu yazar (bit deseni korunur). */
static void put_f64(uint8_t *p, double v)
{
    uint64_t bits = 0u;
    (void)memcpy((uint8_t *)&bits, (const uint8_t *)&v, sizeof(bits));
    put_u64(p, bits);
}

size_t sptf_priv_strnlen(const char *s, size_t limit)
{
    size_t n = 0u;
    if (s != NULL)
    {
        while ((n < limit) && (s[n] != '\0'))
        {
            n++;
        }
    }
    return n;
}

/** @brief Metni sabit uzunluklu alana yazar, kalanını 0 ile doldurur (uzunluk önceden doğrulanmış olmalı). */
static void put_text(uint8_t *p, size_t field, const char *s)
{
    const size_t n = sptf_priv_strnlen(s, field);
    if (n > 0u)
    {
        (void)memcpy(p, (const uint8_t *)s, n);
    }
    if (n < field)
    {
        (void)memset(&p[n], 0, field - n);
    }
}

/** @brief binary64 değeri sonlu mu (üs alanı tümü 1 değil: ±sonsuz ve NaN değil). Kayan nokta karşılaştırması yok. */
static bool is_finite(double v)
{
    uint64_t bits = 0u;
    (void)memcpy((uint8_t *)&bits, (const uint8_t *)&v, sizeof(bits));
    return ((bits >> 52u) & 0x7FFu) != 0x7FFu;
}

/** @brief Metin alana sığıyor mu (UTF-8 bayt sayısı ≤ alan; `\0` gerekmez). */
static bool text_fits(const char *s, size_t field)
{
    return sptf_priv_strnlen(s, field + 1u) <= field;
}

/* ----------------------------------------------------------------------------------------------- CRC-32 */
#if (SPTF_CFG_CRC != 0)

uint32_t sptf_crc32_update(uint32_t crc, const uint8_t *data, size_t len)
{
#if (SPTF_CFG_CRC_TABLE != 0)
    /* CRC-32/ISO-HDLC tablosu (yansıtılmış polinom 0xEDB88320) */
    static const uint32_t crc_table[256] = {
        0x00000000u, 0x77073096u, 0xEE0E612Cu, 0x990951BAu, 0x076DC419u, 0x706AF48Fu,
        0xE963A535u, 0x9E6495A3u, 0x0EDB8832u, 0x79DCB8A4u, 0xE0D5E91Eu, 0x97D2D988u,
        0x09B64C2Bu, 0x7EB17CBDu, 0xE7B82D07u, 0x90BF1D91u, 0x1DB71064u, 0x6AB020F2u,
        0xF3B97148u, 0x84BE41DEu, 0x1ADAD47Du, 0x6DDDE4EBu, 0xF4D4B551u, 0x83D385C7u,
        0x136C9856u, 0x646BA8C0u, 0xFD62F97Au, 0x8A65C9ECu, 0x14015C4Fu, 0x63066CD9u,
        0xFA0F3D63u, 0x8D080DF5u, 0x3B6E20C8u, 0x4C69105Eu, 0xD56041E4u, 0xA2677172u,
        0x3C03E4D1u, 0x4B04D447u, 0xD20D85FDu, 0xA50AB56Bu, 0x35B5A8FAu, 0x42B2986Cu,
        0xDBBBC9D6u, 0xACBCF940u, 0x32D86CE3u, 0x45DF5C75u, 0xDCD60DCFu, 0xABD13D59u,
        0x26D930ACu, 0x51DE003Au, 0xC8D75180u, 0xBFD06116u, 0x21B4F4B5u, 0x56B3C423u,
        0xCFBA9599u, 0xB8BDA50Fu, 0x2802B89Eu, 0x5F058808u, 0xC60CD9B2u, 0xB10BE924u,
        0x2F6F7C87u, 0x58684C11u, 0xC1611DABu, 0xB6662D3Du, 0x76DC4190u, 0x01DB7106u,
        0x98D220BCu, 0xEFD5102Au, 0x71B18589u, 0x06B6B51Fu, 0x9FBFE4A5u, 0xE8B8D433u,
        0x7807C9A2u, 0x0F00F934u, 0x9609A88Eu, 0xE10E9818u, 0x7F6A0DBBu, 0x086D3D2Du,
        0x91646C97u, 0xE6635C01u, 0x6B6B51F4u, 0x1C6C6162u, 0x856530D8u, 0xF262004Eu,
        0x6C0695EDu, 0x1B01A57Bu, 0x8208F4C1u, 0xF50FC457u, 0x65B0D9C6u, 0x12B7E950u,
        0x8BBEB8EAu, 0xFCB9887Cu, 0x62DD1DDFu, 0x15DA2D49u, 0x8CD37CF3u, 0xFBD44C65u,
        0x4DB26158u, 0x3AB551CEu, 0xA3BC0074u, 0xD4BB30E2u, 0x4ADFA541u, 0x3DD895D7u,
        0xA4D1C46Du, 0xD3D6F4FBu, 0x4369E96Au, 0x346ED9FCu, 0xAD678846u, 0xDA60B8D0u,
        0x44042D73u, 0x33031DE5u, 0xAA0A4C5Fu, 0xDD0D7CC9u, 0x5005713Cu, 0x270241AAu,
        0xBE0B1010u, 0xC90C2086u, 0x5768B525u, 0x206F85B3u, 0xB966D409u, 0xCE61E49Fu,
        0x5EDEF90Eu, 0x29D9C998u, 0xB0D09822u, 0xC7D7A8B4u, 0x59B33D17u, 0x2EB40D81u,
        0xB7BD5C3Bu, 0xC0BA6CADu, 0xEDB88320u, 0x9ABFB3B6u, 0x03B6E20Cu, 0x74B1D29Au,
        0xEAD54739u, 0x9DD277AFu, 0x04DB2615u, 0x73DC1683u, 0xE3630B12u, 0x94643B84u,
        0x0D6D6A3Eu, 0x7A6A5AA8u, 0xE40ECF0Bu, 0x9309FF9Du, 0x0A00AE27u, 0x7D079EB1u,
        0xF00F9344u, 0x8708A3D2u, 0x1E01F268u, 0x6906C2FEu, 0xF762575Du, 0x806567CBu,
        0x196C3671u, 0x6E6B06E7u, 0xFED41B76u, 0x89D32BE0u, 0x10DA7A5Au, 0x67DD4ACCu,
        0xF9B9DF6Fu, 0x8EBEEFF9u, 0x17B7BE43u, 0x60B08ED5u, 0xD6D6A3E8u, 0xA1D1937Eu,
        0x38D8C2C4u, 0x4FDFF252u, 0xD1BB67F1u, 0xA6BC5767u, 0x3FB506DDu, 0x48B2364Bu,
        0xD80D2BDAu, 0xAF0A1B4Cu, 0x36034AF6u, 0x41047A60u, 0xDF60EFC3u, 0xA867DF55u,
        0x316E8EEFu, 0x4669BE79u, 0xCB61B38Cu, 0xBC66831Au, 0x256FD2A0u, 0x5268E236u,
        0xCC0C7795u, 0xBB0B4703u, 0x220216B9u, 0x5505262Fu, 0xC5BA3BBEu, 0xB2BD0B28u,
        0x2BB45A92u, 0x5CB36A04u, 0xC2D7FFA7u, 0xB5D0CF31u, 0x2CD99E8Bu, 0x5BDEAE1Du,
        0x9B64C2B0u, 0xEC63F226u, 0x756AA39Cu, 0x026D930Au, 0x9C0906A9u, 0xEB0E363Fu,
        0x72076785u, 0x05005713u, 0x95BF4A82u, 0xE2B87A14u, 0x7BB12BAEu, 0x0CB61B38u,
        0x92D28E9Bu, 0xE5D5BE0Du, 0x7CDCEFB7u, 0x0BDBDF21u, 0x86D3D2D4u, 0xF1D4E242u,
        0x68DDB3F8u, 0x1FDA836Eu, 0x81BE16CDu, 0xF6B9265Bu, 0x6FB077E1u, 0x18B74777u,
        0x88085AE6u, 0xFF0F6A70u, 0x66063BCAu, 0x11010B5Cu, 0x8F659EFFu, 0xF862AE69u,
        0x616BFFD3u, 0x166CCF45u, 0xA00AE278u, 0xD70DD2EEu, 0x4E048354u, 0x3903B3C2u,
        0xA7672661u, 0xD06016F7u, 0x4969474Du, 0x3E6E77DBu, 0xAED16A4Au, 0xD9D65ADCu,
        0x40DF0B66u, 0x37D83BF0u, 0xA9BCAE53u, 0xDEBB9EC5u, 0x47B2CF7Fu, 0x30B5FFE9u,
        0xBDBDF21Cu, 0xCABAC28Au, 0x53B39330u, 0x24B4A3A6u, 0xBAD03605u, 0xCDD70693u,
        0x54DE5729u, 0x23D967BFu, 0xB3667A2Eu, 0xC4614AB8u, 0x5D681B02u, 0x2A6F2B94u,
        0xB40BBE37u, 0xC30C8EA1u, 0x5A05DF1Bu, 0x2D02EF8Du
    };
#endif
    uint32_t c = ~crc;
    size_t i;
    if (data == NULL)
    {
        return crc;
    }
    for (i = 0u; i < len; i++)
    {
#if (SPTF_CFG_CRC_TABLE != 0)
        c = crc_table[(c ^ (uint32_t)data[i]) & 0xFFu] ^ (c >> 8u);
#else
        uint32_t bit;
        c ^= (uint32_t)data[i];
        for (bit = 0u; bit < 8u; bit++)
        {
            if ((c & 1u) != 0u)
            {
                c = (c >> 1u) ^ 0xEDB88320u;
            }
            else
            {
                c >>= 1u;
            }
        }
#endif
    }
    return ~c;
}

/** @brief Çerçevenin CRC'si: donanım kancası tanımlıysa o, değilse yazılım. */
static uint32_t frame_crc(const uint8_t *p, size_t n)
{
#if defined(SPTF_CFG_CRC32_HOOK)
    return SPTF_CFG_CRC32_HOOK(p, n);
#else
    return sptf_crc32_update(0u, p, n);
#endif
}
#endif /* SPTF_CFG_CRC */

/** @brief CRC isteniyorsa çerçevenin son 4 baytına yazar. */
static void finish_crc(uint8_t *p, size_t frame_len, bool crc)
{
#if (SPTF_CFG_CRC != 0)
    if (crc)
    {
        const size_t body = frame_len - SPTF_CRC_SIZE;
        put_u32(&p[body], frame_crc(p, body));
    }
#else
    (void)p;
    (void)frame_len;
    (void)crc;
#endif
}

/** @brief CRC isteği bu derlemede karşılanabilir mi. */
static bool crc_supported(bool crc)
{
#if (SPTF_CFG_CRC != 0)
    (void)crc;
    return true;
#else
    return !crc;
#endif
}

/* --------------------------------------------------------------------------------------------- tipler */

uint32_t sptf_sample_size(sptf_sample_type_t type)
{
    uint32_t size;
    switch (type)
    {
    case SPTF_INT8:
    case SPTF_UINT8:
        size = 1u;
        break;
    case SPTF_INT16:
    case SPTF_UINT16:
    case SPTF_FLOAT16:
        size = 2u;
        break;
    case SPTF_INT24:
    case SPTF_UINT24:
        size = 3u;
        break;
    case SPTF_INT32:
    case SPTF_UINT32:
    case SPTF_FLOAT32:
    case SPTF_CINT16:
        size = 4u;
        break;
    case SPTF_INT64:
    case SPTF_UINT64:
    case SPTF_FLOAT64:
    case SPTF_CINT32:
    case SPTF_CFLOAT32:
        size = 8u;
        break;
    case SPTF_CFLOAT64:
        size = 16u;
        break;
    default:
        size = 0u;
        break;
    }
    return size;
}

/**
 * @brief Bayt sırası çevrilirken kullanılan bileşen boyutu: karmaşıkta yarısı, 24 bit paketlide 1
 *        (çağıran zaten küçük uçlu paketler), diğerlerinde tipin kendisi.
 */
static uint32_t element_size(sptf_sample_type_t type)
{
    uint32_t elem = sptf_sample_size(type);
    if ((type == SPTF_INT24) || (type == SPTF_UINT24))
    {
        elem = 1u;
    }
    else if ((type == SPTF_CINT16) || (type == SPTF_CINT32) || (type == SPTF_CFLOAT32) || (type == SPTF_CFLOAT64))
    {
        elem /= 2u;
    }
    else
    {
        /* tek bileşenli tip: boyut aynen */
    }
    return elem;
}

/**
 * @brief Örnek baytlarını kabloya kopyalar. Küçük uçlu hedefte düz kopya; büyük uçlu hedefte her
 *        @p elem baytlık bileşen ters çevrilir.
 */
static void copy_samples(uint8_t *dst, const uint8_t *src, size_t nbytes, uint32_t elem)
{
#if (SPTF_CFG_HOST_BIG_ENDIAN != 0)
    if (elem > 1u)
    {
        size_t i;
        uint32_t j;
        for (i = 0u; (i + elem) <= nbytes; i += elem)
        {
            for (j = 0u; j < elem; j++)
            {
                dst[i + j] = src[(i + elem) - 1u - j];
            }
        }
    }
    else
    {
        (void)memcpy(dst, src, nbytes);
    }
#else
    (void)elem;
    (void)memcpy(dst, src, nbytes);
#endif
}

sptf_status_t sptf_config_check(const sptf_stream_config_t *cfg)
{
    bool ok;
    if (cfg == NULL)
    {
        return SPTF_ERR_ARG;
    }
    /* NaN her karşılaştırmada yanlış döner: aralık denetimi NaN'ı da reddeder. */
    ok = (cfg->sample_rate_hz >= SPTF_MIN_SAMPLE_RATE) && (cfg->sample_rate_hz <= SPTF_MAX_SAMPLE_RATE);
    ok = ok && (cfg->channel_count != 0u) && (cfg->channel_count <= SPTF_MAX_CHANNELS) && (cfg->channels != NULL);
    ok = ok && (sptf_sample_size(cfg->sample_type) != 0u);
    ok = ok && ((cfg->layout == SPTF_LAYOUT_INTERLEAVED) || (cfg->layout == SPTF_LAYOUT_PLANAR));
    ok = ok && crc_supported(cfg->crc) && text_fits(cfg->name, SPTF_NAME_SIZE);
    if (ok)
    {
        uint32_t i;
        for (i = 0u; (i < cfg->channel_count) && ok; i++)
        {
            const sptf_channel_t *ch = &cfg->channels[i];
            ok = text_fits(ch->name, SPTF_NAME_SIZE) && text_fits(ch->unit, SPTF_UNIT_SIZE) && is_finite(ch->scale) &&
                 is_finite(ch->offset);
        }
    }
    return ok ? SPTF_OK : SPTF_ERR_CONFIG;
}

uint32_t sptf_descriptor_size(uint16_t channel_count, bool crc)
{
    return SPTF_HEADER_SIZE + SPTF_DESC_BODY_SIZE + (SPTF_CHANNEL_SIZE * (uint32_t)channel_count) +
           (crc ? SPTF_CRC_SIZE : 0u);
}

/* ---------------------------------------------------------------------------------------- kodlayıcılar */

/** @brief 32 baytlık ortak başlığı yazar. */
static void put_header(uint8_t *p, sptf_frame_type_t type, const sptf_frame_info_t *info, uint16_t stream_id,
                       size_t frame_len)
{
    p[0] = 0x53u; /* 'S' */
    p[1] = 0x50u; /* 'P' */
    p[2] = 0x54u; /* 'T' */
    p[3] = 0x46u; /* 'F' */
    p[4] = (uint8_t)SPTF_PROTOCOL_VERSION;
    p[5] = (uint8_t)type;
    put_u16(&p[6], info->crc ? 1u : 0u);
    put_u32(&p[8], (uint32_t)frame_len);
    put_u16(&p[12], stream_id);
    put_u16(&p[14], info->desc_rev);
    put_u32(&p[16], info->seq);
    put_u32(&p[20], 0u);
    put_u64(&p[24], info->timestamp_ns);
}

sptf_status_t sptf_encode_descriptor(uint8_t *buf, size_t cap, const sptf_frame_info_t *info,
                                     const sptf_stream_config_t *cfg, size_t *out_len)
{
    sptf_status_t st;
    size_t len;
    uint32_t i;
    if ((buf == NULL) || (info == NULL) || (cfg == NULL) || (out_len == NULL))
    {
        return SPTF_ERR_ARG;
    }
    st = sptf_config_check(cfg);
    if (st != SPTF_OK)
    {
        return st;
    }
    if (!crc_supported(info->crc))
    {
        return SPTF_ERR_CONFIG;
    }
    len = (size_t)sptf_descriptor_size(cfg->channel_count, info->crc);
    if (len > cap)
    {
        return SPTF_ERR_SIZE;
    }
    put_header(buf, SPTF_FRAME_DESCRIPTOR, info, cfg->stream_id, len);
    put_f64(&buf[32], cfg->sample_rate_hz);
    put_u16(&buf[40], cfg->channel_count);
    buf[42] = (uint8_t)cfg->sample_type;
    buf[43] = (uint8_t)cfg->layout;
    put_u32(&buf[44], 0u);
    put_text(&buf[48], SPTF_NAME_SIZE, cfg->name);
    for (i = 0u; i < cfg->channel_count; i++)
    {
        uint8_t *c = &buf[(SPTF_HEADER_SIZE + SPTF_DESC_BODY_SIZE) + (SPTF_CHANNEL_SIZE * i)];
        put_text(c, SPTF_NAME_SIZE, cfg->channels[i].name);
        put_text(&c[32], SPTF_UNIT_SIZE, cfg->channels[i].unit);
        put_f64(&c[48], cfg->channels[i].scale);
        put_f64(&c[56], cfg->channels[i].offset);
    }
    finish_crc(buf, len, info->crc);
    *out_len = len;
    return SPTF_OK;
}

sptf_status_t sptf_priv_encode_data(uint8_t *buf, size_t cap, const sptf_frame_info_t *info,
                                    const sptf_stream_config_t *cfg, uint64_t first_sample,
                                    const uint8_t *block, uint32_t block_count, uint32_t offset,
                                    uint32_t count, size_t *out_len)
{
    uint32_t size;
    uint64_t payload;
    uint64_t len;
    if ((buf == NULL) || (info == NULL) || (cfg == NULL) || (block == NULL) || (out_len == NULL) || (count == 0u) ||
        (count > block_count) || (offset > (block_count - count)) || (first_sample >= SPTF_PRIV_MAX_FIRST_SAMPLE))
    {
        return SPTF_ERR_ARG;
    }
    size = sptf_sample_size(cfg->sample_type);
    if ((size == 0u) || (cfg->channel_count == 0u) || (cfg->channel_count > SPTF_MAX_CHANNELS) ||
        ((cfg->layout != SPTF_LAYOUT_INTERLEAVED) && (cfg->layout != SPTF_LAYOUT_PLANAR)) ||
        (!crc_supported(info->crc)))
    {
        return SPTF_ERR_CONFIG;
    }
    payload = (uint64_t)count * (uint64_t)size * (uint64_t)cfg->channel_count;
    len = (uint64_t)SPTF_HEADER_SIZE + (uint64_t)SPTF_DATA_BODY_SIZE + payload +
          (info->crc ? (uint64_t)SPTF_CRC_SIZE : 0u);
    if (len > (uint64_t)cap)
    {
        return SPTF_ERR_SIZE;
    }
    put_header(buf, SPTF_FRAME_DATA, info, cfg->stream_id, (size_t)len);
    put_u64(&buf[32], first_sample);
    put_u32(&buf[40], count);
    put_u32(&buf[44], 0u);
    {
        uint8_t *dst = &buf[SPTF_HEADER_SIZE + SPTF_DATA_BODY_SIZE];
        const uint32_t elem = element_size(cfg->sample_type);
        if (cfg->layout == SPTF_LAYOUT_INTERLEAVED)
        {
            const size_t start = (size_t)offset * (size_t)size * (size_t)cfg->channel_count;
            copy_samples(dst, &block[start], (size_t)payload, elem);
        }
        else
        {
            const size_t run = (size_t)count * (size_t)size;
            uint32_t c;
            for (c = 0u; c < cfg->channel_count; c++)
            {
                const size_t src = (((size_t)c * (size_t)block_count) + (size_t)offset) * (size_t)size;
                copy_samples(&dst[(size_t)c * run], &block[src], run, elem);
            }
        }
    }
    finish_crc(buf, (size_t)len, info->crc);
    *out_len = (size_t)len;
    return SPTF_OK;
}

sptf_status_t sptf_encode_data(uint8_t *buf, size_t cap, const sptf_frame_info_t *info,
                               const sptf_stream_config_t *cfg, uint64_t first_sample,
                               const void *samples, uint32_t count, size_t *out_len)
{
    return sptf_priv_encode_data(buf, cap, info, cfg, first_sample, (const uint8_t *)samples, count, 0u, count,
                                 out_len);
}

sptf_status_t sptf_encode_marker(uint8_t *buf, size_t cap, const sptf_frame_info_t *info,
                                 uint64_t sample_index, uint32_t code, const char *text, size_t *out_len)
{
    size_t n;
    size_t len;
    if ((buf == NULL) || (info == NULL) || (out_len == NULL))
    {
        return SPTF_ERR_ARG;
    }
    if (!crc_supported(info->crc))
    {
        return SPTF_ERR_CONFIG;
    }
    n = sptf_priv_strnlen(text, 0x10000u);
    if (n > 0xFFFFu)
    {
        return SPTF_ERR_SIZE;
    }
    len = SPTF_HEADER_SIZE + SPTF_MARKER_BODY_SIZE + n + (info->crc ? SPTF_CRC_SIZE : 0u);
    if (len > cap)
    {
        return SPTF_ERR_SIZE;
    }
    put_header(buf, SPTF_FRAME_MARKER, info, info->stream_id, len);
    put_u64(&buf[32], sample_index);
    put_u32(&buf[40], code);
    put_u16(&buf[44], (uint16_t)n);
    put_u16(&buf[46], 0u);
    if (n > 0u)
    {
        (void)memcpy(&buf[SPTF_HEADER_SIZE + SPTF_MARKER_BODY_SIZE], (const uint8_t *)text, n);
    }
    finish_crc(buf, len, info->crc);
    *out_len = len;
    return SPTF_OK;
}
