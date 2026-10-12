/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    sptf_test.c
 * @brief   SPTF-UDP birim testleri (ana makinede derlenir; bağımlılık yok).
 *
 * Çıkış kodu 0 = tüm testler geçti. `-DSPTF_CFG_HOST_BIG_ENDIAN=1` ile derlenirse girişler büyük uçlu
 * hazırlanır ve kablo çıktısının yine küçük uçlu olduğu doğrulanır.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "sptf.h"

static int g_fail = 0;
static int g_checks = 0;

/** @brief Koşul yanlışsa satırı yazar ve hatayı sayar. */
#define CHECK(cond)                                                                                              \
    do                                                                                                           \
    {                                                                                                            \
        g_checks++;                                                                                              \
        if (!(cond))                                                                                             \
        {                                                                                                        \
            g_fail++;                                                                                            \
            (void)printf("HATA %s:%d: %s\n", __FILE__, __LINE__, #cond);                                         \
        }                                                                                                        \
    } while (0)

/* ------------------------------------------------------------------------------------------- sahte taşıma */
#define MAX_DG 64
typedef struct
{
    uint8_t  dg[MAX_DG][1600];
    size_t   len[MAX_DG];
    int      count;
    int      fail_acquire_at; /* bu numaralı acquire NULL döner (-1: hiç) */
    int      acquires;
    int      releases;        /* len == 0 ile çağrılan send */
    uint8_t  scratch[70000];
    sptf_status_t send_result;
} mock_t;

static uint8_t *mock_acquire(void *ctx, size_t len)
{
    mock_t *m = (mock_t *)ctx;
    int n = m->acquires++;
    if ((n == m->fail_acquire_at) || (len > sizeof(m->scratch)))
    {
        return NULL;
    }
    return m->scratch;
}

static sptf_status_t mock_send(void *ctx, uint8_t *buf, size_t len)
{
    mock_t *m = (mock_t *)ctx;
    if (len == 0u)
    {
        m->releases++;
        return SPTF_OK;
    }
    if (m->send_result != SPTF_OK)
    {
        return m->send_result;
    }
    if ((m->count < MAX_DG) && (len <= sizeof(m->dg[0])))
    {
        (void)memcpy(m->dg[m->count], buf, len);
        m->len[m->count] = len;
    }
    m->count++;
    return SPTF_OK;
}

static uint64_t g_now = 0u;
static uint64_t fake_clock(void *ctx)
{
    (void)ctx;
    return g_now;
}

static uint32_t rd32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8u) | ((uint32_t)p[2] << 16u) | ((uint32_t)p[3] << 24u);
}
static uint16_t rd16(const uint8_t *p)
{
    return (uint16_t)((uint32_t)p[0] | ((uint32_t)p[1] << 8u));
}
static uint64_t rd64(const uint8_t *p)
{
    return (uint64_t)rd32(p) | ((uint64_t)rd32(&p[4]) << 32u);
}

/** @brief 16 bitlik değeri hedefin (yapılandırılan) bayt sırasıyla yazar. */
static void host16(uint8_t *p, uint16_t v)
{
#if (SPTF_CFG_HOST_BIG_ENDIAN != 0)
    p[0] = (uint8_t)(v >> 8u);
    p[1] = (uint8_t)(v & 0xFFu);
#else
    p[0] = (uint8_t)(v & 0xFFu);
    p[1] = (uint8_t)(v >> 8u);
#endif
}

static const sptf_channel_t CH2[2] = {{"a", "V", 1.0, 0.0}, {"b", "A", 0.5, -1.0}};

static sptf_stream_config_t cfg_int16(uint16_t channels, const sptf_channel_t *ch)
{
    sptf_stream_config_t c;
    (void)memset(&c, 0, sizeof(c));
    c.stream_id = 1u;
    c.name = "test";
    c.sample_rate_hz = 1000.0;
    c.sample_type = SPTF_INT16;
    c.layout = SPTF_LAYOUT_INTERLEAVED;
    c.channel_count = channels;
    c.channels = ch;
    c.crc = false;
    return c;
}

static void link_with(sptf_link_t *link, mock_t *m, uint32_t max_dg, bool clock)
{
    sptf_link_config_t lc;
    (void)memset(m, 0, sizeof(*m));
    m->fail_acquire_at = -1;
    m->send_result = SPTF_OK;
    (void)memset(&lc, 0, sizeof(lc));
    lc.transport.acquire = mock_acquire;
    lc.transport.send = mock_send;
    lc.transport.ctx = m;
    lc.max_datagram = max_dg;
    lc.clock = clock ? fake_clock : NULL;
    CHECK(sptf_link_init(link, &lc) == SPTF_OK);
}

/* --------------------------------------------------------------------------------------------- testler */

static void test_crc(void)
{
#if (SPTF_CFG_CRC != 0)
    const uint8_t s[] = "123456789";
    uint32_t c = sptf_crc32_update(0u, s, 9u);
    CHECK(c == 0xCBF43926u);
    c = sptf_crc32_update(0u, s, 4u);
    c = sptf_crc32_update(c, &s[4], 5u);
    CHECK(c == 0xCBF43926u);
    CHECK(sptf_crc32_update(0u, s, 0u) == 0u);
#else
    /* CRC derlenmedi: CRC isteyen yapılandırma reddedilir */
    sptf_stream_config_t c = cfg_int16(2u, CH2);
    c.crc = true;
    CHECK(sptf_config_check(&c) == SPTF_ERR_CONFIG);
#endif
}

/** @brief SPTF_Protokol_v1.md §12 test vektörü (bayt bayt). */
static void test_vector(void)
{
#if (SPTF_CFG_CRC != 0)
    static const uint8_t expect[60] = {
        0x53, 0x50, 0x54, 0x46, 0x01, 0x02, 0x01, 0x00, 0x3C, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x42, 0x0F, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x64, 0x00, 0xFF, 0xFF, 0xC8, 0x00, 0xFE, 0xFF, 0xA8, 0x40, 0x1E, 0x68};
    uint8_t samples[8];
    uint8_t out[64];
    size_t n = 0u;
    sptf_stream_config_t c = cfg_int16(2u, CH2);
    sptf_frame_info_t info = {1u, 1u, 1u, 1000000u, true};
    host16(&samples[0], 100u);
    host16(&samples[2], 0xFFFFu);
    host16(&samples[4], 200u);
    host16(&samples[6], 0xFFFEu);
    CHECK(sptf_encode_data(out, sizeof(out), &info, &c, 0u, samples, 2u, &n) == SPTF_OK);
    CHECK(n == 60u);
    CHECK(memcmp(out, expect, 60u) == 0);
    CHECK(sptf_encode_data(out, 59u, &info, &c, 0u, samples, 2u, &n) == SPTF_ERR_SIZE);
    CHECK(sptf_encode_data(out, sizeof(out), &info, &c, (uint64_t)1u << 62, samples, 2u, &n) == SPTF_ERR_ARG);
    CHECK(sptf_encode_data(out, sizeof(out), &info, &c, 0u, samples, 0u, &n) == SPTF_ERR_ARG);
#endif
}

static void test_config_validation(void)
{
    static const sptf_channel_t bad_scale[1] = {{"x", "", (double)NAN, 0.0}};
    static const sptf_channel_t long_unit[1] = {{"x", "0123456789abcdefX", 1.0, 0.0}};
    static const sptf_channel_t full_names[1] = {{"0123456789abcdef0123456789abcdef", "0123456789abcdef", 1.0, 0.0}};
    sptf_stream_config_t c = cfg_int16(2u, CH2);
    CHECK(sptf_config_check(&c) == SPTF_OK);
    CHECK(sptf_config_check(NULL) == SPTF_ERR_ARG);
    c.sample_rate_hz = 0.0;
    CHECK(sptf_config_check(&c) == SPTF_ERR_CONFIG);
    c.sample_rate_hz = (double)NAN;
    CHECK(sptf_config_check(&c) == SPTF_ERR_CONFIG);
    c.sample_rate_hz = 2.0e12;
    CHECK(sptf_config_check(&c) == SPTF_ERR_CONFIG);
    c = cfg_int16(0u, CH2);
    CHECK(sptf_config_check(&c) == SPTF_ERR_CONFIG);
    c = cfg_int16(2u, NULL);
    CHECK(sptf_config_check(&c) == SPTF_ERR_CONFIG);
    c = cfg_int16(2u, CH2);
    c.sample_type = (sptf_sample_type_t)0x13; /* NOLINT(clang-analyzer-optin.core.EnumCastOutOfRange): kasıtlı geçersiz değer */
    CHECK(sptf_config_check(&c) == SPTF_ERR_CONFIG);
    c = cfg_int16(2u, CH2);
    c.layout = (sptf_layout_t)2; /* NOLINT(clang-analyzer-optin.core.EnumCastOutOfRange): kasıtlı geçersiz değer */
    CHECK(sptf_config_check(&c) == SPTF_ERR_CONFIG);
    c = cfg_int16(2u, CH2);
    c.name = "0123456789abcdef0123456789abcdefX"; /* 33 bayt */
    CHECK(sptf_config_check(&c) == SPTF_ERR_CONFIG);
    c.name = "0123456789abcdef0123456789abcdef"; /* tam 32 bayt: geçerli */
    CHECK(sptf_config_check(&c) == SPTF_OK);
    c = cfg_int16(1u, bad_scale);
    CHECK(sptf_config_check(&c) == SPTF_ERR_CONFIG);
    c = cfg_int16(1u, long_unit);
    CHECK(sptf_config_check(&c) == SPTF_ERR_CONFIG);
    c = cfg_int16(1u, full_names);
    CHECK(sptf_config_check(&c) == SPTF_OK);
    CHECK(sptf_sample_size(SPTF_CFLOAT64) == 16u);
    CHECK(sptf_sample_size(SPTF_INT24) == 3u);
    CHECK(sptf_sample_size((sptf_sample_type_t)0) == 0u); /* NOLINT(clang-analyzer-optin.core.EnumCastOutOfRange): kasıtlı geçersiz değer */
}

static void test_link_init_errors(void)
{
    sptf_link_t link;
    sptf_link_config_t lc;
    mock_t m;
    (void)memset(&lc, 0, sizeof(lc));
    CHECK(sptf_link_init(NULL, &lc) == SPTF_ERR_ARG);
    CHECK(sptf_link_init(&link, &lc) == SPTF_ERR_CONFIG); /* taşıma yok */
    lc.transport.acquire = mock_acquire;
    lc.transport.send = mock_send;
    lc.transport.ctx = &m;
    lc.max_datagram = 65508u;
    CHECK(sptf_link_init(&link, &lc) == SPTF_ERR_CONFIG);
    lc.max_datagram = 0u;
    lc.descriptor_period_ms = 1001u;
    CHECK(sptf_link_init(&link, &lc) == SPTF_ERR_CONFIG);
    lc.descriptor_period_ms = 0u;
    CHECK(sptf_link_init(&link, &lc) == SPTF_OK);
    CHECK(link.cfg.max_datagram == SPTF_CFG_MAX_DATAGRAM);
    CHECK(link.cfg.descriptor_period_ms == SPTF_CFG_DESCRIPTOR_PERIOD_MS);
}

/** @brief 22 kanal DESCRIPTOR'ı 1472 bayta sığmaz; 21 sığar. Aynı stream_id reddedilir. */
static void test_stream_init_limits(void)
{
    static sptf_channel_t many[22];
    sptf_link_t link;
    mock_t m;
    sptf_stream_t s1;
    sptf_stream_t s2;
    sptf_stream_config_t c;
    uint32_t i;
    for (i = 0u; i < 22u; i++)
    {
        many[i].name = "k";
        many[i].unit = "";
        many[i].scale = 1.0;
        many[i].offset = 0.0;
    }
    link_with(&link, &m, 0u, false);
    c = cfg_int16(22u, many);
    CHECK(sptf_stream_init(&s1, &link, &c) == SPTF_ERR_SIZE);
    c = cfg_int16(21u, many);
    CHECK(sptf_stream_init(&s1, &link, &c) == SPTF_OK);
    CHECK(sptf_descriptor_size(21u, false) == 1424u);
    CHECK(sptf_stream_init(&s1, &link, &c) == SPTF_ERR_ARG);   /* zaten ekli */
    CHECK(sptf_stream_init(&s2, &link, &c) == SPTF_ERR_CONFIG); /* aynı id */
    c.stream_id = 2u;
    CHECK(sptf_stream_init(&s2, &link, &c) == SPTF_OK);
    CHECK(sptf_stream_deinit(&s2) == SPTF_OK);
    CHECK(sptf_stream_write(&s2, many, 1u, 0u) == SPTF_ERR_STATE);
    CHECK(sptf_stream_init(&s2, &link, &c) == SPTF_OK);
}

/** @brief İlk yazma: DESCRIPTOR + bölünmüş DATA; seq, first_sample, sayaçlar. */
static void test_write_split(void)
{
    sptf_link_t link;
    mock_t m;
    sptf_stream_t s;
    sptf_stream_config_t c = cfg_int16(2u, CH2);
    static uint8_t block[1000 * 4];
    uint32_t i;
    uint32_t per;
    for (i = 0u; i < 2000u; i++)
    {
        host16(&block[(size_t)2u * i], (uint16_t)i);
    }
    link_with(&link, &m, 0u, false);
    CHECK(sptf_stream_init(&s, &link, &c) == SPTF_OK);
    per = sptf_stream_samples_per_frame(&s);
    CHECK(per == (1472u - 48u) / 4u); /* 356 */
    CHECK(sptf_stream_write(&s, block, 1000u, SPTF_TS_NONE) == SPTF_OK);
    /* DESCRIPTOR, 356, 356, DESCRIPTOR (saatsiz periyot 500 örnek: 712 ≥ 500), 288 */
    CHECK(m.count == 5);
    CHECK(m.dg[0][5] == 1u && m.len[0] == 208u && m.dg[3][5] == 1u);
    CHECK(rd32(&m.dg[0][16]) == 0u && rd32(&m.dg[1][16]) == 1u && rd32(&m.dg[4][16]) == 4u);
    CHECK(rd64(&m.dg[1][32]) == 0u && rd32(&m.dg[1][40]) == 356u);
    CHECK(rd64(&m.dg[2][32]) == 356u);
    CHECK(rd64(&m.dg[4][32]) == 712u && rd32(&m.dg[4][40]) == 288u && m.len[4] == 48u + 288u * 4u);
    CHECK(rd16(&m.dg[4][48]) == 1424u); /* 712. örneğin 0. kanalı = 2 · 712 */
    CHECK(rd16(&m.dg[1][14]) == 1u);   /* desc_rev */
    CHECK(s.stats.data_frames == 3u && s.stats.desc_frames == 2u && s.stats.samples_sent == 1000u);
    CHECK(link.stats.datagrams == 5u && link.stats.bytes == (2u * 208u) + (2u * 1472u) + 48u + (288u * 4u));
    /* sonraki DESCRIPTOR 712 + 500 = 1212. örnekte */
    m.count = 0;
    CHECK(sptf_stream_write(&s, block, 10u, SPTF_TS_NONE) == SPTF_OK);
    CHECK(m.count == 1 && m.dg[0][5] == 2u && rd64(&m.dg[0][32]) == 1000u && rd32(&m.dg[0][16]) == 5u);
    m.count = 0;
    CHECK(sptf_stream_write(&s, block, 202u, SPTF_TS_NONE) == SPTF_OK);
    CHECK(m.count == 1 && rd32(&m.dg[0][16]) == 6u);
    m.count = 0;
    CHECK(sptf_stream_write(&s, block, 1u, SPTF_TS_NONE) == SPTF_OK);
    CHECK(m.count == 2 && m.dg[0][5] == 1u && rd64(&m.dg[1][32]) == 1212u);
}

/** @brief Planar blok bölündüğünde her çerçeve kendi kanal dilimlerini taşır. */
static void test_planar_split(void)
{
    static const sptf_channel_t ch3[3] = {{"x", "", 1.0, 0.0}, {"y", "", 1.0, 0.0}, {"z", "", 1.0, 0.0}};
    sptf_link_t link;
    mock_t m;
    sptf_stream_t s;
    sptf_stream_config_t c = cfg_int16(3u, ch3);
    static uint8_t block[3 * 100 * 2];
    uint32_t ch;
    uint32_t i;
    c.layout = SPTF_LAYOUT_PLANAR;
    for (ch = 0u; ch < 3u; ch++)
    {
        for (i = 0u; i < 100u; i++)
        {
            host16(&block[(size_t)2u * ((ch * 100u) + i)], (uint16_t)((ch * 1000u) + i));
        }
    }
    link_with(&link, &m, 260u, false);
    CHECK(sptf_stream_init(&s, &link, &c) == SPTF_ERR_SIZE); /* DESCRIPTOR (272 B) sığmaz */
    link_with(&link, &m, 300u, false);
    CHECK(sptf_stream_init(&s, &link, &c) == SPTF_OK);
    CHECK(sptf_stream_samples_per_frame(&s) == 42u);
    CHECK(sptf_stream_write(&s, block, 100u, SPTF_TS_NONE) == SPTF_OK);
    CHECK(m.count == 4); /* DESCRIPTOR + 42 + 42 + 16 */
    /* 2. DATA (örnek 42..83): kanal 1'in ilk elemanı 1042, kanal 2'nin son elemanı 2083 */
    CHECK(rd32(&m.dg[2][40]) == 42u);
    CHECK(rd16(&m.dg[2][48]) == 42u);
    CHECK(rd16(&m.dg[2][48u + (42u * 2u)]) == 1042u);
    CHECK(rd16(&m.dg[2][48u + (2u * 42u * 2u) + (41u * 2u)]) == 2083u);
    CHECK(rd32(&m.dg[3][40]) == 16u && rd16(&m.dg[3][48u + (2u * 16u * 2u) + (15u * 2u)]) == 2099u);
}

/** @brief Tampon alınamazsa kalan örnekler kayıp sayılır, first_sample ilerler (alıcıda tam boşluk). */
static void test_nomem_drop(void)
{
    sptf_link_t link;
    mock_t m;
    sptf_stream_t s;
    sptf_stream_config_t c = cfg_int16(2u, CH2);
    static uint8_t block[1000 * 4];
    link_with(&link, &m, 0u, false);
    CHECK(sptf_stream_init(&s, &link, &c) == SPTF_OK);
    m.fail_acquire_at = 2; /* DESCRIPTOR (0), DATA 356 (1), DATA (2) başarısız */
    CHECK(sptf_stream_write(&s, block, 1000u, SPTF_TS_NONE) == SPTF_ERR_NOMEM);
    CHECK(m.count == 2);
    CHECK(s.stats.samples_sent == 356u && s.stats.samples_dropped == 644u);
    CHECK(s.next_sample == 1000u && link.stats.err_nomem == 1u);
    m.count = 0;
    CHECK(sptf_stream_write(&s, block, 10u, SPTF_TS_NONE) == SPTF_OK);
    CHECK(m.count == 2 && rd64(&m.dg[1][32]) == 1000u); /* DESCRIPTOR (periyot) + DATA, boşluk 356..999 */
    /* gönderim hatası: örnekler yine kayıp sayılır */
    m.send_result = SPTF_ERR_IO;
    CHECK(sptf_stream_write(&s, block, 10u, SPTF_TS_NONE) == SPTF_ERR_IO);
    CHECK(s.next_sample == 1020u && s.stats.samples_dropped == 654u && link.stats.err_io == 1u);
    /* ilk DESCRIPTOR gönderilemezse veri de gönderilmez, DESCRIPTOR bekler */
    link_with(&link, &m, 0u, false);
    CHECK(sptf_stream_init(&s, &link, &c) == SPTF_OK);
    m.fail_acquire_at = 0;
    CHECK(sptf_stream_write(&s, block, 5u, SPTF_TS_NONE) == SPTF_ERR_NOMEM);
    CHECK(m.count == 0 && s.desc_pending && s.next_sample == 5u);
    CHECK(sptf_link_poll(&link) == SPTF_OK && m.count == 1 && m.dg[0][5] == 1u && !s.desc_pending);
}

/** @brief Saatli mod: zaman damgaları, AUTO, DESCRIPTOR periyodu ve poll. */
static void test_clock(void)
{
    sptf_link_t link;
    mock_t m;
    sptf_stream_t s;
    sptf_stream_config_t c = cfg_int16(2u, CH2);
    static uint8_t block[1000 * 4];
    link_with(&link, &m, 0u, true);
    CHECK(sptf_stream_init(&s, &link, &c) == SPTF_OK);
    g_now = 5000000000u; /* 5 s */
    CHECK(sptf_stream_write(&s, block, 1000u, SPTF_TS_AUTO) == SPTF_OK);
    /* 1 kHz, 1000 örnek: ilk örnek 4 s'de; 2. çerçeve 356 ms sonra */
    CHECK(rd64(&m.dg[0][24]) == 5000000000u);           /* DESCRIPTOR = şimdi */
    CHECK(rd64(&m.dg[1][24]) == 4000000000u);
    CHECK(rd64(&m.dg[2][24]) == 4356000000u);
    m.count = 0;
    g_now += 499000000u;
    CHECK(sptf_link_poll(&link) == SPTF_OK && m.count == 0);
    g_now += 1000000u;
    CHECK(sptf_link_poll(&link) == SPTF_OK && m.count == 1 && m.dg[0][5] == 1u);
    m.count = 0;
    CHECK(sptf_stream_write(&s, block, 2u, 123456789u) == SPTF_OK);
    CHECK(m.count == 1 && rd64(&m.dg[0][24]) == 123456789u);
    m.count = 0;
    CHECK(sptf_stream_marker(&s, SPTF_MARKER_AT_NEXT, 42u, "durum = RUN", SPTF_TS_AUTO) == SPTF_OK);
    CHECK(m.count == 1 && m.dg[0][5] == 3u && rd64(&m.dg[0][32]) == 1002u && rd32(&m.dg[0][40]) == 42u);
    CHECK(rd16(&m.dg[0][44]) == 11u && memcmp(&m.dg[0][48], "durum = RUN", 11u) == 0 && m.len[0] == 59u);
    CHECK(rd64(&m.dg[0][24]) == g_now);
}

/** @brief Yeniden yapılandırma: desc_rev artar, first_sample 0'a döner, DESCRIPTOR hemen gider. */
static void test_reconfigure(void)
{
    sptf_link_t link;
    mock_t m;
    sptf_stream_t s;
    sptf_stream_config_t c = cfg_int16(2u, CH2);
    static uint8_t block[64];
    link_with(&link, &m, 0u, false);
    CHECK(sptf_stream_init(&s, &link, &c) == SPTF_OK);
    CHECK(sptf_stream_write(&s, block, 8u, SPTF_TS_NONE) == SPTF_OK);
    c.sample_rate_hz = 2000.0;
#if (SPTF_CFG_CRC != 0)
    c.crc = true;
#endif
    CHECK(sptf_stream_reconfigure(&s, &c) == SPTF_OK);
    m.count = 0;
    CHECK(sptf_stream_write(&s, block, 8u, SPTF_TS_NONE) == SPTF_OK);
    CHECK(m.count == 2 && rd16(&m.dg[0][14]) == 2u && rd16(&m.dg[1][14]) == 2u);
#if (SPTF_CFG_CRC != 0)
    CHECK(rd64(&m.dg[1][32]) == 0u && rd16(&m.dg[1][6]) == 1u);
    CHECK(rd32(&m.dg[1][m.len[1] - 4u]) == sptf_crc32_update(0u, m.dg[1], m.len[1] - 4u));
#else
    CHECK(rd64(&m.dg[1][32]) == 0u && rd16(&m.dg[1][6]) == 0u);
#endif
    c.stream_id = 9u;
    CHECK(sptf_stream_reconfigure(&s, &c) == SPTF_ERR_CONFIG);
}

/** @brief Geçersiz çağrılar ve büyük işaret metni. */
static void test_misc_errors(void)
{
    static char big[1500];
    sptf_link_t link;
    mock_t m;
    sptf_stream_t s;
    sptf_stream_t never;
    sptf_stream_config_t c = cfg_int16(2u, CH2);
    uint8_t out[256];
    size_t n = 0u;
    sptf_frame_info_t info = {3u, 1u, 0u, 0u, false};
    (void)memset(big, 'x', sizeof(big) - 1u);
    (void)memset(&never, 0, sizeof(never));
    link_with(&link, &m, 0u, false);
    CHECK(sptf_stream_init(&s, &link, &c) == SPTF_OK);
    CHECK(sptf_stream_write(NULL, out, 1u, 0u) == SPTF_ERR_ARG);
    CHECK(sptf_stream_write(&s, NULL, 1u, 0u) == SPTF_ERR_ARG);
    CHECK(sptf_stream_write(&s, out, 0u, 0u) == SPTF_ERR_ARG);
    CHECK(sptf_stream_write(&never, out, 1u, 0u) == SPTF_ERR_STATE);
    CHECK(sptf_stream_marker(&s, 0u, 1u, big, 0u) == SPTF_ERR_SIZE);
    CHECK(sptf_link_poll(NULL) == SPTF_ERR_ARG);
    CHECK(sptf_encode_marker(out, sizeof(out), &info, 7u, 1u, NULL, &n) == SPTF_OK && n == 48u);
    CHECK(rd16(&out[12]) == 3u && rd16(&out[44]) == 0u);
    CHECK(sptf_encode_descriptor(out, sizeof(out), &info, &c, &n) == SPTF_OK && n == 208u);
    CHECK(sptf_encode_descriptor(out, 207u, &info, &c, &n) == SPTF_ERR_SIZE);
    CHECK(m.releases == 0);
}

/** @brief SPTF_UDP_Uygulama_Protokolu.md Ek C: DESCRIPTOR ve MARKER test vektörleri (bayt bayt). */
static void test_vectors_desc_marker(void)
{
#if (SPTF_CFG_CRC != 0)
    static const uint8_t desc_expect[148] = {
        0x53, 0x50, 0x54, 0x46, 0x01, 0x01, 0x01, 0x00, 0x94, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x8F, 0x40, 0x01, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x41, 0x44, 0x43, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x49, 0x61, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x41, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0xFC, 0xA9, 0xF1, 0xD2, 0x4D, 0x62, 0x50, 0x3F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0xBA, 0x27, 0x3D, 0xFC};
    static const uint8_t marker_expect[55] = {
        0x53, 0x50, 0x54, 0x46, 0x01, 0x03, 0x01, 0x00, 0x37, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
        0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x84, 0x1E, 0x00, 0x00, 0x00, 0x00, 0x00,
        0xE8, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2A, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
        0x52, 0x55, 0x4E, 0x37, 0x7D, 0x55, 0x30};
    static const sptf_channel_t ch[1] = {{"Ia", "A", 0.001, 0.0}};
    sptf_stream_config_t c = cfg_int16(1u, ch);
    sptf_frame_info_t info = {1u, 1u, 0u, 0u, true};
    uint8_t out[160];
    size_t n = 0u;
    c.name = "ADC";
    CHECK(sptf_encode_descriptor(out, sizeof(out), &info, &c, &n) == SPTF_OK);
    CHECK(n == 148u && memcmp(out, desc_expect, 148u) == 0);
    info.seq = 5u;
    info.timestamp_ns = 2000000u;
    CHECK(sptf_encode_marker(out, sizeof(out), &info, 1000u, 42u, "RUN", &n) == SPTF_OK);
    CHECK(n == 55u && memcmp(out, marker_expect, 55u) == 0);
#endif
}

int main(void)
{
    test_crc();
    test_vector();
    test_vectors_desc_marker();
    test_config_validation();
    test_link_init_errors();
    test_stream_init_limits();
    test_write_split();
    test_planar_split();
    test_nomem_drop();
    test_clock();
    test_reconfigure();
    test_misc_errors();
    (void)printf("%s: %d denetim, %d hata (bayt sırası: %s)\n", (g_fail == 0) ? "GECTI" : "KALDI", g_checks, g_fail,
                 (SPTF_CFG_HOST_BIG_ENDIAN != 0) ? "buyuk uclu" : "kucuk uclu");
    return (g_fail == 0) ? 0 : 1;
}
