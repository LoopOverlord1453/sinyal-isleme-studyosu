/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    sptf_dump.c
 * @brief   Çapraz doğrulama aracı: bir senaryonun ürettiği datagramları dosyaya döker.
 *
 * Python testi (tests/test_embedded_sptf.py) aynı senaryoyu `spt.protocol.builder` ile üretip bayt bayt
 * karşılaştırır ve datagramları arayüzün gerçek alıcısına (StreamStore) verir.
 *
 * Kullanım:
 * @verbatim
 * sptf_dump OUT TYPE LAYOUT CRC CHANNELS BLOCK BLOCKS MAX_DATAGRAM FS TS0 SEED
 * @endverbatim
 * Çıktı: art arda `[u32 uzunluk (küçük uçlu)][datagram]` kayıtları.
 * Blok baytları: `b[i] = (i · 131 + SEED + 17 · blok) mod 256`. İlk bloktan sonra bir işaret gönderilir.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sptf.h"

typedef struct
{
    FILE *f;
    uint8_t buf[70000];
} dump_t;

static uint8_t *dump_acquire(void *ctx, size_t len)
{
    dump_t *d = (dump_t *)ctx;
    return (len <= sizeof(d->buf)) ? d->buf : NULL;
}

static sptf_status_t dump_send(void *ctx, uint8_t *buf, size_t len)
{
    dump_t *d = (dump_t *)ctx;
    uint8_t hdr[4];
    if (len == 0u)
    {
        return SPTF_OK;
    }
    hdr[0] = (uint8_t)(len & 0xFFu);
    hdr[1] = (uint8_t)((len >> 8u) & 0xFFu);
    hdr[2] = (uint8_t)((len >> 16u) & 0xFFu);
    hdr[3] = (uint8_t)((len >> 24u) & 0xFFu);
    if ((fwrite(hdr, 1u, 4u, d->f) != 4u) || (fwrite(buf, 1u, len, d->f) != len))
    {
        return SPTF_ERR_IO;
    }
    return SPTF_OK;
}

static char g_names[256][12];
static sptf_channel_t g_ch[256];
static dump_t g_dump;

int main(int argc, char **argv)
{
    sptf_link_t link;
    sptf_stream_t s;
    sptf_link_config_t lc;
    sptf_stream_config_t c;
    uint8_t *block;
    uint32_t i;
    uint32_t b;
    uint32_t block_n;
    uint32_t blocks;
    uint32_t seed;
    uint64_t ts0;
    size_t nbytes;
    sptf_status_t st;
    if (argc != 12)
    {
        (void)fprintf(stderr, "kullanim: sptf_dump OUT TYPE LAYOUT CRC CHANNELS BLOCK BLOCKS MAXDG FS TS0 SEED\n");
        return 2;
    }
    (void)memset(&c, 0, sizeof(c));
    c.stream_id = 7u;
    c.name = "dump akisi";
    c.sample_type = (sptf_sample_type_t)strtoul(argv[2], NULL, 0);
    c.layout = (sptf_layout_t)strtoul(argv[3], NULL, 0);
    c.crc = strtoul(argv[4], NULL, 0) != 0u;
    c.channel_count = (uint16_t)strtoul(argv[5], NULL, 0);
    block_n = (uint32_t)strtoul(argv[6], NULL, 0);
    blocks = (uint32_t)strtoul(argv[7], NULL, 0);
    c.sample_rate_hz = strtod(argv[9], NULL);
    ts0 = (uint64_t)strtoull(argv[10], NULL, 0);
    seed = (uint32_t)strtoul(argv[11], NULL, 0);
    for (i = 0u; i < c.channel_count; i++)
    {
        (void)snprintf(g_names[i], sizeof(g_names[i]), "k%u", (unsigned)i);
        g_ch[i].name = g_names[i];
        g_ch[i].unit = (i % 2u == 0u) ? "V" : "\xC2\xB0" "C"; /* "°C" UTF-8 */
        g_ch[i].scale = 1.0 + 0.5 * (double)i;
        g_ch[i].offset = -(double)i;
    }
    c.channels = g_ch;
    g_dump.f = fopen(argv[1], "wb");
    if (g_dump.f == NULL)
    {
        return 3;
    }
    (void)memset(&lc, 0, sizeof(lc));
    lc.transport.acquire = dump_acquire;
    lc.transport.send = dump_send;
    lc.transport.ctx = &g_dump;
    lc.max_datagram = (uint32_t)strtoul(argv[8], NULL, 0);
    st = sptf_link_init(&link, &lc);
    if (st == SPTF_OK)
    {
        st = sptf_stream_init(&s, &link, &c);
    }
    if (st != SPTF_OK)
    {
        (void)fclose(g_dump.f);
        (void)printf("init %d\n", (int)st);
        return 4;
    }
    nbytes = (size_t)block_n * sptf_sample_size(c.sample_type) * c.channel_count;
    block = (nbytes > 0u) ? (uint8_t *)malloc(nbytes) : NULL;
    if (block == NULL)
    {
        (void)fclose(g_dump.f);
        return 5;
    }
    for (b = 0u; b < blocks; b++)
    {
        for (i = 0u; i < nbytes; i++)
        {
            block[i] = (uint8_t)(((uint32_t)i * 131u + seed + 17u * b) & 0xFFu);
        }
        st = sptf_stream_write(&s, block, block_n, (ts0 != 0u) ? (ts0 + (uint64_t)b * 1000u) : 0u);
        if (st != SPTF_OK)
        {
            (void)printf("write %d\n", (int)st);
            break;
        }
        if (b == 0u)
        {
            st = sptf_stream_marker(&s, SPTF_MARKER_AT_NEXT, 0xC0DEu, "i\xC5\x9F" "aret \xC3\xBC", 777u);
            if (st != SPTF_OK)
            {
                (void)printf("marker %d\n", (int)st);
                break;
            }
        }
    }
    free(block);
    (void)fclose(g_dump.f);
    if (st != SPTF_OK)
    {
        return 6;
    }
    (void)printf("ok %u %u\n", (unsigned)link.stats.datagrams, (unsigned)sptf_stream_samples_per_frame(&s));
    return 0;
}
