/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    sptf_dump.c
 * @brief   Çapraz doğrulama aracı: kütüphanenin ürettiği datagramları dosyaya yazar.
 *
 * Gerçek sptf_udp API'si lwIP saplaması üzerinden çalıştırılır; saplamanın yakaladığı
 * her datagram dosyaya [u32 uzunluk][yük] olarak yazılır. tests/test_embedded_sptf.py
 * aynı senaryoyu Signal Processing Toolbox'ın Python kodlayıcısıyla üretip baytları
 * karşılaştırır ve datagramları arayüzün alıcısından geçirir.
 *
 * Kullanım:
 *   sptf_dump <dosya> <tip> <yerleşim> <crc> <kanal> <blok_örnek> <blok> <datagram> <hız> <ts0> <tohum>
 *
 * Senaryo: stream_id 7, ad "dump stream", kanal i = "c<i>", birim çift kanalda "V", tek
 * kanalda "°C", ölçek 1 + 0,5 i, ofset -i. Blok b'nin k. baytı (k * 131 + tohum + 17 b) mod 256.
 * ts0 0 değilse blok b'nin zaman damgası ts0 + 1000 b'dir. İlk bloktan sonra bir MARKER
 * gönderilir (kod 0xC0DE, metin "event μs" (Yunan mu, U+03BC), zaman 777).
 *
 * Test kodu yalnız PC'de çalışır ve MISRA kapsamı dışındadır.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lwip_stub.h"
#include "sptf_frame.h"
#include "sptf_udp.h"

#define MAX_CHANNELS        (256U)
#define MAX_BLOCK_BYTES     (4U * 1024U * 1024U)

static sptf_channel_t channels[MAX_CHANNELS];
static char names[MAX_CHANNELS][8];
static uint8_t block[MAX_BLOCK_BYTES];

static int write_datagrams(const char *path)
{
    FILE *file = fopen(path, "wb");
    uint32_t index;
    int result = 0;

    if (file == NULL)
    {
        result = 1;
    }
    else
    {
        for (index = 0U; index < stub_datagram_count(); index++)
        {
            uint32_t length = 0U;
            const uint8_t *data = stub_datagram(index, &length);
            const uint8_t prefix[4] =
            {
                (uint8_t)(length & 0xFFU), (uint8_t)((length >> 8U) & 0xFFU),
                (uint8_t)((length >> 16U) & 0xFFU), (uint8_t)((length >> 24U) & 0xFFU)
            };

            if ((fwrite(prefix, 1U, 4U, file) != 4U) || (fwrite(data, 1U, length, file) != length))
            {
                result = 1;
            }
        }

        if (fclose(file) != 0)
        {
            result = 1;
        }
    }

    return result;
}

int main(int argc, char **argv)
{
    sptf_udp_config_t config;
    sptf_udp_stream_t *stream = NULL;
    sptf_udp_stats_t stats;
    uint32_t channel_count;
    uint32_t block_samples;
    uint32_t block_count;
    uint32_t sample_size;
    uint32_t block_bytes;
    uint64_t first_time;
    uint32_t seed;
    uint32_t index;
    uint32_t b;
    sptf_status_t status;

    if (argc != 12)
    {
        (void)fprintf(stderr, "usage: sptf_dump out type layout crc channels samples blocks datagram rate ts0 seed\n");
        return 2;
    }

    channel_count = (uint32_t)strtoul(argv[5], NULL, 10);
    block_samples = (uint32_t)strtoul(argv[6], NULL, 10);
    block_count = (uint32_t)strtoul(argv[7], NULL, 10);
    first_time = (uint64_t)strtoull(argv[10], NULL, 10);
    seed = (uint32_t)strtoul(argv[11], NULL, 10);

    if ((channel_count == 0U) || (channel_count > MAX_CHANNELS))
    {
        (void)fprintf(stderr, "invalid channel count\n");
        return 2;
    }

    for (index = 0U; index < channel_count; index++)
    {
        (void)snprintf(names[index], sizeof(names[index]), "c%u", (unsigned)index);
        channels[index].name = names[index];
        channels[index].unit = ((index % 2U) == 0U) ? "V" : "\xC2\xB0" "C";
        channels[index].scale = 1.0 + (0.5 * (double)index);
        channels[index].offset = -(double)index;
    }

    stub_reset();
    (void)sptf_udp_config_init(&config);
    config.desc.name = "dump stream";
    config.desc.channels = channels;
    config.desc.sample_rate_hz = strtod(argv[9], NULL);
    config.desc.channel_count = (uint16_t)channel_count;
    config.desc.sample_type = (uint8_t)strtoul(argv[2], NULL, 0);
    config.desc.layout = (uint8_t)strtoul(argv[3], NULL, 10);
    config.remote_ip[0] = 10U;
    config.remote_ip[1] = 0U;
    config.remote_ip[2] = 0U;
    config.remote_ip[3] = 2U;
    config.stream_id = 7U;
    config.max_datagram_size = (uint16_t)strtoul(argv[8], NULL, 10);
    config.crc_enabled = (strtoul(argv[4], NULL, 10) != 0U);

    status = sptf_udp_open(&config, &stream);

    if (status != SPTF_STATUS_OK)
    {
        (void)fprintf(stderr, "open failed: %d\n", (int)status);
        return 1;
    }

    sample_size = sptf_sample_type_size(config.desc.sample_type);
    block_bytes = block_samples * sample_size * channel_count;

    if (block_bytes > MAX_BLOCK_BYTES)
    {
        (void)fprintf(stderr, "block too large\n");
        return 2;
    }

    for (b = 0U; b < block_count; b++)
    {
        const uint64_t time_ns = (first_time != 0U) ? (first_time + ((uint64_t)b * 1000U)) : 0U;

        for (index = 0U; index < block_bytes; index++)
        {
            block[index] = (uint8_t)(((index * 131U) + seed + (17U * b)) & 0xFFU);
        }

        status = sptf_udp_send_samples(stream, block, block_samples, time_ns);

        if (status != SPTF_STATUS_OK)
        {
            (void)fprintf(stderr, "send failed: %d\n", (int)status);
            return 1;
        }

        if (b == 0U)
        {
            (void)sptf_udp_get_stats(stream, &stats);
            status = sptf_udp_send_marker(stream, stats.next_first_sample, 0xC0DEU, "event \xCE\xBC" "s", 777U);

            if (status != SPTF_STATUS_OK)
            {
                (void)fprintf(stderr, "marker failed: %d\n", (int)status);
                return 1;
            }
        }
    }

    if ((stub_live_pbufs() != 0U) || (write_datagrams(argv[1]) != 0))
    {
        (void)fprintf(stderr, "write failed\n");
        return 1;
    }

    (void)printf("ok %u %u\n", (unsigned)stub_datagram_count(),
                 (unsigned)sptf_max_samples_per_frame(sample_size * channel_count,
                                                      (uint32_t)config.max_datagram_size, config.crc_enabled));
    (void)sptf_udp_close(stream);

    return 0;
}
