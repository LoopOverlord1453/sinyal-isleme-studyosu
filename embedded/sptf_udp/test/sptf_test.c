/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    sptf_test.c
 * @brief   SPTF-UDP kütüphanesinin PC birim testleri (lwIP saplamasıyla).
 *
 * Derleme (embedded/sptf_udp klasöründen):
 *   gcc -std=c99 -Wall -Wextra -Werror -IInc -Itest/stub
 *       Src/sptf_crc32.c Src/sptf_frame.c Src/sptf_udp.c test/stub/lwip_stub.c test/sptf_test.c
 * Çekirdek kilitli yapılandırma için ayrıca -DNO_SYS=0 verilir.
 *
 * Test kodu yalnız PC'de çalışır; çıktı için <stdio.h> kullanır ve MISRA kapsamı dışındadır.
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "lwip/opt.h"
#include "lwip_stub.h"
#include "sptf_crc32.h"
#include "sptf_frame.h"
#include "sptf_udp.h"

/* ============================================================== test altyapısı */

static uint32_t checks_run;
static uint32_t checks_failed;

static void check(bool condition, const char *description, int line)
{
    checks_run++;

    if (!condition)
    {
        checks_failed++;
        (void)printf("  FAIL (line %d): %s\n", line, description);
    }
}

#define CHECK(condition)    check((condition), #condition, __LINE__)

static uint16_t get_u16(const uint8_t *data, uint32_t index)
{
    return (uint16_t)((uint32_t)data[index] | ((uint32_t)data[index + 1U] << 8U));
}

static uint32_t get_u32(const uint8_t *data, uint32_t index)
{
    return (uint32_t)data[index] | ((uint32_t)data[index + 1U] << 8U) | ((uint32_t)data[index + 2U] << 16U)
           | ((uint32_t)data[index + 3U] << 24U);
}

static uint64_t get_u64(const uint8_t *data, uint32_t index)
{
    return (uint64_t)get_u32(data, index) | ((uint64_t)get_u32(data, index + 4U) << 32U);
}

/** data[offset..] beklenen baytlarla aynı mı; yakalanmamış datagramda (NULL) çökmeden yanlış döner. */
static bool same_bytes(const uint8_t *data, uint32_t offset, const void *expected, uint32_t length)
{
    return (data != NULL) && (memcmp(&data[offset], expected, (size_t)length) == 0);
}

/** data[index]; yakalanmamış datagramda 0. */
static uint8_t byte_at(const uint8_t *data, uint32_t index)
{
    return (data != NULL) ? data[index] : 0U;
}

/* Yakalanan datagramın alanları */
typedef struct
{
    const uint8_t *data;
    uint32_t length;
    uint8_t frame_type;
    uint16_t flags;
    uint32_t frame_len;
    uint16_t stream_id;
    uint16_t desc_rev;
    uint32_t seq;
    uint64_t timestamp_ns;
    uint64_t first_sample;              /* DATA: first_sample, MARKER: sample_index */
    uint32_t count;                     /* DATA: sample_count, MARKER: code */
} frame_view_t;

static frame_view_t captured(uint32_t index)
{
    frame_view_t view;

    (void)memset(&view, 0, sizeof(view));
    view.data = stub_datagram(index, &view.length);

    if ((view.data != NULL) && (view.length >= SPTF_HEADER_SIZE))
    {
        view.frame_type = view.data[5];
        view.flags = get_u16(view.data, 6U);
        view.frame_len = get_u32(view.data, 8U);
        view.stream_id = get_u16(view.data, 12U);
        view.desc_rev = get_u16(view.data, 14U);
        view.seq = get_u32(view.data, 16U);
        view.timestamp_ns = get_u64(view.data, 24U);

        if ((view.frame_type != SPTF_FRAME_DESCRIPTOR) && (view.length >= 48U))
        {
            view.first_sample = get_u64(view.data, 32U);
            view.count = get_u32(view.data, 40U);
        }
    }

    return view;
}

/** Bütün yakalanan datagramların genel kurallara uyduğunu sınar. */
static void check_all_datagrams(uint32_t max_datagram)
{
    uint32_t index;

    for (index = 0U; index < stub_datagram_count(); index++)
    {
        const frame_view_t view = captured(index);

        CHECK(view.length >= SPTF_HEADER_SIZE);
        CHECK(view.length <= max_datagram);
        CHECK(same_bytes(view.data, 0U, "SPTF", 4U));
        CHECK(byte_at(view.data, 4U) == SPTF_PROTOCOL_VERSION);
        CHECK(view.frame_len == view.length);
        CHECK((view.flags & 0xFFFEU) == 0U);
        CHECK(get_u32(view.data, 20U) == 0U);

        if ((view.flags & SPTF_FLAG_CRC32) != 0U)
        {
            const uint32_t stored = get_u32(view.data, view.length - 4U);

            CHECK(stored == sptf_crc32_compute(view.data, view.length - 4U));
        }
    }
}

static void check_no_leaks(void)
{
    CHECK(stub_live_pbufs() == 0U);
    CHECK(stub_lock_violations() == 0U);
}

/* ============================================================== akış tanımları */

static const sptf_channel_t two_currents[2] =
{
    { "Ia", "A", 0.001, 0.0 },
    { "Ib", "A", 0.001, 0.0 }
};

static sptf_stream_desc_t current_desc(void)
{
    sptf_stream_desc_t desc;

    desc.name = "Motor currents";
    desc.channels = two_currents;
    desc.sample_rate_hz = 1000.0;
    desc.channel_count = 2U;
    desc.sample_type = SPTF_SAMPLE_INT16;
    desc.layout = SPTF_LAYOUT_INTERLEAVED;

    return desc;
}

static sptf_udp_config_t default_config(uint16_t stream_id)
{
    sptf_udp_config_t config;

    (void)sptf_udp_config_init(&config);
    config.desc = current_desc();
    config.remote_ip[0] = 192U;
    config.remote_ip[1] = 168U;
    config.remote_ip[2] = 1U;
    config.remote_ip[3] = 100U;
    config.stream_id = stream_id;
    config.max_datagram_size = SPTF_STANDARD_DATAGRAM_SIZE;

    return config;
}

static uint8_t ramp_block[65536];

static const uint8_t *ramp(uint32_t sample_count, uint32_t bytes_per_instant)
{
    uint32_t index;

    for (index = 0U; (index < (sample_count * bytes_per_instant)) && (index < sizeof(ramp_block)); index++)
    {
        ramp_block[index] = (uint8_t)(index & 0xFFU);
    }

    return ramp_block;
}

/* ================================================================ CRC-32 */

static void test_crc32(void)
{
    const uint8_t text[9] = { '1', '2', '3', '4', '5', '6', '7', '8', '9' };
    uint32_t partial;

    CHECK(sptf_crc32_compute(text, 9U) == 0xCBF43926U);
    CHECK(sptf_crc32_compute(NULL, 0U) == 0U);

    partial = sptf_crc32_update(SPTF_CRC32_INITIAL_VALUE, text, 4U);
    partial = sptf_crc32_update(partial, &text[4], 5U);
    CHECK(sptf_crc32_finish(partial) == 0xCBF43926U);
}

/* ======================================================== test vektörleri (Ek C) */

static void test_vector_data(void)
{
    static const uint8_t expected[60] =
    {
        0x53, 0x50, 0x54, 0x46, 0x01, 0x02, 0x01, 0x00, 0x3C, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x42, 0x0F, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x64, 0x00, 0xFF, 0xFF, 0xC8, 0x00, 0xFE, 0xFF, 0xA8, 0x40, 0x1E, 0x68
    };
    const int16_t values[4] = { 100, -1, 200, -2 };
    const sptf_stream_desc_t desc = current_desc();
    sptf_frame_header_t header = { 1000000U, 1U, 1U, 1U, true };
    sptf_data_slice_t slice;
    uint8_t buffer[128];
    uint32_t size = 0U;

    slice.block = (const uint8_t *)values;
    slice.first_sample = 0U;
    slice.block_length = 2U;
    slice.offset = 0U;
    slice.count = 2U;

    CHECK(sptf_encode_data(&header, &desc, &slice, buffer, sizeof(buffer), &size) == SPTF_STATUS_OK);
    CHECK(size == 60U);
    CHECK(memcmp(buffer, expected, sizeof(expected)) == 0);

    header.crc_enabled = false;
    CHECK(sptf_encode_data(&header, &desc, &slice, buffer, sizeof(buffer), &size) == SPTF_STATUS_OK);
    CHECK(size == 56U);
    CHECK(buffer[6] == 0U);
    CHECK(get_u32(buffer, 8U) == 56U);
}

static void test_vector_descriptor(void)
{
    static const uint8_t expected[148] =
    {
        0x53, 0x50, 0x54, 0x46, 0x01, 0x01, 0x01, 0x00, 0x94, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x8F, 0x40, 0x01, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x41, 0x44, 0x43, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x49, 0x61, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x41, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0xFC, 0xA9, 0xF1, 0xD2, 0x4D, 0x62, 0x50, 0x3F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0xBA, 0x27, 0x3D, 0xFC
    };
    const sptf_channel_t channel = { "Ia", "A", 0.001, 0.0 };
    sptf_stream_desc_t desc;
    const sptf_frame_header_t header = { 0U, 0U, 1U, 1U, true };
    uint8_t buffer[256];
    uint32_t size = 0U;

    desc.name = "ADC";
    desc.channels = &channel;
    desc.sample_rate_hz = 1000.0;
    desc.channel_count = 1U;
    desc.sample_type = SPTF_SAMPLE_INT16;
    desc.layout = SPTF_LAYOUT_INTERLEAVED;

    CHECK(sptf_encode_descriptor(&header, &desc, buffer, sizeof(buffer), &size) == SPTF_STATUS_OK);
    CHECK(size == 148U);
    CHECK(memcmp(buffer, expected, sizeof(expected)) == 0);
    CHECK(sptf_encode_descriptor(&header, &desc, buffer, 147U, &size) == SPTF_STATUS_BUFFER_TOO_SMALL);
}

static void test_vector_marker(void)
{
    static const uint8_t expected[55] =
    {
        0x53, 0x50, 0x54, 0x46, 0x01, 0x03, 0x01, 0x00, 0x37, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
        0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x84, 0x1E, 0x00, 0x00, 0x00, 0x00, 0x00,
        0xE8, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2A, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
        0x52, 0x55, 0x4E, 0x37, 0x7D, 0x55, 0x30
    };
    const sptf_frame_header_t header = { 2000000U, 5U, 1U, 1U, true };
    uint8_t buffer[128];
    uint32_t size = 0U;

    CHECK(sptf_encode_marker(&header, 1000U, 42U, "RUN", buffer, sizeof(buffer), &size) == SPTF_STATUS_OK);
    CHECK(size == 55U);
    CHECK(memcmp(buffer, expected, sizeof(expected)) == 0);

    /* Metin tampona tam sığar ya da sığmaz. */
    CHECK(sptf_encode_marker(&header, 0U, 0U, "RUN", buffer, 55U, &size) == SPTF_STATUS_OK);
    CHECK(sptf_encode_marker(&header, 0U, 0U, "RUNS", buffer, 55U, &size) == SPTF_STATUS_TEXT_TOO_LONG);
    CHECK(sptf_encode_marker(&header, 0U, 0U, "", buffer, 52U, &size) == SPTF_STATUS_OK);
    CHECK(size == 52U);
    CHECK(sptf_encode_marker(&header, 0U, 0U, "", buffer, 51U, &size) == SPTF_STATUS_BUFFER_TOO_SMALL);
}

/* ============================================================ boyut hesapları */

static void test_sizes(void)
{
    const uint8_t types[17] =
    {
        SPTF_SAMPLE_INT8, SPTF_SAMPLE_UINT8, SPTF_SAMPLE_INT16, SPTF_SAMPLE_UINT16, SPTF_SAMPLE_INT24,
        SPTF_SAMPLE_UINT24, SPTF_SAMPLE_INT32, SPTF_SAMPLE_UINT32, SPTF_SAMPLE_INT64, SPTF_SAMPLE_UINT64,
        SPTF_SAMPLE_FLOAT16, SPTF_SAMPLE_FLOAT32, SPTF_SAMPLE_FLOAT64, SPTF_SAMPLE_CINT16,
        SPTF_SAMPLE_CINT32, SPTF_SAMPLE_CFLOAT32, SPTF_SAMPLE_CFLOAT64
    };
    const uint32_t sizes[17] = { 1U, 1U, 2U, 2U, 3U, 3U, 4U, 4U, 8U, 8U, 2U, 4U, 8U, 4U, 8U, 8U, 16U };
    uint32_t index;

    for (index = 0U; index < 17U; index++)
    {
        CHECK(sptf_sample_type_size(types[index]) == sizes[index]);
    }

    CHECK(sptf_sample_type_size(0x00U) == 0U);
    CHECK(sptf_sample_type_size(0x0BU) == 0U);
    CHECK(sptf_sample_type_size(0x24U) == 0U);

    /* Ek B.1 ve B.2 */
    CHECK(sptf_max_samples_per_frame(2U, 1472U, false) == 712U);
    CHECK(sptf_max_samples_per_frame(2U, 1472U, true) == 710U);
    CHECK(sptf_max_samples_per_frame(16U, 1472U, true) == 88U);
    CHECK(sptf_max_samples_per_frame(24U, 1472U, false) == 59U);
    CHECK(sptf_max_samples_per_frame(2U, 8972U, false) == 4462U);
    CHECK(sptf_max_samples_per_frame(4U, 48U, false) == 0U);
    CHECK(sptf_descriptor_frame_size(21U, true) <= 1472U);
    CHECK(sptf_descriptor_frame_size(22U, false) > 1472U);
    CHECK(sptf_descriptor_frame_size(1U, false) == 144U);
    CHECK(sptf_data_frame_size(4U, 356U, false) == 1472U);
    CHECK(sptf_data_frame_size(0U, 1U, false) == 0U);
    CHECK(sptf_data_frame_size(4096U, 0xFFFFFFF0U, false) == 0U);
}

/* =============================================================== doğrulama [G-32] */

static void test_validation(void)
{
    char long_name[34];
    sptf_channel_t channel = { "c", "V", 1.0, 0.0 };
    sptf_stream_desc_t desc = current_desc();

    CHECK(sptf_check_stream_desc(&desc) == SPTF_STATUS_OK);
    CHECK(sptf_check_stream_desc(NULL) == SPTF_STATUS_NULL_POINTER);

    desc.sample_rate_hz = 0.0009;
    CHECK(sptf_check_stream_desc(&desc) == SPTF_STATUS_INVALID_RATE);
    desc.sample_rate_hz = 0.001;
    CHECK(sptf_check_stream_desc(&desc) == SPTF_STATUS_OK);
    desc.sample_rate_hz = 1.0e12;
    CHECK(sptf_check_stream_desc(&desc) == SPTF_STATUS_OK);
    desc.sample_rate_hz = 1.1e12;
    CHECK(sptf_check_stream_desc(&desc) == SPTF_STATUS_INVALID_RATE);
    desc.sample_rate_hz = (double)NAN;
    CHECK(sptf_check_stream_desc(&desc) == SPTF_STATUS_INVALID_RATE);
    desc.sample_rate_hz = (double)INFINITY;
    CHECK(sptf_check_stream_desc(&desc) == SPTF_STATUS_INVALID_RATE);

    desc = current_desc();
    desc.channel_count = 0U;
    CHECK(sptf_check_stream_desc(&desc) == SPTF_STATUS_INVALID_CHANNELS);
    desc.channel_count = 257U;
    CHECK(sptf_check_stream_desc(&desc) == SPTF_STATUS_INVALID_CHANNELS);

    desc = current_desc();
    desc.sample_type = 0x13U;
    CHECK(sptf_check_stream_desc(&desc) == SPTF_STATUS_INVALID_SAMPLE_TYPE);
    desc = current_desc();
    desc.layout = 2U;
    CHECK(sptf_check_stream_desc(&desc) == SPTF_STATUS_INVALID_LAYOUT);

    /* Metin sınırları: 32 bayt sığar, 33 bayt sığmaz. */
    (void)memset(long_name, 'n', sizeof(long_name));
    long_name[32] = '\0';
    desc = current_desc();
    desc.name = long_name;
    CHECK(sptf_check_stream_desc(&desc) == SPTF_STATUS_OK);
    long_name[32] = 'n';
    long_name[33] = '\0';
    CHECK(sptf_check_stream_desc(&desc) == SPTF_STATUS_TEXT_TOO_LONG);

    desc = current_desc();
    desc.channels = &channel;
    desc.channel_count = 1U;
    channel.unit = "0123456789ABCDEF";
    CHECK(sptf_check_stream_desc(&desc) == SPTF_STATUS_OK);
    channel.unit = "0123456789ABCDEFG";
    CHECK(sptf_check_stream_desc(&desc) == SPTF_STATUS_TEXT_TOO_LONG);
    channel.unit = "V";
    channel.name = NULL;
    CHECK(sptf_check_stream_desc(&desc) == SPTF_STATUS_NULL_POINTER);
    channel.name = "c";
    channel.scale = (double)INFINITY;
    CHECK(sptf_check_stream_desc(&desc) == SPTF_STATUS_NOT_FINITE);
    channel.scale = 1.0;
    channel.offset = (double)NAN;
    CHECK(sptf_check_stream_desc(&desc) == SPTF_STATUS_NOT_FINITE);
    channel.offset = -1.0e300;
    CHECK(sptf_check_stream_desc(&desc) == SPTF_STATUS_OK);
}

static void test_full_name_has_no_terminator(void)
{
    static const char name32[33] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ012345";
    sptf_stream_desc_t desc = current_desc();
    const sptf_frame_header_t header = { 0U, 0U, 1U, 1U, false };
    uint8_t buffer[512];
    uint32_t size = 0U;

    desc.name = name32;
    CHECK(sptf_encode_descriptor(&header, &desc, buffer, sizeof(buffer), &size) == SPTF_STATUS_OK);
    CHECK(memcmp(&buffer[48], name32, 32U) == 0);
    CHECK(buffer[80] == (uint8_t)'I');
}

/* ======================================================== DATA kodlayıcısı */

static void test_planar_and_slices(void)
{
    static const sptf_channel_t channels[3] = { { "a", "", 1.0, 0.0 }, { "b", "", 1.0, 0.0 }, { "c", "", 1.0, 0.0 } };
    const int16_t block[15] = { 0, 1, 2, 3, 4, 100, 101, 102, 103, 104, 200, 201, 202, 203, 204 };
    const int16_t expected[9] = { 1, 2, 3, 101, 102, 103, 201, 202, 203 };
    sptf_stream_desc_t desc;
    const sptf_frame_header_t header = { 0U, 0U, 9U, 1U, false };
    sptf_data_slice_t slice;
    uint8_t buffer[256];
    uint32_t size = 0U;

    desc.name = "planar";
    desc.channels = channels;
    desc.sample_rate_hz = 10.0;
    desc.channel_count = 3U;
    desc.sample_type = SPTF_SAMPLE_INT16;
    desc.layout = SPTF_LAYOUT_PLANAR;

    slice.block = (const uint8_t *)block;
    slice.first_sample = 77U;
    slice.block_length = 5U;
    slice.offset = 1U;
    slice.count = 3U;

    CHECK(sptf_encode_data(&header, &desc, &slice, buffer, sizeof(buffer), &size) == SPTF_STATUS_OK);
    CHECK(size == (48U + 18U));
    CHECK(get_u64(buffer, 32U) == 77U);
    CHECK(get_u32(buffer, 40U) == 3U);
    CHECK(memcmp(&buffer[48], expected, sizeof(expected)) == 0);

    /* Aynı blok interleaved okunursa örnek anları ardışık alınır. */
    desc.layout = SPTF_LAYOUT_INTERLEAVED;
    slice.block_length = 5U;
    slice.offset = 2U;
    slice.count = 2U;
    CHECK(sptf_encode_data(&header, &desc, &slice, buffer, sizeof(buffer), &size) == SPTF_STATUS_OK);
    CHECK(memcmp(&buffer[48], &block[6], 12U) == 0);

    /* Aralık ve tampon denetimleri */
    slice.offset = 4U;
    slice.count = 2U;
    CHECK(sptf_encode_data(&header, &desc, &slice, buffer, sizeof(buffer), &size) == SPTF_STATUS_INVALID_ARGUMENT);
    slice.offset = 0U;
    slice.count = 0U;
    CHECK(sptf_encode_data(&header, &desc, &slice, buffer, sizeof(buffer), &size) == SPTF_STATUS_INVALID_ARGUMENT);
    slice.count = 5U;
    CHECK(sptf_encode_data(&header, &desc, &slice, buffer, 77U, &size) == SPTF_STATUS_BUFFER_TOO_SMALL);
    CHECK(sptf_encode_data(&header, &desc, &slice, buffer, 78U, &size) == SPTF_STATUS_OK);
    slice.block = NULL;
    CHECK(sptf_encode_data(&header, &desc, &slice, buffer, sizeof(buffer), &size) == SPTF_STATUS_NULL_POINTER);

    /* first_sample < 2^62 */
    slice.block = (const uint8_t *)block;
    slice.count = 1U;
    slice.first_sample = SPTF_FIRST_SAMPLE_LIMIT - 1U;
    CHECK(sptf_encode_data(&header, &desc, &slice, buffer, sizeof(buffer), &size) == SPTF_STATUS_OK);
    slice.count = 2U;
    CHECK(sptf_encode_data(&header, &desc, &slice, buffer, sizeof(buffer), &size) == SPTF_STATUS_COUNTER_LIMIT);
}

/* =================================================================== UDP: açma */

static void test_open_validation(void)
{
    sptf_udp_config_t config = default_config(1U);
    sptf_udp_stream_t *stream = NULL;

    CHECK(sptf_udp_config_init(NULL) == SPTF_STATUS_NULL_POINTER);
    CHECK(sptf_udp_open(NULL, &stream) == SPTF_STATUS_NULL_POINTER);
    CHECK(sptf_udp_open(&config, NULL) == SPTF_STATUS_NULL_POINTER);

    config.remote_ip[0] = 0U;
    CHECK(sptf_udp_open(&config, &stream) == SPTF_STATUS_INVALID_ADDRESS);
    config.remote_ip[0] = 224U;
    CHECK(sptf_udp_open(&config, &stream) == SPTF_STATUS_INVALID_ADDRESS);
    config.remote_ip[0] = 255U;
    CHECK(sptf_udp_open(&config, &stream) == SPTF_STATUS_INVALID_ADDRESS);

    config = default_config(1U);
    config.remote_port = 0U;
    CHECK(sptf_udp_open(&config, &stream) == SPTF_STATUS_INVALID_ARGUMENT);
    config = default_config(1U);
    config.descriptor_interval_ms = 0U;
    CHECK(sptf_udp_open(&config, &stream) == SPTF_STATUS_INVALID_ARGUMENT);
    config.descriptor_interval_ms = 1001U;
    CHECK(sptf_udp_open(&config, &stream) == SPTF_STATUS_INVALID_ARGUMENT);
    config = default_config(1U);
    config.max_datagram_size = (uint16_t)(SPTF_CFG_MAX_DATAGRAM_SIZE + 1U);
    CHECK(sptf_udp_open(&config, &stream) == SPTF_STATUS_INVALID_ARGUMENT);
    config = default_config(1U);
    config.max_datagram_size = 143U;            /* 1 kanallı DESCRIPTOR 144 bayt */
    config.desc.channel_count = 1U;
    CHECK(sptf_udp_open(&config, &stream) == SPTF_STATUS_DATAGRAM_TOO_SMALL);
    config.max_datagram_size = 144U;
    CHECK(sptf_udp_open(&config, &stream) == SPTF_STATUS_OK);
    CHECK(sptf_udp_close(stream) == SPTF_STATUS_OK);

    config = default_config(1U);
    config.desc.sample_rate_hz = 0.0;
    CHECK(sptf_udp_open(&config, &stream) == SPTF_STATUS_INVALID_RATE);

    /* 22 kanallık DESCRIPTOR 1472 baytlık datagrama sığmaz ([G-08]). */
    {
        static sptf_channel_t many[22];
        uint32_t index;

        for (index = 0U; index < 22U; index++)
        {
            many[index].name = "x";
            many[index].unit = "";
            many[index].scale = 1.0;
            many[index].offset = 0.0;
        }

        config = default_config(1U);
        config.desc.channels = many;
        config.desc.channel_count = 22U;
        CHECK(sptf_udp_open(&config, &stream) == SPTF_STATUS_DATAGRAM_TOO_SMALL);
        config.desc.channel_count = 21U;
        CHECK(sptf_udp_open(&config, &stream) == SPTF_STATUS_OK);
        CHECK(sptf_udp_close(stream) == SPTF_STATUS_OK);
    }

    CHECK(stub_live_pcbs() == 0U);
}

static void test_pool_and_network_errors(void)
{
    sptf_udp_stream_t *streams[SPTF_CFG_MAX_STREAMS];
    sptf_udp_stream_t *extra = NULL;
    sptf_udp_config_t config;
    uint32_t index;

    for (index = 0U; index < SPTF_CFG_MAX_STREAMS; index++)
    {
        config = default_config((uint16_t)(10U + index));
        CHECK(sptf_udp_open(&config, &streams[index]) == SPTF_STATUS_OK);
    }

    config = default_config(10U);
    CHECK(sptf_udp_open(&config, &extra) == SPTF_STATUS_DUPLICATE_STREAM_ID);
    config = default_config(99U);
    CHECK(sptf_udp_open(&config, &extra) == SPTF_STATUS_NO_FREE_STREAM);

    for (index = 0U; index < SPTF_CFG_MAX_STREAMS; index++)
    {
        CHECK(sptf_udp_close(streams[index]) == SPTF_STATUS_OK);
    }

    CHECK(stub_live_pcbs() == 0U);
    CHECK(sptf_udp_close(streams[0]) == SPTF_STATUS_NOT_OPEN);
    CHECK(sptf_udp_close(NULL) == SPTF_STATUS_NULL_POINTER);
    CHECK(sptf_udp_send_samples(streams[0], ramp_block, 1U, 0U) == SPTF_STATUS_NOT_OPEN);

    stub_fail_next_udp_new();
    config = default_config(1U);
    CHECK(sptf_udp_open(&config, &extra) == SPTF_STATUS_NETWORK_ERROR);
    stub_fail_next_bind();
    CHECK(sptf_udp_open(&config, &extra) == SPTF_STATUS_NETWORK_ERROR);
    CHECK(stub_live_pcbs() == 0U);

    /* Hatadan sonra yuva yeniden kullanılabilir. */
    CHECK(sptf_udp_open(&config, &extra) == SPTF_STATUS_OK);
    CHECK(sptf_udp_close(extra) == SPTF_STATUS_OK);
}

/* ======================================================== UDP: akış kuralları */

static void test_descriptor_first_and_split(void)
{
    sptf_udp_config_t config = default_config(3U);
    sptf_udp_stream_t *stream = NULL;
    sptf_udp_stats_t stats;
    frame_view_t view;

    config.local_port = 40000U;
    stub_reset();
    CHECK(sptf_udp_open(&config, &stream) == SPTF_STATUS_OK);
    CHECK(stub_datagram_count() == 0U);           /* açarken çerçeve gönderilmez */

    /* 1000 örnek: 356 + 356 + 288; periyot 500 örnek -> 712. örnekte DESCRIPTOR tekrarı */
    CHECK(sptf_udp_send_samples(stream, ramp(1000U, 4U), 1000U, 0U) == SPTF_STATUS_OK);
    CHECK(stub_datagram_count() == 5U);

    view = captured(0U);
    CHECK((view.frame_type == SPTF_FRAME_DESCRIPTOR) && (view.seq == 0U) && (view.desc_rev == 1U));
    CHECK(view.stream_id == 3U);
    CHECK((view.data != NULL) && (get_u16(view.data, 40U) == 2U));
    CHECK(byte_at(view.data, 42U) == SPTF_SAMPLE_INT16);
    view = captured(1U);
    CHECK((view.frame_type == SPTF_FRAME_DATA) && (view.seq == 1U) && (view.first_sample == 0U)
          && (view.count == 356U));
    CHECK(same_bytes(view.data, 48U, ramp_block, 356U * 4U));
    view = captured(2U);
    CHECK((view.frame_type == SPTF_FRAME_DATA) && (view.seq == 2U) && (view.first_sample == 356U)
          && (view.count == 356U));
    CHECK(same_bytes(view.data, 48U, &ramp_block[356U * 4U], 356U * 4U));
    view = captured(3U);
    CHECK((view.frame_type == SPTF_FRAME_DESCRIPTOR) && (view.seq == 3U));
    view = captured(4U);
    CHECK((view.frame_type == SPTF_FRAME_DATA) && (view.seq == 4U) && (view.first_sample == 712U)
          && (view.count == 288U));

    CHECK(stub_datagram_remote_ip(0U) == 0xC0A80164U);
    CHECK(stub_datagram_remote_port(0U) == SPTF_DEFAULT_UDP_PORT);
    CHECK(stub_datagram_local_port(4U) == 40000U);

    CHECK(sptf_udp_get_stats(stream, &stats) == SPTF_STATUS_OK);
    CHECK(stats.next_first_sample == 1000U);
    CHECK((stats.samples_sent == 1000U) && (stats.samples_lost == 0U));
    CHECK((stats.frames_sent == 5U) && (stats.descriptors_sent == 2U) && (stats.next_seq == 5U));
    CHECK(stats.described);
    check_all_datagrams(1472U);
    check_no_leaks();

    /* Bağımsız değişken denetimleri hiçbir şey göndermez. */
    CHECK(sptf_udp_send_samples(stream, NULL, 1U, 0U) == SPTF_STATUS_NULL_POINTER);
    CHECK(sptf_udp_send_samples(stream, ramp_block, 0U, 0U) == SPTF_STATUS_INVALID_ARGUMENT);
    CHECK(sptf_udp_send_samples(stream, ramp_block, 0x40000000U, 0U) == SPTF_STATUS_INVALID_ARGUMENT);
    CHECK(stub_datagram_count() == 5U);
    CHECK(sptf_udp_get_stats(stream, NULL) == SPTF_STATUS_NULL_POINTER);
    CHECK(sptf_udp_close(stream) == SPTF_STATUS_OK);
}

static void test_timestamps_and_crc(void)
{
    sptf_udp_config_t config = default_config(4U);
    sptf_udp_stream_t *stream = NULL;

    config.crc_enabled = true;
    config.max_datagram_size = 600U;           /* (600 - 52) / 4 = 137 örnek */
    stub_reset();
    CHECK(sptf_udp_open(&config, &stream) == SPTF_STATUS_OK);
    CHECK(sptf_udp_send_samples(stream, ramp(300U, 4U), 300U, 1000000000U) == SPTF_STATUS_OK);
    CHECK(stub_datagram_count() == 4U);
    CHECK(captured(0U).timestamp_ns == 1000000000U);
    CHECK((captured(1U).timestamp_ns == 1000000000U) && (captured(1U).count == 137U));
    CHECK(captured(2U).timestamp_ns == 1137000000U);
    CHECK((captured(3U).timestamp_ns == 1274000000U) && (captured(3U).count == 26U));
    CHECK((captured(1U).flags & SPTF_FLAG_CRC32) != 0U);
    check_all_datagrams(600U);
    check_no_leaks();
    CHECK(sptf_udp_close(stream) == SPTF_STATUS_OK);
}

static void test_descriptor_period(void)
{
    sptf_udp_config_t config = default_config(5U);
    sptf_udp_stream_t *stream = NULL;
    uint32_t block;
    uint32_t descriptors = 0U;

    /* Örnek sayısıyla: 1000 Hz, 100 ms -> her 100 örnekte bir; 25'lik bloklar */
    config.descriptor_interval_ms = 100U;
    stub_reset();
    CHECK(sptf_udp_open(&config, &stream) == SPTF_STATUS_OK);

    for (block = 0U; block < 12U; block++)
    {
        CHECK(sptf_udp_send_samples(stream, ramp(25U, 4U), 25U, 0U) == SPTF_STATUS_OK);
    }

    for (block = 0U; block < stub_datagram_count(); block++)
    {
        if (captured(block).frame_type == SPTF_FRAME_DESCRIPTOR)
        {
            descriptors++;
            CHECK((block % 5U) == 0U);         /* D, 4 DATA, D, ... */
        }
    }

    CHECK(descriptors == 3U);
    CHECK(sptf_udp_close(stream) == SPTF_STATUS_OK);

    /* Süreyle: 10 örneklik bloklar 150 ms arayla -> her bloktan önce DESCRIPTOR */
    stub_reset();
    CHECK(sptf_udp_open(&config, &stream) == SPTF_STATUS_OK);

    for (block = 0U; block < 4U; block++)
    {
        const uint64_t time_ns = 1000000000U + ((uint64_t)block * 150000000U);

        CHECK(sptf_udp_send_samples(stream, ramp(10U, 4U), 10U, time_ns) == SPTF_STATUS_OK);
    }

    CHECK(stub_datagram_count() == 8U);

    for (block = 0U; block < 8U; block++)
    {
        CHECK(captured(block).frame_type == (((block % 2U) == 0U) ? SPTF_FRAME_DESCRIPTOR : SPTF_FRAME_DATA));
    }

    CHECK(sptf_udp_close(stream) == SPTF_STATUS_OK);

    /* Süreyle: 50 ms arayla -> her iki bloktan birinde */
    stub_reset();
    CHECK(sptf_udp_open(&config, &stream) == SPTF_STATUS_OK);

    for (block = 0U; block < 4U; block++)
    {
        const uint64_t time_ns = 1000000000U + ((uint64_t)block * 50000000U);

        CHECK(sptf_udp_send_samples(stream, ramp(10U, 4U), 10U, time_ns) == SPTF_STATUS_OK);
    }

    CHECK(stub_datagram_count() == 6U);
    CHECK(captured(0U).frame_type == SPTF_FRAME_DESCRIPTOR);
    CHECK(captured(3U).frame_type == SPTF_FRAME_DESCRIPTOR);
    CHECK(sptf_udp_close(stream) == SPTF_STATUS_OK);
    check_no_leaks();
}

/* ======================================================== UDP: hata yönetimi */

static void test_buffer_and_send_errors(void)
{
    sptf_udp_config_t config = default_config(6U);
    sptf_udp_stream_t *stream = NULL;
    sptf_udp_stats_t stats;

    /* [G-29] tampon yok: çerçeve üretilmez, seq tüketilmez, örnekler kayıp, first_sample ilerler */
    stub_reset();
    CHECK(sptf_udp_open(&config, &stream) == SPTF_STATUS_OK);
    stub_fail_alloc_call(1U);
    CHECK(sptf_udp_send_samples(stream, ramp(100U, 4U), 100U, 0U) == SPTF_STATUS_NO_BUFFER);
    CHECK(sptf_udp_send_samples(stream, ramp(100U, 4U), 100U, 0U) == SPTF_STATUS_OK);
    CHECK(stub_datagram_count() == 2U);
    CHECK((captured(1U).seq == 1U) && (captured(1U).first_sample == 100U));
    CHECK(sptf_udp_get_stats(stream, &stats) == SPTF_STATUS_OK);
    CHECK((stats.samples_lost == 100U) && (stats.samples_sent == 100U) && (stats.buffer_errors == 1U));
    CHECK(sptf_udp_close(stream) == SPTF_STATUS_OK);

    /* [G-30] gönderim hatası: seq tüketilir */
    stub_reset();
    CHECK(sptf_udp_open(&config, &stream) == SPTF_STATUS_OK);
    stub_fail_send_call(1U);
    CHECK(sptf_udp_send_samples(stream, ramp(100U, 4U), 100U, 0U) == SPTF_STATUS_SEND_FAILED);
    CHECK(sptf_udp_send_samples(stream, ramp(100U, 4U), 100U, 0U) == SPTF_STATUS_OK);
    CHECK(stub_datagram_count() == 2U);
    CHECK((captured(1U).seq == 2U) && (captured(1U).first_sample == 100U));
    CHECK(sptf_udp_get_stats(stream, &stats) == SPTF_STATUS_OK);
    CHECK((stats.samples_lost == 100U) && (stats.send_errors == 1U) && (stats.next_seq == 3U));
    CHECK(sptf_udp_close(stream) == SPTF_STATUS_OK);

    /* [G-09], [G-31] ilk DESCRIPTOR gönderilemedi: DATA gönderilmez, blok kayıp, sonra yeniden denenir */
    stub_reset();
    CHECK(sptf_udp_open(&config, &stream) == SPTF_STATUS_OK);
    stub_fail_send_call(0U);
    CHECK(sptf_udp_send_samples(stream, ramp(100U, 4U), 100U, 0U) == SPTF_STATUS_DESCRIPTOR_PENDING);
    CHECK(stub_datagram_count() == 0U);
    CHECK(sptf_udp_get_stats(stream, &stats) == SPTF_STATUS_OK);
    CHECK((!stats.described) && (stats.samples_lost == 100U) && (stats.next_first_sample == 100U));
    CHECK(sptf_udp_send_marker(stream, 0U, 1U, "x", 0U) == SPTF_STATUS_OK);
    CHECK(stub_datagram_count() == 2U);
    CHECK((captured(0U).frame_type == SPTF_FRAME_DESCRIPTOR) && (captured(0U).seq == 1U));
    CHECK((captured(1U).frame_type == SPTF_FRAME_MARKER) && (captured(1U).seq == 2U));
    CHECK(sptf_udp_send_samples(stream, ramp(10U, 4U), 10U, 0U) == SPTF_STATUS_OK);
    CHECK((captured(2U).frame_type == SPTF_FRAME_DATA) && (captured(2U).first_sample == 100U));
    CHECK(sptf_udp_close(stream) == SPTF_STATUS_OK);

    /* Yayındayken tekrar DESCRIPTOR'ı başarısız: DATA sürer, DESCRIPTOR sonraki çerçevede yinelenir */
    config.descriptor_interval_ms = 100U;       /* 100 örnek */
    config.max_datagram_size = 248U;            /* (248 - 48) / 4 = 50 örnek */
    stub_reset();
    CHECK(sptf_udp_open(&config, &stream) == SPTF_STATUS_OK);
    CHECK(sptf_udp_send_samples(stream, ramp(100U, 4U), 100U, 0U) == SPTF_STATUS_OK);   /* D, 50, 50 */
    stub_fail_send_call(3U);
    CHECK(sptf_udp_send_samples(stream, ramp(100U, 4U), 100U, 0U) == SPTF_STATUS_SEND_FAILED);
    CHECK(stub_datagram_count() == 6U);         /* D, 50, 50 | (D başarısız, seq 3), 50, D, 50 */
    CHECK((captured(3U).frame_type == SPTF_FRAME_DATA) && (captured(3U).first_sample == 100U)
          && (captured(3U).seq == 4U));
    CHECK((captured(4U).frame_type == SPTF_FRAME_DESCRIPTOR) && (captured(4U).seq == 5U));
    CHECK((captured(5U).frame_type == SPTF_FRAME_DATA) && (captured(5U).first_sample == 150U)
          && (captured(5U).seq == 6U));
    CHECK(sptf_udp_get_stats(stream, &stats) == SPTF_STATUS_OK);
    CHECK((stats.samples_lost == 0U) && (stats.samples_sent == 200U) && (stats.send_errors == 1U));
    CHECK(sptf_udp_close(stream) == SPTF_STATUS_OK);
    check_no_leaks();
}

/* ============================================================== MARKER ve yeniden yapılandırma */

static void test_marker(void)
{
    sptf_udp_config_t config = default_config(7U);
    sptf_udp_stream_t *stream = NULL;
    char long_text[1500];
    frame_view_t view;

    stub_reset();
    CHECK(sptf_udp_open(&config, &stream) == SPTF_STATUS_OK);
    CHECK(sptf_udp_send_marker(stream, 1000U, 42U, "state = RUN", 5000U) == SPTF_STATUS_OK);
    CHECK(stub_datagram_count() == 2U);
    CHECK(captured(0U).frame_type == SPTF_FRAME_DESCRIPTOR);
    view = captured(1U);
    CHECK((view.frame_type == SPTF_FRAME_MARKER) && (view.seq == 1U) && (view.timestamp_ns == 5000U));
    CHECK((view.first_sample == 1000U) && (view.count == 42U));
    CHECK((view.data != NULL) && (get_u16(view.data, 44U) == 11U));
    CHECK(same_bytes(view.data, 48U, "state = RUN", 11U));

    (void)memset(long_text, 'm', sizeof(long_text));
    long_text[1424] = '\0';                     /* 1472 - 48 = 1424 bayt sığar */
    CHECK(sptf_udp_send_marker(stream, 0U, 0U, long_text, 0U) == SPTF_STATUS_OK);
    long_text[1424] = 'm';
    long_text[1425] = '\0';
    CHECK(sptf_udp_send_marker(stream, 0U, 0U, long_text, 0U) == SPTF_STATUS_TEXT_TOO_LONG);
    CHECK(sptf_udp_send_marker(stream, 0U, 0U, NULL, 0U) == SPTF_STATUS_NULL_POINTER);
    CHECK(stub_datagram_count() == 3U);
    check_all_datagrams(1472U);
    CHECK(sptf_udp_close(stream) == SPTF_STATUS_OK);
    check_no_leaks();
}

static void test_reconfigure(void)
{
    sptf_udp_config_t config = default_config(8U);
    sptf_udp_stream_t *stream = NULL;
    sptf_stream_desc_t desc = current_desc();
    sptf_udp_stats_t stats;
    frame_view_t view;

    stub_reset();
    CHECK(sptf_udp_open(&config, &stream) == SPTF_STATUS_OK);
    CHECK(sptf_udp_send_samples(stream, ramp(10U, 4U), 10U, 0U) == SPTF_STATUS_OK);

    desc.sample_rate_hz = 2000.0;
    CHECK(sptf_udp_reconfigure(stream, &desc) == SPTF_STATUS_OK);
    CHECK(stub_datagram_count() == 3U);
    view = captured(2U);
    CHECK((view.frame_type == SPTF_FRAME_DESCRIPTOR) && (view.desc_rev == 2U) && (view.seq == 2U));

    CHECK(sptf_udp_send_samples(stream, ramp(10U, 4U), 10U, 0U) == SPTF_STATUS_OK);
    view = captured(3U);
    CHECK((view.frame_type == SPTF_FRAME_DATA) && (view.desc_rev == 2U) && (view.first_sample == 0U));

    /* Geçersiz tanım akışı değiştirmez. */
    desc.channel_count = 0U;
    CHECK(sptf_udp_reconfigure(stream, &desc) == SPTF_STATUS_INVALID_CHANNELS);
    CHECK(sptf_udp_reconfigure(stream, NULL) == SPTF_STATUS_NULL_POINTER);
    CHECK(sptf_udp_get_stats(stream, &stats) == SPTF_STATUS_OK);
    CHECK((stats.desc_rev == 2U) && (stats.next_first_sample == 10U));
    CHECK(sptf_udp_close(stream) == SPTF_STATUS_OK);

    /* desc_rev 65535'ten sonra 0 */
    config.initial_desc_rev = 65535U;
    CHECK(sptf_udp_open(&config, &stream) == SPTF_STATUS_OK);
    desc = current_desc();
    CHECK(sptf_udp_reconfigure(stream, &desc) == SPTF_STATUS_OK);
    CHECK(sptf_udp_get_stats(stream, &stats) == SPTF_STATUS_OK);
    CHECK(stats.desc_rev == 0U);

    /* El ile DESCRIPTOR */
    stub_reset();
    CHECK(sptf_udp_send_descriptor(stream, 77U) == SPTF_STATUS_OK);
    CHECK((stub_datagram_count() == 1U) && (captured(0U).timestamp_ns == 77U));
    CHECK(sptf_udp_send_descriptor(NULL, 0U) == SPTF_STATUS_NULL_POINTER);
    CHECK(sptf_udp_close(stream) == SPTF_STATUS_OK);
    check_no_leaks();
}

/* ====================================================================== ana */

int main(void)
{
#if defined(SPTF_CFG_LWIP_CORE_LOCKING)
    const bool locking = (SPTF_CFG_LWIP_CORE_LOCKING != 0U);
#else
    const bool locking = (NO_SYS == 0);
#endif

    stub_reset();
    stub_expect_locking(locking);

    test_crc32();
    test_vector_data();
    test_vector_descriptor();
    test_vector_marker();
    test_sizes();
    test_validation();
    test_full_name_has_no_terminator();
    test_planar_and_slices();
    test_open_validation();
    test_pool_and_network_errors();
    test_descriptor_first_and_split();
    test_timestamps_and_crc();
    test_descriptor_period();
    test_buffer_and_send_errors();
    test_marker();
    test_reconfigure();

    CHECK(stub_live_pcbs() == 0U);
    CHECK(stub_lock_violations() == 0U);
    CHECK(locking ? (stub_lock_count() > 0U) : (stub_lock_count() == 0U));

    if (checks_failed == 0U)
    {
        (void)printf("PASSED: %u checks (core locking %s)\n", (unsigned)checks_run, locking ? "on" : "off");
    }
    else
    {
        (void)printf("FAILED: %u of %u checks\n", (unsigned)checks_failed, (unsigned)checks_run);
    }

    return (checks_failed == 0U) ? 0 : 1;
}
