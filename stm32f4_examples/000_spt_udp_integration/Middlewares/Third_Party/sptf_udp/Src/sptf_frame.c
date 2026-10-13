/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    sptf_frame.c
 * @brief   SPTF v1 çerçeve kodlayıcıları (taşımadan bağımsız).
 *
 * Bütün alanlar tampona bayt bayt, küçük uçlu yazılır; tamponun hizalaması ve
 * hedefin bayt sırası varsayılmaz. Örnek verisi kablo düzeninde verildiği için
 * olduğu gibi kopyalanır.
 *
 * Copyright (C) 2026, Kubilay Közleme <burhankozleme@gmail.com>
 */

#include "sptf_frame.h"

#include <float.h>
#include <stddef.h>
#include <string.h>

#include "sptf_crc32.h"

/*
 * DESCRIPTOR'daki f64 alanları (örnekleme hızı, ölçek, ofset) IEEE 754 binary64
 * olarak yazılır. double'ın bit gösterimi doğrudan kopyalandığı için hedefte
 * double'ın binary64 olması gerekir (Cortex-M için AAPCS, PC için x86-64 ABI).
 */
#if (FLT_RADIX != 2) || (DBL_MANT_DIG != 53) || (DBL_MAX_EXP != 1024)
#error "double must be an IEEE 754 binary64 type"
#endif

/* ------------------------------------------------- ortak başlık alan ofsetleri */

#define OFFSET_MAGIC                (0U)
#define OFFSET_VERSION              (4U)
#define OFFSET_FRAME_TYPE           (5U)
#define OFFSET_FLAGS                (6U)
#define OFFSET_FRAME_LENGTH         (8U)
#define OFFSET_STREAM_ID            (12U)
#define OFFSET_DESC_REV             (14U)
#define OFFSET_SEQ                  (16U)
#define OFFSET_HEADER_RESERVED      (20U)
#define OFFSET_TIMESTAMP            (24U)

/* ---------------------------------------------- DESCRIPTOR gövdesi alan ofsetleri */

#define OFFSET_SAMPLE_RATE          (32U)
#define OFFSET_CHANNEL_COUNT        (40U)
#define OFFSET_SAMPLE_TYPE          (42U)
#define OFFSET_LAYOUT               (43U)
#define OFFSET_DESC_RESERVED        (44U)
#define OFFSET_STREAM_NAME          (48U)
#define OFFSET_CHANNEL_RECORDS      (80U)

/* ------------------------------------------- ChannelInfo kaydı içindeki ofsetler */

#define CHANNEL_FIELD_NAME          (0U)
#define CHANNEL_FIELD_UNIT          (32U)
#define CHANNEL_FIELD_SCALE         (48U)
#define CHANNEL_FIELD_OFFSET        (56U)

/* ---------------------------------------------------- DATA gövdesi alan ofsetleri */

#define OFFSET_FIRST_SAMPLE         (32U)
#define OFFSET_SAMPLE_COUNT         (40U)
#define OFFSET_DATA_RESERVED        (44U)
#define OFFSET_SAMPLES              (48U)

/* -------------------------------------------------- MARKER gövdesi alan ofsetleri */

#define OFFSET_SAMPLE_INDEX         (32U)
#define OFFSET_MARKER_CODE          (40U)
#define OFFSET_TEXT_LENGTH          (44U)
#define OFFSET_MARKER_RESERVED      (46U)
#define OFFSET_TEXT                 (48U)

/* ---------------------------------------------------------------- sabit değerler */

/* magic alanı: ASCII "SPTF" */
#define MAGIC_BYTE_0                (0x53U)
#define MAGIC_BYTE_1                (0x50U)
#define MAGIC_BYTE_2                (0x54U)
#define MAGIC_BYTE_3                (0x46U)

#define BYTE_MASK                   (0xFFU)
#define BITS_PER_BYTE               (8U)
#define BYTES_PER_U64               (8U)

/* ======================================================= küçük uçlu yazıcılar */

static void put_u8(uint8_t *buffer, uint32_t index, uint8_t value)
{
    buffer[index] = value;
}

static void put_u16(uint8_t *buffer, uint32_t index, uint16_t value)
{
    buffer[index] = (uint8_t)(value & BYTE_MASK);
    buffer[index + 1U] = (uint8_t)((uint32_t)value >> BITS_PER_BYTE);
}

static void put_u32(uint8_t *buffer, uint32_t index, uint32_t value)
{
    buffer[index] = (uint8_t)(value & BYTE_MASK);
    buffer[index + 1U] = (uint8_t)((value >> 8U) & BYTE_MASK);
    buffer[index + 2U] = (uint8_t)((value >> 16U) & BYTE_MASK);
    buffer[index + 3U] = (uint8_t)((value >> 24U) & BYTE_MASK);
}

static void put_u64(uint8_t *buffer, uint32_t index, uint64_t value)
{
    uint32_t position;

    for (position = 0U; position < BYTES_PER_U64; position++)
    {
        const uint32_t shift = position * BITS_PER_BYTE;

        buffer[index + position] = (uint8_t)((value >> shift) & BYTE_MASK);
    }
}

/** Hedefin küçük uçlu olup olmadığını, bir tamsayının ilk baytına bakarak belirler. */
static bool host_is_little_endian(void)
{
    const uint32_t probe = 1U;
    const uint8_t *probe_bytes = (const uint8_t *)&probe;

    return probe_bytes[0] == 1U;
}

/**
 * double'ın binary64 gösterimini küçük uçlu yazar. Nesnenin baytlarına unsigned char
 * işaretçisiyle erişilir (MISRA C:2012 Kural 11.3 istisnası); double'ın bayt sırası
 * tamsayılarınkiyle aynı kabul edilir (Cortex-M ve x86-64'te böyledir).
 */
static void put_f64(uint8_t *buffer, uint32_t index, sptf_float64_t value)
{
    const uint8_t *representation = (const uint8_t *)&value;
    const bool little_endian = host_is_little_endian();
    uint32_t position;

    for (position = 0U; position < BYTES_PER_U64; position++)
    {
        uint32_t source = position;

        if (!little_endian)
        {
            source = (BYTES_PER_U64 - 1U) - position;
        }

        buffer[index + position] = representation[source];
    }
}

/**
 * Metni sabit uzunluklu alana yazar; alanın kalanını '\0' ile doldurur.
 * Metnin alana sığdığı önceden doğrulanmıştır. Alanı tam dolduran metinde
 * sonlandırıcı yazılmaz (SPT-PRT-002 [G-23]).
 */
static void put_text(uint8_t *buffer, uint32_t index, uint32_t field_size, const char *text)
{
    uint32_t position;
    bool ended = false;

    for (position = 0U; position < field_size; position++)
    {
        uint8_t byte_value = 0U;

        if (!ended)
        {
            if (text[position] == '\0')
            {
                ended = true;
            }
            else
            {
                byte_value = (uint8_t)text[position];
            }
        }

        buffer[index + position] = byte_value;
    }
}

/* ============================================================== yardımcılar */

/**
 * Metnin '\0'a kadarki bayt sayısını verir; en fazla limit + 1 bayt okur.
 * Dönüş değeri limit'ten büyükse metin limit'e sığmıyor demektir.
 */
static uint32_t text_length(const char *text, uint32_t limit)
{
    uint32_t length = 0U;
    bool ended = false;

    while ((!ended) && (length <= limit))
    {
        if (text[length] == '\0')
        {
            ended = true;
        }
        else
        {
            length++;
        }
    }

    return length;
}

/** NaN ve sonsuz değerler bu karşılaştırmaların ikisini birden sağlamaz. */
static bool is_finite(sptf_float64_t value)
{
    return (value >= -DBL_MAX) && (value <= DBL_MAX);
}

static uint32_t crc_size(bool crc_enabled)
{
    uint32_t size = 0U;

    if (crc_enabled)
    {
        size = SPTF_CRC_SIZE;
    }

    return size;
}

static sptf_status_t check_channels(const sptf_channel_t *channels, uint16_t channel_count)
{
    sptf_status_t status = SPTF_STATUS_OK;
    uint32_t channel = 0U;

    while ((status == SPTF_STATUS_OK) && (channel < (uint32_t)channel_count))
    {
        const sptf_channel_t *info = &channels[channel];

        if ((info->name == NULL) || (info->unit == NULL))
        {
            status = SPTF_STATUS_NULL_POINTER;
        }
        else if ((text_length(info->name, SPTF_NAME_FIELD_SIZE) > SPTF_NAME_FIELD_SIZE)
                 || (text_length(info->unit, SPTF_UNIT_FIELD_SIZE) > SPTF_UNIT_FIELD_SIZE))
        {
            status = SPTF_STATUS_TEXT_TOO_LONG;
        }
        else if ((!is_finite(info->scale)) || (!is_finite(info->offset)))
        {
            status = SPTF_STATUS_NOT_FINITE;
        }
        else
        {
            channel++;
        }
    }

    return status;
}

/** DATA çerçevesi için gereken alanları (tip, kanal sayısı, yerleşim) doğrular. */
static sptf_status_t check_data_desc(const sptf_stream_desc_t *desc)
{
    sptf_status_t status = SPTF_STATUS_OK;

    if ((desc->channel_count < 1U) || (desc->channel_count > SPTF_MAX_CHANNELS))
    {
        status = SPTF_STATUS_INVALID_CHANNELS;
    }
    else if (sptf_sample_type_size(desc->sample_type) == 0U)
    {
        status = SPTF_STATUS_INVALID_SAMPLE_TYPE;
    }
    else if ((desc->layout != SPTF_LAYOUT_INTERLEAVED) && (desc->layout != SPTF_LAYOUT_PLANAR))
    {
        status = SPTF_STATUS_INVALID_LAYOUT;
    }
    else
    {
        status = SPTF_STATUS_OK;
    }

    return status;
}

/** Örnek aralığının blok içinde kaldığını ve sayaç sınırını aşmadığını doğrular. */
static sptf_status_t check_slice(const sptf_data_slice_t *slice, uint32_t bytes_per_instant)
{
    sptf_status_t status = SPTF_STATUS_OK;

    if (slice->block == NULL)
    {
        status = SPTF_STATUS_NULL_POINTER;
    }
    else if ((slice->count == 0U) || (slice->offset > slice->block_length)
             || (slice->count > (slice->block_length - slice->offset)))
    {
        status = SPTF_STATUS_INVALID_ARGUMENT;
    }
    else if (slice->block_length > (UINT32_MAX / bytes_per_instant))
    {
        status = SPTF_STATUS_INVALID_ARGUMENT;
    }
    else if ((slice->first_sample >= SPTF_FIRST_SAMPLE_LIMIT)
             || ((uint64_t)slice->count > (SPTF_FIRST_SAMPLE_LIMIT - slice->first_sample)))
    {
        status = SPTF_STATUS_COUNTER_LIMIT;
    }
    else
    {
        status = SPTF_STATUS_OK;
    }

    return status;
}

static void put_header(uint8_t *buffer, const sptf_frame_header_t *header, uint8_t frame_type, uint32_t frame_size)
{
    uint16_t flags = 0U;

    if (header->crc_enabled)
    {
        flags = SPTF_FLAG_CRC32;
    }

    put_u8(buffer, OFFSET_MAGIC, MAGIC_BYTE_0);
    put_u8(buffer, OFFSET_MAGIC + 1U, MAGIC_BYTE_1);
    put_u8(buffer, OFFSET_MAGIC + 2U, MAGIC_BYTE_2);
    put_u8(buffer, OFFSET_MAGIC + 3U, MAGIC_BYTE_3);
    put_u8(buffer, OFFSET_VERSION, SPTF_PROTOCOL_VERSION);
    put_u8(buffer, OFFSET_FRAME_TYPE, frame_type);
    put_u16(buffer, OFFSET_FLAGS, flags);
    put_u32(buffer, OFFSET_FRAME_LENGTH, frame_size);
    put_u16(buffer, OFFSET_STREAM_ID, header->stream_id);
    put_u16(buffer, OFFSET_DESC_REV, header->desc_rev);
    put_u32(buffer, OFFSET_SEQ, header->seq);
    put_u32(buffer, OFFSET_HEADER_RESERVED, 0U);
    put_u64(buffer, OFFSET_TIMESTAMP, header->timestamp_ns);
}

/** Çerçevenin son 4 baytına, önceki baytların CRC-32 değerini yazar. */
static void put_crc(uint8_t *buffer, uint32_t frame_size)
{
    const uint32_t covered = frame_size - SPTF_CRC_SIZE;

    put_u32(buffer, covered, sptf_crc32_compute(buffer, covered));
}

static void put_samples(uint8_t *buffer, const sptf_stream_desc_t *desc, const sptf_data_slice_t *slice,
                        uint32_t bytes_per_instant)
{
    if (desc->layout == SPTF_LAYOUT_INTERLEAVED)
    {
        const uint32_t source_index = slice->offset * bytes_per_instant;
        const uint32_t length = slice->count * bytes_per_instant;

        (void)memcpy(&buffer[OFFSET_SAMPLES], &slice->block[source_index], (size_t)length);
    }
    else
    {
        const uint32_t sample_size = sptf_sample_type_size(desc->sample_type);
        const uint32_t run_length = slice->count * sample_size;
        uint32_t channel;

        /* Her kanalın bloğu block_length örnek uzunluğundadır; aynı aralık sırayla alınır. */
        for (channel = 0U; channel < (uint32_t)desc->channel_count; channel++)
        {
            const uint32_t source_index = ((channel * slice->block_length) + slice->offset) * sample_size;
            const uint32_t target_index = OFFSET_SAMPLES + (channel * run_length);

            (void)memcpy(&buffer[target_index], &slice->block[source_index], (size_t)run_length);
        }
    }
}

/* ================================================================== boyutlar */

uint32_t sptf_sample_type_size(uint8_t sample_type)
{
    uint32_t size;

    switch (sample_type)
    {
        case SPTF_SAMPLE_INT8:
        case SPTF_SAMPLE_UINT8:
            size = 1U;
            break;

        case SPTF_SAMPLE_INT16:
        case SPTF_SAMPLE_UINT16:
        case SPTF_SAMPLE_FLOAT16:
            size = 2U;
            break;

        case SPTF_SAMPLE_INT24:
        case SPTF_SAMPLE_UINT24:
            size = 3U;
            break;

        case SPTF_SAMPLE_INT32:
        case SPTF_SAMPLE_UINT32:
        case SPTF_SAMPLE_FLOAT32:
        case SPTF_SAMPLE_CINT16:
            size = 4U;
            break;

        case SPTF_SAMPLE_INT64:
        case SPTF_SAMPLE_UINT64:
        case SPTF_SAMPLE_FLOAT64:
        case SPTF_SAMPLE_CINT32:
        case SPTF_SAMPLE_CFLOAT32:
            size = 8U;
            break;

        case SPTF_SAMPLE_CFLOAT64:
            size = 16U;
            break;

        default:
            size = 0U;
            break;
    }

    return size;
}

uint32_t sptf_bytes_per_instant(const sptf_stream_desc_t *desc)
{
    uint32_t bytes = 0U;

    if (desc != NULL)
    {
        if ((desc->channel_count >= 1U) && (desc->channel_count <= SPTF_MAX_CHANNELS))
        {
            bytes = sptf_sample_type_size(desc->sample_type) * (uint32_t)desc->channel_count;
        }
    }

    return bytes;
}

uint32_t sptf_descriptor_frame_size(uint16_t channel_count, bool crc_enabled)
{
    const uint32_t records = (uint32_t)channel_count * SPTF_CHANNEL_RECORD_SIZE;

    return (SPTF_HEADER_SIZE + SPTF_DESCRIPTOR_BODY_SIZE) + records + crc_size(crc_enabled);
}

uint32_t sptf_data_frame_size(uint32_t bytes_per_instant, uint32_t sample_count, bool crc_enabled)
{
    const uint32_t overhead = (SPTF_HEADER_SIZE + SPTF_DATA_BODY_SIZE) + crc_size(crc_enabled);
    uint32_t size = 0U;

    if ((bytes_per_instant > 0U) && (sample_count <= ((UINT32_MAX - overhead) / bytes_per_instant)))
    {
        size = overhead + (sample_count * bytes_per_instant);
    }

    return size;
}

uint32_t sptf_max_samples_per_frame(uint32_t bytes_per_instant, uint32_t frame_size_limit, bool crc_enabled)
{
    const uint32_t overhead = (SPTF_HEADER_SIZE + SPTF_DATA_BODY_SIZE) + crc_size(crc_enabled);
    uint32_t count = 0U;

    if ((bytes_per_instant > 0U) && (frame_size_limit > overhead))
    {
        count = (frame_size_limit - overhead) / bytes_per_instant;
    }

    return count;
}

/* ================================================================= doğrulama */

sptf_status_t sptf_check_stream_desc(const sptf_stream_desc_t *desc)
{
    sptf_status_t status = SPTF_STATUS_OK;

    if (desc == NULL)
    {
        status = SPTF_STATUS_NULL_POINTER;
    }
    else if ((desc->name == NULL) || (desc->channels == NULL))
    {
        status = SPTF_STATUS_NULL_POINTER;
    }
    else if (!((desc->sample_rate_hz >= SPTF_MIN_SAMPLE_RATE_HZ) && (desc->sample_rate_hz <= SPTF_MAX_SAMPLE_RATE_HZ)))
    {
        status = SPTF_STATUS_INVALID_RATE;
    }
    else if (text_length(desc->name, SPTF_NAME_FIELD_SIZE) > SPTF_NAME_FIELD_SIZE)
    {
        status = SPTF_STATUS_TEXT_TOO_LONG;
    }
    else
    {
        status = check_data_desc(desc);

        if (status == SPTF_STATUS_OK)
        {
            status = check_channels(desc->channels, desc->channel_count);
        }
    }

    return status;
}

/* =============================================================== kodlayıcılar */

sptf_status_t sptf_encode_descriptor(const sptf_frame_header_t *header, const sptf_stream_desc_t *desc,
                                     uint8_t *buffer, uint32_t buffer_size, uint32_t *frame_size)
{
    sptf_status_t status = SPTF_STATUS_OK;

    if ((header == NULL) || (buffer == NULL) || (frame_size == NULL))
    {
        status = SPTF_STATUS_NULL_POINTER;
    }
    else
    {
        status = sptf_check_stream_desc(desc);
    }

    if (status == SPTF_STATUS_OK)
    {
        const uint32_t size = sptf_descriptor_frame_size(desc->channel_count, header->crc_enabled);

        if (size > buffer_size)
        {
            status = SPTF_STATUS_BUFFER_TOO_SMALL;
        }
        else
        {
            uint32_t channel;

            put_header(buffer, header, SPTF_FRAME_DESCRIPTOR, size);
            put_f64(buffer, OFFSET_SAMPLE_RATE, desc->sample_rate_hz);
            put_u16(buffer, OFFSET_CHANNEL_COUNT, desc->channel_count);
            put_u8(buffer, OFFSET_SAMPLE_TYPE, desc->sample_type);
            put_u8(buffer, OFFSET_LAYOUT, desc->layout);
            put_u32(buffer, OFFSET_DESC_RESERVED, 0U);
            put_text(buffer, OFFSET_STREAM_NAME, SPTF_NAME_FIELD_SIZE, desc->name);

            for (channel = 0U; channel < (uint32_t)desc->channel_count; channel++)
            {
                const sptf_channel_t *info = &desc->channels[channel];
                const uint32_t record = OFFSET_CHANNEL_RECORDS + (channel * SPTF_CHANNEL_RECORD_SIZE);

                put_text(buffer, record + CHANNEL_FIELD_NAME, SPTF_NAME_FIELD_SIZE, info->name);
                put_text(buffer, record + CHANNEL_FIELD_UNIT, SPTF_UNIT_FIELD_SIZE, info->unit);
                put_f64(buffer, record + CHANNEL_FIELD_SCALE, info->scale);
                put_f64(buffer, record + CHANNEL_FIELD_OFFSET, info->offset);
            }

            if (header->crc_enabled)
            {
                put_crc(buffer, size);
            }

            *frame_size = size;
        }
    }

    return status;
}

sptf_status_t sptf_encode_data(const sptf_frame_header_t *header, const sptf_stream_desc_t *desc,
                               const sptf_data_slice_t *slice, uint8_t *buffer, uint32_t buffer_size,
                               uint32_t *frame_size)
{
    sptf_status_t status = SPTF_STATUS_OK;
    uint32_t bytes_per_instant = 0U;

    if ((header == NULL) || (desc == NULL) || (slice == NULL) || (buffer == NULL) || (frame_size == NULL))
    {
        status = SPTF_STATUS_NULL_POINTER;
    }
    else
    {
        status = check_data_desc(desc);
    }

    if (status == SPTF_STATUS_OK)
    {
        bytes_per_instant = sptf_bytes_per_instant(desc);
        status = check_slice(slice, bytes_per_instant);
    }

    if (status == SPTF_STATUS_OK)
    {
        const uint32_t size = sptf_data_frame_size(bytes_per_instant, slice->count, header->crc_enabled);

        if ((size == 0U) || (size > buffer_size))
        {
            status = SPTF_STATUS_BUFFER_TOO_SMALL;
        }
        else
        {
            put_header(buffer, header, SPTF_FRAME_DATA, size);
            put_u64(buffer, OFFSET_FIRST_SAMPLE, slice->first_sample);
            put_u32(buffer, OFFSET_SAMPLE_COUNT, slice->count);
            put_u32(buffer, OFFSET_DATA_RESERVED, 0U);
            put_samples(buffer, desc, slice, bytes_per_instant);

            if (header->crc_enabled)
            {
                put_crc(buffer, size);
            }

            *frame_size = size;
        }
    }

    return status;
}

sptf_status_t sptf_encode_marker(const sptf_frame_header_t *header, uint64_t sample_index, uint32_t code,
                                 const char *text, uint8_t *buffer, uint32_t buffer_size, uint32_t *frame_size)
{
    sptf_status_t status = SPTF_STATUS_OK;

    if ((header == NULL) || (text == NULL) || (buffer == NULL) || (frame_size == NULL))
    {
        status = SPTF_STATUS_NULL_POINTER;
    }
    else
    {
        const uint32_t overhead = (SPTF_HEADER_SIZE + SPTF_MARKER_BODY_SIZE) + crc_size(header->crc_enabled);

        if (buffer_size < overhead)
        {
            status = SPTF_STATUS_BUFFER_TOO_SMALL;
        }
        else
        {
            uint32_t room = buffer_size - overhead;
            uint32_t text_bytes;

            if (room > SPTF_MAX_MARKER_TEXT)
            {
                room = SPTF_MAX_MARKER_TEXT;
            }

            text_bytes = text_length(text, room);

            if (text_bytes > room)
            {
                status = SPTF_STATUS_TEXT_TOO_LONG;
            }
            else
            {
                const uint32_t size = overhead + text_bytes;
                uint32_t position;

                put_header(buffer, header, SPTF_FRAME_MARKER, size);
                put_u64(buffer, OFFSET_SAMPLE_INDEX, sample_index);
                put_u32(buffer, OFFSET_MARKER_CODE, code);
                put_u16(buffer, OFFSET_TEXT_LENGTH, (uint16_t)text_bytes);
                put_u16(buffer, OFFSET_MARKER_RESERVED, 0U);

                for (position = 0U; position < text_bytes; position++)
                {
                    buffer[OFFSET_TEXT + position] = (uint8_t)text[position];
                }

                if (header->crc_enabled)
                {
                    put_crc(buffer, size);
                }

                *frame_size = size;
            }
        }
    }

    return status;
}
