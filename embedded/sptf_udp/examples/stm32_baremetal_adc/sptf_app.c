/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    sptf_app.c
 * @brief   Örnek: çıplak metal STM32, ADC + DMA çift tampon, SPTF-UDP gönderimi.
 *
 * DMA, 2 x 256 örnek anlık dairesel tampona yazar. Yarım ve tam dolma kesmelerinde
 * yalnız sayaçlar artırılır; dolan yarı, ana döngüde sptf_udp_send_samples() ile
 * gönderilir (SPT-PRT-002 [G-28]: ağ çağrıları kesme dışında). Ana döngü bir yarıyı
 * DMA onu yeniden yazmadan önce gönderemezse "overrun" işareti (MARKER) gönderilir.
 *
 * Akış: "ADC1", 4 kanal, uint16, interleaved (ADC tarama + DMA çıktısıyla aynı düzen),
 * 10 kHz, ölçek 3,3 V / 4096. Gerilim, arayüzde volt olarak görünür.
 *
 * Copyright (C) 2026, Kubilay Közleme <burhankozleme@gmail.com>
 */

#include "sptf_app.h"

#include <stddef.h>
#include <stdint.h>

#include "sptf_udp.h"

/* ----------------------------------------------------------------- ayarlar */

#define ADC_CHANNELS            (4U)            /* tarama dizisindeki kanal sayısı */
#define BLOCK_SAMPLES           (256U)          /* yarım tampondaki örnek anı */
#define SAMPLE_RATE_HZ          (10000.0)       /* TIM2 TRGO */
#define ADC_FULL_SCALE_V        (3.3)
#define ADC_LEVELS              (4096.0)        /* 12 bit */

/* 16 bitlik örnekler 32 bitlik kelimelere ikişer sığar (HAL_ADC_Start_DMA uint32_t ister). */
#define HALF_BUFFER_WORDS       ((BLOCK_SAMPLES * ADC_CHANNELS) / 2U)
#define BUFFER_WORDS            (2U * HALF_BUFFER_WORDS)
#define DMA_TRANSFERS           (2U * BLOCK_SAMPLES * ADC_CHANNELS)

#define STREAM_ID               (1U)
#define NANOSECONDS_PER_MS      (1000000U)

#define MARKER_START            (1U)
#define MARKER_OVERRUN          (2U)

/* ------------------------------------------------------------------- durum */

/* DMA hedefi. DMA yarım kelime yazar; gönderimde baytlara unsigned char ile erişilir. */
static uint32_t adc_buffer[BUFFER_WORDS];

/* Kesmede artırılır, ana döngüde okunur. Her sayaç tek yazara sahiptir. */
static volatile uint32_t half_events;
static volatile uint32_t full_events;

static ADC_HandleTypeDef *adc_handle;
static sptf_udp_stream_t *adc_stream;

/* ------------------------------------------------------------------ yardımcılar */

static uint64_t now_ns(void)
{
    /* HAL tik sayacı 1 ms çözünürlüklüdür; yalnız DESCRIPTOR zamanlaması için yeterlidir. */
    return (uint64_t)HAL_GetTick() * NANOSECONDS_PER_MS;
}

static void send_half(uint32_t first_word, uint32_t missed_events)
{
    const uint8_t *samples = (const uint8_t *)&adc_buffer[first_word];
    sptf_udp_stats_t stats;

    if (missed_events > 0U)
    {
        /* Yarı, gönderilmeden önce DMA tarafından yeniden yazıldı: olayı sinyal üzerinde işaretle. */
        if (sptf_udp_get_stats(adc_stream, &stats) == SPTF_STATUS_OK)
        {
            (void)sptf_udp_send_marker(adc_stream, stats.next_first_sample, MARKER_OVERRUN, "overrun", now_ns());
        }
    }

    /* Hatalar akışın sayaçlarına yazılır; kayıp örnekler arayüzde boşluk olarak görünür. */
    (void)sptf_udp_send_samples(adc_stream, samples, BLOCK_SAMPLES, now_ns());
}

/* ================================================================== arayüz */

bool sptf_app_init(ADC_HandleTypeDef *adc)
{
    /* Kütüphane tablonun adresini saklar: statik ömürlü olmalıdır. */
    static const sptf_channel_t adc_channels[ADC_CHANNELS] =
    {
        { "AIN0 (PA3)", "V", ADC_FULL_SCALE_V / ADC_LEVELS, 0.0 },
        { "AIN1 (PC0)", "V", ADC_FULL_SCALE_V / ADC_LEVELS, 0.0 },
        { "AIN2 (PC3)", "V", ADC_FULL_SCALE_V / ADC_LEVELS, 0.0 },
        { "AIN3 (PB1)", "V", ADC_FULL_SCALE_V / ADC_LEVELS, 0.0 }
    };
    sptf_udp_config_t config;
    bool started = false;

    if ((adc != NULL) && (sptf_udp_config_init(&config) == SPTF_STATUS_OK))
    {
        config.desc.name = "ADC1";
        config.desc.channels = adc_channels;
        config.desc.sample_rate_hz = SAMPLE_RATE_HZ;
        config.desc.channel_count = ADC_CHANNELS;
        config.desc.sample_type = SPTF_SAMPLE_UINT16;
        config.desc.layout = SPTF_LAYOUT_INTERLEAVED;

        /* Signal Processing Toolbox'ın çalıştığı PC */
        config.remote_ip[0] = 192U;
        config.remote_ip[1] = 168U;
        config.remote_ip[2] = 1U;
        config.remote_ip[3] = 10U;
        config.stream_id = STREAM_ID;
        config.crc_enabled = true;

        if (sptf_udp_open(&config, &adc_stream) == SPTF_STATUS_OK)
        {
            adc_handle = adc;

            if (HAL_ADC_Start_DMA(adc, adc_buffer, DMA_TRANSFERS) == HAL_OK)
            {
                (void)sptf_udp_send_marker(adc_stream, 0U, MARKER_START, "acquisition started", now_ns());
                started = true;
            }
        }
    }

    return started;
}

void sptf_app_process(void)
{
    static uint32_t half_handled = 0U;
    static uint32_t full_handled = 0U;
    const uint32_t half_now = half_events;
    const uint32_t full_now = full_events;

    if (adc_stream != NULL)
    {
        /* Yarım dolma her zaman tam dolmadan önce gelir: önce ilk yarı gönderilir. */
        if (half_now != half_handled)
        {
            send_half(0U, (half_now - half_handled) - 1U);
            half_handled = half_now;
        }

        if (full_now != full_handled)
        {
            send_half(HALF_BUFFER_WORDS, (full_now - full_handled) - 1U);
            full_handled = full_now;
        }
    }
}

/* ============================================================ HAL geri çağrıları */

void HAL_ADC_ConvHalfCpltCallback(ADC_HandleTypeDef *hadc)
{
    if (hadc == adc_handle)
    {
        half_events++;
    }
}

void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef *hadc)
{
    if (hadc == adc_handle)
    {
        full_events++;
    }
}
