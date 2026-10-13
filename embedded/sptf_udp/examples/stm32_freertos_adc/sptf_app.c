/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    sptf_app.c
 * @brief   Örnek: FreeRTOS (CMSIS-RTOS2) görevi, ADC + DMA çift tampon, SPTF-UDP gönderimi.
 *
 * DMA kesmeleri yalnız olay sayaçlarını artırır ve görevi bir thread flag ile uyandırır.
 * Görev dolan yarıları sptf_udp_send_samples() ile gönderir. Kütüphane lwIP çağrılarını
 * çekirdek kilidi (LOCK_TCPIP_CORE) altında yapar; görev tcpip iş parçacığından bağımsız
 * çalışır. Görev bir yarıyı DMA onu yeniden yazmadan önce gönderemezse "overrun" işareti
 * (MARKER) gönderilir.
 *
 * STM32F7 / STM32H7: D-Cache açıksa DMA tamponu önbelleksiz bir bölgeye (MPU) konmalı ya
 * da her yarı okunmadan önce SCB_InvalidateDCache_by_Addr() ile geçersiz kılınmalıdır.
 *
 * Copyright (C) 2026, Kubilay Közleme <burhankozleme@gmail.com>
 */

#include "sptf_app.h"

#include <stddef.h>
#include <stdint.h>

#include "cmsis_os2.h"
#include "sptf_udp.h"

/* ----------------------------------------------------------------- ayarlar */

#define ADC_CHANNELS            (4U)
#define BLOCK_SAMPLES           (256U)
#define SAMPLE_RATE_HZ          (10000.0)
#define ADC_FULL_SCALE_V        (3.3)
#define ADC_LEVELS              (4096.0)

#define HALF_BUFFER_WORDS       ((BLOCK_SAMPLES * ADC_CHANNELS) / 2U)
#define BUFFER_WORDS            (2U * HALF_BUFFER_WORDS)
#define DMA_TRANSFERS           (2U * BLOCK_SAMPLES * ADC_CHANNELS)

#define STREAM_ID               (1U)
#define TASK_STACK_BYTES        (2048U)
#define FLAG_DATA_READY         (0x00000001U)
#define NANOSECONDS_PER_TICK    (1000000U)      /* configTICK_RATE_HZ = 1000 */

#define MARKER_START            (1U)
#define MARKER_OVERRUN          (2U)

/* ------------------------------------------------------------------- durum */

static uint32_t adc_buffer[BUFFER_WORDS];

static volatile uint32_t half_events;
static volatile uint32_t full_events;

static ADC_HandleTypeDef *adc_handle;
static osThreadId_t task_handle;
static sptf_udp_stream_t *adc_stream;

/* ------------------------------------------------------------------ yardımcılar */

static uint64_t now_ns(void)
{
    return (uint64_t)osKernelGetTickCount() * NANOSECONDS_PER_TICK;
}

static bool open_stream(void)
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
    bool opened = false;

    if (sptf_udp_config_init(&config) == SPTF_STATUS_OK)
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

        opened = (sptf_udp_open(&config, &adc_stream) == SPTF_STATUS_OK);
    }

    return opened;
}

static void send_half(uint32_t first_word, uint32_t missed_events)
{
    const uint8_t *samples = (const uint8_t *)&adc_buffer[first_word];
    sptf_udp_stats_t stats;

    if ((missed_events > 0U) && (sptf_udp_get_stats(adc_stream, &stats) == SPTF_STATUS_OK))
    {
        (void)sptf_udp_send_marker(adc_stream, stats.next_first_sample, MARKER_OVERRUN, "overrun", now_ns());
    }

    (void)sptf_udp_send_samples(adc_stream, samples, BLOCK_SAMPLES, now_ns());
}

static void acquisition_task(void *argument)
{
    uint32_t half_handled = 0U;
    uint32_t full_handled = 0U;
    bool running = open_stream();

    (void)argument;

    if (running)
    {
        running = (HAL_ADC_Start_DMA(adc_handle, adc_buffer, DMA_TRANSFERS) == HAL_OK);
    }

    if (running)
    {
        (void)sptf_udp_send_marker(adc_stream, 0U, MARKER_START, "acquisition started", now_ns());
    }

    while (running)
    {
        const uint32_t flags = osThreadFlagsWait(FLAG_DATA_READY, osFlagsWaitAny, osWaitForever);

        if ((flags & osFlagsError) == 0U)
        {
            const uint32_t half_now = half_events;
            const uint32_t full_now = full_events;

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

    osThreadExit();
}

/* ================================================================== arayüz */

bool sptf_app_start(ADC_HandleTypeDef *adc)
{
    static const osThreadAttr_t task_attributes =
    {
        .name = "sptf",
        .stack_size = TASK_STACK_BYTES,
        .priority = osPriorityAboveNormal
    };
    bool created = false;

    if ((adc != NULL) && (task_handle == NULL))
    {
        adc_handle = adc;
        task_handle = osThreadNew(acquisition_task, NULL, &task_attributes);
        created = (task_handle != NULL);
    }

    return created;
}

/* ============================================================ HAL geri çağrıları */

void HAL_ADC_ConvHalfCpltCallback(ADC_HandleTypeDef *hadc)
{
    if (hadc == adc_handle)
    {
        half_events++;
        (void)osThreadFlagsSet(task_handle, FLAG_DATA_READY);
    }
}

void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef *hadc)
{
    if (hadc == adc_handle)
    {
        full_events++;
        (void)osThreadFlagsSet(task_handle, FLAG_DATA_READY);
    }
}
