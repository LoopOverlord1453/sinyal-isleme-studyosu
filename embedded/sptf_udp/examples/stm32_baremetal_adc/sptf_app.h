/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    sptf_app.h
 * @brief   Örnek: çıplak metal (NO_SYS = 1) STM32, ADC + DMA örneklerini SPTF-UDP ile gönderir.
 *
 * CubeMX yapılandırması (STM32F429ZI, NUCLEO-F429ZI):
 *   - ETH + LwIP, NO_SYS = 1 (RTOS yok), statik IP.
 *   - ADC1: 12 bit, tarama 4 kanal, sürekli dönüşüm kapalı, harici tetik TIM2 TRGO.
 *   - ADC1 DMA: dairesel, çevre birimi ve bellek veri genişliği "Half Word", bellek adres artışı açık.
 *   - TIM2: güncelleme olayı 10 kHz, TRGO = Update.
 *
 * Ana döngü:
 *   (void)sptf_app_init(&hadc1);
 *   (void)HAL_TIM_Base_Start(&htim2);
 *   for (;;)
 *   {
 *       MX_LWIP_Process();
 *       sptf_app_process();
 *   }
 *
 * Copyright (C) 2026, Kubilay Közleme <burhankozleme@gmail.com>
 */

#ifndef SPTF_APP_H
#define SPTF_APP_H

#include <stdbool.h>

#include "main.h"

/**
 * @brief  Akışı açar ve ADC'yi DMA ile başlatır.
 * @param  adc  CubeMX'in oluşturduğu ADC tanıtıcısı (ör. &hadc1).
 * @return true: akış açıldı ve DMA başladı.
 */
bool sptf_app_init(ADC_HandleTypeDef *adc);

/**
 * @brief  Dolan yarım tamponları gönderir. Ana döngüden, MX_LWIP_Process() ile birlikte
 *         çağrılır; kesme içinden çağrılmamalıdır.
 */
void sptf_app_process(void);

#endif /* SPTF_APP_H */
