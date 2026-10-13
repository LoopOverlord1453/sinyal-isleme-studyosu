/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    sptf_app.h
 * @brief   Örnek: FreeRTOS (CMSIS-RTOS2) ile STM32, ADC + DMA örneklerini SPTF-UDP ile gönderir.
 *
 * CubeMX yapılandırması (STM32F429ZI, NUCLEO-F429ZI):
 *   - FREERTOS: CMSIS_V2.
 *   - ETH + LwIP, NO_SYS = 0. LWIP_TCPIP_CORE_LOCKING = 1 (lwIP 2.1 varsayılanı); kütüphane
 *     lwIP çağrılarını LOCK_TCPIP_CORE altında yapar.
 *   - ADC1: 12 bit, tarama 4 kanal, harici tetik TIM2 TRGO; DMA dairesel, "Half Word".
 *   - TIM2: güncelleme olayı 10 kHz, TRGO = Update.
 *
 * Kullanım (CubeMX'in varsayılan görevinde, MX_LWIP_Init() çağrısından sonra):
 *   (void)sptf_app_start(&hadc1);
 *   (void)HAL_TIM_Base_Start(&htim2);
 *
 * Copyright (C) 2026, Kubilay Közleme <burhankozleme@gmail.com>
 */

#ifndef SPTF_APP_H
#define SPTF_APP_H

#include <stdbool.h>

#include "main.h"

/**
 * @brief  Gönderim görevini oluşturur. Görev akışı açar, ADC'yi DMA ile başlatır ve
 *         dolan yarım tamponları gönderir.
 * @param  adc  CubeMX'in oluşturduğu ADC tanıtıcısı (ör. &hadc1).
 * @return true: görev oluşturuldu.
 */
bool sptf_app_start(ADC_HandleTypeDef *adc);

#endif /* SPTF_APP_H */
