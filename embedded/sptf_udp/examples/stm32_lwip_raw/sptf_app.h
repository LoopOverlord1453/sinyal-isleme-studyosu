/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    sptf_app.h
 * @brief   STM32 + lwIP (çıplak metal, `NO_SYS = 1`) örnek uygulaması: ADC DMA çift tamponu → SPTF-UDP.
 *
 * STM32CubeMX projesine eklenir. Gerekenler:
 *  - LWIP etkin (`NO_SYS = 1`), `MX_LWIP_Init()` / `MX_LWIP_Process()` çağrılıyor.
 *  - ADC1: 2 kanal tarama, harici tetik (ör. TIM3 TRGO @ 20 kHz), DMA dairesel, yarım kelime (uint16).
 *
 * Kullanım (main.c):
 * @code
 * MX_LWIP_Init();
 * sptf_app_init();
 * while (1) {
 *     MX_LWIP_Process();
 *     sptf_app_process();
 * }
 * @endcode
 *
 * @ingroup sptf_examples
 */
#ifndef SPTF_APP_H
#define SPTF_APP_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup sptf_examples Örnekler
 * @brief    Kullanıma hazır entegrasyon örnekleri.
 * @{
 */

/**
 * @brief Bağlantıyı, akışı ve ADC DMA'sını başlatır. `MX_LWIP_Init()`'ten sonra bir kez çağrılır.
 * @return 0 başarılı, negatif ::sptf_status_t hata kodu.
 */
int sptf_app_init(void);

/**
 * @brief Ana döngüden sürekli çağrılır: hazır yarım tamponu gönderir, ~100 ms'de bir DESCRIPTOR zamanlamasını yürütür.
 */
void sptf_app_process(void);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* SPTF_APP_H */
