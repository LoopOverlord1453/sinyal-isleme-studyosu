/*
 * Signal Processing Toolbox — DAC dalga tablosu "wave"
 * Dalga: Çoklu ton, 1 kHz + 3 kHz + 6 kHz, genlik 1.80984 V, ofset 1.6496 V
 * Hedef: STM32 HAL — DAC + TIM6 TRGO + DMA dairesel
 * Tablo: N = 250 örnek, M = 1 periyot; fs = 250 kHz → f = M·fs/N = 1000 Hz (istenen 1000 Hz, hata +0.000 ppm)
 * DAC: 12 bit, İşaretsiz (0 … 2^b−1), sağa hizalı, uint16_t; 0 … 3.3 V, 1 LSB = 0.805664 mV
 * Bellek: tablo 500 bayt (const → flash)
 * Bu dosya otomatik üretilmiştir; elle değiştirmeyin, aracı yeniden çalıştırın.
 */
#ifndef WAVE_H
#define WAVE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WAVE_N                   250u   /* kanal başına örnek (DDS: 2^k seviye) */
#define WAVE_CHANNELS            1u
#define WAVE_TABLE_LEN           250u   /* dizi eleman sayısı */
#define WAVE_PERIODS             1u   /* tablodaki periyot sayısı (M) */
#define WAVE_FS_HZ               250000.0   /* DAC güncelleme hızı */
#define WAVE_F_HZ                1000.0   /* gerçekleşen frekans */
#define WAVE_F_TARGET_HZ         1000.0
#define WAVE_DAC_BITS            12u
#define WAVE_DAC_SHIFT           0u   /* sola hizalama kaydırması */
#define WAVE_DAC_LEFT_ALIGNED    0
#define WAVE_DAC_TWOS_COMPLEMENT 0
#define WAVE_VOUT_MIN_V          0.0
#define WAVE_VOUT_MAX_V          3.3
#define WAVE_TABLE_BYTES         500u

typedef uint16_t wave_word_t;   /* DAC'a yazılan kelime */

/* STM32 tarzı zamanlayıcı: fs = TIM_CLK / ((PSC + 1) * (ARR + 1)); TIM_CLK = 90 MHz
 * (CubeMX 'APB1 Timer clocks' ile aynı olmalı). Gerçekleşen fs = 250000 Hz (+0.000 ppm). */
#define WAVE_TIM_CLOCK_HZ        90000000.0
#define WAVE_TIM_PSC             0u
#define WAVE_TIM_ARR             359u
#define WAVE_TIM_ARR_MAX         65535u
#define WAVE_TIM_FS_HZ           250000.0

extern const wave_word_t *const wave_table;   /* WAVE_TABLE_LEN eleman */

uint32_t wave_arr_for_hz(double hz);   /* çıkış frekansı hz için ARR (PSC sabit) */

/* HAL sabitleri (yalnızca USE_HAL_DRIVER tanımlıyken kullanılır) */
#define WAVE_HAL_DAC_ALIGN       DAC_ALIGN_12B_R
#define WAVE_HAL_DAC_CHANNEL     DAC_CHANNEL_1
#define WAVE_HAL_DUAL_REG        DHR12RD   /* stereo (ikili) mod veri yazmacı */

#if defined(USE_HAL_DRIVER)
#ifndef WAVE_HAL_HEADER
#define WAVE_HAL_HEADER "main.h"   /* CubeMX projesi; gerekirse ör. "stm32f4xx_hal.h" */
#endif
#include WAVE_HAL_HEADER
HAL_StatusTypeDef wave_start(DAC_HandleTypeDef *hdac, TIM_HandleTypeDef *htim);
HAL_StatusTypeDef wave_stop(DAC_HandleTypeDef *hdac, TIM_HandleTypeDef *htim);
void wave_set_arr(TIM_HandleTypeDef *htim, uint32_t arr);   /* çalışırken frekans değiştir */
#endif

#ifdef __cplusplus
}
#endif

#endif /* WAVE_H */
