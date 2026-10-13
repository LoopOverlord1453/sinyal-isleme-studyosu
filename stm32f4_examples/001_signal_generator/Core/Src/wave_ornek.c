/* Signal Processing Toolbox — "wave" kullanım örneği (STM32 HAL — DAC + TIM6 TRGO + DMA dairesel) */
/* CubeMX projesinde main.c içine (USER CODE bölümlerine) ekleyin. */
#include "main.h"
#include "wave.h"

extern DAC_HandleTypeDef hdac;    /* F4/F7: hdac — G4/L4/H7: hdac1 (CubeMX adına göre düzeltin) */
extern TIM_HandleTypeDef htim6;

void wave_example_start(void)
{
    /* USER CODE BEGIN 2 içinde, MX_DAC_Init() ve MX_TIM6_Init() sonrasında çağırın */
    if (wave_start(&hdac, &htim6) != HAL_OK) {
        Error_Handler();
    }
}

/* Çıkış frekansını çalışırken değiştirme (tablo aynı, DAC güncelleme hızı değişir): */
void wave_example_set_freq(double hz)
{
    wave_set_arr(&htim6, wave_arr_for_hz(hz));
}
