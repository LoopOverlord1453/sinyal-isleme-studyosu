/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    sptf_app.c
 * @brief   STM32 + lwIP raw örneği: 2 kanal ADC (DMA çift tampon) → "Motor akımları" akışı.
 *
 * Akış: DMA yarım / tam kesmesi yalnız bayrak kurar (ISR'de ağ çağrısı yok). Ana döngü hazır yarıyı
 * (gerekirse D-önbelleği geçersiz kılıp) ::sptf_stream_write ile gönderir. DMA bir yarıyı, ana döngü onu
 * göndermeden yeniden doldurursa taşma sayılır ve arayüzde görünen bir işaret (MARKER) gönderilir.
 *
 * Bellek: ADC tamponu DMA'nın erişebildiği RAM'de olmalıdır (STM32H7: D2 SRAM, `.RAM_D2` vb.;
 * bağlayıcı betiğinize göre ::SPTF_APP_DMA_SECTION). lwIP yığını (`LWIP_RAM_HEAP_POINTER`) Ethernet DMA'sı için
 * önbelleksiz bölgede olmalıdır (CubeMX MPU ayarı).
 *
 * @ingroup sptf_examples
 */
#include "sptf_app.h"

#include "main.h"

#include "lwip/ip_addr.h"

#include "sptf.h"
#include "sptf_port_lwip_raw.h"

/* -------------------------------------------------------------------------------------- ayarlar (#ifndef) */
#ifndef SPTF_APP_PC_IP
/** @brief Arayüzün çalıştığı PC'nin IPv4 adresi (4 bayt, virgülle). */
#define SPTF_APP_PC_IP 192, 168, 1, 10
#endif
#ifndef SPTF_APP_SAMPLE_RATE_HZ
/** @brief ADC tetik frekansı (kanal başına örnekleme hızı). */
#define SPTF_APP_SAMPLE_RATE_HZ 20000.0
#endif
#ifndef SPTF_APP_BLOCK
/** @brief Yarım tampondaki örnek anı sayısı. 2 kanal × uint16 için 356'nın altı tek datagrama sığar. */
#define SPTF_APP_BLOCK 256u
#endif
#ifndef SPTF_APP_DMA_SECTION
/** @brief ADC tamponunun bölüm özniteliği (ör. `__attribute__((section(".RAM_D2")))`). */
#define SPTF_APP_DMA_SECTION
#endif

/** @brief Virgüllü IP makrosunu `IP_ADDR4`'e dört ayrı argüman olarak geçirir (C99 değişken argümanlı makro). */
#define APP_IP_ADDR4(ip, ...) IP_ADDR4(ip, __VA_ARGS__)

#define APP_CHANNELS 2u                                     /**< ADC tarama kanalı. */
#define APP_HALF     (SPTF_APP_BLOCK * APP_CHANNELS)        /**< Yarım tampondaki uint16 eleman. */

extern ADC_HandleTypeDef hadc1;                             /**< CubeMX'in ürettiği ADC tanıtıcısı. */

/** @brief DMA çift tamponu (32 bayt hizalı: D-önbellek satırı). */
SPTF_APP_DMA_SECTION static uint16_t s_adc[2u * APP_HALF] __attribute__((aligned(32)));

static volatile uint8_t  s_ready;      /**< bit0: ilk yarı hazır, bit1: ikinci yarı hazır. */
static volatile uint32_t s_overruns;   /**< Gönderilmeden üzerine yazılan yarım tampon sayısı. */
static uint32_t          s_reported;   /**< MARKER ile bildirilen taşma sayısı. */
static uint32_t          s_last_poll;  /**< Son DESCRIPTOR zamanlaması (ms). */

static sptf_lwip_raw_t s_port;
static sptf_link_t     s_link;
static sptf_stream_t   s_stream;

/** @brief Kanal tanımları: 12 bit ADC, Vref 3.3 V, 0.1 Ω şönt + ×20 kazanç → A. */
static const sptf_channel_t s_channels[APP_CHANNELS] = {
    {"Ia", "A", 3.3 / 4096.0 / (0.1 * 20.0), -0.825}, /* ofset: 1.65 V orta nokta */
    {"Ib", "A", 3.3 / 4096.0 / (0.1 * 20.0), -0.825},
};

/** @brief Akış yapılandırması. */
static const sptf_stream_config_t s_cfg = {
    1u, "Motor akımları", SPTF_APP_SAMPLE_RATE_HZ, SPTF_UINT16, SPTF_LAYOUT_INTERLEAVED, APP_CHANNELS, s_channels, false,
};

/**
 * @brief DWT çevrim sayacından 64 bit monoton zaman (ns).
 *
 * 32 bit sayaç 480 MHz'de ~8.9 s'de taşar; fonksiyon bu süreden sık çağrıldığı sürece (her gönderimde) doğrudur.
 */
static uint64_t board_time_ns(void *ctx)
{
    static uint32_t last = 0u;
    static uint64_t high = 0u;
    const uint32_t now = DWT->CYCCNT;
    const uint64_t hz = (uint64_t)SystemCoreClock;
    uint64_t cycles;
    (void)ctx;
    if (now < last)
    {
        high += (uint64_t)1u << 32u;
    }
    last = now;
    cycles = high | (uint64_t)now;
    return ((cycles / hz) * 1000000000u) + (((cycles % hz) * 1000000000u) / hz);
}

/** @brief DWT çevrim sayacını açar. */
static void dwt_enable(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
#if defined(__CORE_CM7_H_GENERIC)
    DWT->LAR = 0xC5ACCE55u; /* Cortex-M7 (STM32F7 / H7): DWT yazma kilidini aç */
#endif
    DWT->CYCCNT = 0u;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

/**
 * @brief HAL: DMA ilk yarıyı doldurdu (kesme bağlamı: yalnız bayrak).
 * @param[in] hadc  Kesmeyi üreten ADC tanıtıcısı.
 */
void HAL_ADC_ConvHalfCpltCallback(ADC_HandleTypeDef *hadc)
{
    if (hadc == &hadc1)
    {
        if ((s_ready & 1u) != 0u)
        {
            s_overruns++;
        }
        s_ready |= 1u;
    }
}

/**
 * @brief HAL: DMA ikinci yarıyı doldurdu (kesme bağlamı: yalnız bayrak).
 * @param[in] hadc  Kesmeyi üreten ADC tanıtıcısı.
 */
void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef *hadc)
{
    if (hadc == &hadc1)
    {
        if ((s_ready & 2u) != 0u)
        {
            s_overruns++;
        }
        s_ready |= 2u;
    }
}

int sptf_app_init(void)
{
    sptf_link_config_t lc = {0};
    ip_addr_t pc;
    sptf_status_t st;
    dwt_enable();
    APP_IP_ADDR4(&pc, SPTF_APP_PC_IP);
    st = sptf_lwip_raw_open(&s_port, &pc, (u16_t)SPTF_UDP_PORT, 0u);
    if (st == SPTF_OK)
    {
        sptf_lwip_raw_transport(&s_port, &lc.transport);
        lc.clock = board_time_ns;
        st = sptf_link_init(&s_link, &lc);
    }
    if (st == SPTF_OK)
    {
        st = sptf_stream_init(&s_stream, &s_link, &s_cfg);
    }
    if ((st == SPTF_OK) && (HAL_ADC_Start_DMA(&hadc1, (uint32_t *)s_adc, 2u * APP_HALF) != HAL_OK))
    {
        st = SPTF_ERR_IO;
    }
    return (int)st;
}

/** @brief Bir yarım tamponu gönderir. */
static void send_half(uint32_t half)
{
    uint16_t *p = &s_adc[half * APP_HALF];
#if defined(__DCACHE_PRESENT) && (__DCACHE_PRESENT == 1U)
    SCB_InvalidateDCache_by_Addr(p, (int32_t)(APP_HALF * sizeof(uint16_t)));
#endif
    (void)sptf_stream_write(&s_stream, p, SPTF_APP_BLOCK, SPTF_TS_AUTO);
}

void sptf_app_process(void)
{
    uint8_t ready;
    uint32_t now_ms;
    const uint32_t primask = __get_PRIMASK();
    __disable_irq();
    ready = s_ready;
    s_ready = 0u;
    __set_PRIMASK(primask);
    if ((ready & 1u) != 0u)
    {
        send_half(0u);
    }
    if ((ready & 2u) != 0u)
    {
        send_half(1u);
    }
    if (s_overruns != s_reported)
    {
        s_reported = s_overruns;
        (void)sptf_stream_marker(&s_stream, SPTF_MARKER_AT_NEXT, 0xDEADu, "ADC taşması (ana döngü yavaş)", SPTF_TS_AUTO);
    }
    now_ms = HAL_GetTick();
    if ((now_ms - s_last_poll) >= 100u)
    {
        s_last_poll = now_ms;
        (void)sptf_link_poll(&s_link);
    }
}
