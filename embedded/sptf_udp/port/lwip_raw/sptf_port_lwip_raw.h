/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    sptf_port_lwip_raw.h
 * @brief   lwIP raw API taşıma portu (sıfır kopya: çerçeve doğrudan pbuf'a yazılır).
 *
 * Çıplak metal (`NO_SYS = 1`) ve STM32CubeMX / Xilinx lwIP projeleri için önerilen porttur. lwIP 2.x ile uyumludur.
 *
 * Bağlam kuralı (lwIP raw API'nin kuralıdır):
 *  - `NO_SYS = 1`: tüm çağrılar lwIP'nin çalıştığı ana döngüden yapılmalıdır (kesmeden değil).
 *  - RTOS + `LWIP_TCPIP_CORE_LOCKING = 1`: port her çağrıda `LOCK_TCPIP_CORE()` alır; herhangi bir görevden çağrılabilir.
 *  - RTOS, çekirdek kilidi yok: çağrıları tcpip iş parçacığına taşıyın ya da `port/socket` portunu kullanın.
 *
 * Kilit davranışı `SPTF_LWIP_LOCK()` / `SPTF_LWIP_UNLOCK()` makrolarıyla değiştirilebilir.
 *
 * @ingroup sptf_port
 */
#ifndef SPTF_PORT_LWIP_RAW_H
#define SPTF_PORT_LWIP_RAW_H

#include "lwip/ip_addr.h"
#include "lwip/pbuf.h"
#include "lwip/udp.h"

#include "sptf.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup sptf_port Taşıma portları
 * @brief    ::sptf_transport_t gerçeklemeleri.
 * @{
 */

/** @brief lwIP raw portu bağlamı. Alanları doğrudan değiştirmeyin. */
typedef struct
{
    struct udp_pcb *pcb;      /**< UDP denetim bloğu. */
    ip_addr_t       dst_ip;   /**< Hedef (PC) adresi. */
    u16_t           dst_port; /**< Hedef port (varsayılan ::SPTF_UDP_PORT). */
    struct pbuf    *pending;  /**< `acquire` ile alınıp henüz gönderilmemiş pbuf. */
} sptf_lwip_raw_t;

/**
 * @brief UDP denetim bloğunu oluşturur ve hedefi ayarlar.
 * @param[out] port        Port bağlamı (statik ya da kalıcı bellekte).
 * @param[in]  dst_ip      Hedef PC adresi.
 * @param[in]  dst_port    Hedef port (genelde ::SPTF_UDP_PORT).
 * @param[in]  local_port  Yerel kaynak portu (0 → lwIP seçer).
 * @retval SPTF_OK         Başarılı.
 * @retval SPTF_ERR_ARG    NULL argüman.
 * @retval SPTF_ERR_NOMEM  `udp_new()` başarısız (MEMP_NUM_UDP_PCB artırın).
 * @retval SPTF_ERR_IO     `udp_bind()` başarısız (port kullanımda).
 */
sptf_status_t sptf_lwip_raw_open(sptf_lwip_raw_t *port, const ip_addr_t *dst_ip, u16_t dst_port, u16_t local_port);

/**
 * @brief Hedef adresi / portu değiştirir (ör. PC'nin IP'si DHCP ile değişti).
 * @param[in,out] port      Açık port.
 * @param[in]     dst_ip    Yeni hedef adres.
 * @param[in]     dst_port  Yeni hedef port.
 * @retval SPTF_OK, SPTF_ERR_ARG
 */
sptf_status_t sptf_lwip_raw_set_destination(sptf_lwip_raw_t *port, const ip_addr_t *dst_ip, u16_t dst_port);

/**
 * @brief Denetim bloğunu kapatır; bekleyen pbuf varsa serbest bırakır.
 * @param[in,out] port  Port.
 */
void sptf_lwip_raw_close(sptf_lwip_raw_t *port);

/**
 * @brief ::sptf_transport_t yapısını bu portla doldurur (::sptf_link_config_t::transport için).
 * @param[in]  port  Açık port.
 * @param[out] out   Doldurulacak taşıma yapısı.
 */
void sptf_lwip_raw_transport(sptf_lwip_raw_t *port, sptf_transport_t *out);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* SPTF_PORT_LWIP_RAW_H */
