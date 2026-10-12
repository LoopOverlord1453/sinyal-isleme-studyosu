/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    sptf_port_socket.h
 * @brief   BSD soket taşıma portu: lwIP socket API (FreeRTOS vb.), POSIX (Linux, Zephyr, embedded Linux) ve
 *          Windows (Winsock 2).
 *
 * Seçim derleme zamanında yapılır:
 *  - `SPTF_SOCKET_LWIP` tanımlıysa `lwip/sockets.h` (`LWIP_SOCKET = 1`, `LWIP_COMPAT_SOCKETS` önerilir),
 *  - değilse `_WIN32` tanımlıysa Winsock 2 (`ws2_32` ile bağlayın),
 *  - aksi hâlde POSIX.
 *
 * Datagram tamponu port nesnesinin içindedir (bir kopya: tampon → yığın). Soket varsayılan olarak bloklayan
 * kipte açılır.
 *
 * @ingroup sptf_port
 */
#ifndef SPTF_PORT_SOCKET_H
#define SPTF_PORT_SOCKET_H

#include "sptf.h"

#if defined(SPTF_SOCKET_LWIP)
#include "lwip/sockets.h"
/** @brief Platformun soket tanıtıcısı. */
typedef int sptf_socket_fd_t;
#elif defined(_WIN32)
#include <winsock2.h>
/** @brief Platformun soket tanıtıcısı. */
typedef SOCKET sptf_socket_fd_t;
#else
#include <netinet/in.h>
/** @brief Platformun soket tanıtıcısı. */
typedef int sptf_socket_fd_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @addtogroup sptf_port
 * @{
 */

/** @brief Port içindeki datagram tamponu (bayt); ::sptf_link_config_t::max_datagram'dan küçük olamaz. */
#ifndef SPTF_SOCKET_BUFFER_SIZE
#define SPTF_SOCKET_BUFFER_SIZE SPTF_CFG_MAX_DATAGRAM
#endif

/** @brief Soket portu bağlamı. Alanları doğrudan değiştirmeyin. */
typedef struct
{
    sptf_socket_fd_t   fd;     /**< UDP soketi. */
    struct sockaddr_in dst;    /**< Hedef (PC) adresi. */
    bool               open;   /**< Soket açık mı. */
    bool               busy;   /**< Tampon `acquire` ile verildi, `send` bekleniyor. */
    uint8_t            buf[SPTF_SOCKET_BUFFER_SIZE]; /**< Datagram tamponu (kodlayıcı bayt bayt yazar: hizalama gerekmez). */
} sptf_socket_t;

/**
 * @brief UDP soketini açar ve hedefi ayarlar. Windows'ta Winsock'u da başlatır.
 * @param[out] port      Port bağlamı.
 * @param[in]  dst_ipv4  Hedef PC adresi, 4 bayt, ağ sırasıyla (ör. `{192, 168, 1, 10}`).
 * @param[in]  dst_port  Hedef port (genelde ::SPTF_UDP_PORT).
 * @retval SPTF_OK       Başarılı.
 * @retval SPTF_ERR_ARG  NULL argüman.
 * @retval SPTF_ERR_IO   Soket açılamadı.
 */
sptf_status_t sptf_socket_open(sptf_socket_t *port, const uint8_t dst_ipv4[4], uint16_t dst_port);

/**
 * @brief Hedef adresi / portu değiştirir.
 * @param[in,out] port      Açık port.
 * @param[in]     dst_ipv4  Yeni hedef adres (4 bayt).
 * @param[in]     dst_port  Yeni hedef port.
 * @retval SPTF_OK, SPTF_ERR_ARG
 */
sptf_status_t sptf_socket_set_destination(sptf_socket_t *port, const uint8_t dst_ipv4[4], uint16_t dst_port);

/**
 * @brief Soketi kapatır (Windows'ta Winsock'u da bırakır).
 * @param[in,out] port  Port.
 */
void sptf_socket_close(sptf_socket_t *port);

/**
 * @brief ::sptf_transport_t yapısını bu portla doldurur.
 * @param[in]  port  Açık port.
 * @param[out] out   Doldurulacak taşıma yapısı.
 */
void sptf_socket_transport(sptf_socket_t *port, sptf_transport_t *out);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* SPTF_PORT_SOCKET_H */
