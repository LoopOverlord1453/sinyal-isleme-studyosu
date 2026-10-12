/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    sptf_port_lwip_raw.c
 * @brief   lwIP raw API taşıma portu.
 *
 * `acquire` tek parça bir `PBUF_RAM` pbuf ayırır (yükü bitişiktir); kütüphane çerçeveyi doğrudan
 * `pbuf->payload`'a yazar; `send` `udp_sendto()` ile gönderip pbuf'ı bırakır. Ara kopya yoktur.
 *
 * @ingroup sptf_port
 */
#include "sptf_port_lwip_raw.h"

#include "lwip/err.h"

#if defined(NO_SYS) && defined(LWIP_TCPIP_CORE_LOCKING)
#if (NO_SYS == 0) && (LWIP_TCPIP_CORE_LOCKING != 0)
/** @brief RTOS + çekirdek kilidi: raw API çağrıları `LOCK_TCPIP_CORE()` altında yapılır. */
#define SPTF_PRIV_CORE_LOCK 1
#endif
#endif

#if !defined(SPTF_LWIP_LOCK)
#if defined(SPTF_PRIV_CORE_LOCK)
#include "lwip/tcpip.h"
/** @brief lwIP çekirdek kilidini alır (RTOS + çekirdek kilidi açıkken). */
#define SPTF_LWIP_LOCK()   LOCK_TCPIP_CORE()
/** @brief lwIP çekirdek kilidini bırakır. */
#define SPTF_LWIP_UNLOCK() UNLOCK_TCPIP_CORE()
#else
#define SPTF_LWIP_LOCK()   ((void)0)
#define SPTF_LWIP_UNLOCK() ((void)0)
#endif
#endif

/** @brief lwIP hata kodunu kütüphane durum koduna çevirir. */
static sptf_status_t map_err(err_t err)
{
    sptf_status_t st;
    switch (err)
    {
    case ERR_OK:
        st = SPTF_OK;
        break;
    case ERR_MEM:
    case ERR_BUF:
        st = SPTF_ERR_NOMEM;
        break;
    case ERR_WOULDBLOCK:
    case ERR_INPROGRESS:
        st = SPTF_ERR_BUSY;
        break;
    default:
        st = SPTF_ERR_IO;
        break;
    }
    return st;
}

/** @brief ::sptf_transport_t::acquire — tek parça PBUF_RAM ayırır. */
static uint8_t *lwip_acquire(void *ctx, size_t len)
{
    sptf_lwip_raw_t *port = (sptf_lwip_raw_t *)ctx;
    struct pbuf *pb = NULL;
    if ((port == NULL) || (len == 0u) || (len > 0xFFFFu))
    {
        return NULL;
    }
    SPTF_LWIP_LOCK();
    if (port->pending != NULL)
    {
        /* Sözleşme ihlali (acquire/send eşleşmedi): sızıntıyı önle. */
        (void)pbuf_free(port->pending);
        port->pending = NULL;
    }
    pb = pbuf_alloc(PBUF_TRANSPORT, (u16_t)len, PBUF_RAM);
    port->pending = pb;
    SPTF_LWIP_UNLOCK();
    return (pb != NULL) ? (uint8_t *)pb->payload : NULL;
}

/** @brief ::sptf_transport_t::send — `udp_sendto()` ve pbuf'ı bırakma. */
static sptf_status_t lwip_send(void *ctx, uint8_t *buf, size_t len)
{
    sptf_lwip_raw_t *port = (sptf_lwip_raw_t *)ctx;
    sptf_status_t st = SPTF_OK;
    struct pbuf *pb;
    if (port == NULL)
    {
        return SPTF_ERR_ARG;
    }
    SPTF_LWIP_LOCK();
    pb = port->pending;
    port->pending = NULL;
    if ((pb == NULL) || ((uint8_t *)pb->payload != buf) || (len > (size_t)pb->tot_len))
    {
        st = SPTF_ERR_ARG;
    }
    else if (len > 0u)
    {
        pbuf_realloc(pb, (u16_t)len);
        st = map_err(udp_sendto(port->pcb, pb, &port->dst_ip, port->dst_port));
    }
    else
    {
        /* len == 0: yalnız serbest bırak */
    }
    if (pb != NULL)
    {
        (void)pbuf_free(pb);
    }
    SPTF_LWIP_UNLOCK();
    return st;
}

sptf_status_t sptf_lwip_raw_open(sptf_lwip_raw_t *port, const ip_addr_t *dst_ip, u16_t dst_port, u16_t local_port)
{
    sptf_status_t st = SPTF_OK;
    if ((port == NULL) || (dst_ip == NULL))
    {
        return SPTF_ERR_ARG;
    }
    port->pending = NULL;
    port->dst_port = dst_port;
    ip_addr_copy(port->dst_ip, *dst_ip);
    SPTF_LWIP_LOCK();
    port->pcb = udp_new();
    if (port->pcb == NULL)
    {
        st = SPTF_ERR_NOMEM;
    }
    else if (udp_bind(port->pcb, IP_ANY_TYPE, local_port) != (err_t)ERR_OK)
    {
        udp_remove(port->pcb);
        port->pcb = NULL;
        st = SPTF_ERR_IO;
    }
    else
    {
        /* hazır */
    }
    SPTF_LWIP_UNLOCK();
    return st;
}

sptf_status_t sptf_lwip_raw_set_destination(sptf_lwip_raw_t *port, const ip_addr_t *dst_ip, u16_t dst_port)
{
    if ((port == NULL) || (dst_ip == NULL))
    {
        return SPTF_ERR_ARG;
    }
    SPTF_LWIP_LOCK();
    ip_addr_copy(port->dst_ip, *dst_ip);
    port->dst_port = dst_port;
    SPTF_LWIP_UNLOCK();
    return SPTF_OK;
}

void sptf_lwip_raw_close(sptf_lwip_raw_t *port)
{
    if (port != NULL)
    {
        SPTF_LWIP_LOCK();
        if (port->pending != NULL)
        {
            (void)pbuf_free(port->pending);
            port->pending = NULL;
        }
        if (port->pcb != NULL)
        {
            udp_remove(port->pcb);
            port->pcb = NULL;
        }
        SPTF_LWIP_UNLOCK();
    }
}

void sptf_lwip_raw_transport(sptf_lwip_raw_t *port, sptf_transport_t *out)
{
    if (out != NULL)
    {
        out->acquire = lwip_acquire;
        out->send = lwip_send;
        out->ctx = port;
    }
}
