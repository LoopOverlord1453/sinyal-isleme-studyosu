/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    udp.h
 * @brief   lwIP saplaması: UDP raw API (lwip/udp.h ile aynı imzalar).
 */

#ifndef LWIP_STUB_UDP_H
#define LWIP_STUB_UDP_H

#include "lwip/arch.h"
#include "lwip/err.h"
#include "lwip/ip_addr.h"
#include "lwip/pbuf.h"

struct udp_pcb
{
    u16_t local_port;
    u8_t in_use;
    u8_t bound;
};

struct udp_pcb *udp_new(void);
void udp_remove(struct udp_pcb *pcb);
err_t udp_bind(struct udp_pcb *pcb, const ip_addr_t *ipaddr, u16_t port);
err_t udp_sendto(struct udp_pcb *pcb, struct pbuf *p, const ip_addr_t *dst_ip, u16_t dst_port);

#endif /* LWIP_STUB_UDP_H */
