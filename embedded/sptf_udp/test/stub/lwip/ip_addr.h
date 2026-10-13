/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    ip_addr.h
 * @brief   lwIP saplaması: IPv4 adres tipi (lwip/ip_addr.h, yalnız IPv4 yapılandırması).
 */

#ifndef LWIP_STUB_IP_ADDR_H
#define LWIP_STUB_IP_ADDR_H

#include "lwip/arch.h"

typedef struct ip4_addr
{
    u32_t addr;                     /* ağ bayt sırası */
} ip4_addr_t;

typedef ip4_addr_t ip_addr_t;

extern const ip_addr_t ip_addr_any;

#define IP_ADDR_ANY                             (&ip_addr_any)
#define ip_addr_set_ip4_u32_val(ipaddr, val)    ((ipaddr).addr = (val))

#endif /* LWIP_STUB_IP_ADDR_H */
