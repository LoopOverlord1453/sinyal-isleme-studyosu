/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    opt.h
 * @brief   lwIP saplaması: yapılandırma (lwip/opt.h karşılığı).
 *
 * Değerler derleyici komut satırından değiştirilir: -DNO_SYS=0 ile çekirdek kilitli
 * (FreeRTOS) yapılandırma sınanır.
 */

#ifndef LWIP_STUB_OPT_H
#define LWIP_STUB_OPT_H

#include "lwip/arch.h"

#ifndef NO_SYS
#define NO_SYS                      1
#endif

#ifndef LWIP_TCPIP_CORE_LOCKING
#define LWIP_TCPIP_CORE_LOCKING     1
#endif

#ifndef LWIP_UDP
#define LWIP_UDP                    1
#endif

#endif /* LWIP_STUB_OPT_H */
