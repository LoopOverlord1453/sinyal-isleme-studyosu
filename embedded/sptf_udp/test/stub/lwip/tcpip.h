/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    tcpip.h
 * @brief   lwIP saplaması: çekirdek kilidi (lwip/tcpip.h karşılığı).
 *
 * Kilit çağrıları sayılır; saplama her lwIP çağrısında kilidin tutulduğunu denetler.
 */

#ifndef LWIP_STUB_TCPIP_H
#define LWIP_STUB_TCPIP_H

void stub_lock_tcpip_core(void);
void stub_unlock_tcpip_core(void);

#define LOCK_TCPIP_CORE()       stub_lock_tcpip_core()
#define UNLOCK_TCPIP_CORE()     stub_unlock_tcpip_core()

#endif /* LWIP_STUB_TCPIP_H */
