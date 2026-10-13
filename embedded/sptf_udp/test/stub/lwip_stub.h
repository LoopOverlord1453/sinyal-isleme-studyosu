/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    lwip_stub.h
 * @brief   lwIP saplamasının test kontrol arayüzü.
 *
 * Saplama gönderilen her datagramı (yükü, hedef adresi ve portları) saklar, istenen
 * çağrıda tampon ya da gönderim hatası üretir ve lwIP'ye her girişte çekirdek kilidinin
 * beklendiği gibi tutulup tutulmadığını sayar.
 */

#ifndef LWIP_STUB_H
#define LWIP_STUB_H

#include <stdbool.h>
#include <stdint.h>

/** Saklanabilecek en fazla datagram sayısı. */
#define STUB_MAX_DATAGRAMS          (8192U)

/** Datagram yüklerinin toplam üst sınırı (bayt). */
#define STUB_ARENA_SIZE             (8U * 1024U * 1024U)

/** Aynı anda en fazla UDP denetim bloğu. */
#define STUB_MAX_PCBS               (32U)

/** Hata enjeksiyonunda tutulabilecek en fazla çağrı indeksi. */
#define STUB_MAX_FAULTS             (16U)

/** Saplamayı ilk durumuna getirir: datagramlar, sayaçlar ve hatalar silinir. */
void stub_reset(void);

/** lwIP çağrılarında çekirdek kilidinin tutulmasının beklenip beklenmediği. */
void stub_expect_locking(bool expected);

/** Belirtilen pbuf_alloc() çağrısı (sıfırdan sayılır) NULL döndürür. */
void stub_fail_alloc_call(uint32_t call_index);

/** Belirtilen udp_sendto() çağrısı (sıfırdan sayılır) ERR_RTE döndürür. */
void stub_fail_send_call(uint32_t call_index);

/** Sonraki udp_new() çağrısı NULL döndürür. */
void stub_fail_next_udp_new(void);

/** Sonraki udp_bind() çağrısı ERR_USE döndürür. */
void stub_fail_next_bind(void);

uint32_t stub_datagram_count(void);
const uint8_t *stub_datagram(uint32_t index, uint32_t *length);
uint32_t stub_datagram_remote_ip(uint32_t index);       /* ana makine bayt sırası */
uint16_t stub_datagram_remote_port(uint32_t index);
uint16_t stub_datagram_local_port(uint32_t index);

uint32_t stub_alloc_calls(void);
uint32_t stub_send_calls(void);
uint32_t stub_live_pbufs(void);
uint32_t stub_live_pcbs(void);
uint32_t stub_lock_violations(void);
uint32_t stub_lock_count(void);

#endif /* LWIP_STUB_H */
