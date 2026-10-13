/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    pbuf.h
 * @brief   lwIP saplaması: paket tamponu (lwip/pbuf.h ile aynı imzalar).
 */

#ifndef LWIP_STUB_PBUF_H
#define LWIP_STUB_PBUF_H

#include "lwip/arch.h"
#include "lwip/err.h"

typedef enum
{
    PBUF_TRANSPORT = 54,
    PBUF_IP = 34,
    PBUF_LINK = 14,
    PBUF_RAW_TX = 0,
    PBUF_RAW = 1
} pbuf_layer;

typedef enum
{
    PBUF_RAM = 640,
    PBUF_ROM = 1,
    PBUF_REF = 65,
    PBUF_POOL = 386
} pbuf_type;

struct pbuf
{
    struct pbuf *next;
    void *payload;
    u16_t tot_len;
    u16_t len;
    u8_t type_internal;
    u8_t flags;
    u8_t ref;
    u8_t if_idx;
};

struct pbuf *pbuf_alloc(pbuf_layer l, u16_t length, pbuf_type type);
u8_t pbuf_free(struct pbuf *p);
err_t pbuf_take(struct pbuf *buf, const void *dataptr, u16_t len);

#endif /* LWIP_STUB_PBUF_H */
