/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    lwip_stub.c
 * @brief   PC birim testleri için lwIP raw API saplaması.
 *
 * Gerçek lwIP'nin davranışı taklit edilir: udp_sendto() pbuf'u bırakmaz, pbuf_take()
 * tamponun boyunu aşan kopyayı reddeder, çekirdek kilidi özyinelemesizdir. Gönderilen
 * datagramlar bir bellek alanında sırayla saklanır.
 */

#include "lwip_stub.h"

#include <string.h>

#include "lwip/def.h"
#include "lwip/err.h"
#include "lwip/ip_addr.h"
#include "lwip/pbuf.h"
#include "lwip/tcpip.h"
#include "lwip/udp.h"

#define STUB_MAX_PBUFS          (8U)
#define STUB_PBUF_CAPACITY      (65535U)
#define STUB_EPHEMERAL_PORT     (49152U)

typedef struct
{
    uint32_t offset;
    uint32_t length;
    uint32_t remote_ip;
    uint16_t remote_port;
    uint16_t local_port;
} stub_record_t;

typedef struct
{
    struct pbuf header;
    bool in_use;
    uint8_t payload[STUB_PBUF_CAPACITY];
} stub_pbuf_t;

const ip_addr_t ip_addr_any = { 0U };

static stub_record_t records[STUB_MAX_DATAGRAMS];
static uint8_t arena[STUB_ARENA_SIZE];
static uint32_t record_count;
static uint32_t arena_used;

static stub_pbuf_t pbuf_pool[STUB_MAX_PBUFS];
static struct udp_pcb pcb_pool[STUB_MAX_PCBS];

static uint32_t alloc_faults[STUB_MAX_FAULTS];
static uint32_t alloc_fault_count;
static uint32_t send_faults[STUB_MAX_FAULTS];
static uint32_t send_fault_count;
static bool fail_udp_new;
static bool fail_bind;

static uint32_t alloc_calls;
static uint32_t send_calls;
static uint32_t live_pbufs;
static uint32_t live_pcbs;

static bool locking_expected;
static uint32_t lock_depth;
static uint32_t lock_count;
static uint32_t lock_violations;

/* ------------------------------------------------------------------ yardımcılar */

static void check_lock(void)
{
    const uint32_t expected_depth = locking_expected ? 1U : 0U;

    if (lock_depth != expected_depth)
    {
        lock_violations++;
    }
}

static bool in_list(const uint32_t *list, uint32_t count, uint32_t value)
{
    bool found = false;
    uint32_t index;

    for (index = 0U; index < count; index++)
    {
        if (list[index] == value)
        {
            found = true;
        }
    }

    return found;
}

static uint32_t swap32(uint32_t value)
{
    return ((value & 0x000000FFU) << 24U) | ((value & 0x0000FF00U) << 8U)
           | ((value & 0x00FF0000U) >> 8U) | ((value & 0xFF000000U) >> 24U);
}

/* -------------------------------------------------------------- kontrol arayüzü */

void stub_reset(void)
{
    uint32_t index;

    record_count = 0U;
    arena_used = 0U;
    alloc_fault_count = 0U;
    send_fault_count = 0U;
    fail_udp_new = false;
    fail_bind = false;
    alloc_calls = 0U;
    send_calls = 0U;
    lock_depth = 0U;
    lock_count = 0U;
    lock_violations = 0U;

    for (index = 0U; index < STUB_MAX_PBUFS; index++)
    {
        pbuf_pool[index].in_use = false;
    }

    live_pbufs = 0U;
}

void stub_expect_locking(bool expected)
{
    locking_expected = expected;
}

void stub_fail_alloc_call(uint32_t call_index)
{
    if (alloc_fault_count < STUB_MAX_FAULTS)
    {
        alloc_faults[alloc_fault_count] = call_index;
        alloc_fault_count++;
    }
}

void stub_fail_send_call(uint32_t call_index)
{
    if (send_fault_count < STUB_MAX_FAULTS)
    {
        send_faults[send_fault_count] = call_index;
        send_fault_count++;
    }
}

void stub_fail_next_udp_new(void)
{
    fail_udp_new = true;
}

void stub_fail_next_bind(void)
{
    fail_bind = true;
}

uint32_t stub_datagram_count(void)
{
    return record_count;
}

const uint8_t *stub_datagram(uint32_t index, uint32_t *length)
{
    const uint8_t *data = NULL;

    if (index < record_count)
    {
        data = &arena[records[index].offset];
        *length = records[index].length;
    }
    else
    {
        *length = 0U;
    }

    return data;
}

uint32_t stub_datagram_remote_ip(uint32_t index)
{
    return (index < record_count) ? records[index].remote_ip : 0U;
}

uint16_t stub_datagram_remote_port(uint32_t index)
{
    return (index < record_count) ? records[index].remote_port : (uint16_t)0U;
}

uint16_t stub_datagram_local_port(uint32_t index)
{
    return (index < record_count) ? records[index].local_port : (uint16_t)0U;
}

uint32_t stub_alloc_calls(void)
{
    return alloc_calls;
}

uint32_t stub_send_calls(void)
{
    return send_calls;
}

uint32_t stub_live_pbufs(void)
{
    return live_pbufs;
}

uint32_t stub_live_pcbs(void)
{
    return live_pcbs;
}

uint32_t stub_lock_violations(void)
{
    return lock_violations;
}

uint32_t stub_lock_count(void)
{
    return lock_count;
}

/* ------------------------------------------------------------ çekirdek kilidi */

void stub_lock_tcpip_core(void)
{
    if (lock_depth != 0U)
    {
        lock_violations++;              /* gerçek kilit özyinelemesizdir */
    }

    lock_depth++;
    lock_count++;
}

void stub_unlock_tcpip_core(void)
{
    if (lock_depth == 0U)
    {
        lock_violations++;
    }
    else
    {
        lock_depth--;
    }
}

/* ------------------------------------------------------------------ lwIP API */

u32_t lwip_htonl(u32_t x)
{
    return swap32(x);                   /* PC küçük uçludur */
}

struct pbuf *pbuf_alloc(pbuf_layer l, u16_t length, pbuf_type type)
{
    struct pbuf *result = NULL;
    const uint32_t call = alloc_calls;
    uint32_t index;

    check_lock();
    alloc_calls++;

    if ((l == PBUF_TRANSPORT) && (type == PBUF_RAM) && (!in_list(alloc_faults, alloc_fault_count, call)))
    {
        for (index = 0U; (index < STUB_MAX_PBUFS) && (result == NULL); index++)
        {
            if (!pbuf_pool[index].in_use)
            {
                pbuf_pool[index].in_use = true;
                pbuf_pool[index].header.next = NULL;
                pbuf_pool[index].header.payload = pbuf_pool[index].payload;
                pbuf_pool[index].header.tot_len = length;
                pbuf_pool[index].header.len = length;
                pbuf_pool[index].header.ref = 1U;
                result = &pbuf_pool[index].header;
                live_pbufs++;
            }
        }
    }

    return result;
}

u8_t pbuf_free(struct pbuf *p)
{
    u8_t freed = 0U;
    uint32_t index;

    check_lock();

    for (index = 0U; index < STUB_MAX_PBUFS; index++)
    {
        if ((p == &pbuf_pool[index].header) && pbuf_pool[index].in_use)
        {
            pbuf_pool[index].in_use = false;
            live_pbufs--;
            freed = 1U;
        }
    }

    return freed;
}

err_t pbuf_take(struct pbuf *buf, const void *dataptr, u16_t len)
{
    err_t result = (err_t)ERR_OK;

    check_lock();

    if ((buf == NULL) || (dataptr == NULL) || (buf->tot_len < len))
    {
        result = (err_t)ERR_ARG;
    }
    else
    {
        (void)memcpy(buf->payload, dataptr, (size_t)len);
    }

    return result;
}

struct udp_pcb *udp_new(void)
{
    struct udp_pcb *result = NULL;
    uint32_t index;

    check_lock();

    if (fail_udp_new)
    {
        fail_udp_new = false;
    }
    else
    {
        for (index = 0U; (index < STUB_MAX_PCBS) && (result == NULL); index++)
        {
            if (pcb_pool[index].in_use == 0U)
            {
                pcb_pool[index].in_use = 1U;
                pcb_pool[index].bound = 0U;
                pcb_pool[index].local_port = 0U;
                result = &pcb_pool[index];
                live_pcbs++;
            }
        }
    }

    return result;
}

void udp_remove(struct udp_pcb *pcb)
{
    check_lock();

    if ((pcb != NULL) && (pcb->in_use != 0U))
    {
        pcb->in_use = 0U;
        live_pcbs--;
    }
}

err_t udp_bind(struct udp_pcb *pcb, const ip_addr_t *ipaddr, u16_t port)
{
    err_t result = (err_t)ERR_OK;

    check_lock();

    if ((pcb == NULL) || (ipaddr == NULL))
    {
        result = (err_t)ERR_ARG;
    }
    else if (fail_bind)
    {
        fail_bind = false;
        result = (err_t)ERR_USE;
    }
    else
    {
        pcb->bound = 1U;
        pcb->local_port = (port != 0U) ? port : (u16_t)STUB_EPHEMERAL_PORT;
    }

    return result;
}

err_t udp_sendto(struct udp_pcb *pcb, struct pbuf *p, const ip_addr_t *dst_ip, u16_t dst_port)
{
    err_t result = (err_t)ERR_OK;
    const uint32_t call = send_calls;

    check_lock();
    send_calls++;

    if ((pcb == NULL) || (p == NULL) || (dst_ip == NULL) || (pcb->bound == 0U))
    {
        result = (err_t)ERR_ARG;
    }
    else if (in_list(send_faults, send_fault_count, call))
    {
        result = (err_t)ERR_RTE;
    }
    else if ((record_count >= STUB_MAX_DATAGRAMS) || ((arena_used + (uint32_t)p->tot_len) > STUB_ARENA_SIZE))
    {
        result = (err_t)ERR_MEM;
    }
    else
    {
        records[record_count].offset = arena_used;
        records[record_count].length = (uint32_t)p->tot_len;
        records[record_count].remote_ip = swap32(dst_ip->addr);
        records[record_count].remote_port = dst_port;
        records[record_count].local_port = pcb->local_port;
        (void)memcpy(&arena[arena_used], p->payload, (size_t)p->tot_len);
        arena_used += (uint32_t)p->tot_len;
        record_count++;
    }

    return result;
}
