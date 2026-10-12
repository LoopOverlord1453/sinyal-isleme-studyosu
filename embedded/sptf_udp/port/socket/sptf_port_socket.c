/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    sptf_port_socket.c
 * @brief   BSD soket taşıma portu (lwIP socket / POSIX / Winsock 2).
 * @ingroup sptf_port
 */
#include "sptf_port_socket.h"

#include <string.h>

#if defined(SPTF_SOCKET_LWIP)
#include "lwip/errno.h"
/** @brief Geçersiz soket tanıtıcısı. */
#define SPTF_INVALID_FD (-1)
/** @brief Soketi kapatır. */
#define sptf_close_fd(fd) ((void)lwip_close(fd))
#elif defined(_WIN32)
/** @brief Geçersiz soket tanıtıcısı. */
#define SPTF_INVALID_FD INVALID_SOCKET
/** @brief Soketi kapatır. */
#define sptf_close_fd(fd) ((void)closesocket(fd))
#else
#include <errno.h>
#include <sys/socket.h>
#include <unistd.h>
/** @brief Geçersiz soket tanıtıcısı. */
#define SPTF_INVALID_FD (-1)
/** @brief Soketi kapatır. */
#define sptf_close_fd(fd) ((void)close(fd))
#endif

/** @brief Hedef adres yapısını doldurur. */
static void fill_dst(struct sockaddr_in *dst, const uint8_t ip[4], uint16_t port)
{
    uint32_t host = ((uint32_t)ip[0] << 24u) | ((uint32_t)ip[1] << 16u) | ((uint32_t)ip[2] << 8u) | (uint32_t)ip[3];
    (void)memset(dst, 0, sizeof(*dst));
    dst->sin_family = AF_INET;
    dst->sin_port = htons(port);
    dst->sin_addr.s_addr = htonl(host);
}

/** @brief ::sptf_transport_t::acquire — port içindeki tamponu verir. */
static uint8_t *socket_acquire(void *ctx, size_t len)
{
    sptf_socket_t *port = (sptf_socket_t *)ctx;
    if ((port == NULL) || (!port->open) || (len == 0u) || (len > sizeof(port->buf)))
    {
        return NULL;
    }
    port->busy = true;
    return port->buf;
}

/** @brief ::sptf_transport_t::send — `sendto()`. */
static sptf_status_t socket_send(void *ctx, uint8_t *buf, size_t len)
{
    sptf_socket_t *port = (sptf_socket_t *)ctx;
    sptf_status_t st = SPTF_OK;
    if ((port == NULL) || (buf != port->buf) || (!port->busy))
    {
        return SPTF_ERR_ARG;
    }
    port->busy = false;
    if (len > 0u)
    {
#if defined(_WIN32) && !defined(SPTF_SOCKET_LWIP)
        const int sent = sendto(port->fd, (const char *)buf, (int)len, 0, (const struct sockaddr *)&port->dst,
                                (int)sizeof(port->dst));
        const int err = (sent < 0) ? WSAGetLastError() : 0;
        const bool transient = (err == WSAEWOULDBLOCK) || (err == WSAENOBUFS);
#else
        const ssize_t sent = sendto(port->fd, buf, len, 0, (const struct sockaddr *)&port->dst,
                                    (socklen_t)sizeof(port->dst));
        const int err = (sent < 0) ? errno : 0;   /* errno, sendto'dan hemen sonra okunur */
        const bool transient = (err == EAGAIN) || (err == EWOULDBLOCK) || (err == ENOBUFS);
#endif
        if (sent < 0)
        {
            st = transient ? SPTF_ERR_BUSY : SPTF_ERR_IO;
        }
        else if ((size_t)sent != len)
        {
            st = SPTF_ERR_IO;
        }
        else
        {
            /* gönderildi */
        }
    }
    return st;
}

sptf_status_t sptf_socket_open(sptf_socket_t *port, const uint8_t dst_ipv4[4], uint16_t dst_port)
{
    if ((port == NULL) || (dst_ipv4 == NULL))
    {
        return SPTF_ERR_ARG;
    }
    (void)memset(port, 0, sizeof(*port));
#if defined(_WIN32) && !defined(SPTF_SOCKET_LWIP)
    {
        WSADATA wsa;
        if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0)
        {
            return SPTF_ERR_IO;
        }
    }
#endif
    port->fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (port->fd == SPTF_INVALID_FD)
    {
#if defined(_WIN32) && !defined(SPTF_SOCKET_LWIP)
        (void)WSACleanup();
#endif
        return SPTF_ERR_IO;
    }
    fill_dst(&port->dst, dst_ipv4, dst_port);
    port->open = true;
    port->busy = false;
    return SPTF_OK;
}

sptf_status_t sptf_socket_set_destination(sptf_socket_t *port, const uint8_t dst_ipv4[4], uint16_t dst_port)
{
    if ((port == NULL) || (dst_ipv4 == NULL))
    {
        return SPTF_ERR_ARG;
    }
    fill_dst(&port->dst, dst_ipv4, dst_port);
    return SPTF_OK;
}

void sptf_socket_close(sptf_socket_t *port)
{
    if ((port != NULL) && port->open)
    {
        sptf_close_fd(port->fd);
        port->open = false;
#if defined(_WIN32) && !defined(SPTF_SOCKET_LWIP)
        (void)WSACleanup();
#endif
    }
}

void sptf_socket_transport(sptf_socket_t *port, sptf_transport_t *out)
{
    if (out != NULL)
    {
        out->acquire = socket_acquire;
        out->send = socket_send;
        out->ctx = port;
    }
}
