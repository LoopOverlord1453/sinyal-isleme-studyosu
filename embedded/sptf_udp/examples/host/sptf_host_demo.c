/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    sptf_host_demo.c
 * @brief   PC'den (Windows / Linux) gerçek zamanlı SPTF-UDP göndericisi: kartsız deneme ve uçtan uca test.
 *
 * Kullanım:
 * @verbatim
 * sptf_host_demo [IP=127.0.0.1] [PORT=50505] [SÜRE_s=10 (0 = sürekli)] [MOD=demo|ramp] [RAMPA_HIZI_Hz=50000]
 * @endverbatim
 *  - `demo`: akış 1 "Motor akımları" (3 × int16, 20 kHz, üç faz 50 Hz + 5. harmonik + gürültü),
 *            akış 2 "IMU" (6 × float32, 1 kHz), her 2 s'de durum işareti.
 *  - `ramp`: akış 9 "Rampa" (4 × int32), değer = 4 · örnek + kanal; 1000. örnekte işaret (test için).
 *
 * Derleme (MinGW): `gcc -std=c99 -O2 -I../../include -I../../port/socket ../../src/\*.c
 * ../../port/socket/sptf_port_socket.c sptf_host_demo.c -lws2_32 -o sptf_host_demo.exe`
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sptf.h"
#include "sptf_port_socket.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <time.h>
#endif

/** @brief Monoton saat (ns). */
static uint64_t host_clock_ns(void *ctx)
{
    (void)ctx;
#if defined(_WIN32)
    LARGE_INTEGER f;
    LARGE_INTEGER c;
    (void)QueryPerformanceFrequency(&f);
    (void)QueryPerformanceCounter(&c);
    return (uint64_t)((double)c.QuadPart * (1.0e9 / (double)f.QuadPart));
#else
    struct timespec ts;
    (void)clock_gettime(CLOCK_MONOTONIC, &ts);
    return ((uint64_t)ts.tv_sec * 1000000000u) + (uint64_t)ts.tv_nsec;
#endif
}

/** @brief ~1 ms uyur. */
static void sleep_1ms(void)
{
#if defined(_WIN32)
    Sleep(1);
#else
    struct timespec ts = {0, 1000000L};
    (void)nanosleep(&ts, NULL);
#endif
}

/** @brief "a.b.c.d" IPv4 metnini ayrıştırır (strtoul: dönüşüm hataları denetlenir). */
static bool parse_ipv4(const char *text, uint8_t ip[4])
{
    const char *p = text;
    int k;
    for (k = 0; k < 4; k++)
    {
        char *end = NULL;
        const unsigned long v = strtoul(p, &end, 10);
        const char want = (k < 3) ? '.' : '\0';
        if ((end == p) || (v > 255u) || (*end != want))
        {
            return false;
        }
        ip[k] = (uint8_t)v;
        p = end + 1;
    }
    return true;
}

static sptf_socket_t g_port;
static sptf_link_t g_link;

/* ------------------------------------------------------------------------------------------- demo akışları */
#define MOTOR_FS   20000.0
#define MOTOR_BLK  200u
#define IMU_FS     1000.0
#define IMU_BLK    10u

static const sptf_channel_t motor_ch[3] = {
    {"Ia", "A", 10.0 / 32768.0, 0.0}, {"Ib", "A", 10.0 / 32768.0, 0.0}, {"Ic", "A", 10.0 / 32768.0, 0.0}};
static const sptf_stream_config_t motor_cfg = {1u, "Motor akımları", MOTOR_FS, SPTF_INT16, SPTF_LAYOUT_INTERLEAVED,
                                               3u, motor_ch, false};
static const sptf_channel_t imu_ch[6] = {{"ax", "g", 1.0, 0.0},       {"ay", "g", 1.0, 0.0},
                                         {"az", "g", 1.0, 0.0},       {"gx", "\xC2\xB0/s", 1.0, 0.0},
                                         {"gy", "\xC2\xB0/s", 1.0, 0.0}, {"gz", "\xC2\xB0/s", 1.0, 0.0}};
static const sptf_stream_config_t imu_cfg = {2u, "IMU", IMU_FS, SPTF_FLOAT32, SPTF_LAYOUT_INTERLEAVED, 6u, imu_ch,
                                             true};

/** @brief Basit doğrusal eşlenik üreteç: [-1, 1) gürültü. */
static double noise(uint32_t *state)
{
    *state = (*state * 1664525u) + 1013904223u;
    return ((double)(*state >> 8u) / 8388608.0) - 1.0;
}

static int run_demo(double seconds)
{
    static sptf_stream_t motor;
    static sptf_stream_t imu;
    static int16_t mbuf[MOTOR_BLK * 3u];
    static float ibuf[IMU_BLK * 6u];
    const double two_pi = 6.283185307179586;
    uint32_t rng = 12345u;
    uint64_t n_motor = 0u;
    uint64_t n_imu = 0u;
    uint64_t next_marker = 0u;
    int state = 0;
    const uint64_t t0 = host_clock_ns(NULL);
    if ((sptf_stream_init(&motor, &g_link, &motor_cfg) != SPTF_OK) ||
        (sptf_stream_init(&imu, &g_link, &imu_cfg) != SPTF_OK))
    {
        (void)printf("akış başlatılamadı\n");
        return 1;
    }
    for (;;)
    {
        const double t = (double)(host_clock_ns(NULL) - t0) * 1.0e-9;
        if ((seconds > 0.0) && (t >= seconds))
        {
            break;
        }
        while ((double)(n_motor + MOTOR_BLK) <= t * MOTOR_FS)
        {
            uint32_t i;
            uint32_t ph;
            for (i = 0u; i < MOTOR_BLK; i++)
            {
                const double tt = (double)(n_motor + i) / MOTOR_FS;
                for (ph = 0u; ph < 3u; ph++)
                {
                    const double a = (two_pi * 50.0 * tt) - ((double)ph * two_pi / 3.0);
                    const double amp = (6.0 * sin(a)) + (0.6 * sin(5.0 * a)) + (0.05 * noise(&rng));
                    mbuf[(i * 3u) + ph] = (int16_t)(amp * 32768.0 / 10.0);
                }
            }
            (void)sptf_stream_write(&motor, mbuf, MOTOR_BLK, SPTF_TS_AUTO);
            n_motor += MOTOR_BLK;
        }
        while ((double)(n_imu + IMU_BLK) <= t * IMU_FS)
        {
            uint32_t i;
            for (i = 0u; i < IMU_BLK; i++)
            {
                const double tt = (double)(n_imu + i) / IMU_FS;
                ibuf[(i * 6u) + 0u] = (float)((0.10 * sin(two_pi * 0.5 * tt)) + (0.01 * noise(&rng)));
                ibuf[(i * 6u) + 1u] = (float)((0.05 * cos(two_pi * 0.3 * tt)) + (0.01 * noise(&rng)));
                ibuf[(i * 6u) + 2u] = (float)(1.0 + (0.01 * noise(&rng)));
                ibuf[(i * 6u) + 3u] = (float)((20.0 * cos(two_pi * 0.5 * tt)) + (0.5 * noise(&rng)));
                ibuf[(i * 6u) + 4u] = (float)((-10.0 * sin(two_pi * 0.3 * tt)) + (0.5 * noise(&rng)));
                ibuf[(i * 6u) + 5u] = (float)(0.5 * noise(&rng));
            }
            (void)sptf_stream_write(&imu, ibuf, IMU_BLK, SPTF_TS_AUTO);
            n_imu += IMU_BLK;
        }
        if (n_motor >= next_marker)
        {
            state = !state;
            (void)sptf_stream_marker(&motor, SPTF_MARKER_AT_NEXT, (uint32_t)state,
                                     state ? "durum = RUN" : "durum = IDLE", SPTF_TS_AUTO);
            next_marker = n_motor + (uint64_t)(2.0 * MOTOR_FS);
        }
        (void)sptf_link_poll(&g_link);
        sleep_1ms();
    }
    (void)printf("motor: %llu örnek, imu: %llu örnek, %lu datagram, kayıp örnek %llu\n",
                 (unsigned long long)motor.stats.samples_sent, (unsigned long long)imu.stats.samples_sent,
                 (unsigned long)g_link.stats.datagrams,
                 (unsigned long long)(motor.stats.samples_dropped + imu.stats.samples_dropped));
    return 0;
}

/* ---------------------------------------------------------------------------------------- rampa (test) */
static int run_ramp(double seconds, double rate)
{
    static const sptf_channel_t ch[4] = {{"r0", "", 1.0, 0.0}, {"r1", "", 1.0, 0.0},
                                         {"r2", "", 1.0, 0.0}, {"r3", "", 1.0, 0.0}};
    static sptf_stream_t s;
    static int32_t buf[1000u * 4u];
    sptf_stream_config_t c = {9u, "Rampa", 50000.0, SPTF_INT32, SPTF_LAYOUT_INTERLEAVED, 4u, ch, true};
    const uint64_t total = (uint64_t)((seconds * rate) + 0.5); /* en yakın tamsayı (0.3 · 20000 = 5999.99…) */
    const uint32_t blk = (rate >= 100000.0) ? 1000u : (uint32_t)((rate / 100.0) + 1.0);
    const uint64_t t0 = host_clock_ns(NULL);
    uint64_t n = 0u;
    bool marked = false;
    c.sample_rate_hz = rate;
    if (sptf_stream_init(&s, &g_link, &c) != SPTF_OK)
    {
        (void)printf("akış başlatılamadı\n");
        return 1;
    }
    while (n < total)
    {
        const double t = (double)(host_clock_ns(NULL) - t0) * 1.0e-9;
        while ((n < total) && ((double)n <= t * rate))
        {
            const uint32_t k = ((total - n) < blk) ? (uint32_t)(total - n) : blk;
            uint32_t i;
            uint32_t j;
            for (i = 0u; i < k; i++)
            {
                for (j = 0u; j < 4u; j++)
                {
                    buf[(i * 4u) + j] = (int32_t)(((n + i) * 4u) + j);
                }
            }
            if (sptf_stream_write(&s, buf, k, SPTF_TS_AUTO) != SPTF_OK)
            {
                (void)printf("yazma hatası\n");
            }
            n += k;
            if ((!marked) && (n >= 1000u))
            {
                (void)sptf_stream_marker(&s, 1000u, 7u, "rampa", SPTF_TS_AUTO);
                marked = true;
            }
        }
        sleep_1ms();
    }
    (void)printf("rampa: %llu örnek, %lu datagram, kayıp %llu\n", (unsigned long long)s.stats.samples_sent,
                 (unsigned long)g_link.stats.datagrams, (unsigned long long)s.stats.samples_dropped);
    return (s.stats.samples_dropped == 0u) ? 0 : 1;
}

int main(int argc, char **argv)
{
    uint8_t ip[4] = {127u, 0u, 0u, 1u};
    uint16_t port = (uint16_t)SPTF_UDP_PORT;
    double seconds = 10.0;
    const char *mode = "demo";
    double ramp_rate = 50000.0;
    sptf_link_config_t lc;
    int rc;
#if defined(_WIN32)
    (void)SetConsoleOutputCP(65001u); /* UTF-8 konsol çıktısı */
#endif
    if ((argc > 1) && (!parse_ipv4(argv[1], ip)))
    {
        (void)printf("geçersiz IP: %s\n", argv[1]);
        return 2;
    }
    if (argc > 2)
    {
        port = (uint16_t)strtoul(argv[2], NULL, 10);
    }
    if (argc > 3)
    {
        seconds = strtod(argv[3], NULL);
    }
    if (argc > 4)
    {
        mode = argv[4];
    }
    if (argc > 5)
    {
        ramp_rate = strtod(argv[5], NULL);
    }
    if (sptf_socket_open(&g_port, ip, port) != SPTF_OK)
    {
        (void)printf("soket açılamadı\n");
        return 3;
    }
    (void)memset(&lc, 0, sizeof(lc));
    sptf_socket_transport(&g_port, &lc.transport);
    lc.clock = host_clock_ns;
    if (sptf_link_init(&g_link, &lc) != SPTF_OK)
    {
        return 4;
    }
    (void)printf("SPTF-UDP %u.%u.%u -> %u.%u.%u.%u:%u, mod %s\n", SPTF_LIB_VERSION_MAJOR, SPTF_LIB_VERSION_MINOR,
                 SPTF_LIB_VERSION_PATCH, ip[0], ip[1], ip[2], ip[3], (unsigned)port, mode);
    rc = (strcmp(mode, "ramp") == 0) ? run_ramp(seconds, ramp_rate) : run_demo(seconds);
    sptf_socket_close(&g_port);
    return rc;
}
