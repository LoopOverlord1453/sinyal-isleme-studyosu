/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    sptf_config.h
 * @brief   SPTF-UDP kütüphanesinin derleme zamanı ayarları.
 *
 * Her ayarın bir varsayılanı vardır. Varsayılan, derleyici komut satırından
 * (-DSPTF_CFG_MAX_STREAMS=2U gibi) ya da bu başlık eklenmeden önce tanımlanarak
 * değiştirilir. Bütün ayarlar önişlemcide sınanır; geçersiz bir değer derlemeyi
 * #error ile durdurur.
 *
 * Copyright (C) 2026, Kubilay Közleme <burhankozleme@gmail.com>
 */

#ifndef SPTF_CONFIG_H
#define SPTF_CONFIG_H

/**
 * Aynı anda açık tutulabilecek en fazla akış sayısı.
 *
 * Akış nesneleri kütüphane içindeki statik bir havuzdan verilir; dinamik bellek
 * kullanılmaz. Her akış SPTF_CFG_MAX_DATAGRAM_SIZE bayt çerçeve tamponu taşır.
 */
#ifndef SPTF_CFG_MAX_STREAMS
#define SPTF_CFG_MAX_STREAMS            (4U)
#endif

/**
 * Bir datagramın (UDP yükünün) bayt cinsinden üst sınırı.
 *
 * Standart Ethernet'te 1472 bayttır (1500 MTU - 20 IPv4 - 8 UDP). Yolun tamamı
 * jumbo çerçeve destekliyorsa 8972 bayta kadar çıkarılabilir. Akış başına RAM
 * kullanımını doğrudan belirler.
 */
#ifndef SPTF_CFG_MAX_DATAGRAM_SIZE
#define SPTF_CFG_MAX_DATAGRAM_SIZE      (1472U)
#endif

/*
 * SPTF_CFG_LWIP_CORE_LOCKING: lwIP çağrılarının çekirdek kilidiyle (LOCK_TCPIP_CORE)
 * korunması.
 *
 *   1U  Kütüphane fonksiyonları uygulama görevlerinden çağrılır (FreeRTOS, NO_SYS = 0,
 *       LWIP_TCPIP_CORE_LOCKING = 1). Her lwIP çağrısı kilit altında yapılır.
 *   0U  Çıplak metal (NO_SYS = 1) ya da çağrılar zaten tcpip iş parçacığından yapılıyor.
 *
 * Burada tanımlanmazsa sptf_udp.c, lwIP'nin NO_SYS ve LWIP_TCPIP_CORE_LOCKING
 * ayarlarına bakarak seçer. Değer lwIP başlıklarına bağlı olduğu için denetimi de
 * sptf_udp.c içinde yapılır.
 */

/* ------------------------------------------------------------------ denetimler */

#if (SPTF_CFG_MAX_STREAMS < 1U) || (SPTF_CFG_MAX_STREAMS > 16U)
#error "SPTF_CFG_MAX_STREAMS must be in the range 1..16"
#endif

#if (SPTF_CFG_MAX_DATAGRAM_SIZE < 256U) || (SPTF_CFG_MAX_DATAGRAM_SIZE > 65507U)
#error "SPTF_CFG_MAX_DATAGRAM_SIZE must be in the range 256..65507"
#endif

#endif /* SPTF_CONFIG_H */
