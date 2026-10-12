/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    sptf_config.h
 * @brief   SPTF-UDP kütüphanesinin derleme zamanı yapılandırması.
 *
 * Varsayılanlar çoğu proje için uygundur. Değiştirmek için iki yol vardır:
 *  -# Derleyiciye makro verin: `-DSPTF_CFG_CRC_TABLE=0`
 *  -# Kendi başlığınızı yazın ve `-DSPTF_USER_CONFIG_FILE="\"sptf_user_config.h\""` ile gösterin;
 *     o dosyadaki tanımlar buradaki varsayılanların yerine geçer.
 *
 * @ingroup sptf_config
 */
#ifndef SPTF_CONFIG_H
#define SPTF_CONFIG_H

#if defined(SPTF_USER_CONFIG_FILE)
#include SPTF_USER_CONFIG_FILE
#endif

/**
 * @defgroup sptf_config Yapılandırma
 * @brief    Derleme zamanı seçenekleri (tümü `#ifndef` korumalıdır).
 * @{
 */

/**
 * @brief CRC-32 desteği (1 = derlenir, 0 = derlenmez).
 *
 * UDP'nin kendi sağlama toplamı olduğu için CRC isteğe bağlıdır. 0 yapılırsa CRC isteyen akış
 * yapılandırmaları ::SPTF_ERR_CONFIG ile reddedilir ve CRC kodu imaja girmez.
 */
#ifndef SPTF_CFG_CRC
#define SPTF_CFG_CRC 1
#endif

/**
 * @brief CRC-32 yöntemi: 1 = 256 girişli tablo (1 KiB flash, ~8 kat hızlı), 0 = bit bit (tablo yok).
 */
#ifndef SPTF_CFG_CRC_TABLE
#define SPTF_CFG_CRC_TABLE 1
#endif

/**
 * @brief İsteğe bağlı donanım CRC kancası.
 *
 * Tanımlanırsa kütüphane CRC'yi `uint32_t SPTF_CFG_CRC32_HOOK(const uint8_t *data, size_t len)` ile hesaplar.
 * Fonksiyon CRC-32/ISO-HDLC sonucunu (yansıtılmış, init 0xFFFFFFFF, xorout 0xFFFFFFFF; kontrol değeri
 * `"123456789"` → 0xCBF43926) döndürmelidir. Örnek: STM32 CRC birimi, giriş/çıkış ters çevirme açık.
 */
#if defined(__DOXYGEN__)
#define SPTF_CFG_CRC32_HOOK my_hw_crc32
#endif

/**
 * @brief Varsayılan en büyük UDP yükü (bayt). 1472 = Ethernet MTU 1500 − IPv4 (20) − UDP (8).
 *
 * Bağlantı yapılandırmasında (::sptf_link_config_t::max_datagram) akış başına daha küçük ya da jumbo çerçeve
 * için daha büyük (en fazla 65507) bir değer verilebilir.
 */
#ifndef SPTF_CFG_MAX_DATAGRAM
#define SPTF_CFG_MAX_DATAGRAM 1472u
#endif

/**
 * @brief DESCRIPTOR tekrar periyodu (ms). Protokol en fazla 1000 ms ister; varsayılan yarısıdır.
 */
#ifndef SPTF_CFG_DESCRIPTOR_PERIOD_MS
#define SPTF_CFG_DESCRIPTOR_PERIOD_MS 500u
#endif

/**
 * @brief Çalışma zamanı doğrulama makrosu. Varsayılan boştur; hata ayıklarken örn. `assert` bağlayın.
 *
 * Kütüphane hatalı argümanları zaten durum kodlarıyla bildirir; bu makro yalnızca iç tutarlılık içindir.
 */
#ifndef SPTF_CFG_ASSERT
#define SPTF_CFG_ASSERT(expr) ((void)0)
#endif

/**
 * @brief Hedefin bayt sırası: 1 = büyük uçlu (big-endian), 0 = küçük uçlu.
 *
 * GCC / Clang'de `__BYTE_ORDER__` ile otomatik bulunur; bulunamazsa küçük uçlu varsayılır (tüm Cortex-M,
 * RISC-V, Xtensa ve x86 küçük uçludur). Büyük uçlu hedefte örnekler kabloya yazılırken bileşen bileşen
 * çevrilir (kablo biçimi her zaman küçük uçludur).
 */
#ifndef SPTF_CFG_HOST_BIG_ENDIAN
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
#define SPTF_CFG_HOST_BIG_ENDIAN 1
#else
#define SPTF_CFG_HOST_BIG_ENDIAN 0
#endif
#endif

/** @} */

#endif /* SPTF_CONFIG_H */
