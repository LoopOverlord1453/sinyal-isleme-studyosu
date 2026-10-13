/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    sptf_crc32.h
 * @brief   SPTF çerçeveleri için CRC-32/ISO-HDLC hesabı.
 *
 * Algoritma Ethernet ve zlib ile aynıdır: polinom 0x04C11DB7 (yansıtılmış biçimi
 * 0xEDB88320), başlangıç 0xFFFFFFFF, çıkışta 0xFFFFFFFF ile XOR, giriş ve çıkış
 * yansıtılmış. Kontrol değeri: "123456789" için 0xCBF43926.
 *
 * Hesap 256 girişli sabit bir tabloyla (1 KiB, flash) bayt bayt yapılır. Fonksiyonlar
 * durum tutmaz; aynı anda birden fazla görevden çağrılabilir.
 *
 * Copyright (C) 2026, Kubilay Közleme <burhankozleme@gmail.com>
 */

#ifndef SPTF_CRC32_H
#define SPTF_CRC32_H

#include <stdint.h>

/** Parça parça hesapta ilk sptf_crc32_update() çağrısına verilecek başlangıç değeri. */
#define SPTF_CRC32_INITIAL_VALUE    (0xFFFFFFFFU)

/**
 * @brief  CRC ara değerini verilen baytlarla günceller.
 *
 * Çıkış XOR'u uygulanmaz; son değer sptf_crc32_finish() ile alınır.
 *
 * @param  crc     Önceki ara değer (ilk çağrıda SPTF_CRC32_INITIAL_VALUE).
 * @param  data    Baytlar. length 0 ise NULL olabilir.
 * @param  length  Bayt sayısı.
 * @return Yeni ara değer. data NULL ise crc değişmeden döner.
 */
uint32_t sptf_crc32_update(uint32_t crc, const uint8_t *data, uint32_t length);

/**
 * @brief  Ara değere çıkış XOR'unu uygular.
 * @param  crc  sptf_crc32_update() sonucu.
 * @return Çerçeveye yazılacak CRC-32 değeri.
 */
uint32_t sptf_crc32_finish(uint32_t crc);

/**
 * @brief  Tek parça bir bayt dizisinin CRC-32 değerini hesaplar.
 * @param  data    Baytlar. length 0 ise NULL olabilir.
 * @param  length  Bayt sayısı.
 * @return CRC-32 değeri.
 */
uint32_t sptf_crc32_compute(const uint8_t *data, uint32_t length);

#endif /* SPTF_CRC32_H */
