/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    sptf_internal.h
 * @brief   Kütüphane içi yardımcılar (uygulama kodu bu başlığı kullanmamalıdır).
 * @internal
 */
#ifndef SPTF_INTERNAL_H
#define SPTF_INTERNAL_H

#include "sptf.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Kablodaki en büyük `first_sample` (hariç): 2^62. */
#define SPTF_PRIV_MAX_FIRST_SAMPLE ((uint64_t)1u << 62)

/**
 * @brief Daha büyük bir örnek bloğunun `[offset, offset + count)` dilimini DATA çerçevesi olarak yazar.
 *
 * Interleaved blokta dilim bitişiktir. Planar blokta kanal `c`'nin dilimi `block + (c · block_count + offset) · boyut`
 * adresindedir; çerçeveye kanal kanal kopyalanır.
 *
 * @param[out] buf           Hedef tampon.
 * @param[in]  cap           Tampon kapasitesi.
 * @param[in]  info          Başlık alanları.
 * @param[in]  cfg           Akış yapılandırması.
 * @param[in]  first_sample  Dilimin ilk örneğinin akış içi indeksi.
 * @param[in]  block         Bloğun başı.
 * @param[in]  block_count   Bloktaki kanal başına örnek sayısı (planar adım).
 * @param[in]  offset        Dilimin blok içindeki başlangıcı.
 * @param[in]  count         Dilimdeki örnek sayısı (≥ 1).
 * @param[out] out_len       Yazılan bayt.
 * @return ::SPTF_OK, ::SPTF_ERR_ARG, ::SPTF_ERR_CONFIG ya da ::SPTF_ERR_SIZE.
 */
sptf_status_t sptf_priv_encode_data(uint8_t *buf, size_t cap, const sptf_frame_info_t *info,
                                    const sptf_stream_config_t *cfg, uint64_t first_sample,
                                    const uint8_t *block, uint32_t block_count, uint32_t offset,
                                    uint32_t count, size_t *out_len);

/**
 * @brief `\0`'a kadar en fazla @p limit bayt sayar.
 * @param[in] s      Metin (NULL → 0).
 * @param[in] limit  Üst sınır.
 * @return min(strlen(s), limit).
 */
size_t sptf_priv_strnlen(const char *s, size_t limit);

#ifdef __cplusplus
}
#endif

#endif /* SPTF_INTERNAL_H */
