/* SPDX-License-Identifier: LGPL-2.1-only */
/*
 * Çalışma zamanı sinyal yükleme protokolünün (SPTG) kart tarafının arayüzünü içerir: kart
 * çalışırken stüdyo (PC) sinyal tablosunu Ethernet üzerinden UDP ile gönderir; kart tabloyu RAM'e
 * alır ve DAC1'den (PA4) çalar. Derleme ve yükleme gerekmez.
 *
 * Protokol istek - yanıt biçimindedir. PC her isteğe bir yanıt bekler; yanıt gelmezse aynı isteği
 * aynı sıra numarasıyla yeniden gönderir. Kart, sıra numarası ve türü son işlediği istekle aynı
 * olan isteği yeniden işlemez, önceki yanıtı yineler. Sayılar küçük uçludur (little-endian).
 *
 * Başlık (istek ve yanıt, 8 bayt):
 *
 * 	Ofset 	Boyut 	Tip 		Alan 				Değer
 * 	0 		4 		char[4] 	magic 				"SPTG"
 * 	4 		1 		uint8 		version 			1
 * 	5 		1 		uint8 		message_type 		İstek: 1 INFO, 2 BEGIN, 3 DATA, 4 START, 5 STOP;
 * 												yanıt: istek | 0x80
 * 	6 		2 		uint16 		sequence_number 	PC verir; yanıtta aynen döner
 *
 * İstekler:
 *
 * 	INFO 	Gövde yok. Kartın durumunu ve sınırlarını sorar; hiçbir şeyi değiştirmez
 * 	BEGIN 	Tablo tanımı (aşağıda). Çalan sinyali durdurur, yüklemeyi başlatır
 * 	DATA 	byte_offset (uint32) + tablo baytları. Baytları tablonun o konumuna yazar
 * 	START 	Gövde yok. Tablonun CRC-32'sini doğrular ve çalmaya başlar
 * 	STOP 	Gövde yok. Çalan sinyali durdurur
 *
 * BEGIN gövdesi (ofset 8, 20 bayt):
 *
 * 	Ofset 	Boyut 	Tip 	Alan 				Değer
 * 	8 		1 		uint8 	output 				1 (DAC1); bu kart başka çıkışı kabul etmez
 * 	9 		1 		uint8 	channel_count 		1
 * 	10 		1 		uint8 	dac_bits 			12
 * 	11 		1 		uint8 	alignment 			0 sağa, 1 sola hizalı
 * 	12 		2 		uint16 	timer_prescaler 	TIM6 PSC
 * 	14 		2 		uint16 	timer_autoreload 	TIM6 ARR (en az 1)
 * 	16 		4 		uint32 	timer_clock_hz 		PC'nin PSC / ARR hesabında kullandığı zamanlayıcı saati
 * 	20 		4 		uint32 	sample_count 		Örnek sayısı (1 .. 16384)
 * 	24 		4 		uint32 	table_crc32 		Tablo baytlarının CRC-32'si (IEEE 802.3 / zlib)
 *
 * Tablo baytları: DAC kelimeleri, örnek başına 2 bayt (küçük uçlu).
 *
 * Yanıt gövdesi (ofset 8, 20 bayt; bütün yanıtlarda aynı):
 *
 * 	Ofset 	Boyut 	Tip 	Alan 					Değer
 * 	8 		2 		uint16 	status 					0 tamam; 1 istek, 2 sürüm, 3 tanım, 4 saat, 5 boyut,
 * 													6 sıra, 7 CRC, 8 donanım hatası
 * 	10 		1 		uint8 	player_state 			0 sinyal yok, 1 çalışıyor, 3 başlatma hatası,
 * 													4 DMA alt akışı, 5 durduruldu
 * 	11 		1 		uint8 	signal_source 			0 yok, 1 derlenmiş sinyal, 2 çalışma zamanı tablosu
 * 	12 		4 		uint32 	timer_clock_hz 			TIM6'nın giriş saati
 * 	16 		4 		uint32 	table_capacity_bytes 	Tablo için ayrılan RAM
 * 	20 		2 		uint16 	max_chunk_bytes 		Bir DATA isteğinde en fazla tablo baytı
 * 	22 		2 		uint16 	underrun_count 			DAC DMA alt akışı sayısı (65535'te durur)
 * 	24 		4 		uint32 	table_crc32 			Çalan çalışma zamanı tablosunun CRC-32'si (yoksa 0)
 *
 * Kimlik doğrulama yoktur: porta erişen her uç sinyali değiştirebilir; yalnızca kapalı test ağında
 * kullanılır.
 *
 * Copyright (C) 2026, Kubilay Közleme
 * Author: Kubilay Közleme <burhankozleme@gmail.com>
 */

#ifndef INC_SIGNAL_REMOTE_H_
#define INC_SIGNAL_REMOTE_H_

#include <stdint.h>


/*
 * Protokol ve kaynak sınırları:
 *
 * 	SIGNAL_REMOTE_UDP_PORT 				Kartın dinlediği UDP portu
 * 	SIGNAL_REMOTE_TABLE_CAPACITY_BYTES 	Tablo için ayrılan RAM; 16384 örnek alır
 * 	SIGNAL_REMOTE_REPLY_BYTES 			Yanıt uzunluğu (başlık + gövde)
 *
 */
#define SIGNAL_REMOTE_UDP_PORT	50506U
#define SIGNAL_REMOTE_TABLE_CAPACITY_BYTES	32768U
#define SIGNAL_REMOTE_REPLY_BYTES	28U



/*
 * Bir istek datagramını işler ve yanıtını üretir. request gelen datagram, request_length_bytes
 * uzunluğudur; yanıt reply tamponuna yazılır, reply_capacity_bytes en az SIGNAL_REMOTE_REPLY_BYTES
 * olmalıdır. Ağdan bağımsızdır; bütün çağrılar tek bir görevden yapılır.
 *
 * Gönderilecek yanıtın uzunluğunu (SIGNAL_REMOTE_REPLY_BYTES) döner. Datagram bu protokole ait
 * değilse (kısa, magic farklı ya da kendisi bir yanıt) 0 döner ve yanıt gönderilmez.
 */
uint32_t signal_remote_handle_datagram(const uint8_t *request, uint32_t request_length_bytes, uint8_t *reply,
                                       uint32_t reply_capacity_bytes);



/*
 * SIGNAL_REMOTE_UDP_PORT'u dinler; gelen her datagramı signal_remote_handle_datagram() ile işleyip
 * yanıtını gönderene yollar. lwIP kurulduktan (MX_LWIP_Init()) sonra bir görevden çağrılır ve geri
 * dönmez. Yalnızca bağlantı oluşturulamaz ya da porta bağlanamazsa döner.
 */
void signal_remote_run(void);

#endif /* INC_SIGNAL_REMOTE_H_ */
