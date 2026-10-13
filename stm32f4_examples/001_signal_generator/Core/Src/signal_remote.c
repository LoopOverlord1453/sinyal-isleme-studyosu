// SPDX-License-Identifier: LGPL-2.1-only
/*
 * Çalışma zamanı sinyal yükleme protokolünün (SPTG) kart tarafıdır: stüdyonun "Karta Gönder…"
 * penceresinden UDP ile gelen sinyal tablosunu RAM'e alır ve DAC1'den (PA4) çalar. Arayüz ve
 * çerçeve biçimi signal_remote.h'dedir.
 *
 * Dosyanın bölümleri:
 *
 * 	1 	Tablo belleği ve durum
 * 	2 	DAC1 çıkışı: zamanlayıcı saati, durdurma, başlatma
 * 	3 	İstekler: BEGIN, DATA, START, STOP ve yanıt
 * 	4 	UDP dinleyicisi (lwIP netconn)
 *
 * Yükleme sırası: BEGIN çalan sinyali durdurur ve tablo tanımını saklar; DATA istekleri baytları
 * tablodaki konumlarına yazar; START tablonun CRC-32'sini doğrular ve çalmaya başlar. Tek tablo
 * tamponu vardır: yükleme sırasında çıkış kapalıdır.
 *
 * Copyright (C) 2026, Kubilay Közleme
 * Author: Kubilay Közleme <burhankozleme@gmail.com>
 */

#include "signal_remote.h"

#include "main.h"
#include "dac.h"
#include "tim.h"
#include "cmsis_os.h"
#include "lwip/api.h"
#include "sptf_crc32.h"

#include <stdbool.h>
#include <string.h>


/*
 * Çerçeve sabitleri (bkz. signal_remote.h):
 *
 * 	SIGNAL_REMOTE_PROTOCOL_VERSION 		Başlıktaki sürüm
 * 	SIGNAL_REMOTE_HEADER_BYTES 			Başlık uzunluğu
 * 	SIGNAL_REMOTE_BEGIN_BODY_BYTES 		BEGIN gövdesinin uzunluğu
 * 	SIGNAL_REMOTE_DATA_PREFIX_BYTES 	DATA gövdesinde tablo baytlarından önceki byte_offset alanı
 * 	SIGNAL_REMOTE_MAX_DATAGRAM_BYTES 	Kabul edilen en büyük datagram (parçalanmayan UDP yükü)
 * 	SIGNAL_REMOTE_MAX_CHUNK_BYTES 		Bir DATA isteğindeki en fazla tablo baytı
 * 	SIGNAL_REMOTE_OFFSET_x 				Alanın datagram başından ofseti
 * 	SIGNAL_REMOTE_REPLY_FLAG 			Yanıtın message_type alanına eklenen bayrak
 *
 */
#define SIGNAL_REMOTE_PROTOCOL_VERSION	1U
#define SIGNAL_REMOTE_HEADER_BYTES	8U
#define SIGNAL_REMOTE_BEGIN_BODY_BYTES	20U
#define SIGNAL_REMOTE_DATA_PREFIX_BYTES	4U
#define SIGNAL_REMOTE_MAX_DATAGRAM_BYTES	1472U
#define SIGNAL_REMOTE_MAX_CHUNK_BYTES	( SIGNAL_REMOTE_MAX_DATAGRAM_BYTES - SIGNAL_REMOTE_HEADER_BYTES - \
                                          SIGNAL_REMOTE_DATA_PREFIX_BYTES )
#define SIGNAL_REMOTE_OFFSET_VERSION	4U
#define SIGNAL_REMOTE_OFFSET_MESSAGE_TYPE	5U
#define SIGNAL_REMOTE_OFFSET_SEQUENCE_NUMBER	6U
#define SIGNAL_REMOTE_OFFSET_BODY	8U
#define SIGNAL_REMOTE_REPLY_FLAG	0x80U



/*
 * İstek türleri (başlıktaki message_type).
 */
#define SIGNAL_REMOTE_MESSAGE_INFO	1U
#define SIGNAL_REMOTE_MESSAGE_BEGIN	2U
#define SIGNAL_REMOTE_MESSAGE_DATA	3U
#define SIGNAL_REMOTE_MESSAGE_START	4U
#define SIGNAL_REMOTE_MESSAGE_STOP	5U



/*
 * Bu kartın kabul ettiği tablo:
 *
 * 	SIGNAL_REMOTE_OUTPUT_DAC1 		BEGIN'deki output alanının tek geçerli değeri
 * 	SIGNAL_REMOTE_DAC_BITS 			DAC çözünürlüğü
 * 	SIGNAL_REMOTE_SAMPLE_BYTES 		Bir örneğin tablodaki uzunluğu
 * 	SIGNAL_REMOTE_MAX_SAMPLE_COUNT 	Tabloya sığan en fazla örnek
 *
 */
#define SIGNAL_REMOTE_OUTPUT_DAC1	1U
#define SIGNAL_REMOTE_DAC_BITS	12U
#define SIGNAL_REMOTE_SAMPLE_BYTES	2U
#define SIGNAL_REMOTE_MAX_SAMPLE_COUNT	( SIGNAL_REMOTE_TABLE_CAPACITY_BYTES / SIGNAL_REMOTE_SAMPLE_BYTES )



/*
 * netconn_recv() hata dönerse yeniden denemeden önceki bekleme.
 */
#define SIGNAL_REMOTE_RECEIVE_RETRY_DELAY_MS	10U



/*
 * Yanıttaki durum:
 *
 * 	SIGNAL_REMOTE_OK 			İstek yerine getirildi
 * 	SIGNAL_REMOTE_MESSAGE_ERR 	Bilinmeyen istek türü ya da gövde uzunluğu yanlış
 * 	SIGNAL_REMOTE_VERSION_ERR 	Protokol sürümü desteklenmiyor
 * 	SIGNAL_REMOTE_ARG_ERR 		Tablo tanımı geçersiz (çıkış, kanal, çözünürlük, hizalama, örnek
 * 								sayısı ya da ARR)
 * 	SIGNAL_REMOTE_CLOCK_ERR 	PC'nin kullandığı zamanlayıcı saati kartınkinden farklı
 * 	SIGNAL_REMOTE_SIZE_ERR 		Tablo ayrılan RAM'e sığmıyor ya da DATA tablonun dışına taşıyor
 * 	SIGNAL_REMOTE_STATE_ERR 	Yükleme başlatılmadan (BEGIN) DATA ya da START geldi
 * 	SIGNAL_REMOTE_CRC_ERR 		Yüklenen tablonun CRC-32'si BEGIN'dekiyle aynı değil
 * 	SIGNAL_REMOTE_PLAYER_ERR 	DAC, DMA ya da zamanlayıcı başlatılamadı
 *
 */
typedef enum
{
    SIGNAL_REMOTE_OK = 0,
    SIGNAL_REMOTE_MESSAGE_ERR,
    SIGNAL_REMOTE_VERSION_ERR,
    SIGNAL_REMOTE_ARG_ERR,
    SIGNAL_REMOTE_CLOCK_ERR,
    SIGNAL_REMOTE_SIZE_ERR,
    SIGNAL_REMOTE_STATE_ERR,
    SIGNAL_REMOTE_CRC_ERR,
    SIGNAL_REMOTE_PLAYER_ERR
} signal_remote_status_t;



/*
 * Yanıttaki player_state ve signal_source değerleri.
 */
#define SIGNAL_REMOTE_PLAYER_NO_SIGNAL	0U
#define SIGNAL_REMOTE_PLAYER_RUNNING	1U
#define SIGNAL_REMOTE_PLAYER_START_ERR	3U
#define SIGNAL_REMOTE_PLAYER_UNDERRUN_ERR	4U
#define SIGNAL_REMOTE_PLAYER_STOPPED	5U

#define SIGNAL_REMOTE_SOURCE_NONE	0U
#define SIGNAL_REMOTE_SOURCE_BUILT_IN	1U
#define SIGNAL_REMOTE_SOURCE_RUNTIME	2U



/* ------------------------------------------------------------------------------------------------
 * 1 	Tablo belleği ve durum
 * ------------------------------------------------------------------------------------------------ */

/*
 * Başlıktaki tanıtıcı ("SPTG").
 */
static const uint8_t protocol_magic[4] = { (uint8_t)'S', (uint8_t)'P', (uint8_t)'T', (uint8_t)'G' };



/*
 * Çalışma zamanı tablosu. DMA tabloyu 16 bitlik kelimeler olarak okur; DATA istekleri ise bayt
 * bayt yazar (table_bytes).
 */
static uint16_t table_samples[SIGNAL_REMOTE_MAX_SAMPLE_COUNT];
static uint8_t *const table_bytes = (uint8_t *)table_samples;



/*
 * BEGIN ile gelen tablo tanımı ve yükleme durumu:
 *
 * 	table_sample_count 		Tablodaki örnek sayısı
 * 	table_length_bytes 		Tablonun bayt cinsinden uzunluğu
 * 	table_hal_alignment 	HAL hizalama sabiti (DAC_ALIGN_12B_R ya da DAC_ALIGN_12B_L)
 * 	table_timer_prescaler 	TIM6 PSC
 * 	table_timer_autoreload 	TIM6 ARR
 * 	table_expected_crc32 	BEGIN ile gelen CRC-32
 * 	is_loading 				true: BEGIN alındı, START bekleniyor
 * 	active_table_crc32 		Çalan çalışma zamanı tablosunun CRC-32'si; yoksa 0
 *
 */
static uint32_t table_sample_count = 0U;
static uint32_t table_length_bytes = 0U;
static uint32_t table_hal_alignment = DAC_ALIGN_12B_R;
static uint16_t table_timer_prescaler = 0U;
static uint16_t table_timer_autoreload = 0U;
static uint32_t table_expected_crc32 = 0U;
static bool is_loading = false;
static uint32_t active_table_crc32 = 0U;



/*
 * Yanıtta bildirilen durum. player_state ve underrun_count kesmeden de (DMA alt akışı) yazılır.
 */
static volatile uint8_t player_state = SIGNAL_REMOTE_PLAYER_NO_SIGNAL;
static uint8_t signal_source = SIGNAL_REMOTE_SOURCE_NONE;
static volatile uint32_t underrun_count = 0U;



/*
 * Alınan istek ve üretilen yanıt datagramları. Görev yığınını büyütmemek için statiktir; yalnızca
 * dinleyen görev kullanır.
 */
static uint8_t request_datagram[SIGNAL_REMOTE_MAX_DATAGRAM_BYTES];
static uint8_t reply_datagram[SIGNAL_REMOTE_REPLY_BYTES];



/*
 * field'dan küçük uçlu 16 bitlik sayı okur.
 */
static uint16_t read_u16(const uint8_t *field)
{
    return (uint16_t)( (uint16_t)field[0] | (uint16_t)( (uint16_t)field[1] << 8U ) );
}



/*
 * field'dan küçük uçlu 32 bitlik sayı okur.
 */
static uint32_t read_u32(const uint8_t *field)
{
    return (uint32_t)field[0] | ( (uint32_t)field[1] << 8U ) | ( (uint32_t)field[2] << 16U ) |
        ( (uint32_t)field[3] << 24U );
}



/*
 * value sayısını field'a küçük uçlu 16 bit olarak yazar.
 */
static void write_u16(uint8_t *field, uint16_t value)
{
    field[0] = (uint8_t)( value & 0xFFU );
    field[1] = (uint8_t)( ( value >> 8U ) & 0xFFU );
}



/*
 * value sayısını field'a küçük uçlu 32 bit olarak yazar.
 */
static void write_u32(uint8_t *field, uint32_t value)
{
    field[0] = (uint8_t)( value & 0xFFU );
    field[1] = (uint8_t)( ( value >> 8U ) & 0xFFU );
    field[2] = (uint8_t)( ( value >> 16U ) & 0xFFU );
    field[3] = (uint8_t)( ( value >> 24U ) & 0xFFU );
}



/* ------------------------------------------------------------------------------------------------
 * 2 	DAC1 çıkışı
 * ------------------------------------------------------------------------------------------------ */

/*
 * TIM6'nın giriş saatini (Hz) döner. PC, örnekleme hızını veren PSC / ARR değerlerini bu saate
 * göre hesaplar.
 *
 * APB1 bölücüsü 1 ise zamanlayıcı saati PCLK1'dir, değilse PCLK1'in iki katıdır. TIMPRE biti
 * kuruluysa bölücü 2 ya da 4 iken HCLK, daha büyükken PCLK1'in dört katıdır.
 */
static uint32_t timer_clock_hz(void)
{
    RCC_ClkInitTypeDef clock_configuration = { 0 };
    uint32_t flash_latency = 0U;
    const uint32_t apb1_clock_hz = HAL_RCC_GetPCLK1Freq();
    uint32_t clock_hz = apb1_clock_hz;

    HAL_RCC_GetClockConfig(&clock_configuration, &flash_latency);

    if( clock_configuration.APB1CLKDivider != RCC_HCLK_DIV1 )
    {
        clock_hz = apb1_clock_hz * 2U;

#if defined(RCC_DCKCFGR_TIMPRE)
        if( ( RCC->DCKCFGR & RCC_DCKCFGR_TIMPRE ) != 0U )
        {
            if( ( clock_configuration.APB1CLKDivider == RCC_HCLK_DIV2 ) ||
                ( clock_configuration.APB1CLKDivider == RCC_HCLK_DIV4 ) )
            {
                clock_hz = HAL_RCC_GetHCLKFreq();
            }
            else
            {
                clock_hz = apb1_clock_hz * 4U;
            }
        }
#endif
    }

    return clock_hz;
}



/*
 * Zamanlayıcıyı, DAC1'i ve DMA akışını durdurur. Hiçbiri çalışmıyorken de çağrılabilir; main.c'nin
 * başlattığı derlenmiş sinyali de durdurur.
 */
static void stop_output(void)
{
    (void)HAL_TIM_Base_Stop(&htim6);
    (void)HAL_DAC_Stop_DMA(&hdac, DAC_CHANNEL_1);
    __HAL_DAC_CLEAR_FLAG(&hdac, DAC_FLAG_DMAUDR1);
}



/*
 * Yüklenen tabloyu DAC1'den çalmaya başlar: TIM6'ya tablonun PSC / ARR değerlerini yazar, DMA'yı
 * tabloya dairesel bağlar ve zamanlayıcıyı çalıştırır. TIM6'nın her güncellemesi DAC1'i tetikler;
 * DMA sıradaki örneği veri yazmacına taşır.
 *
 * Başarıda HAL_OK, aksi halde başarısız olan HAL çağrısının sonucu döner.
 */
static HAL_StatusTypeDef start_output(void)
{
    HAL_StatusTypeDef hal_status;

    stop_output();

    __HAL_TIM_SET_PRESCALER(&htim6, (uint32_t)table_timer_prescaler);
    __HAL_TIM_SET_AUTORELOAD(&htim6, (uint32_t)table_timer_autoreload);
    __HAL_TIM_SET_COUNTER(&htim6, 0U);

    /* PSC ve ARR ön yüklemelidir: güncelleme olayı değerleri hemen yükler (DAC henüz kapalı). */
    htim6.Instance->EGR = TIM_EGR_UG;

    hal_status = HAL_DAC_Start_DMA(&hdac, DAC_CHANNEL_1, (const uint32_t *)(uintptr_t)table_samples,
                                   table_sample_count, table_hal_alignment);

    if( hal_status == HAL_OK )
    {
        hal_status = HAL_TIM_Base_Start(&htim6);
    }

    return hal_status;
}



/*
 * HAL'in DAC1 DMA alt akışı geri çağrısıdır (TIM6_DAC kesmesinden): DMA bir örneği zamanında
 * taşıyamamıştır. HAL DMA isteğini kapatmıştır; çıkış son değerde kalır. dac_handle kullanılmaz.
 */
void HAL_DAC_DMAUnderrunCallbackCh1(DAC_HandleTypeDef *dac_handle)
{
    (void)dac_handle;

    underrun_count++;
    player_state = SIGNAL_REMOTE_PLAYER_UNDERRUN_ERR;
}



/* ------------------------------------------------------------------------------------------------
 * 3 	İstekler
 * ------------------------------------------------------------------------------------------------ */

/*
 * BEGIN isteğini işler: body gövdesindeki tablo tanımını doğrular, çalan sinyali durdurur ve
 * yüklemeyi başlatır. Tanım geçersizse hiçbir şey değişmez; çalan sinyal sürer.
 *
 * Başarıda SIGNAL_REMOTE_OK döner. Gövde uzunluğu yanlışsa SIGNAL_REMOTE_MESSAGE_ERR, tanım
 * geçersizse SIGNAL_REMOTE_ARG_ERR, zamanlayıcı saati uyuşmuyorsa SIGNAL_REMOTE_CLOCK_ERR, tablo
 * sığmıyorsa SIGNAL_REMOTE_SIZE_ERR döner.
 */
static signal_remote_status_t handle_begin(const uint8_t *body, uint32_t body_length_bytes)
{
    uint8_t output;
    uint8_t channel_count;
    uint8_t dac_bits;
    uint8_t alignment;
    uint16_t timer_autoreload;
    uint32_t sample_count;

    if( body_length_bytes != SIGNAL_REMOTE_BEGIN_BODY_BYTES )
    {
        return SIGNAL_REMOTE_MESSAGE_ERR;
    }

    output = body[0];
    channel_count = body[1];
    dac_bits = body[2];
    alignment = body[3];
    timer_autoreload = read_u16(&body[6]);
    sample_count = read_u32(&body[12]);

    if( ( output != SIGNAL_REMOTE_OUTPUT_DAC1 ) || ( channel_count != 1U ) || ( dac_bits != SIGNAL_REMOTE_DAC_BITS ) ||
        ( alignment > 1U ) || ( sample_count == 0U ) || ( timer_autoreload == 0U ) )
    {
        return SIGNAL_REMOTE_ARG_ERR;
    }

    if( read_u32(&body[8]) != timer_clock_hz() )
    {
        return SIGNAL_REMOTE_CLOCK_ERR;
    }

    if( sample_count > SIGNAL_REMOTE_MAX_SAMPLE_COUNT )
    {
        return SIGNAL_REMOTE_SIZE_ERR;
    }

    /* Tablo tamponu yeniden yazılacak: onu okuyan DMA önce durdurulur. */
    stop_output();
    player_state = SIGNAL_REMOTE_PLAYER_STOPPED;
    signal_source = SIGNAL_REMOTE_SOURCE_NONE;

    table_sample_count = sample_count;
    table_length_bytes = sample_count * SIGNAL_REMOTE_SAMPLE_BYTES;
    table_hal_alignment = ( alignment != 0U ) ? DAC_ALIGN_12B_L : DAC_ALIGN_12B_R;
    table_timer_prescaler = read_u16(&body[4]);
    table_timer_autoreload = timer_autoreload;
    table_expected_crc32 = read_u32(&body[16]);
    active_table_crc32 = 0U;
    is_loading = true;

    return SIGNAL_REMOTE_OK;
}



/*
 * DATA isteğini işler: body gövdesindeki tablo baytlarını byte_offset konumuna yazar.
 *
 * Başarıda SIGNAL_REMOTE_OK döner. Yükleme başlatılmadıysa SIGNAL_REMOTE_STATE_ERR, gövdede tablo
 * baytı yoksa SIGNAL_REMOTE_MESSAGE_ERR, baytlar tablonun dışına taşıyorsa SIGNAL_REMOTE_SIZE_ERR
 * döner.
 */
static signal_remote_status_t handle_data(const uint8_t *body, uint32_t body_length_bytes)
{
    uint32_t byte_offset;
    uint32_t chunk_length_bytes;

    if( is_loading == false )
    {
        return SIGNAL_REMOTE_STATE_ERR;
    }

    if( body_length_bytes <= SIGNAL_REMOTE_DATA_PREFIX_BYTES )
    {
        return SIGNAL_REMOTE_MESSAGE_ERR;
    }

    byte_offset = read_u32(&body[0]);
    chunk_length_bytes = body_length_bytes - SIGNAL_REMOTE_DATA_PREFIX_BYTES;

    if( ( chunk_length_bytes > table_length_bytes ) || ( byte_offset > ( table_length_bytes - chunk_length_bytes ) ) )
    {
        return SIGNAL_REMOTE_SIZE_ERR;
    }

    (void)memcpy(&table_bytes[byte_offset], &body[SIGNAL_REMOTE_DATA_PREFIX_BYTES], chunk_length_bytes);

    return SIGNAL_REMOTE_OK;
}



/*
 * START isteğini işler: yüklenen tablonun CRC-32'sini BEGIN'dekiyle karşılaştırır ve tabloyu
 * çalmaya başlar. CRC uyuşmazsa yükleme açık kalır; eksik DATA yeniden gönderilip START
 * yinelenebilir.
 *
 * Tablo çalınıyorsa SIGNAL_REMOTE_OK döner. Yükleme başlatılmadıysa SIGNAL_REMOTE_STATE_ERR, gövde
 * boş değilse SIGNAL_REMOTE_MESSAGE_ERR, CRC uyuşmuyorsa SIGNAL_REMOTE_CRC_ERR, donanım
 * başlatılamazsa SIGNAL_REMOTE_PLAYER_ERR döner.
 */
static signal_remote_status_t handle_start(uint32_t body_length_bytes)
{
    uint32_t loaded_table_crc32;

    if( is_loading == false )
    {
        return SIGNAL_REMOTE_STATE_ERR;
    }

    if( body_length_bytes != 0U )
    {
        return SIGNAL_REMOTE_MESSAGE_ERR;
    }

    loaded_table_crc32 = sptf_crc32_compute(table_bytes, table_length_bytes);

    if( loaded_table_crc32 != table_expected_crc32 )
    {
        return SIGNAL_REMOTE_CRC_ERR;
    }

    is_loading = false;

    if( start_output() != HAL_OK )
    {
        stop_output();
        player_state = SIGNAL_REMOTE_PLAYER_START_ERR;
        return SIGNAL_REMOTE_PLAYER_ERR;
    }

    player_state = SIGNAL_REMOTE_PLAYER_RUNNING;
    signal_source = SIGNAL_REMOTE_SOURCE_RUNTIME;
    active_table_crc32 = loaded_table_crc32;

    return SIGNAL_REMOTE_OK;
}



/*
 * STOP isteğini işler: çalan sinyali durdurur ve açık yüklemeyi iptal eder.
 *
 * Başarıda SIGNAL_REMOTE_OK, gövde boş değilse SIGNAL_REMOTE_MESSAGE_ERR döner.
 */
static signal_remote_status_t handle_stop(uint32_t body_length_bytes)
{
    if( body_length_bytes != 0U )
    {
        return SIGNAL_REMOTE_MESSAGE_ERR;
    }

    stop_output();
    player_state = SIGNAL_REMOTE_PLAYER_STOPPED;
    signal_source = SIGNAL_REMOTE_SOURCE_NONE;
    is_loading = false;
    active_table_crc32 = 0U;

    return SIGNAL_REMOTE_OK;
}



/*
 * İsteğin yanıtını reply tamponuna yazar: başlık (message_type yanıt bayrağıyla, sequence_number
 * aynen), remote_status ve kartın o anki durumu.
 */
static void write_reply(uint8_t *reply, uint8_t message_type, uint16_t sequence_number,
                        signal_remote_status_t remote_status)
{
    const uint32_t reported_underrun_count = underrun_count;
    uint8_t reported_state = player_state;
    uint8_t reported_source = signal_source;
    uint8_t *body = &reply[SIGNAL_REMOTE_OFFSET_BODY];

    /* Bu modül henüz bir şey çalmadıysa ama DAC1 açıksa, main.c derlenmiş sinyali başlatmıştır. */
    if( ( reported_state == SIGNAL_REMOTE_PLAYER_NO_SIGNAL ) && ( ( hdac.Instance->CR & DAC_CR_EN1 ) != 0U ) )
    {
        reported_state = SIGNAL_REMOTE_PLAYER_RUNNING;
        reported_source = SIGNAL_REMOTE_SOURCE_BUILT_IN;
    }

    (void)memcpy(reply, protocol_magic, sizeof(protocol_magic));
    reply[SIGNAL_REMOTE_OFFSET_VERSION] = (uint8_t)SIGNAL_REMOTE_PROTOCOL_VERSION;
    reply[SIGNAL_REMOTE_OFFSET_MESSAGE_TYPE] = (uint8_t)( message_type | SIGNAL_REMOTE_REPLY_FLAG );
    write_u16(&reply[SIGNAL_REMOTE_OFFSET_SEQUENCE_NUMBER], sequence_number);

    write_u16(&body[0], (uint16_t)remote_status);
    body[2] = reported_state;
    body[3] = reported_source;
    write_u32(&body[4], timer_clock_hz());
    write_u32(&body[8], SIGNAL_REMOTE_TABLE_CAPACITY_BYTES);
    write_u16(&body[12], (uint16_t)SIGNAL_REMOTE_MAX_CHUNK_BYTES);
    write_u16(&body[14], ( reported_underrun_count > 0xFFFFU ) ? (uint16_t)0xFFFFU : (uint16_t)reported_underrun_count);
    write_u32(&body[16], active_table_crc32);
}



uint32_t signal_remote_handle_datagram(const uint8_t *request, uint32_t request_length_bytes, uint8_t *reply,
                                       uint32_t reply_capacity_bytes)
{
    /*
     * Yinelenen isteği tanımak için saklanan son yanıt:
     *
     * 	has_cached_reply 		true: aşağıdaki alanlar geçerli
     * 	cached_message_type 	Son işlenen isteğin türü
     * 	cached_sequence_number 	Son işlenen isteğin sıra numarası
     * 	cached_reply 			O isteğe verilen yanıt
     *
     */
    static bool has_cached_reply = false;
    static uint8_t cached_message_type = 0U;
    static uint16_t cached_sequence_number = 0U;
    static uint8_t cached_reply[SIGNAL_REMOTE_REPLY_BYTES];

    signal_remote_status_t remote_status;
    const uint8_t *body;
    uint32_t body_length_bytes;
    uint8_t message_type;
    uint16_t sequence_number;

    if( ( request == NULL ) || ( reply == NULL ) || ( reply_capacity_bytes < SIGNAL_REMOTE_REPLY_BYTES ) ||
        ( request_length_bytes < SIGNAL_REMOTE_HEADER_BYTES ) )
    {
        return 0U;
    }

    if( memcmp(request, protocol_magic, sizeof(protocol_magic)) != 0 )
    {
        return 0U;
    }

    message_type = request[SIGNAL_REMOTE_OFFSET_MESSAGE_TYPE];
    sequence_number = read_u16(&request[SIGNAL_REMOTE_OFFSET_SEQUENCE_NUMBER]);

    if( ( message_type & SIGNAL_REMOTE_REPLY_FLAG ) != 0U )
    {
        return 0U;
    }

    /* Yanıtı kaybolan istek yeniden gelir: yeniden işlenmez, önceki yanıt yinelenir. */
    if( ( has_cached_reply == true ) && ( message_type != SIGNAL_REMOTE_MESSAGE_INFO ) &&
        ( message_type == cached_message_type ) && ( sequence_number == cached_sequence_number ) )
    {
        (void)memcpy(reply, cached_reply, SIGNAL_REMOTE_REPLY_BYTES);
        return SIGNAL_REMOTE_REPLY_BYTES;
    }

    body = &request[SIGNAL_REMOTE_OFFSET_BODY];
    body_length_bytes = request_length_bytes - SIGNAL_REMOTE_HEADER_BYTES;

    if( request[SIGNAL_REMOTE_OFFSET_VERSION] != SIGNAL_REMOTE_PROTOCOL_VERSION )
    {
        remote_status = SIGNAL_REMOTE_VERSION_ERR;
    }
    else if( message_type == SIGNAL_REMOTE_MESSAGE_INFO )
    {
        remote_status = ( body_length_bytes == 0U ) ? SIGNAL_REMOTE_OK : SIGNAL_REMOTE_MESSAGE_ERR;
    }
    else if( message_type == SIGNAL_REMOTE_MESSAGE_BEGIN )
    {
        remote_status = handle_begin(body, body_length_bytes);
    }
    else if( message_type == SIGNAL_REMOTE_MESSAGE_DATA )
    {
        remote_status = handle_data(body, body_length_bytes);
    }
    else if( message_type == SIGNAL_REMOTE_MESSAGE_START )
    {
        remote_status = handle_start(body_length_bytes);
    }
    else if( message_type == SIGNAL_REMOTE_MESSAGE_STOP )
    {
        remote_status = handle_stop(body_length_bytes);
    }
    else
    {
        remote_status = SIGNAL_REMOTE_MESSAGE_ERR;
    }

    write_reply(reply, message_type, sequence_number, remote_status);

    if( message_type != SIGNAL_REMOTE_MESSAGE_INFO )
    {
        (void)memcpy(cached_reply, reply, SIGNAL_REMOTE_REPLY_BYTES);
        cached_message_type = message_type;
        cached_sequence_number = sequence_number;
        has_cached_reply = true;
    }

    return SIGNAL_REMOTE_REPLY_BYTES;
}



/* ------------------------------------------------------------------------------------------------
 * 4 	UDP dinleyicisi
 * ------------------------------------------------------------------------------------------------ */

/*
 * reply_datagram'ın ilk reply_length_bytes baytını sender_address adresine ve sender_port portuna
 * yollar. Tampon ayrılamazsa yanıt gönderilmez; PC isteği yineler.
 */
static void send_reply(struct netconn *connection, const ip_addr_t *sender_address, u16_t sender_port,
                       uint32_t reply_length_bytes)
{
    struct netbuf *reply_netbuf = netbuf_new();

    if( reply_netbuf != NULL )
    {
        void *reply_payload = netbuf_alloc(reply_netbuf, (u16_t)reply_length_bytes);

        if( reply_payload != NULL )
        {
            (void)memcpy(reply_payload, reply_datagram, reply_length_bytes);
            (void)netconn_sendto(connection, reply_netbuf, sender_address, sender_port);
        }

        netbuf_delete(reply_netbuf);
    }
}



void signal_remote_run(void)
{
    struct netconn *connection = netconn_new(NETCONN_UDP);

    if( connection == NULL )
    {
        return;
    }

    if( netconn_bind(connection, IP_ADDR_ANY, (u16_t)SIGNAL_REMOTE_UDP_PORT) != (err_t)ERR_OK )
    {
        (void)netconn_delete(connection);
        return;
    }

    for( ;; )
    {
        struct netbuf *request_netbuf = NULL;

        if( netconn_recv(connection, &request_netbuf) == (err_t)ERR_OK )
        {
            const uint32_t request_length_bytes = (uint32_t)netbuf_len(request_netbuf);
            const ip_addr_t sender_address = *netbuf_fromaddr(request_netbuf);
            const u16_t sender_port = netbuf_fromport(request_netbuf);
            uint32_t reply_length_bytes = 0U;

            /* Sınırdan uzun datagram bu protokole ait olamaz; yanıtsız atılır. */
            if( ( request_length_bytes != 0U ) && ( request_length_bytes <= SIGNAL_REMOTE_MAX_DATAGRAM_BYTES ) )
            {
                (void)netbuf_copy(request_netbuf, request_datagram, (u16_t)request_length_bytes);
                reply_length_bytes = signal_remote_handle_datagram(request_datagram, request_length_bytes,
                                                                   reply_datagram, SIGNAL_REMOTE_REPLY_BYTES);
            }

            /* İstek tamponu yanıt için yer açmak üzere önce bırakılır. */
            netbuf_delete(request_netbuf);

            if( reply_length_bytes != 0U )
            {
                send_reply(connection, &sender_address, sender_port, reply_length_bytes);
            }
        }
        else
        {
            (void)osDelay(SIGNAL_REMOTE_RECEIVE_RETRY_DELAY_MS);
        }
    }
}
