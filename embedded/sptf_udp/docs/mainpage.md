# SPTF-UDP {#mainpage}

Mikrodenetleyicideki örnekleri Ethernet üzerinden **Signal Processing Toolbox** arayüzüne gerçek zamanlı gönderen,
taşınabilir C99 kütüphanesi. Kablo biçimi SPTF v1'dir, taşıma yalnızca UDP'dir (varsayılan port 50505).

## Katmanlar

| Katman | İçerik |
|---|---|
| @ref sptf_types | Veri tipleri (17), yerleşim, durum kodları, yapılandırma |
| @ref sptf_frame | Durumsuz çerçeve kodlayıcılar ve CRC-32 (ağsız kullanım: dosya, UART köprüsü) |
| @ref sptf_link | UDP göndericisi: akış durumu, datagrama göre bölme, DESCRIPTOR tekrarı, işaretler |
| @ref sptf_port | lwIP raw (sıfır kopya), BSD soket (lwIP socket / POSIX / Winsock) |
| @ref sptf_config | Derleme zamanı seçenekleri |
| @ref sptf_examples | STM32 + lwIP ADC DMA örneği |

## Beş adımda

1. `include/`, `src/` ve bir `port/` klasörünü projeye ekleyin.
2. Portu açın: `sptf_lwip_raw_open(&port, &pc_ip, SPTF_UDP_PORT, 0)`.
3. Bağlantıyı kurun: `sptf_link_init(&link, &cfg)` (taşıma + isteğe bağlı saat).
4. Akışı tanımlayın: `sptf_stream_init(&stream, &link, &stream_cfg)`.
5. Her blokta: `sptf_stream_write(&stream, samples, n, SPTF_TS_AUTO)`.

Gerisini kütüphane yapar: DESCRIPTOR'ı ilk yazımda ve en geç 500 ms'de bir gönderir, büyük blokları datagram
boyutuna böler, `first_sample` / `seq` sayaçlarını yürütür, gönderilemeyen örnekleri alıcıda tam uzunlukta bir
boşluk olarak görünecek biçimde sayar.

## Belgeler

- Üst seviye protokol: `docs/SPTF_UDP_Uygulama_Protokolu.md`
- Çerçeve formatı: `docs/SPTF_Protokol_v1.md`
- Kullanım kılavuzu: `docs/SPTF_UDP_Kullanim_Kilavuzu.md`
