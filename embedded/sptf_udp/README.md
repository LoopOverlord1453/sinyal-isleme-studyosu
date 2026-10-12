# SPTF-UDP — gömülü gönderici kütüphanesi

Mikrodenetleyicinizdeki örnekleri Ethernet üzerinden **Signal Processing Toolbox**'a gerçek zamanlı gönderir.
C99, dinamik bellek yok, ~3.3–4.4 KB flash (Cortex-M, GCC -Os), statik RAM 0 B.

```c
sptf_link_config_t lc = {0};
sptf_lwip_raw_transport(&port, &lc.transport);     /* ya da sptf_socket_transport() */
sptf_link_init(&link, &lc);
sptf_stream_init(&adc, &link, &adc_cfg);
sptf_stream_write(&adc, half_buffer, 256u, SPTF_TS_AUTO);   /* her DMA yarım tamponunda */
```

| Klasör | İçerik |
|---|---|
| `include/` | `sptf.h` genel API, `sptf_config.h` derleme seçenekleri |
| `src/` | Çerçeve kodlayıcı + CRC-32, akış yönetimi |
| `port/lwip_raw/` | lwIP raw API (sıfır kopya; çıplak metal ve FreeRTOS çekirdek kilidi) |
| `port/socket/` | BSD soket: lwIP socket, POSIX, Windows Winsock |
| `examples/stm32_lwip_raw/` | STM32 + lwIP: ADC DMA çift tampon → akış |
| `examples/host/` | PC'den demo / test göndericisi |
| `test/` | Birim testleri, Python çapraz doğrulama aracı |
| `docs/` | `Doxyfile`, API ana sayfası (`doxygen docs/Doxyfile` → `docs/api/html`) |

**Belgeler**

- Kullanım kılavuzu: [../../docs/SPTF_UDP_Kullanim_Kilavuzu.md](../../docs/SPTF_UDP_Kullanim_Kilavuzu.md)
- Üst seviye protokol (normatif): [../../docs/SPTF_UDP_Uygulama_Protokolu.md](../../docs/SPTF_UDP_Uygulama_Protokolu.md)
- Çerçeve formatı (bayt düzeni): [../../docs/SPTF_Protokol_v1.md](../../docs/SPTF_Protokol_v1.md)

**Doğrulama:** `tests/test_embedded_sptf.py`. C birim testleri, 68 yapılandırmada Python referansıyla bayt bayt
eşitlik ve gerçek UDP üzerinden uçtan uca kayıpsız aktarım denetlenir.

Lisans: GNU LGPL 2.1.
