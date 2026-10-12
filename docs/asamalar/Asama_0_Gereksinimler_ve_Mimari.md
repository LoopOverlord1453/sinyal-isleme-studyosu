# Aşama 0 — Gereksinimler ve Mimari

**Durum:** ✅ Tamamlandı

## Amaç
Projenin ne olduğunu, ne olmadığını ve hangi teknolojilerle yapılacağını netleştirmek.
Bu dosya, sonraki aşamalarda dönüp bakılacak **karar kaydıdır**.

## Alınan kararlar

| Konu | Karar |
|---|---|
| Ürün | PC'de çalışan, gerçek zamanlı sinyal görselleştirme ve analiz arayüzü |
| Gömülü taraf | Uygulama PC'de çalışır. Cihaz tarafı için yalnız gönderici kütüphanesi sağlanır: [`embedded/sptf_udp`](../../embedded/sptf_udp) (SPTF-UDP, C99) |
| Sinyal türü | Bağımsız. Formatta gelen her sinyal analiz edilebilmeli |
| Veri yolu | Ethernet, **yalnız UDP**, port 50505. TCP desteklenmez (karar: 2026-10-12). Uygulama katmanı: [SPTF_UDP_Uygulama_Protokolu.md](../SPTF_UDP_Uygulama_Protokolu.md) |
| Veri formatı | **SPTF v1** — kendini tanımlayan ikili çerçeve formatı. Bkz. [SPTF_Protokol_v1.md](../SPTF_Protokol_v1.md) |
| Kayıt dosyası | `.sptr` — gelen çerçeveler değiştirilmeden, alınma zamanlarıyla saklanır |
| Dil | Python 3.14, proje içi sanal ortam (`.venv`) |
| Arayüz | PyQt6 + pyqtgraph |
| Sinyal işleme | Hazır kütüphaneler: numpy, scipy (ileride PyWavelets) |
| İkonlar | qtawesome (Phosphor seti) |
| Tema | TerraFlux esinli koyu tema (#080808 zemin, #EC7C22 vurgu, Inter yazı tipi) |

## SPTF formatının özeti
- **DESCRIPTOR** — örnekleme hızı, kanal sayısı, veri tipi, kanal adı, birim, ölçek/ofset. Saniyede en az bir kez tekrarlanır.
- **DATA** — 64 bit `first_sample` sayacı + örnekler (interleaved veya planar).
- **MARKER** — debug olay işareti (kod + metin).
- 17 veri tipi: int8…int64, uint8…uint64, int24, float16/32/64, karmaşık (cint16, cint32, cfloat32, cfloat64).
- İsteğe bağlı CRC-32, little-endian, 32 baytlık ortak başlık.

## Mimari (PC tarafı)

```
 Kaynaklar                     Arayüz (PyQt6)
 Cihaz      ──UDP─┐
 Simülatör  ──UDP─┼─► Alıcı thread ─► Akış tamponları ─► DSP zinciri ─► Grafik + ölçüm
 .sptr dosya ─────┘   (çözme, kayıp)   (ring buffer)      (scipy)        (pyqtgraph)
                           └─► Kaydedici (.sptr)
```

- Alıcı, arayüzden ayrı bir **süreçte** çalışır ve örnekleri paylaşımlı belleğe yazar (Aşama 1'de gigabit
  hedefi için thread yerine süreç seçildi); arayüz saniyede ~30 kez yenilenir.
- Tamponlar ham veri tipini tutar; fiziksel birime çevirme gösterim anında yapılır.
- Canlı filtreler durum (state) taşır; blok sınırlarında süreksizlik olmaz.

## Klasör yapısı

```
run.py                 Başlatıcı (kendini .venv ile yeniden başlatır)
docs/                  Protokol dokümanı ve aşama notları
spt/protocol/          SPTF çözme / oluşturma            (Aşama 1)
spt/sources/           UDP alıcı, kayıt/oynatma, simülatör (Aşama 1)
spt/dsp/               Filtreler, spektral analiz, ölçümler (Aşama 2–4)
spt/gui/               Arayüz: tema, komutlar, paneller, görünümler
```

## Aşama haritası

| Aşama | Başlık | Durum |
|---|---|---|
| 0 | Gereksinimler ve mimari | ✅ Tamamlandı |
| 1 | Altyapı ve canlı veri akışı | ✅ Tamamlandı |
| 2 | Spektral analiz ve ölçüm araçları | ⏳ Bekliyor |
| 3 | Filtre tasarımı ve işleme | ⏳ Bekliyor |
| 4 | İleri analiz ve veri içe/dışa aktarma | ⏳ Bekliyor |
| 5 | Genişletmeler ve dağıtım | ⏳ Bekliyor |
