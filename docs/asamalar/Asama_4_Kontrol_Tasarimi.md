# Aşama 4 — Kontrol Tasarımı, Sistem Tanımlama ve İHA Otopilotu

> Durum: **tamamlandı** (ilk sürüm). Mimari, ortak kurallar ve doğrulama ilkesi aşağıda; sonuçlar en sonda.

## Amaç
Kullanıcı sisteminin giriş / çıkış verisini (canlı akış, kayıt, Çalışma Alanı anlık görüntüsü, CSV / MAT) verir;
arayüz MATLAB'deki iş akışının karşılığını sunar:

1. **Sistem tanımlama** (System Identification Toolbox): doğrusal (ARX, ARMAX, OE, BJ, tfest, procest, n4sid / ssest,
   impulseest, spa / etfe) ve doğrusal olmayan (nlarx, Hammerstein-Wiener) modeller; compare (fit %), resid, predict.
2. **Analiz** (Control System Toolbox): açık / kapalı çevrim; adım, darbe, başlangıç koşulu, lsim; Bode, Nyquist,
   Nichols, tekil değerler, kök yer eğrisi, kutup-sıfır; kazanç / faz payları, bant genişliği, stepinfo, damp;
   kontrol edilebilirlik / gözlenebilirlik, Gramyenler, normlar. **Sürekli** ve **ayrık** zamanda (c2d / d2c).
3. **Tasarım**: klasik (PID — paralel / standart, türev filtresi, ayrıklaştırma yöntemleri, pidtune; öncül / ardıl
   (lead / lag), çentik, kök yer eğrisi ve Bode ile etkileşimli) ve modern (kutup yerleştirme, LQR / LQI, Kalman,
   LQG, gözlemci), **MPC** (MPC Toolbox anlamlarıyla; kısıtlı, ayrık).
4. **Blok diyagram** (Simulink benzeri): sürükle-bırak bloklar (kaynaklar, toplam, kazanç, TF / SS, PID, ileri
   besleme, doyum, ölü bölge, hız sınırlayıcı, gecikme, ZOH, kestiriciler, MPC, İHA modelleri …), çok hızlı
   (sürekli bitki + ayrık kontrolcü) benzetim, gerçek veriyle benzetim, çalışma noktasında doğrusallaştırma.
5. **İHA / drone otopilotu**: 6 serbestlik dereceli dörtrotor (ve sabit kanat) modelleri, karıştırıcı (mixer),
   kademeli (açı → açısal hız, irtifa, konum) otopilot şablonları; LQR / MPC ile tasarım.
6. **C kodu** (mevcut MCU kod üretim altyapısıyla): ayrık PID (anti-windup), durum geri beslemesi + gözlemci, Kalman,
   MPC (QP çözücüsüyle); ölçülen bellek ve işlem maliyeti.

Ayrık zaman (sayısal kontrol) birinci sınıf vatandaştır; her model ve tasarım sürekli zamanda da analiz edilebilir.

## Doğrulama ilkesi: MATLAB kahini
Bu makinede MATLAB R2025b (Control System, System Identification, MPC, Robust Control, Simulink, Simulink Control
Design, UAV, Aerospace araç kutuları) kurulu. **Her deterministik hesap MATLAB'in kendisiyle karşılaştırılır.**

- `tools/matlab_oracle.py`: MATLAB betiğini toplu kipte (`-batch`) çalıştırır, `out` struct'ını JSON olarak
  `tests/data/matlab/<ad>.json` dosyasına yazar (betik özetiyle). Testler bu dosyaları okur: **testler MATLAB'siz
  çalışır**; betik değişirse MATLAB varken yeniden üretilir. MATLAB çağrıları kilitle sıraya konur.
- Karmaşık sayılar: betikte `spt_c(x)` → [gerçel, sanal] sütunları. N-boyutlu dizi JSON'da `[i][j][k]` (MATLAB
  indisleriyle aynı) iç içe listedir; sütun vektörü düz listedir.
- Toleranslar: deterministik ve kapalı biçimli hesaplar (c2d, margin, lqr, place, step, bode, arx …) **göreli 1e-8**
  (koşul sayısı kötü durumlarda gerekçeli olarak gevşetilir); yinelemeli kestirimler (armax / oe / bj / tfest /
  ssest, MPC QP) aynı tanım ve optimumda **1e-5–1e-4**; MATLAB'in algoritması yayımlanmamış olanlarda (pidtune,
  nlarx eğitimi, n4sid ağırlık seçimi) birebirlik değil **tanımın sağlandığı** (ör. hedef faz payı / geçiş frekansı
  tam tutturuldu, fit % MATLAB'inkinden kötü değil) test edilir ve bu fark belgede açıkça yazılır.
- Her modül, MATLAB belgesindeki tanımı (ör. stepinfo'nun SettlingTime tanımı, margin'in ayrık sistemde frekans
  aralığı, c2d 'matched' kuralları) **MATLAB'den ölçerek** doğrular; bellekten tanım varsayılmaz.

## Mimari
```
spt/control/                 Qt'siz hesap kütüphanesi (numpy / scipy)
  lti.py                     LTI tabanı; TransferFunction (tf), ZPK (zpk), StateSpace (ss); gecikmeler; aritmetik
  convert.py                 tf ↔ ss ↔ zpk, minreal, balreal, balred, modred, ss2ss
  connect.py                 series, parallel, feedback, append, connect (adlarla), loopsens
  discretize.py              c2d, d2c, d2d ('zoh','foh','tustin'+prewarp,'impulse','matched'), pade, absorb_delay
  timeresp.py                step, impulse, initial, lsim, stepinfo, lsiminfo
  freqresp.py                freqresp, evalfr, bode, nyquist, nichols, sigma, margin, allmargin, bandwidth, dcgain,
                             norm (H2 / H∞), getPeakGain
  analysis.py                pole, zero / tzero, damp, pzmap, rlocus, ctrb, obsv, gram, covar, isstable, dsort / esort
  classical.py               PID nesneleri (pid / pidstd, Ts ve IFormula / DFormula), pidtune, lead / lag / notch
  modern.py                  place, acker, lqr, dlqr, lqrd, lqi, lqry, kalman, kalmd, dlqe, lqg, reg, estim, care, dare
  mpc/                       qp.py (Goldfarb–Idnani çift aktif küme = MATLAB KWIK'in temeli), mpc.py (mpc nesnesi,
                             mpcmove, sim, varsayılan kestirici / bozucu modelleri)
  sysid/                     iddata, ön işleme, arx / armax / oe / bj / tfest / procest / n4sid / ssest /
                             impulseest / spa / etfe, compare / resid / predict / sim, nlarx, nlhw, sıra seçimi
  diagram/                   blok kütüphanesi, model grafiği, benzetim motoru, doğrusallaştırma, .sptd (JSON) dosyası
  uav/                       dörtrotor 6-DOF, sabit kanat, motor / pervane, karıştırıcı, trim, otopilot şablonları
spt/codegen/control_c.py     kontrolcü C kodu (PID, durum geri beslemesi + gözlemci, Kalman, MPC)
spt/gui/views/control_view.py   "Kontrol Tasarımı" görünümü (sekmeler: Modeller · Sistem Tanımlama · Analiz ·
spt/gui/control/                  Tasarım · Blok Diyagram); blok diyagram düzenleyicisi
tests/test_control_*.py      MATLAB referanslı testler (tests/data/matlab/*.json)
```

## Ortak kurallar (sözleşme) — tüm alt modüller uyar
**Örnekleme zamanı `Ts`:** `0.0` = sürekli, `> 0` = ayrık (saniye). (MATLAB'in −1 "belirtilmemiş" Ts'i desteklenmez.)
Farklı Ts'li modeller birleştirilemez (MATLAB gibi hata).

**Polinomlar:** azalan kuvvetler, 1-B `float64` dizi (MATLAB `tf`): `[1, 2, 1]` = s² + 2s + 1 (ayrıkta z). Ayrık
TF'de değişken varsayılan `z`; `z^-1` gösterimi yalnız görüntüleme seçeneğidir.

**Sınıflar** (`spt/control/lti.py`), hepsi `LTI` tabanından:
- ortak: `Ts`, `input_names`, `output_names` (tuple[str]), `name`, `input_delay` (nu,), `output_delay` (ny,) —
  sürekli zamanda saniye, ayrıkta **tamsayı örnek**; `ninputs`, `noutputs`, `shape = (ny, nu)`, `is_discrete`.
- `TransferFunction(num, den, Ts=0.0, ...)`: `num[i][j]`, `den[i][j]` (ny × nu liste, her biri 1-B dizi) ve
  `io_delay` (ny, nu). SISO kısayolu: `tf([1], [1, 2, 1])`, `tf('s')`, `tf('z', Ts)`.
- `ZPK(z, p, k, Ts=0.0)`: `z[i][j]`, `p[i][j]` (karmaşık 1-B), `k` (ny, nu).
- `StateSpace(A, B, C, D, Ts=0.0, state_names=...)`: yoğun `float64` matrisler. Tanımlayıcı (E) ve iç gecikme
  (internal delay) yok; kapalı çevrimde gecikme gerekiyorsa ayrıkta durum olarak (tam), sürekli zamanda Padé ile
  (uyarıyla) ya da benzetimde tam gecikme tamponuyla.
- Aritmetik: `+ - * /` (sağdan / soldan sayı ile), `**` (tamsayı), `sys[i, j]` (alt sistem), `-sys`. Tür önceliği
  MATLAB gibi **ss > zpk > tf**. Sonuç Ts'si ortak Ts.
- Fabrika fonksiyonları MATLAB adlarıyla: `tf`, `zpk`, `ss`, `pid`, `pidstd`; dönüştürme: `tf(sys)`, `ss(sys)`,
  `zpk(sys)` (MATLAB gibi aynı fonksiyon dönüştürür).

**Fonksiyon dönüş biçimleri MATLAB'le aynı eksen düzeninde** (karşılaştırma kolaylığı):
- `step / impulse / initial / lsim`: `y` şekli **(nt, ny, nu)** (initial / lsim: (nt, ny)), `t` (nt,), isteğe bağlı
  `x` (nt, nx, nu).
- `bode`: `mag` (mutlak, dB değil), `phase` (derece), şekil **(ny, nu, nw)**; `w` rad/s. `freqresp`: karmaşık
  (ny, nu, nw). `nyquist`: (re, im) aynı şekil. `sigma`: (min(ny,nu), nw).
- `margin(sys)` → `(Gm, Pm, Wcg, Wcp)` (Gm mutlak, Pm derece; payı olmayan için `inf`), `allmargin` sözlüklerle.
- `stepinfo` → sözlük (`RiseTime`, `TransientTime`, `SettlingTime`, `SettlingMin`, `SettlingMax`, `Overshoot`,
  `Undershoot`, `Peak`, `PeakTime`) MATLAB R2025b tanımıyla (eşikler parametre: `SettlingTimeThreshold=0.02`,
  `RiseTimeLimits=(0.1, 0.9)`).
- `damp` → `(wn, zeta, p)`; ayrık sistemde MATLAB gibi `s = log(z)/Ts` üzerinden.
- Varsayılan zaman / frekans ızgaraları bizim seçimimizdir (MATLAB'inki yayımlanmamış); karşılaştırma testleri açık
  ızgara verir.

**Adlar:** Python API'si MATLAB fonksiyon adlarını korur (`c2d`, `lqr`, `pidtune`, `n4sid` …); seçenekler anahtar
sözcükle (`c2d(sys, Ts, method="tustin", prewarp=w)`). Hata iletileri Türkçe ve kullanıcının diliyle
(`ControlError(ValueError)`); iç API adı iletiye yazılmaz.

**Sayısal kurallar:** scipy'nin sağlam rutinleri (`expm`, `solve_continuous_are`, `solve_discrete_are`, `schur`,
`eig`, `place_poles`, `cont2discrete`) tercih edilir; sıfır / kutup için genelleştirilmiş özdeğer (Rosenbrock);
frekans cevabı durum uzayında Hessenberg biçimiyle; polinom kökleri yalnız düşük derecede. Koşulu kötü işlemler
(yüksek dereceli tf ↔ ss) uyarı verir. Her fonksiyonun belgesinde MATLAB karşılığı ve bilinen farkı yazılır.

**Veri kaynakları:** canlı akış / kayıt / Çalışma Alanı anlık görüntüsü (mevcut `handle` arayüzü: `desc`, `ring`,
`fs`), CSV / TXT, MATLAB `.mat` (v5 / v7; `scipy.io.loadmat`). v7.3 (HDF5) için kullanıcıdan v7 kaydetmesi istenir.

**Koordinatlar (İHA):** havacılık standardı **NED** (x kuzey, y doğu, z aşağı), gövde ekseni x ileri, y sağ, z aşağı;
Euler ZYX (yaw-pitch-roll). Arayüzdeki 3B görünüm (z yukarı) için dönüşüm görüntülemede yapılır.

**Dosyalar:** model çalışma alanı `.sptm` (JSON), blok diyagram `.sptd` (JSON); sürüm alanı taşır.

## Arayüz tasarımı ("Kontrol Tasarımı" görünümü)
Üst gezinmede yeni görünüm (`view.control`). Görünüm içinde beş sekme; her sekme diğer görünümlerin bileşenlerini
kullanır (`PageHeader`, `Card`, `CollapsibleSection`, `make_plot`, `Crosshair`, `DataTips`, `PlotAnalysis`, ayrık
pencereye alma, ölçülmüş renk paleti — aynı görünümde **hiçbir renk iki seride kullanılmaz**, her grafik ayrı kart).

1. **Modeller** (MATLAB çalışma alanının karşılığı): model listesi (ad, tür tf / zpk / ss / pid / tanımlanmış /
   mpc, Ts, giriş-çıkış, derece, kararlılık rozeti); yeni model ifadeyle (`1/(s^2+2*s+1)`, `ss(...)`, `tf(...)`),
   `.sptm` içe / dışa aktarma; c2d / d2c, minreal, balred; seçili modelin MATLAB biçimli gösterimi, kutup / sıfır /
   damp tablosu ve hızlı grafikler.
2. **Sistem Tanımlama** (System Identification uygulaması): veri (akış / kayıt / anlık görüntü / CSV / MAT), giriş /
   çıkış kanalları, Ts, zaman aralığı (grafikte seçilir), ön işleme (detrend, filtre, yeniden örnekleme, tahmin /
   doğrulama bölmesi), yöntem ve dereceler, sonuç: veri grafiği, compare (fit %), artık analizi, frekans cevabı
   karşılaştırması, fit tablosu → "Modellere ekle".
3. **Analiz** (Linear System Analyzer): bitki G, kontrolcü C, algılayıcı H, ön filtre F seçimi; açık çevrim L, kapalı
   çevrim T, S, KS, GS; sürekli / ayrık görünüm (c2d yöntemi ve Ts); seçilebilir grafik ızgarası (adım, darbe, Bode,
   Nyquist, Nichols, kök yer eğrisi, kutup-sıfır, tekil değer, lsim — gerçek veriyle); karakteristik işaretleri (tepe,
   yerleşme, yükselme, kalıcı durum), pay ve stepinfo tabloları; birden çok sistem üst üste (ayrı renkler).
4. **Kontrolcü Tasarımı** (Control System Designer + PID Tuner): mimari seçimi (MATLAB'in standart yapıları: tek
   çevrim, ön filtreli, ileri beslemeli, kademeli iç / dış çevrim); kompansatör düzenleyici (kazanç, kutup, sıfır,
   integratör, lead / lag / çentik) — etkileşimli kök yer eğrisi (kapalı çevrim kutbunu sürükle → kazanç) ve açık
   çevrim Bode düzenleyicisi; PID ayarlayıcı (tür, biçim, ayrıklaştırma, "cevap süresi / dayanıklılık" kaydırıcıları →
   hedef geçiş frekansı ve faz payı); modern: kutup yerleştirme, LQR / LQI (Q, R; Bryson kuralı yardımcısı), Kalman /
   LQG, gözlemci; MPC (ufuklar, ağırlıklar, kısıtlar, referansla benzetim); her değişiklikte kapalı çevrim cevabı ve
   paylar canlı güncellenir; "C Kodu Üret…" (MCU özellikleriyle).
5. **Blok Diyagram** (Simulink): kategori paleti (arama), ızgaralı tuval (yakınlaştır / kaydır, paletten sürükle,
   portları dik açılı tellerle bağla, seç / taşı / sil / kopyala / geri al), blok parametre paneli, araç çubuğu (yeni /
   aç / kaydet `.sptd`, **otopilot şablonları**, çalıştır / durdur, çözücü ve bitiş zamanı), kapsam (scope)
   sonuç grafikleri (veri ipuçlarıyla), "Doğrusallaştır" (seçili G/Ç noktaları → Modeller / Analiz), Veri kaynağı
   bloğunun akış / kayıt / dosyaya bağlanması.

## Sınırlar (dürüstçe)
- MATLAB'in yayımlanmamış algoritmaları (pidtune'un ayarlama yöntemi, ssest başlangıcı, nlarx ağ eğitimi, otomatik
  ızgaralar) birebir kopyalanamaz: aynı **tanım ve hedef** uygulanır, sonuç ölçütlerle karşılaştırılır.
- Simulink'in tamamı değil, kontrol tasarımı için gereken blok kümesi; değişken adımlı çözücü olarak Dormand–Prince
  (ode45) ve sabit adımlı ode1 / ode4.

## Sonuçlar (MATLAB R2025b ile karşılaştırma)
| Modül | Dosyalar | Test | MATLAB ile en büyük fark |
|---|---|---|---|
| LTI çekirdeği ve analiz | `spt/control/{lti,convert,connect,discretize,timeresp,freqresp,analysis,parse}.py` | 360+ | çoğunda 1e-13–1e-16; margin / stepinfo(sys) bizde kesin, MATLAB ızgaralı (ör. 2/(s−1) için Wcp = √3) |
| Tasarım | `classical.py`, `modern.py` | 25 | lqr / dlqr / lqrd / lqi / kalman / lqg ~1e-8 altı; tf(pid) birebir; pidtune hedef PM / ωc tam |
| MPC | `spt/control/mpc/` | 98 (C dahil) | QP ≤2e-14, sim u / y ≤3e-13 (MATLAB'in kendi .m kodundan tanımlar) |
| Sistem tanımlama | `spt/control/sysid/` | 69 | ARX / compare / spa 1e-10; armax / oe / bj ≤2e-6; tfest / procest 1e-4; n4sid %5 (MATLAB ayrıntısı yayımlanmamış) |
| Blok diyagram motoru | `spt/control/diagram/` | 95 | Simulink sabit adım ≤1.1e-14, ode45 ≤1.3e-10, linearize 1e-8 |
| İHA modelleri + otopilot | `spt/control/uav/` | 92 | ode45, sembolik Jacobian, trim ≤1e-11; 5 otopilot şablonu kararlı |
| Kontrolcü C kodu | `spt/codegen/control_c.py`, `mpc_c.py` | 52 + 38 | PID (Simulink anti-windup dahil) / Kalman / LQG 1e-13–1e-16; tüm Cortex-M × gcc / clang -Werror |
| Arayüz | `spt/gui/views/control_view.py`, `spt/gui/control/*` | 13 + 14 + 14 + 20 | grafik verileri kütüphaneyle birebir |

Bağımsız denetimde (MPC + C kodu) bulunan 12 bulgunun 10'u düzeltildi (MaxIterations sınırı MATLAB kuralına
getirildi, NaN ölçüm Türkçe hata, float sabit aralığı, Ts bütçesi …), biri kısmen (float'ta kötü koşullu Hessian
yalnız uyarı — double önerilir), biri bilinen sınır olarak yazıldı. Arayüz ajanlarının bildirdiği iki kütüphane hatası
düzeltildi: ayrık çevrimde ωTs küçükken `margin`'in geçişi kaçırması (yoğun tarama + Brent yedeği; 48 durumda 0 kaçırma)
ve son değeri 0 olan cevapta `stepinfo`'nun sıfır döndürmesi (MATLAB anlamları: aşım ∞, yerleşme NaN).

Son kısa denetim turu (5 alan, 29 bulgu; hepsi yeniden üretilip düzeltildi, regresyon testi eklendi):
LTI (DelayMargin kuralı, sahte geçiş eleme, eşit tepe / ayrık SettlingMin, ifadede karmaşık argüman reddi);
blok diyagram (Mux / Demux sanal yönlendirme, sabit adımda akışlı çalıştırma + anında iptal, yinelenen sinyal adı);
sistem tanımlama (FOH ssest, sabit girişte spa, kısa veri, yetersiz uyarım uyarısı); C kodu (SISO SS iyi koşullu
ikinci derece bölümlerden, float32 katsayı kararlılık denetimi, zpk doğrudan gerçekleme, Kalman `My`);
tasarım (kararsız bitkide pidtune P / I işareti, `PhaseMargin` tüm geçişlerin en küçüğü, ayrık PI pencere taşması,
`kalmd` H ≠ 0 reddi, `lqg` current + Nn ≠ 0 MATLAB `lqg` ile 1e-14, `place` kontrol edilemeyen kip / bağımlı B).

Bilinen sınırlar: iç (internal) gecikme yok (sürekli kapalı çevrimde Padé önerilir); MPC'de gecikmeli bitki, karışık
kısıt, uyarlamalı / doğrusal olmayan MPC yok; pidtune / ssest başlangıcı / nlarx eğitimi MATLAB'le birebir değil
(aynı tanım ve hedef); çözümü olmayan gecikmeli kararsız bitkide (ör. e^{−0.3s}/(s−0.5), P tipi) otomatik ωc
araması tüm adayları taradığı için ~1 dk sürebilir (arka planda çalışır); bağlantı (`connect` / `lft` / `loopsens`)
çevrim dışı sürekli gecikmeyi taşımaz; varsayılan dörtrotor parametreleri yayımlanmış bir F450 örneğidir — ölçülmüş değerlerinizi
kullanın.
