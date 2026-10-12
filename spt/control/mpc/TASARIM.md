# MPC ve QP çözücüsü — tasarım notları

Bu paket MATLAB Model Predictive Control Toolbox'ın (R2025b) doğrusal `mpc` nesnesini aynı matematikle uygular.
Tanımlar MATLAB'in kurulu kaynak dosyalarından (`toolbox/mpc/mpcutils/mpc_buildmat.m`, `mpc_constraintcoef.m`,
`mpc_calculatehessian.m`, `@mpcdata/update.m`, `mpc_chkmodel.m`, `mpc_augmentID/OD.m`, `mpc_chkmnmodel.m`,
`mpc_estimator.m`, `mpcblock_optimizer.m`, `mpcloop_engine.m`, `@mpc/ss.m`) okunarak alınmış, sonra MATLAB'in
`getCodeGenerationData` çıktısıyla **matris matris** doğrulanmıştır (bkz. "Doğrulama"). QP çözücüsü KWIK p-kod
olduğundan Goldfarb–Idnani yöntemiyle bağımsız yazılmıştır (kesin konveks QP'nin çözümü tektir).

Notasyon: p tahmin ufku, m kontrol ufku (blok vektörü `moves`, Σ moves = p), n_mv / n_md / n_ud giriş sayıları,
n_y = n_mo + n_uo çıkış sayıları, n_v = n_md + 1 (son eleman DX ofseti için sabit 1). İndisler 0 tabanlıdır.

## 1. İç model

### 1.1 Bitki hazırlığı (`_prep_plant` = `mpc_chkmodel`)
1. Sinyal türleri: `setmpcsignals(plant, MV=, MD=, UD=, MO=, UO=)`; belirtilmeyen girişler MV, çıkışlar MO.
   MV'den çıkışa doğrudan geçiş (D[:, MV] ≠ 0) hatadır.
2. Nominal noktada doğrusallaştırma: `Model.Nominal = {X, U, Y, DX}`. Sapma modeli
   x̃⁺ = A x̃ + B ũ + DX (sürekli: ẋ̃ = A x̃ + B ũ + DX). DX, B'ye **ek bir giriş sütunu** olarak eklenir
   (girişi her zaman 1 olan "ofset" ölçülen bozucusu).
3. Sürekli bitki: artırılmış (A, [B DX]) çifti denetleyici Ts'iyle ZOH: `expm([[A, B̄],[0, 0]]·Ts)`.
   (Ayrık bitkinin Ts'i farklıysa hata; MATLAB d2d yapar.)
4. Boyutsuzlaştırma (ölçek çarpanları `ScaleFactor`): B ← B·diag([S_u; 1]), C ← diag(1/S_y)·C,
   D ← diag(1/S_y)·D·diag([S_u; 1]). Durumlar ölçeklenmez. Ofsetler: u_off = U/S_u, y_off = Y/S_y, x_off = X.

### 1.2 Bozucu ve gürültü modelleri (artırılmış model)
**Giriş bozucusu (UD)** — `mpc_augmentID`: varsayılan modelde UD kanalları sırayla denenir; kanal j için
A₁ = [[A, b_j],[0, 1]], C₁ = [C, d_j] ile z = 1'de PBH testi `rank([I − A₁; C₁(MO,:)]) = n+1` geçerse
**Ts kazançlı tümlev** (x⁺ = x + Ts·w, d = x), geçmezse birim statik kazanç (d = w). Özel model (Model.Disturbance,
birim beyaz gürültüyle sürülen LTI) boyutsuzlaştırılır (C, D satırları 1/S_ud). Artırma:
A ← [[A, B_d C̄],[0, Ā]], C ← [C, D_d C̄]; UD sütunları gürültü girişleriyle değişir (B_d D̄; B̄).
Giriş sırası: [MV | MD | ofset | ID gürültüleri].

**Çıkış bozucusu** — `mpc_augmentOD`: ölçülen çıkışlar, çıkış ağırlıklarının ufuk boyunca toplamına göre
**büyükten küçüğe** (eşitlikte indis sırası) denenir; her biri için tümlev durumu (C(ct, yeni) = 1) eklenince z = 1
PBH testi geçerse eklenir. Girişi **Ts kazançlı** beyaz gürültü: B ← [[B, 0],[0, Ts·I]]. `setoutdist(obj, 'model',
sys)` özel model (ny çıkışlı; DC kazancı 0 ise hiç eklenmez). Örnek: çift tümlev bitkide (drone ekseni) çıkış
tümlevi gözlenemez olur → eklenmez (bitkinin kendi tümlevleri ofsetsiz izlemeyi sağlar).

**Ölçüm gürültüsü** — `mpc_chkmnmodel`: varsayılan her MO'da birim varyanslı beyaz gürültü (boyutsuz D = I).
Özel model (Model.Noise) C, D satırları 1/S_y. min eig(D Dᵀ) < 1e-6 ise D ← D + 1e-3·I (MATLAB kuralı). Gürültü
durumları artırılmış modelin sonuna eklenir ve **yalnız kestiricide** kullanılır; QP yalnız ilk n_xQP = n_xp +
n_xdist durumu görür.

### 1.3 Kestirici (`_kalman` = `mpc_estimator` + `kalman`)
Bilinmeyen (stokastik) girişler: artırılmış modelin **ofset hariç tüm** girişleri — MV'ler, MD'ler, bozucu ve gürültü
beyaz gürültüleri, hepsi birim varyanslı. G = B(:, bilinmeyen), H = D(MO, bilinmeyen):
Q = GGᵀ, R = HHᵀ, N = GHᵀ. P, ayrık filtre DARE'sinin çözümü:

    P = A P Aᵀ − (A P Cmᵀ + N)(Cm P Cmᵀ + R)⁻¹(A P Cmᵀ + N)ᵀ + Q
    M = P Cmᵀ (Cm P Cmᵀ + R)⁻¹            (ölçüm güncellemesi)
    L = (A P Cmᵀ + N)(Cm P Cmᵀ + R)⁻¹      (zaman güncellemesi; N ≠ 0 iken L ≠ A M)

Çevrimiçi: e = y_m − Cm x(k|k−1) − Dvm v(k); x(k|k) = x(k|k−1) + M e;
x(k+1|k) = A x(k|k−1) + Bu u(k) + Bv v(k) + L e. `getEstimator` L, M'yi mühendislik birimine çevirir (sütun j / S_y).

## 2. QP formülasyonu (yoğun, durum eleme ile)

### 2.1 Tahmin modeli (`_prediction` = `mpc_constraintcoef`)
Artırılmış model (n_xQP durum; Bu, Bv = [B_md, B_ofset], C, Dv = [D_md, 0]) ile, Δu = [Δu(k); …; Δu(k+p−1)],
v = [v(k); …; v(k+p)] (her v(i) sonunda 1), x = x(k|k), u₋₁ = u(k−1):

    y(k+i) = C Aⁱ x + Σ_{j<i} C A^{i−1−j} Bu u(k+j) + Σ_{j<i} C A^{i−1−j} Bv v(k+j) + Dv v(k+i),   i = 1..p
    u(k+j) = u₋₁ + Σ_{l≤j} Δu(k+l)

ile Y = Sx·x + Su1·u₋₁ + Su·ΔU + Hv·V. Sx satır blokları C Aⁱ; Su1 = Σ_{j<i} C Aʲ Bu; Su alt üçgen Toeplitz
(blok (i, j) = Σ C A^l Bu); Hv blok (i, j<i) = C A^{i−1−j} Bv, (i, i) = Dv.

**Blok hamleler:** ΔU = Jm·z (z: n_blok·n_mv serbest hamle). Jm, her bloğun ilk adımına birim matris koyar
(m tamsayıysa bloklar [1, …, 1, p − m + 1]). U = I1·u₋₁ + I2Jm·z, I1 = 1_p ⊗ I, I2Jm = (tril(1_{p×p}) ⊗ I)·Jm.

### 2.2 Maliyet
Ağırlıklar **boyutsuz** sinyallere uygulanır (W = ağırlık²; tekdüze ya da zamanla değişen satırlar):

    J = Σ_{i=1..p} ‖W_y (y(k+i) − r(k+i))‖² + Σ_{i=0..p−1} ‖W_u (u(k+i) − u_hedef)‖²
        + Σ_{i=0..p−1} ‖W_Δu Δu(k+i)‖² + ρ_ε ε²

    J(z) = zᵀ H z + 2 fᵀ z + sabit,    QP: min ½ zᵀ H z + fᵀ z  (= J/2)
    H   = SuJmᵀ Wy SuJm + Jmᵀ WΔu Jm + I2Jmᵀ Wu I2Jm            (+ ρ_ε köşegende, kısıt varsa)
    f   = Kxᵀ x + Krᵀ R + Ku1ᵀ u₋₁ + Kvᵀ V + Kutᵀ U_hedef
    Kx  = Sxᵀ Wy SuJm,  Kr = −Wy SuJm,  Ku1 = Su1ᵀ Wy SuJm + I1ᵀ Wu I2Jm,  Kv = Hvᵀ Wy SuJm,  Kut = −Wu I2Jm

Varsayılanlar (MATLAB): W_y = 1 (n_mv < n_y ise ölçülenler önce olmak üzere ilk n_mv çıkış 1, kalanlar 0),
W_u = 0, W_Δu = 0.1, ρ_ε = Weights.ECR = 1e5·max(tüm ağırlıklar) — **kurulum anında** hesaplanır; ağırlıklar
sonradan değişse de ECR korunur (MATLAB set davranışı). `Info.Cost` = J (yarısı değil), boyutsuz.

### 2.3 Kısıtlar (`mpc_buildmat`) — Ac z ≥ b
Tam satır listesi (sıra korunur, `Mrows` silinmeden önceki indisleri tutar):

    [ y ≤ ymax (p·n_y) ; y ≥ ymin (p·n_y) ; u ≤ umax (p·n_mv) ; u ≥ umin (p·n_mv) ;
      Δu ≤ Δumax (n_serbest) ; Δu ≥ Δumin (n_serbest) ]

- Sınırlar boyutsuz ve ofsetsiz: (Min/ScaleFactor) − ofset; zamanla değişen sınırlar ufka son değer tekrarlanarak
  açılır (p'den uzunsa kesilir).
- u kısıtları yalnız her bloğun ilk adımında tutulur; blok içindeki en sıkı sınırla (umin: max, umax: min) ve en küçük
  ECR ile sıkılaştırılır (u blok boyunca sabit olduğundan kesin). Bloklanmış adımların Δu satırları atılır (Δu = 0).
- ±∞ sınırlı satırlar atılır. Kalan satır kalmazsa problem kısıtsızdır: tek çözüm z = −H⁻¹f (gevşek değişken yok).
- **Gevşek değişken (ECR):** kısıt varsa z'nin sonuna ε eklenir; her satır ε·V ile gevşer (V = MinECR / MaxECR /
  RateMinECR / RateMaxECR; MV'ler varsayılan sert V = 0, çıkışlar yumuşak V = 1). Açık ε ≥ 0 satırı yoktur (ε < 0
  hiçbir zaman maliyeti düşürmez). H(ε, ε) = ρ_ε.

Satır biçimi: Ac·z ≥ −(Mlim + Mx·x + Mu1·u₋₁ + Mv·V), Ac = −[Mu, Me]. Örn. y ≤ ymax satırı:
−SuJm z + V ε ≥ −ymax + Sx x + Su1 u₋₁ + Hv V.

### 2.4 Hessian denetimi (`mpc_checkhessian`)
Cholesky H = L Lᵀ; başarısız ya da min diag(L) ≤ 10√eps ise: ‖H‖∞ ≥ 1e10 → hata (kararsız bitki + uzun ufuk);
değilse köşegene sırayla 10^k·10√eps (k = 0..4, birikimli) eklenip yeniden denenir (uyarı). QP çözücüsü bu
(düzeltilmiş) Linv = L⁻¹ ile çalışır; maliyet hesabı özgün H ile (MATLAB gibi).

## 3. Çevrimiçi hesap (`mpcmove`) — C koduna taşınacak döngü
Her örnekte (boyutsuz, ofsetsiz büyüklüklerle):

1. y = y_m/S_y − y_off; R (p·n_y) = r/S_y − y_off (önizleme yoksa p kez tekrar); V ((p+1)·n_v), son eleman 1.
2. Kestirim: e = y − Cm x̂ − Dvm v(k); x̂_kk = x̂ + M e; x_QP = x̂_kk[:n_xQP].
3. f (n_serbest) = Kxᵀ x_QP + Krᵀ R + Ku1ᵀ u₋₁ + Kvᵀ V + Kutᵀ U_hedef; b = −(Mlim + Mx x_QP + Mu1 u₋₁ + Mv V).
4. QP (qp.py, Linv ile, önceki aktif kümeden sıcak başlangıç). Durum < 0 (olursuz) ya da 0 (yineleme sınırı;
   UseSuboptimalSolution kapalı) ise z = 0 → **MV önceki değerinde tutulur**.
5. u = u₋₁ + z[0:n_mv] + u_off; mühendislik birimi u·S_u.
6. x̂⁺ = A x̂ + Bu (u − u_off) + Bv v(k) + L e; u₋₁ ← u − u_off.

MCU için sadeleştirme (önizlemesiz kullanımda kesin eşdeğer): R, V ve U_hedef ufuk boyunca sabitse
Krᵀ R = (Σ_i Kr_iᵀ) r, Kvᵀ V = (Σ_i Kv_iᵀ) v, Kutᵀ U_hedef sabit vektör, Mv·V = (Σ_i Mv_i) v önceden toplanır.
Bellek (double): H/Linv n_z², Ac n_c·n_z, Kx n_xQP·n_s, Kr_sum n_y·n_s, Ku1 n_mv·n_s, Kv_sum n_v·n_s, Mx n_c·n_xQP,
Mu1 n_c·n_mv, Mv_sum n_c·n_v, Mlim n_c, kestirici A, Bu, Bv, Cm, Dvm, L, M (n_x² + …); QP çalışma alanı
2n_z² + 5n_z + n_c. İşlem: kestirim O(n_x²), f O(n_s·n_xQP), b O(n_c·n_xQP), QP yineleme başına O(n_z² + n_c·n_z).

`MPCData` (obj.data) tüm bu matrisleri MATLAB `getCodeGenerationData` adlarıyla taşır (H, Hinv, Linv, Kx, Kr,
Ku1, Kv, Kut, Ac, Mlim, Mx, Mu1, Mv, Mrows, A, B, C, D, Bu, Bv, Cm, Dv, Dvm, L, M, Pcov, utarget, Uscale, Yscale,
uoff, yoff, xpoff, …).

## 4. QP çözücüsü (qp.py)
Goldfarb–Idnani çift aktif küme: kısıtsız en iyiden başlar, her adımda en çok ihlal edilen kısıtı ekler, çift
olurluğu korumak için gerekirse kısıt çıkarır. J = L⁻ᵀQ ve R (L⁻¹N = Q[R; 0]) Givens yansımalarıyla güncellenir
(yeniden çarpanlara ayırma yok; O(n²) ekleme / çıkarma). Ayrıntı ve C bellek düzeni qp.py modül belgesinde.
Durum kodları MATLAB `mpcActiveSetSolver` ile aynı anlamda (> 0 yineleme sayısı, 0 sınır, −1 olursuz, −2 sayısal).

## 5. Benzetim, eşdeğer kontrolcü, cloffset
- `sim` = `mpcloop_engine`: her adımda ölçüm (ofset farkları ve çıkış gürültüsü dahil), kestirim, QP, giriş
  gürültüsü eklenmiş u bitkiye. Model uyumsuzluğu: `mpcsimopt(Model=gerçek_bitki)` (aynı MV / MD / MO düzeni,
  UD sayısı farklı olabilir; nominal belirtilmezse denetleyicininki).
- `ss(obj, 'rvot')` = kısıtsız denetleyicinin doğrusal eşdeğeri: z = −[I 0]·H⁻¹·f'den
  u = Fx x̂ + (I + Fu) u₋₁ + Dym y_m + Dr r + …, durumlar [x̂(k|k−1); u(k−1)].
- `cloffset`: kısıtsız kapalı çevrimin (iç model + ss kontrolcüsü) MO'ya eklenen birim basamak bozucudan MO'ya DC
  kazancı; z = 1'deki yapısal olarak kontrol edilemeyen bozucu tümlevleri atılarak (MATLAB `dcgain` + sminreal).

## 6. Doğrulama (MATLAB R2025b, `tests/data/matlab/mpc_*.json`)
- **QP:** 16 problem (rastgele 5–30 değişken / 11–120 kısıt, eşitlikli, dejenere köşe, yinelenen satır, hepsi
  pasif, kötü koşullu (cond 1e6) kutu, iki tür çözümsüz, zayıf aktif): `quadprog` (etkin küme, 1e-12 tolerans) ve
  `mpcActiveSetSolver` ile x farkı ≤ 2e-14 (göreli), aktif küme ve çarpanlar aynı; exitflag 12/16 birebir
  (kalanlarda aynı çözüm, farklı yol); sıcak başlangıç 1 yineleme (MATLAB ile aynı).
- **MPC (8 durum):** SISO sürekli kısıtsız; SISO MV + oran kısıtlı (mpcmove ayrıntıları, çözümsüz durum); MIMO 2×2
  yumuşak OV + ölçek çarpanı + MV hedefi + referans önizlemesi; MD (doğrudan geçişli, N ≠ 0) + nominal (X, U, Y, DX);
  çift tümlev (drone ekseni: UO hız kısıtı, zamanla değişen MV sınırı, blok hamle [2 3 5]); model uyumsuzluğu + UD
  (varsayılan giriş bozucu tümlevi); özel giriş bozucu / ölçüm gürültüsü / çıkış bozucu modelleri + zamanla
  değişen ağırlık ve sınır + çıkış gürültüsü; ölçek çarpanları + MD önizlemesi. İç veri (30 matris) en büyük göreli
  fark 5e-13; `sim` u, y en büyük mutlak fark 3.3e-13; mpcmove Uopt / Yopt / Xopt / Cost / durum ~1e-14,
  Iterations (sıcak başlangıçlı) 8/8 birebir; getEstimator ~1e-15, ss(mpcobj) ≤ 1e-13, cloffset ~1e-15.
- **Yineleme sayıları (ölçüldü):** MATLAB kod üretim yolu kapalı çevrimleri (8 durum) 426/430 adımda birebir
  (farklı 4 adımda çözüm yine aynı, u farkı ≤ 1.3e-12), mpcmove dizisi 8/8.
- **Yineleme sınırı (`mpc_denetim_cases.json`):** MATLAB kuralı ölçüldü — QP adımı (kısıt ekleme / çıkarma,
  sıcak başlangıçtaki negatif çarpanlı çıkarmalar dahil) sayısı ≤ MaxIterations; bildirilen yineleme = 1 + adım
  (en çok MaxIterations + 1). 1 + 4 adımlık adım: MaxIterations 1–3 → 0, 4–6 → 5 (6/6 birebir); drone ekseni
  MaxIterations = 3: u, y ~1e-13.

## 7. Bilinçli farklar ve sınırlar
- Desteklenmeyen (açık ControlError): gecikmeli bitki (MATLAB absorbDelay), ayrık bitkinin farklı Ts'e yeniden
  örneklenmesi (d2d), köşegen dışı (cell) ağırlıklar, karma (custom) kısıtlar, tamsayı / ikili MV'ler, uyarlamalı /
  LTV / açık (explicit) MPC, özel kestirici (`setEstimator 'custom'`), çevrimiçi ağırlık / sınır değişimi
  (mpcmoveopt). Sürekli zamanlı özel bozucu modelleri ZOH ile ayrıklaştırılır.
- Çözümsüz MPC QP'sinde MATLAB KWIK bazen −2 döndürür; bu çözücü −1 (her ikisi de `QPCode = 'infeasible'`, MV
  tutulur). Yineleme sayıları KWIK'inkinden farklı olabilir (seçim kuralı yayımlanmamış).
- **Yineleme sınırına takılan adımlarda MATLAB ile eşleşme mümkün değildir:** sınır aynı kuralla uygulanır ama
  KWIK'in kısıt seçim yolu yayımlanmadığından sınıra takılan adımın hangisi olduğu ve (UseSuboptimalSolution
  açıkken) alt-en iyi nokta yol bağımlıdır; sonrasında MV dizileri ayrışabilir (ör. drone, MaxIterations = 2).
  Sınıra hiç takılmayan tasarımlarda çözüm birebir aynıdır.
- `mpcmove` / `mpcstate` NaN / ±∞ ölçüm, referans, MD ve durum değerlerini MATLAB gibi reddeder (ControlError);
  `sim` sinyallerinde sütun sayısı tam olmalıdır (ny > 1 iken skaler referans hata, MATLAB InvalidSetpointSize).
- Bilinen sınır (C kodu): float'ta QP'nin doğruluğu Hessian koşuluna bağlıdır; `mpc_c.numeric_warnings`
  koşul × FLT_EPSILON > 1e-3 ya da Hessian düzeltmesi (BadH) durumunda uyarır — böyle tasarımlarda double kullanın.
- `cloffset` ölçek çarpanları 1 değilken MATLAB boyutlu / boyutsuz matrisleri karıştırır; burada tutarlı
  mühendislik birimleri kullanılır (ölçek 1 iken birebir).
- `ss(obj, 't')` MV hedefi girişi tek (zamanla değişmeyen) hedef için.
- Bitki LTI sınıfı değilse (tf / zpk) `spt.control.lti.ss` ile çevrilir; MATLAB kendi gerçeklemesini kullanır
  (u, y aynı; durum koordinatları farklı olabilir).
