"""Senaryo 2 — İrtifa: barometre + ivmeölçer (baro-ataletsel füzyon).

Girişler
  baro_p   : barometre basıncı (Pa | hPa | kPa | mbar | bar | psi)          ┐ en az biri; ikisi de eşlenirse
  baro_alt : barometrik irtifa (m | cm | mm | km | ft)                     ┘ örnekte basınç varsa o kullanılır
  acc_z    : dikey özgül kuvvet, yerçekimi DAHİL (m/s² | g | mg): durağanken +g okur. Varsayım: sensör yatay ya da
             değer dünya çerçevesine döndürülmüş (ör. bir AHRS'nin R(q)·a çıktısının z bileşeni). Eğik gövde
             ivmesi doğrudan verilirse cos(eğim) kadar ölçek hatası sapma kestirimine karışır.
  Barometre seyrek olabilir (ör. 200 Hz akışta 25 Hz): aradaki örnekler NaN.

Çıktılar: irtifa h (m), dikey hız v (m/s, yukarı +).

Model (tüm Kalman türevleri, durum x = [h, v, b], b: ivme sapması; T = örnek aralığı)
  a     = f − g − b                       (f: ölçülen özgül kuvvet, kontrol girişi)
  h⁺    = h + v·T + ½·a·T²,   v⁺ = v + a·T,   b⁺ = b + w_b
  Q     = σ_a²·G·Gᵀ + diag(0, 0, σ_b²·T),  G = [T²/2, T, 0]ᵀ   (σ_a: örnek başına ivme gürültüsü)
  Ölçüm : baro irtifa  z = h + n                         (doğrusal)
          basınç       z = p0·(1 − h/44330.77)^5.25588 + n   (ISA barometrik formül, doğrusal değil)
Algoritmalar
  complementary : klasik baro-ataletsel tamamlayıcı filtre, geçiş zaman sabiti τ. e = z_h − ĥ ile
                  ĥ += k1·e·T_b, v̂ += k2·e·T_b, b̂ −= k3·e·T_b (T_b: son baro ölçümünden bu yana geçen süre,
                  τ/3 ile sınırlı). Sapma açık: k = (3/τ, 3/τ², 1/τ³) → üçlü kutup −1/τ; kapalı: (2/τ, 1/τ², 0).
  kf            : doğrusal KF; basınç önce ISA ile irtifaya çevrilir, R = (σ_p / |dp/dh|)².
  ekf           : basınç doğrudan ölçüm, H = dp/dh (ĥ'de), R = σ_p². (Yalnız baro irtifa varsa KF ile aynıdır.)
  ukf           : basınç ölçümü kokusuz dönüşümle (2n + 1 = 7 sigma noktası). Süreç doğrusal olduğundan UT
                  öngörüsü KF öngörüsüyle birebir aynıdır; kapalı biçimde uygulanır.
  pf            : parçacık filtresi (h, v, b parçacıkları), olabilirlik basınçta / irtifada Gauss, Liu–West
                  yumuşatmalı sistematik yeniden örnekleme; sabit tohum (tekrarlanabilir).
Başlatma: ilk geçerli baro ölçümünde h = z_h, v = 0, b = 0; P0 = diag(R_h, (1 m/s)², σ_b0²). Öncesi NaN.
Eksik ivme (NaN / inf): son geçerli değer tutulur ve her eksik örnekte hıza 2 m/s/√s rastgele yürüyüş eklenir
(örnekleme hızından bağımsız; bilinmeyen ivme kalıcı olabilir); filtre durmaz, barometre taşır.
p0 = 0 seçilirse ilk basınç örneği h = 0 kabul edilir (göreli irtifa).
"""
from __future__ import annotations

import math

import numpy as np

from spt.dsp.fusion_scenarios.base import (
    GRAVITY,
    NAN,
    UNITS_ACC,
    UNITS_LENGTH,
    UNITS_PRESSURE,
    Choice,
    Estimator,
    FusionError,
    Input,
    Output,
    Param,
    Scenario,
    Simulation,
    liu_west_resample,
    normalized_weights,
)

ISA_H = 44330.77          # m
ISA_N = 5.25588
P_STD = 101325.0          # Pa, ISA deniz seviyesi
V0_SIGMA = 1.0            # m/s, başlangıç hız belirsizliği
MISS_ACC_DENSITY = 2.0    # m/s/√s, ivme örneği yokken hıza eklenen rastgele yürüyüş (σ_a² += 2²/T)
_BIG = 1e12


def isa_pressure(h, p0: float):
    """ISA barometrik formül: p = p0·(1 − h/44330.77)^5.25588 (dizi ya da skaler)."""
    if isinstance(h, np.ndarray):
        return p0 * np.power(np.maximum(1.0 - h / ISA_H, 1e-9), ISA_N)
    return p0 * max(1.0 - h / ISA_H, 1e-9) ** ISA_N


def isa_altitude(p, p0: float):
    """Ters ISA: h = 44330.77·(1 − (p/p0)^(1/5.25588))."""
    if isinstance(p, np.ndarray):
        return ISA_H * (1.0 - np.power(p / p0, 1.0 / ISA_N))
    return ISA_H * (1.0 - (p / p0) ** (1.0 / ISA_N))


def isa_dpdh(h: float, p0: float) -> float:
    """dp/dh (Pa/m, negatif): deniz seviyesinde ≈ −12.0 Pa/m."""
    return -p0 * ISA_N / ISA_H * max(1.0 - h / ISA_H, 1e-9) ** (ISA_N - 1.0)


INPUTS = (
    Input("baro_p", "Barometre basıncı", ("",), UNITS_PRESSURE, False,
          ("basinc", "pressure", "press", "baro", "barometre", "barometer", "prs"),
          "Statik basınç. p0 ve ISA formülüyle irtifaya çevrilir"),
    Input("baro_alt", "Barometrik irtifa", ("",), UNITS_LENGTH, False,
          ("baro alt", "baro irtifa", "baro yukseklik", "baro h", "irtifa", "altitude", "alt", "yukseklik",
           "height", "altimetre", "altimeter"),
          "Sensörün kendi hesapladığı irtifa (basınç eşlenmemişse kullanılır)"),
    Input("acc_z", "Dikey ivme (yerçekimi dahil)", ("",), UNITS_ACC, True,
          ("acc z", "accel z", "ivme z", "az", "dikey ivme", "ivme dikey", "acc up", "vertical acc",
           "acc vert", "acc world z", "lin acc z"),
          "Dünya çerçevesinde yukarı yönlü özgül kuvvet: durağanken +g (sensör yatay ya da AHRS ile döndürülmüş)"),
)
OUTPUTS = (
    Output("alt", "İrtifa", "m", False, ("irtifa", "alt", "altitude", "yukseklik", "height", "h")),
    Output("vz", "Dikey hız", "m/s", False, ("dikey hiz", "vz", "vertical speed", "vertical velocity",
                                            "climb", "tirmanma", "vspeed", "vel z", "hiz z", "v z")),
)
ONE_OF = (("baro_p", "baro_alt"),)
PARAMS = (
    Param("sigma_p", "σ basınç (Pa)", 4.0, 1e-3, 1e4, 3,
          "Basınç ölçüm gürültüsü (basınç girişi). Deniz seviyesinde 1 m ≈ 12 Pa"),
    Param("sigma_h", "σ baro irtifa (m)", 0.4, 1e-4, 1e3, 4, "Barometrik irtifa girişinin ölçüm gürültüsü"),
    Param("sigma_acc", "σ ivme (m/s²)", 0.1, 1e-5, 100.0, 4,
          "İvmeölçerin örnek başına beyaz gürültüsü (akış hızında RMS)"),
    Param("sigma_bias", "σ ivme sapması (m/s²/√s)", 0.005, 0.0, 10.0, 5, "İvme sapmasının rastgele yürüyüşü"),
    Param("bias0", "Başlangıç sapma σ (m/s²)", 0.3, 1e-5, 50.0, 4, "İvme sapmasının başlangıç belirsizliği"),
    Param("p0", "p0 — referans basınç (Pa)", P_STD, 0.0, 2e5, 1,
          "h = 0'daki basınç (ISA deniz seviyesi 101325 Pa). 0: ilk basınç örneği h = 0 (göreli irtifa)"),
    Param("g", "Yerçekimi g (m/s²)", GRAVITY, 9.7, 9.9, 5, "Yerel yerçekimi; küçük hatasını sapma kestirimi emer"),
)
PF_PARAMS = (
    Param("particles", "Parçacık sayısı", 500, 50, 20_000, 0, "Daha çok parçacık: daha doğru, daha yavaş"),
    Param("seed", "Tohum", 1, 0, 1_000_000, 0, "Rastgele sayı üretecinin tohumu (tekrarlanabilirlik)"),
)


class _AltitudeBase(Estimator):
    scenario_params = PARAMS
    inputs = INPUTS
    outputs = OUTPUTS
    one_of = ONE_OF

    def _setup(self) -> None:
        sv = self.sv
        if 0.0 < sv["p0"] < 1000.0:
            raise FusionError("p0 (referans basınç) 0 (otomatik) ya da ≥ 1000 Pa olmalı")
        self.p0: float | None = sv["p0"] or None
        self.g = sv["g"]
        self.inited = False
        self.f_last: float | None = None
        self.n_baro = 0
        self.k = 0                   # mutlak örnek sayacı
        self._init_state()

    def _init_state(self) -> None:
        pass

    @staticmethod
    def _lists(x):
        p, h = x.get("baro_p"), x.get("baro_alt")
        return (None if p is None else p.tolist(), None if h is None else h.tolist(), x["acc_z"].tolist())

    def _baro(self, pl, hl, k):
        """k. örnekteki baro ölçümü: (True, basınç Pa) | (False, irtifa m) | None. Basınç öncelikli."""
        if pl is not None:
            v = pl[k]
            if 0.0 < v < _BIG:
                if self.p0 is None:
                    self.p0 = v
                return True, v
        if hl is not None:
            v = hl[k]
            if -_BIG < v < _BIG:
                return False, v
        return None

    def _alt_of(self, meas) -> tuple[float, float]:
        """Ölçümün irtifa karşılığı ve varyansı (doğrusallaştırılmış)."""
        is_p, z = meas
        if is_p:
            h = isa_altitude(z, self.p0)
            return h, (self.sv["sigma_p"] / abs(isa_dpdh(h, self.p0))) ** 2
        return z, self.sv["sigma_h"] ** 2

    def _accel(self, f: float) -> tuple[float | None, bool]:
        """Geçerli ivme ya da tutulan son değer (hiç yoksa None → çağıran a = 0 alır); ikinci değer: eksik mi."""
        if -_BIG < f < _BIG:
            self.f_last = f
            return f, False
        return self.f_last, True

    def _diag_common(self) -> dict[str, str]:
        d = {"Baro ölçümü": str(self.n_baro)}
        if self.p0 is not None and "baro_p" in (self._keys or ()):
            d["p0 (Pa)"] = f"{self.p0:.1f}"
        return d


# ============================================================================ tamamlayıcı filtre
class AltitudeComplementary(_AltitudeBase):
    key, name = "complementary", "Tamamlayıcı (baro-ataletsel)"
    params = (
        Param("tau", "Geçiş zaman sabiti τ (s)", 2.0, 0.02, 1000.0, 3,
              "τ'dan hızlı değişimler ivmeölçerden, yavaş değişimler barometreden gelir"),
        Choice("bias", "İvme sapması kestirimi", "on",
               (("on", "Açık (3. derece, k = 3/τ, 3/τ², 1/τ³)"), ("off", "Kapalı (2. derece, k = 2/τ, 1/τ²)")),
               "Açıkken sabit ivme sapması kalıcı irtifa hatası bırakmaz"),
    )

    def _init_state(self) -> None:
        self.h = self.v = self.b = 0.0
        self.k_last = 0

    def _run(self, x, n, dt):
        pl, hl, al = self._lists(x)
        out = np.empty((n, 2))
        tau = self.values["tau"]
        if self.values["bias"] == "on":
            k1, k2, k3 = 3.0 / tau, 3.0 / tau ** 2, 1.0 / tau ** 3
        else:
            k1, k2, k3 = 2.0 / tau, 1.0 / tau ** 2, 0.0
        t_max = tau / 3.0
        g, T = self.g, dt
        T2 = 0.5 * T * T
        h, v, b = self.h, self.v, self.b
        for k in range(n):
            kabs = self.k + k
            f, _ = self._accel(al[k])
            if f is None:
                f = g + b
            meas = self._baro(pl, hl, k)
            if not self.inited:
                if meas is None:
                    out[k] = NAN, NAN
                    continue
                h, v, b = self._alt_of(meas)[0], 0.0, 0.0
                self.inited, self.k_last = True, kabs
                self.n_baro += 1
                out[k] = h, v
                continue
            a = f - g - b
            h += v * T + a * T2
            v += a * T
            if meas is not None:
                e = self._alt_of(meas)[0] - h
                tb = min((kabs - self.k_last) * T, t_max)
                h += k1 * e * tb
                v += k2 * e * tb
                b -= k3 * e * tb
                self.k_last = kabs
                self.n_baro += 1
            out[k] = h, v
        self.k += n
        self.h, self.v, self.b = h, v, b
        return out

    def diagnostics(self):
        d = {"İvme sapması (m/s²)": f"{self.b:+.4f}"}
        d.update(self._diag_common())
        return d


# ============================================================================ Kalman ailesi
class _AltitudeKalman(_AltitudeBase):
    """KF / EKF / UKF ortak öngörü (kapalı biçim, skaler) — alt sınıflar `_update` yazar."""

    def _init_state(self) -> None:
        self.x = (0.0, 0.0, 0.0)
        self.P = (0.0,) * 6        # p00 p01 p02 p11 p12 p22 (simetrik)

    def _update(self, h, v, b, P, meas):  # pragma: no cover - alt sınıflar
        raise NotImplementedError

    @staticmethod
    def _linear(h, v, b, P, z, hh, pred, R):
        """Skaler ölçüm güncellemesi, H = [hh, 0, 0]."""
        p00, p01, p02, p11, p12, p22 = P
        c0, c1, c2 = hh * p00, hh * p01, hh * p02
        s = hh * c0 + R
        k0, k1, k2 = c0 / s, c1 / s, c2 / s
        y = z - pred
        return (h + k0 * y, v + k1 * y, b + k2 * y,
                (p00 - k0 * c0, p01 - k0 * c1, p02 - k0 * c2, p11 - k1 * c1, p12 - k1 * c2, p22 - k2 * c2))

    def _run(self, x, n, dt):
        pl, hl, al = self._lists(x)
        out = np.empty((n, 2))
        sv = self.sv
        g, T = self.g, dt
        TT = T * T
        s_, r_ = -0.5 * TT, -T
        qa_ok, qa_miss = sv["sigma_acc"] ** 2, sv["sigma_acc"] ** 2 + MISS_ACC_DENSITY ** 2 / T
        q22 = sv["sigma_bias"] ** 2 * T
        h, v, b = self.x
        p00, p01, p02, p11, p12, p22 = self.P
        for k in range(n):
            f, missing = self._accel(al[k])
            if f is None:
                f = g + b
            meas = self._baro(pl, hl, k)
            if not self.inited:
                if meas is None:
                    out[k] = NAN, NAN
                    continue
                h, r0 = self._alt_of(meas)
                v = b = 0.0
                p00, p01, p02, p11, p12, p22 = r0, 0.0, 0.0, V0_SIGMA ** 2, 0.0, sv["bias0"] ** 2
                self.inited = True
                self.n_baro += 1
                out[k] = h, v
                continue
            # ---- öngörü: x⁺ = F·x + B·(f − g), P⁺ = F·P·Fᵀ + Q  (F = [[1, T, −T²/2], [0, 1, −T], [0, 0, 1]])
            a = f - g - b
            h += v * T + 0.5 * a * TT
            v += a * T
            qa = qa_miss if missing else qa_ok
            m00 = p00 + T * p01 + s_ * p02
            m01 = p01 + T * p11 + s_ * p12
            m02 = p02 + T * p12 + s_ * p22
            m11 = p11 + r_ * p12
            m12 = p12 + r_ * p22
            p00 = m00 + T * m01 + s_ * m02 + qa * TT * TT * 0.25
            p01 = m01 + r_ * m02 + qa * TT * T * 0.5
            p02 = m02
            p11 = m11 + r_ * m12 + qa * TT
            p12 = m12
            p22 = p22 + q22
            if meas is not None:
                h, v, b, (p00, p01, p02, p11, p12, p22) = self._update(h, v, b, (p00, p01, p02, p11, p12, p22),
                                                                       meas)
                self.n_baro += 1
            out[k] = h, v
        self.k += n
        self.x = (h, v, b)
        self.P = (p00, p01, p02, p11, p12, p22)
        return out

    def diagnostics(self):
        h, v, b = self.x
        P = self.P
        d = {"İvme sapması (m/s²)": f"{b:+.4f}"}
        if self.inited:
            d["σ irtifa (m)"] = f"{math.sqrt(max(P[0], 0.0)):.3f}"
            d["σ dikey hız (m/s)"] = f"{math.sqrt(max(P[3], 0.0)):.3f}"
            d["σ sapma (m/s²)"] = f"{math.sqrt(max(P[5], 0.0)):.4f}"
        d.update(self._diag_common())
        return d


class AltitudeKF(_AltitudeKalman):
    key, name = "kf", "KF (doğrusal)"

    def _update(self, h, v, b, P, meas):
        z, r = self._alt_of(meas)
        return self._linear(h, v, b, P, z, 1.0, h, r)


class AltitudeEKF(_AltitudeKalman):
    key, name = "ekf", "EKF (basınç ölçümlü)"

    def _update(self, h, v, b, P, meas):
        is_p, z = meas
        if not is_p:
            return self._linear(h, v, b, P, z, 1.0, h, self.sv["sigma_h"] ** 2)
        return self._linear(h, v, b, P, z, isa_dpdh(h, self.p0), isa_pressure(h, self.p0),
                            self.sv["sigma_p"] ** 2)


class AltitudeUKF(_AltitudeKalman):
    key, name = "ukf", "UKF (kokusuz)"

    def _update(self, h, v, b, P, meas):
        is_p, z = meas
        if not is_p:  # doğrusal ölçümde UT güncellemesi KF güncellemesinin aynısıdır
            return self._linear(h, v, b, P, z, 1.0, h, self.sv["sigma_h"] ** 2)
        p0 = self.p0
        p00, p01, p02, p11, p12, p22 = P
        # √(n·P), n = 3 (λ = 0): alt üçgen Cholesky
        l00 = math.sqrt(max(3.0 * p00, 1e-300))
        l10, l20 = 3.0 * p01 / l00, 3.0 * p02 / l00
        l11 = math.sqrt(max(3.0 * p11 - l10 * l10, 1e-300))
        l21 = (3.0 * p12 - l20 * l10) / l11
        l22 = math.sqrt(max(3.0 * p22 - l20 * l20 - l21 * l21, 1e-300))
        cols = ((l00, l10, l20), (0.0, l11, l21), (0.0, 0.0, l22))
        devs = [c for c in cols] + [(-c[0], -c[1], -c[2]) for c in cols]     # merkez dışı 6 nokta (x'e göre)
        z0 = isa_pressure(h, p0)
        zs = [isa_pressure(h + d[0], p0) for d in devs]
        zm = sum(zs) / 6.0                                                    # wm = (0, 1/6, …)
        dz0 = z0 - zm
        dzs = [zz - zm for zz in zs]
        pzz = 2.0 * dz0 * dz0 + sum(d * d for d in dzs) / 6.0 + self.sv["sigma_p"] ** 2   # wc = (2, 1/6, …)
        pxz = [sum(d[j] * e for d, e in zip(devs, dzs)) / 6.0 for j in range(3)]
        k0, k1, k2 = pxz[0] / pzz, pxz[1] / pzz, pxz[2] / pzz
        y = z - zm
        return (h + k0 * y, v + k1 * y, b + k2 * y,
                (p00 - k0 * k0 * pzz, p01 - k0 * k1 * pzz, p02 - k0 * k2 * pzz,
                 p11 - k1 * k1 * pzz, p12 - k1 * k2 * pzz, p22 - k2 * k2 * pzz))


# ============================================================================ parçacık filtresi
class AltitudePF(_AltitudeBase):
    key, name = "pf", "Parçacık filtresi"
    params = PF_PARAMS

    def _init_state(self) -> None:
        self.rng = np.random.default_rng(int(self.values["seed"]))
        self.X: np.ndarray | None = None       # (3, N): h, v, b
        self.w: np.ndarray | None = None
        self.ess = 0.0
        self.resamples = 0

    def _run(self, x, n, dt):
        pl, hl, al = self._lists(x)
        out = np.empty((n, 2))
        sv, rng = self.sv, self.rng
        N = int(self.values["particles"])
        g, T = self.g, dt
        T2 = 0.5 * T * T
        sa_ok, sa_miss = sv["sigma_acc"], math.sqrt(sv["sigma_acc"] ** 2 + MISS_ACC_DENSITY ** 2 / T)
        sb = sv["sigma_bias"] * math.sqrt(T)
        X, w = self.X, self.w
        for k in range(n):
            f, missing = self._accel(al[k])
            meas = self._baro(pl, hl, k)
            if X is None:
                if meas is None:
                    out[k] = NAN, NAN
                    continue
                h0, r0 = self._alt_of(meas)
                X = np.empty((3, N))
                X[0] = h0 + math.sqrt(r0) * rng.standard_normal(N)
                X[1] = V0_SIGMA * rng.standard_normal(N)
                X[2] = sv["bias0"] * rng.standard_normal(N)
                w = np.full(N, 1.0 / N)
                self.inited = True
                self.ess = float(N)
                self.n_baro += 1
                out[k] = float(w @ X[0]), float(w @ X[1])
                continue
            e = rng.standard_normal((2, N))
            if f is None:                       # hiç ivme gelmedi: a = 0 (+ gürültü)
                a = sa_miss * e[0]
            else:
                a = (f - g) - X[2] + (sa_miss if missing else sa_ok) * e[0]
            X[0] += X[1] * T + a * T2
            X[1] += a * T
            X[2] += sb * e[1]
            if meas is not None:
                is_p, z = meas
                if is_p:
                    r = (z - isa_pressure(X[0], self.p0)) / sv["sigma_p"]
                else:
                    r = (z - X[0]) / sv["sigma_h"]
                w = normalized_weights(np.log(np.maximum(w, 1e-300)) - 0.5 * r * r)
                self.ess = 1.0 / float(w @ w)
                if self.ess < 0.5 * N:
                    X = liu_west_resample(rng, X, w)
                    w = np.full(N, 1.0 / N)
                    self.resamples += 1
                self.n_baro += 1
            out[k] = float(w @ X[0]), float(w @ X[1])
        self.k += n
        self.X, self.w = X, w
        return out

    def diagnostics(self):
        d = {}
        if self.X is not None:
            d["İvme sapması (m/s²)"] = f"{float(self.w @ self.X[2]):+.4f}"
            d["ESS"] = f"{self.ess:.0f}"
            d["Yeniden örnekleme"] = str(self.resamples)
        d.update(self._diag_common())
        return d


ALGORITHMS: dict[str, type[Estimator]] = {cls.key: cls for cls in (
    AltitudeComplementary, AltitudeKF, AltitudeEKF, AltitudeUKF, AltitudePF)}


def raw(x: dict, n: int, sv: dict, state: dict) -> np.ndarray:
    """Füzyonsuz: irtifa doğrudan barometreden (basınç varsa ISA ile, yoksa baro irtifa girişi), yalnız baro
    örneklerinde (arası NaN). Dikey hız NaN: seyrek ve gürültülü barometrenin sonlu farkı (25 Hz, σ 0.3 m →
    σ_v ≈ 10 m/s) bilgi taşımaz ve parçalar arası durum gerektirir. p0 = 0 (otomatik) ise ilk geçerli basınç
    `state["_p0"]`'a yazılır / oradan okunur (kestiricilerle aynı referans)."""
    out = np.full((n, 2), np.nan)
    p = x.get("baro_p")
    if p is not None:
        ok = np.isfinite(p) & (p > 0)
        p0 = sv["p0"] or state.get("_p0")
        if not p0 and ok.any():
            p0 = float(p[ok][0])
            state["_p0"] = p0
        if p0:
            out[ok, 0] = isa_altitude(p[ok], p0)
    h = x.get("baro_alt")
    if h is not None:
        fill = np.isnan(out[:, 0]) & np.isfinite(h)
        out[fill, 0] = h[fill]
    return out


# ============================================================================ simülatör
def smooth_steps(t: np.ndarray, steps) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Yumuşak basamakların toplamı: (t0, T, A) → A·S((t − t0)/T), S(u) = u − sin(2πu)/2π (S' ve S'' sürekli).

    Konum, hız ve ivme analitik döner.
    """
    pos, vel, acc = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    two_pi = 2 * np.pi
    for t0, dur, amp in steps:
        u = np.clip((t - t0) / dur, 0.0, 1.0)
        inside = (t > t0) & (t < t0 + dur)
        pos += amp * (u - np.sin(two_pi * u) / two_pi)
        vel += np.where(inside, amp / dur * (1 - np.cos(two_pi * u)), 0.0)
        acc += np.where(inside, amp / dur ** 2 * two_pi * np.sin(two_pi * u), 0.0)
    return pos, vel, acc


def simulate(duration: float, fs: float, seed: int, baro_rate: float = 25.0, baro: str = "pressure",
             sigma_p: float = 4.0, sigma_h: float = 0.35, sigma_acc: float = 0.08, acc_bias: float = 0.12,
             h0: float = 120.0, p0: float = P_STD, acc_gap: tuple[float, float] | None = None) -> Simulation:
    """Dikey uçuş profili (kalkış, tırmanma, alçalma, salınımlı askı) — analitik h, v, a.

    acc_z = a + g + b(t) + gürültü (b: sabit + yavaş sinüs sapma), baro `baro_rate` Hz'de (aradaki örnekler NaN):
    baro="pressure" → ISA basınç + σ_p gürültü; "altitude" → h + σ_h gürültü. acc_gap: (başlangıç s, süre s)
    ivme kaybı (NaN) — dayanıklılık denemesi için.
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration * fs))
    t = np.arange(n) / fs
    steps = [(5.0, 10.0, 35.0), (25.0, 6.0, -10.0), (40.0, 12.0, 25.0), (62.0, 15.0, -40.0), (85.0, 5.0, 8.0),
             (100.0, 8.0, -12.0)]
    steps = [s for s in steps if s[0] < duration]
    pos, vel, acc = smooth_steps(t, steps)
    two_pi = 2 * np.pi
    osc = 0.5 * np.sin(two_pi * 0.15 * t) + 0.15 * np.sin(two_pi * 0.9 * t + 0.3)
    dosc = 0.5 * two_pi * 0.15 * np.cos(two_pi * 0.15 * t) + 0.15 * two_pi * 0.9 * np.cos(two_pi * 0.9 * t + 0.3)
    ddosc = -0.5 * (two_pi * 0.15) ** 2 * np.sin(two_pi * 0.15 * t) \
        - 0.15 * (two_pi * 0.9) ** 2 * np.sin(two_pi * 0.9 * t + 0.3)
    h, v, a = h0 + pos + osc, vel + dosc, acc + ddosc
    bias = acc_bias + 0.02 * np.sin(two_pi * t / 90.0)
    f = a + GRAVITY + bias + sigma_acc * rng.standard_normal(n)
    if acc_gap is not None:
        f[(t >= acc_gap[0]) & (t < acc_gap[0] + acc_gap[1])] = np.nan
    step = max(1, int(round(fs / baro_rate)))
    on = np.zeros(n, dtype=bool)
    on[::step] = True
    inputs = {"acc_z": f}
    noise = rng.standard_normal(n)
    if baro == "pressure":
        inputs["baro_p"] = np.where(on, isa_pressure(h, p0) + sigma_p * noise, np.nan)
    elif baro == "altitude":
        inputs["baro_alt"] = np.where(on, h + sigma_h * noise, np.nan)
    else:
        raise FusionError(f"baro 'pressure' ya da 'altitude' olmalı (verilen: {baro!r})")
    notes = (f"Dikey uçuş profili {duration:g} s @ {fs:g} Hz; baro {fs / step:g} Hz "
             f"({'basınç, σ = %g Pa' % sigma_p if baro == 'pressure' else 'irtifa, σ = %g m' % sigma_h}), "
             f"p0 = {p0:.0f} Pa; ivme σ = {sigma_acc:g} m/s², sapma ≈ {acc_bias:+g} m/s² (± 0.02 yavaş).")
    values = {"sigma_p": sigma_p, "sigma_h": sigma_h, "sigma_acc": sigma_acc, "p0": p0}
    return Simulation(fs, inputs, np.column_stack((h, v)), notes, values)


SCENARIO = Scenario(
    key="altitude",
    name="İrtifa: barometre + ivmeölçer",
    description=("Barometre (basınç ya da irtifa) ile dikey ivmeyi birleştirerek irtifa ve dikey hız kestirir. "
                 "Barometre gürültülü ama sapmasız, ivmeölçer hızlı ama integrali kayar; füzyon ikisinin iyi "
                 "yanını alır ve ivme sapmasını da kestirir. İvme dünya çerçevesinde (yukarı +, yerçekimi dahil) "
                 "olmalıdır."),
    inputs=INPUTS,
    outputs=OUTPUTS,
    params=PARAMS,
    algorithms=ALGORITHMS,
    simulator=simulate,
    one_of=ONE_OF,
    default_fs=200.0,
    raw_fn=raw,
)
