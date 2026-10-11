"""Senaryo 4 — Doğrusal çok sensörlü Kalman: aynı büyüklüğü (konum / hız / ivme) ölçen 1–6 sensörün füzyonu.

Tipik kullanımlar
  - Aynı sıcaklığı / mesafeyi ölçen 3 yedekli sensör (farklı gürültü, farklı hız): hepsi "konum" türü.
  - Enkoder konumu + ivmeölçer: konum + ivme türü, model "sabit ivme".
  - Ultrasonik + lidar mesafe + enkoder hızı: konum + konum + hız, model "sabit hız".
"Konum" burada ölçülen büyüklüğün kendisidir (sıcaklık, seviye, mesafe …); hız ve ivme onun 1. ve 2. türevi.

Girişler: s1 … s6 (hepsi isteğe bağlı, en az biri). Her yuvanın türü (konum | hız | ivme) ve ölçüm gürültüsü σ
senaryo parametresidir. Yuvalar farklı hızda olabilir (NaN = o örnekte ölçüm yok). Birimler tutarlı olmalı
(ör. konum m, hız m/s, ivme m/s²); "ham" birim değeri olduğu gibi kullanır.

Çıktılar (sabit): konum, hız, ivme. Model derecesinin kapsamadığı türevler modelin varsaydığı değerdir:
"sabit seviye"de hız = ivme = 0, "sabit hız"da ivme = 0 (model türevi). Yani ivme yalnız "sabit ivme"
modelinde kestirilir.

Model (durum x = [p, ṗ, p̈][:d], d = derece + 1; T = örnek aralığı)
  F[i][j] = T^(j−i)/(j−i)!  (j ≥ i);  en yüksek türevin türevi beyaz gürültü, yoğunluğu σ_q²:
  Q[i][j] = σ_q²·T^(2d−1−i−j) / ((2d−1−i−j)·(d−1−i)!·(d−1−j)!)
     (d = 1: σ_q²T;  d = 2: σ_q²[[T³/3, T²/2], [T²/2, T]];  d = 3: [[T⁵/20, T⁴/8, T³/6], …])
  Ölçüm: z = x[tür] + n,  R = σ_yuva²  — örnekte hangi yuvalar varsa sıralı skaler güncelleme.
Algoritmalar
  weighted_mean : taban çizgisi — her örnekte o an gelen aynı türdeki ölçümlerin ters varyans ağırlıklı
                  ortalaması (zamanda füzyon yok; ölçüm gelmeyen örnekte son değer tutulur). Hiç sensörü olmayan
                  tür için çıktı NaN.
  kf            : Kalman filtresi (çok hızlı, NaN'ları atlayan sıralı skaler güncelleme).
  pf            : parçacık filtresi (aynı model; doğrusal-Gauss durumda KF'ye yakınsar, karşılaştırma için).
Başlatma: konum türünde yuva eşlenmişse ilk konum ölçümünün geldiği örnekte: p = o örnekteki konum ölçümlerinin
ters varyans ortalaması (varyansı 1/Σ(1/R)); hız / ivme o örnekte ölçüldüyse onlar, değilse 0 ± σ0. Konum yuvası
hiç yoksa konum göreli başlar (ilk ölçüm anında 0, varyans 0).
"""
from __future__ import annotations

import math

import numpy as np

from spt.dsp.fusion_scenarios.base import (
    NAN,
    UNITS_GENERIC,
    Choice,
    Estimator,
    FusionError,
    Input,
    Output,
    Param,
    Scenario,
    Simulation,
    chol_psd,
    liu_west_resample,
    name_tokens,
    normalized_weights,
)

SLOTS = 6
ORDERS = (("seviye", "Sabit seviye (0. derece)"), ("hiz", "Sabit hız (1. derece)"),
          ("ivme", "Sabit ivme (2. derece)"))
KINDS = (("konum", "Konum (büyüklüğün kendisi)"), ("hiz", "Hız (1. türev)"), ("ivme", "İvme (2. türev)"))
LEVEL = {"seviye": 0, "konum": 0, "hiz": 1, "ivme": 2}
KIND_OF_LEVEL = ("konum", "hiz", "ivme")
ORDER_OF_LEVEL = ("seviye", "hiz", "ivme")
_BIG = 1e300

WORDS = ("mesafe", "distance", "range", "konum", "pos", "position", "seviye", "level", "sicaklik", "temp",
         "temperature", "enkoder", "encoder", "hiz", "speed", "vel", "velocity", "ivme", "acc", "accel", "lidar",
         "sonar", "ultrason", "ultrasonik", "kizilotesi", "infrared", "ir", "tof", "laser", "lazer", "sensor",
         "olcum", "deger", "value", "basinc", "pressure", "nem", "humidity", "agirlik", "weight", "yuk", "load",
         "akim", "current", "gerilim", "voltage", "ntc", "pt100", "termokupl", "thermocouple", "potansiyometre",
         "pot", "lvdt")
_SPEED_WORDS = ("hiz", "speed", "vel", "velocity", "rpm")
_ACC_WORDS = ("ivme", "acc", "accel", "hizlanma")
_SPEED_UNITS = ("m/s", "cm/s", "mm/s", "km/h")
_ACC_UNITS = ("m/s²", "g", "mg")

INPUTS = tuple(Input(f"s{i}", f"Sensör {i}", ("",), UNITS_GENERIC, False, WORDS,
                     "Türü (konum / hız / ivme) ve σ'sı senaryo parametrelerinden")
               for i in range(1, SLOTS + 1))
OUTPUTS = (
    Output("konum", "Konum", "birim", False, ("konum", "pos", "position", "mesafe", "distance", "range", "seviye",
                                             "level", "sicaklik", "temp", "temperature", "deger", "value", "x")),
    Output("hiz", "Hız", "birim/s", False, ("hiz", "speed", "vel", "velocity", "v", "turev")),
    Output("ivme", "İvme", "birim/s²", False, ("ivme", "acc", "accel", "a")),
)
ONE_OF = (tuple(i.key for i in INPUTS),)
PARAMS = (
    Choice("order", "Model derecesi", "hiz", ORDERS,
           "Sabit seviye: yavaş değişen büyüklük; sabit hız: türevi de izlenir; sabit ivme: ivme ölçen sensör için"),
    Param("sigma_q", "Süreç gürültüsü σ_q", 0.5, 0.0, 1e9, 6,
          "En yüksek türevin değişiminin beyaz gürültü yoğunluğu (√PSD). Seviye: birim/√s; sabit hız: "
          "birim/s²/√Hz (ivme); sabit ivme: birim/s³/√Hz (sarsıntı). Büyük → hızlı tepki, az yumuşatma"),
    Param("sigma0", "Başlangıç türev σ", 1.0, 1e-9, 1e9, 6,
          "Başlangıçta ölçülmeyen hız (birim/s) ve ivmenin (birim/s²) belirsizliği"),
) + tuple(p for i in range(1, SLOTS + 1) for p in (
    Choice(f"s{i}_kind", f"Sensör {i} türü", "konum", KINDS, "Bu yuvadaki sensör neyi ölçüyor"),
    Param(f"s{i}_sigma", f"Sensör {i} σ", 1.0, 1e-12, 1e12, 6, "Ölçüm gürültüsü (giriş biriminde, RMS)"),
))
PF_PARAMS = (
    Param("particles", "Parçacık sayısı", 500, 50, 20_000, 0, "Daha çok parçacık: daha doğru, daha yavaş"),
    Param("seed", "Tohum", 1, 0, 1_000_000, 0, "Rastgele sayı üretecinin tohumu (tekrarlanabilirlik)"),
)


def model_matrices(d: int, T: float, sigma_q: float) -> tuple[list[list[float]], list[list[float]]]:
    """Ayrık F ve Q (d = durum boyutu 1…3); bkz. modül belgesi."""
    F = [[T ** (j - i) / math.factorial(j - i) if j >= i else 0.0 for j in range(d)] for i in range(d)]
    q2 = sigma_q * sigma_q
    Q = [[q2 * T ** (2 * d - 1 - i - j) / ((2 * d - 1 - i - j) * math.factorial(d - 1 - i)
                                          * math.factorial(d - 1 - j)) for j in range(d)] for i in range(d)]
    return F, Q


def suggest(inputs: dict[str, tuple[int, ...]], units: dict[str, str], desc) -> dict:
    """auto_map önerisi: yuva türleri (birimden ya da addan) ve bunları kapsayan model derecesi."""
    values: dict = {}
    need = 1
    for key in inputs:
        name = desc.channels[inputs[key][0]].name
        toks = name_tokens(name)
        unit = units.get(key, "")
        if unit in _ACC_UNITS or any(t.startswith(w) for t in toks for w in _ACC_WORDS):
            kind = "ivme"
        elif unit in _SPEED_UNITS or any(t.startswith(w) for t in toks for w in _SPEED_WORDS):
            kind = "hiz"
        else:
            kind = "konum"
        values[f"{key}_kind"] = kind
        need = max(need, LEVEL[kind])
    values["order"] = ORDER_OF_LEVEL[need]
    return values


class _LinearBase(Estimator):
    scenario_params = PARAMS
    inputs = INPUTS
    outputs = OUTPUTS
    one_of = ONE_OF

    def _setup(self) -> None:
        self.d = LEVEL[self.sv["order"]] + 1
        self.inited = False
        self.n_meas = [0] * SLOTS
        self._model: tuple[float, list, list] | None = None
        self._init_state()

    def _init_state(self) -> None:
        pass

    def _check_inputs(self, keys: frozenset[str]) -> None:
        sv = self.sv
        order = sv["order"]
        for i in range(1, SLOTS + 1):
            kind = sv[f"s{i}_kind"]
            if f"s{i}" in keys and LEVEL[kind] > LEVEL[order]:
                raise FusionError(f"Sensör {i} '{dict(KINDS)[kind]}' ölçüyor; bunun için model derecesi en az "
                                  f"'{dict(ORDERS)[ORDER_OF_LEVEL[LEVEL[kind]]]}' olmalı")

    def _slots(self, x) -> list[tuple[int, int, float, list]]:
        """Eşlenmiş yuvalar: (yuva no, tür seviyesi, R, değer listesi)."""
        sv = self.sv
        return [(i, LEVEL[sv[f"s{i + 1}_kind"]], sv[f"s{i + 1}_sigma"] ** 2, x[f"s{i + 1}"].tolist())
                for i in range(SLOTS) if f"s{i + 1}" in x]

    @property
    def has_position_slot(self) -> bool:
        return any(LEVEL[self.sv[f"s{i}_kind"]] == 0 for i in range(1, SLOTS + 1) if f"s{i}" in (self._keys or ()))

    def _matrices(self, T):
        if self._model is None or self._model[0] != T:
            F, Q = model_matrices(self.d, T, self.sv["sigma_q"])
            self._model = (T, F, Q)
        return self._model[1], self._model[2]

    def _measurements(self, slots, k) -> list[tuple[int, float, float]]:
        meas = []
        for i, lvl, r, vals in slots:
            z = vals[k]
            if -_BIG < z < _BIG:
                meas.append((lvl, z, r))
                self.n_meas[i] += 1
        return meas

    def _initial(self, meas) -> tuple[list[float], list[float]] | None:
        """Başlangıç durumu ve (köşegen) varyansı; konum yuvası varken konum ölçümü yoksa None (beklenir)."""
        if not meas:
            return None
        has_pos = self.has_position_slot
        d = self.d
        x0, p0 = [0.0] * d, [self.sv["sigma0"] ** 2] * d
        for lvl in range(d):
            ms = [(z, r) for level, z, r in meas if level == lvl]
            if ms:
                info = sum(1.0 / r for _, r in ms)
                x0[lvl], p0[lvl] = sum(z / r for z, r in ms) / info, 1.0 / info
            elif lvl == 0:
                if has_pos:
                    return None
                p0[0] = 0.0          # konum yuvası yok: göreli konum (ilk ölçüm anında 0)
        return x0, p0

    def _out(self, state) -> tuple[float, float, float]:
        d = self.d
        return state[0], (state[1] if d > 1 else 0.0), (state[2] if d > 2 else 0.0)

    def _diag_common(self) -> dict[str, str]:
        used = [f"s{i + 1}: {c}" for i, c in enumerate(self.n_meas) if f"s{i + 1}" in (self._keys or ())]
        return {"Model": dict(ORDERS)[self.sv["order"]], "Ölçüm sayısı": ", ".join(used)}


# ============================================================================ taban çizgisi
class LinearWeightedMean(_LinearBase):
    key, name = "weighted_mean", "Ağırlıklı ortalama (anlık)"

    def _init_state(self) -> None:
        self.hold = [NAN, NAN, NAN]

    def _run(self, x, n, dt):
        slots = self._slots(x)
        out = np.empty((n, 3))
        hold = self.hold
        for k in range(n):
            meas = self._measurements(slots, k)
            if meas:
                for lvl in range(3):
                    ms = [(z, r) for level, z, r in meas if level == lvl]
                    if ms:
                        info = sum(1.0 / r for _, r in ms)
                        hold[lvl] = sum(z / r for z, r in ms) / info
            out[k] = hold
        return out

    def diagnostics(self):
        return self._diag_common()


# ============================================================================ Kalman filtresi
class LinearKF(_LinearBase):
    key, name = "kf", "Kalman filtresi"

    def _init_state(self) -> None:
        self.x: list[float] = [0.0] * self.d
        self.P: list[list[float]] = [[0.0] * self.d for _ in range(self.d)]

    def _run(self, x, n, dt):
        slots = self._slots(x)
        F, Q = self._matrices(dt)
        d = self.d
        rd = range(d)
        out = np.empty((n, 3))
        xs, P = self.x, self.P
        for k in range(n):
            meas = self._measurements(slots, k)
            if not self.inited:
                start = self._initial(meas)
                if start is None:
                    out[k] = NAN, NAN, NAN
                    continue
                xs = start[0]
                P = [[start[1][i] if i == j else 0.0 for j in rd] for i in rd]
                self.inited = True
                out[k] = self._out(xs)
                continue
            # ---- öngörü: x = F·x, P = F·P·Fᵀ + Q  (F üst üçgen)
            xs = [sum(F[i][j] * xs[j] for j in range(i, d)) for i in rd]
            FP = [[sum(F[i][m] * P[m][j] for m in range(i, d)) for j in rd] for i in rd]
            P = [[sum(FP[i][m] * F[j][m] for m in range(j, d)) + Q[i][j] for j in rd] for i in rd]
            # ---- sıralı skaler güncellemeler, H = e_tür
            for c, z, r in meas:
                pc = P[c]
                s = pc[c] + r
                kg = [P[i][c] / s for i in rd]
                y = z - xs[c]
                xs = [xs[i] + kg[i] * y for i in rd]
                P = [[P[i][j] - kg[i] * pc[j] for j in rd] for i in rd]
            out[k] = self._out(xs)
        self.x, self.P = xs, P
        return out

    def diagnostics(self):
        d = {}
        if self.inited:
            d["σ konum"] = f"{math.sqrt(max(self.P[0][0], 0.0)):.4g}"
            if self.d > 1:
                d["σ hız"] = f"{math.sqrt(max(self.P[1][1], 0.0)):.4g}"
            if self.d > 2:
                d["σ ivme"] = f"{math.sqrt(max(self.P[2][2], 0.0)):.4g}"
        d.update(self._diag_common())
        return d


# ============================================================================ parçacık filtresi
class LinearPF(_LinearBase):
    key, name = "pf", "Parçacık filtresi"
    params = PF_PARAMS

    def _init_state(self) -> None:
        self.rng = np.random.default_rng(int(self.values["seed"]))
        self.X: np.ndarray | None = None      # (d, N)
        self.w: np.ndarray | None = None
        self.ess = 0.0
        self.resamples = 0

    def _run(self, x, n, dt):
        slots = self._slots(x)
        F, Q = self._matrices(dt)
        Fm, Lq = np.array(F), chol_psd(np.array(Q))
        d, N, rng = self.d, int(self.values["particles"]), self.rng
        out = np.empty((n, 3))
        X, w = self.X, self.w
        for k in range(n):
            meas = self._measurements(slots, k)
            if X is None:
                start = self._initial(meas)
                if start is None:
                    out[k] = NAN, NAN, NAN
                    continue
                X = np.array(start[0])[:, None] + np.sqrt(np.array(start[1]))[:, None] * rng.standard_normal((d, N))
                w = np.full(N, 1.0 / N)
                self.inited, self.ess = True, float(N)
                out[k] = self._out(X @ w)
                continue
            X = Fm @ X + Lq @ rng.standard_normal((d, N))
            if meas:
                logl = np.zeros(N)
                for c, z, r in meas:
                    e = z - X[c]
                    logl -= (0.5 / r) * (e * e)
                w = normalized_weights(np.log(np.maximum(w, 1e-300)) + logl)
                self.ess = 1.0 / float(w @ w)
                if self.ess < 0.5 * N:
                    X = liu_west_resample(rng, X, w)
                    w = np.full(N, 1.0 / N)
                    self.resamples += 1
            out[k] = self._out(X @ w)
        self.X, self.w = X, w
        return out

    def diagnostics(self):
        d = {}
        if self.X is not None:
            d["ESS"] = f"{self.ess:.0f}"
            d["Yeniden örnekleme"] = str(self.resamples)
        d.update(self._diag_common())
        return d


ALGORITHMS: dict[str, type[Estimator]] = {cls.key: cls for cls in (LinearWeightedMean, LinearKF, LinearPF)}


def raw(x: dict, n: int, sv: dict, state: dict) -> np.ndarray:
    """Füzyonsuz: her örnekte o an gelen aynı türdeki ölçümlerin ters varyans (1/σ²) ağırlıklı ortalaması —
    konum ← konum türü yuvalar, hız ← hız türü, ivme ← ivme türü; o örnekte (ya da hiç) ölçüm yoksa NaN.
    (`weighted_mean` taban çizgisinin son değeri tutmayan, durumsuz hali.)"""
    num, den = np.zeros((3, n)), np.zeros((3, n))
    for i in range(1, SLOTS + 1):
        z = x.get(f"s{i}")
        if z is None:
            continue
        lvl, r = LEVEL[sv[f"s{i}_kind"]], sv[f"s{i}_sigma"] ** 2
        ok = np.isfinite(z)
        num[lvl] += np.where(ok, z, 0.0) / r
        den[lvl] += ok / r
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan).T


# ============================================================================ simülatör
DEFAULT_SENSORS = (("konum", 0.03, 100.0), ("konum", 0.10, 20.0), ("konum", 0.20, 50.0), ("hiz", 0.02, 100.0))
SENSOR_NAMES = ("lidar", "ultrasonik", "kızılötesi", "enkoder")


def truth_profile(t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analitik konum (m), hız, ivme: 3 m civarında farklı frekanslı salınımlar (taşıyıcı / platform)."""
    comps = ((1.0, 0.05, 0.0), (0.4, 0.13, 0.7), (0.15, 0.4, 1.3))
    p, v, a = np.full_like(t, 3.0), np.zeros_like(t), np.zeros_like(t)
    for amp, f, ph in comps:
        w = 2 * np.pi * f
        p += amp * np.sin(w * t + ph)
        v += amp * w * np.cos(w * t + ph)
        a -= amp * w * w * np.sin(w * t + ph)
    return p, v, a


def simulate(duration: float, fs: float, seed: int, sensors=DEFAULT_SENSORS, order: str = "hiz",
             sigma_q: float | None = None) -> Simulation:
    """Konum p(t) analitik; her sensör (tür, σ, hız Hz) kendi hızında (aradaki örnekler NaN), beyaz gürültülü.

    Varsayılan: lidar (konum, σ 0.03 m, 100 Hz), ultrasonik (konum, σ 0.10 m, 20 Hz), kızılötesi (konum, σ 0.20 m,
    50 Hz), enkoder hızı (hız, σ 0.02 m/s, 100 Hz). sigma_q verilmezse modele göre ayarlanmış değer önerilir.
    """
    if len(sensors) > SLOTS:
        raise FusionError(f"En fazla {SLOTS} sensör")
    rng = np.random.default_rng(seed)
    n = int(round(duration * fs))
    t = np.arange(n) / fs
    truth = np.column_stack(truth_profile(t))
    inputs: dict[str, np.ndarray] = {}
    # varsayılan σ_q: bu profilde RMSE taramasıyla ayarlandı (sabit hız 0.5, sabit ivme 1.0)
    values: dict = {"order": order, "sigma_q": sigma_q if sigma_q is not None else {"seviye": 1.0, "hiz": 0.5,
                                                                                     "ivme": 1.0}[order]}
    parts = []
    for i, (kind, sigma, rate) in enumerate(sensors, start=1):
        step = max(1, int(round(fs / rate)))
        on = np.zeros(n, dtype=bool)
        on[(i - 1) % step::step] = True
        inputs[f"s{i}"] = np.where(on, truth[:, LEVEL[kind]] + sigma * rng.standard_normal(n), np.nan)
        values[f"s{i}_kind"] = kind
        values[f"s{i}_sigma"] = sigma
        name = SENSOR_NAMES[i - 1] if sensors is DEFAULT_SENSORS else f"s{i}"
        parts.append(f"{name}: {kind}, σ = {sigma:g}, {fs / step:g} Hz")
    notes = f"Salınımlı konum {duration:g} s @ {fs:g} Hz; " + "; ".join(parts) + "."
    return Simulation(fs, inputs, truth, notes, values)


SCENARIO = Scenario(
    key="linear_multi",
    name="Doğrusal çok sensörlü Kalman (konum / hız / ivme)",
    description=("Aynı büyüklüğü ölçen 1–6 sensörü (yedekli sıcaklık / mesafe sensörleri, enkoder + ivmeölçer, "
                 "ultrasonik + lidar …) farklı gürültü ve hızlarıyla birleştirir. Her yuvanın türü (konum, hız, "
                 "ivme) ve σ'sı seçilir; model sabit seviye / sabit hız / sabit ivme."),
    inputs=INPUTS,
    outputs=OUTPUTS,
    params=PARAMS,
    algorithms=ALGORITHMS,
    simulator=simulate,
    one_of=ONE_OF,
    default_fs=100.0,
    suggest=suggest,
    raw_fn=raw,
)
