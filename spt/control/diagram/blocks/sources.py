"""Kaynak blokları (Simulink "Sources" kütüphanesi anlamlarıyla).

Zaman tabanlı kaynaklar sürekli örnekleme zamanında (Ts = 0) çözücünün ara adımlarında da kendi zamanlarıyla
değerlendirilir (Simulink sabit adımlı çözücüde de böyledir: ode4 ara adımında Basamak, ara adımın zamanına bakar).
Değişken adımlı çözücüde süreksizlik anları (Basamak anı, darbe kenarları, veri noktaları) "zaman olayı" olarak
bildirilir; çözücü bu anlara tam basar.
"""
from __future__ import annotations

import math

import numpy as np

from spt.control.diagram.block import (CONSTANT_TS, CONTINUOUS_TS, Param, SampleTime, SourceBlock, as_vec,
                                       p_ts, register_block)

_TOL = 1e-9


def _bcast(*vals: np.ndarray) -> int:
    sizes = {as_vec(v).size for v in vals}
    big = [s for s in sizes if s != 1]
    if len(set(big)) > 1:
        return -1
    return big[0] if big else 1


class _VecSource(SourceBlock):
    """Vektör parametreleri skaler genişletmeyle ortak genişliğe yayılan kaynak."""
    _vec_params: tuple[str, ...] = ()

    def check_params(self) -> None:
        if _bcast(*[self.params[k] for k in self._vec_params]) < 0:
            self.error("vektör parametrelerin uzunlukları uyumsuz (skaler ya da aynı uzunlukta olmalı).")

    def source_width(self) -> int:
        return _bcast(*[self.params[k] for k in self._vec_params])

    def setup(self, in_widths, out_widths):
        w = out_widths[0]
        self._v = {k: np.broadcast_to(as_vec(self.params[k]), (w,)).copy() for k in self._vec_params}


@register_block
class Step(_VecSource):
    type_name = "Step"
    display_name = "Basamak"
    category = "sources"
    description = "t < basamak anı iken başlangıç, sonra son değer."
    simulink = "Step"
    PARAMS = (
        Param("step_time", "float", 1.0, "Basamak anı", unit="s", allow_inf=True),
        Param("initial", "vector", 0.0, "Başlangıç değeri"),
        Param("final", "vector", 1.0, "Son değer"),
        p_ts(0.0),
    )
    _vec_params = ("initial", "final")

    def major(self, t, x, xd, u):
        self._after = t >= self.params["step_time"]

    def output(self, t, x, xd, u):
        if self.ctx.freeze_events and not self.ctx.major:
            after = self._after
        else:
            after = t >= self.params["step_time"]
        return [self._v["final"] if after else self._v["initial"]]

    def next_event(self, t):
        st = self.params["step_time"]
        return st if t < st else math.inf


@register_block
class Ramp(_VecSource):
    type_name = "Ramp"
    display_name = "Rampa"
    category = "sources"
    description = "Başlangıç anından sonra sabit eğimle artan sinyal: y = y0 + eğim·(t − t0)."
    simulink = "Ramp"
    PARAMS = (
        Param("slope", "vector", 1.0, "Eğim", unit="1/s"),
        Param("start_time", "float", 0.0, "Başlangıç anı", unit="s"),
        Param("initial", "vector", 0.0, "Başlangıç çıkışı"),
    )
    _vec_params = ("slope", "initial")

    def sample_time(self):
        return CONTINUOUS_TS

    def output(self, t, x, xd, u):
        st = self.params["start_time"]
        if t < st:
            return [self._v["initial"]]
        return [self._v["initial"] + self._v["slope"] * (t - st)]

    def next_event(self, t):
        st = self.params["start_time"]
        return st if t < st else math.inf


@register_block
class SineWave(_VecSource):
    type_name = "SineWave"
    display_name = "Sinüs"
    category = "sources"
    description = "Zaman tabanlı sinüs: y = genlik·sin(ω·t + faz) + ofset."
    simulink = "Sine Wave (Time based)"
    PARAMS = (
        Param("amplitude", "vector", 1.0, "Genlik"),
        Param("bias", "vector", 0.0, "Ofset (bias)"),
        Param("frequency", "vector", 1.0, "Açısal frekans ω", unit="rad/s"),
        Param("phase", "vector", 0.0, "Faz", unit="rad"),
        p_ts(0.0),
    )
    _vec_params = ("amplitude", "bias", "frequency", "phase")

    def output(self, t, x, xd, u):
        v = self._v
        return [v["amplitude"] * np.sin(v["frequency"] * t + v["phase"]) + v["bias"]]


@register_block
class Constant(_VecSource):
    type_name = "Constant"
    display_name = "Sabit"
    category = "sources"
    description = "Sabit değer (skaler ya da vektör)."
    simulink = "Constant"
    PARAMS = (
        Param("value", "vector", 1.0, "Değer", min_len=1),
        p_ts(float("inf"), hint="inf: sabit (varsayılan), 0: sürekli, Ts > 0: ayrık."),
    )
    _vec_params = ("value",)

    def output(self, t, x, xd, u):
        return [self._v["value"]]

    def linearize(self, t, x, xd, u):
        w = self.out_widths[0]
        return np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((w, 0)), np.zeros((w, 0))


@register_block
class Ground(SourceBlock):
    type_name = "Ground"
    display_name = "Toprak (sıfır)"
    category = "sources"
    description = "Sıfır sinyali (kullanılmayan girişleri bağlamak için)."
    simulink = "Ground"
    PARAMS = (Param("width", "int", 1, "Genişlik", min=1),)

    def sample_time(self):
        return CONSTANT_TS

    def source_width(self):
        return self.params["width"]

    def output(self, t, x, xd, u):
        return [np.zeros(self.params["width"])]


@register_block
class PulseGenerator(_VecSource):
    type_name = "PulseGenerator"
    display_name = "Darbe üreteci"
    category = "sources"
    description = "Zaman tabanlı darbe: gecikmeden sonra her periyodun başındaki genişlik boyunca genlik, aksi 0."
    simulink = "Pulse Generator (Time based)"
    PARAMS = (
        Param("amplitude", "vector", 1.0, "Genlik"),
        Param("period", "float", 1.0, "Periyot", unit="s", min=0.0, strict_min=True),
        Param("width", "float", 50.0, "Darbe genişliği (periyodun yüzdesi)", unit="%", min=0.0, max=100.0),
        Param("delay", "float", 0.0, "Faz gecikmesi", unit="s", min=0.0),
        p_ts(0.0),
    )
    _vec_params = ("amplitude",)

    def _frac(self, t: float) -> tuple[int, float]:
        n = (t - self.params["delay"]) / self.params["period"]
        k = math.floor(n + _TOL * max(1.0, abs(n)))
        return k, n - k

    def _is_on(self, t: float) -> bool:
        if t < self.params["delay"] - _TOL * self.params["period"]:
            return False
        _, frac = self._frac(t)
        return frac < self.params["width"] / 100.0 - _TOL

    def major(self, t, x, xd, u):
        self._on = self._is_on(t)

    def output(self, t, x, xd, u):
        on = self._on if (self.ctx.freeze_events and not self.ctx.major) else self._is_on(t)
        return [self._v["amplitude"] if on else np.zeros_like(self._v["amplitude"])]

    def next_event(self, t):
        if not self.ts.is_continuous:
            return math.inf
        P, d, w = self.params["period"], self.params["delay"], self.params["width"] / 100.0
        if t < d - _TOL * P:
            return d
        k, frac = self._frac(t)
        fall = d + (k + w) * P
        if frac < w - _TOL and fall > t:
            return fall
        return d + (k + 1) * P


@register_block
class Clock(SourceBlock):
    type_name = "Clock"
    display_name = "Saat"
    category = "sources"
    description = "Benzetim zamanı t."
    simulink = "Clock"
    PARAMS = ()

    def sample_time(self):
        return CONTINUOUS_TS

    def output(self, t, x, xd, u):
        return [np.array([t])]


class _RandomBase(_VecSource):
    """Örnekleme anlarında yeni rastgele değer üreten ayrık kaynak (tohumlu; benzetim başında yeniden başlar)."""

    def allowed_sample_times(self):
        return "discrete"

    def reset(self):
        self._rng = np.random.default_rng(int(self.params["seed"]))
        self._cur = self._draw()

    def output(self, t, x, xd, u):
        return [self._cur]

    def commit(self, t, x, xd, u, y):
        self._cur = self._draw()

    def _draw(self) -> np.ndarray:
        raise NotImplementedError


@register_block
class RandomNumber(_RandomBase):
    type_name = "RandomNumber"
    display_name = "Rastgele sayı (Gauss)"
    category = "sources"
    description = "Her örnekleme anında N(ortalama, varyans) dağılımlı değer (tohumlu, tekrarlanabilir)."
    simulink = "Random Number"
    PARAMS = (
        Param("mean", "vector", 0.0, "Ortalama"),
        Param("variance", "vector", 1.0, "Varyans", min=0.0),
        Param("seed", "int", 0, "Tohum", min=0),
        p_ts(0.1),
    )
    _vec_params = ("mean", "variance")

    def _draw(self):
        w = self.out_widths[0]
        return self._v["mean"] + np.sqrt(self._v["variance"]) * self._rng.standard_normal(w)


@register_block
class UniformRandomNumber(_RandomBase):
    type_name = "UniformRandomNumber"
    display_name = "Rastgele sayı (tekdüze)"
    category = "sources"
    description = "Her örnekleme anında [en küçük, en büyük) aralığında tekdüze değer (tohumlu)."
    simulink = "Uniform Random Number"
    PARAMS = (
        Param("minimum", "vector", -1.0, "En küçük"),
        Param("maximum", "vector", 1.0, "En büyük"),
        Param("seed", "int", 0, "Tohum", min=0),
        p_ts(0.1),
    )
    _vec_params = ("minimum", "maximum")

    def check_params(self):
        super().check_params()
        if np.any(as_vec(self.params["maximum"]) < as_vec(self.params["minimum"])):
            self.error("en büyük değer en küçükten küçük olamaz.")

    def _draw(self):
        w = self.out_widths[0]
        lo, hi = self._v["minimum"], self._v["maximum"]
        return lo + (hi - lo) * self._rng.random(w)


@register_block
class BandLimitedWhiteNoise(_RandomBase):
    type_name = "BandLimitedWhiteNoise"
    display_name = "Bant sınırlı beyaz gürültü"
    category = "sources"
    description = ("Sürekli sistemlere beyaz gürültü: y = √(güç / Ts)·N(0, 1) her Ts'de (güç = tek taraflı değil, "
                   "Simulink'teki 'Noise power' = güç spektral yoğunluğu yüksekliği).")
    simulink = "Band-Limited White Noise"
    PARAMS = (
        Param("power", "vector", 0.1, "Gürültü gücü (PSD yüksekliği)", min=0.0),
        Param("seed", "int", 23341, "Tohum", min=0),
        p_ts(0.1),
    )
    _vec_params = ("power",)

    def _draw(self):
        w = self.out_widths[0]
        return np.sqrt(self._v["power"] / self.ts.period) * self._rng.standard_normal(w)


@register_block
class DataSource(SourceBlock):
    type_name = "DataSource"
    display_name = "Veri kaynağı"
    category = "sources"
    description = ("Ölçülmüş (t, değer) dizisinden sinyal: doğrusal ara değerleme ya da sıfır dereceli tutma (ZOH); "
                   "veri dışında tutma / doğrusal / sıfır. Gerçek veriyle benzetim için.")
    simulink = "From Workspace"
    PARAMS = (
        Param("time", "vector", [0.0, 1.0], "Zaman noktaları", unit="s", min_len=1),
        Param("values", "matrix", [[0.0], [1.0]], "Değerler (satır = zaman noktası, sütun = kanal)", orient="col"),
        Param("interp", "choice", "linear", "Ara değerleme",
              choices={"linear": "Doğrusal", "zoh": "Sıfır dereceli tutma (en son örnek)"}),
        Param("extrap", "choice", "hold", "Veri aralığı dışında",
              choices={"hold": "İlk / son değeri tut", "linear": "Doğrusal dış değerleme", "zero": "Sıfır"}),
        p_ts(0.0),
    )

    def check_params(self):
        t = self.params["time"]
        v = self.params["values"]
        if v.shape[0] != t.size:
            if v.shape[0] == 1 and v.shape[1] == t.size:      # satır vektörü verildiyse sütuna çevir
                self.params["values"] = v = v.T.copy()
            else:
                self.error(f"değer satır sayısı ({v.shape[0]}) zaman noktası sayısına ({t.size}) eşit olmalı.")
        if v.shape[1] < 1:
            self.error("en az bir kanal gerekli.")
        if np.any(np.diff(t) < 0):
            self.error("zaman noktaları azalmayan sırada olmalı.")
        if not np.isfinite(t).all():
            self.error("zaman noktaları sonlu olmalı.")

    def source_width(self):
        return int(self.params["values"].shape[1])

    def setup(self, in_widths, out_widths):
        self._t = self.params["time"]
        self._y = self.params["values"]

    def _value(self, t: float) -> np.ndarray:
        tt, yy = self._t, self._y
        n = tt.size
        ext = self.params["extrap"]
        if n == 1:
            return yy[0].copy() if (ext != "zero" or t == tt[0]) else np.zeros(yy.shape[1])
        k = int(np.searchsorted(tt, t, side="right")) - 1
        if k < 0:
            if ext == "hold":
                return yy[0].copy()
            if ext == "zero":
                return np.zeros(yy.shape[1])
            j = int(np.searchsorted(tt, tt[0], side="right"))     # ilk farklı zaman
            j = min(j, n - 1)
            if tt[j] == tt[0]:
                return yy[0].copy()
            return yy[0] + (yy[j] - yy[0]) * (t - tt[0]) / (tt[j] - tt[0])
        if k >= n - 1:
            if t == tt[-1] or ext == "hold":
                return yy[-1].copy()
            if ext == "zero":
                return np.zeros(yy.shape[1])
            j = int(np.searchsorted(tt, tt[-1], side="left")) - 1  # son farklı zamandan önceki nokta
            if j < 0:
                return yy[-1].copy()
            return yy[-1] + (yy[-1] - yy[j]) * (t - tt[-1]) / (tt[-1] - tt[j])
        if self.params["interp"] == "zoh":
            return yy[k].copy()
        t0, t1 = tt[k], tt[k + 1]
        return yy[k] + (yy[k + 1] - yy[k]) * ((t - t0) / (t1 - t0))

    def major(self, t, x, xd, u):
        self._held = self._value(t)

    def output(self, t, x, xd, u):
        if self.params["interp"] == "zoh" and self.ctx.freeze_events and not self.ctx.major:
            return [self._held.copy()]
        return [self._value(t)]

    def next_event(self, t):
        if not self.ts.is_continuous:
            return math.inf
        k = int(np.searchsorted(self._t, t, side="right"))
        return float(self._t[k]) if k < self._t.size else math.inf


__all__ = ["Step", "Ramp", "SineWave", "Constant", "Ground", "PulseGenerator", "Clock", "RandomNumber",
           "UniformRandomNumber", "BandLimitedWhiteNoise", "DataSource", "SampleTime"]
