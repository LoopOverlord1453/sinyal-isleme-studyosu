"""Ayrık bloklar (Simulink "Discrete" kütüphanesi anlamlarıyla).

Ayrık bloklar yalnız kendi örnekleme anlarında çıkış hesaplar ve durum günceller; arada çıkış tutulur (ZOH).
Örnekleme zamanı −1 (miras) verilirse girişten alınır; sürekli bir sinyalden miras alınamaz (hata).
"""
from __future__ import annotations

import numpy as np

from spt.control.diagram import _lti
from spt.control.diagram.block import Block, Param, as_vec, expand, p_ts, register_block, width_join
from spt.control.diagram.blocks.continuous import LTIBlock, check_abcd


@register_block
class DiscreteTransferFcn(LTIBlock):
    type_name = "DiscreteTransferFcn"
    display_name = "Ayrık transfer fonksiyonu"
    category = "discrete"
    description = "H(z) = pay(z) / payda(z); z'nin azalan kuvvetleri (MATLAB tf(num, den, Ts) gibi)."
    simulink = "Discrete Transfer Fcn"
    discrete_only = True
    PARAMS = (
        Param("num", "matrix", [1.0], "Pay katsayıları (z'nin azalan kuvvetleri)"),
        Param("den", "vector", [1.0, -0.5], "Payda katsayıları (z'nin azalan kuvvetleri)", min_len=1),
        p_ts(-1.0),
    )

    def _build(self):
        return _lti.tf2ss(self.params["num"], self.params["den"])


@register_block
class DiscreteFilter(LTIBlock):
    type_name = "DiscreteFilter"
    display_name = "Ayrık filtre"
    category = "discrete"
    description = "H(z) = (b0 + b1 z⁻¹ + …) / (a0 + a1 z⁻¹ + …); z⁻¹'in artan kuvvetleri (DSP / filter() gibi)."
    simulink = "Discrete Filter"
    discrete_only = True
    PARAMS = (
        Param("num", "vector", [1.0], "Pay b (z⁻¹'in artan kuvvetleri)", min_len=1),
        Param("den", "vector", [1.0, -0.5], "Payda a (z⁻¹'in artan kuvvetleri)", min_len=1),
        p_ts(-1.0),
    )

    def _build(self):
        b, a = as_vec(self.params["num"]), as_vec(self.params["den"])
        if a[0] == 0:
            self.error("paydanın ilk katsayısı (a0) sıfır olamaz.")
        n = max(b.size, a.size)
        bz = np.concatenate([b, np.zeros(n - b.size)])
        az = np.concatenate([a, np.zeros(n - a.size)])
        return _lti.tf2ss(bz, az)


@register_block
class DiscreteStateSpace(LTIBlock):
    type_name = "DiscreteStateSpace"
    display_name = "Ayrık durum uzayı"
    category = "discrete"
    description = "x[k+1] = A x[k] + B u[k], y[k] = C x[k] + D u[k]."
    simulink = "Discrete State-Space"
    discrete_only = True
    PARAMS = (
        Param("A", "matrix", [[0.5]], "A matrisi"),
        Param("B", "matrix", [[1.0]], "B matrisi", orient="col"),
        Param("C", "matrix", [[1.0]], "C matrisi"),
        Param("D", "matrix", [[0.0]], "D matrisi"),
        Param("x0", "vector", 0.0, "Başlangıç durumu"),
        p_ts(-1.0),
    )

    def _build(self):
        return check_abcd(self, self.params["A"], self.params["B"], self.params["C"], self.params["D"])

    def _x0(self):
        return self.params["x0"]


@register_block
class UnitDelay(Block):
    type_name = "UnitDelay"
    display_name = "Birim gecikme"
    category = "discrete"
    description = "y[k] = u[k − 1] (z⁻¹); başlangıç koşulu y[0]."
    simulink = "Unit Delay"
    PARAMS = (Param("x0", "vector", 0.0, "Başlangıç koşulu"), p_ts(-1.0))

    def allowed_sample_times(self):
        return "discrete"

    def feedthrough(self):
        return [False]

    def propagate_widths(self, in_widths, guess=False):
        return [width_join([in_widths[0], as_vec(self.params["x0"]).size], guess)]

    def setup(self, in_widths, out_widths):
        w = out_widths[0]
        if in_widths[0] not in (1, w) or as_vec(self.params["x0"]).size not in (1, w):
            self.error(f"giriş ({in_widths[0]}) ve başlangıç koşulu genişlikleri uyumsuz.")
        self.nxd = w

    def xd0(self):
        return expand(self.params["x0"], self.nxd).copy()

    def output(self, t, x, xd, u):
        return [xd.copy()]

    def update(self, t, x, xd, u):
        return np.broadcast_to(u[0], (self.nxd,)).astype(float).copy()

    def linearize(self, t, x, xd, u):
        w = self.nxd
        B = np.eye(w) if self.in_widths[0] == w else np.ones((w, 1))
        return np.zeros((w, w)), B, np.eye(w), np.zeros((w, self.in_widths[0]))


_DTI_METHODS = {"forward_euler": "İleri Euler", "backward_euler": "Geri Euler", "trapezoidal": "Trapez"}


@register_block
class DiscreteIntegrator(Block):
    type_name = "DiscreteIntegrator"
    display_name = "Ayrık integratör"
    category = "discrete"
    description = ("Ayrık integral K·Ts·(…): İleri Euler y[n] = x[n]; Geri Euler y[n] = x[n] + K·Ts·u[n]; "
                   "Trapez y[n] = x[n] + K·Ts/2·u[n]. Başlangıç koşulu durum x[0]'a uygulanır (Simulink 'Auto').")
    simulink = "Discrete-Time Integrator"
    PARAMS = (
        Param("method", "choice", "forward_euler", "Yöntem", choices=_DTI_METHODS),
        Param("K", "float", 1.0, "Kazanç K"),
        Param("x0", "vector", 0.0, "Başlangıç koşulu"),
        Param("upper", "vector", float("inf"), "Üst doyum sınırı", allow_inf=True),
        Param("lower", "vector", float("-inf"), "Alt doyum sınırı", allow_inf=True),
        p_ts(-1.0),
    )

    def allowed_sample_times(self):
        return "discrete"

    def feedthrough(self):
        return [self.params["method"] != "forward_euler"]

    def propagate_widths(self, in_widths, guess=False):
        sizes = [as_vec(self.params[k]).size for k in ("x0", "upper", "lower")]
        return [width_join([in_widths[0]] + sizes, guess)]

    def setup(self, in_widths, out_widths):
        w = out_widths[0]
        if in_widths[0] not in (1, w):
            self.error("giriş genişliği başlangıç koşuluyla uyumsuz.")
        self.nxd = w
        self._U = expand(self.params["upper"], w)
        self._L = expand(self.params["lower"], w)
        if np.any(self._U < self._L):
            self.error("üst sınır alt sınırdan küçük olamaz.")
        self._lim = bool(np.any(np.isfinite(self._U)) or np.any(np.isfinite(self._L)))

    def xd0(self):
        x0 = expand(self.params["x0"], self.nxd).copy()
        return np.clip(x0, self._L, self._U) if self._lim else x0

    def _gain(self) -> float:
        return self.params["K"] * self.ts.period

    def _sat(self, v):
        return np.clip(v, self._L, self._U) if self._lim else v

    def output(self, t, x, xd, u):
        m = self.params["method"]
        if m == "forward_euler":
            return [self._sat(xd.copy())]
        kt = self._gain() if m == "backward_euler" else 0.5 * self._gain()
        return [self._sat(xd + kt * u[0])]

    def update(self, t, x, xd, u):
        m = self.params["method"]
        kt = self._gain()
        if m == "forward_euler":
            return self._sat(xd + kt * u[0])
        if m == "backward_euler":
            return self._sat(xd + kt * u[0])
        y = self._sat(xd + 0.5 * kt * u[0])
        return self._sat(y + 0.5 * kt * u[0])

    def linearize(self, t, x, xd, u):
        w = self.nxd
        E = np.eye(w) if self.in_widths[0] == w else np.ones((w, 1))
        kt = self._gain()
        m = self.params["method"]
        Dk = 0.0 if m == "forward_euler" else (kt if m == "backward_euler" else 0.5 * kt)
        return np.eye(w), kt * E, np.eye(w), Dk * E


@register_block
class ZeroOrderHold(Block):
    type_name = "ZeroOrderHold"
    display_name = "Sıfır dereceli tutucu"
    category = "discrete"
    description = "Girişi örnekleme anlarında örnekler ve bir sonraki ana kadar tutar."
    simulink = "Zero-Order Hold"
    PARAMS = (p_ts(0.1),)

    def allowed_sample_times(self):
        return "discrete"

    def output(self, t, x, xd, u):
        return [u[0].copy()]

    def setup(self, in_widths, out_widths):
        pass

    def linearize(self, t, x, xd, u):
        w = self.out_widths[0]
        return np.zeros((0, 0)), np.zeros((0, w)), np.zeros((w, 0)), np.eye(w)


@register_block
class DiscreteDerivative(Block):
    type_name = "DiscreteDerivative"
    display_name = "Ayrık türev"
    category = "discrete"
    description = "y[n] = K·(u[n] − u[n−1]) / Ts; başlangıçtaki önceki ölçekli giriş K·u[−1]/Ts verilebilir."
    simulink = "Discrete Derivative"
    PARAMS = (
        Param("K", "float", 1.0, "Kazanç K"),
        Param("x0", "vector", 0.0, "Önceki ölçekli giriş K·u[−1]/Ts"),
        p_ts(-1.0),
    )

    def allowed_sample_times(self):
        return "discrete"

    def propagate_widths(self, in_widths, guess=False):
        return [width_join([in_widths[0], as_vec(self.params["x0"]).size], guess)]

    def setup(self, in_widths, out_widths):
        w = out_widths[0]
        if in_widths[0] not in (1, w):
            self.error("giriş genişliği başlangıç koşuluyla uyumsuz.")
        self.nxd = w

    def xd0(self):
        return expand(self.params["x0"], self.nxd).copy()

    def output(self, t, x, xd, u):
        return [self.params["K"] * u[0] / self.ts.period - xd]

    def update(self, t, x, xd, u):
        return np.broadcast_to(self.params["K"] * u[0] / self.ts.period, (self.nxd,)).astype(float).copy()

    def linearize(self, t, x, xd, u):
        w = self.nxd
        E = np.eye(w) if self.in_widths[0] == w else np.ones((w, 1))
        g = self.params["K"] / self.ts.period
        return np.zeros((w, w)), g * E, -np.eye(w), g * E


@register_block
class Memory(Block):
    type_name = "Memory"
    display_name = "Bellek"
    category = "discrete"
    description = ("Bir önceki ana adımdaki girişi verir (sürekli modellerde cebirsel döngüyü kırmak için); çıkış "
                   "ara adımlarda sabittir. Ayrık hızda birim gecikmeyle aynıdır.")
    simulink = "Memory"
    steady_passthrough = True
    PARAMS = (Param("x0", "vector", 0.0, "Başlangıç çıkışı"),)

    def feedthrough(self):
        return [False]

    def propagate_widths(self, in_widths, guess=False):
        return [width_join([in_widths[0], as_vec(self.params["x0"]).size], guess)]

    def setup(self, in_widths, out_widths):
        w = out_widths[0]
        if in_widths[0] not in (1, w):
            self.error("giriş genişliği başlangıç çıkışıyla uyumsuz.")
        self._w = w

    def reset(self):
        self._prev = expand(self.params["x0"], self._w).astype(float).copy()
        self._out = self._prev.copy()

    def major(self, t, x, xd, u):
        self._out = self._prev.copy()

    def output(self, t, x, xd, u):
        # ara adımda sabit (Simulink "fixed in minor step"): çıkış yalnız ana adımda değişir
        return [self._out.copy()]

    def commit(self, t, x, xd, u, y):
        self._prev = np.broadcast_to(u[0], (self._w,)).astype(float).copy()

    def linearize(self, t, x, xd, u):
        w = self._w
        E = np.eye(w) if self.in_widths[0] == w else np.ones((w, 1))
        if self.ts.is_discrete:                   # birim gecikme
            return np.zeros((w, w)), E, np.eye(w), np.zeros((w, self.in_widths[0]))
        return np.zeros((0, 0)), np.zeros((0, self.in_widths[0])), np.zeros((w, 0)), E


__all__ = ["DiscreteTransferFcn", "DiscreteFilter", "DiscreteStateSpace", "UnitDelay", "DiscreteIntegrator",
           "ZeroOrderHold", "DiscreteDerivative", "Memory"]
