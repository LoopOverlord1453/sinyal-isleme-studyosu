"""Doğrusal olmayan bloklar (Simulink "Discontinuities", "Lookup Tables", "Signal Routing/Switch", "Math" ve
"User-Defined Functions/Fcn" anlamlarıyla).

Kip (mode) anlamları
--------------------
- Sabit adımlı çözücülerde (ode1 / ode4) bloklar ara adımlarda da doğrudan değerlendirilir (Simulink gibi). İstisna:
  **Röle** konumunu yalnız ana adımda değiştirir (Simulink'te de böyle; ara adımlarda son ana adımdaki konum).
- Değişken adımlı çözücüde (ode45) sıfır geçiş etkinse (varsayılan) süreksiz bloklar kiplerini ana adımda belirler,
  ara adımlarda kip dondurulur; kip sınırının geçildiği an çözücünün sıfır geçiş aramasıyla bulunur.

Doğrusallaştırma (Simulink Control Design varsayılanları, MATLAB'den ölçüldü): Doyum, Ölü bölge, Hız sınırlayıcı ve
Nicemleyici "kazanç olarak doğrusallaştır" seçeneğiyle 1 (kapatılırsa bölgeye göre 1 / 0), Röle 0, Mutlak değer işaret,
Anahtar seçili kol, arama tablosu bulunduğu parçanın eğimi.
"""
from __future__ import annotations

import numpy as np

from spt.control.diagram.block import Block, Param, as_vec, expand, p_ts, register_block, width_join
from spt.control.diagram.expr import Expression, prepare_env, split_expressions


def _static(D: np.ndarray):
    return np.zeros((0, 0)), np.zeros((0, D.shape[1])), np.zeros((D.shape[0], 0)), D


class _ElementwiseNL(Block):
    """Tek girişli, eleman bazında doğrusal olmayan blok (vektör parametreler skaler genişletmeli)."""
    _vec_params: tuple[str, ...] = ()

    def propagate_widths(self, in_widths, guess=False):
        return [width_join([in_widths[0]] + [as_vec(self.params[k]).size for k in self._vec_params], guess)]

    def setup(self, in_widths, out_widths):
        w = out_widths[0]
        if in_widths[0] != w:
            self.error(f"giriş genişliği ({in_widths[0]}) parametre uzunluklarıyla ({w}) uyumsuz.")
        for k in self._vec_params:
            if as_vec(self.params[k]).size not in (1, w):
                self.error(f"'{self._param_defs[k].label}' 1 ya da {w} elemanlı olmalı.")
        self._v = {k: expand(self.params[k], w) for k in self._vec_params}
        self._mode = np.zeros(w, dtype=int)

    def _frozen(self) -> bool:
        return self.ctx.zc and not self.ctx.major


@register_block
class Saturation(_ElementwiseNL):
    type_name = "Saturation"
    display_name = "Doyum"
    category = "nonlinear"
    description = "y = min(max(u, alt), üst)."
    simulink = "Saturation"
    PARAMS = (
        Param("upper", "vector", 0.5, "Üst sınır", allow_inf=True),
        Param("lower", "vector", -0.5, "Alt sınır", allow_inf=True),
        Param("linearize_as_gain", "bool", True, "Doğrusallaştırmada kazanç (1) say"),
        p_ts(-1.0),
    )
    _vec_params = ("upper", "lower")

    def check_params(self):
        U, L = as_vec(self.params["upper"]), as_vec(self.params["lower"])
        n = max(U.size, L.size)
        if (U.size not in (1, n)) or (L.size not in (1, n)) or np.any(np.broadcast_to(U, n) < np.broadcast_to(L, n)):
            self.error("üst sınır alt sınırdan küçük olamaz (ve uzunluklar uyumlu olmalı).")

    def major(self, t, x, xd, u):
        if self.ctx.zc:
            v = u[0]
            self._mode = np.where(v > self._v["upper"], 1, np.where(v < self._v["lower"], -1, 0))

    def output(self, t, x, xd, u):
        U, L = self._v["upper"], self._v["lower"]
        if self._frozen():
            return [np.where(self._mode == 1, U, np.where(self._mode == -1, L, u[0]))]
        return [np.minimum(np.maximum(u[0], L), U)]

    def zero_crossings(self, t, x, xd, u):
        return np.concatenate([u[0] - self._v["upper"], u[0] - self._v["lower"]])

    def linearize(self, t, x, xd, u):
        w = self.out_widths[0]
        if self.params["linearize_as_gain"]:
            return _static(np.eye(w))
        v = u[0]
        inside = (v >= self._v["lower"]) & (v <= self._v["upper"])
        return _static(np.diag(inside.astype(float)))


@register_block
class DeadZone(_ElementwiseNL):
    type_name = "DeadZone"
    display_name = "Ölü bölge"
    category = "nonlinear"
    description = "başlangıç ≤ u ≤ bitiş iken 0; dışında u − bitiş ya da u − başlangıç."
    simulink = "Dead Zone"
    PARAMS = (
        Param("start", "vector", -0.5, "Ölü bölge başlangıcı"),
        Param("end", "vector", 0.5, "Ölü bölge bitişi"),
        Param("linearize_as_gain", "bool", True, "Doğrusallaştırmada kazanç (1) say"),
        p_ts(-1.0),
    )
    _vec_params = ("start", "end")

    def check_params(self):
        a, b = as_vec(self.params["start"]), as_vec(self.params["end"])
        n = max(a.size, b.size)
        if (a.size not in (1, n)) or (b.size not in (1, n)) or np.any(np.broadcast_to(b, n) < np.broadcast_to(a, n)):
            self.error("bitiş başlangıçtan küçük olamaz (ve uzunluklar uyumlu olmalı).")

    def major(self, t, x, xd, u):
        if self.ctx.zc:
            v = u[0]
            self._mode = np.where(v > self._v["end"], 1, np.where(v < self._v["start"], -1, 0))

    def output(self, t, x, xd, u):
        v = u[0]
        a, b = self._v["start"], self._v["end"]
        mode = self._mode if self._frozen() else np.where(v > b, 1, np.where(v < a, -1, 0))
        return [np.where(mode == 1, v - b, np.where(mode == -1, v - a, 0.0))]

    def zero_crossings(self, t, x, xd, u):
        return np.concatenate([u[0] - self._v["end"], u[0] - self._v["start"]])

    def linearize(self, t, x, xd, u):
        w = self.out_widths[0]
        if self.params["linearize_as_gain"]:
            return _static(np.eye(w))
        v = u[0]
        outside = (v > self._v["end"]) | (v < self._v["start"])
        return _static(np.diag(outside.astype(float)))


@register_block
class RateLimiter(_ElementwiseNL):
    type_name = "RateLimiter"
    display_name = "Hız sınırlayıcı"
    category = "nonlinear"
    description = ("Çıkışın değişim hızını [düşme, yükselme] aralığında sınırlar: Δt = t − (son ana adım), "
                   "y = y_önceki + kırp(u − y_önceki, düşme·Δt, yükselme·Δt). Sürekli kipte ilk çıkış u'dur.")
    simulink = "Rate Limiter"
    steady_passthrough = False
    PARAMS = (
        Param("rising", "vector", 1.0, "Yükselme hızı sınırı", unit="1/s", allow_inf=True),
        Param("falling", "vector", -1.0, "Düşme hızı sınırı", unit="1/s", allow_inf=True),
        Param("x0", "vector", 0.0, "Başlangıç koşulu (ayrık kipte y[−1])"),
        Param("linearize_as_gain", "bool", True, "Doğrusallaştırmada kazanç (1) say"),
        p_ts(0.0, hint="0: sürekli (varsayılan), −1: miras (girişten; ayrık olabilir), Ts > 0: ayrık."),
    )
    _vec_params = ("rising", "falling", "x0")

    def check_params(self):
        r, f = as_vec(self.params["rising"]), as_vec(self.params["falling"])
        n = max(r.size, f.size)
        if (r.size not in (1, n)) or (f.size not in (1, n)) or np.any(np.broadcast_to(r, n) < np.broadcast_to(f, n)):
            self.error("yükselme sınırı düşme sınırından küçük olamaz.")

    def reset(self):
        w = self.out_widths[0]
        if self.ts.is_discrete:
            self._prev = expand(self.params["x0"], w).astype(float).copy()
            self._tprev = None
        else:
            self._prev = None
            self._tprev = None

    def _limit(self, t: float, v: np.ndarray) -> np.ndarray:
        if self.ts.is_discrete:
            dt = self.ts.period
        else:
            if self._prev is None:
                return v.copy()
            dt = t - self._tprev
        prev = self._prev
        rate = v - prev
        rise = dt * self._v["rising"]
        fall = dt * self._v["falling"]
        return np.where(rate > rise, prev + rise, np.where(rate < fall, prev + fall, v))

    def output(self, t, x, xd, u):
        return [self._limit(t, u[0])]

    def commit(self, t, x, xd, u, y):
        self._prev = y[0].copy()
        self._tprev = t

    def trim_output(self, t, x, xd, u):
        return [u[0].copy()]

    def linearize(self, t, x, xd, u):
        # kararlı hâlde y = u: seçenekten bağımsız kazanç 1 (MATLAB linearize'da da 'kazanç olarak' kapalıyken 1)
        w = self.out_widths[0]
        return _static(np.eye(w))


@register_block
class Quantizer(_ElementwiseNL):
    type_name = "Quantizer"
    display_name = "Nicemleyici"
    category = "nonlinear"
    description = "y = q·round(u / q) (yarım sıfırdan uzağa, MATLAB round gibi)."
    simulink = "Quantizer"
    PARAMS = (
        Param("interval", "vector", 0.5, "Nicemleme aralığı q", min=0.0, strict_min=True),
        Param("linearize_as_gain", "bool", True, "Doğrusallaştırmada kazanç (1) say"),
        p_ts(-1.0),
    )
    _vec_params = ("interval",)

    def output(self, t, x, xd, u):
        q = self._v["interval"]
        r = u[0] / q
        return [q * (np.sign(r) * np.floor(np.abs(r) + 0.5))]

    def linearize(self, t, x, xd, u):
        w = self.out_widths[0]
        return _static(np.eye(w) if self.params["linearize_as_gain"] else np.zeros((w, w)))


@register_block
class Relay(_ElementwiseNL):
    type_name = "Relay"
    display_name = "Röle (histerezisli)"
    category = "nonlinear"
    description = ("u ≥ açma noktası → açık çıkışı; u ≤ kapama noktası → kapalı çıkışı; arada önceki konum. Konum "
                   "yalnız ana adımlarda değişir; başlangıç konumu kapalı.")
    simulink = "Relay"
    PARAMS = (
        Param("on_point", "vector", 0.1, "Açma noktası (eşik)"),
        Param("off_point", "vector", -0.1, "Kapama noktası (eşik)"),
        Param("on_value", "vector", 1.0, "Açıkken çıkış"),
        Param("off_value", "vector", 0.0, "Kapalıyken çıkış"),
        p_ts(-1.0),
    )
    _vec_params = ("on_point", "off_point", "on_value", "off_value")

    def check_params(self):
        a, b = as_vec(self.params["on_point"]), as_vec(self.params["off_point"])
        n = max(a.size, b.size)
        if (a.size not in (1, n)) or (b.size not in (1, n)) or np.any(np.broadcast_to(a, n) < np.broadcast_to(b, n)):
            self.error("açma noktası kapama noktasından küçük olamaz.")

    def reset(self):
        self._on = np.zeros(self.out_widths[0], dtype=bool)

    def _next_state(self, v: np.ndarray, on: np.ndarray) -> np.ndarray:
        return (v >= self._v["on_point"]) | (~(v <= self._v["off_point"]) & on)

    def major(self, t, x, xd, u):
        self._on = self._next_state(u[0], self._on)

    def output(self, t, x, xd, u):
        return [np.where(self._on, self._v["on_value"], self._v["off_value"])]

    def zero_crossings(self, t, x, xd, u):
        return np.where(self._on, u[0] - self._v["off_point"], u[0] - self._v["on_point"])

    def trim_output(self, t, x, xd, u):
        on = self._next_state(u[0], np.zeros(self.out_widths[0], dtype=bool))
        return [np.where(on, self._v["on_value"], self._v["off_value"])]

    def linearize(self, t, x, xd, u):
        w = self.out_widths[0]
        return _static(np.zeros((w, w)))


@register_block
class Abs(Block):
    type_name = "Abs"
    display_name = "Mutlak değer"
    category = "nonlinear"
    description = "y = |u|."
    simulink = "Abs"
    PARAMS = (p_ts(-1.0),)

    def setup(self, in_widths, out_widths):
        self._mode = np.ones(out_widths[0])

    def major(self, t, x, xd, u):
        if self.ctx.zc:
            self._mode = np.where(u[0] < 0, -1.0, 1.0)

    def output(self, t, x, xd, u):
        if self.ctx.zc and not self.ctx.major:
            return [self._mode * u[0]]
        return [np.abs(u[0])]

    def zero_crossings(self, t, x, xd, u):
        return u[0].copy()

    def linearize(self, t, x, xd, u):
        return _static(np.diag(np.sign(u[0])))


@register_block
class Sign(Block):
    type_name = "Sign"
    display_name = "İşaret"
    category = "nonlinear"
    description = "y = sign(u) ∈ {−1, 0, 1}."
    simulink = "Sign"
    PARAMS = (p_ts(-1.0),)

    def setup(self, in_widths, out_widths):
        self._mode = np.zeros(out_widths[0])

    def major(self, t, x, xd, u):
        if self.ctx.zc:
            self._mode = np.sign(u[0])

    def output(self, t, x, xd, u):
        if self.ctx.zc and not self.ctx.major:
            return [self._mode.copy()]
        return [np.sign(u[0])]

    def zero_crossings(self, t, x, xd, u):
        return u[0].copy()

    def linearize(self, t, x, xd, u):
        w = self.out_widths[0]
        return _static(np.zeros((w, w)))


@register_block
class Lookup1D(Block):
    type_name = "Lookup1D"
    display_name = "Arama tablosu (1-B)"
    category = "nonlinear"
    description = "Kırılma noktaları ve tablo değerleriyle ara değerleme (eleman bazında)."
    simulink = "1-D Lookup Table"
    PARAMS = (
        Param("breakpoints", "vector", [-1.0, 0.0, 1.0], "Kırılma noktaları (kesin artan)", min_len=1),
        Param("table", "vector", [-1.0, 0.0, 1.0], "Tablo değerleri", min_len=1),
        Param("interp", "choice", "linear", "Ara değerleme",
              choices={"linear": "Doğrusal", "flat": "Düz (alttaki nokta)", "nearest": "En yakın"}),
        Param("extrap", "choice", "linear", "Dış değerleme", choices={"linear": "Doğrusal", "clip": "Kırp"}),
        p_ts(-1.0),
    )

    def check_params(self):
        bp, tb = self.params["breakpoints"], self.params["table"]
        if bp.size != tb.size:
            self.error(f"kırılma noktası ({bp.size}) ve tablo ({tb.size}) uzunlukları eşit olmalı.")
        if np.any(np.diff(bp) <= 0):
            self.error("kırılma noktaları kesin artan olmalı.")

    def _segment(self, v: float) -> int:
        bp = self.params["breakpoints"]
        k = int(np.searchsorted(bp, v, side="right")) - 1
        return min(max(k, 0), bp.size - 2)

    def _eval(self, v: float) -> tuple[float, float]:
        bp, tb = self.params["breakpoints"], self.params["table"]
        n = bp.size
        if n == 1:
            return float(tb[0]), 0.0
        mode, ext = self.params["interp"], self.params["extrap"]
        if v < bp[0] or v > bp[-1]:
            if ext == "clip" or mode != "linear":
                return float(tb[0] if v < bp[0] else tb[-1]), 0.0
        k = self._segment(v)
        if mode == "flat":
            j = 0 if v < bp[0] else (n - 1 if v >= bp[-1] else k)
            return float(tb[j]), 0.0
        if mode == "nearest":
            if v >= bp[-1]:
                return float(tb[-1]), 0.0
            if v <= bp[0]:
                return float(tb[0]), 0.0
            j = k + 1 if (v - bp[k]) >= (bp[k + 1] - v) else k
            return float(tb[j]), 0.0
        s = (tb[k + 1] - tb[k]) / (bp[k + 1] - bp[k])
        return float(tb[k] + s * (v - bp[k])), float(s)

    def output(self, t, x, xd, u):
        return [np.array([self._eval(float(v))[0] for v in u[0]])]

    def linearize(self, t, x, xd, u):
        return _static(np.diag([self._eval(float(v))[1] for v in u[0]]))


_CRITERIA = {"ge": "u2 ≥ eşik", "gt": "u2 > eşik", "ne0": "u2 ≠ 0"}


@register_block
class Switch(Block):
    type_name = "Switch"
    display_name = "Anahtar"
    category = "nonlinear"
    description = "Ölçüt sağlanırsa 1. giriş, değilse 3. giriş geçer (2. giriş denetim); eleman bazında."
    simulink = "Switch"
    PARAMS = (
        Param("criteria", "choice", "ge", "Ölçüt", choices=_CRITERIA),
        Param("threshold", "float", 0.0, "Eşik"),
        p_ts(-1.0),
    )

    def n_inputs(self):
        return 3

    def input_labels(self):
        return ["u1", "u2 (denetim)", "u3"]

    def propagate_widths(self, in_widths, guess=False):
        return [width_join(list(in_widths), guess)]

    def setup(self, in_widths, out_widths):
        w = out_widths[0]
        for i, wi in enumerate(in_widths):
            if wi not in (1, w):
                self.error(f"{i + 1}. giriş genişliği ({wi}) uyumsuz (çıkış {w}).")
        self._mode = np.ones(w, dtype=bool)

    def _pass(self, c: np.ndarray) -> np.ndarray:
        k, T = self.params["criteria"], self.params["threshold"]
        if k == "ge":
            return c >= T
        if k == "gt":
            return c > T
        return c != 0

    def major(self, t, x, xd, u):
        if self.ctx.zc:
            self._mode = np.broadcast_to(self._pass(u[1]), (self.out_widths[0],)).copy()

    def output(self, t, x, xd, u):
        w = self.out_widths[0]
        sel = self._mode if (self.ctx.zc and not self.ctx.major) else self._pass(u[1])
        return [np.broadcast_to(np.where(sel, u[0], u[2]), (w,)).astype(float).copy()]

    def zero_crossings(self, t, x, xd, u):
        T = 0.0 if self.params["criteria"] == "ne0" else self.params["threshold"]
        return u[1] - T

    def linearize(self, t, x, xd, u):
        w = self.out_widths[0]
        sel = np.broadcast_to(self._pass(u[1]), (w,)).astype(float)
        blocks = []
        for i, wi in enumerate(self.in_widths):
            if i == 1:
                blocks.append(np.zeros((w, wi)))
                continue
            d = sel if i == 0 else 1.0 - sel
            blocks.append(np.diag(d) if wi == w else d.reshape(-1, 1))
        return _static(np.hstack(blocks))


@register_block
class MinMax(Block):
    type_name = "MinMax"
    display_name = "En küçük / en büyük"
    category = "nonlinear"
    description = "Girişlerin eleman bazında en küçüğü / en büyüğü; tek girişte elemanların en küçüğü / büyüğü."
    simulink = "MinMax"
    PARAMS = (
        Param("function", "choice", "max", "İşlev", choices={"min": "En küçük", "max": "En büyük"}),
        Param("inputs", "int", 2, "Giriş sayısı", min=1, max=64),
        p_ts(-1.0),
    )

    def n_inputs(self):
        return self.params["inputs"]

    def propagate_widths(self, in_widths, guess=False):
        if self.n_inputs() == 1:
            return [1]
        return [width_join(list(in_widths), guess)]

    def setup(self, in_widths, out_widths):
        w = out_widths[0]
        if self.n_inputs() > 1:
            for i, wi in enumerate(in_widths):
                if wi not in (1, w):
                    self.error(f"{i + 1}. giriş genişliği ({wi}) uyumsuz (çıkış {w}).")
        self._sel = np.zeros(w, dtype=int)

    def _stack(self, u):
        if self.n_inputs() == 1:
            return u[0].reshape(-1, 1)                      # (eleman, 1) → seçim elemanlar arasında
        w = self.out_widths[0]
        return np.stack([np.broadcast_to(ui, (w,)) for ui in u], axis=0)   # (giriş, eleman)

    def _select(self, u) -> np.ndarray:
        S = self._stack(u)
        fn = np.argmin if self.params["function"] == "min" else np.argmax
        return np.atleast_1d(fn(S, axis=0))

    def major(self, t, x, xd, u):
        if self.ctx.zc:
            self._sel = self._select(u)

    def output(self, t, x, xd, u):
        S = self._stack(u)
        sel = self._sel if (self.ctx.zc and not self.ctx.major) else self._select(u)
        return [np.array([S[sel[j], j] for j in range(S.shape[1])], dtype=float)]

    def zero_crossings(self, t, x, xd, u):
        S = self._stack(u)
        cur = np.array([S[self._sel[j], j] for j in range(S.shape[1])])
        return (S - cur).ravel()

    def linearize(self, t, x, xd, u):
        sel = self._select(u)
        if self.n_inputs() == 1:
            D = np.zeros((1, u[0].size))
            D[0, sel[0]] = 1.0
            return _static(D)
        w = self.out_widths[0]
        blocks = []
        for i, wi in enumerate(self.in_widths):
            d = (sel == i).astype(float)
            blocks.append(np.diag(d) if wi == w else d.reshape(-1, 1))
        return _static(np.hstack(blocks))


@register_block
class Fcn(Block):
    type_name = "Fcn"
    display_name = "İfade"
    category = "nonlinear"
    description = ("Güvenli matematiksel ifade: u (1 elemanlıysa yalın), u(1), u(2) … (1 tabanlı), t. ';' ile "
                   "ayrılmış birden çok ifade vektör çıkış verir. Ör. 'u(1)^2 + sin(u(2))'.")
    simulink = "Fcn"
    PARAMS = (
        Param("expr", "expr", "u", "İfade", hint="Fonksiyonlar: sin cos tan exp log sqrt abs min max atan2 sat …"),
        p_ts(-1.0),
    )

    def check_params(self):
        parts = split_expressions(self.params["expr"])
        if not parts:
            self.error("ifade boş.")
        try:
            self._exprs = [Expression(p, scalars=("t",), vectors={"u": None}) for p in parts]
        except Exception as exc:  # ControlError
            self.error(str(exc))

    def propagate_widths(self, in_widths, guess=False):
        return [len(self._exprs)]

    def setup(self, in_widths, out_widths):
        for e in self._exprs:
            e.check_widths({"u": in_widths[0]}, self.name)

    def output(self, t, x, xd, u):
        env = prepare_env({"u": u[0], "t": t})
        try:
            return [np.array([e.eval_prepared(env) for e in self._exprs])]
        except Exception as exc:
            self.error(f"t = {t:.6g} s: {exc}")
            raise


__all__ = ["Saturation", "DeadZone", "RateLimiter", "Quantizer", "Relay", "Abs", "Sign", "Lookup1D", "Switch",
           "MinMax", "Fcn"]
