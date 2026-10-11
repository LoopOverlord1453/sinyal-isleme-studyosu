"""Doğrusal sürekli bloklar ve temel matematik (Simulink "Continuous" ve "Math Operations" anlamlarıyla).

Doğrusal zamanla değişmez bloklar ortak `LTIBlock` tabanını kullanır: x' = A x + B u (sürekli) ya da
x[k+1] = A x[k] + B u[k] (ayrık), y = C x + D u; doğrudan geçiş D ≠ 0 ise vardır. Doğrusallaştırmaları tamdır.
"""
from __future__ import annotations

import bisect

import numpy as np

from spt.control.diagram import _lti
from spt.control.errors import ControlError
from spt.control.diagram.block import (CONTINUOUS_TS, Block, Param, as_vec, expand, p_ts, register_block,
                                       width_join)


# ============================================================================ LTI tabanı
class LTIBlock(Block):
    """Durum uzayı gerçeklemeli doğrusal blok tabanı. Alt sınıf `_build()` ile (A, B, C, D) döndürür ve isteğe bağlı
    `_x0()` başlangıç durumunu verir. Sürekli mi ayrık mı olduğu çözülmüş örnekleme zamanından belirlenir."""
    discrete_only = False
    continuous_only = False

    def check_params(self) -> None:
        try:
            A, B, C, D = self._build()
        except ControlError as exc:
            msg = str(exc)
            if msg.startswith(f"'{self.name}' bloğu"):
                raise
            self.error(msg[0].lower() + msg[1:] if msg else msg)
        self._A, self._B, self._C, self._D = (np.atleast_2d(np.asarray(m, dtype=np.float64)) for m in (A, B, C, D))
        n = self._A.shape[0] if self._A.size else 0
        if n == 0:
            self._A = np.zeros((0, 0))
            self._B = np.zeros((0, self._D.shape[1]))
            self._C = np.zeros((self._D.shape[0], 0))
        self._n = n
        x0 = as_vec(self._x0())
        if x0.size not in (1, n) and not (n == 0 and x0.size <= 1):
            self.error(f"başlangıç durumu {n} elemanlı olmalı ({x0.size} verildi).")
        self._x0v = np.broadcast_to(x0, (n,)).copy() if n else np.zeros(0)

    def _build(self):
        raise NotImplementedError

    def _x0(self):
        return 0.0

    def allowed_sample_times(self) -> str:
        if self.discrete_only:
            return "discrete"
        if self.continuous_only:
            return "continuous"
        return "any"

    def port_widths(self) -> list[int]:
        """Giriş portlarının genişlikleri (B / D sütunları bu sırayla bölünür). Varsayılan: tek port."""
        return [self._B.shape[1]]

    def n_inputs(self):
        return len(self.port_widths())

    def feedthrough(self):
        out, c = [], 0
        for w in self.port_widths():
            out.append(bool(np.any(self._D[:, c:c + w] != 0)))
            c += w
        return out

    def propagate_widths(self, in_widths, guess=False):
        return [self._C.shape[0]]

    def setup(self, in_widths, out_widths):
        for i, (got, need) in enumerate(zip(in_widths, self.port_widths())):
            if got != need:
                lab = self.input_labels()[i]
                self.error(f"'{lab}' girişinin genişliği {need} olmalı ({got} bağlandı).")
        if self.ts.is_discrete:
            self.nx, self.nxd = 0, self._n
        else:
            self.nx, self.nxd = self._n, 0
        self._multi = len(self.port_widths()) > 1

    def _u(self, u):
        return np.concatenate(u) if self._multi else u[0]

    def x0(self):
        return self._x0v.copy() if self.nx else np.zeros(0)

    def xd0(self):
        return self._x0v.copy() if self.nxd else np.zeros(0)

    def output(self, t, x, xd, u):
        s = xd if self.nxd else x
        uu = self._u(u)
        if self._n:
            return [self._C @ s + self._D @ uu]
        return [self._D @ uu]

    def derivatives(self, t, x, xd, u):
        return self._A @ x + self._B @ self._u(u)

    def update(self, t, x, xd, u):
        return self._A @ xd + self._B @ self._u(u)

    def linearize(self, t, x, xd, u):
        return self._A.copy(), self._B.copy(), self._C.copy(), self._D.copy()

    def state_space(self):
        """(A, B, C, D) gerçeklemesi (bloğun kendi hızında)."""
        return self._A.copy(), self._B.copy(), self._C.copy(), self._D.copy()


# ============================================================================ matematik
@register_block
class Gain(Block):
    type_name = "Gain"
    display_name = "Kazanç"
    category = "continuous"
    description = "y = K·u (eleman bazında) ya da y = K·u (matris çarpımı)."
    simulink = "Gain"
    PARAMS = (
        Param("K", "matrix", 1.0, "Kazanç K"),
        Param("mode", "choice", "elementwise", "Çarpım",
              choices={"elementwise": "Eleman bazında (K.*u)", "matrix": "Matris (K*u)"}),
        p_ts(-1.0),
    )

    def check_params(self):
        K = self.params["K"]
        if self.params["mode"] == "elementwise" and min(K.shape) > 1:
            self.error("eleman bazında çarpımda K vektör ya da skaler olmalı; matris için 'Matris (K*u)' seçin.")

    def _matrix(self) -> bool:
        return self.params["mode"] == "matrix" and self.params["K"].size > 1

    def propagate_widths(self, in_widths, guess=False):
        K = self.params["K"]
        if self._matrix():
            return [K.shape[0]]
        return [width_join([K.size, in_widths[0]], guess)]

    def setup(self, in_widths, out_widths):
        K = self.params["K"]
        if self._matrix():
            if in_widths[0] != K.shape[1]:
                self.error(f"K {K.shape[0]}×{K.shape[1]}; giriş genişliği {K.shape[1]} olmalı ({in_widths[0]} "
                           "bağlandı).")
            self._K = K
        else:
            w = out_widths[0]
            k = K.ravel()
            if k.size not in (1, w) or in_widths[0] not in (1, w):
                self.error(f"K ({k.size} eleman) ile giriş ({in_widths[0]}) genişlikleri uyumsuz.")
            self._k = np.broadcast_to(k, (w,)).copy()

    def output(self, t, x, xd, u):
        if self._matrix():
            return [self._K @ u[0]]
        return [self._k * u[0]]

    def linearize(self, t, x, xd, u):
        w = self.out_widths[0]
        if self._matrix():
            D = self._K.copy()
        else:
            D = np.diag(self._k) if self.in_widths[0] == w else self._k.reshape(-1, 1)
        return np.zeros((0, 0)), np.zeros((0, self.in_widths[0])), np.zeros((w, 0)), D


def _expand_matrix(wi: int, w: int) -> np.ndarray:
    return np.ones((w, 1)) if wi == 1 and w != 1 else np.eye(w)


@register_block
class Sum(Block):
    type_name = "Sum"
    display_name = "Toplam"
    category = "continuous"
    description = "İşaretli toplam: '+-' gibi işaret dizisi (| boşluk). Tek girişte elemanları toplar."
    simulink = "Sum"
    PARAMS = (
        Param("signs", "signs", "+-", "İşaretler", hint="Her giriş için '+' ya da '-'; '|' yalnız görünüm boşluğu."),
        p_ts(-1.0),
    )

    def _signs(self) -> list[float]:
        return [1.0 if c == "+" else -1.0 for c in self.params["signs"] if c in "+-"]

    def n_inputs(self):
        return len(self._signs())

    def input_labels(self):
        return ["+" if s > 0 else "−" for s in self._signs()]

    def propagate_widths(self, in_widths, guess=False):
        if self.n_inputs() == 1:
            return [1]
        return [width_join(list(in_widths), guess)]

    def setup(self, in_widths, out_widths):
        w = out_widths[0]
        if self.n_inputs() > 1:
            for i, wi in enumerate(in_widths):
                if wi not in (1, w):
                    self.error(f"{i + 1}. giriş genişliği ({wi}) diğerleriyle uyumsuz (çıkış {w}); yalnız skaler "
                               "genişletilebilir.")
        self._s = self._signs()

    def output(self, t, x, xd, u):
        s = self._s
        if len(s) == 1:
            return [np.array([s[0] * float(np.sum(u[0]))])]
        acc = s[0] * u[0]
        for si, ui in zip(s[1:], u[1:]):
            acc = acc + si * ui
        if acc.shape[0] != self.out_widths[0]:
            acc = np.broadcast_to(acc, (self.out_widths[0],)).copy()
        return [acc]

    def linearize(self, t, x, xd, u):
        w = self.out_widths[0]
        if len(self._s) == 1:
            D = self._s[0] * np.ones((1, self.in_widths[0]))
        else:
            D = np.hstack([si * _expand_matrix(wi, w) for si, wi in zip(self._s, self.in_widths)])
        return np.zeros((0, 0)), np.zeros((0, D.shape[1])), np.zeros((w, 0)), D


@register_block
class Product(Block):
    type_name = "Product"
    display_name = "Çarpım"
    category = "continuous"
    description = "Eleman bazında çarpım / bölüm: '**', '*/' … Tek '*' girişinde elemanları çarpar."
    simulink = "Product (Element-wise)"
    PARAMS = (
        Param("ops", "str", "**", "İşlemler", hint="Her giriş için '*' (çarp) ya da '/' (böl)."),
        p_ts(-1.0),
    )

    def check_params(self):
        ops = self.params["ops"].replace(" ", "")
        if not ops or any(c not in "*/" for c in ops):
            self.error("işlemler yalnız '*' ve '/' içermeli.")
        if ops == "/":
            self.error("tek girişte yalnız '*' (elemanları çarp) desteklenir.")
        self._ops = ops

    def n_inputs(self):
        return len(self._ops)

    def input_labels(self):
        return ["×" if c == "*" else "÷" for c in self._ops]

    def propagate_widths(self, in_widths, guess=False):
        if len(self._ops) == 1:
            return [1]
        return [width_join(list(in_widths), guess)]

    def setup(self, in_widths, out_widths):
        w = out_widths[0]
        if len(self._ops) > 1:
            for i, wi in enumerate(in_widths):
                if wi not in (1, w):
                    self.error(f"{i + 1}. giriş genişliği ({wi}) diğerleriyle uyumsuz.")

    def output(self, t, x, xd, u):
        if len(self._ops) == 1:
            return [np.array([float(np.prod(u[0]))])]
        acc = np.ones(self.out_widths[0])
        for c, ui in zip(self._ops, u):
            if c == "*":
                acc = acc * ui
            else:
                if np.any(ui == 0):
                    self.error(f"sıfıra bölme (t = {t:.6g} s).")
                acc = acc / ui
        return [acc]

    def linearize(self, t, x, xd, u):
        w = self.out_widths[0]
        if len(self._ops) == 1:
            v = u[0]
            D = np.array([[float(np.prod(np.delete(v, j))) for j in range(v.size)]])
            return np.zeros((0, 0)), np.zeros((0, v.size)), np.zeros((1, 0)), D
        f = []
        for c, ui in zip(self._ops, u):
            ue = np.broadcast_to(ui, (w,)).astype(float)
            if c == "/" and np.any(ue == 0):
                self.error("doğrusallaştırma noktasında sıfıra bölme.")
            f.append(ue if c == "*" else 1.0 / ue)
        blocks = []
        for j, (c, ui) in enumerate(zip(self._ops, u)):
            others = np.ones(w)
            for i, fi in enumerate(f):
                if i != j:
                    others = others * fi
            ue = np.broadcast_to(ui, (w,)).astype(float)
            d = others if c == "*" else -others / (ue * ue)
            blocks.append(np.diag(d) if ui.size == w else d.reshape(-1, 1))
        D = np.hstack(blocks)
        return np.zeros((0, 0)), np.zeros((0, D.shape[1])), np.zeros((w, 0)), D


# ============================================================================ integratör / türev
@register_block
class Integrator(Block):
    type_name = "Integrator"
    display_name = "İntegratör"
    category = "continuous"
    description = "y = ∫u dt + x0; isteğe bağlı doyum sınırları (sınırdayken türev sıfırlanır, Simulink gibi)."
    simulink = "Integrator"
    PARAMS = (
        Param("x0", "vector", 0.0, "Başlangıç koşulu"),
        Param("upper", "vector", float("inf"), "Üst doyum sınırı", allow_inf=True),
        Param("lower", "vector", float("-inf"), "Alt doyum sınırı", allow_inf=True),
    )

    def check_params(self):
        U, L = as_vec(self.params["upper"]), as_vec(self.params["lower"])
        if U.size > 1 and L.size > 1 and U.size != L.size:
            self.error("sınır vektörlerinin uzunlukları uyumsuz.")
        if np.any(np.broadcast_to(U, max(U.size, L.size)) < np.broadcast_to(L, max(U.size, L.size))):
            self.error("üst sınır alt sınırdan küçük olamaz.")

    def sample_time(self):
        return CONTINUOUS_TS

    def allowed_sample_times(self):
        return "continuous"

    def feedthrough(self):
        return [False]

    def propagate_widths(self, in_widths, guess=False):
        sizes = [as_vec(self.params[k]).size for k in ("x0", "upper", "lower")]
        return [width_join([in_widths[0]] + sizes, guess)]

    def setup(self, in_widths, out_widths):
        w = out_widths[0]
        if in_widths[0] not in (1, w):
            self.error(f"giriş genişliği ({in_widths[0]}) başlangıç koşuluyla ({w}) uyumsuz.")
        for k in ("x0", "upper", "lower"):
            if as_vec(self.params[k]).size not in (1, w):
                self.error(f"'{self._param_defs[k].label}' {w} elemanlı olmalı.")
        self.nx = w
        self._U = expand(self.params["upper"], w)
        self._L = expand(self.params["lower"], w)
        self._limited = bool(np.any(np.isfinite(self._U)) or np.any(np.isfinite(self._L)))
        x0 = expand(self.params["x0"], w)
        self._x0v = np.clip(x0, self._L, self._U) if self._limited else x0

    def x0(self):
        return self._x0v.copy()

    def output(self, t, x, xd, u):
        if self._limited:
            return [np.clip(x, self._L, self._U)]
        return [x.copy()]

    def derivatives(self, t, x, xd, u):
        du = np.broadcast_to(u[0], (self.nx,)).astype(float)
        if not self._limited:
            return du.copy()
        usat = x >= self._U
        lsat = x <= self._L
        free = (~usat & ~lsat) | (lsat & (du > 0)) | (usat & (du < 0))
        return np.where(free, du, 0.0)

    def project(self, x):
        if self._limited and (np.any(x > self._U) or np.any(x < self._L)):
            return np.clip(x, self._L, self._U)
        return None

    def zero_crossings(self, t, x, xd, u):
        if not self._limited:
            return None
        return np.concatenate([x - self._U, x - self._L])

    def linearize(self, t, x, xd, u):
        w = self.nx
        B = np.eye(w) if self.in_widths[0] == w else np.ones((w, 1))
        return np.zeros((w, w)), B, np.eye(w), np.zeros((w, self.in_widths[0]))


@register_block
class FilteredDerivative(LTIBlock):
    type_name = "FilteredDerivative"
    display_name = "Türev (filtreli)"
    category = "continuous"
    description = "Filtreli türev: y = N·s / (s + N) · u (N → ∞ iken saf türev); başlangıçta x0 = u(0) seçilebilir."
    simulink = "Transfer Fcn [N 0]/[1 N] (Simulink Derivative'in filtreli karşılığı)"
    continuous_only = True
    PARAMS = (
        Param("N", "float", 100.0, "Filtre katsayısı N", unit="rad/s", min=0.0, strict_min=True),
        Param("x0", "float", 0.0, "Filtre başlangıç durumu"),
    )

    def _build(self):
        N = self.params["N"]
        return [[-N]], [[N]], [[-N]], [[N]]

    def _x0(self):
        return self.params["x0"]

    def sample_time(self):
        return CONTINUOUS_TS


# ============================================================================ transfer fonksiyonu / durum uzayı
@register_block
class TransferFcn(LTIBlock):
    type_name = "TransferFcn"
    display_name = "Transfer fonksiyonu"
    category = "continuous"
    description = "G(s) = pay(s) / payda(s); azalan kuvvetler. Pay matrisse her satır bir çıkıştır (tek giriş)."
    simulink = "Transfer Fcn"
    continuous_only = True
    PARAMS = (
        Param("num", "matrix", [1.0], "Pay katsayıları (azalan kuvvetler)"),
        Param("den", "vector", [1.0, 1.0], "Payda katsayıları (azalan kuvvetler)", min_len=1),
    )

    def _build(self):
        return _lti.tf2ss(self.params["num"], self.params["den"])

    def sample_time(self):
        return CONTINUOUS_TS


@register_block
class StateSpace(LTIBlock):
    type_name = "StateSpace"
    display_name = "Durum uzayı"
    category = "continuous"
    description = "x' = A x + B u, y = C x + D u; başlangıç durumu x0."
    simulink = "State-Space"
    continuous_only = True
    PARAMS = (
        Param("A", "matrix", [[-1.0]], "A matrisi"),
        Param("B", "matrix", [[1.0]], "B matrisi", orient="col"),
        Param("C", "matrix", [[1.0]], "C matrisi"),
        Param("D", "matrix", [[0.0]], "D matrisi"),
        Param("x0", "vector", 0.0, "Başlangıç durumu"),
    )

    def _build(self):
        return check_abcd(self, self.params["A"], self.params["B"], self.params["C"], self.params["D"])

    def _x0(self):
        return self.params["x0"]

    def sample_time(self):
        return CONTINUOUS_TS


def check_abcd(block: Block, A, B, C, D):
    """A, B, C, D boyut denetimi (0 durumlu saf kazanç dahil)."""
    A = np.atleast_2d(np.asarray(A, dtype=np.float64))
    D = np.atleast_2d(np.asarray(D, dtype=np.float64))
    n = A.shape[0] if A.size else 0
    if n and A.shape[0] != A.shape[1]:
        block.error(f"A kare olmalı ({A.shape[0]}×{A.shape[1]} verildi).")
    B = np.asarray(B, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    if n == 0:
        if D.size == 0:
            block.error("durum yoksa D boş olamaz.")
        return np.zeros((0, 0)), np.zeros((0, D.shape[1])), np.zeros((D.shape[0], 0)), D
    B = np.atleast_2d(B)
    if B.shape[0] != n:
        if B.shape[1] == n and B.shape[0] == 1:
            B = B.T
        else:
            block.error(f"B satır sayısı {n} olmalı ({B.shape[0]}×{B.shape[1]} verildi).")
    C = np.atleast_2d(C)
    if C.shape[1] != n:
        block.error(f"C sütun sayısı {n} olmalı ({C.shape[0]}×{C.shape[1]} verildi).")
    m, p = B.shape[1], C.shape[0]
    if D.shape != (p, m):
        if D.size == 1 and float(D.ravel()[0]) == 0.0:
            D = np.zeros((p, m))
        else:
            block.error(f"D {p}×{m} olmalı ({D.shape[0]}×{D.shape[1]} verildi).")
    return A, B, C, D


@register_block
class ZeroPole(LTIBlock):
    type_name = "ZeroPole"
    display_name = "Sıfır-kutup"
    category = "continuous"
    description = "G(s) = k·Π(s − z) / Π(s − p); karmaşık sıfır / kutuplar eşlenik çiftler hâlinde."
    simulink = "Zero-Pole"
    continuous_only = True
    PARAMS = (
        Param("zeros", "cvector", [], "Sıfırlar"),
        Param("poles", "cvector", [-1.0], "Kutuplar"),
        Param("gain", "float", 1.0, "Kazanç k"),
    )

    def _build(self):
        num, den = _lti.zpk2tf(self.params["zeros"], self.params["poles"], self.params["gain"])
        return _lti.tf2ss(num, den)

    def sample_time(self):
        return CONTINUOUS_TS


# ============================================================================ taşıma gecikmesi
@register_block
class TransportDelay(Block):
    type_name = "TransportDelay"
    display_name = "Taşıma gecikmesi"
    category = "continuous"
    description = ("y(t) = u(t − τ); ana adımlarda saklanan girişler arasında doğrusal ara değerleme (τ adımdan "
                   "küçükse son iki noktadan doğrusal dış değerleme; Simulink gibi). t − τ ≤ t0 iken başlangıç "
                   "çıkışı. Doğrusallaştırmada Padé yaklaşımı (derece 0: kazanç 1).")
    simulink = "Transport Delay"
    steady_passthrough = True
    PARAMS = (
        Param("delay", "float", 1.0, "Gecikme τ", unit="s", min=0.0),
        Param("initial", "vector", 0.0, "Başlangıç çıkışı"),
        Param("pade_order", "int", 0, "Padé derecesi (doğrusallaştırma için)", min=0, max=20),
    )

    def sample_time(self):
        return CONTINUOUS_TS

    def allowed_sample_times(self):
        return "continuous"

    def feedthrough(self):
        return [self.params["delay"] == 0.0]

    def propagate_widths(self, in_widths, guess=False):
        return [width_join([in_widths[0], as_vec(self.params["initial"]).size], guess)]

    def setup(self, in_widths, out_widths):
        w = out_widths[0]
        if in_widths[0] != w:
            self.error(f"giriş genişliği ({in_widths[0]}) başlangıç çıkışıyla ({w}) uyumsuz.")
        if as_vec(self.params["initial"]).size not in (1, w):
            self.error(f"başlangıç çıkışı {w} elemanlı olmalı.")
        self._init = expand(self.params["initial"], w)

    def reset(self):
        self._tb: list[float] = []
        self._ub: list[np.ndarray] = []

    def output(self, t, x, xd, u):
        tau = self.params["delay"]
        if tau == 0.0:
            return [u[0].copy()]
        tm = t - tau
        tb = self._tb
        if tm <= self.ctx.t_start or not tb:
            return [self._init.copy()]
        k = bisect.bisect_right(tb, tm) - 1
        if k < 0:
            return [self._init.copy()]
        if k >= len(tb) - 1:
            # gecikme adımdan küçükse t − τ tamponun ötesindedir: son iki noktadan doğrusal dış değerleme
            # (tek nokta varsa o değer) — Simulink ile aynı
            if len(tb) < 2:
                return [self._ub[-1].copy()]
            t0, t1 = tb[-2], tb[-1]
            a = (tm - t0) / (t1 - t0)
            return [self._ub[-2] + a * (self._ub[-1] - self._ub[-2])]
        t0, t1 = tb[k], tb[k + 1]
        a = (tm - t0) / (t1 - t0)
        return [self._ub[k] + a * (self._ub[k + 1] - self._ub[k])]

    def commit(self, t, x, xd, u, y):
        if self._tb and t <= self._tb[-1]:
            self._ub[-1] = u[0].copy()
            return
        self._tb.append(t)
        self._ub.append(u[0].copy())
        tau = self.params["delay"]
        if len(self._tb) > 4096:                     # eski noktaları at (t − τ'dan önceki en az bir nokta kalır)
            k = bisect.bisect_right(self._tb, t - tau) - 2
            if k > 2048:
                del self._tb[:k]
                del self._ub[:k]

    def linearize(self, t, x, xd, u):
        w = self.out_widths[0]
        num, den = _lti.pade(self.params["delay"], self.params["pade_order"])
        a, b, c, d = _lti.tf2ss(num, den)
        I = np.eye(w)
        return np.kron(I, a), np.kron(I, b), np.kron(I, c), np.kron(I, d)


def lti_block(sys, name: str | None = None) -> Block:
    """LTI nesnesinden (ördek tipleme: A,B,C,D,Ts ya da num,den,Ts) uygun blok: sürekli → Durum Uzayı / Transfer
    Fonksiyonu, ayrık → Ayrık Durum Uzayı / Ayrık Transfer Fonksiyonu."""
    from spt.control.diagram.blocks.discrete import DiscreteStateSpace, DiscreteTransferFcn
    data = _lti.lti_data(sys)
    Ts = data["Ts"]
    if "A" in data:
        if Ts > 0:
            return DiscreteStateSpace(name, A=data["A"], B=data["B"], C=data["C"], D=data["D"], Ts=Ts)
        return StateSpace(name, A=data["A"], B=data["B"], C=data["C"], D=data["D"])
    if Ts > 0:
        return DiscreteTransferFcn(name, num=data["num"], den=data["den"], Ts=Ts)
    return TransferFcn(name, num=data["num"], den=data["den"])


__all__ = ["LTIBlock", "Gain", "Sum", "Product", "Integrator", "FilteredDerivative", "TransferFcn", "StateSpace",
           "ZeroPole", "TransportDelay", "lti_block", "check_abcd"]
