"""Bitki blokları ve eklenti taban sınıfları (MPC, İHA modelleri gibi başka modüllerin blokları için).

- `ExprStateSpace`: ifadelerle tanımlı doğrusal olmayan durum uzayı (dosyaya kaydedilebilir; ör. sarkaç, motor).
- `ContinuousPlant`: Python'da yazılan sürekli bitki eklentileri için taban (f, g, isteğe bağlı analitik Jacobian).
- `DiscreteController`: kendi iç belleğini (her türlü Python nesnesi) yöneten ayrık kontrolcü eklentileri için taban
  (ör. MPC: `compute` hesaplar, bellek yalnız ana adımda `commit` ile ilerler).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from spt.control.diagram.block import Block, Param, as_vec, p_ts, register_block
from spt.control.diagram.expr import Expression, prepare_env, split_expressions


@register_block
class ExprStateSpace(Block):
    type_name = "ExprStateSpace"
    display_name = "Doğrusal olmayan sistem (ifade)"
    category = "plants"
    description = ("x' = f(x, u, t), y = g(x, u, t) ifadelerle (';' ile ayrılmış, her durum / çıkış için bir ifade). "
                   "Değişkenler x(1) … x(n), u(1) … u(m), t. Ts > 0 iken f bir sonraki durumu verir (x[k+1]).")
    simulink = "S-Function / MATLAB Function (bitki)"
    PARAMS = (
        Param("f", "expr", "x(2); -9.81*sin(x(1)) - 0.5*x(2) + u(1)", "Durum türevleri f (her durum için)"),
        Param("g", "expr", "x(1)", "Çıkışlar g"),
        Param("x0", "vector", [0.0, 0.0], "Başlangıç durumu", min_len=1),
        Param("inputs", "int", 1, "Giriş genişliği m", min=0),
        p_ts(0.0, hint="0: sürekli (f = x'), Ts > 0: ayrık (f = x[k+1])."),
    )

    def check_params(self):
        n = as_vec(self.params["x0"]).size
        m = self.params["inputs"]
        fs, gs = split_expressions(self.params["f"]), split_expressions(self.params["g"])
        if len(fs) != n:
            self.error(f"f'de {n} ifade olmalı (durum sayısı = x0 uzunluğu), {len(fs)} verildi.")
        if not gs:
            self.error("en az bir çıkış ifadesi gerekli.")
        vecs = {"x": n, "u": m} if m else {"x": n}
        try:
            self._f = [Expression(e, scalars=("t",), vectors=vecs) for e in fs]
            self._g = [Expression(e, scalars=("t",), vectors=vecs) for e in gs]
            for e in self._f + self._g:
                e.check_widths(vecs, self.name)
        except Exception as exc:
            self.error(str(exc))
        self._n, self._m = n, m
        st = self.params["Ts"]
        if st.is_inherited or st.is_constant:
            self.error("örnekleme zamanı 0 (sürekli) ya da pozitif (ayrık) olmalı.")

    def n_inputs(self):
        return 1 if self._m else 0

    def feedthrough(self):
        return [any("u" in e.used for e in self._g)] if self._m else []

    def propagate_widths(self, in_widths, guess=False):
        return [len(self._g)]

    def setup(self, in_widths, out_widths):
        if self._m and in_widths[0] != self._m:
            self.error(f"giriş genişliği {self._m} olmalı ({in_widths[0]} bağlandı).")
        if self.ts.is_discrete:
            self.nx, self.nxd = 0, self._n
        else:
            self.nx, self.nxd = self._n, 0

    def x0(self):
        return as_vec(self.params["x0"]).copy() if self.nx else np.zeros(0)

    def xd0(self):
        return as_vec(self.params["x0"]).copy() if self.nxd else np.zeros(0)

    def _env(self, t, s, u):
        env = {"x": s, "t": t}
        if self._m:
            env["u"] = u[0]
        return env

    def _call(self, exprs, env, t):
        env = prepare_env(env)
        try:
            return np.array([e.eval_prepared(env) for e in exprs])
        except Exception as exc:
            self.error(f"t = {t:.6g} s: {exc}")
            raise

    def output(self, t, x, xd, u):
        s = xd if self.nxd else x
        return [self._call(self._g, self._env(t, s, u), t)]

    def derivatives(self, t, x, xd, u):
        return self._call(self._f, self._env(t, x, u), t)

    def update(self, t, x, xd, u):
        return self._call(self._f, self._env(t, xd, u), t)


class ContinuousPlant(Block):
    """Sürekli bitki eklenti tabanı.

    Alt sınıf şunları tanımlar: `type_name`, `display_name`, `category` (varsayılan "plants"), `PARAMS`;
    `dims()` → (nx, nu, ny); `f(t, x, u)` → x' (nx,); `g(t, x, u)` → y (ny,); `g_uses_u` (y u'ya doğrudan bağlı mı;
    varsayılan False); isteğe bağlı `initial_state()` ve `jacobian(t, x, u)` → (A, B, C, D) analitik doğrusallaştırma.
    Tek vektör giriş (nu) ve tek vektör çıkış (ny) vardır; nu = 0 ise girişsizdir."""
    category = "plants"
    g_uses_u = False

    def dims(self) -> tuple[int, int, int]:
        raise NotImplementedError

    def f(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def g(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def initial_state(self) -> np.ndarray:
        return np.zeros(self.dims()[0])

    def jacobian(self, t, x, u):
        return None

    # --- Block sözleşmesine bağlama
    def sample_time(self):
        from spt.control.diagram.block import CONTINUOUS_TS
        return CONTINUOUS_TS

    def allowed_sample_times(self):
        return "continuous"

    def n_inputs(self):
        return 1 if self.dims()[1] else 0

    def feedthrough(self):
        return [bool(self.g_uses_u)] if self.dims()[1] else []

    def propagate_widths(self, in_widths, guess=False):
        return [self.dims()[2]]

    def setup(self, in_widths, out_widths):
        nx, nu, _ = self.dims()
        if nu and in_widths[0] != nu:
            self.error(f"giriş genişliği {nu} olmalı ({in_widths[0]} bağlandı).")
        self.nx = nx

    def x0(self):
        return np.asarray(self.initial_state(), dtype=float).copy()

    def _uin(self, u):
        return u[0] if self.dims()[1] else np.zeros(0)

    def output(self, t, x, xd, u):
        return [np.asarray(self.g(t, x, self._uin(u)), dtype=float)]

    def derivatives(self, t, x, xd, u):
        return np.asarray(self.f(t, x, self._uin(u)), dtype=float)

    def linearize(self, t, x, xd, u):
        J = self.jacobian(t, x, self._uin(u))
        if J is None:
            return None
        return tuple(np.atleast_2d(np.asarray(m, dtype=float)) for m in J)


class DiscreteController(Block):
    """Ayrık kontrolcü eklenti tabanı (iç belleği Python nesnesi olan kontrolcüler; ör. MPC).

    Alt sınıf şunları tanımlar: `PARAMS` (örnekleme zamanı 'Ts' dahil), `port_widths_in()` → giriş portu
    genişlikleri listesi, `output_width()`, `initial_memory()` → bellek nesnesi, `compute(t, u_list, memory)` →
    (y, yeni_bellek). `compute` belleği **yerinde değiştirmemeli** (yeni nesne / kopya döndürmeli): çıkış her örnekleme
    anında hesaplanır, bellek ancak ana adım tamamlanınca (`commit`) ilerletilir. Doğrudan geçiş varsayılan olarak
    vardır (`feedthrough_flags` ile değiştirilebilir); isteğe bağlı `jacobian` doğrusallaştırma için."""
    category = "controllers"
    feedthrough_flags: tuple[bool, ...] | None = None

    def port_widths_in(self) -> list[int]:
        raise NotImplementedError

    def output_width(self) -> int:
        raise NotImplementedError

    def initial_memory(self) -> Any:
        return None

    def compute(self, t: float, u: list[np.ndarray], memory: Any) -> tuple[np.ndarray, Any]:
        raise NotImplementedError

    def allowed_sample_times(self):
        return "discrete"

    def n_inputs(self):
        return len(self.port_widths_in())

    def feedthrough(self):
        f = self.feedthrough_flags
        return list(f) if f is not None else [True] * self.n_inputs()

    def propagate_widths(self, in_widths, guess=False):
        return [self.output_width()]

    def setup(self, in_widths, out_widths):
        for i, (got, need) in enumerate(zip(in_widths, self.port_widths_in())):
            if got != need:
                self.error(f"{i + 1}. giriş genişliği {need} olmalı ({got} bağlandı).")

    def reset(self):
        self._mem = self.initial_memory()
        self._pending = None
        self._y = np.zeros(self.output_width())

    def output(self, t, x, xd, u):
        y, self._pending = self.compute(t, u, self._mem)
        return [np.asarray(y, dtype=float)]

    def commit(self, t, x, xd, u, y):
        if self._pending is not None:
            self._mem = self._pending
            self._pending = None


__all__ = ["ExprStateSpace", "ContinuousPlant", "DiscreteController"]
