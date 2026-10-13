"""LTI modelleri: transfer fonksiyonu (tf), sıfır-kutup-kazanç (zpk) ve durum uzayı (ss).

MATLAB Control System Toolbox'ın `tf`, `zpk`, `ss` nesnelerinin karşılığı. Sözleşme
(docs/SPT-DEV-001_Gelistirme_Asamalari.docx, Aşama 4):

- `Ts = 0.0` sürekli, `Ts > 0` ayrık zaman (saniye). Farklı Ts'li modeller birleştirilemez; statik kazanç (dinamiği ve
  gecikmesi olmayan model) her Ts'ye uyar (MATLAB gibi).
- Polinomlar azalan kuvvetlerde 1-B `float64` dizi: `[1, 2, 1]` = s² + 2s + 1 (ayrıkta z). MATLAB gibi pay / payda
  normalleştirilmez (`tf(1, [2, 4])` olduğu gibi saklanır); payı sıfır olan elemanın paydası 1 yapılır.
- Gecikmeler: `input_delay` (nu,), `output_delay` (ny,), tf / zpk'de ayrıca `io_delay` (ny, nu). Sürekli zamanda
  saniye, ayrıkta tamsayı örnek. Bir G/Ç kanalının toplam gecikmesi = giriş + çıkış + io (`total_delay()`).
- Aritmetik (`+ - * /`, `**` tamsayı, `-sys`, `sys[i, j]`, `inv`) MATLAB gibi; tür önceliği ss > zpk > tf. SISO model
  MIMO modelle toplanır / çarpılırsa her elemana uygulanır (MATLAB'in skaler genişletmesi). tf toplamı / çarpımı
  MATLAB'in formülleriyle (pay·payda çapraz çarpımı; ortak payda aranmaz, sadeleştirme yapılmaz — `minreal`).
- Gecikmeli modellerin aritmetiği: tf / zpk'de iç gecikmeler G/Ç gecikmesine katlanır (MATLAB gibi); durum uzayında
  giriş / çıkış gecikmesi olarak gösterilebiliyorsa öyle tutulur. Gösterilemiyorsa (MATLAB burada ss'nin "iç gecikme"
  (InternalDelay) özelliğini kullanır) sürekli zamanda açık bir hata verilir ve Padé yaklaşımı (`pade`) önerilir;
  ayrık zamanda gecikme z⁻¹ durumları olarak modele tam katılır.

Bilinen farklar: MATLAB'in iç gecikmesi, tanımlayıcı (E) matrisi, Ts = −1 ve karmaşık katsayılı modeller desteklenmez.
"""
from __future__ import annotations

import math
import numbers
from typing import Any

import numpy as np

from spt.control.errors import ControlError

EPS = float(np.finfo(float).eps)

DELAY_INTERNAL_MSG = (
    "Bu işlem, gecikmeyi giriş / çıkışa taşınamayan bir iç gecikme gerektiriyor; sürekli zamanda bu desteklenmez. "
    "Gecikmeyi önce Padé yaklaşımıyla rasyonel hale getirin (pade(model, N)) ya da modeli ayrıklaştırın "
    "(ayrık zamanda gecikme z⁻¹ durumları olarak modele tam katılır).")


# ---------------------------------------------------------------------------------------------------------------
# Doğrulama yardımcıları
# ---------------------------------------------------------------------------------------------------------------
def _is_number(x: Any) -> bool:
    return isinstance(x, numbers.Number) and not isinstance(x, bool)


def _check_Ts(Ts: Any) -> float:
    try:
        val = float(Ts)
    except (TypeError, ValueError):
        raise ControlError("Örnekleme zamanı (Ts) bir sayı olmalı.") from None
    if not math.isfinite(val) or val < 0.0:
        raise ControlError("Örnekleme zamanı Ts ≥ 0 olmalı (0 = sürekli zaman); MATLAB'in Ts = −1 "
                           "(belirtilmemiş) kullanımı desteklenmez.")
    return val


def _real_array(x: Any, what: str) -> np.ndarray:
    a = np.asarray(x)
    if a.dtype == object:
        try:
            a = a.astype(complex)
        except (TypeError, ValueError):
            raise ControlError(f"{what} sayısal olmalı.") from None
    if np.iscomplexobj(a):
        if np.any(a.imag != 0):
            raise ControlError(f"{what} gerçel olmalı (karmaşık katsayılı modeller desteklenmez).")
        a = a.real
    try:
        a = a.astype(float)
    except (TypeError, ValueError):
        raise ControlError(f"{what} sayısal olmalı.") from None
    if not np.all(np.isfinite(a)):
        raise ControlError(f"{what} sonlu sayılardan oluşmalı (NaN / sonsuz olamaz).")
    return a


def _names(names: Any, n: int, kind: str) -> tuple[str, ...]:
    """MATLAB gibi ad listesi: None → boş adlar; tek ad + çok kanal → 'u(1)', 'u(2)' …"""
    if names is None:
        return ("",) * n
    if isinstance(names, str):
        if n == 1:
            return (names,)
        if names == "":
            return ("",) * n
        return tuple(f"{names}({i + 1})" for i in range(n))
    out = tuple("" if v is None else str(v) for v in names)
    if len(out) != n:
        raise ControlError(f"{kind} adı sayısı ({len(out)}) {kind} sayısıyla ({n}) aynı olmalı.")
    return out


def _delay_vec(d: Any, n: int, Ts: float, kind: str) -> np.ndarray:
    if d is None:
        return np.zeros(n)
    a = _real_array(np.atleast_1d(d), f"{kind} gecikmesi").ravel()
    if a.size == 1 and n != 1:
        a = np.full(n, a[0])
    if a.size != n:
        raise ControlError(f"{kind} gecikmesi {n} elemanlı olmalı ({a.size} verildi).")
    return _check_delays(a, Ts, kind)


def _check_delays(a: np.ndarray, Ts: float, kind: str) -> np.ndarray:
    if np.any(a < 0):
        raise ControlError(f"{kind} gecikmesi negatif olamaz.")
    if Ts > 0:
        r = np.round(a)
        if np.any(np.abs(a - r) > 1e-9 * np.maximum(1.0, np.abs(a))):
            raise ControlError(f"Ayrık zamanda {kind.lower()} gecikmesi örnek sayısı (tamsayı) olmalı.")
        a = r
    return a.astype(float)


def _io_delay(d: Any, ny: int, nu: int, Ts: float) -> np.ndarray:
    if d is None:
        return np.zeros((ny, nu))
    a = _real_array(d, "G/Ç gecikmesi")
    if a.ndim == 0 or a.size == 1:
        a = np.full((ny, nu), float(a.ravel()[0]) if a.size else 0.0)
    if a.ndim == 1 and a.size == ny * nu:
        a = a.reshape(ny, nu)
    if a.shape != (ny, nu):
        raise ControlError(f"G/Ç gecikmesi {ny}×{nu} olmalı.")
    return _check_delays(a, Ts, "G/Ç").reshape(ny, nu)


def _same(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-12 * max(1.0, abs(a), abs(b))


def _vec_same(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and all(_same(float(x), float(y)) for x, y in zip(a.ravel(), b.ravel()))


def _shape_str(shape: tuple[int, int]) -> str:
    return f"{shape[0]}×{shape[1]}"


# ---------------------------------------------------------------------------------------------------------------
# Polinom yardımcıları (azalan kuvvetler)
# ---------------------------------------------------------------------------------------------------------------
def _strip(p: np.ndarray) -> np.ndarray:
    nz = np.flatnonzero(p)
    return p[nz[0]:].copy() if nz.size else np.zeros(1)


def _poly(p: Any, what: str) -> np.ndarray:
    a = _real_array(np.atleast_1d(p), what)
    if a.ndim != 1:
        raise ControlError(f"{what} 1-B katsayı dizisi olmalı.")
    if a.size == 0:
        return np.zeros(1)
    return _strip(a)


def _pzero(p: np.ndarray) -> bool:
    return not np.any(p)


def _padd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """İki polinomun toplamı; yuvarlama düzeyindeki (toplananlara göre) katsayılar tam sıfır sayılır."""
    n = max(a.size, b.size)
    A = np.zeros(n)
    B = np.zeros(n)
    A[n - a.size:] = a
    B[n - b.size:] = b
    S = A + B
    S[np.abs(S) <= 8.0 * EPS * (np.abs(A) + np.abs(B))] = 0.0
    return _strip(S)


def _pmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _strip(np.convolve(a, b))


def _ppow_shift(p: np.ndarray, d: int) -> np.ndarray:
    """p(z)·z^d."""
    if d <= 0:
        return p.copy()
    return np.concatenate([p, np.zeros(int(d))])


def cplxpair(r: Any, tol: float | None = None, *, what: str = "Kökler") -> np.ndarray:
    """Karmaşık kökleri eşlenik çiftlerine eşler ve çiftleri tam eşlenik yapar (gerçel sistem için).

    Sıra korunur: her (+) üye kendi (−) eşiyle yan yana gelir; gerçel kökler (|sanal| ≤ tol·|r|) tam gerçel olur.
    Eşi bulunamayan karmaşık kök hata verir."""
    r = np.asarray(r, dtype=complex).ravel()
    if r.size == 0:
        return r
    if not np.all(np.isfinite(r)):
        raise ControlError(f"{what} sonlu olmalı.")
    tol = 100.0 * EPS if tol is None else tol
    scale = np.maximum(np.abs(r), 1.0)
    is_real = np.abs(r.imag) <= tol * scale
    used = np.zeros(r.size, bool)
    out: list[complex] = []
    for i in range(r.size):
        if used[i]:
            continue
        used[i] = True
        if is_real[i]:
            out.append(complex(r[i].real, 0.0))
            continue
        cand = [j for j in range(r.size) if not used[j] and not is_real[j]]
        if not cand:
            raise ControlError(f"{what}: {r[i]:.6g} karmaşık kökünün eşleniği yok (gerçel katsayılı model gerekir).")
        dist = np.abs(r[cand] - np.conj(r[i]))
        k = int(np.argmin(dist))
        if dist[k] > max(1e-6, 1e3 * tol) * scale[i]:
            raise ControlError(f"{what}: {r[i]:.6g} karmaşık kökünün eşleniği yok (gerçel katsayılı model gerekir).")
        j = cand[k]
        used[j] = True
        re = 0.5 * (r[i].real + r[j].real)
        im = 0.5 * (abs(r[i].imag) + abs(r[j].imag))
        out.extend([complex(re, im), complex(re, -im)])
    return np.asarray(out, dtype=complex)


def poly_from_roots(r: Any) -> np.ndarray:
    """Köklerden gerçel, monik polinom (eşlenik çiftler ikinci derece çarpanlar olarak, gerçel aritmetikle)."""
    r = cplxpair(r, tol=1e-9)
    p = np.ones(1)
    i = 0
    while i < r.size:
        if r[i].imag == 0.0:
            p = np.convolve(p, [1.0, -r[i].real])
            i += 1
        else:
            p = np.convolve(p, [1.0, -2.0 * r[i].real, r[i].real ** 2 + r[i].imag ** 2])
            i += 2
    return p


def roots(p: np.ndarray) -> np.ndarray:
    """Polinom kökleri (eşlenik çiftler tam eşlenik)."""
    p = _strip(np.asarray(p, float))
    if p.size <= 1:
        return np.zeros(0, complex)
    r = np.roots(p).astype(complex)
    return cplxpair(r, tol=1e-10)


# ---------------------------------------------------------------------------------------------------------------
# Taban sınıf
# ---------------------------------------------------------------------------------------------------------------
class LTI:
    """Tüm LTI modellerin tabanı (MATLAB `DynamicSystem` / `lti`)."""

    __array_ufunc__ = None      # numpy dizisi soldayken işlemi bize bıraksın
    _priority = 0
    _kind = "lti"

    def __init__(self, ny: int, nu: int, Ts: float = 0.0, *, input_names=None, output_names=None, name: str = "",
                 input_delay=None, output_delay=None):
        self._ny = int(ny)
        self._nu = int(nu)
        self.Ts = _check_Ts(Ts)
        self.input_names = _names(input_names, self._nu, "Giriş")
        self.output_names = _names(output_names, self._ny, "Çıkış")
        self.name = "" if name is None else str(name)
        self.input_delay = _delay_vec(input_delay, self._nu, self.Ts, "Giriş")
        self.output_delay = _delay_vec(output_delay, self._ny, self.Ts, "Çıkış")

    # --- boyutlar -------------------------------------------------------------------------------------------
    @property
    def ninputs(self) -> int:
        return self._nu

    @property
    def noutputs(self) -> int:
        return self._ny

    @property
    def shape(self) -> tuple[int, int]:
        return (self._ny, self._nu)

    @property
    def is_discrete(self) -> bool:
        return self.Ts > 0.0

    @property
    def is_siso(self) -> bool:
        return self._ny == 1 and self._nu == 1

    def _meta(self) -> dict:
        return dict(input_names=self.input_names, output_names=self.output_names, name=self.name,
                    input_delay=self.input_delay.copy(), output_delay=self.output_delay.copy())

    # --- gecikmeler -----------------------------------------------------------------------------------------
    def _io(self) -> np.ndarray:
        return np.zeros(self.shape)

    def total_delay(self) -> np.ndarray:
        """Her G/Ç kanalının toplam gecikmesi (ny, nu): giriş + çıkış + io (sürekli: s, ayrık: örnek)."""
        return self.input_delay[None, :] + self.output_delay[:, None] + self._io()

    def has_delay(self) -> bool:
        return bool(np.any(self.input_delay) or np.any(self.output_delay) or np.any(self._io()))

    def is_static(self) -> bool:
        """Dinamiği ve gecikmesi olmayan saf kazanç mı?"""
        raise NotImplementedError

    def _structure(self) -> np.ndarray:
        """(ny, nu) bool: giriş j çıkış i'yi (yapısal olarak) etkiliyor mu?"""
        raise NotImplementedError

    # --- aritmetik ------------------------------------------------------------------------------------------
    def __add__(self, other):
        return _arith(self, other, "+")

    def __radd__(self, other):
        return _arith(other, self, "+")

    def __sub__(self, other):
        return _arith(self, _negate(other), "+")

    def __rsub__(self, other):
        return _arith(other, -self, "+")

    def __mul__(self, other):
        return _arith(self, other, "*")

    def __rmul__(self, other):
        return _arith(other, self, "*")

    __matmul__ = __mul__
    __rmatmul__ = __rmul__

    def __truediv__(self, other):
        if isinstance(other, LTI):
            return _arith(self, inv(other), "*")
        K = _gain_matrix(other)
        if K.shape == (1, 1):
            if K[0, 0] == 0:
                raise ControlError("Sıfıra bölme.")
            return _arith(self, 1.0 / K[0, 0], "*")
        if K.shape[0] != K.shape[1]:
            raise ControlError("Bölen kazanç matrisi kare olmalı.")
        try:
            return _arith(self, np.linalg.inv(K), "*")
        except np.linalg.LinAlgError:
            raise ControlError("Bölen kazanç matrisi tekil.") from None

    def __rtruediv__(self, other):
        return _arith(other, inv(self), "*")

    def __neg__(self):
        return self._neg()

    def __pos__(self):
        return self.copy()

    def __pow__(self, k):
        if not _is_number(k) or float(k) != int(k):
            raise ControlError("Model yalnız tamsayı kuvvete yükseltilebilir.")
        k = int(k)
        if self._ny != self._nu:
            raise ControlError("Kuvvet yalnız kare (giriş sayısı = çıkış sayısı) modelde tanımlı.")
        if k == 0:
            return self._static(np.eye(self._ny))
        base = self if k > 0 else inv(self)
        k = abs(k)
        result = None
        while k:
            if k & 1:
                result = base if result is None else result * base
            k >>= 1
            if k:
                base = base * base
        return result

    def __iter__(self):
        raise TypeError("LTI modeli üzerinde yinelenemez; alt sistem için sys[i, j] kullanın.")

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 2:
            if self._ny == 1 and not isinstance(key, tuple):
                key = (0, key)
            elif self._nu == 1 and not isinstance(key, tuple):
                key = (key, 0)
            else:
                raise ControlError("Alt sistem için iki indis verin: sys[çıkışlar, girişler].")
        rows = _index(key[0], self._ny)
        cols = _index(key[1], self._nu)
        return self._sub(rows, cols)

    # --- alt sınıfların sağladıkları ------------------------------------------------------------------------
    def _static(self, K: np.ndarray) -> "LTI":
        raise NotImplementedError

    def _retime(self, Ts: float) -> "LTI":
        out = self.copy()
        out.Ts = Ts
        return out

    def _neg(self) -> "LTI":
        raise NotImplementedError

    def _sub(self, rows: np.ndarray, cols: np.ndarray) -> "LTI":
        raise NotImplementedError

    def copy(self) -> "LTI":
        raise NotImplementedError

    def _add(self, other: "LTI") -> "LTI":
        raise NotImplementedError

    def _mul(self, other: "LTI") -> "LTI":
        raise NotImplementedError

    def _expand(self, shape: tuple[int, int]) -> "LTI":
        raise NotImplementedError

    # --- sözlük / JSON -------------------------------------------------------------------------------------
    def to_dict(self) -> dict:
        d = {"type": self._kind, "version": 1, "Ts": self.Ts, "shape": [self._ny, self._nu],
             "input_names": list(self.input_names), "output_names": list(self.output_names), "name": self.name,
             "input_delay": self.input_delay.tolist(), "output_delay": self.output_delay.tolist()}
        d.update(self._data_dict())
        return d

    def _data_dict(self) -> dict:
        raise NotImplementedError

    @staticmethod
    def from_dict(d: dict) -> "LTI":
        """`to_dict` çıktısından model (JSON ile kaydedilmiş modeller)."""
        if not isinstance(d, dict) or "type" not in d:
            raise ControlError("Geçersiz model kaydı.")
        kind = d["type"]
        ny, nu = (int(v) for v in d.get("shape", (1, 1)))
        meta = dict(Ts=d.get("Ts", 0.0), input_names=d.get("input_names"), output_names=d.get("output_names"),
                    name=d.get("name", ""), input_delay=d.get("input_delay"), output_delay=d.get("output_delay"))
        if kind == "tf":
            return TransferFunction(_grid_from_json(d["num"], ny, nu), _grid_from_json(d["den"], ny, nu),
                                    io_delay=d.get("io_delay"), **meta)
        if kind == "zpk":
            zz = [[_cvec_from_json(v) for v in row] for row in d["z"]]
            pp = [[_cvec_from_json(v) for v in row] for row in d["p"]]
            return ZPK(zz, pp, np.asarray(d["k"], float).reshape(ny, nu), io_delay=d.get("io_delay"), **meta)
        if kind == "ss":
            nx = int(d.get("nx", 0))
            A = np.asarray(d["A"], float).reshape(nx, nx)
            B = np.asarray(d["B"], float).reshape(nx, nu)
            C = np.asarray(d["C"], float).reshape(ny, nx)
            D = np.asarray(d["D"], float).reshape(ny, nu)
            return StateSpace(A, B, C, D, state_names=d.get("state_names"), **meta)
        raise ControlError(f"Bilinmeyen model türü: {kind!r}")

    # --- gösterim -------------------------------------------------------------------------------------------
    def __repr__(self) -> str:
        return self.__str__()

    def _footer(self, what: str) -> list[str]:
        lines = []
        if self.is_discrete:
            lines.append(f"Örnekleme zamanı: {_fmt(self.Ts)} s")
            lines.append(f"Ayrık zamanlı {what}.")
        else:
            lines.append(f"Sürekli zamanlı {what}.")
        unit = "örnek" if self.is_discrete else "s"
        if np.any(self.input_delay):
            lines.append(f"Giriş gecikmesi ({unit}): " + ", ".join(_fmt(v) for v in self.input_delay))
        if np.any(self.output_delay):
            lines.append(f"Çıkış gecikmesi ({unit}): " + ", ".join(_fmt(v) for v in self.output_delay))
        return lines


def _index(k: Any, n: int) -> np.ndarray:
    try:
        idx = np.arange(n)[k]
    except (IndexError, TypeError):
        raise ControlError(f"Geçersiz indis: {k!r} (boyut {n}).") from None
    return np.atleast_1d(idx).astype(int)


def _grid_from_json(g, ny, nu):
    return [[np.asarray(g[i][j], float) for j in range(nu)] for i in range(ny)]


def _cvec_from_json(v) -> np.ndarray:
    a = np.asarray(v, float)
    if a.size == 0:
        return np.zeros(0, complex)
    a = a.reshape(-1, 2)
    return a[:, 0] + 1j * a[:, 1]


def _cvec_to_json(v: np.ndarray) -> list:
    return [[float(x.real), float(x.imag)] for x in np.asarray(v, complex)]


# ---------------------------------------------------------------------------------------------------------------
# Transfer fonksiyonu
# ---------------------------------------------------------------------------------------------------------------
def _poly_grid(x: Any, what: str) -> list[list[np.ndarray]]:
    """SISO polinom (sayı / 1-B dizi) ya da MIMO iç içe liste x[i][j] → ızgara."""
    if isinstance(x, np.ndarray) and x.dtype != object:
        if x.ndim <= 1:
            return [[_poly(x, what)]]
        raise ControlError(f"MIMO {what} iç içe liste olarak verilmeli: {what}[i][j] bir polinom.")
    if _is_number(x):
        return [[_poly([x], what)]]
    if not isinstance(x, (list, tuple, np.ndarray)) or len(x) == 0:
        raise ControlError(f"{what} boş ya da geçersiz.")
    if all(_is_number(v) for v in x):
        return [[_poly(np.asarray(x, dtype=complex if any(isinstance(v, complex) for v in x) else float), what)]]
    rows = []
    for row in x:
        if not isinstance(row, (list, tuple, np.ndarray)):
            raise ControlError(f"MIMO {what}: her satır polinomlardan oluşan bir liste olmalı.")
        rows.append([_poly(p, what) for p in row])
    if len({len(r) for r in rows}) != 1 or len(rows[0]) == 0:
        raise ControlError(f"MIMO {what}: bütün satırlar aynı sayıda eleman içermeli.")
    return rows


class TransferFunction(LTI):
    """Transfer fonksiyonu modeli (MATLAB `tf`): `num[i][j]`, `den[i][j]` azalan kuvvetli polinomlar.

    Uygun olmayan (pay derecesi > payda derecesi) tf saklanabilir (MATLAB gibi) ama durum uzayına çevrilemez."""

    _priority = 1
    _kind = "tf"

    def __init__(self, num, den, Ts: float = 0.0, *, io_delay=None, input_delay=None, output_delay=None,
                 input_names=None, output_names=None, name: str = ""):
        N = _poly_grid(num, "Pay")
        Dn = _poly_grid(den, "Payda")
        ny, nu = len(N), len(N[0])
        if (len(Dn), len(Dn[0])) != (ny, nu):
            if len(Dn) == 1 and len(Dn[0]) == 1:
                Dn = [[Dn[0][0].copy() for _ in range(nu)] for _ in range(ny)]
            else:
                raise ControlError(f"Pay ({ny}×{nu}) ve payda ({len(Dn)}×{len(Dn[0])}) boyutları aynı olmalı.")
        for i in range(ny):
            for j in range(nu):
                if _pzero(Dn[i][j]):
                    raise ControlError("Payda sıfır polinom olamaz.")
                if _pzero(N[i][j]):
                    N[i][j] = np.zeros(1)
                    Dn[i][j] = np.ones(1)
        super().__init__(ny, nu, Ts, input_names=input_names, output_names=output_names, name=name,
                         input_delay=input_delay, output_delay=output_delay)
        self._num = N
        self._den = Dn
        self.io_delay = _io_delay(io_delay, ny, nu, self.Ts)

    # --- veri -----------------------------------------------------------------------------------------------
    @property
    def num(self) -> list[list[np.ndarray]]:
        return [[p.copy() for p in row] for row in self._num]

    @property
    def den(self) -> list[list[np.ndarray]]:
        return [[p.copy() for p in row] for row in self._den]

    def tfdata(self, squeeze: bool = False):
        """(num, den); `squeeze=True` ve SISO ise 1-B diziler (MATLAB `tfdata(sys, 'v')`)."""
        if squeeze and self.is_siso:
            return self._num[0][0].copy(), self._den[0][0].copy()
        return self.num, self.den

    def _io(self) -> np.ndarray:
        return self.io_delay

    def copy(self) -> "TransferFunction":
        return TransferFunction(self.num, self.den, self.Ts, io_delay=self.io_delay.copy(), **self._meta())

    def is_static(self) -> bool:
        return (not self.has_delay() and all(n.size == 1 and d.size == 1 for rn, rd in zip(self._num, self._den)
                                              for n, d in zip(rn, rd)))

    def is_proper(self) -> bool:
        return all(n.size <= d.size for rn, rd in zip(self._num, self._den) for n, d in zip(rn, rd))

    def _structure(self) -> np.ndarray:
        return np.array([[not _pzero(n) for n in row] for row in self._num], dtype=bool).reshape(self.shape)

    def _static(self, K: np.ndarray) -> "TransferFunction":
        K = np.atleast_2d(K)
        return TransferFunction([[np.array([v]) for v in row] for row in K], [[np.ones(1)] * K.shape[1]] * K.shape[0],
                                self.Ts)

    def _neg(self) -> "TransferFunction":
        out = self.copy()
        out._num = [[-p for p in row] for row in out._num]
        return out

    def _sub(self, rows, cols) -> "TransferFunction":
        return TransferFunction([[self._num[i][j] for j in cols] for i in rows],
                                [[self._den[i][j] for j in cols] for i in rows], self.Ts,
                                io_delay=self.io_delay[np.ix_(rows, cols)], input_delay=self.input_delay[cols],
                                output_delay=self.output_delay[rows],
                                input_names=[self.input_names[j] for j in cols],
                                output_names=[self.output_names[i] for i in rows])

    def _expand(self, shape) -> "TransferFunction":
        p, m = shape
        return TransferFunction([[self._num[0][0]] * m for _ in range(p)], [[self._den[0][0]] * m for _ in range(p)],
                                self.Ts, io_delay=np.full(shape, self.io_delay[0, 0]),
                                input_delay=np.full(m, self.input_delay[0]),
                                output_delay=np.full(p, self.output_delay[0]))

    def _diag(self, k: int) -> "TransferFunction":
        """SISO modelin k×k köşegen kopyası (skaler çarpım için)."""
        tot = float(self.total_delay()[0, 0])
        N = [[self._num[0][0] if i == j else np.zeros(1) for j in range(k)] for i in range(k)]
        D = [[self._den[0][0] if i == j else np.ones(1) for j in range(k)] for i in range(k)]
        return TransferFunction(N, D, self.Ts, io_delay=np.eye(k) * tot)

    def _add(self, other: "TransferFunction") -> "TransferFunction":
        return _rat_add(self, other)

    def _mul(self, other: "TransferFunction") -> "TransferFunction":
        return _rat_mul(self, other)

    def _data_dict(self) -> dict:
        return {"num": [[p.tolist() for p in row] for row in self._num],
                "den": [[p.tolist() for p in row] for row in self._den], "io_delay": self.io_delay.tolist()}

    def __str__(self) -> str:
        var = "z" if self.is_discrete else "s"
        blocks = []
        for j in range(self._nu):
            for i in range(self._ny):
                body = _frac_lines(_poly_str(self._num[i][j], var), _poly_str(self._den[i][j], var),
                                   _delay_prefix(self.io_delay[i, j], self.is_discrete))
                if self.is_siso:
                    blocks.append(body)
                else:
                    blocks.append([f"  {_io_label(self, i, j)}:"] + ["  " + ln for ln in body])
        lines = [_title(self)] if self.name else []
        for b in blocks:
            lines.append(" ")
            lines.extend(b)
        lines.append(" ")
        lines.extend(self._footer("transfer fonksiyonu"))
        return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------
# Sıfır-kutup-kazanç
# ---------------------------------------------------------------------------------------------------------------
def _root_grid(x: Any, ny: int | None, nu: int | None, what: str) -> list[list[np.ndarray]]:
    def vec(v):
        a = np.asarray(v, dtype=complex).ravel() if not (isinstance(v, (list, tuple)) and len(v) == 0) else \
            np.zeros(0, complex)
        return cplxpair(a, tol=1e-9, what=what)

    if x is None:
        x = []
    if isinstance(x, np.ndarray) and x.dtype != object and x.ndim <= 1:
        return [[vec(x)]]
    if _is_number(x):
        return [[vec([x])]]
    if isinstance(x, (list, tuple)) and (len(x) == 0 or all(_is_number(v) for v in x)):
        if ny is not None and (ny, nu) != (1, 1) and len(x) == 0:
            return [[np.zeros(0, complex) for _ in range(nu)] for _ in range(ny)]
        return [[vec(x)]]
    rows = []
    for row in x:
        if not isinstance(row, (list, tuple, np.ndarray)):
            raise ControlError(f"MIMO {what}: her satır kök dizilerinden oluşan bir liste olmalı.")
        rows.append([vec(v) for v in row])
    if len({len(r) for r in rows}) != 1:
        raise ControlError(f"MIMO {what}: bütün satırlar aynı sayıda eleman içermeli.")
    return rows


class ZPK(LTI):
    """Sıfır-kutup-kazanç modeli (MATLAB `zpk`): G_ij(s) = k_ij · Π(s − z) / Π(s − p)."""

    _priority = 2
    _kind = "zpk"

    def __init__(self, z, p, k, Ts: float = 0.0, *, io_delay=None, input_delay=None, output_delay=None,
                 input_names=None, output_names=None, name: str = ""):
        K = _real_array(np.atleast_2d(k) if np.ndim(k) <= 2 else k, "Kazanç")
        if K.ndim != 2:
            raise ControlError("Kazanç (k) skaler ya da ny×nu matris olmalı.")
        ny, nu = K.shape
        Z = _root_grid(z, ny, nu, "Sıfırlar")
        P = _root_grid(p, ny, nu, "Kutuplar")
        if (len(Z), len(Z[0])) != (ny, nu) or (len(P), len(P[0])) != (ny, nu):
            raise ControlError(f"Sıfır / kutup ızgarası kazanç matrisiyle ({ny}×{nu}) aynı boyutta olmalı.")
        for i in range(ny):
            for j in range(nu):
                if K[i, j] == 0.0:
                    Z[i][j] = np.zeros(0, complex)
                    P[i][j] = np.zeros(0, complex)
        super().__init__(ny, nu, Ts, input_names=input_names, output_names=output_names, name=name,
                         input_delay=input_delay, output_delay=output_delay)
        self._z = Z
        self._p = P
        self._k = K
        self.io_delay = _io_delay(io_delay, ny, nu, self.Ts)

    @property
    def z(self) -> list[list[np.ndarray]]:
        return [[v.copy() for v in row] for row in self._z]

    @property
    def p(self) -> list[list[np.ndarray]]:
        return [[v.copy() for v in row] for row in self._p]

    @property
    def k(self) -> np.ndarray:
        return self._k.copy()

    def zpkdata(self, squeeze: bool = False):
        """(z, p, k); `squeeze=True` ve SISO ise diziler / skaler (MATLAB `zpkdata(sys, 'v')`)."""
        if squeeze and self.is_siso:
            return self._z[0][0].copy(), self._p[0][0].copy(), float(self._k[0, 0])
        return self.z, self.p, self.k

    def _io(self) -> np.ndarray:
        return self.io_delay

    def copy(self) -> "ZPK":
        return ZPK(self.z, self.p, self._k.copy(), self.Ts, io_delay=self.io_delay.copy(), **self._meta())

    def is_static(self) -> bool:
        return not self.has_delay() and all(z.size == 0 and p.size == 0 for rz, rp in zip(self._z, self._p)
                                            for z, p in zip(rz, rp))

    def is_proper(self) -> bool:
        return all(z.size <= p.size for rz, rp in zip(self._z, self._p) for z, p in zip(rz, rp))

    def _structure(self) -> np.ndarray:
        return self._k != 0.0

    def _static(self, K: np.ndarray) -> "ZPK":
        K = np.atleast_2d(K)
        return ZPK([[[] for _ in range(K.shape[1])] for _ in range(K.shape[0])],
                   [[[] for _ in range(K.shape[1])] for _ in range(K.shape[0])], K, self.Ts)

    def _neg(self) -> "ZPK":
        out = self.copy()
        out._k = -out._k
        return out

    def _sub(self, rows, cols) -> "ZPK":
        return ZPK([[self._z[i][j] for j in cols] for i in rows], [[self._p[i][j] for j in cols] for i in rows],
                   self._k[np.ix_(rows, cols)], self.Ts, io_delay=self.io_delay[np.ix_(rows, cols)],
                   input_delay=self.input_delay[cols], output_delay=self.output_delay[rows],
                   input_names=[self.input_names[j] for j in cols], output_names=[self.output_names[i] for i in rows])

    def _expand(self, shape) -> "ZPK":
        p, m = shape
        return ZPK([[self._z[0][0]] * m for _ in range(p)], [[self._p[0][0]] * m for _ in range(p)],
                   np.full(shape, self._k[0, 0]), self.Ts, io_delay=np.full(shape, self.io_delay[0, 0]),
                   input_delay=np.full(m, self.input_delay[0]), output_delay=np.full(p, self.output_delay[0]))

    def _diag(self, k: int) -> "ZPK":
        tot = float(self.total_delay()[0, 0])
        Z = [[self._z[0][0] if i == j else np.zeros(0, complex) for j in range(k)] for i in range(k)]
        P = [[self._p[0][0] if i == j else np.zeros(0, complex) for j in range(k)] for i in range(k)]
        return ZPK(Z, P, np.eye(k) * self._k[0, 0], self.Ts, io_delay=np.eye(k) * tot)

    def _add(self, other: "ZPK") -> "ZPK":
        return _rat_add(self, other)

    def _mul(self, other: "ZPK") -> "ZPK":
        return _rat_mul(self, other)

    def _data_dict(self) -> dict:
        return {"z": [[_cvec_to_json(v) for v in row] for row in self._z],
                "p": [[_cvec_to_json(v) for v in row] for row in self._p],
                "k": self._k.tolist(), "io_delay": self.io_delay.tolist()}

    def __str__(self) -> str:
        var = "z" if self.is_discrete else "s"
        blocks = []
        for j in range(self._nu):
            for i in range(self._ny):
                k = float(self._k[i, j])
                if k == 0.0:
                    body = ["0"]
                else:
                    num = _factors_str(self._z[i][j], var)
                    den = _factors_str(self._p[i][j], var)
                    if abs(k) != 1.0 or not num:
                        num = _fmt(k) + (" " + num if num else "")
                    elif k < 0:
                        num = "-" + num
                    body = _frac_lines(num, den, _delay_prefix(self.io_delay[i, j], self.is_discrete)) if den else \
                        [_delay_prefix(self.io_delay[i, j], self.is_discrete) + num]
                if self.is_siso:
                    blocks.append(body)
                else:
                    blocks.append([f"  {_io_label(self, i, j)}:"] + ["  " + ln for ln in body])
        lines = [_title(self)] if self.name else []
        for b in blocks:
            lines.append(" ")
            lines.extend(b)
        lines.append(" ")
        lines.extend(self._footer("sıfır-kutup-kazanç modeli"))
        return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------
# Durum uzayı
# ---------------------------------------------------------------------------------------------------------------
def _ss_matrices(A, B, C, D):
    A = _real_array(A, "A matrisi")
    if A.size == 0:
        A = np.zeros((0, 0))
    A = np.atleast_2d(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ControlError("A matrisi kare olmalı.")
    n = A.shape[0]
    Dm = _real_array(D, "D matrisi")
    Bm = _real_array(B, "B matrisi") if B is not None else np.zeros((n, 0))
    Cm = _real_array(C, "C matrisi") if C is not None else np.zeros((0, n))
    # B
    if Bm.size == 0:
        m = Dm.shape[1] if Dm.ndim == 2 else (1 if Dm.size == 1 else 0)
        Bm = np.zeros((n, m))
    elif Bm.ndim <= 1:
        if n == 1:
            Bm = Bm.reshape(1, -1)
        elif Bm.size == n:
            Bm = Bm.reshape(n, 1)
        else:
            raise ControlError(f"B matrisi {n} satırlı olmalı.")
    if Bm.ndim != 2 or Bm.shape[0] != n:
        raise ControlError(f"B matrisi {n} satırlı olmalı (A {n}×{n}).")
    m = Bm.shape[1]
    # C
    if Cm.size == 0:
        p = Dm.shape[0] if Dm.ndim == 2 else (1 if Dm.size == 1 else 0)
        Cm = np.zeros((p, n))
    elif Cm.ndim <= 1:
        if n == 1:
            Cm = Cm.reshape(-1, 1)
        elif Cm.size == n:
            Cm = Cm.reshape(1, n)
        else:
            raise ControlError(f"C matrisi {n} sütunlu olmalı.")
    if Cm.ndim != 2 or Cm.shape[1] != n:
        raise ControlError(f"C matrisi {n} sütunlu olmalı (A {n}×{n}).")
    p = Cm.shape[0]
    # D
    if Dm.size == 1 and (p, m) != (1, 1):
        if Dm.ravel()[0] != 0.0:
            raise ControlError(f"D matrisi {p}×{m} olmalı.")
        Dm = np.zeros((p, m))
    elif Dm.size == p * m:
        Dm = Dm.reshape(p, m)
    else:
        raise ControlError(f"D matrisi {p}×{m} olmalı.")
    return A.copy(), Bm.copy(), Cm.copy(), Dm.copy()


class StateSpace(LTI):
    """Durum uzayı modeli (MATLAB `ss`): ẋ = Ax + Bu, y = Cx + Du (ayrıkta x[k+1] = Ax[k] + Bu[k])."""

    _priority = 3
    _kind = "ss"

    def __init__(self, A, B, C, D, Ts: float = 0.0, *, state_names=None, input_delay=None, output_delay=None,
                 input_names=None, output_names=None, name: str = ""):
        A, B, C, D = _ss_matrices(A, B, C, D)
        super().__init__(C.shape[0], B.shape[1], Ts, input_names=input_names, output_names=output_names,
                         name=name, input_delay=input_delay, output_delay=output_delay)
        self.A, self.B, self.C, self.D = A, B, C, D
        self.state_names = _names(state_names, A.shape[0], "Durum")

    @property
    def nx(self) -> int:
        return self.A.shape[0]

    nstates = nx

    def ssdata(self):
        return self.A.copy(), self.B.copy(), self.C.copy(), self.D.copy()

    def copy(self) -> "StateSpace":
        return StateSpace(self.A, self.B, self.C, self.D, self.Ts, state_names=self.state_names, **self._meta())

    def is_static(self) -> bool:
        return self.nx == 0 and not self.has_delay()

    def _structure(self) -> np.ndarray:
        n = self.nx
        S = self.D != 0.0
        if n == 0:
            return S
        adj = (self.A != 0.0)                      # adj[i, j]: x_j → x_i
        reach = (self.B != 0.0).copy()             # (n, m): hangi durumlara ulaşılıyor
        for _ in range(n):
            new = reach | (adj.astype(int) @ reach.astype(int) > 0)
            if np.array_equal(new, reach):
                break
            reach = new
        return S | ((self.C != 0.0).astype(int) @ reach.astype(int) > 0)

    def _static(self, K: np.ndarray) -> "StateSpace":
        K = np.atleast_2d(np.asarray(K, float))
        return StateSpace(np.zeros((0, 0)), np.zeros((0, K.shape[1])), np.zeros((K.shape[0], 0)), K, self.Ts)

    def _neg(self) -> "StateSpace":
        out = self.copy()
        out.C = -out.C
        out.D = -out.D
        return out

    def _sub(self, rows, cols) -> "StateSpace":
        return StateSpace(self.A, self.B[:, cols], self.C[rows, :], self.D[np.ix_(rows, cols)], self.Ts,
                          state_names=self.state_names, input_delay=self.input_delay[cols],
                          output_delay=self.output_delay[rows], input_names=[self.input_names[j] for j in cols],
                          output_names=[self.output_names[i] for i in rows])

    def _expand(self, shape) -> "StateSpace":
        p, m = shape
        return StateSpace(self.A, self.B @ np.ones((1, m)), np.ones((p, 1)) @ self.C, self.D[0, 0] * np.ones((p, m)),
                          self.Ts, state_names=self.state_names, input_delay=np.full(m, self.input_delay[0]),
                          output_delay=np.full(p, self.output_delay[0]))

    def _diag(self, k: int) -> "StateSpace":
        n = self.nx
        A = np.kron(np.eye(k), self.A)
        B = np.kron(np.eye(k), self.B)
        C = np.kron(np.eye(k), self.C)
        D = np.kron(np.eye(k), self.D)
        tot = float(self.total_delay()[0, 0])
        names = tuple(f"{s}_{c + 1}" if s else "" for c in range(k) for s in self.state_names) if n else None
        return StateSpace(A, B, C, D, self.Ts, state_names=names, input_delay=np.full(k, tot))

    def _add(self, other: "StateSpace") -> "StateSpace":
        return _ss_add(self, other)

    def _mul(self, other: "StateSpace") -> "StateSpace":
        return _ss_mul(self, other)

    def _data_dict(self) -> dict:
        return {"nx": self.nx, "A": self.A.tolist(), "B": self.B.tolist(), "C": self.C.tolist(),
                "D": self.D.tolist(), "state_names": list(self.state_names)}

    def __str__(self) -> str:
        xs = [s or f"x{i + 1}" for i, s in enumerate(self.state_names)]
        us = [s or f"u{i + 1}" for i, s in enumerate(self.input_names)]
        ys = [s or f"y{i + 1}" for i, s in enumerate(self.output_names)]
        lines = [_title(self)] if self.name else []
        if self.nx:
            for label, M, rn, cn in (("A", self.A, xs, xs), ("B", self.B, xs, us), ("C", self.C, ys, xs),
                                     ("D", self.D, ys, us)):
                lines.append(" ")
                lines.append(f"  {label} = ")
                lines.extend(_matrix_lines(M, rn, cn))
        else:
            lines.append(" ")
            lines.append("  D = ")
            lines.extend(_matrix_lines(self.D, ys, us))
        lines.append(" ")
        lines.extend(self._footer("durum uzayı modeli" if self.nx else "statik kazanç"))
        return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------
# Dönüştürme ve aritmetik çekirdeği
# ---------------------------------------------------------------------------------------------------------------
def _gain_matrix(x: Any) -> np.ndarray:
    if _is_number(x):
        if isinstance(x, complex) and x.imag != 0:
            raise ControlError("Karmaşık kazanç desteklenmez.")
        return np.array([[float(np.real(x))]])
    a = _real_array(x, "Kazanç matrisi")
    if a.ndim == 0:
        return a.reshape(1, 1)
    if a.ndim != 2:
        raise ControlError("Statik kazanç skaler ya da 2-B matris olmalı.")
    return a


def _negate(x: Any):
    if isinstance(x, LTI):
        return -x
    return -_gain_matrix(x)


def _as_class(sys: LTI, cls: type) -> LTI:
    if isinstance(sys, cls):
        return sys
    from spt.control import convert
    if cls is TransferFunction:
        return convert.to_tf(sys)
    if cls is ZPK:
        return convert.to_zpk(sys)
    if cls is StateSpace:
        return convert.to_ss(sys)
    raise ControlError("Bilinmeyen model türü.")


def common_Ts(*systems: LTI) -> float:
    """Birleştirilecek modellerin ortak Ts'si (statik kazançlar uyar); uyumsuzsa hata."""
    Ts = None
    for s in systems:
        if s.is_static():
            continue
        if Ts is None:
            Ts = s.Ts
        elif not _same(Ts, s.Ts):
            a = "sürekli" if Ts == 0 else f"Ts = {_fmt(Ts)} s"
            b = "sürekli" if s.Ts == 0 else f"Ts = {_fmt(s.Ts)} s"
            raise ControlError(f"Örnekleme zamanları farklı modeller birleştirilemez ({a} ve {b}).")
    if Ts is None:
        Ts = max((s.Ts for s in systems), default=0.0)
    return Ts


def _promote(a: Any, b: Any) -> tuple[LTI, LTI]:
    if not isinstance(a, LTI):
        a = b._static(_gain_matrix(a))
    if not isinstance(b, LTI):
        b = a._static(_gain_matrix(b))
    cls = type(a) if a._priority >= b._priority else type(b)
    a = _as_class(a, cls)
    b = _as_class(b, cls)
    Ts = common_Ts(a, b)
    if a.Ts != Ts:
        a = a._retime(Ts)
    if b.Ts != Ts:
        b = b._retime(Ts)
    return a, b


def _arith(a: Any, b: Any, op: str) -> LTI:
    a, b = _promote(a, b)
    if op == "+":
        if a.shape != b.shape:
            if a.shape == (1, 1):
                a = a._expand(b.shape)
            elif b.shape == (1, 1):
                b = b._expand(a.shape)
            else:
                raise ControlError(f"Toplanan modellerin boyutları uyumsuz: {_shape_str(a.shape)} ve "
                                   f"{_shape_str(b.shape)}.")
        return a._add(b)
    if a.shape[1] != b.shape[0]:
        if a.shape == (1, 1):
            p, m = b.shape
            return a._diag(p)._mul(b) if p <= m else b._mul(a._diag(m))
        if b.shape == (1, 1):
            p, m = a.shape
            return b._diag(p)._mul(a) if p <= m else a._mul(b._diag(m))
        raise ControlError(f"Çarpılan modellerin boyutları uyumsuz: {_shape_str(a.shape)} · {_shape_str(b.shape)} "
                           "(soldakinin giriş sayısı sağdakinin çıkış sayısına eşit olmalı).")
    return a._mul(b)


def _merge(x: tuple[str, ...], y: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(u if (u == v or not v) else (v if not u else "") for u, v in zip(x, y))


# --- tf / zpk (rasyonel) -------------------------------------------------------------------------------------------
def _entry(sys: LTI, i: int, j: int):
    if isinstance(sys, TransferFunction):
        return (sys._num[i][j], sys._den[i][j])
    return (sys._z[i][j], sys._p[i][j], float(sys._k[i, j]))


def _entry_zero(sys: LTI, e) -> bool:
    return _pzero(e[0]) if isinstance(sys, TransferFunction) else e[2] == 0.0


def _entry_absorb(sys: LTI, e, d: float):
    """Ayrık elemana z^-d gecikmesini kutup olarak kat."""
    d = int(round(d))
    if d <= 0:
        return e
    if isinstance(sys, TransferFunction):
        return (e[0], _ppow_shift(e[1], d))
    return (e[0], np.concatenate([e[1], np.zeros(d, complex)]), e[2])


def _entry_add(sys: LTI, a, b):
    if isinstance(sys, TransferFunction):
        return (_padd(_pmul(a[0], b[1]), _pmul(b[0], a[1])), _pmul(a[1], b[1]))
    # zpk: payı çarpanlarından açıp topla (MATLAB gibi; ortak kutuplar sadeleştirilmez)
    na = a[2] * poly_from_roots(a[0])
    nb = b[2] * poly_from_roots(b[0])
    num = _padd(_pmul(na, poly_from_roots(b[1])), _pmul(nb, poly_from_roots(a[1])))
    p = np.concatenate([a[1], b[1]])
    if _pzero(num):
        return (np.zeros(0, complex), np.zeros(0, complex), 0.0)
    return (roots(num), p, float(num[0]))


def _entry_mul(sys: LTI, a, b):
    if isinstance(sys, TransferFunction):
        return (_pmul(a[0], b[0]), _pmul(a[1], b[1]))
    return (np.concatenate([a[0], b[0]]), np.concatenate([a[1], b[1]]), a[2] * b[2])


def _entry_zero_value(sys: LTI):
    if isinstance(sys, TransferFunction):
        return (np.zeros(1), np.ones(1))
    return (np.zeros(0, complex), np.zeros(0, complex), 0.0)


def _build(proto: LTI, E: list[list], io: np.ndarray, Ts: float, **meta) -> LTI:
    ny, nu = len(E), len(E[0]) if E else 0
    if isinstance(proto, TransferFunction):
        return TransferFunction([[e[0] for e in row] for row in E], [[e[1] for e in row] for row in E], Ts,
                                io_delay=io, **meta)
    return ZPK([[e[0] for e in row] for row in E], [[e[1] for e in row] for row in E],
               np.array([[e[2] for e in row] for row in E], float).reshape(ny, nu), Ts, io_delay=io, **meta)


def _rat_add(a: LTI, b: LTI) -> LTI:
    ny, nu = a.shape
    Ts = a.Ts
    if _vec_same(a.input_delay, b.input_delay) and _vec_same(a.output_delay, b.output_delay):
        ind, outd = a.input_delay.copy(), a.output_delay.copy()
        ioa, iob = a._io(), b._io()
    else:
        ind, outd = np.zeros(nu), np.zeros(ny)
        ioa, iob = a.total_delay(), b.total_delay()
    E, io = [], np.zeros((ny, nu))
    for i in range(ny):
        row = []
        for j in range(nu):
            ea, eb = _entry(a, i, j), _entry(b, i, j)
            za, zb = _entry_zero(a, ea), _entry_zero(b, eb)
            if za and zb:
                row.append(_entry_zero_value(a))
                continue
            if za:
                row.append(eb)
                io[i, j] = iob[i, j]
                continue
            if zb:
                row.append(ea)
                io[i, j] = ioa[i, j]
                continue
            ta, tb = float(ioa[i, j]), float(iob[i, j])
            if not _same(ta, tb):
                if Ts == 0:
                    raise ControlError(DELAY_INTERNAL_MSG)
                t = min(ta, tb)
                ea, eb = _entry_absorb(a, ea, ta - t), _entry_absorb(b, eb, tb - t)
                ta = t
            e = _entry_add(a, ea, eb)
            row.append(e)
            io[i, j] = 0.0 if _entry_zero(a, e) else ta
        E.append(row)
    return _build(a, E, io, Ts, input_delay=ind, output_delay=outd,
                  input_names=_merge(a.input_names, b.input_names),
                  output_names=_merge(a.output_names, b.output_names))


def _rat_mul(a: LTI, b: LTI) -> LTI:
    """a · b (önce b, sonra a): iç gecikmeler (a girişi + b çıkışı + io'lar) G/Ç gecikmesine katlanır."""
    p, q = a.shape
    m = b.shape[1]
    Ts = a.Ts
    ioa, iob = a._io(), b._io()
    E, io = [], np.zeros((p, m))
    for i in range(p):
        row = []
        for j in range(m):
            terms = []
            for k in range(q):
                ea, eb = _entry(a, i, k), _entry(b, k, j)
                if _entry_zero(a, ea) or _entry_zero(b, eb):
                    continue
                terms.append((float(ioa[i, k] + a.input_delay[k] + b.output_delay[k] + iob[k, j]), ea, eb))
            if not terms:
                row.append(_entry_zero_value(a))
                continue
            dmin = min(t[0] for t in terms)
            if any(not _same(t[0], dmin) for t in terms):
                if Ts == 0:
                    raise ControlError(DELAY_INTERNAL_MSG)
            acc = None
            for d, ea, eb in terms:
                e = _entry_mul(a, ea, eb)
                if not _same(d, dmin):
                    e = _entry_absorb(a, e, d - dmin)
                acc = e if acc is None else _entry_add(a, acc, e)
            row.append(acc)
            io[i, j] = 0.0 if _entry_zero(a, acc) else dmin
        E.append(row)
    return _build(a, E, io, Ts, input_delay=b.input_delay.copy(), output_delay=a.output_delay.copy(),
                  input_names=b.input_names, output_names=a.output_names)


# --- durum uzayı ---------------------------------------------------------------------------------------------------
def factor_delays(T: np.ndarray, mask: np.ndarray, tol: float = 1e-12):
    """T[i, j] = in[j] + out[i] (mask'teki elemanlarda) olacak negatif olmayan (in, out) bulur.

    Her bağlı bileşende gecikme mümkün olduğunca girişe yüklenir (MATLAB'in tercihi gibi). Bulunamazsa None."""
    p, m = T.shape
    ind = np.full(m, np.nan)
    outd = np.full(p, np.nan)
    for start in range(p):
        if not np.isnan(outd[start]) or not mask[start].any():
            continue
        outd[start] = 0.0
        comp_rows, comp_cols = [start], []
        stack = [("r", start)]
        while stack:
            kind, idx = stack.pop()
            if kind == "r":
                for j in np.flatnonzero(mask[idx]):
                    v = T[idx, j] - outd[idx]
                    if np.isnan(ind[j]):
                        ind[j] = v
                        comp_cols.append(j)
                        stack.append(("c", j))
                    elif abs(ind[j] - v) > tol * max(1.0, abs(v)):
                        return None
            else:
                for i in np.flatnonzero(mask[:, idx]):
                    v = T[i, idx] - ind[idx]
                    if np.isnan(outd[i]):
                        outd[i] = v
                        comp_rows.append(i)
                        stack.append(("r", i))
                    elif abs(outd[i] - v) > tol * max(1.0, abs(v)):
                        return None
        c = -min(outd[comp_rows])
        if c > min(ind[comp_cols]) + tol * max(1.0, abs(c)):
            return None
        ind[comp_cols] -= c
        outd[comp_rows] += c
    ind = np.where(np.isnan(ind), 0.0, np.maximum(ind, 0.0))
    outd = np.where(np.isnan(outd), 0.0, np.maximum(outd, 0.0))
    return ind, outd


def ss_absorb(sys: "StateSpace", inputs: bool = True, outputs: bool = True) -> "StateSpace":
    """Ayrık modelin giriş / çıkış gecikmelerini z⁻¹ durumları olarak modele katar (tam)."""
    if not sys.is_discrete:
        raise ControlError("Gecikme yalnız ayrık modelde durumlara katılabilir; sürekli zamanda pade kullanın.")
    A, B, C, D = sys.A.copy(), sys.B.copy(), sys.C.copy(), sys.D.copy()
    names = list(sys.state_names)
    ind = sys.input_delay.copy()
    outd = sys.output_delay.copy()
    if inputs:
        for j in range(sys.ninputs):
            d = int(round(ind[j]))
            if d <= 0:
                continue
            n = A.shape[0]
            An = np.zeros((n + d, n + d))
            An[:n, :n] = A
            An[:n, n + d - 1] = B[:, j]
            for i in range(1, d):
                An[n + i, n + i - 1] = 1.0
            Bn = np.vstack([B, np.zeros((d, B.shape[1]))])
            Bn[:n, j] = 0.0
            Bn[n, j] = 1.0
            Cn = np.hstack([C, np.zeros((C.shape[0], d))])
            Cn[:, n + d - 1] = D[:, j]
            D[:, j] = 0.0
            A, B, C = An, Bn, Cn
            names += [""] * d
            ind[j] = 0.0
    if outputs:
        for i in range(sys.noutputs):
            d = int(round(outd[i]))
            if d <= 0:
                continue
            n = A.shape[0]
            An = np.zeros((n + d, n + d))
            An[:n, :n] = A
            An[n, :n] = C[i]
            for k in range(1, d):
                An[n + k, n + k - 1] = 1.0
            Bn = np.vstack([B, np.zeros((d, B.shape[1]))])
            Bn[n] = D[i]
            Cn = np.hstack([C, np.zeros((C.shape[0], d))])
            Cn[i] = 0.0
            Cn[i, n + d - 1] = 1.0
            D[i] = 0.0
            A, B, C = An, Bn, Cn
            names += [""] * d
            outd[i] = 0.0
    return StateSpace(A, B, C, D, sys.Ts, state_names=names, input_delay=ind, output_delay=outd,
                      input_names=sys.input_names, output_names=sys.output_names, name=sys.name)


def _blkdiag(*Ms: np.ndarray) -> np.ndarray:
    r = sum(M.shape[0] for M in Ms)
    c = sum(M.shape[1] for M in Ms)
    out = np.zeros((r, c))
    i = j = 0
    for M in Ms:
        out[i:i + M.shape[0], j:j + M.shape[1]] = M
        i += M.shape[0]
        j += M.shape[1]
    return out


def _ss_add(a: StateSpace, b: StateSpace) -> StateSpace:
    if b.is_static() and not b.D.any():
        return a.copy()
    if a.is_static() and not a.D.any():
        return b.copy()
    ind, outd = a.input_delay, a.output_delay
    if not (_vec_same(a.input_delay, b.input_delay) and _vec_same(a.output_delay, b.output_delay)):
        Sa, Sb = a._structure(), b._structure()
        Ta, Tb = a.total_delay(), b.total_delay()
        both = Sa & Sb
        res = None
        if np.all(np.abs(Ta[both] - Tb[both]) <= 1e-12 * np.maximum(1.0, Ta[both])):
            res = factor_delays(np.where(Sa, Ta, Tb), Sa | Sb)
        if res is None:
            if not a.is_discrete:
                raise ControlError(DELAY_INTERNAL_MSG)
            a, b = ss_absorb(a), ss_absorb(b)
            ind, outd = a.input_delay, a.output_delay
        else:
            ind, outd = res
    return StateSpace(_blkdiag(a.A, b.A), np.vstack([a.B, b.B]), np.hstack([a.C, b.C]), a.D + b.D, a.Ts,
                      state_names=a.state_names + b.state_names, input_delay=ind, output_delay=outd,
                      input_names=_merge(a.input_names, b.input_names),
                      output_names=_merge(a.output_names, b.output_names))


def _ss_mul(a: StateSpace, b: StateSpace) -> StateSpace:
    """a · b (önce b): x = [x_a; x_b]."""
    e = a.input_delay + b.output_delay
    ind, outd = b.input_delay.copy(), a.output_delay.copy()
    if np.any(e > 0):
        Sa, Sb = a._structure(), b._structure()
        p, m = a.noutputs, b.ninputs
        T = np.zeros((p, m))
        mask = np.zeros((p, m), bool)
        ok = True
        for i in range(p):
            for j in range(m):
                ks = np.flatnonzero(Sa[i] & Sb[:, j])
                if ks.size == 0:
                    continue
                if np.ptp(e[ks]) > 1e-12 * max(1.0, float(np.max(e[ks]))):
                    ok = False
                    break
                mask[i, j] = True
                T[i, j] = outd[i] + e[ks[0]] + ind[j]
            if not ok:
                break
        res = factor_delays(T, mask) if ok else None
        if res is None:
            if not a.is_discrete:
                raise ControlError(DELAY_INTERNAL_MSG)
            a = ss_absorb(a, inputs=True, outputs=False)
            b = ss_absorb(b, inputs=False, outputs=True)
        else:
            ind, outd = res
    na, nb = a.nx, b.nx
    A = np.zeros((na + nb, na + nb))
    A[:na, :na] = a.A
    A[:na, na:] = a.B @ b.C
    A[na:, na:] = b.A
    B = np.vstack([a.B @ b.D, b.B])
    C = np.hstack([a.C, a.D @ b.C])
    D = a.D @ b.D
    return StateSpace(A, B, C, D, a.Ts, state_names=a.state_names + b.state_names, input_delay=ind,
                      output_delay=outd, input_names=b.input_names, output_names=a.output_names)


# ---------------------------------------------------------------------------------------------------------------
# Fabrika fonksiyonları (MATLAB adlarıyla)
# ---------------------------------------------------------------------------------------------------------------
_META_KEYS = ("input_names", "output_names", "name", "input_delay", "output_delay")


def _apply_meta(sys: LTI, kw: dict) -> LTI:
    if not kw:
        return sys
    sys = sys.copy()
    for key, val in kw.items():
        if key == "input_names":
            sys.input_names = _names(val, sys.ninputs, "Giriş")
        elif key == "output_names":
            sys.output_names = _names(val, sys.noutputs, "Çıkış")
        elif key == "name":
            sys.name = str(val)
        elif key == "input_delay":
            sys.input_delay = _delay_vec(val, sys.ninputs, sys.Ts, "Giriş")
        elif key == "output_delay":
            sys.output_delay = _delay_vec(val, sys.noutputs, sys.Ts, "Çıkış")
        elif key == "io_delay" and isinstance(sys, (TransferFunction, ZPK)):
            sys.io_delay = _io_delay(val, sys.noutputs, sys.ninputs, sys.Ts)
        elif key == "state_names" and isinstance(sys, StateSpace):
            sys.state_names = _names(val, sys.nx, "Durum")
        else:
            raise ControlError(f"Bilinmeyen model özelliği: {key}")
    return sys


def tf(*args, **kw) -> TransferFunction:
    """Transfer fonksiyonu (MATLAB `tf`).

    - `tf(num, den, Ts=0)` — SISO: 1-B katsayılar; MIMO: `num[i][j]`, `den[i][j]` iç içe listeler.
    - `tf('s')`, `tf('z', Ts)` — Laplace / z değişkeni (`G = 1 / (s**2 + 2*s + 1)`).
    - `tf(K)` — statik kazanç (skaler ya da matris); `tf(sys)` — başka modelden dönüştürme.
    Anahtar sözcükler: `Ts`, `io_delay`, `input_delay`, `output_delay`, `input_names`, `output_names`, `name`."""
    Ts = kw.pop("Ts", None)
    if len(args) == 1 and isinstance(args[0], LTI):
        out = _as_class(args[0], TransferFunction)
        if out is args[0]:
            out = out.copy()
        return _apply_meta(out, kw)
    if len(args) in (1, 2) and isinstance(args[0], str):
        var = args[0]
        if len(args) == 2:
            Ts = args[1]
        if var == "s":
            if Ts not in (None, 0, 0.0):
                raise ControlError("'s' değişkeni sürekli zamandır; ayrık zaman için tf('z', Ts) kullanın.")
            return _apply_meta(TransferFunction([1.0, 0.0], [1.0], 0.0), kw)
        if var in ("z", "q"):
            if Ts is None or float(Ts) <= 0:
                raise ControlError("'z' değişkeni için örnekleme zamanı (Ts > 0) verin: tf('z', Ts).")
            return _apply_meta(TransferFunction([1.0, 0.0], [1.0], Ts), kw)
        raise ControlError(f"Bilinmeyen değişken: {var!r} ('s' ya da 'z').")
    if len(args) == 1:
        K = _gain_matrix(args[0])
        return _apply_meta(TransferFunction(np.ones(1), np.ones(1))._static(K)._retime(Ts or 0.0), kw)
    if len(args) == 3:
        if Ts is not None:
            raise ControlError("Ts iki kez verildi.")
        Ts = args[2]
    elif len(args) != 2:
        raise ControlError("tf(num, den[, Ts]) biçiminde çağırın.")
    return TransferFunction(args[0], args[1], 0.0 if Ts is None else Ts, **kw)


def zpk(*args, **kw) -> ZPK:
    """Sıfır-kutup-kazanç modeli (MATLAB `zpk`).

    - `zpk(z, p, k, Ts=0)` — SISO: sıfır / kutup dizileri ve kazanç; MIMO: `z[i][j]`, `p[i][j]`, `k` (ny×nu).
    - `zpk('s')`, `zpk('z', Ts)`; `zpk(K)` statik kazanç; `zpk(sys)` dönüştürme."""
    Ts = kw.pop("Ts", None)
    if len(args) == 1 and isinstance(args[0], LTI):
        out = _as_class(args[0], ZPK)
        if out is args[0]:
            out = out.copy()
        return _apply_meta(out, kw)
    if len(args) in (1, 2) and isinstance(args[0], str):
        return zpk(tf(*args, **({} if Ts is None else {"Ts": Ts})), **kw)
    if len(args) == 1:
        K = _gain_matrix(args[0])
        return _apply_meta(ZPK([], [], 1.0)._static(K)._retime(Ts or 0.0), kw)
    if len(args) == 4:
        if Ts is not None:
            raise ControlError("Ts iki kez verildi.")
        Ts = args[3]
    elif len(args) != 3:
        raise ControlError("zpk(z, p, k[, Ts]) biçiminde çağırın.")
    return ZPK(args[0], args[1], args[2], 0.0 if Ts is None else Ts, **kw)


def ss(*args, **kw) -> StateSpace:
    """Durum uzayı modeli (MATLAB `ss`).

    - `ss(A, B, C, D, Ts=0)`; `ss(D)` statik kazanç; `ss(sys)` dönüştürme (tf / zpk gerçeklemesi);
      `ss(sys, 'minimal')` en küçük gerçekleme (MATLAB gibi)."""
    Ts = kw.pop("Ts", None)
    if args and isinstance(args[0], LTI):
        if len(args) == 2 and args[1] == "minimal":
            from spt.control.convert import minreal
            return _apply_meta(minreal(_as_class(args[0], StateSpace)), kw)
        if len(args) != 1:
            raise ControlError("ss(sys) ya da ss(sys, 'minimal') biçiminde çağırın.")
        out = _as_class(args[0], StateSpace)
        if out is args[0]:
            out = out.copy()
        return _apply_meta(out, kw)
    if len(args) == 1:
        K = _gain_matrix(args[0])
        return _apply_meta(StateSpace(np.zeros((0, 0)), None, None, K, Ts or 0.0), kw)
    if len(args) == 5:
        if Ts is not None:
            raise ControlError("Ts iki kez verildi.")
        Ts = args[4]
    elif len(args) != 4:
        raise ControlError("ss(A, B, C, D[, Ts]) biçiminde çağırın.")
    return StateSpace(*args[:4], 0.0 if Ts is None else Ts, **kw)


def inv(sys: Any) -> LTI:
    """Modelin tersi (MATLAB `inv`): kare model; durum uzayında D tersinir olmalı. Gecikmeli modelin tersi
    nedensel olmadığından hata verir."""
    if not isinstance(sys, LTI):
        K = _gain_matrix(sys)
        try:
            return np.linalg.inv(K) if K.shape != (1, 1) else 1.0 / K[0, 0]
        except (np.linalg.LinAlgError, ZeroDivisionError):
            raise ControlError("Kazanç matrisi tekil; tersi alınamaz.") from None
    if sys.noutputs != sys.ninputs:
        raise ControlError("Tersi yalnız kare (giriş sayısı = çıkış sayısı) modelde tanımlı.")
    if sys.has_delay():
        raise ControlError("Gecikmeli modelin tersi bir öngörü (negatif gecikme) gerektirir; nedensel değildir.")
    names = dict(input_names=sys.output_names, output_names=sys.input_names)
    if isinstance(sys, TransferFunction) and sys.is_siso:
        n, d = sys._num[0][0], sys._den[0][0]
        if _pzero(n):
            raise ControlError("Sıfır transfer fonksiyonunun tersi yok.")
        return TransferFunction(d, n, sys.Ts, **names)
    if isinstance(sys, ZPK) and sys.is_siso:
        k = float(sys._k[0, 0])
        if k == 0.0:
            raise ControlError("Sıfır modelin tersi yok.")
        return ZPK(sys._p[0][0], sys._z[0][0], 1.0 / k, sys.Ts, **names)
    s = _as_class(sys, StateSpace)
    try:
        Di = np.linalg.inv(s.D)
    except np.linalg.LinAlgError:
        raise ControlError("D matrisi tekil olduğundan durum uzayı modelinin tersi uygun (proper) değil.") from None
    if np.linalg.cond(s.D) > 1e12:
        raise ControlError("D matrisi tekile çok yakın; ters model sayısal olarak güvenilmez.")
    out = StateSpace(s.A - s.B @ Di @ s.C, s.B @ Di, -Di @ s.C, Di, s.Ts, state_names=s.state_names, **names)
    return _as_class(out, type(sys)) if not isinstance(sys, StateSpace) else out


# ---------------------------------------------------------------------------------------------------------------
# Gösterim (MATLAB düzenine benzer)
# ---------------------------------------------------------------------------------------------------------------
def _fmt(x: float) -> str:
    x = float(x)
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    s = f"{x:.4g}"
    if "e" in s:
        m, e = s.split("e")
        e = int(e)
        s = f"{m}e{'-' if e < 0 else ''}{abs(e):02d}"
    return s


def _poly_str(p: np.ndarray, var: str) -> str:
    p = _strip(np.asarray(p, float))
    n = p.size - 1
    terms = []
    for k, c in enumerate(p):
        if c == 0.0:
            continue
        power = n - k
        v = "" if power == 0 else (var if power == 1 else f"{var}^{power}")
        mag = abs(c)
        coef = "" if (mag == 1.0 and v) else _fmt(mag)
        body = f"{coef} {v}".strip() if coef and v else (coef or v)
        if not terms:
            terms.append(("-" if c < 0 else "") + body)
        else:
            terms.append((" - " if c < 0 else " + ") + body)
    return "".join(terms) if terms else "0"


def _factors_str(r: np.ndarray, var: str) -> str:
    """zpk çarpanları: (s+2) (s^2 + 6s + 25); tekrarlılar (s+1)^2."""
    r = np.asarray(r, complex)
    facs: list[str] = []
    i = 0
    # MATLAB sırası: önce orijindeki kökler, sonra gerçel çarpanlar, sonra ikinci derece çarpanlar
    rr = [x for x in r if x == 0] + [x for x in r if x.imag == 0 and x != 0] + [x for x in r if x.imag != 0]
    while i < len(rr):
        x = rr[i]
        if x.imag == 0.0:
            v = x.real
            facs.append(var if v == 0.0 else (f"({var}-{_fmt(v)})" if v > 0 else f"({var}+{_fmt(-v)})"))
            i += 1
        else:
            b = -2.0 * x.real
            c = x.real ** 2 + x.imag ** 2
            mid = "" if b == 0 else (f" + {_fmt(b)}{var}" if b > 0 else f" - {_fmt(-b)}{var}")
            facs.append(f"({var}^2{mid} + {_fmt(c)})")
            i += 2
    out: list[str] = []
    for f in facs:
        if f not in out:
            cnt = facs.count(f)
            out.append(f if cnt == 1 else f"{f}^{cnt}")
    return " ".join(out)


def _delay_prefix(d: float, discrete: bool) -> str:
    if d == 0:
        return ""
    return f"z^(-{_fmt(d)}) * " if discrete else f"exp(-{_fmt(d)}*s) * "


def _frac_lines(num: str, den: str, prefix: str = "") -> list[str]:
    if den in ("1", ""):
        return ["  " + prefix + num]
    w = max(len(num), len(den))
    pad = " " * len(prefix)
    return ["  " + pad + " " * ((w - len(num)) // 2) + num,
            "  " + prefix + "-" * w,
            "  " + pad + " " * ((w - len(den)) // 2) + den]


def _io_label(sys: LTI, i: int, j: int) -> str:
    u = sys.input_names[j] or f"giriş {j + 1}"
    y = sys.output_names[i] or f"çıkış {i + 1}"
    return f"{u} → {y}"


def _title(sys: LTI) -> str:
    return f"{sys.name} ="


def _matrix_lines(M: np.ndarray, rn: list[str], cn: list[str]) -> list[str]:
    cells = [[_fmt(v) for v in row] for row in M]
    w = max([len(c) for c in cn] + [len(c) for row in cells for c in row] + [1])
    rw = max([len(r) for r in rn] + [1])
    lines = ["   " + " " * rw + "".join(f"  {c:>{w}}" for c in cn)]
    for r, row in zip(rn, cells):
        lines.append("   " + f"{r:<{rw}}" + "".join(f"  {c:>{w}}" for c in row))
    return lines
