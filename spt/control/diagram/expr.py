"""Blok diyagram için güvenli ifade değerlendirici (İfade bloğu, ifade tabanlı durum uzayı, sayısal parametreler).

Python `eval` / `compile` kullanılmaz: ifade `ast` ile çözümlenir, beyaz listeyle doğrulanır (yalnız sayı sabitleri,
izinli adlar / fonksiyonlar ve aritmetik / karşılaştırma) ve skaler `float` kapanışlarına (closure) derlenir. Benzetimde
her küçük adımda çağrıldığı için numpy değil `math` kullanılır (hızlı skaler hesap).

MATLAB / Simulink Fcn bloğu alışkanlıkları desteklenir:
- `^` üs (`**` ile aynı), `~=` eşit değil, `&&` / `||` / `&` / `|` mantıksal ve / veya, `~` değil;
- vektör değişkenler 1 tabanlı indekslenir: `u(1)`, `x(3)` (Simulink Fcn bloğu gibi); kısaltma `u1`, `x3`;
- karşılaştırmalar 1.0 / 0.0 üretir.
Tanımsız sonuç (sıfıra bölme, log(0), negatif sayının kesirli üssü …) `ControlError` yükseltir; karmaşık sayı asla
üretilmez.
"""
from __future__ import annotations

import ast
import functools
import math
import re
from typing import Any, Callable, Iterable

import numpy as np

from spt.control.errors import ControlError

MAX_EXPR_LEN = 2000
_MAX_NODES = 1000
_MAX_DEPTH = 200
MAX_EXPONENT = 1000.0

CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e, "inf": math.inf, "Inf": math.inf}


def _round_half_away(x: float) -> float:
    return math.copysign(math.floor(abs(x) + 0.5), x)


def _sign(x: float) -> float:
    return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)


def _mod(a: float, b: float) -> float:
    """MATLAB mod: sonuç b'nin işaretinde; mod(a, 0) = a."""
    if b == 0:
        return a
    return a - math.floor(a / b) * b


def _rem(a: float, b: float) -> float:
    """MATLAB rem: sonuç a'nın işaretinde; rem(a, 0) = NaN → tanımsız."""
    if b == 0:
        raise ValueError("rem(a, 0)")
    return math.fmod(a, b)


def _sat(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def _pow(a: float, b: float) -> float:
    return math.pow(a, b)


# ad → (fonksiyon, en az, en çok argüman, açıklama)
FUNCTIONS: dict[str, tuple[Callable[..., float], int, int, str]] = {
    "sin": (math.sin, 1, 1, "sinüs (radyan)"),
    "cos": (math.cos, 1, 1, "kosinüs (radyan)"),
    "tan": (math.tan, 1, 1, "tanjant (radyan)"),
    "asin": (math.asin, 1, 1, "ters sinüs"),
    "acos": (math.acos, 1, 1, "ters kosinüs"),
    "atan": (math.atan, 1, 1, "ters tanjant"),
    "atan2": (math.atan2, 2, 2, "atan2(y, x), dört bölgeli"),
    "sinh": (math.sinh, 1, 1, "hiperbolik sinüs"),
    "cosh": (math.cosh, 1, 1, "hiperbolik kosinüs"),
    "tanh": (math.tanh, 1, 1, "hiperbolik tanjant"),
    "exp": (math.exp, 1, 1, "e üssü"),
    "log": (math.log, 1, 1, "doğal logaritma"),
    "log10": (math.log10, 1, 1, "10 tabanında logaritma"),
    "log2": (math.log2, 1, 1, "2 tabanında logaritma"),
    "sqrt": (math.sqrt, 1, 1, "karekök"),
    "abs": (abs, 1, 1, "mutlak değer"),
    "sign": (_sign, 1, 1, "işaret (−1, 0, 1)"),
    "floor": (lambda x: float(math.floor(x)), 1, 1, "aşağı yuvarla"),
    "ceil": (lambda x: float(math.ceil(x)), 1, 1, "yukarı yuvarla"),
    "round": (_round_half_away, 1, 1, "en yakına yuvarla (yarım sıfırdan uzağa, MATLAB gibi)"),
    "fix": (lambda x: float(math.trunc(x)), 1, 1, "sıfıra doğru yuvarla"),
    "mod": (_mod, 2, 2, "mod(a, b), sonuç b'nin işaretinde (MATLAB)"),
    "rem": (_rem, 2, 2, "rem(a, b), sonuç a'nın işaretinde (MATLAB)"),
    "min": (min, 2, 16, "en küçük"),
    "max": (max, 2, 16, "en büyük"),
    "hypot": (math.hypot, 2, 2, "√(x² + y²)"),
    "pow": (_pow, 2, 2, "pow(a, b) = a^b"),
    "sat": (_sat, 3, 3, "sat(x, alt, üst): doyum"),
}

_BINOPS: dict[type, Callable[[float, float], float]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: _pow,
    ast.Mod: _mod,
}
_CMP: dict[type, Callable[[float, float], bool]] = {
    ast.Lt: lambda a, b: a < b, ast.LtE: lambda a, b: a <= b, ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b, ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b,
}
_FORBIDDEN: dict[type, str] = {
    ast.Attribute: "öznitelik erişimi (nokta '.') kullanılamaz",
    ast.Subscript: "köşeli parantezle indeksleme kullanılamaz; u(1) biçimini kullanın",
    ast.Lambda: "lambda kullanılamaz",
    ast.Tuple: "virgülle ayrılmış ifade kullanılamaz",
    ast.List: "liste kullanılamaz",
    ast.Dict: "sözlük kullanılamaz",
    ast.Set: "küme kullanılamaz",
    ast.JoinedStr: "metin kullanılamaz",
    ast.NamedExpr: "':=' ataması kullanılamaz",
    ast.Starred: "'*' açma kullanılamaz",
}


def _translate(text: str) -> str:
    """MATLAB sözdizimini Python'a çevirir (uzunluğu koruyarak değil; konumlar yaklaşık)."""
    s = text.replace("\r", " ").replace("\n", " ")
    s = s.replace("~=", "!=").replace("&&", " and ").replace("||", " or ")
    s = s.replace("&", " and ").replace("|", " or ").replace("^", "**")
    s = re.sub(r"~(?!=)", " not ", s)
    return s


class Expression:
    """Doğrulanmış ve derlenmiş skaler ifade.

    `scalars`: skaler değişken adları (ör. ("t",)); `vectors`: ad → genişlik (ör. {"u": 3}); genişlik None ise
    indeks sınırı denetlenmez (genişlik sonra belirlenir, `check_widths` ile denetlenir). `u1`, `u2` … kısaltmaları
    vektör değişkenler için kendiliğinden tanınır; genişliği 1 olan vektör değişken yalın `u` olarak da kullanılabilir.
    """

    def __init__(self, text: str, scalars: Iterable[str] = (), vectors: dict[str, int | None] | None = None):
        if not isinstance(text, str):
            raise ControlError("İfade metin olmalı.")
        if len(text) > MAX_EXPR_LEN:
            raise ControlError(f"İfade çok uzun ({len(text)} karakter; en fazla {MAX_EXPR_LEN}).")
        self.text = text
        self.scalars = tuple(scalars)
        self.vectors = dict(vectors or {})
        src = _translate(text).strip()
        if not src:
            raise ControlError("İfade boş.")
        try:
            tree = ast.parse(src, mode="eval")
        except SyntaxError as exc:
            raise ControlError(f"İfadede sözdizimi hatası: {exc.msg} ('{text}').") from None
        except (ValueError, RecursionError, MemoryError):
            raise ControlError(f"İfade çözümlenemedi ('{text}').") from None
        self._count = 0
        self.indices: dict[str, int] = {}          # vektör değişken → kullanılan en büyük indeks (1 tabanlı)
        self.uses_bare: set[str] = set()            # yalın kullanılan vektör değişkenler (genişlik 1 olmalı)
        self.used: set[str] = set()
        self._fn = self._compile(tree.body, 0)

    # ------------------------------------------------------------------ derleme
    def _fail(self, msg: str) -> None:
        raise ControlError(f"{msg} ('{self.text}').")

    def _compile(self, node: ast.AST, depth: int) -> Callable[[dict], float]:
        self._count += 1
        if self._count > _MAX_NODES:
            self._fail(f"İfade çok karmaşık (en fazla {_MAX_NODES} öğe)")
        if depth > _MAX_DEPTH:
            self._fail("İfade çok derin iç içe")
        d = depth + 1
        if isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                self._fail("Yalnız gerçel sayı sabitleri kullanılabilir")
            try:
                fv = float(v)
            except OverflowError:
                fv = math.inf
            if not math.isfinite(fv):
                self._fail("Sayı çok büyük")
            return lambda env, _v=fv: _v
        if isinstance(node, ast.Name):
            name = node.id
            if name in CONSTANTS:
                return lambda env, _v=CONSTANTS[name]: _v
            if name in self.scalars:
                self.used.add(name)
                return lambda env, _n=name: env[_n]
            if name in self.vectors:
                self.used.add(name)
                self.uses_bare.add(name)
                self.indices[name] = max(self.indices.get(name, 0), 1)
                return lambda env, _n=name: env[_n][0]
            m = re.fullmatch(r"([A-Za-z_]+)(\d+)", name)
            if m and m.group(1) in self.vectors and int(m.group(2)) >= 1:
                var, idx = m.group(1), int(m.group(2))
                self.used.add(var)
                self.indices[var] = max(self.indices.get(var, 0), idx)
                return lambda env, _n=var, _i=idx - 1: env[_n][_i]
            if name in FUNCTIONS:
                self._fail(f"'{name}' bir fonksiyondur; {name}(…) biçiminde çağırın")
            allowed = list(self.scalars) + [f"{v}(i)" for v in self.vectors] + ["pi", "e"]
            self._fail(f"Bilinmeyen ad '{name}'; kullanılabilir adlar: {', '.join(allowed)}")
        if isinstance(node, ast.BinOp):
            op = _BINOPS.get(type(node.op))
            if op is None:
                self._fail("Bu işleç kullanılamaz (yalnız + − * / ^ %)")
            a, b = self._compile(node.left, d), self._compile(node.right, d)
            if isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Constant):
                ev = node.right.value
                if isinstance(ev, (int, float)) and abs(float(ev)) > MAX_EXPONENT:
                    self._fail(f"Üs çok büyük (|üs| ≤ {MAX_EXPONENT:g})")
            return lambda env, _a=a, _b=b, _op=op: _op(_a(env), _b(env))
        if isinstance(node, ast.UnaryOp):
            a = self._compile(node.operand, d)
            if isinstance(node.op, ast.USub):
                return lambda env, _a=a: -_a(env)
            if isinstance(node.op, ast.UAdd):
                return a
            if isinstance(node.op, ast.Not):
                return lambda env, _a=a: 1.0 if _a(env) == 0 else 0.0
            self._fail("Bu tekli işleç kullanılamaz")
        if isinstance(node, ast.Compare):
            parts = [self._compile(node.left, d)] + [self._compile(c, d) for c in node.comparators]
            ops = []
            for op in node.ops:
                f = _CMP.get(type(op))
                if f is None:
                    self._fail("Bu karşılaştırma kullanılamaz (yalnız < <= > >= == ~=)")
                ops.append(f)

            def cmp(env, _p=parts, _o=ops):
                left = _p[0](env)
                for f, nxt in zip(_o, _p[1:]):
                    right = nxt(env)
                    if not f(left, right):
                        return 0.0
                    left = right
                return 1.0
            return cmp
        if isinstance(node, ast.BoolOp):
            vals = [self._compile(v, d) for v in node.values]
            if isinstance(node.op, ast.And):
                return lambda env, _v=vals: 1.0 if all(f(env) != 0 for f in _v) else 0.0
            return lambda env, _v=vals: 1.0 if any(f(env) != 0 for f in _v) else 0.0
        if isinstance(node, ast.IfExp):
            c, a, b = self._compile(node.test, d), self._compile(node.body, d), self._compile(node.orelse, d)
            return lambda env, _c=c, _a=a, _b=b: _a(env) if _c(env) != 0 else _b(env)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                self._fail("Yalnız izinli fonksiyonlar adıyla çağrılabilir")
            name = node.func.id
            if node.keywords:
                self._fail(f"'{name}' çağrısında anahtar sözcüklü argüman kullanılamaz")
            if name in self.vectors:             # u(i): 1 tabanlı indeks (sabit tamsayı)
                if len(node.args) != 1 or not isinstance(node.args[0], ast.Constant) \
                        or isinstance(node.args[0].value, bool) or not isinstance(node.args[0].value, int) \
                        or node.args[0].value < 1:
                    self._fail(f"'{name}(i)' indeksi 1 ya da daha büyük sabit tamsayı olmalı")
                idx = int(node.args[0].value)
                self.used.add(name)
                self.indices[name] = max(self.indices.get(name, 0), idx)
                return lambda env, _n=name, _i=idx - 1: env[_n][_i]
            if name not in FUNCTIONS:
                self._fail(f"İzin verilmeyen fonksiyon '{name}'; izinliler: {', '.join(FUNCTIONS)}")
            fn, lo, hi, _ = FUNCTIONS[name]
            if not lo <= len(node.args) <= hi:
                rng = f"{lo}" if lo == hi else f"{lo}–{hi}"
                self._fail(f"'{name}' {rng} argüman alır, {len(node.args)} verildi")
            args = [self._compile(a, d) for a in node.args]
            if len(args) == 1:
                a0 = args[0]
                return lambda env, _f=fn, _a=a0: _f(_a(env))
            return lambda env, _f=fn, _a=args: _f(*[g(env) for g in _a])
        self._fail(_FORBIDDEN.get(type(node), f"'{type(node).__name__}' öğesi ifadede kullanılamaz"))
        raise AssertionError  # pragma: no cover

    # ------------------------------------------------------------------ kullanım
    def check_widths(self, widths: dict[str, int], owner: str = "") -> None:
        """Vektör değişkenlerin gerçek genişliklerine göre indeksleri denetler."""
        pre = f"'{owner}' bloğu: " if owner else ""
        for var, idx in self.indices.items():
            w = widths.get(var)
            if w is None:
                continue
            if idx > w:
                raise ControlError(f"{pre}ifade {var}({idx}) kullanıyor ama '{var}' genişliği {w} ('{self.text}').")
        for var in self.uses_bare:
            w = widths.get(var)
            if w is not None and w != 1:
                raise ControlError(f"{pre}'{var}' {w} elemanlı; yalın '{var}' yerine {var}(1) … {var}({w}) "
                                   f"kullanın ('{self.text}').")

    def __call__(self, env: dict[str, Any]) -> float:
        return self.eval_prepared(prepare_env(env))

    def eval_prepared(self, env: dict[str, Any]) -> float:
        """`prepare_env` ile hazırlanmış (Python float / liste) ortamda değerlendirme."""
        try:
            v = self._fn(env)
        except ZeroDivisionError:
            raise ControlError(f"İfade sıfıra bölme üretti ('{self.text}').") from None
        except (ValueError, OverflowError):
            raise ControlError(f"İfade tanımsız ya da taşan değer üretti ('{self.text}'; ör. log(0), "
                               "negatif sayının karekökü / kesirli üssü).") from None
        if isinstance(v, complex):  # pragma: no cover - math fonksiyonları karmaşık üretmez
            raise ControlError(f"İfade karmaşık sayı üretti ('{self.text}').")
        v = float(v)
        if not math.isfinite(v):
            raise ControlError(f"İfade sonsuz ya da tanımsız değer üretti ('{self.text}').")
        return v


def prepare_env(env: dict[str, Any]) -> dict[str, Any]:
    """numpy değerlerini Python float / listeye çevirir (sıfıra bölme sessizce inf değil hata üretsin)."""
    return {k: (v.tolist() if isinstance(v, np.ndarray) else float(v)) for k, v in env.items()}


def split_expressions(text: str) -> list[str]:
    """';' ile ayrılmış ifadeleri böler (parantez içindeki ';' yoktur, boş parçalar atılır)."""
    parts = [p.strip() for p in str(text).split(";")]
    return [p for p in parts if p]


# ============================================================================ sayısal parametre metni
@functools.lru_cache(maxsize=512)
def constant(text: str) -> float:
    """Değişkensiz sabit ifade ("2*pi", "1e-3", "-inf", "sqrt(2)/2") → float."""
    t = text.strip()
    low = t.lower()
    if low in ("inf", "+inf", "infinity"):
        return math.inf
    if low in ("-inf", "-infinity"):
        return -math.inf
    if low == "nan":
        return math.nan
    try:
        return float(t)
    except ValueError:
        pass
    e = Expression(t)
    return e({})


def _split_top(s: str, seps: str) -> list[str]:
    """Parantez dışındaki ayırıcılarla böler."""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if depth == 0 and ch in seps:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return out


def parse_numeric(text: str) -> np.ndarray:
    """MATLAB tarzı sayı / vektör / matris metni → 2-B dizi (satırlar ';', elemanlar ',' ya da boşlukla).

    "[1 2; 3 4]" → [[1, 2], [3, 4]], "1e-3" → [[0.001]], "[0 2*pi]" → [[0, 6.283…]]. Boşluk ayırıcı olduğundan
    eleman içi ifadelerde boşluk kullanmayın ya da elemanları virgülle ayırın ("[1 + 2, 3]")."""
    s = str(text).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    s = s.strip()
    if not s:
        return np.zeros((0, 0))
    rows = []
    for row in _split_top(s, ";"):
        row = row.strip()
        if not row:
            continue
        elems = _split_top(row, ",") if "," in row else _split_top(row, " \t")
        vals = []
        for el in elems:
            el = el.strip()
            if not el:
                continue
            try:
                vals.append(constant(el))
            except ControlError as exc:
                raise ControlError(f"'{el}' sayıya çevrilemedi: {exc}") from None
        rows.append(vals)
    if not rows:
        return np.zeros((0, 0))
    n = len(rows[0])
    if any(len(r) != n for r in rows):
        raise ControlError(f"Matris satırlarının eleman sayıları farklı ('{text}').")
    return np.array(rows, dtype=np.float64)
