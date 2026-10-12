"""Kullanıcının yazdığı model ifadesini güvenli biçimde modele çevirir (eval kullanılmaz).

Desteklenen yazım (MATLAB'e benzer):
- Değişkenler: `s` (sürekli), `z` (ayrık; Ts > 0 verilmeli); sabitler `pi`, sanal birim `i` / `j` (`2+3i`).
- İşlemler: `+ - * /`, `^` (ya da `**`; tamsayı üs modelde), parantez, tekli eksi (MATLAB önceliğiyle: -2^2 = −4).
- Fonksiyonlar: `tf(num, den[, Ts])`, `zpk(z, p, k[, Ts])`, `ss(A, B, C, D[, Ts])`, `exp(-T*s)` (sürekli gecikme),
  `sqrt(x)`. Ts verilmezse `parse_model`'in Ts'si kullanılır.
- Matrisler: `[1 2; 3 4]`, `[1, -2]` (MATLAB kuralı: boşluktan sonra gelen ve ardından boşluk gelmeyen +/- yeni
  eleman başlatır: `[1 -2]` iki eleman, `[1 - 2]` bir eleman); model elemanlı matris `[G1 G2; G3 G4]` MIMO model kurar.

Hatalarda konumlu Türkçe ileti verilir (`ControlError`, `.position` özniteliği 1 tabanlı karakter konumu).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from spt.control.errors import ControlError
from spt.control.lti import LTI, StateSpace, TransferFunction, ZPK, ss, tf, zpk
from spt.control import convert

MAX_LEN = 10_000
MAX_DEPTH = 200
MAX_POWER = 100

_TOKEN_RE = re.compile(r"""
    (?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?[ij]?)
  | (?P<id>[A-Za-z_][A-Za-z_0-9]*)
  | (?P<op>\*\*|\.\*|\./|\.\^|[-+*/^(),;\[\]'])
  | (?P<nl>\n)
  | (?P<ws>[ \t\r]+)
""", re.VERBOSE)


@dataclass
class _Tok:
    kind: str           # num, id, op, nl, end
    text: str
    pos: int            # 1 tabanlı
    sp_before: bool = False
    sp_after: bool = False


def _err(msg: str, pos: int) -> ControlError:
    e = ControlError(f"{pos}. karakterde: {msg}")
    e.position = pos
    return e


def _tokenize(text: str) -> list[_Tok]:
    toks: list[_Tok] = []
    i = 0
    space = False
    while i < len(text):
        m = _TOKEN_RE.match(text, i)
        if not m:
            raise _err(f"geçersiz karakter {text[i]!r}", i + 1)
        kind = m.lastgroup
        if kind == "ws":
            space = True
            if toks:
                toks[-1].sp_after = True
        else:
            toks.append(_Tok(kind, m.group(), i + 1, sp_before=space))
            space = False
        i = m.end()
    toks.append(_Tok("end", "", len(text) + 1, sp_before=space))
    return toks


class _Parser:
    def __init__(self, text: str, Ts: float):
        self.text = text
        self.toks = _tokenize(text)
        self.i = 0
        self.Ts = Ts
        self.ctx: list[str] = []          # 'm' (matris) / 'p' (parantez)
        self.depth = 0
        self.used_s = self.used_z = False

    # --- yardımcılar ---------------------------------------------------------------------------------------
    @property
    def tok(self) -> _Tok:
        return self.toks[self.i]

    def peek(self, k: int = 1) -> _Tok:
        return self.toks[min(self.i + k, len(self.toks) - 1)]

    def take(self) -> _Tok:
        t = self.toks[self.i]
        self.i += 1
        return t

    def expect(self, text: str) -> _Tok:
        t = self.tok
        if t.kind == "op" and t.text == text:
            return self.take()
        found = "ifade sonu" if t.kind == "end" else repr(t.text)
        raise _err(f"{text!r} bekleniyordu, {found} bulundu", t.pos)

    def in_matrix(self) -> bool:
        return bool(self.ctx) and self.ctx[-1] == "m"

    def skip_nl(self):
        while self.tok.kind == "nl" and not self.in_matrix():
            self.take()

    # --- dilbilgisi -----------------------------------------------------------------------------------------
    def parse(self):
        self.skip_nl()
        if self.tok.kind == "end":
            raise _err("ifade boş", 1)
        v = self.expr()
        self.skip_nl()
        if self.tok.kind != "end":
            t = self.tok
            hint = " (çarpma için '*' eksik olabilir)" if t.kind in ("id", "num") or t.text == "(" else ""
            raise _err(f"beklenmeyen {t.text!r}{hint}", t.pos)
        return v

    def _element_break(self) -> bool:
        """Matris içinde +/- yeni eleman mı başlatıyor (MATLAB boşluk kuralı)?"""
        t = self.tok
        return self.in_matrix() and t.sp_before and not t.sp_after

    def expr(self):
        self._enter()
        left, _ = self.term()
        while self.tok.kind == "op" and self.tok.text in "+-" and not self._element_break():
            op = self.take()
            right, _ = self.term()
            left = _binop(left, right, op.text, op.pos)
        self.depth -= 1
        return left

    def term(self):
        left, lpos = self.unary()
        while self.tok.kind == "op" and self.tok.text in ("*", "/", ".*", "./"):
            op = self.take()
            right, _ = self.unary()
            left = _binop(left, right, op.text, op.pos)
        return left, lpos

    def unary(self):
        t = self.tok
        if t.kind == "op" and t.text in "+-":
            self.take()
            v, _ = self.unary()
            return (_neg(v, t.pos) if t.text == "-" else v), t.pos
        return self.power()

    def power(self):
        base, pos = self.postfix()
        while self.tok.kind == "op" and self.tok.text in ("^", "**", ".^"):
            op = self.take()
            ex = self.powarg()
            base = _pow(base, ex, op.pos)
        return base, pos

    def powarg(self):
        t = self.tok
        if t.kind == "op" and t.text in "+-":
            self.take()
            v = self.powarg()
            return _neg(v, t.pos) if t.text == "-" else v
        v, _ = self.postfix()
        return v

    def postfix(self):
        v, pos = self.primary()
        while self.tok.kind == "op" and self.tok.text == "'" and not self.tok.sp_before:
            t = self.take()
            if isinstance(v, np.ndarray):
                v = v.T
            elif not isinstance(v, (int, float, complex)):
                raise _err("devrik (') yalnız sayı / matris için", t.pos)
        return v, pos

    def primary(self):
        t = self.tok
        if t.kind == "num":
            self.take()
            txt = t.text
            if txt[-1] in "ij":
                return complex(0.0, float(txt[:-1])), t.pos
            return float(txt), t.pos
        if t.kind == "id":
            self.take()
            if self.tok.kind == "op" and self.tok.text == "(" and not (self.in_matrix() and self.tok.sp_before):
                return self.call(t), t.pos
            return self.name(t), t.pos
        if t.kind == "op" and t.text == "(":
            self.take()
            self.ctx.append("p")
            v = self.expr()
            self.skip_nl()
            self.expect(")")
            self.ctx.pop()
            return v, t.pos
        if t.kind == "op" and t.text == "[":
            return self.matrix(), t.pos
        if t.kind == "end":
            raise _err("ifade beklenirken giriş bitti", t.pos)
        raise _err(f"beklenmeyen {t.text!r}", t.pos)

    def _enter(self):
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise _err("ifade çok derin iç içe", self.tok.pos)

    def name(self, t: _Tok):
        n = t.text
        if n == "s":
            if self.Ts > 0:
                raise _err("ayrık modelde (Ts > 0) 's' değil 'z' kullanın", t.pos)
            self.used_s = True
            return tf("s")
        if n == "z":
            if not self.Ts > 0:
                raise _err("'z' değişkeni için örnekleme zamanı (Ts > 0) verin", t.pos)
            self.used_z = True
            return tf("z", self.Ts)
        if n == "pi":
            return math.pi
        if n in ("i", "j"):
            return 1j
        if n in ("Inf", "inf"):
            return math.inf
        raise _err(f"bilinmeyen ad {n!r} (s, z, pi, i, j; fonksiyonlar tf, zpk, ss, exp, sqrt)", t.pos)

    def call(self, t: _Tok):
        self.expect("(")
        self.ctx.append("p")
        args = []
        self.skip_nl()
        if not (self.tok.kind == "op" and self.tok.text == ")"):
            while True:
                args.append((self.expr(), self.tok.pos))
                self.skip_nl()
                if self.tok.kind == "op" and self.tok.text == ",":
                    self.take()
                    self.skip_nl()
                    continue
                break
        self.expect(")")
        self.ctx.pop()
        return _call(t.text, [a for a, _ in args], t.pos, self.Ts)

    def matrix(self):
        start = self.expect("[")
        self.ctx.append("m")
        rows: list[list] = [[]]
        while True:
            t = self.tok
            if t.kind == "end":
                raise _err("']' bekleniyordu, ifade bitti", t.pos)
            if t.kind == "op" and t.text == "]":
                self.take()
                break
            if (t.kind == "op" and t.text == ";") or t.kind == "nl":
                self.take()
                if rows[-1]:
                    rows.append([])
                continue
            if t.kind == "op" and t.text == ",":
                self.take()
                continue
            if rows[-1] and not t.sp_before and not (t.kind == "op" and t.text in "+-"):
                raise _err(f"matris elemanları arasında boşluk ya da ',' bekleniyordu ({t.text!r})", t.pos)
            rows[-1].append((self.expr(), t.pos))
        self.ctx.pop()
        if rows and not rows[-1]:
            rows.pop()
        return _build_matrix(rows, start.pos)


# ---------------------------------------------------------------------------------------------------------------
# Değerlendirme (beyaz liste)
# ---------------------------------------------------------------------------------------------------------------
def _is_num(v) -> bool:
    return isinstance(v, (int, float, complex))


def _as_lti_ok(v, pos):
    if isinstance(v, LTI):
        return v
    if _is_num(v):
        if isinstance(v, complex) and v.imag != 0:
            raise _err("karmaşık sayı modelle birleştirilemez", pos)
        return float(np.real(v))
    if isinstance(v, np.ndarray):
        if np.iscomplexobj(v) and np.any(v.imag != 0):
            raise _err("karmaşık matris modelle birleştirilemez", pos)
        return np.real(v).astype(float)
    raise _err("geçersiz işlenen", pos)


def _binop(a, b, op: str, pos: int):
    try:
        if isinstance(a, LTI) or isinstance(b, LTI):
            a, b = _as_lti_ok(a, pos), _as_lti_ok(b, pos)
            if op == "+":
                return a + b
            if op == "-":
                return a - b
            if op in ("*", ".*"):
                return a * b
            if op in ("/", "./"):
                return a / b
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            if isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.size > 1 and b.size > 1:
                return a @ b
            return a * b
        if op == ".*":
            return a * b
        if op in ("/", "./"):
            if isinstance(b, np.ndarray) and b.size > 1 and op == "/":
                raise _err("matrisle sağdan bölme desteklenmez", pos)
            return a / b
    except ControlError as e:
        if getattr(e, "position", None) is None:
            raise _err(str(e), pos) from None
        raise
    except (ValueError, ZeroDivisionError, TypeError) as e:
        raise _err(f"işlem yapılamadı ({e})", pos) from None
    raise _err(f"bilinmeyen işlem {op!r}", pos)


def _neg(v, pos):
    if isinstance(v, (LTI, np.ndarray)) or _is_num(v):
        return -v
    raise _err("geçersiz işlenen", pos)


def _pow(base, ex, pos):
    if isinstance(ex, LTI) or isinstance(ex, np.ndarray) and ex.size != 1:
        raise _err("üs bir sayı olmalı", pos)
    if isinstance(ex, np.ndarray):
        ex = ex.ravel()[0]
    if isinstance(base, LTI):
        if isinstance(ex, complex) or float(ex) != int(ex):
            raise _err("model yalnız tamsayı kuvvete yükseltilebilir", pos)
        if abs(int(ex)) > MAX_POWER:
            raise _err(f"üs çok büyük (|üs| ≤ {MAX_POWER})", pos)
        try:
            return base ** int(ex)
        except ControlError as e:
            raise _err(str(e), pos) from None
    if isinstance(base, np.ndarray) and base.size > 1:
        if base.ndim != 2 or base.shape[0] != base.shape[1] or float(np.real(ex)) != int(np.real(ex)):
            raise _err("matris kuvveti için kare matris ve tamsayı üs gerekir", pos)
        return np.linalg.matrix_power(base, int(np.real(ex)))
    try:
        b = complex(base.ravel()[0]) if isinstance(base, np.ndarray) else base
        r = b ** ex
        return r.real if isinstance(r, complex) and r.imag == 0 else r
    except (OverflowError, ZeroDivisionError, ValueError) as e:
        raise _err(f"kuvvet hesaplanamadı ({e})", pos) from None


def _vec(v, pos, what: str) -> np.ndarray:
    if isinstance(v, LTI):
        raise _err(f"{what} sayı dizisi olmalı", pos)
    a = np.atleast_1d(np.asarray(v))
    if a.ndim == 2 and min(a.shape) > 1:
        raise _err(f"{what} vektör olmalı", pos)
    return a.ravel()


def _mat(v, pos, what: str) -> np.ndarray:
    if isinstance(v, LTI):
        raise _err(f"{what} sayı matrisi olmalı", pos)
    a = np.asarray(v)
    if np.iscomplexobj(a):
        if np.any(a.imag != 0):
            raise _err(f"{what} gerçel olmalı", pos)
        a = a.real
    return np.atleast_2d(a.astype(float)) if a.size else np.zeros((0, 0))


def _realv(v, pos, what):
    """Sayı / dizi gerçel olmalı: sanal kısmı sıfırdan farklıysa sessizce atmak yerine hata (MATLAB'de karmaşık
    katsayılı model ayrı bir kavramdır; burada desteklenmez)."""
    a = np.asarray(v)
    if np.iscomplexobj(a):
        if np.any(a.imag != 0):
            raise _err(f"{what} gerçel olmalı (karmaşık sayı verildi)", pos)
        a = a.real
    return a.item() if a.ndim == 0 else a


def _ts_arg(args, idx, Ts, pos):
    if len(args) > idx:
        v = args[idx]
        if not _is_num(v) or isinstance(v, complex):
            raise _err("örnekleme zamanı bir sayı olmalı", pos)
        return float(v)
    return Ts


def _call(name: str, args: list, pos: int, Ts: float):
    try:
        if name == "tf":
            if len(args) not in (2, 3):
                raise _err("tf(pay, payda[, Ts]) biçiminde yazın", pos)
            return tf(_realv(_vec(args[0], pos, "pay"), pos, "pay"), _realv(_vec(args[1], pos, "payda"), pos, "payda"),
                      _ts_arg(args, 2, Ts, pos))
        if name == "zpk":
            if len(args) not in (3, 4):
                raise _err("zpk(sıfırlar, kutuplar, kazanç[, Ts]) biçiminde yazın", pos)
            k = args[2]
            if not _is_num(k) and not (isinstance(k, np.ndarray) and k.size == 1):
                raise _err("kazanç bir sayı olmalı", pos)
            kk = complex(np.asarray(k).ravel()[0])
            if kk.imag != 0:
                raise _err("kazanç gerçel olmalı", pos)
            return zpk(_vec(args[0], pos, "sıfırlar"), _vec(args[1], pos, "kutuplar"), kk.real,
                       _ts_arg(args, 3, Ts, pos))
        if name == "ss":
            if len(args) not in (4, 5):
                raise _err("ss(A, B, C, D[, Ts]) biçiminde yazın", pos)
            A, B, C, D = (_mat(a, pos, w) for a, w in zip(args[:4], "ABCD"))
            return ss(A, B, C, D, _ts_arg(args, 4, Ts, pos))
        if name == "exp":
            if len(args) != 1:
                raise _err("exp tek argüman alır", pos)
            return _exp(args[0], pos)
        if name == "sqrt":
            if len(args) != 1 or not (_is_num(args[0])):
                raise _err("sqrt bir sayı alır", pos)
            v = complex(args[0]) ** 0.5
            return v.real if v.imag == 0 else v
        # MATLAB'deki model işlemleri (varsayılanlarıyla; ayrıştırıcıda metin değişmezi yok → yöntem seçilemez)
        if name in ("c2d", "d2c", "feedback", "series", "parallel", "minreal"):
            from spt.control import connect as _cn
            from spt.control import discretize as _dz
            for a in args:
                if not isinstance(a, LTI):
                    _realv(a, pos, "argüman")
            sys_args = [a if isinstance(a, LTI) else tf(np.real(np.atleast_1d(a)), [1.0], Ts) if _is_num(a) else a
                        for a in args]
            if name == "c2d":
                if len(args) != 2 or not isinstance(args[0], LTI) or not _is_num(args[1]):
                    raise _err("c2d(model, Ts) biçiminde yazın (yöntem: sıfır dereceli tutucu, MATLAB varsayılanı)",
                               pos)
                return _dz.c2d(args[0], float(np.real(args[1])), "zoh")
            if name == "d2c":
                if len(args) != 1 or not isinstance(args[0], LTI):
                    raise _err("d2c(model) biçiminde yazın (yöntem: sıfır dereceli tutucu)", pos)
                return _dz.d2c(args[0], "zoh")
            if name == "feedback":
                if not 1 <= len(args) <= 3 or not isinstance(args[0], LTI):
                    raise _err("feedback(G[, H[, işaret]]) biçiminde yazın (işaret −1: negatif geri besleme)", pos)
                sign = -1 if len(args) < 3 else args[2]
                if not _is_num(sign) or np.real(sign) not in (-1, 1):
                    raise _err("feedback işareti −1 ya da +1 olmalı", pos)
                H = sys_args[1] if len(args) >= 2 else 1.0
                return _cn.feedback(args[0], H, int(np.real(sign)))
            if name in ("series", "parallel"):
                if len(args) != 2 or not any(isinstance(a, LTI) for a in args):
                    raise _err(f"{name}(model1, model2) biçiminde yazın", pos)
                return (_cn.series if name == "series" else _cn.parallel)(sys_args[0], sys_args[1])
            if len(args) != 1 or not isinstance(args[0], LTI):
                raise _err("minreal(model) biçiminde yazın", pos)
            return convert.minreal(args[0])
        if name == "pid":
            if not 1 <= len(args) <= 5 or not all(_is_num(a) for a in args):
                raise _err("pid(Kp[, Ki[, Kd[, Tf[, Ts]]]]) biçiminde yazın (sayılarla)", pos)
            from spt.control.classical import pid as _pid
            vals = [float(_realv(a, pos, "PID parametresi")) for a in args] + [0.0] * (5 - len(args))
            return _pid(vals[0], vals[1], vals[2], vals[3], vals[4] if len(args) == 5 else Ts)
    except ControlError as e:
        if getattr(e, "position", None) is None:
            raise _err(str(e), pos) from None
        raise
    raise _err(f"bilinmeyen fonksiyon {name!r} (tf, zpk, ss, exp, sqrt, c2d, d2c, feedback, series, parallel, "
               "minreal, pid)", pos)


def _exp(arg, pos):
    if _is_num(arg):
        return complex(np.exp(arg)) if isinstance(arg, complex) else math.exp(arg)
    if isinstance(arg, LTI):
        if arg.is_discrete:
            raise _err("ayrık zamanda gecikme için z^-k kullanın", pos)
        t = convert.to_tf(arg)
        if not t.is_siso or t.has_delay():
            raise _err("exp(-T*s) biçiminde gecikme bekleniyordu", pos)
        n, d = t._num[0][0], t._den[0][0]
        if d.size != 1 or n.size != 2 or n[1] != 0:
            raise _err("exp(-T*s) biçiminde gecikme bekleniyordu (üs s'nin katı olmalı)", pos)
        T = -n[0] / d[0]
        if T < 0:
            raise _err("gecikme negatif olamaz (exp(-T*s), T ≥ 0)", pos)
        return TransferFunction([1.0], [1.0], 0.0, io_delay=T)
    raise _err("exp argümanı sayı ya da -T*s olmalı", pos)


def _build_matrix(rows: list[list], pos: int):
    if not rows:
        return np.zeros((0, 0))
    if any(isinstance(v, LTI) for row in rows for v, _ in row):
        return _lti_concat(rows, pos)
    out_rows = []
    for row in rows:
        parts = []
        for v, p in row:
            a = np.asarray(v)
            if a.ndim == 0:
                a = a.reshape(1, 1)
            elif a.ndim == 1:
                a = a.reshape(1, -1)
            parts.append(a)
        h = {p.shape[0] for p in parts}
        if len(h) != 1:
            raise _err("matris satırındaki elemanların satır sayıları uyuşmuyor", row[0][1])
        out_rows.append(np.hstack(parts))
    w = {r.shape[1] for r in out_rows}
    if len(w) != 1:
        raise _err("matris satırlarının uzunlukları farklı", pos)
    m = np.vstack(out_rows)
    if np.iscomplexobj(m) and not np.any(m.imag):
        m = m.real
    return m


def _lti_concat(rows, pos):
    """[G1 G2; G3 G4] → MIMO model (blok birleştirme)."""
    blocks = []
    for row in rows:
        brow = []
        for v, p in row:
            if not isinstance(v, LTI):
                v = tf(_mat(v, p, "eleman"))
            brow.append(v)
        blocks.append(brow)
    cls = max((type(b) for row in blocks for b in row), key=lambda c: c._priority)
    from spt.control.lti import common_Ts
    Ts = common_Ts(*[b for row in blocks for b in row])
    for row in blocks:
        if len({b.noutputs for b in row}) != 1:
            raise _err("model matrisinin satırındaki blokların çıkış sayıları farklı", pos)
    if len({sum(b.ninputs for b in row) for row in blocks}) != 1:
        raise _err("model matrisinin satırlarının giriş sayıları farklı", pos)
    blocks = [[b if b.Ts == Ts else b._retime(Ts) for b in row] for row in blocks]
    if cls is StateSpace:
        out = _ss_concat(blocks)
        if out is not None:
            return convert.sminreal(out)
        return convert.sminreal(convert.to_ss(_sum_concat(blocks)))
    # tf / zpk: elemanlar doğrudan yerleştirilir (her elemanın toplam gecikmesi io gecikmesi olur)
    ny = sum(row[0].noutputs for row in blocks)
    nu = sum(b.ninputs for b in blocks[0])
    io = np.zeros((ny, nu))
    is_tf = cls is TransferFunction
    A_ = [[None] * nu for _ in range(ny)]
    B_ = [[None] * nu for _ in range(ny)]
    K = np.zeros((ny, nu))
    r0 = 0
    for row in blocks:
        c0 = 0
        for b in row:
            bb = convert.to_tf(b) if is_tf else convert.to_zpk(b)
            tot = bb.total_delay()
            for i in range(bb.noutputs):
                for j in range(bb.ninputs):
                    if is_tf:
                        A_[r0 + i][c0 + j], B_[r0 + i][c0 + j] = bb._num[i][j], bb._den[i][j]
                    else:
                        A_[r0 + i][c0 + j], B_[r0 + i][c0 + j] = bb._z[i][j], bb._p[i][j]
                        K[r0 + i, c0 + j] = bb._k[i, j]
                    io[r0 + i, c0 + j] = tot[i, j]
            c0 += b.ninputs
        r0 += row[0].noutputs
    if is_tf:
        return TransferFunction(A_, B_, Ts, io_delay=io)
    return ZPK(A_, B_, K, Ts, io_delay=io)


def _ss_concat(blocks):
    """Durum uzayı blok birleştirme (satır: yatay, sonra dikey). Gecikmeler blok sınırlarında uyuşmuyorsa None."""
    from spt.control.lti import _blkdiag
    rows = []
    for row in blocks:
        ss_row = [convert.to_ss(b) for b in row]
        od = ss_row[0].output_delay
        if any(not np.allclose(b.output_delay, od) for b in ss_row):
            return None
        rows.append(StateSpace(_blkdiag(*[b.A for b in ss_row]), _blkdiag(*[b.B for b in ss_row]),
                               np.hstack([b.C for b in ss_row]), np.hstack([b.D for b in ss_row]), ss_row[0].Ts,
                               input_delay=np.concatenate([b.input_delay for b in ss_row]), output_delay=od))
    ind = rows[0].input_delay
    if any(not np.allclose(r.input_delay, ind) for r in rows):
        return None
    return StateSpace(_blkdiag(*[r.A for r in rows]), np.vstack([r.B for r in rows]), _blkdiag(*[r.C for r in rows]),
                      np.vstack([r.D for r in rows]), rows[0].Ts, input_delay=ind,
                      output_delay=np.concatenate([r.output_delay for r in rows]))


def _sum_concat(blocks):
    """Genel (gecikmeleri uyuşmayan) durum: Σ E_i · G_ij · F_j."""
    ny = sum(row[0].noutputs for row in blocks)
    nu = sum(b.ninputs for b in blocks[0])
    total = None
    r0 = 0
    for row in blocks:
        c0 = 0
        for b in row:
            E = np.zeros((ny, b.noutputs))
            E[r0:r0 + b.noutputs, :] = np.eye(b.noutputs)
            F = np.zeros((b.ninputs, nu))
            F[:, c0:c0 + b.ninputs] = np.eye(b.ninputs)
            term = E * b * F
            total = term if total is None else total + term
            c0 += b.ninputs
        r0 += row[0].noutputs
    return total


def parse_model(text: str, Ts: float = 0.0) -> LTI:
    """Metin ifadesinden model (güvenli; eval yok). Sayı / matris sonucu statik kazanç tf olur.

    Örnekler: '1/(s^2+2*s+1)', 'exp(-0.2*s)*5/(s+1)', 'tf([1],[1 2 1])', 'zpk([],[-1 -2],3)',
    'ss([0 1;-2 -3],[0;1],[1 0],0)', '(z-0.5)/(z^2-z+0.3)' (Ts verilmişse)."""
    if not isinstance(text, str):
        raise ControlError("Model ifadesi metin olmalı.")
    if len(text) > MAX_LEN:
        raise ControlError(f"İfade çok uzun (en çok {MAX_LEN} karakter).")
    Ts = float(Ts)
    if not (Ts >= 0 and math.isfinite(Ts)):
        raise ControlError("Ts ≥ 0 olmalı.")
    p = _Parser(text, Ts)
    try:
        v = p.parse()
    except RecursionError:
        raise ControlError("İfade çok derin iç içe.") from None
    if isinstance(v, LTI):
        return v
    if _is_num(v):
        if isinstance(v, complex) and v.imag != 0:
            raise ControlError("Sonuç karmaşık bir sayı; model değil.")
        return tf(float(np.real(v)), Ts=Ts)
    if isinstance(v, np.ndarray):
        m = _mat(v, 1, "Sonuç")
        if m.size == 0:
            raise ControlError("Sonuç boş matris; model değil.")
        return tf(m, Ts=Ts)
    raise ControlError("İfade bir model üretmedi.")


__all__ = ["parse_model"]
