"""Model ifadesi ayrıştırıcısı (parse_model): MATLAB yazımı, güvenlik (eval yok) ve konumlu Türkçe hatalar.

Ayrıştırılan modeller doğrudan kurucularla (tf / zpk / ss) aynı olmalı; karşılaştırma frekans cevabıyla (1e-13)."""
from __future__ import annotations

import numpy as np
import pytest

from spt.control.errors import ControlError
from spt.control.lti import StateSpace, TransferFunction, ZPK, ss, tf, zpk
from spt.control.freqresp import freqresp
from spt.control.parse import parse_model

W = np.array([0.1, 0.7, 2.3, 9.0])


def _same(a, b, w=W):
    Ha, Hb = freqresp(a, w), freqresp(b, w)
    assert np.allclose(Ha, Hb, rtol=1e-13, atol=0)


CASES = [
    ("1/(s^2+2*s+1)", 0.0, tf([1], [1, 2, 1])),
    ("exp(-0.2*s)*5/(s+1)", 0.0, tf([5], [1, 1], io_delay=0.2)),
    ("tf([1],[1 2 1])", 0.0, tf([1], [1, 2, 1])),
    ("zpk([],[-1 -2],3)", 0.0, zpk([], [-1, -2], 3)),
    ("ss([0 1;-2 -3],[0;1],[1 0],0)", 0.0, ss([[0, 1], [-2, -3]], [[0], [1]], [[1, 0]], 0)),
    ("(z-0.5)/(z^2-z+0.3)", 0.1, tf([1, -0.5], [1, -1, 0.3], 0.1)),
    ("zpk([-1+2i -1-2i],[-3 -4 -5],2)", 0.0, zpk([-1 + 2j, -1 - 2j], [-3, -4, -5], 2)),
    ("2^-1*s/(s+1)", 0.0, tf([0.5, 0], [1, 1])),
    ("-2^2", 0.0, tf(-4.0)),
    ("tf([1 -2],[1 - 2 3])", 0.0, tf([1, -2], [-1, 3])),          # MATLAB boşluk kuralı
    ("s**2/(s+1)**2", 0.0, tf([1, 0, 0], [1, 2, 1])),
    ("10*(s+1)/(s*(s+10))", 0.0, tf([10, 10], [1, 10, 0])),
    ("z^-2*0.5/(z-0.8)", 0.1, tf([0.5], [1, -0.8], 0.1, io_delay=2)),
    ("tf([1 0.5],[1 -0.9])", 0.2, tf([1, 0.5], [1, -0.9], 0.2)),
    ("exp(-1.5*s)/s", 0.0, tf([1], [1, 0], io_delay=1.5)),
    ("pi/(s+pi)", 0.0, tf([np.pi], [1, np.pi])),
    ("ss([-1 0; 0 -2], [1 0; 0 1], [1 1], [0 0])", 0.0, ss([[-1, 0], [0, -2]], np.eye(2), [[1, 1]], [[0, 0]])),
]


@pytest.mark.parametrize("text,Ts,ref", CASES)
def test_parse_equals_constructor(text, Ts, ref):
    G = parse_model(text, Ts)
    assert G.shape == ref.shape and G.Ts == ref.Ts
    w = W[W < np.pi / Ts] if Ts else W
    _same(G, ref, w)


def test_parse_types_and_mimo():
    assert isinstance(parse_model("zpk([],[-1],2)"), ZPK)
    assert isinstance(parse_model("ss(-1,1,1,0)"), StateSpace)
    assert isinstance(parse_model("1/(s+1)"), TransferFunction)
    G = parse_model("[1/(s+1) 2; 0 s/(s+2)]")
    assert G.shape == (2, 2)
    ref = tf([[[1], [2]], [[0], [1, 0]]], [[[1, 1], [1]], [[1], [1, 2]]])
    _same(G, ref)
    H = parse_model("[ss(-1,1,1,0), 1/(s+3)]")
    assert isinstance(H, StateSpace) and H.shape == (1, 2)
    _same(H, tf([[[1], [1]]], [[[1, 1], [1, 3]]]))
    K = parse_model("[1 2; 3 4]")
    assert K.is_static() and np.allclose([[float(v[0]) for v in row] for row in K.num], [[1, 2], [3, 4]])


@pytest.mark.parametrize("text,Ts,pos", [
    ("1/(s+1", 0.0, 7),
    ("2s+1", 0.0, 2),
    ("1/(z+1)", 0.0, 4),
    ("1/(s+1)", 0.1, 4),
    ("foo(s)", 0.0, 1),
    ("s^0.5", 0.0, 2),
    ("[1 2; 3]", 0.0, 1),
    ("import os", 0.0, 1),
    ('__import__("os")', 0.0, 12),
    ("s@2", 0.0, 2),
    ("exp(-s^2)", 0.0, 1),
    ("exp(0.5*s)/(s+1)", 0.0, 1),
    ("", 0.0, 1),
    ("(s+1))", 0.0, 6),
    ("tf([1],[1 1]", 0.0, 13),
    ("s^1000", 0.0, 2),
])
def test_parse_errors_have_position(text, Ts, pos):
    with pytest.raises(ControlError) as ei:
        parse_model(text, Ts)
    assert getattr(ei.value, "position", None) == pos, str(ei.value)
    assert str(ei.value).startswith(f"{pos}. karakterde")


def test_parse_is_safe_against_python_code():
    for text in ("().__class__", "open('x')", "lambda: 0", "[x for x in s]", "s.__dict__", "1if 1else 0"):
        with pytest.raises(ControlError):
            parse_model(text)
    with pytest.raises(ControlError):
        parse_model("(" * 500 + "s" + ")" * 500)
    with pytest.raises(ControlError):
        parse_model("s" * 20000)
