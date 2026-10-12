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


def test_matlab_model_functions_in_expressions():
    """MATLAB kullanıcısının yazacağı model işlemleri: c2d (zoh), d2c, feedback, series, parallel, minreal, pid —
    sonuçlar kütüphane fonksiyonlarıyla birebir; yanlış kullanımda konumlu Türkçe hata."""
    from spt.control.connect import feedback
    from spt.control.discretize import c2d
    from spt.control.freqresp import freqresp
    from spt.control.classical import pid

    w = [0.1, 1.0, 10.0]
    G = tf([1.0], [1.0, 2.0, 1.0])
    assert np.allclose(freqresp(parse_model("c2d(1/(s^2+2*s+1), 0.05)"), w), freqresp(c2d(G, 0.05, "zoh"), w),
                       rtol=0, atol=1e-14)
    C = pid(2.0, 1.0, 0.5, 0.01)
    L = parse_model("feedback(pid(2, 1, 0.5, 0.01)/(s^2+2*s+1), 1)")
    assert np.allclose(freqresp(L, w), freqresp(feedback(C * G, 1.0), w), rtol=1e-12, atol=0)
    assert np.allclose(freqresp(parse_model("d2c(c2d(1/(s+1), 0.1))"), w), freqresp(tf([1.0], [1.0, 1.0]), w),
                       rtol=1e-10)
    m = parse_model("minreal((s+1)/((s+1)*(s+2)))")
    assert m.ninputs == 1 and np.allclose(freqresp(m, w), freqresp(tf([1.0], [1.0, 2.0]), w), rtol=1e-12)
    for bad in ("c2d(1/(s+1))", "feedback(1/(s+1), 1, 3)", "pid(1/(s+1))", "foo(1)"):
        with pytest.raises(ControlError):
            parse_model(bad)


@pytest.mark.parametrize("text", ["series(1/(s+1), 2i)", "feedback(1/(s+1), 1+3i)", "parallel(1/(s+1), 1+1i)",
                                  "c2d(1/(s+1), 0.1+5i)", "pid(1+2i)", "pid(1, 2i)", "tf([1 2i],[1 1])"])
def test_parse_rejects_complex_arguments(text):
    with pytest.raises(ControlError, match="gerçel"):
        parse_model(text)


def test_parse_large_model_matrix_is_fast():
    import time
    txt = "[" + ";".join(" ".join(f"1/(s+{i + j + 1})" for j in range(25)) for i in range(25)) + "]"
    t0 = time.perf_counter()
    G = parse_model(txt)
    assert time.perf_counter() - t0 < 5.0 and G.shape == (25, 25)
    _same(G[3, 7], tf([1], [1, 11]))
    H = parse_model("[ss(-1,1,1,0) ss(-2,1,1,0); 0 exp(-0.5*s)/(s+3)]")
    assert H.shape == (2, 2)
    ref = tf([[[1], [1]], [[0], [1]]], [[[1, 1], [1, 2]], [[1], [1, 3]]], io_delay=[[0, 0], [0, 0.5]])
    _same(H, ref)
