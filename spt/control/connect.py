"""Model bağlantıları: series, parallel, feedback, append, connect (adlarla), sumblk, loopsens, lft.

MATLAB karşılıkları (R2025b'den ölçülen davranışlar):
- `feedback(sys1, sys2=1, sign=-1, feedin, feedout)`: SISO tf'de MATLAB'in formülü num = n1·d2,
  den = d1·d2 − sign·n1·n2 (sadeleştirme yok); diğerlerinde durum uzayında kapalı çevrim. Sonuç türü önceliğe göre.
- `connect(blok1, …, inputs, outputs)`: bloklar giriş / çıkış adlarıyla bağlanır; bir iç sinyalin adı giriş olarak
  verilirse dış giriş o noktaya TOPLANIR (analiz noktası; MATLAB gibi; aynı adlı çıkış ise toplama ÖNCESİ blok çıkışını
  verir, MATLAB'den ölçüldü); dış girişe bağlı olmayan blok girişleri sıfır
  kabul edilir (MATLAB gibi, hata vermez); aynı çıkış adı iki blokta varsa hata (MATLAB gibi). Sonuç ss (MATLAB gibi),
  yapısal olarak bağlı olmayan durumlar atılır.
- `sumblk('e = r - y', n)`: statik toplama bloğu (tf), çok kanallıda e(1) … adları.
- `loopsens(P, C)`: Si, Ti, Li, So, To, Lo, PSi, CSo, Poles, Stable (negatif geri besleme).
- Gecikme: sürekli zamanda geri besleme halkasındaki gecikme iç gecikme gerektirir → açık hata (pade önerisi);
  ayrık zamanda gecikmeler duruma tam katılır.
"""
from __future__ import annotations

import re

import numpy as np

from spt.control.errors import ControlError
from spt.control.lti import (
    DELAY_INTERNAL_MSG, LTI, StateSpace, TransferFunction, ZPK, _as_class, _blkdiag, _gain_matrix, _padd, _pmul,
    _pzero, common_Ts, ss_absorb)
from spt.control import convert


def _lift(x) -> LTI:
    """Sayı / matris → statik kazanç tf (en düşük öncelik: sonuç türünü model belirler, MATLAB gibi)."""
    if isinstance(x, LTI):
        return x
    K = _gain_matrix(x)
    return TransferFunction([[np.array([v]) for v in row] for row in K], [[np.ones(1)] * K.shape[1]] * K.shape[0])


def _target_class(*systems: LTI):
    return max((type(s) for s in systems), key=lambda c: c._priority)


def _ss_nodelay(sys: LTI, Ts: float) -> StateSpace:
    """Durum uzayı, gecikmeler kapalı çevrimde gösterilemeyeceği için: ayrıkta duruma kat, süreklide hata."""
    s = convert.to_ss(sys)
    if s.Ts != Ts:
        s = s.copy()
        s.Ts = Ts
    if s.has_delay():
        if Ts == 0:
            raise ControlError(DELAY_INTERNAL_MSG)
        s = ss_absorb(s)
    return s


def _close(A, B, C, D, K):
    """u = v + K·y statik bağlantısıyla kapalı çevrim (v → y). Cebirsel döngü tekilse hata."""
    p = C.shape[0]
    E = np.eye(p) - D @ K
    if p and np.linalg.cond(E) > 1e12:
        raise ControlError("Cebirsel döngü tekil (I − D·K tersinir değil): bağlantı iyi tanımlı değil.")
    Ei = np.linalg.inv(E) if p else E
    Cc = Ei @ C
    Dc = Ei @ D
    Ac = A + B @ K @ Cc
    Bc = B @ (np.eye(B.shape[1]) + K @ Dc)
    return Ac, Bc, Cc, Dc


def _append_ss(systems: list[StateSpace]) -> StateSpace:
    A = _blkdiag(*[s.A for s in systems])
    B = _blkdiag(*[s.B for s in systems])
    C = _blkdiag(*[s.C for s in systems])
    D = _blkdiag(*[s.D for s in systems])
    names = sum((list(s.state_names) for s in systems), [])
    return StateSpace(A, B, C, D, systems[0].Ts if systems else 0.0, state_names=names,
                      input_names=sum((list(s.input_names) for s in systems), []),
                      output_names=sum((list(s.output_names) for s in systems), []))


def append(*systems) -> LTI:
    """Blok köşegen birleştirme (MATLAB `append`): girişler ve çıkışlar art arda; gecikmeler korunur."""
    if not systems:
        raise ControlError("En az bir model verin.")
    syss = [_lift(s) for s in systems]
    Ts = common_Ts(*syss)
    cls = _target_class(*syss)
    syss = [_as_class(s, cls) for s in syss]
    syss = [s._retime(Ts) if s.Ts != Ts else s for s in syss]
    if cls is StateSpace:
        out = _append_ss([StateSpace(s.A, s.B, s.C, s.D, Ts, state_names=s.state_names,
                                     input_names=s.input_names, output_names=s.output_names) for s in syss])
        out.input_delay = np.concatenate([s.input_delay for s in syss])
        out.output_delay = np.concatenate([s.output_delay for s in syss])
        return out
    ny = sum(s.noutputs for s in syss)
    nu = sum(s.ninputs for s in syss)
    io = np.zeros((ny, nu))
    r0 = c0 = 0
    if cls is TransferFunction:
        N = [[np.zeros(1) for _ in range(nu)] for _ in range(ny)]
        Dn = [[np.ones(1) for _ in range(nu)] for _ in range(ny)]
    else:
        N = [[np.zeros(0, complex) for _ in range(nu)] for _ in range(ny)]
        Dn = [[np.zeros(0, complex) for _ in range(nu)] for _ in range(ny)]
        K = np.zeros((ny, nu))
    for s in syss:
        tot = s.total_delay()
        for i in range(s.noutputs):
            for j in range(s.ninputs):
                if cls is TransferFunction:
                    N[r0 + i][c0 + j], Dn[r0 + i][c0 + j] = s._num[i][j], s._den[i][j]
                else:
                    N[r0 + i][c0 + j], Dn[r0 + i][c0 + j], K[r0 + i, c0 + j] = s._z[i][j], s._p[i][j], s._k[i, j]
                io[r0 + i, c0 + j] = tot[i, j]
        r0 += s.noutputs
        c0 += s.ninputs
    names = dict(input_names=sum((list(s.input_names) for s in syss), []),
                 output_names=sum((list(s.output_names) for s in syss), []))
    if cls is TransferFunction:
        return TransferFunction(N, Dn, Ts, io_delay=io, **names)
    return ZPK(N, Dn, K, Ts, io_delay=io, **names)


def series(sys1, sys2, outputs1=None, inputs2=None) -> LTI:
    """Seri bağlantı (MATLAB `series`): sys2·sys1; outputs1 / inputs2 ile kısmi bağlantı."""
    s1, s2 = _lift(sys1), _lift(sys2)
    if outputs1 is not None:
        s1 = s1[list(np.atleast_1d(outputs1)), :]
    if inputs2 is not None:
        s2 = s2[:, list(np.atleast_1d(inputs2))]
    return s2 * s1


def parallel(sys1, sys2, in1=None, in2=None, out1=None, out2=None) -> LTI:
    """Paralel bağlantı (MATLAB `parallel`): varsayılan sys1 + sys2; in1/in2 ortak girişler, out1/out2 toplanan
    çıkışlar (diğer giriş / çıkışlar eklenir)."""
    s1, s2 = _lift(sys1), _lift(sys2)
    if in1 is None and in2 is None and out1 is None and out2 is None:
        return s1 + s2
    i1 = list(np.atleast_1d(in1 if in1 is not None else range(s1.ninputs)).astype(int))
    i2 = list(np.atleast_1d(in2 if in2 is not None else range(s2.ninputs)).astype(int))
    o1 = list(np.atleast_1d(out1 if out1 is not None else range(s1.noutputs)).astype(int))
    o2 = list(np.atleast_1d(out2 if out2 is not None else range(s2.noutputs)).astype(int))
    if len(i1) != len(i2) or len(o1) != len(o2):
        raise ControlError("Ortak giriş ve toplanan çıkış indis listeleri aynı uzunlukta olmalı.")
    m1, m2, p1, p2 = s1.ninputs, s2.ninputs, s1.noutputs, s2.noutputs
    free1 = [j for j in range(m1) if j not in i1]
    free2 = [j for j in range(m2) if j not in i2]
    nv = len(free1) + len(i1) + len(free2)
    Mi = np.zeros((m1 + m2, nv))
    c = 0
    for j in free1:
        Mi[j, c] = 1
        c += 1
    for a, b in zip(i1, i2):
        Mi[a, c] = 1
        Mi[m1 + b, c] = 1
        c += 1
    for j in free2:
        Mi[m1 + j, c] = 1
        c += 1
    zf1 = [i for i in range(p1) if i not in o1]
    zf2 = [i for i in range(p2) if i not in o2]
    nz = len(zf1) + len(o1) + len(zf2)
    Mo = np.zeros((nz, p1 + p2))
    r = 0
    for i in zf1:
        Mo[r, i] = 1
        r += 1
    for a, b in zip(o1, o2):
        Mo[r, a] = 1
        Mo[r, p1 + b] = 1
        r += 1
    for i in zf2:
        Mo[r, p1 + i] = 1
        r += 1
    return Mo * append(s1, s2) * Mi


def feedback(sys1, sys2=1.0, sign: int = -1, feedin=None, feedout=None) -> LTI:
    """Geri besleme (MATLAB `feedback(sys1, sys2, feedin, feedout, sign)`); varsayılan negatif (sign = −1).

    sys1'in feedin girişlerine sys2'nin çıkışı (sign ile) eklenir, sys1'in feedout çıkışları sys2'yi sürer.
    Sonuç girişleri / çıkışları sys1'inkilerdir."""
    if sign not in (1, -1, 1.0, -1.0):
        raise ControlError("İşaret +1 ya da −1 olmalı.")
    s1, s2 = _lift(sys1), _lift(sys2)
    Ts = common_Ts(s1, s2)
    cls = _target_class(s1, s2)
    fin = list(range(s1.ninputs)) if feedin is None else [int(v) for v in np.atleast_1d(feedin)]
    fout = list(range(s1.noutputs)) if feedout is None else [int(v) for v in np.atleast_1d(feedout)]
    if s2.shape != (len(fin), len(fout)):
        raise ControlError(f"Geri besleme modeli {len(fin)}×{len(fout)} olmalı (feedin × feedout).")
    if cls is TransferFunction and s1.is_siso and s2.is_siso and not s1.has_delay() and not s2.has_delay():
        a, b = convert.to_tf(s1), convert.to_tf(s2)
        n1, d1, n2, d2 = a._num[0][0], a._den[0][0], b._num[0][0], b._den[0][0]
        num = _pmul(n1, d2)
        den = _padd(_pmul(d1, d2), -float(sign) * _pmul(n1, n2))
        if _pzero(den):
            raise ControlError("Kapalı çevrim paydası sıfır (cebirsel döngü tekil).")
        return TransferFunction(num, den, Ts, input_names=s1.input_names, output_names=s1.output_names)
    a, b = _ss_nodelay(s1, Ts), _ss_nodelay(s2, Ts)
    big = _append_ss([a, b])
    m1, p1 = a.ninputs, a.noutputs
    K = np.zeros((big.ninputs, big.noutputs))
    for idx, j in enumerate(fin):
        K[j, p1 + idx] = float(sign)
    for idx, i in enumerate(fout):
        K[m1 + idx, i] = 1.0
    Ac, Bc, Cc, Dc = _close(big.A, big.B, big.C, big.D, K)
    out = StateSpace(Ac, Bc[:, :m1], Cc[:p1], Dc[:p1, :m1], Ts, state_names=big.state_names,
                     input_names=s1.input_names, output_names=s1.output_names)
    if cls is StateSpace:
        return out
    return convert.to_tf(out) if cls is TransferFunction else convert.to_zpk(out)


def lft(sys1, sys2, nu: int | None = None, ny: int | None = None) -> StateSpace:
    """Yıldız çarpımı / doğrusal kesirli dönüşüm (MATLAB `lft(sys1, sys2, nu, ny)`).

    sys1'in son nu çıkışı sys2'nin ilk nu girişini sürer; sys2'nin ilk ny çıkışı sys1'in son ny girişine döner.
    Sonuç girişleri [sys1'in ilk girişleri; sys2'nin son girişleri], çıkışları [sys1'in ilk çıkışları; sys2'nin son
    çıkışları]. nu / ny verilmezse: sys2 daha küçükse alt LFT (sys2'nin bütün G/Ç'si), değilse üst LFT."""
    s1, s2 = _lift(sys1), _lift(sys2)
    Ts = common_Ts(s1, s2)
    if nu is None or ny is None:
        if s2.ninputs <= s1.noutputs and s2.noutputs <= s1.ninputs:
            nu, ny = s2.ninputs, s2.noutputs           # alt LFT: sys2'nin bütün G/Ç'si
        else:
            nu, ny = s1.noutputs, s1.ninputs           # üst LFT: sys1'in bütün G/Ç'si
    nu, ny = int(nu), int(ny)
    if nu > s1.noutputs or nu > s2.ninputs or ny > s2.noutputs or ny > s1.ninputs:
        raise ControlError("lft boyutları uyumsuz.")
    a, b = _ss_nodelay(s1, Ts), _ss_nodelay(s2, Ts)
    big = _append_ss([a, b])
    m1, p1 = a.ninputs, a.noutputs
    K = np.zeros((big.ninputs, big.noutputs))
    for k in range(nu):                       # y1 (son nu çıkış) → u2 (ilk nu giriş)
        K[m1 + k, p1 - nu + k] = 1.0
    for k in range(ny):                       # y2 (ilk ny çıkış) → u1 (son ny giriş)
        K[m1 - ny + k, p1 + k] = 1.0
    Ac, Bc, Cc, Dc = _close(big.A, big.B, big.C, big.D, K)
    ins = list(range(m1 - ny)) + list(range(m1 + nu, big.ninputs))
    outs = list(range(p1 - nu)) + list(range(p1 + ny, big.noutputs))
    names_in = [big.input_names[i] for i in ins]
    names_out = [big.output_names[i] for i in outs]
    return convert.sminreal(StateSpace(Ac, Bc[:, ins], Cc[outs], Dc[np.ix_(outs, ins)], Ts,
                                       state_names=big.state_names, input_names=names_in, output_names=names_out))


# ---------------------------------------------------------------------------------------------------------------
# Adlarla bağlantı
# ---------------------------------------------------------------------------------------------------------------
def _expand_names(names, available: list[str]) -> list[str]:
    if isinstance(names, str):
        names = [names]
    out = []
    for nm in names:
        nm = str(nm)
        if nm in available:
            out.append(nm)
            continue
        vec = [a for a in available if re.fullmatch(re.escape(nm) + r"\(\d+\)", a)]
        if vec:
            out.extend(sorted(vec, key=lambda a: int(a[len(nm) + 1:-1])))
            continue
        out.append(nm)
    return out


def sumblk(expr: str, n: int = 1, Ts: float = 0.0) -> TransferFunction:
    """Toplama bloğu (MATLAB `sumblk`): 'e = r - y + w' → çıkış e, girişler r, y, w (işaretleriyle).
    n > 1 ise her sinyal n kanallı (e(1) … e(n))."""
    if not isinstance(expr, str) or expr.count("=") != 1:
        raise ControlError("Toplama ifadesi 'e = r - y' biçiminde olmalı.")
    lhs, rhs = (t.strip() for t in expr.split("="))
    if not re.fullmatch(r"[A-Za-z_][\w.]*", lhs):
        raise ControlError(f"Geçersiz çıkış adı: {lhs!r}")
    terms = re.findall(r"([+-]?)\s*([A-Za-z_][\w.]*)", rhs)
    rest = re.sub(r"([+-]?)\s*([A-Za-z_][\w.]*)", "", rhs).strip()
    if not terms or rest:
        raise ControlError(f"Toplama ifadesi çözümlenemedi: {rhs!r}")
    n = int(n)
    if n < 1:
        raise ControlError("Kanal sayısı n ≥ 1 olmalı.")
    signs = [(-1.0 if sg == "-" else 1.0) for sg, _ in terms]
    names = [nm for _, nm in terms]
    D = np.hstack([sg * np.eye(n) for sg in signs])
    ins = sum(([nm] if n == 1 else [f"{nm}({k + 1})" for k in range(n)] for nm in names), [])
    outs = [lhs] if n == 1 else [f"{lhs}({k + 1})" for k in range(n)]
    return TransferFunction([[np.array([v]) for v in row] for row in D], [[np.ones(1)] * D.shape[1]] * n, Ts,
                            input_names=ins, output_names=outs)


def connect(*args, inputs=None, outputs=None) -> StateSpace:
    """Adlarla blok bağlantısı (MATLAB `connect(sys1, …, sysN, inputs, outputs)`).

    Son iki konumsal argüman (ya da `inputs=`, `outputs=`) dış giriş ve çıkış ad listeleridir. Her blok girişi aynı
    adlı blok çıkışına bağlanır; dış giriş adı bir blok çıkışıyla aynıysa o sinyale eklenir (analiz noktası).
    Sonuç: ss (adlar dış girişler / çıkışlar)."""
    blocks = list(args)
    if inputs is None or outputs is None:
        if len(blocks) < 3:
            raise ControlError("connect(blok1, …, girişler, çıkışlar) biçiminde çağırın.")
        outputs = blocks.pop()
        inputs = blocks.pop()
    if not blocks or not all(isinstance(b, LTI) for b in blocks):
        raise ControlError("Bağlanacak bloklar model olmalı.")
    Ts = common_Ts(*blocks)
    ss_blocks = [_ss_nodelay(b, Ts) for b in blocks]
    big = _append_ss(ss_blocks)
    unames = list(big.input_names)
    ynames = list(big.output_names)
    for nm in ynames:
        if not nm:
            raise ControlError("Bütün blok çıkışlarının adı olmalı (output_names).")
    dup = {nm for nm in ynames if ynames.count(nm) > 1}
    if dup:
        raise ControlError(f"Çıkış adı birden çok blokta var: {', '.join(sorted(dup))} (çıkış adları tekil olmalı).")
    known = sorted(set(unames) | set(ynames))
    ins = _expand_names(inputs, known)
    outs = _expand_names(outputs, known)
    nU, nY, nR = len(unames), len(ynames), len(ins)
    yidx = {nm: k for k, nm in enumerate(ynames)}
    K = np.zeros((nU, nY))
    N = np.zeros((nU, nR))
    for i, nm in enumerate(unames):
        if nm in yidx:
            K[i, yidx[nm]] = 1.0
    for e, nm in enumerate(ins):
        if nm in yidx:                                      # iç sinyale eklenen giriş
            for i, un in enumerate(unames):
                if un == nm:
                    N[i, e] = 1.0
        else:
            hit = False
            for i, un in enumerate(unames):
                if un == nm:
                    N[i, e] = 1.0
                    hit = True
            if not hit:
                raise ControlError(f"Dış giriş {nm!r} hiçbir blok girişine ya da sinyale bağlı değil.")
    Ac, Bc, Cc, Dc = _close(big.A, big.B, big.C, big.D, K)
    # v = N·r: kapalı çevrimin girişi blok girişleri
    Bv, Dv = Bc @ N, Dc @ N
    rows_C, rows_D = [], []
    for nm in outs:
        if nm in yidx:                    # blok çıkışı (eklenen dış giriş öncesi değer, MATLAB gibi)
            k = yidx[nm]
            c, d = Cc[k].copy(), Dv[k].copy()
        elif nm in ins:
            c, d = np.zeros(Ac.shape[0]), np.zeros(nR)
            d[ins.index(nm)] = 1.0
        else:
            raise ControlError(f"Çıkış {nm!r} bulunamadı (blok çıkışı ya da dış giriş olmalı).")
        rows_C.append(c)
        rows_D.append(d)
    Cn = np.array(rows_C).reshape(len(outs), Ac.shape[0])
    Dn = np.array(rows_D).reshape(len(outs), nR)
    return convert.sminreal(StateSpace(Ac, Bv, Cn, Dn, Ts, state_names=big.state_names, input_names=ins,
                                       output_names=outs))


def loopsens(P, C) -> dict:
    """Geri besleme döngüsü duyarlılıkları (MATLAB `loopsens(P, C)`, negatif geri besleme): Si, Ti, Li (giriş
    döngüsü L = C·P), So, To, Lo (çıkış döngüsü L = P·C), PSi = P·Si, CSo = C·So, Poles (kapalı çevrim), Stable."""
    p, c = _lift(P), _lift(C)
    Ts = common_Ts(p, c)
    ps, cs = _ss_nodelay(p, Ts), _ss_nodelay(c, Ts)
    if cs.ninputs != ps.noutputs or cs.noutputs != ps.ninputs:
        raise ControlError("C'nin boyutu Pᵀ ile uyumlu olmalı (C: nu×ny).")
    m, q = ps.ninputs, ps.noutputs
    Li = cs * ps
    Lo = ps * cs
    Im = StateSpace(np.zeros((0, 0)), np.zeros((0, m)), np.zeros((m, 0)), np.eye(m), Ts)
    Iq = StateSpace(np.zeros((0, 0)), np.zeros((0, q)), np.zeros((q, 0)), np.eye(q), Ts)
    Si = feedback(Im, Li)
    Ti = feedback(Li, Im)
    So = feedback(Iq, Lo)
    To = feedback(Lo, Iq)
    PSi = feedback(ps, cs)
    CSo = feedback(cs, ps)
    poles = np.linalg.eigvals(PSi.A) if PSi.nx else np.zeros(0, complex)
    if Ts > 0:
        stable = bool(np.all(np.abs(poles) < 1 - 1e-12))
    else:
        stable = bool(np.all(poles.real < -1e-12 * np.maximum(1.0, np.abs(poles))))
    return dict(Si=Si, Ti=Ti, Li=Li, So=So, To=To, Lo=Lo, PSi=PSi, CSo=CSo, Poles=poles, Stable=stable)
