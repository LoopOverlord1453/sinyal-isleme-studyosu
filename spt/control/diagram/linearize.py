"""Blok diyagramın çalışma noktasında doğrusallaştırılması ve çalışma noktası (trim) bulma.

Doğrusallaştırma (`linearize`; Simulink Control Design `linearize` + `linio` anlamlarıyla)
----------------------------------------------------------------------------------------
- Blok blok: her bloğun yerel modeli (A, B, C, D) analitik (LTI bloklarında tam; doğrusal olmayanlarda tanımlıysa)
  ya da merkezi farkla (adım ε^(1/3)·max(1, |v|)) bulunur; sonra bağlantılarla kapalı biçimde birleştirilir:
  y = (I − D·M)⁻¹ (C x + D E w).
- Doğrusallaştırma noktaları bir sinyal (blok çıkış portu) üzerindedir: 'input' (girişe w eklenir), 'output'
  (ölçüm), 'inout' (w eklenir, eklenmiş sinyal ölçülür), 'outin' (önce ölçülür sonra w eklenir), 'openinput'
  (çevrim açılır, aşağı akış w), 'openoutput' (ölçülür, çevrim açılır), 'looptransfer' (ölçülür, açılır, w enjekte:
  çevrim transferi — işaretiyle, negatif geri beslemede −L), 'loopbreak' (yalnız açma).
- G/Ç yolu üzerinde olmayan bloklar atılır (Simulink "BlockReduction").
- **Hız kuralı:** yol üzerindeki bloklar arasında ayrık blok varsa hedef Ts varsayılan olarak en yavaş ayrık hızdır
  (Simulink: "longest sample time"); `Ts` ile değiştirilebilir. Ayrık hedefte sürekli kısım (sürekli hızdaki tüm
  bloklar) **bir bütün olarak** en hızlı ayrık hızda ZOH ile ayrıklaştırılır, o hızdaki bloklarla birleştirilir, grup
  olarak bir sonraki (daha yavaş) hıza ZOH ile taşınır (tam katta matris kuvveti, kesin) … hedef hıza kadar. Bu,
  örneklenmiş veri sisteminin tam modelidir ve MATLAB `linearize` ile birebirdir (ölçüldü: ZOH → G1 → G2 zincirinde
  c2d(G1·G2), blok blok değil; iki hızlı denetleyicide 1e-15). Sürekli hedefte (Ts = 0) en yavaş ayrık gruptan
  başlanıp hızlıya taşınır ve sonunda grup olarak ZOH tersiyle (d2c) sürekliye çevrilir (MATLAB ile aynı;
  `rate_conversion='tustin'` seçilebilir). Hedeften yavaş ayrık bloklar (yalnız Ts açıkça küçük verilirse) blok
  düzeyinde dönüştürülür.
- Varsayılanlar (MATLAB'den ölçüldü): Doyum / Ölü bölge / Hız sınırlayıcı / Nicemleyici kazanç 1 ("kazanç olarak
  doğrusallaştır" kapalıyken Doyum / Ölü bölge bölgeye göre 1 / 0, Nicemleyici 0, Hız sınırlayıcı yine 1), Röle 0,
  Taşıma gecikmesi Padé (derece 0: kazanç 1), PID doyumu kazanç 1.

Çalışma noktası (`findop`; MATLAB `findop` + `operspec` anlamlarıyla)
-------------------------------------------------------------------
Bilinmeyenler: bilinmeyen durumlar, serbest kök girişler. Koşullar: kararlı hâl istenen sürekli durumların türevi 0,
ayrık durumların x[k+1] − x[k] = 0, bilinen çıkışların hedef değeri, taşıma gecikmesi / bellek gibi blokların
kararlı hâlde çıkış = giriş. Sınırlı doğrusal olmayan en küçük kareler (scipy `least_squares`, 'trf').
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from spt.control.diagram import _lti
from spt.control.diagram.block import Block, as_vec
from spt.control.diagram.engine import CompiledModel, OperatingPoint, SimResult, compile_model
from spt.control.diagram.model import Model
from spt.control.errors import ControlError

POINT_TYPES = {
    "input": "Giriş (w eklenir)",
    "output": "Çıkış (ölçüm)",
    "inout": "Giriş + çıkış (w eklenir, eklenmiş sinyal ölçülür)",
    "outin": "Çıkış + giriş (ölçülür, sonra w eklenir)",
    "openinput": "Açık çevrim girişi",
    "openoutput": "Açık çevrim çıkışı",
    "looptransfer": "Çevrim transferi",
    "loopbreak": "Çevrim açma",
}
_HAS_INPUT = {"input", "inout", "outin", "openinput", "looptransfer"}
_HAS_OUTPUT = {"output", "inout", "outin", "openoutput", "looptransfer"}
_OPENS = {"openinput", "openoutput", "looptransfer", "loopbreak"}


@dataclass(frozen=True)
class LinPoint:
    """Doğrusallaştırma noktası: `block` bloğunun `port` çıkışındaki (0 tabanlı) sinyal üzerinde."""
    block: str
    port: int = 0
    type: str = "input"
    name: str | None = None


def linio(block: Any, port: int = 0, type: str = "input", name: str | None = None) -> LinPoint:
    """MATLAB `linio` karşılığı (port 0 tabanlı)."""
    if isinstance(block, Block):
        block = block.name
    if type not in POINT_TYPES:
        raise ControlError(f"Bilinmeyen doğrusallaştırma noktası türü '{type}'; geçerliler: {', '.join(POINT_TYPES)}.")
    return LinPoint(str(block), int(port), type, name)


@dataclass
class LinearSystem:
    """Doğrusallaştırma sonucu: x' = A x + B u, y = C x + D u (Ts > 0 ise ayrık)."""
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    Ts: float = 0.0
    input_names: list[str] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)
    state_names: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        return self.D.shape

    def freqresp(self, w: Any) -> np.ndarray:
        """Karmaşık frekans cevabı (ny, nu, nw); w rad/s (ayrıkta z = e^(jωTs))."""
        w = np.atleast_1d(np.asarray(w, dtype=float))
        out = np.empty(self.D.shape + (w.size,), dtype=complex)
        for k, wk in enumerate(w):
            s = np.exp(1j * wk * self.Ts) if self.Ts > 0 else 1j * wk
            out[:, :, k] = _lti.freqresp(self.A, self.B, self.C, self.D, s)
        return out

    def evalfr(self, s: complex) -> np.ndarray:
        return _lti.freqresp(self.A, self.B, self.C, self.D, s)

    def poles(self) -> np.ndarray:
        return np.linalg.eigvals(self.A) if self.A.size else np.zeros(0, dtype=complex)

    def dcgain(self) -> np.ndarray:
        return np.real(self.evalfr(1.0 if self.Ts > 0 else 0.0))

    def to_lti(self) -> Any:
        """`spt.control.lti.StateSpace` nesnesine çevirir (LTI çekirdeği kuruluysa)."""
        try:
            from spt.control.lti import StateSpace  # type: ignore
        except ImportError as exc:
            raise ControlError(f"LTI çekirdeği yüklenemedi: {exc}") from None
        return StateSpace(self.A, self.B, self.C, self.D, Ts=self.Ts)


# ============================================================================ blok Jacobian'ı
def block_jacobian(b: Block, t: float, x: np.ndarray, xd: np.ndarray, u: list[np.ndarray], *,
                   analytic: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bloğun yerel doğrusallaştırması (A, B, C, D). Önce `b.linearize`, yoksa merkezi fark."""
    nin = sum(b.in_widths)
    nout = sum(b.out_widths)
    if analytic:
        r = b.linearize(t, x, xd, u)
        if r is not None:
            A, B, C, D = (np.atleast_2d(np.asarray(m, dtype=float)) for m in r)
            n = A.shape[0] if A.size else 0
            A = A.reshape(n, n) if n else np.zeros((0, 0))
            B = B.reshape(n, nin) if n else np.zeros((0, nin))
            C = C.reshape(nout, n) if n else np.zeros((nout, 0))
            D = D.reshape(nout, nin) if D.size else np.zeros((nout, nin))
            return A, B, C, D
    if b.nx and b.nxd:
        raise ControlError(f"'{b.name}' bloğu hem sürekli hem ayrık durum içeriyor; sayısal doğrusallaştırma "
                           "desteklenmez.")
    cont = b.nx > 0
    s0 = np.array(x if cont else xd, dtype=float)
    ns = s0.size
    widths = list(b.in_widths)
    uf = np.concatenate([np.asarray(v, dtype=float) for v in u]) if u else np.zeros(0)
    cuts = np.cumsum([0] + widths)

    def split(v: np.ndarray) -> list[np.ndarray]:
        return [v[cuts[k]:cuts[k + 1]].copy() for k in range(len(widths))]

    def fg(s: np.ndarray, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ul = split(uv)
        xs = s if cont else np.zeros(0)
        xds = s if not cont else np.asarray(xd, dtype=float)
        y = b.output(t, xs, xds, ul)
        yv = np.concatenate([np.asarray(v, dtype=float).ravel() for v in y]) if y else np.zeros(0)
        if ns:
            f = b.derivatives(t, xs, xds, ul) if cont else b.update(t, xs, xds, ul)
            f = np.asarray(f, dtype=float).ravel()
        else:
            f = np.zeros(0)
        return f, yv

    eps3 = np.finfo(float).eps ** (1.0 / 3.0)
    A = np.zeros((ns, ns))
    C = np.zeros((nout, ns))
    B = np.zeros((ns, nin))
    D = np.zeros((nout, nin))
    for j in range(ns):
        h = eps3 * max(1.0, abs(s0[j]))
        sp, sm = s0.copy(), s0.copy()
        sp[j] += h
        sm[j] -= h
        fp, yp = fg(sp, uf)
        fm, ym = fg(sm, uf)
        A[:, j] = (fp - fm) / (2 * h)
        C[:, j] = (yp - ym) / (2 * h)
    for j in range(nin):
        h = eps3 * max(1.0, abs(uf[j]))
        up, um = uf.copy(), uf.copy()
        up[j] += h
        um[j] -= h
        fp, yp = fg(s0, up)
        fm, ym = fg(s0, um)
        B[:, j] = (fp - fm) / (2 * h)
        D[:, j] = (yp - ym) / (2 * h)
    return A, B, C, D


# ============================================================================ çalışma noktası değerlendirmesi
def _set_lin_ctx(cm: CompiledModel, t: float) -> None:
    ctx = cm.ctx
    ctx.major = True
    ctx.zc = False
    ctx.freeze_events = False
    ctx.linearizing = True
    ctx.t_start = t


def _op_states(cm: CompiledModel, op: OperatingPoint | None) -> tuple[float, np.ndarray, list[np.ndarray]]:
    t = float(op.t) if op is not None else 0.0
    X = np.concatenate([np.asarray(b.x0(), dtype=float).ravel() for b in cm.blocks]) if cm.nx else np.zeros(0)
    XD = [np.asarray(b.xd0(), dtype=float).ravel().copy() for b in cm.blocks]
    if op is not None:
        for name, v in op.x.items():
            i = cm.index.get(name)
            if i is None:
                raise ControlError(f"Çalışma noktasındaki '{name}' bloğu modelde yok.")
            v = as_vec(v)
            if v.size != cm.blocks[i].nx:
                raise ControlError(f"'{name}' bloğunun sürekli durumu {cm.blocks[i].nx} elemanlı olmalı.")
            X[cm.xsl[i]] = v
        for name, v in op.xd.items():
            i = cm.index.get(name)
            if i is None:
                raise ControlError(f"Çalışma noktasındaki '{name}' bloğu modelde yok.")
            v = as_vec(v)
            if v.size != cm.blocks[i].nxd:
                raise ControlError(f"'{name}' bloğunun ayrık durumu {cm.blocks[i].nxd} elemanlı olmalı.")
            XD[i] = v.copy()
    return t, X, XD


def _set_inputs(cm: CompiledModel, inputs: dict[int, np.ndarray]) -> None:
    for port, i in cm.inports.items():
        b = cm.blocks[i]
        w = b.out_widths[0]
        v = as_vec(inputs.get(port, 0.0))
        if v.size not in (1, w):
            raise ControlError(f"{port} numaralı girişin değeri {w} elemanlı olmalı.")
        val = np.broadcast_to(v, (w,)).astype(float).copy()
        b.value_fn = lambda tt, _v=val: _v.copy()


def _eval_signals(cm: CompiledModel, t: float, X: np.ndarray, XD: list[np.ndarray],
                  z: dict[int, np.ndarray] | None = None) -> list[list[np.ndarray]]:
    """Çalışma noktasında tüm blok çıkışları. Kararlı hâlde çıkış = giriş olan bloklar (taşıma gecikmesi, bellek)
    için `z` verilmişse o değer, verilmemişse sabit nokta yinelemesi kullanılır."""
    blocks = cm.blocks
    Y = [[np.zeros(w) for w in b.out_widths] for b in blocks]
    passthru = [i for i, b in enumerate(blocks) if getattr(b, "steady_passthrough", False)]
    zz = {i: (z[i].copy() if z is not None and i in z else None) for i in passthru}

    def one_pass() -> None:
        for i in cm.order:
            b = blocks[i]
            xs = X[cm.xsl[i]] if b.nx else np.zeros(0)
            u = [Y[s][p] for s, p in cm.src[i]]
            if i in zz and zz[i] is not None:
                Y[i] = [zz[i].copy()]
                continue
            y = b.trim_output(t, xs, XD[i], u)
            if y is None:
                y = b.output(t, xs, XD[i], u)
            Y[i] = [np.asarray(v, dtype=float) for v in y]

    one_pass()
    if passthru and (z is None):
        for _ in range(200):
            diff = 0.0
            for i in passthru:
                s, p = cm.src[i][0]
                uval = np.broadcast_to(Y[s][p], (blocks[i].out_widths[0],)).astype(float)
                old = Y[i][0]
                diff = max(diff, float(np.max(np.abs(uval - old))) if uval.size else 0.0)
                zz[i] = uval.copy()
            one_pass()
            if diff <= 1e-13 * (1 + max(float(np.max(np.abs(zz[i]))) for i in passthru)):
                break
        else:
            raise ControlError("Çalışma noktasında taşıma gecikmesi / bellek bloklarının kararlı hâl değeri "
                               "bulunamadı (yineleme yakınsamadı); çalışma noktasını findop ile hesaplayın.")
    return Y


def _signals_from_op(cm: CompiledModel, op: OperatingPoint) -> list[list[np.ndarray]] | None:
    if not op.signals:
        return None
    Y = []
    for i, b in enumerate(cm.blocks):
        row = []
        for p, w in enumerate(b.out_widths):
            v = op.signals.get((cm.names[i], p))
            if v is None or np.size(v) != w:
                return None
            row.append(np.asarray(v, dtype=float).copy())
        Y.append(row)
    return Y


# ============================================================================ birleştirme
def _blkdiag(mats: list[np.ndarray], rows: list[int], cols: list[int]) -> np.ndarray:
    out = np.zeros((sum(rows), sum(cols)))
    r = c = 0
    for m, nr, nc in zip(mats, rows, cols):
        if nr and nc:
            out[r:r + nr, c:c + nc] = m
        r += nr
        c += nc
    return out


def _d2c_tustin(A, B, C, D, T: float):
    """Tustin tersi (MATLAB d2c 'tustin'): z = (1 + sT/2)/(1 − sT/2). M = (I + Ad)⁻¹ ile
    A = (2/T)·M(Ad − I), B = 2·M·Bd, C = (2/T)·Cd·M, D = Dd − Cd·M·Bd."""
    n = A.shape[0]
    I = np.eye(n)
    try:
        Mi = np.linalg.inv(A + I)
    except np.linalg.LinAlgError:
        raise ControlError("Tustin tersi tanımsız: ayrık blokta z = −1 kutbu var.") from None
    return (2.0 / T) * Mi @ (A - I), 2.0 * Mi @ B, (2.0 / T) * C @ Mi, D - C @ Mi @ B


def linearize(model: Model, io: list[LinPoint] | LinPoint, op: OperatingPoint | SimResult | None = None, *,
              Ts: float | None = None, rate_conversion: str = "zoh", prune: bool = True) -> LinearSystem:
    """Modeli `io` noktaları arasında `op` çalışma noktasında doğrusallaştırır.

    op: None (başlangıç durumları, kök girişler 0, t = 0), `OperatingPoint` (findop ya da benzetim sonu) ya da
    `SimResult` (son anı). Ts: hedef örnekleme zamanı (None: kural; 0: sürekli). rate_conversion: 'zoh' | 'tustin'.
    """
    if isinstance(io, LinPoint):
        io = [io]
    if not io:
        raise ControlError("En az bir doğrusallaştırma noktası gerekli.")
    if rate_conversion not in ("zoh", "tustin"):
        raise ControlError("Hız dönüşüm yöntemi 'zoh' ya da 'tustin' olmalı.")
    if isinstance(op, SimResult):
        op = op.final
    cm = compile_model(model)
    blocks = cm.blocks
    nb = len(blocks)
    # ---------------- noktalar
    pts = []
    seen = set()
    for pnt in io:
        if not isinstance(pnt, LinPoint):
            raise ControlError("Doğrusallaştırma noktaları linio(...) ile verilmeli.")
        i, p = cm.find_signal((pnt.block, pnt.port))
        if (i, p) in seen:
            raise ControlError(f"'{pnt.block}' bloğunun {p + 1}. çıkışında birden çok doğrusallaştırma noktası var; "
                               "tek noktada uygun türü (ör. 'inout', 'looptransfer') kullanın.")
        seen.add((i, p))
        pts.append((i, p, pnt.type, pnt.name or cm.signal_key(i, p)))
    if not any(t in _HAS_INPUT for _, _, t, _ in pts) or not any(t in _HAS_OUTPUT for _, _, t, _ in pts):
        raise ControlError("En az bir giriş türü ve bir çıkış türü doğrusallaştırma noktası gerekli.")
    # ---------------- çalışma noktası
    t, X, XD = _op_states(cm, op)
    _set_lin_ctx(cm, t)
    for b in blocks:
        b.reset()
    _set_inputs(cm, op.inputs if op is not None else {})
    Y = _signals_from_op(cm, op) if op is not None else None
    if Y is None:
        Y = _eval_signals(cm, t, X, XD)
    # ---------------- bağlantı yapısı
    dests: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for j in range(nb):
        for q, (s, p) in enumerate(cm.src[j]):
            dests.setdefault((s, p), []).append((j, q))
    opened = {(i, p) for i, p, typ, _ in pts if typ in _OPENS}
    # ---------------- yol üzerindeki bloklar
    keep = set(range(nb))
    if prune:
        succ = [set() for _ in range(nb)]
        pred = [set() for _ in range(nb)]
        for (s, p), lst in dests.items():
            if (s, p) in opened:
                continue
            for j, _ in lst:
                succ[s].add(j)
                pred[j].add(s)
        fwd: set[int] = set()
        stack = [j for i, p, typ, _ in pts if typ in _HAS_INPUT for j, _ in dests.get((i, p), [])]
        while stack:
            k = stack.pop()
            if k not in fwd:
                fwd.add(k)
                stack.extend(succ[k])
        bwd: set[int] = set()
        stack = [i for i, p, typ, _ in pts if typ in _HAS_OUTPUT]
        while stack:
            k = stack.pop()
            if k not in bwd:
                bwd.add(k)
                stack.extend(pred[k])
        keep = fwd & bwd
    # ---------------- yerel modeller
    loc: dict[int, tuple] = {}
    for i in range(nb):
        b = blocks[i]
        if i not in keep:
            continue
        xs = X[cm.xsl[i]] if b.nx else np.zeros(0)
        u = [Y[s][p] for s, p in cm.src[i]]
        A, B, C, D = block_jacobian(b, t, xs, XD[i], u)
        loc[i] = (A, B, C, D)
    # ---------------- hedef hız
    disc_rates = [blocks[i].ts.period for i in keep if blocks[i].ts.is_discrete]
    if Ts is None:
        Ts_t = max(disc_rates) if disc_rates else 0.0
    else:
        Ts_t = float(Ts)
        if Ts_t < 0 or not math.isfinite(Ts_t):
            raise ControlError("Hedef örnekleme zamanı 0 (sürekli) ya da pozitif olmalı.")
    # ---------------- sinyal / giriş indeksleri
    yoff, uoff = {}, {}
    ny = nu = 0
    for i in range(nb):
        for p, w in enumerate(blocks[i].out_widths):
            yoff[(i, p)] = (ny, w)
            ny += w
        for q, w in enumerate(blocks[i].in_widths):
            uoff[(i, q)] = (nu, w)
            nu += w
    w_off, z_off = [], []
    nw = nz = 0
    in_names, out_names = [], []
    for i, p, typ, name in pts:
        w = blocks[i].out_widths[p]
        if typ in _HAS_INPUT:
            w_off.append((i, p, nw, w))
            in_names += [name] if w == 1 else [f"{name}({k + 1})" for k in range(w)]
            nw += w
        if typ in _HAS_OUTPUT:
            z_off.append((i, p, nz, w, typ))
            out_names += [name] if w == 1 else [f"{name}({k + 1})" for k in range(w)]
            nz += w
    M = np.zeros((nu, ny))
    E = np.zeros((nu, nw))
    F = np.zeros((nz, ny))
    G = np.zeros((nz, nw))
    for (s, p), lst in dests.items():
        if s not in keep or (s, p) in opened:
            continue
        y0, w = yoff[(s, p)]
        for j, q in lst:
            if j not in keep:
                continue
            u0, wu = uoff[(j, q)]
            M[u0:u0 + wu, y0:y0 + w] = np.eye(w) if wu == w else np.ones((wu, w)) / w
    for i, p, w0, w in w_off:
        for j, q in dests.get((i, p), []):
            if j not in keep:
                continue
            u0, wu = uoff[(j, q)]
            E[u0:u0 + wu, w0:w0 + w] = np.eye(w)
    for i, p, z0, w, typ in z_off:
        y0, _ = yoff[(i, p)]
        if i in keep:
            F[z0:z0 + w, y0:y0 + w] = np.eye(w)
        if typ == "inout":
            wo = [o for o in w_off if o[0] == i and o[1] == p][0]
            G[z0:z0 + w, wo[2]:wo[2] + w] = np.eye(w)
    # ---------------- birimler: her blok x⁺ / x' = A x + By·y + Bw·w, y_blok = C x + Dy·y + Dw·w
    is_cont = {i: cm.rate_of[i] < 0 for i in range(nb)}        # sürekli ya da sabit hız
    units_by_rate: dict[float, list[_Unit]] = {}
    for i in sorted(keep):
        b = blocks[i]
        A, B, C, D = loc[i]
        ui = [k for q in range(len(b.in_widths)) for k in range(uoff[(i, q)][0], uoff[(i, q)][0] + uoff[(i, q)][1])]
        yi = [k for p in range(len(b.out_widths)) for k in range(yoff[(i, p)][0], yoff[(i, p)][0] + yoff[(i, p)][1])]
        Mi, Ei = M[ui, :], E[ui, :]
        n = A.shape[0]
        names = [cm.names[i]] if n == 1 else [f"{cm.names[i]}({k + 1})" for k in range(n)]
        rate = 0.0 if is_cont[i] else b.ts.period
        unit = _Unit(A, B @ Mi, B @ Ei, C, D @ Mi, D @ Ei, yi, names)
        if rate > 0 and Ts_t > 0 and rate > Ts_t:
            # hedef hızdan yavaş blok (yalnız Ts açıkça küçük verildiyse): blok düzeyinde dönüşüm
            unit = _convert(unit, rate, Ts_t, rate_conversion)
            rate = Ts_t
        units_by_rate.setdefault(rate, []).append(unit)
    cont_units = units_by_rate.pop(0.0, [])
    S: _Unit | None = None
    if Ts_t > 0:
        # sürekli kısım en hızlı ayrık hızda bütün olarak ayrıklaştırılır, o hızın bloklarıyla birleştirilir ve
        # grup olarak bir sonraki (yavaş) hıza taşınır … hedef hıza kadar (MATLAB linearize ile aynı sonuç)
        seq = sorted(set(units_by_rate) | {Ts_t})
        cur = 0.0
        S = _merge(cont_units) if cont_units else None
        for r in seq:
            if S is not None:
                S = _convert(S, cur, r, rate_conversion)
            S = _merge(([S] if S is not None else []) + units_by_rate.get(r, []))
            cur = r
    else:
        # sürekli hedef: en yavaş ayrık gruptan başlayarak hızlıya, en sonda bütün olarak sürekliye
        cur = None
        for r in sorted(units_by_rate, reverse=True):
            if S is not None:
                S = _convert(S, cur, r, rate_conversion)
            S = _merge(([S] if S is not None else []) + units_by_rate[r])
            cur = r
        if S is not None:
            S = _convert(S, cur, 0.0, rate_conversion)
        S = _merge(([S] if S is not None else []) + cont_units)
    if S is None:
        S = _Unit(np.zeros((0, 0)), np.zeros((0, ny)), np.zeros((0, nw)), np.zeros((0, 0)), np.zeros((0, ny)),
                  np.zeros((0, nw)), [], [])
    Ysel = S.y
    Fy = F[:, Ysel] if Ysel else np.zeros((nz, 0))
    Acl, Bcl = S.A, S.Bw
    Ccl = Fy @ S.C
    Dcl = Fy @ S.Dw + G
    return LinearSystem(Acl, Bcl, Ccl, Dcl, Ts_t, in_names, out_names, list(S.names))


@dataclass
class _Unit:
    """Doğrusal birim: x' (ya da x⁺) = A x + By·y + Bw·w;  y[ys] = C x + Dy·y + Dw·w (y: tüm sinyaller)."""
    A: np.ndarray
    By: np.ndarray
    Bw: np.ndarray
    C: np.ndarray
    Dy: np.ndarray
    Dw: np.ndarray
    y: list[int]
    names: list[str]


def _merge(units: list[_Unit]) -> _Unit:
    """Birimleri aynı hızda birleştirir; aralarındaki cebirsel bağıntılar çözülür (I − D_iç)⁻¹."""
    if len(units) == 1 and not units[0].y:
        return units[0]
    ns = [u.A.shape[0] for u in units]
    A = _blkdiag([u.A for u in units], ns, ns)
    By = np.vstack([u.By for u in units])
    Bw = np.vstack([u.Bw for u in units])
    C = _blkdiag([u.C for u in units], [len(u.y) for u in units], ns)
    Dy = np.vstack([u.Dy for u in units])
    Dw = np.vstack([u.Dw for u in units])
    Y = [k for u in units for k in u.y]
    names = [nm for u in units for nm in u.names]
    if not Y:
        return _Unit(A, By, Bw, C, Dy, Dw, Y, names)
    Dyy = Dy[:, Y]
    Dy_ext = Dy.copy()
    Dy_ext[:, Y] = 0.0
    By_int = By[:, Y]
    By_ext = By.copy()
    By_ext[:, Y] = 0.0
    T = np.eye(len(Y)) - Dyy
    cond = np.linalg.cond(T) if T.size else 1.0
    if not math.isfinite(cond) or cond > 1e14:
        raise ControlError("Doğrusallaştırmada çözülemeyen cebirsel döngü (I − D·M tekil): döngü kazancı tam 1 olan "
                           "doğrudan geçişli bir çevrim var.")
    n = A.shape[0]
    ny = Dy.shape[1]
    Q = np.linalg.solve(T, np.hstack([C, Dy_ext, Dw]))
    Qc, Qy, Qw = Q[:, :n], Q[:, n:n + ny], Q[:, n + ny:]
    return _Unit(A + By_int @ Qc, By_ext + By_int @ Qy, Bw + By_int @ Qw, Qc, Qy, Qw, Y, names)


def _convert(u: _Unit, T_from: float, T_to: float, method: str) -> _Unit:
    """Birimi hız değiştirir (0 = sürekli): girişler (y, w) dönüşüm aralığında tutulmuş kabul edilir (ZOH)."""
    if T_from == T_to:
        return u
    n = u.A.shape[0]
    ny = u.By.shape[1]
    Bfull = np.hstack([u.By, u.Bw])
    Dfull = np.hstack([u.Dy, u.Dw])
    used = np.flatnonzero(np.any(Bfull != 0, axis=0) | np.any(Dfull != 0, axis=0))
    B, D = Bfull[:, used], Dfull[:, used]
    A, C = u.A, u.C
    if n:
        if T_from == 0.0:
            A, B, C, D = _lti.c2d(A, B, C, D, T_to, method)
        elif T_to == 0.0:
            if method == "tustin":
                A, B, C, D = _d2c_tustin(A, B, C, D, T_from)
            else:
                A, B = _lti.d2c_zoh(A, B, T_from)
        elif method == "tustin":
            A, B, C, D = _d2c_tustin(A, B, C, D, T_from)
            A, B, C, D = _lti.c2d(A, B, C, D, T_to, "tustin")
        else:
            A, B = _lti.resample_zoh(A, B, T_from, T_to)
    Bn = np.zeros_like(Bfull)
    Dn = np.zeros_like(Dfull)
    Bn[:, used] = B
    Dn[:, used] = D
    return _Unit(A, Bn[:, :ny], Bn[:, ny:], C, Dn[:, :ny], Dn[:, ny:], list(u.y), list(u.names))


# ============================================================================ trim
@dataclass
class OperSpec:
    """Çalışma noktası belirtimi (MATLAB operspec karşılığı). Sözlük değerleri:
    states[blok]  = {"x": tahmin/değer, "known": bool ya da maske, "steady": bool ya da maske, "min", "max"}
    inputs[port]  = {"u": tahmin/değer, "known": bool ya da maske, "min", "max"}
    outputs[ref]  = {"y": hedef, "known": bool ya da maske, "min", "max"}   (ref: kök çıkış portu numarası ya da
                    sinyal adı / "Blok" / "Blok:2")
    """
    t: float = 0.0
    states: dict[str, dict[str, Any]] = field(default_factory=dict)
    inputs: dict[int, dict[str, Any]] = field(default_factory=dict)
    outputs: dict[Any, dict[str, Any]] = field(default_factory=dict)


def operspec(model: Model) -> OperSpec:
    """Varsayılan belirtim: durumlar başlangıç değerinde, bilinmiyor, kararlı hâl; kök girişler 0, serbest."""
    cm = compile_model(model)
    sp = OperSpec()
    for i, b in enumerate(cm.blocks):
        if b.nx:
            sp.states[cm.names[i]] = {"x": np.asarray(b.x0(), float), "known": False, "steady": True,
                                      "min": -np.inf, "max": np.inf}
        if b.nxd:
            sp.states[cm.names[i]] = {"x": np.asarray(b.xd0(), float), "known": False, "steady": True,
                                      "min": -np.inf, "max": np.inf}
    for port, i in cm.inports.items():
        w = cm.blocks[i].out_widths[0]
        sp.inputs[port] = {"u": np.zeros(w), "known": False, "min": -np.inf, "max": np.inf}
    return sp


@dataclass
class TrimResult:
    op: OperatingPoint
    success: bool
    max_residual: float
    message: str
    residuals: dict[str, np.ndarray]
    nfev: int


def _mask(v: Any, n: int) -> np.ndarray:
    a = np.atleast_1d(np.asarray(v, dtype=bool)).ravel()
    return np.broadcast_to(a, (n,)).copy() if a.size == 1 else a


def _vals(v: Any, n: int, label: str) -> np.ndarray:
    a = as_vec(v)
    if a.size not in (1, n):
        raise ControlError(f"{label}: {n} elemanlı olmalı.")
    return np.broadcast_to(a, (n,)).astype(float).copy()


def findop(model: Model, spec: OperSpec | None = None, *, tol: float = 1e-6, max_nfev: int | None = None) \
        -> TrimResult:
    """Kararlı hâl çalışma noktası (MATLAB findop karşılığı). Sonuç `TrimResult.op` doğrudan `linearize`'a ya da
    `simulate(..., initial=op)`'a verilebilir."""
    cm = compile_model(model)
    if spec is None:
        spec = operspec(model)
    t = float(spec.t)
    _set_lin_ctx(cm, t)
    blocks = cm.blocks
    for b in blocks:
        b.reset()
    t0, X0, XD0 = _op_states(cm, None)
    # ---------------- bilinmeyenler
    var_lo, var_hi, var_x0 = [], [], []
    slots: list[tuple[str, int, np.ndarray]] = []   # (tür, blok/port, maske)
    st_info = {}
    for i, b in enumerate(blocks):
        n = b.nx or b.nxd
        if not n:
            continue
        name = cm.names[i]
        s = spec.states.get(name, {})
        x = _vals(s.get("x", X0[cm.xsl[i]] if b.nx else XD0[i]), n, f"'{name}' durumu")
        known = _mask(s.get("known", False), n)
        steady = _mask(s.get("steady", True), n)
        lo = _vals(s.get("min", -np.inf), n, f"'{name}' alt sınırı")
        hi = _vals(s.get("max", np.inf), n, f"'{name}' üst sınırı")
        st_info[i] = (x, known, steady)
        free = ~known
        slots.append(("x", i, free))
        var_x0 += list(np.clip(x[free], lo[free], hi[free]))
        var_lo += list(lo[free])
        var_hi += list(hi[free])
    in_info = {}
    for port, i in cm.inports.items():
        w = blocks[i].out_widths[0]
        s = spec.inputs.get(port, {})
        u = _vals(s.get("u", 0.0), w, f"{port}. giriş")
        known = _mask(s.get("known", False), w)
        lo = _vals(s.get("min", -np.inf), w, f"{port}. giriş alt sınırı")
        hi = _vals(s.get("max", np.inf), w, f"{port}. giriş üst sınırı")
        in_info[port] = (u, known)
        free = ~known
        slots.append(("u", port, free))
        var_x0 += list(np.clip(u[free], lo[free], hi[free]))
        var_lo += list(lo[free])
        var_hi += list(hi[free])
    passthru = [i for i, b in enumerate(blocks) if getattr(b, "steady_passthrough", False)]
    for i in passthru:
        w = blocks[i].out_widths[0]
        slots.append(("z", i, np.ones(w, dtype=bool)))
        var_x0 += [0.0] * w
        var_lo += [-np.inf] * w
        var_hi += [np.inf] * w
    out_specs = []
    for ref, s in spec.outputs.items():
        if isinstance(ref, (int, np.integer)) and not isinstance(ref, bool):
            oi = cm.outports.get(int(ref))
            if oi is None:
                raise ControlError(f"{ref} numaralı kök çıkış portu yok.")
            si, sp_ = cm.src[oi][0]
            label = f"{ref}. çıkış"
        else:
            si, sp_ = cm.find_signal(ref)
            label = f"'{ref}'"
        w = blocks[si].out_widths[sp_]
        y = _vals(s.get("y", 0.0), w, f"{label} hedefi")
        known = _mask(s.get("known", "y" in s), w)
        lo = _vals(s.get("min", -np.inf), w, f"{label} alt sınırı")
        hi = _vals(s.get("max", np.inf), w, f"{label} üst sınırı")
        out_specs.append((si, sp_, y, known, lo, hi, label))
    z0 = np.array(var_x0, dtype=float)
    lb = np.array(var_lo, dtype=float)
    ub = np.array(var_hi, dtype=float)

    def unpack(zv: np.ndarray):
        X = X0.copy()
        XD = [v.copy() for v in XD0]
        U = {p: v[0].copy() for p, v in in_info.items()}
        Z = {}
        k = 0
        for kind, key, free in slots:
            nf = int(free.sum())
            if kind == "x":
                x, _, _ = st_info[key]
                v = x.copy()
                v[free] = zv[k:k + nf]
                if blocks[key].nx:
                    X[cm.xsl[key]] = v
                else:
                    XD[key] = v
            elif kind == "u":
                v = U[key]
                v[free] = zv[k:k + nf]
            else:
                Z[key] = zv[k:k + nf].copy()
            k += nf
        return X, XD, U, Z

    names_res: list[tuple[str, int]] = []

    def residual(zv: np.ndarray) -> np.ndarray:
        X, XD, U, Z = unpack(zv)
        _set_inputs(cm, U)
        Y = _eval_signals(cm, t, X, XD, Z)
        res = []
        names_res.clear()
        for i, (x, known, steady) in st_info.items():
            b = blocks[i]
            u = [Y[s][p] for s, p in cm.src[i]]
            if b.nx:
                d = np.asarray(b.derivatives(t, X[cm.xsl[i]], XD[i], u), float)
            else:
                d = np.asarray(b.update(t, np.zeros(0), XD[i], u), float) - XD[i]
            res.append(d[steady])
            names_res.append((cm.names[i], int(steady.sum())))
        for i in passthru:
            s, p = cm.src[i][0]
            uval = np.broadcast_to(Y[s][p], (blocks[i].out_widths[0],))
            res.append(Z[i] - uval)
            names_res.append((f"{cm.names[i]} (kararlı hâl)", uval.size))
        for si, sp_, y, known, lo, hi, label in out_specs:
            v = Y[si][sp_]
            r = np.where(known, v - y, 0.0)
            r = r + np.where(v > hi, v - hi, 0.0) + np.where(v < lo, v - lo, 0.0)
            res.append(r)
            names_res.append((label, v.size))
        return np.concatenate(res) if res else np.zeros(0)

    r0 = residual(z0)
    nfev = 1
    if z0.size and r0.size:
        try:
            sol = least_squares(residual, z0, bounds=(lb, ub), method="trf", xtol=1e-15, ftol=1e-15, gtol=1e-15,
                                x_scale="jac", max_nfev=max_nfev or 200 * (z0.size + 1))
        except ValueError as exc:
            raise ControlError(f"Çalışma noktası çözülemedi: {exc}") from None
        zsol = sol.x
        nfev += int(sol.nfev)
    else:
        zsol = z0
    rf = residual(zsol)
    X, XD, U, Z = unpack(zsol)
    _set_inputs(cm, U)
    Y = _eval_signals(cm, t, X, XD, Z)
    op = OperatingPoint(t)
    for i, b in enumerate(blocks):
        if b.nx:
            op.x[cm.names[i]] = X[cm.xsl[i]].copy()
        if b.nxd:
            op.xd[cm.names[i]] = XD[i].copy()
        for p in range(b.n_outputs()):
            op.signals[(cm.names[i], p)] = Y[i][p].copy()
    op.inputs = {p: v.copy() for p, v in U.items()}
    resid = {}
    k = 0
    for nm, n in names_res:
        resid[nm] = rf[k:k + n].copy()
        k += n
    mx = float(np.max(np.abs(rf))) if rf.size else 0.0
    ok = mx <= tol
    if ok:
        msg = f"Çalışma noktası koşulları sağlandı (en büyük artık {mx:.3g})."
    else:
        worst = max(resid.items(), key=lambda kv: float(np.max(np.abs(kv[1]))) if kv[1].size else 0.0)[0]
        msg = (f"Çalışma noktası koşulları tam sağlanamadı: en büyük artık {mx:.3g} ('{worst}'). Bilinen değerleri, "
               "sınırları ya da başlangıç tahminini gözden geçirin.")
    return TrimResult(op, ok, mx, msg, resid, nfev)


__all__ = ["linio", "LinPoint", "LinearSystem", "linearize", "block_jacobian", "OperSpec", "operspec", "findop",
           "TrimResult", "POINT_TYPES"]
