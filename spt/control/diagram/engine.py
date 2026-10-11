"""Blok diyagram benzetim motoru (Simulink anlamlarıyla).

Derleme (`compile_model`)
------------------------
1. Model kopyalanır ve alt sistemler düzleştirilir ("Alt/Blok" tam adları).
2. Sinyal genişlikleri yayılır (geri besleme döngülerinde eksik bilgiyle tahmin + son denetim).
3. Örnekleme zamanları yayılır: miras bloklar girişlerinden — herhangi biri sürekliyse sürekli; hepsi sabitse sabit;
   ayrıklar tek hızsa o hız, farklı hızlar birbirinin tam katıysa en hızlısı, değilse "örnekleme zamanı çatışması".
   Girişsiz miras bloklar süreklidir. Ayrık bloklar sürekli / sabit hız alamaz.
4. Yürütme sırası: doğrudan geçişli girişlere göre topolojik sıra (aynı düzeyde ekleme sırası korunur). Sıralanamayan
   bloklar **cebirsel döngü** oluşturur; döngüdeki bloklar adlarıyla bildirilir.

Benzetim (`simulate` / `Simulator`)
----------------------------------
Her ana adımda: (1) sıralı çıkışlar (ayrık bloklar yalnız örnekleme anlarında; arada çıkış tutulur), (2) kayıt,
(3) ayrık durum güncellemesi + iş belleği, (4) sürekli durumların çözücüyle ilerletilmesi; çözücü ara adımlarında
yalnız sürekli hızdaki bloklar yeniden hesaplanır (Simulink ana / ara adım anlamı).

Çözücüler: `ode1` (Euler) ve `ode4` (klasik Runge–Kutta 4) sabit adımlı — ayrık hızlar adımın tam katı olmalı;
`ode45` (Dormand–Prince 5(4), FSAL, MATLAB ode45 adım denetimi, 4. dereceden sürekli ara değerleme) değişken adımlı:
ayrık örnekleme anlarına, blok zaman olaylarına (Basamak anı, darbe kenarı, veri noktaları) ve istenen çıktı
anlarına tam basar; sıfır geçiş etkinse süreksiz bloklar ara adımda kiplerini dondurur ve kip sınırının geçildiği an
ara değerlemeyle (sekant + ikiye bölme) bulunup ana adım o ana konur.
"""
from __future__ import annotations

import heapq
import math
import time as _time
import warnings
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable

import numpy as np

from spt.control.diagram.block import (CONSTANT_TS, CONTINUOUS_TS, Block, SampleTime, SimContext, as_vec,
                                       block_class)
from spt.control.diagram.model import Model
from spt.control.errors import ControlError, ControlWarning

SOLVERS: dict[str, str] = {
    "ode1": "Euler (sabit adım)",
    "ode4": "Runge–Kutta 4 (sabit adım)",
    "ode45": "Dormand–Prince 5(4) (değişken adım)",
}
R_CONT = -1
R_CONST = -2
_EMPTY = np.zeros(0)


# ============================================================================ derlenmiş model
@dataclass
class CompiledModel:
    name: str
    blocks: list[Block]
    names: list[str]
    index: dict[str, int]
    src: list[list[tuple[int, int]]]
    order: list[int]
    rate_of: list[int]
    rates: list[SampleTime]
    xsl: list[slice]
    nx: int
    signal_entries: dict[tuple[int, int], dict[str, Any]]
    inports: dict[int, int]
    outports: dict[int, int]
    ctx: SimContext

    def signal_key(self, i: int, port: int) -> str:
        e = self.signal_entries.get((i, port))
        if e and e.get("name"):
            return e["name"]
        if self.blocks[i].n_outputs() == 1:
            return self.names[i]
        return f"{self.names[i]}:{port + 1}"

    def find_signal(self, ref: Any) -> tuple[int, int]:
        """Sinyal başvurusu → (blok, port): sinyal adı, "Blok" (1. çıkış), "Blok:2" (1 tabanlı) ya da (ad, port)."""
        if isinstance(ref, tuple) and len(ref) == 2:
            name, port = ref
            if isinstance(name, Block):
                name = name.name
            i = self.index.get(str(name))
            if i is None:
                raise ControlError(f"'{name}' adlı blok bulunamadı.")
            if not 0 <= int(port) < self.blocks[i].n_outputs():
                raise ControlError(f"'{name}' bloğunun {int(port) + 1}. çıkışı yok.")
            return i, int(port)
        if isinstance(ref, Block):
            ref = ref.name
        ref = str(ref)
        for (i, p), e in self.signal_entries.items():
            if e.get("name") == ref:
                return i, p
        if ref in self.index:
            i = self.index[ref]
            if self.blocks[i].n_outputs() < 1:
                raise ControlError(f"'{ref}' bloğunun çıkışı yok.")
            return i, 0
        if ":" in ref:
            name, _, p = ref.rpartition(":")
            if name in self.index and p.isdigit():
                return self.find_signal((name, int(p) - 1))
        raise ControlError(f"'{ref}' adlı sinyal ya da blok bulunamadı.")

    def is_continuous(self, i: int) -> bool:
        return self.rate_of[i] == R_CONT


# ============================================================================ düzleştirme
def _flatten(model: Model):
    from spt.control.diagram.blocks.ports import Subsystem, Wire
    blocks: list[tuple[str, Block]] = []
    conns: list[tuple[tuple[str, int], tuple[str, int]]] = []
    signals: dict[tuple[str, int], dict[str, Any]] = {}

    def rec(m: Model, prefix: str, root: bool) -> None:
        def out_ep(bname: str, port: int) -> tuple[str, int]:
            b = m.blocks[bname]
            if isinstance(b, Subsystem):
                return (f"{prefix}{bname}/{b.outports()[port].name}", 0)
            return (prefix + bname, port)

        def in_ep(bname: str, port: int) -> tuple[str, int]:
            b = m.blocks[bname]
            if isinstance(b, Subsystem):
                return (f"{prefix}{bname}/{b.inports()[port].name}", 0)
            return (prefix + bname, port)

        for name, b in m.blocks.items():
            full = prefix + name
            if isinstance(b, Subsystem):
                b.inports()
                b.outports()
                rec(b.system, full + "/", False)
            elif not root and b.type_name in ("Inport", "Outport"):
                w = Wire(full)
                w.meta = {"port_of": b.type_name}
                blocks.append((full, w))
            else:
                b.name = full
                blocks.append((full, b))
        for c in m.connections:
            if c.src not in m.blocks or c.dst not in m.blocks:
                raise ControlError(f"Bağlantı var olmayan bloğa işaret ediyor ({prefix}{c.src} → {prefix}{c.dst}).")
            conns.append((out_ep(c.src, c.src_port), in_ep(c.dst, c.dst_port)))
        for (bn, p), entry in m.signals.items():
            if bn not in m.blocks:
                continue
            e = dict(entry)
            if e.get("name") and prefix:
                e["name"] = prefix + e["name"]
            signals[out_ep(bn, p)] = e

    rec(model, "", True)
    return blocks, conns, signals


# ============================================================================ derleme
def _gcd_period(periods: list[float]) -> float:
    fr = [Fraction(p).limit_denominator(10 ** 9) for p in periods]
    g = fr[0]
    for f in fr[1:]:
        a, b = g, f
        # kesirlerin EBOB'u: gcd(a/b, c/d) = gcd(a·d, c·b) / (b·d)
        num = math.gcd(a.numerator * b.denominator, b.numerator * a.denominator)
        g = Fraction(num, a.denominator * b.denominator)
    return float(g)


def _multiple(a: float, b: float) -> bool:
    """a, b'nin tam katı mı (göreli 1e-9)?"""
    r = a / b
    return abs(r - round(r)) <= 1e-9 * max(1.0, r)


def compile_model(model: Model) -> CompiledModel:
    """Modeli benzetim / doğrusallaştırma için derler (modelin kendisi değişmez; kopyası kullanılır)."""
    if not isinstance(model, Model):
        raise ControlError("Derlenecek nesne bir blok diyagram modeli değil.")
    m = model.copy()
    flat, conns, sigs = _flatten(m)
    names = [n for n, _ in flat]
    blocks = [b for _, b in flat]
    index = {n: i for i, n in enumerate(names)}
    if len(index) != len(names):
        raise ControlError("Düzleştirilmiş modelde aynı tam ada sahip bloklar var.")
    n = len(blocks)
    for b in blocks:
        if type(b).__name__ != "Wire" and b.type_name and b.type_name != "Wire":
            block_class(b.type_name)  # tembel kayıtları doğrula
    # ---------------- bağlantılar
    src: list[list[tuple[int, int] | None]] = [[None] * b.n_inputs() for b in blocks]
    for (sn, sp), (dn, dp) in conns:
        si, di = index.get(sn), index.get(dn)
        if si is None or di is None:
            raise ControlError(f"Bağlantı bulunamayan bloğa işaret ediyor ({sn} → {dn}).")
        if not 0 <= sp < blocks[si].n_outputs():
            raise ControlError(f"'{sn}' bloğunun {sp + 1}. çıkışı yok.")
        if not 0 <= dp < blocks[di].n_inputs():
            raise ControlError(f"'{dn}' bloğunun {dp + 1}. girişi yok.")
        if src[di][dp] is not None:
            raise ControlError(f"'{dn}' bloğunun {dp + 1}. girişine birden çok sinyal bağlı.")
        src[di][dp] = (si, sp)
    for i, lst in enumerate(src):
        for p, s in enumerate(lst):
            if s is None:
                raise ControlError(f"'{names[i]}' bloğunun {p + 1}. girişi bağlanmamış.")
    srcs: list[list[tuple[int, int]]] = [list(s) for s in src]  # type: ignore[arg-type]
    # ---------------- genişlikler
    outw: list[list[int | None]] = [[None] * b.n_outputs() for b in blocks]

    def inw(i: int) -> list[int | None]:
        return [outw[s][p] for s, p in srcs[i]]

    def assign(i: int, ws: list[int | None]) -> bool:
        changed = False
        if len(ws) != len(outw[i]):
            raise ControlError(f"'{names[i]}' bloğu {len(outw[i])} çıkış bildirdi ama {len(ws)} genişlik verdi.")
        for p, w in enumerate(ws):
            if w is None:
                continue
            w = int(w)
            if w < 1:
                raise ControlError(f"'{names[i]}' bloğunun {p + 1}. çıkış genişliği geçersiz ({w}).")
            if outw[i][p] is None:
                outw[i][p] = w
                changed = True
            elif outw[i][p] != w:
                raise ControlError(f"'{names[i]}' bloğunda sinyal genişliği çelişkisi: {p + 1}. çıkış hem "
                                   f"{outw[i][p]} hem {w} elemanlı olmalı (bağlı sinyallerin genişliklerini "
                                   "denetleyin).")
        return changed

    for _ in range(4 * n + 10):
        progress = True
        while progress:
            progress = False
            for i in range(n):
                if assign(i, list(blocks[i].propagate_widths(inw(i), guess=False))):
                    progress = True
        if all(w is not None for ws in outw for w in ws):
            break
        guessed = False
        for i in range(n):
            if any(w is None for w in outw[i]):
                if assign(i, list(blocks[i].propagate_widths(inw(i), guess=True))):
                    guessed = True
                    break
        if not guessed:
            for i in range(n):
                outw[i] = [1 if w is None else w for w in outw[i]]
            break
    for i in range(n):
        final = blocks[i].propagate_widths(inw(i), guess=True)
        for p, (w0, w1) in enumerate(zip(outw[i], final)):
            if w1 is not None and int(w1) != w0:
                raise ControlError(f"'{names[i]}' bloğunda sinyal genişliği çelişkisi ({p + 1}. çıkış: {w0} ve "
                                   f"{w1}).")
    # ---------------- örnekleme zamanları
    declared = [b.sample_time() for b in blocks]
    res: list[SampleTime | None] = [None if s.is_inherited else s for s in declared]

    def rule(i: int, partial: bool) -> SampleTime | None:
        ins = [res[s] for s, _ in srcs[i]]
        if not ins:
            return CONTINUOUS_TS
        known = [x for x in ins if x is not None]
        if len(known) < len(ins) and not partial:
            if any(x.is_continuous for x in known):
                return CONTINUOUS_TS
            return None
        if not known:
            return None
        if any(x.is_continuous for x in known):
            return CONTINUOUS_TS
        disc = [x for x in known if x.is_discrete]
        if not disc:
            return CONSTANT_TS if len(known) == len(ins) else None
        fast = min(disc, key=lambda s: (s.period, s.offset))
        for d in disc:
            ok = _multiple(d.period, fast.period) and _multiple(abs(d.offset - fast.offset) + fast.period,
                                                               fast.period)
            if not ok:
                periods = ", ".join(sorted({f"{x.period:g}" for x in disc}))
                raise ControlError(f"Örnekleme zamanı çatışması: '{names[i]}' bloğu birbirinin tam katı olmayan "
                                   f"hızlardaki sinyalleri ({periods} s) birleştiriyor; bloğa açık bir örnekleme "
                                   "zamanı verin ya da hızları uyumlu seçin.")
        return fast

    inh = [i for i in range(n) if declared[i].is_inherited]
    for _ in range(4 * n + 10):
        progress = True
        while progress:
            progress = False
            for i in inh:
                if res[i] is None:
                    r = rule(i, False)
                    if r is not None:
                        res[i] = r
                        progress = True
        if all(r is not None for r in res):
            break
        done = False
        for i in inh:
            if res[i] is None:
                r = rule(i, True)
                if r is not None:
                    res[i] = r
                    done = True
                    break
        if not done:
            for i in inh:
                if res[i] is None:
                    res[i] = CONTINUOUS_TS
            break
    for _ in range(2 * n + 4):                      # tahminlerin tutarlılığı (sabit nokta)
        changed = False
        for i in inh:
            r = rule(i, False)
            if r is not None and r != res[i]:
                res[i] = r
                changed = True
        if not changed:
            break
    else:
        raise ControlError("Örnekleme zamanları tutarlı biçimde belirlenemedi (miras bloklardan oluşan döngü); "
                           "döngüdeki bir bloğa açık örnekleme zamanı verin.")
    for i, b in enumerate(blocks):
        ts = res[i]
        assert ts is not None
        allowed = b.allowed_sample_times()
        if allowed == "discrete" and not ts.is_discrete:
            why = "sürekli" if ts.is_continuous else "sabit"
            raise ControlError(f"'{names[i]}' bloğu ({b.display_name}) ayrık bir blok; {why} örnekleme zamanı "
                               "alamaz. Örnekleme zamanını (Ts > 0) açıkça verin ya da ayrık bir sinyale bağlayın.")
        if allowed == "continuous" and not ts.is_continuous:
            raise ControlError(f"'{names[i]}' bloğu ({b.display_name}) yalnız sürekli zamanda çalışır.")
        b.ts = ts
    # ---------------- kurulum
    ctx = SimContext()
    for i, b in enumerate(blocks):
        b.ctx = ctx
        b.in_widths = [int(outw[s][p]) for s, p in srcs[i]]  # type: ignore[arg-type]
        b.out_widths = [int(w) for w in outw[i]]  # type: ignore[arg-type]
        b.setup(list(b.in_widths), list(b.out_widths))
        if b.nx and not b.ts.is_continuous:
            raise ControlError(f"'{names[i]}' bloğunun sürekli durumu var ama örnekleme zamanı sürekli değil.")
        if b.nxd and not b.ts.is_discrete:
            raise ControlError(f"'{names[i]}' bloğunun ayrık durumu var ama örnekleme zamanı ayrık değil.")
    # ---------------- sıralama ve cebirsel döngü
    succ: list[list[int]] = [[] for _ in range(n)]
    indeg = [0] * n
    for i, b in enumerate(blocks):
        ft = list(b.feedthrough())
        if len(ft) != len(srcs[i]):
            raise ControlError(f"'{names[i]}' bloğu {len(srcs[i])} giriş için {len(ft)} doğrudan geçiş bayrağı "
                               "verdi.")
        for (s, _), f in zip(srcs[i], ft):
            if f:
                succ[s].append(i)
                indeg[i] += 1
    heap = [i for i in range(n) if indeg[i] == 0]
    heapq.heapify(heap)
    order: list[int] = []
    while heap:
        i = heapq.heappop(heap)
        order.append(i)
        for j in succ[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                heapq.heappush(heap, j)
    if len(order) < n:
        raise ControlError(_loop_message(names, succ, [i for i in range(n) if indeg[i] > 0]))
    # ---------------- hızlar ve durumlar
    rates: list[SampleTime] = sorted({b.ts for b in blocks if b.ts.is_discrete}, key=lambda s: (s.period, s.offset))
    rate_of = []
    for b in blocks:
        if b.ts.is_continuous:
            rate_of.append(R_CONT)
        elif b.ts.is_constant:
            rate_of.append(R_CONST)
        else:
            rate_of.append(rates.index(b.ts))
    xsl, k = [], 0
    for b in blocks:
        xsl.append(slice(k, k + b.nx))
        k += b.nx
    sig_entries = {}
    for (bn, p), e in sigs.items():
        i = index.get(bn)
        if i is not None and 0 <= p < blocks[i].n_outputs():
            sig_entries[(i, p)] = e
    inports, outports = {}, {}
    for i, b in enumerate(blocks):
        if b.type_name in ("Inport", "Outport"):
            d = inports if b.type_name == "Inport" else outports
            num = b.params["port"]
            if num in d:
                raise ControlError(f"Kök modelde iki {'giriş' if b.type_name == 'Inport' else 'çıkış'} portu aynı "
                                   f"numarayı ({num}) kullanıyor.")
            d[num] = i
    return CompiledModel(m.name, blocks, names, index, srcs, order, rate_of, rates, xsl, k, sig_entries, inports,
                         outports, ctx)


def _loop_message(names: list[str], succ: list[list[int]], remaining: list[int]) -> str:
    """Sıralanamayan bloklarda bir döngü bulur ve Türkçe ileti üretir."""
    rem = set(remaining)
    # yalnız döngü üzerindeki düğümler: kalanlardan başlayıp kalanlar içinde DFS ile çevrim
    start = remaining[0]
    for s0 in remaining:
        path, pos = [], {}
        node = s0
        while node not in pos:
            pos[node] = len(path)
            path.append(node)
            nxt = [j for j in succ[node] if j in rem]
            if not nxt:
                break
            node = nxt[0]
        else:
            cyc = path[pos[node]:]
            start = None
            break
    if start is not None:  # pragma: no cover - kalan küme her zaman döngü içerir
        cyc = remaining
    seq = " → ".join(f"'{names[i]}'" for i in cyc + [cyc[0]])
    return (f"Cebirsel döngü: {seq} ({len(cyc)} blok). Bu bloklar çıkışlarını aynı anda girişlerinden "
            "hesaplıyor (doğrudan geçiş); döngüye durum içeren bir blok (İntegratör, Birim gecikme, Bellek, transfer "
            "fonksiyonu …) ekleyin.")


# ============================================================================ sonuç
@dataclass
class OperatingPoint:
    """Çalışma noktası: zaman, blok durumları (tam adlarla), kök giriş değerleri ve isteğe bağlı tüm sinyaller."""
    t: float = 0.0
    x: dict[str, np.ndarray] = field(default_factory=dict)
    xd: dict[str, np.ndarray] = field(default_factory=dict)
    inputs: dict[int, np.ndarray] = field(default_factory=dict)
    signals: dict[tuple[str, int], np.ndarray] = field(default_factory=dict)

    def copy(self) -> "OperatingPoint":
        return OperatingPoint(self.t, {k: v.copy() for k, v in self.x.items()},
                              {k: v.copy() for k, v in self.xd.items()},
                              {k: np.array(v, dtype=float) for k, v in self.inputs.items()},
                              {k: v.copy() for k, v in self.signals.items()})


@dataclass
class ScopeData:
    t: np.ndarray
    data: list[np.ndarray]
    labels: list[str]


@dataclass
class SimResult:
    """Benzetim sonucu. `t` (nt,), `signals[ad]` (nt, genişlik), `scopes[blok]`, `outputs[port]` (nt, genişlik)."""
    t: np.ndarray
    signals: dict[str, np.ndarray]
    scopes: dict[str, ScopeData]
    outputs: dict[int, np.ndarray]
    final: OperatingPoint
    stats: dict[str, Any]
    cancelled: bool = False

    def signal(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """(t, değer); tek elemanlı sinyalde değer (nt,) biçimindedir."""
        if name not in self.signals:
            raise ControlError(f"'{name}' sinyali kaydedilmedi; kayıt için sinyali işaretleyin (log) ya da "
                               "log_all=True verin.")
        y = self.signals[name]
        return self.t, (y[:, 0] if y.shape[1] == 1 else y)

    def operating_point(self) -> OperatingPoint:
        """Benzetimin son anındaki çalışma noktası (doğrusallaştırma / devam için)."""
        return self.final.copy()


# ============================================================================ Dormand–Prince katsayıları
_C = np.array([0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0])
_A = [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
]
_E = np.array([71 / 57600, 0.0, -71 / 16695, 71 / 1920, -17253 / 339200, 22 / 525, -1 / 40])
_BI = np.array([                       # MATLAB ntrp45: y(t+s·h) = y + h·K·BI·[s; s²; s³; s⁴]
    [1.0, -183 / 64, 37 / 12, -145 / 128],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 1500 / 371, -1000 / 159, 1000 / 371],
    [0.0, -125 / 32, 125 / 12, -375 / 64],
    [0.0, 9477 / 3392, -729 / 106, 25515 / 6784],
    [0.0, -11 / 7, 11 / 3, -55 / 28],
    [0.0, 3 / 2, -4.0, 5 / 2],
])


# ============================================================================ simülatör
def _input_fn(spec: Any, width: int, label: str) -> Callable[[float], np.ndarray]:
    """Kök giriş tanımı → t'nin fonksiyonu: sabit, (t, değerler) (doğrusal, uçlarda tutma) ya da çağrılabilir."""
    if callable(spec):
        def f(t: float, _s=spec) -> np.ndarray:
            v = np.atleast_1d(np.asarray(_s(t), dtype=float)).ravel()
            return np.broadcast_to(v, (width,)).copy() if v.size == 1 else v
        return f
    if isinstance(spec, tuple) and len(spec) == 2:
        tt = np.asarray(spec[0], dtype=float).ravel()
        vv = np.asarray(spec[1], dtype=float)
        if vv.ndim == 1:
            vv = vv.reshape(-1, 1)
        if vv.shape[0] != tt.size or vv.shape[1] not in (1, width) or np.any(np.diff(tt) < 0):
            raise ControlError(f"{label}: (t, değerler) biçimi hatalı (satır sayısı = zaman sayısı, {width} kanal, "
                               "artan zaman).")

        def g(t: float) -> np.ndarray:
            out = np.array([np.interp(t, tt, vv[:, j]) for j in range(vv.shape[1])])
            return np.broadcast_to(out, (width,)).copy() if out.size == 1 else out
        return g
    v = as_vec(spec)
    if v.size not in (1, width):
        raise ControlError(f"{label}: {width} elemanlı değer bekleniyordu.")
    const = np.broadcast_to(v, (width,)).astype(float).copy()
    return lambda t: const.copy()


class Simulator:
    """Derlenmiş model üzerinde benzetim (arayüz için ilerleme / iptal destekli).

    inputs: kök Giriş portları için {port numarası ya da blok adı: sabit | (t, değerler) | f(t)}.
    initial: başlangıç durumu ({blok tam adı: değer} ya da `OperatingPoint`); verilmeyenler blok parametresinden.
    """

    def __init__(self, model: Model | CompiledModel, *, inputs: dict | None = None,
                 initial: OperatingPoint | dict | None = None):
        self.cm = model if isinstance(model, CompiledModel) else compile_model(model)
        cm = self.cm
        self.blocks = cm.blocks
        for key, spec in (inputs or {}).items():
            if isinstance(key, str):
                i = cm.index.get(key)
                if i is None or cm.blocks[i].type_name != "Inport":
                    raise ControlError(f"'{key}' adlı kök giriş portu yok.")
            else:
                i = cm.inports.get(int(key))
                if i is None:
                    raise ControlError(f"{key} numaralı kök giriş portu yok.")
            b = cm.blocks[i]
            b.value_fn = _input_fn(spec, b.out_widths[0], f"'{b.name}' girişi")
        self._initial = initial

    # ------------------------------------------------------------------ hazırlık
    def _setup_run(self, t0: float) -> None:
        cm = self.cm
        cm.ctx.t_start = t0
        cm.ctx.major = True
        cm.ctx.linearizing = False
        for b in self.blocks:
            b.reset()
        self.X = np.concatenate([np.asarray(b.x0(), dtype=float).ravel() for b in self.blocks]) if cm.nx \
            else np.zeros(0)
        self.XD = [np.asarray(b.xd0(), dtype=float).ravel().copy() for b in self.blocks]
        for i, b in enumerate(self.blocks):
            if self.XD[i].size != b.nxd:
                raise ControlError(f"'{cm.names[i]}' bloğunun başlangıç ayrık durumu {b.nxd} elemanlı olmalı.")
        if self.X.size != cm.nx:
            raise ControlError("Başlangıç sürekli durum boyutu tutarsız.")
        init = self._initial
        if init is not None:
            xs = init.x if isinstance(init, OperatingPoint) else init
            xds = init.xd if isinstance(init, OperatingPoint) else {}
            for name, val in xs.items():
                i = cm.index.get(name)
                if i is None:
                    raise ControlError(f"Başlangıç durumu verilen '{name}' bloğu yok.")
                b = self.blocks[i]
                v = as_vec(val)
                if b.nx and v.size == b.nx:
                    self.X[cm.xsl[i]] = v
                elif b.nxd and v.size == b.nxd:
                    self.XD[i] = v.copy()
                else:
                    raise ControlError(f"'{name}' bloğunun başlangıç durumu {b.nx or b.nxd} elemanlı olmalı.")
            for name, val in xds.items():
                i = cm.index.get(name)
                if i is None:
                    raise ControlError(f"Başlangıç durumu verilen '{name}' bloğu yok.")
                v = as_vec(val)
                if v.size != self.blocks[i].nxd:
                    raise ControlError(f"'{name}' bloğunun ayrık durumu {self.blocks[i].nxd} elemanlı olmalı.")
                self.XD[i] = v.copy()
            if isinstance(init, OperatingPoint):
                for port, val in init.inputs.items():
                    i = cm.inports.get(int(port))
                    if i is not None and self.blocks[i].value_fn is None:
                        b = self.blocks[i]
                        b.value_fn = _input_fn(val, b.out_widths[0], f"'{b.name}' girişi")
        self.Y = [[np.zeros(w) for w in b.out_widths] for b in self.blocks]
        self._cont_order = [i for i in cm.order if cm.rate_of[i] == R_CONT]
        self._deriv = [i for i in cm.order if self.blocks[i].nx]
        self._proj = [type(b).project is not Block.project for b in self.blocks]
        self._zc = [i for i in cm.order if cm.rate_of[i] == R_CONT
                    and type(self.blocks[i]).zero_crossings is not Block.zero_crossings]
        self._evt = [i for i, b in enumerate(self.blocks) if type(b).next_event is not Block.next_event]
        self._checked = False

    def _u(self, i: int) -> list[np.ndarray]:
        Y = self.Y
        return [Y[s][p] for s, p in self.cm.src[i]]

    # ------------------------------------------------------------------ ana adım
    def _major(self, t: float, hits: list[bool]) -> None:
        cm = self.cm
        cm.ctx.major = True
        X, XD, blocks, rate_of, xsl = self.X, self.XD, self.blocks, cm.rate_of, cm.xsl
        first = not self._checked
        for i in cm.order:
            r = rate_of[i]
            if r >= 0 and not hits[r]:
                # ilk ana adımda örnekleme anı gelmemiş (ofsetli) ayrık bloğun çıkışı: doğrudan geçişi yoksa
                # durumundan (ör. birim gecikmenin başlangıç koşulu), varsa sıfır
                if first and not any(blocks[i].feedthrough()):
                    y0 = blocks[i].output(t, _EMPTY, XD[i], self._u(i))
                    self._check_out(i, y0)
                    self.Y[i] = y0
                continue
            b = blocks[i]
            if b.nx:
                xs = X[xsl[i]]
                if self._proj[i]:
                    pr = b.project(xs)
                    if pr is not None:
                        X[xsl[i]] = pr
                        xs = X[xsl[i]]
            else:
                xs = _EMPTY
            u = self._u(i)
            b.major(t, xs, XD[i], u)
            y = b.output(t, xs, XD[i], u)
            if not self._checked:
                self._check_out(i, y)
            self.Y[i] = y
        self._checked = True

    def _check_out(self, i: int, y: Any) -> None:
        b = self.blocks[i]
        if not isinstance(y, list) or len(y) != b.n_outputs():
            raise ControlError(f"'{self.cm.names[i]}' bloğu {b.n_outputs()} çıkış yerine {len(y)} döndürdü.")
        for p, (v, w) in enumerate(zip(y, b.out_widths)):
            if not isinstance(v, np.ndarray) or v.shape != (w,):
                raise ControlError(f"'{self.cm.names[i]}' bloğunun {p + 1}. çıkışı {w} elemanlı olmalı "
                                   f"({np.shape(v)} döndü).")

    def _update(self, t: float, hits: list[bool]) -> None:
        cm = self.cm
        X, XD, blocks, rate_of, xsl = self.X, self.XD, self.blocks, cm.rate_of, cm.xsl
        new: list[tuple[int, np.ndarray]] = []
        for i in cm.order:
            r = rate_of[i]
            if r == R_CONST or (r >= 0 and not hits[r]):
                continue
            b = blocks[i]
            xs = X[xsl[i]] if b.nx else _EMPTY
            u = self._u(i)
            if b.nxd:
                v = np.asarray(b.update(t, xs, XD[i], u), dtype=float).ravel()
                if v.size != b.nxd:
                    raise ControlError(f"'{cm.names[i]}' bloğunun güncellenen durumu {b.nxd} elemanlı olmalı.")
                new.append((i, v))
            b.commit(t, xs, XD[i], u, self.Y[i])
        for i, v in new:
            XD[i] = v

    # ------------------------------------------------------------------ ara adım
    def _derivs(self, t: float, X: np.ndarray) -> np.ndarray:
        cm = self.cm
        dX = np.empty(cm.nx)
        xsl, XD, blocks = cm.xsl, self.XD, self.blocks
        for i in self._deriv:
            sl = xsl[i]
            dX[sl] = blocks[i].derivatives(t, X[sl], XD[i], self._u(i))
        return dX

    def _minor_outputs(self, t: float, X: np.ndarray) -> None:
        cm = self.cm
        cm.ctx.major = False
        xsl, XD, blocks = cm.xsl, self.XD, self.blocks
        for i in self._cont_order:
            b = blocks[i]
            self.Y[i] = b.output(t, X[xsl[i]] if b.nx else _EMPTY, XD[i], self._u(i))

    def _f(self, t: float, X: np.ndarray) -> np.ndarray:
        self._minor_outputs(t, X)
        return self._derivs(t, X)

    def _zc_values(self, t: float, X: np.ndarray) -> np.ndarray:
        cm = self.cm
        out = []
        for i in self._zc:
            b = self.blocks[i]
            z = b.zero_crossings(t, X[cm.xsl[i]] if b.nx else _EMPTY, self.XD[i], self._u(i))
            if z is not None:
                out.append(np.asarray(z, dtype=float).ravel())
        return np.concatenate(out) if out else _EMPTY

    # ------------------------------------------------------------------ kayıt
    def _log_init(self, log_all: bool, extra: list) -> None:
        cm = self.cm
        keys: dict[str, tuple[int, int]] = {}
        for (i, p), e in cm.signal_entries.items():
            if e.get("log"):
                keys[cm.signal_key(i, p)] = (i, p)
        if log_all:
            for i, b in enumerate(self.blocks):
                for p in range(b.n_outputs()):
                    keys.setdefault(cm.signal_key(i, p), (i, p))
        for ref in extra:
            i, p = cm.find_signal(ref)
            key = ref if isinstance(ref, str) else cm.signal_key(i, p)
            keys[key] = (i, p)
        self._log_keys = keys
        self._log_t: list[float] = []
        self._log_sig: dict[str, list[np.ndarray]] = {k: [] for k in keys}
        self._scopes = [i for i, b in enumerate(self.blocks) if b.type_name == "Scope"]
        self._log_scope: dict[int, list[list[np.ndarray]]] = {i: [[] for _ in self.cm.src[i]] for i in self._scopes}
        self._log_out: dict[int, list[np.ndarray]] = {p: [] for p in cm.outports}

    def _log(self, t: float) -> None:
        self._log_t.append(t)
        Y = self.Y
        for k, (i, p) in self._log_keys.items():
            self._log_sig[k].append(Y[i][p].copy())
        for i in self._scopes:
            for j, (s, p) in enumerate(self.cm.src[i]):
                self._log_scope[i][j].append(Y[s][p].copy())
        for port, i in self.cm.outports.items():
            s, p = self.cm.src[i][0]
            self._log_out[port].append(Y[s][p].copy())

    def _result(self, t_end: float, stats: dict, cancelled: bool) -> SimResult:
        cm = self.cm
        t = np.array(self._log_t)

        def stack(lst: list[np.ndarray], w: int) -> np.ndarray:
            return np.vstack(lst) if lst else np.zeros((0, w))
        signals = {k: stack(v, self.blocks[self._log_keys[k][0]].out_widths[self._log_keys[k][1]])
                   for k, v in self._log_sig.items()}
        scopes = {}
        for i in self._scopes:
            b = self.blocks[i]
            scopes[cm.names[i]] = ScopeData(t, [stack(v, w) for v, w in zip(self._log_scope[i], b.in_widths)],
                                            b.channel_labels())
        outputs = {port: stack(v, self.blocks[cm.outports[port]].in_widths[0]) for port, v in self._log_out.items()}
        op = OperatingPoint(t_end)
        for i, b in enumerate(self.blocks):
            if b.nx:
                op.x[cm.names[i]] = self.X[cm.xsl[i]].copy()
            if b.nxd:
                op.xd[cm.names[i]] = self.XD[i].copy()
            for p in range(b.n_outputs()):
                op.signals[(cm.names[i], p)] = self.Y[i][p].copy()
        for port, i in cm.inports.items():
            op.inputs[port] = self.Y[i][0].copy()
        return SimResult(t, signals, scopes, outputs, op, stats, cancelled)

    # ------------------------------------------------------------------ çalıştırma
    def run(self, stop_time: float, *, solver: str = "ode4", step: float | None = None, start_time: float = 0.0,
            rtol: float = 1e-3, atol: float | str = "auto", max_step: float | None = None,
            initial_step: float | None = None, zero_crossing: bool = True, output_times: Any = None,
            decimation: int = 1, log_all: bool = False, log: list | tuple = (),
            progress: Callable[[float, float], Any] | None = None, cancel: Any = None,
            max_steps: int = 50_000_000) -> SimResult:
        if solver not in SOLVERS:
            raise ControlError(f"Bilinmeyen çözücü '{solver}'; geçerliler: {', '.join(SOLVERS)}.")
        t0, tf = float(start_time), float(stop_time)
        if not (math.isfinite(t0) and math.isfinite(tf)) or tf <= t0:
            raise ControlError("Bitiş zamanı başlangıçtan büyük ve sonlu olmalı.")
        if decimation < 1:
            raise ControlError("Seyreltme (decimation) en az 1 olmalı.")
        ot = None
        if output_times is not None:
            ot = np.unique(np.asarray(output_times, dtype=float).ravel())
            if ot.size == 0 or ot[0] < t0 - 1e-12 * max(1.0, abs(t0)) or ot[-1] > tf + 1e-12 * max(1.0, abs(tf)):
                raise ControlError("Çıktı zamanları [başlangıç, bitiş] aralığında olmalı.")
        self._setup_run(t0)
        self._log_init(log_all, list(log))
        wall0 = _time.perf_counter()
        self._cancel_fn = _cancel_checker(cancel)
        self._progress = progress
        self._last_prog = wall0
        if solver == "ode45":
            stats, cancelled = self._run_variable(t0, tf, rtol, atol, max_step, initial_step, zero_crossing, ot,
                                                  max_steps)
        else:
            stats, cancelled = self._run_fixed(t0, tf, solver, step, ot, decimation, max_steps)
        stats["solver"] = solver
        stats["wall_time"] = _time.perf_counter() - wall0
        return self._result(self._t_end, stats, cancelled)

    def _tick(self, t: float, t0: float, tf: float) -> bool:
        """İlerleme bildirimi ve iptal denetimi; iptal istendiyse True."""
        now = _time.perf_counter()
        if self._progress is not None and now - self._last_prog > 0.1:
            self._last_prog = now
            self._progress((t - t0) / (tf - t0), t)
        return bool(self._cancel_fn and self._cancel_fn())

    def _hits(self, t: float) -> list[bool]:
        return [r.hit(t) for r in self.cm.rates]

    def _auto_step(self, t0: float, tf: float) -> float:
        rates = self.cm.rates
        span = (tf - t0) / 50.0
        if rates:
            base = _gcd_period([r.period for r in rates] + [r.offset for r in rates if r.offset > 0])
            if self.cm.nx and base > span:
                return base / math.ceil(base / span - 1e-9)
            return base
        return span

    def _run_fixed(self, t0, tf, solver, step, ot, decimation, max_steps):
        cm = self.cm
        h = float(step) if step is not None else self._auto_step(t0, tf)
        if not (h > 0 and math.isfinite(h)):
            raise ControlError("Sabit adım pozitif olmalı.")
        for r in cm.rates:
            if not _multiple(r.period, h) or (r.offset and not _multiple(r.offset, h)):
                raise ControlError(f"Sabit adımlı çözücüde ayrık örnekleme zamanı (Ts = {r.period:g} s"
                                   + (f", ofset {r.offset:g} s" if r.offset else "")
                                   + f") adımın (h = {h:g} s) tam katı olmalı.")
            if not _multiple(t0 + r.period, h):
                raise ControlError("Başlangıç zamanı adımın tam katı olmalı (ayrık hızlarla hizalı).")
        nsteps = int(math.floor((tf - t0) / h + 1e-9))
        times = [t0 + k * h for k in range(nsteps + 1)]
        if tf - times[-1] > 1e-9 * h:
            times.append(tf)
        else:
            times[-1] = tf
        if len(times) > max_steps:
            raise ControlError(f"Adım sayısı ({len(times)}) üst sınırı ({max_steps}) aşıyor; adımı büyütün.")
        log_set = None
        if ot is not None:
            log_set = set()
            arr = np.array(times)
            for tt in ot:
                k = int(np.argmin(np.abs(arr - tt)))
                if abs(arr[k] - tt) > 1e-9 * max(h, abs(tt)):
                    raise ControlError(f"Çıktı zamanı {tt:g} s sabit adım ızgarasında değil (h = {h:g} s); ode45 "
                                       "kullanın ya da adımı uyumlu seçin.")
                log_set.add(k)
        rk4 = solver == "ode4"
        cm.ctx.zc = False
        cm.ctx.freeze_events = False
        cancelled = False
        last = len(times) - 1
        for k, t in enumerate(times):
            hits = self._hits(t)
            self._major(t, hits)
            if (log_set is None and (k % decimation == 0 or k == last)) or (log_set is not None and k in log_set):
                self._log(t)
            if k == last:
                break
            # Simulink sırası: çıkışlar → güncelleme → türevler; türevler ana adım çıkış aşamasının sinyalleriyle
            # (güncelleme öncesi iş belleğiyle) hesaplanır, bu yüzden k1 güncellemeden önce alınır.
            if cm.nx:
                k1 = self._derivs(t, self.X)
            self._update(t, hits)
            hk = times[k + 1] - t
            if cm.nx:
                X = self.X
                if rk4:
                    k2 = self._f(t + hk / 2, X + (hk / 2) * k1)
                    k3 = self._f(t + hk / 2, X + (hk / 2) * k2)
                    k4 = self._f(t + hk, X + hk * k3)
                    Xn = X + (hk / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
                else:
                    Xn = X + hk * k1
                if not np.all(np.isfinite(Xn)):
                    raise ControlError(_blowup(t + hk))
                self.X = Xn
            if k % 64 == 0 and self._tick(t, t0, tf):
                cancelled = True
                break
        self._t_end = self._log_t[-1] if self._log_t else t0
        return {"steps": len(times) - 1, "step": h, "rejected": 0, "zc_events": 0}, cancelled

    # ------------------------------------------------------------------ değişken adım
    def _run_variable(self, t0, tf, rtol, atol, max_step, h0, zc_on, ot, max_steps):
        cm = self.cm
        if not (0 < rtol < 1):
            raise ControlError("Göreli tolerans 0 ile 1 arasında olmalı.")
        nx = cm.nx
        auto_atol = isinstance(atol, str)
        if auto_atol:
            if atol != "auto":
                raise ControlError("Mutlak tolerans sayı ya da 'auto' olmalı.")
            maxabs = np.abs(self.X).copy()
            atol_v = np.maximum(1e-6, rtol * maxabs)
        else:
            atol_v = np.broadcast_to(as_vec(atol), (nx,)).astype(float).copy() if nx else np.zeros(0)
            if np.any(atol_v <= 0):
                raise ControlError("Mutlak tolerans pozitif olmalı.")
        hmax = float(max_step) if max_step else (tf - t0) / 50.0
        cm.ctx.zc = bool(zc_on)
        cm.ctx.freeze_events = True
        has_zc = bool(self._zc)
        zc_resume = -math.inf
        warned = False
        t = t0
        ot_i = 0
        stats = {"steps": 0, "rejected": 0, "zc_events": 0, "f_evals": 0}
        tol_t = lambda tt: 1e-12 * max(1.0, abs(tt))  # noqa: E731

        def next_target(tt: float) -> float:
            tgt = tf
            for r in cm.rates:
                tgt = min(tgt, r.next_hit(tt))
            for i in self._evt:
                e = self.blocks[i].next_event(tt)
                if e > tt + tol_t(tt):
                    tgt = min(tgt, e)
            if ot is not None:
                nonlocal ot_i
                while ot_i < ot.size and ot[ot_i] <= tt + tol_t(tt):
                    ot_i += 1
                if ot_i < ot.size:
                    tgt = min(tgt, ot[ot_i])
            return tgt

        log_j = 0

        def outputs_at(tt: float) -> list[bool]:
            nonlocal log_j
            hits = self._hits(tt)
            self._major(tt, hits)
            if ot is None:
                self._log(tt)
            else:                                  # her çıktı anı yalnız bir kez kaydedilir
                while log_j < ot.size and ot[log_j] < tt - 10 * tol_t(tt):
                    log_j += 1
                if log_j < ot.size and abs(ot[log_j] - tt) <= 10 * tol_t(tt):
                    self._log(tt)
                    log_j += 1
            return hits

        h = None
        consecutive_zc = 0
        cancelled = False
        while True:
            active = bool(zc_on) and t >= zc_resume
            cm.ctx.zc = active
            detect_zc = active and has_zc
            hits = outputs_at(t)
            if t >= tf - tol_t(tf):
                break
            if stats["steps"] >= max_steps:
                raise ControlError(f"Adım sayısı üst sınırına ({max_steps}) ulaşıldı (t = {t:.6g} s).")
            X = self.X
            if nx:
                f0 = self._derivs(t, X)            # güncelleme öncesi sinyallerle (Simulink sırası)
                stats["f_evals"] += 1
                zc0 = self._zc_values(t, X) if detect_zc else _EMPTY
            self._update(t, hits)
            target = next_target(t)
            if nx == 0:
                t = min(target, t + hmax)
                stats["steps"] += 1
                if stats["steps"] % 64 == 0 and self._tick(t, t0, tf):
                    cancelled = True
                    break
                continue
            hmin = 16 * np.finfo(float).eps * max(abs(t), 1e-300)
            if h is None:
                if h0:
                    h = float(h0)
                else:
                    thr = atol_v / rtol
                    rh = np.max(np.abs(f0) / np.maximum(np.abs(X), thr)) / (0.8 * rtol ** 0.2)
                    h = min(hmax, tf - t0)
                    if h * rh > 1:
                        h = 1.0 / rh
                h = max(h, hmin)
            h = min(h, hmax)
            dist = target - t
            if 1.1 * h >= dist:
                h = dist
            failed = False
            K = np.empty((7, nx))
            K[0] = f0
            while True:
                for s in range(1, 7):
                    acc = X.copy()
                    for j, a in enumerate(_A[s]):
                        if a:
                            acc += (h * a) * K[j]
                    if s == 6:
                        Xn = acc
                    K[s] = self._f(t + _C[s] * h, acc)
                stats["f_evals"] += 6
                errv = h * (_E @ K)
                sc = np.maximum(atol_v, rtol * np.maximum(np.abs(X), np.abs(Xn)))
                err = float(np.max(np.abs(errv) / sc))
                if not np.all(np.isfinite(Xn)) or not math.isfinite(err):
                    err = math.inf
                if err <= 1.0:
                    break
                stats["rejected"] += 1
                if h <= hmin:
                    raise ControlError(f"t = {t:.9g} s anında adım en küçük değere indi ama tolerans "
                                       "sağlanamadı (süreksizlik ya da sert sistem); toleransı gevşetin ya da "
                                       "sabit adımlı çözücü deneyin.")
                if not failed:
                    failed = True
                    fac = 0.1 if not math.isfinite(err) else max(0.1, 0.8 * err ** -0.2)
                    h = max(hmin, h * fac)
                else:
                    h = max(hmin, 0.5 * h)
            t_new = t + h
            if abs(t_new - target) <= tol_t(target):
                t_new = target
            # sıfır geçiş
            located = False
            if detect_zc and zc0.size:
                zc1 = self._zc_values(t_new, Xn)
                # işaret değişimi ya da tam sıfırdan ayrılma (adım kırılma anına tam düştüyse)
                if zc1.size == zc0.size and (np.any(zc0 * zc1 < 0) or np.any((zc0 == 0) & (zc1 != 0))):
                    t_new, Xn = self._locate(t, X, h, K, zc0, zc1, t_new, Xn)
                    located = True
                    stats["zc_events"] += 1
            # adım önerisi (MATLAB ode45)
            if not failed:
                temp = 1.25 * max(err, 1e-300) ** 0.2
                h = h / temp if temp > 0.2 else 5.0 * h
            if located and (t_new - t) <= 1e3 * tol_t(t_new):
                consecutive_zc += 1
                if consecutive_zc > 100:
                    # kayma kipi (sliding mode): sıfır geçiş araması geçici olarak askıya alınır; bu aralıkta
                    # bloklar doğrudan değerlendirilir ve doğruluğu adım denetimi sağlar (Simulink 'adaptive'
                    # sıfır geçiş algoritmasının yaklaşımı)
                    zc_resume = t_new + (tf - t0) / 100.0
                    consecutive_zc = 0
                    stats["zc_suppressed"] = stats.get("zc_suppressed", 0) + 1
                    if not warned:
                        warned = True
                        warnings.warn(f"t ≈ {t_new:.6g} s civarında art arda sıfır geçişi (kayma kipi / "
                                      "'chattering'); sıfır geçiş araması kısa süre askıya alındı.", ControlWarning,
                                      stacklevel=3)
            else:
                consecutive_zc = 0
            self.X = Xn
            t = t_new
            if auto_atol:
                maxabs = np.maximum(maxabs, np.abs(Xn))
                atol_v = np.maximum(1e-6, rtol * maxabs)
            stats["steps"] += 1
            if stats["steps"] % 64 == 0 and self._tick(t, t0, tf):
                cancelled = True
                break
        self._t_end = self._log_t[-1] if self._log_t else t0
        stats["step"] = None
        return stats, cancelled

    def _locate(self, t, X, h, K, zc0, zc1, t_new, Xn):
        """[t, t_new] içinde ilk kip sınırı geçişini bulur; (olay anı, durum) döndürür (geçişin hemen sağı)."""
        ta, tb = t, t_new
        ga, gb = zc0, zc1

        def X_at(tt: float) -> np.ndarray:
            s = (tt - t) / h
            return X + h * (K.T @ (_BI @ np.array([s, s * s, s ** 3, s ** 4])))

        Xb = Xn
        for it in range(200):
            tol = 4 * np.finfo(float).eps * max(abs(ta), abs(tb), 1e-300) + 1e-15 * (t_new - t)
            if tb - ta <= tol:
                break
            cross = (ga * gb < 0) | ((ga == 0) & (gb != 0))
            if it % 3 == 2 or not np.any(cross):
                tm = 0.5 * (ta + tb)
            else:
                ga_c, gb_c = ga[cross], gb[cross]
                cand = ta + (tb - ta) * ga_c / (ga_c - gb_c)
                tm = float(np.min(cand))
                lo, hi = ta + 0.02 * (tb - ta), tb - 0.02 * (tb - ta)
                tm = min(max(tm, lo), hi)
            Xm = X_at(tm)
            self._minor_outputs(tm, Xm)
            gm = self._zc_values(tm, Xm)
            if np.any(ga * gm < 0) or np.any((gm == 0) & (ga != 0)) or np.any((ga == 0) & (gm != 0)):
                tb, gb, Xb = tm, gm, Xm
            else:
                ta, ga = tm, gm
        return tb, Xb


def _blowup(t: float) -> str:
    return (f"Benzetim t = {t:.6g} s anında sayısal olarak ıraksadı (NaN / sonsuz durum): sistem kararsız ya da "
            "adım çok büyük.")


def _cancel_checker(cancel: Any) -> Callable[[], bool] | None:
    if cancel is None:
        return None
    if hasattr(cancel, "is_set"):
        return cancel.is_set
    if callable(cancel):
        return cancel
    raise ControlError("İptal nesnesi çağrılabilir ya da is_set() yöntemli olmalı.")


def simulate(model: Model, stop_time: float | None = None, *, inputs: dict | None = None,
             initial: OperatingPoint | dict | None = None, **options: Any) -> SimResult:
    """Modeli benzetir. Seçenekler (verilmeyenler `model.settings`'ten):
    solver ('ode1' | 'ode4' | 'ode45'), step (sabit adım), start_time, rtol, atol ('auto' ya da sayı), max_step,
    initial_step, zero_crossing, output_times (bu anlarda kayıt; ode45 tam basar), decimation, log_all, log
    (ek kaydedilecek sinyaller), progress(oran, t), cancel (is_set() ya da çağrılabilir), max_steps."""
    st = model.settings
    if stop_time is None:
        stop_time = float(st.get("stop_time", 10.0))
    opts = {"solver": st.get("solver", "ode4"), "step": st.get("step"), "start_time": st.get("start_time", 0.0),
            "rtol": st.get("rtol", 1e-3), "atol": st.get("atol", "auto"), "max_step": st.get("max_step"),
            "zero_crossing": st.get("zero_crossing", True)}
    opts.update(options)
    sim = Simulator(model, inputs=inputs, initial=initial)
    return sim.run(stop_time, **opts)


__all__ = ["compile_model", "CompiledModel", "Simulator", "simulate", "SimResult", "ScopeData", "OperatingPoint",
           "SOLVERS"]
