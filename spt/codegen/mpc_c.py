"""Doğrusal MPC için MCU C kodu üreteci (çevrimiçi adım + Goldfarb–Idnani QP), ayak izi ve hesap maliyeti.

Üretilen C, `spt.control.mpc` (MATLAB R2025b MPC Toolbox ile ~1e-13 doğrulanmış) çevrimiçi hesabının — MATLAB'in
`mpcmoveCodeGeneration` ile ürettiği kodun — karşılığıdır. Tüm matrisler `MPCData` (MATLAB `getCodeGenerationData`)
adlarıyla ve **boyutsuz** (ölçekli) iç biçimde `static const` olarak flash'a gömülür; boyutlar `#define`, sayı türü
`<ÖNEK>_REAL` (float / double), durum çağıranın yapısında (malloc yok, global yok, yeniden girişli).

Her örnekte (Ts) adım (TASARIM.md §3 ile birebir):
  1. y = ym / S_y − y_off; r, v aynı biçimde (önizleme: satır dizisi, eksik satırlar sonuncuyla doldurulur).
  2. e = y − Cm x̂ − Dvm v(k); x̂(k|k) = x̂ + M e.
  3. f = Kxᵀ x̂ + Krᵀ R + Ku1ᵀ u₋₁ + Kvᵀ V + Kutᵀ U_hedef;  b = −(Mlim + Mx x̂ + Mu1 u₋₁ + Mv V).
     V'nin her bloğunun son elemanı (DX ofseti) her zaman 1, U_hedef sabittir: bu terimler üretim anında `f0` ve
     `Mlim0` vektörlerine önceden toplanır; önizleme kapalıysa Kr / Kv / Mv ufuk boyunca toplanmış saklanır
     (R, V ufuk boyunca sabit olduğundan matematiksel olarak aynı; yalnız toplama sırası → yuvarlama düzeyi).
  4. QP (qp.py ile aynı algoritma, aynı seçim kuralları, aynı yineleme sayımı ve durum kodları): J = Linvᵀ,
     önceki aktif kümeden sıcak başlangıç, en çok MaxIterations yineleme. Durum < 0 ya da 0 (alt-en iyi kapalı) →
     Δu = 0: MV önceki değerinde tutulur (MATLAB gibi). UseSuboptimalSolution açıksa yineleme sınırında alt-en iyi
     hamle MV / oran sınırlarına kırpılır (mpc_clipping).
  5. u = u₋₁ + Δu + u_off (mühendislik birimi: × S_u); x̂(k+1|k) = A x̂ + Bu (u − u_off) + Bv v(k) + L e.

Python / MATLAB'den bilinçli farklar (hepsi belgelenir, testlerle ölçülür):
  - Ölçek çarpanına bölme yerine önceden hesaplanmış tersle çarpma; sabit terimlerin önceden toplanması; Givens
    yarıçapı √(a² + b²) (taşma / alttan taşmada ölçekli yedek) — yuvarlama düzeyinde. Double'da Python'la fark
    ~1e-13, yineleme sayıları ve aktif kümeler aynıdır.
  - Sonlu olmayan (NaN / ±∞) ölçüm / referans / MD ya da geçersiz satır sayısı: adım yapılmaz, MV tutulur, durum
    değişmez, `<ÖNEK>_STATUS_BAD_INPUT` (−3) döner (Python hata fırlatır / NaN yayar).
  - Tek duyarlıkta QP'nin doğrusal bağımlılık eşiği DEP_TOL = 1e3·FLT_EPSILON (double: qp.py ile aynı 1e6·DBL_EPSILON).
  - Optimizer.MaxIterations = 0 desteklenmez (MATLAB'de de pozitif tamsayı olmalıdır).

Ayak izi (`footprint`): yerel derleyici varsa ÖLÇÜLÜR (nesne bölümleri, sizeof, libm dahil bağlanmış imaj farkı,
disassembly'den en kötü yığın); yoksa / IAR / Xtensa / RISC-V için ölçülmüş değerden TAHMİN (satırda yazar).
Maliyet (`cost`): adım başına işlem sayıları ana makinede C++ sayaç türüyle ÖLÇÜLÜR (kapalı çevrim senaryosu);
en kötü durum (MaxIterations yinelemesi) kodun yapısından türetilmiş ANALİTİK ÜST SINIRDIR (testte her ölçülen
adımın bu sınırın altında kaldığı doğrulanır); çevrimler MCU profil tablosundan TAHMİN edilir.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
import warnings
from dataclasses import dataclass, field

import numpy as np

from spt.codegen import toolchain as tc
from spt.control.errors import ControlWarning
from spt.codegen.fusion_c import FootprintRow, GeneratedFile, _count_prelude, write_files
from spt.codegen.mcu import (BASIC_OPS, COMPILERS, IAR_LIBM_FACTOR, IAR_SIZE_FACTOR, LIBRARY_OPS, MCU_PROFILES,
                             OPS, OPT_LEVELS, McuProfile, cycle_key, profile)

__all__ = ["MpcCodegenOptions", "MpcCostRow", "HostRun", "GeneratedFile", "FootprintRow", "check_options", "plan",
           "generate", "write_files", "state_size", "host_program", "run_steps", "run_closed_loop", "footprint",
           "cost", "step_ops_bound", "usage_markdown", "file_names", "default_prefix", "numeric_warnings",
           "budget_advice", "ram_bytes", "RAM_PERCENT_NOTE"]

COST_MIN_STEPS = 60


# ============================================================================ seçenekler ve sonuç türleri
@dataclass(frozen=True)
class MpcCodegenOptions:
    real: str = "float"                  # "float" | "double"
    mcu: str = "cortex-m4f"
    compiler: str = "gcc"
    opt: str = "-Os"
    clock_mhz: float = 168.0
    flash_kb: float = 1024.0
    ram_kb: float = 192.0
    prefix: str = "mpc"                  # dosya adları <önek>.h / <önek>.c, tanımlayıcılar <önek>_step …
    ref_preview: bool = False            # referans önizlemesi (p satıra kadar r dizisi)
    md_preview: bool = False             # ölçülen bozucu önizlemesi (p + 1 satıra kadar v dizisi)


@dataclass(frozen=True)
class MpcCostRow:
    scenario: str
    ops_mean: dict[str, float]           # adım başına ortalama işlem sayıları (ölçülen)
    cycles_mean: float                   # ölçülen senaryoda ortalama çevrim (tahmin: işlem × tablo)
    cycles_max: float                    # ölçülen senaryodaki en pahalı adım
    cycles_wc: float                     # analitik en kötü durum (MaxIterations yineleme) üst sınırı
    cycles_iter: float                   # bir QP yinelemesinin en kötü durum üst sınırı
    us_mean: float
    us_max: float
    us_wc: float
    load_mean: float                     # CPU yükü (%) @ fs = 1/Ts
    load_max: float
    load_wc: float
    iters_mean: float
    iters_max: int
    max_iter: int
    iter_budget: int                     # Ts bütçesine sığan en çok yineleme (CPU %100)
    status_counts: dict[str, int]
    method: str


@dataclass(frozen=True)
class HostRun:
    u: np.ndarray                        # (T, nmv) mühendislik birimi
    status: np.ndarray                   # (T,) durum kodu (> 0 yineleme)
    slack: np.ndarray                    # (T,)
    x: np.ndarray                        # (T, nx) adım sonrası x̂(k+1|k) (bitki kısmı Nominal.X dahil)
    y: np.ndarray                        # (T, ny) kapalı çevrimde bitki çıkışı (açık çevrimde 0)
    ops: np.ndarray | None = None        # (T, len(OPS)) adım başına işlem sayıları


# ============================================================================ seçenek denetimi
_PREFIX_RE = re.compile(r"[A-Za-z](?:_?[A-Za-z0-9])*")


def default_prefix(name: str | None) -> str:
    """Model adından geçerli bir önek (harfle başlar, yalnız harf / rakam / tek alt çizgi; Türkçe harfler ASCII'ye)."""
    tr = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(name or "").translate(tr)).strip("_").lower()
    s = re.sub(r"_+", "_", s)
    if not s or not s[0].isalpha():
        s = "mpc" + ("_" + s if s else "")
    s = s[:40].rstrip("_") or "mpc"
    return s + "_mpc" if _std_clash(s) else s


#: <stdint.h> tür / makro aileleri: önek "int16" → int16_t türü ve INT16_C makrosu, "intmax" → intmax_t / INTMAX_C.
_STDINT_RE = re.compile(r"u?int(?:8|16|32|64|max|ptr|_least(?:8|16|32|64)|_fast(?:8|16|32|64))", re.I)
#: <önek>_t türü standart bir typedef'le çakışan önekler (stddef / stdint / math / stdlib / time / stdio / signal …).
_STD_TYPEDEF_PREFIXES = frozenset({"size", "ssize", "ptrdiff", "wchar", "wint", "max_align", "float", "double",
                                   "div", "ldiv", "lldiv", "fpos", "clock", "time", "mbstate", "fenv", "fexcept",
                                   "sig_atomic", "errno", "char16", "char32"})
#: <önek>.h dosya adı standart başlıkla aynıysa (`-I.` ile) sistem başlığının yerine geçer.
_STD_HEADERS = frozenset({"assert", "complex", "ctype", "errno", "fenv", "float", "inttypes", "iso646", "limits",
                          "locale", "math", "setjmp", "signal", "stdalign", "stdarg", "stdatomic", "stdbool",
                          "stddef", "stdint", "stdio", "stdlib", "stdnoreturn", "string", "tgmath", "threads",
                          "time", "uchar", "wchar", "wctype"})


def _std_clash(prefix: str) -> str:
    """Önek C standart kitaplığının bir adıyla çakışıyorsa nedeni (yoksa '')."""
    p = (prefix or "").lower()
    if _STDINT_RE.fullmatch(p):
        return f"<stdint.h>: {p}_t türü ve {p.upper()}_C / {p.upper()}_MAX makroları"
    if p in _STD_TYPEDEF_PREFIXES:
        return f"standart tür {p}_t"
    if p in _STD_HEADERS:
        return f"standart başlık <{p}.h>"
    return ""


def check_options(opts: MpcCodegenOptions) -> None:
    if opts.real not in ("float", "double"):
        raise ValueError("Sayı türü 'float' ya da 'double' olmalı")
    profile(opts.mcu)
    if opts.compiler not in COMPILERS:
        raise ValueError(f"Bilinmeyen derleyici: {opts.compiler!r}")
    if opts.opt not in OPT_LEVELS:
        raise ValueError(f"İyileştirme düzeyi {OPT_LEVELS} içinden olmalı")
    if not _PREFIX_RE.fullmatch(opts.prefix or ""):
        raise ValueError("Önek harfle başlamalı; harf, rakam ve tek alt çizgi içerebilir (başta / sonda ya da çift alt "
                         "çizgi C'de saklı adlar üretir: _x, a__b)")
    clash = _std_clash(opts.prefix)
    if clash:
        raise ValueError(f"Önek '{opts.prefix}' C standart kitaplığıyla çakışır ({clash}); başka bir önek seçin.")
    if opts.clock_mhz <= 0 or opts.flash_kb <= 0 or opts.ram_kb <= 0:
        raise ValueError("Saat, flash ve RAM pozitif olmalı")


def file_names(prefix: str) -> tuple[str, str, str, str]:
    p = prefix.lower()
    return f"{p}.h", f"{p}.c", f"{p}_ornek.c", f"KULLANIM_{p}.md"


# ============================================================================ veri planı
@dataclass
class _Array:
    name: str
    values: np.ndarray                   # düz (satır öncelikli)
    size: str                            # C boyut ifadesi
    doc: str


@dataclass
class _Plan:
    """Üretim için gereken her şey (MPCData'dan türetilmiş, boyutsuz)."""
    name: str
    Ts: float
    p: int
    moves: tuple[int, ...]
    nx: int
    nxp: int
    nxQP: int
    nmv: int
    nmd: int
    nmy: int
    ny: int
    nfree: int
    nz: int
    nc: int
    isQP: bool
    max_iter: int
    tol: float
    warm: bool
    subopt: bool
    ref_preview: bool
    md_preview: bool
    has_bv_off: bool
    arrays: list[_Array] = field(default_factory=list)
    mv_names: tuple[str, ...] = ()
    mo_names: tuple[str, ...] = ()
    ov_names: tuple[str, ...] = ()
    md_names: tuple[str, ...] = ()
    mv_units: tuple[str, ...] = ()
    ov_units: tuple[str, ...] = ()
    md_units: tuple[str, ...] = ()
    myindex: tuple[int, ...] = ()
    uyindex: tuple[int, ...] = ()
    bounds: dict = field(default_factory=dict)    # KULLANIM için kısıt özeti
    scales: dict = field(default_factory=dict)

    def array(self, name: str) -> np.ndarray:
        for a in self.arrays:
            if a.name == name:
                return a.values
        raise KeyError(name)


def _mpc_data(mpcobj):
    try:
        return mpcobj.data
    except Exception as exc:  # noqa: BLE001 — ControlError dahil: arayüze anlaşılır ileti
        raise ValueError(f"MPC iç verisi hesaplanamadı: {exc}") from None


def plan(mpcobj, opts: MpcCodegenOptions | None = None, name: str | None = None) -> _Plan:
    """MPC nesnesinden üretilecek sabitler ve boyutlar (önizleme seçeneklerine göre toplanmış / tam)."""
    opts = opts or MpcCodegenOptions()
    d = _mpc_data(mpcobj)
    if d.max_iter < 1 and d.isQP:
        raise ValueError("Optimizer.MaxIterations en az 1 olmalı (0 yinelemeli kırpma kipi C'de desteklenmiyor).")
    if d.nmy == 0 or d.nmv == 0:
        raise ValueError("MPC'nin en az bir MV'si ve bir ölçülen çıkışı olmalı.")
    p, nmv, nmd, nv, ny, nmy = d.p, d.nmv, d.nmd, d.nv, d.ny, d.nmy
    nxQP, nx, nfree = d.nxQP, d.nx, d.nfree
    nc = int(d.Ac.shape[0]) if d.isQP else 0
    nz = d.degrees if d.isQP else nfree
    if nc > 30000 or nz > 30000:
        raise ValueError("Kısıt / değişken sayısı C'deki int sayaçları için çok büyük.")
    arrays: list[_Array] = []

    def add(nm, vals, size, doc):
        arrays.append(_Array(nm, np.asarray(vals, dtype=float).ravel(), size, doc))

    mv, md, mo = d.mvindex, d.mdindex, d.myindex
    # ---- ölçek ve nominal (ofset) sabitleri
    add("ym_iscale", 1.0 / d.Yscale[mo], "SPT_NYM", "1 / ölçek çarpanı (ölçülen çıkışlar, OV.ScaleFactor)")
    add("ym_off", d.yoff[mo], "SPT_NYM", "Nominal.Y / ölçek (ölçülen çıkışlar, boyutsuz)")
    add("r_iscale", 1.0 / d.Yscale, "SPT_NY", "1 / ölçek çarpanı (tüm çıkışlar; referans)")
    add("r_off", d.yoff, "SPT_NY", "Nominal.Y / ölçek (tüm çıkışlar, boyutsuz)")
    if nmd:
        add("md_iscale", 1.0 / d.Uscale[md], "SPT_NMD", "1 / ölçek çarpanı (ölçülen bozucular, DV.ScaleFactor)")
        add("md_off", d.uoff[md], "SPT_NMD", "Nominal.U / ölçek (ölçülen bozucular, boyutsuz)")
    add("mv_scale", d.Uscale[mv], "SPT_NMV", "MV ölçek çarpanı (MV.ScaleFactor)")
    add("mv_iscale", 1.0 / d.Uscale[mv], "SPT_NMV", "1 / MV ölçek çarpanı")
    add("mv_off", d.mvoff, "SPT_NMV", "Nominal.U / ölçek (MV'ler, boyutsuz)")
    add("xp_off", d.xpoff, "SPT_NXP", "Nominal.X (bitki durumları; ölçeklenmez)")
    # ---- kestirici (MPCData: A, Bu, Bv, Cm, Dvm, L, M)
    add("A", d.A, "SPT_NX * SPT_NX", "A — artırılmış kestirici modeli (bitki + bozucu + gürültü)")
    add("Bu", d.Bu, "SPT_NX * SPT_NMV", "Bu — MV giriş matrisi")
    if nmd:
        add("Bv", d.Bv[:, :nmd], "SPT_NX * SPT_NMD", "Bv — ölçülen bozucu sütunları (ofset sütunu ayrı)")
    bv_off = d.Bv[:, nmd]
    has_bv_off = bool(np.any(bv_off != 0))
    if has_bv_off:
        add("Bv_off", bv_off, "SPT_NX", "Bv'nin ofset (Nominal.DX) sütunu — girişi her zaman 1")
    add("Cm", d.Cm, "SPT_NYM * SPT_NX", "Cm — ölçülen çıkış matrisi")
    if nmd:
        add("Dvm", d.Dvm[:, :nmd], "SPT_NYM * SPT_NMD", "Dvm — ölçülen bozucudan ölçülen çıkışa doğrudan geçiş")
    if np.any(d.Dvm[:, nmd] != 0):          # MPC yapısı gereği sıfırdır (D'nin ofset sütunu 0)
        raise ValueError("Beklenmeyen veri: Dvm'nin ofset sütunu sıfır değil.")
    add("L", d.L, "SPT_NX * SPT_NYM", "L — Kalman zaman güncellemesi kazancı")
    add("M", d.M, "SPT_NX * SPT_NYM", "M — Kalman ölçüm güncellemesi kazancı")
    # ---- QP doğrusal terimi f (nfree)
    add("KxT", d.Kx.T, "SPT_NFREE * SPT_NXQP", "Kxᵀ")
    add("Ku1T", d.Ku1.T, "SPT_NFREE * SPT_NMV", "Ku1ᵀ")
    Kr = d.Kr.reshape(p, ny, nfree)
    if opts.ref_preview:
        add("KrT", Kr.reshape(p * ny, nfree).T, "SPT_NFREE * SPT_P * SPT_NY", "Krᵀ (önizleme: tüm ufuk)")
    else:
        add("KrT", Kr.sum(axis=0).T, "SPT_NFREE * SPT_NY", "(Σᵢ Krᵢ)ᵀ — referans ufuk boyunca sabit")
    Kv = d.Kv.reshape(p + 1, nv, nfree)
    if nmd:
        if opts.md_preview:
            add("KvT", Kv[:, :nmd, :].reshape((p + 1) * nmd, nfree).T, "SPT_NFREE * (SPT_P + 1) * SPT_NMD",
                "Kvᵀ, ölçülen bozucu satırları (önizleme: tüm ufuk)")
        else:
            add("KvT", Kv[:, :nmd, :].sum(axis=0).T, "SPT_NFREE * SPT_NMD",
                "(Σᵢ Kvᵢ)ᵀ, ölçülen bozucu satırları — MD ufuk boyunca sabit")
    f0 = Kv[:, nmd, :].sum(axis=0) + d.Kut.T @ d.utarget
    add("f0", f0, "SPT_NFREE", "Kvᵀ·[ofset = 1] + Kutᵀ·U_hedef (sabit kısım)")
    bounds = {}
    if d.isQP:
        Lv = d.Linv
        add("Linv", np.concatenate([Lv[i, :i + 1] for i in range(nz)]), "SPT_NZ * (SPT_NZ + 1) / 2",
            "Linv = L⁻¹ (H = L Lᵀ), alt üçgen satır satır paketli")
        add("Ac", d.Ac, "SPT_NC * SPT_NZ", "Ac — kısıtlar Ac z ≥ b")
        Mv = d.Mv.reshape(nc, p + 1, nv)
        add("Mlim0", d.Mlim + Mv[:, :, nmd].sum(axis=1), "SPT_NC", "Mlim + Mv·[ofset = 1] (sabit kısım)")
        add("Mx", d.Mx, "SPT_NC * SPT_NXQP", "Mx")
        add("Mu1", d.Mu1, "SPT_NC * SPT_NMV", "Mu1")
        if nmd:
            if opts.md_preview:
                add("Mv", Mv[:, :, :nmd].reshape(nc, (p + 1) * nmd), "SPT_NC * (SPT_P + 1) * SPT_NMD",
                    "Mv, ölçülen bozucu sütunları (önizleme: tüm ufuk)")
            else:
                add("Mv", Mv[:, :, :nmd].sum(axis=1), "SPT_NC * SPT_NMD",
                    "Σᵢ Mvᵢ, ölçülen bozucu sütunları — MD ufuk boyunca sabit")
        if d.use_suboptimal:
            cl = _clip_bounds(d)
            for k, doc in (("umin", "u alt sınırı"), ("umax", "u üst sınırı"), ("dumin", "Δu alt sınırı"),
                           ("dumax", "Δu üst sınırı")):
                add(f"clip_{k}", cl[k], "SPT_NMV", f"İlk hamle {doc} (boyutsuz; ±∞ → ±REAL_MAX) — kırpma")
        bounds = _bounds_summary(mpcobj, d)
    else:
        add("Kmpc", -d.Hinv[:nmv, :nfree], "SPT_NMV * SPT_NFREE", "−H⁻¹'in ilk n_mv satırı (kısıtsız çözüm)")
    names_mv = tuple(str(mpcobj.MV[i].Name) for i in range(nmv))
    names_ov = tuple(str(mpcobj.OV[i].Name) for i in range(ny))
    names_dv = tuple(str(mpcobj.DV[i].Name) for i in range(nmd))
    units_mv = tuple(str(mpcobj.MV[i].get("Units") or "") for i in range(nmv))
    units_ov = tuple(str(mpcobj.OV[i].get("Units") or "") for i in range(ny))
    units_dv = tuple(str(mpcobj.DV[i].get("Units") or "") for i in range(nmd))
    scales = {"mv": d.Uscale[mv].tolist(), "ov": d.Yscale.tolist(), "md": d.Uscale[md].tolist(),
              "mv_nom": (d.mvoff * d.Uscale[mv]).tolist(), "ov_nom": (d.yoff * d.Yscale).tolist(),
              "md_nom": (d.mdoff * d.Uscale[md]).tolist(), "x_nom": d.xpoff.tolist()}
    for a in arrays:
        if not a.name.startswith("clip_") and not np.all(np.isfinite(a.values)):   # kırpma: ±∞ → ±REAL_MAX
            raise ValueError(f"MPC verisinde sonlu olmayan değer ({a.name}).")
    if opts.real == "float":
        huge = [a.name for a in arrays if a.name not in _BOUND_ARRAYS
                and np.any(np.isfinite(a.values) & (np.abs(a.values) > FLT_MAX))]
        if huge:
            raise ValueError(f"{', '.join(huge)} sabitleri float aralığını aşıyor (|x| > {FLT_MAX:.3g}): sayı türü "
                             "double seçin ya da ölçek çarpanlarını (ScaleFactor) sinyal aralığına göre düzeltin.")
    return _Plan(name=str(name or "mpc"), Ts=float(d.Ts), p=p, moves=tuple(int(m) for m in d.moves), nx=nx,
                 nxp=d.nxp, nxQP=nxQP, nmv=nmv, nmd=nmd, nmy=nmy, ny=ny, nfree=nfree, nz=nz, nc=nc, isQP=d.isQP,
                 max_iter=int(d.max_iter), tol=float(d.tol), warm=bool(d.use_warm_start),
                 subopt=bool(d.use_suboptimal) and d.isQP, ref_preview=bool(opts.ref_preview),
                 md_preview=bool(opts.md_preview) and nmd > 0, has_bv_off=has_bv_off, arrays=arrays,
                 mv_names=names_mv, mo_names=tuple(names_ov[i] for i in mo), ov_names=names_ov, md_names=names_dv,
                 mv_units=units_mv, ov_units=units_ov, md_units=units_dv, myindex=tuple(int(i) for i in mo),
                 uyindex=tuple(int(i) for i in d.uyindex), bounds=bounds, scales=scales)


def _clip_bounds(d) -> dict[str, np.ndarray]:
    """MATLAB mpc_clipping / mpc.py _clip: ilk adımın u ve Δu sınırları (boyutsuz; u sınırı mutlak = Mlim + u_off)."""
    nmv, p, ny, nfree = d.nmv, d.p, d.ny, d.nfree
    nylim, pnu = 2 * p * ny, p * nmv
    out = {k: np.full(nmv, v) for k, v in (("umin", -np.inf), ("umax", np.inf), ("dumin", -np.inf),
                                            ("dumax", np.inf))}
    for i, ix in enumerate(d.Mrows):
        if nylim <= ix < nylim + nmv:
            out["umax"][ix - nylim] = d.Mlim[i] + d.mvoff[ix - nylim]
        elif nylim + pnu <= ix < nylim + pnu + nmv:
            out["umin"][ix - nylim - pnu] = -d.Mlim[i] + d.mvoff[ix - nylim - pnu]
        elif nylim + 2 * pnu <= ix < nylim + 2 * pnu + nmv:
            out["dumax"][ix - nylim - 2 * pnu] = d.Mlim[i]
        elif nylim + 2 * pnu + nfree <= ix < nylim + 2 * pnu + nfree + nmv:
            out["dumin"][ix - nylim - 2 * pnu - nfree] = -d.Mlim[i]
    return out


def _bounds_summary(mpcobj, d) -> dict:
    """KULLANIM için kısıt özeti (mühendislik birimi, MPC nesnesindeki değerler)."""
    def fmt(v):
        a = np.atleast_1d(np.asarray(v, dtype=float)).ravel()
        if a.size == 0 or np.all(~np.isfinite(a)):
            return "—"
        return ", ".join(f"{x:g}" for x in a[:4]) + (" …" if a.size > 4 else "")
    rows = []
    for i in range(d.nmv):
        s = mpcobj.MV[i]
        rows.append(("MV", str(s.Name), fmt(s.Min), fmt(s.Max), fmt(s.RateMin), fmt(s.RateMax),
                     "sert" if np.all(np.asarray(s.MinECR, float) == 0) and np.all(np.asarray(s.MaxECR, float) == 0)
                     else "yumuşak"))
    for i in range(d.ny):
        s = mpcobj.OV[i]
        if np.any(np.isfinite(np.asarray(s.Min, float))) or np.any(np.isfinite(np.asarray(s.Max, float))):
            rows.append(("OV", str(s.Name), fmt(s.Min), fmt(s.Max), "—", "—",
                         "sert" if np.all(np.asarray(s.MinECR, float) == 0) and
                         np.all(np.asarray(s.MaxECR, float) == 0) else "yumuşak"))
    return {"rows": rows, "ecr": float(d.Wecr)}


# ============================================================================ C metni
_BANNER = ("Signal Processing Toolbox tarafından üretildi. Python referansı: spt/control/mpc "
           "(MATLAB R2025b ile doğrulandı).")


FLT_MIN = 1.1754943508222875e-38       # en küçük normal float
FLT_MAX = 3.4028234663852886e+38
FLT_EPSILON = 1.1920928955078125e-07
DBL_EPSILON = float(np.finfo(float).eps)
#: Sınır vektörleri: |x| > FLT_MAX değerleri float'ta ±FLT_MAX'a kırpılabilir (pratikte "sınır yok").
_BOUND_ARRAYS = frozenset({"Mlim0", "clip_umin", "clip_umax", "clip_dumin", "clip_dumax"})


def _lit(v: float, bound: bool = False) -> str:
    """C sabiti. Aynı .c hem float hem double derlenir: |x| < FLT_MIN → SPT_CT (float'ta 0), sınırlarda
    |x| > FLT_MAX → SPT_CH (float'ta FLT_MAX); matrislerde |x| > FLT_MAX olduğu gibi yazılır (float derlemesi
    #error ile durdurulur, bkz. _source / plan)."""
    v = float(v)
    if math.isinf(v):
        return "SPT_REAL_MAX" if v > 0 else "(-SPT_REAL_MAX)"
    if v != 0.0 and abs(v) < 1e-300:
        v = 0.0
    a = abs(v)
    r = repr(a)
    if not any(c in r for c in ".eE"):
        r += ".0"
    sign = "-" if v < 0 else ""
    if 0.0 < a < FLT_MIN:
        return f"({sign}SPT_CT({r}))"
    if a > FLT_MAX and bound:
        return f"({sign}SPT_CH({r}))"
    return f"SPT_C({sign}{r})"


def _huge_matrix_arrays(pl) -> list[str]:
    """|x| > FLT_MAX (sonlu) eleman içeren sınır dışı (matris / vektör) sabitler — float'ta temsil edilemez."""
    return [a.name for a in pl.arrays if a.name not in _BOUND_ARRAYS
            and np.any(np.isfinite(a.values) & (np.abs(a.values) > FLT_MAX))]


def _c_array(a: _Array, per_line: int = 3) -> str:
    bound = a.name in _BOUND_ARRAYS
    items = [_lit(v, bound) for v in a.values]
    lines = []
    for i in range(0, len(items), per_line):
        lines.append("    " + ", ".join(items[i:i + per_line]) + ("," if i + per_line < len(items) else ""))
    return (f"/** {a.doc} */\nstatic const SPT_REAL spt_{a.name}[{a.size}] = {{\n" + "\n".join(lines) + "\n};")


def _fill(text: str, **kw: str) -> str:
    for k, v in kw.items():
        text = text.replace("{{" + k + "}}", v)
    left = re.findall(r"\{\{[A-Z_0-9]+\}\}", text)
    assert not left, left
    return text


def _prefixed(text: str, prefix: str) -> str:
    p = prefix.lower()
    text = text.replace('"spt.h"', f'"{p}.h"').replace("spt.h", f"{p}.h").replace("spt.c", f"{p}.c")
    text = text.replace("spt_ornek.c", f"{p}_ornek.c").replace("KULLANIM_spt.md", f"KULLANIM_{p}.md")
    text = re.sub(r"\bspt_", p + "_", text)
    text = re.sub(r"\bSPT_", prefix.upper() + "_", text)
    lines = text.split("\n")                       # fonksiyon imzası devam satırlarını '(' sonrasına hizala
    for i in range(1, len(lines)):
        m = re.match(r"(?:static )?(?:int|void) \w+\(", lines[i - 1])
        if m and lines[i - 1].rstrip().endswith(",") and re.match(r" +(const |int |\w+_REAL )", lines[i]):
            lines[i] = " " * len(m.group(0)) + lines[i].strip()
    return "\n".join(lines)


def _sig_list(names, units, idx=None) -> str:
    out = []
    names = [str(n).replace("*/", "* /") for n in names]
    units = [str(u).replace("*/", "* /") for u in units] if units else units
    for i, n in enumerate(names):
        u = f" [{units[i]}]" if units and units[i] else ""
        k = f" (bitki çıkışı {idx[i] + 1})" if idx is not None else ""
        out.append(f"{i}: {n}{u}{k}")
    return "; ".join(out) if out else "—"


_HEADER = r"""/**
 * @file spt.h
 * @brief MPC denetleyicisi "{{NAME}}" — çevrimiçi adım (MCU: malloc yok, sabit boyutlu diziler, yeniden girişli).
 *
 * {{BANNER}}
 * Ayrıntılı kullanım: KULLANIM_spt.md
 *
 * Denetleyici: Ts = {{TS}} s, tahmin ufku p = {{P}}, hamle blokları {{MOVES}}; MV {{NMV}}, MD {{NMD}},
 * ölçülen çıkış {{NYM}}, ölçülmeyen çıkış {{NUO}}; QP: {{NZ}} değişken, {{NC}} kısıt.
 *   MV (u)   : {{MV_LIST}}
 *   MO (ym)  : {{MO_LIST}}
 *   OV (ref) : {{OV_LIST}}
 *   MD (md)  : {{MD_LIST}}
 *
 * Her Ts saniyede bir spt_step() çağrılır (MATLAB mpcmove / mpcmoveCodeGeneration ile aynı hesap):
 *   1. Boyutsuzlaştırma: y = ym / ScaleFactor − Nominal.Y (referans ve ölçülen bozucu da aynı biçimde).
 *   2. Kalman ölçüm güncellemesi: e = y − Cm x̂(k|k−1) − Dvm v(k);  x̂(k|k) = x̂(k|k−1) + M e.
 *   3. QP doğrusal terimi f = Kxᵀ x̂ + Krᵀ r + Ku1ᵀ u(k−1) + Kvᵀ v + Kutᵀ u_hedef ve kısıt sağ tarafı
 *      b = −(Mlim + Mx x̂ + Mu1 u(k−1) + Mv v)  (sabit kısımlar üretimde önceden toplandı: f0, Mlim0).
 *   4. QP: min ½ zᵀ H z + fᵀ z,  Ac z ≥ b — Goldfarb–Idnani çift aktif küme; H = L Lᵀ'nin Linv = L⁻¹ çarpanı
 *      önceden hesaplı; önceki adımın aktif kümesinden sıcak başlangıç; en çok SPT_MAX_ITER yineleme.
 *   5. u(k) = u(k−1) + Δu(k) (ilk hamle). QP çözümsüz / sayısal hata / yineleme sınırı → MV önceki değerinde
 *      tutulur (MATLAB gibi).
 *   6. Kestirici zaman güncellemesi: x̂(k+1|k) = A x̂(k|k−1) + Bu u(k) + Bv v(k) + L e.
 *
 * Uyarı: -ffast-math / -ffinite-math-only ile derlemeyin (NaN / sonsuz giriş denetimleri kaldırılır).
 */
#ifndef SPT_H
#define SPT_H

#include <stddef.h>
#include <stdint.h>

#ifndef SPT_USE_DOUBLE
#define SPT_USE_DOUBLE {{USE_DOUBLE}}       /* 1: double, 0: float (üretimde seçilen: {{REAL}}) */
#endif

#if defined(SPT_CUSTOM_REAL)
/* SPT_REAL, SPT_C, SPT_REAL_MAX, SPT_EPS, SPT_SQRT, SPT_FABS önceden tanımlı (ana makine sayacı). */
#elif SPT_USE_DOUBLE
#include <float.h>
#include <math.h>
/** Hesap türü: double (yalnız çift duyarlık FPU'lu çekirdekte — ör. Cortex-M7 — donanımda). */
typedef double SPT_REAL;
#define SPT_C(x) (x)
#define SPT_REAL_MAX DBL_MAX
#define SPT_EPS DBL_EPSILON
#ifndef SPT_SQRT
#define SPT_SQRT(x) sqrt(x)
#endif
#ifndef SPT_FABS
#define SPT_FABS(x) fabs(x)
#endif
#else
#include <float.h>
#include <math.h>
/** Hesap türü: float (tek duyarlık FPU'lu Cortex-M4F / M33F / M7'de donanımda). */
typedef float SPT_REAL;
#define SPT_C(x) (x##f)
#define SPT_REAL_MAX FLT_MAX
#define SPT_EPS FLT_EPSILON
#ifndef SPT_SQRT
#define SPT_SQRT(x) sqrtf(x)
#endif
#ifndef SPT_FABS
#define SPT_FABS(x) fabsf(x)
#endif
#endif

/* Aralık dışı sabitler (aynı .c hem float hem double derlenebilsin; -Werror temiz):
 *   SPT_CT(x): |x| < FLT_MIN olan sabit — double'da tam değer, float'ta 0 (alttan taşma / normal altı yok);
 *   SPT_CH(x): |x| > FLT_MAX olan SINIR sabiti (Mlim, kırpma sınırları) — double'da tam değer, float'ta FLT_MAX
 *              (float'ta pratikte sınırsız). Matrislerde |x| > FLT_MAX ise .c dosyası float'ta #error verir. */
#ifndef SPT_CT
#if defined(SPT_CUSTOM_REAL) || SPT_USE_DOUBLE
#define SPT_CT(x) SPT_C(x)
#define SPT_CH(x) SPT_C(x)
#else
#define SPT_CT(x) SPT_C(0.0)
#define SPT_CH(x) SPT_REAL_MAX
#endif
#endif

/* ---- boyutlar (üretim anında sabit) ----------------------------------------------------------------------------- */
#define SPT_NX    {{NX}}      /* kestirici durumu: bitki {{NXP}} + bozucu {{NXD}} + ölçüm gürültüsü {{NXN}} */
#define SPT_NXP   {{NXP}}      /* bitki durumu */
#define SPT_NXQP  {{NXQP}}      /* QP'nin gördüğü durum (bitki + bozucu) */
#define SPT_NMV   {{NMV}}      /* manipüle edilen değişken (MV, çıkış u) */
#define SPT_NMD   {{NMD}}      /* ölçülen bozucu (MD, giriş md) */
#define SPT_NYM   {{NYM}}      /* ölçülen çıkış (MO, giriş ym) */
#define SPT_NY    {{NY}}      /* tüm çıkışlar (OV; referans satırının boyu) */
#define SPT_P     {{P}}      /* tahmin ufku */
#define SPT_NFREE {{NFREE}}      /* serbest hamle (Δu blokları × MV) */
#define SPT_NZ    {{NZ}}      /* QP değişkeni (serbest hamle{{SLACK_NOTE}}) */
#define SPT_NC    {{NC}}      /* QP kısıt satırı (0: kısıtsız, kapalı biçim) */
#define SPT_TS    {{TS_LIT}}   /* örnekleme zamanı (s): spt_step() tam bu periyotla çağrılmalı */
#define SPT_REF_PREVIEW {{REF_PREVIEW}}   /* 1: ref en çok SPT_P satır (r(k+1) … r(k+p)), 0: tek satır */
#define SPT_MD_PREVIEW  {{MD_PREVIEW}}   /* 1: md en çok SPT_P + 1 satır (v(k) … v(k+p)), 0: tek satır */
{{QP_CONFIG}}
/* ---- durum kodları (MATLAB mpcActiveSetSolver / mpcmove info.Iterations anlamıyla) ------------------------------ */
#define SPT_STATUS_MAXITER    0      /* yineleme sınırı: MV tutuldu (QPCode 'unreliable') */
#define SPT_STATUS_INFEASIBLE (-1)   /* QP çözümsüz: MV tutuldu (QPCode 'infeasible') */
#define SPT_STATUS_NUMERICAL  (-2)   /* sayısal hata (NaN / sonsuz): MV tutuldu */
#define SPT_STATUS_BAD_INPUT  (-3)   /* giriş NaN / sonsuz ya da satır sayısı geçersiz: adım yapılmadı */

#if defined(__cplusplus) && !defined(SPT_CUSTOM_REAL)
extern "C" {
#endif

/** Denetleyici durumu (tamamı burada; global değişken yok). Statik ayırın: `static spt_t mpc;`. */
typedef struct {
    SPT_REAL x[SPT_NX];          /* x̂(k|k−1) − [Nominal.X; 0]: kestirici durumu (boyutsuz sapma) */
    SPT_REAL u1[SPT_NMV];        /* u(k−1)/ScaleFactor − u_off: son MV (boyutsuz sapma) */
    SPT_REAL slack;              /* son QP'nin gevşek değişkeni ε (yumuşak kısıt ihlali ölçüsü; kısıtsızda 0) */
{{QP_REAL_FIELDS}}    int status;                  /* son adımın durum kodu (> 0: QP yineleme sayısı) */
    int iter_max;                /* görülen en büyük yineleme sayısı (bütçe izleme) */
{{QP_INT_FIELDS}}    uint32_t steps;              /* toplam adım */
    uint32_t n_ok;               /* çözülen adım (durum > 0) */
    uint32_t n_maxiter;          /* yineleme sınırına takılan adım */
    uint32_t n_infeasible;       /* çözümsüz QP */
    uint32_t n_numerical;        /* sayısal hata */
    uint32_t n_bad_input;        /* geçersiz giriş */
{{QP_CHAR_FIELDS}}} spt_t;

/** MATLAB mpcstate varsayılanı: bitki durumu = Nominal.X, bozucu / gürültü durumları 0, son MV = nominal MV,
 *  aktif küme boş, sayaçlar sıfır. */
void spt_init(spt_t *s);

/**
 * Bir kontrol adımı (her Ts'de bir kez). Tüm büyüklükler mühendislik biriminde (ScaleFactor / Nominal dahil).
 * @param ym       ölçülen çıkışlar [SPT_NYM] (NULL: nominal değer)
 * @param ref      referans, satır öncelikli [ref_rows × SPT_NY]; satır i → y(k+1+i). NULL: Nominal.Y.
 *                 SPT_REF_PREVIEW 0 iken ref_rows 1 olmalı; 1 iken en çok SPT_P satır kullanılır, eksik satırlar
 *                 son satırla doldurulur (MATLAB mpcmove önizleme kuralı).
 * @param md       ölçülen bozucular [md_rows × SPT_NMD]; satır i → v(k+i) (NULL: nominal; SPT_NMD = 0 ise yok
 *                 sayılır). SPT_MD_PREVIEW 0 iken md_rows 1 olmalı; 1 iken en çok SPT_P + 1 satır.
 * @param u        çıkış: uygulanacak MV [SPT_NMV] (hata durumlarında önceki MV)
 * @return         > 0: QP yineleme sayısı (çözüldü), 0 / −1 / −2: MV tutuldu, −3: geçersiz giriş (durum değişmez)
 */
int spt_step(spt_t *s, const SPT_REAL *ym, const SPT_REAL *ref, int ref_rows,
             const SPT_REAL *md, int md_rows, SPT_REAL u[SPT_NMV]);

/** Son MV'yi ayarlar (el → otomatik geçişte darbesiz devralma: o anki eyleyici komutunu verin). */
void spt_set_last_mv(spt_t *s, const SPT_REAL u[SPT_NMV]);
/** Son uygulanan MV (mühendislik birimi). */
void spt_get_last_mv(const spt_t *s, SPT_REAL u[SPT_NMV]);
/** Kestirici durumu x̂(k|k−1) = [bitki (Nominal.X dahil); bozucu; gürültü] (MATLAB mpcstate Plant / Disturbance /
 *  Noise sırasıyla). */
void spt_get_state(const spt_t *s, SPT_REAL x[SPT_NX]);
void spt_set_state(spt_t *s, const SPT_REAL x[SPT_NX]);

#if defined(__cplusplus) && !defined(SPT_CUSTOM_REAL)
}
#endif

#endif /* SPT_H */
"""

_QP_CONFIG = r"""#ifndef SPT_MAX_ITER
#define SPT_MAX_ITER {{MAX_ITER}}    /* Optimizer.MaxIterations (en kötü durum süresini bu belirler; bkz. KULLANIM) */
#endif
#ifndef SPT_TOL
#define SPT_TOL {{TOL}}   /* Optimizer.ConstraintTolerance (boyutsuz kısıt ihlali) */
#endif
#ifndef SPT_DEP_TOL
#if defined(SPT_CUSTOM_REAL) || SPT_USE_DOUBLE
#define SPT_DEP_TOL SPT_C(2.220446049250313e-10)   /* 1e6·DBL_EPSILON: doğrusal bağımlılık eşiği (qp.py) */
#else
#define SPT_DEP_TOL SPT_C(1.1920929e-4)            /* 1e3·FLT_EPSILON: tek duyarlıkta bağımlılık eşiği */
#endif
#endif
#define SPT_WARM_START {{WARM}}   /* Optimizer.UseWarmStart */
#define SPT_SUBOPTIMAL {{SUBOPT}}   /* Optimizer.UseSuboptimalSolution (sınırda kırpılmış alt-en iyi hamle) */
#if SPT_MAX_ITER < 1
#error "SPT_MAX_ITER en az 1 olmalı"
#endif
"""

_QP_REAL_FIELDS = """    SPT_REAL qJ[SPT_NZ * SPT_NZ];  /* QP: J = L⁻ᵀQ (satır öncelikli) */
    SPT_REAL qR[SPT_NZ * SPT_NZ];  /* QP: R (üst üçgen) */
    SPT_REAL qx[SPT_NZ];           /* QP çözümü z = [Δu bloklar; ε] */
    SPT_REAL qd[SPT_NZ];           /* QP çalışma vektörleri */
    SPT_REAL qz[SPT_NZ];
    SPT_REAL qr[SPT_NZ];
    SPT_REAL qu[SPT_NZ];           /* aktif kısıt çarpanları */
    SPT_REAL qb[SPT_NC];           /* kısıt sağ tarafı b */
"""
_QP_INT_FIELDS = "    int qact[SPT_NZ];            /* aktif kısıt indisleri */\n"
_QP_CHAR_FIELDS = "    unsigned char iA[SPT_NC];    /* aktif küme (1 = aktif): sonraki adımın sıcak başlangıcı */\n"

_QP_SOURCE = r"""
/* ==================================================================================================================
 * QP çözücüsü — Goldfarb–Idnani çift aktif küme (spt/control/mpc/qp.py ile aynı algoritma ve durum kodları)
 *   min ½ zᵀ H z + fᵀ z,  Ac z ≥ b.  J = L⁻ᵀQ, L⁻¹N = Q[R; 0] Givens dönüşümleriyle güncellenir (JᵀHJ = I).
 *   Yineleme sayımı MATLAB'inkiyle aynı kural: 1 + ekleme / çıkarma adımları (sıcak başlangıç çıkarmaları dahil).
 * ================================================================================================================ */
#define SPT_QJ(i, j) (s->qJ[(i) * SPT_NZ + (j)])
#define SPT_QR(i, j) (s->qR[(i) * SPT_NZ + (j)])
#define SPT_QP_RUN (-100)

/** √(a² + b²); taşma / alttan taşmada ölçekli biçim (Givens yarıçapı). */
static SPT_REAL spt_hypot(SPT_REAL a, SPT_REAL b)
{
    SPT_REAL h = SPT_SQRT(a * a + b * b), t;
    if (h > SPT_C(0.0) && h <= SPT_REAL_MAX) {
        return h;
    }
    a = SPT_FABS(a);
    b = SPT_FABS(b);
    if (a < b) {
        t = a;
        a = b;
        b = t;
    }
    if (!(a > SPT_C(0.0)) || !(a <= SPT_REAL_MAX)) {
        return a;                                  /* 0, sonsuz ya da NaN (NaN sayısal hata olarak yakalanır) */
    }
    t = b / a;
    return a * SPT_SQRT(SPT_C(1.0) + t * t);
}

/** qd = Jᵀ a */
static void spt_qp_jt(spt_t *s, const SPT_REAL *a)
{
    int i, j;
    for (j = 0; j < SPT_NZ; ++j) {
        SPT_REAL acc = SPT_C(0.0);
        for (i = 0; i < SPT_NZ; ++i) {
            acc += SPT_QJ(i, j) * a[i];
        }
        s->qd[j] = acc;
    }
}

/** qr[0..q) = R⁻¹ qd[0..q) (üst üçgen geri yerine koyma) */
static void spt_qp_back(spt_t *s, int q)
{
    int i, k;
    for (i = q - 1; i >= 0; --i) {
        SPT_REAL acc = s->qd[i];
        for (k = i + 1; k < q; ++k) {
            acc -= SPT_QR(i, k) * s->qr[k];
        }
        s->qr[i] = acc / SPT_QR(i, i);
    }
}

/** qz = J[:, q..] qd[q..] (birincil adım yönü) */
static void spt_qp_dir(spt_t *s, int q)
{
    int i, j;
    for (i = 0; i < SPT_NZ; ++i) {
        SPT_REAL acc = SPT_C(0.0);
        for (j = q; j < SPT_NZ; ++j) {
            acc += SPT_QJ(i, j) * s->qd[j];
        }
        s->qz[i] = acc;
    }
}

/** ‖qd[from..]‖ */
static SPT_REAL spt_qp_norm(const spt_t *s, int from)
{
    SPT_REAL acc = SPT_C(0.0);
    int j;
    for (j = from; j < SPT_NZ; ++j) {
        acc += s->qd[j] * s->qd[j];
    }
    return SPT_SQRT(acc);
}

/** aᵀ qx − b */
static SPT_REAL spt_qp_slack(const spt_t *s, int i)
{
    const SPT_REAL *a = spt_Ac + i * SPT_NZ;
    SPT_REAL acc = SPT_C(0.0);
    int j;
    for (j = 0; j < SPT_NZ; ++j) {
        acc += a[j] * s->qx[j];
    }
    return acc - s->qb[i];
}

/** Kısıt ekleme: qd[q..]'yu Givens dönüşümleriyle e_q doğrultusuna döndür (J'nin q.. sütunlarına uygula);
 *  R'nin q. sütunu = qd[0..q]. */
static void spt_qp_add_column(spt_t *s, int q)
{
    int i, j;
    for (j = SPT_NZ - 1; j > q; --j) {
        SPT_REAL a = s->qd[j - 1], b = s->qd[j], h, c, sn;
        if (b == SPT_C(0.0)) {
            continue;
        }
        h = spt_hypot(a, b);
        c = a / h;
        sn = b / h;
        s->qd[j - 1] = h;
        s->qd[j] = SPT_C(0.0);
        for (i = 0; i < SPT_NZ; ++i) {
            SPT_REAL t1 = SPT_QJ(i, j - 1), t2 = SPT_QJ(i, j);
            SPT_QJ(i, j - 1) = c * t1 + sn * t2;
            SPT_QJ(i, j) = sn * t1 - c * t2;
        }
    }
    for (i = 0; i <= q; ++i) {
        SPT_QR(i, q) = s->qd[i];
    }
}

/** Aktif kümeden l. elemanı çıkar (q: önceki boyut); yeni boyutu döndürür. */
static int spt_qp_drop(spt_t *s, int l, int q)
{
    int i, j, k;
    for (i = l; i < q - 1; ++i) {
        for (k = 0; k < i + 2; ++k) {
            SPT_QR(k, i) = SPT_QR(k, i + 1);
        }
        s->qu[i] = s->qu[i + 1];
        s->qact[i] = s->qact[i + 1];
    }
    for (k = 0; k < SPT_NZ; ++k) {
        SPT_QR(k, q - 1) = SPT_C(0.0);
    }
    q -= 1;
    for (j = l; j < q; ++j) {
        SPT_REAL a = SPT_QR(j, j), b = SPT_QR(j + 1, j), h, c, sn;
        if (b == SPT_C(0.0)) {
            continue;
        }
        h = spt_hypot(a, b);
        c = a / h;
        sn = b / h;
        SPT_QR(j, j) = h;
        SPT_QR(j + 1, j) = SPT_C(0.0);
        for (k = j + 1; k < q; ++k) {
            SPT_REAL t1 = SPT_QR(j, k), t2 = SPT_QR(j + 1, k);
            SPT_QR(j, k) = c * t1 + sn * t2;
            SPT_QR(j + 1, k) = sn * t1 - c * t2;
        }
        for (i = 0; i < SPT_NZ; ++i) {
            SPT_REAL t1 = SPT_QJ(i, j), t2 = SPT_QJ(i, j + 1);
            SPT_QJ(i, j) = c * t1 + sn * t2;
            SPT_QJ(i, j + 1) = sn * t1 - c * t2;
        }
    }
    return q;
}
{{WARM_FUNCS}}
/** x sonlu mu (NaN / ±∞ yok). */
static int spt_qp_finite(const spt_t *s)
{
    int i;
    for (i = 0; i < SPT_NZ; ++i) {
        if (!(SPT_FABS(s->qx[i]) <= SPT_REAL_MAX)) {
            return 0;
        }
    }
    return 1;
}

/** QP'yi çözer: sonuç s->qx, aktif küme s->iA (sıcak başlangıç girdisi de odur). Durum kodunu döndürür. */
static int spt_qp_solve(spt_t *s, const SPT_REAL *f)
{
    int i, j, l, p, q = 0, iters = 1, status = SPT_QP_RUN;
    /* J = Linvᵀ, R = 0, x = −H⁻¹ f = −Linvᵀ (Linv f) (kısıtsız en iyi) */
    for (i = 0; i < SPT_NZ; ++i) {
        for (j = 0; j < SPT_NZ; ++j) {
            SPT_QJ(i, j) = (j >= i) ? spt_Linv[j * (j + 1) / 2 + i] : SPT_C(0.0);
            SPT_QR(i, j) = SPT_C(0.0);
        }
    }
    for (j = 0; j < SPT_NZ; ++j) {
        const SPT_REAL *row = spt_Linv + j * (j + 1) / 2;
        SPT_REAL acc = SPT_C(0.0);
        for (i = 0; i <= j; ++i) {
            acc += row[i] * f[i];
        }
        s->qd[j] = acc;
    }
    for (i = 0; i < SPT_NZ; ++i) {
        SPT_REAL acc = SPT_C(0.0);
        for (j = i; j < SPT_NZ; ++j) {
            acc += spt_Linv[j * (j + 1) / 2 + i] * s->qd[j];
        }
        s->qx[i] = -acc;
    }
{{WARM_START}}
    while (status == SPT_QP_RUN) {
        SPT_REAL smin = SPT_C(0.0), up = SPT_C(0.0);
        /* 1. en çok ihlal edilen pasif kısıt */
        if (!spt_qp_finite(s)) {
            status = SPT_STATUS_NUMERICAL;
            break;
        }
        p = -1;
        for (i = 0; i < SPT_NC; ++i) {
            if (!s->iA[i]) {
                SPT_REAL si = spt_qp_slack(s, i);
                if (p < 0 || si < smin) {
                    smin = si;
                    p = i;
                }
            }
        }
        if (p < 0 || smin >= -SPT_TOL) {
            status = iters;                           /* en iyi (KKT sağlandı) */
            break;
        }
        /* 2. adım döngüsü (aynı p için) */
        for (;;) {
            SPT_REAL nd, nd2, t1 = SPT_C(0.0), t2 = SPT_C(0.0), t, rmax = SPT_C(0.0), rtol;
            int t2ok;
            if (!spt_qp_finite(s)) {
                status = SPT_STATUS_NUMERICAL;
                break;
            }
            spt_qp_jt(s, spt_Ac + p * SPT_NZ);
            spt_qp_back(s, q);
            nd = spt_qp_norm(s, 0);
            nd2 = (q < SPT_NZ) ? spt_qp_norm(s, q) : SPT_C(0.0);
            for (j = 0; j < q; ++j) {
                SPT_REAL aj = SPT_FABS(s->qr[j]);
                if (aj > rmax) {
                    rmax = aj;
                }
            }
            rtol = SPT_DEP_TOL * rmax;               /* yuvarlama düzeyindeki r_j > 0 sayılmaz */
            l = -1;
            for (j = 0; j < q; ++j) {
                if (s->qr[j] > rtol) {
                    SPT_REAL ratio = s->qu[j] / s->qr[j];
                    if (l < 0 || ratio < t1) {
                        t1 = ratio;
                        l = j;
                    }
                }
            }
            t2ok = !(nd2 <= SPT_DEP_TOL * nd);
            if (t2ok) {
                t2 = -spt_qp_slack(s, p) / (nd2 * nd2);
            }
            if (l < 0 && !t2ok) {
                status = SPT_STATUS_INFEASIBLE;
                break;
            }
            if (iters >= SPT_MAX_ITER) {
                status = SPT_STATUS_MAXITER;
                break;
            }
            ++iters;
            if (!t2ok) {                              /* yalnız çift adım: l çıkarılır */
                for (j = 0; j < q; ++j) {
                    s->qu[j] -= t1 * s->qr[j];
                }
                up += t1;
                s->iA[s->qact[l]] = 0;
                q = spt_qp_drop(s, l, q);
                continue;
            }
            t = (l < 0 || t2 <= t1) ? t2 : t1;
            spt_qp_dir(s, q);
            for (i = 0; i < SPT_NZ; ++i) {
                s->qx[i] += t * s->qz[i];
            }
            for (j = 0; j < q; ++j) {
                s->qu[j] -= t * s->qr[j];
            }
            up += t;
            if (l < 0 || t2 <= t1) {                  /* tam adım: p eklenir */
                spt_qp_add_column(s, q);
                s->qu[q] = up;
                s->qact[q] = p;
                s->iA[p] = 1;
                ++q;
                break;
            }
            s->iA[s->qact[l]] = 0;                    /* kısmi adım: l çıkarılır */
            q = spt_qp_drop(s, l, q);
        }
    }
    return status;
}
"""

_WARM_FUNCS = r"""
/** Önceki aktif kümedeki i. kısıtı tam adımla ekle (sıcak başlangıç). Bağımlıysa eklemez; yeni q'yu döndürür. */
static int spt_qp_force_add(spt_t *s, int i, int q)
{
    SPT_REAL nd, nd2, t;
    int j;
    if (q >= SPT_NZ) {
        return q;
    }
    spt_qp_jt(s, spt_Ac + i * SPT_NZ);
    nd = spt_qp_norm(s, 0);
    nd2 = spt_qp_norm(s, q);
    if (nd == SPT_C(0.0) || nd2 <= SPT_DEP_TOL * nd) {
        return q;
    }
    spt_qp_back(s, q);
    spt_qp_dir(s, q);
    t = -spt_qp_slack(s, i) / (nd2 * nd2);
    for (j = 0; j < SPT_NZ; ++j) {
        s->qx[j] += t * s->qz[j];
    }
    for (j = 0; j < q; ++j) {
        s->qu[j] -= t * s->qr[j];
    }
    spt_qp_add_column(s, q);
    s->qu[q] = t;
    s->qact[q] = i;
    s->iA[i] = 1;
    return q + 1;
}

/** Aktif küme üzerinde kapalı biçim: y₁ = R⁻ᵀ b_A, x = J₁ y₁ − J₂ J₂ᵀ f, u = R⁻¹ (y₁ + J₁ᵀ f). */
static void spt_qp_recompute(spt_t *s, const SPT_REAL *f, int q)
{
    int i, j, k;
    for (i = 0; i < q; ++i) {                         /* qz[0..q) = y₁ (Rᵀ alt üçgen: ileri yerine koyma) */
        SPT_REAL acc = s->qb[s->qact[i]];
        for (k = 0; k < i; ++k) {
            acc -= SPT_QR(k, i) * s->qz[k];
        }
        s->qz[i] = acc / SPT_QR(i, i);
    }
    for (j = 0; j < SPT_NZ; ++j) {                    /* qd = Jᵀ f (+ y₁ ilk q elemana) */
        SPT_REAL acc = SPT_C(0.0);
        for (i = 0; i < SPT_NZ; ++i) {
            acc += SPT_QJ(i, j) * f[i];
        }
        s->qd[j] = (j < q) ? acc + s->qz[j] : acc;
    }
    for (i = 0; i < SPT_NZ; ++i) {                    /* x = J₁ y₁ − J₂ (J₂ᵀ f) */
        SPT_REAL a1 = SPT_C(0.0), a2 = SPT_C(0.0);
        for (j = 0; j < q; ++j) {
            a1 += SPT_QJ(i, j) * s->qz[j];
        }
        for (j = q; j < SPT_NZ; ++j) {
            a2 += SPT_QJ(i, j) * s->qd[j];
        }
        s->qx[i] = a1 - a2;
    }
    for (i = q - 1; i >= 0; --i) {                    /* u = R⁻¹ (y₁ + J₁ᵀ f) */
        SPT_REAL acc = s->qd[i];
        for (k = i + 1; k < q; ++k) {
            acc -= SPT_QR(i, k) * s->qu[k];
        }
        s->qu[i] = acc / SPT_QR(i, i);
    }
}
"""

_WARM_START = r"""    {
        /* sıcak başlangıç: önceki aktif kısıtlar tam adımla eklenir, negatif çarpanlılar (en negatifi önce)
           çıkarılır ve kapalı biçimle yeniden hesaplanır (çıkarmalar yineleme sayılır — MATLAB gibi) */
        int added = 0;
        for (i = 0; i < SPT_NC; ++i) {
            if (s->iA[i]) {
                int q0 = q;
                s->iA[i] = 0;
                q = spt_qp_force_add(s, i, q);
                added |= (q != q0);
            }
        }
        while (added) {
            SPT_REAL worst = SPT_C(0.0);
            int wpos = -1;
            spt_qp_recompute(s, f, q);
            for (j = 0; j < q; ++j) {
                if (s->qu[j] < worst) {
                    worst = s->qu[j];
                    wpos = j;
                }
            }
            if (wpos < 0) {
                break;
            }
            s->iA[s->qact[wpos]] = 0;
            q = spt_qp_drop(s, wpos, q);
            ++iters;
        }
    }
"""

_COLD_START = r"""    for (i = 0; i < SPT_NC; ++i) {
        s->iA[i] = 0;                                 /* UseWarmStart kapalı: her adım boş kümeden */
    }
"""


def _gemv_helpers() -> str:
    return r"""
/** y += M x (M satır öncelikli rows × cols) */
static void spt_gemv_add(SPT_REAL *y, const SPT_REAL *M, const SPT_REAL *x, int rows, int cols)
{
    int i, j;
    for (i = 0; i < rows; ++i) {
        SPT_REAL acc = y[i];
        for (j = 0; j < cols; ++j) {
            acc += M[i * cols + j] * x[j];
        }
        y[i] = acc;
    }
}

/** y −= M x */
static void spt_gemv_sub(SPT_REAL *y, const SPT_REAL *M, const SPT_REAL *x, int rows, int cols)
{
    int i, j;
    for (i = 0; i < rows; ++i) {
        SPT_REAL acc = y[i];
        for (j = 0; j < cols; ++j) {
            acc -= M[i * cols + j] * x[j];
        }
        y[i] = acc;
    }
}

/** Sonlu mu (NaN / ±∞ değil). */
static int spt_finite_n(const SPT_REAL *v, int n)
{
    int i;
    for (i = 0; i < n; ++i) {
        if (!(SPT_FABS(v[i]) <= SPT_REAL_MAX)) {
            return 0;
        }
    }
    return 1;
}
"""


def _step_source(pl: _Plan) -> str:
    """spt_step(): MPC çevrimiçi adımı (plana göre gereksiz kısımlar üretilmez)."""
    L: list[str] = []
    a = L.append
    a("int spt_step(spt_t *s, const SPT_REAL *ym, const SPT_REAL *ref, int ref_rows,")
    a("             const SPT_REAL *md, int md_rows, SPT_REAL u[SPT_NMV])")
    a("{")
    a("    SPT_REAL e[SPT_NYM], xk[SPT_NX], xn[SPT_NX], f[SPT_NZ], ua[SPT_NMV];")
    if pl.nmd:
        a("    SPT_REAL v0[SPT_NMD];")
    a("    int i, j, status, ok;")
    if pl.nmd == 0:
        a("    (void)md;")
        a("    (void)md_rows;")
    a("    /* 0. giriş denetimi: geçersizse adım yapılmaz, MV tutulur, durum değişmez */")
    a("    ok = (ym == NULL || spt_finite_n(ym, SPT_NYM));")
    a("    if (ok && ref != NULL) {")
    if pl.ref_preview:
        a("        int rows = (ref_rows < SPT_P) ? ref_rows : SPT_P;     /* fazla satırlar kullanılmaz */")
        a("        ok = (ref_rows >= 1 && spt_finite_n(ref, rows * SPT_NY));")
    else:
        a("        ok = (ref_rows == 1 && spt_finite_n(ref, SPT_NY));    /* önizlemesiz üretildi: tek satır */")
    a("    }")
    if pl.nmd:
        a("    if (ok && md != NULL) {")
        if pl.md_preview:
            a("        int rows = (md_rows < SPT_P + 1) ? md_rows : SPT_P + 1;")
            a("        ok = (md_rows >= 1 && spt_finite_n(md, rows * SPT_NMD));")
        else:
            a("        ok = (md_rows == 1 && spt_finite_n(md, SPT_NMD));")
        a("    }")
    a("    if (!ok) {")
    a("        spt_get_last_mv(s, u);")
    a("        s->status = SPT_STATUS_BAD_INPUT;")
    a("        s->steps++;")
    a("        s->n_bad_input++;")
    a("        return SPT_STATUS_BAD_INPUT;")
    a("    }")
    a("    /* 1. ölçüm ve ölçülen bozucu: boyutsuz sapma (y = ym / S − y_off) */")
    a("    for (i = 0; i < SPT_NYM; ++i) {")
    a("        e[i] = (ym != NULL) ? ym[i] * spt_ym_iscale[i] - spt_ym_off[i] : SPT_C(0.0);")
    a("    }")
    if pl.nmd:
        a("    for (i = 0; i < SPT_NMD; ++i) {")
        a("        v0[i] = (md != NULL) ? md[i] * spt_md_iscale[i] - spt_md_off[i] : SPT_C(0.0);")
        a("    }")
    a("    /* 2. Kalman ölçüm güncellemesi: e = y − Cm x̂ − Dvm v(k);  x̂(k|k) = x̂ + M e */")
    a("    spt_gemv_sub(e, spt_Cm, s->x, SPT_NYM, SPT_NX);")
    if pl.nmd:
        a("    spt_gemv_sub(e, spt_Dvm, v0, SPT_NYM, SPT_NMD);")
    a("    for (i = 0; i < SPT_NX; ++i) {")
    a("        xk[i] = s->x[i];")
    a("    }")
    a("    spt_gemv_add(xk, spt_M, e, SPT_NX, SPT_NYM);")
    a("    /* 3. f = Kxᵀ x̂ + Ku1ᵀ u(k−1) + Krᵀ r + Kvᵀ v + [Kvᵀ ofset + Kutᵀ u_hedef] */")
    a("    for (j = 0; j < SPT_NFREE; ++j) {")
    a("        f[j] = spt_f0[j];")
    a("    }")
    if pl.nz > pl.nfree:
        a("    f[SPT_NFREE] = SPT_C(0.0);                    /* gevşek değişkenin doğrusal terimi */")
    a("    spt_gemv_add(f, spt_KxT, xk, SPT_NFREE, SPT_NXQP);")
    a("    spt_gemv_add(f, spt_Ku1T, s->u1, SPT_NFREE, SPT_NMV);")
    a("    if (ref != NULL) {")
    if pl.ref_preview:
        a("        for (i = 0; i < SPT_P; ++i) {                 /* satır i → y(k+1+i); eksik satırlar = son satır */")
        a("            const SPT_REAL *row = ref + (i < ref_rows ? i : ref_rows - 1) * SPT_NY;")
        a("            int k;")
        a("            for (k = 0; k < SPT_NY; ++k) {")
        a("                SPT_REAL rk = row[k] * spt_r_iscale[k] - spt_r_off[k];")
        a("                for (j = 0; j < SPT_NFREE; ++j) {")
        a("                    f[j] += spt_KrT[j * (SPT_P * SPT_NY) + i * SPT_NY + k] * rk;")
        a("                }")
        a("            }")
        a("        }")
    else:
        a("        SPT_REAL rs[SPT_NY];")
        a("        for (i = 0; i < SPT_NY; ++i) {")
        a("            rs[i] = ref[i] * spt_r_iscale[i] - spt_r_off[i];")
        a("        }")
        a("        spt_gemv_add(f, spt_KrT, rs, SPT_NFREE, SPT_NY);")
    a("    }")
    if pl.nmd:
        if pl.md_preview:
            a("    if (md != NULL) {")
            a("        for (i = 0; i <= SPT_P; ++i) {                /* satır i → v(k+i); eksikler = son satır */")
            a("            const SPT_REAL *row = md + (i < md_rows ? i : md_rows - 1) * SPT_NMD;")
            a("            int k;")
            a("            for (k = 0; k < SPT_NMD; ++k) {")
            a("                SPT_REAL vk = row[k] * spt_md_iscale[k] - spt_md_off[k];")
            a("                for (j = 0; j < SPT_NFREE; ++j) {")
            a("                    f[j] += spt_KvT[j * ((SPT_P + 1) * SPT_NMD) + i * SPT_NMD + k] * vk;")
            a("                }")
            if pl.isQP:
                a("                for (j = 0; j < SPT_NC; ++j) {")
                a("                    s->qb[j] -= spt_Mv[j * ((SPT_P + 1) * SPT_NMD) + i * SPT_NMD + k] * vk;")
                a("                }")
            a("            }")
            a("        }")
            a("    }")
        else:
            a("    spt_gemv_add(f, spt_KvT, v0, SPT_NFREE, SPT_NMD);")
    a("{{SOLVE}}")
    a("    /* 5. u(k) = u(k−1) + Δu(k) + u_off (MV mühendislik birimi: × ScaleFactor) */")
    a("    for (i = 0; i < SPT_NMV; ++i) {")
    a("        ua[i] = s->u1[i] + s_du(i) + spt_mv_off[i];")
    a("    }")
    if pl.subopt:
        a("#if SPT_SUBOPTIMAL")
        a("    if (status == SPT_STATUS_MAXITER) {   /* alt-en iyi hamle: ilk adım u / Δu sınırlarına kırpılır */")
        a("        for (i = 0; i < SPT_NMV; ++i) {")
        a("            SPT_REAL prev = s->u1[i] + spt_mv_off[i], du = ua[i] - prev;")
        a("            if (du > spt_clip_dumax[i]) {")
        a("                ua[i] = prev + spt_clip_dumax[i];")
        a("            } else if (du < spt_clip_dumin[i]) {")
        a("                ua[i] = prev + spt_clip_dumin[i];")
        a("            }")
        a("            if (ua[i] < spt_clip_umin[i]) {")
        a("                ua[i] = spt_clip_umin[i];")
        a("            }")
        a("            if (ua[i] > spt_clip_umax[i]) {")
        a("                ua[i] = spt_clip_umax[i];")
        a("            }")
        a("        }")
        a("    }")
        a("#endif")
    a("    /* 6. Kalman zaman güncellemesi: x̂(k+1|k) = A x̂ + Bu (u − u_off) + Bv v(k) + L e */")
    a("    for (i = 0; i < SPT_NMV; ++i) {")
    a("        u[i] = ua[i] * spt_mv_scale[i];")
    a("        ua[i] -= spt_mv_off[i];")
    a("        s->u1[i] = ua[i];")
    a("    }")
    a("    for (i = 0; i < SPT_NX; ++i) {")
    a("        xn[i] = SPT_C(0.0);")
    a("    }")
    a("    spt_gemv_add(xn, spt_A, s->x, SPT_NX, SPT_NX);")
    a("    spt_gemv_add(xn, spt_Bu, ua, SPT_NX, SPT_NMV);")
    if pl.nmd:
        a("    spt_gemv_add(xn, spt_Bv, v0, SPT_NX, SPT_NMD);")
    if pl.has_bv_off:
        a("    for (i = 0; i < SPT_NX; ++i) {")
        a("        xn[i] += spt_Bv_off[i];")
        a("    }")
    a("    spt_gemv_add(xn, spt_L, e, SPT_NX, SPT_NYM);")
    a("    for (i = 0; i < SPT_NX; ++i) {")
    a("        s->x[i] = xn[i];")
    a("    }")
    a("    /* durum sayaçları */")
    a("    s->status = status;")
    a("    s->steps++;")
    a("    if (status > 0) {")
    a("        s->n_ok++;")
    a("        if (status > s->iter_max) {")
    a("            s->iter_max = status;")
    a("        }")
    a("    } else if (status == SPT_STATUS_MAXITER) {")
    a("        s->n_maxiter++;")
    a("    } else if (status == SPT_STATUS_INFEASIBLE) {")
    a("        s->n_infeasible++;")
    a("    } else {")
    a("        s->n_numerical++;")
    a("    }")
    a("    return status;")
    a("}")
    text = "\n".join(L) + "\n"
    if pl.isQP:
        solve = []
        b = solve.append
        b("    /* 4. QP: b = −(Mlim + Mx x̂ + Mu1 u(k−1) + Mv v), sonra Goldfarb–Idnani */")
        b("    for (i = 0; i < SPT_NC; ++i) {")
        b("        SPT_REAL acc = -spt_Mlim0[i];")
        b("        for (j = 0; j < SPT_NXQP; ++j) {")
        b("            acc -= spt_Mx[i * SPT_NXQP + j] * xk[j];")
        b("        }")
        b("        for (j = 0; j < SPT_NMV; ++j) {")
        b("            acc -= spt_Mu1[i * SPT_NMV + j] * s->u1[j];")
        b("        }")
        if pl.nmd and not pl.md_preview:
            b("        for (j = 0; j < SPT_NMD; ++j) {")
            b("            acc -= spt_Mv[i * SPT_NMD + j] * v0[j];")
            b("        }")
        if pl.md_preview:
            b("        s->qb[i] += acc;                           /* MD önizleme katkısı yukarıda eklendi */")
        else:
            b("        s->qb[i] = acc;")
        b("    }")
        b("    status = spt_qp_solve(s, f);")
        b("    if (status < 0 || (status == SPT_STATUS_MAXITER && !SPT_SUBOPTIMAL)) {")
        b("        for (i = 0; i < SPT_NZ; ++i) {")
        b("            s->qx[i] = SPT_C(0.0);                 /* çözüm yok: Δu = 0 → MV tutulur */")
        b("        }")
        b("    }")
        b("    s->slack = s->qx[SPT_NZ - 1];")
        text = text.replace("{{SOLVE}}", "\n".join(solve))
        text = text.replace("s_du(i)", "s->qx[i]")
        if pl.md_preview:
            # qb önce sıfırlanmalı (MD önizleme döngüsü qb'ye yazar): f hazırlığının başına ekle
            text = text.replace("    for (j = 0; j < SPT_NFREE; ++j) {\n        f[j] = spt_f0[j];\n    }\n",
                                "    for (j = 0; j < SPT_NFREE; ++j) {\n        f[j] = spt_f0[j];\n    }\n"
                                "    for (j = 0; j < SPT_NC; ++j) {\n        s->qb[j] = SPT_C(0.0);\n    }\n", 1)
    else:
        solve = ["    /* 4. kısıtsız: Δu = −[H⁻¹]₁ f (kapalı biçim) */",
                 "    status = 1;",
                 "    s->slack = SPT_C(0.0);",
                 "    for (i = 0; i < SPT_NMV; ++i) {",
                 "        SPT_REAL acc = SPT_C(0.0);",
                 "        for (j = 0; j < SPT_NFREE; ++j) {",
                 "            acc += spt_Kmpc[i * SPT_NFREE + j] * f[j];",
                 "        }",
                 "        ua[i] = acc;",
                 "    }"]
        text = text.replace("{{SOLVE}}", "\n".join(solve))
        text = text.replace("ua[i] = s->u1[i] + s_du(i) + spt_mv_off[i];", "ua[i] = s->u1[i] + ua[i] + spt_mv_off[i];")
    return text


_API_SOURCE = r"""
void spt_init(spt_t *s)
{
    size_t i;
    unsigned char *b = (unsigned char *)s;
    for (i = 0; i < sizeof(*s); ++i) {
        b[i] = 0u;                                    /* tüm alanlar sıfır (gerçek sayılar dahil, aşağıda açıkça) */
    }
    for (i = 0; i < (size_t)SPT_NX; ++i) {
        s->x[i] = SPT_C(0.0);
    }
    for (i = 0; i < (size_t)SPT_NMV; ++i) {
        s->u1[i] = SPT_C(0.0);
    }
    s->slack = SPT_C(0.0);
    s->status = 1;
}

void spt_set_last_mv(spt_t *s, const SPT_REAL u[SPT_NMV])
{
    int i;
    for (i = 0; i < SPT_NMV; ++i) {
        s->u1[i] = u[i] * spt_mv_iscale[i] - spt_mv_off[i];
    }
}

void spt_get_last_mv(const spt_t *s, SPT_REAL u[SPT_NMV])
{
    int i;
    for (i = 0; i < SPT_NMV; ++i) {
        u[i] = (s->u1[i] + spt_mv_off[i]) * spt_mv_scale[i];
    }
}

void spt_get_state(const spt_t *s, SPT_REAL x[SPT_NX])
{
    int i;
    for (i = 0; i < SPT_NXP; ++i) {
        x[i] = s->x[i] + spt_xp_off[i];
    }
    for (i = SPT_NXP; i < SPT_NX; ++i) {
        x[i] = s->x[i];
    }
}

void spt_set_state(spt_t *s, const SPT_REAL x[SPT_NX])
{
    int i;
    for (i = 0; i < SPT_NXP; ++i) {
        s->x[i] = x[i] - spt_xp_off[i];
    }
    for (i = SPT_NXP; i < SPT_NX; ++i) {
        s->x[i] = x[i];
    }
}
"""


def _header(pl: _Plan, opts: MpcCodegenOptions) -> str:
    qp = pl.nc > 0
    qp_cfg = _fill(_QP_CONFIG, MAX_ITER=str(pl.max_iter), TOL=_lit(pl.tol), WARM=str(int(pl.warm)),
                   SUBOPT=str(int(pl.subopt))) if qp else ""
    nxd = pl.nxQP - pl.nxp
    return _fill(_HEADER, NAME=pl.name.replace("*/", "* /"), BANNER=_BANNER, TS=f"{pl.Ts:g}", TS_LIT=_lit(pl.Ts),
                 P=str(pl.p), MOVES="[" + " ".join(str(m) for m in pl.moves) + "]", NMV=str(pl.nmv),
                 NMD=str(pl.nmd), NYM=str(pl.nmy), NUO=str(len(pl.uyindex)), NZ=str(pl.nz), NC=str(pl.nc),
                 MV_LIST=_sig_list(pl.mv_names, pl.mv_units),
                 MO_LIST=_sig_list(pl.mo_names, [pl.ov_units[i] for i in pl.myindex], pl.myindex),
                 OV_LIST=_sig_list(pl.ov_names, pl.ov_units), MD_LIST=_sig_list(pl.md_names, pl.md_units),
                 USE_DOUBLE="1" if opts.real == "double" else "0", REAL=opts.real, NX=str(pl.nx), NXP=str(pl.nxp),
                 NXD=str(nxd), NXN=str(pl.nx - pl.nxQP), NXQP=str(pl.nxQP), NY=str(pl.ny), NFREE=str(pl.nfree),
                 SLACK_NOTE=" + gevşek ε" if pl.nz > pl.nfree else "", REF_PREVIEW=str(int(pl.ref_preview)),
                 MD_PREVIEW=str(int(pl.md_preview)), QP_CONFIG=qp_cfg,
                 QP_REAL_FIELDS=_QP_REAL_FIELDS if qp else "", QP_INT_FIELDS=_QP_INT_FIELDS if qp else "",
                 QP_CHAR_FIELDS=_QP_CHAR_FIELDS if qp else "")


def _source(pl: _Plan) -> str:
    name = pl.name.replace("*/", "* /")
    head = (f"/**\n * @file spt.c\n * @brief MPC denetleyicisi \"{name}\" — sabit veriler (flash) ve çevrimiçi adım."
            f"\n *\n * {_BANNER}\n *\n * Matrisler MPCData (MATLAB getCodeGenerationData) adlarıyla, boyutsuz (ölçekli)"
            " iç biçimdedir.\n * Bu dosyayı elle değiştirmeyin: MPC tasarımı değişince yeniden üretin.\n */\n"
            "#include \"spt.h\"\n\n")
    huge = _huge_matrix_arrays(pl)
    if huge:
        head += ("#if !defined(SPT_CUSTOM_REAL) && !SPT_USE_DOUBLE\n"
                 f"#error \"spt: {', '.join(huge)} sabitleri float aralığını aşıyor (|x| > FLT_MAX): "
                 "SPT_USE_DOUBLE=1 ile derleyin ya da ölçek çarpanlarını düzeltin.\"\n#endif\n\n")
    data = "\n\n".join(_c_array(a) for a in pl.arrays)
    body = _gemv_helpers()
    if pl.nc > 0:
        body += _fill(_QP_SOURCE, WARM_FUNCS=_WARM_FUNCS if pl.warm else "",
                      WARM_START=_WARM_START if pl.warm else _COLD_START)
    return head + data + "\n" + body + _API_SOURCE + "\n" + _step_source(pl)


_EXAMPLE = r"""/**
 * @file spt_ornek.c
 * @brief Örnek: MPC denetleyicisinin sabit örnekleme periyoduyla (Ts = {{TS}} s) çağrılması ve süre ölçümü.
 *
 * {{BANNER}}
 *
 * Donanıma bağlı dört fonksiyonu (olcum_oku, referans_al, eyleyici_yaz, periyot_bekle) kullanıcı yazar (STM32 HAL,
 * Zephyr, FreeRTOS …). spt_ornek_gorev() main() içinden ya da bir RTOS görevinden çağrılır ve geri dönmez.
 */
#include <stdint.h>
#include <stddef.h>
#include "spt.h"

#ifndef ORNEK_CPU_HZ
#define ORNEK_CPU_HZ {{CPU}}   /* çekirdek saati (Hz): DWT->CYCCNT çevrimlerini süreye çevirmek için */
#endif

/* ---- kullanıcı tarafından sağlanır --------------------------------------------------------------------------- */
/** Ölçülen çıkışları okur (mühendislik birimi, MPC tasarımındaki sırayla): {{MO_LIST}} */
extern void olcum_oku(SPT_REAL ym[SPT_NYM]);
/** Referansı verir ({{REF_ROWS}} satır × SPT_NY): {{OV_LIST}} */
extern void referans_al(SPT_REAL *ref);
{{MD_DECL}}/** MV'yi eyleyiciye yazar (mühendislik birimi): {{MV_LIST}} */
extern void eyleyici_yaz(const SPT_REAL u[SPT_NMV]);
/** Bir sonraki Ts anına kadar bekler (donanım zamanlayıcısı bayrağı, vTaskDelayUntil …). */
extern void periyot_bekle(void);
/** Tanılama (isteğe bağlı): durum kodu ve adımın çevrim sayısı. */
extern void tanilama(int durum, uint32_t cevrim);

/* ---- DWT çevrim sayacı (Cortex-M3/M4/M7/M33; diğerlerinde 0) ------------------------------------------------- */
#if defined(__ARM_ARCH_7M__) || defined(__ARM_ARCH_7EM__) || defined(__ARM_ARCH_8M_MAIN__)
#define ORNEK_DWT_CTRL   (*(volatile uint32_t *)0xE0001000u)
#define ORNEK_DWT_CYCCNT (*(volatile uint32_t *)0xE0001004u)
#define ORNEK_DEMCR      (*(volatile uint32_t *)0xE000EDFCu)
static void cyccnt_baslat(void)
{
    ORNEK_DEMCR |= 0x01000000u;
    ORNEK_DWT_CYCCNT = 0u;
    ORNEK_DWT_CTRL |= 1u;
}
static uint32_t cyccnt(void)
{
    return ORNEK_DWT_CYCCNT;
}
#else
static void cyccnt_baslat(void)
{
}
static uint32_t cyccnt(void)
{
    return 0u;
}
#endif

void spt_ornek_gorev(void);

void spt_ornek_gorev(void)
{
    static spt_t ctl;                       /* durum statik: yığında yer kaplamaz */
    SPT_REAL ym[SPT_NYM], ref[{{REF_ROWS}} * SPT_NY], u[SPT_NMV];
{{MD_VARS}}    cyccnt_baslat();
    spt_init(&ctl);
    /* El → otomatik geçişte o anki eyleyici komutuyla başlayın (darbesiz): spt_set_last_mv(&ctl, u_simdiki); */
    for (;;) {
        uint32_t t0, t1;
        int durum;
        periyot_bekle();                    /* Ts = {{TS}} s: ölçüm her periyotta AYNI anda alınmalı */
        olcum_oku(ym);
        referans_al(ref);
{{MD_READ}}        t0 = cyccnt();
        durum = spt_step(&ctl, ym, ref, {{REF_ROWS}}, {{MD_ARGS}}, u);
        t1 = cyccnt();
        eyleyici_yaz(u);                    /* hata durumlarında da u geçerlidir (önceki MV) */
        tanilama(durum, (uint32_t)(t1 - t0));
    }
}
"""


def _example(pl: _Plan, opts: MpcCodegenOptions) -> str:
    ref_rows = "SPT_P" if pl.ref_preview else "1"
    md_rows = "(SPT_P + 1)" if pl.md_preview else "1"
    if pl.nmd:
        md_decl = (f"/** Ölçülen bozucuları verir ({md_rows} satır × SPT_NMD): "
                   f"{_sig_list(pl.md_names, pl.md_units)} */\nextern void bozucu_oku(SPT_REAL *md);\n")
        md_vars = f"    SPT_REAL md[{md_rows} * SPT_NMD];\n"
        md_read = "        bozucu_oku(md);\n"
        md_args = f"md, {md_rows}"
    else:
        md_decl = md_vars = md_read = ""
        md_args = "NULL, 0"
    return _fill(_EXAMPLE, TS=f"{pl.Ts:g}", BANNER=_BANNER, CPU=f"{opts.clock_mhz * 1e6:.0f}u",
                 MO_LIST=_sig_list(pl.mo_names, [pl.ov_units[i] for i in pl.myindex]),
                 OV_LIST=_sig_list(pl.ov_names, pl.ov_units), MV_LIST=_sig_list(pl.mv_names, pl.mv_units),
                 REF_ROWS=ref_rows, MD_DECL=md_decl, MD_VARS=md_vars, MD_READ=md_read, MD_ARGS=md_args)


def _sources(pl: _Plan, opts: MpcCodegenOptions) -> dict[str, str]:
    """Önek uygulanmamış (spt) kaynaklar: ölçüm ve ana makine derlemeleri için."""
    return {"spt.h": _header(pl, opts), "spt.c": _source(pl), "spt_ornek.c": _example(pl, opts)}


_COND_EPS_LIMIT = 1e-3     # koşul × eps bundan büyükse hamlelerin göreli hatası anlamlı olabilir


def numeric_warnings(mpcobj, opts: MpcCodegenOptions | None = None) -> list[str]:
    """Seçilen sayı türünde sayısal güvenilirlik uyarıları (Türkçe; boşsa sorun yok).

    - Hessian düzeltildiyse (MATLAB BadH ≠ 0: tekil / tekile çok yakın H köşegeni artırılır): çözüm, maliyetin
      belirlemediği yönlerde (ör. aynı etkiyi yapan iki MV, sıfır MV ve oran ağırlığı) bu düzeltmeye bağlıdır;
      float'ta bu yönleri yuvarlama belirler.
    - koşul(H_serbest) × eps > 1e-3: hamlelerin göreli hatası ~koşul × eps olabilir. Gevşek değişken ε köşegende
      ayrık olduğundan (yalnız ölçek) hesaba katılmaz; H, QP'nin gerçekten kullandığı (düzeltilmiş) Hessian'dır."""
    opts = opts or MpcCodegenOptions()
    d = _mpc_data(mpcobj)
    nf = d.nfree
    Lf = np.linalg.inv(d.Linv)
    Hf = (Lf @ Lf.T)[:nf, :nf]
    cond = float(np.linalg.cond(Hf))
    flt = opts.real == "float"
    eps = FLT_EPSILON if flt else DBL_EPSILON
    out = []
    if d.BadH:
        out.append(f"QP Hessian'ı tekil ya da tekile çok yakın (MATLAB BadH = {d.BadH}); köşegeni küçük bir değerle "
                   "artırıldı (MATLAB / Python ile aynı düzeltme). Hamleler, maliyetin belirlemediği yönlerde (ör. aynı "
                   "etkiyi yapan MV'ler; MV ve MV oranı ağırlıkları 0) bu düzeltmeye bağlıdır"
                   + ("; float'ta bu yönleri yuvarlama belirler ve MV'ler Python / MATLAB'den çok farklı çıkabilir. "
                      "double kullanın ya da MV / MV oranı ağırlıklarını sıfırdan büyük yapın." if flt else "."))
    if cond * eps > _COND_EPS_LIMIT:
        out.append(f"QP Hessian'ının koşul sayısı {cond:.3g}: {opts.real} ile hamlelerin göreli hatası koşul × eps ≈ "
                   f"{cond * eps:.2g} olabilir"
                   + ("; sert sınırlar da ConstraintTolerance'ın ötesinde birkaç ulp aşılabilir. double kullanın ya da "
                      "MV oranı ağırlığını artırın / ScaleFactor'ları sinyal aralıklarına göre seçin." if flt else
                      ". MV oranı ağırlığını artırın / ScaleFactor'ları sinyal aralıklarına göre seçin."))
    return out


def generate(mpcobj, opts: MpcCodegenOptions | None = None, name: str | None = None,
             footprint_rows: list[FootprintRow] | None = None,
             cost_rows: list[MpcCostRow] | None = None) -> list[GeneratedFile]:
    """<önek>.h, <önek>.c, <önek>_ornek.c ve KULLANIM_<önek>.md (önek uygulanmış). Sayısal güvenilirlik sorunu
    varsa (bkz. :func:`numeric_warnings`) ControlWarning verilir ve KULLANIM'ın başına yazılır."""
    opts = opts or MpcCodegenOptions()
    check_options(opts)
    pl = plan(mpcobj, opts, name)
    for w in numeric_warnings(mpcobj, opts):
        warnings.warn(f"MPC C kodu ({opts.real}): {w}", ControlWarning, stacklevel=2)
    src = _sources(pl, opts)
    files = [GeneratedFile(_prefixed(n, opts.prefix), _prefixed(t, opts.prefix)) for n, t in src.items()]
    files.append(GeneratedFile(file_names(opts.prefix)[3], usage_markdown(mpcobj, opts, name, footprint_rows,
                                                                          cost_rows, pl)))
    return files


# ============================================================================ durum yapısı boyutu (analitik)
def state_size(pl: _Plan, real: str) -> int:
    """sizeof(spt_t): ILP32 (ARM AAPCS) ve x86-64'te aynı (işaretçi alanı yok; gerçek sayılar önce)."""
    r = 8 if real == "double" else 4
    reals = pl.nx + pl.nmv + 1
    ints = 2
    chars = 0
    if pl.nc:
        reals += 2 * pl.nz * pl.nz + 5 * pl.nz + pl.nc
        ints += pl.nz
        chars = pl.nc
    size = reals * r + ints * 4 + 6 * 4 + chars
    align = max(r, 4)
    return -(-size // align) * align


# ============================================================================ ana makine sürücüsü ve sayaç
_DRIVER = r"""/* Ana makine doğrulama / maliyet sürücüsü (MCU'ya eklenmez).
   Giriş: [kip, T, ref_satır, md_satır, ilk_durum] (+ ilk durum) (+ kapalı çevrimde bitki bloğu) + adım verileri.
   Çıkış (adım başına): u[NMV], durum, ε, x̂(k+1|k)[NX], y[NY] (+ sayaç kipinde işlem sayıları). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "spt.h"

#if defined(SPT_CUSTOM_REAL)
unsigned long long spt_ops[SPT_NOPS];
#define DRV_R(x) spt_cnt(x)
#define DRV_D(r) ((r).v)
#define DRV_NOPS SPT_NOPS
#else
#define DRV_R(x) ((SPT_REAL)(x))
#define DRV_D(r) ((double)(r))
#define DRV_NOPS 0
#endif
#define DRV_NMD1 (SPT_NMD + 1)
#define DRV_MAXR (SPT_P + 3)

static FILE *drv_in;
static int drv_fail;
static spt_t drv_st;
static SPT_REAL drv_ym[SPT_NYM], drv_u[SPT_NMV], drv_x[SPT_NX];
static SPT_REAL drv_ref[DRV_MAXR * SPT_NY], drv_md[DRV_MAXR * DRV_NMD1];

static double drv_get(void)
{
    double v = 0.0;
    if (fread(&v, sizeof(double), 1, drv_in) != 1) {
        drv_fail = 1;
    }
    return v;
}

static double *drv_vec(long n)
{
    double *p = (double *)malloc(sizeof(double) * (size_t)(n > 0 ? n : 1));
    long i;
    if (p == NULL) {
        drv_fail = 1;
        return p;
    }
    for (i = 0; i < n; ++i) {
        p[i] = drv_get();
    }
    return p;
}

static void drv_put(FILE *out, double v)
{
    fwrite(&v, sizeof(double), 1, out);
}

int main(int argc, char **argv)
{
    FILE *out;
    long mode, T, rr, vr, t, i, j, nxs = 0, nds = 0;
    double *Ap = NULL, *Bup = NULL, *Bvp = NULL, *Bdp = NULL, *Cp = NULL, *Dvp = NULL, *Ddp = NULL, *xp = NULL;
    double *ypoff = NULL, *upoff = NULL, *dyoff = NULL, *yoffc = NULL, *ysc = NULL, *mvsc = NULL, *mo = NULL;
    double *y = NULL, *xq = NULL, *w = NULL, *dd = NULL, *yn = NULL, *un = NULL, *vp = NULL;
    unsigned long long before[DRV_NOPS + 1];
    if (argc < 3) {
        return 2;
    }
    drv_in = fopen(argv[1], "rb");
    if (drv_in == NULL) {
        return 3;
    }
    mode = (long)drv_get();
    T = (long)drv_get();
    rr = (long)drv_get();
    vr = (long)drv_get();
    spt_init(&drv_st);
    if (drv_get() != 0.0) {
        for (i = 0; i < SPT_NX; ++i) {
            drv_x[i] = DRV_R(drv_get());
        }
        for (i = 0; i < SPT_NMV; ++i) {
            drv_u[i] = DRV_R(drv_get());
        }
        spt_set_state(&drv_st, drv_x);
        spt_set_last_mv(&drv_st, drv_u);
    }
    if (rr > DRV_MAXR || vr > DRV_MAXR) {
        return 4;
    }
    if (mode == 1) {
        nxs = (long)drv_get();
        nds = (long)drv_get();
        Ap = drv_vec(nxs * nxs);
        Bup = drv_vec(nxs * SPT_NMV);
        Bvp = drv_vec(nxs * DRV_NMD1);
        Bdp = drv_vec(nxs * nds);
        Cp = drv_vec(SPT_NY * nxs);
        Dvp = drv_vec(SPT_NY * DRV_NMD1);
        Ddp = drv_vec(SPT_NY * nds);
        xp = drv_vec(nxs);
        ypoff = drv_vec(SPT_NY);
        upoff = drv_vec(SPT_NMV);
        dyoff = drv_vec(SPT_NYM);
        yoffc = drv_vec(SPT_NYM);
        ysc = drv_vec(SPT_NY);
        mvsc = drv_vec(SPT_NMV);
        mo = drv_vec(SPT_NYM);
        y = (double *)malloc(sizeof(double) * (size_t)SPT_NY);
        xq = (double *)malloc(sizeof(double) * (size_t)(nxs > 0 ? nxs : 1));
        w = (double *)malloc(sizeof(double) * (size_t)(SPT_NMV + 1));
        if (y == NULL || xq == NULL || w == NULL) {
            return 5;
        }
    }
    out = fopen(argv[2], "wb");
    if (out == NULL || drv_fail) {
        return 6;
    }
    for (t = 0; t < T; ++t) {
        int status;
        if (mode == 0) {
            for (i = 0; i < SPT_NYM; ++i) {
                drv_ym[i] = DRV_R(drv_get());
            }
        }
        for (i = 0; i < rr * SPT_NY; ++i) {
            drv_ref[i] = DRV_R(drv_get());
        }
        for (i = 0; i < vr * SPT_NMD; ++i) {
            drv_md[i] = DRV_R(drv_get());
        }
        if (mode == 1) {
            vp = drv_vec(SPT_NMD);
            dd = drv_vec(nds);
            yn = drv_vec(SPT_NYM);
            un = drv_vec(SPT_NMV);
            for (i = 0; i < SPT_NY; ++i) {           /* y = Cp xp + Dvp [vp; 1] + Ddp d (boyutsuz bitki sapması) */
                double acc = 0.0;
                for (j = 0; j < nxs; ++j) {
                    acc += Cp[i * nxs + j] * xp[j];
                }
                for (j = 0; j < SPT_NMD; ++j) {
                    acc += Dvp[i * DRV_NMD1 + j] * vp[j];
                }
                acc += Dvp[i * DRV_NMD1 + SPT_NMD];
                for (j = 0; j < nds; ++j) {
                    acc += Ddp[i * nds + j] * dd[j];
                }
                y[i] = acc;
            }
            for (i = 0; i < SPT_NYM; ++i) {
                long k = (long)mo[i];
                drv_ym[i] = DRV_R((y[k] + dyoff[i] + yn[i] + yoffc[i]) * ysc[k]);
            }
        }
        if (drv_fail) {
            return 7;
        }
#if defined(SPT_CUSTOM_REAL)
        memcpy(before, spt_ops, sizeof(spt_ops));
#else
        (void)before;
#endif
        status = spt_step(&drv_st, drv_ym, rr > 0 ? drv_ref : NULL, (int)rr, vr > 0 ? drv_md : NULL, (int)vr,
                          drv_u);
#if defined(SPT_CUSTOM_REAL)
        for (i = 0; i < SPT_NOPS; ++i) {
            before[i] = spt_ops[i] - before[i];          /* yalnız spt_step()'in işlemleri */
        }
#endif
        if (mode == 1) {
            for (i = 0; i < SPT_NMV; ++i) {
                w[i] = DRV_D(drv_u[i]) / mvsc[i] + un[i];        /* uygulanan (boyutsuz, gürültülü) MV */
            }
            for (i = 0; i < nxs; ++i) {
                double acc = 0.0;
                for (j = 0; j < nxs; ++j) {
                    acc += Ap[i * nxs + j] * xp[j];
                }
                for (j = 0; j < SPT_NMV; ++j) {
                    acc += Bup[i * SPT_NMV + j] * (w[j] - upoff[j]);
                }
                for (j = 0; j < SPT_NMD; ++j) {
                    acc += Bvp[i * DRV_NMD1 + j] * vp[j];
                }
                acc += Bvp[i * DRV_NMD1 + SPT_NMD];
                for (j = 0; j < nds; ++j) {
                    acc += Bdp[i * nds + j] * dd[j];
                }
                xq[i] = acc;
            }
            memcpy(xp, xq, sizeof(double) * (size_t)nxs);
        }
        spt_get_state(&drv_st, drv_x);
        for (i = 0; i < SPT_NMV; ++i) {
            drv_put(out, mode == 1 ? w[i] * mvsc[i] : DRV_D(drv_u[i]));
        }
        drv_put(out, (double)status);
        drv_put(out, DRV_D(drv_st.slack));
        for (i = 0; i < SPT_NX; ++i) {
            drv_put(out, DRV_D(drv_x[i]));
        }
        for (i = 0; i < SPT_NY; ++i) {
            drv_put(out, mode == 1 ? (y[i] + ypoff[i]) * ysc[i] : 0.0);
        }
#if defined(SPT_CUSTOM_REAL)
        for (i = 0; i < SPT_NOPS; ++i) {
            drv_put(out, (double)before[i]);
        }
#endif
        if (mode == 1) {
            free(vp);
            free(dd);
            free(yn);
            free(un);
        }
    }
    fclose(out);
    fclose(drv_in);
    return 0;
}
"""


def host_program(mpcobj, opts: MpcCodegenOptions | None = None, *, real: str | None = None,
                 count_ops: bool = False) -> tc.Build:
    """Üretilen C'yi ana makinede derler (doğrulama sürücüsüyle). count_ops: C++ olarak, SPT_REAL = sayaç türü."""
    opts = opts or MpcCodegenOptions()
    check_options(opts)
    pl = plan(mpcobj, opts)
    src = _sources(pl, opts)
    files = {"spt.h": src["spt.h"], "spt.c": src["spt.c"], "spt_host_driver.c": _DRIVER}
    sources = ["spt.c", "spt_host_driver.c"]
    if count_ops:
        files["spt_count_prelude.hpp"] = _count_prelude()
        return tc.host_build(files, sources, cxx=True, extra=["-std=c++17", "-include", "spt_count_prelude.hpp"],
                             opt="-O2", output="spt_mpc_count")
    real = real or opts.real
    return tc.host_build(files, sources, defines=[f"SPT_USE_DOUBLE={1 if real == 'double' else 0}"],
                         extra=["-std=c99", "-Wall", "-Wextra", "-pedantic"], opt="-O2", output=f"spt_mpc_{real}")


def _rows(arr, rows: int, ncol: int, what: str) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if a.ndim <= 1:
        a = a.reshape(1, -1) if a.size == ncol else a.reshape(-1, ncol)
    if a.shape[-1] != ncol:
        raise ValueError(f"{what} {ncol} sütunlu olmalı")
    return a


def _execute(build: tc.Build, blob: bytes, T: int, pl: _Plan, count_ops: bool) -> HostRun:
    if not build.ok:
        raise RuntimeError("Ana makine derlemesi başarısız:\n" + build.log)
    tag = hashlib.sha256(blob).hexdigest()[:16]
    runs = build.workdir / "runs"
    runs.mkdir(exist_ok=True)
    inp, out = runs / f"in_{tag}.bin", runs / f"out_{tag}.bin"
    if not out.exists():
        inp.write_bytes(blob)
        res = tc.run_program(build, [str(inp), str(out)], timeout=600)
        if res.returncode != 0 or not out.exists():
            out.unlink(missing_ok=True)
            raise RuntimeError(f"Sürücü hatası ({res.returncode}): {res.stderr}")
        inp.unlink(missing_ok=True)
    rec = pl.nmv + 2 + pl.nx + pl.ny + (len(OPS) if count_ops else 0)
    raw = np.fromfile(out, dtype="<f8")
    if raw.size != T * rec:
        out.unlink(missing_ok=True)
        raise RuntimeError("Sürücü çıktısı eksik")
    raw = raw.reshape(T, rec)
    k = pl.nmv
    u, status, slack = raw[:, :k], raw[:, k].astype(int), raw[:, k + 1]
    x = raw[:, k + 2:k + 2 + pl.nx]
    y = raw[:, k + 2 + pl.nx:k + 2 + pl.nx + pl.ny]
    ops = raw[:, k + 2 + pl.nx + pl.ny:] if count_ops else None
    return HostRun(u, status, slack, x, y, ops)


def _header_blob(mode: int, T: int, rr: int, vr: int, x0, u0) -> list[float]:
    head = [float(mode), float(T), float(rr), float(vr)]
    if x0 is None and u0 is None:
        return head + [0.0]
    return head + [1.0] + list(np.asarray(x0, float).ravel()) + list(np.asarray(u0, float).ravel())


def run_steps(mpcobj, opts: MpcCodegenOptions, ym, ref=None, md=None, *, x0=None, u0=None, real: str | None = None,
              count_ops: bool = False) -> HostRun:
    """Açık çevrim mpcmove dizisi: her adımda verilen ym (T × nmy), ref (T × satır × ny ya da T × ny) ve md
    (T × satır × nmd ya da T × nmd) mühendislik birimiyle. x0 / u0: başlangıç kestirici durumu ve son MV
    (verilmezse spt_init = mpcstate varsayılanı). ref / md None → NULL (nominal)."""
    pl = plan(mpcobj, opts)
    ym = np.asarray(ym, dtype=float).reshape(-1, pl.nmy)
    T = ym.shape[0]
    refa = None if ref is None else np.asarray(ref, dtype=float).reshape(T, -1, pl.ny)
    mda = None if (md is None or pl.nmd == 0) else np.asarray(md, dtype=float).reshape(T, -1, pl.nmd)
    rr = 0 if refa is None else refa.shape[1]
    vr = 0 if mda is None else mda.shape[1]
    if x0 is not None or u0 is not None:
        x0 = np.zeros(pl.nx) if x0 is None else x0
        u0 = (np.asarray(pl.scales["mv_nom"])) if u0 is None else u0
    blob = _header_blob(0, T, rr, vr, x0, u0)
    for t in range(T):
        blob += list(ym[t])
        if refa is not None:
            blob += list(refa[t].ravel())
        if mda is not None:
            blob += list(mda[t].ravel())
    build = host_program(mpcobj, opts, real=real, count_ops=count_ops)
    return _execute(build, np.asarray(blob, dtype="<f8").tobytes(), T, pl, count_ops)


def run_closed_loop(mpcobj, opts: MpcCodegenOptions, T: int, r=None, v=None, options=None, *,
                    real: str | None = None, count_ops: bool = False) -> HostRun:
    """Kapalı çevrim benzetimi (Python `sim(mpcobj, T, r, v, options)` ile aynı bitki, gürültü ve önizleme
    kuralları; denetleyici = üretilen C). options: mpcsimopt ya da sözlük (Model, PlantInitialState,
    UnmeasuredDisturbance, InputNoise, OutputNoise, RefLookAhead, MDLookAhead). Kısıtları kapatma / açık çevrim
    desteklenmez."""
    mpcmod = importlib.import_module("spt.control.mpc.mpc")   # modül (paket aynı adlı fonksiyonu dışa verir)
    pl = plan(mpcobj, opts)
    d = mpcobj.data
    opt = mpcmod.mpcsimopt() if options is None else (options if isinstance(options, mpcmod.mpcsimopt)
                                                      else mpcmod.mpcsimopt(options))
    for k in ("Constraints", "OpenLoop", "MVSignal", "ControllerInitialState"):
        if opt[k] is not None and not (k == "Constraints" and str(opt[k]).lower() == "on"):
            raise ValueError(f"run_closed_loop: '{k}' seçeneği desteklenmiyor")

    def flag(val):
        return val is not None and (val is True or str(val).lower() == "on")

    ref_prev, md_prev = flag(opt.RefLookAhead), flag(opt.MDLookAhead)
    if ref_prev and not pl.ref_preview:
        raise ValueError("Referans önizlemesi için kod ref_preview=True ile üretilmeli")
    if md_prev and not pl.md_preview and pl.nmd:
        raise ValueError("MD önizlemesi için kod md_preview=True ile üretilmeli")
    sp = mpcmod._sim_plant(mpcobj, d, opt.Model)
    ny, nmd, nmy, nmv, p = d.ny, d.nmd, d.nmy, d.nmv, d.p
    rr_eng = (d.yoff * d.Yscale).reshape(1, ny) if r is None else _rows(r, 0, ny, "Referans")
    vv_eng = (d.mdoff * d.Uscale[d.mdindex]).reshape(1, nmd) if (v is None or nmd == 0) else _rows(v, 0, nmd, "MD")
    dd = mpcmod._signal(opt.UnmeasuredDisturbance, sp["ndp"], sp["dscale"], "Ölçülmeyen bozucu")
    yn = mpcmod._signal(opt.OutputNoise, nmy, d.Yscale[d.myindex], "Çıkış gürültüsü")
    un = mpcmod._signal(opt.InputNoise, nmv, d.Uscale[d.mvindex], "Giriş gürültüsü")

    def get(a, k, n):
        if a is None:
            return np.zeros(n)
        return a[min(k, a.shape[0] - 1)]

    nxs = sp["Ap"].shape[0]
    nds = int(sp["ndp"])
    x0p = sp["xpoff"] if opt.PlantInitialState is None else np.asarray(opt.PlantInitialState, float).ravel()
    xp0 = x0p - sp["xpoff"]
    dvoff = sp["vpoff"] - d.mdoff
    dyoff = sp["ypoff"] - d.yoff
    rr = p if ref_prev else 1
    vr = (p + 1 if md_prev else 1) if nmd else 0
    blob = _header_blob(1, T, rr, vr, None, None)
    blob += [float(nxs), float(nds)]
    for m in (sp["Ap"], sp["Bup"], sp["Bvp"], sp["Bdp"].reshape(nxs, nds), sp["Cp"], sp["Dvp"],
              sp["Ddp"].reshape(ny, nds), xp0, sp["ypoff"], sp["upoff"], dyoff[d.myindex], d.yoff[d.myindex],
              d.Yscale, d.Uscale[d.mvindex], d.myindex.astype(float)):
        blob += list(np.asarray(m, dtype=float).ravel())
    for t in range(T):
        if ref_prev:
            rows = [get(rr_eng, t + i, ny) for i in range(1, p + 1)]
        else:
            rows = [get(rr_eng, t, ny)]
        blob += list(np.concatenate(rows))
        if nmd:
            if md_prev:
                vrows = [get(vv_eng, t + i, nmd) for i in range(p + 1)]
            else:
                vrows = [get(vv_eng, t, nmd)]
            blob += list(np.concatenate(vrows))
        vt = get(vv_eng, t, nmd) / d.Uscale[d.mdindex] - d.mdoff if nmd else np.zeros(0)
        blob += list(vt + dvoff)
        blob += list(get(dd, t, nds))
        blob += list(get(yn, t, nmy))
        blob += list(get(un, t, nmv))
    build = host_program(mpcobj, opts, real=real, count_ops=count_ops)
    return _execute(build, np.asarray(blob, dtype="<f8").tobytes(), T, pl, count_ops)


# ============================================================================ maliyet
def _op_weights(prof: McuProfile, real: str) -> np.ndarray:
    c = prof.cycles
    w = []
    for op in OPS:
        v = c[cycle_key(op, real)]
        if op in BASIC_OPS:
            v += c["mem"]
        elif op in LIBRARY_OPS:
            v += c["call"]
        w.append(v)
    return np.array(w)


def _ops(**kw) -> np.ndarray:
    v = np.zeros(len(OPS))
    for k, n in kw.items():
        v[OPS.index(k)] += n
    return v


def step_ops_bound(pl: _Plan, iterations: int, warm_adds: int | None = None) -> np.ndarray:
    """Bir adımın işlem sayısı ÜST SINIRI (OPS sırasıyla): sabit kısım + sıcak başlangıç + `iterations` QP yinelemesi.

    Kodun yapısından satır satır sayılmıştır (her döngü en kötü q ile); ölçülen her adımın bu sınırın altında
    kaldığı testte doğrulanır. iterations = MaxIterations → en kötü durum yürütme süresi (WCET) sınırı."""
    n, m = pl.nz, pl.nc
    nx, nxq, nmv, nmd, nmy, ny, nf, p = pl.nx, pl.nxQP, pl.nmv, pl.nmd, pl.nmy, pl.ny, pl.nfree, pl.p
    rr = p if pl.ref_preview else 1
    vr = (p + 1 if pl.md_preview else 1) if nmd else 0
    tot = np.zeros(len(OPS))
    # giriş denetimi (abs + cmp her eleman)
    tot += _ops(abs=nmy + rr * ny + vr * nmd, cmp=nmy + rr * ny + vr * nmd)
    # ölçek: y, v
    tot += _ops(mul=nmy + nmd, add=nmy + nmd)
    # e = y − Cm x − Dvm v ; xk = x + M e
    tot += _ops(mul=nmy * nx + nmy * nmd + nx * nmy, add=nmy * nx + nmy * nmd + nx * nmy)
    # f
    tot += _ops(mul=nf * nxq + nf * nmv, add=nf * nxq + nf * nmv)
    tot += _ops(mul=rr * ny * (1 + nf), add=rr * ny * (1 + nf))
    if nmd:
        tot += _ops(mul=vr * nmd * (1 + nf), add=vr * nmd * (1 + nf))
        if pl.md_preview and m:
            tot += _ops(mul=vr * nmd * m, add=vr * nmd * m)
    # u, x güncellemesi
    tot += _ops(add=3 * nmv, mul=nmv)
    tot += _ops(mul=nx * nx + nx * nmv + nx * nmd + nx * nmy, add=nx * nx + nx * nmv + nx * nmd + nx * nmy
                + (nx if pl.has_bv_off else 0))
    if pl.subopt:
        tot += _ops(add=3 * nmv, cmp=4 * nmv)
    if m == 0:
        return tot + _ops(mul=nmv * nf, add=nmv * nf)
    # b
    tot += _ops(neg=m, mul=m * (nxq + nmv + (nmd if not pl.md_preview else 0)),
                add=m * (nxq + nmv + (nmd if not pl.md_preview else 0)) + (m if pl.md_preview else 0))
    # QP başlangıcı: x = −Linvᵀ (Linv f)
    tri = n * (n + 1) // 2
    tot += _ops(mul=2 * tri, add=2 * tri, neg=n)

    def jt():
        return _ops(mul=n * n, add=n * n)

    def back():                                    # q ≤ n
        return _ops(mul=n * (n - 1) // 2, add=n * (n - 1) // 2, div=n)

    def norm(k):
        return _ops(mul=k, add=k, sqrt=1)

    def slack():
        return _ops(mul=n, add=n + 1)

    def finite():
        return _ops(abs=n, cmp=n)

    def hyp():                                     # olağan yol + ölçekli yedek (üst sınır)
        return _ops(mul=4, add=2, sqrt=2, cmp=5, abs=2, div=1)

    def rot(cols):                                 # bir Givens dönüşümü, cols sütun / satır uzunluğu
        return _ops(cmp=1, div=2, mul=4 * cols, add=2 * cols) + hyp()

    add_col = (n - 1) * rot(n)                     # q = 0 en kötü
    drop = (n - 1) * (rot(n) + _ops(mul=4 * n, add=2 * n))   # R satır güncellemesi ≤ 4n çarpma
    dirv = _ops(mul=n * n, add=n * n)
    scan = finite() + m * slack() + _ops(cmp=m) + _ops(cmp=1, neg=1)
    # bir iç adım (yineleme): her terim kendi en kötü q'suyla
    inner = (finite() + jt() + back() + norm(n) + norm(n) + _ops(abs=n, cmp=2 * n, mul=1)
             + _ops(cmp=2 * n, div=n) + _ops(mul=1, cmp=1) + slack() + _ops(neg=1, mul=1, div=1)
             + _ops(cmp=1) + dirv + _ops(mul=2 * n, add=2 * n + 1) + _ops(cmp=1))
    step_iter = inner + np.maximum(add_col, drop) + scan
    if pl.warm:
        recompute = (_ops(mul=n * (n - 1) // 2, add=n * (n - 1) // 2, div=n) + _ops(mul=n * n, add=n * n + n)
                     + _ops(mul=n * n, add=n * n + n) + _ops(mul=n * (n - 1) // 2, add=n * (n - 1) // 2, div=n)
                     + _ops(cmp=n))
        force = (jt() + norm(n) + norm(n) + _ops(cmp=2, mul=1) + back() + dirv + slack() + _ops(neg=1, mul=1, div=1)
                 + _ops(mul=2 * n, add=2 * n) + add_col)
        wa = n if warm_adds is None else min(int(warm_adds), n)
        tot += wa * force + recompute
        step_iter = np.maximum(step_iter, recompute + drop)
    tot += scan + inner                            # ilk tarama + sonlanan (sayılmayan) yarım adım
    tot += max(int(iterations), 0) * step_iter
    return tot


def _cost_scenario(mpcobj, d, T: int):
    """Varsayılan maliyet senaryosu: tüm çıkış referanslarına ±1 ölçek birimi basamak (yarıda işaret değiştirir),
    MD varsa T/4'te +1 ölçek birimi basamak. Kısıtlar bu senaryoda devreye girer / girmez — gerçek görevinizin
    referansıyla ölçmek için cost(..., r=, v=) verin."""
    t = np.arange(T)
    sgn = np.where(t < T // 2, 1.0, -1.0)[:, None]
    r = (d.yoff * d.Yscale)[None, :] + sgn * d.Yscale[None, :]
    v = None
    if d.nmd:
        v = (d.mdoff * d.Uscale[d.mdindex])[None, :] + np.where(t >= T // 4, 1.0, 0.0)[:, None] * \
            d.Uscale[d.mdindex][None, :]
    return r, v


def cost(mpcobj, opts: MpcCodegenOptions | None = None, *, T: int | None = None, r=None, v=None,
         options=None, scenario: str | None = None) -> list[MpcCostRow]:
    """Adım başına işlem sayısı (C++ sayaçla ölçülür; kapalı çevrim) + analitik en kötü durum → çevrim / µs /
    CPU yükü @ fs = 1/Ts (MCU tablosundan tahmin)."""
    opts = opts or MpcCodegenOptions()
    check_options(opts)
    pl = plan(mpcobj, opts)
    d = mpcobj.data
    prof = profile(opts.mcu)
    T = int(T or max(COST_MIN_STEPS, 4 * pl.p))
    label = scenario
    if r is None and v is None:
        r, v = _cost_scenario(mpcobj, d, T)
        label = label or "±1 ölçek birimi referans basamağı (yarıda işaret değiştirir)" + \
            (" + MD basamağı" if d.nmd else "")
    run = run_closed_loop(mpcobj, opts, T, r, v, options, count_ops=True)
    w = _op_weights(prof, opts.real)
    ops = run.ops
    cyc = ops @ w
    fs = 1.0 / pl.Ts
    it = np.where(run.status > 0, run.status, 0)
    iter_cyc = float((step_ops_bound(pl, 1) - step_ops_bound(pl, 0)) @ w) if pl.nc else 0.0
    wc = float(step_ops_bound(pl, pl.max_iter if pl.nc else 0) @ w)
    us = cyc / opts.clock_mhz
    budget_cyc = pl.Ts * opts.clock_mhz * 1e6
    base_cyc = float(step_ops_bound(pl, 0) @ w)
    budget = int(max(0.0, (budget_cyc - base_cyc) // iter_cyc)) if iter_cyc > 0 else 0
    counts = {"çözüldü": int(np.sum(run.status > 0)), "yineleme sınırı": int(np.sum(run.status == 0)),
              "çözümsüz": int(np.sum(run.status == -1)), "sayısal": int(np.sum(run.status == -2)),
              "geçersiz giriş": int(np.sum(run.status == -3))}
    soft = ", double yazılım kayan nokta" if opts.real == "double" and prof.double_is_soft() else ""
    method = (f"işlem sayısı ölçüldü (C++ sayaç, {T} adım kapalı çevrim: {label}); en kötü durum = analitik üst sınır "
              f"(MaxIterations = {pl.max_iter} yineleme, her yineleme en kötü aktif küme boyuyla); çevrim çekirdek "
              f"tablosundan tahmin ({prof.name}, {opts.real}{soft})")
    return [MpcCostRow(scenario=label, ops_mean={op: float(x) for op, x in zip(OPS, ops.mean(axis=0))},
                       cycles_mean=float(cyc.mean()), cycles_max=float(cyc.max()), cycles_wc=wc,
                       cycles_iter=iter_cyc, us_mean=float(us.mean()), us_max=float(us.max()),
                       us_wc=wc / opts.clock_mhz, load_mean=float(us.mean()) * fs / 1e4,
                       load_max=float(us.max()) * fs / 1e4, load_wc=wc / opts.clock_mhz * fs / 1e4,
                       iters_mean=float(it.mean()), iters_max=int(it.max()), max_iter=pl.max_iter if pl.nc else 0,
                       iter_budget=budget, status_counts=counts, method=method)]


# ============================================================================ ayak izi
_FP_CACHE: dict[str, dict] = {}
_FP_VERSION = 1


def _harness(pl: _Plan, baseline: bool) -> str:
    """Ölçüm koşumu: tüm genel API çağrılır, giriş / çıkış volatile (temel: aynı G/Ç, MPC çağrısı yok)."""
    md = "spt_in + SPT_NYM + SPT_NY" if pl.nmd else "NULL"
    calls = "" if baseline else f"""
        spt_out[SPT_NMV] = (SPT_REAL)spt_step(&st, ym, spt_in_nv + SPT_NYM, 1, {md.replace('spt_in', 'spt_in_nv')},
                                             {1 if pl.nmd else 0}, u);
        spt_get_state(&st, x);
        spt_get_last_mv(&st, u2);"""
    copy = "" if not baseline else """
        for (i = 0; i < SPT_NMV; ++i) u[i] = ym[i % SPT_NYM];
        x[0] = u2[0] = ym[0];"""
    init = "" if baseline else "    spt_init(&st);\n    spt_set_last_mv(&st, u);\n    spt_set_state(&st, x);\n"
    state = "" if baseline else "static spt_t st;\n"
    return f"""/* Ayak izi ölçüm koşumu ({'temel' if baseline else 'mpc'}). */
#include <stddef.h>
#include "spt.h"
{state}volatile SPT_REAL spt_in[SPT_NYM + SPT_NY + SPT_NMD + 1];
volatile SPT_REAL spt_out[SPT_NMV + SPT_NX + 2];
static SPT_REAL spt_in_nv[SPT_NYM + SPT_NY + SPT_NMD + 1];
int main(void);
int main(void)
{{
    SPT_REAL ym[SPT_NYM], u[SPT_NMV], u2[SPT_NMV], x[SPT_NX];
    int i;
    for (i = 0; i < SPT_NMV; ++i) u[i] = u2[i] = spt_in[0];
    for (i = 0; i < SPT_NX; ++i) x[i] = spt_in[0];
{init}    for (;;) {{
        for (i = 0; i < SPT_NYM + SPT_NY + SPT_NMD + 1; ++i) spt_in_nv[i] = spt_in[i];
        for (i = 0; i < SPT_NYM; ++i) ym[i] = spt_in_nv[i];{calls}{copy}
        for (i = 0; i < SPT_NMV; ++i) spt_out[i] = u[i] + u2[i];
        for (i = 0; i < SPT_NX; ++i) spt_out[SPT_NMV + 1 + i] = x[i];
    }}
}}
"""


def _measure(mpcobj, opts: MpcCodegenOptions, prof: McuProfile, compiler: str) -> dict | None:
    """Bir profil + derleyici için ölçüm; derleyici yoksa None."""
    tool = tc.cross_tool(compiler)
    if tool is None or tc.cross_flags(prof, compiler) is None:
        return None
    pl = plan(mpcobj, opts)
    src = _sources(pl, opts)
    files = {"spt.h": src["spt.h"], "spt.c": src["spt.c"],
             "spt_sizeof_probe.c": '#include "spt.h"\nunsigned char spt_sizeof_state[sizeof(spt_t)];\n'}
    real_def = f"SPT_USE_DOUBLE={1 if opts.real == 'double' else 0}"
    gcc = tc.find_arm_gcc()
    key_blob = json.dumps([_FP_VERSION, tc.STACK_ANALYSIS_VERSION, sorted(files.items()), prof.key, compiler, opts.opt,
                           real_def, tool.path, tool.version, gcc.version if gcc else None])
    key = hashlib.sha256(key_blob.encode()).hexdigest()[:24]
    cache_file = tc.WORK_ROOT / "footprint" / f"mpc_{key}.json"
    if key in _FP_CACHE:
        return _FP_CACHE[key]
    if cache_file.exists():
        try:
            _FP_CACHE[key] = json.loads(cache_file.read_text(encoding="utf-8"))
            return _FP_CACHE[key]
        except (OSError, ValueError):
            pass
    build = tc.compile_cross(files, ["spt.c", "spt_sizeof_probe.c"], prof, compiler, opts.opt, defines=[real_def],
                             warnings=())
    if not build.ok:
        raise RuntimeError(f"{compiler} {prof.key} derleme hatası:\n{build.log}")
    sz = build.sizes("spt.c")
    syms = tc.elf_symbols(build.objects["spt_sizeof_probe.c"])
    su = build.stack_usage()
    row = {"flash_code": sz.flash, "text": sz.text, "rodata": sz.rodata, "data": sz.data, "bss": sz.bss,
           "ram_state": syms["spt_sizeof_state"].size, "flash_with_libm": None, "stack": None, "stack_own": None,
           "stack_path": [], "su_step": su.get("spt_step"), "object": str(build.objects["spt.c"])}
    if gcc is not None and prof.gcc_flags is not None:
        base = tc.link_cross({"spt.h": files["spt.h"], "spt_fp_base.c": _harness(pl, True)}, ["spt_fp_base.c"], prof,
                             compiler, opts.opt, defines=[real_def])
        img = tc.link_cross({"spt.h": files["spt.h"], "spt.c": files["spt.c"], "spt_fp_harness.c": _harness(pl, False)},
                            ["spt_fp_harness.c", "spt.c"], prof, compiler, opts.opt, defines=[real_def])
        if base.ok and img.ok:
            row["flash_with_libm"] = tc.elf_sizes(img.output).flash - tc.elf_sizes(base.output).flash
            st = tc.stack_analysis(img.output, ["spt_step"])
            own = tc.stack_analysis(img.output, ["spt_step"], include=lambda nm: nm.startswith("spt_"))
            if "spt_step" in st:
                row["stack"] = st["spt_step"].bytes
                row["stack_path"] = list(st["spt_step"].path)
                row["stack_exact"] = st["spt_step"].exact
                row["frame_step"] = st["spt_step"].frame
                row["stack_own"] = own["spt_step"].bytes if "spt_step" in own else None
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(row), encoding="utf-8")
    _FP_CACHE[key] = row
    return row


# Derleyici bulunamazsa son çare: boyut formülü (bayt) — kod ≈ sabit + veri (sabitler tam bilinir).
_REF_CODE = {"none": 3600, "sp": 2600, "dp": 2600}
_REF_OPT = {"-O0": 2.2, "-Os": 1.0, "-O2": 1.25, "-O3": 1.6}


def _reference_row(pl: _Plan, opts: MpcCodegenOptions, prof: McuProfile, compiler: str, why: str) -> FootprintRow:
    r = 8 if opts.real == "double" else 4
    data = sum(a.values.size for a in pl.arrays) * r
    k = _REF_OPT[opts.opt] * prof.density * {"gcc": 1.0, "clang": 1.0, "iar": IAR_SIZE_FACTOR}[compiler]
    code = int(round(_REF_CODE[prof.fpu] * k * (1.6 if opts.real == "double" and prof.double_is_soft() else 1.0)))
    return FootprintRow(pl.name, compiler, code + data, None, state_size(pl, opts.real), int(round(300 * prof.density)),
                        False, f"tahmin: {why}; kod ≈ {code} B (çekirdek / iyileştirmeye göre ölçeklenmiş referans) + "
                               f"sabit veri {data} B (kesin); RAM durum = sizeof (yapı düzeninden kesin)")


def footprint(mpcobj, opts: MpcCodegenOptions | None = None, compilers: tuple[str, ...] | None = None,
              name: str | None = None) -> list[FootprintRow]:
    """Derleyici başına flash / RAM / yığın. Yerel derleyici varsa ölçülür, yoksa tahmin edilir."""
    opts = opts or MpcCodegenOptions()
    check_options(opts)
    prof = profile(opts.mcu)
    pl = plan(mpcobj, opts, name)
    rows: list[FootprintRow] = []
    for comp in tuple(compilers or (opts.compiler,)):
        if comp not in COMPILERS:
            raise ValueError(f"Bilinmeyen derleyici: {comp!r}")
        if comp == "iar" and prof.arch == "xtensa":
            continue
        base_prof = prof if prof.measurable else MCU_PROFILES[prof.estimate_base]
        src_comp = "gcc" if comp == "iar" else comp
        dm = _measure(mpcobj, opts, base_prof, src_comp) if base_prof.measurable else None
        if dm is None:
            why = f"{COMPILERS[src_comp]} bulunamadı" if base_prof.measurable else "derleyici yok"
            rows.append(_reference_row(pl, opts, prof, comp, why))
            continue
        tool = tc.cross_tool(src_comp)
        label = tool.label if tool else src_comp
        note = ""
        if dm.get("stack") is not None:
            libm = [f for f in dm.get("stack_path", []) if not f.startswith("spt_")]
            note = (f"; yığın: objdump statik analizi, en kötü yol {' → '.join(dm.get('stack_path', []))} (libm dahil "
                    f"{dm['stack']} B, yalnız bu kod {dm.get('stack_own')} B)" + (" — libm: " + ", ".join(libm)
                                                                                   if libm else ""))
            if dm.get("stack_exact") is False:
                note += " — çözülemeyen yığın ayarı var, alt sınır"
            note = re.sub(r"\bspt_", opts.prefix.lower() + "_", note)   # ölçüm iç adlarla: kullanıcı önekine çevir

        def sc(v, k):
            return None if v is None else int(round(v * k))

        if comp == "iar":
            rows.append(FootprintRow(pl.name, comp, int(round(dm["flash_code"] * IAR_SIZE_FACTOR * prof.density)),
                                     sc(dm["flash_with_libm"], IAR_LIBM_FACTOR * prof.density), dm["ram_state"],
                                     sc(dm["stack"], prof.density), False,
                                     f"tahmin: IAR yok — {label} {opts.opt} ({base_prof.name}) ölçümü × "
                                     f"{IAR_SIZE_FACTOR}" + (f" × {prof.density} kod yoğunluğu" if not prof.measurable
                                                              else "") + "; RAM durum = sizeof (kesin)",
                                     sc(dm.get("stack_own"), prof.density)))
        elif prof.measurable:
            rows.append(FootprintRow(pl.name, comp, dm["flash_code"], dm["flash_with_libm"], dm["ram_state"],
                                     dm["stack"], True,
                                     f"{label} {opts.opt}, {prof.name}, {opts.real}: ölçüldü (nesne .text {dm['text']} "
                                     f"+ .rodata {dm['rodata']} + .data {dm['data']} B; libm: newlib-nano ile "
                                     f"bağlanmış imaj − boş temel; RAM: sizeof(durum)){note}", dm.get("stack_own")))
        else:
            k = prof.density
            rows.append(FootprintRow(pl.name, comp, int(round(dm["flash_code"] * k)), sc(dm["flash_with_libm"], k),
                                     dm["ram_state"], sc(dm["stack"], k), False,
                                     f"tahmin: {prof.arch} derleyicisi yok — {label} {opts.opt} ({base_prof.name}) "
                                     f"ölçümü × {k} kod yoğunluğu; RAM durum = sizeof (ILP32 düzeni aynı, kesin)",
                                     sc(dm.get("stack_own"), k)))
    return rows


# ============================================================================ kullanım kılavuzu
def _fmt_int(v) -> str:
    if v is None:
        return "—"
    return f"{int(round(v)):,}".replace(",", " ") if math.isfinite(v) else "∞"


#: RAM yüzdesinin TEK tanımı (KULLANIM ve arayüz aynı sayıyı gösterir).
RAM_PERCENT_NOTE = "RAM % = durum yapısı + en kötü yığın (libm dahil; yığın bilinmiyorsa yalnız durum)."


def ram_bytes(row: FootprintRow) -> int:
    """RAM % hesabındaki bayt: durum yapısı (QP çalışma alanı dahil) + en kötü yığın (libm dahil, biliniyorsa)."""
    return int(row.ram_state) + int(row.stack or 0)


def _footprint_md(opts: MpcCodegenOptions, rows: list[FootprintRow]) -> str:
    lines = ["| Derleyici | Flash (kod + sabitler) | Flash (libm dahil) | RAM (durum) | Yığın (libm dahil / bu kod) "
             f"| Flash % ({opts.flash_kb:g} kB) | RAM % ({opts.ram_kb:g} kB) | Ölçüm |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        fl = r.flash_with_libm if r.flash_with_libm is not None else r.flash_code
        ram = ram_bytes(r)
        lines.append(f"| {COMPILERS[r.compiler]} | {_fmt_int(r.flash_code)} B | {_fmt_int(r.flash_with_libm)}"
                     f"{' B' if r.flash_with_libm is not None else ''} | {_fmt_int(r.ram_state)} B | "
                     f"{_fmt_int(r.stack)} / {_fmt_int(r.stack_own)} B | {100 * fl / (opts.flash_kb * 1024):.2f} | "
                     f"{100 * ram / (opts.ram_kb * 1024):.2f} | {'ölçüldü' if r.measured else 'tahmin'} |")
    methods = sorted({r.method for r in rows})
    return "\n".join(lines) + f"\n\n{RAM_PERCENT_NOTE}\n\n" + "\n".join(f"- {m}" for m in methods)


def _cost_md(pl: _Plan, opts: MpcCodegenOptions, rows: list[MpcCostRow]) -> str:
    fs = 1.0 / pl.Ts
    P = opts.prefix.upper()
    out = []
    for c in rows:
        table = [f"| | Çevrim | µs @ {opts.clock_mhz:g} MHz | CPU yükü @ {fs:g} Hz |", "|---|---:|---:|---:|"]
        for lbl, cyc, us, ld in (("Ortalama adım (ölçülen senaryo)", c.cycles_mean, c.us_mean, c.load_mean),
                                 ("En pahalı adım (ölçülen senaryo)", c.cycles_max, c.us_max, c.load_max),
                                 (f"**En kötü durum** ({c.max_iter} yineleme, analitik üst sınır)", c.cycles_wc,
                                  c.us_wc, c.load_wc)):
            table.append(f"| {lbl} | {_fmt_int(cyc)} | {us:.1f} | %{ld:.2f} |")
        table.append(f"| Bir QP yinelemesi (üst sınır) | {_fmt_int(c.cycles_iter)} | "
                     f"{c.cycles_iter / opts.clock_mhz:.2f} | — |")
        counts = ", ".join(f"{k} {v}" for k, v in c.status_counts.items())
        ops = ", ".join(f"{k} {v:.1f}" for k, v in c.ops_mean.items() if v)
        out.append("\n".join(table) + f"""

- Senaryo: {c.scenario}. Yineleme: ortalama {c.iters_mean:.2f}, en çok {c.iters_max} (sınır `{P}_MAX_ITER` =
  {c.max_iter}). Durumlar: {counts}.
- **Ts bütçesi:** {pl.Ts:g} s × {opts.clock_mhz:g} MHz = {_fmt_int(pl.Ts * opts.clock_mhz * 1e6)} çevrim → CPU'nun
  tamamı ayrılsa en çok **{c.iter_budget}** yineleme sığar.{_budget_note(c, P)}
- {c.method}.

Adım başına ortalama işlem sayıları (ölçülen): {ops}.""")
    return "\n\n".join(out)


def _budget_note(c: MpcCostRow, P: str) -> str:
    if c.iter_budget >= c.max_iter:
        return " En kötü durum bile bütçeye sığıyor."
    need = min(max(c.iters_max + 2, 1), max(c.iter_budget, 1))
    return (f" **En kötü durum Ts'yi aşıyor:** `{P}_MAX_ITER`'ı ölçülen en çok yinelemenin ({c.iters_max}) üstünde "
            f"ve bütçenin altında bir değere indirin (ör. `-D{P}_MAX_ITER={need}`); sınıra takılan adımda MV önceki "
            "değerinde tutulur (güvenli ama o adımda kontrol yok). Diğer çözümler: daha hızlı çekirdek / saat, daha "
            "kısa ufuk ya da daha az hamle bloğu, daha az kısıt (ör. çıkış kısıtlarını ufkun ilk adımlarıyla "
            "sınırlamak).")


def usage_markdown(mpcobj, opts: MpcCodegenOptions | None = None, name: str | None = None,
                   footprint_rows: list[FootprintRow] | None = None, cost_rows: list[MpcCostRow] | None = None,
                   pl: _Plan | None = None) -> str:
    """Türkçe kullanım kılavuzu (KULLANIM_<önek>.md)."""
    opts = opts or MpcCodegenOptions()
    check_options(opts)
    pl = pl or plan(mpcobj, opts, name)
    p, P = opts.prefix.lower(), opts.prefix.upper()
    prof = profile(opts.mcu)
    h, c, ex, _ = file_names(opts.prefix)
    fs = 1.0 / pl.Ts
    sc = pl.scales

    def sig_rows(kind, names, units, scale, nom, extra=None):
        out = []
        for i, n in enumerate(names):
            u = units[i] if units and units[i] else "—"
            out.append(f"| {kind} | {i} | {n} | {u} | {scale[i]:g} | {nom[i]:g} |" + (f" {extra[i]} |" if extra else
                                                                                      ""))
        return out

    sig = ["| Tür | İndis | Ad | Birim | ScaleFactor | Nominal |", "|---|---:|---|---|---:|---:|"]
    sig += sig_rows("MV (u)", pl.mv_names, pl.mv_units, sc["mv"], sc["mv_nom"])
    sig += sig_rows("MO (ym)", pl.mo_names, [pl.ov_units[i] for i in pl.myindex], [sc["ov"][i] for i in pl.myindex],
                    [sc["ov_nom"][i] for i in pl.myindex])
    sig += sig_rows("OV (ref)", pl.ov_names, pl.ov_units, sc["ov"], sc["ov_nom"])
    sig += sig_rows("MD (md)", pl.md_names, pl.md_units, sc["md"], sc["md_nom"])
    if pl.bounds.get("rows"):
        con = ["| Sinyal | Ad | Min | Max | Oran min | Oran max | Tür |", "|---|---|---:|---:|---:|---:|---|"]
        con += [f"| {k} | {n} | {a} | {b} | {c_} | {d_} | {t} |" for k, n, a, b, c_, d_, t in pl.bounds["rows"]]
        con_md = "\n".join(con) + (f"\n\nYumuşak kısıtlar tek bir gevşek değişkenle (ε ≥ 0) gevşer; ihlal cezası "
                                   f"ρ_ε = {pl.bounds['ecr']:g} (Weights.ECR). `s.slack` son adımın ε değeridir: 0'dan "
                                   "büyükse en az bir yumuşak kısıt ihlal edilmiştir (ör. çıkış sınırı aşılmak zorunda "
                                   "kaldı).")
    else:
        con_md = "Bu tasarımda kısıt yok: QP yerine kapalı biçim (Δu = −[H⁻¹]₁ f) kullanılır, yineleme yoktur."
    ref_rows = f"en çok {pl.p} satır (`ref_rows`)" if pl.ref_preview else "tek satır (`ref_rows = 1`)"
    md_rows = (f"en çok {pl.p + 1} satır" if pl.md_preview else "tek satır") if pl.nmd else "MD yok (`NULL, 0`)"
    fpu = {"none": "FPU yok: tüm kayan nokta yazılımda; float bile onlarca çevrim — MPC için en az Cortex-M4F önerilir",
           "sp": "tek duyarlık FPU: `float` önerilir; `double` yazılım kayan noktadır (10–100× yavaş)",
           "dp": "çift duyarlık FPU: `double` da donanımda (yine de `float` daha hızlı ve küçük)"}[prof.fpu]
    flags = " ".join(prof.gcc_flags) if prof.gcc_flags else "(üretici SDK'sının bayrakları)"
    fp_md = _footprint_md(opts, footprint_rows) if footprint_rows else \
        "_Bu dosya ölçümsüz üretildi; ayak izi tablosu için `footprint()` sonuçlarıyla yeniden üretin._"
    cost_md = _cost_md(pl, opts, cost_rows) if cost_rows else \
        "_Bu dosya ölçümsüz üretildi; maliyet tablosu için `cost()` sonuçlarıyla yeniden üretin._"
    real_note = ""
    if opts.real == "double" and prof.double_is_soft():
        real_note = (f"\n> **Uyarı:** `double` seçildi ama {prof.name} çekirdeğinde çift duyarlık donanımı yok; tüm "
                     "hesap yazılım kayan noktayla yapılır. Mümkünse `float` kullanın.\n")
    md_arg = "md, 1" if pl.nmd else "NULL, 0"
    ref_dim = f"{P}_P * {P}_NY" if pl.ref_preview else f"{P}_NY"
    md_decl = f"\n    {P}_REAL md[{P}_NMD];" if pl.nmd else ""
    md_read = "\n    bozucu_oku(md);" if pl.nmd else ""
    held = "kırpılmış alt-en iyi hamle" if pl.subopt else "önceki değer tutulur"
    api = [("`{p}_init(&s)`", "MATLAB `mpcstate` varsayılanı: bitki Nominal.X, bozucu / gürültü 0, son MV nominal"),
           ("`{p}_step(&s, ym, ref, ref_rows, md, md_rows, u)`", "Bir kontrol adımı (aşağıda); dönüş durum kodu"),
           ("`{p}_set_last_mv(&s, u)` / `{p}_get_last_mv(&s, u)`",
            "Son MV (el → otomatik darbesiz geçiş: eyleyicinin o anki değeri)"),
           ("`{p}_get_state(&s, x)` / `{p}_set_state(&s, x)`",
            "Kestirici durumu x̂(k\\|k−1) = [bitki (Nominal.X dahil); bozucu; gürültü]")]
    api_md = "\n".join(f"| {a.format(p=p)} | {b} |" for a, b in api)
    codes = [("> 0", "QP çözüldü; değer yineleme sayısı (MATLAB `info.Iterations`)", "u(k) = u(k−1) + Δu"),
             (f"`{P}_STATUS_MAXITER` (0)", f"`{P}_MAX_ITER` yinelemede bitmedi (MATLAB 'unreliable')", held),
             (f"`{P}_STATUS_INFEASIBLE` (−1)", "Sert kısıtlar birlikte sağlanamıyor (ör. oran sınırı MV sınırına "
              "yetişemiyor)", "önceki değer tutulur"),
             (f"`{P}_STATUS_NUMERICAL` (−2)", "NaN / sonsuz (bozuk veri, aşırı kötü koşul)", "önceki değer tutulur"),
             (f"`{P}_STATUS_BAD_INPUT` (−3)", "`ym` / `ref` / `md` NaN / sonsuz ya da satır sayısı geçersiz",
              "önceki değer; durum değişmez")]
    codes_md = "\n".join(f"| {a} | {b} | {c_} |" for a, b, c_ in codes)
    pv = ("açık" if pl.ref_preview else "kapalı", "açık" if pl.md_preview else "kapalı")
    if pl.nc:
        warm = ("Çözücü önceki adımın aktif kümesinden başlar (`s.iA`): kararlı durumda çoğu adım **1** yinelemede "
                "biter." if pl.warm else "Sıcak başlangıç kapalı (Optimizer.UseWarmStart): her adım boş kümeden "
                "başlar.")
        wc_md = f"""- {warm}
- En kötü durum süresi `{P}_MAX_ITER` ile sınırlıdır (varsayılan MATLAB'in `4·max(30, n_z + n_c)` = {pl.max_iter}).
  Gerçek zamanlı garanti için en kötü durum çevrimi Ts bütçesine sığmalıdır; sığmıyorsa `{P}_MAX_ITER`'ı ölçülen
  en çok yinelemenin (aşağıdaki tablo; sahada `s.iter_max`) biraz üstüne indirin. Sınıra takılan adımda MV
  {"kırpılmış alt-en iyi hamleyle" if pl.subopt else "önceki değerinde"} kalır."""
    else:
        wc_md = ("Kısıt olmadığından QP ve yineleme yoktur: her adımın süresi sabittir (en kötü durum = ortalama).")
    qp_line = (f"QP: {pl.nz} değişken ({pl.nfree} serbest hamle + ε), {pl.nc} kısıt satırı, en çok {pl.max_iter} "
               f"yineleme, sıcak başlangıç {'açık' if pl.warm else 'kapalı'}, alt-en iyi çözüm "
               f"{'açık (kırpılır)' if pl.subopt else 'kapalı (MV tutulur)'}") if pl.nc else "QP yok (kısıtsız)"
    return f"""# MPC denetleyicisi „{pl.name}" — C kodu kullanım kılavuzu

Signal Processing Toolbox tarafından üretildi. Python referansı: `spt/control/mpc` (MATLAB R2025b MPC Toolbox ile
matris matris ve benzetimde ~1e-13 doğrulanmış). Üretilen C, MATLAB'in `mpcmoveCodeGeneration` ile ürettiği kodun
karşılığıdır: aynı kestirici, aynı QP formülasyonu, aynı aktif küme çözücüsü (Goldfarb–Idnani), aynı durum kodları.

**Denetleyici:** Ts = {pl.Ts:g} s (**{fs:g} Hz**), tahmin ufku p = {pl.p}, hamle blokları {list(pl.moves)} ·
MV {pl.nmv}, MD {pl.nmd}, MO {pl.nmy}, UO {len(pl.uyindex)} · kestirici {pl.nx} durum · {qp_line}.

**Seçimler:** sayı türü `{opts.real}` · MCU {prof.name} ({prof.examples}) · derleyici {COMPILERS[opts.compiler]}
`{opts.opt}` · saat {opts.clock_mhz:g} MHz · referans önizlemesi {pv[0]} · MD önizlemesi {pv[1]}
{real_note}
## Dosyalar

| Dosya | İçerik |
|---|---|
| `{h}` | Boyutlar (`#define`), sayı türü `{P}_REAL`, durum yapısı `{p}_t`, API |
| `{c}` | Sabit matrisler (`static const`, flash) ve çevrimiçi adım + QP çözücüsü |
| `{ex}` | Sabit periyotla çağıran örnek görev (sensör / eyleyici fonksiyonları kullanıcıya ait) |

`malloc` yok, global değişken yok, özyineleme yok; tüm döngülerin üst sınırı derleme anında sabit. Tüm durum
`{p}_t` yapısındadır: fonksiyonlar yeniden girişlidir, aynı koddan birden çok denetleyici örneği çalıştırılabilir.
Birden çok farklı MPC (ör. irtifa + yatay konum) için her birini farklı önekle üretin.

## Sinyaller ve birimler

Tüm API **mühendislik birimiyle** çalışır; ölçek çarpanları (ScaleFactor) ve nominal değerler (Model.Nominal) içeride
uygulanır: `y_boyutsuz = ym / ScaleFactor − Nominal / ScaleFactor`. Dizilerin sırası aşağıdaki indislerdir.

{chr(10).join(sig)}

## Hızlı başlangıç

```c
#include "{h}"

static {p}_t mpc;                                  /* durum: statik (yığında yer kaplamaz) */

void baslat(void) {{ {p}_init(&mpc); }}

/* Donanım zamanlayıcısıyla TAM {pl.Ts:g} s'de bir (= {fs:g} Hz) çağırın */
void kontrol_periyodu(void)
{{
    {P}_REAL ym[{P}_NYM], ref[{ref_dim}], u[{P}_NMV];{md_decl}
    olcum_oku(ym);                                 /* ölçülen çıkışlar */
    referans_al(ref);                              /* hedef */{md_read}
    int durum = {p}_step(&mpc, ym, ref, {pl.p if pl.ref_preview else 1}, {md_arg}, u);
    eyleyici_yaz(u);                               /* durum ne olursa olsun u geçerlidir */
    (void)durum;                                   /* > 0: çözüldü (yineleme), ≤ 0: MV tutuldu */
}}
```

## API

| Fonksiyon | Açıklama |
|---|---|
{api_md}

`ref`: {ref_rows}; satır i → y(k+1+i) (önizleme: eksik satırlar son satırla doldurulur, MATLAB kuralı); `NULL` →
nominal çıkış. `md`: {md_rows}; satır i → v(k+i). `ym = NULL` → nominal (yalnız test için).

**Durum kodları** (`s.status`, dönüş değeri):

| Kod | Anlam | MV |
|---|---|---|
{codes_md}

Sayaçlar: `s.steps`, `s.n_ok`, `s.n_maxiter`, `s.n_infeasible`, `s.n_numerical`, `s.n_bad_input`, `s.iter_max`
(görülen en büyük yineleme — sahada bütçe izleme için). MV tutulduğunda kestirici yine güncellenir (uygulanan MV ile),
sonraki adım normal devam eder (MATLAB davranışı). Art arda çok sayıda 0 / −1 bir tasarım ya da sensör sorununa işaret
eder: üst katmanda güvenli kipe geçin.

## Projeye ekleme

### STM32CubeIDE (GCC)
1. `{h}` → `Core/Inc`, `{c}` → `Core/Src` (ya da ayrı klasör + *Paths and Symbols*).
2. *MCU Settings*: FPU ve ABI. Bu çekirdek için: `{flags}`.
3. *MCU GCC Linker → Libraries*: `m` (`-lm`; yalnız `sqrtf` / `fabsf` kullanılır).
4. **`-ffast-math` kullanmayın** (NaN / sonsuz denetimleri kaldırılır). `-O2` ya da `-Os` önerilir; `-O0` 2–4×
   yavaştır.
5. İsteğe bağlı komut satırı makroları: `{P}_USE_DOUBLE=0/1`, `{P}_MAX_ITER=n`, `{P}_TOL=…`.

### Keil MDK (Arm Compiler 6) / IAR EWARM
Dosyaları gruba ekleyin, C99; *Floating Point Hardware* = Single (M4F / M33F) ya da Double (M7). Fast-math kapalı.
AC6 clang tabanlıdır (boyutlar yerelde clang ile ölçüldü); IAR boyutları GCC ölçümünden **tahmindir**.

### C++ projeleri (Arduino, ESP-IDF, Zephyr C++)
Başlık `extern "C"` korumalıdır; `.c` dosyası C++ derleyicisiyle de uyarısız derlenir.

## Çağırma hızı ve zamanlama

- `{p}_step()` **tam Ts = {pl.Ts:g} s** periyotla çağrılmalıdır: tahmin modeli, kestirici ve bozucu tümlevleri bu
  periyotla ayrıklaştırılmıştır. Farklı periyot kararlılığı bozar — Ts değişecekse MPC'yi yeni Ts ile yeniden
  tasarlayıp kodu yeniden üretin.
- Bir donanım zamanlayıcısı {fs:g} Hz'de kesme üretsin; kesmede yalnız bayrak kurup hesabı yüksek öncelikli bir
  görevde (`vTaskDelayUntil`) yapın. Ölçüm her periyotta aynı anda alınmalı; MV hesaplanır hesaplanmaz uygulanmalı
  (hesap süresi kadar gecikme MATLAB modelinde yoktur — süre Ts'nin küçük bir kesri olmalı).
- Süreyi sahada ölçün: `{p}_step()` çağrısını iki `DWT->CYCCNT` okumasıyla çevreleyin (bkz. `{ex}`).
  Aşağıdaki çevrimler tahmindir (gerçek değer flash bekleme durumuna / önbelleğe bağlı).

## Ölçekleme ve nominal değerler

İç model boyutsuzdur (MATLAB gibi): sinyaller kendi ScaleFactor'larıyla bölünür. ScaleFactor'ları sinyallerin
çalışma aralığına yakın seçin (ör. ±4 m/s² ivme için 4): QP'nin koşulu ve **float doğruluğu** bunlara bağlıdır.
Nominal (çalışma) noktası farklı olan bitkilerde (ör. askıda itki) mühendislik birimindeki değerleri doğrudan verin;
nominal değerler içeride çıkarılır / eklenir.

## Kısıtlar

{con_md}

## Sıcak başlangıç, yineleme sınırı ve en kötü durum süresi

{wc_md}

## Ölçülen ayak izi

{fp_md}

## Hesap maliyeti

{cost_md}

Maliyet modeli: çevrim = Σ temel işlem × (çekirdek çevrimi + bellek / dal ek yükü). İşlem sayıları **ölçülmüştür**
(üretilen C, ana makinede C++ sayaç türüyle derlenip kapalı çevrimde çalıştırılır); en kötü durum, kodun yapısından
türetilmiş **analitik üst sınırdır** (her ölçülen adımın altında kaldığı testte doğrulanır); çekirdek çevrim
tabloları **tahmindir** (`spt/codegen/mcu.py`). CPU yükü = µs/adım × fs / 10⁴ (%).

## Sayı türü ve FPU

{prof.name}: {fpu}. `{P}_USE_DOUBLE` ile seçilir. Sabitler `{P}_C()` ile doğru türde yazılmıştır (float modunda
istemsiz double yükseltmesi yok; `-Wdouble-promotion` temiz). Float'ta QP'nin doğrusal bağımlılık eşiği
`{P}_DEP_TOL = 1e3·FLT_EPSILON`'dur (double: 1e6·DBL_EPSILON, Python / MATLAB ile aynı). Kısıt toleransı
`{P}_TOL` = {pl.tol:g} boyutsuz birimdir; float'ta çok küçük ölçek çarpanlarıyla birlikte yuvarlama düzeyine
yaklaşırsa büyütün.

## Python / MATLAB referansından farklar

- Yuvarlama düzeyinde: ölçek çarpanına bölme yerine tersle çarpma; sabit terimler (ofset, MV hedefi; önizlemesizde
  referans / MD toplamları) üretimde önceden toplanır; Givens yarıçapı √(a² + b²). Double'da Python ile fark ~1e-13,
  yineleme sayıları ve durum kodları aynıdır (testlerle doğrulandı).
- Geçersiz giriş (NaN / sonsuz) Python'da hata ya da NaN yayılımıdır; burada adım yapılmaz (−3), MV ve durum korunur.
- MATLAB KWIK çözümsüz MPC QP'sinde bazen −2 döndürür, bu çözücü −1 (ikisinde de MV tutulur); yineleme sayıları
  KWIK'inkinden farklı olabilir (çözüm aynıdır).
- Çevrimiçi ağırlık / sınır değişimi, değişken ufuk, özel kestirici desteklenmez (Python MPC ile aynı sınırlar).
"""
