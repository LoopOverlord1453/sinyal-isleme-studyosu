"""Genel (senaryo tabanlı) füzyon kestiricileri için C99 kod üreteci, ayak izi (flash / RAM / yığın) ve hesap maliyeti.

Kapsam (`SUPPORTED`): `spt.dsp.fusion_generic` senaryolarından irtifa (altitude: complementary, kf, ekf, ukf, pf),
2B konum (position2d: dead_reckoning, ekf, ukf, pf) ve doğrusal çok sensörlü Kalman (linear_multi: weighted_mean, kf,
pf). Yönelim senaryosu `spt.codegen.fusion_c` ile üretilir (aynı kurallar, aynı araç zinciri).

Üretilen C, `spt/dsp/fusion_scenarios/*.py` Python kestiricilerinin birebir karşılığıdır: aynı denklemler; aynı senaryo
(Param / Choice) ve algoritma parametreleri (anlam, birim, varsayılan); aynı başlatma (irtifa: ilk geçerli baro
ölçümü; 2B konum: GNSS izi d_init'i aşınca kiriş başlığı; doğrusal: ilk konum ölçümü ya da göreli konum); NaN = "bu
örnekte ölçüm yok" kuralı ve çok hızlı (seyrek sensörlü) akış; otomatik p0 (ilk basınç) ve enlem / boylamın açıldığı
ilk GNSS düzeltmesi; eksik ivme / jiroskopta son değerin tutulup gürültünün büyütülmesi; çıktılar `Scenario.outputs`
sırasında (açılar rad). Üretilen API yalnız eşlenen girişleri (`GenericCodegenOptions.inputs`) alır.

Python'dan bilinçli farklar (çift hassasiyette Python ile en büyük fark < 1e-6 olarak testlerde doğrulanır):
  - Yuvarlama düzeyinde: bazı bölmeler tersle çarpma; simetrik kovaryansların üst üçgeni + ayna; 2B EKF'de F·P·Fᵀ
    seyrek biçimde; tamsayı kuvvetler çarpımla; UKF ağırlıklı toplamları ortak çarpanla.
  - Parçacık filtreleri: rastgele sayı üreteci farklı (xorshift32 + Marsaglia kutupsal), yeniden örneklemede parçacık
    sırası farklı (yerinde kopyalama), ucuz olabilirliklerde ağırlık güncellemesi log yerine iki geçişli çarpım,
    doğrusal PF'nin süreç gürültüsü kökü kapalı biçimde (q·D·chol(C)) — sonuçlar istatistiksel olarak eşdeğerdir.
  - 2B konum: başlatma için GNSS geçmişi sabit boyutlu halka tampon (SPT_POS_FIX_HISTORY = 160 düzeltme, en az
    30 s / 159 aralıkla: her GNSS hızında 30 s; Python aynı kuralı kullanır); enlem / boylam her zaman double alınır
    (float32 enlem ~0.5 m çözünürlük verir).
  - UKF'lerde Cholesky başarısızsa `chol_psd` özdeğer yedeği Jacobi ile (aynı sigma kümesi); 2×2 yenilik tekilse
    güncelleme atlanır (Python LinAlgError fırlatırdı).
  - Python'un FusionError verdiği parametre hataları (p0 ∈ (0, 1000) Pa; yuva türü > model derecesi) C'de init() /
    reset() dönüşüyle (0) bildirilir; step() o durumda hep 0 döner.

Ayak izi (`footprint`): yerel derleyici varsa ÖLÇÜLÜR (nesne bölümleri, sizeof, libm dahil bağlanmış imaj farkı,
disassembly'den en kötü yığın); yoksa ölçülmüş bir referanstan TAHMİN edilir. Maliyet (`cost`): adım başına işlem
sayıları ana makinede C++ sayaç türüyle ÖLÇÜLÜR (`fusion_c` ile aynı düzenek; pow = exp + log + mul sayılır), çevrimler
MCU profilindeki tablodan TAHMİN edilir.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass

import numpy as np

from spt.codegen import toolchain as tc
from spt.codegen.fusion_c import (CostRow, FootprintRow, GeneratedFile, _c_float, _count_prelude, _fill, _fmt_int,
                                  _op_weights, _prefixed, write_files)
from spt.codegen.mcu import (COMPILERS, IAR_LIBM_FACTOR, IAR_SIZE_FACTOR, MCU_PROFILES, OPS, OPT_LEVELS, McuProfile,
                             profile)
from spt.dsp import fusion_generic as fg
from spt.dsp.fusion_scenarios import linear_multi as lin_mod
from spt.dsp.fusion_scenarios import position2d as pos_mod
from spt.dsp.fusion_scenarios.base import Choice, check_input_keys, input_arrays, resolve_values

__all__ = ["CostRow", "FootprintRow", "GeneratedFile", "GenericCodegenOptions", "HostRun", "SUPPORTED", "cost",
           "footprint", "generate", "host_program", "run_generated", "state_size", "usage_markdown", "usage_names",
           "write_files"]

SUPPORTED: dict[str, tuple[str, ...]] = {
    "altitude": ("complementary", "kf", "ekf", "ukf", "pf"),
    "position2d": ("dead_reckoning", "ekf", "ukf", "pf"),
    "linear_multi": ("weighted_mean", "kf", "pf"),
}
DETERMINISTIC: dict[str, tuple[str, ...]] = {s: tuple(a for a in algos if a != "pf") for s, algos in SUPPORTED.items()}
GEN_ROOT = tc.WORK_ROOT / "generic"   # SPT_CODEGEN_DIR'e uyar (derleme önbelleğiyle aynı yerde)
COST_SAMPLES = 4000          # maliyet ölçümünde simüle edilen adım sayısı
COST_SEED = 7
FIX_HISTORY = pos_mod.INIT_HISTORY   # 2B konum başlatma geçmişi (en az 30 s / 159 aralıklı düzeltmeler)

_SAB = {"altitude": "alt", "position2d": "pos", "linear_multi": "lin"}
_PYMOD = {"altitude": "altitude", "position2d": "position2d", "linear_multi": "linear_multi"}
_CHOICE_INDEX = {("altitude", "complementary", "bias"): {"off": 0, "on": 1},
                 ("linear_multi", None, "order"): {"seviye": 0, "hiz": 1, "ivme": 2},
                 ("linear_multi", None, "kind"): {"konum": 0, "hiz": 1, "ivme": 2}}


# ============================================================================ seçenekler ve sonuç türleri
@dataclass(frozen=True)
class GenericCodegenOptions:
    scenario: str                              # "altitude" | "position2d" | "linear_multi"
    algorithms: tuple[str, ...]
    scenario_values: dict                      # Param / Choice değerleri (eksik → varsayılan); "_" önekliler yok sayılır
    params: dict                               # algoritma başına değerler {algo: {anahtar: değer}}
    inputs: tuple[str, ...]                    # eşlenen giriş anahtarları; üretilen API tam olarak bunları alır
    real: str = "float"                        # "float" | "double"
    mcu: str = "cortex-m4f"
    compiler: str = "gcc"
    opt: str = "-Os"
    clock_mhz: float = 168.0
    fs_hz: float = 100.0
    flash_kb: float = 1024.0
    ram_kb: float = 192.0
    prefix: str = "spt"


@dataclass(frozen=True)
class HostRun:
    out: np.ndarray                  # (N, len(outputs)); kestirim yoksa NaN (Python ile aynı)
    ops: np.ndarray | None = None    # (N, len(OPS)) adım başına işlem sayıları (count_ops)


@dataclass(frozen=True)
class _Resolved:
    scenario: str
    sab: str
    algorithms: tuple[str, ...]      # SUPPORTED sırasında
    values: dict                     # senaryo değerleri (doğrulanmış, varsayılanlarla tam)
    params: dict                     # {algo: doğrulanmış değerler}
    inputs: tuple[str, ...]          # Scenario.inputs sırasında


def _scen(key: str):
    if key not in SUPPORTED:
        raise ValueError(f"Desteklenmeyen senaryo: {key!r} (seçenekler: {', '.join(SUPPORTED)}; yönelim için "
                         "spt.codegen.fusion_c)")
    return fg.SCENARIOS[key]


def _resolve(opts: GenericCodegenOptions) -> _Resolved:
    """Seçenekleri doğrular (Python kestiricisiyle aynı kurallar; hata → ValueError / FusionError)."""
    scen = _scen(opts.scenario)
    if not opts.algorithms:
        raise ValueError("En az bir algoritma seçilmeli")
    for a in opts.algorithms:
        if a not in SUPPORTED[opts.scenario]:
            raise ValueError(f"{scen.name}: C karşılığı olmayan algoritma {a!r} "
                             f"(seçenekler: {', '.join(SUPPORTED[opts.scenario])})")
    for a in opts.params:
        if a not in SUPPORTED[opts.scenario]:
            raise ValueError(f"Bilinmeyen algoritma parametresi: {a!r}")
    if opts.real not in ("float", "double"):
        raise ValueError("real 'float' ya da 'double' olmalı")
    profile(opts.mcu)
    if opts.compiler not in COMPILERS:
        raise ValueError(f"Bilinmeyen derleyici: {opts.compiler!r}")
    if opts.opt not in OPT_LEVELS:
        raise ValueError(f"İyileştirme düzeyi {OPT_LEVELS} içinden olmalı")
    if not re.fullmatch(r"[A-Za-z](?:_?[A-Za-z0-9])*", opts.prefix):  # saklı tanımlayıcı (_X, __) üretmez
        raise ValueError("Önek harfle başlamalı; harf, rakam ve tek alt çizgi içerebilir (başta / sonda ya da çift alt "
                         "çizgi C'de saklı adlar üretir: _x, a__b)")
    if not (opts.clock_mhz > 0 and opts.fs_hz > 0):
        raise ValueError("clock_mhz ve fs_hz pozitif olmalı")
    keys = tuple(opts.inputs)
    if len(set(keys)) != len(keys):
        raise ValueError("Girişler tekrarsız olmalı")
    check_input_keys(scen.inputs, scen.one_of, keys)
    values = resolve_values(scen.params, opts.scenario_values, scen.name)
    algos = tuple(a for a in SUPPORTED[opts.scenario] if a in set(opts.algorithms))
    params = {}
    for a in SUPPORTED[opts.scenario]:
        if a in algos or a in opts.params:
            est = fg.make_estimator(opts.scenario, a, opts.scenario_values, opts.params.get(a) or {})
            est._check_inputs(frozenset(keys))          # ör. yuva türü ↔ model derecesi, p0 kuralı (_setup)
            params[a] = dict(est.values)
    inputs = tuple(i.key for i in scen.inputs if i.key in set(keys))
    return _Resolved(opts.scenario, _SAB[opts.scenario], algos, values, params, inputs)


def check_options(opts: GenericCodegenOptions) -> None:
    _resolve(opts)


def _particles(r: _Resolved) -> int:
    return int(r.params.get("pf", {}).get("particles", 500))


# ============================================================================ C yapıları (tek kaynak: alan listeleri)
# Alan: (tür, ad, boyutlar, açıklama). Tür: "real" | "f64" | "u32" | "int" | "u16" | iç içe yapı adı ("spt_alt_cfg").
# Boyut: tamsayı ya da makro ifadesi (sizeof hesabında `macros` ile değerlendirilir). Yapı C metni ve analitik
# sizeof aynı listeden üretilir (bkz. state_size; testlerde ölçülen sizeof ile karşılaştırılır).
_CTYPES = {"real": "SPT_REAL", "f64": "double", "u32": "uint32_t", "int": "int", "u16": "uint16_t"}


def _struct_c(name: str, brief: str, fields: list) -> str:
    lines = [f"/** {brief} */", f"typedef struct {name} {{"]
    for ctype, fname, dims, comment in fields:
        ct = _CTYPES.get(ctype, ctype + "_t")
        decl = f"    {ct} {fname}{''.join(f'[{d}]' for d in dims)};"
        lines.append(f"{decl:<52}/**< {comment} */" if len(decl) < 52 else f"{decl} /**< {comment} */")
    lines.append(f"}} {name}_t;")
    return "\n".join(lines) + "\n"


def _sizeof(fields: list, real: str, macros: dict, registry: dict) -> tuple[int, int]:
    off, align = 0, 1
    for ctype, _name, dims, _c in fields:
        if ctype == "real":
            s = a = 8 if real == "double" else 4
        elif ctype == "f64":
            s = a = 8
        elif ctype in ("u32", "int"):
            s = a = 4
        elif ctype == "u16":
            s = a = 2
        else:
            s, a = _sizeof(registry[ctype], real, macros, registry)
        count = 1
        for d in dims:
            count *= d if isinstance(d, int) else int(eval(d, {"__builtins__": {}}, dict(macros)))
        off = -(-off // a) * a + s * count
        align = max(align, a)
    return -(-off // align) * align, align


def _cfg_fields(r: _Resolved) -> list:
    scen = fg.SCENARIOS[r.scenario]
    if r.scenario == "linear_multi":
        return [("int", "order", (), "model derecesi: 0 sabit seviye, 1 sabit hız, 2 sabit ivme"),
                ("real", "sigma_q", (), "süreç gürültüsü σ_q (en yüksek türevin √PSD'si)"),
                ("real", "sigma0", (), "başlangıçta ölçülmeyen türevlerin σ'sı"),
                ("int", "kind", (6,), "yuva türleri s1…s6: 0 konum, 1 hız, 2 ivme"),
                ("real", "sigma", (6,), "yuva ölçüm gürültüleri σ s1…s6 (giriş biriminde)")]
    return [("real", p.key, (), p.label) for p in scen.params]


def _core_fields(r: _Resolved) -> list:
    if r.scenario == "altitude":
        return [("real", "out", (2,), "çıktılar [irtifa (m), dikey hız (m/s)]; kestirim yoksa NaN"),
                ("real", "p0", (), "kullanılan referans basınç (Pa); otomatikte ilk geçerli basınç"),
                ("real", "f_last", (), "son geçerli ivme (m/s²) — eksik örneklerde tutulur"),
                ("u32", "k", (), "örnek sayacı"),
                ("u32", "n_baro", (), "kullanılan baro ölçümü sayısı (tanılama)"),
                ("int", "p0_ok", (), "p0 belirlendi mi"),
                ("int", "f_ok", (), "f_last geçerli mi (hiç ivme gelmediyse 0)"),
                ("int", "ready", (), "1: filtre başlatıldı"),
                ("int", "ok", (), "1: parametreler geçerli")]
    if r.scenario == "position2d":
        h = "SPT_POS_FIX_HISTORY"
        return [("f64", "lat0", (), "enlem / boylam orijini (rad, ilk geçerli düzeltme)"),
                ("f64", "lon0", (), "orijin boylamı (rad)"),
                ("real", "out", (4,), "çıktılar [x (m), y (m), başlık ψ (rad), hız (m/s)]; yoksa NaN"),
                ("real", "theta", (), "başlatma öncesi ham jiroskop integrali (rad)"),
                ("real", "last_fix", (2,), "son GNSS düzeltmesi (m) — başlatma öncesi çıktı"),
                ("real", "w_last", (), "son geçerli jiroskop (rad/s)"),
                ("real", "ll_kx", (), "enlem/boylam → x çarpanı N(φ0)·cos φ0 (m/rad)"),
                ("real", "ll_ky", (), "enlem → y çarpanı M(φ0) (m/rad)"),
                ("real", "fx", (h,), "başlatma geçmişi: düzeltme x (m)"),
                ("real", "fy", (h,), "başlatma geçmişi: düzeltme y (m)"),
                ("real", "fth", (h,), "başlatma geçmişi: o andaki jiroskop integrali (rad)"),
                ("u32", "fk", (h,), "başlatma geçmişi: örnek no"),
                ("u32", "k", (), "örnek sayacı"),
                ("u32", "n_gnss", (), "GNSS düzeltmesi sayısı (tanılama)"),
                ("u32", "init_k", (), "başlatma örneği (tanılama)"),
                ("int", "head", (), "halka tampon başı"),
                ("int", "nfix", (), "tampondaki düzeltme sayısı"),
                ("int", "has_last_fix", (), "last_fix geçerli mi"),
                ("int", "has_w", (), "w_last geçerli mi"),
                ("int", "has_origin", (), "lat0 / lon0 geçerli mi"),
                ("int", "ready", (), "1: filtre başlatıldı"),
                ("int", "ok", (), "1: parametreler geçerli")]
    return [("real", "out", (3,), "çıktılar [konum, hız, ivme]; kestirim yoksa NaN"),
            ("u32", "n_meas", (6,), "yuva başına kullanılan ölçüm sayısı (tanılama)"),
            ("int", "d", (), "durum boyutu = derece + 1"),
            ("int", "has_pos", (), "eşlenmiş konum türü yuva var mı"),
            ("int", "ready", (), "1: filtre başlatıldı"),
            ("int", "ok", (), "1: parametreler geçerli")]


_RNG_FIELDS = [("u32", "s", (), "xorshift32 durumu (sıfır olamaz)"),
               ("int", "has_spare", (), "spare geçerli mi"),
               ("real", "spare", (), "kutupsal yöntemin sakladığı ikinci normal sayı")]


def _algo_fields(r: _Resolved, algo: str) -> list:
    s = r.sab
    cfg = (f"spt_{s}_cfg", "cfg", (), "senaryo parametreleri (çalışma anında değiştirilebilir)")
    core = (f"spt_{s}_core", "c", (), "senaryo durumu ve çıktılar")
    if algo == "pf":
        n = f"SPT_{s.upper()}_PF_PARTICLES"
        dim = {"alt": 3, "pos": 6, "lin": "SPT_LIN_PF_DMAX"}[s]
        work = {"alt": 27, "pos": 108, "lin": "3 * SPT_LIN_PF_DMAX * SPT_LIN_PF_DMAX"}[s]
        fields = [cfg, ("u32", "seed", (), "tohum (reset ile etkinleşir)"), core,
                  ("spt_rng", "rng", (), "rastgele sayı üreteci"),
                  ("real", "ess", (), "etkin parçacık sayısı (son ölçüm)")]
        if s == "pos":
            fields.append(("real", "heading", (), "son çıktı başlığı (dairesel ortalamanın referansı, rad)"))
            fields.append(("u32", "since_fix", (), "konum yürüyüşü birikmiş örnek sayısı"))
        fields += [("u32", "resamples", (), "yeniden örnekleme sayacı (tanılama)"),
                   ("real", "w", (n,), "normalize ağırlıklar"),
                   ("real", "X", (n, dim), "parçacıklar"),
                   ("real", "work", (work,), "yeniden örnekleme çalışma alanı (kovaryans, kök)"),
                   ("u16", "cnt", (n,), "yeniden örneklemede kopya sayıları (geçici)")]
        return fields
    if r.scenario == "altitude":
        if algo == "complementary":
            return [cfg, ("real", "tau", (), "geçiş zaman sabiti τ (s)"),
                    ("int", "bias", (), "ivme sapması kestirimi: 1 açık (3. derece), 0 kapalı"), core,
                    ("real", "h", (), "irtifa (m)"), ("real", "v", (), "dikey hız (m/s)"),
                    ("real", "b", (), "ivme sapması (m/s²)"), ("u32", "k_last", (), "son baro ölçümünün örneği")]
        return [cfg, core, ("real", "x", (3,), "durum [h (m), v (m/s), b (m/s²)]"),
                ("real", "P", (6,), "kovaryans [p00 p01 p02 p11 p12 p22] (simetrik)")]
    if r.scenario == "position2d":
        if algo == "dead_reckoning":
            return [cfg, core, ("real", "x", (4,), "durum [x, y, ψ, v]")]
        fields = [cfg, core, ("real", "x", (6,), "durum [x (m), y (m), ψ (rad), v (m/s), b_g (rad/s), s]"),
                  ("real", "P", (6, 6), "kovaryans (her zaman tam simetrik)")]
        if algo == "ukf":
            fields += [("real", "L", (6, 6), "√(6·P) (sütunlar sigma yönleri)"),
                       ("real", "X", (13, 6), "sigma noktaları"),
                       ("real", "work", (72,), "6·P ve özdeğer yedeği çalışma alanı"),
                       ("u32", "fallbacks", (), "Cholesky başarısız → özdeğer yedeği sayacı (tanılama)")]
        return fields
    if algo == "weighted_mean":
        return [cfg, core, ("real", "hold", (3,), "tutulan son ağırlıklı ortalamalar [konum, hız, ivme]")]
    return [cfg, core, ("real", "x", (3,), "durum [p, ṗ, p̈][:d]"), ("real", "P", (3, 3), "kovaryans (d×d kullanılır)")]


def _macros(r: _Resolved) -> dict:
    m = {"SPT_POS_FIX_HISTORY": FIX_HISTORY, "SPT_LIN_PF_DMAX": lin_mod.LEVEL[r.values.get("order", "hiz")] + 1
         if r.scenario == "linear_multi" else 3}
    m[f"SPT_{r.sab.upper()}_PF_PARTICLES"] = _particles(r)
    return m


def _registry(r: _Resolved) -> dict:
    return {f"spt_{r.sab}_cfg": _cfg_fields(r), f"spt_{r.sab}_core": _core_fields(r), "spt_rng": _RNG_FIELDS}


def state_size(opts: GenericCodegenOptions, algorithm: str) -> int:
    """sizeof(durum yapısı) — yapı düzeninden (ILP32 ARM AAPCS ve x86-64'te aynı: işaretçi alanı yok)."""
    r = _resolve(opts)
    return _sizeof(_algo_fields(r, algorithm), opts.real, _macros(r), _registry(r))[0]


# ============================================================================ C şablonları
_BANNER = """Signal Processing Toolbox tarafından üretildi; Python referansı: spt/dsp/fusion_scenarios/.
 * Ayrıntılı kullanım: KULLANIM.md"""

_COMMON_H = r"""/**
 * @file spt_fusion_generic_common.h
 * @brief Genel (senaryo tabanlı) sensör füzyonu ortak tanımları: sayı türü, matematik makroları, NaN (= ölçüm yok),
 *        Python ile aynı açı sarma, rastgele sayı üreteci, özdeğer yedekli Cholesky, parçacık ağırlıkları ve
 *        Liu–West yeniden örnekleme.
 *
 * {{BANNER}}
 *
 * Kurallar (tüm senaryolar; Python spt.dsp.fusion_generic ile aynı):
 *  - Girişler SI birimindedir (Pa, m, m/s², rad/s, m/s; enlem / boylam rad). Birim dönüşümünü çağıran yapar.
 *  - Giriş NaN (SPT_NAN) = "bu örnekte ölçüm yok" (seyrek sensör, kesinti); sonsuz / aşırı büyük değerler de yok
 *    sayılır. Vektör girişlerde (GNSS) NULL işaretçi de "yok" demektir.
 *  - Çıktı NaN = henüz kestirim yok (Python'un NaN verdiği örneklerde). Açısal çıktılar radyandır.
 *  - Tüm durum çağıranın yapısındadır: malloc ve global değişken yok, fonksiyonlar yeniden girişlidir.
 *
 * Yapılandırma (derleyici komut satırından ya da bu dosyadan önce tanımlanabilir):
 *  - SPT_USE_DOUBLE : 1 → double, 0 → float (üretimde seçilen: {{REAL}}).
 *  - SPT_SQRT, SPT_SIN, SPT_COS, SPT_ATAN2, SPT_EXP, SPT_LOG, SPT_FABS, SPT_FMOD, SPT_HYPOT, SPT_POW: matematik
 *    fonksiyonları (ör. CMSIS-DSP ile değiştirilebilir; doğruluk sorumluluğu kullanıcıdadır).
 *  - Yönelim füzyonu (spt_fusion_common.h) aynı derleme biriminde kullanılabilir (ekleme sırası serbest): ortak sayı
 *    türü ve makrolar ilk eklenen başlıktan gelir. İki üretimi aynı sayı türüyle (float / double) yapın.
 *
 * Uyarı: -ffast-math / -ffinite-math-only ile derlemeyin; NaN (eksik ölçüm) denetimleri kaldırılır.
 */
#ifndef SPT_FUSION_GENERIC_COMMON_H
#define SPT_FUSION_GENERIC_COMMON_H

#include <stddef.h>
#include <stdint.h>

#ifndef SPT_USE_DOUBLE
#define SPT_USE_DOUBLE {{USE_DOUBLE}}
#endif

#ifndef SPT_FUSION_SHARED_DEFS
/* ---- yönelim ve genel füzyon başlıklarının ortak tanımları: hangisi önce eklenirse oradan gelir -------------- */
#define SPT_FUSION_SHARED_DEFS
#if defined(SPT_CUSTOM_REAL)
/* SPT_REAL türü ve tüm SPT_xxx makroları bu dosyadan önce tanımlanır (ana makinedeki işlem sayacı testi). */
#elif SPT_USE_DOUBLE
#include <float.h>
#include <math.h>
/** Hesap türü: double (çift hassasiyet FPU'su olmayan çekirdeklerde yazılım kayan noktadır, yavaştır). */
typedef double SPT_REAL;
#define SPT_C(x) (x)
#define SPT_REAL_MAX DBL_MAX
#define SPT_EPS DBL_EPSILON
#ifndef SPT_SQRT
#define SPT_SQRT(x) sqrt(x)
#endif
#ifndef SPT_SIN
#define SPT_SIN(x) sin(x)
#endif
#ifndef SPT_COS
#define SPT_COS(x) cos(x)
#endif
#ifndef SPT_ATAN2
#define SPT_ATAN2(y, x) atan2((y), (x))
#endif
#ifndef SPT_ASIN
#define SPT_ASIN(x) asin(x)
#endif
#ifndef SPT_EXP
#define SPT_EXP(x) exp(x)
#endif
#ifndef SPT_LOG
#define SPT_LOG(x) log(x)
#endif
#ifndef SPT_FABS
#define SPT_FABS(x) fabs(x)
#endif
#ifndef SPT_FMOD
#define SPT_FMOD(x, y) fmod((x), (y))
#endif
#ifndef SPT_HYPOT
#define SPT_HYPOT(x, y) hypot((x), (y))
#endif
#else
#include <float.h>
#include <math.h>
/** Hesap türü: float (Cortex-M4F / M7 / M33F FPU'sunda donanımda). Sabitler SPT_C() ile float sabiti olur. */
typedef float SPT_REAL;
#define SPT_C(x) (x##f)
#define SPT_REAL_MAX FLT_MAX
#define SPT_EPS FLT_EPSILON
#ifndef SPT_SQRT
#define SPT_SQRT(x) sqrtf(x)
#endif
#ifndef SPT_SIN
#define SPT_SIN(x) sinf(x)
#endif
#ifndef SPT_COS
#define SPT_COS(x) cosf(x)
#endif
#ifndef SPT_ATAN2
#define SPT_ATAN2(y, x) atan2f((y), (x))
#endif
#ifndef SPT_ASIN
#define SPT_ASIN(x) asinf(x)
#endif
#ifndef SPT_EXP
#define SPT_EXP(x) expf(x)
#endif
#ifndef SPT_LOG
#define SPT_LOG(x) logf(x)
#endif
#ifndef SPT_FABS
#define SPT_FABS(x) fabsf(x)
#endif
#ifndef SPT_FMOD
#define SPT_FMOD(x, y) fmodf((x), (y))
#endif
#ifndef SPT_HYPOT
#define SPT_HYPOT(x, y) hypotf((x), (y))
#endif
#endif

#ifndef SPT_FROM_INT
#define SPT_FROM_INT(i) ((SPT_REAL)(i))
#endif
#ifndef SPT_U32_TO_REAL
#define SPT_U32_TO_REAL(u) ((SPT_REAL)(u))
#endif

#define SPT_PI           SPT_C(3.14159265358979323846)
#define SPT_NEG_PI       SPT_C(-3.14159265358979323846)
#define SPT_TWO_PI       SPT_C(6.28318530717958647693)
#define SPT_THREE_PI     SPT_C(9.42477796076937971539)
#define SPT_NEG_THREE_PI SPT_C(-9.42477796076937971539)
#define SPT_D2R          SPT_C(0.0174532925199432957692)

#if defined(__cplusplus) && !defined(SPT_CUSTOM_REAL)
#define SPT_BEGIN_DECLS extern "C" {
#define SPT_END_DECLS }
#else
#define SPT_BEGIN_DECLS
#define SPT_END_DECLS
#endif

/** Sonlu mu (NaN / ±sonsuz değil). */
static inline int spt_finite(SPT_REAL v)
{
    return v == v && v <= SPT_REAL_MAX && v >= -SPT_REAL_MAX;
}
#endif /* !SPT_FUSION_SHARED_DEFS */

/* ---- genel füzyona özgü tanımlar --------------------------------------------------------------------------------- */
#if defined(SPT_CUSTOM_REAL)
/* SPT_POW, SPT_NAN, SPT_TINY, SPT_HUGE önceden tanımlanır. */
#elif SPT_USE_DOUBLE
#ifndef SPT_POW
#define SPT_POW(x, y) pow((x), (y))
#endif
/** "Ölçüm yok" / "kestirim yok" değeri. */
#define SPT_NAN ((double)NAN)
#define SPT_TINY SPT_C(1e-300)        /* Python'daki ağırlık / pivot alt sınırı */
#define SPT_HUGE SPT_C(1e300)         /* doğrusal senaryonun geçerli ölçüm sınırı */
#else
#ifndef SPT_POW
#define SPT_POW(x, y) powf((x), (y))
#endif
/** "Ölçüm yok" / "kestirim yok" değeri. */
#define SPT_NAN NAN
#define SPT_TINY FLT_MIN              /* float'ta 1e-300 sıfıra iner: en küçük normal sayı */
#define SPT_HUGE FLT_MAX
#endif
#define SPT_FOUR_PI SPT_C(12.5663706143591729539)
#define SPT_LW_A SPT_C(0.98)          /* Liu–West çekirdek daraltma katsayısı a */
#define SPT_LW_K SPT_C(0.0596)        /* 1 − a² + şişirme (0.02): yeniden örnekleme gürültüsü kovaryans çarpanı */
#define SPT_LW_DMAX 6

/** Açıyı [−π, π) aralığına sarar — Python (a + π) % 2π − π ile bit düzeyinde aynı (yaygın durumda fmod yok). */
static inline SPT_REAL spt_wrap(SPT_REAL a)
{
    SPT_REAL t = a + SPT_PI;
    if (t >= SPT_C(0.0)) {
        if (t < SPT_TWO_PI) {
            return t - SPT_PI;
        }
        if (t < SPT_FOUR_PI) {
            return (t - SPT_TWO_PI) - SPT_PI;    /* Sterbenz: çıkarma kesin, fmod ile aynı */
        }
    } else if (t >= -SPT_TWO_PI) {
        return (t + SPT_TWO_PI) - SPT_PI;
    }
    t = SPT_FMOD(t, SPT_TWO_PI);
    if (t < SPT_C(0.0)) {
        t += SPT_TWO_PI;
    }
    return t - SPT_PI;
}

/* ---- rastgele sayı üreteci (parçacık filtreleri) ------------------------------------------------------------------ */
/** xorshift32 + Marsaglia kutupsal normal üreteci (Python numpy PCG64'ten farklı; istatistiksel olarak eşdeğer). */
{{RNG_STRUCT}}
/** Tohum karıştırma (murmur3 son adımı); xorshift32 durumu sıfır olamaz. */
static inline void spt_rng_seed(spt_rng_t *r, uint32_t seed)
{
    uint32_t x = seed + 0x9E3779B9u;
    x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
    x = (x ^ (x >> 13)) * 0xC2B2AE35u;
    x ^= x >> 16;
    r->s = x ? x : 0x6D2B79F5u;
    r->has_spare = 0;
    r->spare = SPT_C(0.0);
}

/** xorshift32 (Marsaglia 13/17/5). */
static inline uint32_t spt_rng_next(spt_rng_t *r)
{
    uint32_t x = r->s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    r->s = x;
    return x;
}

/** [0, 1) tekdüze sayı (üst 24 bit). */
static inline SPT_REAL spt_rng_uniform(spt_rng_t *r)
{
    return SPT_U32_TO_REAL(spt_rng_next(r) >> 8) * SPT_C(5.9604644775390625e-08);
}

/** Standart normal sayı: Marsaglia kutupsal yöntemi (çiftin ikincisi saklanır). */
static inline SPT_REAL spt_rng_normal(spt_rng_t *r)
{
    SPT_REAL u, v, r2, f;
    if (r->has_spare) {
        r->has_spare = 0;
        return r->spare;
    }
    do {
        u = SPT_C(2.0) * spt_rng_uniform(r) - SPT_C(1.0);
        v = SPT_C(2.0) * spt_rng_uniform(r) - SPT_C(1.0);
        r2 = u * u + v * v;
    } while (r2 >= SPT_C(1.0) || r2 == SPT_C(0.0));
    f = SPT_SQRT(SPT_C(-2.0) * SPT_LOG(r2) / r2);
    r->spare = v * f;
    r->has_spare = 1;
    return u * f;
}

/* ---- yarı-tanımlı matris karekökü (Python chol_psd) -------------------------------------------------------------- */
/** Yedek: simetrik ½(A + Aᵀ)'nin dönüşümlü Jacobi özdeğer ayrışımı, L = V·diag(√max(λ, 0)) (Python eigh yedeği;
 *  sigma kümesi {±Lⱼ} özvektör işaret / sırasından bağımsızdır). work: n·n. */
static inline void spt_eig_sqrt(int n, const SPT_REAL *a, SPT_REAL *l, SPT_REAL *work)
{
    SPT_REAL *m = work;
    int i, j, k, p, q, sweep;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            m[i * n + j] = SPT_C(0.5) * (a[i * n + j] + a[j * n + i]);
            l[i * n + j] = (i == j) ? SPT_C(1.0) : SPT_C(0.0);
        }
    }
    for (sweep = 0; sweep < 50; ++sweep) {
        SPT_REAL off = SPT_C(0.0), diag = SPT_C(0.0);
        for (i = 0; i < n; ++i) {
            diag += m[i * n + i] * m[i * n + i];
            for (j = i + 1; j < n; ++j) {
                off += m[i * n + j] * m[i * n + j];
            }
        }
        if (!(off > SPT_EPS * SPT_EPS * diag)) {
            break;
        }
        for (p = 0; p < n - 1; ++p) {
            for (q = p + 1; q < n; ++q) {
                SPT_REAL apq = m[p * n + q], theta, t, c, sn;
                if (apq == SPT_C(0.0)) {
                    continue;
                }
                theta = (m[q * n + q] - m[p * n + p]) / (SPT_C(2.0) * apq);
                t = SPT_C(1.0) / (SPT_FABS(theta) + SPT_SQRT(theta * theta + SPT_C(1.0)));
                if (theta < SPT_C(0.0)) {
                    t = -t;
                }
                c = SPT_C(1.0) / SPT_SQRT(t * t + SPT_C(1.0));
                sn = t * c;
                for (k = 0; k < n; ++k) {
                    SPT_REAL akp = m[k * n + p], akq = m[k * n + q];
                    m[k * n + p] = c * akp - sn * akq;
                    m[k * n + q] = sn * akp + c * akq;
                }
                for (k = 0; k < n; ++k) {
                    SPT_REAL apk = m[p * n + k], aqk = m[q * n + k];
                    m[p * n + k] = c * apk - sn * aqk;
                    m[q * n + k] = sn * apk + c * aqk;
                }
                for (k = 0; k < n; ++k) {
                    SPT_REAL vkp = l[k * n + p], vkq = l[k * n + q];
                    l[k * n + p] = c * vkp - sn * vkq;
                    l[k * n + q] = sn * vkp + c * vkq;
                }
            }
        }
    }
    for (j = 0; j < n; ++j) {
        SPT_REAL lam = m[j * n + j];
        SPT_REAL f = (lam > SPT_C(0.0)) ? SPT_SQRT(lam) : SPT_C(0.0);
        for (i = 0; i < n; ++i) {
            l[i * n + j] *= f;
        }
    }
}

/**
 * L·Lᵀ = A (n×n, satır düzeni; Python chol_psd). Cholesky (alt üçgen, üst kısım sıfır); bir pivot ≤ 0 ya da NaN ise
 * özdeğer yedeği (tam matris). work: n·n gerçek sayı.
 * @return 1: Cholesky, 0: özdeğer yedeği kullanıldı
 */
static inline int spt_chol_psd(int n, const SPT_REAL *a, SPT_REAL *l, SPT_REAL *work)
{
    int i, j, k;
    for (j = 0; j < n; ++j) {
        SPT_REAL d = a[j * n + j], id;
        for (k = 0; k < j; ++k) {
            d -= l[j * n + k] * l[j * n + k];
        }
        if (!(d > SPT_C(0.0))) {
            spt_eig_sqrt(n, a, l, work);
            return 0;
        }
        d = SPT_SQRT(d);
        l[j * n + j] = d;
        id = SPT_C(1.0) / d;
        for (i = j + 1; i < n; ++i) {
            SPT_REAL acc = a[i * n + j];
            for (k = 0; k < j; ++k) {
                acc -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = acc * id;
        }
        for (i = 0; i < j; ++i) {
            l[i * n + j] = SPT_C(0.0);
        }
    }
    return 1;
}

/* ---- parçacık ağırlıkları -------------------------------------------------------------------------------------- */
/** Eşit ağırlık 1/n. @return ESS = n */
static inline SPT_REAL spt_pf_uniform(SPT_REAL *w, int n)
{
    SPT_REAL v = SPT_C(1.0) / SPT_FROM_INT(n);
    int i;
    for (i = 0; i < n; ++i) {
        w[i] = v;
    }
    return SPT_FROM_INT(n);
}

/** w: log-ağırlıklar → normalize ağırlıklar (Python normalized_weights: NaN / −∞ → 0; hiç sonlu yoksa eşit).
 *  @return ESS = 1 / Σw² */
static inline SPT_REAL spt_pf_from_log(SPT_REAL *w, int n)
{
    SPT_REAL m = SPT_C(0.0), sum = SPT_C(0.0), sum2 = SPT_C(0.0), is;
    int i, any = 0;
    for (i = 0; i < n; ++i) {
        if (spt_finite(w[i]) && (!any || w[i] > m)) {
            m = w[i];
            any = 1;
        }
    }
    if (!any) {
        return spt_pf_uniform(w, n);
    }
    for (i = 0; i < n; ++i) {
        SPT_REAL v = spt_finite(w[i]) ? SPT_EXP(w[i] - m) : SPT_C(0.0);
        w[i] = v;
        sum += v;
        sum2 += v * v;
    }
    is = SPT_C(1.0) / sum;
    for (i = 0; i < n; ++i) {
        w[i] *= is;
    }
    return sum * sum / sum2;
}

/** İki geçişli olabilirlik güncellemesi, 1. geçiş: en büyük sonlu log-olabilirlik. */
static inline void spt_pf_max(SPT_REAL ll, SPT_REAL *lmax, int *any)
{
    if (spt_finite(ll) && (!*any || ll > *lmax)) {
        *lmax = ll;
        *any = 1;
    }
}

/** 2. geçiş, tek parçacık: w ← max(w, tiny)·exp(ll − lmax) (= Python exp(log max(w, 1e-300) + ll − m) oranı;
 *  NaN olabilirlik → 0). */
static inline void spt_pf_mul(SPT_REAL *w, SPT_REAL ll, SPT_REAL lmax, SPT_REAL *sum, SPT_REAL *sum2)
{
    SPT_REAL v = *w;
    if (v < SPT_TINY) {
        v = SPT_TINY;
    }
    v = (ll == ll) ? v * SPT_EXP(ll - lmax) : SPT_C(0.0);
    *w = v;
    *sum += v;
    *sum2 += v * v;
}

/** 2. geçişten sonra: Σ = sum ile normalize. @return ESS */
static inline SPT_REAL spt_pf_scale(SPT_REAL *w, int n, SPT_REAL sum, SPT_REAL sum2)
{
    SPT_REAL is = SPT_C(1.0) / sum;
    int i;
    for (i = 0; i < n; ++i) {
        w[i] *= is;
    }
    return sum * sum / sum2;
}

/**
 * Sistematik yeniden örnekleme + Liu–West çekirdek yumuşatma (Python liu_west_resample: a = 0.98, şişirme %2).
 * x: n parçacık × stride (satır düzeni; ilk d ≤ 6 bileşen kullanılır), w: normalize ağırlıklar (çıkışta 1/n), cnt: n
 * geçici sayaç, work: 3·d·d, angle: dairesel işlenen bileşen (−1: yok). Seçilen parçacıklar ağırlıklı ortalamaya a
 * oranında çekilir, N(0, (1 − a² + 0.02)·Σ) gürültü eklenir. Python'dan fark: rastgele sayılar ve parçacık sırası
 * (yerinde kopyalama) — istatistiksel olarak eşdeğer.
 */
static inline void spt_lw_resample(SPT_REAL *x, int n, int stride, int d, SPT_REAL *w, uint16_t *cnt,
                                   spt_rng_t *rng, int angle, SPT_REAL *work)
{
    SPT_REAL mean[SPT_LW_DMAX], dev[SPT_LW_DMAX], nz[SPT_LW_DMAX];
    SPT_REAL *cov = work, *l = work + d * d, *tmp = work + 2 * d * d;
    SPT_REAL sn = SPT_C(0.0), cs = SPT_C(0.0), inv_n, u, cum = SPT_C(0.0);
    int i, j = 0, a, c, chol, free_slot = 0;
    for (a = 0; a < d; ++a) {
        mean[a] = SPT_C(0.0);
    }
    for (i = 0; i < n; ++i) {
        const SPT_REAL *xi = x + i * stride;
        for (a = 0; a < d; ++a) {
            mean[a] += w[i] * xi[a];
        }
        if (angle >= 0) {
            sn += w[i] * SPT_SIN(xi[angle]);
            cs += w[i] * SPT_COS(xi[angle]);
        }
    }
    if (angle >= 0) {
        mean[angle] = SPT_ATAN2(sn, cs);
    }
    for (a = 0; a < d * d; ++a) {
        cov[a] = SPT_C(0.0);
    }
    for (i = 0; i < n; ++i) {
        const SPT_REAL *xi = x + i * stride;
        for (a = 0; a < d; ++a) {
            dev[a] = xi[a] - mean[a];
        }
        if (angle >= 0) {
            dev[angle] = spt_wrap(dev[angle]);
        }
        for (a = 0; a < d; ++a) {
            SPT_REAL wa = w[i] * dev[a];
            for (c = a; c < d; ++c) {
                cov[a * d + c] += wa * dev[c];
            }
        }
    }
    for (a = 0; a < d; ++a) {
        for (c = a; c < d; ++c) {
            cov[a * d + c] *= SPT_LW_K;
            cov[c * d + a] = cov[a * d + c];
        }
    }
    chol = spt_chol_psd(d, cov, l, tmp);
    /* kopya sayıları: (u + j)/n konumlarından cum_{i−1} < konum ≤ cum_i olanlar (Python searchsorted) */
    inv_n = SPT_C(1.0) / SPT_FROM_INT(n);
    u = spt_rng_uniform(rng);
    for (i = 0; i < n; ++i) {
        int cc = 0;
        cum += w[i];
        while (j < n && (u + SPT_FROM_INT(j)) * inv_n <= cum) {
            ++cc;
            ++j;
        }
        cnt[i] = (uint16_t)cc;
    }
    cnt[n - 1] = (uint16_t)(cnt[n - 1] + (n - j));   /* Python c[−1] = 1: yuvarlama artığı sona */
    /* yerinde kopyalama: fazla kopyalar seçilmemiş (sayısı 0) yuvalara */
    for (i = 0; i < n; ++i) {
        while (cnt[i] > 1) {
            while (cnt[free_slot] != 0) {
                ++free_slot;
            }
            for (a = 0; a < d; ++a) {
                x[free_slot * stride + a] = x[i * stride + a];
            }
            cnt[free_slot] = 1;
            cnt[i] = (uint16_t)(cnt[i] - 1);
        }
    }
    /* çekirdek: x ← ortalama + a·(x − ortalama) + L·n */
    for (i = 0; i < n; ++i) {
        SPT_REAL *xi = x + i * stride;
        for (a = 0; a < d; ++a) {
            dev[a] = xi[a] - mean[a];
            nz[a] = spt_rng_normal(rng);
        }
        if (angle >= 0) {
            dev[angle] = spt_wrap(dev[angle]);
        }
        for (a = 0; a < d; ++a) {
            SPT_REAL v = mean[a] + SPT_LW_A * dev[a];
            int cmax = chol ? a + 1 : d;
            for (c = 0; c < cmax; ++c) {
                v += l[a * d + c] * nz[c];
            }
            xi[a] = v;
        }
        if (angle >= 0) {
            xi[angle] = spt_wrap(xi[angle]);
        }
        w[i] = inv_n;
    }
}

#endif /* SPT_FUSION_GENERIC_COMMON_H */
"""

# ---------------------------------------------------------------------------- irtifa ortak başlığı
_ALT_COMMON_H = r"""/**
 * @file spt_alt_common.h
 * @brief İrtifa (barometre + ivmeölçer) senaryosunun ortak parçaları: senaryo parametreleri, ISA barometrik formül,
 *        baro / ivme örneği kuralları, Kalman ailesinin ortak öngörüsü ve doğrusal ölçüm güncellemesi.
 *
 * {{BANNER}}
 * Python karşılığı: spt/dsp/fusion_scenarios/altitude.py (_AltitudeBase, _AltitudeKalman).
 *
 * Model (durum [h, v, b]: irtifa m, dikey hız m/s yukarı +, ivme sapması m/s²; T = örnek aralığı):
 *   a = f − g − b,  h⁺ = h + v·T + ½·a·T²,  v⁺ = v + a·T,  b⁺ = b + w_b
 *   Ölçüm: baro irtifa z = h + n, ya da basınç z = p0·(1 − h/44330.77)^5.25588 + n (ISA).
 * Girişler: baro_p (Pa), baro_alt (m), acc_z (m/s², dünya z yukarı, yerçekimi DAHİL: durağanken +g). NaN = yok.
 * İkisi de eşlenirse örnekte basınç varsa o kullanılır. Eksik ivmede son değer tutulur ve hıza 2 m/s/√s yürüyüş
 * eklenir. p0 = 0: ilk geçerli basınç h = 0 kabul edilir (göreli irtifa).
 */
#ifndef SPT_ALT_COMMON_H
#define SPT_ALT_COMMON_H

#include "spt_fusion_generic_common.h"

/* ---- Senaryo parametreleri: derleme anı varsayılanları (#ifndef korumalı; defaults() ile s.cfg'ye kopyalanır) ---- */
{{PARAM_DEFINES}}

/* ---- Eşlenen girişler (üretim anında sabit; API yalnız bunları alır — değiştirmeyin) */
#define SPT_ALT_HAS_BARO_P {{HAS_BARO_P}}
#define SPT_ALT_HAS_BARO_ALT {{HAS_BARO_ALT}}

/* ---- Model sabitleri (Python altitude.py ile aynı) */
#define SPT_ISA_H         SPT_C(44330.77)              /* m */
#define SPT_ISA_INV_H     SPT_C(2.2557695253206746e-05) /* 1 / 44330.77 */
#define SPT_ISA_N         SPT_C(5.25588)
#define SPT_ISA_INV_N     SPT_C(0.1902630958088845)    /* 1 / 5.25588 */
#define SPT_ISA_N1        SPT_C(4.25588)               /* 5.25588 − 1 */
#define SPT_ISA_DPDH_K    SPT_C(0.00011856053932742429) /* 5.25588 / 44330.77 */
#define SPT_ALT_V0_SIGMA  SPT_C(1.0)                   /* m/s, başlangıç hız belirsizliği */
#define SPT_ALT_MISS_ACC2 SPT_C(4.0)                   /* (2 m/s/√s)²: ivme yokken hız yürüyüşü */
#define SPT_ALT_BIG       SPT_C(1e12)                  /* geçerli ölçüm sınırı */

{{CFG_STRUCT}}
{{CORE_STRUCT}}
/** Senaryo parametrelerini derleme anı varsayılanlarına getirir. */
static inline void spt_alt_cfg_defaults(spt_alt_cfg_t *c)
{
{{CFG_DEFAULTS}}
}

/** Senaryo durumunu sıfırlar ve p0 kuralını denetler (Python _setup: p0 = 0 otomatik ya da ≥ 1000 Pa).
 *  @return 1: parametreler geçerli */
static inline int spt_alt_core_reset(spt_alt_core_t *c, const spt_alt_cfg_t *cfg)
{
    c->out[0] = SPT_NAN;
    c->out[1] = SPT_NAN;
    c->ok = (cfg->p0 == SPT_C(0.0) || cfg->p0 >= SPT_C(1000.0));
    c->p0 = cfg->p0;
    c->p0_ok = (cfg->p0 != SPT_C(0.0));
    c->f_last = SPT_C(0.0);
    c->f_ok = 0;
    c->k = 0u;
    c->n_baro = 0u;
    c->ready = 0;
    return c->ok;
}

/** ISA barometrik formül: p = p0·max(1 − h/44330.77, 1e-9)^5.25588. */
static inline SPT_REAL spt_isa_pressure(SPT_REAL h, SPT_REAL p0)
{
    SPT_REAL u = SPT_C(1.0) - h * SPT_ISA_INV_H;
    if (u < SPT_C(1e-9)) {
        u = SPT_C(1e-9);
    }
    return p0 * SPT_POW(u, SPT_ISA_N);
}

/** Ters ISA: h = 44330.77·(1 − (p/p0)^(1/5.25588)). */
static inline SPT_REAL spt_isa_altitude(SPT_REAL p, SPT_REAL p0)
{
    return SPT_ISA_H * (SPT_C(1.0) - SPT_POW(p / p0, SPT_ISA_INV_N));
}

/** dp/dh (Pa/m, negatif; deniz seviyesinde ≈ −12 Pa/m). */
static inline SPT_REAL spt_isa_dpdh(SPT_REAL h, SPT_REAL p0)
{
    SPT_REAL u = SPT_C(1.0) - h * SPT_ISA_INV_H;
    if (u < SPT_C(1e-9)) {
        u = SPT_C(1e-9);
    }
    return -p0 * SPT_ISA_DPDH_K * SPT_POW(u, SPT_ISA_N1);
}

/**
 * Bu örnekteki baro ölçümü (Python _baro): basınç öncelikli; p0 otomatikse ilk geçerli basınç p0 olur.
 * @return 1: basınç (*z Pa), 2: baro irtifa (*z m), 0: ölçüm yok
 */
static inline int spt_alt_baro(spt_alt_core_t *c, SPT_REAL p, SPT_REAL h, SPT_REAL *z)
{
#if SPT_ALT_HAS_BARO_P
    if (p > SPT_C(0.0) && p < SPT_ALT_BIG) {
        if (!c->p0_ok) {
            c->p0 = p;
            c->p0_ok = 1;
        }
        *z = p;
        return 1;
    }
#else
    (void)p;
#endif
#if SPT_ALT_HAS_BARO_ALT
    if (h > -SPT_ALT_BIG && h < SPT_ALT_BIG) {
        *z = h;
        return 2;
    }
#else
    (void)h;
#endif
    (void)c;
    return 0;
}

/** Ölçümün irtifa karşılığı (m). */
static inline SPT_REAL spt_alt_h_of(const spt_alt_core_t *c, int kind, SPT_REAL z)
{
    return (kind == 1) ? spt_isa_altitude(z, c->p0) : z;
}

/** Ölçümün irtifa karşılığı ve varyansı (Python _alt_of; basınçta R = (σ_p / |dp/dh|)²). */
static inline SPT_REAL spt_alt_of(const spt_alt_core_t *c, const spt_alt_cfg_t *cfg, int kind, SPT_REAL z,
                                  SPT_REAL *var)
{
    if (kind == 1) {
        SPT_REAL h = spt_isa_altitude(z, c->p0);
        SPT_REAL r = cfg->sigma_p / SPT_FABS(spt_isa_dpdh(h, c->p0));
        *var = r * r;
        return h;
    }
    *var = cfg->sigma_h * cfg->sigma_h;
    return z;
}

/**
 * Geçerli ivme ya da tutulan son değer (Python _accel).
 * @return 0: geçerli, 1: eksik (son değer tutuldu), 2: hiç ivme gelmedi (çağıran f = g + b alır → a = 0)
 */
static inline int spt_alt_accel(spt_alt_core_t *c, SPT_REAL f, SPT_REAL *out)
{
    if (f > -SPT_ALT_BIG && f < SPT_ALT_BIG) {
        c->f_last = f;
        c->f_ok = 1;
        *out = f;
        return 0;
    }
    *out = c->f_last;
    return c->f_ok ? 1 : 2;
}

/** Kalman ailesi başlangıcı: h = z_h, v = b = 0; P0 = diag(R_h, (1 m/s)², σ_b0²). */
static inline void spt_alt_kalman_start(SPT_REAL x[3], SPT_REAL P[6], const spt_alt_cfg_t *cfg, SPT_REAL h,
                                        SPT_REAL r0)
{
    x[0] = h;
    x[1] = SPT_C(0.0);
    x[2] = SPT_C(0.0);
    P[0] = r0;
    P[1] = SPT_C(0.0);
    P[2] = SPT_C(0.0);
    P[3] = SPT_ALT_V0_SIGMA * SPT_ALT_V0_SIGMA;
    P[4] = SPT_C(0.0);
    P[5] = cfg->bias0 * cfg->bias0;
}

/** Kalman ailesi öngörüsü (kapalı biçim): x⁺ = F·x + B·(f − g), P⁺ = F·P·Fᵀ + Q,
 *  F = [[1, T, −T²/2], [0, 1, −T], [0, 0, 1]], Q = σ_a²·G·Gᵀ + diag(0, 0, σ_b²·T). */
static inline void spt_alt_kalman_predict(SPT_REAL x[3], SPT_REAL P[6], const spt_alt_cfg_t *cfg, SPT_REAL f,
                                          int missing, SPT_REAL T)
{
    SPT_REAL TT = T * T, s_ = SPT_C(-0.5) * TT, r_ = -T;
    SPT_REAL qa = cfg->sigma_acc * cfg->sigma_acc, q22 = cfg->sigma_bias * cfg->sigma_bias * T;
    SPT_REAL a = f - cfg->g - x[2];
    SPT_REAL m00, m01, m02, m11, m12;
    if (missing) {
        qa += SPT_ALT_MISS_ACC2 / T;
    }
    x[0] += x[1] * T + SPT_C(0.5) * a * TT;
    x[1] += a * T;
    m00 = P[0] + T * P[1] + s_ * P[2];
    m01 = P[1] + T * P[3] + s_ * P[4];
    m02 = P[2] + T * P[4] + s_ * P[5];
    m11 = P[3] + r_ * P[4];
    m12 = P[4] + r_ * P[5];
    P[0] = m00 + T * m01 + s_ * m02 + qa * TT * TT * SPT_C(0.25);
    P[1] = m01 + r_ * m02 + qa * TT * T * SPT_C(0.5);
    P[2] = m02;
    P[3] = m11 + r_ * m12 + qa * TT;
    P[4] = m12;
    P[5] += q22;
}

/** Skaler ölçüm güncellemesi, H = [hh, 0, 0] (Python _linear). */
static inline void spt_alt_linear_update(SPT_REAL x[3], SPT_REAL P[6], SPT_REAL z, SPT_REAL hh, SPT_REAL pred,
                                         SPT_REAL R)
{
    SPT_REAL c0 = hh * P[0], c1 = hh * P[1], c2 = hh * P[2];
    SPT_REAL is = SPT_C(1.0) / (hh * c0 + R);
    SPT_REAL k0 = c0 * is, k1 = c1 * is, k2 = c2 * is, y = z - pred;
    x[0] += k0 * y;
    x[1] += k1 * y;
    x[2] += k2 * y;
    P[0] -= k0 * c0;
    P[1] -= k0 * c1;
    P[2] -= k0 * c2;
    P[3] -= k1 * c1;
    P[4] -= k1 * c2;
    P[5] -= k2 * c2;
}

#endif /* SPT_ALT_COMMON_H */
"""

# ---------------------------------------------------------------------------- 2B konum ortak başlığı
_POS_COMMON_H = r"""/**
 * @file spt_pos_common.h
 * @brief 2B konum (GNSS + jiroskop + hız / odometri) senaryosunun ortak parçaları: senaryo parametreleri, giriş
 *        kuralları, enlem / boylam → yerel metre, süreç gürültüsü ve GNSS izinden başlatma.
 *
 * {{BANNER}}
 * Python karşılığı: spt/dsp/fusion_scenarios/position2d.py (_Pos2DBase).
 *
 * Model (CTRV benzeri tek tekerlek; durum [x, y, ψ, v, b_g, s], ω = gyro_z − b_g, T = örnek aralığı):
 *   ψm = ψ + ½·ω·T;  x⁺ = x + v·T·cos ψm;  y⁺ = y + v·T·sin ψm;  ψ⁺ = ψ + ω·T;  v⁺ = v + a·T ya da yürüyüş
 *   Ölçümler: GNSS z = (x, y) + n;  odometri z = s·v + n
 * Başlatma: GNSS izi d_init metreyi aşınca — konum son düzeltme, başlık kirişin yönü + kiriş boyunca dönülen açının
 * yarısı, hız odometriden ya da kiriş / süre. Öncesinde x, y = son GNSS düzeltmesi; başlık ve hız NaN.
 * Eksik jiroskop: son değer tutulur, başlığa 0.2 rad/√s yürüyüş eklenir. ψ: doğu ekseninden saat yönünün tersine.
 */
#ifndef SPT_POS_COMMON_H
#define SPT_POS_COMMON_H

#include "spt_fusion_generic_common.h"

/* ---- Senaryo parametreleri: derleme anı varsayılanları (#ifndef korumalı; defaults() ile s.cfg'ye kopyalanır) ---- */
{{PARAM_DEFINES}}

/** Başlatma kirişi için saklanan en çok GNSS düzeltmesi. Düzeltmeler en az 30 s / (SPT_POS_FIX_HISTORY − 1) aralıkla
 *  saklanır (5.3 Hz'e kadar hepsi): geçmiş her GNSS hızında son 30 s'yi kapsar (Python ile aynı kural). Değiştirirseniz
 *  Python referansından ayrılırsınız. RAM: düzeltme başına 3 gerçek sayı + 4 bayt. */
#ifndef SPT_POS_FIX_HISTORY
#define SPT_POS_FIX_HISTORY {{FIX_HISTORY}}
#endif

/* ---- Eşlenen girişler (üretim anında sabit; API yalnız bunları alır — değiştirmeyin) */
#define SPT_POS_HAS_GNSS {{HAS_GNSS}}
#define SPT_POS_HAS_GNSS_LL {{HAS_GNSS_LL}}
#define SPT_POS_HAS_SPEED {{HAS_SPEED}}
#define SPT_POS_HAS_ACC_FWD {{HAS_ACC_FWD}}

/* ---- Model sabitleri (Python position2d.py ile aynı) */
#define SPT_POS_INIT_WINDOW SPT_C(30.0)                /* s, başlatma geçmişi */
#define SPT_POS_FIX_SPACING (SPT_POS_INIT_WINDOW / SPT_FROM_INT(SPT_POS_FIX_HISTORY - 1))  /* s, saklama aralığı */
#define SPT_POS_MISS_RATE2  SPT_C(0.04000000000000001) /* 0.2² (rad²/s): jiroskop yokken başlık yürüyüşü */
#define SPT_POS_SCALE_RW2   SPT_C(1e-08)               /* (1e-4)²: odometri ölçeği yürüyüşü */
#define SPT_POS_BIG         SPT_C(1e12)                /* geçerli ölçüm sınırı */
#define SPT_POS_SQRT2       SPT_C(1.4142135623730951)
#define SPT_WGS84_A         SPT_C(6378137.0)
#define SPT_WGS84_E2        SPT_C(6.69437999014e-3)

{{CFG_STRUCT}}
{{CORE_STRUCT}}
/** Bir örneğin doğrulanmış girişleri (yalnız adım içinde). */
typedef struct spt_pos_in {
    SPT_REAL gz, zx, zy, spd, acc;   /* jiroskop (tutulan), GNSS (m), hız, ivme */
    int gmiss, fix, has_spd, has_acc;
} spt_pos_in_t;

/** Süreç gürültüsü varyansları (Python _noise). */
typedef struct spt_pos_noise {
    SPT_REAL pos, psi, psi_miss, v_rw, v_acc, bg, s;
} spt_pos_noise_t;

/** Senaryo parametrelerini derleme anı varsayılanlarına getirir. */
static inline void spt_pos_cfg_defaults(spt_pos_cfg_t *c)
{
{{CFG_DEFAULTS}}
}

/** Senaryo durumunu sıfırlar (Python _setup). @return 1 */
static inline int spt_pos_core_reset(spt_pos_core_t *c)
{
    int i;
    for (i = 0; i < 4; ++i) {
        c->out[i] = SPT_NAN;
    }
    c->lat0 = 0.0;
    c->lon0 = 0.0;
    c->theta = SPT_C(0.0);
    c->last_fix[0] = SPT_C(0.0);
    c->last_fix[1] = SPT_C(0.0);
    c->w_last = SPT_C(0.0);
    c->ll_kx = SPT_C(0.0);
    c->ll_ky = SPT_C(0.0);
    c->k = 0u;
    c->n_gnss = 0u;
    c->init_k = 0u;
    c->head = 0;
    c->nfix = 0;
    c->has_last_fix = 0;
    c->has_w = 0;
    c->has_origin = 0;
    c->ready = 0;
    c->ok = 1;
    return 1;
}

/** Sonlu mu (double; enlem / boylam). */
static inline int spt_finite_d(double v)
{
    return v == v && v <= DBL_MAX && v >= -DBL_MAX;
}

/** Enlem / boylam (rad, double) → ilk geçerli düzeltme etrafında yerel doğu / kuzey metre (eşdikdörtgen, WGS-84
 *  yarıçapları; Python ll_to_local). Fark double'da alınır: float modunda da cm çözünürlüğü korunur. */
static inline void spt_pos_ll_to_local(spt_pos_core_t *c, double lat, double lon, SPT_REAL *x, SPT_REAL *y)
{
    if (!c->has_origin) {
        SPT_REAL l0 = (SPT_REAL)lat, s = SPT_SIN(l0);
        SPT_REAL q = SPT_C(1.0) - SPT_WGS84_E2 * (s * s);
        c->lat0 = lat;
        c->lon0 = lon;
        c->has_origin = 1;
        c->ll_kx = SPT_WGS84_A / SPT_SQRT(q) * SPT_COS(l0);
        c->ll_ky = SPT_WGS84_A * (SPT_C(1.0) - SPT_WGS84_E2) / SPT_POW(q, SPT_C(1.5));
    }
    *x = c->ll_kx * spt_wrap((SPT_REAL)(lon - c->lon0));
    *y = c->ll_ky * (SPT_REAL)(lat - c->lat0);
}

/** Süreç gürültüsü varyansları (Python _noise; σ_gyro °/s, σ_gbias °/s/√s). */
static inline void spt_pos_noise(const spt_pos_cfg_t *cfg, SPT_REAL T, spt_pos_noise_t *q)
{
    SPT_REAL g = cfg->sigma_gyro * SPT_D2R * T, gb = cfg->sigma_gbias * SPT_D2R, a = cfg->sigma_accf * T;
    q->pos = cfg->sigma_pos * cfg->sigma_pos * T;
    q->psi = g * g;
    q->psi_miss = g * g + SPT_POS_MISS_RATE2 * T;
    q->v_rw = cfg->sigma_vrw * cfg->sigma_vrw * T;
    q->v_acc = a * a;
    q->bg = gb * gb * T;
    q->s = SPT_POS_SCALE_RW2 * T;
}

/** Başlangıç kovaryansı köşegeni (Python _p0). */
static inline void spt_pos_p0(const spt_pos_cfg_t *cfg, SPT_REAL d, SPT_REAL span, SPT_REAL v, int has_speed,
                              SPT_REAL p0[6])
{
    SPT_REAL sg = cfg->sigma_gnss, s_psi, s_v, gb = cfg->gbias0 * SPT_D2R, sc = SPT_C(0.01) * cfg->scale0;
    s_psi = SPT_POS_SQRT2 * sg / d + SPT_C(3.0) * SPT_D2R;
    if (has_speed) {
        s_v = cfg->sigma_speed + SPT_C(0.01) * cfg->scale0 * SPT_FABS(v);
    } else {
        s_v = SPT_POS_SQRT2 * sg / span + SPT_C(0.5);
    }
    p0[0] = sg * sg;
    p0[1] = sg * sg;
    p0[2] = s_psi * s_psi;
    p0[3] = s_v * s_v;
    p0[4] = gb * gb;
    p0[5] = sc * sc;
}

/** Başlatma öncesi (Python _pre_init): jiroskop açısını biriktirir, GNSS geçmişini tutar; kiriş d_init'i aşınca 1
 *  döner, st = [x, y, ψ, v] ve p0 (köşegen varyans) doldurulur. */
static inline int spt_pos_pre_init(spt_pos_core_t *c, const spt_pos_cfg_t *cfg, uint32_t kabs,
                                   const spt_pos_in_t *in, SPT_REAL T, SPT_REAL st[4], SPT_REAL p0[6])
{
    SPT_REAL horizon;
    int i, m;
    c->theta += in->gz * T;
    if (!in->fix) {
        return 0;
    }
    c->last_fix[0] = in->zx;
    c->last_fix[1] = in->zy;
    c->has_last_fix = 1;
    horizon = SPT_POS_INIT_WINDOW / T;
    while (c->nfix > 0 && SPT_FROM_INT(kabs - c->fk[c->head]) > horizon) {
        c->head = (c->head + 1) % SPT_POS_FIX_HISTORY;
        c->nfix--;
    }
    for (m = c->nfix - 1; m >= 0; --m) {             /* en yeniden eskiye: d ≥ d_init olan ilk saklı düzeltme */
        SPT_REAL dx, dy, d;
        i = (c->head + m) % SPT_POS_FIX_HISTORY;
        dx = in->zx - c->fx[i];
        dy = in->zy - c->fy[i];
        d = SPT_HYPOT(dx, dy);
        if (d >= cfg->d_init) {
            SPT_REAL span = SPT_FROM_INT(kabs - c->fk[i]) * T;
            st[0] = in->zx;
            st[1] = in->zy;
            st[2] = spt_wrap(SPT_ATAN2(dy, dx) + SPT_C(0.5) * (c->theta - c->fth[i]));
            st[3] = in->has_spd ? in->spd : d / span;
            spt_pos_p0(cfg, d, span, st[3], in->has_spd, p0);
            return 1;
        }
    }
    /* seyreltilmiş geçmiş: son saklanandan en az 30 s / (H − 1) sonraki düzeltme saklanır (tampon dolarsa en eski düşer) */
    i = (c->head + c->nfix - 1) % SPT_POS_FIX_HISTORY;
    if (c->nfix == 0 || SPT_FROM_INT(kabs - c->fk[i]) * T >= SPT_POS_FIX_SPACING) {
        if (c->nfix == SPT_POS_FIX_HISTORY) {
            c->head = (c->head + 1) % SPT_POS_FIX_HISTORY;
            c->nfix--;
        }
        i = (c->head + c->nfix) % SPT_POS_FIX_HISTORY;
        c->fk[i] = kabs;
        c->fx[i] = in->zx;
        c->fy[i] = in->zy;
        c->fth[i] = c->theta;
        c->nfix++;
    }
    return 0;
}

/**
 * Örnek başı (Python _run döngüsü): giriş kuralları ve başlatma.
 * @return 0: başlatılmadı (çıktı: son GNSS düzeltmesi, başlık / hız NaN), 1: bu örnekte başlatıldı (st, p0 dolu;
 *         çağıran durumu kurar), 2: başlatılmış, normal adım (in dolu)
 */
static inline int spt_pos_begin(spt_pos_core_t *c, const spt_pos_cfg_t *cfg, const SPT_REAL gnss[2],
                                const double gnss_ll[2], SPT_REAL gyro_z, SPT_REAL speed, SPT_REAL acc_fwd,
                                SPT_REAL T, spt_pos_in_t *in, SPT_REAL st[4], SPT_REAL p0[6])
{
    uint32_t kabs = c->k;
    SPT_REAL zx = SPT_NAN, zy = SPT_NAN;
    c->k = kabs + 1u;
    if (gyro_z > -SPT_POS_BIG && gyro_z < SPT_POS_BIG) {
        c->w_last = gyro_z;
        c->has_w = 1;
        in->gz = gyro_z;
        in->gmiss = 0;
    } else {
        in->gz = c->has_w ? c->w_last : SPT_C(0.0);
        in->gmiss = 1;
    }
#if SPT_POS_HAS_GNSS
    (void)gnss_ll;                                   /* ikisi de eşlenirse metre girişi kullanılır (Python gibi) */
    if (gnss != NULL) {
        zx = gnss[0];
        zy = gnss[1];
    }
#else
    (void)gnss;
    if (gnss_ll != NULL && spt_finite_d(gnss_ll[0]) && spt_finite_d(gnss_ll[1])) {
        spt_pos_ll_to_local(c, gnss_ll[0], gnss_ll[1], &zx, &zy);
    }
#endif
    in->zx = zx;
    in->zy = zy;
    in->fix = zx > -SPT_POS_BIG && zx < SPT_POS_BIG && zy > -SPT_POS_BIG && zy < SPT_POS_BIG;
#if SPT_POS_HAS_SPEED
    in->has_spd = speed > -SPT_POS_BIG && speed < SPT_POS_BIG;
    in->spd = speed;
#else
    (void)speed;
    in->has_spd = 0;
    in->spd = SPT_C(0.0);
#endif
#if SPT_POS_HAS_ACC_FWD
    in->has_acc = acc_fwd > -SPT_POS_BIG && acc_fwd < SPT_POS_BIG;
    in->acc = acc_fwd;
#else
    (void)acc_fwd;
    in->has_acc = 0;
    in->acc = SPT_C(0.0);
#endif
    if (in->fix) {
        c->n_gnss++;
    }
    if (!c->ready) {
        if (spt_pos_pre_init(c, cfg, kabs, in, T, st, p0)) {
            c->ready = 1;
            c->init_k = kabs;
            c->out[0] = st[0];
            c->out[1] = st[1];
            c->out[2] = st[2];
            c->out[3] = st[3];
            return 1;
        }
        c->out[0] = c->has_last_fix ? c->last_fix[0] : SPT_NAN;
        c->out[1] = c->has_last_fix ? c->last_fix[1] : SPT_NAN;
        c->out[2] = SPT_NAN;
        c->out[3] = SPT_NAN;
        return 0;
    }
    return 2;
}

#endif /* SPT_POS_COMMON_H */
"""

# ---------------------------------------------------------------------------- doğrusal ortak başlık
_LIN_COMMON_H = r"""/**
 * @file spt_lin_common.h
 * @brief Doğrusal çok sensörlü (1–6 yuva: konum / hız / ivme) senaryonun ortak parçaları: senaryo parametreleri,
 *        ölçüm toplama, başlatma ve ayrık model (F, Q).
 *
 * {{BANNER}}
 * Python karşılığı: spt/dsp/fusion_scenarios/linear_multi.py (_LinearBase, model_matrices).
 *
 * Model (durum x = [p, ṗ, p̈][:d], d = derece + 1): F[i][j] = T^(j−i)/(j−i)!, en yüksek türevin türevi beyaz gürültü
 * (yoğunluk σ_q²), Q[i][j] = σ_q²·T^(2d−1−i−j) / ((2d−1−i−j)·(d−1−i)!·(d−1−j)!). Ölçüm: z = x[tür] + n, R = σ_yuva².
 * Örnekte hangi yuvalar varsa (NaN olmayan) sıralı skaler güncelleme. Çıktılar [konum, hız, ivme]; modelin
 * kapsamadığı türevler 0.
 * Başlatma: konum yuvası eşlenmişse ilk konum ölçümünün geldiği örnekte (aynı türdeki ölçümlerin ters varyans
 * ortalaması); hız / ivme o örnekte ölçüldüyse onlar, değilse 0 ± σ0. Konum yuvası yoksa konum göreli (0, varyans 0).
 */
#ifndef SPT_LIN_COMMON_H
#define SPT_LIN_COMMON_H

#include "spt_fusion_generic_common.h"

/* ---- Senaryo parametreleri: derleme anı varsayılanları (#ifndef korumalı; defaults() ile s.cfg'ye kopyalanır) ---- */
{{PARAM_DEFINES}}

/* ---- Eşlenen yuvalar (bit i → s(i+1); üretim anında sabit, API yalnız bunları alır — değiştirmeyin) */
#define SPT_LIN_MAPPED {{MAPPED}}u

{{CFG_STRUCT}}
{{CORE_STRUCT}}
/** Bir örneğin ölçümleri (yuva sırasıyla; yalnız adım içinde). */
typedef struct spt_lin_meas {
    int m;                 /* ölçüm sayısı */
    int lvl[6];            /* tür seviyesi: 0 konum, 1 hız, 2 ivme */
    SPT_REAL z[6];         /* ölçüm */
    SPT_REAL r[6];         /* R = σ² */
} spt_lin_meas_t;

/** Senaryo parametrelerini derleme anı varsayılanlarına getirir. */
static inline void spt_lin_cfg_defaults(spt_lin_cfg_t *c)
{
{{CFG_DEFAULTS}}
}

/** Senaryo durumunu sıfırlar ve parametreleri denetler (Python _setup / _check_inputs: eşlenen her yuvanın türü ≤
 *  model derecesi; dmax: durum dizisi boyutu). @return 1: geçerli */
static inline int spt_lin_core_reset(spt_lin_core_t *c, const spt_lin_cfg_t *cfg, int dmax)
{
    int i;
    c->ok = cfg->order >= 0 && cfg->order <= 2 && cfg->order + 1 <= dmax;
    c->has_pos = 0;
    for (i = 0; i < 6; ++i) {
        if ((SPT_LIN_MAPPED >> i) & 1u) {
            if (cfg->kind[i] < 0 || cfg->kind[i] > cfg->order) {
                c->ok = 0;
            }
            if (cfg->kind[i] == 0) {
                c->has_pos = 1;
            }
        }
        c->n_meas[i] = 0u;
    }
    c->d = cfg->order + 1;
    c->out[0] = SPT_NAN;
    c->out[1] = SPT_NAN;
    c->out[2] = SPT_NAN;
    c->ready = 0;
    return c->ok;
}

/** Bu örnekteki ölçümler (Python _measurements): eşlenen yuvalar sırasıyla, geçerli (sonlu) değerler. */
static inline void spt_lin_measurements(spt_lin_core_t *c, const spt_lin_cfg_t *cfg, const SPT_REAL zin[6],
                                        spt_lin_meas_t *m)
{
    int i;
    m->m = 0;
    for (i = 0; i < 6; ++i) {
        if (((SPT_LIN_MAPPED >> i) & 1u) && zin[i] > -SPT_HUGE && zin[i] < SPT_HUGE) {
            m->lvl[m->m] = cfg->kind[i];
            m->z[m->m] = zin[i];
            m->r[m->m] = cfg->sigma[i] * cfg->sigma[i];
            m->m++;
            c->n_meas[i]++;
        }
    }
}

/** Aynı türdeki ölçümlerin ters varyans ortalaması: *num = Σ z/R, *info = Σ 1/R. @return o türde ölçüm var mı */
static inline int spt_lin_level_mean(const spt_lin_meas_t *m, int lvl, SPT_REAL *mean, SPT_REAL *info)
{
    SPT_REAL num = SPT_C(0.0), inf = SPT_C(0.0);
    int j, any = 0;
    for (j = 0; j < m->m; ++j) {
        if (m->lvl[j] == lvl) {
            inf += SPT_C(1.0) / m->r[j];
            any = 1;
        }
    }
    if (!any) {
        return 0;
    }
    for (j = 0; j < m->m; ++j) {
        if (m->lvl[j] == lvl) {
            num += m->z[j] / m->r[j];
        }
    }
    *mean = num / inf;
    *info = inf;
    return 1;
}

/** Başlangıç durumu ve köşegen varyansı (Python _initial). @return 0: beklenir (konum yuvası var, ölçümü yok) */
static inline int spt_lin_initial(const spt_lin_core_t *c, const spt_lin_cfg_t *cfg, const spt_lin_meas_t *m,
                                  SPT_REAL x0[3], SPT_REAL p0[3])
{
    int l;
    if (!m->m) {
        return 0;
    }
    for (l = 0; l < 3; ++l) {
        x0[l] = SPT_C(0.0);
        p0[l] = cfg->sigma0 * cfg->sigma0;
    }
    for (l = 0; l < c->d; ++l) {
        SPT_REAL mean, info;
        if (spt_lin_level_mean(m, l, &mean, &info)) {
            x0[l] = mean;
            p0[l] = SPT_C(1.0) / info;
        } else if (l == 0) {
            if (c->has_pos) {
                return 0;
            }
            p0[0] = SPT_C(0.0);       /* konum yuvası yok: göreli konum (ilk ölçüm anında 0) */
        }
    }
    return 1;
}

/** Ayrık F ve Q (d = 1…3; Python model_matrices). */
static inline void spt_lin_model(int d, SPT_REAL T, SPT_REAL sigma_q, SPT_REAL F[3][3], SPT_REAL Q[3][3])
{
    SPT_REAL q2 = sigma_q * sigma_q, T2 = T * T, T3 = T2 * T;
    int i, j;
    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            F[i][j] = (i == j) ? SPT_C(1.0) : SPT_C(0.0);
            Q[i][j] = SPT_C(0.0);
        }
    }
    F[0][1] = T;
    F[1][2] = T;
    F[0][2] = T2 * SPT_C(0.5);
    if (d == 1) {
        Q[0][0] = q2 * T;
    } else if (d == 2) {
        Q[0][0] = q2 * T3 * SPT_C(0.33333333333333333333);
        Q[0][1] = q2 * T2 * SPT_C(0.5);
        Q[1][1] = q2 * T;
    } else {
        Q[0][0] = q2 * (T3 * T2) * SPT_C(0.05);
        Q[0][1] = q2 * (T2 * T2) * SPT_C(0.125);
        Q[0][2] = q2 * T3 * SPT_C(0.16666666666666666667);
        Q[1][1] = q2 * T3 * SPT_C(0.33333333333333333333);
        Q[1][2] = q2 * T2 * SPT_C(0.5);
        Q[2][2] = q2 * T;
    }
    Q[1][0] = Q[0][1];
    Q[2][0] = Q[0][2];
    Q[2][1] = Q[1][2];
}

/** Çıktılar [konum, hız, ivme]: modelin kapsamadığı türevler 0 (Python _out). */
static inline void spt_lin_set_out(spt_lin_core_t *c, const SPT_REAL x[3])
{
    c->out[0] = x[0];
    c->out[1] = (c->d > 1) ? x[1] : SPT_C(0.0);
    c->out[2] = (c->d > 2) ? x[2] : SPT_C(0.0);
}

#endif /* SPT_LIN_COMMON_H */
"""

# ---------------------------------------------------------------------------- algoritma kaynakları (çekirdek fonksiyon)
# Çekirdek imzaları (tüm girişler; eşlenmeyenler NaN / NULL ile çağrılır — derleyici o dalları atar):
#   alt: (s, baro_p, baro_alt, acc_z, dt)   pos: (s, gnss, gnss_ll, gyro_z, speed, acc_fwd, dt)   lin: (s, z[6], dt)
_SOURCES: dict[tuple[str, str], str] = {}

_SOURCES[("altitude", "complementary")] = r"""
void spt_alt_complementary_defaults(spt_alt_complementary_t *s)
{
    spt_alt_cfg_defaults(&s->cfg);
{{SET_DEFAULTS}}
    (void)spt_alt_complementary_reset(s);
}

int spt_alt_complementary_reset(spt_alt_complementary_t *s)
{
    s->h = SPT_C(0.0);
    s->v = SPT_C(0.0);
    s->b = SPT_C(0.0);
    s->k_last = 0u;
    return spt_alt_core_reset(&s->c, &s->cfg);
}

/* Python AltitudeComplementary._run, bir örnek. e = z_h − ĥ ile ĥ += k1·e·T_b, v̂ += k2·e·T_b, b̂ −= k3·e·T_b. */
static int spt_alt_complementary_core(spt_alt_complementary_t *s, SPT_REAL baro_p, SPT_REAL baro_alt, SPT_REAL acc_z,
                                      SPT_REAL dt)
{
    spt_alt_core_t *c = &s->c;
    SPT_REAL f, z = SPT_C(0.0), a, it, k1, k2, k3, T2 = SPT_C(0.5) * dt * dt;
    uint32_t kabs = c->k;
    int kind;
    if (!c->ok) {
        return 0;
    }
    c->k = kabs + 1u;
    if (spt_alt_accel(c, acc_z, &f) == 2) {
        f = s->cfg.g + s->b;                        /* hiç ivme gelmedi: a = 0 */
    }
    kind = spt_alt_baro(c, baro_p, baro_alt, &z);
    if (!c->ready) {
        if (!kind) {
            return 0;
        }
        s->h = spt_alt_h_of(c, kind, z);
        s->v = SPT_C(0.0);
        s->b = SPT_C(0.0);
        s->k_last = kabs;
        c->ready = 1;
        c->n_baro++;
        c->out[0] = s->h;
        c->out[1] = s->v;
        return 1;
    }
    it = SPT_C(1.0) / s->tau;
    if (s->bias) {                                   /* üçlü kutup −1/τ: k = (3/τ, 3/τ², 1/τ³) */
        k1 = SPT_C(3.0) * it;
        k2 = k1 * it;
        k3 = it * it * it;
    } else {                                         /* (2/τ, 1/τ², 0) */
        k1 = SPT_C(2.0) * it;
        k2 = it * it;
        k3 = SPT_C(0.0);
    }
    a = f - s->cfg.g - s->b;
    s->h += s->v * dt + a * T2;
    s->v += a * dt;
    if (kind) {
        SPT_REAL e = spt_alt_h_of(c, kind, z) - s->h;
        SPT_REAL tb = SPT_FROM_INT(kabs - s->k_last) * dt, tmax = s->tau / SPT_C(3.0);
        if (tb > tmax) {
            tb = tmax;
        }
        s->h += k1 * e * tb;
        s->v += k2 * e * tb;
        s->b -= k3 * e * tb;
        s->k_last = kabs;
        c->n_baro++;
    }
    c->out[0] = s->h;
    c->out[1] = s->v;
    return 1;
}

SPT_REAL spt_alt_complementary_acc_bias(const spt_alt_complementary_t *s)
{
    return s->b;
}
"""

_ALT_KALMAN_COMMON = r"""
void spt_{{A}}_defaults(spt_{{A}}_t *s)
{
    spt_alt_cfg_defaults(&s->cfg);
    (void)spt_{{A}}_reset(s);
}

int spt_{{A}}_reset(spt_{{A}}_t *s)
{
    int i;
    for (i = 0; i < 3; ++i) {
        s->x[i] = SPT_C(0.0);
    }
    for (i = 0; i < 6; ++i) {
        s->P[i] = SPT_C(0.0);
    }
    return spt_alt_core_reset(&s->c, &s->cfg);
}
{{UPDATE}}
/* Python _AltitudeKalman._run, bir örnek: öngörü (kapalı biçim) + baro ölçümü varsa {{UPDATE_NAME}} güncellemesi. */
static int spt_{{A}}_core(spt_{{A}}_t *s, SPT_REAL baro_p, SPT_REAL baro_alt, SPT_REAL acc_z, SPT_REAL dt)
{
    spt_alt_core_t *c = &s->c;
    SPT_REAL f, z = SPT_C(0.0);
    int miss, kind;
    if (!c->ok) {
        return 0;
    }
    c->k++;
    miss = spt_alt_accel(c, acc_z, &f);
    if (miss == 2) {
        f = s->cfg.g + s->x[2];                     /* hiç ivme gelmedi: a = 0 */
    }
    kind = spt_alt_baro(c, baro_p, baro_alt, &z);
    if (!c->ready) {
        SPT_REAL r0, h0;
        if (!kind) {
            return 0;
        }
        h0 = spt_alt_of(c, &s->cfg, kind, z, &r0);
        spt_alt_kalman_start(s->x, s->P, &s->cfg, h0, r0);
        c->ready = 1;
        c->n_baro++;
        c->out[0] = s->x[0];
        c->out[1] = s->x[1];
        return 1;
    }
    spt_alt_kalman_predict(s->x, s->P, &s->cfg, f, miss != 0, dt);
    if (kind) {
        spt_{{A}}_measure(s, kind, z);
        c->n_baro++;
    }
    c->out[0] = s->x[0];
    c->out[1] = s->x[1];
    return 1;
}

SPT_REAL spt_{{A}}_acc_bias(const spt_{{A}}_t *s)
{
    return s->x[2];
}
"""

_ALT_UPDATES = {
    "kf": r"""
/* KF: basınç önce ISA ile irtifaya çevrilir, R = (σ_p / |dp/dh|)²; H = [1, 0, 0]. */
static void spt_alt_kf_measure(spt_alt_kf_t *s, int kind, SPT_REAL z)
{
    SPT_REAL r, h = spt_alt_of(&s->c, &s->cfg, kind, z, &r);
    spt_alt_linear_update(s->x, s->P, h, SPT_C(1.0), s->x[0], r);
}
""",
    "ekf": r"""
/* EKF: basınç doğrudan ölçüm, H = dp/dh (ĥ'de), R = σ_p²; baro irtifada KF ile aynı. */
static void spt_alt_ekf_measure(spt_alt_ekf_t *s, int kind, SPT_REAL z)
{
    if (kind == 1) {
        SPT_REAL p0 = s->c.p0;
        SPT_REAL hh = spt_isa_dpdh(s->x[0], p0), pred = spt_isa_pressure(s->x[0], p0);
        spt_alt_linear_update(s->x, s->P, z, hh, pred, s->cfg.sigma_p * s->cfg.sigma_p);
    } else {
        spt_alt_linear_update(s->x, s->P, z, SPT_C(1.0), s->x[0], s->cfg.sigma_h * s->cfg.sigma_h);
    }
}
""",
    "ukf": r"""
/* UKF: basınç ölçümü kokusuz dönüşümle (2n + 1 = 7 sigma noktası, α = 1, β = 2, κ = 0 → Wm = (0, 1/6 …),
   Wc = (2, 1/6 …)); √(3·P) alt üçgen Cholesky (pivot 1e-300 ile sınırlı). Süreç doğrusal: öngörü KF ile aynı. */
static void spt_alt_ukf_measure(spt_alt_ukf_t *s, int kind, SPT_REAL z)
{
    SPT_REAL *P = s->P, p0 = s->c.p0, h = s->x[0];
    SPT_REAL l00, l10, l20, l11, l21, l22, t, cols[3][3], devs[6][3], zs[6], dzs[6], z0, zm, dz0, pzz, pxz[3];
    SPT_REAL k0, k1, k2, y;
    int i, j;
    if (kind != 1) {                                 /* doğrusal ölçümde UT güncellemesi KF'nin aynısıdır */
        spt_alt_linear_update(s->x, P, z, SPT_C(1.0), h, s->cfg.sigma_h * s->cfg.sigma_h);
        return;
    }
    t = SPT_C(3.0) * P[0];
    l00 = SPT_SQRT(t > SPT_TINY ? t : SPT_TINY);
    l10 = SPT_C(3.0) * P[1] / l00;
    l20 = SPT_C(3.0) * P[2] / l00;
    t = SPT_C(3.0) * P[3] - l10 * l10;
    l11 = SPT_SQRT(t > SPT_TINY ? t : SPT_TINY);
    l21 = (SPT_C(3.0) * P[4] - l20 * l10) / l11;
    t = SPT_C(3.0) * P[5] - l20 * l20 - l21 * l21;
    l22 = SPT_SQRT(t > SPT_TINY ? t : SPT_TINY);
    cols[0][0] = l00;
    cols[0][1] = l10;
    cols[0][2] = l20;
    cols[1][0] = SPT_C(0.0);
    cols[1][1] = l11;
    cols[1][2] = l21;
    cols[2][0] = SPT_C(0.0);
    cols[2][1] = SPT_C(0.0);
    cols[2][2] = l22;
    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            devs[i][j] = cols[i][j];
            devs[3 + i][j] = -cols[i][j];
        }
    }
    z0 = spt_isa_pressure(h, p0);
    zm = SPT_C(0.0);
    for (i = 0; i < 6; ++i) {                        /* h + 0 = h: yalnız ilk sigma çifti basıncı değiştirir */
        zs[i] = (devs[i][0] == SPT_C(0.0)) ? z0 : spt_isa_pressure(h + devs[i][0], p0);
        zm += zs[i];
    }
    zm = zm / SPT_C(6.0);
    dz0 = z0 - zm;
    pzz = SPT_C(0.0);
    for (i = 0; i < 6; ++i) {
        dzs[i] = zs[i] - zm;
        pzz += dzs[i] * dzs[i];
    }
    pzz = SPT_C(2.0) * dz0 * dz0 + pzz / SPT_C(6.0) + s->cfg.sigma_p * s->cfg.sigma_p;
    for (j = 0; j < 3; ++j) {
        SPT_REAL acc = SPT_C(0.0);
        for (i = 0; i < 6; ++i) {
            acc += devs[i][j] * dzs[i];
        }
        pxz[j] = acc / SPT_C(6.0);
    }
    k0 = pxz[0] / pzz;
    k1 = pxz[1] / pzz;
    k2 = pxz[2] / pzz;
    y = z - zm;
    s->x[0] += k0 * y;
    s->x[1] += k1 * y;
    s->x[2] += k2 * y;
    P[0] -= k0 * k0 * pzz;
    P[1] -= k0 * k1 * pzz;
    P[2] -= k0 * k2 * pzz;
    P[3] -= k1 * k1 * pzz;
    P[4] -= k1 * k2 * pzz;
    P[5] -= k2 * k2 * pzz;
}
""",
}
for _k in ("kf", "ekf", "ukf"):
    _SOURCES[("altitude", _k)] = _ALT_KALMAN_COMMON.replace("{{UPDATE}}", _ALT_UPDATES[_k]) \
        .replace("{{UPDATE_NAME}}", _k.upper()).replace("{{A}}", f"alt_{_k}")

_SOURCES[("altitude", "pf")] = r"""
void spt_alt_pf_defaults(spt_alt_pf_t *s)
{
    spt_alt_cfg_defaults(&s->cfg);
{{SET_DEFAULTS}}
    (void)spt_alt_pf_reset(s);
}

int spt_alt_pf_reset(spt_alt_pf_t *s)
{
    spt_rng_seed(&s->rng, s->seed);
    s->ess = SPT_C(0.0);
    s->resamples = 0u;
    return spt_alt_core_reset(&s->c, &s->cfg);
}

static void spt_alt_pf_out(spt_alt_pf_t *s)
{
    SPT_REAL h = SPT_C(0.0), v = SPT_C(0.0);
    int i;
    for (i = 0; i < SPT_ALT_PF_PARTICLES; ++i) {
        h += s->w[i] * s->X[i][0];
        v += s->w[i] * s->X[i][1];
    }
    s->c.out[0] = h;
    s->c.out[1] = v;
}

/* Python AltitudePF._run, bir örnek: (h, v, b) parçacıkları; olabilirlik basınçta / irtifada Gauss; ESS < N/2 ise
   Liu–West yumuşatmalı sistematik yeniden örnekleme. */
static int spt_alt_pf_core(spt_alt_pf_t *s, SPT_REAL baro_p, SPT_REAL baro_alt, SPT_REAL acc_z, SPT_REAL dt)
{
    const int n = SPT_ALT_PF_PARTICLES;
    spt_alt_core_t *c = &s->c;
    const spt_alt_cfg_t *cfg = &s->cfg;
    SPT_REAL f, z = SPT_C(0.0), T2 = SPT_C(0.5) * dt * dt, sa, sb, fg;
    int miss, kind, i;
    if (!c->ok) {
        return 0;
    }
    c->k++;
    miss = spt_alt_accel(c, acc_z, &f);
    kind = spt_alt_baro(c, baro_p, baro_alt, &z);
    if (!c->ready) {
        SPT_REAL r0, h0, sr;
        if (!kind) {
            return 0;
        }
        h0 = spt_alt_of(c, cfg, kind, z, &r0);
        sr = SPT_SQRT(r0);
        for (i = 0; i < n; ++i) {
            SPT_REAL *x = s->X[i];
            x[0] = h0 + sr * spt_rng_normal(&s->rng);
            x[1] = SPT_ALT_V0_SIGMA * spt_rng_normal(&s->rng);
            x[2] = cfg->bias0 * spt_rng_normal(&s->rng);
        }
        s->ess = spt_pf_uniform(s->w, n);
        c->ready = 1;
        c->n_baro++;
        spt_alt_pf_out(s);
        return 1;
    }
    /* ---- öngörü: a = (f − g) − b + σ·e (hiç ivme yoksa a = σ·e), sapma yürüyüşü */
    sa = miss ? SPT_SQRT(cfg->sigma_acc * cfg->sigma_acc + SPT_ALT_MISS_ACC2 / dt) : cfg->sigma_acc;
    sb = cfg->sigma_bias * SPT_SQRT(dt);
    fg = f - cfg->g;
    for (i = 0; i < n; ++i) {
        SPT_REAL *x = s->X[i];
        SPT_REAL e0 = spt_rng_normal(&s->rng), e1 = spt_rng_normal(&s->rng);
        SPT_REAL a = (miss == 2) ? sa * e0 : fg - x[2] + sa * e0;
        x[0] += x[1] * dt + a * T2;
        x[1] += a * dt;
        x[2] += sb * e1;
    }
    /* ---- baro olabilirliği */
    if (kind == 1) {                                 /* basınç: log-ağırlıklarla (olabilirlik pahalı: tek geçiş) */
        SPT_REAL is = SPT_C(1.0) / cfg->sigma_p;
        for (i = 0; i < n; ++i) {
            SPT_REAL r = (z - spt_isa_pressure(s->X[i][0], c->p0)) * is;
            SPT_REAL w = (s->w[i] < SPT_TINY) ? SPT_TINY : s->w[i];
            s->w[i] = SPT_LOG(w) - SPT_C(0.5) * r * r;
        }
        s->ess = spt_pf_from_log(s->w, n);
    } else if (kind == 2) {                          /* baro irtifa: iki geçiş */
        SPT_REAL is = SPT_C(1.0) / cfg->sigma_h, lmax = SPT_C(0.0), sum = SPT_C(0.0), sum2 = SPT_C(0.0);
        int any = 0;
        for (i = 0; i < n; ++i) {
            SPT_REAL r = (z - s->X[i][0]) * is;
            spt_pf_max(SPT_C(-0.5) * r * r, &lmax, &any);
        }
        if (any) {
            for (i = 0; i < n; ++i) {
                SPT_REAL r = (z - s->X[i][0]) * is;
                spt_pf_mul(&s->w[i], SPT_C(-0.5) * r * r, lmax, &sum, &sum2);
            }
            s->ess = spt_pf_scale(s->w, n, sum, sum2);
        } else {
            s->ess = spt_pf_uniform(s->w, n);
        }
    }
    if (kind) {
        if (s->ess < SPT_C(0.5) * SPT_FROM_INT(n)) {
            spt_lw_resample(&s->X[0][0], n, 3, 3, s->w, s->cnt, &s->rng, -1, s->work);
            s->resamples++;
        }
        c->n_baro++;
    }
    spt_alt_pf_out(s);
    return 1;
}

SPT_REAL spt_alt_pf_acc_bias(const spt_alt_pf_t *s)
{
    SPT_REAL b = SPT_C(0.0);
    int i;
    for (i = 0; i < SPT_ALT_PF_PARTICLES; ++i) {
        b += s->w[i] * s->X[i][2];
    }
    return b;
}
"""

# ---------------------------------------------------------------------------- 2B konum
_POS_HEAD = r"""
void spt_{{A}}_defaults(spt_{{A}}_t *s)
{
    spt_pos_cfg_defaults(&s->cfg);
{{SET_DEFAULTS}}
    (void)spt_{{A}}_reset(s);
}
"""

_SOURCES[("position2d", "dead_reckoning")] = _POS_HEAD + r"""
int spt_pos_dead_reckoning_reset(spt_pos_dead_reckoning_t *s)
{
    int i;
    for (i = 0; i < 4; ++i) {
        s->x[i] = SPT_C(0.0);
    }
    return spt_pos_core_reset(&s->c);
}

/* Python Pos2DDeadReckoning: yalnız integral (başlatmadan sonra GNSS kullanılmaz, sapma / ölçek düzeltilmez). */
static int spt_pos_dead_reckoning_core(spt_pos_dead_reckoning_t *s, const SPT_REAL gnss[2], const double gnss_ll[2],
                                       SPT_REAL gyro_z, SPT_REAL speed, SPT_REAL acc_fwd, SPT_REAL dt)
{
    spt_pos_in_t in;
    SPT_REAL st[4], p0[6], pm, *x = s->x;
    int r, i;
    if (!s->c.ok) {
        return 0;
    }
    r = spt_pos_begin(&s->c, &s->cfg, gnss, gnss_ll, gyro_z, speed, acc_fwd, dt, &in, st, p0);
    if (r == 0) {
        return 0;
    }
    if (r == 1) {
        for (i = 0; i < 4; ++i) {
            x[i] = st[i];
        }
        return 1;
    }
    if (in.has_spd) {
        x[3] = in.spd;
    } else if (in.has_acc) {
        x[3] += in.acc * dt;
    }
    pm = x[2] + SPT_C(0.5) * in.gz * dt;
    x[0] += x[3] * dt * SPT_COS(pm);
    x[1] += x[3] * dt * SPT_SIN(pm);
    x[2] = spt_wrap(x[2] + in.gz * dt);
    for (i = 0; i < 4; ++i) {
        s->c.out[i] = x[i];
    }
    return 1;
}
"""

_POS_KF_COMMON = r"""
int spt_{{A}}_reset(spt_{{A}}_t *s)
{
    int i, j;
    for (i = 0; i < 6; ++i) {
        s->x[i] = SPT_C(0.0);
        for (j = 0; j < 6; ++j) {
            s->P[i][j] = SPT_C(0.0);
        }
    }{{RESET_EXTRA}}
    return spt_pos_core_reset(&s->c);
}

/* Başlangıç (Python _start): x = [x, y, ψ, v, 0, 1], P = diag(p0). */
static void spt_{{A}}_start(spt_{{A}}_t *s, const SPT_REAL st[4], const SPT_REAL p0[6])
{
    int i, j;
    for (i = 0; i < 4; ++i) {
        s->x[i] = st[i];
    }
    s->x[4] = SPT_C(0.0);
    s->x[5] = SPT_C(1.0);
    for (i = 0; i < 6; ++i) {
        for (j = 0; j < 6; ++j) {
            s->P[i][j] = (i == j) ? p0[i] : SPT_C(0.0);
        }
    }
}

/** Süreç gürültüsü köşegeni (Python _qdiag). */
static void spt_{{A}}_qdiag(const spt_{{A}}_t *s, SPT_REAL T, const spt_pos_in_t *in, SPT_REAL qd[6])
{
    spt_pos_noise_t q;
    spt_pos_noise(&s->cfg, T, &q);
    qd[0] = q.pos;
    qd[1] = q.pos;
    qd[2] = in->gmiss ? q.psi_miss : q.psi;
    qd[3] = in->has_acc ? q.v_acc : q.v_rw;
    qd[4] = q.bg;
    qd[5] = q.s;
}

SPT_REAL spt_{{A}}_gyro_bias(const spt_{{A}}_t *s)
{
    return s->x[4];
}

SPT_REAL spt_{{A}}_odo_scale(const spt_{{A}}_t *s)
{
    return s->x[5];
}
"""

_SOURCES[("position2d", "ekf")] = _POS_HEAD + _POS_KF_COMMON.replace("{{RESET_EXTRA}}", "") + r"""
/** P ← P − kg·phᵀ (üst üçgen + ayna; Python sıralı skaler güncelleme). */
static void spt_pos_ekf_correct(spt_pos_ekf_t *s, const SPT_REAL ph[6], SPT_REAL denom, SPT_REAL innov)
{
    SPT_REAL kg[6], is = SPT_C(1.0) / denom;
    int a, b;
    for (a = 0; a < 6; ++a) {
        kg[a] = ph[a] * is;
        s->x[a] += kg[a] * innov;
    }
    for (a = 0; a < 6; ++a) {
        for (b = a; b < 6; ++b) {
            s->P[a][b] -= kg[a] * ph[b];
            s->P[b][a] = s->P[a][b];
        }
    }
}

/* Python Pos2DEKF._step: Jacobian kapalı biçim; F·P·Fᵀ yalnız F'nin I'dan farklı 7 elemanıyla (seyrek). */
static int spt_pos_ekf_core(spt_pos_ekf_t *s, const SPT_REAL gnss[2], const double gnss_ll[2], SPT_REAL gyro_z,
                            SPT_REAL speed, SPT_REAL acc_fwd, SPT_REAL dt)
{
    SPT_REAL (*P)[6] = s->P, *x = s->x;
    spt_pos_in_t in;
    SPT_REAL st[4], p0[6], qd[6], M[3][6], ph[6];
    SPT_REAL w, pm, c, sn, vT, a02, a03, a04, a12, a13, a14, a24, v;
    int r, i, j;
    if (!s->c.ok) {
        return 0;
    }
    r = spt_pos_begin(&s->c, &s->cfg, gnss, gnss_ll, gyro_z, speed, acc_fwd, dt, &in, st, p0);
    if (r == 0) {
        return 0;
    }
    if (r == 1) {
        spt_pos_ekf_start(s, st, p0);
        return 1;
    }
    /* ---- öngörü */
    v = x[3];
    w = in.gz - x[4];
    pm = x[2] + SPT_C(0.5) * w * dt;
    c = SPT_COS(pm);
    sn = SPT_SIN(pm);
    vT = v * dt;
    a02 = -vT * sn;
    a03 = dt * c;
    a04 = SPT_C(0.5) * vT * dt * sn;
    a12 = vT * c;
    a13 = dt * sn;
    a14 = SPT_C(-0.5) * vT * dt * c;
    a24 = -dt;
    x[0] += vT * c;
    x[1] += vT * sn;
    x[2] += w * dt;
    if (in.has_acc) {
        x[3] = v + in.acc * dt;
    }
    for (j = 0; j < 6; ++j) {                        /* M = F·P, satır 0–2 (3–5 = P) */
        M[0][j] = P[0][j] + a02 * P[2][j] + a03 * P[3][j] + a04 * P[4][j];
        M[1][j] = P[1][j] + a12 * P[2][j] + a13 * P[3][j] + a14 * P[4][j];
        M[2][j] = P[2][j] + a24 * P[4][j];
    }
    P[0][0] = M[0][0] + a02 * M[0][2] + a03 * M[0][3] + a04 * M[0][4];
    for (i = 0; i < 2; ++i) {
        P[i][1] = M[i][1] + a12 * M[i][2] + a13 * M[i][3] + a14 * M[i][4];
    }
    for (i = 0; i < 3; ++i) {
        P[i][2] = M[i][2] + a24 * M[i][4];
        for (j = 3; j < 6; ++j) {
            P[i][j] = M[i][j];
        }
    }
    spt_pos_ekf_qdiag(s, dt, &in, qd);
    for (i = 0; i < 6; ++i) {
        P[i][i] += qd[i];
        for (j = i + 1; j < 6; ++j) {
            P[j][i] = P[i][j];
        }
    }
    /* ---- GNSS: x ve y sıralı skaler güncelleme, R = σ_gnss² */
    if (in.fix) {
        SPT_REAL rr = s->cfg.sigma_gnss * s->cfg.sigma_gnss;
        for (i = 0; i < 2; ++i) {
            for (j = 0; j < 6; ++j) {
                ph[j] = P[j][i];
            }
            spt_pos_ekf_correct(s, ph, ph[i] + rr, (i == 0 ? in.zx : in.zy) - x[i]);
        }
    }
    /* ---- odometri: z = s·v, H = [0, 0, 0, s, 0, v] */
    if (in.has_spd) {
        SPT_REAL sc = x[5], vv = x[3];
        for (j = 0; j < 6; ++j) {
            ph[j] = P[j][3] * sc + P[j][5] * vv;
        }
        spt_pos_ekf_correct(s, ph, sc * ph[3] + vv * ph[5] + s->cfg.sigma_speed * s->cfg.sigma_speed,
                            in.spd - sc * vv);
    }
    x[2] = spt_wrap(x[2]);
    for (i = 0; i < 4; ++i) {
        s->c.out[i] = x[i];
    }
    return 1;
}
"""

_SOURCES[("position2d", "ukf")] = _POS_HEAD + _POS_KF_COMMON.replace(
    "{{RESET_EXTRA}}", "\n    s->fallbacks = 0u;") + r"""
#define SPT_POS_UKF_W  SPT_C(0.083333333333333333333)   /* Wm = Wc = 1/12 (i ≥ 1) */
#define SPT_POS_UKF_W0 SPT_C(2.0)                        /* Wc0 (Wm0 = 0) */

/** Sigma noktaları (Python _sigma): X₀ = x, X₁₊ⱼ = x + Lⱼ, X₇₊ⱼ = x − Lⱼ, L = chol_psd(6·P). */
static void spt_pos_ukf_sigma(spt_pos_ukf_t *s)
{
    SPT_REAL *a6 = s->work;
    int i, j;
    for (i = 0; i < 6; ++i) {
        for (j = 0; j < 6; ++j) {
            a6[i * 6 + j] = SPT_C(6.0) * s->P[i][j];
        }
    }
    if (!spt_chol_psd(6, a6, &s->L[0][0], s->work + 36)) {
        s->fallbacks++;
    }
    for (i = 0; i < 6; ++i) {
        s->X[0][i] = s->x[i];
        for (j = 0; j < 6; ++j) {
            s->X[1 + j][i] = s->x[i] + s->L[i][j];
            s->X[7 + j][i] = s->x[i] - s->L[i][j];
        }
    }
}

/** P ← P − K·Pzz·Kᵀ ve x ← x + K·e (üst üçgen + ayna). K: 6×m, m ≤ 2. */
static void spt_pos_ukf_apply(spt_pos_ukf_t *s, int m, SPT_REAL K[6][2], SPT_REAL Pzz[2][2], const SPT_REAL e[2])
{
    SPT_REAL KP[6][2];
    int a, b, c;
    for (a = 0; a < 6; ++a) {
        SPT_REAL acc = SPT_C(0.0);
        for (c = 0; c < m; ++c) {
            acc += K[a][c] * e[c];
        }
        s->x[a] += acc;
        for (c = 0; c < m; ++c) {
            SPT_REAL t = SPT_C(0.0);
            for (b = 0; b < m; ++b) {
                t += K[a][b] * Pzz[b][c];
            }
            KP[a][c] = t;
        }
    }
    for (a = 0; a < 6; ++a) {
        for (b = a; b < 6; ++b) {
            SPT_REAL t = SPT_C(0.0);
            for (c = 0; c < m; ++c) {
                t += KP[a][c] * K[b][c];
            }
            s->P[a][b] -= t;
            s->P[b][a] = s->P[a][b];
        }
    }
}

/* Python Pos2DUKF._step: 13 sigma noktası (α = 1, β = 2, κ = 0), başlık dairesel ortalamayla; GNSS 2×2 ve odometri
   skaler kokusuz güncelleme. */
static int spt_pos_ukf_core(spt_pos_ukf_t *s, const SPT_REAL gnss[2], const double gnss_ll[2], SPT_REAL gyro_z,
                            SPT_REAL speed, SPT_REAL acc_fwd, SPT_REAL dt)
{
    SPT_REAL (*X)[6] = s->X, (*P)[6] = s->P, *x = s->x;
    spt_pos_in_t in;
    SPT_REAL st[4], p0[6], qd[6], ref, acc;
    int r, i, a, c;
    if (!s->c.ok) {
        return 0;
    }
    r = spt_pos_begin(&s->c, &s->cfg, gnss, gnss_ll, gyro_z, speed, acc_fwd, dt, &in, st, p0);
    if (r == 0) {
        return 0;
    }
    if (r == 1) {
        spt_pos_ukf_start(s, st, p0);
        return 1;
    }
    /* ---- öngörü: sigma noktaları modelden geçirilir */
    spt_pos_ukf_sigma(s);
    for (i = 0; i < 13; ++i) {
        SPT_REAL w = in.gz - X[i][4];
        SPT_REAL pm = X[i][2] + SPT_C(0.5) * w * dt, vT = X[i][3] * dt;
        X[i][0] += vT * SPT_COS(pm);
        X[i][1] += vT * SPT_SIN(pm);
        X[i][2] += w * dt;
        if (in.has_acc) {
            X[i][3] += in.acc * dt;
        }
    }
    for (a = 0; a < 6; ++a) {
        acc = SPT_C(0.0);
        for (i = 1; i < 13; ++i) {
            acc += X[i][a];
        }
        x[a] = acc * SPT_POS_UKF_W;
    }
    ref = X[0][2];                                   /* dairesel ortalama (merkez noktaya göre) */
    acc = SPT_C(0.0);
    for (i = 1; i < 13; ++i) {
        acc += spt_wrap(X[i][2] - ref);
    }
    x[2] = ref + acc * SPT_POS_UKF_W;
    for (i = 0; i < 13; ++i) {                       /* X ← D = X − x (başlık sarılı) */
        for (a = 0; a < 6; ++a) {
            X[i][a] -= x[a];
        }
        X[i][2] = spt_wrap(X[i][2]);
    }
    spt_pos_ukf_qdiag(s, dt, &in, qd);
    for (a = 0; a < 6; ++a) {
        for (c = a; c < 6; ++c) {
            SPT_REAL sum = SPT_C(0.0);
            for (i = 1; i < 13; ++i) {
                sum += X[i][a] * X[i][c];
            }
            P[a][c] = sum * SPT_POS_UKF_W + SPT_POS_UKF_W0 * X[0][a] * X[0][c];
            P[c][a] = P[a][c];
        }
        P[a][a] += qd[a];
    }
    /* ---- GNSS: Z = sigma noktalarının (x, y)'si, R = σ²·I */
    if (in.fix) {
        SPT_REAL zm[2], Pzz[2][2], Pi[2][2], K[6][2], e[2], dZ[13][2], det, rr = s->cfg.sigma_gnss * s->cfg.sigma_gnss;
        int b;
        spt_pos_ukf_sigma(s);
        for (b = 0; b < 2; ++b) {
            acc = SPT_C(0.0);
            for (i = 1; i < 13; ++i) {
                acc += X[i][b];
            }
            zm[b] = acc * SPT_POS_UKF_W;
        }
        for (i = 0; i < 13; ++i) {
            dZ[i][0] = X[i][0] - zm[0];
            dZ[i][1] = X[i][1] - zm[1];
        }
        for (b = 0; b < 2; ++b) {
            for (c = b; c < 2; ++c) {
                SPT_REAL sum = SPT_C(0.0);
                for (i = 1; i < 13; ++i) {
                    sum += dZ[i][b] * dZ[i][c];
                }
                Pzz[b][c] = sum * SPT_POS_UKF_W + SPT_POS_UKF_W0 * dZ[0][b] * dZ[0][c];
                Pzz[c][b] = Pzz[b][c];
            }
            Pzz[b][b] += rr;
        }
        det = Pzz[0][0] * Pzz[1][1] - Pzz[0][1] * Pzz[1][0];
        if (det != SPT_C(0.0) && spt_finite(det)) {
            SPT_REAL id = SPT_C(1.0) / det;
            Pi[0][0] = Pzz[1][1] * id;
            Pi[0][1] = -Pzz[0][1] * id;
            Pi[1][0] = -Pzz[1][0] * id;
            Pi[1][1] = Pzz[0][0] * id;
            for (a = 0; a < 6; ++a) {                /* Pxz (D₀ = 0) ve K = Pxz·Pzz⁻¹ */
                SPT_REAL p0x = SPT_C(0.0), p1x = SPT_C(0.0);
                for (i = 1; i < 13; ++i) {
                    SPT_REAL d = X[i][a] - x[a];
                    p0x += d * dZ[i][0];
                    p1x += d * dZ[i][1];
                }
                p0x *= SPT_POS_UKF_W;
                p1x *= SPT_POS_UKF_W;
                K[a][0] = p0x * Pi[0][0] + p1x * Pi[1][0];
                K[a][1] = p0x * Pi[0][1] + p1x * Pi[1][1];
            }
            e[0] = in.zx - zm[0];
            e[1] = in.zy - zm[1];
            spt_pos_ukf_apply(s, 2, K, Pzz, e);
        }
    }
    /* ---- odometri: Z = s·v (skaler) */
    if (in.has_spd) {
        SPT_REAL zs[13], zm, pzz, K[6][2], Pzz[2][2], e[2];
        spt_pos_ukf_sigma(s);
        acc = SPT_C(0.0);
        for (i = 0; i < 13; ++i) {
            zs[i] = X[i][5] * X[i][3];
            if (i > 0) {
                acc += zs[i];
            }
        }
        zm = acc * SPT_POS_UKF_W;
        acc = SPT_C(0.0);
        for (i = 0; i < 13; ++i) {
            zs[i] -= zm;
            if (i > 0) {
                acc += zs[i] * zs[i];
            }
        }
        pzz = acc * SPT_POS_UKF_W + SPT_POS_UKF_W0 * zs[0] * zs[0] + s->cfg.sigma_speed * s->cfg.sigma_speed;
        if (pzz != SPT_C(0.0)) {
            SPT_REAL ip = SPT_C(1.0) / pzz;
            for (a = 0; a < 6; ++a) {
                SPT_REAL sum = SPT_C(0.0);
                for (i = 1; i < 13; ++i) {
                    sum += zs[i] * (X[i][a] - x[a]);
                }
                K[a][0] = sum * SPT_POS_UKF_W * ip;
                K[a][1] = SPT_C(0.0);
            }
            Pzz[0][0] = pzz;
            Pzz[0][1] = SPT_C(0.0);
            Pzz[1][0] = SPT_C(0.0);
            Pzz[1][1] = SPT_C(0.0);
            e[0] = in.spd - zm;
            e[1] = SPT_C(0.0);
            spt_pos_ukf_apply(s, 1, K, Pzz, e);
        }
    }
    x[2] = spt_wrap(x[2]);
    for (i = 0; i < 4; ++i) {
        s->c.out[i] = x[i];
    }
    return 1;
}
"""

_SOURCES[("position2d", "pf")] = _POS_HEAD + r"""
int spt_pos_pf_reset(spt_pos_pf_t *s)
{
    spt_rng_seed(&s->rng, s->seed);
    s->ess = SPT_C(0.0);
    s->heading = SPT_C(0.0);
    s->since_fix = 0u;
    s->resamples = 0u;
    return spt_pos_core_reset(&s->c);
}

/* Başlangıç (Python _start): parçacıklar [x, y, ψ, v, 0, 1] + √p0·N(0, 1). */
static void spt_pos_pf_start(spt_pos_pf_t *s, const SPT_REAL st[4], const SPT_REAL p0[6])
{
    SPT_REAL mean[6], sd[6];
    int i, a;
    for (a = 0; a < 4; ++a) {
        mean[a] = st[a];
    }
    mean[4] = SPT_C(0.0);
    mean[5] = SPT_C(1.0);
    for (a = 0; a < 6; ++a) {
        sd[a] = SPT_SQRT(p0[a]);
    }
    for (i = 0; i < SPT_POS_PF_PARTICLES; ++i) {
        for (a = 0; a < 6; ++a) {
            s->X[i][a] = mean[a] + sd[a] * spt_rng_normal(&s->rng);
        }
    }
    s->ess = spt_pf_uniform(s->w, SPT_POS_PF_PARTICLES);
    s->since_fix = 0u;
    s->heading = st[2];
}

/* Python Pos2DPF._step: 6 boyutlu parçacıklar; odometri hareket modeline kontrol girişi (v = (z + n)/s), ağırlıklar
   yalnız GNSS'te; konum yürüyüşü GNSS anına ertelenir; ESS < N/2 → Liu–West (başlık dairesel). */
static int spt_pos_pf_core(spt_pos_pf_t *s, const SPT_REAL gnss[2], const double gnss_ll[2], SPT_REAL gyro_z,
                           SPT_REAL speed, SPT_REAL acc_fwd, SPT_REAL dt)
{
    const int n = SPT_POS_PF_PARTICLES;
    spt_pos_in_t in;
    spt_pos_noise_t q;
    SPT_REAL st[4], p0[6], sd_om, sd_v, sd_bg, ref, acc, mx = SPT_C(0.0), my = SPT_C(0.0), mv = SPT_C(0.0);
    int r, i;
    if (!s->c.ok) {
        return 0;
    }
    r = spt_pos_begin(&s->c, &s->cfg, gnss, gnss_ll, gyro_z, speed, acc_fwd, dt, &in, st, p0);
    if (r == 0) {
        return 0;
    }
    if (r == 1) {
        spt_pos_pf_start(s, st, p0);
        return 1;
    }
    spt_pos_noise(&s->cfg, dt, &q);
    sd_om = (in.gmiss ? SPT_SQRT(q.psi_miss) : SPT_SQRT(q.psi)) / dt;
    sd_v = in.has_acc ? SPT_SQRT(q.v_acc) : SPT_SQRT(q.v_rw);
    sd_bg = SPT_SQRT(q.bg);
    for (i = 0; i < n; ++i) {
        SPT_REAL *x = s->X[i];
        SPT_REAL e0 = spt_rng_normal(&s->rng), e1 = spt_rng_normal(&s->rng), e2 = spt_rng_normal(&s->rng);
        SPT_REAL om = in.gz - x[4] + sd_om * e0, pm, vT;
        if (in.has_spd) {
            x[3] = (in.spd + s->cfg.sigma_speed * e1) / x[5];
        } else if (in.has_acc) {
            x[3] += in.acc * dt + sd_v * e1;
        } else {
            x[3] += sd_v * e1;
        }
        pm = x[2] + SPT_C(0.5) * om * dt;
        vT = x[3] * dt;
        x[0] += vT * SPT_COS(pm);
        x[1] += vT * SPT_SIN(pm);
        x[2] += om * dt;
        x[4] += sd_bg * e2;
    }
    s->since_fix++;
    if (in.fix) {
        SPT_REAL sd_pos = SPT_SQRT(q.pos), ig, lmax = SPT_C(0.0), sum = SPT_C(0.0), sum2 = SPT_C(0.0);
        int any = 0;
        if (sd_pos > SPT_C(0.0)) {                   /* ertelenmiş konum yürüyüşü */
            SPT_REAL k = sd_pos * SPT_SQRT(SPT_FROM_INT(s->since_fix));
            for (i = 0; i < n; ++i) {
                s->X[i][0] += k * spt_rng_normal(&s->rng);
                s->X[i][1] += k * spt_rng_normal(&s->rng);
            }
        }
        s->since_fix = 0u;
        ig = SPT_C(-0.5) / (s->cfg.sigma_gnss * s->cfg.sigma_gnss);
        for (i = 0; i < n; ++i) {
            SPT_REAL dx = in.zx - s->X[i][0], dy = in.zy - s->X[i][1];
            spt_pf_max((dx * dx + dy * dy) * ig, &lmax, &any);
        }
        if (any) {
            for (i = 0; i < n; ++i) {
                SPT_REAL dx = in.zx - s->X[i][0], dy = in.zy - s->X[i][1];
                spt_pf_mul(&s->w[i], (dx * dx + dy * dy) * ig, lmax, &sum, &sum2);
            }
            s->ess = spt_pf_scale(s->w, n, sum, sum2);
        } else {
            s->ess = spt_pf_uniform(s->w, n);
        }
        if (s->ess < SPT_C(0.5) * SPT_FROM_INT(n)) {
            spt_lw_resample(&s->X[0][0], n, 6, 6, s->w, s->cnt, &s->rng, 2, s->work);
            s->resamples++;
        }
    }
    ref = s->heading;                                /* dairesel ortalama, önceki başlığa göre */
    acc = SPT_C(0.0);
    for (i = 0; i < n; ++i) {
        const SPT_REAL *x = s->X[i];
        mx += s->w[i] * x[0];
        my += s->w[i] * x[1];
        mv += s->w[i] * x[3];
        acc += s->w[i] * spt_wrap(x[2] - ref);
    }
    s->heading = spt_wrap(ref + acc);
    s->c.out[0] = mx;
    s->c.out[1] = my;
    s->c.out[2] = s->heading;
    s->c.out[3] = mv;
    return 1;
}

SPT_REAL spt_pos_pf_gyro_bias(const spt_pos_pf_t *s)
{
    SPT_REAL b = SPT_C(0.0);
    int i;
    for (i = 0; i < SPT_POS_PF_PARTICLES; ++i) {
        b += s->w[i] * s->X[i][4];
    }
    return b;
}

SPT_REAL spt_pos_pf_odo_scale(const spt_pos_pf_t *s)
{
    SPT_REAL b = SPT_C(0.0);
    int i;
    for (i = 0; i < SPT_POS_PF_PARTICLES; ++i) {
        b += s->w[i] * s->X[i][5];
    }
    return b;
}
"""

# ---------------------------------------------------------------------------- doğrusal
_SOURCES[("linear_multi", "weighted_mean")] = r"""
void spt_lin_weighted_mean_defaults(spt_lin_weighted_mean_t *s)
{
    spt_lin_cfg_defaults(&s->cfg);
    (void)spt_lin_weighted_mean_reset(s);
}

int spt_lin_weighted_mean_reset(spt_lin_weighted_mean_t *s)
{
    s->hold[0] = SPT_NAN;
    s->hold[1] = SPT_NAN;
    s->hold[2] = SPT_NAN;
    return spt_lin_core_reset(&s->c, &s->cfg, 3);
}

/* Python LinearWeightedMean: o an gelen aynı türdeki ölçümlerin ters varyans ortalaması (zamanda füzyon yok); ölçüm
   gelmeyen türde son değer tutulur, hiç sensörü olmayan tür NaN. */
static int spt_lin_weighted_mean_core(spt_lin_weighted_mean_t *s, const SPT_REAL zin[6], SPT_REAL dt)
{
    spt_lin_core_t *c = &s->c;
    spt_lin_meas_t m;
    int l;
    (void)dt;
    if (!c->ok) {
        return 0;
    }
    spt_lin_measurements(c, &s->cfg, zin, &m);
    if (m.m) {
        for (l = 0; l < 3; ++l) {
            SPT_REAL mean, info;
            if (spt_lin_level_mean(&m, l, &mean, &info)) {
                s->hold[l] = mean;
            }
        }
        c->ready = 1;
    }
    for (l = 0; l < 3; ++l) {
        c->out[l] = s->hold[l];
    }
    return c->ready;
}
"""

_SOURCES[("linear_multi", "kf")] = r"""
void spt_lin_kf_defaults(spt_lin_kf_t *s)
{
    spt_lin_cfg_defaults(&s->cfg);
    (void)spt_lin_kf_reset(s);
}

int spt_lin_kf_reset(spt_lin_kf_t *s)
{
    int i, j;
    for (i = 0; i < 3; ++i) {
        s->x[i] = SPT_C(0.0);
        for (j = 0; j < 3; ++j) {
            s->P[i][j] = SPT_C(0.0);
        }
    }
    return spt_lin_core_reset(&s->c, &s->cfg, 3);
}

/* Python LinearKF._run, bir örnek: x = F·x, P = F·P·Fᵀ + Q (F üst üçgen), ardından o örnekteki her ölçüm için
   sıralı skaler güncelleme (H = e_tür). Toplama sırası Python ile aynıdır. */
static int spt_lin_kf_core(spt_lin_kf_t *s, const SPT_REAL zin[6], SPT_REAL dt)
{
    spt_lin_core_t *c = &s->c;
    spt_lin_meas_t m;
    SPT_REAL F[3][3], Q[3][3], FP[3][3], xn[3], (*P)[3] = s->P, *x = s->x;
    int d, i, j, k;
    if (!c->ok) {
        return 0;
    }
    spt_lin_measurements(c, &s->cfg, zin, &m);
    d = c->d;
    if (!c->ready) {
        SPT_REAL x0[3], p0[3];
        if (!spt_lin_initial(c, &s->cfg, &m, x0, p0)) {
            return 0;
        }
        for (i = 0; i < 3; ++i) {
            x[i] = x0[i];
            for (j = 0; j < 3; ++j) {
                P[i][j] = (i == j) ? p0[i] : SPT_C(0.0);
            }
        }
        c->ready = 1;
        spt_lin_set_out(c, x);
        return 1;
    }
    spt_lin_model(d, dt, s->cfg.sigma_q, F, Q);
    for (i = 0; i < d; ++i) {
        SPT_REAL acc = SPT_C(0.0);
        for (j = i; j < d; ++j) {
            acc += F[i][j] * x[j];
        }
        xn[i] = acc;
    }
    for (i = 0; i < d; ++i) {
        x[i] = xn[i];
        for (j = 0; j < d; ++j) {
            SPT_REAL acc = SPT_C(0.0);
            for (k = i; k < d; ++k) {
                acc += F[i][k] * P[k][j];
            }
            FP[i][j] = acc;
        }
    }
    for (i = 0; i < d; ++i) {
        for (j = 0; j < d; ++j) {
            SPT_REAL acc = SPT_C(0.0);
            for (k = j; k < d; ++k) {
                acc += FP[i][k] * F[j][k];
            }
            P[i][j] = acc + Q[i][j];
        }
    }
    for (k = 0; k < m.m; ++k) {
        int cc = m.lvl[k];
        SPT_REAL pc[3], kg[3], sv, y;
        for (j = 0; j < d; ++j) {
            pc[j] = P[cc][j];
        }
        sv = pc[cc] + m.r[k];
        for (i = 0; i < d; ++i) {
            kg[i] = P[i][cc] / sv;
        }
        y = m.z[k] - x[cc];
        for (i = 0; i < d; ++i) {
            x[i] += kg[i] * y;
            for (j = 0; j < d; ++j) {
                P[i][j] -= kg[i] * pc[j];
            }
        }
    }
    spt_lin_set_out(c, x);
    return 1;
}
"""

_SOURCES[("linear_multi", "pf")] = r"""
/** Süreç gürültüsü kökü kapalı biçimde: Q = σ_q²·D·C·D, D = diag(T^(d−1−i)·√T) → L = |σ_q|·D·chol(C) (C sabit;
 *  Python chol_psd(Q) ile aynı matris, sayısal Cholesky ve yedek gerekmez; σ_q = 0 → L = 0). */
static void spt_lin_pf_noise_root(int d, SPT_REAL T, SPT_REAL sigma_q, SPT_REAL L[3][3])
{
    SPT_REAL st = SPT_SQRT(T), q = SPT_FABS(sigma_q);
    int i, j;
    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            L[i][j] = SPT_C(0.0);
        }
    }
    if (d == 1) {
        L[0][0] = q * st;
    } else if (d == 2) {
        SPT_REAL d0 = q * T * st, d1 = q * st;
        L[0][0] = d0 * SPT_C(0.57735026918962576451);   /* 1/√3 */
        L[1][0] = d1 * SPT_C(0.86602540378443864676);   /* √3/2 */
        L[1][1] = d1 * SPT_C(0.5);
    } else {
        SPT_REAL d0 = q * T * T * st, d1 = q * T * st, d2 = q * st;
        L[0][0] = d0 * SPT_C(0.22360679774997896964);   /* 1/√20 */
        L[1][0] = d1 * SPT_C(0.55901699437494742410);   /* √20/8 */
        L[1][1] = d1 * SPT_C(0.14433756729740644113);   /* 1/√48 */
        L[2][0] = d2 * SPT_C(0.74535599249992989880);   /* √20/6 */
        L[2][1] = d2 * SPT_C(0.57735026918962576451);   /* √48/12 */
        L[2][2] = d2 * SPT_C(0.33333333333333333333);   /* 1/3 */
    }
}

void spt_lin_pf_defaults(spt_lin_pf_t *s)
{
    spt_lin_cfg_defaults(&s->cfg);
{{SET_DEFAULTS}}
    (void)spt_lin_pf_reset(s);
}

int spt_lin_pf_reset(spt_lin_pf_t *s)
{
    spt_rng_seed(&s->rng, s->seed);
    s->ess = SPT_C(0.0);
    s->resamples = 0u;
    return spt_lin_core_reset(&s->c, &s->cfg, SPT_LIN_PF_DMAX);
}

static void spt_lin_pf_out(spt_lin_pf_t *s)
{
    SPT_REAL m[3];
    int i, a, d = s->c.d;
    for (a = 0; a < 3; ++a) {
        m[a] = SPT_C(0.0);
    }
    for (i = 0; i < SPT_LIN_PF_PARTICLES; ++i) {
        for (a = 0; a < d; ++a) {
            m[a] += s->w[i] * s->X[i][a];
        }
    }
    spt_lin_set_out(&s->c, m);
}

/* Python LinearPF._run, bir örnek: X ← F·X + L·N(0, I); ölçüm varsa Gauss olabilirlik (iki geçiş), ESS < N/2 →
   Liu–West yeniden örnekleme. */
static int spt_lin_pf_core(spt_lin_pf_t *s, const SPT_REAL zin[6], SPT_REAL dt)
{
    const int n = SPT_LIN_PF_PARTICLES;
    spt_lin_core_t *c = &s->c;
    spt_lin_meas_t m;
    SPT_REAL F[3][3], Q[3][3], L[3][3], hr[6];
    int d, i, a, b, k;
    if (!c->ok) {
        return 0;
    }
    spt_lin_measurements(c, &s->cfg, zin, &m);
    d = c->d;
    if (!c->ready) {
        SPT_REAL x0[3], p0[3], sd[3];
        if (!spt_lin_initial(c, &s->cfg, &m, x0, p0)) {
            return 0;
        }
        for (a = 0; a < d; ++a) {
            sd[a] = SPT_SQRT(p0[a]);
        }
        for (i = 0; i < n; ++i) {
            for (a = 0; a < d; ++a) {
                s->X[i][a] = x0[a] + sd[a] * spt_rng_normal(&s->rng);
            }
        }
        s->ess = spt_pf_uniform(s->w, n);
        c->ready = 1;
        spt_lin_pf_out(s);
        return 1;
    }
    spt_lin_model(d, dt, s->cfg.sigma_q, F, Q);
    spt_lin_pf_noise_root(d, dt, s->cfg.sigma_q, L);
    for (i = 0; i < n; ++i) {
        SPT_REAL *x = s->X[i], nz[3], xn[3];
        for (a = 0; a < d; ++a) {
            nz[a] = spt_rng_normal(&s->rng);
        }
        for (a = 0; a < d; ++a) {
            SPT_REAL acc = SPT_C(0.0);
            for (b = a; b < d; ++b) {
                acc += F[a][b] * x[b];
            }
            for (b = 0; b <= a; ++b) {
                acc += L[a][b] * nz[b];
            }
            xn[a] = acc;
        }
        for (a = 0; a < d; ++a) {
            x[a] = xn[a];
        }
    }
    if (m.m) {
        SPT_REAL lmax = SPT_C(0.0), sum = SPT_C(0.0), sum2 = SPT_C(0.0);
        int any = 0;
        for (k = 0; k < m.m; ++k) {
            hr[k] = SPT_C(-0.5) / m.r[k];
        }
        for (i = 0; i < n; ++i) {
            SPT_REAL ll = SPT_C(0.0);
            for (k = 0; k < m.m; ++k) {
                SPT_REAL e = m.z[k] - s->X[i][m.lvl[k]];
                ll += hr[k] * (e * e);
            }
            spt_pf_max(ll, &lmax, &any);
        }
        if (any) {
            for (i = 0; i < n; ++i) {
                SPT_REAL ll = SPT_C(0.0);
                for (k = 0; k < m.m; ++k) {
                    SPT_REAL e = m.z[k] - s->X[i][m.lvl[k]];
                    ll += hr[k] * (e * e);
                }
                spt_pf_mul(&s->w[i], ll, lmax, &sum, &sum2);
            }
            s->ess = spt_pf_scale(s->w, n, sum, sum2);
        } else {
            s->ess = spt_pf_uniform(s->w, n);
        }
        if (s->ess < SPT_C(0.5) * SPT_FROM_INT(n)) {
            spt_lw_resample(&s->X[0][0], n, SPT_LIN_PF_DMAX, d, s->w, s->cnt, &s->rng, -1, s->work);
            s->resamples++;
        }
    }
    spt_lin_pf_out(s);
    return 1;
}
"""

# ---------------------------------------------------------------------------- girişler: C imzası ve belgeler
_INPUT_DOC = {
    "baro_p": "barometre basıncı (Pa); NaN / ≤ 0 = ölçüm yok",
    "baro_alt": "barometrik irtifa (m); NaN = yok (örnekte basınç da varsa basınç kullanılır)",
    "acc_z": "dikey özgül kuvvet (m/s², dünya z yukarı, yerçekimi DAHİL: durağanken +g); NaN = yok (son değer "
             "tutulur)",
    "gnss": "GNSS konumu [x doğu, y kuzey] (m, yerel ENU) ya da NULL; NaN / NULL = düzeltme yok",
    "gnss_ll": "GNSS [enlem, boylam] (rad, her zaman double) ya da NULL; ilk geçerli düzeltme orijin olur",
    "gyro_z": "sapma (yaw) hızı (rad/s, z yukarı, saat yönünün tersi +); NaN = yok (son değer tutulur)",
    "speed": "hız ölçümü / odometri (m/s); NaN = yok",
    "acc_fwd": "boylamsal ivme (m/s², yerçekimi bileşeni çıkarılmış); NaN = yok",
}
_CORE_ARGS = {"altitude": ("baro_p", "baro_alt", "acc_z"),
              "position2d": ("gnss", "gnss_ll", "gyro_z", "speed", "acc_fwd")}


def _is_vec(key: str) -> bool:
    return key in ("gnss", "gnss_ll")


def _param_decl(key: str) -> str:
    if key == "gnss":
        return "const SPT_REAL gnss[2]"
    if key == "gnss_ll":
        return "const double gnss_ll[2]"
    return f"SPT_REAL {key}"


def _input_doc(r: _Resolved, key: str) -> str:
    if r.scenario == "linear_multi":
        i = int(key[1:])
        kind = r.values[f"{key}_kind"]
        return (f"sensör {i} ölçümü (tür: {dict(lin_mod.KINDS)[kind].split(' (')[0].lower()}, σ = "
                f"{r.values[f'{key}_sigma']:g}; giriş biriminde); NaN = yok")
    return _INPUT_DOC[key]


def _signature(r: _Resolved) -> str:
    return ", ".join(_param_decl(k) for k in r.inputs)


def _core_call(r: _Resolved, algo: str) -> str:
    a = f"{r.sab}_{algo}"
    if r.scenario == "linear_multi":
        vals = ", ".join(f"s{i}" if f"s{i}" in r.inputs else "SPT_NAN" for i in range(1, 7))
        return f"    const SPT_REAL z[6] = {{{vals}}};\n    return spt_{a}_core(s, z, dt);"
    args = []
    for k in _CORE_ARGS[r.scenario]:
        if k in r.inputs:
            args.append(k)
        else:
            args.append("NULL" if _is_vec(k) else "SPT_NAN")
    return f"    return spt_{a}_core(s, {', '.join(args)}, dt);"


# ---------------------------------------------------------------------------- başlık / kaynak dosyaları
_HEADER = r"""/**
 * @file spt_{{A}}.h
 * @brief {{BRIEF}}
 *
 * {{BANNER}}
 * Python karşılığı: spt.dsp.fusion_scenarios.{{PYMOD}}.{{PYCLASS}}
 * ({{SCEN_NAME}} / {{ALGO_NAME}}).
 *
 * Kullanım:
 *   static spt_{{A}}_t s;
 *   spt_{{A}}_init(&s);
 *   her örnekte: spt_{{A}}_step(&s, {{ARGS_SHORT}}, dt);
 *                spt_{{A}}_output(&s, out);   out = {{OUT_DOC}}
 * Eksik ölçüm: SPT_NAN (vektör girişte NULL da olur). Durum yapısı çağıranındır: malloc ve global yok, yeniden
 * girişlidir (aynı yapıya aynı anda iki bağlamdan erişmeyin).
 */
#ifndef SPT_{{AU}}_H
#define SPT_{{AU}}_H

#include "spt_{{SAB}}_common.h"
{{PARAM_DEFINES}}
{{STRUCT}}
SPT_BEGIN_DECLS

/** Parametreleri derleme anı varsayılanlarına (#define) getirir ve durumu sıfırlar (reset). */
void spt_{{A}}_defaults(spt_{{A}}_t *s);

/**
 * Varsayılan parametrelerle başlatır (= defaults). Kestirici ilk uygun ölçümde step() içinde kendiliğinden başlar
 * (Python ile aynı).
 * @return 1: parametreler geçerli, 0: geçersiz (Python'da FusionError; step() hep 0 döner, çıktılar NaN)
 */
int spt_{{A}}_init(spt_{{A}}_t *s);

/** Durumu başlangıca döndürür, parametreleri korur (Python Estimator.reset()); çalışma anında değiştirilen
 *  parametreleri de denetler. @return 1: parametreler geçerli */
int spt_{{A}}_reset(spt_{{A}}_t *s);

/**
 * Bir örnek işler — Python Estimator.run'ın örnek döngüsüyle birebir (NaN = bu örnekte ölçüm yok).
{{INPUT_DOCS}}
 * @param dt örnekleme aralığı (s, > 0 ve sonlu; Python'daki 1/fs). Değilse örnek atlanır, durum değişmez, 0 döner.
 * @return 1: kestirim var (başlatıldı), 0: henüz yok{{RET_NOTE}}
 */
int spt_{{A}}_step(spt_{{A}}_t *s, {{SIG}}, SPT_REAL dt);

/** step() ile aynı (dönüş değeri yok). */
void spt_{{A}}_update(spt_{{A}}_t *s, {{SIG}}, SPT_REAL dt);

/** Çıktılar (Python Scenario.outputs sırasıyla): {{OUT_DOC}}. Kestirim yoksa NaN. */
void spt_{{A}}_output(const spt_{{A}}_t *s, SPT_REAL out[{{K}}]);
{{GETTER_DECLS}}{{EXTRA_DECLS}}
SPT_END_DECLS

#endif /* SPT_{{AU}}_H */
"""

_API_IMPL = r"""
int spt_{{A}}_init(spt_{{A}}_t *s)
{
    spt_{{A}}_defaults(s);
    return s->c.ok;
}

int spt_{{A}}_step(spt_{{A}}_t *s, {{SIG}}, SPT_REAL dt)
{
    if (!(dt > SPT_C(0.0)) || !spt_finite(dt)) {
        return 0;   /* geçersiz dt (≤ 0, NaN, sonsuz; Python'da FusionError): örnek atlanır, durum değişmez */
    }
{{CALL}}
}

void spt_{{A}}_update(spt_{{A}}_t *s, {{SIG}}, SPT_REAL dt)
{
    (void)spt_{{A}}_step(s, {{ARGS}}, dt);
}

void spt_{{A}}_output(const spt_{{A}}_t *s, SPT_REAL out[{{K}}])
{
{{OUT_COPY}}
}
{{GETTERS}}"""

_EXTRA_DECLS = {
    ("altitude", "complementary"): "acc_bias",
    ("altitude", "kf"): "acc_bias", ("altitude", "ekf"): "acc_bias", ("altitude", "ukf"): "acc_bias",
    ("altitude", "pf"): "acc_bias",
    ("position2d", "ekf"): "gyro_odo", ("position2d", "ukf"): "gyro_odo", ("position2d", "pf"): "gyro_odo",
}


def _extra_decls(r: _Resolved, algo: str) -> str:
    a = f"{r.sab}_{algo}"
    kind = _EXTRA_DECLS.get((r.scenario, algo))
    if kind == "acc_bias":
        cost = " (O(N) ağırlıklı ortalama)" if algo == "pf" else ""
        return f"\n/** İvme sapması kestirimi (m/s²){cost}. */\nSPT_REAL spt_{a}_acc_bias(const spt_{a}_t *s);\n"
    if kind == "gyro_odo":
        cost = " (O(N) ağırlıklı ortalama)" if algo == "pf" else ""
        return (f"\n/** Jiroskop sapması kestirimi b_g (rad/s){cost}. */\nSPT_REAL spt_{a}_gyro_bias(const spt_{a}_t *s);\n"
                f"\n/** Odometri ölçek çarpanı kestirimi s (≈ 1){cost}. */\n"
                f"SPT_REAL spt_{a}_odo_scale(const spt_{a}_t *s);\n")
    return ""


def _out_doc(r: _Resolved) -> str:
    outs = fg.SCENARIOS[r.scenario].outputs
    unit = {"°": "rad"}
    return "[" + ", ".join(f"{o.key} ({unit.get(o.unit, o.unit)})" for o in outs) + "]"


_BRIEFS = {
    ("altitude", "complementary"): "İrtifa: klasik baro-ataletsel tamamlayıcı filtre (geçiş zaman sabiti τ, isteğe "
                                   "bağlı ivme sapması kestirimi).",
    ("altitude", "kf"): "İrtifa: doğrusal Kalman filtresi (durum h, v, ivme sapması; basınç ISA ile irtifaya "
                        "çevrilir).",
    ("altitude", "ekf"): "İrtifa: genişletilmiş Kalman filtresi (basınç doğrudan ölçüm, H = dp/dh).",
    ("altitude", "ukf"): "İrtifa: kokusuz Kalman filtresi (basınç ölçümü 7 sigma noktasıyla; öngörü KF ile aynı).",
    ("altitude", "pf"): "İrtifa: parçacık filtresi (h, v, b parçacıkları; Liu–West yumuşatmalı sistematik yeniden "
                        "örnekleme).",
    ("position2d", "dead_reckoning"): "2B konum: ölü hesap (yalnız integral; taban çizgisi).",
    ("position2d", "ekf"): "2B konum: genişletilmiş Kalman filtresi (x, y, ψ, v, jiroskop sapması, odometri ölçeği).",
    ("position2d", "ukf"): "2B konum: kokusuz Kalman filtresi (13 sigma noktası, başlık dairesel ortalamayla).",
    ("position2d", "pf"): "2B konum: parçacık filtresi (6 boyutlu durum; odometri kontrol girişi, ağırlık GNSS'te).",
    ("linear_multi", "weighted_mean"): "Doğrusal çok sensörlü: anlık ters varyans ağırlıklı ortalama (taban "
                                       "çizgisi).",
    ("linear_multi", "kf"): "Doğrusal çok sensörlü Kalman filtresi (sabit seviye / hız / ivme; çok hızlı sıralı "
                            "skaler güncelleme).",
    ("linear_multi", "pf"): "Doğrusal çok sensörlü parçacık filtresi (aynı model; karşılaştırma için).",
}


def _choice_index(scenario: str, algo: str | None, key: str, value: str) -> int:
    k = "kind" if re.fullmatch(r"s\d_kind", key) else key
    return _CHOICE_INDEX[(scenario, algo, k)][value]


def _define(macro: str, literal: str, label: str, tip: str, default) -> str:
    tip = f" — {tip}" if tip else ""
    dflt = default if isinstance(default, str) else f"{default:g}"
    return f"/** {label}{tip} (Python varsayılanı {dflt}) */\n#ifndef {macro}\n#define {macro} {literal}\n#endif"


def _scenario_block(r: _Resolved) -> tuple[str, str]:
    """(senaryo #define bloğu, cfg_defaults gövdesi)."""
    scen = fg.SCENARIOS[r.scenario]
    S = r.sab.upper()
    defines, sets = [], []
    for p in scen.params:
        macro = f"SPT_{S}_{p.key.upper()}"
        v = r.values[p.key]
        if isinstance(p, Choice):
            idx = _choice_index(r.scenario, None, p.key, v)
            opts = ", ".join(f"{_choice_index(r.scenario, None, p.key, k)} {lab.split(' (')[0]}" for k, lab in p.options)
            defines.append(_define(macro, str(idx), f"{p.label} ({opts})", p.tip, p.default))
        else:
            defines.append(_define(macro, _c_float(v), p.label, p.tip, p.default))
        m = re.fullmatch(r"s(\d)_(kind|sigma)", p.key)
        if m:
            i = int(m.group(1)) - 1
            if m.group(2) == "kind":
                sets.append(f"    c->kind[{i}] = {macro};")
            else:
                sets.append(f"    c->sigma[{i}] = (SPT_REAL)({macro});")
        elif isinstance(p, Choice):
            sets.append(f"    c->{p.key} = {macro};")
        else:
            sets.append(f"    c->{p.key} = (SPT_REAL)({macro});")
    if r.scenario == "linear_multi":
        defines.append("/** Doğrusal parçacık filtresinin durum dizisi boyutu (çalışma anında derece en çok buna kadar "
                       "artırılabilir) */\n#ifndef SPT_LIN_PF_DMAX\n#define SPT_LIN_PF_DMAX (SPT_LIN_ORDER + 1)\n#endif")
    return "\n".join(defines), "\n".join(sets)


def _algo_block(r: _Resolved, algo: str) -> tuple[str, str]:
    """(algoritma #define bloğu, defaults() içi atamalar)."""
    cls = fg.SCENARIOS[r.scenario].algorithms[algo]
    A = f"{r.sab}_{algo}".upper()
    defines, sets = [], []
    vals = r.params[algo]
    for p in cls.params:
        macro = f"SPT_{A}_{p.key.upper()}"
        v = vals[p.key]
        if algo == "pf" and p.key == "particles":
            defines.append(_define(macro, str(int(v)), p.label + " (derleme anı: dizi boyutu)", p.tip, p.default))
            continue
        if algo == "pf" and p.key == "seed":
            defines.append(_define(macro, f"{int(v)}u", p.label, p.tip, p.default))
            sets.append(f"    s->seed = (uint32_t)({macro});")
        elif isinstance(p, Choice):
            idx = _choice_index(r.scenario, algo, p.key, v)
            opts = ", ".join(f"{_choice_index(r.scenario, algo, p.key, k)} {lab.split(' (')[0]}" for k, lab in p.options)
            defines.append(_define(macro, str(idx), f"{p.label} ({opts})", p.tip, p.default))
            sets.append(f"    s->{p.key} = {macro};")
        else:
            defines.append(_define(macro, _c_float(v), p.label, p.tip, p.default))
            sets.append(f"    s->{p.key} = (SPT_REAL)({macro});")
    return ("\n" + "\n".join(defines) + "\n") if defines else "", "\n".join(sets)


def _common_header(opts: GenericCodegenOptions, r: _Resolved) -> str:
    rng = _struct_c("spt_rng", "Rastgele sayı üreteci durumu.", _RNG_FIELDS)
    return _fill(_COMMON_H, BANNER=_BANNER, REAL=opts.real, USE_DOUBLE="1" if opts.real == "double" else "0",
                 RNG_STRUCT=rng)


def _scenario_header(r: _Resolved) -> str:
    defines, sets = _scenario_block(r)
    scen_title = fg.SCENARIOS[r.scenario].name
    cfg = _struct_c(f"spt_{r.sab}_cfg", f"Senaryo parametreleri ({scen_title}; Python Scenario.params) — çalışma "
                    "anında değiştirilebilir.", _cfg_fields(r))
    core = _struct_c(f"spt_{r.sab}_core", "Senaryo durumu (Python taban sınıfı) ve çıktılar.", _core_fields(r))
    kw = dict(BANNER=_BANNER, PARAM_DEFINES=defines, CFG_STRUCT=cfg, CORE_STRUCT=core, CFG_DEFAULTS=sets)
    if r.scenario == "altitude":
        return _fill(_ALT_COMMON_H, HAS_BARO_P=str(int("baro_p" in r.inputs)),
                     HAS_BARO_ALT=str(int("baro_alt" in r.inputs)), **kw)
    if r.scenario == "position2d":
        return _fill(_POS_COMMON_H, FIX_HISTORY=str(FIX_HISTORY), HAS_GNSS=str(int("gnss" in r.inputs)),
                     HAS_GNSS_LL=str(int("gnss_ll" in r.inputs)), HAS_SPEED=str(int("speed" in r.inputs)),
                     HAS_ACC_FWD=str(int("acc_fwd" in r.inputs)), **kw)
    mask = sum(1 << (int(k[1:]) - 1) for k in r.inputs)
    return _fill(_LIN_COMMON_H, MAPPED=f"0x{mask:02X}", **kw)


def _algo_files(r: _Resolved, algo: str) -> tuple[str, str]:
    scen = fg.SCENARIOS[r.scenario]
    cls = scen.algorithms[algo]
    a = f"{r.sab}_{algo}"
    defines, sets = _algo_block(r, algo)
    outs = scen.outputs
    k = len(outs)
    struct = _struct_c(f"spt_{a}", f"{cls.name} durumu ({scen.name}).", _algo_fields(r, algo))
    sig = _signature(r)
    args = ", ".join(r.inputs)
    getter_decls = "".join(f"\n/** {o.label} ({'rad' if o.angle else o.unit}); kestirim yoksa NaN. */\n"
                           f"SPT_REAL spt_{a}_{o.key}(const spt_{a}_t *s);\n" for o in outs)
    getters = "".join(f"\nSPT_REAL spt_{a}_{o.key}(const spt_{a}_t *s)\n{{\n    return s->c.out[{i}];\n}}\n"
                      for i, o in enumerate(outs))
    docs = "\n".join(f" * @param {key} {_input_doc(r, key)}" for key in r.inputs)
    ret_note = {"position2d": " (başlatma öncesi x, y son GNSS düzeltmesi olabilir)",
                "linear_multi": " (ağırlıklı ortalamada: ölçülmeyen türlerin çıktısı NaN kalır)"}.get(r.scenario, "")
    header = _fill(_HEADER, A=a, AU=a.upper(), BRIEF=_BRIEFS[(r.scenario, algo)], BANNER=_BANNER,
                   PYMOD=_PYMOD[r.scenario], PYCLASS=cls.__name__, SCEN_NAME=scen.name, ALGO_NAME=cls.name,
                   ARGS_SHORT=args, OUT_DOC=_out_doc(r), SAB=r.sab, PARAM_DEFINES=defines, STRUCT=struct,
                   INPUT_DOCS=docs, RET_NOTE=ret_note, SIG=sig, K=str(k), GETTER_DECLS=getter_decls,
                   EXTRA_DECLS=_extra_decls(r, algo))
    body = _SOURCES[(r.scenario, algo)].replace("{{A}}", a)
    body = _fill(body.replace("{{SET_DEFAULTS}}\n", sets + "\n" if sets else ""))
    out_copy = "\n".join(f"    out[{i}] = s->c.out[{i}];" for i in range(k))
    impl = _fill(_API_IMPL, A=a, SIG=sig, CALL=_core_call(r, algo), ARGS=args, K=str(k), OUT_COPY=out_copy,
                 GETTERS=getters)
    source = (f"/**\n * @file spt_{a}.c\n * @brief {cls.name} — uygulama (Python: spt.dsp.fusion_scenarios."
              f"{_PYMOD[r.scenario]}.{cls.__name__}).\n *\n * {_BANNER}\n */\n#include \"spt_{a}.h\"\n"
              + body + impl)
    return header, source


_WIDTH = 120


def _words(text: str, width: int) -> list[str]:
    lines, cur = [], ""
    for w in text.split(" "):
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    return lines + [cur]


def _split_args(line: str) -> tuple[str, list[str], str] | None:
    """'… f(a, b, c) …' → ('… f(', ['a,', 'b,', 'c'], ') …') — virgülleri düzey 1'de olan ilk parantez grubu."""
    for p, ch0 in enumerate(line):
        if ch0 != "(":
            continue
        depth, start, parts = 0, p + 1, []
        for i in range(p, len(line)):
            ch = line[i]
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
                if depth == 0:
                    parts.append(line[start:i].strip())
                    if len(parts) > 1:
                        return line[:p + 1], [x + "," for x in parts[:-1]] + [parts[-1]], line[i:]
                    break
            elif ch == "," and depth == 1:
                parts.append(line[start:i].strip())
                start = i + 1
    return None


def _depth0_commas(line: str) -> list[int]:
    depth, out = 0, []
    for i, ch in enumerate(line):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            out.append(i)
    return out


def _wrap_line(line: str, width: int) -> list[str]:
    if len(line) <= width:
        return [line]
    indent = line[:len(line) - len(line.lstrip())]
    s = line.strip()
    if re.fullmatch(r"/\* -{2,}.*-{2,} \*/", s):                         # süs çizgisi: tireleri kısalt
        return [re.sub(r"-+ \*/$", lambda m: "-" * max(2, len(m.group(0)) - 3 - (len(line) - width)) + " */", line)]
    m = re.fullmatch(r"/\*\* (.*) \*/", s)                                # tek satırlık belge yorumu → blok
    if m:
        return [f"{indent}/**"] + [f"{indent} * {x}" for x in _words(m.group(1), width - len(indent) - 3)] + \
            [f"{indent} */"]
    m = re.fullmatch(r"/\* (.*) \*/", s)                                  # tek satırlık yorum → çok satır
    if m:
        parts = _words(m.group(1), width - len(indent) - 6)
        return [f"{indent}/* {parts[0]}"] + [f"{indent}   {x}" for x in parts[1:-1]] + \
            ([f"{indent}   {parts[-1]} */"] if len(parts) > 1 else [f"{indent}/* {parts[0]} */"])
    m = re.fullmatch(r"(\S.*?)\s+(/\*\*?<? .*\*/)", s)                    # kod + sondaki yorum → yorum üstte
    if m and not s.startswith(("/*", "*")):
        comment = re.sub(r"^/\*\*< ", "/** ", m.group(2))
        return _wrap_line(indent + comment, width) + _wrap_line(indent + m.group(1), width)
    m = re.fullmatch(r"(\s*\*\s+)(\S.*)", line)                          # blok yorum içi (" * …", " *  - …")
    if m:
        head, body = m.group(1), m.group(2)
        cont = head + "  " if body.startswith(("@", "- ")) else head
        parts = _words(body, width - len(cont))
        return [f"{head}{parts[0]}"] + [f"{cont}{x}" for x in parts[1:]]
    if s.startswith(("#", "/*", "*", "//")):
        return [line]
    sp = _split_args(line)
    if sp:                                                                # çağrı / imza: virgüllerde, paranteze hizalı
        head, args, tail = sp
        pad = " " * len(head)
        out, cur = [], head
        for i, a in enumerate(args):
            piece = a + (tail if i == len(args) - 1 else "")
            if cur != head and len(cur) + 1 + len(piece) > width:
                out.append(cur.rstrip())
                cur = pad + piece
            else:
                cur = cur + ("" if cur == head else " ") + piece
        out.append(cur)
        if all(len(x) <= width for x in out) or len(out) > 1:
            return [y for x in out for y in (_wrap_line(x, width) if len(x) > width and x != line else [x])]
    commas = _depth0_commas(line)
    if commas and s.endswith(";"):                                        # çok bildirimli satır
        cut = max((c for c in commas if c < width), default=commas[0])
        return [line[:cut + 1]] + _wrap_line(indent + " " * 8 + line[cut + 1:].strip(), width)
    for op in (" && ", " || ", " + ", " - ", " * ", " / ", " "):          # son çare: işleç / boşlukta böl
        cut = line.rfind(op, len(indent) + 1, width)
        if cut > len(indent) + 8:
            return [line[:cut].rstrip()] + _wrap_line(indent + " " * 8 + line[cut:].strip(), width)
    return [line]


def _join_continuations(text: str) -> str:
    """Şablonlarda elle bölünmüş imza / çağrı satırlarını (açık parantez + satır sonu virgül) birleştirir."""
    lines, out, i = text.split("\n"), [], 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if not s.startswith(("#", "/*", "*", "//")) and "/*" not in line:
            while (line.rstrip().endswith(",") and line.count("(") > line.count(")") and i + 1 < len(lines)
                   and "/*" not in lines[i + 1]):
                i += 1
                line = line.rstrip() + " " + lines[i].strip()
        out.append(line)
        i += 1
    return "\n".join(out)


def _wrap_c(text: str, width: int = _WIDTH) -> str:
    """Üretilen C satırlarını `width` sütuna sarar (yalnız biçim; anlam değişmez)."""
    return "\n".join(y for line in _join_continuations(text).split("\n") for y in _wrap_line(line, width))


def _sources(opts: GenericCodegenOptions, r: _Resolved | None = None, wrap: bool = True) -> list[GeneratedFile]:
    """C dosyaları (önek uygulanmamış: spt_*)."""
    r = r or _resolve(opts)
    files = [GeneratedFile("spt_fusion_generic_common.h", _common_header(opts, r)),
             GeneratedFile(f"spt_{r.sab}_common.h", _scenario_header(r))]
    for algo in r.algorithms:
        h, c = _algo_files(r, algo)
        files += [GeneratedFile(f"spt_{r.sab}_{algo}.h", h), GeneratedFile(f"spt_{r.sab}_{algo}.c", c)]
    return [GeneratedFile(f.name, _wrap_c(f.text)) for f in files] if wrap else files


# ============================================================================ örnek kullanım dosyası
def _example(opts: GenericCodegenOptions, r: _Resolved) -> str:
    scen = fg.SCENARIOS[r.scenario]
    first = f"{r.sab}_{r.algorithms[0]}"
    k = len(scen.outputs)
    incs = "\n".join(f'#include "spt_{r.sab}_{a}.h"' for a in r.algorithms)
    states = "\n".join(f"    static spt_{r.sab}_{a}_t f_{a};   /* {scen.algorithms[a].name} */" for a in r.algorithms)
    inits = "\n".join(f"    (void)spt_{r.sab}_{a}_init(&f_{a});" for a in r.algorithms)
    decls, params, args = [], [], []
    for key in r.inputs:
        if key == "gnss":
            decls.append("    SPT_REAL gnss[2];")
            params.append("SPT_REAL gnss[2]")
            args.append("gnss_var ? gnss : NULL")
        elif key == "gnss_ll":
            decls.append("    double gnss_ll[2];")
            params.append("double gnss_ll[2]")
            args.append("gnss_var ? gnss_ll : NULL")
        else:
            decls.append(f"    SPT_REAL {key};")
            params.append(f"SPT_REAL *{key}")
            args.append(key)
    has_vec = any(_is_vec(k_) for k_ in r.inputs)
    read_args = ", ".join(key if _is_vec(key) else f"&{key}" for key in r.inputs)
    if has_vec:
        params.append("int *gnss_var")
        read_args += ", &gnss_var"
    argstr = ", ".join(args)
    others = "\n".join(f"        (void)spt_{r.sab}_{a}_step(&f_{a}, {argstr}, dt);" for a in r.algorithms[1:])
    if others:
        others = "\n        /* diğer seçili algoritmalar (karşılaştırma için; gerekmiyorsa silin) */\n" + others
    out_names = ", ".join(f"{o.key} {'rad' if o.angle else o.unit}" for o in scen.outputs)
    in_doc = "\n".join(f" *   {key}: {_input_doc(r, key)}" for key in r.inputs)
    gnss_doc = ("\n *  *gnss_var: bu örnekte yeni GNSS düzeltmesi var mı (yoksa NULL geçilir)." if has_vec else "")
    return _fill(r"""/**
 * @file ornek_kullanim.c
 * @brief Örnek: sabit örnekleme hızında füzyon adımı ve çıktıların okunması ({{SCEN}}).
 *
 * {{BANNER}}
 *
 * Donanıma bağlı üç fonksiyon (sensor_oku, ornek_bekle, sonuc_kullan) kullanıcı tarafından yazılır (STM32 HAL,
 * Zephyr, ESP-IDF …). fuzyon_gorevi() main() içinden ya da bir RTOS görevinden çağrılır ve geri dönmez.
 */
#include <stdint.h>
#include <stddef.h>
{{INCLUDES}}

#ifndef ORNEK_FS_HZ
#define ORNEK_FS_HZ {{FS}}            /* akış (en hızlı sensör) örnekleme frekansı (Hz) */
#endif

/* ---- kullanıcı tarafından sağlanır --------------------------------------------------------------------------- */
/** Bir akış örneği okur (SI birimleri). Bu örnekte gelmeyen (seyrek) sensör için SPT_NAN yazın:
{{IN_DOC}}{{GNSS_DOC}}
 *  @return 1: yeni örnek, 0: yok */
extern int sensor_oku({{PARAMS}});
/** Bir sonraki örnekleme anına kadar bekler (zamanlayıcı kesmesinin kurduğu bayrak, vTaskDelayUntil …). */
extern void ornek_bekle(void);
/** Sonucu kullanır (UART, CAN, kontrol döngüsü …): out = [{{OUTS}}] (NaN = henüz kestirim yok). */
extern void sonuc_kullan(const SPT_REAL out[{{K}}]);

void fuzyon_gorevi(void);

void fuzyon_gorevi(void)
{
{{STATES}}
{{DECLS}}{{GNSS_DECL}}
    SPT_REAL out[{{K}}];
    const SPT_REAL dt = (SPT_REAL)(1.0 / ORNEK_FS_HZ);
{{INITS}}
    for (;;) {
        ornek_bekle();
        if (!sensor_oku({{READ_ARGS}})) {
            continue;
        }
        /* step(): NaN girişleri "ölçüm yok" sayar, ilk uygun ölçümde kendiliğinden başlatır */
        if (spt_{{FIRST}}_step(&f_{{FIRST_ALGO}}, {{ARGS}}, dt)) {
            spt_{{FIRST}}_output(&f_{{FIRST_ALGO}}, out);
            sonuc_kullan(out);
        }{{OTHERS}}
    }
}
""", SCEN=scen.name, BANNER=_BANNER, INCLUDES=incs, FS=_c_float(opts.fs_hz), IN_DOC=in_doc, GNSS_DOC=gnss_doc,
        PARAMS=", ".join(params), OUTS=out_names, K=str(k), STATES=states, DECLS="\n".join(decls),
        GNSS_DECL="\n    int gnss_var = 0;" if has_vec else "", INITS=inits, READ_ARGS=read_args, FIRST=first,
        FIRST_ALGO=r.algorithms[0], ARGS=argstr, OTHERS=others)


# ============================================================================ üretim
def generate(opts: GenericCodegenOptions, footprint_rows: list[FootprintRow] | None = None,
             cost_rows: list[CostRow] | None = None) -> list[GeneratedFile]:
    """Ortak başlık, senaryo başlığı, algoritma başına .h / .c, ornek_kullanim_<senaryo>.c ve KULLANIM_<senaryo>.md
    (önek uygulanmış).

    footprint_rows / cost_rows verilirse ölçüm tabloları KULLANIM.md'ye eklenir."""
    r = _resolve(opts)
    # Senaryoya özgü adlar: yönelim üreticisinin ornek_kullanim.c / KULLANIM.md dosyalarıyla aynı klasöre yazılabilir
    doc, example = usage_names(r.sab)
    files = _sources(opts, r, wrap=False) + [
        GeneratedFile(example, _example(opts, r).replace("@file ornek_kullanim.c", f"@file {example}"))]
    files = [GeneratedFile(_prefixed(f.name, opts.prefix),
                           _wrap_c(_prefixed(f.text.replace("kullanım: KULLANIM.md", f"kullanım: {doc}"), opts.prefix)))
             for f in files]
    files.append(GeneratedFile(doc, usage_markdown(opts, footprint_rows, cost_rows)))
    return files


def usage_names(sab: str) -> tuple[str, str]:
    """(kullanım kılavuzu, örnek dosya) adları: KULLANIM_<senaryo>.md, ornek_kullanim_<senaryo>.c."""
    return f"KULLANIM_{sab}.md", f"ornek_kullanim_{sab}.c"


# ============================================================================ ana makine sürücüsü
_DRIVER = r"""/* Ana makine doğrulama / maliyet sürücüsü (MCU'ya eklenmez). Giriş dosyası: [N, dt, sütun] + N×sütun double
   (eşlenen girişler Scenario.inputs sırasıyla; vektörler 2 sütun); çıkış: N×K çıktı (NaN = yok) (+ sayaç modunda
   N×SPT_NOPS işlem sayısı). Vektör girişin iki bileşeni de NaN ise NULL geçilir (NULL yolu da sınanır). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
{{INCLUDES}}

#if defined(SPT_CUSTOM_REAL)
unsigned long long spt_ops[SPT_NOPS];
#define SPT_DRV_REAL(x) spt_cnt(x)
#define SPT_DRV_DBL(r) ((r).v)
#define SPT_DRV_NOPS SPT_NOPS
#else
#define SPT_DRV_REAL(x) ((SPT_REAL)(x))
#define SPT_DRV_DBL(r) ((double)(r))
#define SPT_DRV_NOPS 0
#endif

static double *spt_drv_in, *spt_drv_out, *spt_drv_ops;
static long spt_drv_n, spt_drv_cols;
static double spt_drv_dt;

static void spt_drv_snap(unsigned long long *before)
{
#if defined(SPT_CUSTOM_REAL)
    memcpy(before, spt_ops, sizeof(spt_ops));
#else
    (void)before;
#endif
}

static void spt_drv_count(long k, const unsigned long long *before)
{
#if defined(SPT_CUSTOM_REAL)
    int i;
    for (i = 0; i < SPT_NOPS; ++i) {
        spt_drv_ops[k * SPT_NOPS + i] = (double)(spt_ops[i] - before[i]);
    }
#else
    (void)k;
    (void)before;
#endif
}

{{BOTH_NAN}}{{RUNNERS}}
int main(int argc, char **argv)
{
    FILE *f;
    double hdr[3];
    size_t total;
    if (argc < 4) {
        return 2;
    }
    f = fopen(argv[2], "rb");
    if (!f || fread(hdr, sizeof(double), 3, f) != 3) {
        return 3;
    }
    spt_drv_n = (long)hdr[0];
    spt_drv_dt = hdr[1];
    spt_drv_cols = (long)hdr[2];
    total = (size_t)spt_drv_n * (size_t)spt_drv_cols;
    spt_drv_in = (double *)malloc(sizeof(double) * (total + 1));
    spt_drv_out = (double *)malloc(sizeof(double) * {{K}} * ((size_t)spt_drv_n + 1));
    spt_drv_ops = (double *)malloc(sizeof(double) * (SPT_DRV_NOPS + 1) * ((size_t)spt_drv_n + 1));
    if (!spt_drv_in || !spt_drv_out || !spt_drv_ops || fread(spt_drv_in, sizeof(double), total, f) != total) {
        return 4;
    }
    fclose(f);
{{DISPATCH}}
    {
        return 5;
    }
    f = fopen(argv[3], "wb");
    if (!f) {
        return 6;
    }
    fwrite(spt_drv_out, sizeof(double), (size_t){{K}} * (size_t)spt_drv_n, f);
    if (SPT_DRV_NOPS > 0) {
        fwrite(spt_drv_ops, sizeof(double), (size_t)(SPT_DRV_NOPS * spt_drv_n), f);
    }
    fclose(f);
    return 0;
}
"""

_RUNNER = r"""
static void spt_drv_run_{{ALGO}}(void)
{
    static spt_{{A}}_t st;
    SPT_REAL in[{{NCOL}}], out[{{K}}];
    unsigned long long before[SPT_DRV_NOPS + 1];
    long k;
    int i;
    spt_{{A}}_defaults(&st);
    for (k = 0; k < spt_drv_n; ++k) {
        const double *row = spt_drv_in + spt_drv_cols * k;
{{PTR_DECLS}}
        for (i = 0; i < {{NCOL}}; ++i) {
            in[i] = SPT_DRV_REAL(row[i]);
        }
{{PTRS}}
        spt_drv_snap(before);
        (void)spt_{{A}}_step(&st, {{ARGS}}, SPT_DRV_REAL(spt_drv_dt));
        spt_drv_count(k, before);
        spt_{{A}}_output(&st, out);
        for (i = 0; i < {{K}}; ++i) {
            spt_drv_out[{{K}} * k + i] = SPT_DRV_DBL(out[i]);
        }
    }
}
"""


def _columns(r: _Resolved) -> list[tuple[str, int, int]]:
    """(giriş, ilk sütun, genişlik)."""
    out, col = [], 0
    for key in r.inputs:
        w = 2 if _is_vec(key) else 1
        out.append((key, col, w))
        col += w
    return out


def _driver_source(r: _Resolved) -> str:
    k = len(fg.SCENARIOS[r.scenario].outputs)
    cols = _columns(r)
    ncol = sum(w for _, _, w in cols)
    runners, dispatch = [], []
    ptr_decls, ptrs, args = [], [], []
    for key, c0, w in cols:
        if key == "gnss":
            ptr_decls.append("        const SPT_REAL *p_gnss;")
            ptrs.append(f"        p_gnss = spt_drv_both_nan(row + {c0}) ? NULL : &in[{c0}];")
            args.append("p_gnss")
        elif key == "gnss_ll":
            ptr_decls.append("        const double *p_ll;")
            ptrs.append(f"        p_ll = spt_drv_both_nan(row + {c0}) ? NULL : row + {c0};")
            args.append("p_ll")
        else:
            args.append(f"in[{c0}]")
    for algo in r.algorithms:
        runners.append(_fill(_RUNNER, ALGO=algo, A=f"{r.sab}_{algo}", NCOL=str(ncol), K=str(k),
                             PTR_DECLS="\n".join(ptr_decls), PTRS="\n".join(ptrs), ARGS=", ".join(args)))
        dispatch.append(f'    if (strcmp(argv[1], "{algo}") == 0) {{\n        spt_drv_run_{algo}();\n    }} else')
    both_nan = ("\nstatic int spt_drv_both_nan(const double *v)\n{\n    return v[0] != v[0] && v[1] != v[1];\n}\n"
                if ptrs else "")
    return _fill(_DRIVER, INCLUDES="\n".join(f'#include "spt_{r.sab}_{a}.h"' for a in r.algorithms),
                 RUNNERS="".join(runners), DISPATCH="\n".join(dispatch), K=str(k), BOTH_NAN=both_nan)


_COUNT_EXTRA = r"""
/* Genel füzyon ekleri: pow = exp + log + mul olarak sayılır (newlib powf ≈ bu maliyet). */
inline spt_cnt spt_cnt_pow(spt_cnt a, spt_cnt b)
{ ++spt_ops[SPT_OP_EXP]; ++spt_ops[SPT_OP_LOG]; ++spt_ops[SPT_OP_MUL]; return spt_cnt(std::pow(a.v, b.v)); }
#define SPT_POW(x, y) spt_cnt_pow((x), (y))
#define SPT_NAN spt_cnt(NAN)
#define SPT_TINY spt_cnt(1e-300)
#define SPT_HUGE spt_cnt(1e300)
"""


def host_program(opts: GenericCodegenOptions, *, real: str | None = None, count_ops: bool = False) -> tc.Build:
    """Üretilen C'yi ana makinede derler (doğrulama sürücüsüyle). count_ops: C++ olarak, SPT_REAL = sayaç türü."""
    r = _resolve(opts)
    real = real or opts.real
    files = {f.name: f.text for f in _sources(opts, r)}
    files["spt_host_driver.c"] = _driver_source(r)
    sources = [f"spt_{r.sab}_{a}.c" for a in r.algorithms] + ["spt_host_driver.c"]
    if count_ops:
        files["spt_count_prelude.hpp"] = _count_prelude() + _COUNT_EXTRA
        return tc.host_build(files, sources, cxx=True, extra=["-std=c++17", "-include", "spt_count_prelude.hpp"],
                             opt="-O2", output="spt_gcount")
    return tc.host_build(files, sources, defines=[f"SPT_USE_DOUBLE={1 if real == 'double' else 0}"],
                         extra=["-std=c99", "-Wall", "-Wextra", "-pedantic"], opt="-O2", output=f"spt_ghost_{real}")


def _input_matrix(r: _Resolved, inputs: dict, n: int) -> np.ndarray:
    scen = fg.SCENARIOS[r.scenario]
    arrays, _ = input_arrays(scen.inputs, {k: v for k, v in inputs.items() if k in r.inputs})
    data = np.full((n, sum(w for _, _, w in _columns(r))), np.nan)
    for key, c0, w in _columns(r):
        if key in arrays:
            a = arrays[key]
            data[:, c0:c0 + w] = a.reshape(n, w)
    return data


def run_generated(opts: GenericCodegenOptions, algorithm: str, inputs: dict, dt: float, *,
                  real: str | None = None, count_ops: bool = False) -> HostRun:
    """Üretilen C algoritmasını ana makinede verilen girişlerle (SI, NaN = yok; Estimator.run ile aynı sözlük)
    çalıştırır. Eşlenmeyen anahtarlar yok sayılır; eşlenen ama verilmeyen girişler tümü NaN kabul edilir."""
    r = _resolve(opts)
    if algorithm not in r.algorithms:
        raise ValueError(f"{algorithm!r} seçili algoritmalar arasında değil")
    build = host_program(opts, real=real, count_ops=count_ops)
    if not build.ok:
        raise RuntimeError("Ana makine derlemesi başarısız:\n" + build.log)
    n = len(next(iter(inputs.values())))
    data = _input_matrix(r, inputs, n)
    k = len(fg.SCENARIOS[r.scenario].outputs)
    blob = np.r_[float(n), float(dt), float(data.shape[1]), data.ravel()].astype("<f8").tobytes()
    tag = hashlib.sha256(blob + str(build.workdir).encode()).hexdigest()[:20]
    runs = GEN_ROOT / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    inp, out = runs / f"in_{tag}.bin", runs / f"out_{algorithm}_{tag}.bin"
    if not out.exists():
        inp.write_bytes(blob)
        res = tc.run_program(build, [algorithm, str(inp), str(out)], timeout=600)
        inp.unlink(missing_ok=True)
        if res.returncode != 0 or not out.exists():
            raise RuntimeError(f"Sürücü hatası ({res.returncode}): {res.stderr}")
    raw = np.fromfile(out, dtype="<f8")
    y = raw[:k * n].reshape(n, k)
    ops = raw[k * n:].reshape(n, len(OPS)) if count_ops else None
    return HostRun(y, ops)


# ============================================================================ ayak izi
_FP_CACHE: dict[str, list[dict]] = {}
_FP_VERSION = 1


def _harness(r: _Resolved, algo: str, baseline: bool) -> str:
    a = f"{r.sab}_{algo}"
    k = len(fg.SCENARIOS[r.scenario].outputs)
    cols = _columns(r)
    nin = sum(w for key, _, w in cols if key != "gnss_ll")
    lines, args = [], []
    idx = 0
    for key, _c0, w in cols:
        if key == "gnss":
            lines.append(f"        v[0] = spt_in[{idx}];\n        v[1] = spt_in[{idx + 1}];")
            args.append("v")
            idx += 2
        elif key == "gnss_ll":
            lines.append("        ll[0] = spt_in_ll[0];\n        ll[1] = spt_in_ll[1];")
            args.append("ll")
        else:
            args.append(f"spt_in[{idx}]")
            idx += 1
    extra = _extra_decls(r, algo)
    getters = re.findall(r"SPT_REAL (spt_\w+)\(const", extra)
    gsum = "".join(f" + {g}(&st)" for g in getters)
    outs = "".join(f" + spt_{a}_{o.key}(&st)" for o in fg.SCENARIOS[r.scenario].outputs)
    calls = "" if baseline else f"""
        spt_out[7] = SPT_FROM_INT(spt_{a}_step(&st, {', '.join(args)}, spt_in[{nin}]));
        spt_{a}_update(&st, {', '.join(args)}, spt_in[{nin}]);
        spt_{a}_output(&st, o);
        spt_out[5] = SPT_C(0.0){outs}{gsum};
        if (spt_in[{nin}] < SPT_C(0.0)) {{
            spt_out[6] = SPT_FROM_INT(spt_{a}_reset(&st));
        }}"""
    copy = "".join(f"        o[{i}] = spt_in[{i % (nin + 1)}];\n" for i in range(k)) if baseline else ""
    init = "" if baseline else f"    spt_out[6] = SPT_FROM_INT(spt_{a}_init(&st));\n"
    state = "" if baseline else f"static spt_{a}_t st;\n"
    vec_decl = ("    SPT_REAL v[2];\n" if "gnss" in r.inputs else "") + ("    double ll[2];\n" if "gnss_ll" in r.inputs
                                                                          else "")
    vec_use = ""
    if baseline and "gnss" in r.inputs:
        vec_use += "        o[0] = o[0] + v[0] + v[1];\n"
    if baseline and "gnss_ll" in r.inputs:
        vec_use += "        o[0] = o[0] + (SPT_REAL)(ll[0] + ll[1]);\n"
    return f"""/* Ayak izi ölçüm koşumu ({'temel' if baseline else a}): tüm genel API çağrılır, giriş / çıkış volatile. */
#include "spt_{a}.h"
{state}volatile SPT_REAL spt_in[{nin + 1}];
volatile double spt_in_ll[2];
volatile SPT_REAL spt_out[8];
int main(void);
int main(void)
{{
{vec_decl}    SPT_REAL o[{k}];
    int i;
{init}    for (;;) {{
{chr(10).join(lines)}{calls}
{copy}{vec_use}        for (i = 0; i < {k}; ++i) {{
            spt_out[i] = o[i];
        }}
    }}
}}
"""


def _measure(opts: GenericCodegenOptions, prof: McuProfile, compiler: str) -> list[dict] | None:
    """Bir profil + derleyici için algoritma başına ölçüm; derleyici yoksa None."""
    tool = tc.cross_tool(compiler)
    if tool is None or tc.cross_flags(prof, compiler) is None:
        return None
    r = _resolve(opts)
    files = {f.name: f.text for f in _sources(opts, r)}
    real_def = f"SPT_USE_DOUBLE={1 if opts.real == 'double' else 0}"
    probe = "\n".join(f'#include "spt_{r.sab}_{a}.h"' for a in r.algorithms) + "\n" + \
        "\n".join(f"unsigned char spt_sizeof_{a}[sizeof(spt_{r.sab}_{a}_t)];" for a in r.algorithms) + "\n"
    files["spt_sizeof_probe.c"] = probe
    gcc = tc.find_arm_gcc()
    key_blob = json.dumps([_FP_VERSION, tc.STACK_ANALYSIS_VERSION, sorted(files.items()), prof.key, compiler, opts.opt,
                           real_def,
                           tool.path, tool.version, gcc.version if gcc else None])
    key = hashlib.sha256(key_blob.encode()).hexdigest()[:24]
    cache_file = GEN_ROOT / "footprint" / f"{key}.json"
    if key in _FP_CACHE:
        return _FP_CACHE[key]
    if cache_file.exists():
        try:
            _FP_CACHE[key] = json.loads(cache_file.read_text(encoding="utf-8"))
            return _FP_CACHE[key]
        except (OSError, ValueError):
            pass
    sources = [f"spt_{r.sab}_{a}.c" for a in r.algorithms] + ["spt_sizeof_probe.c"]
    build = tc.compile_cross(files, sources, prof, compiler, opts.opt, defines=[real_def], warnings=())
    if not build.ok:
        raise RuntimeError(f"{compiler} {prof.key} derleme hatası:\n{build.log}")
    syms = tc.elf_symbols(build.objects["spt_sizeof_probe.c"])
    su = build.stack_usage()
    rows = []
    linkable = gcc is not None and prof.gcc_flags is not None
    base_flash = None
    common = {n: t for n, t in files.items() if n.endswith("_common.h")}
    if linkable:
        a0 = r.algorithms[0]
        bfiles = dict(common, **{f"spt_{r.sab}_{a0}.h": files[f"spt_{r.sab}_{a0}.h"],
                                 "spt_fp_base.c": _harness(r, a0, True)})
        base = tc.link_cross(bfiles, ["spt_fp_base.c"], prof, compiler, opts.opt, defines=[real_def])
        base_flash = tc.elf_sizes(base.output).flash if base.ok else None
    for algo in r.algorithms:
        a = f"{r.sab}_{algo}"
        sz = build.sizes(f"spt_{a}.c")
        row = {"algorithm": algo, "flash_code": sz.flash, "ram_state": syms[f"spt_sizeof_{algo}"].size,
               "flash_with_libm": None, "stack": None, "stack_path": [], "su_core": su.get(f"spt_{a}_step")}
        if linkable and base_flash is not None:
            hfiles = dict(common, **{f"spt_{a}.h": files[f"spt_{a}.h"], f"spt_{a}.c": files[f"spt_{a}.c"],
                                     "spt_fp_harness.c": _harness(r, algo, False)})
            img = tc.link_cross(hfiles, ["spt_fp_harness.c", f"spt_{a}.c"], prof, compiler, opts.opt,
                                defines=[real_def])
            if img.ok:
                row["flash_with_libm"] = tc.elf_sizes(img.output).flash - base_flash
                entries = [f"spt_{a}_step", f"spt_{a}_reset"]
                st = tc.stack_analysis(img.output, entries)
                own = tc.stack_analysis(img.output, entries, include=lambda name: name.startswith("spt_"))
                if f"spt_{a}_step" in st:
                    info = st[f"spt_{a}_step"]
                    row["stack"] = info.bytes
                    row["stack_path"] = list(info.path)
                    row["stack_exact"] = info.exact
                    row["stack_own"] = own[f"spt_{a}_step"].bytes if f"spt_{a}_step" in own else None
                    row["frame_step"] = info.frame
        rows.append(row)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(rows), encoding="utf-8")
    _FP_CACHE[key] = rows
    return rows


# Derleyici bulunamazsa son çare: Cortex-M4F, arm-none-eabi-gcc 13.3 -Os, float ölçümü (bayt; PF 500 parçacık).
# (flash_code, flash_with_libm, stack) — ölçeklenerek kullanılır, satırda "tahmin" yazar.
_REFERENCE = {
    ("altitude", "complementary"): (690, 2900, 136), ("altitude", "kf"): (1000, 3220, 176),
    ("altitude", "ekf"): (1100, 3320, 112), ("altitude", "ukf"): (1460, 3730, 352),
    ("altitude", "pf"): (3010, 6440, 264),
    ("position2d", "dead_reckoning"): (1180, 6730, 560), ("position2d", "ekf"): (2180, 7770, 680),
    ("position2d", "ukf"): (3820, 9470, 792), ("position2d", "pf"): (4090, 10950, 704),
    ("linear_multi", "weighted_mean"): (520, 720, 128), ("linear_multi", "kf"): (1430, 1620, 320),
    ("linear_multi", "pf"): (3630, 5220, 488),
}
_REF_CORE = {"none": 1.6, "sp": 1.0, "dp": 1.0}
_REF_OPT = {"-O0": 2.2, "-Os": 1.0, "-O2": 1.25, "-O3": 1.6}


def _reference_row(opts: GenericCodegenOptions, r: _Resolved, prof: McuProfile, compiler: str, algo: str,
                   why: str) -> FootprintRow:
    f, fl, st = _REFERENCE[(r.scenario, algo)]
    k = _REF_CORE[prof.fpu] * _REF_OPT[opts.opt] * prof.density
    if opts.real == "double" and prof.double_is_soft() and prof.fpu != "none":
        k *= 1.5
    k *= {"gcc": 1.0, "clang": 1.0, "iar": IAR_SIZE_FACTOR}[compiler]
    return FootprintRow(algo, compiler, int(round(f * k)), int(round(fl * k)), state_size(opts, algo),
                        int(round(st * prof.density)), False,
                        f"tahmin: {why}; önceden ölçülmüş Cortex-M4F GCC -Os referansından ölçeklendi (×{k:.2f}); RAM "
                        f"durum = sizeof (yapı düzeninden kesin)")


def footprint(opts: GenericCodegenOptions, compilers: tuple[str, ...] | None = None) -> list[FootprintRow]:
    """Algoritma × derleyici başına flash / RAM / yığın. Yerel derleyici varsa ölçülür, yoksa tahmin edilir."""
    r = _resolve(opts)
    prof = profile(opts.mcu)
    compilers = tuple(compilers or (opts.compiler,))
    rows: list[FootprintRow] = []
    measured: dict[tuple[str, str], list[dict] | None] = {}

    def get(p: McuProfile, comp: str) -> list[dict] | None:
        if (p.key, comp) not in measured:
            measured[(p.key, comp)] = _measure(opts, p, comp) if p.measurable else None
        return measured[(p.key, comp)]

    for comp in compilers:
        if comp not in COMPILERS:
            raise ValueError(f"Bilinmeyen derleyici: {comp!r}")
        if comp == "iar" and prof.arch == "xtensa":
            continue                                     # IAR'ın Xtensa derleyicisi yok
        base_prof = prof if prof.measurable else MCU_PROFILES[prof.estimate_base]
        src_comp = "gcc" if comp == "iar" else comp
        data = get(base_prof, src_comp)
        if data is None:
            why = f"{COMPILERS[src_comp]} bulunamadı" if base_prof.measurable else "derleyici yok"
            rows += [_reference_row(opts, r, prof, comp, a, why) for a in r.algorithms]
            continue
        tool = tc.cross_tool(src_comp)
        tool_label = f"{tool.label}" if tool else src_comp
        for d in data:
            stack_note = ""
            if d.get("stack") is not None:
                libm = [f for f in d.get("stack_path", []) if not f.startswith("spt_")]
                stack_note = (f"; yığın: objdump statik analizi, en kötü yol step → çekirdek (libm dahil "
                              f"{d['stack']} B, yalnız bu kod {d.get('stack_own')} B"
                              + (f"; en derin libm zinciri {' → '.join(libm)}" if libm else "") + ")")
                if d.get("stack_exact") is False:
                    stack_note += " — çözülemeyen yığın ayarı var, alt sınır"

            def scaled(v: int | None, k: float) -> int | None:
                return None if v is None else int(round(v * k))

            if comp == "iar":
                rows.append(FootprintRow(
                    d["algorithm"], comp, int(round(d["flash_code"] * IAR_SIZE_FACTOR * prof.density)),
                    scaled(d["flash_with_libm"], IAR_LIBM_FACTOR * prof.density), d["ram_state"],
                    scaled(d["stack"], prof.density), False,
                    f"tahmin: IAR yok — {tool_label} {opts.opt} ({base_prof.name}) ölçümü × {IAR_SIZE_FACTOR}"
                    + (f" × {prof.density} kod yoğunluğu" if not prof.measurable else "")
                    + "; RAM durum = sizeof (ABI aynı, kesin)", scaled(d.get("stack_own"), prof.density)))
            elif prof.measurable:
                rows.append(FootprintRow(
                    d["algorithm"], comp, d["flash_code"], d["flash_with_libm"], d["ram_state"], d["stack"], True,
                    f"{tool_label} {opts.opt}, {prof.name}, {opts.real}: ölçüldü (nesne .text+.rodata+.data; "
                    f"libm: newlib-nano ile bağlanmış imaj − boş temel; RAM: sizeof){stack_note}",
                    d.get("stack_own")))
            else:
                k = prof.density
                rows.append(FootprintRow(
                    d["algorithm"], comp, int(round(d["flash_code"] * k)), scaled(d["flash_with_libm"], k),
                    d["ram_state"], scaled(d["stack"], k), False,
                    f"tahmin: {prof.arch} derleyicisi yok — {tool_label} {opts.opt} ({base_prof.name}) ölçümü × "
                    f"{k} kod yoğunluğu; RAM durum = sizeof (ILP32 düzeni aynı, kesin)", scaled(d.get("stack_own"), k)))
    return rows


# ============================================================================ maliyet
def _cost_data(r: _Resolved, fs: float) -> tuple[dict, float]:
    """Eşlenen girişlere uygun simülasyon (maliyet ölçümü için): girişler + dt."""
    scen = fg.SCENARIOS[r.scenario]
    dur = COST_SAMPLES / fs
    if r.scenario == "altitude":
        sim = scen.simulate(dur, fs, COST_SEED, baro="pressure" if "baro_p" in r.inputs else "altitude")
    elif r.scenario == "position2d":
        sim = scen.simulate(dur, fs, COST_SEED, speed="speed" in r.inputs, acc="acc_fwd" in r.inputs,
                            latlon="gnss" not in r.inputs)
    else:
        rates = (100.0, 20.0, 50.0, 100.0, 25.0, 10.0)
        sensors = tuple((r.values[f"s{i}_kind"], r.values[f"s{i}_sigma"], rates[i - 1]) for i in range(1, 7))
        sim = scen.simulate(dur, fs, COST_SEED, sensors=sensors, order=r.values["order"])
    return sim.inputs, sim.dt


def cost(opts: GenericCodegenOptions) -> list[CostRow]:
    """Adım başına işlem sayısı (ölçülür; başlatma sonrası ortalama, en kötü adım başlatma dahil) → çevrim / süre /
    CPU yükü (MCU tablosundan tahmin)."""
    r = _resolve(opts)
    prof = profile(opts.mcu)
    inputs, dt = _cost_data(r, opts.fs_hz)
    weights = _op_weights(prof, opts.real)
    rows = []
    for a in r.algorithms:
        run = run_generated(opts, a, inputs, dt, count_ops=True)
        valid = np.isfinite(run.out).all(axis=1)
        start = int(np.argmax(valid)) + 1 if valid.any() else 0
        ops = run.ops[start:] if start < len(run.ops) else run.ops
        mean = ops.mean(axis=0)
        cyc_all = run.ops @ weights
        cycles = float((ops @ weights).mean())
        us = cycles / opts.clock_mhz
        rows.append(CostRow(
            a, {op: float(v) for op, v in zip(OPS, mean)}, cycles, us, us * opts.fs_hz / 1e4,
            1e6 / us if us > 0 else math.inf,
            f"işlem sayısı ölçüldü (C++ sayaç, simüle veri {len(run.ops)} adım @ {opts.fs_hz:g} Hz; ortalama başlatma "
            f"sonrası {len(ops)} adım, en kötü adım başlatma dahil; pow = exp + log + mul), çevrim çekirdek "
            f"tablosundan tahmin ({prof.name}, {opts.real}"
            f"{', double yazılım kayan nokta' if opts.real == 'double' and prof.double_is_soft() else ''})",
            False, float(cyc_all.max()), float(cyc_all.max()) / opts.clock_mhz))
    return rows


# ============================================================================ kullanım kılavuzu
def _title(r: _Resolved, algo: str) -> str:
    return fg.SCENARIOS[r.scenario].algorithms[algo].name


def _footprint_table(opts: GenericCodegenOptions, r: _Resolved, rows: list[FootprintRow]) -> str:
    lines = ["| Algoritma | Derleyici | Flash (kod) | Flash (libm dahil) | RAM (durum) | Yığın (libm dahil / bu kod) "
             f"| Flash % ({opts.flash_kb:g} kB) | RAM % ({opts.ram_kb:g} kB) | Ölçüm |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for x in rows:
        total_flash = x.flash_with_libm if x.flash_with_libm is not None else x.flash_code
        ram = x.ram_state + (x.stack or 0)
        lines.append(f"| {_title(r, x.algorithm)} | {COMPILERS[x.compiler]} | {_fmt_int(x.flash_code)} B | "
                     f"{_fmt_int(x.flash_with_libm)}{' B' if x.flash_with_libm is not None else ''} | "
                     f"{_fmt_int(x.ram_state)} B | {_fmt_int(x.stack)} / {_fmt_int(x.stack_own)} B | "
                     f"{100 * total_flash / (opts.flash_kb * 1024):.1f} | {100 * ram / (opts.ram_kb * 1024):.1f} | "
                     f"{'ölçüldü' if x.measured else 'tahmin'} |")
    methods = sorted({re.sub(r"; yığın: .*$", "", x.method) for x in rows})
    return ("\n".join(lines) + "\n\nRAM % = durum + en kötü yığın (libm dahil). Libm dahil yığın muhafazakârdır "
            "(newlib sinf / cosf / powf'un büyük argüman yolları statik olarak en derindir); \"bu kod\" sütunu libm "
            "fonksiyonlarını 0 baytlık yaprak sayar.\n\n" + "\n".join(f"- {m}" for m in methods))


def _cost_table(opts: GenericCodegenOptions, r: _Resolved, rows: list[CostRow]) -> str:
    lines = [f"| Algoritma | Çevrim / adım (ort.) | En kötü adım | µs / adım @ {opts.clock_mhz:g} MHz | "
             f"CPU yükü @ {opts.fs_hz:g} Hz | En yüksek hız |", "|---|---:|---:|---:|---:|---:|"]
    for x in rows:
        lines.append(f"| {_title(r, x.algorithm)} | {_fmt_int(x.cycles)} | {_fmt_int(x.cycles_max)} | "
                     f"{x.us_per_update:.1f} | %{x.cpu_load_percent:.2f} | {_fmt_int(x.max_rate_hz)} Hz |")
    ops = ["| Algoritma | " + " | ".join(OPS) + " |", "|---|" + "---:|" * len(OPS)]
    for x in rows:
        ops.append(f"| {_title(r, x.algorithm)} | " + " | ".join(f"{x.ops[o]:.1f}" for o in OPS) + " |")
    return "\n".join(lines) + "\n\n" + (rows[0].method if rows else "") + \
        "\n\nAdım başına ortalama işlem sayıları (ölçülen):\n\n" + "\n".join(ops)


_SCEN_NOTES = {
    "altitude": """- **Başlatma:** ilk geçerli baro ölçümünde h = z_h (basınçta ISA ile), v = 0, b = 0;
  P0 = diag(R_h, (1 m/s)², σ_b0²). Öncesinde çıktılar NaN, `step()` 0 döner.
- **Basınç / baro irtifa:** ikisi de eşlenirse örnekte basınç varsa o kullanılır (basınç ≤ 0 ya da NaN → yok).
- **p0 = 0 (otomatik):** ilk geçerli basınç örneği h = 0 kabul edilir (göreli irtifa); kullanılan değer `s.c.p0`.
  0 < p0 < 1000 Pa geçersizdir (`init()` 0 döner).
- **Eksik ivme (NaN / sonsuz):** son geçerli değer tutulur ve her eksik örnekte hıza 2 m/s/√s rastgele yürüyüş eklenir
  (Kalman ailesi Q'ya, parçacık filtresi gürültüye); hiç ivme gelmemişse a = 0. Filtre durmaz, barometre taşır.
- **İvme ekseni:** dünya çerçevesinde yukarı +, yerçekimi dahil (durağanken +g). Eğik bir gövdede doğrudan ivmeölçer
  z'si verilirse cos(eğim) ölçek hatası sapma kestirimine karışır; bir AHRS'nin R(q)·a çıktısının z'sini kullanın.""",
    "position2d": """- **Başlatma:** GNSS izi `d_init` metreyi aşınca: konum son düzeltme, başlık kirişin yönü + kiriş
  boyunca jiroskopla dönülen açının yarısı, hız odometriden ya da kiriş / süre (ileri hareket varsayılır). Öncesinde
  `out[0..1]` son GNSS düzeltmesi (yoksa NaN), başlık ve hız NaN; `step()` 0 döner.
- **Başlatma geçmişi:** son 30 s'nin GNSS düzeltmeleri `SPT_POS_FIX_HISTORY` (160) boyutlu halka tamponda, en az
  30 s / 159 ≈ 0.19 s aralıkla tutulur (5.3 Hz'e kadar her düzeltme): her GNSS hızında 30 s kapsanır ve başlatma
  Python ile aynıdır. RAM: düzeltme başına 3 gerçek sayı + 4 bayt.
- **GNSS (metre / enlem-boylam):** ikisi de eşlenirse metre girişi kullanılır, enlem / boylam yok sayılır (Python ile
  aynı). Enlem / boylam ilk geçerli düzeltme etrafında yerel teğet düzleme açılır; giriş her zaman `double` (rad)
  alınır (float32 enlem ~0.5 m çözünürlük verir). Orijin `s.c.lat0 / s.c.lon0`.
- **Eksik jiroskop:** son geçerli değer tutulur, başlığa 0.2 rad/√s yürüyüş eklenir. GNSS kesintisinde (tünel) ölü
  hesapla devam edilir.
- **Başlık:** doğu ekseninden saat yönünün tersine (rad); pusula başlığı = 90° − ψ.""",
    "linear_multi": """- **Başlatma:** konum türünde yuva eşlenmişse ilk konum ölçümünün geldiği örnekte: p = o örnekteki
  konum ölçümlerinin ters varyans ortalaması (varyans 1/Σ(1/R)); hız / ivme o örnekte ölçüldüyse onlar, değilse
  0 ± σ0. Konum yuvası hiç yoksa konum göreli başlar (ilk ölçüm anında 0). Eşlenmiş ama hiç veri gelmeyen konum yuvası
  da "konum yuvası var" sayılır (Python ile aynı).
- **Model derecesi:** sabit seviye / hız / ivme (`s.cfg.order`); eşlenen her yuvanın türü dereceden büyük olamaz
  (`init()` / `reset()` 0 döner). Modelin kapsamadığı türev çıktıları 0'dır (ör. sabit hızda ivme = 0).
- **Ağırlıklı ortalama:** zamanda füzyon yoktur; ölçüm gelmeyen örnekte son değer tutulur, hiç sensörü olmayan tür NaN.
- **Parçacık filtresi:** durum dizisi `SPT_LIN_PF_DMAX` (varsayılan derece + 1) boyutludur.""",
}


def usage_markdown(opts: GenericCodegenOptions, fp: list[FootprintRow] | None = None,
                   cost: list[CostRow] | None = None) -> str:
    """Türkçe kullanım kılavuzu (KULLANIM_<senaryo>.md)."""
    r = _resolve(opts)
    p, P = opts.prefix.lower(), opts.prefix.upper()
    prof = profile(opts.mcu)
    scen = fg.SCENARIOS[r.scenario]
    float_pos_note = ("""

**2B konumda float ve orijinden uzaklık:** konum (x, y) orijine göre `float` tutulur (enlem / boylam girişinde orijin
ilk düzeltmedir; metre girişinde sizin koordinatlarınızın orijini). Float'ın çözünürlüğü uzaklıkla kabardığından her
adımdaki küçük yer değiştirmenin yuvarlaması birikir. Denetimde ölçülen (EKF, 100 Hz, 10⁶ adım): orijinden **20 km'ye
kadar** float ile double aynı doğrulukta (fark ≤ %2); **70–90 km'de** konum RMSE ~2× (0.52–0.64 m / 0.30 m), hız RMSE
~2.4× (0.105 / 0.043 m/s). Geniş alanda çalışan araçlarda **double** seçin; ya da enlem / boylamda araç durduğunda
`reset()` ile orijini yenileyin, metre girişinde koordinatları yakın bir orijine göre (double'da çıkarıp) verin."""
                      if r.scenario == "position2d" and opts.real == "float" else "")
    s = r.sab
    first = f"{s}_{r.algorithms[0]}"
    k = len(scen.outputs)
    file_rows = [f"| `{p}_fusion_generic_common.h` | Sayı türü (`{P}_REAL`), matematik makroları, `{P}_NAN`, açı sarma, "
                 "rastgele sayı üreteci, Cholesky, Liu–West (static inline) |",
                 f"| `{p}_{s}_common.h` | {scen.name}: senaryo parametreleri (`#define`), giriş kuralları, ortak "
                 "model parçaları |"]
    for a in r.algorithms:
        file_rows.append(f"| `{p}_{s}_{a}.h` / `{p}_{s}_{a}.c` | {_title(r, a)}: durum yapısı `{p}_{s}_{a}_t`, API |")
    file_rows.append(f"| `{usage_names(r.sab)[1]}` | Sabit hızda çağıran örnek görev (sensör okuma / bekleme kullanıcıya "
                     "ait) |")
    sig = _signature(r).replace("SPT_", f"{P}_")
    in_rows = []
    for key in r.inputs:
        inp = scen.input(key)
        decl = _param_decl(key).replace("SPT_", f"{P}_")
        in_rows.append(f"| `{key}` | {inp.label} | `{decl}` | {_input_doc(r, key)} |")
    out_rows = [f"| {i} | `{o.key}` | {o.label} | {'rad' if o.angle else o.unit} | `{p}_{first}_{o.key}(&s)` |"
                for i, o in enumerate(scen.outputs)]
    params = []
    for prm in scen.params:
        v = r.values[prm.key]
        m = re.fullmatch(r"s(\d)_(kind|sigma)", prm.key)
        field = (f"s.cfg.{'kind' if m.group(2) == 'kind' else 'sigma'}[{int(m.group(1)) - 1}]" if m
                 else f"s.cfg.{prm.key}")
        val = f"{_choice_index(r.scenario, None, prm.key, v)} ({v})" if isinstance(prm, Choice) else f"{v:g}"
        dflt = prm.default if isinstance(prm, Choice) else f"{prm.default:g}"
        params.append(f"| senaryo | `{P}_{s.upper()}_{prm.key.upper()}` | {prm.label} | {val} | {dflt} | `{field}` |")
    for a in r.algorithms:
        cls = scen.algorithms[a]
        for prm in cls.params:
            v = r.params[a][prm.key]
            macro = f"{P}_{s.upper()}_{a.upper()}_{prm.key.upper()}"
            if a == "pf" and prm.key == "particles":
                runtime = "derleme anı (dizi boyutu)"
            elif a == "pf" and prm.key == "seed":
                runtime = "`s.seed` + `reset()`"
            else:
                runtime = f"`s.{prm.key}`"
            val = f"{_choice_index(r.scenario, a, prm.key, v)} ({v})" if isinstance(prm, Choice) else f"{v:g}"
            dflt = prm.default if isinstance(prm, Choice) else f"{prm.default:g}"
            params.append(f"| {_title(r, a)} | `{macro}` | {prm.label} | {val} | {dflt} | {runtime} |")
    extra = []
    for a in r.algorithms:
        kind = _EXTRA_DECLS.get((r.scenario, a))
        if kind == "acc_bias":
            extra.append(f"- `{p}_{s}_{a}_acc_bias(&s)`: ivme sapması kestirimi (m/s²).")
        elif kind == "gyro_odo":
            extra.append(f"- `{p}_{s}_{a}_gyro_bias(&s)` (rad/s) / `{p}_{s}_{a}_odo_scale(&s)`: jiroskop sapması ve "
                         "odometri ölçek kestirimi.")
    pf_note = ""
    if "pf" in r.algorithms:
        n = _particles(r)
        pf_note = (f"\n## Parçacık filtresi\n\n- Parçacık sayısı derleme anında `{P}_{s.upper()}_PF_PARTICLES` "
                   f"(bu üretimde {n}); durum yapısı büyüktür ({_fmt_int(state_size(opts, 'pf'))} B, `{opts.real}`) → "
                   "`static` ya da global ayırın, yığına koymayın.\n- Rastgele sayı üreteci xorshift32 + Marsaglia "
                   "kutupsal (Python numpy'den farklı): sonuçlar istatistiksel olarak eşdeğerdir, örnek örnek aynı "
                   "değildir. Tohum `s.seed` (`reset()` ile etkinleşir).\n- Ağırlık güncellemesi ve yeniden örnekleme "
                   "yalnız ölçüm örneklerindedir; en kötü adım (yeniden örnekleme) ortalamanın birkaç katıdır "
                   "(maliyet tablosu). `s.ess`, `s.resamples` tanılama içindir.\n")
        if r.scenario == "position2d":
            pf_note += ("- Not (Python tasarımıyla aynı davranış): odometri ölçeği parçacıkları çeşitliliğini yitirirse "
                        "GNSS ağırlıkları yanlılığı düzeltemez ve konum yavaşça ıraksayabilir (varsayılan simülasyonda "
                        "tohumların ~%10'unda, Python'da da aynı oranda). Üretimde EKF / UKF önerilir.\n")
    real_note = ""
    if opts.real == "double" and prof.double_is_soft():
        real_note = (f"\n> **Uyarı:** `double` seçildi ama {prof.name} çekirdeğinde çift hassasiyet donanımı yok; tüm "
                     "hesap yazılım kayan noktayla yapılır. Mümkünse `float` kullanın.\n")
    fpu = {"none": "FPU yok: `-mfloat-abi=soft` (tüm kayan nokta yazılımda)",
           "sp": "tek hassasiyet FPU: `float` önerilir; `double` yazılım kayan noktadır (10–100× yavaş)",
           "dp": "çift hassasiyet FPU: `double` de donanımda (yine de `float` daha hızlı ve küçük)"}[prof.fpu]
    flags = " ".join(prof.gcc_flags) if prof.gcc_flags else "(bu çekirdek için bayraklar üreticinin SDK'sına bakın)"
    fp_md = _footprint_table(opts, r, fp) if fp else \
        "_Bu dosya ölçümsüz üretildi; ayak izi tablosu için `footprint()` sonuçlarıyla yeniden üretin._"
    cost_md = _cost_table(opts, r, cost) if cost else \
        "_Bu dosya ölçümsüz üretildi; maliyet tablosu için `cost()` sonuçlarıyla yeniden üretin._"
    args = ", ".join(r.inputs)
    tune_key = {"altitude": "sigma_acc", "position2d": "sigma_gnss", "linear_multi": "sigma_q"}[r.scenario]
    tune_val = _c_float(r.values[tune_key] * 2) + ("f" if opts.real == "float" else "")
    nan_example = {
        "altitude": f"spt_{first}_step(&f, baro_hazir ? p_pa : {P}_NAN, acc_z, dt);",
        "position2d": f"spt_{first}_step(&f, {args}, dt);   /* GNSS yoksa NULL ya da NaN */",
        "linear_multi": f"spt_{first}_step(&f, {', '.join(f'{x}_hazir ? {x}_deger : {P}_NAN' for x in r.inputs[:2])}"
                        f"{', ...' if len(r.inputs) > 2 else ''}, dt);",
    }[r.scenario].replace("spt_", f"{p}_")
    text = f"""# {scen.name} — C kodu kullanım kılavuzu

Signal Processing Toolbox tarafından üretildi. Python referansı: `spt/dsp/fusion_scenarios/{_PYMOD[r.scenario]}.py`
(aynı denklemler, aynı senaryo / algoritma parametreleri, aynı başlatma, aynı NaN = "ölçüm yok" kuralı ve çıktılar).

**Seçimler:** algoritmalar {", ".join(_title(r, a) for a in r.algorithms)} · eşlenen girişler {", ".join(f"`{x}`" for x in r.inputs)}
· sayı türü `{opts.real}` · MCU {prof.name} ({prof.examples}) · derleyici {COMPILERS[opts.compiler]} `{opts.opt}` ·
saat {opts.clock_mhz:g} MHz · akış örnekleme hızı {opts.fs_hz:g} Hz
{real_note}
## Dosyalar

| Dosya | İçerik |
|---|---|
{chr(10).join(file_rows)}

Tüm durum çağıranın yapısındadır: `malloc` yok, global değişken yok, fonksiyonlar yeniden girişlidir. Kullandığınız
algoritmanın `.c` dosyası ve iki ortak başlık yeterlidir.

## Hızlı başlangıç

```c
#include "{p}_{first}.h"

static {p}_{first}_t f;                       /* durum (statik: yığında yer kaplamaz) */

void baslat(void) {{ (void){p}_{first}_init(&f); }}

/* zamanlayıcı kesmesinden ya da RTOS görevinden {opts.fs_hz:g} Hz'de; bu örnekte gelmeyen sensör: {P}_NAN */
void fuzyon_adimi({sig})
{{
    {P}_REAL out[{k}];
    if ({p}_{first}_step(&f, {args}, {P}_C({1.0 / opts.fs_hz!r}))) {{
        {p}_{first}_output(&f, out);         /* {_out_doc(r)} */
    }}
}}
```

## API (her algoritma için aynı; `X` ∈ {{{", ".join(f"{s}_{a}" for a in r.algorithms)}}})

| Fonksiyon | Açıklama |
|---|---|
| `{p}_X_defaults(&s)` | Parametreleri derleme anı varsayılanlarına getirir, durumu sıfırlar |
| `{p}_X_init(&s)` | `defaults`; parametreler geçersizse 0 döner (Python FusionError) |
| `{p}_X_reset(&s)` | Durumu başlangıca döndürür, parametreleri korur (çalışma anında değişenleri denetler) |
| `{p}_X_step(&s, {args}, dt)` | **Önerilen.** Bir örnek: NaN girişler "ölçüm yok", ilk uygun ölçümde kendiliğinden başlar; kestirim varsa 1 |
| `{p}_X_update(&s, {args}, dt)` | `step` ile aynı, dönüş değeri yok |
| `{p}_X_output(&s, out)` | Çıktılar Python `Scenario.outputs` sırasıyla; kestirim yoksa NaN |
| `{p}_X_<çıktı>(&s)` | Tek çıktı (aşağıdaki tablo) |

{chr(10).join(extra)}

## Girişler

| Anahtar | Giriş | C türü | Birim / kural |
|---|---|---|---|
{chr(10).join(in_rows)}

Girişler **SI birimindedir** (Python ile aynı); ham sensör birimlerini çağıran çevirir (ör. hPa × 100 → Pa,
°/s × π/180 → rad/s, g × 9.80665 → m/s²).

## Çıktılar

| # | Anahtar | Çıktı | Birim | Tek çıktı okuma |
|---|---|---|---|---|
{chr(10).join(out_rows)}

## Eksik ölçüm (NaN) ve çok hızlı sensörler

- `step()` her **akış örneğinde** çağrılır (en hızlı sensörün hızında, `dt` = 1/fs). Bu örnekte gelmeyen (seyrek)
  sensörün değeri `{P}_NAN` verilir: ör. 100 Hz akışta 5 Hz GNSS için 20 örnekten 19'unda NaN. Sonsuz ve aşırı büyük
  değerler de "yok" sayılır; vektör girişlerde NULL işaretçi de olur. Bu, Python'daki NaN kuralının aynısıdır.
- "Geçerli" bayrağı tutan sürücüler için: `{nan_example}`
- `-ffast-math` / `-ffinite-math-only` ile **derlemeyin** (NaN denetimleri kaldırılır).

## Senaryoya özgü kurallar

{_SCEN_NOTES[r.scenario]}

## Parametreler

Varsayılanlar derleme anında `#define` ile gelir (`#ifndef` korumalı: komut satırından ezilebilir) ve `defaults()` ile
yapıya kopyalanır. Çalışma anında yapı alanından değiştirilebilir (ör. `f.cfg.{tune_key} = {tune_val};`); değişiklikler
bir sonraki `step()`'te etkilidir (başlangıç belirsizlikleri ve tohum `reset()` ile).

| Kapsam | Makro | Anlam | Bu üretim | Python varsayılanı | Çalışma anında |
|---|---|---|---:|---:|---|
{chr(10).join(params)}
{pf_note}
## Projeye ekleme

### STM32CubeIDE (GCC)
1. Başlıkları `Core/Inc`, `.c` dosyasını `Core/Src` altına kopyalayın (ya da ayrı klasörü include yoluna ekleyin).
2. *MCU Settings*: FPU ve ABI. Bu çekirdek için bayraklar: `{flags}`.
3. *MCU GCC Linker → Libraries*: `m` (`-lm`). newlib-nano varsayılandır.
4. İsteğe bağlı *Preprocessor*: `{P}_USE_DOUBLE=0/1`, parametre makroları. **`-ffast-math` kullanmayın.**

### Keil MDK (Arm Compiler 6) / IAR EWARM
Dosyaları projeye ekleyin, include yolunu verin; FPU: "Single Precision" (M4F / M33F) ya da "Double Precision" (M7);
dil C99 ya da üstü; *Fast math* kapalı. AC6 clang tabanlıdır (boyutlar "Clang" satırlarına yakındır); IAR boyutları
**tahmindir** (GCC ölçümü × {IAR_SIZE_FACTOR}).

### ESP32 (ESP-IDF) / RISC-V
Dosyaları bir bileşenin `srcs` listesine ekleyin. ESP32'nin FPU'su tek hassasiyettir (`float`). FPU'suz çekirdeklerde
tüm kayan nokta yazılımdadır.

## Sabit hızda çağırma ve dt

- Parametrelerdeki örnek başına gürültüler (σ ivme, σ jiroskop …) akış hızına göredir: algoritmayı **sabit hızda**
  çağırın (donanım zamanlayıcısı; kesmede bayrak, hesap ana döngüde ya da RTOS görevinde).
- `dt` saniyedir; sabit hızda `dt = 1/fs`. Hız oynuyorsa ölçülen dt verilebilir (Cortex-M'de `DWT->CYCCNT`).
- Sensör FIFO'sundan toplu okuyorsanız örnekleri sırayla, her biri için kendi dt'siyle işleyin.

## Sayı türü ve FPU

{prof.name}: {fpu}. `{P}_USE_DOUBLE` ile seçilir. Sabitler `{P}_C()` makrosuyla doğru türdedir (`-Wdouble-promotion`
temiz). Matematik fonksiyonları `{P}_SQRT`, `{P}_POW` … makrolarıyla değiştirilebilir. `float`'ta sonuçlar Python'dan
~10⁻⁷ göreli farklıdır; doğruluk (RMSE) testlerdeki senaryolarda Python'un %5'i içindedir.{float_pos_note}

## Ölçülen ayak izi

{fp_md}

## Hesap maliyeti

{cost_md}

Maliyet modeli: çevrim = Σ temel işlem × (çekirdek çevrimi + bellek / dal ek yükü) + Σ kütüphane çağrısı × (fonksiyon
çevrimi + çağrı ek yükü). İşlem sayıları **ölçülmüştür** (üretilen C, ana makinede C++ sayaç türüyle simüle veriyle
çalıştırılır; pow = exp + log + mul); çekirdek çevrim tabloları **tahmindir** (`spt/codegen/mcu.py`).
CPU yükü = µs/adım × fs / 10⁴ (%).

## Python referansından farklar

- Yuvarlama düzeyinde: bazı bölmeler tersle çarpma, simetrik kovaryansların üst üçgeni, seyrek F·P·Fᵀ, tamsayı
  kuvvetler çarpımla. Çift hassasiyette Python ile en büyük fark < 10⁻⁶ (çıktı biriminde) olarak doğrulanmıştır
  (eksik / NaN ölçümlü akışlar dahil; NaN konumları aynı).
- Parçacık filtreleri: rastgele sayı üreteci, yeniden örneklemede parçacık sırası ve (ucuz olabilirliklerde) iki
  geçişli ağırlık güncellemesi farklı → istatistiksel olarak eşdeğer (RMSE testlerde Python × 1.5 içinde).
- UKF: Cholesky başarısızsa özdeğer yedeği Jacobi ile (aynı sigma kümesi); tekil yenilik kovaryansında güncelleme
  atlanır (Python hata fırlatırdı).
- Parametre hataları (Python FusionError) `init()` / `reset()` dönüşüyle bildirilir.
{"- 2B konum: başlatma geçmişi sabit boyutlu (`SPT_POS_FIX_HISTORY`); enlem / boylam girişi double." if r.scenario == "position2d" else ""}

## İş parçacığı güvenliği

Fonksiyonlar yeniden girişlidir ve global durum tutmaz. Aynı durum yapısını iki bağlamdan (kesme + görev) kullanacaksanız
erişimi kendiniz koruyun ya da sonucu çift tampon / kuyrukla aktarın.
"""
    return text.replace("spt_", f"{p}_").replace("SPT_", f"{P}_") if p != "spt" else text
