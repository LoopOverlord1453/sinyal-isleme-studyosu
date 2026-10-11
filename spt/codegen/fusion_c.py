"""Füzyon algoritmaları için C99 kod üreteci, ayak izi (flash / RAM / yığın) ve hesap maliyeti.

Üretilen C, `spt.dsp.fusion` içindeki Python algoritmalarının birebir karşılığıdır: aynı denklemler, aynı parametre
anlamları ve varsayılanları, ilk geçerli ivme (+ manyetometre) örneğinden `initial_quaternion` ile başlatma, aynı
doğrusal ivme gürültü büyütmesi (`_acc_r`), aynı yalnız-başlık manyetometre güncellemesi, aynı geçersiz örnek
kuralları (sonlu olmayan / sıfır ivme ya da manyetometre o güncellemeyi atlar; sonlu olmayan jiroskop durumu unutur).

Python'dan bilinçli farklar (hepsi yuvarlama düzeyinde; çift hassasiyette doğrulanır, bkz. testler):
  - Bölmeler mümkün olduğunca tersle çarpmaya çevrildi; simetrik matrislerin yalnız üst üçgeni hesaplanıp aynalandı.
  - Başlık için sin / cos(roll), sin / cos(pitch) özdeşliklerle (trig çağrısı yok), iki açının farkı tek atan2 ile
    (karmaşık bölme) hesaplanır; EKF Joseph biçimi kapalı formda (P − K·uᵀ − u·Kᵀ + s·K·Kᵀ) açılmıştır.
  - Parçacık filtresinin rastgele sayı üreteci farklıdır (xorshift32 + Marsaglia kutupsal); sonuç istatistiksel
    olarak eşdeğerdir, örnek örnek aynı değildir. Yeniden örneklemede parçacık sırası farklıdır (yerinde kopyalama).
  - UKF'de Cholesky başarısızsa Python `eigh` yedeği Jacobi özdeğer ayrışımıyla yapılır (aynı sigma kümesi).
  - 3×3 / 1×1 yenilik kovaryansı tekilse güncelleme atlanır (Python burada LinAlgError fırlatırdı).
  - Başlatmada ivme / manyetometre geçerliliği ‖v‖₂ > 0 ile denetlenir (Python |v|₁ > 0): karesi taşan ya da sıfıra
    inen bileşenler (float'ta ~1e-19'dan küçük, ~1.8e19'dan büyük) C'de geçersiz sayılır — normalleştirmede 0/0 ya da
    sonsuz üretmesin diye (MCU güvenliği). Gerçek sensör verisinde fark oluşmaz.
  - step() dt ≤ 0 / NaN / sonsuz örneği atlar (0 döner, durum değişmez); Python run() sabit dt alır.

Ayak izi (`footprint`): yerel derleyici varsa ÖLÇÜLÜR (nesne bölümleri, sizeof, libm dahil bağlanmış imaj farkı,
disassembly'den en kötü yığın); IAR, Xtensa ve RISC-V için ölçülmüş bir değerden TAHMİN edilir (yöntem satırda yazar).
Maliyet (`cost`): adım başına işlem sayıları ana makinede C++ sayaç türüyle ÖLÇÜLÜR, çevrimler MCU profilindeki
tablodan TAHMİN edilir.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from spt.codegen import toolchain as tc
from spt.codegen.mcu import (BASIC_OPS, COMPILERS, IAR_LIBM_FACTOR, IAR_SIZE_FACTOR, LIBRARY_OPS, MCU_PROFILES,
                             OPS, OPT_LEVELS, McuProfile, cycle_key, profile)
from spt.dsp import fusion as fu

COST_SAMPLES = 4000      # maliyet ölçümünde simüle edilen adım sayısı (fs = 200 Hz'de 20 s)
COST_SEED = 7
_TITLES = {"mahony": "Mahony", "madgwick": "Madgwick", "kalman": "Kalman (eksen başına)", "ekf": "EKF (çarpımsal)",
           "ukf": "UKF (kokusuz)", "pf": "Parçacık filtresi"}
_BIAS_GETTER = {"mahony": "gyro_bias", "madgwick": None, "kalman": "euler_bias", "ekf": "gyro_bias",
                "ukf": "gyro_bias", "pf": "gyro_bias"}


# ============================================================================ seçenekler ve sonuç türleri
@dataclass(frozen=True)
class CodegenOptions:
    algorithms: tuple[str, ...]
    params: dict[str, dict[str, float]] = field(default_factory=dict)  # eksik → fusion.py varsayılanı
    real: str = "float"                  # "float" | "double"
    mcu: str = "cortex-m4f"
    compiler: str = "gcc"
    opt: str = "-Os"
    clock_mhz: float = 168.0
    fs_hz: float = 200.0
    flash_kb: float = 1024.0
    ram_kb: float = 192.0
    use_mag: bool = True
    prefix: str = "spt"


@dataclass(frozen=True)
class GeneratedFile:
    name: str
    text: str


@dataclass(frozen=True)
class FootprintRow:
    algorithm: str
    compiler: str
    flash_code: int                  # algoritmanın kendi kodu + sabitleri (bayt)
    flash_with_libm: int | None      # libm / libgcc (yazılım kayan nokta) dahil bağlanmış imaj farkı (bayt)
    ram_state: int                   # sizeof(durum yapısı) (bayt)
    stack: int | None                # en kötü yol yığın kullanımı, libm dahil (bayt; muhafazakâr üst sınır)
    measured: bool
    method: str
    stack_own: int | None = None     # yalnız üretilen kodun en kötü yolu (libm fonksiyonları 0 baytlık yaprak)


@dataclass(frozen=True)
class CostRow:
    algorithm: str
    ops: dict[str, float]            # adım başına ortalama işlem sayıları
    cycles: float                    # adım başına ortalama çevrim (tahmin)
    us_per_update: float
    cpu_load_percent: float
    max_rate_hz: float
    method: str
    mag: bool = False                # True: manyetometreli (MARG) adımlar
    cycles_max: float = 0.0          # en pahalı adım (ör. parçacık yeniden örneklemesi)
    us_max: float = 0.0


@dataclass(frozen=True)
class HostRun:
    q: np.ndarray                    # (N, 4); çıktı yoksa NaN
    bias: np.ndarray                 # (3,) son durumdaki sapma (Madgwick: sıfır)
    ops: np.ndarray | None = None    # (N, len(OPS)) adım başına işlem sayıları


def _resolved_params(key: str, given: dict[str, float] | None) -> dict[str, float]:
    cls = fu.ALGORITHMS[key]
    given = dict(given or {})
    out = {}
    for p in cls.params:
        v = float(given.pop(p.key, p.default))
        if not math.isfinite(v):
            raise ValueError(f"{key}.{p.key}: sonlu bir sayı olmalı")
        if not p.lo <= v <= p.hi:  # Python ile aynı aralık (dışında ör. σ = 0 kalıcı NaN üretir)
            raise ValueError(f"{cls.name}: '{p.label}' = {v:g} geçersiz; izin verilen aralık [{p.lo:g}, {p.hi:g}]")
        out[p.key] = v
    if given:
        raise ValueError(f"{key}: bilinmeyen parametre(ler) {sorted(given)}")
    if key == "pf":
        out["particles"] = float(int(round(out["particles"])))
        out["seed"] = float(int(round(out["seed"])) & 0xFFFFFFFF)
    return out


def check_options(opts: CodegenOptions) -> None:
    if not opts.algorithms:
        raise ValueError("En az bir algoritma seçilmeli")
    for a in opts.algorithms:
        if a not in fu.ALGORITHMS:
            raise ValueError(f"Bilinmeyen algoritma: {a!r} (seçenekler: {', '.join(fu.ALGORITHMS)})")
    for a in opts.params:
        if a not in fu.ALGORITHMS:
            raise ValueError(f"Bilinmeyen algoritma parametresi: {a!r}")
        _resolved_params(a, opts.params[a])
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
    if opts.clock_mhz <= 0 or opts.fs_hz <= 0:
        raise ValueError("clock_mhz ve fs_hz pozitif olmalı")


def _ordered(opts: CodegenOptions) -> list[str]:
    """Seçilen algoritmalar fusion.ALGORITHMS sırasında, tekrarsız."""
    return [k for k in fu.ALGORITHMS if k in set(opts.algorithms)]


# ============================================================================ C şablonları
# Şablonlar 'spt_' / 'SPT_' önekiyle yazılır; _prefixed() kullanıcı önekine çevirir. {{AD}} yer tutuculardır.
_BANNER = """Signal Processing Toolbox tarafından üretildi; Python referansı: spt/dsp/fusion.py.
 * Ayrıntılı kullanım: KULLANIM.md"""

_COMMON_H = r"""/**
 * @file spt_fusion_common.h
 * @brief Yönelim füzyonu ortak tanımları: sayı türü, matematik makroları, kuaterniyon yardımcıları.
 *
 * {{BANNER}}
 *
 * Kurallar (tüm algoritmalar):
 *  - Gövde ve dünya çerçevesi sağ el kuralı, z YUKARI. Durağan ivmeölçer +g·z_gövde ölçer (özgül kuvvet).
 *  - q = [w, x, y, z] gövde → dünya döndürmesi (v_dünya = R(q)·v_gövde); kinematik q̇ = ½·q ⊗ [0, ω_gövde].
 *  - Euler açıları ZYX (aerospace): R = Rz(yaw)·Ry(pitch)·Rx(roll); yaw manyetik kuzeyden (dünya +x) ölçülür.
 *  - Birimler: jiroskop rad/s, dt saniye, açılar radyan. İvmeölçer ve manyetometre birimden bağımsızdır (yalnız
 *    yönleri kullanılır; m/s², g, µT, ham LSB olabilir).
 *
 * Yapılandırma (derleyici komut satırından ya da bu dosyadan önce tanımlanabilir):
 *  - SPT_USE_DOUBLE : 1 → double, 0 → float (üretimde seçilen: {{REAL}}).
 *  - SPT_USE_MAG    : 0 → manyetometre kodu derlenmez (flash tasarrufu; mag parametresi yok sayılır).
 *  - SPT_SQRT, SPT_SIN, SPT_COS, SPT_ATAN2, SPT_ASIN, SPT_EXP, SPT_LOG, SPT_FABS, SPT_FMOD: matematik fonksiyonları
 *    (ör. CMSIS-DSP arm_sqrt_f32 ya da hızlı sinüs ile değiştirilebilir; doğruluk sorumluluğu kullanıcıdadır).
 *
 * Uyarı: -ffast-math / -ffinite-math-only ile derlemeyin; geçersiz örnek (NaN / sonsuz) denetimleri kaldırılır.
 */
#ifndef SPT_FUSION_COMMON_H
#define SPT_FUSION_COMMON_H

#include <stddef.h>
#include <stdint.h>

#ifndef SPT_USE_DOUBLE
#define SPT_USE_DOUBLE {{USE_DOUBLE}}
#endif
#ifndef SPT_USE_MAG
#define SPT_USE_MAG {{USE_MAG}}
#endif

#ifndef SPT_FUSION_SHARED_DEFS
/* ---- yönelim ve genel füzyon başlıklarının ortak tanımları: hangisi önce eklenirse oradan gelir -------------- */
#define SPT_FUSION_SHARED_DEFS
#if defined(SPT_CUSTOM_REAL)
/* SPT_REAL türü, SPT_C, SPT_REAL_MAX, SPT_EPS, SPT_FROM_INT, SPT_U32_TO_REAL ve tüm SPT_xxx matematik makroları
   bu dosyadan önce tanımlanır (ör. ana makinedeki işlem sayacı testi). */
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

/** Üç bileşenin hepsi sonlu mu (jiroskop örneği denetimi). */
static inline int spt_finite3(const SPT_REAL v[3])
{
    return spt_finite(v[0]) && spt_finite(v[1]) && spt_finite(v[2]);
}

/**
 * Ölçüm kullanılabilir mi (Python _valid): bileşenler sonlu ve norm > 0.
 * @return Geçerliyse norm, değilse 0.
 */
static inline SPT_REAL spt_norm3_valid(const SPT_REAL v[3])
{
    SPT_REAL n = SPT_SQRT(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    return (n > SPT_C(0.0) && n <= SPT_REAL_MAX) ? n : SPT_C(0.0);
}

/** c = a ⊗ b (Hamilton çarpımı, [w, x, y, z]). c, a ya da b ile aynı dizi olmamalı. */
static inline void spt_quat_mul(const SPT_REAL a[4], const SPT_REAL b[4], SPT_REAL c[4])
{
    c[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
    c[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
    c[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
    c[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
}

/** out = in / |in| (out, in ile aynı olabilir). */
static inline void spt_quat_normalize(const SPT_REAL in[4], SPT_REAL out[4])
{
    SPT_REAL n = SPT_C(1.0) / SPT_SQRT(in[0] * in[0] + in[1] * in[1] + in[2] * in[2] + in[3] * in[3]);
    out[0] = in[0] * n;
    out[1] = in[1] * n;
    out[2] = in[2] * n;
    out[3] = in[3] * n;
}

/** e = exp(v): dönme vektörü (rad, eksen·açı) → birim kuaterniyon; k = sin(θ/2)/θ (θ → 0'da ½). */
static inline void spt_quat_exp(SPT_REAL vx, SPT_REAL vy, SPT_REAL vz, SPT_REAL e[4])
{
    SPT_REAL angle = SPT_SQRT(vx * vx + vy * vy + vz * vz);
    SPT_REAL k;
    if (angle == SPT_C(0.0)) {
        e[0] = SPT_C(1.0);
        e[1] = SPT_C(0.0);
        e[2] = SPT_C(0.0);
        e[3] = SPT_C(0.0);
        return;
    }
    e[0] = SPT_COS(SPT_C(0.5) * angle);
    k = (angle > SPT_C(1e-12)) ? SPT_SIN(SPT_C(0.5) * angle) / angle : SPT_C(0.5);
    e[1] = vx * k;
    e[2] = vy * k;
    e[3] = vz * k;
}

/** q ← normalize(q ⊗ exp([vx, vy, vz])) (Python _qmul_exp). */
static inline void spt_qmul_exp(SPT_REAL q[4], SPT_REAL vx, SPT_REAL vy, SPT_REAL vz)
{
    SPT_REAL e[4], t[4];
    spt_quat_exp(vx, vy, vz, e);
    spt_quat_mul(q, e, t);
    spt_quat_normalize(t, q);
}

/** Birim kuaterniyon → dönme vektörü (kısa yol, Python quat_log). */
static inline void spt_quat_log(const SPT_REAL q[4], SPT_REAL v[3])
{
    SPT_REAL w = q[0], x = q[1], y = q[2], z = q[3];
    SPT_REAL s, k;
    if (w < SPT_C(0.0)) {
        w = -w;
        x = -x;
        y = -y;
        z = -z;
    }
    s = SPT_SQRT(x * x + y * y + z * z);
    k = (s > SPT_C(1e-12)) ? SPT_C(2.0) * SPT_ATAN2(s, w) / s : SPT_C(2.0);
    v[0] = x * k;
    v[1] = y * k;
    v[2] = z * k;
}

/** Tahmini yerçekimi yönü gövdede: R(q)ᵀ·ẑ = R(q)'nun 3. satırı. */
static inline void spt_gravity_body(const SPT_REAL q[4], SPT_REAL v[3])
{
    v[0] = SPT_C(2.0) * (q[1] * q[3] - q[0] * q[2]);
    v[1] = SPT_C(2.0) * (q[2] * q[3] + q[0] * q[1]);
    v[2] = SPT_C(1.0) - SPT_C(2.0) * (q[1] * q[1] + q[2] * q[2]);
}

/** ZYX Euler → kuaterniyon (rad). */
static inline void spt_quat_from_euler(SPT_REAL roll, SPT_REAL pitch, SPT_REAL yaw, SPT_REAL q[4])
{
    SPT_REAL cr = SPT_COS(SPT_C(0.5) * roll), sr = SPT_SIN(SPT_C(0.5) * roll);
    SPT_REAL cp = SPT_COS(SPT_C(0.5) * pitch), sp = SPT_SIN(SPT_C(0.5) * pitch);
    SPT_REAL cy = SPT_COS(SPT_C(0.5) * yaw), sy = SPT_SIN(SPT_C(0.5) * yaw);
    q[0] = cr * cp * cy + sr * sp * sy;
    q[1] = sr * cp * cy - cr * sp * sy;
    q[2] = cr * sp * cy + sr * cp * sy;
    q[3] = cr * cp * sy - sr * sp * cy;
}

/** Kuaterniyon → ZYX Euler (rad): roll, yaw ∈ [−π, π], pitch ∈ [−π/2, π/2]. */
static inline void spt_euler_from_quat(const SPT_REAL q[4], SPT_REAL *roll, SPT_REAL *pitch, SPT_REAL *yaw)
{
    SPT_REAL sp = SPT_C(2.0) * (q[0] * q[2] - q[3] * q[1]);
    if (sp > SPT_C(1.0)) {
        sp = SPT_C(1.0);
    } else if (sp < SPT_C(-1.0)) {
        sp = SPT_C(-1.0);
    }
    *roll = SPT_ATAN2(SPT_C(2.0) * (q[0] * q[1] + q[2] * q[3]), SPT_C(1.0) - SPT_C(2.0) * (q[1] * q[1] + q[2] * q[2]));
    *pitch = SPT_ASIN(sp);
    *yaw = SPT_ATAN2(SPT_C(2.0) * (q[0] * q[3] + q[1] * q[2]), SPT_C(1.0) - SPT_C(2.0) * (q[2] * q[2] + q[3] * q[3]));
}

/**
 * İlk yönelim (Python initial_quaternion + normalize): ivmeölçerden roll / pitch, geçerli manyetometre varsa eğim
 * düzeltmeli yaw, yoksa yaw = 0. acc geçerli olmalı (çağıran denetler). İşlem sırası Python ile birebirdir (hypot,
 * numpy einsum toplama sırası, bölme): ilk adımda gradyan ≈ 0 olduğundan Madgwick'in normalize adımının yönü bu
 * yuvarlama artıklarına bağlıdır.
 */
static inline void spt_initial_quat(const SPT_REAL acc[3], const SPT_REAL mag[3], SPT_REAL q[4])
{
    SPT_REAL roll = SPT_ATAN2(acc[1], acc[2]);
    SPT_REAL pitch = SPT_ATAN2(-acc[0], SPT_HYPOT(acc[1], acc[2]));
    SPT_REAL yaw = SPT_C(0.0), n;
#if SPT_USE_MAG
    if (mag != NULL && spt_norm3_valid(mag) > SPT_C(0.0)) {
        SPT_REAL cr = SPT_COS(roll), sr = SPT_SIN(roll), cp = SPT_COS(pitch), sp = SPT_SIN(pitch);
        yaw = SPT_ATAN2(sr * mag[2] - cr * mag[1], cp * mag[0] + sp * sr * mag[1] + sp * cr * mag[2]);
    }
#else
    (void)mag;
#endif
    spt_quat_from_euler(roll, pitch, yaw, q);
    n = SPT_SQRT((q[0] * q[0] + q[2] * q[2]) + (q[1] * q[1] + q[3] * q[3]));
    q[0] = q[0] / n;
    q[1] = q[1] / n;
    q[2] = q[2] / n;
    q[3] = q[3] / n;
}

/** Açıyı [−π, π) aralığına sarar (Python (a + π) % 2π − π); yaygın durumda yalnız karşılaştırma yapılır. */
static inline SPT_REAL spt_wrap_pi(SPT_REAL a)
{
    if (a >= SPT_NEG_PI && a < SPT_PI) {
        return a;
    }
    if (a >= SPT_PI && a < SPT_THREE_PI) {
        return a - SPT_TWO_PI;
    }
    if (a < SPT_NEG_PI && a >= SPT_NEG_THREE_PI) {
        return a + SPT_TWO_PI;
    }
    a = SPT_FMOD(a + SPT_PI, SPT_TWO_PI);
    if (a < SPT_C(0.0)) {
        a += SPT_TWO_PI;
    }
    return a - SPT_PI;
}

/** wrap(atan2(n1, d1) − atan2(n2, d2)) tek atan2 ile: arg((d1 + i·n1)·(d2 − i·n2)). */
static inline SPT_REAL spt_angle_diff(SPT_REAL n1, SPT_REAL d1, SPT_REAL n2, SPT_REAL d2)
{
    if ((n1 == SPT_C(0.0) && d1 == SPT_C(0.0)) || (n2 == SPT_C(0.0) && d2 == SPT_C(0.0))) {
        return spt_wrap_pi(SPT_ATAN2(n1, d1) - SPT_ATAN2(n2, d2));
    }
    return SPT_ATAN2(n1 * d2 - d1 * n2, d1 * d2 + n1 * n2);
}

/**
 * Eğim düzeltmeli manyetik başlık, q'nun roll / pitch'iyle (Python tilt_compensated_heading(mag, roll(q),
 * pitch(q))): başlık = atan2(*num, *den). sin / cos özdeşliklerle (trig çağrısı yok): roll = atan2(A, B) →
 * sin = A/√(A²+B²), cos = B/√(A²+B²); pitch = asin(s) → sin = s, cos = √(1 − s²).
 */
static inline void spt_heading_vec(const SPT_REAL q[4], const SPT_REAL mag[3], SPT_REAL *num, SPT_REAL *den)
{
    SPT_REAL a = SPT_C(2.0) * (q[0] * q[1] + q[2] * q[3]);
    SPT_REAL b = SPT_C(1.0) - SPT_C(2.0) * (q[1] * q[1] + q[2] * q[2]);
    SPT_REAL sp = SPT_C(2.0) * (q[0] * q[2] - q[3] * q[1]);
    SPT_REAL r = SPT_SQRT(a * a + b * b);
    SPT_REAL sr = SPT_C(0.0), cr = SPT_C(1.0), cp;
    if (r > SPT_C(0.0)) {
        SPT_REAL ir = SPT_C(1.0) / r;
        sr = a * ir;
        cr = b * ir;
    }
    if (sp > SPT_C(1.0)) {
        sp = SPT_C(1.0);
    } else if (sp < SPT_C(-1.0)) {
        sp = SPT_C(-1.0);
    }
    cp = SPT_SQRT(SPT_C(1.0) - sp * sp);
    *num = sr * mag[2] - cr * mag[1];
    *den = cp * mag[0] + sp * sr * mag[1] + sp * cr * mag[2];
}

/** 3×3 ters (kofaktör / determinant). Tekilse 0 döner (çağıran güncellemeyi atlar). */
static inline int spt_inv3(SPT_REAL a[3][3], SPT_REAL inv[3][3])
{
    SPT_REAL c00 = a[1][1] * a[2][2] - a[1][2] * a[2][1];
    SPT_REAL c01 = a[1][2] * a[2][0] - a[1][0] * a[2][2];
    SPT_REAL c02 = a[1][0] * a[2][1] - a[1][1] * a[2][0];
    SPT_REAL det = a[0][0] * c00 + a[0][1] * c01 + a[0][2] * c02;
    SPT_REAL id;
    if (!(det != SPT_C(0.0) && spt_finite(det))) {
        return 0;
    }
    id = SPT_C(1.0) / det;
    inv[0][0] = c00 * id;
    inv[0][1] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * id;
    inv[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * id;
    inv[1][0] = c01 * id;
    inv[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * id;
    inv[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * id;
    inv[2][0] = c02 * id;
    inv[2][1] = (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * id;
    inv[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * id;
    return 1;
}

/** İvmeölçer ölçüm varyansı: |a| yerçekimi kestiriminden saptıkça büyür (Python _acc_r). */
static inline SPT_REAL spt_acc_r(SPT_REAL *g_ref, int *g_ref_valid, SPT_REAL n, SPT_REAL sigma_acc)
{
    SPT_REAL dev = SPT_C(0.0);
    if (!*g_ref_valid) {
        *g_ref = n;
        *g_ref_valid = 1;
    } else {
        *g_ref += SPT_C(0.001) * (n - *g_ref);
    }
    if (*g_ref > SPT_C(0.0)) {
        dev = SPT_FABS(n / *g_ref - SPT_C(1.0)) * SPT_C(50.0);   /* (sapma / 0.02) */
    }
    return sigma_acc * sigma_acc * (SPT_C(1.0) + dev * dev);
}

/** EKF / UKF süreç gürültüsü (Python _process_cov): Q_θθ = qa·I, Q_θb = qc·I, Q_bb = qd·I. */
static inline void spt_process_noise(SPT_REAL sigma_gyro, SPT_REAL sigma_bias, SPT_REAL dt, SPT_REAL *qa,
                                     SPT_REAL *qc, SPT_REAL *qd)
{
    SPT_REAL sg = sigma_gyro * SPT_D2R, sb = sigma_bias * SPT_D2R;
    sg = sg * sg;
    sb = sb * sb;
    *qa = sg * dt + sb * dt * dt * dt / SPT_C(3.0);
    *qc = SPT_C(-0.5) * sb * dt * dt;
    *qd = sb * dt;
}

#endif /* SPT_FUSION_COMMON_H */
"""

# Her algoritmada aynı olan başlık bildirimleri ({{A}} = algoritma adı, {{AU}} = büyük harf)
_API_DECLS = r"""
/** Parametreleri derleme anı varsayılanlarına getirir, durumu "başlatılmamış" yapar (step() ilk geçerli örnekte
 *  kendisi başlatır). */
void spt_{{A}}_defaults(spt_{{A}}_t *s);

/**
 * Varsayılan parametreler + ilk örnekten başlatma (= defaults + reset).
 * @param acc ivmeölçer (herhangi bir birim; sonlu ve sıfırdan farklı olmalı)
 * @param mag manyetometre ya da NULL (geçersizse yok sayılır → yaw = 0)
 * @return 1: başlatıldı, 0: acc geçersiz (durum başlatılmadı)
 */
int spt_{{A}}_init(spt_{{A}}_t *s, const SPT_REAL acc[3], const SPT_REAL mag[3]);

/** Durumu ilk örnekten yeniden başlatır, parametreleri korur (Python Estimator.reset(initial_quaternion(…))). */
int spt_{{A}}_reset(spt_{{A}}_t *s, const SPT_REAL acc[3], const SPT_REAL mag[3]);

/**
 * Bir örnek işler (durum başlatılmış olmalı).
 * @param gyro jiroskop (rad/s, gövde çerçevesi; sonlu olmalı — denetimli akış için step())
 * @param acc  ivmeölçer; sonlu değilse ya da sıfırsa ivme güncellemesi atlanır
 * @param mag  manyetometre ya da NULL; geçersizse başlık güncellemesi atlanır
 * @param dt   örnekleme aralığı (s)
 */
void spt_{{A}}_update(spt_{{A}}_t *s, const SPT_REAL gyro[3], const SPT_REAL acc[3], const SPT_REAL mag[3],
                      SPT_REAL dt);

/**
 * Denetimli akış (Python Estimator.run ile aynı kurallar): jiroskop sonlu değilse durum unutulur ve 0 döner; durum
 * başlatılmamışsa ilk geçerli ivme (+ geçerli manyetometre) örneğinden başlatılır, ardından update() çalışır.
 * dt > 0 ve sonlu olmalı; değilse (ör. aynı zaman damgası, sayaç taşması) örnek atlanır, durum değişmez, 0 döner.
 * @return 1: bu örnek için geçerli yönelim var, 0: yok (Python'da NaN)
 */
int spt_{{A}}_step(spt_{{A}}_t *s, const SPT_REAL gyro[3], const SPT_REAL acc[3], const SPT_REAL mag[3],
                   SPT_REAL dt);

/** Yönelim kuaterniyonu [w, x, y, z] (gövde → dünya). */
void spt_{{A}}_quat(const spt_{{A}}_t *s, SPT_REAL q[4]);

/** ZYX Euler açıları (rad): roll, yaw ∈ [−π, π], pitch ∈ [−π/2, π/2]. İşaretçiler NULL olmamalı. */
void spt_{{A}}_euler(const spt_{{A}}_t *s, SPT_REAL *roll, SPT_REAL *pitch, SPT_REAL *yaw);
"""

_API_IMPL = r"""
int spt_{{A}}_init(spt_{{A}}_t *s, const SPT_REAL acc[3], const SPT_REAL mag[3])
{
    spt_{{A}}_defaults(s);
    return spt_{{A}}_reset(s, acc, mag);
}

int spt_{{A}}_step(spt_{{A}}_t *s, const SPT_REAL gyro[3], const SPT_REAL acc[3], const SPT_REAL mag[3],
                   SPT_REAL dt)
{
    if (!(dt > SPT_C(0.0)) || !spt_finite(dt)) {
        return 0;   /* geçersiz dt (≤ 0, NaN, sonsuz): örnek atlanır, durum değişmez */
    }
    if (!spt_finite3(gyro)) {
        s->ready = 0;
        return 0;
    }
    if (!s->ready && !spt_{{A}}_reset(s, acc, mag)) {
        return 0;
    }
    spt_{{A}}_update(s, gyro, acc, mag, dt);
    return 1;
}
"""

_QUAT_EULER_IMPL = r"""
void spt_{{A}}_quat(const spt_{{A}}_t *s, SPT_REAL q[4])
{
    q[0] = s->{{Q}}[0];
    q[1] = s->{{Q}}[1];
    q[2] = s->{{Q}}[2];
    q[3] = s->{{Q}}[3];
}

void spt_{{A}}_euler(const spt_{{A}}_t *s, SPT_REAL *roll, SPT_REAL *pitch, SPT_REAL *yaw)
{
    spt_euler_from_quat(s->{{Q}}, roll, pitch, yaw);
}
"""

_HEADER = r"""/**
 * @file spt_{{A}}.h
 * @brief {{BRIEF}}
 *
 * {{BANNER}}
 * Python karşılığı: spt.dsp.fusion.{{PYCLASS}}.
 *
 * Kullanım: spt_{{A}}_t s; spt_{{A}}_defaults(&s); her örnekte spt_{{A}}_step(&s, gyro, acc, mag, dt);
 * ardından spt_{{A}}_quat() / spt_{{A}}_euler(). Durum yapısı çağıranındır: malloc ve global yok, yeniden
 * girişlidir (aynı yapıya aynı anda iki bağlamdan erişmeyin).
 */
#ifndef SPT_{{AU}}_H
#define SPT_{{AU}}_H

#include "spt_fusion_common.h"

/* ---- Varsayılan parametreler (derleme anı; #define ile ezilebilir, çalışma anında yapı alanlarından değişir) */
{{PARAM_DEFINES}}

SPT_BEGIN_DECLS

{{STRUCT}}
{{API}}

{{EXTRA_DECLS}}
SPT_END_DECLS

#endif /* SPT_{{AU}}_H */
"""

_STRUCTS = {
    "mahony": r"""/** Mahony durumu (Python Mahony: q, _integral, kp, ki). */
typedef struct spt_mahony {
    SPT_REAL q[4];          /**< yönelim [w, x, y, z], gövde → dünya */
    SPT_REAL integral[3];   /**< PI integral terimi (rad/s); jiroskop sapması kestirimi = −integral */
    SPT_REAL kp;            /**< oransal kazanç (1/s) — çalışma anında değiştirilebilir */
    SPT_REAL ki;            /**< integral kazancı (1/s²); 0 → sapma öğrenilmez */
    int ready;              /**< 1: durum başlatıldı */
} spt_mahony_t;
""",
    "madgwick": r"""/** Madgwick durumu (Python Madgwick: q, beta). */
typedef struct spt_madgwick {
    SPT_REAL q[4];          /**< yönelim [w, x, y, z], gövde → dünya */
    SPT_REAL beta;          /**< gradyan adımı (rad/s) — çalışma anında değiştirilebilir */
    int ready;              /**< 1: durum başlatıldı */
} spt_madgwick_t;
""",
    "kalman": r"""/** Eksen başına Kalman durumu (Python EulerKalman: açı + Euler hızı sapması, 2×2 kovaryans × 3). */
typedef struct spt_kalman {
    SPT_REAL angle[3];      /**< roll, pitch, yaw (rad) */
    SPT_REAL bias[3];       /**< Euler HIZI sapması (rad/s) — gövde jiroskop sapması değildir */
    SPT_REAL P[3][4];       /**< eksen başına kovaryans [p00, p01, p10, p11] */
    SPT_REAL sigma_angle;   /**< açı süreç gürültüsü (°/√s) */
    SPT_REAL sigma_bias;    /**< sapma rastgele yürüyüşü (°/s/√s) */
    SPT_REAL sigma_meas;    /**< ivmeölçer / manyetometre açısı gürültüsü (°) */
    int ready;              /**< 1: durum başlatıldı */
} spt_kalman_t;
""",
    "ekf": r"""/** Çarpımsal EKF (MEKF) durumu: hata durumu [δθ (3), δb (3)], nominal q ve b. */
typedef struct spt_ekf {
    SPT_REAL q[4];          /**< nominal yönelim [w, x, y, z] */
    SPT_REAL b[3];          /**< jiroskop sapması kestirimi (rad/s) */
    SPT_REAL P[6][6];       /**< hata kovaryansı (rad², rad²/s, rad²/s²) — her zaman tam simetrik */
    SPT_REAL g_ref;         /**< yerçekimi büyüklüğünün yavaş kestirimi (ivmeölçer biriminde) */
    SPT_REAL sigma_gyro;    /**< jiroskop beyaz gürültüsü (°/s/√Hz) */
    SPT_REAL sigma_bias;    /**< sapma rastgele yürüyüşü (°/s/√s) */
    SPT_REAL sigma_acc;     /**< normalize ivme yönü gürültüsü (örnek başına) */
    SPT_REAL sigma_heading; /**< manyetik başlık gürültüsü (°, örnek başına) */
    int g_ref_valid;        /**< g_ref başlatıldı mı */
    int ready;              /**< 1: durum başlatıldı */
} spt_ekf_t;
""",
    "ukf": r"""/** UKF (USQUE) durumu: hata durumu [δθ, δb], 13 sigma noktası; çalışma alanı da yapıdadır (yığın küçük kalır). */
typedef struct spt_ukf {
    SPT_REAL q[4];          /**< nominal yönelim [w, x, y, z] */
    SPT_REAL b[3];          /**< jiroskop sapması kestirimi (rad/s) */
    SPT_REAL P[6][6];       /**< hata kovaryansı — her zaman tam simetrik */
    SPT_REAL g_ref;         /**< yerçekimi büyüklüğünün yavaş kestirimi */
    SPT_REAL sigma_gyro;    /**< jiroskop beyaz gürültüsü (°/s/√Hz) */
    SPT_REAL sigma_bias;    /**< sapma rastgele yürüyüşü (°/s/√s) */
    SPT_REAL sigma_acc;     /**< normalize ivme yönü gürültüsü (örnek başına) */
    SPT_REAL sigma_heading; /**< manyetik başlık gürültüsü (°, örnek başına) */
    /* ---- çalışma alanı: adım içinde geçici (durum değil) */
    SPT_REAL L[6][6];       /**< √(6·P): sütunları sigma yönleri */
    SPT_REAL A[6][6];       /**< özdeğer yedeği (Jacobi) çalışma matrisi */
    SPT_REAL qs[13][4];     /**< sigma kuaterniyonları */
    SPT_REAL bs[13][3];     /**< sigma sapmaları */
    SPT_REAL dz[13][3];     /**< dönme farkları / ölçüm sigmaları */
    uint32_t fallbacks;     /**< Cholesky başarısız → özdeğer yedeği sayacı (tanılama) */
    int g_ref_valid;        /**< g_ref başlatıldı mı */
    int ready;              /**< 1: durum başlatıldı */
} spt_ukf_t;
""",
    "pf": r"""/** Parçacık filtresi durumu: SPT_PF_PARTICLES adet kuaterniyon + sapma parçacığı (yapı büyüktür: statik ayırın). */
typedef struct spt_pf {
    SPT_REAL q[SPT_PF_PARTICLES][4];  /**< parçacık yönelimleri */
    SPT_REAL b[SPT_PF_PARTICLES][3];  /**< parçacık jiroskop sapmaları (rad/s) */
    SPT_REAL logw[SPT_PF_PARTICLES];  /**< log ağırlıklar */
    SPT_REAL q_out[4];                /**< son ağırlıklı ortalama yönelim (çıktı) */
    SPT_REAL g_ref;                   /**< yerçekimi büyüklüğünün yavaş kestirimi */
    SPT_REAL ess;                     /**< etkin parçacık sayısı (son adım) */
    SPT_REAL spare;                   /**< kutupsal yöntemin sakladığı ikinci normal sayı */
    SPT_REAL sigma_gyro;              /**< öngörü gürültüsü (°/s/√Hz) */
    SPT_REAL sigma_bias;              /**< sapma parçacıklarının yürüyüşü (°/s/√s) */
    SPT_REAL sigma_acc;               /**< ivmeölçer olabilirlik genişliği (normalize) */
    SPT_REAL sigma_heading;           /**< başlık olabilirlik genişliği (°) */
    uint32_t rng;                     /**< xorshift32 durumu */
    uint32_t seed;                    /**< tohum (reset ile etkinleşir) */
    uint32_t resamples;               /**< yeniden örnekleme sayacı (tanılama) */
    int g_ref_valid;                  /**< g_ref başlatıldı mı */
    int has_spare;                    /**< spare geçerli mi */
    int ready;                        /**< 1: durum başlatıldı */
    uint16_t cnt[SPT_PF_PARTICLES];   /**< yeniden örneklemede kopya sayıları (geçici) */
} spt_pf_t;
""",
}

_EXTRA_DECLS = {
    "mahony": r"""/** Jiroskop sapması kestirimi (rad/s, gövde) = −integral (Python Mahony.bias). */
void spt_mahony_gyro_bias(const spt_mahony_t *s, SPT_REAL bias[3]);
""",
    "madgwick": "",
    "kalman": r"""/**
 * Euler HIZI sapması kestirimi (rad/s) — Python EulerKalman.bias (bias_is_euler = True). Gövde jiroskop sapması
 * DEĞİLDİR: eksen başına model, b_euler ≈ T(roll, pitch)·b_jiro'nun zaman ortalamasını kestirir (yatay çevresinde
 * jiroskop sapmasına yakın, büyük eğimde değil).
 */
void spt_kalman_euler_bias(const spt_kalman_t *s, SPT_REAL bias[3]);
""",
    "ekf": r"""/** Jiroskop sapması kestirimi (rad/s, gövde). */
void spt_ekf_gyro_bias(const spt_ekf_t *s, SPT_REAL bias[3]);
""",
    "ukf": r"""/** Jiroskop sapması kestirimi (rad/s, gövde). */
void spt_ukf_gyro_bias(const spt_ukf_t *s, SPT_REAL bias[3]);
""",
    "pf": r"""/** Jiroskop sapması kestirimi (rad/s): parçacık sapmalarının ağırlıklı ortalaması (O(N), N adet exp). */
void spt_pf_gyro_bias(const spt_pf_t *s, SPT_REAL bias[3]);
""",
}

_BRIEFS = {
    "mahony": "Mahony tamamlayıcı filtresi (SO(3) üzerinde PI düzeltmeli; x-io referans denklemleri).",
    "madgwick": "Madgwick gradyan iniş filtresi (makaledeki amaç fonksiyonu ve Jacobian'lar; IMU ve MARG).",
    "kalman": "Eksen başına Kalman (açı + Euler hızı sapması; Euler hızları tam kinematikle, yaw eğim düzeltmeli "
              "başlıkla).",
    "ekf": "Çarpımsal genişletilmiş Kalman filtresi (MEKF): hata durumu δθ (3) + jiroskop sapması (3); manyetometre "
           "yalnız başlığı düzeltir (kazanç h'ye izdüşümlü, Joseph biçimi).",
    "ukf": "Kokusuz Kalman filtresi (USQUE): aynı hata durumu, 13 sigma noktası (α = 1, β = 2, κ = 0); manyetometre "
           "yalnız başlığı düzeltir (Z = h·δθ).",
    "pf": "Parçacık filtresi: kuaterniyon + sapma parçacıkları, sistematik yeniden örnekleme ve pürüzlendirme; "
          "başlık olabilirliği dünya z etrafındaki twist açısıyla.",
}
_PYCLASS = {"mahony": "Mahony", "madgwick": "Madgwick", "kalman": "EulerKalman", "ekf": "MEKF", "ukf": "UKF",
            "pf": "ParticleFilter"}

_SOURCES = {
    "mahony": r"""
void spt_mahony_defaults(spt_mahony_t *s)
{
{{SET_DEFAULTS}}
    s->ready = 0;
}

int spt_mahony_reset(spt_mahony_t *s, const SPT_REAL acc[3], const SPT_REAL mag[3])
{
    if (!(spt_norm3_valid(acc) > SPT_C(0.0))) {
        s->ready = 0;
        return 0;
    }
    spt_initial_quat(acc, mag, s->q);
    s->integral[0] = SPT_C(0.0);
    s->integral[1] = SPT_C(0.0);
    s->integral[2] = SPT_C(0.0);
    s->ready = 1;
    return 1;
}

void spt_mahony_update(spt_mahony_t *s, const SPT_REAL gyro[3], const SPT_REAL acc[3], const SPT_REAL mag[3],
                       SPT_REAL dt)
{
    SPT_REAL w = s->q[0], x = s->q[1], y = s->q[2], z = s->q[3];
    SPT_REAL gx = gyro[0], gy = gyro[1], gz = gyro[2];
    SPT_REAL an = spt_norm3_valid(acc);
    SPT_REAL h, nw, nx, ny, nz, n;
#if !SPT_USE_MAG
    (void)mag;
#endif
    if (an > SPT_C(0.0)) {
        SPT_REAL ia = SPT_C(1.0) / an;
        SPT_REAL ax = acc[0] * ia, ay = acc[1] * ia, az = acc[2] * ia;
        /* tahmini yerçekimi yönü (gövde) ve ölçülenle çapraz çarpım hatası */
        SPT_REAL vx = SPT_C(2.0) * (x * z - w * y), vy = SPT_C(2.0) * (y * z + w * x);
        SPT_REAL vz = w * w - x * x - y * y + z * z;
        SPT_REAL ex = ay * vz - az * vy, ey = az * vx - ax * vz, ez = ax * vy - ay * vx;
#if SPT_USE_MAG
        SPT_REAL mn = (mag != NULL) ? spt_norm3_valid(mag) : SPT_C(0.0);
        if (mn > SPT_C(0.0)) {
            /* manyetik alanın dünyadaki referans yönü b = [√(hx²+hy²), 0, bz] ve gövdedeki tahmini */
            SPT_REAL im = SPT_C(1.0) / mn;
            SPT_REAL mx = mag[0] * im, my = mag[1] * im, mz = mag[2] * im;
            SPT_REAL hx = mx * (SPT_C(1.0) - SPT_C(2.0) * (y * y + z * z)) + my * SPT_C(2.0) * (x * y - w * z)
                          + mz * SPT_C(2.0) * (x * z + w * y);
            SPT_REAL hy = mx * SPT_C(2.0) * (x * y + w * z) + my * (SPT_C(1.0) - SPT_C(2.0) * (x * x + z * z))
                          + mz * SPT_C(2.0) * (y * z - w * x);
            SPT_REAL bz = mx * SPT_C(2.0) * (x * z - w * y) + my * SPT_C(2.0) * (y * z + w * x)
                          + mz * (SPT_C(1.0) - SPT_C(2.0) * (x * x + y * y));
            SPT_REAL bx = SPT_SQRT(hx * hx + hy * hy);
            SPT_REAL wx = bx * (SPT_C(1.0) - SPT_C(2.0) * (y * y + z * z)) + bz * SPT_C(2.0) * (x * z - w * y);
            SPT_REAL wy = bx * SPT_C(2.0) * (x * y - w * z) + bz * SPT_C(2.0) * (w * x + y * z);
            SPT_REAL wz = bx * SPT_C(2.0) * (x * z + w * y) + bz * (SPT_C(1.0) - SPT_C(2.0) * (x * x + y * y));
            ex += my * wz - mz * wy;
            ey += mz * wx - mx * wz;
            ez += mx * wy - my * wx;
        }
#endif
        if (s->ki > SPT_C(0.0)) {
            s->integral[0] += s->ki * ex * dt;
            s->integral[1] += s->ki * ey * dt;
            s->integral[2] += s->ki * ez * dt;
            gx += s->integral[0];
            gy += s->integral[1];
            gz += s->integral[2];
        }
        gx += s->kp * ex;
        gy += s->kp * ey;
        gz += s->kp * ez;
    }
    /* q̇ = ½·q ⊗ [0, ω], Euler adımı ve normalizasyon */
    h = SPT_C(0.5) * dt;
    nw = w - h * (x * gx + y * gy + z * gz);
    nx = x + h * (w * gx + y * gz - z * gy);
    ny = y + h * (w * gy - x * gz + z * gx);
    nz = z + h * (w * gz + x * gy - y * gx);
    n = SPT_C(1.0) / SPT_SQRT(nw * nw + nx * nx + ny * ny + nz * nz);
    s->q[0] = nw * n;
    s->q[1] = nx * n;
    s->q[2] = ny * n;
    s->q[3] = nz * n;
}

void spt_mahony_gyro_bias(const spt_mahony_t *s, SPT_REAL bias[3])
{
    bias[0] = -s->integral[0];
    bias[1] = -s->integral[1];
    bias[2] = -s->integral[2];
}
""",
    "madgwick": r"""
void spt_madgwick_defaults(spt_madgwick_t *s)
{
{{SET_DEFAULTS}}
    s->ready = 0;
}

int spt_madgwick_reset(spt_madgwick_t *s, const SPT_REAL acc[3], const SPT_REAL mag[3])
{
    if (!(spt_norm3_valid(acc) > SPT_C(0.0))) {
        s->ready = 0;
        return 0;
    }
    spt_initial_quat(acc, mag, s->q);
    s->ready = 1;
    return 1;
}

void spt_madgwick_update(spt_madgwick_t *s, const SPT_REAL gyro[3], const SPT_REAL acc[3], const SPT_REAL mag[3],
                         SPT_REAL dt)
{
    SPT_REAL w = s->q[0], x = s->q[1], y = s->q[2], z = s->q[3];
    SPT_REAL gx = gyro[0], gy = gyro[1], gz = gyro[2];
    /* jiroskop kinematiği: q̇ = ½·q ⊗ [0, ω] */
    SPT_REAL qw = SPT_C(0.5) * (-x * gx - y * gy - z * gz);
    SPT_REAL qx = SPT_C(0.5) * (w * gx + y * gz - z * gy);
    SPT_REAL qy = SPT_C(0.5) * (w * gy - x * gz + z * gx);
    SPT_REAL qz = SPT_C(0.5) * (w * gz + x * gy - y * gx);
    SPT_REAL an = spt_norm3_valid(acc);
    SPT_REAL n;
#if !SPT_USE_MAG
    (void)mag;
#endif
    if (an > SPT_C(0.0)) {
        /* bölme (tersle çarpma değil): başlatmadan hemen sonra f ≈ 0 ve normalize gradyanın yönü yuvarlama
           artığına bağlı — Python ile aynı işlem sırası aynı adımı verir */
        SPT_REAL ax = acc[0] / an, ay = acc[1] / an, az = acc[2] / an;
        /* amaç fonksiyonu f ve gradyanı Jᵀ·f (makaledeki kapalı biçim) */
        SPT_REAL f1 = SPT_C(2.0) * (x * z - w * y) - ax;
        SPT_REAL f2 = SPT_C(2.0) * (w * x + y * z) - ay;
        SPT_REAL f3 = SPT_C(2.0) * (SPT_C(0.5) - x * x - y * y) - az;
        SPT_REAL sw = SPT_C(-2.0) * y * f1 + SPT_C(2.0) * x * f2;
        SPT_REAL sx = SPT_C(2.0) * z * f1 + SPT_C(2.0) * w * f2 - SPT_C(4.0) * x * f3;
        SPT_REAL sy = SPT_C(-2.0) * w * f1 + SPT_C(2.0) * z * f2 - SPT_C(4.0) * y * f3;
        SPT_REAL sz = SPT_C(2.0) * x * f1 + SPT_C(2.0) * y * f2;
        SPT_REAL norm;
#if SPT_USE_MAG
        SPT_REAL mn = (mag != NULL) ? spt_norm3_valid(mag) : SPT_C(0.0);
        if (mn > SPT_C(0.0)) {
            SPT_REAL mx = mag[0] / mn, my = mag[1] / mn, mz = mag[2] / mn;
            SPT_REAL hx = mx * (SPT_C(1.0) - SPT_C(2.0) * (y * y + z * z)) + my * SPT_C(2.0) * (x * y - w * z)
                          + mz * SPT_C(2.0) * (x * z + w * y);
            SPT_REAL hy = mx * SPT_C(2.0) * (x * y + w * z) + my * (SPT_C(1.0) - SPT_C(2.0) * (x * x + z * z))
                          + mz * SPT_C(2.0) * (y * z - w * x);
            SPT_REAL bz = mx * SPT_C(2.0) * (x * z - w * y) + my * SPT_C(2.0) * (y * z + w * x)
                          + mz * (SPT_C(1.0) - SPT_C(2.0) * (x * x + y * y));
            SPT_REAL bx = SPT_SQRT(hx * hx + hy * hy);
            SPT_REAL f4 = SPT_C(2.0) * bx * (SPT_C(0.5) - y * y - z * z) + SPT_C(2.0) * bz * (x * z - w * y) - mx;
            SPT_REAL f5 = SPT_C(2.0) * bx * (x * y - w * z) + SPT_C(2.0) * bz * (w * x + y * z) - my;
            SPT_REAL f6 = SPT_C(2.0) * bx * (w * y + x * z) + SPT_C(2.0) * bz * (SPT_C(0.5) - x * x - y * y) - mz;
            sw += SPT_C(-2.0) * bz * y * f4 + (SPT_C(-2.0) * bx * z + SPT_C(2.0) * bz * x) * f5
                  + SPT_C(2.0) * bx * y * f6;
            sx += SPT_C(2.0) * bz * z * f4 + (SPT_C(2.0) * bx * y + SPT_C(2.0) * bz * w) * f5
                  + (SPT_C(2.0) * bx * z - SPT_C(4.0) * bz * x) * f6;
            sy += (SPT_C(-4.0) * bx * y - SPT_C(2.0) * bz * w) * f4 + (SPT_C(2.0) * bx * x + SPT_C(2.0) * bz * z) * f5
                  + (SPT_C(2.0) * bx * w - SPT_C(4.0) * bz * y) * f6;
            sz += (SPT_C(-4.0) * bx * z + SPT_C(2.0) * bz * x) * f4 + (SPT_C(-2.0) * bx * w + SPT_C(2.0) * bz * y) * f5
                  + SPT_C(2.0) * bx * x * f6;
        }
#endif
        norm = SPT_SQRT(sw * sw + sx * sx + sy * sy + sz * sz);
        if (norm > SPT_C(0.0)) {
            SPT_REAL k = s->beta / norm;
            qw -= k * sw;
            qx -= k * sx;
            qy -= k * sy;
            qz -= k * sz;
        }
    }
    w += qw * dt;
    x += qx * dt;
    y += qy * dt;
    z += qz * dt;
    n = SPT_C(1.0) / SPT_SQRT(w * w + x * x + y * y + z * z);
    s->q[0] = w * n;
    s->q[1] = x * n;
    s->q[2] = y * n;
    s->q[3] = z * n;
}
""",
    "kalman": r"""
void spt_kalman_defaults(spt_kalman_t *s)
{
{{SET_DEFAULTS}}
    s->ready = 0;
}

int spt_kalman_reset(spt_kalman_t *s, const SPT_REAL acc[3], const SPT_REAL mag[3])
{
    SPT_REAL q0[4];
    int i;
    if (!(spt_norm3_valid(acc) > SPT_C(0.0))) {
        s->ready = 0;
        return 0;
    }
    spt_initial_quat(acc, mag, q0);
    spt_euler_from_quat(q0, &s->angle[0], &s->angle[1], &s->angle[2]);
    for (i = 0; i < 3; ++i) {
        s->bias[i] = SPT_C(0.0);
        s->P[i][0] = SPT_C(0.01);
        s->P[i][1] = SPT_C(0.0);
        s->P[i][2] = SPT_C(0.0);
        s->P[i][3] = SPT_C(1e-4);
    }
    s->ready = 1;
    return 1;
}

void spt_kalman_update(spt_kalman_t *s, const SPT_REAL gyro[3], const SPT_REAL acc[3], const SPT_REAL mag[3],
                       SPT_REAL dt)
{
    SPT_REAL qa = s->sigma_angle * SPT_D2R, qb = s->sigma_bias * SPT_D2R, r = s->sigma_meas * SPT_D2R;
    SPT_REAL sr = SPT_SIN(s->angle[0]), cr = SPT_COS(s->angle[0]);
    SPT_REAL sp = SPT_SIN(s->angle[1]), cp0 = SPT_COS(s->angle[1]);
    SPT_REAL acp = SPT_FABS(cp0), cp, tp, rates[3], meas[3];
    int has[3] = {0, 0, 0};
    int i;
    qa = qa * qa;
    qb = qb * qb;
    r = r * r;
    if (acp < SPT_C(1e-6)) {
        acp = SPT_C(1e-6);   /* copysign(max(|cos|, 1e-6), cos): pitch ±90° tekilliği */
    }
    cp = (cp0 < SPT_C(0.0)) ? -acp : acp;
    tp = sp / cp;
    /* Euler hızları: tam kinematik (gövde jiroskobundan) */
    rates[0] = gyro[0] + sr * tp * gyro[1] + cr * tp * gyro[2];
    rates[1] = cr * gyro[1] - sr * gyro[2];
    rates[2] = (sr * gyro[1] + cr * gyro[2]) / cp;
    meas[0] = SPT_C(0.0);
    meas[1] = SPT_C(0.0);
    meas[2] = SPT_C(0.0);
    if (spt_norm3_valid(acc) > SPT_C(0.0)) {
        meas[0] = SPT_ATAN2(acc[1], acc[2]);
        meas[1] = SPT_ATAN2(-acc[0], SPT_SQRT(acc[1] * acc[1] + acc[2] * acc[2]));
        has[0] = 1;
        has[1] = 1;
    }
#if SPT_USE_MAG
    if (mag != NULL && spt_norm3_valid(mag) > SPT_C(0.0)) {
        /* eğim düzeltmeli başlık: önceki adımın roll / pitch'iyle (Python ile aynı) */
        meas[2] = SPT_ATAN2(sr * mag[2] - cr * mag[1], cp0 * mag[0] + sp * sr * mag[1] + sp * cr * mag[2]);
        has[2] = 1;
    }
#else
    (void)mag;
#endif
    for (i = 0; i < 3; ++i) {
        SPT_REAL *p = s->P[i];
        s->angle[i] += dt * (rates[i] - s->bias[i]);
        p[0] += dt * (dt * p[3] - p[1] - p[2] + qa);
        p[1] -= dt * p[3];
        p[2] -= dt * p[3];
        p[3] += qb * dt;
        if (has[i]) {
            SPT_REAL innov = spt_wrap_pi(meas[i] - s->angle[i]);
            SPT_REAL is = SPT_C(1.0) / (p[0] + r);
            SPT_REAL k0 = p[0] * is, k1 = p[2] * is;
            SPT_REAL p00 = p[0], p01 = p[1];
            s->angle[i] += k0 * innov;
            s->bias[i] += k1 * innov;
            p[0] -= k0 * p00;
            p[1] -= k0 * p01;
            p[2] -= k1 * p00;
            p[3] -= k1 * p01;
        }
        s->angle[i] = spt_wrap_pi(s->angle[i]);
    }
}

void spt_kalman_quat(const spt_kalman_t *s, SPT_REAL q[4])
{
    spt_quat_from_euler(s->angle[0], s->angle[1], s->angle[2], q);
}

void spt_kalman_euler(const spt_kalman_t *s, SPT_REAL *roll, SPT_REAL *pitch, SPT_REAL *yaw)
{
    *roll = s->angle[0];
    *pitch = s->angle[1];
    *yaw = s->angle[2];
}

void spt_kalman_euler_bias(const spt_kalman_t *s, SPT_REAL bias[3])
{
    bias[0] = s->bias[0];
    bias[1] = s->bias[1];
    bias[2] = s->bias[2];
}
""",
    "ekf": r"""
void spt_ekf_defaults(spt_ekf_t *s)
{
{{SET_DEFAULTS}}
    s->ready = 0;
}

int spt_ekf_reset(spt_ekf_t *s, const SPT_REAL acc[3], const SPT_REAL mag[3])
{
    int i, j;
    if (!(spt_norm3_valid(acc) > SPT_C(0.0))) {
        s->ready = 0;
        return 0;
    }
    spt_initial_quat(acc, mag, s->q);
    for (i = 0; i < 6; ++i) {
        for (j = 0; j < 6; ++j) {
            s->P[i][j] = SPT_C(0.0);
        }
    }
    for (i = 0; i < 3; ++i) {
        s->b[i] = SPT_C(0.0);
        s->P[i][i] = SPT_C(0.0076154354946677734);          /* (5°)² rad² */
        s->P[i + 3][i + 3] = SPT_C(7.6154354946677734e-05); /* (0.5°/s)² */
    }
    s->g_ref = SPT_C(0.0);
    s->g_ref_valid = 0;
    s->ready = 1;
    return 1;
}

/** (T·Φᵀ)[i][j], j < 3: Φ'nin j. satırı = (I − [w×])_j | −dt·e_j */
static SPT_REAL spt_ekf_phi_t(const SPT_REAL t[6], int j, SPT_REAL wx, SPT_REAL wy, SPT_REAL wz, SPT_REAL dt)
{
    if (j == 0) {
        return t[0] + wz * t[1] - wy * t[2] - dt * t[3];
    }
    if (j == 1) {
        return t[1] - wz * t[0] + wx * t[2] - dt * t[4];
    }
    return t[2] + wy * t[0] - wx * t[1] - dt * t[5];
}

void spt_ekf_update(spt_ekf_t *s, const SPT_REAL gyro[3], const SPT_REAL acc[3], const SPT_REAL mag[3],
                    SPT_REAL dt)
{
    SPT_REAL (*P)[6] = s->P;
    SPT_REAL T[3][6], qa, qc, qd, an;
    SPT_REAL wx = (gyro[0] - s->b[0]) * dt, wy = (gyro[1] - s->b[1]) * dt, wz = (gyro[2] - s->b[2]) * dt;
    int i, j, k;

    /* ---- öngörü: nominal q jiroskopla; P ← Φ·P·Φᵀ + Q, Φ = [[I − [ω·dt×], −I·dt], [0, I]] */
    spt_process_noise(s->sigma_gyro, s->sigma_bias, dt, &qa, &qc, &qd);
    spt_qmul_exp(s->q, wx, wy, wz);
    for (j = 0; j < 6; ++j) {
        T[0][j] = P[0][j] + wz * P[1][j] - wy * P[2][j] - dt * P[3][j];
        T[1][j] = P[1][j] - wz * P[0][j] + wx * P[2][j] - dt * P[4][j];
        T[2][j] = P[2][j] + wy * P[0][j] - wx * P[1][j] - dt * P[5][j];
    }
    for (i = 0; i < 3; ++i) {
        for (j = i; j < 3; ++j) {
            P[i][j] = spt_ekf_phi_t(T[i], j, wx, wy, wz, dt);
        }
        for (j = 3; j < 6; ++j) {
            P[i][j] = T[i][j];
        }
        P[i][i] += qa;
        P[i][i + 3] += qc;
        P[i + 3][i + 3] += qd;
    }
    for (i = 0; i < 6; ++i) {
        for (j = i + 1; j < 6; ++j) {
            P[j][i] = P[i][j];
        }
    }

    /* ---- ivmeölçer: v = R(q)ᵀ·ẑ, H = [[v×], 0] */
    an = spt_norm3_valid(acc);
    if (an > SPT_C(0.0)) {
        SPT_REAL v[3], HP[3][6], S[3][3], Si[3][3], K[6][3], e[3], dx[6];
        SPT_REAL r = spt_acc_r(&s->g_ref, &s->g_ref_valid, an, s->sigma_acc);
        SPT_REAL ia = SPT_C(1.0) / an;
        spt_gravity_body(s->q, v);
        for (j = 0; j < 6; ++j) {
            HP[0][j] = v[1] * P[2][j] - v[2] * P[1][j];
            HP[1][j] = v[2] * P[0][j] - v[0] * P[2][j];
            HP[2][j] = v[0] * P[1][j] - v[1] * P[0][j];
        }
        for (i = 0; i < 3; ++i) {
            S[i][0] = v[1] * HP[i][2] - v[2] * HP[i][1];
            S[i][1] = v[2] * HP[i][0] - v[0] * HP[i][2];
            S[i][2] = v[0] * HP[i][1] - v[1] * HP[i][0];
            S[i][i] += r;
        }
        if (spt_inv3(S, Si)) {
            for (j = 0; j < 6; ++j) {
                for (i = 0; i < 3; ++i) {
                    K[j][i] = Si[i][0] * HP[0][j] + Si[i][1] * HP[1][j] + Si[i][2] * HP[2][j];
                }
            }
            e[0] = acc[0] * ia - v[0];
            e[1] = acc[1] * ia - v[1];
            e[2] = acc[2] * ia - v[2];
            for (j = 0; j < 6; ++j) {
                dx[j] = K[j][0] * e[0] + K[j][1] * e[1] + K[j][2] * e[2];
            }
            spt_qmul_exp(s->q, dx[0], dx[1], dx[2]);
            for (k = 0; k < 3; ++k) {
                s->b[k] += dx[3 + k];
            }
            for (i = 0; i < 6; ++i) {
                for (j = i; j < 6; ++j) {
                    P[i][j] -= K[i][0] * HP[0][j] + K[i][1] * HP[1][j] + K[i][2] * HP[2][j];
                    P[j][i] = P[i][j];
                }
            }
        }
    }

#if SPT_USE_MAG
    /* ---- manyetometre: yalnız başlık; H = [h, 0], h = R(q)'nun 3. satırı (dünya z ekseni gövdede) */
    if (mag != NULL && spt_norm3_valid(mag) > SPT_C(0.0)) {
        const SPT_REAL *q = s->q;
        SPT_REAL hn, hd, innov, h[3], Ph[6], K[6], sv, is, kt, kb;
        spt_heading_vec(q, mag, &hn, &hd);
        innov = spt_angle_diff(hn, hd, SPT_C(2.0) * (q[0] * q[3] + q[1] * q[2]),
                               SPT_C(1.0) - SPT_C(2.0) * (q[2] * q[2] + q[3] * q[3]));   /* wrap(başlık − yaw) */
        spt_gravity_body(q, h);
        for (i = 0; i < 6; ++i) {
            Ph[i] = P[i][0] * h[0] + P[i][1] * h[1] + P[i][2] * h[2];
        }
        sv = h[0] * Ph[0] + h[1] * Ph[1] + h[2] * Ph[2] + s->sigma_heading * SPT_D2R * (s->sigma_heading * SPT_D2R);
        is = SPT_C(1.0) / sv;
        for (i = 0; i < 6; ++i) {
            K[i] = Ph[i] * is;
        }
        /* kazanç h'ye izdüşürülür: yalnız dünya z etrafındaki dönme ve o eksendeki sapma düzeltilir */
        kt = h[0] * K[0] + h[1] * K[1] + h[2] * K[2];
        kb = h[0] * K[3] + h[1] * K[4] + h[2] * K[5];
        for (k = 0; k < 3; ++k) {
            K[k] = h[k] * kt;
            K[3 + k] = h[k] * kb;
        }
        spt_qmul_exp(s->q, K[0] * innov, K[1] * innov, K[2] * innov);
        for (k = 0; k < 3; ++k) {
            s->b[k] += K[3 + k] * innov;
        }
        /* Joseph biçimi (I − K·H)·P·(I − K·H)ᵀ + K·Kᵀ·r = P − K·uᵀ − u·Kᵀ + s·K·Kᵀ, u = P·Hᵀ, s = H·P·Hᵀ + r */
        for (i = 0; i < 6; ++i) {
            for (j = i; j < 6; ++j) {
                P[i][j] += sv * K[i] * K[j] - K[i] * Ph[j] - Ph[i] * K[j];
                P[j][i] = P[i][j];
            }
        }
    }
#else
    (void)mag;
#endif
}

void spt_ekf_gyro_bias(const spt_ekf_t *s, SPT_REAL bias[3])
{
    bias[0] = s->b[0];
    bias[1] = s->b[1];
    bias[2] = s->b[2];
}
""",
    "ukf": r"""
#define SPT_UKF_W  SPT_C(0.083333333333333333333)   /* Wm = Wc = 1/12 (i ≥ 1) */
#define SPT_UKF_W0 SPT_C(2.0)                        /* Wc0 (Wm0 = 0) */

void spt_ukf_defaults(spt_ukf_t *s)
{
{{SET_DEFAULTS}}
    s->ready = 0;
}

int spt_ukf_reset(spt_ukf_t *s, const SPT_REAL acc[3], const SPT_REAL mag[3])
{
    int i, j;
    if (!(spt_norm3_valid(acc) > SPT_C(0.0))) {
        s->ready = 0;
        return 0;
    }
    spt_initial_quat(acc, mag, s->q);
    for (i = 0; i < 6; ++i) {
        for (j = 0; j < 6; ++j) {
            s->P[i][j] = SPT_C(0.0);
        }
    }
    for (i = 0; i < 3; ++i) {
        s->b[i] = SPT_C(0.0);
        s->P[i][i] = SPT_C(0.0076154354946677734);          /* (5°)² rad² */
        s->P[i + 3][i + 3] = SPT_C(7.6154354946677734e-05); /* (0.5°/s)² */
    }
    s->g_ref = SPT_C(0.0);
    s->g_ref_valid = 0;
    s->fallbacks = 0u;
    s->ready = 1;
    return 1;
}

/**
 * Python eigh yedeği: L = V·diag(√(6·max(λ, 1e-18))), V / λ simetrik P'nin özvektör / değerleri (dönüşümlü
 * Jacobi). Sigma kümesi {±L_j} özvektör işaret / sırasından bağımsızdır → Python ile aynı küme.
 */
static void spt_ukf_eig_sqrt(spt_ukf_t *s)
{
    SPT_REAL (*a)[6] = s->A, (*v)[6] = s->L;
    int i, j, k, p, r, sweep;
    s->fallbacks++;
    for (i = 0; i < 6; ++i) {
        for (j = 0; j < 6; ++j) {
            a[i][j] = SPT_C(0.5) * (s->P[i][j] + s->P[j][i]);
            v[i][j] = (i == j) ? SPT_C(1.0) : SPT_C(0.0);
        }
    }
    for (sweep = 0; sweep < 50; ++sweep) {
        SPT_REAL off = SPT_C(0.0), diag = SPT_C(0.0);
        for (i = 0; i < 6; ++i) {
            diag += a[i][i] * a[i][i];
            for (j = i + 1; j < 6; ++j) {
                off += a[i][j] * a[i][j];
            }
        }
        if (!(off > SPT_EPS * SPT_EPS * diag)) {
            break;
        }
        for (p = 0; p < 5; ++p) {
            for (r = p + 1; r < 6; ++r) {
                SPT_REAL apr = a[p][r], theta, t, c, sn;
                if (apr == SPT_C(0.0)) {
                    continue;
                }
                theta = (a[r][r] - a[p][p]) / (SPT_C(2.0) * apr);
                t = SPT_C(1.0) / (SPT_FABS(theta) + SPT_SQRT(theta * theta + SPT_C(1.0)));
                if (theta < SPT_C(0.0)) {
                    t = -t;
                }
                c = SPT_C(1.0) / SPT_SQRT(t * t + SPT_C(1.0));
                sn = t * c;
                for (k = 0; k < 6; ++k) {
                    SPT_REAL akp = a[k][p], akr = a[k][r];
                    a[k][p] = c * akp - sn * akr;
                    a[k][r] = sn * akp + c * akr;
                }
                for (k = 0; k < 6; ++k) {
                    SPT_REAL apk = a[p][k], ark = a[r][k];
                    a[p][k] = c * apk - sn * ark;
                    a[r][k] = sn * apk + c * ark;
                }
                for (k = 0; k < 6; ++k) {
                    SPT_REAL vkp = v[k][p], vkr = v[k][r];
                    v[k][p] = c * vkp - sn * vkr;
                    v[k][r] = sn * vkp + c * vkr;
                }
            }
        }
    }
    for (j = 0; j < 6; ++j) {
        SPT_REAL lam = a[j][j], f;
        if (!(lam > SPT_C(1e-18))) {
            lam = SPT_C(1e-18);
        }
        f = SPT_SQRT(SPT_C(6.0) * lam);
        for (i = 0; i < 6; ++i) {
            v[i][j] *= f;
        }
    }
}

/** L ← √(6·P): Cholesky (Python np.linalg.cholesky(N·P)); pivot ≤ 0 ise özdeğer yedeği. */
static void spt_ukf_sqrt(spt_ukf_t *s)
{
    SPT_REAL (*L)[6] = s->L;
    int i, j, k;
    for (j = 0; j < 6; ++j) {
        SPT_REAL d = SPT_C(6.0) * s->P[j][j], id;
        for (k = 0; k < j; ++k) {
            d -= L[j][k] * L[j][k];
        }
        if (!(d > SPT_C(0.0))) {
            spt_ukf_eig_sqrt(s);
            return;
        }
        d = SPT_SQRT(d);
        L[j][j] = d;
        id = SPT_C(1.0) / d;
        for (i = j + 1; i < 6; ++i) {
            SPT_REAL acc = SPT_C(6.0) * s->P[i][j];
            for (k = 0; k < j; ++k) {
                acc -= L[i][k] * L[j][k];
            }
            L[i][j] = acc * id;
        }
        for (i = 0; i < j; ++i) {
            L[i][j] = SPT_C(0.0);
        }
    }
}

/** Sigma sapma vektörü D_i'nin a. bileşeni (a < 3: δθ, a ≥ 3: δb). */
static SPT_REAL spt_ukf_d(const spt_ukf_t *s, int i, int a)
{
    return (a < 3) ? s->dz[i][a] : s->bs[i][a - 3];
}

/**
 * Ortak ölçüm güncellemesi (Python UKF._update). Z_i: s->dz[i][0..m-1] (yerinde dZ'ye çevrilir), z: ölçüm,
 * r: ölçüm varyansı (R = r·I). X_0 = 0, X_{1+j} = +L_j, X_{7+j} = −L_j.
 */
static void spt_ukf_correct(spt_ukf_t *s, int m, const SPT_REAL *z, SPT_REAL r)
{
    SPT_REAL (*Z)[3] = s->dz, (*P)[6] = s->P;
    SPT_REAL zm[3], Pzz[3][3], Pi[3][3], Pxz[6][3], K[6][3], KP[6][3], dx[6];
    int i, j, a, c, e;
    for (a = 0; a < m; ++a) {
        SPT_REAL acc = SPT_C(0.0);
        for (i = 1; i < 13; ++i) {
            acc += Z[i][a];
        }
        zm[a] = acc * SPT_UKF_W;
    }
    for (i = 0; i < 13; ++i) {
        for (a = 0; a < m; ++a) {
            Z[i][a] -= zm[a];
        }
    }
    for (a = 0; a < m; ++a) {
        for (c = a; c < m; ++c) {
            SPT_REAL acc = SPT_C(0.0);
            for (i = 1; i < 13; ++i) {
                acc += Z[i][a] * Z[i][c];
            }
            Pzz[a][c] = acc * SPT_UKF_W + SPT_UKF_W0 * Z[0][a] * Z[0][c];
            Pzz[c][a] = Pzz[a][c];
        }
        Pzz[a][a] += r;
    }
    for (a = 0; a < 6; ++a) {
        for (c = 0; c < m; ++c) {
            SPT_REAL acc = SPT_C(0.0);
            for (j = 0; j < 6; ++j) {
                acc += s->L[a][j] * (Z[1 + j][c] - Z[7 + j][c]);
            }
            Pxz[a][c] = acc * SPT_UKF_W;
        }
    }
    if (m == 1) {
        if (!(Pzz[0][0] != SPT_C(0.0))) {
            return;
        }
        Pi[0][0] = SPT_C(1.0) / Pzz[0][0];
    } else if (!spt_inv3(Pzz, Pi)) {
        return;
    }
    for (a = 0; a < 6; ++a) {      /* K = Pxz·Pzz⁻ᵀ (Python solve(Pzz, Pxzᵀ)ᵀ) */
        for (c = 0; c < m; ++c) {
            SPT_REAL acc = SPT_C(0.0);
            for (e = 0; e < m; ++e) {
                acc += Pxz[a][e] * Pi[c][e];
            }
            K[a][c] = acc;
        }
    }
    for (a = 0; a < 6; ++a) {
        SPT_REAL acc = SPT_C(0.0);
        for (c = 0; c < m; ++c) {
            acc += K[a][c] * (z[c] - zm[c]);
        }
        dx[a] = acc;
    }
    spt_qmul_exp(s->q, dx[0], dx[1], dx[2]);
    for (a = 0; a < 3; ++a) {
        s->b[a] += dx[3 + a];
    }
    for (a = 0; a < 6; ++a) {      /* P ← P − K·Pzz·Kᵀ (üst üçgen + ayna) */
        for (c = 0; c < m; ++c) {
            SPT_REAL acc = SPT_C(0.0);
            for (e = 0; e < m; ++e) {
                acc += K[a][e] * Pzz[e][c];
            }
            KP[a][c] = acc;
        }
    }
    for (a = 0; a < 6; ++a) {
        for (c = a; c < 6; ++c) {
            SPT_REAL acc = SPT_C(0.0);
            for (e = 0; e < m; ++e) {
                acc += KP[a][e] * K[c][e];
            }
            P[a][c] -= acc;
            P[c][a] = P[a][c];
        }
    }
}

void spt_ukf_update(spt_ukf_t *s, const SPT_REAL gyro[3], const SPT_REAL acc[3], const SPT_REAL mag[3],
                    SPT_REAL dt)
{
    SPT_REAL (*P)[6] = s->P, (*L)[6] = s->L;
    SPT_REAL qa, qc, qd, mth[3], bp[3], qm[4], t[4], e[4], an;
    int i, j, a, c;

    /* ---- öngörü: sigma noktaları q ⊗ exp(X_i[:3]), b + X_i[3:], jiroskopla ilerletilir */
    spt_process_noise(s->sigma_gyro, s->sigma_bias, dt, &qa, &qc, &qd);
    spt_ukf_sqrt(s);
    for (a = 0; a < 4; ++a) {
        s->qs[0][a] = s->q[a];
    }
    for (a = 0; a < 3; ++a) {
        s->bs[0][a] = s->b[a];
    }
    for (j = 0; j < 6; ++j) {
        spt_quat_exp(L[0][j], L[1][j], L[2][j], e);
        spt_quat_mul(s->q, e, s->qs[1 + j]);
        e[1] = -e[1];   /* exp(−v): kosinüs ve sinc çift → yalnız vektör kısmının işareti */
        e[2] = -e[2];
        e[3] = -e[3];
        spt_quat_mul(s->q, e, s->qs[7 + j]);
        for (a = 0; a < 3; ++a) {
            s->bs[1 + j][a] = s->b[a] + L[3 + a][j];
            s->bs[7 + j][a] = s->b[a] - L[3 + a][j];
        }
    }
    for (i = 0; i < 13; ++i) {
        spt_quat_exp((gyro[0] - s->bs[i][0]) * dt, (gyro[1] - s->bs[i][1]) * dt, (gyro[2] - s->bs[i][2]) * dt, e);
        spt_quat_mul(s->qs[i], e, t);
        spt_quat_normalize(t, s->qs[i]);
    }
    /* δθ_i = log(q̄* ⊗ q_i), q̄ = q_0; ağırlıklı ortalamalar ve kovaryans */
    qm[0] = s->qs[0][0];
    qm[1] = -s->qs[0][1];
    qm[2] = -s->qs[0][2];
    qm[3] = -s->qs[0][3];
    for (i = 0; i < 13; ++i) {
        spt_quat_mul(qm, s->qs[i], t);
        spt_quat_log(t, s->dz[i]);
    }
    for (a = 0; a < 3; ++a) {
        SPT_REAL st = SPT_C(0.0), sb = SPT_C(0.0);
        for (i = 1; i < 13; ++i) {
            st += s->dz[i][a];
            sb += s->bs[i][a];
        }
        mth[a] = st * SPT_UKF_W;
        bp[a] = sb * SPT_UKF_W;
    }
    for (i = 0; i < 13; ++i) {
        for (a = 0; a < 3; ++a) {
            s->dz[i][a] -= mth[a];
            s->bs[i][a] -= bp[a];
        }
    }
    for (a = 0; a < 6; ++a) {
        for (c = a; c < 6; ++c) {
            SPT_REAL sum = SPT_C(0.0);
            for (i = 1; i < 13; ++i) {
                sum += spt_ukf_d(s, i, a) * spt_ukf_d(s, i, c);
            }
            P[a][c] = sum * SPT_UKF_W + SPT_UKF_W0 * spt_ukf_d(s, 0, a) * spt_ukf_d(s, 0, c);
        }
    }
    for (a = 0; a < 3; ++a) {
        P[a][a] += qa;
        P[a][a + 3] += qc;
        P[a + 3][a + 3] += qd;
    }
    for (a = 0; a < 6; ++a) {
        for (c = a + 1; c < 6; ++c) {
            P[c][a] = P[a][c];
        }
    }
    for (a = 0; a < 4; ++a) {
        s->q[a] = s->qs[0][a];
    }
    spt_qmul_exp(s->q, mth[0], mth[1], mth[2]);
    for (a = 0; a < 3; ++a) {
        s->b[a] = bp[a];
    }

    /* ---- ivmeölçer: Z_i = R(q ⊗ exp(X_i[:3]))ᵀ·ẑ */
    an = spt_norm3_valid(acc);
    if (an > SPT_C(0.0)) {
        SPT_REAL z[3], ia = SPT_C(1.0) / an;
        SPT_REAL r = spt_acc_r(&s->g_ref, &s->g_ref_valid, an, s->sigma_acc);
        spt_ukf_sqrt(s);
        spt_gravity_body(s->q, s->dz[0]);
        for (j = 0; j < 6; ++j) {
            spt_quat_exp(L[0][j], L[1][j], L[2][j], e);
            spt_quat_mul(s->q, e, t);
            spt_gravity_body(t, s->dz[1 + j]);
            e[1] = -e[1];
            e[2] = -e[2];
            e[3] = -e[3];
            spt_quat_mul(s->q, e, t);
            spt_gravity_body(t, s->dz[7 + j]);
        }
        z[0] = acc[0] * ia;
        z[1] = acc[1] * ia;
        z[2] = acc[2] * ia;
        spt_ukf_correct(s, 3, z, r);
    }

#if SPT_USE_MAG
    /* ---- manyetometre: başlık, dünya z etrafındaki dönme olarak (Z_i = h·δθ_i, h = R(q)'nun 3. satırı) */
    if (mag != NULL && spt_norm3_valid(mag) > SPT_C(0.0)) {
        const SPT_REAL *q = s->q;
        SPT_REAL hn, hd, h[3], innov, rh = s->sigma_heading * SPT_D2R;
        spt_heading_vec(q, mag, &hn, &hd);
        innov = spt_angle_diff(hn, hd, SPT_C(2.0) * (q[0] * q[3] + q[1] * q[2]),
                               SPT_C(1.0) - SPT_C(2.0) * (q[2] * q[2] + q[3] * q[3]));   /* wrap(başlık − yaw) */
        spt_gravity_body(q, h);
        spt_ukf_sqrt(s);
        s->dz[0][0] = SPT_C(0.0);
        for (j = 0; j < 6; ++j) {
            SPT_REAL zj = L[0][j] * h[0] + L[1][j] * h[1] + L[2][j] * h[2];
            s->dz[1 + j][0] = zj;
            s->dz[7 + j][0] = -zj;
        }
        spt_ukf_correct(s, 1, &innov, rh * rh);
    }
#else
    (void)mag;
#endif
}

void spt_ukf_gyro_bias(const spt_ukf_t *s, SPT_REAL bias[3])
{
    bias[0] = s->b[0];
    bias[1] = s->b[1];
    bias[2] = s->b[2];
}
""",
    "pf": r"""
#define SPT_PF_LOGW_MIN SPT_C(-690.77552789821368)   /* log(1e-300): Python'daki ağırlık alt sınırı */

/** Tohum karıştırma (murmur3 son adımı); xorshift32 durumu sıfır olamaz. */
static uint32_t spt_pf_hash(uint32_t x)
{
    x += 0x9E3779B9u;
    x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
    x = (x ^ (x >> 13)) * 0xC2B2AE35u;
    x ^= x >> 16;
    return x ? x : 0x6D2B79F5u;
}

/** xorshift32 (Marsaglia 13/17/5). */
static uint32_t spt_pf_next(spt_pf_t *s)
{
    uint32_t x = s->rng;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    s->rng = x;
    return x;
}

/** [0, 1) tekdüze sayı (üst 24 bit). */
static SPT_REAL spt_pf_uniform(spt_pf_t *s)
{
    return SPT_U32_TO_REAL(spt_pf_next(s) >> 8) * SPT_C(5.9604644775390625e-08);
}

/** Standart normal sayı: Marsaglia kutupsal yöntemi (çiftin ikincisi saklanır). */
static SPT_REAL spt_pf_normal(spt_pf_t *s)
{
    SPT_REAL u, v, r2, f;
    if (s->has_spare) {
        s->has_spare = 0;
        return s->spare;
    }
    do {
        u = SPT_C(2.0) * spt_pf_uniform(s) - SPT_C(1.0);
        v = SPT_C(2.0) * spt_pf_uniform(s) - SPT_C(1.0);
        r2 = u * u + v * v;
    } while (r2 >= SPT_C(1.0) || r2 == SPT_C(0.0));
    f = SPT_SQRT(SPT_C(-2.0) * SPT_LOG(r2) / r2);
    s->spare = v * f;
    s->has_spare = 1;
    return u * f;
}

void spt_pf_defaults(spt_pf_t *s)
{
{{SET_DEFAULTS}}
    s->ready = 0;
}

int spt_pf_reset(spt_pf_t *s, const SPT_REAL acc[3], const SPT_REAL mag[3])
{
    const int n = SPT_PF_PARTICLES;
    SPT_REAL q0[4], sd, lw;
    int i, k;
    if (!(spt_norm3_valid(acc) > SPT_C(0.0))) {
        s->ready = 0;
        return 0;
    }
    spt_initial_quat(acc, mag, q0);
    s->rng = spt_pf_hash(s->seed);
    s->has_spare = 0;
    s->spare = SPT_C(0.0);
    sd = SPT_C(3.0) * SPT_D2R;                 /* yönelim yayılımı 3° */
    for (i = 0; i < n; ++i) {
        SPT_REAL vx, vy, vz;
        vx = sd * spt_pf_normal(s);
        vy = sd * spt_pf_normal(s);
        vz = sd * spt_pf_normal(s);
        for (k = 0; k < 4; ++k) {
            s->q[i][k] = q0[k];
        }
        spt_qmul_exp(s->q[i], vx, vy, vz);
    }
    sd = SPT_C(0.3) * SPT_D2R;                 /* sapma yayılımı 0.3°/s */
    for (i = 0; i < n; ++i) {
        for (k = 0; k < 3; ++k) {
            s->b[i][k] = sd * spt_pf_normal(s);
        }
    }
    lw = -SPT_LOG(SPT_FROM_INT(n));
    for (i = 0; i < n; ++i) {
        s->logw[i] = lw;
    }
    for (k = 0; k < 4; ++k) {
        s->q_out[k] = q0[k];
    }
    s->g_ref = SPT_C(0.0);
    s->g_ref_valid = 0;
    s->ess = SPT_FROM_INT(n);
    s->resamples = 0u;
    s->ready = 1;
    return 1;
}

/** Sistematik yeniden örnekleme (Python searchsorted ile aynı atama) + pürüzlendirme; logw = log w − max. */
static void spt_pf_resample(spt_pf_t *s, SPT_REAL inv_sum)
{
    const int n = SPT_PF_PARTICLES;
    SPT_REAL inv_n = SPT_C(1.0) / SPT_FROM_INT(n), u = spt_pf_uniform(s), cum = SPT_C(0.0), sd, lw;
    int i, j = 0, k, free_slot = 0;
    /* parçacık i'nin kopya sayısı: (u + j)/n konumlarından cum_{i−1} < konum ≤ cum_i olanlar */
    for (i = 0; i < n; ++i) {
        int c = 0;
        cum += SPT_EXP(s->logw[i]) * inv_sum;
        while (j < n && (u + SPT_FROM_INT(j)) * inv_n <= cum) {
            ++c;
            ++j;
        }
        s->cnt[i] = (uint16_t)c;
    }
    s->cnt[n - 1] = (uint16_t)(s->cnt[n - 1] + (n - j));   /* min(idx, n − 1): yuvarlama artığı */
    /* yerinde kopyalama: fazla kopyalar seçilmemiş (sayısı 0) yuvalara */
    for (i = 0; i < n; ++i) {
        while (s->cnt[i] > 1) {
            while (s->cnt[free_slot] != 0) {
                ++free_slot;
            }
            for (k = 0; k < 4; ++k) {
                s->q[free_slot][k] = s->q[i][k];
            }
            for (k = 0; k < 3; ++k) {
                s->b[free_slot][k] = s->b[i][k];
            }
            s->cnt[free_slot] = 1;
            s->cnt[i] = (uint16_t)(s->cnt[i] - 1);
        }
    }
    sd = SPT_C(0.2) * SPT_D2R;                 /* pürüzlendirme: 0.2° yönelim, 0.01°/s sapma */
    for (i = 0; i < n; ++i) {
        SPT_REAL vx, vy, vz;
        vx = sd * spt_pf_normal(s);
        vy = sd * spt_pf_normal(s);
        vz = sd * spt_pf_normal(s);
        spt_qmul_exp(s->q[i], vx, vy, vz);
    }
    sd = SPT_C(0.01) * SPT_D2R;
    for (i = 0; i < n; ++i) {
        for (k = 0; k < 3; ++k) {
            s->b[i][k] += sd * spt_pf_normal(s);
        }
    }
    lw = -SPT_LOG(SPT_FROM_INT(n));
    for (i = 0; i < n; ++i) {
        s->logw[i] = lw;
    }
    s->resamples++;
}

void spt_pf_update(spt_pf_t *s, const SPT_REAL gyro[3], const SPT_REAL acc[3], const SPT_REAL mag[3],
                   SPT_REAL dt)
{
    const int n = SPT_PF_PARTICLES;
    SPT_REAL sqdt = SPT_SQRT(dt);
    SPT_REAL sg = s->sigma_gyro * SPT_D2R / sqdt;   /* spektral yoğunluk → örnek başına σ */
    SPT_REAL sb = s->sigma_bias * SPT_D2R * sqdt;
    SPT_REAL an, lmax, sum, sum2, mean[4], nrm;
    const SPT_REAL *ref;
    int i, k, imax;

    /* ---- öngörü: gürültülü jiroskopla her parçacık; sapma rastgele yürüyüşü */
    for (i = 0; i < n; ++i) {
        SPT_REAL ox, oy, oz;
        ox = gyro[0] - s->b[i][0] + sg * spt_pf_normal(s);
        oy = gyro[1] - s->b[i][1] + sg * spt_pf_normal(s);
        oz = gyro[2] - s->b[i][2] + sg * spt_pf_normal(s);
        spt_qmul_exp(s->q[i], ox * dt, oy * dt, oz * dt);
        for (k = 0; k < 3; ++k) {
            s->b[i][k] += sb * spt_pf_normal(s);
        }
    }

    /* ---- ivmeölçer olabilirliği (doğrusal ivmede genişler) */
    an = spt_norm3_valid(acc);
    if (an > SPT_C(0.0)) {
        SPT_REAL dev, sa2, k2, ia = SPT_C(1.0) / an, ax, ay, az;
        if (!s->g_ref_valid) {
            s->g_ref = an;
            s->g_ref_valid = 1;
        } else {
            s->g_ref += SPT_C(0.001) * (an - s->g_ref);
        }
        dev = SPT_FABS(an / s->g_ref - SPT_C(1.0)) * SPT_C(50.0);
        sa2 = s->sigma_acc * s->sigma_acc * (SPT_C(1.0) + dev * dev);
        k2 = SPT_C(0.5) / sa2;
        ax = acc[0] * ia;
        ay = acc[1] * ia;
        az = acc[2] * ia;
        for (i = 0; i < n; ++i) {
            const SPT_REAL *q = s->q[i];
            SPT_REAL ex = ax - SPT_C(2.0) * (q[1] * q[3] - q[0] * q[2]);
            SPT_REAL ey = ay - SPT_C(2.0) * (q[2] * q[3] + q[0] * q[1]);
            SPT_REAL ez = az - (SPT_C(1.0) - SPT_C(2.0) * (q[1] * q[1] + q[2] * q[2]));
            s->logw[i] -= (ex * ex + ey * ey + ez * ez) * k2;
        }
    }

#if SPT_USE_MAG
    /* ---- başlık olabilirliği: en olası parçacığın eğimiyle başlık; parçacık başlığı ona göre dünya z etrafındaki
       twist açısıyla: r = q_i ⊗ q_ref*, twist = 2·atan2(r_z, r_w) */
    if (mag != NULL && spt_norm3_valid(mag) > SPT_C(0.0)) {
        const SPT_REAL *qb;
        SPT_REAL hn, hd, d, k2, sh = s->sigma_heading * SPT_D2R;
        int best = 0;
        for (i = 1; i < n; ++i) {
            if (s->logw[i] > s->logw[best]) {
                best = i;
            }
        }
        qb = s->q[best];
        spt_heading_vec(qb, mag, &hn, &hd);
        d = spt_angle_diff(SPT_C(2.0) * (qb[0] * qb[3] + qb[1] * qb[2]),
                           SPT_C(1.0) - SPT_C(2.0) * (qb[2] * qb[2] + qb[3] * qb[3]), hn, hd);  /* yaw_ref − başlık */
        k2 = SPT_C(0.5) / (sh * sh);
        for (i = 0; i < n; ++i) {
            const SPT_REAL *q = s->q[i];
            SPT_REAL rw = q[0] * qb[0] + q[1] * qb[1] + q[2] * qb[2] + q[3] * qb[3];
            SPT_REAL rz = q[3] * qb[0] - q[0] * qb[3] + q[2] * qb[1] - q[1] * qb[2];
            SPT_REAL e = spt_wrap_pi(d + SPT_C(2.0) * SPT_ATAN2(rz, rw));
            s->logw[i] -= e * e * k2;
        }
    }
#else
    (void)mag;
#endif

    /* ---- normalizasyon, ağırlıklı ortalama (en ağır parçacığa göre işaret hizalı), ESS */
    lmax = s->logw[0];
    imax = 0;
    for (i = 1; i < n; ++i) {
        if (s->logw[i] > lmax) {
            lmax = s->logw[i];
            imax = i;
        }
    }
    ref = s->q[imax];
    sum = SPT_C(0.0);
    sum2 = SPT_C(0.0);
    for (k = 0; k < 4; ++k) {
        mean[k] = SPT_C(0.0);
    }
    for (i = 0; i < n; ++i) {
        const SPT_REAL *q = s->q[i];
        SPT_REAL w;
        s->logw[i] -= lmax;
        w = SPT_EXP(s->logw[i]);
        sum += w;
        sum2 += w * w;
        if (q[0] * ref[0] + q[1] * ref[1] + q[2] * ref[2] + q[3] * ref[3] < SPT_C(0.0)) {
            w = -w;
        }
        for (k = 0; k < 4; ++k) {
            mean[k] += w * q[k];
        }
    }
    nrm = SPT_C(1.0) / SPT_SQRT(mean[0] * mean[0] + mean[1] * mean[1] + mean[2] * mean[2] + mean[3] * mean[3]);
    for (k = 0; k < 4; ++k) {
        s->q_out[k] = mean[k] * nrm;
    }
    s->ess = sum * sum / sum2;
    if (s->ess < SPT_C(0.5) * SPT_FROM_INT(n)) {
        spt_pf_resample(s, SPT_C(1.0) / sum);
    } else {
        SPT_REAL ls = SPT_LOG(sum);
        for (i = 0; i < n; ++i) {
            SPT_REAL v = s->logw[i] - ls;
            s->logw[i] = (v > SPT_PF_LOGW_MIN) ? v : SPT_PF_LOGW_MIN;
        }
    }
}

void spt_pf_gyro_bias(const spt_pf_t *s, SPT_REAL bias[3])
{
    const int n = SPT_PF_PARTICLES;
    SPT_REAL lmax = s->logw[0], sum = SPT_C(0.0), acc[3];
    int i, k;
    for (i = 1; i < n; ++i) {
        if (s->logw[i] > lmax) {
            lmax = s->logw[i];
        }
    }
    acc[0] = SPT_C(0.0);
    acc[1] = SPT_C(0.0);
    acc[2] = SPT_C(0.0);
    for (i = 0; i < n; ++i) {
        SPT_REAL w = SPT_EXP(s->logw[i] - lmax);
        sum += w;
        for (k = 0; k < 3; ++k) {
            acc[k] += w * s->b[i][k];
        }
    }
    for (k = 0; k < 3; ++k) {
        bias[k] = acc[k] / sum;
    }
}
""",
}
_QUAT_FIELD = {"mahony": "q", "madgwick": "q", "ekf": "q", "ukf": "q", "pf": "q_out"}   # kalman: açıdan


def _c_float(v: float) -> str:
    r = repr(float(v))
    return r if any(c in r for c in ".eE") else r + ".0"


def _fill(text: str, **kw: str) -> str:
    for k, v in kw.items():
        text = text.replace("{{" + k + "}}", v)
    left = re.findall(r"\{\{[A-Z_]+\}\}", text)
    assert not left, left
    return text


def _prefixed(text: str, prefix: str) -> str:
    if prefix != "spt":
        text = re.sub(r"\bspt_", prefix.lower() + "_", text)
        text = re.sub(r"\bSPT_", prefix.upper() + "_", text)
    return _align(text)


def _align(text: str) -> str:
    """Fonksiyon imzalarının devam satırlarını açılış parantezine hizalar (önek uzunluğu değişebilir)."""
    lines = text.split("\n")
    for i in range(1, len(lines)):
        prev, cur = lines[i - 1], lines[i]
        m = re.match(r"^(?:static )?(?:inline )?(?:void|int|uint32_t|\w+_REAL) \w+\(", prev)
        if m and prev.rstrip().endswith(",") and cur.startswith(" ") and re.match(r"(const )?\w+_REAL\b", cur.strip()):
            lines[i] = " " * prev.index("(", m.start()) + " " + cur.strip()
    return "\n".join(lines)


def _param_block(key: str, values: dict[str, float]) -> tuple[str, str]:
    """(#define bloğu, defaults() gövdesi)."""
    cls = fu.ALGORITHMS[key]
    defines, sets = [], []
    for p in cls.params:
        macro = f"SPT_{key.upper()}_{p.key.upper()}"
        v = values[p.key]
        if key == "pf" and p.key == "particles":
            literal = str(int(v))
        elif key == "pf" and p.key == "seed":
            literal = f"{int(v)}u"
        else:
            literal = _c_float(v)
        tip = f" — {p.tip}" if p.tip else ""
        defines.append(f"/** {p.label}{tip} (Python varsayılanı {p.default:g}) */\n"
                       f"#ifndef {macro}\n#define {macro} {literal}\n#endif")
        if key == "pf" and p.key == "particles":
            continue
        if key == "pf" and p.key == "seed":
            sets.append(f"    s->seed = (uint32_t)({macro});")
        else:
            sets.append(f"    s->{p.key} = (SPT_REAL)({macro});")
    return "\n".join(defines), "\n".join(sets)


def _algo_files(key: str, values: dict[str, float]) -> tuple[str, str]:
    defines, sets = _param_block(key, values)
    api = _fill(_API_DECLS, A=key).strip("\n")
    header = _fill(_HEADER, A=key, AU=key.upper(), BRIEF=_BRIEFS[key], BANNER=_BANNER, PYCLASS=_PYCLASS[key],
                   PARAM_DEFINES=defines, STRUCT=_STRUCTS[key], API=api, EXTRA_DECLS=_EXTRA_DECLS[key])
    body = _fill(_SOURCES[key], SET_DEFAULTS=sets)
    quat = "" if key == "kalman" else _fill(_QUAT_EULER_IMPL, A=key, Q=_QUAT_FIELD[key])
    source = (f"/**\n * @file spt_{key}.c\n * @brief {_TITLES[key]} — uygulama (Python: spt.dsp.fusion."
              f"{_PYCLASS[key]}).\n *\n * {_BANNER}\n */\n#include \"spt_{key}.h\"\n"
              + body + _fill(_API_IMPL, A=key) + quat)
    return header, source


def _common_header(opts: CodegenOptions) -> str:
    return _fill(_COMMON_H, BANNER=_BANNER, REAL=opts.real, USE_DOUBLE="1" if opts.real == "double" else "0",
                 USE_MAG="1" if opts.use_mag else "0")


def _example(opts: CodegenOptions) -> str:
    algos = _ordered(opts)
    inc = "\n".join(f'#include "spt_{a}.h"' for a in algos)
    states = "\n".join(f"    static spt_{a}_t f_{a};   /* {_TITLES[a]} */" for a in algos)
    defaults = "\n".join(f"    spt_{a}_defaults(&f_{a});" for a in algos)
    first = algos[0]
    steps = "\n".join(f"        (void)spt_{a}_step(&f_{a}, gyro, acc, mag_ok ? mag : NULL, dt);" for a in algos[1:])
    others = ("\n        /* diğer seçili algoritmalar (karşılaştırma için; gerekmiyorsa silin) */\n" + steps) if steps \
        else ""
    return _fill(r"""/**
 * @file ornek_kullanim.c
 * @brief Örnek: sabit örnekleme hızında füzyon güncellemesi ve Euler açılarının okunması.
 *
 * {{BANNER}}
 *
 * Donanıma bağlı üç fonksiyon (imu_oku, ornek_bekle, sonuc_kullan) kullanıcı tarafından yazılır (STM32 HAL,
 * Zephyr, ESP-IDF …). fuzyon_gorevi() main() içinden ya da bir RTOS görevinden çağrılır ve geri dönmez.
 */
#include <stdint.h>
#include <stddef.h>
{{INCLUDES}}

#ifndef ORNEK_FS_HZ
#define ORNEK_FS_HZ {{FS}}            /* örnekleme frekansı (Hz) — zamanlayıcı bu hızda tetiklemeli */
#endif
#ifndef ORNEK_CPU_HZ
#define ORNEK_CPU_HZ {{CPU}}       /* çekirdek saati (Hz) — DWT->CYCCNT çevrimlerini süreye çevirmek için */
#endif
#ifndef ORNEK_DT_OLC
#define ORNEK_DT_OLC 0                /* 1: dt her örnekte DWT->CYCCNT ile ölçülür, 0: sabit 1/fs (önerilen) */
#endif

/* ---- kullanıcı tarafından sağlanır --------------------------------------------------------------------------- */
/** Bir örnek okur. gyro: rad/s; acc / mag: herhangi bir birim (yalnız yön kullanılır). Eksenler gövde çerçevesine
 *  (sağ el, z yukarı) hizalanmış olmalı. *mag_gecerli: bu örnekte yeni manyetometre verisi var mı.
 *  @return 1: yeni örnek, 0: yok */
extern int imu_oku(SPT_REAL gyro[3], SPT_REAL acc[3], SPT_REAL mag[3], int *mag_gecerli);
/** Bir sonraki örnekleme anına kadar bekler (zamanlayıcı kesmesinin kurduğu bayrak, vTaskDelayUntil …). */
extern void ornek_bekle(void);
/** Sonucu kullanır (UART, CAN, kontrol döngüsü …): açılar derece, güncellemenin süresi çevrim. */
extern void sonuc_kullan(SPT_REAL roll_deg, SPT_REAL pitch_deg, SPT_REAL yaw_deg, uint32_t cevrim);

/* ---- DWT çevrim sayacı (Cortex-M3/M4/M7/M33; M0+ ve diğerlerinde 0 döner) ------------------------------------- */
#if defined(__ARM_ARCH_7M__) || defined(__ARM_ARCH_7EM__) || defined(__ARM_ARCH_8M_MAIN__)
#define ORNEK_DWT_CTRL   (*(volatile uint32_t *)0xE0001000u)
#define ORNEK_DWT_CYCCNT (*(volatile uint32_t *)0xE0001004u)
#define ORNEK_DEMCR      (*(volatile uint32_t *)0xE000EDFCu)
static void cyccnt_baslat(void)
{
    ORNEK_DEMCR |= 0x01000000u;   /* TRCENA */
    ORNEK_DWT_CYCCNT = 0u;
    ORNEK_DWT_CTRL |= 1u;         /* CYCCNTENA */
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

void fuzyon_gorevi(void);

void fuzyon_gorevi(void)
{
{{STATES}}
    SPT_REAL gyro[3], acc[3], mag[3], roll, pitch, yaw;
    const SPT_REAL dt_sabit = (SPT_REAL)(1.0 / ORNEK_FS_HZ);
    int mag_ok = 0;
#if ORNEK_DT_OLC
    uint32_t onceki = 0u;
    int ilk = 1;
#endif
{{DEFAULTS}}
    cyccnt_baslat();
    for (;;) {
        SPT_REAL dt = dt_sabit;
        uint32_t t0, t1;
        int ok;
        ornek_bekle();
        if (!imu_oku(gyro, acc, mag, &mag_ok)) {
            continue;
        }
#if ORNEK_DT_OLC
        {
            uint32_t simdi = cyccnt();
            if (!ilk && simdi != onceki) {
                dt = (SPT_REAL)((double)(uint32_t)(simdi - onceki) / ORNEK_CPU_HZ);
            }
            onceki = simdi;
            ilk = 0;
        }
#endif
        t0 = cyccnt();
        /* step(): geçersiz jiroskopta durumu unutur, ilk geçerli ivme (+ manyetometre) örneğinde kendisi başlatır */
        ok = spt_{{FIRST}}_step(&f_{{FIRST}}, gyro, acc, mag_ok ? mag : NULL, dt);
        t1 = cyccnt();{{OTHERS}}
        if (ok) {
            spt_{{FIRST}}_euler(&f_{{FIRST}}, &roll, &pitch, &yaw);
            sonuc_kullan(roll * SPT_C(57.295779513082321), pitch * SPT_C(57.295779513082321),
                         yaw * SPT_C(57.295779513082321), (uint32_t)(t1 - t0));
        }
    }
}
""", BANNER=_BANNER, INCLUDES=inc, FS=_c_float(opts.fs_hz), CPU=_c_float(opts.clock_mhz * 1e6), STATES=states,
                 DEFAULTS=defaults, FIRST=first, OTHERS=others)


def _sources(opts: CodegenOptions) -> list[GeneratedFile]:
    """C dosyaları (önek uygulanmamış: spt_*)."""
    files = [GeneratedFile("spt_fusion_common.h", _common_header(opts))]
    for key in _ordered(opts):
        header, source = _algo_files(key, _resolved_params(key, opts.params.get(key)))
        files += [GeneratedFile(f"spt_{key}.h", header), GeneratedFile(f"spt_{key}.c", source)]
    return files


def generate(opts: CodegenOptions, footprint_rows: list[FootprintRow] | None = None,
             cost_rows: list[CostRow] | None = None) -> list[GeneratedFile]:
    """Ortak başlık, algoritma başına .h / .c, ornek_kullanim.c ve KULLANIM.md (önek uygulanmış).

    footprint_rows / cost_rows verilirse ölçüm tabloları KULLANIM.md'ye eklenir."""
    check_options(opts)
    files = _sources(opts) + [GeneratedFile("ornek_kullanim.c", _example(opts))]
    files = [GeneratedFile(_prefixed(f.name, opts.prefix), _prefixed(f.text, opts.prefix)) for f in files]
    files.append(GeneratedFile("KULLANIM.md", usage_markdown(opts, footprint_rows, cost_rows)))
    return files


def write_files(files: list[GeneratedFile], folder: str | Path) -> list[Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for f in files:
        p = folder / f.name
        p.write_text(f.text, encoding="utf-8", newline="\n")
        out.append(p)
    return out


# ============================================================================ durum yapısı boyutu (analitik)
def _layout(key: str, particles: int) -> tuple[int, int, int]:
    """(gerçek sayı, 4 baytlık tamsayı, 2 baytlık tamsayı) alan sayıları — yapılar bu sırayla tanımlıdır."""
    return {"mahony": (9, 1, 0), "madgwick": (5, 1, 0), "kalman": (21, 1, 0), "ekf": (48, 2, 0),
            "ukf": (250, 3, 0), "pf": (8 * particles + 11, 6, particles)}[key]


def state_size(key: str, real: str, particles: int = 600) -> int:
    """sizeof(durum yapısı): ILP32 (ARM AAPCS, RISC-V ilp32, Xtensa) ve x86-64'te aynı (işaretçi alanı yok)."""
    r = 8 if real == "double" else 4
    reals, words, halves = _layout(key, particles)
    size = reals * r + words * 4 + halves * 2
    return -(-size // r) * r


def _particles(opts: CodegenOptions) -> int:
    return int(_resolved_params("pf", opts.params.get("pf"))["particles"])


# ============================================================================ ayak izi
_FP_CACHE: dict[str, list[dict]] = {}
_FP_VERSION = 4


def _harness(key: str, use_mag: bool, baseline: bool) -> str:
    getter = _BIAS_GETTER[key]
    bias = f"spt_{key}_{getter}(&st, b);" if getter else "b[0] = b[1] = b[2] = SPT_C(0.0);"
    mag = "m" if use_mag else "NULL"
    calls = "" if baseline else f"""
        spt_out[7] = SPT_FROM_INT(spt_{key}_step(&st, g, a, {mag}, spt_in[9]));
        spt_{key}_quat(&st, q);
        spt_{key}_euler(&st, &r, &p, &y);
        {bias}"""
    copy = "        for (i = 0; i < 4; ++i) q[i] = g[i % 3] + a[i % 3] + m[i % 3];\n        r = p = y = b[0] = b[1] = " \
           "b[2] = spt_in[9];\n" if baseline else ""
    init = "" if baseline else f"    (void)spt_{key}_init(&st, a, {mag});\n"
    state = "" if baseline else f"static spt_{key}_t st;\n"
    return f"""/* Ayak izi ölçüm koşumu ({'temel' if baseline else key}): tüm genel API çağrılır, giriş / çıkış volatile. */
#include "spt_{key}.h"
{state}volatile SPT_REAL spt_in[10];
volatile SPT_REAL spt_out[8];
int main(void);
int main(void)
{{
    SPT_REAL g[3], a[3], m[3], q[4], r, p, y, b[3];
    int i;
    for (i = 0; i < 3; ++i) {{ g[i] = spt_in[i]; a[i] = spt_in[3 + i]; m[i] = spt_in[6 + i]; }}
{init}    for (;;) {{
        for (i = 0; i < 3; ++i) {{ g[i] = spt_in[i]; a[i] = spt_in[3 + i]; m[i] = spt_in[6 + i]; }}{calls}
{copy}        for (i = 0; i < 4; ++i) spt_out[i] = q[i];
        spt_out[4] = r + p + y;
        spt_out[5] = b[0] + b[1] + b[2];
    }}
}}
"""


def _measure(opts: CodegenOptions, prof: McuProfile, compiler: str) -> list[dict] | None:
    """Bir profil + derleyici için algoritma başına ölçüm; derleyici yoksa None."""
    tool = tc.cross_tool(compiler)
    if tool is None or tc.cross_flags(prof, compiler) is None:
        return None
    algos = _ordered(opts)
    files = {f.name: f.text for f in _sources(opts)}
    real_def = f"SPT_USE_DOUBLE={1 if opts.real == 'double' else 0}"
    probe = "\n".join(f'#include "spt_{a}.h"' for a in algos) + "\n" + \
        "\n".join(f"unsigned char spt_sizeof_{a}[sizeof(spt_{a}_t)];" for a in algos) + "\n"
    files["spt_sizeof_probe.c"] = probe
    gcc = tc.find_arm_gcc()
    key_blob = json.dumps([_FP_VERSION, tc.STACK_ANALYSIS_VERSION, sorted(files.items()), prof.key, compiler, opts.opt,
                           real_def, opts.use_mag,
                           tool.path, tool.version, gcc.version if gcc else None])
    key = hashlib.sha256(key_blob.encode()).hexdigest()[:24]
    cache_file = tc.WORK_ROOT / "footprint" / f"{key}.json"
    if key in _FP_CACHE:
        return _FP_CACHE[key]
    if cache_file.exists():
        try:
            _FP_CACHE[key] = json.loads(cache_file.read_text(encoding="utf-8"))
            return _FP_CACHE[key]
        except (OSError, ValueError):
            pass
    sources = [f"spt_{a}.c" for a in algos] + ["spt_sizeof_probe.c"]
    build = tc.compile_cross(files, sources, prof, compiler, opts.opt, defines=[real_def], warnings=())
    if not build.ok:
        raise RuntimeError(f"{compiler} {prof.key} derleme hatası:\n{build.log}")
    syms = tc.elf_symbols(build.objects["spt_sizeof_probe.c"])
    su = build.stack_usage()
    rows = []
    linkable = gcc is not None and prof.gcc_flags is not None
    base_flash = None
    if linkable:
        bfiles = {"spt_fusion_common.h": files["spt_fusion_common.h"],
                  f"spt_{algos[0]}.h": files[f"spt_{algos[0]}.h"],
                  "spt_fp_base.c": _harness(algos[0], opts.use_mag, True)}
        base = tc.link_cross(bfiles, ["spt_fp_base.c"], prof, compiler, opts.opt, defines=[real_def])
        base_flash = tc.elf_sizes(base.output).flash if base.ok else None
    for a in algos:
        sz = build.sizes(f"spt_{a}.c")
        row = {"algorithm": a, "flash_code": sz.flash, "ram_state": syms[f"spt_sizeof_{a}"].size,
               "flash_with_libm": None, "stack": None, "stack_path": [], "su_update": su.get(f"spt_{a}_update")}
        if linkable and base_flash is not None:
            hfiles = {"spt_fusion_common.h": files["spt_fusion_common.h"], f"spt_{a}.h": files[f"spt_{a}.h"],
                      f"spt_{a}.c": files[f"spt_{a}.c"], "spt_fp_harness.c": _harness(a, opts.use_mag, False)}
            img = tc.link_cross(hfiles, ["spt_fp_harness.c", f"spt_{a}.c"], prof, compiler, opts.opt,
                                defines=[real_def])
            if img.ok:
                row["flash_with_libm"] = tc.elf_sizes(img.output).flash - base_flash
                entries = [f"spt_{a}_step", f"spt_{a}_update"]
                st = tc.stack_analysis(img.output, entries)
                own = tc.stack_analysis(img.output, entries, include=lambda name: name.startswith("spt_"))
                if f"spt_{a}_step" in st:
                    info = st[f"spt_{a}_step"]
                    row["stack"] = info.bytes
                    row["stack_path"] = list(info.path)
                    row["stack_exact"] = info.exact
                    row["stack_own"] = own[f"spt_{a}_step"].bytes if f"spt_{a}_step" in own else None
                    upd = st.get(f"spt_{a}_update")
                    row["stack_update"] = upd.bytes if upd else None
                    row["frame_update"] = upd.frame if upd else None
        rows.append(row)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(rows), encoding="utf-8")
    _FP_CACHE[key] = rows
    return rows


# Derleyici bulunamazsa son çare: Cortex-M4F, arm-none-eabi-gcc 13.3 -Os, float, manyetometreli ölçümü (bayt).
# (flash_code, flash_with_libm, stack) — ölçeklenerek kullanılır, satırda "tahmin" yazar.
_REFERENCE = {"mahony": (800, 2600, 72), "madgwick": (1000, 2700, 64), "kalman": (1100, 3900, 96),
              "ekf": (2800, 5500, 360), "ukf": (4300, 7600, 330), "pf": (2400, 6600, 140)}
_REF_CORE = {"none": 1.6, "sp": 1.0, "dp": 1.0}
_REF_OPT = {"-O0": 2.2, "-Os": 1.0, "-O2": 1.25, "-O3": 1.6}


def _reference_row(opts: CodegenOptions, prof: McuProfile, compiler: str, key: str, why: str) -> FootprintRow:
    f, fl, st = _REFERENCE[key]
    k = _REF_CORE[prof.fpu] * _REF_OPT[opts.opt] * prof.density
    if opts.real == "double" and prof.double_is_soft() and prof.fpu != "none":
        k *= 1.5
    k *= {"gcc": 1.0, "clang": 1.0, "iar": IAR_SIZE_FACTOR}[compiler]
    return FootprintRow(key, compiler, int(round(f * k)), int(round(fl * k)),
                        state_size(key, opts.real, _particles(opts)), int(round(st * prof.density)), False,
                        f"tahmin: {why}; önceden ölçülmüş Cortex-M4F GCC -Os referansından ölçeklendi "
                        f"(×{k:.2f}); RAM durum = sizeof (yapı düzeninden kesin)")


def footprint(opts: CodegenOptions, compilers: tuple[str, ...] | None = None) -> list[FootprintRow]:
    """Algoritma × derleyici başına flash / RAM / yığın. Yerel derleyici varsa ölçülür, yoksa tahmin edilir."""
    check_options(opts)
    prof = profile(opts.mcu)
    compilers = tuple(compilers or (opts.compiler,))
    algos = _ordered(opts)
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
            rows += [_reference_row(opts, prof, comp, a, why) for a in algos]
            continue
        tool = tc.cross_tool(src_comp)
        tool_label = f"{tool.label}" if tool else src_comp
        for d in data:
            stack_note = ""
            if d.get("stack") is not None:
                libm = [f for f in d.get("stack_path", []) if not f.startswith("spt_")]
                stack_note = (f"; yığın: objdump statik analizi, en kötü yol step → update / reset (libm dahil "
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
                    f"tahmin: IAR yok — {tool_label} {opts.opt} ({base_prof.name}) ölçümü × {IAR_SIZE_FACTOR} "
                    f"(yayımlanmış karşılaştırmalarda IAR genelde GCC'den %5–15 küçük)"
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


# ============================================================================ ana makine sürücüsü ve işlem sayacı
_COUNT_PRELUDE = r"""/* İşlem sayacı: SPT_REAL yerine geçen C++ türü (yalnız ana makinede maliyet ölçümü için). */
#include <cmath>
#include <cfloat>
#include <cstdint>
#define SPT_NOPS {{NOPS}}
enum { {{ENUM}} };
extern unsigned long long spt_ops[SPT_NOPS];
struct spt_cnt {
    double v;
    spt_cnt() : v(0.0) {}
    explicit spt_cnt(double x) : v(x) {}
};
inline spt_cnt operator+(spt_cnt a, spt_cnt b) { ++spt_ops[SPT_OP_ADD]; return spt_cnt(a.v + b.v); }
inline spt_cnt operator-(spt_cnt a, spt_cnt b) { ++spt_ops[SPT_OP_ADD]; return spt_cnt(a.v - b.v); }
inline spt_cnt operator*(spt_cnt a, spt_cnt b) { ++spt_ops[SPT_OP_MUL]; return spt_cnt(a.v * b.v); }
inline spt_cnt operator/(spt_cnt a, spt_cnt b) { ++spt_ops[SPT_OP_DIV]; return spt_cnt(a.v / b.v); }
inline spt_cnt operator-(spt_cnt a) { ++spt_ops[SPT_OP_NEG]; return spt_cnt(-a.v); }
inline spt_cnt &operator+=(spt_cnt &a, spt_cnt b) { ++spt_ops[SPT_OP_ADD]; a.v += b.v; return a; }
inline spt_cnt &operator-=(spt_cnt &a, spt_cnt b) { ++spt_ops[SPT_OP_ADD]; a.v -= b.v; return a; }
inline spt_cnt &operator*=(spt_cnt &a, spt_cnt b) { ++spt_ops[SPT_OP_MUL]; a.v *= b.v; return a; }
inline spt_cnt &operator/=(spt_cnt &a, spt_cnt b) { ++spt_ops[SPT_OP_DIV]; a.v /= b.v; return a; }
inline bool operator<(spt_cnt a, spt_cnt b) { ++spt_ops[SPT_OP_CMP]; return a.v < b.v; }
inline bool operator>(spt_cnt a, spt_cnt b) { ++spt_ops[SPT_OP_CMP]; return a.v > b.v; }
inline bool operator<=(spt_cnt a, spt_cnt b) { ++spt_ops[SPT_OP_CMP]; return a.v <= b.v; }
inline bool operator>=(spt_cnt a, spt_cnt b) { ++spt_ops[SPT_OP_CMP]; return a.v >= b.v; }
inline bool operator==(spt_cnt a, spt_cnt b) { ++spt_ops[SPT_OP_CMP]; return a.v == b.v; }
inline bool operator!=(spt_cnt a, spt_cnt b) { ++spt_ops[SPT_OP_CMP]; return a.v != b.v; }
inline spt_cnt spt_cnt_sqrt(spt_cnt a) { ++spt_ops[SPT_OP_SQRT]; return spt_cnt(std::sqrt(a.v)); }
inline spt_cnt spt_cnt_sin(spt_cnt a) { ++spt_ops[SPT_OP_SIN]; return spt_cnt(std::sin(a.v)); }
inline spt_cnt spt_cnt_cos(spt_cnt a) { ++spt_ops[SPT_OP_COS]; return spt_cnt(std::cos(a.v)); }
inline spt_cnt spt_cnt_atan2(spt_cnt a, spt_cnt b) { ++spt_ops[SPT_OP_ATAN2]; return spt_cnt(std::atan2(a.v, b.v)); }
inline spt_cnt spt_cnt_asin(spt_cnt a) { ++spt_ops[SPT_OP_ASIN]; return spt_cnt(std::asin(a.v)); }
inline spt_cnt spt_cnt_exp(spt_cnt a) { ++spt_ops[SPT_OP_EXP]; return spt_cnt(std::exp(a.v)); }
inline spt_cnt spt_cnt_log(spt_cnt a) { ++spt_ops[SPT_OP_LOG]; return spt_cnt(std::log(a.v)); }
inline spt_cnt spt_cnt_fabs(spt_cnt a) { ++spt_ops[SPT_OP_ABS]; return spt_cnt(std::fabs(a.v)); }
inline spt_cnt spt_cnt_fmod(spt_cnt a, spt_cnt b) { ++spt_ops[SPT_OP_FMOD]; return spt_cnt(std::fmod(a.v, b.v)); }
inline spt_cnt spt_cnt_hypot(spt_cnt a, spt_cnt b)   /* yalnız başlatmada: √(a² + b²) kadar sayılır */
{ ++spt_ops[SPT_OP_SQRT]; spt_ops[SPT_OP_MUL] += 2; ++spt_ops[SPT_OP_ADD]; return spt_cnt(std::hypot(a.v, b.v)); }
inline spt_cnt spt_cnt_from_int(long long i) { ++spt_ops[SPT_OP_CONV]; return spt_cnt((double)i); }
inline spt_cnt spt_cnt_rng(uint32_t u) { ++spt_ops[SPT_OP_RNG]; return spt_cnt((double)u); }
typedef spt_cnt SPT_REAL;
#define SPT_CUSTOM_REAL 1
#define SPT_C(x) spt_cnt(x)
#define SPT_REAL_MAX spt_cnt(DBL_MAX)
#define SPT_EPS spt_cnt(DBL_EPSILON)
#define SPT_SQRT(x) spt_cnt_sqrt(x)
#define SPT_SIN(x) spt_cnt_sin(x)
#define SPT_COS(x) spt_cnt_cos(x)
#define SPT_ATAN2(y, x) spt_cnt_atan2((y), (x))
#define SPT_ASIN(x) spt_cnt_asin(x)
#define SPT_EXP(x) spt_cnt_exp(x)
#define SPT_LOG(x) spt_cnt_log(x)
#define SPT_FABS(x) spt_cnt_fabs(x)
#define SPT_FMOD(x, y) spt_cnt_fmod((x), (y))
#define SPT_HYPOT(x, y) spt_cnt_hypot((x), (y))
#define SPT_FROM_INT(i) spt_cnt_from_int(i)
#define SPT_U32_TO_REAL(u) spt_cnt_rng(u)
"""

_DRIVER = r"""/* Ana makine doğrulama / maliyet sürücüsü (MCU'ya eklenmez). Giriş: [N, dt, mag] + N×9 double (jiro, ivme,
   mag); çıkış: N×4 kuaterniyon (NaN = çıktı yok) + 3 sapma (+ sayaç modunda N×SPT_NOPS işlem sayısı). */
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
static long spt_drv_n;
static int spt_drv_mag;
static double spt_drv_dt, spt_drv_nan;
static double spt_drv_bias[3];

static void spt_drv_load(long k, SPT_REAL g[3], SPT_REAL a[3], SPT_REAL m[3])
{
    const double *row = spt_drv_in + 9 * k;
    int i;
    for (i = 0; i < 3; ++i) {
        g[i] = SPT_DRV_REAL(row[i]);
        a[i] = SPT_DRV_REAL(row[3 + i]);
        m[i] = SPT_DRV_REAL(row[6 + i]);
    }
}

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

static void spt_drv_quat(long k, int ok, const SPT_REAL q[4])
{
    int i;
    for (i = 0; i < 4; ++i) {
        spt_drv_out[4 * k + i] = ok ? SPT_DRV_DBL(q[i]) : spt_drv_nan;
    }
}
{{RUNNERS}}
int main(int argc, char **argv)
{
    FILE *f;
    double hdr[3];
    long total;
    if (argc < 4) {
        return 2;
    }
    spt_drv_nan = strtod("nan", NULL);
    f = fopen(argv[2], "rb");
    if (!f || fread(hdr, sizeof(double), 3, f) != 3) {
        return 3;
    }
    spt_drv_n = (long)hdr[0];
    spt_drv_dt = hdr[1];
    spt_drv_mag = hdr[2] != 0.0;
    spt_drv_in = (double *)malloc(sizeof(double) * 9 * (size_t)spt_drv_n);
    spt_drv_out = (double *)malloc(sizeof(double) * 4 * (size_t)spt_drv_n);
    spt_drv_ops = (double *)malloc(sizeof(double) * (SPT_DRV_NOPS + 1) * (size_t)spt_drv_n);
    if (!spt_drv_in || !spt_drv_out || !spt_drv_ops
        || fread(spt_drv_in, sizeof(double), 9 * (size_t)spt_drv_n, f) != 9 * (size_t)spt_drv_n) {
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
    total = 4 * spt_drv_n;
    fwrite(spt_drv_out, sizeof(double), (size_t)total, f);
    fwrite(spt_drv_bias, sizeof(double), 3, f);
    if (SPT_DRV_NOPS > 0) {
        fwrite(spt_drv_ops, sizeof(double), (size_t)(SPT_DRV_NOPS * spt_drv_n), f);
    }
    fclose(f);
    return 0;
}
"""

_RUNNER = r"""
static void spt_drv_run_{{A}}(void)
{
    static spt_{{A}}_t st;
    SPT_REAL g[3], a[3], m[3], q[4], b[3];
    unsigned long long before[SPT_DRV_NOPS + 1];
    long k;
    int i, ok;
    spt_{{A}}_defaults(&st);
    for (k = 0; k < spt_drv_n; ++k) {
        spt_drv_load(k, g, a, m);
        spt_drv_snap(before);
        ok = spt_{{A}}_step(&st, g, a, spt_drv_mag ? m : NULL, SPT_DRV_REAL(spt_drv_dt));
        spt_drv_count(k, before);
        if (ok) {
            spt_{{A}}_quat(&st, q);
        }
        spt_drv_quat(k, ok, q);
    }
    {{BIAS}}
    for (i = 0; i < 3; ++i) {
        spt_drv_bias[i] = SPT_DRV_DBL(b[i]);
    }
}
"""


def _driver_source(algos: list[str]) -> str:
    runners = []
    dispatch = []
    for a in algos:
        g = _BIAS_GETTER[a]
        bias = f"spt_{a}_{g}(&st, b);" if g else "b[0] = SPT_C(0.0);\n    b[1] = SPT_C(0.0);\n    b[2] = SPT_C(0.0);"
        runners.append(_fill(_RUNNER, A=a, BIAS=bias))
        dispatch.append(f'    if (strcmp(argv[1], "{a}") == 0) {{\n        spt_drv_run_{a}();\n    }} else')
    return _fill(_DRIVER, INCLUDES="\n".join(f'#include "spt_{a}.h"' for a in algos), RUNNERS="".join(runners),
                 DISPATCH="\n".join(dispatch))


def _count_prelude() -> str:
    enum = ", ".join(f"SPT_OP_{op.upper()}" for op in OPS)
    return _fill(_COUNT_PRELUDE, NOPS=str(len(OPS)), ENUM=enum)


def host_program(opts: CodegenOptions, *, real: str | None = None, count_ops: bool = False) -> tc.Build:
    """Üretilen C'yi ana makinede derler (doğrulama sürücüsüyle). count_ops: C++ olarak, SPT_REAL = sayaç türü."""
    check_options(opts)
    algos = _ordered(opts)
    real = real or opts.real
    files = {f.name: f.text for f in _sources(opts)}
    files["spt_host_driver.c"] = _driver_source(algos)
    sources = [f"spt_{a}.c" for a in algos] + ["spt_host_driver.c"]
    if count_ops:
        files["spt_count_prelude.hpp"] = _count_prelude()
        return tc.host_build(files, sources, cxx=True, extra=["-std=c++17", "-include", "spt_count_prelude.hpp"],
                             opt="-O2", output="spt_count")
    return tc.host_build(files, sources, defines=[f"SPT_USE_DOUBLE={1 if real == 'double' else 0}"],
                         extra=["-std=c99", "-Wall", "-Wextra", "-pedantic"], opt="-O2", output=f"spt_host_{real}")


def run_generated(opts: CodegenOptions, algorithm: str, gyro: np.ndarray, acc: np.ndarray,
                  mag: np.ndarray | None, dt: float, *, real: str | None = None,
                  count_ops: bool = False) -> HostRun:
    """Üretilen C algoritmasını ana makinede verilen veriyle çalıştırır (step() akışı, Python run() ile aynı kurallar)."""
    if algorithm not in _ordered(opts):
        raise ValueError(f"{algorithm!r} seçili algoritmalar arasında değil")
    build = host_program(opts, real=real, count_ops=count_ops)
    if not build.ok:
        raise RuntimeError("Ana makine derlemesi başarısız:\n" + build.log)
    gyro = np.asarray(gyro, dtype=np.float64)
    n = len(gyro)
    data = np.empty((n, 9))
    data[:, 0:3] = gyro
    data[:, 3:6] = acc
    data[:, 6:9] = 0.0 if mag is None else mag
    blob = np.r_[float(n), float(dt), 0.0 if mag is None else 1.0, data.ravel()].astype("<f8").tobytes()
    tag = hashlib.sha256(blob).hexdigest()[:16]
    runs = build.workdir / "runs"
    runs.mkdir(exist_ok=True)
    inp, out = runs / f"in_{tag}.bin", runs / f"out_{algorithm}_{tag}.bin"
    if not out.exists():
        inp.write_bytes(blob)
        res = tc.run_program(build, [algorithm, str(inp), str(out)], timeout=600)
        if res.returncode != 0 or not out.exists():
            raise RuntimeError(f"Sürücü hatası ({res.returncode}): {res.stderr}")
        inp.unlink(missing_ok=True)
    raw = np.fromfile(out, dtype="<f8")
    q = raw[:4 * n].reshape(n, 4)
    bias = raw[4 * n:4 * n + 3].copy()
    ops = raw[4 * n + 3:].reshape(n, len(OPS)) if count_ops else None
    return HostRun(q, bias, ops)


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


def cost(opts: CodegenOptions) -> list[CostRow]:
    """Adım başına işlem sayısı (ölçülür) → çevrim / süre / CPU yükü (MCU tablosundan tahmin)."""
    check_options(opts)
    prof = profile(opts.mcu)
    n = COST_SAMPLES
    sim = fu.simulate_imu(duration=n / opts.fs_hz, fs=opts.fs_hz, seed=COST_SEED)
    dt = 1.0 / opts.fs_hz
    weights = _op_weights(prof, opts.real)
    rows = []
    variants = (True, False) if opts.use_mag else (False,)
    for a in _ordered(opts):
        for use_mag in variants:
            run = run_generated(opts, a, sim.gyro, sim.acc, sim.mag if use_mag else None, dt, count_ops=True)
            ops = run.ops[1:]                     # ilk adım (başlatma dahil) hariç
            mean = ops.mean(axis=0)
            cyc = ops @ weights
            cycles = float(cyc.mean())
            us = cycles / opts.clock_mhz
            us_max = float(cyc.max()) / opts.clock_mhz
            rows.append(CostRow(
                a, {op: float(v) for op, v in zip(OPS, mean)}, cycles, us, us * opts.fs_hz / 1e4,
                1e6 / us if us > 0 else math.inf,
                f"işlem sayısı ölçüldü (C++ sayaç, {len(ops)} adım, {'MARG' if use_mag else 'IMU'}), çevrim çekirdek "
                f"tablosundan tahmin ({prof.name}, {opts.real}{', double yazılım kayan nokta' if opts.real == 'double' and prof.double_is_soft() else ''})",
                use_mag, float(cyc.max()), us_max))
    return rows


# ============================================================================ kullanım kılavuzu
def _fmt_int(v: int | float | None) -> str:
    if v is None:
        return "—"
    return f"{int(round(v)):,}".replace(",", " ") if math.isfinite(v) else "∞"


def _footprint_table(opts: CodegenOptions, rows: list[FootprintRow]) -> str:
    lines = ["| Algoritma | Derleyici | Flash (kod) | Flash (libm dahil) | RAM (durum) | Yığın (libm dahil / bu kod) "
             f"| Flash % ({opts.flash_kb:g} kB) | RAM % ({opts.ram_kb:g} kB) | Ölçüm |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        total_flash = r.flash_with_libm if r.flash_with_libm is not None else r.flash_code
        ram = r.ram_state + (r.stack or 0)
        lines.append(f"| {_TITLES[r.algorithm]} | {COMPILERS[r.compiler]} | {_fmt_int(r.flash_code)} B | "
                     f"{_fmt_int(r.flash_with_libm)}{' B' if r.flash_with_libm is not None else ''} | "
                     f"{_fmt_int(r.ram_state)} B | {_fmt_int(r.stack)} / {_fmt_int(r.stack_own)} B | "
                     f"{100 * total_flash / (opts.flash_kb * 1024):.1f} | {100 * ram / (opts.ram_kb * 1024):.1f} | "
                     f"{'ölçüldü' if r.measured else 'tahmin'} |")
    methods = sorted({re.sub(r"; yığın: .*$", "", r.method) for r in rows})
    stack = next((re.search(r"yığın: (objdump statik analizi, en kötü yol [^(]*)", r.method) for r in rows
                  if "yığın:" in r.method), None)
    return ("\n".join(lines) + "\n\nRAM % = durum + en kötü yığın (libm dahil). Libm dahil yığın muhafazakârdır: "
            "newlib sinf / cosf'in büyük açı indirgemesi (__kernel_rem_pio2f, ≈ 450 B yerel dizi) statik olarak en "
            "derin yoldur ama bu koddaki küçük açılarda çalışmaz; \"bu kod\" sütunu libm fonksiyonlarını 0 baytlık yaprak "
            "sayar.\n\n" + "\n".join(f"- {m}" for m in methods)
            + (f"\n- Yığın: {stack.group(1).strip()}." if stack else ""))


def _cost_table(opts: CodegenOptions, rows: list[CostRow]) -> str:
    lines = [f"| Algoritma | Mod | Çevrim / adım (ort.) | En kötü adım | µs / adım @ {opts.clock_mhz:g} MHz | "
             f"CPU yükü @ {opts.fs_hz:g} Hz | En yüksek hız |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {_TITLES[r.algorithm]} | {'MARG' if r.mag else 'IMU'} | {_fmt_int(r.cycles)} | "
                     f"{_fmt_int(r.cycles_max)} | {r.us_per_update:.1f} | %{r.cpu_load_percent:.2f} | "
                     f"{_fmt_int(r.max_rate_hz)} Hz |")
    ops = ["| Algoritma | Mod | " + " | ".join(OPS) + " |", "|---|---|" + "---:|" * len(OPS)]
    for r in rows:
        ops.append(f"| {_TITLES[r.algorithm]} | {'MARG' if r.mag else 'IMU'} | "
                   + " | ".join(f"{r.ops[o]:.1f}" for o in OPS) + " |")
    return "\n".join(lines) + "\n\n" + (rows[0].method if rows else "") + \
        "\n\nAdım başına ortalama işlem sayıları (ölçülen):\n\n" + "\n".join(ops)


def usage_markdown(opts: CodegenOptions, footprint_rows: list[FootprintRow] | None = None,
                   cost_rows: list[CostRow] | None = None) -> str:
    """Türkçe kullanım kılavuzu (KULLANIM.md)."""
    check_options(opts)
    p, P = opts.prefix.lower(), opts.prefix.upper()
    prof = profile(opts.mcu)
    algos = _ordered(opts)
    first = algos[0]
    file_rows = [f"| `{p}_fusion_common.h` | Sayı türü (`{P}_REAL`), matematik makroları, kuaterniyon yardımcıları "
                 f"(static inline) |"]
    for a in algos:
        file_rows.append(f"| `{p}_{a}.h` / `{p}_{a}.c` | {_TITLES[a]}: durum yapısı `{p}_{a}_t`, API |")
    file_rows.append("| `ornek_kullanim.c` | Sabit hızda çağıran örnek görev (sensör okuma / bekleme kullanıcıya ait) |")
    params = []
    for a in algos:
        vals = _resolved_params(a, opts.params.get(a))
        for prm in fu.ALGORITHMS[a].params:
            macro = f"{P}_{a.upper()}_{prm.key.upper()}"
            runtime = "derleme anı (dizi boyutu)" if (a, prm.key) == ("pf", "particles") else \
                f"`s.{prm.key}`" + (" + `reset()`" if (a, prm.key) == ("pf", "seed") else "")
            params.append(f"| {_TITLES[a]} | `{macro}` | {prm.label} | {vals[prm.key]:g} | {prm.default:g} | "
                          f"{runtime} |")
    runtime_example = "örnek yok: seçili algoritmanın çalışma anı parametresi yok"
    for a in algos:  # yalnız seçili algoritmanın gerçek alanı, seçilen sayı türünde
        prm = next((x for x in fu.ALGORITHMS[a].params if (a, x.key) != ("pf", "particles")), None)
        if prm is not None:
            v = _resolved_params(a, opts.params.get(a))[prm.key]
            lit = f"{float(v)!r}" + ("f" if opts.real == "float" else "")
            runtime_example = f"`{p}_{a}_t s; … s.{prm.key} = {lit};`"
            break
    diag = []
    if "pf" in algos:
        diag.append("parçacık filtresinde `s.ess` (etkin parçacık sayısı) ve `s.resamples`")
    if "ukf" in algos:
        diag.append("UKF'de `s.fallbacks` (özdeğer yedeği sayısı)")
    text = "; ".join(diag)
    diag_line = f"- {text[:1].upper()}{text[1:]} tanılama içindir.\n" if diag else ""
    bias_lines = []
    for a in algos:
        g = _BIAS_GETTER[a]
        if g == "gyro_bias":
            bias_lines.append(f"- `{p}_{a}_gyro_bias(&s, b)`: gövde jiroskop sapması kestirimi (rad/s).")
        elif g == "euler_bias":
            bias_lines.append(f"- `{p}_{a}_euler_bias(&s, b)`: **Euler hızı** sapması (rad/s) — gövde jiroskop "
                              "sapması değildir (yatay çevresinde ona yakındır, büyük eğimde değil).")
        else:
            bias_lines.append(f"- {_TITLES[a]}: sapma kestirmez.")
    fpu = {"none": "FPU yok: `-mfloat-abi=soft` (tüm kayan nokta yazılımda; float bile onlarca çevrim)",
           "sp": "tek hassasiyet FPU: `float` önerilir; `double` yazılım kayan noktadır (10–100× yavaş)",
           "dp": "çift hassasiyet FPU: `double` de donanımda (yine de `float` daha hızlı ve küçük)"}[prof.fpu]
    flags = " ".join(prof.gcc_flags) if prof.gcc_flags else "(bu çekirdek için derleyici bayrakları üreticinin SDK'sına bakın)"
    real_note = ""
    if opts.real == "double" and prof.double_is_soft():
        real_note = (f"\n> **Uyarı:** `double` seçildi ama {prof.name} çekirdeğinde çift hassasiyet donanımı yok; "
                     "tüm hesap yazılım kayan noktayla yapılır. Mümkünse `float` kullanın.\n")
    fp_md = _footprint_table(opts, footprint_rows) if footprint_rows else \
        "_Bu dosya ölçümsüz üretildi; ayak izi tablosu için `footprint()` sonuçlarıyla yeniden üretin._"
    cost_md = _cost_table(opts, cost_rows) if cost_rows else \
        "_Bu dosya ölçümsüz üretildi; maliyet tablosu için `cost()` sonuçlarıyla yeniden üretin._"
    text = f"""# Yönelim füzyonu — C kodu kullanım kılavuzu

Signal Processing Toolbox tarafından üretildi. Python referansı: `spt/dsp/fusion.py` (aynı denklemler, aynı
parametreler, aynı başlatma ve geçersiz örnek kuralları).

**Seçimler:** algoritmalar {", ".join(_TITLES[a] for a in algos)} · sayı türü `{opts.real}` · MCU {prof.name}
({prof.examples}) · derleyici {COMPILERS[opts.compiler]} `{opts.opt}` · saat {opts.clock_mhz:g} MHz ·
örnekleme {opts.fs_hz:g} Hz · manyetometre {"var" if opts.use_mag else "yok (`" + P + "_USE_MAG = 0`)"}
{real_note}
## Dosyalar

| Dosya | İçerik |
|---|---|
{chr(10).join(file_rows)}

Tüm durum çağıranın yapısındadır: `malloc` yok, global değişken yok, fonksiyonlar yeniden girişlidir. Yalnızca
kullandığınız algoritmanın `.c` dosyasını ve ortak başlığı projeye eklemeniz yeterlidir.

## Hızlı başlangıç

```c
#include "{p}_{first}.h"

static {p}_{first}_t f;                  /* durum (statik: yığında yer kaplamaz) */

void baslat(void) {{ {p}_{first}_defaults(&f); }}

/* zamanlayıcı kesmesinden ya da RTOS görevinden {opts.fs_hz:g} Hz'de */
void fuzyon_adimi(const {P}_REAL gyro[3], const {P}_REAL acc[3], const {P}_REAL mag[3] /* ya da NULL */)
{{
    {P}_REAL roll, pitch, yaw, q[4];
    if ({p}_{first}_step(&f, gyro, acc, mag, {P}_C({1.0 / opts.fs_hz!r}))) {{
        {p}_{first}_euler(&f, &roll, &pitch, &yaw);   /* rad */
        {p}_{first}_quat(&f, q);                      /* [w, x, y, z], gövde → dünya */
    }}
}}
```

## API (her algoritma için aynı)

| Fonksiyon | Açıklama |
|---|---|
| `{p}_X_defaults(&s)` | Parametreleri derleme anı varsayılanlarına getirir, durumu "başlatılmamış" yapar |
| `{p}_X_init(&s, acc, mag)` | `defaults` + `reset`; acc geçersizse 0 döner |
| `{p}_X_reset(&s, acc, mag)` | Durumu bu örnekten yeniden başlatır (parametreler korunur) |
| `{p}_X_update(&s, gyro, acc, mag, dt)` | Bir örnek işler (durum başlatılmış ve gyro sonlu olmalı) |
| `{p}_X_step(&s, gyro, acc, mag, dt)` | **Önerilen.** Geçersiz jiroskopta durumu unutur, ilk geçerli ivmede başlatır, sonra `update`; geçerli çıktı varsa 1 |
| `{p}_X_quat(&s, q)` / `{p}_X_euler(&s, &r, &p, &y)` | Sonuç: kuaterniyon / ZYX Euler (rad) |

`X` ∈ {{{", ".join(algos)}}}. Sapma okuma:

{chr(10).join(bias_lines)}

## Projeye ekleme

### STM32CubeIDE (GCC)
1. `{p}_fusion_common.h`, `{p}_<algoritma>.h/.c` dosyalarını `Core/Inc` ve `Core/Src` altına kopyalayın (ya da
   ayrı bir kaynak klasörü ekleyip *Properties → C/C++ General → Paths and Symbols* ile include yoluna ekleyin).
2. *Properties → C/C++ Build → Settings → MCU Settings*: Floating-point unit ve ABI. Bu çekirdek için bayraklar:
   `{flags}`.
3. Matematik kütüphanesi: *MCU GCC Linker → Libraries* altına `m` ekleyin (`-lm`). newlib-nano varsayılandır.
4. İsteğe bağlı: *Preprocessor* altında `{P}_USE_DOUBLE=0/1`, `{P}_USE_MAG=0/1`, parametre makroları.
5. **`-ffast-math` kullanmayın** (geçersiz örnek denetimleri NaN karşılaştırmalarına dayanır).

### Keil MDK (Arm Compiler 6)
1. Dosyaları bir *Group*'a ekleyin; include yolunu *Options for Target → C/C++ (AC6) → Include Paths*'e yazın.
2. *Target* sekmesi: *Floating Point Hardware* = "Single Precision" (M4F / M33F) ya da "Double Precision" (M7).
3. Dil: *C99* ya da üstü. *Optimization*: `-Os balanced` ya da `-O2`. `-ffast-math` (*Fast math*) kapalı kalsın.
4. AC6 clang tabanlıdır; bu kodun boyutu yerelde clang ile ölçülmüştür ("Clang / Arm Compiler 6" satırları).

### IAR EWARM
1. Dosyaları projeye ekleyin, *Options → C/C++ Compiler → Preprocessor* ile include yolunu verin.
2. *General Options → Target → FPU*: VFPv4 single (M4F), VFPv5 double (M7), VFPv5 single (M33F).
3. *Language*: C99 (ya da Standard C), *Optimizations*: High / Size ya da Speed. Kaynak UTF-8'dir (yalnız
   yorumlarda Türkçe karakter var).
4. IAR bu makinede olmadığından IAR boyutları **tahmindir** (GCC ölçümü × {IAR_SIZE_FACTOR}).

### ESP32 (ESP-IDF) / RISC-V
Dosyaları bir bileşenin `srcs` listesine ekleyin. ESP32'nin FPU'su tek hassasiyettir (`float` kullanın; bölme ve
karekök yazılım yinelemesidir). FPU'suz RISC-V (RV32IMAC) çekirdeklerde tüm kayan nokta yazılımdadır.

## Sabit hızda çağırma ve dt

- Algoritmalar **sabit örnekleme hızında** çağrılmak üzere ayarlanmıştır (parametrelerdeki ivme / başlık
  gürültüleri örnek başınadır). Bir donanım zamanlayıcısını {opts.fs_hz:g} Hz'e kurun; kesmede yalnız bir bayrak
  kurup hesabı ana döngüde ya da yüksek öncelikli bir RTOS görevinde (`vTaskDelayUntil`) yapın.
- `dt` saniyedir. Sabit hızda `dt = 1/fs` verin. Hız oynuyorsa dt'yi ölçün: Cortex-M3/M4/M7/M33'te
  `DWT->CYCCNT` (bkz. `ornek_kullanim.c`, `ORNEK_DT_OLC 1`): `dt = (CYCCNT_şimdi − CYCCNT_önceki) / SystemCoreClock`.
- Sensör FIFO'sundan toplu okuyorsanız örnekleri sırayla, her biri için kendi dt'siyle işleyin.
- Hesap süresini ölçmek için `step()` çağrısını iki `DWT->CYCCNT` okumasıyla çevreleyin; aşağıdaki maliyet tablosu
  çekirdek tablosundan **tahmindir**, gerçek değer bellek bekleme durumlarına (flash wait state, önbellek) bağlıdır.

## Sensör eksen hizalaması

Gövde çerçevesi **sağ el kuralı, z yukarı**: kart yatay ve durağanken ivmeölçer yaklaşık `[0, 0, +g]` okumalı;
burun +x yönünde, +y sola. Jiroskop aynı eksenlerde ve sağ el kuralıyla (z etrafında saat yönünün tersi pozitif),
**rad/s** olmalı (°/s ise `× π/180`). Manyetometre de aynı eksenlere döndürülmelidir — birçok 9 eksenli modülde
manyetometre eksenleri farklıdır. Örnek (MPU-9250: AK8963 manyetometresi x↔y değişik, z ters):

```c
/* ham → gövde (z yukarı, sağ el) */
gyro[0] = gx_dps * {P}_D2R;  gyro[1] = gy_dps * {P}_D2R;  gyro[2] = gz_dps * {P}_D2R;
acc[0]  = ax;                acc[1]  = ay;                acc[2]  = az;
mag[0]  = my_ut;             mag[1]  = mx_ut;             mag[2]  = -mz_ut;   /* AK8963 eksenleri */
```

Doğrulama: kartı yatay tutun (roll ≈ pitch ≈ 0), burnu yukarı kaldırın (pitch negatif: ZYX'te +y etrafında
pozitif dönme burnu aşağı indirir), sağ kanadı aşağı yatırın (roll pozitif), saat yönünün tersine çevirin (yaw artar).
İvmeölçer ve manyetometrenin birimi önemsizdir (yalnız yön kullanılır), jiroskop birimi önemlidir.

## Başlatma, geçersiz örnekler, kayıp

- `step()` Python `Estimator.run` ile aynı kuralları uygular: sonlu olmayan jiroskop örneği **kayıp** sayılır
  (durum unutulur, 0 döner); sonra ilk **geçerli** (sonlu, sıfırdan farklı) ivme örneğinde ivmeölçerden roll /
  pitch, geçerli manyetometre varsa eğim düzeltmeli yaw ile yeniden başlatılır (yoksa yaw = 0).
- Sonlu olmayan ya da sıfır ivme / manyetometre örneği yalnız o adımdaki ilgili güncellemeyi atlar.
- `dt` > 0 ve sonlu olmalıdır. Ölçülen dt kullanıyorsanız (FIFO zaman damgası, DWT) aynı zaman damgası (dt = 0),
  geriye giden ya da taşan sayaç (dt < 0) ve NaN / sonsuz dt **atlanır**: `step()` 0 döner, durum değişmez.
  (`update()` dt'yi denetlemez; doğrudan kullanıyorsanız denetim sizdedir.)
- `update()` doğrudan kullanılıyorsa başlatmayı (`init` / `reset`) ve jiroskop denetimini çağıran yapar.
- İlk örnekler sırasında kartın durağan olması (ya da en azından doğrusal ivmenin küçük olması) yakınsamayı
  hızlandırır. Manyetometre yoksa (`mag = NULL`) yaw gözlemlenemez ve yavaşça kayar.

## Kalibrasyon

- **Jiroskop sapması:** Mahony / EKF / UKF / parçacık filtresi sapmayı çalışırken kestirir; yine de açılışta
  1–2 s durağan ortalama alıp çıkarmak yakınsamayı hızlandırır. Madgwick sapma kestirmez (önceden çıkarın).
- **Manyetometre sert demir (hard iron):** kartı tüm yönlerde döndürüp her eksenin (min + max)/2 değerini
  çıkarın (ya da küre uydurma). **Yumuşak demir (soft iron):** elipsoid uydurmayla 3×3 düzeltme matrisi. Kalibre
  edilmemiş manyetometre başlıkta sabit / yönle değişen hataya yol açar; EKF / UKF / parçacık filtresinde bozulma
  roll / pitch'e sızmaz.
- **İvmeölçer:** eksen başına kazanç / ofset (6 yüz yöntemi) eğim doğruluğunu artırır.

## Parametreler

Varsayılanlar derleme anında `#define` ile gelir (`#ifndef` korumalı: komut satırından ezilebilir) ve `defaults()`
ile yapıya kopyalanır; çalışma anında yapı alanından değiştirilebilir ({runtime_example}).

| Algoritma | Makro | Anlam | Bu üretim | Python varsayılanı | Çalışma anında |
|---|---|---|---:|---:|---|
{chr(10).join(params)}

Not: ivmeölçer / başlık gürültüleri örnek başınadır; örnekleme hızını çok artırırsanız (ör. 1 kHz) σ ivme / σ ölçüm
değerlerini büyütün (doğrusal ivme zamanla ilişkilidir).

## Sonuçları okuma

- `quat()`: q = [w, x, y, z], gövde → dünya (v_dünya = R(q)·v_gövde). Kontrol döngüleri için tercih edin
  (tekillik yok).
- `euler()`: ZYX (aerospace) roll / pitch / yaw, radyan; derece için `× 57.2958`. pitch ±90° yakınında roll ve yaw
  tanımsızlaşır (gimbal kilidi).
{diag_line}
## İş parçacığı güvenliği

Fonksiyonlar yeniden girişlidir ve global durum tutmaz. Aynı durum yapısını iki bağlamdan (ör. kesme + görev)
kullanacaksanız erişimi kendiniz koruyun (kesmeyi kısa süre kapatma, mutex) ya da sonucu bir çift tampon / kuyrukla
aktarın. Farklı yapılar birbirinden bağımsızdır.

## Sayı türü ve FPU

{prof.name}: {fpu}. `{P}_USE_DOUBLE` ile seçilir. Sabitler `{P}_C()` makrosuyla doğru türde yazılmıştır (float
modunda istemsiz double yükseltmesi yok; `-Wdouble-promotion` temiz). Matematik fonksiyonları `{P}_SQRT`,
`{P}_SIN`, … makrolarıyla değiştirilebilir (ör. CMSIS-DSP `arm_sin_f32`); doğruluk sorumluluğu kullanıcıdadır.

## Ölçülen ayak izi

{fp_md}

## Hesap maliyeti

{cost_md}

Maliyet modeli: çevrim = Σ temel işlem × (çekirdek çevrimi + bellek / dal ek yükü) + Σ kütüphane çağrısı ×
(fonksiyon çevrimi + çağrı ek yükü). İşlem sayıları **ölçülmüştür** (üretilen C, ana makinede C++ sayaç türüyle
derlenip simüle IMU verisiyle çalıştırılır); çekirdek çevrim tabloları **tahmindir** (`spt/codegen/mcu.py`).
CPU yükü = µs/adım × fs / 10⁴ (%).

## Python referansından farklar

- Yuvarlama düzeyinde: bölmeler tersle çarpma; simetrik kovaryansların yalnız üst üçgeni; başlıkta sin/cos
  özdeşlikleri ve açı farkı için tek atan2; EKF Joseph biçimi kapalı formda. Çift hassasiyette Python ile
  en büyük açı farkı < 10⁻⁶ rad olarak doğrulanmıştır.
- Parçacık filtresi: rastgele sayı üreteci farklı (xorshift32 + Marsaglia kutupsal) → sonuçlar istatistiksel olarak
  eşdeğer, örnek örnek aynı değil; yeniden örneklemede parçacık sırası farklı (yerinde kopyalama).
- UKF: Cholesky başarısız olursa Python'daki `eigh` yedeği Jacobi özdeğer ayrışımıyla yapılır (aynı sigma kümesi;
  çok katlı özdeğerlerde özvektör seçimi farklı olabilir ama eşit derecede geçerlidir).
- Tekil yenilik kovaryansında güncelleme atlanır (Python hata fırlatırdı; pratikte r > 0 olduğu için oluşmaz).
- `float` modunda sonuçlar Python'dan ~10⁻⁶ göreli farklıdır; doğruluk (RMSE) farkı testlerde < 0.1°'dir.
"""
    return text
