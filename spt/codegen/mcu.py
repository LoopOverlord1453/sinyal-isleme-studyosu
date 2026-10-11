"""Mikrodenetleyici profilleri: derleyici bayrakları ve işlem başına çevrim (cycle) tabloları.

Çevrim tablolarındaki TÜM sayılar TAHMİNDİR (bu makinede çevrim doğru bir simülatör / kart yok). Kaynaklar ve
varsayımlar:
  - FPU'lu çekirdeklerde temel işlemler çekirdek teknik başvuru kılavuzlarından (TRM) gelir: Cortex-M4F / M33F
    FPv4 / FPv5-SP: VADD / VSUB / VMUL 1, VDIV.F32 / VSQRT.F32 14 çevrim, VCMP + VMRS + dal ≈ 3, VCVT 1.
    Cortex-M7 FPv5-DP: tek hassasiyet aynı, VDIV.F64 / VSQRT.F64 ≈ 30, VMUL.F64 2 çevrim (çift yayınlı boru hattı).
  - GCC varsayılanı (-fmath-errno) sqrtf'i VSQRT + NaN denetimi + (nadiren) kütüphane çağrısı olarak üretir: +3.
  - FPU'su olmayan çekirdeklerde (M0+, M3, RV32IMAC) ve tek hassasiyet FPU'lu çekirdeklerde `double` işlemleri
    YAZILIM kayan noktadır (libgcc __aeabi_dadd …): onlarca – yüzlerce çevrim. Çift hassasiyet FPU yalnız M7'de var.
    M0+ (ARMv6-M) libgcc'nin genel C soft-fp kodunu kullanır ve 32×32→64 çarpması yoktur: M3'ten 2 – 3 kat yavaştır.
  - libm (newlib) aşkın fonksiyonları: sinf / cosf / atan2f / asinf / expf / logf için çekirdeğin temel işlem
    maliyetinden (polinom + aralık indirgeme ≈ 15 – 30 işlem) türetilmiş yaklaşık değerler. newlib, yalnız çift
    hassasiyet FPU'lu ARM'da (M7) "yeni" (double tabanlı) expf / logf kodunu kullanır; diğerlerinde eski float kodu.
  - "mem": temel aritmetik işlem başına ortalama yükleme / saklama / dal / döngü ek yükü (çevrim). Maliyet modeli
    (fusion_c.cost): çevrim = Σ temel_işlem·(c_işlem + mem) + Σ kütüphane_çağrısı·(c_fonksiyon + call).
  - Xtensa (ESP32) ve RISC-V profilleri için bu makinede derleyici yoktur: kod boyutu ölçülmüş bir ARM profilinden
    kod yoğunluğu çarpanıyla TAHMİN edilir (bkz. `density`, `estimate_base`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Sayılan işlemler (C++ sayaç sınıfıyla ölçülür) → tablo anahtarı (float / double)
OPS = ("add", "mul", "div", "sqrt", "cmp", "conv", "neg", "abs", "fmod", "sin", "cos", "atan2", "asin", "exp",
       "log", "rng")
BASIC_OPS = frozenset(("add", "mul", "div", "sqrt", "cmp", "conv", "neg", "abs"))   # + "mem" ek yükü
LIBRARY_OPS = frozenset(("fmod", "sin", "cos", "atan2", "asin", "exp", "log"))      # + "call" ek yükü
_FLOAT_KEYS = {"add": "fadd", "mul": "fmul", "div": "fdiv", "sqrt": "fsqrt", "cmp": "fcmp", "conv": "fconv",
               "neg": "fneg", "abs": "fabs", "fmod": "fmodf", "sin": "sinf", "cos": "cosf", "atan2": "atan2f",
               "asin": "asinf", "exp": "expf", "log": "logf", "rng": "rng"}
_DOUBLE_KEYS = {"add": "dadd", "mul": "dmul", "div": "ddiv", "sqrt": "dsqrt", "cmp": "dcmp", "conv": "dconv",
                "neg": "dneg", "abs": "dabs", "fmod": "fmod", "sin": "sin", "cos": "cos", "atan2": "atan2",
                "asin": "asin", "exp": "exp", "log": "log", "rng": "rng"}


def cycle_key(op: str, real: str) -> str:
    """Sayılan işlem adı → profil çevrim tablosu anahtarı ("add" + "float" → "fadd", "double" → "dadd")."""
    return (_DOUBLE_KEYS if real == "double" else _FLOAT_KEYS)[op]


def _table(f: tuple[float, ...], d: tuple[float, ...], rng: float, call: float, mem: float) -> dict[str, float]:
    """f / d: add, mul, div, sqrt, cmp, conv, neg, abs, fmod, sin, cos, atan2, asin, exp, log (çevrim)."""
    names = OPS[:-1]
    table = {_FLOAT_KEYS[n]: float(v) for n, v in zip(names, f)}
    table.update({_DOUBLE_KEYS[n]: float(v) for n, v in zip(names, d)})
    table.update(rng=float(rng), call=float(call), mem=float(mem))
    return table


@dataclass(frozen=True)
class McuProfile:
    key: str
    name: str
    fpu: str                                   # "none" | "sp" | "dp"
    gcc_flags: tuple[str, ...] | None          # None → bu çekirdek için yerel derleyici yok
    clang_flags: tuple[str, ...] | None
    cycles: dict[str, float] = field(compare=False, repr=False)
    default_clock_mhz: float = 100.0
    examples: str = ""
    arch: str = "arm"                          # "arm" | "xtensa" | "riscv"
    flash_kb: float = 1024.0                   # örnek parça için tipik bellek (arayüz varsayılanı)
    ram_kb: float = 128.0
    estimate_base: str | None = None           # yerel derleyici yoksa boyutun türetildiği ölçülü profil
    density: float = 1.0                       # kod boyutu çarpanı (estimate_base'e göre)
    note: str = ""

    @property
    def measurable(self) -> bool:
        return self.gcc_flags is not None or self.clang_flags is not None

    def double_is_soft(self) -> bool:
        return self.fpu != "dp"


# ---------------------------------------------------------------------------- çevrim tabloları (TAHMİN)
#                 add  mul   div   sqrt cmp conv neg abs fmod  sin   cos  atan2  asin  exp   log
_M0P = _table((90, 110, 400, 650, 45, 50, 2, 2, 500, 2200, 2200, 3200, 2800, 2600, 2500),
              (160, 330, 1300, 2500, 60, 70, 3, 3, 1500, 8000, 8000, 11000, 9500, 8500, 8000),
              rng=10, call=8, mem=3)
_M3 = _table((45, 40, 150, 350, 25, 25, 1, 1, 250, 900, 900, 1300, 1100, 950, 900),
             (75, 90, 450, 900, 35, 35, 2, 2, 700, 3000, 3000, 4200, 3500, 3200, 3000),
             rng=6, call=6, mem=3)
_M4F = _table((1, 1, 14, 17, 3, 1, 1, 1, 60, 110, 110, 160, 130, 120, 110),
              (60, 50, 350, 800, 25, 25, 2, 2, 400, 2500, 2500, 3500, 3000, 2600, 2500),
              rng=5, call=5, mem=2)
_M33F = _table((1, 1, 14, 17, 3, 1, 1, 1, 55, 105, 105, 150, 125, 115, 105),
               (55, 45, 330, 750, 22, 22, 2, 2, 380, 2400, 2400, 3300, 2900, 2500, 2400),
               rng=5, call=5, mem=2)
_M7 = _table((1, 1, 14, 16, 2, 1, 1, 1, 40, 60, 60, 90, 80, 45, 45),
             (1, 2, 30, 32, 2, 1, 1, 1, 80, 110, 110, 160, 130, 80, 80),
             rng=3, call=4, mem=1)
_ESP32 = _table((2, 2, 45, 60, 3, 2, 1, 1, 120, 150, 150, 220, 200, 180, 160),
                (70, 90, 450, 900, 30, 35, 2, 2, 600, 3500, 3500, 5000, 4200, 3600, 3500),
                rng=5, call=8, mem=3)
_RV_SOFT = _table((60, 55, 180, 450, 30, 30, 2, 2, 350, 1300, 1300, 1800, 1500, 1350, 1300),
                  (100, 140, 600, 1300, 40, 45, 2, 2, 900, 4500, 4500, 6000, 5000, 4600, 4500),
                  rng=6, call=5, mem=3)
_RV_F = _table((1, 1, 20, 22, 2, 2, 1, 1, 80, 130, 130, 190, 160, 140, 130),
               (100, 140, 600, 1300, 40, 45, 2, 2, 900, 4500, 4500, 6000, 5000, 4600, 4500),
               rng=4, call=4, mem=2)

_THUMB = ("-mthumb",)
MCU_PROFILES: dict[str, McuProfile] = {p.key: p for p in (
    McuProfile("cortex-m0plus", "Cortex-M0+ (FPU yok, yazılım kayan nokta)", "none",
               ("-mcpu=cortex-m0plus", *_THUMB, "-mfloat-abi=soft"),
               ("--target=thumbv6m-none-eabi", "-mcpu=cortex-m0plus", "-mfloat-abi=soft"),
               _M0P, 64.0, "STM32G0, STM32L0, RP2040, SAMD21, MSPM0", flash_kb=128, ram_kb=36),
    McuProfile("cortex-m3", "Cortex-M3 (FPU yok, yazılım kayan nokta)", "none",
               ("-mcpu=cortex-m3", *_THUMB, "-mfloat-abi=soft"),
               ("--target=thumbv7m-none-eabi", "-mcpu=cortex-m3", "-mfloat-abi=soft"),
               _M3, 72.0, "STM32F1, STM32F2, STM32L1, LPC1768", flash_kb=128, ram_kb=20),
    McuProfile("cortex-m4f", "Cortex-M4F (FPU tek hassasiyet)", "sp",
               ("-mcpu=cortex-m4", *_THUMB, "-mfpu=fpv4-sp-d16", "-mfloat-abi=hard"),
               ("--target=thumbv7em-none-eabihf", "-mcpu=cortex-m4", "-mfpu=fpv4-sp-d16", "-mfloat-abi=hard"),
               _M4F, 168.0, "STM32F4, STM32G4, STM32L4, nRF52840, NXP Kinetis K64", flash_kb=1024, ram_kb=192),
    McuProfile("cortex-m7", "Cortex-M7 (FPU çift hassasiyet)", "dp",
               ("-mcpu=cortex-m7", *_THUMB, "-mfpu=fpv5-d16", "-mfloat-abi=hard"),
               ("--target=thumbv7em-none-eabihf", "-mcpu=cortex-m7", "-mfpu=fpv5-d16", "-mfloat-abi=hard"),
               _M7, 480.0, "STM32H7, STM32F76x/F77x, NXP i.MX RT10xx", flash_kb=2048, ram_kb=1024,
               note="Önbellek / TCM varsayılır; flash'tan önbelleksiz çalışmada çevrimler belirgin artar."),
    McuProfile("cortex-m33f", "Cortex-M33F (FPU tek hassasiyet, ARMv8-M)", "sp",
               ("-mcpu=cortex-m33", *_THUMB, "-mfpu=fpv5-sp-d16", "-mfloat-abi=hard"),
               ("--target=thumbv8m.main-none-eabihf", "-mcpu=cortex-m33", "-mfpu=fpv5-sp-d16",
                "-mfloat-abi=hard"),
               _M33F, 160.0, "STM32U5, STM32L5, nRF5340, Renesas RA6M5", flash_kb=1024, ram_kb=256),
    McuProfile("esp32", "ESP32 (Xtensa LX6, FPU tek hassasiyet) — tahmin", "sp", None, None,
               _ESP32, 240.0, "ESP32-WROOM / WROVER (ESP32-S3: LX7, benzer)", arch="xtensa",
               flash_kb=4096, ram_kb=320, estimate_base="cortex-m4f", density=1.25,
               note="Xtensa derleyicisi yok: boyut Cortex-M4F GCC ölçümü × 1.25 (24/16 bit komutlar, pencereli "
                    "ABI, literal havuzları). Bölme / karekök donanımda yok (yazılım yinelemesi)."),
    McuProfile("rv32imac", "RISC-V RV32IMAC (FPU yok) — tahmin", "none", None, None,
               _RV_SOFT, 108.0, "GD32VF103, CH32V203, ESP32-C3 (RV32IMC)", arch="riscv",
               flash_kb=128, ram_kb=32, estimate_base="cortex-m3", density=1.20,
               note="RISC-V derleyicisi yok: boyut Cortex-M3 GCC ölçümü × 1.20 (RV32IMC ≈ Thumb-2'den %10 – 25 "
                    "büyük; Embench IoT karşılaştırmaları)."),
    McuProfile("rv32imfc", "RISC-V RV32IMFC (FPU tek hassasiyet) — tahmin", "sp", None, None,
               _RV_F, 144.0, "CH32V307 (QingKe V4F), GD32VW553", arch="riscv",
               flash_kb=256, ram_kb=64, estimate_base="cortex-m4f", density=1.15,
               note="RISC-V derleyicisi yok: boyut Cortex-M4F GCC ölçümü × 1.15."),
)}

COMPILERS: dict[str, str] = {"gcc": "GCC (arm-none-eabi-gcc)", "clang": "Clang / Arm Compiler 6 (Keil MDK)",
                             "iar": "IAR EWARM"}
OPT_LEVELS = ("-O0", "-Os", "-O2", "-O3")

# IAR EWARM bu makinede yok: boyut, aynı ayarla ölçülen GCC değerinden türetilir. Yayımlanmış CoreMark / EEMBC kod
# boyutu karşılaştırmalarında IAR (High-Size) çoğunlukla GCC -Os'tan %5 – 15 küçük kod üretir → 0.90 (TAHMİN).
IAR_SIZE_FACTOR = 0.90
# IAR DLIB matematik kütüphanesi newlib-nano'dan farklıdır; libm dahil toplam için aynı çarpan (TAHMİN).
IAR_LIBM_FACTOR = 0.90


def profile(key: str) -> McuProfile:
    try:
        return MCU_PROFILES[key]
    except KeyError:
        raise ValueError(f"Bilinmeyen MCU profili: {key!r} (seçenekler: {', '.join(MCU_PROFILES)})") from None
