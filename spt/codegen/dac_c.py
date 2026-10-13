"""DAC çıkışı için C kodu üreteci: dalga tablosu (.h/.c), hedefe özgü sürme kodu, örnek dosya ve KULLANIM.md.

Hedefler (`TARGETS`):
  - stm32_hal_dma: DAC + TIM6 TRGO tetik + DMA dairesel; HAL_DAC_Start_DMA ve doğru hizalama sabiti
    (DAC_ALIGN_12B_R / DAC_ALIGN_12B_L / DAC_ALIGN_8B_R). PSC/ARR makroları `timer_config` ile hesaplanır.
  - generic_isr: taşınabilir; zamanlayıcı kesmesi her örnekte `<ad>_isr_tick()` çağırır, kullanıcı
    `dac_write()` fonksiyonunu yazar.
  - dds: 32 bit faz akümülatörlü üreteç, çalışırken frekans değişimi (`<ad>_dds_set_freq`), 2^k tablo,
    isteğe bağlı doğrusal ara değerleme. Python modeli `spt.dsp.siggen.dds_sequence` ile bit düzeyinde aynı.

Yalnız STM32 desteklenir; generic_isr ve dds STM32 zamanlayıcı kesmesiyle (dahili DAC ya da SPI / I2C / I2S üzerinden
harici DAC) kullanılır.

Üretilen çekirdek (.h/.c) satıcı başlığı olmadan `-std=c99 -Wall -Wextra -pedantic -Werror` ile temiz derlenir:
HAL'e özgü bölümler `USE_HAL_DRIVER` (STM32 CubeMX) korumasının içindedir. Örnek dosyalar HAL başlıklarına ihtiyaç
duyabilir.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from spt import APP_NAME
from spt.dsp.siggen import (ACC_BITS, DAC_CODINGS, DacSpec, Metrics, SignalError, Table, TimerConfig, WaveSpec,
                            dds_frac_bits, describe, metrics, timer_config)


@dataclass(frozen=True)
class GeneratedFile:
    """Üretilen dosya: ad (uzantılı) ve içerik."""
    name: str
    text: str


TARGETS: dict[str, str] = {
    "stm32_hal_dma": "STM32 HAL — DAC + TIM6 TRGO + DMA dairesel",
    "generic_isr": "Genel — zamanlayıcı kesmesi (ISR) + dac_write()",
    "dds": "DDS — faz akümülatörü, çalışırken frekans değişimi",
}

# Kod boyutu / CPU tahminleri (Cortex-M3/M4, -O2). TAHMİNDİR; kesin değer derlenmiş nesne dosyasından ölçülmeli.
_CODE_EST = {"stm32_hal_dma": 160, "generic_isr": 48, "dds": 110}
_CYCLES_EST = {"stm32_hal_dma": 0.0, "generic_isr": 40.0, "dds": 50.0}
_DDS_INTERP_EXTRA_CODE = 60
_DDS_INTERP_EXTRA_CYCLES = 15.0


# ============================================================================ yardımcılar
def _ident(name: str) -> str:
    """Addan C tanımlayıcısı: geçersiz karakterler alt çizgi, ardışık / baştaki / sondaki alt çizgiler atılır (çift alt
    çizgi ve _Büyük C / C++'ta saklıdır; makrolar <AD>_XXX biçiminde çift alt çizgi üretmesin), rakamla başlarsa w_."""
    ident = re.sub(r"_+", "_", re.sub(r"[^0-9A-Za-z_]", "_", (name or "").strip())).strip("_") or "wave"
    return "w_" + ident if ident[0].isdigit() else ident


def _c_double(v: float) -> str:
    v = float(v)
    if not math.isfinite(v):
        raise SignalError("C koduna sonlu olmayan sayı yazılamaz.")
    r = repr(v)
    return r if any(c in r for c in ".e") else r + ".0"


def _comment(text: str) -> str:
    """C yorumu içinde yorum sonlandırıcıları etkisizleştirilir."""
    return text.replace("*/", "* /").replace("/*", "/ *")


def _fmt_hz(f: float) -> str:
    a = abs(f)
    if a >= 1e6:
        return f"{f / 1e6:.9g} MHz"
    if a >= 1e3:
        return f"{f / 1e3:.9g} kHz"
    return f"{f:.9g} Hz"


def _c_word(v: int, signed: bool, bits: int) -> str:
    v = int(v)
    if signed:
        if bits == 32 and v == -(1 << 31):
            return "(-2147483647 - 1)"
        return str(v)
    digits = bits // 4
    return f"0x{v:0{digits}X}" + ("u" if bits == 32 else "")


def _c_array(values: np.ndarray, signed: bool, bits: int) -> list[str]:
    per = {8: 16, 16: 12, 32: 8}[bits]
    items = [_c_word(v, signed, bits) for v in np.asarray(values).reshape(-1)]
    return ["    " + ", ".join(items[i:i + per]) + "," for i in range(0, len(items), per)]


def _uint(v: int) -> str:
    return f"{int(v)}u"


@dataclass
class _Ctx:
    ident: str
    P: str
    spec: WaveSpec
    dac: DacSpec
    table: Table
    target: str
    m: Metrics
    tc: TimerConfig | None
    interpolate: bool
    dac_channel: int
    cpu_clock_hz: float
    timer_bits: int
    notes: list[str]


# ============================================================================ doğrulama
def _validate(ctx: _Ctx) -> None:
    t, dac, table = ctx.target, ctx.dac, ctx.table
    if t == "dds":
        if table.mode != "dds":
            raise SignalError("DDS platformu için “Tablo kipi”ni DDS seçin.")
        return
    if table.mode != "exact":
        raise SignalError("Bu platform tabloyu fs'te sırayla çalar: “Tablo kipi”ni “Tam periyot” seçin "
                          "(DDS tablosu için platform olarak DDS'i seçin).")
    if t == "stm32_hal_dma":
        if dac.bits not in (8, 12):
            raise SignalError("STM32 dahili DAC'ı 8 ya da 12 bittir: “STM32 DAC 12 bit” ya da “STM32 DAC 8 bit” "
                              "hazır ayarını seçin.")
        if dac.signed:
            raise SignalError("STM32 dahili DAC'ı işaretsiz kod kullanır: Kodlama'yı “İşaretsiz” seçin.")
        if dac.bits == 8 and (dac.align != "right" or dac.word_bits != 8):
            raise SignalError("STM32 8 bit modu yalnızca sağa hizalıdır ve 8 bit kelime kullanır: “STM32 DAC 8 bit” "
                              "hazır ayarını seçin.")
        if dac.bits == 12 and dac.word_bits != 16:
            raise SignalError("STM32 12 bit modu 16 bitlik kelime (yarım kelime DMA) kullanır: Kelime'yi 16 bit "
                              "seçin.")
        if ctx.dac_channel not in (1, 2):
            raise SignalError("STM32 DAC kanalı 1 ya da 2 olmalı.")


# ============================================================================ başlık (.h)
def _banner(ctx: _Ctx, what: str) -> list[str]:
    t, dac, m = ctx.table, ctx.dac, ctx.m
    lines = [f"{APP_NAME} — {what} \"{ctx.ident}\"",
             f"Dalga: {describe(ctx.spec)}"]
    if t.spec2 is not None and t.channels == 2:
        lines.append(f"Kanal 2: {describe(t.spec2)}")
    lines.append(f"Hedef: {TARGETS[ctx.target]}")
    if t.mode == "dds":
        lines.append(f"DDS: tablo 2^{t.table_bits} = {t.N} seviye (tek periyot), 32 bit faz akümülatörü; "
                     f"fs = {_fmt_hz(t.fs)}, W = {t.tuning_word} → f = {t.f_real:.9g} Hz "
                     f"(istenen {t.f_target:.9g} Hz), çözünürlük {t.freq_resolution:.6g} Hz")
    elif t.periodic:
        lines.append(f"Tablo: N = {t.N} örnek, M = {t.periods} periyot; fs = {_fmt_hz(t.fs)} → "
                     f"f = M·fs/N = {t.f_real:.9g} Hz (istenen {t.f_target:.9g} Hz, hata {t.freq_error_ppm:+.3f} ppm)")
    else:
        lines.append(f"Tablo: N = {t.N} örnek (periyodik olmayan); fs = {_fmt_hz(t.fs)}, döngü süresi "
                     f"{t.N / t.fs:.6g} s")
    assert dac.out_max is not None
    lines.append(f"DAC: {dac.bits} bit, {DAC_CODINGS[dac.coding]}, {'sola' if dac.align == 'left' else 'sağa'} "
                 f"hizalı, {dac.c_type}; {dac.out_min:g} … {dac.out_max:g} V, 1 LSB = {dac.lsb * 1e3:.6g} mV")
    lines.append(f"Bellek: tablo {m.memory_bytes} bayt (const → flash)")
    if t.clipped:
        lines.append(f"UYARI: {t.clipped} örnek DAC aralığı dışındaydı ve kırpıldı")
    lines.append("Bu dosya otomatik üretilmiştir; elle değiştirmeyin, aracı yeniden çalıştırın.")
    return ["/*"] + [f" * {_comment(s)}" for s in lines] + [" */"]


def _timer_macros(ctx: _Ctx) -> list[str]:
    tc, P = ctx.tc, ctx.P
    if tc is None:
        return []
    arr_max = (1 << ctx.timer_bits) - 1
    return [
        f"/* STM32 tarzı zamanlayıcı: fs = TIM_CLK / ((PSC + 1) * (ARR + 1)); TIM_CLK = {_fmt_hz(tc.timer_clock_hz)}",
        f" * (CubeMX 'APB1 Timer clocks' ile aynı olmalı). Gerçekleşen fs = {tc.fs_real:.9g} Hz "
        f"({tc.error_ppm:+.3f} ppm). */",
        f"#define {P}_TIM_CLOCK_HZ        {_c_double(tc.timer_clock_hz)}",
        f"#define {P}_TIM_PSC             {_uint(tc.psc)}",
        f"#define {P}_TIM_ARR             {_uint(tc.arr)}",
        f"#define {P}_TIM_ARR_MAX         {_uint(arr_max)}",
        f"#define {P}_TIM_FS_HZ           {_c_double(tc.fs_real)}",
        "",
    ]


def _header(ctx: _Ctx) -> str:
    t, dac, P, ident = ctx.table, ctx.dac, ctx.P, ctx.ident
    guard = f"{P}_H"
    out = _banner(ctx, "DAC dalga tablosu")
    out += [f"#ifndef {guard}", f"#define {guard}", "", "#include <stdint.h>", "",
            "#ifdef __cplusplus", 'extern "C" {', "#endif", ""]
    out += [
        f"#define {P}_N                   {_uint(t.N)}   /* kanal başına örnek (DDS: 2^k seviye) */",
        f"#define {P}_CHANNELS            {_uint(t.channels)}",
        f"#define {P}_TABLE_LEN           {_uint(t.N * (1 if t.mode == 'dds' else t.channels))}"
        "   /* dizi eleman sayısı */",
        f"#define {P}_PERIODS             {_uint(t.periods)}   /* tablodaki periyot sayısı (M) */",
        f"#define {P}_FS_HZ               {_c_double(t.fs)}   /* DAC güncelleme hızı */",
        f"#define {P}_F_HZ                {_c_double(t.f_real)}   /* gerçekleşen frekans */",
        f"#define {P}_F_TARGET_HZ         {_c_double(t.f_target)}",
        f"#define {P}_DAC_BITS            {_uint(dac.bits)}",
        f"#define {P}_DAC_SHIFT           {_uint(dac.shift)}   /* sola hizalama kaydırması */",
        f"#define {P}_DAC_LEFT_ALIGNED    {1 if dac.align == 'left' else 0}",
        f"#define {P}_DAC_TWOS_COMPLEMENT {1 if dac.signed else 0}",
        f"#define {P}_VOUT_MIN_V          {_c_double(dac.out_min)}",
        f"#define {P}_VOUT_MAX_V          {_c_double(dac.out_max or 0.0)}",
        f"#define {P}_TABLE_BYTES         {_uint(t.memory_bytes)}",
        "",
        f"typedef {dac.c_type} {ident}_word_t;   /* DAC'a yazılan kelime */",
        "",
    ]
    tgt = ctx.target
    out += _timer_macros(ctx)
    if tgt == "dds":
        k = t.table_bits or 0
        out += [
            f"#define {P}_DDS_TABLE_BITS      {_uint(k)}",
            f"#define {P}_DDS_SIZE            (UINT32_C(1) << {P}_DDS_TABLE_BITS)",
            f"#define {P}_DDS_FRAC_BITS       {_uint(dds_frac_bits(k))}   /* ara değerleme kesir biti */",
            f"#define {P}_DDS_STEP            UINT32_C({t.tuning_word})   /* {t.f_real:.9g} Hz için ayar kelimesi */",
            f"#define {P}_DDS_STEP_PER_HZ     {_c_double(2.0 ** ACC_BITS / t.fs)}   /* 2^32 / fs */",
            f"#define {P}_DDS_RESOLUTION_HZ   {_c_double(t.fs / 2.0 ** ACC_BITS)}   /* fs / 2^32 */",
            f"#ifndef {P}_DDS_INTERP",
            f"#define {P}_DDS_INTERP          {1 if ctx.interpolate else 0}   /* 1: doğrusal ara değerleme */",
            "#endif",
            "",
            f"typedef {dac.level_c_type} {ident}_level_t;   /* tabloda işaretsiz seviye 0 … 2^b − 1 */",
            "",
            "typedef struct {",
            "    uint32_t phase;   /* faz akümülatörü (2^32 = bir periyot) */",
            "    uint32_t step;    /* ayar kelimesi: f = step * fs / 2^32 */",
            f"}} {ident}_dds_t;",
            "",
            f"extern const {ident}_level_t *const {ident}_dds_levels;",
            "",
            f"void {ident}_dds_init({ident}_dds_t *g);                    /* faz 0, step = {P}_DDS_STEP */",
            f"void {ident}_dds_set_step({ident}_dds_t *g, uint32_t step);",
            f"void {ident}_dds_set_freq({ident}_dds_t *g, double hz);     /* 0 … fs/2'ye kırpılır */",
            f"void {ident}_dds_set_phase({ident}_dds_t *g, uint32_t phase);",
            f"{ident}_word_t {ident}_dds_next({ident}_dds_t *g);          /* sıradaki DAC kelimesi */",
            f"{ident}_word_t {ident}_level_to_word(uint32_t level);       /* seviye → DAC kelimesi */",
        ]
    else:
        out += [f"extern const {ident}_word_t *const {ident}_table;   /* {P}_TABLE_LEN eleman"
                + (", serpiştirilmiş [k1, k2, k1, k2 …]" if t.channels == 2 else "") + " */", ""]
    if tgt in ("stm32_hal_dma", "generic_isr"):
        out += [f"uint32_t {ident}_arr_for_hz(double hz);   /* çıkış frekansı hz için ARR (PSC sabit) */"]
    if tgt == "generic_isr":
        if t.channels == 2:
            out += [f"void dac_write_stereo({ident}_word_t ch1, {ident}_word_t ch2);   /* KULLANICI yazar */"]
        else:
            out += [f"void dac_write({ident}_word_t code);   /* KULLANICI yazar: kodu DAC'a gönder */"]
        out += [f"void {ident}_isr_tick(void);   /* zamanlayıcı kesmesinde fs hızında çağırın */",
                f"void {ident}_reset(void);"]
    if tgt == "stm32_hal_dma":
        out += _stm32_header(ctx)
    out += ["", "#ifdef __cplusplus", "}", "#endif", "", f"#endif /* {guard} */", ""]
    return "\n".join(out)


def _stm32_header(ctx: _Ctx) -> list[str]:
    P, ident, dac = ctx.P, ctx.ident, ctx.dac
    align = {(12, "right"): "DAC_ALIGN_12B_R", (12, "left"): "DAC_ALIGN_12B_L", (8, "right"): "DAC_ALIGN_8B_R"}
    reg = {(12, "right"): "DHR12RD", (12, "left"): "DHR12LD", (8, "right"): "DHR8RD"}[(dac.bits, dac.align)]
    ch = f"DAC_CHANNEL_{ctx.dac_channel}"
    return [
        "",
        "/* HAL sabitleri (yalnızca USE_HAL_DRIVER tanımlıyken kullanılır) */",
        f"#define {P}_HAL_DAC_ALIGN       {align[(dac.bits, dac.align)]}",
        f"#define {P}_HAL_DAC_CHANNEL     {ch}",
        f"#define {P}_HAL_DUAL_REG        {reg}   /* stereo (ikili) mod veri yazmacı */",
        "",
        "#if defined(USE_HAL_DRIVER)",
        f"#ifndef {P}_HAL_HEADER",
        f"#define {P}_HAL_HEADER \"main.h\"   /* CubeMX projesi; gerekirse ör. \"stm32f4xx_hal.h\" */",
        "#endif",
        f"#include {P}_HAL_HEADER",
        f"HAL_StatusTypeDef {ident}_start(DAC_HandleTypeDef *hdac, TIM_HandleTypeDef *htim);",
        f"HAL_StatusTypeDef {ident}_stop(DAC_HandleTypeDef *hdac, TIM_HandleTypeDef *htim);",
        f"void {ident}_set_arr(TIM_HandleTypeDef *htim, uint32_t arr);   /* çalışırken frekans değiştir */",
        "#endif",
    ]


# ============================================================================ kaynak (.c)
def _table_block(ctx: _Ctx) -> list[str]:
    t, dac, P, ident = ctx.table, ctx.dac, ctx.P, ctx.ident
    out = [f"#ifndef {P}_TABLE_IN_RAM",
           f"#define {P}_TABLE_QUAL const   /* const: tablo flash'ta (bağlayıcı .rodata'ya koyar) */",
           "#else",
           f"#define {P}_TABLE_QUAL         /* -D{P}_TABLE_IN_RAM: tablo RAM'de (DMA flash'a erişemiyorsa) */",
           "#endif", ""]
    if t.mode == "dds":
        out += [f"static {P}_TABLE_QUAL {ident}_level_t {ident}_dds_levels_data[{P}_DDS_SIZE] = {{"]
        out += _c_array(t.levels, False, 8 * dac.level_bytes)
        out += ["};", f"const {ident}_level_t *const {ident}_dds_levels = {ident}_dds_levels_data;", ""]
    elif t.channels == 2:
        # İkili DAC yazmacına tek DMA aktarımıyla kanal çifti yazılır (STM32: 12 bitte Word, 8 bitte Half Word):
        # tablo 4 bayt hizalı olmalı. uint32_t üyeli birleşim bunu her C99 derleyicisinde garanti eder (özniteliksiz).
        out += ["/* stereo: serpiştirilmiş [kanal1, kanal2, kanal1, kanal2 …]; birleşim = 4 bayt hiza (ikili DMA) */",
                f"static {P}_TABLE_QUAL union {{",
                f"    {ident}_word_t w[{P}_TABLE_LEN];",
                "    uint32_t align;",
                f"}} {ident}_table_store = {{ {{"]
        out += _c_array(t.flat_codes, dac.signed, dac.word_bits)
        out += ["} };", f"#define {ident}_table_data ({ident}_table_store.w)",
                f"const {ident}_word_t *const {ident}_table = {ident}_table_data;", ""]
    else:
        out += [f"static {P}_TABLE_QUAL {ident}_word_t {ident}_table_data[{P}_TABLE_LEN] = {{"]
        out += _c_array(t.flat_codes, dac.signed, dac.word_bits)
        out += ["};", f"const {ident}_word_t *const {ident}_table = {ident}_table_data;", ""]
    return out


def _arr_for_hz(ctx: _Ctx) -> list[str]:
    P, ident = ctx.P, ctx.ident
    return [
        f"uint32_t {ident}_arr_for_hz(double hz)",
        "{",
        "    double a;",
        "    if (!(hz > 0.0)) {",
        f"        return {P}_TIM_ARR_MAX;",
        "    }",
        f"    a = {P}_TIM_CLOCK_HZ * (double){P}_PERIODS",
        f"        / ((double)({P}_TIM_PSC + 1u) * (double){P}_N * hz) + 0.5;",
        f"    if (a >= (double){P}_TIM_ARR_MAX + 1.0) {{",
        f"        return {P}_TIM_ARR_MAX;",
        "    }",
        "    if (a < 2.0) {",
        "        return 1u;",
        "    }",
        "    return (uint32_t)a - 1u;",
        "}",
        "",
    ]


def _level_to_word_body(ctx: _Ctx) -> str:
    dac, P, ident = ctx.dac, ctx.P, ctx.ident
    if dac.signed:
        return (f"    return ({ident}_word_t)(((int32_t)level - (int32_t){1 << (dac.bits - 1)}L) "
                f"* (int32_t){1 << dac.shift}L);")
    if dac.shift:
        return f"    return ({ident}_word_t)(level << {P}_DAC_SHIFT);"
    return f"    return ({ident}_word_t)level;"


def _dds_code(ctx: _Ctx) -> list[str]:
    P, ident = ctx.P, ctx.ident
    return [
        f"{ident}_word_t {ident}_level_to_word(uint32_t level)",
        "{",
        _level_to_word_body(ctx),
        "}",
        "",
        f"void {ident}_dds_init({ident}_dds_t *g)",
        "{",
        "    g->phase = 0u;",
        f"    g->step = {P}_DDS_STEP;",
        "}",
        "",
        f"void {ident}_dds_set_step({ident}_dds_t *g, uint32_t step)",
        "{",
        "    g->step = step;   /* 32 bit yazma: 32 bit MCU'larda atomik */",
        "}",
        "",
        f"void {ident}_dds_set_phase({ident}_dds_t *g, uint32_t phase)",
        "{",
        "    g->phase = phase;",
        "}",
        "",
        "/* step = floor(hz * 2^32/fs + 0.5); Python: spt.dsp.siggen.dds_tuning ile aynı işlem sırası.",
        " * double kullanır (FPU'su tek hassasiyetli MCU'da yazılımla; seyrek çağırın). */",
        f"void {ident}_dds_set_freq({ident}_dds_t *g, double hz)",
        "{",
        "    double s;",
        "    if (!(hz > 0.0)) {",
        "        hz = 0.0;",
        "    }",
        f"    if (hz > {P}_FS_HZ / 2.0) {{",
        f"        hz = {P}_FS_HZ / 2.0;",
        "    }",
        f"    s = hz * {P}_DDS_STEP_PER_HZ + 0.5;",
        "    g->step = (uint32_t)s;",
        "}",
        "",
        f"{ident}_word_t {ident}_dds_next({ident}_dds_t *g)",
        "{",
        "    uint32_t ph = g->phase;",
        f"    uint32_t idx = ph >> (32u - {P}_DDS_TABLE_BITS);",
        f"    uint32_t level = {ident}_dds_levels_data[idx];",
        f"#if {P}_DDS_INTERP",
        f"    uint32_t frac = (ph >> (32u - {P}_DDS_TABLE_BITS - {P}_DDS_FRAC_BITS))",
        f"                    & ((UINT32_C(1) << {P}_DDS_FRAC_BITS) - 1u);",
        f"    uint32_t next = {ident}_dds_levels_data[(idx + 1u) & ({P}_DDS_SIZE - 1u)];",
        f"    level = (uint32_t)(((uint64_t)level * ((UINT32_C(1) << {P}_DDS_FRAC_BITS) - frac)",
        "                        + (uint64_t)next * frac",
        f"                        + (UINT32_C(1) << ({P}_DDS_FRAC_BITS - 1u))) >> {P}_DDS_FRAC_BITS);",
        "#endif",
        "    g->phase = ph + g->step;",
        f"    return {ident}_level_to_word(level);",
        "}",
        "",
    ]


def _generic_code(ctx: _Ctx) -> list[str]:
    P, ident = ctx.P, ctx.ident
    if ctx.table.channels == 2:
        write = [f"    dac_write_stereo({ident}_table_data[2u * i], {ident}_table_data[2u * i + 1u]);"]
    else:
        write = [f"    dac_write({ident}_table_data[i]);"]
    return [
        f"static volatile uint32_t {ident}_pos = 0u;",
        "",
        "/* Önce DAC'a yaz, sonra indeksi ilerlet: kesme gecikmesindeki dalgalanma çıkışa en az yansır. */",
        f"void {ident}_isr_tick(void)",
        "{",
        f"    uint32_t i = {ident}_pos;",
        *write,
        f"    {ident}_pos = (i + 1u < {P}_N) ? i + 1u : 0u;",
        "}",
        "",
        f"void {ident}_reset(void)",
        "{",
        f"    {ident}_pos = 0u;",
        "}",
        "",
    ]


def _stm32_code(ctx: _Ctx) -> list[str]:
    P, ident = ctx.P, ctx.ident
    stereo = ctx.table.channels == 2
    out = [
        "#if defined(USE_HAL_DRIVER)",
        "/* CubeMX: DAC tetik = 'Timer 6 Trigger Out event'; TIM6 TRGO = Update Event; DMA dairesel. */",
        f"HAL_StatusTypeDef {ident}_start(DAC_HandleTypeDef *hdac, TIM_HandleTypeDef *htim)",
        "{",
        "    HAL_StatusTypeDef st;",
        f"    __HAL_TIM_SET_PRESCALER(htim, {P}_TIM_PSC);",
        f"    __HAL_TIM_SET_AUTORELOAD(htim, {P}_TIM_ARR);",
        "    __HAL_TIM_SET_COUNTER(htim, 0u);",
        "    htim->Instance->EGR = TIM_EGR_UG;   /* PSC'yi hemen yükle (DAC henüz kapalı) */",
    ]
    if stereo:
        out += [
            "    /* İkili mod: iki kanal tek DMA akışıyla, ikili veri yazmacına (DMA veri genişliği: 12 bitte Word,",
            "     * 8 bitte Half Word). HAL_DAC_Start_DMA tek kanal içindir; burada doğrudan sürülür. */",
            f"    st = HAL_DMA_Start(hdac->DMA_Handle1, (uint32_t)(uintptr_t){ident}_table_data,",
            f"                       (uint32_t)(uintptr_t)&hdac->Instance->{P}_HAL_DUAL_REG, {P}_N);",
            "    if (st != HAL_OK) {",
            "        return st;",
            "    }",
            "    SET_BIT(hdac->Instance->CR, DAC_CR_DMAEN1);",
            "    __HAL_DAC_ENABLE(hdac, DAC_CHANNEL_1);",
            "    __HAL_DAC_ENABLE(hdac, DAC_CHANNEL_2);",
        ]
    else:
        out += [
            f"    st = HAL_DAC_Start_DMA(hdac, {P}_HAL_DAC_CHANNEL, (uint32_t *)(uintptr_t){ident}_table_data,",
            f"                           {P}_N, {P}_HAL_DAC_ALIGN);",
            "    if (st != HAL_OK) {",
            "        return st;",
            "    }",
        ]
    out += [
        "    return HAL_TIM_Base_Start(htim);",
        "}",
        "",
        f"HAL_StatusTypeDef {ident}_stop(DAC_HandleTypeDef *hdac, TIM_HandleTypeDef *htim)",
        "{",
        "    (void)HAL_TIM_Base_Stop(htim);",
    ]
    if stereo:
        out += ["    CLEAR_BIT(hdac->Instance->CR, DAC_CR_DMAEN1);",
                "    __HAL_DAC_DISABLE(hdac, DAC_CHANNEL_2);",
                "    __HAL_DAC_DISABLE(hdac, DAC_CHANNEL_1);",
                "    return HAL_DMA_Abort(hdac->DMA_Handle1);"]
    else:
        out += [f"    return HAL_DAC_Stop_DMA(hdac, {P}_HAL_DAC_CHANNEL);"]
    out += [
        "}",
        "",
        "/* ARR önyükleme (ARPE) açıksa değişiklik bir sonraki güncellemede, kesintisiz uygulanır. */",
        f"void {ident}_set_arr(TIM_HandleTypeDef *htim, uint32_t arr)",
        "{",
        "    __HAL_TIM_SET_AUTORELOAD(htim, arr);",
        "}",
        "#endif /* USE_HAL_DRIVER */",
        "",
    ]
    return out


def _source(ctx: _Ctx) -> str:
    out = _banner(ctx, "DAC dalga tablosu")
    out += [f'#include "{ctx.ident}.h"', ""]
    out += _table_block(ctx)
    tgt = ctx.target
    if tgt in ("stm32_hal_dma", "generic_isr"):
        out += _arr_for_hz(ctx)
    if tgt == "dds":
        out += _dds_code(ctx)
    elif tgt == "generic_isr":
        out += _generic_code(ctx)
    elif tgt == "stm32_hal_dma":
        out += _stm32_code(ctx)
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out) + "\n"


# ============================================================================ örnek dosya
def _example(ctx: _Ctx) -> GeneratedFile:
    ident, P, t, tgt = ctx.ident, ctx.P, ctx.table, ctx.target
    stereo = t.channels == 2
    head = [f"/* {APP_NAME} — \"{ident}\" kullanım örneği ({_comment(TARGETS[tgt])}) */"]
    if tgt == "stm32_hal_dma":
        body = head + [
            "/* CubeMX projesinde main.c içine (USER CODE bölümlerine) ekleyin. */",
            '#include "main.h"',
            f'#include "{ident}.h"',
            "",
            "extern DAC_HandleTypeDef hdac;    /* F4/F7: hdac — G4/L4/H7: hdac1 (CubeMX adına göre düzeltin) */",
            "extern TIM_HandleTypeDef htim6;",
            "",
            f"void {ident}_example_start(void)",
            "{",
            "    /* USER CODE BEGIN 2 içinde, MX_DAC_Init() ve MX_TIM6_Init() sonrasında çağırın */",
            f"    if ({ident}_start(&hdac, &htim6) != HAL_OK) {{",
            "        Error_Handler();",
            "    }",
            "}",
            "",
            "/* Çıkış frekansını çalışırken değiştirme (tablo aynı, DAC güncelleme hızı değişir): */",
            f"void {ident}_example_set_freq(double hz)",
            "{",
            f"    {ident}_set_arr(&htim6, {ident}_arr_for_hz(hz));",
            "}",
            "",
        ]
        if stereo:
            body.insert(len(head) + 1, "/* Stereo: DAC kanal 1 ve 2 tetik = TIM6 TRGO; DMA yalnızca DAC_CH1'de, "
                                       "veri genişliği Word (12 bit) / Half Word (8 bit). */")
        return GeneratedFile(f"{ident}_ornek.c", "\n".join(body))
    if tgt == "generic_isr":
        if stereo:
            impl = [f"void dac_write_stereo({ident}_word_t ch1, {ident}_word_t ch2)",
                    "{",
                    "    /* Örn. STM32: DAC->DHR12R1 = ch1; DAC->DHR12R2 = ch2; */",
                    "    (void)ch1;",
                    "    (void)ch2;",
                    "}"]
        else:
            impl = [f"void dac_write({ident}_word_t code)",
                    "{",
                    "    /* Örn. STM32 dahili DAC : DAC->DHR12R1 = code;",
                    "     * Örn. SPI DAC MCP4921  : spi_send16(0x3000u | (code & 0x0FFFu));",
                    "     * Örn. I2C DAC MCP4725  : (kesmede I2C yavaştır; tampon + DMA önerilir) */",
                    "    (void)code;",
                    "}"]
        body = head + [
            f'#include "{ident}.h"',
            "",
            "/* 1) DAC'a yazma fonksiyonu (donanımınıza göre doldurun): */",
            *impl,
            "",
            f"/* 2) Zamanlayıcıyı fs = {t.fs:.9g} Hz kesme üretecek şekilde kurun (STM32 tarzı PSC/ARR:",
            f" *    {P}_TIM_PSC / {P}_TIM_ARR) ve kesme fonksiyonunda çağırın: */",
            "void TIMx_IRQHandler(void)",
            "{",
            "    /* önce kesme bayrağını temizleyin (ör. TIM6->SR = 0;) */",
            f"    {ident}_isr_tick();",
            "}",
            "",
        ]
        return GeneratedFile(f"{ident}_ornek.c", "\n".join(body))
    # dds
    body = head + [
        f'#include "{ident}.h"',
        "",
        f"static {ident}_dds_t gen;",
        "",
        f"void dac_write({ident}_word_t code);   /* donanımınıza göre yazın (bkz. KULLANIM.md) */",
        f"void dac_write({ident}_word_t code)",
        "{",
        "    (void)code;   /* ör. DAC->DHR12R1 = code; */",
        "}",
        "",
        f"void {ident}_example_init(void)",
        "{",
        f"    {ident}_dds_init(&gen);              /* tasarım frekansı: {P}_F_HZ */",
        f"    {ident}_dds_set_freq(&gen, 440.0);   /* istediğiniz an, kesme açıkken de güvenli */",
        "}",
        "",
        f"/* fs = {t.fs:.9g} Hz zamanlayıcı kesmesi */",
        "void TIMx_IRQHandler(void)",
        "{",
        "    /* önce kesme bayrağını temizleyin */",
        f"    dac_write({ident}_dds_next(&gen));",
        "}",
        "",
    ]
    return GeneratedFile(f"{ident}_ornek.c", "\n".join(body))


# ============================================================================ bellek raporu
def memory_report(table: Table, dac: DacSpec, target: str, *, interpolate: bool = False,
                  cpu_clock_hz: float = 168e6) -> dict[str, Any]:
    """Bellek ve CPU özeti. Kod boyutu ve döngü sayıları TAHMİNDİR (Cortex-M3/M4, -O2; satıcı sürücüleri
    hariç); kesin boyut derlenmiş nesne dosyasından ölçülmelidir.

    Anahtarlar: target, table_flash_bytes, table_ram_bytes_if_in_ram, dma_buffer_ram_bytes, state_ram_bytes,
    ram_bytes, code_size_estimate_bytes, code_size_is_estimate (True), cpu_cycles_per_sample_estimate,
    cpu_load_pct_estimate, cpu_clock_hz, notes (Türkçe).
    """
    if target not in TARGETS:
        raise SignalError(f"Bilinmeyen hedef '{target}'; geçerliler: {', '.join(TARGETS)}.")
    if dac != table.dac:
        raise SignalError("DacSpec tablonunkiyle aynı değil.")
    notes: list[str] = []
    tab = table.memory_bytes
    state = {"stm32_hal_dma": 0, "generic_isr": 4, "dds": 8}[target]
    code = _CODE_EST[target] + (_DDS_INTERP_EXTRA_CODE if (target == "dds" and interpolate) else 0)
    cycles = _CYCLES_EST[target]
    if target == "dds" and interpolate:
        cycles += _DDS_INTERP_EXTRA_CYCLES
    if target == "stm32_hal_dma":
        notes.append("DMA dairesel: örnek başına CPU yükü ≈ 0 (yalnızca veri yolu çevrimleri). HAL DAC/TIM/DMA "
                     "sürücüleri (birkaç kB) CubeMX projesinde zaten vardır, tahmine dahil değil.")
    if target == "generic_isr":
        notes.append("ISR ≈ 40 çevrim/örnek (Cortex-M kesme giriş/çıkış ≈ 24 + gövde); dac_write maliyeti hariç "
                     "(dahili DAC yazmacı birkaç çevrim, SPI DAC çok daha fazla).")
    if target == "dds":
        notes.append("DDS ≈ 50 çevrim/örnek kesme dahil (ara değerlemeyle ≈ 65); dac_write hariç. "
                     "dds_set_freq double kullanır: FPU'su tek hassasiyetli MCU'da yazılım kütüphanesi "
                     "(≈ 1–2 kB, başka yerde double yoksa) eklenir.")
    load = cycles * table.fs / cpu_clock_hz * 100.0
    return {
        "target": target,
        "table_flash_bytes": tab,
        "table_ram_bytes_if_in_ram": tab,
        "dma_buffer_ram_bytes": 0,   # STM32 DMA tabloyu yerinde okur (ek tampon yok)
        "state_ram_bytes": state,
        "ram_bytes": state,
        "code_size_estimate_bytes": code,
        "code_size_is_estimate": True,
        "cpu_cycles_per_sample_estimate": cycles,
        "cpu_load_pct_estimate": load,
        "cpu_clock_hz": float(cpu_clock_hz),
        "notes": notes,
    }


# ============================================================================ KULLANIM.md
def _f(v: float, fmt: str = ".4g", unit: str = "") -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    return f"{v:{fmt}}{(' ' + unit) if unit else ''}"


def _usage(ctx: _Ctx, files: list[str], mem: dict[str, Any]) -> str:
    t, dac, m, P, ident, tgt, tc = ctx.table, ctx.dac, ctx.m, ctx.P, ctx.ident, ctx.target, ctx.tc
    assert dac.out_max is not None
    L: list[str] = [f"# {ident} — DAC sinyal üreteci kullanım kılavuzu", "",
                    f"{APP_NAME} tarafından üretildi. Hedef: **{TARGETS[tgt]}**.", "", "## Özet", "",
                    "| Özellik | Değer |", "|---|---|",
                    f"| Dalga | {describe(ctx.spec)} |"]
    if t.spec2 is not None and t.channels == 2:
        L.append(f"| Kanal 2 | {describe(t.spec2)} |")
    if t.mode == "dds":
        L.append(f"| Tablo | 2^{t.table_bits} = {t.N} seviye, tek periyot (DDS) |")
        L.append(f"| Ayar kelimesi | W = {t.tuning_word} (çözünürlük fs/2^32 = {t.freq_resolution:.6g} Hz) |")
    else:
        L.append(f"| Tablo | N = {t.N} örnek, M = {t.periods} periyot"
                 f"{'' if t.periodic else ' (periyodik olmayan, döngüyle tekrarlanır)'} |")
    L += [f"| DAC güncelleme hızı | fs = {t.fs:.9g} Hz (her {1e6 / t.fs:.6g} µs bir örnek) |",
          f"| Gerçekleşen frekans | {t.f_real:.9g} Hz (istenen {t.f_target:.9g} Hz, hata {t.freq_error_hz:+.6g} Hz"
          f" / {t.freq_error_ppm:+.3f} ppm) |",
          f"| DAC | {dac.bits} bit, {DAC_CODINGS[dac.coding]}, {'sola' if dac.align == 'left' else 'sağa'} hizalı, "
          f"`{dac.c_type}`; {dac.out_min:g} … {dac.out_max:g} V, 1 LSB = {dac.lsb * 1e3:.4g} mV |",
          f"| Tablo belleği | {t.memory_bytes} bayt (flash, `const`) |",
          f"| Kırpılan örnek | {t.clipped} |", ""]
    if tc is not None:
        L += [f"Zamanlayıcı (STM32 tarzı, TIM_CLK = {tc.timer_clock_hz:.9g} Hz): **PSC = {tc.psc}, ARR = {tc.arr}** "
              f"→ fs = {tc.fs_real:.9g} Hz ({tc.error_ppm:+.3f} ppm).", ""]

    L += ["## Tablo nedir?", ""]
    if t.mode == "dds":
        L += [f"`{ident}_dds_levels`, dalganın **bir periyodunu** {t.N} eşit faz adımında tutan işaretsiz DAC "
              "seviyeleridir (0 … 2^b − 1, hizalama/kodlama öncesi). Her örnekte 32 bit faz akümülatörüne ayar "
              f"kelimesi eklenir; üst {t.table_bits} bit tablo indeksidir. `{ident}_level_to_word()` seviyeyi "
              "DAC kelimesine (kodlama + hizalama) çevirir.", ""]
    else:
        L += [f"`{ident}_table`, DAC'a fs hızında sırayla ve döngüyle yazılacak **hazır DAC kelimeleridir** "
              f"(kodlama ve {'sola' if dac.align == 'left' else 'sağa'} hizalama uygulanmış). "
              + (f"Tabloda tam {t.periods} periyot vardır; son örnekten ilk örneğe geçiş süreksizlik yaratmaz. "
                 "M > 1 ise frekans M·fs/N ile istenen değere çok daha yakın tutturulur."
                 if t.periodic else "Tablo periyodik olmayan bir kaydı/sinyali tutar ve döngüyle tekrar çalınır.")
              + (" Stereo: elemanlar serpiştirilmiştir (kanal 1, kanal 2, kanal 1, …)." if t.channels == 2 else ""),
              ""]
    L += ["## Dosyalar", ""]
    desc = {f"{ident}.h": "makrolar (N, fs, f, PSC/ARR, hizalama), tipler ve fonksiyon bildirimleri",
            f"{ident}.c": "tablo (`static const`, flash) ve hedefe özgü sürme kodu",
            "KULLANIM.md": "bu kılavuz"}
    for fn in files:
        L.append(f"- `{fn}` — {desc.get(fn, 'kullanım örneği')}")
    L += ["", f"Tabloyu RAM'e almak için (ör. DMA'nın flash'a erişemediği aileler) `-D{P}_TABLE_IN_RAM` ile "
          "derleyin.", "", "## Entegrasyon", ""]
    L += _usage_target(ctx)
    L += ["", "## Frekansı değiştirme", ""]
    if t.mode == "dds":
        L += [f"- Çalışırken: `{ident}_dds_set_freq(&g, hz)` (double, 0 … fs/2) ya da tam sayı `{ident}_dds_set_step(&g, "
              f"step)`; step = round(hz · 2^32 / fs). Çözünürlük {t.freq_resolution:.6g} Hz. Faz sürekli kalır "
              "(atlama/tıklama olmaz).",
              "- Dalga şeklini değiştirmek için tabloyu yeniden üretin."]
    else:
        L += [f"- **ARR ile (tablo aynı):** f_çıkış = M · TIM_CLK / ((PSC+1)(ARR+1) · N). `{ident}_arr_for_hz(hz)`"
              " uygun ARR'yi verir"
              + (f", `{ident}_set_arr(&htim6, arr)` uygular" if tgt == "stm32_hal_dma" else "")
              + ". ARR önyükleme (ARPE) açıkken değişiklik kesintisizdir; çözünürlük ARR adımıyla sınırlıdır "
              "(yüksek frekansta kaba)."]
        L += ["- **Tabloyu yeniden üretmek:** farklı frekans için N/M yeniden seçilir (en hassas yol).",
              "- **İnce ve hızlı ayar gerekiyorsa** `dds` hedefini kullanın (fs/2^32 çözünürlük, faz sürekli)."]
    L += ["", "## Çıkış filtresi (yeniden yapılandırma)", "",
          "DAC çıkışı basamaklıdır (sıfır dereceli tutma); fs'nin katları çevresinde görüntüler (imge) oluşur.", "",
          f"- Anlamlı içeriğin en yüksek frekansı: {_f(m.content_max_hz, '.6g', 'Hz')} (temel bileşene göre −40 dBc eşiği)",
          f"- İlk görüntü: {_f(m.first_image_hz, '.6g', 'Hz')} (temel bileşenin görüntüsü fs − f = {_f(m.image_hz, '.6g', 'Hz')})",
          f"- Sıfır dereceli tutma düşüşü (içerik üst sınırında): {_f(m.zoh_droop_db, '.3f', 'dB')}"]
    if math.isfinite(m.filter_cutoff_hz):
        fc = m.filter_cutoff_hz
        r1 = 1e3
        c_rc = 1.0 / (2 * math.pi * r1 * fc)
        r2 = 10e3
        c2 = 1.0 / (2 * math.pi * fc * r2 * math.sqrt(2.0))
        L += [f"- Önerilen alçak geçiren köşe frekansı: **{fc:.4g} Hz** (içerik ile görüntünün geometrik ortası); "
              f"ilk görüntüyü {m.image_atten_db:g} dB bastırmak için **{m.filter_order}. derece Butterworth** (tahmin).",
              f"- 1. derece RC örneği: R = 1 kΩ, C = {c_rc * 1e9:.4g} nF (en yakın standart değere yuvarlayın).",
              f"- 2. derece Sallen-Key (birim kazanç, Butterworth Q = 0.707): R1 = R2 = 10 kΩ, C2 (toprağa) = "
              f"{c2 * 1e9:.4g} nF, C1 (geri besleme) = {2 * c2 * 1e9:.4g} nF."]
    else:
        L += ["- İçerik fs/2'ye kadar uzanıyor (keskin kenarlı dalga ya da gürültü): görüntüler içerikle örtüşür. "
              "Korunması gereken en yüksek harmoniğe göre köşe seçin ya da fs'yi artırın."]
    sr_need = math.pi * m.content_max_hz * m.vpp if math.isfinite(m.content_max_hz) else math.nan
    L += ["", "## Tamponlama / op-amp", "",
          f"- Ardışık örnekler arası en büyük sıçrama {m.max_step_v * 1e3:.4g} mV → filtresiz çıkışta "
          f"{m.max_slew_v_per_s / 1e6:.4g} V/µs basamak eğimi; filtre sonrası gereken eğim hızı ≈ π · f_içerik · Vpp "
          f"= {_f(sr_need / 1e6, '.3g', 'V/µs')}. Op-amp eğim hızı (SR) bunun en az 2–3 katı, kazanç-bant genişliği "
          "çarpımı filtre köşesinin ≥ 10 katı olsun.",
          "- DAC çıkışını doğrudan düşük empedanslı yüke bağlamayın; gerilim izleyici (buffer) kullanın."]
    if tgt == "stm32_hal_dma":
        L += ["- STM32 dahili DAC tamponu (Output Buffer = Enable) tipik olarak ≥ 5 kΩ / ≤ 50 pF yük sürer ve çıkış "
              "≈ 0.2 V … VDDA − 0.2 V aralığında doğrusaldır (F4 veri sayfası; ailenize göre doğrulayın). Tam 0 V / "
              "VREF gerekiyorsa tamponu kapatıp harici rail-to-rail op-amp kullanın."]
    L += ["", "## Bellek ve CPU maliyeti", "",
          f"- Tablo: **{mem['table_flash_bytes']} bayt flash** (RAM'e alınırsa aynı miktar RAM).",
          f"- RAM: {mem['ram_bytes']} bayt (durum {mem['state_ram_bytes']} B).",
          f"- Kod boyutu: ≈ {mem['code_size_estimate_bytes']} bayt (**tahmin**; kesin değer için derlenmiş nesne "
          "dosyasının boyutuna bakın)."]
    cyc = mem["cpu_cycles_per_sample_estimate"]
    if cyc == 0:
        L.append("- CPU: DMA ile ≈ %0 (örnek başına yalnızca veri yolu çevrimleri).")
    else:
        L.append(f"- CPU: ≈ {cyc:.0f} çevrim/örnek (**tahmin**) → {ctx.cpu_clock_hz / 1e6:.4g} MHz çekirdekte "
                 f"≈ %{mem['cpu_load_pct_estimate']:.2f} yük (fs = {t.fs:.6g} Hz).")
    for n in mem["notes"]:
        L.append(f"- {n}")
    L += ["", "## Ölçümler (nicemlenmiş çıkış)", "", "| Ölçüm | Değer |", "|---|---|",
          f"| Vmin / Vmax / Vpp | {m.vmin:.4g} / {m.vmax:.4g} / {m.vpp:.4g} V |",
          f"| Ortalama (DC) / RMS / AC RMS | {m.mean:.4g} / {m.rms:.4g} / {m.ac_rms:.4g} V |",
          f"| Tepe faktörü (AC) | {_f(m.crest_factor, '.4g')} |",
          f"| THD (ilk 10 harmonik) | {_f(m.thd_pct, '.4g', '%')} ({_f(m.thd_db, '.2f', 'dB')}) |",
          f"| SFDR | {_f(m.sfdr_dbc, '.1f', 'dBc')} |",
          f"| SNR / SINAD / ENOB | {_f(m.snr_db, '.1f', 'dB')} / {_f(m.sinad_db, '.1f', 'dB')} / "
          f"{_f(m.enob, '.2f', 'bit')} |",
          f"| Nicemleme SNR'si | {_f(m.quant_snr_db, '.1f', 'dB')} |",
          f"| Analiz | {m.analysis} |", ""]
    for n in m.notes:
        L.append(f"- {n}")
    L += ["", "## Sınırlamalar", "",
          "- Ölçümler ideal DAC modeline göredir (INL/DNL, kazanç/ofset hatası, yerleşme süresi ve glitch dahil "
          "değil). Gerçek çıkışı osiloskop / spektrum analizörüyle doğrulayın.",
          "- Zamanlayıcı saati (TIM_CLK) ve kristal toleransı frekansı doğrudan etkiler (tipik ±20–50 ppm)."]
    if tgt == "stm32_hal_dma":
        L += ["- STM32 DAC güncelleme hızı aileye bağlıdır (F4: ≈ 1 MS/s üst sınır, yerleşme ≈ 3–6 µs); veri "
              "sayfanıza bakın.",
              "- Bazı ailelerde (ör. H7) DMA'nın erişebildiği bellek bölgeleri ve önbellek tutarlılığı dikkat ister; "
              f"gerekirse `-D{P}_TABLE_IN_RAM` ve uygun bölge."]
    if tgt == "dds":
        L += [f"- Faz kesme spurları ≈ −{6.02 * (t.table_bits or 0):.0f} dBc (ara değerleme ile belirgin biçimde "
              "daha düşük). fs/f küçüldükçe (periyot başına az örnek) bozulma artar."]
    warns = list(t.warnings) + ctx.notes
    if warns:
        L += ["", "## Uyarılar", ""] + [f"- {w}" for w in warns]
    L.append("")
    return "\n".join(L)


def _usage_target(ctx: _Ctx) -> list[str]:
    t, dac, P, ident, tgt, tc = ctx.table, ctx.dac, ctx.P, ctx.ident, ctx.target, ctx.tc
    if tgt == "stm32_hal_dma":
        assert tc is not None
        stereo = t.channels == 2
        width = ("Word / Word" if dac.bits == 12 else "Half Word / Half Word") if stereo else \
            ("Half Word / Half Word" if dac.bits == 12 else "Byte / Byte")
        align = {(12, "right"): "DAC_ALIGN_12B_R", (12, "left"): "DAC_ALIGN_12B_L",
                 (8, "right"): "DAC_ALIGN_8B_R"}[(dac.bits, dac.align)]
        pin = "PA4 (OUT1) ve PA5 (OUT2)" if stereo else ("PA4 (OUT1)" if ctx.dac_channel == 1 else "PA5 (OUT2)")
        return [
            "### STM32CubeMX ayarları", "",
            f"1. **DAC**: {'OUT1 ve OUT2' if stereo else f'OUT{ctx.dac_channel}'} → \"Connected to external pin only\""
            f" ({pin}). Output Buffer: Enable. Trigger: **Timer 6 Trigger Out event**.",
            f"2. **TIM6**: Activated. Prescaler (PSC) = **{tc.psc}**, Counter Period (ARR) = **{tc.arr}**, "
            "auto-reload preload = Enable, Trigger Event Selection (TRGO) = **Update Event**. "
            f"(TIM6 saati {tc.timer_clock_hz:.9g} Hz varsayıldı; Clock Configuration'daki \"APB1 Timer clocks\" "
            "farklıysa kodu doğru `timer_clock_hz` ile yeniden üretin.)",
            f"3. **DMA** (DAC → DMA Settings → Add, {'yalnızca DAC_CH1' if stereo else f'DAC_CH{ctx.dac_channel}'}): "
            "Direction = Memory To Peripheral, Mode = **Circular**, Increment Address = Memory, "
            f"Data Width (Peripheral / Memory) = **{width}**.",
            "4. Generate Code. `" + f"{ident}.h` ve `{ident}.c` dosyalarını Core/Inc ve Core/Src'ye ekleyin.",
            "",
            "### Kod", "",
            f"`main.c` → USER CODE BEGIN 2: `{ident}_start(&hdac, &htim6);` (F4/F7: `hdac`, G4/L4/H7: `hdac1`).",
            "",
            f"- Hizalama sabiti: `{align}` ({dac.bits} bit, {'sola' if dac.align == 'left' else 'sağa'} hizalı; "
            f"tablo elemanı `{dac.c_type}`).",
            f"- `{ident}_start` PSC/ARR'yi `{P}_TIM_PSC` / `{P}_TIM_ARR` ile yazar, DMA'yı dairesel başlatır, "
            "TIM6'yı çalıştırır. Durdurmak için `" + f"{ident}_stop`.",
            "- HAL başlığı varsayılan olarak `main.h`; değiştirmek için `-D" + f"{P}_HAL_HEADER=\\\"stm32g4xx_hal.h\\\"`.",
        ] + (["- Stereo: iki kanal ikili veri yazmacına (`" + {(12, "right"): "DHR12RD", (12, "left"): "DHR12LD",
                                                             (8, "right"): "DHR8RD"}[(dac.bits, dac.align)]
              + "`) tek DMA akışıyla yazılır; bu kısım HAL_DAC_Start_DMA kullanmaz ve ailenize göre doğrulanmalıdır."]
             if stereo else [])
    if tgt == "generic_isr":
        lines = [
            f"1. `dac_write{'_stereo' if t.channels == 2 else ''}()` fonksiyonunu yazın: kodu DAC'a gönderir "
            f"(dahili DAC yazmacı, SPI/I2C DAC …). Kod tipi `{ident}_word_t` = `{dac.c_type}`.",
            f"2. Bir zamanlayıcıyı **fs = {t.fs:.9g} Hz** kesme üretecek şekilde kurun.",
            f"3. Kesme fonksiyonunda `{ident}_isr_tick();` çağırın (önce DAC'a yazar, sonra indeksi ilerletir).",
            "",
            "Kesme gecikmesindeki dalgalanma (jitter) çıkışta faz gürültüsü/spur olarak görünür; kesmeyi yüksek "
            "öncelikli yapın ve DAC'ı tetikli (zamanlayıcı tetikli DAC yazmacı) kullanabiliyorsanız tercih edin.",
        ]
        if tc is not None:
            lines.insert(2, f"   STM32 tarzı zamanlayıcıda TIM_CLK = {tc.timer_clock_hz:.9g} Hz için PSC = {tc.psc}, "
                            f"ARR = {tc.arr} (`{P}_TIM_PSC`, `{P}_TIM_ARR`).")
        return lines
    # dds
    lines = [
        f"1. `{ident}.h` / `{ident}.c` dosyalarını projeye ekleyin.",
        f"2. Bir üreteç tanımlayın ve başlatın: `static {ident}_dds_t g; {ident}_dds_init(&g);` "
        f"(tasarım frekansı {t.f_real:.9g} Hz).",
        f"3. fs = {t.fs:.9g} Hz zamanlayıcı kesmesinde: `dac_write({ident}_dds_next(&g));`.",
        f"4. Frekansı istediğiniz an değiştirin: `{ident}_dds_set_freq(&g, 1234.5);`.",
        "",
        f"Ara değerleme: `-D{P}_DDS_INTERP=1` (ya da üretimde interpolate=True) ile komşu iki seviye arasında "
        f"{dds_frac_bits(t.table_bits or 0)} bit kesirle doğrusal ara değer alınır (faz kesme spurları azalır, "
        "≈ 15 çevrim ek maliyet).",
    ]
    if tc is not None:
        lines.insert(3, f"   STM32 tarzı zamanlayıcıda PSC = {tc.psc}, ARR = {tc.arr} (TIM_CLK = "
                        f"{tc.timer_clock_hz:.9g} Hz).")
    return lines


# ============================================================================ ana giriş
def generate(spec: WaveSpec, dac: DacSpec, table: Table, target: str, name: str = "wave", *,
             fs: float | None = None, timer_clock_hz: float = 84e6, timer_bits: int = 16,
             cpu_clock_hz: float = 168e6, dac_channel: int = 1, interpolate: bool = False,
             image_atten_db: float = 40.0) -> list[GeneratedFile]:
    """Hedef için dosyaları üretir: `<ad>.h`, `<ad>.c`, `<ad>_ornek.c`, `KULLANIM.md`.

    spec / dac, tabloyu üreten nesnelerle aynı olmalıdır (denetlenir). fs verilirse tablonun fs'iyle aynı
    olmalıdır. timer_clock_hz / timer_bits: STM32 tarzı PSC/ARR hesabı (stm32_hal_dma, generic_isr, dds).
    dac_channel: STM32 DAC kanalı (1/2, mono). interpolate: DDS'te doğrusal ara değerleme varsayılanı.
    """
    if target not in TARGETS:
        raise SignalError(f"Bilinmeyen hedef '{target}'; geçerliler: {', '.join(TARGETS)}.")
    if table.spec != spec:
        raise SignalError("Tablo bu WaveSpec ile üretilmemiş; make_table'ı yeniden çağırın.")
    if table.dac != dac:
        raise SignalError("Tablo bu DacSpec ile üretilmemiş; make_table'ı yeniden çağırın.")
    if fs is not None and abs(float(fs) - table.fs) > 1e-9 * table.fs:
        raise SignalError(f"fs ({fs:g} Hz) tablonun fs'inden ({table.fs:g} Hz) farklı; tabloyu bu fs ile üretin.")
    ident = _ident(name)
    notes: list[str] = []
    tc = timer_config(table.fs, timer_clock_hz, timer_bits)
    if abs(tc.error_ppm) > 1.0:
        f_out = table.f_real * tc.fs_real / table.fs
        notes.append(f"Zamanlayıcı fs'yi tam üretemiyor: {tc.fs_real:.9g} Hz ({tc.error_ppm:+.2f} ppm) → çıkış "
                     f"frekansı {f_out:.9g} Hz olur. Tabloyu make_table(..., fs=timer_config(...).fs_real) ile "
                     "üretmek frekansı tam tutturur.")
    if target == "stm32_hal_dma" and table.fs > 1e6:
        notes.append("fs > 1 MHz: çoğu STM32 DAC'ının güncelleme hızını aşar (veri sayfasına bakın).")
    m = metrics(table, image_atten_db=image_atten_db, dds_interpolate=interpolate and table.mode == "dds")
    ctx = _Ctx(ident=ident, P=ident.upper(), spec=spec, dac=dac, table=table, target=target, m=m, tc=tc,
               interpolate=bool(interpolate), dac_channel=int(dac_channel), cpu_clock_hz=float(cpu_clock_hz),
               timer_bits=int(timer_bits), notes=notes)
    _validate(ctx)
    header = GeneratedFile(f"{ident}.h", _header(ctx))
    source = GeneratedFile(f"{ident}.c", _source(ctx))
    example = _example(ctx)
    mem = memory_report(table, dac, target, interpolate=interpolate, cpu_clock_hz=cpu_clock_hz)
    files = [header.name, source.name, example.name, "KULLANIM.md"]
    usage = GeneratedFile("KULLANIM.md", _usage(ctx, files, mem))
    return [header, source, example, usage]
