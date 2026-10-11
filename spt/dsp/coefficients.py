"""Katsayı kuantizasyonu (float32 / Q31 / Q15) ve dışa aktarma (CSV, C başlık dosyası, CMSIS-DSP).

Sabit noktada IIR katsayıları:
  - L∞ bölüm ölçeklemesi (isteğe bağlı, sabit noktada varsayılan): her bölümün çıkışında |H| tepe değeri 1
    olacak şekilde pay (b) katsayıları ölçeklenir; toplam kazanç değişmez. Ara düğümlerde taşma önlenir ve
    scipy'nin tüm kazancı ilk bölüme topladığı yüksek dereceli tasarımlarda (ilk bölüm b ≈ 1e-20) katsayılar
    Q15'te sıfıra yuvarlanmaz.
  - Q15 / Q31'de [−1, 1) dışına taşan katsayılar (ör. a1 ≈ −1.9) CMSIS-DSP'deki gibi tek bir `postShift` ile
    saklanır: kod = round(katsayı · 2^(B−1) / 2^postShift); filtre çıkışı 2^postShift ile çarpılır.

Sabit noktada FIR katsayıları da aynı kuralla saklanır (ör. türev alıcı / yüksek kazançlı FIR'de |b| ≫ 1):
CMSIS-DSP'nin FIR fonksiyonlarında kaydırma parametresi olmadığından çıkış, uygulamada 2^postShift ile
(doygunlukla sola kaydırarak) çarpılmalıdır. postShift = 0 ise hiçbir şey gerekmez.

CMSIS-DSP biquad düzeni: her bölüm {b0, b1, b2, a1, a2}; a1 ve a2 ters işaretlidir
(y[n] = b0·x[n] + b1·x[n−1] + b2·x[n−2] + a1·y[n−1] + a2·y[n−2]). Q15'te her bölüm 6 eleman:
{b0, 0, b1, b2, a1, a2}. CMSIS FIR katsayıları zamanda ters sıradadır: {b[N−1], …, b[0]}.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from spt import APP_NAME
from spt.dsp.filters import Design, FilterError, _response_at, analyze, mask_for

NUMBER_FORMATS = {"float64": "float64 (double)", "float32": "float32", "q31": "Q31 (int32)", "q15": "Q15 (int16)"}
FIXED_BITS = {"q15": 16, "q31": 32}
IIR_STRUCTURES = {"df1": "Direct Form I", "df2t": "Direct Form II Transposed"}


@dataclass(frozen=True, eq=False)
class Quantized:
    fmt: str
    sos: np.ndarray | None          # kuantize edilmiş gerçek değerler (standart işaret, a0 = 1)
    b: np.ndarray | None
    codes: np.ndarray | None        # tamsayı kodlar — IIR: (k, 5) [b0, b1, b2, −a1, −a2]; FIR: (N,)
    post_shift: int = 0
    clipped: int = 0
    scaled: bool = False

    def as_design(self, d: Design) -> Design:
        return Design(d.spec, sos=self.sos, b=self.b, order=d.order)


def _section_response(sec: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Tek ikinci derece bölümün cevabı (birim çemberdeki kutupta inf / NaN: kuantize kararsız bölüm)."""
    e1 = np.exp(-1j * w)
    e2 = e1 * e1
    with np.errstate(divide="ignore", invalid="ignore"):
        return (sec[0] + sec[1] * e1 + sec[2] * e2) / (sec[3] + sec[4] * e1 + sec[5] * e2)


def scale_sections(sos: np.ndarray, points: int | None = None) -> np.ndarray:
    """L∞ bölüm ölçeklemesi: i. bölümün çıkışındaki birikimli cevabın tepe değeri 1 olur; son bölüm toplam
    kazancı geri getirir (ölçek çarpanlarının çarpımı 1). Sinüzoidal kararlı hâl için taşmayı önler; geniş bantlı
    en kötü girdi (ℓ1 normu) daha büyük olabilir.

    Izgara sıklığı kutupların birim çembere uzaklığına göre seçilir (dar bantlı bölümlerin tepesi kaçmasın)."""
    sos = np.array(sos, dtype=np.float64)
    if len(sos) < 2:
        return sos
    if points is None:
        radius = max((float(np.abs(np.roots(sec[3:])).max()) if np.any(sec[4:]) else 0.0) for sec in sos)
        # çok bölümlü kaskadda (ör. 480 bölümlü tarak) bellek / süre sınırı; tepe yine de ara değerlemeyle bulunur
        top = 1 << 20 if len(sos) <= 24 else 1 << 18 if len(sos) <= 100 else 1 << 16
        points = int(np.clip(16 / max(1 - radius, 1e-9), 4096, top))
    w = np.linspace(0, np.pi, points)
    original = sos.copy()
    cum = np.ones(points, dtype=complex)  # özgün kaskadın birikimli cevabı: bölüm başına bir çarpım (O(K))
    exact_refine = len(sos) <= 24
    previous = 1.0
    for i in range(len(sos)):
        if i < len(sos) - 1:
            cum *= _section_response(original[i], w)
            mag = np.abs(cum)
            k = int(np.argmax(mag))
            if exact_refine:  # tepe ızgara noktaları arasında kalmasın: çevresinde ince değerlendirme
                fine = np.linspace(w[max(k - 1, 0)], w[min(k + 1, len(w) - 1)], 257)
                fine_cum = np.ones(len(fine), dtype=complex)
                for sec in original[:i + 1]:
                    fine_cum *= _section_response(sec, fine)
                peak = float(max(mag[k], np.abs(fine_cum).max()))
            else:  # çok bölümde O(K²) ince arama yerine dB ölçeğinde parabolik ara değerleme
                peak = float(mag[k])
                if 0 < k < len(mag) - 1 and mag[k - 1] > 0 and mag[k + 1] > 0:
                    y0, y1, y2 = np.log(mag[k - 1]), np.log(mag[k]), np.log(mag[k + 1])
                    den = y0 - 2 * y1 + y2
                    if den < 0:
                        peak = float(np.exp(y1 - 0.125 * (y2 - y0) ** 2 / den))
            target = 1.0 / peak if peak > 0 else 1.0
        else:
            target = 1.0
        sos[i, :3] *= target / previous
        previous = target
    return sos


def _fixed_codes(values: np.ndarray, fmt: str) -> tuple[np.ndarray, int]:
    """Katsayıları B bitlik koda çevirir: en büyük |katsayı| / 2^postShift < 1 olacak en küçük postShift."""
    one = 2.0 ** (FIXED_BITS[fmt] - 1)
    peak, shift = float(np.abs(values).max()) if values.size else 0.0, 0
    while peak / 2.0 ** shift > (one - 1) / one:
        shift += 1
    codes = np.clip(np.round(values * (one / 2.0 ** shift)), -one, one - 1).astype(np.int64)
    return codes, shift


@lru_cache(maxsize=16)  # tasarımlar değişmez: aynı tasarım / biçim için (dışa aktarma + rapor) bir kez
def quantize(d: Design, fmt: str, scale: bool | None = None) -> Quantized:
    if not d.is_linear:
        raise FilterError("Medyan filtre doğrusal değildir; katsayısı yoktur")
    if fmt not in NUMBER_FORMATS:
        raise FilterError(f"Bilinmeyen sayı biçimi: {fmt}")
    fixed = fmt in FIXED_BITS
    if d.kind == "iir":
        scale = fixed if scale is None else scale
        sos = scale_sections(d.sos) if scale else np.array(d.sos, dtype=np.float64)
        sos = sos / sos[:, 3:4]  # a0 = 1 (scipy zaten böyle üretir)
        if fmt == "float64":
            return Quantized(fmt, sos, None, None, scaled=scale)
        if fmt == "float32":
            q = sos.astype(np.float32).astype(np.float64)
            q[:, 3] = 1.0
            return Quantized(fmt, q, None, None, scaled=scale)
        one = 2.0 ** (FIXED_BITS[fmt] - 1)
        coeffs = np.column_stack((sos[:, 0], sos[:, 1], sos[:, 2], -sos[:, 4], -sos[:, 5]))
        codes, shift = _fixed_codes(coeffs, fmt)
        real = codes * (2.0 ** shift / one)
        qsos = np.column_stack((real[:, :3], np.ones(len(real)), -real[:, 3], -real[:, 4]))
        return Quantized(fmt, qsos, None, codes, shift, 0, scale)
    b = np.asarray(d.b, dtype=np.float64)
    if fmt == "float64":
        return Quantized(fmt, None, b.copy(), None)
    if fmt == "float32":
        return Quantized(fmt, None, b.astype(np.float32).astype(np.float64), None)
    one = 2.0 ** (FIXED_BITS[fmt] - 1)
    codes, shift = _fixed_codes(b, fmt)
    return Quantized(fmt, None, codes * (2.0 ** shift / one), codes, shift)


@dataclass(frozen=True)
class QuantReport:
    pass_dev_db: float               # geçirme bandında (ideal kazancın −3 dB'lik bölgesi) en büyük |Δ|
    dev_db: float                    # ideal kazancın −60 dB'ye kadar olan bölgesinde en büyük |Δ|
    stop_peak_db: float | None       # durdurma bandı tepesi (kuantize) — maske ya da bastırma hedefi varsa
    ideal_stop_peak_db: float | None
    max_pole_radius: float
    clipped: int

    @property
    def stable(self) -> bool:
        return self.max_pole_radius < 1.0


def _magnitude_db(d: Design, freqs: np.ndarray) -> np.ndarray:
    """Yalnız genlik (dB); rapor grup gecikmesine ihtiyaç duymaz. IIR'de bölümler vektörel çarpılır."""
    if d.kind == "iir":
        w = 2 * np.pi * np.asarray(freqs, dtype=np.float64) / d.fs
        h = np.ones(len(w), dtype=complex)
        for sec in d.sos:
            h *= _section_response(sec, w)
    else:
        h = _response_at(d, freqs)
    with np.errstate(invalid="ignore"):
        return 20 * np.log10(np.maximum(np.abs(h), 1e-15))


def quantization_report(d: Design, q: Quantized) -> QuantReport:
    ideal = analyze(d)
    quant_db = _magnitude_db(q.as_design(d), ideal.freqs)
    top = ideal.magnitude_db.max()
    passband = ideal.magnitude_db >= top - 3.0
    visible = ideal.magnitude_db >= top - max(60.0, d.spec.atten_db)
    mask = mask_for(d.spec)
    stop = None
    if mask is not None:
        stop = np.zeros_like(passband)
        for lo, hi in mask.stopbands:
            stop |= (ideal.freqs >= lo) & (ideal.freqs <= hi)
    elif d.spec.family == "iir" and d.spec.method in ("cheby2", "ellip"):
        stop = ideal.magnitude_db <= -d.spec.atten_db + 0.5  # eşdalgalı durdurma bandı (geçiş bölgesi hariç)
    has_stop = stop is not None and bool(stop.any())
    radius = 0.0
    if q.sos is not None:
        radius = max(float(np.abs(np.roots(sec[3:])).max()) if np.any(sec[4:]) else 0.0 for sec in q.sos)
    return QuantReport(
        float(np.abs(quant_db[passband] - ideal.magnitude_db[passband]).max()),
        float(np.abs(quant_db[visible] - ideal.magnitude_db[visible]).max()),
        float(quant_db[stop].max()) if has_stop else None,
        float(ideal.magnitude_db[stop].max()) if has_stop else None,
        radius, q.clipped)


# ============================================================================ dışa aktarma
def _identifier(name: str) -> str:
    ident = re.sub(r"[^0-9A-Za-z_]", "_", name.strip()) or "flt"
    return f"_{ident}" if ident[0].isdigit() else ident


def _c_number(value: float, fmt: str) -> str:
    if fmt in FIXED_BITS:
        return str(int(value))
    text = f"{value:.9g}" if fmt == "float32" else f"{value:.17g}"
    if not any(c in text for c in ".en"):
        text += ".0"
    return text + ("f" if fmt == "float32" else "")


def _rows(values: list[str], per_line: int, indent: str = "    ") -> list[str]:
    return [indent + ", ".join(values[i:i + per_line]) + "," for i in range(0, len(values), per_line)]


def _summary_lines(d: Design, q: Quantized) -> list[str]:
    s = d.spec
    lines = [f"{d.describe()}", f"fs = {s.fs:g} Hz · {d.structure_text()}"]
    if q.fmt in FIXED_BITS:
        bits = FIXED_BITS[q.fmt] - 1
        lines.append(f"Biçim: {NUMBER_FORMATS[q.fmt]}, postShift = {q.post_shift} "
                     f"(katsayı = kod · 2^{q.post_shift} / 2^{bits})")
        if d.kind == "fir" and q.post_shift:
            lines.append(f"DİKKAT: FIR çıkışı 2^{q.post_shift} ile çarpılmalı (doygunlukla {q.post_shift} bit sola "
                         "kaydırın); CMSIS-DSP FIR fonksiyonlarında kaydırma parametresi yoktur")
    else:
        lines.append(f"Biçim: {NUMBER_FORMATS[q.fmt]}")
    if q.scaled:
        lines.append("Bölüm ölçekleme: L∞ (her bölüm çıkışında |H| ≤ 1, toplam kazanç korunur)")
    if q.clipped:
        lines.append(f"UYARI: {q.clipped} katsayı [−1, 1) aralığına sığmadığı için kırpıldı")
    return lines


def to_csv(d: Design, fmt: str = "float64", scale: bool | None = None) -> str:
    q = quantize(d, fmt, scale)
    out = [f"# {APP_NAME} — filtre katsayıları"] + [f"# {line}" for line in _summary_lines(d, q)]
    fixed = fmt in FIXED_BITS
    if d.kind == "iir":
        if fixed:
            out.append("# Satır başına bir ikinci derece bölüm, standart işaret: "
                       "y = b0·x + b1·x1 + b2·x2 − a1·y1 − a2·y2 (a0 = 1)")
            out.append("b0,b1,b2,a1,a2")
            for c in q.codes:
                out.append(",".join(str(int(v)) for v in (c[0], c[1], c[2], -c[3], -c[4])))
        else:
            out.append("# Satır başına bir ikinci derece bölüm (scipy / MATLAB SOS düzeni)")
            out.append("b0,b1,b2,a0,a1,a2")
            for row in q.sos:
                out.append(",".join(_c_number(v, "float64") for v in row))
    else:
        out.append("n,b")
        values = q.codes if fixed else q.b
        for i, v in enumerate(values):
            out.append(f"{i},{int(v) if fixed else _c_number(v, 'float64')}")
    return "\n".join(out) + "\n"


_CMSIS_TYPES = {"float32": "float32_t", "float64": "float64_t", "q31": "q31_t", "q15": "q15_t"}
_PLAIN_TYPES = {"float32": "float", "float64": "double", "q31": "int32_t", "q15": "int16_t"}


def to_c(d: Design, fmt: str = "float32", cmsis: bool = True, structure: str = "df1", name: str = "flt",
         scale: bool | None = None, block_size: int = 32) -> str:
    """C başlık dosyası. `cmsis=True`: CMSIS-DSP yapıları ve katsayı düzeni; aksi halde düz C dizileri."""
    q = quantize(d, fmt, scale)
    ident = _identifier(name)
    macro = ident.upper()
    ctype = (_CMSIS_TYPES if cmsis else _PLAIN_TYPES)[fmt]
    head = [f" * {line}" for line in _summary_lines(d, q)]
    body: list[str] = []
    usage: list[str] = []

    if d.kind == "iir":
        k = len(q.sos)
        if cmsis:
            if fmt == "float64":
                structure = "df2t"  # CMSIS-DSP'de çift hassasiyetli biquad yalnızca DF2T
            if fmt in FIXED_BITS:
                structure = "df1"   # CMSIS-DSP sabit nokta biquad yalnızca DF1
            if fmt in FIXED_BITS:
                codes = q.codes
                per = 6 if fmt == "q15" else 5
                values = []
                for c in codes:
                    values += ([c[0], 0, c[1], c[2], c[3], c[4]] if fmt == "q15" else list(c))
                values = [_c_number(v, fmt) for v in values]
                order_note = "{b0, 0, b1, b2, a1, a2}" if fmt == "q15" else "{b0, b1, b2, a1, a2}"
            else:
                per = 5
                values = [_c_number(v, fmt) for row in q.sos for v in (row[0], row[1], row[2], -row[4], -row[5])]
                order_note = "{b0, b1, b2, a1, a2}"
            head.append(f" * Yapı: CMSIS-DSP biquad {IIR_STRUCTURES[structure]} · {k} bölüm")
            head.append(f" * Bölüm başına {order_note}; a1, a2 CMSIS kuralı gereği TERS İŞARETLİ:")
            head.append(" *   y[n] = b0·x[n] + b1·x[n−1] + b2·x[n−2] + a1·y[n−1] + a2·y[n−2]")
            body.append(f"#define {macro}_NUM_STAGES {k}")
            if fmt in FIXED_BITS:
                body.append(f"#define {macro}_POST_SHIFT {q.post_shift}")
            body.append("")
            body.append(f"static const {ctype} {ident}_coeffs[{per} * {macro}_NUM_STAGES] = {{")
            for i in range(k):
                body.append(f"    /* bölüm {i + 1} */ " + ", ".join(values[i * per:(i + 1) * per]) + ",")
            body.append("};")
            suffix = {"float32": "f32", "float64": "f64", "q31": "q31", "q15": "q15"}[fmt]
            if structure == "df2t":
                inst, init, run, state = (f"arm_biquad_cascade_df2T_instance_{suffix}",
                                          f"arm_biquad_cascade_df2T_init_{suffix}",
                                          f"arm_biquad_cascade_df2T_{suffix}", f"2 * {macro}_NUM_STAGES")
            else:
                inst, init, run, state = (f"arm_biquad_casd_df1_inst_{suffix}", f"arm_biquad_cascade_df1_init_{suffix}",
                                          f"arm_biquad_cascade_df1_{suffix}", f"4 * {macro}_NUM_STAGES")
            shift_arg = f", {macro}_POST_SHIFT" if fmt in FIXED_BITS else ""
            usage = [
                f"static {ctype} {ident}_state[{state}];",
                f"static {inst} {ident};",
                f"{init}(&{ident}, {macro}_NUM_STAGES, {ident}_coeffs, {ident}_state{shift_arg});",
                f"{run}(&{ident}, src, dst, blockSize);",
            ]
        else:
            if fmt in FIXED_BITS:
                rows = [[_c_number(v, fmt) for v in (c[0], c[1], c[2], -c[3], -c[4])] for c in q.codes]
                cols, layout = 5, "{b0, b1, b2, a1, a2}"
            else:
                rows = [[_c_number(v, fmt) for v in row] for row in q.sos]
                cols, layout = 6, "{b0, b1, b2, a0, a1, a2}"
            head.append(f" * Yapı: {k} ikinci derece bölüm (SOS), satır başına {layout}, standart işaret:")
            head.append(" *   y[n] = b0·x[n] + b1·x[n−1] + b2·x[n−2] − a1·y[n−1] − a2·y[n−2]")
            body.append(f"#define {macro}_NUM_SECTIONS {k}")
            if fmt in FIXED_BITS:
                body.append(f"#define {macro}_POST_SHIFT {q.post_shift}")
            body.append("")
            body.append(f"static const {ctype} {ident}_sos[{macro}_NUM_SECTIONS][{cols}] = {{")
            for row in rows:
                body.append("    {" + ", ".join(row) + "},")
            body.append("};")
    else:
        codes = q.codes if fmt in FIXED_BITS else q.b
        taps = list(codes)
        if cmsis:
            if fmt == "q15" and (len(taps) % 2 or len(taps) < 4):
                pad = (len(taps) % 2) + max(0, 4 - len(taps) - (len(taps) % 2))
                taps = taps + [0] * pad
                head.append(f" * Not: arm_fir_init_q15 çift ve ≥ 4 katsayı ister; sona {pad} sıfır eklendi "
                            "(cevap değişmez)")
            taps = taps[::-1]
            head.append(f" * Yapı: CMSIS-DSP FIR · {len(taps)} katsayı, ZAMANDA TERS SIRA: {{b[N−1], …, b[0]}}")
            suffix = {"float32": "f32", "float64": "f64", "q31": "q31", "q15": "q15"}[fmt]
            usage = [
                f"#define {macro}_BLOCK_SIZE {block_size}",
                f"static {ctype} {ident}_state[{macro}_NUM_TAPS + {macro}_BLOCK_SIZE - 1];",
                f"static arm_fir_instance_{suffix} {ident};",
                f"arm_fir_init_{suffix}(&{ident}, {macro}_NUM_TAPS, {ident}_coeffs, {ident}_state, "
                f"{macro}_BLOCK_SIZE);",
                f"arm_fir_{suffix}(&{ident}, src, dst, {macro}_BLOCK_SIZE);",
            ]
            array_name = f"{ident}_coeffs"
        else:
            head.append(f" * Yapı: FIR · {len(taps)} katsayı, doğal sıra: y[n] = Σ b[k]·x[n−k]")
            array_name = f"{ident}_b"
        body.append(f"#define {macro}_NUM_TAPS {len(taps)}")
        if fmt in FIXED_BITS:
            body.append(f"#define {macro}_POST_SHIFT {q.post_shift}  /* çıkış · 2^POST_SHIFT */")
            if cmsis and q.post_shift:
                usage.append(f"arm_shift_{fmt}(dst, {macro}_POST_SHIFT, dst, {macro}_BLOCK_SIZE);"
                             "  // doygunlukla 2^POST_SHIFT ile çarpar")
        body.append("")
        body.append(f"static const {ctype} {array_name}[{macro}_NUM_TAPS] = {{")
        body += _rows([_c_number(v, fmt) for v in taps], 8 if fmt in FIXED_BITS else 4)
        body.append("};")

    guard = f"{macro}_COEFFS_H"
    out = ["/*", f" * {APP_NAME} — filtre katsayıları", *head, " */",
           f"#ifndef {guard}", f"#define {guard}", ""]
    out.append('#include "arm_math.h"' if cmsis else "#include <stdint.h>")
    out.append("")
    out += body
    if usage:
        out += ["", "/* Kullanım:"] + [f" *   {line}" for line in usage] + [" */"]
    out += ["", f"#endif /* {guard} */", ""]
    return "\n".join(out)
