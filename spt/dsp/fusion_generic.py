"""Genel (senaryo tabanlı) sensör füzyonu — arayüz ve motorun kullanacağı tek giriş noktası.

Yalnızca IMU yönelimi değil, farklı sensör birleşimleri için "senaryo"lar tanımlar. Her senaryo; kullanıcının kanal
eşleyeceği girişleri (`Input`), kestirilen çıktıları (`Output`), tüm algoritmaların paylaştığı parametreleri
(`Param` / `Choice`), sıralı algoritmalarını (ilki en basit taban çizgisi) ve gerçeği tam bilinen analitik
simülatörünü içerir.

Senaryolar (`SCENARIOS` sırasıyla)
  orientation  : Yönelim (IMU / MARG) — `spt.dsp.fusion` algoritmalarının uyarlayıcısı (roll, pitch, yaw)
  altitude     : İrtifa: barometre + ivmeölçer (irtifa, dikey hız)
  position2d   : 2B konum: GNSS + jiroskop + hız / odometri (x, y, başlık, hız)
  linear_multi : Doğrusal çok sensörlü Kalman, 1–6 sensör (konum, hız, ivme)
Modeller, denklemler ve varsayımlar her senaryonun modül belgesindedir (`spt.dsp.fusion_scenarios.*`).

Ortak kurallar
  - `Estimator.run(inputs, dt)` girişleri SI'da alır: {giriş anahtarı: (n,) ya da (n, eksen)} dizileri.
    NaN = "bu örnekte ölçüm yok" (seyrek sensör). Çıktı (n, len(outputs)) float64; başlatılmamış örnekler NaN.
  - Açısal çıktılar radyandır (`Output.angle`); gösterimde `Output.display_scale` (= 180/π) ile çarpın.
    `errors` açı hatalarını ±π'ye sarar.
  - Bloklar halinde çağrı tek seferlik çağrıyla birebir aynı sonucu verir (parçacık filtreleri dahil).
  - Geçersiz parametre / eksik giriş: Türkçe `FusionError` (kullanıcıya doğrudan gösterilebilir).

Motor / arayüz entegrasyonu (öneri)
    scen = SCENARIOS["altitude"]
    mapping = auto_map(scen, desc)                       # ya da kullanıcının elle yaptığı Mapping
    scen.check(mapping.inputs, scenario_values)          # başlatmadan önce doğrulama (FusionError)
    # işçi süreçte (algoritma başına): yalnızca anahtarlar ve sözlükler gönderilir (pickle dostu)
    est = make_estimator(scen.key, "kf", scenario_values, algo_values)
    for parça in akış:                                   # columns: (n, kanal) fiziksel değerler
        y = est.run(gather_inputs(scen, mapping, columns), 1 / fs)
    ref = gather_reference(scen, mapping, desc, columns) # (n, k) iç birimde, eşlenmeyen NaN
    err = errors(scen, y, ref); rmse(err)                # sonuç tablosu; est.diagnostics(), est.cost_us
    raw = scen.raw(x, raw_state)                         # gri "ham (füzyonsuz)" eğri, (n, k); raw_state:
                                                         # akış boyunca aynı sözlük (_p0 / _origin saklar)
  Akış kaybında (gap) motor, yönelimdeki gibi kestiriciyi yeniden oluşturabilir (`est.reset()` de olur).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from spt.dsp.fusion import Param
from spt.dsp.fusion_scenarios import altitude, linear_multi, orientation, position2d
from spt.dsp.fusion_scenarios.base import (
    D2R,
    Choice,
    Estimator,
    FusionError,
    Input,
    Output,
    Scenario,
    Simulation,
    name_tokens,
    normalize_unit,
    unit_key,
    wrap,
)

__all__ = [
    "Choice", "Estimator", "FusionError", "Input", "Mapping", "Output", "Param", "SCENARIOS", "Scenario",
    "Simulation", "auto_map", "describe", "errors", "gather_inputs", "gather_reference", "make_estimator", "name_tokens",
    "reference_scale", "rmse", "to_display",
]

SCENARIOS: dict[str, Scenario] = {s.key: s for s in (
    orientation.SCENARIO, altitude.SCENARIO, position2d.SCENARIO, linear_multi.SCENARIO)}


@dataclass(frozen=True)
class Mapping:
    """Kanal eşlemesi.

    inputs   : giriş anahtarı → kanal indeksleri (eksen sırasıyla; skalerde tek eleman)
    units    : giriş anahtarı → `Input.units` anahtarı (SI çarpanı `Input.units[...]`)
    reference: çıktı anahtarı → referans (gerçek değer) kanalı. Açısal çıktıda kanal birimi "rad" içermiyorsa
               derece kabul edilir (°, deg, boş); diğer çıktılar olduğu gibi (çıktının biriminde) kullanılır.
    values   : önerilen senaryo değerleri (ör. linear_multi'de birimden / addan çıkarılan yuva türleri ve model
               derecesi). Arayüz isterse senaryo formuna uygular; boş olabilir.
    """
    inputs: dict[str, tuple[int, ...]]
    units: dict[str, str]
    reference: dict[str, int]
    values: dict = field(default_factory=dict)


def _scenario(scenario: Scenario | str) -> Scenario:
    if isinstance(scenario, Scenario):
        return scenario
    try:
        return SCENARIOS[scenario]
    except KeyError:
        raise FusionError(f"Bilinmeyen senaryo: {scenario!r} (geçerli: {', '.join(SCENARIOS)})") from None


def make_estimator(scenario_key: str, algo_key: str, scenario_values: dict | None = None,
                   algo_values: dict | None = None) -> Estimator:
    """Senaryonun algoritmasını oluşturur (parametreler doğrulanır; hata → FusionError)."""
    scen = _scenario(scenario_key)
    cls = scen.algorithms.get(algo_key)
    if cls is None:
        raise FusionError(f"{scen.name}: bilinmeyen algoritma {algo_key!r} (geçerli: {', '.join(scen.algorithms)})")
    return cls(scenario_values or {}, **(algo_values or {}))


# ============================================================================ otomatik eşleme
_REF_WORDS = ("ref", "referans", "reference", "true", "truth", "gt", "gercek", "ground")
_AXIS_ALIASES = {"x": ("x", "east", "dogu"), "y": ("y", "north", "kuzey"), "z": ("z", "up", "yukari"),
                 "lat": ("lat", "enlem", "latitude"), "lon": ("lon", "lng", "boylam", "longitude", "long")}


@lru_cache(maxsize=None)
def _split_words(words: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(w.split()) for w in words)


def _tok_score(word: str, tok: str) -> float:
    """Tam eşleşme 1; ≥ 3 harfli sözcük kanal parçasının başıysa 0.75 ("press" ~ "pressure")."""
    if tok == word:
        return 1.0
    return 0.75 if len(word) >= 3 and tok.startswith(word) else 0.0


def _word_score(words: tuple[str, ...], toks: list[str]) -> float:
    """En iyi anahtar sözcüğün puanı: çok sözcüklü anahtarın ("acc z") tüm parçaları bulunmalı."""
    best = 0.0
    for parts in _split_words(words):
        total = 0.0
        for part in parts:
            m = max((_tok_score(part, t) for t in toks), default=0.0)
            if m == 0.0:
                total = 0.0
                break
            total += m
        best = max(best, total)
    return best


def _extended(toks: list[str]) -> list[str]:
    """Bitişik eksen soneklerini de ayırır: "accz" → ek olarak "acc", "z"."""
    extra = [p for t in toks if len(t) >= 3 and t[-1] in "xyz" and t[:-1].isalpha() for p in (t[:-1], t[-1])]
    return toks + extra


def _axis(toks: list[str], axes: tuple[str, ...]) -> tuple[int, list[str]] | None:
    """Vektör girişinde eksen: sondan başa eksen parçası ("x", "lat" …) ya da bitişik sonek ("ax", "gyrox")."""
    for pos in range(len(toks) - 1, -1, -1):
        for i, axis in enumerate(axes):
            if toks[pos] in _AXIS_ALIASES.get(axis, (axis,)):
                return i, toks
    last = toks[-1] if toks else ""
    if len(last) >= 2 and last[:-1].isalpha():
        for i, axis in enumerate(axes):
            if len(axis) == 1 and last[-1] == axis:
                return i, toks[:-1] + [last[:-1], last[-1]]
    return None


def _unit_bonus(units: dict[str, float], unit: str) -> tuple[float, str | None]:
    key = unit_key(units, unit)
    return (2.0 if key is not None else 0.0), key


def _output_unit_bonus(out: Output, unit: str) -> float:
    u = normalize_unit(unit)
    if not u:
        return 0.0
    if out.angle:
        return 2.0 if u in ("°", "rad") else 0.0
    return 2.0 if u == normalize_unit(out.unit) else 0.0


def auto_map(scenario: Scenario | str, desc) -> Mapping | None:
    """Kanal adlarından ve birimlerinden eşleme önerir; zorunlu girişler eşlenemezse None.

    Adlar normalleştirilir (NFKD, birleşik işaretler atılır, ı/İ → i, CamelCase bölünür, küçük harf):
    "İvme X", "ivme_x", "acc_x", "AccX", "ax", "GyroZ", "baro", "basinc", "pressure", "gps_x", "lat", "enlem",
    "hiz", "speed", "odom" … tanınır. Puan = anahtar sözcük eşleşmesi (+2 birim de girişin birimlerinden biriyse);
    her kanal en çok bir yuvaya, yüksek puandan başlayarak atanır. "ref", "true", "gt", "gercek" … içeren kanallar
    girişe değil, çıktının referansına eşlenir. Birim anahtarı kanal biriminden seçilir (°/s ↔ rad/s ↔ mdps,
    Pa / hPa / kPa / mbar, m, ° …); tanınmazsa girişin ilk (varsayılan) birimi.
    """
    scen = _scenario(scenario)
    plain: list[tuple[int, list[str], str]] = []
    refs: list[tuple[int, list[str], str]] = []
    for i, ch in enumerate(desc.channels):
        toks = name_tokens(ch.name)
        if not toks:
            continue
        rest = [t for t in toks if not any(_tok_score(w, t) for w in _REF_WORDS)]
        if len(rest) < len(toks):
            if rest:
                refs.append((i, rest, ch.unit or ""))
        else:
            plain.append((i, toks, ch.unit or ""))

    # ---- girişler: (puan, kanal, giriş sırası, eksen) adayları, açgözlü atama
    cands = []
    for order, inp in enumerate(scen.inputs):
        for ch, toks, unit in plain:
            if inp.scalar:
                score, axis = _word_score(inp.words, _extended(toks)), 0
            else:
                found = _axis(toks, inp.axes)
                if found is None:
                    continue
                axis, words_toks = found
                score = _word_score(inp.words, words_toks)
            if score > 0:
                cands.append((score + _unit_bonus(inp.units, unit)[0], ch, order, axis))
    cands.sort(key=lambda c: (-c[0], c[1], c[2], c[3]))
    used: set[int] = set()
    slots: dict[tuple[int, int], int] = {}
    for _, ch, order, axis in cands:
        if ch in used or (order, axis) in slots:
            continue
        slots[(order, axis)] = ch
        used.add(ch)
    inputs: dict[str, tuple[int, ...]] = {}
    units: dict[str, str] = {}
    for order, inp in enumerate(scen.inputs):
        idx = [slots.get((order, a)) for a in range(inp.width)]
        if any(i is None for i in idx):
            continue        # eksik eksenli vektör eşlenmez
        inputs[inp.key] = tuple(idx)
        units[inp.key] = unit_key(inp.units, desc.channels[idx[0]].unit or "") or next(iter(inp.units))
    if any(inp.required and inp.key not in inputs for inp in scen.inputs):
        return None
    if any(not set(group) & set(inputs) for group in scen.one_of):
        return None

    # ---- referans kanalları
    rcands = []
    for order, out in enumerate(scen.outputs):
        for ch, toks, unit in refs:
            score = _word_score(out.words, _extended(toks))
            if score > 0:
                rcands.append((score + _output_unit_bonus(out, unit), ch, order))
    rcands.sort(key=lambda c: (-c[0], c[1], c[2]))
    reference: dict[str, int] = {}
    rused: set[int] = set()
    for _, ch, order in rcands:
        key = scen.outputs[order].key
        if ch in rused or key in reference:
            continue
        reference[key] = ch
        rused.add(ch)
    reference = {o.key: reference[o.key] for o in scen.outputs if o.key in reference}
    values = scen.suggest(inputs, units, desc) if scen.suggest is not None else {}
    return Mapping(inputs, units, reference, values)


# ============================================================================ veri hazırlama ve değerlendirme
def gather_inputs(scenario: Scenario | str, mapping: Mapping, columns: np.ndarray) -> dict[str, np.ndarray]:
    """(n, kanal) fiziksel değerlerden eşlenmiş girişleri SI'da çıkarır (`Estimator.run` için)."""
    scen = _scenario(scenario)
    cols = np.asarray(columns, dtype=np.float64)
    if cols.ndim != 2:
        raise FusionError("Kanal verisi (n, kanal) biçiminde olmalı")
    out: dict[str, np.ndarray] = {}
    for key, idx in mapping.inputs.items():
        inp = scen.input(key)
        if len(idx) != inp.width:
            raise FusionError(f"'{inp.label}' için {inp.width} kanal gerekli, {len(idx)} verildi")
        if any(not 0 <= i < cols.shape[1] for i in idx):
            raise FusionError(f"'{inp.label}' için geçersiz kanal indeksi: {idx}")
        unit = mapping.units.get(key) or next(iter(inp.units))
        if unit not in inp.units:
            raise FusionError(f"'{inp.label}' için bilinmeyen birim {unit!r} (geçerli: {', '.join(inp.units)})")
        a = cols[:, list(idx)] * inp.units[unit]
        out[key] = a[:, 0].copy() if inp.scalar else a
    return out


def reference_scale(output: Output, unit: str) -> float:
    """Referans kanal değeri → iç birim çarpanı: açısal çıktıda "rad" yoksa derece kabul edilir."""
    if output.angle:
        return 1.0 if "rad" in normalize_unit(unit) else D2R
    return 1.0


def gather_reference(scenario: Scenario | str, mapping: Mapping, desc, columns: np.ndarray) -> np.ndarray:
    """(n, len(outputs)) referans değerleri iç birimde (açılar rad); eşlenmeyen çıktı sütunları NaN."""
    scen = _scenario(scenario)
    cols = np.asarray(columns, dtype=np.float64)
    out = np.full((len(cols), len(scen.outputs)), np.nan)
    for j, o in enumerate(scen.outputs):
        ch = mapping.reference.get(o.key)
        if ch is not None:
            out[:, j] = cols[:, ch] * reference_scale(o, desc.channels[ch].unit or "")
    return out


def errors(scenario: Scenario | str, estimate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """(n, k) kestirim − referans (iç birimde); açısal sütunlar [−π, π) aralığına sarılır, NaN korunur."""
    scen = _scenario(scenario)
    est = np.asarray(estimate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    k = len(scen.outputs)
    if est.ndim != 2 or est.shape[1] != k or ref.shape != est.shape:
        raise FusionError(f"Kestirim ve referans (n, {k}) aynı şekilde olmalı: {est.shape} / {ref.shape}")
    with np.errstate(invalid="ignore"):
        err = est - ref
    for j, o in enumerate(scen.outputs):
        if o.angle:
            err[:, j] = wrap(err[:, j])
    return err


def rmse(err: np.ndarray) -> np.ndarray:
    """Sütun başına karekök ortalama kare (NaN'lar atlanır; hiç sonlu değer yoksa NaN)."""
    e = np.asarray(err, dtype=np.float64)
    if e.ndim == 1:
        e = e[:, None]
    ok = np.isfinite(e)
    count = ok.sum(axis=0)
    total = np.where(ok, e, 0.0) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, np.sqrt(total.sum(axis=0) / np.maximum(count, 1)), np.nan)


def to_display(scenario: Scenario | str, values: np.ndarray) -> np.ndarray:
    """İç birimden gösterim birimine (açısal sütunlar rad → °)."""
    scen = _scenario(scenario)
    scale = np.array([o.display_scale for o in scen.outputs])
    return np.asarray(values, dtype=np.float64) * scale


def describe(scenario: Scenario | str) -> str:
    """Senaryonun kısa metin özeti (girişler, çıktılar, algoritmalar) — günlük / ipucu için."""
    scen = _scenario(scenario)
    ins = ", ".join(f"{i.label}{'' if i.required else ' (isteğe bağlı)'}" for i in scen.inputs)
    outs = ", ".join(f"{o.label} ({o.unit})" for o in scen.outputs)
    algos = ", ".join(cls.name for cls in scen.algorithms.values())
    return f"{scen.name}\nGirişler: {ins}\nÇıktılar: {outs}\nAlgoritmalar: {algos}"

