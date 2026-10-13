"""Sinyal Test Uygulaması: kanal / akış / yapılandırma tanımları, sinyal üreteçleri ve hazır senaryolar.

Üreteçler sürekli zamanlıdır: `StreamGen.physical(n0, k)` mutlak örnek indeksinden (n0) itibaren k örnek üretir;
ardışık bloklar tek seferde üretilenle birebir aynıdır (faz örnek indeksinden, gürültü kanal başına sıralı
üreteçten gelir). Bu yüzden frekans tablo uzunluğuna yuvarlanmaz (simülatörün döngü tablosundan farkı) ve akış
saatlerce dikişsiz sürer. Faz, frekans ve örnekleme hızı tamsayıysa tam aritmetikle hesaplanır (bkz. waveforms).

Bu modül Qt kullanmaz (gönderim motoru ayrı süreçte çalışır).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields, replace
from fractions import Fraction
from typing import Any

import numpy as np

from spt.dsp import waveforms as wf
from spt.dsp.siggen import _ecg
from spt.protocol.builder import to_wire
from spt.protocol.format import (
    MAX_CHANNELS,
    MAX_DATAGRAM,
    SAMPLE_TYPES_BY_NAME,
    STANDARD_DATAGRAM,
    UDP_PORT,
    ChannelInfo,
    Descriptor,
    Layout,
    SampleType,
    samples_per_frame,
)

# Sinyal türleri: anahtar → (görünen ad, kullanılan ek parametreler)
SIGNAL_KINDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "sine": ("Sinüs", ()),
    "square": ("Kare", ("duty",)),
    "triangle": ("Üçgen", ()),
    "sawtooth": ("Testere", ()),
    "chirp": ("Chirp (f0 → f1)", ("f1", "sweep_s")),
    "noise": ("Beyaz gürültü", ()),
    "multitone": ("Çok tonlu (harmonikli)", ("harmonics",)),
    "pulse": ("Darbe treni", ("duty",)),
    "dc": ("DC", ()),
    "am": ("AM", ("mod_freq", "depth")),
    "fm": ("FM", ("mod_freq", "deviation")),
    "ecg": ("EKG", ()),
    "step": ("Adım (kare, 0 … genlik)", ("duty",)),
    "ramp": ("Rampa (0 … genlik)", ()),
    "iq": ("I/Q (karmaşık ton)", ()),
}
# Ek parametrelerin görünen adı ve birimi
PARAM_LABELS = {"duty": ("Görev oranı", ""), "f1": ("Bitiş frekansı f1", " Hz"),
                "sweep_s": ("Tarama süresi", " s"), "harmonics": ("Harmonik sayısı", ""),
                "mod_freq": ("Modülasyon frekansı", " Hz"), "depth": ("Modülasyon derinliği", ""),
                "deviation": ("Frekans sapması", " Hz")}
LAYOUTS = {"interleaved": Layout.INTERLEAVED, "planar": Layout.PLANAR}
DATAGRAMS = (STANDARD_DATAGRAM, 8972, MAX_DATAGRAM)
MAX_FS = 200e6            # gönderilebilecek en yüksek örnekleme hızı (Hz)
MAX_STREAMS = 32
# Döngü tablosu: yüksek hızlı (fs · kanal ≥ TABLE_MIN_RATE), gürültüsüz ve periyodik akışlarda ortak periyot bir kez
# üretilip tekrar gönderilir (doğrudan hesap ≈ 10 MS/s ile sınırlı; tablodan gönderim yalnız kopyadır)
TABLE_MIN_RATE = 2e6
TABLE_MAX = 1 << 20
TABLE_MIN = 1 << 16
_PERIODIC = {"sine", "square", "triangle", "sawtooth", "pulse", "step", "ramp", "multitone", "ecg", "iq", "dc", "am",
             "fm", "chirp"}


@dataclass
class ChannelSpec:
    name: str = "ch0"
    unit: str = ""
    kind: str = "sine"
    freq: float = 50.0          # Hz (EKG'de kalp hızı / 60)
    amp: float = 1.0            # tepe genliği (gürültüde ≈ tepe: σ = genlik / 3)
    offset: float = 0.0
    phase_deg: float = 0.0
    noise: float = 0.0          # eklenen beyaz gürültünün σ'sı (fiziksel birimde)
    duty: float = 0.5           # kare / darbe / adım
    f1: float = 0.0             # chirp bitiş frekansı (0: 10 × f0)
    sweep_s: float = 1.0        # chirp tarama süresi
    harmonics: int = 5          # çok tonlu: 1/h genlikli h = 1 … N
    mod_freq: float = 5.0       # AM / FM
    depth: float = 0.5          # AM derinliği (0 … 1)
    deviation: float = 50.0     # FM tepe frekans sapması (Hz)


@dataclass
class StreamSpec:
    stream_id: int = 1
    name: str = "akış"
    fs: float = 10_000.0
    sample_type: str = "int16"
    layout: str = "interleaved"
    datagram: int = STANDARD_DATAGRAM
    crc: bool = False
    marker_every: float = 0.0    # s (0 = kapalı)
    frame_ms: float = 10.0       # en uzun paket aralığı: düşük hızda datagram dolmadan da gönderilir (gecikme)
    enabled: bool = True
    rev: int = 1                 # desc_rev: tanımlayıcıyı değiştiren her düzenlemede artar
    channels: list[ChannelSpec] = field(default_factory=lambda: [ChannelSpec()])

    @property
    def stype(self) -> SampleType:
        return SAMPLE_TYPES_BY_NAME[self.sample_type]

    def bytes_per_second(self) -> float:
        return self.fs * self.stype.size * len(self.channels)


@dataclass
class Impairments:
    """Yapay ağ bozulmaları (test için): yalnız DATA çerçevelerine uygulanır."""
    loss_pct: float = 0.0        # atılan çerçeve oranı
    jitter_ms: float = 0.0       # gönderim zamanına eklenen |N(0, σ)| gecikmenin σ'sı
    reorder_pct: float = 0.0     # bir sonraki çerçeveyle yer değiştiren çerçeve oranı
    duplicate_pct: float = 0.0   # iki kez gönderilen çerçeve oranı

    @property
    def active(self) -> bool:
        return any((self.loss_pct, self.jitter_ms, self.reorder_pct, self.duplicate_pct))


@dataclass
class SenderConfig:
    host: str = "127.0.0.1"
    port: int = UDP_PORT
    bind: str = ""               # yerel IPv4 (boş: işletim sistemi seçer)
    scenario: str = ""
    impairments: Impairments = field(default_factory=Impairments)
    streams: list[StreamSpec] = field(default_factory=list)

    def enabled_streams(self) -> list[StreamSpec]:
        return [s for s in self.streams if s.enabled and s.channels]


# ---------------------------------------------------------------------------------------------------- JSON
def config_to_dict(cfg: SenderConfig) -> dict[str, Any]:
    return asdict(cfg)


def _known(cls, data: Any) -> dict[str, Any]:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in dict(data or {}).items() if k in names}


def _clamp(x: float, lo: float, hi: float) -> float:
    return min(max(float(x), lo), hi)


def channel_from_dict(data: Any) -> ChannelSpec:
    ch = ChannelSpec(**_known(ChannelSpec, data))
    kind = ch.kind if ch.kind in SIGNAL_KINDS else "sine"
    return replace(ch, kind=kind, name=str(ch.name)[:31], unit=str(ch.unit)[:15],
                   freq=_clamp(ch.freq, 0.0, MAX_FS), amp=float(ch.amp), offset=float(ch.offset),
                   phase_deg=float(ch.phase_deg), noise=abs(float(ch.noise)), duty=_clamp(ch.duty, 0.0, 1.0),
                   f1=_clamp(ch.f1, 0.0, MAX_FS), sweep_s=_clamp(ch.sweep_s, 1e-3, 3600.0),
                   harmonics=int(_clamp(ch.harmonics, 1, 64)), mod_freq=_clamp(ch.mod_freq, 0.0, MAX_FS),
                   depth=_clamp(ch.depth, 0.0, 1.0), deviation=abs(float(ch.deviation)))


def stream_from_dict(data: Any) -> StreamSpec:
    raw = _known(StreamSpec, data)
    channels = [channel_from_dict(c) for c in raw.pop("channels", [])][:MAX_CHANNELS]
    st = StreamSpec(**raw, channels=channels or [ChannelSpec()])
    return replace(st, stream_id=int(_clamp(st.stream_id, 0, 0xFFFF)), name=str(st.name)[:31],
                   fs=_clamp(st.fs, 1e-3, MAX_FS),
                   sample_type=st.sample_type if st.sample_type in SAMPLE_TYPES_BY_NAME else "int16",
                   layout=st.layout if st.layout in LAYOUTS else "interleaved",
                   datagram=int(_clamp(st.datagram, 128, MAX_DATAGRAM)), crc=bool(st.crc),
                   marker_every=_clamp(st.marker_every, 0.0, 3600.0), frame_ms=_clamp(st.frame_ms, 0.1, 10_000.0),
                   enabled=bool(st.enabled),
                   rev=int(_clamp(st.rev, 1, 0xFFFF)))


def config_from_dict(data: Any) -> SenderConfig:
    """Kayıttan / dosyadan yapılandırma; bilinmeyen alanlar atılır, değerler geçerli aralığa çekilir."""
    raw = _known(SenderConfig, data)
    imp = Impairments(**_known(Impairments, raw.pop("impairments", {})))
    imp = Impairments(_clamp(imp.loss_pct, 0, 100), _clamp(imp.jitter_ms, 0, 10_000),
                      _clamp(imp.reorder_pct, 0, 100), _clamp(imp.duplicate_pct, 0, 100))
    streams = [stream_from_dict(s) for s in raw.pop("streams", [])][:MAX_STREAMS]
    cfg = SenderConfig(**raw, impairments=imp, streams=streams)
    return replace(cfg, host=str(cfg.host).strip() or "127.0.0.1", port=int(_clamp(cfg.port, 1, 65535)),
                   bind=str(cfg.bind or "").strip(), scenario=str(cfg.scenario or ""))


def descriptor_key(st: StreamSpec) -> tuple:
    """Tanımlayıcıyı (dolayısıyla desc_rev'i) etkileyen alanlar: değişirse rev artırılmalı."""
    return (st.stream_id, st.name, st.fs, st.sample_type, st.layout,
            tuple((c.name, c.unit) + channel_scaling_inputs(c) for c in st.channels))


def channel_scaling_inputs(ch: ChannelSpec) -> tuple:
    return (ch.kind, ch.amp, ch.offset, ch.noise)


def frame_samples(spec: StreamSpec, desc: Descriptor | None = None) -> int:
    """DATA çerçevesi başına örnek: datagrama sığan en çok, ama en çok `frame_ms` kadar veri. Düşük hızlı akışta
    datagram dolana kadar beklemek gecikme yaratır (500 Hz EKG, 2 kanal int16: 1472 B'lık datagram ≈ 0.7 s); gerçek
    cihazlar gibi ≈ 10 ms'de bir paket gönderilir."""
    desc = desc or StreamGen(spec).desc
    fit = samples_per_frame(desc, spec.datagram, spec.crc)
    return max(1, min(fit, round(spec.fs * spec.frame_ms / 1000.0)))


# ---------------------------------------------------------------------------------------------------- üreteç
def channel_scale(stype: SampleType, ch: ChannelSpec) -> tuple[float, float]:
    """Tamsayı tiplerde (scale, offset): fiziksel = ham · scale + offset. Kanalın ofseti etrafında ±1.25 · (genlik +
    4σ gürültü) aralığı tam ölçeğe oturur (kırpma payı %25). Kayan noktalı tiplerde değer doğrudan gönderilir."""
    if stype.is_float:
        return 1.0, 0.0
    peak = abs(ch.amp) + 4.0 * ch.noise if ch.kind != "dc" else 4.0 * ch.noise
    half = 1.25 * peak if peak > 0 else max(1.0, 0.1 * abs(ch.offset))
    full = (1 << (stype.bits - 1)) - 1
    scale = half / full
    if stype.is_complex:               # karmaşık tamsayıda ofset taşınmaz (re / im ayrı ölçeklenmez)
        return scale, 0.0
    return scale, (ch.offset if stype.signed else ch.offset - (full + 1) * scale)


class StreamGen:
    """Bir akışın örnek üreteci: tanımlayıcı + mutlak indeksten kablo tipinde blok."""

    def __init__(self, spec: StreamSpec, seed: int = 1):
        self.spec = spec
        stype = spec.stype
        self.stype = stype
        scales = [channel_scale(stype, ch) for ch in spec.channels]
        infos = tuple(ChannelInfo(ch.name or f"ch{i}", ch.unit, sc, off)
                      for i, (ch, (sc, off)) in enumerate(zip(spec.channels, scales)))
        self.desc = Descriptor(spec.stream_id, spec.rev, float(spec.fs), stype, LAYOUTS[spec.layout],
                               spec.name, infos)
        self._scales = scales
        self._rngs = [np.random.default_rng((seed, spec.stream_id, i)) for i in range(len(spec.channels))]
        self.period = None                # döngü tablosu kullanılıyorsa ortak periyot (örnek)
        self._table = None
        if spec.fs * len(spec.channels) >= TABLE_MIN_RATE:
            self.period = common_period(spec)
            if self.period is not None:
                reps = max(1, -(-TABLE_MIN // self.period))
                self._table = self._compute(0, self.period * reps)

    def physical(self, n0: int, k: int) -> np.ndarray:
        """(k, kanal) fiziksel değerler (karmaşık tipte karmaşık)."""
        n = np.arange(n0, n0 + k, dtype=np.int64)
        cols = [channel_values(ch, n, self.spec.fs, rng, self.stype.is_complex)
                for ch, rng in zip(self.spec.channels, self._rngs)]
        return np.stack(cols, axis=1)

    def wire(self, n0: int, k: int) -> np.ndarray:
        """(k, kanal) kablo tipinde, bitişik. Döngü tablosu varsa tablodan (dilim ya da sarmalı indeksleme)."""
        table = self._table
        if table is None:
            return self._compute(n0, k)
        length = len(table)
        i = n0 % length
        if i + k <= length:
            return table[i:i + k]
        return np.ascontiguousarray(table[(np.arange(n0, n0 + k) % length)])

    def _compute(self, n0: int, k: int) -> np.ndarray:
        phys = self.physical(n0, k)
        st = self.stype
        if st.is_float:
            raw = phys
        else:
            scale = np.array([s for s, _ in self._scales])
            offset = np.array([o for _, o in self._scales])
            raw = phys / scale if st.is_complex else (phys - offset) / scale
            raw = np.rint(raw.real) + 1j * np.rint(raw.imag) if st.is_complex else np.rint(raw)
            # to_wire int24'te yuvarlamaz, keser (1 LSB'ye varan hata): ham değer burada yuvarlanır
        return np.ascontiguousarray(to_wire(raw, st))


def _cycles(freq: float, fs: float) -> int | None:
    """freq / fs = p / q (en sade) ise periyot q örnek; değerler kullanıcının yazdığı ondalık sayılar olduğundan
    `Fraction(repr(x))` tam karşılığı verir (ör. 123.4 Hz, 10 kHz → 50000 örnek)."""
    if freq == 0:
        return 1
    q = (Fraction(repr(float(freq))) / Fraction(repr(float(fs)))).denominator
    return q if q <= TABLE_MAX else None


def common_period(spec: StreamSpec) -> int | None:
    """Akışın tüm kanallarının ortak periyodu (örnek); gürültü ya da periyodik olmayan kanal varsa / çok uzunsa
    None (doğrudan hesap)."""
    period = 1
    for ch in spec.channels:
        if ch.kind not in _PERIODIC or ch.noise > 0:
            return None
        parts = [_cycles(ch.freq, spec.fs)]
        if ch.kind in ("am", "fm"):
            parts.append(_cycles(ch.mod_freq, spec.fs))
        elif ch.kind == "chirp":
            sweep = Fraction(repr(float(ch.sweep_s))) * Fraction(repr(float(spec.fs)))
            parts = [sweep.numerator if sweep.denominator == 1 and sweep.numerator <= TABLE_MAX else None]
        for q in parts:
            if q is None:
                return None
            period = math.lcm(period, q)
            if period > TABLE_MAX:
                return None
    return period


def channel_values(ch: ChannelSpec, n: np.ndarray, fs: float, rng: np.random.Generator,
                   complex_out: bool = False) -> np.ndarray:
    """n (mutlak örnek indeksleri) için kanalın fiziksel değeri: ofset + sinyal + gürültü."""
    k = len(n)
    ph = np.mod(wf.phase(n, ch.freq, fs) + ch.phase_deg / 360.0, 1.0)
    t = n / fs
    kind, a = ch.kind, ch.amp
    if kind == "sine":
        x = a * np.sin(2 * np.pi * ph)
    elif kind == "square":
        x = a * wf.square(ph, ch.duty)
    elif kind == "triangle":
        x = a * wf.triangle(ph)
    elif kind == "sawtooth":
        x = a * wf.sawtooth(ph)
    elif kind == "chirp":
        f0 = ch.freq
        f1 = ch.f1 or 10.0 * f0
        tl = np.mod(t, ch.sweep_s)
        x = a * np.sin(2 * np.pi * (f0 * tl + (f1 - f0) * tl * tl / (2 * ch.sweep_s)) + math.radians(ch.phase_deg))
    elif kind == "noise":
        x = a / 3.0 * (rng.standard_normal(k) if not complex_out
                       else (rng.standard_normal(k) + 1j * rng.standard_normal(k)) / math.sqrt(2))
    elif kind == "multitone":
        hs = np.arange(1, ch.harmonics + 1)
        weights = 1.0 / hs
        x = a * (np.sin(2 * np.pi * np.mod(np.outer(ph, hs), 1.0)) @ weights) / weights.sum()
    elif kind in ("pulse", "step"):
        x = a * (ph < ch.duty)
    elif kind == "dc":
        x = np.zeros(k)
    elif kind == "am":
        x = a * (1 + ch.depth * np.sin(2 * np.pi * ch.mod_freq * t)) * np.sin(2 * np.pi * ph) / (1 + ch.depth)
    elif kind == "fm":
        beta = ch.deviation / ch.mod_freq if ch.mod_freq > 0 else 0.0
        x = a * np.sin(2 * np.pi * ph + beta * np.sin(2 * np.pi * ch.mod_freq * t))
    elif kind == "ecg":
        x = a * _ecg(ph, 1.0 / ch.freq if ch.freq > 0 else 1.0)
    elif kind == "ramp":
        x = a * ph
    elif kind == "iq":
        x = a * np.exp(2j * np.pi * ph) if complex_out else a * np.cos(2 * np.pi * ph)
    else:
        raise ValueError(f"bilinmeyen sinyal türü: {kind}")
    x = x + ch.offset
    if ch.noise > 0 and kind != "noise":
        if complex_out:
            x = x + ch.noise * (rng.standard_normal(k) + 1j * rng.standard_normal(k)) / math.sqrt(2)
        else:
            x = x + ch.noise * rng.standard_normal(k)
    if complex_out and not np.iscomplexobj(x):
        x = x.astype(np.complex128)
    return x


# ---------------------------------------------------------------------------------------------------- senaryolar
def _ch(name: str, unit: str, kind: str, freq: float, amp: float, **kw) -> ChannelSpec:
    return ChannelSpec(name=name, unit=unit, kind=kind, freq=freq, amp=amp, **kw)


def _grid() -> list[StreamSpec]:
    v = 230 * math.sqrt(2)
    return [StreamSpec(1, "sebeke-3faz", 10_000.0, "int16", channels=[
        _ch("V_L1", "V", "sine", 50.0, v, noise=1.5),
        _ch("V_L2", "V", "sine", 50.0, v, phase_deg=-120.0, noise=1.5),
        _ch("V_L3", "V", "sine", 50.0, v, phase_deg=120.0, noise=1.5),
        _ch("I_L1", "A", "multitone", 50.0, 12.0, harmonics=7, noise=0.05),
    ])]


def _imu() -> list[StreamSpec]:
    return [StreamSpec(2, "imu-6eksen", 1000.0, "int16", channels=[
        _ch("acc_x", "m/s²", "sine", 2.0, 0.6, noise=0.04),
        _ch("acc_y", "m/s²", "sine", 1.3, 0.4, phase_deg=90.0, noise=0.04),
        _ch("acc_z", "m/s²", "sine", 0.7, 0.2, offset=9.81, noise=0.04),
        _ch("gyr_x", "°/s", "sine", 1.1, 25.0, noise=0.4),
        _ch("gyr_y", "°/s", "triangle", 0.5, 15.0, noise=0.4),
        _ch("gyr_z", "°/s", "sine", 0.25, 40.0, noise=0.4),
    ])]


def _audio() -> list[StreamSpec]:
    return [StreamSpec(3, "ses-stereo", 48_000.0, "int16", channels=[
        _ch("sol", "FS", "sine", 440.0, 0.5, noise=0.002),
        _ch("sag", "FS", "chirp", 100.0, 0.5, f1=8000.0, sweep_s=2.0),
    ])]


def _ecg_stream() -> list[StreamSpec]:
    return [StreamSpec(4, "ekg", 500.0, "int16", channels=[
        _ch("DII", "mV", "ecg", 1.2, 1.2, noise=0.02),
        _ch("solunum", "", "sine", 0.25, 0.5, noise=0.01),
    ])]


def _vibration() -> list[StreamSpec]:
    return [StreamSpec(5, "titresim", 25_600.0, "float32", channels=[
        _ch("x", "g", "multitone", 120.0, 2.0, harmonics=6, noise=0.08),
        _ch("y", "g", "am", 3000.0, 1.0, mod_freq=87.0, depth=0.8, noise=0.05),
        _ch("z", "g", "noise", 0.0, 0.6),
    ])]


def _tones() -> list[StreamSpec]:
    return [StreamSpec(6, "test-tonlari", 48_000.0, "float32", channels=[
        _ch("sinus_1k", "V", "sine", 1000.0, 1.0),
        _ch("kare_100", "V", "square", 100.0, 1.0, duty=0.3),
        _ch("ucgen_250", "V", "triangle", 250.0, 1.0),
        _ch("testere_50", "V", "sawtooth", 50.0, 1.0),
        _ch("darbe_10", "V", "pulse", 10.0, 1.0, duty=0.05),
        _ch("fm_2k", "V", "fm", 2000.0, 1.0, mod_freq=5.0, deviation=500.0),
        _ch("rampa_2", "V", "ramp", 2.0, 1.0),
        _ch("dc", "V", "dc", 0.0, 0.0, offset=1.65, noise=0.002),
    ])]


def _iq() -> list[StreamSpec]:
    return [StreamSpec(7, "iq-250k", 250_000.0, "cint16", channels=[
        _ch("rx0", "", "iq", 12_500.0, 0.8, noise=0.01),
    ])]


def _high_rate() -> list[StreamSpec]:
    return [StreamSpec(8, "hizli-1MS", 1_000_000.0, "int16", channels=[
        _ch("ch0", "V", "sine", 10_000.0, 1.0, noise=0.01),
        _ch("ch1", "V", "square", 1000.0, 0.5),
    ])]


# Seçilebilir senaryolar (görünen ad → akış listesi üreten fonksiyon); "Hepsi" ilk altısını birlikte gönderir
SCENARIOS = {
    "Şebeke 3 faz 50 Hz + harmonik": _grid,
    "IMU 6 eksen 1 kHz": _imu,
    "Ses stereo 48 kHz": _audio,
    "EKG 500 Hz": _ecg_stream,
    "Titreşim 25.6 kHz": _vibration,
    "Test tonları (çok tonlu)": _tones,
    "I/Q 250 kS/s (karmaşık)": _iq,
    "Yüksek hız 1 MS/s": _high_rate,
}
ALL = "Hepsi"
_ALL_PARTS = ("Şebeke 3 faz 50 Hz + harmonik", "IMU 6 eksen 1 kHz", "Ses stereo 48 kHz", "EKG 500 Hz",
              "Titreşim 25.6 kHz", "Test tonları (çok tonlu)", "I/Q 250 kS/s (karmaşık)")


def scenario_names() -> list[str]:
    return [*SCENARIOS, ALL]


def scenario(name: str) -> list[StreamSpec]:
    """Senaryonun akışları (her çağrıda yeni nesneler). "Hepsi": yüksek hız dışındaki tüm senaryolar."""
    if name == ALL:
        return [s for part in _ALL_PARTS for s in SCENARIOS[part]()]
    try:
        return SCENARIOS[name]()
    except KeyError:
        raise ValueError(f"bilinmeyen senaryo: {name}") from None


def default_config() -> SenderConfig:
    return SenderConfig(scenario=ALL, streams=scenario(ALL))
