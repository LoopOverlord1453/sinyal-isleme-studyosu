"""samples_signal_data/ klasöründeki örnek .sptr kayıtlarını üretir (tekrarlanabilir, sabit tohumlu).

Kullanım:  .venv\\Scripts\\python tools\\make_sample_recordings.py

Her kayıt, gerçek bir cihazın UDP ile göndereceği çerçevelerin aynısını içerir: periyodik DESCRIPTOR,
1472 baytlık DATA çerçeveleri, MARKER olayları ve gerçekçi alınma zamanları.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import signal as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spt.console import utf8_stdio  # noqa: E402
from spt.dsp import waveforms as wf  # noqa: E402
from spt.protocol.builder import build_data, build_descriptor, build_marker, to_wire  # noqa: E402
from spt.protocol.format import (  # noqa: E402
    SAMPLE_TYPES_BY_NAME,
    STANDARD_DATAGRAM,
    ChannelInfo,
    Descriptor,
    Layout,
    samples_per_frame,
)
from spt.protocol.recording import SptrWriter  # noqa: E402

OUT_DIR = ROOT / "samples_signal_data"
START_NS = 1_760_000_000_000_000_000  # sabit başlangıç zamanı (tekrarlanabilirlik için)
MAX_FRAME_SECONDS = 0.05


@dataclass
class StreamSpec:
    stream_id: int
    name: str
    fs: float
    type_name: str
    channels: list[ChannelInfo]
    values: np.ndarray                       # (N, kanal) fiziksel değerler (karmaşık olabilir)
    layout: Layout = Layout.INTERLEAVED
    markers: list[tuple[float, int, str]] = field(default_factory=list)  # (zaman s, kod, metin)
    drop: float = 0.0                         # rastgele çerçeve kaybı oranı
    burst_gap: tuple[float, float] | None = None  # (başlangıç s, süre s) toplu kayıp
    reset_at: float | None = None             # cihaz reset anı (first_sample 0'a döner)


def int_channel(name: str, unit: str, type_name: str, full_scale: float, center: float = 0.0) -> ChannelInfo:
    """Tamsayı tipte ±full_scale aralığını kapsayan ölçekli kanal (işaretsizde orta nokta = center)."""
    st = SAMPLE_TYPES_BY_NAME[type_name]
    half = 1 << (st.bits - 1)
    scale = full_scale / (half - 1)
    offset = center - half * scale if not st.signed else center
    return ChannelInfo(name, unit, scale, offset)


def to_raw(values: np.ndarray, spec: StreamSpec) -> np.ndarray:
    st = SAMPLE_TYPES_BY_NAME[spec.type_name]
    if st.is_float:
        return to_wire(values, st)
    scales = np.array([c.scale for c in spec.channels])
    offsets = np.array([c.offset for c in spec.channels])
    raw = values / scales if st.is_complex else (values - offsets) / scales
    return to_wire(raw, st)


def stream_frames(spec: StreamSpec, rng: np.random.Generator) -> list[tuple[int, bytes]]:
    st = SAMPLE_TYPES_BY_NAME[spec.type_name]
    desc = Descriptor(spec.stream_id, 1, spec.fs, st, spec.layout, spec.name, tuple(spec.channels))
    # Datagrama sığan kadar, ama gecikme için çerçeve başına en fazla 50 ms veri (yavaş akışlarda küçük çerçeve)
    spf = min(samples_per_frame(desc, STANDARD_DATAGRAM), max(1, int(spec.fs * MAX_FRAME_SECONDS)))
    wire = to_raw(spec.values, spec)
    n = wire.shape[0]
    frames: list[tuple[int, bytes]] = []
    seq = 0
    device_offset = 0      # reset sonrası first_sample'ın sıfırdan başlaması için
    next_desc = 0.0
    reset_done = spec.reset_at is None
    pending_markers = sorted(spec.markers)

    def rx_time(t: float) -> int:
        return START_NS + int(t * 1e9) + int(rng.uniform(80e3, 120e3))  # ağ + yığın gecikmesi

    for start in range(0, n, spf):
        count = min(spf, n - start)
        t_end = (start + count) / spec.fs
        t_start = start / spec.fs
        if not reset_done and t_start >= spec.reset_at:
            reset_done = True
            device_offset = start
            seq = 0
            next_desc = t_start
        if t_start >= next_desc:
            frames.append((rx_time(t_start), build_descriptor(desc, seq, int((t_start) * 1e9))))
            seq += 1
            next_desc = t_start + 1.0
        while pending_markers and pending_markers[0][0] <= t_end:
            t_m, code, text = pending_markers.pop(0)
            index = int(t_m * spec.fs) - device_offset
            frames.append((rx_time(t_m), build_marker(spec.stream_id, 1, seq, int(t_m * 1e9), index, code, text)))
            seq += 1
        lost = spec.drop and rng.random() < spec.drop
        if spec.burst_gap and spec.burst_gap[0] <= t_start < spec.burst_gap[0] + spec.burst_gap[1]:
            lost = True
        first = start - device_offset
        frame = build_data(desc, seq, int(t_start * 1e9), first, wire[start:start + count])
        seq += 1
        if not lost:
            frames.append((rx_time(t_end), frame))
    return frames


def write_recording(path: Path, specs: list[StreamSpec], seed: int) -> None:
    rng = np.random.default_rng(seed)
    frames = [f for spec in specs for f in stream_frames(spec, rng)]
    frames.sort(key=lambda f: f[0])
    with SptrWriter(path, created_unix_ns=START_NS) as w:
        for rx, frame in frames:
            w.write(frame, rx)
    print(f"{path.name:58s} {path.stat().st_size / 1e6:6.2f} MB  {len(frames):6d} çerçeve")


# ---------------------------------------------------------------- örnek sinyaller
def t_axis(fs: float, seconds: float) -> np.ndarray:
    return np.arange(int(fs * seconds)) / fs


def sample_power_line(rng) -> list[StreamSpec]:
    fs, t = 10_000.0, t_axis(10_000.0, 10.0)
    a = (1.2 * np.sin(2 * np.pi * 50 * t) + 0.18 * np.sin(2 * np.pi * 150 * t)
         + 0.35 * np.sin(2 * np.pi * 1200 * t) + 0.01 * rng.standard_normal(t.size))
    b = (0.8 * np.sin(2 * np.pi * 50 * t - 2 * np.pi / 3) + 0.2 * np.sin(2 * np.pi * 3100 * t)
         + 0.008 * rng.standard_normal(t.size))
    ch = [int_channel("faz_A", "V", "int16", 2.5), int_channel("faz_B", "V", "int16", 2.5)]
    return [StreamSpec(1, "sebeke-olcum", fs, "int16", ch, np.column_stack((a, b)))]


def sample_chirp(rng) -> list[StreamSpec]:
    fs, t = 20_000.0, t_axis(20_000.0, 8.0)
    x = 0.9 * sps.chirp(t % 4.0, f0=20, t1=4.0, f1=8000, method="logarithmic") + 0.003 * rng.standard_normal(t.size)
    markers = [(0.0, 1, "tarama başladı"), (4.0, 1, "tarama başladı")]
    return [StreamSpec(1, "frekans-tarama", fs, "float32", [ChannelInfo("chirp", "V")], x[:, None], markers=markers)]


def sample_waveforms(rng) -> list[StreamSpec]:
    fs = 50_000.0
    n = np.arange(int(fs * 4.0))
    sq = wf.square(wf.phase(n, 1000, fs)) * 3.0
    tri = wf.triangle(wf.phase(n, 500, fs)) * 2.0
    saw = wf.sawtooth(wf.phase(n, 250, fs)) * 1.0
    ch = [int_channel(n, "V", "int16", 4.0) for n in ("kare_1kHz", "ucgen_500Hz", "testere_250Hz")]
    return [StreamSpec(1, "dalga-sekilleri", fs, "int16", ch, np.column_stack((sq, tri, saw)), layout=Layout.PLANAR)]


def sample_imu(rng) -> list[StreamSpec]:
    fs, t = 1000.0, t_axis(1000.0, 30.0)
    motor = np.clip((t - 8) / 3, 0, 1) * np.clip((24 - t) / 2, 0, 1)   # 8 s'de kalkış, 22–24 s'de duruş
    vib = motor * (0.35 * np.sin(2 * np.pi * 120 * t) + 0.12 * np.sin(2 * np.pi * 240 * t + 0.4))
    noise = lambda s: s * rng.standard_normal(t.size)  # noqa: E731
    acc = [vib + noise(0.01), 0.6 * vib + noise(0.01), 1.0 + 0.4 * vib + noise(0.01)]
    gyro = [motor * 2.0 * np.sin(2 * np.pi * 0.3 * t) + noise(0.05), noise(0.05), motor * 180.0 + noise(0.5)]
    ch = [ChannelInfo(n, "g") for n in ("ivme_x", "ivme_y", "ivme_z")] + \
         [ChannelInfo(n, "°/s") for n in ("jiro_x", "jiro_y", "jiro_z")]
    markers = [(8.0, 10, "motor_start"), (22.0, 11, "motor_stop")]
    return [StreamSpec(1, "imu", fs, "float32", ch, np.column_stack(acc + gyro), markers=markers)]


def sample_audio(rng) -> list[StreamSpec]:
    fs = 48_000.0
    t = t_axis(fs, 6.0)
    notes = [(0.0, 440.0), (1.5, 554.37), (3.0, 659.25), (4.5, 880.0)]
    left = np.zeros(t.size)
    right = np.zeros(t.size)
    markers = []
    for i, (start, f) in enumerate(notes):
        env = np.where(t >= start, np.exp(-(t - start) * 1.2), 0.0)
        tone = env * (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t))
        left += 0.35 * tone
        right += 0.25 * np.roll(tone, int(0.0004 * fs))
        markers.append((start, 20 + i, f"nota {f:.0f} Hz"))
    left += 1e-4 * rng.standard_normal(t.size)
    right += 1e-4 * rng.standard_normal(t.size)
    ch = [ChannelInfo("sol", "FS", 1 / (1 << 23)), ChannelInfo("sag", "FS", 1 / (1 << 23))]
    return [StreamSpec(1, "ses-24bit", fs, "int24", ch, np.column_stack((left, right)), markers=markers)]


def sample_iq(rng) -> list[StreamSpec]:
    fs, t = 1_000_000.0, t_axis(1_000_000.0, 1.0)
    x = (0.5 * np.exp(2j * np.pi * 125_000 * t) + 0.05 * np.exp(-2j * np.pi * 300_000 * t)
         + 0.002 * (rng.standard_normal(t.size) + 1j * rng.standard_normal(t.size)))
    ch = [ChannelInfo("iq", "FS", 1 / 32767)]
    return [StreamSpec(1, "iq-alici", fs, "cint16", ch, x[:, None])]


def sample_ecg(rng) -> list[StreamSpec]:
    fs, t = 500.0, t_axis(500.0, 20.0)
    hr = 72 / 60
    phase = (t * hr) % 1.0
    wave = lambda c, w, a: a * np.exp(-((phase - c) ** 2) / (2 * w ** 2))  # noqa: E731
    ecg = wave(0.18, 0.025, 0.12) - wave(0.33, 0.009, 0.15) + wave(0.36, 0.011, 1.1) \
        - wave(0.39, 0.010, 0.25) + wave(0.62, 0.045, 0.3)
    x = ecg + 0.25 * np.sin(2 * np.pi * 50 * t) + 0.3 * np.sin(2 * np.pi * 0.25 * t) \
        + 0.02 * rng.standard_normal(t.size)
    # 12 bit ADC, uint16 içinde: ham 0..4095 ↔ −2..+2 mV (işaretsiz tip + ofset örneği)
    ch = [ChannelInfo("ekg", "mV", 4.0 / 4096, -2.0)]
    return [StreamSpec(1, "ekg-12bit", fs, "uint16", ch, x[:, None],
                       markers=[(5.0, 30, "hasta hareket etti"), (12.0, 31, "elektrot kontrol")])]


def sample_lossy(rng) -> list[StreamSpec]:
    fs, t = 10_000.0, t_axis(10_000.0, 10.0)
    x = np.sin(2 * np.pi * 100 * t) + 0.02 * rng.standard_normal(t.size)
    ch = [int_channel("sinyal", "V", "int16", 1.5)]
    return [StreamSpec(1, "kayipli-akis", fs, "int16", ch, x[:, None], drop=0.02, burst_gap=(3.0, 0.5),
                       reset_at=6.0, markers=[(6.0, 99, "cihaz reset")])]


def sample_multi_stream(rng) -> list[StreamSpec]:
    fs1, t1 = 25_600.0, t_axis(25_600.0, 10.0)
    bearing = (0.2 * np.sin(2 * np.pi * 29.5 * t1) + 0.05 * np.sin(2 * np.pi * 3200 * t1)
               * (1 + np.sign(np.sin(2 * np.pi * 7.3 * t1))) + 0.01 * rng.standard_normal(t1.size))
    vib = np.column_stack((bearing, 0.7 * bearing + 0.01 * rng.standard_normal(t1.size)))
    ch1 = [int_channel("rulman_x", "g", "int16", 0.5), int_channel("rulman_y", "g", "int16", 0.5)]
    fs2, t2 = 10.0, t_axis(10.0, 10.0)
    temp = 24.0 + 0.6 * t2 + 0.05 * rng.standard_normal(t2.size)
    return [StreamSpec(1, "titresim", fs1, "int16", ch1, vib),
            StreamSpec(2, "sicaklik", fs2, "float32", [ChannelInfo("sicaklik", "°C")], temp[:, None])]


def sample_imu_reference(rng) -> list[StreamSpec]:
    """Sensör füzyonu için 9 eksenli IMU + gerçek yönelim (referans) kanalları.

    Yönelim analitik bir yörüngeden gelir; jiroskopta sabit sapma ve gürültü, ivmeölçerde doğrusal ivme ve gürültü,
    manyetometrede gürültü vardır. ref_roll / ref_pitch / ref_yaw gerçek açılardır (kestirim hatası ölçmek için).
    """
    from spt.dsp.fusion import simulate_imu

    sim = simulate_imu(duration=60.0, fs=200.0, seed=int(rng.integers(1, 1_000_000)))
    deg = 180.0 / np.pi
    ch = ([ChannelInfo(f"jiro_{a}", "°/s") for a in "xyz"] + [ChannelInfo(f"ivme_{a}", "m/s²") for a in "xyz"]
          + [ChannelInfo(f"mag_{a}", "µT") for a in "xyz"]
          + [ChannelInfo(f"ref_{n}", "°") for n in ("roll", "pitch", "yaw")])
    values = np.column_stack((sim.gyro * deg, sim.acc, sim.mag, sim.truth * deg))
    bias = ", ".join(f"{v:+.3f}" for v in sim.gyro_bias * deg)
    markers = [(0.0, 40, f"jiroskop sapmasi [{bias}] deg/s")]
    return [StreamSpec(1, "imu-9eksen-referansli", 200.0, "float32", ch, values, markers=markers)]


SAMPLES = [
    ("01_sinus_50Hz_harmonik_2kanal_int16_10kHz.sptr", sample_power_line),
    ("02_chirp_20Hz_8kHz_float32_20kHz.sptr", sample_chirp),
    ("03_kare_ucgen_testere_3kanal_planar_int16_50kHz.sptr", sample_waveforms),
    ("04_imu_6kanal_float32_1kHz.sptr", sample_imu),
    ("05_ses_24bit_stereo_48kHz.sptr", sample_audio),
    ("06_iq_cint16_1MSps.sptr", sample_iq),
    ("07_ekg_50Hz_parazitli_uint16_500Hz.sptr", sample_ecg),
    ("08_kayipli_akis_ve_reset_int16_10kHz.sptr", sample_lossy),
    ("09_coklu_akis_titresim_ve_sicaklik.sptr", sample_multi_stream),
    ("10_imu_9eksen_yonelim_referansli_200Hz.sptr", sample_imu_reference),
]


def main() -> int:
    utf8_stdio()
    OUT_DIR.mkdir(exist_ok=True)
    skipped = []
    for seed, (name, make) in enumerate(SAMPLES, start=1):
        try:
            write_recording(OUT_DIR / name, make(np.random.default_rng(seed)), seed)
        except OSError as exc:
            skipped.append(name)
            print(f"{name:58s} ATLANDI: {exc.strerror or exc} (arayüzde oynatılıyor olabilir)")
    if skipped:
        print("\nAtlanan dosyalar için arayüzdeki oynatmayı durdurup komutu tekrar çalıştırın.")
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
