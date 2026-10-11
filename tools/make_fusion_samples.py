"""Genel sensör füzyonu senaryoları için örnek kayıtlar (11–13) üretir — samples_signal_data/ altına.

Kullanım:  .venv\\Scripts\\python tools\\make_fusion_samples.py

Kayıtlar `tools/make_sample_recordings.py` ile aynı yazıcıyı kullanır (periyodik DESCRIPTOR, DATA çerçeveleri,
MARKER olayları, gerçekçi alınma zamanları); veriler `spt.dsp.fusion_generic` senaryolarının analitik
simülatörlerinden gelir, gerçek değerler `ref_*` kanallarındadır. Seyrek sensörler (barometre 25 Hz, GNSS 5 Hz,
ultrasonik 20 Hz …) aynı akışta, ölçüm olmayan örneklerde NaN taşır. Tüm kayıtlar sabit tohumludur.

  11  İrtifa: basinc (Pa, 25 Hz) + ivme_z (m/s², 200 Hz) + ref_irtifa / ref_dikey_hiz
  12  2B konum: gps_x / gps_y (m, 5 Hz, 3 s kesinti) + jiro_z (°/s) + hiz (m/s, 100 Hz) + ref_x / ref_y /
      ref_baslik / ref_hiz
  13  Doğrusal çok sensörlü: mesafe_lidar / mesafe_ultrasonik / mesafe_kizilotesi (m) + enkoder_hiz (m/s) +
      ref_konum / ref_hiz
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import make_sample_recordings as msr  # noqa: E402

from spt.console import utf8_stdio  # noqa: E402
from spt.dsp import fusion_generic as fg  # noqa: E402
from spt.dsp.fusion_scenarios import altitude, linear_multi, position2d  # noqa: E402
from spt.protocol.format import SAMPLE_TYPES_BY_NAME, ChannelInfo, Descriptor  # noqa: E402

R2D = 180.0 / np.pi
MARKER_SCENARIO = 50     # senaryo açıklaması
MARKER_EVENT = 51        # olay (kesinti başı / sonu …)


def sample_altitude(rng: np.random.Generator) -> list[msr.StreamSpec]:
    """İrtifa: 200 Hz akış; barometre basıncı 25 Hz (aradaki örnekler NaN), dikey ivme, gerçek irtifa / hız."""
    fs = 200.0
    sim = altitude.simulate(120.0, fs, int(rng.integers(1, 1_000_000)), baro_rate=25.0, sigma_p=4.0,
                            sigma_acc=0.08, acc_bias=0.12)
    ch = [ChannelInfo("basinc", "Pa"), ChannelInfo("ivme_z", "m/s²"), ChannelInfo("ref_irtifa", "m"),
          ChannelInfo("ref_dikey_hiz", "m/s")]
    values = np.column_stack((sim.inputs["baro_p"], sim.inputs["acc_z"], sim.truth[:, 0], sim.truth[:, 1]))
    text = ("senaryo=altitude (irtifa: barometre + ivmeolcer). basinc 25 Hz (arada NaN), sigma_p=4 Pa, "
            "p0=101325 Pa (ISA); ivme_z dunya yukari ozgul kuvvet (+g dahil), sigma=0.08 m/s2, "
            "sapma ~+0.12 m/s2")
    markers = [(0.0, MARKER_SCENARIO, text), (5.0, MARKER_EVENT, "tirmanma +35 m"),
               (62.0, MARKER_EVENT, "alcalma -40 m")]
    return [msr.StreamSpec(1, "irtifa-baro-ivme", fs, "float32", ch, values, markers=markers)]


def sample_position(rng: np.random.Generator) -> list[msr.StreamSpec]:
    """2B konum: 100 Hz akış; GNSS 5 Hz (+ 3 s kesinti), jiroskop z (°/s), odometri hızı, gerçek durum."""
    fs = 100.0
    outage = (60.0, 3.0)
    sim = position2d.simulate(120.0, fs, int(rng.integers(1, 1_000_000)), gnss_rate=5.0, sigma_gnss=2.5,
                              outage=outage, gyro_bias_dps=0.4, sigma_gyro_dps=0.15, odo_scale=1.03,
                              sigma_speed=0.05)
    ch = [ChannelInfo("gps_x", "m"), ChannelInfo("gps_y", "m"), ChannelInfo("jiro_z", "°/s"),
          ChannelInfo("hiz", "m/s"), ChannelInfo("ref_x", "m"), ChannelInfo("ref_y", "m"),
          ChannelInfo("ref_baslik", "°"), ChannelInfo("ref_hiz", "m/s")]
    values = np.column_stack((sim.inputs["gnss"], sim.inputs["gyro_z"] * R2D, sim.inputs["speed"],
                              sim.truth[:, 0], sim.truth[:, 1], sim.truth[:, 2] * R2D, sim.truth[:, 3]))
    text = ("senaryo=position2d (GNSS + jiroskop + odometri). gps_x/gps_y yerel ENU m, 5 Hz (arada NaN), "
            "sigma=2.5 m; jiro_z sapma +0.4 deg/s, sigma=0.15 deg/s; hiz olcek 1.03, sigma=0.05 m/s; "
            "baslik dogudan saat yonunun tersine")
    markers = [(0.0, MARKER_SCENARIO, text), (outage[0], MARKER_EVENT, "GNSS kesintisi basladi"),
               (outage[0] + outage[1], MARKER_EVENT, "GNSS kesintisi bitti")]
    return [msr.StreamSpec(1, "arac-gnss-odometri", fs, "float32", ch, values, markers=markers)]


def sample_linear(rng: np.random.Generator) -> list[msr.StreamSpec]:
    """Doğrusal çok sensörlü: 3 yedekli mesafe sensörü (farklı gürültü / hız) + enkoder hızı, gerçek konum / hız."""
    fs = 100.0
    sim = linear_multi.simulate(60.0, fs, int(rng.integers(1, 1_000_000)))
    ch = [ChannelInfo("mesafe_lidar", "m"), ChannelInfo("mesafe_ultrasonik", "m"),
          ChannelInfo("mesafe_kizilotesi", "m"), ChannelInfo("enkoder_hiz", "m/s"),
          ChannelInfo("ref_konum", "m"), ChannelInfo("ref_hiz", "m/s")]
    values = np.column_stack([sim.inputs[f"s{i}"] for i in range(1, 5)] + [sim.truth[:, 0], sim.truth[:, 1]])
    text = ("senaryo=linear_multi (dogrusal cok sensorlu Kalman). s1 lidar konum sigma=0.03 m 100 Hz; "
            "s2 ultrasonik konum sigma=0.10 m 20 Hz; s3 kizilotesi konum sigma=0.20 m 50 Hz; "
            "s4 enkoder hiz sigma=0.02 m/s 100 Hz; model sabit hiz, sigma_q=0.5")
    return [msr.StreamSpec(1, "coklu-mesafe-enkoder", fs, "float32", ch, values,
                           markers=[(0.0, MARKER_SCENARIO, text)])]


SAMPLES = [
    ("11_irtifa_baro_25Hz_ivme_200Hz_referansli.sptr", sample_altitude, "altitude"),
    ("12_arac_gnss_5Hz_jiro_odometri_100Hz_referansli.sptr", sample_position, "position2d"),
    ("13_coklu_mesafe_sensoru_enkoder_100Hz_referansli.sptr", sample_linear, "linear_multi"),
]


def _mapping_summary(scenario_key: str, specs: list[msr.StreamSpec]) -> str:
    spec = specs[0]
    desc = Descriptor(spec.stream_id, 1, spec.fs, SAMPLE_TYPES_BY_NAME[spec.type_name], spec.layout, spec.name,
                      tuple(spec.channels))
    m = fg.auto_map(scenario_key, desc)
    if m is None:
        return "auto_map: EŞLENEMEDİ"
    ins = ", ".join(f"{k}←{'/'.join(desc.channels[i].name for i in v)} [{m.units[k]}]" for k, v in m.inputs.items())
    refs = ", ".join(f"{k}←{desc.channels[i].name}" for k, i in m.reference.items())
    return f"auto_map({scenario_key}): {ins}; referans: {refs}"


def main() -> int:
    utf8_stdio()
    msr.OUT_DIR.mkdir(exist_ok=True)
    skipped = []
    for seed, (name, make, scenario_key) in enumerate(SAMPLES, start=11):
        specs = make(np.random.default_rng(seed))
        try:
            msr.write_recording(msr.OUT_DIR / name, specs, seed)
        except OSError as exc:
            skipped.append(name)
            print(f"{name:58s} ATLANDI: {exc.strerror or exc} (arayüzde oynatılıyor olabilir)")
            continue
        print(f"    {_mapping_summary(scenario_key, specs)}")
    if skipped:
        print("\nAtlanan dosyalar için arayüzdeki oynatmayı durdurup komutu tekrar çalıştırın.")
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
