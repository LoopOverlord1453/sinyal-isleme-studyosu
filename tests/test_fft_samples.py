"""Örnek kayıtlardaki bilinen bileşenlerin FFT ile doğru ölçüldüğünü doğrular (tools/verify_fft_samples.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import verify_fft_samples as verify  # noqa: E402


def test_all_known_components_match_theory():
    failed = [f"{c.sample}/{c.channel}/{c.what}: beklenen {c.expected:+.3f}, ölçülen {c.measured:+.3f} {c.unit}"
              for c in verify.run_checks() if not c.ok]
    assert not failed, "\n".join(failed)


def test_gap_free_window_avoids_amplitude_loss():
    dirty, clean = verify.gap_effect()
    assert dirty < -1.0          # kayıp içeren pencere genliği belirgin düşürür
    assert abs(clean) < 0.05     # kayıpsız pencere doğru okur (0 dB)
