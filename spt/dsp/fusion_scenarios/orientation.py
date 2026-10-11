"""Senaryo 1 — Yönelim (IMU / MARG): mevcut `spt.dsp.fusion` algoritmalarının genel API'ye uyarlayıcısı.

Algoritmalar yeniden yazılmaz: her biri `spt.dsp.fusion.ALGORITHMS` içindeki sınıfın kendisini çalıştırır, parametre
listesi de aynıdır. Uyarlayıcının eklediği tek şey çok hızlı / eksik veri davranışıdır:

  - Jiroskop veya ivmeölçer satırı sonlu değilse (NaN / inf) çıktı NaN olur ve algoritma bir sonraki geçerli
    satırda ivmeölçer (+ manyetometre) ile temiz biçimde yeniden başlatılır (füzyon motorunun akış kaybındaki
    davranışının aynısı).
  - Manyetometre daha seyrek olabilir (NaN = o örnekte ölçüm yok): son geçerli örnek tutulur (sıfır-dereceden
    tutma). Manyetometre eşlenmişse başlangıçta (ve her yeniden başlatmada) ilk manyetometre örneği en fazla
    0.5 s beklenir ki başlangıç başlığı (yaw) doğru olsun; gelmezse manyetometresiz başlanır.
  - Manyetometrenin birimi algoritmalar için önemsizdir (yalnızca yönü kullanılır); SI'da (T) verilir.
Çıktılar: roll, pitch, yaw (ZYX, rad). Kurallar için `spt.dsp.fusion` modül belgesine bakın.
"""
from __future__ import annotations

import numpy as np

from spt.dsp import fusion as fu
from spt.dsp.fusion_scenarios.base import (
    R2D,
    UNITS_ACC,
    UNITS_MAG,
    UNITS_RATE,
    Estimator,
    Input,
    Output,
    Scenario,
    Simulation,
    fmt_vec,
)

MAG_WAIT_S = 0.5     # ilk manyetometre örneği için en fazla bekleme

INPUTS = (
    Input("gyro", "Jiroskop", ("x", "y", "z"), UNITS_RATE, True,
          ("gyro", "gyr", "jiro", "jiroskop", "omega", "rate", "angular", "g"),
          "Gövde açısal hızı (sağ el, z yukarı)"),
    Input("acc", "İvmeölçer", ("x", "y", "z"), UNITS_ACC, True,
          ("acc", "accel", "ivme", "ivmeolcer", "a"),
          "Özgül kuvvet; durağanken +g·ẑ_gövde ölçer"),
    Input("mag", "Manyetometre", ("x", "y", "z"), UNITS_MAG, False,
          ("mag", "manyet", "manyetometre", "compass", "pusula", "m"),
          "İsteğe bağlı; yalnızca yönü kullanılır. Seyrek olabilir (son örnek tutulur)"),
)
OUTPUTS = (
    Output("roll", "Roll", "°", True, ("roll", "yuvarlanma", "phi")),
    Output("pitch", "Pitch", "°", True, ("pitch", "yunuslama", "theta")),
    Output("yaw", "Yaw", "°", True, ("yaw", "sapma", "heading", "baslik", "psi")),
)


class _OrientationBase(Estimator):
    """`impl` (spt.dsp.fusion.Estimator alt sınıfı) üzerine ince uyarlayıcı."""
    impl: type[fu.Estimator] = fu.Mahony
    inputs = INPUTS
    outputs = OUTPUTS

    def _setup(self) -> None:
        self._est: fu.Estimator | None = None    # geçerli satır gelince kurulur
        self._mag: np.ndarray | None = None      # tutulan son manyetometre örneği
        self._wait = 0                           # ilk manyetometre için beklenen örnek sayısı
        self.restarts = 0
        self._bias = np.zeros(3)

    def _restart(self) -> None:
        if self._est is not None:
            self._bias = np.array(self._est.bias, dtype=np.float64)
            self.restarts += 1
        self._est = None
        self._mag = None
        self._wait = 0

    def _run(self, x, n, dt):
        gyro, acc = x["gyro"], x["acc"]
        mag = x.get("mag")
        out = np.full((n, 3), np.nan)
        valid = np.isfinite(gyro).all(axis=1) & np.isfinite(acc).all(axis=1)
        bad = np.flatnonzero(~valid)
        if mag is not None:
            mag_ok = np.isfinite(mag).all(axis=1) & (np.abs(mag).sum(axis=1) > 0)
            fresh = np.flatnonzero(mag_ok)
        wait_max = max(1, int(round(MAG_WAIT_S / dt)))
        k = 0
        while k < n:
            if not valid[k]:
                self._restart()
                k += 1
                continue
            with_mag = mag is not None and (self._mag is not None or mag_ok[k])
            if self._est is None and mag is not None and not with_mag and self._wait < wait_max:
                self._wait += 1          # manyetometre bekleniyor: çıktı NaN
                k += 1
                continue
            end = int(bad[np.searchsorted(bad, k)]) if len(bad) and bad[-1] >= k else n
            if mag is not None and not with_mag:  # manyetometre gelene kadar manyetometresiz parça
                j = np.searchsorted(fresh, k)
                if j < len(fresh):
                    end = min(end, int(fresh[j]))
            if self._est is None:
                self._est = self.impl(**self.values)
            m = None
            if with_mag:
                seg_ok = mag_ok[k:end]
                idx = np.where(seg_ok, np.arange(end - k), -1)
                np.maximum.accumulate(idx, out=idx)
                m = np.empty((end - k, 3))
                held = idx < 0
                m[~held] = mag[k:end][idx[~held]]
                if held.any():
                    m[held] = self._mag
                self._mag = m[-1].copy()
            q = self._est.run(gyro[k:end], acc[k:end], m, dt)
            out[k:end] = np.column_stack(fu.euler_from_quat(q))
            k = end
        return out

    def diagnostics(self) -> dict[str, str]:
        est = self._est
        bias = np.array(est.bias, dtype=np.float64) if est is not None else self._bias
        text = fmt_vec(bias * R2D) if getattr(self.impl, "estimates_bias", True) else "— (kestirmez)"
        if getattr(self.impl, "bias_is_euler", False):  # eksen başına Kalman: Euler hızı sapması (gövde değil)
            text += " (Euler hızı)"
        d = {"Jiroskop sapması (°/s)": text}
        if est is not None and hasattr(est, "ess"):
            d["ESS"] = f"{float(est.ess):.0f}"
        if self.restarts:
            d["Yeniden başlatma"] = str(self.restarts)
        return d


def _adapter(cls: type[fu.Estimator]) -> type[_OrientationBase]:
    name = f"Orientation{cls.__name__}"
    adapter = type(name, (_OrientationBase,), {"key": cls.key, "name": cls.name, "params": tuple(cls.params),
                                                "impl": cls, "__module__": __name__, "__qualname__": name})
    globals()[name] = adapter    # süreçler arası pickle için modül düzeyinde ad
    return adapter


ALGORITHMS: dict[str, type[Estimator]] = {key: _adapter(cls) for key, cls in fu.ALGORITHMS.items()}


def raw(x: dict, n: int, sv: dict, state: dict) -> np.ndarray:
    """Füzyonsuz: roll / pitch ivmeölçerden (durağan varsayım), yaw eğim düzeltmeli manyetik başlık
    (manyetometre yoksa ya da o örnekte NaN ise NaN). Jiroskop kullanılmaz."""
    out = np.full((n, 3), np.nan)
    acc = x.get("acc")
    if acc is None:
        return out
    roll, pitch = fu.accel_roll_pitch(acc)
    out[:, 0], out[:, 1] = roll, pitch
    mag = x.get("mag")
    if mag is not None:
        ok = np.isfinite(mag).all(axis=1) & (np.abs(mag).sum(axis=1) > 0)
        out[:, 2] = np.where(ok, fu.tilt_compensated_heading(mag, roll, pitch), np.nan)
    return out


def simulate(duration: float, fs: float, seed: int, mag: bool = True, **kw) -> Simulation:
    """`spt.dsp.fusion.simulate_imu` (gyro rad/s, acc m/s², mag µT → T). kw: simulate_imu parametreleri."""
    sim = fu.simulate_imu(duration=duration, fs=fs, seed=seed, **kw)
    inputs = {"gyro": sim.gyro, "acc": sim.acc}
    if mag:
        inputs["mag"] = sim.mag * 1e-6
    notes = (f"Analitik yönelim yörüngesi, {duration:g} s @ {fs:g} Hz. Jiroskop sapması "
             f"{fmt_vec(sim.gyro_bias * R2D)} °/s; ivmeölçerde doğrusal ivme ve gürültü"
             + ("; manyetometre (µT → T)." if mag else "; manyetometre yok."))
    return Simulation(fs, inputs, sim.truth, notes, {})


SCENARIO = Scenario(
    key="orientation",
    name="Yönelim (IMU / MARG)",
    description=("Jiroskop + ivmeölçer (+ manyetometre) ile roll / pitch / yaw kestirimi. Mevcut 6 yönelim "
                 "algoritması (Mahony, Madgwick, Kalman, EKF, UKF, parçacık filtresi) aynen kullanılır."),
    inputs=INPUTS,
    outputs=OUTPUTS,
    params=(),
    algorithms=ALGORITHMS,
    simulator=simulate,
    default_fs=200.0,
    raw_fn=raw,
)
