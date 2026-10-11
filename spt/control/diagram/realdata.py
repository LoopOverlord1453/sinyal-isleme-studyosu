"""Gerçek (ölçülmüş) veriyle benzetim ve karşılaştırma.

Ölçülen giriş zaman serisi bir Veri kaynağı bloğuna yerleştirilir, model ölçüm anlarında (ode45 bu anlara tam
basar; sabit adımda ızgara üzerinde olmalıdır) benzetilir ve seçilen çıkış ölçülen çıkışla karşılaştırılır.

Uyum (fit) yüzdesi MATLAB `compare` tanımıyla (NRMSE): fit = 100·(1 − ‖y − ŷ‖ / ‖y − ortalama(y)‖), kanal başına.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from spt.control.diagram.engine import SimResult, simulate
from spt.control.diagram.model import Model
from spt.control.errors import ControlError


def fit_percent(y: Any, yhat: Any) -> np.ndarray:
    """Kanal başına uyum yüzdesi (MATLAB compare 'NRMSE'): 100·(1 − ‖y − ŷ‖/‖y − ȳ‖). y sabitse NaN."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if yhat.ndim == 1:
        yhat = yhat.reshape(-1, 1)
    if y.shape != yhat.shape:
        raise ControlError(f"Ölçülen ({y.shape}) ve benzetilen ({yhat.shape}) çıkış boyutları farklı.")
    num = np.linalg.norm(y - yhat, axis=0)
    den = np.linalg.norm(y - y.mean(axis=0), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, 100.0 * (1.0 - num / den), np.nan)


@dataclass
class DataComparison:
    """Gerçek veriyle benzetim sonucu: t, benzetilen ve ölçülen çıkış (nt, ny), hata, fit %, RMSE, en büyük hata."""
    t: np.ndarray
    y_sim: np.ndarray
    y_meas: np.ndarray | None
    error: np.ndarray | None
    fit: np.ndarray | None
    rmse: np.ndarray | None
    max_error: np.ndarray | None
    result: SimResult

    def summary(self) -> str:
        if self.fit is None:
            return "Benzetim tamamlandı (karşılaştırılacak ölçüm verilmedi)."
        parts = []
        for k in range(self.fit.size):
            parts.append(f"kanal {k + 1}: uyum %{self.fit[k]:.2f}, RMSE {self.rmse[k]:.4g}, "
                         f"en büyük hata {self.max_error[k]:.4g}")
        return "; ".join(parts)


def simulate_with_data(model: Model, source: str | dict[str, Any], t: Any, u: Any = None, *, output: Any,
                       y_meas: Any = None, interp: str | None = None, solver: str | None = None,
                       step: float | None = None, **sim_options: Any) -> DataComparison:
    """Ölçülen giriş verisiyle benzetim.

    source: Veri kaynağı bloğunun adı (veri `u`) ya da {blok adı: değerler} (birden çok kaynak; hepsi aynı `t`).
    t: ölçüm zamanları (artan). u: (nt,) ya da (nt, kanal). output: karşılaştırılacak sinyal (ad, "Blok", "Blok:2"
    ya da kök çıkış portu numarası). y_meas: ölçülen çıkış (nt,) ya da (nt, kanal) — verilirse fit / hata hesaplanır.
    interp: Veri kaynağının ara değerlemesi ('linear' / 'zoh'; None: bloktaki ayar). solver: None ise ode45 (ölçüm
    anlarına tam basar); 'ode4' / 'ode1' için `step` verilmezse veri aralığı tekdüze ise o aralık kullanılır.
    """
    t = np.asarray(t, dtype=float).ravel()
    if t.size < 2 or np.any(np.diff(t) <= 0):
        raise ControlError("Ölçüm zamanları en az iki noktalı ve kesin artan olmalı.")
    sources = source if isinstance(source, dict) else {source: u}
    m = model.copy()
    for name, data in sources.items():
        if data is None:
            raise ControlError(f"'{name}' veri kaynağı için değer verilmedi.")
        blk = m.block(name)
        if blk.type_name != "DataSource":
            raise ControlError(f"'{name}' bir Veri kaynağı bloğu değil.")
        vals = np.asarray(data, dtype=float)
        if vals.ndim == 1:
            vals = vals.reshape(-1, 1)
        if vals.shape[0] != t.size:
            raise ControlError(f"'{name}' verisinin satır sayısı ({vals.shape[0]}) zaman sayısına ({t.size}) eşit "
                               "olmalı.")
        upd: dict[str, Any] = {"time": t, "values": vals}
        if interp is not None:
            upd["interp"] = interp
        blk.set_params(**upd)
    solver = solver or "ode45"
    opts = dict(sim_options)
    if solver in ("ode1", "ode4") and step is None:
        dt = np.diff(t)
        if np.max(np.abs(dt - dt[0])) > 1e-9 * dt[0]:
            raise ControlError("Ölçüm aralığı tekdüze değil; sabit adımlı çözücü için 'step' verin ya da ode45 "
                               "kullanın.")
        step = float(dt[0])
    log = []
    if not (isinstance(output, (int, np.integer)) and not isinstance(output, bool)):
        log = [output]
    res = simulate(m, float(t[-1]), solver=solver, step=step, start_time=float(t[0]), output_times=t, log=log,
                   **opts)
    if res.cancelled:
        raise ControlError("Benzetim iptal edildi.")
    if log:
        key = output if isinstance(output, str) else list(res.signals)[-1]
        ys = res.signals[key]
    else:
        if int(output) not in res.outputs:
            raise ControlError(f"{output} numaralı kök çıkış portu yok.")
        ys = res.outputs[int(output)]
    if ys.shape[0] != t.size or np.max(np.abs(res.t - t)) > 1e-9 * max(1.0, abs(t[-1])):
        raise ControlError("Benzetim çıktı anları ölçüm anlarıyla eşleşmedi.")
    if y_meas is None:
        return DataComparison(t, ys, None, None, None, None, None, res)
    ym = np.asarray(y_meas, dtype=float)
    if ym.ndim == 1:
        ym = ym.reshape(-1, 1)
    if ym.shape != ys.shape:
        raise ControlError(f"Ölçülen çıkış boyutu ({ym.shape}) benzetilen çıkışla ({ys.shape}) aynı olmalı.")
    err = ym - ys
    return DataComparison(t, ys, ym, err, fit_percent(ym, ys), np.sqrt(np.mean(err ** 2, axis=0)),
                          np.max(np.abs(err), axis=0), res)


__all__ = ["fit_percent", "simulate_with_data", "DataComparison"]
