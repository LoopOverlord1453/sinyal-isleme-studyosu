"""Sistem tanımlama testlerinin ortak yardımcıları: deterministik veri kümeleri ve MATLAB kahini sarmalayıcısı.

Veri Python'da (sabit tohumla) üretilir ve `tests/data/matlab/sysid_data/<ad>.csv` dosyasına %.17g ile yazılır;
hem Python testleri hem MATLAB betikleri **aynı CSV'yi** okur (MATLAB: `readmatrix`). CSV varsa yeniden üretilmez
(numpy dağılım akışları sürümle değişse de veri sabit kalır).

MATLAB betikleri CSV klasörünü `{DATA}` yer tutucusuyla anar; betik özeti (sha256) yer tutuculu metin + kullanılan
CSV'lerin özetinden hesaplanır, böylece depo taşınsa da kayıtlı referans geçerli kalır, veri değişirse yeniden üretilir.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "tests" / "data" / "matlab" / "sysid_data"
FIXTURE_DIR = ROOT / "tests" / "data" / "matlab"


# --------------------------------------------------------------------------------------------- veri üreticileri
def _zoh_ct(num, den, u, Ts, delay=0.0, sub=50):
    """Sürekli zamanlı sistemin ZOH girişli tam (ince ızgaralı, ZOH ara değerli) benzetimi — bağımsız referans.

    Gecikme ince adımın tam katı olmalı (anahtarlanma anları ince ızgarada)."""
    h = Ts / sub
    nd = delay / h
    if abs(nd - round(nd)) > 1e-9:
        raise ValueError("gecikme ince adımın katı olmalı")
    nd = int(round(nd))
    n = len(u)
    uf = np.repeat(np.asarray(u, float), sub)
    uf = np.concatenate([np.zeros(nd), uf])[:n * sub]
    t = np.arange(n * sub) * h
    sysc = signal.lti(num, den)
    _, yf, _ = signal.lsim(sysc, uf, t, interp=False)
    return yf[::sub]


def _held_noise(rng, n, hold, scale=1.0):
    k = -(-n // hold)
    return np.repeat(rng.standard_normal(k) * scale, hold)[:n]


def _gen_arx1():
    rng = np.random.default_rng(101)
    n = 600
    u = rng.standard_normal(n)
    e = 0.2 * rng.standard_normal(n)
    A, B, C = [1, -1.5, 0.7], [0, 1.0, 0.5], [1, -0.8, 0.2]
    y = signal.lfilter(B, A, u) + signal.lfilter(C, A, e)
    return np.column_stack([y, u])


def _gen_oe1():
    rng = np.random.default_rng(202)
    n = 800
    u = rng.standard_normal(n)
    e = 0.1 * rng.standard_normal(n)
    y = signal.lfilter([0, 0, 0.4, 0.25], [1, -1.2, 0.5], u) + e
    return np.column_stack([y, u])


def _gen_bj1():
    rng = np.random.default_rng(303)
    n = 1500
    u = rng.standard_normal(n)
    e = 0.2 * rng.standard_normal(n)
    y = signal.lfilter([0, 0.5, 0.2], [1, -1.1, 0.3], u) + signal.lfilter([1, 0.6], [1, -0.9], e)
    return np.column_stack([y, u])


def _gen_miso1():
    rng = np.random.default_rng(404)
    n = 1000
    u1 = rng.standard_normal(n)
    u2 = _held_noise(rng, n, 3)
    e = 0.1 * rng.standard_normal(n)
    A = [1, -0.7]
    y = (signal.lfilter([0, 0.8, 0.3], A, u1) + signal.lfilter([0, 0, -0.5], A, u2)
         + signal.lfilter([1], A, e))
    return np.column_stack([y, u1, u2])


def _gen_ct1():
    """G(s) = (2s+8)/(s²+1.6s+4)·e^{-0.12 s}, Ts = 0.05 (gecikme 2.4 örnek); P2DZU: K=2, Tw=0.5, ζ=0.4, Tz=0.25."""
    rng = np.random.default_rng(505)
    n, Ts = 1000, 0.05
    u = _held_noise(rng, n, 6)
    y0 = _zoh_ct([2, 8], [1, 1.6, 4], u, Ts, delay=0.12, sub=50)
    y = y0 + 0.05 * rng.standard_normal(n)
    return np.column_stack([y, u])


def _gen_proc1():
    """P1D: K=1.5, Tp1=2, Td=0.35 (y1) ve P2D: K=-0.8, Tp1=1.5, Tp2=0.4, Td=0.25 (y2); Ts = 0.1."""
    rng = np.random.default_rng(606)
    n, Ts = 600, 0.1
    u = _held_noise(rng, n, 15)
    y1 = _zoh_ct([1.5], [2.0, 1.0], u, Ts, delay=0.35, sub=20) + 0.02 * rng.standard_normal(n)
    y2 = _zoh_ct([-0.8], np.polymul([1.5, 1], [0.4, 1]), u, Ts, delay=0.25, sub=20) + 0.02 * rng.standard_normal(n)
    return np.column_stack([y1, y2, u])


def _gen_mimo1():
    """2 giriş, 2 çıkış, 4. dereceden yenilik (innovations) biçimli ayrık sistem."""
    rng = np.random.default_rng(707)
    n = 2000
    A = np.array([[0.75, 0.35, 0, 0], [-0.35, 0.75, 0, 0], [0, 0, 0.5, 0.0], [0, 0, 0.2, -0.4]])
    B = np.array([[1.0, 0.0], [0.5, 0.3], [0.0, 1.0], [0.2, -0.6]])
    C = np.array([[1.0, 0.0, 0.5, 0.0], [0.0, 0.4, 0.0, 1.0]])
    K = np.array([[0.3, 0.0], [0.0, 0.2], [0.1, 0.1], [0.0, 0.25]])
    u = np.column_stack([rng.standard_normal(n), _held_noise(rng, n, 2)])
    e = 0.1 * rng.standard_normal((n, 2))
    x = np.zeros(4)
    y = np.zeros((n, 2))
    for t in range(n):
        y[t] = C @ x + e[t]
        x = A @ x + B @ u[t] + K @ e[t]
    return np.column_stack([y, u])


def _gen_ts1():
    rng = np.random.default_rng(808)
    n = 1000
    e = rng.standard_normal(n)
    y = signal.lfilter([1, 0.5], [1, -1.2, 0.6], e)
    return y[:, None]


def _gen_fir1():
    rng = np.random.default_rng(909)
    n = 300
    u = rng.standard_normal(n)
    g = signal.lfilter([0, 0.3, 0.2], [1, -1.4, 0.6], np.r_[1.0, np.zeros(59)])
    y = signal.lfilter(g, [1], u) + 0.3 * rng.standard_normal(n)
    return np.column_stack([y, u])


def _gen_nl1():
    """Doğrusal olmayan ARX: y = 0.5y1 − 0.2y2 + tanh(1.5u1) + 0.15·u1·y1 + 0.1u2 + e."""
    rng = np.random.default_rng(1010)
    n = 1500
    u = np.clip(_held_noise(rng, n, 2, 0.9), -2.5, 2.5)
    e = 0.02 * rng.standard_normal(n)
    y = np.zeros(n)
    for t in range(n):
        y1 = y[t - 1] if t >= 1 else 0.0
        y2 = y[t - 2] if t >= 2 else 0.0
        u1 = u[t - 1] if t >= 1 else 0.0
        u2 = u[t - 2] if t >= 2 else 0.0
        y[t] = 0.5 * y1 - 0.2 * y2 + np.tanh(1.5 * u1) + 0.15 * u1 * y1 + 0.1 * u2 + e[t]
    return np.column_stack([y, u])


def _gen_hw1():
    """Hammerstein–Wiener: v = doyum(u, [-0.8, 1.0]); x = B/F v; y = ölü bölge(x, [-0.2, 0.3]) + e."""
    rng = np.random.default_rng(1111)
    n = 1500
    u = rng.uniform(-2, 2, -(-n // 3)).repeat(3)[:n]
    v = np.clip(u, -0.8, 1.0)
    x = signal.lfilter([0, 0.6, 0.3], [1, -0.9, 0.2], v)
    y = np.where(x > 0.3, x - 0.3, np.where(x < -0.2, x + 0.2, 0.0)) + 0.01 * rng.standard_normal(n)
    return np.column_stack([y, u])


GENERATORS = {
    "arx1": _gen_arx1, "oe1": _gen_oe1, "bj1": _gen_bj1, "miso1": _gen_miso1, "ct1": _gen_ct1,
    "proc1": _gen_proc1, "mimo1": _gen_mimo1, "ts1": _gen_ts1, "fir1": _gen_fir1, "nl1": _gen_nl1,
    "hw1": _gen_hw1,
}


def dataset(name: str) -> np.ndarray:
    """Veri kümesini (N, sütun) CSV'den okur; yoksa üretip yazar."""
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        arr = GENERATORS[name]()
        np.savetxt(path, arr, delimiter=",", fmt="%.17g")
    arr = np.loadtxt(path, delimiter=",", ndmin=2)
    return arr


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------------- MATLAB kahini
def oracle(name: str, script: str, datasets: tuple[str, ...] = ()) -> dict:
    """`tools.matlab_oracle.fixture` gibi, ama `{DATA}` yer tutuculu ve CSV özetli (taşınabilir) önbellek."""
    from tools.matlab_oracle import OracleUnavailable, run

    for d in datasets:
        dataset(d)
    key = script.strip() + "\n%% data:" + ",".join(f"{d}={_file_sha(DATA_DIR / f'{d}.csv')}" for d in datasets)
    sha = hashlib.sha256(key.encode("utf-8")).hexdigest()
    path = FIXTURE_DIR / f"{name}.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("script_sha") == sha:
            return data["out"]
    real = script.replace("{DATA}", DATA_DIR.as_posix())
    try:
        out = run(real)
    except OracleUnavailable as exc:  # pragma: no cover - MATLAB'siz makinede
        pytest.skip(f"MATLAB referansı yok: {exc}")
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"script_sha": sha, "matlab": "R2025b", "datasets": list(datasets),
                                "script": script.strip(), "out": out}, ensure_ascii=False, indent=1,
                               allow_nan=True), encoding="utf-8")
    return out


def arr(x) -> np.ndarray:
    """JSON'dan gelen MATLAB dizisini numpy'ye çevirir (None → NaN)."""
    if x is None:
        return np.array(np.nan)
    return np.asarray(x, dtype=float)
