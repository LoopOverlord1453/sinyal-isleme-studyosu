"""Tanımlama verisi (MATLAB `iddata` karşılığı), ön işleme ve veri okuyucular.

- `IDData(y, u=None, Ts=1.0, ...)`: zaman bölgesi, eşit aralıklı, çok girişli / çok çıkışlı, çok deneyli veri.
  Girişsiz (u=None) veri zaman serisidir. MATLAB'deki gibi `tstart` varsayılanı **Ts**'dir (R2025b'de ölçüldü:
  `iddata(y,u,0.1).Tstart = 0.1`); örnekleme anları `tstart + k·Ts`, k = 0…N−1.
- Ön işleme: `detrend` / `retrend` (MATLAB TrendInfo gibi geri alınabilir), `idfilt`, `resample`, alt aralık /
  kanal / deney seçimi (`data[örnekler, çıkışlar, girişler, deneyler]`, `getexp`), `merge`, `split`.
- Eksik örnekler (NaN): tahmin fonksiyonları açık hata verir; `fill_missing` doğrusal ara değerle ya da ARX modeliyle
  (misdata benzeri, kendi algoritmamız — ayrıntı fonksiyonda) doldurur.
- Okuyucular: `iddata_from_handle` (canlı akış / kayıt / anlık görüntü), `read_csv`, `read_mat` (+ `mat_variables`).
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal

from spt.control.errors import ControlError

__all__ = ["IDData", "iddata", "merge", "TrendInfo", "detrend", "retrend", "idfilt", "resample", "split",
           "fill_missing", "iddata_from_handle", "read_csv", "read_mat", "mat_variables"]


def _as_2d(x, what: str) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if np.iscomplexobj(x):
        raise ControlError(f"{what} karmaşık olamaz (gerçel ölçüm verisi gerekli)")
    if a.ndim == 0:
        a = a.reshape(1, 1)
    elif a.ndim == 1:
        a = a[:, None]
    elif a.ndim != 2:
        raise ControlError(f"{what} (örnek × kanal) biçiminde 1-B ya da 2-B olmalı")
    return np.ascontiguousarray(a)


def _names(given, n: int, prefix: str) -> tuple[str, ...]:
    if given is None:
        return tuple(f"{prefix}{i + 1}" for i in range(n))
    if isinstance(given, str):
        given = (given,)
    given = tuple(str(g) for g in given)
    if len(given) != n:
        raise ControlError(f"{len(given)} ad verildi, {n} kanal var")
    return given


def _units(given, n: int) -> tuple[str, ...]:
    if given is None:
        return ("",) * n
    if isinstance(given, str):
        given = (given,)
    given = tuple(str(g) for g in given)
    if len(given) != n:
        raise ControlError(f"{len(given)} birim verildi, {n} kanal var")
    return given


def _is_multi(x) -> bool:
    return isinstance(x, (list, tuple)) and len(x) > 0 and all(isinstance(e, np.ndarray) for e in x)


class IDData:
    """Giriş / çıkış ölçüm verisi (MATLAB `iddata`).

    Parametreler
    - y: (N,) ya da (N, ny) dizi; çok deneyli veri için dizilerden oluşan liste / demet.
    - u: (N,) ya da (N, nu) dizi, deney listesi ya da None (zaman serisi).
    - Ts: örnekleme aralığı (s, > 0).
    - tstart: ilk örneğin zamanı (deney başına liste olabilir); varsayılan Ts (MATLAB gibi).
    - input_names / output_names / input_units / output_units, name, experiment_names.
    - intersample: girişlerin örnekler arası davranışı ('zoh' ya da 'foh'; sürekli zamanlı modellerin
      benzetiminde kullanılır). Tek değer ya da giriş başına demet.

    Özellikler: `y`, `u` (tek deneyde dizi, çok deneyde liste — MATLAB gibi), `ys`, `us` (her zaman liste),
    `Ts`, `tstart`, `ny`, `nu`, `ne`, `N` (tek deneyde int, çok deneyde liste), `time`, `is_timeseries`."""

    def __init__(self, y=None, u=None, Ts: float = 1.0, *, tstart=None, input_names=None, output_names=None,
                 input_units=None, output_units=None, name: str = "", intersample="zoh",
                 experiment_names=None):
        if y is None:
            raise ControlError("Çıkış verisi (y) verilmeli")
        ys = [_as_2d(e, "Çıkış verisi") for e in y] if _is_multi(y) else [_as_2d(y, "Çıkış verisi")]
        if u is None:
            us = [np.zeros((len(e), 0)) for e in ys]
        elif _is_multi(u):
            us = [_as_2d(e, "Giriş verisi") for e in u]
        else:
            us = [_as_2d(u, "Giriş verisi")]
        if len(us) != len(ys):
            raise ControlError(f"Giriş {len(us)} deney, çıkış {len(ys)} deney içeriyor")
        ny, nu = ys[0].shape[1], us[0].shape[1]
        for k, (a, b) in enumerate(zip(ys, us)):
            if a.shape[1] != ny or b.shape[1] != nu:
                raise ControlError("Tüm deneylerde kanal sayıları aynı olmalı")
            if len(a) != len(b):
                raise ControlError(f"Deney {k + 1}: giriş ({len(b)}) ve çıkış ({len(a)}) örnek sayıları farklı")
            if len(a) == 0:
                raise ControlError("Veri boş")
            if np.isinf(a).any() or np.isinf(b).any():
                raise ControlError("Veride sonsuz değer var")
        if ny == 0:
            raise ControlError("En az bir çıkış kanalı gerekli")
        Ts = float(Ts)
        if not np.isfinite(Ts) or Ts <= 0:
            raise ControlError("Örnekleme aralığı Ts pozitif olmalı (zaman bölgesi verisi)")
        self._y, self._u = ys, us
        self.Ts = Ts
        ne = len(ys)
        if tstart is None:
            ts = [Ts] * ne
        elif np.ndim(tstart) == 0:
            ts = [float(tstart)] * ne
        else:
            ts = [float(t) for t in tstart]
            if len(ts) != ne:
                raise ControlError("tstart deney sayısı kadar olmalı")
        self._tstart = ts
        self.output_names = _names(output_names, ny, "y")
        self.input_names = _names(input_names, nu, "u")
        self.output_units = _units(output_units, ny)
        self.input_units = _units(input_units, nu)
        if isinstance(intersample, str):
            intersample = (intersample,) * nu
        intersample = tuple(str(s).lower() for s in intersample)
        if len(intersample) != nu or any(s not in ("zoh", "foh") for s in intersample):
            raise ControlError("Örnekler arası davranış giriş başına 'zoh' ya da 'foh' olmalı")
        self.intersample = intersample
        self.name = str(name)
        self.experiment_names = _names(experiment_names, ne, "Exp")

    # ------------------------------------------------------------------ erişim
    @property
    def ys(self) -> list[np.ndarray]:
        return self._y

    @property
    def us(self) -> list[np.ndarray]:
        return self._u

    @property
    def y(self):
        return self._y[0] if len(self._y) == 1 else list(self._y)

    @property
    def u(self):
        return self._u[0] if len(self._u) == 1 else list(self._u)

    @property
    def ny(self) -> int:
        return self._y[0].shape[1]

    @property
    def nu(self) -> int:
        return self._u[0].shape[1]

    @property
    def ne(self) -> int:
        return len(self._y)

    @property
    def N(self):
        n = [len(e) for e in self._y]
        return n[0] if len(n) == 1 else n

    @property
    def lengths(self) -> list[int]:
        return [len(e) for e in self._y]

    @property
    def tstart(self):
        return self._tstart[0] if len(self._tstart) == 1 else list(self._tstart)

    @property
    def tstarts(self) -> list[float]:
        return list(self._tstart)

    @property
    def time(self):
        """Örnekleme anları (s) — tek deneyde dizi, çok deneyde liste."""
        t = [ts + self.Ts * np.arange(len(e)) for ts, e in zip(self._tstart, self._y)]
        return t[0] if len(t) == 1 else t

    @property
    def is_timeseries(self) -> bool:
        return self.nu == 0

    @property
    def has_missing(self) -> bool:
        return any(np.isnan(a).any() for a in self._y + self._u)

    def size(self) -> tuple:
        """MATLAB `size(data)`: (N, ny, nu, ne)."""
        return (self.N, self.ny, self.nu, self.ne)

    def __len__(self) -> int:
        return sum(self.lengths)

    def __repr__(self) -> str:
        kind = "Zaman serisi" if self.is_timeseries else "Zaman bölgesi verisi"
        n = self.N
        return (f"<IDData {kind}: {n} örnek, {self.ny} çıkış, {self.nu} giriş, {self.ne} deney, Ts={self.Ts:g} s"
                + (f", '{self.name}'" if self.name else "") + ">")

    def _clone(self, ys, us, tstart=None, out_idx=None, in_idx=None, exp_idx=None, **kw) -> IDData:
        out_idx = range(self.ny) if out_idx is None else out_idx
        in_idx = range(self.nu) if in_idx is None else in_idx
        exp_idx = range(self.ne) if exp_idx is None else exp_idx
        args = dict(
            Ts=self.Ts, tstart=[self._tstart[i] for i in exp_idx] if tstart is None else tstart,
            output_names=[self.output_names[i] for i in out_idx], input_names=[self.input_names[i] for i in in_idx],
            output_units=[self.output_units[i] for i in out_idx], input_units=[self.input_units[i] for i in in_idx],
            name=self.name, intersample=tuple(self.intersample[i] for i in in_idx),
            experiment_names=[self.experiment_names[i] for i in exp_idx])
        args.update(kw)
        return IDData(list(ys), list(us), **args)

    def copy(self) -> IDData:
        return self._clone([a.copy() for a in self._y], [b.copy() for b in self._u])

    def _chan(self, sel, names, what) -> list[int]:
        n = len(names)
        if sel is None or (isinstance(sel, slice) and sel == slice(None)):
            return list(range(n))
        if isinstance(sel, (str, int, np.integer)):
            sel = [sel]
        if isinstance(sel, slice):
            return list(range(n))[sel]
        out = []
        for s in sel:
            if isinstance(s, str):
                if s not in names:
                    raise ControlError(f"'{s}' adlı {what} yok")
                out.append(names.index(s))
            else:
                i = int(s)
                if not -n <= i < n:
                    raise ControlError(f"{what} indisi {i} aralık dışında")
                out.append(i % n)
        return out

    def select(self, samples=None, outputs=None, inputs=None, experiments=None) -> IDData:
        """Alt küme (MATLAB `data(samples, outputs, inputs, experiments)`); örnek seçimi her deneye uygulanır.

        `samples`: dilim ya da (bitişik) indis dizisi — örnekleme anları korunur (tstart kayar)."""
        ei = self._chan(experiments, self.experiment_names, "deney")
        oi = self._chan(outputs, self.output_names, "çıkış")
        ii = self._chan(inputs, self.input_names, "giriş")
        if not oi:
            raise ControlError("En az bir çıkış seçilmeli")
        ys, us, ts = [], [], []
        for e in ei:
            y, u, t0 = self._y[e], self._u[e], self._tstart[e]
            if samples is not None:
                idx = np.arange(len(y))[samples] if isinstance(samples, slice) else np.asarray(samples, int)
                idx = np.atleast_1d(idx)
                if idx.size == 0:
                    raise ControlError("Seçilen örnek aralığı boş")
                if idx.size > 1 and np.any(np.diff(idx) != 1):
                    raise ControlError("Örnek seçimi bitişik ve artan olmalı (seyreltme için resample kullanın)")
                if idx[0] < 0 or idx[-1] >= len(y):
                    raise ControlError("Örnek indisi aralık dışında")
                y, u = y[idx], u[idx]
                t0 = t0 + idx[0] * self.Ts
            ys.append(y[:, oi])
            us.append(u[:, ii])
            ts.append(t0)
        return self._clone(ys, us, tstart=ts, out_idx=oi, in_idx=ii, exp_idx=ei)

    def __getitem__(self, key) -> IDData:
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) > 4:
            raise ControlError("En çok 4 indis: örnekler, çıkışlar, girişler, deneyler")
        key = key + (None,) * (4 - len(key))
        s = key[0]
        if isinstance(s, slice) and s == slice(None):
            s = None
        return self.select(s, key[1], key[2], key[3])

    def getexp(self, experiments) -> IDData:
        """Deney(ler)i seçer (MATLAB `getexp`); 0 tabanlı indis ya da ad."""
        return self.select(experiments=experiments)

    def with_signals(self, ys, us=None) -> IDData:
        """Aynı üst veriyle yeni sinyaller (iç kullanım: filtrelenmiş / ön işlenmiş veri)."""
        if us is None:
            us = self._u
        return self._clone(ys, us)


def iddata(y, u=None, Ts: float = 1.0, **kw) -> IDData:
    """MATLAB `iddata(y, u, Ts)` kısayolu."""
    return IDData(y, u, Ts, **kw)


def merge(*datas: IDData) -> IDData:
    """Deneyleri birleştirir (MATLAB `merge`): kanal sayıları ve Ts aynı olmalı."""
    if not datas:
        raise ControlError("Birleştirilecek veri yok")
    d0 = datas[0]
    ys, us, ts, names = [], [], [], []
    for d in datas:
        if d.ny != d0.ny or d.nu != d0.nu:
            raise ControlError("Birleştirilen verilerin kanal sayıları aynı olmalı")
        if abs(d.Ts - d0.Ts) > 1e-12 * d0.Ts:
            raise ControlError("Birleştirilen verilerin örnekleme aralıkları aynı olmalı")
        ys += d.ys
        us += d.us
        ts += d.tstarts
        names += list(d.experiment_names)
    if len(set(names)) != len(names):
        names = [f"Exp{i + 1}" for i in range(len(names))]
    return IDData(ys, us, d0.Ts, tstart=ts, output_names=d0.output_names, input_names=d0.input_names,
                  output_units=d0.output_units, input_units=d0.input_units, name=d0.name,
                  intersample=d0.intersample, experiment_names=names)


def split(data: IDData, fraction: float = 0.5) -> tuple[IDData, IDData]:
    """Tahmin / doğrulama bölmesi: her deneyin ilk `fraction` kısmı tahmin, kalanı doğrulama verisi."""
    if not 0 < fraction < 1:
        raise ControlError("Bölme oranı 0 ile 1 arasında olmalı")
    est, val = [], []
    for e in range(data.ne):
        d = data.getexp(e)
        n = d.N
        k = int(round(n * fraction))
        if k < 1 or k >= n:
            raise ControlError("Bölme sonrası parçalardan biri boş kalıyor")
        est.append(d[:k])
        val.append(d[k:])
    return (est[0] if len(est) == 1 else merge(*est)), (val[0] if len(val) == 1 else merge(*val))


def check_complete(data: IDData, what: str = "Tahmin") -> None:
    """NaN içeren veride açık hata (eksik örnekleri doldurmak için fill_missing önerilir)."""
    for k, (y, u) in enumerate(zip(data.ys, data.us)):
        bad = np.isnan(y).any(axis=1) | (np.isnan(u).any(axis=1) if u.shape[1] else False)
        if bad.any():
            first = int(np.flatnonzero(bad)[0])
            exp = f"deney {k + 1}, " if data.ne > 1 else ""
            raise ControlError(f"{what} verisinde {int(bad.sum())} eksik (NaN) örnek var ({exp}ilki {first}. örnek). "
                               "Eksik örnekleri doldurun (Eksikleri doldur: doğrusal ya da modelle) ya da eksiksiz "
                               "bir aralık seçin.")


# ======================================================================================== trend
@dataclass
class TrendInfo:
    """MATLAB `TrendInfo`: deney başına kaydırma ve eğim. Değerler (ne, kanal) dizileridir; eğim birimi 1/s.

    Trend: kanal(k) = offset + slope·(k·Ts), k = 0… ilk örnekten itibaren (R2025b'de ölçüldü)."""
    input_offset: np.ndarray
    output_offset: np.ndarray
    input_slope: np.ndarray
    output_slope: np.ndarray
    Ts: float
    data_name: str = ""

    @property
    def InputOffset(self):
        return self.input_offset[0] if len(self.input_offset) == 1 else self.input_offset

    @property
    def OutputOffset(self):
        return self.output_offset[0] if len(self.output_offset) == 1 else self.output_offset

    @property
    def InputSlope(self):
        return self.input_slope[0] if len(self.input_slope) == 1 else self.input_slope

    @property
    def OutputSlope(self):
        return self.output_slope[0] if len(self.output_slope) == 1 else self.output_slope


def _fit_trend(x: np.ndarray, Ts: float, order: int):
    n, m = x.shape
    if m == 0:
        return np.zeros(0), np.zeros(0)
    if np.isnan(x).any():
        off = np.zeros(m)
        slope = np.zeros(m)
        t = np.arange(n) * Ts
        for j in range(m):
            ok = ~np.isnan(x[:, j])
            if order == 0 or ok.sum() < 2:
                off[j] = np.mean(x[ok, j]) if ok.any() else 0.0
            else:
                slope[j], off[j] = np.polyfit(t[ok], x[ok, j], 1)
        return off, slope
    if order == 0:
        return x.mean(axis=0), np.zeros(m)
    t = np.arange(n) * Ts
    M = np.column_stack([np.ones(n), t])
    coef = np.linalg.lstsq(M, x, rcond=None)[0]
    return coef[0], coef[1]


def detrend(data: IDData, order: int = 0, *, return_trend: bool = False):
    """Ortalama (order=0) ya da doğrusal trend (order=1) çıkarır (MATLAB `detrend(data, order)`).

    `return_trend=True` ise (veri, TrendInfo) döner; `retrend` ile geri eklenir. Doğrusal trend, örnekleme
    zamanına (ilk örnekten itibaren) karşı en küçük karelerle uydurulur."""
    if order not in (0, 1):
        raise ControlError("Trend derecesi 0 (ortalama) ya da 1 (doğrusal) olmalı")
    ys, us = [], []
    io, oo, is_, os_ = [], [], [], []
    for y, u in zip(data.ys, data.us):
        t = (np.arange(len(y)) * data.Ts)[:, None]
        a, b = _fit_trend(y, data.Ts, order)
        c, d = _fit_trend(u, data.Ts, order)
        ys.append(y - a - t * b)
        us.append(u - c - t * d if u.shape[1] else u.copy())
        oo.append(a)
        os_.append(b)
        io.append(c)
        is_.append(d)
    T = TrendInfo(np.array(io).reshape(data.ne, data.nu), np.array(oo).reshape(data.ne, data.ny),
                  np.array(is_).reshape(data.ne, data.nu), np.array(os_).reshape(data.ne, data.ny),
                  data.Ts, data.name)
    out = data.with_signals(ys, us)
    return (out, T) if return_trend else out


def retrend(data: IDData, trend: TrendInfo) -> IDData:
    """`detrend`'in çıkardığı trendi geri ekler (MATLAB `retrend`)."""
    if trend.output_offset.shape[1] != data.ny or trend.input_offset.shape[1] != data.nu:
        raise ControlError("Trend bilgisi verinin kanal sayısına uymuyor")
    if len(trend.output_offset) not in (1, data.ne):
        raise ControlError("Trend bilgisi verinin deney sayısına uymuyor")
    ys, us = [], []
    for k, (y, u) in enumerate(zip(data.ys, data.us)):
        e = k if len(trend.output_offset) > 1 else 0
        t = (np.arange(len(y)) * data.Ts)[:, None]
        ys.append(y + trend.output_offset[e] + t * trend.output_slope[e])
        us.append(u + trend.input_offset[e] + t * trend.input_slope[e] if u.shape[1] else u.copy())
    return data.with_signals(ys, us)


# ======================================================================================== süzme
def _filter_coeffs(filt, Ts):
    """{num, den} (q⁻¹ katsayıları) ya da LTI benzeri nesne → (b, a) ayrık katsayılar."""
    if isinstance(filt, (tuple, list)) and len(filt) == 2:
        return np.atleast_1d(np.asarray(filt[0], float)), np.atleast_1d(np.asarray(filt[1], float))
    if hasattr(filt, "_dtf"):  # tanımlama modeli (IDPoly …): ölçüm modeli G
        b, a = filt._siso_dtf(Ts)
        return b, a
    num = getattr(filt, "num", None)
    den = getattr(filt, "den", None)
    if num is not None and den is not None:  # spt.control.lti TransferFunction (azalan kuvvetler, z)
        fTs = float(getattr(filt, "Ts", 0.0))
        if fTs <= 0:
            raise ControlError("Süzgeç ayrık zamanlı olmalı (sürekli zamanlı süzgeci önce ayrıklaştırın)")
        if abs(fTs - Ts) > 1e-9 * Ts:
            raise ControlError("Süzgecin örnekleme aralığı veriyle aynı olmalı")
        n = np.atleast_1d(np.asarray(num[0][0] if isinstance(num, (list, tuple)) else num, float))
        d = np.atleast_1d(np.asarray(den[0][0] if isinstance(den, (list, tuple)) else den, float))
        n = np.pad(n, (len(d) - len(n), 0)) if len(n) < len(d) else n
        return n, d  # z'nin azalan kuvvetleri = q⁻¹'in artan kuvvetleri (aynı uzunlukta)
    raise ControlError("Süzgeç {pay, payda} ya da ayrık transfer fonksiyonu olmalı")


def idfilt(data: IDData, filt=None, *, order: int | None = None, band=None, kind: str = "low",
           causal: bool = True) -> IDData:
    """Veriyi süzer (MATLAB `idfilt`): tüm giriş ve çıkış kanalları aynı süzgeçten geçer.

    - `filt={num, den}` (q⁻¹ katsayıları, MATLAB `filter` anlamı) ya da ayrık transfer fonksiyonu.
    - ya da `order=N, band=Wn` (Nyquist'e göre normalize, 0<Wn<1; bant için [Wl, Wh]) ile Butterworth
      (`kind`: 'low', 'high', 'bandpass', 'bandstop'; MATLAB `butter` ile aynı tasarım).
    - `causal=False`: sıfır fazlı (ileri-geri) süzme (`filtfilt`, MATLAB filtfilt ile aynı kenar işlemi)."""
    if filt is None:
        if order is None or band is None:
            raise ControlError("Süzgeç ya da (derece, bant) verilmeli")
        Wn = np.atleast_1d(np.asarray(band, float))
        if np.any(Wn <= 0) or np.any(Wn >= 1):
            raise ControlError("Kesim frekansları Nyquist'e göre 0 ile 1 arasında olmalı")
        btype = {"low": "lowpass", "high": "highpass", "bandpass": "bandpass", "bandstop": "bandstop"}.get(kind)
        if btype is None:
            raise ControlError("Süzgeç türü 'low', 'high', 'bandpass' ya da 'bandstop' olmalı")
        if len(Wn) == 2 and kind in ("low", "high"):
            btype = "bandpass"
        b, a = signal.butter(int(order), Wn if len(Wn) > 1 else Wn[0], btype=btype)
    else:
        b, a = _filter_coeffs(filt, data.Ts)
    if a[0] == 0:
        raise ControlError("Süzgeç paydasının ilk katsayısı sıfır olamaz")

    def run(x):
        if x.shape[1] == 0:
            return x.copy()
        if np.isnan(x).any():
            raise ControlError("Eksik (NaN) örnek içeren veri süzülemez; önce doldurun")
        if causal:
            return signal.lfilter(b, a, x, axis=0)
        padlen = 3 * (max(len(a), len(b)) - 1)
        if len(x) <= padlen:
            raise ControlError("Sıfır fazlı süzme için veri çok kısa")
        return signal.filtfilt(b, a, x, axis=0, padtype="odd", padlen=padlen)

    return data.with_signals([run(y) for y in data.ys], [run(u) for u in data.us])


def resample(data: IDData, P: int, Q: int, order: int = 10) -> IDData:
    """Örnekleme hızını P/Q oranında değiştirir — MATLAB `resample(iddata, P, Q)` tanımı (R2025b kaynağıyla aynı):

    - çıkışlar (ve Q ≠ 1 ise girişler): sinyal baştan ve sondan N örneklik sabit uzatmayla gömülür, Signal Processing
      `resample(x, P, Q, n)` süzgeciyle (Kaiser β=5 pencereli firls, uzunluk 2·n·max(P,Q)+1) yeniden örneklenir ve
      ⌊N·P/Q⌋+1. örnekten ⌈N·P/Q⌉ örnek alınır; Q ≠ 1 iken 'zoh' girişler 'foh' olur;
    - saf yukarı örneklemede (Q = 1) girişler tam: 'zoh' tekrarla, 'foh' doğrusal ara değerle;
    - özgün zaman aralığını (tstart … son örnek) aşan örnekler atılır. Yeni Ts = Ts·Q/P, tstart korunur."""
    P0, Q0 = int(P), int(Q)
    if P0 < 1 or Q0 < 1:
        raise ControlError("P ve Q pozitif tamsayı olmalı")
    g = np.gcd(P0, Q0)
    P, Q = P0 // g, Q0 // g
    if P == Q:
        return data.copy()
    pq = max(P, Q)
    fc = 1.0 / (2 * pq)
    L = 2 * order * pq + 1
    if order > 0:
        h = signal.firls(L, [0, 2 * fc, 2 * fc, 1], [1, 1, 0, 0]) * signal.windows.kaiser(L, 5.0)
    else:
        h = np.ones(P)
    h = P * h / h.sum()
    Tsr = data.Ts * Q0 / P0

    def embedded(x):
        ns = len(x)
        E = np.vstack([np.repeat(x[:1], ns, axis=0), x, np.repeat(x[-1:], ns, axis=0)])
        r = _mat_resample(E, P, Q, h)
        x0 = int(np.floor(ns * P / Q))
        return r[x0:x0 + int(np.ceil(ns * P / Q))]

    ys, us, inter = [], [], list(data.intersample)
    for y, u in zip(data.ys, data.us):
        if np.isnan(y).any() or np.isnan(u).any():
            raise ControlError("Eksik (NaN) örnek içeren veri yeniden örneklenemez; önce doldurun")
        ns = len(y)
        yr = embedded(y)
        if u.shape[1] == 0:
            ur = np.zeros((len(yr), 0))
        elif Q == 1:
            ur = np.zeros((ns * P, u.shape[1]))
            for j in range(u.shape[1]):
                if data.intersample[j] == "zoh":
                    ur[:, j] = np.repeat(u[:, j], P)
                else:  # foh: örnekler arası doğrusal, son örnek tutulur
                    seg = u[:-1, j][:, None] + (u[1:, j] - u[:-1, j])[:, None] * (np.arange(P) / P)[None, :]
                    ur[:, j] = np.r_[seg.ravel(), np.repeat(u[-1, j], P)]
            ur = ur[:len(yr)] if len(ur) >= len(yr) else ur
        else:
            ur = embedded(u)
            inter = ["foh" if s == "zoh" else s for s in inter]
        n = min(len(yr), len(ur)) if u.shape[1] else len(yr)
        t_orig_end = (ns - 1) * data.Ts
        keep = np.arange(n) * Tsr <= t_orig_end + 1e-12 * max(abs(t_orig_end), 1.0)
        ys.append(yr[:n][keep])
        us.append(ur[:n][keep])
    out = data._clone(ys, us, intersample=tuple(inter))
    out.Ts = Tsr
    return out


def _mat_resample(x: np.ndarray, p: int, q: int, h: np.ndarray) -> np.ndarray:
    """MATLAB `resample` (uniform) gövdesi: upfirdn + gecikme dengelemesi."""
    Lx = x.shape[0]
    Lhalf = (len(h) - 1) // 2
    # MATLAB: süzgeci önden sıfırla doldurur (nz = q − mod(Lhalf, q); hizalıysa da q kadar) — aynen
    nz = q - Lhalf % q
    h = np.concatenate([np.zeros(nz), h])
    delay = (Lhalf + nz) // q
    Ly = int(np.ceil(Lx * p / q))
    Lh = len(h)
    nz1 = 0
    while int(np.ceil(((Lx - 1) * p + Lh + nz1) / q)) - delay < Ly:
        nz1 += 1
    h = np.concatenate([h, np.zeros(nz1)])
    y = signal.upfirdn(h, x, up=p, down=q, axis=0)
    return y[delay:delay + Ly]


# ======================================================================================== eksik veri
def fill_missing(data: IDData, method: str = "linear", *, orders=None, max_iter: int = 20,
                 tol: float = 1e-7) -> IDData:
    """Eksik (NaN) örnekleri doldurur.

    - 'linear': kanal başına doğrusal ara değer (uçlarda en yakın değer).
    - 'model': MATLAB `misdata` benzeri (kendi algoritmamız): doğrusal ara değerle başlar, sonra dönüşümlü olarak
      (i) ARX modelini kestirir, (ii) eksik değerleri, modelin bir adım öngörü hatalarının kareleri toplamını en
      küçükleyen değerlerle (doğrusal en küçük kareler; ARX hataları veride doğrusaldır) değiştirir. `orders`
      [na, nb, nk] (varsayılan her kanal için [4, 4, 1]). Ölçülen değerlere dokunulmaz."""
    if method not in ("linear", "model"):
        raise ControlError("Doldurma yöntemi 'linear' ya da 'model' olmalı")

    def lin(x):
        x = x.copy()
        for j in range(x.shape[1]):
            c = x[:, j]
            bad = np.isnan(c)
            if bad.all():
                raise ControlError("Bir kanalın tüm örnekleri eksik")
            if bad.any():
                idx = np.arange(len(c))
                c[bad] = np.interp(idx[bad], idx[~bad], c[~bad])
        return x

    filled = data.with_signals([lin(y) for y in data.ys], [lin(u) for u in data.us])
    if method == "linear" or not data.has_missing:
        return filled
    from spt.control.sysid.linear import _arx_regression, _arx_orders  # döngüsel içe aktarmayı önler

    ny, nu = data.ny, data.nu
    if orders is None:
        orders = [4, 4, 1] if nu else [4]
    na, nb, nk = _arx_orders(orders, ny, nu)
    masks = [(np.isnan(y), np.isnan(u)) for y, u in zip(data.ys, data.us)]
    ys = [y.copy() for y in filled.ys]
    us = [u.copy() for u in filled.us]
    prev = None
    for _ in range(max_iter):
        cur = filled.with_signals(ys, us)
        theta = []
        for i in range(ny):
            Phi, Y = _arx_regression(cur, i, na, nb, nk, start="estimate")
            theta.append(np.linalg.lstsq(Phi, Y, rcond=None)[0])
        # Eksik değerler için doğrusal en küçük kareler: e = Y − Φθ, her ikisi de veride doğrusal.
        new_ys, new_us = [], []
        for k in range(data.ne):
            my, mu = masks[k]
            miss = np.concatenate([np.flatnonzero(my.ravel(order="F")),
                                   my.size + np.flatnonzero(mu.ravel(order="F"))])
            if len(miss) == 0:
                new_ys.append(ys[k])
                new_us.append(us[k])
                continue
            base = np.concatenate([ys[k].ravel(order="F"), us[k].ravel(order="F")])

            def resid_of(vec):
                yy = vec[:my.size].reshape(my.shape, order="F")
                uu = vec[my.size:].reshape(mu.shape, order="F")
                d1 = filled.getexp(k).with_signals([yy], [uu])
                r = []
                for i in range(ny):
                    Phi, Y = _arx_regression(d1, i, na, nb, nk, start="estimate")
                    r.append(Y - Phi @ theta[i])
                return np.concatenate(r)

            v0 = base.copy()
            v0[miss] = 0.0
            r0 = resid_of(v0)
            M = np.empty((len(r0), len(miss)))
            for j, m in enumerate(miss):
                v = v0.copy()
                v[m] = 1.0
                M[:, j] = resid_of(v) - r0
            sol = np.linalg.lstsq(M, -r0, rcond=None)[0]
            v = base.copy()
            v[miss] = sol
            new_ys.append(v[:my.size].reshape(my.shape, order="F"))
            new_us.append(v[my.size:].reshape(mu.shape, order="F"))
        ys, us = new_ys, new_us
        vec = np.concatenate([np.concatenate([a.ravel(), b.ravel()]) for a, b in zip(ys, us)])
        if prev is not None and np.linalg.norm(vec - prev) <= tol * max(np.linalg.norm(vec), 1e-300):
            break
        prev = vec
    return filled.with_signals(ys, us)


# ======================================================================================== okuyucular
def iddata_from_handle(handle, inputs=(), outputs=(), n0: int | None = None, n1: int | None = None, *,
                       name: str | None = None) -> IDData:
    """Canlı akış / kayıt / anlık görüntü tutamacından veri (fiziksel birimde).

    `inputs` / `outputs`: kanal indisleri ya da adları (`handle.desc.channels[i].name`). `[n0, n1)` mutlak örnek
    aralığı (varsayılan: tamponda geçerli tüm veri). Kayıp bölgeler NaN olur (tahminden önce doldurulmalı ya da
    eksiksiz aralık seçilmeli). Karmaşık örnek tipli kanallar desteklenmez."""
    from spt.dsp.convert import physical_column

    desc = handle.desc
    chans = list(desc.channels)
    names = [c.name or f"ch{i}" for i, c in enumerate(chans)]

    def resolve(sel, what):
        if isinstance(sel, (str, int, np.integer)):
            sel = [sel]
        out = []
        for s in sel:
            if isinstance(s, str):
                if s not in names:
                    raise ControlError(f"'{s}' adlı kanal yok ({what})")
                out.append(names.index(s))
            else:
                i = int(s)
                if not 0 <= i < len(chans):
                    raise ControlError(f"Kanal indisi {i} aralık dışında ({what})")
                out.append(i)
        return out

    oi, ii = resolve(outputs, "çıkış"), resolve(inputs, "giriş")
    if not oi:
        raise ControlError("En az bir çıkış kanalı seçilmeli")
    if set(oi) & set(ii):
        raise ControlError("Bir kanal hem giriş hem çıkış olamaz")
    if desc.sample_type.is_complex:
        raise ControlError("Karmaşık (I/Q) örnekli akış sistem tanımlamada kullanılamaz")
    ring = handle.ring
    lo = ring.valid_from()
    hi = handle.latest_index()
    a = lo if n0 is None else max(int(n0), lo)
    b = hi if n1 is None else min(int(n1), hi)
    if b <= a:
        raise ControlError("Seçilen aralıkta veri yok")
    first, raw = ring.read(a, b)
    if len(raw) == 0:
        raise ControlError("Seçilen aralıkta veri yok")
    if raw.ndim == 1:
        raw = raw[:, None]

    def col(i):
        return np.asarray(physical_column(raw[:, i], desc.sample_type, chans[i]), float)

    Y = np.column_stack([col(i) for i in oi])
    U = np.column_stack([col(i) for i in ii]) if ii else None
    for gs, ge in ring.gaps_between(first, first + len(raw)):
        s, e = max(gs, first) - first, min(ge, first + len(raw)) - first
        Y[s:e] = np.nan
        if U is not None:
            U[s:e] = np.nan
    fs = float(handle.fs)
    return IDData(Y, U, 1.0 / fs, tstart=float(handle.t0) + first / fs,
                  output_names=[names[i] for i in oi], input_names=[names[i] for i in ii],
                  output_units=[chans[i].unit for i in oi], input_units=[chans[i].unit for i in ii],
                  name=name if name is not None else (desc.name or getattr(handle, "key", "")))


def _sniff(text: str):
    sample = text[:20000]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t ")
        delim = dialect.delimiter
    except csv.Error:
        delim = "," if "," in sample else (";" if ";" in sample else None)
    return delim


def _parse_float(s: str) -> float:
    s = s.strip()
    if s == "" or s.lower() in ("nan", "na", "null"):
        return np.nan
    return float(s)


def read_csv(path, *, outputs, inputs=(), Ts: float | None = None, time_column=None, delimiter: str | None = None,
             decimal: str = ".", name: str | None = None) -> IDData:
    """CSV / TXT dosyasından veri. Sütunlar indisle (0 tabanlı) ya da başlık adıyla seçilir.

    Örnekleme aralığı `Ts` ile verilir ya da `time_column` (zaman sütunu) eşit aralıklıysa ondan bulunur. Boş /
    'nan' hücreler eksik örnek (NaN) olur. Ondalık ayırıcı virgülse `decimal=','` (sütun ayırıcı ';' olmalı)."""
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    if delimiter is None:
        delimiter = _sniff(text)
    rows = list(csv.reader(io.StringIO(text), delimiter=delimiter or " ", skipinitialspace=True))
    rows = [[c for c in r if c != ""] if (delimiter in (None, " ")) else r for r in rows]
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        raise ControlError("Dosya boş")

    def num(c):
        return _parse_float(c.replace(decimal, ".") if decimal != "." else c)

    header = None
    try:
        [num(c) for c in rows[0]]
    except ValueError:
        header = [c.strip() for c in rows[0]]
        rows = rows[1:]
    if not rows:
        raise ControlError("Dosyada sayısal satır yok")
    width = max(len(r) for r in rows)
    try:
        M = np.array([[num(c) for c in r] + [np.nan] * (width - len(r)) for r in rows], float)
    except ValueError as exc:
        raise ControlError(f"Sayıya çevrilemeyen hücre: {exc}") from None

    def idx(sel):
        if isinstance(sel, (str, int, np.integer)):
            sel = [sel]
        out = []
        for s in sel:
            if isinstance(s, str):
                if header is None or s not in header:
                    raise ControlError(f"'{s}' adlı sütun yok")
                out.append(header.index(s))
            else:
                i = int(s)
                if not 0 <= i < width:
                    raise ControlError(f"Sütun {i} yok (dosyada {width} sütun var)")
                out.append(i)
        return out

    oi, ii = idx(outputs), idx(inputs)
    if not oi:
        raise ControlError("En az bir çıkış sütunu seçilmeli")
    tstart = None
    if time_column is not None:
        t = M[:, idx(time_column)[0]]
        if np.isnan(t).any():
            raise ControlError("Zaman sütununda boş hücre var")
        dt = np.diff(t)
        if len(dt) == 0 or np.any(dt <= 0):
            raise ControlError("Zaman sütunu artan olmalı")
        Ts_t = float(np.median(dt))
        if np.max(np.abs(dt - Ts_t)) > 1e-6 * max(Ts_t, 1e-300) + 1e-12 * np.max(np.abs(t)):
            raise ControlError("Zaman sütunu eşit aralıklı değil (sistem tanımlama eşit aralıklı veri ister)")
        if Ts is not None and abs(Ts - Ts_t) > 1e-6 * Ts:
            raise ControlError(f"Verilen Ts ({Ts:g}) zaman sütunuyla ({Ts_t:g}) uyuşmuyor")
        Ts = Ts_t
        tstart = float(t[0])
    if Ts is None:
        raise ControlError("Örnekleme aralığı (Ts) ya da zaman sütunu verilmeli")
    hn = (lambda i: header[i]) if header else (lambda i: f"col{i}")
    return IDData(M[:, oi], M[:, ii] if ii else None, Ts, tstart=tstart,
                  output_names=[hn(i) for i in oi], input_names=[hn(i) for i in ii],
                  name=name if name is not None else Path(path).stem)


def mat_variables(path) -> dict:
    """`.mat` dosyasındaki sayısal değişkenler: ad → şekil. v7.3 (HDF5) dosyada açık hata."""
    from scipy import io as sio

    try:
        info = sio.whosmat(str(path))
    except NotImplementedError:
        raise ControlError("MATLAB v7.3 (HDF5) dosyası okunamıyor; MATLAB'de save(..., '-v7') ile kaydedin") from None
    except ValueError as exc:
        raise ControlError(f".mat dosyası okunamadı: {exc}") from None
    return {n: tuple(s) for n, s, cls in info if cls in ("double", "single", "int8", "int16", "int32", "int64",
                                                        "uint8", "uint16", "uint32", "uint64", "logical")}


def read_mat(path, *, y: str, u: str | list | None = None, Ts: float | None = None, time: str | None = None,
             name: str | None = None) -> IDData:
    """`.mat` (v5 / v7) dosyasından veri: çıkış / giriş değişken adları (sütunlar kanal), Ts ya da zaman değişkeni.

    MATLAB `iddata` nesnesi doğrudan okunamaz (sınıf nesnesi); `y = data.y; u = data.u; Ts = data.Ts` olarak
    düz diziler kaydedilmelidir."""
    from scipy import io as sio

    try:
        d = sio.loadmat(str(path), squeeze_me=False)
    except NotImplementedError:
        raise ControlError("MATLAB v7.3 (HDF5) dosyası okunamıyor; MATLAB'de save(..., '-v7') ile kaydedin") from None
    except ValueError as exc:
        raise ControlError(f".mat dosyası okunamadı: {exc}") from None

    def get(nm):
        if nm not in d:
            raise ControlError(f"'{nm}' değişkeni dosyada yok")
        v = d[nm]
        if not isinstance(v, np.ndarray) or v.dtype.kind not in "fiub":
            raise ControlError(f"'{nm}' sayısal dizi değil (iddata gibi nesneler için y/u dizilerini kaydedin)")
        v = np.asarray(v, float)
        if v.ndim == 2 and v.shape[0] == 1 and v.shape[1] > 1:
            v = v.T
        return v

    Y = get(y)
    if isinstance(u, (list, tuple)):
        U = np.column_stack([get(n) for n in u])
        unames = list(u)
    elif u is not None:
        U = get(u)
        unames = [u] if U.shape[1] == 1 else [f"{u}{i + 1}" for i in range(U.shape[1])]
    else:
        U, unames = None, []
    if Ts is None and time is None and "Ts" in d:
        Ts = float(np.asarray(d["Ts"]).ravel()[0])
    tstart = None
    if time is not None:
        t = get(time).ravel()
        dt = np.diff(t)
        Ts_t = float(np.median(dt))
        if np.any(dt <= 0) or np.max(np.abs(dt - Ts_t)) > 1e-6 * Ts_t + 1e-12 * np.max(np.abs(t)):
            raise ControlError("Zaman değişkeni eşit aralıklı ve artan olmalı")
        Ts, tstart = Ts_t, float(t[0])
    if Ts is None:
        raise ControlError("Örnekleme aralığı (Ts) ya da zaman değişkeni verilmeli")
    ynames = [y] if Y.shape[1] == 1 else [f"{y}{i + 1}" for i in range(Y.shape[1])]
    return IDData(Y, U, Ts, tstart=tstart, output_names=ynames, input_names=unames,
                  name=name if name is not None else Path(path).stem)
