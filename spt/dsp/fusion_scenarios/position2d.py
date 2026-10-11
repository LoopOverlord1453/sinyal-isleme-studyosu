"""Senaryo 3 — 2B konum: GNSS + jiroskop + hız (odometri), kara aracı / robot.

Girişler
  gnss    : GNSS konumu yerel ENU metre (x doğu, y kuzey)                      ┐ en az biri; ikisi de eşlenirse
  gnss_ll : GNSS enlem / boylam (° | rad). İlk geçerli düzeltme etrafında yerel ┘ metre girişi kullanılır
            teğet düzleme çevrilir (eşdikdörtgen; WGS-84 meridyen ve ilk düşey yarıçaplarıyla):
            x = N(φ0)·cos φ0·(λ − λ0), y = M(φ0)·(φ − φ0) — birkaç km içinde cm düzeyinde doğru.
            Not: float32 kanal enlem/boylamda ~0.5 m çözünürlük verir; mümkünse float64 ya da metre kullanın.
  gyro_z  : sapma (yaw) hızı, z yukarı, saat yönünün tersi + (rad/s | °/s | mdps) — zorunlu
  speed   : hız ölçümü (tekerlek / odometri; m/s | km/h …) — isteğe bağlı; ölçek hatası kestirilir
  acc_fwd : boylamsal ivme (m/s² | g) — isteğe bağlı; hızı sürmek için kullanılır (sapması modellenmez)
  GNSS seyrektir (ör. 100 Hz akışta 5 Hz): aradaki örnekler NaN; kesinti (tünel) de NaN'dır.

Çıktılar: x, y (m), başlık ψ (doğu ekseninden saat yönünün tersine; pusula başlığı = 90° − ψ), hız v (m/s).

Model (CTRV benzeri tek tekerlek; durum [x, y, ψ, v, b_g, s], T = örnek aralığı, ω = gyro_z − b_g)
  ψm = ψ + ½·ω·T;   x⁺ = x + v·T·cos ψm;   y⁺ = y + v·T·sin ψm;   ψ⁺ = ψ + ω·T
  v⁺ = v + a·T (acc_fwd varsa) ya da rastgele yürüyüş;   b_g, s rastgele yürüyüş
  Ölçümler: GNSS z = (x, y) + n;   odometri z = s·v + n   (s: odometri ölçek çarpanı, ≈ 1)
  Q = diag(σ_pos²T, σ_pos²T, (σ_g·T)², q_v, σ_bg²T, (1e-4)²T),  q_v = (σ_a·T)² ya da σ_vrw²·T
Algoritmalar
  dead_reckoning : yalnız integral (taban çizgisi): başlatmadan sonra GNSS kullanılmaz, jiroskop sapması ve
                   odometri ölçeği düzeltilmez.
  ekf            : genişletilmiş Kalman (Jacobian kapalı biçim), ölçümler sıralı skaler güncellemeyle.
  ukf            : kokusuz Kalman, 2n + 1 = 13 sigma noktası (α = 1, β = 2, κ = 0), başlık dairesel ortalamayla.
  pf             : parçacık filtresi (6 boyutlu durum), Liu–West yumuşatmalı sistematik yeniden örnekleme.
                   Odometri burada ölçüm değil hareket modelinin kontrol girişidir (robotikteki MCL gibi):
                   v = (z + n)/s; ağırlıklar yalnız GNSS düzeltmelerinde güncellenir (100 Hz hız olabilirliği
                   her örnekte yeniden örneklemeye ve çökmeye yol açar). Konumun toplamsal yürüyüşü GNSS anına
                   ertelenir (dağılım aynı), ölçeğin 1e-4/√s yürüyüşü ihmal edilir.
Başlatma (hepsinde aynı): GNSS izi `d_init` metreyi aşınca — konum son düzeltme, başlık kirişin yönü + kiriş
boyunca jiroskopla dönülen açının yarısı (sabit dönüş hızında kirişin yönü orta noktadaki başlıktır), hız
odometriden ya da kiriş/süre. İleri hareket varsayılır. Öncesinde x, y son GNSS düzeltmesi; başlık, hız NaN.
Eksik jiroskop (NaN / inf): son geçerli değer tutulur, her eksik örnekte başlığa 0.2 rad/√s rastgele yürüyüş eklenir.
"""
from __future__ import annotations

import math

import numpy as np

from spt.dsp.fusion_scenarios.altitude import smooth_steps
from spt.dsp.fusion_scenarios.base import (
    D2R,
    NAN,
    R2D,
    UNITS_ACC,
    UNITS_ANGLE,
    UNITS_LENGTH,
    UNITS_RATE,
    UNITS_SPEED,
    Estimator,
    Input,
    Output,
    Param,
    Scenario,
    Simulation,
    chol_psd,
    liu_west_resample,
    normalized_weights,
    ut_weights,
    wrap,
    wrap_scalar,
)

WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3
INIT_WINDOW_S = 30.0       # başlatma kirişi için saklanan GNSS geçmişi
INIT_HISTORY = 160         # geçmişte en çok bu kadar düzeltme (MCU'da sabit boyutlu tampon; C ile aynı kural)
INIT_SPACING_S = INIT_WINDOW_S / (INIT_HISTORY - 1)   # saklanan düzeltmeler arası en az süre: her GNSS hızında 30 s
MISS_RATE_DENSITY = 0.2    # rad/√s, jiroskop örneği yokken başlığa eklenen rastgele yürüyüş
SCALE_RW = 1e-4            # 1/√s, odometri ölçeğinin yürüyüşü
_BIG = 1e12

INPUTS = (
    Input("gnss", "GNSS konumu (yerel ENU)", ("x", "y"), UNITS_LENGTH, False,
          ("gps", "gnss", "konum", "pos", "position", "enu", "utm"),
          "Yerel doğu (x) / kuzey (y) metre. Seyrek olabilir (NaN = düzeltme yok)"),
    Input("gnss_ll", "GNSS enlem / boylam", ("lat", "lon"), UNITS_ANGLE, False,
          ("gps", "gnss", "lat", "lon", "enlem", "boylam", "latitude", "longitude", "lng"),
          "İlk düzeltme etrafında yerel teğet düzleme çevrilir (metre girişi yoksa)"),
    Input("gyro_z", "Sapma hızı (jiroskop z)", ("",), UNITS_RATE, True,
          ("gyro z", "gyr z", "jiro z", "gz", "yaw rate", "omega z", "rate z", "sapma hizi", "donus hizi",
           "yaw hizi", "w z"),
          "Dikey eksen etrafında açısal hız; saat yönünün tersi pozitif (z yukarı)"),
    Input("speed", "Hız (odometri)", ("",), UNITS_SPEED, False,
          ("hiz", "speed", "odom", "odometri", "odometry", "velocity", "vel", "tekerlek", "wheel", "v"),
          "Tekerlek / odometri hızı; ölçek hatası kestirilir"),
    Input("acc_fwd", "Boylamsal ivme", ("",), UNITS_ACC, False,
          ("acc x", "accel x", "ivme x", "ax", "ileri ivme", "boylamsal ivme", "acc fwd", "acc long",
           "hizlanma"),
          "Araç ekseninde ileri ivme; hızı sürer (yerçekimi bileşeni çıkarılmış olmalı)"),
)
OUTPUTS = (
    Output("x", "x (doğu)", "m", False, ("x", "dogu", "east", "pos x", "konum x", "gps x")),
    Output("y", "y (kuzey)", "m", False, ("y", "kuzey", "north", "pos y", "konum y", "gps y")),
    Output("heading", "Başlık ψ", "°", True, ("baslik", "heading", "yaw", "psi", "yon", "course", "rota")),
    Output("speed", "Hız", "m/s", False, ("hiz", "speed", "v", "velocity", "vel")),
)
ONE_OF = (("gnss", "gnss_ll"),)
PARAMS = (
    Param("sigma_gnss", "σ GNSS (m)", 2.5, 0.001, 1000.0, 3, "Eksen başına GNSS konum gürültüsü"),
    Param("sigma_gyro", "σ jiroskop (°/s)", 0.2, 1e-5, 100.0, 4, "Örnek başına jiroskop beyaz gürültüsü (RMS)"),
    Param("sigma_gbias", "σ jiro sapması (°/s/√s)", 0.005, 0.0, 10.0, 5, "Jiroskop sapmasının rastgele yürüyüşü"),
    Param("gbias0", "Başlangıç jiro sapma σ (°/s)", 0.5, 1e-5, 100.0, 4, "Jiroskop sapmasının başlangıç belirsizliği"),
    Param("sigma_speed", "σ hız ölçümü (m/s)", 0.1, 1e-5, 100.0, 4, "Odometri hız ölçümünün gürültüsü"),
    Param("sigma_vrw", "Hız değişimi (m/s/√s)", 1.0, 1e-5, 100.0, 4,
          "İvme girişi yokken hızın rastgele yürüyüşü (aracın hızlanma kabiliyeti)"),
    Param("sigma_accf", "σ boylamsal ivme (m/s²)", 0.2, 1e-5, 100.0, 4, "İvme girişi varken örnek başına gürültüsü"),
    Param("sigma_pos", "Konum süreç gürültüsü (m/√s)", 0.1, 0.0, 100.0, 4, "Kayma / yanal hareket payı"),
    Param("scale0", "Başlangıç odometri ölçek σ (%)", 5.0, 0.0, 50.0, 3, "Odometri ölçek çarpanının belirsizliği"),
    Param("d_init", "Başlatma mesafesi (m)", 20.0, 0.5, 10_000.0, 2,
          "Başlığı GNSS izinden kurmak için gereken yol; büyük → daha doğru başlangıç başlığı, geç başlama"),
)
PF_PARAMS = (
    Param("particles", "Parçacık sayısı", 500, 50, 20_000, 0, "Daha çok parçacık: daha doğru, daha yavaş"),
    Param("seed", "Tohum", 1, 0, 1_000_000, 0, "Rastgele sayı üretecinin tohumu (tekrarlanabilirlik)"),
)


def ll_to_local(lat, lon, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    """Enlem / boylam (rad) → (lat0, lon0) etrafında yerel doğu / kuzey metre (eşdikdörtgen, WGS-84 yarıçapları)."""
    s2 = math.sin(lat0) ** 2
    n_rad = WGS84_A / math.sqrt(1.0 - WGS84_E2 * s2)
    m_rad = WGS84_A * (1.0 - WGS84_E2) / (1.0 - WGS84_E2 * s2) ** 1.5
    return n_rad * math.cos(lat0) * wrap(np.asarray(lon) - lon0), m_rad * (np.asarray(lat) - lat0)


def local_to_ll(x, y, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    """`ll_to_local`'ın tersi (rad)."""
    s2 = math.sin(lat0) ** 2
    n_rad = WGS84_A / math.sqrt(1.0 - WGS84_E2 * s2)
    m_rad = WGS84_A * (1.0 - WGS84_E2) / (1.0 - WGS84_E2 * s2) ** 1.5
    return lat0 + np.asarray(y) / m_rad, lon0 + np.asarray(x) / (n_rad * math.cos(lat0))


class _Pos2DBase(Estimator):
    """Ortak: giriş hazırlama, GNSS (enlem/boylam) dönüşümü, başlatma ve örnek döngüsü.

    Alt sınıflar `_start(state, p0_diag)` ve `_step(gz, gmiss, fix, zx, zy, spd, acc, T) → (x, y, ψ, v)` yazar.
    """
    scenario_params = PARAMS
    inputs = INPUTS
    outputs = OUTPUTS
    one_of = ONE_OF

    def _setup(self) -> None:
        self.inited = False
        self.k = 0
        self.theta = 0.0                 # başlatma öncesi ham jiroskop integrali
        self.fixes: list[tuple[int, float, float, float]] = []
        self.last_fix: tuple[float, float] | None = None
        self.origin: tuple[float, float] | None = None
        self.w_last: float | None = None
        self.n_gnss = 0
        self.init_k: int | None = None
        self._init_state()

    def _init_state(self) -> None:
        pass

    # ---------------------------------------------------------------- yardımcılar
    def _noise(self, T: float) -> dict:
        sv = self.sv
        return {
            "pos": sv["sigma_pos"] ** 2 * T,
            "psi": (sv["sigma_gyro"] * D2R * T) ** 2,
            "psi_miss": (sv["sigma_gyro"] * D2R * T) ** 2 + MISS_RATE_DENSITY ** 2 * T,
            "v_rw": sv["sigma_vrw"] ** 2 * T,
            "v_acc": (sv["sigma_accf"] * T) ** 2,
            "bg": (sv["sigma_gbias"] * D2R) ** 2 * T,
            "s": SCALE_RW ** 2 * T,
        }

    def _gnss_lists(self, x, n):
        if "gnss" in x:
            g = x["gnss"]
            return g[:, 0].tolist(), g[:, 1].tolist()
        ll = x["gnss_ll"]
        ok = np.isfinite(ll).all(axis=1)
        if self.origin is None and ok.any():
            i = int(np.argmax(ok))
            self.origin = (float(ll[i, 0]), float(ll[i, 1]))
        if self.origin is None:
            return [NAN] * n, [NAN] * n
        with np.errstate(invalid="ignore"):
            gx, gy = ll_to_local(ll[:, 0], ll[:, 1], *self.origin)
        gx[~ok] = np.nan
        gy[~ok] = np.nan
        return gx.tolist(), gy.tolist()

    def _pre_init(self, kabs, gz, fix, zx, zy, spd, T):
        """Başlatma öncesi: jiroskop açısını biriktir, GNSS geçmişini tut; kiriş yeterliyse başlangıç durumu."""
        self.theta += gz * T
        if not fix:
            return None
        self.last_fix = (zx, zy)
        fixes = self.fixes
        horizon = INIT_WINDOW_S / T
        while fixes and kabs - fixes[0][0] > horizon:
            fixes.pop(0)
        d_init = self.sv["d_init"]
        for k0, x0, y0, th0 in reversed(fixes):  # en yeniden eskiye: d ≥ d_init olan ilk saklı düzeltme
            d = math.hypot(zx - x0, zy - y0)
            if d >= d_init:
                span = (kabs - k0) * T
                psi = wrap_scalar(math.atan2(zy - y0, zx - x0) + 0.5 * (self.theta - th0))
                v = spd if spd is not None else d / span
                return (zx, zy, psi, v), self._p0(d, span, v, spd is not None)
        # Seyreltilmiş geçmiş: düzeltme, son saklanandan en az INIT_SPACING_S sonraysa saklanır (≤ 5.3 Hz GNSS'te hepsi)
        if not fixes or (kabs - fixes[-1][0]) * T >= INIT_SPACING_S:
            fixes.append((kabs, zx, zy, self.theta))
            if len(fixes) > INIT_HISTORY:
                fixes.pop(0)
        return None

    def _p0(self, d, span, v, has_speed):
        sv = self.sv
        sg = sv["sigma_gnss"]
        s_psi = math.sqrt(2.0) * sg / d + 3.0 * D2R
        if has_speed:
            s_v = sv["sigma_speed"] + 0.01 * sv["scale0"] * abs(v)
        else:
            s_v = math.sqrt(2.0) * sg / span + 0.5
        return (sg ** 2, sg ** 2, s_psi ** 2, s_v ** 2, (sv["gbias0"] * D2R) ** 2, (0.01 * sv["scale0"]) ** 2)

    # ---------------------------------------------------------------- örnek döngüsü
    def _run(self, x, n, dt):
        gl = x["gyro_z"].tolist()
        xl, yl = self._gnss_lists(x, n)
        sl = x["speed"].tolist() if "speed" in x else None
        al = x["acc_fwd"].tolist() if "acc_fwd" in x else None
        out = np.empty((n, 4))
        for k in range(n):
            gz = gl[k]
            if -_BIG < gz < _BIG:
                self.w_last, gmiss = gz, False
            else:
                gz, gmiss = (self.w_last if self.w_last is not None else 0.0), True
            zx, zy = xl[k], yl[k]
            fix = -_BIG < zx < _BIG and -_BIG < zy < _BIG
            spd = sl[k] if sl is not None and -_BIG < sl[k] < _BIG else None
            acc = al[k] if al is not None and -_BIG < al[k] < _BIG else None
            if fix:
                self.n_gnss += 1
            if not self.inited:
                start = self._pre_init(self.k + k, gz, fix, zx, zy, spd, dt)
                if start is None:
                    lx, ly = self.last_fix if self.last_fix is not None else (NAN, NAN)
                    out[k] = lx, ly, NAN, NAN
                    continue
                self.inited, self.init_k = True, self.k + k
                self._start(*start)
                st = start[0]
                out[k] = st[0], st[1], st[2], st[3]
                continue
            out[k] = self._step(gz, gmiss, fix, zx, zy, spd, acc, dt)
        self.k += n
        return out

    def _start(self, state, p0) -> None:  # pragma: no cover - alt sınıflar
        raise NotImplementedError

    def _step(self, gz, gmiss, fix, zx, zy, spd, acc, T):  # pragma: no cover - alt sınıflar
        raise NotImplementedError

    def _diag_common(self) -> dict[str, str]:
        d = {"GNSS düzeltmesi": str(self.n_gnss)}
        if self.init_k is not None:
            d["Başlatma örneği"] = str(self.init_k)
        return d


# ============================================================================ ölü hesap (taban çizgisi)
class Pos2DDeadReckoning(_Pos2DBase):
    key, name = "dead_reckoning", "Ölü hesap (yalnız integral)"

    def _init_state(self) -> None:
        self.s = (0.0, 0.0, 0.0, 0.0)

    def _start(self, state, p0) -> None:
        self.s = state

    def _step(self, gz, gmiss, fix, zx, zy, spd, acc, T):
        px, py, psi, v = self.s
        if spd is not None:
            v = spd
        elif acc is not None:
            v += acc * T
        pm = psi + 0.5 * gz * T
        px += v * T * math.cos(pm)
        py += v * T * math.sin(pm)
        psi = wrap_scalar(psi + gz * T)
        self.s = (px, py, psi, v)
        return self.s

    def diagnostics(self):
        return self._diag_common()


# ============================================================================ EKF
class Pos2DEKF(_Pos2DBase):
    key, name = "ekf", "EKF"

    def _init_state(self) -> None:
        self.x: np.ndarray | None = None
        self.P: np.ndarray | None = None
        self._F = np.eye(6)
        self._q: tuple[float, dict] | None = None

    def _start(self, state, p0) -> None:
        self.x = np.array([*state, 0.0, 1.0])
        self.P = np.diag(p0)

    def _qdiag(self, T, gmiss, has_acc):
        if self._q is None or self._q[0] != T:
            q = self._noise(T)
            self._q = (T, {(gm, ha): np.diag([q["pos"], q["pos"], q["psi_miss" if gm else "psi"],
                                              q["v_acc" if ha else "v_rw"], q["bg"], q["s"]])
                           for gm in (False, True) for ha in (False, True)})
        return self._q[1][(gmiss, has_acc)]

    def _step(self, gz, gmiss, fix, zx, zy, spd, acc, T):
        x, P, F = self.x, self.P, self._F
        px, py, psi, v, bg, sc = x.tolist()
        w = gz - bg
        pm = psi + 0.5 * w * T
        c, s = math.cos(pm), math.sin(pm)
        vT = v * T
        F[0, 2], F[0, 3], F[0, 4] = -vT * s, T * c, 0.5 * vT * T * s
        F[1, 2], F[1, 3], F[1, 4] = vT * c, T * s, -0.5 * vT * T * c
        F[2, 4] = -T
        x[0], x[1], x[2] = px + vT * c, py + vT * s, psi + w * T
        if acc is not None:
            x[3] = v + acc * T
        P = F @ P @ F.T + self._qdiag(T, gmiss, acc is not None)
        sv = self.sv
        if fix:
            r = sv["sigma_gnss"] ** 2
            for i, z in ((0, zx), (1, zy)):
                ph = P[:, i].copy()
                kg = ph / (ph[i] + r)
                x += kg * (z - x[i])
                P -= kg[:, None] * ph
        if spd is not None:
            sc, v = x[5], x[3]
            ph = P[:, 3] * sc + P[:, 5] * v
            kg = ph / (sc * ph[3] + v * ph[5] + sv["sigma_speed"] ** 2)
            x += kg * (spd - sc * v)
            P -= kg[:, None] * ph
        x[2] = wrap_scalar(x[2])
        self.P = 0.5 * (P + P.T)
        return x[0], x[1], x[2], x[3]

    def diagnostics(self):
        d = {}
        if self.x is not None:
            d["Jiroskop sapması (°/s)"] = f"{self.x[4] * R2D:+.4f}"
            d["Odometri ölçeği"] = f"{self.x[5]:.4f}"
            d["σ konum (m)"] = f"{math.sqrt(max(self.P[0, 0] + self.P[1, 1], 0.0)):.2f}"
            d["σ başlık (°)"] = f"{math.sqrt(max(self.P[2, 2], 0.0)) * R2D:.2f}"
        d.update(self._diag_common())
        return d


# ============================================================================ UKF
def _wrap_fast(a: np.ndarray) -> np.ndarray:
    """Filtre içi (NaN'sız) hızlı açı sarma."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class Pos2DUKF(Pos2DEKF):
    key, name = "ukf", "UKF (kokusuz)"
    _WM, _WC = ut_weights(6)

    def _init_state(self) -> None:
        super()._init_state()
        self._X = np.empty((13, 6))

    def _sigma(self, x, P):
        """2n + 1 sigma noktası: x, x ± √(n·P) sütunları → (13, 6)."""
        Lt = chol_psd(6.0 * P).T
        X = self._X
        X[:] = x
        X[1:7] += Lt
        X[7:] -= Lt
        return X

    def _ut_update(self, x, P, Z, z, R):
        """Sigma noktalarından (Z: (13, m)) kokusuz ölçüm güncellemesi."""
        wm, wc = self._WM, self._WC
        D = self._X - x
        zm = wm @ Z
        dZ = Z - zm
        if Z.ndim == 1:                                       # skaler ölçüm: matris tersi yok
            pzz = float(wc @ (dZ * dZ)) + R
            K = (wc * dZ) @ D / pzz
            return x + K * (z - zm), P - (K[:, None] * K) * pzz
        pzz = (dZ.T * wc) @ dZ + R
        K = np.linalg.solve(pzz, ((D.T * wc) @ dZ).T).T
        return x + K @ (z - zm), P - K @ pzz @ K.T

    def _step(self, gz, gmiss, fix, zx, zy, spd, acc, T):
        wm, wc = self._WM, self._WC
        X = self._sigma(self.x, self.P)
        w = gz - X[:, 4]
        pm = X[:, 2] + 0.5 * w * T
        vT = X[:, 3] * T
        X[:, 0] += vT * np.cos(pm)
        X[:, 1] += vT * np.sin(pm)
        X[:, 2] += w * T
        if acc is not None:
            X[:, 3] += acc * T
        x = wm @ X
        ref = X[0, 2]
        x[2] = ref + wm @ _wrap_fast(X[:, 2] - ref)          # dairesel ortalama (merkez noktaya göre)
        D = X - x
        D[:, 2] = _wrap_fast(D[:, 2])
        P = (D.T * wc) @ D + self._qdiag(T, gmiss, acc is not None)
        sv = self.sv
        if fix:
            X = self._sigma(x, P)
            x, P = self._ut_update(x, P, X[:, :2], np.array((zx, zy)), np.eye(2) * sv["sigma_gnss"] ** 2)
        if spd is not None:
            X = self._sigma(x, P)
            x, P = self._ut_update(x, P, X[:, 5] * X[:, 3], spd, sv["sigma_speed"] ** 2)
        x[2] = wrap_scalar(x[2])
        self.x, self.P = x, 0.5 * (P + P.T)
        return x[0], x[1], x[2], x[3]


# ============================================================================ parçacık filtresi
class Pos2DPF(_Pos2DBase):
    key, name = "pf", "Parçacık filtresi"
    params = PF_PARAMS

    def _init_state(self) -> None:
        self.rng = np.random.default_rng(int(self.values["seed"]))
        self.X: np.ndarray | None = None      # (6, N): x, y, ψ, v, b_g, s
        self.w: np.ndarray | None = None
        self.ess = 0.0
        self.resamples = 0
        self._sd: tuple[float, dict] | None = None

    def _start(self, state, p0) -> None:
        N = int(self.values["particles"])
        mean = np.array([*state, 0.0, 1.0])
        self.X = mean[:, None] + np.sqrt(np.array(p0))[:, None] * self.rng.standard_normal((6, N))
        self.w = np.full(N, 1.0 / N)
        self.ess = float(N)
        self._since_fix = 0          # konum yürüyüşü birikmiş örnek sayısı
        self._heading = state[2]

    def _std(self, T):
        if self._sd is None or self._sd[0] != T:
            q = self._noise(T)
            self._sd = (T, {k: math.sqrt(v) for k, v in q.items()})
        return self._sd[1]

    def _step(self, gz, gmiss, fix, zx, zy, spd, acc, T):
        X, w, rng = self.X, self.w, self.rng
        sd = self._std(T)
        sv = self.sv
        N = X.shape[1]
        e = rng.standard_normal((3, N))
        om = gz - X[4] + (sd["psi_miss"] if gmiss else sd["psi"]) / T * e[0]
        # Odometri hareket modeline kontrol girişi olarak girer (robotikteki MCL gibi): v = (z + n) / s.
        # Ölçüm olarak 100 Hz'de ağırlıklandırmak her örnekte yeniden örneklemeye ve çökmeye yol açar.
        if spd is not None:
            X[3] = (spd + sv["sigma_speed"] * e[1]) / X[5]
        elif acc is not None:
            X[3] += acc * T + sd["v_acc"] * e[1]
        else:
            X[3] += sd["v_rw"] * e[1]
        pm = X[2] + 0.5 * om * T
        vT = X[3] * T
        X[0] += vT * np.cos(pm)
        X[1] += vT * np.sin(pm)
        X[2] += om * T
        X[4] += sd["bg"] * e[2]
        # Konumun toplamsal yürüyüşü (σ_pos) dinamiğe geri beslenmez: GNSS anına ertelenip toplu eklenir
        # (ölçüm anlarında dağılım birebir aynı, örnek başına 2·N rastgele sayı tasarrufu). Ölçeğin çok küçük
        # yürüyüşü (1e-4/√s) ihmal edilir; çeşitliliği yeniden örneklemedeki Liu–West çekirdeği korur.
        self._since_fix += 1
        if fix:
            if sd["pos"] > 0:
                X[0:2] += sd["pos"] * math.sqrt(self._since_fix) * rng.standard_normal((2, N))
            self._since_fix = 0
            r2 = ((zx - X[0]) ** 2 + (zy - X[1]) ** 2) / sv["sigma_gnss"] ** 2
            w = normalized_weights(np.log(np.maximum(w, 1e-300)) - 0.5 * r2)
            self.ess = 1.0 / float(w @ w)
            if self.ess < 0.5 * N:
                X = liu_west_resample(rng, X, w, angle_rows=(2,))
                w = np.full(N, 1.0 / N)
                self.resamples += 1
        self.X, self.w = X, w
        ref = self._heading                    # dairesel ortalama, önceki başlığa göre (trigonometrisiz)
        heading = wrap_scalar(ref + float(w @ ((X[2] - ref + math.pi) % (2.0 * math.pi) - math.pi)))
        self._heading = heading
        return float(w @ X[0]), float(w @ X[1]), heading, float(w @ X[3])

    def diagnostics(self):
        d = {}
        if self.X is not None:
            m = self.X @ self.w
            d["Jiroskop sapması (°/s)"] = f"{m[4] * R2D:+.4f}"
            d["Odometri ölçeği"] = f"{m[5]:.4f}"
            d["ESS"] = f"{self.ess:.0f}"
            d["Yeniden örnekleme"] = str(self.resamples)
        d.update(self._diag_common())
        return d


ALGORITHMS: dict[str, type[Estimator]] = {cls.key: cls for cls in (
    Pos2DDeadReckoning, Pos2DEKF, Pos2DUKF, Pos2DPF)}


def raw(x: dict, n: int, sv: dict, state: dict) -> np.ndarray:
    """Füzyonsuz: x, y doğrudan GNSS'ten (yalnız düzeltme örneklerinde, arası NaN; enlem/boylam kestiricilerle
    aynı biçimde ilk geçerli düzeltme etrafında açılır — referans `state["_origin"]` = (enlem, boylam) rad);
    başlık NaN (tek sensörden doğrudan ölçülmez); hız = odometri ölçümü (eşlenmişse, ölçek düzeltmesiz)."""
    out = np.full((n, 4), np.nan)
    if "gnss" in x:
        g = x["gnss"]
        ok = np.isfinite(g).all(axis=1)
        out[ok, :2] = g[ok]
    elif "gnss_ll" in x:
        ll = x["gnss_ll"]
        ok = np.isfinite(ll).all(axis=1)
        origin = state.get("_origin")
        if origin is None and ok.any():
            i = int(np.argmax(ok))
            origin = (float(ll[i, 0]), float(ll[i, 1]))
            state["_origin"] = origin
        if origin is not None and ok.any():
            gx, gy = ll_to_local(ll[ok, 0], ll[ok, 1], *origin)
            out[ok, 0], out[ok, 1] = gx, gy
    if "speed" in x:
        out[:, 3] = x["speed"]
    return out


# ============================================================================ simülatör
def simulate(duration: float, fs: float, seed: int, gnss_rate: float = 5.0, sigma_gnss: float = 2.5,
             outage: tuple[float, float] | None = (60.0, 10.0), gyro_bias_dps: float = 0.4,
             sigma_gyro_dps: float = 0.15, odo_scale: float = 1.03, sigma_speed: float = 0.05,
             speed: bool = True, acc: bool = False, latlon: bool = False, lat0_deg: float = 41.0,
             lon0_deg: float = 29.0, psi0_deg: float = 30.0, oversample: int = 10) -> Simulation:
    """Araç benzeri yörünge: dönüşler (sin² profilli sapma hızı) ve hız değişimleri — başlık ve hız analitik,
    konum ince ızgarada (oversample×) yamuk integrali (hata ≪ 1 mm).

    Sensörler: jiroskop = ω + sabit sapma + gürültü; odometri = ölçek·v + gürültü; GNSS `gnss_rate` Hz, σ eksen
    başına, `outage` (başlangıç s, süre s) boyunca yok. acc=True: boylamsal ivme (v̇ + 0.05 m/s² gürültü);
    latlon=True: GNSS enlem / boylam (rad) olarak (gnss_ll), aksi halde yerel metre (gnss).
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration * fs))
    t = np.arange(n) / fs
    tf = np.arange(n * oversample) / (fs * oversample)
    turns = [(12, 6, 90), (30, 5, -60), (45, 8, 120), (63, 6, -90), (80, 4, 45), (95, 10, -180), (112, 5, 60),
             (130, 6, -90), (150, 8, 135), (170, 5, -45)]
    turns = [(a, b, c * D2R) for a, b, c in turns if a < duration]
    speeds = [(5, 6, 4.0), (25, 4, -6.0), (40, 5, 5.0), (58, 6, -3.0), (75, 8, 6.0), (92, 5, -7.0), (108, 6, 4.0),
              (125, 5, -4.0), (145, 6, 3.0), (165, 4, -2.0)]
    speeds = [s for s in speeds if s[0] < duration]
    psi0 = psi0_deg * D2R

    def profile(tt):
        dpsi, omega, _ = smooth_steps(tt, turns)
        dv, vdot, _ = smooth_steps(tt, speeds)
        return psi0 + dpsi, omega, 10.0 + dv, vdot

    psi_f, _, v_f, _ = profile(tf)
    vx, vy = v_f * np.cos(psi_f), v_f * np.sin(psi_f)
    dtf = 1.0 / (fs * oversample)
    px = np.concatenate(([0.0], np.cumsum(0.5 * (vx[1:] + vx[:-1]) * dtf)))[::oversample]
    py = np.concatenate(([0.0], np.cumsum(0.5 * (vy[1:] + vy[:-1]) * dtf)))[::oversample]
    psi, omega, v, vdot = profile(t)
    bias = gyro_bias_dps * D2R
    inputs = {"gyro_z": omega + bias + sigma_gyro_dps * D2R * rng.standard_normal(n)}
    if speed:
        inputs["speed"] = odo_scale * v + sigma_speed * rng.standard_normal(n)
    if acc:
        inputs["acc_fwd"] = vdot + 0.05 * rng.standard_normal(n)
    step = max(1, int(round(fs / gnss_rate)))
    on = np.zeros(n, dtype=bool)
    on[::step] = True
    if outage is not None:
        on &= ~((t >= outage[0]) & (t < outage[0] + outage[1]))
    gx = np.where(on, px + sigma_gnss * rng.standard_normal(n), np.nan)
    gy = np.where(on, py + sigma_gnss * rng.standard_normal(n), np.nan)
    if latlon:
        lat, lon = local_to_ll(gx, gy, lat0_deg * D2R, lon0_deg * D2R)
        inputs["gnss_ll"] = np.column_stack((lat, lon))
    else:
        inputs["gnss"] = np.column_stack((gx, gy))
    truth = np.column_stack((px, py, wrap(psi), v))
    notes = (f"Araç yörüngesi {duration:g} s @ {fs:g} Hz; GNSS {fs / step:g} Hz, σ = {sigma_gnss:g} m"
             + (f", kesinti {outage[0]:g}–{outage[0] + outage[1]:g} s" if outage else "")
             + f"; jiroskop sapması {gyro_bias_dps:+g} °/s, σ = {sigma_gyro_dps:g} °/s"
             + (f"; odometri ölçeği {odo_scale:g}, σ = {sigma_speed:g} m/s" if speed else "; odometri yok")
             + (f"; GNSS enlem/boylam (başlangıç {lat0_deg:g}°, {lon0_deg:g}°)" if latlon else "") + ".")
    values = {"sigma_gnss": sigma_gnss, "sigma_gyro": sigma_gyro_dps, "sigma_speed": sigma_speed}
    return Simulation(fs, inputs, truth, notes, values)


SCENARIO = Scenario(
    key="position2d",
    name="2B konum: GNSS + jiroskop + hız (odometri)",
    description=("Seyrek ve gürültülü GNSS konumunu jiroskop (sapma hızı) ve tekerlek hızıyla birleştirerek "
                 "x, y, başlık ve hız kestirir; jiroskop sapmasını ve odometri ölçek hatasını da öğrenir. "
                 "GNSS kesintisinde (tünel) ölü hesapla devam eder."),
    inputs=INPUTS,
    outputs=OUTPUTS,
    params=PARAMS,
    algorithms=ALGORITHMS,
    simulator=simulate,
    one_of=ONE_OF,
    default_fs=100.0,
    raw_fn=raw,
)
