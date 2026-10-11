"""Sensör füzyonu: IMU (jiroskop + ivmeölçer) ve MARG (+ manyetometre) ile yönelim kestirimi.

Kurallar (tüm algoritmalar ve test verisi aynı tanımı kullanır):
  - Gövde ve dünya çerçevesi sağ el kuralı, z yukarı. Durağan ivmeölçer +g·ẑ_gövde ölçer (özgül kuvvet).
  - q = [w, x, y, z] gövde → dünya döndürmesi: v_dünya = R(q)·v_gövde. Kinematik: q̇ = ½·q ⊗ [0, ω_gövde].
  - Euler açıları ZYX (aerospace): R = Rz(yaw)·Ry(pitch)·Rx(roll). Yaw, manyetik kuzeyden (dünya +x) ölçülür.
  - Manyetik alan dünya çerçevesinde [B·cos I, 0, −B·sin I] (I: eğim açısı); algoritmalar yalnızca yönünü kullanır.

Algoritmalar:
  - Mahony: SO(3) üzerinde tamamlayıcı filtre (PI düzeltmeli), x-io referans uygulamasıyla aynı denklemler.
  - Madgwick: gradyan iniş, makaledeki amaç fonksiyonu ve Jacobian'larla (IMU ve MARG).
  - Kalman: eksen başına klasik açı + jiroskop sapması Kalman'ı (gömülüde yaygın "TKJ" yapısı); Euler hızları
    tam kinematikle hesaplanır, yaw manyetometreden eğim düzeltmeli başlıkla.
  - EKF: çarpımsal (hata durumlu) genişletilmiş Kalman — MEKF; durum: yönelim hatası δθ (3) + jiroskop sapması (3).
  - UKF: aynı hata durumunda kokusuz (unscented) Kalman (USQUE yaklaşımı), 2n+1 sigma noktası.
  - Parçacık filtresi: kuaterniyon + sapma parçacıkları, sistematik yeniden örnekleme ve pürüzlendirme.
EKF / UKF / parçacık filtresinde manyetometre yalnızca başlığı (yaw) düzeltir; böylece manyetik bozulma roll /
pitch'e sızmaz. Başlık hatası dünya z ekseni etrafındaki dönme olarak ölçülür (EKF / UKF: h = R(q)[2, :] izdüşümü;
parçacık: dünya z etrafındaki "twist" açısı); ZYX yaw'ın tan(pitch) ile büyüyen eğim duyarlılığı kullanılmaz, bu
yüzden pitch ±90°'ye yaklaşsa da başlık düzeltilir. İvmeölçer ölçüm gürültüsü, |a| yerçekiminden saptıkça (doğrusal
ivme) otomatik büyütülür.

Geçersiz örnekler: sonlu olmayan jiroskop örneği kayıp gibi işlenir (çıktı NaN, algoritma sonraki geçerli örnekte
ivmeölçer / manyetometreden yeniden başlar); sonlu olmayan ya da sıfır ivme / manyetometre örneği o adımda yalnızca
o ölçüm güncellemesini atlar.

Ayar notu: ivmeölçer ve başlık gürültüleri örnek başınadır. Doğrusal ivme beyaz değil (zamanla ilişkili) olduğundan
örnekleme hızı çok artırılınca EKF / UKF / Kalman ivmeölçere fazla güvenir; yüksek fs'de σ ivme / σ ölçüm büyütülmeli.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

GRAVITY = 9.80665


# ============================================================================ kuaterniyon araçları
def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton çarpımı; son eksen [w, x, y, z], diğer eksenler yayınlanır (broadcast)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack((aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw), axis=-1)


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.asarray(q, dtype=np.float64) * np.array([1.0, -1.0, -1.0, -1.0])


def quat_exp(rotvec: np.ndarray) -> np.ndarray:
    """Dönme vektörü (rad, eksen·açı) → birim kuaterniyon. sin(θ/2)/θ = ½·sinc(θ/2π): sıfırda da tanımlı."""
    v = np.asarray(rotvec, dtype=np.float64)
    angle = np.sqrt(np.einsum("...i,...i->...", v, v))[..., None]
    return np.concatenate((np.cos(0.5 * angle), v * (0.5 * np.sinc(angle / (2 * np.pi)))), axis=-1)


def quat_log(q: np.ndarray) -> np.ndarray:
    """Birim kuaterniyon → dönme vektörü (kısa yol)."""
    q = np.asarray(q, dtype=np.float64)
    q = np.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    s = np.sqrt(np.einsum("...i,...i->...", v, v))[..., None]
    angle = 2 * np.arctan2(s, q[..., :1])
    k = np.where(s > 1e-12, angle / np.maximum(s, 1e-300), 2.0)
    return v * k


def normalize(q: np.ndarray) -> np.ndarray:
    return q / np.sqrt(np.einsum("...i,...i->...", q, q))[..., None]


def quat_from_euler(roll, pitch, yaw) -> np.ndarray:
    cr, sr = np.cos(np.asarray(roll) / 2), np.sin(np.asarray(roll) / 2)
    cp, sp = np.cos(np.asarray(pitch) / 2), np.sin(np.asarray(pitch) / 2)
    cy, sy = np.cos(np.asarray(yaw) / 2), np.sin(np.asarray(yaw) / 2)
    return np.stack((cr * cp * cy + sr * sp * sy,
                     sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy,
                     cr * cp * sy - sr * sp * cy), axis=-1)


def euler_from_quat(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(roll, pitch, yaw) radyan; ZYX."""
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def rotmat(q: np.ndarray) -> np.ndarray:
    """R(q): gövde → dünya. Son iki eksen 3×3."""
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack((
        np.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)), axis=-1),
        np.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)), axis=-1),
        np.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), axis=-1),
    ), axis=-2)


def gravity_body(q: np.ndarray) -> np.ndarray:
    """Tahmini yerçekimi yönü gövdede: R(q)ᵀ·ẑ (R'nin 3. satırı)."""
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), axis=-1)


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def wrap(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def accel_roll_pitch(acc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Durağan varsayımıyla ivmeölçerden roll ve pitch (rad)."""
    acc = np.asarray(acc, dtype=np.float64)
    ax, ay, az = acc[..., 0], acc[..., 1], acc[..., 2]
    return np.arctan2(ay, az), np.arctan2(-ax, np.hypot(ay, az))


def tilt_compensated_heading(mag: np.ndarray, roll, pitch) -> np.ndarray:
    """Eğim düzeltmeli manyetik başlık (yaw, rad)."""
    mag = np.asarray(mag, dtype=np.float64)
    mx, my, mz = mag[..., 0], mag[..., 1], mag[..., 2]
    cr, sr, cp, sp = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch)
    return np.arctan2(sr * mz - cr * my, cp * mx + sp * sr * my + sp * cr * mz)


def initial_quaternion(acc: np.ndarray, mag: np.ndarray | None = None) -> np.ndarray:
    roll, pitch = accel_roll_pitch(acc)
    yaw = tilt_compensated_heading(mag, roll, pitch) if mag is not None else 0.0
    return quat_from_euler(roll, pitch, yaw)


# ============================================================================ algoritmalar
@dataclass(frozen=True)
class Param:
    key: str
    label: str
    default: float
    lo: float
    hi: float
    decimals: int
    tip: str = ""


class Estimator:
    key = ""
    name = ""
    params: tuple[Param, ...] = ()
    bias_is_euler = False
    estimates_bias = True

    def __init__(self, **values):
        for p in self.params:
            setattr(self, p.key, float(values.get(p.key, p.default)))
        self.q: np.ndarray | None = None
        self.bias = np.zeros(3)
        self.seconds = 0.0   # toplam hesap süresi (örnek başına maliyet için)
        self.samples = 0

    def reset(self, q0: np.ndarray) -> None:
        self.q = normalize(np.asarray(q0, dtype=np.float64))
        self.bias = np.zeros(3)

    def run(self, gyro: np.ndarray, acc: np.ndarray, mag: np.ndarray | None, dt: float) -> np.ndarray:
        """Blok işler (durum bloklar arasında taşınır); örnek başına kuaterniyon (N, 4) döndürür.

        Sonlu olmayan jiroskop örnekleri kayıp gibidir: çıktı NaN, durum unutulur ve algoritma sonraki ilk geçerli
        ivme (ve varsa manyetometre) örneğinden yeniden başlar. Blok sınırları sonucu değiştirmez."""
        gyro = np.asarray(gyro, dtype=np.float64)
        acc = np.asarray(acc, dtype=np.float64)
        mag = None if mag is None else np.asarray(mag, dtype=np.float64)
        out = np.full((len(gyro), 4), np.nan)
        start = time.perf_counter()
        for a, b, ok in _finite_runs(np.isfinite(gyro).all(axis=1)):
            if not ok:
                self.q = None
                continue
            if self.q is None:  # ilk geçerli ivme örneğinden başlat (öncesi NaN)
                valid = np.isfinite(acc[a:b]).all(axis=1) & (np.abs(acc[a:b]).sum(axis=1) > 0)
                if not valid.any():
                    continue
                a += int(np.argmax(valid))
                m0 = mag[a] if mag is not None and _valid(mag[a]) else None
                self.reset(initial_quaternion(acc[a], m0))
            out[a:b] = self._run(gyro[a:b], acc[a:b], None if mag is None else mag[a:b], dt)
        self.seconds += time.perf_counter() - start
        self.samples += len(out)
        return out

    @property
    def cost_us(self) -> float:
        return self.seconds / self.samples * 1e6 if self.samples else 0.0

    def _run(self, gyro, acc, mag, dt) -> np.ndarray:  # pragma: no cover - alt sınıflar
        raise NotImplementedError


def _unit(v):
    """Birim vektör; sıfır ya da sonlu olmayan vektörde None (o ölçüm atlanır)."""
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return (v[0] / n, v[1] / n, v[2] / n) if 0 < n < math.inf else None


def _valid(v) -> bool:
    """Ölçüm kullanılabilir mi: tüm bileşenler sonlu ve norm sıfırdan büyük."""
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return 0 < n < math.inf


def _finite_runs(ok: np.ndarray) -> list[tuple[int, int, bool]]:
    if len(ok) == 0:
        return []
    edges = np.flatnonzero(np.diff(ok.astype(np.int8))) + 1
    bounds = np.r_[0, edges, len(ok)]
    return [(int(a), int(b), bool(ok[a])) for a, b in zip(bounds[:-1], bounds[1:])]


class Mahony(Estimator):
    key, name = "mahony", "Mahony"
    params = (Param("kp", "Kp", 1.0, 0.0, 50.0, 3, "Oransal kazanç: ivmeölçer / manyetometre düzeltme hızı (1/s)"),
              Param("ki", "Ki", 0.05, 0.0, 10.0, 4, "İntegral kazancı: jiroskop sapmasını öğrenir (1/s²)"))

    def reset(self, q0):
        super().reset(q0)
        self._integral = [0.0, 0.0, 0.0]

    def _run(self, gyro, acc, mag, dt):
        w, x, y, z = self.q
        ix, iy, iz = self._integral
        kp, ki = self.kp, self.ki
        out = np.empty((len(gyro), 4))
        for k in range(len(gyro)):
            gx, gy, gz = gyro[k]
            a = _unit(acc[k])
            if a is not None:
                ax, ay, az = a
                vx, vy, vz = 2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z
                ex, ey, ez = ay * vz - az * vy, az * vx - ax * vz, ax * vy - ay * vx
                m = _unit(mag[k]) if mag is not None else None
                if m is not None:
                    mx, my, mz = m
                    hx = mx * (1 - 2 * (y * y + z * z)) + my * 2 * (x * y - w * z) + mz * 2 * (x * z + w * y)
                    hy = mx * 2 * (x * y + w * z) + my * (1 - 2 * (x * x + z * z)) + mz * 2 * (y * z - w * x)
                    bz = mx * 2 * (x * z - w * y) + my * 2 * (y * z + w * x) + mz * (1 - 2 * (x * x + y * y))
                    bx = math.sqrt(hx * hx + hy * hy)
                    wx = bx * (1 - 2 * (y * y + z * z)) + bz * 2 * (x * z - w * y)
                    wy = bx * 2 * (x * y - w * z) + bz * 2 * (w * x + y * z)
                    wz = bx * 2 * (x * z + w * y) + bz * (1 - 2 * (x * x + y * y))
                    ex += my * wz - mz * wy
                    ey += mz * wx - mx * wz
                    ez += mx * wy - my * wx
                if ki > 0:
                    ix += ki * ex * dt
                    iy += ki * ey * dt
                    iz += ki * ez * dt
                    gx, gy, gz = gx + ix, gy + iy, gz + iz
                gx, gy, gz = gx + kp * ex, gy + kp * ey, gz + kp * ez
            h = 0.5 * dt
            w, x, y, z = (w - h * (x * gx + y * gy + z * gz), x + h * (w * gx + y * gz - z * gy),
                          y + h * (w * gy - x * gz + z * gx), z + h * (w * gz + x * gy - y * gx))
            n = 1.0 / math.sqrt(w * w + x * x + y * y + z * z)
            w, x, y, z = w * n, x * n, y * n, z * n
            out[k] = (w, x, y, z)
        self.q = out[-1].copy() if len(out) else self.q
        self._integral = [ix, iy, iz]
        self.bias = -np.array(self._integral)
        return out


class Madgwick(Estimator):
    key, name = "madgwick", "Madgwick"
    estimates_bias = False  # β tek parametre: jiroskop sapması kestirilmez
    params = (Param("beta", "β", 0.05, 0.0, 5.0, 4,
                    "Gradyan adımı (rad/s): büyük β ivmeölçere / manyetometreye daha hızlı yakınsar, daha gürültülü"),)

    def _run(self, gyro, acc, mag, dt):
        w, x, y, z = self.q
        beta = self.beta
        out = np.empty((len(gyro), 4))
        for k in range(len(gyro)):
            gx, gy, gz = gyro[k]
            qw = 0.5 * (-x * gx - y * gy - z * gz)
            qx = 0.5 * (w * gx + y * gz - z * gy)
            qy = 0.5 * (w * gy - x * gz + z * gx)
            qz = 0.5 * (w * gz + x * gy - y * gx)
            a = _unit(acc[k])
            if a is not None:
                ax, ay, az = a
                f1 = 2 * (x * z - w * y) - ax
                f2 = 2 * (w * x + y * z) - ay
                f3 = 2 * (0.5 - x * x - y * y) - az
                sw = -2 * y * f1 + 2 * x * f2
                sx = 2 * z * f1 + 2 * w * f2 - 4 * x * f3
                sy = -2 * w * f1 + 2 * z * f2 - 4 * y * f3
                sz = 2 * x * f1 + 2 * y * f2
                m = _unit(mag[k]) if mag is not None else None
                if m is not None:
                    mx, my, mz = m
                    hx = mx * (1 - 2 * (y * y + z * z)) + my * 2 * (x * y - w * z) + mz * 2 * (x * z + w * y)
                    hy = mx * 2 * (x * y + w * z) + my * (1 - 2 * (x * x + z * z)) + mz * 2 * (y * z - w * x)
                    bz = mx * 2 * (x * z - w * y) + my * 2 * (y * z + w * x) + mz * (1 - 2 * (x * x + y * y))
                    bx = math.sqrt(hx * hx + hy * hy)
                    f4 = 2 * bx * (0.5 - y * y - z * z) + 2 * bz * (x * z - w * y) - mx
                    f5 = 2 * bx * (x * y - w * z) + 2 * bz * (w * x + y * z) - my
                    f6 = 2 * bx * (w * y + x * z) + 2 * bz * (0.5 - x * x - y * y) - mz
                    sw += -2 * bz * y * f4 + (-2 * bx * z + 2 * bz * x) * f5 + 2 * bx * y * f6
                    sx += 2 * bz * z * f4 + (2 * bx * y + 2 * bz * w) * f5 + (2 * bx * z - 4 * bz * x) * f6
                    sy += (-4 * bx * y - 2 * bz * w) * f4 + (2 * bx * x + 2 * bz * z) * f5 + (2 * bx * w - 4 * bz * y) * f6
                    sz += (-4 * bx * z + 2 * bz * x) * f4 + (-2 * bx * w + 2 * bz * y) * f5 + 2 * bx * x * f6
                norm = math.sqrt(sw * sw + sx * sx + sy * sy + sz * sz)
                if norm > 0:
                    qw -= beta * sw / norm
                    qx -= beta * sx / norm
                    qy -= beta * sy / norm
                    qz -= beta * sz / norm
            w, x, y, z = w + qw * dt, x + qx * dt, y + qy * dt, z + qz * dt
            n = 1.0 / math.sqrt(w * w + x * x + y * y + z * z)
            w, x, y, z = w * n, x * n, y * n, z * n
            out[k] = (w, x, y, z)
        self.q = out[-1].copy() if len(out) else self.q
        return out


class EulerKalman(Estimator):
    key, name = "kalman", "Kalman (eksen başına)"
    bias_is_euler = True  # `bias` gövde jiroskop sapması değil, Euler hızı sapması
    params = (Param("sigma_angle", "σ açı (°/√s)", 0.5, 1e-4, 100.0, 4, "Açı süreç gürültüsü"),
              Param("sigma_bias", "σ sapma (°/s/√s)", 0.05, 1e-5, 10.0, 5, "Jiroskop sapması rastgele yürüyüşü"),
              Param("sigma_meas", "σ ölçüm (°)", 3.0, 1e-3, 90.0, 3,
                    "İvmeölçer / manyetometre açısının gürültüsü (doğrusal ivme de bunu büyütür)"))

    def reset(self, q0):
        super().reset(q0)
        roll, pitch, yaw = (float(v) for v in euler_from_quat(self.q))
        self._angle = [roll, pitch, yaw]
        self._bias = [0.0, 0.0, 0.0]
        self._P = [[[0.01, 0.0], [0.0, 1e-4]] for _ in range(3)]

    def _run(self, gyro, acc, mag, dt):
        d2r = math.pi / 180
        qa, qb = (self.sigma_angle * d2r) ** 2, (self.sigma_bias * d2r) ** 2
        r = (self.sigma_meas * d2r) ** 2
        angle, bias, P = self._angle, self._bias, self._P
        out = np.empty((len(gyro), 3))
        for k in range(len(gyro)):
            gx, gy, gz = gyro[k]
            roll, pitch = angle[0], angle[1]
            sr, cr = math.sin(roll), math.cos(roll)
            cp = math.cos(pitch)
            cp = math.copysign(max(abs(cp), 1e-6), cp)
            tp = math.sin(pitch) / cp
            rates = (gx + sr * tp * gy + cr * tp * gz, cr * gy - sr * gz, (sr * gy + cr * gz) / cp)
            ax, ay, az = acc[k]
            meas = ([math.atan2(ay, az), math.atan2(-ax, math.hypot(ay, az)), None] if _valid(acc[k])
                    else [None, None, None])
            if mag is not None and _valid(mag[k]):
                meas[2] = float(tilt_compensated_heading(mag[k], angle[0], angle[1]))
            for i in range(3):
                p = P[i]
                angle[i] += dt * (rates[i] - bias[i])
                p[0][0] += dt * (dt * p[1][1] - p[0][1] - p[1][0] + qa)
                p[0][1] -= dt * p[1][1]
                p[1][0] -= dt * p[1][1]
                p[1][1] += qb * dt
                if meas[i] is not None:
                    innov = (meas[i] - angle[i] + math.pi) % (2 * math.pi) - math.pi
                    s = p[0][0] + r
                    k0, k1 = p[0][0] / s, p[1][0] / s
                    angle[i] += k0 * innov
                    bias[i] += k1 * innov
                    p00, p01 = p[0][0], p[0][1]
                    p[0][0] -= k0 * p00
                    p[0][1] -= k0 * p01
                    p[1][0] -= k1 * p00
                    p[1][1] -= k1 * p01
                angle[i] = (angle[i] + math.pi) % (2 * math.pi) - math.pi
            out[k] = angle
        # Eksen başına model Euler hızlarının sapmasını kestirir (b_euler ≈ T(φ, θ)·b_jiro'nun zaman ortalaması);
        # yatay çevresinde jiroskop sapmasına yakındır, büyük eğimde değildir — arayüz bunu ayrıca belirtir.
        self.bias = np.array(bias)
        q = quat_from_euler(out[:, 0], out[:, 1], out[:, 2])
        self.q = q[-1].copy() if len(q) else self.q
        return q


class _ErrorStateFilter(Estimator):
    """EKF / UKF ortak: hata durumu δx = [δθ (gövde), δb], nominal q ve b ayrı tutulur."""
    params = (Param("sigma_gyro", "σ jiroskop (°/s/√Hz)", 0.3, 1e-4, 100.0, 4, "Jiroskop beyaz gürültüsü"),
              Param("sigma_bias", "σ sapma (°/s/√s)", 0.01, 1e-6, 10.0, 5, "Sapmanın rastgele yürüyüşü"),
              Param("sigma_acc", "σ ivme (normalize)", 0.05, 1e-4, 2.0, 4,
                    "Normalize ivmeölçer yönünün gürültüsü; doğrusal ivmede otomatik büyür"),
              Param("sigma_heading", "σ başlık (°)", 5.0, 0.01, 180.0, 3, "Manyetik başlık ölçümünün gürültüsü"))

    def reset(self, q0):
        super().reset(q0)
        d2r = math.pi / 180
        self.P = np.diag([(5 * d2r) ** 2] * 3 + [(0.5 * d2r) ** 2] * 3)
        self._g_ref = None

    def _noise(self):
        d2r = math.pi / 180
        return (self.sigma_gyro * d2r) ** 2, (self.sigma_bias * d2r) ** 2

    def _acc_r(self, acc_k):
        n = float(np.linalg.norm(acc_k))
        if self._g_ref is None:
            self._g_ref = n
        else:
            self._g_ref += 0.001 * (n - self._g_ref)  # yerçekimi büyüklüğünün yavaş kestirimi (birimden bağımsız)
        dev = abs(n / self._g_ref - 1.0) if self._g_ref > 0 else 0.0
        return (self.sigma_acc ** 2) * (1.0 + (dev / 0.02) ** 2)  # doğrusal ivmede ölçüme daha az güven

    def _process_cov(self, dt):
        sg, sb = self._noise()
        Q = np.zeros((6, 6))
        Q[:3, :3] = np.eye(3) * (sg * dt + sb * dt ** 3 / 3)
        Q[:3, 3:] = Q[3:, :3] = -np.eye(3) * sb * dt ** 2 / 2
        Q[3:, 3:] = np.eye(3) * sb * dt
        return Q

    def _heading_r(self):
        return (self.sigma_heading * math.pi / 180) ** 2


def _qmul_exp(q, vx, vy, vz):
    """q ⊗ exp([vx, vy, vz]) — tek kuaterniyon için saf Python (numpy çağrı yükü yok), birim normlu döner."""
    angle = math.sqrt(vx * vx + vy * vy + vz * vz)
    c = math.cos(0.5 * angle)
    k = math.sin(0.5 * angle) / angle if angle > 1e-12 else 0.5
    ew, ex, ey, ez = c, vx * k, vy * k, vz * k
    w, x, y, z = q
    w, x, y, z = (w * ew - x * ex - y * ey - z * ez, w * ex + x * ew + y * ez - z * ey,
                  w * ey - x * ez + y * ew + z * ex, w * ez + x * ey - y * ex + z * ew)
    n = 1.0 / math.sqrt(w * w + x * x + y * y + z * z)
    return (w * n, x * n, y * n, z * n)


class MEKF(_ErrorStateFilter):
    key, name = "ekf", "EKF (çarpımsal)"

    def _run(self, gyro, acc, mag, dt):
        q = tuple(float(v) for v in self.q)
        b = [float(v) for v in self.bias]
        P = self.P
        Q = self._process_cov(dt)
        r_h = self._heading_r()
        out = np.empty((len(gyro), 4))
        Phi = np.eye(6)
        Phi[:3, 3:] = -np.eye(3) * dt
        V = np.zeros((3, 3))
        for k in range(len(gyro)):
            # ---- öngörü: nominal kuaterniyon jiroskopla, hata kovaryansı Φ ile
            wx, wy, wz = (gyro[k, 0] - b[0]) * dt, (gyro[k, 1] - b[1]) * dt, (gyro[k, 2] - b[2]) * dt
            q = _qmul_exp(q, wx, wy, wz)
            Phi[0, 1], Phi[0, 2], Phi[1, 0] = wz, -wy, -wz  # I − [ω·dt ×]
            Phi[1, 2], Phi[2, 0], Phi[2, 1] = wx, wy, -wx
            P = Phi @ P @ Phi.T + Q
            # ---- ivmeölçer: tahmini yerçekimi yönü v = R(q)ᵀ·ẑ, H = [[v×], 0]
            ax, ay, az = acc[k]
            n = math.sqrt(ax * ax + ay * ay + az * az)
            if 0 < n < math.inf:
                w, x, y, z = q
                vx, vy, vz = 2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)
                V[0, 1], V[0, 2], V[1, 0], V[1, 2], V[2, 0], V[2, 1] = -vz, vy, vz, -vx, -vy, vx
                HP = V @ P[:3]                                  # (3, 6)
                S = HP[:, :3] @ V.T + np.eye(3) * self._acc_r(acc[k])
                K = np.linalg.solve(S, HP).T                    # (6, 3)
                dx = K @ np.array((ax / n - vx, ay / n - vy, az / n - vz))
                q = _qmul_exp(q, dx[0], dx[1], dx[2])
                b[0] += dx[3]
                b[1] += dx[4]
                b[2] += dx[5]
                P = P - K @ HP
                P = 0.5 * (P + P.T)
            # ---- manyetometre: yalnızca başlık (yaw); H = dünya z ekseni (R'nin 3. satırı)
            if mag is not None and _valid(mag[k]):
                mx, my, mz = mag[k]
                w, x, y, z = q
                roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
                pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
                yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                sr, cr, sp, cp = math.sin(roll), math.cos(roll), math.sin(pitch), math.cos(pitch)
                heading = math.atan2(sr * mz - cr * my, cp * mx + sp * sr * my + sp * cr * mz)
                innov = (heading - yaw + math.pi) % (2 * math.pi) - math.pi
                h = np.array((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)))
                Ph = P[:, :3] @ h                           # (6,)
                s_ = float(h @ Ph[:3]) + r_h
                K = Ph / s_
                # Kazanç h'ye izdüşürülür: başlık ölçümü yalnız dünya z etrafındaki dönmeyi ve o eksendeki
                # sapmayı düzeltir (kovaryans çapraz terimleri üzerinden roll / pitch'e bozulma sızmaz).
                K[:3] = h * float(h @ K[:3])
                K[3:] = h * float(h @ K[3:])
                dx = K * innov
                q = _qmul_exp(q, dx[0], dx[1], dx[2])
                b[0] += dx[3]
                b[1] += dx[4]
                b[2] += dx[5]
                IKH = np.eye(6)
                IKH[:, :3] -= np.outer(K, h)
                P = IKH @ P @ IKH.T + np.outer(K, K) * r_h  # Joseph biçimi: kazanç optimal olmasa da P geçerli
                P = 0.5 * (P + P.T)
            out[k] = q
        self.q, self.bias, self.P = np.array(q), np.array(b), P
        return out


class UKF(_ErrorStateFilter):
    key, name = "ukf", "UKF (kokusuz)"
    N = 6
    # α = 1, β = 2, κ = 0: λ = 0 → tüm ağırlıklar pozitif, sigma yayılımı √(n·P)
    _WM = np.r_[0.0, np.full(12, 1 / 12)]
    _WC = np.r_[2.0, np.full(12, 1 / 12)]

    def _sigma(self, P):
        try:
            L = np.linalg.cholesky(self.N * P)
        except np.linalg.LinAlgError:
            vals, vecs = np.linalg.eigh(0.5 * (P + P.T))
            L = vecs @ np.diag(np.sqrt(np.clip(vals, 1e-18, None) * self.N))
        return np.vstack((np.zeros(self.N), L.T, -L.T))  # (13, 6)

    def _run(self, gyro, acc, mag, dt):
        q, b, P = self.q, self.bias, self.P
        Q = self._process_cov(dt)
        r_h = self._heading_r()
        wm, wc = self._WM, self._WC
        out = np.empty((len(gyro), 4))
        for k in range(len(gyro)):
            # ---- öngörü: sigma noktalarını jiroskopla ilerlet
            X = self._sigma(P)
            qs = quat_mul(q, quat_exp(X[:, :3]))
            bs = b + X[:, 3:]
            qs = normalize(quat_mul(qs, quat_exp((gyro[k] - bs) * dt)))
            q_mean = qs[0]
            dth = quat_log(quat_mul(quat_conj(q_mean), qs))
            mean_th = wm @ dth
            b_pred = wm @ bs
            D = np.hstack((dth - mean_th, bs - b_pred))
            P = (D.T * wc) @ D + Q
            q = np.array(_qmul_exp(q_mean, *mean_th))  # tek kuaterniyon: saf Python daha hızlı
            b = b_pred
            # ---- ivmeölçer güncellemesi
            if _valid(acc[k]):
                n = np.linalg.norm(acc[k])
                X = self._sigma(P)
                Z = gravity_body(quat_mul(q, quat_exp(X[:, :3])))
                q, b, P = self._update(q, b, P, X, Z, acc[k] / n, np.eye(3) * self._acc_r(acc[k]))
            # ---- manyetometre: başlık, dünya z etrafındaki dönme olarak (Z_i = h·δθ_i, h = R(q)[2, :]).
            # Sigma noktalarının ZYX yaw'ı kullanılmaz: o, eğime tan(pitch) kadar duyarlıdır ve bozulmayı roll /
            # pitch'e taşır, pitch ±90° yakınında da başlık kazancını sıfıra indirir.
            if mag is not None and _valid(mag[k]):
                roll, pitch, yaw = euler_from_quat(q)
                heading = float(tilt_compensated_heading(mag[k], roll, pitch))
                w, x, y, z = q
                h = np.array((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)))
                X = self._sigma(P)
                Z = (X[:, :3] @ h)[:, None]
                innov = np.array([float(wrap(heading - float(yaw)))])
                q, b, P = self._update(q, b, P, X, Z, innov, np.array([[r_h]]))
            out[k] = q
        self.q, self.bias, self.P = q, b, P
        return out

    def _update(self, q, b, P, X, Z, z, R):
        wm, wc = self._WM, self._WC
        z_mean = wm @ Z
        dZ = Z - z_mean
        Pzz = (dZ.T * wc) @ dZ + R
        Pxz = (X.T * wc) @ dZ
        K = np.linalg.solve(Pzz, Pxz.T).T
        dx = K @ (z - z_mean)
        q = np.array(_qmul_exp(q, *dx[:3]))
        b = b + dx[3:]
        P = P - K @ Pzz @ K.T
        return q, b, 0.5 * (P + P.T)


class ParticleFilter(Estimator):
    key, name = "pf", "Parçacık filtresi"
    params = (Param("particles", "Parçacık sayısı", 600, 50, 20_000, 0, "Daha çok parçacık: daha doğru, daha yavaş"),
              Param("sigma_gyro", "σ jiroskop (°/s/√Hz)", 0.5, 1e-3, 100.0, 4, "Öngörü gürültüsü"),
              Param("sigma_bias", "σ sapma (°/s/√s)", 0.02, 1e-6, 10.0, 5, "Sapma parçacıklarının yürüyüşü"),
              Param("sigma_acc", "σ ivme (normalize)", 0.08, 1e-3, 2.0, 4, "Olabilirlik genişliği (ivmeölçer)"),
              Param("sigma_heading", "σ başlık (°)", 8.0, 0.1, 180.0, 3, "Olabilirlik genişliği (manyetik başlık)"),
              Param("seed", "Tohum", 1, 0, 1_000_000, 0, "Rastgele sayı üretecinin tohumu (tekrarlanabilirlik)"))

    def reset(self, q0):
        super().reset(q0)
        n = int(self.particles)
        self._rng = np.random.default_rng(int(self.seed))
        d2r = math.pi / 180
        self._q = normalize(quat_mul(self.q, quat_exp(self._rng.normal(0, 3 * d2r, (n, 3)))))
        self._b = self._rng.normal(0, 0.3 * d2r, (n, 3))
        self._logw = np.full(n, -math.log(n))
        self._g_ref = None
        self.ess = float(n)
        self.resamples = 0

    def _run(self, gyro, acc, mag, dt):
        d2r = math.pi / 180
        rng = self._rng
        n = len(self._q)
        sg = self.sigma_gyro * d2r / math.sqrt(dt)        # spektral yoğunluk → örnek başına σ
        sb = self.sigma_bias * d2r * math.sqrt(dt)
        sh = self.sigma_heading * d2r
        out = np.empty((len(gyro), 4))
        qs, bs, logw = self._q, self._b, self._logw
        for k in range(len(gyro)):
            omega = gyro[k] - bs + rng.normal(0, sg, (n, 3))
            qs = normalize(quat_mul(qs, quat_exp(omega * dt)))
            bs = bs + rng.normal(0, sb, (n, 3))
            norm = np.linalg.norm(acc[k])
            if _valid(acc[k]):
                if self._g_ref is None:
                    self._g_ref = norm
                else:
                    self._g_ref += 0.001 * (norm - self._g_ref)
                dev = abs(norm / self._g_ref - 1.0)
                sa2 = self.sigma_acc ** 2 * (1.0 + (dev / 0.02) ** 2)
                e = acc[k] / norm - gravity_body(qs)
                logw = logw - np.einsum("ij,ij->i", e, e) / (2 * sa2)
            if mag is not None and _valid(mag[k]):
                # Başlık en olası parçacığın eğimiyle ölçülür; her parçacığın başlığı, ona göre dünya z ekseni
                # etrafındaki "twist" açısıyla (r = q_i ⊗ q_ref*, twist = 2·atan2(r_z, r_w)) bulunur — eğimden
                # bağımsızdır, pitch ±90° yakınında da tanımlıdır.
                best = qs[np.argmax(logw)]
                roll, pitch, yaw_ref = euler_from_quat(best)
                heading = tilt_compensated_heading(mag[k], roll, pitch)
                r = quat_mul(qs, quat_conj(best))
                eh = wrap(yaw_ref + 2 * np.arctan2(r[:, 3], r[:, 0]) - heading)
                logw = logw - eh ** 2 / (2 * sh ** 2)
            logw = logw - logw.max()
            weights = np.exp(logw)
            weights /= weights.sum()
            logw = np.log(np.maximum(weights, 1e-300))
            ref = qs[np.argmax(weights)]
            aligned = np.where((qs @ ref)[:, None] < 0, -qs, qs)
            out[k] = normalize(weights @ aligned)
            self.ess = 1.0 / float(np.sum(weights ** 2))
            if self.ess < n / 2:  # sistematik yeniden örnekleme + pürüzlendirme (parçacık çeşitliliği)
                positions = (rng.random() + np.arange(n)) / n
                idx = np.minimum(np.searchsorted(np.cumsum(weights), positions), n - 1)
                qs = normalize(quat_mul(qs[idx], quat_exp(rng.normal(0, 0.2 * d2r, (n, 3)))))
                bs = bs[idx] + rng.normal(0, 0.01 * d2r, (n, 3))
                logw = np.full(n, -math.log(n))
                self.resamples += 1
        self._q, self._b, self._logw = qs, bs, logw
        weights = np.exp(logw - logw.max())
        weights /= weights.sum()
        self.bias = weights @ bs
        self.q = out[-1].copy() if len(out) else self.q
        return out


ALGORITHMS: dict[str, type[Estimator]] = {cls.key: cls for cls in (Mahony, Madgwick, EulerKalman, MEKF, UKF,
                                                                   ParticleFilter)}


def make_estimator(key: str, **params) -> Estimator:
    return ALGORITHMS[key](**params)


# ============================================================================ değerlendirme ve test verisi
def angle_error_deg(estimate_rad: np.ndarray, reference_rad: np.ndarray) -> np.ndarray:
    return np.degrees(wrap(np.asarray(estimate_rad) - np.asarray(reference_rad)))


def rms(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(np.sqrt(np.mean(v ** 2))) if len(v) else float("nan")


@dataclass
class SimulatedImu:
    fs: float
    gyro: np.ndarray       # (N, 3) rad/s (sapma + gürültü dahil)
    acc: np.ndarray        # (N, 3) m/s²
    mag: np.ndarray        # (N, 3) µT
    truth: np.ndarray      # (N, 3) roll, pitch, yaw — rad
    gyro_bias: np.ndarray  # (3,) rad/s


def simulate_imu(duration: float = 60.0, fs: float = 200.0, seed: int = 1, gyro_bias=(0.02, -0.015, 0.01),
                 gyro_noise: float = 0.005, acc_noise: float = 0.03, mag_noise: float = 0.4,
                 field_ut: float = 50.0, inclination_deg: float = 60.0, linear_acc: float = 0.2) -> SimulatedImu:
    """Analitik yönelim yörüngesinden kusursuz bilinen gerçek değerli IMU / MARG verisi üretir.

    Gövde açısal hızı Euler açılarının türevlerinden tam kinematikle hesaplanır; ivmeölçer R(q)ᵀ·[0, 0, g] + doğrusal
    ivme + gürültü, manyetometre R(q)ᵀ·m_dünya + gürültü ölçer. Jiroskopta sabit sapma ve beyaz gürültü vardır.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(round(duration * fs))) / fs
    two_pi = 2 * np.pi
    d2r = np.pi / 180
    # (genlik °, frekans Hz, faz) bileşenleri; yaw ayrıca yavaş bir tam tur döner
    roll = d2r * (30 * np.sin(two_pi * 0.05 * t) + 10 * np.sin(two_pi * 0.23 * t + 0.5))
    pitch = d2r * (20 * np.sin(two_pi * 0.07 * t + 1.0) + 8 * np.sin(two_pi * 0.31 * t))
    yaw_u = two_pi * 0.01 * t + d2r * 40 * np.sin(two_pi * 0.11 * t)
    droll = d2r * (30 * two_pi * 0.05 * np.cos(two_pi * 0.05 * t) + 10 * two_pi * 0.23 * np.cos(two_pi * 0.23 * t + 0.5))
    dpitch = d2r * (20 * two_pi * 0.07 * np.cos(two_pi * 0.07 * t + 1.0) + 8 * two_pi * 0.31 * np.cos(two_pi * 0.31 * t))
    dyaw = two_pi * 0.01 + d2r * 40 * two_pi * 0.11 * np.cos(two_pi * 0.11 * t)
    sr, cr, sp, cp = np.sin(roll), np.cos(roll), np.sin(pitch), np.cos(pitch)
    p = droll - dyaw * sp
    q = dpitch * cr + dyaw * sr * cp
    r = -dpitch * sr + dyaw * cr * cp
    bias = np.asarray(gyro_bias, dtype=np.float64)
    gyro = np.column_stack((p, q, r)) + bias + gyro_noise * rng.standard_normal((len(t), 3))
    quat = quat_from_euler(roll, pitch, yaw_u)
    Rt = np.swapaxes(rotmat(quat), -1, -2)  # dünya → gövde
    lin_world = linear_acc * np.column_stack((np.sin(two_pi * 0.4 * t), np.cos(two_pi * 0.27 * t),
                                              0.5 * np.sin(two_pi * 0.6 * t)))
    acc = np.einsum("nij,nj->ni", Rt, lin_world + np.array([0.0, 0.0, GRAVITY]))
    acc += acc_noise * rng.standard_normal(acc.shape)
    inc = inclination_deg * d2r
    m_world = field_ut * np.array([np.cos(inc), 0.0, -np.sin(inc)])
    mag = np.einsum("nij,j->ni", Rt, m_world) + mag_noise * rng.standard_normal((len(t), 3))
    return SimulatedImu(fs, gyro, acc, mag, np.column_stack((roll, pitch, wrap(yaw_u))), bias)
