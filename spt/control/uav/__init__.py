"""İHA / drone dinamik modelleri (Aşama 4) — arayüzsüz, numpy / scipy.

- `quadrotor`: çok rotorlu 6-DOF (NED, kuaterniyonlu), motor dinamiği, karıştırıcı (PX4 / Betaflight doygunluk
  önceliği), askı trimi, analitik doğrusallaştırma, standart küçük açı modeli.
- `fixedwing`: Beard & McLain sabit kanat 6-DOF (Aerosonde), rüzgar / Dryden, trim (Va, γ, R), boylamsal / yanal
  doğrusal modeller, kitabın transfer fonksiyonu katsayıları, mod çözümlemesi.
- `common`: döndürmeler, kuaterniyonlar, sayısal Jacobian, benzetim, `LinearModel`, motor / pervane yardımcıları,
  algılayıcı modelleri (IMU, barometre, GPS, manyetometre).
- `autopilot`: otopilot kazanç tasarımı (kademeli PID — pidtune, LQR tutum — dlqr, MPC tahmin modeli, sabit kanat
  ardışık döngü kapama — Beard & McLain Böl. 6), basamak ölçütleri.
- `diagram_blocks`: blok diyagram eklenti blokları (dörtrotor / sabit kanat bitkisi, karıştırıcı, ters karıştırıcı,
  ivme → tutum + itki, doğrusal MPC, IMU / barometre / GPS) — diyagram kütüphanesi kendiliğinden yükler.
- `diagram_templates`: düzenleyicinin otopilot şablonları (`TEMPLATES`).

Ortak model arabirimi (blok diyagram motoru için): `nx`, `nu`, `state_names`, `input_names`, `output_names`,
`derivatives(x, u, t)`, `output(x, u, t)`, `linearize(x0, u0) → LinearModel(A, B, C, D, adlar)`.
"""
from spt.control.uav.common import DrydenGust, ImuModel, LinearModel, SimResult, simulate
from spt.control.uav.fixedwing import (
    FixedWing, FixedWingParams, TrimResult, book_transfer_functions, lateral, longitudinal, mode_analysis,
    transfer_function_coefficients, wind_function,
)
from spt.control.uav.quadrotor import (
    Mixer, Quadrotor, QuadrotorParams, axis_transfer_functions, hover_small_angle_model,
)

__all__ = [
    "DrydenGust", "FixedWing", "FixedWingParams", "ImuModel", "LinearModel", "Mixer", "Quadrotor", "QuadrotorParams",
    "SimResult", "TrimResult", "axis_transfer_functions", "book_transfer_functions", "hover_small_angle_model",
    "lateral", "longitudinal", "mode_analysis", "simulate", "transfer_function_coefficients", "wind_function",
]
