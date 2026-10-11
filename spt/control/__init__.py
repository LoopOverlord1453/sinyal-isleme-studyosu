"""Kontrol tasarımı, sistem tanımlama, MPC, blok diyagram ve İHA modelleri (Aşama 4).

Mimari ve ortak kurallar: docs/asamalar/Asama_4_Kontrol_Tasarimi.md. MATLAB adlarıyla toplu erişim:

    from spt.control import tf, c2d, step, margin, pidtune, lqr, mpc
    G = tf([1], [1, 2, 1]); Gd = c2d(G, 0.01, "zoh")

Adlar tembel yüklenir (PEP 562): paket içe aktarılınca alt modüller (scipy dahil) hemen yüklenmez ve alt modüllerin
birbirini içe aktarması döngü oluşturmaz. Alt paketler (`sysid`, `diagram`, `uav`) doğrudan içe aktarılır.
"""
from __future__ import annotations

import importlib

from spt.control.errors import ControlError, ControlWarning

_EXPORTS: dict[str, tuple[str, ...]] = {
    "lti": ("tf", "zpk", "ss", "TransferFunction", "ZPK", "StateSpace", "LTI"),
    "convert": ("minreal", "balreal", "balred", "modred", "canon", "hsvd", "ss2ss"),
    "connect": ("feedback", "series", "parallel", "append", "connect", "sumblk", "lft", "loopsens"),
    "discretize": ("c2d", "d2c", "d2d", "pade"),
    "timeresp": ("step", "impulse", "initial", "lsim", "stepinfo", "lsiminfo"),
    "freqresp": ("freqresp", "evalfr", "bode", "nyquist", "nichols", "sigma", "margin", "allmargin", "bandwidth",
                 "dcgain", "norm", "getPeakGain"),
    "analysis": ("pole", "zero", "tzero", "pzmap", "damp", "rlocus", "ctrb", "obsv", "gram", "covar", "isstable"),
    "parse": ("parse_model",),
    "classical": ("pid", "pidstd", "pid2", "pidstd2", "pidtune", "leadlag", "lead", "lag", "notch", "rlocfind"),
    "modern": ("place", "acker", "lqr", "dlqr", "lqrd", "lqi", "lqry", "kalman", "kalmd", "lqg", "reg", "estim",
               "care", "dare"),
    "mpc": ("mpc", "MPC", "mpcstate", "mpcmove", "setmpcsignals", "mpcsimopt"),
}
_WHERE = {name: module for module, names in _EXPORTS.items() for name in names}

__all__ = ["ControlError", "ControlWarning", *_WHERE]


def __getattr__(name: str):
    module = _WHERE.get(name)
    if module is None:
        raise AttributeError(f"module 'spt.control' has no attribute {name!r}")
    value = getattr(importlib.import_module(f"spt.control.{module}"), name)
    globals()[name] = value  # sonraki erişimler doğrudan
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
