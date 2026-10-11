"""Sistem tanımlama (MATLAB System Identification Toolbox karşılığı) — Aşama 4.

Mimari ve doğrulama ilkesi: docs/asamalar/Asama_4_Kontrol_Tasarimi.md. Hesaplar MATLAB R2025b ile
karşılaştırılmıştır (tests/test_control_sysid_*.py, referanslar tests/data/matlab/sysid_*.json).

Modüller: data (IDData, ön işleme, okuyucular) · models (idpoly/idss/idtf/idproc/idfrd) · linear (arx, armax, oe,
bj, polyest, ar) · ctime (tfest, procest) · subspace (n4sid, ssest) · nonparam (impulseest, spa, etfe, covf,
delayest, arxstruc, selstruc, struc) · evaluate (compare, predict, sim, pe, resid) · nonlinear (nlarx, nlhw).
"""
from spt.control.sysid.ctime import procest, tfest
from spt.control.sysid.data import (IDData, TrendInfo, check_complete, detrend, fill_missing, iddata,
                                    iddata_from_handle, idfilt, mat_variables, merge, read_csv, read_mat, resample,
                                    retrend, split)
from spt.control.sysid.evaluate import CompareResult, ResidResult, compare, pe, predict, resid, sim
from spt.control.sysid.linear import ar, armax, arx, bj, oe, polyest
from spt.control.sysid.models import (IDFRD, IDSS, IDTF, Fit, IDModel, IDPoly, IDProc, Report, idpoly, idproc, idss,
                                      idtf)
from spt.control.sysid.nonlinear import (IDNLARX, IDNLHW, CustomRegressor, DeadZone, Linear, PiecewiseLinear,
                                         Poly1D, PolyRegressor, Saturation, SigmoidNet, UnitGain, WaveletNet, nlarx,
                                         nlhw)
from spt.control.sysid.nonparam import arxstruc, covf, delayest, etfe, impulseest, selstruc, spa, struc
from spt.control.sysid.subspace import d2c_ss, n4sid, ssest

__all__ = [
    # veri
    "IDData", "TrendInfo", "iddata", "merge", "split", "detrend", "retrend", "idfilt", "resample", "fill_missing",
    "check_complete", "iddata_from_handle", "read_csv", "read_mat", "mat_variables",
    # modeller
    "IDModel", "IDPoly", "IDSS", "IDTF", "IDProc", "IDFRD", "Fit", "Report", "idpoly", "idss", "idtf", "idproc",
    "IDNLARX", "IDNLHW",
    # doğrusal kestiriciler
    "arx", "armax", "oe", "bj", "polyest", "ar", "tfest", "procest", "n4sid", "ssest", "d2c_ss",
    # parametrik olmayan, yapı seçimi
    "impulseest", "spa", "etfe", "covf", "delayest", "arxstruc", "selstruc", "struc",
    # doğrusal olmayan
    "nlarx", "nlhw", "Linear", "SigmoidNet", "WaveletNet", "PolyRegressor", "CustomRegressor", "Saturation",
    "DeadZone", "PiecewiseLinear", "Poly1D", "UnitGain",
    # değerlendirme
    "sim", "predict", "compare", "pe", "resid", "CompareResult", "ResidResult",
]
