"""Kontrolcü blokları.

PID Controller (Simulink "PID Controller" / "PID Controller (2DOF)" anlamlarıyla; iç yapısı MATLAB R2025b'den
okunarak doğrulandı)
--------------------------------------------------------------------------------------------------------------
Paralel:  u = P·e_p + I·∫e_i + D·N/(1 + N/s)·e_d        İdeal:  u = P·(e_p + I·∫e_i + D·N/(1 + N/s)·e_d)
1-DOF'ta e_p = e_i = e_d = e; 2-DOF'ta e_p = b·r − y, e_i = r − y, e_d = c·r − y.

- İntegratör girişi (preInt) = I·e_i [+ Kt·(TR − u_sat) izleme] [+ Kb·(u_sat − u_ön) geri hesaplama]; sıkıştırmada
  (clamping) ölü bölge(u_ön) ≠ 0 ve işareti integratör girişinin işaretiyle aynıysa (sgn: > 0 → +1, aksi −1)
  integratör girişi 0 yapılır. İdeal biçimde integratör ve türev P'den önceki birimdedir (P toplamdan sonra çarpar).
- Ayrık zamanda integratör yöntemleri (Ts·I çarpanıyla): İleri Euler Ts/(z−1), Geri Euler Ts·z/(z−1), Trapez
  Ts/2·(z+1)/(z−1); türev filtresi D·N/(1 + N·α(z)) aynı α ile. Geri Euler / Trapez integratörde anti-windup ve
  izleme kararları cebirsel döngü olmaması için integratörün **durumuyla** (o anki giriş katkısı olmadan) hesaplanan
  doyum öncesi toplamla verilir (Simulink'in "Sum Fdbk" yolu).
- Geri Euler / Trapez filtrede Simulink filtre başlangıç koşulunu kullanmaz (bu blok da kullanmaz); İleri Euler ve
  sürekli filtrede kullanılır.
- Doğrusallaştırmada doyum kazanç 1 sayılır (Simulink'in iç Saturation bloğunun varsayılanı), sıkıştırma etkin
  değil kabul edilir; böylece geri hesaplama terimi sıfırdır.
- Sürekli zamanda filtresiz türev desteklenmez (uygun olmayan transfer fonksiyonu; N verin).
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from spt.control.diagram import _lti
from spt.control.diagram.block import Block, Param, SampleTime, p_ts, register_block
from spt.control.diagram.blocks.continuous import LTIBlock, check_abcd
from spt.control.errors import ControlError

_METHODS = {"forward_euler": "İleri Euler", "backward_euler": "Geri Euler", "trapezoidal": "Trapez"}
_KFAC = {"forward_euler": 0.0, "backward_euler": 1.0, "trapezoidal": 0.5}


def _pid_params(two_dof: bool) -> tuple[Param, ...]:
    ps = [
        Param("controller", "choice", "PID", "Kontrolcü türü",
              choices={"PID": "PID", "PI": "PI", "PD": "PD", "P": "P", "I": "I"}, group="Ana"),
        Param("form", "choice", "parallel", "Biçim",
              choices={"parallel": "Paralel: P + I/s + D·N/(1+N/s)", "ideal": "İdeal: P·(1 + I/s + D·N/(1+N/s))"},
              group="Ana"),
        Param("P", "float", 1.0, "Oransal kazanç P", group="Ana"),
        Param("I", "float", 1.0, "İntegral kazancı I", group="Ana"),
        Param("D", "float", 0.0, "Türev kazancı D", group="Ana"),
        Param("N", "float", 100.0, "Türev filtre katsayısı N", unit="rad/s", min=0.0, strict_min=True, group="Ana"),
        Param("use_filter", "bool", True, "Türev filtresi kullan", group="Ana"),
    ]
    if two_dof:
        ps += [Param("b", "float", 1.0, "Oransal ayar ağırlığı b", group="Ana"),
               Param("c", "float", 1.0, "Türev ayar ağırlığı c", group="Ana")]
    ps += [
        Param("integrator_method", "choice", "forward_euler", "İntegratör yöntemi (ayrık)", choices=_METHODS,
              group="Ayrık"),
        Param("filter_method", "choice", "forward_euler", "Filtre yöntemi (ayrık)", choices=_METHODS, group="Ayrık"),
        Param("xi0", "float", 0.0, "İntegratör başlangıç koşulu", group="Başlangıç"),
        Param("xf0", "float", 0.0, "Filtre başlangıç koşulu", group="Başlangıç"),
        Param("limit_output", "bool", False, "Çıkışı sınırla (doyum)", group="Doyum"),
        Param("upper", "float", float("inf"), "Üst doyum sınırı", allow_inf=True, group="Doyum"),
        Param("lower", "float", float("-inf"), "Alt doyum sınırı", allow_inf=True, group="Doyum"),
        Param("anti_windup", "choice", "none", "Anti-windup yöntemi",
              choices={"none": "Yok", "back_calculation": "Geri hesaplama (Kb)", "clamping": "Sıkıştırma (clamping)"},
              group="Doyum"),
        Param("Kb", "float", 1.0, "Geri hesaplama katsayısı Kb", group="Doyum"),
        Param("tracking", "bool", False, "İzleme modu (TR girişi)", group="İzleme"),
        Param("Kt", "float", 1.0, "İzleme katsayısı Kt", group="İzleme"),
        p_ts(0.0, hint="0: sürekli PID, Ts > 0: ayrık PID, −1: girişten miras."),
    ]
    return tuple(ps)


class _PIDBase(Block):
    two_dof = False
    category = "controllers"

    # ------------------------------------------------------------------ yapı
    def check_params(self):
        p = self.params
        if p["upper"] < p["lower"]:
            self.error("üst doyum sınırı alt sınırdan küçük olamaz.")
        if p["anti_windup"] != "none" and not p["limit_output"]:
            self.error("anti-windup için çıkış sınırlaması (doyum) açık olmalı.")
        c = p["controller"]
        self._hasP = c != "I"
        self._hasI = c in ("PID", "PI", "I")
        self._hasD = c in ("PID", "PD")
        self._ideal = p["form"] == "ideal" and self._hasP

    def n_inputs(self):
        return (2 if self.two_dof else 1) + (1 if self.params["tracking"] else 0)

    def input_labels(self):
        labs = ["Ref", "y"] if self.two_dof else ["e"]
        if self.params["tracking"]:
            labs.append("TR")
        return labs

    def _discrete(self) -> bool:
        return self.ts.is_discrete

    def feedthrough(self):
        disc = self._discrete()
        kf = _KFAC[self.params["integrator_method"]] if disc else 0.0
        main = self._hasP or self._hasD or (self._hasI and kf > 0)
        ft = [main] * (2 if self.two_dof else 1)
        if self.params["tracking"]:
            ft.append(self._hasI and kf > 0)
        return ft

    def propagate_widths(self, in_widths, guess=False):
        known = [w for w in in_widths if w is not None]
        if len(known) == len(in_widths) and known:
            return [max(known)]
        return [max(known)] if (guess and known) else [None]

    def setup(self, in_widths, out_widths):
        w = out_widths[0]
        for i, wi in enumerate(in_widths):
            if wi not in (1, w):
                self.error(f"'{self.input_labels()[i]}' girişinin genişliği ({wi}) uyumsuz.")
        self._w = w
        disc = self._discrete()
        p = self.params
        if self._hasD and not disc and not p["use_filter"]:
            self.error("sürekli zamanda filtresiz türev desteklenmez; 'Türev filtresi kullan'ı açın (N).")
        nI = w if self._hasI else 0
        nD = w if self._hasD else 0
        if disc:
            self.nx, self.nxd = 0, nI + nD
        else:
            self.nx, self.nxd = nI + nD, 0
        self._nI = nI
        self._lin = False
        self._sat_mode = np.zeros(w, dtype=int)
        self._cmem_on = False
        self._cmem = np.zeros(w, dtype=bool)
        self._cmem_store = np.zeros(w, dtype=bool)
        self._raw = 0.0
        # sabitler (hız için)
        Ts = self.ts.period if disc else 0.0
        N = p["N"]
        self._disc, self._Ts = disc, Ts
        self._P, self._I, self._D = p["P"], p["I"], p["D"]
        self._Kb, self._Kt = p["Kb"], p["Kt"]
        self._U, self._L = p["upper"], p["lower"]
        self._lim = bool(p["limit_output"])
        self._aw = p["anti_windup"]
        self._trk = bool(p["tracking"])
        self._b = p.get("b", 1.0)
        self._c = p.get("c", 1.0)
        self._kf = _KFAC[p["integrator_method"]] if disc else 0.0
        self._sgnP = self._P if self._ideal else 1.0
        # türev yolu: Dout = g·(D·e_d − c·xf); ayrık filtre güncellemesi kipine göre
        fm = p["filter_method"]
        if not disc:
            self._fg, self._fc, self._fkind, self._fa = N, 1.0, "cont", 0.0
        elif not p["use_filter"]:
            self._fg, self._fc, self._fkind, self._fa = 1.0 / Ts, Ts, "diff", 0.0
        elif fm == "forward_euler":
            self._fg, self._fc, self._fkind, self._fa = N, 1.0, "fe", 0.0
        else:
            a = 1.0 / (1.0 + N * Ts) if fm == "backward_euler" else (2.0 - N * Ts) / (2.0 + N * Ts)
            g = N * a if fm == "backward_euler" else 2.0 * N / (2.0 + N * Ts)
            self._fg, self._fc, self._fkind, self._fa = g, 1.0 - a, "dtf", a

    def _init_states(self) -> np.ndarray:
        w = self._w
        parts = []
        if self._hasI:
            parts.append(np.full(w, self.params["xi0"]))
        if self._hasD:
            use_ic = self._fkind in ("cont", "fe")
            parts.append(np.full(w, self.params["xf0"] if use_ic else 0.0))
        return np.concatenate(parts) if parts else np.zeros(0)

    def x0(self):
        return self._init_states() if self.nx else np.zeros(0)

    def xd0(self):
        return self._init_states() if self.nxd else np.zeros(0)

    # ------------------------------------------------------------------ hesap (eleman bazında; skaler ve vektör)
    def _core(self, xi, xf, ep, ei, ed, tr, sat_mode, clamp_frozen):
        """Tek kanal (float) ya da vektör (numpy) için aynı denklemler. sat_mode / clamp_frozen None değilse
        dondurulmuş kipler kullanılır. Dönüş: (u, pre_sat, pre_fb, pre_int, Dout)."""
        lin = self._lin
        if self._hasD:
            Dout = self._fg * (self._D * ed - self._fc * xf)
        else:
            Dout = 0.0 * ed
        P = self._P
        hasP = self._hasP
        ideal = self._ideal
        if ideal:
            pre_fb = P * (ep + xi + Dout)
        else:
            pre_fb = (P * ep if hasP else 0.0) + xi + Dout
        lim = self._lim and not lin
        U, L = self._U, self._L
        scalar = isinstance(ep, float)
        if lim:
            if sat_mode is None:
                post_fb = (U if pre_fb > U else (L if pre_fb < L else pre_fb)) if scalar \
                    else np.minimum(np.maximum(pre_fb, L), U)
            else:
                post_fb = _frozen_sat(pre_fb, sat_mode, U, L, scalar)
        else:
            post_fb = pre_fb
        if self._hasI:
            pre_int = self._I * ei
            if self._trk:
                pre_int = pre_int + self._Kt * (tr - post_fb)
            raw = pre_int
            aw = self._aw
            if not lin and aw == "back_calculation":
                pre_int = pre_int + self._Kb * (post_fb - pre_fb)
            elif not lin and aw == "clamping":
                if clamp_frozen is not None:
                    clamp = clamp_frozen
                elif scalar:
                    dz = pre_fb - U if pre_fb > U else (pre_fb - L if pre_fb < L else 0.0)
                    clamp = dz != 0.0 and ((dz > 0.0) == (pre_int * self._sgnP > 0.0))
                else:
                    dz = np.where(pre_fb > U, pre_fb - U, np.where(pre_fb < L, pre_fb - L, 0.0))
                    clamp = (dz != 0) & ((dz > 0) == (pre_int * self._sgnP > 0))
                if scalar:
                    if clamp:
                        pre_int = 0.0
                else:
                    pre_int = np.where(clamp, 0.0, pre_int)
            i_out = xi + self._kf * self._Ts * pre_int if self._kf else xi
        else:
            pre_int = raw = 0.0 * ep
            i_out = xi
        if ideal:
            pre_sat = P * (ep + i_out + Dout)
        else:
            pre_sat = (P * ep if hasP else 0.0) + i_out + Dout
        if lim:
            if sat_mode is None:
                uo = (U if pre_sat > U else (L if pre_sat < L else pre_sat)) if scalar \
                    else np.minimum(np.maximum(pre_sat, L), U)
            else:
                uo = _frozen_sat(pre_sat, sat_mode, U, L, scalar)
        else:
            uo = pre_sat
        self._raw = raw
        return uo, pre_sat, pre_fb, pre_int, Dout

    def _eval(self, s: np.ndarray, u: list[np.ndarray], frozen: bool = False):
        """Durum s ve girişlerden ara büyüklükler. Dönüş: (u, pre_sat, pre_fb, pre_int, Dout, ed) — tek kanalda
        float, çok kanalda dizi."""
        nI = self._nI
        if self._w == 1:
            xi = float(s[0]) if self._hasI else 0.0
            xf = float(s[nI]) if self._hasD else 0.0
            if self.two_dof:
                r, y = float(u[0][0]), float(u[1][0])
                ep, ei, ed = self._b * r - y, r - y, self._c * r - y
            else:
                ep = ei = ed = float(u[0][0])
            tr = float(u[-1][0]) if self._trk else 0.0
            sm = int(self._sat_mode[0]) if frozen else None
            cf = bool(self._cmem[0]) if self._cmem_on and not self._lin else None
        else:
            w = self._w
            xi = s[:nI] if self._hasI else np.zeros(w)
            xf = s[nI:] if self._hasD else np.zeros(w)
            if self.two_dof:
                r = np.broadcast_to(u[0], (w,)).astype(float)
                y = np.broadcast_to(u[1], (w,)).astype(float)
                ep, ei, ed = self._b * r - y, r - y, self._c * r - y
            else:
                ep = ei = ed = np.broadcast_to(u[0], (w,)).astype(float)
            tr = np.broadcast_to(u[-1], (w,)).astype(float) if self._trk else np.zeros(w)
            sm = self._sat_mode if frozen else None
            cf = self._cmem if self._cmem_on and not self._lin else None
        return self._core(xi, xf, ep, ei, ed, tr, sm, cf) + (ed,)

    def _frozen(self) -> bool:
        return self.ctx.zc and not self.ctx.major and not self._disc

    def reset(self):
        # sürekli zamanda sıkıştırma kararı Simulink'teki gibi bir Bellek bloğundan geçer: bir önceki ana adımda
        # verilen karar kullanılır (başlangıç: sıkıştırma yok)
        self._cmem_on = (not self._disc) and self._hasI and self._lim and self._aw == "clamping"
        self._cmem = np.zeros(self._w, dtype=bool)          # kullanılan (Bellek çıkışı)
        self._cmem_store = np.zeros(self._w, dtype=bool)    # saklanan (Bellek girişi, son ana adım)

    def major(self, t, x, xd, u):
        if self._cmem_on:
            # Simulink Bellek bloğu ara adımda sabittir: [t_k, t_k+1) boyunca t_k−1'deki karar geçerli
            self._cmem = self._cmem_store.copy()
        if self.ctx.zc and not self._disc and self._lim:
            uo, pre_sat, pre_fb, pre_int, Dout, ed = self._eval(x, u, frozen=False)
            v = np.atleast_1d(np.asarray(pre_sat, dtype=float))
            self._sat_mode = np.where(v > self._U, 1, np.where(v < self._L, -1, 0))

    def commit(self, t, x, xd, u, y):
        if self._cmem_on:
            _, _, pre_fb, _, _, _ = self._eval(x, u, frozen=False)
            pf = np.atleast_1d(np.asarray(pre_fb, dtype=float))
            raw = np.atleast_1d(np.asarray(self._raw, dtype=float))
            U, L = self._U, self._L
            dz = np.where(pf > U, pf - U, np.where(pf < L, pf - L, 0.0))
            eq = np.sign(dz) == np.sign(raw)
            if self._ideal:
                P = self._P
                ok = (eq & (P > 0)) | (~eq & (np.sign(P) != 1))
            else:
                ok = eq
            self._cmem_store = (dz != 0) & ok

    def output(self, t, x, xd, u):
        s = xd if self._disc else x
        uo = self._eval(s, u, self._frozen())[0]
        return [np.array([uo]) if self._w == 1 else uo]

    def derivatives(self, t, x, xd, u):
        _, _, _, pre_int, Dout, _ = self._eval(x, u, self._frozen())
        if self._w == 1:
            if self._hasI and self._hasD:
                return np.array([pre_int, Dout])
            return np.array([pre_int if self._hasI else Dout]) if (self._hasI or self._hasD) else np.zeros(0)
        parts = ([pre_int] if self._hasI else []) + ([Dout] if self._hasD else [])
        return np.concatenate(parts) if parts else np.zeros(0)

    def update(self, t, x, xd, u):
        _, _, _, pre_int, Dout, ed = self._eval(xd, u, False)
        Ts = self._Ts
        nI = self._nI
        parts = []
        if self._hasI:
            parts.append(np.atleast_1d(xd[:nI] + Ts * pre_int))
        if self._hasD:
            xf = xd[nI:]
            k = self._fkind
            if k == "diff":
                nxt = self._D * ed / Ts
            elif k == "fe":
                nxt = xf + Ts * Dout
            else:
                nxt = self._fa * xf + self._D * ed
            parts.append(np.broadcast_to(np.atleast_1d(np.asarray(nxt, dtype=float)), (self._w,)))
        return np.concatenate(parts) if parts else np.zeros(0)

    def zero_crossings(self, t, x, xd, u):
        if self._disc or not self._lim:
            return None
        uo, pre_sat, pre_fb, pre_int, Dout, ed = self._eval(x, u, frozen=True)
        ps = np.atleast_1d(np.asarray(pre_sat, dtype=float))
        z = [ps - self._U, ps - self._L]
        if self._aw == "clamping":
            pf = np.atleast_1d(np.asarray(pre_fb, dtype=float))
            raw = np.broadcast_to(np.atleast_1d(np.asarray(self._raw, dtype=float)), (self._w,))
            z += [pf - self._U, pf - self._L, raw]
        return np.concatenate(z)

    def linearize(self, t, x, xd, u):
        from spt.control.diagram.linearize import block_jacobian
        self._lin = True
        try:
            return block_jacobian(self, t, x, xd, u, analytic=False)
        finally:
            self._lin = False


def _frozen_sat(v, mode, U, L, scalar):
    if scalar:
        return U if mode == 1 else (L if mode == -1 else v)
    return np.where(mode == 1, U, np.where(mode == -1, L, v))


@register_block
class PIDController(_PIDBase):
    type_name = "PIDController"
    display_name = "PID kontrolcü"
    description = ("Simulink PID Controller: paralel / ideal, filtreli türev (N), sürekli ya da ayrık (İleri / Geri "
                   "Euler, Trapez), çıkış doyumu, anti-windup (geri hesaplama / sıkıştırma), izleme modu.")
    simulink = "PID Controller"
    PARAMS = _pid_params(False)


@register_block
class PIDController2DOF(_PIDBase):
    type_name = "PIDController2DOF"
    display_name = "PID kontrolcü (2-DOF)"
    description = ("İki serbestlik dereceli PID: u = P(b·r − y) + I/s(r − y) + D·N/(1+N/s)(c·r − y); girişler "
                   "Ref ve y.")
    simulink = "PID Controller (2DOF)"
    two_dof = True
    PARAMS = _pid_params(True)


# ============================================================================ doğrusal kompanzatörler
def _ts_value(block: Block) -> float:
    st: SampleTime = block.params["Ts"]
    return st.period


def _disc_params() -> tuple[Param, ...]:
    return (
        Param("method", "choice", "tustin", "Ayrıklaştırma yöntemi (Ts > 0)",
              choices={"tustin": "Tustin (çift doğrusal)", "zoh": "Sıfır dereceli tutucu"}),
        Param("prewarp", "float", 0.0, "Ön bükme frekansı (Tustin; 0: yok)", unit="rad/s", min=0.0),
        p_ts(0.0, hint="0: sürekli, Ts > 0: verilen yöntemle ayrıklaştırılmış."),
    )


class _CompBase(LTIBlock):
    category = "controllers"

    def _ss_c(self):
        raise NotImplementedError

    def _build(self):
        st: SampleTime = self.params["Ts"]
        if st.is_inherited or st.is_constant:
            self.error("örnekleme zamanı 0 (sürekli) ya da pozitif (ayrık) olmalı.")
        A, B, C, D = self._ss_c()
        if st.is_discrete:
            return _lti.c2d(A, B, C, D, st.period, self.params["method"], self.params["prewarp"])
        return A, B, C, D


@register_block
class Feedforward(LTIBlock):
    type_name = "Feedforward"
    display_name = "İleri besleme"
    category = "controllers"
    description = ("İleri besleme kazancı ya da transfer fonksiyonu: Ts = 0 iken pay / payda s'de, Ts > 0 iken z'de; "
                   "yalın kazanç için pay = [K], payda = [1].")
    simulink = "Gain / Transfer Fcn"
    PARAMS = (
        Param("num", "vector", [1.0], "Pay (azalan kuvvetler)", min_len=1),
        Param("den", "vector", [1.0], "Payda (azalan kuvvetler)", min_len=1),
        p_ts(0.0, hint="0: sürekli (s), Ts > 0: ayrık (z), −1: miras (yalnız yalın kazançta)."),
    )

    def _build(self):
        num, den = self.params["num"], self.params["den"]
        st: SampleTime = self.params["Ts"]
        A, B, C, D = _lti.tf2ss(num, den)
        if (st.is_inherited or st.is_constant) and A.size:
            self.error("dinamik ileri beslemede örnekleme zamanı 0 (s) ya da pozitif (z) olmalı.")
        return A, B, C, D


@register_block
class LeadLag(_CompBase):
    type_name = "LeadLag"
    display_name = "Öncül-ardıl"
    description = "C(s) = K·(s/ωz + 1)/(s/ωp + 1); ωz < ωp öncül (faz ilerletir), ωz > ωp ardıl."
    simulink = "Transfer Fcn (Lead-Lag)"
    PARAMS = (
        Param("K", "float", 1.0, "Kazanç K"),
        Param("wz", "float", 1.0, "Sıfır frekansı ωz", unit="rad/s", min=0.0, strict_min=True),
        Param("wp", "float", 10.0, "Kutup frekansı ωp", unit="rad/s", min=0.0, strict_min=True),
    ) + _disc_params()

    def _ss_c(self):
        K, wz, wp = self.params["K"], self.params["wz"], self.params["wp"]
        return _lti.tf2ss([K / wz, K], [1.0 / wp, 1.0])


@register_block
class Notch(_CompBase):
    type_name = "Notch"
    display_name = "Çentik filtre"
    description = ("C(s) = (s² + 2ζz·ω0·s + ω0²)/(s² + 2ζp·ω0·s + ω0²); ω0'da derinlik ζz/ζp. Ayrıkta varsayılan "
                   "Tustin, ön bükme ω0'da (çentik frekansı korunur).")
    simulink = "Transfer Fcn (Notch)"
    PARAMS = (
        Param("w0", "float", 10.0, "Çentik frekansı ω0", unit="rad/s", min=0.0, strict_min=True),
        Param("zeta_z", "float", 0.01, "Sıfır sönümü ζz (derinlik)", min=0.0),
        Param("zeta_p", "float", 0.5, "Kutup sönümü ζp (genişlik)", min=0.0, strict_min=True),
        Param("method", "choice", "tustin", "Ayrıklaştırma yöntemi (Ts > 0)",
              choices={"tustin": "Tustin (çift doğrusal)", "zoh": "Sıfır dereceli tutucu"}),
        Param("prewarp_at_notch", "bool", True, "Tustin'de ω0'da ön bükme"),
        p_ts(0.0, hint="0: sürekli, Ts > 0: ayrıklaştırılmış."),
    )

    def _build(self):
        st: SampleTime = self.params["Ts"]
        if st.is_inherited or st.is_constant:
            self.error("örnekleme zamanı 0 (sürekli) ya da pozitif (ayrık) olmalı.")
        w0, zz, zp = self.params["w0"], self.params["zeta_z"], self.params["zeta_p"]
        A, B, C, D = _lti.tf2ss([1.0, 2 * zz * w0, w0 * w0], [1.0, 2 * zp * w0, w0 * w0])
        if st.is_discrete:
            pw = w0 if (self.params["prewarp_at_notch"] and self.params["method"] == "tustin") else 0.0
            if pw and pw * st.period / 2 >= np.pi / 2:
                self.error("çentik frekansı Nyquist frekansından (π/Ts) küçük olmalı.")
            return _lti.c2d(A, B, C, D, st.period, self.params["method"], pw)
        return A, B, C, D


@register_block
class StateFeedback(LTIBlock):
    type_name = "StateFeedback"
    display_name = "Durum geri beslemesi"
    category = "controllers"
    description = "u = Nr·r − K·x (LQR / kutup yerleştirme kazancı); referans girişi isteğe bağlı."
    simulink = "Gain (−K)"
    PARAMS = (
        Param("K", "matrix", [[1.0]], "Kazanç matrisi K (m×n)"),
        Param("use_reference", "bool", False, "Referans girişi (r)"),
        Param("Nr", "matrix", [[1.0]], "Referans kazancı Nr (m×p ya da skaler)"),
        p_ts(-1.0),
    )

    def _build(self):
        K = self.params["K"]
        m, n = K.shape
        if self.params["use_reference"]:
            Nr = self.params["Nr"]
            if Nr.size == 1:
                Nr = float(Nr.ravel()[0]) * np.eye(m)
            if Nr.shape[0] != m:
                self.error(f"Nr {m} satırlı olmalı.")
            D = np.hstack([-K, Nr])
        else:
            D = -K
        return np.zeros((0, 0)), np.zeros((0, D.shape[1])), np.zeros((m, 0)), D

    def port_widths(self):
        n = self.params["K"].shape[1]
        if self.params["use_reference"]:
            return [n, self._D.shape[1] - n]
        return [n]

    def input_labels(self):
        return ["x", "r"] if self.params["use_reference"] else ["x"]


@register_block
class Observer(LTIBlock):
    type_name = "Observer"
    display_name = "Gözlemci (Luenberger)"
    category = "controllers"
    description = ("x̂' = A x̂ + B u + L (y − C x̂ − D u) (sürekli) ya da x̂[k+1] = … (ayrık öngörücü biçim); "
                   "çıkış x̂.")
    simulink = "State-Space (gözlemci)"
    PARAMS = (
        Param("A", "matrix", [[0.0, 1.0], [0.0, 0.0]], "A"),
        Param("B", "matrix", [[0.0], [1.0]], "B", orient="col"),
        Param("C", "matrix", [[1.0, 0.0]], "C"),
        Param("D", "matrix", [[0.0]], "D"),
        Param("L", "matrix", [[10.0], [25.0]], "Gözlemci kazancı L (n×p)", orient="col"),
        Param("x0", "vector", 0.0, "Başlangıç kestirimi x̂0"),
        p_ts(0.0, hint="0: sürekli gözlemci, Ts > 0: ayrık (öngörücü) gözlemci."),
    )

    def _build(self):
        A, B, C, D = check_abcd(self, self.params["A"], self.params["B"], self.params["C"], self.params["D"])
        n, m, p = A.shape[0], B.shape[1], C.shape[0]
        L = self.params["L"]
        if L.shape != (n, p):
            if L.size == n * p:
                L = L.reshape(n, p)
            else:
                self.error(f"L {n}×{p} olmalı ({L.shape[0]}×{L.shape[1]} verildi).")
        st: SampleTime = self.params["Ts"]
        if st.is_inherited or st.is_constant:
            self.error("örnekleme zamanı 0 (sürekli) ya da pozitif (ayrık) olmalı.")
        self._m, self._p = m, p
        Ao = A - L @ C
        Bo = np.hstack([B - L @ D, L])
        return Ao, Bo, np.eye(n), np.zeros((n, m + p))

    def _x0(self):
        return self.params["x0"]

    def port_widths(self):
        return [self._m, self._p]

    def input_labels(self):
        return ["u", "y"]


@register_block
class KalmanFilter(Block):
    type_name = "KalmanFilter"
    display_name = "Ayrık Kalman filtresi"
    category = "controllers"
    description = ("x[n+1] = A x + B u + G w, y = C x + D u + v (E[ww'] = Q, E[vv'] = R, E[wv'] = N). 'Süzgeç' "
                   "biçiminde çıkış x̂[n|n], 'öngörücü' biçiminde x̂[n|n−1]; kararlı hâl (DARE) ya da zamanla "
                   "değişen kazanç.")
    simulink = "Kalman Filter"
    PARAMS = (
        Param("A", "matrix", [[1.0, 0.1], [0.0, 1.0]], "A"),
        Param("B", "matrix", [[0.005], [0.1]], "B", orient="col"),
        Param("C", "matrix", [[1.0, 0.0]], "C"),
        Param("D", "matrix", [[0.0]], "D"),
        Param("G", "matrix", [], "G (boş: birim matris)"),
        Param("Q", "matrix", [[0.01, 0.0], [0.0, 0.01]], "Süreç gürültüsü kovaryansı Q"),
        Param("R", "matrix", [[0.1]], "Ölçüm gürültüsü kovaryansı R"),
        Param("N", "matrix", [], "Çapraz kovaryans N (boş: 0)"),
        Param("x0", "vector", 0.0, "Başlangıç kestirimi x̂[0|−1]"),
        Param("P0", "matrix", [], "Başlangıç kovaryansı P[0|−1] (boş: birim)"),
        Param("form", "choice", "filter", "Kestirim biçimi",
              choices={"filter": "Süzgeç x̂[n|n] (güncel ölçümü kullanır)", "predictor": "Öngörücü x̂[n|n−1]"}),
        Param("steady_state", "bool", True, "Kararlı hâl kazancı (DARE)"),
        p_ts(-1.0),
    )

    def check_params(self):
        A, B, C, D = check_abcd(self, self.params["A"], self.params["B"], self.params["C"], self.params["D"])
        n, p = A.shape[0], C.shape[0]
        if n == 0:
            self.error("en az bir durum gerekli.")
        G = self.params["G"]
        G = np.eye(n) if G.size == 0 else G
        if G.shape[0] != n:
            if G.ndim == 2 and G.shape[0] == 1 and G.shape[1] == n:
                G = G.T
            else:
                self.error(f"G {n} satırlı olmalı.")
        nw = G.shape[1]
        Q = self.params["Q"]
        if Q.size == 1 and nw != 1:
            Q = float(Q.ravel()[0]) * np.eye(nw)
        R = self.params["R"]
        if R.size == 1 and p != 1:
            R = float(R.ravel()[0]) * np.eye(p)
        if Q.shape != (nw, nw):
            self.error(f"Q {nw}×{nw} olmalı.")
        if R.shape != (p, p):
            self.error(f"R {p}×{p} olmalı.")
        Nn = self.params["N"]
        Nn = np.zeros((nw, p)) if Nn.size == 0 else Nn
        if Nn.shape != (nw, p):
            self.error(f"N {nw}×{p} olmalı.")
        if not np.allclose(Q, Q.T) or not np.allclose(R, R.T):
            self.error("Q ve R simetrik olmalı.")
        if np.min(np.linalg.eigvalsh((R + R.T) / 2)) <= 0:
            self.error("R pozitif tanımlı olmalı.")
        P0 = self.params["P0"]
        P0 = np.eye(n) if P0.size == 0 else (float(P0.ravel()[0]) * np.eye(n) if P0.size == 1 else P0)
        if P0.shape != (n, n):
            self.error(f"P0 {n}×{n} olmalı.")
        x0 = np.atleast_1d(self.params["x0"])
        if x0.size not in (1, n):
            self.error(f"x0 {n} elemanlı olmalı.")
        self._A, self._B, self._C, self._D, self._G = A, B, C, D, G
        self._Q, self._R, self._N, self._P0 = Q, R, Nn, P0
        self._x0v = np.broadcast_to(x0, (n,)).astype(float).copy()
        self._n, self._m, self._p = n, B.shape[1], p
        self._Pss = None
        if self.params["steady_state"]:
            GQG = G @ Q @ G.T
            S = G @ Nn
            try:
                P = sla.solve_discrete_are(A.T, C.T, GQG, R, s=S)
            except (np.linalg.LinAlgError, ValueError) as exc:
                self.error(f"kararlı hâl Kalman kazancı bulunamadı (DARE çözümsüz: {exc}); (A, C) "
                           "gözlenebilir / (A, G·√Q) kararlılaştırılabilir olmalı.")
            self._Pss = (P + P.T) / 2

    def n_inputs(self):
        return 2

    def input_labels(self):
        return ["u", "y"]

    def allowed_sample_times(self):
        return "discrete"

    def feedthrough(self):
        f = self.params["form"] == "filter"
        return [f, f]

    def propagate_widths(self, in_widths, guess=False):
        return [self._n]

    def setup(self, in_widths, out_widths):
        if in_widths[0] != self._m or in_widths[1] != self._p:
            self.error(f"girişler u ({self._m}) ve y ({self._p}) genişliğinde olmalı (bağlanan {in_widths[0]}, "
                       f"{in_widths[1]}).")
        self.nxd = self._n

    def xd0(self):
        return self._x0v.copy()

    def reset(self):
        self._P = self._Pss.copy() if self._Pss is not None else self._P0.copy()

    def _gains(self):
        A, C, G, R, Nn = self._A, self._C, self._G, self._R, self._N
        P = self._P
        S = C @ P @ C.T + R
        Si = np.linalg.inv(S)
        M = P @ C.T @ Si
        L = (A @ P @ C.T + G @ Nn) @ Si
        return M, L, S

    def gains(self) -> dict[str, np.ndarray]:
        """Güncel kazançlar (kararlı hâlde sabit): M (yenilik → x̂[n|n]), L (öngörücü), P (P[n|n−1])."""
        M, L, _ = self._gains()
        return {"M": M, "L": L, "P": self._P.copy()}

    def output(self, t, x, xd, u):
        if self.params["form"] == "predictor":
            return [xd.copy()]
        M, _, _ = self._gains()
        e = u[1] - self._C @ xd - self._D @ u[0]
        return [xd + M @ e]

    def update(self, t, x, xd, u):
        _, L, _ = self._gains()
        e = u[1] - self._C @ xd - self._D @ u[0]
        return self._A @ xd + self._B @ u[0] + L @ e

    def commit(self, t, x, xd, u, y):
        if self._Pss is None:
            M, L, S = self._gains()
            A, G, Q = self._A, self._G, self._Q
            P = A @ self._P @ A.T + G @ Q @ G.T - L @ S @ L.T
            self._P = (P + P.T) / 2

    def linearize(self, t, x, xd, u):
        M, L, _ = self._gains()
        A, B, C, D = self._A, self._B, self._C, self._D
        n = self._n
        Al = A - L @ C
        Bl = np.hstack([B - L @ D, L])
        if self.params["form"] == "predictor":
            return Al, Bl, np.eye(n), np.zeros((n, self._m + self._p))
        Cl = np.eye(n) - M @ C
        Dl = np.hstack([-M @ D, M])
        return Al, Bl, Cl, Dl


__all__ = ["PIDController", "PIDController2DOF", "Feedforward", "LeadLag", "Notch", "StateFeedback", "Observer",
           "KalmanFilter", "ControlError"]
