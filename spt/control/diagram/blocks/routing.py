"""Sinyal yönlendirme blokları: Mux, Demux, Seçici, Vektör birleştirme (Simulink anlamlarıyla).

Seçici indeksleri Simulink / MATLAB gibi **1 tabanlıdır** (arayüzde görünen parametre).
"""
from __future__ import annotations

import numpy as np

from spt.control.diagram.block import Block, Param, p_ts, register_block


def _static(D: np.ndarray):
    return np.zeros((0, 0)), np.zeros((0, D.shape[1])), np.zeros((D.shape[0], 0)), D


@register_block
class Mux(Block):
    type_name = "Mux"
    display_name = "Mux (birleştirici)"
    category = "routing"
    description = "Girişleri art arda tek vektör sinyalde birleştirir."
    simulink = "Mux"
    PARAMS = (Param("inputs", "int", 2, "Giriş sayısı", min=1, max=256), p_ts(-1.0))

    def n_inputs(self):
        return self.params["inputs"]

    def propagate_widths(self, in_widths, guess=False):
        if all(w is not None for w in in_widths):
            return [int(sum(in_widths))]
        if guess:
            return [int(sum(w if w is not None else 1 for w in in_widths))]
        return [None]

    def setup(self, in_widths, out_widths):
        if sum(in_widths) != out_widths[0]:
            self.error(f"çıkış genişliği ({out_widths[0]}) giriş genişlikleri toplamına ({sum(in_widths)}) eşit "
                       "değil (bir geri besleme döngüsünde genişlik tahmini tutmadı; girişlerin genişliğini "
                       "belirginleştirin).")

    def output(self, t, x, xd, u):
        return [np.concatenate(u)]

    def linearize(self, t, x, xd, u):
        w = self.out_widths[0]
        return _static(np.eye(w))


@register_block
class VectorConcatenate(Mux):
    type_name = "VectorConcatenate"
    display_name = "Vektör birleştirme"
    category = "routing"
    description = "Girişleri art arda birleştirir (veri yolu / vektör oluşturma)."
    simulink = "Vector Concatenate"


@register_block
class Demux(Block):
    type_name = "Demux"
    display_name = "Demux (ayırıcı)"
    category = "routing"
    description = "Vektör sinyali parçalara ayırır: eşit parçalar ya da verilen genişlikler."
    simulink = "Demux"
    PARAMS = (
        Param("outputs", "int", 2, "Çıkış sayısı", min=1, max=256),
        Param("widths", "vector", [], "Çıkış genişlikleri (boş: eşit böl)"),
        p_ts(-1.0),
    )

    def check_params(self):
        w = self.params["widths"]
        if w.size:
            if w.size != self.params["outputs"]:
                self.error(f"genişlik listesi {self.params['outputs']} elemanlı olmalı.")
            if np.any(w < 1) or np.any(w != np.round(w)):
                self.error("genişlikler pozitif tam sayı olmalı.")

    def n_outputs(self):
        return self.params["outputs"]

    def propagate_widths(self, in_widths, guess=False):
        n = self.params["outputs"]
        ws = self.params["widths"]
        if ws.size:
            return [int(v) for v in ws]
        w = in_widths[0]
        if w is None:
            return [1] * n if guess else [None] * n
        if w % n:
            self.error(f"{w} elemanlı giriş {n} eşit parçaya bölünemiyor; genişlikleri verin.")
        return [w // n] * n

    def setup(self, in_widths, out_widths):
        if sum(out_widths) != in_widths[0]:
            self.error(f"çıkış genişlikleri toplamı ({sum(out_widths)}) giriş genişliğine ({in_widths[0]}) eşit "
                       "olmalı.")
        self._cuts = np.cumsum([0] + list(out_widths))

    def output(self, t, x, xd, u):
        v = u[0]
        c = self._cuts
        return [v[c[i]:c[i + 1]].copy() for i in range(len(c) - 1)]

    def linearize(self, t, x, xd, u):
        w = self.in_widths[0]
        return _static(np.eye(w))


@register_block
class Selector(Block):
    type_name = "Selector"
    display_name = "Seçici"
    category = "routing"
    description = "Giriş vektöründen verilen (1 tabanlı) indekslerdeki elemanları seçer."
    simulink = "Selector"
    PARAMS = (Param("indices", "vector", [1.0], "İndeksler (1 tabanlı)", min=1.0, min_len=1), p_ts(-1.0))

    def check_params(self):
        idx = self.params["indices"]
        if np.any(idx != np.round(idx)):
            self.error("indeksler tam sayı olmalı.")

    def propagate_widths(self, in_widths, guess=False):
        return [int(self.params["indices"].size)]

    def setup(self, in_widths, out_widths):
        idx = self.params["indices"].astype(int)
        if int(idx.max()) > in_widths[0]:
            self.error(f"indeks {int(idx.max())} giriş genişliğini ({in_widths[0]}) aşıyor.")
        self._idx = idx - 1

    def output(self, t, x, xd, u):
        return [u[0][self._idx]]

    def linearize(self, t, x, xd, u):
        D = np.zeros((self._idx.size, self.in_widths[0]))
        D[np.arange(self._idx.size), self._idx] = 1.0
        return _static(D)


__all__ = ["Mux", "VectorConcatenate", "Demux", "Selector"]
