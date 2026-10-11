"""Gözlem blokları: Skop / Kayıt (benzetim sonuçlarını tutar) ve Sonlandırıcı."""
from __future__ import annotations

import numpy as np

from spt.control.diagram.block import Block, Param, register_block


class _Sink(Block):
    def n_outputs(self):
        return 0

    def propagate_widths(self, in_widths, guess=False):
        return []

    def output(self, t, x, xd, u):
        return []

    def linearize(self, t, x, xd, u):
        n = sum(self.in_widths)
        return np.zeros((0, 0)), np.zeros((0, n)), np.zeros((0, 0)), np.zeros((0, n))


@register_block
class Scope(_Sink):
    type_name = "Scope"
    display_name = "Skop / kayıt"
    category = "sinks"
    description = ("Girişlerini her kayıt anında saklar; sonuçlar benzetim sonucunda bu bloğun adıyla "
                   "(SimResult.scopes) bulunur.")
    simulink = "Scope / To Workspace"
    PARAMS = (
        Param("inputs", "int", 1, "Giriş sayısı", min=1, max=64),
        Param("labels", "str", "", "Kanal adları (virgülle; boş: giriş adları)"),
    )

    def n_inputs(self):
        return self.params["inputs"]

    def channel_labels(self) -> list[str]:
        given = [s.strip() for s in self.params["labels"].split(",") if s.strip()]
        n = self.n_inputs()
        return [given[i] if i < len(given) else f"{self.name}:{i + 1}" for i in range(n)]


@register_block
class Terminator(_Sink):
    type_name = "Terminator"
    display_name = "Sonlandırıcı"
    category = "sinks"
    description = "Kullanılmayan çıkışları bağlamak için (hiçbir şey yapmaz)."
    simulink = "Terminator"
    PARAMS = ()


__all__ = ["Scope", "Terminator"]
