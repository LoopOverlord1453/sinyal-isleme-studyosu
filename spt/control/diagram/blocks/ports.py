"""Portlar ve alt sistemler: Giriş portu (Inport), Çıkış portu (Outport), Alt sistem (Subsystem).

- Kök modelde Giriş portu dış giriştir (benzetimde `inputs`, trim / doğrusallaştırmada çalışma noktası değeri);
  Çıkış portu model çıkışıdır (sonuçta `outputs`).
- Alt sistem içindeki Giriş / Çıkış portları alt sistem bloğunun portlarıdır (port numarası 1 tabanlı). Derlemede
  alt sistemler düzleştirilir (sanal alt sistem; Simulink "virtual subsystem" anlamı): iç bloklar "Alt/Blok" tam
  adıyla, portlar özdeşlik bağlantısı (tel) olarak yer alır.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from spt.control.diagram.block import Block, Param, SourceBlock, p_ts, register_block


@register_block
class Inport(SourceBlock):
    type_name = "Inport"
    display_name = "Giriş portu"
    category = "ports"
    description = "Kök modelde dış giriş; alt sistemde alt sistem bloğunun girişi."
    simulink = "In1"
    is_virtual = True
    PARAMS = (
        Param("port", "int", 1, "Port numarası", min=1),
        Param("width", "int", 1, "Genişlik (kök modelde)", min=1),
        p_ts(-1.0),
    )

    def __init__(self, name=None, **params):
        super().__init__(name, **params)
        self.value_fn: Callable[[float], np.ndarray] | None = None

    def source_width(self):
        return self.params["width"]

    def output(self, t, x, xd, u):
        if self.value_fn is None:
            return [np.zeros(self.params["width"])]
        return [self.value_fn(t)]

    def linearize(self, t, x, xd, u):
        w = self.params["width"]
        return np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((w, 0)), np.zeros((w, 0))


@register_block
class Outport(Block):
    type_name = "Outport"
    display_name = "Çıkış portu"
    category = "ports"
    description = "Kök modelde model çıkışı (kaydedilir); alt sistemde alt sistem bloğunun çıkışı."
    simulink = "Out1"
    is_virtual = True
    PARAMS = (Param("port", "int", 1, "Port numarası", min=1),)

    def n_outputs(self):
        return 0

    def propagate_widths(self, in_widths, guess=False):
        return []

    def output(self, t, x, xd, u):
        return []

    def linearize(self, t, x, xd, u):
        n = self.in_widths[0]
        return np.zeros((0, 0)), np.zeros((0, n)), np.zeros((0, 0)), np.zeros((0, n))


class Wire(Block):
    """Düzleştirmede alt sistem portlarının yerini alan özdeşlik bloğu (kayıtlı değil)."""
    type_name = "Wire"
    display_name = "Tel"
    category = "ports"
    is_virtual = True
    PARAMS = ()

    def output(self, t, x, xd, u):
        return [u[0]]

    def linearize(self, t, x, xd, u):
        w = self.in_widths[0]
        return np.zeros((0, 0)), np.zeros((0, w)), np.zeros((w, 0)), np.eye(w)


@register_block
class Subsystem(Block):
    type_name = "Subsystem"
    display_name = "Alt sistem"
    category = "ports"
    description = "Blokları gruplar; içindeki Giriş / Çıkış portları bloğun portlarıdır."
    simulink = "Subsystem"
    is_virtual = True
    PARAMS = ()

    def __init__(self, name=None, system: Any = None, **params):
        super().__init__(name, **params)
        if system is None:
            from spt.control.diagram.model import Model
            system = Model(self.name)
        self.system = system

    def _ports(self, kind: str) -> list[Block]:
        ports = [b for b in self.system.blocks.values() if b.type_name == kind]
        nums = sorted(b.params["port"] for b in ports)
        if nums != list(range(1, len(nums) + 1)):
            self.error(f"{'giriş' if kind == 'Inport' else 'çıkış'} port numaraları 1'den başlayıp ardışık "
                       f"olmalı (bulunan: {nums}).")
        return sorted(ports, key=lambda b: b.params["port"])

    def inports(self) -> list[Block]:
        return self._ports("Inport")

    def outports(self) -> list[Block]:
        return self._ports("Outport")

    def n_inputs(self):
        return len(self.inports())

    def n_outputs(self):
        return len(self.outports())

    def input_labels(self):
        return [b.name for b in self.inports()]

    def output_labels(self):
        return [b.name for b in self.outports()]

    def output(self, t, x, xd, u):  # pragma: no cover - düzleştirilir
        raise AssertionError("alt sistem düzleştirilmeden yürütülemez")

    def clone(self, name=None):
        b = Subsystem(name or self.name, system=self.system.copy())
        b.meta = dict(self.meta)
        return b


__all__ = ["Inport", "Outport", "Wire", "Subsystem"]
