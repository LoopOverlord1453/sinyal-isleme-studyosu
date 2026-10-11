"""Blok kütüphanesi: modüller içe aktarıldığında bloklar kaydolur.

Eklenti keşfi: aşağıdaki isteğe bağlı modüller varsa (başka alt paketlerin blokları; ör. MPC, İHA) içe aktarılır ve
içlerindeki `@register_block` sınıfları kütüphaneye eklenir. Modül yoksa sessizce geçilir; varsa ama hata verirse
`ControlWarning` ile bildirilir (kütüphanenin geri kalanı çalışır).
"""
from __future__ import annotations

import importlib
import importlib.util
import warnings

from spt.control.diagram.blocks import (continuous, controllers, discrete, nonlinear, plants, ports, routing,
                                        sinks, sources)
from spt.control.errors import ControlWarning

BUILTIN_MODULES = (sources, continuous, discrete, nonlinear, controllers, routing, sinks, ports, plants)

PLUGIN_MODULES: tuple[str, ...] = (
    "spt.control.mpc.diagram_blocks",
    "spt.control.uav.diagram_blocks",
)


def load_plugins(modules: tuple[str, ...] = PLUGIN_MODULES) -> list[str]:
    """İsteğe bağlı eklenti modüllerini yükler; yüklenenlerin adlarını döndürür."""
    loaded = []
    for name in modules:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            continue
        try:
            importlib.import_module(name)
            loaded.append(name)
        except Exception as exc:  # eklenti hatası kütüphaneyi bozmamalı
            warnings.warn(f"Blok eklentisi '{name}' yüklenemedi: {exc}", ControlWarning, stacklevel=2)
    return loaded


load_plugins()
