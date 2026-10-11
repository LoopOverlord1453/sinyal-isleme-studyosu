"""İşlem zinciri: sıralı filtre aşamaları (ör. DC kaldır → bant geçiren → notch).

Aşamalar Hz cinsinden tanımlıdır; zincir her akışa o akışın örnekleme hızıyla yeniden tasarlanarak uygulanır.
Böylece aynı zincir 10 kHz'lik ve 48 kHz'lik iki akışa da doğru kesim frekanslarıyla uygulanabilir.
Zincir JSON olarak (.sptchain) kaydedilip başka oturumlarda / akışlarda kullanılabilir.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from spt.dsp.filters import Design, FilterError, FilterSpec, apply_offline, design

CHAIN_FORMAT = "spt-chain"
CHAIN_VERSION = 1


_UIDS = itertools.count(1)


@dataclass(frozen=True)
class ChainStage:
    spec: FilterSpec
    enabled: bool = True
    # Oturum içi kimlik: aç / kapat (`replace`) ve sıralama değişse de korunur; düzenlenen adımın zincirdeki yeri
    # indeksle değil bununla bulunur. Eşitlik ve kayda katılmaz.
    uid: int = field(default_factory=lambda: next(_UIDS), compare=False)


class ProcessingChain:
    """Değişmez zincir: her düzenleme yeni bir zincir döndürür (geri al / yinele ve önbellek için güvenli)."""

    def __init__(self, stages=()):
        self.stages: tuple[ChainStage, ...] = tuple(stages)
        self._cache: dict[float, tuple[list[Design], list[str]]] = {}

    # ---------------------------------------------------------------- düzenleme
    def added(self, stage: ChainStage) -> ProcessingChain:
        return ProcessingChain(self.stages + (stage,))

    def index_of(self, uid: int) -> int | None:
        return next((i for i, s in enumerate(self.stages) if s.uid == uid), None)

    def replaced(self, index: int, stage: ChainStage) -> ProcessingChain:
        stages = list(self.stages)
        stages[index] = stage
        return ProcessingChain(stages)

    def removed(self, index: int) -> ProcessingChain:
        return ProcessingChain(s for i, s in enumerate(self.stages) if i != index)

    def moved(self, index: int, to: int) -> ProcessingChain:
        stages = list(self.stages)
        stages.insert(to, stages.pop(index))
        return ProcessingChain(stages)

    def toggled(self, index: int, enabled: bool) -> ProcessingChain:
        return self.replaced(index, replace(self.stages[index], enabled=enabled))

    def __eq__(self, other) -> bool:
        return isinstance(other, ProcessingChain) and self.stages == other.stages

    def __hash__(self) -> int:
        return hash(self.stages)

    def __len__(self) -> int:
        return len(self.stages)

    @property
    def enabled_count(self) -> int:
        return sum(1 for s in self.stages if s.enabled)

    def describe(self) -> str:
        names = [s.spec.label().split(" · ")[0] for s in self.stages if s.enabled]
        return " → ".join(names) if names else "boş"

    # ---------------------------------------------------------------- tasarım
    def designs(self, fs: float) -> tuple[list[Design], list[str]]:
        """Etkin aşamaların `fs` için tasarımları ve tasarlanamayanlar için hata iletileri."""
        cached = self._cache.get(fs)
        if cached is not None:
            return cached
        designs, errors = [], []
        for i, stage in enumerate(self.stages, 1):
            if not stage.enabled:
                continue
            try:
                designs.append(design(replace(stage.spec, fs=fs)))
            except FilterError as exc:
                errors.append(f"{i}. adım ({stage.spec.label()}): {exc}")
        self._cache[fs] = (designs, errors)
        return designs, errors

    def apply(self, x: np.ndarray, fs: float, zero_phase: bool = False) -> np.ndarray:
        designs, errors = self.designs(fs)
        if errors:
            raise FilterError("; ".join(errors))
        return apply_offline(designs, x, zero_phase)

    # ---------------------------------------------------------------- kalıcılık
    def to_dict(self) -> dict:
        return {"format": CHAIN_FORMAT, "version": CHAIN_VERSION,
                "stages": [{"enabled": s.enabled, "spec": s.spec.to_dict()} for s in self.stages]}

    @classmethod
    def from_dict(cls, data: dict) -> ProcessingChain:
        if not isinstance(data, dict) or data.get("format") != CHAIN_FORMAT:
            raise FilterError("Bu dosya bir işlem zinciri değil")
        if int(data.get("version", 0)) > CHAIN_VERSION:
            raise FilterError("İşlem zinciri dosyası bu sürümden daha yeni")
        stages = []
        for item in data.get("stages", []):
            stages.append(ChainStage(FilterSpec.from_dict(item.get("spec", {})), bool(item.get("enabled", True))))
        return cls(stages)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ProcessingChain:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise FilterError(f"Zincir dosyası okunamadı: {exc}") from exc
        return cls.from_dict(data)
