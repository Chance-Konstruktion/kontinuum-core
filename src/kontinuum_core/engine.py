"""
Main public API for KONTINUUM Core.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .amygdala import Amygdala
from .anterior_cingulate import AnteriorCingulate
from .basal_ganglia import BasalGanglia
from .cerebellum import Cerebellum
from .entorhinal_cortex import EntorhinalCortex
from .hippocampus import Hippocampus
from .hypothalamus import Hypothalamus
from .insula import Insula
from .locus_coeruleus import LocusCoeruleus
from .metaplasticity import Metaplasticity
from .neurorhythms import Neurorhythms
from .nucleus_accumbens import NucleusAccumbens
from .predictive_processing import PredictiveProcessing
from .prefrontal_cortex import PrefrontalCortex
from .reticular import Reticular
from .sleep_consolidation import SleepConsolidation
from .spatial_cortex import SpatialCortex
from .thalamus import Thalamus


@dataclass
class EngineSnapshot:
    surprise: float = 0.0
    anomaly: bool = False
    learning_state: str = "cold_start"
    tick_count: int = 0
    predictions: List[Any] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


class KontinuumEngine:
    def __init__(self):
        self.thalamus = Thalamus()
        self.hippocampus = Hippocampus()
        self.predictive = PredictiveProcessing()
        self.cerebellum = Cerebellum()
        self.basal_ganglia = BasalGanglia()
        self.neurorhythms = Neurorhythms()
        self.sleep_consolidation = SleepConsolidation()
        self.metaplasticity = Metaplasticity()
        self.amygdala = Amygdala()
        self.insula = Insula()
        self.hypothalamus = Hypothalamus()
        self.spatial_cortex = SpatialCortex()
        self.prefrontal_cortex = PrefrontalCortex()
        self.anterior_cingulate = AnteriorCingulate()
        self.entorhinal_cortex = EntorhinalCortex()
        self.locus_coeruleus = LocusCoeruleus()
        self.nucleus_accumbens = NucleusAccumbens()
        self.reticular = Reticular()
        self.tick_count = 0

    def observe(self, event: Dict[str, Any]) -> EngineSnapshot:
        """Process a single event and return current snapshot."""
        token = self.thalamus.update(event)
        memory = self.hippocampus.update(token)
        predictions = self.predictive.update(memory)
        surprise = float(event.get("surprise", 0.0) or 0.0)

        self.cerebellum.update({"predictions": predictions})
        self.basal_ganglia.update({"surprise": surprise})
        self.tick_count += 1
        return EngineSnapshot(
            tick_count=self.tick_count,
            surprise=surprise,
            predictions=[predictions],
            extra={"token": token, "memory": memory},
        )

    def evaluate(self, context: Optional[Dict[str, Any]] = None) -> EngineSnapshot:
        """Force an evaluation cycle."""
        return self.observe(context or {})
