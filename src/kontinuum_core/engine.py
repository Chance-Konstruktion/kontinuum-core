"""Main public API for KONTINUUM Core.

The KontinuumEngine wires all 18 neuro-inspired modules into a single
pipeline. It is HA-free and can be used standalone or embedded in a
host integration (ha-kontinuum, ha-kontinuum-lite).

Pipeline (per event):
    thalamus.process()   → token_id + normalized state
    hypothalamus.absorb()→ homeostasis update + optional transition
    insula.process()     → mode detection + optional transition
    context vector       → time(9) + hypothalamus(9) + insula(3) = 21
    hippocampus.predict()→ top-k predictions
    predictive.compute_surprise() + get_learn_weight()
    hippocampus.learn(learn_weight)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
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

ANOMALY_THRESHOLD = 0.7


@dataclass
class EngineSnapshot:
    surprise: float = 0.0
    anomaly: bool = False
    learning_state: str = "cold_start"
    tick_count: int = 0
    token_id: Optional[int] = None
    token: Optional[str] = None
    predictions: List[Any] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


class KontinuumEngine:
    """Neuro-inspired learning engine. HA-free."""

    def __init__(self):
        self.thalamus = Thalamus()
        self.hippocampus = Hippocampus()
        self.predictive = PredictiveProcessing()
        self.cerebellum = Cerebellum()
        self.basal_ganglia = BasalGanglia()
        self.neurorhythms = Neurorhythms()
        self.sleep_consolidation = SleepConsolidation()
        self.metaplasticity = Metaplasticity(
            brain_modules={
                "hippocampus": self.hippocampus,
                "predictive": self.predictive,
            }
        )
        self.amygdala = Amygdala()
        self.insula = Insula()
        self.hypothalamus = Hypothalamus()
        self.spatial_cortex = SpatialCortex()
        self.prefrontal_cortex = PrefrontalCortex(self.amygdala)
        self.anterior_cingulate = AnteriorCingulate()
        self.entorhinal_cortex = EntorhinalCortex()
        self.locus_coeruleus = LocusCoeruleus()
        self.nucleus_accumbens = NucleusAccumbens()
        self.reticular = Reticular()
        self.tick_count = 0

    # ------------------------------------------------------------------
    # Entity registration
    # ------------------------------------------------------------------
    def register_entity(self, entity_id: str, **kwargs) -> None:
        self.thalamus.register_entity(entity_id, **kwargs)

    # ------------------------------------------------------------------
    # Main observation pipeline
    # ------------------------------------------------------------------
    def observe(self, event: Optional[Dict[str, Any]] = None) -> EngineSnapshot:
        self.tick_count += 1
        event = event or {}
        entity_id = event.get("entity_id")
        new_state = event.get("new_state")

        if not entity_id or new_state is None:
            return self._snapshot(extra={"skipped": "no_entity_or_state"})

        timestamp = event.get("timestamp") or datetime.now(timezone.utc)
        old_state = event.get("old_state")

        signal = self.thalamus.process(entity_id, new_state, old_state, timestamp)
        if signal is None:
            return self._snapshot(extra={"skipped": "filtered"})

        token_id = signal["token_id"]
        token = signal["token"]
        room = signal["room"]
        semantic = signal["semantic"]
        state = signal["state"]

        # Homeostasis absorption (energy/climate side-channel)
        if self.hypothalamus.is_hypothalamus_signal(semantic):
            self.hypothalamus.absorb(room, semantic, state, entity_id)

        # Mode detection (insula may emit transition tokens; ignored here
        # but kept in insula state so get_mode_context() reflects them)
        self.insula.process(semantic, state, room, token)

        # Build 21-dim context vector
        ctx = (
            list(self.thalamus.encode_time_context(timestamp))
            + list(self.hypothalamus.get_context_vector())
            + list(self.insula.get_mode_context())
        )

        predictions = self.hippocampus.predict(ctx, top_k=5)
        surprise = self.predictive.compute_surprise(token_id, predictions)
        learn_weight = self.predictive.get_learn_weight()
        self.hippocampus.learn(token_id, ctx, timestamp, learn_weight=learn_weight)

        return self._snapshot(
            surprise=surprise,
            anomaly=surprise >= ANOMALY_THRESHOLD,
            token_id=token_id,
            token=token,
            predictions=predictions,
        )

    def evaluate(self, context: Optional[Dict[str, Any]] = None) -> EngineSnapshot:
        return self.observe(context or {})

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def _learning_state(self) -> str:
        n = self.hippocampus.total_events
        acc = self.hippocampus.accuracy
        if n < 100:
            return "cold_start"
        if n < 1000 or acc < 0.3:
            return "learning"
        return "stable"

    def _snapshot(
        self,
        surprise: float = 0.0,
        anomaly: bool = False,
        token_id: Optional[int] = None,
        token: Optional[str] = None,
        predictions: Optional[List[Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> EngineSnapshot:
        return EngineSnapshot(
            surprise=float(surprise),
            anomaly=bool(anomaly),
            learning_state=self._learning_state(),
            tick_count=self.tick_count,
            token_id=token_id,
            token=token,
            predictions=predictions or [],
            extra=extra or {},
        )
