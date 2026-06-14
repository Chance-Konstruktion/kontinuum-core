"""Regression guards for the 0.3.1 foundation fixes.

Covered:
- ranking/decision key off the EVENT timestamp's hour, not wall clock (so
  replayed / seeded streams land in the right accumbens bucket);
- bounded growth of the two fully-persisted maps that had no eviction;
- metaplasticity state is now included in engine persistence.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from kontinuum_core import KontinuumEngine
from kontinuum_core.hippocampus import Hippocampus
from kontinuum_core.nucleus_accumbens import NucleusAccumbens


def test_remembered_decision_keys_on_event_hour_not_wallclock():
    engine = KontinuumEngine()
    dec = SimpleNamespace(token="kitchen.light.on", token_id=7, entity_id="light.k")

    engine._remember_decision(dec, None, bucket=0, room="kitchen",
                              timestamp=datetime(2026, 1, 1, 3, 0, tzinfo=timezone.utc))
    assert engine._last_decision_ctx["state_key"].endswith("|3")

    engine._remember_decision(dec, None, bucket=0, room="kitchen",
                              timestamp=datetime(2026, 1, 1, 14, 0, tzinfo=timezone.utc))
    assert engine._last_decision_ctx["state_key"].endswith("|14")


def test_nucleus_accumbens_growth_is_bounded():
    na = NucleusAccumbens()
    for i in range(NucleusAccumbens.MAX_ENTRIES + 500):
        na.reinforce(f"state_{i}", f"action_{i}", reward=1.0)
    assert len(na.values) <= NucleusAccumbens.MAX_ENTRIES
    assert len(na.success_counts) <= NucleusAccumbens.MAX_ENTRIES


def test_hippocampus_durations_eviction_is_bounded():
    h = Hippocampus()
    for i in range(Hippocampus.MAX_DURATION_KEYS + 300):
        h.durations[f"{i}_{i + 1}"] = [float(i)]
    h._evict_durations()
    assert len(h.durations) <= Hippocampus.MAX_DURATION_KEYS


def test_metaplasticity_is_included_in_engine_persistence():
    engine = KontinuumEngine()
    blob = engine.to_dict()
    assert "metaplasticity" in blob["modules"]
    # and it round-trips without error
    restored = KontinuumEngine()
    restored.from_dict(blob)
