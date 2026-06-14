"""Workflow boundary tests for safety and attention gates."""
from __future__ import annotations

import pytest

from kontinuum_core import KontinuumEngine


def test_safety_critical_lock_actions_are_vetoed_at_the_risk_boundary():
    """The safety boundary must refuse lock actions even at high confidence."""
    engine = KontinuumEngine()

    assessment = engine.amygdala.assess(
        "entrance.lock.locked",
        "lock",
        "entrance",
        "locked",
        confidence=0.99,
    )

    assert assessment["decision"] == "VETO"
    assert assessment["risk"] == 1.0
    assert engine.amygdala.total_vetoes == 1


@pytest.mark.xfail(reason="Engine.observe() currently wires Reticular but never calls should_process().")
def test_engine_attention_gate_filters_repeated_entity_bursts():
    """The full observe workflow should use the Reticular burst filter."""
    engine = KontinuumEngine()
    for idx in range(engine.reticular.BURST_LIMIT + 2):
        engine.observe({"entity_id": "sensor.noisy_temperature", "new_state": str(idx)})

    assert engine.reticular.filtered_events > 0
    assert engine.reticular.total_cooldowns > 0
