"""Workflow boundary tests for safety and attention gates."""
from __future__ import annotations

from datetime import datetime, timezone

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


def test_engine_attention_gate_filters_repeated_entity_bursts():
    """observe() now routes events through the Reticular burst filter.

    A single entity flooding within the burst window (identical event-time here)
    must be throttled: the gate trips a cooldown and the surplus events come back
    as ``burst_filtered`` instead of feeding learning.
    """
    engine = KontinuumEngine()
    engine.register_entity(
        "binary_sensor.hallway_motion", ha_area="hallway",
        domain="binary_sensor", device_class="motion",
    )
    ts = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    burst_filtered = 0
    for idx in range(40):
        snap = engine.observe({
            "entity_id": "binary_sensor.hallway_motion",
            "new_state": "on" if idx % 2 == 0 else "off",
            "timestamp": ts,  # identical event-time => a genuine burst
        })
        if snap.extra.get("skipped") == "burst_filtered":
            burst_filtered += 1

    assert burst_filtered > 0
    assert engine.reticular.filtered_events > 0
    assert engine.reticular.total_cooldowns > 0
