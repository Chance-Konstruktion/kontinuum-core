"""Reward loop (dopamine / nucleus accumbens) — explicit-feedback semantics.

Design decision (Option 1, conservative): the reward loop is driven ONLY by the
host-facing ``engine.feedback()`` call — accumbens ``reinforce()`` and the
neurorhythms dopamine outcome fire nowhere else in the core. Autonomous implicit
learning (``prefrontal_cortex.check_implicit_positives``) stays unwired.

These tests lock that contract in:
* ``feedback(True)`` moves the accumbens bias positive and registers a
  dopamine outcome (success_counts grows).
* ``feedback(False)`` moves the accumbens bias negative.
* Pure observation without any feedback (shadow / observer operation) produces
  **zero** dopamine bursts — no accidental autonomous learning.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from kontinuum_core import KontinuumEngine


def _build_engine() -> KontinuumEngine:
    e = KontinuumEngine()
    e.register_entity("switch.kitchen", ha_area="kitchen", domain="switch")
    e.register_entity("light.bedroom_lamp", ha_area="bedroom", domain="light")
    return e


def _drive_pattern(e: KontinuumEngine, cycles: int):
    base = datetime(2026, 6, 13, 19, 0, 0, tzinfo=timezone.utc)
    steps = [
        ("switch.kitchen", "on"),
        ("light.bedroom_lamp", "on"),
        ("switch.kitchen", "off"),
        ("light.bedroom_lamp", "off"),
    ]
    n = 0
    for _ in range(cycles):
        for entity_id, state in steps:
            e.observe({
                "entity_id": entity_id, "new_state": state,
                "timestamp": base + timedelta(minutes=n),
            })
            n += 1


def _remembered_key(e):
    ctx = e._last_decision_ctx
    assert ctx is not None, "no decision remembered to reinforce"
    return ctx["state_key"], ctx["action_key"]


def test_explicit_positive_feedback_reinforces_accumbens_and_dopamine():
    e = _build_engine()
    _drive_pattern(e, cycles=30)
    e.observe({"entity_id": "switch.kitchen", "new_state": "on",
               "timestamp": datetime(2026, 6, 14, 7, 0, 0, tzinfo=timezone.utc)})
    state_key, action_key = _remembered_key(e)

    bias_before = e.nucleus_accumbens.get_bias(state_key, action_key)
    successes_before = e.nucleus_accumbens.success_counts.get((state_key, action_key), 0)

    assert e.feedback(True) is True

    assert e.nucleus_accumbens.get_bias(state_key, action_key) > bias_before
    assert e.nucleus_accumbens.success_counts.get((state_key, action_key), 0) \
        == successes_before + 1


def test_explicit_negative_feedback_pushes_accumbens_negative():
    e = _build_engine()
    _drive_pattern(e, cycles=30)
    e.observe({"entity_id": "switch.kitchen", "new_state": "on",
               "timestamp": datetime(2026, 6, 14, 7, 0, 0, tzinfo=timezone.utc)})
    state_key, action_key = _remembered_key(e)

    bias_before = e.nucleus_accumbens.get_bias(state_key, action_key)
    assert e.feedback(False) is True
    assert e.nucleus_accumbens.get_bias(state_key, action_key) < bias_before


def test_observation_without_feedback_produces_no_dopamine_bursts():
    """Shadow / observer behaviour: the reward loop is host-driven, so a run
    that never calls feedback() must not fire any dopamine burst on its own."""
    e = _build_engine()
    _drive_pattern(e, cycles=50)

    assert e.neurorhythms.total_bursts == 0
    # And no (state, action) has been credited a success without feedback.
    assert all(v == 0 for v in e.nucleus_accumbens.success_counts.values()) \
        or not e.nucleus_accumbens.success_counts
