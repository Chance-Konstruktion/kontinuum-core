"""Tests that lock in the full module wiring of KontinuumEngine.observe().

Before this wiring the engine instantiated 18 modules but only drove 6 in
its pipeline; the rest were dead weight. These tests assert that every
decision-relevant module now actually receives data and influences the
snapshot, that the cognitive-control loop is closed, that the host-facing
reward loop (feedback) reaches the outcome modules, and that the whole
engine state round-trips through to_dict/from_dict.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from kontinuum_core import KontinuumEngine
from kontinuum_core.predictive_processing import (
    ANOMALY_DEFAULT_THRESHOLD,
    PredictiveProcessing,
)


def _build_engine() -> KontinuumEngine:
    e = KontinuumEngine()
    e.register_entity("switch.kitchen", ha_area="kitchen", domain="switch")
    e.register_entity("light.bedroom_lamp", ha_area="bedroom", domain="light")
    return e


def _drive_pattern(e: KontinuumEngine, cycles: int):
    """Feed a repeating switch.on -> light.on -> switch.off -> light.off loop.

    Returns the list of snapshots produced (one per processed event).
    """
    base = datetime(2026, 6, 13, 19, 0, 0, tzinfo=timezone.utc)
    steps = [
        ("switch.kitchen", "on"),
        ("light.bedroom_lamp", "on"),
        ("switch.kitchen", "off"),
        ("light.bedroom_lamp", "off"),
    ]
    snaps = []
    n = 0
    for c in range(cycles):
        for entity_id, state in steps:
            ts = base + timedelta(minutes=n)
            snaps.append(e.observe({
                "entity_id": entity_id,
                "new_state": state,
                "timestamp": ts,
            }))
            n += 1
    return snaps


# ---------------------------------------------------------------------------
# Module activation
# ---------------------------------------------------------------------------

def test_observe_drives_every_decision_module():
    """A run must leave a trace in each newly-wired module, not just 6."""
    e = _build_engine()
    snaps = _drive_pattern(e, cycles=30)
    processed = [s for s in snaps if "skipped" not in s.extra]
    assert processed, "no events were processed"

    # Predictive saw every processed event.
    assert e.predictive.total_events == len(processed)
    # Locus Coeruleus raised arousal from the event density.
    assert e.locus_coeruleus.get_arousal() > 0.2
    # Neurorhythms modulated (and tracked) the learning rate.
    assert e.neurorhythms.total_synaptic_load > 0
    # Basal ganglia formed Q-values from passive observation.
    assert len(e.basal_ganglia.q_values) > 0
    # Anterior cingulate observed a decision round per processed event.
    assert (e.anterior_cingulate.total_agreements
            + e.anterior_cingulate.total_conflicts) == len(processed)
    # Cerebellum compiled reflex rules out of hippocampus memory.
    assert len(e.cerebellum.rules) > 0


def test_snapshot_surfaces_rich_module_output():
    """The snapshot.extra must expose the signals that used to be discarded."""
    e = _build_engine()
    snaps = _drive_pattern(e, cycles=30)
    processed = [s for s in snaps if "skipped" not in s.extra]
    last = processed[-1]
    for key in ("anomaly_threshold", "arousal", "cognitive_control",
                "conflict_level", "dopamine", "raw_prediction_count"):
        assert key in last.extra, f"missing extra key: {key}"
    # surprise invariant preserved across the richer pipeline.
    assert all(0.0 <= s.surprise <= 1.0 for s in processed)
    # At least one processed event produced an advisory PFC decision.
    assert any("decision" in s.extra for s in processed)


def test_default_mode_is_advisory_only():
    """In the default SHADOW mode the engine recommends but never executes."""
    e = _build_engine()
    snaps = _drive_pattern(e, cycles=30)
    for s in snaps:
        dec = s.extra.get("decision")
        if dec is not None:
            assert dec["stage"] == "OBSERVE"


# ---------------------------------------------------------------------------
# Closed cognitive-control loop
# ---------------------------------------------------------------------------

def test_cognitive_control_damps_ranking_confidence():
    """High ACC cognitive_control must lower the re-ranked confidence."""
    e = _build_engine()
    # Make the token decodable so the ranking helper can resolve it.
    e.observe({"entity_id": "light.bedroom_lamp", "new_state": "on"})
    tok = e.thalamus.token_to_id["bedroom.light.on"]
    pred = [(tok, 0.9, 0.8, "test", 20)]

    e.anterior_cingulate.cognitive_control = 0.0
    relaxed = e._rank_predictions(list(pred), bucket=0, room="bedroom")[0][2]
    e.anterior_cingulate.cognitive_control = 1.0
    conflicted = e._rank_predictions(list(pred), bucket=0, room="bedroom")[0][2]

    assert conflicted < relaxed, "cognitive_control did not damp confidence"


def test_entorhinal_anticipation_boosts_expected_room():
    """A token in the anticipated next room must be ranked higher."""
    e = _build_engine()
    e.observe({"entity_id": "light.bedroom_lamp", "new_state": "on"})
    tok = e.thalamus.token_to_id["bedroom.light.on"]
    pred = [(tok, 0.9, 0.8, "test", 20)]

    e._expected_next_room = None
    base_conf = e._rank_predictions(list(pred), bucket=0, room="bedroom")[0][2]
    e._expected_next_room = "bedroom"
    boosted_conf = e._rank_predictions(list(pred), bucket=0, room="bedroom")[0][2]

    assert boosted_conf > base_conf, "entorhinal anticipation did not boost"


def test_entorhinal_learns_room_transitions():
    """Confirmed spatial 'entered' tokens must populate the transition map."""
    e = KontinuumEngine()
    # Drive the spatial channel directly with confirmed entries.
    e._last_room = "kitchen"
    for room in ("bedroom", "kitchen", "bedroom"):
        # Simulate what observe() does with a confirmed 'entered' token.
        if e._last_room and e._last_room != room:
            e.entorhinal_cortex.observe_transition(e._last_room, room)
        e._last_room = room
    assert e.entorhinal_cortex.predict_next_room("kitchen") == "bedroom"


# ---------------------------------------------------------------------------
# Host-facing reward loop
# ---------------------------------------------------------------------------

def test_feedback_without_decision_is_noop():
    e = KontinuumEngine()
    assert e.feedback(True) is False


def test_feedback_reinforces_reward_modules():
    """feedback() must reach accumbens / basal ganglia / neurorhythms / ACC."""
    e = _build_engine()
    snaps = _drive_pattern(e, cycles=30)
    # A decision must be pending for the most recent qualifying event.
    assert e._last_decision_ctx is not None or any(
        "decision" in s.extra for s in snaps
    )
    # Drive one more switch.on so a fresh decision is remembered.
    e.observe({
        "entity_id": "switch.kitchen", "new_state": "on",
        "timestamp": datetime(2026, 6, 14, 7, 0, 0, tzinfo=timezone.utc),
    })
    assert e._last_decision_ctx is not None, "no decision remembered to reinforce"

    errors_before = e.anterior_cingulate.total_errors + e.anterior_cingulate.total_correct
    assert e.feedback(True) is True
    # Reward modules learned something.
    assert len(e.nucleus_accumbens.values) > 0
    assert (e.anterior_cingulate.total_errors
            + e.anterior_cingulate.total_correct) == errors_before + 1
    # Decision is consumed (one feedback per decision).
    assert e._last_decision_ctx is None
    assert e.feedback(True) is False


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_engine_state_round_trips():
    e = _build_engine()
    _drive_pattern(e, cycles=20)
    blob = e.to_dict()

    e2 = KontinuumEngine()
    e2.from_dict(blob)

    assert e2.tick_count == e.tick_count
    assert e2.hippocampus.total_events == e.hippocampus.total_events
    assert e2._last_room == e._last_room
    assert len(e2.predictive.surprise_history) == len(e.predictive.surprise_history)
    assert len(e2.cerebellum.rules) == len(e.cerebellum.rules)


def test_predictive_surprise_history_persists():
    """B: the adaptive anomaly threshold must survive a save/load cycle."""
    p = PredictiveProcessing()
    for i in range(40):
        # Alternate predicted / unpredicted to build a non-trivial spread.
        p.compute_surprise(i, predictions=[])
    assert len(p.surprise_history) == 40
    threshold_before = p.anomaly_threshold()
    assert threshold_before != ANOMALY_DEFAULT_THRESHOLD  # adaptive engaged

    p2 = PredictiveProcessing()
    p2.from_dict(p.to_dict())
    assert len(p2.surprise_history) == 40
    assert p2.anomaly_threshold() == threshold_before
