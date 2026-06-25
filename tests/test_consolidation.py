"""Sleep-consolidation is now wired and correct.

Before this, SleepConsolidation.consolidate() was never called by the engine and
referenced attributes that don't exist (hippocampus.ngram_counts,
cerebellum.extract_rules, basal_ganglia.q_table) — so it would have crashed if
invoked. These tests lock in that it runs, keeps the hippocampus totals
consistent, resets synaptic load, and is actually triggered after a quiet spell.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

from kontinuum_core import KontinuumEngine


def _warm(engine, cycles=60):
    engine.register_entity("switch.k", ha_area="kitchen", domain="switch")
    engine.register_entity("light.b", ha_area="bedroom", domain="light")
    base = datetime(2026, 1, 1, 19, 0, tzinfo=timezone.utc)
    n = 0
    for _ in range(cycles):
        for eid, st in [("switch.k", "on"), ("light.b", "on"),
                        ("switch.k", "off"), ("light.b", "off")]:
            engine.observe({"entity_id": eid, "new_state": st,
                            "timestamp": base + timedelta(minutes=n)})
            n += 1
    return engine


def test_consolidate_runs_and_keeps_totals_consistent():
    e = _warm(KontinuumEngine())
    load_before = e.neurorhythms.total_synaptic_load
    assert load_before > 0

    stats = e.sleep_consolidation.consolidate(
        e.hippocampus, e.cerebellum, e.basal_ganglia, e.neurorhythms)

    # Phase 1 must keep totals == sum of per-token transition weights.
    for bucket, ngrams in e.hippocampus.transitions.items():
        for ng, toks in ngrams.items():
            assert abs(e.hippocampus.totals[bucket][ng] - sum(toks.values())) < 1e-6

    # Phase 5 homeostasis must reset the accumulated synaptic load (the value
    # that previously grew forever because consolidation never ran).
    assert e.neurorhythms.total_synaptic_load == 0.0
    assert e.sleep_consolidation.total_consolidations == 1
    assert stats["patterns_reinforced"] >= 0 and "homeostasis_factor" in stats


def test_consolidate_does_not_break_prediction():
    """After consolidation the engine must still produce valid predictions."""
    e = _warm(KontinuumEngine())
    e.sleep_consolidation.consolidate(
        e.hippocampus, e.cerebellum, e.basal_ganglia, e.neurorhythms)
    snap = e.observe({"entity_id": "switch.k", "new_state": "on",
                      "timestamp": datetime(2026, 2, 1, 19, 0, tzinfo=timezone.utc)})
    assert 0.0 <= snap.surprise <= 1.0
    assert len(snap.predictions) <= 5


def test_engine_triggers_consolidation_after_a_quiet_period():
    e = _warm(KontinuumEngine())
    # Force the "quiet spell" preconditions, then one more event must run it.
    e.sleep_consolidation.last_consolidation_ts = 0.0
    e.sleep_consolidation.events_since_last = 60
    e._last_event_ts = time.time() - 3600  # >30 min since last event
    before = e.sleep_consolidation.total_consolidations

    e.observe({"entity_id": "switch.k", "new_state": "on",
               "timestamp": datetime(2026, 3, 1, 19, 0, tzinfo=timezone.utc)})

    assert e.sleep_consolidation.total_consolidations == before + 1


def test_tick_consolidates_during_idle_without_any_event():
    """The core fix: a quiet spell must consolidate via the host heartbeat,
    with no further event arriving (the old path could only fire on an event,
    which by definition meant it was *not* quiet)."""
    e = _warm(KontinuumEngine())
    e.sleep_consolidation.last_consolidation_ts = 0.0
    e.sleep_consolidation.events_since_last = 60
    e._last_event_ts = time.time() - 3600  # >30 min since last event
    before = e.sleep_consolidation.total_consolidations

    stats = e.tick()

    assert stats is not None
    assert e.sleep_consolidation.total_consolidations == before + 1


def test_tick_is_a_noop_when_not_quiet():
    e = _warm(KontinuumEngine())
    e.sleep_consolidation.events_since_last = 60
    e._last_event_ts = time.time()  # busy: event just happened
    before = e.sleep_consolidation.total_consolidations

    assert e.tick() is None
    assert e.sleep_consolidation.total_consolidations == before


def test_force_consolidation_bypasses_all_gates():
    e = _warm(KontinuumEngine())
    # No quiet spell, fresh cooldown, but force must still run.
    e.sleep_consolidation.last_consolidation_ts = time.time()
    e.sleep_consolidation.events_since_last = 0
    e._last_event_ts = time.time()
    before = e.sleep_consolidation.total_consolidations

    stats = e.force_consolidation()

    assert "homeostasis_factor" in stats
    assert e.sleep_consolidation.total_consolidations == before + 1
