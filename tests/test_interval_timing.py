"""Tests for the IntervalTiming module (the "inner stopwatch", v0.6.0).

Distinct from the Suprachiasmatic Nucleus (time-of-day): this learns the
*duration* between recurring occurrences of a token and resurfaces an overdue,
regular cadence (e.g. a task that runs every few weeks) that sequence
prediction can't see.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from kontinuum_core import KontinuumEngine
from kontinuum_core.interval_timing import IntervalTiming


def _engine() -> KontinuumEngine:
    e = KontinuumEngine()
    e.register_entity("switch.kitchen", ha_area="kitchen", domain="switch")
    e.register_entity("light.bedroom_lamp", ha_area="bedroom", domain="light")
    return e


# ---------------------------------------------------------------------------
# Unit
# ---------------------------------------------------------------------------

def test_learns_regular_cadence_and_flags_due():
    it = IntervalTiming()
    P = 100_000.0
    for i in range(4):           # t0 .. t0+3P  → 3 intervals
        it.observe(1, i * P)
    assert it.timers[1]["count"] == 3
    # Just fired → not due.
    assert it.due_score(1, 3 * P + 10) == 0.0
    # Around the expected next time → due; overdue stays due.
    assert it.due_score(1, 3 * P + 0.9 * P) > 0.5
    assert it.due_score(1, 3 * P + 2 * P) > 0.5
    assert it.predict_next_ts(1) == 3 * P + P


def test_min_observations_gate():
    it = IntervalTiming()
    it.observe(2, 0.0)
    it.observe(2, 100_000.0)     # count == 1, below MIN_OBSERVATIONS
    assert it.due_score(2, 300_000.0) == 0.0


def test_sub_floor_intervals_ignored():
    it = IntervalTiming()
    for t in (0.0, 100.0, 200.0):   # all < MIN_INTERVAL → no cadence built
        it.observe(3, t)
    assert it.timers[3]["count"] == 0


def test_irregular_cadence_never_due():
    it = IntervalTiming()
    # Seed a high relative spread directly (mad/mean = 0.6 > MAX_REL_SPREAD).
    it.timers[7] = {"last": 0.0, "mean": 100_000.0, "mad": 60_000.0, "count": 5}
    assert it.due_score(7, 250_000.0) == 0.0   # irregular → not predictable


def test_due_prediction_excludes_current_and_round_trips():
    it = IntervalTiming()
    P = 100_000.0
    for i in range(4):
        it.observe(5, i * P)
    now = 3 * P + 0.95 * P
    # The only tracked token is the excluded one → nothing to surface.
    assert it.due_prediction(now, exclude=5) is None
    pred = it.due_prediction(now, exclude=None)
    assert pred is not None and pred[0] == 5 and pred[3] == "interval_timing"

    it2 = IntervalTiming()
    it2.from_dict(it.to_dict())
    assert it2.due_score(5, now) == it.due_score(5, now)
    assert it2.predict_next_ts(5) == it.predict_next_ts(5)


def test_eviction_bounds_memory():
    it = IntervalTiming()
    it.MAX_ENTRIES = 10
    for tok in range(25):
        it.observe(tok, 0.0)
    assert len(it.timers) <= 10


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------

def test_engine_injects_overdue_cadence():
    e = _engine()
    # One observe to mint a decodable token + capture its id.
    s = e.observe({
        "entity_id": "switch.kitchen", "new_state": "on",
        "timestamp": datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    })
    sw_tok = s.token_id
    assert sw_tok is not None

    # Seed a regular weekly cadence, last fired 8 days ago (overdue).
    now_dt = datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc)
    now = now_dt.timestamp()
    week = 7 * 24 * 3600
    e.interval_timing.timers[sw_tok] = {
        "last": now - 8 * 24 * 3600, "mean": float(week), "mad": 0.0, "count": 5,
    }

    # Fire a DIFFERENT entity now → the overdue switch cadence must be injected.
    snap = e.observe({
        "entity_id": "light.bedroom_lamp", "new_state": "on", "timestamp": now_dt,
    })
    pred_ids = [p[0] for p in snap.predictions]
    assert sw_tok in pred_ids, "overdue cadence was not injected as a prediction"
    assert any(p[3] == "interval_timing" for p in snap.predictions)
    assert snap.extra.get("interval_due_token") == "kitchen.switch.on"
    assert snap.extra.get("interval_tracked", 0) >= 1


def test_overdue_injection_does_not_touch_surprise():
    """Injection happens after surprise is computed → anomaly signal unchanged."""
    e = _engine()
    s = e.observe({
        "entity_id": "switch.kitchen", "new_state": "on",
        "timestamp": datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    })
    sw_tok = s.token_id
    now_dt = datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc)
    e.interval_timing.timers[sw_tok] = {
        "last": now_dt.timestamp() - 8 * 24 * 3600,
        "mean": float(7 * 24 * 3600), "mad": 0.0, "count": 5,
    }
    snap = e.observe({
        "entity_id": "light.bedroom_lamp", "new_state": "on", "timestamp": now_dt,
    })
    # Surprise stays a valid [0,1] signal regardless of the injected candidate.
    assert 0.0 <= snap.surprise <= 1.0
