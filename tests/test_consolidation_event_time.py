"""Sleep-consolidation now computes in EVENT time, not wall-clock.

Before this fix ``should_consolidate`` read ``time.time()`` internally, so a
burst replay (thousands of events processed within milliseconds of wall-clock)
never saw a quiet window and consolidated **zero** times — invisible and
untestable in replay/backtest. These tests lock in the event-time behaviour:

a) *Replay fires*: accelerated event time with simulated quiet spells → fires.
b) *Fallback fires*: sustained load without any quiet window → fires once after
   ``MAX_EVENTS_BEFORE_FORCED``, cooldown respected.
c) *Busy phase stays silent*: dense events, no quiet, under the fallback → 0.
d) *Persistence roundtrip*: to_dict → from_dict keeps counters/timestamps.

The gates are tested on ``should_consolidate`` directly with an injected event
clock, so no real ``sleep`` and no wall-clock dependency is involved.
"""
from __future__ import annotations

from kontinuum_core.sleep_consolidation import (
    SleepConsolidation,
    QUIET_THRESHOLD,
    MIN_EVENTS_FOR_CONSOLIDATION,
    COOLDOWN_SECONDS,
    MAX_EVENTS_BEFORE_FORCED,
    MAX_INTERVAL_SECONDS,
)


def test_replay_with_quiet_spells_fires_in_event_time():
    """3000 events on an accelerated EVENT clock with periodic quiet spells
    must consolidate (> 0), unlike the old wall-clock burst replay (== 0)."""
    s = SleepConsolidation()
    vt = 1_000_000.0          # virtual event time (epoch seconds)
    last_event_ts = 0.0
    EVENT_SPACING = 60        # one event per minute during the day
    fired = 0

    for i in range(3000):
        s.observe_event()
        if s.should_consolidate(vt, last_event_ts):
            s.consolidate_stub = None  # (no modules here; gate test only)
            s.last_consolidation_ts = vt
            s.events_since_last = 0
            s.total_consolidations += 1
            fired += 1
        # event time advances; every ~90 events a 40-min quiet spell (night)
        vt += EVENT_SPACING
        last_event_ts = vt
        if i % 90 == 89:
            vt += QUIET_THRESHOLD + 600

    assert fired > 0
    assert s.total_consolidations > 0


def test_fallback_fires_under_sustained_load_without_quiet():
    """No quiet window ever, but after MAX_EVENTS_BEFORE_FORCED events exactly
    one consolidation is forced; the cooldown is still respected afterwards."""
    s = SleepConsolidation()
    vt = 2_000_000.0
    last_event_ts = 0.0
    fired = 0

    # Feed just enough events to cross the forced-event threshold, all with a
    # tiny event-time gap (< QUIET_THRESHOLD) so the quiet path never applies.
    for _ in range(MAX_EVENTS_BEFORE_FORCED):
        s.observe_event()
        # last_event_ts is always "just now" in event time → never quiet
        last_event_ts = vt
        if s.should_consolidate(vt, last_event_ts):
            s.last_consolidation_ts = vt
            s.events_since_last = 0
            s.total_consolidations += 1
            fired += 1
        vt += 1  # 1 second between events → dense, no quiet window

    assert fired == 1

    # Cooldown holds: immediately after, even at the fallback we do not re-fire.
    for _ in range(MAX_EVENTS_BEFORE_FORCED):
        s.observe_event()
        last_event_ts = vt
        assert not s.should_consolidate(vt, last_event_ts)
        vt += 1  # still inside COOLDOWN_SECONDS


def test_busy_phase_under_fallback_stays_silent():
    """Dense events, no quiet spell and below the fallback threshold → the
    consolidator must not fight live use: zero consolidations."""
    s = SleepConsolidation()
    vt = 3_000_000.0
    fired = 0

    n = MAX_EVENTS_BEFORE_FORCED - 1  # strictly below the forced-event limit
    assert n >= MIN_EVENTS_FOR_CONSOLIDATION
    for _ in range(n):
        s.observe_event()
        last_event_ts = vt  # always "just now" → never quiet
        if s.should_consolidate(vt, last_event_ts):
            fired += 1
        vt += 1  # 1 s apart, well under QUIET_THRESHOLD and MAX_INTERVAL

    assert fired == 0
    assert s.total_consolidations == 0


def test_interval_fallback_fires_after_max_interval_seconds():
    """Even without a quiet window, once MAX_INTERVAL_SECONDS of event time have
    passed since the last consolidation, a run is forced (cooldown permitting)."""
    s = SleepConsolidation()
    t0 = 4_000_000.0
    s.last_consolidation_ts = t0
    s.events_since_last = MIN_EVENTS_FOR_CONSOLIDATION  # enough events
    # Just under the interval: not yet (busy, no quiet).
    assert not s.should_consolidate(t0 + MAX_INTERVAL_SECONDS - 1, t0 + MAX_INTERVAL_SECONDS - 1)
    # At/after the interval: forced.
    assert s.should_consolidate(t0 + MAX_INTERVAL_SECONDS, t0 + MAX_INTERVAL_SECONDS)


def test_cooldown_blocks_all_paths():
    s = SleepConsolidation()
    t0 = 5_000_000.0
    s.last_consolidation_ts = t0
    s.events_since_last = MAX_EVENTS_BEFORE_FORCED  # would force, but...
    # Inside cooldown: nothing fires regardless of event count.
    assert not s.should_consolidate(t0 + COOLDOWN_SECONDS - 1, t0)
    # Cooldown elapsed + a quiet window → fires.
    assert s.should_consolidate(t0 + COOLDOWN_SECONDS + 1, t0)


def test_persistence_roundtrip_keeps_counters_and_timestamps():
    s = SleepConsolidation()
    s.last_consolidation_ts = 12345.678
    s.events_since_last = 17
    s.total_consolidations = 42
    s.total_dream_connections = 9
    s.last_homeostasis_factor = 0.97

    restored = SleepConsolidation()
    restored.from_dict(s.to_dict())

    assert restored.last_consolidation_ts == s.last_consolidation_ts
    assert restored.events_since_last == s.events_since_last
    assert restored.total_consolidations == s.total_consolidations
    assert restored.total_dream_connections == s.total_dream_connections
    assert restored.last_homeostasis_factor == s.last_homeostasis_factor


def test_old_persisted_state_loads_without_new_fields():
    """Backward compatibility: a dict from an older version (no new fields)
    must load via .get() defaults without raising."""
    s = SleepConsolidation()
    s.from_dict({"last_consolidation_ts": 111.0, "total_consolidations": 3})
    assert s.last_consolidation_ts == 111.0
    assert s.total_consolidations == 3
    assert s.events_since_last == 0  # default
