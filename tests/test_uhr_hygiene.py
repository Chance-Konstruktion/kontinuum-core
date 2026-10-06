"""Tests for the clock hygiene (issue #2, §7 step 6, part 1) and the
measurement findings 1+2 (MR !4, note 13693) — TRIMMED onto !8's clock
per review 03:19: the Locus-Coeruleus part of the original MR is now
main's (``test_clock.py`` from MR !8 covers it — "Entscheidung:
wirf deinen eigenen Locus-Coeruleus-Teil raus"); what remains here is
the Entorhinal prune in event time and the two Messstand findings:
the raw hippocampus list rides the snapshot (extra["raw_predictions"]
as "Hippocampus pur"), the "Engine vor Ranking" line carries REAL
confidence (not the fixed 1.0 that turned its ECE into a plain error
rate), and both raw lines empty themselves instead of going stale.
"""
from __future__ import annotations

from datetime import datetime, timezone

from kontinuum_core import KontinuumEngine
from kontinuum_core.entorhinal_cortex import EntorhinalCortex
from benchmarks.spur.messstand import _Rohspion

VORHER = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def _observe_engine(e: KontinuumEngine, entity_id: str, state: str,
                    ts: datetime):
    e.thalamus.entity_last_token.pop(entity_id, None)
    return e.observe({
        "entity_id": entity_id,
        "new_state": state,
        "old_state": None,
        "timestamp": ts,
    })


def test_prune_cadence_runs_in_event_time():
    """Two events 25 h apart in EVENT time (prune interval: 24 h): the
    second event triggers the maintenance — and the prune stamp carries
    the EVENT time, not the wall clock."""
    e = KontinuumEngine()
    e.register_entity("sensor.a", ha_area="keller", domain="switch")  # switch: on/off ist die gueltige Ladder
    t0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 1, 2, 13, 30, tzinfo=timezone.utc)
    _observe_engine(e, "sensor.a", "on", t0)
    _observe_engine(e, "sensor.a", "off", t1)
    stamp = e.entorhinal_cortex.last_prune_ts
    assert stamp == t1.timestamp()


def test_engine_prune_gate_rides_the_explicit_clock():
    """Review 03:19 («Für den Rückfall nimm self._clock() statt der
    Wanduhr»): the engine hands its EXPLICIT clock to the entorhinal —
    a hand-called prune without a timestamp stamps on the CONFIG clock
    (1000.0), not the wall clock. Events WITH timestamps keep stamping
    in event time (the engine path from the test above)."""
    uhr_wert = 1000.0
    e = KontinuumEngine(config={"clock": lambda: uhr_wert})
    e.register_entity("sensor.a", ha_area="keller", domain="switch")  # switch: on/off ist die gueltige Ladder
    _observe_engine(e, "sensor.a", "on",
                    datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc))
    _observe_engine(e, "sensor.a", "off",
                    datetime(2026, 1, 2, 13, 30, tzinfo=timezone.utc))
    # beide Ereignisse tragen Zeitstempel ⇒ der Takt stempelt Ereigniszeit:
    assert e.entorhinal_cortex.last_prune_ts == \
        datetime(2026, 1, 2, 13, 30, tzinfo=timezone.utc).timestamp()
    # der Rückfall einer Hand-rufenden prune ist DIE UHR, nicht die Wanduhr:
    e.entorhinal_cortex.prune_old_transitions()
    assert e.entorhinal_cortex.last_prune_ts == uhr_wert


def test_prune_without_event_time_falls_back_to_the_clock():
    """Direct call without a timestamp: the injected clock stamps —
    !8's fallback pattern (``self._clock()`` statt der Wanduhr, review
    03:19) also for the prune gate. A bare EntorhinalCortex() keeps
    the wall clock (live path unchanged)."""
    ent = EntorhinalCortex(clock=lambda: 1234.5)
    ent.prune_old_transitions()
    assert ent.last_prune_ts == 1234.5

    bar = EntorhinalCortex()
    before = bar.last_prune_ts
    bar.prune_old_transitions()
    assert bar.last_prune_ts > before  # die Wanduhr als Standard
    assert bar.last_prune_ts > VORHER.timestamp()  # ...die JETZIGE


def test_raw_predictions_ride_the_snapshot():
    """Befund 1, first half: the RAW hippocampus list rides the snapshot
    as (token, prob, conf) — the "Hippocampus pur" line needs no spy."""
    e = KontinuumEngine()
    e.register_entity("sensor.a", ha_area="keller", domain="switch")  # switch: on/off ist die gueltige Ladder
    for n in range(14):
        snap = _observe_engine(e, "sensor.a", "on" if n % 2 else "off",
                               datetime(2026, 1, 1, 12, n // 2, n % 2, tzinfo=timezone.utc))
    roh = (snap.extra or {}).get("raw_predictions")
    assert isinstance(roh, list)
    if roh:  # a learned pattern exists after six alternating events
        for eintrag in roh:
            assert isinstance(eintrag[0], str) and eintrag[0]
            assert 0.0 <= float(eintrag[1]) <= 1.0
            assert 0.0 <= float(eintrag[2]) <= 1.0


def test_spion_carries_real_confidence_and_empties():
    """Befund 2: the spy's confidence comes from the tuple (place 3) —
    it is NOT the fixed 1.0 that made the ECE a plain error rate.
    Befund 1, second half: leeren() empties the line (no stale carry)."""
    e = KontinuumEngine()
    e.register_entity("sensor.a", ha_area="keller", domain="switch")  # switch: on/off ist die gueltige Ladder
    spion = _Rohspion(e)
    zeilen = []
    for n in range(14):
        _observe_engine(e, "sensor.a", "on" if n % 2 else "off",
                        datetime(2026, 1, 1, 12, n // 2, n % 2, tzinfo=timezone.utc))
        zeilen.append(spion.vorhersage())
    gefuellt = [z for z in zeilen if z]
    assert gefuellt, "der Spion fing nie (kein Ranking-Eingang getroffen)"
    konfidenzen = [conf for z in gefuellt for _, conf in z]
    assert min(konfidenzen) < 1.0, "die feste 1.0-Luege lebt noch"
    spion.leeren()
    assert spion.vorhersage() == []
