from __future__ import annotations

import json
import os
import statistics
import time
from datetime import datetime, timedelta, timezone

import pytest

from kontinuum_core import (
    KontinuumEngine,
    build_llm_context,
    extract_json,
    normalize_proposal,
    render_llm_context,
)

ROOMS = [
    "bedroom", "bathroom", "kitchen", "living", "hallway", "office",
    "garage", "outdoor", "utility", "kidsroom", "guestroom", "dining",
]
DOMAINS = ["binary_sensor", "light", "switch", "sensor", "climate", "media_player"]
STATES = {
    "binary_sensor": ["on", "off"],
    "light": ["on", "off"],
    "switch": ["on", "off"],
    "sensor": ["18.5", "19.0", "21.0", "23.5", "45", "52"],
    "climate": ["heat", "idle", "cool"],
    "media_player": ["playing", "paused", "off"],
}


def _entity(domain: str, room: str, idx: int) -> str:
    if domain == "binary_sensor":
        return f"binary_sensor.motion_{room}_{idx}"
    if domain == "sensor":
        return f"sensor.{room}_temperature_{idx}"
    return f"{domain}.{room}_{idx}"


def _register_house(engine: KontinuumEngine, per_room: int = 2) -> list[str]:
    entities: list[str] = []
    for room in ROOMS:
        for domain in DOMAINS:
            for idx in range(per_room):
                entity_id = _entity(domain, room, idx)
                engine.register_entity(entity_id, ha_area=room, domain=domain)
                entities.append(entity_id)
    return entities


def _state_for(entity_id: str, i: int) -> str:
    domain = entity_id.split(".", 1)[0]
    values = STATES[domain]
    return values[(i + len(entity_id)) % len(values)]


def _event(entity_id: str, i: int, start: datetime) -> dict:
    return {
        "entity_id": entity_id,
        "new_state": _state_for(entity_id, i),
        "old_state": _state_for(entity_id, i + 1),
        "timestamp": start + timedelta(seconds=i * 17),
    }


def test_high_volume_mixed_home_load_keeps_snapshots_bounded_and_persistent():
    engine = KontinuumEngine()
    entities = _register_house(engine, per_room=3)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)

    snapshots = []
    seen_per_entity = {entity_id: 0 for entity_id in entities}
    for i in range(3000):
        entity_id = entities[(i * 37) % len(entities)]
        seen_per_entity[entity_id] += 1
        snap = engine.observe(_event(entity_id, seen_per_entity[entity_id], start))
        snapshots.append(snap)
        assert snap.tick_count == i + 1
        assert 0.0 <= snap.surprise <= 1.0
        assert isinstance(snap.anomaly, bool)
        assert len(snap.predictions) <= 6
        if "skipped" not in snap.extra:
            assert snap.extra["raw_prediction_count"] <= 5
            assert 0.0 <= snap.extra["anomaly_threshold"] <= 1.0
            assert 0.0 <= snap.extra["arousal"] <= 1.0
            assert 0.0 <= snap.extra["cognitive_control"] <= 1.0
            assert 0.0 <= snap.extra["conflict_level"] <= 1.0

    assert engine.tick_count == 3000
    assert engine.hippocampus.total_events > 2500
    assert engine._learning_state() == "stable"
    assert len(engine.thalamus.token_to_id) <= engine.thalamus.MAX_VOCAB_SIZE
    assert any(s.anomaly for s in snapshots)
    assert any(s.predictions for s in snapshots[200:])

    restored = KontinuumEngine()
    restored.from_dict(json.loads(json.dumps(engine.to_dict())))
    assert restored.tick_count == engine.tick_count
    assert restored.hippocampus.total_events == engine.hippocampus.total_events
    assert restored.thalamus.token_to_id == engine.thalamus.token_to_id


def test_raspberry_pi_4_performance_budget_for_representative_core_workload():
    engine = KontinuumEngine()
    entities = _register_house(engine, per_room=2)
    start = datetime(2026, 2, 1, tzinfo=timezone.utc)
    total_events = 1500

    t0 = time.perf_counter()
    for i in range(total_events):
        engine.observe(_event(entities[i % len(entities)], i, start))
    elapsed = time.perf_counter() - t0
    events_per_second = total_events / elapsed

    assert engine.tick_count == total_events
    assert len(engine.thalamus.token_to_id) > 100

    if os.getenv("KONTINUUM_RPI4_PERF") == "1":
        assert events_per_second >= 250, (
            f"Raspberry Pi 4 budget missed: {events_per_second:.1f} events/s")
    else:
        assert events_per_second >= 50, (
            f"portable CI smoke budget missed: {events_per_second:.1f} events/s")


def test_codex_style_llm_round_trip_can_read_context_and_return_actionable_json():
    engine = KontinuumEngine()
    entities = _register_house(engine, per_room=1)
    start = datetime(2026, 3, 1, tzinfo=timezone.utc)
    for i in range(240):
        engine.observe(_event(entities[i % len(entities)], i, start))

    context = build_llm_context(engine, top_k=5)
    prompt_payload = render_llm_context(context)
    assert "KONTINUUM home-brain state" in prompt_payload
    assert "surprise/confidence are 0–1" in prompt_payload
    assert context["schema_version"] == 1
    assert len(context["prediction"]["expected_next"]) <= 5

    codex_reply = f"""
    Ich habe den Core-Kontext gelesen und schlage nur eine Shadow-Aktion vor.
    ```json
    {{
      "agent": "chatgpt-codex",
      "action": "explain_anomaly",
      "entity_id": "{entities[0]}",
      "reason": "High surprise needs operator-readable explanation before acting.",
      "priority": "87.6",
      "veto": "false"
    }}
    ```
    """

    parsed = extract_json(codex_reply)
    proposal = normalize_proposal(codex_reply)
    assert parsed["agent"] == "chatgpt-codex"
    assert proposal == {
        "agent": "chatgpt-codex",
        "action": "explain_anomaly",
        "entity_id": entities[0],
        "reason": "High surprise needs operator-readable explanation before acting.",
        "priority": 88,
        "veto": False,
        "valid": True,
    }


@pytest.mark.parametrize("events", [1, 10, 100, 1000])
def test_learning_state_boundaries_under_monotonic_event_growth(events):
    engine = KontinuumEngine()
    entities = _register_house(engine, per_room=1)
    start = datetime(2026, 4, 1, tzinfo=timezone.utc)
    for i in range(events):
        engine.observe(_event(entities[i % len(entities)], i, start))

    learned_events = engine.hippocampus.total_events
    if learned_events < 100:
        assert engine._learning_state() == "cold_start"
    elif learned_events < 1000 or engine.hippocampus.accuracy < 0.3:
        assert engine._learning_state() == "learning"
    else:
        assert engine._learning_state() in {"learning", "stable"}


def test_repeated_invalid_and_filtered_events_do_not_pollute_memory():
    engine = KontinuumEngine()
    for i in range(500):
        missing = engine.observe({"new_state": "on"})
        ignored = engine.observe({
            "entity_id": f"sensor.kontinuum_internal_{i}",
            "new_state": str(i),
            "timestamp": datetime(2026, 5, 1, tzinfo=timezone.utc),
        })
        assert missing.extra == {"skipped": "no_entity_or_state"}
        assert ignored.extra == {"skipped": "filtered"}

    assert engine.tick_count == 1000
    assert engine.hippocampus.total_events == 0
    assert engine.thalamus.token_to_id == {}


def test_surprise_distribution_remains_finite_during_burst_then_idle_pattern():
    engine = KontinuumEngine()
    entities = _register_house(engine, per_room=1)
    start = datetime(2026, 6, 1, tzinfo=timezone.utc)
    surprises = []

    for burst in range(20):
        base = start + timedelta(hours=burst * 6)
        for j in range(60):
            snap = engine.observe(_event(entities[(burst + j) % len(entities)], burst * 60 + j, base))
            surprises.append(snap.surprise)
        idle_snap = engine.observe({
            "entity_id": "sensor.living_temperature_0",
            "new_state": "20.0",
            "old_state": "19.5",
            "timestamp": base + timedelta(hours=5, minutes=59),
        })
        surprises.append(idle_snap.surprise)

    assert all(0.0 <= s <= 1.0 for s in surprises)
    assert statistics.fmean(surprises[-100:]) < 0.9
