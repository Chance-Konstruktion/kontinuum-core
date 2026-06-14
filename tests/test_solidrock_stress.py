"""Solidrock-style integration and stress tests for the public core API.

These tests intentionally exercise realistic, noisy host/LLM boundaries without
changing production code: long routines, filtered noise, persistence restarts,
and a Codex/ChatGPT-like proposal loop.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from kontinuum_core import (
    KontinuumEngine,
    build_llm_context,
    extract_json,
    normalize_proposal,
    render_llm_context,
)
from kontinuum_core.thalamus import Thalamus


def _register_house(engine: KontinuumEngine) -> None:
    entities = [
        ("binary_sensor.hallway_motion", "hallway", "binary_sensor", "motion", "Hallway motion"),
        ("light.kitchen_ceiling", "kitchen", "light", "", "Kitchen ceiling"),
        ("sensor.kitchen_temperature", "kitchen", "sensor", "temperature", "Kitchen temperature"),
        ("sensor.livingroom_illuminance", "livingroom", "sensor", "illuminance", "Livingroom lux"),
        ("climate.livingroom", "livingroom", "climate", "", "Livingroom climate"),
        ("binary_sensor.entrance_door", "entrance", "binary_sensor", "door", "Entrance door"),
        ("device_tracker.alex_phone", "hallway", "device_tracker", "", "Alex phone"),
        ("media_player.livingroom_tv", "livingroom", "media_player", "", "Livingroom TV"),
    ]
    for entity_id, area, domain, device_class, name in entities:
        engine.register_entity(
            entity_id,
            ha_area=area,
            domain=domain,
            device_class=device_class,
            friendly_name=name,
        )


def _routine_events(start: datetime, cycles: int) -> list[dict]:
    pattern = [
        ("device_tracker.alex_phone", "home", 0),
        ("binary_sensor.entrance_door", "on", 1),
        ("binary_sensor.hallway_motion", "on", 2),
        ("light.kitchen_ceiling", "on", 3),
        ("sensor.kitchen_temperature", "21.5", 4),
        ("sensor.livingroom_illuminance", "8", 5),
        ("climate.livingroom", "heat", 6),
        ("media_player.livingroom_tv", "playing", 7),
        ("media_player.livingroom_tv", "off", 35),
        ("light.kitchen_ceiling", "off", 36),
        ("binary_sensor.hallway_motion", "off", 37),
        ("binary_sensor.entrance_door", "off", 38),
    ]
    events = []
    for cycle in range(cycles):
        day = cycle // 2
        evening_offset = cycle % 2 * 120
        base = start + timedelta(days=day, minutes=evening_offset)
        for entity_id, state, minute in pattern:
            events.append({
                "entity_id": entity_id,
                "new_state": state,
                "timestamp": base + timedelta(minutes=minute),
            })
    return events


def test_engine_survives_long_noisy_mixed_home_stream_and_keeps_outputs_bounded():
    engine = KontinuumEngine()
    _register_house(engine)
    start = datetime(2026, 1, 5, 18, 0, tzinfo=timezone.utc)

    last = None
    for i, event in enumerate(_routine_events(start, cycles=140)):
        last = engine.observe(event)
        if i % 9 == 0:
            skipped = engine.observe({"entity_id": "sensor.unregistered_noise", "new_state": "123"})
            assert skipped.extra == {"skipped": "filtered"}
        if i % 13 == 0:
            duplicate = engine.observe({**event, "timestamp": event["timestamp"] + timedelta(seconds=1)})
            assert duplicate.extra == {"skipped": "filtered"}

    assert last is not None
    assert engine.tick_count > engine.hippocampus.total_events
    assert engine.hippocampus.total_events >= 1000
    assert len(engine.hippocampus.buffer) <= 30
    assert len(engine.thalamus.token_to_id) <= Thalamus.MAX_VOCAB_SIZE
    assert len(last.predictions) <= 5
    assert 0.0 <= last.surprise <= 1.0
    assert 0.0 <= last.extra["anomaly_threshold"] <= 1.0
    assert last.extra["raw_prediction_count"] <= 5


def test_restart_roundtrip_preserves_predictions_for_continued_learning():
    source = KontinuumEngine()
    _register_house(source)
    start = datetime(2026, 2, 1, 18, 0, tzinfo=timezone.utc)
    for event in _routine_events(start, cycles=80):
        source.observe(event)

    restored = KontinuumEngine()
    restored.from_dict(json.loads(json.dumps(source.to_dict())))

    probe_time = start + timedelta(days=50)
    probe = {"entity_id": "device_tracker.alex_phone", "new_state": "not_home", "timestamp": probe_time}
    source_snapshot = source.observe(probe)
    restored_snapshot = restored.observe(probe)

    assert restored.tick_count == source.tick_count
    assert restored.hippocampus.total_events == source.hippocampus.total_events
    assert restored_snapshot.token == source_snapshot.token
    assert [p[:4] for p in restored_snapshot.predictions] == [p[:4] for p in source_snapshot.predictions]


def test_chatgpt_codex_style_context_prompt_and_proposal_roundtrip():
    engine = KontinuumEngine()
    _register_house(engine)
    start = datetime(2026, 3, 10, 18, 0, tzinfo=timezone.utc)
    for event in _routine_events(start, cycles=35):
        engine.observe(event)

    context = build_llm_context(engine, top_k=4)
    prompt_payload = render_llm_context(context)
    assert "KONTINUUM home-brain state" in prompt_payload
    assert "Expected next:" in prompt_payload
    assert context["schema_version"] == 1
    assert len(context["prediction"]["expected_next"]) <= 4

    codex_reply = """
    I inspected the KONTINUUM context. I will stay in SHADOW mode and return
    only the machine-readable proposal for the core:

    ```json
    {
      "agent": "chatgpt-codex",
      "action": "light.turn_on",
      "entity": "light.kitchen_ceiling",
      "reason": "Low illuminance and learned evening routine suggest kitchen light next.",
      "priority": "82.6",
      "veto": "false"
    }
    ```
    """
    extracted = extract_json(codex_reply)
    proposal = normalize_proposal(codex_reply)

    assert extracted["agent"] == "chatgpt-codex"
    assert proposal == {
        "agent": "chatgpt-codex",
        "action": "light.turn_on",
        "entity_id": "light.kitchen_ceiling",
        "reason": "Low illuminance and learned evening routine suggest kitchen light next.",
        "priority": 83,
        "veto": False,
        "valid": True,
    }


def test_llm_parser_ignores_decoy_json_and_uses_first_balanced_payload():
    reply = """
    Thought process omitted. Example schema: {"action": "do.not_use", "priority": 1}
    Final answer:
    {"action": "climate.set_temperature", "entity_id": "climate.livingroom", "priority": 77}
    """
    assert extract_json(reply) == {"action": "do.not_use", "priority": 1}
    normalized = normalize_proposal(reply, agent="codex")
    assert normalized["agent"] == "codex"
    assert normalized["action"] == "do.not_use"
    assert normalized["priority"] == 1


def test_thalamus_auto_mode_filters_high_volume_infrastructure_noise_but_keeps_behavior():
    t = Thalamus()
    t.track_mode = "auto"

    noisy = [
        "sensor.nas_cpu_percent",
        "sensor.pve_node_netin",
        "sensor.router_uptime",
        "sensor.speedtest_download",
        "binary_sensor.supervisor_running",
    ]
    for entity_id in noisy:
        domain = entity_id.split(".")[0]
        assert t.should_track(entity_id, domain=domain) is False

    assert t.should_track("binary_sensor.kitchen_motion", domain="binary_sensor") is True
    assert t.should_track("sensor.kitchen_temperature", domain="sensor") is True
    assert t.should_track("sensor.router_uptime", domain="sensor", labels=["kontinuum"]) is True


def test_engine_rejects_incomplete_boundary_events_without_learning_tokens():
    engine = KontinuumEngine()

    missing_entity = engine.observe({"new_state": "on"})
    missing_state = engine.observe({"entity_id": "light.kitchen_ceiling"})
    none_event = engine.observe(None)

    assert missing_entity.extra == {"skipped": "no_entity_or_state"}
    assert missing_state.extra == {"skipped": "no_entity_or_state"}
    assert none_event.extra == {"skipped": "no_entity_or_state"}
    assert engine.tick_count == 3
    assert engine.hippocampus.total_events == 0
    assert engine.thalamus.token_to_id == {}


def test_thalamus_vocabulary_pressure_prunes_stale_tokens_but_continues_tokenizing():
    t = Thalamus()
    t.register_entity(
        "climate.test_chamber",
        ha_area="office",
        domain="climate",
        friendly_name="Office test chamber",
    )

    newest_signal = None
    for i in range(Thalamus.MAX_VOCAB_SIZE + 250):
        newest_signal = t.process("climate.test_chamber", f"mode_{i}", f"mode_{i - 1}")

    assert newest_signal is not None
    assert newest_signal["token"] == f"office.climate.mode_{Thalamus.MAX_VOCAB_SIZE + 249}"
    assert len(t.token_to_id) <= Thalamus.MAX_VOCAB_SIZE
    assert t.decode_token(1) == "?1"
    assert t.decode_token(newest_signal["token_id"]) == newest_signal["token"]
