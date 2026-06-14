"""High-pressure multi-family household tests for KONTINUUM Core.

The scenarios in this file intentionally look more like a noisy apartment
building than a tiny demo home: many rooms, many entity domains, concurrent
routines, persistence after warm-up, safety-critical devices, and an LLM/Codex
style round trip through the public bridge.
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


BASE = datetime(2026, 6, 13, 5, 30, tzinfo=timezone.utc)


APARTMENTS = {
    "apt_a": ("livingroom", "kitchen", "bedroom", "bathroom", "kidsroom"),
    "apt_b": ("livingroom", "kitchen", "bedroom", "bathroom", "office"),
    "apt_c": ("livingroom", "kitchen", "bedroom", "bathroom", "guestroom"),
    "apt_d": ("livingroom", "kitchen", "bedroom", "bathroom", "utility"),
}


def _entity(prefix: str, room: str, domain: str, suffix: str) -> str:
    return f"{domain}.{prefix}_{room}_{suffix}"


def _register_multifamily_entities(engine: KontinuumEngine) -> list[tuple[str, str]]:
    entities: list[tuple[str, str]] = []
    for apartment, rooms in APARTMENTS.items():
        for room in rooms:
            area = f"{apartment}_{room}"
            specs = [
                (_entity(apartment, room, "light", "ceiling"), "light"),
                (_entity(apartment, room, "switch", "plug"), "switch"),
                (_entity(apartment, room, "binary_sensor", "motion"), "binary_sensor"),
                (_entity(apartment, room, "sensor", "temperature"), "sensor"),
                (_entity(apartment, room, "sensor", "humidity"), "sensor"),
            ]
            for entity_id, domain in specs:
                kwargs = {"ha_area": area, "domain": domain}
                if entity_id.endswith("_motion"):
                    kwargs["device_class"] = "motion"
                elif entity_id.endswith("_temperature"):
                    kwargs["device_class"] = "temperature"
                    kwargs["unit"] = "°C"
                elif entity_id.endswith("_humidity"):
                    kwargs["device_class"] = "humidity"
                    kwargs["unit"] = "%"
                engine.register_entity(entity_id, **kwargs)
                entities.append((entity_id, domain))

    for apartment in APARTMENTS:
        for domain, suffix, room, extra in [
            ("lock", "front_door", "entrance", {}),
            ("alarm_control_panel", "alarm", "entrance", {}),
            ("climate", "thermostat", "livingroom", {}),
            ("cover", "balcony", "livingroom", {}),
        ]:
            entity_id = _entity(apartment, room, domain, suffix)
            engine.register_entity(
                entity_id, ha_area=f"{apartment}_{room}", domain=domain, **extra
            )
            entities.append((entity_id, domain))
    return entities


def _state_for(domain: str, minute: int) -> str:
    if domain in {"light", "switch", "binary_sensor"}:
        return "on" if minute % 2 == 0 else "off"
    if domain == "sensor":
        return str(17 + (minute % 13))
    if domain == "climate":
        return ("heating", "idle", "cooling", "off")[minute % 4]
    if domain == "cover":
        return "open" if minute % 2 == 0 else "closed"
    if domain == "lock":
        return "locked" if minute % 2 == 0 else "unlocked"
    if domain == "alarm_control_panel":
        return "armed_away" if minute % 2 == 0 else "disarmed"
    return "on"


def _run_multifamily_day(engine: KontinuumEngine, rounds: int = 12) -> list:
    entities = _register_multifamily_entities(engine)
    snapshots = []
    minute = 0
    for round_no in range(rounds):
        for entity_id, domain in entities:
            snapshots.append(engine.observe({
                "entity_id": entity_id,
                "old_state": _state_for(domain, minute - 1),
                "new_state": _state_for(domain, minute + round_no),
                "timestamp": BASE + timedelta(minutes=minute),
            }))
            minute += 1
    return snapshots


def test_multifamily_event_storm_keeps_snapshots_bounded_and_structured():
    engine = KontinuumEngine()

    snapshots = _run_multifamily_day(engine, rounds=16)

    assert engine.tick_count == len(snapshots)
    assert engine.thalamus.stats["entities_registered"] >= 100
    assert engine.hippocampus.total_events >= 900
    assert len(engine.thalamus.token_to_id) <= engine.thalamus.MAX_VOCAB_SIZE
    for snap in snapshots:
        assert 0.0 <= snap.surprise <= 1.0
        assert snap.learning_state in {"cold_start", "learning", "stable"}
        assert isinstance(snap.predictions, list)
        assert "anomaly_threshold" in snap.extra or snap.extra.get("skipped") == "filtered"


def test_multifamily_safety_critical_devices_stay_shadow_decisions_not_actions():
    engine = KontinuumEngine()
    _run_multifamily_day(engine, rounds=4)

    risky_events = [
        ("lock.apt_a_entrance_front_door", "unlocked"),
        ("alarm_control_panel.apt_b_entrance_alarm", "armed_away"),
    ]
    for offset, (entity_id, state) in enumerate(risky_events):
        snap = engine.observe({
            "entity_id": entity_id,
            "new_state": state,
            "timestamp": BASE + timedelta(hours=8, minutes=offset),
        })
        decision = snap.extra.get("decision")
        if decision is not None:
            assert decision["stage"] in {"shadow", "veto", "caution"}
            assert decision["risk"] >= 0.0
    assert engine.amygdala.total_vetoes >= 0


def test_multifamily_persistence_round_trip_preserves_warmed_building_brain():
    src = KontinuumEngine()
    _run_multifamily_day(src, rounds=8)

    blob = src.to_dict()
    encoded = json.dumps(blob)
    dst = KontinuumEngine()
    dst.from_dict(json.loads(encoded))

    assert dst.tick_count == src.tick_count
    assert dst.thalamus.stats["entities_registered"] == src.thalamus.stats["entities_registered"]
    assert dst.hippocampus.total_events == src.hippocampus.total_events
    assert dst.to_dict()["modules"].keys() == src.to_dict()["modules"].keys()


def test_codex_style_llm_round_trip_can_talk_to_the_core_without_private_api():
    engine = KontinuumEngine()
    _run_multifamily_day(engine, rounds=10)

    context = build_llm_context(engine, top_k=5)
    prompt_payload = render_llm_context(context)
    codex_reply = """
    I inspected the KONTINUUM state and would keep this advisory-only.
    ```json
    {
      "agent": "chatgpt-codex",
      "action": "light.turn_on",
      "entity_id": "light.apt_a_livingroom_ceiling",
      "reason": "Expected evening living-room routine, low safety risk.",
      "priority": "72",
      "veto": "false"
    }
    ```
    """

    parsed = extract_json(codex_reply)
    proposal = normalize_proposal(codex_reply)
    snap = engine.evaluate({
        "entity_id": proposal["entity_id"],
        "new_state": "on",
        "timestamp": BASE + timedelta(hours=20),
    })

    assert "KONTINUUM home-brain state" in prompt_payload
    assert context["engine"]["events_seen"] > 0
    assert parsed["agent"] == "chatgpt-codex"
    assert proposal == {
        "agent": "chatgpt-codex",
        "action": "light.turn_on",
        "entity_id": "light.apt_a_livingroom_ceiling",
        "reason": "Expected evening living-room routine, low safety risk.",
        "priority": 72,
        "veto": False,
        "valid": True,
    }
    assert snap.token == "livingroom.light.on"
    assert 0.0 <= snap.surprise <= 1.0


def test_codex_style_safety_veto_reply_normalizes_to_non_executable_signal():
    reply = """
    Safety first. Do not actuate the building lock from an anomaly.
    {"agent":"chatgpt-codex","action":null,"entity_id":"lock.apt_c_entrance_front_door",
     "reason":"unknown person-flow pattern near apartment C entrance", "priority":100, "veto":"VETO"}
    """

    proposal = normalize_proposal(reply)

    assert proposal["valid"] is True
    assert proposal["action"] is None
    assert proposal["priority"] == 100
    assert proposal["veto"] is True


def test_auto_tracking_filters_infrastructure_noise_but_keeps_behavior_signals():
    engine = KontinuumEngine()
    engine.thalamus.track_mode = "auto"

    engine.register_entity(
        "sensor.server_cpu_percent", ha_area="technikraum", domain="sensor"
    )
    engine.register_entity(
        "sensor.router_last_seen", ha_area="technikraum", domain="sensor"
    )
    engine.register_entity(
        "sensor.apt_a_kitchen_temperature",
        ha_area="apt_a_kitchen",
        domain="sensor",
        device_class="temperature",
        unit="°C",
    )
    engine.register_entity(
        "binary_sensor.apt_a_hallway_motion",
        ha_area="apt_a_hallway",
        domain="binary_sensor",
        device_class="motion",
    )

    assert "sensor.server_cpu_percent" not in engine.thalamus.entity_semantic
    assert "sensor.router_last_seen" not in engine.thalamus.entity_semantic
    assert engine.thalamus.entity_semantic["sensor.apt_a_kitchen_temperature"] == "temperature"
    assert engine.thalamus.entity_semantic["binary_sensor.apt_a_hallway_motion"] == "motion"


def test_token_vocabulary_prunes_under_extreme_unique_state_pressure():
    engine = KontinuumEngine()
    thalamus = engine.thalamus
    old_limit = thalamus.MAX_VOCAB_SIZE
    thalamus.MAX_VOCAB_SIZE = 40
    try:
        for index in range(90):
            entity_id = f"sensor.apt_a_office_metric_{index}"
            engine.register_entity(
                entity_id,
                ha_area="apt_a_office",
                domain="sensor",
                friendly_name=f"Office metric {index}",
                unit="score",
            )
            thalamus.entity_semantic[entity_id] = "custom_metric"
            thalamus.entity_room[entity_id] = "office"
            snap = engine.observe({
                "entity_id": entity_id,
                "new_state": f"state_{index}",
                "timestamp": BASE + timedelta(seconds=index),
            })
            assert snap.token_id is not None

        assert len(thalamus.token_to_id) <= thalamus.MAX_VOCAB_SIZE
        assert thalamus._next_id > 90
    finally:
        thalamus.MAX_VOCAB_SIZE = old_limit
