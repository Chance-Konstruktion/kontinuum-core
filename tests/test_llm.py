"""Tests for the LLM data contract (kontinuum_core.llm).

Two directions: the engine→LLM context export must be robust and carry the
anomaly signal; the LLM→engine parsing must survive the sloppiness real models
produce (code fences, prose, stringly-typed fields).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from kontinuum_core import (
    KontinuumEngine,
    build_llm_context,
    render_llm_context,
    extract_json,
    normalize_proposal,
)
from kontinuum_core.llm import CONTEXT_SCHEMA_VERSION


def _warm_engine() -> KontinuumEngine:
    e = KontinuumEngine()
    e.register_entity("switch.kitchen", ha_area="kitchen", domain="switch")
    e.register_entity("light.bedroom_lamp", ha_area="bedroom", domain="light")
    base = datetime(2026, 6, 13, 19, 0, 0, tzinfo=timezone.utc)
    steps = [("switch.kitchen", "on"), ("light.bedroom_lamp", "on"),
             ("switch.kitchen", "off"), ("light.bedroom_lamp", "off")]
    n = 0
    for _ in range(30):
        for entity_id, state in steps:
            e.observe({"entity_id": entity_id, "new_state": state,
                       "timestamp": base + timedelta(minutes=n)})
            n += 1
    return e


# --------------------------------------------------------------------------
# Engine -> LLM
# --------------------------------------------------------------------------
def test_context_has_versioned_schema_and_anomaly_signal():
    ctx = build_llm_context(_warm_engine())
    assert ctx["schema_version"] == CONTEXT_SCHEMA_VERSION
    # The anomaly signal (the whole point of feeding the LLM) must be present
    # and well-formed.
    ano = ctx["anomaly"]
    assert 0.0 <= ano["surprise"] <= 1.0
    assert 0.0 <= ano["threshold"] <= 1.0
    assert isinstance(ano["is_anomaly"], bool)
    # Scales must be documented so the model can interpret the numbers.
    assert "surprise" in ctx["scales"]
    assert ctx["engine"]["events_seen"] > 0


def test_context_never_raises_on_a_blank_engine():
    """A cold engine (no observations) must still produce a valid context."""
    ctx = build_llm_context(KontinuumEngine())
    assert ctx["engine"]["learning_maturity"] == "cold_start"
    assert ctx["prediction"]["expected_next"] == []


def test_render_context_is_labeled_text_with_scales():
    text = render_llm_context(build_llm_context(_warm_engine()))
    assert "KONTINUUM" in text
    assert "Anomaly:" in text
    assert "Scales:" in text  # the model is told the 0–1 meaning


# --------------------------------------------------------------------------
# LLM -> Engine : extract_json
# --------------------------------------------------------------------------
def test_extract_plain_json():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_already_parsed():
    assert extract_json({"a": 1}) == {"a": 1}


def test_extract_from_markdown_fence():
    raw = '```json\n{"action": "light.turn_on", "priority": 80}\n```'
    assert extract_json(raw) == {"action": "light.turn_on", "priority": 80}


def test_extract_from_surrounding_prose():
    raw = 'Sure! Here is my answer:\n{"action": null, "priority": 0}\nHope that helps.'
    assert extract_json(raw) == {"action": None, "priority": 0}


def test_extract_handles_nested_objects():
    raw = 'noise {"a": {"b": [1, 2]}, "c": "}"} trailing'
    assert extract_json(raw) == {"a": {"b": [1, 2]}, "c": "}"}


def test_extract_garbage_returns_none():
    assert extract_json("totally not json") is None
    assert extract_json(42) is None


# --------------------------------------------------------------------------
# LLM -> Engine : normalize_proposal
# --------------------------------------------------------------------------
def test_normalize_coerces_stringly_typed_fields():
    out = normalize_proposal(
        '{"action": "light.turn_on", "entity_id": "light.k", '
        '"reason": "dark", "priority": "85", "veto": "false"}',
        agent="comfort",
    )
    assert out["valid"] is True
    assert out["agent"] == "comfort"
    assert out["priority"] == 85 and isinstance(out["priority"], int)
    assert out["veto"] is False


def test_normalize_clamps_priority_and_parses_veto_variants():
    assert normalize_proposal('{"priority": 150}')["priority"] == 100
    assert normalize_proposal('{"priority": -5}')["priority"] == 0
    for truthy in ('{"veto": true}', '{"veto": "yes"}', '{"veto": 1}', '{"veto": "VETO"}'):
        assert normalize_proposal(truthy)["veto"] is True


def test_normalize_maps_nullish_to_none():
    out = normalize_proposal('{"action": "null", "entity_id": "", "reason": "keine"}')
    assert out["action"] is None
    assert out["entity_id"] is None
    assert out["reason"] == ""


def test_normalize_unparseable_is_marked_invalid():
    out = normalize_proposal("the model rambled and never produced json", agent="safety")
    assert out["valid"] is False
    assert out["action"] is None and out["priority"] == 0
    assert out["agent"] == "safety"  # agent still stamped for traceability


def test_normalize_handles_fenced_reply_end_to_end():
    raw = '```\n{"action": "climate.set", "priority": 60, "veto": false}\n```'
    out = normalize_proposal(raw)
    assert out["valid"] is True
    assert out["action"] == "climate.set" and out["priority"] == 60
