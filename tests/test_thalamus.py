"""Tests for Thalamus.process() — the entry point of the observe pipeline."""
from __future__ import annotations

from kontinuum_core.thalamus import Thalamus


def test_get_or_create_token_is_stable_and_mints_new():
    t = Thalamus()
    a = t.get_or_create_token("kitchen.light.on")
    b = t.get_or_create_token("kitchen.light.on")  # same token -> same id
    c = t.get_or_create_token("bedroom.motion.on")  # new token -> new id
    assert a == b
    assert a != c
    assert t.decode_token(a) == "kitchen.light.on"
    assert t.decode_token(c) == "bedroom.motion.on"


def _register(t: Thalamus, entity_id: str, area: str = "bedroom") -> None:
    """Convenience: register an entity with all the bits Thalamus needs to
    actually emit a token (room must resolve, semantic must resolve)."""
    domain = entity_id.split(".")[0]
    # device_class hints help _resolve_semantic for binary_sensor.
    device_class = "motion" if domain == "binary_sensor" else ""
    t.register_entity(
        entity_id,
        ha_area=area,
        device_class=device_class,
        domain=domain,
        friendly_name=entity_id.split(".")[-1].replace("_", " "),
    )


def test_unregistered_entity_returns_none():
    """An entity that was never registered is invisible to Thalamus."""
    t = Thalamus()
    assert t.process("sensor.unknown_entity", "1", None) is None


def test_registered_entity_emits_token_dict():
    """A registered entity with a real state change yields a token dict."""
    t = Thalamus()
    _register(t, "binary_sensor.bedroom_motion")
    signal = t.process("binary_sensor.bedroom_motion", "on", "off")
    assert signal is not None
    for key in ("token_id", "token", "room", "semantic", "state"):
        assert key in signal, f"missing key: {key}"
    assert signal["room"] == "bedroom"
    assert isinstance(signal["token_id"], int)
    assert signal["token"].startswith("bedroom.")


def test_same_token_twice_in_a_row_is_filtered():
    """Thalamus de-duplicates: repeating the same state→token transition
    must return None on the second call. This is what stops the engine
    from learning that 'light is still on' over and over again."""
    t = Thalamus()
    _register(t, "binary_sensor.bedroom_motion")
    first = t.process("binary_sensor.bedroom_motion", "on", "off")
    second = t.process("binary_sensor.bedroom_motion", "on", "on")
    assert first is not None
    assert second is None  # filtered


def test_token_ids_are_stable_and_incrementing():
    """Each unique token gets one stable id; new tokens get fresh ids."""
    t = Thalamus()
    _register(t, "binary_sensor.bedroom_motion")
    _register(t, "binary_sensor.kitchen_motion", area="kitchen")
    sig_a = t.process("binary_sensor.bedroom_motion", "on", "off")
    sig_b = t.process("binary_sensor.kitchen_motion", "on", "off")
    assert sig_a["token_id"] != sig_b["token_id"]
    # Re-encounter token A: same id.
    t.process("binary_sensor.bedroom_motion", "off", "on")
    sig_a_again = t.process("binary_sensor.bedroom_motion", "on", "off")
    assert sig_a_again["token_id"] == sig_a["token_id"]


def test_encode_time_context_returns_9_dim_vector():
    """The 21-dim engine context vector includes 9 dims from Thalamus."""
    from datetime import datetime, timezone

    t = Thalamus()
    vec = t.encode_time_context(datetime(2026, 5, 19, 14, 30, tzinfo=timezone.utc))
    assert len(vec) == 9
    for component in vec:
        assert isinstance(component, (int, float))
