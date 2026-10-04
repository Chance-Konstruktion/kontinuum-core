"""Tests for the energy connection (issue #4, Flug 2608): the hypothalamus
transition tokens (house.energy.* / house.climate.*) are LEARNED like sensor
tokens instead of being discarded by observe().

Before this connection absorb()'s return value fell on the floor: the brain
never saw the house's own state changes (battery critical, solar charging,
climate comfort) — only the sluggish 9-dim context vector. These tests lock
in: the transition rides the snapshot (extra["house_transition"]), the minted
token decodes back to its name, the hippocampus counts the learn, and the
none-path (no transition) keeps the extra field honest (None, not noise).

The connection ships BEHIND A SWITCH (Claude's order on #4): the constructor
parameter learn_house_transitions defaults to False — main keeps the old
behavior (observe, don't learn) until the Stufe-1 ablation (#2) shows a
stable win. The learning tests opt in explicitly; the default-off test locks
in the old behavior plus the honest observation.
"""
from __future__ import annotations

from datetime import datetime, timezone

from kontinuum_core import KontinuumEngine


def _build_engine(**kwargs) -> KontinuumEngine:
    e = KontinuumEngine(learn_house_transitions=True, **kwargs)
    e.register_entity("sensor.battery", ha_area="keller", domain="sensor")
    e.register_entity("sensor.solar", ha_area="dach", domain="sensor")
    return e


def _observe(e: KontinuumEngine, entity_id: str, state: str, minute: int):
    e.thalamus.entity_last_token.pop(entity_id, None)
    return e.observe({
        "entity_id": entity_id,
        "new_state": state,
        "old_state": None,
        "timestamp": datetime(2026, 10, 3, 12, minute, tzinfo=timezone.utc),
    })


def _open_cooldown(e: KontinuumEngine):
    """The energy transition carries a 60 s wall-clock cooldown (a named
    finding of Denkrunde #3) — a test cannot wait for it, so the test opens
    the gate the same way the clock would."""
    e.hypothalamus._last_energy_event_time = 0.0


def test_energy_transition_is_learned_not_discarded():
    e = _build_engine()
    _open_cooldown(e)
    snap = _observe(e, "sensor.battery", "critical", 0)

    # The transition rides the snapshot ...
    assert snap.extra.get("house_transition") == "house.energy.critical"
    # ... the token was minted and decodes back to its name ...
    assert e.thalamus.decode_token(
        e.thalamus.token_to_id["house.energy.critical"]
    ) == "house.energy.critical"
    # ... and the hippocampus counted the house learn ON TOP of the sensor
    # learn (two events left this one observe() call: sensor + house).
    assert e.hippocampus.total_events == 2


def test_no_transition_keeps_the_extra_field_honest():
    e = _build_engine()
    _open_cooldown(e)
    _observe(e, "sensor.battery", "critical", 0)
    # Same state again: no significant change, no transition — the extra
    # stays None (the truth, not noise) and only the sensor learn happened.
    snap = _observe(e, "sensor.battery", "critical", 1)
    assert snap.extra.get("house_transition") is None
    assert e.hippocampus.total_events == 3  # 2 (first call) + 1 sensor


def test_house_tokens_share_the_markov_memory():
    e = _build_engine()
    # Battery normal -> solar charging: both house tokens must live in the
    # SAME token vocabulary the thalamus decodes from. (The module's
    # precedence law: battery <= 0 dominates — a critical battery reports
    # "critical" even with the sun shining, so "charging" needs a live
    # battery first. The test learned that from the module, not the wish.)
    _open_cooldown(e)
    # (The state-word ladder is coarse: "normal" carries level 1 = low;
    # only "full" reaches level 3, so "full" is the honest path to a
    # "house.energy.normal" token. The module taught the test again.)
    snap1 = _observe(e, "sensor.battery", "full", 0)
    _open_cooldown(e)
    snap2 = _observe(e, "sensor.solar", "high", 1)

    assert snap1.extra.get("house_transition") == "house.energy.normal"
    assert snap2.extra.get("house_transition") == "house.energy.charging"
    for name in ("house.energy.normal", "house.energy.charging"):
        assert e.thalamus.decode_token(e.thalamus.token_to_id[name]) == name
    # Four events: two sensor learns + two house learns.
    assert e.hippocampus.total_events == 4


def test_default_off_observes_but_does_not_learn():
    """The switch defaults to OFF: main keeps the old behavior (observe,
    don't learn) until the ablation decides. The observation stays honest
    (the transition still rides the snapshot), but no house token is minted
    into the vocabulary and the hippocampus counts only the sensor learn."""
    e = KontinuumEngine()  # deliberately the bare default — no opt-in
    e.register_entity("sensor.battery", ha_area="keller", domain="sensor")
    _open_cooldown(e)
    snap = _observe(e, "sensor.battery", "critical", 0)

    assert snap.extra.get("house_transition") == "house.energy.critical"
    assert "house.energy.critical" not in e.thalamus.token_to_id
    assert e.hippocampus.total_events == 1
