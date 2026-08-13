"""Tests for LLM-seeded priors (kontinuum_core.priors).

Two things to lock in: the parser survives sloppy LLM output, and seeding
actually gives the engine a day-1 head start — it recognizes the household
routine immediately instead of treating everything as new.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from kontinuum_core import (
    KontinuumEngine,
    parse_home_prior,
    seed_engine_from_prior,
)

_PRIOR = {
    "rooms": ["bedroom", "bathroom", "kitchen", "living"],
    "routines": [
        {"name": "morning", "events": [
            {"hour": 7, "minute": 0, "room": "bedroom"},
            {"hour": 7, "minute": 10, "room": "bathroom"},
            {"hour": 7, "minute": 25, "room": "kitchen"},
            {"hour": 8, "minute": 0, "room": "living"},
        ]},
        {"name": "evening", "events": [
            {"hour": 19, "minute": 0, "room": "kitchen"},
            {"hour": 20, "minute": 0, "room": "living"},
            {"hour": 22, "minute": 30, "room": "bathroom"},
            {"hour": 23, "minute": 0, "room": "bedroom"},
        ]},
    ],
}

_MORNING = [(7, 0, "bedroom"), (7, 10, "bathroom"), (7, 25, "kitchen"), (8, 0, "living")]


def _replay_morning(engine, day) -> float:
    surprises = []
    for hh, mm, room in _MORNING:
        for state, delta in (("on", 0), ("off", 2)):
            snap = engine.observe({
                "entity_id": f"binary_sensor.motion_{room}",
                "new_state": state,
                "timestamp": day.replace(hour=hh, minute=mm) + timedelta(minutes=delta),
            })
            if "skipped" not in snap.extra:
                surprises.append(snap.surprise)
    return sum(surprises) / len(surprises)


# --------------------------------------------------------------------------
# parse_home_prior
# --------------------------------------------------------------------------
def test_parse_accepts_fenced_json_and_normalizes():
    raw = '```json\n{"rooms":["Bed Room"],"routines":[{"name":"m","events":' \
          '[{"hour":"7","minute":"5","room":"Bed Room"}]}]}\n```'
    prior = parse_home_prior(raw)
    assert prior["valid"] is True
    assert prior["rooms"] == ["bed_room"]
    assert prior["routines"][0]["events"][0] == {"hour": 7, "minute": 5, "room": "bed_room"}


def test_parse_drops_malformed_events_and_derives_rooms():
    raw = {"routines": [{"name": "x", "events": [
        {"hour": 7, "minute": 0, "room": "kitchen"},
        {"hour": 99, "minute": 0, "room": "kitchen"},   # bad hour -> dropped
        {"room": "kitchen"},                              # no hour -> dropped
        {"hour": 8, "minute": 0},                         # no room -> dropped
    ]}]}
    prior = parse_home_prior(raw)
    assert len(prior["routines"][0]["events"]) == 1
    assert prior["rooms"] == ["kitchen"]  # derived from events


def test_parse_garbage_is_invalid():
    assert parse_home_prior("the model never answered in json")["valid"] is False
    assert parse_home_prior({"routines": []})["valid"] is False


# --------------------------------------------------------------------------
# seed_engine_from_prior
# --------------------------------------------------------------------------
def test_seeding_is_a_noop_for_invalid_prior():
    engine = KontinuumEngine()
    out = seed_engine_from_prior(engine, parse_home_prior("nope"))
    assert out["seeded"] is False
    assert engine.hippocampus.total_events == 0


def test_seeding_gives_a_day_one_head_start():
    """A seeded engine recognizes the routine on day 1; an unseeded one doesn't."""
    prior = parse_home_prior(_PRIOR)
    probe = datetime(2026, 3, 1, tzinfo=timezone.utc)

    seeded = KontinuumEngine()
    summary = seed_engine_from_prior(seeded, prior, days=30)
    assert summary["seeded"] is True and summary["events"] > 0

    unseeded = KontinuumEngine()
    for room in prior["rooms"]:
        unseeded.register_entity(f"binary_sensor.motion_{room}", ha_area=room,
                                 domain="binary_sensor", device_class="motion")

    seeded_routine = _replay_morning(seeded, probe + timedelta(days=31))
    unseeded_routine = _replay_morning(unseeded, probe)

    # The seeded brain finds the routine far less surprising than a blank one...
    assert seeded_routine < 0.3, f"seeded routine still surprising: {seeded_routine:.3f}"
    assert unseeded_routine > 0.5, f"unseeded baseline too low: {unseeded_routine:.3f}"
    assert seeded_routine < unseeded_routine * 0.5

    # ...yet still flags an off-pattern event (kitchen at 03:00) as surprising.
    anomaly = seeded.observe({
        "entity_id": "binary_sensor.motion_kitchen", "new_state": "on",
        "timestamp": (probe + timedelta(days=33)).replace(hour=3, minute=0),
    })
    assert anomaly.surprise > seeded_routine * 2, (
        f"off-pattern not surprising enough: {anomaly.surprise:.3f} vs routine {seeded_routine:.3f}")
