"""LLM-seeded priors — give the engine a day-1 head start.

The companion to :mod:`kontinuum_core.llm`'s context export. That path lets a
model *read* the engine; this path lets a model *seed* it. At setup the LLM is
asked to describe the home's typical routines as JSON (see
:data:`HOME_PRIOR_PROMPT`); :func:`parse_home_prior` validates that reply and
:func:`seed_engine_from_prior` replays it through the engine's normal learning
path, so KONTINUUM starts out already expecting the household's rhythm instead of
learning everything from a blank slate.

Design choice: priors are seeded by **replaying synthetic events through
``observe()``**, not by poking module internals. That reuses the real, tested
learning machinery, keeps the prior just a (low-evidence) starting point, and
lets real observations naturally override it over time — a wrong guess from the
LLM decays as the home's actual behaviour accumulates.

HA-free, pure stdlib.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .llm import extract_json

PRIOR_SCHEMA_VERSION = 1

# The instruction handed to the LLM at setup. Kept here so every integration
# asks for the same, parseable shape.
HOME_PRIOR_PROMPT = (
    "You are bootstrapping KONTINUUM, a neuro-inspired smart-home brain, for a "
    "new home. Describe the household's TYPICAL daily routines so the brain has "
    "a sensible starting expectation before it has observed anything.\n"
    "Reply with ONLY JSON of this shape:\n"
    '{"rooms": ["bedroom", "kitchen", ...], '
    '"routines": [{"name": "morning", "events": ['
    '{"hour": 7, "minute": 0, "room": "bedroom"}, '
    '{"hour": 7, "minute": 15, "room": "kitchen"}]}]}\n'
    "Use 24h time. List each routine's events in the order they usually happen. "
    "Only include rooms a person actually moves through."
)


def _int_in(v: Any, lo: int, hi: int) -> Optional[int]:
    try:
        n = int(v)
    except (TypeError, ValueError):
        return None
    return n if lo <= n <= hi else None


def _clean_room(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip().lower().replace(" ", "_")
    return s or None


def parse_home_prior(raw: Any) -> Dict[str, Any]:
    """Validate/normalize an LLM home description into a clean prior.

    Forgiving like :func:`~kontinuum_core.llm.normalize_proposal`: survives code
    fences / prose, drops malformed events and routines, derives ``rooms`` from
    the routines if absent. Returns ``valid=False`` when nothing usable remains.
    """
    data = extract_json(raw)
    if not isinstance(data, dict):
        return {"schema_version": PRIOR_SCHEMA_VERSION, "rooms": [],
                "routines": [], "valid": False}

    routines: List[Dict[str, Any]] = []
    for r in data.get("routines", []) or []:
        if not isinstance(r, dict):
            continue
        events = []
        for e in r.get("events", []) or []:
            if not isinstance(e, dict):
                continue
            room = _clean_room(e.get("room"))
            hour = _int_in(e.get("hour"), 0, 23)
            minute = _int_in(e.get("minute", 0), 0, 59)
            if room is None or hour is None or minute is None:
                continue
            events.append({"hour": hour, "minute": minute, "room": room})
        if events:
            routines.append({"name": str(r.get("name", "routine")), "events": events})

    rooms = [r for r in (_clean_room(x) for x in (data.get("rooms") or [])) if r]
    if not rooms:  # derive from routine events
        seen = []
        for r in routines:
            for e in r["events"]:
                if e["room"] not in seen:
                    seen.append(e["room"])
        rooms = seen

    return {
        "schema_version": PRIOR_SCHEMA_VERSION,
        "rooms": rooms,
        "routines": routines,
        "valid": bool(routines),
    }


def seed_engine_from_prior(engine, prior: Dict[str, Any], *, days: int = 30,
                           start: Optional[datetime] = None,
                           sensor_prefix: str = "binary_sensor.motion_") -> Dict[str, Any]:
    """Warm the engine by replaying a parsed prior through ``observe()``.

    Registers a motion sensor per room and replays each routine's events
    (``on`` then ``off``) across ``days`` synthetic days, so the engine learns
    the household rhythm before it has seen any real event. No-op for an invalid
    or empty prior.
    """
    if not prior or not prior.get("routines"):
        return {"seeded": False, "events": 0, "rooms": [], "routines": []}

    rooms = prior.get("rooms") or []
    for room in rooms:
        engine.register_entity(f"{sensor_prefix}{room}", ha_area=room,
                               domain="binary_sensor", device_class="motion")

    start = start or datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    seeded = 0
    for d in range(days):
        day = start + timedelta(days=d)
        for routine in prior["routines"]:
            for ev in routine["events"]:
                ts = day.replace(hour=ev["hour"], minute=ev["minute"])
                entity = f"{sensor_prefix}{ev['room']}"
                for state, delta in (("on", 0), ("off", 2)):
                    engine.observe({"entity_id": entity, "new_state": state,
                                    "timestamp": ts + timedelta(minutes=delta)})
                    seeded += 1

    return {
        "seeded": True,
        "events": seeded,
        "rooms": rooms,
        "routines": [r["name"] for r in prior["routines"]],
    }
