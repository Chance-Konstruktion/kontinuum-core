"""Schema-versioning guarantees for KontinuumEngine persistence.

The brain is serialized to disk by the host (the HA integrations) and read
back on the next start. These tests lock in that the on-disk format carries a
``schema_version`` and that a restore degrades safely across versions instead
of loading a half-understood brain.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from kontinuum_core import KontinuumEngine


def _warm_engine() -> KontinuumEngine:
    """Return an engine with some learned state to round-trip."""
    e = KontinuumEngine()
    base = datetime(2026, 6, 13, 19, 0, 0, tzinfo=timezone.utc)
    steps = [("switch.kitchen", "on"), ("light.bedroom_lamp", "on"),
             ("switch.kitchen", "off"), ("light.bedroom_lamp", "off")]
    n = 0
    for _ in range(20):
        for entity_id, state in steps:
            e.observe({"entity_id": entity_id, "new_state": state,
                       "timestamp": base + timedelta(minutes=n)})
            n += 1
    return e


def test_to_dict_stamps_current_schema_version():
    snap = _warm_engine().to_dict()
    assert snap["schema_version"] == KontinuumEngine.SCHEMA_VERSION


def test_round_trip_restores_state():
    src = _warm_engine()
    blob = src.to_dict()
    dst = KontinuumEngine()
    dst.from_dict(blob)
    assert dst.tick_count == src.tick_count
    assert dst.to_dict()["modules"].keys() == blob["modules"].keys()


def test_legacy_blob_without_version_is_accepted():
    """Brains written by kontinuum-core <= 0.1.2 carry no schema_version."""
    blob = _warm_engine().to_dict()
    blob.pop("schema_version")  # simulate the pre-versioning 0.1.2 layout
    dst = KontinuumEngine()
    dst.from_dict(blob)
    assert dst.tick_count == blob["tick_count"]


def test_future_schema_is_refused_and_cold_starts():
    """A newer-than-known schema must not be partially applied."""
    blob = _warm_engine().to_dict()
    blob["schema_version"] = KontinuumEngine.SCHEMA_VERSION + 1
    blob["tick_count"] = 999_999
    dst = KontinuumEngine()  # fresh: tick_count == 0
    dst.from_dict(blob)
    assert dst.tick_count == 0, "future-schema brain must be ignored, not loaded"
