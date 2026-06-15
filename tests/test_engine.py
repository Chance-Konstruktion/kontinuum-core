"""Smoke tests for the KontinuumEngine public contract.

These tests pin the surface described in the roadmap (section 4 –
Core-API) so accidental regressions in the engine constructor or the
observe-pipeline are caught before they reach the HA integrations.

The tests deliberately avoid asserting on the *values* produced by the
neuro-modules (those are non-trivial and covered by per-module tests);
they only assert on structure, wiring, and the documented invariants.
"""
from __future__ import annotations

from kontinuum_core import (
    KontinuumEngine,
    MemoryState,
    Observation,
    Prediction,
    Scheduler,
)
from kontinuum_core.engine import EngineSnapshot


class FakeScheduler:
    """Minimal Scheduler-Protocol implementation for tests."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def schedule_interval(self, callback, seconds):
        self.calls.append((callback, seconds))
        return lambda: None


# ---------------------------------------------------------------------------
# Public-API surface
# ---------------------------------------------------------------------------

def test_public_exports():
    """Roadmap section 4 pins the public API surface."""
    import kontinuum_core

    assert set(kontinuum_core.__all__) == {
        "KontinuumEngine",
        "Scheduler",
        "Observation",
        "Prediction",
        "MemoryState",
        "build_llm_context",
        "render_llm_context",
        "extract_json",
        "normalize_proposal",
        "HOME_PRIOR_PROMPT",
        "parse_home_prior",
        "seed_engine_from_prior",
    }
    assert kontinuum_core.__version__ == "0.5.0"


def test_data_types_default_construct():
    """Observation/Prediction/MemoryState are pure dataclasses."""
    assert Observation().payload == {}
    pred = Prediction()
    assert pred.expected == {}
    assert pred.confidence == 0.0
    assert MemoryState().data == {}


def test_scheduler_protocol_is_runtime_checkable():
    """`isinstance(obj, Scheduler)` must work for duck-typed objects."""
    assert isinstance(FakeScheduler(), Scheduler)


# ---------------------------------------------------------------------------
# Constructor contract
# ---------------------------------------------------------------------------

def test_constructor_defaults():
    """KontinuumEngine() must work without any arguments."""
    e = KontinuumEngine()
    assert e.config == {}
    assert e.scheduler is None
    assert e.tick_count == 0


def test_constructor_accepts_full_contract():
    """KontinuumEngine accepts (config, scheduler, storage_path)."""
    s = FakeScheduler()
    e = KontinuumEngine(config={"foo": 1}, scheduler=s, storage_path="/tmp/x")
    assert e.config == {"foo": 1}
    assert e.scheduler is s


def test_metaplasticity_receives_scheduler_and_full_brain_dict():
    """Metaplasticity must see scheduler + all 7 modules it tracks."""
    s = FakeScheduler()
    e = KontinuumEngine(scheduler=s)

    assert e.metaplasticity._scheduler is s
    assert set(e.metaplasticity._brain_modules.keys()) == {
        "hippocampus",
        "predictive",
        "cerebellum",
        "basal_ganglia",
        "reticular",
        "accumbens",
        "locus",
    }


def test_metaplasticity_start_calls_scheduler():
    """metaplasticity.start() must register an interval on the scheduler."""
    s = FakeScheduler()
    e = KontinuumEngine(scheduler=s)
    e.metaplasticity.start(interval_hours=24)

    assert len(s.calls) == 1
    callback, seconds = s.calls[0]
    assert seconds == 24 * 3600
    assert callable(callback)


def test_metaplasticity_start_without_scheduler_is_a_noop():
    """Without a scheduler, start() must not raise."""
    e = KontinuumEngine(scheduler=None)
    e.metaplasticity.start(interval_hours=24)  # must not raise


# ---------------------------------------------------------------------------
# Observe pipeline
# ---------------------------------------------------------------------------

def test_observe_without_event_returns_skipped_snapshot():
    """Empty observe() must return a well-formed snapshot, not raise."""
    e = KontinuumEngine()
    snap = e.observe()
    assert isinstance(snap, EngineSnapshot)
    assert snap.extra == {"skipped": "no_entity_or_state"}
    assert snap.tick_count == 1


def test_observe_with_payload_advances_tick():
    """Real payload must run through the pipeline and tick the counter."""
    e = KontinuumEngine()
    snap = e.observe({"entity_id": "sensor.test", "new_state": "21.5"})
    assert isinstance(snap, EngineSnapshot)
    assert snap.tick_count == 1
    assert 0.0 <= snap.surprise <= 1.0


def test_evaluate_is_an_alias_for_observe():
    """The service entry-point keeps the same shape as observe()."""
    e = KontinuumEngine()
    snap = e.evaluate({"entity_id": "sensor.test", "new_state": "x"})
    assert isinstance(snap, EngineSnapshot)
    assert snap.tick_count == 1


# ---------------------------------------------------------------------------
# Module wiring
# ---------------------------------------------------------------------------

def test_engine_exposes_all_18_modules():
    """All 18 brain modules from the roadmap table must be reachable."""
    e = KontinuumEngine()
    expected = {
        "thalamus",
        "hippocampus",
        "predictive",
        "cerebellum",
        "basal_ganglia",
        "neurorhythms",
        "sleep_consolidation",
        "amygdala",
        "insula",
        "hypothalamus",
        "spatial_cortex",
        "prefrontal_cortex",
        "anterior_cingulate",
        "entorhinal_cortex",
        "locus_coeruleus",
        "nucleus_accumbens",
        "reticular",
        "metaplasticity",
    }
    for name in expected:
        assert getattr(e, name) is not None, f"missing module: {name}"


def test_register_entity_delegates_to_thalamus():
    """KontinuumEngine.register_entity must forward to Thalamus."""
    e = KontinuumEngine()
    # Thalamus.register_entity accepts (entity_id, ha_area, device_class,
    # domain, friendly_name, unit, labels). The engine's wrapper forwards
    # **kwargs verbatim — passing a known keyword must not raise.
    e.register_entity("sensor.bedroom_temp", ha_area="bedroom")
    snap = e.observe({"entity_id": "sensor.bedroom_temp", "new_state": "21.5"})
    assert snap.tick_count == 1
