"""Tests for PredictiveProcessing.compute_surprise() — surprise + learn_weight."""
from __future__ import annotations

from kontinuum_core.predictive_processing import (
    MAX_LEARN_WEIGHT,
    MIN_LEARN_WEIGHT,
    PredictiveProcessing,
)


def test_initial_state():
    """Fresh instance is empty / at default baseline."""
    p = PredictiveProcessing()
    assert p.total_events == 0
    assert p.current_surprise == 0.0
    assert 0.0 < p.baseline_surprise < 1.0


def test_unpredicted_token_yields_high_surprise():
    """No matching prediction → maximum prediction-error component."""
    p = PredictiveProcessing()
    s = p.compute_surprise(token_id=42, predictions=[])
    # 70% prediction (1.0) + 30% novelty (1.0) → ~1.0 capped.
    assert s >= 0.9
    assert p.total_events == 1


def test_perfectly_predicted_token_yields_low_surprise():
    """Token matches the top prediction with prob=1.0, conf=1.0 → low."""
    p = PredictiveProcessing()
    # Warm up familiarity so novelty is low, too.
    for _ in range(60):
        p.compute_surprise(token_id=7, predictions=[(7, 1.0, 1.0, "x", 60)])
    final = p.compute_surprise(token_id=7, predictions=[(7, 1.0, 1.0, "x", 60)])
    assert final < 0.2


def test_learn_weight_clamped_to_documented_range():
    """learn_weight stays within [MIN_LEARN_WEIGHT, MAX_LEARN_WEIGHT]."""
    p = PredictiveProcessing()
    # Drive a surprising event.
    p.compute_surprise(token_id=99, predictions=[])
    high = p.get_learn_weight()
    assert MIN_LEARN_WEIGHT <= high <= MAX_LEARN_WEIGHT

    # Drive expected events.
    for _ in range(80):
        p.compute_surprise(token_id=1, predictions=[(1, 1.0, 1.0, "x", 80)])
    low = p.get_learn_weight()
    assert MIN_LEARN_WEIGHT <= low <= MAX_LEARN_WEIGHT
    # And the expected case must yield a *smaller* weight than the surprise case.
    assert low < high


def test_familiarity_reduces_surprise_over_repetitions():
    """Same token, no predictions: surprise drops as the token becomes familiar."""
    p = PredictiveProcessing()
    first = p.compute_surprise(token_id=5, predictions=[])
    for _ in range(40):
        p.compute_surprise(token_id=5, predictions=[])
    later = p.compute_surprise(token_id=5, predictions=[])
    assert later < first


def test_surprise_history_capped_at_100():
    """surprise_history is a bounded ring buffer (deque maxlen=100)."""
    p = PredictiveProcessing()
    for i in range(150):
        p.compute_surprise(token_id=i, predictions=[])
    assert len(p.surprise_history) == 100
    assert p.total_events == 150


def test_to_dict_round_trip_preserves_persistent_state():
    """Serialisation contract: persistent fields (familiarity, totals,
    baseline, max_surprise) round-trip. Volatile fields like
    current_surprise and surprise_history deliberately reset to defaults
    on load — they are not part of the learned state."""
    p = PredictiveProcessing()
    for i in range(20):
        p.compute_surprise(token_id=i % 5, predictions=[])

    snapshot = p.to_dict()
    fresh = PredictiveProcessing()
    fresh.from_dict(snapshot)

    assert fresh.total_events == p.total_events
    assert fresh.total_surprises == p.total_surprises
    assert fresh.total_expected == p.total_expected
    assert fresh.baseline_surprise == p.baseline_surprise
    assert fresh.max_surprise == p.max_surprise
    assert fresh._token_familiarity == p._token_familiarity
