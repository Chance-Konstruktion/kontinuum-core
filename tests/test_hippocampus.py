"""Tests for the Hippocampus n-gram memory module."""
from __future__ import annotations

from datetime import datetime, timezone

from kontinuum_core.hippocampus import Hippocampus


def _ctx(time_marker: float = 0.0) -> list:
    """Minimal 21-dim context: 9 time + 9 hypothalamus + 3 insula.

    The exact values do not matter for most tests; we just need the
    vector to be the right length so `_context_bucket()` does not
    short-circuit on `len(ctx) < 7`."""
    return [time_marker] * 21


def test_fresh_instance_has_no_memory():
    h = Hippocampus()
    assert h.total_events == 0
    assert h.predict(_ctx()) == []


def test_learn_increments_total_events():
    h = Hippocampus()
    h.learn(token_id=1, ctx=_ctx(), timestamp=datetime.now(timezone.utc))
    assert h.total_events == 1


def test_predict_returns_list_of_tuples():
    h = Hippocampus()
    now = datetime.now(timezone.utc)
    # Build a tiny sequence so n-grams have something to chew on.
    for token in [1, 2, 1, 2, 1, 2]:
        h.learn(token_id=token, ctx=_ctx(), timestamp=now)
    preds = h.predict(_ctx(), top_k=3)
    assert isinstance(preds, list)
    assert len(preds) <= 3
    for p in preds:
        # (token_id, prob, conf, source, n_obs)
        assert len(p) == 5
        assert isinstance(p[0], int)
        assert 0.0 <= p[1] <= 1.0
        assert 0.0 <= p[2] <= 1.0


def test_accuracy_starts_at_zero():
    """No shadow predictions yet → accuracy is 0.0 (not NaN, not crash)."""
    h = Hippocampus()
    assert h.accuracy == 0.0


def test_top_k_caps_output():
    h = Hippocampus()
    now = datetime.now(timezone.utc)
    for token in [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5]:
        h.learn(token_id=token, ctx=_ctx(), timestamp=now)
    assert len(h.predict(_ctx(), top_k=2)) <= 2
    assert len(h.predict(_ctx(), top_k=10)) <= 10


def test_buffer_keeps_recent_tokens():
    """The n-gram buffer is a bounded deque (maxlen=30)."""
    h = Hippocampus()
    now = datetime.now(timezone.utc)
    for token in range(50):
        h.learn(token_id=token, ctx=_ctx(), timestamp=now)
    assert len(h.buffer) <= 30


def test_small_sample_probability_is_shrunk():
    """count/(total + smoothing): a 2-out-of-2 pattern must not claim
    probability 1.0 — small samples get shrunk toward uncertainty."""
    h = Hippocampus()
    now = datetime.now(timezone.utc)
    for token in [1, 2] * 4:
        h.learn(token_id=token, ctx=_ctx(), timestamp=now)
    preds = h.predict(_ctx())
    assert preds, "expected at least one prediction"
    assert all(p[1] < 1.0 for p in preds)


def test_multi_order_evidence_beats_single_order():
    """A token supported by several n-gram orders accumulates evidence
    and ends up with higher confidence than max-of-one-order would give."""
    h = Hippocampus()
    now = datetime.now(timezone.utc)
    # Strict alternation: the buffer ends in token 2, so the next
    # expected token (1) is supported by 1-,2-,3- and 4-grams at once.
    for token in [1, 2] * 20:
        h.learn(token_id=token, ctx=_ctx(), timestamp=now)
    preds = h.predict(_ctx())
    assert preds
    top = preds[0]
    assert top[0] == 1
    # Evidence from 4 orders → confidence saturates near 1.0.
    assert top[2] > 0.8


def test_decay_prunes_forgotten_entries():
    """Decay removes entries below DECAY_PRUNE_THRESHOLD and cleans up
    empty n-grams/buckets instead of keeping micro-weights forever."""
    h = Hippocampus()
    now = datetime.now(timezone.utc)
    for token in [1, 2, 3, 1, 2, 3]:
        h.learn(token_id=token, ctx=_ctx(), timestamp=now)
    assert h.transitions

    # ~2000 days of decay shrinks every weight below the prune threshold.
    h._apply_decay(2000)
    assert not h.transitions
    assert not h.totals
