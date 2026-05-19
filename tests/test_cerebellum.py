"""Tests for the Cerebellum reflex/rule module."""
from __future__ import annotations

from kontinuum_core.cerebellum import Cerebellum


def test_fresh_instance_has_no_rules():
    c = Cerebellum()
    assert c.rules == {}
    assert c.chunks == []
    assert c._total_fired == 0


def test_check_without_rules_returns_none():
    """check() on an empty cerebellum must not raise — just returns None."""
    c = Cerebellum()
    assert c.check(token_id=42) is None


def test_compile_rules_on_empty_hippocampus_is_safe():
    """A blank Hippocampus has no transitions → no rules emitted, no crash."""
    from kontinuum_core.hippocampus import Hippocampus

    c = Cerebellum()
    c.compile_rules(Hippocampus())
    assert c.rules == {}


def test_set_context_is_a_simple_setter():
    """set_context(bucket) just stores the bucket for later check() use."""
    c = Cerebellum()
    c.set_context(7)
    assert c._current_bucket == 7
    c.set_context(None)
    assert c._current_bucket is None


def test_stats_returns_dict_with_documented_keys():
    """The stats property is the diagnostic surface — its shape is stable."""
    c = Cerebellum()
    s = c.stats
    assert isinstance(s, dict)
    # The current stats payload is consumed by sensor attributes — these
    # keys are the contract.
    for key in ("rules_count", "rules_1gram", "rules_2gram", "rules_3gram",
                "chunks_count", "total_fires", "success_rate"):
        assert key in s, f"missing stats key: {key}"


def test_to_dict_from_dict_round_trip():
    """A blank cerebellum serialises and round-trips without losing identity."""
    c = Cerebellum()
    blob = c.to_dict()
    fresh = Cerebellum()
    fresh.from_dict(blob)
    assert fresh.rules == c.rules
    assert fresh.chunks == c.chunks
