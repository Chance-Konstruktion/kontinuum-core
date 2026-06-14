"""Regression guard built on the replay benchmark.

This is the end-to-end "does it actually learn" check: train on a routine,
inject out-of-distribution events, and assert the surprise signal still
separates them. Thresholds are deliberately loose so the test guards against
real regressions (the engine going blind) rather than pinning exact numbers.
"""
from __future__ import annotations

from benchmarks.replay import run_benchmark


def test_surprise_separates_anomalies_from_routine():
    res = run_benchmark(train_days=30, eval_days=8)
    assert res.n_normal > 0 and res.n_anomaly > 0
    # Injected anomalies must be more surprising than the learned routine...
    assert res.mean_surprise_anomaly > res.mean_surprise_normal
    # ...and the ranking separation must stay well above chance (0.5).
    # Observed ~0.99; 0.8 leaves wide regression headroom without flaking.
    assert res.auc > 0.8, f"separation collapsed: AUC={res.auc:.3f}"
