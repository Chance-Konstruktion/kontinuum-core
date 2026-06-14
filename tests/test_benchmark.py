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


def test_anomaly_flag_has_usable_recall_when_converged():
    """The built-in ``anomaly`` flag must actually fire on anomalies.

    The replay benchmark is deterministic, so these are hard gates. They lock
    in the robust (median + MAD) adaptive threshold: before it, the 0.55 floor
    made the flag fire on only ~6% of injected anomalies (recall 0.06). Measured
    in the converged regime (enough training that the routine is learned),
    observed precision 0.96 / recall 0.76 — the bounds leave regression
    headroom without flaking.
    """
    res = run_benchmark(train_days=40, eval_days=12)
    assert res.auc > 0.9, f"ranking separation regressed: AUC={res.auc:.3f}"
    assert res.recall >= 0.6, f"anomaly-flag recall regressed: {res.recall:.3f}"
    assert res.precision >= 0.85, f"too many false alarms: precision={res.precision:.3f}"
