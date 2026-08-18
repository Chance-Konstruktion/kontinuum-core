"""Regression guard built on the replay benchmark.

This is the end-to-end "does it actually learn" check: train on a routine,
inject out-of-distribution events, and assert the surprise signal still
separates them. Thresholds are deliberately loose so the test guards against
real regressions (the engine going blind) rather than pinning exact numbers.
"""
from __future__ import annotations

from benchmarks.replay import run_benchmark, run_drift_benchmark


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

    The replay benchmark is deterministic *within* a platform, but it was
    not the same across them: identical code produced recall 0.5556 on
    Linux and 0.7778 to 0.9028 on Windows -- with the same AUC and the
    same precision. The ranking never differed; only the threshold did.

    The cause was ``ANOMALY_MIN_THRESHOLD = 0.10``. Every missed anomaly
    was missed at exactly that value, at a median distance of 0.006 from
    it: the floor sat inside the anomaly cloud, so the last decimal place
    decided a third of the cases. See ``benchmarks/untere_klammer.py``.

    Asserting recall alone would not have caught that -- 0.76 looked fine
    on the machine it was measured on. So the second assertion below
    measures the fragility itself.
    """
    res = run_benchmark(train_days=40, eval_days=12)
    assert res.auc > 0.9, f"ranking separation regressed: AUC={res.auc:.3f}"
    assert res.recall >= 0.9, f"anomaly-flag recall regressed: {res.recall:.3f}"


def test_the_anomaly_threshold_does_not_sit_in_the_anomaly_cloud():
    """How many anomalies are decided by the last decimal place?

    This is the assertion that would have caught issue #1 before it
    reached another machine. Recall says how many were found here;
    this says how many of them were a coin toss.

    An anomaly whose surprise is within ``BORDERLINE`` of the floor is
    not detected, it is guessed: a different ``libm``, a different
    rounding, and it lands on the other side. At the old floor of 0.10
    that was true for 24 of 72 anomalies -- exactly the third that
    changed between platforms.
    """
    from kontinuum_core.predictive_processing import ANOMALY_MIN_THRESHOLD

    BORDERLINE = 0.01     # the gap between the three measured platforms
    BUDGET = 5            # 24 at the old floor, 1 at the current one

    res = run_benchmark(train_days=40, eval_days=12)
    anomalies = [s for s, l in zip(res.scores, res.labels) if l == 1]
    assert anomalies, "no anomalies in the benchmark -- nothing was measured"

    borderline = [s for s in anomalies
                  if abs(s - ANOMALY_MIN_THRESHOLD) <= BORDERLINE]
    assert len(borderline) <= BUDGET, (
        f"{len(borderline)} of {len(anomalies)} anomalies lie within "
        f"{BORDERLINE} of the floor {ANOMALY_MIN_THRESHOLD} -- those are "
        "decided by rounding, not by detection, and this benchmark will "
        "report a different recall on the next machine"
    )
    assert res.precision >= 0.85, f"too many false alarms: precision={res.precision:.3f}"


def test_engine_detects_and_readapts_to_concept_drift():
    """A harder probe than static separation: plasticity.

    Train on routine A, switch the world to routine B. The engine must (a) be
    clearly surprised right after the switch — it noticed the change — and
    (b) re-learn B so surprise settles back down. Observed: 6.8x spike over
    baseline, then back below baseline within a few days. Bounds leave headroom.
    """
    d = run_drift_benchmark(train_days=40, drift_days=20)
    # (a) detection: the new routine is markedly more surprising at first.
    assert d.spike > 2.5 * d.baseline, (
        f"drift not detected: spike={d.spike:.3f} baseline={d.baseline:.3f}")
    # (b) re-adaptation: surprise comes back down as B is learned...
    assert d.adapted < 0.5 * d.spike, (
        f"did not re-adapt: adapted={d.adapted:.3f} spike={d.spike:.3f}")
    # ...all the way back to roughly the old steady state.
    assert d.adapted <= d.baseline * 1.5, (
        f"re-adaptation incomplete: adapted={d.adapted:.3f} baseline={d.baseline:.3f}")


def test_a_restless_but_normal_home_does_not_set_off_the_flag():
    """The other side of the coin -- and the one nobody could measure.

    Every run of this benchmark used to play the same routine to the
    minute, with anomalies mixed in. Precision needs anomalies to be
    defined, so a home with none was not expressible at all. And that is
    exactly the home ``ANOMALY_MIN_THRESHOLD`` was introduced for.

    The consequence: the whole suite passed with the floor at 0.0. The
    upper side was guarded, the lower side was not, and the constant that
    caused issue #1 could be moved anywhere without a test objecting.

    Here the routine wanders by up to twenty minutes -- somebody sleeping
    in, a late bath. Nothing is an anomaly. The measured rate is 3.8%;
    the budget leaves room without letting the detector become jumpy.
    """
    res = run_benchmark(train_days=40, eval_days=12,
                        jitter_minutes=20, with_anomalies=False)

    assert res.n_normal > 200, (
        f"only {res.n_normal} normal events -- with nothing to count, a "
        "false-alarm rate of 0.0 means nothing. This guard exists because "
        "three measurements in a row read 0.000 from an empty list"
    )
    assert res.n_anomaly == 0, "with_anomalies=False still injected anomalies"

    assert res.false_alarm_rate <= 0.05, (
        f"false alarms on a normal-but-restless home: "
        f"{res.false_alarm_rate:.4f}. The flag is firing on somebody "
        "sleeping in, and an alarm that cries wolf gets switched off"
    )


def test_the_flag_still_fires_when_something_really_happens():
    """The non-empty guard for the test above.

    A detector that never fires trivially satisfies a false-alarm budget.
    Read together, the two tests say: quiet when nothing happens, loud
    when something does.
    """
    res = run_benchmark(train_days=40, eval_days=12, jitter_minutes=20)

    assert res.n_anomaly > 0, "no anomalies injected -- nothing was measured"
    fired = sum(1 for f, l in zip(res.flags, res.labels) if f and l == 1)
    assert fired > 0, (
        "not a single injected anomaly set the flag -- the previous test's "
        "clean false-alarm rate is worthless"
    )
