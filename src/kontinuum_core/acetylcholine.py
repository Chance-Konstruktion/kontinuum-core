"""Acetylcholine – expected uncertainty / context-adaptive learning rate.

Biological inspiration (Yu & Dayan, 2005): acetylcholine signals *expected*
uncertainty — the known, irreducible noise of a familiar context — while
noradrenaline signals *unexpected* uncertainty (genuine surprise). KONTINUUM
already has the noradrenergic side (Locus Coeruleus arousal + Predictive
surprise). The cholinergic side was missing: in a context that is *reliably*
noisy (a busy hallway, a chatty multi-sensor room), a single surprising event
should NOT trigger strong learning — the surprise is expected there. This
module keeps a per-context-bucket estimate of expected surprise and lowers the
learning rate where noise is the norm, so the engine stops chasing irreducible
jitter while still learning fast in clean contexts.

Performance: one dict lookup + EMA per event, bounded map. ~0 ms.
"""

from __future__ import annotations


class Acetylcholine:
    ALPHA = 0.05
    MAX_ENTRIES = 2000
    MIN_SAMPLES = 20       # observations before a bucket's estimate is trusted
    MAX_REDUCTION = 0.4    # at most -40 % learning in a reliably noisy bucket

    def __init__(self):
        self.expected = {}   # bucket -> EMA of surprise (expected uncertainty)
        self.counts = {}     # bucket -> n

    def observe(self, bucket: int, surprise: float) -> None:
        e = self.expected.get(bucket)
        if e is None:
            self.expected[bucket] = float(surprise)
        else:
            self.expected[bucket] = e + self.ALPHA * (float(surprise) - e)
        self.counts[bucket] = self.counts.get(bucket, 0) + 1
        if len(self.expected) > self.MAX_ENTRIES:
            self._evict()

    def learn_gain(self, bucket: int) -> float:
        """Multiplier (<=1.0) for the learning rate in ``bucket``.

        Neutral until MIN_SAMPLES; then high expected uncertainty (a reliably
        noisy context) reduces the rate by up to MAX_REDUCTION.
        """
        if self.counts.get(bucket, 0) < self.MIN_SAMPLES:
            return 1.0
        eu = max(0.0, min(1.0, self.expected.get(bucket, 0.0)))
        return 1.0 - self.MAX_REDUCTION * eu

    def mean_expected(self) -> float:
        if not self.expected:
            return 0.0
        return sum(self.expected.values()) / len(self.expected)

    def _evict(self):
        """Drop the least-observed buckets first (least informative)."""
        order = sorted(self.counts.items(), key=lambda kv: kv[1])
        for bucket, _ in order[: len(self.expected) - self.MAX_ENTRIES]:
            self.expected.pop(bucket, None)
            self.counts.pop(bucket, None)

    @property
    def stats(self) -> dict:
        return {
            "tracked_buckets": len(self.expected),
            "mean_expected_uncertainty": round(self.mean_expected(), 3),
        }

    def to_dict(self) -> dict:
        return {
            "expected": {str(k): v for k, v in self.expected.items()},
            "counts": {str(k): v for k, v in self.counts.items()},
        }

    def from_dict(self, data: dict):
        self.expected = {}
        self.counts = {}
        for k, v in data.get("expected", {}).items():
            try:
                self.expected[int(k)] = float(v)
            except (ValueError, TypeError):
                continue
        for k, v in data.get("counts", {}).items():
            try:
                self.counts[int(k)] = int(v)
            except (ValueError, TypeError):
                continue
