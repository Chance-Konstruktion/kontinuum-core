"""Interval Timing – the brain's stopwatch for recurring events.

Biological inspiration: distinct from the circadian clock (Suprachiasmatic
Nucleus = time-of-day), interval timing is the brain's *duration* sense — the
striatal / cerebellar "pacemaker-accumulator" that learns how long things
usually take and anticipates the next occurrence. It is what lets you feel that
the bins go out "about now" or the robot vacuum runs "every few weeks" —
cadences a sequence model (hippocampal n-grams) cannot capture, because those
events are neither adjacent in the event stream nor tied to a fixed clock hour
or weekday.

For each recurring event token it keeps an EMA of the inter-event interval and
its relative spread (regularity), then flags a token as *due* once enough
regular observations have accrued and roughly one interval has elapsed since it
last fired. Due tokens can be re-surfaced as low-priority predictions so an
overdue routine reappears on its own — without polluting the surprise signal
(injection happens after surprise is computed).

Performance: one dict lookup + a few EMAs per event, bounded map. ~0 ms.
"""

from __future__ import annotations

from typing import List, Optional, Tuple


class IntervalTiming:
    ALPHA = 0.3              # intervals are few → fairly responsive EMA
    MIN_OBSERVATIONS = 3     # intervals seen before a cadence is trusted
    MIN_INTERVAL = 600.0     # ignore sub-10-min repeats (bursts / flapping)
    MAX_REL_SPREAD = 0.4     # mad/mean above this → too irregular to predict
    DUE_RATIO_START = 0.75   # start considering "due" at 75 % of the interval
    MAX_ENTRIES = 2000

    def __init__(self):
        # token_id -> {"last": ts, "mean": s, "mad": s, "count": n}
        self.timers = {}

    # ------------------------------------------------------------------
    def observe(self, token_id: int, now: float) -> None:
        """Record an occurrence of ``token_id`` at epoch ``now`` (seconds)."""
        rec = self.timers.get(token_id)
        if rec is None:
            self.timers[token_id] = {"last": float(now), "mean": 0.0,
                                     "mad": 0.0, "count": 0}
            if len(self.timers) > self.MAX_ENTRIES:
                self._evict()
            return

        interval = float(now) - rec["last"]
        rec["last"] = float(now)
        # Sub-floor gaps are bursts / the same occurrence, not a cadence.
        if interval < self.MIN_INTERVAL:
            return

        if rec["count"] == 0:
            rec["mean"] = interval
            rec["mad"] = 0.0
        else:
            a = self.ALPHA
            rec["mean"] += a * (interval - rec["mean"])
            rec["mad"] += a * (abs(interval - rec["mean"]) - rec["mad"])
        rec["count"] += 1

    # ------------------------------------------------------------------
    def due_score(self, token_id: int, now: float) -> float:
        """How "due" ``token_id`` is right now, in [0, 1].

        0 until the cadence is trusted (enough regular observations) and at
        least ~75 % of the typical interval has elapsed; ramps to 1 around the
        expected time and *stays* 1 while overdue. Irregular cadences score 0.
        """
        rec = self.timers.get(token_id)
        if rec is None or rec["count"] < self.MIN_OBSERVATIONS or rec["mean"] <= 0:
            return 0.0
        rel_spread = rec["mad"] / rec["mean"]
        if rel_spread > self.MAX_REL_SPREAD:
            return 0.0
        ratio = (float(now) - rec["last"]) / rec["mean"]
        if ratio < self.DUE_RATIO_START:
            return 0.0
        regularity = 1.0 - min(1.0, rel_spread / self.MAX_REL_SPREAD)
        dueness = min(1.0, ratio)
        return max(0.0, min(1.0, regularity * dueness))

    def due_prediction(self, now: float, exclude: Optional[int] = None,
                       threshold: float = 0.5) -> Optional[Tuple]:
        """Top overdue, regular token as an injectable prediction tuple.

        Returns ``(token_id, score, score, "interval_timing", count)`` or None.
        ``exclude`` skips the token currently firing (it is not "due", it is
        happening).
        """
        best_id, best_score = None, threshold
        for token_id, rec in self.timers.items():
            if token_id == exclude:
                continue
            score = self.due_score(token_id, now)
            if score > best_score:
                best_id, best_score = token_id, score
        if best_id is None:
            return None
        return (best_id, best_score, best_score, "interval_timing",
                self.timers[best_id]["count"])

    def due_count(self, now: float, threshold: float = 0.5) -> int:
        return sum(1 for tid in self.timers
                   if self.due_score(tid, now) >= threshold)

    def predict_next_ts(self, token_id: int) -> Optional[float]:
        rec = self.timers.get(token_id)
        if rec is None or rec["count"] < self.MIN_OBSERVATIONS:
            return None
        return rec["last"] + rec["mean"]

    # ------------------------------------------------------------------
    def _evict(self):
        """Bound memory: drop the least-observed timers first."""
        order = sorted(self.timers.items(), key=lambda kv: kv[1]["count"])
        for token_id, _ in order[: len(self.timers) - self.MAX_ENTRIES]:
            self.timers.pop(token_id, None)

    @property
    def stats(self) -> dict:
        regular = sum(1 for r in self.timers.values()
                      if r["count"] >= self.MIN_OBSERVATIONS
                      and r["mean"] > 0 and r["mad"] / r["mean"] <= self.MAX_REL_SPREAD)
        return {"tracked": len(self.timers), "regular_cadences": regular}

    def to_dict(self) -> dict:
        return {"timers": {str(k): v for k, v in self.timers.items()}}

    def from_dict(self, data: dict):
        self.timers = {}
        for k, v in data.get("timers", {}).items():
            try:
                self.timers[int(k)] = {
                    "last": float(v.get("last", 0.0)),
                    "mean": float(v.get("mean", 0.0)),
                    "mad": float(v.get("mad", 0.0)),
                    "count": int(v.get("count", 0)),
                }
            except (ValueError, TypeError, AttributeError):
                continue
