"""Lateral Habenula – anti-reward / disappointment memory.

Biological inspiration: the lateral habenula is the brain's "anti-reward"
hub. It fires when an outcome is *worse* than expected and, via the RMTg,
suppresses dopamine. KONTINUUM already models the positive pole (dopamine
bursts in Neurorhythms, Go/NoGo Q-values in the Basal Ganglia, habit bias in
the Nucleus Accumbens); the habenula adds the missing negative pole: a memory
of *systematic disappointment* per (context, action). When the user keeps
rejecting / overriding the same suggestion in the same context, the habenula
learns to suppress it during ranking — so the system stops nagging instead of
re-proposing a move it has repeatedly been told off for.

Performance: one dict lookup + EMA per feedback, bounded map. ~0 ms per event.
"""

from __future__ import annotations


class LateralHabenula:
    """Tracks per-(state, action) disappointment and suppresses chronic losers."""

    alpha = 0.25
    # Bound the map so a long-lived install can't grow it without limit
    # (mirrors NucleusAccumbens.MAX_ENTRIES).
    MAX_ENTRIES = 4000
    # Below this an entry is considered relieved and dropped.
    RELEASE_FLOOR = 0.02
    # A "still notable" suppression, for active_count() / dashboards.
    ACTIVE_LEVEL = 0.1

    def __init__(self):
        # (state_key, action_key) -> disappointment in [0, 1]
        self.disappointment = {}

    def get_suppression(self, state: str, action: str) -> float:
        return float(self.disappointment.get((state, action), 0.0))

    def punish(self, state: str, action: str, strength: float = 1.0) -> float:
        """A negative outcome (override / reject): raise disappointment."""
        key = (state, action)
        old = self.disappointment.get(key, 0.0)
        new = min(1.0, old + self.alpha * strength * (1.0 - old))
        self.disappointment[key] = new
        if len(self.disappointment) > self.MAX_ENTRIES:
            self._evict()
        return new

    def relieve(self, state: str, action: str, strength: float = 1.0) -> float:
        """A positive outcome: decay disappointment back toward zero."""
        key = (state, action)
        old = self.disappointment.get(key, 0.0)
        if old <= 0.0:
            return 0.0
        new = old - self.alpha * strength * old
        if new < self.RELEASE_FLOOR:
            self.disappointment.pop(key, None)
            return 0.0
        self.disappointment[key] = new
        return new

    def active_count(self) -> int:
        return sum(1 for v in self.disappointment.values() if v >= self.ACTIVE_LEVEL)

    def _evict(self):
        """Bound memory: keep the strongest (most-disappointing) entries."""
        keep = dict(sorted(self.disappointment.items(),
                           key=lambda kv: kv[1], reverse=True)[:self.MAX_ENTRIES])
        self.disappointment = keep

    @property
    def stats(self) -> dict:
        vals = list(self.disappointment.values())
        return {
            "tracked": len(vals),
            "active_suppressions": self.active_count(),
            "max_suppression": round(max(vals), 3) if vals else 0.0,
        }

    def to_dict(self) -> dict:
        return {
            "disappointment": {
                f"{k[0]}||{k[1]}": v for k, v in self.disappointment.items()
            }
        }

    def from_dict(self, data: dict):
        self.disappointment = {}
        for key, val in data.get("disappointment", {}).items():
            if "||" in key:
                s, a = key.split("||", 1)
                self.disappointment[(s, a)] = float(val)
