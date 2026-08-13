"""BDNF – use-dependent structural protection (the "vitamin" layer).

Biological inspiration: BDNF (brain-derived neurotrophic factor) is the growth
factor behind use-dependent plasticity — synapses that are repeatedly *used
and rewarded* are structurally strengthened and protected, while idle ones are
pruned. It is the closest analogue to the "keep the brain healthy" role people
associate with vitamins / nootropics: not a moment-to-moment signal, but
background maintenance of what has proven worth keeping.

KONTINUUM forgets on purpose (decay + sleep-consolidation pruning), which is
healthy — but blanket forgetting can erode a *proven* routine during a quiet
spell. BDNF tracks which action tokens belong to repeatedly-correct reflexes
and marks them protected, so consolidation floors (instead of deleting) the
transitions that feed them. Trophic support fades slowly, so a routine that
stops being used / correct eventually loses protection.

Performance: one dict op per reinforcement, bounded map. ~0 ms.
"""

from __future__ import annotations


class Bdnf:
    GAIN = 0.3
    DECAY = 0.98               # applied once per consolidation cycle
    PROTECT_THRESHOLD = 0.5
    DROP_FLOOR = 0.05
    MAX_ENTRIES = 1000

    def __init__(self):
        # token_id -> trophic support in [0, 1]
        self.trophic = {}

    def reinforce(self, token_id: int, amount: float = 1.0) -> float:
        v = self.trophic.get(token_id, 0.0)
        v = min(1.0, v + self.GAIN * amount * (1.0 - v))
        self.trophic[token_id] = v
        if len(self.trophic) > self.MAX_ENTRIES:
            self._evict()
        return v

    def decay_all(self) -> None:
        """Fade trophic support (call once per consolidation cycle)."""
        for k in list(self.trophic.keys()):
            nv = self.trophic[k] * self.DECAY
            if nv < self.DROP_FLOOR:
                del self.trophic[k]
            else:
                self.trophic[k] = nv

    def is_protected(self, token_id: int) -> bool:
        return self.trophic.get(token_id, 0.0) >= self.PROTECT_THRESHOLD

    def protection(self, token_id: int) -> float:
        return float(self.trophic.get(token_id, 0.0))

    def protected_count(self) -> int:
        return sum(1 for v in self.trophic.values() if v >= self.PROTECT_THRESHOLD)

    def _evict(self):
        """Bound memory: keep the most trophically-supported tokens."""
        keep = dict(sorted(self.trophic.items(),
                           key=lambda kv: kv[1], reverse=True)[:self.MAX_ENTRIES])
        self.trophic = keep

    @property
    def stats(self) -> dict:
        return {
            "tracked": len(self.trophic),
            "protected": self.protected_count(),
        }

    def to_dict(self) -> dict:
        return {"trophic": {str(k): v for k, v in self.trophic.items()}}

    def from_dict(self, data: dict):
        self.trophic = {}
        for k, v in data.get("trophic", {}).items():
            try:
                self.trophic[int(k)] = float(v)
            except (ValueError, TypeError):
                continue
