"""Subthalamic Nucleus (STN) – the global "hold your horses" brake.

Biological inspiration: via the hyperdirect pathway the STN implements a fast
global brake on the basal ganglia. When cortical conflict is high — several
actions competing, none clearly best — the STN raises the decision threshold
and *buys time* instead of letting the fastest option win. The Anterior
Cingulate already measures conflict and damps confidence; the STN turns that
into an explicit recommendation to **wait one more event** under uncertainty
rather than act now. Serotonin (patience) tunes how readily it holds: a
content system happily waits for certainty, a frustrated one acts sooner.

Performance: pure arithmetic, no maps. ~0 ms per event.
"""

from __future__ import annotations


class SubthalamicNucleus:
    CONFLICT_WEIGHT = 0.6
    MARGIN_WEIGHT = 0.4
    # A margin (top1 - top2 confidence) at or above this is "decisive".
    DECISIVE_MARGIN = 0.3
    BASE_HOLD_THRESHOLD = 0.55

    def __init__(self):
        self.brake = 0.0
        self.total_holds = 0
        self.total_evaluated = 0

    def compute_brake(self, conflict_level: float, top_conf: float,
                      runner_up_conf: float) -> float:
        """Combine module conflict with the top-candidate margin into a brake."""
        margin = max(0.0, float(top_conf) - float(runner_up_conf))
        uncertainty = 1.0 - min(1.0, margin / self.DECISIVE_MARGIN)
        raw = (self.CONFLICT_WEIGHT * max(0.0, min(1.0, conflict_level))
               + self.MARGIN_WEIGHT * uncertainty)
        self.brake = max(0.0, min(1.0, raw))
        return self.brake

    def should_hold(self, patience: float = 0.5) -> bool:
        """True when the system should wait instead of acting.

        ``patience`` (serotonin, 0..1) lowers the threshold: a patient system
        holds more readily under uncertainty, an impatient one acts.
        """
        self.total_evaluated += 1
        threshold = self.BASE_HOLD_THRESHOLD - 0.2 * (patience - 0.5) * 2.0
        threshold = max(0.2, min(0.9, threshold))
        hold = self.brake >= threshold
        if hold:
            self.total_holds += 1
        return hold

    @property
    def stats(self) -> dict:
        return {
            "brake": round(self.brake, 3),
            "total_holds": self.total_holds,
            "total_evaluated": self.total_evaluated,
            "hold_rate": round(self.total_holds / max(1, self.total_evaluated), 3),
        }

    def to_dict(self) -> dict:
        return {
            "total_holds": self.total_holds,
            "total_evaluated": self.total_evaluated,
        }

    def from_dict(self, data: dict):
        self.total_holds = int(data.get("total_holds", 0))
        self.total_evaluated = int(data.get("total_evaluated", 0))
