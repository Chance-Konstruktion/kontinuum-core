"""Serotonin – mood / patience / temporal discounting.

Biological inspiration: serotonin (raphe nuclei) sets a slow behavioural
baseline. High serotonergic tone is associated with *patience* — a willingness
to wait for a delayed, more certain reward — and behavioural calm; low tone
with impulsivity and exploration. KONTINUUM models the fast neuromodulators
(dopamine, noradrenaline) but had no slow "mood" baseline. Serotonin here is a
slow EMA of how well things are going (positive vs negative feedback, chaos),
exposed as a ``patience`` signal that tunes the Subthalamic Nucleus' readiness
to wait under uncertainty.

Performance: a single EMA. ~0 ms per event.
"""

from __future__ import annotations


class Serotonin:
    ALPHA = 0.02         # slow mood baseline
    BASELINE = 0.5
    ANOMALY_DIP = 0.01   # gentle pull-down on chaotic events

    def __init__(self):
        self.level = self.BASELINE

    def reward(self, positive: bool) -> float:
        target = 1.0 if positive else 0.0
        self.level += self.ALPHA * (target - self.level)
        self.level = max(0.0, min(1.0, self.level))
        return self.level

    def observe(self, anomaly: bool) -> None:
        """A surprising / anomalous event nudges mood down a touch."""
        if anomaly:
            self.level = max(0.0, self.level - self.ANOMALY_DIP * self.level)

    def get_patience(self) -> float:
        return self.level

    @property
    def stats(self) -> dict:
        mood = ("content" if self.level > 0.6
                else "frustrated" if self.level < 0.35 else "neutral")
        return {"level": round(self.level, 3), "mood": mood}

    def to_dict(self) -> dict:
        return {"level": self.level}

    def from_dict(self, data: dict):
        self.level = float(data.get("level", self.BASELINE))
