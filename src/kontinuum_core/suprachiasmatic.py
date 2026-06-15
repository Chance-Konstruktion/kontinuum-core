"""Suprachiasmatic Nucleus (SCN) – the *learned* internal clock.

Biological inspiration: the SCN is the master circadian pacemaker. It is not
hard-wired to wall-clock noon — it *entrains* to the actual light/activity
cycle the organism experiences. KONTINUUM's Neurorhythms module already
modulates learning with a fixed cosine peaked at 08:00; that is a population
prior, not *this* household. The SCN learns the home's own activity profile
over the day and emits a small correction (±15 %) centred on 1.0, so a
night-shift household's "morning" plasticity peak drifts to when that home is
actually awake — without touching the population default until enough has been
observed.

Performance: a 24-bin EMA update per event. ~0 ms.
"""

from __future__ import annotations


class SuprachiasmaticNucleus:
    ALPHA = 0.01           # slow entrainment
    MAX_DEVIATION = 0.15   # at most ±15 % nudge to the learning rate
    WARMUP = 200           # events before the SCN influences anything

    def __init__(self):
        # Smoothed probability that an event falls in each hour of the day.
        self.activity = [0.0] * 24
        self.total = 0

    def observe(self, hour: int) -> None:
        h = int(hour) % 24
        a = self.ALPHA
        for i in range(24):
            target = 1.0 if i == h else 0.0
            self.activity[i] += a * (target - self.activity[i])
        self.total += 1

    def phase_gain(self, hour: int) -> float:
        """Correction multiplier (~1.0) for the learning rate at ``hour``.

        >1.0 in hours this home is reliably active, <1.0 in its quiet hours.
        Neutral (1.0) until WARMUP events have been seen.
        """
        if self.total < self.WARMUP:
            return 1.0
        mean = sum(self.activity) / 24.0
        if mean <= 0.0:
            return 1.0
        rel = (self.activity[int(hour) % 24] - mean) / mean
        rel = max(-1.0, min(1.0, rel))
        return 1.0 + self.MAX_DEVIATION * rel

    def peak_hour(self) -> int:
        return max(range(24), key=lambda i: self.activity[i])

    @property
    def stats(self) -> dict:
        return {
            "total": self.total,
            "entrained": self.total >= self.WARMUP,
            "peak_hour": self.peak_hour() if self.total else 0,
        }

    def to_dict(self) -> dict:
        return {"activity": list(self.activity), "total": self.total}

    def from_dict(self, data: dict):
        act = data.get("activity", [])
        if isinstance(act, list) and len(act) == 24:
            self.activity = [float(x) for x in act]
        self.total = int(data.get("total", 0))
