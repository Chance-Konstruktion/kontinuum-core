"""Cortisol – the slow systemic stress hormone (HPA axis).

Biological inspiration: cortisol is the body's *slow* stress signal (hours,
not seconds — the fast one, adrenaline / noradrenaline, is the Locus
Coeruleus). It rises under sustained, unpredictable load and shifts the
organism into a conservative, defensive posture. KONTINUUM had only the fast
arousal signal. Cortisol here integrates sustained surprise, anomalies and
user overrides into a slow stress level; when the home has been chaotic lately
(guests, illness, a move) it makes the engine act more cautiously — trusting
learned routines less and lowering autonomous confidence — until calm returns.

Performance: a single EMA. ~0 ms per event.
"""

from __future__ import annotations


class Cortisol:
    BASELINE = 0.1
    RELAX = 0.01            # slow return toward baseline per event
    SURPRISE_GATE = 0.5     # only surprise above this is "stressful"
    SURPRISE_GAIN = 0.1
    ANOMALY_BUMP = 0.05
    OVERRIDE_BUMP = 0.08
    MAX_DAMPING = 0.3       # up to -30 % ranking confidence at full stress

    def __init__(self):
        self.level = self.BASELINE

    def observe(self, surprise: float, anomaly: bool) -> float:
        rise = 0.0
        if surprise > self.SURPRISE_GATE:
            rise += (surprise - self.SURPRISE_GATE) * self.SURPRISE_GAIN
        if anomaly:
            rise += self.ANOMALY_BUMP
        self.level += rise
        # Slow homeostatic relaxation toward baseline.
        self.level += (self.BASELINE - self.level) * self.RELAX
        self.level = max(0.0, min(1.0, self.level))
        return self.level

    def stress_event(self, strength: float = 1.0) -> float:
        """A discrete stressor (e.g. a user override / rejection)."""
        self.level = max(0.0, min(1.0, self.level + self.OVERRIDE_BUMP * strength))
        return self.level

    def damping(self) -> float:
        """Confidence multiplier in [1-MAX_DAMPING, 1.0]; 1.0 at baseline."""
        excess = max(0.0, self.level - self.BASELINE) / (1.0 - self.BASELINE)
        return 1.0 - self.MAX_DAMPING * excess

    @property
    def stats(self) -> dict:
        state = ("calm" if self.level < 0.25
                 else "stressed" if self.level > 0.5 else "elevated")
        return {
            "level": round(self.level, 3),
            "state": state,
            "damping": round(self.damping(), 3),
        }

    def to_dict(self) -> dict:
        return {"level": self.level}

    def from_dict(self, data: dict):
        self.level = float(data.get("level", self.BASELINE))
