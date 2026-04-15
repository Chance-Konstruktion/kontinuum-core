"""Locus Coeruleus: tracks arousal from event density."""

from __future__ import annotations

import time
from collections import deque


class LocusCoeruleus:
    WINDOW_SECONDS = 60

    def __init__(self):
        self.events = deque(maxlen=2000)
        self.arousal = 0.2

    def observe_event(self):
        now = time.time()
        self.events.append(now)
        self._recompute(now)

    def _recompute(self, now: float):
        recent = [t for t in self.events if now - t <= self.WINDOW_SECONDS]
        density = len(recent) / float(self.WINDOW_SECONDS)
        target = min(1.0, density * 2.5)
        self.arousal = self.arousal * 0.9 + target * 0.1

    def get_arousal(self) -> float:
        return max(0.0, min(1.0, self.arousal))

    def to_dict(self) -> dict:
        return {"events": list(self.events), "arousal": self.arousal}

    def from_dict(self, data: dict):
        self.events = deque(data.get("events", [])[-2000:], maxlen=2000)
        self.arousal = float(data.get("arousal", 0.2))
