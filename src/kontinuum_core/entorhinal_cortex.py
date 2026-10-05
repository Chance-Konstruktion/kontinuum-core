"""Entorhinal Cortex: lightweight room-transition map."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Callable, Optional


class EntorhinalCortex:
    def __init__(self, clock: Optional[Callable[[], float]] = None):
        # Clock hygiene (issue #2, §7 step 6, part 1 — on !8's clock, per
        # review 03:19): the engine injects its EXPLICIT clock (Abnahme
        # 13642's pattern — Hypothalamus/LocusCoeruleus carry it the same
        # way); the standard is the wall clock for bare callers.
        self._clock = clock or time.time
        self.transitions = defaultdict(lambda: defaultdict(int))
        self.last_prune_ts = 0.0

    def observe_transition(self, from_room: str, to_room: str):
        if not from_room or not to_room or from_room == to_room:
            return
        self.transitions[from_room][to_room] += 1

    def predict_next_room(self, room: str):
        options = self.transitions.get(room, {})
        if not options:
            return None
        return max(options.items(), key=lambda x: x[1])[0]

    def prune_old_transitions(self, ratio: float = 0.05,
                              now_ts: Optional[float] = None):
        """Prunes the weakest transitions (default 5%) to cap long-term growth.

        Clock hygiene (issue #2, §7 step 6, part 1 — Befund 3 in MR !4
        note 13693, on !8's clock per review 03:19): the stamp runs in
        EVENT time when the caller carries it (the engine's maintenance
        cadence does), so backtests prune on the same schedule a live
        home would. Falls back to the injected clock (the wall clock
        for bare callers — the live path stays what it was)."""
        stamp = now_ts if now_ts is not None else self._clock()
        flat = []
        for from_room, targets in self.transitions.items():
            for to_room, count in targets.items():
                flat.append((from_room, to_room, count))
        if len(flat) < 20:
            self.last_prune_ts = stamp
            return
        flat.sort(key=lambda x: x[2])
        n_remove = max(1, int(len(flat) * ratio))
        for from_room, to_room, _ in flat[:n_remove]:
            if to_room in self.transitions[from_room]:
                del self.transitions[from_room][to_room]
        self.last_prune_ts = stamp

    def to_dict(self) -> dict:
        return {
            "transitions": {k: dict(v) for k, v in self.transitions.items()},
            "last_prune_ts": self.last_prune_ts,
        }

    def from_dict(self, data: dict):
        self.transitions = defaultdict(lambda: defaultdict(int))
        for k, v in data.get("transitions", {}).items():
            self.transitions[k] = defaultdict(int, v)
        self.last_prune_ts = float(data.get("last_prune_ts", 0.0))
