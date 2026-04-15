"""Insula module for KONTINUUM Core.

Biological inspiration: The insula integrates bodily signals into a
feeling of the current state. Here it detects mode:
sleeping, waking_up, active, relaxing, cooking, away.

Uses circadian priors (day/night via update_sun) for biologically
plausible mode detection.
"""

import logging
import time
from collections import deque

_LOGGER = logging.getLogger(__name__)


class Mode:
    SLEEPING = "sleeping"
    WAKING_UP = "waking_up"
    ACTIVE = "active"
    RELAXING = "relaxing"
    COOKING = "cooking"
    AWAY = "away"


# Weights: (semantic, state) → {mode: weight}
MODE_SIGNALS = {
    ("light", "off"): {Mode.SLEEPING: 0.3, Mode.AWAY: 0.2},
    ("light", "on"): {Mode.ACTIVE: 0.2, Mode.WAKING_UP: 0.2},
    ("media", "playing"): {Mode.RELAXING: 0.4, Mode.ACTIVE: 0.1},
    ("media", "off"): {Mode.SLEEPING: 0.2},
    ("media", "standby"): {Mode.RELAXING: 0.1},
    ("motion", "on"): {Mode.ACTIVE: 0.15},
    ("motion", "off"): {Mode.SLEEPING: 0.05, Mode.AWAY: 0.05},
    ("door", "on"): {Mode.ACTIVE: 0.2, Mode.WAKING_UP: 0.1},
    ("tracker", "home"): {Mode.ACTIVE: 0.3},
    ("tracker", "away"): {Mode.AWAY: 0.8},
    ("spatial", "entered"): {Mode.ACTIVE: 0.2},
    ("spatial", "left"): {Mode.ACTIVE: 0.1},
    ("switch", "on"): {Mode.ACTIVE: 0.15},
    ("switch", "off"): {Mode.RELAXING: 0.1},
    ("climate", "heating"): {Mode.ACTIVE: 0.05},
    ("climate", "off"): {Mode.AWAY: 0.1},
    ("cpu", "high"): {Mode.ACTIVE: 0.35},
    ("gpu", "high"): {Mode.ACTIVE: 0.3, Mode.RELAXING: 0.1},
    ("steps", "high"): {Mode.ACTIVE: 0.4},
    ("steps", "active"): {Mode.ACTIVE: 0.2},
    ("heartrate", "high"): {Mode.ACTIVE: 0.35},
    ("sleep", "deep"): {Mode.SLEEPING: 0.6},
    ("bed_presence", "on"): {Mode.SLEEPING: 0.35, Mode.RELAXING: 0.2},
    ("co2", "high"): {Mode.AWAY: 0.15, Mode.SLEEPING: 0.05},
    ("gaming", "on"): {Mode.ACTIVE: 0.3, Mode.RELAXING: 0.15},
}

ROOM_MODE_HINTS = {
    "bedroom": {Mode.SLEEPING: 0.2, Mode.RELAXING: 0.1},
    "kitchen": {Mode.COOKING: 0.3, Mode.ACTIVE: 0.1},
    "livingroom": {Mode.RELAXING: 0.2},
    "bathroom": {Mode.WAKING_UP: 0.15},
    "office": {Mode.ACTIVE: 0.2},
    "outdoor": {Mode.ACTIVE: 0.1, Mode.AWAY: 0.05},
    "garage": {Mode.ACTIVE: 0.1, Mode.AWAY: 0.1},
}


class Insula:
    """Mode detector for KONTINUUM."""

    ALL_MODES = [
        Mode.SLEEPING, Mode.WAKING_UP, Mode.ACTIVE,
        Mode.RELAXING, Mode.COOKING, Mode.AWAY,
    ]

    MODE_INDEX = {
        Mode.SLEEPING: 0, Mode.WAKING_UP: 1, Mode.ACTIVE: 2,
        Mode.RELAXING: 3, Mode.COOKING: 4, Mode.AWAY: 5,
    }

    TRANSITION_COOLDOWN = 300
    MIN_CONFIDENCE = 0.40
    DECAY_PER_MINUTE = 0.02
    WINDOW_SIZE = 15
    INACTIVITY_THRESHOLD = 1800

    def __init__(self):
        self.mode_probs = {m: 0.0 for m in self.ALL_MODES}
        self.mode_probs[Mode.ACTIVE] = 0.3
        self.current_mode = Mode.ACTIVE
        self.last_mode_change = 0
        self.total_mode_changes = 0
        self.event_window = deque(maxlen=self.WINDOW_SIZE)
        self.last_event_time = 0
        self._is_daylight = True

    def process(self, semantic: str, state: str, room: str,
                token: str = "") -> dict:
        """Processes an event for mode detection.

        Returns a transition token dict on mode change, else None.
        """
        now = time.time()
        self.last_event_time = now
        self.event_window.append((semantic, state, room, now))

        sig_key = (semantic, state)
        if sig_key in MODE_SIGNALS:
            for mode, weight in MODE_SIGNALS[sig_key].items():
                self.mode_probs[mode] += weight

        self._apply_room_hints(room)
        self._apply_circadian_priors()
        self._apply_decay(now)
        self._check_inactivity(now)

        total = sum(self.mode_probs.values())
        if total > 0:
            for mode in self.ALL_MODES:
                self.mode_probs[mode] /= total

        best_mode = max(self.mode_probs, key=self.mode_probs.get)
        best_conf = self.mode_probs[best_mode]

        if (best_mode != self.current_mode and
                best_conf >= self.MIN_CONFIDENCE and
                (now - self.last_mode_change) >= self.TRANSITION_COOLDOWN):

            old_mode = self.current_mode
            self.current_mode = best_mode
            self.last_mode_change = now
            self.total_mode_changes += 1

            _LOGGER.info("Insula: %s → %s (conf=%.2f)", old_mode, best_mode, best_conf)

            return {
                "token": f"mode.{best_mode}",
                "room": "house",
                "semantic": "mode",
                "state": best_mode,
            }

        return None

    def update_sun(self, is_daylight: bool):
        """Updates daylight state for circadian priors."""
        self._is_daylight = is_daylight

    def _apply_circadian_priors(self):
        if not self._is_daylight:
            self.mode_probs[Mode.SLEEPING] += 0.15
            self.mode_probs[Mode.RELAXING] += 0.05
        else:
            self.mode_probs[Mode.ACTIVE] += 0.05

    def _apply_room_hints(self, room: str):
        if room in ROOM_MODE_HINTS:
            for mode, bonus in ROOM_MODE_HINTS[room].items():
                self.mode_probs[mode] += bonus * 0.5

    def _apply_decay(self, now: float):
        if len(self.event_window) < 2:
            return
        oldest = self.event_window[0][3]
        elapsed = (now - oldest) / 60.0
        if elapsed > 0:
            for mode in self.ALL_MODES:
                self.mode_probs[mode] *= max(0.0, 1.0 - self.DECAY_PER_MINUTE * elapsed * 0.1)

    def _check_inactivity(self, now: float):
        if self.last_event_time > 0:
            inactive = now - self.last_event_time
            if inactive > self.INACTIVITY_THRESHOLD:
                self.mode_probs[Mode.SLEEPING] += 0.1
                self.mode_probs[Mode.AWAY] += 0.05

    def get_mode_context(self) -> list:
        mode_idx = self.MODE_INDEX.get(self.current_mode, 2) / 5.0
        confidence = self.mode_probs.get(self.current_mode, 0.0)
        duration = min((time.time() - self.last_mode_change) / 3600.0, 1.0) if self.last_mode_change else 0.0
        return [mode_idx, confidence, duration]

    def get_mode_index(self) -> int:
        return self.MODE_INDEX.get(self.current_mode, 2)

    def to_dict(self) -> dict:
        return {
            "current_mode": self.current_mode,
            "mode_probs": self.mode_probs,
            "last_mode_change": self.last_mode_change,
            "total_mode_changes": self.total_mode_changes,
            "last_event_time": self.last_event_time,
        }

    def from_dict(self, data: dict):
        self.current_mode = data.get("current_mode", Mode.ACTIVE)
        self.mode_probs = data.get("mode_probs", {m: 0.0 for m in self.ALL_MODES})
        for m in self.ALL_MODES:
            if m not in self.mode_probs:
                self.mode_probs[m] = 0.0
        self.last_mode_change = data.get("last_mode_change", 0)
        self.total_mode_changes = data.get("total_mode_changes", 0)
        self.last_event_time = data.get("last_event_time", 0)

    @property
    def stats(self) -> dict:
        return {
            "current_mode": self.current_mode,
            "confidence": round(self.mode_probs.get(self.current_mode, 0), 3),
            "total_mode_changes": self.total_mode_changes,
            "mode_probs": {m: round(p, 3) for m, p in self.mode_probs.items() if p > 0.01},
        }
