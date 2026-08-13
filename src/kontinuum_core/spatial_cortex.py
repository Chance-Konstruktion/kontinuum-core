"""Spatial Cortex for KONTINUUM Core.

Learns room sequences (A → B), predicts the next room and applies
three-layer anti-bounce (hysteresis, confirmation, cooldown).
"""

import logging
import time

_LOGGER = logging.getLogger(__name__)

SPATIAL_SEMANTICS = {"motion", "presence", "door", "tracker", "co2", "bed_presence"}

SIGNAL_WEIGHTS = {
    "tracker": 0.9,
    "presence": 0.7,
    "motion": 0.5,
    "door": 0.2,
    "co2": 0.25,
    "bed_presence": 0.8,
}

DECAY_RATES = {
    "tracker": 900,    # 15 min
    "presence": 600,   # 10 min
    "motion": 300,     # 5 min
    "door": 180,       # 3 min
    "co2": 900,        # 15 min
    "bed_presence": 1200,  # 20 min
}


class RoomState:
    def __init__(self, name: str):
        self.name = name
        self.probability = 0.0
        self.signals = {}
        self.motion_count = 0
        self.last_motion_time = 0


class SpatialCortex:
    TRANSITION_COOLDOWN = 60
    HYSTERESIS_FACTOR = 1.2
    CONFIRMATION_TIME = 30
    ACTIVE_COOLDOWN = 1800
    ACTIVE_MIN_MOTIONS = 5
    OCCUPIED_THRESHOLD = 0.25

    # Pseudo rooms ignored when determining the current room
    IGNORED_ROOMS = {"area_unknown"}

    def __init__(self):
        self.rooms = {}
        self.current_room = "unknown"
        self.last_transition_time = 0
        self.proposed_room = None
        self.proposed_since = 0
        self.total_transitions = 0
        self.bounces_prevented = 0
        self._last_active_token_time = 0
        # "room_a→room_b" → count
        self.movement_memory = {}

    def is_spatial_signal(self, semantic: str) -> bool:
        return semantic in SPATIAL_SEMANTICS

    def absorb(self, room: str, semantic: str, state: str,
               entity_id: str = "") -> list:
        if not room or room == "unknown" or room in self.IGNORED_ROOMS:
            return []

        now = time.time()

        if room not in self.rooms:
            self.rooms[room] = RoomState(room)

        rs = self.rooms[room]

        active_signal = state in ("on", "home", "open", "detected")
        if semantic == "co2" and state in ("elevated", "high"):
            active_signal = True
        if semantic == "bed_presence" and state in ("on", "occupied"):
            active_signal = True

        if active_signal:
            weight = SIGNAL_WEIGHTS.get(semantic, 0.2)
            sig_key = f"{entity_id}_{semantic}"
            rs.signals[sig_key] = (weight, now)

            if semantic == "motion":
                rs.motion_count += 1
                rs.last_motion_time = now
        elif state in ("off", "away", "closed", "clear", "not_home", "good"):
            sig_key = f"{entity_id}_{semantic}"
            rs.signals.pop(sig_key, None)

            if semantic == "tracker" and room == self.current_room:
                rs.probability *= 0.5

        self._update_probabilities(now)

        tokens = []
        transition = self._detect_transition(now)
        if transition:
            tokens.extend(transition)

        active = self._check_active_token(room, now)
        if active:
            tokens.append(active)

        return tokens

    def _update_probabilities(self, now: float):
        for room_name, rs in self.rooms.items():
            prob = 0.0
            expired = []

            for sig_key, (weight, sig_time) in rs.signals.items():
                semantic = sig_key.split("_")[-1] if "_" in sig_key else "motion"
                max_age = DECAY_RATES.get(semantic, 180)
                age = now - sig_time

                if age > max_age:
                    expired.append(sig_key)
                    continue

                freshness = 1.0 - (age / max_age)
                prob += weight * freshness

            for key in expired:
                del rs.signals[key]

            rs.probability = min(1.0, prob)

    def _detect_transition(self, now: float) -> list:
        if not self.rooms:
            return []

        valid_rooms = [r for r in self.rooms.values() if r.name not in self.IGNORED_ROOMS]
        if not valid_rooms:
            return []
        best_room = max(valid_rooms, key=lambda r: r.probability)

        if best_room.probability < self.OCCUPIED_THRESHOLD:
            return []

        if best_room.name == self.current_room:
            self.proposed_room = None
            return []

        current_prob = self.rooms[self.current_room].probability if self.current_room in self.rooms else 0.0

        # Layer 1: hysteresis
        if best_room.probability < current_prob * self.HYSTERESIS_FACTOR:
            self.bounces_prevented += 1
            return []

        # Layer 2: confirmation
        if self.proposed_room != best_room.name:
            self.proposed_room = best_room.name
            self.proposed_since = now
            return []

        if (now - self.proposed_since) < self.CONFIRMATION_TIME:
            return []

        # Layer 3: cooldown
        if (now - self.last_transition_time) < self.TRANSITION_COOLDOWN:
            self.bounces_prevented += 1
            return []

        old_room = self.current_room
        new_room = best_room.name
        self.current_room = new_room
        self.last_transition_time = now
        self.total_transitions += 1
        self.proposed_room = None

        # Learn movement pattern
        if old_room != "unknown":
            move_key = f"{old_room}→{new_room}"
            self.movement_memory[move_key] = self.movement_memory.get(move_key, 0) + 1

        tokens = []
        if old_room != "unknown":
            tokens.append({
                "token": f"person.left.{old_room}",
                "room": old_room,
                "semantic": "spatial",
                "state": "left",
            })
        tokens.append({
            "token": f"person.entered.{new_room}",
            "room": new_room,
            "semantic": "spatial",
            "state": "entered",
        })

        _LOGGER.info("Spatial: %s → %s (prob=%.2f)", old_room, new_room, best_room.probability)
        return tokens

    def _check_active_token(self, room: str, now: float) -> dict:
        if room != self.current_room:
            return None

        rs = self.rooms.get(room)
        if not rs:
            return None

        if rs.motion_count < self.ACTIVE_MIN_MOTIONS:
            return None

        if (now - self._last_active_token_time) < self.ACTIVE_COOLDOWN:
            return None

        self._last_active_token_time = now
        rs.motion_count = 0

        return {
            "token": f"person.active.{room}",
            "room": room,
            "semantic": "spatial",
            "state": "active",
        }

    def get_current_location(self) -> str:
        return self.current_room

    def predict_next_room(self) -> list:
        if self.current_room == "unknown":
            return []

        prefix = f"{self.current_room}→"
        candidates = []
        for key, count in self.movement_memory.items():
            if key.startswith(prefix) and count >= 2:
                target = key.split("→", 1)[1]
                candidates.append((target, count))

        total = sum(c for _, c in candidates)
        if total == 0:
            return []

        return [(room, round(count / total, 3))
                for room, count in sorted(candidates, key=lambda x: -x[1])][:3]

    def to_dict(self) -> dict:
        rooms_data = {}
        for name, rs in self.rooms.items():
            rooms_data[name] = {
                "probability": rs.probability,
                "motion_count": rs.motion_count,
                "last_motion_time": rs.last_motion_time,
            }
        return {
            "current_room": self.current_room,
            "rooms": rooms_data,
            "last_transition_time": self.last_transition_time,
            "total_transitions": self.total_transitions,
            "bounces_prevented": self.bounces_prevented,
            "movement_memory": self.movement_memory,
        }

    def from_dict(self, data: dict):
        self.current_room = data.get("current_room", "unknown")
        self.last_transition_time = data.get("last_transition_time", 0)
        self.total_transitions = data.get("total_transitions", 0)
        self.bounces_prevented = data.get("bounces_prevented", 0)
        self.movement_memory = data.get("movement_memory", {})
        for name, rd in data.get("rooms", {}).items():
            rs = RoomState(name)
            rs.probability = rd.get("probability", 0)
            rs.motion_count = rd.get("motion_count", 0)
            rs.last_motion_time = rd.get("last_motion_time", 0)
            self.rooms[name] = rs

    @property
    def stats(self) -> dict:
        presence = {}
        for name, rs in self.rooms.items():
            if rs.probability > 0.01:
                presence[name] = round(rs.probability, 3)
        next_room = self.predict_next_room()
        return {
            "current_room": self.current_room,
            "total_transitions": self.total_transitions,
            "bounces_prevented": self.bounces_prevented,
            "presence_map": presence,
            "predicted_next": next_room[0] if next_room else None,
            "movement_patterns": len(self.movement_memory),
        }
