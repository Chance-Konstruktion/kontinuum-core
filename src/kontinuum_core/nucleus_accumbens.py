"""Nucleus Accumbens: fast reward amplifier for action habits."""

from __future__ import annotations


class NucleusAccumbens:
    alpha = 0.2

    def __init__(self):
        self.values = {}
        self.success_counts = {}

    def get_bias(self, state: str, action: str) -> float:
        return float(self.values.get((state, action), 0.0))

    def reinforce(self, state: str, action: str, reward: float) -> float:
        key = (state, action)
        old = self.values.get(key, 0.0)
        new = old + self.alpha * (float(reward) - old)
        self.values[key] = max(-1.0, min(1.5, new))
        if reward > 0:
            self.success_counts[key] = self.success_counts.get(key, 0) + 1
        return self.values[key]

    def to_dict(self) -> dict:
        return {
            "values": {f"{k[0]}||{k[1]}": v for k, v in self.values.items()},
            "success_counts": {f"{k[0]}||{k[1]}": v for k, v in self.success_counts.items()},
        }

    def from_dict(self, data: dict):
        self.values = {}
        self.success_counts = {}
        for key, val in data.get("values", {}).items():
            if "||" in key:
                s, a = key.split("||", 1)
                self.values[(s, a)] = val
        for key, val in data.get("success_counts", {}).items():
            if "||" in key:
                s, a = key.split("||", 1)
                self.success_counts[(s, a)] = val
