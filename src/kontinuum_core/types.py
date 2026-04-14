"""Public data types for KONTINUUM Core."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Observation:
    """Single input event delivered to the engine."""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class Prediction:
    """Engine output describing what the engine expects next."""
    expected: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class MemoryState:
    """Serializable engine state for persistence."""
    data: dict[str, Any] = field(default_factory=dict)
