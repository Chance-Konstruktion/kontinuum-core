"""EntorhinalCortex module for KONTINUUM Core (Home-Assistant-free)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

_LOGGER = logging.getLogger(__name__)


@dataclass
class EntorhinalCortexState:
    """Mutable internal state for EntorhinalCortex."""

    values: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)


class EntorhinalCortex:
    """Generic, synchronous placeholder implementation for EntorhinalCortex."""

    def __init__(self) -> None:
        self.state = EntorhinalCortexState()

    def reset(self) -> None:
        self.state = EntorhinalCortexState()

    def update(self, signal: Dict[str, Any] | None = None) -> Dict[str, Any]:
        signal = signal or {}
        self.state.values.update(signal)
        self.state.history.append(dict(signal))
        return dict(self.state.values)
