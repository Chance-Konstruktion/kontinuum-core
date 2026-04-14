"""Scheduler protocol for KONTINUUM Core.

The HA integration provides an adapter wrapping async_track_time_interval;
non-HA users can pass any object with the same shape (e.g. a threading-
based scheduler for tests).
"""
from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class Scheduler(Protocol):
    """Minimal scheduler contract used by core modules."""

    def schedule_interval(self, callback: Callable[[], None], seconds: float) -> Callable[[], None]:
        """Register `callback` to be invoked every `seconds`. Return a cancel-handle."""
        ...
