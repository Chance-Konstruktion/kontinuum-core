"""KONTINUUM Core - neuro-inspired learning engine (HA-free)."""
from .engine import KontinuumEngine
from .scheduler import Scheduler
from .types import MemoryState, Observation, Prediction
from .llm import (
    build_llm_context,
    render_llm_context,
    extract_json,
    normalize_proposal,
)

__version__ = "0.2.0"
__all__ = [
    "KontinuumEngine",
    "Scheduler",
    "Observation",
    "Prediction",
    "MemoryState",
    # LLM bridge — the engine's language-model data contract.
    "build_llm_context",
    "render_llm_context",
    "extract_json",
    "normalize_proposal",
]
