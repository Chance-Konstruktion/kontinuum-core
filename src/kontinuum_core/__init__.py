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
from .priors import (
    HOME_PRIOR_PROMPT,
    parse_home_prior,
    seed_engine_from_prior,
)

__version__ = "0.3.0"
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
    # LLM-seeded priors — day-1 head start.
    "HOME_PRIOR_PROMPT",
    "parse_home_prior",
    "seed_engine_from_prior",
]
