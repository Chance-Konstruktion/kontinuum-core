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

try:
    # Written at build time by setuptools-scm from the Git tag.
    from ._version import version as __version__
except Exception:  # pragma: no cover - source tree without a built _version.py
    try:
        from importlib.metadata import PackageNotFoundError, version as _pkg_version

        try:
            __version__ = _pkg_version("kontinuum-core")
        except PackageNotFoundError:
            __version__ = "0.0.0"
    except Exception:
        __version__ = "0.0.0"
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
