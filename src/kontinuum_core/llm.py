"""LLM bridge — a stable, language-model-friendly data contract.

KONTINUUM is a sub-symbolic engine; an LLM is the optional language / reasoning
layer on top of it (see the ``cortex`` layer in the Home-Assistant integration).
For that pairing to work well, two data paths have to be *clean*:

1. **Engine → LLM** (:func:`build_llm_context`, :func:`render_llm_context`):
   expose what the engine knows in a way a model can reason over reliably —
   explicit 0–1 scales with their meaning, the **anomaly / surprise signal**,
   the top-k expected next events, and the learning maturity. A flat dump of
   un-explained numbers (``dopamine: 0.123``) is hard for a model to use; this
   labels every value and states its scale.

2. **LLM → Engine** (:func:`extract_json`, :func:`normalize_proposal`): turn a
   model's reply — which in practice arrives wrapped in ``` fences, padded with
   prose, or with ``priority`` as the string ``"80"`` — into a strict, validated
   proposal the engine can act on without its consensus arithmetic breaking.

HA-free, pure stdlib. Nothing here imports Home Assistant or an HTTP client, so
it is fully unit-tested in the core suite and reusable by every integration.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union

# Bumped when the shape of build_llm_context() changes incompatibly.
CONTEXT_SCHEMA_VERSION = 1

# The strict shape every LLM proposal is coerced into.
PROPOSAL_FIELDS = ("agent", "action", "entity_id", "reason", "priority", "veto")


# ===========================================================================
# Engine → LLM : context export
# ===========================================================================
# Engine attribute -> candidate keys in ha-kontinuum's brain dict, so the
# context export works whether handed a KontinuumEngine or that plain dict of
# module instances (the Pro integration drives modules directly, not the engine).
_BRAIN_KEYS = {
    "predictive": ("predictive",),
    "hippocampus": ("hippocampus",),
    "thalamus": ("thalamus",),
    "spatial_cortex": ("spatial_cortex", "spatial"),
    "insula": ("insula",),
    "hypothalamus": ("hypothalamus",),
    "basal_ganglia": ("basal_ganglia",),
    "cerebellum": ("cerebellum",),
}


def _as_engine_view(source):
    """Normalize a KontinuumEngine OR a brain dict into an attribute view."""
    if not isinstance(source, Mapping):
        return source
    view = SimpleNamespace()
    for attr, keys in _BRAIN_KEYS.items():
        setattr(view, attr, next((source[k] for k in keys if k in source), None))
    view.tick_count = source.get("tick_count", 0)
    return view


def _maturity(events: int) -> str:
    if events < 100:
        return "cold_start"
    if events < 2000:
        return "warming"
    return "mature"


def _safe(fn, default=None):
    """Call ``fn`` (a zero-arg lambda) and swallow any failure.

    The context export must never raise just because a module's API drifted or a
    value is momentarily unavailable — a degraded context is better than none.
    """
    try:
        return fn()
    except Exception:  # noqa: BLE001 — robustness is the whole point here
        return default


def _expected_next(engine, top_k: int) -> List[Dict[str, Any]]:
    def _build():
        ctx = (
            list(engine.thalamus.encode_time_context(datetime.now(timezone.utc)))
            + list(engine.hypothalamus.get_context_vector())
            + list(engine.insula.get_mode_context())
        )
        preds = engine.hippocampus.predict(ctx, top_k=top_k) or []
        out: List[Dict[str, Any]] = []
        for p in preds[:top_k]:
            conf = p[2] if len(p) > 2 else (p[1] if len(p) > 1 else 0.0)
            out.append({
                "event": _safe(lambda: engine.thalamus.decode_token(p[0]),
                               str(p[0])),
                "confidence": round(float(conf), 3),
            })
        return out
    return _safe(_build, []) or []


def build_llm_context(engine, *, top_k: int = 3) -> Dict[str, Any]:
    """Serialize the engine state into an LLM-optimized, versioned snapshot.

    ``engine`` may be a :class:`KontinuumEngine` or a brain dict of module
    instances (as the ha-kontinuum integration uses). Defensive by design: any
    individual field that can't be read degrades to a safe default rather than
    raising. The output is plain JSON-able data.
    """
    engine = _as_engine_view(engine)
    pred = getattr(engine, "predictive", None)
    surprise = float(_safe(lambda: pred.current_surprise, 0.0) or 0.0)
    threshold = float(_safe(lambda: pred.anomaly_threshold(), 0.0) or 0.0)
    events = int(_safe(lambda: engine.hippocampus.total_events, 0) or 0)

    mode = _safe(lambda: engine.insula.current_mode)
    mode = _safe(lambda: getattr(mode, "value", None)) or (
        str(mode) if mode is not None else None
    )

    return {
        "schema_version": CONTEXT_SCHEMA_VERSION,
        "engine": {
            "tick_count": int(_safe(lambda: engine.tick_count, 0) or 0),
            "events_seen": events,
            "learning_maturity": _maturity(events),
        },
        "now": {
            "mode": mode,
            "room": _safe(lambda: engine.spatial_cortex.current_room),
        },
        "prediction": {
            "expected_next": _expected_next(engine, top_k),
        },
        "anomaly": {
            "surprise": round(surprise, 3),
            "threshold": round(threshold, 3),
            "is_anomaly": surprise >= threshold and threshold > 0.0,
            "baseline_surprise": round(
                float(_safe(lambda: pred.baseline_surprise, 0.0) or 0.0), 3),
            "average_surprise": round(
                float(_safe(lambda: pred.get_average_surprise(), 0.0) or 0.0), 3),
        },
        "learning": {
            "accuracy": _safe(lambda: round(float(engine.hippocampus.accuracy), 3)),
            "habits": _safe(lambda: int(engine.basal_ganglia.total_habits)),
            "reflex_rules": _safe(lambda: len(engine.cerebellum.rules)),
            "dopamine": _safe(lambda: round(float(engine.basal_ganglia.dopamine_signal), 3)),
        },
        # Every 0–1 field above, with what its ends mean — so the model never
        # has to guess the scale.
        "scales": {
            "confidence": "0 = no idea, 1 = certain",
            "surprise": "0 = fully expected, 1 = totally surprising",
            "threshold": "surprise at/above this flags an anomaly",
            "is_anomaly": "true = the last event was anomalous for this home",
            "dopamine": "reward signal, higher = more rewarding context",
            "learning_maturity": "cold_start < warming < mature (more = more learned)",
        },
    }


def render_llm_context(ctx: Dict[str, Any]) -> str:
    """Render :func:`build_llm_context` output as compact, labeled prose.

    Models reason more reliably over labeled text with inline scales than over a
    raw JSON blob, so this is the recommended payload for a prompt.
    """
    eng = ctx.get("engine", {})
    now = ctx.get("now", {})
    ano = ctx.get("anomaly", {})
    learn = ctx.get("learning", {})
    preds = ctx.get("prediction", {}).get("expected_next", [])

    pred_str = ", ".join(
        f"{p['event']} ({p['confidence']:.0%})" for p in preds
    ) or "—"

    flag = "YES" if ano.get("is_anomaly") else "no"
    lines = [
        "KONTINUUM home-brain state",
        f"- Learning: {eng.get('learning_maturity', '?')} "
        f"({eng.get('events_seen', 0)} events seen)",
        f"- Mode: {now.get('mode', '?')} | Room: {now.get('room', '?')}",
        f"- Expected next: {pred_str}",
        f"- Anomaly: {flag}  (surprise {ano.get('surprise', 0):.2f} vs "
        f"threshold {ano.get('threshold', 0):.2f}; "
        f"home baseline {ano.get('baseline_surprise', 0):.2f})",
        f"- Prediction accuracy: "
        + (f"{learn['accuracy']:.0%}" if learn.get("accuracy") is not None else "?")
        + f" | habits {learn.get('habits', '?')} | reflexes {learn.get('reflex_rules', '?')}",
        "Scales: surprise/confidence are 0–1 (0=expected/unsure, 1=surprising/certain).",
    ]
    return "\n".join(lines)


# ===========================================================================
# LLM → Engine : response parsing + normalization
# ===========================================================================
_FENCE_RE = re.compile(r"```[a-zA-Z0-9]*")


def extract_json(raw: Any) -> Optional[Union[dict, list]]:
    """Best-effort extraction of a JSON value from a model reply.

    Handles the real-world sloppiness: already-parsed dict/list, ```json fenced
    blocks, and a JSON object embedded in surrounding prose. Returns ``None`` if
    nothing parseable is found.
    """
    if isinstance(raw, (dict, list)):
        return raw
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if "```" in text:
        text = _FENCE_RE.sub("", text).replace("```", "").strip()
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        pass
    snippet = _first_balanced(text)
    if snippet is not None:
        try:
            return json.loads(snippet)
        except Exception:  # noqa: BLE001
            return None
    return None


def _first_balanced(text: str) -> Optional[str]:
    """Return the first balanced {...} or [...] substring, skipping strings."""
    openers = {"{": "}", "[": "]"}
    for i, ch in enumerate(text):
        if ch not in openers:
            continue
        close = openers[ch]
        depth = 0
        in_str = False
        esc = False
        for j in range(i, len(text)):
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == ch:
                depth += 1
            elif c == close:
                depth -= 1
                if depth == 0:
                    return text[i:j + 1]
        break
    return None


_NULLISH = {"", "null", "none", "n/a", "na", "-", "keine", "kein"}
_TRUTHY = {"true", "1", "yes", "y", "ja", "veto"}


def _opt_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return None if s.lower() in _NULLISH else s


def _priority(v: Any) -> int:
    try:
        n = int(round(float(v)))
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, n))


def _bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if v is None:
        return False
    return str(v).strip().lower() in _TRUTHY


def normalize_proposal(raw: Any, agent: Optional[str] = None) -> Dict[str, Any]:
    """Coerce a model reply into the strict proposal schema the engine acts on.

    Always returns a dict with all of :data:`PROPOSAL_FIELDS` plus ``valid``.
    Field coercion is forgiving: ``priority`` clamps to 0–100 from str/float,
    ``veto`` accepts ``"true"``/``"yes"``/``1``, ``action``/``entity_id`` map
    empty/``"null"`` to ``None``. ``valid=False`` signals an unparseable reply,
    so the caller can ignore it instead of acting on garbage.
    """
    data = extract_json(raw)
    if not isinstance(data, dict):
        return {
            "agent": agent, "action": None, "entity_id": None,
            "reason": "unparseable LLM response", "priority": 0,
            "veto": False, "valid": False,
        }
    return {
        "agent": agent if agent is not None else _opt_str(data.get("agent")),
        "action": _opt_str(data.get("action")),
        "entity_id": _opt_str(data.get("entity_id", data.get("entity"))),
        "reason": _opt_str(data.get("reason")) or "",
        "priority": _priority(data.get("priority", 0)),
        "veto": _bool(data.get("veto", False)),
        "valid": True,
    }
