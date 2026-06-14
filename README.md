# KONTINUUM Core

Pure-Python, neuro-inspired learning engine extracted from
[KONTINUUM](https://github.com/Chance-Konstruktion/ha-kontinuum). No Home
Assistant dependency — usable from any Python project. **Zero runtime
dependencies** (standard library only), Python 3.9+.

> **Part of the 3-repo family:**
> **kontinuum-core** (this repo, HA-free Python package on PyPI) ·
> [`ha-kontinuum`](https://github.com/Chance-Konstruktion/ha-kontinuum) (full HA Pro integration with UI) ·
> [`ha-kontinuum-lite`](https://github.com/Chance-Konstruktion/ha-kontinuum-lite) (slim HA integration, no UI)

## Installation

```bash
pip install kontinuum-core
```

## Usage

```python
from kontinuum_core import KontinuumEngine

engine = KontinuumEngine()
engine.register_entity("binary_sensor.motion_kitchen", ha_area="kitchen", domain="binary_sensor")

snap = engine.observe({"entity_id": "binary_sensor.motion_kitchen", "new_state": "on"})
print(snap.surprise, snap.anomaly)   # 0..1 surprise, bool anomaly flag
```

## What it does

One `observe()` pipeline drives 18 brain-region modules: thalamic tokenization,
hippocampal n-gram memory, predictive **surprise** with a robust (median + MAD)
adaptive **anomaly** threshold, cerebellar reflexes, basal-ganglia habits, an
attention (reticular) **burst filter** for noise, and **sleep consolidation**
(replay / prune / dream-recombine / synaptic homeostasis during quiet spells).

State is **persistent and bounded**: `engine.to_dict()` / `from_dict()`
round-trip the full learned brain (with a `schema_version` guard), and the
learned maps are capped — safe to run for years on a Raspberry Pi.

## LLM integration contract (`kontinuum_core.llm`)

The engine is the sub-symbolic brain; an LLM is the optional language / reasoning
layer on top.

- **`build_llm_context(engine_or_brain)` / `render_llm_context(ctx)`** — export
  the state (anomaly signal, expected-next events, learning maturity) with
  **explicit 0–1 scales** so a model can reason over it reliably.
- **`extract_json(reply)` / `normalize_proposal(reply)`** — turn a model's
  (often sloppy: code-fenced, prose-wrapped, stringly-typed) reply into a
  strict, validated action proposal.

## Day-1 priors (`kontinuum_core.priors`)

`parse_home_prior(llm_reply)` + `seed_engine_from_prior(engine, prior)` let an
LLM describe the home at setup, so the engine starts already expecting the
household routine instead of from a blank slate.

## Benchmark

```bash
python benchmarks/replay.py
```

A replay benchmark + concept-drift stress test (also runs as a CI quality gate):
the surprise signal separates anomalies from routine at AUC ≈ 0.99 and re-adapts
after a routine change.

## License

AGPL-3.0 – see LICENSE file.
