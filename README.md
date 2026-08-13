# KONTINUUM Core

[![Tests](https://github.com/Chance-Konstruktion/kontinuum-core/actions/workflows/tests.yaml/badge.svg?branch=main)](https://github.com/Chance-Konstruktion/kontinuum-core/actions/workflows/tests.yaml)
[![PyPI version](https://img.shields.io/pypi/v/kontinuum-core.svg)](https://pypi.org/project/kontinuum-core/)
[![Python versions](https://img.shields.io/pypi/pyversions/kontinuum-core.svg)](https://pypi.org/project/kontinuum-core/)
[![License: AGPL-3.0](https://img.shields.io/pypi/l/kontinuum-core.svg)](https://github.com/Chance-Konstruktion/kontinuum-core/blob/main/LICENSE)

Pure-Python, neuro-inspired learning engine extracted from
[KONTINUUM](https://github.com/Chance-Konstruktion/ha-kontinuum). No Home
Assistant dependency — usable from any Python project. **Zero runtime
dependencies** (standard library only), Python 3.9+.

> **Part of the KONTINUUM family:**
> **kontinuum-core** (this repo, HA-free Python package on PyPI) ·
> [`ha-kontinuum`](https://github.com/Chance-Konstruktion/ha-kontinuum) (full HA Pro integration with UI) ·
> [`ha-kontinuum-lite`](https://github.com/Chance-Konstruktion/ha-kontinuum-lite) (slim HA integration, no UI) ·
> [`kontinuum-AI-anomaly`](https://github.com/Chance-Konstruktion/kontinuum-AI-anomaly) (anomaly / novelty monitor for **agent action streams**, built on this engine)

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

One `observe()` pipeline drives 26 brain-region modules: thalamic tokenization,
hippocampal n-gram memory, predictive **surprise** with a robust (median + MAD)
adaptive **anomaly** threshold, cerebellar reflexes, basal-ganglia habits, an
attention (reticular) **burst filter** for noise, and **sleep consolidation**
(replay / prune / dream-recombine / synaptic homeostasis during quiet spells).

Since 0.5.0 the set also models the missing slow signals — a **lateral
habenula** (anti-reward: stop re-proposing rejected actions), a **subthalamic
nucleus** ("hold your horses" under conflict), a **learned circadian clock**
(suprachiasmatic), an **interval-timing** stopwatch for recurring cadences
(e.g. "every few weeks"), and the **cortisol / acetylcholine / serotonin** modulators
plus **BDNF** use-dependent protection. All are scalar EMAs or tiny bounded
maps, start neutral, and add ~0 ms/event.

State is **persistent and bounded**: `engine.to_dict()` / `from_dict()`
round-trip the full learned brain (with a `schema_version` guard), and the
learned maps are capped — safe to run for years on a Raspberry Pi.

Every `observe()` returns an `EngineSnapshot` with the live observability
signals — `surprise` (0–1 prediction error) and the adaptive `anomaly` flag —
which the HA integrations surface directly as entities (`sensor.kontinuum_surprise`,
`binary_sensor.kontinuum_anomaly`, …). The engine stands on its own; any LLM is
strictly an optional layer on top.

## Documentation

Full reference in [`docs/`](docs/):
[**MODULES.md**](docs/MODULES.md) (all 26 brain modules) ·
[**PIPELINE.md**](docs/PIPELINE.md) (the `observe()` flow, `EngineSnapshot.extra`
fields, the reward loop, persistence).

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

## Built on top of core

The engine is designed for smart-home event streams, but it is a general
sequence-learner. [**`kontinuum-AI-anomaly`**](https://github.com/Chance-Konstruktion/kontinuum-AI-anomaly)
(PyPI: `ai-kontinuum-monitor`) points it at an **AI agent's action log** instead
and adds the monitor layer core deliberately leaves out — robust scoring,
anomaly history, alerting (escalation / snooze), a dashboard, multi-agent
cross-stream correlation, strategy presets and an LLM-feedback loop. Core stays
**untouched** there; it is a thin, additive layer, which is what keeps this
engine equally usable from the Home Assistant integrations and from an
agent-monitoring context.

```bash
pip install ai-kontinuum-monitor   # pulls in kontinuum-core
```

## Benchmark

```bash
python benchmarks/replay.py
```

A replay benchmark + concept-drift stress test (also runs as a CI quality gate):
the surprise signal separates anomalies from routine at AUC ≈ 0.99 and re-adapts
after a routine change.

## License

AGPL-3.0 – see LICENSE file.
