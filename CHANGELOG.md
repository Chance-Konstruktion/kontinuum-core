# Changelog

## 0.1.3 (2026-06-14)

### Persistence schema versioning

- **`to_dict()` now stamps a `schema_version`** (currently `1`) into the
  serialized brain. `from_dict()` refuses to restore a brain written by a
  *newer* schema than the running engine understands — it logs a warning and
  cold-starts instead of loading a half-understood state that could corrupt
  learning. Brains with an older or missing version (anything written by
  `kontinuum-core <= 0.1.2`) are treated as schema `1` and load exactly as
  before, so the change is fully backward compatible.

### Replay benchmark (offline evaluation)

- **`benchmarks/replay.py`** — the first end-to-end "does it actually learn"
  harness. It synthesises a learnable multi-room household routine, trains the
  engine on it, then replays a held-out period with out-of-distribution events
  injected (motion in the kitchen at 03:00, the bedroom at 14:00, …) and
  reports how well the `surprise` signal separates anomalies from routine
  (mean surprise, Mann-Whitney AUC, and precision/recall of the built-in
  `anomaly` flag). Runs standalone (`python benchmarks/replay.py`, exits
  non-zero if separation collapses) and is not shipped in the wheel.
- **`tests/test_benchmark.py`** turns the harness into a regression guard
  (asserts AUC stays well above chance). On the synthetic routine the surprise
  signal separates anomalies at AUC ≈ 0.99.
- **`tests/test_persistence_schema.py`** locks in the round-trip + version
  guarantees above.

## 0.1.2 (2026-06-13)

### Engine wiring — all 18 modules now influence the decision

- **Full neuro-pipeline in `KontinuumEngine.observe()`:** the engine
  instantiated 18 modules but its pipeline only ever drove 6 (thalamus,
  hypothalamus, insula, hippocampus, predictive, metaplasticity). The other
  twelve were dead weight for any standalone / `ha-kontinuum-lite` user. The
  observe path now drives the proven wiring from the full integration:
  - **Locus Coeruleus** arousal (event density) → modulates ranking and the
    Reticular burst filter (`reticular.set_arousal_source(locus)`).
  - **Neurorhythms** register every surprise and modulate the learning rate
    (circadian rhythm + dopamine bursts) before the hippocampus learns.
  - **Basal ganglia** passively observe each event (Q-values) and re-rank
    predictions by Go/NoGo priority.
  - **Cerebellum** checks for a fired reflex rule each event, injects a
    confident reflex as a top prediction, and compiles new rules out of
    hippocampus memory every 50 events.
  - **Nucleus accumbens** habit-bias and **entorhinal** next-room
    anticipation feed the ranking.
  - **Anterior cingulate** scores the conflict between the module votes; its
    `cognitive_control` (EMA of conflict + error rate) damps ranking
    confidence by up to 25% on the *next* event — a genuinely closed control
    loop (previously the ACC output was consumed nowhere).
  - **Prefrontal cortex** turns the ranked candidates into an advisory
    `Decision`. The core stays in `SHADOW` mode: it recommends, never acts.
  - **Amygdala** risk assessment runs inside the PFC evaluation.
- **`KontinuumEngine.feedback(positive)`** closes the reward loops a pure
  `observe()` stream cannot: a host that executed (or saw the user
  accept/override) the last decision calls it to teach basal ganglia
  (TD Q-update + habits), nucleus accumbens (habit bias), neurorhythms
  (dopamine burst/dip), the cerebellum rule that fired, and the ACC error
  monitor. Without it these outcome modules only observe.
- **`snapshot.extra`** now surfaces the rich signals that used to be
  discarded: `anomaly_threshold`, `arousal`, `cognitive_control`,
  `conflict_level`, `dopamine`, `expected_next_room`, the fired `reflex` and
  the advisory `decision`.
- **`KontinuumEngine.to_dict()` / `from_dict()`** serialize the full engine
  state (all 17 stateful modules + room/anticipation wiring), so the
  now-active modules survive a restart instead of cold-starting.

### Prediction & anomaly quality

- **Adaptive anomaly threshold:** anomalies are now flagged relative to the
  home's own surprise distribution (baseline + 2σ of the last 100 surprises,
  clamped to [0.55, 0.95]) instead of a fixed `0.7`. A chaotic home raises
  the bar, a very predictable one lowers it. Falls back to `0.7` until
  30 samples exist (`PredictiveProcessing.anomaly_threshold()`).
- **Confidence-calibrated misses:** an unpredicted token is no longer
  automatic maximum surprise. The miss surprise scales with the model's own
  confidence — full surprise only when a high-confidence prediction was
  wrong; cold-start misses are moderate. Kills the "everything is an
  anomaly on day one" failure mode.
- **Probability shrinkage (Hippocampus):** transition probabilities use
  `count / (total + 1)` instead of `count / total`, so a 2-out-of-2 pattern
  claims 0.67 instead of 1.0. Vanishes for large samples.
- **Evidence combination (Hippocampus):** predictions supported by multiple
  n-gram orders / neighbor buckets now accumulate their evidence
  (`Σ prob × weight`) instead of taking only the best single source —
  agreement across orders raises confidence.
- **Cerebellum confidence recovery:** rule successes slowly restore
  confidence (×1.02, capped 0.99); previously only failures moved it
  (×0.95), permanently burying rules after one bad streak.

### Fixed
- `PredictiveProcessing.from_dict` now converts token-familiarity keys back
  to `int`. After a JSON persistence round-trip the keys were strings, so
  every familiarity lookup missed and novelty silently reset on restart.
- `PredictiveProcessing` now persists `surprise_history`. It was omitted from
  `to_dict`, so after every restart the adaptive anomaly threshold (and
  `average_surprise`) fell back to the fixed `0.7` default for the first
  30 events before re-learning the home's surprise level. The threshold now
  survives the restart.
- `Hippocampus._apply_decay` prunes entries below 0.05 and drops empty
  n-grams/buckets — micro-weights no longer accumulate unbounded over months.
- `__version__` now reads `"0.1.1"` then `"0.1.2"` (matches `pyproject.toml`
  again after the 0.1.0 → 0.1.1 typo). Both HA integrations pin
  `kontinuum-core>=0.1.1`; before this fix a runtime introspection
  would still report the stale `0.1.0` literal.

### Changed
- `KontinuumEngine.__init__` now accepts the full roadmap contract:
  `config: dict | None`, `scheduler: Scheduler | None`,
  `storage_path: str | None`. The scheduler is wired into
  `Metaplasticity` so hosts can opt-in to its 24 h adaptation loop
  by passing a `Scheduler`-Protocol instance (e.g. the `HAScheduler`
  shipped by both HA integrations).
- `Metaplasticity` inside the engine now sees the full brain-module
  dict (hippocampus, predictive, cerebellum, basal_ganglia, reticular,
  accumbens, locus) instead of just hippocampus + predictive.

### Added
- `tests/` directory with pytest-driven smoke tests. Initial coverage:
  - `test_engine.py` (13) — public API surface, constructor contract,
    Metaplasticity wiring, observe pipeline, the full 18-module surface.
  - `test_thalamus.py` (5) — entry point of the observe pipeline.
  - `test_predictive_processing.py` (7) — surprise + learn_weight.
  - `test_hippocampus.py` (6) — n-gram memory.
  - `test_cerebellum.py` (6) — rule/chunk engine.
  - `test_engine_wiring.py` (10) — locks in that every decision module is
    driven, the cognitive-control loop is closed, entorhinal anticipation
    boosts the ranking, `feedback()` reaches the reward modules, and the
    engine state (incl. `surprise_history`) round-trips.
  - Total: 53 tests, < 0.2 s wall.
- `[project.optional-dependencies].dev = ["pytest>=7"]` for
  `pip install -e ".[dev]"`.
- `[tool.pytest.ini_options]` configures `testpaths = ["tests"]` and
  `pythonpath = ["src"]` so `python -m pytest` works from repo root.
- `.github/workflows/tests.yaml` runs pytest against Python 3.9–3.12
  on every push/PR to main.
- `.gitignore` for the usual Python build/test artefacts.
- README banner linking to the two HA integrations.

## 0.1.1 (2026-04-16)

### Fixed
- `engine.py` now uses the real brain-module interfaces. The previous
  version called non-existent `.update()` methods on Thalamus,
  Hippocampus and PredictiveProcessing and could not actually run.

### Added
- Full observation pipeline in `KontinuumEngine.observe()`:
  `thalamus.process()` → `hypothalamus.absorb()` → `insula.process()`
  → 21-dim context vector → `hippocampus.predict()` →
  `predictive.compute_surprise()` → `hippocampus.learn(learn_weight)`.
- `EngineSnapshot` now exposes `token_id`, `token`, and a real
  `learning_state` (`cold_start` / `learning` / `stable`) derived from
  `hippocampus.total_events` and `hippocampus.accuracy`.
- `register_entity()` delegates to `Thalamus.register_entity()`.
- `Metaplasticity` is wired with the engine's brain modules so its
  adaptation pass can actually find them.

## 0.1.0 (2026-04-10)

Initial PyPI release: all 18 neuro-inspired brain modules extracted
from the Home Assistant integration into an HA-free core package.
