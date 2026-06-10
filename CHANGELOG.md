# Changelog

## 0.1.2 (Unreleased)

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
  - Total: 37 tests, < 0.1 s wall.
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
