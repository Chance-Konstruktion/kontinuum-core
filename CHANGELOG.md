# Changelog

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
