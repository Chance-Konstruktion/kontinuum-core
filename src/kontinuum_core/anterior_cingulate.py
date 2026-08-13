"""Anterior Cingulate Cortex (ACC) – conflict monitor and error tracker.

Biological inspiration: The ACC detects conflicts between competing
action candidates and signals when uncertainty is too high. It
modulates the decision threshold: many conflicts → act more cautiously,
high clarity → act faster.

Responsibilities:
- Detect conflicts between modules (e.g. Hippocampus vs Amygdala)
- Track prediction errors (expectation vs reality)
- Adjust confidence thresholds dynamically
- Signal "cognitive control" when uncertainty is high

Performance: pure arithmetic, no I/O, no ML. ~0 ms per event.
"""

import logging
from collections import deque

_LOGGER = logging.getLogger(__name__)

# How many decisions to keep in the rolling window
HISTORY_SIZE = 200
# EMA factor for conflict rate
EMA_ALPHA = 0.05
# Thresholds for conflict level
CONFLICT_LOW = 0.2
CONFLICT_HIGH = 0.6


class AnteriorCingulateCortex:
    """Monitors conflicts between modules and adapts decision thresholds."""

    def __init__(self):
        self._conflict_history = deque(maxlen=HISTORY_SIZE)
        self._error_history = deque(maxlen=HISTORY_SIZE)

        self.conflict_level = 0.0      # 0.0 = no conflict, 1.0 = max conflict
        self.error_rate = 0.0          # fraction of wrong predictions
        self.cognitive_control = 0.0   # how hard to brake

        self.total_conflicts = 0
        self.total_agreements = 0
        self.total_errors = 0
        self.total_correct = 0
        self.threshold_adjustments = 0

        self.confidence_threshold = 0.5
        self._last_update_ts = 0.0

    def observe_decision(self, proposals: list) -> float:
        """Measures the conflict level of a decision round.

        proposals: list of module suggestions, e.g.:
          [{"source": "hippocampus", "action": "light.on", "confidence": 0.8},
           {"source": "amygdala", "action": "veto", "confidence": 0.6}]

        Returns: conflict value in [0.0, 1.0]
        """
        if not proposals or len(proposals) < 2:
            self._conflict_history.append(0.0)
            self._update_ema()
            self.total_agreements += 1
            return 0.0

        actions = [p.get("action", "") for p in proposals]
        unique_actions = set(actions)

        disagreement = (len(unique_actions) - 1) / max(1, len(actions) - 1)

        has_veto = any(p.get("action") == "veto" or p.get("veto", False) for p in proposals)
        if has_veto:
            disagreement = min(1.0, disagreement + 0.3)

        confidences = [p.get("confidence", 0.5) for p in proposals]
        if len(confidences) > 1:
            mean_conf = sum(confidences) / len(confidences)
            variance = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)
            disagreement = min(1.0, disagreement + variance * 0.5)

        self._conflict_history.append(disagreement)
        self._update_ema()

        if disagreement > CONFLICT_LOW:
            self.total_conflicts += 1
        else:
            self.total_agreements += 1

        return disagreement

    def observe_outcome(self, was_correct: bool):
        """Records whether the last action was correct."""
        self._error_history.append(0.0 if was_correct else 1.0)

        if was_correct:
            self.total_correct += 1
        else:
            self.total_errors += 1

        self.error_rate = (
            EMA_ALPHA * (0.0 if was_correct else 1.0)
            + (1.0 - EMA_ALPHA) * self.error_rate
        )

        self._adjust_threshold()

    def get_adjusted_threshold(self, base_confidence: float = 0.5) -> float:
        """Returns an adjusted confidence threshold.

        High conflict → threshold raised (more cautious).
        Low conflict  → threshold lowered (act faster).
        """
        return max(0.2, min(0.95, base_confidence + self.cognitive_control * 0.3))

    def should_defer_to_cortex(self) -> bool:
        """Returns True when uncertainty is high enough to consult the cortex (LLM)."""
        return self.conflict_level > CONFLICT_HIGH or self.error_rate > 0.4

    def _update_ema(self):
        if self._conflict_history:
            latest = self._conflict_history[-1]
            self.conflict_level = (
                EMA_ALPHA * latest + (1.0 - EMA_ALPHA) * self.conflict_level
            )
        self.cognitive_control = min(1.0, self.conflict_level * 0.6 + self.error_rate * 0.4)

    def _adjust_threshold(self):
        old = self.confidence_threshold

        if self.error_rate > 0.3:
            self.confidence_threshold = min(0.9, self.confidence_threshold + 0.02)
        elif self.error_rate < 0.1 and self.conflict_level < CONFLICT_LOW:
            self.confidence_threshold = max(0.3, self.confidence_threshold - 0.01)

        if abs(old - self.confidence_threshold) > 0.001:
            self.threshold_adjustments += 1

    @property
    def stats(self) -> dict:
        return {
            "conflict_level": round(self.conflict_level, 3),
            "error_rate": round(self.error_rate, 3),
            "cognitive_control": round(self.cognitive_control, 3),
            "confidence_threshold": round(self.confidence_threshold, 3),
            "total_conflicts": self.total_conflicts,
            "total_agreements": self.total_agreements,
            "total_errors": self.total_errors,
            "total_correct": self.total_correct,
            "threshold_adjustments": self.threshold_adjustments,
        }

    def to_dict(self) -> dict:
        return {
            "conflict_level": self.conflict_level,
            "error_rate": self.error_rate,
            "cognitive_control": self.cognitive_control,
            "confidence_threshold": self.confidence_threshold,
            "total_conflicts": self.total_conflicts,
            "total_agreements": self.total_agreements,
            "total_errors": self.total_errors,
            "total_correct": self.total_correct,
            "threshold_adjustments": self.threshold_adjustments,
        }

    def from_dict(self, data: dict):
        self.conflict_level = data.get("conflict_level", 0.0)
        self.error_rate = data.get("error_rate", 0.0)
        self.cognitive_control = data.get("cognitive_control", 0.0)
        self.confidence_threshold = data.get("confidence_threshold", 0.5)
        self.total_conflicts = data.get("total_conflicts", 0)
        self.total_agreements = data.get("total_agreements", 0)
        self.total_errors = data.get("total_errors", 0)
        self.total_correct = data.get("total_correct", 0)
        self.threshold_adjustments = data.get("threshold_adjustments", 0)


# Backwards-compatible alias for engine.py imports.
AnteriorCingulate = AnteriorCingulateCortex
