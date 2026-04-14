"""
Predictive Processing – Surprise-Based Learning.

Biologisches Vorbild: Das Gehirn generiert ständig Vorhersagen über
die nächsten sensorischen Eingaben. Nur die ABWEICHUNG (Prediction Error /
Surprise) wird wirklich verarbeitet und gelernt. Erwartete Events werden
fast ignoriert.

Dieses Modul berechnet für jedes eingehende Event einen Surprise-Wert
(0.0 = komplett erwartet, 1.0 = totale Überraschung) und stellt diesen
als Lerngewicht zur Verfügung.

Effekte:
- Erwartete Muster (Licht an um 7:00 wie jeden Tag) → kaum Lernen
- Unerwartete Events (Licht an um 3 Uhr nachts) → starkes Lernen
- Neue Events (noch nie gesehen) → maximales Lernen
- Verbessert Signal/Rausch-Verhältnis aller Module

Performance: 1 Lookup + Arithmetik pro Event. ~0 ms.
"""

import logging
import time
from collections import deque

_LOGGER = logging.getLogger(__name__)

# EMA-Faktoren
SURPRISE_EMA_ALPHA = 0.08     # Wie schnell passt sich das Baseline-Surprise an
NOVELTY_DECAY = 0.995         # Wie schnell verliert ein neues Token seine Neuheit

# Surprise → Lerngewicht Mapping
# surprise 0.0 → weight 0.2 (erwartetes Event: wenig lernen, aber nicht null)
# surprise 0.5 → weight 1.0 (normales Surprise-Level)
# surprise 1.0 → weight 2.5 (totale Überraschung: viel lernen)
MIN_LEARN_WEIGHT = 0.2
MAX_LEARN_WEIGHT = 2.5


class PredictiveProcessing:
    """Berechnet Surprise-Signale für jedes Event."""

    def __init__(self):
        # Surprise-Historie (EMA)
        self.baseline_surprise = 0.3   # Erwartetes Surprise-Niveau
        self.current_surprise = 0.0    # Letzter Surprise-Wert
        self.learn_weight = 1.0        # Aktuelles Lerngewicht

        # Token-Bekanntheit: wie oft wurde jedes Token schon gesehen
        self._token_familiarity = {}   # token_id → Zähler (exponentiell gewichtet)

        # Statistiken
        self.total_events = 0
        self.total_surprises = 0       # Events mit surprise > 0.6
        self.total_expected = 0        # Events mit surprise < 0.2
        self.max_surprise = 0.0
        self.surprise_history = deque(maxlen=100)  # Letzte 100 Surprise-Werte

    def compute_surprise(self, token_id: int, predictions: list) -> float:
        """
        Berechnet den Surprise-Wert für ein eingehendes Event.

        Args:
            token_id: Das tatsächlich eingetretene Event-Token
            predictions: Aktuelle Vorhersagen [(token_id, prob, conf, source, n_obs), ...]

        Returns:
            Surprise-Wert 0.0-1.0
        """
        self.total_events += 1

        # ── Komponente 1: Vorhersage-Fehler ──
        # War dieses Token in den Vorhersagen?
        prediction_surprise = 1.0  # Default: nicht vorhergesagt = maximale Überraschung

        if predictions:
            for pred in predictions:
                pred_token = pred[0]
                pred_prob = pred[1] if len(pred) > 1 else 0.0
                pred_conf = pred[2] if len(pred) > 2 else 0.0

                if pred_token == token_id:
                    # Token war vorhergesagt!
                    # Surprise = 1 - (Wahrscheinlichkeit × Konfidenz)
                    prediction_surprise = 1.0 - (pred_prob * pred_conf)
                    break

            # Bonus: War es die TOP-Vorhersage?
            if predictions[0][0] == token_id:
                top_conf = predictions[0][2] if len(predictions[0]) > 2 else 0.0
                prediction_surprise *= (1.0 - top_conf * 0.3)  # Extra-Reduktion

        # ── Komponente 2: Neuheit (Token-Bekanntheit) ──
        familiarity = self._token_familiarity.get(token_id, 0.0)
        novelty = max(0.0, 1.0 - familiarity / 50.0)  # Nach 50 Sichtungen = vertraut

        # Bekanntheit aktualisieren
        self._token_familiarity[token_id] = familiarity + 1.0

        # Alle Token-Bekanntheit leicht decayen (vergessen)
        if self.total_events % 100 == 0:
            self._decay_familiarity()

        # ── Kombiniertes Surprise-Signal ──
        # 70% Prediction Error + 30% Novelty
        surprise = prediction_surprise * 0.7 + novelty * 0.3
        surprise = max(0.0, min(1.0, surprise))

        # ── Baseline-Anpassung (was ist "normal"?) ──
        # Surprise relativ zur Baseline bewerten
        relative_surprise = surprise - self.baseline_surprise
        # Baseline nachführen
        self.baseline_surprise = (
            SURPRISE_EMA_ALPHA * surprise
            + (1.0 - SURPRISE_EMA_ALPHA) * self.baseline_surprise
        )

        self.current_surprise = surprise

        # ── Lerngewicht berechnen ──
        # Surprise → Lerngewicht (nicht-linear: Überraschungen zählen überproportional)
        if surprise < 0.15:
            # Fast komplett erwartet → minimales Lernen
            self.learn_weight = MIN_LEARN_WEIGHT
            self.total_expected += 1
        elif surprise > 0.6:
            # Überraschung → starkes Lernen (quadratisch skaliert)
            self.learn_weight = min(MAX_LEARN_WEIGHT, 1.0 + (surprise - 0.3) ** 2 * 5.0)
            self.total_surprises += 1
        else:
            # Normal → lineares Lernen
            self.learn_weight = 0.5 + surprise * 1.5
            self.learn_weight = max(MIN_LEARN_WEIGHT, min(MAX_LEARN_WEIGHT, self.learn_weight))

        # Tracking
        self.max_surprise = max(self.max_surprise, surprise)
        self.surprise_history.append(surprise)

        return surprise

    def get_learn_weight(self) -> float:
        """Gibt das aktuelle Lerngewicht zurück (0.2 - 2.5)."""
        return self.learn_weight

    def get_average_surprise(self) -> float:
        """Durchschnittliches Surprise-Level der letzten 100 Events."""
        if not self.surprise_history:
            return 0.5
        return sum(self.surprise_history) / len(self.surprise_history)

    def _decay_familiarity(self):
        """Alle Token-Bekanntheit leicht vergessen (Novelty-Recovery)."""
        to_remove = []
        for token_id in self._token_familiarity:
            self._token_familiarity[token_id] *= NOVELTY_DECAY
            if self._token_familiarity[token_id] < 0.1:
                to_remove.append(token_id)
        for token_id in to_remove:
            del self._token_familiarity[token_id]

    @property
    def stats(self) -> dict:
        return {
            "current_surprise": round(self.current_surprise, 3),
            "baseline_surprise": round(self.baseline_surprise, 3),
            "learn_weight": round(self.learn_weight, 3),
            "average_surprise": round(self.get_average_surprise(), 3),
            "total_events": self.total_events,
            "total_surprises": self.total_surprises,
            "total_expected": self.total_expected,
            "surprise_ratio": round(
                self.total_surprises / max(1, self.total_events), 3
            ),
            "max_surprise": round(self.max_surprise, 3),
        }

    def to_dict(self) -> dict:
        return {
            "baseline_surprise": self.baseline_surprise,
            "total_events": self.total_events,
            "total_surprises": self.total_surprises,
            "total_expected": self.total_expected,
            "max_surprise": self.max_surprise,
            "token_familiarity": dict(self._token_familiarity),
        }

    def from_dict(self, data: dict):
        self.baseline_surprise = data.get("baseline_surprise", 0.3)
        self.total_events = data.get("total_events", 0)
        self.total_surprises = data.get("total_surprises", 0)
        self.total_expected = data.get("total_expected", 0)
        self.max_surprise = data.get("max_surprise", 0.0)
        self._token_familiarity = data.get("token_familiarity", {})
