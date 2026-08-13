"""
╔══════════════════════════════════════════════════════════════════╗
║  KONTINUUM – Basalganglien                                      ║
║  Handlungsauswahl & Belohnungslernen                            ║
║                                                                  ║
║  Biologisches Vorbild:                                           ║
║  Die Basalganglien steuern die Handlungsauswahl durch zwei      ║
║  Pfade: Der direkte Pfad (Go) verstärkt belohnte Aktionen,     ║
║  der indirekte Pfad (NoGo) unterdrückt bestrafte. Dopamin      ║
║  signalisiert den Reward Prediction Error – die Differenz       ║
║  zwischen erwartetem und tatsächlichem Ergebnis.                ║
║                                                                  ║
║  v0.13.0 – Erstimplementation:                                  ║
║  - Q-Value-Tabelle: (Kontext-Bucket, Aktion) → Erwarteter Wert ║
║  - TD-Learning mit adaptiver Lernrate (Dopamin-Signal)          ║
║  - Gewohnheitsbildung: Häufig erfolgreiche Aktionen = Habits    ║
║  - Striatum: Aktionsprioritäten für den PFC                     ║
╚══════════════════════════════════════════════════════════════════╝
"""

import logging
import time
from collections import defaultdict

_LOGGER = logging.getLogger(__name__)


class BasalGanglia:
    """
    Belohnungslernen und Handlungsauswahl.

    Architektur:
        Striatum (Q-Values)  →  Direkt (Go) / Indirekt (NoGo)
                ↑
        Dopamin-Signal (RPE) ←  Tatsächliches Ergebnis
                ↑
        PFC meldet Outcome   ←  User-Override oder implizites Akzeptieren

    Q-Value Update (TD-Learning):
        Q(bucket, action) += α * δ
        δ = reward - Q(bucket, action)    # Reward Prediction Error
        α = Lernrate (adaptiv via RPE)
    """

    # ── Konfiguration ──────────────────────────────────────────
    LEARNING_RATE = 0.1        # Basis-Lernrate (α)
    alpha = 0.1
    DISCOUNT_FACTOR = 0.9      # Wie stark zukünftige Belohnungen zählen (γ)
    HABIT_THRESHOLD = 10       # Ab N Erfolgen ohne Failure → Habit
    HABIT_BOOST = 0.2          # Bonus für Habits bei Aktionspriorisierung
    MAX_Q_ENTRIES = 2000       # Max Q-Value Einträge (Memory-Begrenzung)

    # Belohnungswerte
    REWARD_POSITIVE = 1.0      # User hat Aktion nicht rückgängig gemacht
    REWARD_NEGATIVE = -1.0     # User hat Override ausgelöst
    REWARD_NEUTRAL = 0.0       # Keine Info (Beobachtung)

    def __init__(self):
        # Q-Value Tabelle: "bucket:token_id" → Q-Wert
        self.q_values = defaultdict(float)
        # Minimal-FAL kompatible Tabelle: (state, action) -> q
        self.q_table = {}

        # Habit-Tracker: token_id → {successes, failures, is_habit}
        self.habits = defaultdict(lambda: {"successes": 0, "failures": 0, "is_habit": False})

        # Dopamin-Signal: Letzte RPE-Werte für adaptive Lernrate
        self._recent_rpe = []  # Letzte 20 RPEs
        self._last_rpe = 0.0

        # Pending Actions: Warten auf Outcome
        # entity_id → {token_id, bucket, timestamp, token_str}
        self.pending_actions = {}

        # Statistik
        self.total_updates = 0
        self.total_habits = 0
        self.total_positive = 0
        self.total_negative = 0


    # ── Minimal FAL API ─────────────────────────────────────

    def get_q_value(self, state: str, action: str) -> float:
        """Liest den Q-Wert für den FAL-State/Action-Key."""
        return float(self.q_table.get((state, action), 0.1))

    def update_q_value(self, state: str, action: str, reward: float):
        """Minimales Q-Learning-Update: Q = Q + alpha * (reward - Q)."""
        old_q = self.get_q_value(state, action)
        new_q = old_q + self.alpha * (float(reward) - old_q)
        self.q_table[(state, action)] = new_q
        return new_q

    # ── Striatum: Aktionspriorisierung ─────────────────────────

    def get_action_priority(self, token_id: int, bucket: int) -> float:
        """
        Berechnet die Aktionspriorität basierend auf Q-Value + Habit-Bonus.

        Returns: Prioritäts-Score [-1.0, 2.0+]
            > 0.0 = Direct Pathway (Go) → Aktion verstärken
            < 0.0 = Indirect Pathway (NoGo) → Aktion unterdrücken
        """
        q_key = f"{bucket}:{token_id}"
        q_value = self.q_values.get(q_key, 0.0)

        # Habit-Bonus: Bewährte Aktionen bekommen Extra-Boost
        habit = self.habits.get(token_id)
        if habit and habit["is_habit"]:
            q_value += self.HABIT_BOOST

        return q_value

    def rank_actions(self, candidates: list, bucket: int) -> list:
        """
        Sortiert Aktionskandidaten nach Basalganglien-Priorität.

        Args:
            candidates: [(token_id, prob, conf, source, n_obs), ...]
            bucket: Aktueller Kontext-Bucket

        Returns:
            Sortierte Liste mit zusätzlichem priority-Score.
        """
        ranked = []
        for candidate in candidates:
            token_id, prob, conf, source = candidate[:4]
            n_obs = candidate[4] if len(candidate) > 4 else 0
            priority = self.get_action_priority(token_id, bucket)
            ranked.append((token_id, prob, conf, source, n_obs, priority))

        # Sortiere: Höchste Priorität zuerst (Go-Pathway dominiert)
        ranked.sort(key=lambda x: x[5], reverse=True)
        return ranked

    # ── Dopamin-Signal: Reward Processing ──────────────────────

    def register_action(self, entity_id: str, token_id: int,
                        bucket: int, token_str: str = ""):
        """
        Registriert eine ausgeführte Aktion für späteres Outcome-Tracking.
        Wird vom PFC aufgerufen wenn eine Aktion ausgeführt wird.
        """
        self.pending_actions[entity_id] = {
            "token_id": token_id,
            "bucket": bucket,
            "timestamp": time.time(),
            "token_str": token_str,
        }

    def process_outcome(self, entity_id: str, positive: bool):
        """
        Verarbeitet das Outcome einer Aktion (TD-Learning).

        Wird aufgerufen wenn:
        - positive=True: User hat Aktion nicht rückgängig gemacht (300s)
        - positive=False: User hat Override innerhalb von 60s ausgelöst
        """
        action = self.pending_actions.pop(entity_id, None)
        if not action:
            return

        token_id = action["token_id"]
        bucket = action["bucket"]
        reward = self.REWARD_POSITIVE if positive else self.REWARD_NEGATIVE

        # ── TD-Update ──
        q_key = f"{bucket}:{token_id}"
        old_q = self.q_values.get(q_key, 0.0)

        # Reward Prediction Error (Dopamin-Signal)
        rpe = reward - old_q
        self._last_rpe = rpe
        self._recent_rpe.append(rpe)
        if len(self._recent_rpe) > 20:
            self._recent_rpe = self._recent_rpe[-20:]

        # Adaptive Lernrate: Größerer RPE → schnelleres Lernen
        alpha = self.LEARNING_RATE * (1.0 + abs(rpe) * 0.5)
        alpha = min(alpha, 0.5)  # Cap bei 50%

        # Q-Value Update
        new_q = old_q + alpha * rpe
        new_q = max(-1.0, min(2.0, new_q))  # Clamp
        self.q_values[q_key] = new_q

        # ── Habit-Tracking ──
        habit = self.habits[token_id]
        if positive:
            habit["successes"] += 1
            self.total_positive += 1
            # Habit-Bildung: N Erfolge ohne Failure
            if habit["successes"] >= self.HABIT_THRESHOLD and not habit["is_habit"]:
                habit["is_habit"] = True
                self.total_habits += 1
                _LOGGER.info(
                    "Basalganglien: Neue Gewohnheit! Token %d (%s) – %d Erfolge",
                    token_id, action.get("token_str", "?"), habit["successes"],
                )
        else:
            habit["failures"] += 1
            habit["successes"] = max(0, habit["successes"] - 3)  # Rückschritt
            self.total_negative += 1
            # Habit verlieren bei Failure
            if habit["is_habit"]:
                habit["is_habit"] = False
                self.total_habits = max(0, self.total_habits - 1)
                _LOGGER.info(
                    "Basalganglien: Gewohnheit verloren! Token %d (%s)",
                    token_id, action.get("token_str", "?"),
                )

        self.total_updates += 1

        _LOGGER.debug(
            "Basalganglien: Q(%s)=%.3f→%.3f, RPE=%.3f, α=%.3f, %s",
            q_key, old_q, new_q, rpe, alpha,
            "POSITIV" if positive else "NEGATIV",
        )

    def process_observation(self, token_id: int, bucket: int):
        """
        Passives Lernen: Beobachtet welche Token in welchem Kontext auftreten.
        Stärkt Q-Values leicht für natürlich vorkommende Muster.
        """
        q_key = f"{bucket}:{token_id}"
        old_q = self.q_values.get(q_key, 0.0)

        # Sehr kleine positive Verstärkung für beobachtete Muster
        # (= das Haus "will" diesen Zustand)
        nudge = 0.01
        new_q = old_q + nudge * (1.0 - abs(old_q))  # Schwächer wenn schon stark
        self.q_values[q_key] = max(-1.0, min(2.0, new_q))

    # ── Cleanup & Memory Management ───────────────────────────

    def cleanup_pending(self):
        """Entfernt abgelaufene pending actions (> 600s)."""
        now = time.time()
        expired = [eid for eid, a in self.pending_actions.items()
                   if (now - a["timestamp"]) > 600]
        for eid in expired:
            del self.pending_actions[eid]

    def _evict_q_values(self):
        """Entfernt die schwächsten Q-Values wenn Limit erreicht."""
        if len(self.q_values) <= self.MAX_Q_ENTRIES:
            return
        # Entferne Einträge nah an 0 (wenig Information)
        scored = sorted(self.q_values.items(), key=lambda x: abs(x[1]))
        to_remove = len(self.q_values) - self.MAX_Q_ENTRIES
        for key, _ in scored[:to_remove]:
            del self.q_values[key]

    # ── Persistence ────────────────────────────────────────────

    def to_dict(self) -> dict:
        self._evict_q_values()
        return {
            "q_values": dict(self.q_values),
            "habits": {
                str(k): v for k, v in self.habits.items()
                if v["successes"] > 0 or v["failures"] > 0
            },
            "total_updates": self.total_updates,
            "total_habits": self.total_habits,
            "total_positive": self.total_positive,
            "total_negative": self.total_negative,
        }

    def from_dict(self, data: dict):
        self.q_values = defaultdict(float, data.get("q_values", {}))
        for k, v in data.get("habits", {}).items():
            try:
                self.habits[int(k)] = v
            except (ValueError, TypeError):
                pass
        self.total_updates = data.get("total_updates", 0)
        self.total_habits = data.get("total_habits", 0)
        self.total_positive = data.get("total_positive", 0)
        self.total_negative = data.get("total_negative", 0)

    @property
    def dopamine_signal(self) -> float:
        """Aktuelles Dopamin-Niveau (durchschnittlicher RPE)."""
        if not self._recent_rpe:
            return 0.0
        return sum(self._recent_rpe) / len(self._recent_rpe)

    @property
    def stats(self) -> dict:
        n_positive_q = sum(1 for v in self.q_values.values() if v > 0.1)
        n_negative_q = sum(1 for v in self.q_values.values() if v < -0.1)
        habit_list = [
            {"token_id": tid, "successes": h["successes"], "failures": h["failures"]}
            for tid, h in self.habits.items()
            if h["is_habit"]
        ]
        return {
            "total_updates": self.total_updates,
            "total_habits": self.total_habits,
            "total_positive": self.total_positive,
            "total_negative": self.total_negative,
            "q_entries": len(self.q_values),
            "go_actions": n_positive_q,
            "nogo_actions": n_negative_q,
            "dopamine_signal": round(self.dopamine_signal, 3),
            "last_rpe": round(self._last_rpe, 3),
            "active_habits": habit_list[:5],
        }
