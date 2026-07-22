"""
╔══════════════════════════════════════════════════════════════════╗
║  KONTINUUM – Präfrontaler Kortex                                ║
║  Entscheidungsinstanz: Soll ich handeln?                        ║
║                                                                  ║
║  Biologisches Vorbild:                                           ║
║  Der PFC wägt ab, plant und entscheidet. Er integriert alle    ║
║  Informationen und bestimmt die optimale Handlung.              ║
║  Er lernt auch aus implizitem Feedback: Wenn der User eine     ║
║  KONTINUUM-Aktion innerhalb von 60s rückgängig macht, ist      ║
║  das negatives Feedback.                                         ║
║                                                                  ║
║  v0.18.0 – Betriebsmodi:                                       ║
║  • shadow  = Nur beobachten, nichts ausführen                   ║
║  • confirm = Bestätigung anfordern vor Ausführung               ║
║  • active  = Selbständig schalten (freigeschaltete Semantiken) ║
╚══════════════════════════════════════════════════════════════════╝
"""

import logging
import time

_LOGGER = logging.getLogger(__name__)

MODE_SHADOW = "shadow"
MODE_CONFIRM = "confirm"
MODE_ACTIVE = "active"
VALID_MODES = {MODE_SHADOW, MODE_CONFIRM, MODE_ACTIVE}

ACTIONABLE_SEMANTICS = {
    "light", "switch", "fan", "cover", "climate",
    "media", "automation", "vacuum",
}

STATE_TO_SERVICE = {
    "light": {"on": "turn_on", "off": "turn_off"},
    "switch": {"on": "turn_on", "off": "turn_off"},
    "fan": {"on": "turn_on", "off": "turn_off"},
    "media": {"playing": "media_play", "paused": "media_pause", "off": "turn_off"},
    "cover": {"open": "open_cover", "closed": "close_cover"},
    "climate": {"heating": "set_hvac_mode", "cooling": "set_hvac_mode", "off": "turn_off"},
    "automation": {"on": "turn_on"},
    "vacuum": {"on": "start", "off": "return_to_base"},
}


class Decision:
    """Eine Entscheidung des PFC."""
    OBSERVE = "OBSERVE"
    PREPARE = "PREPARE"
    SUGGEST = "SUGGEST"
    CONFIRM = "CONFIRM"
    EXECUTE = "EXECUTE"

    def __init__(self):
        self.token = ""
        self.token_id = 0
        self.entity_id = ""
        self.confidence = 0.0
        self.utility = 0.0
        self.risk = 0.0
        self.n_obs = 0
        self.stage = self.OBSERVE
        self.source = ""
        self.reasons = []

    def to_dict(self):
        return {
            "token": self.token,
            "entity_id": self.entity_id,
            "confidence": self.confidence,
            "utility": self.utility,
            "risk": self.risk,
            "n_obs": self.n_obs,
            "stage": self.stage,
            "source": self.source,
            "reasons": self.reasons,
        }


class PrefrontalCortex:
    """Entscheidungsinstanz von KONTINUUM."""

    UTILITY_THRESHOLD_SUGGEST = 0.4
    UTILITY_THRESHOLD_EXECUTE = 0.6
    OVERRIDE_WINDOW = 60
    IMPLICIT_POSITIVE_DELAY = 300
    MIN_OBS_SUGGEST = 15
    MIN_OBS_EXECUTE = 30

    def __init__(self, amygdala):
        self.amygdala = amygdala
        self.shadow_mode = True
        self.operation_mode = MODE_SHADOW
        self.total_decisions = 0
        self.total_executions = 0
        self.total_confirms = 0
        self.overrides_detected = 0
        self.own_actions = {}
        self.utility_weights = {}
        self._feedback_log = []
        self.activated_semantics = set()
        self._pending_confirms = {}

    def evaluate(self, predictions: list, thalamus,
                 basal_ganglia=None, bucket: int = 0) -> Decision:
        best_decision = None
        best_utility = -1

        for prediction in predictions:
            token_id, prob, conf, source = prediction[:4]
            n_obs = prediction[4] if len(prediction) > 4 else 0

            token = thalamus.decode_token(token_id)
            parts = token.split(".")
            if len(parts) != 3:
                continue

            room, semantic, state = parts
            if semantic not in ACTIONABLE_SEMANTICS:
                continue

            assessment = self.amygdala.assess(token, semantic, room, state, conf)

            if assessment["decision"] == "VETO":
                continue

            risk = assessment["risk"]
            weight = self.utility_weights.get(semantic, 1.0)

            q_boost = 0.0
            if basal_ganglia:
                q_boost = basal_ganglia.get_action_priority(token_id, bucket) * 0.2

            utility = conf * weight - risk * 0.5 + q_boost

            if utility > best_utility:
                best_utility = utility
                d = Decision()
                d.token = token
                d.token_id = token_id
                d.confidence = conf
                d.utility = utility
                d.risk = risk
                d.source = source
                d.n_obs = n_obs
                d.reasons = assessment["reasons"]

                candidates = thalamus.resolve_entities(token)
                d.entity_id = candidates[0] if candidates else ""

                if not d.entity_id:
                    d.stage = Decision.OBSERVE
                    d.reasons = d.reasons + ["keine Entity auflösbar (Thalamus)"]
                    best_decision = d
                    continue

                if n_obs < self.MIN_OBS_SUGGEST:
                    d.stage = Decision.OBSERVE
                    d.reasons = d.reasons + [f"n={n_obs} < {self.MIN_OBS_SUGGEST} (zu wenig Daten)"]
                elif self.operation_mode == MODE_SHADOW:
                    d.stage = Decision.OBSERVE
                elif (utility >= self.UTILITY_THRESHOLD_EXECUTE
                      and n_obs >= self.MIN_OBS_EXECUTE
                      and (self.operation_mode in (MODE_ACTIVE, MODE_CONFIRM)
                           or semantic in self.activated_semantics)):
                    if self.operation_mode == MODE_CONFIRM:
                        d.stage = Decision.CONFIRM
                    else:
                        d.stage = Decision.EXECUTE
                elif utility >= self.UTILITY_THRESHOLD_SUGGEST:
                    d.stage = Decision.SUGGEST
                else:
                    d.stage = Decision.OBSERVE

                best_decision = d

        if best_decision:
            self.total_decisions += 1
            if best_decision.stage == Decision.EXECUTE:
                self.total_executions += 1
            elif best_decision.stage == Decision.CONFIRM:
                self.total_confirms += 1

        return best_decision

    def get_service_call(self, decision: Decision) -> dict:
        parts = decision.token.split(".")
        if len(parts) != 3:
            return None
        room, semantic, state = parts
        services = STATE_TO_SERVICE.get(semantic, {})
        service = services.get(state)
        if not service:
            return None
        data = {"entity_id": decision.entity_id}
        if semantic == "climate" and state != "off":
            data["hvac_mode"] = state
        return {
            "domain": semantic,
            "service": service,
            "entity_id": decision.entity_id,
            "data": data,
        }

    def is_own_action(self, entity_id: str) -> bool:
        action = self.own_actions.get(entity_id)
        if not action:
            return False
        return (time.time() - action["time"]) < 10

    def mark_own_action(self, entity_id: str, token: str = "", semantic: str = ""):
        self.own_actions[entity_id] = {
            "time": time.time(),
            "token": token,
            "semantic": semantic,
        }

    def check_override(self, entity_id: str, new_state: str, amygdala=None) -> bool:
        action = self.own_actions.get(entity_id)
        if not action:
            return False
        elapsed = time.time() - action["time"]
        if elapsed > self.OVERRIDE_WINDOW:
            return False
        if elapsed < 2:
            return False
        self.overrides_detected += 1
        token = action.get("token", "")
        _LOGGER.info("Override erkannt: %s (nach %.0fs) – negatives Feedback", entity_id, elapsed)
        if amygdala and token:
            amygdala.learn_from_feedback(token, "negative")
        self._feedback_log.append({
            "time": time.time(), "entity_id": entity_id,
            "token": token, "feedback": "negative", "delay": elapsed,
        })
        if len(self._feedback_log) > 100:
            self._feedback_log = self._feedback_log[-100:]
        semantic = action.get("semantic", "")
        if semantic:
            current = self.utility_weights.get(semantic, 1.0)
            self.utility_weights[semantic] = max(0.1, current - 0.05)
        del self.own_actions[entity_id]
        return True

    def check_implicit_positives(self, amygdala):
        """EXPERIMENTELL / NICHT VERDRAHTET — im Core aktuell nirgends aufgerufen.

        Impliziter Lernpfad: eine eigene Aktion, die nach ``IMPLICIT_POSITIVE_DELAY``
        nicht überschrieben wurde, gilt als schwach positiv. Bewusst **nicht** an
        den Reward-Loop (Nucleus Accumbens / Neurorhythms-Dopamin) angebunden — der
        Reward-Loop läuft ausschließlich über das explizite ``engine.feedback()``
        (siehe dessen Docstring) und gehört an die Host-/HA-Integration, nicht ins
        autonome Core-Verhalten. Autonomes implizites Lernen wäre eine bewusste
        Produktentscheidung mit eigenen Leitplanken (Selbstverstärkung begrenzen,
        Event-Zeit statt Wall-Clock, abschaltbar, explizites Feedback gewinnt).

        Achtung: nutzt ``time.time()`` (Wall-Clock) und wäre im Replay damit
        unsichtbar/untestbar — vor einer echten Aktivierung erst auf Event-Zeit
        umstellen.
        """
        now = time.time()
        to_remove = []
        for entity_id, action in list(self.own_actions.items()):
            elapsed = now - action["time"]
            if elapsed > self.IMPLICIT_POSITIVE_DELAY:
                token = action.get("token", "")
                if token and amygdala:
                    amygdala.learn_from_feedback(token, "positive")
                semantic = action.get("semantic", "")
                if semantic:
                    current = self.utility_weights.get(semantic, 1.0)
                    self.utility_weights[semantic] = min(2.0, current + 0.01)
                to_remove.append(entity_id)
        for eid in to_remove:
            del self.own_actions[eid]
        return to_remove if to_remove else None

    def learn_from_feedback(self, semantic: str, positive: bool):
        if positive:
            current = self.utility_weights.get(semantic, 1.0)
            self.utility_weights[semantic] = min(2.0, current + 0.02)
        else:
            current = self.utility_weights.get(semantic, 1.0)
            self.utility_weights[semantic] = max(0.1, current - 0.05)

    def set_operation_mode(self, mode: str):
        if mode not in VALID_MODES:
            _LOGGER.warning("Ungültiger Modus: '%s'. Erlaubt: %s", mode, VALID_MODES)
            return False
        old = self.operation_mode
        self.operation_mode = mode
        self.shadow_mode = (mode == MODE_SHADOW)
        _LOGGER.info("Betriebsmodus: %s → %s", old, mode)
        return True

    def queue_confirm(self, decision: Decision, reasoning: str = "",
                      context: dict = None) -> str:
        confirm_id = f"c_{int(time.time())}_{decision.token_id}"
        self._pending_confirms[confirm_id] = {
            "decision": decision,
            "timestamp": time.time(),
            "reasoning": reasoning or "",
            "context": context or {},
        }
        self.total_confirms += 1
        cutoff = time.time() - 600
        self._pending_confirms = {
            k: v for k, v in self._pending_confirms.items()
            if v["timestamp"] > cutoff
        }
        return confirm_id

    def get_pending_confirm(self, confirm_id: str):
        entry = self._pending_confirms.pop(confirm_id, None)
        if entry:
            return entry["decision"]
        return None

    def peek_pending_confirm(self, confirm_id: str) -> dict:
        return self._pending_confirms.get(confirm_id)

    def get_all_pending_confirms(self) -> list:
        now = time.time()
        result = []
        for cid, entry in self._pending_confirms.items():
            d = entry["decision"]
            parts = d.token.split(".")
            room = parts[0] if len(parts) == 3 else ""
            semantic = parts[1] if len(parts) == 3 else ""
            action_label = parts[2] if len(parts) == 3 else d.token
            ctx = entry.get("context") or {}
            result.append({
                "id": cid,
                "token": d.token,
                "entity_id": d.entity_id,
                "room": room,
                "semantic": semantic,
                "action": action_label,
                "confidence": round(d.confidence, 3),
                "utility": round(d.utility, 3),
                "risk": round(d.risk, 3),
                "n_obs": d.n_obs,
                "source": d.source,
                "reasons": list(d.reasons or []),
                "reasoning": entry.get("reasoning", ""),
                "context": {
                    "mode": ctx.get("mode"),
                    "time_bucket": ctx.get("time_bucket"),
                    "bucket_id": ctx.get("bucket_id"),
                    "rule_key": ctx.get("rule_key"),
                    "rule_order": ctx.get("rule_order"),
                },
                "age_s": int(now - entry["timestamp"]),
                "expires_in_s": max(0, 600 - int(now - entry["timestamp"])),
            })
        result.sort(key=lambda r: -r["age_s"])
        return result

    def reject_pending(self, confirm_id: str, basal_ganglia=None,
                       amygdala=None) -> dict:
        entry = self._pending_confirms.pop(confirm_id, None)
        if not entry:
            return None
        decision = entry["decision"]
        parts = decision.token.split(".")
        semantic = parts[1] if len(parts) == 3 else ""
        if semantic:
            self.learn_from_feedback(semantic, positive=False)
        if amygdala and decision.token:
            try:
                amygdala.learn_from_feedback(decision.token, "negative")
            except Exception:
                pass
        if basal_ganglia and decision.entity_id:
            try:
                if decision.entity_id not in basal_ganglia.pending_actions:
                    basal_ganglia.register_action(
                        decision.entity_id, decision.token_id,
                        entry.get("context", {}).get("bucket_id", 0),
                        decision.token,
                    )
                basal_ganglia.process_outcome(decision.entity_id, positive=False)
            except Exception:
                pass
        self._feedback_log.append({
            "time": time.time(), "entity_id": decision.entity_id,
            "token": decision.token, "feedback": "rejected",
        })
        if len(self._feedback_log) > 100:
            self._feedback_log = self._feedback_log[-100:]
        return {
            "id": confirm_id,
            "token": decision.token,
            "entity_id": decision.entity_id,
            "semantic": semantic,
        }

    def to_dict(self) -> dict:
        return {
            "shadow_mode": self.shadow_mode,
            "operation_mode": self.operation_mode,
            "total_decisions": self.total_decisions,
            "total_executions": self.total_executions,
            "total_confirms": self.total_confirms,
            "overrides_detected": self.overrides_detected,
            "utility_weights": self.utility_weights,
            "activated_semantics": list(self.activated_semantics),
            "feedback_log": self._feedback_log[-20:],
        }

    def from_dict(self, data: dict):
        self.operation_mode = data.get("operation_mode", MODE_SHADOW)
        self.shadow_mode = data.get("shadow_mode", self.operation_mode == MODE_SHADOW)
        self.total_decisions = data.get("total_decisions", 0)
        self.total_executions = data.get("total_executions", 0)
        self.total_confirms = data.get("total_confirms", 0)
        self.overrides_detected = data.get("overrides_detected", 0)
        self.utility_weights = data.get("utility_weights", {})
        self.activated_semantics = set(data.get("activated_semantics", []))
        self._feedback_log = data.get("feedback_log", [])

    @property
    def stats(self) -> dict:
        return {
            "shadow_mode": self.shadow_mode,
            "operation_mode": self.operation_mode,
            "total_decisions": self.total_decisions,
            "total_executions": self.total_executions,
            "total_confirms": self.total_confirms,
            "overrides_detected": self.overrides_detected,
            "activated_semantics": list(self.activated_semantics),
            "utility_weights": self.utility_weights,
            "override_rate": f"{self.overrides_detected / max(1, self.total_decisions):.1%}",
            "pending_confirms": len(self._pending_confirms),
        }
