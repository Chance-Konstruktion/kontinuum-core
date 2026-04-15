"""Amygdala module for KONTINUUM Core.

Biological inspiration: Risk assessment and safety gate. The amygdala
evaluates proposed actions emotionally – above all for danger – and can
trigger a rapid "STOP!" before the prefrontal cortex has finished
reasoning.
"""

import logging
import time

_LOGGER = logging.getLogger(__name__)

NEVER_AUTO = {"alarm", "lock"}

RISK_SCORES = {
    "light": 0.05, "media": 0.05, "automation": 0.1,
    "switch": 0.2, "fan": 0.15, "cover": 0.3,
    "climate": 0.35, "water_heater": 0.4, "vacuum": 0.25,
    "lock": 0.9, "alarm": 1.0,
}

SAFE_STATES = {
    "light": {"on", "off"},
    "media": {"playing", "paused", "standby", "off"},
    "switch": {"on", "off"},
    "fan": {"on", "off"},
    "climate": {"heating", "cooling", "idle", "off"},
    "cover": {"open", "closed"},
    "automation": {"on"},
}


class Amygdala:
    """Assesses the risk of a proposed action."""

    def __init__(self):
        self.learned_risk = {}
        self.veto_log = []
        self.total_assessments = 0
        self.total_vetoes = 0
        self.total_cautions = 0

    def assess(self, token: str, semantic: str, room: str,
               state: str, confidence: float,
               context: dict = None) -> dict:
        """Assesses the risk of an action."""
        self.total_assessments += 1
        context = context or {}
        reasons = []

        # 1. Absolute vetoes
        if semantic in NEVER_AUTO:
            self.total_vetoes += 1
            reason = f"{semantic} is safety-critical"
            reasons.append(reason)
            self._log_veto(token, reason)
            return {
                "decision": "VETO", "risk": 1.0,
                "reasons": reasons, "required_confidence": float("inf"),
            }

        system_mode = context.get("system_mode", "home")
        if system_mode in ("away", "vacation"):
            if semantic not in ("automation",):
                reasons.append(f"System in {system_mode} mode")

        # 2. Base risk
        base_risk = RISK_SCORES.get(semantic, 0.3)

        # 2b. SAFE_STATES check
        if semantic in SAFE_STATES:
            if state not in SAFE_STATES[semantic]:
                base_risk = min(1.0, base_risk + 0.2)
                reasons.append(f"Unknown state '{state}' for {semantic}: +0.2")

        # 3. Context modifiers
        risk = base_risk

        if system_mode in ("away", "vacation"):
            risk = min(1.0, risk + 0.3)
            reasons.append("Away mode: +0.3 risk")

        is_night = context.get("is_night", False)
        if is_night and semantic == "light" and state == "on":
            risk = min(1.0, risk + 0.1)
            reasons.append("Light on at night: +0.1")

        learned_key = f"{room}.{semantic}.{state}"
        if learned_key in self.learned_risk:
            modifier = self.learned_risk[learned_key]
            risk = max(0.0, min(1.0, risk + modifier))
            reasons.append(f"Learned: {modifier:+.2f}")

        # 4. Confidence thresholds
        if risk < 0.15:
            required_conf = 0.3
        elif risk < 0.3:
            required_conf = 0.5
        elif risk < 0.5:
            required_conf = 0.65
        elif risk < 0.7:
            required_conf = 0.8
        else:
            required_conf = 0.95

        # 5. Decision
        if risk >= 0.95:
            decision = "VETO"
            self.total_vetoes += 1
            reasons.append(f"Risk {risk:.2f} >= 0.95")
            self._log_veto(token, reasons[-1])
        elif confidence < required_conf:
            decision = "CAUTION"
            self.total_cautions += 1
            reasons.append(f"Conf {confidence:.0%} < {required_conf:.0%}")
        else:
            decision = "ALLOW"
            reasons.append(f"Conf {confidence:.0%} >= {required_conf:.0%}")

        return {
            "decision": decision, "risk": round(risk, 3),
            "reasons": reasons, "required_confidence": required_conf,
        }

    def learn_from_feedback(self, token: str, feedback: str):
        """Adjusts risk based on user feedback."""
        if feedback == "negative":
            current = self.learned_risk.get(token, 0.0)
            self.learned_risk[token] = min(0.5, current + 0.05)
        elif feedback == "positive":
            current = self.learned_risk.get(token, 0.0)
            self.learned_risk[token] = max(-0.2, current - 0.02)

    def _log_veto(self, token: str, reason: str):
        self.veto_log.append({"timestamp": time.time(), "token": token, "reason": reason})
        if len(self.veto_log) > 50:
            self.veto_log = self.veto_log[-50:]
        _LOGGER.info("VETO: %s – %s", token, reason)

    def to_dict(self) -> dict:
        return {
            "learned_risk": self.learned_risk,
            "total_assessments": self.total_assessments,
            "total_vetoes": self.total_vetoes,
            "total_cautions": self.total_cautions,
        }

    def from_dict(self, data: dict):
        self.learned_risk = data.get("learned_risk", {})
        self.total_assessments = data.get("total_assessments", 0)
        self.total_vetoes = data.get("total_vetoes", 0)
        self.total_cautions = data.get("total_cautions", 0)

    @property
    def stats(self) -> dict:
        return {
            "total_assessments": self.total_assessments,
            "total_vetoes": self.total_vetoes,
            "total_cautions": self.total_cautions,
            "learned_risks": len(self.learned_risk),
            "veto_rate": f"{self.total_vetoes / max(1, self.total_assessments):.1%}",
            "recent_vetos": self.veto_log[-5:],
        }
