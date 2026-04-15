import gzip
import json
import logging
import os
from datetime import datetime, timezone
from typing import Callable, Optional

_LOGGER = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "hippocampus": {"learning_rate": 0.05, "forgetting_rate": 0.001},
    "cerebellum": {"learning_rate": 0.1, "forgetting_rate": 0.005, "min_confidence": 0.5},
    "basal_ganglia": {"learning_rate": 0.1, "q_discount": 0.9},
    "reticular": {"priority_threshold": 0.3},
    "accumbens": {"reward_scaling": 1.0},
    "locus": {"arousal_sensitivity": 0.5},
}


class MetaPlasticity:
    """
    Lernt, wie gut andere Module lernen, und passt deren Parameter dynamisch an.

    HA-frei: Statt HomeAssistant werden injiziert:
      storage_path  – Verzeichnis für persistente Daten (optional)
      scheduler     – Scheduler-Protocol-Instanz (optional)
      brain_modules – Dict der anderen Brain-Module (optional)
      event_callback – Callable(event_type, data) für Events (optional)
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        scheduler=None,
        brain_modules: Optional[dict] = None,
        event_callback: Optional[Callable] = None,
    ):
        self._storage_path = storage_path
        self._scheduler = scheduler
        self._brain_modules = brain_modules or {}
        self._event_callback = event_callback
        self.data = {
            "version": 1,
            "last_update": None,
            "module_params": {},
            "metric_history": {},
        }

    @property
    def _path(self) -> Optional[str]:
        if not self._storage_path:
            return None
        return os.path.join(self._storage_path, "metaplasticity.json.gz")

    def load(self):
        self._init_defaults()
        path = self._path
        if not path or not os.path.exists(path):
            return
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                loaded = json.load(f)
            self.data.update(loaded or {})
            self._init_defaults()
        except Exception:
            _LOGGER.exception("MetaPlasticity load failed")

    def save(self):
        path = self._path
        if not path:
            return
        tmp = path + ".tmp"
        payload = dict(self.data)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with gzip.open(tmp, "wt", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(tmp, path)
        except Exception:
            _LOGGER.exception("MetaPlasticity save failed")

    def _init_defaults(self):
        params = self.data.setdefault("module_params", {})
        for module, defaults in DEFAULT_PARAMS.items():
            current = params.get(module, {})
            merged = dict(defaults)
            merged.update(current)
            params[module] = merged
        self.data.setdefault("metric_history", {})

    def start(self, interval_hours: int = 24):
        """Startet die periodische Parameter-Anpassung via Scheduler."""
        if self._scheduler:
            self._scheduler.schedule_interval(
                self._update_parameters, interval_hours * 3600
            )

    def get_params(self, module_name: str) -> dict:
        self._init_defaults()
        return dict(self.data["module_params"].get(module_name, {}))

    def set_params(self, module_name: str, new_values: dict):
        self._init_defaults()
        base = self.data["module_params"].setdefault(module_name, {})
        base.update(new_values or {})

    def _update_parameters(self, _now=None):
        try:
            metrics_by_module = self._collect_metrics()
            for module_name, metrics in metrics_by_module.items():
                self._adjust_parameters(module_name, metrics)
            self.data["last_update"] = datetime.now(timezone.utc).isoformat()
            self.save()
            if self._event_callback:
                self._event_callback("kontinuum_metaplasticity_updated", {
                    "last_update": self.data["last_update"],
                    "modules": list(metrics_by_module.keys()),
                })
        except Exception:
            _LOGGER.exception("MetaPlasticity update failed")

    def _collect_metrics(self) -> dict:
        brain = self._brain_modules
        hp = brain.get("hippocampus")
        cb = brain.get("cerebellum")
        bg = brain.get("basal_ganglia")
        ret = brain.get("reticular")
        acc = brain.get("accumbens")
        loc = brain.get("locus")

        return {
            "hippocampus": {
                "error": max(0.0, 1.0 - getattr(hp, "accuracy", 0.0)) if hp else 0.5,
                "success_rate": getattr(hp, "accuracy", 0.0) if hp else 0.0,
                "confidence": getattr(hp, "accuracy", 0.0) if hp else 0.0,
            },
            "cerebellum": {
                "error": 0.2,
                "success_rate": float((cb.stats.get("success_rate", "0%").strip("%") or 0)) / 100.0 if cb else 0.0,
                "confidence": cb.MIN_CONFIDENCE if cb else 0.0,
            },
            "basal_ganglia": {
                "error": max(0.0, 0.3 - abs(bg.stats.get("dopamine_signal", 0.0))) if bg else 0.3,
                "success_rate": 0.0 if not bg else bg.total_positive / max(1, bg.total_positive + bg.total_negative),
                "confidence": max(0.0, min(1.0, (bg.stats.get("go_actions", 0) / max(1, bg.stats.get("q_entries", 1))))) if bg else 0.0,
            },
            "reticular": {
                "error": min(1.0, (ret.filtered_events / 1000.0)) if ret else 0.0,
                "success_rate": 1.0 - min(1.0, (ret.filtered_events / 2000.0)) if ret else 0.5,
                "confidence": ret.get_priority("global") if ret else 0.5,
            },
            "accumbens": {
                "error": 0.2,
                "success_rate": min(1.0, len(acc.success_counts) / 100.0) if acc else 0.0,
                "confidence": min(1.0, len(acc.values) / 200.0) if acc else 0.0,
            },
            "locus": {
                "error": 0.0 if not loc else abs(0.4 - loc.get_arousal()),
                "success_rate": 0.7 if loc else 0.0,
                "confidence": 1.0 if loc else 0.0,
            },
        }

    def _adjust_parameters(self, module_name: str, metrics: dict):
        self._init_defaults()
        params = self.data["module_params"].setdefault(module_name, {})
        hist = self.data["metric_history"].setdefault(module_name, [])
        hist.append(metrics)
        if len(hist) > 10:
            hist[:] = hist[-10:]

        error = sum(m.get("error", 0.0) for m in hist) / len(hist)
        success_rate = sum(m.get("success_rate", 0.0) for m in hist) / len(hist)
        confidence = sum(m.get("confidence", 0.0) for m in hist) / len(hist)

        old = dict(params)
        current_lr = float(params.get("learning_rate", 0.1))
        current_fr = float(params.get("forgetting_rate", 0.005))

        if error > 0.2:
            params["learning_rate"] = min(0.3, current_lr * 1.1)
            params["forgetting_rate"] = max(0.0005, current_fr * 0.95)
        elif error < 0.05 and success_rate > 0.8:
            params["learning_rate"] = max(0.02, current_lr * 0.98)
            params["forgetting_rate"] = min(0.01, current_fr * 1.02)

        if success_rate > 0.8 and confidence > 0.6:
            if "q_discount" in params:
                params["q_discount"] = min(0.99, float(params["q_discount"]) + 0.01)
        if len(hist) >= 10 and self._no_improvement(hist):
            params["forgetting_rate"] = min(0.02, float(params.get("forgetting_rate", 0.005)) * 1.1)

        if params != old:
            _LOGGER.info("MetaPlasticity adjusted %s: %s -> %s", module_name, old, params)

    @staticmethod
    def _no_improvement(hist: list) -> bool:
        first = hist[:5]
        last = hist[-5:]
        first_error = sum(x.get("error", 0) for x in first) / max(1, len(first))
        last_error = sum(x.get("error", 0) for x in last) / max(1, len(last))
        return last_error >= first_error * 0.98

    def to_dict(self) -> dict:
        return dict(self.data)

    def from_dict(self, data: dict):
        self.data.update(data or {})
        self._init_defaults()


# Backwards-compatible alias (engine.py imports as `Metaplasticity`)
Metaplasticity = MetaPlasticity
