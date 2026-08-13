"""Hypothalamus module for KONTINUUM Core.

Biological inspiration: Homeostasis monitor for energy, climate and
environment. The hypothalamus regulates inner balance – temperature,
hunger, sleep-wake cycle. Here it absorbs ~95% of energy/climate
noise events and forwards tokens only when the environment changes
significantly.
"""

import logging
import time

_LOGGER = logging.getLogger(__name__)

ENERGY_SEMANTICS = {"battery", "voltage", "power", "energy", "solar", "grid"}
CLIMATE_SEMANTICS = {"temperature", "humidity", "illuminance", "pressure", "co2"}

STATE_TO_LEVEL = {
    "critical": 0, "low": 0, "none": 0, "cold": 0, "dry": 0, "dark": 0,
    "export": 0, "good": 0,
    "niedrig": 1, "wenig": 1, "cool": 1, "dim": 1, "minimal": 1,
    "moderate": 1, "normal": 1,
    "medium": 2, "mittel": 2, "comfort": 2, "bright": 2, "import": 2,
    "full": 3, "high": 3, "viel": 3, "warm": 3, "very_bright": 3,
    "very_hot": 4, "hot": 4, "humid": 3, "poor": 3,
    "lädt": 3, "charging": 3,
}


class HomeoState:
    """Current homeostasis state."""

    def __init__(self):
        self.battery_state = 2
        self.solar_state = 0
        self.power_consumption = 1
        self.grid_state = 1
        self.indoor_temp = 2
        self.outdoor_temp = 1
        self.humidity = 1
        self.brightness = 1
        self.heating_active = False
        self.heating_demand = 0
        # Trend tracking
        self._prev_temp = None
        self._prev_temp_time = 0
        self._prev_battery = None
        self._prev_battery_time = 0
        self._prev_solar = None
        self._prev_solar_time = 0
        self.trend_temp = 0.0       # Δlevel/h normalised [-1, 1]
        self.trend_battery = 0.0
        self.trend_solar = 0.0

    def to_context_vector(self) -> list:
        """9-dimensional normalised context vector."""
        return [
            self.battery_state / 3.0,
            self.solar_state / 3.0,
            self.power_consumption / 3.0,
            min(self.indoor_temp / 4.0, 1.0),
            self.brightness / 3.0,
            1.0 if self.heating_active else 0.0,
            max(0.0, min(1.0, (self.trend_temp + 1) / 2)),
            max(0.0, min(1.0, (self.trend_battery + 1) / 2)),
            max(0.0, min(1.0, (self.trend_solar + 1) / 2)),
        ]

    def update_trend(self, semantic: str, level: int):
        """Computes trend coefficients from level deltas."""
        now = time.time()
        if semantic == "temperature":
            if self._prev_temp is not None and (now - self._prev_temp_time) > 60:
                hours = (now - self._prev_temp_time) / 3600
                self.trend_temp = max(-1, min(1, (level - self._prev_temp) / max(hours, 0.1)))
            self._prev_temp = level
            self._prev_temp_time = now
        elif semantic == "battery":
            if self._prev_battery is not None and (now - self._prev_battery_time) > 60:
                hours = (now - self._prev_battery_time) / 3600
                self.trend_battery = max(-1, min(1, (level - self._prev_battery) / max(hours, 0.1)))
            self._prev_battery = level
            self._prev_battery_time = now
        elif semantic == "solar":
            if self._prev_solar is not None and (now - self._prev_solar_time) > 60:
                hours = (now - self._prev_solar_time) / 3600
                self.trend_solar = max(-1, min(1, (level - self._prev_solar) / max(hours, 0.1)))
            self._prev_solar = level
            self._prev_solar_time = now


class Hypothalamus:
    """Homeostasis monitor for KONTINUUM."""

    ENERGY_COOLDOWN = 60
    CLIMATE_COOLDOWN = 1800

    def __init__(self):
        self.state = HomeoState()
        self.events_absorbed = 0
        self.last_energy_state = None
        self.last_climate_state = None
        self._last_energy_event_time = 0
        self._last_climate_event_time = 0

    def is_hypothalamus_signal(self, semantic: str) -> bool:
        return semantic in ENERGY_SEMANTICS or semantic in CLIMATE_SEMANTICS

    def absorb(self, room: str, semantic: str, state: str,
               entity_id: str = "") -> dict:
        """Absorbs an energy/climate event and updates state.

        Returns a transition token dict on significant change, else None.
        """
        self.events_absorbed += 1
        level = STATE_TO_LEVEL.get(state, 1)

        if semantic in ("temperature", "battery", "solar"):
            self.state.update_trend(semantic, level)

        if semantic == "battery":
            self.state.battery_state = level
        elif semantic == "solar":
            self.state.solar_state = level
        elif semantic == "power":
            self.state.power_consumption = level
        elif semantic == "grid":
            self.state.grid_state = level
        elif semantic == "voltage":
            pass
        elif semantic == "energy":
            pass
        elif semantic == "temperature":
            if "outdoor" in (room or "") or "außen" in (entity_id or "").lower():
                self.state.outdoor_temp = level
            else:
                self.state.indoor_temp = level
        elif semantic == "humidity":
            self.state.humidity = level
        elif semantic == "illuminance":
            self.state.brightness = level
        elif semantic == "co2":
            pass

        if semantic == "climate":
            self.state.heating_active = state in ("heating", "heat")

        if semantic in ENERGY_SEMANTICS:
            return self._check_energy_transition()
        if semantic in CLIMATE_SEMANTICS:
            return self._check_climate_transition()
        return None

    def _check_energy_transition(self) -> dict:
        battery = self.state.battery_state
        solar = self.state.solar_state
        current = (battery, solar)

        if current == self.last_energy_state:
            return None

        now = time.time()
        if (now - self._last_energy_event_time) < self.ENERGY_COOLDOWN:
            self.last_energy_state = current
            return None

        self.last_energy_state = current
        self._last_energy_event_time = now

        if battery <= 0:
            energy_state = "critical"
        elif battery <= 1 and solar <= 1:
            energy_state = "low"
        elif solar >= 2:
            energy_state = "charging"
        else:
            energy_state = "normal"

        _LOGGER.info("Hypothalamus: energy transition → house.energy.%s", energy_state)
        return {
            "token": f"house.energy.{energy_state}",
            "room": "house",
            "semantic": "energy_state",
            "state": energy_state,
        }

    def _check_climate_transition(self) -> dict:
        current = self.state.indoor_temp

        if current == self.last_climate_state:
            return None

        now = time.time()
        if (now - self._last_climate_event_time) < self.CLIMATE_COOLDOWN:
            self.last_climate_state = current
            return None

        self.last_climate_state = current
        self._last_climate_event_time = now

        temp_names = {0: "cold", 1: "cool", 2: "comfort", 3: "warm", 4: "hot"}
        climate_state = temp_names.get(current, "comfort")

        return {
            "token": f"house.climate.{climate_state}",
            "room": "house",
            "semantic": "climate_state",
            "state": climate_state,
        }

    def get_context_vector(self) -> list:
        return self.state.to_context_vector()

    def get_energy_summary(self) -> dict:
        battery_names = {0: "critical", 1: "low", 2: "normal", 3: "charging"}
        solar_names = {0: "none", 1: "low", 2: "medium", 3: "high"}
        power_names = {0: "none", 1: "low", 2: "medium", 3: "high"}
        return {
            "battery": battery_names.get(self.state.battery_state, "?"),
            "solar": solar_names.get(self.state.solar_state, "?"),
            "consumption": power_names.get(self.state.power_consumption, "?"),
            "heating_active": self.state.heating_active,
            "indoor_temp": self.state.indoor_temp,
            "outdoor_temp": self.state.outdoor_temp,
        }

    def to_dict(self) -> dict:
        return {
            "battery_state": self.state.battery_state,
            "solar_state": self.state.solar_state,
            "power_consumption": self.state.power_consumption,
            "grid_state": self.state.grid_state,
            "indoor_temp": self.state.indoor_temp,
            "outdoor_temp": self.state.outdoor_temp,
            "humidity": self.state.humidity,
            "brightness": self.state.brightness,
            "heating_active": self.state.heating_active,
            "heating_demand": self.state.heating_demand,
            "events_absorbed": self.events_absorbed,
            "last_energy_state": self.last_energy_state,
            "last_climate_state": self.last_climate_state,
            "last_energy_event_time": self._last_energy_event_time,
            "last_climate_event_time": self._last_climate_event_time,
            "trend_temp": self.state.trend_temp,
            "trend_battery": self.state.trend_battery,
            "trend_solar": self.state.trend_solar,
        }

    def from_dict(self, data: dict):
        self.state.battery_state = data.get("battery_state", 2)
        self.state.solar_state = data.get("solar_state", 0)
        self.state.power_consumption = data.get("power_consumption", 1)
        self.state.grid_state = data.get("grid_state", 1)
        self.state.indoor_temp = data.get("indoor_temp", 2)
        self.state.outdoor_temp = data.get("outdoor_temp", 1)
        self.state.humidity = data.get("humidity", 1)
        self.state.brightness = data.get("brightness", 1)
        self.state.heating_active = data.get("heating_active", False)
        self.state.heating_demand = data.get("heating_demand", 0)
        self.events_absorbed = data.get("events_absorbed", 0)
        self.last_energy_state = data.get("last_energy_state")
        self.last_climate_state = data.get("last_climate_state")
        self._last_energy_event_time = data.get("last_energy_event_time", 0)
        self._last_climate_event_time = data.get("last_climate_event_time", 0)
        self.state.trend_temp = data.get("trend_temp", 0.0)
        self.state.trend_battery = data.get("trend_battery", 0.0)
        self.state.trend_solar = data.get("trend_solar", 0.0)

    @property
    def stats(self) -> dict:
        return {
            "events_absorbed": self.events_absorbed,
            "energy": self.get_energy_summary(),
            "context_vector": self.state.to_context_vector(),
        }
