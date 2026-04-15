"""
╔══════════════════════════════════════════════════════════════════╗
║  KONTINUUM – Thalamus                                           ║
║  Sensorisches Tor: Filtert, normalisiert, tokenisiert           ║
║                                                                  ║
║  Biologisches Vorbild:                                           ║
║  Der Thalamus ist die Relaisstation des Gehirns. Er filtert     ║
║  sensorische Eingaben, entfernt Rauschen und leitet relevante   ║
║  Signale an die richtigen Verarbeitungszentren weiter.          ║
╚══════════════════════════════════════════════════════════════════╝
"""

import logging
import math
import re
from pathlib import Path
from datetime import datetime, timezone

_LOGGER = logging.getLogger(__name__)

# ── Raum-Zuordnung ──────────────────────────────────────────────

HA_AREA_MAP = {
    "schlafzimmer": "bedroom", "bedroom": "bedroom",
    "wohnzimmer": "livingroom", "living": "livingroom", "wohnbereich": "livingroom",
    "küche": "kitchen", "kitchen": "kitchen",
    "bad": "bathroom", "bath": "bathroom", "badezimmer": "bathroom",
    "flur": "hallway", "hall": "hallway", "corridor": "hallway", "diele": "hallway",
    "büro": "office", "office": "office", "arbeitszimmer": "office",
    "kinderzimmer": "kidsroom", "kids": "kidsroom",
    "garten": "outdoor", "garden": "outdoor", "terrasse": "outdoor",
    "balkon": "outdoor", "balcony": "outdoor", "outdoor": "outdoor",
    "einfahrt": "outdoor", "carport": "outdoor", "hof": "outdoor",
    "garage": "garage",
    "keller": "basement", "basement": "basement",
    "hauswirtschaft": "utility", "utility": "utility",
    "waschküche": "utility", "laundry": "utility",
    "technikraum": "utility", "heizungsraum": "utility",
    "dachboden": "attic", "attic": "attic",
    "eingang": "entrance", "entrance": "entrance", "windfang": "entrance",
    "gästezimmer": "guestroom", "guest": "guestroom",
    "esszimmer": "diningroom", "dining": "diningroom",
}

NAME_ROOM_HINTS = {
    # Deutsch
    "schlaf": "bedroom", "bett": "bedroom", "bedroom": "bedroom",
    "wohn": "livingroom", "living": "livingroom", "lounge": "livingroom",
    "küche": "kitchen", "kitchen": "kitchen", "kochen": "kitchen",
    "bad": "bathroom", "bath": "bathroom", "dusche": "bathroom", "wc": "bathroom",
    "büro": "office", "office": "office", "desk": "office", "arbeit": "office",
    "flur": "hallway", "hall": "hallway", "gang": "hallway", "diele": "hallway",
    "eingang": "entrance", "entrance": "entrance", "haustür": "entrance",
    "garten": "outdoor", "garden": "outdoor", "outdoor": "outdoor",
    "terrasse": "outdoor", "balkon": "outdoor", "terrace": "outdoor",
    "einfahrt": "outdoor", "carport": "outdoor",
    "garage": "garage",
    "keller": "basement", "basement": "basement",
    "wasch": "utility", "utility": "utility", "hauswirtschaft": "utility",
    "heizung": "utility", "technik": "utility", "server": "utility",
    "kind": "kidsroom", "kids": "kidsroom", "nursery": "kidsroom",
    "gäste": "guestroom", "guest": "guestroom",
    "dach": "attic", "attic": "attic", "speicher": "attic",
    "esszimmer": "diningroom", "dining": "diningroom",
    "fernseh": "livingroom", "tv": "livingroom", "wiz_wohn": "livingroom",
    # IP-Kamera Patterns
    "ipcam": "outdoor",
}

# ── Semantik-Zuordnung ──────────────────────────────────────────

DOMAIN_SEMANTIC = {
    "light": "light",
    "switch": "switch",
    "fan": "fan",
    "cover": "cover",
    "climate": "climate",
    "media_player": "media",
    "vacuum": "vacuum",
    "lock": "lock",
    "alarm_control_panel": "alarm",
    "automation": "automation",
    "water_heater": "water_heater",
}

SENSOR_KEYWORDS = {
    "temperature": "temperature", "temp": "temperature",
    "humidity": "humidity", "feucht": "humidity",
    "power": "power", "watt": "power", "leistung": "power",
    "energy": "energy", "energie": "energy", "kwh": "energy",
    "battery": "battery", "batterie": "battery", "akku": "battery",
    "voltage": "voltage", "spannung": "voltage",
    "illuminance": "illuminance", "lux": "illuminance", "hell": "illuminance",
    "pressure": "pressure", "druck": "pressure",
    "solar": "solar", "pv_": "solar", "photovoltaic": "solar",
    "grid": "grid", "netz": "grid",
    "co2": "co2", "carbon": "co2",
    "cpu": "cpu", "processor": "cpu",
    "gpu": "gpu", "graphics": "gpu",
    "tpms": "tpms", "tire": "tpms", "reifen": "tpms",
    "heart": "heartrate", "pulse": "heartrate", "puls": "heartrate",
    "step": "steps", "schritt": "steps",
    "sleep": "sleep",
    "gaming": "gaming",
    "network": "network", "dhcp": "network", "router": "network",
    "wallbox": "wallbox", "charger": "wallbox", "laden": "wallbox",
    "bed": "bed_presence", "bett": "bed_presence",
    "screen": "screen", "display": "screen",
}


class Thalamus:
    """
    Sensorisches Tor von KONTINUUM.

    Aufgaben:
    1. Entity → Raum zuordnen (aus HA Areas, Names, IDs)
    2. Entity → Semantischer Typ (light, motion, temperature, ...)
    3. State-Change → Token erzeugen (z.B. "bedroom.light.on")
    4. Duplikate filtern (gleicher Token hintereinander)
    5. Token-Vokabular verwalten (bidirektionale Map)
    """

    # Domains die wir komplett ignorieren
    IGNORED_DOMAINS = {
        "persistent_notification", "scene", "script", "zone",
        "input_boolean", "input_number", "input_select", "input_text",
        "input_datetime", "counter", "timer", "group", "logbook",
        "recorder", "homeassistant", "frontend", "system_log",
        "updater", "update", "tts", "stt", "conversation",
    }

    IGNORED_PATTERNS = [
        re.compile(r"^sensor\.kontinuum_"),
        re.compile(r"_linkquality$"),
        re.compile(r"_signal_strength$"),
        re.compile(r"_last_seen$"),
        re.compile(r"_pre_release$"),
        re.compile(r"^switch\.\w+_pre_release"),
        re.compile(r"_cell_voltage[s]?_\d+"),
        re.compile(r"_cell_\d+"),
        re.compile(r"^sensor\.pve.*_net(in|out|_in|_out)"),
        re.compile(r"^sensor\.pve.*_(netin|netout|read|write)"),
        re.compile(r"_netin$"), re.compile(r"_netout$"),
        re.compile(r"_uptime$"), re.compile(r"_runtime$"),
        re.compile(r"_operating_hours$"), re.compile(r"_run_time$"),
        re.compile(r"_disk_(read|write)$"),
        re.compile(r"^sensor\.home_assistant_"),
        re.compile(r"^sensor\.hassio_"),
        re.compile(r"^binary_sensor\.hassio_"),
        re.compile(r"^sensor\.supervisor_"),
        re.compile(r"^binary_sensor\.supervisor_"),
        re.compile(r"^binary_sensor\.\w+_running$"),
        re.compile(r"_cpu_prozent$"),
        re.compile(r"_cpu_percent$"),
        re.compile(r"_memory_percent$"),
        re.compile(r"_speicher_prozent$"),
        re.compile(r"^sensor\.pve_"),
        re.compile(r"^sensor\.proxmox_"),
        re.compile(r"^binary_sensor\.pve_"),
        re.compile(r"^sensor\.nas_"),
        re.compile(r"^sensor\.synology_"),
        re.compile(r"^sensor\.truenas_"),
        re.compile(r"^sensor\.qnap_"),
        re.compile(r"^sensor\.unraid_"),
        re.compile(r"^sensor\.docker_"),
        re.compile(r"^sensor\.portainer_"),
        re.compile(r"^sensor\.server_"),
        re.compile(r"^sensor\.pi_hole_"),
        re.compile(r"^sensor\.adguard_"),
        re.compile(r"^sensor\.speedtest_"),
        re.compile(r"^sensor\.unifi_"),
        re.compile(r"^sensor\.fritzbox_"),
        re.compile(r"^sensor\.glances_"),
    ]

    MAX_VOCAB_SIZE = 5000

    def __init__(self):
        self.entity_room = {}
        self.entity_semantic = {}
        self.token_to_id = {}
        self.id_to_token = {}
        self._next_id = 1
        self.entity_last_token = {}
        self._sun_elevation = 0.0
        self._sun_is_daylight = False
        self._unassigned_entities = {}
        self._unassigned_event_counts = {}
        self._ignored_entities = set()
        self._included_entities = set()
        self.track_mode = "standard"
        self.stats = {
            "entities_registered": 0,
            "rooms_discovered": 0,
            "tokens_filtered": 0,
            "events_processed": 0,
        }
        self._known_rooms = set()
        self.custom_semantic_rules = []
        self.custom_thresholds = {}

    def should_track(self, entity_id: str, domain: str = "",
                     labels: list = None) -> bool:
        label_names = [l.lower().strip() for l in (labels or [])]

        if any(n in ("ignore_kontinuum", "ignore kontinuum") for n in label_names):
            self._ignored_entities.add(entity_id)
            self.stats["entities_ignored"] = len(self._ignored_entities)
            return False

        has_include_label = any(n == "kontinuum" for n in label_names)
        if has_include_label:
            self._included_entities.add(entity_id)

        if not domain:
            domain = entity_id.split(".")[0] if "." in entity_id else ""

        if domain in self.IGNORED_DOMAINS:
            if has_include_label:
                return True
            return False

        for pat in self.IGNORED_PATTERNS:
            if pat.search(entity_id):
                if has_include_label:
                    return True
                return False

        if self.track_mode == "labeled":
            return has_include_label

        if self.track_mode == "auto":
            if has_include_label:
                return True
            return self._heuristic_filter(entity_id, domain)

        return True

    BEHAVIOR_DOMAINS = {
        "light", "switch", "binary_sensor", "climate", "cover",
        "fan", "media_player", "lock", "vacuum", "alarm_control_panel",
        "person", "device_tracker", "sensor",
    }

    RELEVANT_SENSOR_PATTERNS = [
        re.compile(r"temperature|temp"),
        re.compile(r"humidity|feucht"),
        re.compile(r"illumin|lux|helligkeit"),
        re.compile(r"power|energy|energie|watt|kwh"),
        re.compile(r"presence|pr[äa]senz|occupancy|besetzt"),
        re.compile(r"motion|bewegung"),
        re.compile(r"door|t[üu]r|window|fenster"),
        re.compile(r"contact|kontakt"),
        re.compile(r"co2|voc|air_quality|luft"),
    ]

    NOISE_SENSOR_PATTERNS = [
        re.compile(r"battery|batterie|akku"),
        re.compile(r"signal|rssi|linkquality"),
        re.compile(r"update|firmware|version"),
        re.compile(r"uptime|laufzeit"),
        re.compile(r"ip_address|mac_address"),
        re.compile(r"last_seen|zuletzt_gesehen"),
    ]

    def _heuristic_filter(self, entity_id: str, domain: str) -> bool:
        if domain not in self.BEHAVIOR_DOMAINS:
            return False
        if domain != "sensor":
            return True
        eid_lower = entity_id.lower()
        for pat in self.NOISE_SENSOR_PATTERNS:
            if pat.search(eid_lower):
                return False
        for pat in self.RELEVANT_SENSOR_PATTERNS:
            if pat.search(eid_lower):
                return True
        return False

    def register_entity(self, entity_id: str, ha_area: str = "",
                        device_class: str = "", domain: str = "",
                        friendly_name: str = "", unit: str = "",
                        labels: list = None):
        if not entity_id:
            return
        if not domain:
            domain = entity_id.split(".")[0] if "." in entity_id else ""
        if not self.should_track(entity_id, domain, labels):
            return
        room = self._resolve_room(entity_id, ha_area, friendly_name, labels)
        semantic = self._resolve_semantic(entity_id, domain, device_class,
                                          friendly_name, unit)
        if not semantic:
            return
        if room == "unknown":
            self.stats["entities_filtered"] = self.stats.get("entities_filtered", 0) + 1
            self._unassigned_entities[entity_id] = {
                "semantic": semantic,
                "name": friendly_name or entity_id,
                "domain": domain,
            }
            suggested = self.suggest_room(entity_id)
            if suggested:
                room = suggested
            else:
                return
        self.entity_room[entity_id] = room
        self.entity_semantic[entity_id] = semantic
        if room not in self._known_rooms:
            self._known_rooms.add(room)
            self.stats["rooms_discovered"] = len(self._known_rooms)
        self.stats["entities_registered"] = len(self.entity_semantic)

    def _resolve_room(self, entity_id: str, ha_area: str, friendly_name: str,
                       labels: list = None) -> str:
        if ha_area:
            area_lower = ha_area.lower().strip()
            for pattern, room in HA_AREA_MAP.items():
                if pattern == area_lower or pattern in area_lower:
                    return room
            slug = re.sub(r"[^a-z0-9_]", "", area_lower.replace(" ", "_"))
            if slug:
                return slug
        if labels:
            for label in labels:
                label_lower = label.lower().strip()
                for pattern, room in HA_AREA_MAP.items():
                    if pattern == label_lower or pattern in label_lower:
                        return room
        if friendly_name:
            name_lower = friendly_name.lower()
            for hint, room in NAME_ROOM_HINTS.items():
                if hint in name_lower:
                    return room
        eid_lower = entity_id.lower()
        for hint, room in NAME_ROOM_HINTS.items():
            if hint in eid_lower:
                return room
        return "unknown"

    def _resolve_semantic(self, entity_id: str, domain: str,
                          device_class: str, friendly_name: str,
                          unit: str) -> str:
        custom_semantic = self._resolve_custom_semantic(
            entity_id, domain, device_class, friendly_name, unit)
        if custom_semantic:
            return custom_semantic
        if domain in DOMAIN_SEMANTIC:
            return DOMAIN_SEMANTIC[domain]
        if domain == "binary_sensor":
            dc = (device_class or "").lower()
            if dc in ("motion", "occupancy"):
                return "motion"
            if dc in ("door", "window", "garage_door", "opening"):
                return "door"
            if dc in ("presence", "connectivity"):
                return "presence"
            if dc in ("running", "problem"):
                return "network"
            name = (friendly_name or entity_id).lower()
            if any(k in name for k in ("motion", "bewegung", "pir")):
                return "motion"
            if any(k in name for k in ("door", "tür", "fenster", "window")):
                return "door"
            if any(k in name for k in ("presence", "anwesen", "besetzt")):
                return "presence"
            if any(k in name for k in ("bett", "bed", "sleep", "schlaf")):
                return "bed_presence"
            if any(k in name for k in ("gaming", "game")):
                return "gaming"
            if any(k in name for k in ("ping", "netz", "network", "online")):
                return "network"
            return "binary"
        if domain == "sensor":
            return self._classify_sensor(entity_id, device_class,
                                          friendly_name, unit)
        if domain in ("device_tracker", "person"):
            return "tracker"
        if domain in ("button", "number"):
            return None
        return None

    def _classify_sensor(self, entity_id: str, device_class: str,
                         friendly_name: str, unit: str) -> str:
        dc = (device_class or "").lower()
        name = (friendly_name or entity_id).lower()
        unit_lower = (unit or "").lower()
        unit_map = {
            "temperature": "temperature", "humidity": "humidity",
            "battery": "battery", "power": "power", "energy": "energy",
            "voltage": "voltage", "illuminance": "illuminance",
            "pressure": "pressure", "co2": "co2", "aqi": "co2", "speed": "network",
        }
        if dc in unit_map:
            return unit_map[dc]
        if unit_lower in ("°c", "°f"):
            return "temperature"
        if unit_lower in ("%",) and "battery" in name:
            return "battery"
        if unit_lower in ("w", "kw"):
            return "power"
        if unit_lower in ("kwh", "wh"):
            return "energy"
        if unit_lower in ("v",):
            return "voltage"
        if unit_lower in ("lx", "lux"):
            return "illuminance"
        if unit_lower in ("bpm",):
            return "heartrate"
        if unit_lower in ("steps", "step"):
            return "steps"
        if unit_lower in ("ms", "mb/s", "mbit/s"):
            return "network"
        for keyword, semantic in SENSOR_KEYWORDS.items():
            if keyword in name or keyword in entity_id.lower():
                return semantic
        return None

    def _resolve_custom_semantic(self, entity_id: str, domain: str,
                                 device_class: str, friendly_name: str,
                                 unit: str) -> str:
        eid = entity_id.lower()
        dc = (device_class or "").lower()
        name = (friendly_name or "").lower()
        unit_lower = (unit or "").lower()
        for rule in self.custom_semantic_rules:
            target = rule.get("semantic")
            if not target:
                continue
            r_domain = (rule.get("domain") or "").lower()
            r_device_class = (rule.get("device_class") or "").lower()
            r_entity_regex = rule.get("entity_regex")
            r_name_contains = (rule.get("name_contains") or "").lower()
            r_unit = (rule.get("unit") or "").lower()
            if r_domain and r_domain != domain:
                continue
            if r_device_class and r_device_class != dc:
                continue
            if r_unit and r_unit != unit_lower:
                continue
            if r_name_contains and r_name_contains not in name:
                continue
            if r_entity_regex and not re.search(r_entity_regex, eid):
                continue
            return target
        return None

    def process(self, entity_id: str, new_state: str, old_state: str,
                timestamp=None) -> dict:
        if entity_id in self._ignored_entities:
            return None
        if entity_id not in self.entity_semantic:
            return None
        semantic = self.entity_semantic[entity_id]
        room = self.entity_room.get(entity_id, "unknown")
        if room == "unknown":
            self.stats["tokens_filtered"] += 1
            return None
        state = self._normalize_state(semantic, new_state)
        if not state:
            return None
        token = f"{room}.{semantic}.{state}"
        if self.entity_last_token.get(entity_id) == token:
            self.stats["tokens_filtered"] += 1
            return None
        self.entity_last_token[entity_id] = token
        if token not in self.token_to_id:
            if len(self.token_to_id) >= self.MAX_VOCAB_SIZE:
                cutoff = self._next_id - int(self.MAX_VOCAB_SIZE * 0.9)
                stale = [t for t, i in self.token_to_id.items() if i < cutoff]
                for t in stale:
                    old_id = self.token_to_id.pop(t)
                    self.id_to_token.pop(old_id, None)
                _LOGGER.debug("Vokabular beschnitten: %d Tokens entfernt", len(stale))
            self.token_to_id[token] = self._next_id
            self.id_to_token[self._next_id] = token
            self._next_id += 1
        token_id = self.token_to_id[token]
        self.stats["events_processed"] += 1
        if not timestamp:
            timestamp = datetime.now(timezone.utc)
        return {
            "token_id": token_id,
            "token": token,
            "entity_id": entity_id,
            "room": room,
            "semantic": semantic,
            "state": state,
            "timestamp": timestamp,
        }

    def _normalize_state(self, semantic: str, state: str) -> str:
        if not state or state in ("unknown", "unavailable"):
            return None
        state_lower = state.lower().strip()
        if semantic in ("light", "switch", "fan", "automation",
                        "motion", "door", "presence", "binary",
                        "gaming", "bed_presence", "screen", "wallbox"):
            if state_lower in ("on", "true", "open", "detected", "home"):
                return "on"
            if state_lower in ("off", "false", "closed", "clear", "not_home"):
                return "off"
            return state_lower
        if semantic == "media":
            if state_lower in ("playing", "on"):
                return "playing"
            if state_lower in ("paused",):
                return "paused"
            if state_lower in ("idle", "standby"):
                return "standby"
            if state_lower in ("off", "unavailable"):
                return "off"
            return state_lower
        if semantic == "climate":
            return state_lower
        if semantic == "cover":
            if state_lower in ("open", "opening"):
                return "open"
            if state_lower in ("closed", "closing"):
                return "closed"
            return state_lower
        if semantic == "tracker":
            if state_lower in ("home", "zu hause"):
                return "home"
            if state_lower in ("not_home", "away", "abwesend"):
                return "away"
            return state_lower
        if semantic in ("temperature", "humidity", "power", "energy",
                        "battery", "voltage", "illuminance", "pressure",
                        "solar", "grid", "co2", "cpu", "gpu", "tpms",
                        "heartrate", "steps", "network", "sleep"):
            return self._bucket_value(semantic, state)
        try:
            float(state_lower)
            return self._bucket_value("_generic", state_lower)
        except (ValueError, TypeError):
            pass
        return state_lower if len(state_lower) < 30 else None

    def _bucket_value(self, semantic: str, state: str) -> str:
        try:
            val = float(state)
        except (ValueError, TypeError):
            return state.lower() if state else None
        if semantic == "temperature":
            if val < 10: return "cold"
            elif val < 18: return "cool"
            elif val < 24: return "comfort"
            elif val < 30: return "warm"
            else: return "very_hot"
        elif semantic == "humidity":
            if val < 30: return "dry"
            elif val < 60: return "normal"
            else: return "humid"
        elif semantic == "power":
            if val < 100: return "low"
            elif val < 1000: return "medium"
            else: return "high"
        elif semantic == "energy":
            return "consumed"
        elif semantic == "battery":
            if val < 10: return "critical"
            elif val < 30: return "low"
            elif val < 80: return "medium"
            else: return "full"
        elif semantic == "voltage":
            if val < 210: return "low"
            elif val < 240: return "normal"
            else: return "high"
        elif semantic == "illuminance":
            if val < 10: return "dark"
            elif val < 200: return "dim"
            elif val < 1000: return "bright"
            else: return "very_bright"
        elif semantic == "solar":
            if val < 10: return "none"
            elif val < 500: return "low"
            elif val < 2000: return "medium"
            else: return "high"
        elif semantic == "grid":
            if val < 0: return "export"
            elif val < 100: return "minimal"
            else: return "import"
        elif semantic == "co2":
            if val < 600: return "good"
            elif val < 1000: return "elevated"
            else: return "high"
        elif semantic in ("cpu", "gpu"):
            low = self._threshold(semantic, "low", 35)
            medium = self._threshold(semantic, "medium", 70)
            if val < low: return "low"
            if val < medium: return "medium"
            return "high"
        elif semantic == "tpms":
            low = self._threshold("tpms", "low", 2.2)
            high = self._threshold("tpms", "high", 2.8)
            if val < low: return "low"
            if val > high: return "high"
            return "normal"
        elif semantic == "heartrate":
            low = self._threshold("heartrate", "low", 55)
            high = self._threshold("heartrate", "high", 110)
            if val < low: return "low"
            if val > high: return "high"
            return "normal"
        elif semantic == "steps":
            if val < self._threshold("steps", "low", 1000): return "low"
            if val < self._threshold("steps", "high", 7000): return "active"
            return "high"
        elif semantic == "network":
            if val < 1: return "idle"
            elif val < 10: return "low"
            elif val < 100: return "medium"
            else: return "high"
        elif semantic == "sleep":
            if val <= 0: return "awake"
            elif val < 30: return "light"
            elif val < 70: return "deep"
            else: return "rem"
        if val < 20: return "very_low"
        elif val < 40: return "low"
        elif val < 60: return "medium"
        elif val < 80: return "high"
        else: return "very_high"

    def _threshold(self, semantic: str, level: str, default: float) -> float:
        sem_cfg = self.custom_thresholds.get(semantic, {})
        try:
            return float(sem_cfg.get(level, default))
        except (TypeError, ValueError):
            return default

    def load_custom_profiles(self, path: str):
        p = Path(path)
        if not p.exists():
            _LOGGER.debug("Thalamus: Kein Custom-Profil (%s) – übersprungen", path)
            return
        try:
            import json
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as err:
            _LOGGER.warning("Thalamus: Profil konnte nicht geladen werden (%s): %s", path, err)
            return
        self.custom_semantic_rules = data.get("semantic_rules", []) or []
        self.custom_thresholds = data.get("thresholds", {}) or {}

    def decode_token(self, token_id: int) -> str:
        return self.id_to_token.get(token_id, f"?{token_id}")

    def resolve_entities(self, token: str) -> list:
        parts = token.split(".")
        if len(parts) != 3:
            return []
        room, semantic, state = parts
        return [
            eid for eid, sem in self.entity_semantic.items()
            if sem == semantic and self.entity_room.get(eid) == room
        ]

    def update_sun(self, elevation: float, is_daylight: bool):
        self._sun_elevation = elevation
        self._sun_is_daylight = is_daylight

    def encode_time_context(self, timestamp) -> list:
        if hasattr(timestamp, 'hour'):
            dt = timestamp
        else:
            try:
                dt = datetime.fromisoformat(str(timestamp))
            except (ValueError, TypeError):
                dt = datetime.now(timezone.utc)
        hour = dt.hour + dt.minute / 60.0
        dow = dt.weekday()
        month = dt.month
        sun_norm = max(0.0, min(1.0, (self._sun_elevation + 18) / 108))
        return [
            math.sin(2 * math.pi * hour / 24),
            math.cos(2 * math.pi * hour / 24),
            math.sin(2 * math.pi * dow / 7),
            math.cos(2 * math.pi * dow / 7),
            math.sin(2 * math.pi * month / 12),
            math.cos(2 * math.pi * month / 12),
            1.0 if dow >= 5 else 0.0,
            sun_norm,
            1.0 if self._sun_is_daylight else 0.0,
        ]

    def track_unassigned_event(self, entity_id: str):
        if entity_id in self._unassigned_entities:
            self._unassigned_event_counts[entity_id] = (
                self._unassigned_event_counts.get(entity_id, 0) + 1
            )

    SUGGEST_HINTS = {
        "schreibtisch": "office", "laptop": "office", "monitor": "office",
        "drucker": "office", "printer": "office", "scanner": "office",
        "fernseher": "livingroom", "couch": "livingroom", "sofa": "livingroom",
        "receiver": "livingroom", "soundbar": "livingroom", "beamer": "livingroom",
        "herd": "kitchen", "ofen": "kitchen", "spül": "kitchen",
        "mikrowelle": "kitchen", "kühlschrank": "kitchen", "fridge": "kitchen",
        "geschirrspüler": "kitchen", "dunstabzug": "kitchen", "kaffee": "kitchen",
        "coffee": "kitchen", "toaster": "kitchen",
        "dusch": "bathroom", "wanne": "bathroom", "spiegel": "bathroom",
        "waschbecken": "bathroom", "toilette": "bathroom", "wc_": "bathroom",
        "waschmaschine": "utility", "trockner": "utility", "dryer": "utility",
        "bügel": "utility",
        "matratze": "bedroom", "nachttisch": "bedroom", "nachtlicht": "bedroom",
        "wecker": "bedroom", "alarm_clock": "bedroom",
        "spielzeug": "kidsroom", "toy": "kidsroom",
        "rasen": "outdoor", "mäher": "outdoor", "bewässer": "outdoor",
        "sprinkler": "outdoor", "pool": "outdoor", "grill": "outdoor",
        "briefkasten": "outdoor", "mailbox": "outdoor",
        "treppen": "hallway", "stair": "hallway",
        "haustür": "entrance", "klingel": "entrance", "doorbell": "entrance",
    }

    def suggest_room(self, entity_id: str) -> str:
        info = self._unassigned_entities.get(entity_id, {})
        name = info.get("name", entity_id).lower()
        eid = entity_id.lower()
        import re as _re
        words = set(_re.split(r'[._\-\s]+', eid + " " + name))
        best_match = None
        best_len = 0
        all_hints = {**NAME_ROOM_HINTS, **HA_AREA_MAP, **self.SUGGEST_HINTS}
        for hint, room in all_hints.items():
            if hint in name or hint in eid:
                if len(hint) > best_len:
                    best_match = room
                    best_len = len(hint)
            elif hint in words:
                if len(hint) > best_len:
                    best_match = room
                    best_len = len(hint)
        return best_match

    def get_unassigned_report(self, top_n: int = 10) -> list:
        entries = []
        for eid, info in self._unassigned_entities.items():
            count = self._unassigned_event_counts.get(eid, 0)
            suggestion = self.suggest_room(eid)
            entries.append((
                eid, count, info.get("semantic", "?"),
                info.get("name", eid), suggestion,
            ))
        entries.sort(key=lambda x: -x[1])
        return entries[:top_n]

    def to_dict(self) -> dict:
        return {
            "entity_room": self.entity_room,
            "entity_semantic": self.entity_semantic,
            "token_to_id": self.token_to_id,
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()},
            "next_id": self._next_id,
            "entity_last_token": self.entity_last_token,
            "known_rooms": list(self._known_rooms),
            "unassigned_entities": self._unassigned_entities,
            "unassigned_event_counts": self._unassigned_event_counts,
            "custom_semantic_rules": self.custom_semantic_rules,
            "custom_thresholds": self.custom_thresholds,
        }

    def from_dict(self, data: dict):
        self.entity_room = data.get("entity_room", {})
        self.entity_semantic = data.get("entity_semantic", {})
        self.token_to_id = data.get("token_to_id", {})
        self.id_to_token = {int(k): v for k, v in data.get("id_to_token", {}).items()}
        self._next_id = data.get("next_id", 1)
        self.entity_last_token = data.get("entity_last_token", {})
        self._known_rooms = set(data.get("known_rooms", []))
        self._unassigned_entities = data.get("unassigned_entities", {})
        self._unassigned_event_counts = data.get("unassigned_event_counts", {})
        self.custom_semantic_rules = data.get("custom_semantic_rules", [])
        self.custom_thresholds = data.get("custom_thresholds", {})
        self.stats["entities_registered"] = len(self.entity_semantic)
        self.stats["rooms_discovered"] = len(self._known_rooms)
