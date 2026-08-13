"""
Neurorhythms – Biologische Rhythmen & Dopamin-Dynamik.

Kein neues Gehirnareal, sondern das TIMING-System das alles moduliert.

1. Circadiane Modulation:
   Das Gehirn lernt morgens (hohe Plastizität) anders als nachts
   (niedrige Plastizität, Konsolidierung). Alle Lernraten werden
   durch einen Tagesrhythmus moduliert.

2. Dopamin-Bursts (Phasisches Dopamin):
   Normales Dopamin = langsames Q-Learning (tonisch).
   Dopamin-BURST = Überraschung + positives Outcome gleichzeitig.
   Ein Burst verstärkt genau dieses Muster überproportional stark.
   Biologisch: "Das war unerwartet GUT → sofort merken!"

3. Synaptic Homeostasis (SHY):
   Während aktiver Phasen werden Synapsen stärker (Lernen).
   Nachts/In Ruhe werden ALLE Gewichte proportional herunterskaliert.
   Verhindert Sättigung, erhält Signal-Rausch-Verhältnis.

Performance: Nur Arithmetik + Zeitabfragen. ~0 ms pro Event.
"""

import math
import time
import logging
from collections import deque

_LOGGER = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# Circadiane Modulation
# ═══════════════════════════════════════════════════════════════

# Lernrate-Multiplikator über 24h (biologisch inspiriert)
# Morgens (6-10): Hohe Plastizität (1.3x)
# Mittags (11-14): Moderate (1.0x)
# Nachmittags (15-18): Leicht erhöht (1.1x) – "zweites Hoch"
# Abends (19-22): Abnehmend (0.8x)
# Nachts (23-5): Niedrig (0.5x) – Konsolidierungsphase

def _circadian_base(hour: int) -> float:
    """Basis-Circadian-Kurve als Cosinus mit Peak um 8 Uhr."""
    # Peak bei 8:00, Tief bei 2:00
    phase = (hour - 8) / 24.0 * 2 * math.pi
    return 0.5 + 0.5 * math.cos(phase)  # Range 0.0 - 1.0


# ═══════════════════════════════════════════════════════════════
# Dopamin-Burst Detektion
# ═══════════════════════════════════════════════════════════════

# Ein Burst tritt auf wenn:
# 1. Surprise > BURST_SURPRISE_THRESHOLD (unerwartet)
# 2. UND das Outcome positiv war (kein Override innerhalb von 60s)
# Burst-Stärke: surprise × positive_outcome → Multiplikator 3-8x
BURST_SURPRISE_THRESHOLD = 0.5
BURST_MIN_MULTIPLIER = 3.0
BURST_MAX_MULTIPLIER = 8.0
BURST_COOLDOWN = 30.0  # Sekunden zwischen Bursts (Refractory Period)


class Neurorhythms:
    """Moduliert Lernraten durch biologische Rhythmen und Dopamin-Bursts."""

    def __init__(self):
        # ── Circadian ──
        self.circadian_multiplier = 1.0
        self._last_circadian_update = 0

        # ── Dopamin-Bursts ──
        self.dopamine_level = 0.0         # Tonisches Level (0.0-1.0)
        self.burst_active = False          # Ist gerade ein Burst aktiv?
        self.burst_multiplier = 1.0        # Aktueller Burst-Multiplikator
        self.last_burst_ts = 0.0
        self._pending_surprise = 0.0       # Surprise des letzten Events (wartet auf Outcome)
        self._pending_token = None         # Token des letzten Events
        self._pending_ts = 0.0

        # Burst-Historie (für Dashboard-Visualisierung)
        self.burst_history = deque(maxlen=50)
        self.total_bursts = 0
        self.strongest_burst = 0.0

        # ── Synaptic Homeostasis ──
        self.total_synaptic_load = 0.0     # Kumulierte "Synapsenstärke" seit letztem Reset
        self.homeostasis_factor = 1.0      # Skalierungsfaktor (wird nachts angewendet)
        self._events_since_homeostasis = 0

    # ══════════════════════════════════════════════════════════
    # Circadian
    # ══════════════════════════════════════════════════════════

    def get_circadian_multiplier(self, hour: int = None) -> float:
        """
        Gibt den aktuellen circadianen Lernraten-Multiplikator zurück.
        Range: 0.5 (Nacht) bis 1.3 (Morgen-Peak).
        """
        now = time.time()
        # Nur alle 60s neu berechnen
        if now - self._last_circadian_update > 60 or hour is not None:
            if hour is None:
                import datetime as dt
                hour = dt.datetime.now().hour

            base = _circadian_base(hour)
            # Skalierung: 0.0 → 0.5x, 0.5 → 1.0x, 1.0 → 1.3x
            self.circadian_multiplier = 0.5 + base * 0.8
            self._last_circadian_update = now

        return self.circadian_multiplier

    # ══════════════════════════════════════════════════════════
    # Dopamin-Bursts
    # ══════════════════════════════════════════════════════════

    def register_surprise(self, token_id: int, surprise: float):
        """
        Registriert ein überraschendes Event. Wenn innerhalb von 60s
        ein positives Outcome kommt → Dopamin-Burst.
        """
        if surprise > BURST_SURPRISE_THRESHOLD:
            self._pending_surprise = surprise
            self._pending_token = token_id
            self._pending_ts = time.time()

        # Tonisches Dopamin anpassen (langsam, EMA)
        self.dopamine_level = 0.95 * self.dopamine_level + 0.05 * surprise

    def register_outcome(self, token_id: int, positive: bool) -> float:
        """
        Registriert das Outcome einer Aktion/Vorhersage.
        Returns: Burst-Multiplikator (1.0 = kein Burst, 3-8x = Burst!)
        """
        now = time.time()
        self.burst_active = False
        self.burst_multiplier = 1.0

        # Prüfe ob wir ein pending surprise haben das zum Outcome passt
        if (positive
                and self._pending_surprise > BURST_SURPRISE_THRESHOLD
                and now - self._pending_ts < 60.0
                and now - self.last_burst_ts > BURST_COOLDOWN):

            # DOPAMIN-BURST! Unerwartet + positiv = maximales Lernen
            burst_strength = self._pending_surprise  # 0.5 - 1.0
            self.burst_multiplier = (
                BURST_MIN_MULTIPLIER
                + (burst_strength - BURST_SURPRISE_THRESHOLD)
                / (1.0 - BURST_SURPRISE_THRESHOLD)
                * (BURST_MAX_MULTIPLIER - BURST_MIN_MULTIPLIER)
            )
            self.burst_multiplier = min(BURST_MAX_MULTIPLIER, self.burst_multiplier)

            self.burst_active = True
            self.last_burst_ts = now
            self.total_bursts += 1
            self.strongest_burst = max(self.strongest_burst, self.burst_multiplier)

            self.burst_history.append({
                "ts": now,
                "token": self._pending_token,
                "surprise": self._pending_surprise,
                "multiplier": round(self.burst_multiplier, 1),
            })

            _LOGGER.info(
                "DOPAMIN-BURST! surprise=%.2f, multiplier=%.1fx, token=%s",
                self._pending_surprise, self.burst_multiplier, self._pending_token,
            )

            # Tonisches Level boosten
            self.dopamine_level = min(1.0, self.dopamine_level + 0.3)

        elif not positive and self._pending_surprise > BURST_SURPRISE_THRESHOLD:
            # Unerwartet + negativ = Dopamin-Dip (lernt auch, aber anders)
            self.dopamine_level = max(0.0, self.dopamine_level - 0.15)

        # Pending zurücksetzen
        self._pending_surprise = 0.0
        self._pending_token = None

        return self.burst_multiplier

    # ══════════════════════════════════════════════════════════
    # Synaptic Homeostasis
    # ══════════════════════════════════════════════════════════

    def track_learning(self, learn_weight: float):
        """Trackt kumulierte Lernstärke für Homeostasis."""
        self.total_synaptic_load += learn_weight
        self._events_since_homeostasis += 1

    def compute_homeostasis_factor(self) -> float:
        """
        Berechnet den Homeostasis-Skalierungsfaktor.
        Nach vielen Events mit hohem Lernen → stärkere Herunterskalierung.
        Wird während Sleep Consolidation angewendet.
        """
        if self._events_since_homeostasis < 50:
            return 1.0  # Nicht genug Daten

        avg_load = self.total_synaptic_load / self._events_since_homeostasis
        # Wenn durchschnittliche Last > 1.0 → System hat viel gelernt → herunterskalieren
        # Factor: 0.85 (starke Skalierung) bis 0.98 (kaum Skalierung)
        if avg_load > 1.2:
            self.homeostasis_factor = 0.85
        elif avg_load > 0.8:
            self.homeostasis_factor = 0.92
        else:
            self.homeostasis_factor = 0.98

        return self.homeostasis_factor

    def reset_homeostasis(self):
        """Nach Anwendung der Homeostasis (z.B. in Sleep Consolidation)."""
        self.total_synaptic_load = 0.0
        self._events_since_homeostasis = 0
        self.homeostasis_factor = 1.0

    # ══════════════════════════════════════════════════════════
    # Kombiniertes Lerngewicht
    # ══════════════════════════════════════════════════════════

    def modulate_learning(self, base_weight: float, hour: int = None) -> float:
        """
        Kombiniert alle Rhythmen zu einem finalen Lerngewicht.

        base_weight: Aus Predictive Processing (Surprise-basiert)
        Returns: Finales Lerngewicht (alle Modulationen kombiniert)
        """
        circadian = self.get_circadian_multiplier(hour)
        burst = self.burst_multiplier if self.burst_active else 1.0

        # Kombination: base × circadian × burst
        final = base_weight * circadian * burst
        final = max(0.05, min(10.0, final))  # Clamp

        # Homeostasis tracken
        self.track_learning(final)

        # Burst nach Verwendung zurücksetzen
        if self.burst_active:
            self.burst_active = False
            self.burst_multiplier = 1.0

        return final

    # ══════════════════════════════════════════════════════════
    # Persistenz
    # ══════════════════════════════════════════════════════════

    @property
    def stats(self) -> dict:
        return {
            "circadian_multiplier": round(self.circadian_multiplier, 2),
            "dopamine_level": round(self.dopamine_level, 3),
            "total_bursts": self.total_bursts,
            "strongest_burst": round(self.strongest_burst, 1),
            "homeostasis_factor": round(self.homeostasis_factor, 2),
            "synaptic_load": round(self.total_synaptic_load, 1),
            "burst_active": self.burst_active,
        }

    def to_dict(self) -> dict:
        return {
            "dopamine_level": self.dopamine_level,
            "total_bursts": self.total_bursts,
            "strongest_burst": self.strongest_burst,
            "total_synaptic_load": self.total_synaptic_load,
            "events_since_homeostasis": self._events_since_homeostasis,
        }

    def from_dict(self, data: dict):
        self.dopamine_level = data.get("dopamine_level", 0.0)
        self.total_bursts = data.get("total_bursts", 0)
        self.strongest_burst = data.get("strongest_burst", 0.0)
        self.total_synaptic_load = data.get("total_synaptic_load", 0.0)
        self._events_since_homeostasis = data.get("events_since_homeostasis", 0)
