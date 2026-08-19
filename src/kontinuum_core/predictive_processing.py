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
from statistics import median

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

# Adaptiver Anomalie-Schwellwert (robust gegen Kontamination):
#   Schwelle = Median + ANOMALY_MAD_FACTOR × (MAD_TO_STD × MAD) der letzten
#   Surprise-Werte -- gerechnet auf der um Auffaellige BEREINIGTEN Historie,
#   nach unten auf Median x ANOMALY_FLOOR_FACTOR und nach oben auf
#   ANOMALY_MAX_THRESHOLD begrenzt. Naeheres in anomaly_threshold().
# Warum Median/MAD statt Mittelwert/Std: Anomalien sind selten und LAUT. In
# einem Mittelwert+Std-Schätzer heben genau die Ausreißer, die wir flaggen
# wollen, die Schwelle gegen sich selbst an → der Recall bricht ein. Median und
# MAD (Median der absoluten Abweichungen) ignorieren die seltenen Spitzen und
# schätzen das *normale* Surprise-Niveau des Zuhauses; MAD_TO_STD macht aus der
# MAD einen konsistenten Std-Schätzer für normalverteilte Daten.
# 3.0 stammte aus der Zeit, als Median und MAD noch aus der ganzen
# Historie kamen -- also inklusive der Anomalien, die die Schwelle gegen
# sich selbst anhoben. Mit der bereinigten Historie (siehe
# anomaly_threshold) liegt die Schwelle tiefer und richtiger, und 3.0
# waere zu streng. Gemessen auf beiden Maschinen:
#
#   k     Recall  Precision  Grenzfaelle
#   1.8   1.0000     1.0000            0
#   2.0   1.0000     1.0000            0
#   2.2   1.0000     1.0000            0   <- Mitte des Plateaus
#   2.5   1.0000     1.0000            0
#   2.8   0.9861     1.0000           22
#   3.0   0.9028     1.0000           25
#
# 2.2 liegt in der Mitte des flachen Bereichs und nicht an seinem Rand.
# Ein Wert am Rand ist wieder nur ein gluecklicher Wuerfelwurf -- die
# Lehre aus 0.55, 0.10 und 0.07.
ANOMALY_MAD_FACTOR = 2.2
MAD_TO_STD = 1.4826
# Untergrenze — RELATIV, nicht absolut.
#
# Die Geschichte dieser Zeile ist eine Kette fester Zahlen: 0.55 (der Flag
# feuerte praktisch nie), dann 0.10 (Issue #1: der Boden lag mitten in der
# Anomalie-Wolke), dann 0.07. Jede war auf genau einem Rechner gemessen.
#
# Denn die Beträge wandern. Derselbe Code, dieselben Seeds, dieselbe AUC
# auf sechs Stellen (0.9996) — und trotzdem:
#
#                        Windows/3.14   Linux/3.13
#   Median normal            0.0814       0.0575
#   MAD normal               0.0110       0.0107
#   Median Anomalie          0.1372       0.1301
#   Abstand unterste Anom.   1.39 MAD     2.48 MAD
#
# Die Streuung ist gleich, die Anomalien liegen fast gleich — aber die
# NORMALEN Ereignisse liegen auf Windows höher. Der Abstand zwischen
# normal und auffällig ist dort nur halb so groß. Eine feste Zahl schneidet
# deshalb an verschiedenen Stellen durch dieselbe Rangliste.
#
# Der Boden ist dabei NICHT der Hebel, auch wenn drei Anläufe an ihm
# gedreht haben. Gemessen mit der bereinigten Schwelle (k = 2.2):
#
#   Boden              Recall  Precision  Grenzfaelle
#   nur 0.02           1.0000     1.0000            0
#   fest 0.07          1.0000     1.0000            0
#   Median × 1.2       1.0000     1.0000            0
#   Median × 1.5       1.0000     1.0000           13
#
# Bis × 1.2 greift er in diesem Zuhause gar nicht -- er ist reine
# Vorsorge fuer ein sehr starres. Ab × 1.5 wird er bindend und landet in
# der Anomalie-Wolke: dieselbe Falle wie 0.10 damals, nur relativ
# formuliert. Deshalb × 1.2 und keine "sichere" Reserve obendrauf.
#
# Ein Perzentil wäre der naheliegende Griff und ist der falsche: Es setzt
# eine feste Anomalierate voraus. Im Benchmark sind 21 % der Ereignisse
# auffällig, in einem echten Zuhause weit unter 1 % — dasselbe Perzentil
# liegt einmal mitten in der Wolke und einmal mitten im Normalen.
ANOMALY_FLOOR_FACTOR = 1.2

# Reiner Schutz gegen Null, keine Kalibrierung: In einem vollkommen
# starren Zuhause wäre der Median 0 und damit auch der relative Boden —
# dann wäre jedes Ereignis eine Anomalie. Diese Zahl darf niemandem als
# Stellschraube dienen; wer hier dreht, dreht am falschen Ende.
ANOMALY_ABSOLUTE_FLOOR = 0.02
ANOMALY_MAX_THRESHOLD = 0.95
ANOMALY_DEFAULT_THRESHOLD = 0.7
ANOMALY_MIN_SAMPLES = 30

# Kalibrierung des Vorhersage-Fehlers bei nicht vorhergesagten Tokens:
# Ein Miss zählt nur dann als volle Überraschung, wenn das Modell selbst
# zuversichtlich war. Im Cold Start (keine/unsichere Vorhersagen) ist
# "nicht vorhergesagt" der Normalfall und kein starkes Signal.
UNPREDICTED_BASE_SURPRISE = 0.6


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
        # Kalibriert nach Modell-Zuversicht: Ein Miss bei einem unsicheren
        # Modell (Cold Start) ist weniger überraschend als ein Miss bei
        # einem Modell, das eine hochkonfidente Vorhersage hatte.
        model_conf = 0.0
        for pred in predictions or []:
            if len(pred) > 2 and pred[2] > model_conf:
                model_conf = pred[2]
        prediction_surprise = (
            UNPREDICTED_BASE_SURPRISE
            + (1.0 - UNPREDICTED_BASE_SURPRISE) * model_conf
        )

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

    def anomaly_threshold(self) -> float:
        """Robust-adaptiver Anomalie-Schwellwert.

        Die Schwelle wird aus dem eigenen Surprise-Niveau des Zuhauses
        abgeleitet: ``Median + k × (MAD_TO_STD × MAD)`` der letzten
        Surprise-Werte. Median und MAD sind robust gegen die seltenen,
        lauten Anomalien – anders als Mittelwert+Std, den genau diese
        Ausreißer nach oben verzerren (sie höben die Schwelle gegen sich
        selbst an). Ein lautes/chaotisches Zuhause hebt die Schwelle, ein
        sehr vorhersagbares senkt sie bis auf die Untergrenze. Vor
        ANOMALY_MIN_SAMPLES Beobachtungen gilt der Default.

        Die Untergrenze ist **relativ**: ``Median x ANOMALY_FLOOR_FACTOR``.
        Eine feste Zahl dort war dreimal in Folge auf genau einem Rechner
        richtig -- die Surprise-Betraege wandern zwischen Plattformen,
        die Rangfolge nicht. Naeheres oben bei der Konstante.
        """
        n = len(self.surprise_history)
        if n < ANOMALY_MIN_SAMPLES:
            return ANOMALY_DEFAULT_THRESHOLD
        data = list(self.surprise_history)

        # Zwei Durchgänge. Median und MAD sind robust gegen einzelne
        # Ausreißer, aber nicht gegen 20 % davon: In der Historie stehen
        # auch die Anomalien, und sie heben die Schwelle gegen sich
        # selbst an. Genau das war die Ursache des Plattform-Unterschieds
        # — wie viele Auffällige gerade im Fenster stehen, entscheidet
        # über die Schwelle, und das schwankt.
        #
        # Der erste Durchgang liefert eine grobe Trennung, der zweite
        # rechnet nur noch mit dem, was darunter liegt: dem *normalen*
        # Niveau dieses Zuhauses.
        med = median(data)
        mad = median([abs(x - med) for x in data])
        grob = med + ANOMALY_MAD_FACTOR * MAD_TO_STD * mad

        # `or data`: Liegt nichts unter der groben Schwelle, ist das
        # Fenster zu klein oder alles gleich laut. Dann mit allem
        # weiterrechnen statt mit einer leeren Liste — median([]) wirft,
        # und ein Absturz an dieser Stelle sähe aus wie ein Datenfehler.
        ruhig = [x for x in data if x < grob] or data
        med = median(ruhig)
        mad = median([abs(x - med) for x in ruhig])
        threshold = med + ANOMALY_MAD_FACTOR * MAD_TO_STD * mad
        # Der Boden wandert mit dem Niveau des Zuhauses mit. Eine feste
        # Zahl hier hat dreimal in Folge auf genau einem Rechner gepasst.
        floor = max(ANOMALY_ABSOLUTE_FLOOR, med * ANOMALY_FLOOR_FACTOR)
        # Die Obergrenze zuletzt, und ueber ALLES. Beim ersten Versuch
        # stand sie nur um den adaptiven Teil -- in einem chaotischen
        # Zuhause (Median 0.72) hob der Boden die Schwelle auf 1.08, ueber
        # den Wertebereich hinaus. Dort ist kein Ereignis mehr auffaellig,
        # und der Flag waere still ausgegangen.
        return min(ANOMALY_MAX_THRESHOLD, max(floor, threshold))

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
            "anomaly_threshold": round(self.anomaly_threshold(), 3),
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
            # Ohne die Surprise-Historie würde anomaly_threshold() nach jedem
            # Neustart für ANOMALY_MIN_SAMPLES Events auf den Default (0.7)
            # zurückfallen – die adaptive Schwelle (und average_surprise)
            # vergäße das gelernte Surprise-Niveau des Zuhauses.
            "surprise_history": list(self.surprise_history),
        }

    def from_dict(self, data: dict):
        self.baseline_surprise = data.get("baseline_surprise", 0.3)
        self.total_events = data.get("total_events", 0)
        self.total_surprises = data.get("total_surprises", 0)
        self.total_expected = data.get("total_expected", 0)
        self.max_surprise = data.get("max_surprise", 0.0)
        # JSON-Roundtrip macht aus int-Keys Strings → zurückkonvertieren,
        # sonst greift die Novelty-Erkennung nach einem Neustart ins Leere.
        self._token_familiarity = {
            int(k): v for k, v in data.get("token_familiarity", {}).items()
        }
        # Adaptive Anomalie-Schwelle über den Neustart hinweg erhalten.
        self.surprise_history = deque(
            (float(s) for s in data.get("surprise_history", [])), maxlen=100
        )
