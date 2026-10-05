"""Die explizite Uhr (Abnahme 13642, Notiz 13198): Ereigniszeit statt Wanduhr.

Vorher lasen Locus Coeruleus und die Hypothalamus-Cooldowns die Wanduhr:
Im Replay (alles laeuft in Sekunden ab) saettigte das Arousal auf 1,0 und
das Burst-Gate filterte strenger als live. Jetzt gilt: hat ein Ereignis
eine Zeit, rechnet die Engine damit; ohne Zeitstempel greift die
konfigurierbare Uhr (Standard: Wanduhr — Live-Betrieb unveraendert).

Der wichtigste Test ist der von Claude verlangte Satz: „Replay = gleiche
Arousal-Kurve wie bei echter Zeit" — zwei Engines, dieselben
Ereigniszeiten, einmal als Zeitstempel, einmal ueber die Uhr.
"""
from __future__ import annotations

import random
import time
from datetime import datetime, timezone

from kontinuum_core import KontinuumEngine
from kontinuum_core.hypothalamus import Hypothalamus
from kontinuum_core.locus_coeruleus import LocusCoeruleus


class FakeClock:
    """Eine Uhr, die man stellt — der Messstand-Fall."""

    def __init__(self, start: float = 0.0):
        self.t = start

    def set(self, t: float) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


# ---------------------------------------------------------------------------
# Die Module direkt
# ---------------------------------------------------------------------------


def test_locus_coeruleus_rechnet_mit_ereigniszeit():
    """100 Ereignisse im 6-Sekunden-Takt (10 Minuten Strom): Wer die
    Ereigniszeit liest, sieht ~10 Ereignisse im 60-s-Fenster
    (Arousal ~0,42). Wer die Wanduhr liest, sieht alle 100 in
    Millisekunden (Arousal ~1,0)."""
    zeiten = [6.0 * i for i in range(100)]

    ereigniszeit = LocusCoeruleus()
    for t in zeiten:
        ereigniszeit.observe_event(t)
    ruhig = ereigniszeit.get_arousal()
    assert 0.35 < ruhig < 0.6, ruhig

    wanduhr = LocusCoeruleus()
    for _ in zeiten:
        wanduhr.observe_event()  # kein now -> Uhr (Wanduhr)
    assert wanduhr.get_arousal() > 0.9, wanduhr.get_arousal()


def test_hypothalamus_cooldown_rechnet_mit_ereigniszeit():
    """Achtung: absorb() bekommt die GEBUCKETTEN Zustaende (so reicht
    die Engine sie herein) — nicht die Rohwerte."""
    h = Hypothalamus()
    t0 = 1_000_000.0
    # 1. Batterie-Wechsel setzt den Zeitstempel der letzten Energie-Sache.
    assert h.absorb("utility", "battery", "medium", "sensor.b", now=t0) is not None
    assert h._last_energy_event_time == t0
    # 2. Zehn Sekunden spaeter: im Cooldown (60 s) -> unterdrueckt,
    #    der Zeitstempel bleibt stehen.
    assert h.absorb("utility", "battery", "full", "sensor.b", now=t0 + 10) is None
    assert h.last_energy_state == (3, 0)
    assert h._last_energy_event_time == t0
    # 3. Nach 61 s: ausserhalb -> Uebergang mit neuem Zeitstempel.
    assert h.absorb("utility", "battery", "low", "sensor.b", now=t0 + 61) is not None
    assert h._last_energy_event_time == t0 + 61


# ---------------------------------------------------------------------------
# Die Engine
# ---------------------------------------------------------------------------


def _engine(clock=None) -> KontinuumEngine:
    engine = KontinuumEngine(config={"clock": clock} if clock else None)
    # Determinismus (dieselbe Zeile wie in benchmarks/replay.py).
    engine.sleep_consolidation._rng = random.Random(0)
    engine.register_entity("binary_sensor.motion_wohnzimmer",
                           ha_area="wohnzimmer", domain="binary_sensor")
    return engine


def _ereignis(i: int, t: float, mit_zeitstempel: bool) -> dict:
    daten = {
        "entity_id": "binary_sensor.motion_wohnzimmer",
        "new_state": "on" if i % 2 == 0 else "off",
    }
    if mit_zeitstempel:
        daten["timestamp"] = datetime.fromtimestamp(t, tz=timezone.utc)
    return daten


def test_replay_gleiche_arousal_kurve_wie_echte_zeit():
    """Claudes Auflage, woertlich: dieselben Ereigniszeiten, einmal als
    Zeitstempel (live), einmal ueber die Uhr (Replay ohne Zeitstempel) —
    die Arousal-Kurve muss ZUGLEICH sein."""
    zeiten = [1_000_000.0 + 6.0 * i for i in range(30)]
    live = _engine()
    uhr = FakeClock(zeiten[0])
    replay = _engine(uhr)
    for i, t in enumerate(zeiten):
        schnapp_live = live.observe(_ereignis(i, t, True))
        # Die Uhr MUSS mitlaufen — sonst stehen alle Replay-Ereignisse
        # auf derselben Zeit und das Burst-Gate filtert sie (genau der
        # Mechanismus, den die Ereigniszeit heilt; beim ersten Testlauf
        # ist mir das selbst passiert).
        uhr.set(t)
        schnapp_replay = replay.observe(_ereignis(i, t, False))
        assert schnapp_live.extra["arousal"] == schnapp_replay.extra["arousal"], i
        # Nicht nur das Arousal: die ganze Ereignisantwort ist gleich —
        # Surprise, Lernzustand, Entscheidung.
        assert schnapp_live.surprise == schnapp_replay.surprise, i
        assert schnapp_live.learning_state == schnapp_replay.learning_state, i
        assert schnapp_live.token == schnapp_replay.token, i
    assert live.locus_coeruleus.get_arousal() == \
        replay.locus_coeruleus.get_arousal()


def test_ohne_zeitstempel_und_ohne_uhr_bleibt_die_wanduhr():
    """Live-Verhalten unveraendert: kein Zeitstempel, keine konfigurierte
    Uhr -> die Module bekommen die WANDUHR (hier direkt bezeugt am
    Zeitstempel, den der Locus Coeruleus ablegt)."""
    engine = _engine()
    vorher = time.time()
    engine.observe(_ereignis(0, 0.0, False))
    nachher = time.time()
    gestempelt = engine.locus_coeruleus.events[-1]
    assert vorher - 1.0 <= gestempelt <= nachher + 1.0, gestempelt


def test_engine_gibt_ereigniszeit_an_den_hypothalamus_weiter():
    """Ein Batterie-Ereignis mit Zeitstempel: der Cooldown-Zeitstempel
    des Hypothalamus traegt die EREIGNISZEIT, nicht die Wanduhr."""
    engine = _engine()
    engine.register_entity("sensor.batterie", ha_area="utility",
                           domain="sensor", device_class="battery")
    t = 1_700_000_000.0
    engine.observe({
        "entity_id": "sensor.batterie",
        "new_state": "55",
        "timestamp": datetime.fromtimestamp(t, tz=timezone.utc),
    })
    assert engine.hypothalamus._last_energy_event_time == t
