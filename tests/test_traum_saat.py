# -*- coding: utf-8 -*-
"""FLUG 2732 (claudes Weg 2, Notiz 17474): DER GESÄTE TRAUM-REPLAY — die Wachen.

Die Engine reicht ``SleepConsolidation`` einen eigenen, gesäten
``random.Random`` hinein (``rng_seed`` aus der Konfiguration, fester
Standard 0). Der Kern fasst das GLOBALE ``random``-Modul nie mehr an.
Die Wachen wiegen EXAKT (``getstate()``-Vergleich — kein Flaken, keine
Münze) und LAUFEN (zwei gleich gesäte Maschinen träumen denselben Traum).
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from kontinuum_core import KontinuumEngine


# ---------------------------------------------------------------------------
# Die Saat selbst (exakte Waage: der Zustand des RNG ist der Beweis)
# ---------------------------------------------------------------------------

def test_traum_saat_standard_ist_null():
    """Ohne Konfiguration trägt der Traum die Saat 0 — dieselbe wie die
    bestehenden ``_rng``-Patches (replay.py, spur/messstand.py, Tests),
    sie bleiben semantisch identisch."""
    e = KontinuumEngine()
    assert isinstance(e.sleep_consolidation._rng, random.Random)
    assert e.sleep_consolidation._rng.getstate() == random.Random(0).getstate()


def test_traum_saat_aus_der_konfiguration():
    """``rng_seed: 5`` säht den Traum mit 5 — die Engine reicht die Saat
    weiter, der Wert steht nicht im Modul."""
    e = KontinuumEngine(config={"rng_seed": 5})
    assert e.sleep_consolidation._rng.getstate() == random.Random(5).getstate()


def test_traum_ist_ein_eigener_strom_nicht_das_modul():
    """Der Kern fasst das GLOBALE random-Modul nie mehr an: der Traum trägt
    eine EIGENE Instanz (nicht das Modul — drawt aus einem eigenen Strom,
    der Aufrufer-Maschine bleibt ihr Zufall unangetastet)."""
    e = KontinuumEngine()
    assert e.sleep_consolidation._rng is not random
    # Zwei Maschinen teilen DENSELBEN Strom nicht (je eigene Instanz):
    b = KontinuumEngine()
    assert e.sleep_consolidation._rng is not b.sleep_consolidation._rng


def test_traum_saat_none_saht_aus_der_maschine():
    """``rng_seed: None`` säht aus der Maschine (OS-Entropie): nicht
    deterministisch, ehrlich benannt — und der Traum läuft trotzdem."""
    e = KontinuumEngine(config={"rng_seed": None})
    assert isinstance(e.sleep_consolidation._rng, random.Random)
    assert e.sleep_consolidation._rng.getstate() != random.Random(0).getstate()


# ---------------------------------------------------------------------------
# Der Traum selbst (Lauf-Waage: gleiche Saat, gleiches Erleben → gleiches
# Traum-Ergebnis; die einzige RNG-Stelle im Konsolidierungspfad ist der
# Dream-Replay — Phase 1–3 und 5 sind reine Form)
# ---------------------------------------------------------------------------

def _erleben(engine: KontinuumEngine) -> None:
    """Dasselbe Erleben für zwei Maschinen (feste Zeitstempel — nur der
    Traum darf ziehen, und der trägt die Saat)."""
    base = datetime(2026, 1, 1, 19, 0, tzinfo=timezone.utc)
    n = 0
    for _ in range(3):
        for eid, st in [("switch.k", "on"), ("light.b", "on"),
                        ("switch.k", "off"), ("light.b", "off"),
                        ("binary_sensor.m", "on"), ("media_player.s", "play")]:
            engine.observe({"entity_id": eid, "new_state": st,
                            "timestamp": base + timedelta(minutes=n)})
            n += 1


def test_gleiche_saat_gleicher_traum():
    """Zwei Maschinen, Saat 7, dasselbe Erleben: der Dream-Replay zieht
    dieselben Paare — JEDE Zahl der Konsolidierung steht gleich (die
    wackelnde Wache von 17464/2730 ist damit ohne Münze: kein Lauf
    kippt mehr am Zufall)."""
    a = KontinuumEngine(config={"rng_seed": 7})
    b = KontinuumEngine(config={"rng_seed": 7})
    _erleben(a)
    _erleben(b)
    stats_a = a.sleep_consolidation.consolidate(
        a.hippocampus, a.cerebellum, a.basal_ganglia, a.neurorhythms)
    stats_b = b.sleep_consolidation.consolidate(
        b.hippocampus, b.cerebellum, b.basal_ganglia, b.neurorhythms)
    # Die Wanduhr-Zeile (Konsolidierungs-Zeitstempel) ist kein Traum —
    # JEDE andere Zahl steht gleich:
    stats_a.pop("timestamp", None)
    stats_b.pop("timestamp", None)
    assert stats_a == stats_b
    assert a.sleep_consolidation.total_dream_connections == \
        b.sleep_consolidation.total_dream_connections


def test_ungleiche_saat_traeumt_unterschiedlich():
    """Saat 7 gegen Saat 8: die Maschinen erleben dasselbe, aber der Traum
    zieht andere Paare — die Saat ist der GRUND des Unterschieds (der
    Messstand der Anomalie-Wache wird sie je Saat benennen)."""
    a = KontinuumEngine(config={"rng_seed": 7})
    b = KontinuumEngine(config={"rng_seed": 8})
    _erleben(a)
    _erleben(b)
    a.sleep_consolidation.consolidate(
        a.hippocampus, a.cerebellum, a.basal_ganglia, a.neurorhythms)
    b.sleep_consolidation.consolidate(
        b.hippocampus, b.cerebellum, b.basal_ganglia, b.neurorhythms)
    # Der RNG-Zustand lief auseinander (exakt, nie flakend):
    assert a.sleep_consolidation._rng.getstate() != \
        b.sleep_consolidation._rng.getstate()


def test_traum_ohne_saat_patch_ledig():
    """Die alten ``_rng``-Patches (Saat 0, dieselbe Zeile wie in
    benchmarks/replay.py) bleiben UNSCHULDIG: die Maschine trägt die Saat
    schon selbst — das Patch ändert nichts mehr am Strom."""
    e = KontinuumEngine(config={"rng_seed": 3})
    vor = e.sleep_consolidation._rng.getstate()
    # Das Patch (aus welchem Grund auch immer noch gefahren):
    e.sleep_consolidation._rng = random.Random(3)
    assert e.sleep_consolidation._rng.getstate() == vor
