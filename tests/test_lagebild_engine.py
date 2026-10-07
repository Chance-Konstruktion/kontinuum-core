"""Stufe 3 in der Engine: Lagebild vor dem Thalamus, Börse statt Ranking,
Speicherstand bitgleich, alte Gehirne laden weiter."""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone

from benchmarks.spur.messstand import registriere_engine
from benchmarks.spur.simulator import simuliere
from benchmarks.spur.zeitgeber import Zeitgeber
from kontinuum_core import KontinuumEngine
from kontinuum_core.association_cortex import WEG

START = datetime(2026, 1, 5, 18, 0, tzinfo=timezone.utc)


def _ereignis(ev) -> dict:
    return {"entity_id": ev.entity, "new_state": ev.zustand,
            "old_state": ev.alt, "timestamp": ev.ts}


def test_lagebild_sieht_was_der_thalamus_verwirft():
    """Reifendrucksensoren, die mit dem Auto wegfahren, melden
    ``unavailable``. Der Thalamus verwirft das Ereignis, das Lagebild führt
    den Sensor als ``weg`` — und die Engine verliert dabei nichts."""
    e = KontinuumEngine()
    e.register_entity("sensor.reifen_vl", ha_area="garage", domain="sensor")
    e.observe({"entity_id": "sensor.reifen_vl", "new_state": "2.4", "timestamp": START})
    snap = e.observe({"entity_id": "sensor.reifen_vl", "new_state": "unavailable",
                      "timestamp": START + timedelta(minutes=10)})
    assert e.association_cortex.zustand["sensor.reifen_vl"] == WEG
    assert snap.extra.get("skipped") == "filtered"


def test_tracker_ohne_raum_ist_ein_ziel_der_anwesenheit():
    e = KontinuumEngine()
    e.register_entity("light.flur", ha_area="flur", domain="light")
    t = START
    for tag in range(3):
        for stunde, zustand in ((7, "not_home"), (17, "home")):
            t = START.replace(hour=stunde) + timedelta(days=tag)
            e.observe({"entity_id": "device_tracker.handy", "new_state": zustand, "timestamp": t})
            e.observe({"entity_id": "light.flur", "new_state": "on" if zustand == "home" else "off",
                       "timestamp": t + timedelta(minutes=1)})
    bild = e.lagebild()
    assert "device_tracker.handy" in bild["anwesenheit"]
    assert bild["stats"]["takte"] > 0


def test_boerse_fuehrt_die_vorhersage_und_die_alte_kette_reist_mit():
    sim = simuliere("geraete", tage=4, saat=1)
    e = KontinuumEngine()
    registriere_engine(e, sim.spur.kopf.entitaeten)
    for ev in sim.spur.ereignisse:
        snap = e.observe(_ereignis(ev))
    assert snap.predictions, "die Börse schweigt"
    assert all(p[3] in ("claustrum", "cerebellum", "interval_timing") for p in snap.predictions)
    assert len(snap.predictions) <= KontinuumEngine.CLAUSTRUM_TOP
    assert "predictions_alt" in snap.extra
    claustrum = snap.extra["claustrum"]
    assert claustrum["ueberraschung_bits"] is not None
    assert set(claustrum["gewichte"]) == {"sequenz", "zeit", "folgezeit", "lage", "extern"}


def test_engine_setzt_nach_dem_laden_bitgleich_fort():
    """Gespeichert, als JSON geschrieben (so wie Home Assistant es tut),
    neu geladen: Danach sagen beide Engines Ereignis für Ereignis dasselbe."""
    sim = simuliere("geraete", tage=6, saat=2)
    ereignisse = sim.spur.ereignisse
    mitte = len(ereignisse) // 2
    with Zeitgeber() as uhr:
        uhr.stelle(ereignisse[0].ts)
        a = KontinuumEngine()
        registriere_engine(a, sim.spur.kopf.entitaeten)
        for ev in ereignisse[:mitte]:
            uhr.stelle(ev.ts)
            a.observe(_ereignis(ev))
        b = KontinuumEngine()
        registriere_engine(b, sim.spur.kopf.entitaeten)
        b.from_dict(json.loads(json.dumps(a.to_dict())))
        # Der Traum-Zufall reist nicht mit; beide bekommen dieselbe Saat.
        a.sleep_consolidation._rng = random.Random(1)
        b.sleep_consolidation._rng = random.Random(1)
        for ev in ereignisse[mitte:]:
            uhr.stelle(ev.ts)
            sa = a.observe(_ereignis(ev))
            sb = b.observe(_ereignis(ev))
            assert sa.predictions == sb.predictions
            assert sa.extra.get("claustrum") == sb.extra.get("claustrum")
    assert a.claustrum.letzte_vorhersage == b.claustrum.letzte_vorhersage
    assert a.lagebild() == b.lagebild()


def test_altes_gehirn_ohne_stufe3_laedt_und_startet_beide_frisch():
    sim = simuliere("klassisch", tage=3, saat=1)
    alt = KontinuumEngine(claustrum=False, lagebild=False)
    registriere_engine(alt, sim.spur.kopf.entitaeten)
    for ev in sim.spur.ereignisse:
        alt.observe(_ereignis(ev))
    blob = json.loads(json.dumps(alt.to_dict()))
    assert "claustrum" not in blob["modules"]
    assert "association_cortex" not in blob["modules"]
    neu = KontinuumEngine()
    neu.from_dict(blob)
    assert neu.tick_count == alt.tick_count
    assert neu.claustrum.ereignisse == 0
    assert neu.association_cortex.ereignisse == 0
    # ... und umgekehrt: ein neues Gehirn in einer Engine ohne Stufe 3
    ohne = KontinuumEngine(claustrum=False, lagebild=False)
    ohne.from_dict(json.loads(json.dumps(neu.to_dict())))
    assert ohne.claustrum is None and ohne.association_cortex is None


def test_alte_kette_bleibt_abrufbar():
    sim = simuliere("klassisch", tage=3, saat=1)
    e = KontinuumEngine(claustrum=False, lagebild=False)
    registriere_engine(e, sim.spur.kopf.entitaeten)
    for ev in sim.spur.ereignisse:
        snap = e.observe(_ereignis(ev))
    assert all(p[3] != "claustrum" for p in (snap.predictions or []))
    assert "claustrum" not in snap.extra
    assert e.lagebild() == {}


def test_rueckmeldungen_wirken_weiter_auf_den_vorschlag():
    """Vorhersage und Vorschlag sind zweierlei: Die Börse sagt vorher, der
    Vorschlag hört weiter auf den Nutzer. Unterdrückt die Habenula alles
    (chronisch abgelehnt), sinkt die Konfidenz des Vorschlags — die
    Vorhersage der Börse bleibt Ereignis für Ereignis dieselbe."""
    sim = simuliere("geraete", tage=5, saat=3)
    with Zeitgeber() as uhr:
        uhr.stelle(sim.spur.ereignisse[0].ts)
        frei = KontinuumEngine()
        genervt = KontinuumEngine()
        genervt.habenula.get_suppression = lambda *_: 1.0
        for e in (frei, genervt):
            registriere_engine(e, sim.spur.kopf.entitaeten)
            e.sleep_consolidation._rng = random.Random(0)
        verglichen = 0
        for ev in sim.spur.ereignisse:
            uhr.stelle(ev.ts)
            a = frei.observe(_ereignis(ev))
            b = genervt.observe(_ereignis(ev))
            assert a.predictions == b.predictions
            da, db = a.extra.get("decision"), b.extra.get("decision")
            if da and db and da.get("token") == db.get("token"):
                assert db["confidence"] <= da["confidence"]  # Boden 0,05
                verglichen += db["confidence"] < da["confidence"]
    assert verglichen > 20
