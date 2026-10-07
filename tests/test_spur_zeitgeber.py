"""Selbsttests des Zeitgebers (Messmittel zu Protokoll § 6, Schritt 6).

Der Befund, der ihn erzwungen hat (05.10.2026, aruba k=4, jeweils
identischer Strom): Die gemessene Trefferquote wanderte mit der WANDUHR
des Messlaufs — 39,1 % um 08:50 lokal (circadiane Rate 1,30), 41,0 % um
19:13 (0,51), 41,3 % um 22:51 (0,55); zwei Laeufe zur selben Stunde
waren bis aufs Ereignis identisch. Ursache: ``modulate_learning`` ohne
Stunde zieht ``datetime.now().hour``, und die Cooldowns (Kleinhirn
300 s, Reticular 5 s, Spatial 60/1800 s) messen Wandzeit, nicht
Hauszeit. Der Zeitgeber schimt die Uhr auf die Ereigniszeit — dieselbe
Semantik, die Schritt 6 in den Kern tragen soll (clock-Parameter wie !8).
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone

from benchmarks.spur.messstand import messe_ursprung
from benchmarks.spur.simulator import simuliere
from benchmarks.spur.zeitgeber import Zeitgeber
from kontinuum_core import KontinuumEngine
from kontinuum_core.neurorhythms import Neurorhythms


def test_zeitgeber_stellt_die_uhr_und_gibt_sie_zurueck():
    echte = time.time()
    moment = datetime(2011, 6, 1, 8, 30, tzinfo=timezone.utc)
    with Zeitgeber() as uhr:
        uhr.stelle(moment)
        assert time.time() == moment.timestamp()
    # Nach dem Block tickt die Wanduhr wieder (kein Leck).
    assert abs(time.time() - echte) < 5.0


def test_zeitgeber_stunde_kommt_aus_der_ereigniszone():
    """Die Hauszeit des Ereignisses, nicht die Zone des Messrechners:
    23:30+05:00 ist Stunde 23, egal wo der Prozess laeuft."""
    uhr = Zeitgeber()
    uhr.stelle(datetime(2011, 6, 1, 23, 30,
                        tzinfo=timezone(timedelta(hours=5))))
    assert uhr.stunde() == 23
    uhr.stelle(datetime(2011, 6, 1, 23, 30, tzinfo=timezone.utc))
    assert uhr.stunde() == 23


def test_zeitgeber_gibt_die_ereignisstunde_an_die_lernrate():
    """``modulate_learning`` ohne Stunde zieht sonst die Wanduhr-Stunde.
    Mit Zeitgeber gilt die Ereignisstunde — 08:00 ist der circadiane
    Peak (Multiplikator 1,3), 20:00 der Tiefpunkt (0,5)."""
    # Der feste Faktor ist seit 06.10.2026 aus (neutral 1,0); fuer diesen
    # Test des Zeitgebers wird er gezielt eingeschaltet.
    rhythm = Neurorhythms()
    rhythm.CIRCADIAN_FEST = True
    with Zeitgeber() as uhr:
        uhr.stelle(datetime(2011, 6, 1, 8, 0, tzinfo=timezone.utc))
        assert abs(rhythm.modulate_learning(1.0) - 1.3) < 1e-9
        rhythm2 = Neurorhythms()
        rhythm2.CIRCADIAN_FEST = True
        uhr.stelle(datetime(2011, 6, 1, 20, 0, tzinfo=timezone.utc))
        assert abs(rhythm2.modulate_learning(1.0) - 0.5) < 1e-9


def test_messstand_haelt_die_uhr_auf_dem_ereignis():
    """Im Messkern steht die Uhr waehrend ``observe`` auf dem Ereignis —
    die synthetischen Ereignisse liegen Monate von der Wanduhr entfernt,
    also ist die Naehe ein Beweis und kein Zufall."""
    sim = simuliere("klassisch", tage=42, saat=3)
    erwartet = sim.spur.ereignisse[0].ts
    gesehen = []

    def bauer():
        engine = KontinuumEngine()
        original = engine.observe

        def merken(ereignis):
            gesehen.append(abs(time.time() - ereignis["timestamp"].timestamp()))
            return original(ereignis)

        engine.observe = merken  # type: ignore[method-assign]
        return engine

    messe_ursprung(sim.spur, train_wochen=4, engine_bauer=bauer)
    assert gesehen, "die Engine hat kein Ereignis gesehen"
    assert max(gesehen) < 1.0, f"Wanduhr statt Ereigniszeit: {max(gesehen)}s"


def test_zeitgeber_macht_die_zeitzone_des_laufs_egal():
    """Der Kern-Befund als Waechter: MIT Zeitgeber liefern zwei Laeufe in
    verschiedenen Prozess-Zonen (UTC vs. Asia/Tokyo, 9 Stunden
    Unterschied) identische Zahlen. OHNE Zeitgeber taten sie das nicht —
    92/115 vs. 88/118 Top-1-Treffer auf diesem Strom (Saat 1, 42 Tage,
    Ursprung 4)."""
    if not hasattr(time, "tzset"):
        return  # nur POSIX; die CI faehrt Linux
    sim = simuliere("klassisch", tage=42, saat=1)
    alte_zone = os.environ.get("TZ")
    try:
        ergebnisse = []
        for zone in ("UTC", "Asia/Tokyo"):
            os.environ["TZ"] = zone
            time.tzset()
            u = messe_ursprung(sim.spur, train_wochen=4)
            ergebnisse.append(
                {name: (b.gesamt, b.top1, b.top3)
                 for name, b in u.systeme.items()}
            )
        assert ergebnisse[0] == ergebnisse[1]
    finally:
        if alte_zone is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = alte_zone
        time.tzset()
