"""Simulierter Haushalt mit zwei Personen, Auto, Fernseher und PC.

Der Prüfstand für die Anwesenheit (Assoziationskortex). Vorlage ist die
Art Zusammenhang, die ein Haus von allein lernen soll:

    „Sind die Reifendrucksensoren weggefahren, ist der Fernseher aus und
    der PC an, dann ist jemand da. Gilt dasselbe, aber der Fernseher ist an
    und der PC im Standby, dann ist diese Person weg.“

Zwei Personen teilen sich ein Auto. Person A arbeitet oft zu Hause am PC,
Person B schaut abends fern. Wer mit dem Auto wegfährt, nimmt die vier
Reifendrucksensoren mit: Sie melden sich ab (``unavailable``), wenn das
Auto außer Funkweite ist. Dazu kommen Funkaussetzer, während das Auto
daheim steht — die Sensoren sind also kein sauberes Signal, nur ein Indiz.

Die Handys melden sich als ``device_tracker`` (home/not_home), so wie ein
Router das sieht: verspätet, mit WLAN-Schlaf-Flattern und mit Phasen
``unknown``. Sie sind im Training die einzigen Etiketten. Die WAHRE
Anwesenheit kennt nur der Prüfstand.

Alles ist geseedet: gleiche Saat, gleiche Ereignisse.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Sequence, Tuple

from .spur import Entitaet, Ereignis

ZEITZONE = timezone(timedelta(hours=2))

REIFEN = ("sensor.reifen_vl", "sensor.reifen_vr", "sensor.reifen_hl", "sensor.reifen_hr")

ENTITAETEN: Sequence[Entitaet] = (
    *(Entitaet(r, "garage", "sensor", None, "tpms") for r in REIFEN),
    Entitaet("media_player.fernseher", "wohnzimmer", "media_player"),
    Entitaet("sensor.fernseher_leistung", "wohnzimmer", "sensor", "power"),
    Entitaet("binary_sensor.rechner_online", "buero", "binary_sensor", "connectivity"),
    Entitaet("sensor.rechner_leistung", "buero", "sensor", "power"),
    Entitaet("light.wohnzimmer", "wohnzimmer", "light"),
    Entitaet("light.buero", "buero", "light"),
    Entitaet("light.kueche", "kueche", "light"),
    Entitaet("sensor.wasserkocher_leistung", "kueche", "sensor", "power"),
    Entitaet("device_tracker.handy_a", "flur", "device_tracker"),
    Entitaet("device_tracker.handy_b", "flur", "device_tracker"),
)

PERSONEN = ("a", "b")
TRACKER = {"a": "device_tracker.handy_a", "b": "device_tracker.handy_b"}


@dataclass
class Haushalt:
    """Ergebnis: Ereignisse (roh, mit unavailable), die Wahrheit je Person
    als Liste von Abwesenheits-Intervallen, und die Entitätentabelle."""

    entitaeten: Sequence[Entitaet]
    ereignisse: List[Ereignis]
    abwesend: Dict[str, List[Tuple[datetime, datetime]]]
    start: datetime
    tage: int
    bericht: Dict[str, object] = field(default_factory=dict)

    def zuhause(self, person: str, zeit: datetime) -> bool:
        for von, bis in self.abwesend[person]:
            if von <= zeit < bis:
                return False
        return True


def _um(tag: datetime, stunde: float, rng: random.Random, streuung_min: int = 0) -> datetime:
    basis = tag + timedelta(hours=stunde)
    if streuung_min:
        basis += timedelta(minutes=rng.randint(-streuung_min, streuung_min))
    return basis


def _tagesplan(tag: datetime, rng: random.Random) -> Tuple[Dict[str, List[Tuple[datetime, datetime]]], List[Tuple[datetime, datetime]]]:
    """Abwesenheiten je Person und die Fahrten des Autos an einem Tag."""
    wochenende = tag.weekday() >= 5
    weg: Dict[str, List[Tuple[datetime, datetime]]] = {"a": [], "b": []}
    auto: List[Tuple[datetime, datetime]] = []

    if not wochenende:
        a_modus = rng.choices(["homeoffice", "buero", "frei"], [0.5, 0.4, 0.1])[0]
        b_arbeitet = rng.random() < 0.75
        a_auto = False
        if a_modus == "buero":
            von, bis = _um(tag, 7.7, rng, 15), _um(tag, 17.2, rng, 25)
            weg["a"].append((von, bis))
            auto.append((von, bis))
            a_auto = True
        elif a_modus == "frei" and rng.random() < 0.6:
            von = _um(tag, 10.5, rng, 60)
            bis = von + timedelta(minutes=rng.randint(60, 150))
            weg["a"].append((von, bis))
            auto.append((von, bis))
        if b_arbeitet:
            von, bis = _um(tag, 7.4, rng, 10), _um(tag, 16.5, rng, 30)
            weg["b"].append((von, bis))
            # B nimmt das Auto, wenn A es nicht braucht (sonst Bus/Rad)
            if not a_auto and rng.random() < 0.6 and not any(
                    v < bis and von < b for v, b in auto):
                auto.append((von, bis))
        # Abends geht B manchmal noch weg, mit Auto wenn es da ist
        if rng.random() < 0.25:
            von = _um(tag, 19.0, rng, 30)
            bis = von + timedelta(minutes=rng.randint(90, 180))
            weg["b"].append((von, bis))
            if not any(v < bis and von < b for v, b in auto) and rng.random() < 0.7:
                auto.append((von, bis))
    else:
        if rng.random() < 0.5:  # gemeinsamer Ausflug
            von = _um(tag, 13.0, rng, 60)
            bis = von + timedelta(minutes=rng.randint(150, 300))
            weg["a"].append((von, bis))
            weg["b"].append((von, bis))
            auto.append((von, bis))
        if rng.random() < 0.3:  # A allein unterwegs (Sport, mit Auto)
            von = _um(tag, 9.5, rng, 45)
            bis = von + timedelta(minutes=rng.randint(60, 120))
            if not any(v < bis and von < b for v, b in weg["a"]):
                weg["a"].append((von, bis))
                if not any(v < bis and von < b for v, b in auto):
                    auto.append((von, bis))
    for p in weg:
        weg[p].sort()
    auto.sort()
    return weg, auto


def _da(intervalle: List[Tuple[datetime, datetime]], von: datetime, bis: datetime) -> bool:
    """Ist die Person im ganzen Fenster [von, bis) zu Hause?"""
    return not any(v < bis and von < b for v, b in intervalle)


def simuliere_haushalt(tage: int = 70, saat: int = 1,
                       start: datetime = None) -> Haushalt:
    rng = random.Random(saat)
    start = start or datetime(2026, 3, 2, 0, 0, tzinfo=ZEITZONE)
    ereignisse: List[Ereignis] = []
    abwesend: Dict[str, List[Tuple[datetime, datetime]]] = {"a": [], "b": []}
    autofahrten: List[Tuple[datetime, datetime]] = []

    def e(zeit: datetime, entity: str, zustand: str) -> None:
        ereignisse.append(Ereignis(ts=zeit, entity=entity, zustand=zustand))

    for nummer in range(tage):
        tag = start + timedelta(days=nummer)
        weg, auto = _tagesplan(tag, rng)
        for p in PERSONEN:
            abwesend[p].extend(weg[p])
        autofahrten.extend(auto)
        wochenende = tag.weekday() >= 5
        a_weg, b_weg = weg["a"], weg["b"]

        # -- PC (Person A) ------------------------------------------------
        fenster = []
        if not wochenende:
            fenster += [(8.5, 12.0, 0.95), (13.0, 17.0, 0.9), (20.0, 23.0, 0.55)]
        else:
            fenster += [(10.0, 12.5, 0.5), (20.5, 23.5, 0.5)]
        for von_h, bis_h, p in fenster:
            von, bis = _um(tag, von_h, rng, 15), _um(tag, bis_h, rng, 20)
            if rng.random() < p and _da(a_weg, von, bis):
                e(von, "binary_sensor.rechner_online", "on")
                e(von + timedelta(seconds=40), "sensor.rechner_leistung", f"{rng.uniform(85, 130):.1f}")
                e(bis, "sensor.rechner_leistung", f"{rng.uniform(2.0, 4.0):.1f}")
                e(bis + timedelta(seconds=20), "binary_sensor.rechner_online", "off")
                if bis_h >= 17.5 or von_h >= 17.5:
                    e(von + timedelta(minutes=2), "light.buero", "on")
                    e(bis + timedelta(minutes=1), "light.buero", "off")

        # -- Fernseher (Person B, abends; nachmittags wenn A weg) ---------
        tv = []
        abend_von, abend_bis = _um(tag, 19.5, rng, 25), _um(tag, 22.6, rng, 30)
        if rng.random() < 0.8 and _da(b_weg, abend_von, abend_bis):
            tv.append((abend_von, abend_bis))
        elif wochenende and rng.random() < 0.6 and _da(a_weg, abend_von, abend_bis):
            tv.append((abend_von, abend_bis))  # A schaut allein
        nachmittag_von, nachmittag_bis = _um(tag, 15.0, rng, 30), _um(tag, 17.0, rng, 30)
        if (rng.random() < 0.4 and _da(b_weg, nachmittag_von, nachmittag_bis)
                and not _da(a_weg, nachmittag_von, nachmittag_bis)):
            tv.append((nachmittag_von, nachmittag_bis))
        for von, bis in tv:
            e(von, "media_player.fernseher", "playing")
            e(von + timedelta(seconds=30), "sensor.fernseher_leistung", f"{rng.uniform(70, 95):.1f}")
            e(bis, "media_player.fernseher", "standby")
            e(bis + timedelta(seconds=30), "sensor.fernseher_leistung", f"{rng.uniform(0.6, 1.2):.1f}")
            if bis.hour >= 18 or von.hour >= 18:
                e(von + timedelta(minutes=1), "light.wohnzimmer", "on")
                e(bis + timedelta(minutes=2), "light.wohnzimmer", "off")

        # -- Küche: Wasserkocher morgens und nachmittags -----------------
        for stunde, wer in ((6.8, ("a", "b")), (15.2, ("a",)), (18.3, ("a", "b"))):
            zeit = _um(tag, stunde, rng, 20)
            if any(_da(weg[p], zeit, zeit + timedelta(minutes=5)) for p in wer) and rng.random() < 0.7:
                e(zeit, "light.kueche", "on")
                e(zeit + timedelta(seconds=30), "sensor.wasserkocher_leistung", f"{rng.uniform(1900, 2200):.0f}")
                e(zeit + timedelta(minutes=3), "sensor.wasserkocher_leistung", "0.0")
                e(zeit + timedelta(minutes=rng.randint(6, 20)), "light.kueche", "off")

        # -- Handys (Etiketten): verspätet, flatternd, manchmal unknown --
        for p in PERSONEN:
            tracker = TRACKER[p]
            for von, bis in weg[p]:
                e(von + timedelta(minutes=rng.randint(4, 15)), tracker, "not_home")
                e(bis + timedelta(minutes=rng.randint(1, 8)), tracker, "home")
            for _ in range(rng.randint(0, 2)):  # WLAN-Schlaf zu Hause
                zeit = _um(tag, rng.uniform(0.5, 23.0), rng)
                if _da(weg[p], zeit, zeit + timedelta(minutes=25)):
                    e(zeit, tracker, "not_home")
                    e(zeit + timedelta(minutes=rng.randint(8, 20)), tracker, "home")
            if rng.random() < 0.08:  # Handy aus oder Router-Neustart
                zeit = _um(tag, rng.uniform(1.0, 22.0), rng)
                e(zeit, tracker, "unknown")
                bis = zeit + timedelta(minutes=rng.randint(20, 120))
                e(bis, tracker, "home" if _da(weg[p], bis, bis + timedelta(seconds=1)) else "not_home")

    # -- Auto: die vier Reifen ------------------------------------------
    autofahrten.sort()
    for von, bis in autofahrten:
        for reifen in REIFEN:
            e(von + timedelta(minutes=rng.randint(12, 18)), reifen, "unavailable")
            e(bis + timedelta(minutes=rng.randint(1, 4)), reifen, f"{rng.uniform(2.35, 2.45):.2f}")
    # Funkaussetzer bei stehendem Auto (je Reifen ~1x pro Tag, 20-40 min)
    for nummer in range(tage):
        tag = start + timedelta(days=nummer)
        for reifen in REIFEN:
            if rng.random() < 0.6:
                zeit = _um(tag, rng.uniform(0.0, 23.0), rng)
                bis = zeit + timedelta(minutes=rng.randint(20, 40))
                if _da(autofahrten, zeit - timedelta(minutes=20), bis + timedelta(minutes=5)):
                    e(zeit, reifen, "unavailable")
                    e(bis, reifen, f"{rng.uniform(2.35, 2.45):.2f}")

    # Startzustand: alles aus, Auto da, beide daheim
    anfang = start - timedelta(minutes=1)
    for reifen in REIFEN:
        ereignisse.append(Ereignis(ts=anfang, entity=reifen, zustand="2.40"))
    for entity, zustand in (("media_player.fernseher", "standby"), ("sensor.fernseher_leistung", "0.9"),
                            ("binary_sensor.rechner_online", "off"), ("sensor.rechner_leistung", "3.0"),
                            ("light.wohnzimmer", "off"), ("light.buero", "off"), ("light.kueche", "off"),
                            ("sensor.wasserkocher_leistung", "0.0"),
                            (TRACKER["a"], "home"), (TRACKER["b"], "home")):
        ereignisse.append(Ereignis(ts=anfang, entity=entity, zustand=zustand))

    ereignisse.sort(key=lambda x: (x.ts, x.entity))
    for p in PERSONEN:
        abwesend[p].sort()
    anteil = {}
    for p in PERSONEN:
        summe = sum((b - v).total_seconds() for v, b in abwesend[p])
        anteil[p] = round(summe / (tage * 86400.0), 3)
    return Haushalt(
        entitaeten=ENTITAETEN,
        ereignisse=ereignisse,
        abwesend=abwesend,
        start=start,
        tage=tage,
        bericht={"ereignisse": len(ereignisse), "abwesenheitsanteil": anteil,
                 "autofahrten": len(autofahrten)},
    )


__all__ = ["Haushalt", "simuliere_haushalt", "ENTITAETEN", "PERSONEN", "TRACKER", "REIFEN"]
