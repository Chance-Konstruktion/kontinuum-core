"""Der Simulations-Erzeuger der Kontinuum-Spur (Stufe 1, Schritt 2).

Zwei Haustypen, geseedet, direkt im Spur-Format — damit der Messstand
oeffentlich geuebt werden kann, bevor echte Daten da sind:

* **klassisch** — Bewegung, Licht, Tueren, Temperatur/Feuchte; ein
  Tagesablauf mit Streuung, Wochenend-Variante.
* **geraete** — Heizung, Leistung, PV, Batterie, Netz, Wallbox,
  Server-Rauschen; dazu optional ein **Leck-Tag** (Heizluefter laeuft
  22–06 Uhr durch, Marke "anomalie" nur fuer die Diagnose).

Der Erzeuger baut den ROHEN Strom (so wie ein Sensor ihn liefern
wuerde) und duennt ihn mit dem echten Thalamus aus. Damit zeigt der
Bericht ehrlich, wie viel davon nach den Bucket-Eimern uebrig bleibt —
genau der Boden, auf dem die Hypothalamus-Probe C (#2, Notiz 13430)
ihre Zahl bekommt.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from .spur import Entitaet, Ereignis, Kopf, Spur, ausduennen

#: Feste Zone des Simulators — Ortszeit ist Pflicht (Protokoll § 2).
ZEITZONE = timezone(timedelta(hours=2))


@dataclass
class Simulation:
    """Ergebnis: die ausgeuennnte Spur plus der ehrliche Bericht
    (roh vs. behalten), damit der Eimer-Verlust sichtbar ist."""

    spur: Spur
    roh_anzahl: int
    leck_roh: int
    leck_spur: int
    bericht: Dict[str, object] = field(default_factory=dict)


KLASSISCH_ENTITAETEN: Sequence[Entitaet] = (
    Entitaet("binary_sensor.bad_bewegung", "bad", "binary_sensor", "motion"),
    Entitaet("binary_sensor.kueche_bewegung", "kueche", "binary_sensor", "motion"),
    Entitaet("binary_sensor.wohnzimmer_bewegung", "wohnzimmer", "binary_sensor", "motion"),
    Entitaet("binary_sensor.schlafzimmer_bewegung", "schlafzimmer", "binary_sensor", "motion"),
    Entitaet("binary_sensor.flur_bewegung", "flur", "binary_sensor", "motion"),
    Entitaet("binary_sensor.haustuer", "flur", "binary_sensor", "door"),
    Entitaet("light.bad", "bad", "light"),
    Entitaet("light.kueche", "kueche", "light"),
    Entitaet("light.wohnzimmer", "wohnzimmer", "light"),
    Entitaet("light.schlafzimmer", "schlafzimmer", "light"),
    Entitaet("sensor.kueche_temperatur", "kueche", "sensor", "temperature"),
    Entitaet("sensor.wohnzimmer_temperatur", "wohnzimmer", "sensor", "temperature"),
    Entitaet("sensor.bad_feuchte", "bad", "sensor", "humidity"),
)

GERAETE_ENTITAETEN: Sequence[Entitaet] = (
    Entitaet("climate.heizung", "hauswirtschaft", "climate"),
    Entitaet("sensor.wohnzimmer_temperatur", "wohnzimmer", "sensor", "temperature"),
    Entitaet("sensor.aussen_temperatur", "outdoor", "sensor", "temperature"),
    Entitaet("sensor.heizung_leistung", "hauswirtschaft", "sensor", "power"),
    Entitaet("sensor.haus_leistung", "hauswirtschaft", "sensor", "power"),
    Entitaet("sensor.pv_leistung", "outdoor", "sensor", "solar"),
    Entitaet("sensor.batterie_ladestand", "keller", "sensor", "battery"),
    Entitaet("sensor.netz_bezug", "hauswirtschaft", "sensor", "grid"),
    Entitaet("sensor.wohnzimmer_co2", "wohnzimmer", "sensor", "carbon_dioxide", "co2"),
    Entitaet("switch.wallbox", "garage", "switch", None, "wallbox"),
    Entitaet("sensor.keller_server_cpu", "keller", "sensor", None, "cpu"),
    Entitaet("light.wohnzimmer", "wohnzimmer", "light"),
    Entitaet("binary_sensor.haustuer", "flur", "binary_sensor", "door"),
)


def _zeit(tag: datetime, stunde: int, minute: int, versatz: int) -> datetime:
    """Tag + Uhrzeit (+ Streuung) in der festen Zone."""
    basis = tag.replace(hour=stunde, minute=minute, second=0, microsecond=0)
    return basis + timedelta(minutes=versatz)


def _paar(zeit: datetime, entity: str, dauer_min: int) -> List[Ereignis]:
    """Ein Binaer-Ereignis mit Abschaltung."""
    return [
        Ereignis(ts=zeit, entity=entity, zustand="on", alt="off"),
        Ereignis(ts=zeit + timedelta(minutes=dauer_min), entity=entity,
                 zustand="off", alt="on"),
    ]


def _klassischer_tag(tag: datetime, wochenende: bool,
                     rng: random.Random, jitter: int) -> List[Ereignis]:
    s = 45 if not wochenende else 100
    ereignisse: List[Ereignis] = []

    def streuung() -> int:
        return rng.randint(-jitter, jitter)

    ereignisse += _paar(_zeit(tag, 6, 40, streuung() + s // 2), "binary_sensor.bad_bewegung", 9)
    ereignisse += _paar(_zeit(tag, 7, 0, streuung() + s // 3), "binary_sensor.kueche_bewegung", 22)
    ereignisse += _paar(_zeit(tag, 7, 0, streuung() + s // 3), "light.kueche", 25)
    ereignisse += _paar(_zeit(tag, 7, 25, streuung() + s // 2), "binary_sensor.haustuer", 2)
    ereignisse += _paar(_zeit(tag, 7, 30, streuung() + s // 2), "binary_sensor.flur_bewegung", 3)
    if not wochenende:
        ereignisse += _paar(_zeit(tag, 12, 30, streuung()), "binary_sensor.kueche_bewegung", 25)
        ereignisse += _paar(_zeit(tag, 17, 40, streuung()), "binary_sensor.haustuer", 2)
        ereignisse += _paar(_zeit(tag, 17, 45, streuung()), "binary_sensor.flur_bewegung", 3)
        ereignisse += _paar(_zeit(tag, 18, 0, streuung()), "light.kueche", 60)
        ereignisse += _paar(_zeit(tag, 18, 5, streuung()), "binary_sensor.wohnzimmer_bewegung", 240)
    else:
        ereignisse += _paar(_zeit(tag, 9, 30, streuung()), "binary_sensor.wohnzimmer_bewegung", 300)
        ereignisse += _paar(_zeit(tag, 19, 0, streuung()), "light.kueche", 45)
    ereignisse += _paar(_zeit(tag, 18, 0, streuung() + s // 2), "light.wohnzimmer", 300)
    ereignisse += _paar(_zeit(tag, 21, 30, streuung()), "light.bad", 25)
    ereignisse += _paar(_zeit(tag, 22, 10, streuung()), "binary_sensor.schlafzimmer_bewegung", 6)
    ereignisse += _paar(_zeit(tag, 22, 15, streuung()), "light.schlafzimmer", 25)

    # Temperatur/Feuchte: Tagesbogen mit Bucket-Wechseln (cool -> comfort -> cool)
    ereignisse.append(Ereignis(_zeit(tag, 6, 55, streuung()), "sensor.kueche_temperatur", "19.4"))
    ereignisse.append(Ereignis(_zeit(tag, 12, 0, streuung()), "sensor.kueche_temperatur", "20.6"))
    ereignisse.append(Ereignis(_zeit(tag, 21, 0, streuung()), "sensor.kueche_temperatur", "18.7"))
    ereignisse.append(Ereignis(_zeit(tag, 7, 5, streuung()), "sensor.wohnzimmer_temperatur", "18.2"))
    ereignisse.append(Ereignis(_zeit(tag, 17, 30, streuung()), "sensor.wohnzimmer_temperatur", "21.1"))
    ereignisse.append(Ereignis(_zeit(tag, 7, 20, streuung()), "sensor.bad_feuchte", "28"))
    ereignisse.append(Ereignis(_zeit(tag, 8, 10, streuung()), "sensor.bad_feuchte", "62"))
    ereignisse.append(Ereignis(_zeit(tag, 20, 30, streuung()), "sensor.bad_feuchte", "44"))
    return ereignisse


def _geraete_tag(tag: datetime, rng: random.Random, jitter: int,
                 leck: bool) -> List[Ereignis]:
    """Ein Geraete-Haus-Tag. `leck` = heute Nacht laeuft der Heizluefter
    von 22:00 bis 06:00 des Folgetags durch."""
    ereignisse: List[Ereignis] = []

    def streuung() -> int:
        return rng.randint(-jitter, jitter)

    marke = "anomalie" if leck else None

    # Heizung: zwei Heizfenster, dazwischen idle
    ereignisse.append(Ereignis(_zeit(tag, 5, 30, streuung()), "climate.heizung", "heating", "idle"))
    ereignisse.append(Ereignis(_zeit(tag, 5, 31, streuung()), "sensor.heizung_leistung", "1800"))
    ereignisse.append(Ereignis(_zeit(tag, 7, 0, streuung()), "sensor.heizung_leistung", "800"))
    ereignisse.append(Ereignis(_zeit(tag, 9, 0, streuung()), "climate.heizung", "idle", "heating"))
    ereignisse.append(Ereignis(_zeit(tag, 9, 1, streuung()), "sensor.heizung_leistung", "30"))
    ereignisse.append(Ereignis(_zeit(tag, 16, 30, streuung()), "climate.heizung", "heating", "idle"))
    ereignisse.append(Ereignis(_zeit(tag, 16, 31, streuung()), "sensor.heizung_leistung", "1500"))
    ereignisse.append(Ereignis(_zeit(tag, 22, 0, streuung()), "climate.heizung", "idle", "heating"))
    ereignisse.append(Ereignis(_zeit(tag, 22, 1, streuung()), "sensor.heizung_leistung", "0"))

    # Temperaturen (Bucket-Wechsel cold/cool/comfort)
    ereignisse.append(Ereignis(_zeit(tag, 6, 0, streuung()), "sensor.aussen_temperatur", "7.5"))
    ereignisse.append(Ereignis(_zeit(tag, 14, 0, streuung()), "sensor.aussen_temperatur", "15.2"))
    ereignisse.append(Ereignis(_zeit(tag, 22, 30, streuung()), "sensor.aussen_temperatur", "9.1"))
    ereignisse.append(Ereignis(_zeit(tag, 6, 5, streuung()), "sensor.wohnzimmer_temperatur", "19.6"))
    ereignisse.append(Ereignis(_zeit(tag, 17, 0, streuung()), "sensor.wohnzimmer_temperatur", "21.2"))
    ereignisse.append(Ereignis(_zeit(tag, 23, 0, streuung()), "sensor.wohnzimmer_temperatur", "20.4"))

    # PV-Bogen und Batterie
    for stunde, wert in ((7, "80"), (10, "600"), (13, "2400"), (17, "900"), (20, "0")):
        ereignisse.append(Ereignis(_zeit(tag, stunde, 5, streuung()), "sensor.pv_leistung", wert))
    for stunde, wert in ((8, "55"), (13, "85"), (19, "45"), (23, "22")):
        ereignisse.append(Ereignis(_zeit(tag, stunde, 20, streuung()), "sensor.batterie_ladestand", wert))
    ereignisse.append(Ereignis(_zeit(tag, 13, 30, streuung()), "sensor.netz_bezug", "-120"))
    ereignisse.append(Ereignis(_zeit(tag, 20, 30, streuung()), "sensor.netz_bezug", "350"))

    # Hausleistung (aggregiert) + Server-Rauschen
    for stunde, wert in ((6, "250"), (12, "420"), (18, "880"), (22, "180")):
        ereignisse.append(Ereignis(_zeit(tag, stunde, 10, streuung()), "sensor.haus_leistung", wert))
    for stunde, wert in ((3, "22"), (9, "48"), (14, "78"), (19, "51"), (23, "30")):
        ereignisse.append(Ereignis(_zeit(tag, stunde, 40, streuung()), "sensor.keller_server_cpu", wert))
    ereignisse.append(Ereignis(_zeit(tag, 11, 0, streuung()), "sensor.wohnzimmer_co2", "720"))
    ereignisse.append(Ereignis(_zeit(tag, 21, 30, streuung()), "sensor.wohnzimmer_co2", "430"))

    # Wallbox an manchen Abenden; Tuer und Licht
    if rng.random() < 0.6:
        ereignisse += _paar(_zeit(tag, 18, 15, streuung()), "switch.wallbox", 180)
    ereignisse += _paar(_zeit(tag, 7, 20, streuung()), "binary_sensor.haustuer", 2)
    ereignisse += _paar(_zeit(tag, 17, 50, streuung()), "binary_sensor.haustuer", 2)
    ereignisse += _paar(_zeit(tag, 18, 0, streuung()), "light.wohnzimmer", 300)

    if leck:
        # Das Leck: der Heizluefter laeuft ab 22:00 DURCH. Der rohe
        # Sensor meldet alle 10 Minuten (~800 W, leicht schwankend);
        # die Eimer sehen davon fast nichts (genau das ist der Befund).
        for i in range(6 * 8):  # 22:00 bis 06:00, alle 10 Minuten
            ts = _zeit(tag, 22, 0, 0) + timedelta(minutes=10 * i)
            wert = 800.0 + rng.uniform(-5.0, 5.0)
            ereignisse.append(Ereignis(ts=ts, entity="sensor.heizung_leistung",
                                       zustand=f"{wert:.1f}", marke=marke))
        ereignisse.append(Ereignis(_zeit(tag, 22, 5, 0), "sensor.haus_leistung",
                                   "900", marke=marke))
    return ereignisse


def simuliere(
    haus_typ: str = "klassisch",
    tage: int = 14,
    saat: int = 1,
    start: Optional[datetime] = None,
    mit_leck_tag: bool = True,
    jitter_minuten: int = 6,
) -> Simulation:
    """Erzeugt eine geseedete Spur. Gleiche Saat ⇒ gleiche Bytes."""
    if haus_typ not in ("klassisch", "geraete"):
        raise ValueError(f"haus_typ ist 'klassisch' oder 'geraete': {haus_typ!r}")
    if tage < 1:
        raise ValueError("tage mindestens 1")
    start = start or datetime(2026, 3, 2, 0, 0, 0, tzinfo=ZEITZONE)
    rng = random.Random(saat)
    entitaeten = KLASSISCH_ENTITAETEN if haus_typ == "klassisch" else GERAETE_ENTITAETEN

    leck_tag_index = max(2, tage // 2)
    roh: List[Ereignis] = []
    for nummer in range(tage):
        tag = start + timedelta(days=nummer)
        if haus_typ == "klassisch":
            wochenende = tag.weekday() >= 5
            roh += _klassischer_tag(tag, wochenende, rng, jitter_minuten)
        else:
            roh += _geraete_tag(tag, rng, jitter_minuten,
                                leck=mit_leck_tag and nummer == leck_tag_index)

    roh.sort(key=lambda e: (e.ts, e.entity, e.zustand))
    behalten = ausduennen(entitaeten, roh)
    zeitraum = (roh[0].ts.isoformat(), roh[-1].ts.isoformat())
    kopf = Kopf(
        haus=f"sim-{haus_typ}-s{saat}",
        haus_typ=haus_typ,
        quelle="simulation",
        zeitzone="Europe/Berlin",
        konverter="benchmarks/spur/simulator.py 1",
        entitaeten=entitaeten,
        lizenz="eigener Erzeuger (Simulation)",
        zeitraum=zeitraum,
    )
    spur = Spur(kopf=kopf, ereignisse=behalten)
    leck_roh = sum(1 for e in roh if e.marke == "anomalie")
    leck_spur = sum(1 for e in behalten if e.marke == "anomalie")
    bericht = {
        "roh": len(roh),
        "behalten": len(behalten),
        "eimer_verlust": len(roh) - len(behalten),
        "leck_roh": leck_roh,
        "leck_behalten": leck_spur,
    }
    return Simulation(spur=spur, roh_anzahl=len(roh),
                      leck_roh=leck_roh, leck_spur=leck_spur, bericht=bericht)
