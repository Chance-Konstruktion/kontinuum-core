"""CASAS -> Kontinuum-Spur (Stufe 1, Schritt 4).

Quelle: Zenodo-Record 17180309, „CASAS Smart Home dataset (aruba, cairo,
milan, tulum)" — Lizenz **CC BY 4.0**, Pflichtzitat unten (im Kopf der
Spur). Archiv `new_labeled_data.zip`, MD5
`86954063e1d2099d288f59227b19a749`; der CI-Job laedt es selbst und
prueft die Pruefsumme — die Rohdaten liegen NIE im Repo.

Format je Zeile (Leerraum/Tab getrennt):
``YYYY-MM-DD HH:MM:SS.ffffff  SENSOR  WERT  [AKTIVITAET  [begin|end]]``
Sensoren: ``M`` Bewegung, ``D`` Tuer, ``T`` Temperatur, ``L`` Licht;
unbekannte Praefixe werden gezaehlt und uebersprungen (kein stilles
Schlucken).

Ehrliche Grenzen (im Bericht und im Kopf sichtbar):

* **Zeitzone:** Die Dateien tragen lokale Zeit OHNE Zone; die Haeuser
  stehen in Walla Walla, Washington -> ``America/Los_Angeles``. Die
  Umstellungstage (DST) sind damit an zwei Stunden im Jahr mehrdeutig;
  wir nehmen die erste Auslegung (fold=0). Fehlt die Zeitzonen-Datenbank
  (slim-Images), faellt der Konverter auf eine FESTE Zone -08:00 zurueck
  und sagt das im Bericht — keine stille Sommerzeit-Luege.
* **Aktivitaets-Labels gehoeren nicht in die Spur** (Datensparsamkeit,
  Abnahme 13642); sie werden nur gezaehlt (spaetere Diagnose).
* **Jeder Sensor ist sein eigener Raum** (``raum = Sensor-ID``), sonst
  vermischten sich gleiche Semantiken zu einem Token (PIPELINE.md).
"""
from __future__ import annotations

import argparse
import heapq
import re
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone as _timezone
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from .spur import (
    Entitaet,
    Ereignis,
    Kopf,
    duenne_aus_strom,
    schreibe_spur_strom,
)

#: Praefix -> (domain, semantik). Was hier fehlt, wird gezaehlt und
#: uebersprungen — nie geraten.
PRAEFIX: Dict[str, Tuple[str, str]] = {
    "m": ("binary_sensor", "motion"),
    "d": ("binary_sensor", "door"),
    "t": ("sensor", "temperature"),
    "l": ("light", "light"),
}

ZEITZONEN_NAME = "America/Los_Angeles"
ZITAT = (
    "Cook, D., Crandall, A., Thomas, B., & Krishnan, N. (2013). CASAS: "
    "A smart home in a box. IEEE Computer, 46(7), 62-69. "
    "doi:10.1109/MC.2012.328"
)
LIZENZ = "CC BY 4.0"
KONVERTER = "benchmarks/spur/casas.py 1"
ARCHIV_MD5 = "86954063e1d2099d288f59227b19a749"

#: Unordnung, die die Fenster-Sortierung tragen kann (Ereignisse).
FENSTER = 50_000

#: Semantiken, deren Wert eine ZAHL ist (mit optionalem
#: Einheiten-Schwanz — die Quelle liefert vereinzelt "28.55c").
NUMERISCHE_SEMANTIK = frozenset({
    "temperature", "humidity", "power", "energy", "battery", "voltage",
    "illuminance", "pressure", "co2", "solar", "grid", "cpu", "gpu",
})

_ZAHL = re.compile(r"^[+-]?\d+(?:\.\d+)?")

#: Grundmenge der CASAS-Haeuser (tulum ist in zwei Dateien geteilt).
HAEUSER = ("aruba", "cairo", "milan", "tulum1", "tulum2")


class CasasFehler(ValueError):
    """Formfehler der Quelle — mit Zeilennummer, nie still."""


@dataclass(frozen=True)
class Zeile:
    ts: datetime
    sensor: str
    wert: str
    aktivitaet: Optional[str]


_MUSTER = re.compile(
    r"^(?P<datum>\d{4}-\d{2}-\d{2})\s+(?P<zeit>\d{2}:\d{2}:\d{2}(?:\.\d+)?)"
    r"\s+(?P<sensor>\S+)\s+(?P<wert>\S+)(?:\s+(?P<rest>.*))?$"
)


def _normalisiere(wert: str, semantik: str, zaehler: dict) -> str:
    """Zahlen-Sensoren von Einheiten-Schwaenzen befreien ("28.55c" ->
    "28.55"). Kein stilles Geradebiegen: jede Aenderung wird gezaehlt
    und steht im Bericht. Nicht-Zahlen bleiben unangetastet."""
    if semantik not in NUMERISCHE_SEMANTIK:
        return wert
    treffer = _ZAHL.match(wert)
    if not treffer:
        return wert
    sauber = treffer.group(0)
    if sauber != wert:
        zaehler["normalisierte_werte"] = zaehler.get("normalisierte_werte", 0) + 1
    return sauber


def _zeitzone():
    """Echte Zone, wenn die Datenbank da ist — sonst feste Zone MIT
    Ansage (der Bericht nennt den Rueckfall)."""
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(ZEITZONEN_NAME), ZEITZONEN_NAME
    except Exception:
        return _timezone(timedelta(hours=-8)), "feste Zone -08:00 (DST nicht abgebildet)"


def zerlege_zeile(zeile: str, zonen_info, nummer: int = 0) -> Zeile:
    """Eine CASAS-Zeile -> Zeile. Der Aktivitaetstext (kann Leerzeichen
    enthalten) wird abgeschnitten und nur gezaehlt."""
    treffer = _MUSTER.match(zeile.strip())
    if not treffer:
        raise CasasFehler(f"Zeile {nummer} nicht lesbar: {zeile[:60]!r}")
    zone, _name = zonen_info
    # fromisoformat kann vor Python 3.11 nur 3 oder 6 Nachkommastellen
    hms, punkt, bruch = treffer.group("zeit").partition(".")
    zeit = hms + (punkt + bruch.ljust(6, "0")[:6] if punkt else "")
    ts = datetime.fromisoformat(
        f"{treffer.group('datum')}T{zeit}"
    ).replace(tzinfo=zone)
    rest = (treffer.group("rest") or "").strip() or None
    return Zeile(ts=ts, sensor=treffer.group("sensor").lower(),
                 wert=treffer.group("wert"), aktivitaet=rest)


def entitaeten_fuer(sensoren: Iterable[str]) -> Tuple[List[Entitaet], List[str]]:
    """Baut die Entitaetentabelle; unbekannte Praefixe kommen als Liste
    zurueck (der Aufrufer zaehlt sie und laesst sie weg)."""
    tabelle: List[Entitaet] = []
    unbekannt: List[str] = []
    for sensor in sorted(set(sensoren)):
        praefix = sensor[:1]
        if praefix not in PRAEFIX:
            unbekannt.append(sensor)
            continue
        domain, semantik = PRAEFIX[praefix]
        tabelle.append(
            Entitaet(id=sensor, raum=sensor, domain=domain, semantik=semantik)
        )
    return tabelle, unbekannt


def _rohe_zeilen(archiv: str, dateiname: str) -> Iterator[Tuple[int, str]]:
    with zipfile.ZipFile(archiv) as paket:
        namen = paket.namelist()
        if dateiname not in namen:
            raise CasasFehler(
                f"{dateiname} nicht im Archiv; vorhanden: {namen}"
            )
        with paket.open(dateiname) as datei:
            for nummer, roh in enumerate(datei, start=1):
                yield nummer, roh.decode("utf-8", errors="replace")


def _geordnet(roh: Iterable[Ereignis], fenster: int, zaehler: dict
              ) -> Iterator[Ereignis]:
    """Fenster-Sortierung fuer leicht unordentliche Quellen.

    Die CASAS-Dateien sind fast sortiert — aruba hat aber einen
    Ausreisser-Block (2011-05-23), in dem eine spaetere Zeile vor einer
    frueheren steht. Ein Heap in Dateireihenfolge gibt Ereignisse in
    Zeitfolge heraus, solange die Unordnung kleiner als ``fenster`` ist;
    ``verschoben`` zaehlt die Inversionen relativ zur Datei (Diagnose,
    kein stilles Geradebiegen)."""
    h = []
    letzte_datei = None
    nummer = 0
    for e in roh:
        if letzte_datei is not None and e.ts < letzte_datei:
            zaehler["verschoben"] = zaehler.get("verschoben", 0) + 1
        elif letzte_datei is None or e.ts > letzte_datei:
            letzte_datei = e.ts
        nummer += 1
        heapq.heappush(h, (e.ts, e.entity, e.zustand, nummer, e))
        if len(h) > fenster:
            yield heapq.heappop(h)[4]
    while h:
        yield heapq.heappop(h)[4]


def konvertiere(
    archiv: str,
    haus: str,
    ausgabe: Optional[str] = None,
    wochen: Optional[int] = None,
    dateiname: Optional[str] = None,
) -> dict:
    """Wandelt ein CASAS-Haus in eine Kontinuum-Spur.

    Zwei Durchgaenge: erst die Sensoren sammeln (Entitaetentabelle),
    dann STREAMEND parsen und mit dem ECHTEN Thalamus ausduennen. Der
    rohe Strom wird nie materialisiert (1,7 Mio. Zeilen fuer aruba)."""
    dateiname = dateiname or f"{haus}.txt"
    zonen_info = _zeitzone()

    # Durchgang 1: Sensoren sammeln, Zeitraum bestimmen.
    sensoren: set = set()
    zeilen_je_praefix: Counter = Counter()
    zeilen_gesamt = 0
    mit_aktivitaet = 0
    erste_ts: Optional[datetime] = None
    letzte_ts: Optional[datetime] = None
    grenze: Optional[datetime] = None
    for nummer, roh in _rohe_zeilen(archiv, dateiname):
        if not roh.strip():
            continue
        zeile = zerlege_zeile(roh, zonen_info, nummer)
        zeilen_gesamt += 1
        sensoren.add(zeile.sensor)
        zeilen_je_praefix[zeile.sensor[:1]] += 1
        if zeile.aktivitaet:
            mit_aktivitaet += 1
        if erste_ts is None:
            erste_ts = zeile.ts
            if wochen is not None:
                grenze = erste_ts + timedelta(days=7 * wochen)
        if grenze is not None and zeile.ts >= grenze:
            break
        letzte_ts = zeile.ts

    tabelle, unbekannt = entitaeten_fuer(sensoren)
    bekannte = {e.id for e in tabelle}
    semantik_von = {e.id: (e.semantik or "") for e in tabelle}
    unbekannt_praefixe = Counter(sensor[:1] for sensor in unbekannt)

    # Durchgang 2 (streamend): unbekannte Sensoren zaehlen, Werte
    # normalisieren, Rest ausduennen — ueber den echten Thalamus.
    unbekannt_zeilen: Counter = Counter()
    zaehler: dict = {}
    grenze2 = (
        erste_ts + timedelta(days=7 * wochen) if (wochen and erste_ts) else None
    )

    def strom() -> Iterator[Ereignis]:
        for nummer, roh in _rohe_zeilen(archiv, dateiname):
            if not roh.strip():
                continue
            zeile = zerlege_zeile(roh, zonen_info, nummer)
            if zeile.sensor not in bekannte:
                unbekannt_zeilen[zeile.sensor[:1]] += 1
                continue
            if grenze2 is not None and zeile.ts >= grenze2:
                return
            wert = _normalisiere(zeile.wert, semantik_von[zeile.sensor], zaehler)
            yield Ereignis(ts=zeile.ts, entity=zeile.sensor, zustand=wert)

    geordnet = _geordnet(strom(), FENSTER, zaehler)
    kopf = Kopf(
        haus=f"casas-{haus}",
        haus_typ="klassisch",
        quelle="casas",
        zeitzone=zonen_info[1],
        konverter=KONVERTER,
        entitaeten=tabelle,
        lizenz=LIZENZ,
        zitat=ZITAT,
        zeitraum=(
            (erste_ts.isoformat(), letzte_ts.isoformat())
            if erste_ts is not None and letzte_ts is not None else None
        ),
    )
    behalten_strom = duenne_aus_strom(tabelle, geordnet)
    if ausgabe:
        anzahl = schreibe_spur_strom(kopf, behalten_strom, ausgabe)
    else:
        anzahl = sum(1 for _ in behalten_strom)  # nur fuer kleine Proben
    tage = (
        (letzte_ts - erste_ts).days + 1
        if erste_ts is not None and letzte_ts is not None else 0
    )
    roh_bekannt = zeilen_gesamt - sum(unbekannt_zeilen.values())
    return {
        "haus": kopf.haus,
        "datei": dateiname,
        "zeilen": zeilen_gesamt,
        "zeilen_je_praefix": dict(sorted(zeilen_je_praefix.items())),
        "sensoren": len(sensoren),
        "entitaeten": len(tabelle),
        "unbekannte_sensoren": len(unbekannt),
        "unbekannte_praefixe": dict(sorted(unbekannt_praefixe.items())),
        "unbekannte_zeilen": dict(sorted(unbekannt_zeilen.items())),
        "aktivitaetszeilen": mit_aktivitaet,
        "ereignisse_vor_ausduennung": roh_bekannt,
        "behalten": anzahl,
        "eimer_verlust": roh_bekannt - anzahl,
        "verschobene_zeilen": zaehler.get("verschoben", 0),
        "normalisierte_werte": zaehler.get("normalisierte_werte", 0),
        "tage": tage,
        "erste_ts": erste_ts.isoformat() if erste_ts else None,
        "letzte_ts": letzte_ts.isoformat() if letzte_ts else None,
        "zeitzone": zonen_info[1],
    }


def _cli(argv=None) -> int:
    zerleger = argparse.ArgumentParser(
        prog="casas",
        description="CASAS-Haus -> Kontinuum-Spur (JSONL). Rohdaten bleiben draussen.",
    )
    zerleger.add_argument("--archiv", required=True,
                          help="Pfad zur new_labeled_data.zip")
    zerleger.add_argument("--haus", required=True, choices=HAEUSER)
    zerleger.add_argument("--ausgabe", required=True, help="Ziel .jsonl")
    zerleger.add_argument("--wochen", type=int, default=None,
                          help="nur die ersten N Wochen (Tests/CI)")
    args = zerleger.parse_args(argv)
    try:
        bericht = konvertiere(args.archiv, args.haus, args.ausgabe,
                              wochen=args.wochen)
    except CasasFehler as fehler:
        print(f"FEHLER: {fehler}", file=sys.stderr)
        return 1
    for schluessel, wert in bericht.items():
        print(f"{schluessel}: {wert}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())


__all__ = [
    "CasasFehler",
    "HAEUSER",
    "PRAEFIX",
    "ZEITZONEN_NAME",
    "ZITAT",
    "LIZENZ",
    "ARCHIV_MD5",
    "zerlege_zeile",
    "entitaeten_fuer",
    "konvertiere",
]
