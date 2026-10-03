"""Die Kontinuum-Spur (Stufe 1, Issue #77/Leit-Ticket #2).

Ein gemeinsames Ereignisformat fuer drei Quellen — echte HA-Historie,
CASAS, Simulation — damit der Messstand (Vorhersage des naechsten
Ereignisses gegen B0/B1/B2) auf allen dreien dieselbe Maschine fahren
kann. Entworfen als Vorschlag in kontinuum-core#2, Notiz 13188; die
Nachtraege 13198 (Uhr-Zeit) und 13430 (Bucket-Feinheit, Gegenprobe,
Vokabular-Luecken) sind eingearbeitet.

Kernsaetze des Formats:

* **Token-Granularitaet bleibt die der Engine** — die Spur traegt
  Entity, Zustand und Zeit; die Tokenisierung selbst macht eine
  frische {@code Thalamus}-Instanz mit denselben Tabellen wie das
  Replay. Kein Nachbau, keine zweite Wahrheit.
* **Ortszeit mit Zone ist Pflicht.** Die Stunde steckt im Zeitkontext
  (`thalamus.encode_time_context`) und im SCN — eine Umrechnung auf
  UTC wuerde den Haushaltstag verschieben.
* **Verlustfreie Ausduennung.** Aufeinanderfolgende Ereignisse
  derselben Entity mit gleichem Engine-Token werden entfernt — genau
  die, die der Thalamus selbst verwerfen wuerde. Die Engine sieht
  danach denselben Strom; der Beweis ist der Token-Lauf.
* **Diagnose bleibt dran, Wahrheit nicht.** Optionale `marke`-Felder
  (z. B. "anomalie" fuer ein eingebautes Leck) sind NUR fuer
  Diagnosen; kein Vorhersager darf sie sehen.
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Allow `python -m benchmarks.spur.pruefe_spur` (und direkte Aufrufe)
# ohne editable install — dasselbe Muster wie benchmarks/replay.py.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from kontinuum_core.hypothalamus import (
    CLIMATE_SEMANTICS,
    ENERGY_SEMANTICS,
    STATE_TO_LEVEL,
)
from kontinuum_core.thalamus import HA_AREA_MAP, Thalamus

SPUR_FORMAT = "kontinuum-spur/1"
#: Stand der Kategorie-Tabellen — steigt, wenn sich eine Zuordnung aendert.
KATEGORIEN_STAND = 1

HAUS_TYPEN = ("klassisch", "geraete")
QUELLEN = ("ha-historie", "casas", "simulation")

#: Semantik -> Kategorie, je Haustyp. Jedes Token faellt in GENAU EINE
#: Kategorie; Unbekanntes landet in "sonstiges" (nie stillschweigend weg).
KATEGORIEN: Dict[str, Dict[str, str]] = {
    "klassisch": {
        "light": "licht",
        "switch": "licht",
        "motion": "bewegung",
        "presence": "bewegung",
        "bed_presence": "bewegung",
        "door": "tueren",
        "lock": "tueren",
        "cover": "tueren",
        "climate": "klima",
        "temperature": "klima",
        "humidity": "klima",
        "illuminance": "klima",
        "media": "geraete",
        "vacuum": "geraete",
        "gaming": "geraete",
        "screen": "geraete",
        "network": "infrastruktur",
        "cpu": "infrastruktur",
        "gpu": "infrastruktur",
    },
    "geraete": {
        "light": "licht",
        "switch": "licht",
        "motion": "bewegung",
        "presence": "bewegung",
        "door": "bewegung",
        "temperature": "klima",
        "humidity": "klima",
        "co2": "klima",
        "illuminance": "klima",
        "climate": "heizung",
        "water_heater": "heizung",
        "power": "energie",
        "energy": "energie",
        "solar": "energie",
        "grid": "energie",
        "battery": "energie",
        "voltage": "energie",
        "wallbox": "energie",
        "media": "geraete",
        "vacuum": "geraete",
        "gaming": "geraete",
        "screen": "geraete",
        "network": "infrastruktur",
        "cpu": "infrastruktur",
        "gpu": "infrastruktur",
    },
}
SONSTIGES = "sonstiges"

#: Ein Raum-Slug ist klein, ohne Umlaute/Leerraum — die Engine normalisiert
#: ihn ueber HA_AREA_MAP weiter (z. B. "kueche" -> "kitchen").
RAUM_MUSTER = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class SpurFehler(ValueError):
    """Format-, Tokenisierungs- oder Ordnungsfehler — mit Namen, nie still."""


# ---------------------------------------------------------------------------
# Datentypen
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Entitaet:
    """Eine Entity der Spur. `semantik` uebersteuert (wenn gesetzt) die
    Ableitung aus domain/geraeteklasse — als feste Zuordnungsregel, damit
    CASAS-Sensornamen (m001 …) ohne Rate-Spiel tokenisierbar sind."""

    id: str
    raum: str
    domain: str
    geraeteklasse: Optional[str] = None
    semantik: Optional[str] = None

    def als_json(self) -> dict:
        daten = {"id": self.id, "raum": self.raum, "domain": self.domain}
        if self.geraeteklasse:
            daten["geraeteklasse"] = self.geraeteklasse
        if self.semantik:
            daten["semantik"] = self.semantik
        return daten


@dataclass(frozen=True)
class Ereignis:
    """Ein Zustandswechsel. `marke` ist NUR Diagnose (z. B. "anomalie")
    und darf keinem Vorhersager gezeigt werden."""

    ts: datetime
    entity: str
    zustand: str
    alt: Optional[str] = None
    marke: Optional[str] = None

    def als_json(self) -> dict:
        daten = {
            "ts": self.ts.isoformat(),
            "entity": self.entity,
            "zustand": self.zustand,
        }
        if self.alt is not None:
            daten["alt"] = self.alt
        if self.marke is not None:
            daten["marke"] = self.marke
        return daten


@dataclass(frozen=True)
class Kopf:
    """Der Kopf der Spur: WER, WO, WOHER, WIE LANGE — und die
    Entitaetentabelle, aus der die Tokenisierung beweisbar wird."""

    haus: str
    haus_typ: str
    quelle: str
    zeitzone: str
    konverter: str
    entitaeten: Sequence[Entitaet]
    lizenz: Optional[str] = None
    zeitraum: Optional[Tuple[str, str]] = None

    def als_json(self) -> dict:
        daten = {
            "format": SPUR_FORMAT,
            "haus": self.haus,
            "haus_typ": self.haus_typ,
            "quelle": self.quelle,
            "zeitzone": self.zeitzone,
            "konverter": self.konverter,
            "lizenz": self.lizenz if self.lizenz is not None else "",
            "entitaeten": [e.als_json() for e in self.entitaeten],
            "kategorien_stand": KATEGORIEN_STAND,
        }
        if self.zeitraum:
            daten["zeitraum"] = list(self.zeitraum)
        return daten


@dataclass
class Spur:
    kopf: Kopf
    ereignisse: List[Ereignis] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Zeit
# ---------------------------------------------------------------------------


def lies_ts(text: str) -> datetime:
    """ISO-8601 MIT Zone (oder 'Z'). Eine naive Zeit ist ein Fehler:
    Ortszeit ohne Zone ist nicht eindeutig, und der Haushaltstag haengt
    an der Zone (Protokoll § 2, Regel 1)."""
    if not isinstance(text, str) or not text:
        raise SpurFehler(f"Zeitstempel fehlt oder ist kein Text: {text!r}")
    roh = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        zeit = datetime.fromisoformat(roh)
    except ValueError as problem:
        raise SpurFehler(f"Zeitstempel nicht lesbar: {text!r} ({problem})")
    if zeit.tzinfo is None or zeit.tzinfo.utcoffset(zeit) is None:
        raise SpurFehler(
            f"Zeitstempel ohne Zone: {text!r} — Ortszeit braucht ein Offset "
            "(z. B. +02:00); eine UTC-Umrechnung wuerde den Haushaltstag "
            "verschieben (Protokoll § 2, Regel 1)."
        )
    return zeit


# ---------------------------------------------------------------------------
# Kategorien
# ---------------------------------------------------------------------------


def kategorie(haus_typ: str, semantik: str) -> str:
    """Die Kategorie eines Tokens — genau eine, nie keine."""
    tabelle = KATEGORIEN.get(haus_typ)
    if tabelle is None:
        raise SpurFehler(f"Unbekannter haus_typ: {haus_typ!r}")
    return tabelle.get(semantik, SONSTIGES)


def token_zerlegen(token: str) -> Tuple[str, str, str]:
    """`raum.semantik.zustand` -> Teile. Ein Token mit Punkt im Zustand
    (z. B. climate "heat_cool") bleibt heil: es wird an den ersten zwei
    Punkten getrennt."""
    teile = token.split(".", 2)
    if len(teile) != 3:
        raise SpurFehler(f"Token nicht in raum.semantik.zustand zerlegbar: {token!r}")
    return teile[0], teile[1], teile[2]


# ---------------------------------------------------------------------------
# Tokenisierung — die eine Wahrheit ist der Thalamus
# ---------------------------------------------------------------------------


def registriere(entitaeten: Sequence[Entitaet]) -> Thalamus:
    """Baut einen Thalamus mit genau dieser Entitaetentabelle. Eine
    Entity, die die Engine verwerfen wuerde (kein Raum, keine Semantik),
    ist ein Formatfehler — nicht eine stille Luecke."""
    thalamus = Thalamus()
    for e in entitaeten:
        if not RAUM_MUSTER.match(e.raum or ""):
            raise SpurFehler(f"Raum kein Slug: {e.id}: {e.raum!r}")
        if e.semantik:
            thalamus.custom_semantic_rules.append(
                {
                    "semantic": e.semantik,
                    "entity_regex": "^" + re.escape(e.id.lower()) + "$",
                }
            )
    for e in entitaeten:
        thalamus.register_entity(
            e.id,
            ha_area=e.raum,
            domain=e.domain,
            device_class=e.geraeteklasse or "",
        )
    fehlend = [e.id for e in entitaeten if e.id not in thalamus.entity_semantic]
    if fehlend:
        bericht = thalamus.get_diagnostics()
        raise SpurFehler(
            "Nicht registrierbar (die Engine wuerde ihre Ereignisse stumm "
            f"verwerfen): {fehlend} — Diagnose: {bericht}"
        )
    return thalamus


def tokenisieren(spur: Spur) -> List[str]:
    """Faehrt die Spur durch einen frischen Thalamus und liefert die
    Token-Folge. Jedes Ereignis MUSS ein Token ergeben (sonst
    SpurFehler mit Index) — die ausgeuennnte Spur enthaelt per
    Konstruktion keine Ereignisse mehr, die die Engine verwirft."""
    thalamus = registriere(spur.kopf.entitaeten)
    tokens: List[str] = []
    for i, e in enumerate(spur.ereignisse):
        signal = thalamus.process(e.entity, e.zustand, e.alt, e.ts)
        if signal is None:
            raise SpurFehler(
                f"Ereignis {i} ({e.entity} = {e.zustand!r}) ergibt kein "
                "Token — die Spur ist nicht ausgeuennnt oder die "
                "Entitaetentabelle passt nicht zum Strom."
            )
        tokens.append(signal["token"])
    return tokens


# ---------------------------------------------------------------------------
# Verlustfreie Ausduennung
# ---------------------------------------------------------------------------


def ausduennen(
    entitaeten: Sequence[Entitaet],
    roh: Iterable[Ereignis],
) -> List[Ereignis]:
    """Entfernt genau die Ereignisse, die die Engine selbst verwerfen
    wuerde (unavailable/unknown, gleiches Token in Folge je Entity) —
    mit dem ECHTEN Thalamus, nicht mit einem Nachbau. Ein Ereignis
    einer unbekannten Entity ist dagegen ein Fehler."""
    thalamus = registriere(entitaeten)
    fehlend = set()
    behalten: List[Ereignis] = []
    for e in roh:
        signal = thalamus.process(e.entity, e.zustand, e.alt, e.ts)
        if signal is not None:
            behalten.append(e)
            continue
        if e.entity not in thalamus.entity_semantic:
            fehlend.add(e.entity)
    if fehlend:
        raise SpurFehler(
            "Ereignisse von nicht registrierten Entities: "
            f"{sorted(fehlend)} — die Entitaetentabelle ist unvollstaendig."
        )
    return behalten


# ---------------------------------------------------------------------------
# Lesen / Schreiben
# ---------------------------------------------------------------------------


def _kopf_aus_json(daten: dict) -> Kopf:
    fehlend = [k for k in ("haus", "haus_typ", "quelle", "zeitzone", "konverter")
               if not daten.get(k)]
    if fehlend:
        raise SpurFehler(f"Kopf unvollstaendig, es fehlt: {fehlend}")
    if daten.get("haus_typ") not in HAUS_TYPEN:
        raise SpurFehler(f"haus_typ unbekannt: {daten.get('haus_typ')!r} "
                         f"(erlaubt: {HAUS_TYPEN})")
    if daten.get("quelle") not in QUELLEN:
        raise SpurFehler(f"quelle unbekannt: {daten.get('quelle')!r} "
                         f"(erlaubt: {QUELLEN})")
    entitaeten = []
    for roh in daten.get("entitaeten") or []:
        if not roh.get("id") or not roh.get("raum") or not roh.get("domain"):
            raise SpurFehler(f"Entitaet unvollstaendig: {roh!r}")
        entitaeten.append(
            Entitaet(
                id=roh["id"],
                raum=roh["raum"],
                domain=roh["domain"],
                geraeteklasse=roh.get("geraeteklasse"),
                semantik=roh.get("semantik"),
            )
        )
    if not entitaeten:
        raise SpurFehler("Entitaetentabelle ist leer — ohne sie ist keine "
                         "Tokenisierung beweisbar.")
    kennungen = [e.id for e in entitaeten]
    if len(set(kennungen)) != len(kennungen):
        doppelt = sorted({k for k in kennungen if kennungen.count(k) > 1})
        raise SpurFehler(f"Entitaet doppelt in der Tabelle: {doppelt}")
    zeitraum = daten.get("zeitraum")
    if zeitraum is not None and (not isinstance(zeitraum, list) or len(zeitraum) != 2):
        raise SpurFehler(f"zeitraum muss [start, ende] sein: {zeitraum!r}")
    return Kopf(
        haus=daten["haus"],
        haus_typ=daten["haus_typ"],
        quelle=daten["quelle"],
        zeitzone=daten["zeitzone"],
        konverter=daten["konverter"],
        entitaeten=entitaeten,
        lizenz=daten.get("lizenz"),
        zeitraum=(zeitraum[0], zeitraum[1]) if zeitraum else None,
    )


def lies_spur(pfad) -> Spur:
    """Liest eine Kontinuum-Spur (JSONL). Erste Zeile Kopf, danach ein
    Ereignis je Zeile. Ordnung wird geprueft: nicht absteigend."""
    text = Path(pfad).read_text(encoding="utf-8")
    zeilen = [z for z in text.split("\n") if z.strip()]
    if not zeilen:
        raise SpurFehler(f"Leere Spur: {pfad}")
    try:
        kopf_roh = json.loads(zeilen[0])
    except json.JSONDecodeError as problem:
        raise SpurFehler(f"Kopfzeile kein JSON: {problem}")
    if kopf_roh.get("format") != SPUR_FORMAT:
        raise SpurFehler(
            f"Kopf traegt {kopf_roh.get('format')!r}, erwartet {SPUR_FORMAT!r}"
        )
    kopf = _kopf_aus_json(kopf_roh)
    bekannt = {e.id for e in kopf.entitaeten}
    ereignisse: List[Ereignis] = []
    letzte_zeit: Optional[datetime] = None
    for nummer, zeile in enumerate(zeilen[1:], start=2):
        try:
            roh = json.loads(zeile)
        except json.JSONDecodeError as problem:
            raise SpurFehler(f"Zeile {nummer} kein JSON: {problem}")
        for pflicht in ("ts", "entity", "zustand"):
            if not roh.get(pflicht) and roh.get(pflicht) != 0:
                raise SpurFehler(f"Zeile {nummer}: '{pflicht}' fehlt")
        if roh["entity"] not in bekannt:
            raise SpurFehler(
                f"Zeile {nummer}: Entity {roh['entity']!r} steht nicht in "
                "der Entitaetentabelle."
            )
        zeit = lies_ts(roh["ts"])
        if letzte_zeit is not None and zeit < letzte_zeit:
            raise SpurFehler(
                f"Zeile {nummer}: Zeit faellt ({roh['ts']} nach "
                f"{letzte_zeit.isoformat()}) — die Spur muss nicht "
                "absteigend sortiert sein."
            )
        letzte_zeit = zeit
        ereignisse.append(
            Ereignis(
                ts=zeit,
                entity=roh["entity"],
                zustand=str(roh["zustand"]),
                alt=roh.get("alt"),
                marke=roh.get("marke"),
            )
        )
    return Spur(kopf=kopf, ereignisse=ereignisse)


def schreibe_spur(spur: Spur, pfad) -> None:
    """Schreibt die Spur als JSONL (UTF-8, LF). Der Kopf traegt die
    Entitaetentabelle; `marke` reist nur mit, wenn sie gesetzt ist."""
    ziel = Path(pfad)
    ziel.parent.mkdir(parents=True, exist_ok=True)
    zeilen = [json.dumps(spur.kopf.als_json(), ensure_ascii=False)]
    for e in spur.ereignisse:
        zeilen.append(json.dumps(e.als_json(), ensure_ascii=False))
    ziel.write_text("\n".join(zeilen) + "\n", encoding="utf-8", newline="\n")


# ---------------------------------------------------------------------------
# Uebersicht (nur Aggregate — fuer den Waechter und die Tafel)
# ---------------------------------------------------------------------------


def uebersicht(spur: Spur) -> dict:
    """Aggregate einer Spur: Zeilen, Tage, Entities, Ereignisse je
    Kategorie, Vokabular — und die Vokabular-Luecken des Hypothalamus
    (Semantik/Zustand-Paare, die das Level-Lexikon nicht kennt; Befund
    aus der Denkrunde #3, Kontinuum-Notiz 13430)."""
    tokens = tokenisieren(spur)
    haus_typ = spur.kopf.haus_typ
    je_kategorie: Dict[str, int] = {}
    luecken_je_kategorie: Dict[str, int] = {}
    luecken_paare = set()
    tage = set()
    vokabular = set()
    for token, e in zip(tokens, spur.ereignisse):
        raum, semantik, zustand = token_zerlegen(token)
        kat = kategorie(haus_typ, semantik)
        je_kategorie[kat] = je_kategorie.get(kat, 0) + 1
        vokabular.add(token)
        tage.add(e.ts.date().isoformat())
        if semantik in ENERGY_SEMANTICS or semantik in CLIMATE_SEMANTICS:
            if zustand not in STATE_TO_LEVEL:
                luecken_je_kategorie[kat] = luecken_je_kategorie.get(kat, 0) + 1
                luecken_paare.add(f"{semantik}.{zustand}")
    marken: Dict[str, int] = {}
    for e in spur.ereignisse:
        if e.marke:
            marken[e.marke] = marken.get(e.marke, 0) + 1
    return {
        "haus": spur.kopf.haus,
        "haus_typ": haus_typ,
        "quelle": spur.kopf.quelle,
        "ereignisse": len(spur.ereignisse),
        "tage": len(tage),
        "entitaeten": len(spur.kopf.entitaeten),
        "vokabular": len(vokabular),
        "je_kategorie": dict(sorted(je_kategorie.items())),
        "hypothalamus_luecken": dict(sorted(luecken_je_kategorie.items())),
        "hypothalamus_luecken_paare": sorted(luecken_paare),
        "marken": dict(sorted(marken.items())),
    }


__all__ = [
    "SPUR_FORMAT",
    "KATEGORIEN_STAND",
    "HAUS_TYPEN",
    "QUELLEN",
    "KATEGORIEN",
    "SONSTIGES",
    "SpurFehler",
    "Entitaet",
    "Ereignis",
    "Kopf",
    "Spur",
    "lies_ts",
    "kategorie",
    "token_zerlegen",
    "registriere",
    "tokenisieren",
    "ausduennen",
    "lies_spur",
    "schreibe_spur",
    "uebersicht",
]
