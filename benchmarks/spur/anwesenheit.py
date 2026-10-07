"""Der Anwesenheits-Messstand: Wer ist da, wenn das Handy schweigt?

Protokoll (dieselbe Strenge wie der Messkern, Protokoll #2):

* **Training** (Wochen 1..k): Der Assoziationskortex sieht alle
  Gerätezustände UND die Tracker (``device_tracker.*``) als Etiketten.
  Die Tracker sind verrauscht (verspätet, WLAN-Flattern, ``unknown``) —
  wie im echten Haus. Die wahre Anwesenheit sieht niemand.
* **Test** (Woche k+1): Die Tracker werden zu Beginn auf ``unknown``
  gesetzt, ihre Meldungen verschwinden. Alle 5 Minuten Ereigniszeit sagt
  der Kortex aus der Lage, wie wahrscheinlich die Person zu Hause ist.
  Verglichen wird mit der WAHREN Anwesenheit.
* **Gegner** (lernen aus denselben Etiketten im Training):
  - ``P0`` Mehrheit: der häufigere Zustand der Trainingszeit;
  - ``P1`` Uhrzeit-Gewohnheit: P(zu Hause | Werktag/Wochenende, Stunde);
  - ``P2`` Aktivität: zu Hause, wenn im Haus in den letzten 60 Minuten
    irgendein Gerät geschaltet hat (die klassische Belegungsregel).
* **Kennzahlen** je Person: Trefferquote (Schwelle 0,5), balancierte
  Trefferquote (Mittel aus „daheim richtig“ und „weg richtig“, damit eine
  Mehrheitsregel nicht glänzt), Brier-Wert.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from kontinuum_core.association_cortex import AssociationCortex, NUMERISCHE_STUFEN
from kontinuum_core.thalamus import Thalamus

from .spur import Entitaet, Ereignis

TAKT = timedelta(minutes=5)
SYSTEME = ("Kortex", "P0", "P1", "P2")


@dataclass
class Bilanz:
    n: int = 0
    treffer: int = 0
    brier: float = 0.0
    daheim_n: int = 0
    daheim_treffer: int = 0
    weg_n: int = 0
    weg_treffer: int = 0

    def zaehle(self, p_daheim: float, wahr_daheim: bool) -> None:
        self.n += 1
        richtig = (p_daheim >= 0.5) == wahr_daheim
        self.treffer += richtig
        self.brier += (p_daheim - (1.0 if wahr_daheim else 0.0)) ** 2
        if wahr_daheim:
            self.daheim_n += 1
            self.daheim_treffer += richtig
        else:
            self.weg_n += 1
            self.weg_treffer += richtig

    @property
    def quote(self) -> float:
        return self.treffer / self.n if self.n else 0.0

    @property
    def balanciert(self) -> float:
        teile = []
        if self.daheim_n:
            teile.append(self.daheim_treffer / self.daheim_n)
        if self.weg_n:
            teile.append(self.weg_treffer / self.weg_n)
        return sum(teile) / len(teile) if teile else 0.0

    @property
    def brier_mittel(self) -> float:
        return self.brier / self.n if self.n else 0.0


@dataclass
class AnwesenheitsErgebnis:
    ursprung: int
    je_person: Dict[str, Dict[str, Bilanz]] = field(default_factory=dict)
    tracker_roh: Dict[str, Bilanz] = field(default_factory=dict)

    def zeilen(self) -> List[str]:
        aus = []
        for person, systeme in self.je_person.items():
            teile = []
            for name, b in systeme.items():
                teile.append(f"{name} {b.quote:6.1%} bal {b.balanciert:6.1%} Brier {b.brier_mittel:.3f}")
            roh = self.tracker_roh.get(person)
            zusatz = f" | Tracker selbst {roh.quote:.1%}" if roh and roh.n else ""
            aus.append(f"k={self.ursprung} {person}: " + " | ".join(teile) + zusatz)
        return aus


def _thalamus(entitaeten: Sequence[Entitaet]) -> Thalamus:
    thalamus = Thalamus()
    for e in entitaeten:
        if e.semantik:
            thalamus.custom_semantic_rules.append({
                "semantic": e.semantik,
                "entity_regex": "^" + e.id.lower().replace(".", r"\.") + "$",
            })
    for e in entitaeten:
        thalamus.register_entity(e.id, ha_area=e.raum, domain=e.domain,
                                 device_class=e.geraeteklasse or "")
    return thalamus


def zufuehren(kortex: AssociationCortex, thalamus: Thalamus, ereignis: Ereignis) -> None:
    """Ein Rohereignis in den Kortex — so, wie die Engine es tut."""
    semantik = thalamus.entity_semantic.get(ereignis.entity)
    normiert = None
    if semantik:
        try:
            normiert = thalamus._normalize_state(semantik, ereignis.zustand)
        except Exception:  # noqa: BLE001 - ein kaputter Wert ist kein Absturz
            normiert = None
    kortex.setze(ereignis.entity, ereignis.zustand, ereignis.ts,
                 normiert=normiert, numerisch=semantik in NUMERISCHE_STUFEN)


def messe_anwesenheit(
    entitaeten: Sequence[Entitaet],
    ereignisse: Sequence[Ereignis],
    ziele: Dict[str, str],
    wahrheit: Callable[[str, datetime], bool],
    train_wochen: int,
    kortex_bauer: Optional[Callable[[], AssociationCortex]] = None,
) -> AnwesenheitsErgebnis:
    """Ein rollierender Ursprung. ``ziele`` bildet Person → Tracker-Entität."""
    start = ereignisse[0].ts
    grenze = start + timedelta(days=7 * train_wochen)
    ende = grenze + timedelta(days=7)
    kortex = (kortex_bauer or AssociationCortex)()
    thalamus = _thalamus(entitaeten)
    tracker_person = {t: p for p, t in ziele.items()}

    # Gegner-Zähler (aus den Tracker-Etiketten im Training, je 5-min-Takt)
    p0 = {p: [0.0, 0.0] for p in ziele}                      # [daheim, weg]
    p1 = {p: {} for p in ziele}                              # (we, h) -> [daheim, weg]
    letzte_aktivitaet: Optional[datetime] = None
    tracker_zustand: Dict[str, str] = {}

    ergebnis = AnwesenheitsErgebnis(ursprung=train_wochen)
    for p in ziele:
        ergebnis.je_person[p] = {s: Bilanz() for s in SYSTEME}
        ergebnis.tracker_roh[p] = Bilanz()

    naechster = start.replace(second=0, microsecond=0) + TAKT
    versteckt = False

    def takt_bewerten(zeit: datetime) -> None:
        kortex.tick(zeit)
        we = 1 if zeit.weekday() >= 5 else 0
        for person, tracker in ziele.items():
            wahr = wahrheit(person, zeit)
            auskunft = kortex.anwesenheit(tracker, zeit)
            p_k = auskunft["zuhause"] if auskunft["zuhause"] is not None else 0.5
            b = ergebnis.je_person[person]
            b["Kortex"].zaehle(p_k, wahr)
            d, w = p0[person]
            b["P0"].zaehle(1.0 if d >= w else 0.0, wahr)
            zelle = p1[person].get((we, zeit.hour))
            if zelle and sum(zelle) > 0:
                b["P1"].zaehle((zelle[0] + 0.5) / (sum(zelle) + 1.0), wahr)
            else:
                b["P1"].zaehle(1.0 if d >= w else 0.0, wahr)
            aktiv = letzte_aktivitaet is not None and zeit - letzte_aktivitaet <= timedelta(minutes=60)
            b["P2"].zaehle(1.0 if aktiv else 0.0, wahr)

    def takt_lernen(zeit: datetime) -> None:
        we = 1 if zeit.weekday() >= 5 else 0
        for person, tracker in ziele.items():
            z = tracker_zustand.get(tracker)
            if z not in ("home", "not_home"):
                continue
            i = 0 if z == "home" else 1
            p0[person][i] += 1.0
            zelle = p1[person].setdefault((we, zeit.hour), [0.0, 0.0])
            zelle[i] += 1.0

    for ereignis in ereignisse:
        if ereignis.ts >= ende:
            break
        # fällige Takte VOR diesem Ereignis
        while naechster <= ereignis.ts and naechster < ende:
            if naechster >= grenze:
                if not versteckt:
                    for tracker in ziele.values():
                        kortex.setze(tracker, "unknown", grenze)
                    versteckt = True
                takt_bewerten(naechster)
                # Wie gut wäre der Tracker selbst gewesen (nur Diagnose)?
                for person, tracker in ziele.items():
                    z = tracker_zustand.get(tracker)
                    if z in ("home", "not_home"):
                        ergebnis.tracker_roh[person].zaehle(
                            1.0 if z == "home" else 0.0, wahrheit(person, naechster))
            else:
                takt_lernen(naechster)
            naechster += TAKT
        if ereignis.entity in tracker_person:
            tracker_zustand[ereignis.entity] = ereignis.zustand
            if ereignis.ts >= grenze:
                continue  # im Test schweigen die Tracker
        else:
            letzte_aktivitaet = ereignis.ts
        zufuehren(kortex, thalamus, ereignis)
    return ergebnis


__all__ = ["Bilanz", "AnwesenheitsErgebnis", "messe_anwesenheit", "zufuehren", "SYSTEME"]
