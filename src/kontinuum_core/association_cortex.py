"""Assoziationskortex — das Lagebild: alle Zustände gegen alle.

Biologisches Vorbild: Die Assoziationsrinde verknüpft, was mehrere Sinne
gleichzeitig melden, zu einer Lage. „Auto weg + Fernseher aus + PC an“
sagt kein einzelner Sensor, erst die Kombination.

Was der Kortex zählt (alles inkrementell, gedeckelt, Pi-tauglich):

1. **Gerätezustände im Takt (alle 5 Minuten Ereigniszeit).** Zu jedem Takt
   steht jede beobachtete Entität in genau einem Zustand, und
   „unavailable“ ist ein Zustand (``weg``): Reifendrucksensoren, die mit dem
   Auto wegfahren, melden genau das. Der Thalamus wirft solche Ereignisse
   weg, der Kortex nicht.

2. **Paar-Tafel „alle gegen alle“.** Für jedes Paar von Merkmalen
   (Entität = Zustand) zählt er die Takte, in denen beide galten. Daraus
   P(B | A) und der Lift P(B | A) / P(B): „Wenn der PC an ist, ist der
   Fernseher zu 95 % aus.“ Paarweise, mit Absicht: Die volle Tafel aller
   Kombinationen wächst mit 2^n (40 Geräte ergeben 10^12 Kombinationen) und
   passt auf keinen Rechner. Höhere Kombinationen setzt der Bayes-Schluss
   unten aus vielen Paaren zusammen.

3. **Anwesenheit (Ziele).** ``person.*`` und ``device_tracker.*`` sind Ziele.
   Ist ihr Zustand bekannt, lernt der Kortex, welche Lage dazu passt
   (Naive Bayes über alle Merkmale samt Dauer im Zustand und Tageszeit).
   Ist er unbekannt (Handy aus, Tracker „unknown“), schließt er aus der Lage
   zurück und nennt seine Belege. Das ist der Bayes-Sensor, den man in Home
   Assistant sonst von Hand mit Wahrscheinlichkeiten füttern muss: Hier
   lernt er sich selbst.

4. **Was folgt in welcher Lage (Ereignis-Assoziation).** Für jedes Merkmal,
   welches Ereignis als nächstes kam. Das ist der Lage-Experte des Claustrums:
   Ein Licht, das gerade an ist, geht als nächstes aus, nicht an. Ein
   Bewegungsmelder, der seit zwei Minuten an ist, fällt bald ab. Eine reine
   Ereigniskette weiß das nicht, das Lagebild schon.
"""

from __future__ import annotations

import heapq
import math
from array import array
from collections import deque
from typing import Any, Callable, Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

#: Dauer im aktuellen Zustand, Grenzen in Sekunden: <2 min, <15 min, <1 h, <4 h, länger.
DAUER_GRENZEN = (120.0, 900.0, 3600.0, 14400.0)

#: Zustände, die „Gerät ist nicht erreichbar“ heißen.
WEG = "weg"
UNBEKANNT_ZUSTAENDE = ("unknown", "unbekannt", "none", "")
WEG_ZUSTAENDE = ("unavailable",)

#: Bekannte Zustände eines Ziels, die „zu Hause“ heißen.
ZUHAUSE = ("home", "zu hause", "on")


def dauer_eimer(sekunden: float) -> int:
    for i, grenze in enumerate(DAUER_GRENZEN):
        if sekunden < grenze:
            return i
    return len(DAUER_GRENZEN)


def ist_ziel(entity_id: str) -> bool:
    """Ziele sind Personen und ihre Tracker."""
    return entity_id.startswith("person.") or entity_id.startswith("device_tracker.")


def _versatz(zeit) -> Optional[float]:
    try:
        abstand = zeit.utcoffset()
    except (AttributeError, TypeError, ValueError):
        return None
    return abstand.total_seconds() if abstand is not None else None


def _ts(zeit) -> Optional[float]:
    if zeit is None:
        return None
    if isinstance(zeit, (int, float)):
        return float(zeit)
    try:
        return zeit.timestamp()
    except (AttributeError, TypeError, ValueError, OverflowError, OSError):
        return None


#: Semantiken, deren Messwert der Kortex in eigene Gerätestufen teilt.
NUMERISCHE_STUFEN = frozenset({"power", "solar", "grid", "current"})


class Stufen:
    """Lernt die Stufen eines Messwerts aus seiner eigenen Verteilung.

    Der Thalamus eimert Leistung fest (<100 W niedrig, <1000 W mittel).
    Ein PC mit 3 W Standby und 100 W Betrieb, ein Fernseher mit 1 W und
    80 W: beides „niedrig“, an und aus sind unsichtbar. Hier entscheidet
    die Verteilung des Geräts selbst: ein Histogramm über log2(1 + |W|)
    in halben Schritten, und jedes Tal zwischen zwei besetzten Gruppen
    ist eine Stufengrenze. Die Grenzen werden nur in Abständen neu
    bestimmt (20, 50, 100, 200, … Werte), damit „s1“ nicht jeden Tag
    etwas anderes heißt.
    """

    BREITE = 0.5
    MAX_STUFEN = 4

    def __init__(self):
        self.hist: Dict[int, float] = {}
        self.n = 0
        self.grenzen: List[int] = []
        self._naechste_pruefung = 20

    def stufe(self, wert: float) -> str:
        eimer = int(math.log2(1.0 + abs(wert)) / self.BREITE)
        self.hist[eimer] = self.hist.get(eimer, 0.0) + 1.0
        self.n += 1
        if self.n >= self._naechste_pruefung:
            self._grenzen_neu()
            self._naechste_pruefung = int(self._naechste_pruefung * 2.5)
            if self.n > 20000:
                for k in list(self.hist):
                    self.hist[k] *= 0.5
        stufe = 0
        for g in self.grenzen:
            if eimer >= g:
                stufe += 1
        return f"s{stufe}"

    def _grenzen_neu(self) -> None:
        if not self.hist:
            return
        oben = max(self.hist.values())
        schwelle = oben * 0.02
        grenzen = []
        in_gruppe = False
        for eimer in range(min(self.hist), max(self.hist) + 1):
            besetzt = self.hist.get(eimer, 0.0) > schwelle
            if besetzt and not in_gruppe and eimer != min(self.hist):
                grenzen.append(eimer)
            in_gruppe = besetzt
        self.grenzen = grenzen[: self.MAX_STUFEN - 1]

    def to_dict(self) -> Dict[str, Any]:
        return {"hist": [[k, v] for k, v in self.hist.items()], "n": self.n,
                "grenzen": self.grenzen, "pruefung": self._naechste_pruefung}

    def from_dict(self, daten: Dict[str, Any]) -> None:
        self.hist = {int(k): float(v) for k, v in daten.get("hist", [])}
        self.n = int(daten.get("n", 0))
        self.grenzen = [int(g) for g in daten.get("grenzen", [])]
        self._naechste_pruefung = int(daten.get("pruefung", 20))


class AssociationCortex:
    """Lagebild, Paar-Tafel, Anwesenheit und Lage-Experte in einem Modul."""

    #: Takt der Zustandsaufnahme in Sekunden Ereigniszeit.
    TAKT_S = 300.0
    #: So viele Takte holt ein Ereignis nach einer Pause höchstens nach (1 Tag).
    MAX_NACHHOLEN = 288
    #: Entitäten im Blick (Merkmale). Ziele kommen immer dazu.
    MAX_ENTITAETEN = 96
    #: Entitäten in der Paar-Tafel (quadratischer Speicher).
    MAX_PAAR_ENTITAETEN = 48
    MAX_PAAR_MERKMALE = 192
    #: Glättung der Ereignis-Assoziation (Stärke des Vorwissens P(y)).
    BETA = 2.0
    #: Ein Merkmal merkt sich höchstens so viele Folge-Ereignisse.
    MAX_FOLGEN = 64
    FOLGE_DECKEL = 4000.0
    #: Glättung der Anwesenheits-Tafeln, in Takten: Fünf-Minuten-Takte
    #: derselben Episode sind keine unabhängigen Zeugen; zwölf Takte (eine
    #: Stunde) Vorwissen halten einen einzelnen Tag davon ab, alles zu
    #: entscheiden.
    ALPHA = 12.0
    #: Merkmalsfamilien der Anwesenheit (Abschaltprobe im Messstand)
    MERKMAL_DAUER = True
    MERKMAL_ABSCHNITT = True

    def __init__(self, dauer_in_folge: bool = True, paar_tafel: bool = True,
                 ziel: Optional[Callable[[str], bool]] = None):
        self.dauer_in_folge = dauer_in_folge
        self.paar_tafel = paar_tafel
        #: Wer ist Ziel der Anwesenheit? Standard: ``person.*`` und jeder
        #: ``device_tracker.*``. Ein Wirt mit vielen Trackern (Router legen
        #: einen je Gerät im Netz an) gibt hier nur Personen und ihre eigenen
        #: Tracker an — der Tracker des PCs ist dann ein Indiz der Lage,
        #: kein Ziel.
        self.ziel_pruefer: Callable[[str], bool] = ziel or ist_ziel
        # -- Lage ------------------------------------------------------
        self.zustand: Dict[str, str] = {}
        self.seit: Dict[str, float] = {}
        self.stufen: Dict[str, Stufen] = {}
        #: letzte Zustandsänderung irgendeiner Nicht-Ziel-Entität (Stille)
        self.letzte_aktivitaet: Optional[float] = None
        self.wechsel: Dict[str, float] = {}
        self.im_blick: List[str] = []
        self._im_blick_set = set()
        self.ziele: List[str] = []
        self.jetzt: Optional[float] = None
        #: Abstand der Ortszeit zu UTC (Sekunden), aus dem letzten Zeitstempel
        self.versatz = 0.0
        self._naechster_takt: Optional[float] = None
        self.takte = 0
        self.ereignisse = 0
        # -- Paar-Tafel ------------------------------------------------
        self.paar_index: Dict[Tuple[str, str], int] = {}
        self.paar_merkmale: List[Tuple[str, str]] = []
        self.paar = array("d", [0.0]) * (self.MAX_PAAR_MERKMALE * self.MAX_PAAR_MERKMALE)
        self.paar_takte = 0.0
        self._paar_offen = 0
        # -- Anwesenheit je Ziel: [weg, daheim] ---------------------------
        self.anw_n: Dict[str, List[float]] = {}
        self.anw: Dict[str, List[Dict[str, float]]] = {}
        #: Uhrzeit-Gewohnheit je Ziel: "Wochenende:Stunde" -> [weg, daheim]
        self.anw_zeit: Dict[str, Dict[str, List[float]]] = {}
        #: gelernte Mischung je Ziel: [Achse, Uhrzeit] plus je Entität ein Gewicht
        self.anw_w: Dict[str, List[float]] = {}
        self._anw_g2: Dict[str, List[float]] = {}
        self.anw_v: Dict[str, Dict[str, float]] = {}
        self._anw_v_g2: Dict[str, Dict[str, float]] = {}
        #: Takte warten einen Tag, bevor sie in die Zählung gehen (s. _takt)
        self._anw_warteschlange: Dict[str, Any] = {}
        self.anw_bilanz: Dict[str, List[float]] = {}  # [n, brier_summe, treffer]
        # -- Ereignis-Assoziation (Lage-Experte) ------------------------
        self.folge: Dict[Hashable, Dict[Hashable, float]] = {}
        self.folge_n: Dict[Hashable, float] = {}
        #: je Merkmal die drei häufigsten Folgen, inkrementell gepflegt
        self.folge_top: Dict[Hashable, List[Hashable]] = {}
        self.ziel_n: Dict[Hashable, float] = {}
        self.ziel_summe = 0.0
        self.vertrauen: Dict[str, float] = {}
        self._vertrauen_g2: Dict[str, float] = {}
        self._schnapp: List[Tuple[str, Hashable]] = []
        self._kandidaten: List[Hashable] = []
        self._boni: List[Tuple[str, Dict[int, float]]] = []
        self._q: List[float] = []

    # ==================================================================
    # Lage pflegen
    # ==================================================================
    def setze(self, entity_id: str, zustand: Optional[str], zeit=None,
              normiert: Optional[str] = None, numerisch: bool = False) -> None:
        """Ein Zustandswechsel. ``None``/unknown heißt „unbekannt“,
        unavailable heißt ``weg``. Vorher werden verpasste Takte nachgeholt,
        denn bis eben galt noch die alte Lage.

        ``normiert``: der Zustand, wie der Thalamus ihn liest (z. B. ein
        Temperatur-Eimer); ``numerisch``: der Rohwert ist eine Leistung o. Ä.
        und bekommt eigene Gerätestufen (``Stufen``)."""
        ts = _ts(zeit)
        if ts is None:
            ts = self.jetzt if self.jetzt is not None else 0.0
        versatz = _versatz(zeit)
        if versatz is not None:
            self.versatz = versatz
        self._takte_bis(ts)
        self.jetzt = ts if self.jetzt is None else max(self.jetzt, ts)
        self.ereignisse += 1
        neu = self._normalisiere(zustand)
        if neu not in (WEG, "unbekannt"):
            if numerisch:
                try:
                    wert = float(str(zustand).strip())
                except (TypeError, ValueError):
                    wert = None
                if wert is not None and math.isfinite(wert):
                    stufen = self.stufen.get(entity_id)
                    if stufen is None:
                        stufen = Stufen()
                        self.stufen[entity_id] = stufen
                    neu = stufen.stufe(wert)
                elif normiert:
                    neu = normiert
            elif normiert:
                neu = normiert
        alt = self.zustand.get(entity_id)
        if alt == neu:
            return
        self.zustand[entity_id] = neu
        self.seit[entity_id] = ts
        ziel = self.ziel_pruefer(entity_id)
        if not ziel:
            self.letzte_aktivitaet = ts
        self.wechsel[entity_id] = self.wechsel.get(entity_id, 0.0) + 1.0
        # Die Prüfung des Wirts darf sich ändern (in Home Assistant: wem
        # gehört welcher Tracker) — Ziel wird, wer es JETZT ist, auch wenn
        # er schon als Merkmal im Blick war; wer es nicht mehr ist, geht.
        if ziel:
            if entity_id not in self.ziele:
                self.ziele.append(entity_id)
            if entity_id not in self._im_blick_set:
                self._blick_dazu(entity_id)
        else:
            if entity_id in self.ziele:
                self.ziele.remove(entity_id)
            if (entity_id not in self._im_blick_set
                    and len(self.im_blick) < self.MAX_ENTITAETEN):
                self._blick_dazu(entity_id)
        if self.ereignisse % 2000 == 0:
            self._blick_pflegen()

    @staticmethod
    def _normalisiere(zustand: Optional[str]) -> str:
        if zustand is None:
            return "unbekannt"
        text = str(zustand).strip().lower()
        if text in WEG_ZUSTAENDE:
            return WEG
        if text in UNBEKANNT_ZUSTAENDE:
            return "unbekannt"
        if text == "not_home" or text == "away":
            return "not_home"
        return text[:40]

    def _blick_dazu(self, entity_id: str) -> None:
        self.im_blick.append(entity_id)
        self._im_blick_set.add(entity_id)

    def _blick_pflegen(self) -> None:
        """Bei sehr vielen Entitäten: die mit den meisten Wechseln bleiben im
        Blick (Ziele immer), Zähler halbieren, damit alte Ruhe vergeht."""
        if len(self.wechsel) <= self.MAX_ENTITAETEN:
            for e in list(self.wechsel):
                self.wechsel[e] *= 0.5
            return
        kandidaten = [e for e in self.wechsel if not self.ziel_pruefer(e)]
        beste = heapq.nlargest(self.MAX_ENTITAETEN, kandidaten, key=self.wechsel.__getitem__)
        neu = list(self.ziele) + [e for e in beste if e not in self.ziele]
        self.im_blick = neu
        self._im_blick_set = set(neu)
        for e in list(self.wechsel):
            self.wechsel[e] *= 0.5

    # ==================================================================
    # Takt: Paar-Tafel und Anwesenheit
    # ==================================================================
    def _takte_bis(self, ts: float) -> None:
        if self._naechster_takt is None:
            self._naechster_takt = (math.floor(ts / self.TAKT_S) + 1) * self.TAKT_S
            return
        nachgeholt = 0
        while self._naechster_takt <= ts:
            if nachgeholt < self.MAX_NACHHOLEN:
                self._takt(self._naechster_takt)
                nachgeholt += 1
            self._naechster_takt += self.TAKT_S
            if nachgeholt >= self.MAX_NACHHOLEN and self._naechster_takt <= ts:
                # Lange Pause: der Rest des Lochs zählt nicht, sonst lernte
                # ein abgeschalteter Server "alles aus" als Normalfall.
                self._naechster_takt = (math.floor(ts / self.TAKT_S) + 1) * self.TAKT_S
                break
        # Zwischen zwei Ereignissen ändert sich kein Zustand: alle eben
        # nachgeholten Takte sahen dieselbe Lage. Die Paar-Tafel bekommt sie
        # deshalb in EINEM Schritt (Anzahl × Lage), nicht Takt für Takt —
        # höchstens einmal je Ereignis und höchstens einmal je Takt.
        self._paar_nachtragen()

    def tick(self, zeit) -> None:
        """Herzschlag ohne Ereignis (Host-Timer): holt fällige Takte nach."""
        ts = _ts(zeit)
        if ts is not None:
            self._takte_bis(ts)
            self.jetzt = ts if self.jetzt is None else max(self.jetzt, ts)

    def _merkmale(self, t: float, ohne: Optional[str] = None) -> List[str]:
        """Merkmale zum Zeitpunkt t: Entität=Zustand@Dauer, dasselbe noch
        einmal je Tagesabschnitt (``§``), plus Tageszeit und Stille.

        Die Abschnitts-Kopie ist die eine Wechselwirkung, die Naive Bayes
        sonst nicht sieht: „Auto steht da“ heißt abends „alle daheim“,
        werktags um zehn aber nur „wer weg ist, fuhr Bus“.

        Ziele (Personen, ihre Tracker) sind nie Merkmal: Der Tracker einer
        Person ist die Quelle ihres Etiketts — als Indiz spräche das
        Lagebild dem Handy nur nach, statt ihm zu widersprechen, wenn es im
        Büro liegt. Geschlossen wird allein aus den Geräten."""
        aus = []
        lokal = t + self.versatz
        stunde = int((lokal % 86400.0) // 3600.0)
        wochenende = 1 if int((lokal // 86400.0 + 3) % 7) >= 5 else 0
        abschnitt = f"{wochenende}:{stunde // 3}"
        ist_ziel_ = self.ziel_pruefer
        for e in self.im_blick:
            if e == ohne or ist_ziel_(e):
                continue
            z = self.zustand.get(e)
            if z is None or z == "unbekannt":
                continue
            aus.append(f"{e}={z}")
            if self.MERKMAL_DAUER:
                d = dauer_eimer(t - self.seit.get(e, t))
                aus.append(f"{e}={z}@{d}")
            if self.MERKMAL_ABSCHNITT:
                aus.append(f"{e}={z}§{abschnitt}")
        aus.append(f"zeit:{abschnitt}")
        if self.letzte_aktivitaet is not None:
            aus.append(f"stille@{dauer_eimer(t - self.letzte_aktivitaet)}")
        return aus

    def _takt(self, t: float) -> None:
        self.takte += 1
        if self.paar_tafel:
            self._paar_offen += 1
        for ziel in self.ziele:
            zustand = self.zustand.get(ziel)
            if zustand is None or zustand == "unbekannt" or zustand == WEG:
                continue
            daheim = 1 if zustand in ZUHAUSE else 0
            merkmale = self._merkmale(t, ohne=ziel)
            self._anwesenheit_pruefen(ziel, daheim, t, merkmale)
            # Erst nach einem Tag in die Zählung: Fünf-Minuten-Takte derselben
            # Episode gleichen sich fast vollständig. Zählte der Takt sofort,
            # sähe die Prüfung oben immer ihre eigene Episode und hielte die
            # Lage für unschlagbar — die Mischung lernte dann, der Uhrzeit
            # nicht mehr zu trauen. Mit einem Tag Abstand prüft sie ehrlich,
            # über Tage hinweg.
            schlange = self._anw_warteschlange.setdefault(ziel, deque())
            schlange.append((t, daheim, merkmale))
            while schlange and schlange[0][0] <= t - self.ANW_VERZUG_S:
                t_alt, daheim_alt, merkmale_alt = schlange.popleft()
                self._anwesenheit_zaehlen(ziel, t_alt, daheim_alt, merkmale_alt)

    #: Verzögerung der Anwesenheits-Zählung (Sekunden Ereigniszeit).
    ANW_VERZUG_S = 86400.0

    def _anwesenheit_zaehlen(self, ziel: str, t: float, daheim: int,
                             merkmale: Sequence[str]) -> None:
        n = self.anw_n.setdefault(ziel, [0.0, 0.0])
        n[daheim] += 1.0
        tafeln = self.anw.setdefault(ziel, [{}, {}])
        tafel = tafeln[daheim]
        for m in merkmale:
            tafel[m] = tafel.get(m, 0.0) + 1.0
        slot = self._slot(t)
        zelle = self.anw_zeit.setdefault(ziel, {}).setdefault(slot, [0.0, 0.0])
        zelle[daheim] += 1.0
        if n[0] + n[1] > 50000:
            for i in (0, 1):
                n[i] *= 0.5
                for m in list(tafeln[i]):
                    tafeln[i][m] *= 0.5
            for z in self.anw_zeit[ziel].values():
                z[0] *= 0.5
                z[1] *= 0.5

    def _paar_nachtragen(self) -> None:
        if self._paar_offen <= 0:
            return
        anzahl = float(self._paar_offen)
        self._paar_offen = 0
        self._paar_takt(anzahl)

    def _paar_takt(self, anzahl: float = 1.0) -> None:
        idx = []
        for e in self.im_blick[: self.MAX_PAAR_ENTITAETEN]:
            z = self.zustand.get(e)
            if z is None:
                continue
            schluessel = (e, z)
            i = self.paar_index.get(schluessel)
            if i is None:
                if len(self.paar_merkmale) >= self.MAX_PAAR_MERKMALE:
                    continue
                i = len(self.paar_merkmale)
                self.paar_index[schluessel] = i
                self.paar_merkmale.append(schluessel)
            idx.append(i)
        breite = self.MAX_PAAR_MERKMALE
        tafel = self.paar
        for i in idx:
            zeile = i * breite
            for j in idx:
                tafel[zeile + j] += anzahl
        self.paar_takte += anzahl
        if self.paar_takte > 200000.0:
            for k in range(len(tafel)):
                tafel[k] *= 0.5
            self.paar_takte *= 0.5

    # ==================================================================
    # Anwesenheit
    # ==================================================================
    def _slot(self, t: float) -> str:
        lokal = t + self.versatz
        stunde = int((lokal % 86400.0) // 3600.0)
        wochenende = 1 if int((lokal // 86400.0 + 3) % 7) >= 5 else 0
        return f"{wochenende}:{stunde}"

    @staticmethod
    def _quelle(merkmal: str) -> str:
        """Zu welcher Entität (oder Pseudo-Entität) gehört ein Merkmal?"""
        if merkmal.startswith("zeit:"):
            return "zeit"
        if merkmal.startswith("stille@"):
            return "stille"
        if "§" in merkmal:
            return merkmal.split("=", 1)[0] + "§"
        if "@" in merkmal:
            return merkmal.split("=", 1)[0] + "@"
        return merkmal.split("=", 1)[0]

    def _lage_beitraege(self, ziel: str, merkmale: Sequence[str]) -> List[Tuple[str, str, float]]:
        """Je Merkmal der Bayes-Beitrag log P(m | daheim) / P(m | weg) —
        positiv spricht für daheim. Rückgabe: (Quelle, Merkmal, Beitrag)."""
        n = self.anw_n.get(ziel)
        if not n or n[0] + n[1] <= 0:
            return []
        weg, daheim = self.anw[ziel]
        a = self.ALPHA
        aus = []
        for m in merkmale:
            beitrag = math.log(((daheim.get(m, 0.0) + a) / (n[1] + 2 * a))
                               / ((weg.get(m, 0.0) + a) / (n[0] + 2 * a)))
            aus.append((self._quelle(m), m, beitrag))
        return aus

    def _zeit_logit(self, ziel: str, t: float) -> float:
        zelle = self.anw_zeit.get(ziel, {}).get(self._slot(t))
        if not zelle:
            n = self.anw_n.get(ziel, [0.0, 0.0])
            return math.log((n[1] + 0.5) / (n[0] + 0.5))
        return math.log((zelle[1] + 0.5) / (zelle[0] + 0.5))

    #: Startgewichte: Der Kortex beginnt als reine Uhrzeit-Gewohnheit
    #: (Gewicht 1) und gibt den Geräten erst Gewicht, wenn sie sich in der
    #: ehrlichen Prüfung bewähren — wie das Claustrum, das mit der Kette
    #: beginnt. So ist er von Anfang an nie schlechter als die Gewohnheit.
    #: Das gelernte Gewicht je Entität holt außerdem zurück, was Naive Bayes
    #: bei korrelierten Zeugen (vier Reifen eines Autos) vierfach zählt.
    V_START = 0.0
    W_ZEIT_START = 1.0
    #: Lernrate der Mischung (AdaGrad). Gemessen am 06.10.2026 auf Aruba,
    #: Tulum1 und dem simulierten Haushalt: 0,1 schlägt 0,3 und 1,0.
    ANW_RATE = 0.1

    def _p_daheim(self, ziel: str, t: float, merkmale: Sequence[str]):
        beitraege = self._lage_beitraege(ziel, merkmale)
        l_zeit = self._zeit_logit(ziel, t)
        w = self.anw_w.get(ziel, [0.0, self.W_ZEIT_START])
        v = self.anw_v.get(ziel, {})
        logit = w[0] + w[1] * l_zeit
        for quelle, _, b in beitraege:
            logit += v.get(quelle, self.V_START) * b
        logit = max(-30.0, min(30.0, logit))
        return 1.0 / (1.0 + math.exp(-logit)), l_zeit, beitraege, w, v

    def _anwesenheit_pruefen(self, ziel: str, daheim: int, t: float,
                             merkmale: Sequence[str]) -> None:
        """Vor dem Lernen: Wie gut hätte die Lage das Ziel erraten? Daraus
        lernt die logistische Mischung — je Person und je Entität: Wer feste
        Arbeitszeiten hat, verrät sich über die Uhr; wer zu Hause am PC
        arbeitet, über die Geräte; und vier Reifen eines Autos sind
        zusammen nur ein Zeuge."""
        n = self.anw_n.get(ziel)
        if not n or n[0] + n[1] < 50:
            return
        p, l_zeit, beitraege, w, v = self._p_daheim(ziel, t, merkmale)
        fehler = daheim - p
        # AdaGrad: Die Schritte schrumpfen mit der gesammelten Erfahrung.
        # RMSprop machte auch bei reinem Rauschen volle Schritte, und die
        # Gewichte wanderten über Wochen zufällig (gemessen: Tulum, 4 Wochen).
        rate = self.ANW_RATE
        g2 = self._anw_g2.setdefault(ziel, [0.0, 0.0])
        neu = list(w)
        for i, x in enumerate((1.0, l_zeit)):
            g = fehler * x
            g2[i] += g * g
            neu[i] = w[i] + rate * g / (math.sqrt(g2[i]) + 1e-6)
        neu[0] = max(-5.0, min(5.0, neu[0]))
        neu[1] = max(0.0, min(3.0, neu[1]))
        self.anw_w[ziel] = neu
        v_neu = self.anw_v.setdefault(ziel, {})
        vg2 = self._anw_v_g2.setdefault(ziel, {})
        for quelle, _, b in beitraege:
            g = fehler * b
            vg2[quelle] = vg2.get(quelle, 0.0) + g * g
            alt_v = v_neu.get(quelle, self.V_START)
            v_neu[quelle] = max(0.0, min(2.0, alt_v + rate * g / (math.sqrt(vg2[quelle]) + 1e-6)))
        bilanz = self.anw_bilanz.setdefault(ziel, [0.0, 0.0, 0.0])
        bilanz[0] += 1.0
        bilanz[1] += (p - daheim) ** 2
        bilanz[2] += 1.0 if (p >= 0.5) == bool(daheim) else 0.0

    def anwesenheit(self, ziel: str, zeit=None) -> Dict[str, Any]:
        """Was die Lage über das Ziel sagt — auch (gerade) wenn es unbekannt ist.

        Rückgabe: ``zuhause`` (Wahrscheinlichkeit), ``belege`` (die stärksten
        Merkmale als (Merkmal, Beitrag), positiv spricht für daheim, schon mit
        dem gelernten Gewicht der Lage verrechnet), ``uhrzeit`` (Beitrag der
        Gewohnheit), ``gewichte`` und ``takte`` (Lernumfang)."""
        t = _ts(zeit)
        if t is None:
            t = self.jetzt if self.jetzt is not None else 0.0
        n = self.anw_n.get(ziel)
        if not n or n[0] + n[1] <= 0:
            return {"zuhause": None, "belege": [], "takte": 0, "uhrzeit": None,
                    "gewichte": None, "wahrscheinlichster": None, "verteilung": {}}
        merkmale = self._merkmale(t, ohne=ziel)
        p, l_zeit, beitraege, w, v = self._p_daheim(ziel, t, merkmale)
        gewichtet = [(m, v.get(q, self.V_START) * b) for q, m, b in beitraege]
        stark = sorted(gewichtet, key=lambda x: -abs(x[1]))[:6]
        return {
            "zuhause": p,
            "wahrscheinlichster": "home" if p >= 0.5 else "not_home",
            "verteilung": {"home": p, "not_home": 1.0 - p},
            "belege": [(m, round(b, 2)) for m, b in stark],
            "uhrzeit": round(w[1] * l_zeit, 2),
            "gewichte": {"achse": round(w[0], 3), "uhrzeit": round(w[1], 3),
                         "entitaeten": {q: round(x, 3) for q, x in sorted(v.items())}},
            "takte": int(n[0] + n[1]),
        }

    # ==================================================================
    # Paar-Tafel lesen: Zusammenhänge
    # ==================================================================
    def zusammenhaenge(self, entity_id: Optional[str] = None, top: int = 10,
                       min_takte: float = 12.0) -> List[Dict[str, Any]]:
        """Die stärksten Zusammenhänge (Lift) der Paar-Tafel.

        Jeder Eintrag: wenn ``wenn`` gilt, gilt ``dann`` mit Wahrscheinlichkeit
        ``p`` — und das ist ``lift``-mal so oft wie sonst."""
        self._paar_nachtragen()
        if self.paar_takte <= 0:
            return []
        breite = self.MAX_PAAR_MERKMALE
        n = len(self.paar_merkmale)
        diag = [self.paar[i * breite + i] for i in range(n)]
        aus = []
        for i in range(n):
            e_i, z_i = self.paar_merkmale[i]
            if entity_id is not None and e_i != entity_id:
                continue
            if diag[i] < min_takte:
                continue
            for j in range(n):
                e_j, z_j = self.paar_merkmale[j]
                if e_j == e_i or diag[j] <= 0:
                    continue
                gemeinsam = self.paar[i * breite + j]
                if gemeinsam < min_takte:
                    continue
                p = gemeinsam / diag[i]
                lift = p / (diag[j] / self.paar_takte)
                aus.append({
                    "wenn": f"{e_i}={z_i}", "dann": f"{e_j}={z_j}",
                    "p": round(p, 3), "lift": round(lift, 2),
                    "takte": int(gemeinsam),
                })
        aus.sort(key=lambda d: (-(d["lift"] * min(1.0, d["p"] * 2)), d["wenn"], d["dann"]))
        return aus[:top]

    def p_gemeinsam(self, a: Tuple[str, str], b: Tuple[str, str]) -> Optional[float]:
        """P(b | a) aus der Paar-Tafel, None wenn eines der Merkmale fehlt."""
        self._paar_nachtragen()
        i = self.paar_index.get(a)
        j = self.paar_index.get(b)
        if i is None or j is None:
            return None
        breite = self.MAX_PAAR_MERKMALE
        basis = self.paar[i * breite + i]
        if basis <= 0:
            return None
        return self.paar[i * breite + j] / basis

    # ==================================================================
    # Lage-Experte für das Claustrum
    # ==================================================================
    def _folge_merkmal(self, e: str, t: float) -> Optional[Hashable]:
        z = self.zustand.get(e)
        if z is None:
            return None
        if self.dauer_in_folge:
            return (e, z, dauer_eimer(t - self.seit.get(e, t)))
        return (e, z)

    def bereite_folge(self, zeit=None) -> None:
        """Friert die Merkmale des Vorhersagemoments ein."""
        t = _ts(zeit)
        if t is None:
            t = self.jetzt if self.jetzt is not None else 0.0
        schnapp = []
        for e in self.im_blick:
            if self.ziel_pruefer(e):
                continue
            m = self._folge_merkmal(e, t)
            if m is not None:
                schnapp.append((e, m))
        self._schnapp = schnapp

    def _p_ziel(self, y: Hashable) -> float:
        vokabular = len(self.ziel_n) + 1
        return (self.ziel_n.get(y, 0.0) + 0.5) / (self.ziel_summe + 0.5 * vokabular)

    def kandidaten_folge(self, k: int = 10) -> List[Hashable]:
        """Die Ereignisse, die in DIESER Lage am stärksten über ihrer
        Grundhäufigkeit liegen (je Merkmal das Beste, dann gerankt)."""
        punkte: Dict[Hashable, float] = {}
        for e, m in self._schnapp:
            z = self.folge.get(m)
            if not z:
                continue
            n = self.folge_n.get(m, 0.0)
            if n < 3:
                continue
            for y in self.folge_top.get(m) or heapq.nlargest(3, z, key=z.__getitem__):
                lift = (z[y] / n) / self._p_ziel(y)
                if lift > punkte.get(y, 0.0):
                    punkte[y] = lift
        return heapq.nlargest(k, punkte, key=punkte.__getitem__)

    def lifts_folge(self, kandidaten: Sequence[Hashable]) -> List[float]:
        """log P(y) + Σ_e v_e · Bonus_e(y) je Kandidat (bis auf eine
        Konstante, die die Mischung ohnehin kürzt)."""
        index = {y: i for i, y in enumerate(kandidaten)}
        p_ziel = [self._p_ziel(y) for y in kandidaten]
        werte = [math.log(p) for p in p_ziel]
        # Die heißeste Schleife der Engine (Merkmale × Kandidaten): BETA·P(y)
        # einmal je Kandidat statt je Paar, Nachschlagen über lokale Namen.
        # Rechenweg und Reihenfolge wie zuvor, also bitgleiche Werte.
        nenner = [self.BETA * p for p in p_ziel]
        log1p = math.log1p
        folge = self.folge
        vertrauen = self.vertrauen
        index_get = index.get
        n_index = len(index)
        boni: List[Tuple[str, Dict[int, float]]] = []
        for e, m in self._schnapp:
            z = folge.get(m)
            if not z:
                continue
            if len(z) <= n_index:
                eintraege = {i: log1p(c / nenner[i]) for y, c in z.items()
                             if (i := index_get(y)) is not None}
            else:
                eintraege = {i: log1p(c / nenner[i]) for y, i in index.items()
                             if (c := z.get(y))}
            if not eintraege:
                continue
            v = vertrauen.get(e, 0.5)
            for i, b in eintraege.items():
                werte[i] += v * b
            boni.append((e, eintraege))
        # eigene Verteilung merken, damit das Vertrauen lernen kann
        oben = max(werte)
        exps = [math.exp(w - oben) for w in werte]
        summe = sum(exps)
        self._q = [x / summe for x in exps]
        self._kandidaten = list(kandidaten)
        self._boni = boni
        return werte

    def lerne_folge(self, y: Hashable) -> None:
        """Das Ereignis ``y`` ist eingetreten: Vertrauen je Entität nachführen
        (Gradient der eigenen Log-Likelihood), dann die Zählung."""
        if self._boni and self._kandidaten:
            try:
                ziel = self._kandidaten.index(y)
            except ValueError:
                ziel = -1
            if ziel >= 0:
                q = self._q
                for e, eintraege in self._boni:
                    erwartung = sum([q[i] * b for i, b in eintraege.items()])
                    grad = eintraege.get(ziel, 0.0) - erwartung
                    g2 = 0.999 * self._vertrauen_g2.get(e, 0.0) + 0.001 * grad * grad
                    self._vertrauen_g2[e] = g2
                    v = self.vertrauen.get(e, 0.5) + 0.01 * grad / (math.sqrt(g2) + 1e-6)
                    self.vertrauen[e] = max(0.0, min(3.0, v))
        for e, m in self._schnapp:
            z = self.folge.get(m)
            if z is None:
                z = {}
                self.folge[m] = z
            z[y] = z.get(y, 0.0) + 1.0
            top = self.folge_top.get(m)
            if top is None:
                top = []
                self.folge_top[m] = top
            if y in top:
                top.sort(key=z.__getitem__, reverse=True)
            elif len(top) < 3:
                top.append(y)
                top.sort(key=z.__getitem__, reverse=True)
            elif z[y] > z[top[-1]]:
                top[-1] = y
                top.sort(key=z.__getitem__, reverse=True)
            n = self.folge_n.get(m, 0.0) + 1.0
            if n > self.FOLGE_DECKEL or len(z) > 2 * self.MAX_FOLGEN:
                if len(z) > self.MAX_FOLGEN:
                    behalten = set(heapq.nlargest(self.MAX_FOLGEN, z, key=z.__getitem__))
                    for k in list(z):
                        if k not in behalten:
                            del z[k]
                if n > self.FOLGE_DECKEL:
                    for k in list(z):
                        z[k] *= 0.5
                n = sum(z.values())
                self.folge_top[m] = heapq.nlargest(3, z, key=z.__getitem__)
            self.folge_n[m] = n
        self.ziel_n[y] = self.ziel_n.get(y, 0.0) + 1.0
        self.ziel_summe += 1.0
        if self.ziel_summe > 200000.0:
            for k in list(self.ziel_n):
                self.ziel_n[k] *= 0.5
            self.ziel_summe *= 0.5
        self._schnapp = []
        self._boni = []

    # ==================================================================
    # Auskunft und Speicher
    # ==================================================================
    @property
    def stats(self) -> Dict[str, Any]:
        bilanz = {
            ziel: {
                "takte_geprueft": int(b[0]),
                "brier": round(b[1] / b[0], 4) if b[0] else None,
                "treffer": round(b[2] / b[0], 4) if b[0] else None,
            }
            for ziel, b in self.anw_bilanz.items()
        }
        return {
            "entitaeten_im_blick": len(self.im_blick),
            "ziele": list(self.ziele),
            "takte": self.takte,
            "paar_merkmale": len(self.paar_merkmale),
            "folge_merkmale": len(self.folge),
            "anwesenheit": bilanz,
        }

    def to_dict(self) -> Dict[str, Any]:
        self._paar_nachtragen()
        n = len(self.paar_merkmale)
        breite = self.MAX_PAAR_MERKMALE
        paar = [[i, j, self.paar[i * breite + j]]
                for i in range(n) for j in range(n) if self.paar[i * breite + j] > 0]
        return {
            "zustand": self.zustand,
            "seit": self.seit,
            "stufen": {e: st.to_dict() for e, st in self.stufen.items()},
            "letzte_aktivitaet": self.letzte_aktivitaet,
            "wechsel": self.wechsel,
            "im_blick": self.im_blick,
            "ziele": self.ziele,
            "jetzt": self.jetzt,
            "versatz": self.versatz,
            "naechster_takt": self._naechster_takt,
            "takte": self.takte,
            "ereignisse": self.ereignisse,
            "paar_merkmale": [list(m) for m in self.paar_merkmale],
            "paar": paar,
            "paar_takte": self.paar_takte,
            "anw_n": self.anw_n,
            "anw": self.anw,
            "anw_zeit": self.anw_zeit,
            "anw_w": self.anw_w,
            "anw_v": self.anw_v,
            "anw_warteschlange": {z: [[t, d, list(m)] for t, d, m in q]
                                  for z, q in self._anw_warteschlange.items()},
            "anw_bilanz": self.anw_bilanz,
            "folge": [[list(m), [[y, c] for y, c in z.items()]] for m, z in self.folge.items()],
            # die gepflegte Rangfolge reist mit: bei Gleichstand entscheidet
            # ihre Geschichte, und die kennt ein neu gebautes nlargest nicht
            "folge_top": [[list(m), list(top)] for m, top in self.folge_top.items()],
            "ziel_n": [[y, c] for y, c in self.ziel_n.items()],
            "vertrauen": self.vertrauen,
            "vertrauen_g2": self._vertrauen_g2,
            "anw_g2": self._anw_g2,
            "anw_v_g2": self._anw_v_g2,
            # offener Vorhersage-Stand des Lage-Experten (bitgleiche Fortsetzung)
            "schnapp": [[e, list(m)] for e, m in self._schnapp],
            "kandidaten": list(self._kandidaten),
            "boni": [[e, [[i, b] for i, b in eintraege.items()]] for e, eintraege in self._boni],
            "q": list(self._q),
            "paar_offen": self._paar_offen,
        }

    def from_dict(self, daten: Dict[str, Any]) -> None:
        self.zustand = dict(daten.get("zustand", {}))
        self.seit = {k: float(v) for k, v in daten.get("seit", {}).items()}
        self.stufen = {}
        for e, roh in daten.get("stufen", {}).items():
            st = Stufen()
            st.from_dict(roh)
            self.stufen[e] = st
        self.letzte_aktivitaet = daten.get("letzte_aktivitaet")
        self.wechsel = {k: float(v) for k, v in daten.get("wechsel", {}).items()}
        self.im_blick = list(daten.get("im_blick", []))
        self._im_blick_set = set(self.im_blick)
        # Ziele nach der Prüfung des Wirts: Wer keins mehr ist, bleibt
        # Merkmal; sein Gelerntes schadet nicht und verdrängt sich nicht.
        self.ziele = [z for z in daten.get("ziele", []) if self.ziel_pruefer(z)]
        self.jetzt = daten.get("jetzt")
        self.versatz = float(daten.get("versatz", 0.0))
        self._naechster_takt = daten.get("naechster_takt")
        self.takte = int(daten.get("takte", 0))
        self.ereignisse = int(daten.get("ereignisse", 0))
        self.paar_merkmale = [tuple(m) for m in daten.get("paar_merkmale", [])][: self.MAX_PAAR_MERKMALE]
        self.paar_index = {m: i for i, m in enumerate(self.paar_merkmale)}
        self.paar = array("d", [0.0]) * (self.MAX_PAAR_MERKMALE * self.MAX_PAAR_MERKMALE)
        breite = self.MAX_PAAR_MERKMALE
        for i, j, w in daten.get("paar", []):
            if i < breite and j < breite:
                self.paar[int(i) * breite + int(j)] = float(w)
        self.paar_takte = float(daten.get("paar_takte", 0.0))
        self.anw_n = {z: [float(v[0]), float(v[1])] for z, v in daten.get("anw_n", {}).items()
                      if isinstance(v, list) and len(v) == 2}
        self.anw = {
            z: [{m: float(x) for m, x in t[0].items()}, {m: float(x) for m, x in t[1].items()}]
            for z, t in daten.get("anw", {}).items() if isinstance(t, list) and len(t) == 2
        }
        self.anw_zeit = {z: {k: [float(v[0]), float(v[1])] for k, v in d.items()}
                         for z, d in daten.get("anw_zeit", {}).items()}
        self.anw_w = {z: [float(x) for x in w][:2] for z, w in daten.get("anw_w", {}).items()
                      if isinstance(w, list) and len(w) >= 2}
        self.anw_v = {z: {q: float(x) for q, x in d.items()}
                      for z, d in daten.get("anw_v", {}).items()}
        self._anw_warteschlange = {
            z: deque((float(t), int(d), list(m)) for t, d, m in q)
            for z, q in daten.get("anw_warteschlange", {}).items()
        }
        self.anw_bilanz = {k: [float(x) for x in v] for k, v in daten.get("anw_bilanz", {}).items()}
        self.folge = {}
        self.folge_n = {}
        self.folge_top = {}
        for m, eintraege in daten.get("folge", []):
            schluessel = tuple(m)
            z = {(tuple(y) if isinstance(y, list) else y): float(c) for y, c in eintraege}
            self.folge[schluessel] = z
            self.folge_n[schluessel] = sum(z.values())
            self.folge_top[schluessel] = heapq.nlargest(3, z, key=z.__getitem__)
        for m, top in daten.get("folge_top", []):
            schluessel = tuple(m)
            z = self.folge.get(schluessel)
            if z is not None:
                gepflegt = [(tuple(y) if isinstance(y, list) else y) for y in top]
                if all(y in z for y in gepflegt):
                    self.folge_top[schluessel] = gepflegt
        self.ziel_n = {(tuple(y) if isinstance(y, list) else y): float(c)
                       for y, c in daten.get("ziel_n", [])}
        self.ziel_summe = sum(self.ziel_n.values())
        self.vertrauen = {k: float(v) for k, v in daten.get("vertrauen", {}).items()}
        self._vertrauen_g2 = {k: float(v) for k, v in daten.get("vertrauen_g2", {}).items()}
        self._anw_g2 = {k: [float(x) for x in v] for k, v in daten.get("anw_g2", {}).items()}
        self._anw_v_g2 = {k: {q: float(x) for q, x in d.items()}
                          for k, d in daten.get("anw_v_g2", {}).items()}
        self._schnapp = [(e, tuple(m)) for e, m in daten.get("schnapp", [])]
        self._kandidaten = [(tuple(y) if isinstance(y, list) else y)
                            for y in daten.get("kandidaten", [])]
        self._boni = [(e, {int(i): float(b) for i, b in eintraege})
                      for e, eintraege in daten.get("boni", [])]
        self._q = [float(x) for x in daten.get("q", [])]
        self._paar_offen = int(daten.get("paar_offen", 0))


def lage_setzen(kortex: AssociationCortex, thalamus: Any, entity_id: str,
                zustand: Any, zeit=None) -> None:
    """Eine Zustandsänderung ins Lagebild, gelesen wie der Thalamus liest.

    Für jeden Wirt gleich (Engine, Home-Assistant-Integration), und zwar VOR
    dem Thalamus aufzurufen: Auch was er gleich verwirft (``unavailable``,
    Entitäten ohne Raum), gehört ins Lagebild. Der Thalamus liefert die
    Semantik und die Lesart des Werts; Leistungen bekommen eigene
    Gerätestufen. Ignorierte Entitäten und solche ohne Semantik bleiben
    draußen — außer Personen und Trackern, den Zielen der Anwesenheit."""
    if entity_id in getattr(thalamus, "_ignored_entities", ()):
        return
    semantic = thalamus.entity_semantic.get(entity_id)
    if semantic is None:
        unassigned = getattr(thalamus, "_unassigned_entities", {}).get(entity_id)
        semantic = unassigned.get("semantic") if unassigned else None
    if semantic is None and not kortex.ziel_pruefer(entity_id):
        return
    roh = None if zustand is None else str(zustand)
    normiert = None
    if semantic and roh is not None:
        try:
            normiert = thalamus._normalize_state(semantic, roh)
        except Exception:  # noqa: BLE001 - ein kaputter Wert ist kein Absturz
            normiert = None
    kortex.setze(entity_id, roh, zeit, normiert=normiert,
                 numerisch=semantic in NUMERISCHE_STUFEN)
