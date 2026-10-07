"""Claustrum — die Börse der Vorhersagen.

Biologisches Vorbild: Das Claustrum ist eine dünne Schicht unter der
Inselrinde, verbunden mit fast allen Rindenarealen. Crick und Koch haben es
als „Dirigenten“ beschrieben, der die Beiträge der Areale bündelt. Hier tut
es genau das: Es hört alle Vorhersager an und lernt im Betrieb, wem es in
welcher Lage trauen kann.

Warum es das gibt (Messstand kontinuum-core#2, nachgemessen am 06.10.2026):
Die alte Kette verlor auf der Simulation gegen eine reine 1-Gramm-Markov-
Kette (B2), klassisch −10,2 %, Geräte-Haus −4,4 %. Zwei Ursachen:

1. Der Hippocampus zersplittert sein Gedächtnis in bis zu 96 Kontext-Eimer
   und hat im klassischen Haus bei ~70 % der Ereignisse keinen einzigen
   Kandidaten über seiner Schwelle.
2. Das Ranking stellt eingeschleuste Kandidaten (Intervall, Reflex) mit
   fester Konfidenz vor richtige, ohne je zu prüfen, ob sie treffen.

Die Antwort ist keine weitere Regel, sondern Messung im Betrieb: Jeder
Experte nennt für jeden Kandidaten eine Wahrscheinlichkeit. Das Claustrum
legt sie log-linear übereinander (ein Produkt der Experten, wie die Mischer
in PAQ-Kompressoren) und schiebt nach jedem Ereignis die Gewichte in
Richtung dessen, der recht hatte (Gradient der Log-Likelihood, RMSprop).
Ein Experte, der nichts taugt, landet bei Gewicht ~0 und kann nicht mehr
schaden. Die Gewichte gelten global plus je Mischkontext (Semantik des
letzten Ereignisses), denn nach einer Bewegung trägt ein anderer Experte
als nach einem Leistungssprung.

Experten (alle in Ereigniszeit, keine Wanduhr):

* ``sequenz``   – Markov-Kette variabler Ordnung 1..3, Witten-Bell-geglättet,
  ohne Eimer. Der ehrliche Nachfolger von B2; die Mischung startet mit
  Gewicht 1 auf ihr, also nie schlechter als die Kette selbst.
* ``zeit``      – (Werktag/Wochenende, Stunde ±1) → nächstes Ereignis.
* ``folgezeit`` – (letztes Ereignis, Tagesabschnitt) → nächstes Ereignis.
* ``lage``      – der Assoziationskortex: was folgt, wenn die Geräte gerade
  so stehen (``association_cortex.py``).
* ``extern``    – was der Aufrufer dazugibt (Hippocampus, Reflex) — als
  Lift gegen die Grundrate, geglättet und gedeckelt: Wer nicht genannt wird,
  bleibt neutral. (Ungeglättet hieß „nicht genannt“ ``log(EPS)`` — eine
  einzige überfällige Kadenz drückte alles andere auf 0 %.)

Pi-Budget: O(Experten × Kandidaten) je Ereignis (≤ 6 × ≤ 48), alle Zähler
gedeckelt (Halbieren statt Wachsen, seltene Kontexte werden verdrängt).
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

#: Kleinste Wahrscheinlichkeit, die ein Experte nennen darf. Darunter wird
#: abgeschnitten, sonst kippt ein einzelner Experte mit log(0) die Mischung.
EPS = 1e-6

#: Externe Stimmen: ``p = (1−λ)·e + λ·Grundrate``, als Lift gegen die
#: Grundrate (nicht Genannte → 0, also neutral), höchstens EXTERN_DECKEL
#: Nats — eine Stimme darf überzeugen, aber nicht allein entscheiden.
EXTERN_GLAETTUNG = 0.5
EXTERN_DECKEL = 4.0

#: Die Experten in fester Reihenfolge (Spalten der Mischung).
EXPERTEN = ("sequenz", "zeit", "folgezeit", "lage", "extern")


def _top(zaehler: Dict[Any, float], k: int) -> List[Any]:
    """Die k größten Schlüssel eines Zählers (Gleichstand: Einfügereihenfolge)."""
    if len(zaehler) <= k:
        return sorted(zaehler, key=zaehler.__getitem__, reverse=True)
    return heapq.nlargest(k, zaehler, key=zaehler.__getitem__)


def _halbiere(zaehler: Dict[Any, float], rest: float = 0.25) -> float:
    """Halbiert alle Zähler, wirft winzige Reste weg, gibt die neue Summe."""
    summe = 0.0
    for schluessel in list(zaehler):
        wert = zaehler[schluessel] * 0.5
        if wert < rest:
            del zaehler[schluessel]
        else:
            zaehler[schluessel] = wert
            summe += wert
    return summe


def _paare(zaehler: Dict[Any, float]) -> List[List[Any]]:
    """Zähler als JSON-sichere Liste (Token-IDs bleiben Zahlen)."""
    return [[schluessel, wert] for schluessel, wert in zaehler.items()]


def _schluessel(roh: Any) -> Any:
    """Kontext-Schlüssel aus JSON zurück (Listen werden wieder Tupel)."""
    if isinstance(roh, list):
        return tuple(_schluessel(teil) for teil in roh)
    return roh


# ---------------------------------------------------------------------------
# Experte: Sequenz (Markov variabler Ordnung, Witten-Bell)
# ---------------------------------------------------------------------------


class Sequenz:
    """P(nächstes | letzte n Ereignisse), n = 1..ordnung, Witten-Bell.

    Witten-Bell mischt jede Ordnung mit der nächstkürzeren: Je mehr
    verschiedene Nachfolger ein Kontext schon hatte, desto mehr traut er der
    kürzeren Ordnung. Ein Kontext, den man zweimal gesehen hat, kann so
    nicht mehr eine Kette überstimmen, die man tausendmal gesehen hat —
    genau das passierte im alten Hippocampus (4-Gramm-Gewicht 0,95 schon ab
    zwei Beobachtungen).
    """

    def __init__(self, ordnung: int = 3,
                 deckel: Sequence[float] = (4096.0, 1024.0, 512.0),
                 max_kontexte: int = 20000, uni_deckel: float = 100000.0):
        self.ordnung = ordnung
        deckel = list(deckel)
        while len(deckel) < ordnung:
            deckel.append(deckel[-1])
        self.deckel = deckel
        self.max_kontexte = max_kontexte
        self.uni_deckel = uni_deckel
        self.zaehler: Dict[tuple, Dict[Hashable, float]] = {}
        self.summe: Dict[tuple, float] = {}
        self.uni: Dict[Hashable, float] = {}
        self.uni_summe = 0.0
        self.verlauf: deque = deque(maxlen=ordnung)
        self._kontexte: List[tuple] = []

    # -- Vorhersage ----------------------------------------------------
    def bereite(self) -> None:
        """Merkt sich die Kontexte der aktuellen Lage (kürzester zuerst)."""
        h = tuple(self.verlauf)
        kontexte = []
        for n in range(1, len(h) + 1):
            ctx = h[-n:]
            if ctx not in self.zaehler:
                break
            kontexte.append(ctx)
        self._kontexte = kontexte

    def p_uni(self, y: Hashable) -> float:
        vokabular = len(self.uni) + 1
        return (self.uni.get(y, 0.0) + 0.5) / (self.uni_summe + 0.5 * vokabular)

    def p(self, y: Hashable, bis: Optional[int] = None) -> float:
        """Witten-Bell-Wahrscheinlichkeit; ``bis`` begrenzt die Ordnung."""
        wert = self.p_uni(y)
        kontexte = self._kontexte if bis is None else self._kontexte[:bis]
        for ctx in kontexte:
            z = self.zaehler[ctx]
            verschiedene = len(z)
            wert = (z.get(y, 0.0) + verschiedene * wert) / (self.summe[ctx] + verschiedene)
        return wert

    def belege(self, y: Hashable) -> int:
        """Wie oft folgte y auf den längsten bekannten Kontext? (n_obs)"""
        for ctx in reversed(self._kontexte):
            wert = self.zaehler[ctx].get(y)
            if wert:
                return int(round(wert))
        return int(round(self.uni.get(y, 0.0)))

    def kandidaten(self, k: int = 15) -> List[Hashable]:
        aus: List[Hashable] = []
        gesehen = set()
        for ctx in reversed(self._kontexte):  # längster Kontext zuerst
            for t in _top(self.zaehler[ctx], k):
                if t not in gesehen:
                    gesehen.add(t)
                    aus.append(t)
        for t in _top(self.uni, 5):
            if t not in gesehen:
                gesehen.add(t)
                aus.append(t)
        return aus

    # -- Lernen --------------------------------------------------------
    def lerne(self, y: Hashable) -> None:
        self.uni[y] = self.uni.get(y, 0.0) + 1.0
        self.uni_summe += 1.0
        if self.uni_summe > self.uni_deckel:
            self.uni_summe = _halbiere(self.uni)
        h = tuple(self.verlauf)
        for n in range(1, len(h) + 1):
            ctx = h[-n:]
            z = self.zaehler.get(ctx)
            if z is None:
                z = {}
                self.zaehler[ctx] = z
                self.summe[ctx] = 0.0
            z[y] = z.get(y, 0.0) + 1.0
            summe = self.summe[ctx] + 1.0
            if summe > self.deckel[n - 1]:
                summe = _halbiere(z)
            self.summe[ctx] = summe
        self.verlauf.append(y)
        if len(self.zaehler) > self.max_kontexte:
            self._verdraengen()

    def _verdraengen(self) -> None:
        """Seltene Kontexte gehen, die häufigsten drei Viertel bleiben."""
        behalten = int(self.max_kontexte * 0.75)
        rang = heapq.nlargest(behalten, self.summe, key=self.summe.__getitem__)
        bleibt = set(rang)
        for ctx in list(self.zaehler):
            if ctx not in bleibt:
                del self.zaehler[ctx]
                del self.summe[ctx]

    # -- Speicher ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "ordnung": self.ordnung,
            "kontexte": [[list(ctx), _paare(z)] for ctx, z in self.zaehler.items()],
            "uni": _paare(self.uni),
            "verlauf": list(self.verlauf),
        }

    def from_dict(self, daten: dict) -> None:
        self.zaehler = {}
        self.summe = {}
        for ctx, paare in daten.get("kontexte", []):
            z = {_schluessel(t): float(w) for t, w in paare}
            self.zaehler[tuple(_schluessel(t) for t in ctx)] = z
            self.summe[tuple(_schluessel(t) for t in ctx)] = sum(z.values())
        self.uni = {_schluessel(t): float(w) for t, w in daten.get("uni", [])}
        self.uni_summe = sum(self.uni.values())
        self.verlauf = deque((_schluessel(t) for t in daten.get("verlauf", [])),
                             maxlen=self.ordnung)


# ---------------------------------------------------------------------------
# Experte: Zeitgewohnheit und Folge je Tagesabschnitt
# ---------------------------------------------------------------------------


def _wochenende(zeit) -> int:
    return 1 if zeit.weekday() >= 5 else 0


class Zeitgewohnheit:
    """(Werktag/Wochenende, Stunde) → nächstes Ereignis.

    Wie Gegner B1, nur ehrlicher geglättet: die Nachbarstunden zählen mit
    (¼ · h−1 + ½ · h + ¼ · h+1), jede Stunde fällt nach Witten-Bell auf die
    Grundhäufigkeit zurück. Gelernt wird mit der Stunde des
    VORHERSAGEMOMENTS, wie bei B1.
    """

    NACHBARN = ((-1, 0.25), (0, 0.5), (1, 0.25))

    def __init__(self, deckel: float = 2000.0):
        self.deckel = deckel
        self.slots: Dict[int, Dict[Hashable, float]] = {}
        self.summe: Dict[int, float] = {}
        self._slot: Optional[int] = None

    @staticmethod
    def slot(zeit) -> int:
        return _wochenende(zeit) * 24 + zeit.hour

    def bereite(self, zeit) -> None:
        self._slot = self.slot(zeit) if zeit is not None else None

    def _nachbar(self, versatz: int) -> int:
        tagtyp, stunde = divmod(self._slot, 24)
        return tagtyp * 24 + (stunde + versatz) % 24

    def kandidaten(self, k: int = 8) -> List[Hashable]:
        if self._slot is None:
            return []
        z = self.slots.get(self._slot)
        return _top(z, k) if z else []

    def p(self, y: Hashable, basis: float) -> float:
        if self._slot is None:
            return basis
        wert = 0.0
        for versatz, gewicht in self.NACHBARN:
            s = self._nachbar(versatz)
            z = self.slots.get(s)
            if not z:
                wert += gewicht * basis
                continue
            verschiedene = len(z)
            wert += gewicht * (z.get(y, 0.0) + verschiedene * basis) / (self.summe[s] + verschiedene)
        return wert

    def lerne(self, y: Hashable) -> None:
        if self._slot is None:
            return
        z = self.slots.setdefault(self._slot, {})
        z[y] = z.get(y, 0.0) + 1.0
        summe = self.summe.get(self._slot, 0.0) + 1.0
        if summe > self.deckel:
            summe = _halbiere(z)
        self.summe[self._slot] = summe

    def to_dict(self) -> dict:
        return {"slots": [[s, _paare(z)] for s, z in self.slots.items()]}

    def from_dict(self, daten: dict) -> None:
        self.slots = {}
        self.summe = {}
        for s, paare in daten.get("slots", []):
            z = {_schluessel(t): float(w) for t, w in paare}
            self.slots[int(s)] = z
            self.summe[int(s)] = sum(z.values())


class FolgeZeit:
    """(letztes Ereignis, Tagesabschnitt) → nächstes Ereignis.

    Acht Abschnitte zu je drei Stunden, getrennt nach Werktag/Wochenende.
    Rückfall nach Witten-Bell auf die Sequenz erster Ordnung — so bleibt
    ein seltener Abschnitt nie stumm, er klingt dann wie die Kette.
    """

    def __init__(self, deckel: float = 1024.0, max_kontexte: int = 20000):
        self.deckel = deckel
        self.max_kontexte = max_kontexte
        self.zaehler: Dict[tuple, Dict[Hashable, float]] = {}
        self.summe: Dict[tuple, float] = {}
        self._kontext: Optional[tuple] = None

    def bereite(self, letztes: Optional[Hashable], zeit) -> None:
        if letztes is None or zeit is None:
            self._kontext = None
            return
        self._kontext = (letztes, _wochenende(zeit) * 8 + zeit.hour // 3)

    def kandidaten(self, k: int = 8) -> List[Hashable]:
        z = self.zaehler.get(self._kontext) if self._kontext else None
        return _top(z, k) if z else []

    def p(self, y: Hashable, basis: float) -> float:
        z = self.zaehler.get(self._kontext) if self._kontext else None
        if not z:
            return basis
        verschiedene = len(z)
        return (z.get(y, 0.0) + verschiedene * basis) / (self.summe[self._kontext] + verschiedene)

    def lerne(self, y: Hashable) -> None:
        if self._kontext is None:
            return
        z = self.zaehler.get(self._kontext)
        if z is None:
            z = {}
            self.zaehler[self._kontext] = z
            self.summe[self._kontext] = 0.0
        z[y] = z.get(y, 0.0) + 1.0
        summe = self.summe[self._kontext] + 1.0
        if summe > self.deckel:
            summe = _halbiere(z)
        self.summe[self._kontext] = summe
        if len(self.zaehler) > self.max_kontexte:
            behalten = set(heapq.nlargest(int(self.max_kontexte * 0.75), self.summe,
                                          key=self.summe.__getitem__))
            for ctx in list(self.zaehler):
                if ctx not in behalten:
                    del self.zaehler[ctx]
                    del self.summe[ctx]

    def to_dict(self) -> dict:
        return {"kontexte": [[list(ctx), _paare(z)] for ctx, z in self.zaehler.items()]}

    def from_dict(self, daten: dict) -> None:
        self.zaehler = {}
        self.summe = {}
        for ctx, paare in daten.get("kontexte", []):
            schluessel = tuple(_schluessel(t) for t in ctx)
            z = {_schluessel(t): float(w) for t, w in paare}
            self.zaehler[schluessel] = z
            self.summe[schluessel] = sum(z.values())


# ---------------------------------------------------------------------------
# Der Mischer
# ---------------------------------------------------------------------------


class Mischer:
    """Log-lineare Mischung mit gelernten Gewichten (global + je Kontext).

    q(y) ∝ exp(Σ_i w_i · log p_i(y)), w = w_global + w_kontext.

    Nach jedem Ereignis y*: g_i = log p_i(y*) − E_q[log p_i] — wer dem
    Treffer mehr Wahrscheinlichkeit gab als die Mischung im Mittel, gewinnt
    Gewicht. Schrittweite nach RMSprop, damit Experten mit großen und
    kleinen Log-Werten gleich schnell lernen.
    """

    W_MIN = -1.0
    W_MAX = 4.0

    def __init__(self, namen: Sequence[str] = EXPERTEN,
                 start: Optional[Dict[str, float]] = None,
                 lernrate: float = 0.02, lernrate_kontext: float = 0.02,
                 vergessen: float = 0.999):
        self.namen = list(namen)
        start = start or {"sequenz": 1.0}
        self.global_w = [float(start.get(n, 0.0)) for n in self.namen]
        self.global_g2 = [0.0] * len(self.namen)
        self.kontext_w: Dict[Hashable, List[float]] = {}
        self.kontext_g2: Dict[Hashable, List[float]] = {}
        self.lernrate = lernrate
        self.lernrate_kontext = lernrate_kontext
        self.vergessen = vergessen
        self.updates = 0

    def gewichte(self, kontext: Hashable) -> List[float]:
        lokal = self.kontext_w.get(kontext)
        if lokal is None:
            return list(self.global_w)
        return [g + l for g, l in zip(self.global_w, lokal)]

    def mische(self, kontext: Hashable, logp: List[List[float]]) -> List[float]:
        """Wahrscheinlichkeiten über die Kandidaten (Zeilen von ``logp``)."""
        w = self.gewichte(kontext)
        werte = [sum(wi * xi for wi, xi in zip(w, zeile)) for zeile in logp]
        oben = max(werte)
        exps = [math.exp(v - oben) for v in werte]
        summe = sum(exps)
        return [e / summe for e in exps]

    def lerne(self, kontext: Hashable, logp: List[List[float]], q: List[float],
              ziel: int) -> None:
        n = len(self.namen)
        grad = []
        for i in range(n):
            erwartung = 0.0
            for zeile, qi in zip(logp, q):
                erwartung += qi * zeile[i]
            grad.append(logp[ziel][i] - erwartung)
        self._schritt(self.global_w, self.global_g2, grad, self.lernrate)
        lokal = self.kontext_w.get(kontext)
        if lokal is None:
            lokal = [0.0] * n
            self.kontext_w[kontext] = lokal
            self.kontext_g2[kontext] = [0.0] * n
        self._schritt(lokal, self.kontext_g2[kontext], grad, self.lernrate_kontext,
                      lokal=True)
        self.updates += 1

    def _schritt(self, w: List[float], g2: List[float], grad: List[float],
                 rate: float, lokal: bool = False) -> None:
        for i, g in enumerate(grad):
            g2[i] = self.vergessen * g2[i] + (1.0 - self.vergessen) * g * g
            schritt = rate * g / (math.sqrt(g2[i]) + 1e-6)
            neu = w[i] + schritt
            if lokal:
                # Ein Kontext darf das Globale nur um ±1,5 verschieben.
                neu = max(-1.5, min(1.5, neu))
            else:
                neu = max(self.W_MIN, min(self.W_MAX, neu))
            w[i] = neu

    def to_dict(self) -> dict:
        return {
            "namen": self.namen,
            "global_w": self.global_w,
            "global_g2": self.global_g2,
            "kontext": [[k, w, self.kontext_g2.get(k, [0.0] * len(w))]
                        for k, w in self.kontext_w.items()],
            "updates": self.updates,
        }

    def from_dict(self, daten: dict) -> None:
        namen = daten.get("namen") or self.namen
        index = {n: i for i, n in enumerate(namen)}

        def ordne(werte: List[float], vorgabe: float = 0.0) -> List[float]:
            return [float(werte[index[n]]) if n in index and index[n] < len(werte)
                    else vorgabe for n in self.namen]

        self.global_w = ordne(daten.get("global_w", []))
        self.global_g2 = ordne(daten.get("global_g2", []))
        self.kontext_w = {}
        self.kontext_g2 = {}
        for k, w, g2 in daten.get("kontext", []):
            schluessel = _schluessel(k)
            self.kontext_w[schluessel] = ordne(w)
            self.kontext_g2[schluessel] = ordne(g2)
        self.updates = int(daten.get("updates", 0))


# ---------------------------------------------------------------------------
# Das Claustrum
# ---------------------------------------------------------------------------


class Claustrum:
    """Experten + Mischer, predict-then-learn in einem Aufruf.

    Ablauf je Ereignis ``beobachte(y, zeit, ...)``:

    1. Die gespeicherte Vorhersage (gemacht nach dem vorigen Ereignis) wird
       am tatsächlichen ``y`` gemessen; der Mischer lernt daraus.
    2. Alle Experten lernen ``y``.
    3. Die neue Vorhersage für das NÄCHSTE Ereignis entsteht und wird
       zurückgegeben (Liste aus ``(token, wahrscheinlichkeit)``).

    Der Lage-Experte (Assoziationskortex) bekommt seine Gerätezustände vom
    Aufrufer über ``lage.setze(...)`` — VOR ``beobachte``; er friert seine
    Merkmale im Vorhersagemoment ein, damit nie ein Zustand des Ziels in
    seine eigene Vorhersage rutscht.
    """

    MAX_KANDIDATEN = 48

    def __init__(self, lage: Any = None, ordnung: int = 3,
                 lernrate: float = 0.02, aus: Sequence[str] = ()):
        self.sequenz = Sequenz(ordnung=ordnung)
        self.zeit = Zeitgewohnheit()
        self.folgezeit = FolgeZeit()
        self.lage = lage
        #: Abschaltprobe: diese Experten bleiben stumm (Messstand).
        self.aus = set(aus)
        self.mischer = Mischer(EXPERTEN, lernrate=lernrate, lernrate_kontext=lernrate)
        self._stand: Optional[Tuple[Hashable, List[Hashable], List[List[float]],
                                    List[float]]] = None
        self.letzte_vorhersage: List[Tuple[Hashable, float]] = []
        self.ereignisse = 0
        self.bewertet = 0
        self.treffer = 0
        #: Log-Verlust (Bits) des letzten Ereignisses — die ehrliche Überraschung.
        self.letzte_ueberraschung: Optional[float] = None

    # ------------------------------------------------------------------
    def beobachte(self, y: Hashable, zeit=None, kontext: Hashable = None,
                  extern: Optional[Dict[Hashable, float]] = None
                  ) -> List[Tuple[Hashable, float]]:
        self.ereignisse += 1
        # 1) die alte Vorhersage am Ziel messen
        if self._stand is not None:
            alt_kontext, kandidaten, logp, q = self._stand
            try:
                ziel = kandidaten.index(y)
            except ValueError:
                ziel = -1
            self.bewertet += 1
            if ziel >= 0:
                if ziel == max(range(len(q)), key=q.__getitem__):
                    self.treffer += 1
                self.letzte_ueberraschung = -math.log2(max(q[ziel], EPS))
                self.mischer.lerne(alt_kontext, logp, q, ziel)
            else:
                self.letzte_ueberraschung = -math.log2(EPS)
        # 2) die Experten lernen
        self.sequenz.lerne(y)
        self.zeit.lerne(y)
        self.folgezeit.lerne(y)
        if self.lage is not None and "lage" not in self.aus:
            self.lage.lerne_folge(y)
        # 3) neu vorhersagen
        return self._vorhersage(y, zeit, kontext, extern)

    def _vorhersage(self, letztes: Hashable, zeit, kontext: Hashable,
                    extern: Optional[Dict[Hashable, float]]
                    ) -> List[Tuple[Hashable, float]]:
        self.sequenz.bereite()
        self.zeit.bereite(zeit)
        self.folgezeit.bereite(letztes, zeit)
        lage_an = self.lage is not None and "lage" not in self.aus
        if lage_an:
            self.lage.bereite_folge(zeit)

        kandidaten: List[Hashable] = []
        gesehen = set()

        def nimm(liste) -> None:
            for t in liste:
                if t not in gesehen and len(kandidaten) < self.MAX_KANDIDATEN:
                    gesehen.add(t)
                    kandidaten.append(t)

        nimm(self.sequenz.kandidaten(15))
        if "folgezeit" not in self.aus:
            nimm(self.folgezeit.kandidaten(8))
        if "zeit" not in self.aus:
            nimm(self.zeit.kandidaten(8))
        if lage_an:
            nimm(self.lage.kandidaten_folge(10))
        if extern and "extern" not in self.aus:
            nimm(sorted(extern, key=extern.__getitem__, reverse=True))
        if not kandidaten:
            self._stand = None
            self.letzte_vorhersage = []
            return []

        lage_lifts = self.lage.lifts_folge(kandidaten) if lage_an else None
        extern_an = bool(extern) and "extern" not in self.aus
        extern_summe = max(1.0, sum(extern.values())) if extern_an else 1.0
        log_lambda = math.log(EXTERN_GLAETTUNG)
        logp: List[List[float]] = []
        for i, y in enumerate(kandidaten):
            p_seq = max(self.sequenz.p(y), EPS)
            basis = self.sequenz.p_uni(y)
            p_eins = self.sequenz.p(y, bis=1)
            zeile = [
                math.log(p_seq),
                0.0 if "zeit" in self.aus else math.log(max(self.zeit.p(y, basis), EPS)),
                0.0 if "folgezeit" in self.aus else
                math.log(max(self.folgezeit.p(y, p_eins), EPS)),
                lage_lifts[i] if lage_lifts is not None else 0.0,
                min(EXTERN_DECKEL, math.log(
                    (1.0 - EXTERN_GLAETTUNG) * extern.get(y, 0.0) / extern_summe
                    / max(basis, EPS) + EXTERN_GLAETTUNG) - log_lambda)
                if extern_an else 0.0,
            ]
            logp.append(zeile)
        q = self.mischer.mische(kontext, logp)
        self._stand = (kontext, kandidaten, logp, q)
        rang = sorted(range(len(kandidaten)), key=lambda i: (-q[i], i))
        self.letzte_vorhersage = [(kandidaten[i], q[i]) for i in rang]
        return self.letzte_vorhersage

    # ------------------------------------------------------------------
    @property
    def trefferquote(self) -> float:
        return self.treffer / self.bewertet if self.bewertet else 0.0

    @property
    def stats(self) -> dict:
        return {
            "ereignisse": self.ereignisse,
            "trefferquote": round(self.trefferquote, 4),
            "gewichte": dict(zip(self.mischer.namen,
                                 (round(w, 3) for w in self.mischer.global_w))),
            "kontexte_mischer": len(self.mischer.kontext_w),
            "kontexte_sequenz": len(self.sequenz.zaehler),
        }

    def to_dict(self) -> dict:
        stand = None
        if self._stand is not None:
            kontext, kandidaten, logp, q = self._stand
            stand = [kontext, list(kandidaten), logp, q]
        return {
            "sequenz": self.sequenz.to_dict(),
            "zeit": self.zeit.to_dict(),
            "folgezeit": self.folgezeit.to_dict(),
            "mischer": self.mischer.to_dict(),
            "ereignisse": self.ereignisse,
            "bewertet": self.bewertet,
            "treffer": self.treffer,
            # Der offene Vorhersage-Stand reist mit: Nach dem Laden lernt
            # der Mischer aus dem nächsten Ereignis genau wie ohne Neustart.
            "stand": stand,
            "zeit_slot": self.zeit._slot,
            "folgezeit_kontext": list(self.folgezeit._kontext) if self.folgezeit._kontext else None,
            "letzte_vorhersage": [[t, q] for t, q in self.letzte_vorhersage],
        }

    def from_dict(self, daten: dict) -> None:
        self.sequenz.from_dict(daten.get("sequenz", {}))
        self.zeit.from_dict(daten.get("zeit", {}))
        self.folgezeit.from_dict(daten.get("folgezeit", {}))
        self.mischer.from_dict(daten.get("mischer", {}))
        self.ereignisse = int(daten.get("ereignisse", 0))
        self.bewertet = int(daten.get("bewertet", 0))
        self.treffer = int(daten.get("treffer", 0))
        stand = daten.get("stand")
        if stand:
            kontext, kandidaten, logp, q = stand
            self._stand = (_schluessel(kontext), [_schluessel(t) for t in kandidaten],
                           [[float(x) for x in zeile] for zeile in logp],
                           [float(x) for x in q])
        else:
            self._stand = None
        slot = daten.get("zeit_slot")
        self.zeit._slot = int(slot) if slot is not None else None
        kontext = daten.get("folgezeit_kontext")
        self.folgezeit._kontext = tuple(_schluessel(t) for t in kontext) if kontext else None
        self.letzte_vorhersage = [(_schluessel(t), float(q))
                                  for t, q in daten.get("letzte_vorhersage", [])]


def boersen_liste(claustrum: Claustrum, token: Hashable, zeit, kontext: Hashable,
                  hippocampus: Optional[Sequence[Sequence[Any]]] = None,
                  reflex: Optional[Tuple[Hashable, float, int]] = None,
                  faellig: Optional[Hashable] = None,
                  top: int = 5) -> List[Tuple[Hashable, float, float, str, int]]:
    """Ein Ereignis an die Börse; ihre Vorhersage im Listenformat der Engine
    ``(token, p, konfidenz, quelle, belege)``. Für jeden Wirt gleich (Engine,
    Home-Assistant-Integration).

    * ``hippocampus``: dessen Rohliste ``(token, p, …)`` — eine Stimme im Rat.
    * ``reflex``: ``(ziel, konfidenz, belege)`` eines sicheren Reflexes —
      Stimme im Rat und, führt die Börse ihn nicht ohnehin unter den besten
      ``top``, Anwärter auf den LETZTEN Platz.
    * ``faellig``: eine überfällige Kadenz der Intervall-Uhr — keine Stimme
      („überfällig“ heißt „bald“, nicht „als Nächstes“), nur Anwärter auf den
      letzten Platz.
    """
    extern: Dict[Hashable, float] = {}
    for eintrag in hippocampus or []:
        extern[eintrag[0]] = max(extern.get(eintrag[0], 0.0), float(eintrag[1]))
    if reflex is not None:
        extern[reflex[0]] = max(extern.get(reflex[0], 0.0), float(reflex[1]))
    liste = claustrum.beobachte(token, zeit, kontext=kontext, extern=extern)
    tupel = [
        (tok, round(q, 4), round(q, 4), "claustrum", claustrum.sequenz.belege(tok))
        for tok, q in liste[:top]
    ]
    q_von = dict(liste)
    anwaerter = (("cerebellum", reflex[0] if reflex is not None else None,
                  reflex[2] if reflex is not None else 0),
                 ("interval_timing", faellig, 0))
    for quelle, ziel, belege in anwaerter:
        if ziel is None or any(t[0] == ziel for t in tupel):
            continue
        q = round(q_von.get(ziel, 0.0), 4)
        eintrag = (ziel, q, q, quelle, belege)
        if len(tupel) >= top:
            tupel[-1] = eintrag
        else:
            tupel.append(eintrag)
    return tupel
