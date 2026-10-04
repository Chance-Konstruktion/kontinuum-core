"""Die drei dummen Gegner B0/B1/B2 (Stufe 1, Protokoll § 3).

Sie sind absichtlich stark gebaut (Rueckfall auf B0, Glaettung,
deterministische Gleichstaende), damit ein Sieg der Engine nicht durch
Strohmaenner entsteht:

* **B0** — das haeufigste Folgeereignis (Unigramm).
* **B1** — die Uhrzeit-Gewohnheit: (Wochentag, Stunde) des Ereignisses,
  das gerade beobachtet wurde -> haeufigstes Folgeereignis; unbekannter
  Schluessel -> B0.
* **B2** — die reine 1-Gramm-Markov-Kette: P(naechstes | letztes);
  unbekannter Kontext -> B0.

Gemeinsame Ordnung (genau wie im Protokoll):

* **Predict-then-learn ist Pflicht des Aufrufers** (wie bei der Engine:
  `predict` vor `learn`). Der ehrliche Fluss je Ereignis ist:
  `vorhersage(ts_aktuell)` -> mit dem tatsaechlichen Token vergleichen
  -> `beobachte(token, ts_aktuell)`. Diese Klassen erzwingen die
  Ordnung nicht — sie lesen nur, was bis dahin beobachtet wurde.
* **Gleichstand:** Haeufigkeit absteigend, dann Token lexikografisch
  aufsteigend — nie Zufall, nie Einfuegereihenfolge.
* **Glaettung:** alpha = 0,5 gegen das FESTE Vokabular; damit sind die
  Wahrscheinlichkeiten fuer die Kalibrierung definiert, und die Summe
  ueber das Vokabular ist 1. Ein Token ausserhalb des Vokabulars ist
  ein Aufruferfehler MIT NAMEN (das Vokabular kommt aus der Spur und
  ist vollstaendig — Existenzwissen, kein Verhaltenswissen).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class Zaehler:
    """Gezaehlte Verteilung mit Glaettung gegen ein festes Vokabular."""

    vokabular: Sequence[str]
    alpha: float = 0.5
    zaehler: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._erlaubt = set(self.vokabular)

    def beobachte(self, token: str) -> None:
        if token not in self._erlaubt:
            raise ValueError(
                f"Token ausserhalb des Vokabulars: {token!r} — das "
                "Vokabular ist die komplette Tokenliste der Spur."
            )
        self.zaehler[token] = self.zaehler.get(token, 0.0) + 1.0

    def gesamt(self) -> float:
        return sum(self.zaehler.values())

    def verteilung(self) -> List[Tuple[str, float]]:
        """(Token, Wahrscheinlichkeit) fuer das ganze Vokabular —
        geglaettet, in deterministischer Reihenfolge."""
        gesamt = self.gesamt()
        nenner = gesamt + self.alpha * len(self.vokabular)
        if nenner <= 0:
            return []
        geordnet = sorted(
            ((t, self.zaehler.get(t, 0.0)) for t in self.vokabular),
            key=lambda paar: (-paar[1], paar[0]),
        )
        return [(t, (n + self.alpha) / nenner) for t, n in geordnet]

    def top(self, k: int = 3) -> List[Tuple[str, float]]:
        return self.verteilung()[:k]


class B0:
    """Das haeufigste Folgeereignis (Unigramm)."""

    name = "B0"

    def __init__(self, vokabular: Sequence[str], alpha: float = 0.5):
        self._unigramm = Zaehler(vokabular, alpha)

    def beobachte(self, token: str, zeit: Optional[datetime] = None) -> None:
        self._unigramm.beobachte(token)

    def vorhersage(self, zeit: datetime, k: int = 3) -> List[Tuple[str, float]]:
        return self._unigramm.top(k)


class B1:
    """Die Uhrzeit-Gewohnheit: (Wochentag, Stunde) -> Folgeereignis.

    Gelernt wird der UEBERGANG: Der Schluessel kommt von der Zeit des
    Ereignisses, das den Uebergang ausgeloest hat (der Vorhersagemoment),
    der gezaehlte Token ist das NAECHSTE Ereignis. Vorhersage und
    Lernen benutzen damit denselben Schluessel — und zwischen zwei
    Ereignissen ueber eine Stundengrenze bleibt der Schluessel der des
    Vorhersagemoments, nicht der des Ziels.
    """

    name = "B1"

    def __init__(self, vokabular: Sequence[str], alpha: float = 0.5):
        self.vokabular = list(vokabular)
        self.alpha = alpha
        self._nach_zeit: Dict[Tuple[int, int], Zaehler] = {}
        self._b0 = B0(vokabular, alpha)
        self._letzter_schluessel: Optional[Tuple[int, int]] = None

    def beobachte(self, token: str, zeit: Optional[datetime] = None) -> None:
        if zeit is None:
            raise ValueError(
                "B1 braucht die Zeit des Ereignisses (Wochentag/Stunde) — "
                "ohne sie waere es B0 mit Umweg."
            )
        if self._letzter_schluessel is not None:
            tabelle = self._nach_zeit.get(self._letzter_schluessel)
            if tabelle is None:
                tabelle = Zaehler(self.vokabular, self.alpha)
                self._nach_zeit[self._letzter_schluessel] = tabelle
            tabelle.beobachte(token)
        self._b0.beobachte(token)
        self._letzter_schluessel = (zeit.weekday(), zeit.hour)

    def vorhersage(self, zeit: datetime, k: int = 3) -> List[Tuple[str, float]]:
        tabelle = self._nach_zeit.get((zeit.weekday(), zeit.hour))
        if tabelle is None or tabelle.gesamt() == 0:
            return self._b0.vorhersage(zeit, k)
        return tabelle.top(k)


class B2:
    """Die reine 1-Gramm-Markov-Kette — ohne Kontext, ohne Buckets."""

    name = "B2"

    def __init__(self, vokabular: Sequence[str], alpha: float = 0.5):
        self.vokabular = list(vokabular)
        self.alpha = alpha
        self._paare: Dict[str, Zaehler] = {}
        self._b0 = B0(vokabular, alpha)
        self._letzter: Optional[str] = None

    def beobachte(self, token: str, zeit: Optional[datetime] = None) -> None:
        self._b0.beobachte(token)
        if self._letzter is not None:
            tabelle = self._paare.get(self._letzter)
            if tabelle is None:
                tabelle = Zaehler(self.vokabular, self.alpha)
                self._paare[self._letzter] = tabelle
            tabelle.beobachte(token)
        self._letzter = token

    def vorhersage(self, zeit: datetime, k: int = 3) -> List[Tuple[str, float]]:
        tabelle = self._paare.get(self._letzter) if self._letzter else None
        if tabelle is None or tabelle.gesamt() == 0:
            return self._b0.vorhersage(zeit, k)
        return tabelle.top(k)


def gegner(vokabular: Sequence[str], alpha: float = 0.5) -> Dict[str, object]:
    """Die Drei, frisch und mit demselben Vokabular."""
    return {
        "B0": B0(vokabular, alpha),
        "B1": B1(vokabular, alpha),
        "B2": B2(vokabular, alpha),
    }


__all__ = ["Zaehler", "B0", "B1", "B2", "gegner"]
