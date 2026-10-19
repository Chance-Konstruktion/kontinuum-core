"""Locus Coeruleus: tracks arousal from event density."""

from __future__ import annotations

import time
from collections import deque
from typing import Callable, Optional


def _endlich(x) -> bool:
    """Ist x eine endliche Zahl? Bewusst ohne math.isfinite: das wirft
    bei sehr grossen ints OverflowError; x - x == 0 gilt fuer jede
    endliche Zahl und nur fuer sie (NaN/inf ergeben NaN)."""
    return x - x == 0


class LocusCoeruleus:
    WINDOW_SECONDS = 60
    MAX_EVENTS = 2000

    def __init__(self, clock: Optional[Callable[[], float]] = None):
        # Die Uhr (Abnahme 13642): Standard Wanduhr; wer Ereigniszeit
        # hat, reicht sie JE AUFRUF herein (der Engine-Weg). Eine
        # konfigurierte Uhr ist der Rueckfall, wenn kein Ereignis-Zeit-
        # stempel vorliegt — so bleibt der Live-Betrieb unveraendert.
        self._clock = clock or time.time
        self.events = deque(maxlen=self.MAX_EVENTS)
        self.arousal = 0.2
        # Gleitfenster (Schritt 6c): dieselbe Zaehlung wie der Bestands-
        # filter ueber ``self.events``, aber inkrementell gepflegt.
        # ``None`` = Neuaufbau faellig. Das Fenster haengt an
        # ``self.events``: wer ``events`` ausserhalb von ``observe_event``
        # ersetzt (so wie ``from_dict``), setzt es auf ``None``.
        self._fenster: Optional[deque] = None
        # Nicht-endliche Werte im Deque (NaN/inf; ueber die Engine nicht
        # erreichbar, die Klasse soll aber auch direkt gerufen dasselbe
        # sehen wie der Bestand): solange einer drin liegt, wird neu
        # aufgebaut, denn seine Fensterzugehoerigkeit springt mit now.
        self._nicht_endlich = 0

    def observe_event(self, now: Optional[float] = None) -> None:
        """Ein Ereignis zaehlen. ``now`` = Ereigniszeit (Epoch-Sekunden);
        ohne Angabe gilt die Uhr (Standard: Wanduhr)."""
        if now is None:
            now = self._clock()
        # Was die volle Ereignis-Deque gleich verdraengt, VOR dem
        # Anhaengen merken: lag es noch im Fenster, muss es dort
        # gestrichen werden (sonst zaehlte der Puffer es weiter).
        verdraengt = (self.events[0]
                      if len(self.events) == self.MAX_EVENTS else None)
        if verdraengt is not None and not _endlich(verdraengt):
            self._nicht_endlich -= 1
        self.events.append(now)
        if not _endlich(now):
            self._nicht_endlich += 1
        self._recompute(now, verdraengt)

    def _recompute(self, now: float, verdraengt: Optional[float] = None):
        # Bestand: bei JEDEM Ereignis die letzten 2000 Zeitstempel
        # durchzaehlen (O(2000) je Ereignis; im Profil der teuerste
        # Einzelposten). Das Fenster ist dieselbe Menge, haelt sich aber
        # selbst aktuell: laeuft die Ereigniszeit vorwaerts, werden links
        # die abgelaufenen abgeschnitten und rechts wird angehaengt; ein
        # Element, das die Ereignis-Deque verdraengt und noch im Fenster
        # lag, wird dort gestrichen. Ein Neuaufbau (die Bestandsformel
        # woertlich) steht nur bei (a) erstem Aufruf / from_dict,
        # (b) Ruecksprung der Ereigniszeit (gemischte Stroeme mit/ohne
        # Zeitstempel) und (c) nicht-endlichen Werten im Deque. Die
        # Zaehlung ist damit in JEDEM Fall identisch zur Bestandsformel;
        # im gemessenen, monotonen Spur-Strom laeuft ausschliesslich der
        # O(1)-Weg (der auch dichte Strome traegt — gemessen).
        neu_aufbau = (
            self._fenster is None
            or self._nicht_endlich
            or (self._fenster and now < self._fenster[-1])
        )
        if not neu_aufbau:
            while (self._fenster
                   and now - self._fenster[0] > self.WINDOW_SECONDS):
                self._fenster.popleft()
            if (verdraengt is not None
                    and now - verdraengt <= self.WINDOW_SECONDS):
                # Das Verdraengte liegt noch im Fenster und ist dort
                # beweisbar vorhanden (Fenster = Bestandsformel ueber
                # events). Faellt die Annahme je, ist der Neuaufbau die
                # Bestandsformel selbst — nie ein falsches Ergebnis.
                try:
                    self._fenster.remove(verdraengt)
                except ValueError:
                    neu_aufbau = True
        if neu_aufbau:
            self._fenster = deque(
                sorted(t for t in self.events
                       if now - t <= self.WINDOW_SECONDS),
                maxlen=self.MAX_EVENTS,
            )
        else:
            self._fenster.append(now)
        density = len(self._fenster) / float(self.WINDOW_SECONDS)
        target = min(1.0, density * 2.5)
        self.arousal = self.arousal * 0.9 + target * 0.1

    def get_arousal(self) -> float:
        return max(0.0, min(1.0, self.arousal))

    def to_dict(self) -> dict:
        return {"events": list(self.events), "arousal": self.arousal}

    def from_dict(self, data: dict):
        self.events = deque(data.get("events", [])[-2000:],
                            maxlen=self.MAX_EVENTS)
        self.arousal = float(data.get("arousal", 0.2))
        # Das Fenster haengt an self.events — nach dem Ersetzen neu aufbauen.
        self._fenster = None
        self._nicht_endlich = sum(1 for t in self.events if not _endlich(t))
