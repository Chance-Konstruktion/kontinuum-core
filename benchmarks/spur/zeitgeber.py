"""Der Zeitgeber: Ereigniszeit fuer den Replay (Messmittel zu Schritt 6).

Im Haus liegen Ereignisse Sekunden auseinander; im Replay rast der Strom
in Millisekunden durch. Jede Wanduhr-Stelle im Kern, die Abstaende oder
Tagesgrenzen misst, sieht damit voellig andere Zeiten:

* Kleinhirn-Regeln (``RULE_COOLDOWN`` 300 s) koennen im Replay praktisch
  nie ein zweites Mal feuern,
* Reticular (5 s), Spatial (60/1800 s), Insula, Basalganglien, Prefrontal
  (10 s) und die Amygdala haben dieselbe Bauart,
* die Tages-Decay des Hippocampus rechnet in Kalendertagen der Wanduhr,
* die circadiane Lernrate zieht ohne ``hour`` die Wanduhr-Stunde
  (``engine.py`` ruft ``modulate_learning`` ohne Stunde), und
* die Cooldowns haengen damit an der LAUFZEIT des Messlaufs — dasselbe
  Programm liefert auf einer beulasteten Maschine andere Zahlen.

Der Zeitgeber schimt die Uhr auf die **Ereigniszeit**: waehrend des
Laufs liefert ``time.time()`` den Zeitstempel des aktuellen Ereignisses,
und ``modulate_learning`` bekommt die Ereignisstunde. Im Haus ist das
die Wahrheit (dort IST der Zeitstempel die Uhr); im Replay ist es die
einzige ehrliche Zeit. Ergebnis: derselbe Strom ⇒ dieselben Zahlen,
unabhaengig von CPU-Last und Uhrzeit des Laufs.

Das ist ein MESSMITTEL, kein Kern-Eingriff: Solange Schritt 6 (Hygiene)
die Semantik nicht in die Module traegt (clock-Parameter wie in !8),
misst der Messstand mit dem Zeitgeber. Faellt der Kern-Umbau, kann der
Zeitgeber ersatzlos verschwinden — er ist ausdruecklich als Schim
gebaut und nicht als zweite Wahrheit.
"""
from __future__ import annotations

import time as _zeit
from datetime import datetime, timezone as _timezone
from typing import Optional

from kontinuum_core.neurorhythms import Neurorhythms


class Zeitgeber:
    """Setzt die Wanduhr der Engine auf die Ereigniszeit.

    Aufruf im Messkern::

        with Zeitgeber() as uhr:
            for ereignis in strom:
                uhr.stelle(ereignis.ts)     # VOR jedem observe/learn
                engine.observe(...)
    """

    def __init__(self) -> None:
        self._jetzt: float = 0.0
        self._moment: Optional[datetime] = None
        self._echte_time = _zeit.time
        self._echtes_modulate = Neurorhythms.modulate_learning

    # -- Bedienung ---------------------------------------------------
    def stelle(self, moment) -> None:
        """Setzt die Ereigniszeit. Akzeptiert datetime (mit Zone) oder
        eine epoch-Zahl in Sekunden."""
        if isinstance(moment, datetime):
            self._moment = moment
            self._jetzt = moment.timestamp()
        else:
            self._jetzt = float(moment)
            self._moment = datetime.fromtimestamp(self._jetzt, tz=_timezone.utc)

    def stunde(self) -> int:
        """Die Stunde des Ereignisses in SEINER Zone — die Hauszeit.
        (Nicht die Prozess-Zeitzone: die Spur traegt ihre Zone im
        Zeitstempel, und die ist die Wahrheit.)"""
        return self._moment.hour

    # -- Kontext -----------------------------------------------------
    def __enter__(self) -> "Zeitgeber":
        uhr = self

        def jetzt() -> float:
            return uhr._jetzt

        _zeit.time = jetzt
        # engine.py ruft modulate_learning ohne Stunde -> das Modul
        # zieht sonst datetime.now().hour (Wanduhr). Hier bekommt es
        # die Ereignisstunde; wer eine Stunde mitgibt, behaelt sie.
        def modulate(self, base_weight: float, hour: Optional[int] = None):
            if hour is None:
                hour = uhr.stunde()
            return uhr._echtes_modulate(self, base_weight, hour)

        Neurorhythms.modulate_learning = modulate
        return self

    def __exit__(self, *args) -> None:
        _zeit.time = self._echte_time
        Neurorhythms.modulate_learning = self._echtes_modulate

    # Kurzformen fuer den linearen Aufruf im Messkern (kein with-Block,
    # damit der Diff zur parallel laufenden !7-Arbeit klein bleibt).
    an = __enter__
    aus = __exit__


__all__ = ["Zeitgeber"]
