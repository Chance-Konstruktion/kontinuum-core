#!/usr/bin/env python3
"""Laeuft die Testsuite zwoelfmal -- kommt zwoelfmal dasselbe heraus?

Pipeline 377 auf ``main`` meldete am 2026-08-19 um 06:35 UTC eine
Fehlalarmrate von 0.060606 und war rot. Vier andere Laeufe derselben
CI, derselbe Code, melden 0.037879 und sind gruen. Der rote lief,
waehrend 32 andere Jobs den Runner auslasteten.

Ausgeschlossen sind bereits, jeweils mit Messung statt Vermutung:
der gesaete Zufall (Folge Zeichen fuer Zeichen gleich), die Uhrzeit
(eingefroren, mit Kontrollausgabe), die verstrichene Echtzeit
(``time.time()`` schrittweise vorgestellt, bis zu 1 s je Aufruf), eine
fortschreitende ``datetime.now()``, ``PYTHONHASHSEED`` und die
Testreihenfolge (jede Datei einzeln davor, die eigene Datei komplett,
alle neunzehn zusammen).

Bleibt die Frage, die man nicht durch Nachdenken beantwortet: Wackelt
es? Zwoelf Laeufe sagen mehr als zwoelf Hypothesen.

    python3 benchmarks/wackelt_es.py

Ein Test, der bei jedem zwanzigsten Lauf umfaellt, ist schlimmer als
einer, der immer rot ist. Er erzieht die Leute dazu, auf "Wiederholen"
zu druecken -- und irgendwann druecken sie das auch, wenn er recht hat.
"""

from __future__ import annotations

import collections
import re
import subprocess
import sys
from pathlib import Path

WURZEL = Path(__file__).resolve().parents[1]
LAEUFE = 12
MUSTER = re.compile(r"false alarms on a normal-but-restless home: ([\d.]+)")


def einmal() -> tuple[bool, str]:
    erg = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-p",
         "no:cacheprovider", "tests"],
        cwd=WURZEL, capture_output=True, text=True)
    treffer = MUSTER.search(erg.stdout)
    if treffer:
        return erg.returncode == 0, treffer.group(1)
    letzte = [z for z in erg.stdout.strip().splitlines() if z.strip()]
    return erg.returncode == 0, (letzte[-1][:60] if letzte else "?")


def main() -> int:
    ergebnisse = []
    for i in range(1, LAEUFE + 1):
        ok, wert = einmal()
        print(f"  Lauf {i:>2}: {'gruen' if ok else 'ROT  '}  {wert}")
        ergebnisse.append(ok)

    rot = ergebnisse.count(False)
    print()
    if rot == 0:
        print(f"{LAEUFE} von {LAEUFE} gruen. Der eine rote Lauf von 06:35 hat")
        print("sich hier nicht wiederholt -- damit ist er nicht erklaert,")
        print("sondern nur selten. Das ist ein Unterschied, den man")
        print("aufschreiben muss und nicht abhaken darf.")
        return 0
    if rot == LAEUFE:
        print(f"{LAEUFE} von {LAEUFE} rot. Dann wackelt nichts, sondern es ist")
        print("schlicht kaputt -- und der gruene Lauf war der Ausreisser.")
        return 1
    print(f"{rot} von {LAEUFE} rot. Der Test wackelt, aus demselben Commit.")
    print("Eine Zahl, die von Lauf zu Lauf springt, kann keine Konstante")
    print("begruenden -- weder die Schwelle noch die Grenze im Test selbst.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
