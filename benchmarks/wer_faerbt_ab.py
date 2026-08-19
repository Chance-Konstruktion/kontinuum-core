#!/usr/bin/env python3
"""Welcher Test hinterlaesst Zustand, der den Benchmark verschiebt?

Der Befund vom 2026-08-19: Derselbe Aufruf misst

    einzeln aufgerufen           Fehlalarmrate 0.037879
    in der vollen Testsuite      Fehlalarmrate 0.060606

auf derselben Maschine, im selben Commit. Der Unterschied kann also nur
davon kommen, was vorher im selben Prozess gelaufen ist.

Dieses Skript laesst den Benchmark-Test einmal je Testdatei laufen --
immer nur diese eine Datei davor, sonst nichts. Faellt er nach genau
einer davon um, ist der Verursacher gefunden.

    python3 benchmarks/wer_faerbt_ab.py

Warum das ueberhaupt zaehlt: Ein Test, dessen Ergebnis von seinen
Vorgaengern abhaengt, misst nicht mehr das, was in seinem Namen steht.
Er ist gruen oder rot, je nachdem, wer vorher dran war -- und beim
naechsten Umbenennen einer Datei springt er um, ohne dass sich am Code
etwas geaendert hat.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

WURZEL = Path(__file__).resolve().parents[1]
ZIEL = ("tests/test_benchmark.py::"
        "test_a_restless_but_normal_home_does_not_set_off_the_flag")


def lauf(*teile: str) -> tuple[bool, str]:
    erg = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-p",
         "no:cacheprovider", *teile],
        cwd=WURZEL, capture_output=True, text=True)
    zeile = ""
    for z in erg.stdout.splitlines():
        if "false alarms on a normal" in z or "false_alarm_rate" in z:
            zeile = z.strip()[:120]
    return erg.returncode == 0, zeile


def main() -> int:
    allein_ok, allein_text = lauf(ZIEL)
    print(f"allein: {'gruen' if allein_ok else 'ROT'}  {allein_text}")
    if not allein_ok:
        print()
        print("Der Test faellt schon allein um. Dann liegt es nicht an einem")
        print("Vorgaenger, sondern am Test selbst -- und dieses Skript hat")
        print("hier nichts mehr zu suchen.")
        return 1

    dateien = sorted(p.name for p in (WURZEL / "tests").glob("test_*.py")
                     if p.name != "test_benchmark.py")
    schuldige = []
    print(f"\n{len(dateien)} Dateien einzeln davorgeschaltet:\n")
    for name in dateien:
        ok, text = lauf(f"tests/{name}", ZIEL)
        marke = "gruen" if ok else "ROT  "
        print(f"  {marke}  {name:<38} {text}")
        if not ok:
            schuldige.append(name)

    print()
    if schuldige:
        print(f"Verursacher: {', '.join(schuldige)}")
        print("Diese Datei hinterlaesst Zustand, den der Benchmark sieht.")
        return 1
    print("Keine einzelne Datei kippt ihn -- es braucht mehrere zusammen.")
    print("Dann ist der naechste Schritt eine Halbierung ueber die Liste,")
    print("nicht das Durchgehen einzelner Dateien.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
