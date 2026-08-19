#!/usr/bin/env python3
"""Welcher Vorlauf verschiebt den Benchmark?

Der Befund vom 2026-08-19: Derselbe Aufruf misst

    einzeln aufgerufen           Fehlalarmrate 0.037879
    in der vollen Testsuite      Fehlalarmrate 0.060606

auf derselben Maschine, im selben Commit. Der Unterschied kann also nur
davon kommen, was vorher im selben Prozess gelaufen ist.

Der erste Versuch schaltete jede Testdatei einzeln davor -- neunzehnmal
gruen. Es liegt also nicht an einer einzelnen Datei. Bleiben zwei
Moeglichkeiten, und dieses Skript prueft beide:

1. Ein **Nachbar in derselben Datei**. ``test_benchmark.py`` hat sechs
   Tests; laeuft der Recall-Test vorher, ist der Prozess nicht mehr
   frisch.
2. **Mehrere Dateien zusammen.** Dafuer wird halbiert statt einzeln
   durchgegangen: Menge testen, bei Rot in zwei Haelften teilen, mit der
   roten weitermachen. Aus neunzehn Laeufen werden fuenf.

    python3 benchmarks/wer_faerbt_ab.py

Warum das ueberhaupt zaehlt: Ein Test, dessen Ergebnis von seinen
Vorgaengern abhaengt, misst nicht mehr das, was in seinem Namen steht.
Er ist gruen oder rot, je nachdem wer vorher dran war -- und beim
naechsten Umbenennen einer Datei springt er um, ohne dass sich am Code
etwas geaendert hat.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

WURZEL = Path(__file__).resolve().parents[1]
DATEI = "tests/test_benchmark.py"
ZIEL = DATEI + "::test_a_restless_but_normal_home_does_not_set_off_the_flag"


def lauf(*teile: str) -> tuple[bool, str]:
    erg = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-p",
         "no:cacheprovider", *teile],
        cwd=WURZEL, capture_output=True, text=True)
    gemessen = ""
    for z in erg.stdout.splitlines():
        if "false alarms on a normal" in z:
            gemessen = z.strip()[:110]
    return erg.returncode == 0, gemessen


def halbieren(kandidaten: list[str]) -> list[str]:
    """Kleinste Teilmenge finden, die den Zieltest noch kippt."""
    if len(kandidaten) <= 1:
        return kandidaten
    mitte = len(kandidaten) // 2
    for haelfte in (kandidaten[:mitte], kandidaten[mitte:]):
        ok, _ = lauf(*[f"tests/{n}" for n in haelfte], ZIEL)
        print(f"    {'gruen' if ok else 'ROT  '}  {len(haelfte):>2} Dateien: "
              f"{', '.join(n[5:-3] for n in haelfte)}")
        if not ok:
            return halbieren(haelfte)
    # Keine Haelfte allein kippt ihn -- es braucht Stuecke aus beiden.
    # Weiter zu halbieren waere hier falsch: Die Antwort ist die ganze
    # Menge, und das ist eine Aussage und kein Fehlschlag.
    return kandidaten


def main() -> int:
    ok, text = lauf(ZIEL)
    print(f"1. nur der Zieltest             {'gruen' if ok else 'ROT'}  {text}")
    if not ok:
        print("\nEr faellt schon allein um -- dann liegt es nicht am Vorlauf.")
        return 1

    ok_datei, text_datei = lauf(DATEI)
    print(f"2. ganze test_benchmark.py      "
          f"{'gruen' if ok_datei else 'ROT'}  {text_datei}")

    dateien = sorted(p.name for p in (WURZEL / "tests").glob("test_*.py")
                     if p.name != "test_benchmark.py")
    ok_alle, text_alle = lauf(*[f"tests/{n}" for n in dateien], ZIEL)
    print(f"3. alle {len(dateien)} anderen + Ziel     "
          f"{'gruen' if ok_alle else 'ROT'}  {text_alle}")

    if not ok_datei and ok_alle:
        print("\nDer Verursacher sitzt in derselben Datei: Ein Nachbartest in")
        print("test_benchmark.py laeuft vorher und hinterlaesst etwas.")
        return 1

    if ok_alle:
        print("\nWeder die Nachbarn noch die anderen Dateien kippen ihn.")
        print("Dann kommt der Unterschied aus etwas anderem als der")
        print("Reihenfolge -- und diese Spur ist zu Ende.")
        return 0

    print("\n4. halbieren:")
    kleinste = halbieren(dateien)
    print(f"\nKleinste Menge, die ihn kippt ({len(kleinste)}):")
    for n in kleinste:
        print(f"   {n}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
