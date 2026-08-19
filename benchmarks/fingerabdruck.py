#!/usr/bin/env python3
"""Derselbe Benchmark, dieselben Seeds -- und zwei verschiedene Ergebnisse.

Am 2026-08-19 fiel auf: Die Testsuite meldet auf zwei Rechnern
unterschiedliche Fehler, aus demselben Commit.

    Windows, Python 3.14   recall 0.694   Fehlalarme 3.79%
    CI (Linux, 3.13)       recall >= 0.9  Fehlalarme 6.06%

Beides deterministisch -- fuenf Laeufe hintereinander auf derselben
Maschine liefern jedes Mal dieselbe Zahl. Ausgeschlossen wurden bereits:
die Uhrzeit (eingefroren, aendert nichts), die Hash-Streuung
(PYTHONHASHSEED 0/1/42/12345, aendert nichts), Umgebungsvariablen (es
werden keine gelesen) und Fremdpakete (es gibt keine).

Das ist wichtiger als jede einzelne Zahl: Eine Messung, die von der
Maschine abhaengt, kann keine Konstante begruenden. ANOMALY_MIN_THRESHOLD
wurde auf einem Rechner eingestellt, dessen Zahlen der Rechner, der die
Tests ausfuehrt, nicht reproduziert.

    python3 benchmarks/fingerabdruck.py

Gibt aus, was diese Maschine misst. Zusammen mit der Ausgabe aus der CI
zeigt der Vergleich, WO die Wege auseinanderlaufen -- nicht nur, DASS sie
es tun.
"""

from __future__ import annotations

import os
import platform
import sys

sys.path.insert(0, os.path.dirname(__file__))

from replay import _build_engine, run_benchmark  # noqa: E402


def zufallsprobe() -> list[float]:
    """Liefert der gesaete Zufall auf beiden Maschinen dieselbe Folge?

    Wenn hier schon Unterschiede stehen, ist die Ursache das
    Zufallsmodul und nichts weiter oben. Wenn nicht, liegt es an der
    Verarbeitung -- und diese Zeile schliesst den einfachsten Verdacht
    aus, statt ihn im Raum zu lassen.
    """
    import random
    w = random.Random(7)
    return [round(w.random(), 12) for _ in range(5)]


def main() -> int:
    print(f"Python    {platform.python_version()}  ({sys.platform})")
    print(f"Maschine  {platform.machine()}")
    print(f"Zufall    {zufallsprobe()}")
    print()

    mit = run_benchmark(train_days=40, eval_days=12)
    ohne = run_benchmark(train_days=40, eval_days=12,
                         jitter_minutes=20, with_anomalies=False)

    print("mit Anomalien (train_days=40, eval_days=12, seed=7):")
    print(f"  n_normal            {mit.n_normal}")
    print(f"  n_anomaly           {mit.n_anomaly}")
    print(f"  auc                 {mit.auc:.6f}")
    print(f"  recall              {mit.recall:.6f}")
    print(f"  mean_surprise_norm  {mit.mean_surprise_normal:.9f}")
    print(f"  mean_surprise_anom  {mit.mean_surprise_anomaly:.9f}")
    print()
    print("ohne Anomalien, jitter_minutes=20:")
    print(f"  n_normal            {ohne.n_normal}")
    print(f"  n_anomaly           {ohne.n_anomaly}")
    print(f"  false_alarm_rate    {ohne.false_alarm_rate:.6f}")
    print(f"  mean_surprise_norm  {ohne.mean_surprise_normal:.9f}")
    print()

    # Die erste Stelle, an der sich die Wege trennen koennen: Wie viele
    # Ereignisse kommen ueberhaupt an, und was macht die Maschine daraus?
    # Sind n_normal und n_anomaly gleich und die Surprise-Mittel
    # verschieden, liegt es an der Berechnung. Sind schon die Zahlen
    # verschieden, liegt es am Aufbau des Laufs.
    motor = _build_engine()
    print(f"Motor     {type(motor).__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
