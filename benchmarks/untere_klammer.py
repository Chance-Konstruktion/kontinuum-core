#!/usr/bin/env python3
"""Was die UNTERE Klammer kostet -- die Stellschraube, die wirklich wirkt.

Der Regressionstest verlangt ``recall >= 0.6``. Auf Linux kommen 0.5556
heraus, auf Windows 0.7778 bis 0.9028 -- bei praktisch gleicher AUC. Die
Rangfolge der Ueberraschungswerte ist also auf allen Plattformen
dieselbe; nur die Schwelle faellt anders. Das Ticket vermutete die
letzten Nachkommastellen einer ``libm``.

Diese Messung zeigt etwas Genaueres. Wenn man jede Anomalie mit der
Schwelle paart, die in ihrem Moment galt:

    56 getroffen, 16 verpasst
    verpasst: Abstand zur Schwelle  min=0.00068  median=0.00597  max=0.02238
    Schwelle bei ALLEN Verpassten:  0.1

Alle sechzehn scheitern an derselben Zahl, und zwar an
``ANOMALY_MIN_THRESHOLD`` -- der unteren Klammer. Die
Median-plus-MAD-Rechnung liefert dort einen kleineren Wert und wird vom
``max(...)`` angehoben. Nicht die adaptive Statistik entscheidet diese
Faelle, sondern eine Konstante.

Und diese Konstante sitzt mitten in der Anomalie-Wolke: Zwoelf der
sechzehn liegen naeher als 0.01 an ihr. Deshalb entscheidet dort
tatsaechlich Rauschen -- nicht weil die Rechnung wackelt, sondern weil
die Grenze an der dichtesten Stelle liegt.

Dieses Skript faehrt die untere Klammer durch und schreibt neben Recall
und Precision auch auf, WIE KNAPP es jeweils ist: wie viele Anomalien
innerhalb eines Haaresbreite um die Schwelle liegen. Diese Spalte ist
die eigentliche Antwort auf das Ticket. Ein Wert mit hoher Trefferquote
und vielen knappen Faellen ist keine gute Einstellung -- er ist nur ein
gluecklicher Wuerfelwurf, der auf dem naechsten Rechner anders faellt.

    python3 benchmarks/untere_klammer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import kontinuum_core.predictive_processing as pp  # noqa: E402
from benchmarks.replay import run_benchmark  # noqa: E402

# Von deutlich empfindlicher bis ueber den heutigen Wert hinaus. 0.100
# ist der Stand heute.
KLAMMERN = (0.060, 0.070, 0.080, 0.085, 0.090, 0.095, 0.100, 0.110, 0.120)

# Wie nah an der Schwelle gilt als "knapp". 0.01 ist die Groessenordnung,
# in der die drei gemessenen Plattformen auseinanderliegen.
HAARESBREITE = 0.01

TRAIN_TAGE, EVAL_TAGE = 40, 12


def main() -> int:
    original = pp.ANOMALY_MIN_THRESHOLD
    print(f"Python {sys.version.split()[0]} auf {sys.platform}")
    print(f"Benchmark: train={TRAIN_TAGE} Tage, eval={EVAL_TAGE} Tage")
    print(f"heutiger Wert: ANOMALY_MIN_THRESHOLD = {original}\n")
    print(f"{'Klammer':>8}  {'Recall':>7}  {'Precision':>9}  {'AUC':>7}  {'knapp':>6}")
    print("-" * 50)

    try:
        for wert in KLAMMERN:
            pp.ANOMALY_MIN_THRESHOLD = wert
            res = run_benchmark(train_days=TRAIN_TAGE, eval_days=EVAL_TAGE)
            anomalien = [s for s, l in zip(res.scores, res.labels) if l == 1]
            knapp = sum(1 for s in anomalien if abs(s - wert) <= HAARESBREITE)
            marke = "  <- heute" if wert == original else ""
            print(f"{wert:8.3f}  {res.recall:7.4f}  {res.precision:9.4f}  "
                  f"{res.auc:7.4f}  {knapp:6d}{marke}")
    finally:
        pp.ANOMALY_MIN_THRESHOLD = original

    print()
    print(f"'knapp' = Anomalien, die hoechstens {HAARESBREITE} von der Schwelle")
    print("entfernt liegen. Diese Faelle entscheidet auf einem anderen Rechner")
    print("eine andere Zahl. Wer eine Einstellung nur nach Recall waehlt und")
    print("diese Spalte nicht ansieht, waehlt einen Wuerfelwurf.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
