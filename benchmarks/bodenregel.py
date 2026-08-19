#!/usr/bin/env python3
"""Welche Untergrenze haelt auch auf einem anderen Rechner?

``anomaly_threshold()`` rechnet ``Median + 3 x 1.4826 x MAD`` und klemmt
das Ergebnis nach unten auf eine feste Zahl. Genau diese Zahl ist das
Problem: Auf Windows misst derselbe Code eine mittlere Surprise von
0.0772, in der CI 0.0600 -- rund 22 % Unterschied, bei einer AUC, die
auf sechs Stellen uebereinstimmt. Die Rangfolge ist ueberall gleich, nur
die Betraege wandern. Eine feste Untergrenze schneidet dadurch an
verschiedenen Stellen durch dieselbe Rangliste.

Dieses Skript vergleicht deshalb keine Zahlen, sondern **Regeln**: Eine
Untergrenze, die aus den beobachteten Werten selbst hervorgeht, wandert
mit ihnen mit.

Die entscheidende Spalte ist nicht Recall, sondern **knapp**: wie viele
Anomalien naeher als eine Haaresbreite an der Schwelle liegen. Ueber die
entscheidet auf dem naechsten Rechner eine andere Nachkommastelle. Eine
Regel mit gutem Recall und vielen knappen Faellen ist kein Fund, sondern
ein gluecklicher Wuerfelwurf.

    python3 benchmarks/bodenregel.py

Vergleiche die Ausgabe mit der aus der CI. Eine Regel taugt nur, wenn
sie auf beiden dieselbe Entscheidung trifft.
"""

from __future__ import annotations

import sys
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import kontinuum_core.predictive_processing as pp  # noqa: E402
from benchmarks.replay import run_benchmark  # noqa: E402

HAARESBREITE = 0.01
TRAIN_TAGE, EVAL_TAGE = 40, 12

# Reiner Schutz gegen Null, keine Kalibrierung: Ohne ihn koennte die
# Schwelle in einem vollkommen starren Zuhause auf 0 fallen und jedes
# Ereignis waere eine Anomalie.
NOTBODEN = 0.02
MAD_TO_STD_LOKAL = 1.4826


def _quantil(werte: list[float], q: float) -> float:
    """Perzentil mit linearer Interpolation, ohne numpy."""
    if not werte:
        return 0.0
    s = sorted(werte)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    unten = int(pos)
    oben = min(unten + 1, len(s) - 1)
    rest = pos - unten
    return s[unten] * (1 - rest) + s[oben] * rest


# Jede Regel bekommt die beobachteten Surprise-Werte und liefert die
# Untergrenze. `fest` ist der heutige Stand und dient als Vergleich.
REGELN = {
    "fest 0.07 (heute)": lambda data, med: 0.07,
    "Median x 1.2": lambda data, med: med * 1.2,
    "Median x 1.5": lambda data, med: med * 1.5,
    "Median x 2.0": lambda data, med: med * 2.0,
    "Perzentil 75": lambda data, med: _quantil(data, 0.75),
    "Perzentil 85": lambda data, med: _quantil(data, 0.85),
    "Perzentil 90": lambda data, med: _quantil(data, 0.90),
    "kein Boden": lambda data, med: NOTBODEN,
}


def mit_regel(regel):
    """Ersetzt anomaly_threshold durch dieselbe Rechnung mit anderer Grenze.

    Die tatsaechlich benutzten Schwellen werden mitgeschrieben. Ohne das
    liesse sich "knapp" nicht ausrechnen: Bei einer mitwandernden Grenze
    gibt es keine eine Zahl mehr, gegen die man vergleichen koennte.
    """
    benutzt: list[float] = []

    def anomaly_threshold(self) -> float:
        n = len(self.surprise_history)
        if n < pp.ANOMALY_MIN_SAMPLES:
            return pp.ANOMALY_DEFAULT_THRESHOLD
        data = list(self.surprise_history)
        med = median(data)
        mad = median([abs(x - med) for x in data])
        adaptiv = med + pp.ANOMALY_MAD_FACTOR * pp.MAD_TO_STD * mad
        boden = max(NOTBODEN, regel(data, med))
        wert = min(pp.ANOMALY_MAX_THRESHOLD, max(boden, adaptiv))
        benutzt.append(wert)
        return wert

    return anomaly_threshold, benutzt


def verteilung() -> None:
    """Wie sieht die Surprise-Verteilung auf DIESER Maschine aus?

    Die Regeln oben aendern nichts am Plattform-Unterschied. Dann liegt
    er nicht an der Grenze, sondern an dem, was gemessen wird. Diese
    Zahlen zeigen, ob die Verteilung nur verschoben ist -- dann waere
    jede massstabsfreie Regel die Loesung -- oder ob sie eine andere
    Form hat. Im zweiten Fall rechnet das Modell selbst verschieden, und
    keine Schwellenregel der Welt haelt das zusammen.
    """
    res = run_benchmark(train_days=TRAIN_TAGE, eval_days=EVAL_TAGE)
    normal = [s for s, l in zip(res.scores, res.labels) if l == 0]
    anomal = [s for s, l in zip(res.scores, res.labels) if l == 1]
    med = median(normal)
    mad = median([abs(x - med) for x in normal])
    print()
    print("Verteilung der Surprise-Werte:")
    print(f"  normal   n={len(normal):<4} Median {med:.4f}  MAD {mad:.4f}")
    for q in (0.50, 0.75, 0.90, 0.95, 0.99):
        print(f"           q{int(q*100):<3} {_quantil(normal, q):.4f}")
    print(f"  Anomalie n={len(anomal):<4} Median {median(anomal):.4f}")
    for q in (0.05, 0.10, 0.25, 0.50):
        print(f"           q{int(q*100):<3} {_quantil(anomal, q):.4f}")
    # Der massstabsfreie Abstand: Wie viele MAD liegen zwischen dem
    # normalen Niveau und der untersten Anomalie? Ist diese Zahl auf
    # beiden Maschinen gleich, ist die Verteilung nur skaliert.
    if mad:
        print(f"  Abstand in MAD-Einheiten:")
        print(f"    unterste Anomalie  {(min(anomal) - med) / (MAD_TO_STD_LOKAL * mad):.3f}")
        print(f"    q10 der Anomalien  {(_quantil(anomal, 0.10) - med) / (MAD_TO_STD_LOKAL * mad):.3f}")
        print(f"    q95 der normalen   {(_quantil(normal, 0.95) - med) / (MAD_TO_STD_LOKAL * mad):.3f}")


def main() -> int:
    print(f"Python {sys.version.split()[0]} auf {sys.platform}")
    print(f"Benchmark: train={TRAIN_TAGE} Tage, eval={EVAL_TAGE} Tage\n")
    print(f"{'Regel':<20} {'Recall':>7} {'Prec':>7} {'AUC':>7} "
          f"{'Fehlal.':>8} {'Schwelle':>9} {'knapp':>6}")
    print("-" * 70)

    original = pp.PredictiveProcessing.anomaly_threshold
    try:
        for name, regel in REGELN.items():
            ersatz, benutzt = mit_regel(regel)
            pp.PredictiveProcessing.anomaly_threshold = ersatz

            benutzt.clear()
            mit = run_benchmark(train_days=TRAIN_TAGE, eval_days=EVAL_TAGE)
            typisch = median(benutzt) if benutzt else float("nan")
            anomalien = [s for s, l in zip(mit.scores, mit.labels) if l == 1]
            knapp = sum(1 for s in anomalien
                        if abs(s - typisch) <= HAARESBREITE)

            ohne = run_benchmark(train_days=TRAIN_TAGE, eval_days=EVAL_TAGE,
                                 jitter_minutes=20, with_anomalies=False)

            print(f"{name:<20} {mit.recall:>7.4f} {mit.precision:>7.4f} "
                  f"{mit.auc:>7.4f} {ohne.false_alarm_rate:>8.4f} "
                  f"{typisch:>9.4f} {knapp:>6d}")
    finally:
        pp.PredictiveProcessing.anomaly_threshold = original

    verteilung()

    print()
    print(f"'Schwelle' ist der Median der tatsaechlich benutzten Schwellen.")
    print(f"'knapp' zaehlt Anomalien, die hoechstens {HAARESBREITE} davon")
    print("entfernt liegen -- die Faelle, die auf einem anderen Rechner")
    print("anders ausgehen. Diese Spalte entscheidet, nicht der Recall.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
