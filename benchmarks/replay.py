"""Replay benchmark: does the engine actually separate anomalies from routine?

Unit tests prove the engine *runs*; this harness measures whether it *learns
something useful*. It synthesises a realistic multi-room household routine
(motion sensors firing on a daily schedule), trains the engine on it for a
number of days, then replays a held-out period in which a handful of clearly
out-of-distribution events are injected (motion in the kitchen at 03:00, in the
bedroom at 14:00, ...). For every evaluated event we record the engine's
``surprise`` and its boolean ``anomaly`` flag, then report how well surprise
separates injected anomalies from normal routine.

Run it directly::

    python benchmarks/replay.py

It exits non-zero if separation collapses (AUC <= 0.5), so it doubles as a
smoke gate. The reusable pieces are imported by ``tests/test_benchmark.py``.
"""
from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

# Allow `python benchmarks/replay.py` without an editable install.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kontinuum_core import KontinuumEngine  # noqa: E402

# --------------------------------------------------------------------------
# Scenario
# --------------------------------------------------------------------------
# A learnable daily routine: (hour, minute, room). Each entry fires the room's
# motion sensor on, then off two minutes later.
ROUTINE: List[Tuple[int, int, str]] = [
    (7, 0, "bedroom"),
    (7, 10, "bathroom"),
    (7, 25, "kitchen"),
    (7, 45, "living"),
    (8, 5, "hallway"),
    (12, 30, "kitchen"),
    (13, 0, "living"),
    (18, 0, "kitchen"),
    (19, 0, "living"),
    (21, 30, "bathroom"),
    (22, 0, "bedroom"),
]

# Out-of-distribution events: (room, time) combinations the routine never
# produces. Same sensors, impossible moments.
ANOMALIES: List[Tuple[int, int, str]] = [
    (3, 0, "kitchen"),    # nobody cooks at 3am
    (4, 0, "bathroom"),   # ...
    (14, 0, "bedroom"),   # bedroom is idle every afternoon
]

# A *different* learnable routine, for the concept-drift stress test. The home's
# schedule changes (e.g. a new resident / shift work): same sensors, new times
# and room order. A good engine should be surprised right after the switch, then
# re-learn the new routine and settle back down.
ROUTINE_B: List[Tuple[int, int, str]] = [
    (10, 0, "living"),
    (10, 30, "kitchen"),
    (13, 0, "bedroom"),
    (15, 0, "bathroom"),
    (16, 0, "hallway"),
    (20, 0, "living"),
    (23, 30, "kitchen"),
    (1, 0, "bathroom"),
    (2, 0, "bedroom"),
]

ROOMS = sorted({r for *_, r in ROUTINE})


def _sensor(room: str) -> str:
    return f"binary_sensor.motion_{room}"


def _build_engine() -> KontinuumEngine:
    e = KontinuumEngine()
    # Sleep consolidation now fires during the (event-time) replay and its
    # dream-replay phase samples at random. Give it a seeded RNG so this
    # quality-gate replay stays deterministic instead of depending on the
    # global ``random`` state.
    e.sleep_consolidation._rng = random.Random(0)
    for room in ROOMS:
        e.register_entity(_sensor(room), ha_area=room, domain="binary_sensor")
    return e


def _emit(engine: KontinuumEngine, room: str, ts: datetime):
    """Fire a room's motion sensor on then off; return processed surprises."""
    out = []
    for state, delta in (("on", 0), ("off", 2)):
        snap = engine.observe({
            "entity_id": _sensor(room),
            "new_state": state,
            "timestamp": ts + timedelta(minutes=delta),
        })
        if "skipped" not in snap.extra:
            out.append(snap)
    return out


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def roc_auc(labels: List[int], scores: List[float]) -> float:
    """Mann-Whitney AUC: P(score(anomaly) > score(normal)). 0.5 == chance."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


@dataclass
class BenchmarkResult:
    n_normal: int = 0
    n_anomaly: int = 0
    mean_surprise_normal: float = 0.0
    mean_surprise_anomaly: float = 0.0
    auc: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    labels: List[int] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    # Die tatsaechlich gesetzten Flags. Ohne sie laesst sich die
    # Fehlalarm-Seite gar nicht messen: Precision braucht Anomalien, und
    # ein ruhiges Zuhause hat keine. Genau dafuer ist die untere Klammer
    # da -- und genau das war bisher nicht pruefbar.
    flags: List[bool] = field(default_factory=list)

    @property
    def false_alarm_rate(self) -> float:
        """Anteil der normalen Ereignisse, die faelschlich Alarm ausloesen.

        Anders als Precision auch dann definiert, wenn es ueberhaupt keine
        Anomalien gibt -- und das ist der Fall, den die untere Klammer
        schuetzen soll.
        """
        normal = [f for f, l in zip(self.flags, self.labels) if l == 0]
        return sum(normal) / len(normal) if normal else 0.0

    def report(self) -> str:
        return (
            "KONTINUUM Core — replay benchmark\n"
            "---------------------------------\n"
            f"samples            : {self.n_normal} normal / {self.n_anomaly} anomaly\n"
            f"mean surprise      : {self.mean_surprise_normal:.3f} normal  "
            f"vs {self.mean_surprise_anomaly:.3f} anomaly\n"
            f"separation (AUC)   : {self.auc:.3f}   (0.5 = chance, 1.0 = perfect)\n"
            f"anomaly-flag P/R/F1: {self.precision:.2f} / {self.recall:.2f} / {self.f1:.2f}\n"
        )


def run_benchmark(train_days: int = 40, eval_days: int = 12,
                  start: datetime | None = None,
                  jitter_minutes: int = 0, with_anomalies: bool = True,
                  seed: int = 7) -> BenchmarkResult:
    """Replay-Benchmark. Zwei Schalter, die vorher gefehlt haben.

    ``jitter_minutes`` streut die Routine in der Auswertung um bis zu
    N Minuten -- gelernt wird weiter auf die Minute genau. Das ist ein
    unruhiges, aber voellig normales Zuhause: mal fuenf Minuten spaeter
    ins Bad. Nichts davon ist eine Anomalie.

    ``with_anomalies=False`` spielt gar keine Anomalien ein.

    Beides zusammen macht die **Fehlalarm-Seite** ueberhaupt erst
    messbar. Bisher lief hier immer dieselbe glatte Routine mit
    eingestreuten Anomalien -- und Precision ist ohne Anomalien nicht
    definiert. Die untere Klammer ANOMALY_MIN_THRESHOLD existiert genau
    fuer diesen Fall, und ihr Nutzen war damit nie nachpruefbar.

    Die Streuung ist geseedet: gleicher ``seed``, gleiches Ergebnis.
    """
    import random as _random

    wuerfel = _random.Random(seed)

    def _versatz() -> timedelta:
        if not jitter_minutes:
            return timedelta(0)
        return timedelta(minutes=wuerfel.randint(-jitter_minutes, jitter_minutes))

    engine = _build_engine()
    start = start or datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    # --- train on the routine only ---
    day = start
    for _ in range(train_days):
        for hh, mm, room in ROUTINE:
            _emit(engine, room, day.replace(hour=hh, minute=mm))
        day += timedelta(days=1)

    # --- evaluate: routine (label 0) interleaved with anomalies (label 1) ---
    labels: List[int] = []
    scores: List[float] = []
    flags: List[bool] = []
    for _ in range(eval_days):
        for hh, mm, room in ROUTINE:
            for snap in _emit(engine, room,
                              day.replace(hour=hh, minute=mm) + _versatz()):
                labels.append(0)
                scores.append(snap.surprise)
                flags.append(snap.anomaly)
        for hh, mm, room in (ANOMALIES if with_anomalies else []):
            for snap in _emit(engine, room, day.replace(hour=hh, minute=mm)):
                labels.append(1)
                scores.append(snap.surprise)
                flags.append(snap.anomaly)
        day += timedelta(days=1)

    res = BenchmarkResult(labels=labels, scores=scores, flags=flags)
    norm = [s for s, l in zip(scores, labels) if l == 0]
    anom = [s for s, l in zip(scores, labels) if l == 1]
    res.n_normal, res.n_anomaly = len(norm), len(anom)
    res.mean_surprise_normal = sum(norm) / len(norm) if norm else 0.0
    res.mean_surprise_anomaly = sum(anom) / len(anom) if anom else 0.0
    res.auc = roc_auc(labels, scores)

    tp = sum(1 for f, l in zip(flags, labels) if f and l == 1)
    fp = sum(1 for f, l in zip(flags, labels) if f and l == 0)
    fn = sum(1 for f, l in zip(flags, labels) if not f and l == 1)
    res.precision = tp / (tp + fp) if (tp + fp) else 0.0
    res.recall = tp / (tp + fn) if (tp + fn) else 0.0
    res.f1 = (2 * res.precision * res.recall / (res.precision + res.recall)
              if (res.precision + res.recall) else 0.0)
    return res


@dataclass
class DriftResult:
    baseline: float = 0.0   # steady-state mean surprise on the old routine
    spike: float = 0.0      # mean surprise on the first day of the new routine
    adapted: float = 0.0    # mean surprise on the last day of the new routine
    per_day: List[float] = field(default_factory=list)

    def report(self) -> str:
        days = "  ".join(f"{s:.2f}" for s in self.per_day)
        return (
            "KONTINUUM Core — concept-drift stress test\n"
            "------------------------------------------\n"
            f"baseline surprise (old routine) : {self.baseline:.3f}\n"
            f"spike     (1st day, new routine): {self.spike:.3f}\n"
            f"adapted   (last day, new routine): {self.adapted:.3f}\n"
            f"per-day mean surprise after switch: {days}\n"
        )


def _day_mean_surprise(engine, routine, day) -> float:
    vals = []
    for hh, mm, room in routine:
        for snap in _emit(engine, room, day.replace(hour=hh, minute=mm)):
            vals.append(snap.surprise)
    return sum(vals) / len(vals) if vals else 0.0


def run_drift_benchmark(train_days: int = 40, drift_days: int = 20,
                        start: datetime | None = None) -> DriftResult:
    """Train on routine A, switch to routine B, watch detection + re-adaptation."""
    engine = _build_engine()
    start = start or datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    day = start
    for _ in range(train_days):
        for hh, mm, room in ROUTINE:
            _emit(engine, room, day.replace(hour=hh, minute=mm))
        day += timedelta(days=1)

    # Steady-state baseline on the *old* routine (what "normal" feels like now).
    baseline = _day_mean_surprise(engine, ROUTINE, day)
    day += timedelta(days=1)

    # Switch the world to routine B and watch surprise rise then re-settle.
    per_day = []
    for _ in range(drift_days):
        per_day.append(_day_mean_surprise(engine, ROUTINE_B, day))
        day += timedelta(days=1)

    return DriftResult(
        baseline=baseline, spike=per_day[0], adapted=per_day[-1], per_day=per_day,
    )


def main() -> int:
    res = run_benchmark()
    print(res.report())
    drift = run_drift_benchmark()
    print(drift.report())
    ok = res.auc > 0.5 and drift.spike > drift.baseline and drift.adapted < drift.spike
    if not ok:
        print("FAIL: anomaly separation or drift adaptation collapsed.")
        return 1
    print("OK: anomalies separate from routine; drift is detected and re-adapted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
