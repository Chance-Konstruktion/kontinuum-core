"""Regressionstest zu MR !16: Decay-Pruning und die Summen-Gleichung.

Die Auslagerung liest ihre Rangfolge aus ``totals`` statt aus
``sum(transitions[ngram].values())`` (Schritt 6c). Damit die beiden
Rechnungen dieselbe Auswahl treffen, zieht ``_apply_decay`` beim Prunen
die gestrichene Masse von ``totals`` ab -- sonst wöge die Auslagerung den
toten Schweif mit (Kimis Befund auf MR !16; ohne den Abzug wichen in
dieser Szene 6 von 14 N-Grammen ab).

Der Test lässt den Prune WIRKLICH feuern (Mini-Gewichte, langer Decay)
und prüft drei Dinge:

1. ``totals[ng] == sum(transitions[ng].values())`` (Toleranz 1e-9),
2. die gestrichenen N-Gramme entsprechen der alten ``sum()``-Formel,
3. nach der Auslagerung tragen ``transitions`` und ``totals`` dieselben
   Schlüssel (kein Rest im Summenbuch).

Ohne die Härtung in ``5ee65baf`` fällt Punkt 1 (die gestrichene Masse
lebte in ``totals`` weiter) — der Test hat also Zähne.
"""
from __future__ import annotations

import heapq
from datetime import datetime, timezone

from kontinuum_core.hippocampus import Hippocampus


def _ctx():
    """Ein Kontextvektor mit mindestens sieben Dimensionen (ein Eimer)."""
    return [0.0, 1.0, 0.0, 1.0, 0.5, 0.5, 0.0]


def _szene_mit_prune() -> Hippocampus:
    """Baut eine Szene, in der der Decay-Pruning wirklich feuert.

    ``learn_weight`` wird auf [0.1, 3.0] geklemmt; 0.1 ist das Minimum.
    Nach ``_apply_decay(400)`` (0.993**400 ≈ 0.06) fallen die dünnen
    Ziele 9 unter ``DECAY_PRUNE_THRESHOLD`` (0.05), während die dicken
    Ziele 7 überleben -- genau der Fall, den die Härtung abdeckt.
    """
    h = Hippocampus()
    ctx = _ctx()
    t = datetime(2011, 1, 1, 12, 0, tzinfo=timezone.utc)
    for _ in range(40):
        h.learn(7, ctx, t, learn_weight=3.0)
    for _ in range(3):
        h.learn(9, ctx, t, learn_weight=0.1)
    for _ in range(30):
        h.learn(7, ctx, t, learn_weight=3.0)
    h._apply_decay(400)
    return h


def test_decay_prune_haelt_die_summen_gleichung():
    h = _szene_mit_prune()
    bucket = h._context_bucket(_ctx())
    trans, tots = h.transitions[bucket], h.totals[bucket]
    assert trans, "die Szene muss N-Gramme tragen"

    # Die Szene muss den Prune wirklich gefeuert haben, sonst prüft der
    # Test die falsche Sache (Ziel 9 fehlt nach dem Prune).
    assert all(9 not in ziele for ziele in trans.values())

    for ng, ziele in trans.items():
        assert abs(tots[ng] - sum(ziele.values())) < 1e-9, ng
        assert tots[ng] > 0.0, ng


def test_auslagerung_streicht_wie_die_alte_summen_formel():
    h = _szene_mit_prune()
    bucket = h._context_bucket(_ctx())
    trans, tots = h.transitions[bucket], h.totals[bucket]

    # Die alte Rechnung als Erwartung: die kleinsten nach
    # sum(transitions[ngram].values()). Achtung: `trans` ist das LEBENDE
    # Dict -- der Schnappschuss muss vor der Auslagerung fallen.
    vorher = set(trans)
    zuviel = max(1, len(trans) - 7)
    erwartet = {ng for _, ng in heapq.nsmallest(
        zuviel, [(sum(ziele.values()), ng) for ng, ziele in trans.items()])}

    h.MAX_NGRAMS_PER_BUCKET = 7    # erzwingt die Auslagerung
    h._evict_bucket(bucket)

    assert len(h.transitions[bucket]) == 7
    gestrichen = vorher - set(h.transitions[bucket])
    assert gestrichen == erwartet
    # Kein Rest im Summenbuch: dieselben Schlüssel hüben wie drüben.
    assert set(h.totals[bucket]) == set(h.transitions[bucket])
