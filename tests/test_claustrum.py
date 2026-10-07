"""Tests für das Claustrum — die Börse der Vorhersagen (Stufe 3, #2).

Die Zusagen gegen die dummen Gegner stehen im Messstand
(``test_spur_messstand.py``); hier stehen die Bauteile: die Kette, der
Mischer, die externe Stimme, die Abschaltprobe und die bitgleiche
Fortsetzung nach dem Laden.
"""
from __future__ import annotations

import json
import math
import random
from datetime import datetime, timedelta, timezone

from kontinuum_core.claustrum import (
    EXTERN_DECKEL,
    EXPERTEN,
    Claustrum,
    Mischer,
    Sequenz,
)

START = datetime(2026, 1, 5, 6, 0, tzinfo=timezone.utc)  # ein Montag


def _zeit(n: int) -> datetime:
    return START + timedelta(minutes=3 * n)


def test_sequenz_lernt_eine_kette_und_bleibt_offen_fuer_neues():
    s = Sequenz(ordnung=3)
    for _ in range(20):
        for y in ("a", "b", "c"):
            s.lerne(y)
    s.bereite()  # der Verlauf endet auf „c“ → „a“ folgt
    assert s.kandidaten(3)[0] == "a"
    assert s.p("a") > 0.9
    # Ein nie gesehener Token bleibt möglich, aber unwahrscheinlich.
    assert 0.0 < s.p("x") < 0.05
    assert s.belege("a") >= 19


def test_mischer_verschiebt_gewicht_zum_treffenden_experten():
    """Experte „gut“ gibt dem Ziel immer 80 %, „rauschen“ würfelt. Nach
    300 Ereignissen trägt „gut“ das Gewicht, „rauschen“ nicht."""
    m = Mischer(("gut", "rauschen"), start={})
    wuerfel = random.Random(3)
    for _ in range(300):
        ziel = wuerfel.randrange(3)
        gut = [math.log(0.8 if i == ziel else 0.1) for i in range(3)]
        zufall = wuerfel.randrange(3)
        rauschen = [math.log(0.8 if i == zufall else 0.1) for i in range(3)]
        logp = [[gut[i], rauschen[i]] for i in range(3)]
        q = m.mische("k", logp)
        m.lerne("k", logp, q, ziel)
    w_gut, w_rauschen = m.gewichte("k")
    assert w_gut > 0.5
    assert w_gut > w_rauschen + 0.3


def _kette(c: Claustrum, runden: int = 30) -> None:
    n = 0
    for _ in range(runden):
        for y in ("a", "b", "c"):
            c.beobachte(y, _zeit(n), kontext="licht")
            n += 1


def test_eine_externe_stimme_ueberstimmt_nicht_alles():
    """Rückfall-Test zum 06.10.2026: Die externe Stimme war ungeglättet —
    „nicht genannt“ hieß log(EPS), und eine einzige überfällige Kadenz
    drückte alles andere auf 0 %. Jetzt ist sie ein gedeckelter Lift:
    Nicht Genannte bleiben neutral (0), Genannte höchstens EXTERN_DECKEL."""
    c = Claustrum()
    _kette(c)
    i_extern = EXPERTEN.index("extern")
    c.mischer.global_w[i_extern] = 1.0  # eine voll vertraute Stimme
    liste = c.beobachte("a", _zeit(1000), kontext="licht", extern={"fremd": 1.0})
    q = dict(liste)
    # Die gelernte Kette (a → b) bleibt vorn; die Stimme verschiebt, sie
    # entscheidet nicht allein.
    assert liste[0][0] == "b"
    assert q["fremd"] < 0.5
    _, kandidaten, logp, _ = c._stand
    for y, zeile in zip(kandidaten, logp):
        if y == "fremd":
            assert 0.0 < zeile[i_extern] <= EXTERN_DECKEL
        else:
            assert zeile[i_extern] == 0.0, y


def test_abschaltprobe_laesst_experten_stumm():
    c = Claustrum(aus=("zeit", "folgezeit"))
    _kette(c, runden=5)
    _, _, logp, _ = c._stand
    i_zeit = EXPERTEN.index("zeit")
    i_folge = EXPERTEN.index("folgezeit")
    assert all(z[i_zeit] == 0.0 and z[i_folge] == 0.0 for z in logp)


def test_ueberraschung_in_bits():
    c = Claustrum()
    _kette(c)
    c.beobachte("a", _zeit(500), kontext="licht")
    c.beobachte("b", _zeit(501), kontext="licht")  # erwartet
    erwartet = c.letzte_ueberraschung
    c.beobachte("a", _zeit(502), kontext="licht")  # nach b kommt sonst c
    unerwartet = c.letzte_ueberraschung
    assert erwartet < 1.0 < unerwartet
    assert 0.5 < c.trefferquote <= 1.0


def test_round_trip_setzt_bitgleich_fort():
    """Gespeichert, als JSON geschrieben, neu geladen: Die geladene Börse
    sagt danach Ereignis für Ereignis genau dasselbe wie die durchgelaufene
    — samt offenem Vorhersage-Stand, aus dem der Mischer als Nächstes lernt."""
    wuerfel = random.Random(7)
    folge = ["licht.an", "licht.aus", "tv.an", "tv.aus", "tuer.auf", "tuer.zu"]
    strom = [(wuerfel.choice(folge), _zeit(i), f"k{i % 3}") for i in range(600)]
    a = Claustrum()
    for y, t, k in strom[:400]:
        a.beobachte(y, t, kontext=k, extern={folge[0]: 0.6})
    b = Claustrum()
    b.from_dict(json.loads(json.dumps(a.to_dict())))
    for y, t, k in strom[400:]:
        la = a.beobachte(y, t, kontext=k, extern={folge[0]: 0.6})
        lb = b.beobachte(y, t, kontext=k, extern={folge[0]: 0.6})
        assert la == lb
        assert a.letzte_ueberraschung == b.letzte_ueberraschung
    assert a.mischer.global_w == b.mischer.global_w
    assert a.trefferquote == b.trefferquote
