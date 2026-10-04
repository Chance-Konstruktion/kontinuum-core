"""Selbsttests der Gegner B0/B1/B2 (Stufe 1, Protokoll § 3).

Sie sichern die Fairness-Zusagen: deterministische Gleichstaende,
Glaettung mit Summe 1, Rueckfall auf B0 (die Gegner werden dadurch
STAERKER, nicht schwaecher), B1 lernt den Uebergang am Vorhersagemoment,
und das Vokabular ist die harte Kante (kein stilles Dazulernen).
"""
from __future__ import annotations

from datetime import datetime, timedelta

from benchmarks.spur.gegner import B0, B1, B2, Zaehler, gegner
from benchmarks.spur.simulator import ZEITZONE, simuliere
from benchmarks.spur.spur import tokenisieren

T0 = datetime(2026, 3, 2, 6, 0, tzinfo=ZEITZONE)


def test_zaehler_glaettet_und_summiert_zu_eins():
    z = Zaehler(["a", "b", "c"])
    z.beobachte("a")
    z.beobachte("a")
    z.beobachte("b")
    top = z.top(1)[0]
    assert top[0] == "a"
    assert abs(top[1] - 2.5 / 4.5) < 1e-12
    assert abs(sum(p for _, p in z.verteilung()) - 1.0) < 1e-12


def test_gleichstand_ist_lexikografisch():
    z = Zaehler(["b", "a"])
    z.beobachte("b")
    z.beobachte("a")
    assert [t for t, _ in z.verteilung()] == ["a", "b"]


def test_vokabular_ist_die_harte_kante():
    z = Zaehler(["a"])
    try:
        z.beobachte("fremd")
    except ValueError as fehler:
        assert "ausserhalb des Vokabulars" in str(fehler)
        return
    raise AssertionError("Fremdes Token wurde still angenommen")


def test_b0_ist_das_haeufigste_folgeereignis():
    b0 = B0(["a", "b"])
    for token in ("a", "a", "b", "b", "a"):
        b0.beobachte(token)
    assert b0.vorhersage(T0, 1)[0][0] == "a"


def test_b2_lernt_die_kette():
    b2 = B2(["a", "b"])
    folge = ["a", "b"] * 20
    treffer = 0
    for i, token in enumerate(folge):
        if i > 0:
            vorhersage = b2.vorhersage(T0 + timedelta(minutes=i), 1)
            if vorhersage and vorhersage[0][0] == token:
                treffer += 1
        b2.beobachte(token)
    # Nach zwei Ereignissen ist die Kette gelernt: 38 von 39 Treffern.
    assert treffer == len(folge) - 3 + 1, treffer


def test_b2_rueckfall_auf_b0():
    b2 = B2(["a", "b", "c"])
    for token in ("a", "a", "b"):
        b2.beobachte(token)
    # Kontext "c" nie gesehen -> Unigramm (a) statt Rate-Spiel.
    b2.beobachte("c")  # setzt den Kontext
    assert b2.vorhersage(T0, 1)[0][0] == "a"


def test_b1_lernt_den_uebergang_am_vorhersagemoment():
    b1 = B1(["a", "b", "c"])
    for tag in range(2):
        basis = T0 + timedelta(days=tag)
        for stunde, token in ((6, "a"), (7, "b"), (8, "c")):
            b1.beobachte(token, basis.replace(hour=stunde))
    # Um 07:00 folgte bisher immer "c" -> genau das.
    assert b1.vorhersage(T0.replace(hour=7), 1)[0][0] == "c"
    # Um 03:00 wurde nie etwas gesehen -> Rueckfall auf B0.
    assert b1.vorhersage(T0.replace(hour=3), 1)[0][0] == "a"


def test_b1_braucht_die_zeit():
    b1 = B1(["a"])
    try:
        b1.beobachte("a")
    except ValueError as fehler:
        assert "Zeit" in str(fehler)
        return
    raise AssertionError("B1 lernte ohne Zeit — das waere B0 mit Umweg")


def test_gegner_sind_drei_unabhaengige():
    schar = gegner(["a", "b"])
    assert set(schar) == {"B0", "B1", "B2"}
    schar["B0"].beobachte("a")
    assert schar["B2"].vorhersage(T0, 1)[0][0] in ("a", "b")


def test_protokoll_ordnung_auf_der_simulation():
    """Der ehrliche Fluss auf einer geseedeten Simulation:
    vorhersage(ts) -> beobachte(token, ts). Alle drei Gegner muessen
    dabei lernen; die beiden kluegeren duerfen nicht schlechter sein
    als das Unigramm (sonst waere etwas an der Ordnung verkehrt)."""
    sim = simuliere("klassisch", tage=10, saat=3)
    tokens = tokenisieren(sim.spur)
    vokabular = sorted(set(tokens))
    schar = gegner(vokabular)
    treffer = {name: 0 for name in schar}
    gesamt = 0
    for i, token in enumerate(tokens):
        if i > 0:
            zeit = sim.spur.ereignisse[i - 1].ts
            for name, g in schar.items():
                vorhersage = g.vorhersage(zeit, 1)
                if vorhersage and vorhersage[0][0] == token:
                    treffer[name] += 1
            gesamt += 1
        for name, g in schar.items():
            g.beobachte(token, sim.spur.ereignisse[i].ts)
    assert gesamt > 250
    quote = {name: h / gesamt for name, h in treffer.items()}
    assert quote["B2"] >= quote["B0"], quote
    assert quote["B1"] >= quote["B0"], quote
    assert 0.0 <= min(quote.values()) and max(quote.values()) <= 1.0
    # Ehrlich notiert: In DIESEM synthetischen Haus ist B0 fast blind,
    # weil die haeufigsten Tokens paarweise abwechseln (Licht an/aus) —
    # der argmax des Unigramms ist dann nie das naechste Ereignis. Das
    # ist eine Eigenschaft des Spielzeug-Hauses, keine Schwaeche des
    # Gegners; die echte Tafel weist solche Zahlen mit Begruendung aus.
    assert quote["B0"] <= 0.15, quote
