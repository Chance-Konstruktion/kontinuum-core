"""Selbsttests des Messkerns (Stufe 1, Protokoll §§ 1/4/5).

Struktur-Zusagen (Bilanzen addieren sich, ECE ist ein Mass, zu kurze
Spuren werfen mit Namen) plus zwei CHARAKTERISIERUNGS-Tests: Sie halten
den ersten gemessenen Befund fest, statt ihn zu wuenschen —

* die ROHE Hippocampus-Liste ist auf den Spielzeug-Haeusern besser als
  die gerankte (das Modul-Ranking kostet Top-1), und
* die reine 1-Gramm-Kette B2 schlaegt die Engine auf dem klassischen
  Spielzeug-Haus.

Beide Saetze sind BEFUNDE, keine Wünsche: Wer das Ranking verbessert
oder die Engine staerkt, dreht sie bewusst um — und muss diesen Test
mitziehen. Genau dafuer steht er hier.
"""
from __future__ import annotations

from benchmarks.spur.messstand import ENGINE_ROH, messe, messe_ursprung
from benchmarks.spur.simulator import simuliere

SAAT_KLASSISCH = 1
SAAT_GERAETE = 1


def test_messstand_bewertet_nur_die_testwoche():
    sim = simuliere("klassisch", tage=42, saat=SAAT_KLASSISCH)
    ergebnis = messe_ursprung(sim.spur, train_wochen=4)
    assert ergebnis.test_ereignisse > 0
    for name, bilanz in ergebnis.systeme.items():
        assert bilanz.gesamt == ergebnis.test_ereignisse, name
        assert bilanz.top1 <= bilanz.top3 <= bilanz.gesamt, name


def test_kategorien_und_eimer_addieren_sich():
    sim = simuliere("geraete", tage=42, saat=SAAT_GERAETE)
    ergebnis = messe_ursprung(sim.spur, train_wochen=4)
    for name, bilanz in ergebnis.systeme.items():
        assert sum(bilanz.je_kategorie_gesamt.values()) == bilanz.gesamt, name
        assert sum(bilanz.je_kategorie_treffer.values()) == bilanz.top1, name
        assert 0.0 <= bilanz.ece() <= 1.0, name
        n_eimer = sum(n for _, n, _, _ in bilanz.kalibrierungs_eimer())
        assert n_eimer == bilanz.gesamt, name
        for _, n, konfidenz, quote in bilanz.kalibrierungs_eimer():
            assert 0.0 <= konfidenz <= 1.0 and 0.0 <= quote <= 1.0


def test_zu_kurze_spur_wirft_mit_namen():
    sim = simuliere("klassisch", tage=21, saat=2)
    try:
        messe(sim.spur, min_train_wochen=4)
    except ValueError as fehler:
        assert "Zu wenige Wochen" in str(fehler)
        return
    raise AssertionError("Zu kurze Spur wurde still gemessen")


def test_rollierende_ursprunge():
    sim = simuliere("klassisch", tage=56, saat=3)
    ergebnis = messe(sim.spur, min_train_wochen=4)
    assert [u.ursprung for u in ergebnis.ursprunge] == [4, 5, 6, 7]
    for ursprung in ergebnis.ursprunge:
        assert ursprung.test_ereignisse > 0
        assert ursprung.test_bis > ursprung.test_von


def test_befund_rohliste_schlaegt_ranking():
    """BEFUND (03.10.2026, Simulation, Saat 1, Ursprung 4, geseedet
    deterministisch): Die rohe Hippocampus-Liste traegt die Wahrheit,
    das Modul-Ranking schiebt sie aus Platz 1 — klassisch 63,4 % vs.
    44,9 %, geraete 86,6 % vs. 69,9 %. Dieser Test haelt den Befund
    fest; das Ranking gehoert auf den Pruefstand (Protokoll § 5),
    nicht wegdiskutiert."""
    for typ in ("klassisch", "geraete"):
        sim = simuliere(typ, tage=42, saat=1)
        ergebnis = messe_ursprung(sim.spur, train_wochen=4)
        roh = ergebnis.systeme[ENGINE_ROH].trefferquote()
        gerankt = ergebnis.systeme["Engine"].trefferquote()
        assert roh > gerankt, (typ, roh, gerankt)


def test_befund_b2_ist_der_echte_gegner():
    """BEFUND: Auf dem klassischen Spielzeug-Haus schlaegt die reine
    1-Gramm-Kette (B2) die Engine (50,7 % vs. 44,9 %, Saat 1,
    Ursprung 4, geseedet). Die Engine schlaegt B1 deutlich — aber der
    Sieg ueber B2 ist NICHT geschenkt; genau das soll die Tafel zeigen.
    (Auf dem Geraete-Haus gewinnt die Engine diese eine Testwoche mit
    69,9 % vs. 63,2 % — ueber alle Ursprünge ist es ein Patt; die
    Spanne entscheidet, nicht die Einzelwoche.)"""
    sim = simuliere("klassisch", tage=42, saat=1)
    ergebnis = messe_ursprung(sim.spur, train_wochen=4)
    assert ergebnis.systeme["Engine"].trefferquote() > \
        ergebnis.systeme["B1"].trefferquote()
    assert ergebnis.systeme["B2"].trefferquote() > \
        ergebnis.systeme["Engine"].trefferquote()
