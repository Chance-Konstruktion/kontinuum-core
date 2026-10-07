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

06.10.2026, das Claustrum (Stufe 3): Die Engine ist jetzt die Börse und
schlaegt B2 deutlich. Die beiden alten Befunde bleiben als Tests der
ALTEN Kette stehen (``KontinuumEngine(claustrum=False)``) — sie sind die
Geschichte, an der man den Unterschied misst. Die neuen Zusagen stehen
darunter.
"""
from __future__ import annotations

from datetime import timedelta

from benchmarks.spur.messstand import (
    ENGINE_ALT,
    ENGINE_ROH,
    ENGINE_VOR_RANKING,
    gepaarte_differenz,
    messe,
    messe_ursprung,
)
from benchmarks.spur.simulator import simuliere
from kontinuum_core import KontinuumEngine

SAAT_KLASSISCH = 1
SAAT_GERAETE = 1


def _alte_kette() -> KontinuumEngine:
    """Die Engine vor dem Claustrum: gerankte Kette, kein Lagebild."""
    return KontinuumEngine(claustrum=False, lagebild=False)


def test_messstand_liest_nicht_hinter_dem_testende():
    """Ein Ursprung endet mit seiner Testwoche — kein Ereignis danach
    darf die Engine noch erreichen. Auf den Spielzeug-Haeusern faellt
    das nicht auf, auf CASAS (1,6 Mio Ereignisse) kostet es Stunden.
    Der Beweis laeuft ueber eine aufzeichnende Engine: ihr groesstes
    gesehenes Datum liegt VOR dem Testende."""
    sim = simuliere("klassisch", tage=42, saat=5)
    spur = sim.spur
    test_ende = spur.ereignisse[0].ts + timedelta(days=7 * 4 + 7)
    gesehen = []

    def bauer():
        engine = KontinuumEngine()
        original = engine.observe

        def merken(ereignis):
            gesehen.append(ereignis["timestamp"])
            return original(ereignis)

        engine.observe = merken  # type: ignore[method-assign]
        return engine

    messe_ursprung(spur, train_wochen=4, engine_bauer=bauer)
    assert gesehen, "die Engine hat kein einziges Ereignis gesehen"
    assert max(gesehen) < test_ende, (
        f"hinter dem Testende gelesen: {max(gesehen)} >= {test_ende}"
    )


def test_messstand_bewertet_nur_die_testwoche():
    sim = simuliere("klassisch", tage=42, saat=SAAT_KLASSISCH)
    ergebnis = messe_ursprung(sim.spur, train_wochen=4)
    assert ergebnis.test_ereignisse > 0
    # Befund 1 (MR !4 note 13693), geheilt: die vier SYSTEME tragen JEDES
    # Testereignis; die Roh-Zeilen leeren sich bei leeren/verworfenen
    # Ereignissen ehrlich und zählen deshalb HOECHSTENS so viele.
    assert ENGINE_ALT in ergebnis.systeme, "die alte Kette fehlt als Zeile"
    for name, bilanz in ergebnis.systeme.items():
        if name in (ENGINE_ROH, ENGINE_VOR_RANKING, ENGINE_ALT):
            assert 0 < bilanz.gesamt <= ergebnis.test_ereignisse, name
        else:
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
    """BEFUND (05.10.2026, Simulation, Saat 1, Ursprung 4, geseedet,
    mit Zeitgeber/Ereigniszeit gemessen): Die rohe Liste traegt die
    Wahrheit, das Modul-Ranking schiebt sie aus Platz 1 — klassisch
    50,7 % vs. 45,9 %, geraete 67,4 % vs. 55,2 %. (Die frueheren
    Wanduhr-Zahlen 60,0/43,9 und 83,7/67,8 ueberzeichneten den Abstand:
    die Zahl hing an der Tageszeit des Laufs, siehe
    tests/test_spur_zeitgeber.py.) Dieser Test haelt den Befund fest;
    das Ranking gehoert auf den Pruefstand (Protokoll § 5), nicht
    wegdiskutiert."""
    for typ in ("klassisch", "geraete"):
        sim = simuliere(typ, tage=42, saat=1)
        ergebnis = messe_ursprung(sim.spur, train_wochen=4,
                                  engine_bauer=_alte_kette)
        roh = ergebnis.systeme[ENGINE_ROH].trefferquote()
        gerankt = ergebnis.systeme["Engine"].trefferquote()
        assert roh > gerankt, (typ, roh, gerankt)


def test_gepaarte_differenz_wird_berichtet():
    """Abnahme 13642, Punkt 3: je Haus die gepaarte Differenz
    Engine − bester Gegner als Median [Min–Max] ueber die Ursprünge."""
    sim = simuliere("klassisch", tage=56, saat=1)
    ergebnis = messe(sim.spur, min_train_wochen=4, engine_bauer=_alte_kette)
    differenz = gepaarte_differenz(ergebnis)
    assert differenz["bester_gegner"] == "B2"
    assert len(differenz["differenzen"]) == len(ergebnis.ursprunge)
    assert differenz["min"] <= differenz["median"] <= differenz["max"]
    # Befund der ALTEN Kette (mit Zeitgeber): die Differenz ist NEGATIV
    # (B2 fuehrt) — klassisch −10,2 % [−14,6 .. −4,9], Saat 1, 8 Wochen.
    # Knapp ist sie nicht.
    assert differenz["median"] < 0


def test_befund_b2_ist_der_echte_gegner():
    """BEFUND (05.10.2026, mit Zeitgeber/Ereigniszeit): Auf dem
    klassischen Spielzeug-Haus schlaegt die reine 1-Gramm-Kette (B2)
    die Engine (50,7 % vs. 45,9 %, Saat 1, Ursprung 4, geseedet); auch
    auf dem Geraete-Haus liegt B2 vorn (59,8 % vs. 55,2 %), und ueber
    die vier Ursprünge der 8-Wochen-Messung ist die gepaarte Differenz
    −4,4 % [−7,4 .. −1,7]. Die Engine schlaegt B1 deutlich — aber der
    Sieg ueber B2 ist an keiner Stelle geschenkt; genau das soll die
    Tafel zeigen. (Der fruehere Geraete-„Sieg" +7,1 % war ein
    Wanduhr-Artefakt.)"""
    sim = simuliere("klassisch", tage=42, saat=1)
    ergebnis = messe_ursprung(sim.spur, train_wochen=4,
                              engine_bauer=_alte_kette)
    assert ergebnis.systeme["Engine"].trefferquote() > \
        ergebnis.systeme["B1"].trefferquote()
    assert ergebnis.systeme["B2"].trefferquote() > \
        ergebnis.systeme["Engine"].trefferquote()


def test_claustrum_schlaegt_jeden_gegner_auf_beiden_haustypen():
    """ZUSAGE der Börse (06.10.2026): Auf beiden Spielzeug-Häusern liegt
    die Engine mit Claustrum vor dem besten dummen Gegner, und zwar nicht
    knapp — gemessen klassisch +26 bis +31, Geräte-Haus +30 bis +34
    Punkte über alle vier Ursprünge (Saat 1, 8 Wochen). Gefordert werden
    hier +15, damit eine echte Verschlechterung auffällt und nicht jede
    Nachkommastelle."""
    for typ in ("klassisch", "geraete"):
        sim = simuliere(typ, tage=56, saat=1)
        ergebnis = messe(sim.spur, min_train_wochen=4)
        differenz = gepaarte_differenz(ergebnis)
        assert differenz["min"] > 0.15, (typ, differenz)


def test_claustrum_schlaegt_die_alte_kette_im_selben_lauf():
    """Vorher und Nachher in EINER Messung: Die alte Kette läuft als
    Zeile „Engine alt“ mit. Die Börse muss sie schlagen, und der
    Hippocampus allein (roh) darf ihr nicht davonlaufen.

    Gemessen (4 Saaten × 4 Ursprünge, Top-1): klassisch Engine 82,5 /
    roh 69,3 / alt 53,6; Geräte-Haus 93,5 / 93,2 / 83,6 — im Geräte-Haus
    ist der Hippocampus allein schon fast so gut, die Börse hält mit ihm
    Schritt. Hier läuft EIN Ursprung (Saat 1, ~200 Testereignisse, ein
    Treffer = 0,5 Punkte): klassisch +37 über alt, Geräte-Haus +6, roh
    −1,6. Gefordert wird darum je Haus, was auch im Rauschen hält."""
    mindestens_ueber_alt = {"klassisch": 0.15, "geraete": 0.03}
    for typ in ("klassisch", "geraete"):
        sim = simuliere(typ, tage=42, saat=1)
        ergebnis = messe_ursprung(sim.spur, train_wochen=4)
        neu = ergebnis.systeme["Engine"].trefferquote()
        alt = ergebnis.systeme[ENGINE_ALT].trefferquote()
        roh = ergebnis.systeme[ENGINE_ROH].trefferquote()
        assert neu > alt + mindestens_ueber_alt[typ], (typ, neu, alt)
        assert neu > roh - 0.03, (typ, neu, roh)
