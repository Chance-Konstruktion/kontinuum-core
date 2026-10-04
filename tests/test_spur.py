"""Selbsttests der Kontinuum-Spur (Stufe 1, Schritt 1).

Sie sichern die vier Zusagen des Formats: Ortszeit ist Pflicht, die
Ordnung steht, die Ausduennung ist verlustfrei (der echte Thalamus
verwirft nichts mehr), und Kategorien decken jedes Token genau einmal.
Dazu die zwei Ehrlichkeits-Beweise aus dem Protokoll: der Simulator ist
geseedet (gleiche Saat ⇒ gleiche Bytes), und das Leck-Tag-Verschwinden
in den Leistungs-Eimern steht als Zahl im Bericht — der Boden fuer die
Hypothalamus-Probe C (#2, Notiz 13430).

Bewusst ohne pytest-Abhaengigkeit (wie die Mehrheit der Suite): reine
Asserts plus zwei kleine Helfer. Laeuft damit ueberall, wo die Engine
laeuft — auch ohne Test-Werkzeuge.
"""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from benchmarks.spur import pruefe_spur
from benchmarks.spur.simulator import ZEITZONE, simuliere
from benchmarks.spur.spur import (
    Entitaet,
    Ereignis,
    KATEGORIEN,
    SONSTIGES,
    SPUR_FORMAT,
    SpurFehler,
    ausduennen,
    kategorie,
    lies_spur,
    lies_ts,
    schreibe_spur,
    token_zerlegen,
    tokenisieren,
    uebersicht,
)


def _erwartet_fehler(beschreibung: str, funktion, teil: str) -> None:
    """Wie pytest.raises(match=...): Fehler mit Namen, sonst lauter."""
    try:
        funktion()
    except SpurFehler as fehler:
        assert teil in str(fehler), (
            f"{beschreibung}: Fehler kam, aber ohne '{teil}': {fehler}"
        )
        return
    raise AssertionError(f"{beschreibung}: kein SpurFehler geworfen")


def _kopf_zeile(haus_typ: str = "klassisch") -> str:
    return json.dumps(
        {
            "format": SPUR_FORMAT,
            "haus": "test",
            "haus_typ": haus_typ,
            "quelle": "simulation",
            "zeitzone": "Europe/Berlin",
            "konverter": "test",
            "lizenz": "Test",
            "entitaeten": [
                {"id": "light.wohnzimmer", "raum": "wohnzimmer", "domain": "light"}
            ],
        }
    )


def _schreibe_zeilen(ordner: Path, zeilen) -> Path:
    pfad = ordner / "probe.jsonl"
    pfad.write_text("\n".join(zeilen) + "\n", encoding="utf-8", newline="\n")
    return pfad


# ---------------------------------------------------------------------------
# Format und Ordnung
# ---------------------------------------------------------------------------


def test_rundreise_und_tokenlauf():
    with tempfile.TemporaryDirectory() as ordner:
        spur = simuliere("klassisch", tage=5, saat=7).spur
        pfad = Path(ordner) / "haus.jsonl"
        schreibe_spur(spur, pfad)
        wieder = lies_spur(pfad)
        assert wieder.kopf.haus == spur.kopf.haus
        assert wieder.kopf.haus_typ == "klassisch"
        assert len(wieder.ereignisse) == len(spur.ereignisse)
        assert tokenisieren(wieder) == tokenisieren(spur)


def test_zeit_ohne_zone_ist_ein_fehler():
    with tempfile.TemporaryDirectory() as ordner:
        pfad = _schreibe_zeilen(
            Path(ordner),
            [
                _kopf_zeile(),
                json.dumps(
                    {"ts": "2026-03-02T07:00:00", "entity": "light.wohnzimmer",
                     "zustand": "on"}
                ),
            ],
        )
        _erwartet_fehler("Naive Zeit", lambda: lies_spur(pfad), "ohne Zone")
    # Mit Zone ist dieselbe Zeit gueltig
    assert lies_ts("2026-03-02T07:00:00+02:00").hour == 7


def test_fallende_zeit_ist_ein_fehler():
    with tempfile.TemporaryDirectory() as ordner:
        pfad = _schreibe_zeilen(
            Path(ordner),
            [
                _kopf_zeile(),
                json.dumps({"ts": "2026-03-02T08:00:00+02:00",
                            "entity": "light.wohnzimmer", "zustand": "on"}),
                json.dumps({"ts": "2026-03-02T07:00:00+02:00",
                            "entity": "light.wohnzimmer", "zustand": "off"}),
            ],
        )
        _erwartet_fehler("Fallende Zeit", lambda: lies_spur(pfad), "faellt")


def test_unbekannte_entity_ist_ein_fehler():
    with tempfile.TemporaryDirectory() as ordner:
        pfad = _schreibe_zeilen(
            Path(ordner),
            [
                _kopf_zeile(),
                json.dumps({"ts": "2026-03-02T07:00:00+02:00",
                            "entity": "light.kueche", "zustand": "on"}),
            ],
        )
        _erwartet_fehler("Unbekannte Entity", lambda: lies_spur(pfad),
                         "nicht in")


def test_zitat_und_lizenz_sind_pflicht_bei_fremden_quellen():
    """Abnahme 13642: quelle != simulation verlangt lizenz UND zitat —
    eine Spur ohne Herkunft ist keine Messung."""
    with tempfile.TemporaryDirectory() as ordner:
        kopf = json.loads(_kopf_zeile())
        kopf["quelle"] = "casas"
        kopf["lizenz"] = "CC BY 4.0"
        pfad = _schreibe_zeilen(
            Path(ordner),
            [json.dumps(kopf), json.dumps(
                {"ts": "2026-03-02T07:00:00+02:00",
                 "entity": "light.wohnzimmer", "zustand": "on"})],
        )
        _erwartet_fehler("Zitat fehlt", lambda: lies_spur(pfad), "zitat")
        kopf["zitat"] = "Cook et al. (2013), CASAS"
        _schreibe_zeilen(
            Path(ordner),
            [json.dumps(kopf), json.dumps(
                {"ts": "2026-03-02T07:00:00+02:00",
                 "entity": "light.wohnzimmer", "zustand": "on"})],
        )
        assert lies_spur(pfad).kopf.zitat == "Cook et al. (2013), CASAS"


# ---------------------------------------------------------------------------
# Ausduennung — und der Beweis, dass sie verlustfrei ist
# ---------------------------------------------------------------------------


def test_ausduennung_ist_verlustfrei():
    entitaeten = (Entitaet("sensor.leistung", "hauswirtschaft", "sensor", "power"),)
    t0 = datetime(2026, 3, 2, 6, 0, tzinfo=ZEITZONE)
    roh = [
        Ereignis(t0, "sensor.leistung", "30"),                          # low
        Ereignis(t0 + timedelta(minutes=1), "sensor.leistung", "35"),   # low, gleicher Eimer
        Ereignis(t0 + timedelta(minutes=2), "sensor.leistung", "805"),  # medium
        Ereignis(t0 + timedelta(minutes=3), "sensor.leistung", "799"),  # medium, gleicher Eimer
        Ereignis(t0 + timedelta(minutes=4), "sensor.leistung", "1800"), # high
        Ereignis(t0 + timedelta(minutes=5), "sensor.leistung", "unavailable"),
    ]
    behalten = ausduennen(entitaeten, roh)
    assert [e.zustand for e in behalten] == ["30", "805", "1800"]
    # Der Token-Lauf bestaetigt: die ausgeuennnte Spur traegt je Ereignis ein Token.
    spur = simuliere("geraete", tage=1, saat=3, mit_leck_tag=False).spur
    assert len(tokenisieren(spur)) == len(spur.ereignisse)


def test_engine_ignorierte_entity_wird_benannt():
    """Die Engine verwirft manche Entities per Muster (z. B.
    `^sensor\\.server_` — Infrastruktur-Rauschen). Eine Spur mit so
    einer Entity ist ein Formatfehler MIT NAMEN, kein stilles Loch:
    der Konverter muss sie vorher weglassen (Befund aus der Messbiene-
    Arbeit am Geräte-Haus; die Liste steht in `thalamus.py`)."""
    entitaeten = (Entitaet("sensor.server_cpu", "keller", "sensor", None, "cpu"),)
    _erwartet_fehler(
        "Engine-ignorierte Entity",
        lambda: ausduennen(entitaeten, [Ereignis(
            datetime(2026, 3, 2, 6, 0, tzinfo=ZEITZONE), "sensor.server_cpu", "42")]),
        "Nicht registrierbar",
    )


def test_unbekannte_entity_im_strom_ist_kein_stilles_wegduennen():
    entitaeten = (Entitaet("sensor.leistung", "hauswirtschaft", "sensor", "power"),)
    t0 = datetime(2026, 3, 2, 6, 0, tzinfo=ZEITZONE)
    roh = [Ereignis(t0, "sensor.fremd", "10")]
    _erwartet_fehler(
        "Unregistrierte Entity im Strom",
        lambda: ausduennen(entitaeten, roh),
        "nicht registrierter",
    )


# ---------------------------------------------------------------------------
# Kategorien
# ---------------------------------------------------------------------------


def test_kategorien_decken_jedes_token_genau_einmal():
    for haus_typ in ("klassisch", "geraete"):
        spur = simuliere(haus_typ, tage=6, saat=11).spur
        tabelle = KATEGORIEN[haus_typ]
        for token in tokenisieren(spur):
            _, semantik, _ = token_zerlegen(token)
            kat = kategorie(haus_typ, semantik)
            assert kat
            if semantik in tabelle:
                assert kat == tabelle[semantik]
            else:
                assert kat == SONSTIGES
    assert kategorie("klassisch", "voellig_neu") == SONSTIGES


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


def test_simulator_ist_geseedet():
    with tempfile.TemporaryDirectory() as ordner:
        a = Path(ordner) / "a.jsonl"
        b = Path(ordner) / "b.jsonl"
        c = Path(ordner) / "c.jsonl"
        schreibe_spur(simuliere("geraete", tage=4, saat=5).spur, a)
        schreibe_spur(simuliere("geraete", tage=4, saat=5).spur, b)
        schreibe_spur(simuliere("geraete", tage=4, saat=6).spur, c)
        assert a.read_bytes() == b.read_bytes()
        assert a.read_bytes() != c.read_bytes()


def test_leck_tag_verschwindet_in_den_eimern():
    """Der Befund der Denkrunde #3, als Zahl: Der rohe Sensor meldet das
    Leck 48-mal, die Eimer lassen davon fast nichts uebrig — genau der
    Boden, auf dem Probe C ihre Verbesserung zeigen muss."""
    mit = simuliere("geraete", tage=8, saat=2, mit_leck_tag=True)
    ohne = simuliere("geraete", tage=8, saat=2, mit_leck_tag=False)
    assert ohne.leck_roh == 0
    assert mit.leck_roh >= 40
    assert 0 < mit.leck_spur <= 3
    assert mit.bericht["eimer_verlust"] > 0


def test_diagnose_semantik_laut_name_gegen_vergeben():
    """Befund 13646: Ein Leistungssensor mit device_class power wird
    "power", nie "solar"/"grid" — der Name sagt etwas anderes. Die
    Diagnosezeile zaehlt genau diese Faelle (die Simulation traegt sie
    jetzt realistisch: pv_leistung und netz_bezug haben device_class
    power, wie echte HA-Sensoren)."""
    simulation = simuliere("geraete", tage=10, saat=4)
    bericht = uebersicht(simulation.spur)
    # Der Netz-Sensor (device_class power, Name ..._netz_bezug): die
    # Klasse gewinnt, "grid" ist unerreichbar — genau der Fund 13646.
    assert "power statt grid" in bericht["semantik_abweichungen_paare"]
    # Ausdrueckliche Uebersteuerungen (wallbox, co2, cpu) stehen als
    # eigene Zahl daneben — Absicht, kein Fund.
    assert "wallbox statt switch" in bericht["semantik_abweichungen_paare"]
    assert bericht["semantik_uebersteuerungen"] == 3
    assert bericht["semantik_abweichungen"].get("energie", 0) >= 2
    # Und der Gegentest der Diagnose: pv_leistung zeigt KEINE Abweichung,
    # weil das Stichwort "leistung" in SENSOR_KEYWORDS VOR "pv_" steht —
    # beide Wege sagen "power". Zwei Mechanismen, eine Zeile, die beide
    # sichtbar macht (Klassen-Vorrang UND Stichwort-Reihenfolge).


def test_diagnose_entitaeten_je_token():
    """Befund 13646: raum.semantik.zustand vermischt Entities desselben
    Raums und derselben Semantik. Im Geraete-Haus liegen drei
    Leistungsmesser in hauswirtschaft — die Diagnose muss das zeigen."""
    geraete = uebersicht(simuliere("geraete", tage=10, saat=4).spur)
    assert geraete["entitaeten_je_token"].get("energie", 0) > 0
    klassisch = uebersicht(simuliere("klassisch", tage=10, saat=4).spur)
    assert klassisch["entitaeten_je_token"].get("bewegung", 0) == 0


def test_uebersicht_traegt_die_aggregate():
    simulation = simuliere("geraete", tage=10, saat=4)
    bericht = uebersicht(simulation.spur)
    assert bericht["haus_typ"] == "geraete"
    for pflicht in ("energie", "heizung", "infrastruktur", "klima"):
        assert pflicht in bericht["je_kategorie"], pflicht
    assert bericht["marken"].get("anomalie", 0) >= 1
    # Kimis Vokabular-Luecke: co2 "elevated" fehlt im Hypothalamus-Lexikon.
    assert "co2.elevated" in bericht["hypothalamus_luecken_paare"]


# ---------------------------------------------------------------------------
# Waechter-CLI
# ---------------------------------------------------------------------------


def test_waechter_cli():
    with tempfile.TemporaryDirectory() as ordner:
        spur = simuliere("klassisch", tage=3, saat=9).spur
        pfad = Path(ordner) / "haus.jsonl"
        schreibe_spur(spur, pfad)
        puffer = io.StringIO()
        with contextlib.redirect_stdout(puffer), contextlib.redirect_stderr(puffer):
            code = pruefe_spur.main([str(pfad)])
        ausgabe = puffer.getvalue()
        assert code == 0, ausgabe
        assert "Kategorien" in ausgabe
        assert "gueltig" in ausgabe
        # Der Waechter druckt nur Aggregate — keine Entity-Namen (Datensparsamkeit).
        assert "sensor." not in ausgabe and "light." not in ausgabe

        kaputt = Path(ordner) / "kaputt.jsonl"
        kaputt.write_text("kein json\n", encoding="utf-8", newline="\n")
        puffer = io.StringIO()
        with contextlib.redirect_stdout(puffer), contextlib.redirect_stderr(puffer):
            assert pruefe_spur.main([str(kaputt)]) == 1
        puffer = io.StringIO()
        with contextlib.redirect_stdout(puffer), contextlib.redirect_stderr(puffer):
            assert pruefe_spur.main([]) == 2
