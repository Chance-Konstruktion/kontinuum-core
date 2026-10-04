"""Selbsttests des CASAS-Konverters (Stufe 1, Schritt 4).

Sie laufen auf einer MINI-Probe im echten CASAS-Format (keine Rohdaten
im Repo — die Regel gilt auch fuer Tests). Geprueft werden: Parsen
(Leerraum und Tab, Aktivitaetslabel mit Leerzeichen), die
Entitaetentabelle aus den Praefixen, das Zaehlen unbekannter Sensoren
statt stillem Schlucken, Lizenz+Zitat im Kopf und die Ausduennung.
"""
from __future__ import annotations

import io
import json
import tempfile
import zipfile
from pathlib import Path

from benchmarks.spur import pruefe_spur
from benchmarks.spur.casas import HAEUSER, konvertiere, zerlege_zeile
from benchmarks.spur.spur import lies_spur, tokenisieren, uebersicht

PROBE = """2010-11-04 00:03:50.209589 M003 ON Sleeping begin
2010-11-04 00:03:57.399391 M003 OFF
2010-11-04 00:15:08.984841 T002 21.5
2010-11-04 00:30:19.185547	T003	21
2010-11-04 00:30:19.385336 D001 OPEN
2010-11-04 00:31:02.000000 D001 CLOSE
2010-11-04 00:40:00.000000 X999 ON
2010-11-04 00:41:00.000000 T002 21.7
2010-11-04 00:42:00.000000 T002 21.9
"""


def _mini_archiv(ordner: Path) -> Path:
    pfad = ordner / "mini.zip"
    with zipfile.ZipFile(pfad, "w") as paket:
        paket.writestr("aruba.txt", PROBE)
    return pfad


def test_zerlege_zeile_leerraum_und_tab():
    zonen = (None, None)
    from datetime import timezone as _tz
    zonen = (_tz.utc, "UTC")
    a = zerlege_zeile("2010-11-04 00:03:50.2 M003 ON Sleeping begin", zonen, 1)
    b = zerlege_zeile("2010-11-04 00:30:19.1\tT003\t21", zonen, 2)
    assert a.sensor == "m003" and a.wert == "ON"
    assert a.aktivitaet == "Sleeping begin"
    assert b.sensor == "t003" and b.wert == "21" and b.aktivitaet is None


def test_konverter_baut_tabelle_thinnt_und_zaehlt_unbekanntes():
    with tempfile.TemporaryDirectory() as ordner:
        archiv = _mini_archiv(Path(ordner))
        ziel = Path(ordner) / "spur.jsonl"
        bericht = konvertiere(str(archiv), "aruba", str(ziel))
        assert bericht["zeilen"] == 9
        assert bericht["unbekannte_sensoren"] == 1
        assert bericht["unbekannte_zeilen"] == {"x": 1}
        assert bericht["aktivitaetszeilen"] == 1
        # T002 21.5/21.7/21.9 liegen alle im Eimer "comfort" -> EIN
        # Ereignis (verlustfrei fuer die Engine, nicht fuer die
        # Nachkommastelle). T003 ist eine EIGENE Entity — ihr "comfort"
        # verschmilzt nicht mit dem von T002. Also: 2 (M) + 1 (T002)
        # + 1 (T003) + 2 (D) = 6.
        assert bericht["behalten"] == 6, bericht
        spur = lies_spur(ziel)
        assert spur.kopf.lizenz == "CC BY 4.0"
        assert "Cook" in spur.kopf.zitat
        assert spur.kopf.quelle == "casas"
        assert "America/Los_Angeles" in spur.kopf.zeitzone
        namen = sorted(e.id for e in spur.kopf.entitaeten)
        assert namen == ["d001", "m003", "t002", "t003"]
        assert len(tokenisieren(spur)) == len(spur.ereignisse)
        assert pruefe_spur.pruefe(ziel)["ereignisse"] == 6


def test_wochen_grenze():
    with tempfile.TemporaryDirectory() as ordner:
        archiv = _mini_archiv(Path(ordner))
        bericht = konvertiere(str(archiv), "aruba", None, wochen=1)
        assert bericht["behalten"] == 6  # alles liegt im ersten Tag
        assert bericht["tage"] == 1


def test_haeuser_liste_ist_vollstaendig():
    assert set(HAEUSER) == {"aruba", "cairo", "milan", "tulum1", "tulum2"}
