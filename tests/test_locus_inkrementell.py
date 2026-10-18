"""Der inkrementelle Gleitfenster-Pfad des Locus Coeruleus (Schritt 6c).

Der Bestandsfilter zaehlt bei jedem Ereignis alle letzten 2000
Zeitstempel durch (O(2000) je Ereignis); die Optimierung pflegt das
60-s-Fenster stattdessen inkrementell. Die Abnahme-Bedingung ist scharf:
die Zaehlung muss auf JEDEM Strom identisch zur Bestandsformel sein.
Dieser Test fuehrt die Bestandsformel als Orakel mit und vergleicht nach
JEDEM Ereignis Arousal UND Fensterzaehlung — nicht nur am Ende, damit
kein Zwischenzustand abweichen kann.

Die kleinen Parameter (Fenster 10 s, Kapazitaet 7) machen Verdraengungs-
und Ruecksprungpfade mit wenigen Ereignissen erreichbar; die grossen
Durchlaeufe (60 s/2000) pruefen den gemessenen Fall. Die Aufbau-Zaehler
bezeugen die Kostenform: der duenne Strom UND der dichte monotone Strom
laufen ohne Neuaufbau (sonst waere die Aenderung wirkungslos bzw. im
dichten Fall sogar teurer als der Bestand — die Mikro-Sonde hat genau
das an der ersten Fassung gemessen); nur Rueckspruenge bauen neu auf.
"""
from __future__ import annotations

import random
from collections import deque

import pytest

from kontinuum_core.locus_coeruleus import LocusCoeruleus


class _Bestand:
    """Die Bestandsformel, woertlich (locus_coeruleus.py vor Schritt 6c)."""

    def __init__(self, fenster: float = 60.0, kapazitaet: int = 2000):
        self.fenster = fenster
        self.events = deque(maxlen=kapazitaet)
        self.arousal = 0.2

    def observe_event(self, now) -> None:
        self.events.append(now)
        recent = [t for t in self.events if now - t <= self.fenster]
        density = len(recent) / float(self.fenster)
        target = min(1.0, density * 2.5)
        self.arousal = self.arousal * 0.9 + target * 0.1


class _Klein(LocusCoeruleus):
    """Winzige Parameter — jeder Randpfad mit wenigen Ereignissen erreichbar."""

    WINDOW_SECONDS = 10
    MAX_EVENTS = 7


def _vergleiche(strom, klein: bool = False, name: str = ""):
    """Fuehrt beide Fassungen ueber den Strom und vergleicht nach JEDEM
    Ereignis die Fensterzaehlung exakt (und damit bitgleich das Arousal —
    dieselbe Rechnung, sobald die Zaehlung stimmt)."""
    neu = _Klein() if klein else LocusCoeruleus()
    alt = _Bestand(fenster=neu.WINDOW_SECONDS, kapazitaet=neu.MAX_EVENTS)
    for i, t in enumerate(strom):
        neu.observe_event(t)
        alt.observe_event(t)
        erwartet = sum(1 for x in alt.events if t - x <= alt.fenster)
        assert len(neu._fenster) == erwartet, (
            f"{name}: Schritt {i}, t={t!r}: Fenster {len(neu._fenster)} "
            f"statt {erwartet}")
        assert neu.arousal == alt.arousal, (
            f"{name}: Schritt {i}, t={t!r}: Arousal {neu.arousal!r} "
            f"statt {alt.arousal!r}")
    assert list(neu.events) == list(alt.events), f"{name}: Ereignis-Deque"
    return neu


def _strom_monoton(rng, n, schritt=30.0):
    t = 1_000.0
    for _ in range(n):
        t += rng.uniform(0.0, 2.0 * schritt)
        yield t


def _strom_gemischt(rng, n, fenster):
    """Monotone Strecken, Duplikate, exakte Fenstergrenze, Rueckspruenge,
    dichte Schuebe — alles, was die Waechter ausloesen koennen."""
    t = 1_000.0
    i = 0
    while i < n:
        wahl = rng.random()
        if wahl < 0.45:
            t += rng.uniform(0.0, 2.0 * fenster)
            i += 1
        elif wahl < 0.60:                      # Duplikat (t unveraendert)
            i += 1
        elif wahl < 0.75:                      # exakte Grenze now - t == fenster
            t += fenster
            i += 1
        elif wahl < 0.90:                      # Ruecksprung
            t += rng.uniform(-3.0 * fenster, 0.0)
            i += 1
        else:                                  # dichter Schub
            for _ in range(min(rng.randint(3, 40), n - i)):
                t += rng.uniform(0.0, 0.01 * max(fenster, 1.0))
                i += 1
        yield t


def _zaehle_aufbauten(lc, strom):
    """Neuaufbau = das Fenster-Objekt wird ersetzt (die Pflege mutiert in
    place). Referenzen festhalten, damit ids nicht wiederverwendet werden."""
    objekte = [lc._fenster]
    for t in strom:
        lc.observe_event(t)
        objekte.append(lc._fenster)
    return sum(1 for a, b in zip(objekte, objekte[1:]) if a is not b)


# ---------------------------------------------------------------------------
# Die Randfaelle von Hand
# ---------------------------------------------------------------------------


def test_grenze_ist_inklusiv():
    """``now - t == WINDOW`` zaehlt noch; eine Winzigkeit spaeter nicht mehr.
    Die Pflege schneidet mit ``>`` ab, der Bestand filtert mit ``<=``."""
    _vergleiche([0.0, 10.0, 20.0, 20.000001, 20.0, 20.000002], klein=True,
                name="grenze")


def test_ruecksprung_dann_vorwaerts():
    """Der Fallstrick fuer einen naiven Puffer: nach dem Ruecksprung
    (9.9 -> 4.5) steht das alte Element VOR dem neuen. Ein Vorwaerts-
    Ereignis (15.6) laesst das abgelaufene 4.5 dann hinter lebenden
    Elementen links liegen — wer es behaelt, zaehlt zu viel."""
    _vergleiche([9.9, 4.5, 10.0, 15.6, 15.7, 20.0, 26.0], klein=True,
                name="ruecksprung")


def test_verdraengtes_im_fenster_wird_gestrichen():
    """Kapazitaet voll, alle Elemente im Fenster: jede Neuankunft
    verdraengt Lebendes. Der Puffer muss es streichen — wer es stehen
    laesst, zaehlt dauerhaft zu viel."""
    _vergleiche([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                 1.0, 1.1, 2.0, 9.0, 10.5, 11.6], klein=True,
                name="kapazitaet")


def test_nicht_endliche_zeit_wie_bestand():
    """NaN/inf koennen ueber die Engine nicht entstehen (der Zeitstempel
    kommt aus datetime), aber wer die Klasse direkt ruft, soll dasselbe
    sehen wie der Bestand: die Formel woertlich."""
    _vergleiche([0.5, 1.0, float("inf"), 2.0, float("nan"), 3.0,
                 -float("inf"), 4.0], klein=True, name="nicht-endlich")


def test_from_dict_hinterlaesst_frischen_puffer():
    rng = random.Random(21)
    lc = LocusCoeruleus()
    vor = list(_strom_monoton(rng, 300, schritt=5.0))
    for t in vor:
        lc.observe_event(t)
    daten = lc.to_dict()
    neu = LocusCoeruleus()
    neu.from_dict(daten)
    assert neu._fenster is None                  # haengt an events
    assert neu.to_dict() == daten                # Zustand unveraendert
    rest = [vor[-1] + 0.5, vor[-1] + 90.0, vor[-1] + 91.0, vor[-1] + 91.0]
    for t in rest:
        lc.observe_event(t)
        neu.observe_event(t)
    assert neu.arousal == lc.arousal
    assert list(neu.events) == list(lc.events)


# ---------------------------------------------------------------------------
# Zufallsstrae me gegen das Orakel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("saat", range(12))
def test_zufallsspur_klein(saat):
    """2500 gemischte Ereignisse auf winzigen Parametern: Verdraengung,
    Ruecksprung, Grenze und Schub in jeder Reihenfolge."""
    rng = random.Random(saat)
    _vergleiche(list(_strom_gemischt(rng, 2500, 10.0)), klein=True,
                name=f"klein/{saat}")


@pytest.mark.parametrize("saat", range(3))
def test_zufallsspur_gross(saat):
    """Dieselben Muster auf den echten Parametern (60 s/2000)."""
    rng = random.Random(100 + saat)
    _vergleiche(list(_strom_gemischt(rng, 4000, 60.0)), name=f"gross/{saat}")


def test_dichter_schub_ueber_der_kapazitaet():
    """2000er-Deque voll, alles im Fenster, dann ausduennen: der dichteste
    Fall mit dem groessten Puffer, inklusive Verdraengung von Lebendem
    bei jedem Ereignis."""
    rng = random.Random(7)
    t = 1_000.0
    strom = []
    for _ in range(2600):
        t += rng.uniform(0.0, 0.05)
        strom.append(t)
    for _ in range(300):
        t += 30.0
        strom.append(t)
    _vergleiche(strom, name="schub")


# ---------------------------------------------------------------------------
# Die Struktur: laeuft der schnelle Weg wirklich?
# ---------------------------------------------------------------------------


def test_duenne_spur_laeuft_ohne_neuaufbau():
    """Der gemessene Fall (CASAS/aruba duenn): Ereigniszeit laeuft
    vorwaerts, das verdraengte Element ist laengst abgelaufen — genau EIN
    Aufbau (der erste), danach nur noch Pflege. Ohne diese Eigenschaft
    waere die Aenderung wirkungslos."""
    rng = random.Random(11)
    lc = LocusCoeruleus()
    strom = list(_strom_monoton(rng, 20_000))
    assert _zaehle_aufbauten(lc, strom) == 1


def test_dichter_strom_laeuft_ohne_neuaufbau():
    """Gegenprobe fuer die Kostenform: auch der dichteste monotone Strom
    (mehr als MAX_EVENTS Ereignisse im Fenster, jede Neuankunft
    verdraengt Lebendes) laeuft nach dem ersten Aufbau ohne Neuaufbau —
    die erste Fassung baute hier bei JEDEM Ereignis neu auf und war
    dadurch messbar teurer als der Bestand (Mikro-Sonde 17.10.)."""
    rng = random.Random(12)
    lc = LocusCoeruleus()
    t = 1_000.0
    strom = []
    for _ in range(2_200):                    # monoton und dicht: alles im Fenster
        t += rng.uniform(0.0, 0.001)
        strom.append(t)
    assert _zaehle_aufbauten(lc, strom) == 1


def test_rueckspruenge_bauen_neu_auf():
    """Der ehrliche Rueckfall: ein Ruecksprung der Ereigniszeit (gemischte
    Stroeme mit/ohne Zeitstempel) kann nicht gepflegt werden — dort steht
    die Bestandsformel woertlich. Ergebnis exakt, Kosten wie der Bestand."""
    rng = random.Random(13)
    t = 1_000.0
    strom = []
    for i in range(600):
        t += rng.uniform(0.0, 30.0)
        if i % 7 == 0:
            t -= rng.uniform(5.0, 40.0)       # Ruecksprung
        strom.append(t)
    lc = _vergleiche(strom, name="rueckspruenge")
    assert _zaehle_aufbauten(lc, strom) > 1
