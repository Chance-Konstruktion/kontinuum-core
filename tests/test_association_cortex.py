"""Tests für den Assoziationskortex — das Lagebild (Stufe 3, #2).

Gerätestufen, ``unavailable`` als Zustand, die Paar-Tafel, der
Lage-Experte, die Anwesenheit auf dem simulierten Haushalt und die
bitgleiche Fortsetzung nach dem Laden.
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone

from benchmarks.spur.anwesenheit import _thalamus, messe_anwesenheit, zufuehren
from benchmarks.spur.haushalt import TRACKER, simuliere_haushalt
from kontinuum_core.association_cortex import WEG, AssociationCortex, Stufen

START = datetime(2026, 1, 5, 6, 0, tzinfo=timezone.utc)  # ein Montag


def test_stufen_trennen_standby_und_betrieb():
    """3 W Standby und 100 W Betrieb sind für den Thalamus beide „niedrig“;
    die gelernten Stufen trennen sie, Rauschen innerhalb einer Stufe nicht."""
    st = Stufen()
    wuerfel = random.Random(1)
    for _ in range(200):
        st.stufe(wuerfel.uniform(2.5, 3.5) if wuerfel.random() < 0.6 else wuerfel.uniform(90, 110))
    assert st.stufe(3.0) != st.stufe(100.0)
    assert st.stufe(2.8) == st.stufe(3.3)
    assert st.stufe(95.0) == st.stufe(105.0)


def test_unavailable_ist_ein_zustand_und_unknown_keiner():
    k = AssociationCortex()
    k.setze("sensor.reifen_vl", "2.4", START, normiert="ok")
    assert k.zustand["sensor.reifen_vl"] == "ok"
    k.setze("sensor.reifen_vl", "unavailable", START + timedelta(minutes=5))
    assert k.zustand["sensor.reifen_vl"] == WEG
    k.setze("sensor.reifen_vl", "unknown", START + timedelta(minutes=10))
    assert k.zustand["sensor.reifen_vl"] == "unbekannt"


def test_paar_tafel_findet_was_zusammen_gilt():
    """Fernseher und Leiste gehen immer gemeinsam an; die Lampe würfelt."""
    k = AssociationCortex()
    wuerfel = random.Random(2)
    t = START
    for _ in range(300):
        an = wuerfel.random() < 0.4
        k.setze("media_player.tv", "on" if an else "off", t)
        k.setze("switch.leiste", "on" if an else "off", t)
        k.setze("light.lampe", "on" if wuerfel.random() < 0.5 else "off", t)
        t += timedelta(minutes=30)
    assert k.p_gemeinsam(("media_player.tv", "on"), ("switch.leiste", "on")) > 0.95
    lampe = k.p_gemeinsam(("media_player.tv", "on"), ("light.lampe", "on"))
    assert 0.2 < lampe < 0.8
    oben = k.zusammenhaenge(top=4)
    assert any(z["wenn"] == "media_player.tv=on" and z["dann"] == "switch.leiste=on"
               for z in oben)


def test_lage_experte_weiss_was_eine_kette_vergisst():
    """Zwischen zwei Lampen-Ereignissen liegt beliebig viel Rauschen — eine
    Kette dritter Ordnung verliert den Faden, das Lagebild nicht: Ist die
    Lampe an, kommt als Nächstes „aus“, nicht „an“."""
    k = AssociationCortex()
    wuerfel = random.Random(4)
    lampe, t = "off", START
    for _ in range(1500):
        k.bereite_folge(t)
        if wuerfel.random() < 0.3:
            lampe = "on" if lampe == "off" else "off"
            y, entity, zustand = f"lampe.{lampe}", "light.lampe", lampe
        else:
            zustand = "on" if wuerfel.random() < 0.5 else "off"
            y, entity = f"rauschen.{zustand}", "switch.rauschen"
        k.lifts_folge(["lampe.on", "lampe.off", "rauschen.on", "rauschen.off"])
        k.lerne_folge(y)
        t += timedelta(minutes=2)
        k.setze(entity, zustand, t)
    for zustand, erwartet, falsch in (("on", "lampe.off", "lampe.on"),
                                       ("off", "lampe.on", "lampe.off")):
        k.setze("light.lampe", zustand, t)
        k.bereite_folge(t)
        werte = dict(zip(["lampe.on", "lampe.off"], k.lifts_folge(["lampe.on", "lampe.off"])))
        assert werte[erwartet] > werte[falsch] + 1.0, (zustand, werte)
        k.lerne_folge("rauschen.on")


def test_anwesenheit_aus_der_lage_schlaegt_die_uhr():
    """Der simulierte Haushalt: Person A arbeitet oft zu Hause am PC, ihre
    Anwesenheit verrät die Lage (Auto, PC, Fernseher), nicht die Uhr.
    Person B hat feste Zeiten — dort ist die Uhrzeit-Gewohnheit schon
    stark, der Kortex darf nicht spürbar schlechter sein.

    Gemessen (35 Tage, 4 Wochen Training, ausgewogene Trefferquote):
    A Saat 1 94,6 % gegen bestenfalls 61,7 %, Saat 3 91,7 % gegen 77,4 %;
    B 95,4 % gegen 91,2 % und 88,4 % gegen 91,6 %."""
    for saat in (1, 3):
        h = simuliere_haushalt(tage=35, saat=saat)
        erg = messe_anwesenheit(h.entitaeten, h.ereignisse, dict(TRACKER), h.zuhause, 4)
        for person, vorsprung in (("a", 0.10), ("b", -0.05)):
            b = erg.je_person[person]
            bester = max(b[g].balanciert for g in ("P0", "P1", "P2"))
            assert b["Kortex"].balanciert > bester + vorsprung, (saat, person, erg.zeilen())
        a = erg.je_person["a"]
        assert a["Kortex"].brier_mittel < min(a[g].brier_mittel for g in ("P0", "P1", "P2"))


def test_anwesenheit_nennt_ihre_belege():
    h = simuliere_haushalt(tage=21, saat=1)
    k = AssociationCortex()
    thalamus = _thalamus(h.entitaeten)
    for ereignis in h.ereignisse:
        zufuehren(k, thalamus, ereignis)
    ende = h.ereignisse[-1].ts
    k.setze(TRACKER["a"], "unknown", ende)  # das Handy schweigt
    auskunft = k.anwesenheit(TRACKER["a"], ende)
    assert 0.0 <= auskunft["zuhause"] <= 1.0
    assert auskunft["takte"] > 1000
    assert auskunft["belege"], "ohne Belege ist die Auskunft nicht prüfbar"
    assert all(isinstance(m, str) and isinstance(b, float) for m, b in auskunft["belege"])


def _lauf(k: AssociationCortex, thalamus, ereignisse):
    """Wie die Engine: Lage setzen, Lage-Experte lernt und sagt vorher."""
    spur = []
    kandidaten = ["light.wohnzimmer=on", "light.wohnzimmer=off",
                  "media_player.fernseher=on", "media_player.fernseher=off"]
    for ereignis in ereignisse:
        k.lerne_folge(f"{ereignis.entity}={ereignis.zustand}")
        zufuehren(k, thalamus, ereignis)
        k.bereite_folge(ereignis.ts)
        spur.append(tuple(k.lifts_folge(kandidaten)))
        spur.append(tuple(k.kandidaten_folge(5)))
    return spur


def test_round_trip_setzt_bitgleich_fort():
    h = simuliere_haushalt(tage=21, saat=2)
    thalamus = _thalamus(h.entitaeten)
    mitte = len(h.ereignisse) // 2
    a = AssociationCortex()
    _lauf(a, thalamus, h.ereignisse[:mitte])
    b = AssociationCortex()
    b.from_dict(json.loads(json.dumps(a.to_dict())))
    assert _lauf(a, thalamus, h.ereignisse[mitte:]) == _lauf(b, thalamus, h.ereignisse[mitte:])
    ende = h.ereignisse[-1].ts
    for ziel in TRACKER.values():
        assert a.anwesenheit(ziel, ende) == b.anwesenheit(ziel, ende)
    assert a.zusammenhaenge(top=20) == b.zusammenhaenge(top=20)


def test_der_wirt_bestimmt_die_ziele():
    """Router legen einen Tracker je Gerät an. Gibt der Wirt nur Personen
    als Ziel an, ist der Tracker des PCs ein Indiz der Lage, kein Ziel —
    auch nach dem Laden eines Stands, in dem er noch Ziel war."""
    nur_personen = AssociationCortex(ziel=lambda e: e.startswith("person."))
    nur_personen.setze("person.a", "home", START)
    nur_personen.setze("device_tracker.pc", "home", START)
    assert nur_personen.ziele == ["person.a"]
    assert "device_tracker.pc" in nur_personen.im_blick
    nur_personen.bereite_folge(START)
    assert any(e == "device_tracker.pc" for e, _ in nur_personen._schnapp)

    alle = AssociationCortex()
    alle.setze("person.a", "home", START)
    alle.setze("device_tracker.pc", "home", START)
    assert alle.ziele == ["person.a", "device_tracker.pc"]
    neu = AssociationCortex(ziel=lambda e: e.startswith("person."))
    neu.from_dict(json.loads(json.dumps(alle.to_dict())))
    assert neu.ziele == ["person.a"]


def test_ziel_darf_wechseln():
    """In Home Assistant ändert sich, wem welcher Tracker gehört: Wer
    Ziel wird, wird es auch, wenn er schon als Merkmal im Blick war."""
    besitz = set()
    k = AssociationCortex(ziel=lambda e: e.startswith("person.") or e in besitz)
    k.setze("device_tracker.handy", "home", START)
    assert k.ziele == [] and "device_tracker.handy" in k.im_blick
    besitz.add("device_tracker.handy")
    k.setze("device_tracker.handy", "not_home", START + timedelta(minutes=5))
    assert k.ziele == ["device_tracker.handy"]
    besitz.clear()
    k.setze("device_tracker.handy", "home", START + timedelta(minutes=10))
    assert k.ziele == []


def test_die_lage_spricht_dem_handy_nicht_nach():
    """Die Anwesenheit schließt allein aus den Geräten: Was der Tracker
    selbst meldet (oder ein anderer Tracker), ändert die Schätzung nicht.
    Liegt das Handy daheim, während Auto und PC weg sind, widerspricht
    das Lagebild ihm — statt es nachzusprechen."""
    h = simuliere_haushalt(tage=21, saat=1)
    k = AssociationCortex()
    thalamus = _thalamus(h.entitaeten)
    for ereignis in h.ereignisse:
        zufuehren(k, thalamus, ereignis)
    ende = h.ereignisse[-1].ts
    schaetzungen = []
    for meldet in ("home", "not_home", "unknown"):
        k.setze(TRACKER["a"], meldet, ende)
        k.setze(TRACKER["b"], meldet, ende)
        schaetzungen.append(k.anwesenheit(TRACKER["a"], ende)["zuhause"])
    assert schaetzungen[0] == schaetzungen[1] == schaetzungen[2]


def _zusammenhaenge_referenz(k, entity_id=None, top=10, min_takte=12.0):
    """Die erste Fassung: rundet und sortiert jedes Paar der Tafel."""
    k._paar_nachtragen()
    if k.paar_takte <= 0:
        return []
    breite = k.MAX_PAAR_MERKMALE
    n = len(k.paar_merkmale)
    diag = [k.paar[i * breite + i] for i in range(n)]
    aus = []
    for i in range(n):
        e_i, z_i = k.paar_merkmale[i]
        if entity_id is not None and e_i != entity_id:
            continue
        if diag[i] < min_takte:
            continue
        for j in range(n):
            e_j, z_j = k.paar_merkmale[j]
            if e_j == e_i or diag[j] <= 0:
                continue
            gemeinsam = k.paar[i * breite + j]
            if gemeinsam < min_takte:
                continue
            p = gemeinsam / diag[i]
            lift = p / (diag[j] / k.paar_takte)
            aus.append({
                "wenn": f"{e_i}={z_i}", "dann": f"{e_j}={z_j}",
                "p": round(p, 3), "lift": round(lift, 2),
                "takte": int(gemeinsam),
            })
    aus.sort(key=lambda d: (-(d["lift"] * min(1.0, d["p"] * 2)), d["wenn"], d["dann"]))
    return aus[:top]


def test_zusammenhaenge_bitgleich():
    """Die schnelle Lesung (nur runden, was gewinnen kann) gibt dieselbe
    Liste wie die erste Fassung — auch an den Rändern: kleine und volle
    Tafel, eine Entität, top größer als alle Paare, top 0."""
    for saat, anzahl, ereignisse in ((1, 6, 1500), (2, 40, 12000), (3, 120, 30000)):
        k = AssociationCortex()
        wuerfel = random.Random(saat)
        gewicht = [wuerfel.random() ** 3 for _ in range(anzahl)]
        t = START
        for nr in range(ereignisse):
            t += timedelta(seconds=wuerfel.randint(5, 120))
            e = wuerfel.choices(range(anzahl), gewicht)[0]
            k.setze(f"switch.e{e}", wuerfel.choice(["on", "off", "idle"]), t)
            if nr % 300 == 0:
                k.setze("person.a", wuerfel.choice(["home", "not_home"]), t)
        for kw in ({}, {"top": 50}, {"top": 100000}, {"top": 0}, {"min_takte": 1.0},
                   {"entity_id": "switch.e1"}, {"entity_id": "person.a", "top": 3}):
            assert k.zusammenhaenge(**kw) == _zusammenhaenge_referenz(k, **kw), (saat, kw)
