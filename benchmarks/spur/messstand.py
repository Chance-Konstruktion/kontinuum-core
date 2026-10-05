"""Der Messkern (Stufe 1, Protokoll §§ 1/4/5): Engine gegen B0/B1/B2.

Eine Schleife, vier Systeme, ein Strom — predict-then-learn fuer alle:

    fuer jedes Ereignis i:
        vorhersage fuer token_i  <- Zustand nach Ereignis i-1
        vergleiche               (Top-1/Top-3, Kategorie, Kalibrierung)
        lerne Ereignis i         (alle vier, immer — auch im Training)

Die Engine macht ihre Ordnung selbst (predict vor learn in `observe()`);
ihre Vorhersage fuer token_i ist die des Snapshots von Ereignis i-1.
Verworfene Ereignisse (Reticular-Gate, gleiches Token) aendern das ZIEL
nicht — die Spur bleibt der Strom; die Engine behaelt in dem Fall ihre
letzte Vorhersage (sie hat nichts Neues gesehen).

Das Ziel ist immer der TOKEN-STROM der Spur (`tokenisieren`), nie eine
Engine-Auskunft — sonst maesse man die Engine gegen sich selbst.

Dazu die rollierenden Ursprunge aus §4: k Wochen Training, Woche k+1
Test, k = min … W-1. Jede Zahl traegt ihren Ursprung; die Tafel zeigt
spaeter Median [Min-Max]. Saaten gibt es in der Simulation (>= 5), bei
echten Spuren sind die Ursprunge die Wiederholung.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from kontinuum_core import KontinuumEngine

from .gegner import gegner
from .spur import Entitaet, Spur, kategorie, token_zerlegen, tokenisieren

#: Kalibrierungs-Eimer (0-0,1, …, 0,9-1,0) — Protokoll § 5.
EIMER = 10

#: Die vier Systeme des Messstands, in fester Reihenfolge.
SYSTEME = ("Engine", "B0", "B1", "B2")

#: Dazu die ROHE Hippocampus-Liste vor dem Modul-Ranking — Protokoll § 5
#: („Rohliste vs. gerankt"): Erst der Vergleich zeigt, ob die Module die
#: Reihenfolge verbessern oder verschlechtern.
ENGINE_ROH = "Engine roh"


@dataclass
class Systembilanz:
    """Was ein System in einem Ursprung geleistet hat."""

    name: str
    top1: int = 0
    top3: int = 0
    gesamt: int = 0
    #: Treffer und Gesamtzahl je Kategorie des TATSAECHLICHEN Tokens.
    je_kategorie_treffer: Dict[str, int] = field(default_factory=dict)
    je_kategorie_gesamt: Dict[str, int] = field(default_factory=dict)
    #: (Konfidenz der Top-1-Vorhersage, Treffer) je bewertetem Paar.
    kalibrierung: List[Tuple[float, int]] = field(default_factory=list)

    def trefferquote(self) -> float:
        return self.top1 / self.gesamt if self.gesamt else 0.0

    def trefferquote3(self) -> float:
        return self.top3 / self.gesamt if self.gesamt else 0.0

    def kategorien(self) -> Dict[str, float]:
        return {
            kat: self.je_kategorie_treffer.get(kat, 0) / n
            for kat, n in sorted(self.je_kategorie_gesamt.items())
            if n
        }

    def _eimer(self) -> Dict[int, List[Tuple[float, int]]]:
        eimer: Dict[int, List[Tuple[float, int]]] = {}
        for konfidenz, treffer in self.kalibrierung:
            nummer = min(EIMER - 1, max(0, int(konfidenz * EIMER)))
            eimer.setdefault(nummer, []).append((konfidenz, treffer))
        return eimer

    def ece(self) -> float:
        """Expected Calibration Error ueber die 10 Eimer, n-gewichtet."""
        eimer = self._eimer()
        gesamt = len(self.kalibrierung)
        if not gesamt:
            return 0.0
        fehler = 0.0
        for werte in eimer.values():
            n = len(werte)
            schnitt = sum(k for k, _ in werte) / n
            quote = sum(t for _, t in werte) / n
            fehler += n / gesamt * abs(schnitt - quote)
        return fehler

    def kalibrierungs_eimer(self) -> List[Tuple[int, int, float, float]]:
        """(Eimer, n, mittlere Konfidenz, Trefferquote) — nur gefuellte."""
        ausgabe = []
        for nummer, werte in sorted(self._eimer().items()):
            n = len(werte)
            ausgabe.append((
                nummer, n,
                sum(k for k, _ in werte) / n,
                sum(t for _, t in werte) / n,
            ))
        return ausgabe

    def bericht(self) -> str:
        zeilen = [
            f"{self.name:<6} Top-1 {self.trefferquote():6.1%}  "
            f"Top-3 {self.trefferquote3():6.1%}  (n={self.gesamt})  "
            f"ECE {self.ece():5.1%}",
        ]
        for kat, quote in self.kategorien().items():
            n = self.je_kategorie_gesamt[kat]
            zeilen.append(f"       {kat:<14} {quote:6.1%}  (n={n})")
        return "\n".join(zeilen)


@dataclass
class UrsprungsErgebnis:
    """Ein rollierender Ursprung: k Wochen Training, Woche k+1 Test."""

    ursprung: int
    test_von: datetime
    test_bis: datetime
    test_ereignisse: int
    systeme: Dict[str, Systembilanz]

    def bericht(self) -> str:
        kopf = (
            f"Ursprung {self.ursprung}: Training {self.ursprung} Wochen, "
            f"Testwoche ab {self.test_von.date()} "
            f"({self.test_ereignisse} Ereignisse im Strom)"
        )
        return "\n".join([kopf] + [b.bericht() for b in self.systeme.values()])


@dataclass
class MessErgebnis:
    spur: Spur
    ursprunge: List[UrsprungsErgebnis] = field(default_factory=list)

    def bericht(self) -> str:
        teile = [
            f"MESSSTAND — {self.spur.kopf.haus} ({self.spur.kopf.haus_typ}), "
            f"{len(self.spur.ereignisse)} Ereignisse, "
            f"{len(self.ursprunge)} Ursprünge",
            "=" * 62,
        ]
        for ursprung in self.ursprunge:
            teile.append(ursprung.bericht())
            teile.append("-" * 62)
        return "\n".join(teile)


# ---------------------------------------------------------------------------
# Registrierung: dieselbe Tabelle wie die Spur, dieselben Uebersteuerungen
# ---------------------------------------------------------------------------


def registriere_engine(engine: KontinuumEngine,
                       entitaeten: Sequence[Entitaet]) -> None:
    """Registriert die Entitaetentabelle auf der Engine — mit denselben
    Semantik-Uebersteuerungen wie `spur.registriere`; sonst faende die
    Engine andere Tokens als der Token-Lauf-Beweis."""
    for e in entitaeten:
        if e.semantik:
            engine.thalamus.custom_semantic_rules.append({
                "semantic": e.semantik,
                "entity_regex": "^" + re.escape(e.id.lower()) + "$",
            })
    for e in entitaeten:
        engine.register_entity(
            e.id, ha_area=e.raum, domain=e.domain,
            device_class=e.geraeteklasse or "",
        )


# ---------------------------------------------------------------------------
# Die Schleife
# ---------------------------------------------------------------------------


def _engine_vorhersage(snapshot, engine: KontinuumEngine,
                       letzte: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """Token + Konfidenz aus einem Engine-Snapshot. Verworfene Ereignisse
    (leere Liste) lassen die letzte gueltige stehen — die Engine hat
    nichts Neues gesehen."""
    if not snapshot.predictions:
        return letzte
    ausgabe: List[Tuple[str, float]] = []
    for eintrag in snapshot.predictions:
        token = engine.thalamus.decode_token(eintrag[0])
        if not token or token.startswith("?"):
            continue
        ausgabe.append((token, float(eintrag[2])))
    return ausgabe or letzte


class _Rohspion:
    """Fängt die ROHE Hippocampus-Liste ab, bevor `_rank_predictions` sie
    umsortiert (Protokoll § 5). Eine Messstand-Huelle: Sie beobachtet,
    sie veraendert nichts — der Originalaufruf laeuft unveraendert."""

    def __init__(self, engine: KontinuumEngine):
        self.engine = engine
        self._roh: List[str] = []
        self._original = engine._rank_predictions

        def fangen(vorhersagen, *args, **kwargs):
            self._roh = [
                self.engine.thalamus.decode_token(eintrag[0])
                for eintrag in (vorhersagen or [])
            ]
            return self._original(vorhersagen, *args, **kwargs)

        engine._rank_predictions = fangen  # type: ignore[method-assign]

    def vorhersage(self) -> List[Tuple[str, float]]:
        # Konfidenz der Rohliste: dieselbe Anzeige wie im Snapshot
        # (Platz 3 der Tupel) — die Rohliste traegt (token, prob, conf,
        # source, n_obs); hier zaehlt die Konfidenz.
        return [(token, 1.0) for token in self._roh]


def messe_ursprung(spur: Spur, train_wochen: int,
                   engine_bauer: Optional[Callable[[], KontinuumEngine]] = None,
                   rohliste: bool = True) -> UrsprungsErgebnis:
    """Ein rollierender Ursprung: die ersten k Wochen lernen, Woche k+1
    bewerten. Alle vier Systeme sehen denselben Strom in derselben
    Ordnung; gezielt wird nur die Testwoche."""
    if train_wochen < 1:
        raise ValueError("train_wochen mindestens 1")
    if not spur.ereignisse:
        raise ValueError("Leere Spur — nichts zu messen.")

    ziele = tokenisieren(spur)  # das Ziel ist der Strom, nicht die Engine
    start = spur.ereignisse[0].ts
    grenze_train = start + timedelta(days=7 * train_wochen)
    test_ende = grenze_train + timedelta(days=7)

    vokabular = sorted(set(ziele))  # Existenzwissen (§ 1 Regel 3)
    bauer = engine_bauer or (lambda: KontinuumEngine())

    def _gebaut() -> KontinuumEngine:
        maschine = bauer()
        # Determinismus: Die Schlaf-Konsolidierung traegt eine
        # Traum-Replay-Phase, die mit dem GLOBALEN Zufall sampelt —
        # ohne feste Saat haengt jede Zahl am Vorlauf im selben Prozess
        # (derselbe Fund wie benchmarks/replay.py; dort steht die
        # gleiche Zeile). Gleiche Spur ⇒ gleiche Zahl, sonst ist keine
        # Messung reproduzierbar.
        maschine.sleep_consolidation._rng = random.Random(0)
        return maschine

    engine = _gebaut()
    registriere_engine(engine, spur.kopf.entitaeten)

    namen = list(SYSTEME) + ([ENGINE_ROH] if rohliste else [])
    systeme = {name: Systembilanz(name) for name in namen}
    schar = gegner(vokabular)
    spion = _Rohspion(engine) if rohliste else None

    letzte_engine: List[Tuple[str, float]] = []
    letzte_roh: List[Tuple[str, float]] = []
    for i, ereignis in enumerate(spur.ereignisse):
        if ereignis.ts >= test_ende:
            # Nach dem Testfenster aendert kein Ereignis mehr eine Zahl:
            # die Vorhersagen fuer die Testwoche stehen fest, danach wird
            # nichts mehr bewertet. Der Leser erzwingt nicht-absteigende
            # Zeit (spur.py), also ist der Abbruch exakt. Ohne ihn liefe
            # jeder Ursprung ueber die GANZE Spur — bei CASAS (1,6 Mio
            # Ereignisse) der Unterschied zwischen Minuten und Stunden.
            break
        # 1) Bewerten (nur Testwoche): die Vorhersage fuer token_i
        #    entstand aus dem Zustand nach Ereignis i-1 — auch am
        #    Rand: die letzte Trainingsvorhersage zielt auf das erste
        #    Testereignis (Protokoll § 1 Regel 4).
        if i > 0 and grenze_train <= ereignis.ts < test_ende:
            ziel = ziele[i]
            kat = kategorie(spur.kopf.haus_typ, token_zerlegen(ziel)[1])
            zeit_davor = spur.ereignisse[i - 1].ts
            vorhersagen = {
                "Engine": letzte_engine,
                "B0": schar["B0"].vorhersage(zeit_davor, 3),
                "B1": schar["B1"].vorhersage(zeit_davor, 3),
                "B2": schar["B2"].vorhersage(zeit_davor, 3),
            }
            if spion is not None:
                vorhersagen[ENGINE_ROH] = letzte_roh
            for name, liste in vorhersagen.items():
                bilanz = systeme[name]
                bilanz.gesamt += 1
                bilanz.je_kategorie_gesamt[kat] = \
                    bilanz.je_kategorie_gesamt.get(kat, 0) + 1
                if liste:
                    if liste[0][0] == ziel:
                        bilanz.top1 += 1
                        bilanz.je_kategorie_treffer[kat] = \
                            bilanz.je_kategorie_treffer.get(kat, 0) + 1
                    if any(t == ziel for t, _ in liste[:3]):
                        bilanz.top3 += 1
                    bilanz.kalibrierung.append(
                        (float(liste[0][1]), 1 if liste[0][0] == ziel else 0)
                    )

        # 2) Lernen — alle vier, jedes Ereignis, auch im Training.
        schnappschuss = engine.observe({
            "entity_id": ereignis.entity,
            "new_state": ereignis.zustand,
            "old_state": ereignis.alt,
            "timestamp": ereignis.ts,
        })
        letzte_engine = _engine_vorhersage(schnappschuss, engine, letzte_engine)
        if spion is not None:
            letzte_roh = spion.vorhersage()
        token_i = ziele[i]
        for name, g in schar.items():
            if name == "B1":
                g.beobachte(token_i, ereignis.ts)
            else:
                g.beobachte(token_i)

    test_ereignisse = sum(
        1 for e in spur.ereignisse if grenze_train <= e.ts < test_ende
    )
    return UrsprungsErgebnis(
        ursprung=train_wochen,
        test_von=grenze_train,
        test_bis=test_ende,
        test_ereignisse=test_ereignisse,
        systeme=systeme,
    )


def messe(spur: Spur, min_train_wochen: int = 4,
          engine_bauer: Optional[Callable[[], KontinuumEngine]] = None
          ) -> MessErgebnis:
    """Rollierende Ursprunge: k = min_train_wochen … W-1, Testwoche k+1.
    Zu wenige Wochen sind kein stiller Abbruch, sondern eine Ausnahme
    mit Namen (§ 4: die Spanne braucht Wiederholungen)."""
    tage = (spur.ereignisse[-1].ts - spur.ereignisse[0].ts).days + 1
    wochen = tage // 7
    if wochen <= min_train_wochen:
        raise ValueError(
            f"Zu wenige Wochen fuer rollierende Ursprünge: {wochen} Wochen "
            f"vorhanden, gebraucht > {min_train_wochen} Training + 1 Test. "
            "Die Spur verlaengern (Protokoll § 4: >= 10 Wochen empfohlen)."
        )
    ergebnis = MessErgebnis(spur=spur)
    for k in range(min_train_wochen, wochen):
        ergebnis.ursprunge.append(messe_ursprung(spur, k, engine_bauer))
    return ergebnis


def gepaarte_differenz(ergebnis: MessErgebnis,
                       gegner: Sequence[str] = ("B0", "B1", "B2")) -> dict:
    """Die gepaarte Differenz Engine − bester Gegner (Abnahme 13642, Punkt 3).

    „Bester Gegner“ ist der mit der höchsten Median-Top-1-Quote über die
    Ursprünge; die Differenz wird JE URSPRUNG gebildet und dann als
    Median [Min–Max] berichtet — so erkennt man einen knappen Sieg als
    knapp. (Die mittlere Differenz über die bewerteten Paare ist mit der
    Quotendifferenz identisch; die Paarung ist der gleiche Strom.)"""
    import statistics

    quoten = {
        name: statistics.median(
            [u.systeme[name].trefferquote() for u in ergebnis.ursprunge]
        )
        for name in gegner
    }
    bester = max(quoten, key=lambda name: (quoten[name], name))
    differenzen = [
        u.systeme["Engine"].trefferquote() - u.systeme[bester].trefferquote()
        for u in ergebnis.ursprunge
    ]
    return {
        "bester_gegner": bester,
        "gegner_median": quoten[bester],
        "differenzen": differenzen,
        "median": statistics.median(differenzen),
        "min": min(differenzen),
        "max": max(differenzen),
    }


def _demo() -> int:
    """Kleiner Aufruf zum Anschauen (und fuer die spaetere Tafel):

        python -m benchmarks.spur.messstand [klassisch|geraete] [wochen] [saat]
    """
    import sys

    from .simulator import simuliere

    haus_typ = sys.argv[1] if len(sys.argv) > 1 else "klassisch"
    wochen = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    saat = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    simulation = simuliere(haus_typ, tage=wochen * 7, saat=saat)
    ergebnis = messe(simulation.spur, min_train_wochen=4)
    print(ergebnis.bericht())
    differenz = gepaarte_differenz(ergebnis)
    print(f"Gepaarte Differenz (Engine − {differenz['bester_gegner']}): "
          f"Median {differenz['median']:+.1%} "
          f"[{differenz['min']:+.1%} .. {differenz['max']:+.1%}]")
    print(f"Spur: roh {simulation.roh_anzahl} -> behalten "
          f"{len(simulation.spur.ereignisse)} (Eimer-Verlust "
          f"{simulation.bericht['eimer_verlust']})")
    if haus_typ == "geraete":
        print(f"Leck-Tag: roh {simulation.leck_roh} -> behalten "
              f"{simulation.leck_spur}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_demo())


__all__ = [
    "EIMER",
    "SYSTEME",
    "ENGINE_ROH",
    "Systembilanz",
    "UrsprungsErgebnis",
    "MessErgebnis",
    "registriere_engine",
    "messe_ursprung",
    "messe",
    "gepaarte_differenz",
]
