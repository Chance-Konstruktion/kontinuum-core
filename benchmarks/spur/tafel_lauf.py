#!/usr/bin/env python3
"""Tafel-Neuaufnahme: alle fuenf CASAS-Haeuser, Zeitgeber an.

Die Messung des Messbienen-Laufs (kontinuum-core#2, Stufe 1): rollierende
Ursprünge je Haus, je Ursprung lernen die ersten k Wochen, bewertet wird
Woche k+1. Regel fuer die Ursprünge: bis zu 6, gleichmäßig von 4 bis
W-1 (W = Wochen des Hauses) — dieselbe Regel wie in der Tafel vom
07.10.2026, damit Staende vergleichbar bleiben.

Ausgabe: JSON + Klartext, je Haus Median Top-1 aller Systeme (inkl. der
Roh-Zeilen des Messstands: Hippocampus pur und — je nach Stand — „vor
Ranking" oder „Engine alt") und die gepaarten Differenzen Engine−B2 und
pur−B2.

    python -m benchmarks.spur.tafel_lauf --daten ORDNER --ziel ORDNER STAND

`--daten` zeigt auf einen Ordner mit aruba.jsonl, tulum2.jsonl, ... (die
konvertierten CASAS-Spuren, siehe benchmarks/spur/casas.py); `--ziel` ist
der Ablageort der Ergebnisse. Beide gehoeren NICHT ins Repo (echte
Haushistorie, .gitignore).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from benchmarks.spur.messstand import (
    ENGINE_ROH,
    ENGINE_VOR_RANKING,
    messe_ursprung,
)
from benchmarks.spur.spur import lies_spur

HAEUSER = ["aruba", "tulum2", "tulum1", "milan", "cairo"]


def ursprungs_satz(wochen: int, max_n: int = 6) -> list[int]:
    von, bis = 4, wochen - 1
    if bis < von:
        return []
    n = min(max_n, bis - von + 1)
    if n == 1:
        return [von]
    werte = sorted({int(round(von + i * (bis - von) / (n - 1)))
                    for i in range(n)})
    return werte


def main() -> int:
    p = argparse.ArgumentParser(prog="tafel_lauf")
    p.add_argument("--daten", required=True,
                   help="Ordner mit <haus>.jsonl (CASAS-Spuren)")
    p.add_argument("--ziel", required=True,
                   help="Ordner fuer tafel_<stand>.{json,txt,log}")
    p.add_argument("stand", help="Etikett des Stands (Commit-Kurzhash)")
    args = p.parse_args()

    daten, ziel = Path(args.daten), Path(args.ziel)
    ziel.mkdir(parents=True, exist_ok=True)
    tafel = {"stand": args.stand, "zeitgeber": True, "haeuser": {}}
    zeilen = [f"TAFEL — Stand {args.stand}, Zeitgeber an, "
              f"Ursprünge: bis zu 6 gleichmäßig 4..W-1", "=" * 66]
    for haus in HAEUSER:
        pfad = daten / f"{haus}.jsonl"
        if not pfad.exists():
            print(f"FEHLT: {pfad}", flush=True)
            continue
        t0 = time.time()
        spur = lies_spur(pfad)
        tage = (spur.ereignisse[-1].ts - spur.ereignisse[0].ts).days + 1
        wochen = tage // 7
        ks = ursprungs_satz(wochen)
        print(f"### {haus}: {len(spur.ereignisse)} Ereignisse, {wochen} Wochen, "
              f"Ursprünge {ks}", flush=True)
        eintrag = {"ereignisse": len(spur.ereignisse), "wochen": wochen,
                   "ursprunge": [], "median": {}}
        systeme = None
        for k in ks:
            uk = messe_ursprung(spur, k)
            systeme = list(uk.systeme)
            werte = {n: {"top1": b.top1, "top3": b.top3, "gesamt": b.gesamt}
                     for n, b in uk.systeme.items()}
            eintrag["ursprunge"].append({"k": k, "werte": werte,
                                         "test_ereignisse": uk.test_ereignisse})
            kurz = " ".join(f"{n}={b.trefferquote():.1%}"
                            for n, b in uk.systeme.items())
            print(f"  k={k:>2}: {kurz}", flush=True)

        def median(name):
            return statistics.median(
                [u["werte"][name]["top1"] / u["werte"][name]["gesamt"]
                 for u in eintrag["ursprunge"] if u["werte"][name]["gesamt"]])

        eintrag["median"] = {n: median(n) for n in systeme}
        # Gepaarte Differenzen (Engine−B2, pur−B2) je Ursprung.
        for schluessel, links in (("gepaart_engine_b2", "Engine"),
                                  ("gepaart_pur_b2", ENGINE_ROH)):
            diffs = [
                u["werte"][links]["top1"] / u["werte"][links]["gesamt"]
                - u["werte"]["B2"]["top1"] / u["werte"]["B2"]["gesamt"]
                for u in eintrag["ursprunge"]
                if u["werte"][links]["gesamt"] and u["werte"]["B2"]["gesamt"]
            ]
            eintrag[schluessel] = {
                "median": statistics.median(diffs),
                "min": min(diffs), "max": max(diffs), "je_ursprung": diffs,
            }
        eintrag["minuten"] = (time.time() - t0) / 60
        tafel["haeuser"][haus] = eintrag
        med = eintrag["median"]
        zweite = ENGINE_VOR_RANKING if ENGINE_VOR_RANKING in med else "Engine alt"
        zeilen.append(
            f"{haus:>7} (W={wochen:>2}, {len(ks)} Ursprünge, "
            f"{eintrag['minuten']:.0f} min)\n"
            f"   Median Top-1: Engine {med.get('Engine', 0):.1%} | "
            f"{zweite} {med.get(zweite, 0):.1%} | "
            f"pur {med.get(ENGINE_ROH, 0):.1%} | B0 {med.get('B0', 0):.1%} | "
            f"B1 {med.get('B1', 0):.1%} | B2 {med.get('B2', 0):.1%}\n"
            f"   gepaart Engine−B2 {eintrag['gepaart_engine_b2']['median']:+.1%} "
            f"[{eintrag['gepaart_engine_b2']['min']:+.1%} .. "
            f"{eintrag['gepaart_engine_b2']['max']:+.1%}] | "
            f"pur−B2 {eintrag['gepaart_pur_b2']['median']:+.1%} "
            f"[{eintrag['gepaart_pur_b2']['min']:+.1%} .. "
            f"{eintrag['gepaart_pur_b2']['max']:+.1%}]"
        )
        print("\n".join(zeilen[-4:]), flush=True)

    (ziel / f"tafel_{args.stand}.json").write_text(
        json.dumps(tafel, ensure_ascii=False, indent=1), encoding="utf-8")
    (ziel / f"tafel_{args.stand}.txt").write_text(
        "\n".join(zeilen) + "\n", encoding="utf-8")
    print("\n".join(zeilen), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
