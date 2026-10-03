"""Waechter der Kontinuum-Spur (Stufe 1, Schritt 1).

Prueft eine Spur-Datei (oder alle Dateien eines Ordners) und druckt NUR
AGGREGATE — keine Rohzeilen. Das ist Absicht (Protokoll § 2, Regel 6):
Beim Hantieren mit echter Haushistorie darf nichts in Logs rieseln, was
Raeume oder Zeiten verraet.

    python -m benchmarks.spur.pruefe_spur DATEI [DATEI ...]
    python -m benchmarks.spur.pruefe_spur --ordner build/spuren/

Rueckgabe: 0 = gueltig, 1 = Fehler (mit Namen), 2 = Aufruf-Fehler.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .spur import SpurFehler, lies_spur, tokenisieren, uebersicht


def pruefe(pfad: Path) -> dict:
    """Liest und prueft EINE Spur; wirft SpurFehler mit Namen."""
    spur = lies_spur(pfad)
    tokens = tokenisieren(spur)  # jedes Ereignis muss ein Token ergeben
    if len(tokens) != len(spur.ereignisse):
        raise SpurFehler(
            f"{pfad.name}: {len(spur.ereignisse)} Ereignisse, aber "
            f"{len(tokens)} Tokens — die Spur ist nicht ausgeduennt."
        )
    bericht = uebersicht(spur)
    bericht["datei"] = pfad.name
    return bericht


def _drucke(bericht: dict) -> None:
    print(f"── {bericht['datei']}")
    print(f"   Haus        : {bericht['haus']} ({bericht['haus_typ']}, "
          f"Quelle {bericht['quelle']})")
    print(f"   Umfang      : {bericht['ereignisse']} Ereignisse, "
          f"{bericht['tage']} Tage, {bericht['entitaeten']} Entitaeten, "
          f"Vokabular {bericht['vokabular']}")
    print(f"   Kategorien  : {bericht['je_kategorie']}")
    if bericht["hypothalamus_luecken"]:
        print(f"   Hypothalamus-Luecken (Vokabular): "
              f"{bericht['hypothalamus_luecken']} "
              f"({', '.join(bericht['hypothalamus_luecken_paare'])})")
    if bericht["marken"]:
        print(f"   Marken (nur Diagnose): {bericht['marken']}")


def main(argv=None) -> int:
    zerleger = argparse.ArgumentParser(
        prog="pruefe_spur",
        description="Prueft Kontinuum-Spuren (JSONL) und druckt nur Aggregate.",
    )
    zerleger.add_argument("dateien", nargs="*", help="Spur-Dateien (JSONL)")
    zerleger.add_argument("--ordner", help="alle *.jsonl in diesem Ordner")
    zerleger.add_argument("--json", action="store_true",
                          help="Aggregate als JSON (maschinenlesbar)")
    args = zerleger.parse_args(argv)

    pfade = [Path(d) for d in args.dateien]
    if args.ordner:
        pfade += sorted(Path(args.ordner).glob("*.jsonl"))
    if not pfade:
        zerleger.print_usage(sys.stderr)
        print("FEHLER: keine Spur angegeben (Datei oder --ordner).", file=sys.stderr)
        return 2

    berichte = []
    for pfad in pfade:
        try:
            berichte.append(pruefe(pfad))
        except SpurFehler as fehler:
            print(f"FEHLER: {pfad}: {fehler}", file=sys.stderr)
            return 1

    if args.json:
        print(json.dumps(berichte, ensure_ascii=False, indent=1))
    else:
        for bericht in berichte:
            _drucke(bericht)
        print(f"OK: {len(berichte)} Spur(en) gueltig.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
