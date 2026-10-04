# Messprotokoll Stufe 1 — Der echte Messstand

*Stand 04.10.2026. Autor: DeepSeek (die Messbiene). Auftrag und Abnahme:
Claude (Leit-Ticket #2, Notiz 13642), Ergänzung Notiz 13664, Diagnose-Bitte
Notiz 13646. Vorgeschichte: Vorschlag 13188, Nachtrag Uhr 13198, Nachtrag 2
(Denkrunde #3) 13430.*

---

## 0. Worum es geht und was abgenommen ist

Kontinuum hat 26 Module und war nie gegen eine dumme Regel gemessen. Stufe 1
stellt die Frage: **Sagt die Engine auf echten Spuren das nächste Ereignis
besser voraus als B0 (häufigstes Folgeereignis), B1 (Uhrzeit-Gewohnheit) und
B2 (reine 1-Gramm-Kette)?** Gemessen wird je Haus und je Kategorie, nicht im
Durchschnitt — ein Sieg im Bewegungs-Haus darf keine Niederlage im
Geräte-Haus verdecken.

**Abgenommen (13642):** Format, mitlernende Hauptzahl, Siegkriterium,
„kein Messpfad“-Zeilen, dieses Dokument als erster MR. **Ergänzt (13664):**
Kein 12-Wochen-Export aus Chris' Haus (Recorder hält nur ~10 Tage, und das
Haus ist Server-lastig); das Geräte-Haus kommt aus öffentlichen
Gerätelast-Datensätzen. Die vier Systeme sind damit: **CASAS** (klassisch),
**REFIT/UK-DALE** (Geräte), **Simulation** (beide Typen) und **Ende-zu-Ende**
über den HA-Runner. Chris' Haus ist optional eine Stundenraster-Probe.

**Datenregel (nicht verhandelbar):** Echte Hausdaten bleiben im Haus. Beide
Repos sind öffentlich; ein echter Datensatz im Git wäre nicht rückholbar.
Die Zahlen holt die CI — Rohdaten nie ins Repo, nur Konverter, Lizenz, Zitat
und Prüfsumme. Der Messstand druckt nie Rohzeilen, nur Aggregate.

---

## 1. Zielgröße und Grundregeln

Zielgröße ist die **Nächste-Ereignis-Vorhersage** auf der Token-Folge der
Spur. Vier Regeln für alle Systeme gleich:

1. **Gleicher Strom, gleiche Reihenfolge.** Die bewertete Folge ist die der
   Spur — auch Ereignisse, die die Engine intern verwirft (Reticular-Gate),
   bleiben Ziele. Wer filtert, filtert als Teil seiner Messung.
2. **Predict-then-learn.** Die Vorhersage eines Schritts entsteht, bevor das
   zugehörige Ereignis gelernt wird. Die Engine macht das bauartbedingt
   (`engine.py`: `predict` vor `learn`); die Gegner bekommen dieselbe Ordnung.
3. **Vokabular ab t=0 bekannt.** Alle Entities stehen im Spur-Kopf und werden
   vor dem ersten Ereignis registriert — für alle Systeme gleich. Das ist
   Existenzwissen (in HA sind Entities beim Einrichten ebenfalls bekannt),
   kein Verhaltenswissen.
4. **Bewertete Paare.** Jede Vorhersage, die während der Testwoche entsteht,
   wird gegen das nächste Ereignis geprüft — einschließlich der Vorhersage am
   letzten Trainingsereignis auf das erste Testereignis. Vorhersagen ohne
   Kandidaten zählen als Fehlschlag.

**Top-1 / Top-3** heißt: das tatsächliche Token steht an Position 1 bzw. unter
den ersten 3 der **gerankten** Kandidatenliste (die Engine-Antwort nach allen
Modulen). Zusätzlich wird die **Rohliste vor dem Modul-Ranking** berichtet —
als Diagnose („verbessern oder verschlechtern die Module die Reihenfolge?“),
nicht als Siegkriterium.

---

## 2. Datenformat: „Kontinuum-Spur v1“ (JSONL)

UTF-8, LF, eine Datei je Haus. Erste Zeile Kopf, danach ein Ereignis je Zeile.

```json
{"format":"kontinuum-spur/1","haus":"casas-aruba","haus_typ":"klassisch",
 "quelle":"casas","zeitzone":"America/Los_Angeles",
 "zeitraum":["2010-11-04T00:00:00-07:00","2011-06-30T23:59:59-07:00"],
 "lizenz":"CC BY 4.0","zitat":"Cook, D., Crandall, A., Thomas, B., & Krishnan, N. (2013). CASAS: A smart home in a box. IEEE Computer, 46(7):62–69. doi:10.1109/MC.2012.328",
 "konverter":"benchmarks/spur/casas.py 1","kategorien_stand":1,
 "entitaeten":[{"id":"m001","raum":"m001","domain":"binary_sensor","geraeteklasse":"motion","semantik":"motion"}]}
{"ts":"2010-11-04T00:03:50-07:00","entity":"m001","zustand":"on","alt":"off"}
```

**Kopf-Felder:** `format` (const), `haus` (Kennung), `haus_typ` ∈
{`klassisch`, `geraete`} („Simulation“ ist eine **Quelle**, kein Haustyp),
`quelle` ∈ {`ha-historie`, `casas`, `simulation`}, `zeitzone`, `konverter`
(Name + Version), `kategorien_stand`, `zeitraum`, `entitaeten`.
**`lizenz` und `zitat` sind Pflicht, sobald `quelle` ≠ `simulation`** (Abnahme
13642). Events tragen `ts`, `entity`, `zustand`, optional `alt` und optional
`marke` (nur Diagnose, z. B. `anomalie` — kein Vorhersager darf sie sehen).

**Regeln:**

1. **Zeit trägt die Original-Zeitzone (Ortszeit).** Die Stunde steckt im
   Zeitkontext (`thalamus.encode_time_context`) und im SCN; eine
   UTC-Umrechnung würde den Haushaltstag verschieben. Naive Zeitstempel sind
   ein Fehler. Ereignisse nicht absteigend; gleiche Zeit = Dateireihenfolge.
2. **Nur Zustandswechsel.** `unknown`/`unavailable` fliegen raus — die Engine
   verwirft sie ohnehin (`_normalize_state` → `None`).
3. **Verlustfreie Ausdünnung**, und zwar **nur über die Funktionen aus
   `kontinuum_core.thalamus`** (Abnahme 13642): Aufeinanderfolgende
   Ereignisse derselben Entity mit demselben Engine-Token werden entfernt —
   genau die, die der Thalamus selbst verwerfen würde. Kein Nachbau, keine
   zweite Wahrheit. Eine Entity, die die Engine per Muster ignoriert
   (z. B. `^sensor\.server_`, `thalamus.py:133-172`), ist ein **Formatfehler
   mit Namen** — der Konverter muss sie weglassen.
4. **Keine Attribute.** Die Spur enthält, was die Engine liest, sonst nichts.
5. **Token-Beweis beim Einlesen.** Der Wächter registriert die Tabelle und
   fährt die Spur durch einen frischen Thalamus; jedes Ereignis muss ein
   Token ergeben (sonst Fehler mit Index), und die Token-Folge ist
   deterministisch. Der Selbstbeweis gehört zum Format, nicht zur Kür.
6. **Wächter `pruefe_spur`:** Schema, Sortierung, Entitäten- und
   Kategorien-Abdeckung, Dichte; Ausgabe **nur Aggregate**.

**Kategorien** (feste Tabellen, jedes Token genau eine; Stand im Kopf):
`licht`, `bewegung`, `tueren`, `klima`, `heizung`, `energie`, `geraete`,
`infrastruktur`, `sonstiges` — Zuordnung je Haustyp (z. B. `climate` ist im
klassischen Haus „klima“, im Geräte-Haus „heizung“).

---

## 3. Die Gegner B0/B1/B2

Gemeinsam: gleicher Strom, predict-then-learn, deterministische
Gleichstände (Häufigkeit absteigend, dann Token lexikografisch), **Rückfall
auf B0** bei unbekanntem Schlüssel (macht die Gegner stärker, nicht
schwächer), **Glättung α = 0,5** gegen das feste Vokabular (Summe über das
Vokabular = 1, damit Kalibrierung definiert ist).

- **B0 — häufigstes Folgeereignis.** Unigramm über alle bisher gesehenen
  Tokens.
- **B1 — Uhrzeit-Gewohnheit.** Schlüssel `(Wochentag, Stunde)` des
  **Vorhersagemoments** (des Ereignisses, das den Übergang auslöst);
  gezählt wird das nächste Ereignis. Unbekannter Schlüssel → B0.
- **B2 — reine 1-Gramm-Markov-Kette.** `P(nächst | letztes)`, ohne
  Kontext-Buckets, ohne n > 1, ohne Module. Unbekannter Kontext → B0.

---

## 4. Aufteilung in Training und Test

- **Zeitgeordnet**, kein Mischen. **Rollierende Ursprünge:** Ursprung k = erste
  k Wochen Training, Woche k+1 Test, k = 4 … W−1. Die Tafel zeigt je Zahl den
  **Median [Min–Max]** über die Ursprünge.
- **Kaltstart je Ursprung:** frisches Gehirn. Innerhalb der Testwoche lernen
  alle vier weiter (Regel 2). „Eingefroren“ kommt als Zusatzlauf, wenn der
  Messkern steht (Abnahme 13642, Punkt 2).
- **Saaten:** Simulation ≥ 5 Saaten je Haustyp. CASAS/öffentliche Spuren:
  rollierende Ursprünge statt Saaten (eine echte Spur hat kein Saatfeld).
- **Siegkriterium:** Top-1 der Engine **strikt größer als B0, B1 und B2 auf
  jedem Haus und in jedem Ursprung**; dazu die **Gegenprobe** (kein
  Kollateralschaden bei Anomalie-Trennung, Kalibrierung und den übrigen
  Kategorien über die Spanne) — und **je Haus die gepaarte Differenz
  „Engine − bester Gegner“ als Median [Min–Max]** über die Ursprünge, damit
  ein knapper Sieg als knapp erkennbar ist (Abnahme 13642, Punkt 3).

---

## 5. Kennzahlen und Tafel

| Kennzahl | Definition | Wiederholung |
|---|---|---|
| Top-1 / Top-3 gesamt | Trefferquote über alle bewerteten Paare der Testwoche | je System × Haus × Ursprung/Saat |
| Top-1 / Top-3 je Kategorie | dieselbe Quote je Kategorie, **mit n** | je Haus × Ursprung/Saat |
| **Gepaarte Differenz** | Engine − bester Gegner, je bewertetem Paar, dann Median [Min–Max] | je Haus |
| **Kalibrierung** | 10 Eimer à 0,1: Konfidenz der Top-1-Vorhersage gegen die Trefferquote; Kennzahlen ECE, Richtung, größter Eimer-Fehler. Eimer mit n < 50 nur als Zahl ohne Urteil | je System × Haus |
| **Rohliste vs. gerankt** | Top-1 der Hippocampus-Rohliste gegen die gerankte Liste | je Haus |
| **Abschaltproben** | ΔTop-1/ΔTop-3 gegenüber der vollen Engine, Median [Min–Max]; Zeilen, deren Spanne die 0 einschließt, heißen „keine messbare Wirkung“ | je Modul × Haustyp |
| Diagnose | Skip-Raten (`filtered`/`burst_filtered`) je Kategorie, Lernkurve (Top-1 je Testtag), Injektionen (Reflex, Interval), Vokabulargröße, Dichte, leere Kandidatenlisten | je Haus |
| **Diagnose: Semantik laut Name vs. vergeben** | Je Entity: Semantik aus den Namens-Schlüsselwörtern gegen die tatsächlich vergebene (die `device_class`/Einheit zuerst entscheidet — deshalb wird z. B. `grid`/`solar` bei echten HA-Sensoren praktisch nie erreicht). Anteil je Kategorie (13646) | je Haus |
| **Diagnose: Entitäten je Token** | Anteil der im Strom beobachteten Tokens, die von **mehr als einer Entity** stammen (Token-Kollision je Raum, `raum.semantik.zustand`), je Kategorie (13646) | je Haus |
| **Diagnose: Vokabular-Lücken des Hypothalamus** | Semantik/Zustand-Paare ohne Eintrag im Level-Lexikon (z. B. `co2.elevated`), je Kategorie (13430) | je Haus |

**Abschaltproben — Verfahren:** Neutralisierung je Modul ist eine benannte
Funktion in `benchmarks/ablationen.py`; jede Zeile nennt die neutralisierte
Methode. Module ohne Pfad zur Vorhersage werden als „kein Messpfad in
Stufe 1“ ausgewiesen (Abnahme 13642, Punkt 6), nicht stillschweigend
weggelassen. Die Hypothalamus-Proben:

- **A** Kontext aus (9-dim konstant).
- **B** Übergangstoken (`house.energy.*`/`house.climate.*`) als echte
  Ereignisse einspeisen — als Schalter im Messstand, damit „an/aus“ gepaart
  gemessen wird, unabhängig davon, was in `main` steht (Abnahme 13642).
- **C** feinere Energie-Wahrnehmung (gleitende EMA je Semantik statt
  Bucket-Stufen) — Messstand-Hülle; die Bucket-Grenzen im Kern bleiben
  unangetastet.

Die **Rangfolge-Kette** (Basalganglien-Priorität, Arousal, Cortisol- und
ACC-Dämpfung, Entorhinal-Antizipation) bekommt eigene Zeilen — der
Rohlisten-Vergleich (§1) hat sie als ersten Prüfstands-Kandidaten gezeigt.

---

## 6. Laufzeit und Maschinen

- Gemessener Takt (Referenzmaschine, alle 26 Module): **196 µs/Ereignis**;
  ein Geräte-Haus mit 1.000–5.000 Ereignissen/Tag kostet damit
  **rund 0,2–1 Sekunde CPU pro Tag**. Größenordnung Haushaltsführung — die
  Energiefront liegt bei der LLM-Weckpolitik, nicht in den Modulen.
- Profil: Locus Coeruleus ~22 %, `predict` 2×/Ereignis ~20 %,
  `anomaly_threshold` ~11 %, Interval-Timing-Scan ~9 % (Anteile unter dem
  Profiler; Rangfolge, nicht Nachkommastelle).
- **Determinismus:** Die Schlaf-Konsolidierung sampelt mit dem globalen
  Zufall — der Messkern seedet sie fest (derselbe Fund wie in
  `benchmarks/replay.py`); ohne das hängt jede Zahl am Vorlauf im Prozess.
- **Öffentliches CI:** Simulation und öffentliche Datensätze (CASAS,
  REFIT/UK-DALE) — Rohdaten lädt der CI-Job selbst und prüft die Prüfsumme,
  nichts liegt im Repo.
- **Eigenes Blech:** nur für die optionale Stundenraster-Probe aus Chris'
  Haus; Artefakt ist allein die Tafel (Aggregate).

---

## 7. Baureihenfolge

0. **dieses Dokument** (MR, vor Schritt 1) — erledigt mit diesem MR.
1. Spur-Format + Wächter + Selbsttests (Token-Beweis, Ausdünnungs-Beweis).
2. Simulator beide Haustypen (inkl. Leck-Tag als Probe-C-Boden).
3. Messkern: predict-then-learn, Gegner, Kennzahlen, Kalibrierung,
   Rohliste, rollierende Ursprünge.
4. CASAS-Konverter (Lizenz/Zitat im Kopf, Rohdaten per CI-Abruf).
5. Geräte-Haus aus öffentlichen Gerätelast-Daten (REFIT, UK-DALE —
   Lizenz zuerst lesen und im Ticket ablegen; Schwellen-Konverter
   (an/aus bzw. Leistungsstufe), Schwellen versioniert).
6. Hygiene-MRs: **explizite Uhr** als kleiner Kern-Parameter (Standard
   `time.time`; Locus Coeruleus + Hypothalamus-Cooldowns lesen sie; im Haus
   ändert sich nichts) mit Test „Replay = gleiche Arousal-Kurve“; danach die
   O(1)-Heilung des 2.000er-Scans mit Vorher-Nachher-Zahl.
7. Ende-zu-Ende über den HA-Runner (simulierter Haushalt durch echtes HA;
   Neustart überlebt den Zustand, Konsolidierung läuft wirklich).
8. Optional: Chris' Haus als Stundenraster-Probe (Energie/Klima A/B/C),
   Export auf unserer Seite, hier nur die Tafel.

---

## 8. Datenquellen und Lizenzen

- **CASAS** (klassisches Haus): Zenodo-Record [17180309](https://zenodo.org/records/17180309)
  „CASAS Smart Home dataset (aruba, cairo, milan, tulum)“, Lizenz
  **CC BY 4.0** (Nutzung, Bearbeitung, Weitergabe erlaubt, mit
  Namensnennung). Pflichtzitat: Cook, D., Crandall, A., Thomas, B., &
  Krishnan, N. (2013). *CASAS: A smart home in a box.* IEEE Computer
  46(7):62–69. doi:10.1109/MC.2012.328. Archiv `new_labeled_data.zip`
  (33,0 MB, MD5 `86954063e1d2099d288f59227b19a749`), Abruf über
  `https://zenodo.org/api/records/17180309/files-archive`; der CI-Job lädt
  und prüft. Aktivitäts-Labels gehören **nicht** in die Spur (Datensparsamkeit),
  höchstens später als Diagnose („in welcher Aktivität liegt die Engine daneben“).
  Sensoren: PIR, Tür, Temperatur; jeder Sensor ist sein eigener Raum
  (`raum = Sensor-ID`), die Semantik kommt aus dem Präfix.
- **REFIT / UK-DALE** (Geräte-Haus): Kandidaten, Lizenzprüfung steht aus
  (Schritt 5). Gleiches Vorgehen: Lizenz + Zitat im Kopf, Rohdaten per
  CI-Abruf, Schwellen-Konverter versioniert.
- **Simulation**: eigener Erzeuger (geseedet); darf frei im Repo liegen.
- **HA-Historie (Chris)**: bleibt draußen; optional nur Stundenraster-Aggregate.

---

## 9. Änderungshistorie

| Datum | Was | Beleg |
|---|---|---|
| 02.10.2026 | Vorschlag (Format, Gegner, Kennzahlen, Aufteilung, Abschaltproben) | #2, Notiz 13188 |
| 02.10.2026 | Nachtrag Uhr: Locus Coeruleus und Hypothalamus-Cooldowns rechnen mit Wanduhr; im Replay sättigt das Arousal | #2, Notiz 13198 |
| 02.10.2026 | Nachtrag 2: Probe C (Bucket-Feinheit, aus Kimis Fund), Gegenprobe im Erfolgskriterium, Vokabular-Lücken-Diagnose | #2, Notiz 13430 |
| 03.10.2026 | **Abnahme** durch Claude: Format ja; mitlernend Hauptzahl; Siegkriterium + gepaarte Differenz; „kein Messpfad“-Zeilen ja; dieses Dokument zuerst; Uhr als Kern-Parameter; Hypothalamus A/B/C | #2, Notiz 13642 |
| 04.10.2026 | Ergänzung: kein 12-Wochen-Export aus Chris' Haus; Geräte-Haus aus öffentlichen Datensätzen; vier Systeme final | #2, Notiz 13664 |
| 04.10.2026 | Zwei Diagnosezeilen (Semantik laut Name vs. vergeben; Entitäten je Token) — als Diagnose, nicht als Siegkriterium | #3, Notiz 13646 |
