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
| 06.10.2026 | Stufe 3: Börse (Claustrum) und Lagebild (Assoziationskortex); Zirkadian-Befund; Zeile „Engine alt“ | #2, §10 |
| 07.10.2026 | Ergebnis Stufe 3: Simulation, fünf CASAS-Häuser (15 Ursprünge), Anwesenheit, Anomalie, Laufzeit auf PC und ARM | #2, §10 |

---

## 10. Stufe 3: Lagebild und Börse

*Stand 07.10.2026. Autor: Claude (Leit-Ticket #2). Gemessen mit dem
Messkern aus §1–§5, unverändert; neu ist nur die Zeile „Engine alt“: die
alte Kandidatenliste läuft im selben Durchgang mit (`extra["predictions_alt"]`),
Vorher und Nachher stehen damit in EINER Messung.*

### 10.1 Befund vor dem Umbau

- **Die alte Kette verlor gegen B2.** Nachgemessen: klassisch −10,2 %,
  Geräte-Haus −4,4 % (Saat 1, 8 Wochen). Zwei Ursachen. Erstens zersplittert
  der Hippocampus sein Gedächtnis in bis zu 96 Kontext-Eimer; im klassischen
  Haus hat er bei rund 70 % der Ereignisse keinen einzigen Kandidaten über
  seiner Schwelle. Zweitens stellt das Ranking eingeschleuste Kandidaten
  (Reflex, überfällige Kadenz) mit fester Konfidenz nach vorn, ohne je zu
  prüfen, ob sie treffen.
- **Der Thalamus sieht nicht, was zusammengehört.** `unavailable` wirft er
  weg — genau das melden Reifendrucksensoren, die mit dem Auto wegfahren.
  Leistung kennt er nur in festen Eimern (unter 100 W „niedrig“): 3 W Standby
  und 100 W Betrieb eines PCs sind dasselbe.
- **Die zirkadiane Lernrate hing an der Uhr des Messlaufs.** Eine feste
  Kosinuskurve (Tief um 20:00, nicht nachts), im Engine-Pfad mit der
  Wanduhr-Stunde gefüttert: Derselbe Strom lernte je nach Uhrzeit des Laufs
  anders. Im Anomalie-Benchmark mit Ereignisstunde (`benchmarks/replay.py`,
  wie live, ohne Zeitgeber) gemessen: AUC 0,948, P 0,36, R 0,25.

### 10.2 Was gebaut wurde

- **Claustrum** (`claustrum.py`), die Börse: Sequenz (Markov 1..3,
  Witten-Bell, ohne Eimer), Uhrzeit-Gewohnheit, Folge je Tagesabschnitt,
  Lage-Experte und die externen Stimmen (Hippocampus, sicherer Reflex)
  werden log-linear gemischt. Die Gewichte lernt sie im Betrieb aus dem, was
  eintrat (Gradient der Log-Likelihood, RMSprop, global plus je Semantik des
  letzten Ereignisses). Die Intervall-Uhr stimmt nicht mit; eine überfällige
  Kadenz und ein Reflex, den die Börse nicht ohnehin führt, bekommen
  höchstens den letzten Platz.
- **Assoziationskortex** (`association_cortex.py`), das Lagebild: jede
  Entität in genau einem Zustand, `unavailable` als `weg`, Leistungen in
  gelernten Gerätestufen; alle 5 Minuten Ereigniszeit eine Paar-Tafel „alle
  gegen alle“; Anwesenheit je `person`/`device_tracker` allein aus den
  Geräten (Naive Bayes über Zustand, Dauer und Tagesabschnitt, gestapelt mit
  der Uhrzeit-Gewohnheit, Gewichte je Entität mit AdaGrad, Zählung einen Tag
  verzögert); dazu der Lage-Experte der Börse.
- **Engine:** Lagebild VOR dem Thalamus, Börse statt Ranking für
  `snapshot.predictions`. Vorhersage und Vorschlag sind getrennt: Was der PFC
  vorschlägt, rankt weiter mit den Rückmeldungen (Habenula, Accumbens,
  Basalganglien), jetzt auf den Kandidaten der Börse. `surprise`/`anomaly`
  bleiben auf dem Hippocampus-Pfad.
- **Neurorhythms:** feste Kurve aus (`CIRCADIAN_FEST = False`, Faktor 1,0);
  die Engine reicht die Ereignisstunde durch.

### 10.3 Nächstes Ereignis (Top-1, Engine-Ebene)

**Simulation** (4 Saaten × 4 Ursprünge, 8 Wochen; n = 3.279 / 3.839):

| Haus | Engine | Engine alt | Engine roh | B2 | B1 | B0 |
|---|---|---|---|---|---|---|
| klassisch | **82,5** | 53,6 | 69,3 | 56,2 | 23,5 | 0,0 |
| Geräte | **93,5** | 83,6 | 93,2 | 60,8 | 31,2 | 8,8 |

Abschaltproben in denselben Läufen: ohne Lage-Experte 81,9 / 92,2 (der
Experte bringt +0,6 / +1,3), ohne externe Stimme 82,4 / 93,8 (keine
messbare Wirkung — der Hippocampus sagt nichts, was die Sequenz nicht
schon weiß; sein Gewicht lernt die Börse leicht negativ).

**CASAS** (je Ursprung k Wochen Training, Woche k+1 Test; alle Systeme im
selben Strom):

| Haus | k | Testereignisse | Engine | Engine alt | Engine roh | B2 | B1 | Δ B2 | Δ alt | Top-3 Engine / B2 |
|---|---|---|---|---|---|---|---|---|---|---|
| aruba | 4 | 53.093 | **51,2** | 41,7 | 42,0 | 39,2 | 5,6 | **+12,1** | +9,6 | 82,1 / 61,3 |
| aruba | 8 | 61.583 | **51,8** | 42,0 | 42,6 | 40,2 | 7,5 | **+11,6** | +9,8 | 82,2 / 61,2 |
| aruba | 12 | 47.088 | **55,9** | 44,6 | 45,0 | 42,4 | 6,5 | **+13,4** | +11,3 | 86,2 / 65,0 |
| cairo | 4 | 72.401 | **48,6** | 37,2 | 37,7 | 34,4 | 3,9 | **+14,3** | +11,5 | 82,0 / 60,0 |
| cairo | 5 | 73.587 | **46,0** | 35,3 | 35,9 | 32,9 | 5,7 | **+13,2** | +10,8 | 79,2 / 57,1 |
| cairo | 6 | 71.050 | **49,2** | 37,9 | 38,5 | 35,4 | 5,6 | **+13,8** | +11,3 | 81,9 / 60,5 |
| milan | 4 | 12.440 | **54,1** | 47,4 | 47,6 | 44,9 | 4,1 | **+9,2** | +6,7 | 82,2 / 64,9 |
| milan | 6 | 42.209 | **52,0** | 43,5 | 43,9 | 40,9 | 3,8 | **+11,2** | +8,6 | 81,6 / 62,3 |
| milan | 8 | 47.350 | **53,9** | 45,1 | 45,8 | 42,5 | 6,2 | **+11,5** | +8,8 | 82,5 / 63,5 |
| tulum1 | 4 | 17.248 | **54,9** | 51,2 | 51,8 | 45,8 | 11,7 | **+9,1** | +3,8 | 89,4 / 71,3 |
| tulum1 | 6 | 20.415 | **43,1** | 37,7 | 38,4 | 33,6 | 6,2 | **+9,5** | +5,4 | 76,1 / 53,9 |
| tulum1 | 8 | 21.459 | **44,3** | 36,7 | 37,1 | 34,1 | 7,2 | **+10,2** | +7,6 | 79,1 / 55,4 |
| tulum2 | 4 | 57.422 | **47,0** | 38,5 | 38,9 | 35,6 | 4,6 | **+11,4** | +8,5 | 80,1 / 58,5 |
| tulum2 | 8 | 49.270 | **44,0** | 35,7 | 35,8 | 32,7 | 4,3 | **+11,3** | +8,4 | 77,0 / 55,6 |
| tulum2 | 12 | 48.222 | **48,1** | 38,8 | 39,4 | 35,6 | 5,1 | **+12,5** | +9,4 | 81,0 / 58,3 |

Über alle 15 Ursprünge: **+9,1 bis +14,3 Punkte über B2**, +3,8 bis +11,5 über die alte Kandidatenliste; Top-3 76,1–89,4 % (alte Liste 55,8–78,2 %, B2 53,9–71,3 %).

**Siegkriterium (§4) erfüllt:** Die Engine liegt in jedem Haus und in jedem
Ursprung strikt vor B0, B1 und B2. `Engine alt` ist die alte
Kandidatenliste VOR dem Ranking (Hippocampus, Reflex, Intervall) — die
stärkere Lesart der alten Kette, denn das Ranking machte sie schlechter
(§5, Rohliste). Auch sie und der Hippocampus allein (`Engine roh`) liegen
in jedem Ursprung zurück.

**Gegenprobe:**

- **Kalibrierung** (ECE der Top-1-Konfidenz): Engine 3,8–9,0 % auf der
  Simulation, 4,1 % auf Milan (k=4); die alte Kette 13–50 %. In der Mitte
  ist die Börse leicht übersicher (Eimer 0,7–0,8: Quote 0,62).
- **Anomalie-Trennung:** siehe 10.5 — besser, nicht schlechter.

### 10.4 Anwesenheit (Lagebild)

Protokoll (`benchmarks/spur/anwesenheit.py`): Fünf-Minuten-Takte; im
Training sind die Tracker die Etiketten, in der Testwoche werden sie
versteckt (`unknown`). Gegner: P0 (häufigster Zustand), P1
(Uhrzeit-Gewohnheit je Wochenende/Stunde), P2 (Aktivität in der letzten
Stunde). Kennzahl: ausgewogene Trefferquote (Mittel aus „daheim richtig“
und „weg richtig“), denn wer immer „daheim“ sagt, hat sonst schon 80 %.

| Haus | Ursprung | Lagebild | bester Gegner | Brier Lagebild |
|---|---|---|---|---|
| Simulation A (Saat 1/2/3) | k=4 | 94,1 / 85,5 / 91,7 | 61,7 / 50,0 / 77,4 | 0,041 / 0,108 / 0,090 |
| Simulation B (Saat 1/2/3) | k=4 | 95,4 / 95,2 / 88,3 | 91,2 / 91,7 / 91,6 | 0,031 / 0,027 / 0,063 |
| CASAS Aruba | k=4 / 8 / 12 | 88,1 / 87,8 / 83,5 | 71,4 / 60,7 / 63,7 | 0,039 / 0,031 / 0,053 |
| CASAS Tulum1 | k=4 / 6 / 8 | 88,8 / **50,1** / 88,8 | 91,0 / 87,6 / 88,2 | 0,090 / 0,103 / 0,103 |

Simulierter Haushalt: zwei Personen, ein Auto mit vier
Reifendrucksensoren, Fernseher, PC (Router-Erreichbarkeit und Leistung),
Handys als Tracker mit Verspätung und WLAN-Flattern. Person A arbeitet oft
zu Hause am PC — ihre Anwesenheit verrät die Lage, nicht die Uhr. Person B
hat feste Zeiten; dort ist die Uhrzeit schon stark. CASAS hat keine
Tracker: Das Etikett stammt aus den Aktivitäten `Leave_Home`/`Enter_Home`.

**Bekannte Schwäche (Tulum1, k=6):** Eine ganze Testwoche lang erkennt das
Lagebild keine Abwesenheit (p(daheim) während der Abwesenheit im Median
0,70, daheim 0,98 — die Reihenfolge stimmt, die Schwelle nicht); die
Uhrzeit allein hatte 87,6 %. Tulum ist eine Wohnung für zwei, das Etikett
gehört einer Person, und die Geräte zeigen oft die andere. Die Lage lernt
dann eine Achse Richtung „daheim“ (+2,1) und traut der Uhr wenig (0,25).
Eine Mischung mit der Uhrzeit nach jüngster Bilanz half nicht — im Training
war das Lage-Modell besser. Offen für Stufe 4: Etiketten je Person bei
mehreren Bewohnern, oder eine Kalibrierung, die ohne Etikett weiterlernt.

Die Tracker sind nie Merkmal: Ihr Zustand ist das Etikett. Als Indiz spräche
das Lagebild dem Handy nur nach, statt ihm zu widersprechen, wenn es im Büro
liegt (Test `test_die_lage_spricht_dem_handy_nicht_nach`).

### 10.5 Anomalie

| Messung | vorher | nachher |
|---|---|---|
| Anomalie-Benchmark mit Ereignisstunde (`replay.py`-Routine, ohne Zeitgeber) | AUC 0,948, P 0,36, R 0,25 | AUC 0,999, P 1,00, R 1,00 |
| dasselbe, Routine mit ±10 min Streuung | — | AUC 0,978, P 0,91, R 1,00 |
| Überraschung der Börse (Bits) als Anomalie-Signal | — | AUC 0,73, P 0,52, R 0,51 |
| kontinuum-ai-anomaly, Messstand (5 Saaten) | Tafel | bitgleich |

Die Börse ist der bessere Vorhersager, aber kein besserer Anomalie-Melder:
Ihr Log-Verlust bleibt Telemetrie (`extra["claustrum"]["ueberraschung_bits"]`),
`surprise`/`anomaly` kommen weiter aus dem Hippocampus-Pfad.

### 10.6 Laufzeit (Raspberry Pi als Mindestmaß)

Je Ereignis, ganze Engine, Cairo (dichtes Haus), erste 6.000 Ereignisse:

| Maschine | alte Engine | neu (Börse + Lagebild) | neu ohne Lage-Experte |
|---|---|---|---|
| PC (Ryzen, Python 3.14) | 0,69 ms | 1,40 ms | 0,92 ms |
| Huawei P smart 2019 (Kirin 710, ARM64, Termux, Python 3.14) | 11,5 ms | 21,6 ms | 14,9 ms |

Die Börse kostet rund ein Drittel mehr, der Lage-Experte noch einmal so
viel. Ein Raspberry Pi 4 (Cortex-A72, 1,5–1,8 GHz) sollte in derselben
Größenordnung wie das Telefon liegen — gemessen ist das nicht. Ein Haus mit
5.000 Ereignissen am Tag braucht damit auf ARM rund 110 Sekunden CPU am Tag,
ein dichtes wie Cairo (~10.000) rund 220 — unter 0,3 % eines Kerns.

Gespeichertes Gehirn nach 100.000 Cairo-Ereignissen: 2,1 MB JSON (gzip
0,52 MB) statt 0,7 MB (0,25 MB); davon Börse 1,0 MB, Lagebild 0,3 MB.

Zwei bitgleiche Beschleunigungen gehören dazu (Profil auf Cairo): Die
Hippocampus-Eviction sortierte bei fast jedem Ereignis alle 1.000 N-Gramme
eines Eimers, um ein bis drei zu streichen (20 % der Engine-Zeit, jetzt
`nsmallest`); die Lage-Schleife (Merkmale × Kandidaten) rechnet `BETA·P(y)`
einmal je Kandidat statt je Paar. Zusammen −15 %.

Speicher ist gedeckelt: Sequenz und Folge je 20.000 Kontexte (dann
verdrängt), Lagebild 96 Entitäten im Blick, Paar-Tafel 192 × 192 Merkmale,
je Merkmal höchstens 128 Folgen.

### 10.7 Bitgleichheit

- Nach `to_dict()` → JSON → `from_dict()` setzt die Engine Ereignis für
  Ereignis bitgleich fort (Test `test_engine_setzt_nach_dem_laden_bitgleich_fort`);
  Börse und Lagebild speichern dafür ihren offenen Vorhersage-Stand mit.
- Die Beschleunigungen (Hippocampus-Eviction, Lage-Schleife) und der Umbau
  auf `lage_setzen()`/`boersen_liste()` sind bitgleich: 20.000 Ereignisse
  Cairo, gleicher Hash der Vorhersagen und des Gesamtzustands vorher und
  nachher.
