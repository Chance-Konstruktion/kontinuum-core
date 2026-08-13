# KONTINUUM Core — Modul-Referenz

Die Engine vereint **26 neuro-inspirierte Module** in *einer* `observe()`-Pipeline.
Alle sind reine Statistik/Arithmetik (kein ML), laufen in **~0 ms pro Event**,
sind in der Größe begrenzt (laufen jahrelang auf einem Raspberry Pi) und werden
über `to_dict()`/`from_dict()` persistiert.

> **Lesehilfe:** *Signal* = was das Modul nach außen liefert · *Kosten* = Aufwand
> pro Event · *Persistenz* = wird im Brain gespeichert (round-trip).
> Die genaue Pipeline-Reihenfolge und alle Snapshot-Felder stehen in
> [`PIPELINE.md`](PIPELINE.md).

Inhalt:
1. [Wahrnehmung & Eingang](#1-wahrnehmung--eingang)
2. [Gedächtnis & Vorhersage](#2-gedächtnis--vorhersage)
3. [Belohnung & Entscheidung](#3-belohnung--entscheidung)
4. [Rhythmen, Botenstoffe & Hormone](#4-rhythmen-botenstoffe--hormone)
5. [Wartung & Meta-Lernen](#5-wartung--meta-lernen)
6. [Cortex (optionale LLM-Schicht)](#6-cortex--optionale-llm-schicht)

---

## 1. Wahrnehmung & Eingang

| Modul (Datei) | Hirnregion | Funktion | Signal | Persistenz |
|---|---|---|---|---|
| **Thalamus** (`thalamus.py`) | Sensorisches Tor | Filtert Events, mappt Entity → `room.semantic.state`-**Token**, kennt Sonnenstand, baut den 9-dim Zeitkontext (Stunde, Wochentag, Monat/Saison, Wochenende, Sonne) | `token_id`, Zeitvektor | ✅ |
| **Formatio Reticularis** (`reticular.py`) | Aufmerksamkeits-/Burst-Filter | Unterdrückt Ereignis-Stürme desselben Entities per Cooldown (z. B. flatternder Kontakt). Arousal-moduliert vom Locus Coeruleus | `should_process()` | ✅ |
| **Locus Coeruleus** (`locus_coeruleus.py`) | Noradrenalin / Arousal | Misst Ereignisdichte (EMA über 60 s) → Aufmerksamkeitslevel; macht das Ranking bei Betrieb reaktiver | `arousal` 0–1 | ✅ |
| **Hypothalamus** (`hypothalamus.py`) | Homöostase | Absorbiert ~95 % der Energie-/Klima-Rauschevents, liefert einen 9-dim Kontext (Batterie, Solar, Verbrauch, Temperatur-Trends …) | Kontextvektor (9) | ✅ |
| **Insula** (`insula.py`) | Interozeption | Erkennt **Modus**: `sleeping, waking_up, active, relaxing, cooking, away` (nutzt zirkadiane Priors) | `current_mode`, Modus-Kontext (3) | ✅ |
| **Spatial Cortex** (`spatial_cortex.py`) | Raumwahrnehmung | Lernt Raum-Sequenzen (A → B), entprellt Präsenz (Hysterese/Confirm/Cooldown), liefert „entered <room>"-Tokens | Raumübergänge | ✅ |
| **Entorhinaler Cortex** (`entorhinal_cortex.py`) | Grid-/Übergangskarte | Lernt die Transitions-Map zwischen Räumen, sagt den nächsten Raum vorher, pruned selten genutzte Pfade (1×/Tag) | `predict_next_room()` | ✅ |

## 2. Gedächtnis & Vorhersage

| Modul (Datei) | Hirnregion | Funktion | Signal | Persistenz |
|---|---|---|---|---|
| **Hippocampus** (`hippocampus.py`) | Episodisches Gedächtnis | Lernt Sequenzen als **1- bis 4-Gramm-Markov-Ketten** pro Kontext-Bucket; sagt Top-k nächste Tokens vorher | `predict()`-Liste | ✅ |
| **Predictive Processing** (`predictive_processing.py`) | Prediction-Error | Berechnet **Surprise** (0–1) aus Vorhersage-Fehler + Neuheit; adaptive **Anomalie-Schwelle** (Median + MAD, robust); leitet das **Lerngewicht** ab | `surprise`, `anomaly_threshold`, `learn_weight` | ✅ |
| **Cerebellum** (`cerebellum.py`) | Reflexe / Prozeduren | Extrahiert stabile Routinen als deterministische Regeln (kontext-bucket-bewusst), bildet **Chunks** (Mehrschritt-Prozeduren) | gefeuerte Regel | ✅ |
| **Interval Timing** (`interval_timing.py`) | Striatal-cerebelläre „Stoppuhr" | Lernt **Dauern zwischen Wiederholungen** (EMA des Intervalls + Streuung) und markiert eine regelmäßige Kadenz als **„fällig"** (z. B. „alle 4 Wochen"). Überfällige Kadenzen werden als Vorhersage injiziert | `due_score`, `due_prediction()` | ✅ |

> **Wichtig:** *Interval Timing* ≠ *Suprachiasmatischer Nukleus*. Der SCN ist die
> **Tagesuhr** (Uhrzeit); Interval Timing ist das **Dauer-/Kadenzgefühl**.

## 3. Belohnung & Entscheidung

| Modul (Datei) | Hirnregion | Funktion | Signal | Persistenz |
|---|---|---|---|---|
| **Basalganglien** (`basal_ganglia.py`) | Striatum / Dopamin | Handlungsauswahl per **Go/NoGo**, **Q-Values** (TD-Lernen), Gewohnheitsbildung; Dopamin = Reward-Prediction-Error | `dopamine_signal`, Aktionspriorität | ✅ |
| **Nucleus Accumbens** (`nucleus_accumbens.py`) | Belohnungs-Verstärker | Schneller Habit-Bias pro `(state, action)` aus Nutzer-Feedback | `get_bias()` | ✅ |
| **Laterale Habenula** (`habenula.py`) | Anti-Reward | Merkt **systematische Enttäuschung** pro `(state, action)` und **unterdrückt** chronisch abgelehnte Vorschläge im Ranking (Schluss mit „Nerven") | `get_suppression()` | ✅ |
| **Amygdala** (`amygdala.py`) | Risiko-/Sicherheits-Gate | Bewertet vorgeschlagene Aktionen auf Gefahr, kann ein schnelles **Veto** auslösen, bevor der PFC fertig ist | `risk`, Veto | ✅ |
| **Präfrontaler Kortex** (`prefrontal_cortex.py`) | Entscheidung | Wägt Nutzen vs. Risiko (+ Q-Boost) ab, wählt die beste Aktion und die **Stufe** (OBSERVE/SUGGEST/CONFIRM/EXECUTE) je Betriebsmodus | `Decision` | ✅ |
| **Anteriorer Cingulärer Cortex** (`anterior_cingulate.py`) | Konfliktmonitor | Misst Uneinigkeit der Modul-Stimmen + Fehlerrate → **`cognitive_control`** dämpft die Confidence der nächsten Runde | `conflict_level`, `cognitive_control` | ✅ |
| **Nucleus Subthalamicus** (`subthalamic_nucleus.py`) | „Hold your horses"-Bremse | Bei hohem Konflikt + knapper Marge empfiehlt er **Warten** statt Handeln (Serotonin/Geduld justiert die Schwelle); stuft eine handlungsbereite Entscheidung auf OBSERVE zurück | `brake`, `should_hold()` | ✅ (Zähler) |

## 4. Rhythmen, Botenstoffe & Hormone

| Modul (Datei) | Botenstoff/Region | Funktion | Signal | Persistenz |
|---|---|---|---|---|
| **Neurorhythms** (`neurorhythms.py`) | Zirkadian + phasisches Dopamin | Moduliert die Lernrate über den Tag (fester Cosinus), **Dopamin-Bursts** bei Überraschung+Erfolg, **Synaptic Homeostasis** | Lernraten-Multiplikator | ✅ |
| **Nucleus Suprachiasmaticus** (`suprachiasmatic.py`) | Innere Tagesuhr | **Lernt** das echte Aktivitätsprofil des Haushalts und korrigiert die Lernrate ±15 % (entraint; Nachtschicht-tauglich) | `phase_gain(hour)` | ✅ |
| **Serotonin** (`serotonin.py`) | Stimmung / Geduld | Langsame Baseline aus Erfolg/Frust → **Geduld**, steuert die Wartebereitschaft des STN | `get_patience()` | ✅ |
| **Acetylcholin** (`acetylcholine.py`) | Erwartete Unsicherheit | Schätzt pro Kontext-Bucket den „normalen" Rauschpegel und **dämpft das Lernen** dort, wo Jitter die Norm ist (Komplement zum NA-Surprise) | `learn_gain(bucket)` | ✅ |
| **Cortisol** (`cortisol.py`) | Stress-Hormon (langsam) | Integriert anhaltende Surprise/Anomalie/Overrides → macht das Ranking in chaotischen Phasen **bis −30 % vorsichtiger** | `level`, `damping()` | ✅ |

## 5. Wartung & Meta-Lernen

| Modul (Datei) | Region | Funktion | Persistenz |
|---|---|---|---|
| **Sleep Consolidation** (`sleep_consolidation.py`) | Schlaf / Replay | In ruhigen Phasen (≥30 min still, ≥50 Events, max 1×/h): schwache Muster vergessen, starke verstärken, **Dream-Replay** (Cross-Context-Generalisierung), Q-Smoothing, synaptische Homöostase | ✅ |
| **Metaplastizität** (`metaplasticity.py`) | Meta-Lernen | Beobachtet, *wie gut* andere Module lernen, und passt deren Parameter (Lernraten etc.) periodisch (24 h) an | ✅ (eigene Datei) |
| **BDNF** (`bdnf.py`) | Neurotropher Schutz („Vitamin"-Schicht) | Schützt die Token bewährter Reflexe vor pauschalem Vergessen in der Konsolidierung (Mindestgewicht statt Löschen); trophischer Support klingt bei Nichtnutzung ab | ✅ |

## 6. Cortex — optionale LLM-Schicht

> **KONTINUUM funktioniert vollständig ohne LLM.** Der Cortex ist ein optionales
> Upgrade und lebt in der HA-Integration (`ha-kontinuum`), nicht im Core.

Der Core stellt nur den **Daten-Vertrag** bereit (`kontinuum_core.llm`):

- `build_llm_context()` / `render_llm_context()` — exportiert den Brain-Zustand
  (Anomalie-Signal, erwartete nächste Events, Lern-Reife) mit **expliziten
  0–1-Skalen**, damit ein Modell zuverlässig darüber argumentieren kann.
- `extract_json()` / `normalize_proposal()` — macht aus einer (oft unsauberen:
  code-fenced, prosa-umhüllten) Modell-Antwort einen strikten, validierten
  Aktions-Vorschlag.
- `kontinuum_core.priors` — `parse_home_prior()` + `seed_engine_from_prior()`
  lassen ein LLM das Zuhause beim Setup beschreiben (Tag-1-Vorsprung).

Die konkrete Cortex-Einrichtung (Agents, Provider inkl. **Custom/OpenAI-kompatibel
/ OpenCLAW**) ist in [`ha-kontinuum/docs/SETTINGS.md`](https://github.com/Chance-Konstruktion/ha-kontinuum/blob/main/docs/SETTINGS.md) beschrieben.

---

## Architektur-Überblick

```
                         ACC (Konfliktmonitor) ── cognitive_control
                          │
Thalamus → Hippocampus → Cerebellum → PFC ─────────────→ Entscheidung
   │           │            │           │
Reticular   Predictive   IntervalTiming Amygdala (Veto)
   │        (Surprise)        │           │
Locus C.   Neurorhythms   Basalganglien  Habenula (Anti-Reward)
   │        SCN/ACh           │           │
Hypothalamus  Serotonin   Nuc. Accumbens  STN (Hold)
   │           Cortisol
Insula / Spatial / Entorhinal

[Sleep Consolidation]  → nachts/ruhig: Replay, Prune, Dream, Homöostase
[Metaplastizität]      → alle 24 h: Lernraten aller Module
[BDNF]                 → schützt bewährte Routinen vor dem Vergessen
[Cortex / LLM]         → optional, oben drauf
```
