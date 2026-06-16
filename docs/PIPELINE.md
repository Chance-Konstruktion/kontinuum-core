# KONTINUUM Core — Pipeline, Snapshot & Persistenz

Wie ein Event durch die Engine läuft, was zurückkommt und wie der Zustand
gespeichert wird. Modul-Details: [`MODULES.md`](MODULES.md).

## Minimal-API

```python
from kontinuum_core import KontinuumEngine

engine = KontinuumEngine()
engine.register_entity("binary_sensor.motion_kitchen", ha_area="kitchen",
                       domain="binary_sensor")

snap = engine.observe({
    "entity_id": "binary_sensor.motion_kitchen",
    "new_state": "on",
    "timestamp": datetime.now(timezone.utc),   # optional
    "old_state": "off",                          # optional
})
print(snap.surprise, snap.anomaly, snap.predictions)

# Outcome-Lernen (vom Host getrieben): hat der Nutzer die letzte Empfehlung
# akzeptiert (True) oder rückgängig gemacht (False)?
engine.feedback(positive=True)

blob = engine.to_dict()      # vollständiges Brain serialisieren
engine.from_dict(blob)        # wiederherstellen
```

## Per-Event-Ablauf von `observe()`

1. **Thalamus** filtert + tokenisiert → `token_id`, `room`, `semantic`, `state`.
   Gefilterte/leere Events → früher `EngineSnapshot` mit `extra["skipped"]`.
2. **Formatio Reticularis** Burst-Gate (sonst `skipped="burst_filtered"`).
3. **Locus Coeruleus** + **Sleep**-Zähler beobachten das Event.
4. **Hypothalamus** absorbiert Energie/Klima; **Spatial/Entorhinal** aktualisieren
   die Raumkarte; **Insula** bestimmt den Modus.
5. **Kontextvektor** (21-dim: Zeit 9 + Hypothalamus 9 + Insula 3) → **Bucket**.
6. **Hippocampus.predict()** (vor dem Lernen) → **Predictive.compute_surprise()**
   + Anomalie-Schwelle. **→ `surprise` wird hier final festgelegt.**
7. **Lerngewicht** = Predictive × **Neurorhythms** (Zirkadian/Burst) ×
   **Suprachiasmaticus** (gelernte Tagesphase) × **Acetylcholin** (erwartete
   Unsicherheit). **Hippocampus.learn(weight)**.
8. **Cortisol** + **Serotonin** beobachten (Surprise/Anomalie).
9. **Basalganglien** (passive Q-Beobachtung), **Cerebellum** (Reflex-Check).
10. Kandidatenliste bauen: Reflex-Injektion + **Interval-Timing**-Injektion
    (überfällige Kadenz) → **Re-Ranking** (`_rank_predictions`): Q-Priorität,
    Accumbens-Bias, Arousal, **Habenula-Suppression**, **Cortisol-Dämpfung**,
    ACC-`cognitive_control`, Entorhinal-Antizipation.
11. **PFC.evaluate()** (Amygdala-Risiko inklusive) → `Decision`.
12. **ACC** misst Konflikt; **STN** kann eine handlungsbereite Entscheidung
    auf OBSERVE **zurückstufen** (Hold).
13. Entscheidung merken (für `feedback()`), Interval-Timing **diese**
    Wiederholung festhalten, ggf. Wartung (`_maybe_maintain`).

> **Reihenfolge-Garantie:** Alles ab Schritt 10 (Ranking/Entscheidung/Injektion)
> verändert **nur** Vorhersagen/Entscheidung, **nicht** `surprise`/`anomaly`
> (in Schritt 6 berechnet). Habenula, Cortisol, STN und Interval-Timing sind
> daher für die Anomalie-Erkennung „nebenwirkungsfrei".

## `EngineSnapshot`

| Feld | Bedeutung |
|---|---|
| `surprise` | Prediction-Error 0–1 |
| `anomaly` | `surprise >= anomaly_threshold` |
| `learning_state` | `cold_start` / `learning` / `stable` |
| `tick_count` | verarbeitete `observe()`-Aufrufe |
| `token_id`, `token` | aktuelles Event-Token |
| `predictions` | gerankte `(token_id, prob, conf, source, n_obs)` |
| `extra` | Reiche Modul-Telemetrie (siehe unten) |

### `snapshot.extra` — Feld-Referenz

| Schlüssel | Quelle / Bedeutung |
|---|---|
| `anomaly_threshold` | adaptive Schwelle (Median+MAD) |
| `arousal` | Locus Coeruleus 0–1 |
| `cognitive_control`, `conflict_level` | ACC |
| `dopamine` | Basalganglien (mittlerer RPE) |
| `expected_next_room` | Entorhinal-Antizipation |
| `raw_prediction_count` | Anzahl Hippocampus-Rohvorhersagen |
| `should_consolidate` | steht eine Schlaf-Konsolidierung an? |
| `cortisol` | Stress-Level 0–1 |
| `serotonin` | Stimmungs-/Geduld-Level 0–1 |
| `acetylcholine` | mittlere erwartete Unsicherheit |
| `scn_gain` | aktueller Suprachiasmaticus-Lernraten-Faktor (~1.0) |
| `stn_brake`, `stn_hold` | Subthalamicus-Bremse / wurde gehalten? |
| `habenula_active` | Anzahl aktiv unterdrückter `(state, action)` |
| `bdnf_protected` | Anzahl geschützter Token |
| `interval_tracked` | Anzahl beobachteter Kadenzen |
| `interval_due_token` | gerade injizierte überfällige Kadenz (falls vorhanden) |
| `reflex` | gefeuerte Cerebellum-Regel (falls vorhanden) |
| `decision` | PFC-Entscheidung (token, stage, confidence, utility, risk, reasons …) |

## `feedback(positive)` — Reward-Loop

Der Host meldet das Ergebnis der letzten Empfehlung. Das schließt die
Lernschleifen: Basalganglien (TD-Q-Update), Nucleus Accumbens, Neurorhythms
(Dopamin-Burst/-Dip), Cerebellum (Regel-Confidence), **Serotonin** (Stimmung),
**Laterale Habenula** (Enttäuschung relieved/punished), **Cortisol** (Stress bei
Ablehnung), **BDNF** (Schutz bewährter Aktionen) und ACC (Fehlerrate).
Gibt `True` zurück, wenn eine Entscheidung zum Verstärken vorlag (eine pro
Entscheidung).

## Persistenz

- `to_dict()` / `from_dict()` round-trippen **alle** zustandsbehafteten Module
  plus Engine-Wiring (`tick_count`, `last_room`, …).
- **`SCHEMA_VERSION`** schützt davor, ein *neueres* Brain in eine *ältere* Engine
  zu laden (dann Cold-Start statt halb verstandenem Zustand). Das aktuelle
  Modul-Set ist **additiv** dazugekommen: ein altes Brain ohne die neuen Keys
  stellt diese Module mit Defaults her, eine alte Engine ignoriert unbekannte
  Keys — `SCHEMA_VERSION` bleibt daher **1**.
- Alle gelernten Maps sind **gedeckelt** (LRU-/Stärke-basiertes Eviction) → die
  Datei wächst nicht unbegrenzt, jahrelanger Pi-Betrieb ist sicher.

## Benchmark / Quality-Gate

```bash
python benchmarks/replay.py
```

Trainiert eine Tagesroutine, injiziert Out-of-Distribution-Events und misst, wie
gut `surprise` Anomalien von Routine trennt (**AUC ≈ 0.99**) + ein
Concept-Drift-Stresstest (erkennt den Wechsel, re-adaptiert). Läuft auch als
CI-Gate.
