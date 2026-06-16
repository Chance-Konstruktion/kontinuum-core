# KONTINUUM Core — Dokumentation

- **[MODULES.md](MODULES.md)** — Referenz aller **26 Module** (Funktion, Signal,
  Kosten, Persistenz), gruppiert nach Wahrnehmung, Gedächtnis/Vorhersage,
  Belohnung/Entscheidung, Botenstoffe/Hormone und Wartung.
- **[PIPELINE.md](PIPELINE.md)** — Per-Event-Ablauf von `observe()`, vollständige
  `EngineSnapshot.extra`-Feldreferenz, der `feedback()`-Reward-Loop, Persistenz
  (`to_dict`/`from_dict`, `SCHEMA_VERSION`) und das Benchmark-Gate.

Die Engine ist HA-frei. Die nutzerseitige Einrichtung der HA-Integration
(Presets, Betriebsmodi, Tracking, **Cortex/LLM** inkl. OpenCLAW, Services) ist in
`ha-kontinuum/docs/SETTINGS.md` dokumentiert.
