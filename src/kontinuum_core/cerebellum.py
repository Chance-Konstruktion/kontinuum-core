"""
╔══════════════════════════════════════════════════════════════════╗
║  KONTINUUM – Cerebellum                                         ║
║  Routinen-Erkennung: Feste Reflexe aus dem Gedächtnis          ║
║                                                                  ║
║  Biologisches Vorbild:                                           ║
║  Das Cerebellum speichert eingeübte Bewegungsabläufe als        ║
║  automatische Reflexe. Hier extrahiert es hochfrequente         ║
║  Event-Paare als deterministische Regeln.                       ║
║                                                                  ║
║  v0.8.0 – Schwellen per Preset konfigurierbar                  ║
║  v0.13.0 – N-Gram Regeln: 2-gram und 3-gram Sequenzen         ║
║  → Kontextreichere Muster (z.B. motion+door → light)           ║
╚══════════════════════════════════════════════════════════════════╝
"""

import logging
import time
from collections import deque

_LOGGER = logging.getLogger(__name__)


class CerebellumRule:
    """Eine gelernte Routine."""
    def __init__(self, trigger: int, target: int, confidence: float,
                 avg_delay: float = 0, ngram_order: int = 1,
                 trigger_sequence: tuple = None,
                 context_buckets: set = None):
        self.trigger = trigger  # Letztes Token der Sequenz (für check())
        self.target = target
        self.confidence = confidence
        self.avg_delay = avg_delay
        self.ngram_order = ngram_order  # 1, 2, 3 oder 4
        self.trigger_sequence = trigger_sequence or (trigger,)  # Volle Sequenz
        # v0.22.0: Kontext-Buckets in denen diese Regel beobachtet wurde.
        # None = kontextfrei (Legacy / globales Muster).
        self.context_buckets = set(context_buckets) if context_buckets else set()
        self.successes = 0
        self.failures = 0
        self.last_fired = 0


class CerebellumChunk:
    """Eine erkannte Sequenz aus mehreren Regeln (Hierarchisches Chunking).

    v0.22.0: Wenn das Cerebellum wiederkehrend A→B→C beobachtet
    (Regelketten in denen `target` der ersten Regel = `trigger` der nächsten),
    wird das als ein höherer "Chunk" gespeichert. Dies entspricht der
    Bildung einer Prozedur ("Filmabend starten") aus Einzelschritten.
    """
    def __init__(self, rule_keys: list, tokens: list, confidence: float):
        self.rule_keys = list(rule_keys)
        self.tokens = list(tokens)  # Token-IDs in Reihenfolge
        self.confidence = confidence
        self.length = len(rule_keys)

    def to_dict(self) -> dict:
        return {
            "rule_keys": list(self.rule_keys),
            "tokens": list(self.tokens),
            "confidence": round(self.confidence, 3),
            "length": self.length,
        }


class Cerebellum:
    """Routinen-Erkennung von KONTINUUM."""

    # Defaults für "ausgeglichen" – werden vom Config Flow überschrieben
    MIN_OBSERVATIONS = 5
    MIN_CONFIDENCE = 0.75
    MAX_RULES = 50
    RULE_COOLDOWN = 300
    MIN_DELAY = 2.0  # Sekunden – darunter = Hardware-Kopplung, kein Verhalten

    # v0.13.0: Confidence-Schwellen pro N-Gram Ordnung
    # Höhere N-Grams sind kontextreicher → niedrigere Schwelle nötig
    NGRAM_CONF_THRESHOLDS = {1: None, 2: 0.60, 3: 0.45, 4: 0.35}  # None = self.MIN_CONFIDENCE

    def __init__(self):
        self.rules = {}  # "trigger_target" → CerebellumRule
        self._recent_buffer = deque(maxlen=15)  # Letzte Token für Sequenz-Check
        self._total_fired = 0
        # v0.22.0: Hierarchisches Chunking – erkannte Mehrschritt-Prozeduren
        self.chunks = []  # list[CerebellumChunk]
        # Aktueller Bucket – wird von außen gesetzt, damit check() Kontext
        # bevorzugen kann (siehe __init__.py: cerebellum.set_context(bucket))
        self._current_bucket = None

    def compile_rules(self, hippocampus):
        """
        Extrahiert Routinen aus dem Hippocampus-Gedächtnis.

        v0.13.0: Unterstützt 1-gram, 2-gram und 3-gram Regeln.
        - 1-gram: A → B (wie bisher, conf >= 75%)
        - 2-gram: (A, B) → C (kontextreicher, conf >= 60%)
        - 3-gram: (A, B, C) → D (sehr spezifisch, conf >= 45%)
        """
        candidates = {}
        # v0.22.0: Sammle für jede Regel die Kontext-Buckets, in denen sie
        # signifikant beobachtet wurde. Das ermöglicht "(token, context) → token"
        # statt rein sequenzielles Lernen.
        rule_buckets = {}

        for bucket in hippocampus.transitions:
            for ngram, targets in hippocampus.transitions[bucket].items():
                n = len(ngram)
                if n not in (1, 2, 3, 4):
                    continue

                # Letztes Token der Sequenz als Trigger
                trigger = ngram[-1]
                total = hippocampus.totals[bucket][ngram]

                if total < self.MIN_OBSERVATIONS:
                    continue

                # Confidence-Schwelle abhängig von N-Gram Ordnung
                min_conf = self.NGRAM_CONF_THRESHOLDS.get(n, self.MIN_CONFIDENCE)
                if min_conf is None:
                    min_conf = self.MIN_CONFIDENCE

                for target, count in targets.items():
                    # Self-Loop filtern
                    if trigger == target:
                        continue

                    conf = count / total
                    if conf < min_conf:
                        continue

                    # Key: für N-gram Regeln die ganze Sequenz kodieren
                    if n == 1:
                        key = f"{trigger}_{target}"
                    else:
                        seq_str = ",".join(str(t) for t in ngram)
                        key = f"seq:{seq_str}_{target}"

                    # N-gram Bonus: Höhere N-grams = kontextreicher = besser
                    ngram_bonus = 1.0 + (n - 1) * 0.15  # 2-gram: 1.15, 3-gram: 1.3
                    effective_score = conf * count * ngram_bonus

                    # Bucket-Set tracken (auch für Kandidaten die nicht
                    # gewinnen – damit haben wir Kontext für die finale Regel)
                    rule_buckets.setdefault(key, set()).add(bucket)

                    if key not in candidates or effective_score > candidates[key][0] * candidates[key][1] * candidates[key][6]:
                        # Delay berechnen (immer vom letzten Token zum Target)
                        dur_key = f"{trigger}_{target}"
                        durs = hippocampus.durations.get(dur_key, [])
                        avg_delay = sorted(durs)[len(durs) // 2] if len(durs) >= 3 else 60

                        # Unter MIN_DELAY = Hardware-Kopplung
                        if avg_delay < self.MIN_DELAY:
                            continue

                        candidates[key] = (conf, count, trigger, target, avg_delay, n, ngram_bonus, ngram)

        sorted_rules = sorted(candidates.items(),
                               key=lambda x: x[1][0] * x[1][1] * x[1][6], reverse=True)
        sorted_rules = sorted_rules[:self.MAX_RULES]

        new_rules = {}
        for key, (conf, count, trigger, target, avg_delay, n, _, ngram) in sorted_rules:
            rule = CerebellumRule(
                trigger, target, conf, avg_delay,
                ngram_order=n, trigger_sequence=ngram,
                context_buckets=rule_buckets.get(key),
            )
            if key in self.rules:
                rule.successes = self.rules[key].successes
                rule.failures = self.rules[key].failures
                # Bestehende Bucket-Beobachtungen erhalten (vereinigen)
                if self.rules[key].context_buckets:
                    rule.context_buckets |= self.rules[key].context_buckets
            new_rules[key] = rule

        self.rules = new_rules
        # v0.22.0: Chunks (Mehrschritt-Prozeduren) aus den Regeln ableiten
        self.chunks = self._detect_chunks()
        n_by_order = {1: 0, 2: 0, 3: 0, 4: 0}
        for r in self.rules.values():
            n_by_order[r.ngram_order] = n_by_order.get(r.ngram_order, 0) + 1
        _LOGGER.info(
            "Cerebellum: %d Routinen kompiliert (1=%d, 2=%d, 3=%d, 4=%d), %d Chunks",
            len(self.rules), n_by_order[1], n_by_order[2], n_by_order[3], n_by_order[4],
            len(self.chunks),
        )

    def set_context(self, bucket):
        """Setzt den aktuellen Kontext-Bucket für check().

        v0.22.0: Wird vor jedem check()-Aufruf vom Setup gesetzt.
        Erlaubt der Regel-Suche, Kontext-passende Routinen zu bevorzugen.
        """
        self._current_bucket = bucket

    # ── v0.22.0: Hierarchisches Chunking ──────────────────────
    MAX_CHUNK_LEN = 5
    MIN_CHUNK_CONFIDENCE = 0.45

    def _detect_chunks(self) -> list:
        """Erkennt zusammengehörende Mehrschritt-Sequenzen aus den Regeln.

        Strategie: Wenn rule_a.target == rule_b.trigger und beide Regeln
        ausreichend Konfidenz haben, bilden sie einen Chunk.
        Längere Ketten entstehen rekursiv (greedy, max MAX_CHUNK_LEN).

        Das ist ein erster Schritt von "Vorhersage" zu "Absichtserkennung":
        Wiederkehrende Sequenzen werden als Einheit erkannt und können
        später als Trigger für Szenen oder komplexe Automatisierungen dienen.
        """
        # Index: trigger → list of (key, rule)
        by_trigger = {}
        for key, rule in self.rules.items():
            if rule.confidence < self.MIN_CHUNK_CONFIDENCE:
                continue
            by_trigger.setdefault(rule.trigger, []).append((key, rule))

        chunks = []
        # Starte von Regeln mit ausreichender Confidence; folge target-Ketten
        for start_key, start_rule in self.rules.items():
            if start_rule.confidence < self.MIN_CHUNK_CONFIDENCE:
                continue
            chain_keys = [start_key]
            chain_tokens = [start_rule.trigger, start_rule.target]
            chain_conf = start_rule.confidence
            current_target = start_rule.target

            while len(chain_keys) < self.MAX_CHUNK_LEN:
                next_candidates = by_trigger.get(current_target, [])
                # Beste Folgeregel (höchste Konfidenz, vermeidet Zyklen)
                best = None
                for k, r in next_candidates:
                    if k in chain_keys:
                        continue
                    if r.target in chain_tokens:
                        continue  # Zyklus vermeiden
                    if best is None or r.confidence > best[1].confidence:
                        best = (k, r)
                if not best:
                    break
                chain_keys.append(best[0])
                chain_tokens.append(best[1].target)
                chain_conf *= best[1].confidence  # multiplikative Kettenwahrscheinlichkeit
                current_target = best[1].target

            if len(chain_keys) >= 2:
                chunks.append(CerebellumChunk(
                    rule_keys=chain_keys,
                    tokens=chain_tokens,
                    confidence=chain_conf,
                ))

        # Deduplizieren (gleiche Token-Sequenz nur 1×) und nach Confidence sortieren
        seen = set()
        unique = []
        for c in sorted(chunks, key=lambda c: (-c.length, -c.confidence)):
            sig = tuple(c.tokens)
            if sig in seen:
                continue
            seen.add(sig)
            unique.append(c)
        # Top 20 reichen für die UI/Auswertung
        return unique[:20]

    def check(self, token_id: int, current_bucket=None):
        """
        Prüft ob ein Token eine Routine triggert.

        v0.13.0: Unterstützt Sequenz-Matching für 2-gram und 3-gram Regeln.
        v0.21.0: Fuzzy-Matching – erlaubt 1 Token Abweichung bei längeren
        Sequenzen. Kürzere Sequenzen (1-gram, 2-gram) werden bevorzugt
        wenn exakte Matches fehlen.
        v0.22.0: Kontext-Bonus – Regeln, die im aktuellen Kontext-Bucket
        beobachtet wurden, bekommen +20% Score. Regeln aus völlig fremdem
        Kontext bekommen -25% Abzug. So lernt das Cerebellum
        "(token, context) → token" statt nur "token → token".
        """
        self._recent_buffer.append(token_id)
        if current_bucket is None:
            current_bucket = self._current_bucket
        now = time.time()
        best_rule = None
        best_score = -1  # Scoring: höher = besserer Match

        for key, rule in self.rules.items():
            if rule.trigger != token_id:
                continue
            if (now - rule.last_fired) < self.RULE_COOLDOWN:
                continue

            if rule.ngram_order == 1:
                # 1-gram: Trigger allein reicht → exakter Match
                score = 1.0 * rule.confidence
            else:
                seq = rule.trigger_sequence
                buf = list(self._recent_buffer)
                if len(buf) < len(seq):
                    continue

                recent = tuple(buf[-len(seq):])

                # Exakter Match → volle Punktzahl
                if recent == seq:
                    score = rule.ngram_order * rule.confidence
                else:
                    # Fuzzy-Match: Wie viele Tokens stimmen überein?
                    matches = sum(1 for a, b in zip(recent, seq) if a == b)
                    match_ratio = matches / len(seq)

                    # Mindestens 50% Übereinstimmung + letztes Token muss stimmen
                    # (letztes Token = trigger, stimmt schon per Definition)
                    if match_ratio < 0.5:
                        continue

                    # Fuzzy-Score: Abzug für jede Abweichung
                    score = rule.ngram_order * rule.confidence * match_ratio * 0.8

            # v0.22.0: Kontext-Modulation
            if current_bucket is not None and rule.context_buckets:
                if current_bucket in rule.context_buckets:
                    score *= 1.2
                else:
                    # Fremder Kontext – Regel ist nur "halb gültig"
                    score *= 0.75

            if score > best_score:
                best_rule = rule
                best_score = score

        return best_rule
    
    def mark_fired(self, rule: CerebellumRule):
        """Markiert eine Regel als gefeuert."""
        rule.last_fired = time.time()
    
    def record_outcome(self, rule_key: str, success: bool):
        """Zeichnet Erfolg/Misserfolg einer Regel auf."""
        if rule_key in self.rules:
            if success:
                self.rules[rule_key].successes += 1
            else:
                self.rules[rule_key].failures += 1
                self.rules[rule_key].confidence *= 0.95
    
    def to_dict(self) -> dict:
        rules_data = {}
        for key, rule in self.rules.items():
            rules_data[key] = {
                "trigger": rule.trigger,
                "target": rule.target,
                "confidence": rule.confidence,
                "avg_delay": rule.avg_delay,
                "successes": rule.successes,
                "failures": rule.failures,
                "ngram_order": rule.ngram_order,
                "trigger_sequence": list(rule.trigger_sequence),
                # v0.22.0: Kontext-Buckets mit serialisieren
                "context_buckets": sorted(rule.context_buckets) if rule.context_buckets else [],
            }
        return {
            "rules": rules_data,
            "chunks": [c.to_dict() for c in self.chunks],
        }

    def from_dict(self, data: dict):
        for key, rd in data.get("rules", {}).items():
            seq = tuple(rd.get("trigger_sequence", [rd["trigger"]]))
            ctx_buckets = set(rd.get("context_buckets", []) or [])
            self.rules[key] = CerebellumRule(
                rd["trigger"], rd["target"], rd["confidence"],
                rd.get("avg_delay", 60),
                ngram_order=rd.get("ngram_order", 1),
                trigger_sequence=seq,
                context_buckets=ctx_buckets,
            )
            self.rules[key].successes = rd.get("successes", 0)
            self.rules[key].failures = rd.get("failures", 0)
        # v0.22.0: Chunks rekonstruieren (best-effort, Strukturen nur)
        for cd in data.get("chunks", []) or []:
            try:
                self.chunks.append(CerebellumChunk(
                    rule_keys=cd.get("rule_keys", []),
                    tokens=cd.get("tokens", []),
                    confidence=cd.get("confidence", 0.0),
                ))
            except Exception:  # pragma: no cover
                continue

    @property
    def stats(self) -> dict:
        total_fires = self._total_fired
        total_success = sum(r.successes for r in self.rules.values())
        n_by_order = {1: 0, 2: 0, 3: 0, 4: 0}
        ctx_aware = 0
        for r in self.rules.values():
            n_by_order[r.ngram_order] = n_by_order.get(r.ngram_order, 0) + 1
            if r.context_buckets:
                ctx_aware += 1
        return {
            "rules_count": len(self.rules),
            "rules_1gram": n_by_order[1],
            "rules_2gram": n_by_order[2],
            "rules_3gram": n_by_order[3],
            "rules_4gram": n_by_order[4],
            "rules_context_aware": ctx_aware,
            "chunks_count": len(self.chunks),
            "total_fires": total_fires,
            "success_rate": f"{total_success / max(1, total_fires):.0%}",
            "top_rules": [
                {"key": k, "conf": round(r.confidence, 2),
                 "fires": r.successes + r.failures, "order": r.ngram_order,
                 "ctx_buckets": len(r.context_buckets)}
                for k, r in sorted(self.rules.items(),
                                    key=lambda x: x[1].confidence, reverse=True)[:5]
            ],
            "top_chunks": [c.to_dict() for c in self.chunks[:5]],
        }
