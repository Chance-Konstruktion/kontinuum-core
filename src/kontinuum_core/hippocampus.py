"""
╔══════════════════════════════════════════════════════════════════╗
║  KONTINUUM – Hippocampus                                        ║
║  Gedächtnissystem: Arbeitsgedächtnis + Konsolidierung           ║
║                                                                  ║
║  v0.12.0 – Intelligenz-Upgrade:                                ║
║  - Mode-Index korrigiert (ctx[-3] statt ctx[12])               ║
║  - Phase 4: Wochentag-Gedächtnis (Werktag vs Wochenende)      ║
║  - Kontextvektor: [time(9) + hypothalamus(9) + insula(3)]     ║
║  = Bis zu 96 Buckets. Robusteres Bucket-Addressing.            ║
║                                                                  ║
║  v0.13.0 – Lernbeschleunigung:                                 ║
║  - Multi-Window Shadow Validation (60s, 300s, 1800s)           ║
║  - Früheres Bucketing (Phase 2 ab 100 statt 2000 Events)      ║
║  - Accuracy pro Zeithorizont getrackt                          ║
║                                                                  ║
║  v0.15.0 – Stichprobengröße (n_obs):                           ║
║  - Predictions enthalten jetzt n_obs (Beobachtungsanzahl)      ║
║  - PFC kann Patterns nach statistischer Signifikanz gaten      ║
╚══════════════════════════════════════════════════════════════════╝
"""

import logging
import math
import time
from collections import defaultdict, deque
from datetime import datetime, timezone

_LOGGER = logging.getLogger(__name__)


class Hippocampus:
    """
    Gedächtnissystem von KONTINUUM.

    Kontext-Bucket (v0.12.0):
        Bis zu 96 Buckets = 6 Zeit × 4 Modi × 2 Energie × 2 Tagestyp
        Kontextvektor: [time(9) + hypothalamus(9) + insula(3)] = 21 Dims
        Mode-Index: ctx[-3] (robust, unabhängig von Hypothalamus-Dimensionen)
    """

    NGRAM_SIZES = [1, 2, 3, 4]

    TIME_BLOCKS = 6
    MODE_GROUPS = 4
    ENERGY_LEVELS = 2
    DAY_TYPES = 2
    TOTAL_BUCKETS = TIME_BLOCKS * MODE_GROUPS * ENERGY_LEVELS * DAY_TYPES  # 96

    DECAY_RATE = 0.993        # Default für "ausgeglichen"
    MIN_OBSERVATIONS = 2
    NGRAM_WEIGHTS = {1: 0.15, 2: 0.4, 3: 0.7, 4: 0.95}
    MAX_NGRAMS_PER_BUCKET = 1000
    
    # Multi-Window Shadow Validation (v0.13.0)
    SHADOW_WINDOWS = [60, 300, 1800]  # 1min, 5min, 30min

    def __init__(self):
        self.transitions = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )
        self.totals = defaultdict(lambda: defaultdict(float))
        self.durations = defaultdict(list)
        self.buffer = deque(maxlen=30)
        self.last_event_time = None
        self.shadow_predictions = deque(maxlen=200)
        self.shadow_hits = 0
        self.shadow_misses = 0
        self.shadow_total = 0
        # Per-Window Accuracy (v0.13.0)
        self.shadow_hits_by_window = {w: 0 for w in self.SHADOW_WINDOWS}
        self.shadow_total_by_window = {w: 0 for w in self.SHADOW_WINDOWS}
        self.total_events = 0
        self.last_decay_day = int(time.time() / 86400)
    
    def _context_bucket(self, ctx: list) -> int:
        """
        Kontext-Vektor → diskreter Bucket.

        ADAPTIV (v0.13.0 – beschleunigt):
        - < 100 Events: 6 Buckets (nur Zeit)
        - >= 100 Events: 24 Buckets (Zeit × Modus)
        - >= 500 Events: 48 Buckets (Zeit × Modus × Energie)
        - >= 2000 Events: 96 Buckets (Zeit × Modus × Energie × Tagestyp)
        """
        if len(ctx) < 7:
            return 0

        # ── Zeit: Stunde aus sin/cos → 6 Blöcke ──
        hour = math.atan2(ctx[0], ctx[1]) * 24 / (2 * math.pi)
        if hour < 0:
            hour += 24
        time_b = min(5, int(hour / 4))

        # Phase 1: Nur Zeit (< 100 Events)
        if self.total_events < 100:
            return time_b

        # ── Modus: ctx[-3] = mode_index/5.0 (Insula ist immer die letzten 3 Dims) ──
        if len(ctx) >= 3:
            mode_val = ctx[-3]
            if mode_val < 0.1:
                mode_b = 0      # sleeping
            elif mode_val < 0.5:
                mode_b = 1      # waking_up, active
            elif mode_val < 0.7:
                mode_b = 2      # relaxing
            else:
                mode_b = 3      # cooking, away
        else:
            mode_b = 1

        # Phase 2: Zeit × Modus (< 500 Events)
        if self.total_events < 500:
            return time_b * self.MODE_GROUPS + mode_b

        # ── Energie: ctx[9] = battery_state/3.0 → 2 Stufen ──
        if len(ctx) > 9 and ctx[9] is not None:
            energy_b = 0 if ctx[9] < 0.33 else 1
        else:
            energy_b = 1

        # Phase 3: Zeit × Modus × Energie (< 2000 Events)
        if self.total_events < 2000:
            return (time_b * self.MODE_GROUPS + mode_b) * self.ENERGY_LEVELS + energy_b

        # ── Tagestyp: ctx[6] = is_weekend → 2 Gruppen (v0.12.0) ──
        daytype_b = 1 if (len(ctx) > 6 and ctx[6] > 0.5) else 0

        # Phase 4: Voll (>= 2000 Events)
        return ((time_b * self.MODE_GROUPS + mode_b) * self.ENERGY_LEVELS + energy_b) * self.DAY_TYPES + daytype_b
    
    def _neighbor_buckets(self, bucket: int) -> list:
        """Nachbar-Buckets: adaptiv je nach Phase."""
        # Phase 1: 6 Buckets (nur Zeit)
        if self.total_events < 100:
            return [(bucket - 1) % self.TIME_BLOCKS,
                    (bucket + 1) % self.TIME_BLOCKS]

        # Phase 2: 24 Buckets (Zeit × Modus)
        if self.total_events < 500:
            mode_b = bucket % self.MODE_GROUPS
            time_b = bucket // self.MODE_GROUPS
            return [
                ((time_b - 1) % self.TIME_BLOCKS) * self.MODE_GROUPS + mode_b,
                ((time_b + 1) % self.TIME_BLOCKS) * self.MODE_GROUPS + mode_b,
            ]

        # Phase 3: 48 Buckets (Zeit × Modus × Energie)
        if self.total_events < 2000:
            energy_b = bucket % self.ENERGY_LEVELS
            rest = bucket // self.ENERGY_LEVELS
            mode_b = rest % self.MODE_GROUPS
            time_b = rest // self.MODE_GROUPS

            neighbors = []
            for dt in [-1, 1]:
                nt = (time_b + dt) % self.TIME_BLOCKS
                neighbors.append((nt * self.MODE_GROUPS + mode_b) * self.ENERGY_LEVELS + energy_b)
            other_e = 1 - energy_b
            neighbors.append((time_b * self.MODE_GROUPS + mode_b) * self.ENERGY_LEVELS + other_e)
            return neighbors

        # Phase 4: 96 Buckets (Zeit × Modus × Energie × Tagestyp, v0.12.0)
        daytype_b = bucket % self.DAY_TYPES
        rest = bucket // self.DAY_TYPES
        energy_b = rest % self.ENERGY_LEVELS
        rest2 = rest // self.ENERGY_LEVELS
        mode_b = rest2 % self.MODE_GROUPS
        time_b = rest2 // self.MODE_GROUPS

        neighbors = []
        for dt in [-1, 1]:
            nt = (time_b + dt) % self.TIME_BLOCKS
            neighbors.append(((nt * self.MODE_GROUPS + mode_b) * self.ENERGY_LEVELS + energy_b) * self.DAY_TYPES + daytype_b)
        # Anderer Energielevel
        other_e = 1 - energy_b
        neighbors.append(((time_b * self.MODE_GROUPS + mode_b) * self.ENERGY_LEVELS + other_e) * self.DAY_TYPES + daytype_b)
        # Anderer Tagestyp (Werktag ↔ Wochenende)
        other_d = 1 - daytype_b
        neighbors.append(((time_b * self.MODE_GROUPS + mode_b) * self.ENERGY_LEVELS + energy_b) * self.DAY_TYPES + other_d)
        return neighbors
    
    def learn(self, token_id: int, ctx: list, timestamp, learn_weight: float = 1.0):
        """Lernt aus einem neuen Event. learn_weight moduliert die Lernstärke (Predictive Processing)."""
        self.total_events += 1

        current_day = int(time.time() / 86400)
        if current_day > self.last_decay_day:
            self._apply_decay(current_day - self.last_decay_day)
            self.last_decay_day = current_day

        if self.last_event_time:
            try:
                duration = (timestamp - self.last_event_time).total_seconds()
            except (TypeError, AttributeError):
                duration = 0.0
        else:
            duration = 0.0
        self.last_event_time = timestamp
        
        bucket = self._context_bucket(ctx)
        self._validate_shadow(token_id, timestamp)
        
        seq = list(self.buffer)
        # learn_weight: Surprise-moduliert (0.2 = erwartet, 2.5 = Überraschung)
        weight = max(0.1, min(3.0, learn_weight))
        for n in self.NGRAM_SIZES:
            if len(seq) >= n:
                ngram = tuple(seq[-n:])
                self.transitions[bucket][ngram][token_id] += weight
                self.totals[bucket][ngram] += weight
        
        if len(self.transitions[bucket]) > self.MAX_NGRAMS_PER_BUCKET:
            self._evict_bucket(bucket)
        
        if seq and 0 < duration < 7200:
            dur_key = f"{seq[-1]}_{token_id}"
            durs = self.durations[dur_key]
            durs.append(duration)
            if len(durs) > 50:
                self.durations[dur_key] = durs[-50:]
        
        self.buffer.append(token_id)
    
    def _evict_bucket(self, bucket: int):
        """LFU Eviction – behält die Top-K N-Gramme."""
        trans = self.transitions[bucket]
        tots = self.totals[bucket]
        scored = [(sum(trans[ng].values()), ng) for ng in trans]
        scored.sort(reverse=True)
        keep = set(ng for _, ng in scored[:self.MAX_NGRAMS_PER_BUCKET])
        for _, ngram in scored[self.MAX_NGRAMS_PER_BUCKET:]:
            del trans[ngram]
            if ngram in tots:
                del tots[ngram]
    
    def predict(self, ctx: list, top_k: int = 5) -> list:
        """
        Sagt die wahrscheinlichsten nächsten Events vorher.

        Returns:
            Liste von (token_id, prob, conf, source, n_obs) Tupeln.
            n_obs = Anzahl Beobachtungen für dieses Pattern (Stichprobengröße).
        """
        seq = list(self.buffer)
        bucket = self._context_bucket(ctx)
        scores = self._score_predictions(seq, bucket)

        results = []
        for token_id, data in scores.items():
            # Konfidenz: prob × weight + Bonus für viele Beobachtungen
            n_obs = data["n_obs"]
            obs_bonus = min(n_obs / 100.0, 0.3)  # Bis zu +0.3 für >100 Beobachtungen
            conf = min(data["prob"] * data["weight"] * 2 + obs_bonus, 1.0)
            if conf >= 0.20:  # Höhere Schwelle = weniger Rauschen
                results.append((token_id, round(data["prob"], 4),
                                round(conf, 4), data["source"],
                                data["n_obs"]))

        # Sortierung: Konfidenz × Wahrscheinlichkeit (stärkere Gewichtung auf Konfidenz)
        results.sort(key=lambda x: x[2] * 0.7 + x[1] * 0.3, reverse=True)

        for tok_id, prob, conf, src, _n in results[:3]:
            if conf >= 0.3:
                self.shadow_predictions.append((
                    self.last_event_time or datetime.now(timezone.utc),
                    tok_id, conf
                ))

        return results[:top_k]
    
    def _score_predictions(self, seq: list, bucket: int) -> dict:
        """Berechnet Vorhersage-Scores mit Stichprobengröße."""
        scores = defaultdict(lambda: {"prob": 0, "weight": 0, "source": "", "n_obs": 0})
        neighbors = self._neighbor_buckets(bucket)

        for n in self.NGRAM_SIZES:
            if len(seq) < n:
                continue
            ngram = tuple(seq[-n:])
            w = self.NGRAM_WEIGHTS[n]

            for bk in [bucket] + neighbors:
                if bk == bucket:
                    discount = 1.0
                elif bk in neighbors[:2]:
                    discount = 0.4
                else:
                    discount = 0.3

                if ngram not in self.transitions[bk]:
                    continue
                total = self.totals[bk][ngram]
                if total < self.MIN_OBSERVATIONS:
                    continue

                for token_id, count in self.transitions[bk][ngram].items():
                    prob = (count / total) * discount
                    eff_weight = w * discount
                    if prob * eff_weight > scores[token_id]["prob"] * scores[token_id]["weight"]:
                        scores[token_id] = {
                            "prob": prob,
                            "weight": eff_weight,
                            "source": f"{n}-gram(n={total:.0f})",
                            "n_obs": int(total),
                        }

        return scores
    
    def predict_duration(self, from_token: int, to_token: int) -> float:
        """Median-Zeitabstand zwischen zwei Events."""
        durs = self.durations.get(f"{from_token}_{to_token}", [])
        if len(durs) < 3:
            return None
        return sorted(durs)[len(durs) // 2]
    
    def _validate_shadow(self, actual_token: int, timestamp):
        """
        Multi-Window Shadow Validation (v0.13.0).

        Prüft Vorhersagen gegen mehrere Zeithorizonte (60s, 300s, 1800s).
        Ein Hit im kürzesten Fenster zählt auch für längere Fenster.
        Die Haupt-Accuracy nutzt das längste Fenster (1800s) als Cutoff.
        """
        from datetime import timedelta
        try:
            max_window = max(self.SHADOW_WINDOWS)
            cutoff = timestamp - timedelta(seconds=max_window)
        except (TypeError, AttributeError):
            return

        matched = False
        surviving = deque(maxlen=200)
        for pred_ts, pred_tok, pred_conf in self.shadow_predictions:
            if pred_ts < cutoff:
                # Abgelaufen: Miss in allen Windows
                self.shadow_misses += 1
                self.shadow_total += 1
                for w in self.SHADOW_WINDOWS:
                    self.shadow_total_by_window[w] = self.shadow_total_by_window.get(w, 0) + 1
                continue
            if pred_tok == actual_token and not matched:
                # Hit: bestimme welche Fenster getroffen wurden
                try:
                    age = (timestamp - pred_ts).total_seconds()
                except (TypeError, AttributeError):
                    age = 0
                self.shadow_hits += 1
                self.shadow_total += 1
                for w in self.SHADOW_WINDOWS:
                    self.shadow_total_by_window[w] = self.shadow_total_by_window.get(w, 0) + 1
                    if age <= w:
                        self.shadow_hits_by_window[w] = self.shadow_hits_by_window.get(w, 0) + 1
                matched = True
                continue
            surviving.append((pred_ts, pred_tok, pred_conf))
        self.shadow_predictions = surviving
    
    def _apply_decay(self, days: int):
        """Exponentieller Decay."""
        factor = self.DECAY_RATE ** days
        for bucket in self.transitions:
            for ngram in self.transitions[bucket]:
                for token in self.transitions[bucket][ngram]:
                    self.transitions[bucket][ngram][token] *= factor
            for ngram in self.totals[bucket]:
                self.totals[bucket][ngram] *= factor
    
    def to_dict(self) -> dict:
        trans = {}
        for b, ngs in self.transitions.items():
            trans[str(b)] = {}
            for ng, toks in ngs.items():
                k = ",".join(str(t) for t in ng)
                trans[str(b)][k] = {str(t): c for t, c in toks.items()}
        tots = {}
        for b, ngs in self.totals.items():
            tots[str(b)] = {}
            for ng, tot in ngs.items():
                tots[str(b)][",".join(str(t) for t in ng)] = tot
        return {
            "transitions": trans, "totals": tots,
            "durations": dict(self.durations),
            "total_events": self.total_events,
            "last_decay_day": self.last_decay_day,
            "buffer": list(self.buffer),
            "last_event_time": self.last_event_time.isoformat() if self.last_event_time else None,
            "shadow_hits": self.shadow_hits,
            "shadow_misses": self.shadow_misses,
            "shadow_total": self.shadow_total,
            "shadow_hits_by_window": self.shadow_hits_by_window,
            "shadow_total_by_window": self.shadow_total_by_window,
        }

    def from_dict(self, data: dict):
        self.total_events = data.get("total_events", 0)
        self.last_decay_day = data.get("last_decay_day", int(time.time() / 86400))
        self.shadow_hits = data.get("shadow_hits", 0)
        self.shadow_misses = data.get("shadow_misses", 0)
        self.shadow_total = data.get("shadow_total", 0)
        # v0.13.0: Per-Window Stats laden
        self.shadow_hits_by_window = {
            int(k): v for k, v in data.get("shadow_hits_by_window", {}).items()
        }
        self.shadow_total_by_window = {
            int(k): v for k, v in data.get("shadow_total_by_window", {}).items()
        }
        # Fehlende Windows initialisieren
        for w in self.SHADOW_WINDOWS:
            self.shadow_hits_by_window.setdefault(w, 0)
            self.shadow_total_by_window.setdefault(w, 0)
        self.durations = defaultdict(list, data.get("durations", {}))
        self.buffer = deque(data.get("buffer", []), maxlen=30)
        lt = data.get("last_event_time")
        if lt:
            try:
                self.last_event_time = datetime.fromisoformat(lt)
            except (ValueError, TypeError):
                pass
        for bs, ngs in data.get("transitions", {}).items():
            b = int(bs)
            for ngs_str, toks in ngs.items():
                ng = tuple(int(t) for t in ngs_str.split(","))
                for ts, c in toks.items():
                    self.transitions[b][ng][int(ts)] = c
        for bs, ngs in data.get("totals", {}).items():
            b = int(bs)
            for ngs_str, tot in ngs.items():
                ng = tuple(int(t) for t in ngs_str.split(","))
                self.totals[b][ng] = tot
    
    @property
    def accuracy(self) -> float:
        if self.shadow_total == 0:
            return 0.0
        return self.shadow_hits / self.shadow_total
    
    @property
    def stats(self) -> dict:
        n_patterns = sum(len(ngs) for ngs in self.transitions.values())
        n_transitions = sum(
            len(ts) for bk in self.transitions.values() for ts in bk.values()
        )
        # Per-Window Accuracy (v0.13.0)
        accuracy_by_window = {}
        for w in self.SHADOW_WINDOWS:
            total = self.shadow_total_by_window.get(w, 0)
            hits = self.shadow_hits_by_window.get(w, 0)
            accuracy_by_window[f"{w}s"] = f"{hits / total:.1%}" if total > 0 else "0.0%"
        return {
            "total_events": self.total_events,
            "patterns": n_patterns, "transitions": n_transitions,
            "buckets_active": len(self.transitions),
            "total_buckets": self.TOTAL_BUCKETS,
            "duration_pairs": len(self.durations),
            "buffer_size": len(self.buffer),
            "memory_kb": round(n_transitions * 32 / 1024, 1),
            "shadow_hits": self.shadow_hits,
            "shadow_misses": self.shadow_misses,
            "shadow_total": self.shadow_total,
            "accuracy": f"{self.accuracy:.1%}",
            "accuracy_by_window": accuracy_by_window,
        }
