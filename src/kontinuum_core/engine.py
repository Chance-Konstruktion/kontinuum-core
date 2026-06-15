"""Main public API for KONTINUUM Core.

The KontinuumEngine wires all 25 neuro-inspired modules into a single
pipeline. It is HA-free and can be used standalone or embedded in a
host integration (ha-kontinuum, ha-kontinuum-lite).

Per-event pipeline (the modules that influence the decision):
    thalamus.process()        → token_id + normalized state
    locus_coeruleus           → arousal (event density)
    hypothalamus.absorb()     → homeostasis side-channel
    spatial_cortex/entorhinal → room map + next-room anticipation
    insula.process()          → mode detection
    context vector            → time(9) + hypothalamus(9) + insula(3) = 21
    hippocampus.predict()     → top-k sequence predictions
    predictive.compute_surprise() + get_learn_weight()
    neurorhythms              → circadian/burst modulation of the rate
    suprachiasmatic           → learned household circadian phase nudge
    acetylcholine             → expected-uncertainty (context) rate damping
    hippocampus.learn(weight)
    cortisol / serotonin      → slow stress / mood hormones (observe)
    basal_ganglia             → passive Q-observation + Go/NoGo re-ranking
    cerebellum                → reflex rule check + reflex injection
    nucleus_accumbens         → habit bias in the ranking
    lateral_habenula          → anti-reward suppression of chronic rejects
    cortisol.damping()        → conservative ranking under sustained stress
    anterior_cingulate        → conflict monitor → cognitive_control damping
    prefrontal_cortex.evaluate→ advisory Decision (SHADOW mode: never acts)
    subthalamic_nucleus       → "hold your horses" brake under conflict

Outcome learning (reward modules) is closed by the host via
:meth:`KontinuumEngine.feedback`; without a host the reward modules only
observe. Periodic maintenance (cerebellum rule compilation, entorhinal
pruning) runs inline on a coarse cadence.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .acetylcholine import Acetylcholine
from .amygdala import Amygdala
from .anterior_cingulate import AnteriorCingulate
from .basal_ganglia import BasalGanglia
from .bdnf import Bdnf
from .cerebellum import Cerebellum
from .cortisol import Cortisol
from .entorhinal_cortex import EntorhinalCortex
from .habenula import LateralHabenula
from .hippocampus import Hippocampus
from .hypothalamus import Hypothalamus
from .insula import Insula
from .locus_coeruleus import LocusCoeruleus
from .metaplasticity import Metaplasticity
from .neurorhythms import Neurorhythms
from .nucleus_accumbens import NucleusAccumbens
from .predictive_processing import PredictiveProcessing
from .prefrontal_cortex import Decision, PrefrontalCortex
from .reticular import Reticular
from .scheduler import Scheduler
from .serotonin import Serotonin
from .sleep_consolidation import SleepConsolidation
from .spatial_cortex import SpatialCortex
from .subthalamic_nucleus import SubthalamicNucleus
from .suprachiasmatic import SuprachiasmaticNucleus
from .thalamus import Thalamus

# Compile cerebellum reflex rules out of hippocampus memory every N events.
# Coarse enough to stay cheap, fine enough that reflexes appear within an
# hour of activity (the HA integration recompiles on a ~10 min wall clock).
COMPILE_EVERY = 50
# Prune the entorhinal room map at most once per day.
ENTORHINAL_PRUNE_SECONDS = 86400

logger = logging.getLogger(__name__)


@dataclass
class EngineSnapshot:
    surprise: float = 0.0
    anomaly: bool = False
    learning_state: str = "cold_start"
    tick_count: int = 0
    token_id: Optional[int] = None
    token: Optional[str] = None
    predictions: List[Any] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


class KontinuumEngine:
    """Neuro-inspired learning engine. HA-free.

    Args:
        config: Optional configuration dict (reserved for future use).
        scheduler: Optional :class:`Scheduler` instance. When provided it is
            wired into :class:`Metaplasticity` so periodic parameter updates
            can be activated via ``metaplasticity.start()``. When omitted,
            metaplasticity stays inert (still queryable, no scheduling).
        storage_path: Optional directory for persistent state of
            sub-modules that opt-in (currently Metaplasticity only).
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Scheduler] = None,
        storage_path: Optional[str] = None,
    ):
        self.config = config or {}
        self.scheduler = scheduler
        self.thalamus = Thalamus()
        self.hippocampus = Hippocampus()
        self.predictive = PredictiveProcessing()
        self.cerebellum = Cerebellum()
        self.basal_ganglia = BasalGanglia()
        self.neurorhythms = Neurorhythms()
        self.sleep_consolidation = SleepConsolidation()
        self.amygdala = Amygdala()
        self.insula = Insula()
        self.hypothalamus = Hypothalamus()
        self.spatial_cortex = SpatialCortex()
        self.prefrontal_cortex = PrefrontalCortex(self.amygdala)
        self.anterior_cingulate = AnteriorCingulate()
        self.entorhinal_cortex = EntorhinalCortex()
        self.locus_coeruleus = LocusCoeruleus()
        self.nucleus_accumbens = NucleusAccumbens()
        self.reticular = Reticular()
        # Reticular filtering is arousal-modulated by the Locus Coeruleus.
        self.reticular.set_arousal_source(self.locus_coeruleus)

        # --- Extended neuromodulator / region set (all O(1), Pi-friendly) ---
        # Regions:
        self.habenula = LateralHabenula()          # anti-reward (stop nagging)
        self.subthalamic = SubthalamicNucleus()    # "hold your horses" brake
        self.suprachiasmatic = SuprachiasmaticNucleus()  # learned inner clock
        # Slow neuromodulators / hormones:
        self.serotonin = Serotonin()               # mood / patience
        self.acetylcholine = Acetylcholine()       # expected uncertainty
        self.cortisol = Cortisol()                 # systemic stress hormone
        # Neurotrophic maintenance ("vitamin" layer):
        self.bdnf = Bdnf()                          # use-dependent protection

        self.metaplasticity = Metaplasticity(
            storage_path=storage_path,
            scheduler=scheduler,
            brain_modules={
                "hippocampus": self.hippocampus,
                "predictive": self.predictive,
                "cerebellum": self.cerebellum,
                "basal_ganglia": self.basal_ganglia,
                "reticular": self.reticular,
                "accumbens": self.nucleus_accumbens,
                "locus": self.locus_coeruleus,
            },
        )
        self.tick_count = 0

        # Engine-level cross-event state.
        self._last_room: Optional[str] = None
        self._expected_next_room: Optional[str] = None
        self._last_event_ts: float = 0.0
        # Snapshot of the most recent advisory decision so a host can close
        # the reward loops through feedback().
        self._last_decision_ctx: Optional[Dict[str, Any]] = None
        # Last suprachiasmatic phase nudge, surfaced in the snapshot extra.
        self._last_scn_gain: float = 1.0

    # ------------------------------------------------------------------
    # Entity registration
    # ------------------------------------------------------------------
    def register_entity(self, entity_id: str, **kwargs) -> None:
        self.thalamus.register_entity(entity_id, **kwargs)

    # ------------------------------------------------------------------
    # Main observation pipeline
    # ------------------------------------------------------------------
    def observe(self, event: Optional[Dict[str, Any]] = None) -> EngineSnapshot:
        self.tick_count += 1
        event = event or {}
        entity_id = event.get("entity_id")
        new_state = event.get("new_state")

        if not entity_id or new_state is None:
            return self._snapshot(extra={"skipped": "no_entity_or_state"})

        timestamp = event.get("timestamp") or datetime.now(timezone.utc)
        old_state = event.get("old_state")

        signal = self.thalamus.process(entity_id, new_state, old_state, timestamp)
        if signal is None:
            return self._snapshot(extra={"skipped": "filtered"})

        token_id = signal["token_id"]
        token = signal["token"]
        room = signal["room"]
        semantic = signal["semantic"]
        state = signal["state"]

        # Reticular attention gate: drop repetitive bursts of the SAME entity so
        # sensor noise (a flapping contact, a chatty power meter) doesn't flood
        # learning. Judged by the *event* timestamp, not wall clock, so replayed
        # / simulated streams are gated by real event rate, not replay speed.
        ev_now = timestamp.timestamp() if hasattr(timestamp, "timestamp") else None
        domain = entity_id.split(".")[0] if entity_id else ""
        if not self.reticular.should_process(entity_id, domain=domain, now=ev_now):
            return self._snapshot(extra={"skipped": "burst_filtered"})

        # Arousal + idle tracking (Locus Coeruleus feeds the Reticular gate;
        # Sleep counts events for the host-triggered consolidation cycle).
        self.locus_coeruleus.observe_event()
        self.sleep_consolidation.observe_event()

        # Homeostasis absorption (energy/climate side-channel)
        if self.hypothalamus.is_hypothalamus_signal(semantic):
            self.hypothalamus.absorb(room, semantic, state, entity_id)

        # Spatial map + entorhinal room-transition anticipation. The spatial
        # cortex turns raw presence/motion/tracker signals into debounced
        # "entered <room>" tokens; on each confirmed room entry we learn
        # "where do people go from here" and pre-activate tokens in the
        # expected next room during ranking.
        if semantic and self.spatial_cortex.is_spatial_signal(semantic):
            for tok in self.spatial_cortex.absorb(room, semantic, state, entity_id):
                if tok.get("semantic") == "spatial" and tok.get("state") == "entered":
                    new_room = tok.get("room")
                    if not new_room:
                        continue
                    if self._last_room and self._last_room != new_room:
                        self.entorhinal_cortex.observe_transition(self._last_room, new_room)
                    self._last_room = new_room
                    self._expected_next_room = self.entorhinal_cortex.predict_next_room(new_room)

        # Mode detection (insula may emit transition tokens; kept in insula
        # state so get_mode_context() reflects them).
        self.insula.process(semantic, state, room, token)

        # Build 21-dim context vector
        ctx = (
            list(self.thalamus.encode_time_context(timestamp))
            + list(self.hypothalamus.get_context_vector())
            + list(self.insula.get_mode_context())
        )

        # Context bucket is needed for the acetylcholine rate gate below, so
        # derive it here (pure function of ctx) instead of after learning.
        bucket = self.hippocampus._context_bucket(ctx)

        # Predict (pre-learn) → surprise → learn weight. The adaptive anomaly
        # threshold is read BEFORE compute_surprise so the current event does
        # not shift its own evaluation baseline.
        pre_predictions = self.hippocampus.predict(ctx, top_k=5)
        anomaly_threshold = self.predictive.anomaly_threshold()
        surprise = self.predictive.compute_surprise(token_id, pre_predictions)
        anomaly_flag = surprise >= anomaly_threshold
        learn_weight = self.predictive.get_learn_weight()

        # Neurorhythms: register surprise + modulate the learning rate
        # (circadian rhythm and dopamine bursts).
        self.neurorhythms.register_surprise(token_id, surprise)
        learn_weight = self.neurorhythms.modulate_learning(learn_weight)

        # Suprachiasmatic nucleus: entrain to THIS home's activity rhythm and
        # nudge the learning rate (±15%) toward the household's real day. Starts
        # neutral (1.0) until warmed up, so it never fights the cold-start.
        hour = timestamp.hour if hasattr(timestamp, "hour") else \
            datetime.now(timezone.utc).hour
        self.suprachiasmatic.observe(hour)
        self._last_scn_gain = self.suprachiasmatic.phase_gain(hour)
        learn_weight *= self._last_scn_gain

        # Acetylcholine: read the bucket's *expected* uncertainty BEFORE folding
        # in the current surprise, so a reliably-noisy context damps learning
        # (we don't chase irreducible jitter) without the event judging itself.
        learn_weight *= self.acetylcholine.learn_gain(bucket)
        self.acetylcholine.observe(bucket, surprise)
        learn_weight = max(0.05, min(10.0, learn_weight))

        # Hippocampus learns (weighted by surprise + rhythms + clock + ACh).
        self.hippocampus.learn(token_id, ctx, timestamp, learn_weight=learn_weight)

        # Slow hormones observe every event (ranking-side effects only):
        #   cortisol  – integrates sustained surprise/anomaly into stress,
        #   serotonin – gentle mood dip on anomalies (recovers via feedback).
        self.cortisol.observe(surprise, anomaly_flag)
        self.serotonin.observe(anomaly_flag)

        # Basal ganglia passive observation (the home "wants" this state).
        self.basal_ganglia.process_observation(token_id, bucket)

        # Cerebellum reflex check (fast routines).
        self.cerebellum.set_context(bucket)
        fired_rule = self.cerebellum.check(token_id, current_bucket=bucket)
        fired_rule_key = None
        if fired_rule is not None:
            self.cerebellum.mark_fired(fired_rule)
            self.cerebellum._total_fired += 1
            fired_rule_key = next(
                (k for k, v in self.cerebellum.rules.items() if v is fired_rule),
                None,
            )

        # Build the ranked candidate list. A confident reflex is injected as a
        # high-priority prediction, then the whole list is re-ranked by basal
        # ganglia / accumbens / arousal / ACC-control / entorhinal anticipation.
        raw_predictions = self.hippocampus.predict(ctx, top_k=5)
        predictions = raw_predictions
        if fired_rule is not None and fired_rule.confidence >= 0.7:
            reflex_pred = (
                fired_rule.target,
                fired_rule.confidence,
                fired_rule.confidence,
                "cerebellum",
                fired_rule.successes + 50,  # n_obs bonus for an established reflex
            )
            predictions = [reflex_pred] + (predictions or [])
        if predictions:
            predictions = self._rank_predictions(predictions, bucket, room, timestamp)

        # PFC decision (the amygdala risk assessment runs inside evaluate()).
        # Default operation mode is SHADOW: the core only *recommends*.
        decision = self.prefrontal_cortex.evaluate(
            predictions, self.thalamus, self.basal_ganglia, bucket
        )

        # ACC conflict monitor: how much do the module votes disagree? Feeds
        # cognitive_control, which damps confidence in the NEXT ranking round.
        self.anterior_cingulate.observe_decision(
            self._build_acc_proposals(
                raw_predictions, predictions, fired_rule, decision
            )
        )

        # Subthalamic nucleus "hold your horses": under high conflict + a thin
        # margin between the top two candidates, recommend waiting instead of
        # acting. Serotonin (patience) tunes how readily it holds. In SHADOW the
        # decision is already OBSERVE, so this only ever brakes an actionable
        # stage — a safety net, never a new action.
        top_conf = predictions[0][2] if predictions else 0.0
        runner_up = predictions[1][2] if predictions and len(predictions) > 1 else 0.0
        stn_brake = self.subthalamic.compute_brake(
            self.anterior_cingulate.conflict_level, top_conf, runner_up
        )
        stn_hold = False
        if (decision is not None
                and decision.stage in (Decision.SUGGEST, Decision.CONFIRM, Decision.EXECUTE)
                and self.subthalamic.should_hold(self.serotonin.get_patience())):
            stn_hold = True
            decision.stage = Decision.OBSERVE
            decision.reasons = list(decision.reasons or []) + [
                f"STN-Hold: Konflikt {self.anterior_cingulate.conflict_level:.2f}, "
                f"Marge {max(0.0, top_conf - runner_up):.2f}"
            ]

        self._remember_decision(decision, fired_rule_key, bucket, room, timestamp)

        prev_event_ts = self._last_event_ts
        self._maybe_maintain(timestamp, prev_event_ts)
        self._last_event_ts = time.time()

        return self._snapshot(
            surprise=surprise,
            anomaly=anomaly_flag,
            token_id=token_id,
            token=token,
            predictions=predictions,
            extra=self._build_extra(
                raw_predictions, fired_rule, decision, anomaly_threshold,
                prev_event_ts, stn_brake, stn_hold,
            ),
        )

    def evaluate(self, context: Optional[Dict[str, Any]] = None) -> EngineSnapshot:
        return self.observe(context or {})

    # ------------------------------------------------------------------
    # Outcome learning (host-driven reward loop)
    # ------------------------------------------------------------------
    def feedback(self, positive: bool) -> bool:
        """Teach the reward modules the outcome of the last decision.

        A host that executed (or observed the user accept/override) the
        engine's most recent advisory decision calls this so the outcome
        modules actually learn instead of only observing:

        * basal ganglia – TD Q-update + habit tracking,
        * nucleus accumbens – habit bias for ``(state, action)``,
        * neurorhythms – dopamine burst on surprising success / dip on
          surprising failure,
        * cerebellum – reflex-rule confidence (recover on success, decay on
          failure) when a rule fired,
        * serotonin – slow mood baseline (patience) toward success,
        * lateral habenula – disappointment memory: relieved on success,
          deepened on rejection (anti-reward suppression),
        * cortisol – systemic stress bumped on rejection,
        * bdnf – trophic protection grown for proven actions / reflexes,
        * anterior cingulate – error-rate monitor.

        Returns ``True`` when a decision was available to reinforce.
        The remembered decision is consumed (one feedback per decision).
        """
        ctx = self._last_decision_ctx
        if not ctx:
            return False

        token_id = ctx["token_id"]
        entity_id = ctx["entity_id"]
        if entity_id:
            # Register + resolve together: process_outcome needs a pending
            # action keyed by entity_id.
            self.basal_ganglia.register_action(
                entity_id, token_id, ctx["bucket"], ctx["token"]
            )
            self.basal_ganglia.process_outcome(entity_id, positive)
        self.nucleus_accumbens.reinforce(
            ctx["state_key"], ctx["action_key"], 1.0 if positive else -1.0
        )
        self.neurorhythms.register_outcome(token_id, positive)

        # Slow mood / anti-reward / stress + neurotrophic protection close here.
        self.serotonin.reward(positive)
        if positive:
            # Relieve any accumulated disappointment for this (state, action)
            # and grow trophic support for the action that worked.
            self.habenula.relieve(ctx["state_key"], ctx["action_key"])
            self.bdnf.reinforce(token_id)
        else:
            # Systematic rejection: habenula suppresses it next time, cortisol
            # rises (the home is pushing back).
            self.habenula.punish(ctx["state_key"], ctx["action_key"])
            self.cortisol.stress_event()

        if ctx["rule_key"] is not None:
            self.cerebellum.record_outcome(ctx["rule_key"], positive)
            if positive:
                rule = self.cerebellum.rules.get(ctx["rule_key"])
                if rule is not None:
                    # A proven reflex: protect its target from blanket forgetting.
                    self.bdnf.reinforce(rule.target)
        self.anterior_cingulate.observe_outcome(positive)

        self._last_decision_ctx = None
        return True

    # ------------------------------------------------------------------
    # Ranking / conflict helpers
    # ------------------------------------------------------------------
    def _rank_predictions(self, predictions: List[Any], bucket: int, room: str,
                          timestamp=None) -> List[Any]:
        """Re-rank predictions by Go/NoGo priority, reward bias, arousal,
        cognitive control and next-room anticipation.

        * Basal ganglia Q-priority pushes habits up / aversions down.
        * Nucleus accumbens adds a habit-bias boost.
        * Locus Coeruleus arousal makes the system more reactive when busy.
        * ACC cognitive_control damps confidence by up to 25% when modules
          disagree or outcomes have been going wrong (closed control loop).
        * Cortisol damps confidence by up to 30% when the home has been under
          sustained stress (chaotic / unpredictable spells).
        * Lateral habenula suppresses candidates the user has repeatedly
          rejected in this context (anti-reward → stop nagging).
        * Entorhinal anticipation pre-activates tokens in the expected room.
        """
        mode = self.insula.current_mode
        arousal = self.locus_coeruleus.get_arousal()
        # Hour from the EVENT timestamp, not wall clock, so replayed / seeded
        # streams key reward bias by the event's hour (else it lands in the
        # wrong accumbens bucket during backfill and never matches at inference).
        hour = (timestamp or datetime.now(timezone.utc)).hour
        state_key = f"{room}|{mode}|{hour}"
        arousal_boost = (arousal - 0.3) * 0.15  # -0.045 .. +0.105
        control = max(0.0, min(1.0, getattr(self.anterior_cingulate, "cognitive_control", 0.0)))
        control_damping = 1.0 - 0.25 * control
        # Cortisol: global conservatism under sustained stress (1.0 at baseline).
        cortisol_damping = self.cortisol.damping()
        expected_room = self._expected_next_room

        ranked = []
        for prediction in predictions:
            token_id, prob, conf, source = prediction[:4]
            n_obs = prediction[4] if len(prediction) > 4 else 0
            priority = self.basal_ganglia.get_action_priority(token_id, bucket)
            action_key = self.thalamus.decode_token(token_id)
            reward_boost = self.nucleus_accumbens.get_bias(state_key, action_key) * 0.1
            anticipation_boost = (
                0.05 if (expected_room and action_key.split(".")[0] == expected_room) else 0.0
            )
            # Lateral habenula: down-weight chronically-rejected (state, action)s.
            suppression = self.habenula.get_suppression(state_key, action_key)
            bg_conf = (
                conf + priority * 0.1 + reward_boost + arousal_boost + anticipation_boost
            ) * control_damping * cortisol_damping * (1.0 - 0.6 * suppression)
            bg_conf = max(0.05, min(1.0, bg_conf))
            ranked.append((token_id, prob, bg_conf, source, n_obs))

        ranked.sort(key=lambda x: x[1] * x[2], reverse=True)
        return ranked

    def _build_acc_proposals(self, raw_predictions, ranked_predictions, fired_rule, decision):
        """Assemble per-module votes for the ACC conflict monitor.

        Each source votes for its own top candidate; conflict can only be
        measured from >= 2 votes. The basal-ganglia vote is only counted when
        re-ranking actually flipped the order (not when it merely confirms the
        reflex), and the amygdala adds a veto vote when the decision is risky.
        """
        proposals = []
        raw_top = raw_predictions[0] if raw_predictions else None
        if raw_top:
            proposals.append({
                "source": "hippocampus",
                "action": self.thalamus.decode_token(raw_top[0]),
                "confidence": raw_top[2],
            })
        reflex_target = fired_rule.target if fired_rule is not None else None
        if fired_rule is not None:
            proposals.append({
                "source": "cerebellum",
                "action": self.thalamus.decode_token(fired_rule.target),
                "confidence": fired_rule.confidence,
            })
        ranked_top = ranked_predictions[0] if ranked_predictions else None
        if (ranked_top and raw_top and ranked_top[0] != raw_top[0]
                and ranked_top[0] != reflex_target):
            proposals.append({
                "source": "basal_ganglia",
                "action": self.thalamus.decode_token(ranked_top[0]),
                "confidence": ranked_top[2],
            })
        if decision is not None and getattr(decision, "risk", 0.0) > 0.5:
            proposals.append({
                "source": "amygdala",
                "action": "veto",
                "confidence": decision.risk,
                "veto": True,
            })
        return proposals

    def _remember_decision(self, decision, fired_rule_key, bucket: int, room: str,
                           timestamp=None) -> None:
        """Snapshot the active decision so feedback() can reinforce it."""
        if decision is None or not getattr(decision, "token", ""):
            self._last_decision_ctx = None
            return
        # Event-timestamp hour (see _rank_predictions) so the remembered
        # state_key matches the one used at ranking time, incl. during replay.
        hour = (timestamp or datetime.now(timezone.utc)).hour
        self._last_decision_ctx = {
            "token_id": decision.token_id,
            "token": decision.token,
            "entity_id": decision.entity_id,
            "bucket": bucket,
            "action_key": self.thalamus.decode_token(decision.token_id),
            "state_key": f"{room}|{self.insula.current_mode}|{hour}",
            "rule_key": fired_rule_key,
        }

    def _maybe_maintain(self, timestamp, prev_event_ts: float = 0.0) -> None:
        """Coarse-cadence background maintenance run inline on the event path."""
        if self.tick_count % COMPILE_EVERY == 0:
            self.cerebellum.compile_rules(self.hippocampus)
        now = time.time()
        if now - self.entorhinal_cortex.last_prune_ts > ENTORHINAL_PRUNE_SECONDS:
            self.entorhinal_cortex.prune_old_transitions()
        # Sleep consolidation: during a quiet spell (≥30 min no events, ≥50
        # events since last, ≤1×/h) replay/prune memory, dream-recombine,
        # smooth Q-values and run synaptic homeostasis. Off in busy periods and
        # in tight replays (wall-clock quiet check), so it never fights live use.
        if self.sleep_consolidation.should_consolidate(prev_event_ts):
            self.sleep_consolidation.consolidate(
                self.hippocampus, self.cerebellum,
                self.basal_ganglia, self.neurorhythms,
                bdnf=self.bdnf,
            )

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def _learning_state(self) -> str:
        n = self.hippocampus.total_events
        acc = self.hippocampus.accuracy
        if n < 100:
            return "cold_start"
        if n < 1000 or acc < 0.3:
            return "learning"
        return "stable"

    def _build_extra(self, raw_predictions, fired_rule, decision,
                     anomaly_threshold, prev_event_ts,
                     stn_brake: float = 0.0, stn_hold: bool = False) -> Dict[str, Any]:
        """Surface the rich module outputs that used to be discarded."""
        extra: Dict[str, Any] = {
            "anomaly_threshold": round(anomaly_threshold, 3),
            "arousal": round(self.locus_coeruleus.get_arousal(), 3),
            "cognitive_control": round(self.anterior_cingulate.cognitive_control, 3),
            "conflict_level": round(self.anterior_cingulate.conflict_level, 3),
            "dopamine": round(self.basal_ganglia.dopamine_signal, 3),
            "expected_next_room": self._expected_next_room,
            "raw_prediction_count": len(raw_predictions or []),
            "should_consolidate": self.sleep_consolidation.should_consolidate(prev_event_ts),
            # Extended neuromodulator / region telemetry (all 0-1 scales):
            "cortisol": round(self.cortisol.level, 3),
            "serotonin": round(self.serotonin.level, 3),
            "acetylcholine": round(self.acetylcholine.mean_expected(), 3),
            "scn_gain": round(self._last_scn_gain, 3),
            "stn_brake": round(stn_brake, 3),
            "stn_hold": bool(stn_hold),
            "habenula_active": self.habenula.active_count(),
            "bdnf_protected": self.bdnf.protected_count(),
        }
        if fired_rule is not None:
            extra["reflex"] = {
                "trigger": fired_rule.trigger,
                "target": fired_rule.target,
                "confidence": round(fired_rule.confidence, 3),
                "ngram_order": fired_rule.ngram_order,
            }
        if decision is not None:
            extra["decision"] = {
                "token": decision.token,
                "entity_id": decision.entity_id,
                "stage": decision.stage,
                "confidence": round(decision.confidence, 3),
                "utility": round(decision.utility, 3),
                "risk": round(decision.risk, 3),
                "source": decision.source,
                "n_obs": decision.n_obs,
                "reasons": decision.reasons,
            }
        return extra

    def _snapshot(
        self,
        surprise: float = 0.0,
        anomaly: bool = False,
        token_id: Optional[int] = None,
        token: Optional[str] = None,
        predictions: Optional[List[Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> EngineSnapshot:
        return EngineSnapshot(
            surprise=float(surprise),
            anomaly=bool(anomaly),
            learning_state=self._learning_state(),
            tick_count=self.tick_count,
            token_id=token_id,
            token=token,
            predictions=predictions or [],
            extra=extra or {},
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    # Bumped whenever the on-disk layout of to_dict() changes incompatibly.
    # A restore from a *newer* schema than this build understands is refused
    # (cold start) instead of silently loading a half-understood brain; an
    # older/absent version is accepted and best-effort migrated forward.
    SCHEMA_VERSION = 1

    # Modules whose state is round-tripped by to_dict/from_dict.
    _PERSISTED_MODULES = (
        "thalamus", "hippocampus", "predictive", "cerebellum", "basal_ganglia",
        "neurorhythms", "sleep_consolidation", "amygdala", "insula",
        "hypothalamus", "spatial_cortex", "prefrontal_cortex",
        "anterior_cingulate", "entorhinal_cortex", "locus_coeruleus",
        "nucleus_accumbens", "reticular", "metaplasticity",
        # Extended region / neuromodulator set (additive: older brains that
        # lack these keys simply restore the modules at their init defaults,
        # so SCHEMA_VERSION stays 1 — the layout grew, it didn't change shape).
        "habenula", "subthalamic", "suprachiasmatic", "serotonin",
        "acetylcholine", "cortisol", "bdnf",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full engine state (all stateful modules + wiring)."""
        modules = {}
        for name in self._PERSISTED_MODULES:
            mod = getattr(self, name)
            to_dict = getattr(mod, "to_dict", None)
            if callable(to_dict):
                modules[name] = to_dict()
        return {
            "schema_version": self.SCHEMA_VERSION,
            "tick_count": self.tick_count,
            "last_room": self._last_room,
            "expected_next_room": self._expected_next_room,
            "modules": modules,
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Restore engine state previously produced by :meth:`to_dict`.

        A brain written by a *newer* schema than this build understands is
        rejected (the engine cold-starts) rather than restored half-blind,
        which could corrupt learning. Brains with an older or missing
        ``schema_version`` (e.g. produced by kontinuum-core <= 0.1.2) are
        treated as schema 1 and loaded as before.
        """
        # ``0`` covers the pre-versioning 0.1.2 layout, which is schema 1.
        version = data.get("schema_version", self.SCHEMA_VERSION)
        if version > self.SCHEMA_VERSION:
            logger.warning(
                "Refusing to restore brain: on-disk schema_version %s is newer "
                "than this engine understands (%s). Cold-starting instead.",
                version, self.SCHEMA_VERSION,
            )
            return
        self.tick_count = data.get("tick_count", 0)
        self._last_room = data.get("last_room")
        self._expected_next_room = data.get("expected_next_room")
        modules = data.get("modules", {})
        for name in self._PERSISTED_MODULES:
            if name not in modules:
                continue
            mod = getattr(self, name)
            from_dict = getattr(mod, "from_dict", None)
            if callable(from_dict):
                from_dict(modules[name])
