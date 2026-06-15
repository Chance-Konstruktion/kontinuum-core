"""Tests for the extended region / neuromodulator set (v0.5.0).

Covers the seven modules added on top of the original wiring:

Regions:        LateralHabenula, SubthalamicNucleus, SuprachiasmaticNucleus
Neuromodulators: Serotonin, Acetylcholine, Cortisol
Maintenance:    Bdnf (use-dependent protection)

Each module is unit-tested in isolation, then the engine wiring is checked
end-to-end: the snapshot surfaces the new signals, feedback closes the new
loops, the habenula/cortisol actually re-rank, the STN can brake an actionable
decision, BDNF shields a proven pattern from consolidation pruning, and the
whole extended state round-trips through to_dict/from_dict.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from kontinuum_core import KontinuumEngine
from kontinuum_core.acetylcholine import Acetylcholine
from kontinuum_core.bdnf import Bdnf
from kontinuum_core.cortisol import Cortisol
from kontinuum_core.habenula import LateralHabenula
from kontinuum_core.prefrontal_cortex import Decision
from kontinuum_core.serotonin import Serotonin
from kontinuum_core.sleep_consolidation import SleepConsolidation
from kontinuum_core.subthalamic_nucleus import SubthalamicNucleus
from kontinuum_core.suprachiasmatic import SuprachiasmaticNucleus


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _engine() -> KontinuumEngine:
    e = KontinuumEngine()
    e.register_entity("switch.kitchen", ha_area="kitchen", domain="switch")
    e.register_entity("light.bedroom_lamp", ha_area="bedroom", domain="light")
    return e


def _drive(e: KontinuumEngine, cycles: int):
    """Alternating two-entity loop (dodges the reticular burst filter)."""
    base = datetime(2026, 6, 13, 19, 0, 0, tzinfo=timezone.utc)
    steps = [
        ("switch.kitchen", "on"),
        ("light.bedroom_lamp", "on"),
        ("switch.kitchen", "off"),
        ("light.bedroom_lamp", "off"),
    ]
    snaps, n = [], 0
    for _ in range(cycles):
        for entity_id, state in steps:
            snaps.append(e.observe({
                "entity_id": entity_id, "new_state": state,
                "timestamp": base + timedelta(minutes=n),
            }))
            n += 1
    return [s for s in snaps if "skipped" not in s.extra]


# ---------------------------------------------------------------------------
# LateralHabenula
# ---------------------------------------------------------------------------

def test_habenula_punish_and_relieve():
    h = LateralHabenula()
    assert h.get_suppression("s", "a") == 0.0
    for _ in range(10):
        h.punish("s", "a")
    sup = h.get_suppression("s", "a")
    assert 0.5 < sup <= 1.0
    assert h.active_count() == 1
    for _ in range(40):
        h.relieve("s", "a")
    # Fully relieved entries are dropped.
    assert h.get_suppression("s", "a") == 0.0
    assert ("s", "a") not in h.disappointment


def test_habenula_round_trip():
    h = LateralHabenula()
    h.punish("room|active|7", "light.on", strength=1.0)
    h2 = LateralHabenula()
    h2.from_dict(h.to_dict())
    assert h2.get_suppression("room|active|7", "light.on") == \
        h.get_suppression("room|active|7", "light.on")


# ---------------------------------------------------------------------------
# SubthalamicNucleus
# ---------------------------------------------------------------------------

def test_stn_brake_monotonic():
    stn = SubthalamicNucleus()
    decisive = stn.compute_brake(conflict_level=0.0, top_conf=0.9, runner_up_conf=0.2)
    conflicted = stn.compute_brake(conflict_level=0.9, top_conf=0.5, runner_up_conf=0.49)
    assert conflicted > decisive
    assert 0.0 <= decisive <= 1.0 and 0.0 <= conflicted <= 1.0


def test_stn_patience_lowers_hold_threshold():
    stn = SubthalamicNucleus()
    # Tune the brake (~0.55) to sit between the patient/impatient hold thresholds
    # (0.35 at patience=1.0, 0.75 at patience=0.0).
    stn.compute_brake(conflict_level=0.5, top_conf=0.6, runner_up_conf=0.49)
    assert 0.35 < stn.brake < 0.75
    # Patient system holds; impatient one acts on the same brake.
    assert stn.should_hold(patience=1.0) is True
    assert stn.should_hold(patience=0.0) is False
    assert stn.total_holds == 1  # only the holding call counted


# ---------------------------------------------------------------------------
# SuprachiasmaticNucleus
# ---------------------------------------------------------------------------

def test_scn_neutral_until_warmup_then_entrains():
    scn = SuprachiasmaticNucleus()
    for _ in range(SuprachiasmaticNucleus.WARMUP - 1):
        scn.observe(10)
    assert scn.phase_gain(10) == 1.0   # still warming up → neutral
    for _ in range(80):
        scn.observe(10)
    assert scn.total >= SuprachiasmaticNucleus.WARMUP
    assert scn.phase_gain(10) > 1.05    # this home is active at 10:00
    assert scn.phase_gain(3) < 0.95     # ...and quiet at 03:00
    assert scn.peak_hour() == 10


# ---------------------------------------------------------------------------
# Serotonin
# ---------------------------------------------------------------------------

def test_serotonin_tracks_mood():
    s = Serotonin()
    assert s.get_patience() == Serotonin.BASELINE
    for _ in range(60):
        s.reward(True)
    assert s.get_patience() > 0.6
    for _ in range(120):
        s.reward(False)
    assert s.get_patience() < 0.35


# ---------------------------------------------------------------------------
# Acetylcholine
# ---------------------------------------------------------------------------

def test_acetylcholine_damps_learning_in_noisy_bucket():
    ach = Acetylcholine()
    # Fresh bucket → neutral until MIN_SAMPLES observations.
    ach.observe(7, 0.8)
    assert ach.learn_gain(7) == 1.0
    for _ in range(Acetylcholine.MIN_SAMPLES):
        ach.observe(7, 0.8)         # reliably noisy context
    assert ach.learn_gain(7) < 0.8  # learning is reduced where noise is the norm
    # A clean (low-surprise) bucket stays near full learning.
    for _ in range(Acetylcholine.MIN_SAMPLES):
        ach.observe(9, 0.05)
    assert ach.learn_gain(9) > 0.95


def test_acetylcholine_round_trip():
    ach = Acetylcholine()
    for _ in range(25):
        ach.observe(3, 0.4)
    ach2 = Acetylcholine()
    ach2.from_dict(ach.to_dict())
    assert ach2.learn_gain(3) == ach.learn_gain(3)
    assert ach2.counts[3] == ach.counts[3]


# ---------------------------------------------------------------------------
# Cortisol
# ---------------------------------------------------------------------------

def test_cortisol_rises_then_relaxes():
    c = Cortisol()
    assert c.damping() == 1.0           # baseline → no conservatism
    for _ in range(30):
        c.observe(0.9, anomaly=True)
    assert c.level > 0.4
    assert c.damping() < 1.0            # stressed → more conservative ranking
    for _ in range(2000):
        c.observe(0.0, anomaly=False)
    assert abs(c.level - Cortisol.BASELINE) < 0.05  # relaxed back toward baseline


def test_cortisol_override_bump():
    c = Cortisol()
    before = c.level
    c.stress_event()
    assert c.level > before


# ---------------------------------------------------------------------------
# Bdnf
# ---------------------------------------------------------------------------

def test_bdnf_protects_then_decays():
    b = Bdnf()
    assert not b.is_protected(5)
    for _ in range(4):
        b.reinforce(5)
    assert b.is_protected(5)
    assert b.protected_count() == 1
    for _ in range(200):
        b.decay_all()
    assert not b.is_protected(5)
    assert b.protected_count() == 0


def test_bdnf_protection_floors_weak_transition_in_consolidation():
    """A BDNF-protected target survives consolidation pruning; an idle one not."""
    class _Cb:
        rules = {}
        def compile_rules(self, hippo):  # noqa: D401 - phase-2 no-op stub
            pass

    def _hippo():
        return type("H", (), {
            "transitions": {0: {(1,): {2: 0.4}}},  # 0.4/1.5 = 0.27 < 0.3 → pruned
            "totals": {0: {(1,): 0.4}},
            "durations": {},
        })()

    # Unprotected → the weak transition is forgotten.
    h1 = _hippo()
    SleepConsolidation().consolidate(h1, _Cb())
    assert (1,) not in h1.transitions[0]

    # Protected → it is floored (kept) instead of deleted.
    h2 = _hippo()
    b = Bdnf()
    for _ in range(5):
        b.reinforce(2)
    assert b.is_protected(2)
    SleepConsolidation().consolidate(h2, _Cb(), bdnf=b)
    assert h2.transitions[0][(1,)].get(2, 0.0) >= 0.5


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------

def test_engine_exposes_extended_modules():
    e = _engine()
    for attr in ("habenula", "subthalamic", "suprachiasmatic",
                 "serotonin", "acetylcholine", "cortisol", "bdnf"):
        assert hasattr(e, attr), f"engine missing {attr}"


def test_snapshot_surfaces_neuromodulator_signals():
    e = _engine()
    processed = _drive(e, cycles=30)
    assert processed
    last = processed[-1]
    for key in ("cortisol", "serotonin", "acetylcholine", "scn_gain",
                "stn_brake", "stn_hold", "habenula_active", "bdnf_protected"):
        assert key in last.extra, f"missing extra key: {key}"
    assert 0.0 <= last.extra["cortisol"] <= 1.0
    assert isinstance(last.extra["stn_hold"], bool)


def test_habenula_suppresses_chronically_rejected_action():
    e = _engine()
    e.observe({"entity_id": "light.bedroom_lamp", "new_state": "on"})
    tok = e.thalamus.token_to_id["bedroom.light.on"]
    pred = [(tok, 0.9, 0.8, "test", 20)]

    base = e._rank_predictions(list(pred), bucket=0, room="bedroom")[0][2]

    mode = e.insula.current_mode
    hour = datetime.now(timezone.utc).hour
    state_key = f"bedroom|{mode}|{hour}"
    action_key = e.thalamus.decode_token(tok)
    for _ in range(10):
        e.habenula.punish(state_key, action_key)

    suppressed = e._rank_predictions(list(pred), bucket=0, room="bedroom")[0][2]
    assert suppressed < base, "habenula did not suppress a rejected action"


def test_cortisol_damps_ranking_under_stress():
    e = _engine()
    e.observe({"entity_id": "light.bedroom_lamp", "new_state": "on"})
    tok = e.thalamus.token_to_id["bedroom.light.on"]
    pred = [(tok, 0.9, 0.8, "test", 20)]

    e.cortisol.level = e.cortisol.BASELINE
    calm = e._rank_predictions(list(pred), bucket=0, room="bedroom")[0][2]
    e.cortisol.level = 1.0
    stressed = e._rank_predictions(list(pred), bucket=0, room="bedroom")[0][2]
    assert stressed < calm, "cortisol did not make ranking more conservative"


def test_stn_downgrades_actionable_decision(monkeypatch):
    """Under a (forced) hold the engine turns an actionable stage into OBSERVE."""
    e = _engine()
    t0 = datetime(2026, 6, 13, 19, 0, 0, tzinfo=timezone.utc)
    e.observe({"entity_id": "light.bedroom_lamp", "new_state": "on", "timestamp": t0})
    tok = e.thalamus.token_to_id["bedroom.light.on"]

    d = Decision()
    d.token = "bedroom.light.on"
    d.token_id = tok
    d.entity_id = "light.bedroom_lamp"
    d.stage = Decision.EXECUTE
    d.confidence = 0.9
    d.n_obs = 40
    monkeypatch.setattr(e.prefrontal_cortex, "evaluate", lambda *a, **k: d)
    monkeypatch.setattr(e.subthalamic, "should_hold", lambda *a, **k: True)

    snap = e.observe({
        "entity_id": "light.bedroom_lamp", "new_state": "off",
        "timestamp": t0 + timedelta(hours=1),
    })
    dec = snap.extra.get("decision")
    assert dec is not None
    assert dec["stage"] == "OBSERVE"
    assert any("STN-Hold" in r for r in dec["reasons"])
    assert snap.extra["stn_hold"] is True


def test_feedback_closes_extended_loops():
    e = _engine()
    _drive(e, cycles=30)
    # Remember a fresh decision to reinforce.
    e.observe({
        "entity_id": "switch.kitchen", "new_state": "on",
        "timestamp": datetime(2026, 6, 14, 7, 0, 0, tzinfo=timezone.utc),
    })
    assert e._last_decision_ctx is not None

    cortisol_before = e.cortisol.level
    serotonin_before = e.serotonin.level
    assert e.feedback(False) is True
    # A rejection raises stress and records disappointment.
    assert e.cortisol.level > cortisol_before
    assert e.serotonin.level < serotonin_before
    assert len(e.habenula.disappointment) >= 1


def test_extended_state_round_trips():
    e = _engine()
    e.cortisol.level = 0.6
    e.serotonin.level = 0.7
    e.habenula.punish("s", "a")
    e.bdnf.reinforce(5)
    e.bdnf.reinforce(5)
    e.acetylcholine.observe(3, 0.4)
    for _ in range(5):
        e.suprachiasmatic.observe(9)
    e.subthalamic.total_holds = 4

    e2 = KontinuumEngine()
    e2.from_dict(e.to_dict())

    assert e2.cortisol.level == 0.6
    assert e2.serotonin.level == 0.7
    assert e2.habenula.get_suppression("s", "a") == e.habenula.get_suppression("s", "a")
    assert e2.bdnf.protection(5) == e.bdnf.protection(5)
    assert e2.acetylcholine.expected[3] == e.acetylcholine.expected[3]
    assert e2.suprachiasmatic.total == e.suprachiasmatic.total
    assert e2.subthalamic.total_holds == 4


def test_old_brain_without_new_modules_still_loads():
    """Forward/backward compatibility: a brain missing the new keys cold-starts
    those modules at defaults instead of failing (SCHEMA_VERSION stays 1)."""
    e = _engine()
    _drive(e, cycles=10)
    blob = e.to_dict()
    # Simulate a pre-0.5.0 brain: drop every extended module key.
    for name in ("habenula", "subthalamic", "suprachiasmatic", "serotonin",
                 "acetylcholine", "cortisol", "bdnf"):
        blob["modules"].pop(name, None)

    e2 = KontinuumEngine()
    e2.from_dict(blob)   # must not raise
    assert e2.cortisol.level == Cortisol.BASELINE
    assert e2.serotonin.level == Serotonin.BASELINE
