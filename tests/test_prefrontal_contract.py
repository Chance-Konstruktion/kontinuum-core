"""Cross-module contracts used by the Home Assistant adapters."""

from kontinuum_core import KontinuumEngine
from kontinuum_core.prefrontal_cortex import Decision, PrefrontalCortex


def _decision(token: str, entity_id: str, token_id: int = 7) -> Decision:
    decision = Decision()
    decision.token = token
    decision.entity_id = entity_id
    decision.token_id = token_id
    decision.stage = Decision.CONFIRM
    return decision


def test_media_semantics_map_to_home_assistant_media_player_domain():
    pfc = PrefrontalCortex(amygdala=None)
    for state, service in (
        ("playing", "media_play"),
        ("paused", "media_pause"),
        ("off", "turn_off"),
    ):
        decision = _decision(f"living_room.media.{state}", "media_player.tv")
        call = pfc.get_service_call(decision)
        assert call == {
            "domain": "media_player",
            "service": service,
            "entity_id": "media_player.tv",
            "data": {"entity_id": "media_player.tv"},
        }


def test_non_media_semantics_keep_their_home_assistant_domain():
    pfc = PrefrontalCortex(amygdala=None)
    decision = _decision("kitchen.light.on", "light.kitchen")
    call = pfc.get_service_call(decision)
    assert call["domain"] == "light"
    assert call["service"] == "turn_on"


def test_confirmation_ids_do_not_collide_for_same_second_and_token():
    pfc = PrefrontalCortex(amygdala=None)
    first = _decision("living_room.light.on", "light.ceiling", token_id=7)
    second = _decision("living_room.light.on", "light.lamp", token_id=7)

    first_id = pfc.queue_confirm(first)
    second_id = pfc.queue_confirm(second)

    assert first_id != second_id
    assert pfc.peek_pending_confirm(first_id)["decision"] is first
    assert pfc.peek_pending_confirm(second_id)["decision"] is second


def test_decision_dict_preserves_token_id():
    decision = _decision("kitchen.light.on", "light.kitchen", token_id=42)
    assert decision.to_dict()["token_id"] == 42


def test_engine_snapshot_decision_preserves_token_id():
    engine = KontinuumEngine()
    decision = _decision("kitchen.light.on", "light.kitchen", token_id=42)

    extra = engine._build_extra([], None, decision, 0.2, 0.0)

    assert extra["decision"]["token_id"] == 42
