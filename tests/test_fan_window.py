"""Fan-assist (multi-split same-mode constraint) and window suggestions."""

from custom_components.adaptive_comfort.core import controller
from custom_components.adaptive_comfort.core.types import (
    MODE_AUTO,
    MODE_COOL,
    MODE_FAN,
    MODE_OFF,
    STATE_COOLING,
    ControllerState,
    HouseSnapshot,
    Settings,
    ZoneSnapshot,
)

NOW = 1_000_000.0


def make_zone(zone_id, temp, occupied=None, **kw):
    kw.setdefault("free_float", tuple([temp] * 24))
    kw.setdefault("confidence", 0.9)
    return ZoneSnapshot(
        zone_id=zone_id, name=zone_id, n_rooms=1, temp=temp, occupied=occupied, **kw
    )


def make_snapshot(zones, settings=None, **kw):
    return HouseSnapshot(
        now_ts=NOW,
        local_hour=12.0,
        settings=settings or Settings(hvac_mode=MODE_AUTO),
        zones=zones,
        **kw,
    )


def warmed_state(zones):
    state = ControllerState()
    for zone in zones:
        state.zone_since[zone.zone_id] = NOW - 7200.0
        state.zone_on[zone.zone_id] = zone.is_on
    state.mode_since = NOW - 24 * 3600.0
    return state


def test_fan_assist_for_opposite_demand_zone():
    # Dominant cooling; one occupied room is too cold. On a multi-split it
    # cannot heat, but a dry coil allows fan-only mixing.
    hot = make_zone("hot", 25.5)
    cold = make_zone("cold", 21.0, occupied=True)
    snap = make_snapshot([hot, cold])
    state = warmed_state([hot, cold])
    decision = controller.tick(snap, state)
    by_zone = {c.zone_id: c for c in decision.commands}
    assert by_zone["hot"].hvac_mode == MODE_COOL
    assert by_zone["cold"].hvac_mode == MODE_FAN
    assert by_zone["cold"].reason == "fan_assist"


def test_fan_assist_blocked_by_wet_coil():
    hot = make_zone("hot", 25.5)
    cold = make_zone("cold", 21.0, occupied=True)
    snap = make_snapshot([hot, cold])
    state = warmed_state([hot, cold])
    state.zone_last_cool["cold"] = NOW - 300.0  # cooled 5 min ago: coil wet
    decision = controller.tick(snap, state)
    by_zone = {c.zone_id: c for c in decision.commands}
    assert "cold" not in by_zone or by_zone["cold"].hvac_mode != MODE_FAN


def test_fan_assist_respects_option_and_topology():
    hot = make_zone("hot", 25.5)
    cold = make_zone("cold", 21.0, occupied=True)
    for kwargs in ({"fan_assist": False}, {"multisplit": False}):
        settings = Settings(hvac_mode=MODE_AUTO, **kwargs)
        snap = make_snapshot([hot, cold], settings)
        state = warmed_state([hot, cold])
        decision = controller.tick(snap, state)
        assert not any(c.hvac_mode == MODE_FAN for c in decision.commands)


def test_fan_stops_when_zone_recovers():
    hot = make_zone("hot", 25.5)
    cold = make_zone("cold", 22.4, occupied=True)  # back in band
    snap = make_snapshot([hot, cold])
    state = warmed_state([hot, cold])
    state.zone_fan["cold"] = True
    decision = controller.tick(snap, state)
    by_zone = {c.zone_id: c for c in decision.commands}
    assert by_zone["cold"].hvac_mode == MODE_OFF
    assert by_zone["cold"].reason == "fan_off"
    assert not state.zone_fan["cold"]


def test_wet_coil_tracked_from_head_state():
    zone = make_zone("bed", 23.0, head_state=STATE_COOLING, is_on=True)
    snap = make_snapshot([zone], Settings(hvac_mode=MODE_COOL))
    state = warmed_state([zone])
    controller.tick(snap, state)
    assert state.zone_last_cool["bed"] == NOW


def test_window_suggested_instead_of_cooling():
    settings = Settings(hvac_mode=MODE_AUTO, window_suggest=True)
    zone = make_zone("bed", 25.5, occupied=True)
    snap = make_snapshot([zone], settings, t_out=18.0)
    state = warmed_state([zone])
    decision = controller.tick(snap, state)
    assert decision.window_suggestions == ["bed"]
    assert not any(c.hvac_mode == MODE_COOL for c in decision.commands)


def test_window_grace_expires_then_cools():
    settings = Settings(hvac_mode=MODE_AUTO, window_suggest=True)
    zone = make_zone("bed", 25.5, occupied=True)
    snap = make_snapshot([zone], settings, t_out=18.0)
    state = warmed_state([zone])
    state.window_suggest_since["bed"] = NOW - 1000.0  # grace (900 s) expired
    decision = controller.tick(snap, state)
    by_zone = {c.zone_id: c for c in decision.commands}
    assert by_zone["bed"].hvac_mode == MODE_COOL
    assert decision.window_suggestions == ["bed"]  # advisory stays on


def test_window_not_suggested_without_presence():
    settings = Settings(hvac_mode=MODE_AUTO, window_suggest=True)
    zone = make_zone("bed", 25.5, occupied=None)
    snap = make_snapshot([zone], settings, t_out=18.0, house_occupied=None)
    state = warmed_state([zone])
    decision = controller.tick(snap, state)
    assert decision.window_suggestions == []
    assert any(c.hvac_mode == MODE_COOL for c in decision.commands)


def test_window_not_suggested_when_option_off_or_warm_outside():
    zone = make_zone("bed", 25.5, occupied=True)
    # Option off (default): cool immediately.
    snap = make_snapshot([zone], Settings(hvac_mode=MODE_AUTO), t_out=18.0)
    decision = controller.tick(snap, warmed_state([zone]))
    assert decision.window_suggestions == []
    assert any(c.hvac_mode == MODE_COOL for c in decision.commands)
    # Option on but outdoors not clearly colder.
    settings = Settings(hvac_mode=MODE_AUTO, window_suggest=True)
    snap = make_snapshot([zone], settings, t_out=24.5)
    decision = controller.tick(snap, warmed_state([zone]))
    assert decision.window_suggestions == []
