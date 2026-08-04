"""Tracking setpoint control: delta adaptation, refresh cadence, fallbacks."""

from custom_components.adaptive_comfort.core import controller, power
from custom_components.adaptive_comfort.core.types import (
    MODE_COOL,
    MODE_HEAT,
    ControllerState,
    HouseSnapshot,
    Settings,
    ZoneSnapshot,
)

NOW = 1_000_000.0


def make_zone(zone_id, temp, is_on=False, **kw):
    return ZoneSnapshot(
        zone_id=zone_id,
        name=zone_id,
        n_rooms=1,
        temp=temp,
        is_on=is_on,
        free_float=tuple([temp] * 24),
        confidence=0.9,
        **kw,
    )


def make_snapshot(zones, settings=None, now=NOW, **kw):
    return HouseSnapshot(
        now_ts=now,
        local_hour=12.0,
        settings=settings or Settings(hvac_mode=MODE_COOL, target=23.0),
        zones=zones,
        t_out=30.0,
        **kw,
    )


def warmed_state(zones, now=NOW):
    state = ControllerState()
    for zone in zones:
        state.zone_since[zone.zone_id] = now - 7200.0
        state.zone_on[zone.zone_id] = zone.is_on
    state.mode_since = now - 24 * 3600.0
    return state


def tick(zones, state, now=NOW, settings=None):
    snap = make_snapshot(zones, settings=settings, now=now)
    return controller.tick(snap, state)


def find_cmd(decision, zid):
    return next((c for c in decision.commands if c.zone_id == zid), None)


def test_demand_command_carries_track_delta():
    zone = make_zone("z1", 26.0)  # hot room, out of band -> demand
    state = warmed_state([zone])
    decision = tick([zone], state)
    cmd = find_cmd(decision, "z1")
    assert cmd is not None and cmd.hvac_mode == MODE_COOL
    assert cmd.track_delta is not None
    assert controller.TRACK_DELTA_MIN_K <= cmd.track_delta <= controller.TRACK_DELTA_MAX_K


def test_delta_deepens_out_of_band_and_relaxes_past_target():
    zone = make_zone("z1", 26.0)
    state = warmed_state([zone])
    tick([zone], state)
    deepened = state.zone_track_delta["z1"]

    # Same zone now cooled past the room-frame setpoint: delta must shrink.
    # Keep zone chatter blocking the off so the head stays in the tracking loop
    # (plant-level min_on is separate; this is the in-run adaptation path).
    cold = make_zone("z1", 22.0, is_on=True)
    later = NOW + 600.0
    state.zone_on["z1"] = True
    state.zone_since["z1"] = later - 60.0
    state.zone_last_cmd["z1"] = later - 3600.0  # spacing satisfied
    tick([cold], state, now=later)
    relaxed = state.zone_track_delta["z1"]
    assert relaxed < deepened
    assert relaxed >= controller.TRACK_DELTA_MIN_K


def test_delta_clamped_to_bounds():
    zone = make_zone("z1", 40.0)  # absurdly hot: repeated deepening
    state = warmed_state([zone])
    for i in range(10):
        state.zone_last_cmd["z1"] = NOW + i * 600.0 - 3600.0
        zone = make_zone("z1", 40.0, is_on=True)
        state.zone_on["z1"] = True
        tick([zone], state, now=NOW + i * 600.0)
    assert state.zone_track_delta["z1"] <= controller.TRACK_DELTA_MAX_K


def test_running_zone_gets_refresh_commands_at_spacing():
    zone = make_zone("z1", 26.0, is_on=True)
    state = warmed_state([zone])
    state.zone_last_cmd["z1"] = NOW - controller.COMMAND_SPACING_S - 1.0
    state.zone_last_setpoint["z1"] = 23.35  # matches band center: no setpoint change
    decision = tick([zone], state)
    cmd = find_cmd(decision, "z1")
    assert cmd is not None, "tracking must refresh even without a setpoint change"
    assert cmd.track_delta is not None


def test_head_anchor_drift_refreshes_before_spacing():
    """Inflated entry internal→SP must re-anchor when the head sensor collapses."""
    # Cool chase depth -0.7 → ideal SP = internal - 0.7.
    # Device still at 26.3 from a 27°C entry; live internal is now 24.
    zone = make_zone(
        "z1",
        26.0,
        is_on=True,
        head_internal_temp=24.0,
        device_setpoint=26.3,
    )
    state = warmed_state([zone])
    state.zone_on["z1"] = True
    state.zone_track_delta["z1"] = 0.7
    state.zone_last_setpoint["z1"] = 23.35
    # Spacing not elapsed, but re-anchor min has.
    state.zone_last_cmd["z1"] = NOW - controller.HEAD_REANCHOR_MIN_S - 1.0
    decision = tick([zone], state)
    cmd = find_cmd(decision, "z1")
    assert cmd is not None, "head-anchor drift must refresh before COMMAND_SPACING_S"
    assert cmd.head_depth_k is not None or cmd.track_delta is not None


def test_head_anchor_stable_waits_for_spacing():
    # OOB deepen 0.7→1.2 this tick; device SP already matches the post-adapt depth.
    zone = make_zone(
        "z1",
        26.0,
        is_on=True,
        head_internal_temp=24.0,
        device_setpoint=24.0 - 1.2,
    )
    state = warmed_state([zone])
    state.zone_on["z1"] = True
    state.zone_track_delta["z1"] = 0.7
    state.zone_last_setpoint["z1"] = 23.35
    state.zone_last_cmd["z1"] = NOW - controller.HEAD_REANCHOR_MIN_S - 1.0
    decision = tick([zone], state)
    cmd = find_cmd(decision, "z1")
    assert cmd is None, "aligned head anchor must not spam before COMMAND_SPACING_S"


def test_ideal_device_setpoint_signed_depth():
    assert controller._ideal_device_setpoint(24.0, -0.7, MODE_COOL) == 23.3
    assert controller._ideal_device_setpoint(24.0, 2.0, MODE_COOL) == 26.0
    assert controller._ideal_device_setpoint(22.0, -0.5, MODE_HEAT) == 22.5
    assert controller._ideal_device_setpoint(22.0, 1.5, MODE_HEAT) == 20.5


def test_tracking_disabled_yields_no_delta_and_no_refresh():
    settings = Settings(hvac_mode=MODE_COOL, target=23.0, tracking=False)
    zone = make_zone("z1", 26.0, is_on=True)
    state = warmed_state([zone])
    state.zone_last_cmd["z1"] = NOW - controller.COMMAND_SPACING_S - 1.0
    state.zone_last_setpoint["z1"] = 23.35
    decision = tick([zone], state, settings=settings)
    cmd = find_cmd(decision, "z1")
    assert cmd is None or cmd.track_delta is None


def test_track_delta_state_round_trips():
    state = ControllerState()
    state.zone_track_delta["z1"] = 1.2
    restored = ControllerState.from_dict(state.to_dict())
    assert restored.zone_track_delta == {"z1": 1.2}


def test_outdoor_band_edges():
    assert power.outdoor_band(None) is None
    assert power.outdoor_band(18.0) == "mild"
    assert power.outdoor_band(25.0) == "warm"
    assert power.outdoor_band(29.9) == "warm"
    assert power.outdoor_band(30.0) == "hot"
