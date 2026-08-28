from custom_components.adaptive_comfort.core import controller
from custom_components.adaptive_comfort.core.types import (
    MODE_AUTO,
    MODE_COOL,
    MODE_HEAT,
    MODE_OFF,
    ControllerState,
    HouseSnapshot,
    Settings,
    ZoneSnapshot,
)

NOW = 1_000_000.0


def make_zone(zone_id, temp, occupied=None, free_float=None, confidence=0.9, **kw):
    if free_float is None:
        free_float = [temp] * 24
    return ZoneSnapshot(
        zone_id=zone_id,
        name=zone_id,
        n_rooms=1,
        temp=temp,
        occupied=occupied,
        free_float=tuple(free_float),
        confidence=confidence,
        **kw,
    )


def make_snapshot(zones, settings=None, now=NOW, local_hour=12.0, **kw):
    return HouseSnapshot(
        now_ts=now,
        local_hour=local_hour,
        settings=settings or Settings(hvac_mode=MODE_AUTO),
        zones=zones,
        **kw,
    )


def warmed_state(zones, now=NOW):
    """State whose min-off timers have long expired."""
    state = ControllerState()
    for zone in zones:
        state.zone_since[zone.zone_id] = now - 7200.0
        state.zone_on[zone.zone_id] = zone.is_on
    state.mode_since = now - 24 * 3600.0
    return state


def test_hot_zone_gets_cooled():
    zone = make_zone("bed", 25.0)
    snap = make_snapshot([zone])
    state = warmed_state([zone])
    decision = controller.tick(snap, state)
    commands = {c.zone_id: c for c in decision.commands}
    assert commands["bed"].hvac_mode == MODE_COOL
    # Warm-edge prior: hold near hi - BAND_HOLD_MARGIN_K (quantized 0.5 K).
    assert commands["bed"].setpoint == 23.0
    assert decision.diag["want"]["bed"] == "demand"


def test_away_demand_holds_near_band_edge():
    from custom_components.adaptive_comfort.core import comfort
    from custom_components.adaptive_comfort.core.types import PRESET_AWAY

    settings = Settings(hvac_mode=MODE_COOL, target=22.5, preset=PRESET_AWAY, adaptive_blend=0.0)
    # Far above away band so still in demand.
    zone = make_zone("bed", 30.0)
    snap = make_snapshot([zone], settings, house_occupied=False)
    state = warmed_state([zone])
    decision = controller.tick(snap, state)
    cmd = {c.zone_id: c for c in decision.commands}["bed"]
    center = comfort.band_center(settings, None)
    _lo, hi = comfort.zone_band(settings, center, True, False)
    assert abs(cmd.setpoint - (hi - comfort.BAND_HOLD_MARGIN_K)) < 0.26  # quantized


def test_manual_emits_no_commands_but_keeps_want():
    from custom_components.adaptive_comfort.core.types import PRESET_MANUAL

    settings = Settings(hvac_mode=MODE_AUTO, preset=PRESET_MANUAL)
    zone = make_zone("bed", 26.0, is_on=True)
    snap = make_snapshot([zone], settings)
    state = warmed_state([zone])
    state.mode = MODE_COOL
    state.zone_parked_since["bed"] = NOW
    decision = controller.tick(snap, state)
    assert decision.commands == []
    assert decision.diag["want"]["bed"] == "demand"
    assert "bed" not in state.zone_parked_since  # parks cleared


def test_manual_with_hvac_off_still_force_stops():
    from custom_components.adaptive_comfort.core.types import PRESET_MANUAL

    settings = Settings(hvac_mode=MODE_OFF, preset=PRESET_MANUAL)
    zone = make_zone("bed", 26.0, is_on=True)
    snap = make_snapshot([zone], settings)
    state = warmed_state([zone])
    decision = controller.tick(snap, state)
    assert any(c.hvac_mode == MODE_OFF for c in decision.commands)


def test_manual_does_not_shed():
    """Manual is full hands-off — including contracted-power shedding."""
    from custom_components.adaptive_comfort.core.types import PRESET_MANUAL

    settings = Settings(hvac_mode=MODE_AUTO, preset=PRESET_MANUAL, shedding_enabled=True)
    zone = make_zone("bed", 26.0, is_on=True, draw_w=1200.0)
    snap = make_snapshot([zone], settings, p_demand=3400.0, shed_urgent=True)
    state = warmed_state([zone])
    state.zone_on["bed"] = True
    decision = controller.tick(snap, state)
    assert decision.commands == []
    assert state.shed == {}


def test_manual_clears_latched_shed_and_fan_keeps_coil_clock():
    from custom_components.adaptive_comfort.core.types import PRESET_MANUAL, STATE_COOLING

    settings = Settings(hvac_mode=MODE_AUTO, preset=PRESET_MANUAL, shedding_enabled=True)
    zone = make_zone("bed", 26.0, is_on=True, head_state=STATE_COOLING)
    snap = make_snapshot([zone], settings, p_demand=2000.0)
    state = warmed_state([zone])
    state.shed["bed"] = NOW - 10.0
    state.zone_fan["bed"] = True
    state.last_shed_action = NOW - 5.0
    decision = controller.tick(snap, state)
    assert decision.commands == []
    assert state.shed == {}
    assert state.zone_fan == {}
    assert state.zone_last_cool.get("bed") == NOW


def test_manual_does_not_flip_mode_on_cooling_overshoot():
    """Manual pulldown below the band must not latch heat into MODE_DWELL_S."""
    from custom_components.adaptive_comfort.core.types import PRESET_MANUAL, STATE_COOLING

    settings = Settings(hvac_mode=MODE_AUTO, preset=PRESET_MANUAL)
    zone = make_zone(
        "bed",
        19.5,
        occupied=True,
        is_on=True,
        head_mode=MODE_COOL,
        head_state=STATE_COOLING,
    )
    snap = make_snapshot([zone], settings, t_rm=23.5)
    state = warmed_state([zone])
    state.mode = MODE_COOL
    decision = controller.tick(snap, state)
    assert decision.commands == []
    assert state.mode == MODE_COOL
    assert decision.diag.get("mode_source") == "manual"


def test_coordination_spreads_to_helper_zones():
    hot = make_zone("hot", 25.0)
    ok = make_zone("ok", 22.6)
    cold_edge = make_zone("edge", 21.7)  # at/below lower band: must not join
    snap = make_snapshot([hot, ok, cold_edge])
    state = warmed_state([hot, ok, cold_edge])
    decision = controller.tick(snap, state)
    by_zone = {c.zone_id: c for c in decision.commands}
    assert by_zone["hot"].hvac_mode == MODE_COOL
    assert by_zone["ok"].hvac_mode == MODE_COOL
    assert by_zone["ok"].reason == "helper"
    assert "edge" not in by_zone


def test_coordination_spreads_to_vacant_helpers():
    hot = make_zone("hot", 25.0)
    empty = make_zone("empty", 22.6, occupied=False)
    snap = make_snapshot([hot, empty])
    state = warmed_state([hot, empty])
    decision = controller.tick(snap, state)
    by_zone = {c.zone_id: c for c in decision.commands}
    assert by_zone["empty"].hvac_mode == MODE_COOL
    assert by_zone["empty"].reason == "helper"


def test_coordination_helper_below_center_still_joins():
    hot = make_zone("hot", 25.0)
    cool_side = make_zone("ok", 22.0)  # below center 22.5, above lo 21.8
    snap = make_snapshot([hot, cool_side])
    state = warmed_state([hot, cool_side])
    decision = controller.tick(snap, state)
    by_zone = {c.zone_id: c for c in decision.commands}
    assert by_zone["ok"].hvac_mode == MODE_COOL
    assert by_zone["ok"].reason == "helper"


def test_zone_presence_off_allows_vacant_helpers():
    settings = Settings(hvac_mode=MODE_AUTO, zone_presence_adaptation=False)
    hot = make_zone("hot", 25.0)
    empty = make_zone("empty", 22.6, occupied=False)
    snap = make_snapshot([hot, empty], settings)
    state = warmed_state([hot, empty])
    decision = controller.tick(snap, state)
    by_zone = {c.zone_id: c for c in decision.commands}
    assert "empty" in by_zone


def test_quiet_night_defers_in_band_sleep_room_to_other_zone():
    settings = Settings(hvac_mode=MODE_COOL, zone_quiet_night={"bed": True})
    bed = make_zone("bed", 22.8, occupied=True)
    east = make_zone("east", 22.6, occupied=False)
    snap = make_snapshot([bed, east], settings, local_hour=23.0)
    state = warmed_state([bed, east])
    decision = controller.tick(snap, state)
    by_zone = {c.zone_id: c for c in decision.commands}
    assert "bed" not in by_zone or by_zone["bed"].hvac_mode == MODE_OFF
    assert by_zone["east"].hvac_mode == MODE_COOL
    assert "bed" in decision.diag.get("quiet_night_deferred", [])


def test_quiet_night_ignored_during_day():
    settings = Settings(hvac_mode=MODE_COOL, zone_quiet_night={"bed": True})
    bed = make_zone("bed", 22.8, occupied=True)
    east = make_zone("east", 22.6, occupied=False)
    snap = make_snapshot([bed, east], settings, local_hour=14.0)
    state = warmed_state([bed, east])
    decision = controller.tick(snap, state)
    by_zone = {c.zone_id: c for c in decision.commands}
    assert "east" not in by_zone
    assert not decision.diag.get("quiet_night_deferred")
    assert not decision.diag.get("quiet_night_bank")


def test_quiet_night_banks_conditioning_hold_edge_before_night():
    settings = Settings(hvac_mode=MODE_COOL, zone_quiet_night={"bed": True})
    bed = make_zone("bed", 22.8, occupied=True)
    east = make_zone("east", 22.6, occupied=False)
    snap = make_snapshot([bed, east], settings, local_hour=20.5)
    state = warmed_state([bed, east])
    decision = controller.tick(snap, state)
    by_zone = {c.zone_id: c for c in decision.commands}
    assert by_zone["bed"].hvac_mode == MODE_COOL
    assert by_zone["bed"].setpoint is not None
    assert by_zone["bed"].setpoint < 22.5
    assert "bed" in decision.diag.get("quiet_night_bank", [])
    assert not decision.diag.get("quiet_night_deferred")


def test_quiet_night_defers_oob_sleeper_until_cover_spent():
    settings = Settings(hvac_mode=MODE_AUTO, zone_quiet_night={"bed": True})
    bed = make_zone("bed", 25.0, occupied=True)
    east = make_zone("east", 22.6, occupied=True)
    snap = make_snapshot([bed, east], settings, local_hour=23.0)
    state = warmed_state([bed, east])
    decision = controller.tick(snap, state)
    by_zone = {c.zone_id: c for c in decision.commands}
    assert "bed" not in by_zone or by_zone["bed"].hvac_mode == MODE_OFF
    assert by_zone["east"].hvac_mode == MODE_COOL
    assert "bed" in decision.diag.get("quiet_night_deferred", [])
    assert decision.diag.get("quiet_cover_extended") == "east"


def test_quiet_night_recruits_sleeper_after_extended_cover():
    settings = Settings(hvac_mode=MODE_COOL, zone_quiet_night={"bed": True})
    bed = make_zone("bed", 25.0, occupied=True)
    # Occupied lo ~ 21.8; extended cover lo = 21.8 - 1.5 = 20.3.
    east = make_zone("east", 20.3, occupied=True, is_on=True)
    snap = make_snapshot([bed, east], settings, local_hour=23.0)
    state = warmed_state([bed, east])
    state.zone_on["east"] = True
    decision = controller.tick(snap, state)
    by_zone = {c.zone_id: c for c in decision.commands}
    assert by_zone["bed"].hvac_mode == MODE_COOL
    assert "east" in by_zone
    assert "bed" not in decision.diag.get("quiet_night_deferred", [])


def test_quiet_night_defers_in_band_in_heat():
    settings = Settings(hvac_mode=MODE_HEAT, zone_quiet_night={"bed": True})
    bed = make_zone("bed", 22.0, occupied=True)
    east = make_zone("east", 21.5, occupied=False)
    snap = make_snapshot([bed, east], settings, local_hour=23.0)
    state = warmed_state([bed, east])
    decision = controller.tick(snap, state)
    by_zone = {c.zone_id: c for c in decision.commands}
    assert "bed" not in by_zone or by_zone["bed"].hvac_mode == MODE_OFF
    assert by_zone["east"].hvac_mode == MODE_HEAT
    assert "bed" in decision.diag.get("quiet_night_deferred", [])


def test_quiet_night_solo_zone_still_runs():
    settings = Settings(hvac_mode=MODE_AUTO, zone_quiet_night={"bed": True})
    bed = make_zone("bed", 25.0, occupied=True)
    snap = make_snapshot([bed], settings, local_hour=23.0)
    state = warmed_state([bed])
    decision = controller.tick(snap, state)
    by_zone = {c.zone_id: c for c in decision.commands}
    assert by_zone["bed"].hvac_mode == MODE_COOL


def test_coordination_disabled_runs_demand_only():
    settings = Settings(hvac_mode=MODE_AUTO, coordination=False)
    hot = make_zone("hot", 25.0)
    ok = make_zone("ok", 22.6)
    snap = make_snapshot([hot, ok], settings)
    state = warmed_state([hot, ok])
    decision = controller.tick(snap, state)
    by_zone = {c.zone_id: c for c in decision.commands}
    assert "hot" in by_zone
    assert "ok" not in by_zone


def test_cop_table_unbanded_does_not_consolidate():
    hot = make_zone("hot", 25.0)
    ok = make_zone("ok", 22.6)
    snap = make_snapshot([hot, ok], cop_by_head_count={1: 4.0, 2: 3.0})
    state = warmed_state([hot, ok])
    decision = controller.tick(snap, state)
    by_zone = {c.zone_id: c for c in decision.commands}
    assert by_zone["ok"].reason == "helper"
    assert not decision.diag.get("consolidated_by_cop_table")
    assert decision.diag.get("cop_n_source") == "prior"


def test_cop_table_banded_spread_when_more_heads_win():
    hot = make_zone("hot", 25.0)
    ok = make_zone("ok", 22.6)
    snap = make_snapshot([hot, ok], cop_by_head_count_banded={1: 1.2, 2: 2.8})
    state = warmed_state([hot, ok])
    decision = controller.tick(snap, state)
    by_zone = {c.zone_id: c for c in decision.commands}
    assert by_zone["ok"].reason == "helper"
    assert not decision.diag.get("consolidated_by_cop_table")
    assert decision.diag.get("cop_n_source") == "banded_spread"


def test_cop_table_prefers_banded_cross_n():
    hot = make_zone("hot", 25.0)
    ok = make_zone("ok", 22.6)
    snap = make_snapshot(
        [hot, ok],
        cop_by_head_count={1: 2.0, 2: 4.0},
        cop_by_head_count_banded={1: 4.0, 2: 3.0},
    )
    state = warmed_state([hot, ok])
    decision = controller.tick(snap, state)
    by_zone = {c.zone_id: c for c in decision.commands}
    assert "ok" not in by_zone
    assert decision.diag.get("consolidated_by_cop_table")


def test_min_off_blocks_restart():
    zone = make_zone("bed", 25.0)
    snap = make_snapshot([zone])
    state = ControllerState()
    state.zone_on["bed"] = False
    state.zone_since["bed"] = NOW - 120.0  # turned off 2 min ago, min_off=10
    decision = controller.tick(snap, state)
    assert not any(c.zone_id == "bed" and c.hvac_mode == MODE_COOL for c in decision.commands)


def test_plant_min_on_served_by_residual_park():
    # Satisfied zone would turn off, but plant compression still owes min_on → park.
    zone = make_zone(
        "bed",
        22.5,
        is_on=True,
        head_state="cooling",
        free_float=[22.5] * 24,
        park_residuals=True,
        park_extraction_w=120.0,
        standing_load_w=80.0,
        park_residual_edge_k=2.5,
    )
    settings = Settings(hvac_mode=MODE_COOL, min_on_min=20.0, park_learning=True, multisplit=True)
    snap = make_snapshot([zone], settings, p_ac=400.0, compression_floor_w=75.0)
    state = ControllerState()
    state.zone_on["bed"] = True
    state.zone_since["bed"] = NOW - 300.0  # zone chatter cleared; plant min_on has not
    state.plant_compress_since = NOW - 300.0
    decision = controller.tick(snap, state)
    assert not any(c.hvac_mode == MODE_OFF for c in decision.commands)
    assert (
        any((c.head_depth_k or 0) > 0 for c in decision.commands)
        or "bed" in decision.state.zone_parked_since
    )
    # After plant min_on elapses, hard off is allowed.
    state = decision.state
    state.zone_last_cmd["bed"] = 0.0
    state.plant_compress_since = NOW - 1300.0
    snap2 = make_snapshot([zone], settings, p_ac=40.0, compression_floor_w=75.0)
    # Clear park session to exercise the off path.
    state.zone_parked_since.clear()
    state.zone_park_margin.clear()
    decision = controller.tick(snap2, state)
    assert any(c.hvac_mode == MODE_OFF for c in decision.commands)


def test_prediction_cost_test_ignores_far_small_breach():
    # Room well inside band; free-float barely grazes hi at hour 6 — not worth a run.
    zone = make_zone(
        "bed",
        22.5,
        free_float=[22.5, 22.6, 22.7, 22.8, 22.9, 23.0, 23.35] + [23.0] * 17,
        pred_60m=22.6,
    )
    settings = Settings(hvac_mode=MODE_COOL, min_on_min=20.0)
    # band hi ≈ 23.2 with defaults; peak breach at h=6 is ~0.15 < PREDICT_MARGIN after hi
    snap = make_snapshot([zone], settings)
    state = warmed_state([zone])
    decision = controller.tick(snap, state)
    assert decision.diag["want"]["bed"] == "off"


def test_prediction_cost_test_fires_near_breach():
    zone = make_zone(
        "bed",
        23.0,
        free_float=[23.0, 23.5, 24.0] + [24.2] * 21,
        pred_60m=23.5,
    )
    settings = Settings(hvac_mode=MODE_COOL, min_on_min=20.0)
    snap = make_snapshot([zone], settings)
    state = warmed_state([zone])
    decision = controller.tick(snap, state)
    assert decision.diag["want"]["bed"] == "demand"


def test_plant_handoff_requires_half_min_on_predicted_need():
    zone = make_zone("bed", 22.5, free_float=[22.5] * 24, pred_60m=22.5)
    settings = Settings(hvac_mode=MODE_COOL, min_on_min=20.0)
    snap = make_snapshot([zone], settings, p_ac=400.0, compression_floor_w=75.0)
    state = warmed_state([zone])
    state.plant_compress_since = NOW - 60.0
    decision = controller.tick(snap, state)
    assert "plant_handoff" not in decision.diag
    assert decision.diag["want"]["bed"] == "off"


def test_forced_off_ignores_min_on():
    zone = make_zone("bed", 25.0, is_on=True)
    settings = Settings(hvac_mode=MODE_OFF)
    snap = make_snapshot([zone], settings)
    state = ControllerState()
    state.zone_on["bed"] = True
    state.zone_since["bed"] = NOW - 60.0
    decision = controller.tick(snap, state)
    assert any(c.zone_id == "bed" and c.hvac_mode == MODE_OFF for c in decision.commands)


def test_shedding_prefers_unoccupied_and_overrides_min_on():
    living = make_zone("living", 26.0, occupied=True, is_on=True)
    guest = make_zone("guest", 24.0, occupied=False, is_on=True)
    settings = Settings(hvac_mode=MODE_COOL)
    snap = make_snapshot(
        [living, guest],
        settings,
        p_demand=3400.0,
        p_grid=3400.0,
        p_grid_over_since=30.0,
    )
    state = warmed_state([living, guest])
    state.zone_on = {"living": True, "guest": True}
    state.zone_since = {"living": NOW - 60.0, "guest": NOW - 60.0}  # both just started
    decision = controller.tick(snap, state)
    off = [c.zone_id for c in decision.commands if c.hvac_mode == MODE_OFF]
    assert off == ["guest"]
    assert "guest" in state.shed


def test_urgent_shedding_drops_all_active_zones():
    living = make_zone("living", 26.0, occupied=True, is_on=True)
    guest = make_zone("guest", 24.0, occupied=False, is_on=True)
    settings = Settings(hvac_mode=MODE_COOL)
    snap = make_snapshot(
        [living, guest],
        settings,
        p_demand=3350.0,
        p_grid=600.0,
        shed_urgent=True,
    )
    state = warmed_state([living, guest])
    state.zone_on = {"living": True, "guest": True}
    decision = controller.tick(snap, state)
    off = {c.zone_id for c in decision.commands if c.hvac_mode == MODE_OFF}
    assert off == {"living", "guest"}
    assert set(state.shed) == {"living", "guest"}


def test_shed_restores_with_headroom():
    zone = make_zone("bed", 25.0, draw_w=700.0)
    snap = make_snapshot([zone], Settings(hvac_mode=MODE_COOL), p_demand=1200.0, p_grid=1200.0)
    state = warmed_state([zone])
    state.shed["bed"] = NOW - 600.0
    decision = controller.tick(snap, state)
    assert "bed" not in state.shed
    by_zone = {c.zone_id: c for c in decision.commands}
    assert by_zone["bed"].hvac_mode == MODE_COOL


def test_no_restore_without_headroom():
    zone = make_zone("bed", 25.0, draw_w=700.0)
    snap = make_snapshot([zone], Settings(hvac_mode=MODE_COOL), p_demand=3000.0, p_grid=3000.0)
    state = warmed_state([zone])
    state.shed["bed"] = NOW - 600.0
    controller.tick(snap, state)
    assert "bed" in state.shed


def test_disabled_zone_ignored():
    settings = Settings(hvac_mode=MODE_AUTO, zone_enabled={"bed": False})
    zone = make_zone("bed", 26.0)
    snap = make_snapshot([zone], settings)
    state = warmed_state([zone])
    decision = controller.tick(snap, state)
    assert decision.commands == []


def test_quantize_setpoint():
    assert controller.quantize_setpoint(22.4) == 22.5
    assert controller.quantize_setpoint(22.24) == 22.0
    assert controller.quantize_setpoint(35.0) == 30.0
    assert controller.quantize_setpoint(10.0) == 16.0


def test_heating_demand_in_winter():
    from custom_components.adaptive_comfort.core import comfort

    zone = make_zone("bed", 19.0, free_float=[19.0 - 0.05 * h for h in range(24)])
    settings = Settings(hvac_mode=MODE_AUTO)
    snap = make_snapshot([zone], settings, t_rm=11.0)
    state = warmed_state([zone])
    decision = controller.tick(snap, state)
    by_zone = {c.zone_id: c for c in decision.commands}
    assert by_zone["bed"].hvac_mode == MODE_HEAT
    center = comfort.band_center(settings, 11.0)
    lo, _hi = comfort.zone_band(settings, center, None, None)
    expected = comfort.hold_edge_setpoint(MODE_HEAT, lo, _hi, conditioning=False)
    assert abs(by_zone["bed"].setpoint - controller.quantize_setpoint(expected)) < 0.01


def test_mixing_free_ride_in_band_hold():
    # Rider on the cool side of the band but prediction-demanding: hold-cover
    # is enough to free-ride (no pull-down needed yet).
    lead = make_zone("east", 22.5, is_on=True, standing_load_w=100.0)
    rider = make_zone(
        "west",
        23.0,
        is_on=False,
        standing_load_w=40.0,
        mixing_gain_w=-80.0,
        free_float=[23.0, 23.6, 24.0] + [24.2] * 21,
        pred_60m=23.6,
    )
    settings = Settings(hvac_mode=MODE_COOL)
    snap = make_snapshot([lead, rider], settings)
    state = warmed_state([lead, rider])
    state.zone_since["east"] = NOW - 3600.0
    state.zone_on["east"] = True
    decision = controller.tick(snap, state)
    assert "west" in decision.diag.get("free_riders", [])
    assert decision.diag["want"]["west"] == "free_ride"


def test_mixing_free_ride_oob_needs_pull_down():
    # OOB with flat free-float: hold-cover alone must NOT skip — room stays hot.
    lead = make_zone("east", 23.5, is_on=True, standing_load_w=100.0)
    rider = make_zone(
        "west",
        24.0,
        is_on=False,
        standing_load_w=40.0,
        mixing_gain_w=-80.0,
        free_float=[24.0] * 24,
        pred_60m=24.2,
    )
    settings = Settings(hvac_mode=MODE_COOL)
    snap = make_snapshot([lead, rider], settings)
    state = warmed_state([lead, rider])
    # Sibling on for hours — transport grace expired.
    state.zone_since["east"] = NOW - 3600.0
    state.zone_on["east"] = True
    decision = controller.tick(snap, state)
    assert "west" not in decision.diag.get("free_riders", [])
    assert decision.diag["want"]["west"] == "demand"


def test_mixing_free_ride_oob_when_float_enters_band():
    # OOB now, but free-float enters band within the pull horizon → skip.
    lead = make_zone("east", 22.5, is_on=True, standing_load_w=100.0)
    rider = make_zone(
        "west",
        24.0,
        is_on=False,
        standing_load_w=40.0,
        mixing_gain_w=-80.0,
        free_float=[24.0, 23.5, 23.0, 22.8] + [22.5] * 20,
        pred_60m=23.5,
    )
    settings = Settings(hvac_mode=MODE_COOL)
    snap = make_snapshot([lead, rider], settings)
    state = warmed_state([lead, rider])
    state.zone_since["east"] = NOW - 3600.0
    state.zone_on["east"] = True
    decision = controller.tick(snap, state)
    assert "west" in decision.diag.get("free_riders", [])
    assert decision.diag["want"]["west"] == "free_ride"


def test_mixing_free_ride_transport_grace():
    # OOB, flat free-float, but sibling just started → wait for air to arrive.
    lead = make_zone("east", 23.5, is_on=True, standing_load_w=100.0)
    rider = make_zone(
        "west",
        24.0,
        is_on=False,
        standing_load_w=40.0,
        mixing_gain_w=-80.0,
        free_float=[24.0] * 24,
        pred_60m=24.2,
    )
    settings = Settings(hvac_mode=MODE_COOL)
    snap = make_snapshot([lead, rider], settings)
    state = warmed_state([lead, rider])
    state.zone_on["east"] = True
    state.zone_since["east"] = NOW - 60.0  # inside FREE_RIDE_TRANSPORT_GRACE_S
    decision = controller.tick(snap, state)
    assert "west" in decision.diag.get("free_riders", [])
    assert decision.diag["want"]["west"] == "free_ride"


def test_prefer_continuous_elects_anchor_and_parks():
    zone = make_zone(
        "east",
        22.5,
        is_on=True,
        head_state="cooling",
        park_residuals=True,
        park_extraction_w=150.0,
        standing_load_w=80.0,
        free_float=[22.5] * 24,
    )
    settings = Settings(
        hvac_mode=MODE_COOL,
        prefer_continuous=True,
        park_learning=True,
        multisplit=True,
    )
    snap = make_snapshot([zone], settings)
    state = warmed_state([zone])
    state.zone_on["east"] = True
    decision = controller.tick(snap, state)
    assert decision.state.anchor_zone == "east"
    assert decision.diag["want"]["east"] == "anchor"
    # Satisfied anchor must park (residual), not track a demand setpoint.
    assert (
        any((c.head_depth_k or 0) > 0 for c in decision.commands)
        or "east" in decision.state.zone_parked_since
    )
    assert not any(
        c.zone_id == "east"
        and (c.head_depth_k is None or c.head_depth_k <= 0)
        and c.hvac_mode == MODE_COOL
        for c in decision.commands
    )


def test_prefer_continuous_handoff_when_sibling_covers():
    # Parked west is anchor; east wants cooling and mixing already covers west
    # → release west so residual park does not add compressor load.
    west = make_zone(
        "west",
        22.5,
        is_on=True,
        head_state="cooling",
        park_residuals=True,
        park_extraction_w=100.0,
        standing_load_w=40.0,
        mixing_gain_w=-80.0,
        free_float=[22.5] * 24,
    )
    east = make_zone(
        "east",
        25.0,
        is_on=False,
        standing_load_w=100.0,
        free_float=[25.0] * 24,
    )
    settings = Settings(
        hvac_mode=MODE_COOL,
        prefer_continuous=True,
        park_learning=True,
        multisplit=True,
    )
    snap = make_snapshot([west, east], settings)
    state = warmed_state([west, east])
    state.zone_on["west"] = True
    state.anchor_zone = "west"
    state.anchor_since = NOW - 600.0
    state.zone_parked_since["west"] = NOW - 600.0
    state.zone_park_margin["west"] = 2.0
    decision = controller.tick(snap, state)
    assert decision.state.anchor_zone is None
    assert "west" not in decision.state.zone_parked_since
    # East should be wanted on as demand.
    assert decision.diag["want"]["east"] == "demand"


def test_plant_compress_debounce_survives_brief_dip():
    from custom_components.adaptive_comfort.core import power

    zone = make_zone("bed", 22.5, is_on=True, head_state="cooling", free_float=[22.5] * 24)
    settings = Settings(hvac_mode=MODE_COOL, min_on_min=20.0, park_learning=True, multisplit=True)
    state = warmed_state([zone])
    state.zone_on["bed"] = True
    state.plant_compress_since = NOW - 300.0
    # One tick below the floor must not clear the run clock.
    snap = make_snapshot([zone], settings, p_ac=40.0, compression_floor_w=75.0)
    decision = controller.tick(snap, state)
    assert decision.state.plant_compress_since == NOW - 300.0
    assert decision.state.plant_below_since == NOW
    # Still below after debounce → clear.
    snap2 = make_snapshot(
        [zone],
        settings,
        now=NOW + power.START_DEBOUNCE_S + 1,
        p_ac=40.0,
        compression_floor_w=75.0,
    )
    decision = controller.tick(snap2, decision.state)
    assert decision.state.plant_compress_since == 0.0


def test_pack_stay_parks_two_in_band_heads():
    hot = make_zone(
        "hot",
        22.6,
        is_on=True,
        head_state="cooling",
        free_float=[22.6] * 24,
    )
    ok = make_zone(
        "ok",
        22.0,
        is_on=True,
        head_state="cooling",
        park_residual_edge_k=2.5,
        free_float=[22.0] * 24,
    )
    settings = Settings(hvac_mode=MODE_COOL, park_learning=True, multisplit=True)
    snap = make_snapshot([hot, ok], settings)
    state = warmed_state([hot, ok])
    state.zone_on["hot"] = True
    state.zone_on["ok"] = True
    decision = controller.tick(snap, state)
    assert set(decision.diag.get("pack_stay", [])) == {"hot", "ok"}
    parked = {c.zone_id for c in decision.commands if (c.head_depth_k or 0) > 0} | set(
        decision.state.zone_parked_since
    )
    assert "hot" in parked and "ok" in parked
    assert not any(c.hvac_mode == MODE_OFF for c in decision.commands)
    for zid in ("hot", "ok"):
        depth = decision.state.zone_head_depth_k.get(zid, 0.0)
        assert depth < 1.0


def test_prefer_continuous_keeps_pack_not_one_leftover():
    east = make_zone(
        "east",
        22.6,
        is_on=True,
        head_state="cooling",
        free_float=[22.6] * 24,
    )
    west = make_zone(
        "west",
        22.4,
        is_on=True,
        head_state="cooling",
        free_float=[22.4] * 24,
    )
    settings = Settings(
        hvac_mode=MODE_COOL,
        prefer_continuous=True,
        park_learning=True,
        multisplit=True,
    )
    snap = make_snapshot([east, west], settings)
    state = warmed_state([east, west])
    state.zone_on["east"] = True
    state.zone_on["west"] = True
    decision = controller.tick(snap, state)
    parked = {c.zone_id for c in decision.commands if (c.head_depth_k or 0) > 0} | set(
        decision.state.zone_parked_since
    )
    assert "east" in parked and "west" in parked
    assert not any(c.hvac_mode == MODE_OFF for c in decision.commands)


def test_eco_waits_until_every_zone_needs_burst():
    from custom_components.adaptive_comfort.core.types import PRESET_ECO

    settings = Settings(
        hvac_mode=MODE_COOL,
        preset=PRESET_ECO,
        zone_quiet_night={"west": True},
    )
    west = make_zone("west", 25.0, occupied=True)
    east = make_zone("east", 22.6, occupied=True)
    snap = make_snapshot([west, east], settings, local_hour=23.0)
    state = warmed_state([west, east])
    decision = controller.tick(snap, state)
    assert decision.diag.get("energy_wait") is True
    assert not any(c.hvac_mode == MODE_COOL for c in decision.commands)


def test_eco_runs_full_pack_including_quiet_night():
    from custom_components.adaptive_comfort.core.types import PRESET_ECO

    settings = Settings(
        hvac_mode=MODE_COOL,
        preset=PRESET_ECO,
        zone_quiet_night={"west": True},
    )
    west = make_zone("west", 25.0, occupied=True)
    east = make_zone("east", 25.0, occupied=True)
    snap = make_snapshot([west, east], settings, local_hour=23.0)
    state = warmed_state([west, east])
    decision = controller.tick(snap, state)
    by_zone = {c.zone_id: c for c in decision.commands}
    assert not decision.diag.get("energy_wait")
    assert not decision.diag.get("quiet_night_deferred")
    assert by_zone["west"].hvac_mode == MODE_COOL
    assert by_zone["east"].hvac_mode == MODE_COOL
    assert decision.diag["want"]["west"] == "demand"
    assert decision.diag["want"]["east"] == "demand"


def test_eco_override_starts_immediately():
    from custom_components.adaptive_comfort.core.types import PRESET_ECO

    settings = Settings(hvac_mode=MODE_COOL, preset=PRESET_ECO)
    # Eco hi ≈ 24.7; OVERRIDE_DELTA_K = 2 → start at > 26.7.
    west = make_zone("west", 27.0, occupied=True)
    east = make_zone("east", 22.6, occupied=True)
    snap = make_snapshot([west, east], settings)
    state = warmed_state([west, east])
    decision = controller.tick(snap, state)
    by_zone = {c.zone_id: c for c in decision.commands}
    assert not decision.diag.get("energy_wait")
    assert by_zone["west"].hvac_mode == MODE_COOL
    assert "east" not in by_zone or by_zone["east"].hvac_mode == MODE_OFF


def test_eco_mid_burst_does_not_kill_remaining_demand():
    from custom_components.adaptive_comfort.core.types import PRESET_ECO

    settings = Settings(
        hvac_mode=MODE_COOL,
        preset=PRESET_ECO,
        park_learning=True,
        multisplit=True,
    )
    west = make_zone("west", 25.0, occupied=True, is_on=True, head_state="cooling")
    east = make_zone("east", 22.6, occupied=True, is_on=True, head_state="cooling")
    snap = make_snapshot([west, east], settings)
    state = warmed_state([west, east])
    state.zone_on["west"] = True
    state.zone_on["east"] = True
    decision = controller.tick(snap, state)
    assert not decision.diag.get("energy_wait")
    assert decision.diag["want"]["west"] == "demand"
    assert not any(c.zone_id == "west" and c.hvac_mode == MODE_OFF for c in decision.commands)
    assert (
        "east" in decision.diag.get("pack_stay", []) or "east" in decision.state.zone_parked_since
    )
    assert not any(c.zone_id == "east" and c.hvac_mode == MODE_OFF for c in decision.commands)


def test_away_waits_until_every_zone_needs_burst():
    from custom_components.adaptive_comfort.core.types import PRESET_AWAY

    settings = Settings(hvac_mode=MODE_COOL, preset=PRESET_AWAY)
    west = make_zone("west", 27.0, occupied=True)
    east = make_zone("east", 22.6, occupied=True)
    snap = make_snapshot([west, east], settings, house_occupied=False)
    state = warmed_state([west, east])
    decision = controller.tick(snap, state)
    assert decision.diag.get("energy_wait") is True
    assert not any(c.hvac_mode == MODE_COOL for c in decision.commands)


def test_mixing_does_not_peel_in_band_pack():
    west = make_zone(
        "west",
        22.5,
        is_on=True,
        head_state="cooling",
        mixing_gain_w=-80.0,
        standing_load_w=40.0,
        free_float=[22.5] * 24,
    )
    east = make_zone(
        "east",
        22.6,
        is_on=True,
        head_state="cooling",
        standing_load_w=80.0,
        free_float=[22.6] * 24,
    )
    settings = Settings(hvac_mode=MODE_COOL, park_learning=True, multisplit=True)
    snap = make_snapshot([west, east], settings)
    state = warmed_state([west, east])
    state.zone_on["west"] = True
    state.zone_on["east"] = True
    state.zone_parked_since["west"] = NOW - 600.0
    state.zone_parked_since["east"] = NOW - 600.0
    state.zone_park_margin["west"] = 0.5
    state.zone_park_margin["east"] = 0.5
    state.zone_head_depth_k["west"] = 0.5
    state.zone_head_depth_k["east"] = 0.5
    decision = controller.tick(snap, state)
    assert set(decision.diag.get("pack_stay", [])) == {"west", "east"}
    assert "west" in decision.state.zone_parked_since
    assert "east" in decision.state.zone_parked_since
    assert not any(c.hvac_mode == MODE_OFF for c in decision.commands)


def test_depth_ledger_raises_in_band_chase_cap():
    snap = make_snapshot(
        [make_zone("hot", 22.6), make_zone("ok", 22.4)],
        cop_by_depth={-0.5: 2.0, -2.0: 3.5},
    )
    cap = controller._depth_chase_cap_k(snap, n_heads=2, in_band=True)
    assert cap == 2.0
    empty = make_snapshot([make_zone("hot", 22.6)])
    assert controller._depth_chase_cap_k(empty, n_heads=2, in_band=True) == 0.5
    assert controller._depth_chase_cap_k(empty, n_heads=1, in_band=True) == 2.5


def test_compressing_hold_stays_below_fan_type_shelf():
    zone = make_zone("ok", 22.6, park_residual_max_margin_k=2.5, park_fan_type_min_margin_k=1.0)
    assert controller._compressing_hold_k(zone) == 0.5
    mapped = make_zone("ok", 22.6, park_residual_max_margin_k=2.0, park_residual_edge_k=2.5)
    assert controller._compressing_hold_k(mapped) == 2.0
    unmapped = make_zone("ok", 22.6)
    assert controller._compressing_hold_k(unmapped) == controller.DEPTH_MAX_K


def test_helper_walks_into_shallow_park_not_covering_bin():
    """Recruited helpers hold at live temp (depth 0), then +0.5 K per re-anchor.

    Once depth is stable and positive, spacing does not re-issue the park
    command (frozen hold — not the descending-internal ladder).
    """
    hot = make_zone("hot", 25.0)
    ok = make_zone(
        "ok",
        22.6,
        park_residuals=True,
        park_residual_max_margin_k=2.0,
        park_residual_edge_k=2.5,
        park_margin_bins={"2.0": [150.0, 0.8, 12.0]},
        standing_load_w=80.0,
    )
    snap = make_snapshot([hot, ok])
    state = warmed_state([hot, ok])
    decision = controller.tick(snap, state)
    cmd = {c.zone_id: c for c in decision.commands}["ok"]
    assert cmd.reason == "helper"
    assert cmd.head_depth_k == 0.0
    assert cmd.head_depth_k is None or cmd.head_depth_k <= 0
    assert cmd.setpoint == 22.5  # hold at live 22.6, quantized
    ok_on = make_zone(
        "ok",
        22.6,
        is_on=True,
        park_residuals=True,
        park_residual_max_margin_k=2.0,
        park_residual_edge_k=2.5,
        park_margin_bins={"2.0": [150.0, 0.8, 12.0]},
        standing_load_w=80.0,
    )
    snap2 = make_snapshot([hot, ok_on], now=NOW + controller.COMMAND_SPACING_S)
    decision = controller.tick(snap2, decision.state)
    cmd2 = {c.zone_id: c for c in decision.commands}["ok"]
    assert cmd2.head_depth_k == 0.5
    assert cmd2.head_depth_k is not None and cmd2.head_depth_k > 0
    assert cmd2.head_depth_k < 2.5
    assert cmd2.setpoint == 23.0  # 22.5 + 0.5

    snap3 = make_snapshot([hot, ok_on], now=NOW + 2 * controller.COMMAND_SPACING_S)
    decision = controller.tick(snap3, decision.state)
    # Frozen park hold: spacing must not re-issue SP = internal + depth.
    assert not any(c.zone_id == "ok" for c in decision.commands)
    assert decision.state.zone_head_depth_k["ok"] == 0.5

    racing = make_zone(
        "ok",
        22.0,
        is_on=True,
        park_residuals=True,
        park_residual_max_margin_k=2.0,
        park_residual_edge_k=2.5,
        park_margin_bins={"2.0": [150.0, 0.8, 12.0]},
        standing_load_w=80.0,
        pred_60m=21.7,
    )
    snap4 = make_snapshot([hot, racing], now=NOW + 3 * controller.COMMAND_SPACING_S)
    decision = controller.tick(snap4, decision.state)
    cmd4 = {c.zone_id: c for c in decision.commands}["ok"]
    assert cmd4.head_depth_k == 1.0
    assert cmd4.head_depth_k < 2.5


def test_idle_mapped_ceiling_is_park_margin_not_depth_max():
    zone = make_zone("ok", 22.6, park_margin_bins={"3.0": [0.0, 0.5, 20.0]})
    assert controller._hysteresis_ceiling_k(zone) == controller.PARK_MARGIN_K


def test_demand_cost_skips_dying_nick_when_action_overshoots():
    """0.28 K over hi that free-float returns, while min_on would cross lo."""
    lo, hi = 23.01, 24.21
    zone = make_zone(
        "east",
        24.49,
        free_float=[24.49, 24.10, 23.90] + [23.80] * 21,
        q_hvac_k_per_h=-4.0,
    )
    assert not controller._wants_conditioning(zone, lo, hi, MODE_COOL, 30.0)


def test_demand_cost_starts_when_inaction_stays_hot():
    lo, hi = 23.01, 24.21
    zone = make_zone(
        "east",
        24.49,
        free_float=[24.49, 24.70, 25.00] + [25.20] * 21,
        q_hvac_k_per_h=-1.0,
    )
    assert controller._wants_conditioning(zone, lo, hi, MODE_COOL, 30.0)


def test_demand_present_oob_without_q_hvac_still_starts():
    zone = make_zone("bed", 25.0)
    assert controller._wants_conditioning(zone, 21.8, 23.2, MODE_COOL, 20.0)


def test_in_band_advance_still_fires_when_q_hvac_is_set():
    """Cost-compare used to run in-band and killed COP advance."""
    zone_in = make_zone(
        "z1",
        23.2,
        free_float=(23.2, 23.5, 24.5, 25.0) + (25.0,) * 20,
        q_hvac_k_per_h=-4.0,
    )
    assert controller._wants_conditioning(
        zone_in, 22.5, 23.9, MODE_COOL, 20.0, cop_timing="advance"
    )


def test_present_oob_wrong_sign_q_hvac_still_starts():
    zone = make_zone(
        "east",
        25.0,
        free_float=[25.0] * 24,
        q_hvac_k_per_h=2.0,
    )
    assert controller._wants_conditioning(zone, 21.8, 23.2, MODE_COOL, 20.0)


def test_eco_quiet_night_does_not_condition_hold():
    """Eco/Away/Boost skip quiet-night; Eco must not bank the cold edge."""
    from custom_components.adaptive_comfort.core.types import PRESET_ECO, PRESET_NONE

    eco = Settings(
        hvac_mode=MODE_COOL,
        preset=PRESET_ECO,
        zone_quiet_night={"bed": True},
        adaptive_blend=0.0,
    )
    none = Settings(
        hvac_mode=MODE_COOL,
        preset=PRESET_NONE,
        zone_quiet_night={"bed": True},
        adaptive_blend=0.0,
    )
    bed = make_zone("bed", 22.8, occupied=True)
    east = make_zone("east", 22.6, occupied=False)
    eco_d = controller.tick(
        make_snapshot([bed, east], eco, local_hour=20.5), warmed_state([bed, east])
    )
    none_d = controller.tick(
        make_snapshot([bed, east], none, local_hour=20.5), warmed_state([bed, east])
    )
    assert "bed" in none_d.diag.get("quiet_night_bank", [])
    assert not eco_d.diag.get("quiet_night_bank")
    assert not eco_d.diag.get("quiet_night_deferred")
    eco_cmd = next((c for c in eco_d.commands if c.zone_id == "bed"), None)
    none_cmd = next((c for c in none_d.commands if c.zone_id == "bed"), None)
    assert none_cmd is not None and none_cmd.setpoint is not None
    if eco_cmd is not None and eco_cmd.setpoint is not None:
        assert eco_cmd.setpoint > none_cmd.setpoint


def test_shed_sole_demand_does_not_recruit_helpers():
    """A latched-shed demand zone must not keep recruiting in-band helpers."""
    hot = make_zone("hot", 25.0, occupied=True, draw_w=700.0)
    ok = make_zone("ok", 22.6, occupied=True)
    settings = Settings(hvac_mode=MODE_COOL, coordination=True)
    snap = make_snapshot([hot, ok], settings, p_demand=3000.0, p_grid=3000.0)
    state = warmed_state([hot, ok])
    state.shed["hot"] = NOW - 600.0
    decision = controller.tick(snap, state)
    assert "hot" in state.shed
    by_zone = {c.zone_id: c for c in decision.commands}
    assert "ok" not in by_zone or by_zone["ok"].hvac_mode == MODE_OFF
    assert "ok" not in decision.diag.get("helpers", [])
    assert not any(c.reason == "helper" for c in decision.commands)


def test_want_tokens_match_sensor_enum():
    allowed = {"off", "demand", "helper", "anchor", "free_ride"}
    hot = make_zone("hot", 25.0)
    ok = make_zone("ok", 22.6)
    snap = make_snapshot([hot, ok])
    state = warmed_state([hot, ok])
    decision = controller.tick(snap, state)
    assert set(decision.diag["want"].values()) <= allowed
