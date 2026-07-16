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


def make_snapshot(zones, settings=None, now=NOW, **kw):
    return HouseSnapshot(
        now_ts=now,
        local_hour=12.0,
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
    assert commands["bed"].setpoint == 22.5


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
    # Helper setpoint sits toward the upper band edge (gentle trim).
    assert by_zone["ok"].setpoint > by_zone["hot"].setpoint
    assert "edge" not in by_zone


def test_coordination_skips_unoccupied_helpers():
    hot = make_zone("hot", 25.0)
    empty = make_zone("empty", 22.6, occupied=False)
    snap = make_snapshot([hot, empty])
    state = warmed_state([hot, empty])
    decision = controller.tick(snap, state)
    by_zone = {c.zone_id: c for c in decision.commands}
    assert "empty" not in by_zone


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


def test_cop_table_consolidates_when_fewer_heads_win():
    hot = make_zone("hot", 25.0)
    ok = make_zone("ok", 22.6)
    snap = make_snapshot([hot, ok], cop_by_head_count={1: 4.0, 2: 3.0})
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


def test_min_on_blocks_early_stop():
    # Head is actively cooling (wet coil), room already past the band.
    zone = make_zone("bed", 21.0, is_on=True, head_state="cooling", free_float=[21.0] * 24)
    settings = Settings(hvac_mode=MODE_COOL)
    snap = make_snapshot([zone], settings)
    state = ControllerState()
    state.zone_on["bed"] = True
    state.zone_since["bed"] = NOW - 300.0  # on for 5 min < 20 min minimum
    decision = controller.tick(snap, state)
    assert not any(c.hvac_mode == MODE_OFF for c in decision.commands)
    # After the minimum runtime it may stop.
    state.zone_since["bed"] = NOW - 1300.0
    state.zone_last_cmd["bed"] = 0.0
    decision = controller.tick(snap, state)
    assert any(c.hvac_mode == MODE_OFF for c in decision.commands)


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
    zone = make_zone("bed", 19.0, free_float=[19.0 - 0.05 * h for h in range(24)])
    snap = make_snapshot([zone], t_rm=11.0)
    state = warmed_state([zone])
    decision = controller.tick(snap, state)
    by_zone = {c.zone_id: c for c in decision.commands}
    assert by_zone["bed"].hvac_mode == MODE_HEAT
