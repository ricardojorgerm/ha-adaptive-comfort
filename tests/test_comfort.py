from custom_components.adaptive_comfort.core import comfort
from custom_components.adaptive_comfort.core.types import (
    MODE_COOL,
    MODE_HEAT,
    MODE_OFF,
    ControllerState,
    HouseSnapshot,
    Settings,
    ZoneSnapshot,
)


def make_zone(zone_id="z1", temp=22.5, free_float=(), confidence=0.9, **kwargs):
    return ZoneSnapshot(
        zone_id=zone_id,
        name=zone_id,
        n_rooms=1,
        temp=temp,
        free_float=tuple(free_float),
        confidence=confidence,
        **kwargs,
    )


def make_snapshot(zones, t_rm=None, now=1_000_000.0, aux=()):
    return HouseSnapshot(
        now_ts=now,
        local_hour=12.0,
        settings=Settings(),
        zones=zones,
        t_rm=t_rm,
        aux_indoor=aux,
    )


def test_adaptive_target_clamped():
    assert comfort.adaptive_target(30.0) == 26.0
    assert comfort.adaptive_target(0.0) == 20.0
    assert abs(comfort.adaptive_target(20.0) - (0.33 * 20 + 18.8)) < 1e-9


def test_band_center_blend():
    s = Settings(target=22.5, adaptive_blend=0.0)
    assert comfort.band_center(s, 25.0) == 22.5
    s.adaptive_blend = 1.0
    assert abs(comfort.band_center(s, 25.0) - comfort.adaptive_target(25.0)) < 1e-9


def test_band_widens_when_unoccupied_and_away():
    s = Settings(band_k=0.7)
    lo, hi = comfort.zone_band(s, 22.5, zone_occupied=None, house_occupied=None)
    assert abs((hi - lo) / 2 - 0.7) < 1e-9
    lo, hi = comfort.zone_band(s, 22.5, zone_occupied=False, house_occupied=None)
    assert abs((hi - lo) / 2 - 2.2) < 1e-9
    lo, hi = comfort.zone_band(s, 22.5, zone_occupied=None, house_occupied=False)
    assert abs((hi - lo) / 2 - 3.7) < 1e-9  # auto-away


def test_boost_overrides_presence_away():
    from custom_components.adaptive_comfort.core.types import PRESET_BOOST

    s = Settings(band_k=0.7, preset=PRESET_BOOST, presence_adaptation=True)
    assert comfort.effective_preset(s, house_occupied=False) == PRESET_BOOST
    lo, hi = comfort.zone_band(s, 22.5, zone_occupied=True, house_occupied=False)
    assert abs((hi - lo) / 2 - 0.35) < 1e-9  # half * 0.5, no away widen


def test_boost_skips_vacant_widen():
    from custom_components.adaptive_comfort.core.types import PRESET_BOOST, PRESET_NONE

    s = Settings(band_k=0.7, preset=PRESET_BOOST)
    lo, hi = comfort.zone_band(s, 22.5, zone_occupied=False, house_occupied=True)
    assert abs((hi - lo) / 2 - 0.35) < 1e-9  # not 0.35+1.5
    s_none = Settings(band_k=0.7, preset=PRESET_NONE)
    lo_n, hi_n = comfort.zone_band(s_none, 22.5, zone_occupied=False, house_occupied=True)
    assert abs((hi_n - lo_n) / 2 - 2.2) < 1e-9


def test_demand_setpoint_band_hold_when_away():
    from custom_components.adaptive_comfort.core.types import PRESET_AWAY, PRESET_BOOST, PRESET_NONE

    assert comfort.demand_setpoint(MODE_COOL, 22.5, 21.8, 23.2, True, PRESET_NONE) == 22.5
    assert (
        abs(
            comfort.demand_setpoint(MODE_COOL, 22.5, 18.8, 26.2, True, PRESET_AWAY)
            - (26.2 - comfort.BAND_HOLD_MARGIN_K)
        )
        < 1e-9
    )
    # Boost center-seeks even when vacant.
    assert comfort.demand_setpoint(MODE_COOL, 22.5, 20.0, 25.0, False, PRESET_BOOST) == 22.5


def test_running_mean_seed_and_update():
    t_rm = comfort.update_running_mean(None, 20.0, 1.0)
    assert t_rm == 20.0
    t_rm2 = comfort.update_running_mean(20.0, 30.0, 24.0)
    assert 20.0 < t_rm2 < 22.0


def test_climatology_lisbon_summer_vs_winter():
    assert comfort.climatology_mean(8) > 22.0  # August
    assert comfort.climatology_mean(1) < 13.0  # January
    # Afternoon warmer than dawn.
    assert comfort.climatology_temp(7, 15.0) > comfort.climatology_temp(7, 5.0)


def test_dominant_mode_model_cooling():
    # Trajectory drifts well above the band: expect cooling.
    zone = make_zone(free_float=[24.0 + 0.1 * h for h in range(24)])
    snap = make_snapshot([zone])
    state = ControllerState()
    bands = {"z1": (21.8, 23.2)}
    decision = comfort.dominant_mode(snap, state, bands)
    assert decision.mode == MODE_COOL
    assert decision.source == "model"


def test_dominant_mode_heat_soak_scenario():
    # Cold day (t_rm would say heat) but house free-floats warm: cool bias.
    zone = make_zone(temp=24.5, free_float=[24.5 + 0.05 * h for h in range(24)])
    snap = make_snapshot([zone], t_rm=13.0)
    state = ControllerState()
    decision = comfort.dominant_mode(snap, state, {"z1": (21.8, 23.2)})
    assert decision.mode == MODE_COOL


def test_dominant_mode_deadband_idles():
    zone = make_zone(free_float=[22.5] * 24)
    snap = make_snapshot([zone])
    state = ControllerState()
    decision = comfort.dominant_mode(snap, state, {"z1": (21.8, 23.2)})
    assert decision.mode == MODE_OFF


def test_dominant_mode_dwell_blocks_flip():
    state = ControllerState()
    state.mode = MODE_COOL
    state.mode_since = 1_000_000.0 - 3600.0  # only 1 h in cool
    zone = make_zone(free_float=[19.0] * 24)  # now demands heat
    snap = make_snapshot([zone])
    decision = comfort.dominant_mode(snap, state, {"z1": (21.8, 23.2)})
    assert decision.mode == MODE_COOL
    assert decision.source == "dwell"


def test_emergency_override_beats_dwell():
    now = 1_000_000.0
    state = ControllerState()
    state.mode = MODE_COOL
    state.mode_since = now - 3600.0
    state.override_since = now - 2000.0  # sustained > 30 min
    zone = make_zone(temp=18.0, occupied=True, free_float=[18.0] * 24)
    snap = make_snapshot([zone], now=now)
    decision = comfort.dominant_mode(snap, state, {"z1": (21.8, 23.2)})
    assert decision.mode == MODE_HEAT
    assert decision.source == "override"


def test_fallback_no_model_hot_room_in_summer():
    # Cold start: no fitted model — persistence still feeds the model path.
    zone = make_zone(temp=25.5, confidence=0.0, free_float=())
    snap = make_snapshot([zone], t_rm=23.5)
    state = ControllerState()
    decision = comfort.dominant_mode(snap, state, {"z1": (21.8, 23.2)})
    assert decision.source == "model"
    assert decision.mode == MODE_COOL


def test_fallback_does_not_fight_season_on_small_error():
    # Mild winter, room slightly warm: stay off rather than cool.
    zone = make_zone(temp=23.5, confidence=0.0, free_float=())
    snap = make_snapshot([zone], t_rm=12.0)
    state = ControllerState()
    decision = comfort.dominant_mode(snap, state, {"z1": (21.8, 23.2)})
    assert decision.mode == MODE_OFF


def test_fallback_uses_aux_bathroom_sensor():
    # Zone sensors missing entirely; the bathroom sensor alone shows the
    # house is far too cold, so cold-start arbitration heats.
    zone = make_zone(temp=None, confidence=0.0, free_float=())
    snap = make_snapshot([zone], t_rm=12.0, aux=((17.0, 0.25),))
    state = ControllerState()
    decision = comfort.dominant_mode(snap, state, {"z1": (21.8, 23.2)})
    assert decision.source == "fallback"
    assert decision.mode == MODE_HEAT


def test_model_mode_does_not_heat_summer_warm_house_on_forecast():
    """Regression: bogus cold-deficit forecast must not heat a 24 °C house in July."""
    # Both zones contribute counterfactual free-float; seasonal guard blocks heat.
    west = make_zone(
        temp=24.4,
        free_float=[18.0] * 24,
        is_on=False,
        zone_id="west",
    )
    east = make_zone(
        temp=24.4,
        free_float=[26.0] * 24,
        is_on=True,
        zone_id="east",
    )
    snap = make_snapshot([west, east], t_rm=23.3)
    state = ControllerState(mode=MODE_COOL, mode_since=0.0)
    bands = {"west": (21.7, 26.1), "east": (23.2, 24.6)}
    decision = comfort.dominant_mode(snap, state, bands)
    assert decision.mode != MODE_HEAT
    assert decision.mode in (MODE_COOL, MODE_OFF)


def test_demand_integrals_include_conditioning_zones():
    """On zones still contribute counterfactual free-float (no-AC from current temp)."""
    on = make_zone(zone_id="on", free_float=[30.0] * 24, is_on=True, temp=30.0)
    off = make_zone(zone_id="off", free_float=[22.5] * 24, is_on=False, temp=22.5)
    bands = {"on": (21.8, 23.2), "off": (21.8, 23.2)}
    warm, cold = comfort.demand_integrals([on, off], bands)
    # 8 h * (30 - 23.2) from the on zone; off zone in-band.
    assert abs(warm - 8 * (30.0 - 23.2)) < 1e-9
    assert cold == 0.0


def test_hot_on_zone_keeps_cool_mode():
    """Field regression: occupied hot room that is already on must not drop to off."""
    zone = make_zone(
        temp=27.0,
        free_float=[27.0] * 24,
        is_on=True,
        occupied=True,
        confidence=0.9,
    )
    snap = make_snapshot([zone])
    state = ControllerState(mode=MODE_COOL, mode_since=0.0)
    bands = {"z1": (21.8, 23.2)}
    d1 = comfort.dominant_mode(snap, state, bands)
    assert d1.mode == MODE_COOL
    d2 = comfort.dominant_mode(snap, state, bands)
    assert d2.mode == MODE_COOL


def test_asymmetric_exit_holds_cool_through_deadband():
    # Entered cool earlier; warm excess between EXIT and DEADBAND.
    # 8 * 0.2 = 1.6 → > EXIT(1.0) and < DEADBAND(3.0) → hold cool
    zone2 = make_zone(temp=23.4, free_float=[23.4] * 24)
    snap2 = make_snapshot([zone2])
    state = ControllerState(mode=MODE_COOL, mode_since=0.0)
    bands = {"z1": (21.8, 23.2)}
    decision = comfort.dominant_mode(snap2, state, bands)
    assert decision.mode == MODE_COOL


def test_asymmetric_exit_releases_when_excess_clears():
    zone = make_zone(temp=23.25, free_float=[23.25] * 24)  # 8*0.05 = 0.4 < EXIT
    snap = make_snapshot([zone])
    state = ControllerState(mode=MODE_COOL, mode_since=0.0)
    decision = comfort.dominant_mode(snap, state, {"z1": (21.8, 23.2)})
    assert decision.mode == MODE_OFF


def test_seasonal_guard_does_not_silence_hot_room():
    """House-mean dilution must not force off while one zone is clearly hot."""
    hot = make_zone(zone_id="hot", temp=27.0, free_float=[27.0] * 24)
    cool = make_zone(zone_id="ok", temp=21.5, free_float=[21.5] * 24)
    snap = make_snapshot([hot, cool], t_rm=12.0)  # winter running mean
    state = ControllerState(mode=MODE_COOL, mode_since=0.0)
    bands = {"hot": (21.8, 23.2), "ok": (21.8, 23.2)}
    decision = comfort.dominant_mode(snap, state, bands)
    assert decision.mode == MODE_COOL


def test_low_confidence_uses_temp_persistence():
    zone = make_zone(temp=27.0, free_float=(), confidence=0.0, is_on=True)
    warm, _cold = comfort.demand_integrals([zone], {"z1": (21.8, 23.2)})
    assert abs(warm - 8 * (27.0 - 23.2)) < 1e-9


def test_indoor_deviation_aux_weighting():
    zones = [make_zone(temp=23.0)]
    bands = {"z1": (21.8, 23.2)}
    dev = comfort.indoor_deviation(zones, bands, aux_indoor=((26.0, 1.0),))
    # zone dev = +0.5, aux dev = +3.5 equally weighted -> 2.0
    assert abs(dev - 2.0) < 1e-9
