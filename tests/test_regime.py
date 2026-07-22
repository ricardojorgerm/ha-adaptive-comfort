"""Regime policy and compressor start counting."""

from custom_components.adaptive_comfort.core import controller, power
from custom_components.adaptive_comfort.core.types import (
    MODE_COOL,
    ControllerState,
    HouseSnapshot,
    Settings,
    ZoneSnapshot,
)

NOW = 1_000_000.0


def make_zone(zone_id, temp, n_rooms=1, **kw):
    return ZoneSnapshot(
        zone_id=zone_id,
        name=zone_id,
        n_rooms=n_rooms,
        temp=temp,
        free_float=tuple([temp] * 24),
        confidence=0.9,
        **kw,
    )


def snap(zones, t_out, settings=None, now=NOW):
    return HouseSnapshot(
        now_ts=now,
        local_hour=12.0,
        settings=settings or Settings(hvac_mode=MODE_COOL, target=23.0),
        zones=zones,
        t_out=t_out,
    )


def warmed(zones, now=NOW):
    st = ControllerState()
    for z in zones:
        st.zone_since[z.zone_id] = now - 7200.0
        st.zone_on[z.zone_id] = z.is_on
    st.mode_since = now - 24 * 3600.0
    return st


def test_regime_ventilate_on_cold_outdoors():
    zones = [make_zone("z1", 26.0, standing_load_w=600.0)]
    st = warmed(zones)
    d = controller.tick(snap(zones, t_out=18.0), st)
    assert d.diag["regime"] == "ventilate"


def test_regime_continuous_scales_with_head_count():
    # One-head zone: threshold ≈ 0.6 * 265 ≈ 159 W.
    one = [make_zone("east", 26.0, standing_load_w=170.0, n_rooms=1)]
    st = warmed(one)
    d = controller.tick(snap(one, t_out=30.0), st)
    assert d.diag["regime"] == "continuous"
    # Two-head zone: threshold ≈ 318 W; 300 W night-West sits just below.
    west_low = [make_zone("west", 26.0, standing_load_w=300.0, n_rooms=2)]
    st = warmed(west_low)
    d = controller.tick(snap(west_low, t_out=30.0), st)
    assert d.diag["regime"] == "cycling"
    west_high = [make_zone("west", 26.0, standing_load_w=330.0, n_rooms=2)]
    st = warmed(west_high)
    d = controller.tick(snap(west_high, t_out=30.0), st)
    assert d.diag["regime"] == "continuous"


def test_regime_cycling_on_small_load():
    zones = [make_zone("z1", 26.0, standing_load_w=100.0)]
    st = warmed(zones)
    d = controller.tick(snap(zones, t_out=30.0), st)
    assert d.diag["regime"] == "cycling"


def test_regime_dwell_hysteresis():
    zones = [make_zone("z1", 26.0, standing_load_w=600.0)]
    st = warmed(zones)
    controller.tick(snap(zones, t_out=30.0), st)
    assert st.regime == "continuous"
    # Conditions flip, but within the dwell the regime holds.
    small = [make_zone("z1", 26.0, standing_load_w=100.0)]
    controller.tick(snap(small, t_out=30.0, now=NOW + 120.0), st)
    assert st.regime == "continuous"
    controller.tick(snap(small, t_out=30.0, now=NOW + controller.REGIME_DWELL_S + 60.0), st)
    assert st.regime == "cycling"


def test_regime_off_switch_restores_cycling():
    settings = Settings(hvac_mode=MODE_COOL, target=23.0, auto_regime=False)
    zones = [make_zone("z1", 26.0, standing_load_w=600.0)]
    st = warmed(zones)
    d = controller.tick(snap(zones, t_out=18.0, settings=settings), st)
    assert d.diag["regime"] == "cycling"


def test_continuous_regime_parks_without_sibling():
    zones = [make_zone("sat", 23.0, is_on=True, standing_load_w=600.0)]
    st = warmed(zones)
    st.mode = "cool"
    d = controller.tick(snap(zones, t_out=30.0), st)
    assert st.regime == "continuous"
    cmd = next((c for c in d.commands if c.zone_id == "sat"), None)
    assert cmd is not None and cmd.park is True
    assert "sat" in st.zone_parked_since


def test_start_counter_debounce_and_rate():
    c = power.StartCounter()
    t = NOW
    assert c.update(t, 400.0, "conditioning") is True  # first rise counts
    assert c.update(t + 60, 40.0, "conditioning") is False
    # Brief dip (< debounce) then rise again: not a new start.
    assert c.update(t + 120, 400.0, "conditioning") is False
    # Long coast then rise: a real restart, tagged with its state.
    assert c.update(t + 200, 40.0, "park") is False
    assert c.update(t + 200 + power.START_DEBOUNCE_S + 1, 400.0, "park") is True
    assert c.per_hour(t + 600, window_s=3600.0) == 2.0
    assert c.by_state(t + 600) == {"conditioning": 1, "park": 1}


def test_start_counter_round_trip():
    c = power.StartCounter()
    c.update(NOW, 400.0, "park")
    r = power.StartCounter.from_dict(c.to_dict())
    assert r.events and r.events[0][1] == "park"
    # Edge state restored: a mid-run restart must not recount the same start.
    assert r._above is True
    assert r.update(NOW + 30.0, 420.0, "park") is False


def test_continuous_regime_in_heat():
    from custom_components.adaptive_comfort.core.types import MODE_HEAT

    # Satisfied warm zone wants off; continuous still parks without a sibling.
    zones = [make_zone("sat", 22.0, is_on=True, standing_load_w=600.0)]
    st = warmed(zones)
    st.mode = MODE_HEAT
    settings = Settings(hvac_mode=MODE_HEAT, target=22.0)
    d = controller.tick(snap(zones, t_out=5.0, settings=settings), st)
    assert d.diag["regime"] == "continuous"
    cmd = next((c for c in d.commands if c.zone_id == "sat"), None)
    assert cmd is not None and cmd.park is True


def test_night_ventilate_forces_ventilate_at_target():
    # Daytime: outdoor == coolest target is not enough (needs REGIME_VENT_MARGIN_K).
    zones = [make_zone("z1", 26.0, standing_load_w=600.0)]
    settings = Settings(hvac_mode=MODE_COOL, target=23.0, night_ventilate=True)
    day = HouseSnapshot(
        now_ts=NOW,
        local_hour=14.0,
        settings=settings,
        zones=zones,
        t_out=23.0,
    )
    st = warmed(zones)
    d = controller.tick(day, st)
    assert d.diag["regime"] == "continuous"
    # Overnight with the switch on: outdoor at the coolest target → ventilate.
    night = HouseSnapshot(
        now_ts=NOW + controller.REGIME_DWELL_S + 60.0,
        local_hour=2.0,
        settings=settings,
        zones=zones,
        t_out=23.0,
    )
    d = controller.tick(night, st)
    assert d.diag["regime"] == "ventilate"


def test_night_ventilate_skips_continuous_near_cool():
    # Outdoor a bit warmer than target: not ventilate, but at night skip continuous.
    zones = [make_zone("z1", 26.0, standing_load_w=600.0)]
    settings = Settings(hvac_mode=MODE_COOL, target=23.0, night_ventilate=True)
    night = HouseSnapshot(
        now_ts=NOW,
        local_hour=23.0,
        settings=settings,
        zones=zones,
        t_out=24.5,  # within NIGHT_SKIP_CONT_K of coolest (23)
    )
    st = warmed(zones)
    d = controller.tick(night, st)
    assert d.diag["regime"] == "cycling"


def test_coast_aware_park_entry():
    zones = [
        make_zone(
            "sat",
            23.0,
            is_on=True,
            standing_load_w=100.0,
            park_trickles=True,
            park_extraction_w=200.0,
            park_coast_margin_k=2.5,
        ),
        make_zone("hot", 26.0, is_on=True, standing_load_w=200.0),
    ]
    st = warmed(zones)
    settings = Settings(hvac_mode=MODE_COOL, target=23.0, auto_regime=False)
    d = controller.tick(snap(zones, t_out=30.0, settings=settings), st)
    cmd = next((c for c in d.commands if c.zone_id == "sat" and c.park), None)
    assert cmd is not None
    assert cmd.park_margin == 2.5
