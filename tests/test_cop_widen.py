"""COP-timed efficiency band widen (center fixed)."""

from custom_components.adaptive_comfort.core import comfort, controller
from custom_components.adaptive_comfort.core.types import (
    MODE_COOL,
    MODE_HEAT,
    ControllerState,
    HouseSnapshot,
    Settings,
    ZoneSnapshot,
)

NOW = 1_000_000.0


def make_zone(zone_id, temp, **kw):
    kw.setdefault("free_float", tuple([temp] * 24))
    kw.setdefault("confidence", 0.9)
    return ZoneSnapshot(
        zone_id=zone_id,
        name=zone_id,
        n_rooms=1,
        temp=temp,
        **kw,
    )


def snap(t_out, forecast, cop_by_band=None, settings=None, temp=24.0, mode=MODE_COOL, **kw):
    band = cop_by_band or {}
    return HouseSnapshot(
        now_ts=NOW,
        local_hour=12.0,
        settings=settings or Settings(hvac_mode=mode, target=23.0, adaptive_blend=0.0),
        zones=[make_zone("z1", temp, is_on=True)],
        t_out=t_out,
        forecast_hours=tuple(forecast),
        cop_by_band=band,
        cop_by_band_mode=mode if band else None,
        **kw,
    )


def test_arbitrage_advance_when_hot_ahead():
    s = snap(24.0, [26.0, 33.0, 34.0], cop_by_band={"mild": 2.4, "hot": 1.0})
    timing, ratio = controller._forecast_cop_arbitrage(s, MODE_COOL, {"z1": 23.0})
    assert timing == "advance"
    assert ratio >= controller.COP_ADVANTAGE_ENTER


def test_arbitrage_defer_when_milder_ahead():
    s = snap(31.0, [28.0, 24.0, 23.0], cop_by_band={"mild": 2.4, "hot": 1.0})
    timing, ratio = controller._forecast_cop_arbitrage(s, MODE_COOL, {"z1": 23.0})
    assert timing == "defer"
    assert ratio >= controller.COP_ADVANTAGE_ENTER


def test_no_arbitrage_in_ventilate_territory():
    s = snap(21.0, [33.0, 34.0], cop_by_band={"mild": 2.4, "hot": 1.0})
    timing, _ = controller._forecast_cop_arbitrage(s, MODE_COOL, {"z1": 23.0})
    assert timing == "none"


def test_widen_engages_and_widens_bands():
    # Non-Boost: symmetric widen both edges.
    s = snap(
        24.0,
        [33.0, 34.0],
        cop_by_band={"mild": 2.4, "hot": 1.0},
        settings=Settings(hvac_mode=MODE_COOL, target=23.0, adaptive_blend=0.0),
    )
    st = ControllerState()
    st.zone_since["z1"] = NOW - 7200.0
    st.zone_on["z1"] = True
    st.mode_since = NOW - 24 * 3600.0
    d = controller.tick(s, st)
    assert d.diag.get("cop_timing") == "advance"
    assert d.diag.get("cop_band_widen_k") == controller.COP_WIDEN_MAX_K
    lo, hi = d.diag["bands"]["z1"]
    center = comfort.band_center(s.settings, None)
    base_lo, base_hi = comfort.zone_band(s.settings, center, True, None)
    assert abs((base_lo - lo) - controller.COP_WIDEN_MAX_K) < 1e-6
    assert abs((hi - base_hi) - controller.COP_WIDEN_MAX_K) < 1e-6
    assert abs(((lo + hi) / 2.0) - center) < 1e-6


def test_boost_widen_stretches_cool_lo_only():
    s = snap(
        24.0,
        [33.0, 34.0],
        cop_by_band={"mild": 2.4, "hot": 1.0},
        settings=Settings(hvac_mode=MODE_COOL, target=23.0, adaptive_blend=0.0, preset="boost"),
    )
    st = ControllerState()
    st.zone_since["z1"] = NOW - 7200.0
    st.zone_on["z1"] = True
    st.mode_since = NOW - 24 * 3600.0
    d = controller.tick(s, st)
    assert d.diag.get("cop_timing") == "advance"
    lo, hi = d.diag["bands"]["z1"]
    center = comfort.band_center(s.settings, None)
    base_lo, base_hi = comfort.zone_band(s.settings, center, True, None)
    assert abs((base_lo - lo) - controller.COP_WIDEN_MAX_K) < 1e-6
    assert abs(hi - base_hi) < 1e-6  # Boost far edge stays tight


def test_widen_hysteresis_holds_below_enter():
    s = snap(24.0, [33.0], cop_by_band={"mild": 2.4, "hot": 1.0}, temp=23.0)
    st = ControllerState()
    # Ratio 2.4/1.0 = 2.4 → engage
    w, t = controller._update_cop_widen(st, s, MODE_COOL, {"z1": 23.0}, s.zones)
    assert w == controller.COP_WIDEN_MAX_K and t == "advance"
    assert st.cop_widen_mode == MODE_COOL
    # Soften advantage to between EXIT and ENTER: still hold.
    soft = snap(24.0, [33.0], cop_by_band={"mild": 1.25, "hot": 1.0}, temp=23.0)
    # 1.25/1.0 = 1.25 >= EXIT 1.15, < ENTER 1.3
    w2, t2 = controller._update_cop_widen(st, soft, MODE_COOL, {"z1": 23.0}, soft.zones)
    assert w2 == controller.COP_WIDEN_MAX_K and t2 == "advance"
    # Drop below EXIT with room back inside base band: withdraw.
    weak = snap(24.0, [33.0], cop_by_band={"mild": 1.1, "hot": 1.0}, temp=23.0)
    w3, t3 = controller._update_cop_widen(st, weak, MODE_COOL, {"z1": 23.0}, weak.zones)
    assert w3 == 0.0 and t3 == "none"
    assert st.cop_widen_mode is None


def test_widen_holds_until_room_crosses_base_lo():
    """Cool-advance bank below tight lo must not snap band (→ false heat)."""
    st = ControllerState()
    engage = snap(24.0, [33.0], cop_by_band={"mild": 2.4, "hot": 1.0}, temp=23.0)
    controller._update_cop_widen(st, engage, MODE_COOL, {"z1": 23.0}, engage.zones)
    assert st.cop_timing == "advance"
    # Advantage gone, but room still below base lo (center 23, half 0.7 → lo 22.3).
    weak = snap(24.0, [33.0], cop_by_band={"mild": 1.1, "hot": 1.0}, temp=22.0)
    w, t = controller._update_cop_widen(st, weak, MODE_COOL, {"z1": 23.0}, weak.zones)
    assert w == controller.COP_WIDEN_MAX_K and t == "advance"
    assert st.cop_widen_mode == MODE_COOL
    # Cross back above base lo → may withdraw.
    recovered = snap(24.0, [33.0], cop_by_band={"mild": 1.1, "hot": 1.0}, temp=22.5)
    w2, t2 = controller._update_cop_widen(st, recovered, MODE_COOL, {"z1": 23.0}, recovered.zones)
    assert w2 == 0.0 and t2 == "none"


def test_auto_does_not_heat_while_cool_advance_banked():
    """Room below tight lo but inside widened lo must stay cool in summer."""
    settings = Settings(hvac_mode="auto", target=23.0, adaptive_blend=0.0, band_k=0.7)
    # Banked below base lo (~22.3) but above widened lo (~21.6).
    s = snap(
        24.0,
        [33.0, 34.0],
        cop_by_band={"mild": 2.4, "hot": 1.0},
        settings=settings,
        temp=22.0,
        mode=MODE_COOL,
        t_rm=27.0,  # season_cool
    )
    st = ControllerState()
    st.mode = MODE_COOL
    st.mode_since = NOW - 24 * 3600.0
    st.zone_since["z1"] = NOW - 7200.0
    st.cop_widen_k = controller.COP_WIDEN_MAX_K
    st.cop_timing = "advance"
    st.cop_widen_mode = MODE_COOL
    d = controller.tick(s, st)
    assert d.diag.get("cop_timing") == "advance"
    assert d.diag.get("mode") != MODE_HEAT
    lo, _hi = d.diag["bands"]["z1"]
    assert lo < 22.0  # still widened under the room


def test_prediction_does_not_dig_past_far_edge():
    """Cool-advance prediction must not keep demand once temp ≤ lo.

    Field (Aug 3): free-float still foresaw a hi breach while West was already
    below the (widened) floor — demand stayed on down to ~20.4°C.
    """
    # Trajectory claims a hi breach at hour 2; room is already below lo.
    zone = make_zone(
        "z1",
        21.0,
        free_float=(21.0, 22.0, 24.5, 25.0) + (25.0,) * 20,
    )
    assert not controller._wants_conditioning(
        zone, 22.5, 23.9, MODE_COOL, 20.0, cop_timing="advance"
    )
    # Still in-band on the approach side: prediction may fire.
    zone_in = make_zone(
        "z1",
        23.2,
        free_float=(23.2, 23.5, 24.5, 25.0) + (25.0,) * 20,
    )
    assert controller._wants_conditioning(
        zone_in, 22.5, 23.9, MODE_COOL, 20.0, cop_timing="advance"
    )
    # Heat symmetric: already above hi → no more heat from prediction.
    zone_hot = make_zone(
        "z1",
        24.5,
        free_float=(24.5, 23.0, 21.0, 20.0) + (20.0,) * 20,
    )
    assert not controller._wants_conditioning(
        zone_hot, 22.0, 23.5, MODE_HEAT, 20.0, cop_timing="advance"
    )


def test_heat_advance_when_colder_band_ahead():
    # Now warm, colder (mild) ahead — heat COP now should beat mild if learned so.
    s = snap(
        27.0,
        [24.0, 22.0],
        cop_by_band={"warm": 3.0, "mild": 2.0},
        settings=Settings(hvac_mode=MODE_HEAT, target=21.0, adaptive_blend=0.0),
        temp=19.0,
        mode=MODE_HEAT,
    )
    timing, ratio = controller._forecast_cop_arbitrage(s, MODE_HEAT, {"z1": 21.0})
    assert timing == "advance"
    assert ratio >= controller.COP_ADVANTAGE_ENTER


def test_widen_holds_latched_direction_when_other_wins():
    # Latched advance with ratio still ≥ EXIT; defer briefly outranks but
    # stays below ENTER — must not snap the band narrow.
    st = ControllerState()
    engage = snap(24.0, [33.0], cop_by_band={"mild": 2.4, "hot": 1.0}, temp=23.0)
    controller._update_cop_widen(st, engage, MODE_COOL, {"z1": 23.0}, engage.zones)
    assert st.cop_timing == "advance" and st.cop_widen_k > 0.0
    # Now: advance 1.20 (hold), defer 1.22 (wins winner pick, < ENTER).
    # Forecast has both hotter and milder so both ratios fire.
    mixed = snap(
        28.0,  # warm
        [33.0, 24.0],  # hot ahead + mild ahead
        cop_by_band={"mild": 1.22, "warm": 1.0, "hot": 1.0 / 1.20},
        temp=23.0,
    )
    # warm/hot advance = 1.0 / (1/1.20) = 1.20; mild/warm defer = 1.22 / 1.0 = 1.22
    w, t = controller._update_cop_widen(st, mixed, MODE_COOL, {"z1": 23.0}, mixed.zones)
    assert w == controller.COP_WIDEN_MAX_K and t == "advance"


def test_band_cop_ignores_wrong_mode_table():
    # Snapshot still carries cool-season band COPs; heat mode must not use them.
    s = snap(
        27.0,
        [24.0],
        cop_by_band={"warm": 3.0, "mild": 1.0},
        mode=MODE_COOL,  # tag says cool
    )
    assert controller._band_cop(s, "warm", MODE_HEAT) is None
    assert controller._band_cop(s, "warm", MODE_COOL) == 3.0
    # Heat with no matching table → no cool-prior leak.
    assert controller._forecast_cop_arbitrage(s, MODE_HEAT, {"z1": 21.0}) == ("none", 0.0)


def test_prediction_horizon_stretches_on_advance():
    zone = make_zone("z1", 23.0, free_float=(23.0, 23.2, 23.5, 24.5, 25.0) + (25.0,) * 19)
    # Breach at hour 3; default horizon ~1h would miss; advance looks 4h.
    # Temp 23.0 is in-band (lo=22 hi=23.5) so demand prediction is allowed.
    assert controller._prediction_justifies_run(
        zone, 22.0, 23.5, MODE_COOL, 20.0, cop_timing="advance"
    )
    assert not controller._prediction_justifies_run(
        zone, 22.0, 23.5, MODE_COOL, 20.0, cop_timing="defer"
    )
