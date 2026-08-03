"""Park behavior: estimator classification and controller park decisions."""

from dataclasses import replace

from custom_components.adaptive_comfort.core import controller
from custom_components.adaptive_comfort.core.park import (
    CLASSIFY_MIN_SAMPLES,
    MARGIN_SETTLE_ALPHA,
    ParkEstimator,
    pick_depth_k,
    shallowest_covering_margin_k,
)
from custom_components.adaptive_comfort.core.types import (
    MODE_COOL,
    MODE_HEAT,
    ControllerState,
    HouseSnapshot,
    Settings,
    ZoneSnapshot,
)

NOW = 1_000_000.0


def make_zone(zone_id, temp, is_on=False, n_rooms=1, **kw):
    return ZoneSnapshot(
        zone_id=zone_id,
        name=zone_id,
        n_rooms=n_rooms,
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
    return controller.tick(make_snapshot(zones, settings=settings, now=now), state)


def find_cmd(decision, zid):
    return next((c for c in decision.commands if c.zone_id == zid), None)


# --- estimator -----------------------------------------------------------------


def test_estimator_classifies_residual_head():
    est = ParkEstimator()
    for _ in range(CLASSIFY_MIN_SAMPLES):
        est.update(150.0, True)
    assert est.residuals is True
    assert est.classification == "residual"


def test_estimator_classifies_idler():
    est = ParkEstimator()
    for _ in range(CLASSIFY_MIN_SAMPLES):
        est.update(0.0, False)
    assert est.residuals is False


def test_estimator_unknown_until_enough_samples():
    est = ParkEstimator()
    est.update(150.0, True)
    assert est.residuals is None


def test_estimator_round_trips():
    est = ParkEstimator()
    for _ in range(CLASSIFY_MIN_SAMPLES):
        est.update(90.0, True)
    est.raise_preferred(2.0)
    restored = ParkEstimator.from_dict(est.to_dict())
    assert restored.residuals is est.residuals
    assert restored.samples == est.samples
    assert restored.preferred_margin_k == 2.0


def test_estimator_settle_preferred_blends_down():
    est = ParkEstimator()
    est.preferred_margin_k = 2.0
    est.settle_preferred(1.0)
    expected = 2.0 + MARGIN_SETTLE_ALPHA * (1.0 - 2.0)
    assert abs(est.preferred_margin_k - expected) < 1e-9


def test_park_preferred_state_round_trips():
    state = ControllerState()
    state.zone_park_preferred["z1"] = 2.0
    state.zone_park_probe_entry["z1"] = 3
    state.zone_last_park_abort["z1"] = 123.0
    restored = ControllerState.from_dict(state.to_dict())
    assert restored.zone_park_preferred == {"z1": 2.0}
    assert restored.zone_park_probe_entry == {"z1": 3}
    assert restored.zone_last_park_abort == {"z1": 123.0}


def _residual_bins(**ext_by_margin):
    """Build well-sampled residual-duty margin_bins for pick_depth tests."""
    bins = {}
    for m, ext in ext_by_margin.items():
        bins[f"{float(m):.1f}"] = [float(ext), 1.0, CLASSIFY_MIN_SAMPLES]
    return bins


def test_shallowest_covering_margin():
    bins = _residual_bins(**{"1.5": 50.0, "2.0": 120.0, "3.0": 280.0})
    assert shallowest_covering_margin_k(bins, 100.0) == 2.0
    assert shallowest_covering_margin_k(bins, 300.0) is None


def test_pick_depth_pull_down_uses_chase():
    bins = _residual_bins(**{"2.0": 200.0})
    depth, tag = pick_depth_k(
        mode=MODE_COOL,
        temp=26.0,
        lo=22.0,
        hi=24.0,
        standing_load_w=100.0,
        n_rooms=1,
        park_residuals=True,
        margin_bins=bins,
        residual_edge_k=2.0,
        track_delta=0.7,
    )
    assert depth == -0.7 and tag == "depth_track"


def test_pick_depth_residual_when_bins_cover():
    bins = _residual_bins(**{"1.5": 97.0, "3.0": 289.0})
    depth, tag = pick_depth_k(
        mode=MODE_COOL,
        temp=23.2,
        lo=22.5,
        hi=23.9,
        standing_load_w=100.0,  # cover needs 60 W/head
        n_rooms=1,
        park_residuals=True,
        margin_bins=bins,
        residual_edge_k=3.0,
        track_delta=0.7,
    )
    # Edge bin covers → prefer residual_edge.
    assert depth == 3.0 and tag == "depth_residual"


def test_pick_depth_shallowest_when_edge_does_not_cover():
    bins = _residual_bins(**{"1.5": 100.0, "3.0": 40.0})
    depth, tag = pick_depth_k(
        mode=MODE_COOL,
        temp=23.2,
        lo=22.5,
        hi=23.9,
        standing_load_w=100.0,  # need 60 W
        n_rooms=1,
        park_residuals=True,
        margin_bins=bins,
        residual_edge_k=3.0,  # only 40 W — insufficient
        track_delta=1.0,
    )
    assert depth == 1.5 and tag == "depth_residual"


def test_pick_depth_heat_symmetry():
    bins = _residual_bins(**{"2.0": 150.0})
    depth, tag = pick_depth_k(
        mode=MODE_HEAT,
        temp=21.0,
        lo=20.0,
        hi=22.0,
        standing_load_w=80.0,
        n_rooms=1,
        park_residuals=True,
        margin_bins=bins,
        residual_edge_k=2.0,
        track_delta=0.5,
    )
    assert depth == 2.0 and tag == "depth_residual"
    # Above hi → chase only.
    depth2, tag2 = pick_depth_k(
        mode=MODE_HEAT,
        temp=22.5,
        lo=20.0,
        hi=22.0,
        standing_load_w=80.0,
        n_rooms=1,
        park_residuals=True,
        margin_bins=bins,
        residual_edge_k=2.0,
        track_delta=0.5,
    )
    assert depth2 == -0.5 and tag2 == "depth_track"


def test_demand_uses_residual_depth_without_want_off():
    """In-band demand with covering bins → positive head depth (not off-path park)."""
    bins = _residual_bins(**{"2.0": 200.0})
    zone = replace(
        make_zone(
            "z1",
            23.5,
            is_on=True,
            park_residuals=True,
            park_extraction_w=200.0,
            park_residual_edge_k=2.0,
            park_margin_bins=bins,
            standing_load_w=100.0,
        ),
        free_float=tuple([24.5] * 24),  # prediction keeps cool demand
    )
    settings = Settings(
        hvac_mode=MODE_COOL,
        target=23.0,
        adaptive_blend=0.0,
        band_k=1.0,
        tracking=True,
        park_learning=True,
        multisplit=True,
    )
    state = warmed_state([zone])
    state.zone_on["z1"] = True
    d = tick([zone], state, settings=settings)
    assert "z1" in (d.diag.get("demand") or [])
    cmd = find_cmd(d, "z1")
    assert cmd is not None
    assert cmd.park is True
    assert cmd.head_depth_k == 2.0
    assert cmd.reason == "depth_residual"
    assert "z1" in state.zone_parked_since


def test_demand_pull_down_stays_on_track_delta():
    bins = _residual_bins(**{"2.0": 200.0})
    zone = make_zone(
        "z1",
        26.0,
        is_on=True,
        park_residuals=True,
        park_extraction_w=200.0,
        park_residual_edge_k=2.0,
        park_margin_bins=bins,
        standing_load_w=100.0,
    )
    settings = Settings(
        hvac_mode=MODE_COOL,
        target=23.0,
        adaptive_blend=0.0,
        band_k=1.0,
        tracking=True,
        park_learning=True,
    )
    state = warmed_state([zone])
    state.zone_on["z1"] = True
    d = tick([zone], state, settings=settings)
    cmd = find_cmd(d, "z1")
    assert cmd is not None
    assert cmd.park is False
    assert cmd.head_depth_k is not None and cmd.head_depth_k < 0
    assert cmd.track_delta is not None


def test_park_exit_clears_session_margin():
    satisfied, hot, state = _two_zone_setup(park_residuals=False)
    # Known idler: enters briefly? Actually idler doesn't park — force a park
    # then release via probe window with unknown→use unclassified probe path.
    satisfied, hot, state = _two_zone_setup()
    tick([satisfied, hot], state)
    assert "sat" in state.zone_park_margin
    later = NOW + controller.PARK_PROBE_S + 60.0
    state.zone_last_cmd["sat"] = later - 3600.0
    tick([satisfied, hot], state, now=later)
    assert "sat" not in state.zone_park_margin
    assert "sat" not in state.zone_parked_since
    assert "sat" not in state.zone_park_ref


# --- controller: probe entry ---------------------------------------------------


def _two_zone_setup(satisfied_temp=23.0, park_residuals=None, **satisfied_kw):
    """Satisfied zone (was on, leaves demand) + hot sibling keeping compressor."""
    satisfied = make_zone(
        "sat", satisfied_temp, is_on=True, park_residuals=park_residuals, **satisfied_kw
    )
    hot = make_zone("hot", 27.0, is_on=True)
    state = warmed_state([satisfied, hot])
    return satisfied, hot, state


def test_unclassified_satisfied_zone_gets_probe_parked():
    satisfied, hot, state = _two_zone_setup()
    decision = tick([satisfied, hot], state)
    cmd = find_cmd(decision, "sat")
    assert cmd is not None and cmd.park is True
    assert "sat" in state.zone_parked_since
    # Probe budget is charged at release (only if observed), not at entry.
    assert "sat" not in state.zone_last_park_probe
    assert state.zone_park_probe_entry["sat"] == 0


def test_probe_rate_limited():
    satisfied, hot, state = _two_zone_setup()
    state.zone_last_park_probe["sat"] = NOW - 60.0  # probed a minute ago
    decision = tick([satisfied, hot], state)
    cmd = find_cmd(decision, "sat")
    assert cmd is None or cmd.park is False  # plain off path, no new probe


def test_known_idler_turns_off_not_parked():
    satisfied, hot, state = _two_zone_setup(park_residuals=False)
    decision = tick([satisfied, hot], state)
    cmd = find_cmd(decision, "sat")
    assert cmd is None or cmd.park is False
    assert "sat" not in state.zone_parked_since


def test_known_residual_exploit_parks_when_load_covered():
    satisfied, hot, state = _two_zone_setup(
        park_residuals=True, park_extraction_w=150.0, standing_load_w=180.0
    )
    decision = tick([satisfied, hot], state)
    cmd = find_cmd(decision, "sat")
    assert cmd is not None and cmd.park is True


def test_known_residual_not_parked_when_load_too_big():
    satisfied, hot, state = _two_zone_setup(
        park_residuals=True, park_extraction_w=50.0, standing_load_w=400.0
    )
    decision = tick([satisfied, hot], state)
    cmd = find_cmd(decision, "sat")
    assert cmd is None or cmd.park is False


def test_no_park_without_running_sibling():
    # Compressor would stop anyway: parking has no basis, plain off.
    satisfied = make_zone("sat", 23.0, is_on=True)
    state = warmed_state([satisfied])
    tick([satisfied], state)
    assert "sat" not in state.zone_parked_since


def test_shed_zone_never_parks():
    satisfied, hot, state = _two_zone_setup()
    state.shed["sat"] = NOW
    tick([satisfied, hot], state)
    assert "sat" not in state.zone_parked_since


def test_park_learning_disabled_falls_back_to_off():
    settings = Settings(hvac_mode=MODE_COOL, target=23.0, park_learning=False)
    satisfied, hot, state = _two_zone_setup()
    tick([satisfied, hot], state, settings=settings)
    assert "sat" not in state.zone_parked_since


# --- controller: parked-state lifecycle ---------------------------------------


def test_parked_zone_released_after_probe_window():
    satisfied, hot, state = _two_zone_setup()
    tick([satisfied, hot], state)  # enters probe park at NOW
    later = NOW + controller.PARK_PROBE_S + 60.0
    state.zone_last_cmd["sat"] = later - 3600.0
    tick([satisfied, hot], state, now=later)
    assert "sat" not in state.zone_parked_since  # probe over, released to off


def test_parked_zone_reenters_demand_when_out_of_band():
    satisfied, hot, state = _two_zone_setup(
        park_residuals=True, park_extraction_w=200.0, standing_load_w=100.0
    )
    tick([satisfied, hot], state)  # exploitation park
    assert "sat" in state.zone_parked_since
    later = NOW + controller.PARK_MIN_DWELL_S + 60.0
    warm = make_zone("sat", 26.0, is_on=True, park_residuals=True, park_extraction_w=200.0)
    state.zone_last_cmd["sat"] = later - 3600.0
    decision = tick([warm, hot], state, now=later)
    assert "sat" not in state.zone_parked_since
    cmd = find_cmd(decision, "sat")
    assert cmd is not None and cmd.park is False and cmd.track_delta is not None


def test_parked_zone_refreshes_park_command_on_spacing():
    satisfied, hot, state = _two_zone_setup(
        park_residuals=True, park_extraction_w=200.0, standing_load_w=100.0
    )
    tick([satisfied, hot], state)
    later = NOW + controller.COMMAND_SPACING_S + 10.0
    decision = tick([satisfied, hot], state, now=later)
    cmd = find_cmd(decision, "sat")
    assert cmd is not None and cmd.park is True  # setpoint re-rides internal


# --- overcorrection, margin escalation, heating symmetry -----------------------


def test_parked_zone_released_when_pushed_through_far_edge():
    """Non-idling head out-cools the load: release only below lo - buffer
    (at the floor itself must survive — that is the field bug)."""
    satisfied, hot, state = _two_zone_setup(
        park_residuals=True, park_extraction_w=400.0, standing_load_w=100.0
    )
    tick([satisfied, hot], state)
    assert "sat" in state.zone_parked_since
    soon = NOW + 120.0  # well inside PARK_MIN_DWELL_S
    # Band lo = target - band_k = 22.3; at the floor the session must live.
    at_floor = make_zone("sat", 22.3, is_on=True, park_residuals=True, park_extraction_w=400.0)
    state.zone_last_cmd["sat"] = soon - 3600.0
    tick([at_floor, hot], state, now=soon)
    assert "sat" in state.zone_parked_since
    # Past the buffer: release immediately (sibling still alive, off allowed).
    frozen = make_zone(
        "sat",
        22.3 - controller.PARK_OVERCOOL_BUFFER_K - 0.05,
        is_on=True,
        park_residuals=True,
    )
    later = soon + 60.0
    state.zone_last_cmd["sat"] = later - 3600.0
    decision = tick([frozen, hot], state, now=later)
    assert "sat" not in state.zone_parked_since
    cmd = find_cmd(decision, "sat")
    assert cmd is None or cmd.park is False


def test_margin_escalates_while_room_keeps_cooling():
    satisfied, hot, state = _two_zone_setup(
        park_residuals=True, park_extraction_w=200.0, standing_load_w=100.0
    )
    tick([satisfied, hot], state)
    assert state.zone_park_margin["sat"] == controller.PARK_MARGIN_K
    # Room fell 0.2 K but is still inside the band: deepen, stay parked.
    cooler = make_zone(
        "sat",
        23.0 - 0.2,
        is_on=True,
        park_residuals=True,
        park_extraction_w=200.0,
        standing_load_w=100.0,
    )
    later = NOW + controller.COMMAND_SPACING_S + 10.0
    decision = tick([cooler, hot], state, now=later)
    assert state.zone_park_margin["sat"] == controller.PARK_MARGIN_K + controller.PARK_MARGIN_STEP_K
    assert state.zone_park_preferred["sat"] == state.zone_park_margin["sat"]
    cmd = find_cmd(decision, "sat")
    assert cmd is not None and cmd.park is True
    assert cmd.park_margin == state.zone_park_margin["sat"]


def test_next_park_starts_slightly_below_preferred():
    """Entry undershoots preferred by one step (still >= PARK_MARGIN_K)."""
    satisfied, hot, state = _two_zone_setup(
        park_residuals=True,
        park_extraction_w=200.0,
        standing_load_w=100.0,
        park_preferred_margin_k=2.0,
    )
    decision = tick([satisfied, hot], state)
    cmd = find_cmd(decision, "sat")
    assert cmd is not None and cmd.park is True
    expected = 2.0 - controller.PARK_ENTRY_UNDERSHOOT_K
    assert cmd.park_margin == expected
    assert state.zone_park_margin["sat"] == expected
    assert state.zone_park_preferred["sat"] == 2.0  # memory unchanged at entry


def test_entry_undershoot_floors_at_park_minimum():
    satisfied, hot, state = _two_zone_setup(
        park_residuals=True,
        park_extraction_w=200.0,
        standing_load_w=100.0,
        park_preferred_margin_k=controller.PARK_MARGIN_K,  # already at floor
    )
    decision = tick([satisfied, hot], state)
    cmd = find_cmd(decision, "sat")
    assert cmd is not None and cmd.park_margin == controller.PARK_MARGIN_K


def test_margin_change_refreshes_immediately():
    """Escalation must not wait for COMMAND_SPACING_S to re-command the head."""
    satisfied, hot, state = _two_zone_setup(
        park_residuals=True, park_extraction_w=200.0, standing_load_w=100.0
    )
    tick([satisfied, hot], state)
    state.zone_last_cmd["sat"] = NOW  # spacing not yet elapsed
    cooler = make_zone(
        "sat",
        23.0 - 0.2,
        is_on=True,
        park_residuals=True,
        park_extraction_w=200.0,
        standing_load_w=100.0,
    )
    decision = tick([cooler, hot], state, now=NOW + 60.0)
    cmd = find_cmd(decision, "sat")
    assert cmd is not None and cmd.park is True
    assert cmd.park_margin == controller.PARK_MARGIN_K + controller.PARK_MARGIN_STEP_K


def test_margin_exhaustion_idles_head():
    satisfied, hot, state = _two_zone_setup(
        park_residuals=True, park_extraction_w=200.0, standing_load_w=100.0
    )
    tick([satisfied, hot], state)
    state.zone_park_margin["sat"] = controller.PARK_MARGIN_MAX_K  # already maxed
    state.zone_park_ref["sat"] = 23.0
    cooler = make_zone(
        "sat",
        23.0 - 0.2,
        is_on=True,
        park_residuals=True,
        park_extraction_w=200.0,
        standing_load_w=100.0,
    )
    tick([cooler, hot], state, now=NOW + 60.0)
    assert "sat" not in state.zone_parked_since  # idled due to overcorrection


def test_margin_relaxes_when_room_drifts_back():
    satisfied, hot, state = _two_zone_setup(
        park_residuals=True, park_extraction_w=200.0, standing_load_w=100.0
    )
    tick([satisfied, hot], state)
    state.zone_park_margin["sat"] = 2.0
    state.zone_park_ref["sat"] = 23.0
    warmer = make_zone(
        "sat",
        23.0 + 0.2,
        is_on=True,
        park_residuals=True,
        park_extraction_w=200.0,
        standing_load_w=100.0,
    )
    tick([warmer, hot], state, now=NOW + 60.0)
    assert state.zone_park_margin["sat"] == 1.5


def test_heating_park_entry_and_far_edge_release():
    """Heat: satisfied zone parks while a cold sibling runs; pushing past
    band hi releases immediately (symmetric to cool far-edge)."""
    from custom_components.adaptive_comfort.core.types import MODE_HEAT

    settings = Settings(hvac_mode=MODE_HEAT, target=23.0)
    # Above center → not a heat helper; in-band → not demand → park candidate.
    satisfied = make_zone(
        "sat",
        23.6,
        is_on=True,
        park_residuals=True,
        park_extraction_w=200.0,
        standing_load_w=100.0,
    )
    cold = make_zone("cold", 19.0, is_on=True)  # sibling keeps compressor on
    state = warmed_state([satisfied, cold])
    decision = tick([satisfied, cold], state, settings=settings)
    cmd = find_cmd(decision, "sat")
    assert cmd is not None and cmd.park is True
    assert "sat" in state.zone_parked_since

    # Overheat past band hi (center 23 ± 0.7 → hi 23.7): release immediately.
    soon = NOW + 120.0
    hot_room = make_zone("sat", 27.0, is_on=True, park_residuals=True, park_extraction_w=200.0)
    state.zone_last_cmd["sat"] = soon - 3600.0
    decision = tick([hot_room, cold], state, settings=settings, now=soon)
    assert "sat" not in state.zone_parked_since
    cmd = find_cmd(decision, "sat")
    assert cmd is None or cmd.park is False


def test_heating_margin_escalates_while_room_keeps_warming():
    from custom_components.adaptive_comfort.core.types import MODE_HEAT

    settings = Settings(hvac_mode=MODE_HEAT, target=23.0)
    # Stay inside band (hi = 23.7) while still rising enough to escalate.
    satisfied = make_zone(
        "sat",
        23.3,
        is_on=True,
        park_residuals=True,
        park_extraction_w=200.0,
        standing_load_w=100.0,
    )
    cold = make_zone("cold", 19.0, is_on=True)
    state = warmed_state([satisfied, cold])
    tick([satisfied, cold], state, settings=settings)
    entry = state.zone_park_margin["sat"]
    warmer = make_zone(
        "sat",
        23.3 + 0.2,
        is_on=True,
        park_residuals=True,
        park_extraction_w=200.0,
        standing_load_w=100.0,
    )
    decision = tick([warmer, cold], state, settings=settings, now=NOW + 60.0)
    assert state.zone_park_margin["sat"] == entry + controller.PARK_MARGIN_STEP_K
    cmd = find_cmd(decision, "sat")
    assert cmd is not None and cmd.park is True
    assert cmd.park_margin == state.zone_park_margin["sat"]


# --- field-bug regressions: floor entry, probe budget, run-out ------------------


def test_park_entered_at_band_floor_survives():
    """Zones exit demand AT the floor, so entry temp == lo must not trip the
    overcorrection release (the field failure: every probe died in one tick)."""
    satisfied, hot, state = _two_zone_setup(satisfied_temp=23.0)
    tick([satisfied, hot], state)
    assert "sat" in state.zone_parked_since
    # Next tick, same temperature (at/near the floor, inside the buffer):
    later = NOW + 65.0
    tick([satisfied, hot], state, now=later)
    assert "sat" in state.zone_parked_since, "buffer must keep floor-parked zones alive"


def test_stillborn_probe_charges_retry_not_budget():
    satisfied, hot, state = _two_zone_setup()
    tick([satisfied, hot], state)  # probe park, park_samples == 0
    assert "sat" in state.zone_park_probe_entry
    # Probe window elapses with zero observations gathered:
    later = NOW + controller.PARK_PROBE_S + 60.0
    state.zone_last_cmd["sat"] = later - 3600.0
    tick([satisfied, hot], state, now=later)
    assert "sat" not in state.zone_parked_since
    assert "sat" not in state.zone_last_park_probe, "no observations -> budget unchanged"
    assert state.zone_last_park_abort.get("sat") == later
    # Immediately satisfied again: retry clock blocks a new probe...
    tick([satisfied, hot], state, now=later + 120.0)
    assert "sat" not in state.zone_parked_since
    # ...but after the short retry interval a new probe is allowed (the zone
    # has run again in the meantime, so it is on and past min_on).
    retry_at = later + controller.PARK_PROBE_RETRY_S + 60.0
    state.zone_on["sat"] = True
    state.zone_since["sat"] = retry_at - 3600.0
    state.zone_last_cmd["sat"] = retry_at - 3600.0
    tick([satisfied, hot], state, now=retry_at)
    assert "sat" in state.zone_parked_since


def test_observed_probe_charges_budget():
    satisfied, hot, state = _two_zone_setup()
    tick([satisfied, hot], state)  # probe park at samples == 0
    later = NOW + controller.PARK_PROBE_S + 60.0
    observed = make_zone("sat", 23.0, is_on=True, park_samples=4)
    state.zone_last_cmd["sat"] = later - 3600.0
    tick([observed, hot], state, now=later)
    assert "sat" not in state.zone_parked_since
    assert state.zone_last_park_probe.get("sat") == later
    assert "sat" not in state.zone_last_park_abort


def test_park_ok_now_keep_temp_only():
    hot = make_zone("hot", 27.0)
    ok = make_zone("ok", 23.0)
    assert controller._park_ok_now(hot, 21.8, 23.2, MODE_COOL) is False
    assert controller._park_ok_now(ok, 21.8, 23.2, MODE_COOL) is True


def test_runout_zone_parks_for_free_observation():
    """Plant min_on still running: satisfied zone residual-parks (no sibling
    needed). Zone chatter is short; the compressor run clock is what holds."""
    zone = make_zone("z1", 23.0, is_on=True, head_mode=MODE_COOL)
    state = ControllerState()
    state.zone_on["z1"] = True
    state.zone_since["z1"] = NOW - 300.0  # zone chatter cleared
    state.plant_compress_since = NOW - 300.0
    state.mode_since = NOW - 24 * 3600.0
    snap = make_snapshot([zone], now=NOW, p_ac=400.0, compression_floor_w=75.0)
    decision = controller.tick(snap, state)
    cmd = find_cmd(decision, "z1")
    assert cmd is not None and cmd.park is True
    assert "z1" in state.zone_parked_since
    assert "z1" not in state.zone_park_probe_entry  # free: no budget involved


def test_runout_does_not_park_while_still_hot():
    """Unfinished pull-down: mode idle + min_on must not fan-type park a hot room."""
    zone = make_zone("z1", 27.0, is_on=True, head_mode=MODE_COOL)
    zone.free_float = tuple([22.5] * 24)  # mode drops idle; room still hot
    state = ControllerState(mode=MODE_COOL, mode_since=0.0)
    state.zone_on["z1"] = True
    state.zone_since["z1"] = NOW - 300.0
    decision = tick([zone], state)
    assert "z1" not in state.zone_parked_since
    cmd = find_cmd(decision, "z1")
    assert cmd is None or cmd.park is False


def test_runout_park_survives_probe_window_while_min_on_blocks():
    zone = make_zone("z1", 23.0, is_on=True, head_mode=MODE_COOL)
    state = ControllerState()
    state.zone_on["z1"] = True
    state.zone_since["z1"] = NOW - 60.0
    state.plant_compress_since = NOW - 60.0
    state.mode_since = NOW - 24 * 3600.0
    snap = make_snapshot([zone], now=NOW, p_ac=400.0, compression_floor_w=75.0)
    controller.tick(snap, state)
    # Probe window elapses but plant min_on still owes runtime:
    later = NOW + controller.PARK_PROBE_S + 30.0
    state.zone_last_cmd["z1"] = later - 3600.0
    snap2 = make_snapshot(
        [make_zone("z1", 23.0, is_on=True, head_mode=MODE_COOL)],
        now=later,
        p_ac=400.0,
        compression_floor_w=75.0,
    )
    controller.tick(snap2, state)
    assert "z1" in state.zone_parked_since, "plant min_on active -> stay parked"


def test_runout_park_releases_when_off_allowed_and_no_sibling():
    zone = make_zone("z1", 23.0, is_on=True, head_mode=MODE_COOL)
    state = ControllerState()
    state.zone_on["z1"] = True
    state.zone_since["z1"] = NOW - 300.0
    state.mode_since = NOW - 24 * 3600.0
    tick([zone], state)
    # Past min_on and past dwell, no sibling wants on: release to off.
    later = NOW + 1500.0  # 25 min after zone_since: min_on satisfied
    state.zone_last_cmd["z1"] = later - 3600.0
    tick([make_zone("z1", 23.0, is_on=True, head_mode=MODE_COOL)], state, now=later)
    assert "z1" not in state.zone_parked_since


def test_runout_falls_back_to_min_delta_when_park_learning_off():
    from custom_components.adaptive_comfort.core.types import MODE_AUTO

    settings = Settings(hvac_mode=MODE_AUTO, target=23.0, park_learning=False)
    zone = make_zone("z1", 23.0, is_on=True, head_mode=MODE_COOL)
    state = ControllerState()
    state.zone_on["z1"] = True
    # Inside zone chatter: off blocked; park_learning off → tracked runout.
    state.zone_since["z1"] = NOW - 60.0
    state.mode_since = NOW - 24 * 3600.0
    snap = HouseSnapshot(now_ts=NOW, local_hour=12.0, settings=settings, zones=[zone], t_out=23.0)
    decision = controller.tick(snap, state)
    cmd = find_cmd(decision, "z1")
    if decision.state.mode not in ("cool", "heat"):
        assert cmd is not None and cmd.reason == "runout"
        assert cmd.track_delta == controller.TRACK_DELTA_MIN_K
    else:
        assert cmd is None or cmd.park is False


# --- power-gated observation and hysteresis learning ---------------------------


def test_gate_observation_solo_fan_floor():
    from custom_components.adaptive_comfort.core.park import fan_floor_w, gate_observation

    floor_1 = fan_floor_w(1)  # 20 + 55 = 75
    # Solo park drawing fan-only power: extraction is phantom, forced to 0.
    ext, active = gate_observation(floor_1 - 20.0, True, 250.0)
    assert ext == 0.0 and active is False
    # Solo park with real compression: extraction stands, active True.
    ext, active = gate_observation(320.0, True, 250.0)
    assert ext == 250.0 and active is True
    # Sibling conditioning (ambiguous power): judge by extraction magnitude.
    ext, active = gate_observation(600.0, False, 10.0)
    assert active is False
    ext, active = gate_observation(600.0, False, 200.0)
    assert active is True
    # No power reading at all: same magnitude fallback.
    ext, active = gate_observation(None, True, 200.0)
    assert active is True


def test_gate_observation_scales_fan_floor_with_heads():
    from custom_components.adaptive_comfort.core.park import fan_floor_w, gate_observation

    # Two-head fan-type park ~100-130 W must stay below the gate (20+55*2=130).
    floor_2 = fan_floor_w(2)
    assert floor_2 == 130.0
    ext, active = gate_observation(floor_2 - 10.0, True, 250.0, n_heads=2)
    assert ext == 0.0 and active is False
    ext, active = gate_observation(floor_2 + 50.0, True, 250.0, n_heads=2)
    assert ext == 250.0 and active is True
    # Tunable per-head: 20 + 70*2 = 160; 140 W fan-type park stays fan-only.
    ext, active = gate_observation(140.0, True, 250.0, n_heads=2, per_head_w=70.0)
    assert ext == 0.0 and active is False


def test_multi_room_park_exploit_uses_per_room_load():
    """Zone-total standing load must not block exploit when per-head covers it."""
    from custom_components.adaptive_comfort.core.controller import _park_exploit_ok

    # Two rooms: zone load 200 W, sensed-room extraction 120 W covers 100 W/room.
    zone = make_zone(
        "duo",
        23.0,
        n_rooms=2,
        park_residuals=True,
        park_extraction_w=120.0,
        standing_load_w=200.0,
    )
    assert _park_exploit_ok(zone, MODE_COOL) is True
    # Same numbers without n_rooms scaling would have failed (120 < 0.7*200).
    under = make_zone(
        "duo",
        23.0,
        n_rooms=2,
        park_residuals=True,
        park_extraction_w=50.0,
        standing_load_w=200.0,
    )
    assert _park_exploit_ok(under, MODE_COOL) is False


def test_multi_room_zone_parks_and_exploits_with_sibling():
    """Mirrored zone with adequate per-head residual parks when a sibling runs."""
    sat = make_zone(
        "sat",
        23.0,
        is_on=True,
        n_rooms=2,
        park_residuals=True,
        park_extraction_w=150.0,
        standing_load_w=200.0,  # 100 W/room; 150 covers it
    )
    hot = make_zone("hot", 26.0, is_on=True, standing_load_w=200.0)
    state = warmed_state([sat, hot])
    decision = tick([sat, hot], state)
    assert "sat" in state.zone_parked_since
    cmd = find_cmd(decision, "sat")
    assert cmd is not None and cmd.park is True


def test_margin_bins_learn_hysteresis_map():
    from custom_components.adaptive_comfort.core.park import (
        CLASSIFY_MIN_SAMPLES,
        ParkEstimator,
    )

    est = ParkEstimator()
    # Shallow margin: hysteresis keeps re-engaging compression.
    for _ in range(CLASSIFY_MIN_SAMPLES):
        est.update(300.0, True, margin_k=1.0)
    # Deep margin: head is fan-type (no compression).
    for _ in range(CLASSIFY_MIN_SAMPLES):
        est.update(0.0, False, margin_k=2.5)
    assert est.margin_bins["1.0"][1] > 0.7  # compression duty high
    assert est.margin_bins["2.5"][1] < 0.3  # fan-type region
    assert est.fan_type_min_margin_k() == 2.5
    assert est.fan_only_ratio is not None and 0.3 < est.fan_only_ratio < 0.7


def test_margin_bins_round_trip():
    from custom_components.adaptive_comfort.core.park import ParkEstimator

    est = ParkEstimator()
    for _ in range(8):
        est.update(150.0, True, margin_k=1.4)  # bins to "1.5"
    restored = ParkEstimator.from_dict(est.to_dict())
    assert "1.5" in restored.margin_bins
    assert restored.margin_bins["1.5"][2] == 8


def test_margin_bin_edges():
    from custom_components.adaptive_comfort.core.park import margin_bin

    assert margin_bin(None) is None
    assert margin_bin(1.0) == "1.0"
    assert margin_bin(1.24) == "1.0"
    assert margin_bin(1.26) == "1.5"
    assert margin_bin(3.0) == "3.0"


def test_control_state_key():
    from custom_components.adaptive_comfort.core.power import control_state_key

    assert control_state_key(True, False) == "conditioning"
    assert control_state_key(False, True) == "park"
    assert control_state_key(True, True) == "mixed"
    assert control_state_key(False, False) is None


# --- residual-hold vs fan-type: entry target, live coil-dry, settle bounds -----


def test_residual_hold_margin_prefers_deepest_compressing_bin():
    from custom_components.adaptive_comfort.core.park import ParkEstimator

    est = ParkEstimator()
    # 2.5 K: still meaningfully compressing.
    for _ in range(CLASSIFY_MIN_SAMPLES):
        est.update(300.0, True, margin_k=2.5)
    # 3.0 K: fan-type, no compression.
    for _ in range(CLASSIFY_MIN_SAMPLES):
        est.update(0.0, False, margin_k=3.0)
    assert est.residual_max_margin_k() == 2.5
    assert est.fan_type_min_margin_k() == 3.0
    # Edge bisects between the two to finer-than-0.5 K resolution.
    assert est.residual_edge_k() == 2.75


def test_residual_edge_falls_back_without_a_clean_pair():
    from custom_components.adaptive_comfort.core.park import ParkEstimator

    est = ParkEstimator()
    for _ in range(CLASSIFY_MIN_SAMPLES):
        est.update(300.0, True, margin_k=1.5)
    # Only a compressing bin is known: edge falls back to it.
    assert est.residual_edge_k() == 1.5
    assert est.fan_type_min_margin_k() is None


def test_current_is_fan_type_reads_live_margin_bin():
    from custom_components.adaptive_comfort.core.park import ParkEstimator

    est = ParkEstimator()
    for _ in range(CLASSIFY_MIN_SAMPLES):
        est.update(300.0, True, margin_k=1.5)
    for _ in range(CLASSIFY_MIN_SAMPLES):
        est.update(0.0, False, margin_k=3.0)
    assert est.current_is_fan_type(1.5) is False
    assert est.current_is_fan_type(3.0) is True
    assert est.current_is_fan_type(2.0) is None  # no evidence at that bin
    assert est.current_is_fan_type(None) is None


def test_park_entry_margin_prefers_residual_hold_over_fan_type():
    """Bins show 2.5 K still compressing / 3.0 K fan-type: entry targets the
    residual-hold depth, not the (shallower-cost but non-conditioning)
    fan-type shelf."""
    zone = make_zone(
        "sat",
        23.0,
        park_fan_type_min_margin_k=3.0,
        park_residual_edge_k=2.5,
    )
    state = ControllerState()
    assert controller._park_entry_margin(zone, state) == 2.5
    assert controller._park_preferred(zone, state) == 2.5


def test_park_entry_ignores_fan_margin_without_residual_hold():
    """No residual-hold evidence yet: entry undershoots preferred rather
    than jumping straight to the fan-type shelf."""
    zone = make_zone(
        "sat",
        23.0,
        park_fan_type_min_margin_k=3.0,
        park_preferred_margin_k=2.0,
    )
    state = ControllerState()
    assert controller._park_entry_margin(zone, state) == 2.0 - controller.PARK_ENTRY_UNDERSHOOT_K


def test_fan_type_park_released_within_coil_dry():
    """A live fan-type park session inside the coil-dry window prefers a
    true off, reusing fan_assist's wet-coil policy instead of holding a
    park that blows air over a wet coil for no compressor output."""
    satisfied, hot, state = _two_zone_setup()
    tick([satisfied, hot], state)  # probe park entered at NOW
    assert "sat" in state.zone_parked_since
    soon = NOW + 120.0  # well inside dwell/probe window
    state.zone_last_cool["sat"] = soon - 60.0  # cooled recently -> coil wet
    state.zone_last_cmd["sat"] = soon - 3600.0
    fan_type_zone = make_zone("sat", 23.0, is_on=True, park_current_is_fan_type=True)
    decision = tick([fan_type_zone, hot], state, now=soon)
    assert "sat" not in state.zone_parked_since
    cmd = find_cmd(decision, "sat")
    assert cmd is None or cmd.park is False


def test_residual_live_session_ignores_coil_dry():
    """A currently-compressing (residual) live session is conditioning, not
    a fan-type hold: COIL_DRY must not force it off."""
    satisfied, hot, state = _two_zone_setup(
        park_residuals=True, park_extraction_w=200.0, standing_load_w=100.0
    )
    tick([satisfied, hot], state)
    assert "sat" in state.zone_parked_since
    soon = NOW + 120.0
    state.zone_last_cool["sat"] = soon - 60.0  # coil wet, but session is residual
    state.zone_last_cmd["sat"] = soon - 3600.0
    residual_zone = make_zone(
        "sat",
        23.0,
        is_on=True,
        park_residuals=True,
        park_extraction_w=200.0,
        standing_load_w=100.0,
        park_current_is_fan_type=False,
    )
    tick([residual_zone, hot], state, now=soon)
    assert "sat" in state.zone_parked_since  # not released by coil-dry


def test_settle_does_not_blend_toward_overshoot_margin():
    """Out-of-band release is an overshoot/failed hold, not a proven
    residual-hold depth: preferred must not blend toward it. Before this
    fix an undershoot-then-relax sequence could even pull preferred *down*
    on a failed hold."""
    satisfied, hot, state = _two_zone_setup(
        park_residuals=True,
        park_extraction_w=200.0,
        standing_load_w=100.0,
        park_preferred_margin_k=2.0,
    )
    tick([satisfied, hot], state)  # entry undershoots to 1.5
    assert state.zone_park_preferred["sat"] == 2.0
    soon = NOW + 60.0
    warm = make_zone("sat", 26.0, is_on=True, park_residuals=True, park_extraction_w=200.0)
    state.zone_last_cmd["sat"] = soon - 3600.0
    tick([warm, hot], state, now=soon)
    assert "sat" not in state.zone_parked_since
    assert state.zone_park_preferred["sat"] == 2.0  # unchanged, not blended down


def test_relax_bounded_by_learned_residual_edge():
    """Relaxing on warming must not walk the margin down past the learned
    residual-hold edge into full-conditioning depths."""
    satisfied, hot, state = _two_zone_setup(
        park_residuals=True,
        park_extraction_w=200.0,
        standing_load_w=100.0,
        park_residual_edge_k=2.0,
    )
    tick([satisfied, hot], state)  # entry at the residual-hold edge: 2.0
    assert state.zone_park_margin["sat"] == 2.0
    warmer = make_zone(
        "sat",
        23.0 + 0.2,
        is_on=True,
        park_residuals=True,
        park_extraction_w=200.0,
        standing_load_w=100.0,
        park_residual_edge_k=2.0,
    )
    tick([warmer, hot], state, now=NOW + 60.0)
    # Would relax to 1.5 (2.0 - PARK_MARGIN_STEP_K) without the bound; the
    # learned residual edge holds it at 2.0 instead.
    assert state.zone_park_margin["sat"] == 2.0


def test_power_debounce_ignores_brief_crossings():
    from custom_components.adaptive_comfort.core.park import (
        PARK_DUTY_DEBOUNCE_S,
        PowerDebounce,
    )

    d = PowerDebounce()
    t = NOW
    assert d.settle(t, 320.0) is None  # first sighting arms, no sample yet
    assert d.settle(t + PARK_DUTY_DEBOUNCE_S - 1.0, 320.0) is None  # still settling
    assert d.settle(t + PARK_DUTY_DEBOUNCE_S, 320.0) is True
    # Brief dip below the floor: re-arm, no false inactive sample.
    assert d.settle(t + PARK_DUTY_DEBOUNCE_S + 60.0, 40.0) is None
    assert d.settle(t + PARK_DUTY_DEBOUNCE_S + 90.0, 320.0) is None  # flipped back
    # Sustained fan-type draw then settles inactive.
    assert d.settle(t + 500.0, 40.0) is None
    assert d.settle(t + 500.0 + PARK_DUTY_DEBOUNCE_S, 40.0) is False
    d.reset()
    assert d.above is None and d.since is None
