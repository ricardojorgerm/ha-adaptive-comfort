"""Park behavior: estimator classification and controller park decisions."""

from custom_components.adaptive_comfort.core import controller
from custom_components.adaptive_comfort.core.park import (
    CLASSIFY_MIN_SAMPLES,
    MARGIN_SETTLE_ALPHA,
    ParkEstimator,
)
from custom_components.adaptive_comfort.core.types import (
    MODE_COOL,
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


def make_snapshot(zones, settings=None, now=NOW):
    return HouseSnapshot(
        now_ts=now,
        local_hour=12.0,
        settings=settings or Settings(hvac_mode=MODE_COOL, target=23.0),
        zones=zones,
        t_out=30.0,
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


def test_estimator_classifies_trickler():
    est = ParkEstimator()
    for _ in range(CLASSIFY_MIN_SAMPLES):
        est.update(150.0, True)
    assert est.trickles is True
    assert est.classification == "trickle"


def test_estimator_classifies_idler():
    est = ParkEstimator()
    for _ in range(CLASSIFY_MIN_SAMPLES):
        est.update(0.0, False)
    assert est.trickles is False


def test_estimator_unknown_until_enough_samples():
    est = ParkEstimator()
    est.update(150.0, True)
    assert est.trickles is None


def test_estimator_round_trips():
    est = ParkEstimator()
    for _ in range(CLASSIFY_MIN_SAMPLES):
        est.update(90.0, True)
    est.raise_preferred(2.0)
    restored = ParkEstimator.from_dict(est.to_dict())
    assert restored.trickles is est.trickles
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


def test_park_exit_clears_session_margin():
    satisfied, hot, state = _two_zone_setup(park_trickles=False)
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


def _two_zone_setup(satisfied_temp=23.0, park_trickles=None, **satisfied_kw):
    """Satisfied zone (was on, leaves demand) + hot sibling keeping compressor."""
    satisfied = make_zone(
        "sat", satisfied_temp, is_on=True, park_trickles=park_trickles, **satisfied_kw
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
    satisfied, hot, state = _two_zone_setup(park_trickles=False)
    decision = tick([satisfied, hot], state)
    cmd = find_cmd(decision, "sat")
    assert cmd is None or cmd.park is False
    assert "sat" not in state.zone_parked_since


def test_known_trickler_exploit_parks_when_load_covered():
    satisfied, hot, state = _two_zone_setup(
        park_trickles=True, park_extraction_w=150.0, standing_load_w=180.0
    )
    decision = tick([satisfied, hot], state)
    cmd = find_cmd(decision, "sat")
    assert cmd is not None and cmd.park is True


def test_known_trickler_not_parked_when_load_too_big():
    satisfied, hot, state = _two_zone_setup(
        park_trickles=True, park_extraction_w=50.0, standing_load_w=400.0
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
        park_trickles=True, park_extraction_w=200.0, standing_load_w=100.0
    )
    tick([satisfied, hot], state)  # exploitation park
    assert "sat" in state.zone_parked_since
    later = NOW + controller.PARK_MIN_DWELL_S + 60.0
    warm = make_zone("sat", 26.0, is_on=True, park_trickles=True, park_extraction_w=200.0)
    state.zone_last_cmd["sat"] = later - 3600.0
    decision = tick([warm, hot], state, now=later)
    assert "sat" not in state.zone_parked_since
    cmd = find_cmd(decision, "sat")
    assert cmd is not None and cmd.park is False and cmd.track_delta is not None


def test_parked_zone_refreshes_park_command_on_spacing():
    satisfied, hot, state = _two_zone_setup(
        park_trickles=True, park_extraction_w=200.0, standing_load_w=100.0
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
        park_trickles=True, park_extraction_w=400.0, standing_load_w=100.0
    )
    tick([satisfied, hot], state)
    assert "sat" in state.zone_parked_since
    soon = NOW + 120.0  # well inside PARK_MIN_DWELL_S
    # Band lo = target - band_k = 22.3; at the floor the session must live.
    at_floor = make_zone("sat", 22.3, is_on=True, park_trickles=True, park_extraction_w=400.0)
    state.zone_last_cmd["sat"] = soon - 3600.0
    tick([at_floor, hot], state, now=soon)
    assert "sat" in state.zone_parked_since
    # Past the buffer: release immediately (sibling still alive, off allowed).
    frozen = make_zone(
        "sat",
        22.3 - controller.PARK_OVERCOOL_BUFFER_K - 0.05,
        is_on=True,
        park_trickles=True,
    )
    later = soon + 60.0
    state.zone_last_cmd["sat"] = later - 3600.0
    decision = tick([frozen, hot], state, now=later)
    assert "sat" not in state.zone_parked_since
    cmd = find_cmd(decision, "sat")
    assert cmd is None or cmd.park is False


def test_margin_escalates_while_room_keeps_cooling():
    satisfied, hot, state = _two_zone_setup(
        park_trickles=True, park_extraction_w=200.0, standing_load_w=100.0
    )
    tick([satisfied, hot], state)
    assert state.zone_park_margin["sat"] == controller.PARK_MARGIN_K
    # Room fell 0.2 K but is still inside the band: deepen, stay parked.
    cooler = make_zone(
        "sat",
        23.0 - 0.2,
        is_on=True,
        park_trickles=True,
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
        park_trickles=True,
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
        park_trickles=True,
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
        park_trickles=True, park_extraction_w=200.0, standing_load_w=100.0
    )
    tick([satisfied, hot], state)
    state.zone_last_cmd["sat"] = NOW  # spacing not yet elapsed
    cooler = make_zone(
        "sat",
        23.0 - 0.2,
        is_on=True,
        park_trickles=True,
        park_extraction_w=200.0,
        standing_load_w=100.0,
    )
    decision = tick([cooler, hot], state, now=NOW + 60.0)
    cmd = find_cmd(decision, "sat")
    assert cmd is not None and cmd.park is True
    assert cmd.park_margin == controller.PARK_MARGIN_K + controller.PARK_MARGIN_STEP_K


def test_margin_exhaustion_idles_head():
    satisfied, hot, state = _two_zone_setup(
        park_trickles=True, park_extraction_w=200.0, standing_load_w=100.0
    )
    tick([satisfied, hot], state)
    state.zone_park_margin["sat"] = controller.PARK_MARGIN_MAX_K  # already maxed
    state.zone_park_ref["sat"] = 23.0
    cooler = make_zone(
        "sat",
        23.0 - 0.2,
        is_on=True,
        park_trickles=True,
        park_extraction_w=200.0,
        standing_load_w=100.0,
    )
    tick([cooler, hot], state, now=NOW + 60.0)
    assert "sat" not in state.zone_parked_since  # idled due to overcorrection


def test_margin_relaxes_when_room_drifts_back():
    satisfied, hot, state = _two_zone_setup(
        park_trickles=True, park_extraction_w=200.0, standing_load_w=100.0
    )
    tick([satisfied, hot], state)
    state.zone_park_margin["sat"] = 2.0
    state.zone_park_ref["sat"] = 23.0
    warmer = make_zone(
        "sat",
        23.0 + 0.2,
        is_on=True,
        park_trickles=True,
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
        park_trickles=True,
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
    hot_room = make_zone("sat", 27.0, is_on=True, park_trickles=True, park_extraction_w=200.0)
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
        park_trickles=True,
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
        park_trickles=True,
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


def test_runout_zone_parks_for_free_observation():
    """A zone wanting off but blocked by min_on parks (no sibling needed, no
    probe budget): the compressor runs regardless, so observation is free.
    This is the field case: East satisfied at ~25 C, sibling idle, min_on
    running out - previously it just waited and hard-cooled at a stale
    setpoint, then turned off having learned nothing."""
    zone = make_zone("z1", 23.0, is_on=True, head_mode=MODE_COOL)
    state = ControllerState()
    state.zone_on["z1"] = True
    state.zone_since["z1"] = NOW - 300.0  # inside min_on: off blocked
    state.mode_since = NOW - 24 * 3600.0
    decision = tick([zone], state)
    cmd = find_cmd(decision, "z1")
    assert cmd is not None and cmd.park is True
    assert "z1" in state.zone_parked_since
    assert "z1" not in state.zone_park_probe_entry  # free: no budget involved


def test_runout_park_survives_probe_window_while_min_on_blocks():
    zone = make_zone("z1", 23.0, is_on=True, head_mode=MODE_COOL)
    state = ControllerState()
    state.zone_on["z1"] = True
    state.zone_since["z1"] = NOW - 60.0  # just turned on: min_on has ~19 min left
    state.mode_since = NOW - 24 * 3600.0
    tick([zone], state)
    # Probe window (15 min) elapses but min_on (20 min) still blocks off:
    later = NOW + controller.PARK_PROBE_S + 30.0
    state.zone_last_cmd["z1"] = later - 3600.0
    tick([make_zone("z1", 23.0, is_on=True, head_mode=MODE_COOL)], state, now=later)
    assert "z1" in state.zone_parked_since, "off not allowed yet -> stay parked"


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
    state.zone_since["z1"] = NOW - 300.0
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
    from custom_components.adaptive_comfort.core.park import gate_observation

    # Solo park drawing fan-only power: extraction is phantom, forced to 0.
    ext, active = gate_observation(35.0, True, 250.0)
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
    from custom_components.adaptive_comfort.core.park import FAN_FLOOR_W, gate_observation

    # Two mirrored heads coasting ~120 W would look like compression with a
    # single-head floor (90 W); scaled floor must treat it as fan-only.
    coast = FAN_FLOOR_W * 2 - 20.0
    ext, active = gate_observation(coast, True, 250.0, n_heads=2)
    assert ext == 0.0 and active is False
    ext, active = gate_observation(FAN_FLOOR_W * 2 + 50.0, True, 250.0, n_heads=2)
    assert ext == 250.0 and active is True


def test_multi_room_park_exploit_uses_per_room_load():
    """Zone-total standing load must not block exploit when per-head covers it."""
    from custom_components.adaptive_comfort.core.controller import _park_exploit_ok

    # Two rooms: zone load 200 W, sensed-room extraction 120 W covers 100 W/room.
    zone = make_zone(
        "duo",
        23.0,
        n_rooms=2,
        park_trickles=True,
        park_extraction_w=120.0,
        standing_load_w=200.0,
    )
    assert _park_exploit_ok(zone, MODE_COOL) is True
    # Same numbers without n_rooms scaling would have failed (120 < 0.7*200).
    under = make_zone(
        "duo",
        23.0,
        n_rooms=2,
        park_trickles=True,
        park_extraction_w=50.0,
        standing_load_w=200.0,
    )
    assert _park_exploit_ok(under, MODE_COOL) is False


def test_multi_room_zone_parks_and_exploits_with_sibling():
    """Mirrored zone with adequate per-head trickle parks when a sibling runs."""
    sat = make_zone(
        "sat",
        23.0,
        is_on=True,
        n_rooms=2,
        park_trickles=True,
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
    # Deep margin: head coasts fan-only.
    for _ in range(CLASSIFY_MIN_SAMPLES):
        est.update(0.0, False, margin_k=2.5)
    assert est.margin_bins["1.0"][1] > 0.7  # compression duty high
    assert est.margin_bins["2.5"][1] < 0.3  # coast region
    assert est.coast_margin_k() == 2.5
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
    # Sustained coast then settles inactive.
    assert d.settle(t + 500.0, 40.0) is None
    assert d.settle(t + 500.0 + PARK_DUTY_DEBOUNCE_S, 40.0) is False
    d.reset()
    assert d.above is None and d.since is None
