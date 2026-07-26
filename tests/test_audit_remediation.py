"""Tests for the runtime-honesty / outdoor / thermal audit remediation."""

from __future__ import annotations

import random

from custom_components.adaptive_comfort.core import controller, power
from custom_components.adaptive_comfort.core.controller import (
    WINDOW_GRACE_S,
    _select_regime,
    _window_would_help,
    quantize_setpoint,
)
from custom_components.adaptive_comfort.core.rls import RLS
from custom_components.adaptive_comfort.core.thermal import ThermalModel
from custom_components.adaptive_comfort.core.types import (
    MODE_AUTO,
    MODE_COOL,
    ControllerState,
    Settings,
)
from tests.test_controller import NOW, make_snapshot, make_zone, warmed_state


def test_quantize_setpoint_honours_device_step():
    assert quantize_setpoint(22.4, step=0.5) == 22.5
    assert quantize_setpoint(22.4, step=1.0) == 22.0
    assert quantize_setpoint(22.6, step=1.0) == 23.0
    assert quantize_setpoint(22.4, step=0.0) == 22.5  # invalid → 0.5


def test_baseline_learned_slot_no_median_fallback():
    model = power.BaselineModel()
    model.update(8.0, 300.0)
    assert model.value(8.0, fallback=False) == 300.0
    assert model.value(20.0, fallback=False) is None
    assert model.value(20.0, fallback=True) == 300.0


def test_estimated_active_ac_draw_empty_active_ignores_pac():
    assert power.estimated_active_ac_draw_w([], 800.0) == 800.0
    assert power.estimated_active_ac_draw_w([], None) == 0.0


def test_outdoor_airflow_ignores_k_mix():
    model = ThermalModel(volume_m3=30.0)
    model.fits[False].theta[0] = 0.4
    model.fits[False].theta[1] = 0.5
    model.fits[False].samples = 1000
    assert abs(model.outdoor_airflow_m3h(False) - 0.4 * 30.0) < 1e-9
    assert model.outdoor_airflow_m3h(False) < model.airflow_m3h


def test_update_c_eff_uses_door_open_ua():
    """Regime-scaled UA for open vs closed must differ (the update_c_eff input)."""
    model = ThermalModel(volume_m3=30.0)
    model.fits[True].theta[0] = 0.8
    model.fits[True].theta[1] = 0.05
    model.fits[True].samples = 1000
    model.fits[False].theta[0] = 0.15
    model.fits[False].theta[1] = 0.05
    model.fits[False].samples = 1000
    assert model.ua_out_w_per_k(True) > model.ua_out_w_per_k(False) * 2
    # Door-open path must not raise (uses ua_out_w_per_k, not closed-only props).
    model.update_c_eff(
        800.0,
        False,
        24.0,
        30.0,
        -3.0,
        12.0,
        door_open=True,
        t_house_other=22.0,
    )
    assert 1.0 <= model.furniture_factor <= 12.0


def test_rls_trace_capped_under_weak_excitation():
    rls = RLS(2, lam=0.998, p0=100.0, trace_max=500.0)
    for _ in range(5000):
        # Near-zero phi → weak excitation; forgetting would wind P up.
        rls.update([1e-9, 0.0], 0.0)
    assert rls.trace <= 500.0 + 1e-6


def test_ventilate_blocked_when_t_out_synthetic():
    zone = make_zone("bed", 26.0, standing_load_w=50.0)
    snap = make_snapshot(
        [zone],
        Settings(hvac_mode=MODE_AUTO, auto_regime=True),
        t_out=15.0,
        t_out_synthetic=True,
    )
    centers = {"bed": 22.5}
    assert _select_regime(snap, MODE_COOL, centers) != "ventilate"


def test_window_not_suggested_when_t_out_synthetic():
    zone = make_zone("bed", 25.5, occupied=True)
    assert not _window_would_help(zone, 18.0, MODE_COOL, True, t_out_synthetic=True)
    assert _window_would_help(zone, 18.0, MODE_COOL, True, t_out_synthetic=False)


def test_window_grace_survives_brief_disqualification():
    # Force cool so mode does not fall to off when the zone briefly enters band
    # (off clears window_suggest_* entirely).
    settings = Settings(hvac_mode=MODE_COOL, window_suggest=True, auto_regime=False)
    hot = make_zone("bed", 25.5, occupied=True)
    state = warmed_state([hot])
    controller.tick(make_snapshot([hot], settings, t_out=18.0), state)
    since = state.window_suggest_since["bed"]
    # Leave suggest pool by entering the comfort band.
    cool_enough = make_zone("bed", 22.5, occupied=True)
    controller.tick(
        make_snapshot([cool_enough], settings, t_out=18.0, now=NOW + 60.0),
        state,
    )
    assert state.window_suggest_since["bed"] == since
    assert "bed" in state.window_suggest_out_since
    # Hot again: same since (no re-arm); still inside grace → no cool.
    decision = controller.tick(
        make_snapshot([hot], settings, t_out=18.0, now=NOW + 120.0),
        state,
    )
    assert state.window_suggest_since["bed"] == since
    assert not any(c.hvac_mode == MODE_COOL for c in decision.commands)


def test_window_grace_rearms_after_sustained_out():
    settings = Settings(hvac_mode=MODE_COOL, window_suggest=True, auto_regime=False)
    hot = make_zone("bed", 25.5, occupied=True)
    state = warmed_state([hot])
    controller.tick(make_snapshot([hot], settings, t_out=18.0), state)
    cool_enough = make_zone("bed", 22.5, occupied=True)
    t_leave = NOW + 60.0
    controller.tick(make_snapshot([cool_enough], settings, t_out=18.0, now=t_leave), state)
    assert "bed" in state.window_suggest_out_since
    t_clear = t_leave + WINDOW_GRACE_S + 1.0
    controller.tick(make_snapshot([cool_enough], settings, t_out=18.0, now=t_clear), state)
    assert "bed" not in state.window_suggest_since
    t2 = t_clear + 60.0
    controller.tick(make_snapshot([hot], settings, t_out=18.0, now=t2), state)
    assert abs(state.window_suggest_since["bed"] - t2) < 1e-6


def test_controller_state_persists_restart_fields():
    st = ControllerState()
    st.zone_last_cool["a"] = 123.0
    st.zone_mode_changes["a"] = [1.0, 2.0]
    st.zone_park_ref["a"] = 24.0
    st.zone_fan["a"] = True
    st.window_suggest_since["a"] = 50.0
    st.window_suggest_out_since["a"] = 60.0
    restored = ControllerState.from_dict(st.to_dict())
    assert restored.zone_last_cool["a"] == 123.0
    assert restored.zone_mode_changes["a"] == [1.0, 2.0]
    assert restored.zone_park_ref["a"] == 24.0
    assert restored.zone_fan["a"] is True
    assert restored.window_suggest_since["a"] == 50.0
    assert restored.window_suggest_out_since["a"] == 60.0


def test_process_power_events_drops_concurrent_same_ts():
    """Coordinator isolation: concurrent switches must not each learn the step."""
    from custom_components.adaptive_comfort.coordinator import (
        EVENT_SETTLE_S,
        AdaptiveComfortRuntime,
    )
    from custom_components.adaptive_comfort.core.power import DrawEstimator
    from custom_components.adaptive_comfort.core.series import TimeSeries

    rt = AdaptiveComfortRuntime.__new__(AdaptiveComfortRuntime)
    rt.draws = DrawEstimator()
    rt.p_load_series = TimeSeries(horizon_s=3600.0)
    ts = 10_000.0
    for t in range(int(ts - 300), int(ts + 300), 10):
        # Step of ~700 W across the transition.
        rt.p_load_series.append(float(t), 200.0 if t < ts else 900.0)
    rt._pending_events = [
        {"ts": ts, "zone": "a", "mode": MODE_COOL, "event_id": 1},
        {"ts": ts, "zone": "b", "mode": MODE_COOL, "event_id": 2},
    ]
    rt._all_events = [(ts, "a", 1), (ts, "b", 2)]
    AdaptiveComfortRuntime._process_power_events(rt, ts + EVENT_SETTLE_S + 1.0)
    assert rt.draws.draw_w("a", MODE_COOL) is None
    assert rt.draws.draw_w("b", MODE_COOL) is None


def test_process_power_events_learns_solo_transition():
    from custom_components.adaptive_comfort.coordinator import (
        EVENT_SETTLE_S,
        AdaptiveComfortRuntime,
    )
    from custom_components.adaptive_comfort.core.power import DrawEstimator
    from custom_components.adaptive_comfort.core.series import TimeSeries

    rt = AdaptiveComfortRuntime.__new__(AdaptiveComfortRuntime)
    rt.draws = DrawEstimator()
    rt.p_load_series = TimeSeries(horizon_s=3600.0)
    ts = 10_000.0
    for t in range(int(ts - 300), int(ts + 300), 10):
        rt.p_load_series.append(float(t), 200.0 if t < ts else 900.0)
    rt._pending_events = [{"ts": ts, "zone": "a", "mode": MODE_COOL, "event_id": 1}]
    rt._all_events = [(ts, "a", 1)]
    AdaptiveComfortRuntime._process_power_events(rt, ts + EVENT_SETTLE_S + 1.0)
    assert abs(rt.draws.draw_w("a", MODE_COOL) - 700.0) < 1.0


def test_react_should_not_run_in_shed_hysteresis_band():
    """Demand between restore and start must not re-enter controller.tick."""
    from custom_components.adaptive_comfort.coordinator import AdaptiveComfortRuntime
    from custom_components.adaptive_comfort.core.power import DrawEstimator
    from custom_components.adaptive_comfort.core.types import ControllerState, Settings

    rt = AdaptiveComfortRuntime.__new__(AdaptiveComfortRuntime)
    rt.settings = Settings(contracted_kva=3.45, shed_start_pct=0.88, shed_restore_pct=0.75)
    rt.controller_state = ControllerState()
    rt.controller_state.shed["bed"] = NOW
    rt.controller_state.last_shed_action = NOW
    rt.zones = {}
    rt.draws = DrawEstimator()
    rt.p_demand = 2800.0  # between 0.75*3450≈2588 and 0.88*3450≈3036
    rt.grid_over_since = None
    rt.shed_urgent = False
    assert not AdaptiveComfortRuntime._shed_needed_now(rt)
    assert not AdaptiveComfortRuntime._react_should_run_control(rt)


def test_react_runs_on_shed_engage():
    from custom_components.adaptive_comfort.coordinator import AdaptiveComfortRuntime
    from custom_components.adaptive_comfort.core.types import ControllerState, Settings

    rt = AdaptiveComfortRuntime.__new__(AdaptiveComfortRuntime)
    rt.settings = Settings(contracted_kva=3.45, shed_start_pct=0.88)
    rt.controller_state = ControllerState()
    rt.zones = {}
    rt.p_demand = 3200.0
    rt.grid_over_since = NOW - 10.0
    rt.shed_urgent = False
    assert AdaptiveComfortRuntime._react_should_run_control(rt)


def test_park_learners_solo_stale_sensible_still_duty_samples():
    """Solo parks: electrical gate runs even when sensible is stale (idler path)."""
    from custom_components.adaptive_comfort.coordinator import (
        FIT_STEP_S,
        PARK_OBS_DELAY_S,
        AdaptiveComfortRuntime,
        ZoneRuntime,
    )
    from custom_components.adaptive_comfort.core.types import (
        ControllerState,
        RoomConfig,
        Settings,
        ZoneConfig,
    )

    cfg = ZoneConfig(
        zone_id="z",
        name="z",
        heads=("climate.a",),
        rooms=(RoomConfig(12.0, 2.5),),
    )
    zone = ZoneRuntime(cfg)
    zone.sensible_w = -500.0  # stale full-conditioning extraction
    zone.sensible_ts = NOW - FIT_STEP_S - 10.0
    rt = AdaptiveComfortRuntime.__new__(AdaptiveComfortRuntime)
    rt.settings = Settings()
    rt.controller_state = ControllerState()
    rt.controller_state.zone_parked_since["z"] = NOW - PARK_OBS_DELAY_S - 10.0
    rt.controller_state.zone_park_margin["z"] = 1.0
    rt.controller_state.mode = MODE_COOL
    rt.zones = {"z": zone}
    # Below fan floor → coast: must still observe (extraction 0), not skip.
    rt.p_load = 50.0
    rt.baseline = power.BaselineModel()
    rt.baseline.update(12.0, 40.0)
    # Pre-settle PowerDebounce past PARK_DUTY_DEBOUNCE_S.
    from custom_components.adaptive_comfort.core import park as park_mod

    floor = park_mod.fan_floor_w(1, per_head_w=rt.settings.fan_floor_per_head_w)
    pac = power.estimate_ac_power(50.0, 40.0)
    assert pac is not None and pac < floor
    zone.park_power.settle(NOW - 120.0, pac, floor_w=floor)
    zone.park_power.settle(NOW - 60.0, pac, floor_w=floor)
    samples_before = zone.park.samples
    AdaptiveComfortRuntime._update_park_learners(rt, NOW, 12.0)
    assert zone.park.samples == samples_before + 1
    assert (zone.park.extraction_w or 0.0) < 60.0


def test_park_learners_nonsolo_skips_stale_sensible():
    from custom_components.adaptive_comfort.coordinator import (
        FIT_STEP_S,
        PARK_OBS_DELAY_S,
        AdaptiveComfortRuntime,
        ZoneRuntime,
    )
    from custom_components.adaptive_comfort.core.types import (
        STATE_COOLING,
        ControllerState,
        RoomConfig,
        Settings,
        ZoneConfig,
    )

    def _zone(zid: str) -> ZoneRuntime:
        cfg = ZoneConfig(
            zone_id=zid,
            name=zid,
            heads=(f"climate.{zid}",),
            rooms=(RoomConfig(12.0, 2.5),),
        )
        return ZoneRuntime(cfg)

    parked = _zone("park")
    parked.sensible_w = -500.0
    parked.sensible_ts = NOW - FIT_STEP_S - 10.0
    sibling = _zone("sib")
    sibling.head_state = STATE_COOLING
    rt = AdaptiveComfortRuntime.__new__(AdaptiveComfortRuntime)
    rt.settings = Settings()
    rt.controller_state = ControllerState()
    rt.controller_state.zone_parked_since["park"] = NOW - PARK_OBS_DELAY_S - 10.0
    rt.controller_state.mode = MODE_COOL
    rt.zones = {"park": parked, "sib": sibling}
    rt.p_load = 900.0
    rt.baseline = power.BaselineModel()
    rt.baseline.update(12.0, 100.0)
    before = parked.park.samples
    AdaptiveComfortRuntime._update_park_learners(rt, NOW, 12.0)
    assert parked.park.samples == before


def test_clear_park_unexpressible_charges_abort():
    from custom_components.adaptive_comfort.core.controller import _clear_park_session
    from custom_components.adaptive_comfort.core.types import ControllerState

    state = ControllerState()
    state.zone_parked_since["z"] = NOW
    state.zone_park_probe_entry["z"] = 0
    _clear_park_session(state, "z", None, NOW + 10.0)
    assert "z" not in state.zone_parked_since
    assert state.zone_last_park_abort.get("z") == NOW + 10.0


def test_honest_pac_positive_without_active_heads():
    assert power.estimate_ac_power(900.0, 300.0) == 600.0
    assert power.estimate_ac_power(200.0, 300.0) == 0.0


def test_rls_still_tracks_with_excitation():
    rng = random.Random(3)
    rls = RLS(1, lam=0.98, trace_max=1e6)
    for _ in range(200):
        rls.update([1.0], 4.0 + rng.gauss(0, 0.01))
    assert abs(rls.theta[0] - 4.0) < 0.15


def test_moisture_schema_wipes_inflated_baseline():
    from custom_components.adaptive_comfort.coordinator import MOISTURE_SCHEMA, ZoneRuntime
    from custom_components.adaptive_comfort.core.types import RoomConfig, ZoneConfig

    cfg = ZoneConfig(
        zone_id="z",
        name="z",
        heads=("climate.a",),
        rooms=(RoomConfig(12.0, 2.5),),
    )
    zone = ZoneRuntime(cfg)
    zone.restore({"moisture_sources_kg_h": 1.5, "moisture_schema": 1})
    assert zone.moisture_sources_kg_h == 0.0
    zone.restore({"moisture_sources_kg_h": 1.5, "moisture_schema": MOISTURE_SCHEMA})
    assert zone.moisture_sources_kg_h == 1.5


def test_effective_forecast_reindexes():
    from custom_components.adaptive_comfort.coordinator import AdaptiveComfortRuntime

    runtime = AdaptiveComfortRuntime.__new__(AdaptiveComfortRuntime)
    runtime.forecast = [20.0 + i for i in range(24)]
    runtime._forecast_fetched_ts = 1_000_000.0
    runtime.t_out = 21.5
    # 1.5 h after fetch: index 0 = live t_out; h=1 → pos 2.5 → blend 22/23.
    out = AdaptiveComfortRuntime._effective_forecast(runtime, 1_000_000.0 + 1.5 * 3600.0)
    assert out[0] == 21.5
    assert abs(out[1] - 22.5) < 0.01
