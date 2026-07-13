"""Decision engine: one pure tick from HouseSnapshot to zone commands.

The runtime (coordinator) builds a HouseSnapshot every 60 s and executes
the returned commands, translating room-coordinate setpoints into
device setpoints via the per-head drift offsets.
"""

from __future__ import annotations

from . import comfort, power
from .types import (
    MODE_AUTO,
    MODE_COOL,
    MODE_FAN,
    MODE_HEAT,
    MODE_OFF,
    STATE_COOLING,
    STATE_FAN_ONLY,
    Command,
    ControllerState,
    Decision,
    HouseSnapshot,
    ZoneSnapshot,
)

PREDICT_MARGIN_K = 0.1
HELPER_BAND_FRACTION = 0.5
MAX_MODE_CHANGES_PER_H = 3
SHED_ACTION_SPACING_S = 30.0
COMMAND_SPACING_S = 180.0
SETPOINT_EPSILON_K = 0.25
COP_TABLE_ADVANTAGE = 1.05
# Fan assist: a multi-split head cannot run opposite to the shared mode, but
# fan-only mixing can nudge an out-of-band room using house air. Running the
# fan over a wet coil re-evaporates condensate accumulated while cooling
# (latent-heat inefficiency: the moisture must be removed again later), so
# fan assist waits for the coil to dry after the last cooling run.
COIL_DRY_S = 30.0 * 60.0
FAN_ASSIST_MIN_DEV_K = 0.3
# Window suggestion: outdoors clearly colder than the room, someone present.
WINDOW_DELTA_K = 2.0
WINDOW_GRACE_S = 15.0 * 60.0


def quantize_setpoint(value: float, minimum: float = 16.0, maximum: float = 30.0) -> float:
    """Round to the 0.5 C steps AC heads accept, clamped to device limits."""
    return min(max(round(value * 2.0) / 2.0, minimum), maximum)


def _comfort_error(zone: ZoneSnapshot, center: float, mode: str) -> float:
    """How much the zone needs conditioning in the mode direction (K, >=0)."""
    if zone.temp is None:
        return 0.0
    if mode == MODE_COOL:
        return max(0.0, zone.temp - center)
    if mode == MODE_HEAT:
        return max(0.0, center - zone.temp)
    return 0.0


def _wants_conditioning(zone: ZoneSnapshot, lo: float, hi: float, mode: str) -> bool:
    """Demand: out of band now, or predicted to exit within the horizon."""
    if zone.temp is None:
        return False
    if mode == MODE_COOL:
        if zone.temp > hi:
            return True
        return zone.pred_60m is not None and zone.pred_60m > hi + PREDICT_MARGIN_K
    if mode == MODE_HEAT:
        if zone.temp < lo:
            return True
        return zone.pred_60m is not None and zone.pred_60m < lo - PREDICT_MARGIN_K
    return False


def _reached_far_edge(zone: ZoneSnapshot, lo: float, hi: float, mode: str) -> bool:
    """True when further conditioning would push the zone out the other side."""
    if zone.temp is None:
        return True
    if mode == MODE_COOL:
        return zone.temp <= lo
    if mode == MODE_HEAT:
        return zone.temp >= hi
    return True


def _transition_allowed(
    state: ControllerState, zone_id: str, now: float, turning_on: bool, s
) -> bool:
    since = state.zone_since.get(zone_id, 0.0)
    elapsed = now - since
    if turning_on and elapsed < s.min_off_min * 60.0:
        return False
    if not turning_on and elapsed < s.min_on_min * 60.0:
        return False
    changes = state.zone_mode_changes.get(zone_id, [])
    recent = [t for t in changes if now - t < 3600.0]
    return len(recent) < MAX_MODE_CHANGES_PER_H


def _record_transition(state: ControllerState, zone_id: str, now: float, on: bool) -> None:
    state.zone_on[zone_id] = on
    state.zone_since[zone_id] = now
    changes = state.zone_mode_changes.setdefault(zone_id, [])
    changes.append(now)
    del changes[:-10]


def _opposite_deviation(zone: ZoneSnapshot, lo: float, hi: float, mode: str) -> float:
    """How far the zone is out of band *against* the dominant mode (K, >=0)."""
    if zone.temp is None:
        return 0.0
    if mode == MODE_COOL:
        return max(0.0, lo - zone.temp)
    if mode == MODE_HEAT:
        return max(0.0, zone.temp - hi)
    return 0.0


def _window_would_help(
    zone: ZoneSnapshot, t_out: float | None, house_occupied: bool | None
) -> bool:
    """Outdoor air is clearly colder than the room and someone can act on it.

    Without any presence information (zone and house both unknown) we do not
    assume a person is available to open a window, and just cool.
    """
    if t_out is None or zone.temp is None:
        return False
    if zone.temp - t_out < WINDOW_DELTA_K:
        return False
    if zone.occupied is True:
        return True
    return zone.occupied is None and house_occupied is True


def tick(snap: HouseSnapshot, state: ControllerState) -> Decision:
    s = snap.settings
    now = snap.now_ts
    commands: list[Command] = []
    diag: dict = {}

    zones = [z for z in snap.zones if z.enabled and s.zone_enabled.get(z.zone_id, True)]

    # 1. Comfort bands.
    centers: dict[str, float] = {}
    bands: dict[str, tuple[float, float]] = {}
    for zone in zones:
        center = comfort.band_center(s, snap.t_rm, s.zone_offsets.get(zone.zone_id, 0.0))
        centers[zone.zone_id] = center
        bands[zone.zone_id] = comfort.zone_band(s, center, zone.occupied, snap.house_occupied)
    diag["bands"] = {z: bands[z] for z in bands}

    # 2. Mode arbitration.
    if s.hvac_mode == MODE_OFF:
        mode = MODE_OFF
        diag["mode_source"] = "forced"
        state.mode = MODE_OFF
    elif s.hvac_mode in (MODE_HEAT, MODE_COOL):
        mode = s.hvac_mode
        diag["mode_source"] = "forced"
        if state.mode != mode:
            state.mode = mode
            state.mode_since = now
    else:  # MODE_AUTO
        decision = comfort.dominant_mode(snap, state, bands)
        mode = decision.mode
        diag["mode_source"] = decision.source
        diag["warm_excess_kh"] = round(decision.warm_excess_kh, 2)
        diag["cold_deficit_kh"] = round(decision.cold_deficit_kh, 2)
    diag["mode"] = mode

    # 3. Demand and helper sets.
    demand = (
        [z for z in zones if _wants_conditioning(z, *bands[z.zone_id], mode)]
        if mode in (MODE_HEAT, MODE_COOL)
        else []
    )
    demand_ids = {z.zone_id for z in demand}

    helpers: list[ZoneSnapshot] = []
    if demand and s.coordination and snap.settings.hvac_mode in (MODE_AUTO, mode):
        for zone in zones:
            if zone.zone_id in demand_ids:
                continue
            if zone.occupied is False:
                continue
            # A helper must have margin in the mode direction: only trim
            # rooms sitting above center (cooling) / below center (heating).
            center = centers[zone.zone_id]
            if zone.temp is None:
                continue
            if mode == MODE_COOL and zone.temp <= center:
                continue
            if mode == MODE_HEAT and zone.temp >= center:
                continue
            if _reached_far_edge(zone, *bands[zone.zone_id], mode):
                continue
            helpers.append(zone)

    # Empirical COP table: consolidate when fewer heads have measurably
    # better efficiency at similar conditions.
    n_spread = sum(z.n_rooms for z in demand) + sum(z.n_rooms for z in helpers)
    n_demand = sum(z.n_rooms for z in demand)
    if helpers and snap.cop_by_head_count:
        cop_spread = snap.cop_by_head_count.get(n_spread)
        cop_demand = snap.cop_by_head_count.get(n_demand)
        if (
            cop_spread is not None
            and cop_demand is not None
            and cop_demand > cop_spread * COP_TABLE_ADVANTAGE
        ):
            helpers = []
            diag["consolidated_by_cop_table"] = True
    diag["demand"] = sorted(demand_ids)
    diag["helpers"] = sorted(z.zone_id for z in helpers)

    want_on = {z.zone_id: (z in demand or z in helpers) for z in zones}

    # Remember when each zone last ran its compressor in cooling: the coil
    # stays wet for a while and fan-only would re-evaporate condensate.
    for zone in zones:
        if zone.head_state == STATE_COOLING:
            state.zone_last_cool[zone.zone_id] = now

    # Window suggestion: when outdoor air is much colder than a room that
    # wants cooling and someone is around to act, propose opening a window
    # and hold mechanical cooling for a grace period. Purely optional; when
    # the option is off (or nobody is detectably present) we just cool.
    window_suggestions: list[str] = []
    if s.window_suggest and mode == MODE_COOL:
        for zone in demand:
            zid = zone.zone_id
            if _window_would_help(zone, snap.t_out, snap.house_occupied):
                since = state.window_suggest_since.setdefault(zid, now)
                window_suggestions.append(zid)
                if now - since < WINDOW_GRACE_S:
                    want_on[zid] = False  # give the user a chance first
            else:
                state.window_suggest_since.pop(zid, None)
        for zid in list(state.window_suggest_since):
            if zid not in window_suggestions:
                state.window_suggest_since.pop(zid, None)
    else:
        state.window_suggest_since.clear()
    diag["window_suggestions"] = window_suggestions

    # Fan assist: on a multi-split every compressor head shares one mode, so
    # a zone out of band in the *opposite* direction cannot be conditioned.
    # Fan-only mixing with house air can still improve comfort - but only
    # over a dry coil (see COIL_DRY_S above for the latent-heat caveat).
    fan_zones: set[str] = set()
    if s.multisplit and s.fan_assist and mode in (MODE_HEAT, MODE_COOL):
        for zone in zones:
            zid = zone.zone_id
            if want_on.get(zid) or zone.occupied is False or zid in state.shed:
                continue
            if _opposite_deviation(zone, *bands[zid], mode) < FAN_ASSIST_MIN_DEV_K:
                continue
            last_cool = state.zone_last_cool.get(zid)
            if last_cool is not None and now - last_cool < COIL_DRY_S:
                continue
            fan_zones.add(zid)
    diag["fan_assist"] = sorted(fan_zones)

    # 4. Shedding overrides everything else.
    shed_active = bool(state.shed)
    if s.shedding_enabled:
        if power.shed_needed(snap.p_grid, s.limit_w, s.shed_start_pct, snap.p_grid_over_since):
            if now - state.last_shed_action >= SHED_ACTION_SPACING_S:
                candidates = [
                    z for z in zones if state.zone_on.get(z.zone_id) and z.zone_id not in state.shed
                ]
                # Unoccupied first, then the zone that needs conditioning least.
                candidates.sort(
                    key=lambda z: (
                        0 if z.occupied is False else 1,
                        _comfort_error(z, centers[z.zone_id], mode),
                    )
                )
                if candidates:
                    victim = candidates[0]
                    state.shed[victim.zone_id] = now
                    state.last_shed_action = now
                    shed_active = True
        elif state.shed:
            # Restore the neediest zone when headroom allows.
            restorable = sorted(
                (z for z in zones if z.zone_id in state.shed),
                key=lambda z: -_comfort_error(z, centers[z.zone_id], mode),
            )
            for zone in restorable:
                if power.restore_allowed(snap.p_grid, s.limit_w, s.shed_restore_pct, zone.draw_w):
                    del state.shed[zone.zone_id]
                    state.last_shed_action = now
                    break
    else:
        state.shed.clear()
    diag["shed"] = sorted(state.shed)

    # 5. Emit commands with cycling guards.
    for zone in zones:
        zid = zone.zone_id
        desired_on = want_on.get(zid, False) and mode in (MODE_HEAT, MODE_COOL)
        if zid in state.shed:
            desired_on = False
        currently_on = state.zone_on.get(zid, zone.is_on)

        transitioned = False
        if desired_on != currently_on:
            # Only a user-forced OFF or shedding bypasses min-runtime; the
            # dominant mode dropping to idle still respects cycling guards.
            forced_off = not desired_on and (s.hvac_mode == MODE_OFF or zid in state.shed)
            if forced_off or _transition_allowed(state, zid, now, desired_on, s):
                _record_transition(state, zid, now, desired_on)
                transitioned = True
            else:
                desired_on = currently_on  # guard blocks the change this tick

        if desired_on and mode not in (MODE_HEAT, MODE_COOL):
            # Zone must keep running out its minimum runtime while the house
            # has gone idle: leave the head as-is, no command this tick.
            continue

        if desired_on:
            center = centers[zid]
            hi = bands[zid][1]
            if zid in demand_ids:
                setpoint = center
                reason = "demand"
            else:
                # Helper zones trim gently toward the band edge.
                offset = HELPER_BAND_FRACTION * (hi - center)
                setpoint = center + offset if mode == MODE_COOL else center - offset
                reason = "helper"
            setpoint = quantize_setpoint(setpoint)
            last_cmd = state.zone_last_cmd.get(zid, 0.0)
            last_sp = state.zone_last_setpoint.get(zid)
            setpoint_changed = last_sp is None or abs(setpoint - last_sp) >= SETPOINT_EPSILON_K
            spacing_ok = now - last_cmd >= COMMAND_SPACING_S
            if transitioned or not zone.is_on or (setpoint_changed and spacing_ok):
                commands.append(Command(zid, mode, setpoint, reason))
                state.zone_last_cmd[zid] = now
                state.zone_last_setpoint[zid] = setpoint
            state.zone_fan[zid] = False
        elif zid in fan_zones:
            head_in_fan = zone.head_state == STATE_FAN_ONLY
            spacing_ok = now - state.zone_last_cmd.get(zid, 0.0) >= COMMAND_SPACING_S
            if not state.zone_fan.get(zid) or (not head_in_fan and spacing_ok):
                commands.append(Command(zid, MODE_FAN, None, "fan_assist"))
                state.zone_last_cmd[zid] = now
            state.zone_fan[zid] = True
        elif state.zone_fan.get(zid):
            # Fan assist no longer needed: stop the fan.
            state.zone_fan[zid] = False
            commands.append(Command(zid, MODE_OFF, None, "fan_off"))
            state.zone_last_cmd[zid] = now
        elif transitioned or (
            zone.is_on and now - state.zone_last_cmd.get(zid, 0.0) >= COMMAND_SPACING_S
        ):
            # Just decided to stop, or the device reports on while it should
            # be off (e.g. manual/remote change) - reassert off, rate-limited.
            reason = "shed" if zid in state.shed else "off"
            commands.append(Command(zid, MODE_OFF, None, reason))
            state.zone_last_cmd[zid] = now
            state.zone_last_setpoint.pop(zid, None)

    diag["shedding_active"] = shed_active
    return Decision(commands, state, diag, window_suggestions)
