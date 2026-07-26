"""Decision engine: one pure tick from HouseSnapshot to zone commands.

The runtime (coordinator) builds a HouseSnapshot every 60 s and executes
the returned commands, translating room-coordinate setpoints into
device setpoints via the per-head drift offsets.
"""

from __future__ import annotations

from . import comfort, power
from .park import MARGIN_SETTLE_ALPHA
from .types import (
    MODE_AUTO,
    MODE_COOL,
    MODE_FAN,
    MODE_HEAT,
    MODE_OFF,
    PRESET_MANUAL,
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
SHED_URGENT_SPACING_S = 3.0
COMMAND_SPACING_S = 180.0
SETPOINT_EPSILON_K = 0.25
# Tracking setpoint control: the commanded device setpoint follows the head's
# internal sensor at a small depth below it (cooling), keeping the inverter's
# perceived error small and constant. The depth adapts to room-frame progress.
TRACK_DELTA_MIN_K = 0.3
TRACK_DELTA_MAX_K = 2.5
TRACK_DELTA_STEP_K = 0.5
TRACK_DELTA_DEFAULT_K = 0.7
# Parked-head characterization/exploitation: setpoint margin above the
# internal reading, minimum dwell in/out of the parked state, probe length
# and per-zone probe spacing while behavior is still unclassified.
PARK_MARGIN_K = 1.0
# Zones normally exit demand at the band floor, so a freshly parked room sits
# AT lo by construction. The overcorrection release must therefore sit a
# buffer BELOW the floor - otherwise every park dies on its first tick (as
# observed in the field: samples stayed 0 and probes burned their budget).
PARK_OVERCOOL_BUFFER_K = 0.4
# A probe that never produced an observation should be cheap to retry.
PARK_PROBE_RETRY_S = 1800.0
# Regime policy thresholds. The compressor cannot run below its floor; the
# only question is who duty-cycles it. When the aggregate standing load can
# feed a meaningful fraction of the floor, continuous (park-held) operation
# avoids controller-imposed off/restart losses; when outdoor air beats the
# compressor, neither should run.
REGIME_VENT_MARGIN_K = 1.0  # outdoor must be this far below the coolest target
# Compressor thermal floor per open head (~295 W electric solo-park median x
# park COP ~2.5-3 ≈ 800 W thermal across the open circuits; 265 x 3 ≈ 800).
REGIME_FLOOR_PER_HEAD_THERMAL_W = 265.0
REGIME_CONT_LOAD_FRACTION = 0.6  # → ~160 W standing load per head
REGIME_DWELL_S = 900.0  # hysteresis on regime switching
# Night outdoor gating (opt-in): widen ventilate and suppress continuous
# park-holds overnight when outdoor air is near the coolest target.
NIGHT_START_H = 22.0
NIGHT_END_H = 8.0
NIGHT_VENT_MARGIN_K = 0.0  # at night, outdoor at/below coolest target is enough
NIGHT_SKIP_CONT_K = 2.0  # outdoor within this of coolest → prefer cycling over continuous


def _is_night(local_hour: float) -> bool:
    return local_hour >= NIGHT_START_H or local_hour < NIGHT_END_H


def _select_regime(snap: HouseSnapshot, mode: str, centers: dict[str, float]) -> str:
    s = snap.settings
    if not s.auto_regime or mode not in (MODE_COOL, MODE_HEAT):
        return "cycling"
    # Free cooling: outdoor beats the coolest zone target. Heat has no
    # symmetric "ventilate" (opening windows when outdoor is warm is rare
    # and already covered by the window-suggestion path).
    # Do not enter ventilate on climatology-after-dropout (t_out_synthetic).
    if mode == MODE_COOL and snap.t_out is not None and not snap.t_out_synthetic and centers:
        coolest = min(centers.values())
        vent_margin = REGIME_VENT_MARGIN_K
        if s.night_ventilate and _is_night(snap.local_hour):
            vent_margin = NIGHT_VENT_MARGIN_K
        if snap.t_out <= coolest - vent_margin:
            return "ventilate"
        # Overnight with outdoor near-cool: don't keep continuous park-holds
        # chewing the compressor floor when free cooling is almost as good.
        if (
            s.night_ventilate
            and _is_night(snap.local_hour)
            and snap.t_out <= coolest + NIGHT_SKIP_CONT_K
        ):
            return "cycling"
    total_load = sum(z.standing_load_w or 0.0 for z in snap.zones if z.enabled)
    total_heads = sum(z.n_rooms for z in snap.zones if z.enabled)
    if total_heads > 0 and total_load >= (
        REGIME_CONT_LOAD_FRACTION * REGIME_FLOOR_PER_HEAD_THERMAL_W * total_heads
    ):
        return "continuous"
    return "cycling"


def _apply_regime_dwell(state: ControllerState, proposed: str, now: float) -> str:
    if state.regime_since == 0.0 or (
        proposed != state.regime and now - state.regime_since >= REGIME_DWELL_S
    ):
        state.regime = proposed
        state.regime_since = now
    return state.regime


PARK_MARGIN_MAX_K = 3.0
PARK_MARGIN_STEP_K = 0.5
# Enter each park one step below the learned preferred depth (still >=
# PARK_MARGIN_K so the setpoint stays on the park side of the internal
# reading). Re-proves the hold without overshooting from a stale high-water.
PARK_ENTRY_UNDERSHOOT_K = PARK_MARGIN_STEP_K
# Room movement in the conditioning direction that counts as "the head is
# not keeping temperature by itself" while parked (per margin decision).
PARK_ADAPT_EPS_K = 0.1
PARK_MIN_DWELL_S = 600.0
PARK_PROBE_S = 900.0
PARK_PROBE_SPACING_S = 6.0 * 3600.0
# Trickle output must plausibly carry the zone's standing load to justify
# exploitation-parking instead of a plain off.
PARK_LOAD_COVER_FRACTION = 0.6
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


def quantize_setpoint(
    value: float,
    minimum: float = 16.0,
    maximum: float = 30.0,
    step: float = 0.5,
) -> float:
    """Round to the device's temperature step, clamped to device limits."""
    step = 0.5 if step is None or step <= 0.0 else float(step)
    rounded = round(value / step) * step
    # Avoid binary float dust (e.g. 22.0000000002) before HA serialises.
    decimals = max(0, min(4, len(f"{step:.4f}".rstrip("0").split(".")[-1])))
    rounded = round(rounded, decimals)
    return min(max(rounded, minimum), maximum)


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


def _park_ok_now(zone: ZoneSnapshot, lo: float, hi: float, pmode: str | None) -> bool:
    """Park is keep-temperature only: never while the room still needs pull-down."""
    if zone.temp is None or pmode not in (MODE_HEAT, MODE_COOL):
        return False
    return zone.temp <= hi if pmode == MODE_COOL else zone.temp >= lo


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


def _park_exploit_ok(zone: ZoneSnapshot, mode: str) -> bool:
    """Known trickler whose parked output plausibly carries the standing load.

    park_extraction_w is sensed-room / per-head; standing_load_w is zone-total
    (x n_rooms). Compare in the per-room frame so mirrored zones are not
    falsely judged unable to cover their load.
    """
    if zone.park_trickles is not True:
        return False
    if zone.park_extraction_w is None:
        return False
    if zone.standing_load_w is None:
        # No load estimate: trickling while satisfied is still calmer than
        # off/on cycling, accept.
        return True
    per_room_load = zone.standing_load_w / max(1, zone.n_rooms)
    return zone.park_extraction_w >= PARK_LOAD_COVER_FRACTION * per_room_load


def _park_preferred(zone: ZoneSnapshot, state: ControllerState) -> float:
    """Learned park depth for this zone, falling back to the base margin.

    Once the hysteresis map has found a coasting depth, prefer that over the
    static default — it is the cheapest hold inside the head's dead-band.
    """
    zid = zone.zone_id
    if zid in state.zone_park_preferred:
        return state.zone_park_preferred[zid]
    if zone.park_coast_margin_k is not None:
        return min(max(zone.park_coast_margin_k, PARK_MARGIN_K), PARK_MARGIN_MAX_K)
    if zone.park_preferred_margin_k is not None:
        return zone.park_preferred_margin_k
    return PARK_MARGIN_K


def _park_entry_margin(zone: ZoneSnapshot, state: ControllerState) -> float:
    """Park entry depth: coast band when mapped, else preferred - undershoot."""
    if zone.park_coast_margin_k is not None:
        # Land inside the coast region; undershooting would re-engage compression.
        return min(max(zone.park_coast_margin_k, PARK_MARGIN_K), PARK_MARGIN_MAX_K)
    preferred = _park_preferred(zone, state)
    return max(preferred - PARK_ENTRY_UNDERSHOOT_K, PARK_MARGIN_K)


def _raise_park_preferred(state: ControllerState, zid: str, margin: float) -> None:
    """High-water: remember at least this depth for the next park entry."""
    prev = state.zone_park_preferred.get(zid, PARK_MARGIN_K)
    state.zone_park_preferred[zid] = max(prev, min(max(margin, PARK_MARGIN_K), PARK_MARGIN_MAX_K))


def _settle_park_preferred(state: ControllerState, zid: str, margin: float) -> None:
    """Blend preferred toward a margin that held without overcorrection."""
    prev = state.zone_park_preferred.get(zid, PARK_MARGIN_K)
    m = min(max(margin, PARK_MARGIN_K), PARK_MARGIN_MAX_K)
    state.zone_park_preferred[zid] = min(
        max(prev + MARGIN_SETTLE_ALPHA * (m - prev), PARK_MARGIN_K), PARK_MARGIN_MAX_K
    )


def _clear_park_session(
    state: ControllerState, zid: str, zone: ZoneSnapshot | None = None, now: float = 0.0
) -> None:
    # Probe bookkeeping: the 6 h probe spacing is only charged when the probe
    # produced at least one observation; a stillborn / unexpressible probe
    # (zone is None, or no new samples) gets the short retry clock instead.
    entry_samples = state.zone_park_probe_entry.pop(zid, None)
    if entry_samples is not None:
        if zone is not None and zone.park_samples > entry_samples:
            state.zone_last_park_probe[zid] = now
            state.zone_last_park_abort.pop(zid, None)
        else:
            state.zone_last_park_abort[zid] = now
            state.zone_last_park_probe.pop(zid, None)
    state.zone_parked_since.pop(zid, None)
    state.zone_park_ref.pop(zid, None)
    state.zone_park_margin.pop(zid, None)


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
    zone: ZoneSnapshot,
    t_out: float | None,
    mode: str,
    house_occupied: bool | None,
    *,
    t_out_synthetic: bool = False,
) -> bool:
    """Outdoor air would move the room toward comfort and someone can act on it.

    Cooling demand: outdoors clearly colder (free cooling / flush).
    Heating demand: outdoors clearly warmer (free warming / ventilate).

    Without any presence information (zone and house both unknown) we do not
    assume a person is available to open a window. Synthetic outdoor (climatology
    after a sensor dropout) must not drive suggestions or grace holds.
    """
    if t_out is None or zone.temp is None or t_out_synthetic:
        return False
    if mode == MODE_COOL and zone.temp - t_out < WINDOW_DELTA_K:
        return False
    if mode == MODE_HEAT and t_out - zone.temp < WINDOW_DELTA_K:
        return False
    if mode not in (MODE_COOL, MODE_HEAT):
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
    preset = comfort.effective_preset(s, snap.house_occupied)
    for zone in zones:
        center = comfort.band_center(s, snap.t_rm, s.zone_offsets.get(zone.zone_id, 0.0))
        centers[zone.zone_id] = center
        bands[zone.zone_id] = comfort.zone_band(s, center, zone.occupied, snap.house_occupied)
    diag["bands"] = {z: bands[z] for z in bands}
    diag["effective_preset"] = preset

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
    helper_ids = {z.zone_id for z in helpers}
    diag["helpers"] = sorted(helper_ids)

    want_on = {z.zone_id: (z in demand or z in helpers) for z in zones}
    diag["want"] = {
        z.zone_id: (
            "demand" if z.zone_id in demand_ids else "helper" if z.zone_id in helper_ids else "off"
        )
        for z in zones
    }

    # Coil-wet bookkeeping always (including Manual — remotes may still cool).
    for zone in zones:
        if zone.head_state == STATE_COOLING:
            state.zone_last_cool[zone.zone_id] = now

    # Manual: full hands-off — no comfort commands, no shedding, no fan assist.
    # Compute want/mode for diagnostics, clear park/shed/fan bookkeeping so
    # sensors and the react path cannot latch a lie, and emit nothing — unless
    # the hub HVAC mode is Off, which still force-stops children.
    if s.preset == PRESET_MANUAL:
        diag["manual"] = True
        for zone in zones:
            if zone.zone_id in state.zone_parked_since:
                _clear_park_session(state, zone.zone_id, zone, now)
        state.shed.clear()
        state.zone_fan.clear()
        state.last_shed_action = 0.0
        if s.hvac_mode != MODE_OFF:
            return Decision([], state, diag, [])

    # ---- Regime policy -------------------------------------------------
    regime = _apply_regime_dwell(state, _select_regime(snap, mode, centers), now)
    diag["regime"] = regime
    # 'ventilate' does not hard-gate: it widens the window-suggestion pool to
    # every warm zone and forces the grace-gated flow (below) regardless of
    # the window_suggest option — open within the grace period, or the
    # compressor proceeds anyway (at its best COP, given the cold outdoors).
    # A hot room is never stranded because nobody was around to open up.

    # Window suggestion: when outdoor air would help a room in demand (cooler
    # outdoors for cooling, warmer outdoors for heating) and someone is around
    # to act, flag the zone as a ventilation opportunity (always advisory).
    # The window_suggest option additionally *delays* mechanical conditioning
    # for a grace period so the user can open a window first; with the option
    # off the suggestion is still reported but conditioning starts immediately.
    # In 'ventilate' regime the grace gate is forced regardless of the option.
    window_suggestions: list[str] = []
    if mode in (MODE_HEAT, MODE_COOL):
        suggest_pool = (
            [z for z in zones if z.temp is not None and z.temp > centers[z.zone_id]]
            if regime == "ventilate" and mode == MODE_COOL
            else demand
        )
        in_pool: set[str] = set()
        for zone in suggest_pool:
            zid = zone.zone_id
            if _window_would_help(
                zone,
                snap.t_out,
                mode,
                snap.house_occupied,
                t_out_synthetic=snap.t_out_synthetic,
            ):
                in_pool.add(zid)
                state.window_suggest_out_since.pop(zid, None)
                since = state.window_suggest_since.setdefault(zid, now)
                window_suggestions.append(zid)
                if (s.window_suggest or regime == "ventilate") and now - since < WINDOW_GRACE_S:
                    want_on[zid] = False  # give the user a chance first
        # Keep the grace clock across brief disqualification (oscillation
        # around WINDOW_DELTA_K). Only clear after WINDOW_GRACE_S continuously
        # out of the pool so the next cool-down episode can re-arm once.
        for zid in list(state.window_suggest_since):
            if zid in in_pool:
                continue
            out_since = state.window_suggest_out_since.setdefault(zid, now)
            if now - out_since >= WINDOW_GRACE_S:
                state.window_suggest_since.pop(zid, None)
                state.window_suggest_out_since.pop(zid, None)
    else:
        state.window_suggest_since.clear()
        state.window_suggest_out_since.clear()
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
    p_demand = snap.p_demand if snap.p_demand is not None else snap.p_grid
    if s.shedding_enabled:
        if power.shed_needed(
            p_demand,
            s.limit_w,
            s.shed_start_pct,
            snap.p_grid_over_since,
            urgent=snap.shed_urgent,
        ):
            spacing = SHED_URGENT_SPACING_S if snap.shed_urgent else SHED_ACTION_SPACING_S
            if now - state.last_shed_action >= spacing:
                candidates = [
                    z for z in zones if state.zone_on.get(z.zone_id) and z.zone_id not in state.shed
                ]
                candidates.sort(
                    key=lambda z: (
                        0 if z.occupied is False else 1,
                        _comfort_error(z, centers[z.zone_id], mode),
                    )
                )
                victims = candidates if snap.shed_urgent else candidates[:1]
                for victim in victims:
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
                if power.restore_allowed(p_demand, s.limit_w, s.shed_restore_pct, zone.draw_w):
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
        # ---- Parked-state management (multi-split, heat and cool) -----------
        # A zone leaving demand/helpers would normally turn off. If park
        # learning is enabled, the compressor stays alive for siblings, and
        # either (a) behavior is unclassified and a probe is due, or (b) the
        # head is a known trickler whose output can carry the standing load,
        # we park instead: setpoint offset from the internal reading, mode
        # kept. Session margin starts at the learned preferred depth and
        # escalates while the room keeps moving in the conditioning direction.
        if zid not in state.zone_park_preferred:
            state.zone_park_preferred[zid] = _park_preferred(zone, state)
        parked_since = state.zone_parked_since.get(zid)
        if parked_since is not None:
            # Direction: dominant mode when conditioning, else the head's own
            # physical mode - run-out parks must survive the house going idle.
            pmode = mode if mode in (MODE_HEAT, MODE_COOL) else zone.head_mode
            off_allowed = (
                s.hvac_mode == MODE_OFF
                or zid in state.shed
                or _transition_allowed(state, zid, now, False, s)
            )
            sibling_alive = any(want_on.get(z.zone_id) and z.zone_id != zid for z in zones)
            dwell_ok = now - parked_since >= PARK_MIN_DWELL_S
            probing = zone.park_trickles is None
            probe_done = probing and now - parked_since >= PARK_PROBE_S
            lo_b, hi_b = bands[zid]
            out_of_band = zone.temp is not None and (
                pmode is not None and (zone.temp > hi_b if pmode == MODE_COOL else zone.temp < lo_b)
            )
            # Far-edge breach: the parked head is out-conditioning the load
            # and pushed the room through the opposite comfort edge. Releases
            # immediately (no dwell): comfort beats characterization.
            overcorrected = zone.temp is not None and (
                pmode is not None
                and (
                    zone.temp <= lo_b - PARK_OVERCOOL_BUFFER_K
                    if pmode == MODE_COOL
                    else zone.temp >= hi_b + PARK_OVERCOOL_BUFFER_K
                )
            )

            # Margin escalation: if the room keeps moving in the conditioning
            # direction while parked, the head is not holding temperature at
            # this depth - raise the setpoint further away from the internal
            # reading before giving up. Only when the margin is maxed and the
            # room still falls (cool) / rises (heat) do we idle the head.
            preferred = state.zone_park_preferred.get(zid, PARK_MARGIN_K)
            margin = state.zone_park_margin.get(zid, preferred)
            margin_exhausted = False
            margin_changed = False
            if zone.temp is not None and pmode in (MODE_HEAT, MODE_COOL):
                ref = state.zone_park_ref.get(zid)
                if ref is None:
                    state.zone_park_ref[zid] = zone.temp
                else:
                    moved = ref - zone.temp if pmode == MODE_COOL else zone.temp - ref
                    if moved >= PARK_ADAPT_EPS_K:
                        if margin < PARK_MARGIN_MAX_K:
                            margin = min(margin + PARK_MARGIN_STEP_K, PARK_MARGIN_MAX_K)
                            state.zone_park_margin[zid] = margin
                            state.zone_park_ref[zid] = zone.temp
                            _raise_park_preferred(state, zid, margin)
                            margin_changed = True
                        else:
                            margin_exhausted = True
                    elif moved <= -PARK_ADAPT_EPS_K:
                        # Drifting back toward the load side: the head eased
                        # off (or idles); relax toward the base margin. A clean
                        # exit then settles preferred toward this lower depth.
                        new_margin = max(margin - PARK_MARGIN_STEP_K, PARK_MARGIN_K)
                        if new_margin != margin:
                            margin = new_margin
                            state.zone_park_margin[zid] = margin
                            margin_changed = True
                        state.zone_park_ref[zid] = zone.temp

            if (overcorrected or margin_exhausted) and not off_allowed:
                # Head cannot hold temperature, but min-runtime forbids off:
                # hold at maximum depth (the gentlest expressible output).
                state.zone_park_margin[zid] = PARK_MARGIN_MAX_K
                diag.setdefault("park_overcorrected_held", []).append(zid)
                if (
                    now - state.zone_last_cmd.get(zid, 0.0) >= COMMAND_SPACING_S
                    and pmode is not None
                ):
                    commands.append(
                        Command(zid, pmode, None, "park", park=True, park_margin=PARK_MARGIN_MAX_K)
                    )
                    state.zone_last_cmd[zid] = now
                continue
            if overcorrected or margin_exhausted:
                # Head cannot hold temperature at any depth: idle it.
                # Keep preferred high-water so the next park starts deeper.
                _clear_park_session(state, zid, zone, now)
                diag.setdefault("park_overcorrected", []).append(zid)
                desired_on = False
            elif out_of_band and (
                dwell_ok
                or (
                    # Under-conditioned breach: comfort beats characterization
                    # both ways (symmetric to overcorrection's immediate release).
                    zone.temp is not None
                    and pmode is not None
                    and (
                        zone.temp >= hi_b + PARK_OVERCOOL_BUFFER_K
                        if pmode == MODE_COOL
                        else zone.temp <= lo_b - PARK_OVERCOOL_BUFFER_K
                    )
                )
            ):
                # Load beat the parked output: fall through to normal
                # demand handling below (zone re-enters as wanting on).
                _settle_park_preferred(state, zid, margin)
                _clear_park_session(state, zid, zone, now)
            elif zid in state.shed or (
                dwell_ok
                and off_allowed
                and (
                    probe_done
                    or zone.park_trickles is False
                    or (
                        regime != "continuous"
                        and (not _park_exploit_ok(zone, pmode) or not sibling_alive)
                    )
                    or pmode is None
                )
            ):
                # Probe finished, head classified as idler, or exploitation
                # no longer justified: release to normal off handling.
                # Settle even when the house mode is already idle (run-out parks).
                # In 'continuous' regime, sibling requirement and exploit check
                # are waived — the load feeds the compressor floor, so holding
                # parked is cheaper than controller-imposed off/restart cycles.
                if zid not in state.shed:
                    _settle_park_preferred(state, zid, margin)
                _clear_park_session(state, zid, zone, now)
                desired_on = False
            else:
                # Stay parked: refresh when spacing elapses *or* margin moved
                # so the head sees the new depth immediately.
                last_cmd = state.zone_last_cmd.get(zid, 0.0)
                if (margin_changed or now - last_cmd >= COMMAND_SPACING_S) and pmode is not None:
                    commands.append(
                        Command(zid, pmode, None, "park", park=True, park_margin=margin)
                    )
                    state.zone_last_cmd[zid] = now
                continue
        elif (
            not desired_on
            and currently_on
            and s.park_learning
            and s.multisplit
            and zid not in state.shed
            and mode in (MODE_HEAT, MODE_COOL)
            and _park_ok_now(zone, *bands[zid], mode)
            and (
                regime == "continuous"
                or any(want_on.get(z.zone_id) and z.zone_id != zid for z in zones)
            )
            and _transition_allowed(state, zid, now, False, s)
        ):
            probe_due = (
                zone.park_trickles is None
                and now - state.zone_last_park_probe.get(zid, 0.0) >= PARK_PROBE_SPACING_S
                and now - state.zone_last_park_abort.get(zid, 0.0) >= PARK_PROBE_RETRY_S
            )
            exploit = zone.park_trickles is True and _park_exploit_ok(zone, mode)
            if probe_due or exploit:
                preferred = _park_preferred(zone, state)
                entry = _park_entry_margin(zone, state)
                state.zone_park_preferred[zid] = preferred
                state.zone_parked_since[zid] = now
                state.zone_park_margin[zid] = entry
                if zone.temp is not None:
                    state.zone_park_ref[zid] = zone.temp
                if probe_due:
                    state.zone_park_probe_entry[zid] = zone.park_samples
                commands.append(Command(zid, mode, None, "park", park=True, park_margin=entry))
                state.zone_last_cmd[zid] = now
                diag.setdefault("parked", []).append(zid)
                continue

        if desired_on != currently_on:
            # Only a user-forced OFF or shedding bypasses min-runtime; the
            # dominant mode dropping to idle still respects cycling guards.
            forced_off = not desired_on and (s.hvac_mode == MODE_OFF or zid in state.shed)
            if forced_off or _transition_allowed(state, zid, now, desired_on, s):
                _record_transition(state, zid, now, desired_on)
                transitioned = True
            else:
                if (
                    not desired_on
                    and currently_on
                    and s.park_learning
                    and zid not in state.shed
                    and zid not in state.zone_parked_since
                    and zone.head_mode in (MODE_HEAT, MODE_COOL)
                    and _park_ok_now(zone, *bands[zid], zone.head_mode)
                ):
                    # Run-out park: the zone wants off but min-runtime forces
                    # it to keep running - the compressor is alive regardless,
                    # so parking is free. Keep-temp only: never park while
                    # still out of band (unfinished pull-down).
                    entry = _park_entry_margin(zone, state)
                    state.zone_parked_since[zid] = now
                    state.zone_park_margin[zid] = entry
                    if zone.temp is not None:
                        state.zone_park_ref[zid] = zone.temp
                    commands.append(
                        Command(zid, zone.head_mode, None, "park", park=True, park_margin=entry)
                    )
                    state.zone_last_cmd[zid] = now
                    diag.setdefault("parked", []).append(zid)
                    continue
                desired_on = currently_on  # guard blocks the change this tick

        if desired_on and mode not in (MODE_HEAT, MODE_COOL):
            # Zone is running out its minimum runtime while the house has
            # gone idle. Do not leave the head at a stale (possibly deep)
            # tracked setpoint: ease it to minimum tracking depth in its own
            # physical direction so it trickles instead of cooling hard into
            # a room nobody asked to condition further.
            if (
                s.tracking
                and zone.is_on
                and zone.head_mode in (MODE_HEAT, MODE_COOL)
                and now - state.zone_last_cmd.get(zid, 0.0) >= COMMAND_SPACING_S
            ):
                state.zone_track_delta[zid] = TRACK_DELTA_MIN_K
                commands.append(
                    Command(zid, zone.head_mode, None, "runout", track_delta=TRACK_DELTA_MIN_K)
                )
                state.zone_last_cmd[zid] = now
            continue

        if desired_on:
            center = centers[zid]
            lo, hi = bands[zid]
            if zid in demand_ids:
                setpoint = comfort.demand_setpoint(mode, center, lo, hi, zone.occupied, preset)
                reason = "demand"
            else:
                # Helper zones trim gently toward the band edge.
                offset = HELPER_BAND_FRACTION * (hi - center)
                setpoint = center + offset if mode == MODE_COOL else center - offset
                reason = "helper"
            setpoint = quantize_setpoint(setpoint)

            # Tracking depth adaptation (room frame): deepen while the room is
            # not converging toward its target edge, relax once it is inside.
            track_delta = None
            if s.tracking and mode in (MODE_HEAT, MODE_COOL):
                delta = state.zone_track_delta.get(zid, TRACK_DELTA_DEFAULT_K)
                if zone.temp is not None:
                    past_target = (
                        zone.temp <= setpoint if mode == MODE_COOL else zone.temp >= setpoint
                    )
                    out_of_band = zone.temp > hi if mode == MODE_COOL else zone.temp < lo
                    if out_of_band:
                        delta += TRACK_DELTA_STEP_K
                    elif past_target:
                        delta -= TRACK_DELTA_STEP_K
                delta = min(max(delta, TRACK_DELTA_MIN_K), TRACK_DELTA_MAX_K)
                state.zone_track_delta[zid] = delta
                track_delta = delta

            last_cmd = state.zone_last_cmd.get(zid, 0.0)
            last_sp = state.zone_last_setpoint.get(zid)
            setpoint_changed = last_sp is None or abs(setpoint - last_sp) >= SETPOINT_EPSILON_K
            spacing_ok = now - last_cmd >= COMMAND_SPACING_S
            # Tracking control re-anchors to the moving internal reading, so
            # refresh commands on every spacing interval while the zone runs.
            refresh = track_delta is not None and zone.is_on and spacing_ok
            if transitioned or not zone.is_on or (setpoint_changed and spacing_ok) or refresh:
                commands.append(Command(zid, mode, setpoint, reason, track_delta=track_delta))
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
            state.zone_track_delta.pop(zid, None)
            _clear_park_session(state, zid, zone, now)

    diag["shedding_active"] = shed_active
    return Decision(commands, state, diag, window_suggestions)
