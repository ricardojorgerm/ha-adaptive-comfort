"""Decision engine: one pure tick from HouseSnapshot to zone commands.

The runtime (coordinator) builds a HouseSnapshot every 60 s and executes
the returned commands, translating room-coordinate setpoints into
device setpoints via the per-head drift offsets.
"""

from __future__ import annotations

import math

from . import comfort, cop_timing, plant, power, regime
from .cop_timing import (
    _apply_cop_widen_bands,
    _clear_cop_widen,
    _outside_base_band_on_widen_side,
    _update_cop_widen,
)
from .head_depth import (
    DEPTH_MAX_K,
    DEPTH_STEP_K,
    PARK_MARGIN_K,
    PARK_MARGIN_MAX_K,
    PARK_MARGIN_STEP_K,
    TRACK_DELTA_DEFAULT_K,
    TRACK_DELTA_MAX_K,
    TRACK_DELTA_MIN_K,
    TRACK_DELTA_STEP_K,
    _chase_floor_k,
    _clamp_depth,
    _clear_park_session,
    _quantize_depth,
    _sync_depth_views,
)
from .park import (
    CLASSIFY_MIN_SAMPLES,
    LOAD_COVER_FRACTION,
    MARGIN_SETTLE_ALPHA,
    pick_depth_k,
    usable_residual_bin,
)
from .plant import (
    _plant_min_on_active,
    _record_transition,
    _transition_allowed,
    _update_plant_compress,
)
from .regime import (
    NIGHT_START_H,
    _apply_regime_dwell,
    _is_night,
    _select_regime,
)
from .types import (
    MODE_AUTO,
    MODE_COOL,
    MODE_FAN,
    MODE_HEAT,
    MODE_OFF,
    PRESET_AWAY,
    PRESET_BOOST,
    PRESET_ECO,
    PRESET_MANUAL,
    PRESET_NONE,
    STATE_COOLING,
    STATE_FAN_ONLY,
    Command,
    ControllerState,
    Decision,
    HouseSnapshot,
    Settings,
    ZoneSnapshot,
)

# Re-export extracted names so tests keep importing via controller.
COP_ADVANTAGE_ENTER = cop_timing.COP_ADVANTAGE_ENTER
COP_ADVANTAGE_EXIT = cop_timing.COP_ADVANTAGE_EXIT
COP_TIMING_LOOKAHEAD_H = cop_timing.COP_TIMING_LOOKAHEAD_H
COP_WIDEN_MAX_K = cop_timing.COP_WIDEN_MAX_K
DEFAULT_BAND_COP_COOL = cop_timing.DEFAULT_BAND_COP_COOL
_arbitrage_ratios = cop_timing._arbitrage_ratios
_band_cop = cop_timing._band_cop
_forecast_cop_arbitrage = cop_timing._forecast_cop_arbitrage
MAX_MODE_CHANGES_PER_H = plant.MAX_MODE_CHANGES_PER_H
ZONE_CHATTER_S = plant.ZONE_CHATTER_S
NIGHT_END_H = regime.NIGHT_END_H
NIGHT_SKIP_CONT_K = regime.NIGHT_SKIP_CONT_K
NIGHT_VENT_MARGIN_K = regime.NIGHT_VENT_MARGIN_K
REGIME_CONT_LOAD_FRACTION = regime.REGIME_CONT_LOAD_FRACTION
REGIME_DWELL_S = regime.REGIME_DWELL_S
REGIME_FLOOR_PER_HEAD_THERMAL_W = regime.REGIME_FLOOR_PER_HEAD_THERMAL_W
REGIME_VENT_MARGIN_K = regime.REGIME_VENT_MARGIN_K

PREDICT_MARGIN_K = 0.1
SHED_ACTION_SPACING_S = 30.0
SHED_URGENT_SPACING_S = 3.0
COMMAND_SPACING_S = 180.0
SETPOINT_EPSILON_K = 0.25
# Re-anchor head depth when live internal drifts vs the device setpoint.
# Chase (negative depth) follows the sensor; positive park depth must not —
# re-issuing SP = internal + margin on a falling supply-air reading is a
# descending ladder (West 22.9→21.0), including the 180 s spacing refresh.
HEAD_REANCHOR_MIN_S = 60.0
HEAD_REANCHOR_EPS_K = 0.5
# Zones normally exit demand at the band floor, so a freshly parked room sits
# AT lo by construction. The overcorrection release must therefore sit a
# buffer BELOW the floor - otherwise every park dies on its first tick (as
# observed in the field: samples stayed 0 and probes burned their budget).
PARK_OVERCOOL_BUFFER_K = 0.4
# A probe that never produced an observation should be cheap to retry.
PARK_PROBE_RETRY_S = 1800.0
# Pre-sleep window: bank a quiet-night zone to the conditioning hold edge
# of the existing band (cool lo / heat hi), then night prefers other zones.
QUIET_NIGHT_BANK_H = 2.0
COP_ADVANCE_HORIZON_H = 4.0  # predictive entry may look this far when advancing
COP_DEFER_HORIZON_H = 1.0  # defer: only near breaches fire predictively


def _is_quiet_night_bank_hour(local_hour: float) -> bool:
    start = NIGHT_START_H - QUIET_NIGHT_BANK_H
    return start <= local_hour < NIGHT_START_H


def _quiet_night_opted(s: Settings, zid: str) -> bool:
    if s.preset in (PRESET_ECO, PRESET_AWAY, PRESET_BOOST):
        return False
    return bool(s.zone_quiet_night.get(zid))


def _quiet_night_active(s: Settings, zid: str, local_hour: float, mode: str) -> bool:
    """Night deferral for zones that should not blow on sleepers."""
    return mode in (MODE_HEAT, MODE_COOL) and _is_night(local_hour) and _quiet_night_opted(s, zid)


def _quiet_night_bank_active(s: Settings, zid: str, local_hour: float, mode: str) -> bool:
    """Pre-sleep: this zone may run to bank the conditioning hold edge."""
    return (
        mode in (MODE_HEAT, MODE_COOL)
        and _is_quiet_night_bank_hour(local_hour)
        and _quiet_night_opted(s, zid)
    )


def _quiet_night_condition_hold(s: Settings, zid: str, local_hour: float, mode: str) -> bool:
    """Demand setpoint uses the conditioning hold edge (same band)."""
    return _quiet_night_active(s, zid, local_hour, mode) or _quiet_night_bank_active(
        s, zid, local_hour, mode
    )


def _quiet_night_bank_needed(zone: ZoneSnapshot, lo: float, hi: float, mode: str) -> bool:
    if zone.temp is None or _reached_far_edge(zone, lo, hi, mode):
        return False
    target = comfort.hold_edge_setpoint(mode, lo, hi, conditioning=True)
    if mode == MODE_COOL:
        return zone.temp > target
    if mode == MODE_HEAT:
        return zone.temp < target
    return False


def _quiet_night_cover(
    zones: list[ZoneSnapshot],
    quiet_ids: set[str],
    bands: dict[str, tuple[float, float]],
    mode: str,
    s: Settings,
) -> ZoneSnapshot | None:
    """One non-quiet zone that can carry standing load instead of the sleep room."""
    cands = []
    for z in zones:
        if z.zone_id in quiet_ids or z.temp is None:
            continue
        lo, hi = bands[z.zone_id]
        ext_lo, ext_hi = _extended_cover_band(s, z, lo, hi, mode)
        # Spent cover still belongs in the pack after the sleeper joins.
        if _reached_far_edge(z, ext_lo, ext_hi, mode) and z.temp is not None:
            if mode == MODE_COOL and z.temp < ext_lo:
                continue
            if mode == MODE_HEAT and z.temp > ext_hi:
                continue
        cands.append(z)
    if not cands:
        return None
    return max(cands, key=lambda z: (z.is_on, z.temp or 0.0))


def _quiet_night_wants_cover(zone: ZoneSnapshot, center: float, mode: str) -> bool:
    """In-band room on the conditioning side of center — house load, not pull-down."""
    if zone.temp is None:
        return False
    if mode == MODE_COOL:
        return zone.temp > center
    if mode == MODE_HEAT:
        return zone.temp < center
    return False


def _apply_quiet_night(
    zones: list[ZoneSnapshot],
    demand: list[ZoneSnapshot],
    helpers: list[ZoneSnapshot],
    s: Settings,
    snap: HouseSnapshot,
    mode: str,
    bands: dict[str, tuple[float, float]],
    centers: dict[str, float],
    *,
    energy_save: bool,
) -> tuple[list[ZoneSnapshot], list[ZoneSnapshot], list[str], str | None]:
    """Prefer other zones at night. Same comfort band as everyone else.

    Cover may run to the extended far edge (vacant-widen slack on the
    conditioning side even if occupied). Quiet rooms stay off that path
    until the cover is spent or OVERRIDE_DELTA_K. Eco/Away skip this.
    """
    if energy_save:
        return demand, helpers, [], None
    quiet_ids = {
        z.zone_id for z in zones if _quiet_night_active(s, z.zone_id, snap.local_hour, mode)
    }
    helpers = [z for z in helpers if z.zone_id not in quiet_ids]
    if not quiet_ids:
        return demand, helpers, [], None
    cover = _quiet_night_cover(zones, quiet_ids, bands, mode, s)
    demand_ids = {z.zone_id for z in demand}
    deferred: list[str] = []
    cover_id = cover.zone_id if cover is not None else None
    cover_spent = False
    if cover is not None:
        lo_c, hi_c = _extended_cover_band(s, cover, *bands[cover.zone_id], mode)
        cover_spent = _reached_far_edge(cover, lo_c, hi_c, mode)
    if cover is not None and not cover_spent:
        kept: list[ZoneSnapshot] = []
        for zone in demand:
            if zone.zone_id not in quiet_ids:
                kept.append(zone)
                continue
            lo_b, hi_b = bands[zone.zone_id]
            if _same_mode_override(zone, lo_b, hi_b, mode):
                kept.append(zone)
                continue
            deferred.append(zone.zone_id)
        demand = kept
        demand_ids = {z.zone_id for z in demand}
        taken = demand_ids | {z.zone_id for z in helpers}
        if cover.zone_id not in taken:
            helpers.append(cover)
        for zone in zones:
            if zone.zone_id not in quiet_ids:
                continue
            if zone.zone_id in demand_ids or zone.zone_id in deferred:
                continue
            if _quiet_night_wants_cover(zone, centers[zone.zone_id], mode):
                deferred.append(zone.zone_id)
    elif cover is not None and cover_spent:
        taken = demand_ids | {z.zone_id for z in helpers}
        if cover.zone_id not in taken and cover.zone_id not in quiet_ids:
            helpers.append(cover)
    return demand, helpers, deferred, cover_id


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
# Residual output must plausibly carry the zone's standing load to justify
# exploitation-parking instead of a plain off.
PARK_LOAD_COVER_FRACTION = LOAD_COVER_FRACTION
COP_TABLE_ADVANTAGE = 1.05


def _cop_n_pair(
    snap: HouseSnapshot, n_demand: int, n_spread: int
) -> tuple[float | None, float | None]:
    """Demand vs demand+helpers COP from the live outdoor band only.

    Unbanded head-count COP is diagnostic (N confounded with weather) and
    must not decide spread vs consolidate.
    """
    banded = snap.cop_by_head_count_banded
    if not banded:
        return None, None
    return banded.get(n_demand), banded.get(n_spread)


def _energy_save_preset(preset: str) -> bool:
    return preset in (PRESET_ECO, PRESET_AWAY)


def _same_mode_override(zone: ZoneSnapshot, lo: float, hi: float, mode: str) -> bool:
    """True when the room is past OVERRIDE_DELTA_K on the demand side."""
    if zone.temp is None:
        return False
    if mode == MODE_COOL:
        return zone.temp > hi + comfort.OVERRIDE_DELTA_K
    if mode == MODE_HEAT:
        return zone.temp < lo - comfort.OVERRIDE_DELTA_K
    return False


def _extended_cover_band(
    s: Settings,
    zone: ZoneSnapshot,
    lo: float,
    hi: float,
    mode: str,
) -> tuple[float, float]:
    """Cover far-edge: vacant-widen slack on the conditioning side if occupied."""
    ext_lo, ext_hi = lo, hi
    if comfort.effective_zone_occupied(s, zone.occupied) is not False:
        if mode == MODE_COOL:
            ext_lo -= comfort.UNOCCUPIED_WIDEN_K
        elif mode == MODE_HEAT:
            ext_hi += comfort.UNOCCUPIED_WIDEN_K
    return comfort.clamp_to_envelope(ext_lo, ext_hi, s)


def _depth_chase_cap_k(snap: HouseSnapshot, n_heads: int, in_band: bool) -> float:
    """In-band pack: shallow chase unless cop_by_depth says a deeper bin wins."""
    if n_heads < 2 or not in_band:
        return TRACK_DELTA_MAX_K
    table = snap.cop_by_depth
    if not table:
        return TRACK_DELTA_MIN_K
    shallow = table.get(-TRACK_DELTA_MIN_K)
    best = TRACK_DELTA_MIN_K
    best_cop = shallow
    for depth, cop in table.items():
        if depth >= 0.0:
            continue
        chase = abs(depth)
        if best_cop is None or cop > best_cop * COP_TABLE_ADVANTAGE:
            best_cop = cop
            best = min(max(chase, TRACK_DELTA_MIN_K), TRACK_DELTA_MAX_K)
    return best


def _compressing_hold_k(zone: ZoneSnapshot) -> float:
    """Slow-walk ceiling: deepest still-compressing depth, not residual-edge.

    Unmapped heads may walk toward DEPTH_MAX; a live fan-type read steps back.
    """
    if zone.park_residual_max_margin_k is not None:
        cap = max(0.0, float(zone.park_residual_max_margin_k))
    else:
        cap = DEPTH_MAX_K
    fan = zone.park_fan_type_min_margin_k
    if fan is not None and fan > 0.0:
        cap = min(cap, float(fan) - DEPTH_STEP_K)
    return max(0.0, _quantize_depth(cap))


# Helper park walk: 0 → 0.5 → 1.0 → 1.5 … per re-anchor. Stop deepening
# when the room is stable; deepen when the far edge is arriving fast, but
# only while the next step still looks like compression. Prefer lingering
# in-band over peeling off.
HELPER_STABLE_K = 0.1
HELPER_EDGE_FAST_S = 45.0 * 60.0


def _far_edge_slack_k(zone: ZoneSnapshot, lo: float, hi: float, mode: str) -> float | None:
    if zone.temp is None:
        return None
    if mode == MODE_COOL:
        return zone.temp - lo
    if mode == MODE_HEAT:
        return hi - zone.temp
    return None


def _helper_toward_far_edge_k(zone: ZoneSnapshot, ref: float | None, mode: str) -> float:
    if zone.temp is None or ref is None:
        return 0.0
    if mode == MODE_COOL:
        return ref - zone.temp
    if mode == MODE_HEAT:
        return zone.temp - ref
    return 0.0


def _helper_too_fast_to_edge(
    zone: ZoneSnapshot,
    lo: float,
    hi: float,
    mode: str,
    toward_k: float,
    dt_s: float,
) -> bool:
    slack = _far_edge_slack_k(zone, lo, hi, mode)
    if slack is None:
        return False
    if zone.pred_60m is not None:
        if mode == MODE_COOL and zone.pred_60m <= lo + 0.2:
            return True
        if mode == MODE_HEAT and zone.pred_60m >= hi - 0.2:
            return True
    if toward_k <= HELPER_STABLE_K or dt_s <= 1.0:
        return False
    rate = toward_k / dt_s
    if rate <= 1e-9:
        return False
    return (slack / rate) < HELPER_EDGE_FAST_S


def _helper_still_compressing(zone: ZoneSnapshot, depth_k: float) -> bool:
    if depth_k <= 0.0:
        return True
    if zone.park_current_is_fan_type is True:
        return False
    return not _extraction_none_at_depth(zone, depth_k)


def _helper_walk_depth(
    state: ControllerState,
    zone: ZoneSnapshot,
    lo: float,
    hi: float,
    mode: str,
    now: float,
) -> float:
    """0, then +0.5 K per re-anchor while useful; hold when stable.

    First command is zero-hold (SP = live reading). The next re-anchor
    enters park at +0.5 K. Further steps only if the room is still moving
    toward the far edge (or arriving too fast) and the next bin should
    still compress. Stable in-band helpers stay put so the pack can linger.
    """
    zid = zone.zone_id
    cap = _compressing_hold_k(zone)
    current = state.zone_head_depth_k.get(zid)
    last = state.zone_last_cmd.get(zid, 0.0)
    ref = state.zone_park_ref.get(zid)
    if current is None:
        if zone.temp is not None:
            state.zone_park_ref[zid] = zone.temp
        return 0.0
    if last <= 0.0 or now - last < COMMAND_SPACING_S:
        return current
    toward = _helper_toward_far_edge_k(zone, ref, mode)
    dt_s = now - last
    too_fast = _helper_too_fast_to_edge(zone, lo, hi, mode, toward, dt_s)
    stable = toward <= HELPER_STABLE_K
    stepped = current
    nxt = min(current + DEPTH_STEP_K, cap)
    if not _helper_still_compressing(zone, current) and current > 0.0:
        stepped = max(current - DEPTH_STEP_K, 0.0)
    elif current < DEPTH_STEP_K - 1e-9 and _helper_still_compressing(zone, DEPTH_STEP_K):
        stepped = DEPTH_STEP_K
    elif stable and not too_fast:
        stepped = current
    elif _helper_still_compressing(zone, nxt):
        stepped = nxt
    else:
        stepped = current
    if zone.temp is not None:
        state.zone_park_ref[zid] = zone.temp
    return stepped


def _pack_residual_ids(
    zones: list[ZoneSnapshot],
    bands: dict[str, tuple[float, float]],
    mode: str,
    state: ControllerState,
    *,
    energy_wait: bool,
    consolidated: bool,
    demand: list[ZoneSnapshot],
    helpers: list[ZoneSnapshot],
    coordination: bool,
    energy_save: bool,
) -> set[str]:
    """In-band already-on coils that should residual-park together.

    Demand and helpers stay on tracking (want_on). Satisfied extras that
    have left those sets park so N and Tevap stay high: N≥2 after the
    burst, or a single in-band leftover while Eco/Away still has demand.
    Solo leftovers with nobody in demand still drop unless
    prefer_continuous elects them as the N=1 anchor. Fan-type members
    drop when another residual coil is already in the pack. Off when
    coordination is down, energy-wait (start gate), or learned consolidate.
    """
    if not coordination or energy_wait or consolidated or mode not in (MODE_HEAT, MODE_COOL):
        return set()
    taken = {z.zone_id for z in demand} | {z.zone_id for z in helpers}
    members: list[ZoneSnapshot] = []
    for zone in zones:
        if zone.zone_id in taken or not zone.enabled or zone.temp is None:
            continue
        currently = zone.is_on or state.zone_on.get(zone.zone_id, False)
        if not currently:
            continue
        lo, hi = bands[zone.zone_id]
        if _reached_far_edge(zone, lo, hi, mode):
            continue
        if not _park_ok_now(zone, lo, hi, mode):
            continue
        members.append(zone)
    residual = {z.zone_id for z in members if z.park_residuals is True}
    if residual:
        members = [z for z in members if z.park_current_is_fan_type is not True]
    ids = {z.zone_id for z in members}
    if len(ids) >= 2:
        return ids
    if ids and demand and energy_save:
        return ids
    return set()


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
# Sibling-sustain: EWMA in-band ratio while a rider zone free-rides a
# conditioning sibling. Opportunistic observation only - see
# _update_sibling_sustain / _sibling_sustain_ok.
SIBLING_SUSTAIN_ALPHA = 0.2
SIBLING_SUSTAIN_MIN_SAMPLES = 6
SIBLING_SUSTAIN_MIN_RATIO = 0.6
# Mixing free-ride for out-of-band rooms: free-float (heads off, house
# mixing included) must enter the band within this many hours — hold-load
# cover alone is not enough (that only stops drift, it does not pull down).
FREE_RIDE_PULL_HORIZON_H = 3
# Transport grace: cold/warm air from a sibling takes minutes to reach a
# far room. While the youngest conditioning sibling is still this young,
# permit free-ride on hold-cover alone so we do not recruit a head for a
# lag that mixing is about to clear.
FREE_RIDE_TRANSPORT_GRACE_S = 900.0


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


def _predicted_breach(
    zone: ZoneSnapshot, lo: float, hi: float, mode: str
) -> tuple[float | None, float]:
    """(time_to_breach_h, peak_breach_k) from free_float, else pred_60m.

    Peak breach is max excursion past the band edge over the scored horizon
    (hours where the trajectory is already out). None ttb means no breach.
    """
    traj = list(zone.free_float) if zone.free_float else []
    if not traj and zone.pred_60m is not None:
        traj = [zone.temp if zone.temp is not None else zone.pred_60m, zone.pred_60m]
    ttb: float | None = None
    peak = 0.0
    for i, t in enumerate(traj[:12]):
        if t is None:
            continue
        if mode == MODE_COOL:
            excess = t - hi
        elif mode == MODE_HEAT:
            excess = lo - t
        else:
            return None, 0.0
        if excess > 0.0:
            if ttb is None:
                ttb = float(i)
            peak = max(peak, excess)
    return ttb, peak


def _prediction_horizon_h(min_on_min: float, cop_timing: str = "none") -> float:
    """How far inaction is scored (h). COP advance/defer stretch this only."""
    min_on_h = max(min_on_min / 60.0, 1.0 / 60.0)
    horizon = max(1.0, 2.0 * min_on_h)
    if cop_timing == "advance":
        horizon = max(horizon, COP_ADVANCE_HORIZON_H)
    elif cop_timing == "defer":
        horizon = min(horizon, max(min_on_h, COP_DEFER_HORIZON_H))
    return horizon


def _prediction_justifies_run(
    zone: ZoneSnapshot,
    lo: float,
    hi: float,
    mode: str,
    min_on_min: float,
    cop_timing: str = "none",
) -> bool:
    """True when predicted breach size and time-to-breach justify one plant min run.

    Uses the prediction — no floor/center deadband that ignores it. A small
    far-away breach fails naturally (ttb ≫ min_on or peak < PREDICT_MARGIN_K).
    COP timing stretches (advance) or shortens (defer) the allowed horizon.
    """
    ttb_h, peak = _predicted_breach(zone, lo, hi, mode)
    if ttb_h is None or peak < PREDICT_MARGIN_K:
        return False
    return ttb_h <= _prediction_horizon_h(min_on_min, cop_timing)


def _hourly_at_path(temps: tuple[float, ...] | list[float], hour: float) -> float:
    """Linear interpolate an hourly trajectory (index 0 = now)."""
    if not temps:
        return 0.0
    if hour <= 0.0:
        return float(temps[0])
    i = int(hour)
    if i >= len(temps) - 1:
        return float(temps[-1])
    frac = hour - i
    return float(temps[i]) * (1.0 - frac) + float(temps[i + 1]) * frac


def _oob_k(temp: float, lo: float, hi: float) -> float:
    return max(0.0, temp - hi) + max(0.0, lo - temp)


def _path_comfort_cost(
    temps: tuple[float, ...] | list[float],
    lo: float,
    hi: float,
    horizon_h: float,
    q_hvac_k_per_h: float = 0.0,
    q_on_h: float | None = None,
) -> tuple[float, float]:
    """(K·h outside [lo, hi], peak OOB K) over the horizon.

    Optional ``q_hvac_k_per_h`` overlays AC for the first ``q_on_h`` hours
    (default: the whole horizon). After that the offset holds so a min_on
    pull-down is scored against the same window as inaction.
    """
    if not temps or horizon_h <= 0.0:
        return 0.0, 0.0
    on_h = horizon_h if q_on_h is None else max(0.0, q_on_h)
    steps = max(1, math.ceil(horizon_h * 6.0))
    dt = horizon_h / steps
    peak = 0.0
    cost = 0.0
    prev = None
    for i in range(steps + 1):
        h = i * dt
        temp = _hourly_at_path(temps, h) + q_hvac_k_per_h * min(h, on_h)
        oob = _oob_k(temp, lo, hi)
        peak = max(peak, oob)
        if prev is not None:
            cost += 0.5 * (prev + oob) * dt
        prev = oob
    return cost, peak


def _wants_conditioning(
    zone: ZoneSnapshot,
    lo: float,
    hi: float,
    mode: str,
    min_on_min: float = 20.0,
    cop_timing: str = "none",
) -> bool:
    """Demand: in-band uses prediction (COP advance); present OOB may nick.

    Far-edge (cool temp≤lo / heat temp≥hi) is already overshoot — do not keep
    digging. In-band never cost-compares a min_on overlay (that killed COP
    advance once the runtime always set q_hvac). Present OOB starts unless a
    same-sign q_hvac shows a min_on run costs more comfort than riding the
    nick. Wrong-sign / missing q_hvac honors the breach.
    """
    if zone.temp is None:
        return False
    if mode == MODE_COOL:
        if zone.temp <= lo:
            return False
        present_oob = zone.temp > hi
    elif mode == MODE_HEAT:
        if zone.temp >= hi:
            return False
        present_oob = zone.temp < lo
    else:
        return False
    if not zone.free_float:
        return present_oob
    if not present_oob:
        return _prediction_justifies_run(zone, lo, hi, mode, min_on_min, cop_timing)
    q_hvac = zone.q_hvac_k_per_h
    if q_hvac is None:
        return True
    if mode == MODE_COOL and q_hvac >= 0.0:
        return True
    if mode == MODE_HEAT and q_hvac <= 0.0:
        return True
    min_on_h = max(min_on_min / 60.0, 1.0 / 60.0)
    horizon_h = max(1.0, 2.0 * min_on_h)
    cost_in, peak_in = _path_comfort_cost(zone.free_float, lo, hi, horizon_h)
    if peak_in < PREDICT_MARGIN_K:
        return False
    cost_act, _peak_act = _path_comfort_cost(zone.free_float, lo, hi, horizon_h, q_hvac, min_on_h)
    return cost_in > cost_act


def _predicted_oob_duration_s(zone: ZoneSnapshot, lo: float, hi: float, mode: str) -> float:
    """Hours of free-float spent out of band, as seconds (handoff gate)."""
    hours = 0
    for t in list(zone.free_float)[:12]:
        if t is None:
            continue
        if (mode == MODE_COOL and t > hi) or (mode == MODE_HEAT and t < lo):
            hours += 1
    if (
        hours == 0
        and zone.pred_60m is not None
        and (
            (mode == MODE_COOL and zone.pred_60m > hi) or (mode == MODE_HEAT and zone.pred_60m < lo)
        )
    ):
        hours = 1
    return float(hours) * 3600.0


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


def _park_exploit_ok(zone: ZoneSnapshot, mode: str) -> bool:
    """Known residual head whose parked output plausibly carries the standing load.

    park_extraction_w is sensed-room / per-head; standing_load_w is zone-total
    (x n_rooms). Compare in the per-room frame so mirrored zones are not
    falsely judged unable to cover their load.
    """
    if zone.park_residuals is not True:
        return False
    if zone.park_extraction_w is None:
        return False
    if zone.standing_load_w is None:
        # No load estimate: residual hold while satisfied is still calmer than
        # off/on cycling, accept.
        return True
    per_room_load = zone.standing_load_w / max(1, zone.n_rooms)
    return zone.park_extraction_w >= PARK_LOAD_COVER_FRACTION * per_room_load


def _command_with_depth(
    zid: str,
    mode: str,
    setpoint: float | None,
    reason: str,
    depth_k: float,
) -> Command:
    """Build a Command from signed head depth (cool: SP = internal + depth)."""
    d = float(depth_k)
    if d > 0.0:
        return Command(zid, mode, setpoint, reason, head_depth_k=d)
    if d < 0.0:
        return Command(
            zid,
            mode,
            setpoint,
            reason,
            track_delta=-d,
            head_depth_k=d,
        )
    return Command(zid, mode, setpoint, reason, head_depth_k=0.0)


def _ideal_device_setpoint(internal: float, depth_k: float, mode: str) -> float:
    """Device SP that realizes signed depth at the live internal reading."""
    if mode == MODE_COOL:
        return internal + depth_k
    return internal - depth_k


def hold_device_setpoint(
    internal: float,
    depth_k: float,
    mode: str,
    last_sp: float | None,
    last_depth: float | None,
) -> float:
    """Device SP for positive depth: freeze last commanded SP; step by Δdepth.

    Re-anchoring ``internal ± depth`` while the supply-air reading falls is
    the West descending ladder. First hold command still uses the live
    internal. Later, unchanged depth keeps the last commanded SP; a depth
    change moves that frozen SP by Δdepth (cool: ``last_sp + Δd``), which
    can step toward internal on a step-down.
    """
    ideal = _ideal_device_setpoint(internal, depth_k, mode)
    if last_sp is None or last_depth is None:
        return ideal
    dd = float(depth_k) - float(last_depth)
    if mode == MODE_COOL:
        return float(last_sp) + dd
    if mode == MODE_HEAT:
        return float(last_sp) - dd
    return ideal


def _head_anchor_drifted(zone: ZoneSnapshot, depth_k: float, mode: str) -> bool:
    """True when device SP no longer matches internal ± depth within epsilon."""
    if mode not in (MODE_HEAT, MODE_COOL):
        return False
    internal = zone.head_internal_temp
    device_sp = zone.device_setpoint
    if internal is None or device_sp is None:
        return False
    ideal = _ideal_device_setpoint(float(internal), depth_k, mode)
    return abs(float(device_sp) - ideal) >= HEAD_REANCHOR_EPS_K


def _depth_command_due(
    state: ControllerState,
    zid: str,
    now: float,
    zone: ZoneSnapshot,
    depth_k: float | None,
    mode: str,
    *,
    force: bool = False,
) -> bool:
    """Emit a depth command on spacing (chase only), forced change, or chase drift.

    Positive park depth is a frozen hold: coordinator would rebuild
    ``SP = internal + depth`` from a falling supply-air reading. Depth
    steps still emit via ``force`` (margin escalate, coil-wet walk).
    """
    if force:
        return True
    if depth_k is not None and depth_k > 0.0:
        return False
    last = state.zone_last_cmd.get(zid, 0.0)
    elapsed = now - last
    if elapsed >= COMMAND_SPACING_S:
        return True
    return (
        depth_k is not None
        and depth_k < 0.0
        and mode in (MODE_HEAT, MODE_COOL)
        and elapsed >= HEAD_REANCHOR_MIN_S
        and _head_anchor_drifted(zone, depth_k, mode)
    )


def _mapped_without_residual(zone: ZoneSnapshot) -> bool:
    """Hysteresis map has samples but no residual bin (idle / unknown only)."""
    if zone.park_residual_max_margin_k is not None:
        return False
    has_samples = False
    for entry in (zone.park_margin_bins or {}).values():
        if len(entry) < 3 or int(entry[2]) < CLASSIFY_MIN_SAMPLES:
            continue
        has_samples = True
        if usable_residual_bin(float(entry[0]), float(entry[1]), float(entry[2])):
            return False
    return has_samples


def _hysteresis_ceiling_k(zone: ZoneSnapshot) -> float:
    """Over-conditioning extreme: deepest still COP/residual-useful depth.

    When the residual map has a proven max (or edge), that caps escalation —
    deeper is COP-useless fan-type hold. While unclassified / unmapped,
    allow full ``DEPTH_MAX_K`` so a session can still learn by probing up.
    Preferred is an *entry* hint, not this ceiling.
    """
    if zone.park_residual_max_margin_k is not None:
        return min(_quantize_depth(zone.park_residual_max_margin_k), DEPTH_MAX_K)
    if _mapped_without_residual(zone):
        return PARK_MARGIN_K
    if zone.park_residual_edge_k is not None:
        return min(_quantize_depth(zone.park_residual_edge_k), DEPTH_MAX_K)
    return DEPTH_MAX_K


def _unintentional_overshoot(
    zone: ZoneSnapshot,
    mode: str,
    base_lo: float,
    base_hi: float,
    *,
    timing: str,
    widen_k: float,
) -> bool:
    """True when the room is past the unwidened far edge for no banked reason.

    COP-advance sitting between the tight and widened edge is intentional.
    Below the tight floor (cool) / above the tight ceiling (heat) by the
    overcool buffer is not.
    """
    if zone.temp is None or mode not in (MODE_HEAT, MODE_COOL):
        return False
    if mode == MODE_COOL:
        if zone.temp > base_lo - PARK_OVERCOOL_BUFFER_K:
            return False
        banked = timing == "advance" and widen_k > 0.0 and zone.temp >= base_lo - widen_k
        return not banked
    if zone.temp < base_hi + PARK_OVERCOOL_BUFFER_K:
        return False
    banked = timing == "advance" and widen_k > 0.0 and zone.temp <= base_hi + widen_k
    return not banked


def _overshoot_ceiling_k(
    zone: ZoneSnapshot, state: ControllerState, depth_k: float, *, walk: bool
) -> float:
    """Lift the learned ceiling only while unintentional overshoot can still deepen."""
    _ = state
    mapped = _hysteresis_ceiling_k(zone)
    if not walk:
        return mapped
    # Live fan-type is the electrical stop. A mapped idle bin at this depth
    # is why we walk — West's 0.5 K shelf was "idle" while the head still
    # compressed as SP followed a falling internal.
    if zone.park_current_is_fan_type is True:
        return max(mapped, _quantize_depth(depth_k))
    return DEPTH_MAX_K


def _extraction_none_at_depth(zone: ZoneSnapshot, depth_k: float) -> bool:
    """True when this positive depth is not moving useful heat / not compressing."""
    if depth_k <= 0.0:
        return False
    if zone.park_current_is_fan_type is True:
        return True
    if zone.park_residuals is False:
        return True
    from .park import CLASSIFY_MIN_SAMPLES, IDLE_MAX_W, RESIDUAL_HOLD_DUTY_MIN, margin_bin

    b = margin_bin(depth_k)
    if b is None:
        return False
    entry = (zone.park_margin_bins or {}).get(b)
    if entry is None or entry[2] < CLASSIFY_MIN_SAMPLES:
        return False
    return entry[1] < RESIDUAL_HOLD_DUTY_MIN or float(entry[0]) <= IDLE_MAX_W


def _shallower_depth_useful(
    zone: ZoneSnapshot,
    depth_k: float,
    *,
    want_conditioning: bool,
    mixing_covered: bool,
) -> bool:
    """Can a lower depth restore compression / useful in-band work?"""
    if mixing_covered and not want_conditioning:
        return False
    from .park import CLASSIFY_MIN_SAMPLES, IDLE_MAX_W, RESIDUAL_HOLD_DUTY_MIN

    for key, (ext, duty, n) in (zone.park_margin_bins or {}).items():
        try:
            m = float(key)
        except (TypeError, ValueError):
            continue
        if (
            m < depth_k - 1e-9
            and n >= CLASSIFY_MIN_SAMPLES
            and duty >= RESIDUAL_HOLD_DUTY_MIN
            and float(ext) > IDLE_MAX_W
        ):
            return True
    # Chase / zero always available when the zone still wants conditioning.
    return want_conditioning


def _adapt_head_depth_k(
    state: ControllerState,
    zid: str,
    zone: ZoneSnapshot,
    mode: str,
    depth_k: float,
    *,
    want_conditioning: bool,
    mixing_covered: bool = False,
    pull_down: bool = False,
    overshoot_walk: bool = False,
) -> tuple[float, bool, bool]:
    """Signed depth adapt. Returns ``(depth, changed, should_off)``.

    Under-conditioning lowers depth through 0 into chase, floored at the
    live tracking chase on the 0.5 grid (``-_chase_floor_k``). Over-
    conditioning raises depth only up to the deepest still COP/residual-
    useful hold (``_hysteresis_ceiling_k``). When extraction is ~none at
    this depth, step down only if a shallower depth can restore useful
    work; otherwise ``should_off`` (future overcooling / no need).
    """
    floor = _chase_floor_k(state, zid)
    ceiling = _overshoot_ceiling_k(zone, state, depth_k, walk=overshoot_walk)
    depth = _clamp_depth(depth_k, floor, ceiling)
    if zone.temp is None or mode not in (MODE_HEAT, MODE_COOL):
        return depth, False, False
    if pull_down:
        # Fast path toward the chase floor when the near edge is breached.
        target = _clamp_depth(min(depth, floor), floor, ceiling)
        if abs(target - depth) > 1e-9:
            state.zone_park_ref[zid] = zone.temp
            return target, True, False
        return depth, False, False

    ref = state.zone_park_ref.get(zid)
    if ref is None:
        state.zone_park_ref[zid] = zone.temp
        return depth, False, False
    moved = ref - zone.temp if mode == MODE_COOL else zone.temp - ref
    changed = False
    should_off = False
    if moved >= PARK_ADAPT_EPS_K:
        # Over-holding: raise depth toward the COP-useful ceiling, then hold
        # there (keep-temp). Far-edge overcorrection is the caller's job —
        # do not soft-release merely because we are already at useful max.
        if depth < ceiling - 1e-9:
            depth = _clamp_depth(depth + DEPTH_STEP_K, floor, ceiling)
            changed = True
            if depth > 0.0:
                _raise_park_preferred(state, zid, depth)
        state.zone_park_ref[zid] = zone.temp
    elif moved <= -PARK_ADAPT_EPS_K:
        # Under-holding / load side: step toward chase floor only.
        none = _extraction_none_at_depth(zone, depth)
        if none and not _shallower_depth_useful(
            zone, depth, want_conditioning=want_conditioning, mixing_covered=mixing_covered
        ):
            # Unintentional overshoot: stay at this depth (retrain), do not
            # fall through into chase. Normal path may idle.
            if not overshoot_walk:
                should_off = True
            state.zone_park_ref[zid] = zone.temp
        else:
            new_depth = _clamp_depth(depth - DEPTH_STEP_K, floor, ceiling)
            if abs(new_depth - depth) > 1e-9:
                depth = new_depth
                changed = True
            elif not want_conditioning and none and depth <= floor + 1e-9:
                # Already at chase floor with no useful shallower depth.
                should_off = True
            state.zone_park_ref[zid] = zone.temp
    return depth, changed, should_off


def _park_preferred(zone: ZoneSnapshot, state: ControllerState) -> float:
    """Learned park depth for this zone, falling back to the base margin.

    Once the hysteresis map has found a residual-hold depth (deepest margin
    still meaningfully compressing, just before the dead-band edge), prefer
    that: it is real compressor output at the cheapest depth that gives it,
    ranked above a fan-type hold (fan running, no output) or a stale default.
    """
    zid = zone.zone_id
    if zid in state.zone_park_preferred:
        pref = state.zone_park_preferred[zid]
    elif zone.park_residual_edge_k is not None:
        pref = min(max(zone.park_residual_edge_k, PARK_MARGIN_K), PARK_MARGIN_MAX_K)
    elif zone.park_preferred_margin_k is not None:
        pref = zone.park_preferred_margin_k
    else:
        pref = PARK_MARGIN_K
    cap = zone.park_residual_max_margin_k
    if cap is not None:
        pref = min(pref, max(float(cap), PARK_MARGIN_K))
    elif _mapped_without_residual(zone):
        pref = min(pref, PARK_MARGIN_K)
    return pref


def _park_entry_margin(zone: ZoneSnapshot, state: ControllerState) -> float:
    """Park entry depth: learned residual-hold edge when mapped, else preferred - undershoot.

    Deliberately does not fall back to fan_type_min_margin_k here: a fan-type
    depth exploits nothing (fan running, no compression), so entry prefers
    either a proven residual-hold depth or the undershoot-from-preferred
    probe, never the fan-type shelf.
    """
    if zone.park_residual_edge_k is not None:
        # Land at the deepest depth still proven to compress; undershooting
        # further would give up real output for no reason.
        return min(max(zone.park_residual_edge_k, PARK_MARGIN_K), PARK_MARGIN_MAX_K)
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


def _sibling_sustain_ok(state: ControllerState, rider_id: str, sibling_ids: list[str]) -> bool:
    """No history yet -> permissive. Learned-poor pairs veto the physical estimate.

    Never blocks purely for lack of data (asymmetric learning is fine, and a
    fresh install must not be denied the model's own physical estimate); a
    pair with enough samples showing the room does NOT actually stay in-band
    overrides an over-optimistic instantaneous mixing calculation.
    """
    data = state.sibling_sustain.get(rider_id)
    if not data:
        return True
    for sid in sibling_ids:
        entry = data.get(sid)
        if entry is None:
            continue
        ratio, samples = entry
        if samples >= SIBLING_SUSTAIN_MIN_SAMPLES and ratio < SIBLING_SUSTAIN_MIN_RATIO:
            return False
    return True


def _update_sibling_sustain(
    state: ControllerState,
    zones: list[ZoneSnapshot],
    mode: str,
    bands: dict[str, tuple[float, float]],
    want_on: dict[str, bool],
) -> None:
    """Observe (never force) whether a free-riding zone stays in-band.

    Ordered and asymmetric by construction: only zones that are already off
    and not wanted on are observed against zones that are actively
    conditioning right now. No reverse-direction probing is ever triggered.
    """
    if mode not in (MODE_HEAT, MODE_COOL):
        return
    conditioning = [z.zone_id for z in zones if z.is_on]
    if not conditioning:
        return
    for zone in zones:
        zid = zone.zone_id
        if zone.is_on or want_on.get(zid) or zone.temp is None:
            continue
        lo, hi = bands.get(zid, (None, None))
        if lo is None or hi is None:
            continue
        in_band = lo <= zone.temp <= hi
        for sib in conditioning:
            if sib == zid:
                continue
            entry = state.sibling_sustain.setdefault(zid, {})
            ratio, samples = entry.get(sib, (1.0 if in_band else 0.0, 0))
            ratio += SIBLING_SUSTAIN_ALPHA * ((1.0 if in_band else 0.0) - ratio)
            entry[sib] = (ratio, samples + 1)


def _mixing_assist_covers_hold(zone: ZoneSnapshot, mode: str) -> bool:
    """True when instantaneous mixing flux covers hold (standing) load."""
    if zone.mixing_gain_w is None or zone.standing_load_w is None:
        return False
    if mode == MODE_COOL:
        assist_w = max(0.0, -zone.mixing_gain_w)
    elif mode == MODE_HEAT:
        assist_w = max(0.0, zone.mixing_gain_w)
    else:
        return False
    per_room_load = zone.standing_load_w / max(1, zone.n_rooms)
    return assist_w >= PARK_LOAD_COVER_FRACTION * per_room_load


def _mixing_will_condition(zone: ZoneSnapshot, lo: float, hi: float, mode: str) -> bool:
    """True when free-float enters the band on the conditioning side.

    free_float already includes house-mixing from siblings (coordinator
    builds t_house_hourly with on-siblings converging to their centres),
    so this is the honest "will mixing pull this room into comfort?" test
    — stronger than hold-load cover alone.
    """
    traj = list(zone.free_float) if zone.free_float else []
    if not traj and zone.pred_60m is not None and zone.temp is not None:
        traj = [zone.temp, zone.pred_60m]
    # Skip hour 0 (current reading — already known OOB); score 1..horizon.
    for t in traj[1 : FREE_RIDE_PULL_HORIZON_H + 1]:
        if t is None:
            continue
        if mode == MODE_COOL and t <= hi:
            return True
        if mode == MODE_HEAT and t >= lo:
            return True
    return False


def _mixing_transport_grace(
    state: ControllerState, conditioning_sibling_ids: list[str], now: float
) -> bool:
    """True while a conditioning sibling is still within transport grace.

    Far rooms lag the instantaneous mixing estimate: air has not arrived
    yet, so the rider looks hotter/colder than it will once the plume
    reaches it. Hold-cover during that window is enough to wait.
    """
    for sid in conditioning_sibling_ids:
        if not state.zone_on.get(sid, False):
            continue
        since = state.zone_since.get(sid, 0.0)
        if since > 0.0 and now - since < FREE_RIDE_TRANSPORT_GRACE_S:
            return True
    return False


def _mixing_free_rider(
    zone: ZoneSnapshot,
    mode: str,
    state: ControllerState,
    conditioning_sibling_ids: list[str],
    lo: float,
    hi: float,
    now: float,
) -> bool:
    """True when a conditioning sibling already covers this zone via mixing.

    In-band (conditioning-side): hold-load cover is enough — the room does
    not need pull-down. Out-of-band: mixing must *condition* (free-float
    enters the band within FREE_RIDE_PULL_HORIZON_H) or we are still inside
    FREE_RIDE_TRANSPORT_GRACE_S of a sibling start (air-travel leniency).
    Sibling-sustain history can veto an over-optimistic physical estimate.
    """
    if not conditioning_sibling_ids:
        return False
    if not _mixing_assist_covers_hold(zone, mode):
        return False
    if not _sibling_sustain_ok(state, zone.zone_id, conditioning_sibling_ids):
        return False
    # Conditioning-side in-band: cool temp≤hi / heat temp≥lo — hold is enough.
    if _park_ok_now(zone, lo, hi, mode):
        return True
    if _mixing_will_condition(zone, lo, hi, mode):
        return True
    return _mixing_transport_grace(state, conditioning_sibling_ids, now)


def _all_zones_satisfied(zones: list[ZoneSnapshot], centers: dict[str, float], mode: str) -> bool:
    """True when every enabled zone with a reading is at/past its own centre."""
    for zone in zones:
        if zone.temp is None or not zone.enabled:
            continue
        center = centers.get(zone.zone_id)
        if center is None:
            continue
        if mode == MODE_COOL and zone.temp > center:
            return False
        if mode == MODE_HEAT and zone.temp < center:
            return False
    return True


def _house_overserve_margin_k(
    zones: list[ZoneSnapshot], bands: dict[str, tuple[float, float]], mode: str
) -> float | None:
    """Head-count-weighted predicted drift past the tightest band edge (K, >0
    means the house is trending past comfort, i.e. the anchor is over-serving).
    """
    weighted = [(z.pred_60m, z.n_rooms) for z in zones if z.pred_60m is not None and z.enabled]
    total_w = sum(w for _v, w in weighted)
    if not weighted or total_w <= 0 or not bands:
        return None
    avg_pred = sum(v * w for v, w in weighted) / total_w
    if mode == MODE_COOL:
        return min(lo for lo, _hi in bands.values()) - avg_pred
    if mode == MODE_HEAT:
        return avg_pred - max(hi for _lo, hi in bands.values())
    return None


def _anchor_should_release(
    state: ControllerState,
    zones: list[ZoneSnapshot],
    centers: dict[str, float],
    bands: dict[str, tuple[float, float]],
    mode: str,
    want_on: dict[str, bool],
    now: float,
) -> tuple[bool, float | None]:
    anchor = next((z for z in zones if z.zone_id == state.anchor_zone), None)
    if anchor is None:
        return True, None
    lo, hi = bands.get(state.anchor_zone, (None, None))
    if anchor.temp is not None and lo is not None and hi is not None:
        if mode == MODE_COOL and anchor.temp <= lo - PARK_OVERCOOL_BUFFER_K:
            return True, None
        if mode == MODE_HEAT and anchor.temp >= hi + PARK_OVERCOOL_BUFFER_K:
            return True, None
    margin = _house_overserve_margin_k(zones, bands, mode)
    # Handoff: another zone is (or will be) conditioning and house mixing
    # already covers this zone's standing load — residual park on the
    # anchor would add compressor load for free. Prefer_continuous only
    # needs *someone* loaded; shift the working head to the demand zone.
    sibs = sorted(
        {
            z.zone_id
            for z in zones
            if z.zone_id != state.anchor_zone and (want_on.get(z.zone_id) or z.is_on)
        }
    )
    if (
        sibs
        and lo is not None
        and hi is not None
        and _mixing_free_rider(anchor, mode, state, sibs, lo, hi, now)
    ):
        return True, margin
    if not _all_zones_satisfied(zones, centers, mode):
        return False, margin
    return (margin is not None and margin > 0.0), margin


def _select_anchor(
    zones: list[ZoneSnapshot],
    mode: str,
    shed: dict,
    quiet_ids: set[str] | None = None,
) -> str | None:
    """Elect the anchor from already-loaded heads: best load coverage with
    least excess, tie-broken by head count (mirrored zones commit every head)
    then mixing centrality. Never recruits an idle zone just to be the anchor.
    Quiet-night sleep rooms lose to any other already-on candidate.
    """
    candidates = [
        z
        for z in zones
        if z.is_on
        and z.enabled
        and z.zone_id not in shed
        and (z.head_mode == mode or z.head_mode is None)
    ]
    if quiet_ids:
        preferred = [z for z in candidates if z.zone_id not in quiet_ids]
        if preferred:
            candidates = preferred
    if not candidates:
        return None
    total_load = sum(z.standing_load_w or 0.0 for z in zones if z.enabled)

    def score(z: ZoneSnapshot) -> tuple:
        coverage = (z.park_extraction_w or 0.0) * z.n_rooms
        covers = coverage >= total_load
        return (
            0 if covers else 1,
            (coverage - total_load) if covers else -coverage,
            -z.n_rooms,
            -(z.mixing_coupling_w_per_k or 0.0),
        )

    return min(candidates, key=score).zone_id


def _update_anchor(
    state: ControllerState,
    snap: HouseSnapshot,
    zones: list[ZoneSnapshot],
    mode: str,
    centers: dict[str, float],
    bands: dict[str, tuple[float, float]],
    want_on: dict[str, bool],
) -> float | None:
    """prefer_continuous: keep already-on heads loaded (demand or park).

    Elects one already-on head as the N=1 leftover when the pack has
    drained. N≥2 stay-loaded is ``pack_ids`` (every in-band coil), not
    this single leftover. Does *not* force demand setpoints — parking is
    allowed. Releases when overserving, overcorrected, or when a
    *demand* sibling's conditioning already covers the leftover via
    mixing (handoff). Pack members are not peeled because mix could
    carry them.

    Returns the current over-serve margin (K) for diagnostics, or None.
    """
    s = snap.settings
    eligible = (
        s.prefer_continuous
        and s.park_learning
        and s.multisplit
        and s.preset != PRESET_MANUAL
        and mode in (MODE_HEAT, MODE_COOL)
    )
    if not eligible:
        state.anchor_zone = None
        state.anchor_since = 0.0
        return None
    zone_ids = {z.zone_id for z in zones}
    if state.anchor_zone is not None and state.anchor_zone not in zone_ids:
        state.anchor_zone = None
        state.anchor_since = 0.0
    if state.anchor_zone is not None:
        should_release, margin = _anchor_should_release(
            state, zones, centers, bands, mode, want_on, snap.now_ts
        )
        if should_release:
            state.anchor_zone = None
            state.anchor_since = 0.0
            return None
        return margin
    candidate = _select_anchor(
        zones,
        mode,
        state.shed,
        {z.zone_id for z in zones if _quiet_night_active(s, z.zone_id, snap.local_hour, mode)},
    )
    if candidate is not None:
        state.anchor_zone = candidate
        state.anchor_since = snap.now_ts
    return None


def _anchor_active(
    state: ControllerState,
    s,
    mode: str,
    zid: str,
    pack_ids: set[str] | None = None,
) -> bool:
    """True while prefer_continuous should keep this head loaded.

    The elected leftover (`anchor_zone`) covers N=1. Pack members
    (`pack_ids`) stay loaded together so continuous does not collapse
    to a single cold coil.
    """
    if (
        not s.prefer_continuous
        or not s.park_learning
        or not s.multisplit
        or s.preset == PRESET_MANUAL
        or mode not in (MODE_HEAT, MODE_COOL)
        or zid in state.shed
    ):
        return False
    if state.anchor_zone == zid:
        return True
    return bool(pack_ids) and zid in pack_ids


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

    # 2. COP-timed efficiency band *before* mode arbitration. Mode must see
    # the widened edges — otherwise a cool-advance bank below the tight lo
    # looks like a cold deficit and auto flips to heat in summer. Use the
    # latched widen mode (or last/forced conditioning mode) for stretch.
    if s.hvac_mode in (MODE_HEAT, MODE_COOL):
        widen_mode: str | None = s.hvac_mode
    elif state.cop_widen_mode in (MODE_HEAT, MODE_COOL):
        widen_mode = state.cop_widen_mode
    elif state.mode in (MODE_HEAT, MODE_COOL):
        widen_mode = state.mode
    else:
        widen_mode = None
    if widen_mode is not None:
        widen_k, cop_timing = _update_cop_widen(state, snap, widen_mode, centers, zones)
        apply_mode = state.cop_widen_mode or widen_mode
        _apply_cop_widen_bands(zones, bands, centers, snap, preset, apply_mode, widen_k)
        if widen_k > 0.0:
            diag["bands"] = {z: bands[z] for z in bands}
    else:
        _clear_cop_widen(state)
        widen_k, cop_timing = 0.0, "none"
    diag["cop_band_widen_k"] = widen_k
    diag["cop_timing"] = cop_timing

    # 3. Mode arbitration on the (possibly widened) bands.
    if s.hvac_mode == MODE_OFF:
        mode = MODE_OFF
        diag["mode_source"] = "forced"
        state.mode = MODE_OFF
        _clear_cop_widen(state)
        widen_k, cop_timing = 0.0, "none"
        # Rebuild tight bands after clearing widen for an off hub.
        for zone in zones:
            bands[zone.zone_id] = comfort.zone_band(
                s, centers[zone.zone_id], zone.occupied, snap.house_occupied
            )
        diag["bands"] = {z: bands[z] for z in bands}
        diag["cop_band_widen_k"] = 0.0
        diag["cop_timing"] = "none"
    elif s.preset == PRESET_MANUAL:
        # User owns the plant. Do not re-arbitrate: a manual pulldown below
        # the band would otherwise elect heat and latch MODE_DWELL_S.
        mode = state.mode if state.mode in (MODE_HEAT, MODE_COOL, MODE_OFF) else MODE_OFF
        diag["mode_source"] = "manual"
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

    # If auto settled on the opposite conditioning mode, refresh widen once
    # so heat/cool stretch matches the live decision.
    if (
        s.hvac_mode == MODE_AUTO
        and mode in (MODE_HEAT, MODE_COOL)
        and state.cop_widen_mode is not None
        and state.cop_widen_mode != mode
        and not (
            state.cop_widen_k > 0.0
            and _outside_base_band_on_widen_side(
                zones, centers, snap, state.cop_widen_mode, state.cop_timing
            )
        )
    ):
        # Only restretch when not mid recovery on the latched side.
        for zone in zones:
            bands[zone.zone_id] = comfort.zone_band(
                s, centers[zone.zone_id], zone.occupied, snap.house_occupied
            )
        widen_k, cop_timing = _update_cop_widen(state, snap, mode, centers, zones)
        _apply_cop_widen_bands(zones, bands, centers, snap, preset, mode, widen_k)
        diag["bands"] = {z: bands[z] for z in bands}
        diag["cop_band_widen_k"] = widen_k
        diag["cop_timing"] = cop_timing

    # Plant compression clock (min_on keys off this, not zone_since).
    _update_plant_compress(state, snap)
    plant_min_on = _plant_min_on_active(state, snap, s)
    diag["plant_min_on"] = plant_min_on
    diag["plant_compress_since"] = state.plant_compress_since

    # 3. Demand and helper sets.
    demand = (
        [
            z
            for z in zones
            if _wants_conditioning(z, *bands[z.zone_id], mode, s.min_on_min, cop_timing)
        ]
        if mode in (MODE_HEAT, MODE_COOL)
        else []
    )
    # Mixing free-ride: drop zones a conditioning sibling already covers
    # via house mixing — hold when in-band, pull-down when free-float
    # shows the band is reachable (with transport grace for air lag).
    conditioning_sibs = [z.zone_id for z in zones if z.is_on]
    free_riders: list[str] = []
    if mode in (MODE_HEAT, MODE_COOL) and conditioning_sibs:
        kept: list[ZoneSnapshot] = []
        for zone in demand:
            sibs = [sid for sid in conditioning_sibs if sid != zone.zone_id]
            lo_b, hi_b = bands[zone.zone_id]
            if sibs and _mixing_free_rider(zone, mode, state, sibs, lo_b, hi_b, now):
                if _quiet_night_bank_active(s, zone.zone_id, snap.local_hour, mode):
                    kept.append(zone)
                else:
                    free_riders.append(zone.zone_id)
            else:
                kept.append(zone)
        demand = kept
    diag["free_riders"] = sorted(free_riders)
    banked: list[str] = []
    if mode in (MODE_HEAT, MODE_COOL) and not _energy_save_preset(preset):
        demand_ids = {z.zone_id for z in demand}
        for zone in zones:
            if zone.zone_id in demand_ids:
                continue
            if not _quiet_night_bank_active(s, zone.zone_id, snap.local_hour, mode):
                continue
            lo_b, hi_b = bands[zone.zone_id]
            if _quiet_night_bank_needed(zone, lo_b, hi_b, mode):
                demand.append(zone)
                demand_ids.add(zone.zone_id)
                banked.append(zone.zone_id)
    diag["quiet_night_bank"] = sorted(banked)
    # Already-shed rooms stay in `demand` until after this tick's shed/restore
    # so a restore can still emit. Helper recruitment ignores them so a
    # latched-shed sole-demand zone cannot recruit in-band helpers.
    live_demand = [z for z in demand if z.zone_id not in state.shed]
    live_demand_ids = {z.zone_id for z in live_demand}

    helpers: list[ZoneSnapshot] = []
    energy_save = _energy_save_preset(preset)
    if (
        live_demand
        and s.coordination
        and snap.settings.hvac_mode in (MODE_AUTO, mode)
        and not energy_save
    ):
        for zone in zones:
            if zone.zone_id in live_demand_ids:
                continue
            if zone.zone_id in free_riders:
                continue
            if zone.zone_id in state.shed:
                continue
            if _quiet_night_active(s, zone.zone_id, snap.local_hour, mode):
                continue
            if zone.temp is None:
                continue
            if _reached_far_edge(zone, *bands[zone.zone_id], mode):
                continue
            helpers.append(zone)
        helpers.sort(key=lambda z: (not z.is_on, z.zone_id))

    # Keep mixing leads on: free-ride drops the rider, not the covering head.
    if free_riders and s.coordination and not energy_save:
        taken = {z.zone_id for z in demand} | {z.zone_id for z in helpers}
        for zone in zones:
            if zone.zone_id in taken or zone.zone_id in free_riders:
                continue
            if zone.zone_id in state.shed:
                continue
            if not zone.is_on or zone.temp is None:
                continue
            if _reached_far_edge(zone, *bands[zone.zone_id], mode):
                continue
            helpers.append(zone)
            taken.add(zone.zone_id)

    # Empirical COP table: consolidate only from the live outdoor band.
    n_demand = sum(z.n_rooms for z in demand if z.zone_id not in state.shed)
    n_spread = n_demand + sum(z.n_rooms for z in helpers)
    cop_demand, cop_spread = _cop_n_pair(snap, n_demand, n_spread)
    cop_n_source = "prior"
    if (
        helpers
        and cop_spread is not None
        and cop_demand is not None
        and cop_demand > cop_spread * COP_TABLE_ADVANTAGE
    ):
        helpers = []
        diag["consolidated_by_cop_table"] = True
        cop_n_source = "banded_consolidate"
    elif cop_demand is not None and cop_spread is not None:
        cop_n_source = "banded_spread"
    diag["cop_n_source"] = cop_n_source

    demand, helpers, quiet_deferred, quiet_cover_id = _apply_quiet_night(
        zones, demand, helpers, s, snap, mode, bands, centers, energy_save=energy_save
    )

    # Eco/Away: wait *to start* until every enabled zone needs conditioning,
    # unless override or the N-table says the smaller demand set wins.
    # Once any head is already running, do not clear remaining demand —
    # satisfied rooms pack-stay instead of killing the hot room.
    energy_wait = False
    if energy_save and mode in (MODE_HEAT, MODE_COOL) and demand:
        enabled = [z for z in zones if z.enabled]
        all_need = all(
            _wants_conditioning(z, *bands[z.zone_id], mode, s.min_on_min, cop_timing)
            for z in enabled
        )
        override = any(_same_mode_override(z, *bands[z.zone_id], mode) for z in enabled)
        n_all = sum(z.n_rooms for z in enabled)
        n_dem = sum(z.n_rooms for z in demand)
        cd, cs = _cop_n_pair(snap, n_dem, n_all)
        consolidate_ok = cd is not None and cs is not None and cd > cs * COP_TABLE_ADVANTAGE
        already_running = any(z.is_on or state.zone_on.get(z.zone_id, False) for z in enabled)
        if not all_need and not override and not consolidate_ok and not already_running:
            demand = []
            helpers = []
            energy_wait = True
    diag["energy_wait"] = energy_wait

    demand_ids = {z.zone_id for z in demand}
    helper_ids = {z.zone_id for z in helpers}
    pack_ids = _pack_residual_ids(
        zones,
        bands,
        mode,
        state,
        energy_wait=energy_wait,
        consolidated=bool(diag.get("consolidated_by_cop_table")),
        demand=[z for z in demand if z.zone_id not in state.shed],
        helpers=helpers,
        coordination=s.coordination,
        energy_save=energy_save,
    )

    diag["demand"] = sorted(demand_ids)
    diag["helpers"] = sorted(helper_ids)
    diag["quiet_night_deferred"] = sorted(quiet_deferred)
    diag["n_heads"] = sum(z.n_rooms for z in demand) + sum(z.n_rooms for z in helpers)
    diag["spread_prior"] = diag.get("cop_n_source") == "prior" and bool(helpers)
    if quiet_cover_id:
        diag["quiet_cover_extended"] = quiet_cover_id
    if pack_ids:
        diag["pack_stay"] = sorted(pack_ids)

    want_on = {z.zone_id: (z.zone_id in demand_ids or z.zone_id in helper_ids) for z in zones}
    for zid in free_riders:
        want_on[zid] = False

    # Plant min_on handoff: while the compressor still owes runtime, a zone
    # that is satisfied may recruit another head only when that zone's
    # predicted out-of-band duration covers ≥ half min_on (no spurious starts).
    if plant_min_on and mode in (MODE_HEAT, MODE_COOL):
        half_min_s = 0.5 * s.min_on_min * 60.0
        any_wanted = any(want_on.values())
        if not any_wanted:
            ordered = sorted(
                zones,
                key=lambda z: (
                    0 if z.is_on else 1,
                    0 if not _quiet_night_active(s, z.zone_id, snap.local_hour, mode) else 1,
                ),
            )
            for zone in ordered:
                if zone.zone_id in state.shed or not zone.enabled:
                    continue
                lo_b, hi_b = bands[zone.zone_id]
                if _predicted_oob_duration_s(zone, lo_b, hi_b, mode) >= half_min_s:
                    want_on[zone.zone_id] = True
                    diag.setdefault("plant_handoff", []).append(zone.zone_id)
                    break

    _update_sibling_sustain(state, zones, mode, bands, want_on)
    # Anchor election after want_on is known so handoff can see demand sibs.
    # Do NOT force want_on[anchor]=True: that would block park entry
    # (park requires not desired_on). Anchor stay-loaded is via the park
    # branch's _anchor_active waiver, not tracked demand setpoints.
    oversve_margin = _update_anchor(state, snap, zones, mode, centers, bands, want_on)
    diag["anchor_zone"] = state.anchor_zone
    diag["anchor_overserve_margin_k"] = oversve_margin

    diag["want"] = {
        z.zone_id: (
            "demand"
            if z.zone_id in demand_ids
            else "helper"
            if z.zone_id in helper_ids
            else "anchor"
            if state.anchor_zone == z.zone_id
            else "free_ride"
            if z.zone_id in free_riders
            else "off"
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
            if (
                want_on.get(zid)
                or comfort.effective_zone_occupied(s, zone.occupied) is False
                or zid in state.shed
                or _quiet_night_active(s, zid, snap.local_hour, mode)
            ):
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
    if state.shed:
        shed_ids = set(state.shed)
        demand = [z for z in demand if z.zone_id not in shed_ids]
        helpers = [z for z in helpers if z.zone_id not in shed_ids]
        demand_ids = {z.zone_id for z in demand}
        helper_ids = {z.zone_id for z in helpers}
        pack_ids -= shed_ids
        for zid in shed_ids:
            want_on[zid] = False
        diag["demand"] = sorted(demand_ids)
        diag["helpers"] = sorted(helper_ids)
        diag["n_heads"] = sum(z.n_rooms for z in demand) + sum(z.n_rooms for z in helpers)
        if pack_ids:
            diag["pack_stay"] = sorted(pack_ids)
        elif "pack_stay" in diag:
            del diag["pack_stay"]

    overshoot_walk: dict[str, bool] = {}
    if mode in (MODE_HEAT, MODE_COOL):
        for zone in zones:
            base_lo, base_hi = comfort.zone_band(
                s, centers[zone.zone_id], zone.occupied, snap.house_occupied
            )
            overshoot_walk[zone.zone_id] = _unintentional_overshoot(
                zone, mode, base_lo, base_hi, timing=cop_timing, widen_k=widen_k
            )
        walking = [zid for zid, flag in overshoot_walk.items() if flag]
        if walking:
            diag["depth_overshoot_walk"] = walking

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
        # head is a known residual head whose output can carry the standing load,
        # we park instead: setpoint offset from the internal reading, mode
        # kept. Session margin starts at the learned preferred depth and
        # escalates while the room keeps moving in the conditioning direction.
        if zid not in state.zone_park_preferred:
            state.zone_park_preferred[zid] = _park_preferred(zone, state)
        parked_since = state.zone_parked_since.get(zid)
        if (
            parked_since is not None
            and desired_on
            and zid not in helper_ids
            and mode
            in (
                MODE_HEAT,
                MODE_COOL,
            )
        ):
            # Demand-side hysteresis depth: continuum adapt; still wants conditioning.
            lo_b, hi_b = bands[zid]
            preferred = state.zone_park_preferred.get(zid, PARK_MARGIN_K)
            depth = state.zone_head_depth_k.get(zid, state.zone_park_margin.get(zid, preferred))
            overcorrected = zone.temp is not None and (
                zone.temp <= lo_b - PARK_OVERCOOL_BUFFER_K
                if mode == MODE_COOL
                else zone.temp >= hi_b + PARK_OVERCOOL_BUFFER_K
            )
            pull_down = not _park_ok_now(zone, lo_b, hi_b, mode)
            walk = overshoot_walk.get(zid, False)
            depth, depth_changed, should_off = _adapt_head_depth_k(
                state,
                zid,
                zone,
                mode,
                depth,
                want_conditioning=True,
                pull_down=pull_down,
                overshoot_walk=walk,
            )
            floor = _chase_floor_k(state, zid)
            ceiling = _overshoot_ceiling_k(zone, state, depth, walk=walk)
            depth = _sync_depth_views(state, zid, depth, now, lo=floor, hi=ceiling, zone=zone)
            if walk and not should_off:
                if _depth_command_due(state, zid, now, zone, depth, mode, force=depth_changed):
                    commands.append(_command_with_depth(zid, mode, None, "depth_residual", depth))
                    state.zone_last_cmd[zid] = now
                state.zone_fan[zid] = False
                continue
            if overcorrected or should_off:
                _clear_park_session(state, zid, zone, now)
                diag.setdefault("park_overcorrected", []).append(zid)
                # Fall through to demand / off handling.
            elif depth > 0.0:
                if _depth_command_due(state, zid, now, zone, depth, mode, force=depth_changed):
                    commands.append(_command_with_depth(zid, mode, None, "depth_residual", depth))
                    state.zone_last_cmd[zid] = now
                state.zone_fan[zid] = False
                continue
            else:
                # Crossed into 0 / chase — fall through; demand emits continuum depth.
                diag.setdefault("depth_continuum", []).append(zid)
        if parked_since is not None and zid in state.zone_parked_since and zid not in helper_ids:
            # Direction: dominant mode when conditioning, else the head's own
            # physical mode - run-out parks must survive the house going idle.
            pmode = mode if mode in (MODE_HEAT, MODE_COOL) else zone.head_mode
            off_allowed = (
                s.hvac_mode == MODE_OFF
                or zid in state.shed
                or _transition_allowed(state, zid, now, False, s)
            )
            sibling_alive = any(want_on.get(z.zone_id) and z.zone_id != zid for z in zones) or (
                zid in pack_ids and len(pack_ids) >= 2
            )
            # Sibling conditioning + mixing already covers this zone's hold
            # load → residual park is pure extra compressor work; release.
            covering_sibs = [
                z.zone_id for z in zones if z.zone_id != zid and (want_on.get(z.zone_id) or z.is_on)
            ]
            lo_b, hi_b = bands[zid]
            ride_mode = mode if mode in (MODE_HEAT, MODE_COOL) else (pmode or "")
            mixing_covered = (
                bool(covering_sibs)
                and ride_mode in (MODE_HEAT, MODE_COOL)
                and _mixing_free_rider(zone, ride_mode, state, covering_sibs, lo_b, hi_b, now)
            )
            dwell_ok = now - parked_since >= PARK_MIN_DWELL_S
            probing = zone.park_residuals is None
            probe_done = probing and now - parked_since >= PARK_PROBE_S
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
            # A fan-type live session still blows air across the coil; inside
            # the coil-dry window that re-evaporates condensate exactly like
            # fan_assist mixing would, so it is released to true off (best)
            # rather than held as a park. Residual sessions are conditioning
            # and are exempt — there is no coil-dry subsystem specific to park.
            last_cool = state.zone_last_cool.get(zid)
            coil_wet_fan_type = (
                zone.park_current_is_fan_type is True
                and last_cool is not None
                and now - last_cool < COIL_DRY_S
            )

            # Continuum depth escalation / under-cond step-down.
            preferred = state.zone_park_preferred.get(zid, PARK_MARGIN_K)
            depth = state.zone_head_depth_k.get(zid, state.zone_park_margin.get(zid, preferred))
            depth_changed = False
            depth_exhausted = False
            if pmode in (MODE_HEAT, MODE_COOL) and zid in pack_ids:
                walked = _helper_walk_depth(state, zone, lo_b, hi_b, pmode, now)
                depth_changed = abs(walked - depth) > 1e-9
                cap = _compressing_hold_k(zone)
                depth = _sync_depth_views(
                    state, zid, walked, now, lo=0.0, hi=max(cap, 0.0), zone=zone
                )
            elif pmode in (MODE_HEAT, MODE_COOL):
                walk = overshoot_walk.get(zid, False)
                depth, depth_changed, depth_exhausted = _adapt_head_depth_k(
                    state,
                    zid,
                    zone,
                    pmode,
                    depth,
                    want_conditioning=False,
                    mixing_covered=mixing_covered,
                    overshoot_walk=walk,
                )
                floor = _chase_floor_k(state, zid)
                ceiling = _overshoot_ceiling_k(zone, state, depth, walk=walk)
                depth = _sync_depth_views(state, zid, depth, now, lo=floor, hi=ceiling, zone=zone)

            walk = overshoot_walk.get(zid, False)
            hold_ceiling = (
                _overshoot_ceiling_k(zone, state, depth, walk=walk)
                if pmode in (MODE_HEAT, MODE_COOL)
                else PARK_MARGIN_MAX_K
            )
            if (overcorrected or depth_exhausted) and not off_allowed:
                # Head cannot hold temperature, but min-runtime forbids off.
                # Unintentional overshoot already stepped +0.5 this tick —
                # keep that depth (do not slam to DEPTH_MAX).
                rewrote = False
                if not (walk and not depth_exhausted):
                    pre = depth
                    depth = _sync_depth_views(
                        state,
                        zid,
                        hold_ceiling,
                        now,
                        lo=_chase_floor_k(state, zid),
                        hi=hold_ceiling,
                        zone=zone,
                    )
                    rewrote = abs(depth - pre) > 1e-9
                diag.setdefault("park_overcorrected_held", []).append(zid)
                if pmode is not None and _depth_command_due(
                    state,
                    zid,
                    now,
                    zone,
                    depth,
                    pmode,
                    force=rewrote or depth_changed,
                ):
                    commands.append(_command_with_depth(zid, pmode, None, "park", depth))
                    state.zone_last_cmd[zid] = now
                continue
            if overcorrected or depth_exhausted:
                # Head cannot hold temperature at any useful depth: idle it.
                _clear_park_session(state, zid, zone, now)
                diag.setdefault("park_overcorrected", []).append(zid)
                desired_on = False
            elif coil_wet_fan_type and off_allowed:
                # Fan-type + wet coil: step depth down (toward chase / off),
                # not a hard soft-release — re-evaporation risk falls as depth
                # leaves positive hysteresis; off only once depth is exhausted.
                if depth > 0.0:
                    depth = _sync_depth_views(
                        state,
                        zid,
                        depth - DEPTH_STEP_K,
                        now,
                        lo=_chase_floor_k(state, zid),
                        hi=hold_ceiling,
                        zone=zone,
                    )
                    diag.setdefault("park_coil_wet_step", []).append(zid)
                    if pmode is not None and _depth_command_due(
                        state, zid, now, zone, depth, pmode, force=True
                    ):
                        commands.append(_command_with_depth(zid, pmode, None, "park", depth))
                        state.zone_last_cmd[zid] = now
                    if depth > 0.0:
                        continue
                _clear_park_session(state, zid, zone, now)
                diag.setdefault("park_coil_wet_released", []).append(zid)
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
                # Load beat the parked output: fall through to normal demand.
                _clear_park_session(state, zid, zone, now)
            elif zid in state.shed or (
                dwell_ok
                and off_allowed
                and not plant_min_on
                and (
                    (
                        mixing_covered
                        and (zid not in pack_ids or any(sid in demand_ids for sid in covering_sibs))
                    )
                    or (
                        zid not in pack_ids
                        and not _anchor_active(state, s, mode, zid, pack_ids)
                        and (
                            probe_done
                            or (
                                # Known idler with no useful shallower depth → off.
                                zone.park_residuals is False
                                and not _shallower_depth_useful(
                                    zone,
                                    depth if depth > 0 else preferred,
                                    want_conditioning=False,
                                    mixing_covered=mixing_covered,
                                )
                            )
                            or (
                                # Soft-release when exploit/sibling fail — but keep
                                # stepping a residual head whose depth can still
                                # move toward the chase floor (weak cover).
                                regime != "continuous"
                                and (not _park_exploit_ok(zone, pmode) or not sibling_alive)
                                and not (
                                    zone.park_residuals is True
                                    and depth > _chase_floor_k(state, zid) + 1e-9
                                )
                            )
                            or pmode is None
                        )
                    )
                )
            ):
                # Soft-release: mixing free-ride, finished probe idler, etc.
                if zid not in state.shed and depth > 0.0:
                    _settle_park_preferred(state, zid, depth)
                _clear_park_session(state, zid, zone, now)
                desired_on = False
            else:
                # Stay parked / continuum: refresh on depth move only while
                # depth is positive (frozen hold). Chase/zero-hold still
                # follows spacing. Weak residual cover stays here so adapt
                # can step depth down rather than soft-releasing to off.
                if depth <= 0.0:
                    # Stepped into chase while want-off — idle if allowed.
                    # Pack-stay zero-hold stays loaded (walk continues below).
                    if off_allowed and not plant_min_on and zid not in pack_ids:
                        _clear_park_session(state, zid, zone, now)
                        desired_on = False
                    elif pmode is not None and _depth_command_due(
                        state, zid, now, zone, depth, pmode, force=depth_changed
                    ):
                        commands.append(_command_with_depth(zid, pmode, None, "park", depth))
                        state.zone_last_cmd[zid] = now
                        continue
                elif pmode is not None and _depth_command_due(
                    state, zid, now, zone, depth, pmode, force=depth_changed
                ):
                    commands.append(_command_with_depth(zid, pmode, None, "park", depth))
                    state.zone_last_cmd[zid] = now
                    continue
                if depth > 0.0 or zid in state.zone_parked_since:
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
                or plant_min_on
                or _anchor_active(state, s, mode, zid, pack_ids)
                or any(want_on.get(z.zone_id) and z.zone_id != zid for z in zones)
                or zid in pack_ids
            )
            and _transition_allowed(state, zid, now, False, s)
        ):
            probe_due = (
                zone.park_residuals is None
                and now - state.zone_last_park_probe.get(zid, 0.0) >= PARK_PROBE_SPACING_S
                and now - state.zone_last_park_abort.get(zid, 0.0) >= PARK_PROBE_RETRY_S
            )
            # Plant min_on / anchor: residual park serves the compressor run
            # even before the head is classified (observations are free —
            # do not charge the probe budget). Exploit or spaced probe otherwise.
            plant_hold = plant_min_on or _anchor_active(state, s, mode, zid, pack_ids)
            exploit = zone.park_residuals is True and _park_exploit_ok(zone, mode)
            if plant_hold or probe_due or exploit or zid in pack_ids:
                preferred = _park_preferred(zone, state)
                lo_p, hi_p = bands[zid]
                if zid in pack_ids:
                    # Continue the helper walk; never jump to residual-edge.
                    entry = _helper_walk_depth(state, zone, lo_p, hi_p, mode, now)
                else:
                    entry = _park_entry_margin(zone, state)
                state.zone_park_preferred[zid] = preferred
                _sync_depth_views(state, zid, entry, now, zone=zone)
                state.zone_parked_since[zid] = now
                if zone.temp is not None:
                    state.zone_park_ref[zid] = zone.temp
                if probe_due and not plant_hold:
                    state.zone_park_probe_entry[zid] = zone.park_samples
                commands.append(_command_with_depth(zid, mode, None, "park", entry))
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
                    # still out of band (unfinished pull-down). Pack members
                    # continue the helper walk instead of jumping to residual-edge.
                    lo_p, hi_p = bands[zid]
                    if zid in pack_ids:
                        entry = _helper_walk_depth(state, zone, lo_p, hi_p, zone.head_mode, now)
                    else:
                        entry = _park_entry_margin(zone, state)
                    _sync_depth_views(state, zid, entry, now, zone=zone)
                    state.zone_parked_since[zid] = now
                    if zone.temp is not None:
                        state.zone_park_ref[zid] = zone.temp
                    commands.append(_command_with_depth(zid, zone.head_mode, None, "park", entry))
                    state.zone_last_cmd[zid] = now
                    diag.setdefault("parked", []).append(zid)
                    continue
                desired_on = currently_on  # guard blocks the change this tick

        walk = overshoot_walk.get(zid, False)
        if (
            walk
            and desired_on
            and currently_on
            and mode in (MODE_HEAT, MODE_COOL)
            and zid not in helper_ids
        ):
            # Min-on / leftover demand while already past the unwidened far
            # edge: raise hysteresis depth instead of tracking-chase.
            preferred = state.zone_park_preferred.get(zid, PARK_MARGIN_K)
            depth = state.zone_head_depth_k.get(zid, state.zone_park_margin.get(zid, preferred))
            depth, depth_changed, should_off = _adapt_head_depth_k(
                state,
                zid,
                zone,
                mode,
                depth,
                want_conditioning=True,
                overshoot_walk=True,
            )
            floor = _chase_floor_k(state, zid)
            ceiling = _overshoot_ceiling_k(zone, state, depth, walk=True)
            depth = _sync_depth_views(state, zid, depth, now, lo=floor, hi=ceiling, zone=zone)
            if not should_off:
                if _depth_command_due(state, zid, now, zone, depth, mode, force=depth_changed):
                    commands.append(_command_with_depth(zid, mode, None, "depth_residual", depth))
                    state.zone_last_cmd[zid] = now
                state.zone_fan[zid] = False
                continue

        if desired_on and mode not in (MODE_HEAT, MODE_COOL):
            # Zone is running out its minimum runtime while the house has
            # gone idle. Do not leave the head at a stale (possibly deep)
            # tracked setpoint: ease it to minimum tracking depth in its own
            # physical direction so it modulates gently instead of cooling hard into
            # a room nobody asked to condition further.
            if s.tracking and zone.is_on and zone.head_mode in (MODE_HEAT, MODE_COOL):
                state.zone_track_delta[zid] = TRACK_DELTA_MIN_K
                if _depth_command_due(state, zid, now, zone, -TRACK_DELTA_MIN_K, zone.head_mode):
                    commands.append(
                        _command_with_depth(zid, zone.head_mode, None, "runout", -TRACK_DELTA_MIN_K)
                    )
                    state.zone_last_cmd[zid] = now
            continue

        if desired_on:
            center = centers[zid]
            lo, hi = bands[zid]
            condition_hold = False
            quiet_cover = diag.get("quiet_cover_extended") == zid
            helper_depth_k: float | None = None
            helper_prev_d: float | None = None
            if zid in demand_ids:
                condition_hold = _quiet_night_condition_hold(s, zid, snap.local_hour, mode)
                if (
                    not condition_hold
                    and preset == PRESET_NONE
                    and s.coordination
                    and mode in (MODE_HEAT, MODE_COOL)
                ):
                    # Coordinated None: hold the reactive edge (cool: warm
                    # return air / higher Tevap; heat: cold return air).
                    setpoint = comfort.hold_edge_setpoint(mode, lo, hi, conditioning=False)
                else:
                    setpoint = comfort.demand_setpoint(
                        mode,
                        center,
                        lo,
                        hi,
                        zone.occupied,
                        preset,
                        s,
                        condition_hold=condition_hold,
                    )
                reason = "demand"
            elif quiet_cover:
                ext_lo, ext_hi = _extended_cover_band(s, zone, lo, hi, mode)
                setpoint = comfort.hold_edge_setpoint(mode, ext_lo, ext_hi, conditioning=True)
                reason = "helper"
                condition_hold = True
                lo, hi = ext_lo, ext_hi
            else:
                # Hold at live room temp (depth 0), then SP follows 0.5 K
                # park steps: 21 → 21.5 → 22 … while still compressing.
                reason = "helper"
                helper_prev_d = state.zone_head_depth_k.get(zid, 0.0)
                helper_depth_k = _helper_walk_depth(state, zone, lo, hi, mode, now)
                prev_d = helper_prev_d
                hold = quantize_setpoint(zone.temp) if zone.temp is not None else center
                hold = min(max(hold, lo), hi)
                last_sp = state.zone_last_setpoint.get(zid)
                if last_sp is None:
                    setpoint = hold
                elif mode == MODE_COOL:
                    base = float(last_sp) - prev_d
                    setpoint = min(max(base + helper_depth_k, lo), hi)
                else:
                    base = float(last_sp) + prev_d
                    setpoint = min(max(base - helper_depth_k, lo), hi)
            setpoint = quantize_setpoint(setpoint)
            pick_lo, pick_hi = lo, hi
            if condition_hold:
                # Chase until the hold edge, not residual-park in mid-band.
                if mode == MODE_COOL:
                    pick_hi = min(hi, setpoint)
                elif mode == MODE_HEAT:
                    pick_lo = max(lo, setpoint)

            # Tracking depth adaptation (room frame): deepen while the room is
            # not converging toward its target edge, relax once it is inside.
            # Continuum head depth is floored at -this chase (0.5 grid).
            track_delta = None
            depth_k: float | None = None
            if s.tracking and mode in (MODE_HEAT, MODE_COOL):
                delta = state.zone_track_delta.get(zid, TRACK_DELTA_DEFAULT_K)
                if zone.temp is not None:
                    past_target = (
                        zone.temp <= setpoint if mode == MODE_COOL else zone.temp >= setpoint
                    )
                    out_of_band = zone.temp > pick_hi if mode == MODE_COOL else zone.temp < pick_lo
                    n_on = sum(
                        z.n_rooms
                        for z in zones
                        if want_on.get(z.zone_id) and z.zone_id not in state.shed
                    )
                    chase_cap = _depth_chase_cap_k(snap, n_on, not out_of_band)
                    if out_of_band:
                        delta += TRACK_DELTA_STEP_K
                    elif past_target:
                        delta -= TRACK_DELTA_STEP_K
                    delta = min(delta, chase_cap)
                delta = min(max(delta, TRACK_DELTA_MIN_K), TRACK_DELTA_MAX_K)
                state.zone_track_delta[zid] = delta
                track_delta = delta
                # Continuum depth from prior adapt / residual pick.
                walk = overshoot_walk.get(zid, False)
                leftover = state.zone_head_depth_k.get(zid)
                if reason == "helper" and not condition_hold and helper_depth_k is not None:
                    cap = _compressing_hold_k(zone)
                    floor = 0.0
                    ceiling = min(_overshoot_ceiling_k(zone, state, helper_depth_k, walk=walk), cap)
                    depth_k = _sync_depth_views(
                        state,
                        zid,
                        helper_depth_k,
                        now,
                        lo=floor,
                        hi=max(ceiling, 0.0),
                        zone=zone,
                    )
                elif (
                    leftover is not None and leftover <= 0.0 and zid not in state.zone_parked_since
                ):
                    # Already stepped into 0/chase via continuum; clamp to live
                    # chase floor after track adapt above. Leftover positive
                    # depth without a session must not skip pick_depth_k
                    # (in-band resurrection is the same B1 class as a hot room).
                    floor = _chase_floor_k(state, zid)
                    held = leftover
                    ceiling = _overshoot_ceiling_k(zone, state, held, walk=walk)
                    depth_k = _sync_depth_views(
                        state,
                        zid,
                        held,
                        now,
                        lo=floor,
                        hi=ceiling,
                        zone=zone,
                    )
                    if reason == "demand":
                        reason = "depth_chase" if depth_k < 0.0 else "depth_hold"
                else:
                    depth_k, depth_tag = pick_depth_k(
                        mode=mode,
                        temp=zone.temp,
                        lo=pick_lo,
                        hi=pick_hi,
                        standing_load_w=zone.standing_load_w,
                        n_rooms=zone.n_rooms,
                        park_residuals=zone.park_residuals,
                        margin_bins=zone.park_margin_bins,
                        track_delta=delta,
                        park_learning=s.park_learning,
                    )
                    if depth_k > 0.0:
                        if reason == "demand":
                            reason = depth_tag
                        floor = _chase_floor_k(state, zid)
                        ceiling = _overshoot_ceiling_k(zone, state, depth_k, walk=walk)
                        depth_k = _sync_depth_views(
                            state, zid, depth_k, now, lo=floor, hi=ceiling, zone=zone
                        )
                        state.zone_park_preferred.setdefault(zid, _park_preferred(zone, state))
                    else:
                        floor = _chase_floor_k(state, zid)
                        ceiling = _overshoot_ceiling_k(zone, state, depth_k, walk=walk)
                        depth_k = _sync_depth_views(
                            state, zid, depth_k, now, lo=floor, hi=ceiling, zone=zone
                        )
                        if reason == "demand":
                            reason = depth_tag

            last_sp = state.zone_last_setpoint.get(zid)
            setpoint_changed = last_sp is None or abs(setpoint - last_sp) >= SETPOINT_EPSILON_K
            # Re-anchor on spacing (chase only), room-setpoint change, helper
            # depth step, or head-internal drift vs the live device SP.
            depth_for_anchor = depth_k
            if depth_for_anchor is None and track_delta is not None:
                depth_for_anchor = -float(track_delta)
            has_depth = depth_for_anchor is not None
            spacing_ok = now - state.zone_last_cmd.get(zid, 0.0) >= COMMAND_SPACING_S
            depth_stepped = (
                helper_depth_k is not None
                and helper_prev_d is not None
                and abs(helper_depth_k - helper_prev_d) > 1e-9
            )
            depth_due = (
                has_depth
                and zone.is_on
                and _depth_command_due(
                    state, zid, now, zone, depth_for_anchor, mode, force=depth_stepped
                )
            )
            if transitioned or not zone.is_on or (setpoint_changed and spacing_ok) or depth_due:
                if depth_k is not None:
                    commands.append(_command_with_depth(zid, mode, setpoint, reason, depth_k))
                else:
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
