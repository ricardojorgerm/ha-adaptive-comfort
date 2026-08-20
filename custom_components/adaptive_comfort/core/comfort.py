"""Comfort targets, presets, presence and dominant-mode arbitration."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .types import (
    MODE_COOL,
    MODE_HEAT,
    MODE_OFF,
    PRESET_AWAY,
    PRESET_BOOST,
    PRESET_ECO,
    PRESET_MANUAL,
    PRESET_NONE,
    ControllerState,
    HouseSnapshot,
    Settings,
    ZoneSnapshot,
)

RUNNING_MEAN_TAU_H = 7.0 * 24.0
# House envelope rails at default target 22.5: +/- (band_k + Away +3 K) -> 18.8-26.2.
ADAPTIVE_MIN_C = 18.8
ADAPTIVE_MAX_C = 26.2
ECO_EXTRA_K = 1.5
AWAY_EXTRA_K = 3.0

# Lisbon climatological normals used as the outdoor fallback when neither an
# outdoor sensor nor a weather entity is available (fresh installs stay sane).
LISBON_MONTHLY_MEAN_C = (11.6, 12.7, 14.9, 15.9, 18.0, 21.2, 23.1, 23.5, 22.1, 18.8, 15.0, 12.4)
LISBON_DIURNAL_AMPLITUDE_K = (3.5, 3.5, 4.0, 4.0, 4.5, 5.0, 5.5, 5.5, 5.0, 4.0, 3.5, 3.5)


def climatology_mean(month: int) -> float:
    """Monthly mean outdoor temperature for the default (Lisbon) climate."""
    return LISBON_MONTHLY_MEAN_C[(month - 1) % 12]


def climatology_temp(month: int, local_hour: float) -> float:
    """Climatological outdoor temperature estimate (mean + diurnal swing)."""
    amplitude = LISBON_DIURNAL_AMPLITUDE_K[(month - 1) % 12]
    return climatology_mean(month) + amplitude * math.cos(
        2.0 * math.pi * (local_hour - 15.0) / 24.0
    )


# Cold-start fallback thresholds on the running-mean outdoor temperature.
FALLBACK_HEAT_BELOW_C = 14.0
FALLBACK_COOL_ABOVE_C = 21.0
FALLBACK_HYSTERESIS_K = 1.0

# Enter/exit thresholds for the MODE_HORIZON_H integral (K*h). Sized so a
# sustained ~0.4 K / ~0.15 K mean excursion enters / holds (8 * 0.375 / 0.125).
MODE_DEADBAND_KH = 3.0
MODE_EXIT_KH = 1.0
MODE_HORIZON_H = 8  # near-term hours for mode arbitration (not full 24 h)
MODE_DWELL_S = 6.0 * 3600.0
OVERRIDE_DELTA_K = 2.0
OVERRIDE_SUSTAIN_S = 30.0 * 60.0
MIN_MODEL_CONFIDENCE = 0.3
UNOCCUPIED_WIDEN_K = 1.5
UNOCCUPIED_WEIGHT = 0.3
# Away/vacant demand trims to the near band edge, not the comfort center.
BAND_HOLD_MARGIN_K = 0.3


def effective_zone_occupied(settings: Settings, occupied: bool | None) -> bool | None:
    """Occupancy for vacant control paths.

    When ``zone_presence_adaptation`` is off, returns ``None`` (unknown) so
    vacant widen, band-hold, helper/fan skips, and integral down-weights do
    not fire — while zone presence sensors still update for diagnostics.
    """
    if not settings.zone_presence_adaptation:
        return None
    return occupied


def update_running_mean(t_rm: float | None, t_out: float, dt_h: float) -> float:
    """Exponential running mean of outdoor temperature (7-day time constant)."""
    if t_rm is None:
        return t_out
    return t_rm + (dt_h / RUNNING_MEAN_TAU_H) * (t_out - t_rm)


def adaptive_target(t_rm: float) -> float:
    """EN 16798 style adaptive comfort temperature, clamped to sane bounds."""
    return min(max(0.33 * t_rm + 18.8, ADAPTIVE_MIN_C), ADAPTIVE_MAX_C)


def house_envelope(settings: Settings) -> tuple[float, float]:
    """Absolute indoor rails: ``target ± (band_k + Away extra)``.

    Default 22.5 +/- 3.7 -> 18.8-26.2 C. Seasonal center may move inside;
    Eco/Away widens cannot command outside.
    """
    half = settings.band_k + AWAY_EXTRA_K
    return settings.target - half, settings.target + half


def clamp_to_envelope(lo: float, hi: float, settings: Settings) -> tuple[float, float]:
    elo, ehi = house_envelope(settings)
    lo_c = min(max(lo, elo), ehi)
    hi_c = min(max(hi, elo), ehi)
    if lo_c > hi_c:
        lo_c, hi_c = elo, ehi
    return lo_c, hi_c


def band_center(settings: Settings, t_rm: float | None, zone_offset: float = 0.0) -> float:
    center = settings.target
    if t_rm is not None and settings.adaptive_blend > 0:
        w = min(max(settings.adaptive_blend, 0.0), 1.0)
        center = (1.0 - w) * settings.target + w * adaptive_target(t_rm)
    center = center + zone_offset
    elo, ehi = house_envelope(settings)
    return min(max(center, elo), ehi)


def effective_preset(settings: Settings, house_occupied: bool | None) -> str:
    """User preset with presence Away, unless Boost (or Manual) wins."""
    preset = settings.preset
    if preset in (PRESET_BOOST, PRESET_MANUAL):
        return preset
    if settings.presence_adaptation and house_occupied is False:
        return PRESET_AWAY
    return preset if preset else PRESET_NONE


def zone_band(
    settings: Settings,
    center: float,
    zone_occupied: bool | None,
    house_occupied: bool | None,
    extra_half_k: float = 0.0,
    extra_lo_k: float | None = None,
    extra_hi_k: float | None = None,
) -> tuple[float, float]:
    """Comfort band (lower, upper) for a zone this tick.

    COP-efficiency slack (center fixed) via ``extra_lo_k`` / ``extra_hi_k``.
    If those are omitted, ``extra_half_k`` widens both sides symmetrically.
    Boost callers typically stretch only the conditioning side (cool: lo;
    heat: hi) so the far reactive edge stays tight.
    """
    half = settings.band_k
    preset = effective_preset(settings, house_occupied)
    if preset == PRESET_ECO:
        half += ECO_EXTRA_K
    elif preset == PRESET_AWAY:
        half += AWAY_EXTRA_K
    elif preset == PRESET_BOOST:
        half = max(0.3, half * 0.5)
    # Manual uses the underlying band for diagnostics; it does not command.
    # Boost overrides vacant widen (same as it overrides presence-Away).
    # Away already prices in empty-house slack — do not stack vacant +1.5 K.
    if effective_zone_occupied(settings, zone_occupied) is False and preset not in (
        PRESET_BOOST,
        PRESET_AWAY,
    ):
        half += UNOCCUPIED_WIDEN_K
    if extra_lo_k is None and extra_hi_k is None:
        extra_lo_k = extra_half_k
        extra_hi_k = extra_half_k
    else:
        extra_lo_k = 0.0 if extra_lo_k is None else extra_lo_k
        extra_hi_k = 0.0 if extra_hi_k is None else extra_hi_k
    lo = center - half - max(0.0, extra_lo_k)
    hi = center + half + max(0.0, extra_hi_k)
    return clamp_to_envelope(lo, hi, settings)


def hold_edge_setpoint(mode: str, lo: float, hi: float, *, conditioning: bool) -> float:
    """``BAND_HOLD_MARGIN_K`` inset from a band edge.

    ``conditioning=True``: cool toward ``lo``, heat toward ``hi`` (bank a
    quiet-night room before sleep). ``False``: the Away/vacant drift edge.
    """
    if mode == MODE_COOL:
        return (lo + BAND_HOLD_MARGIN_K) if conditioning else (hi - BAND_HOLD_MARGIN_K)
    if mode == MODE_HEAT:
        return (hi - BAND_HOLD_MARGIN_K) if conditioning else (lo + BAND_HOLD_MARGIN_K)
    return (lo + hi) / 2.0


def demand_setpoint(
    mode: str,
    center: float,
    lo: float,
    hi: float,
    zone_occupied: bool | None,
    preset: str,
    settings: Settings | None = None,
    *,
    condition_hold: bool = False,
) -> float:
    """Room-frame demand target: center-seek, or band-hold when Away/vacant.

    ``condition_hold`` banks the conditioning edge (cool ``lo``, heat ``hi``).
    Boost always center-seeks (overrides Away, vacant, and condition-hold).
    """
    if preset == PRESET_BOOST:
        return center
    if condition_hold:
        return hold_edge_setpoint(mode, lo, hi, conditioning=True)
    occ = (
        effective_zone_occupied(settings, zone_occupied) if settings is not None else zone_occupied
    )
    band_hold = preset == PRESET_AWAY or occ is False
    if not band_hold:
        return center
    return hold_edge_setpoint(mode, lo, hi, conditioning=False)


def indoor_deviation(
    zones: list[ZoneSnapshot],
    bands: dict[str, tuple[float, float]],
    aux_indoor: tuple[tuple[float, float], ...] = (),
) -> float | None:
    """Weighted mean deviation of indoor temperatures from band centers [K].

    Zone sensors weigh 1.0 (x rooms); auxiliary sensors in unconditioned
    rooms carry their own (smaller) weight and use the mean band center,
    giving a fresh install extra indoor evidence (e.g. a bathroom sensor).
    """
    total = 0.0
    weight_sum = 0.0
    for zone in zones:
        if zone.temp is None or zone.zone_id not in bands:
            continue
        lo, hi = bands[zone.zone_id]
        w = float(zone.n_rooms)
        total += w * (zone.temp - (lo + hi) / 2.0)
        weight_sum += w
    if bands and aux_indoor:
        centers = [(lo + hi) / 2.0 for lo, hi in bands.values()]
        mean_center = sum(centers) / len(centers)
        for temp, w in aux_indoor:
            total += w * (temp - mean_center)
            weight_sum += w
    if weight_sum == 0.0:
        return None
    return total / weight_sum


def fallback_mode(
    t_rm: float | None,
    current: str,
    indoor_dev: float | None = None,
    band_k: float = 0.7,
) -> str:
    """Cold-start rule: live indoor evidence bounded by seasonal sanity.

    With no fitted model yet, act like a sensible band thermostat: condition
    toward the target when indoor readings leave the comfort band, but do not
    fight the season (no cooling in a cold spell, no heating in a heat wave)
    unless the indoor error is large. Falls back to pure outdoor hysteresis
    when no indoor readings exist; t_rm may itself be a climatology estimate.
    """
    season_heat = t_rm is not None and t_rm < FALLBACK_HEAT_BELOW_C
    season_cool = t_rm is not None and t_rm > FALLBACK_COOL_ABOVE_C

    if indoor_dev is not None:
        # Classic thermostat hysteresis in the indoor domain: entering a mode
        # needs the band edge crossed; leaving it needs the house pushed a
        # full band past center in the opposite direction. A latched mode
        # with no demand is harmless (zone band logic keeps heads off), so
        # the wide hysteresis only stabilises direction, not runtime.
        enter = band_k
        exit_ = band_k
        if current == MODE_COOL and indoor_dev > -exit_:
            return MODE_COOL
        if current == MODE_HEAT and indoor_dev < exit_:
            return MODE_HEAT
        if indoor_dev > enter and (not season_heat or indoor_dev > 2.0):
            return MODE_COOL
        if indoor_dev < -enter and (not season_cool or indoor_dev < -2.0):
            return MODE_HEAT
        return MODE_OFF

    if t_rm is None:
        return MODE_OFF
    if current == MODE_HEAT:
        return MODE_HEAT if t_rm < FALLBACK_HEAT_BELOW_C + FALLBACK_HYSTERESIS_K else MODE_OFF
    if current == MODE_COOL:
        return MODE_COOL if t_rm > FALLBACK_COOL_ABOVE_C - FALLBACK_HYSTERESIS_K else MODE_OFF
    if season_heat:
        return MODE_HEAT
    if season_cool:
        return MODE_COOL
    return MODE_OFF


@dataclass
class ModeDecision:
    mode: str
    source: str  # "forced" | "model" | "fallback" | "override" | "dwell"
    warm_excess_kh: float = 0.0
    cold_deficit_kh: float = 0.0


def _mode_trajectory(zone: ZoneSnapshot, horizon_h: int = MODE_HORIZON_H) -> tuple[float, ...]:
    """Near-term no-AC temperatures for mode arbitration.

    Confident free-float is the counterfactual (including while the head is on:
    predict_free never models AC). Low-confidence zones contribute a flat hold
    at the current temperature so an always-conditioned room is not dropped.
    """
    if zone.confidence >= MIN_MODEL_CONFIDENCE and zone.free_float:
        return tuple(zone.free_float[:horizon_h])
    if zone.temp is None:
        return ()
    return (zone.temp,) * horizon_h


def demand_integrals(
    zones: list[ZoneSnapshot],
    bands: dict[str, tuple[float, float]],
    horizon_h: int = MODE_HORIZON_H,
    settings: Settings | None = None,
) -> tuple[float, float]:
    """Occupancy-weighted (warm excess, cold deficit) in K*h over the horizon.

    Integrates each zone's no-AC counterfactual vs its comfort band. Conditioning
    zones are included: predict_free is already "if this zone gets no AC."
    A cooling head does not add cold (it manufactured the undershoot); a heating
    head does not add warm.
    """
    warm = 0.0
    cold = 0.0
    for zone in zones:
        if zone.zone_id not in bands:
            continue
        trajectory = _mode_trajectory(zone, horizon_h)
        if not trajectory:
            continue
        lo, hi = bands[zone.zone_id]
        occ = (
            effective_zone_occupied(settings, zone.occupied)
            if settings is not None
            else zone.occupied
        )
        weight = UNOCCUPIED_WEIGHT if occ is False else 1.0
        weight *= zone.n_rooms  # a mirrored zone is N rooms' worth of demand
        # Active cooling made the room cold; that is not a heat request.
        # Active heating made it hot; that is not a cool request. Manual
        # pulldown (SP 18 overnight) must not elect heat at 25 °C outdoor.
        skip_warm = zone.head_mode == MODE_HEAT
        skip_cold = zone.head_mode == MODE_COOL
        for temp in trajectory:
            if not skip_warm:
                warm += weight * max(0.0, temp - hi)  # 1 h per sample
            if not skip_cold:
                cold += weight * max(0.0, lo - temp)
    return warm, cold


def _emergency_override(
    zones: list[ZoneSnapshot],
    bands: dict[str, tuple[float, float]],
    dominant: str,
    state: ControllerState,
    now_ts: float,
) -> str | None:
    """Temporary opposite mode when an occupied zone is far out of band
    against the dominant mode for a sustained period."""
    opposite = None
    for zone in zones:
        if zone.occupied is False or zone.temp is None or zone.zone_id not in bands:
            continue
        lo, hi = bands[zone.zone_id]
        if (
            dominant == MODE_COOL
            and zone.temp < lo - OVERRIDE_DELTA_K
            and zone.head_mode != MODE_COOL
        ):
            opposite = MODE_HEAT
        elif (
            dominant == MODE_HEAT
            and zone.temp > hi + OVERRIDE_DELTA_K
            and zone.head_mode != MODE_HEAT
        ):
            opposite = MODE_COOL
    if opposite is None:
        state.override_since = 0.0
        if state.override_mode is not None:
            # Keep the override active until the triggering zone is back in
            # band; checked below via the in-band scan.
            still_needed = False
            for zone in zones:
                if zone.temp is None or zone.zone_id not in bands:
                    continue
                lo, hi = bands[zone.zone_id]
                if (
                    state.override_mode == MODE_HEAT
                    and zone.temp < lo
                    and zone.head_mode != MODE_COOL
                ):
                    still_needed = True
                if (
                    state.override_mode == MODE_COOL
                    and zone.temp > hi
                    and zone.head_mode != MODE_HEAT
                ):
                    still_needed = True
            if not still_needed:
                state.override_mode = None
        return state.override_mode
    if state.override_since == 0.0:
        state.override_since = now_ts
    if now_ts - state.override_since >= OVERRIDE_SUSTAIN_S:
        state.override_mode = opposite
    return state.override_mode


def _max_present_excess(
    zones: list[ZoneSnapshot],
    bands: dict[str, tuple[float, float]],
    *,
    cool: bool,
) -> float | None:
    """Largest current out-of-band excursion (K) on the cool or heat side.

    Skip heads that manufactured the excursion (cooling → not a heat need).
    """
    best: float | None = None
    for zone in zones:
        if zone.temp is None or zone.zone_id not in bands:
            continue
        if cool and zone.head_mode == MODE_HEAT:
            continue
        if not cool and zone.head_mode == MODE_COOL:
            continue
        lo, hi = bands[zone.zone_id]
        excess = zone.temp - hi if cool else lo - zone.temp
        if best is None or excess > best:
            best = excess
    return best


def dominant_mode(
    snap: HouseSnapshot,
    state: ControllerState,
    bands: dict[str, tuple[float, float]],
) -> ModeDecision:
    """Model-driven mode arbitration (used when the house climate is auto).

    Predict which way the house free-floats over the horizon and pick the
    mode that opposes the drift; hysteresis lives in the demand domain
    (K*h deadband + minimum dwell) rather than on outdoor temperature.
    """
    now = snap.now_ts
    zones = [z for z in snap.zones if z.enabled]

    override = _emergency_override(zones, bands, state.mode, state, now)
    if override is not None:
        if state.mode != override:
            state.mode = override
            state.mode_since = now
        return ModeDecision(override, "override")

    # Model integrals need at least one confident free-float. Flat temperature
    # persistence alone must not drive cool/heat — in winter a sunlit room
    # 0.4 K over the band would otherwise cross MODE_DEADBAND_KH and the
    # seasonal guard cannot demote (max_hot_k > 0 by construction).
    has_model = any(z.confidence >= MIN_MODEL_CONFIDENCE and z.free_float for z in zones)
    if not has_model:
        dev = indoor_deviation(zones, bands, snap.aux_indoor)
        mode = fallback_mode(snap.t_rm, state.mode, dev, snap.settings.band_k)
        if mode != state.mode:
            state.mode = mode
            state.mode_since = now
        return ModeDecision(mode, "fallback")

    warm, cold = demand_integrals(zones, bands, settings=snap.settings)
    season_heat = snap.t_rm is not None and snap.t_rm < FALLBACK_HEAT_BELOW_C
    season_cool = snap.t_rm is not None and snap.t_rm > FALLBACK_COOL_ABOVE_C
    # Per-zone present excursion (not house-mean): a single hot room must not
    # be silenced by cooler siblings in the seasonal guard.
    max_hot_k = _max_present_excess(zones, bands, cool=True)
    max_cold_k = _max_present_excess(zones, bands, cool=False)

    seasonally_demoted = False
    if warm - cold > MODE_DEADBAND_KH:
        desired = MODE_COOL
        # Winter: ignore forecast-only cool when no zone is above the band now.
        if season_heat and (max_hot_k is None or max_hot_k <= 0.0):
            desired = MODE_OFF
            seasonally_demoted = True
    elif cold - warm > MODE_DEADBAND_KH:
        desired = MODE_HEAT
        # Summer: ignore forecast-only heat when no zone is below the band now.
        if season_cool and (max_cold_k is None or max_cold_k <= 0.0):
            desired = MODE_OFF
            seasonally_demoted = True
    else:
        desired = MODE_OFF

    # Asymmetric exit: hold cool/heat until excess falls below MODE_EXIT_KH.
    # Do not re-promote a mode the seasonal guard just demoted (otherwise a
    # latched cool with forecast-only warm excess stays cool all winter).
    if desired == MODE_OFF and not seasonally_demoted:
        if state.mode == MODE_COOL and warm - cold > MODE_EXIT_KH:
            desired = MODE_COOL
        elif state.mode == MODE_HEAT and cold - warm > MODE_EXIT_KH:
            desired = MODE_HEAT

    if desired != state.mode:
        # Switching between heat and cool (or leaving off) honours the dwell;
        # dropping to off is allowed when exit hysteresis agrees.
        if desired != MODE_OFF and now - state.mode_since < MODE_DWELL_S and state.mode != MODE_OFF:
            return ModeDecision(state.mode, "dwell", warm, cold)
        state.mode = desired
        state.mode_since = now
    return ModeDecision(state.mode, "model", warm, cold)
