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
    ControllerState,
    HouseSnapshot,
    Settings,
    ZoneSnapshot,
)

RUNNING_MEAN_TAU_H = 7.0 * 24.0
ADAPTIVE_MIN_C = 20.0
ADAPTIVE_MAX_C = 26.0

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

MODE_DEADBAND_KH = 1.5
MODE_DWELL_S = 6.0 * 3600.0
OVERRIDE_DELTA_K = 2.0
OVERRIDE_SUSTAIN_S = 30.0 * 60.0
MIN_MODEL_CONFIDENCE = 0.3
UNOCCUPIED_WIDEN_K = 1.5
UNOCCUPIED_WEIGHT = 0.3


def update_running_mean(t_rm: float | None, t_out: float, dt_h: float) -> float:
    """Exponential running mean of outdoor temperature (7-day time constant)."""
    if t_rm is None:
        return t_out
    return t_rm + (dt_h / RUNNING_MEAN_TAU_H) * (t_out - t_rm)


def adaptive_target(t_rm: float) -> float:
    """EN 16798 style adaptive comfort temperature, clamped to sane bounds."""
    return min(max(0.33 * t_rm + 18.8, ADAPTIVE_MIN_C), ADAPTIVE_MAX_C)


def band_center(settings: Settings, t_rm: float | None, zone_offset: float = 0.0) -> float:
    center = settings.target
    if t_rm is not None and settings.adaptive_blend > 0:
        w = min(max(settings.adaptive_blend, 0.0), 1.0)
        center = (1.0 - w) * settings.target + w * adaptive_target(t_rm)
    return center + zone_offset


def zone_band(
    settings: Settings,
    center: float,
    zone_occupied: bool | None,
    house_occupied: bool | None,
) -> tuple[float, float]:
    """Comfort band (lower, upper) for a zone this tick."""
    half = settings.band_k
    preset = settings.preset
    if settings.presence_adaptation and house_occupied is False:
        preset = PRESET_AWAY
    if preset == PRESET_ECO:
        half += 1.5
    elif preset == PRESET_AWAY:
        half += 3.0
    elif preset == PRESET_BOOST:
        half = max(0.3, half * 0.5)
    if zone_occupied is False:
        half += UNOCCUPIED_WIDEN_K
    return center - half, center + half


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


def demand_integrals(
    zones: list[ZoneSnapshot],
    bands: dict[str, tuple[float, float]],
) -> tuple[float, float]:
    """Occupancy-weighted (warm excess, cold deficit) in K*h over the horizon,
    integrated from each zone's hourly free-float trajectory."""
    warm = 0.0
    cold = 0.0
    for zone in zones:
        if not zone.free_float or zone.zone_id not in bands:
            continue
        lo, hi = bands[zone.zone_id]
        weight = UNOCCUPIED_WEIGHT if zone.occupied is False else 1.0
        weight *= zone.n_rooms  # a mirrored zone is N rooms' worth of demand
        for temp in zone.free_float:
            warm += weight * max(0.0, temp - hi)  # 1 h per sample
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
        if dominant == MODE_COOL and zone.temp < lo - OVERRIDE_DELTA_K:
            opposite = MODE_HEAT
        elif dominant == MODE_HEAT and zone.temp > hi + OVERRIDE_DELTA_K:
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
                if state.override_mode == MODE_HEAT and zone.temp < lo:
                    still_needed = True
                if state.override_mode == MODE_COOL and zone.temp > hi:
                    still_needed = True
            if not still_needed:
                state.override_mode = None
        return state.override_mode
    if state.override_since == 0.0:
        state.override_since = now_ts
    if now_ts - state.override_since >= OVERRIDE_SUSTAIN_S:
        state.override_mode = opposite
    return state.override_mode


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

    confident = [z for z in zones if z.free_float and z.confidence >= MIN_MODEL_CONFIDENCE]
    if not confident:
        dev = indoor_deviation(zones, bands, snap.aux_indoor)
        mode = fallback_mode(snap.t_rm, state.mode, dev, snap.settings.band_k)
        if mode != state.mode:
            state.mode = mode
            state.mode_since = now
        return ModeDecision(mode, "fallback")

    warm, cold = demand_integrals(confident, bands)
    if warm - cold > MODE_DEADBAND_KH:
        desired = MODE_COOL
    elif cold - warm > MODE_DEADBAND_KH:
        desired = MODE_HEAT
    else:
        desired = MODE_OFF

    if desired != state.mode:
        # Switching between heat and cool (or leaving off) honours the dwell;
        # dropping to off is always allowed.
        if desired != MODE_OFF and now - state.mode_since < MODE_DWELL_S and state.mode != MODE_OFF:
            return ModeDecision(state.mode, "dwell", warm, cold)
        state.mode = desired
        state.mode_since = now
    return ModeDecision(state.mode, "model", warm, cold)
