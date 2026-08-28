"""COP-timed efficiency-band widen (center fixed)."""

from __future__ import annotations

from . import comfort, power
from .regime import REGIME_VENT_MARGIN_K
from .types import (
    MODE_COOL,
    MODE_HEAT,
    PRESET_BOOST,
    ControllerState,
    HouseSnapshot,
    ZoneSnapshot,
)

COP_TIMING_LOOKAHEAD_H = 6
COP_ADVANTAGE_ENTER = 1.3
COP_ADVANTAGE_EXIT = 1.15
COP_WIDEN_MAX_K = 0.7
DEFAULT_BAND_COP_COOL = {"mild": 2.4, "warm": 1.6, "hot": 1.1}
_BAND_RANK = {"mild": 0, "warm": 1, "hot": 2}


def _band_cop(snap: HouseSnapshot, band: str, mode: str) -> float | None:
    """Learned COP for ``band`` in ``mode``; cool priors only as cool fallback.

    ``snap.cop_by_band`` is filtered to the mode at snapshot build (prior
    tick's controller mode). Use it only when that tag matches ``mode`` —
    otherwise a cool↔heat flip this tick would arbitrage on the wrong
    season's table and skip cool priors because keys exist.
    """
    if snap.cop_by_band_mode == mode and band in snap.cop_by_band:
        return snap.cop_by_band[band]
    if mode == MODE_COOL:
        return DEFAULT_BAND_COP_COOL.get(band)
    return None


def _arbitrage_ratios(
    snap: HouseSnapshot, mode: str, centers: dict[str, float]
) -> tuple[float, float]:
    """Return (advance_ratio, defer_ratio) from band COP vs forecast.

    Either ratio is 0.0 when that direction has no signal. Callers pick a
    winner for enter, but hold/exit must read the *latched* direction's
    ratio so a brief flip of which side wins cannot snap the band narrow.
    """
    if mode not in (MODE_HEAT, MODE_COOL) or not snap.forecast_hours:
        return 0.0, 0.0
    if snap.t_out is None or snap.t_out_synthetic:
        return 0.0, 0.0
    band_now = power.outdoor_band(snap.t_out)
    if band_now is None:
        return 0.0, 0.0
    # Free outdoor air: banking via widen is moot for cool.
    if mode == MODE_COOL and centers and snap.t_out <= min(centers.values()) - REGIME_VENT_MARGIN_K:
        return 0.0, 0.0

    worst = band_now
    best = band_now
    for t in snap.forecast_hours[:COP_TIMING_LOOKAHEAD_H]:
        b = power.outdoor_band(t)
        if b is None:
            continue
        if _BAND_RANK[b] > _BAND_RANK[worst]:
            worst = b
        if _BAND_RANK[b] < _BAND_RANK[best]:
            best = b

    cop_now = _band_cop(snap, band_now, mode)
    if cop_now is None or cop_now <= 0:
        return 0.0, 0.0

    advance_ratio = 0.0
    defer_ratio = 0.0
    if mode == MODE_COOL:
        # Hotter outdoor → worse cool COP → advance when worse ahead.
        if worst != band_now:
            cop_w = _band_cop(snap, worst, mode)
            if cop_w is not None and cop_w > 0:
                advance_ratio = cop_now / cop_w
        # Milder outdoor ahead → defer (wait for better COP).
        if best != band_now:
            cop_b = _band_cop(snap, best, mode)
            if cop_b is not None and cop_b > 0:
                defer_ratio = cop_b / cop_now
    else:
        # Heat: colder outdoor → worse COP. `best` is coldest band in window.
        if _BAND_RANK[best] < _BAND_RANK[band_now]:
            cop_c = _band_cop(snap, best, mode)
            if cop_c is not None and cop_c > 0:
                advance_ratio = cop_now / cop_c
        # Warmer outdoor ahead → better heat COP later → defer.
        if _BAND_RANK[worst] > _BAND_RANK[band_now]:
            cop_w = _band_cop(snap, worst, mode)
            if cop_w is not None and cop_w > 0:
                defer_ratio = cop_w / cop_now
    return advance_ratio, defer_ratio


def _forecast_cop_arbitrage(
    snap: HouseSnapshot, mode: str, centers: dict[str, float]
) -> tuple[str, float]:
    """Return ('advance'|'defer'|'none', advantage_ratio) from band COP vs forecast.

    advance: current outdoor band beats a worse band arriving within the
    lookahead (cool when hot ahead; heat when colder ahead).
    defer: a better band arrives within the lookahead.
    """
    advance_ratio, defer_ratio = _arbitrage_ratios(snap, mode, centers)
    if advance_ratio >= defer_ratio and advance_ratio > 0:
        return "advance", advance_ratio
    if defer_ratio > 0:
        return "defer", defer_ratio
    return "none", 0.0


def _outside_base_band_on_widen_side(
    zones: list[ZoneSnapshot],
    centers: dict[str, float],
    snap: HouseSnapshot,
    mode: str,
    timing: str,
) -> bool:
    """True if any zone still sits outside the unwidened band on the widen side.

    Cool advance banks below the tight lo; heat advance above the tight hi;
    defer floats the far edge. Until every zone has crossed back inside that
    base edge, withdrawing widen would reclassify the bank as the opposite
    mode's demand (summer heat after a cool bank — the field failure).
    """
    if timing not in ("advance", "defer") or mode not in (MODE_HEAT, MODE_COOL):
        return False
    s = snap.settings
    for zone in zones:
        if zone.temp is None or zone.zone_id not in centers:
            continue
        lo, hi = comfort.zone_band(s, centers[zone.zone_id], zone.occupied, snap.house_occupied)
        if mode == MODE_COOL and timing == "advance" and zone.temp < lo:
            return True
        if mode == MODE_HEAT and timing == "advance" and zone.temp > hi:
            return True
        if mode == MODE_COOL and timing == "defer" and zone.temp > hi:
            return True
        if mode == MODE_HEAT and timing == "defer" and zone.temp < lo:
            return True
    return False


def _clear_cop_widen(state: ControllerState) -> None:
    state.cop_widen_k = 0.0
    state.cop_timing = "none"
    state.cop_widen_mode = None


def _update_cop_widen(
    state: ControllerState,
    snap: HouseSnapshot,
    mode: str,
    centers: dict[str, float],
    zones: list[ZoneSnapshot],
) -> tuple[float, str]:
    """Hysteretic efficiency-band widen. Returns (widen_k, timing).

    Enter when either direction clears ENTER. Hold while the *latched*
    direction's own ratio stays ≥ EXIT — not whichever side wins this tick
    — so a brief advance/defer flip cannot snap the half-band narrow.
    Even after COP advantage falls below EXIT, keep the widen until every
    zone has crossed back inside the unwidened band on the widen side.
    """
    if mode not in (MODE_HEAT, MODE_COOL):
        _clear_cop_widen(state)
        return 0.0, "none"

    advance_ratio, defer_ratio = _arbitrage_ratios(snap, mode, centers)
    # Prefer advance on a tie (same rule as _forecast_cop_arbitrage).
    if advance_ratio >= defer_ratio and advance_ratio >= COP_ADVANTAGE_ENTER:
        state.cop_timing = "advance"
        state.cop_widen_k = COP_WIDEN_MAX_K
        state.cop_widen_mode = mode
    elif defer_ratio > advance_ratio and defer_ratio >= COP_ADVANTAGE_ENTER:
        state.cop_timing = "defer"
        state.cop_widen_k = COP_WIDEN_MAX_K
        state.cop_widen_mode = mode
    elif state.cop_widen_k > 0.0:
        hold_mode = state.cop_widen_mode or mode
        if state.cop_timing == "advance":
            hold_ratio = advance_ratio
        elif state.cop_timing == "defer":
            hold_ratio = defer_ratio
        else:
            hold_ratio = 0.0
        # Recompute ratios for the latched widen mode when it differs from
        # the caller mode (recovery after a false opposite-mode flip).
        if hold_mode != mode:
            advance_ratio, defer_ratio = _arbitrage_ratios(snap, hold_mode, centers)
            if state.cop_timing == "advance":
                hold_ratio = advance_ratio
            elif state.cop_timing == "defer":
                hold_ratio = defer_ratio
        if hold_ratio < COP_ADVANTAGE_EXIT:
            if _outside_base_band_on_widen_side(zones, centers, snap, hold_mode, state.cop_timing):
                # Recovery latch: keep efficiency band until temps re-enter.
                state.cop_widen_mode = hold_mode
            else:
                _clear_cop_widen(state)
    else:
        _clear_cop_widen(state)
    return state.cop_widen_k, state.cop_timing


def _apply_cop_widen_bands(
    zones: list[ZoneSnapshot],
    bands: dict[str, tuple[float, float]],
    centers: dict[str, float],
    snap: HouseSnapshot,
    preset: str,
    mode: str,
    widen_k: float,
) -> None:
    """Mutate ``bands`` with the efficiency half-band stretch for ``mode``."""
    if widen_k <= 0.0 or mode not in (MODE_HEAT, MODE_COOL):
        return
    s = snap.settings
    boost = preset == PRESET_BOOST
    if boost and mode == MODE_COOL:
        extra_lo, extra_hi = widen_k, 0.0
    elif boost and mode == MODE_HEAT:
        extra_lo, extra_hi = 0.0, widen_k
    else:
        extra_lo = extra_hi = widen_k
    for zone in zones:
        bands[zone.zone_id] = comfort.zone_band(
            s,
            centers[zone.zone_id],
            zone.occupied,
            snap.house_occupied,
            extra_lo_k=extra_lo,
            extra_hi_k=extra_hi,
        )
