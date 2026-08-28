"""Signed head depth plus distinct chase-floor and hold-session clocks.

``zone_park_margin`` is the only redundant mirror (d > 0). Chase floor
(``zone_track_delta``), hold_since, hold_ref, and probe clocks stay separate.
"""

from __future__ import annotations

from .park import MARGIN_BIN_K, MARGIN_MAX_K, MARGIN_MIN_K
from .types import ControllerState, ZoneSnapshot

PARK_MARGIN_K = MARGIN_MIN_K
PARK_MARGIN_MAX_K = MARGIN_MAX_K
PARK_MARGIN_STEP_K = MARGIN_BIN_K

TRACK_DELTA_MIN_K = 0.5
TRACK_DELTA_MAX_K = 2.5
TRACK_DELTA_STEP_K = 0.5
TRACK_DELTA_DEFAULT_K = 0.5

DEPTH_STEP_K = PARK_MARGIN_STEP_K
DEPTH_MAX_K = PARK_MARGIN_MAX_K
DEPTH_MIN_K = -TRACK_DELTA_MAX_K


class HeadDepth:
    """Per-zone view over ControllerState depth / hold / probe fields."""

    def __init__(self, state: ControllerState, zid: str) -> None:
        self.state = state
        self.zid = zid

    @property
    def current(self) -> float | None:
        return self.state.zone_head_depth_k.get(self.zid)

    @property
    def chase_floor(self) -> float:
        chase = self.state.zone_track_delta.get(self.zid, TRACK_DELTA_DEFAULT_K)
        chase = min(max(float(chase), TRACK_DELTA_MIN_K), TRACK_DELTA_MAX_K)
        return -quantize_depth(chase)

    @property
    def hold_since(self) -> float | None:
        return self.state.zone_parked_since.get(self.zid)

    @property
    def hold_ref(self) -> float | None:
        return self.state.zone_park_ref.get(self.zid)

    def sync(
        self,
        depth_k: float,
        now: float,
        *,
        lo: float | None = None,
        hi: float | None = None,
        zone: ZoneSnapshot | None = None,
    ) -> float:
        """Persist signed depth and mirror park_margin / track_delta views.

        Leaving a hold (``depth <= 0`` while ``parked_since`` is set) goes
        through ``clear_hold`` so probe budget / abort clocks fire. A bare
        pop of ``zone_parked_since`` would skip those clocks and leak
        ``zone_park_ref``.
        """
        depth = clamp_depth(depth_k, lo, hi)
        zid = self.zid
        state = self.state
        if depth <= 0.0 and zid in state.zone_parked_since:
            self.clear_hold(zone, now)
        state.zone_head_depth_k[zid] = depth
        if depth > 0.0:
            state.zone_park_margin[zid] = depth
            if zid not in state.zone_parked_since:
                state.zone_parked_since[zid] = now
            state.zone_track_delta[zid] = max(
                state.zone_track_delta.get(zid, TRACK_DELTA_DEFAULT_K), TRACK_DELTA_MIN_K
            )
        else:
            state.zone_track_delta[zid] = abs(depth) if depth < 0.0 else TRACK_DELTA_MIN_K
        return depth

    def clear_hold(self, zone: ZoneSnapshot | None = None, now: float = 0.0) -> None:
        """Drop the hold session and leftover positive depth (probe bookkeeping)."""
        zid = self.zid
        state = self.state
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
        state.zone_head_depth_k.pop(zid, None)


def quantize_depth(depth_k: float) -> float:
    return round(float(depth_k) / DEPTH_STEP_K) * DEPTH_STEP_K


def clamp_depth(depth_k: float, lo: float | None = None, hi: float | None = None) -> float:
    stepped = quantize_depth(depth_k)
    lo_b = DEPTH_MIN_K if lo is None else float(lo)
    hi_b = DEPTH_MAX_K if hi is None else float(hi)
    if lo_b > hi_b:
        lo_b, hi_b = hi_b, lo_b
    return min(max(stepped, lo_b), hi_b)


def _quantize_depth(depth_k: float) -> float:
    return quantize_depth(depth_k)


def _clamp_depth(depth_k: float, lo: float | None = None, hi: float | None = None) -> float:
    return clamp_depth(depth_k, lo, hi)


def _chase_floor_k(state: ControllerState, zid: str) -> float:
    return HeadDepth(state, zid).chase_floor


def _sync_depth_views(
    state: ControllerState,
    zid: str,
    depth_k: float,
    now: float,
    *,
    lo: float | None = None,
    hi: float | None = None,
    zone: ZoneSnapshot | None = None,
) -> float:
    return HeadDepth(state, zid).sync(depth_k, now, lo=lo, hi=hi, zone=zone)


def _clear_park_session(
    state: ControllerState, zid: str, zone: ZoneSnapshot | None = None, now: float = 0.0
) -> None:
    HeadDepth(state, zid).clear_hold(zone, now)
