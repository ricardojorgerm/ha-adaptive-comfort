"""Plant compression clock and per-zone on/off chatter guards."""

from __future__ import annotations

from . import power
from .types import ControllerState, HouseSnapshot

MAX_MODE_CHANGES_PER_H = 3
# Per-zone anti-chatter dwell (s). Plant-level min_on_min is enforced against
# the compressor run clock, not this — zones may leave demand into residual
# park while the plant minimum is still running.
ZONE_CHATTER_S = 180.0


def _update_plant_compress(state: ControllerState, snap: HouseSnapshot) -> None:
    """Track plant compression from electrical p_ac with StartCounter-class debounce.

    Brief sub-floor dips (inverter modulation, baseline noise) must not clear
    the run clock — otherwise plant min_on never completes. Same 180 s floor
    dwell as StartCounter before we treat compression as ended.
    """
    compressing = snap.p_ac is not None and snap.p_ac >= snap.compression_floor_w
    if compressing:
        if state.plant_compress_since <= 0.0:
            state.plant_compress_since = snap.now_ts
        state.plant_below_since = 0.0
        return
    if state.plant_compress_since <= 0.0:
        state.plant_below_since = 0.0
        return
    if state.plant_below_since <= 0.0:
        state.plant_below_since = snap.now_ts
        return
    if snap.now_ts - state.plant_below_since >= power.START_DEBOUNCE_S:
        state.plant_compress_since = 0.0
        state.plant_below_since = 0.0


def _plant_min_on_active(state: ControllerState, snap: HouseSnapshot, s) -> bool:
    """True while a live compression run has not yet reached min_on_min."""
    if state.plant_compress_since <= 0.0:
        return False
    return snap.now_ts - state.plant_compress_since < s.min_on_min * 60.0


def _transition_allowed(
    state: ControllerState, zone_id: str, now: float, turning_on: bool, s
) -> bool:
    since = state.zone_since.get(zone_id, 0.0)
    elapsed = now - since
    if turning_on and elapsed < s.min_off_min * 60.0:
        return False
    # Zone off uses a short anti-chatter dwell; plant min_on is separate
    # (residual park / handoff keep the compressor loaded).
    if not turning_on and elapsed < ZONE_CHATTER_S:
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
