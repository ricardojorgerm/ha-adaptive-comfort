"""Operating-regime policy: ventilate / continuous / cycling."""

from __future__ import annotations

from .types import MODE_COOL, MODE_HEAT, ControllerState, HouseSnapshot

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
