"""Power pipeline: load composition, baseline, event deltas, shedding math.

The house only has a meter-side (grid) power sensor. The AC draw is
inferred by combining a time-of-day baseline (learned while all heads
are off) with robust step deltas measured when heads switch. An optional
battery sensor lets consumption exceed the grid reading while discharging.
"""

from __future__ import annotations

from collections import deque
from statistics import median

BASELINE_SLOTS = 48  # 30-minute time-of-day slots
EVENT_PRE_S = (-240.0, -15.0)
EVENT_POST_S = (60.0, 240.0)
MIN_PLAUSIBLE_STEP_W = 60.0
MAX_PLAUSIBLE_STEP_W = 4000.0


def outdoor_band(t_out: float | None) -> str | None:
    """Coarse outdoor-condition band for empirical COP bookkeeping.

    Head-count and weather are confounded in practice (many heads run in hot
    afternoons, single heads at mild night), so per-band tables let the two
    effects be separated during analysis.
    """
    if t_out is None:
        return None
    if t_out < 25.0:
        return "mild"
    if t_out < 30.0:
        return "warm"
    return "hot"


COMPRESSION_FLOOR_W = 75.0  # default = fan_floor_w(1); callers pass live floor
START_DEBOUNCE_S = 180.0  # power must stay below the floor this long to arm
START_WINDOW_S = 24.0 * 3600.0


class StartCounter:
    """Counts compressor starts from the electrical record.

    A start is a rising edge of (ac_power > floor) after the draw has been
    below the floor for at least START_DEBOUNCE_S — brief dips from modulation
    do not re-arm. Pass the same fan_floor_w(open_heads) used for park gating
    so multi-head fan-type parks are not counted as compression restarts. Events carry
    the control-state key so inner-loop (park hysteresis) restarts are
    separable from outer-loop (controller cycling) restarts.
    """

    def __init__(self) -> None:
        self.events: list[tuple[float, str]] = []  # (ts, state_key)
        self._above = False
        self._below_since: float | None = None

    def update(
        self,
        now: float,
        ac_w: float | None,
        state_key: str | None,
        floor_w: float | None = None,
    ) -> bool:
        if ac_w is None:
            return False
        floor = COMPRESSION_FLOOR_W if floor_w is None else floor_w
        started = False
        if ac_w > floor:
            armed = self._below_since is not None and now - self._below_since >= START_DEBOUNCE_S
            if not self._above and (armed or not self.events):
                self.events.append((now, state_key or "unknown"))
                started = True
            self._above = True
            self._below_since = None
        else:
            if self._above or self._below_since is None:
                self._below_since = now
            self._above = False
        cutoff = now - 2 * START_WINDOW_S
        while self.events and self.events[0][0] < cutoff:
            self.events.pop(0)
        return started

    def per_hour(self, now: float, window_s: float = START_WINDOW_S) -> float:
        n = sum(1 for ts, _ in self.events if ts >= now - window_s)
        return n / (window_s / 3600.0)

    def current_run_started_at(self) -> float | None:
        """Start timestamp of the compressor run in progress (None if idle).

        The plant-level minimum-run-time policy needs "how long has the
        compressor been continuously above the floor", not per-zone
        `zone_since`: this is that answer straight from the electrical edge
        detector, no separate bookkeeping required (the last recorded start
        event *is* the current run's start while `_above` holds).
        """
        if not self._above or not self.events:
            return None
        return self.events[-1][0]

    def by_state(self, now: float, window_s: float = START_WINDOW_S) -> dict[str, int]:
        out: dict[str, int] = {}
        for ts, key in self.events:
            if ts >= now - window_s:
                out[key] = out.get(key, 0) + 1
        return out

    def to_dict(self) -> dict:
        return {
            "events": [[ts, key] for ts, key in self.events[-500:]],
            "above": self._above,
            "below_since": self._below_since,
        }

    @classmethod
    def from_dict(cls, data: dict) -> StartCounter:
        c = cls()
        for item in data.get("events", []):
            try:
                c.events.append((float(item[0]), str(item[1])))
            except (TypeError, ValueError, IndexError):
                continue
        # Restore edge state so a restart mid-run does not recount the same start.
        if "above" in data:
            c._above = bool(data["above"])
        elif c.events:
            c._above = True
        if data.get("below_since") is not None:
            try:
                c._below_since = float(data["below_since"])
            except (TypeError, ValueError):
                c._below_since = None
        return c


def control_state_key(any_conditioning: bool, any_parked: bool) -> str | None:
    """House-wide control-state label for COP bookkeeping."""
    if any_conditioning and any_parked:
        return "mixed"
    if any_conditioning:
        return "conditioning"
    if any_parked:
        return "park"
    return None


def compose_load(
    p_grid: float,
    p_battery: float | None = None,
    battery_positive_discharging: bool = True,
    known_loads: list[float] | None = None,
) -> float:
    """Total house load attributable to unmonitored devices (incl. the AC)."""
    load = p_grid
    if p_battery is not None:
        discharge = p_battery if battery_positive_discharging else -p_battery
        load += max(0.0, discharge)
    for p in known_loads or []:
        load -= p
    return max(0.0, load)


class BaselineModel:
    """EWMA baseline of non-AC load per 30-min time-of-day slot."""

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.slots: list[float | None] = [None] * BASELINE_SLOTS

    @staticmethod
    def slot_for(local_hour: float) -> int:
        return int(local_hour * 2.0) % BASELINE_SLOTS

    def update(self, local_hour: float, p_load: float) -> None:
        """Feed only while all heads have been off >= 10 min (caller gates)."""
        idx = self.slot_for(local_hour)
        cur = self.slots[idx]
        self.slots[idx] = p_load if cur is None else cur + self.alpha * (p_load - cur)

    def value(self, local_hour: float, *, fallback: bool = True) -> float | None:
        """Return the baseline for this half-hour slot.

        When the current slot has never been learned, the median of other
        slots is diagnostics-grade only: park gating / StartCounter should
        pass ``fallback=False`` so a hot-afternoon gap does not invent
        phantom AC from night medians.
        """
        idx = self.slot_for(local_hour)
        if self.slots[idx] is not None:
            return self.slots[idx]
        if not fallback:
            return None
        known = [s for s in self.slots if s is not None]
        return median(known) if known else None

    def coverage(self) -> dict[str, float | int]:
        """How many of the 48 half-hour slots have a learned (non-fallback) value.

        Hot days barely clear the all-heads-off gate, so before/after kWh
        comparisons need this visible — a sparse baseline invents phantom AC.
        """
        filled = sum(1 for s in self.slots if s is not None)
        return {
            "slots_filled": filled,
            "slots_total": BASELINE_SLOTS,
            "fraction": filled / BASELINE_SLOTS if BASELINE_SLOTS else 0.0,
        }

    def to_dict(self) -> dict:
        return {"slots": list(self.slots)}

    @classmethod
    def from_dict(cls, data: dict) -> BaselineModel:
        model = cls()
        slots = data.get("slots") or []
        if len(slots) == BASELINE_SLOTS:
            model.slots = [None if s is None else float(s) for s in slots]
        return model


def measure_step(pre_samples: list[float], post_samples: list[float]) -> float | None:
    """Robust load step across a head on/off event (medians of both windows)."""
    if len(pre_samples) < 3 or len(post_samples) < 3:
        return None
    return median(post_samples) - median(pre_samples)


class DrawEstimator:
    """Robust per-(zone, mode) electrical-draw statistics from step events.

    Mirrored heads switch together, so a measured step is the combined
    zone draw; callers divide by the head count for per-head figures.
    """

    def __init__(self, max_events: int = 50) -> None:
        self.events: dict[str, deque[float]] = {}

    @staticmethod
    def _key(zone_id: str, mode: str) -> str:
        return f"{zone_id}|{mode}"

    def add_event(self, zone_id: str, mode: str, delta_w: float) -> bool:
        """Add one |step| observation; returns False if rejected as outlier."""
        mag = abs(delta_w)
        if not (MIN_PLAUSIBLE_STEP_W <= mag <= MAX_PLAUSIBLE_STEP_W):
            return False
        key = self._key(zone_id, mode)
        bucket = self.events.setdefault(key, deque(maxlen=50))
        if len(bucket) >= 5:
            med = median(bucket)
            mad = median([abs(x - med) for x in bucket])
            # MAD can be zero for a stable draw; keep a relative floor so
            # gross outliers (another appliance switching) are still rejected.
            threshold = max(5.0 * mad, 0.3 * med, 150.0)
            if abs(mag - med) > threshold:
                return False
        bucket.append(mag)
        return True

    def draw_w(self, zone_id: str, mode: str) -> float | None:
        bucket = self.events.get(self._key(zone_id, mode))
        if bucket:
            return median(bucket)
        # Fall back to the other mode's draw for the same zone.
        for key, other in self.events.items():
            if key.startswith(f"{zone_id}|") and other:
                return median(other)
        return None

    def to_dict(self) -> dict:
        return {"events": {k: list(v) for k, v in self.events.items()}}

    @classmethod
    def from_dict(cls, data: dict) -> DrawEstimator:
        est = cls()
        for key, values in data.get("events", {}).items():
            est.events[key] = deque([float(v) for v in values], maxlen=50)
        return est


def estimate_ac_power(
    p_load: float,
    baseline: float | None,
    p_max_plausible: float = 6000.0,
) -> float | None:
    """Continuous AC power estimate while any head is on."""
    if baseline is None:
        return None
    return min(max(0.0, p_load - baseline), p_max_plausible)


def allocate_power(
    p_ac_w: float,
    zones: list[tuple[str, float, float]],
) -> dict[str, float]:
    """Split total AC power across active zones.

    zones: (zone_id, learned_draw_w, modulation_proxy) for each active zone,
    where modulation_proxy is |T_set - T_in| clamped to a small floor so a
    zone at setpoint still gets standby share.
    """
    weights = {z: max(draw, 1.0) * max(mod, 0.2) for z, draw, mod in zones}
    total = sum(weights.values())
    if total <= 0:
        return dict.fromkeys(weights, 0.0)
    return {z: p_ac_w * w / total for z, w in weights.items()}


# -- contracted-power shedding ------------------------------------------------

SHED_SUSTAINED_S = 5.0
SHED_CRITICAL_PCT = 0.96
SHED_PEAK_WINDOW_S = 120.0
SHED_DEFAULT_START_PCT = 0.88
KNOWN_LOAD_SPIKE_W = 800.0
DEFAULT_AC_DRAW_W = 400.0


def estimated_active_ac_draw_w(
    zone_draws: list[float | None],
    p_ac_est: float | None = None,
) -> float:
    """Conservative in-service AC draw when the meter lags or under-reports."""
    learned = sum(w for w in zone_draws if w is not None)
    if p_ac_est is not None and p_ac_est > 0:
        return max(learned, p_ac_est)
    if learned > 0:
        return learned
    if any(w is None for w in zone_draws):
        return DEFAULT_AC_DRAW_W * len(zone_draws)
    return 0.0


def contracted_demand_w(
    p_grid: float | None,
    known_loads: list[float] | None = None,
    active_ac_w: float = 0.0,
    p_grid_peak: float | None = None,
) -> float | None:
    """Best-effort whole-house demand for shedding (max of meter and parts sum).

    When the grid sensor lags (common with energy-style meters), a large known
    load such as an oven plus running AC can exceed the contract limit even
    though the latest grid sample still looks low.
    """
    parts: list[float] = []
    if p_grid is not None:
        parts.append(p_grid)
    if p_grid_peak is not None:
        parts.append(p_grid_peak)
    known_sum = sum(known_loads or [])
    if known_sum > 0.0 or active_ac_w > 0.0:
        parts.append(known_sum + active_ac_w)
    return max(parts) if parts else None


def shed_needed(
    p_demand: float | None,
    limit_w: float,
    start_pct: float,
    over_since_s: float | None,
    *,
    sustained_s: float = SHED_SUSTAINED_S,
    critical_pct: float = SHED_CRITICAL_PCT,
    urgent: bool = False,
) -> bool:
    """True when contracted demand has exceeded the shed threshold long enough."""
    if p_demand is None:
        return False
    if urgent or p_demand >= critical_pct * limit_w:
        return True
    if p_demand <= start_pct * limit_w:
        return False
    return over_since_s is not None and over_since_s >= sustained_s


def restore_allowed(
    p_grid: float | None,
    limit_w: float,
    restore_pct: float,
    zone_draw_w: float | None,
    margin_w: float = 100.0,
) -> bool:
    """True when there is headroom to restore a zone with the given draw."""
    if p_grid is None:
        return False
    if p_grid >= restore_pct * limit_w:
        return False
    draw = zone_draw_w if zone_draw_w is not None else 800.0  # conservative default
    return (limit_w - p_grid) > (draw + margin_w)
