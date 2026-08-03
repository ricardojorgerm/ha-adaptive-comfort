"""Learned less-conditioned head depth (park / hysteresis hold).

Positive head depth commands a setpoint on the satisfied side of the
internal reading (cool: ``internal + margin``; heat: ``internal - margin``)
while the compressor mode stays active. Devices differ: some thermo-off
cleanly, others hold a keep-temperature residual — modulating at minimal
output in a hysteresis band around the setpoint. On a multi-split, a
"satisfied" head's expansion valve may also pass residual refrigerant
while sibling heads keep the compressor running.

``pick_depth_k`` is the shared selector: demand/helper may choose that
positive depth when residual bins cover standing load (small-house COP
advantage); want-off park / run-out still use the same table. Negative
depth is ordinary tracking (``internal - |delta|`` in cool).

Two axes — do not conflate them:

1. **Thermal classification** (`idle` / `residual` / `unknown`) — EWMA of
   parked *extraction* (W of heat removed from the room model). This is
   not electrical fan watts. `idle` means almost no heat moved (<= IDLE_MAX_W);
   a quiet indoor fan still draws ~60-75 W on the meter.
2. **Electrical compression** (fan floor / duty) — house `p_ac` below
   `fan_floor_w(N)` means no compressor load (fan + electronics only:
   fan-type park). That gates observations and duty bins; it is what
   decides residual vs fan-type for control, not IDLE_MAX_W.

Rather than assuming either device behavior, we park heads deliberately
(bounded probes at first, exploitation once learned) and measure what
happens. The learned thermal class lets the controller prefer park over
hard off when a measured residual head covers standing load.

Entry depth targets the *residual-hold* edge, not the fan-type shelf:
`residual_max_margin_k()` / `residual_edge_k()` find the deepest margin
that is still meaningfully compressing (just before the dead-band edge),
which keeps the head doing real work instead of merely idling its fan.
`fan_type_min_margin_k()` remains for diagnostics (charts the cheapest
no-compression depth) but is no longer used to pick where to park.
Ranking a satisfied zone's options: residual-hold park (real output) >
plain off > fan-type park (fan running, no output) — a live fan-type
session inside the coil-dry window is released to off exactly like
`fan_assist` would be, since blowing air over a wet coil re-evaporates
condensate whether the fan runs standalone or as a keep-temperature park.

preferred_margin_k is the learned park depth (K from the internal
reading) that has held the room: each park session starts one step below
it (floored at the minimum park margin so the setpoint stays on the park
side of the head), may escalate further, and feeds the value back so the
next park converges on what this head needs.
"""

from __future__ import annotations

from .types import MODE_COOL, MODE_HEAT

# Thermal extraction thresholds for ParkEstimator.classification (W of heat
# moved while parked — NOT meter watts / fan draw):
#   idle:     extraction_w <= IDLE_MAX_W   (thermo-off / no useful cooling)
#   unknown:  between the two (keep probing)
#   residual: extraction_w >= RESIDUAL_MIN_W  (residual compression useful)
# Fan electronics alone are typically 60-130 W electric and must not be
# read as these numbers.
RESIDUAL_MIN_W = 60.0
TRICKLE_MIN_W = RESIDUAL_MIN_W  # deprecated alias; remove next release
IDLE_MAX_W = 25.0
CLASSIFY_MIN_SAMPLES = 6
# Electrical truth: below this AC draw there is no compression - just fans
# and electronics - regardless of what hvac_action claims. Field data: one
# indoor fan + electronics sits ~60-75 W; two-head fan-type parks ~100-130 W.
# Floor = base + per_head * open_heads (per_head is HA-tunable).
FAN_FLOOR_BASE_W = 20.0
FAN_FLOOR_PER_HEAD_W = 55.0
# Park duty samples on the 60 s control tick (not the 5 min thermal fit, and
# not sub-minute meter edges). Power must sit on one side of the (per-zone)
# fan floor for this long before a sample counts - brief modulation dips do
# not flip duty.
PARK_DUTY_DEBOUNCE_S = 60.0
MARGIN_BIN_K = 0.5
# Default / bounds kept in sync with controller.PARK_MARGIN_* .
DEFAULT_MARGIN_K = 1.0
MARGIN_MIN_K = 1.0
MARGIN_MAX_K = 3.0
# Pull preferred toward a settled session margin on a clean park exit.
MARGIN_SETTLE_ALPHA = 0.3
# Compression duty at/above this counts as "meaningfully still compressing"
# for the residual-hold search; below it a bin is fan-type (no compression).
# Same threshold as fan_type_min_margin_k's <0.3 so the two searches meet at the
# same boundary rather than leaving a dead zone between them.
RESIDUAL_HOLD_DUTY_MIN = 0.3
# Finer resolution than MARGIN_BIN_K for the dead-band edge: bisecting
# between the deepest still-compressing bin and the shallowest fan-type bin
# above it lands within this many K of the true device hysteresis edge.
EDGE_STEP_K = 0.25
# Fraction of per-head standing load a residual bin must cover to be
# eligible as a hysteresis (positive) depth. Kept in sync with
# controller.PARK_LOAD_COVER_FRACTION.
LOAD_COVER_FRACTION = 0.6


def fan_floor_w(
    n_heads: int = 1,
    per_head_w: float | None = None,
    base_w: float | None = None,
) -> float:
    """Electrical floor below which a solo park has no compression (fan-type).

    This is meter watts, not IDLE_MAX_W. Mirrored heads fan together, so
    house draw scales with head count. Default 20 + 55*N puts East (~75 W)
    and West (~130 W) fan-only parks below the gate while real compression
    (~300 W+) stays above.
    """
    base = FAN_FLOOR_BASE_W if base_w is None else base_w
    per = FAN_FLOOR_PER_HEAD_W if per_head_w is None else per_head_w
    return base + per * max(1, int(n_heads))


def margin_bin(margin_k: float | None) -> str | None:
    """Bin a park margin to MARGIN_BIN_K steps ('1.0', '1.5', ...)."""
    if margin_k is None:
        return None
    return f"{round(margin_k / MARGIN_BIN_K) * MARGIN_BIN_K:.1f}"


def _bin_extraction_w(margin_bins: dict[str, list[float]], margin_k: float) -> float | None:
    """EWMA extraction for the bin containing margin_k, if well sampled."""
    b = margin_bin(margin_k)
    if b is None:
        return None
    entry = margin_bins.get(b)
    if entry is None or entry[2] < CLASSIFY_MIN_SAMPLES:
        return None
    if entry[1] < RESIDUAL_HOLD_DUTY_MIN:
        return None
    return float(entry[0])


def shallowest_covering_margin_k(
    margin_bins: dict[str, list[float]],
    required_w: float,
) -> float | None:
    """Smallest residual-duty margin whose extraction covers ``required_w``."""
    best: float | None = None
    for bin_key, (ext, duty, n) in margin_bins.items():
        if n < CLASSIFY_MIN_SAMPLES or duty < RESIDUAL_HOLD_DUTY_MIN:
            continue
        if float(ext) < required_w:
            continue
        m = float(bin_key)
        if best is None or m < best:
            best = m
    return best


def pick_depth_k(
    *,
    mode: str,
    temp: float | None,
    lo: float,
    hi: float,
    standing_load_w: float | None,
    n_rooms: int,
    park_residuals: bool | None,
    margin_bins: dict[str, list[float]] | None,
    residual_edge_k: float | None,
    track_delta: float,
    park_learning: bool = True,
) -> tuple[float, str]:
    """Signed head depth for cool ``SP = internal + depth`` / heat ``- depth``.

    Negative depth is chase tracking; positive is hysteresis residual hold.
    Pull-down (cool ``temp > hi`` / heat ``temp < lo``) and far-edge
    overshoot never pick positive depth. Unknown / non-residual heads stay
    on track until park probes classify them.
    """
    chase = -abs(track_delta)
    if temp is None or mode not in (MODE_COOL, MODE_HEAT) or not park_learning:
        return chase, "depth_track"
    if (mode == MODE_COOL and temp > hi) or (mode == MODE_HEAT and temp < lo):
        return chase, "depth_track"
    # Past the far edge: demand must not keep coasting deeper.
    if (mode == MODE_COOL and temp <= lo) or (mode == MODE_HEAT and temp >= hi):
        return chase, "depth_track"
    if park_residuals is not True:
        return chase, "depth_track"
    bins = margin_bins or {}
    if standing_load_w is None:
        required = 0.0
    else:
        required = LOAD_COVER_FRACTION * standing_load_w / max(1, int(n_rooms))
    covering = shallowest_covering_margin_k(bins, required)
    if covering is None:
        return chase, "depth_track"
    # Prefer residual_edge when that depth itself covers (entry target).
    if residual_edge_k is not None:
        edge_ext = _bin_extraction_w(bins, residual_edge_k)
        if edge_ext is not None and edge_ext >= required:
            covering = min(max(residual_edge_k, MARGIN_MIN_K), MARGIN_MAX_K)
        else:
            covering = min(max(covering, MARGIN_MIN_K), MARGIN_MAX_K)
    else:
        covering = min(max(covering, MARGIN_MIN_K), MARGIN_MAX_K)
    return covering, "depth_residual"


def gate_observation(
    p_ac_w: float | None,
    solo: bool,
    extraction_w: float,
    n_heads: int = 1,
    per_head_w: float | None = None,
    base_w: float | None = None,
) -> tuple[float, bool]:
    """Power-gate a parked observation.

    solo (no sibling conditioning): the house AC draw belongs to this park
    alone, so below fan_floor_w(...) the session is fan-type (no compression)
    - the thermal model's extraction estimate is phantom and is forced to
    zero (that zero then feeds the idle side of thermal classification).
    extraction_w is sensed-room / per-head frame (same as ParkEstimator).
    Non-solo: electrical attribution is ambiguous; fall back to judging
    activity from the extraction magnitude itself (IDLE_MAX_W).
    """
    if solo and p_ac_w is not None:
        if p_ac_w < fan_floor_w(n_heads, per_head_w=per_head_w, base_w=base_w):
            return 0.0, False
        return max(0.0, extraction_w), True
    return max(0.0, extraction_w), extraction_w > IDLE_MAX_W


class PowerDebounce:
    """Settles AC draw above/below a fan floor before duty samples count.

    Returns None while the side is unknown or still inside the debounce
    window after a crossing; True/False once power has been stable long enough.
    """

    def __init__(self) -> None:
        self.above: bool | None = None
        self.since: float | None = None

    def reset(self) -> None:
        self.above = None
        self.since = None

    def settle(self, now: float, p_ac_w: float | None, floor_w: float | None = None) -> bool | None:
        if p_ac_w is None:
            return None
        floor = fan_floor_w(1) if floor_w is None else floor_w
        above = p_ac_w >= floor
        if self.above is None or self.above != above:
            self.above = above
            self.since = now
            return None
        if self.since is None or now - self.since < PARK_DUTY_DEBOUNCE_S:
            return None
        return above


class ParkEstimator:
    """EWMA of parked thermal extraction, idle/residual class, preferred margin.

    `classification` is thermal (extraction_w). Compression vs fan-type is
    electrical — see fan_floor_w / margin_bins duty / fan_type_min_margin_k.
    """

    def __init__(self, alpha: float = 0.15) -> None:
        self.alpha = alpha
        # Heat moved while parked in the conditioning direction (W, >= 0).
        # Thermal frame — not house meter / fan watts.
        self.extraction_w: float | None = None
        self.active_ratio: float | None = (
            None  # fraction of parked ticks reporting active conditioning
        )
        self.samples = 0
        # Learned park depth to target on the next park entry (K).
        self.preferred_margin_k: float = DEFAULT_MARGIN_K
        # Per-margin hysteresis map: bin -> [ewma extraction W, ewma
        # compression duty, samples]. Duty uses the electrical fan floor:
        # ~0 in the fan-type region, ~1 where hysteresis re-engages
        # compression. That map is the residual vs fan-type axis.
        self.margin_bins: dict[str, list[float]] = {}

    def update(
        self, extraction_w: float, action_active: bool, margin_k: float | None = None
    ) -> None:
        x = max(0.0, extraction_w)
        b = margin_bin(margin_k)
        if b is not None:
            ext, duty, n = self.margin_bins.get(b, [x, 1.0 if action_active else 0.0, 0])
            ext += self.alpha * (x - ext)
            duty += self.alpha * ((1.0 if action_active else 0.0) - duty)
            self.margin_bins[b] = [ext, duty, n + 1]
        if self.extraction_w is None:
            self.extraction_w = x
        else:
            self.extraction_w += self.alpha * (x - self.extraction_w)
        a = 1.0 if action_active else 0.0
        if self.active_ratio is None:
            self.active_ratio = a
        else:
            self.active_ratio += self.alpha * (a - self.active_ratio)
        self.samples += 1

    def raise_preferred(self, margin_k: float) -> None:
        """High-water mark: session needed at least this depth to hold."""
        m = min(max(margin_k, MARGIN_MIN_K), MARGIN_MAX_K)
        if m > self.preferred_margin_k:
            self.preferred_margin_k = m

    def settle_preferred(self, margin_k: float) -> None:
        """Blend preferred toward a margin that held without overcorrection."""
        m = min(max(margin_k, MARGIN_MIN_K), MARGIN_MAX_K)
        self.preferred_margin_k += MARGIN_SETTLE_ALPHA * (m - self.preferred_margin_k)
        self.preferred_margin_k = min(max(self.preferred_margin_k, MARGIN_MIN_K), MARGIN_MAX_K)

    @property
    def classification(self) -> str:
        """Thermal class from extraction EWMA: 'residual' | 'idle' | 'unknown'.

        idle = almost no heat removed (<= IDLE_MAX_W), not 'fan under 30 W
        electric'. Needs CLASSIFY_MIN_SAMPLES. Between IDLE_MAX_W and
        RESIDUAL_MIN_W stays unknown (probe). Electrical fan-type vs residual
        compression is margin_bins duty / fan_floor_w, not this label alone.
        """
        if self.samples < CLASSIFY_MIN_SAMPLES or self.extraction_w is None:
            return "unknown"
        if self.extraction_w >= RESIDUAL_MIN_W:
            return "residual"
        if self.extraction_w <= IDLE_MAX_W:
            return "idle"
        return "unknown"

    @property
    def fan_only_ratio(self) -> float | None:
        """Fraction of parked time not reporting active conditioning (device)."""
        if self.active_ratio is None:
            return None
        return 1.0 - self.active_ratio

    def fan_type_min_margin_k(self) -> float | None:
        """Shallowest margin (K) whose bins are fan-type: mode on, no compression.

        Reads margin_bins for the smallest depth with duty < RESIDUAL_HOLD_DUTY_MIN.
        That is the *fan-type shelf* — useful for charts and for locating the
        dead-band edge next to residual_max_margin_k(). Not an entry target:
        parking here removes no heat and re-humidifies; entry uses
        residual_edge_k() instead.
        """
        best = None
        for bin_key, (_ext, duty, n) in self.margin_bins.items():
            if n >= CLASSIFY_MIN_SAMPLES and duty < RESIDUAL_HOLD_DUTY_MIN:
                m = float(bin_key)
                if best is None or m < best:
                    best = m
        return best

    def fan_park_margin_k(self) -> float | None:
        """Deprecated alias for fan_type_min_margin_k()."""
        return self.fan_type_min_margin_k()

    def coast_margin_k(self) -> float | None:
        """Deprecated alias for fan_type_min_margin_k()."""
        return self.fan_type_min_margin_k()

    def fan_type_margin_k(self) -> float | None:
        """Deprecated alias for fan_type_min_margin_k()."""
        return self.fan_type_min_margin_k()

    def residual_max_margin_k(self) -> float | None:
        """Deepest margin (K) that is still residual: mode on, compressing.

        Walks margin_bins from the compressing side (duty >= threshold) and
        returns the deepest such bin — the hold just before the fan-type shelf.
        Pair with fan_type_min_margin_k() to bracket the dead-band edge.
        """
        best = None
        for bin_key, (_ext, duty, n) in self.margin_bins.items():
            if n >= CLASSIFY_MIN_SAMPLES and duty >= RESIDUAL_HOLD_DUTY_MIN:
                m = float(bin_key)
                if best is None or m > best:
                    best = m
        return best

    def residual_hold_margin_k(self) -> float | None:
        """Deprecated alias for residual_max_margin_k()."""
        return self.residual_max_margin_k()

    def residual_edge_k(self) -> float | None:
        """Residual↔fan-type edge (K), finer than the 0.5 K bin grid.

        Bisects residual_max_margin_k() and fan_type_min_margin_k() when both
        are known; otherwise returns whichever bound exists. This is the
        park *entry* target.
        """
        hold = self.residual_max_margin_k()
        fan = self.fan_type_min_margin_k()
        if hold is not None and fan is not None and fan > hold:
            midpoint = (hold + fan) / 2.0
            return round(midpoint / EDGE_STEP_K) * EDGE_STEP_K
        return hold if hold is not None else fan

    def current_is_fan_type(self, margin_k: float | None) -> bool | None:
        """Live electrical read for a session currently parked at margin_k.

        True: this depth's bin has enough evidence of low duty (fan-type,
        no compression). False: enough evidence of real compression.
        None: no margin, or not enough samples in that bin yet.
        """
        b = margin_bin(margin_k)
        if b is None:
            return None
        entry = self.margin_bins.get(b)
        if entry is None or entry[2] < CLASSIFY_MIN_SAMPLES:
            return None
        return entry[1] < RESIDUAL_HOLD_DUTY_MIN

    def is_fan_type_at(self, margin_k: float | None) -> bool | None:
        """Deprecated alias for current_is_fan_type()."""
        return self.current_is_fan_type(margin_k)

    @property
    def residuals(self) -> bool | None:
        cls = self.classification
        if cls == "unknown":
            return None
        return cls == "residual"

    @property
    def trickles(self) -> bool | None:
        """Deprecated alias for residuals."""
        return self.residuals

    def to_dict(self) -> dict:
        return {
            "extraction_w": self.extraction_w,
            "active_ratio": self.active_ratio,
            "samples": self.samples,
            "preferred_margin_k": self.preferred_margin_k,
            "margin_bins": {k: list(v) for k, v in self.margin_bins.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> ParkEstimator:
        est = cls()
        if data.get("extraction_w") is not None:
            est.extraction_w = float(data["extraction_w"])
        if data.get("active_ratio") is not None:
            est.active_ratio = float(data["active_ratio"])
        est.samples = int(data.get("samples", 0))
        for k, v in data.get("margin_bins", {}).items():
            try:
                est.margin_bins[str(k)] = [float(v[0]), float(v[1]), int(v[2])]
            except (TypeError, ValueError, IndexError):
                continue
        if data.get("preferred_margin_k") is not None:
            est.preferred_margin_k = min(
                max(float(data["preferred_margin_k"]), MARGIN_MIN_K), MARGIN_MAX_K
            )
        return est
