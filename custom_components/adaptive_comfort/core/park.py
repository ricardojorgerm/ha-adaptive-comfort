"""Learned above-setpoint ("parked") head behavior.

When a head is commanded with a setpoint above its internal reading while
the compressor mode stays active, devices differ: some thermo-off cleanly
(idle), others hold a keep-temperature trickle - modulating at minimal
output in a hysteresis band around the setpoint. On a multi-split, a
"satisfied" head's expansion valve may also pass residual refrigerant
while sibling heads keep the compressor running.

Rather than assuming either behavior, we park heads deliberately (bounded
probes at first, exploitation once learned) and measure what happens:
whether the head still reports active conditioning, and how much heat it
actually moves in the conditioning direction (from the zone thermal
model, which separates AC action from free-float drift). The learned
classification lets the controller choose parking over a hard off when
a zone is satisfied but its standing load persists - trickle output then
covers the load without off/on cycling, and without guessing.

preferred_margin_k is the learned park depth (K from the internal
reading) that has held the room: each park session starts one step below
it (floored at the minimum park margin so the setpoint stays on the park
side of the head), may escalate further, and feeds the value back so the
next park converges on what this head needs.
"""

from __future__ import annotations

# A head is considered a "trickler" when parked extraction is meaningfully
# above measurement noise; "idler" when meaningfully below. In between we
# keep probing.
TRICKLE_MIN_W = 60.0
IDLE_MAX_W = 25.0
CLASSIFY_MIN_SAMPLES = 6
# Electrical truth: below this AC draw there is no compression — just fans
# and electronics — regardless of what hvac_action claims. Field data shows
# parked heads cycling between real compression (~300-460 W) and effectively
# fan-only (<60 W) via their internal hysteresis; hvac_action reports
# 'cooling' throughout, so power is the only honest activity signal.
FAN_FLOOR_W = 90.0
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


def fan_floor_w(n_heads: int = 1) -> float:
    """Electrical floor below which a solo park has no compression.

    Mirrored heads in one zone all fan together, so the house draw scales
    with head count; a fixed 90 W floor would mis-read a 2-3 head coast as
    compression.
    """
    return FAN_FLOOR_W * max(1, int(n_heads))


def margin_bin(margin_k: float | None) -> str | None:
    """Bin a park margin to MARGIN_BIN_K steps ('1.0', '1.5', ...)."""
    if margin_k is None:
        return None
    return f"{round(margin_k / MARGIN_BIN_K) * MARGIN_BIN_K:.1f}"


def gate_observation(
    p_ac_w: float | None,
    solo: bool,
    extraction_w: float,
    n_heads: int = 1,
) -> tuple[float, bool]:
    """Power-gate a parked observation.

    solo (no sibling conditioning): the house AC draw belongs to this park
    alone, so below fan_floor_w(n_heads) the heads are coasting fan-only -
    the thermal model's extraction estimate is phantom and is forced to zero.
    extraction_w is sensed-room / per-head frame (same as ParkEstimator).
    Non-solo: electrical attribution is ambiguous; fall back to judging
    activity from the extraction magnitude itself.
    """
    if solo and p_ac_w is not None:
        if p_ac_w < fan_floor_w(n_heads):
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
        floor = FAN_FLOOR_W if floor_w is None else floor_w
        above = p_ac_w >= floor
        if self.above is None or self.above != above:
            self.above = above
            self.since = now
            return None
        if self.since is None or now - self.since < PARK_DUTY_DEBOUNCE_S:
            return None
        return above


class ParkEstimator:
    """EWMA of parked extraction, idle/trickle class, and preferred margin."""

    def __init__(self, alpha: float = 0.15) -> None:
        self.alpha = alpha
        # Heat moved while parked in the conditioning direction (W, >= 0).
        self.extraction_w: float | None = None
        self.active_ratio: float | None = (
            None  # fraction of parked ticks reporting active conditioning
        )
        self.samples = 0
        # Learned park depth to target on the next park entry (K).
        self.preferred_margin_k: float = DEFAULT_MARGIN_K
        # Per-margin hysteresis map: bin -> [ewma extraction W, ewma
        # compression duty, samples]. Reveals where the head's internal
        # dead-band edges sit: duty ~0 in the coast region, ~1 where the
        # hysteresis keeps re-engaging compression.
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
        """'trickle' | 'idle' | 'unknown' (needs more parked observations)."""
        if self.samples < CLASSIFY_MIN_SAMPLES or self.extraction_w is None:
            return "unknown"
        if self.extraction_w >= TRICKLE_MIN_W:
            return "trickle"
        if self.extraction_w <= IDLE_MAX_W:
            return "idle"
        return "unknown"

    @property
    def fan_only_ratio(self) -> float | None:
        """Fraction of parked time coasting without compression."""
        if self.active_ratio is None:
            return None
        return 1.0 - self.active_ratio

    def coast_margin_k(self) -> float | None:
        """Smallest margin bin whose compression duty is low (<0.3): the
        cheapest park depth that lands inside the head's coast region."""
        best = None
        for bin_key, (_ext, duty, n) in self.margin_bins.items():
            if n >= CLASSIFY_MIN_SAMPLES and duty < 0.3:
                m = float(bin_key)
                if best is None or m < best:
                    best = m
        return best

    @property
    def trickles(self) -> bool | None:
        cls = self.classification
        if cls == "unknown":
            return None
        return cls == "trickle"

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
