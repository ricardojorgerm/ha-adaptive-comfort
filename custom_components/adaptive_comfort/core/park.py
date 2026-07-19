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
# Default / bounds kept in sync with controller.PARK_MARGIN_* .
DEFAULT_MARGIN_K = 1.0
MARGIN_MIN_K = 1.0
MARGIN_MAX_K = 3.0
# Pull preferred toward a settled session margin on a clean park exit.
MARGIN_SETTLE_ALPHA = 0.3


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

    def update(self, extraction_w: float, action_active: bool) -> None:
        x = max(0.0, extraction_w)
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
        self.preferred_margin_k = min(
            max(self.preferred_margin_k, MARGIN_MIN_K), MARGIN_MAX_K
        )

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
        }

    @classmethod
    def from_dict(cls, data: dict) -> ParkEstimator:
        est = cls()
        if data.get("extraction_w") is not None:
            est.extraction_w = float(data["extraction_w"])
        if data.get("active_ratio") is not None:
            est.active_ratio = float(data["active_ratio"])
        est.samples = int(data.get("samples", 0))
        if data.get("preferred_margin_k") is not None:
            est.preferred_margin_k = min(
                max(float(data["preferred_margin_k"]), MARGIN_MIN_K), MARGIN_MAX_K
            )
        return est
