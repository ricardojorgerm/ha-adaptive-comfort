"""Per-state sensor drift learning.

AC internal sensors read differently depending on whether air is being
moved: idle units suffer stratification and electronics self-heating,
while a running fan mixes room air past the sensor. We therefore learn
one offset per operating state against the zone's external reference
sensor and use the offset of the *current* state to correct readings
when no external sensor is available (or to translate setpoints, since
the head regulates on its internal sensor).
"""

from __future__ import annotations

from .types import HEAD_STATES, STATE_STANDBY


class DriftEstimator:
    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha
        self.offsets: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def update(self, state: str, t_internal: float, t_ref: float) -> None:
        """Feed one stable-state observation (caller gates on >=10 min stability)."""
        if state not in HEAD_STATES:
            state = STATE_STANDBY
        delta = t_internal - t_ref
        if state in self.offsets:
            self.offsets[state] += self.alpha * (delta - self.offsets[state])
        else:
            self.offsets[state] = delta
        self.counts[state] = self.counts.get(state, 0) + 1

    def offset(self, state: str) -> float:
        """Learned offset for a state, with fallback to any learned state."""
        if state in self.offsets:
            return self.offsets[state]
        if self.offsets:
            return sum(self.offsets.values()) / len(self.offsets)
        return 0.0

    def correct(self, t_internal: float, state: str) -> float:
        """Estimate true room temperature from the internal reading."""
        return t_internal - self.offset(state)

    def to_dict(self) -> dict:
        return {"offsets": dict(self.offsets), "counts": dict(self.counts)}

    @classmethod
    def from_dict(cls, data: dict) -> DriftEstimator:
        est = cls()
        est.offsets = {str(k): float(v) for k, v in data.get("offsets", {}).items()}
        est.counts = {str(k): int(v) for k, v in data.get("counts", {}).items()}
        return est
