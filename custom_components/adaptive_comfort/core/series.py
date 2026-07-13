"""Lightweight time series ring buffer used by the sampler and estimators."""

from __future__ import annotations

from collections import deque
from statistics import median


class TimeSeries:
    """Append-only (ts, value) buffer with a bounded time horizon."""

    def __init__(self, horizon_s: float = 26 * 3600.0) -> None:
        self.horizon_s = horizon_s
        self._data: deque[tuple[float, float]] = deque()

    def append(self, ts: float, value: float) -> None:
        if self._data and ts <= self._data[-1][0]:
            return
        self._data.append((ts, value))
        cutoff = ts - self.horizon_s
        while self._data and self._data[0][0] < cutoff:
            self._data.popleft()

    def __len__(self) -> int:
        return len(self._data)

    @property
    def latest(self) -> tuple[float, float] | None:
        return self._data[-1] if self._data else None

    def window(self, start_ts: float, end_ts: float) -> list[float]:
        return [v for ts, v in self._data if start_ts <= ts <= end_ts]

    def median_window(self, start_ts: float, end_ts: float) -> float | None:
        vals = self.window(start_ts, end_ts)
        return median(vals) if vals else None

    def mean_window(self, start_ts: float, end_ts: float) -> float | None:
        vals = self.window(start_ts, end_ts)
        return sum(vals) / len(vals) if vals else None

    def value_at(self, ts: float, max_age_s: float = 900.0) -> float | None:
        """Most recent value at or before ts, if fresh enough."""
        best: tuple[float, float] | None = None
        for t, v in reversed(self._data):
            if t <= ts:
                best = (t, v)
                break
        if best is None or ts - best[0] > max_age_s:
            return None
        return best[1]
