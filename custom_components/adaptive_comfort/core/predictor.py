"""Predicted-vs-actual scoring for the zone free-float predictor.

Records a prediction (predicted temperature at a fixed horizon, alongside the
model coefficients and forecast inputs that produced it) at the moment it is
made. When the horizon elapses, scores it against the actual observed
temperature -- but *only* when the zone's heads were off for the entire
horizon, so the score is honestly a test of the free-float model and not
contaminated by AC operation the model never claimed to predict.

This module deliberately reports real prediction error (`predicted - actual`)
rather than a comfort-style "distance from setpoint" number -- that confusion
is exactly the bug this module replaces (see coordinator's old
`_update_free_float_bias`, renamed `_update_free_float_deviation`). A
prediction whose horizon elapses while the AC ran is dropped, not scored with
a fudge factor: it is simply not evidence about the free-float model.
"""

from __future__ import annotations

from dataclasses import dataclass

HORIZONS_MIN: tuple[int, ...] = (15, 30, 60)

# Bounded history per (zone, horizon) so bias/MAE track recent performance.
MAX_SCORED_SAMPLES = 300
# A pending prediction whose horizon never resolves cleanly (zone kept
# cycling on/off) is abandoned after this long rather than kept forever.
MAX_PENDING_AGE_S = 3.0 * 3600.0
# Outdoor-forecast miss beyond this many K is tagged: large error here may be
# a bad forecast, not a bad thermal model -- separate the two, don't hide it.
FORECAST_ERROR_TAG_K = 1.5


@dataclass
class PendingPrediction:
    """One not-yet-resolved prediction, awaiting its horizon."""

    ts: float
    zone_id: str
    horizon_min: int
    predicted_t: float
    coeffs: dict
    forecast_t_out: float | None  # outdoor temp the forecast used at due_ts
    t_house_used: float | None = None  # house-mixing input used, for audit

    @property
    def due_ts(self) -> float:
        return self.ts + self.horizon_min * 60.0


@dataclass
class ScoredSample:
    ts: float
    predicted_t: float
    actual_t: float
    error_k: float  # predicted - actual (signed prediction error)
    forecast_t_out: float | None
    actual_t_out: float | None
    forecast_error_k: float | None  # actual_t_out - forecast_t_out
    forecast_suspect: bool = False  # forecast_error_k beyond FORECAST_ERROR_TAG_K


class PredictorScorer:
    """Per-zone predicted-vs-actual scoring at fixed horizons."""

    def __init__(self) -> None:
        self._pending: list[PendingPrediction] = []
        self._scored: dict[tuple[str, int], list[ScoredSample]] = {}

    def record(
        self,
        ts: float,
        zone_id: str,
        horizon_min: int,
        predicted_t: float,
        coeffs: dict,
        forecast_t_out: float | None,
        t_house_used: float | None = None,
    ) -> None:
        self._pending.append(
            PendingPrediction(
                ts=ts,
                zone_id=zone_id,
                horizon_min=horizon_min,
                predicted_t=predicted_t,
                coeffs=dict(coeffs),
                forecast_t_out=forecast_t_out,
                t_house_used=t_house_used,
            )
        )

    def score_due(
        self,
        now_ts: float,
        zone_off_since: dict[str, float | None],
        actual_temp: dict[str, float],
        actual_t_out: float | None,
    ) -> list[ScoredSample]:
        """Resolve every pending prediction whose horizon has elapsed.

        `zone_off_since[zone_id]` is the timestamp the zone's heads have been
        continuously off since (`None` if currently conditioning) -- the same
        signal that gates the free-float RLS fit itself. A prediction only
        scores when the zone was off continuously from *before* it was made
        through `now`; otherwise the AC ran during the horizon and the
        prediction is dropped (not scored, not kept, not fudged).
        """
        remaining: list[PendingPrediction] = []
        newly_scored: list[ScoredSample] = []
        for pending in self._pending:
            if now_ts < pending.due_ts:
                if now_ts - pending.ts < MAX_PENDING_AGE_S:
                    remaining.append(pending)
                continue
            off_since = zone_off_since.get(pending.zone_id)
            actual = actual_temp.get(pending.zone_id)
            free_float_observable = (
                off_since is not None and off_since <= pending.ts and actual is not None
            )
            if free_float_observable:
                error = pending.predicted_t - actual
                forecast_err = None
                suspect = False
                if pending.forecast_t_out is not None and actual_t_out is not None:
                    forecast_err = actual_t_out - pending.forecast_t_out
                    suspect = abs(forecast_err) > FORECAST_ERROR_TAG_K
                sample = ScoredSample(
                    ts=pending.due_ts,
                    predicted_t=pending.predicted_t,
                    actual_t=actual,
                    error_k=error,
                    forecast_t_out=pending.forecast_t_out,
                    actual_t_out=actual_t_out,
                    forecast_error_k=forecast_err,
                    forecast_suspect=suspect,
                )
                key = (pending.zone_id, pending.horizon_min)
                samples = self._scored.setdefault(key, [])
                samples.append(sample)
                if len(samples) > MAX_SCORED_SAMPLES:
                    del samples[: len(samples) - MAX_SCORED_SAMPLES]
                newly_scored.append(sample)
            # else: heads ran during the horizon (or actual missing) -- this
            # is not evidence about the free-float model; drop it silently.
        self._pending = remaining
        return newly_scored

    def stats(
        self, zone_id: str, horizon_min: int, *, exclude_forecast_suspect: bool = False
    ) -> dict | None:
        """Bias (mean signed error) and MAE for one zone/horizon, in K."""
        samples = self._scored.get((zone_id, horizon_min))
        if not samples:
            return None
        use = [s for s in samples if not (exclude_forecast_suspect and s.forecast_suspect)]
        if not use:
            return None
        n = len(use)
        bias = sum(s.error_k for s in use) / n
        mae = sum(abs(s.error_k) for s in use) / n
        return {"bias_k": round(bias, 3), "mae_k": round(mae, 3), "n": n}

    def last_scored_ts(self, zone_id: str, horizon_min: int) -> float | None:
        """Timestamp of the newest scored sample, or None if none yet."""
        samples = self._scored.get((zone_id, horizon_min))
        if not samples:
            return None
        return samples[-1].ts

    def all_stats(self, zone_id: str) -> dict[int, dict]:
        out = {}
        for horizon in HORIZONS_MIN:
            stat = self.stats(zone_id, horizon)
            if stat is not None:
                out[horizon] = stat
        return out

    def pending_count(self, zone_id: str | None = None) -> int:
        if zone_id is None:
            return len(self._pending)
        return sum(1 for p in self._pending if p.zone_id == zone_id)

    def to_dict(self) -> dict:
        return {
            "pending": [
                {
                    "ts": p.ts,
                    "zone_id": p.zone_id,
                    "horizon_min": p.horizon_min,
                    "predicted_t": p.predicted_t,
                    "coeffs": p.coeffs,
                    "forecast_t_out": p.forecast_t_out,
                    "t_house_used": p.t_house_used,
                }
                for p in self._pending
            ],
            "scored": {
                f"{zid}|{horizon}": [
                    {
                        "ts": s.ts,
                        "predicted_t": s.predicted_t,
                        "actual_t": s.actual_t,
                        "error_k": s.error_k,
                        "forecast_t_out": s.forecast_t_out,
                        "actual_t_out": s.actual_t_out,
                        "forecast_error_k": s.forecast_error_k,
                        "forecast_suspect": s.forecast_suspect,
                    }
                    for s in samples
                ]
                for (zid, horizon), samples in self._scored.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> PredictorScorer:
        scorer = cls()
        for item in data.get("pending", []):
            try:
                scorer._pending.append(
                    PendingPrediction(
                        ts=float(item["ts"]),
                        zone_id=str(item["zone_id"]),
                        horizon_min=int(item["horizon_min"]),
                        predicted_t=float(item["predicted_t"]),
                        coeffs=dict(item.get("coeffs") or {}),
                        forecast_t_out=(
                            float(item["forecast_t_out"])
                            if item.get("forecast_t_out") is not None
                            else None
                        ),
                        t_house_used=(
                            float(item["t_house_used"])
                            if item.get("t_house_used") is not None
                            else None
                        ),
                    )
                )
            except (TypeError, ValueError, KeyError):
                continue
        for key, samples in data.get("scored", {}).items():
            zid, _, horizon_str = str(key).rpartition("|")
            try:
                horizon = int(horizon_str)
            except ValueError:
                continue
            resolved: list[ScoredSample] = []
            for item in samples:
                try:
                    resolved.append(
                        ScoredSample(
                            ts=float(item["ts"]),
                            predicted_t=float(item["predicted_t"]),
                            actual_t=float(item["actual_t"]),
                            error_k=float(item["error_k"]),
                            forecast_t_out=(
                                float(item["forecast_t_out"])
                                if item.get("forecast_t_out") is not None
                                else None
                            ),
                            actual_t_out=(
                                float(item["actual_t_out"])
                                if item.get("actual_t_out") is not None
                                else None
                            ),
                            forecast_error_k=(
                                float(item["forecast_error_k"])
                                if item.get("forecast_error_k") is not None
                                else None
                            ),
                            forecast_suspect=bool(item.get("forecast_suspect", False)),
                        )
                    )
                except (TypeError, ValueError, KeyError):
                    continue
            if resolved:
                scorer._scored[(zid, horizon)] = resolved
        return scorer
