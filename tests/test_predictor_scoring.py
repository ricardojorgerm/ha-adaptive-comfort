"""Predicted-vs-actual scoring (core/predictor.py) and t_house_hourly wiring."""

from custom_components.adaptive_comfort.core.predictor import (
    HORIZONS_MIN,
    PredictorScorer,
)
from custom_components.adaptive_comfort.core.thermal import ThermalModel

TS0 = 1_700_000_000.0


def _fitted_model(k_out: float = 0.3, k_mix: float = 0.0) -> ThermalModel:
    model = ThermalModel(volume_m3=30.0)
    model.fits[False].theta[0] = k_out
    model.fits[False].theta[1] = k_mix
    model.fits[False].samples = 1000
    return model


def test_record_creates_pending_with_full_context():
    scorer = PredictorScorer()
    coeffs = {"k_out_h": 0.3, "k_mix_h": 0.1}
    scorer.record(TS0, "z1", 15, predicted_t=22.5, coeffs=coeffs, forecast_t_out=30.0)
    dumped = scorer.to_dict()
    assert len(dumped["pending"]) == 1
    item = dumped["pending"][0]
    assert item["ts"] == TS0
    assert item["zone_id"] == "z1"
    assert item["horizon_min"] == 15
    assert item["predicted_t"] == 22.5
    assert item["coeffs"] == coeffs
    assert item["forecast_t_out"] == 30.0


def test_scores_only_when_zone_off_throughout_horizon():
    """Free-float observable prediction scores; AC-running prediction is dropped."""
    scorer = PredictorScorer()
    scorer.record(TS0, "z1", 15, predicted_t=22.0, coeffs={}, forecast_t_out=None)
    due_ts = TS0 + 15 * 60.0

    # Horizon not yet elapsed: nothing scored, prediction stays pending.
    scored = scorer.score_due(due_ts - 1.0, {"z1": TS0 - 100.0}, {"z1": 21.5}, actual_t_out=None)
    assert scored == []
    assert scorer.pending_count("z1") == 1

    # Horizon elapsed, zone off continuously since before the prediction was made.
    scored = scorer.score_due(due_ts, {"z1": TS0 - 100.0}, {"z1": 21.5}, actual_t_out=None)
    assert len(scored) == 1
    assert abs(scored[0].error_k - 0.5) < 1e-9  # predicted(22.0) - actual(21.5)
    assert scorer.pending_count("z1") == 0
    stats = scorer.stats("z1", 15)
    assert stats == {"bias_k": 0.5, "mae_k": 0.5, "n": 1}
    assert scorer.last_scored_ts("z1", 15) == due_ts


def test_ac_running_during_horizon_is_dropped_not_scored():
    scorer = PredictorScorer()
    scorer.record(TS0, "z2", 30, predicted_t=23.0, coeffs={}, forecast_t_out=None)
    due_ts = TS0 + 30 * 60.0

    # Currently conditioning (off_since None): not free-float observable.
    scored = scorer.score_due(due_ts, {"z2": None}, {"z2": 21.0}, actual_t_out=None)
    assert scored == []
    assert scorer.stats("z2", 30) is None
    assert scorer.pending_count("z2") == 0  # resolved (dropped), not left pending forever


def test_ac_ran_partway_through_horizon_is_dropped():
    """off_since after the prediction ts means the AC ran for part of the horizon."""
    scorer = PredictorScorer()
    scorer.record(TS0, "z3", 15, predicted_t=23.0, coeffs={}, forecast_t_out=None)
    due_ts = TS0 + 15 * 60.0
    # Zone only went off 5 minutes before "now" -- it was conditioning when
    # the prediction was made and for part of the horizon.
    scored = scorer.score_due(due_ts, {"z3": due_ts - 300.0}, {"z3": 22.0}, actual_t_out=None)
    assert scored == []
    assert scorer.stats("z3", 15) is None


def test_forecast_error_tagged_separately_from_model_error():
    scorer = PredictorScorer()
    scorer.record(TS0, "z4", 60, predicted_t=24.0, coeffs={}, forecast_t_out=30.0)
    due_ts = TS0 + 60 * 60.0
    # Actual outdoor temp came in 3 K above forecast: a forecast miss, not
    # necessarily a thermal-model miss -- must be tagged, not hidden.
    scored = scorer.score_due(due_ts, {"z4": TS0 - 1.0}, {"z4": 23.5}, actual_t_out=33.0)
    assert len(scored) == 1
    sample = scored[0]
    assert sample.forecast_error_k == 3.0
    assert sample.forecast_suspect is True
    # Included by default...
    assert scorer.stats("z4", 60)["n"] == 1
    # ...but can be excluded when isolating true model error.
    assert scorer.stats("z4", 60, exclude_forecast_suspect=True) is None


def test_multiple_horizons_scored_independently():
    scorer = PredictorScorer()
    for horizon in HORIZONS_MIN:
        scorer.record(
            TS0, "z5", horizon, predicted_t=20.0 + horizon, coeffs={}, forecast_t_out=None
        )
    for horizon in HORIZONS_MIN:
        due_ts = TS0 + horizon * 60.0
        scorer.score_due(due_ts, {"z5": TS0 - 1.0}, {"z5": 20.0}, actual_t_out=None)
    for horizon in HORIZONS_MIN:
        stats = scorer.stats("z5", horizon)
        assert stats is not None
        assert abs(stats["bias_k"] - horizon) < 1e-9


def test_predictor_scorer_roundtrip_persistence():
    scorer = PredictorScorer()
    scorer.record(TS0, "z6", 15, predicted_t=22.0, coeffs={"k_out_h": 0.2}, forecast_t_out=29.0)
    scorer.score_due(TS0 + 15 * 60.0, {"z6": TS0 - 1.0}, {"z6": 21.0}, actual_t_out=29.5)
    scorer.record(TS0 + 100.0, "z6", 30, predicted_t=22.5, coeffs={}, forecast_t_out=None)

    restored = PredictorScorer.from_dict(scorer.to_dict())
    assert restored.stats("z6", 15) == scorer.stats("z6", 15)
    assert restored.pending_count("z6") == 1


def test_predict_free_moves_differently_with_hourly_house_trajectory():
    """When k_mix is large, an hourly house-other trajectory should pull the
    prediction away from what a frozen (constant) house-other value gives."""
    model = _fitted_model(k_out=0.1, k_mix=0.6)
    t_in = 24.0
    t_out_hourly = [24.0] * 12  # flat outdoor: isolates the mixing term
    frozen_house = 24.0  # "current" reading, held constant

    # Siblings cooling steadily from 24 down toward 18 over the horizon.
    hourly_house = [24.0 - 0.5 * h for h in range(12)]

    traj_frozen = model.predict_free(
        t_in, t_out_hourly, start_hour=12.0, hours=10.0, t_house_other=frozen_house
    )
    traj_hourly = model.predict_free(
        t_in, t_out_hourly, start_hour=12.0, hours=10.0, t_house_hourly=hourly_house
    )
    assert len(traj_frozen) == len(traj_hourly)
    # Early on both start at t_in and haven't diverged much yet...
    assert abs(traj_frozen[0] - traj_hourly[0]) < 1e-9
    # ...but by the end of the horizon the falling house trajectory has
    # pulled the room down further than the frozen constant did.
    assert traj_hourly[-1] < traj_frozen[-1] - 0.5


def test_predict_horizons_matches_predict_free_at_shared_point():
    """Fine-grained horizon integrator agrees with predict_free's own physics."""
    model = _fitted_model(k_out=0.25, k_mix=0.1)
    t_in = 25.0
    t_out_hourly = [30.0] * 6
    t_house = 22.0

    hourly_traj = model.predict_free(
        t_in, t_out_hourly, start_hour=10.0, hours=2.0, t_house_other=t_house
    )
    horizons = model.predict_horizons(
        t_in, t_out_hourly, start_hour=10.0, horizons_min=(60,), t_house_other=t_house
    )
    # Both integrate the same ODE to t=1h; step-size differences should be small.
    assert abs(horizons[60] - hourly_traj[1]) < 0.05


def test_predict_horizons_returns_all_requested_and_is_monotonic_toward_outdoor():
    model = _fitted_model(k_out=0.3, k_mix=0.0)
    preds = model.predict_horizons(20.0, [30.0] * 3, start_hour=12.0, horizons_min=HORIZONS_MIN)
    assert set(preds.keys()) == set(HORIZONS_MIN)
    # Warming toward a hot outdoor: longer horizon -> more warming.
    assert preds[15] < preds[30] < preds[60]
