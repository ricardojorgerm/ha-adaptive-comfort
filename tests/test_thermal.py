from custom_components.adaptive_comfort.core.simulator import SimHouse, SimRoom
from custom_components.adaptive_comfort.core.thermal import (
    COP_PRIOR,
    K_PRIOR,
    DiurnalModel,
    ThermalModel,
)

STEP_H = 5.0 / 60.0


def run_free_response(model: ThermalModel, room: SimRoom, house: SimHouse, hours: float):
    """Feed free-response samples from the simulator into the model."""
    prev = room.temp
    elapsed = 0.0
    while elapsed < hours:
        obs = house.step(STEP_H)
        model.update_free(prev, room.temp, obs["t_out"], STEP_H, obs["hour"])
        prev = room.temp
        elapsed += STEP_H


def test_recovers_exchange_constant():
    room = SimRoom(name="bed", k=0.35, volume_m3=30.0, solar_amplitude=0.3)
    house = SimHouse(rooms=[room])
    model = ThermalModel(room.volume_m3)
    run_free_response(model, room, house, hours=120.0)
    assert abs(model.k(False) - 0.35) < 0.1


def test_anchoring_chain():
    model = ThermalModel(volume_m3=30.0)
    model.fits[False].theta[0] = 0.5
    model.fits[False].theta[1] = 0.0
    model.fits[False].samples = 1000
    assert abs(model.k(False) - 0.5) < 1e-9
    assert abs(model.k_mix(False) - 0.0) < 1e-9
    assert abs(model.ach - 0.5) < 1e-9
    assert abs(model.airflow_m3h - 15.0) < 1e-9
    assert abs(model.ua_w_per_k - 0.34 * 15.0) < 1e-9
    assert abs(model.c_air_wh_per_k - 0.34 * 30.0) < 1e-9


def test_passive_cooling_from_house_mixing():
    """Standby room cools because another zone is cold — k_mix absorbs it, not k_out."""
    model = ThermalModel(volume_m3=30.0)
    t_out = 26.0
    t_in = 24.0
    t_house = 20.0
    true_k_out = 0.08
    true_k_mix = 0.35
    dt_h = STEP_H
    for _ in range(250):
        dtdt = true_k_out * (t_out - t_in) + true_k_mix * (t_house - t_in)
        t_prev = t_in
        t_in += dtdt * dt_h
        model.update_free(t_prev, t_in, t_out, dt_h, 12.0, t_house_other=t_house)
    assert model.k_mix(False) > 0.12
    # Without the mixing regressor this scenario crushes k_out toward zero.
    assert model.k(False) > 0.04


def test_legacy_six_param_fit_migrates():
    from custom_components.adaptive_comfort.core.thermal import _expand_legacy_rls

    legacy = {
        "theta": [0.4, 0.1, 0.0, 0.0, 0.0, 0.0],
        "p": [[1.0 if i == j else 0.0 for j in range(6)] for i in range(6)],
        "samples": 50,
        "lam": 0.998,
    }
    expanded = _expand_legacy_rls(legacy)
    assert len(expanded["theta"]) == 7
    assert expanded["theta"][1] == 0.0
    model = ThermalModel.from_dict({"fit_closed": legacy, "fit_open": legacy}, 30.0)
    assert model.fits[False].theta[0] == 0.4


def test_cold_start_uses_priors():
    model = ThermalModel(volume_m3=30.0)
    assert model.k(False) == K_PRIOR
    assert model.q_hat(12.0) == 0.0
    assert model.cop_effective == COP_PRIOR
    # Prediction works out of the box.
    traj = model.predict_free(25.0, [30.0] * 24, start_hour=12.0)
    assert len(traj) == 25
    assert traj[-1] > traj[0]  # drifts toward hot outdoors


def test_prior_blends_into_fit():
    model = ThermalModel(volume_m3=30.0)
    # A handful of samples should still be dominated by the prior.
    for _ in range(5):
        model.update_free(22.0, 22.05, 30.0, STEP_H, 12.0)
    assert abs(model.k(False) - K_PRIOR) < 0.15


def test_free_float_attractor_above_outdoor_with_gains():
    room = SimRoom(name="bed", k=0.3, solar_amplitude=0.4, internal_gain=0.1)
    house = SimHouse(rooms=[room])
    model = ThermalModel(room.volume_m3)
    run_free_response(model, room, house, hours=72.0)
    # At mid-afternoon the attractor exceeds outdoor temperature.
    assert model.t_eq(25.0, 14.0) > 25.0


def test_cop_estimation_quasi_steady():
    # Steady cooling: room held at fixed temp, known electrical draw.
    model = ThermalModel(volume_m3=30.0)
    model.fits[False].theta[0] = 0.4
    model.fits[False].samples = 1000
    t_in, t_out, p_head = 23.0, 33.0, 500.0
    for _ in range(100):
        model.update_cop(p_head, t_in, t_out, local_hour=15.0)
    # Q = UA * 10 K = 0.34*0.4*30*10 = 40.8 W -> tiny COP, rejected (out of bounds)
    assert model.cop is None or model.cop >= 0.5


def test_diurnal_model_forecast():
    from custom_components.adaptive_comfort.core.simulator import sinusoidal_outdoor

    model = DiurnalModel()
    for step in range(24 * 12):
        hour = (step * 5.0 / 60.0) % 24.0
        model.update(hour, sinusoidal_outdoor(hour))
    prediction = model.predict(15.0)
    assert prediction is not None
    assert abs(prediction - sinusoidal_outdoor(15.0)) < 0.5


def test_thermal_roundtrip_persistence():
    model = ThermalModel(volume_m3=25.0)
    model.fits[False].theta[0] = 0.6
    model.fits[False].samples = 50
    model.cop = 3.3
    restored = ThermalModel.from_dict(model.to_dict(), 25.0)
    assert restored.fits[False].theta[0] == 0.6
    assert restored.cop == 3.3
