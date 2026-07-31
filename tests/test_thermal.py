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
    # UA is fit-consistent: C_eff · k (default furniture_factor = 4).
    assert abs(model.c_air_wh_per_k - 0.34 * 30.0) < 1e-9
    assert abs(model.ua_w_per_k - model.c_eff_wh_per_k * 0.5) < 1e-9
    assert abs(model.ua_w_per_k - 0.34 * 15.0 * model.furniture_factor) < 1e-9


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
    assert model.q_coeffs(False) is None
    diag = model.disturbance_diag(12.0)
    assert diag["q_hat_k_per_h"] == 0.0
    assert diag["coeffs_closed"] is None


def test_disturbance_coeffs_from_fit():
    model = ThermalModel(volume_m3=30.0)
    model.fits[False].theta = [0.3, 0.1, 0.05, 0.02, -0.01, 0.0, 0.0]
    model.fits[False].samples = 100
    coeffs = model.q_coeffs(False)
    assert coeffs is not None
    assert coeffs["a0"] == 0.05
    assert coeffs["a1"] == 0.02
    assert coeffs["b1"] == -0.01
    assert coeffs["samples"] == 100
    diag = model.disturbance_diag(0.0, door_open=False)
    # At hour 0: cos=1, sin=0 → q = a0 + a1 + a2
    assert abs(diag["q_hat_k_per_h"] - 0.07) < 1e-4
    assert diag["coeffs_closed"]["a0"] == 0.05
    assert diag["coeffs_open"] is None


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
    # Steady hold: Q = C_eff · k · ΔT = 40.8·0.4·10 ≈ 163 W → COP ~0.33, rejected.
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


def _cool_capable_model() -> ThermalModel:
    model = ThermalModel(volume_m3=30.0)
    model.fits[False].theta[0] = 0.4
    model.fits[False].samples = 1000
    return model


def test_update_cop_accepts_transient_samples():
    """Small rooms on short cycles are never quasi-steady while conditioning;
    the storage term C*dT/dt makes the transient sample valid (fixes zones
    that reported cop=None forever)."""
    model = _cool_capable_model()
    assert model.cop is None
    # Pulling down at 3 K/h while drawing 100 W: |q| = C*3 + UA*dT is
    # dominated by the storage term and lands inside the plausible COP band.
    model.update_cop(100.0, 24.0, 30.0, 12.0, dtdt_per_h=-3.0)
    assert model.cop is not None and model.cop > 0.0


def test_fit_is_fallback_marks_unlearned_regime():
    model = _cool_capable_model()  # only the door_closed regime is fitted
    assert model.fit_is_fallback(True) is True
    assert model.fit_is_fallback(False) is False


def test_update_cop_includes_latent():
    """Sensible-only heat flow undercounts delivered cooling by the latent
    share; here the sensible-only sample even falls below COP_MIN and is
    rejected (the field bug: implausibly low zone COPs), while the
    latent-inclusive sample lands in the plausible band."""
    a = _cool_capable_model()
    b = _cool_capable_model()
    # |Q_sens| ~ C_eff*|-3 - 0.4*6| ~ 220 W; at 500 W electric -> COP ~0.44 < COP_MIN.
    a.update_cop(500.0, 24.0, 30.0, 12.0, dtdt_per_h=-3.0)
    b.update_cop(500.0, 24.0, 30.0, 12.0, dtdt_per_h=-3.0, latent_w=200.0)
    assert a.cop is None  # sensible-only below COP_MIN
    assert b.cop is not None and b.cop > 0.8  # latent counted as delivered heat


def test_sensible_power_matches_c_eff_times_excess_rate():
    """Watt side must agree with the rate model predict_free integrates."""
    model = _cool_capable_model()
    model.furniture_factor = 4.0
    t_in, t_out, t_house, hour = 24.0, 30.0, 22.0, 12.0
    model.fits[False].theta[1] = 0.3  # k_mix
    dtdt = -1.5
    ff = model.free_float_rate(t_in, t_out, hour, t_house_other=t_house)
    q = model.sensible_power_w(t_in, t_out, dtdt, hour, t_house_other=t_house)
    assert abs(q - model.c_eff_wh_per_k * (dtdt - ff)) < 1e-9
    # Standing load (dtdt=0): AC must cancel free-float warming.
    stand = -model.sensible_power_w(t_in, t_out, 0.0, hour, t_house_other=t_house)
    assert abs(stand - model.c_eff_wh_per_k * ff) < 1e-9
    assert abs(model.ua_mix_w_per_k - model.c_eff_wh_per_k * 0.3) < 1e-9


def test_update_c_eff_recovers_furniture_from_excess_rate():
    """Furniture learning uses Q / excess_rate — not circular UA(=C·k)."""
    model = ThermalModel(volume_m3=30.0)
    model.fits[False].theta[0] = 0.3
    model.fits[False].theta[1] = 0.0
    model.fits[False].samples = 1000
    model.furniture_factor = 2.0
    model.cop = 3.0
    model.cop_samples = 20
    # True C_eff = 0.34*30*5 = 51; cool with known Q and matching dtdt.
    true_ff = 5.0  # furniture
    c_air = 0.34 * 30.0
    c_true = c_air * true_ff
    q_hvac = -3.0 * 600.0  # COP*P, cooling
    # q_hvac = C * (dtdt - ff_rate); ff_rate = k*ΔT = 0.3*10 = 3 at these temps
    t_in, t_out = 23.0, 33.0
    ff_rate = 0.3 * (t_out - t_in)
    dtdt = q_hvac / c_true + ff_rate
    for _ in range(80):
        model.update_c_eff(600.0, False, t_in, t_out, dtdt, 12.0)
    assert abs(model.furniture_factor - true_ff) < 0.35
