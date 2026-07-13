from custom_components.adaptive_comfort.core import power


def test_compose_load_with_battery_discharge():
    # Grid shows 200 W but battery supplies 800 W: real load is 1000 W.
    assert power.compose_load(200.0, 800.0, True) == 1000.0
    # Charging battery does not reduce the measured load.
    assert power.compose_load(1200.0, -500.0, True) == 1200.0
    # Inverted sign convention.
    assert power.compose_load(200.0, -800.0, False) == 1000.0


def test_compose_load_subtracts_known_loads():
    assert power.compose_load(1500.0, None, True, [300.0, 200.0]) == 1000.0


def test_baseline_slots_and_fallback():
    model = power.BaselineModel(alpha=0.5)
    model.update(8.25, 300.0)
    assert model.value(8.4) == 300.0
    # Unknown slot falls back to the median of known slots.
    assert model.value(20.0) == 300.0


def test_measure_step():
    pre = [200.0, 210.0, 205.0]
    post = [900.0, 910.0, 905.0]
    assert abs(power.measure_step(pre, post) - 700.0) < 1e-9
    assert power.measure_step([200.0], post) is None


def test_draw_estimator_rejects_outliers():
    est = power.DrawEstimator()
    for _ in range(10):
        assert est.add_event("z1", "cool", 700.0)
    assert not est.add_event("z1", "cool", 3500.0)
    assert abs(est.draw_w("z1", "cool") - 700.0) < 1e-9
    # Other-mode fallback.
    assert est.draw_w("z1", "heat") == 700.0


def test_estimate_and_allocate():
    assert power.estimate_ac_power(1500.0, 400.0) == 1100.0
    assert power.estimate_ac_power(300.0, 400.0) == 0.0
    alloc = power.allocate_power(1000.0, [("a", 700.0, 2.0), ("b", 700.0, 1.0)])
    assert abs(alloc["a"] - 666.67) < 1.0
    assert abs(sum(alloc.values()) - 1000.0) < 1e-6


def test_shed_thresholds_for_345_kva():
    limit = 3.45 * 1000.0  # the contracted 3.45 kVA -> 3450 W
    assert not power.shed_needed(3000.0, limit, 0.92, None)
    assert not power.shed_needed(3300.0, limit, 0.92, 5.0)  # not sustained
    assert power.shed_needed(3300.0, limit, 0.92, 20.0)
    assert power.restore_allowed(1500.0, limit, 0.75, 700.0)
    assert not power.restore_allowed(2900.0, limit, 0.75, 700.0)
    assert not power.restore_allowed(2500.0, limit, 0.75, 900.0)
