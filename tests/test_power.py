from custom_components.adaptive_comfort.core import power


def test_compose_load_with_battery_discharge():
    # Grid shows 200 W but battery supplies 800 W: real load is 1000 W.
    assert power.compose_load(200.0, 800.0, True) == 1000.0
    # Charging subtracts — it is not AC.
    assert power.compose_load(1200.0, -500.0, True) == 700.0
    # Inverted sign convention (positive = charging): discharge is -p_battery.
    assert power.compose_load(200.0, -800.0, False) == 1000.0
    assert power.compose_load(1050.0, 800.0, False) == 250.0


def test_compose_load_battery_also_in_known_loads_double_subtracts():
    """Footgun: the battery entity must not also be a known-load entity."""
    once = power.compose_load(1050.0, 800.0, False)
    twice = power.compose_load(1050.0, 800.0, False, [800.0])
    assert once == 250.0
    assert twice == 0.0


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
    start = power.SHED_DEFAULT_START_PCT
    assert not power.shed_needed(3000.0, limit, start, None)
    assert not power.shed_needed(3300.0, limit, start, 3.0)  # not sustained
    assert not power.shed_needed(3300.0, limit, start, 6.0)  # warning band needs 30 s
    assert power.shed_needed(3300.0, limit, start, 30.0)
    assert power.shed_needed(3400.0, limit, start, None, urgent=True)
    assert power.shed_needed(3320.0, limit, start, None)  # >= 96% critical
    assert power.restore_allowed(1500.0, limit, 0.75, 700.0)
    assert not power.restore_allowed(2900.0, limit, 0.75, 700.0)
    assert not power.restore_allowed(2500.0, limit, 0.75, 900.0)


def test_contracted_demand_uses_known_loads_when_meter_lags():
    demand = power.contracted_demand_w(
        p_grid=600.0,
        known_loads=[2300.0, 50.0],
        active_ac_w=500.0,
        p_grid_peak=2900.0,
    )
    assert demand == 2900.0
    assert demand >= 2850.0


def test_contracted_demand_discharge_follows_import():
    """Discharge must not inflate the lag term; restore stays on honest demand."""
    demand = power.contracted_demand_w(
        p_grid=200.0,
        known_loads=[100.0],
        active_ac_w=250.0,
        p_grid_peak=200.0,
        discharge_w=800.0,
    )
    assert demand == 200.0
    assert power.restore_allowed(demand, 3450.0, 0.75, 1000.0)


def test_contracted_demand_charge_counts_on_grid():
    demand = power.contracted_demand_w(
        p_grid=1050.0,
        known_loads=[],
        active_ac_w=250.0,
        discharge_w=-800.0,
    )
    assert demand == 1050.0


def test_baseline_schema_wipes_charge_inclusive_slots():
    dirty = power.BaselineModel()
    dirty.update(23.5, 900.0)
    wiped = power.BaselineModel.from_dict({"slots": list(dirty.slots)})
    assert wiped.value(23.5, fallback=False) is None
    kept = power.BaselineModel.from_dict(dirty.to_dict())
    assert kept.value(23.5, fallback=False) == dirty.value(23.5, fallback=False)


def test_energy_merge_max_duplicate_heads_not_watts():
    merge = power.EnergyMerge(power.ENERGY_MERGE_MAX)
    merge.update({"a": 0.0, "b": 0.0})
    delta = merge.update({"a": 250.0, "b": 250.0})
    assert delta == 250.0
    assert merge.house_wh == 250.0
    assert delta != 15_000.0


def test_energy_merge_sum_partitioned():
    merge = power.EnergyMerge(power.ENERGY_MERGE_SUM)
    merge.update({"a": 0.0, "b": 0.0})
    assert merge.update({"a": 100.0, "b": 150.0}) == 250.0


def test_energy_merge_ignores_negative_and_reset():
    merge = power.EnergyMerge()
    merge.update({"a": 500.0})
    assert merge.update({"a": -1.0}) == 0.0
    assert merge.last_wh["a"] == 500.0
    assert merge.update({"a": 0.0}) == 0.0
    assert merge.last_wh["a"] == 0.0
    assert merge.update({"a": 250.0}) == 250.0
