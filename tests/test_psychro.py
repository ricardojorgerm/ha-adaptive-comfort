from custom_components.adaptive_comfort.core import psychro


def test_saturation_pressure_magnus():
    # ~2.34 kPa at 20 C
    assert abs(psychro.saturation_pressure_pa(20.0) - 2333) < 30


def test_humidity_ratio_typical_room():
    w = psychro.humidity_ratio(20.0, 50.0)
    assert 0.006 < w < 0.008


def test_latent_power_for_one_kg_per_hour():
    assert abs(psychro.latent_power_w(1.0) - 680.0) < 1e-6


def test_moisture_removal_balance():
    # Zone drying out (dw/dt < 0) with matched in/out humidity and no
    # sources: everything removed comes from storage.
    removed = psychro.moisture_removal_kg_h(
        volume_m3=30.0,
        airflow_m3h=6.0,
        w_in=0.008,
        w_out=0.008,
        dw_in_dt_per_h=-0.001,
        sources_kg_h=0.0,
    )
    expected = psychro.AIR_DENSITY_KG_M3 * 30.0 * 0.001
    assert abs(removed - expected) < 1e-9


def test_moisture_removal_never_negative():
    removed = psychro.moisture_removal_kg_h(30.0, 6.0, 0.006, 0.010, 0.01, 0.0)
    assert removed == 0.0
