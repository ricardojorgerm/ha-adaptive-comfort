"""Thermal model fan mixing boosts."""

from custom_components.adaptive_comfort.core.thermal import ThermalModel


def test_outdoor_exhaust_increases_modeled_outdoor_loss():
    model = ThermalModel(volume_m3=30.0)
    base = model.sensible_power_w(24.0, 20.0, 0.0, 12.0, t_house_other=23.0)
    boosted = model.sensible_power_w(
        24.0,
        20.0,
        0.0,
        12.0,
        t_house_other=23.0,
        outdoor_exhaust_on=True,
    )
    assert abs(boosted) > abs(base)


def test_indoor_fans_increase_modeled_house_mixing_loss():
    model = ThermalModel(volume_m3=30.0)
    base = model.sensible_power_w(24.0, 26.0, 0.0, 12.0, t_house_other=22.0)
    boosted = model.sensible_power_w(
        24.0,
        26.0,
        0.0,
        12.0,
        t_house_other=22.0,
        indoor_fans_on=True,
    )
    assert abs(boosted) > abs(base)
