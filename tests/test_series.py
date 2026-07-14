"""TimeSeries ring buffer."""

from custom_components.adaptive_comfort.core.series import TimeSeries


def test_value_at_returns_scalar_not_tuple():
    s = TimeSeries()
    s.append(100.0, 0.010)
    s.append(400.0, 0.012)
    earlier = s.value_at(400.0 - 300.0)
    assert isinstance(earlier, float)
    assert earlier == 0.010


def test_value_at_none_when_too_old():
    s = TimeSeries()
    s.append(100.0, 1.0)
    assert s.value_at(1000.0, max_age_s=60.0) is None
