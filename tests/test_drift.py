from custom_components.adaptive_comfort.core.drift import DriftEstimator
from custom_components.adaptive_comfort.core.types import STATE_COOLING, STATE_STANDBY


def test_learns_per_state_offsets():
    est = DriftEstimator(alpha=0.2)
    # Idle head reads 1.5 K high (electronics), running head reads true.
    for _ in range(50):
        est.update(STATE_STANDBY, 23.5, 22.0)
        est.update(STATE_COOLING, 22.05, 22.0)
    assert abs(est.offset(STATE_STANDBY) - 1.5) < 0.1
    assert abs(est.offset(STATE_COOLING) - 0.05) < 0.1
    assert abs(est.correct(23.5, STATE_STANDBY) - 22.0) < 0.1


def test_unknown_state_falls_back_to_mean():
    est = DriftEstimator()
    est.update(STATE_STANDBY, 23.0, 22.0)
    assert est.offset("fan_only") == est.offset(STATE_STANDBY)


def test_no_data_returns_zero_offset():
    est = DriftEstimator()
    assert est.offset(STATE_STANDBY) == 0.0
    assert est.correct(22.0, STATE_STANDBY) == 22.0


def test_roundtrip_persistence():
    est = DriftEstimator()
    est.update(STATE_STANDBY, 23.0, 22.0)
    restored = DriftEstimator.from_dict(est.to_dict())
    assert restored.offsets == est.offsets
