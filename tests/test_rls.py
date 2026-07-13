import random

from custom_components.adaptive_comfort.core.rls import RLS


def test_rls_recovers_linear_model():
    rng = random.Random(1)
    true_theta = [2.0, -3.0, 1.0]
    rls = RLS(3, lam=1.0)
    for _ in range(500):
        phi = [rng.uniform(-1, 1), rng.uniform(-1, 1), 1.0]
        y = sum(t * x for t, x in zip(true_theta, phi, strict=True)) + rng.gauss(0, 0.01)
        rls.update(phi, y)
    for est, true in zip(rls.theta, true_theta, strict=True):
        assert abs(est - true) < 0.05


def test_rls_forgetting_tracks_drift():
    rng = random.Random(2)
    rls = RLS(1, lam=0.98)
    for _ in range(300):
        rls.update([1.0], 5.0 + rng.gauss(0, 0.01))
    assert abs(rls.theta[0] - 5.0) < 0.1
    for _ in range(300):
        rls.update([1.0], 7.0 + rng.gauss(0, 0.01))
    assert abs(rls.theta[0] - 7.0) < 0.1


def test_rls_roundtrip_persistence():
    rls = RLS(2)
    rls.update([1.0, 0.5], 2.0)
    restored = RLS.from_dict(rls.to_dict(), 2)
    assert restored.theta == rls.theta
    assert restored.samples == rls.samples
