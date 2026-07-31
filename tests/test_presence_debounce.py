"""Occupancy debounce: mmWave flicker must not buy a helper run."""

from custom_components.adaptive_comfort.presence import (
    OCCUPIED_STICKY_S,
    VACANCY_DWELL_S,
    OccupancyDebounce,
)


def test_occupied_rising_edge_is_immediate():
    d = OccupancyDebounce()
    assert d.update(True, 1000.0) is True


def test_flicker_false_stays_occupied_inside_sticky():
    d = OccupancyDebounce()
    d.update(True, 1000.0)
    # Brief vacant blip well inside sticky window.
    assert d.update(False, 1000.0 + OCCUPIED_STICKY_S / 2) is True


def test_vacancy_requires_dwell_after_sticky():
    d = OccupancyDebounce()
    d.update(True, 1000.0)
    t = 1000.0 + OCCUPIED_STICKY_S + 1.0
    assert d.update(False, t) is True  # dwell not yet elapsed
    assert d.update(False, t + VACANCY_DWELL_S) is False


def test_unknown_holds_last_effective():
    d = OccupancyDebounce()
    d.update(True, 1000.0)
    assert d.update(None, 1100.0) is True


def test_roundtrip():
    d = OccupancyDebounce()
    d.update(True, 1000.0)
    d.update(False, 1000.0 + OCCUPIED_STICKY_S + VACANCY_DWELL_S)
    restored = OccupancyDebounce.from_dict(d.to_dict())
    assert restored.effective is False
