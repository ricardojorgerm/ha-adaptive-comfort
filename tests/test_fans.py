"""Tests for ventilation fan helpers."""

from __future__ import annotations

from custom_components.adaptive_comfort.fans import fan_entities_on


class _State:
    def __init__(self, state: str) -> None:
        self.state = state


class _Hass:
    class states:
        _map: dict[str, _State | None]

        @classmethod
        def get(cls, entity_id: str) -> _State | None:
            return cls._map.get(entity_id)


def _hass(states: dict[str, str | None]) -> _Hass:
    _Hass.states._map = {k: _State(v) if v is not None else None for k, v in states.items()}  # type: ignore[attr-defined]
    return _Hass()


def test_fan_entities_on_any_active():
    hass = _hass({"fan.bath": "off", "switch.hood": "on"})
    assert fan_entities_on(hass, ("fan.bath", "switch.hood")) is True


def test_fan_entities_on_all_off():
    hass = _hass({"fan.bath": "off"})
    assert fan_entities_on(hass, ("fan.bath",)) is False


def test_fan_entities_on_empty():
    hass = _hass({})
    assert fan_entities_on(hass, ()) is False
