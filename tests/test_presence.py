"""Tests for house/zone presence resolution."""

from __future__ import annotations

from custom_components.adaptive_comfort.presence import house_presence, presence_state


class _State:
    def __init__(self, state: str) -> None:
        self.state = state


class _Hass:
    def __init__(self, states: dict[str, _State | None]) -> None:
        self._states = states

    class states:
        _map: dict[str, _State | None]

        @classmethod
        def get(cls, entity_id: str) -> _State | None:
            return cls._map.get(entity_id)


def _hass(states: dict[str, str | None]) -> _Hass:
    mapped = {k: _State(v) if v is not None else None for k, v in states.items()}
    _Hass.states._map = mapped  # type: ignore[attr-defined]
    return _Hass(mapped)


def test_person_home_and_not_home():
    hass = _hass({"person.ricardo": "home"})
    assert presence_state(hass, "person.ricardo") is True
    hass = _hass({"person.ricardo": "not_home"})
    assert presence_state(hass, "person.ricardo") is False


def test_house_presence_uses_person_when_available():
    hass = _hass({"person.ricardo": "home", "binary_sensor.bed": "off"})
    assert house_presence(hass, "person.ricardo", ["binary_sensor.bed"]) is True


def test_house_presence_falls_back_to_zones_when_person_unavailable():
    hass = _hass({"person.ricardo": None, "binary_sensor.bed": "on"})
    assert house_presence(hass, "person.ricardo", ["binary_sensor.bed"]) is True


def test_house_presence_fallback_all_zones_off():
    hass = _hass(
        {
            "person.ricardo": "unavailable",
            "binary_sensor.bed": "off",
            "binary_sensor.living": "off",
        }
    )
    assert house_presence(hass, "person.ricardo", ["binary_sensor.bed", "binary_sensor.living"]) is False


def test_house_presence_no_sensors():
    hass = _hass({})
    assert house_presence(hass, None, []) is None
