"""House and zone presence resolution."""

from __future__ import annotations

from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import HomeAssistant

HOME_STATES = frozenset({"home", "on"})
AWAY_STATES = frozenset({"not_home", "off", "away"})


def presence_state(hass: HomeAssistant, entity_id: str | None) -> bool | None:
    """Return True/False when occupied/unoccupied, or None when unknown."""
    if not entity_id:
        return None
    state = hass.states.get(entity_id)
    if state is None or state.state in (STATE_UNKNOWN, STATE_UNAVAILABLE):
        return None
    domain = entity_id.split(".", 1)[0]
    if domain == "zone":
        try:
            return float(state.state) > 0
        except (ValueError, TypeError):
            return None
    lowered = state.state.lower()
    if lowered in HOME_STATES:
        return True
    if lowered in AWAY_STATES:
        return False
    return None


def configured_presence_available(hass: HomeAssistant, entity_id: str | None) -> bool:
    """True when a configured house-presence entity is reporting a usable state."""
    if not entity_id:
        return False
    state = hass.states.get(entity_id)
    return state is not None and state.state not in (STATE_UNKNOWN, STATE_UNAVAILABLE)


def house_presence(
    hass: HomeAssistant,
    entity_id: str | None,
    zone_presence_sensors: list[str | None],
) -> bool | None:
    """Whole-home presence from a configured entity, with zone-sensor fallback.

    Accepts person, device_tracker, binary_sensor, group, or zone entities.
    When the configured entity is missing or unavailable, derive presence from
    zone presence sensors: any zone occupied -> home; all explicitly vacant -> away.
    """
    if configured_presence_available(hass, entity_id):
        result = presence_state(hass, entity_id)
        if result is not None:
            return result

    zone_values = [presence_state(hass, sensor) for sensor in zone_presence_sensors if sensor]
    if not zone_values:
        return None
    if any(value is True for value in zone_values):
        return True
    if all(value is False for value in zone_values):
        return False
    return None
