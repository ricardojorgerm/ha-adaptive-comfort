"""Ventilation fan state helpers."""

from __future__ import annotations

from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import HomeAssistant

OFF_STATES = frozenset({"off", "idle", "closed", "0", "false"})


def fan_entities_on(hass: HomeAssistant, entity_ids: tuple[str, ...]) -> bool:
    """True when any configured fan/switch vent entity is actively on."""
    for entity_id in entity_ids:
        state = hass.states.get(entity_id)
        if state is None or state.state in (STATE_UNKNOWN, STATE_UNAVAILABLE):
            continue
        if state.state.lower() not in OFF_STATES:
            return True
    return False
