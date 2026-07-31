"""House and zone presence resolution."""

from __future__ import annotations

from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import HomeAssistant

HOME_STATES = frozenset({"home", "on"})
AWAY_STATES = frozenset({"not_home", "off", "away"})
# Sticky occupied / vacancy dwell so mmWave flicker cannot buy a min_on helper.
OCCUPIED_STICKY_S = 120.0
VACANCY_DWELL_S = 180.0


class OccupancyDebounce:
    """Debounce raw presence into sticky occupied / delayed vacant.

    - Rising edge (vacant→occupied) is immediate.
    - Occupied holds for OCCUPIED_STICKY_S after the last True sample.
    - Vacant only reports after VACANCY_DWELL_S of continuous False.
    - None (unknown) does not change the effective state.
    """

    def __init__(
        self,
        sticky_s: float = OCCUPIED_STICKY_S,
        vacancy_s: float = VACANCY_DWELL_S,
    ) -> None:
        self.sticky_s = sticky_s
        self.vacancy_s = vacancy_s
        self._effective: bool | None = None
        self._last_true_ts: float | None = None
        self._false_since: float | None = None

    @property
    def effective(self) -> bool | None:
        return self._effective

    def update(self, raw: bool | None, now: float) -> bool | None:
        if raw is True:
            self._last_true_ts = now
            self._false_since = None
            self._effective = True
            return self._effective
        if raw is False:
            if self._effective is True:
                idle = (
                    0.0
                    if self._last_true_ts is None
                    else now - self._last_true_ts
                )
                # Still inside sticky window after last True.
                if idle < self.sticky_s:
                    return self._effective
                # Large gap since last True already covers sticky + dwell.
                if idle >= self.sticky_s + self.vacancy_s:
                    self._false_since = now
                    self._effective = False
                    return self._effective
                if self._false_since is None:
                    self._false_since = now
                if now - self._false_since >= self.vacancy_s:
                    self._effective = False
                return self._effective
            if self._effective is None:
                if self._false_since is None:
                    self._false_since = now
                if now - self._false_since >= self.vacancy_s:
                    self._effective = False
            return self._effective
        # Unknown: hold.
        return self._effective

    def to_dict(self) -> dict:
        return {
            "effective": self._effective,
            "last_true_ts": self._last_true_ts,
            "false_since": self._false_since,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> OccupancyDebounce:
        d = cls()
        if not data:
            return d
        eff = data.get("effective")
        d._effective = None if eff is None else bool(eff)
        lt = data.get("last_true_ts")
        d._last_true_ts = None if lt is None else float(lt)
        fs = data.get("false_since")
        d._false_since = None if fs is None else float(fs)
        return d


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
