"""Binary sensors: shedding, per-zone conditioning state."""

from __future__ import annotations

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import DOMAIN
from .coordinator import AdaptiveComfortRuntime
from .entity import AdaptiveComfortEntity, AdaptiveComfortZoneEntity


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    runtime: AdaptiveComfortRuntime = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([SheddingActiveSensor(runtime), WindowSuggestionSensor(runtime)])
    for zone in runtime.zones.values():
        async_add_entities(
            [
                ZoneConditioningSensor(runtime, zone),
                ZoneParkedSensor(runtime, zone),
                ZoneShedSensor(runtime, zone),
                ZoneWindowSuggestionSensor(runtime, zone),
            ],
            config_subentry_id=zone.config.zone_id,
        )


class SheddingActiveSensor(AdaptiveComfortEntity, BinarySensorEntity):
    _attr_translation_key = "shedding_active"
    _attr_device_class = BinarySensorDeviceClass.PROBLEM

    def __init__(self, runtime: AdaptiveComfortRuntime) -> None:
        super().__init__(runtime, "shedding_active")

    @property
    def is_on(self) -> bool:
        return self.runtime.shedding_active

    @property
    def extra_state_attributes(self) -> dict:
        shed = self.runtime.controller_state.shed
        names = {
            zid: self.runtime.zones[zid].config.name for zid in shed if zid in self.runtime.zones
        }
        return {"shed_zones": list(names.values())}


class ZoneConditioningSensor(AdaptiveComfortZoneEntity, BinarySensorEntity):
    _attr_translation_key = "conditioning_active"
    _attr_device_class = BinarySensorDeviceClass.RUNNING

    def __init__(self, runtime: AdaptiveComfortRuntime, zone) -> None:
        super().__init__(runtime, zone, "conditioning_active")

    @property
    def is_on(self) -> bool:
        return self.zone.is_on

    @property
    def extra_state_attributes(self) -> dict:
        return {"head_state": self.zone.head_state}


class ZoneParkedSensor(AdaptiveComfortZoneEntity, BinarySensorEntity):
    """True while the controller is holding the zone in a parked setpoint."""

    _attr_translation_key = "zone_parked"
    _attr_device_class = BinarySensorDeviceClass.RUNNING
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, runtime: AdaptiveComfortRuntime, zone) -> None:
        super().__init__(runtime, zone, "parked")

    @property
    def is_on(self) -> bool:
        return self.zone.config.zone_id in self.runtime.controller_state.zone_parked_since

    @property
    def extra_state_attributes(self) -> dict:
        zid = self.zone.config.zone_id
        st = self.runtime.controller_state
        return {
            "margin_k": st.zone_park_margin.get(zid),
            "preferred_margin_k": st.zone_park_preferred.get(
                zid, self.zone.park.preferred_margin_k
            ),
            "classification": self.zone.park.classification,
        }


class ZoneShedSensor(AdaptiveComfortZoneEntity, BinarySensorEntity):
    _attr_translation_key = "zone_shed"
    _attr_device_class = BinarySensorDeviceClass.PROBLEM
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, runtime: AdaptiveComfortRuntime, zone) -> None:
        super().__init__(runtime, zone, "shed")

    @property
    def is_on(self) -> bool:
        return self.zone.config.zone_id in self.runtime.controller_state.shed


class WindowSuggestionSensor(AdaptiveComfortEntity, BinarySensorEntity):
    """On when opening a window somewhere would beat mechanical conditioning."""

    _attr_translation_key = "window_suggestion"

    def __init__(self, runtime: AdaptiveComfortRuntime) -> None:
        super().__init__(runtime, "window_suggestion")

    @property
    def is_on(self) -> bool:
        return bool(self.runtime.window_suggestions)

    @property
    def extra_state_attributes(self) -> dict:
        names = [
            self.runtime.zones[zid].config.name
            for zid in self.runtime.window_suggestions
            if zid in self.runtime.zones
        ]
        return {"zones": names}


class ZoneWindowSuggestionSensor(AdaptiveComfortZoneEntity, BinarySensorEntity):
    _attr_translation_key = "zone_window_suggestion"

    def __init__(self, runtime: AdaptiveComfortRuntime, zone) -> None:
        super().__init__(runtime, zone, "window_suggestion")

    @property
    def is_on(self) -> bool:
        return self.zone.config.zone_id in self.runtime.window_suggestions
