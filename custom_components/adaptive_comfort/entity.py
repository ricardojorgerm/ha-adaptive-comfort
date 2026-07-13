"""Shared entity bases: hub device and per-zone devices."""

from __future__ import annotations

from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity import Entity

from .const import DOMAIN, SIGNAL_UPDATE
from .coordinator import AdaptiveComfortRuntime, ZoneRuntime


class AdaptiveComfortEntity(Entity):
    _attr_should_poll = False
    _attr_has_entity_name = True

    def __init__(self, runtime: AdaptiveComfortRuntime, suffix: str) -> None:
        self.runtime = runtime
        self._attr_unique_id = f"{runtime.entry.entry_id}_{suffix}"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, runtime.entry.entry_id)},
            name="Adaptive Comfort House",
            manufacturer="Adaptive Comfort",
            model="Whole-home controller",
        )

    async def async_added_to_hass(self) -> None:
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                f"{SIGNAL_UPDATE}_{self.runtime.entry.entry_id}",
                self.async_write_ha_state,
            )
        )


class AdaptiveComfortZoneEntity(AdaptiveComfortEntity):
    def __init__(self, runtime: AdaptiveComfortRuntime, zone: ZoneRuntime, suffix: str) -> None:
        super().__init__(runtime, f"{zone.config.zone_id}_{suffix}")
        self.zone = zone
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, zone.config.zone_id)},
            name=zone.config.name,
            manufacturer="Adaptive Comfort",
            model="Climate zone",
            via_device=(DOMAIN, runtime.entry.entry_id),
        )
