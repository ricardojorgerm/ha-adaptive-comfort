"""The house-wide climate entity."""

from __future__ import annotations

from typing import Any, ClassVar

from homeassistant.components.climate import (
    ClimateEntity,
    ClimateEntityFeature,
    HVACAction,
    HVACMode,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_TEMPERATURE, UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import DOMAIN
from .coordinator import AdaptiveComfortRuntime
from .core.types import (
    MODE_AUTO,
    MODE_COOL,
    MODE_HEAT,
    MODE_OFF,
    PRESET_AWAY,
    PRESET_BOOST,
    PRESET_ECO,
    PRESET_NONE,
)
from .entity import AdaptiveComfortEntity

HVAC_TO_CORE = {
    HVACMode.OFF: MODE_OFF,
    HVACMode.HEAT: MODE_HEAT,
    HVACMode.COOL: MODE_COOL,
    HVACMode.AUTO: MODE_AUTO,
}
CORE_TO_HVAC = {v: k for k, v in HVAC_TO_CORE.items()}


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    runtime: AdaptiveComfortRuntime = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([AdaptiveComfortClimate(runtime)])


class AdaptiveComfortClimate(AdaptiveComfortEntity, ClimateEntity):
    _attr_name = None  # take the device name
    _attr_temperature_unit = UnitOfTemperature.CELSIUS
    _attr_hvac_modes: ClassVar[list[HVACMode]] = [
        HVACMode.OFF,
        HVACMode.HEAT,
        HVACMode.COOL,
        HVACMode.AUTO,
    ]
    _attr_preset_modes: ClassVar[list[str]] = [
        PRESET_NONE,
        PRESET_ECO,
        PRESET_AWAY,
        PRESET_BOOST,
    ]
    _attr_supported_features = (
        ClimateEntityFeature.TARGET_TEMPERATURE
        | ClimateEntityFeature.PRESET_MODE
        | ClimateEntityFeature.TURN_OFF
        | ClimateEntityFeature.TURN_ON
    )
    _attr_min_temp = 16.0
    _attr_max_temp = 28.0
    _attr_target_temperature_step = 0.5

    def __init__(self, runtime: AdaptiveComfortRuntime) -> None:
        super().__init__(runtime, "house_climate")

    @property
    def hvac_mode(self) -> HVACMode:
        return CORE_TO_HVAC.get(self.runtime.settings.hvac_mode, HVACMode.AUTO)

    @property
    def hvac_action(self) -> HVACAction:
        if self.runtime.settings.hvac_mode == MODE_OFF:
            return HVACAction.OFF
        active = self.runtime.controller_state.mode
        any_on = any(z.is_on for z in self.runtime.zones.values())
        if not any_on:
            return HVACAction.IDLE
        if active == MODE_HEAT:
            return HVACAction.HEATING
        if active == MODE_COOL:
            return HVACAction.COOLING
        return HVACAction.IDLE

    @property
    def target_temperature(self) -> float:
        return self.runtime.settings.target

    @property
    def current_temperature(self) -> float | None:
        temps = [z.temp for z in self.runtime.zones.values() if z.temp is not None]
        if not temps:
            return None
        return round(sum(temps) / len(temps), 2)

    @property
    def preset_mode(self) -> str:
        return self.runtime.settings.preset

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        return {
            "dominant_mode": self.runtime.controller_state.mode,
            "mode_source": self.runtime.mode_source,
            "shedding_active": self.runtime.shedding_active,
        }

    async def async_set_temperature(self, **kwargs: Any) -> None:
        if (temp := kwargs.get(ATTR_TEMPERATURE)) is not None:
            self.runtime.settings.target = float(temp)
            self.runtime.request_save()
            self.runtime.notify()

    async def async_set_hvac_mode(self, hvac_mode: HVACMode) -> None:
        self.runtime.settings.hvac_mode = HVAC_TO_CORE.get(hvac_mode, MODE_AUTO)
        self.runtime.request_save()
        self.runtime.notify()

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        if preset_mode in self._attr_preset_modes:
            self.runtime.settings.preset = preset_mode
            self.runtime.request_save()
            self.runtime.notify()

    async def async_turn_off(self) -> None:
        await self.async_set_hvac_mode(HVACMode.OFF)

    async def async_turn_on(self) -> None:
        await self.async_set_hvac_mode(HVACMode.AUTO)
