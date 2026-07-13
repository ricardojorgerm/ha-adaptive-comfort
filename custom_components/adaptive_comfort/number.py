"""Configuration numbers for house tunables and per-zone comfort offsets."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from homeassistant.components.number import NumberEntity, NumberEntityDescription, NumberMode
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import DOMAIN
from .coordinator import AdaptiveComfortRuntime
from .core.types import Settings
from .entity import AdaptiveComfortEntity, AdaptiveComfortZoneEntity


@dataclass(frozen=True, kw_only=True)
class SettingNumberDescription(NumberEntityDescription):
    get_fn: Callable[[Settings], float] = None
    set_fn: Callable[[Settings, float], None] = None


HOUSE_NUMBERS: tuple[SettingNumberDescription, ...] = (
    SettingNumberDescription(
        key="band",
        translation_key="band",
        native_min_value=0.3,
        native_max_value=3.0,
        native_step=0.1,
        native_unit_of_measurement="K",
        mode=NumberMode.BOX,
        entity_category=EntityCategory.CONFIG,
        get_fn=lambda s: s.band_k,
        set_fn=lambda s, v: setattr(s, "band_k", v),
    ),
    SettingNumberDescription(
        key="min_on",
        translation_key="min_on",
        native_min_value=5,
        native_max_value=60,
        native_step=1,
        native_unit_of_measurement="min",
        mode=NumberMode.BOX,
        entity_category=EntityCategory.CONFIG,
        get_fn=lambda s: s.min_on_min,
        set_fn=lambda s, v: setattr(s, "min_on_min", v),
    ),
    SettingNumberDescription(
        key="min_off",
        translation_key="min_off",
        native_min_value=3,
        native_max_value=60,
        native_step=1,
        native_unit_of_measurement="min",
        mode=NumberMode.BOX,
        entity_category=EntityCategory.CONFIG,
        get_fn=lambda s: s.min_off_min,
        set_fn=lambda s, v: setattr(s, "min_off_min", v),
    ),
    SettingNumberDescription(
        key="contracted_kva",
        translation_key="contracted_kva",
        native_min_value=1.15,
        native_max_value=41.4,
        native_step=0.05,
        native_unit_of_measurement="kVA",
        mode=NumberMode.BOX,
        entity_category=EntityCategory.CONFIG,
        get_fn=lambda s: s.contracted_kva,
        set_fn=lambda s, v: setattr(s, "contracted_kva", v),
    ),
    SettingNumberDescription(
        key="shed_start",
        translation_key="shed_start",
        native_min_value=50,
        native_max_value=100,
        native_step=1,
        native_unit_of_measurement="%",
        mode=NumberMode.BOX,
        entity_category=EntityCategory.CONFIG,
        get_fn=lambda s: s.shed_start_pct * 100.0,
        set_fn=lambda s, v: setattr(s, "shed_start_pct", v / 100.0),
    ),
    SettingNumberDescription(
        key="shed_restore",
        translation_key="shed_restore",
        native_min_value=30,
        native_max_value=95,
        native_step=1,
        native_unit_of_measurement="%",
        mode=NumberMode.BOX,
        entity_category=EntityCategory.CONFIG,
        get_fn=lambda s: s.shed_restore_pct * 100.0,
        set_fn=lambda s, v: setattr(s, "shed_restore_pct", v / 100.0),
    ),
    SettingNumberDescription(
        key="adaptive_blend",
        translation_key="adaptive_blend",
        native_min_value=0,
        native_max_value=100,
        native_step=5,
        native_unit_of_measurement="%",
        mode=NumberMode.BOX,
        entity_category=EntityCategory.CONFIG,
        get_fn=lambda s: s.adaptive_blend * 100.0,
        set_fn=lambda s, v: setattr(s, "adaptive_blend", v / 100.0),
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    runtime: AdaptiveComfortRuntime = hass.data[DOMAIN][entry.entry_id]
    async_add_entities(HouseNumber(runtime, description) for description in HOUSE_NUMBERS)
    for zone in runtime.zones.values():
        async_add_entities(
            [ZoneOffsetNumber(runtime, zone)], config_subentry_id=zone.config.zone_id
        )


class HouseNumber(AdaptiveComfortEntity, NumberEntity):
    entity_description: SettingNumberDescription

    def __init__(
        self, runtime: AdaptiveComfortRuntime, description: SettingNumberDescription
    ) -> None:
        super().__init__(runtime, description.key)
        self.entity_description = description

    @property
    def native_value(self) -> float:
        return self.entity_description.get_fn(self.runtime.settings)

    async def async_set_native_value(self, value: float) -> None:
        self.entity_description.set_fn(self.runtime.settings, value)
        self.runtime.request_save()
        self.runtime.notify()


class ZoneOffsetNumber(AdaptiveComfortZoneEntity, NumberEntity):
    _attr_translation_key = "comfort_offset"
    _attr_native_min_value = -3.0
    _attr_native_max_value = 3.0
    _attr_native_step = 0.5
    _attr_native_unit_of_measurement = "K"
    _attr_mode = NumberMode.BOX
    _attr_entity_category = EntityCategory.CONFIG

    def __init__(self, runtime: AdaptiveComfortRuntime, zone) -> None:
        super().__init__(runtime, zone, "comfort_offset")

    @property
    def native_value(self) -> float:
        return self.runtime.settings.zone_offsets.get(self.zone.config.zone_id, 0.0)

    async def async_set_native_value(self, value: float) -> None:
        self.runtime.settings.zone_offsets[self.zone.config.zone_id] = value
        self.runtime.request_save()
        self.runtime.notify()
