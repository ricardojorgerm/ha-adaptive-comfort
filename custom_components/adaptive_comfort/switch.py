"""Feature switches for the house and per-zone participation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from homeassistant.components.switch import SwitchEntity, SwitchEntityDescription
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import DOMAIN
from .coordinator import AdaptiveComfortRuntime
from .core.types import PRESET_MANUAL, Settings
from .entity import AdaptiveComfortEntity, AdaptiveComfortZoneEntity


@dataclass(frozen=True, kw_only=True)
class SettingSwitchDescription(SwitchEntityDescription):
    get_fn: Callable[[Settings], bool] = None
    set_fn: Callable[[Settings, bool], None] = None


HOUSE_SWITCHES: tuple[SettingSwitchDescription, ...] = (
    SettingSwitchDescription(
        key="coordination",
        translation_key="coordination",
        entity_category=EntityCategory.CONFIG,
        get_fn=lambda s: s.coordination,
        set_fn=lambda s, v: setattr(s, "coordination", v),
    ),
    SettingSwitchDescription(
        key="presence_adaptation",
        translation_key="presence_adaptation",
        entity_category=EntityCategory.CONFIG,
        get_fn=lambda s: s.presence_adaptation,
        set_fn=lambda s, v: setattr(s, "presence_adaptation", v),
    ),
    SettingSwitchDescription(
        key="zone_presence_adaptation",
        translation_key="zone_presence_adaptation",
        entity_category=EntityCategory.CONFIG,
        get_fn=lambda s: s.zone_presence_adaptation,
        set_fn=lambda s, v: setattr(s, "zone_presence_adaptation", v),
    ),
    SettingSwitchDescription(
        key="shedding_enabled",
        translation_key="shedding_enabled",
        entity_category=EntityCategory.CONFIG,
        get_fn=lambda s: s.shedding_enabled,
        set_fn=lambda s, v: setattr(s, "shedding_enabled", v),
    ),
    SettingSwitchDescription(
        key="tracking",
        translation_key="tracking",
        entity_category=EntityCategory.CONFIG,
        get_fn=lambda s: s.tracking,
        set_fn=lambda s, v: setattr(s, "tracking", v),
    ),
    SettingSwitchDescription(
        key="park_learning",
        translation_key="park_learning",
        entity_category=EntityCategory.CONFIG,
        get_fn=lambda s: s.park_learning,
        set_fn=lambda s, v: setattr(s, "park_learning", v),
    ),
    SettingSwitchDescription(
        key="auto_regime",
        translation_key="auto_regime",
        entity_category=EntityCategory.CONFIG,
        get_fn=lambda s: s.auto_regime,
        set_fn=lambda s, v: setattr(s, "auto_regime", v),
    ),
    SettingSwitchDescription(
        key="night_ventilate",
        translation_key="night_ventilate",
        entity_category=EntityCategory.CONFIG,
        get_fn=lambda s: s.night_ventilate,
        set_fn=lambda s, v: setattr(s, "night_ventilate", v),
    ),
    SettingSwitchDescription(
        key="prefer_continuous",
        translation_key="prefer_continuous",
        entity_category=EntityCategory.CONFIG,
        get_fn=lambda s: s.prefer_continuous,
        set_fn=lambda s, v: setattr(s, "prefer_continuous", v),
    ),
    SettingSwitchDescription(
        key="fan_assist",
        translation_key="fan_assist",
        entity_category=EntityCategory.CONFIG,
        get_fn=lambda s: s.fan_assist,
        set_fn=lambda s, v: setattr(s, "fan_assist", v),
    ),
    SettingSwitchDescription(
        key="window_suggest",
        translation_key="window_suggest",
        entity_category=EntityCategory.CONFIG,
        get_fn=lambda s: s.window_suggest,
        set_fn=lambda s, v: setattr(s, "window_suggest", v),
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    runtime: AdaptiveComfortRuntime = hass.data[DOMAIN][entry.entry_id]
    async_add_entities(
        [
            *(HouseSwitch(runtime, description) for description in HOUSE_SWITCHES),
            ManualControlSwitch(runtime),
        ]
    )
    for zone in runtime.zones.values():
        async_add_entities(
            [ZoneEnabledSwitch(runtime, zone)], config_subentry_id=zone.config.zone_id
        )


class HouseSwitch(AdaptiveComfortEntity, SwitchEntity):
    entity_description: SettingSwitchDescription

    def __init__(
        self, runtime: AdaptiveComfortRuntime, description: SettingSwitchDescription
    ) -> None:
        super().__init__(runtime, description.key)
        self.entity_description = description

    @property
    def is_on(self) -> bool:
        return self.entity_description.get_fn(self.runtime.settings)

    async def async_turn_on(self, **kwargs: Any) -> None:
        self.entity_description.set_fn(self.runtime.settings, True)
        self.runtime.request_save()
        self.runtime.notify()

    async def async_turn_off(self, **kwargs: Any) -> None:
        self.entity_description.set_fn(self.runtime.settings, False)
        self.runtime.request_save()
        self.runtime.notify()


class ManualControlSwitch(AdaptiveComfortEntity, SwitchEntity):
    """Pause adaptive commanding; leave heads as-is (mirrors climate preset manual)."""

    _attr_translation_key = "manual_control"
    _attr_entity_category = EntityCategory.CONFIG

    def __init__(self, runtime: AdaptiveComfortRuntime) -> None:
        super().__init__(runtime, "manual_control")

    @property
    def is_on(self) -> bool:
        return self.runtime.manual_control

    async def async_turn_on(self, **kwargs: Any) -> None:
        self.runtime.set_preset(PRESET_MANUAL)
        self.runtime.request_save()
        self.runtime.notify()

    async def async_turn_off(self, **kwargs: Any) -> None:
        self.runtime.clear_manual()
        self.runtime.request_save()
        self.runtime.notify()


class ZoneEnabledSwitch(AdaptiveComfortZoneEntity, SwitchEntity):
    _attr_translation_key = "zone_enabled"
    _attr_entity_category = EntityCategory.CONFIG

    def __init__(self, runtime: AdaptiveComfortRuntime, zone) -> None:
        super().__init__(runtime, zone, "enabled")

    @property
    def is_on(self) -> bool:
        return self.runtime.settings.zone_enabled.get(self.zone.config.zone_id, True)

    async def async_turn_on(self, **kwargs: Any) -> None:
        self.runtime.settings.zone_enabled[self.zone.config.zone_id] = True
        self.runtime.request_save()
        self.runtime.notify()

    async def async_turn_off(self, **kwargs: Any) -> None:
        self.runtime.settings.zone_enabled[self.zone.config.zone_id] = False
        self.runtime.request_save()
        self.runtime.notify()
