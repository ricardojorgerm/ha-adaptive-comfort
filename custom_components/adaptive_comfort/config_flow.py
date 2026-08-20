"""Config flow: hub entry plus zone / unconditioned-room subentries."""

from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.core import callback
from homeassistant.helpers.selector import (
    BooleanSelector,
    EntitySelector,
    EntitySelectorConfig,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectSelector,
    SelectSelectorConfig,
    TextSelector,
)

from .const import (
    CONF_AREA,
    CONF_AREAS,
    CONF_BATTERY_POSITIVE_DISCHARGING,
    CONF_BATTERY_POWER,
    CONF_CONSUMPTION,
    CONF_CONTRACTED_KVA,
    CONF_DEFAULT_TARGET,
    CONF_DOOR_SENSOR,
    CONF_ENERGY_MERGE,
    CONF_GRID_POWER,
    CONF_HEADS,
    CONF_HEIGHT,
    CONF_HUMIDITY_SENSOR,
    CONF_INDOOR_FANS,
    CONF_KNOWN_LOADS,
    CONF_MULTISPLIT,
    CONF_NAME,
    CONF_OUTDOOR_EXHAUST_FANS,
    CONF_OUTDOOR_TEMP,
    CONF_POWER_FACTOR,
    CONF_PRESENCE,
    CONF_PRESENCE_SENSOR,
    CONF_ROOM_TYPE,
    CONF_TEMP_SENSOR,
    CONF_WEATHER,
    DEFAULT_CEILING_HEIGHT_M,
    DEFAULT_CONTRACTED_KVA,
    DEFAULT_TARGET_C,
    DOMAIN,
    SUBENTRY_ROOM,
    SUBENTRY_ZONE,
)
from .core.power import ENERGY_MERGE_MAX, ENERGY_MERGE_SUM
from .core.types import ROOM_TYPE_REGULAR, ROOM_TYPE_WET

POWER_SENSOR = EntitySelector(EntitySelectorConfig(domain="sensor", device_class="power"))
ENERGY_SENSOR = EntitySelector(
    EntitySelectorConfig(domain="sensor", device_class="energy", multiple=True)
)
TEMP_SENSOR = EntitySelector(EntitySelectorConfig(domain="sensor", device_class="temperature"))
HUMIDITY_SENSOR = EntitySelector(EntitySelectorConfig(domain="sensor", device_class="humidity"))
PRESENCE_ENTITY = EntitySelector(
    EntitySelectorConfig(domain=["person", "device_tracker", "binary_sensor", "group", "zone"])
)
FAN_ENTITY = EntitySelector(EntitySelectorConfig(domain=["fan", "switch"], multiple=True))


def _entity_tuple(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    if isinstance(value, list):
        return tuple(value)
    return (str(value),)


def _house_schema(defaults: dict[str, Any]) -> vol.Schema:
    def d(key: str, fallback: Any = None) -> dict:
        value = defaults.get(key, fallback)
        return {"suggested_value": value} if value is not None else {}

    return vol.Schema(
        {
            vol.Required(CONF_GRID_POWER, description=d(CONF_GRID_POWER)): POWER_SENSOR,
            vol.Optional(CONF_BATTERY_POWER, description=d(CONF_BATTERY_POWER)): POWER_SENSOR,
            vol.Required(
                CONF_BATTERY_POSITIVE_DISCHARGING,
                default=defaults.get(CONF_BATTERY_POSITIVE_DISCHARGING, True),
            ): BooleanSelector(),
            vol.Optional(CONF_KNOWN_LOADS, description=d(CONF_KNOWN_LOADS)): EntitySelector(
                EntitySelectorConfig(domain="sensor", device_class="power", multiple=True)
            ),
            vol.Optional(CONF_CONSUMPTION, description=d(CONF_CONSUMPTION)): ENERGY_SENSOR,
            vol.Required(
                CONF_ENERGY_MERGE,
                default=defaults.get(CONF_ENERGY_MERGE, ENERGY_MERGE_MAX),
            ): SelectSelector(
                SelectSelectorConfig(
                    options=[ENERGY_MERGE_MAX, ENERGY_MERGE_SUM],
                    translation_key="energy_merge",
                )
            ),
            vol.Optional(CONF_OUTDOOR_TEMP, description=d(CONF_OUTDOOR_TEMP)): TEMP_SENSOR,
            vol.Optional(CONF_WEATHER, description=d(CONF_WEATHER)): EntitySelector(
                EntitySelectorConfig(domain="weather")
            ),
            vol.Optional(CONF_PRESENCE, description=d(CONF_PRESENCE)): PRESENCE_ENTITY,
            vol.Required(
                CONF_MULTISPLIT, default=defaults.get(CONF_MULTISPLIT, True)
            ): BooleanSelector(),
            vol.Required(
                CONF_CONTRACTED_KVA,
                default=defaults.get(CONF_CONTRACTED_KVA, DEFAULT_CONTRACTED_KVA),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=1.15,
                    max=41.4,
                    step=0.05,
                    mode=NumberSelectorMode.BOX,
                    unit_of_measurement="kVA",
                )
            ),
            vol.Required(
                CONF_POWER_FACTOR, default=defaults.get(CONF_POWER_FACTOR, 1.0)
            ): NumberSelector(
                NumberSelectorConfig(min=0.5, max=1.0, step=0.01, mode=NumberSelectorMode.BOX)
            ),
            vol.Required(
                CONF_DEFAULT_TARGET,
                default=defaults.get(CONF_DEFAULT_TARGET, DEFAULT_TARGET_C),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=16,
                    max=28,
                    step=0.5,
                    mode=NumberSelectorMode.BOX,
                    unit_of_measurement="°C",
                )
            ),
        }
    )


class AdaptiveComfortConfigFlow(ConfigFlow, domain=DOMAIN):
    """Hub config flow: one entry per house."""

    VERSION = 1

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        await self.async_set_unique_id(DOMAIN)
        self._abort_if_unique_id_configured()
        if user_input is not None:
            return self.async_create_entry(title="Adaptive Comfort", data=user_input)
        return self.async_show_form(step_id="user", data_schema=_house_schema({}))

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        entry = self._get_reconfigure_entry()
        if user_input is not None:
            return self.async_update_and_abort(entry, data=user_input)
        return self.async_show_form(
            step_id="reconfigure", data_schema=_house_schema(dict(entry.data))
        )

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        return {
            SUBENTRY_ZONE: ZoneSubentryFlowHandler,
            SUBENTRY_ROOM: RoomSubentryFlowHandler,
        }


def _zone_base_schema(defaults: dict[str, Any]) -> vol.Schema:
    def d(key: str) -> dict:
        value = defaults.get(key)
        return {"suggested_value": value} if value is not None else {}

    return vol.Schema(
        {
            vol.Required(CONF_NAME, description=d(CONF_NAME)): TextSelector(),
            vol.Required(CONF_HEADS, description=d(CONF_HEADS)): EntitySelector(
                EntitySelectorConfig(domain="climate", multiple=True)
            ),
            vol.Required(
                CONF_HEIGHT, default=defaults.get(CONF_HEIGHT, DEFAULT_CEILING_HEIGHT_M)
            ): NumberSelector(
                NumberSelectorConfig(
                    min=2.0,
                    max=5.0,
                    step=0.05,
                    mode=NumberSelectorMode.BOX,
                    unit_of_measurement="m",
                )
            ),
            vol.Optional(CONF_TEMP_SENSOR, description=d(CONF_TEMP_SENSOR)): TEMP_SENSOR,
            vol.Optional(
                CONF_HUMIDITY_SENSOR, description=d(CONF_HUMIDITY_SENSOR)
            ): HUMIDITY_SENSOR,
            vol.Optional(CONF_DOOR_SENSOR, description=d(CONF_DOOR_SENSOR)): EntitySelector(
                EntitySelectorConfig(domain="binary_sensor")
            ),
            vol.Optional(
                CONF_PRESENCE_SENSOR, description=d(CONF_PRESENCE_SENSOR)
            ): PRESENCE_ENTITY,
            vol.Optional(CONF_INDOOR_FANS, description=d(CONF_INDOOR_FANS)): FAN_ENTITY,
            vol.Optional(
                CONF_OUTDOOR_EXHAUST_FANS, description=d(CONF_OUTDOOR_EXHAUST_FANS)
            ): FAN_ENTITY,
        }
    )


def _areas_schema(heads: list[str], defaults: list[float] | None = None) -> vol.Schema:
    """One area field per head; areas are per-room and never summed."""
    fields: dict[Any, Any] = {}
    for index, _head in enumerate(heads):
        default = 12.0
        if defaults and index < len(defaults):
            default = defaults[index]
        fields[vol.Required(f"area_{index + 1}", default=default)] = NumberSelector(
            NumberSelectorConfig(
                min=2,
                max=150,
                step=0.5,
                mode=NumberSelectorMode.BOX,
                unit_of_measurement="m²",
            )
        )
    return vol.Schema(fields)


class ZoneSubentryFlowHandler(ConfigSubentryFlow):
    """Add or reconfigure a climatized zone (N mirrored heads, one room each)."""

    def __init__(self) -> None:
        self._base: dict[str, Any] = {}

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> SubentryFlowResult:
        if user_input is not None:
            self._base = user_input
            return await self.async_step_areas()
        return self.async_show_form(step_id="user", data_schema=_zone_base_schema({}))

    async def async_step_areas(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        heads: list[str] = self._base[CONF_HEADS]
        if user_input is not None:
            areas = [float(user_input[f"area_{i + 1}"]) for i in range(len(heads))]
            data = {**self._base, CONF_AREAS: areas}
            data[CONF_INDOOR_FANS] = _entity_tuple(data.get(CONF_INDOOR_FANS))
            data[CONF_OUTDOOR_EXHAUST_FANS] = _entity_tuple(data.get(CONF_OUTDOOR_EXHAUST_FANS))
            name = data.pop(CONF_NAME)
            return self.async_create_entry(title=name, data=data)
        return self.async_show_form(
            step_id="areas",
            data_schema=_areas_schema(heads),
            description_placeholders={"count": str(len(heads))},
        )

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        subentry = self._get_reconfigure_subentry()
        if user_input is not None:
            self._base = user_input
            return await self.async_step_reconfigure_areas()
        defaults = {CONF_NAME: subentry.title, **dict(subentry.data)}
        return self.async_show_form(step_id="reconfigure", data_schema=_zone_base_schema(defaults))

    async def async_step_reconfigure_areas(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        subentry = self._get_reconfigure_subentry()
        heads: list[str] = self._base[CONF_HEADS]
        if user_input is not None:
            areas = [float(user_input[f"area_{i + 1}"]) for i in range(len(heads))]
            data = {**self._base, CONF_AREAS: areas}
            data[CONF_INDOOR_FANS] = _entity_tuple(data.get(CONF_INDOOR_FANS))
            data[CONF_OUTDOOR_EXHAUST_FANS] = _entity_tuple(data.get(CONF_OUTDOOR_EXHAUST_FANS))
            name = data.pop(CONF_NAME)
            return self.async_update_and_abort(self._get_entry(), subentry, title=name, data=data)
        old_areas = list(subentry.data.get(CONF_AREAS, []))
        return self.async_show_form(
            step_id="reconfigure_areas",
            data_schema=_areas_schema(heads, old_areas),
            description_placeholders={"count": str(len(heads))},
        )


def _room_schema(defaults: dict[str, Any]) -> vol.Schema:
    return vol.Schema(
        {
            vol.Required(
                CONF_NAME,
                description=(
                    {"suggested_value": defaults[CONF_NAME]} if CONF_NAME in defaults else {}
                ),
            ): TextSelector(),
            vol.Required(CONF_AREA, default=defaults.get(CONF_AREA, 6.0)): NumberSelector(
                NumberSelectorConfig(
                    min=1,
                    max=150,
                    step=0.5,
                    mode=NumberSelectorMode.BOX,
                    unit_of_measurement="m²",
                )
            ),
            vol.Required(
                CONF_ROOM_TYPE, default=defaults.get(CONF_ROOM_TYPE, ROOM_TYPE_REGULAR)
            ): SelectSelector(
                SelectSelectorConfig(
                    options=[ROOM_TYPE_REGULAR, ROOM_TYPE_WET],
                    translation_key="room_type",
                )
            ),
            vol.Optional(
                CONF_TEMP_SENSOR,
                description=(
                    {"suggested_value": defaults[CONF_TEMP_SENSOR]}
                    if defaults.get(CONF_TEMP_SENSOR)
                    else {}
                ),
            ): TEMP_SENSOR,
            vol.Required(
                CONF_HEIGHT, default=defaults.get(CONF_HEIGHT, DEFAULT_CEILING_HEIGHT_M)
            ): NumberSelector(
                NumberSelectorConfig(
                    min=2.0,
                    max=5.0,
                    step=0.05,
                    mode=NumberSelectorMode.BOX,
                    unit_of_measurement="m",
                )
            ),
        }
    )


class RoomSubentryFlowHandler(ConfigSubentryFlow):
    """Add or reconfigure an unconditioned room (wet/regular)."""

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> SubentryFlowResult:
        if user_input is not None:
            name = user_input.pop(CONF_NAME)
            return self.async_create_entry(title=name, data=user_input)
        return self.async_show_form(step_id="user", data_schema=_room_schema({}))

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        subentry = self._get_reconfigure_subentry()
        if user_input is not None:
            name = user_input.pop(CONF_NAME)
            return self.async_update_and_abort(
                self._get_entry(), subentry, title=name, data=user_input
            )
        defaults = {CONF_NAME: subentry.title, **dict(subentry.data)}
        return self.async_show_form(step_id="reconfigure", data_schema=_room_schema(defaults))
