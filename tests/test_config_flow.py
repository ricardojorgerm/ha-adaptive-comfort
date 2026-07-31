"""Config flow tests (require pytest-homeassistant-custom-component)."""

from __future__ import annotations

import pytest

pytest.importorskip("homeassistant")

# Subentry flows landed in recent HA cores; on older test harnesses the
# integration's config_flow module cannot even import. Skip (not fail) so a
# mismatched local environment reports honestly - CI pins a matching core.
try:
    from homeassistant.config_entries import ConfigSubentryFlow  # noqa: F401
except ImportError:  # pragma: no cover - depends on installed HA version
    pytest.skip(
        "installed homeassistant core lacks ConfigSubentryFlow (subentry flows)",
        allow_module_level=True,
    )

from homeassistant import config_entries
from homeassistant.components.climate import HVACMode
from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.const import UnitOfPower, UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType

from custom_components.adaptive_comfort.const import (
    CONF_AREA,
    CONF_AREAS,
    CONF_CONTRACTED_KVA,
    CONF_DEFAULT_TARGET,
    CONF_GRID_POWER,
    CONF_HEADS,
    CONF_HEIGHT,
    CONF_MULTISPLIT,
    CONF_NAME,
    CONF_POWER_FACTOR,
    CONF_ROOM_TYPE,
    CONF_TEMP_SENSOR,
    DOMAIN,
    SUBENTRY_ROOM,
    SUBENTRY_ZONE,
)
from custom_components.adaptive_comfort.core.types import ROOM_TYPE_WET


def _seed_entities(hass: HomeAssistant) -> None:
    hass.states.async_set(
        "sensor.grid_power",
        "1200",
        {"device_class": SensorDeviceClass.POWER, "unit_of_measurement": UnitOfPower.WATT},
    )
    for entity_id in ("climate.bedroom", "climate.living", "climate.office"):
        hass.states.async_set(
            entity_id,
            HVACMode.OFF,
            {
                "hvac_modes": [HVACMode.OFF, HVACMode.HEAT, HVACMode.COOL, HVACMode.FAN_ONLY],
                "min_temp": 16,
                "max_temp": 30,
                "temperature": 22,
            },
        )
    hass.states.async_set(
        "sensor.bathroom_temp",
        "23.5",
        {
            "device_class": SensorDeviceClass.TEMPERATURE,
            "unit_of_measurement": UnitOfTemperature.CELSIUS,
        },
    )


async def test_user_flow_form(hass: HomeAssistant) -> None:
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "user"


async def test_user_flow_create_entry(hass: HomeAssistant) -> None:
    _seed_entities(hass)
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            CONF_GRID_POWER: "sensor.grid_power",
            CONF_MULTISPLIT: True,
            CONF_CONTRACTED_KVA: 3.45,
            CONF_POWER_FACTOR: 1.0,
            CONF_DEFAULT_TARGET: 22.5,
        },
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["title"] == "Adaptive Comfort"
    assert result["data"][CONF_GRID_POWER] == "sensor.grid_power"


async def test_user_flow_aborts_if_already_configured(hass: HomeAssistant) -> None:
    _seed_entities(hass)
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            CONF_GRID_POWER: "sensor.grid_power",
            CONF_MULTISPLIT: True,
            CONF_CONTRACTED_KVA: 3.45,
            CONF_POWER_FACTOR: 1.0,
            CONF_DEFAULT_TARGET: 22.5,
        },
    )
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] == FlowResultType.ABORT
    assert result["reason"] == "already_configured"


async def test_zone_subentry_flow(hass: HomeAssistant) -> None:
    _seed_entities(hass)
    flow = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    entry = await hass.config_entries.flow.async_configure(
        flow["flow_id"],
        {
            CONF_GRID_POWER: "sensor.grid_power",
            CONF_MULTISPLIT: True,
            CONF_CONTRACTED_KVA: 3.45,
            CONF_POWER_FACTOR: 1.0,
            CONF_DEFAULT_TARGET: 22.5,
        },
    )
    assert entry["type"] == FlowResultType.CREATE_ENTRY
    config_entry = hass.config_entries.async_entries(DOMAIN)[0]

    sub = await hass.config_entries.subentries.async_init(
        (config_entry.entry_id, SUBENTRY_ZONE),
        context={"source": config_entries.SOURCE_USER},
    )
    assert sub["type"] == FlowResultType.FORM
    sub = await hass.config_entries.subentries.async_configure(
        sub["flow_id"],
        {
            CONF_NAME: "Bedroom",
            CONF_HEADS: ["climate.bedroom"],
            CONF_HEIGHT: 2.6,
        },
    )
    assert sub["type"] == FlowResultType.FORM
    assert sub["step_id"] == "areas"
    sub = await hass.config_entries.subentries.async_configure(
        sub["flow_id"],
        {"area_1": 12.0},
    )
    assert sub["type"] == FlowResultType.CREATE_ENTRY
    assert sub["title"] == "Bedroom"
    config_entry = hass.config_entries.async_get_entry(config_entry.entry_id)
    assert config_entry is not None
    stored = next(s for s in config_entry.subentries.values() if s.title == "Bedroom")
    assert stored.data[CONF_AREAS] == [12.0]


async def test_hub_reconfigure_flow(hass: HomeAssistant) -> None:
    _seed_entities(hass)
    flow = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    await hass.config_entries.flow.async_configure(
        flow["flow_id"],
        {
            CONF_GRID_POWER: "sensor.grid_power",
            CONF_MULTISPLIT: True,
            CONF_CONTRACTED_KVA: 3.45,
            CONF_POWER_FACTOR: 1.0,
            CONF_DEFAULT_TARGET: 22.5,
        },
    )
    config_entry = hass.config_entries.async_entries(DOMAIN)[0]
    result = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={
            "source": config_entries.SOURCE_RECONFIGURE,
            "entry_id": config_entry.entry_id,
        },
    )
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "reconfigure"
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            CONF_GRID_POWER: "sensor.grid_power",
            CONF_MULTISPLIT: True,
            CONF_CONTRACTED_KVA: 4.6,
            CONF_POWER_FACTOR: 1.0,
            CONF_DEFAULT_TARGET: 23.0,
        },
    )
    assert result["type"] == FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    config_entry = hass.config_entries.async_get_entry(config_entry.entry_id)
    assert config_entry is not None
    assert config_entry.data[CONF_CONTRACTED_KVA] == 4.6
    assert config_entry.data[CONF_DEFAULT_TARGET] == 23.0


async def test_unconditioned_room_subentry_flow(hass: HomeAssistant) -> None:
    _seed_entities(hass)
    flow = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    await hass.config_entries.flow.async_configure(
        flow["flow_id"],
        {
            CONF_GRID_POWER: "sensor.grid_power",
            CONF_MULTISPLIT: True,
            CONF_CONTRACTED_KVA: 3.45,
            CONF_POWER_FACTOR: 1.0,
            CONF_DEFAULT_TARGET: 22.5,
        },
    )
    config_entry = hass.config_entries.async_entries(DOMAIN)[0]

    sub = await hass.config_entries.subentries.async_init(
        (config_entry.entry_id, SUBENTRY_ROOM),
        context={"source": config_entries.SOURCE_USER},
    )
    sub = await hass.config_entries.subentries.async_configure(
        sub["flow_id"],
        {
            CONF_NAME: "Bathroom",
            CONF_AREA: 5.0,
            CONF_ROOM_TYPE: ROOM_TYPE_WET,
            CONF_HEIGHT: 2.6,
            CONF_TEMP_SENSOR: "sensor.bathroom_temp",
        },
    )
    assert sub["type"] == FlowResultType.CREATE_ENTRY
    assert sub["title"] == "Bathroom"
