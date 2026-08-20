"""Constants for the Adaptive Comfort integration."""

from __future__ import annotations

from homeassistant.const import Platform

DOMAIN = "adaptive_comfort"

PLATFORMS: list[Platform] = [
    Platform.BINARY_SENSOR,
    Platform.CLIMATE,
    Platform.NUMBER,
    Platform.SENSOR,
    Platform.SWITCH,
]

# Subentry types
SUBENTRY_ZONE = "zone"
SUBENTRY_ROOM = "unconditioned_room"

# Main entry config keys
CONF_GRID_POWER = "grid_power_entity"
CONF_BATTERY_POWER = "battery_power_entity"
CONF_BATTERY_POSITIVE_DISCHARGING = "battery_positive_discharging"
CONF_KNOWN_LOADS = "known_load_entities"
CONF_CONSUMPTION = "consumption_entities"
CONF_ENERGY_MERGE = "energy_merge"
CONF_OUTDOOR_TEMP = "outdoor_temp_entity"
CONF_WEATHER = "weather_entity"
CONF_PRESENCE = "presence_entity"
CONF_MULTISPLIT = "multisplit"
CONF_CONTRACTED_KVA = "contracted_kva"
CONF_POWER_FACTOR = "power_factor"
CONF_DEFAULT_TARGET = "default_target"

# Zone subentry keys
CONF_NAME = "name"
CONF_HEADS = "climate_entities"
CONF_AREAS = "room_areas"
CONF_HEIGHT = "ceiling_height"
CONF_TEMP_SENSOR = "temp_sensor"
CONF_HUMIDITY_SENSOR = "humidity_sensor"
CONF_DOOR_SENSOR = "door_sensor"
CONF_PRESENCE_SENSOR = "presence_sensor"
CONF_INDOOR_FANS = "indoor_fan_entities"
CONF_OUTDOOR_EXHAUST_FANS = "outdoor_exhaust_fan_entities"

# Unconditioned-room subentry keys
CONF_AREA = "area_m2"
CONF_ROOM_TYPE = "room_type"

DEFAULT_TARGET_C = 22.5
DEFAULT_CONTRACTED_KVA = 3.45
DEFAULT_CEILING_HEIGHT_M = 2.6

SIGNAL_UPDATE = f"{DOMAIN}_update"

STORAGE_VERSION = 1
