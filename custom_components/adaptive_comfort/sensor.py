"""Diagnostic and telemetry sensors for house and zones."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    PERCENTAGE,
    UnitOfEnergy,
    UnitOfPower,
    UnitOfTemperature,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import DOMAIN
from .coordinator import AdaptiveComfortRuntime, ZoneRuntime
from .entity import AdaptiveComfortEntity, AdaptiveComfortZoneEntity


@dataclass(frozen=True, kw_only=True)
class HouseSensorDescription(SensorEntityDescription):
    value_fn: Callable[[AdaptiveComfortRuntime], Any] = None
    attr_fn: Callable[[AdaptiveComfortRuntime], dict | None] = lambda _r: None


@dataclass(frozen=True, kw_only=True)
class ZoneSensorDescription(SensorEntityDescription):
    value_fn: Callable[[ZoneRuntime, AdaptiveComfortRuntime], Any] = None
    attr_fn: Callable[[ZoneRuntime, AdaptiveComfortRuntime], dict | None] = lambda _z, _r: None


def _round(value: float | None, digits: int = 2) -> float | None:
    return None if value is None else round(value, digits)


HOUSE_SENSORS: tuple[HouseSensorDescription, ...] = (
    HouseSensorDescription(
        key="house_load",
        translation_key="house_load",
        native_unit_of_measurement=UnitOfPower.WATT,
        device_class=SensorDeviceClass.POWER,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda r: _round(r.p_load, 0),
    ),
    HouseSensorDescription(
        key="ac_power_estimate",
        translation_key="ac_power_estimate",
        native_unit_of_measurement=UnitOfPower.WATT,
        device_class=SensorDeviceClass.POWER,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda r: _round(r.p_ac, 0),
    ),
    HouseSensorDescription(
        key="ac_energy",
        translation_key="ac_energy",
        native_unit_of_measurement=UnitOfEnergy.WATT_HOUR,
        device_class=SensorDeviceClass.ENERGY,
        state_class=SensorStateClass.TOTAL_INCREASING,
        value_fn=lambda r: _round(r.energy.house_wh, 0),
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    HouseSensorDescription(
        key="power_headroom",
        translation_key="power_headroom",
        native_unit_of_measurement=UnitOfPower.WATT,
        device_class=SensorDeviceClass.POWER,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda r: _round(r.power_headroom_w, 0),
    ),
    HouseSensorDescription(
        key="outdoor_effective",
        translation_key="outdoor_effective",
        native_unit_of_measurement=UnitOfTemperature.CELSIUS,
        device_class=SensorDeviceClass.TEMPERATURE,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda r: _round(r.t_out, 1),
        attr_fn=lambda r: {"source": r.outdoor_source},
    ),
    HouseSensorDescription(
        key="outdoor_running_mean",
        translation_key="outdoor_running_mean",
        native_unit_of_measurement=UnitOfTemperature.CELSIUS,
        device_class=SensorDeviceClass.TEMPERATURE,
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda r: _round(r.t_rm, 2),
    ),
    HouseSensorDescription(
        key="dominant_mode",
        translation_key="dominant_mode",
        device_class=SensorDeviceClass.ENUM,
        options=["off", "heat", "cool"],
        value_fn=lambda r: r.controller_state.mode,
        attr_fn=lambda r: {"source": r.mode_source},
    ),
    HouseSensorDescription(
        key="effective_preset",
        translation_key="effective_preset",
        device_class=SensorDeviceClass.ENUM,
        options=["none", "eco", "away", "boost", "manual"],
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda r: r.effective_preset,
        attr_fn=lambda r: {
            "preset": r.settings.preset,
            "manual_control": r.manual_control,
            "house_occupied": r.house_occupied,
        },
    ),
    HouseSensorDescription(
        key="free_float_deviation",
        translation_key="free_float_deviation",
        native_unit_of_measurement="K",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda r: _round(r.free_float_deviation, 2),
    ),
    HouseSensorDescription(
        key="warm_excess",
        translation_key="warm_excess",
        native_unit_of_measurement="K·h",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda r: _round(r.warm_excess_kh, 2),
    ),
    HouseSensorDescription(
        key="cold_deficit",
        translation_key="cold_deficit",
        native_unit_of_measurement="K·h",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda r: _round(r.cold_deficit_kh, 2),
    ),
    HouseSensorDescription(
        key="total_sensible",
        translation_key="total_sensible",
        native_unit_of_measurement=UnitOfPower.WATT,
        device_class=SensorDeviceClass.POWER,
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda r: _round(r.total_sensible_w, 0),
    ),
    HouseSensorDescription(
        key="total_latent",
        translation_key="total_latent",
        native_unit_of_measurement=UnitOfPower.WATT,
        device_class=SensorDeviceClass.POWER,
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda r: _round(r.total_latent_w, 0),
    ),
    HouseSensorDescription(
        key="house_cop",
        translation_key="house_cop",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda r: _round(r.house_cop, 2),
        attr_fn=lambda r: {
            "by_mode_heads": {k: round(v[0], 2) for k, v in r.cop_table.items()},
            "by_mode_band": {k: round(v[0], 2) for k, v in r.cop_table_banded.items()},
            "by_mode_state": {k: round(v[0], 2) for k, v in r.cop_table_state.items()},
        },
    ),
    HouseSensorDescription(
        key="model_confidence",
        translation_key="model_confidence",
        native_unit_of_measurement=PERCENTAGE,
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda r: _round(r.model_confidence * 100.0, 0),
    ),
    HouseSensorDescription(
        key="parked_zones",
        translation_key="parked_zones",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda r: len(r.controller_state.zone_parked_since),
        attr_fn=lambda r: {
            "zones": sorted(
                r.zones[zid].config.name
                for zid in r.controller_state.zone_parked_since
                if zid in r.zones
            )
        },
    ),
    HouseSensorDescription(
        key="operating_regime",
        translation_key="operating_regime",
        device_class=SensorDeviceClass.ENUM,
        options=["ventilate", "continuous", "cycling"],
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda r: r.controller_state.regime,
        attr_fn=lambda r: {
            "since": r.controller_state.regime_since,
            "auto_regime": r.settings.auto_regime,
            "night_ventilate": r.settings.night_ventilate,
            "prefer_continuous": r.settings.prefer_continuous,
            "anchor_zone": r.controller_state.anchor_zone,
            "anchor_since": r.controller_state.anchor_since,
            "plant_compress_since": r.controller_state.plant_compress_since,
        },
    ),
    HouseSensorDescription(
        key="compressor_starts_per_hour",
        translation_key="compressor_starts_per_hour",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda r: round(r.starts.per_hour(time.time()), 2),
        attr_fn=lambda r: {
            "by_state_24h": r.starts.by_state(time.time()),
            "events_24h": sum(r.starts.by_state(time.time()).values()),
        },
    ),
)


def _k_exchange_attrs(zone: ZoneRuntime, _r: AdaptiveComfortRuntime) -> dict:
    return {
        "k_out_h-1": round(zone.model.k(zone.door_open), 3),
        "k_mix_h-1": round(zone.model.k_mix(zone.door_open), 3),
    }


def _per_room_airflow(zone: ZoneRuntime, _r: AdaptiveComfortRuntime) -> dict:
    k_out = zone.model.k(zone.door_open)
    return {
        f"room_{i + 1}_m3h": round(k_out * room.volume_m3, 1)
        for i, room in enumerate(zone.config.rooms)
    }


def _drift_attrs(zone: ZoneRuntime, _r: AdaptiveComfortRuntime) -> dict:
    return {
        head: {state: round(off, 2) for state, off in est.offsets.items()}
        for head, est in zone.drift.items()
    }


def _predictor_attrs(zone: ZoneRuntime, r: AdaptiveComfortRuntime) -> dict:
    stats = r.predictor_stats(zone.config.zone_id)
    attrs: dict[str, Any] = {f"{horizon}m": values for horizon, values in stats.items()}
    attrs["pending"] = r.predictor_scorer.pending_count(zone.config.zone_id)
    return attrs


def _predictor_mae_30m(zone: ZoneRuntime, r: AdaptiveComfortRuntime) -> float | None:
    stats = r.predictor_stats(zone.config.zone_id)
    stat = stats.get(30)
    return None if stat is None else stat["mae_k"]


ZONE_SENSORS: tuple[ZoneSensorDescription, ...] = (
    ZoneSensorDescription(
        key="temperature",
        translation_key="zone_temperature",
        native_unit_of_measurement=UnitOfTemperature.CELSIUS,
        device_class=SensorDeviceClass.TEMPERATURE,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda z, r: _round(z.temp, 2),
    ),
    ZoneSensorDescription(
        key="predicted_60m",
        translation_key="predicted_60m",
        native_unit_of_measurement=UnitOfTemperature.CELSIUS,
        device_class=SensorDeviceClass.TEMPERATURE,
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda z, r: _round(z.pred_60m, 2),
    ),
    ZoneSensorDescription(
        key="predictor_error",
        translation_key="predictor_error",
        native_unit_of_measurement="K",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=_predictor_mae_30m,
        attr_fn=_predictor_attrs,
    ),
    ZoneSensorDescription(
        key="k_exchange",
        translation_key="k_exchange",
        native_unit_of_measurement="1/h",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda z, r: _round(z.model.k(z.door_open), 3),
        attr_fn=_k_exchange_attrs,
    ),
    ZoneSensorDescription(
        key="k_mix",
        translation_key="k_mix",
        native_unit_of_measurement="1/h",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda z, r: _round(z.model.k_mix(z.door_open), 3),
    ),
    ZoneSensorDescription(
        key="ach",
        translation_key="ach",
        native_unit_of_measurement="1/h",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda z, r: _round(z.model.ach, 3),
    ),
    ZoneSensorDescription(
        key="airflow",
        translation_key="airflow",
        native_unit_of_measurement="m³/h",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda z, r: _round(z.model.airflow_m3h, 1),
        attr_fn=_per_room_airflow,
    ),
    ZoneSensorDescription(
        key="ua",
        translation_key="ua",
        native_unit_of_measurement="W/K",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda z, r: _round(z.model.ua_w_per_k, 1),
        attr_fn=lambda z, r: {"ua_mix_w_per_k": round(z.model.ua_mix_w_per_k, 1)},
    ),
    ZoneSensorDescription(
        key="c_eff",
        translation_key="c_eff",
        native_unit_of_measurement="Wh/K",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda z, r: _round(0.34 * z.config.total_volume_m3 * z.model.furniture_factor, 1),
    ),
    ZoneSensorDescription(
        key="furniture_factor",
        translation_key="furniture_factor",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda z, r: _round(z.model.furniture_factor, 2),
    ),
    ZoneSensorDescription(
        key="cop",
        translation_key="zone_cop",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda z, r: _round(z.model.cop, 2),
    ),
    ZoneSensorDescription(
        key="sensible_power",
        translation_key="sensible_power",
        native_unit_of_measurement=UnitOfPower.WATT,
        device_class=SensorDeviceClass.POWER,
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda z, r: _round(abs(z.sensible_w) * z.config.n_rooms, 0),
        attr_fn=lambda z, r: {
            "per_room_w": round(abs(z.sensible_w), 1),
            "signed_w": round(z.sensible_w, 1),
        },
    ),
    ZoneSensorDescription(
        key="standing_load",
        translation_key="standing_load",
        native_unit_of_measurement=UnitOfPower.WATT,
        device_class=SensorDeviceClass.POWER,
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda z, r: _round(z.standing_load_w, 0),
        attr_fn=lambda z, r: {
            "per_room_w": (
                None
                if z.standing_load_w is None
                else round(z.standing_load_w / max(1, z.config.n_rooms), 1)
            ),
            "n_rooms": z.config.n_rooms,
        },
    ),
    ZoneSensorDescription(
        key="latent_power",
        translation_key="latent_power",
        native_unit_of_measurement=UnitOfPower.WATT,
        device_class=SensorDeviceClass.POWER,
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda z, r: _round(z.latent_w * z.config.n_rooms, 0),
        attr_fn=lambda z, r: {"per_room_w": round(z.latent_w, 1)},
    ),
    ZoneSensorDescription(
        key="drift_offset",
        translation_key="drift_offset",
        native_unit_of_measurement="K",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda z, r: _round(
            sum(est.offset(z.head_state) for est in z.drift.values()) / len(z.drift)
            if z.drift
            else None,
            2,
        ),
        attr_fn=_drift_attrs,
    ),
    ZoneSensorDescription(
        key="confidence",
        translation_key="zone_confidence",
        native_unit_of_measurement=PERCENTAGE,
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda z, r: _round(z.model.confidence(z.door_open) * 100.0, 0),
    ),
    ZoneSensorDescription(
        key="allocated_power",
        translation_key="allocated_power",
        native_unit_of_measurement=UnitOfPower.WATT,
        device_class=SensorDeviceClass.POWER,
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda z, r: _round(z.allocated_w, 0),
    ),
    ZoneSensorDescription(
        key="head_internal_temp",
        translation_key="head_internal_temp",
        native_unit_of_measurement=UnitOfTemperature.CELSIUS,
        device_class=SensorDeviceClass.TEMPERATURE,
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda z, r: _round(z.head_internal_temp, 2),
        attr_fn=lambda z, r: {
            "device_setpoint": None if z.device_setpoint is None else round(z.device_setpoint, 2),
            "room_temp": None if z.temp is None else round(z.temp, 2),
            "track_delta_k": r.controller_state.zone_track_delta.get(z.config.zone_id),
        },
    ),
    ZoneSensorDescription(
        key="want",
        translation_key="zone_want",
        device_class=SensorDeviceClass.ENUM,
        options=["off", "demand", "helper"],
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda z, r: r.zone_want(z),
    ),
    ZoneSensorDescription(
        key="control_state",
        translation_key="control_state",
        device_class=SensorDeviceClass.ENUM,
        options=[
            "off",
            "demand",
            "helper",
            "park",
            "runout",
            "fan_assist",
            "shed",
            "conditioning",
            "manual",
        ],
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda z, r: r.zone_control_state(z),
        attr_fn=lambda z, r: r.zone_control_attrs(z),
    ),
    ZoneSensorDescription(
        key="track_delta",
        translation_key="track_delta",
        native_unit_of_measurement="K",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda z, r: _round(r.controller_state.zone_track_delta.get(z.config.zone_id), 2),
    ),
    ZoneSensorDescription(
        key="park_classification",
        translation_key="park_classification",
        device_class=SensorDeviceClass.ENUM,
        options=["unknown", "idle", "residual"],
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda z, r: z.park.classification,
        attr_fn=lambda z, r: {
            "park_samples": z.park.samples,
            "park_preferred_margin_k": r.controller_state.zone_park_preferred.get(
                z.config.zone_id, z.park.preferred_margin_k
            ),
            "park_active_ratio": (
                None if z.park.active_ratio is None else round(z.park.active_ratio, 3)
            ),
            "park_fan_only_ratio": (
                None if z.park.fan_only_ratio is None else round(z.park.fan_only_ratio, 3)
            ),
            "park_fan_type_min_margin_k": z.park.fan_type_min_margin_k(),
            "park_residual_max_margin_k": z.park.residual_max_margin_k(),
            "park_residual_edge_k": z.park.residual_edge_k(),
            "park_current_is_fan_type": z.park.current_is_fan_type(
                r.controller_state.zone_park_margin.get(z.config.zone_id)
            ),
            "park_margin_bins": z.park.margin_bins,
        },
    ),
    ZoneSensorDescription(
        key="park_margin",
        translation_key="park_margin",
        native_unit_of_measurement="K",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda z, r: _round(r.controller_state.zone_park_margin.get(z.config.zone_id), 2),
        attr_fn=lambda z, r: {
            "park_preferred_margin_k": r.controller_state.zone_park_preferred.get(
                z.config.zone_id, z.park.preferred_margin_k
            ),
            "park_fan_type_min_margin_k": z.park.fan_type_min_margin_k(),
            "park_residual_max_margin_k": z.park.residual_max_margin_k(),
            "park_residual_edge_k": z.park.residual_edge_k(),
            "park_current_is_fan_type": z.park.current_is_fan_type(
                r.controller_state.zone_park_margin.get(z.config.zone_id)
            ),
        },
    ),
    ZoneSensorDescription(
        key="park_extraction",
        translation_key="park_extraction",
        native_unit_of_measurement=UnitOfPower.WATT,
        device_class=SensorDeviceClass.POWER,
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda z, r: _round(z.park.extraction_w, 0),
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    runtime: AdaptiveComfortRuntime = hass.data[DOMAIN][entry.entry_id]
    async_add_entities(HouseSensor(runtime, description) for description in HOUSE_SENSORS)
    for zone in runtime.zones.values():
        async_add_entities(
            (ZoneSensor(runtime, zone, description) for description in ZONE_SENSORS),
            config_subentry_id=zone.config.zone_id,
        )


class HouseSensor(AdaptiveComfortEntity, SensorEntity):
    entity_description: HouseSensorDescription

    def __init__(
        self, runtime: AdaptiveComfortRuntime, description: HouseSensorDescription
    ) -> None:
        super().__init__(runtime, description.key)
        self.entity_description = description

    @property
    def native_value(self) -> Any:
        return self.entity_description.value_fn(self.runtime)

    @property
    def extra_state_attributes(self) -> dict | None:
        return self.entity_description.attr_fn(self.runtime)


class ZoneSensor(AdaptiveComfortZoneEntity, SensorEntity):
    entity_description: ZoneSensorDescription

    def __init__(
        self,
        runtime: AdaptiveComfortRuntime,
        zone,
        description: ZoneSensorDescription,
    ) -> None:
        super().__init__(runtime, zone, description.key)
        self.entity_description = description

    @property
    def native_value(self) -> Any:
        return self.entity_description.value_fn(self.zone, self.runtime)

    @property
    def extra_state_attributes(self) -> dict | None:
        return self.entity_description.attr_fn(self.zone, self.runtime)
