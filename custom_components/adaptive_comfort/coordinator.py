"""Runtime for Adaptive Comfort: sampling, estimation, control loop.

All math lives in core/; this module adapts Home Assistant state and
services to the pure snapshot/command interface.
"""

from __future__ import annotations

import logging
import time
from datetime import timedelta
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_TEMPERATURE, STATE_ON, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.util import dt as dt_util

from .const import (
    CONF_AREA,
    CONF_AREAS,
    CONF_BATTERY_POSITIVE_DISCHARGING,
    CONF_BATTERY_POWER,
    CONF_CONTRACTED_KVA,
    CONF_DEFAULT_TARGET,
    CONF_DOOR_SENSOR,
    CONF_GRID_POWER,
    CONF_HEADS,
    CONF_HEIGHT,
    CONF_HUMIDITY_SENSOR,
    CONF_INDOOR_FANS,
    CONF_KNOWN_LOADS,
    CONF_MULTISPLIT,
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
    SIGNAL_UPDATE,
    SUBENTRY_ROOM,
    SUBENTRY_ZONE,
)
from .core import comfort, controller, power, psychro
from .fans import fan_entities_on
from .presence import house_presence, presence_state
from .core.drift import DriftEstimator
from .core.power import BaselineModel, DrawEstimator
from .core.series import TimeSeries
from .core.thermal import DiurnalModel, ThermalModel, house_other_temperature
from .core.types import (
    MODE_COOL,
    MODE_FAN,
    MODE_HEAT,
    MODE_OFF,
    ROOM_TYPE_REGULAR,
    ROOM_TYPE_WET,
    STATE_COOLING,
    STATE_FAN_ONLY,
    STATE_HEATING,
    STATE_STANDBY,
    ControllerState,
    HouseSnapshot,
    RoomConfig,
    Settings,
    UnconditionedRoom,
    ZoneConfig,
    ZoneSnapshot,
)
from .storage import AdaptiveComfortStore

_LOGGER = logging.getLogger(__name__)

SAMPLE_INTERVAL = timedelta(seconds=60)
SAVE_INTERVAL = timedelta(minutes=15)
FORECAST_INTERVAL = timedelta(minutes=30)
FIT_STEP_S = 300.0  # 5-minute smoothed steps for the RC fit
STABLE_STATE_S = 600.0  # drift updates need >=10 min in a stable head state
ALL_OFF_SETTLE_S = 900.0  # free-response fit waits 15 min after heads stop
BASELINE_OFF_S = 600.0
EVENT_SETTLE_S = 240.0
QUASI_STEADY_K_H = 0.2
TRANSIENT_K_H = 0.5
COP_TABLE_MIN_SAMPLES = 20
FORECAST_HOURS = 24


def _float_state(hass: HomeAssistant, entity_id: str | None) -> float | None:
    if not entity_id:
        return None
    state = hass.states.get(entity_id)
    if state is None or state.state in (STATE_UNKNOWN, STATE_UNAVAILABLE):
        return None
    try:
        return float(state.state)
    except (ValueError, TypeError):
        return None


def _head_state(hass: HomeAssistant, entity_id: str) -> str:
    """Classify a climate entity into a drift/operating state."""
    state = hass.states.get(entity_id)
    if state is None or state.state in (STATE_UNKNOWN, STATE_UNAVAILABLE, "off"):
        return STATE_STANDBY
    action = state.attributes.get("hvac_action")
    if action == "cooling":
        return STATE_COOLING
    if action == "heating":
        return STATE_HEATING
    if action == "fan":
        return STATE_FAN_ONLY
    if action in ("idle", "off"):
        return STATE_STANDBY
    if state.state == "cool":
        return STATE_COOLING
    if state.state == "heat":
        return STATE_HEATING
    if state.state == "fan_only":
        return STATE_FAN_ONLY
    return STATE_STANDBY


class ZoneRuntime:
    """Live state for one zone."""

    def __init__(self, config: ZoneConfig) -> None:
        self.config = config
        self.model = ThermalModel(config.sensed_room.volume_m3)
        self.drift: dict[str, DriftEstimator] = {head: DriftEstimator() for head in config.heads}
        self.temp_series = TimeSeries()
        self.w_series = TimeSeries(horizon_s=4 * 3600.0)
        self.head_state = STATE_STANDBY
        self.head_state_since = 0.0
        self.all_off_since: float | None = 0.0
        self.last_fit_ts: float | None = None
        self.last_fit_temp: float | None = None
        self.sensible_w = 0.0  # sensed room, signed
        self.latent_w = 0.0  # sensed room
        self.moisture_sources_kg_h = 0.0
        self.temp: float | None = None
        self.rh: float | None = None
        self.door_open = False
        self.indoor_fans_on = False
        self.outdoor_exhaust_on = False
        self.occupied: bool | None = None
        self.free_float: tuple[float, ...] = ()
        self.pred_60m: float | None = None
        self.allocated_w = 0.0

    @property
    def is_on(self) -> bool:
        return self.head_state in (STATE_COOLING, STATE_HEATING)

    def to_dict(self) -> dict:
        return {
            "thermal": self.model.to_dict(),
            "drift": {h: d.to_dict() for h, d in self.drift.items()},
            "moisture_sources_kg_h": self.moisture_sources_kg_h,
        }

    def restore(self, data: dict) -> None:
        if "thermal" in data:
            self.model = ThermalModel.from_dict(data["thermal"], self.config.sensed_room.volume_m3)
        for head, drift_data in data.get("drift", {}).items():
            if head in self.drift:
                self.drift[head] = DriftEstimator.from_dict(drift_data)
        self.moisture_sources_kg_h = float(data.get("moisture_sources_kg_h", 0.0))


class AdaptiveComfortRuntime:
    """Owns sampling, estimation and the 60 s control loop for one entry."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self.entry = entry
        self.store = AdaptiveComfortStore(hass, entry.entry_id)
        self.settings = Settings(
            target=entry.data.get(CONF_DEFAULT_TARGET, DEFAULT_TARGET_C),
            contracted_kva=entry.data.get(CONF_CONTRACTED_KVA, DEFAULT_CONTRACTED_KVA),
            power_factor=entry.data.get(CONF_POWER_FACTOR, 1.0),
        )
        self.multisplit: bool = entry.data.get(CONF_MULTISPLIT, True)
        self.settings.coordination = self.multisplit
        self.settings.multisplit = self.multisplit
        self.zones: dict[str, ZoneRuntime] = {}
        self.rooms: list[UnconditionedRoom] = []
        self._build_zones()
        self.outdoor_source: str = "none"

        self.baseline = BaselineModel()
        self.draws = DrawEstimator()
        self.outdoor_diurnal = DiurnalModel()
        self.controller_state = ControllerState()
        self.t_rm: float | None = None
        self.t_out: float | None = None
        self.house_occupied: bool | None = None
        self.p_grid: float | None = None
        self.p_load: float | None = None
        self.p_ac: float | None = None
        self.p_grid_series = TimeSeries(horizon_s=2 * 3600.0)
        self.p_load_series = TimeSeries(horizon_s=2 * 3600.0)
        self.grid_over_since: float | None = None
        self.forecast: list[float] = []
        self._forecast_ts = 0.0
        self.cop_table: dict[int, tuple[float, int]] = {}  # heads -> (ewma cop, samples)
        self.house_cop: float | None = None
        self.free_float_bias: float | None = None
        self.warm_excess_kh: float | None = None
        self.cold_deficit_kh: float | None = None
        self.window_suggestions: list[str] = []
        self.mode_source: str = "off"
        self.last_diag: dict = {}
        self._pending_events: list[dict] = []
        self._all_events: list[tuple[float, str]] = []
        self._all_off_since: float | None = 0.0
        self._last_sample_ts: float | None = None
        self._unsub: list = []

    # -- config parsing -------------------------------------------------------

    def _build_zones(self) -> None:
        self.zones.clear()
        self.rooms = []
        for subentry_id, subentry in self.entry.subentries.items():
            if subentry.subentry_type == SUBENTRY_ROOM:
                self.rooms.append(
                    UnconditionedRoom(
                        name=subentry.title,
                        area_m2=float(subentry.data.get(CONF_AREA, 6.0)),
                        room_type=subentry.data.get(CONF_ROOM_TYPE, ROOM_TYPE_REGULAR),
                        height_m=subentry.data.get(CONF_HEIGHT, DEFAULT_CEILING_HEIGHT_M),
                        temp_sensor=subentry.data.get(CONF_TEMP_SENSOR),
                    )
                )
                continue
            if subentry.subentry_type != SUBENTRY_ZONE:
                continue
            data = subentry.data
            heads = tuple(data.get(CONF_HEADS, []))
            areas = list(data.get(CONF_AREAS, []))
            if not heads or not areas:
                continue
            height = data.get(CONF_HEIGHT, DEFAULT_CEILING_HEIGHT_M)
            rooms = tuple(RoomConfig(area_m2=float(a), height_m=height) for a in areas)
            config = ZoneConfig(
                zone_id=subentry_id,
                name=subentry.title,
                heads=heads,
                rooms=rooms,
                temp_sensor=data.get(CONF_TEMP_SENSOR),
                humidity_sensor=data.get(CONF_HUMIDITY_SENSOR),
                door_sensor=data.get(CONF_DOOR_SENSOR),
                presence_sensor=data.get(CONF_PRESENCE_SENSOR),
                indoor_fan_entities=tuple(data.get(CONF_INDOOR_FANS, ())),
                outdoor_exhaust_fan_entities=tuple(data.get(CONF_OUTDOOR_EXHAUST_FANS, ())),
            )
            self.zones[subentry_id] = ZoneRuntime(config)

    # -- lifecycle ------------------------------------------------------------

    async def async_setup(self) -> None:
        data = await self.store.async_load()
        self._restore(data)
        self._unsub.append(async_track_time_interval(self.hass, self._async_tick, SAMPLE_INTERVAL))
        self._unsub.append(async_track_time_interval(self.hass, self._async_save, SAVE_INTERVAL))

    async def async_unload(self) -> None:
        for unsub in self._unsub:
            unsub()
        self._unsub.clear()
        await self.store.async_save(self._persist())

    def _restore(self, data: dict) -> None:
        if not data:
            return
        settings = data.get("settings", {})
        for key in (
            "target",
            "band_k",
            "min_on_min",
            "min_off_min",
            "contracted_kva",
            "power_factor",
            "shed_start_pct",
            "shed_restore_pct",
            "adaptive_blend",
        ):
            if key in settings:
                setattr(self.settings, key, float(settings[key]))
        for key in (
            "coordination",
            "presence_adaptation",
            "shedding_enabled",
            "fan_assist",
            "window_suggest",
        ):
            if key in settings:
                setattr(self.settings, key, bool(settings[key]))
        if "hvac_mode" in settings:
            self.settings.hvac_mode = settings["hvac_mode"]
        if "preset" in settings:
            self.settings.preset = settings["preset"]
        self.settings.zone_offsets = dict(settings.get("zone_offsets", {}))
        self.settings.zone_enabled = dict(settings.get("zone_enabled", {}))
        if "t_rm" in data and data["t_rm"] is not None:
            self.t_rm = float(data["t_rm"])
        if "baseline" in data:
            self.baseline = BaselineModel.from_dict(data["baseline"])
        if "draws" in data:
            self.draws = DrawEstimator.from_dict(data["draws"])
        if "outdoor_diurnal" in data:
            self.outdoor_diurnal = DiurnalModel.from_dict(data["outdoor_diurnal"])
        if "controller" in data:
            self.controller_state = ControllerState.from_dict(data["controller"])
        for zone_id, zone_data in data.get("zones", {}).items():
            if zone_id in self.zones:
                self.zones[zone_id].restore(zone_data)
        for key, value in data.get("cop_table", {}).items():
            try:
                self.cop_table[int(key)] = (float(value[0]), int(value[1]))
            except (TypeError, ValueError, IndexError):
                continue

    def _persist(self) -> dict:
        s = self.settings
        return {
            "settings": {
                "target": s.target,
                "band_k": s.band_k,
                "min_on_min": s.min_on_min,
                "min_off_min": s.min_off_min,
                "contracted_kva": s.contracted_kva,
                "power_factor": s.power_factor,
                "shed_start_pct": s.shed_start_pct,
                "shed_restore_pct": s.shed_restore_pct,
                "adaptive_blend": s.adaptive_blend,
                "coordination": s.coordination,
                "presence_adaptation": s.presence_adaptation,
                "shedding_enabled": s.shedding_enabled,
                "fan_assist": s.fan_assist,
                "window_suggest": s.window_suggest,
                "hvac_mode": s.hvac_mode,
                "preset": s.preset,
                "zone_offsets": s.zone_offsets,
                "zone_enabled": s.zone_enabled,
            },
            "t_rm": self.t_rm,
            "baseline": self.baseline.to_dict(),
            "draws": self.draws.to_dict(),
            "outdoor_diurnal": self.outdoor_diurnal.to_dict(),
            "controller": self.controller_state.to_dict(),
            "zones": {zid: zone.to_dict() for zid, zone in self.zones.items()},
            "cop_table": {str(k): [v[0], v[1]] for k, v in self.cop_table.items()},
        }

    async def _async_save(self, _now=None) -> None:
        await self.store.async_save(self._persist())

    def request_save(self) -> None:
        self.store.async_delay_save(self._persist(), 60.0)

    def notify(self) -> None:
        async_dispatcher_send(self.hass, f"{SIGNAL_UPDATE}_{self.entry.entry_id}")

    # -- helpers ---------------------------------------------------------------

    def _local_hour(self) -> float:
        now = dt_util.now()
        return now.hour + now.minute / 60.0 + now.second / 3600.0

    def _read_outdoor(self) -> tuple[float | None, str]:
        """Outdoor temperature and its source: sensor > weather > climatology."""
        value = _float_state(self.hass, self.entry.data.get(CONF_OUTDOOR_TEMP))
        if value is not None:
            return value, "sensor"
        weather = self.entry.data.get(CONF_WEATHER)
        if weather:
            state = self.hass.states.get(weather)
            if state is not None:
                temp = state.attributes.get("temperature")
                if isinstance(temp, (int, float)):
                    return float(temp), "weather"
        # Fresh install with no outdoor data at all: use climatological
        # normals (Lisbon defaults) so seasonal logic stays sane.
        now = dt_util.now()
        return comfort.climatology_temp(now.month, self._local_hour()), "climatology"

    def _read_outdoor_rh(self) -> float | None:
        weather = self.entry.data.get(CONF_WEATHER)
        if weather:
            state = self.hass.states.get(weather)
            if state is not None:
                rh = state.attributes.get("humidity")
                if isinstance(rh, (int, float)):
                    return float(rh)
        return None

    def _volume_readings(self) -> dict[str, tuple[float, float]]:
        """Zone_id -> (temperature, volume m3) for all zones with a reading."""
        readings: dict[str, tuple[float, float]] = {}
        for zone in self.zones.values():
            if zone.temp is None:
                continue
            readings[zone.config.zone_id] = (zone.temp, zone.config.total_volume_m3)
        return readings

    def _aux_volume_readings(self) -> list[tuple[float, float]]:
        """Unconditioned-room (temp, volume) pairs for house-mean mixing."""
        aux: list[tuple[float, float]] = []
        for room in self.rooms:
            temp = _float_state(self.hass, room.temp_sensor)
            if temp is not None:
                aux.append((temp, room.volume_m3))
        return aux

    def _house_other_temp(self, zone_id: str) -> float | None:
        return house_other_temperature(
            zone_id, self._volume_readings(), self._aux_volume_readings()
        )

    def _zone_temp(self, zone: ZoneRuntime) -> float | None:
        """Corrected zone temperature: external sensor, else drift-corrected head."""
        external = _float_state(self.hass, zone.config.temp_sensor)
        if external is not None:
            return external
        readings = []
        for head in zone.config.heads:
            state = self.hass.states.get(head)
            if state is None:
                continue
            internal = state.attributes.get("current_temperature")
            if isinstance(internal, (int, float)):
                readings.append(
                    zone.drift[head].correct(float(internal), _head_state(self.hass, head))
                )
        if readings:
            return sum(readings) / len(readings)
        return None

    async def _async_refresh_forecast(self, now_ts: float) -> None:
        if now_ts - self._forecast_ts < FORECAST_INTERVAL.total_seconds():
            return
        self._forecast_ts = now_ts
        weather = self.entry.data.get(CONF_WEATHER)
        temps: list[float] = []
        if weather:
            try:
                response = await self.hass.services.async_call(
                    "weather",
                    "get_forecasts",
                    {"entity_id": weather, "type": "hourly"},
                    blocking=True,
                    return_response=True,
                )
                forecast = (response or {}).get(weather, {}).get("forecast", [])
                temps = [
                    float(item["temperature"])
                    for item in forecast[:FORECAST_HOURS]
                    if isinstance(item.get("temperature"), (int, float))
                ]
            except Exception:
                _LOGGER.debug("Hourly forecast unavailable from %s", weather, exc_info=True)
        if not temps:
            diurnal = self.outdoor_diurnal.forecast_hours(self._local_hour(), FORECAST_HOURS)
            if diurnal:
                temps = diurnal
            else:
                # Last resort: climatological normals (Lisbon defaults).
                now = dt_util.now()
                hour = self._local_hour()
                temps = [
                    comfort.climatology_temp(now.month, (hour + h) % 24.0)
                    for h in range(FORECAST_HOURS)
                ]
        if temps and self.t_out is not None:
            temps[0] = self.t_out  # anchor the horizon on the live reading
        self.forecast = temps

    # -- main loop --------------------------------------------------------------

    async def _async_tick(self, _now=None) -> None:
        now_ts = time.time()
        local_hour = self._local_hour()
        dt_h = (now_ts - self._last_sample_ts) / 3600.0 if self._last_sample_ts else 1.0 / 60.0
        self._last_sample_ts = now_ts

        self._sample_house(now_ts, local_hour, dt_h)
        await self._async_refresh_forecast(now_ts)
        self._sample_zones(now_ts, local_hour)
        self._process_power_events(now_ts)
        self._update_estimators(now_ts, local_hour)
        await self._async_control(now_ts, local_hour)
        self.notify()

    def _sample_house(self, now_ts: float, local_hour: float, dt_h: float) -> None:
        data = self.entry.data
        self.p_grid = _float_state(self.hass, data.get(CONF_GRID_POWER))
        battery = _float_state(self.hass, data.get(CONF_BATTERY_POWER))
        known = [
            v
            for eid in data.get(CONF_KNOWN_LOADS, [])
            if (v := _float_state(self.hass, eid)) is not None
        ]
        if self.p_grid is not None:
            self.p_grid_series.append(now_ts, self.p_grid)
            self.p_load = power.compose_load(
                self.p_grid,
                battery,
                data.get(CONF_BATTERY_POSITIVE_DISCHARGING, True),
                known,
            )
            self.p_load_series.append(now_ts, self.p_load)
            threshold = self.settings.shed_start_pct * self.settings.limit_w
            if self.p_grid > threshold:
                if self.grid_over_since is None:
                    self.grid_over_since = now_ts
            else:
                self.grid_over_since = None

        self.t_out, self.outdoor_source = self._read_outdoor()
        if self.t_rm is None:
            # Seed the running mean from climatology rather than a single
            # instantaneous reading (which may be a mid-afternoon extreme).
            self.t_rm = comfort.climatology_mean(dt_util.now().month)
        if self.t_out is not None and self.outdoor_source != "climatology":
            self.t_rm = comfort.update_running_mean(self.t_rm, self.t_out, dt_h)
            self.outdoor_diurnal.update(local_hour, self.t_out)
        self.house_occupied = house_presence(
            self.hass,
            data.get(CONF_PRESENCE),
            [zone.config.presence_sensor for zone in self.zones.values()],
        )

    def _sample_zones(self, now_ts: float, local_hour: float) -> None:
        any_on = False
        for zone in self.zones.values():
            cfg = zone.config
            # Head operating state (mirrored heads: most active wins).
            states = [_head_state(self.hass, head) for head in cfg.heads]
            priority = [STATE_COOLING, STATE_HEATING, STATE_FAN_ONLY, STATE_STANDBY]
            new_state = next((p for p in priority if p in states), STATE_STANDBY)
            if new_state != zone.head_state:
                if (new_state in (STATE_COOLING, STATE_HEATING)) != zone.is_on:
                    mode = MODE_HEAT if new_state == STATE_HEATING else MODE_COOL
                    if zone.is_on:  # turning off: attribute mode from previous state
                        mode = MODE_HEAT if zone.head_state == STATE_HEATING else MODE_COOL
                    self._pending_events.append({"ts": now_ts, "zone": cfg.zone_id, "mode": mode})
                    self._all_events.append((now_ts, cfg.zone_id))
                zone.head_state = new_state
                zone.head_state_since = now_ts
            if zone.head_state == STATE_STANDBY:
                if zone.all_off_since is None:
                    zone.all_off_since = now_ts
            else:
                zone.all_off_since = None
                any_on = zone.is_on or any_on

            zone.temp = self._zone_temp(zone)
            if zone.temp is not None:
                zone.temp_series.append(now_ts, zone.temp)
            zone.rh = _float_state(self.hass, cfg.humidity_sensor)
            if zone.rh is not None and zone.temp is not None:
                zone.w_series.append(now_ts, psychro.humidity_ratio(zone.temp, zone.rh))
            door = self.hass.states.get(cfg.door_sensor) if cfg.door_sensor else None
            zone.door_open = door is not None and door.state == STATE_ON
            zone.indoor_fans_on = fan_entities_on(self.hass, cfg.indoor_fan_entities)
            zone.outdoor_exhaust_on = fan_entities_on(
                self.hass, cfg.outdoor_exhaust_fan_entities
            )
            zone.occupied = presence_state(self.hass, cfg.presence_sensor)

            # Drift learning: stable state >= 10 min with an external reference.
            external = _float_state(self.hass, cfg.temp_sensor)
            if external is not None and now_ts - zone.head_state_since >= STABLE_STATE_S:
                for head in cfg.heads:
                    state = self.hass.states.get(head)
                    if state is None:
                        continue
                    internal = state.attributes.get("current_temperature")
                    if isinstance(internal, (int, float)):
                        zone.drift[head].update(zone.head_state, float(internal), external)

        if any_on:
            self._all_off_since = None
        elif self._all_off_since is None:
            self._all_off_since = now_ts

        # Baseline learning while every head has been off long enough.
        if (
            self.p_load is not None
            and self._all_off_since is not None
            and now_ts - self._all_off_since >= BASELINE_OFF_S
        ):
            self.baseline.update(local_hour, self.p_load)

    def _process_power_events(self, now_ts: float) -> None:
        remaining: list[dict] = []
        for event in self._pending_events:
            age = now_ts - event["ts"]
            if age < EVENT_SETTLE_S:
                remaining.append(event)
                continue
            # Reject events with a neighbour too close to isolate the step.
            isolated = all(
                abs(other_ts - event["ts"]) < 1.0 or abs(other_ts - event["ts"]) > EVENT_SETTLE_S
                for other_ts, _zid in self._all_events
            )
            if isolated:
                pre = self.p_load_series.window(event["ts"] - 240.0, event["ts"] - 15.0)
                post = self.p_load_series.window(event["ts"] + 60.0, event["ts"] + 240.0)
                delta = power.measure_step(pre, post)
                if delta is not None:
                    self.draws.add_event(event["zone"], event["mode"], delta)
        self._pending_events = remaining
        self._all_events = [
            (ts, zid) for ts, zid in self._all_events if now_ts - ts < 2 * EVENT_SETTLE_S
        ]

    def _update_estimators(self, now_ts: float, local_hour: float) -> None:
        # Continuous AC power estimate and per-zone allocation.
        active = [z for z in self.zones.values() if z.is_on]
        self.p_ac = None
        if self.p_load is not None:
            if active:
                self.p_ac = power.estimate_ac_power(self.p_load, self.baseline.value(local_hour))
            else:
                self.p_ac = 0.0
        allocations: dict[str, float] = {}
        if self.p_ac and active:
            mode = MODE_HEAT if any(z.head_state == STATE_HEATING for z in active) else MODE_COOL
            allocations = power.allocate_power(
                self.p_ac,
                [
                    (
                        z.config.zone_id,
                        self.draws.draw_w(z.config.zone_id, mode) or 500.0,
                        abs((z.temp or 0.0) - comfort.band_center(self.settings, self.t_rm)),
                    )
                    for z in active
                ],
            )
        for zone in self.zones.values():
            zone.allocated_w = allocations.get(zone.config.zone_id, 0.0)

        total_sensible = 0.0
        total_latent = 0.0
        for zone in self.zones.values():
            self._update_zone_estimators(zone, now_ts, local_hour)
            if zone.is_on:
                total_sensible += abs(zone.sensible_w) * zone.config.n_rooms
                total_latent += zone.latent_w * zone.config.n_rooms

        # Empirical COP table by active head count.
        if self.p_ac and self.p_ac > 100.0 and active:
            n_heads = sum(z.config.n_rooms for z in active)
            house_cop = (total_sensible + total_latent) / self.p_ac
            if 0.3 <= house_cop <= 8.0:
                self.house_cop = house_cop
                prev, count = self.cop_table.get(n_heads, (house_cop, 0))
                self.cop_table[n_heads] = (prev + 0.05 * (house_cop - prev), count + 1)

    def _update_zone_estimators(self, zone: ZoneRuntime, now_ts: float, local_hour: float) -> None:
        if zone.temp is None:
            return
        # 5-minute smoothed temperature step.
        smoothed = zone.temp_series.mean_window(now_ts - 150.0, now_ts + 1.0)
        if smoothed is None:
            return
        if zone.last_fit_ts is None:
            zone.last_fit_ts = now_ts
            zone.last_fit_temp = smoothed
            return
        gap = now_ts - zone.last_fit_ts
        if gap < FIT_STEP_S:
            return
        prev_temp = zone.last_fit_temp
        zone.last_fit_ts = now_ts
        zone.last_fit_temp = smoothed
        if prev_temp is None or self.t_out is None or gap > 2.0 * FIT_STEP_S:
            return
        dt_h = gap / 3600.0
        dtdt = (smoothed - prev_temp) / dt_h

        if zone.all_off_since is not None and now_ts - zone.all_off_since >= ALL_OFF_SETTLE_S:
            # Only fit against measured outdoor data; climatology estimates
            # would corrupt the exchange-constant identification.
            if self.outdoor_source != "climatology":
                zone.model.update_free(
                    prev_temp,
                    smoothed,
                    self.t_out,
                    dt_h,
                    local_hour,
                    zone.door_open,
                    self._house_other_temp(zone.config.zone_id),
                    indoor_fans_on=zone.indoor_fans_on,
                    outdoor_exhaust_on=zone.outdoor_exhaust_on,
                )
            zone.sensible_w = 0.0
            zone.latent_w = 0.0
            self._update_moisture_baseline(zone, dt_h)
            return

        if not zone.is_on or now_ts - zone.head_state_since < STABLE_STATE_S:
            return

        # Conditioning: close COP / capacitance and derive live heat flows.
        per_head_w = zone.allocated_w / zone.config.n_rooms if zone.allocated_w else 0.0
        heating = zone.head_state == STATE_HEATING
        t_house = self._house_other_temp(zone.config.zone_id)
        if per_head_w > 50.0 and self.outdoor_source != "climatology":
            if abs(dtdt) < QUASI_STEADY_K_H:
                zone.model.update_cop(
                    per_head_w, smoothed, self.t_out, local_hour, zone.door_open, t_house
                )
            elif abs(dtdt) > TRANSIENT_K_H:
                zone.model.update_c_eff(
                    per_head_w,
                    heating,
                    smoothed,
                    self.t_out,
                    dtdt,
                    local_hour,
                    zone.door_open,
                    t_house,
                )
        zone.sensible_w = zone.model.sensible_power_w(
            smoothed,
            self.t_out,
            dtdt,
            local_hour,
            zone.door_open,
            t_house,
            indoor_fans_on=zone.indoor_fans_on,
            outdoor_exhaust_on=zone.outdoor_exhaust_on,
        )
        zone.latent_w = self._latent_power(zone, dt_h)

    def _update_moisture_baseline(self, zone: ZoneRuntime, dt_h: float) -> None:
        """Learn indoor moisture generation while the AC is off."""
        if len(zone.w_series) < 2:
            return
        latest = zone.w_series.latest
        earlier = zone.w_series.value_at(latest[0] - FIT_STEP_S) if latest else None
        if latest is None or earlier is None:
            return
        dw_dt = (latest[1] - earlier) / (FIT_STEP_S / 3600.0)
        w_out = self._outdoor_humidity_ratio()
        infiltration = 0.0
        if w_out is not None:
            infiltration = psychro.AIR_DENSITY_KG_M3 * zone.model.airflow_m3h * (w_out - latest[1])
        storage = psychro.AIR_DENSITY_KG_M3 * zone.config.sensed_room.volume_m3 * dw_dt
        sources = max(0.0, storage - infiltration)
        zone.moisture_sources_kg_h += 0.05 * (sources - zone.moisture_sources_kg_h)

    def _outdoor_humidity_ratio(self) -> float | None:
        rh = self._read_outdoor_rh()
        if rh is None or self.t_out is None:
            return None
        return psychro.humidity_ratio(self.t_out, rh)

    def _latent_power(self, zone: ZoneRuntime, dt_h: float) -> float:
        if zone.head_state != STATE_COOLING or len(zone.w_series) < 2:
            return 0.0
        latest = zone.w_series.latest
        earlier = zone.w_series.value_at(latest[0] - FIT_STEP_S) if latest else None
        if latest is None or earlier is None:
            return 0.0
        dw_dt = (latest[1] - earlier) / (FIT_STEP_S / 3600.0)
        w_out = self._outdoor_humidity_ratio()
        removed = psychro.moisture_removal_kg_h(
            zone.config.sensed_room.volume_m3,
            zone.model.airflow_m3h,
            latest[1],
            w_out if w_out is not None else latest[1],
            dw_dt,
            zone.moisture_sources_kg_h,
        )
        return psychro.latent_power_w(removed)

    # -- control -----------------------------------------------------------------

    def _build_snapshot(self, now_ts: float, local_hour: float) -> HouseSnapshot:
        zone_snaps: list[ZoneSnapshot] = []
        forecast = self.forecast or ([self.t_out] * FORECAST_HOURS if self.t_out else [])
        volume_readings = self._volume_readings()
        aux_readings = self._aux_volume_readings()
        for zone in self.zones.values():
            free_float: tuple[float, ...] = ()
            pred_60m = None
            if zone.temp is not None and forecast:
                zid = zone.config.zone_id
                t_house = house_other_temperature(zid, volume_readings, aux_readings)
                trajectory = zone.model.predict_free(
                    zone.temp,
                    forecast,
                    local_hour,
                    hours=FORECAST_HOURS,
                    door_open=zone.door_open,
                    t_house_other=t_house,
                    indoor_fans_on=zone.indoor_fans_on,
                    outdoor_exhaust_on=zone.outdoor_exhaust_on,
                )
                free_float = tuple(trajectory)
                if len(trajectory) > 1:
                    pred_60m = trajectory[1]
            zone.free_float = free_float
            zone.pred_60m = pred_60m
            mode = MODE_HEAT if zone.head_state == STATE_HEATING else MODE_COOL
            zone_snaps.append(
                ZoneSnapshot(
                    zone_id=zone.config.zone_id,
                    name=zone.config.name,
                    n_rooms=zone.config.n_rooms,
                    temp=zone.temp,
                    rh=zone.rh,
                    door_open=zone.door_open,
                    occupied=zone.occupied,
                    is_on=zone.is_on,
                    head_state=zone.head_state,
                    pred_60m=pred_60m,
                    free_float=free_float,
                    confidence=zone.model.confidence(zone.door_open),
                    draw_w=self.draws.draw_w(zone.config.zone_id, mode),
                    enabled=self.settings.zone_enabled.get(zone.config.zone_id, True),
                )
            )
        cop_hints = {
            n: cop for n, (cop, count) in self.cop_table.items() if count >= COP_TABLE_MIN_SAMPLES
        }
        # Sensors in unconditioned rooms (e.g. a bathroom) give a fresh
        # install extra indoor evidence for cold-start mode arbitration.
        aux_indoor: list[tuple[float, float]] = []
        for room in self.rooms:
            temp = _float_state(self.hass, room.temp_sensor)
            if temp is not None:
                weight = 0.25 if room.room_type == ROOM_TYPE_WET else 0.5
                aux_indoor.append((temp, weight))
        return HouseSnapshot(
            now_ts=now_ts,
            local_hour=local_hour,
            settings=self.settings,
            zones=zone_snaps,
            t_out=self.t_out,
            t_rm=self.t_rm,
            house_occupied=self.house_occupied,
            p_grid=self.p_grid,
            p_grid_over_since=(
                now_ts - self.grid_over_since if self.grid_over_since is not None else None
            ),
            forecast_hours=tuple(forecast),
            cop_by_head_count=cop_hints,
            aux_indoor=tuple(aux_indoor),
        )

    async def _async_control(self, now_ts: float, local_hour: float) -> None:
        snapshot = self._build_snapshot(now_ts, local_hour)
        decision = controller.tick(snapshot, self.controller_state)
        self.last_diag = decision.diag
        self.mode_source = decision.diag.get("mode_source", "off")
        self.warm_excess_kh = decision.diag.get("warm_excess_kh")
        self.cold_deficit_kh = decision.diag.get("cold_deficit_kh")
        self.window_suggestions = decision.window_suggestions
        self._update_free_float_bias(snapshot)
        for command in decision.commands:
            await self._async_execute(command)

    def _update_free_float_bias(self, snapshot: HouseSnapshot) -> None:
        center = comfort.band_center(self.settings, self.t_rm)
        biases = []
        for zone in snapshot.zones:
            if zone.free_float:
                mean_traj = sum(zone.free_float) / len(zone.free_float)
                biases.append(mean_traj - center)
        self.free_float_bias = sum(biases) / len(biases) if biases else None

    async def _async_execute(self, command) -> None:
        zone = self.zones.get(command.zone_id)
        if zone is None:
            return
        for head in zone.config.heads:
            state = self.hass.states.get(head)
            if state is None:
                continue
            try:
                if command.hvac_mode == MODE_OFF:
                    await self.hass.services.async_call(
                        "climate",
                        "set_hvac_mode",
                        {"entity_id": head, "hvac_mode": "off"},
                        blocking=False,
                    )
                    continue
                if command.hvac_mode == MODE_FAN:
                    # Fan assist: only if the head actually supports fan_only.
                    supported = state.attributes.get("hvac_modes") or []
                    if "fan_only" in supported:
                        await self.hass.services.async_call(
                            "climate",
                            "set_hvac_mode",
                            {"entity_id": head, "hvac_mode": "fan_only"},
                            blocking=False,
                        )
                    continue
                # Translate the room setpoint into the head's internal-sensor
                # coordinates and quantize to device steps.
                active = STATE_HEATING if command.hvac_mode == MODE_HEAT else STATE_COOLING
                offset = zone.drift[head].offset(active)
                minimum = state.attributes.get("min_temp", 16.0)
                maximum = state.attributes.get("max_temp", 30.0)
                setpoint = controller.quantize_setpoint(
                    (command.setpoint or self.settings.target) + offset,
                    float(minimum),
                    float(maximum),
                )
                if state.state != command.hvac_mode:
                    await self.hass.services.async_call(
                        "climate",
                        "set_hvac_mode",
                        {"entity_id": head, "hvac_mode": command.hvac_mode},
                        blocking=False,
                    )
                await self.hass.services.async_call(
                    "climate",
                    "set_temperature",
                    {"entity_id": head, ATTR_TEMPERATURE: setpoint},
                    blocking=False,
                )
            except Exception:
                _LOGGER.exception("Failed to command %s", head)

    # -- entity-facing helpers ------------------------------------------------------

    @property
    def power_headroom_w(self) -> float | None:
        if self.p_grid is None:
            return None
        return self.settings.limit_w - self.p_grid

    @property
    def shedding_active(self) -> bool:
        return bool(self.controller_state.shed)

    @property
    def total_sensible_w(self) -> float:
        return sum(abs(z.sensible_w) * z.config.n_rooms for z in self.zones.values() if z.is_on)

    @property
    def total_latent_w(self) -> float:
        return sum(z.latent_w * z.config.n_rooms for z in self.zones.values() if z.is_on)

    @property
    def model_confidence(self) -> float:
        if not self.zones:
            return 0.0
        return sum(z.model.confidence(z.door_open) for z in self.zones.values()) / len(self.zones)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "settings": self._persist()["settings"],
            "t_out": self.t_out,
            "t_rm": self.t_rm,
            "p_grid": self.p_grid,
            "p_load": self.p_load,
            "p_ac": self.p_ac,
            "forecast": self.forecast,
            "cop_table": {str(k): v for k, v in self.cop_table.items()},
            "controller_diag": self.last_diag,
            "zones": {
                zid: {
                    "name": z.config.name,
                    "temp": z.temp,
                    "head_state": z.head_state,
                    "door_open": z.door_open,
                    "indoor_fans_on": z.indoor_fans_on,
                    "outdoor_exhaust_on": z.outdoor_exhaust_on,
                    "coupling": self._coupling_diag(z),
                    "c_eff_wh_per_k": z.model.c_eff_wh_per_k,
                    "furniture_factor": z.model.furniture_factor,
                    "cop": z.model.cop,
                    "confidence": z.model.confidence(z.door_open),
                    "drift": {h: d.to_dict() for h, d in z.drift.items()},
                    "sensible_w": z.sensible_w,
                    "latent_w": z.latent_w,
                    "allocated_w": z.allocated_w,
                }
                for zid, z in self.zones.items()
            },
        }

    def _coupling_diag(self, zone: ZoneRuntime) -> dict[str, Any]:
        """Thermal couplings split by destination (outdoor vs house) and door regime.

        k values are in h⁻¹ (equivalent ACH for that leg); UA values in W/K.
        t_house_other is the volume-weighted mean of the *other* zones plus
        unconditioned-room sensors — the temperature the mixing leg pulls toward.
        """
        model = zone.model
        per_regime = {}
        for regime, label in ((False, "door_closed"), (True, "door_open")):
            per_regime[label] = {
                "k_out_h": round(model.k(regime), 4),
                "k_mix_h": round(model.k_mix(regime), 4),
                "fit_samples": model.fit_samples(regime),
                "fit_stage": model.fit_stage(regime),
            }
        active = zone.door_open
        return {
            "outdoor": {
                "k_out_h": round(model.k(active), 4),
                "ua_w_per_k": round(model.ua_w_per_k, 2),
                "description": "envelope exchange with outside air",
            },
            "house_mixing": {
                "k_mix_h": round(model.k_mix(active), 4),
                "ua_mix_w_per_k": round(model.ua_mix_w_per_k, 2),
                "t_house_other": self._house_other_temp(zone.config.zone_id),
                "description": "air mixing with other zones and unconditioned rooms",
            },
            "ach_total": round(model.ach, 4),
            "active_regime": "door_open" if active else "door_closed",
            "regimes": per_regime,
        }


@callback
def get_runtime(hass: HomeAssistant, entry_id: str) -> AdaptiveComfortRuntime:
    return hass.data[DOMAIN][entry_id]
