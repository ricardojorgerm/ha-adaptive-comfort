"""Runtime for Adaptive Comfort: sampling, estimation, control loop.

All math lives in core/; this module adapts Home Assistant state and
services to the pure snapshot/command interface.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import timedelta
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_TEMPERATURE, STATE_ON, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.event import async_track_state_change_event, async_track_time_interval
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
from .core import comfort, controller, park, power, psychro
from .core.drift import DriftEstimator
from .core.power import BaselineModel, DrawEstimator
from .core.series import TimeSeries
from .core.thermal import DiurnalModel, ThermalModel, house_other_temperature
from .core.types import (
    MODE_COOL,
    MODE_FAN,
    MODE_HEAT,
    MODE_OFF,
    PRESET_MANUAL,
    PRESET_NONE,
    ROOM_TYPE_REGULAR,
    ROOM_TYPE_WET,
    STATE_COOLING,
    STATE_FAN_ONLY,
    STATE_HEATING,
    STATE_STANDBY,
    Command,
    ControllerState,
    HouseSnapshot,
    RoomConfig,
    Settings,
    UnconditionedRoom,
    ZoneConfig,
    ZoneSnapshot,
)
from .fans import fan_entities_on
from .presence import house_presence, presence_state
from .storage import AdaptiveComfortStore

_LOGGER = logging.getLogger(__name__)

SAMPLE_INTERVAL = timedelta(seconds=60)
POWER_REACT_DEBOUNCE_S = 2.0
SAVE_INTERVAL = timedelta(minutes=15)
FORECAST_INTERVAL = timedelta(minutes=30)
FIT_STEP_S = 300.0  # 5-minute smoothed steps for the RC fit
STABLE_STATE_S = 600.0  # drift updates need >=10 min in a stable head state
ALL_OFF_SETTLE_S = 900.0  # free-response fit waits 15 min after heads stop
BASELINE_OFF_S = 600.0
EVENT_SETTLE_S = 240.0
QUASI_STEADY_K_H = 0.2
TRANSIENT_K_H = 0.5
PARK_OBS_DELAY_S = 120.0  # parked observations start after this settle time
COP_TABLE_MIN_SAMPLES = 20
# House-COP publication: gate out low-power cycle edges (small heat flow over
# small power swings wildly) and smooth what the sensor reports.
HOUSE_COP_MIN_POWER_W = 250.0
HOUSE_COP_EMA_ALPHA = 0.2
FORECAST_HOURS = 24
# Hold last measured outdoor this long after a sensor/weather dropout before
# falling back to climatology (which then marks t_out_synthetic).
OUTDOOR_LKG_S = 2.0 * 3600.0
# Moisture baselines learned against (k_out+k_mix) are invalid after the
# outdoor-only airflow fix; wipe on restore from older schema.
MOISTURE_SCHEMA = 2
# House/zone COP learned with inflated latent must be re-fit after outdoor-only
# airflow; wipe tables and per-zone COP on restore from older schema.
COP_SCHEMA = 2


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
        self.sensible_ts: float | None = None  # when sensible_w was last fitted
        self.latent_w = 0.0  # sensed room
        # Zone-total free-float heat that must be rejected/supplied to hold T (W).
        self.standing_load_w: float | None = None
        self.park = park.ParkEstimator()
        self.park_power = park.PowerDebounce()
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
        # Live head-frame readings (for tracking/park diagnostics).
        self.head_internal_temp: float | None = None
        self.device_setpoint: float | None = None
        # Last controller command reason for this zone (demand/helper/park/…).
        self.last_control_reason: str | None = None

    @property
    def is_on(self) -> bool:
        return self.head_state in (STATE_COOLING, STATE_HEATING)

    def to_dict(self) -> dict:
        return {
            "park": self.park.to_dict(),
            "thermal": self.model.to_dict(),
            "drift": {h: d.to_dict() for h, d in self.drift.items()},
            "moisture_sources_kg_h": self.moisture_sources_kg_h,
            "moisture_schema": MOISTURE_SCHEMA,
        }

    def restore(self, data: dict) -> None:
        if "park" in data:
            self.park = park.ParkEstimator.from_dict(data["park"])
        if "thermal" in data:
            self.model = ThermalModel.from_dict(data["thermal"], self.config.sensed_room.volume_m3)
        for head, drift_data in data.get("drift", {}).items():
            if head in self.drift:
                self.drift[head] = DriftEstimator.from_dict(drift_data)
        # Inflated (k_out+k_mix) moisture EWMAs are not reusable.
        if int(data.get("moisture_schema", 0)) >= MOISTURE_SCHEMA:
            self.moisture_sources_kg_h = float(data.get("moisture_sources_kg_h", 0.0))
        else:
            self.moisture_sources_kg_h = 0.0


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
        self._prev_house_occupied: bool | None = None
        self._preset_before_manual: str = PRESET_NONE
        self.p_grid: float | None = None
        self.p_load: float | None = None
        self.p_ac: float | None = None
        self.p_demand: float | None = None
        self.known_load_total: float = 0.0
        self.shed_urgent: bool = False
        self._prev_known_total: float | None = None
        self._demand_spike_until: float = 0.0
        self._last_power_react: float = 0.0
        self.p_grid_series = TimeSeries(horizon_s=2 * 3600.0)
        self.p_load_series = TimeSeries(horizon_s=2 * 3600.0)
        self.grid_over_since: float | None = None
        self.forecast: list[float] = []
        self._forecast_ts = 0.0
        self._forecast_fetched_ts: float = 0.0
        self._t_out_lkg: float | None = None
        self._t_out_lkg_ts: float = 0.0
        self._t_out_synthetic: bool = False
        self._event_seq: int = 0
        self._control_lock = asyncio.Lock()
        self._prev_zone_enabled: dict[str, bool] = {}
        # True only when at least one *conditioning* zone refreshed sensible this tick.
        self._conditioning_fit_refreshed: bool = False
        self.cop_table: dict[int, tuple[float, int]] = {}  # heads -> (ewma cop, samples)
        # (heads, outdoor band) -> (ewma cop, samples); diagnostics/analysis only for now.
        self.cop_table_banded: dict[str, tuple[float, int]] = {}
        # control state ('conditioning'|'park'|'mixed') -> (ewma cop, samples)
        self.cop_table_state: dict[str, tuple[float, int]] = {}
        self.starts = power.StartCounter()
        self._last_state_key: str | None = None
        self.house_cop: float | None = None
        self.free_float_bias: float | None = None
        self.warm_excess_kh: float | None = None
        self.cold_deficit_kh: float | None = None
        self.window_suggestions: list[str] = []
        self.mode_source: str = "off"
        self.last_diag: dict = {}
        self._pending_events: list[dict] = []
        # (ts, zone_id, event_id) — isolation compares by id, not equal timestamps.
        self._all_events: list[tuple[float, str, int]] = []
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
            if len(areas) != len(heads):
                _LOGGER.warning(
                    "Zone %s: %d heads but %d areas; aligning to head count",
                    subentry.title,
                    len(heads),
                    len(areas),
                )
                if len(areas) < len(heads):
                    areas = areas + [areas[-1]] * (len(heads) - len(areas))
                else:
                    areas = areas[: len(heads)]
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
        watch = self._power_watch_entities()
        if watch:
            self._unsub.append(
                async_track_state_change_event(self.hass, watch, self._on_power_entity_change)
            )

    @callback
    def _on_power_entity_change(self, event) -> None:
        """React quickly when grid or a known load changes (oven, etc.)."""
        self.hass.async_create_task(self._async_power_react())

    async def _async_power_react(self) -> None:
        """Fast meter path: fresh heads+power, shed only on engage/release.

        Must not run estimator EWMAs or full controller.tick on every meter
        update — tracking-delta / park-margin adapt per tick and would saturate.
        """
        now_ts = time.time()
        if now_ts - self._last_power_react < POWER_REACT_DEBOUNCE_S:
            return
        self._last_power_react = now_ts
        local_hour = self._local_hour()
        self._sample_head_states(now_ts)
        self._sample_power(now_ts, local_hour)
        self._finalize_demand(now_ts)
        if self.settings.shedding_enabled:
            await self._async_control_if_shed_transition(now_ts, local_hour, skip_if_busy=True)
        # Always publish: ac_power_estimate / starts sensors should track the
        # meter at event rate, not wait for the 60 s control tick.
        self.notify()

    def _shed_needed_now(self) -> bool:
        over_s = time.time() - self.grid_over_since if self.grid_over_since is not None else None
        return power.shed_needed(
            self.p_demand,
            self.settings.limit_w,
            self.settings.shed_start_pct,
            over_s,
            urgent=self.shed_urgent,
        )

    def _shed_has_open_candidates(self) -> bool:
        """True when an on zone is not yet in the shed set (another victim possible)."""
        shed = self.controller_state.shed
        return any(z.is_on and z.config.zone_id not in shed for z in self.zones.values())

    def _shed_any_restore_possible(self) -> bool:
        """True when at least one shed zone would pass restore_allowed at current demand."""
        p_demand = self.p_demand
        if p_demand is None:
            return False
        s = self.settings
        for zid in self.controller_state.shed:
            zone = self.zones.get(zid)
            mode = MODE_HEAT if zone is not None and zone.head_state == STATE_HEATING else MODE_COOL
            draw = self.draws.draw_w(zid, mode)
            if power.restore_allowed(p_demand, s.limit_w, s.shed_restore_pct, draw):
                return True
        return False

    def _react_should_run_control(self) -> bool:
        """Whether power-react should enter controller.tick for shedding.

        Must not compare shed_needed to bool(state.shed): start (0.88) and restore
        (0.75) differ, so the hysteresis band would re-enter tick every meter event
        with no state change and saturate zone_track_delta.
        """
        needed = self._shed_needed_now()
        shedding = bool(self.controller_state.shed)
        now_ts = time.time()
        if needed:
            if not shedding:
                return True
            spacing = (
                controller.SHED_URGENT_SPACING_S
                if self.shed_urgent
                else controller.SHED_ACTION_SPACING_S
            )
            if now_ts - self.controller_state.last_shed_action < spacing:
                return False
            return self._shed_has_open_candidates()
        # Below start threshold: only run when a restore can actually proceed.
        return shedding and self._shed_any_restore_possible()

    async def _async_control_if_shed_transition(
        self, now_ts: float, local_hour: float, *, skip_if_busy: bool = False
    ) -> None:
        """Run control only when shed engage, add-victim, or actionable restore."""
        if not self._react_should_run_control():
            return
        await self._async_control(now_ts, local_hour, skip_if_busy=skip_if_busy)

    def _power_watch_entities(self) -> list[str]:
        data = self.entry.data
        entities: list[str] = []
        if grid := data.get(CONF_GRID_POWER):
            entities.append(grid)
        entities.extend(data.get(CONF_KNOWN_LOADS, []))
        return entities

    def _known_load_readings(self) -> list[float]:
        return [
            v
            for eid in self.entry.data.get(CONF_KNOWN_LOADS, [])
            if (v := _float_state(self.hass, eid)) is not None
        ]

    def _finalize_demand(self, now_ts: float) -> None:
        """Conservative contracted demand for shedding (handles meter lag)."""
        known = self._known_load_readings()
        self.known_load_total = sum(known)
        active = [z for z in self.zones.values() if z.is_on]
        mode = MODE_HEAT if any(z.head_state == STATE_HEATING for z in active) else MODE_COOL
        draws = [self.draws.draw_w(z.config.zone_id, mode) for z in active]
        # Empty active + house residual must not count as "active AC draw".
        ac_w = power.estimated_active_ac_draw_w(draws, self.p_ac if active else None)
        peak = self.p_grid_series.max_window(now_ts - power.SHED_PEAK_WINDOW_S, now_ts)
        self.p_demand = power.contracted_demand_w(self.p_grid, known, ac_w, peak)

        if (
            self._prev_known_total is not None
            and self.known_load_total - self._prev_known_total >= power.KNOWN_LOAD_SPIKE_W
        ):
            self._demand_spike_until = now_ts + 90.0
        self._prev_known_total = self.known_load_total

        threshold = self.settings.shed_start_pct * self.settings.limit_w
        if self.p_demand is not None and self.p_demand > threshold:
            if self.grid_over_since is None:
                self.grid_over_since = now_ts
        elif self.p_demand is None or self.p_demand <= threshold:
            peak_demand = power.contracted_demand_w(None, known, ac_w, peak)
            if peak_demand is None or peak_demand <= threshold:
                self.grid_over_since = None

        critical = power.SHED_CRITICAL_PCT * self.settings.limit_w
        self.shed_urgent = now_ts < self._demand_spike_until or (
            self.p_demand is not None and self.p_demand >= critical
        )

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
            "fan_floor_per_head_w",
        ):
            if key in settings:
                setattr(self.settings, key, float(settings[key]))
        for key in (
            "coordination",
            "presence_adaptation",
            "shedding_enabled",
            "fan_assist",
            "tracking",
            "park_learning",
            "auto_regime",
            "night_ventilate",
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
        # Seed so the first tick does not Off-spam zones already disabled.
        self._prev_zone_enabled = {
            zid: bool(self.settings.zone_enabled.get(zid, True)) for zid in self.zones
        }
        if "preset_before_manual" in data:
            self._preset_before_manual = str(data["preset_before_manual"] or PRESET_NONE)
        if "prev_house_occupied" in data:
            prev = data["prev_house_occupied"]
            self._prev_house_occupied = None if prev is None else bool(prev)
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
        # Prefer the higher of controller-working and zone-persisted park depth
        # so a restart cannot forget a learned margin.
        for zid, zone in self.zones.items():
            persisted = zone.park.preferred_margin_k
            working = self.controller_state.zone_park_preferred.get(zid)
            best = max(persisted, working) if working is not None else persisted
            self.controller_state.zone_park_preferred[zid] = best
            zone.park.preferred_margin_k = best
        if int(data.get("cop_schema", 0)) >= COP_SCHEMA:
            for key, value in data.get("cop_table", {}).items():
                try:
                    self.cop_table[int(key)] = (float(value[0]), int(value[1]))
                except (TypeError, ValueError, IndexError):
                    continue
            for key, value in data.get("cop_table_banded", {}).items():
                try:
                    self.cop_table_banded[str(key)] = (float(value[0]), int(value[1]))
                except (TypeError, ValueError, IndexError):
                    continue
            for key, value in data.get("cop_table_state", {}).items():
                try:
                    self.cop_table_state[str(key)] = (float(value[0]), int(value[1]))
                except (TypeError, ValueError, IndexError):
                    continue
        else:
            # Inflated-latent COP history is not reusable.
            self.cop_table.clear()
            self.cop_table_banded.clear()
            self.cop_table_state.clear()
            self.house_cop = None
            for zone in self.zones.values():
                zone.model.cop = None
                zone.model.cop_samples = 0
        if "starts" in data:
            self.starts = power.StartCounter.from_dict(data["starts"])

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
                "tracking": s.tracking,
                "park_learning": s.park_learning,
                "auto_regime": s.auto_regime,
                "night_ventilate": s.night_ventilate,
                "fan_floor_per_head_w": s.fan_floor_per_head_w,
                "window_suggest": s.window_suggest,
                "hvac_mode": s.hvac_mode,
                "preset": s.preset,
                "zone_offsets": s.zone_offsets,
                "zone_enabled": s.zone_enabled,
            },
            "preset_before_manual": self._preset_before_manual,
            "prev_house_occupied": self._prev_house_occupied,
            "t_rm": self.t_rm,
            "baseline": self.baseline.to_dict(),
            "draws": self.draws.to_dict(),
            "outdoor_diurnal": self.outdoor_diurnal.to_dict(),
            "controller": self.controller_state.to_dict(),
            "zones": {zid: zone.to_dict() for zid, zone in self.zones.items()},
            "cop_schema": COP_SCHEMA,
            "cop_table": {str(k): [v[0], v[1]] for k, v in self.cop_table.items()},
            "cop_table_banded": {k: [v[0], v[1]] for k, v in self.cop_table_banded.items()},
            "cop_table_state": {k: [v[0], v[1]] for k, v in self.cop_table_state.items()},
            "starts": self.starts.to_dict(),
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

    def _outdoor_configured(self) -> bool:
        data = self.entry.data
        return bool(data.get(CONF_OUTDOOR_TEMP) or data.get(CONF_WEATHER))

    def _read_outdoor(self) -> tuple[float | None, str]:
        """Outdoor temperature and its source: sensor > weather > LKG > climatology."""
        now_ts = time.time()
        value = _float_state(self.hass, self.entry.data.get(CONF_OUTDOOR_TEMP))
        if value is not None:
            self._t_out_lkg = value
            self._t_out_lkg_ts = now_ts
            self._t_out_synthetic = False
            return value, "sensor"
        weather = self.entry.data.get(CONF_WEATHER)
        if weather:
            state = self.hass.states.get(weather)
            if state is not None:
                temp = state.attributes.get("temperature")
                if isinstance(temp, (int, float)):
                    self._t_out_lkg = float(temp)
                    self._t_out_lkg_ts = now_ts
                    self._t_out_synthetic = False
                    return float(temp), "weather"
        # Configured outdoor dropped out: prefer last-known-good briefly.
        if (
            self._outdoor_configured()
            and self._t_out_lkg is not None
            and now_ts - self._t_out_lkg_ts <= OUTDOOR_LKG_S
        ):
            self._t_out_synthetic = False
            return self._t_out_lkg, "stale"
        # Climatology: OK for virgin installs (no outdoor entity). After a
        # configured source dropout it is synthetic for discretionary actuation.
        now = dt_util.now()
        self._t_out_synthetic = self._outdoor_configured()
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

    def _standing_load_w(self, zone) -> float | None:
        """Estimated heat inflow the zone must reject to hold temperature (W).

        The free-float side of sensible_power_w (dT/dt = 0, no AC term) at
        current conditions: what a parked/off zone gains per second. Used to
        judge whether learned trickle output can carry a satisfied zone.
        """
        if zone.temp is None or self.t_out is None:
            return None
        local_hour = self._local_hour()
        t_house = self._house_other_temp(zone.config.zone_id)
        # sensible_power_w returns heat *added by the AC*; with dtdt=0 its
        # negation is the standing inflow the AC must remove to hold temp.
        inflow = -zone.model.sensible_power_w(
            zone.temp, self.t_out, 0.0, local_hour, zone.door_open, t_house
        )
        # Cooling must reject heat coming in; heating must replace heat
        # going out. Same free-float number, opposite sign of interest.
        if self.controller_state.mode == MODE_HEAT:
            return max(0.0, -inflow) * zone.config.n_rooms
        return max(0.0, inflow) * zone.config.n_rooms

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
        self._forecast_fetched_ts = now_ts

    def _effective_forecast(self, now_ts: float) -> list[float]:
        """Re-index stored forecast so index 0 is 'now', anchored on live t_out."""
        if not self.forecast:
            return [self.t_out] * FORECAST_HOURS if self.t_out is not None else []
        elapsed_h = 0.0
        if self._forecast_fetched_ts > 0.0:
            elapsed_h = max(0.0, (now_ts - self._forecast_fetched_ts) / 3600.0)
        n = len(self.forecast)
        shifted: list[float] = []
        for h in range(FORECAST_HOURS):
            pos = elapsed_h + h
            idx = min(int(pos), n - 1)
            frac = min(pos - idx, 1.0)
            nxt = min(idx + 1, n - 1)
            shifted.append(self.forecast[idx] * (1.0 - frac) + self.forecast[nxt] * frac)
        if self.t_out is not None and shifted:
            shifted[0] = self.t_out
        return shifted

    # -- main loop --------------------------------------------------------------

    async def _async_tick(self, _now=None) -> None:
        now_ts = time.time()
        local_hour = self._local_hour()
        dt_h = (now_ts - self._last_sample_ts) / 3600.0 if self._last_sample_ts else 1.0 / 60.0
        self._last_sample_ts = now_ts

        self._sample_house(now_ts, local_hour, dt_h)
        await self._async_refresh_forecast(now_ts)
        self._sample_head_states(now_ts)
        # Power before environment so baseline learning sees a fresh p_load.
        self._sample_power(now_ts, local_hour)
        self._sample_zone_environment(now_ts, local_hour)
        self._process_power_events(now_ts)
        self._update_estimators(now_ts, local_hour)
        self._finalize_demand(now_ts)
        await self._async_apply_zone_disable_edges()
        await self._async_control(now_ts, local_hour)
        self.notify()

    def _sample_power(self, now_ts: float, local_hour: float) -> None:
        """Grid/load series + electrical p_ac + starts. Safe on the react path."""
        data = self.entry.data
        self.p_grid = _float_state(self.hass, data.get(CONF_GRID_POWER))
        battery = _float_state(self.hass, data.get(CONF_BATTERY_POWER))
        known = self._known_load_readings()
        if self.p_grid is not None:
            self.p_grid_series.append(now_ts, self.p_grid)
            self.p_load = power.compose_load(
                self.p_grid,
                battery,
                data.get(CONF_BATTERY_POSITIVE_DISCHARGING, True),
                known,
            )
            self.p_load_series.append(now_ts, self.p_load)
        self._estimate_p_ac(now_ts, local_hour)

    def _estimate_p_ac(self, now_ts: float, local_hour: float) -> None:
        """Electrical AC residual — never zeroed from hvac_action.

        Baseline learning stays gated on device quiet (_all_off_since). Trade-offs:
        unmodeled non-AC load above the fan floor can latch StartCounter (missed
        starts) or, with a learned slot and heads off, count phantom starts —
        preferred to manufacturing starts from hvac_action idle flicker.
        """
        active = [z for z in self.zones.values() if z.is_on]
        self.p_ac = None
        if self.p_load is not None:
            # Median fallback OK for demand/shedding; park path uses learned slot.
            self.p_ac = power.estimate_ac_power(self.p_load, self.baseline.value(local_hour))
        parked_ids = set(self.controller_state.zone_parked_since)
        self._last_state_key = power.control_state_key(
            any(z.config.zone_id not in parked_ids for z in active),
            any(z.config.zone_id in parked_ids for z in active),
        )
        # StartCounter: never invent compression with no heads open. Prefer a
        # learned baseline slot when counting starts while heads are active.
        if not active:
            starts_pac = 0.0
        else:
            learned = self.baseline.value(local_hour, fallback=False)
            if self.p_load is not None and learned is not None:
                starts_pac = power.estimate_ac_power(self.p_load, learned)
            else:
                starts_pac = self.p_ac
        self.starts.update(
            now_ts,
            starts_pac,
            self._last_state_key,
            floor_w=self._compression_floor_w(active, parked_ids),
        )

    def _park_p_ac(self, local_hour: float) -> float | None:
        """p_ac for park gating — only from a learned baseline slot."""
        if self.p_load is None:
            return None
        learned = self.baseline.value(local_hour, fallback=False)
        if learned is None:
            return None
        return power.estimate_ac_power(self.p_load, learned)

    def _sample_house(self, now_ts: float, local_hour: float, dt_h: float) -> None:
        """Tick-only: outdoor, t_rm, diurnal, presence. No power series."""
        data = self.entry.data
        self.t_out, self.outdoor_source = self._read_outdoor()
        if self.t_rm is None:
            # Seed the running mean from climatology rather than a single
            # instantaneous reading (which may be a mid-afternoon extreme).
            self.t_rm = comfort.climatology_mean(dt_util.now().month)
        if self.t_out is not None and self.outdoor_source != "climatology":
            self.t_rm = comfort.update_running_mean(self.t_rm, self.t_out, dt_h)
            self.outdoor_diurnal.update(local_hour, self.t_out)
        occupied = house_presence(
            self.hass,
            data.get(CONF_PRESENCE),
            [zone.config.presence_sensor for zone in self.zones.values()],
        )
        if (
            self.settings.presence_adaptation
            and self._prev_house_occupied is False
            and occupied is True
        ):
            self.reset_track_deltas()
        self._prev_house_occupied = occupied
        self.house_occupied = occupied

    def _sample_head_states(self, now_ts: float) -> None:
        """Cheap head-state scan + draw-event stamps. Both react and tick."""
        any_busy = False
        for zone in self.zones.values():
            cfg = zone.config
            states = [_head_state(self.hass, head) for head in cfg.heads]
            priority = [STATE_COOLING, STATE_HEATING, STATE_FAN_ONLY, STATE_STANDBY]
            new_state = next((p for p in priority if p in states), STATE_STANDBY)
            if new_state != zone.head_state:
                if (new_state in (STATE_COOLING, STATE_HEATING)) != zone.is_on:
                    mode = MODE_HEAT if new_state == STATE_HEATING else MODE_COOL
                    if zone.is_on:
                        mode = MODE_HEAT if zone.head_state == STATE_HEATING else MODE_COOL
                    self._event_seq += 1
                    eid = self._event_seq
                    self._pending_events.append(
                        {"ts": now_ts, "zone": cfg.zone_id, "mode": mode, "event_id": eid}
                    )
                    self._all_events.append((now_ts, cfg.zone_id, eid))
                zone.head_state = new_state
                zone.head_state_since = now_ts
            if zone.head_state == STATE_STANDBY:
                if zone.all_off_since is None:
                    zone.all_off_since = now_ts
            else:
                zone.all_off_since = None
                # Fan-only draws power — must block house baseline learning.
                any_busy = True
        if any_busy:
            self._all_off_since = None
        elif self._all_off_since is None:
            self._all_off_since = now_ts

    def _sample_zone_environment(self, now_ts: float, local_hour: float) -> None:
        """Tick-only: temps, series, doors/fans/presence, drift, baseline learn."""
        for zone in self.zones.values():
            cfg = zone.config
            zone.temp = self._zone_temp(zone)
            if zone.temp is not None:
                zone.temp_series.append(now_ts, zone.temp)
            zone.rh = _float_state(self.hass, cfg.humidity_sensor)
            if zone.rh is not None and zone.temp is not None:
                zone.w_series.append(now_ts, psychro.humidity_ratio(zone.temp, zone.rh))
            door = self.hass.states.get(cfg.door_sensor) if cfg.door_sensor else None
            zone.door_open = door is not None and door.state == STATE_ON
            zone.indoor_fans_on = fan_entities_on(self.hass, cfg.indoor_fan_entities)
            zone.outdoor_exhaust_on = fan_entities_on(self.hass, cfg.outdoor_exhaust_fan_entities)
            zone.occupied = presence_state(self.hass, cfg.presence_sensor)

            internals: list[float] = []
            setpoints: list[float] = []
            for head in cfg.heads:
                state = self.hass.states.get(head)
                if state is None:
                    continue
                internal = state.attributes.get("current_temperature")
                if isinstance(internal, (int, float)):
                    internals.append(float(internal))
                target = state.attributes.get("temperature")
                if isinstance(target, (int, float)):
                    setpoints.append(float(target))
            zone.head_internal_temp = sum(internals) / len(internals) if internals else None
            zone.device_setpoint = sum(setpoints) / len(setpoints) if setpoints else None

            # Drift learning keys on each head's own state (not aggregate).
            external = _float_state(self.hass, cfg.temp_sensor)
            if external is not None and now_ts - zone.head_state_since >= STABLE_STATE_S:
                for head in cfg.heads:
                    state = self.hass.states.get(head)
                    if state is None:
                        continue
                    internal = state.attributes.get("current_temperature")
                    if isinstance(internal, (int, float)):
                        zone.drift[head].update(
                            _head_state(self.hass, head), float(internal), external
                        )

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
            eid = event.get("event_id")
            # Reject events with a neighbour within EVENT_SETTLE_S (by id).
            isolated = all(
                other_id == eid or abs(other_ts - event["ts"]) > EVENT_SETTLE_S
                for other_ts, _zid, other_id in self._all_events
            )
            if isolated:
                pre = self.p_load_series.window(event["ts"] - 240.0, event["ts"] - 15.0)
                post = self.p_load_series.window(event["ts"] + 60.0, event["ts"] + 240.0)
                delta = power.measure_step(pre, post)
                if delta is not None:
                    self.draws.add_event(event["zone"], event["mode"], delta)
        self._pending_events = remaining
        self._all_events = [
            (ts, zid, eid) for ts, zid, eid in self._all_events if now_ts - ts < 2 * EVENT_SETTLE_S
        ]

    def _compression_floor_w(self, active: list, parked_ids: set[str]) -> float:
        """Same electrical floor as park gating, scaled by open head count."""
        n_open = sum(
            z.config.n_rooms
            for z in self.zones.values()
            if z.is_on or z.config.zone_id in parked_ids
        )
        return park.fan_floor_w(
            max(1, n_open),
            per_head_w=self.settings.fan_floor_per_head_w,
        )

    def _update_estimators(self, now_ts: float, local_hour: float) -> None:
        # p_ac / starts already refreshed by _sample_power on this tick.
        active = [z for z in self.zones.values() if z.is_on]
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

        self._conditioning_fit_refreshed = False
        total_sensible = 0.0
        total_latent = 0.0
        for zone in self.zones.values():
            self._update_zone_estimators(zone, now_ts, local_hour)
            if zone.is_on:
                total_sensible += abs(zone.sensible_w) * zone.config.n_rooms
                total_latent += zone.latent_w * zone.config.n_rooms

        # Park duty/extraction on the 60 s tick (debounced electrically), not
        # gated behind the 5 min thermal fit.
        self._update_park_learners(now_ts, local_hour)

        # House COP only when at least one *conditioning* zone refreshed
        # sensible this tick (free-float zeros must not reopen 60 s sampling).
        if (
            self._conditioning_fit_refreshed
            and self.p_ac
            and self.p_ac > HOUSE_COP_MIN_POWER_W
            and active
        ):
            n_heads = sum(z.config.n_rooms for z in active)
            house_cop = (total_sensible + total_latent) / self.p_ac
            if 0.3 <= house_cop <= 8.0:
                # Publish an EWMA: instantaneous samples at cycle edges divide
                # small heat flows by small powers and swing wildly.
                if self.house_cop is None:
                    self.house_cop = house_cop
                else:
                    self.house_cop += HOUSE_COP_EMA_ALPHA * (house_cop - self.house_cop)
                prev, count = self.cop_table.get(n_heads, (house_cop, 0))
                self.cop_table[n_heads] = (prev + 0.05 * (house_cop - prev), count + 1)
                band = power.outdoor_band(self.t_out)
                if band is not None:
                    key = f"{n_heads}|{band}"
                    prev_b, count_b = self.cop_table_banded.get(key, (house_cop, 0))
                    self.cop_table_banded[key] = (
                        prev_b + 0.05 * (house_cop - prev_b),
                        count_b + 1,
                    )
                # Per-control-state COP: are park-holds the cheapest or the
                # most wasteful kWh in the system? Sampled house-wide.
                skey = self._last_state_key
                if skey is not None:
                    prev_s, count_s = self.cop_table_state.get(skey, (house_cop, 0))
                    self.cop_table_state[skey] = (
                        prev_s + 0.05 * (house_cop - prev_s),
                        count_s + 1,
                    )

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
            zone.sensible_ts = now_ts
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
            latent_w = max(0.0, self._latent_power(zone, dt_h))
            if abs(dtdt) < QUASI_STEADY_K_H:
                zone.model.update_cop(
                    per_head_w,
                    smoothed,
                    self.t_out,
                    local_hour,
                    zone.door_open,
                    t_house,
                    latent_w=latent_w,
                )
            elif abs(dtdt) <= TRANSIENT_K_H:
                # Moderate transient: small rooms on short cycles are never
                # quasi-steady while conditioning (the reason multi-room
                # zones reported cop=null forever). sensible_power_w already
                # carries the C*dT/dt storage term, so the sample is valid.
                # Strong transients (below) stay reserved for c_eff learning.
                zone.model.update_cop(
                    per_head_w,
                    smoothed,
                    self.t_out,
                    local_hour,
                    zone.door_open,
                    t_house,
                    dtdt_per_h=dtdt,
                    latent_w=latent_w,
                )
            else:
                zone.model.update_c_eff(
                    per_head_w,
                    heating,
                    smoothed,
                    self.t_out,
                    dtdt,
                    local_hour,
                    zone.door_open,
                    t_house,
                    indoor_fans_on=zone.indoor_fans_on,
                    outdoor_exhaust_on=zone.outdoor_exhaust_on,
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
        zone.sensible_ts = now_ts
        zone.latent_w = self._latent_power(zone, dt_h)
        self._conditioning_fit_refreshed = True

    def _update_park_learners(self, now_ts: float, local_hour: float) -> None:
        """Feed parked-zone learners once per control tick.

        Activity (compression vs coast) is judged electrically with a
        PARK_DUTY_DEBOUNCE_S settle so brief meter blips do not flip duty.
        Extraction magnitude reuses the latest thermal sensible_w (updated on
        the 5 min fit); stale values from before park entry are skipped so
        honest p_ac cannot promote an idler via frozen full-conditioning W.
        """
        if self.manual_control:
            # Not commanding: never attribute user/remote setpoints to park learning.
            return
        parked_ids = set(self.controller_state.zone_parked_since)
        park_pac = self._park_p_ac(local_hour)
        for zone in self.zones.values():
            zid = zone.config.zone_id
            parked_since = self.controller_state.zone_parked_since.get(zid)
            if parked_since is None:
                zone.park_power.reset()
                continue
            if now_ts - parked_since < PARK_OBS_DELAY_S:
                continue
            stale_sensible = (
                zone.sensible_ts is None
                or zone.sensible_ts < parked_since
                or now_ts - zone.sensible_ts > FIT_STEP_S
            )
            if zone.head_state == STATE_HEATING or self.controller_state.mode == MODE_HEAT:
                raw_extraction = max(0.0, zone.sensible_w)
            else:
                raw_extraction = max(0.0, -zone.sensible_w)
            n_heads = zone.config.n_rooms
            solo = not any(
                other.is_on
                and other.config.zone_id != zid
                and other.config.zone_id not in parked_ids
                for other in self.zones.values()
            )
            if solo:
                if park_pac is None:
                    # Unknown baseline slot: do not fall through to thermal-only
                    # activity (would invent compression from stale sensible).
                    continue
                settled = zone.park_power.settle(
                    now_ts,
                    park_pac,
                    floor_w=park.fan_floor_w(
                        n_heads, per_head_w=self.settings.fan_floor_per_head_w
                    ),
                )
                if settled is None:
                    continue
                # Solo: electrical gate is authoritative. Stale thermal magnitude
                # must not drop the observation (idler classification needs it)
                # and must not feed frozen full-conditioning watts.
                if not settled:
                    extraction, active = 0.0, False
                elif stale_sensible:
                    extraction, active = 0.0, True
                else:
                    extraction, active = max(0.0, raw_extraction), True
            else:
                # Non-solo activity is thermal-judged — need a fresh sensible.
                if stale_sensible:
                    continue
                zone.park_power.reset()
                extraction, active = park.gate_observation(
                    park_pac,
                    solo,
                    raw_extraction,
                    n_heads=n_heads,
                    per_head_w=self.settings.fan_floor_per_head_w,
                )
            margin = self.controller_state.zone_park_margin.get(zid)
            zone.park.update(extraction, active, margin_k=margin)

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
        outdoor_flow = zone.model.outdoor_airflow_m3h(
            zone.door_open, outdoor_exhaust_on=zone.outdoor_exhaust_on
        )
        infiltration = 0.0
        if w_out is not None:
            infiltration = psychro.AIR_DENSITY_KG_M3 * outdoor_flow * (w_out - latest[1])
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
        outdoor_flow = zone.model.outdoor_airflow_m3h(
            zone.door_open, outdoor_exhaust_on=zone.outdoor_exhaust_on
        )
        removed = psychro.moisture_removal_kg_h(
            zone.config.sensed_room.volume_m3,
            outdoor_flow,
            latest[1],
            w_out if w_out is not None else latest[1],
            dw_dt,
            zone.moisture_sources_kg_h,
        )
        return psychro.latent_power_w(removed)

    # -- control -----------------------------------------------------------------

    def _build_snapshot(self, now_ts: float, local_hour: float) -> HouseSnapshot:
        zone_snaps: list[ZoneSnapshot] = []
        forecast = self._effective_forecast(now_ts)
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
            zone.standing_load_w = self._standing_load_w(zone)
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
                    park_trickles=zone.park.trickles,
                    park_extraction_w=zone.park.extraction_w,
                    park_preferred_margin_k=zone.park.preferred_margin_k,
                    park_coast_margin_k=zone.park.coast_margin_k(),
                    park_samples=zone.park.samples,
                    head_mode=(
                        MODE_HEAT
                        if zone.head_state == STATE_HEATING
                        else MODE_COOL
                        if zone.head_state == STATE_COOLING
                        else None
                    ),
                    standing_load_w=zone.standing_load_w,
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
            t_out_synthetic=self._t_out_synthetic,
            t_rm=self.t_rm,
            house_occupied=self.house_occupied,
            p_grid=self.p_grid,
            p_demand=self.p_demand,
            p_grid_over_since=(
                now_ts - self.grid_over_since if self.grid_over_since is not None else None
            ),
            shed_urgent=self.shed_urgent,
            forecast_hours=tuple(forecast),
            cop_by_head_count=cop_hints,
            aux_indoor=tuple(aux_indoor),
        )

    def reset_track_deltas(self) -> None:
        """Clear sticky tracking depth (Manual enter / house vacant→occupied)."""
        self.controller_state.zone_track_delta.clear()

    def _clear_park_sessions(self) -> None:
        """Drop in-flight parks (Manual enter — not commanding)."""
        from .core.controller import _clear_park_session

        now = time.time()
        for zid in list(self.controller_state.zone_parked_since):
            # zone=None: drop session without charging probe spacing.
            _clear_park_session(self.controller_state, zid, None, now)

    def set_preset(self, preset: str) -> None:
        """Apply a climate preset; Manual clears track/park and remembers prior preset."""
        prev = self.settings.preset
        if preset == PRESET_MANUAL and prev != PRESET_MANUAL:
            self._preset_before_manual = prev if prev != PRESET_MANUAL else PRESET_NONE
            self.settings.preset = PRESET_MANUAL
            self.reset_track_deltas()
            self._clear_park_sessions()
            return
        if prev == PRESET_MANUAL and preset != PRESET_MANUAL:
            # Leaving Manual via climate preset picker — honour the requested preset.
            self.settings.preset = preset
            return
        self.settings.preset = preset

    def clear_manual(self) -> None:
        """Exit Manual via the switch, restoring the pre-Manual preset."""
        if not self.manual_control:
            return
        restored = self._preset_before_manual or PRESET_NONE
        if restored == PRESET_MANUAL:
            restored = PRESET_NONE
        self.settings.preset = restored

    @property
    def manual_control(self) -> bool:
        return self.settings.preset == PRESET_MANUAL

    @property
    def effective_preset(self) -> str:
        return comfort.effective_preset(self.settings, self.house_occupied)

    def zone_want(self, zone: ZoneRuntime) -> str:
        """Controller desire for this zone (demand/helper/off) from last tick."""
        return (self.last_diag.get("want") or {}).get(zone.config.zone_id, "off")

    async def _async_apply_zone_disable_edges(self) -> None:
        """On zone_enabled false edge, emit Off once (do not spam while disabled)."""
        for zid, zone in self.zones.items():
            enabled = self.settings.zone_enabled.get(zid, True)
            prev = self._prev_zone_enabled.get(zid, True)
            if prev and not enabled:
                from .core.controller import _clear_park_session

                _clear_park_session(self.controller_state, zid, None, time.time())
                await self._async_execute(
                    Command(zone_id=zid, hvac_mode=MODE_OFF, reason="disabled")
                )
                zone.last_control_reason = "disabled"
            self._prev_zone_enabled[zid] = enabled

    async def _async_control(
        self, now_ts: float, local_hour: float, *, skip_if_busy: bool = False
    ) -> None:
        # React: skip if tick holds the lock (stale queued shed is worse).
        # Tick: await the lock normally.
        if skip_if_busy and self._control_lock.locked():
            return
        async with self._control_lock:
            await self._async_control_locked(now_ts, local_hour)

    async def _async_control_locked(self, now_ts: float, local_hour: float) -> None:
        snapshot = self._build_snapshot(now_ts, local_hour)
        decision = controller.tick(snapshot, self.controller_state)
        self.last_diag = decision.diag
        self.mode_source = decision.diag.get("mode_source", "off")
        self.warm_excess_kh = decision.diag.get("warm_excess_kh")
        self.cold_deficit_kh = decision.diag.get("cold_deficit_kh")
        self.window_suggestions = decision.window_suggestions
        # Persist learned park depth onto each zone's ParkEstimator (survives restart).
        for zid, preferred in decision.state.zone_park_preferred.items():
            zone = self.zones.get(zid)
            if zone is not None:
                zone.park.preferred_margin_k = preferred
        for command in decision.commands:
            zone = self.zones.get(command.zone_id)
            if zone is not None:
                zone.last_control_reason = command.reason
        self._update_free_float_bias(snapshot)
        if self.manual_control:
            # Still force children off when hub HVAC is Off; otherwise hands-off.
            if self.settings.hvac_mode == MODE_OFF:
                for command in decision.commands:
                    if command.hvac_mode == MODE_OFF:
                        await self._async_execute(command)
            return
        for command in decision.commands:
            await self._async_execute(command)

    def zone_control_state(self, zone: ZoneRuntime) -> str:
        """Coarse per-zone action for history charts (what we commanded/held)."""
        if self.manual_control and self.settings.hvac_mode != MODE_OFF:
            return "manual"
        zid = zone.config.zone_id
        st = self.controller_state
        if zid in st.shed:
            return "shed"
        if zid in st.zone_parked_since:
            return "park"
        if st.zone_fan.get(zid):
            return "fan_assist"
        if zone.last_control_reason == "runout":
            return "runout"
        demand = set(self.last_diag.get("demand") or [])
        helpers = set(self.last_diag.get("helpers") or [])
        if zid in demand:
            return "demand"
        if zid in helpers:
            return "helper"
        if st.zone_on.get(zid) or zone.is_on:
            return "conditioning"
        return "off"

    def zone_control_attrs(self, zone: ZoneRuntime) -> dict:
        """Detail for the control_state sensor — replaces separate reason/parked entities."""
        zid = zone.config.zone_id
        st = self.controller_state
        parked = zid in st.zone_parked_since
        attrs: dict = {
            "last_reason": zone.last_control_reason or "none",
            "parked": parked,
            "track_delta_k": st.zone_track_delta.get(zid),
        }
        if parked:
            attrs["park_margin_k"] = st.zone_park_margin.get(zid)
            attrs["park_preferred_margin_k"] = st.zone_park_preferred.get(
                zid, zone.park.preferred_margin_k
            )
            attrs["park_classification"] = zone.park.classification
            attrs["park_extraction_w"] = (
                None if zone.park.extraction_w is None else round(zone.park.extraction_w, 0)
            )
            attrs["park_active_ratio"] = (
                None if zone.park.active_ratio is None else round(zone.park.active_ratio, 3)
            )
            attrs["park_coast_margin_k"] = zone.park.coast_margin_k()
        return attrs

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
        park_expressed = False
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
                minimum = state.attributes.get("min_temp", 16.0)
                maximum = state.attributes.get("max_temp", 30.0)
                internal = state.attributes.get("current_temperature")
                if command.park and isinstance(internal, (int, float)):
                    park_expressed = True
                    margin = command.park_margin or controller.PARK_MARGIN_K
                    # Park: ride just above the internal reading, keeping the
                    # compressor mode, so the device's own above-setpoint
                    # policy (idle vs keep-temperature trickle) expresses
                    # itself and can be measured. Never used for shed zones.
                    raw = (
                        float(internal) + margin
                        if command.hvac_mode == MODE_COOL
                        else float(internal) - margin
                    )
                elif command.park:
                    # This head has no internal reading; try sibling heads.
                    continue
                elif command.track_delta is not None and isinstance(internal, (int, float)):
                    # Tracking control: anchor to the live internal reading so
                    # the head sees a small constant error and stays at low
                    # modulation instead of ramping hard and self-terminating
                    # when the coil chills its own sensor (drift is dynamic
                    # within a run; the static offset over- then under-shoots).
                    delta = command.track_delta
                    raw = (
                        float(internal) - delta
                        if command.hvac_mode == MODE_COOL
                        else float(internal) + delta
                    )
                else:
                    # Fallback: static drift translation (cold start, stale or
                    # missing internal reading, or tracking disabled).
                    offset = zone.drift[head].offset(active)
                    raw = (command.setpoint or self.settings.target) + offset
                step = state.attributes.get("target_temp_step", 0.5)
                try:
                    step_f = float(step) if step is not None else 0.5
                except (TypeError, ValueError):
                    step_f = 0.5
                setpoint = controller.quantize_setpoint(
                    raw, float(minimum), float(maximum), step=step_f
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
        if command.park and not park_expressed:
            # No head could express the park — drop session so the learner does
            # not attribute observations to a park that never executed.
            from .core.controller import _clear_park_session

            _clear_park_session(self.controller_state, command.zone_id, None, time.time())

    # -- entity-facing helpers ------------------------------------------------------

    @property
    def power_headroom_w(self) -> float | None:
        demand = self.p_demand if self.p_demand is not None else self.p_grid
        if demand is None:
            return None
        return self.settings.limit_w - demand

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
            "p_demand": self.p_demand,
            "known_load_total": self.known_load_total,
            "shed_urgent": self.shed_urgent,
            "p_load": self.p_load,
            "p_ac": self.p_ac,
            "forecast": self.forecast,
            "cop_table": {str(k): v for k, v in self.cop_table.items()},
            "cop_table_banded": dict(self.cop_table_banded),
            "cop_table_state": dict(self.cop_table_state),
            "compressor_starts_per_hour_24h": round(self.starts.per_hour(time.time()), 2),
            "compressor_starts_by_state_24h": self.starts.by_state(time.time()),
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
                    "park": z.park.to_dict() | {"classification": z.park.classification},
                    "confidence": z.model.confidence(z.door_open),
                    "drift": {h: d.to_dict() for h, d in z.drift.items()},
                    "sensible_w": z.sensible_w,
                    "latent_w": z.latent_w,
                    "allocated_w": z.allocated_w,
                    "standing_load_w": (
                        None if z.standing_load_w is None else round(z.standing_load_w, 1)
                    ),
                    "disturbance": z.model.disturbance_diag(
                        self._local_hour(), z.door_open
                    ),
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
                "fallback": model.fit_is_fallback(regime),
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
