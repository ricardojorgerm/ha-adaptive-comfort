"""Runtime for Adaptive Comfort: sampling, estimation, control loop.

All math lives in core/; this module adapts Home Assistant state and
services to the pure snapshot/command interface.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from datetime import timedelta
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    ATTR_TEMPERATURE,
    ATTR_UNIT_OF_MEASUREMENT,
    STATE_ON,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
    UnitOfEnergy,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.event import async_track_state_change_event, async_track_time_interval
from homeassistant.util import dt as dt_util

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
from .core.power import BaselineModel, DrawEstimator, EnergyMerge
from .core.predictor import HORIZONS_MIN, PredictorScorer
from .core.series import TimeSeries
from .core.thermal import (
    DiurnalModel,
    ThermalModel,
    apply_rain_outdoor,
    blend_outdoor_trend,
    clearness_index,
    house_other_temperature,
    rain_sink_k_per_h,
)
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
from .presence import OccupancyDebounce, house_presence, presence_state
from .storage import AdaptiveComfortStore

_LOGGER = logging.getLogger(__name__)

SAMPLE_INTERVAL = timedelta(seconds=60)
POWER_REACT_DEBOUNCE_S = 2.0
SAVE_INTERVAL = timedelta(minutes=15)
FORECAST_INTERVAL = timedelta(minutes=30)
FIT_STEP_S = 300.0  # 5-minute smoothed steps for the RC fit
STABLE_STATE_S = 600.0  # drift updates need >=10 min in a stable head state
ALL_OFF_SETTLE_S = 900.0  # free-response fit waits 15 min after heads stop
TRANSIENT_SETTLE_S = 300.0  # q_transient may update after 5 min off
OUTDOOR_TREND_S = 30.0 * 60.0  # live dT_out/dt window
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
# House/zone COP: schema 2 wiped inflated-latent history; schema 3 splits
# heat/cool ledgers (keys '{mode}|…') so seasons no longer mix.
COP_SCHEMA = 4  # depth-binned COP ledger (chase / zero / hysteresis)


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


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
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
        self.last_on_sensible_w: float | None = None  # last non-park conditioning (signed)
        self.latent_w = 0.0  # sensed room
        # Zone-total free-float heat that must be rejected/supplied to hold T (W).
        self.standing_load_w: float | None = None
        self.park = park.ParkEstimator()
        self.park_power = park.PowerDebounce()
        self.moisture_sources_kg_h = 0.0
        self.temp: float | None = None
        self.rh: float | None = None
        self.door_open = True  # no door sensor → assume open (inter-room mixing)
        self.indoor_fans_on = False
        self.outdoor_exhaust_on = False
        self.occupied: bool | None = None
        self.occupancy = OccupancyDebounce()
        self.free_float: tuple[float, ...] = ()
        self.pred_60m: float | None = None
        self.allocated_w = 0.0
        # Live head-frame readings (for tracking/park diagnostics).
        self.head_internal_temp: float | None = None
        self.device_setpoint: float | None = None
        # Frozen hold: last executed positive-depth device SP (not live internal).
        self.last_hold_depth_k: float | None = None
        self.last_hold_device_sp: float | None = None
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
            "occupancy": self.occupancy.to_dict(),
            "moisture_sources_kg_h": self.moisture_sources_kg_h,
            "moisture_schema": MOISTURE_SCHEMA,
        }

    def restore(self, data: dict) -> None:
        if "park" in data:
            self.park = park.ParkEstimator.from_dict(data["park"])
        if "thermal" in data:
            self.model = ThermalModel.from_dict(data["thermal"], self.config.sensed_room.volume_m3)
            # No door sensor: history was learned under default-closed; move it
            # to open so mixing-aware reads do not start from a blank regime.
            if not self.config.door_sensor:
                self.model.migrate_default_closed_to_open()
        for head, drift_data in data.get("drift", {}).items():
            if head in self.drift:
                self.drift[head] = DriftEstimator.from_dict(drift_data)
        # Inflated (k_out+k_mix) moisture EWMAs are not reusable.
        if int(data.get("moisture_schema", 0)) >= MOISTURE_SCHEMA:
            self.moisture_sources_kg_h = float(data.get("moisture_sources_kg_h", 0.0))
        else:
            self.moisture_sources_kg_h = 0.0
        if "occupancy" in data:
            self.occupancy = OccupancyDebounce.from_dict(data["occupancy"])


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
        self.energy = EnergyMerge(
            mode=str(entry.data.get(CONF_ENERGY_MERGE, power.ENERGY_MERGE_MAX))
        )
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
        self.p_discharge: float = 0.0
        self.known_load_total: float = 0.0
        self.shed_urgent: bool = False
        self._prev_known_total: float | None = None
        self._demand_spike_until: float = 0.0
        self._last_power_react: float = 0.0
        self.p_grid_series = TimeSeries(horizon_s=2 * 3600.0)
        self.p_load_series = TimeSeries(horizon_s=2 * 3600.0)
        self.grid_over_since: float | None = None
        self.forecast: list[float] = []
        self.forecast_cloud: list[float | None] = []
        self.forecast_condition: list[str | None] = []
        self.forecast_precip: list[float | None] = []
        self.forecast_precip_prob: list[float | None] = []
        self._t_out_hist: list[tuple[float, float]] = []
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
        self.cop_table: dict[str, tuple[float, int]] = {}  # "{mode}|{heads}" -> (ewma, n)
        # "{mode}|{heads}|{mild|warm|hot}" -> (ewma, n) — weather vs head-count.
        self.cop_table_banded: dict[str, tuple[float, int]] = {}
        # "{mode}|{conditioning|park|mixed}" -> (ewma, n) — park-hold economics.
        self.cop_table_state: dict[str, tuple[float, int]] = {}
        # "{mode}|{±depth}" -> (ewma, n) — continuum depth economics (incl. ≤0).
        self.cop_table_depth: dict[str, tuple[float, int]] = {}
        self.starts = power.StartCounter()
        self._last_state_key: str | None = None
        self.house_cop: float | None = None
        # Mean predicted-vs-band deviation across zones (K); NOT a prediction
        # error metric -- see `_update_free_float_deviation`. Real
        # predicted-vs-actual error lives in `predictor_scorer` below.
        self.free_float_deviation: float | None = None
        self.predictor_scorer = PredictorScorer()
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
        if batt := data.get(CONF_BATTERY_POWER):
            entities.append(batt)
        entities.extend(data.get(CONF_KNOWN_LOADS, []))
        return entities

    def _known_load_readings(self) -> list[float]:
        batt = self.entry.data.get(CONF_BATTERY_POWER)
        return [
            v
            for eid in self.entry.data.get(CONF_KNOWN_LOADS, [])
            if eid != batt and (v := _float_state(self.hass, eid)) is not None
        ]

    def _finalize_demand(self, now_ts: float) -> None:
        """Contracted demand: max(p_grid, 120s peak, known + p_ac - discharge)."""
        known = self._known_load_readings()
        self.known_load_total = sum(known)
        # Residual p_ac only — not max(learned, p_ac). Heads off → no AC term.
        active = any(z.is_on for z in self.zones.values())
        ac_w = self.p_ac if active and self.p_ac is not None else 0.0
        peak = self.p_grid_series.max_window(now_ts - power.SHED_PEAK_WINDOW_S, now_ts)
        self.p_demand = power.contracted_demand_w(self.p_grid, known, ac_w, peak, self.p_discharge)

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
            peak_demand = power.contracted_demand_w(None, known, ac_w, peak, self.p_discharge)
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
            "zone_presence_adaptation",
            "shedding_enabled",
            "fan_assist",
            "tracking",
            "park_learning",
            "auto_regime",
            "night_ventilate",
            "prefer_continuous",
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
        self.settings.zone_quiet_night = {
            str(k): bool(v) for k, v in dict(settings.get("zone_quiet_night") or {}).items()
        }
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
        if "energy" in data:
            self.energy = EnergyMerge.from_dict(
                data["energy"],
                mode=str(self.entry.data.get(CONF_ENERGY_MERGE, power.ENERGY_MERGE_MAX)),
            )
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
        schema = int(data.get("cop_schema", 0))
        if schema >= 3:
            for key, value in data.get("cop_table", {}).items():
                try:
                    self.cop_table[str(key)] = (float(value[0]), int(value[1]))
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
            if schema >= 4:
                for key, value in data.get("cop_table_depth", {}).items():
                    try:
                        self.cop_table_depth[str(key)] = (float(value[0]), int(value[1]))
                    except (TypeError, ValueError, IndexError):
                        continue
        elif schema == 2:
            # Schema 2 mixed heat/cool under bare keys; attribute to cool
            # (field history is cooling-dominated) and start heat fresh.
            self.cop_table = power.migrate_cop_table_keys(data.get("cop_table", {}), kind="heads")
            self.cop_table_banded = power.migrate_cop_table_keys(
                data.get("cop_table_banded", {}), kind="banded"
            )
            self.cop_table_state = power.migrate_cop_table_keys(
                data.get("cop_table_state", {}), kind="state"
            )
        else:
            # Inflated-latent COP history is not reusable.
            self.cop_table.clear()
            self.cop_table_banded.clear()
            self.cop_table_state.clear()
            self.cop_table_depth.clear()
            self.house_cop = None
            for zone in self.zones.values():
                zone.model.cop = None
                zone.model.cop_samples = 0
        if "starts" in data:
            self.starts = power.StartCounter.from_dict(data["starts"])
        if "predictor_scorer" in data:
            self.predictor_scorer = PredictorScorer.from_dict(data["predictor_scorer"])

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
                "zone_presence_adaptation": s.zone_presence_adaptation,
                "shedding_enabled": s.shedding_enabled,
                "fan_assist": s.fan_assist,
                "tracking": s.tracking,
                "park_learning": s.park_learning,
                "auto_regime": s.auto_regime,
                "night_ventilate": s.night_ventilate,
                "prefer_continuous": s.prefer_continuous,
                "fan_floor_per_head_w": s.fan_floor_per_head_w,
                "window_suggest": s.window_suggest,
                "hvac_mode": s.hvac_mode,
                "preset": s.preset,
                "zone_offsets": s.zone_offsets,
                "zone_enabled": s.zone_enabled,
                "zone_quiet_night": s.zone_quiet_night,
            },
            "preset_before_manual": self._preset_before_manual,
            "prev_house_occupied": self._prev_house_occupied,
            "t_rm": self.t_rm,
            "baseline": self.baseline.to_dict(),
            "energy": self.energy.to_dict(),
            "draws": self.draws.to_dict(),
            "outdoor_diurnal": self.outdoor_diurnal.to_dict(),
            "controller": self.controller_state.to_dict(),
            "zones": {zid: zone.to_dict() for zid, zone in self.zones.items()},
            "cop_schema": COP_SCHEMA,
            "cop_table": {k: [v[0], v[1]] for k, v in self.cop_table.items()},
            "cop_table_banded": {k: [v[0], v[1]] for k, v in self.cop_table_banded.items()},
            "cop_table_state": {k: [v[0], v[1]] for k, v in self.cop_table_state.items()},
            "cop_table_depth": {k: [v[0], v[1]] for k, v in self.cop_table_depth.items()},
            "starts": self.starts.to_dict(),
            "predictor_scorer": self.predictor_scorer.to_dict(),
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

    def _standing_load_w(self, zone, solar_scale: float = 1.0) -> float | None:
        """Estimated heat inflow the zone must reject to hold temperature (W).

        The free-float side of sensible_power_w (dT/dt = 0, no AC term) at
        current conditions: what a parked/off zone gains per second. Used to
        judge whether learned residual output can carry a satisfied zone.
        """
        if zone.temp is None or self.t_out is None:
            return None
        local_hour = self._local_hour()
        t_house = self._house_other_temp(zone.config.zone_id)
        # sensible_power_w returns heat *added by the AC*; with dtdt=0 its
        # negation is the standing inflow the AC must remove to hold temp.
        inflow = -zone.model.sensible_power_w(
            zone.temp,
            self.t_out,
            0.0,
            local_hour,
            zone.door_open,
            t_house,
            indoor_fans_on=zone.indoor_fans_on,
            outdoor_exhaust_on=zone.outdoor_exhaust_on,
            solar_scale=solar_scale,
        )
        # Cooling must reject heat coming in; heating must replace heat
        # going out. Same free-float number, opposite sign of interest.
        if self.controller_state.mode == MODE_HEAT:
            return max(0.0, -inflow) * zone.config.n_rooms
        return max(0.0, inflow) * zone.config.n_rooms

    def _q_hvac_k_per_h(self, zone: ZoneRuntime) -> float | None:
        """AC overlay (K/h) for dying-nick cost: last on-period excess, else hold.

        Sign must match the live plant mode. Park-idle ~0 W and a leftover
        cooling overlay while heating are not a pull-down rate.
        """
        c = zone.model.c_eff_wh_per_k
        if c <= 0.0:
            return None
        last = zone.last_on_sensible_w
        mode = self.controller_state.mode
        q = None
        if last is not None and abs(last) > park.IDLE_MAX_W:
            q = last / c
            if (mode == MODE_HEAT and q <= 0.0) or (mode == MODE_COOL and q >= 0.0):
                q = None
        if q is not None:
            return q
        if zone.standing_load_w is None:
            return None
        per_head = zone.standing_load_w / max(1, zone.config.n_rooms)
        if mode == MODE_HEAT:
            return per_head / c
        if mode == MODE_COOL:
            return -per_head / c
        return None

    def _house_other_temp(self, zone_id: str) -> float | None:
        return house_other_temperature(
            zone_id, self._volume_readings(), self._aux_volume_readings()
        )

    def _house_other_hourly(self, zone_id: str, hours: int) -> list[float] | None:
        """Hourly house-other trajectory for predict_free's mixing term.

        Counterfactual: *this* zone's heads are off (what predict_free
        models for the zone being predicted) while siblings keep being
        controlled as usual -- "my heads off, siblings as usual", not the
        house-wide no-AC counterfactual `comfort.demand_integrals` builds
        zone-by-zone from each zone's own predict_free output. Using a
        constant "frozen at right now" house-other reading for a 24 h
        integration silently assumes a sibling stays wherever it happens to
        be this instant even if it is mid-cycle; since siblings under active
        control are pulled toward their own comfort band, a sibling that is
        currently on is modelled converging toward its band centre (what
        control keeps it near) while a sibling that is off is held at its
        last reading (no better forward model without recursing into a full
        multi-zone simulation, which is out of scope here). Volume-weighted
        like the instantaneous `house_other_temperature()`.
        """
        others: list[tuple[float, float, float]] = []  # (now_t, target_t, volume)
        for other_zid, other in self.zones.items():
            if other_zid == zone_id or other.temp is None:
                continue
            vol = other.config.total_volume_m3
            if vol <= 0:
                continue
            if other.is_on:
                target = comfort.band_center(
                    self.settings, self.t_rm, self.settings.zone_offsets.get(other_zid, 0.0)
                )
            else:
                target = other.temp
            others.append((other.temp, target, vol))
        for temp, vol in self._aux_volume_readings():
            others.append((temp, temp, vol))
        if not others:
            return None
        # ~90 min time constant to approach the controlled target -- fast
        # enough to matter within the scoring horizons, slow enough not to
        # pretend a sibling snaps to setpoint instantly.
        tau_h = 1.5
        trajectory: list[float] = []
        for h in range(hours):
            frac = 1.0 - math.exp(-(h + 1) / tau_h)
            total = 0.0
            weight = 0.0
            for now_t, target_t, vol in others:
                total += vol * (now_t + frac * (target_t - now_t))
                weight += vol
            trajectory.append(total / weight if weight > 0 else 0.0)
        return trajectory

    @staticmethod
    def _coeffs_snapshot(zone: ZoneRuntime) -> dict:
        """Model coefficients in effect for one zone right now (predictor audit)."""
        model = zone.model
        door = zone.door_open
        return {
            "k_out_h": round(model.k(door), 5),
            "k_mix_h": round(model.k_mix(door), 5),
            "door_open": door,
            "fit_samples": model.fit_samples(door),
            "fit_stage": model.fit_stage(door),
            "q": model.q_coeffs(door),
        }

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
        clouds: list[float | None] = []
        conditions: list[str | None] = []
        precip: list[float | None] = []
        precip_prob: list[float | None] = []
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
                for item in forecast[:FORECAST_HOURS]:
                    if not isinstance(item.get("temperature"), (int, float)):
                        continue
                    temps.append(float(item["temperature"]))
                    clouds.append(_optional_float(item.get("cloud_coverage")))
                    cond = item.get("condition")
                    conditions.append(str(cond) if cond else None)
                    precip.append(_optional_float(item.get("precipitation")))
                    precip_prob.append(
                        _optional_float(
                            item.get("precipitation_probability", item.get("precip_probability"))
                        )
                    )
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
            clouds = [None] * len(temps)
            conditions = [None] * len(temps)
            precip = [None] * len(temps)
            precip_prob = [None] * len(temps)
        if temps and self.t_out is not None:
            temps[0] = self.t_out  # anchor the horizon on the live reading
        self.forecast = temps
        self.forecast_cloud = clouds
        self.forecast_condition = conditions
        self.forecast_precip = precip
        self.forecast_precip_prob = precip_prob
        self._forecast_fetched_ts = now_ts

    def _outdoor_dtdt_per_h(self, _now_ts: float) -> float | None:
        hist = getattr(self, "_t_out_hist", None) or []
        if len(hist) < 2:
            return None
        oldest_ts, oldest_t = hist[0]
        newest_ts, newest_t = hist[-1]
        span_h = (newest_ts - oldest_ts) / 3600.0
        if span_h < 10.0 / 60.0:
            return None
        return (newest_t - oldest_t) / span_h

    def _shift_hourly_floats(self, series: list[float], elapsed_h: float) -> list[float]:
        n = len(series)
        if n == 0:
            return []
        shifted: list[float] = []
        for h in range(FORECAST_HOURS):
            pos = elapsed_h + h
            idx = min(int(pos), n - 1)
            frac = min(pos - idx, 1.0)
            nxt = min(idx + 1, n - 1)
            shifted.append(series[idx] * (1.0 - frac) + series[nxt] * frac)
        return shifted

    def _shift_hourly_optional(
        self, series: list[float | None], elapsed_h: float
    ) -> list[float | None]:
        n = len(series)
        if n == 0:
            return [None] * FORECAST_HOURS
        out: list[float | None] = []
        for h in range(FORECAST_HOURS):
            pos = elapsed_h + h
            idx = min(int(pos), n - 1)
            frac = min(pos - idx, 1.0)
            nxt = min(idx + 1, n - 1)
            a, b = series[idx], series[nxt]
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                out.append(float(a) * (1.0 - frac) + float(b) * frac)
            else:
                out.append(a if frac < 0.5 else b)
        return out

    def _shift_hourly_text(self, series: list[str | None], elapsed_h: float) -> list[str | None]:
        n = len(series)
        if n == 0:
            return [None] * FORECAST_HOURS
        out: list[str | None] = []
        for h in range(FORECAST_HOURS):
            pos = elapsed_h + h
            idx = min(int(pos + 0.5), n - 1)
            out.append(series[idx])
        return out

    def _effective_forecast(self, now_ts: float) -> list[float]:
        """Re-index stored forecast so index 0 is 'now', with live trend + rain."""
        if not self.forecast:
            return [self.t_out] * FORECAST_HOURS if self.t_out is not None else []
        elapsed_h = 0.0
        if self._forecast_fetched_ts > 0.0:
            elapsed_h = max(0.0, (now_ts - self._forecast_fetched_ts) / 3600.0)
        shifted = self._shift_hourly_floats(self.forecast, elapsed_h)
        shifted = blend_outdoor_trend(shifted, self.t_out, self._outdoor_dtdt_per_h(now_ts))
        precip = self._shift_hourly_optional(
            getattr(self, "forecast_precip", None) or [], elapsed_h
        )
        precip_prob = self._shift_hourly_optional(
            getattr(self, "forecast_precip_prob", None) or [], elapsed_h
        )
        return apply_rain_outdoor(shifted, precip, precip_prob, self.t_out)

    def _forecast_q_hours(self, now_ts: float) -> tuple[list[float], list[float]]:
        """Hourly solar scale and rain-sink (K/h) aligned with `_effective_forecast`."""
        elapsed_h = 0.0
        if self._forecast_fetched_ts > 0.0:
            elapsed_h = max(0.0, (now_ts - self._forecast_fetched_ts) / 3600.0)
        clouds = self._shift_hourly_optional(getattr(self, "forecast_cloud", None) or [], elapsed_h)
        conditions = self._shift_hourly_text(
            getattr(self, "forecast_condition", None) or [], elapsed_h
        )
        precip = self._shift_hourly_optional(
            getattr(self, "forecast_precip", None) or [], elapsed_h
        )
        precip_prob = self._shift_hourly_optional(
            getattr(self, "forecast_precip_prob", None) or [], elapsed_h
        )
        scales = [
            clearness_index(
                clouds[i] if i < len(clouds) else None,
                conditions[i] if i < len(conditions) else None,
            )
            for i in range(FORECAST_HOURS)
        ]
        sinks = [
            rain_sink_k_per_h(
                precip[i] if i < len(precip) else None,
                precip_prob[i] if i < len(precip_prob) else None,
            )
            for i in range(FORECAST_HOURS)
        ]
        return scales, sinks

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
        self._sample_energy()
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
        self.p_discharge = power.battery_discharge_w(
            battery, data.get(CONF_BATTERY_POSITIVE_DISCHARGING, True)
        )
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

    def _energy_reading_wh(self, entity_id: str) -> float | None:
        """Convert an energy sensor sample to Wh. Never a 60 s wattage."""
        state = self.hass.states.get(entity_id)
        if state is None or state.state in (STATE_UNKNOWN, STATE_UNAVAILABLE):
            return None
        try:
            value = float(state.state)
        except (TypeError, ValueError):
            return None
        unit = str(state.attributes.get(ATTR_UNIT_OF_MEASUREMENT) or "")
        if unit in (UnitOfEnergy.WATT_HOUR, "Wh"):
            return value
        return value * 1000.0

    def _sample_energy(self) -> None:
        """Tick-only Wh ledger. Does not replace residual p_ac."""
        raw = self.entry.data.get(CONF_CONSUMPTION) or []
        entities = [raw] if isinstance(raw, str) else list(raw)
        if not entities:
            return
        self.energy.update({eid: self._energy_reading_wh(eid) for eid in entities})

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
        # Depth-aware control-state COP: depth_k ≤ 0 → conditioning, > 0 → park.
        parked_ids = {
            z.config.zone_id
            for z in active
            if power.zone_depth_is_park(
                self._zone_head_depth_k(z.config.zone_id),
                parked_fallback=z.config.zone_id in self.controller_state.zone_parked_since,
            )
        }
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
            self._t_out_hist.append((now_ts, self.t_out))
            cutoff = now_ts - OUTDOOR_TREND_S
            self._t_out_hist = [(ts, t) for ts, t in self._t_out_hist if ts >= cutoff]
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
            # No door sensor: assume open so k_mix / house-coupling is the
            # default regime (inter-room mixing is the common case).
            if cfg.door_sensor:
                zone.door_open = door is not None and door.state == STATE_ON
            else:
                zone.door_open = True
            zone.indoor_fans_on = fan_entities_on(self.hass, cfg.indoor_fan_entities)
            zone.outdoor_exhaust_on = fan_entities_on(self.hass, cfg.outdoor_exhaust_fan_entities)
            raw_occ = presence_state(self.hass, cfg.presence_sensor)
            zone.occupied = zone.occupancy.update(raw_occ, now_ts)

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
        scales, _sinks = self._forecast_q_hours(now_ts)
        solar_scale = scales[0] if scales else 1.0
        for zone in self.zones.values():
            self._update_zone_estimators(zone, now_ts, local_hour, solar_scale=solar_scale)
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
                # Ledger key must match snapshot hint filtering (controller
                # mode) and agree with live head_state — else skip the row.
                mode = power.cop_sample_mode(
                    self.controller_state.mode,
                    any_heating=any(z.head_state == STATE_HEATING for z in active),
                    any_cooling=any(z.head_state == STATE_COOLING for z in active),
                )
                if mode is not None:
                    hkey = power.cop_heads_key(mode, n_heads)
                    prev, count = self.cop_table.get(hkey, (house_cop, 0))
                    self.cop_table[hkey] = (prev + 0.05 * (house_cop - prev), count + 1)
                    band = power.outdoor_band(self.t_out)
                    if band is not None:
                        bkey = power.cop_banded_key(mode, n_heads, band)
                        prev_b, count_b = self.cop_table_banded.get(bkey, (house_cop, 0))
                        self.cop_table_banded[bkey] = (
                            prev_b + 0.05 * (house_cop - prev_b),
                            count_b + 1,
                        )
                    # Per-control-state COP (mode-split): are park-holds the
                    # cheapest or the most wasteful kWh in this mode?
                    skey = self._last_state_key
                    if skey is not None:
                        sk = power.cop_state_key(mode, skey)
                        prev_s, count_s = self.cop_table_state.get(sk, (house_cop, 0))
                        self.cop_table_state[sk] = (
                            prev_s + 0.05 * (house_cop - prev_s),
                            count_s + 1,
                        )
                    # Depth-binned COP (0.5 K grid), including chase / zero hold.
                    for zone in active:
                        depth = self._zone_head_depth_k(zone.config.zone_id)
                        if depth is None:
                            continue
                        dkey = power.cop_depth_key(mode, depth)
                        prev_d, count_d = self.cop_table_depth.get(dkey, (house_cop, 0))
                        self.cop_table_depth[dkey] = (
                            prev_d + 0.05 * (house_cop - prev_d),
                            count_d + 1,
                        )

    def _zone_head_depth_k(self, zid: str) -> float | None:
        """Live signed head depth, with park_margin / track_delta mirrors as fallback."""
        st = self.controller_state
        if zid in st.zone_head_depth_k:
            return float(st.zone_head_depth_k[zid])
        if zid in st.zone_park_margin:
            return float(st.zone_park_margin[zid])
        if zid in st.zone_track_delta:
            return -float(st.zone_track_delta[zid])
        return None

    def _update_zone_estimators(
        self,
        zone: ZoneRuntime,
        now_ts: float,
        local_hour: float,
        *,
        solar_scale: float = 1.0,
    ) -> None:
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

        off_s = None if zone.all_off_since is None else now_ts - zone.all_off_since
        if (
            off_s is not None
            and off_s >= TRANSIENT_SETTLE_S
            and self.outdoor_source != "climatology"
        ):
            t_house = self._house_other_temp(zone.config.zone_id)
            expected = zone.model.free_float_rate(
                smoothed,
                self.t_out,
                local_hour,
                zone.door_open,
                t_house,
                indoor_fans_on=zone.indoor_fans_on,
                outdoor_exhaust_on=zone.outdoor_exhaust_on,
                include_transient=False,
                solar_scale=solar_scale,
            )
            zone.model.update_q_transient(dtdt - expected)
        else:
            zone.model.forget_q_transient(dt_h)

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
                    solar_scale=solar_scale,
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
                    indoor_fans_on=zone.indoor_fans_on,
                    outdoor_exhaust_on=zone.outdoor_exhaust_on,
                    solar_scale=solar_scale,
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
                    indoor_fans_on=zone.indoor_fans_on,
                    outdoor_exhaust_on=zone.outdoor_exhaust_on,
                    solar_scale=solar_scale,
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
                    solar_scale=solar_scale,
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
            solar_scale=solar_scale,
        )
        zone.sensible_ts = now_ts
        if zone.config.zone_id not in self.controller_state.zone_parked_since:
            zone.last_on_sensible_w = zone.sensible_w
        zone.latent_w = self._latent_power(zone, dt_h)
        self._conditioning_fit_refreshed = True

    def _update_park_learners(self, now_ts: float, local_hour: float) -> None:
        """Feed parked-zone learners once per control tick.

        Activity (compression vs fan-type park) is judged electrically with a
        PARK_DUTY_DEBOUNCE_S settle so brief meter blips do not flip duty.
        Extraction magnitude reuses the latest thermal sensible_w (updated on
        the 5 min fit); stale values from before park entry are skipped so
        honest p_ac cannot promote an idler via frozen full-conditioning W.
        """
        if self.manual_control:
            # Not commanding: never attribute user/remote setpoints to park learning.
            for zone in self.zones.values():
                zone.park_power.reset()
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
                other.config.zone_id != zid and (other.is_on or other.config.zone_id in parked_ids)
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
                if not settled:
                    extraction, active = 0.0, False
                elif stale_sensible:
                    # Compression is real; room-rate is unknown. Duty-only so
                    # we do not write ext=0 / active=True (West +3 K poison).
                    margin = self.controller_state.zone_park_margin.get(zid)
                    zone.park.update_duty(True, margin_k=margin)
                    continue
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
        solar_scale, rain_sink = self._forecast_q_hours(now_ts)
        volume_readings = self._volume_readings()
        aux_readings = self._aux_volume_readings()
        for zone in self.zones.values():
            free_float: tuple[float, ...] = ()
            pred_60m = None
            zid = zone.config.zone_id
            t_house = house_other_temperature(zid, volume_readings, aux_readings)
            if zone.temp is not None and forecast:
                t_house_hourly = self._house_other_hourly(zid, FORECAST_HOURS)
                trajectory = zone.model.predict_free(
                    zone.temp,
                    forecast,
                    local_hour,
                    hours=FORECAST_HOURS,
                    door_open=zone.door_open,
                    t_house_other=t_house,
                    t_house_hourly=t_house_hourly,
                    indoor_fans_on=zone.indoor_fans_on,
                    outdoor_exhaust_on=zone.outdoor_exhaust_on,
                    solar_scale_hourly=solar_scale,
                    rain_sink_hourly=rain_sink,
                )
                free_float = tuple(trajectory)
                if len(trajectory) > 1:
                    pred_60m = trajectory[1]
                # Free-float scores only while heads are off; recording during
                # Manual/adaptive conditioning just queues rows that get dropped.
                if zone.all_off_since is not None:
                    self._record_predictions(
                        now_ts,
                        zone,
                        forecast,
                        t_house,
                        t_house_hourly,
                        local_hour,
                        solar_scale,
                        rain_sink,
                    )
            zone.free_float = free_float
            zone.pred_60m = pred_60m
            zone.standing_load_w = self._standing_load_w(
                zone, solar_scale=solar_scale[0] if solar_scale else 1.0
            )
            mixing_gain_w = None
            mixing_coupling = None
            if zone.temp is not None and t_house is not None:
                _k_out, k_mix = zone.model._scaled_k(
                    zone.door_open,
                    indoor_fans_on=zone.indoor_fans_on,
                    outdoor_exhaust_on=zone.outdoor_exhaust_on,
                )
                mixing_coupling = zone.model.c_eff_wh_per_k * k_mix
                mixing_gain_w = mixing_coupling * (t_house - zone.temp)
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
                    park_residuals=zone.park.residuals,
                    park_extraction_w=zone.park.extraction_w,
                    park_preferred_margin_k=zone.park.preferred_margin_k,
                    park_fan_type_min_margin_k=zone.park.fan_type_min_margin_k(),
                    park_residual_max_margin_k=zone.park.residual_max_margin_k(),
                    park_residual_edge_k=zone.park.residual_edge_k(),
                    park_current_is_fan_type=zone.park.current_is_fan_type(
                        self.controller_state.zone_park_margin.get(zone.config.zone_id)
                    ),
                    park_margin_bins={k: list(v) for k, v in zone.park.margin_bins.items()},
                    park_samples=zone.park.samples,
                    head_mode=(
                        MODE_HEAT
                        if zone.head_state == STATE_HEATING
                        else MODE_COOL
                        if zone.head_state == STATE_COOLING
                        else None
                    ),
                    head_internal_temp=zone.head_internal_temp,
                    device_setpoint=zone.device_setpoint,
                    standing_load_w=zone.standing_load_w,
                    mixing_gain_w=mixing_gain_w,
                    mixing_coupling_w_per_k=mixing_coupling,
                    q_hvac_k_per_h=self._q_hvac_k_per_h(zone),
                )
            )
        mode_hint = self.controller_state.mode
        cop_banded_n: dict[int, float] = {}
        if mode_hint in (MODE_HEAT, MODE_COOL):
            cop_hints = power.cop_by_head_count(self.cop_table, mode_hint, COP_TABLE_MIN_SAMPLES)
            cop_by_band = power.cop_by_band(self.cop_table_banded, mode_hint, COP_TABLE_MIN_SAMPLES)
            cop_by_band_mode = mode_hint
            band = power.outdoor_band(self.t_out)
            if band is not None:
                cop_banded_n = power.cop_by_head_count_banded(
                    self.cop_table_banded, mode_hint, band, COP_TABLE_MIN_SAMPLES
                )
            cop_depth = power.cop_by_depth(self.cop_table_depth, mode_hint, COP_TABLE_MIN_SAMPLES)
        else:
            cop_hints = {}
            cop_by_band = {}
            cop_by_band_mode = None
            cop_depth = {}
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
            cop_by_head_count_banded=cop_banded_n,
            cop_by_depth=cop_depth,
            cop_by_band=cop_by_band,
            cop_by_band_mode=cop_by_band_mode,
            aux_indoor=tuple(aux_indoor),
            p_ac=self.p_ac,
            compression_floor_w=self._compression_floor_w(
                [z for z in self.zones.values() if z.is_on],
                {zid for zid, since in self.controller_state.zone_parked_since.items() if since},
            ),
        )

    def _record_predictions(
        self,
        now_ts: float,
        zone: ZoneRuntime,
        forecast: list[float],
        t_house: float | None,
        t_house_hourly: list[float] | None,
        local_hour: float,
        solar_scale_hourly: list[float] | None = None,
        rain_sink_hourly: list[float] | None = None,
    ) -> None:
        """Log a predicted-vs-actual sample at each scoring horizon.

        Uses the finer `predict_horizons` integrator (not the hourly
        `predict_free` trajectory) so 15/30-minute predictions land on the
        actual target time instead of the nearest hour.
        """
        preds = zone.model.predict_horizons(
            zone.temp,
            forecast,
            local_hour,
            HORIZONS_MIN,
            door_open=zone.door_open,
            t_house_other=t_house,
            t_house_hourly=t_house_hourly,
            indoor_fans_on=zone.indoor_fans_on,
            outdoor_exhaust_on=zone.outdoor_exhaust_on,
            solar_scale_hourly=solar_scale_hourly,
            rain_sink_hourly=rain_sink_hourly,
        )
        if not preds:
            return
        coeffs = self._coeffs_snapshot(zone)
        for horizon_min, predicted_t in preds.items():
            due_h = horizon_min / 60.0
            forecast_t_out = None
            if forecast:
                idx = min(int(due_h), len(forecast) - 1)
                frac = min(due_h - idx, 1.0)
                nxt = min(idx + 1, len(forecast) - 1)
                forecast_t_out = forecast[idx] * (1 - frac) + forecast[nxt] * frac
            self.predictor_scorer.record(
                now_ts,
                zone.config.zone_id,
                horizon_min,
                predicted_t,
                coeffs,
                forecast_t_out,
                t_house_used=t_house,
            )

    def _score_predictions(self, now_ts: float) -> None:
        """Resolve pending predictions whose horizon has elapsed (tick-rate is fine)."""
        zone_off_since = {zid: z.all_off_since for zid, z in self.zones.items()}
        actual_temp = {zid: z.temp for zid, z in self.zones.items() if z.temp is not None}
        self.predictor_scorer.score_due(now_ts, zone_off_since, actual_temp, self.t_out)

    def predictor_stats(self, zone_id: str) -> dict[int, dict]:
        """Per-horizon {bias_k, mae_k, n} for one zone (diagnostics/sensors)."""
        return self.predictor_scorer.all_stats(zone_id)

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
            zone = self.zones.get(zid)
            if zone is not None:
                zone.park_power.reset()

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
        self._update_free_float_deviation(snapshot)
        self._score_predictions(now_ts)
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
        head_depth = self._zone_head_depth_k(zid)
        track = st.zone_track_delta.get(zid)
        attrs: dict = {
            "last_reason": zone.last_control_reason or "none",
            "parked": parked,
            "track_delta_k": track,
            "head_depth_k": head_depth,
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
            attrs["park_fan_only_ratio"] = (
                None if zone.park.fan_only_ratio is None else round(zone.park.fan_only_ratio, 3)
            )
            attrs["park_fan_type_min_margin_k"] = zone.park.fan_type_min_margin_k()
            attrs["park_residual_max_margin_k"] = zone.park.residual_max_margin_k()
            attrs["park_residual_edge_k"] = zone.park.residual_edge_k()
            attrs["park_current_is_fan_type"] = zone.park.current_is_fan_type(
                st.zone_park_margin.get(zid)
            )
        return attrs

    def _update_free_float_deviation(self, snapshot: HouseSnapshot) -> None:
        """Mean (predicted no-AC trajectory - band centre) across zones, K.

        This is a *demand* indicator (how far the house wants to drift from
        target if left alone) -- distance-to-setpoint, not prediction error.
        It must never be read as model accuracy; for real predicted-vs-actual
        error see `predictor_scorer` / `predictor_stats()`.
        """
        deviations = []
        center = comfort.band_center(self.settings, self.t_rm)
        for zone in snapshot.zones:
            if zone.free_float:
                mean_traj = sum(zone.free_float) / len(zone.free_float)
                deviations.append(mean_traj - center)
        self.free_float_deviation = sum(deviations) / len(deviations) if deviations else None

    async def _async_execute(self, command) -> None:
        zone = self.zones.get(command.zone_id)
        if zone is None:
            return
        park_expressed = False
        if command.hvac_mode == MODE_OFF:
            zone.last_hold_depth_k = None
            zone.last_hold_device_sp = None
            zone.last_on_sensible_w = None
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
                # Signed head depth: cool SP = internal + depth, heat =
                # internal - depth. Positive = hysteresis hold; negative = chase.
                if command.head_depth_k is not None and isinstance(internal, (int, float)):
                    if command.park or command.head_depth_k > 0.0:
                        park_expressed = True
                    depth = float(command.head_depth_k)
                    if depth > 0.0:
                        raw = controller.hold_device_setpoint(
                            float(internal),
                            depth,
                            command.hvac_mode,
                            zone.last_hold_device_sp,
                            zone.last_hold_depth_k,
                        )
                    else:
                        zone.last_hold_depth_k = None
                        zone.last_hold_device_sp = None
                        raw = (
                            float(internal) + depth
                            if command.hvac_mode == MODE_COOL
                            else float(internal) - depth
                        )
                elif command.park and isinstance(internal, (int, float)):
                    park_expressed = True
                    margin = command.park_margin or controller.PARK_MARGIN_K
                    raw = controller.hold_device_setpoint(
                        float(internal),
                        margin,
                        command.hvac_mode,
                        zone.last_hold_device_sp,
                        zone.last_hold_depth_k,
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
                if (
                    command.head_depth_k is not None
                    and float(command.head_depth_k) > 0.0
                ):
                    zone.last_hold_depth_k = float(command.head_depth_k)
                    zone.last_hold_device_sp = setpoint
                elif command.park:
                    zone.last_hold_depth_k = float(
                        command.park_margin or controller.PARK_MARGIN_K
                    )
                    zone.last_hold_device_sp = setpoint
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
            "house_wh": self.energy.house_wh,
            "baseline_coverage": self.baseline.coverage(),
            "forecast": self.forecast,
            "cop_table": dict(self.cop_table),
            "cop_table_banded": dict(self.cop_table_banded),
            "cop_table_state": dict(self.cop_table_state),
            "cop_table_depth": dict(self.cop_table_depth),
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
                    "park": z.park.to_dict()
                    | {
                        "classification": z.park.classification,
                        "park_fan_type_min_margin_k": z.park.fan_type_min_margin_k(),
                        "park_residual_max_margin_k": z.park.residual_max_margin_k(),
                        "park_residual_edge_k": z.park.residual_edge_k(),
                        "park_current_is_fan_type": z.park.current_is_fan_type(
                            self.controller_state.zone_park_margin.get(zid)
                        ),
                    },
                    "confidence": z.model.confidence(z.door_open),
                    "drift": {h: d.to_dict() for h, d in z.drift.items()},
                    "sensible_w": z.sensible_w,
                    "latent_w": z.latent_w,
                    "allocated_w": z.allocated_w,
                    "standing_load_w": (
                        None if z.standing_load_w is None else round(z.standing_load_w, 1)
                    ),
                    "disturbance": z.model.disturbance_diag(self._local_hour(), z.door_open),
                    "predictor": {
                        "stats_by_horizon_min": self.predictor_stats(zid),
                        "pending": self.predictor_scorer.pending_count(zid),
                    },
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
