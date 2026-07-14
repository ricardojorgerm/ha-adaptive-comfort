"""Shared dataclasses for the Adaptive Comfort core."""

from __future__ import annotations

from dataclasses import dataclass, field

# Head operating-state categories used for drift learning and gating.
STATE_STANDBY = "standby"
STATE_FAN_ONLY = "fan_only"
STATE_HEATING = "heating"
STATE_COOLING = "cooling"
HEAD_STATES = (STATE_STANDBY, STATE_FAN_ONLY, STATE_HEATING, STATE_COOLING)

MODE_OFF = "off"
MODE_HEAT = "heat"
MODE_COOL = "cool"
MODE_AUTO = "auto"
MODE_FAN = "fan_only"

PRESET_NONE = "none"
PRESET_ECO = "eco"
PRESET_AWAY = "away"
PRESET_BOOST = "boost"

ROOM_TYPE_REGULAR = "regular"
ROOM_TYPE_WET = "wet"


@dataclass(frozen=True)
class RoomConfig:
    """One physical room hosting one AC head."""

    area_m2: float
    height_m: float = 2.6

    @property
    def volume_m3(self) -> float:
        return self.area_m2 * self.height_m


@dataclass(frozen=True)
class ZoneConfig:
    """A climatized area: N mirrored heads, one room per head.

    Rooms are never merged: the thermal fit runs on the sensed room
    (index 0 by convention) and extensive quantities are summed over rooms.
    """

    zone_id: str
    name: str
    heads: tuple[str, ...]
    rooms: tuple[RoomConfig, ...]
    temp_sensor: str | None = None
    humidity_sensor: str | None = None
    door_sensor: str | None = None
    presence_sensor: str | None = None

    @property
    def n_rooms(self) -> int:
        return len(self.rooms)

    @property
    def sensed_room(self) -> RoomConfig:
        return self.rooms[0]

    @property
    def total_volume_m3(self) -> float:
        return sum(r.volume_m3 for r in self.rooms)


@dataclass(frozen=True)
class UnconditionedRoom:
    name: str
    area_m2: float
    room_type: str = ROOM_TYPE_REGULAR
    height_m: float = 2.6
    temp_sensor: str | None = None  # optional: contributes weak indoor evidence

    @property
    def volume_m3(self) -> float:
        return self.area_m2 * self.height_m


@dataclass(frozen=True)
class HouseConfig:
    grid_power_entity: str
    battery_power_entity: str | None = None
    battery_positive_discharging: bool = True
    known_load_entities: tuple[str, ...] = ()
    outdoor_temp_entity: str | None = None
    weather_entity: str | None = None
    presence_entity: str | None = None
    multisplit: bool = True
    contracted_kva: float = 3.45
    power_factor: float = 1.0
    default_target: float = 22.5
    unconditioned: tuple[UnconditionedRoom, ...] = ()


@dataclass
class Settings:
    """Runtime-tunable settings, backed by HA number/switch/climate entities."""

    target: float = 22.5
    band_k: float = 0.7
    min_on_min: float = 20.0
    min_off_min: float = 10.0
    contracted_kva: float = 3.45
    power_factor: float = 1.0
    shed_start_pct: float = 0.92
    shed_restore_pct: float = 0.75
    adaptive_blend: float = 0.3
    coordination: bool = True
    presence_adaptation: bool = True
    shedding_enabled: bool = True
    multisplit: bool = True
    fan_assist: bool = True  # fan-only for opposite-demand zones on a multi-split
    # Delay mechanical cooling while a window would beat it (grace period).
    # The ventilation suggestion itself is always reported regardless.
    window_suggest: bool = False
    hvac_mode: str = MODE_AUTO
    preset: str = PRESET_NONE
    zone_offsets: dict[str, float] = field(default_factory=dict)
    zone_enabled: dict[str, bool] = field(default_factory=dict)

    @property
    def limit_w(self) -> float:
        # 3.45 kVA -> 3450 W at PF 1.0 (Portuguese contracted power convention).
        return self.contracted_kva * 1000.0 * self.power_factor


@dataclass
class ZoneSnapshot:
    """Per-zone inputs for one controller tick (precomputed by the runtime)."""

    zone_id: str
    name: str
    n_rooms: int
    temp: float | None  # corrected room temperature
    rh: float | None = None
    door_open: bool = False
    occupied: bool | None = None  # None = no presence sensor
    is_on: bool = False  # any head actively conditioning
    head_state: str = STATE_STANDBY
    pred_60m: float | None = None  # free-float prediction 60 min ahead
    free_float: tuple[float, ...] = ()  # hourly free-float trajectory (24 h)
    confidence: float = 0.0
    draw_w: float | None = None  # learned total electrical draw of the zone
    enabled: bool = True


@dataclass
class HouseSnapshot:
    """Everything the controller needs for one tick."""

    now_ts: float
    local_hour: float
    settings: Settings
    zones: list[ZoneSnapshot]
    t_out: float | None = None
    t_rm: float | None = None  # 7-day running-mean outdoor temperature
    house_occupied: bool | None = None
    p_grid: float | None = None
    p_grid_over_since: float | None = None  # ts since grid power > shed threshold
    forecast_hours: tuple[float, ...] = ()  # hourly outdoor forecast, aligned with free_float
    cop_by_head_count: dict[int, float] = field(default_factory=dict)  # empirical COP hints
    # Temperatures from unconditioned rooms as (temp, weight) pairs; weak
    # extra indoor evidence for cold-start mode arbitration.
    aux_indoor: tuple[tuple[float, float], ...] = ()


@dataclass
class Command:
    """Zone-level actuation request; the runtime fans it out to mirrored heads."""

    zone_id: str
    hvac_mode: str  # MODE_OFF | MODE_HEAT | MODE_COOL
    setpoint: float | None = None  # desired *room* temperature (pre drift translation)
    reason: str = ""


@dataclass
class ControllerState:
    """Persistent controller memory carried between ticks."""

    mode: str = MODE_OFF  # active dominant mode: off/heat/cool
    mode_since: float = 0.0
    override_mode: str | None = None
    override_since: float = 0.0
    zone_on: dict[str, bool] = field(default_factory=dict)
    zone_since: dict[str, float] = field(default_factory=dict)  # last on/off transition
    zone_last_cmd: dict[str, float] = field(default_factory=dict)
    zone_last_setpoint: dict[str, float] = field(default_factory=dict)
    zone_mode_changes: dict[str, list[float]] = field(default_factory=dict)
    zone_fan: dict[str, bool] = field(default_factory=dict)  # fan-assist active
    zone_last_cool: dict[str, float] = field(default_factory=dict)  # coil-wet lockout
    window_suggest_since: dict[str, float] = field(default_factory=dict)
    shed: dict[str, float] = field(default_factory=dict)  # zone_id -> shed ts
    last_shed_action: float = 0.0

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "mode_since": self.mode_since,
            "zone_on": dict(self.zone_on),
            "zone_since": dict(self.zone_since),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ControllerState:
        st = cls()
        st.mode = data.get("mode", MODE_OFF)
        st.mode_since = data.get("mode_since", 0.0)
        st.zone_on = dict(data.get("zone_on", {}))
        st.zone_since = dict(data.get("zone_since", {}))
        return st


@dataclass
class Decision:
    """Controller output for one tick."""

    commands: list[Command]
    state: ControllerState
    diag: dict
    # Zones where opening a window would beat mechanical cooling right now.
    window_suggestions: list[str] = field(default_factory=list)
