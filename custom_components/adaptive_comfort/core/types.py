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
    indoor_fan_entities: tuple[str, ...] = ()
    outdoor_exhaust_fan_entities: tuple[str, ...] = ()

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
    shed_start_pct: float = 0.88
    shed_restore_pct: float = 0.75
    adaptive_blend: float = 0.3
    coordination: bool = True
    presence_adaptation: bool = True
    shedding_enabled: bool = True
    multisplit: bool = True
    fan_assist: bool = True  # fan-only for opposite-demand zones on a multi-split
    # Tracking setpoint control: command device setpoints relative to the
    # head's *live* internal sensor (internal - delta) instead of a static
    # drift translation, so the inverter sees a small, steady error and
    # modulates at low speed instead of ramping and self-terminating.
    tracking: bool = True
    # Characterize and exploit above-setpoint head behavior instead of
    # assuming it: bounded probes measure whether parked heads idle or
    # trickle, then parking replaces hard-off when trickle output can
    # carry a satisfied zone's standing load (multi-split, siblings on).
    park_learning: bool = True
    # Hold mechanical conditioning briefly while the user is expected to ventilate manually.
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
    # Park-behavior knowledge (from ParkEstimator): None until classified.
    park_trickles: bool | None = None
    # Parked observations accumulated so far (probe bookkeeping: a probe
    # only counts against the probe budget if it produced observations).
    park_samples: int = 0
    # Physical conditioning direction of the head right now (MODE_COOL /
    # MODE_HEAT / None), independent of the house's dominant mode. Lets the
    # controller keep tracking a zone that is running out its minimum
    # runtime after the house has gone idle.
    head_mode: str | None = None
    park_extraction_w: float | None = None
    # Learned park depth (K) to start from on the next park entry.
    park_preferred_margin_k: float | None = None
    # Estimated standing heat load of the zone at current conditions (W, >=0);
    # lets the controller judge whether trickle output can carry the zone.
    standing_load_w: float | None = None


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
    p_demand: float | None = None  # conservative demand used for shedding
    p_grid_over_since: float | None = None  # s demand has exceeded shed threshold
    shed_urgent: bool = False  # immediate shed (critical overload or known-load spike)
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
    # When set, the runtime commands the head at (internal - track_delta) in
    # cooling / (internal + track_delta) in heating, falling back to the
    # drift translation of `setpoint` if the internal reading is unusable.
    track_delta: float | None = None
    # Park the head: command a setpoint just above its internal reading
    # while keeping the compressor mode, to observe/exploit the device's
    # own keep-temperature behavior instead of turning off.
    park: bool = False
    # Adaptive park depth: setpoint rides internal + margin (cool) or
    # internal - margin (heat). Escalated by the controller while the room
    # keeps moving in the conditioning direction despite being parked.
    park_margin: float | None = None


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
    zone_track_delta: dict[str, float] = field(default_factory=dict)  # tracking-control depth (K)
    zone_parked_since: dict[str, float] = field(default_factory=dict)  # parked-state entry time
    zone_last_park_probe: dict[str, float] = field(default_factory=dict)  # consumed-probe spacing
    zone_last_park_abort: dict[str, float] = field(
        default_factory=dict
    )  # stillborn-probe retry spacing
    zone_park_probe_entry: dict[str, int] = field(
        default_factory=dict
    )  # park_samples at probe entry
    zone_park_margin: dict[str, float] = field(
        default_factory=dict
    )  # adaptive margin above internal (K)
    zone_park_ref: dict[str, float] = field(
        default_factory=dict
    )  # room temp at last margin decision
    # Working copy of learned park depth; synced to ParkEstimator after each tick.
    zone_park_preferred: dict[str, float] = field(default_factory=dict)
    shed: dict[str, float] = field(default_factory=dict)  # zone_id -> shed ts
    last_shed_action: float = 0.0

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "mode_since": self.mode_since,
            "zone_on": dict(self.zone_on),
            "zone_since": dict(self.zone_since),
            "zone_track_delta": dict(self.zone_track_delta),
            "zone_parked_since": dict(self.zone_parked_since),
            "zone_last_park_probe": dict(self.zone_last_park_probe),
            "zone_park_probe_entry": dict(self.zone_park_probe_entry),
            "zone_park_margin": dict(self.zone_park_margin),
            "zone_last_park_abort": dict(self.zone_last_park_abort),
            "zone_park_preferred": dict(self.zone_park_preferred),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ControllerState:
        st = cls()
        st.mode = data.get("mode", MODE_OFF)
        st.mode_since = data.get("mode_since", 0.0)
        st.zone_on = dict(data.get("zone_on", {}))
        st.zone_since = dict(data.get("zone_since", {}))
        st.zone_track_delta = {
            str(k): float(v) for k, v in data.get("zone_track_delta", {}).items()
        }
        st.zone_parked_since = {
            str(k): float(v) for k, v in data.get("zone_parked_since", {}).items()
        }
        st.zone_last_park_probe = {
            str(k): float(v) for k, v in data.get("zone_last_park_probe", {}).items()
        }
        st.zone_park_probe_entry = {
            str(k): int(v) for k, v in data.get("zone_park_probe_entry", {}).items()
        }
        st.zone_park_margin = {
            str(k): float(v) for k, v in data.get("zone_park_margin", {}).items()
        }
        st.zone_last_park_abort = {
            str(k): float(v) for k, v in data.get("zone_last_park_abort", {}).items()
        }
        st.zone_park_preferred = {
            str(k): float(v) for k, v in data.get("zone_park_preferred", {}).items()
        }
        return st


@dataclass
class Decision:
    """Controller output for one tick."""

    commands: list[Command]
    state: ControllerState
    diag: dict
    # Zones where opening a window would beat mechanical cooling right now.
    window_suggestions: list[str] = field(default_factory=list)
