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
PRESET_MANUAL = "manual"

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
    # House-level: vacant house → Away preset; homecoming clears track deltas.
    presence_adaptation: bool = True
    # Zone-level: vacant widen, vacant band-hold, skip vacant helpers / fan-assist,
    # and down-weight vacant zones in mode integrals. Off → treat occupancy as
    # unknown for those paths (sensors still update for diagnostics).
    zone_presence_adaptation: bool = True
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
    # hold residual compression, then parking replaces hard-off when
    # residual output can carry a satisfied zone's standing load
    # (multi-split, siblings on).
    park_learning: bool = True
    # Regime policy: per-tick choice among 'ventilate' (outdoor beats the
    # compressor: gate cooling demand), 'continuous' (load can feed the
    # compressor floor: hold zones parked instead of cycling them off), and
    # 'cycling' (default behavior with true offs and drained coils).
    auto_regime: bool = True
    # Overnight: when outdoor air is at/near the coolest target, force the
    # ventilate / cycling path instead of continuous park-holds (opt-in;
    # field data showed ~280 W overnight compression against cool outdoor air).
    night_ventilate: bool = False
    # Opt-in: keep exactly one already-loaded "anchor" head parked instead of
    # released to off, so the compressor keeps running continuously for the
    # house's aggregate standing load even when every zone is individually
    # satisfied. Other zones cycle/off normally - this is not the per-zone
    # 'continuous' regime (which parks whichever zones justify it on their
    # own load), it is a single elected anchor. See controller._update_anchor.
    prefer_continuous: bool = False
    # Per-open-head electrical fan floor (W). Gate = 20 + this * heads.
    fan_floor_per_head_w: float = 55.0
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
    door_open: bool = True  # no door sensor → assume open (inter-room mixing)
    occupied: bool | None = None  # None = no presence sensor
    is_on: bool = False  # any head actively conditioning
    head_state: str = STATE_STANDBY
    pred_60m: float | None = None  # free-float prediction 60 min ahead
    free_float: tuple[float, ...] = ()  # hourly free-float trajectory (24 h)
    confidence: float = 0.0
    draw_w: float | None = None  # learned total electrical draw of the zone
    enabled: bool = True
    # Park-behavior knowledge (from ParkEstimator): None until classified.
    park_residuals: bool | None = None
    # Parked observations accumulated so far (probe bookkeeping: a probe
    # only counts against the probe budget if it produced observations).
    park_samples: int = 0
    # Physical conditioning direction of the head right now (MODE_COOL /
    # MODE_HEAT / None), independent of the house's dominant mode. Lets the
    # controller keep tracking a zone that is running out its minimum
    # runtime after the house has gone idle.
    head_mode: str | None = None
    # Live head-frame readings (mean across mirrored heads) for depth re-anchor.
    head_internal_temp: float | None = None
    device_setpoint: float | None = None
    park_extraction_w: float | None = None  # sensed-room / per-head parked output (W)
    # Learned park depth (K) to start from on the next park entry.
    park_preferred_margin_k: float | None = None
    # Shallowest (min) fan-type shelf — mode on, no compression (diagnostics).
    park_fan_type_min_margin_k: float | None = None
    # Deepest (max) still-compressing residual bin (diagnostics).
    park_residual_max_margin_k: float | None = None
    # Entry target: residual↔fan-type edge (bisect of max residual and min
    # fan-type, finer than the 0.5 K grid).
    park_residual_edge_k: float | None = None
    # Is the *current* session margin fan-type? (live electrical class).
    park_current_is_fan_type: bool | None = None
    # Hysteresis map copy for depth selection: bin -> [ext_w, duty, n].
    park_margin_bins: dict[str, list[float]] = field(default_factory=dict)
    # Estimated standing heat load of the *zone* at current conditions
    # (W, >=0, sensed-room inflow x n_rooms). Park extraction is per-head;
    # exploit compares extraction to standing_load_w / n_rooms.
    standing_load_w: float | None = None
    # Signed house-mixing heat flow into the sensed room at current
    # conditions: c_eff * k_mix * (T_house_other - T_zone), sensed-room /
    # per-head frame (W). Positive = heat flowing from the house into this
    # zone. Used to judge whether a sibling's conditioning already covers
    # this zone's own standing load via mixing alone (free-ride policy).
    mixing_gain_w: float | None = None
    # Raw mixing coupling strength c_eff * k_mix (W/K), independent of the
    # instantaneous ΔT - a "how thermally central is this room to the rest
    # of the house" proxy (used as an anchor-selection tie-break).
    mixing_coupling_w_per_k: float | None = None


@dataclass
class HouseSnapshot:
    """Everything the controller needs for one tick."""

    now_ts: float
    local_hour: float
    settings: Settings
    zones: list[ZoneSnapshot]
    t_out: float | None = None
    # True when t_out is climatology after a configured outdoor source dropped
    # out (not a virgin install that never had outdoor). Discretionary
    # actuation (ventilate, window grace) must not trust this number.
    t_out_synthetic: bool = False
    t_rm: float | None = None  # 7-day running-mean outdoor temperature
    house_occupied: bool | None = None
    p_grid: float | None = None
    p_demand: float | None = None  # conservative demand used for shedding
    p_grid_over_since: float | None = None  # s demand has exceeded shed threshold
    shed_urgent: bool = False  # immediate shed (critical overload or known-load spike)
    forecast_hours: tuple[float, ...] = ()  # hourly outdoor forecast, aligned with free_float
    cop_by_head_count: dict[int, float] = field(default_factory=dict)  # empirical COP hints
    # Empirical COP by outdoor band ('mild'|'warm'|'hot') for the mode in
    # ``cop_by_band_mode`` (sample-gated, aggregated across head counts).
    # COP-timed widen: advance/defer predictive demand with a bounded wider
    # half-band (center fixed). Not a center-shift precool.
    cop_by_band: dict[str, float] = field(default_factory=dict)
    # Mode the ``cop_by_band`` map was filtered for (None when empty / idle).
    # Arbitrage must ignore the map when this disagrees with the tick's mode.
    cop_by_band_mode: str | None = None
    # Temperatures from unconditioned rooms as (temp, weight) pairs; weak
    # extra indoor evidence for cold-start mode arbitration.
    aux_indoor: tuple[tuple[float, float], ...] = ()
    # Electrical AC residual (W) and the live fan-floor used as the compression
    # threshold — plant-level min_on keys off these, not per-zone timers.
    p_ac: float | None = None
    compression_floor_w: float = 75.0


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
    # Signed head depth (K): cool SP = internal + head_depth_k, heat
    # SP = internal - head_depth_k. Negative = chase track; positive =
    # hysteresis residual hold. When set, the runtime prefers this over
    # separate track_delta / park_margin translation.
    head_depth_k: float | None = None


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
    # Last time the zone left the window-suggest pool; grace clock is kept
    # across brief disqualification until this ages past WINDOW_GRACE_S.
    window_suggest_out_since: dict[str, float] = field(default_factory=dict)
    zone_track_delta: dict[str, float] = field(default_factory=dict)  # tracking-control depth (K)
    zone_parked_since: dict[str, float] = field(default_factory=dict)  # parked-state entry time
    zone_last_park_probe: dict[str, float] = field(default_factory=dict)  # consumed-probe spacing
    zone_last_park_abort: dict[str, float] = field(
        default_factory=dict
    )  # stillborn-probe retry spacing
    zone_park_probe_entry: dict[str, int] = field(
        default_factory=dict
    )  # park_samples at probe entry
    regime: str = "cycling"  # 'ventilate' | 'continuous' | 'cycling'
    regime_since: float = 0.0
    zone_park_margin: dict[str, float] = field(
        default_factory=dict
    )  # adaptive margin above internal (K)
    zone_park_ref: dict[str, float] = field(
        default_factory=dict
    )  # room temp at last margin decision
    # Working copy of learned park depth; synced to ParkEstimator after each tick.
    zone_park_preferred: dict[str, float] = field(default_factory=dict)
    # Signed head depth continuum (K): cool SP = internal + depth.
    zone_head_depth_k: dict[str, float] = field(default_factory=dict)
    # Deprecated: kept for persistence round-trip of older saves.
    zone_force_chase_until: dict[str, float] = field(default_factory=dict)
    shed: dict[str, float] = field(default_factory=dict)  # zone_id -> shed ts
    last_shed_action: float = 0.0
    # prefer_continuous: the single elected anchor zone (None when the
    # policy is off or no zone currently qualifies).
    anchor_zone: str | None = None
    anchor_since: float = 0.0
    # sibling-sustain: observed, ORDERED (rider_zone -> lead_zone -> [ewma
    # in-band ratio, samples]) evidence that a zone stays in-band via house
    # mixing while a specific sibling conditions. Opportunistic-only: never
    # forces a reverse-direction probe, purely accumulates from ticks where
    # the rider was already off/free-riding.
    sibling_sustain: dict[str, dict[str, tuple[float, int]]] = field(default_factory=dict)
    # Plant compression run clock (epoch s). Set on rising edge of
    # p_ac >= compression_floor; cleared after START_DEBOUNCE_S below the
    # floor (brief inverter dips do not reset min_on). Enforced against
    # this clock, not per-zone zone_since.
    plant_compress_since: float = 0.0
    plant_below_since: float = 0.0  # first sub-floor sample while run live
    # COP-timed efficiency band: hysteretic half-band widen (K) and direction
    # ('advance'|'defer'|'none'). Center stays fixed. Withdraw only after the
    # COP advantage falls below EXIT *and* every zone has crossed back inside
    # the unwidened band on the widen side — so a cool-advance bank cannot
    # snap the floor up under a room and look like a heat demand.
    cop_widen_k: float = 0.0
    cop_timing: str = "none"
    # Mode that engaged the widen (heat/cool); held across ticks so a false
    # opposite-mode flip cannot restretch the wrong edge while recovering.
    cop_widen_mode: str | None = None

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
            "zone_park_ref": dict(self.zone_park_ref),
            "zone_park_preferred": dict(self.zone_park_preferred),
            "zone_head_depth_k": dict(self.zone_head_depth_k),
            "zone_force_chase_until": dict(self.zone_force_chase_until),
            "zone_fan": dict(self.zone_fan),
            "zone_last_cool": dict(self.zone_last_cool),
            "zone_mode_changes": {k: list(v) for k, v in self.zone_mode_changes.items()},
            "window_suggest_since": dict(self.window_suggest_since),
            "window_suggest_out_since": dict(self.window_suggest_out_since),
            "regime": self.regime,
            "regime_since": self.regime_since,
            "anchor_zone": self.anchor_zone,
            "anchor_since": self.anchor_since,
            "plant_compress_since": self.plant_compress_since,
            "plant_below_since": self.plant_below_since,
            "cop_widen_k": self.cop_widen_k,
            "cop_timing": self.cop_timing,
            "cop_widen_mode": self.cop_widen_mode,
            "sibling_sustain": {
                rider: {sib: [ratio, samples] for sib, (ratio, samples) in subs.items()}
                for rider, subs in self.sibling_sustain.items()
            },
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
        st.zone_park_ref = {str(k): float(v) for k, v in data.get("zone_park_ref", {}).items()}
        st.zone_park_preferred = {
            str(k): float(v) for k, v in data.get("zone_park_preferred", {}).items()
        }
        st.zone_head_depth_k = {
            str(k): float(v) for k, v in data.get("zone_head_depth_k", {}).items()
        }
        st.zone_force_chase_until = {
            str(k): float(v) for k, v in data.get("zone_force_chase_until", {}).items()
        }
        st.zone_fan = {str(k): bool(v) for k, v in data.get("zone_fan", {}).items()}
        st.zone_last_cool = {str(k): float(v) for k, v in data.get("zone_last_cool", {}).items()}
        st.zone_mode_changes = {
            str(k): [float(t) for t in v] for k, v in data.get("zone_mode_changes", {}).items()
        }
        st.window_suggest_since = {
            str(k): float(v) for k, v in data.get("window_suggest_since", {}).items()
        }
        st.window_suggest_out_since = {
            str(k): float(v) for k, v in data.get("window_suggest_out_since", {}).items()
        }
        st.regime = str(data.get("regime", "cycling"))
        st.regime_since = float(data.get("regime_since", 0.0))
        st.anchor_zone = data.get("anchor_zone")
        st.anchor_since = float(data.get("anchor_since", 0.0))
        st.plant_compress_since = float(data.get("plant_compress_since", 0.0))
        st.plant_below_since = float(data.get("plant_below_since", 0.0))
        st.cop_widen_k = max(0.0, float(data.get("cop_widen_k", 0.0)))
        timing = str(data.get("cop_timing", "none"))
        st.cop_timing = timing if timing in ("advance", "defer", "none") else "none"
        wmode = data.get("cop_widen_mode")
        st.cop_widen_mode = str(wmode) if wmode in (MODE_HEAT, MODE_COOL) else None
        if st.cop_widen_k <= 0.0 or st.cop_timing == "none":
            st.cop_widen_mode = None
        for rider, subs in data.get("sibling_sustain", {}).items():
            entry: dict[str, tuple[float, int]] = {}
            for sib, value in subs.items():
                try:
                    entry[str(sib)] = (float(value[0]), int(value[1]))
                except (TypeError, ValueError, IndexError):
                    continue
            if entry:
                st.sibling_sustain[str(rider)] = entry
        return st


@dataclass
class Decision:
    """Controller output for one tick."""

    commands: list[Command]
    state: ControllerState
    diag: dict
    # Zones where opening a window would beat mechanical cooling right now.
    window_suggestions: list[str] = field(default_factory=list)
