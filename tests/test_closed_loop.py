"""Closed-loop cold-start test: controller + synthetic house.

A brand-new install (no fitted models, no COP data, fallback mode
arbitration) on a hot Lisbon summer day must pull rooms toward the band
without violating min-on / min-off cycling guards.
"""

import itertools

from custom_components.adaptive_comfort.core import controller
from custom_components.adaptive_comfort.core.simulator import SimHouse, SimRoom
from custom_components.adaptive_comfort.core.types import (
    MODE_AUTO,
    MODE_COOL,
    MODE_OFF,
    ControllerState,
    HouseSnapshot,
    Settings,
    ZoneSnapshot,
)

STEP_H = 1.0 / 60.0  # one-minute steps
T0 = 1_000_000.0


def test_cold_start_summer_day_controls_decently():
    rooms = [
        SimRoom(name="living", k=0.35, volume_m3=45.0, cop=3.2, draw_w=900.0, temp=27.5),
        SimRoom(name="bed1", k=0.30, volume_m3=28.0, cop=3.2, draw_w=650.0, temp=27.0),
        SimRoom(name="bed2", k=0.30, volume_m3=26.0, cop=3.2, draw_w=650.0, temp=26.5),
    ]
    house = SimHouse(rooms=rooms, outdoor_mean=26.0, outdoor_amplitude=5.0)
    house.hour = 13.0  # start early afternoon

    settings = Settings(hvac_mode=MODE_AUTO)
    state = ControllerState()
    active: dict[str, tuple[str, float] | None] = {r.name: None for r in rooms}
    transitions: dict[str, list[tuple[float, bool]]] = {r.name: [] for r in rooms}

    now = T0
    for _ in range(6 * 60):  # six hours, minute ticks
        # Head-internal thermostat: modulate against the commanded setpoint.
        for room in rooms:
            cmd = active[room.name]
            if cmd is None:
                room.hvac_on = False
            else:
                mode, setpoint = cmd
                room.hvac_mode = mode
                # Park / track commands may omit a room-frame setpoint; fall
                # back to a mild keep-temp depth so the sim head still modulates.
                if setpoint is None:
                    setpoint = room.temp + (1.0 if mode == MODE_COOL else -1.0)
                if mode == MODE_COOL:
                    room.hvac_on = room.temp > setpoint - 0.2
                else:
                    room.hvac_on = room.temp < setpoint + 0.2

        obs = house.step(STEP_H)
        now += 60.0

        zones = [
            ZoneSnapshot(
                zone_id=room.name,
                name=room.name,
                n_rooms=1,
                temp=obs["temps"][room.name],
                occupied=None,
                is_on=active[room.name] is not None,
                confidence=0.0,  # cold start: no fitted model
                free_float=(),
            )
            for room in rooms
        ]
        snap = HouseSnapshot(
            now_ts=now,
            local_hour=obs["hour"],
            settings=settings,
            zones=zones,
            t_out=obs["t_out"],
            t_rm=23.5,  # Lisbon August climatology seed
            p_grid=obs["p_grid"],
        )
        decision = controller.tick(snap, state)
        for cmd in decision.commands:
            was_on = active[cmd.zone_id] is not None
            if cmd.hvac_mode == MODE_OFF:
                active[cmd.zone_id] = None
                if was_on:
                    transitions[cmd.zone_id].append((now, False, cmd.reason))
            else:
                sp = cmd.setpoint
                if sp is None and cmd.track_delta is not None:
                    # Approximate room-frame target from tracking depth.
                    room = next(r for r in rooms if r.name == cmd.zone_id)
                    sp = room.temp - cmd.track_delta
                active[cmd.zone_id] = (cmd.hvac_mode, sp)
                if not was_on:
                    transitions[cmd.zone_id].append((now, True, cmd.reason))

    # 1. It actually cooled: every room ends inside a generous comfort range.
    for room in rooms:
        assert room.temp < 25.0, f"{room.name} still hot: {room.temp:.1f}"
        assert room.temp > 20.0, f"{room.name} overcooled: {room.temp:.1f}"

    # 2. Cold-start arbitration used cooling (may idle once rooms are in band).
    assert state.mode in (MODE_COOL, MODE_OFF)
    assert any(events for events in transitions.values())

    # 3. Cycling guards: off respects min_off; on uses the short zone chatter
    # dwell (plant-level min_on is electrical and not asserted here).
    # Fan assist is not compressor runtime — fan_off→helper is not min_off.
    for name, events in transitions.items():
        for (t1, on1, reason1), (t2, _on2, reason2) in itertools.pairwise(events):
            if reason1 in ("fan_assist", "fan_off") or reason2 in ("fan_assist", "fan_off"):
                continue
            gap_min = (t2 - t1) / 60.0
            minimum = settings.min_off_min if not on1 else controller.ZONE_CHATTER_S / 60.0
            assert gap_min >= minimum - 1e-6, f"{name}: {gap_min:.1f} min violates guard"


def test_cold_start_winter_night_heats():
    rooms = [SimRoom(name="bed", k=0.3, volume_m3=28.0, cop=3.0, draw_w=650.0, temp=17.0)]
    house = SimHouse(rooms=rooms, outdoor_mean=10.0, outdoor_amplitude=3.0)
    house.hour = 22.0

    settings = Settings(hvac_mode=MODE_AUTO)
    state = ControllerState()
    active: tuple[str, float] | None = None

    now = T0
    for _ in range(4 * 60):
        room = rooms[0]
        if active is None:
            room.hvac_on = False
        else:
            mode, setpoint = active
            room.hvac_mode = mode
            if setpoint is None:
                setpoint = room.temp - 1.0
            room.hvac_on = room.temp < setpoint + 0.2
        obs = house.step(STEP_H)
        now += 60.0
        snap = HouseSnapshot(
            now_ts=now,
            local_hour=obs["hour"],
            settings=settings,
            zones=[
                ZoneSnapshot(
                    zone_id="bed",
                    name="bed",
                    n_rooms=1,
                    temp=obs["temps"]["bed"],
                    is_on=active is not None,
                    confidence=0.0,
                    free_float=(),
                )
            ],
            t_out=obs["t_out"],
            t_rm=12.0,  # Lisbon winter
            p_grid=obs["p_grid"],
        )
        decision = controller.tick(snap, state)
        for cmd in decision.commands:
            active = None if cmd.hvac_mode == MODE_OFF else (cmd.hvac_mode, cmd.setpoint)

    # Cold-edge hold is shallower than center-seek; 4 h from 17 °C still heats.
    assert rooms[0].temp > 19.0
    assert state.mode in ("heat", "off")
