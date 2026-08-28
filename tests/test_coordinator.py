"""Coordinator command execution: room↔head frame, Manual, unexpressible hold."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from custom_components.adaptive_comfort.coordinator import (
    AdaptiveComfortRuntime,
    ZoneRuntime,
)
from custom_components.adaptive_comfort.core.types import (
    MODE_COOL,
    MODE_HEAT,
    PRESET_MANUAL,
    PRESET_NONE,
    STATE_COOLING,
    Command,
    ControllerState,
    RoomConfig,
    Settings,
    ZoneConfig,
)

HEAD = "climate.a"


def _climate(internal, hvac_mode="cool"):
    return SimpleNamespace(
        state=hvac_mode,
        attributes={
            "current_temperature": internal,
            "min_temp": 16.0,
            "max_temp": 30.0,
            "target_temp_step": 0.5,
            "hvac_modes": ["off", "cool", "heat", "fan_only"],
        },
    )


def _runtime(climates: dict | None = None, heads=(HEAD,)):
    cfg = ZoneConfig(
        zone_id="z",
        name="z",
        heads=heads,
        rooms=(RoomConfig(12.0, 2.5),),
    )
    zone = ZoneRuntime(cfg)
    rt = AdaptiveComfortRuntime.__new__(AdaptiveComfortRuntime)
    rt.settings = Settings()
    rt.controller_state = ControllerState()
    rt.zones = {"z": zone}
    rt._preset_before_manual = None
    calls: list[tuple] = []

    async def async_call(domain, service, data, blocking=False):
        calls.append((domain, service, dict(data)))

    climates = climates if climates is not None else {HEAD: _climate(20.0)}
    rt.hass = SimpleNamespace(
        states=SimpleNamespace(get=climates.get),
        services=SimpleNamespace(async_call=async_call),
    )
    rt._calls = calls
    return rt, zone


def _setpoints(rt) -> list[float]:
    return [data["temperature"] for _d, svc, data in rt._calls if svc == "set_temperature"]


def test_execute_positive_depth_translates_internal_plus_depth():
    rt, zone = _runtime()
    cmd = Command("z", MODE_COOL, 23.0, "park", head_depth_k=1.5)
    asyncio.run(AdaptiveComfortRuntime._async_execute(rt, cmd))
    assert _setpoints(rt) == [21.5]
    assert zone.last_hold_device_sp == 21.5
    assert zone.last_hold_depth_k == 1.5


def test_execute_positive_depth_does_not_reanchor_falling_internal():
    """West ladder: live internal falling must not rewrite a frozen hold SP."""
    rt, _zone = _runtime({HEAD: _climate(20.0)})
    cmd = Command("z", MODE_COOL, 23.0, "park", head_depth_k=1.5)
    asyncio.run(AdaptiveComfortRuntime._async_execute(rt, cmd))
    assert _setpoints(rt) == [21.5]

    rt.hass.states.get = {HEAD: _climate(18.0)}.get
    rt._calls.clear()
    asyncio.run(AdaptiveComfortRuntime._async_execute(rt, cmd))
    assert _setpoints(rt) == [21.5]


def test_execute_heat_positive_depth_freezes_last_commanded_sp():
    rt, _zone = _runtime({HEAD: _climate(20.0, hvac_mode="heat")})
    cmd = Command("z", MODE_HEAT, 21.0, "park", head_depth_k=1.5)
    asyncio.run(AdaptiveComfortRuntime._async_execute(rt, cmd))
    assert _setpoints(rt) == [18.5]
    rt.hass.states.get = {HEAD: _climate(22.0, hvac_mode="heat")}.get
    rt._calls.clear()
    asyncio.run(AdaptiveComfortRuntime._async_execute(rt, cmd))
    assert _setpoints(rt) == [18.5]


def test_execute_negative_depth_chases_live_internal():
    rt, _zone = _runtime({HEAD: _climate(20.0)})
    cmd = Command("z", MODE_COOL, 23.0, "demand", head_depth_k=-0.5, track_delta=0.5)
    asyncio.run(AdaptiveComfortRuntime._async_execute(rt, cmd))
    assert _setpoints(rt) == [19.5]
    rt.hass.states.get = {HEAD: _climate(18.0)}.get
    rt._calls.clear()
    asyncio.run(AdaptiveComfortRuntime._async_execute(rt, cmd))
    assert _setpoints(rt) == [17.5]


def test_execute_hold_without_internal_does_not_drift_fallback():
    """Missing internal on a positive-depth command must not use room+drift."""
    rt, zone = _runtime({HEAD: _climate(None)})
    zone.drift[HEAD].offsets[STATE_COOLING] = -3.0
    rt.controller_state.zone_head_depth_k["z"] = 1.5
    rt.controller_state.zone_parked_since["z"] = 1_000_000.0
    cmd = Command("z", MODE_COOL, 23.0, "park", head_depth_k=1.5)
    asyncio.run(AdaptiveComfortRuntime._async_execute(rt, cmd))
    assert _setpoints(rt) == []
    assert "z" not in rt.controller_state.zone_head_depth_k
    assert "z" not in rt.controller_state.zone_parked_since


def test_execute_hold_uses_sibling_with_internal():
    other = "climate.b"
    climates = {HEAD: _climate(None), other: _climate(20.0)}
    rt, _zone = _runtime(climates, heads=(HEAD, other))
    cmd = Command("z", MODE_COOL, 23.0, "park", head_depth_k=1.5)
    asyncio.run(AdaptiveComfortRuntime._async_execute(rt, cmd))
    temps = {
        (data["entity_id"], data["temperature"])
        for _d, svc, data in rt._calls
        if svc == "set_temperature"
    }
    assert temps == {(other, 21.5)}


def test_manual_enter_clears_leftover_depth_without_session():
    """B1 leftover: positive depth with no parked_since still drops on Manual."""
    rt, zone = _runtime()
    zone.last_hold_device_sp = 21.5
    zone.last_hold_depth_k = 1.5
    rt.controller_state.zone_head_depth_k["z"] = 1.5
    rt.controller_state.zone_track_delta["z"] = 0.5
    rt.settings.preset = PRESET_NONE
    rt.set_preset(PRESET_MANUAL)
    assert rt.settings.preset == PRESET_MANUAL
    assert "z" not in rt.controller_state.zone_head_depth_k
    assert "z" not in rt.controller_state.zone_track_delta
    assert zone.last_hold_device_sp is None
    assert zone.last_hold_depth_k is None


def test_manual_enter_clears_active_hold_session():
    rt, zone = _runtime()
    rt.controller_state.zone_parked_since["z"] = 1.0
    rt.controller_state.zone_park_margin["z"] = 1.5
    rt.controller_state.zone_head_depth_k["z"] = 1.5
    rt.settings.preset = PRESET_NONE
    rt.set_preset(PRESET_MANUAL)
    assert "z" not in rt.controller_state.zone_parked_since
    assert "z" not in rt.controller_state.zone_head_depth_k
    assert zone.last_hold_device_sp is None


def test_zone_control_attrs_prefers_signed_head_depth():
    rt, zone = _runtime()
    rt.controller_state.zone_head_depth_k["z"] = 0.0
    rt.controller_state.zone_track_delta["z"] = 0.5
    attrs = rt.zone_control_attrs(zone)
    assert attrs["head_depth_k"] == 0.0
    assert attrs["track_delta_k"] == 0.5
