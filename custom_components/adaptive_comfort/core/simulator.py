"""Synthetic house used by the test suite.

Ground-truth rooms follow the same 1R1C physics the estimator assumes,
so tests can assert that estimators recover known parameters and that
the controller keeps comfort while honouring its guards.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


def sinusoidal_outdoor(hour: float, mean: float = 18.0, amplitude: float = 6.0) -> float:
    """Diurnal outdoor temperature peaking mid-afternoon (15h)."""
    return mean + amplitude * math.cos(2.0 * math.pi * (hour - 15.0) / 24.0)


@dataclass
class SimRoom:
    """Ground truth for one room."""

    name: str
    k: float = 0.3  # air-exchange coupling [1/h]
    volume_m3: float = 30.0
    furniture_factor: float = 4.0
    solar_amplitude: float = 0.4  # K/h peak of diurnal gain
    solar_peak_hour: float = 14.0
    internal_gain: float = 0.05  # K/h constant gain (electronics, heat soak)
    cop: float = 3.2
    draw_w: float = 700.0  # electrical draw of this room's head at full duty
    temp: float = 22.0
    hvac_on: bool = False
    hvac_mode: str = "cool"
    duty: float = 1.0  # inverter modulation 0..1

    @property
    def c_eff_wh_per_k(self) -> float:
        return 0.34 * self.volume_m3 * self.furniture_factor

    def q_disturbance(self, hour: float) -> float:
        solar = self.solar_amplitude * max(
            0.0, math.cos(2.0 * math.pi * (hour - self.solar_peak_hour) / 24.0)
        )
        return self.internal_gain + solar

    def set_thermostat(self, setpoint: float) -> None:
        """Emulate the head's internal proportional (inverter) thermostat."""
        if self.hvac_mode == "cool":
            error = self.temp - setpoint
        else:
            error = setpoint - self.temp
        self.duty = min(max(error / 1.0 + 0.2, 0.0), 1.0)
        self.hvac_on = self.duty > 0.02

    def step(self, t_out: float, hour: float, dt_h: float) -> float:
        """Advance one step; returns electrical power drawn [W]."""
        q_hvac_w = 0.0
        p_elec = 0.0
        if self.hvac_on:
            sign = 1.0 if self.hvac_mode == "heat" else -1.0
            q_hvac_w = sign * self.cop * self.draw_w * self.duty
            p_elec = self.draw_w * self.duty
        dtdt = (
            self.k * (t_out - self.temp) + self.q_disturbance(hour) + q_hvac_w / self.c_eff_wh_per_k
        )
        self.temp += dtdt * dt_h
        return p_elec


@dataclass
class SimHouse:
    rooms: list[SimRoom]
    outdoor_mean: float = 18.0
    outdoor_amplitude: float = 6.0
    base_load_w: float = 250.0
    noise_k: float = 0.0  # stddev of temperature sensor noise
    seed: int = 42
    hour: float = 0.0
    rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def t_out(self, hour: float | None = None) -> float:
        return sinusoidal_outdoor(
            self.hour if hour is None else hour, self.outdoor_mean, self.outdoor_amplitude
        )

    def base_load(self) -> float:
        # Simple morning/evening bumps on top of a constant floor.
        h = self.hour % 24.0
        bump = 150.0 if 7.0 <= h <= 9.0 or 19.0 <= h <= 22.0 else 0.0
        return self.base_load_w + bump

    def step(self, dt_h: float) -> dict:
        """Advance the whole house; returns observations for this instant."""
        t_out = self.t_out()
        p_ac = sum(room.step(t_out, self.hour, dt_h) for room in self.rooms)
        self.hour += dt_h
        p_grid = self.base_load() + p_ac
        return {
            "hour": self.hour % 24.0,
            "t_out": t_out,
            "p_grid": p_grid,
            "p_ac": p_ac,
            "temps": {
                room.name: room.temp + self.rng.gauss(0.0, self.noise_k) for room in self.rooms
            },
        }
