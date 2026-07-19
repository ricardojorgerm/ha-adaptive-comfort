"""Zone thermal model: free-response RC fit, anchoring, COP, prediction.

Model (1R1C with outdoor + house mixing and a learned diurnal disturbance):

    dT/dt = k_out*(T_out - T_in) + k_mix*(T_house - T_in) + q(t)   [K/h]
    q(t)  = a0 + a1*cos(wt) + b1*sin(wt) + a2*cos(2wt) + b2*sin(2wt)

T_house is the volume-weighted mean temperature of *other* conditioned zones
and optional unconditioned-room sensors (leave-one-out for the zone being fit).

Fitted online with RLS (forgetting 0.998) while the zone's heads are fully off.
Outdoor anchoring via the air-exchange hypothesis:

    V_dot_out = k_out*V  [m3/h],  UA_out = 0.34*k_out*V  [W/K]

Mixing uses the same volumetric heat capacity with k_mix. Total ACH ≈ k_out + k_mix.

The effective capacitance is C_air scaled by a learned furniture factor,
closed against disaggregated AC power (coordinate descent with COP).
"""

from __future__ import annotations

import math

from .psychro import AIR_HEAT_WH_M3K
from .rls import RLS

N_PARAMS = 7
N_PARAMS_LEGACY = 6
OMEGA = 2.0 * math.pi / 24.0  # rad per hour of local time
K_MIN, K_MAX = 0.02, 3.0
K_MIX_MIN, K_MIX_MAX = 0.0, 2.0
FURNITURE_MIN, FURNITURE_MAX = 1.0, 12.0
COP_MIN, COP_MAX = 0.5, 8.0
K_PRIOR = 0.2
K_MIX_PRIOR = 0.15
COP_PRIOR = 3.0
MIN_FIT_SAMPLES = 36
# Effective exchange boost while configured vent fans are running (identification + prediction).
OUTDOOR_FAN_K_OUT_MULT = 2.5
INDOOR_FAN_K_MIX_MULT = 2.0


def house_other_temperature(
    zone_id: str,
    zone_readings: dict[str, tuple[float, float]],
    aux_readings: list[tuple[float, float]] | None = None,
) -> float | None:
    """Volume-weighted indoor mean excluding zone_id.

    zone_readings: zone_id -> (temp C, volume m3)
    aux_readings: optional (temp C, volume m3) from unconditioned rooms
    """
    total = 0.0
    weight = 0.0
    for zid, (temp, vol) in zone_readings.items():
        if zid == zone_id or vol <= 0:
            continue
        total += vol * temp
        weight += vol
    for temp, vol in aux_readings or []:
        if vol <= 0:
            continue
        total += vol * temp
        weight += vol
    if weight <= 0:
        return None
    return total / weight


def _phi(delta_out_in: float, delta_house_in: float, local_hour: float) -> list[float]:
    wt = OMEGA * local_hour
    return [
        delta_out_in,
        delta_house_in,
        1.0,
        math.cos(wt),
        math.sin(wt),
        math.cos(2 * wt),
        math.sin(2 * wt),
    ]


def _expand_legacy_rls(data: dict) -> dict:
    """Upgrade a 6-parameter fit (outdoor-only) to 7-parameter (outdoor + mix)."""
    theta = data.get("theta")
    if not theta or len(theta) != N_PARAMS_LEGACY:
        return data
    p = data.get("p")
    new_theta = [theta[0], 0.0, *theta[1:]]
    new_p = None
    if p and len(p) == N_PARAMS_LEGACY:
        n = N_PARAMS
        new_p = [[100.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        for i in range(N_PARAMS_LEGACY):
            for j in range(N_PARAMS_LEGACY):
                oi = i + 1 if i >= 1 else i
                oj = j + 1 if j >= 1 else j
                new_p[oi][oj] = p[i][j]
    out = dict(data)
    out["theta"] = new_theta
    if new_p is not None:
        out["p"] = new_p
    return out


class ThermalModel:
    """Estimator for one zone, anchored on the sensed room's volume."""

    def __init__(self, volume_m3: float, lam: float = 0.998) -> None:
        self.volume_m3 = volume_m3
        self.fits: dict[bool, RLS] = {False: RLS(N_PARAMS, lam), True: RLS(N_PARAMS, lam)}
        self.furniture_factor = 4.0
        self.cop: float | None = None
        self.cop_samples = 0

    def _scaled_k(
        self,
        door_open: bool,
        *,
        indoor_fans_on: bool = False,
        outdoor_exhaust_on: bool = False,
    ) -> tuple[float, float]:
        out_mult = OUTDOOR_FAN_K_OUT_MULT if outdoor_exhaust_on else 1.0
        mix_mult = INDOOR_FAN_K_MIX_MULT if indoor_fans_on else 1.0
        return self.k(door_open) * out_mult, self.k_mix(door_open) * mix_mult

    def update_free(
        self,
        t_in_prev: float,
        t_in_now: float,
        t_out: float,
        dt_h: float,
        local_hour: float,
        door_open: bool = False,
        t_house_other: float | None = None,
        *,
        indoor_fans_on: bool = False,
        outdoor_exhaust_on: bool = False,
    ) -> float | None:
        """One free-response sample (heads fully off). Returns residual [K/h]."""
        if dt_h <= 0:
            return None
        y = (t_in_now - t_in_prev) / dt_h
        if abs(y) > 8.0:
            return None
        delta_out = t_out - t_in_now
        if outdoor_exhaust_on:
            delta_out *= OUTDOOR_FAN_K_OUT_MULT
        delta_house = 0.0 if t_house_other is None else t_house_other - t_in_now
        if indoor_fans_on:
            delta_house *= INDOOR_FAN_K_MIX_MULT
        fit = self.fits[door_open]
        residual = fit.update(_phi(delta_out, delta_house, local_hour), y)
        fit.theta[0] = min(max(fit.theta[0], K_MIN), K_MAX)
        fit.theta[1] = min(max(fit.theta[1], K_MIX_MIN), K_MIX_MAX)
        return residual

    def _fit(self, door_open: bool) -> RLS:
        fit = self.fits[door_open]
        if fit.samples == 0 and self.fits[not door_open].samples > 0:
            return self.fits[not door_open]
        return fit

    def _blend(self, fit: RLS, prior: float, k_fit: float) -> float:
        if fit.samples < MIN_FIT_SAMPLES:
            w = fit.samples / MIN_FIT_SAMPLES
            return (1.0 - w) * prior + w * k_fit
        return k_fit

    def k(self, door_open: bool = False) -> float:
        """Outdoor exchange constant k_out (h⁻¹)."""
        fit = self._fit(door_open)
        if fit.samples == 0:
            return K_PRIOR
        k_fit = min(max(fit.theta[0], K_MIN), K_MAX)
        return self._blend(fit, K_PRIOR, k_fit)

    def k_mix(self, door_open: bool = False) -> float:
        """Inter-zone / house mixing constant k_mix (h⁻¹)."""
        fit = self._fit(door_open)
        if fit.samples == 0:
            return K_MIX_PRIOR
        k_fit = min(max(fit.theta[1], K_MIX_MIN), K_MIX_MAX)
        return self._blend(fit, K_MIX_PRIOR, k_fit)

    def q_hat(self, local_hour: float, door_open: bool = False) -> float:
        """Disturbance term in K/h (solar/internal gains per unit capacitance)."""
        fit = self._fit(door_open)
        if fit.samples == 0:
            return 0.0
        phi = _phi(0.0, 0.0, local_hour)
        q = sum(t * x for t, x in zip(fit.theta[2:], phi[2:], strict=True))
        if fit.samples < MIN_FIT_SAMPLES:
            q *= fit.samples / MIN_FIT_SAMPLES
        return q

    @property
    def cop_effective(self) -> float:
        if self.cop is None or self.cop_samples < 10:
            return COP_PRIOR
        return self.cop

    def t_eq(
        self,
        t_out: float,
        local_hour: float,
        door_open: bool = False,
        t_house_other: float | None = None,
        *,
        indoor_fans_on: bool = False,
        outdoor_exhaust_on: bool = False,
    ) -> float:
        """Free-float attractor at fixed outdoor and house-other temperatures."""
        k_out, k_mix = self._scaled_k(
            door_open,
            indoor_fans_on=indoor_fans_on,
            outdoor_exhaust_on=outdoor_exhaust_on,
        )
        if t_house_other is None:
            k_mix = 0.0
        q = self.q_hat(local_hour, door_open)
        denom = k_out + k_mix
        if denom <= 1e-9:
            return t_out
        num = k_out * t_out + q
        if t_house_other is not None:
            num += k_mix * t_house_other
        return num / denom

    @property
    def ach(self) -> float:
        """Total equivalent air changes (outdoor + house mixing)."""
        return self.k(False) + self.k_mix(False)

    @property
    def airflow_m3h(self) -> float:
        return self.ach * self.volume_m3

    @property
    def ua_w_per_k(self) -> float:
        return AIR_HEAT_WH_M3K * self.k(False) * self.volume_m3

    @property
    def ua_mix_w_per_k(self) -> float:
        return AIR_HEAT_WH_M3K * self.k_mix(False) * self.volume_m3

    @property
    def c_air_wh_per_k(self) -> float:
        return AIR_HEAT_WH_M3K * self.volume_m3

    @property
    def c_eff_wh_per_k(self) -> float:
        return self.c_air_wh_per_k * self.furniture_factor

    def confidence(self, door_open: bool = False) -> float:
        fit = self._fit(door_open)
        if fit.samples == 0:
            return 0.0
        sample_part = min(1.0, fit.samples / 300.0)
        trace_part = 1.0 / (1.0 + fit.trace / N_PARAMS)
        return sample_part * trace_part

    def fit_samples(self, door_open: bool = False) -> int:
        return self._fit(door_open).samples

    def fit_is_fallback(self, door_open: bool = False) -> bool:
        """True when this regime has no samples of its own and reports the
        other regime's fit (diagnostics would otherwise show two identical
        regimes and imply both were learned independently)."""
        return self.fits[door_open].samples == 0 and self.fits[not door_open].samples > 0

    def fit_stage(self, door_open: bool = False) -> str:
        """How much the k values rely on learned data: prior | blending | fitted."""
        samples = self.fit_samples(door_open)
        if samples == 0:
            return "prior"
        if samples < MIN_FIT_SAMPLES:
            return "blending"
        return "fitted"

    def sensible_power_w(
        self,
        t_in: float,
        t_out: float,
        dtdt_per_h: float,
        local_hour: float,
        door_open: bool = False,
        t_house_other: float | None = None,
        *,
        indoor_fans_on: bool = False,
        outdoor_exhaust_on: bool = False,
    ) -> float:
        """Heat added by the AC (signed W; negative while cooling)."""
        c = self.c_eff_wh_per_k
        k_out, k_mix = self._scaled_k(
            door_open,
            indoor_fans_on=indoor_fans_on,
            outdoor_exhaust_on=outdoor_exhaust_on,
        )
        mix_w = 0.0
        if t_house_other is not None:
            mix_w = AIR_HEAT_WH_M3K * k_mix * self.volume_m3 * (t_house_other - t_in)
        ua_out = AIR_HEAT_WH_M3K * k_out * self.volume_m3
        return (
            c * dtdt_per_h - ua_out * (t_out - t_in) - mix_w - self.q_hat(local_hour, door_open) * c
        )

    def update_cop(
        self,
        p_ac_w: float,
        t_in: float,
        t_out: float,
        local_hour: float,
        door_open: bool = False,
        t_house_other: float | None = None,
        alpha: float = 0.05,
        dtdt_per_h: float = 0.0,
    ) -> None:
        if p_ac_w < 50.0:
            return
        q_hvac = abs(
            self.sensible_power_w(t_in, t_out, dtdt_per_h, local_hour, door_open, t_house_other)
        )
        cop = q_hvac / p_ac_w
        if not (COP_MIN <= cop <= COP_MAX):
            return
        if self.cop is None:
            self.cop = cop
        else:
            self.cop += alpha * (cop - self.cop)
        self.cop_samples += 1

    def update_c_eff(
        self,
        p_ac_w: float,
        heating: bool,
        t_in: float,
        t_out: float,
        dtdt_per_h: float,
        local_hour: float,
        door_open: bool = False,
        t_house_other: float | None = None,
        alpha: float = 0.05,
    ) -> None:
        if p_ac_w < 50.0:
            return
        q_hvac = self.cop_effective * p_ac_w * (1.0 if heating else -1.0)
        denom = dtdt_per_h - self.q_hat(local_hour, door_open)
        if abs(denom) < 0.1:
            return
        mix_w = 0.0
        if t_house_other is not None:
            mix_w = self.ua_mix_w_per_k * (t_house_other - t_in)
        c_est = (q_hvac + self.ua_w_per_k * (t_out - t_in) + mix_w) / denom
        factor = c_est / self.c_air_wh_per_k
        if not (FURNITURE_MIN <= factor <= FURNITURE_MAX):
            return
        self.furniture_factor += alpha * (factor - self.furniture_factor)

    @staticmethod
    def carnot_cop(t_in: float, t_out: float, cooling: bool) -> float | None:
        dt = (t_out - t_in) if cooling else (t_in - t_out)
        if dt <= 0.5:
            return None
        cold = min(t_in, t_out) + 273.15
        hot = max(t_in, t_out) + 273.15
        return cold / (hot - cold) if cooling else hot / (hot - cold)

    def predict_free(
        self,
        t_in: float,
        t_out_hourly: list[float],
        start_hour: float,
        hours: float = 24.0,
        step_h: float = 1.0 / 6.0,
        door_open: bool = False,
        t_house_other: float | None = None,
        t_house_hourly: list[float] | None = None,
        *,
        indoor_fans_on: bool = False,
        outdoor_exhaust_on: bool = False,
    ) -> list[float]:
        """Euler-integrated free-float trajectory; returns hourly samples."""
        if not t_out_hourly:
            return []
        k_out, k_mix = self._scaled_k(
            door_open,
            indoor_fans_on=indoor_fans_on,
            outdoor_exhaust_on=outdoor_exhaust_on,
        )
        if t_house_other is None and not t_house_hourly:
            k_mix = 0.0
        temps: list[float] = []
        t = t_in
        elapsed = 0.0
        next_sample = 0.0
        while elapsed <= hours + 1e-9:
            if elapsed >= next_sample - 1e-9:
                temps.append(t)
                next_sample += 1.0
            idx = min(int(elapsed), len(t_out_hourly) - 1)
            frac = min(elapsed - idx, 1.0)
            nxt = min(idx + 1, len(t_out_hourly) - 1)
            t_out = t_out_hourly[idx] * (1 - frac) + t_out_hourly[nxt] * frac
            if t_house_hourly:
                hi = min(int(elapsed), len(t_house_hourly) - 1)
                hf = min(elapsed - hi, 1.0)
                hn = min(hi + 1, len(t_house_hourly) - 1)
                t_house = t_house_hourly[hi] * (1 - hf) + t_house_hourly[hn] * hf
            else:
                t_house = t_house_other
            hour = (start_hour + elapsed) % 24.0
            mix_term = k_mix * (t_house - t) if t_house is not None else 0.0
            t += step_h * (k_out * (t_out - t) + mix_term + self.q_hat(hour, door_open))
            elapsed += step_h
        return temps

    def to_dict(self) -> dict:
        return {
            "volume_m3": self.volume_m3,
            "fit_closed": self.fits[False].to_dict(),
            "fit_open": self.fits[True].to_dict(),
            "furniture_factor": self.furniture_factor,
            "cop": self.cop,
            "cop_samples": self.cop_samples,
        }

    @classmethod
    def from_dict(cls, data: dict, volume_m3: float) -> ThermalModel:
        model = cls(volume_m3)
        if "fit_closed" in data:
            model.fits[False] = RLS.from_dict(_expand_legacy_rls(data["fit_closed"]), N_PARAMS)
        if "fit_open" in data:
            model.fits[True] = RLS.from_dict(_expand_legacy_rls(data["fit_open"]), N_PARAMS)
        model.furniture_factor = float(data.get("furniture_factor", 4.0))
        cop = data.get("cop")
        model.cop = float(cop) if cop is not None else None
        model.cop_samples = int(data.get("cop_samples", 0))
        return model


class DiurnalModel:
    """Harmonic fit of a signal vs local hour (used to forecast outdoor temp
    when no weather entity is configured)."""

    def __init__(self, lam: float = 0.999) -> None:
        self.fit = RLS(5, lam)

    @staticmethod
    def _phi(local_hour: float) -> list[float]:
        wt = OMEGA * local_hour
        return [1.0, math.cos(wt), math.sin(wt), math.cos(2 * wt), math.sin(2 * wt)]

    def update(self, local_hour: float, value: float) -> None:
        self.fit.update(self._phi(local_hour), value)

    def predict(self, local_hour: float) -> float | None:
        if self.fit.samples < 12:
            return None
        return self.fit.predict(self._phi(local_hour))

    def forecast_hours(self, start_hour: float, hours: int) -> list[float] | None:
        if self.fit.samples < 12:
            return None
        return [self.fit.predict(self._phi((start_hour + h) % 24.0)) for h in range(hours)]

    def to_dict(self) -> dict:
        return self.fit.to_dict()

    @classmethod
    def from_dict(cls, data: dict) -> DiurnalModel:
        model = cls()
        model.fit = RLS.from_dict(data, 5)
        return model
