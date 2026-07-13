"""Zone thermal model: free-response RC fit, anchoring, COP, prediction.

Model (1R1C free response with a learned diurnal disturbance):

    dT/dt = k*(T_out - T_in) + q(t)          [K/h]
    q(t)  = a0 + a1*cos(wt) + b1*sin(wt) + a2*cos(2wt) + b2*sin(2wt)

Fitted online with RLS (forgetting 0.998) while the zone's heads are
fully off. Anchoring to absolute units via the air-exchange hypothesis:

    V_dot = k*V  [m3/h],  UA = 0.34*k*V  [W/K],  C_air = 0.34*V  [Wh/K]

The effective capacitance is C_air scaled by a learned furniture factor,
closed against disaggregated AC power (coordinate descent with COP).
"""

from __future__ import annotations

import math

from .psychro import AIR_HEAT_WH_M3K
from .rls import RLS

N_PARAMS = 6
OMEGA = 2.0 * math.pi / 24.0  # rad per hour of local time
K_MIN, K_MAX = 0.02, 3.0
FURNITURE_MIN, FURNITURE_MAX = 1.0, 12.0
COP_MIN, COP_MAX = 0.5, 8.0
# Cold-start priors: a typical bedroom leaks at ~0.2 ACH-equivalent and a
# modern inverter split delivers ~3 units of heat per unit of electricity.
K_PRIOR = 0.2
COP_PRIOR = 3.0
# Below this many 5-minute samples the fit is too fresh to trust; fall back
# to priors so a new install still predicts and controls decently.
MIN_FIT_SAMPLES = 36


def _phi(delta_out_in: float, local_hour: float) -> list[float]:
    wt = OMEGA * local_hour
    return [
        delta_out_in,
        1.0,
        math.cos(wt),
        math.sin(wt),
        math.cos(2 * wt),
        math.sin(2 * wt),
    ]


class ThermalModel:
    """Estimator for one zone, anchored on the sensed room's volume."""

    def __init__(self, volume_m3: float, lam: float = 0.998) -> None:
        self.volume_m3 = volume_m3
        # Separate regimes per door state; index by bool(door_open).
        self.fits: dict[bool, RLS] = {False: RLS(N_PARAMS, lam), True: RLS(N_PARAMS, lam)}
        self.furniture_factor = 4.0  # typical for furnished rooms; refined online
        self.cop: float | None = None
        self.cop_samples = 0

    # -- fitting -----------------------------------------------------------

    def update_free(
        self,
        t_in_prev: float,
        t_in_now: float,
        t_out: float,
        dt_h: float,
        local_hour: float,
        door_open: bool = False,
    ) -> float | None:
        """One free-response sample (heads fully off). Returns residual [K/h]."""
        if dt_h <= 0:
            return None
        y = (t_in_now - t_in_prev) / dt_h
        if abs(y) > 8.0:  # implausible slope, likely a sensor glitch
            return None
        fit = self.fits[door_open]
        residual = fit.update(_phi(t_out - t_in_now, local_hour), y)
        fit.theta[0] = min(max(fit.theta[0], K_MIN), K_MAX)
        return residual

    def _fit(self, door_open: bool) -> RLS:
        fit = self.fits[door_open]
        if fit.samples == 0 and self.fits[not door_open].samples > 0:
            return self.fits[not door_open]
        return fit

    # -- fitted / derived parameters ---------------------------------------

    def k(self, door_open: bool = False) -> float:
        fit = self._fit(door_open)
        if fit.samples == 0:
            return K_PRIOR  # neutral prior: ~5 h time constant
        k_fit = min(max(fit.theta[0], K_MIN), K_MAX)
        if fit.samples < MIN_FIT_SAMPLES:
            # Blend from prior to fit as evidence accumulates.
            w = fit.samples / MIN_FIT_SAMPLES
            return (1.0 - w) * K_PRIOR + w * k_fit
        return k_fit

    def q_hat(self, local_hour: float, door_open: bool = False) -> float:
        """Disturbance term in K/h (solar/internal gains per unit capacitance)."""
        fit = self._fit(door_open)
        if fit.samples == 0:
            return 0.0
        phi = _phi(0.0, local_hour)
        q = sum(t * x for t, x in zip(fit.theta[1:], phi[1:], strict=True))
        if fit.samples < MIN_FIT_SAMPLES:
            q *= fit.samples / MIN_FIT_SAMPLES  # shrink toward zero early on
        return q

    @property
    def cop_effective(self) -> float:
        """Learned COP, or the inverter-split prior before enough evidence."""
        if self.cop is None or self.cop_samples < 10:
            return COP_PRIOR
        return self.cop

    def t_eq(self, t_out: float, local_hour: float, door_open: bool = False) -> float:
        """Free-float attractor: the temperature the room drifts toward."""
        return t_out + self.q_hat(local_hour, door_open) / self.k(door_open)

    @property
    def ach(self) -> float:
        return self.k(False)

    @property
    def airflow_m3h(self) -> float:
        return self.k(False) * self.volume_m3

    @property
    def ua_w_per_k(self) -> float:
        return AIR_HEAT_WH_M3K * self.k(False) * self.volume_m3

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

    # -- power closure (coordinate descent) ---------------------------------

    def sensible_power_w(
        self,
        t_in: float,
        t_out: float,
        dtdt_per_h: float,
        local_hour: float,
        door_open: bool = False,
    ) -> float:
        """Heat added by the AC (signed W; negative while cooling).

        C*dT/dt = UA*(T_out - T_in) + q*C + Q_hvac
        """
        c = self.c_eff_wh_per_k
        return (
            c * dtdt_per_h
            - self.ua_w_per_k * (t_out - t_in)
            - self.q_hat(local_hour, door_open) * c
        )

    def update_cop(
        self,
        p_ac_w: float,
        t_in: float,
        t_out: float,
        local_hour: float,
        door_open: bool = False,
        alpha: float = 0.05,
    ) -> None:
        """Quasi-steady COP update (call when |dT/dt| is small while conditioning)."""
        if p_ac_w < 50.0:
            return
        q_hvac = abs(self.sensible_power_w(t_in, t_out, 0.0, local_hour, door_open))
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
        alpha: float = 0.05,
    ) -> None:
        """Transient capacitance update with COP held fixed.

        dT/dt = (Q_hvac + UA*(T_out-T_in))/C + q  =>  C = num / (dT/dt - q)
        """
        if p_ac_w < 50.0:
            return
        q_hvac = self.cop_effective * p_ac_w * (1.0 if heating else -1.0)
        denom = dtdt_per_h - self.q_hat(local_hour, door_open)
        if abs(denom) < 0.1:
            return
        c_est = (q_hvac + self.ua_w_per_k * (t_out - t_in)) / denom
        factor = c_est / self.c_air_wh_per_k
        if not (FURNITURE_MIN <= factor <= FURNITURE_MAX):
            return
        self.furniture_factor += alpha * (factor - self.furniture_factor)

    @staticmethod
    def carnot_cop(t_in: float, t_out: float, cooling: bool) -> float | None:
        """Ideal COP bound for sanity checking (second-law efficiency)."""
        dt = (t_out - t_in) if cooling else (t_in - t_out)
        if dt <= 0.5:
            return None
        cold = min(t_in, t_out) + 273.15
        hot = max(t_in, t_out) + 273.15
        return cold / (hot - cold) if cooling else hot / (hot - cold)

    # -- prediction ----------------------------------------------------------

    def predict_free(
        self,
        t_in: float,
        t_out_hourly: list[float],
        start_hour: float,
        hours: float = 24.0,
        step_h: float = 1.0 / 6.0,
        door_open: bool = False,
    ) -> list[float]:
        """Euler-integrated free-float trajectory; returns hourly samples.

        t_out_hourly[i] is the forecast outdoor temperature i hours from now
        (index 0 = now). Values are linearly interpolated between hours.
        """
        if not t_out_hourly:
            return []
        k = self.k(door_open)
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
            hour = (start_hour + elapsed) % 24.0
            t += step_h * (k * (t_out - t) + self.q_hat(hour, door_open))
            elapsed += step_h
        return temps

    # -- persistence ----------------------------------------------------------

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
            model.fits[False] = RLS.from_dict(data["fit_closed"], N_PARAMS)
        if "fit_open" in data:
            model.fits[True] = RLS.from_dict(data["fit_open"], N_PARAMS)
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
