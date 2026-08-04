"""Zone thermal model: free-response RC fit, anchoring, COP, prediction.

Model (1R1C with outdoor + house mixing and a learned diurnal disturbance):

    dT/dt = k_out*(T_out - T_in) + k_mix*(T_house - T_in) + q(t)   [K/h]
    q(t)  = a0 + a1*cos(wt) + b1*sin(wt) + a2*cos(2wt) + b2*sin(2wt)

T_house is the volume-weighted mean temperature of *other* conditioned zones
and optional unconditioned-room sensors (leave-one-out for the zone being fit).

Fitted online with RLS (forgetting 0.998) while the zone's heads are fully off.
`predict_free` integrates that rate model directly.

Power uses the same rates on effective capacitance C_eff = furniture · 0.34 · V:

    UA = C_eff * k   [W/K]
    Q_ac = C_eff * (dT/dt - k_out*ΔT_out - k_mix*ΔT_house - q)   [W]

(Do not use air-only UA = 0.34·k·V for the energy balance — that understates
coupling by furniture_factor and disagrees with the rate model prediction uses.)

Outdoor *volume* flow for moisture stays the air-exchange hypothesis:

    V_dot_out = k_out * V  [m³/h]   (house-mixing must not couple to outdoor w)

Total ACH ≈ k_out + k_mix. C_eff is closed against disaggregated AC power.
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

    def q_coeffs(self, door_open: bool) -> dict[str, float | int] | None:
        """Raw learned harmonic coefficients for one door regime (no fallback).

        q(t) = a0 + a1·cos(ωt) + b1·sin(ωt) + a2·cos(2ωt) + b2·sin(2ωt),
        with ω = 2π/24 and t in local hours. Returns None when that regime
        has never been observed.
        """
        fit = self.fits[door_open]
        if fit.samples == 0 or len(fit.theta) < N_PARAMS:
            return None
        th = fit.theta
        return {
            "a0": round(float(th[2]), 5),
            "a1": round(float(th[3]), 5),
            "b1": round(float(th[4]), 5),
            "a2": round(float(th[5]), 5),
            "b2": round(float(th[6]), 5),
            "samples": int(fit.samples),
        }

    def disturbance_diag(self, local_hour: float, door_open: bool = False) -> dict:
        """Diagnostics for the learned diurnal disturbance q(t)."""
        q = self.q_hat(local_hour, door_open)
        return {
            "q_hat_k_per_h": round(q, 5),
            "q_hat_w": round(q * self.c_eff_wh_per_k, 1),
            "local_hour": round(local_hour % 24.0, 2),
            "active_regime": "door_open" if door_open else "door_closed",
            "fallback": self.fit_is_fallback(door_open),
            "coeffs_closed": self.q_coeffs(False),
            "coeffs_open": self.q_coeffs(True),
        }

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

    def outdoor_airflow_m3h(
        self,
        door_open: bool = False,
        *,
        outdoor_exhaust_on: bool = False,
    ) -> float:
        """Outdoor-exchange volume flow (m³/h) — for moisture coupling to outdoor w.

        House-mixing (`k_mix`) exchanges indoor air with other rooms, not outdoor
        humidity; using total ACH here inflated latent removal and moisture
        baselines.
        """
        k_out, _k_mix = self._scaled_k(
            door_open,
            indoor_fans_on=False,
            outdoor_exhaust_on=outdoor_exhaust_on,
        )
        return k_out * self.volume_m3

    @property
    def c_air_wh_per_k(self) -> float:
        return AIR_HEAT_WH_M3K * self.volume_m3

    @property
    def c_eff_wh_per_k(self) -> float:
        return self.c_air_wh_per_k * self.furniture_factor

    @property
    def ua_w_per_k(self) -> float:
        """Effective outdoor UA (W/K): C_eff · k_out, fit-consistent."""
        return self.c_eff_wh_per_k * self.k(False)

    @property
    def ua_mix_w_per_k(self) -> float:
        """Effective mixing UA (W/K): C_eff · k_mix, fit-consistent."""
        return self.c_eff_wh_per_k * self.k_mix(False)

    def ua_out_w_per_k(
        self,
        door_open: bool = False,
        *,
        indoor_fans_on: bool = False,
        outdoor_exhaust_on: bool = False,
    ) -> float:
        k_out, _ = self._scaled_k(
            door_open,
            indoor_fans_on=indoor_fans_on,
            outdoor_exhaust_on=outdoor_exhaust_on,
        )
        return self.c_eff_wh_per_k * k_out

    def ua_mix_scaled_w_per_k(
        self,
        door_open: bool = False,
        *,
        indoor_fans_on: bool = False,
        outdoor_exhaust_on: bool = False,
    ) -> float:
        _k_out, k_mix = self._scaled_k(
            door_open,
            indoor_fans_on=indoor_fans_on,
            outdoor_exhaust_on=outdoor_exhaust_on,
        )
        return self.c_eff_wh_per_k * k_mix

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

    def migrate_default_closed_to_open(self) -> bool:
        """Move closed-regime learning into open when open is still empty.

        Used for zones with no door sensor after the default flipped from
        closed → open (assume inter-room mixing). Returns True if migrated.
        """
        if self.fits[True].samples > 0 or self.fits[False].samples == 0:
            return False
        self.fits[True] = self.fits[False]
        self.fits[False] = RLS(N_PARAMS, self.fits[True].lam)
        return True

    def fit_stage(self, door_open: bool = False) -> str:
        """How much the k values rely on learned data: prior | blending | fitted."""
        samples = self.fit_samples(door_open)
        if samples == 0:
            return "prior"
        if samples < MIN_FIT_SAMPLES:
            return "blending"
        return "fitted"

    def free_float_rate(
        self,
        t_in: float,
        t_out: float,
        local_hour: float,
        door_open: bool = False,
        t_house_other: float | None = None,
        *,
        indoor_fans_on: bool = False,
        outdoor_exhaust_on: bool = False,
    ) -> float:
        """No-AC dT/dt [K/h] — same rate model `predict_free` integrates."""
        k_out, k_mix = self._scaled_k(
            door_open,
            indoor_fans_on=indoor_fans_on,
            outdoor_exhaust_on=outdoor_exhaust_on,
        )
        mix = k_mix * (t_house_other - t_in) if t_house_other is not None else 0.0
        return k_out * (t_out - t_in) + mix + self.q_hat(local_hour, door_open)

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
        """Heat added by the AC (signed W; negative while cooling).

        C_eff times the excess rate over free-float — identical physics to
        `predict_free` / `free_float_rate`, so standing load and COP use the
        same model the controller's predictions use.
        """
        excess = dtdt_per_h - self.free_float_rate(
            t_in,
            t_out,
            local_hour,
            door_open,
            t_house_other,
            indoor_fans_on=indoor_fans_on,
            outdoor_exhaust_on=outdoor_exhaust_on,
        )
        return self.c_eff_wh_per_k * excess

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
        latent_w: float = 0.0,
    ) -> None:
        if p_ac_w < 50.0:
            return
        # Delivered heat is sensible + latent: dehumidification is real
        # compressor work (omitting it undercounted COP by the latent share,
        # ~30-50% in humid rooms).
        q_hvac = abs(
            self.sensible_power_w(t_in, t_out, dtdt_per_h, local_hour, door_open, t_house_other)
        ) + max(0.0, latent_w)
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
        *,
        indoor_fans_on: bool = False,
        outdoor_exhaust_on: bool = False,
    ) -> None:
        if p_ac_w < 50.0:
            return
        q_hvac = self.cop_effective * p_ac_w * (1.0 if heating else -1.0)
        # Q_ac = C_eff * (dT/dt - free_float_rate) → solve for C without
        # using UA(=C·k), which would be circular in furniture_factor.
        excess = dtdt_per_h - self.free_float_rate(
            t_in,
            t_out,
            local_hour,
            door_open,
            t_house_other,
            indoor_fans_on=indoor_fans_on,
            outdoor_exhaust_on=outdoor_exhaust_on,
        )
        if abs(excess) < 0.1:
            return
        c_est = q_hvac / excess
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

    def predict_horizons(
        self,
        t_in: float,
        t_out_hourly: list[float],
        start_hour: float,
        horizons_min: tuple[int, ...],
        door_open: bool = False,
        t_house_other: float | None = None,
        t_house_hourly: list[float] | None = None,
        *,
        indoor_fans_on: bool = False,
        outdoor_exhaust_on: bool = False,
    ) -> dict[int, float]:
        """Predicted temperature at each of `horizons_min` minutes ahead.

        Same free-float physics as `predict_free` (identical k_out/k_mix/q(t)
        integration), but stepped finely and landing exactly on each requested
        horizon -- `predict_free`'s hourly sampling cadence is too coarse to
        score against 15/30-minute-ahead actuals.
        """
        if not t_out_hourly or not horizons_min:
            return {}
        k_out, k_mix = self._scaled_k(
            door_open,
            indoor_fans_on=indoor_fans_on,
            outdoor_exhaust_on=outdoor_exhaust_on,
        )
        if t_house_other is None and not t_house_hourly:
            k_mix = 0.0
        targets = sorted(set(h for h in horizons_min if h > 0))
        if not targets:
            return {}
        out: dict[int, float] = {}
        t = t_in
        elapsed = 0.0
        max_step_h = 1.0 / 60.0  # 1-minute integration step
        target_i = 0
        while target_i < len(targets):
            target_h = targets[target_i] / 60.0
            while elapsed < target_h - 1e-9:
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
                step_h = min(max_step_h, target_h - elapsed)
                t += step_h * (k_out * (t_out - t) + mix_term + self.q_hat(hour, door_open))
                elapsed += step_h
            out[targets[target_i]] = t
            target_i += 1
        return out

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
