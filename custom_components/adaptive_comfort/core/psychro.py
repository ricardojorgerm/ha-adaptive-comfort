"""Psychrometric helpers (Magnus formula, humidity ratio, latent heat)."""

from __future__ import annotations

import math

P_ATM_PA = 101_325.0
AIR_DENSITY_KG_M3 = 1.2
# Volumetric sensible heat capacity of air: rho * c_p ~= 0.34 Wh/(m3*K)
AIR_HEAT_WH_M3K = 0.34
# Latent heat of vaporisation: ~2450 kJ/kg ~= 680 Wh/kg
H_FG_WH_KG = 680.0


def saturation_pressure_pa(temp_c: float) -> float:
    """Saturation vapour pressure over water (Magnus), in Pa."""
    return 610.94 * math.exp(17.625 * temp_c / (243.04 + temp_c))


def vapour_pressure_pa(temp_c: float, rh_pct: float) -> float:
    return (rh_pct / 100.0) * saturation_pressure_pa(temp_c)


def humidity_ratio(temp_c: float, rh_pct: float, pressure_pa: float = P_ATM_PA) -> float:
    """Humidity ratio w in kg water / kg dry air."""
    p_v = vapour_pressure_pa(temp_c, rh_pct)
    p_v = min(p_v, pressure_pa * 0.99)
    return 0.622 * p_v / (pressure_pa - p_v)


def moisture_removal_kg_h(
    volume_m3: float,
    airflow_m3h: float,
    w_in: float,
    w_out: float,
    dw_in_dt_per_h: float,
    sources_kg_h: float = 0.0,
) -> float:
    """Moisture removed by the AC coil, from the zone moisture balance.

    rho*V*dw/dt = rho*V_dot*(w_out - w_in) + m_sources - m_removed
    """
    infiltration = AIR_DENSITY_KG_M3 * airflow_m3h * (w_out - w_in)
    storage = AIR_DENSITY_KG_M3 * volume_m3 * dw_in_dt_per_h
    removed = infiltration + sources_kg_h - storage
    return max(0.0, removed)


def latent_power_w(moisture_removed_kg_h: float) -> float:
    """Latent power in W for a given condensate removal rate."""
    return moisture_removed_kg_h * H_FG_WH_KG
