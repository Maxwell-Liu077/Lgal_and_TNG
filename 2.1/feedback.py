"""AGN radio-mode strength diagnostics for Phase 2.1."""

from __future__ import annotations

import numpy as np


K_AGN_MSUN_PER_YR = 5.3e-3
RADIO_EFFICIENCY = 0.1
LIGHT_SPEED_CM_S = 2.99792458e10
MSUN_G = 1.98847e33
SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0


def compute_agn_strength(
    *,
    m_hot_msun: float,
    m_bh_msun: float,
    r200c_pkpc: float,
    v200c_km_s: float,
) -> dict:
    """Compute H15 heating rate and A_SAM from averaged physical inputs."""

    try:
        m_hot_msun = float(m_hot_msun)
        m_bh_msun = float(m_bh_msun)
        r200c_pkpc = float(r200c_pkpc)
        v200c_km_s = float(v200c_km_s)
    except (TypeError, ValueError):
        return {"agn_strength": np.nan, "A_SAM": np.nan, "mdot_bh_msun_per_yr": np.nan, "mdot_heat_h15_msun_per_yr": np.nan, "t_dyn_gyr": np.nan}
    values = (m_hot_msun, m_bh_msun, r200c_pkpc, v200c_km_s)
    if any(not np.isfinite(value) for value in values) or m_hot_msun <= 0 or m_bh_msun < 0 or r200c_pkpc <= 0 or v200c_km_s <= 0:
        return {"agn_strength": np.nan, "A_SAM": np.nan, "mdot_bh_msun_per_yr": np.nan, "mdot_heat_h15_msun_per_yr": np.nan, "t_dyn_gyr": np.nan}
    mdot_bh = K_AGN_MSUN_PER_YR * (m_hot_msun / 1.0e11) * (m_bh_msun / 1.0e8)
    power = RADIO_EFFICIENCY * mdot_bh * MSUN_G / SECONDS_PER_YEAR * LIGHT_SPEED_CM_S**2
    mdot_heat = 2.0 * power / (v200c_km_s * 1.0e5) ** 2 / MSUN_G * SECONDS_PER_YEAR
    t_dyn_gyr = r200c_pkpc * 3.085677581491367e21 / (v200c_km_s * 1.0e5) / (1.0e9 * SECONDS_PER_YEAR)
    denominator = m_hot_msun / (t_dyn_gyr * 1.0e9)
    # A zero BH mass is the documented physical -infinity limit.  A zero hot
    # reservoir, by contrast, makes both numerator and denominator vanish and
    # is undefined; it is rejected above as NaN.
    strength = -np.inf if m_bh_msun == 0 else float(np.log10(mdot_heat / denominator))
    return {
        "agn_strength": strength,
        "A_SAM": strength,
        "mdot_bh_msun_per_yr": float(mdot_bh),
        "mdot_heat_h15_msun_per_yr": float(mdot_heat),
        "t_dyn_gyr": float(t_dyn_gyr),
    }
