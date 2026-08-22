"""Henriques-style isothermal cooling calculations."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


G_CGS = 6.67430e-8
MSUN_G = 1.98847e33
KPC_CM = 3.085677581491367e21
PROTON_MASS_G = 1.67262192369e-24
BOLTZMANN_CGS = 1.380649e-16
SECONDS_PER_GYR = 1.0e9 * 365.25 * 24.0 * 3600.0


def compute_isothermal_sam_cooling(
    *,
    m_hot_msun: float,
    z_hot_mass_fraction: float,
    m200c_msun: float,
    r200c_pkpc: float,
    cooling_function: Callable,
    mean_molecular_weight: float = 0.59,
    virial_temperature_factor: float = 35.9,
) -> dict:
    """Evaluate the Henriques S1.4 isothermal cooling model."""

    if not all(np.isfinite(value) for value in (m_hot_msun, m200c_msun, r200c_pkpc)) or m_hot_msun < 0 or m200c_msun <= 0 or r200c_pkpc <= 0:
        raise ValueError("Invalid hot mass, halo mass, or halo radius")
    v200_cms = np.sqrt(G_CGS * m200c_msun * MSUN_G / (r200c_pkpc * KPC_CM))
    v200_kms = float(v200_cms / 1.0e5)
    t_dyn_s = r200c_pkpc * KPC_CM / v200_cms
    t_dyn_gyr = float(t_dyn_s / SECONDS_PER_GYR)
    t_vir = float(virial_temperature_factor * v200_kms**2)
    metallicity = float(z_hot_mass_fraction) if np.isfinite(z_hot_mass_fraction) and z_hot_mass_fraction > 0 else 0.0
    lam = float(cooling_function(t_vir, metallicity))
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("Cooling function returned a non-positive value")
    if m_hot_msun == 0:
        r_cool = 0.0
        rate = 0.0
        mode = "empty"
    else:
        numerator = t_dyn_s * m_hot_msun * MSUN_G * lam
        denominator = 6.0 * np.pi * mean_molecular_weight * PROTON_MASS_G * BOLTZMANN_CGS * t_vir * r200c_pkpc * KPC_CM
        r_cool = float(np.sqrt(numerator / denominator) / KPC_CM)
        fraction = min(r_cool / r200c_pkpc, 1.0)
        rate = float(m_hot_msun / (t_dyn_gyr * 1.0e9) * fraction)
        mode = "cooling_flow" if r_cool < r200c_pkpc else "rapid_infall"
    return {
        "v200c_km_s": v200_kms,
        "t_dyn_gyr": t_dyn_gyr,
        "t_vir_k": t_vir,
        "lambda_erg_cm3_s": lam,
        "r_cool_isothermal_pkpc": r_cool,
        "r_cool_isothermal_over_r200c": float(r_cool / r200c_pkpc),
        "sam_cooling_mode": mode,
        "rate_sam_isothermal_msun_per_yr": rate,
    }
