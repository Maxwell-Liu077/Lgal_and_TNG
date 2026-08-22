"""Henriques-style cooling and supernova-reheating calculations.

All functions are side-effect free and operate on explicit physical inputs.
They therefore remain reusable independently of TNG catalog access.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


G_CGS = 6.67430e-8
MSUN_G = 1.98847e33
KPC_CM = 3.0856775814913673e21
PROTON_MASS_G = 1.67262192369e-24
BOLTZMANN_CGS = 1.380649e-16
SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0
SECONDS_PER_GYR = 1.0e9 * SECONDS_PER_YEAR


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
    """Evaluate Henriques S1.4 isothermal cooling at one endpoint."""

    if m_hot_msun < 0:
        raise ValueError("m_hot_msun cannot be negative")
    if m200c_msun <= 0 or r200c_pkpc <= 0:
        raise ValueError("M200c and R200c must be positive")
    v200c_cm_s = np.sqrt(
        G_CGS * m200c_msun * MSUN_G / (r200c_pkpc * KPC_CM)
    )
    v200c_km_s = float(v200c_cm_s / 1.0e5)
    t_dyn_s = r200c_pkpc * KPC_CM / v200c_cm_s
    t_dyn_gyr = float(t_dyn_s / SECONDS_PER_GYR)
    t_vir_k = float(virial_temperature_factor * v200c_km_s**2)
    metallicity = (
        float(z_hot_mass_fraction)
        if np.isfinite(z_hot_mass_fraction) and z_hot_mass_fraction > 0
        else 0.0
    )
    lambda_cooling = float(cooling_function(t_vir_k, metallicity))
    if not np.isfinite(lambda_cooling) or lambda_cooling <= 0:
        raise ValueError("Cooling function returned a non-positive value")

    if m_hot_msun == 0:
        r_cool_pkpc = 0.0
        rate = 0.0
        mode = "empty"
    else:
        numerator = (
            t_dyn_s * m_hot_msun * MSUN_G * lambda_cooling
        )
        denominator = (
            6.0
            * np.pi
            * mean_molecular_weight
            * PROTON_MASS_G
            * BOLTZMANN_CGS
            * t_vir_k
            * r200c_pkpc
            * KPC_CM
        )
        r_cool_pkpc = float(np.sqrt(numerator / denominator) / KPC_CM)
        fraction = min(r_cool_pkpc / r200c_pkpc, 1.0)
        rate = float(
            m_hot_msun / (t_dyn_gyr * 1.0e9) * fraction
        )
        mode = (
            "cooling_flow"
            if r_cool_pkpc < r200c_pkpc
            else "rapid_infall"
        )
    return {
        "v200c_km_s": v200c_km_s,
        "t_dyn_gyr": t_dyn_gyr,
        "t_vir_k": t_vir_k,
        "lambda_erg_cm3_s": lambda_cooling,
        "r_cool_isothermal_pkpc": r_cool_pkpc,
        "r_cool_isothermal_over_r200c": float(
            r_cool_pkpc / r200c_pkpc
        ),
        "sam_cooling_mode": mode,
        "rate_sam_isothermal_msun_per_yr": rate,
    }


def compute_sam_reheating_loading(
    *,
    vmax_km_s: float,
    v200c_km_s: float,
    epsilon_disk: float = 2.6,
    v_reheat_km_s: float = 480.0,
    beta_disk: float = 0.72,
    eta_halo: float = 0.62,
    v_eject_km_s: float = 100.0,
    beta_halo: float = 0.80,
    v_sn_km_s: float = 630.0,
) -> dict:
    """Return the Henriques et al. (2015) S1.7 mass loading.

    The nominal disc loading from eq. S19 is capped by the total
    supernova energy available through eqs. S16--S17.
    """

    positive = (
        vmax_km_s,
        v200c_km_s,
        epsilon_disk,
        v_reheat_km_s,
        beta_disk,
        eta_halo,
        v_eject_km_s,
        beta_halo,
        v_sn_km_s,
    )
    if not all(np.isfinite(value) and value > 0 for value in positive):
        raise ValueError("S1.7 inputs and parameters must be positive")
    loading_disk = epsilon_disk * (
        0.5 + (vmax_km_s / v_reheat_km_s) ** (-beta_disk)
    )
    coupling_halo = eta_halo * (
        0.5 + (vmax_km_s / v_eject_km_s) ** (-beta_halo)
    )
    loading_energy_cap = coupling_halo * (
        v_sn_km_s / v200c_km_s
    ) ** 2
    loading_effective = min(loading_disk, loading_energy_cap)
    return {
        "sn_loading_disk": float(loading_disk),
        "sn_coupling_halo": float(coupling_halo),
        "sn_loading_energy_cap": float(loading_energy_cap),
        "sn_loading_effective": float(loading_effective),
        "sn_reheating_energy_limited": bool(
            loading_energy_cap < loading_disk
        ),
    }


def compute_interval_sam_reheating(
    *,
    formed_stellar_mass_msun: float,
    dt_gyr: float,
    previous_loading: dict,
    current_loading: dict,
    rate_sam_isothermal_msun_per_yr: float,
) -> dict:
    """Compute interval reheating and a non-negative effective cooling rate."""

    if formed_stellar_mass_msun < 0:
        raise ValueError("formed_stellar_mass_msun cannot be negative")
    if dt_gyr <= 0:
        raise ValueError("dt_gyr must be positive")
    if rate_sam_isothermal_msun_per_yr < 0:
        raise ValueError("SAM cooling rate cannot be negative")
    loading_prev = float(previous_loading["sn_loading_effective"])
    loading_cur = float(current_loading["sn_loading_effective"])
    loading_mean = 0.5 * (loading_prev + loading_cur)
    reheated_mass = loading_mean * formed_stellar_mass_msun
    reheating_rate = reheated_mass / (dt_gyr * 1.0e9)
    effective_rate = max(
        rate_sam_isothermal_msun_per_yr - reheating_rate,
        0.0,
    )
    output = {
        "mstar_formed_interval_msun": float(formed_stellar_mass_msun),
        "sn_loading_effective_mean": float(loading_mean),
        "mass_sam_reheating_msun": float(reheated_mass),
        "rate_sam_reheating_msun_per_yr": float(reheating_rate),
        "rate_sam_sn_effective_msun_per_yr": float(effective_rate),
        "sam_sn_cooling_clipped": bool(
            reheating_rate >= rate_sam_isothermal_msun_per_yr
            and rate_sam_isothermal_msun_per_yr > 0
        ),
    }
    for suffix, values in (
        ("prev", previous_loading),
        ("cur", current_loading),
    ):
        for key, value in values.items():
            output[f"{key}_{suffix}"] = value
    return output
