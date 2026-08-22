"""Mass-bin summaries shared by the baseline and inward-delivery analyses.

Normalization is deliberately applied to each halo before percentile
statistics are computed.  This preserves the existing nonlinear statistical
contract and prevents reconstruction from bin-center masses.  The field
registry contains both analysis variants; only fields present in the supplied
per-halo result are processed, so each workflow retains its original output
schema.
"""

from __future__ import annotations

import numpy as np

from ..utils.config import Phase18Config


RATE_FIELDS = {
    "sam": "rate_sam_isothermal_msun_per_yr",
    "sam_sn": "rate_sam_sn_effective_msun_per_yr",
    "sam_reheating": "rate_sam_reheating_msun_per_yr",
    "tng_cold_to_hot": (
        "rate_tng_cold_to_hot_transition_msun_per_yr"
    ),
    "outer_cold_inflow": "rate_outer_cold_inflow_msun_per_yr",
    "hot_halo_cooling": "rate_hot_halo_cooling_msun_per_yr",
    "cooling_plus_inflow": "rate_cooling_plus_inflow_msun_per_yr",
    "outer_hot_to_inner_hot": (
        "rate_outer_hot_to_inner_hot_msun_per_yr"
    ),
    "outer_cold_to_inner_hot": (
        "rate_outer_cold_to_inner_hot_msun_per_yr"
    ),
    "inner_hot_delivery": "rate_inner_hot_delivery_msun_per_yr",
    "tng_total_inward_delivery": (
        "rate_tng_total_inward_delivery_msun_per_yr"
    ),
}

MASS_FIELDS = {
    "sam_reheating_mass": "mass_sam_reheating_msun",
    "tng_cold_to_hot_mass": (
        "mass_tng_cold_to_hot_transition_msun"
    ),
    "outer_hot_to_inner_hot_mass": (
        "mass_outer_hot_to_inner_hot_msun"
    ),
    "outer_cold_to_inner_hot_mass": (
        "mass_outer_cold_to_inner_hot_msun"
    ),
    "inner_hot_delivery_mass": "mass_inner_hot_delivery_msun",
    "tng_total_inward_delivery_mass": (
        "mass_tng_total_inward_delivery_msun"
    ),
}

NORMALIZATION_FIELDS = {
    "m200c": "m200c_mean_msun",
    "mhot": "m_hot_mean_msun",
    "mcgm": "m_cgm_mean_msun",
    "mstar": "mstar_mean_msun",
}


def compute_binned_statistics(
    halo_results: dict[str, np.ndarray],
    config: Phase18Config,
) -> dict[str, np.ndarray]:
    """Compute median, 16th, and 84th percentiles in fixed mass bins."""

    edges = config.mass_bin_edges
    centers = config.mass_bin_centers
    bin_index = np.asarray(
        halo_results["mass_bin_index"],
        dtype=int,
    )
    output: dict[str, np.ndarray] = {
        "mass_bin_index": np.arange(len(centers), dtype=np.int16),
        "mass_bin_low": edges[:-1],
        "mass_bin_high": edges[1:],
        "mass_bin_center": centers,
        "n_halos": np.asarray(
            [np.count_nonzero(bin_index == i) for i in range(len(centers))],
            dtype=np.int32,
        ),
    }
    for short_name, field in {
        **RATE_FIELDS,
        **MASS_FIELDS,
    }.items():
        # Baseline-only and inward-delivery-only fields share this registry.
        # Skipping absent fields preserves the exact schema of each workflow.
        if field not in halo_results:
            continue
        values = np.asarray(halo_results[field], dtype=float)
        p16 = np.full(len(centers), np.nan)
        median = np.full(len(centers), np.nan)
        p84 = np.full(len(centers), np.nan)
        counts = np.zeros(len(centers), dtype=np.int32)
        for index in range(len(centers)):
            selected = values[bin_index == index]
            selected = selected[np.isfinite(selected)]
            counts[index] = len(selected)
            if len(selected):
                p16[index], median[index], p84[index] = np.percentile(
                    selected,
                    [16.0, 50.0, 84.0],
                )
        output[f"{short_name}_p16"] = p16
        output[f"{short_name}_median"] = median
        output[f"{short_name}_p84"] = p84
        output[f"{short_name}_n_finite"] = counts
    return output


def normalize_halo_rates(
    halo_results: dict[str, np.ndarray],
    *,
    normalization: str,
) -> dict[str, np.ndarray]:
    """Return rate/Mnormalization for every halo in units of Gyr^-1."""

    if normalization not in NORMALIZATION_FIELDS:
        raise ValueError(
            "normalization must be one of: "
            + ", ".join(NORMALIZATION_FIELDS)
        )
    mass_field = NORMALIZATION_FIELDS[normalization]
    masses = np.asarray(halo_results[mass_field], dtype=float)
    valid_mass = np.isfinite(masses) & (masses > 0)
    normalized = {
        key: np.array(values, copy=True)
        for key, values in halo_results.items()
    }
    for field in RATE_FIELDS.values():
        if field not in halo_results:
            continue
        rates = np.asarray(halo_results[field], dtype=float)
        values = np.full(rates.shape, np.nan, dtype=float)
        valid = valid_mass & np.isfinite(rates)
        values[valid] = rates[valid] / masses[valid] * 1.0e9
        normalized[field] = values
    return normalized


def compute_all_normalized_statistics(
    halo_results: dict[str, np.ndarray],
    config: Phase18Config,
) -> dict[str, dict[str, np.ndarray]]:
    """Compute statistics for four endpoint-mean mass normalizations."""

    return {
        normalization: compute_binned_statistics(
            normalize_halo_rates(
                halo_results,
                normalization=normalization,
            ),
            config,
        )
        for normalization in NORMALIZATION_FIELDS
    }
