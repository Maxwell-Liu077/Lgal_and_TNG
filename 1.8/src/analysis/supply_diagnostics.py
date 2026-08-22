"""Reusable statistics for optional supply and feedback diagnostics.

These helpers operate only on existing per-halo arrays.  They never read TNG
files and never construct figures, which keeps analysis independent of I/O and
presentation.
"""

from __future__ import annotations

import numpy as np


NORMALIZATION_MASS_FIELDS = {
    "m200c": "m200c_mean_msun",
    "mhot": "m_hot_mean_msun",
    "mcgm": "m_cgm_mean_msun",
    "mstar": "mstar_mean_msun",
}


def _append_binned_percentiles(
    statistics: dict[str, np.ndarray],
    values: np.ndarray,
    bin_index: np.ndarray,
    *,
    short_name: str,
) -> dict[str, np.ndarray]:
    """Copy statistics and append one median and 16th/84th percentile series."""

    output = {
        key: np.array(value, copy=True)
        for key, value in statistics.items()
    }
    values = np.asarray(values, dtype=float)
    bin_index = np.asarray(bin_index, dtype=int)
    n_bins = len(output["mass_bin_center"])
    p16 = np.full(n_bins, np.nan)
    median = np.full(n_bins, np.nan)
    p84 = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=np.int32)
    for index in range(n_bins):
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


def add_agn_percentiles(
    statistics: dict[str, np.ndarray],
    values: np.ndarray,
    bin_index: np.ndarray,
) -> dict[str, np.ndarray]:
    """Append the AGN-corrected SAM series to existing binned statistics."""

    return _append_binned_percentiles(
        statistics,
        values,
        bin_index,
        short_name="sam_agn",
    )


def compute_central_cold_to_outer_cold_rate(
    halo_results: dict[str, np.ndarray],
    *,
    tracer_mass_msun: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert complete-history cold-to-cold tracer counts into rates."""

    counts = np.asarray(halo_results["n_fate_outer_cold"], dtype=float)
    dt_gyr = np.asarray(halo_results["dt_gyr"], dtype=float)
    rates = np.full(counts.shape, np.nan)
    valid = (
        np.isfinite(counts)
        & (counts >= 0)
        & np.isfinite(dt_gyr)
        & (dt_gyr > 0)
    )
    rates[valid] = counts[valid] * tracer_mass_msun / (dt_gyr[valid] * 1.0e9)
    return rates, valid


def _normalized_combined_rate(
    halo_results: dict[str, np.ndarray],
    *,
    normalization: str,
    include_cold_to_cold: bool,
    tracer_mass_msun: float | None = None,
) -> np.ndarray:
    """Build one combined supply rate and normalize it per halo."""

    if normalization not in NORMALIZATION_MASS_FIELDS:
        raise ValueError("Unknown normalization")
    retained = np.asarray(
        halo_results["rate_cooling_plus_inflow_msun_per_yr"],
        dtype=float,
    )
    cold_to_hot = np.asarray(
        halo_results["rate_tng_cold_to_hot_transition_msun_per_yr"],
        dtype=float,
    )
    components = [retained, cold_to_hot]
    if include_cold_to_cold:
        if tracer_mass_msun is None:
            raise ValueError("tracer_mass_msun is required for cold-to-cold")
        cold_to_cold, _ = compute_central_cold_to_outer_cold_rate(
            halo_results,
            tracer_mass_msun=tracer_mass_msun,
        )
        components.append(cold_to_cold)
    combined = np.sum(np.vstack(components), axis=0)
    masses = np.asarray(
        halo_results[NORMALIZATION_MASS_FIELDS[normalization]],
        dtype=float,
    )
    valid = np.isfinite(masses) & (masses > 0)
    for component in components:
        valid &= np.isfinite(component)
    normalized = np.full(combined.shape, np.nan)
    normalized[valid] = combined[valid] / masses[valid] * 1.0e9
    return normalized


def add_cold_to_hot_supply_percentiles(
    statistics: dict[str, np.ndarray],
    halo_results: dict[str, np.ndarray],
    normalization: str,
) -> dict[str, np.ndarray]:
    """Append the historical Hot + Cold + cold-to-hot normalized curve."""

    values = _normalized_combined_rate(
        halo_results,
        normalization=normalization,
        include_cold_to_cold=False,
    )
    return _append_binned_percentiles(
        statistics,
        values,
        halo_results["mass_bin_index"],
        short_name="cooling_plus_cold_to_hot",
    )


def add_cold_outflow_corrections_percentiles(
    statistics: dict[str, np.ndarray],
    halo_results: dict[str, np.ndarray],
    normalization: str,
    *,
    tracer_mass_msun: float,
) -> dict[str, np.ndarray]:
    """Append the curve containing both central-cold fate corrections."""

    values = _normalized_combined_rate(
        halo_results,
        normalization=normalization,
        include_cold_to_cold=True,
        tracer_mass_msun=tracer_mass_msun,
    )
    return _append_binned_percentiles(
        statistics,
        values,
        halo_results["mass_bin_index"],
        short_name="cooling_plus_cold_to_hot_and_cold",
    )


def paired_bootstrap_summary(
    values: np.ndarray,
    *,
    bin_index: np.ndarray,
    n_bins: int,
    random_seed: int,
    seed_offset: int,
    bootstrap_samples: int = 4000,
) -> dict[str, np.ndarray]:
    """Return medians and paired-bootstrap 68 percent confidence intervals."""

    values = np.asarray(values, dtype=float)
    bin_index = np.asarray(bin_index, dtype=int)
    n_series = values.shape[0]
    median = np.full((n_series, n_bins), np.nan)
    ci16 = np.full((n_series, n_bins), np.nan)
    ci84 = np.full((n_series, n_bins), np.nan)
    counts = np.zeros(n_bins, dtype=np.int32)
    for index in range(n_bins):
        selected = values[:, bin_index == index].T
        selected = selected[np.all(np.isfinite(selected), axis=1)]
        counts[index] = len(selected)
        if not len(selected):
            continue
        median[:, index] = np.median(selected, axis=0)
        rng = np.random.default_rng(
            np.random.SeedSequence([random_seed, int(seed_offset), index])
        )
        draws = rng.integers(
            0,
            len(selected),
            size=(bootstrap_samples, len(selected)),
        )
        bootstrap_medians = np.median(selected[draws], axis=1)
        ci16[:, index], ci84[:, index] = np.percentile(
            bootstrap_medians,
            [16.0, 84.0],
            axis=0,
        )
    return {
        "median": median,
        "ci16": ci16,
        "ci84": ci84,
        "counts": counts,
    }


def build_supply_comparison_statistics(
    halo_results: dict[str, np.ndarray],
    base_statistics: dict[str, np.ndarray],
    *,
    normalization: str,
    tracer_mass_msun: float,
    random_seed: int,
    seed_offset: int,
    bootstrap_samples: int = 4000,
) -> dict[str, np.ndarray]:
    """Build the four paired supply curves used by the comparison figure."""

    cold_to_cold, _ = compute_central_cold_to_outer_cold_rate(
        halo_results,
        tracer_mass_msun=tracer_mass_msun,
    )
    lgal = np.asarray(
        halo_results["rate_sam_isothermal_msun_per_yr"],
        dtype=float,
    )
    retained = np.asarray(
        halo_results["rate_cooling_plus_inflow_msun_per_yr"],
        dtype=float,
    )
    cold_to_hot = np.asarray(
        halo_results["rate_tng_cold_to_hot_transition_msun_per_yr"],
        dtype=float,
    )
    rates = np.vstack(
        [
            lgal,
            retained,
            retained + cold_to_hot,
            retained + cold_to_hot + cold_to_cold,
        ]
    )
    masses = np.asarray(
        halo_results[NORMALIZATION_MASS_FIELDS[normalization]],
        dtype=float,
    )
    normalized = np.full(rates.shape, np.nan)
    valid_mass = np.isfinite(masses) & (masses > 0)
    normalized[:, valid_mass] = rates[:, valid_mass] / masses[valid_mass] * 1.0e9
    return paired_bootstrap_summary(
        normalized,
        bin_index=halo_results["mass_bin_index"],
        n_bins=len(base_statistics["mass_bin_center"]),
        random_seed=random_seed,
        seed_offset=seed_offset,
        bootstrap_samples=bootstrap_samples,
    )
