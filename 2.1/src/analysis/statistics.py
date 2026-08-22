"""Per-halo normalization, mass-bin summaries, and AGN quartile statistics."""

from __future__ import annotations

import numpy as np

from ..utils.config import Phase21Config


RATE_FIELDS = (
    "rate_sam_isothermal_msun_per_yr",
    "rate_cool_iso_msun_per_yr",
    "mdot_bh_msun_per_yr", "mdot_heat_h15_msun_per_yr",
    "rate_total_in_msun_per_yr", "rate_first_in_msun_per_yr", "rate_recycled_in_a_msun_per_yr",
    "rate_stay_in_msun_per_yr", "rate_single_in_msun_per_yr", "rate_recycled_in_b_msun_per_yr",
    "rate_other_in_msun_per_yr", "rate_total_out_msun_per_yr", "rate_stay_out_msun_per_yr",
    "rate_recycled_out_msun_per_yr", "rate_other_out_msun_per_yr", "rate_first_in_eff_msun_per_yr",
    "rate_recycled_in_eff_msun_per_yr", "rate_pf_in_msun_per_yr", "rate_supply_msun_per_yr",
    "rate_feedback_msun_per_yr",
)


def _percentiles(values: np.ndarray) -> tuple[float, float, float, int]:
    """Return p16, median, p84, and finite count."""

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return np.nan, np.nan, np.nan, 0
    p16, median, p84 = np.percentile(finite, [16.0, 50.0, 84.0])
    return float(p16), float(median), float(p84), int(len(finite))


def compute_binned_statistics(halo_results: dict[str, np.ndarray], config: Phase21Config) -> dict[str, np.ndarray]:
    """Compute per-halo p16/median/p84 summaries in final mass bins."""

    centers = config.mass_bin_centers
    output = {"mass_bin_index": np.arange(len(centers), dtype=np.int16), "mass_bin_low": config.mass_bin_edges[:-1], "mass_bin_high": config.mass_bin_edges[1:], "mass_bin_center": centers}
    bins = np.asarray(halo_results.get("mass_bin_index", []), dtype=int)
    output["n_halos"] = np.asarray([np.count_nonzero(bins == index) for index in range(len(centers))], dtype=np.int32)
    for field in RATE_FIELDS:
        if field not in halo_results:
            continue
        values = np.asarray(halo_results[field], dtype=float)
        p16 = np.full(len(centers), np.nan)
        median = np.full(len(centers), np.nan)
        p84 = np.full(len(centers), np.nan)
        counts = np.zeros(len(centers), dtype=np.int32)
        for index in range(len(centers)):
            p16[index], median[index], p84[index], counts[index] = _percentiles(values[bins == index])
        short = field.removeprefix("rate_").removesuffix("_msun_per_yr")
        output[f"{short}_p16"] = p16
        output[f"{short}_median"] = median
        output[f"{short}_p84"] = p84
        output[f"{short}_n_finite"] = counts
    return output


def normalize_halo_rates(halo_results: dict[str, np.ndarray], normalization: str) -> dict[str, np.ndarray]:
    """Normalize each halo rate by its own Mhot or Mstar in Gyr^-1."""

    field = {"mhot": "m_hot_mean_msun", "mstar": "mstar_mean_msun"}.get(normalization)
    if field is None:
        raise ValueError("normalization must be mhot or mstar")
    masses = np.asarray(halo_results[field], dtype=float)
    output = {key: np.asarray(value).copy() for key, value in halo_results.items()}
    valid = np.isfinite(masses) & (masses > 0)
    for rate_field in RATE_FIELDS:
        if rate_field not in output:
            continue
        values = np.asarray(output[rate_field], dtype=float)
        normalized = np.full(values.shape, np.nan)
        normalized[valid & np.isfinite(values)] = values[valid & np.isfinite(values)] / masses[valid & np.isfinite(values)] * 1.0e9
        output[rate_field] = normalized
    return output


def compute_all_normalized_statistics(halo_results: dict[str, np.ndarray], config: Phase21Config) -> dict[str, dict[str, np.ndarray]]:
    """Build mass-bin statistics for both requested normalizations."""

    return {normalization: compute_binned_statistics(normalize_halo_rates(halo_results, normalization), config) for normalization in ("mhot", "mstar")}


def compute_agn_quartile_statistics(halo_results: dict[str, np.ndarray], config: Phase21Config) -> dict[str, np.ndarray]:
    """Summarize average xcool by mass bin and AGN quartile."""

    centers = config.mass_bin_centers
    output = {"mass_bin_center": centers, "quartile": np.arange(1, 5, dtype=np.int8)}
    values = np.asarray(halo_results.get("x_cool_mean", []), dtype=float)
    bins = np.asarray(halo_results.get("mass_bin_index", []), dtype=int)
    quartiles = np.asarray(halo_results.get("agn_quartile", []), dtype=int)
    medians = np.full((4, len(centers)), np.nan)
    p16 = np.full_like(medians, np.nan)
    p84 = np.full_like(medians, np.nan)
    counts = np.zeros_like(medians, dtype=np.int32)
    assigned = np.zeros_like(medians, dtype=np.int32)
    for q in range(1, 5):
        for index in range(len(centers)):
            mask = (bins == index) & (quartiles == q)
            assigned[q - 1, index] = np.count_nonzero(mask)
            p16[q - 1, index], medians[q - 1, index], p84[q - 1, index], counts[q - 1, index] = _percentiles(values[mask])
    output.update({"median": medians, "p16": p16, "p84": p84, "n_finite": counts, "n_assigned": assigned})
    return output


def compute_composition_statistics(halo_results: dict[str, np.ndarray], config: Phase21Config) -> dict[str, np.ndarray]:
    """Compute mass-weighted four-part entry and three-part exit compositions."""

    bins = np.asarray(halo_results.get("mass_bin_index", []), dtype=int)
    n_bins = len(config.mass_bin_centers)
    in_fields = ("rate_first_in_eff_msun_per_yr", "rate_recycled_in_eff_msun_per_yr", "rate_stay_in_msun_per_yr", "rate_other_in_msun_per_yr")
    out_fields = ("rate_stay_out_msun_per_yr", "rate_recycled_out_msun_per_yr", "rate_other_out_msun_per_yr")
    output = {"mass_bin_center": config.mass_bin_centers, "in_component_names": np.asarray(("first-in effective", "recycled-in effective", "stay-in", "other")), "out_component_names": np.asarray(("stay-out", "recycled-out", "other"))}
    for prefix, fields in (("in", in_fields), ("out", out_fields)):
        totals = np.zeros((len(fields), n_bins), dtype=float)
        for component, field in enumerate(fields):
            values = np.asarray(halo_results.get(field, np.zeros(len(bins))), dtype=float)
            for index in range(n_bins):
                selected = values[bins == index]
                totals[component, index] = np.nansum(selected)
        component_total = totals.sum(axis=0)
        total_field = f"rate_total_{prefix}_msun_per_yr"
        denominator = np.zeros(n_bins, dtype=float)
        total_values = np.asarray(halo_results.get(total_field, np.full(len(bins), np.nan)), dtype=float)
        for index in range(n_bins):
            selected = total_values[bins == index]
            denominator[index] = np.nansum(selected) if np.any(np.isfinite(selected)) else component_total[index]
        closure_error = denominator - component_total
        tolerance = 1.0e-12 * np.maximum(1.0, np.abs(denominator))
        if np.any(np.abs(closure_error) > tolerance):
            raise ValueError(f"{prefix} composition does not close to rate_total_{prefix}")
        fractions = np.full_like(totals, np.nan)
        valid = denominator > 0
        fractions[:, valid] = totals[:, valid] / denominator[valid]
        output[f"{prefix}_component_rates"] = totals
        output[f"{prefix}_component_fractions"] = fractions
        output[f"{prefix}_total"] = denominator
        output[f"{prefix}_closure_error"] = closure_error
        count_field = f"n_total_{prefix}"
        counts = np.asarray(halo_results.get(count_field, np.zeros(len(bins))), dtype=float)
        output[f"{prefix}_event_count"] = np.asarray(
            [np.nansum(counts[bins == index]) for index in range(n_bins)],
            dtype=np.int64,
        )
        output[f"{prefix}_halo_count_with_events"] = np.asarray(
            [np.count_nonzero(np.isfinite(counts[bins == index]) & (counts[bins == index] > 0)) for index in range(n_bins)],
            dtype=np.int32,
        )
    return output
