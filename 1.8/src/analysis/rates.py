"""Shared tracer rates and optional total inward-delivery assembly.

The optional extension adds thermally inclusive inner-hot delivery channels
while keeping the baseline channel names and numerical definitions unchanged.
"""

from __future__ import annotations

import numpy as np


def compute_total_inward_delivery(
    transition: dict,
    *,
    n_outer_hot_to_inner_hot: int,
    n_outer_cold_to_inner_hot: int,
    tracer_mass_msun: float,
    dt_gyr: float,
) -> dict:
    """Add successful central supply and inner-hot arrival channels."""

    counts = (
        int(n_outer_hot_to_inner_hot),
        int(n_outer_cold_to_inner_hot),
    )
    if (
        any(count < 0 for count in counts)
        or tracer_mass_msun <= 0
        or dt_gyr <= 0
    ):
        raise ValueError("Inward-delivery inputs must be non-negative")
    n_inner_hot = sum(counts)
    n_total = int(transition["n_cooling_plus_inflow"]) + n_inner_hot
    scale = tracer_mass_msun / (dt_gyr * 1.0e9)
    mass_scale = tracer_mass_msun
    return {
        "n_outer_hot_to_inner_hot": counts[0],
        "mass_outer_hot_to_inner_hot_msun": counts[0] * mass_scale,
        "rate_outer_hot_to_inner_hot_msun_per_yr": counts[0] * scale,
        "n_outer_cold_to_inner_hot": counts[1],
        "mass_outer_cold_to_inner_hot_msun": counts[1] * mass_scale,
        "rate_outer_cold_to_inner_hot_msun_per_yr": counts[1] * scale,
        "n_inner_hot_delivery": n_inner_hot,
        "mass_inner_hot_delivery_msun": n_inner_hot * mass_scale,
        "rate_inner_hot_delivery_msun_per_yr": n_inner_hot * scale,
        "n_tng_total_inward_delivery": n_total,
        "mass_tng_total_inward_delivery_msun": (
            float(transition["mass_cooling_plus_inflow_msun"])
            + n_inner_hot * mass_scale
        ),
        "rate_tng_total_inward_delivery_msun_per_yr": (
            float(transition["rate_cooling_plus_inflow_msun_per_yr"])
            + n_inner_hot * scale
        ),
    }


def _mass_for_ids(
    query_ids: np.ndarray,
    sorted_ids: np.ndarray,
    sorted_masses: np.ndarray,
) -> float:
    """Sum masses for query IDs using the pre-sorted particle catalog."""

    query_ids = np.asarray(query_ids, dtype=np.uint64)
    if len(query_ids) == 0:
        return 0.0
    positions = np.searchsorted(sorted_ids, query_ids)
    valid = positions < len(sorted_ids)
    if not np.all(valid):
        raise ValueError("Matched ParticleID absent from target mass table")
    if not np.all(sorted_ids[positions] == query_ids):
        raise ValueError("Matched ParticleID absent from target mass table")
    return float(np.asarray(sorted_masses)[positions].sum())


def compute_particleid_rates(
    source: dict,
    target: dict,
    *,
    dt_gyr: float,
) -> dict:
    """Approximate 98->99 channels by persistent ParticleIDs."""

    if dt_gyr <= 0:
        raise ValueError("dt_gyr must be positive")
    hot_ids = np.asarray(source["hot_ids"], dtype=np.uint64)
    outer_ids = np.asarray(source["outer_cold_ids"], dtype=np.uint64)
    target_ids = np.asarray(target["target_ids"], dtype=np.uint64)
    hot_cross = np.intersect1d(hot_ids, target_ids, assume_unique=True)
    outer_cross = np.intersect1d(
        outer_ids,
        target_ids,
        assume_unique=True,
    )
    if len(np.intersect1d(hot_cross, outer_cross, assume_unique=True)):
        raise ValueError("ParticleID channels overlap")
    mass_hot = _mass_for_ids(
        hot_cross,
        target_ids,
        target["target_masses_msun"],
    )
    mass_outer = _mass_for_ids(
        outer_cross,
        target_ids,
        target["target_masses_msun"],
    )
    scale = 1.0 / (dt_gyr * 1.0e9)
    return {
        "n_hot_halo_cooling": int(len(hot_cross)),
        "n_outer_cold_inflow": int(len(outer_cross)),
        "n_cooling_plus_inflow": int(
            len(hot_cross) + len(outer_cross)
        ),
        "mass_hot_halo_cooling_msun": mass_hot,
        "mass_outer_cold_inflow_msun": mass_outer,
        "mass_cooling_plus_inflow_msun": mass_hot + mass_outer,
        "rate_hot_halo_cooling_msun_per_yr": mass_hot * scale,
        "rate_outer_cold_inflow_msun_per_yr": mass_outer * scale,
        "rate_cooling_plus_inflow_msun_per_yr": (
            (mass_hot + mass_outer) * scale
        ),
    }


def compute_tracer_rates(
    *,
    tracer_hot_prev: np.ndarray,
    tracer_outer_cold_prev: np.ndarray,
    tracer_central_cold_cur: np.ndarray,
    tracer_central_star_cur: np.ndarray,
    tracer_mass_msun: float,
    dt_gyr: float,
) -> dict:
    """Exact MC-tracer channels for the 98->99 interval."""

    if tracer_mass_msun <= 0 or dt_gyr <= 0:
        raise ValueError("Tracer mass and dt must be positive")
    hot = np.unique(np.asarray(tracer_hot_prev, dtype=np.uint64))
    outer = np.unique(
        np.asarray(tracer_outer_cold_prev, dtype=np.uint64)
    )
    cold = np.unique(
        np.asarray(tracer_central_cold_cur, dtype=np.uint64)
    )
    stars = np.unique(
        np.asarray(tracer_central_star_cur, dtype=np.uint64)
    )
    if len(np.intersect1d(hot, outer, assume_unique=True)):
        raise ValueError("Tracer source channels overlap")
    if len(np.intersect1d(cold, stars, assume_unique=True)):
        raise ValueError("Tracer target channels overlap")
    target = np.union1d(cold, stars)
    cooling = np.intersect1d(hot, target, assume_unique=True)
    inflow = np.intersect1d(outer, target, assume_unique=True)
    if len(np.intersect1d(cooling, inflow, assume_unique=True)):
        raise ValueError("Tracer transition channels overlap")
    n_cooling = len(cooling)
    n_inflow = len(inflow)
    n_total = n_cooling + n_inflow
    scale = tracer_mass_msun / (dt_gyr * 1.0e9)
    return {
        "n_hot_halo_cooling": int(n_cooling),
        "n_outer_cold_inflow": int(n_inflow),
        "n_cooling_plus_inflow": int(n_total),
        "mass_hot_halo_cooling_msun": n_cooling * tracer_mass_msun,
        "mass_outer_cold_inflow_msun": n_inflow * tracer_mass_msun,
        "mass_cooling_plus_inflow_msun": n_total * tracer_mass_msun,
        "rate_hot_halo_cooling_msun_per_yr": n_cooling * scale,
        "rate_outer_cold_inflow_msun_per_yr": n_inflow * scale,
        "rate_cooling_plus_inflow_msun_per_yr": n_total * scale,
        "poisson_fraction_hot_halo_cooling": (
            1.0 / np.sqrt(n_cooling) if n_cooling else np.nan
        ),
        "poisson_fraction_outer_cold_inflow": (
            1.0 / np.sqrt(n_inflow) if n_inflow else np.nan
        ),
        "poisson_fraction_cooling_plus_inflow": (
            1.0 / np.sqrt(n_total) if n_total else np.nan
        ),
    }
