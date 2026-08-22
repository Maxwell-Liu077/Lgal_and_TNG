"""Final statistics and metadata assembly for Phase 1.8."""

from __future__ import annotations

from ..utils.config import Phase18Config
from .mass_statistics import (
    compute_all_normalized_statistics,
    compute_binned_statistics,
)
from ..io.tracer_history import compute_stacked_history_statistics
from ..utils.arrays import _records_to_arrays


def build_phase18_result(
    records: list[dict],
    *,
    config: Phase18Config,
    mode: str,
    dt_gyr: float,
    sample_metadata: dict,
    state_rejected: list[dict],
    membership_rejected: list[dict],
) -> dict:
    """Build the unchanged result tree and metadata contract."""

    halo_results = _records_to_arrays(records)
    binned = compute_binned_statistics(halo_results, config)
    normalized = compute_all_normalized_statistics(
        halo_results,
        config,
    )
    history_statistics = compute_stacked_history_statistics(
        halo_results,
        config,
    )
    metadata = {
        "phase": "1.8",
        "mode": mode,
        "base_path": config.base_path,
        "snap_prev": config.snap_prev,
        "snap_cur": config.snap_cur,
        "dt_gyr": dt_gyr,
        "mass_bin_edges": config.mass_bin_edges.tolist(),
        "sample_limit_per_bin": config.sample_per_bin,
        "selected_halo_count": len(records),
        "random_seed": config.random_seed,
        "state_rejected": state_rejected,
        "membership_rejected": membership_rejected,
        "sample_metadata": sample_metadata,
        "tracer_workers": config.tracer_workers,
        "tracer_backend": config.tracer_parallel_backend,
        "tracer_mass_msun": config.tracer_mass_msun,
        "source_definition": (
            "snap98: tracer hot source 0.1R200c<=r<R200c around "
            "GroupPos; SAM hot reservoir r<R200c; outer cold "
            "0.1R200c<=r<R200c around SubhaloPos; CGM mass includes "
            "all gas at 0.1R200c<=r<R200c around GroupPos; all gas "
            "sets exclude gas bound to non-central Subfind subhalos"
        ),
        "target_definition": (
            "snap99: central cold gas or central-bound real stars at "
            "r<0.1R200c; satellite-bound gas is excluded"
        ),
        "sam_definition": (
            "arithmetic mean of independently evaluated snap98 and "
            "snap99 Henriques S1.4 isothermal cooling rates; "
            "L-GALAXIES (SN) subtracts the S1.7 interval reheating "
            "rate inferred from TNG newly formed stellar mass and is "
            "clipped at zero; AGN feedback is not included"
        ),
        "history_definition": (
            "complete tracer-count fractions: origins of snap99 central "
            "cold gas are classified at snap98; fates of snap98 central "
            "cold gas are classified at snap99. All unmatched records "
            "are retained in the other category so each direction closes"
        ),
        "normalization_definition": (
            "per-halo rate divided by the snap98/snap99 endpoint-mean "
            "mass in Gyr^-1 for M200c, Mhot, Mcgm, and Mstar"
        ),
    }
    return {
        "metadata": metadata,
        "halo_results": halo_results,
        "binned_statistics": binned,
        "normalized_statistics": normalized,
        "history_statistics": history_statistics,
    }

