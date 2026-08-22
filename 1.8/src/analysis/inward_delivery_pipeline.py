"""Top-level orchestration for the integrated inward-delivery extension.

The coordinator shares configuration, physics, sample selection, endpoint
states, caches, and serializers with the baseline workflow.  Only the extra
tracer query, per-halo evaluation, and result metadata differ.
"""

from __future__ import annotations

from pathlib import Path

from ..utils.config import Phase18Config
from ..utils.cosmology import snapshot_interval_gyr
from .inward_delivery_evaluation import evaluate_halos
from .inward_delivery_result import build_phase181_result
from ..io.sample import attach_subfind_membership, load_or_build_sample
from ..io.state_cache import prepare_halo_states
from ..io.inward_delivery_products import prepare_tracer_products


def run_phase18(
    cooling_function,
    *,
    mode: str = "slow",
    config: Phase18Config | None = None,
    cache_dir: str | Path = "output/cache",
    rebuild_sample: bool = False,
    verbose: bool = True,
) -> dict:
    """Run the integrated inward-delivery workflow.

    The function name and signature remain historical for notebook and script
    compatibility.  Only orchestration was changed; validation, cache reuse,
    scientific fields, metadata strings, and returned keys are unchanged.
    """

    if config is None:
        config = Phase18Config()
    if mode != "slow":
        raise ValueError("Phase 1.8 now supports slow MC-tracer mode only")
    try:
        import illustris_python as il
    except ImportError as exc:
        raise ImportError("illustris_python is required for Phase 1.8") from exc

    sample, sample_metadata = load_or_build_sample(
        config,
        cache_dir=cache_dir,
        rebuild=rebuild_sample,
        verbose=verbose,
    )
    if (
        not sample
        or "sample_index" not in sample
        or len(sample["sample_index"]) == 0
    ):
        raise ValueError("No halos were selected")
    sample, membership_rejected = attach_subfind_membership(
        sample,
        config=config,
        verbose=verbose,
    )
    if not sample or len(sample.get("sample_index", [])) == 0:
        raise ValueError("No central-at-both-snapshots halos remain")

    header_prev = il.groupcat.loadHeader(
        config.base_path,
        config.snap_prev,
    )
    header_cur = il.groupcat.loadHeader(
        config.base_path,
        config.snap_cur,
    )
    dt_gyr = snapshot_interval_gyr(
        header_prev,
        header_cur,
        h=config.h,
        omega_m=config.omega_m,
        omega_lambda=config.omega_lambda,
    )
    sample, states, state_rejected = prepare_halo_states(
        sample,
        config=config,
        header_prev=header_prev,
        header_cur=header_cur,
        cache_dir=cache_dir,
        verbose=verbose,
    )
    if not states:
        raise ValueError("No valid halo states were loaded")

    tracer_products = prepare_tracer_products(
        states,
        config=config,
        cache_dir=cache_dir,
        verbose=verbose,
    )
    records = evaluate_halos(
        sample,
        states,
        tracer_products,
        cooling_function,
        config=config,
        header_prev=header_prev,
        header_cur=header_cur,
        dt_gyr=dt_gyr,
    )
    return build_phase181_result(
        records,
        config=config,
        mode=mode,
        dt_gyr=dt_gyr,
        sample_metadata=sample_metadata,
        state_rejected=state_rejected,
        membership_rejected=membership_rejected,
    )
