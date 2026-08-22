"""Additional tracer scans required by the inward-delivery extension.

The expensive snapshot passes remain isolated from per-halo evaluation.  The
call order and cache paths exactly follow the original monolithic pipeline so
existing chunk caches retain their meaning and reuse behavior.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..utils.config import Phase18Config
from .tracer_batch import (
    build_grouped_id_catalog,
    build_parent_catalog,
    build_tracer_id_catalog,
    scan_snapshot_for_catalog,
    scan_snapshot_for_tracer_parents,
    tracer_sets_for_halo,
)
from .tracer_history import lookup_parent_particle_records


def prepare_tracer_products(
    states: list[dict],
    *,
    config: Phase18Config,
    cache_dir: str | Path,
    verbose: bool = True,
) -> dict:
    """Run all shared scans and return reusable per-halo lookup products.

    In addition to the baseline origin/fate queries, this extension resolves the
    snapshot-99 parents of the outer-hot and outer-cold source tracers.  No
    rates or statistics are evaluated in this I/O stage.
    """

    source_catalog = build_parent_catalog(states, role="source")
    target_catalog = build_parent_catalog(states, role="target")
    tracer_cache = Path(cache_dir) / "tracer_chunks"
    source_tracers = scan_snapshot_for_catalog(
        config.base_path,
        config.snap_prev,
        source_catalog,
        cache_dir=tracer_cache,
        block_size=config.tracer_read_block_size,
        max_workers=64,
        parallel_backend="process",
        verbose=verbose,
    )
    target_tracers = scan_snapshot_for_catalog(
        config.base_path,
        config.snap_cur,
        target_catalog,
        cache_dir=tracer_cache,
        block_size=config.tracer_read_block_size,
        max_workers=64,
        parallel_backend="process",
        verbose=verbose,
    )
    source_sets_by_halo = [
        tracer_sets_for_halo(source_tracers, halo_index)
        for halo_index in range(len(states))
    ]
    central_prev_catalog = build_grouped_id_catalog(
        [
            np.asarray(state["source"]["central_cold_ids"], dtype=np.uint64)
            for state in states
        ]
    )
    central_prev_tracers = scan_snapshot_for_catalog(
        config.base_path,
        config.snap_prev,
        central_prev_catalog,
        cache_dir=tracer_cache,
        block_size=config.tracer_read_block_size,
        max_workers=64,
        parallel_backend="process",
        verbose=verbose,
    )
    central_cur_tracers = [
        tracer_sets_for_halo(target_tracers, halo_index)["central_cold"]
        for halo_index in range(len(states))
    ]
    central_prev_tracer_list = [
        np.asarray(
            central_prev_tracers.get(
                halo_index,
                np.empty(0, dtype=np.uint64),
            ),
            dtype=np.uint64,
        )
        for halo_index in range(len(states))
    ]
    origin_query = build_tracer_id_catalog(central_cur_tracers)
    fate_query = build_tracer_id_catalog(central_prev_tracer_list)
    origin_parents = scan_snapshot_for_tracer_parents(
        config.base_path,
        config.snap_prev,
        origin_query,
        cache_dir=tracer_cache,
        block_size=config.tracer_read_block_size,
        max_workers=64,
        parallel_backend="process",
        verbose=verbose,
    )
    fate_parents = scan_snapshot_for_tracer_parents(
        config.base_path,
        config.snap_cur,
        fate_query,
        cache_dir=tracer_cache,
        block_size=config.tracer_read_block_size,
        max_workers=64,
        parallel_backend="process",
        verbose=verbose,
    )
    supply_fate_groups = []
    for source_sets in source_sets_by_halo:
        supply_fate_groups.extend(
            [source_sets["hot"], source_sets["outer_cold"]]
        )
    supply_fate_query = build_grouped_id_catalog(supply_fate_groups)
    supply_fate_parents = scan_snapshot_for_tracer_parents(
        config.base_path,
        config.snap_cur,
        supply_fate_query,
        cache_dir=tracer_cache,
        block_size=config.tracer_read_block_size,
        max_workers=64,
        parallel_backend="process",
        verbose=verbose,
    )
    all_origin_parent_ids = (
        np.concatenate(list(origin_parents.values()))
        if origin_parents
        else np.empty(0, dtype=np.uint64)
    )
    all_fate_parent_ids = (
        np.concatenate(list(fate_parents.values()))
        if fate_parents
        else np.empty(0, dtype=np.uint64)
    )
    all_supply_fate_parent_ids = (
        np.concatenate(list(supply_fate_parents.values()))
        if supply_fate_parents
        else np.empty(0, dtype=np.uint64)
    )
    particle_cache = Path(cache_dir) / "history_particle_chunks"
    origin_particle_records = lookup_parent_particle_records(
        config.base_path,
        config.snap_prev,
        all_origin_parent_ids,
        cache_dir=particle_cache,
        block_size=config.tracer_read_block_size,
        max_workers=64,
        verbose=verbose,
    )
    fate_particle_records = lookup_parent_particle_records(
        config.base_path,
        config.snap_cur,
        all_fate_parent_ids,
        cache_dir=particle_cache,
        block_size=config.tracer_read_block_size,
        max_workers=64,
        verbose=verbose,
    )
    supply_fate_particle_records = lookup_parent_particle_records(
        config.base_path,
        config.snap_cur,
        all_supply_fate_parent_ids,
        cache_dir=particle_cache,
        block_size=config.tracer_read_block_size,
        max_workers=64,
        verbose=verbose,
    )

    return {
        "source_tracers": source_tracers,
        "target_tracers": target_tracers,
        "source_sets_by_halo": source_sets_by_halo,
        "central_cur_tracers": central_cur_tracers,
        "central_prev_tracer_list": central_prev_tracer_list,
        "origin_parents": origin_parents,
        "fate_parents": fate_parents,
        "supply_fate_parents": supply_fate_parents,
        "origin_particle_records": origin_particle_records,
        "fate_particle_records": fate_particle_records,
        "supply_fate_particle_records": supply_fate_particle_records,
    }
