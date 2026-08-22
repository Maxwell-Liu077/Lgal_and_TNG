"""Central-galaxy selection, MPB validation, and sample serialization.

Raw group catalogs are read here, while reusable sample caches are written via
explicit caller-provided paths.  Scientific selection defaults are unchanged.
"""

from __future__ import annotations

import json
import warnings
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from pathlib import Path

import numpy as np

from ..utils.config import Phase18Config
from .mpb import load_two_snapshot_mpb


HALO_FIELDS = [
    "GroupFirstSub",
    "Group_M_Crit200",
    "Group_R_Crit200",
]

SUBHALO_FIELDS = [
    "SubhaloFlag",
    "SubhaloMassInRadType",
]


def load_z0_central_candidates(
    base_path: str,
    config: Phase18Config,
) -> dict[str, np.ndarray]:
    """Load z=0 centrals in the requested two-half-mass-radius Mstar range."""

    try:
        import illustris_python as il
    except ImportError as exc:
        raise ImportError(
            "illustris_python is required to select TNG halos"
        ) from exc

    halos = il.groupcat.loadHalos(
        base_path,
        config.snap_cur,
        fields=HALO_FIELDS,
    )
    if not isinstance(halos, dict):
        raise ValueError("Expected a dictionary from loadHalos")
    subhalos = il.groupcat.loadSubhalos(
        base_path,
        config.snap_cur,
        fields=SUBHALO_FIELDS,
    )
    if not isinstance(subhalos, dict):
        raise ValueError("Expected a dictionary from loadSubhalos")

    first_sub = np.asarray(halos["GroupFirstSub"], dtype=np.int64)
    m200c_msun = (
        np.asarray(halos["Group_M_Crit200"], dtype=float)
        * 1.0e10
        / config.h
    )
    r200c_ckpc_h = np.asarray(
        halos["Group_R_Crit200"],
        dtype=float,
    )
    subhalo_flag = np.asarray(subhalos["SubhaloFlag"], dtype=bool)
    stellar_mass_in_rad_type = np.asarray(
        subhalos["SubhaloMassInRadType"],
        dtype=float,
    )
    if stellar_mass_in_rad_type.ndim != 2 or (
        stellar_mass_in_rad_type.shape[1] <= 4
    ):
        raise ValueError("SubhaloMassInRadType must contain the stellar type")
    first_sub_in_range = (first_sub >= 0) & (first_sub < len(subhalo_flag))
    mstar_msun = np.full(len(first_sub), np.nan, dtype=float)
    flagged = np.zeros(len(first_sub), dtype=bool)
    mstar_msun[first_sub_in_range] = (
        stellar_mass_in_rad_type[first_sub[first_sub_in_range], 4]
        * 1.0e10
        / config.h
    )
    flagged[first_sub_in_range] = subhalo_flag[
        first_sub[first_sub_in_range]
    ]
    valid = (
        first_sub_in_range
        & flagged
        & np.isfinite(mstar_msun)
        & (mstar_msun > 0)
        & (m200c_msun > 0)
        & (r200c_ckpc_h > 0)
    )
    log_mass = np.full(len(mstar_msun), np.nan)
    log_mass[valid] = np.log10(mstar_msun[valid])
    valid &= (
        (log_mass >= config.log_mstar_min)
        & (log_mass <= config.log_mstar_max)
    )

    group_ids = np.where(valid)[0].astype(np.int64)
    subhalo_ids = first_sub[valid]
    stellar_mass = mstar_msun[valid]
    halo_mass = m200c_msun[valid]
    log_mass = log_mass[valid]
    r200c = r200c_ckpc_h[valid]

    edges = config.mass_bin_edges
    bin_index = np.searchsorted(edges, log_mass, side="right") - 1
    bin_index[log_mass == edges[-1]] = len(edges) - 2
    inside = (bin_index >= 0) & (bin_index < len(edges) - 1)
    return {
        "group_id_z0": group_ids[inside],
        "subhalo_id_z0": subhalo_ids[inside],
        "mstar_z0_msun": stellar_mass[inside],
        "log_mstar_z0": log_mass[inside],
        "m200c_z0_msun": halo_mass[inside],
        "log_m200c_z0": np.log10(halo_mass[inside]),
        "r200c_z0_ckpc_h": r200c[inside],
        "mass_bin_index": bin_index[inside].astype(np.int16),
    }


def _records_to_table(records: list[dict]) -> dict[str, np.ndarray]:
    """Convert schema-identical sample records to a dictionary of arrays."""

    if not records:
        return {}
    keys = tuple(records[0])
    if any(tuple(record) != keys for record in records[1:]):
        raise ValueError("Sample records have inconsistent fields")
    return {
        key: np.asarray([record[key] for record in records])
        for key in keys
    }


def _mpb_worker(args):
    """Process-pool adapter that loads one two-snapshot MPB record."""

    base_path, subhalo_id, snap_prev, snap_cur, h = args
    try:
        result = load_two_snapshot_mpb(
            base_path,
            int(subhalo_id),
            snap_prev=snap_prev,
            snap_cur=snap_cur,
            h=h,
        )
        return result, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def select_two_snapshot_sample(
    base_path: str,
    config: Phase18Config,
    *,
    verbose: bool = True,
) -> tuple[dict[str, np.ndarray], dict]:
    """Select up to 50 z=0 centrals per Mstar bin and validate the MPB."""

    candidates = load_z0_central_candidates(base_path, config)
    edges = config.mass_bin_edges
    records: list[dict] = []
    rejected: list[dict] = []
    candidate_counts = []
    selected_counts = []

    serial = (
        config.mpb_parallel_backend == "serial"
        or config.mpb_workers == 1
    )
    executor = None
    if not serial:
        executor_class = (
            ThreadPoolExecutor
            if config.mpb_parallel_backend == "thread"
            else ProcessPoolExecutor
        )
        executor = executor_class(max_workers=config.mpb_workers)
    try:
        for bin_index, (low, high) in enumerate(
            zip(edges[:-1], edges[1:])
        ):
            positions = np.where(
                candidates["mass_bin_index"] == bin_index
            )[0]
            candidate_counts.append(int(len(positions)))
            if len(positions) < config.minimum_bin_warning_count:
                message = (
                    f"Mass bin [{low:.2f}, {high:.2f}) has only "
                    f"{len(positions)} candidate galaxies (<"
                    f"{config.minimum_bin_warning_count}); selecting all."
                )
                warnings.warn(message, RuntimeWarning)
                if verbose:
                    print(f"[WARNING] {message}")
            rng = np.random.default_rng(
                np.random.SeedSequence([config.random_seed, bin_index])
            )
            order = positions[rng.permutation(len(positions))]
            accepted = 0
            cursor = 0
            while (
                accepted < config.sample_per_bin
                and cursor < len(order)
            ):
                remaining = config.sample_per_bin - accepted
                batch_size = (
                    1
                    if serial
                    else max(2 * config.mpb_workers, remaining)
                )
                batch = order[cursor:cursor + batch_size]
                cursor += len(batch)
                resolved = {}
                if serial:
                    position = int(batch[0])
                    subhalo_id = int(
                        candidates["subhalo_id_z0"][position]
                    )
                    resolved[position] = _mpb_worker(
                        (
                            base_path,
                            subhalo_id,
                            config.snap_prev,
                            config.snap_cur,
                            config.h,
                        )
                    )
                else:
                    future_map = {}
                    for position in batch:
                        position = int(position)
                        subhalo_id = int(
                            candidates["subhalo_id_z0"][position]
                        )
                        future = executor.submit(
                            _mpb_worker,
                            (
                                base_path,
                                subhalo_id,
                                config.snap_prev,
                                config.snap_cur,
                                config.h,
                            ),
                        )
                        future_map[future] = position
                    for future in as_completed(future_map):
                        resolved[future_map[future]] = future.result()

                for position in batch:
                    if accepted >= config.sample_per_bin:
                        break
                    position = int(position)
                    subhalo_id = int(
                        candidates["subhalo_id_z0"][position]
                    )
                    mpb, error = resolved[position]
                    if error is not None:
                        rejected.append(
                            {
                                "subhalo_id_z0": subhalo_id,
                                "mass_bin_index": bin_index,
                                "reason": error,
                            }
                        )
                        continue
                    record = {
                        "sample_index": len(records),
                        "mass_bin_index": bin_index,
                        "mass_bin_low": float(low),
                        "mass_bin_high": float(high),
                        "mass_bin_center": float(0.5 * (low + high)),
                        "sample_rank_in_bin": accepted,
                        "group_id_z0": int(
                            candidates["group_id_z0"][position]
                        ),
                    "mstar_catalog_z0_msun": float(
                        candidates["mstar_z0_msun"][position]
                    ),
                    "log_mstar_z0": float(
                        candidates["log_mstar_z0"][position]
                    ),
                    "m200c_catalog_z0_msun": float(
                        candidates["m200c_z0_msun"][position]
                    ),
                        "log_m200c_z0": float(
                            candidates["log_m200c_z0"][position]
                        ),
                        **mpb,
                    }
                    records.append(record)
                    accepted += 1
                    if verbose:
                        print(
                            f"[sample bin {low:.2f},{high:.2f}] "
                            f"{accepted}/"
                            f"{min(config.sample_per_bin, len(order))} "
                            f"sub={subhalo_id}"
                        )
            selected_counts.append(accepted)
            if accepted < config.minimum_bin_warning_count:
                message = (
                    f"Mass bin [{low:.2f}, {high:.2f}) has only "
                    f"{accepted} valid snap98->99 MPB galaxies after "
                    f"validation (<{config.minimum_bin_warning_count})."
                )
                warnings.warn(message, RuntimeWarning)
                if verbose:
                    print(f"[WARNING] {message}")
    finally:
        if executor is not None:
            executor.shutdown()

    table = _records_to_table(records)
    metadata = {
        "base_path": str(base_path),
        "selection_axis": (
            "snapshot-99 SubhaloMassInRadType[:,4] "
            "(stars within twice the stellar half-mass radius)"
        ),
        "random_seed": config.random_seed,
        "mass_bin_edges": edges.tolist(),
        "candidate_counts": candidate_counts,
        "selected_counts": selected_counts,
        "rejected_mpb": rejected,
    }
    return table, metadata


def save_sample_catalog(
    table: dict[str, np.ndarray],
    metadata: dict,
    output_prefix: str | Path,
) -> tuple[Path, Path]:
    """Atomically persist the sample table and JSON metadata under one prefix."""

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    table_path = output_prefix.with_suffix(".npz")
    metadata_path = output_prefix.with_suffix(".json")
    temporary = table_path.with_suffix(".tmp.npz")
    np.savez(temporary, **table)
    temporary.replace(table_path)
    metadata_temporary = metadata_path.with_suffix(".tmp.json")
    metadata_temporary.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    metadata_temporary.replace(metadata_path)
    return table_path, metadata_path


def load_sample_catalog(
    output_prefix: str | Path,
) -> tuple[dict[str, np.ndarray], dict]:
    """Load cached sample arrays and their associated JSON metadata."""

    output_prefix = Path(output_prefix)
    table_path = output_prefix.with_suffix(".npz")
    metadata_path = output_prefix.with_suffix(".json")
    with np.load(table_path, allow_pickle=False) as data:
        table = {key: np.asarray(data[key]) for key in data.files}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return table, metadata
