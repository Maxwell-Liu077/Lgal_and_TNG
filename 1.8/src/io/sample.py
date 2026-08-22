"""Sample-cache and Subfind-membership workflow for Phase 1.8."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..utils.config import Phase18Config
from .region_state import (
    load_subfind_membership_catalog,
    subfind_membership_counts,
)
from .sampling import (
    load_sample_catalog,
    save_sample_catalog,
    select_two_snapshot_sample,
)
from ..utils.arrays import _row


def _sample_cache_tag(config: Phase18Config) -> str:
    """Return the versioned sample tag used by existing Phase 1.8 caches."""

    path_tag = hashlib.sha256(
        str(Path(config.base_path)).encode("utf-8")
    ).hexdigest()[:8]
    return (
        f"sample_v4_mstar2rh_path{path_tag}_s{config.random_seed}"
        f"_m{config.log_mstar_min:.2f}-{config.log_mstar_max:.2f}"
        f"_d{config.mass_bin_width:.2f}"
        f"_n{config.sample_per_bin}"
        f"_snap{config.snap_prev:03d}-{config.snap_cur:03d}"
    ).replace(".", "p")


def load_or_build_sample(
    config: Phase18Config,
    *,
    cache_dir: str | Path,
    rebuild: bool = False,
    verbose: bool = True,
) -> tuple[dict[str, np.ndarray], dict]:
    """Load a valid sample cache or build it from the group catalogue."""

    cache_dir = Path(cache_dir)
    prefix = cache_dir / "samples" / _sample_cache_tag(config)
    if (
        not rebuild
        and prefix.with_suffix(".npz").is_file()
        and prefix.with_suffix(".json").is_file()
    ):
        try:
            if verbose:
                print(f"loading sample cache: {prefix}")
            return load_sample_catalog(prefix)
        except Exception as exc:
            if verbose:
                print(
                    "sample cache is incomplete or invalid; rebuilding: "
                    f"{type(exc).__name__}: {exc}"
                )
    table, metadata = select_two_snapshot_sample(
        config.base_path,
        config,
        verbose=verbose,
    )
    save_sample_catalog(table, metadata, prefix)
    return table, metadata


def attach_subfind_membership(
    sample_table: dict[str, np.ndarray],
    *,
    config: Phase18Config,
    verbose: bool = True,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Attach central/satellite Subfind counts at both endpoints."""

    previous_catalog = load_subfind_membership_catalog(
        config.base_path,
        config.snap_prev,
    )
    current_catalog = load_subfind_membership_catalog(
        config.base_path,
        config.snap_cur,
    )
    membership_rows: list[dict] = []
    valid_indices = []
    rejected = []
    for index in range(len(sample_table["sample_index"])):
        row = _row(sample_table, index)
        try:
            previous = subfind_membership_counts(
                previous_catalog,
                group_id=int(row["group_id_prev"]),
                central_subfind_id=int(row["subfind_id_prev"]),
            )
            current = subfind_membership_counts(
                current_catalog,
                group_id=int(row["group_id_cur"]),
                central_subfind_id=int(row["subfind_id_cur"]),
            )
        except Exception as exc:
            rejected.append(
                {
                    "subhalo_id_z0": int(row["subhalo_id_z0"]),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        membership_rows.append(
            {
                **{
                    f"{key}_prev": value
                    for key, value in previous.items()
                },
                **{
                    f"{key}_cur": value
                    for key, value in current.items()
                },
            }
        )
        valid_indices.append(index)
    valid_indices = np.asarray(valid_indices, dtype=int)
    output = {
        key: np.asarray(values)[valid_indices]
        for key, values in sample_table.items()
    }
    if membership_rows:
        for key in membership_rows[0]:
            output[key] = np.asarray(
                [row[key] for row in membership_rows],
                dtype=np.int64,
            )
    if verbose:
        print(
            "[membership] "
            f"valid={len(valid_indices)}, rejected={len(rejected)}",
            flush=True,
        )
    return output, rejected

