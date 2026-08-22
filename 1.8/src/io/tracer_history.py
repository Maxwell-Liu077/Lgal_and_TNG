"""Shared tracer histories plus the optional inner-hot parent mask.

The complete origin/fate classification is retained, and the extra mask is a
separate deterministic step so inward-delivery measurements can be audited.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ..utils.config import Phase18Config
from .region_state import internal_energy_to_temperature, periodic_radius
from .tracer_batch import snapshot_chunk_paths


ORIGIN_NAMES = (
    "central_cold",
    "outer_hot",
    "central_hot",
    "outer_cold",
    "wind",
    "star",
    "satellite",
    "outside_r200c",
    "black_hole",
    "other",
)

FATE_NAMES = (
    "central_cold",
    "central_star",
    "halo_hot",
    "outer_cold",
    "wind",
    "outside_r200c",
    "satellite",
    "black_hole",
    "other",
)

_PARTICLE_QUERY_IDS: np.ndarray | None = None
_PARTICLE_BLOCK_SIZE: int | None = None


def _membership_positions(
    values: np.ndarray,
    sorted_targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return lookup positions and validity for values in sorted targets."""

    positions = np.searchsorted(sorted_targets, values)
    valid = positions < len(sorted_targets)
    if np.any(valid):
        indices = np.where(valid)[0]
        valid[indices] = (
            sorted_targets[positions[indices]] == values[indices]
        )
    return valid, positions


def _empty_records() -> dict[str, np.ndarray]:
    """Return the fixed empty schema used for unmatched parent queries."""

    return {
        "particle_ids": np.empty(0, dtype=np.uint64),
        "particle_type": np.empty(0, dtype=np.int8),
        "coordinates": np.empty((0, 3), dtype=float),
        "sfr": np.empty(0, dtype=float),
        "internal_energy": np.empty(0, dtype=float),
        "electron_abundance": np.empty(0, dtype=float),
        "formation_time": np.empty(0, dtype=float),
    }


def _scan_particle_chunk(
    path: str | Path,
    query_ids: np.ndarray,
    block_size: int,
) -> dict[str, np.ndarray]:
    """Read physical records for selected tracer ParentIDs."""

    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required for particle lookup") from exc

    parts = {key: [] for key in _empty_records()}
    specifications = (
        (
            "PartType0",
            0,
            (
                "ParticleIDs",
                "Coordinates",
                "StarFormationRate",
                "InternalEnergy",
                "ElectronAbundance",
            ),
        ),
        (
            "PartType4",
            4,
            (
                "ParticleIDs",
                "Coordinates",
                "GFM_StellarFormationTime",
            ),
        ),
        ("PartType5", 5, ("ParticleIDs",)),
    )
    with h5py.File(path, "r") as handle:
        for group_name, particle_type, required in specifications:
            if group_name not in handle:
                continue
            group = handle[group_name]
            missing = [field for field in required if field not in group]
            if missing:
                raise KeyError(
                    f"Missing {group_name} fields {missing} in {path}"
                )
            ids_dataset = group["ParticleIDs"]
            for start in range(0, len(ids_dataset), block_size):
                stop = min(start + block_size, len(ids_dataset))
                ids = np.asarray(
                    ids_dataset[start:stop],
                    dtype=np.uint64,
                )
                valid, _ = _membership_positions(ids, query_ids)
                if not np.any(valid):
                    continue
                count = int(np.count_nonzero(valid))
                selected_ids = ids[valid]
                coordinates = (
                    np.asarray(
                        group["Coordinates"][start:stop],
                        dtype=float,
                    )[valid]
                    if "Coordinates" in group
                    else np.full((count, 3), np.nan)
                )
                parts["particle_ids"].append(selected_ids)
                parts["particle_type"].append(
                    np.full(count, particle_type, dtype=np.int8)
                )
                parts["coordinates"].append(coordinates)
                if particle_type == 0:
                    parts["sfr"].append(
                        np.asarray(
                            group["StarFormationRate"][start:stop],
                            dtype=float,
                        )[valid]
                    )
                    parts["internal_energy"].append(
                        np.asarray(
                            group["InternalEnergy"][start:stop],
                            dtype=float,
                        )[valid]
                    )
                    parts["electron_abundance"].append(
                        np.asarray(
                            group["ElectronAbundance"][start:stop],
                            dtype=float,
                        )[valid]
                    )
                    parts["formation_time"].append(
                        np.full(count, np.nan)
                    )
                elif particle_type == 4:
                    parts["sfr"].append(np.full(count, np.nan))
                    parts["internal_energy"].append(
                        np.full(count, np.nan)
                    )
                    parts["electron_abundance"].append(
                        np.full(count, np.nan)
                    )
                    parts["formation_time"].append(
                        np.asarray(
                            group[
                                "GFM_StellarFormationTime"
                            ][start:stop],
                            dtype=float,
                        )[valid]
                    )
                else:
                    parts["sfr"].append(np.full(count, np.nan))
                    parts["internal_energy"].append(
                        np.full(count, np.nan)
                    )
                    parts["electron_abundance"].append(
                        np.full(count, np.nan)
                    )
                    parts["formation_time"].append(
                        np.full(count, np.nan)
                    )
    if not parts["particle_ids"]:
        return _empty_records()
    return {
        key: np.concatenate(values, axis=0)
        for key, values in parts.items()
    }


def _particle_worker_init(
    query_ids: np.ndarray,
    block_size: int,
) -> None:
    """Install sorted query IDs and block size in a process-pool worker."""

    global _PARTICLE_QUERY_IDS, _PARTICLE_BLOCK_SIZE
    _PARTICLE_QUERY_IDS = query_ids
    _PARTICLE_BLOCK_SIZE = block_size


def _particle_worker(path: str) -> dict[str, np.ndarray]:
    """Scan one particle chunk using the worker's immutable query catalog."""

    if _PARTICLE_QUERY_IDS is None or _PARTICLE_BLOCK_SIZE is None:
        raise RuntimeError("Particle worker was not initialized")
    return _scan_particle_chunk(
        path,
        _PARTICLE_QUERY_IDS,
        _PARTICLE_BLOCK_SIZE,
    )


def _records_fingerprint(
    query_ids: np.ndarray,
    snap_num: int,
) -> str:
    """Hash the sorted parent query and snapshot for particle-record caches."""

    digest = hashlib.sha256()
    digest.update(f"phase18-particle-records-v1-{snap_num}".encode())
    values = np.ascontiguousarray(query_ids, dtype=np.uint64)
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def _save_records(
    path: Path,
    records: dict[str, np.ndarray],
    fingerprint: str,
) -> None:
    """Atomically persist matched parent-particle records and fingerprint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(
        temporary,
        fingerprint=np.asarray(fingerprint),
        **records,
    )
    temporary.replace(path)


def _load_records(
    path: Path,
    fingerprint: str,
) -> dict[str, np.ndarray] | None:
    """Return cached records only when the stored fingerprint is current."""

    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(np.asarray(data["fingerprint"]).item()) != fingerprint:
                return None
            records = {
                key: np.asarray(data[key])
                for key in _empty_records()
            }
        lengths = {
            len(values)
            for values in records.values()
        }
        if len(lengths) != 1:
            return None
        return records
    except Exception:
        return None


def lookup_parent_particle_records(
    base_path: str | Path,
    snap_num: int,
    parent_ids: np.ndarray,
    *,
    cache_dir: str | Path | None,
    block_size: int,
    max_workers: int,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Look up gas/star/wind/BH records for selected ParentIDs."""

    query_ids = np.unique(np.asarray(parent_ids, dtype=np.uint64))
    if len(query_ids) == 0:
        return _empty_records()
    paths = snapshot_chunk_paths(base_path, snap_num)
    fingerprint = _records_fingerprint(query_ids, snap_num)
    cache_root = (
        Path(cache_dir)
        / f"particle_records_snap{snap_num:03d}_{fingerprint[:16]}"
        if cache_dir is not None
        else None
    )
    records_per_chunk: list[dict[str, np.ndarray]] = []
    pending: list[tuple[int, Path, Path | None]] = []
    for chunk_index, path in enumerate(paths):
        cache_path = (
            cache_root / f"chunk_{chunk_index:04d}.npz"
            if cache_root is not None
            else None
        )
        cached = (
            _load_records(cache_path, fingerprint)
            if cache_path is not None
            else None
        )
        if cached is None:
            pending.append((chunk_index, path, cache_path))
        else:
            records_per_chunk.append(cached)
    if verbose:
        print(
            f"[snap {snap_num} particle lookup] chunks={len(paths)}, "
            f"cached={len(paths) - len(pending)}, "
            f"pending={len(pending)}, workers={max_workers}"
        )
    if pending and max_workers == 1:
        for sequence, (_, path, cache_path) in enumerate(
            pending,
            start=1,
        ):
            records = _scan_particle_chunk(path, query_ids, block_size)
            records_per_chunk.append(records)
            if cache_path is not None:
                _save_records(cache_path, records, fingerprint)
            if verbose:
                print(
                    f"[snap {snap_num} particle lookup] "
                    f"{sequence}/{len(pending)}"
                )
    elif pending:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_particle_worker_init,
            initargs=(query_ids, block_size),
        ) as executor:
            future_map = {
                executor.submit(_particle_worker, str(path)): (
                    chunk_index,
                    cache_path,
                )
                for chunk_index, path, cache_path in pending
            }
            completed = 0
            for future in as_completed(future_map):
                chunk_index, cache_path = future_map[future]
                records = future.result()
                records_per_chunk.append(records)
                if cache_path is not None:
                    _save_records(cache_path, records, fingerprint)
                completed += 1
                if verbose:
                    print(
                        f"[snap {snap_num} particle lookup] "
                        f"{completed}/{len(pending)} "
                        f"(chunk index {chunk_index})"
                    )
    nonempty = [
        records
        for records in records_per_chunk
        if len(records["particle_ids"])
    ]
    if not nonempty:
        return _empty_records()
    merged = {
        key: np.concatenate([records[key] for records in nonempty], axis=0)
        for key in _empty_records()
    }
    order = np.argsort(merged["particle_ids"], kind="stable")
    merged = {key: values[order] for key, values in merged.items()}
    ids = merged["particle_ids"]
    if len(ids) > 1 and np.any(ids[1:] == ids[:-1]):
        raise ValueError("ParticleID occurs in multiple particle records")
    return merged


def _matched_record_positions(
    parent_ids: np.ndarray,
    records: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Locate parent IDs in sorted records and return positions plus a mask."""

    record_ids = np.asarray(records["particle_ids"], dtype=np.uint64)
    positions = np.searchsorted(record_ids, parent_ids)
    matched = positions < len(record_ids)
    if np.any(matched):
        indices = np.where(matched)[0]
        matched[indices] = (
            record_ids[positions[indices]] == parent_ids[indices]
        )
    return matched, positions


def classify_parent_states(
    parent_ids: np.ndarray,
    records: dict[str, np.ndarray],
    *,
    branch_state: dict,
    group_center_ckpc_h: np.ndarray,
    subhalo_center_ckpc_h: np.ndarray,
    r200c_ckpc_h: float,
    box_size_ckpc_h: float,
    config: Phase18Config,
    direction: str,
) -> np.ndarray:
    """Classify one ParentID per tracer into an exclusive endpoint state."""

    if direction == "origin":
        names = ORIGIN_NAMES
    elif direction == "fate":
        names = FATE_NAMES
    else:
        raise ValueError("direction must be origin or fate")
    codes = {name: index for index, name in enumerate(names)}
    parent_ids = np.asarray(parent_ids, dtype=np.uint64)
    output = np.full(len(parent_ids), codes["other"], dtype=np.int16)
    matched, positions = _matched_record_positions(parent_ids, records)
    if not np.any(matched):
        return output
    selected = np.where(matched)[0]
    record_position = positions[selected]
    particle_type = np.asarray(
        records["particle_type"][record_position],
        dtype=np.int8,
    )
    selected_ids = parent_ids[selected]

    black_hole = particle_type == 5
    output[selected[black_hole]] = codes["black_hole"]

    type4 = particle_type == 4
    if np.any(type4):
        type4_selected = selected[type4]
        type4_positions = record_position[type4]
        formation_time = np.asarray(
            records["formation_time"][type4_positions],
            dtype=float,
        )
        wind = np.isfinite(formation_time) & (formation_time <= 0)
        output[type4_selected[wind]] = codes["wind"]
        real_star = np.isfinite(formation_time) & (formation_time > 0)
        if direction == "origin":
            output[type4_selected[real_star]] = codes["star"]
        else:
            star_ids = selected_ids[type4][real_star]
            star_positions = type4_positions[real_star]
            central_membership = np.isin(
                star_ids,
                np.asarray(
                    branch_state["central_star_ids_all"],
                    dtype=np.uint64,
                ),
            )
            satellite_membership = np.isin(
                star_ids,
                np.asarray(
                    branch_state["satellite_star_ids"],
                    dtype=np.uint64,
                ),
            )
            radius = periodic_radius(
                records["coordinates"][star_positions],
                subhalo_center_ckpc_h,
                box_size_ckpc_h,
            )
            central = (
                central_membership
                & (radius < config.central_rfrac * r200c_ckpc_h)
            )
            real_selected = type4_selected[real_star]
            output[real_selected[central]] = codes["central_star"]
            output[
                real_selected[satellite_membership & ~central]
            ] = codes["satellite"]

    gas = particle_type == 0
    if not np.any(gas):
        return output
    gas_selected = selected[gas]
    gas_positions = record_position[gas]
    gas_ids = selected_ids[gas]
    satellite = np.isin(
        gas_ids,
        np.asarray(branch_state["satellite_gas_ids"], dtype=np.uint64),
    )
    output[gas_selected[satellite]] = codes["satellite"]
    radius_halo = periodic_radius(
        records["coordinates"][gas_positions],
        group_center_ckpc_h,
        box_size_ckpc_h,
    )
    radius_central = periodic_radius(
        records["coordinates"][gas_positions],
        subhalo_center_ckpc_h,
        box_size_ckpc_h,
    )
    outside = (
        ~satellite
        & (
            radius_halo
            >= config.outer_rmax_rfrac * r200c_ckpc_h
        )
    )
    output[gas_selected[outside]] = codes["outside_r200c"]
    diffuse_inside = ~satellite & ~outside
    temperature = internal_energy_to_temperature(
        records["internal_energy"][gas_positions],
        records["electron_abundance"][gas_positions],
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        log_temperature = np.log10(temperature)
    sfr = np.asarray(records["sfr"][gas_positions], dtype=float)
    cold = (sfr > 0) | (
        (sfr <= 0)
        & np.isfinite(log_temperature)
        & (log_temperature < config.cold_log10_temperature_max)
    )
    hot = (
        (sfr <= 0)
        & np.isfinite(log_temperature)
        & (log_temperature >= config.hot_log10_temperature_min)
    )
    central = (
        diffuse_inside
        & (radius_central < config.central_rfrac * r200c_ckpc_h)
    )
    outer = diffuse_inside & ~central
    if direction == "origin":
        output[gas_selected[central & cold]] = codes["central_cold"]
        output[gas_selected[central & hot]] = codes["central_hot"]
        output[gas_selected[outer & cold]] = codes["outer_cold"]
        output[gas_selected[outer & hot]] = codes["outer_hot"]
    else:
        output[gas_selected[central & cold]] = codes["central_cold"]
        output[gas_selected[outer & cold]] = codes["outer_cold"]
        output[gas_selected[diffuse_inside & hot]] = codes["halo_hot"]
    return output


def inner_hot_parent_mask(
    parent_ids: np.ndarray,
    records: dict[str, np.ndarray],
    *,
    branch_state: dict,
    group_center_ckpc_h: np.ndarray,
    subhalo_center_ckpc_h: np.ndarray,
    r200c_ckpc_h: float,
    box_size_ckpc_h: float,
    config: Phase18Config,
) -> np.ndarray:
    """Identify diffuse hot-gas parents inside 0.1 R200c."""

    parent_ids = np.asarray(parent_ids, dtype=np.uint64)
    output = np.zeros(len(parent_ids), dtype=bool)
    matched, positions = _matched_record_positions(parent_ids, records)
    if not np.any(matched):
        return output
    selected = np.where(matched)[0]
    record_positions = positions[selected]
    particle_type = np.asarray(
        records["particle_type"][record_positions],
        dtype=np.int8,
    )
    gas = particle_type == 0
    if not np.any(gas):
        return output
    gas_selected = selected[gas]
    gas_positions = record_positions[gas]
    gas_ids = parent_ids[gas_selected]
    satellite = np.isin(
        gas_ids,
        np.asarray(branch_state["satellite_gas_ids"], dtype=np.uint64),
    )
    radius_halo = periodic_radius(
        records["coordinates"][gas_positions],
        group_center_ckpc_h,
        box_size_ckpc_h,
    )
    radius_central = periodic_radius(
        records["coordinates"][gas_positions],
        subhalo_center_ckpc_h,
        box_size_ckpc_h,
    )
    temperature = internal_energy_to_temperature(
        records["internal_energy"][gas_positions],
        records["electron_abundance"][gas_positions],
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        log_temperature = np.log10(temperature)
    sfr = np.asarray(records["sfr"][gas_positions], dtype=float)
    inner_hot = (
        ~satellite
        & (radius_halo < config.outer_rmax_rfrac * r200c_ckpc_h)
        & (radius_central < config.central_rfrac * r200c_ckpc_h)
        & (sfr <= 0)
        & np.isfinite(log_temperature)
        & (log_temperature >= config.hot_log10_temperature_min)
    )
    output[gas_selected[inner_hot]] = True
    return output


def history_record(
    *,
    origin_codes: np.ndarray,
    fate_codes: np.ndarray,
    n_origin_expected: int,
    n_fate_expected: int,
    tracer_mass_msun: float,
    dt_gyr: float,
) -> dict:
    """Return complete count/fraction records for both history directions."""

    if len(origin_codes) > n_origin_expected:
        raise ValueError("More origin records than target tracers")
    if len(fate_codes) > n_fate_expected:
        raise ValueError("More fate records than source tracers")
    output: dict[str, float | int] = {
        "n_origin_denominator": int(n_origin_expected),
        "n_fate_denominator": int(n_fate_expected),
    }
    for prefix, codes_array, names, expected in (
        ("origin", origin_codes, ORIGIN_NAMES, n_origin_expected),
        ("fate", fate_codes, FATE_NAMES, n_fate_expected),
    ):
        codes_array = np.asarray(codes_array, dtype=int)
        counts = np.bincount(codes_array, minlength=len(names)).astype(
            np.int64
        )
        missing = int(expected - len(codes_array))
        counts[names.index("other")] += missing
        if int(counts.sum()) != int(expected):
            raise RuntimeError(f"{prefix} history does not close")
        for index, name in enumerate(names):
            count = int(counts[index])
            output[f"n_{prefix}_{name}"] = count
            output[f"fraction_{prefix}_{name}"] = (
                count / expected if expected else np.nan
            )
        output[f"fraction_{prefix}_closure"] = (
            float(counts.sum() / expected) if expected else np.nan
        )
    hot_count = int(
        output["n_fate_halo_hot"]
    )
    output.update(
        {
            "n_tng_cold_to_hot_transition": hot_count,
            "mass_tng_cold_to_hot_transition_msun": (
                hot_count * tracer_mass_msun
            ),
            "rate_tng_cold_to_hot_transition_msun_per_yr": (
                hot_count
                * tracer_mass_msun
                / (dt_gyr * 1.0e9)
            ),
            "fraction_tng_cold_to_hot_transition": (
                hot_count / n_fate_expected
                if n_fate_expected
                else np.nan
            ),
        }
    )
    return output


def compute_stacked_history_statistics(
    halo_results: dict[str, np.ndarray],
    config: Phase18Config,
) -> dict[str, np.ndarray]:
    """Compute tracer-count-weighted fractions in each Mstar bin."""

    bin_index = np.asarray(halo_results["mass_bin_index"], dtype=int)
    centers = config.mass_bin_centers
    output: dict[str, np.ndarray] = {
        "mass_bin_center": centers,
        "mass_bin_low": config.mass_bin_edges[:-1],
        "mass_bin_high": config.mass_bin_edges[1:],
    }
    for prefix, names in (
        ("origin", ORIGIN_NAMES),
        ("fate", FATE_NAMES),
    ):
        denominator = np.asarray(
            halo_results[f"n_{prefix}_denominator"],
            dtype=np.int64,
        )
        fractions = np.full((len(names), len(centers)), np.nan)
        denominator_sum = np.zeros(len(centers), dtype=np.int64)
        for mass_bin in range(len(centers)):
            selected = bin_index == mass_bin
            total = int(denominator[selected].sum())
            denominator_sum[mass_bin] = total
            if total == 0:
                continue
            for state_index, name in enumerate(names):
                count = np.asarray(
                    halo_results[f"n_{prefix}_{name}"],
                    dtype=np.int64,
                )
                fractions[state_index, mass_bin] = (
                    count[selected].sum() / total
                )
        output[f"{prefix}_state_names"] = np.asarray(names)
        output[f"{prefix}_fractions"] = fractions
        output[f"{prefix}_denominator_sum"] = denominator_sum
        closure = np.nansum(fractions, axis=0)
        closure[denominator_sum == 0] = np.nan
        output[f"{prefix}_closure"] = closure
    return output
