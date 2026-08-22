"""Chunked full-snapshot tracer scans with resumable atomic caches.

This low-level data-access module preserves cache fingerprints, chunk file
names, and process-backend behavior.  It contains no plotting or scientific
aggregation logic.
"""

from __future__ import annotations

import hashlib
import re
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from pathlib import Path

import numpy as np


HOT = 0
OUTER_COLD = 1
CENTRAL_COLD = 2
CENTRAL_STAR = 3
N_STATES = 4
STATE_NAMES = (
    "hot",
    "outer_cold",
    "central_cold",
    "central_star",
)

_WORKER_CATALOG: dict[str, np.ndarray] | None = None
_WORKER_BLOCK_SIZE: int | None = None
_WORKER_SCAN_KIND: str | None = None


def _chunk_number(path: Path) -> int:
    """Extract the numeric chunk suffix used to preserve snapshot order."""

    match = re.search(r"\.(\d+)\.hdf5$", path.name)
    return int(match.group(1)) if match else 0


def snapshot_chunk_paths(
    base_path: str | Path,
    snap_num: int,
) -> list[Path]:
    """Return numerically ordered HDF5 chunks for one full snapshot."""

    base_path = Path(base_path)
    tag = f"{int(snap_num):03d}"
    snap_dir = base_path / f"snapdir_{tag}"
    if snap_dir.is_dir():
        paths = list(snap_dir.glob(f"snap_{tag}.*.hdf5"))
    else:
        paths = list(base_path.glob(f"snap_{tag}.*.hdf5"))
        single = base_path / f"snap_{tag}.hdf5"
        if single.is_file():
            paths.append(single)
    paths = sorted(set(paths), key=_chunk_number)
    if not paths:
        raise FileNotFoundError(
            f"No snapshot chunks found for snap={snap_num} in {base_path}"
        )
    return paths


def _build_lookup_catalog(
    id_parts: list[np.ndarray],
    label_parts: list[np.ndarray],
) -> dict[str, np.ndarray]:
    """Build the sorted lookup representation shared by both scan directions."""

    parent_parts = [
        np.asarray(values, dtype=np.uint64)
        for values in id_parts
        if len(values)
    ]
    label_parts = [
        np.asarray(values, dtype=np.int32)
        for values in label_parts
        if len(values)
    ]
    if not parent_parts:
        return {
            "parent_ids": np.empty(0, dtype=np.uint64),
            "primary_labels": np.empty(0, dtype=np.int32),
            "duplicate_unique_indices": np.empty(0, dtype=np.int64),
            "duplicate_offsets": np.zeros(1, dtype=np.int64),
            "duplicate_extra_labels": np.empty(0, dtype=np.int32),
        }

    parents = np.concatenate(parent_parts)
    labels = np.concatenate(label_parts)
    order = np.argsort(parents, kind="stable")
    sorted_parents = parents[order]
    sorted_labels = labels[order]
    unique_parents, first, counts = np.unique(
        sorted_parents,
        return_index=True,
        return_counts=True,
    )
    primary_labels = sorted_labels[first]

    primary_entry = np.zeros(len(sorted_parents), dtype=bool)
    primary_entry[first] = True
    extra_labels = sorted_labels[~primary_entry]
    extra_parent_ids = sorted_parents[~primary_entry]
    if len(extra_parent_ids):
        extra_unique_indices = np.searchsorted(
            unique_parents,
            extra_parent_ids,
        ).astype(np.int64)
        duplicate_unique_indices, extra_counts = np.unique(
            extra_unique_indices,
            return_counts=True,
        )
        duplicate_offsets = np.concatenate(
            [
                np.zeros(1, dtype=np.int64),
                np.cumsum(extra_counts, dtype=np.int64),
            ]
        )
    else:
        duplicate_unique_indices = np.empty(0, dtype=np.int64)
        duplicate_offsets = np.zeros(1, dtype=np.int64)

    if not np.all(counts >= 1):
        raise RuntimeError("Invalid parent catalog")
    return {
        "parent_ids": unique_parents,
        "primary_labels": primary_labels.astype(np.int32, copy=False),
        "duplicate_unique_indices": duplicate_unique_indices,
        "duplicate_offsets": duplicate_offsets,
        "duplicate_extra_labels": extra_labels.astype(
            np.int32,
            copy=False,
        ),
    }


def build_parent_catalog(
    states: list[dict],
    *,
    role: str,
) -> dict[str, np.ndarray]:
    """Build one vectorized ParentID->halo/state lookup."""

    if role == "source":
        specifications = (
            ("hot_ids", HOT),
            ("outer_cold_ids", OUTER_COLD),
        )
        branch = "source"
    elif role == "target":
        specifications = (
            ("central_cold_ids", CENTRAL_COLD),
            ("central_star_ids", CENTRAL_STAR),
        )
        branch = "target"
    else:
        raise ValueError("role must be source or target")

    parent_parts = []
    label_parts = []
    for halo_index, pair_state in enumerate(states):
        state = pair_state[branch]
        for key, state_code in specifications:
            ids = np.asarray(state[key], dtype=np.uint64)
            if len(ids) == 0:
                continue
            parent_parts.append(ids)
            label_parts.append(
                np.full(
                    len(ids),
                    halo_index * N_STATES + state_code,
                    dtype=np.int32,
                )
            )
    return _build_lookup_catalog(parent_parts, label_parts)


def build_tracer_id_catalog(
    tracer_ids_by_halo: list[np.ndarray],
) -> dict[str, np.ndarray]:
    """Build a TracerID lookup with one output label per selected halo."""

    id_parts = []
    label_parts = []
    for halo_index, tracer_ids in enumerate(tracer_ids_by_halo):
        ids = np.unique(np.asarray(tracer_ids, dtype=np.uint64))
        if len(ids) == 0:
            continue
        id_parts.append(ids)
        label_parts.append(
            np.full(len(ids), halo_index, dtype=np.int32)
        )
    return _build_lookup_catalog(id_parts, label_parts)


def build_grouped_id_catalog(
    ids_by_group: list[np.ndarray],
) -> dict[str, np.ndarray]:
    """Build a generic ID lookup with one output label per group."""

    return build_tracer_id_catalog(ids_by_group)


def parent_catalog_fingerprint(
    catalog: dict[str, np.ndarray],
    *,
    snap_num: int,
    purpose: str = "parent_to_tracer",
) -> str:
    """Hash the exact lookup catalog, snapshot, and scan purpose for caching."""

    digest = hashlib.sha256()
    digest.update(
        f"phase18-snap{snap_num}-{purpose}-v2".encode("ascii")
    )
    for key in (
        "parent_ids",
        "primary_labels",
        "duplicate_unique_indices",
        "duplicate_offsets",
        "duplicate_extra_labels",
    ):
        values = np.ascontiguousarray(catalog[key])
        digest.update(key.encode("ascii"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def _append_grouped(
    selected: dict[int, list[np.ndarray]],
    labels: np.ndarray,
    tracer_ids: np.ndarray,
) -> None:
    """Append one block of tracer IDs to every matched output label."""

    if len(labels) == 0:
        return
    order = np.argsort(labels, kind="stable")
    labels = labels[order]
    tracer_ids = tracer_ids[order]
    unique_labels, first = np.unique(labels, return_index=True)
    stops = np.concatenate([first[1:], [len(labels)]])
    for label, start, stop in zip(unique_labels, first, stops):
        selected.setdefault(int(label), []).append(
            tracer_ids[int(start):int(stop)]
        )


def _membership_positions(
    values: np.ndarray,
    sorted_targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return lookup positions and validity for values in sorted targets."""

    positions = np.searchsorted(sorted_targets, values)
    valid = positions < len(sorted_targets)
    if np.any(valid):
        valid_indices = np.where(valid)[0]
        valid[valid_indices] = (
            sorted_targets[positions[valid_indices]]
            == values[valid_indices]
        )
    return valid, positions


def _scan_one_chunk(
    path: str | Path,
    catalog: dict[str, np.ndarray],
    block_size: int,
) -> dict[int, np.ndarray]:
    """Scan one HDF5 chunk for parent-to-tracer matches in bounded blocks."""

    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required to scan PartType3") from exc

    selected: dict[int, list[np.ndarray]] = {}
    parent_lookup = catalog["parent_ids"]
    if len(parent_lookup) == 0:
        return {}

    with h5py.File(path, "r") as handle:
        if "PartType3" not in handle:
            return {}
        group = handle["PartType3"]
        if "ParentID" not in group or "TracerID" not in group:
            raise KeyError(f"Missing PartType3 IDs in {path}")
        parent_dataset = group["ParentID"]
        tracer_dataset = group["TracerID"]
        if len(parent_dataset) != len(tracer_dataset):
            raise ValueError(f"PartType3 lengths differ in {path}")

        for start in range(0, len(parent_dataset), block_size):
            stop = min(start + block_size, len(parent_dataset))
            parents = np.asarray(
                parent_dataset[start:stop],
                dtype=np.uint64,
            )
            valid, positions = _membership_positions(
                parents,
                parent_lookup,
            )
            if not np.any(valid):
                continue
            tracer_ids = np.asarray(
                tracer_dataset[start:stop],
                dtype=np.uint64,
            )[valid]
            unique_indices = positions[valid].astype(
                np.int64,
                copy=False,
            )
            primary_labels = catalog["primary_labels"][unique_indices]
            _append_grouped(selected, primary_labels, tracer_ids)

            duplicate_indices = catalog["duplicate_unique_indices"]
            if len(duplicate_indices) == 0:
                continue
            duplicate_position = np.searchsorted(
                duplicate_indices,
                unique_indices,
            )
            is_duplicate = duplicate_position < len(duplicate_indices)
            if np.any(is_duplicate):
                candidate = np.where(is_duplicate)[0]
                is_duplicate[candidate] = (
                    duplicate_indices[duplicate_position[candidate]]
                    == unique_indices[candidate]
                )
            if not np.any(is_duplicate):
                continue

            duplicate_position = duplicate_position[is_duplicate]
            duplicate_tracers = tracer_ids[is_duplicate]
            offsets = catalog["duplicate_offsets"]
            counts = (
                offsets[duplicate_position + 1]
                - offsets[duplicate_position]
            )
            repeated_tracers = np.repeat(duplicate_tracers, counts)
            total = int(counts.sum())
            repeated_group_starts = np.repeat(
                np.cumsum(counts) - counts,
                counts,
            )
            label_positions = (
                np.repeat(offsets[duplicate_position], counts)
                + np.arange(total)
                - repeated_group_starts
            )
            extra_labels = catalog["duplicate_extra_labels"][
                label_positions
            ]
            _append_grouped(
                selected,
                extra_labels,
                repeated_tracers,
            )

    return {
        label: np.concatenate(parts)
        for label, parts in selected.items()
    }


def _scan_tracer_parents_one_chunk(
    path: str | Path,
    catalog: dict[str, np.ndarray],
    block_size: int,
) -> dict[int, np.ndarray]:
    """Map selected TracerIDs to their ParentIDs in one snapshot chunk."""

    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required to scan PartType3") from exc

    selected: dict[int, list[np.ndarray]] = {}
    tracer_lookup = catalog["parent_ids"]
    if len(tracer_lookup) == 0:
        return {}
    with h5py.File(path, "r") as handle:
        if "PartType3" not in handle:
            return {}
        group = handle["PartType3"]
        if "ParentID" not in group or "TracerID" not in group:
            raise KeyError(f"Missing PartType3 IDs in {path}")
        tracer_dataset = group["TracerID"]
        parent_dataset = group["ParentID"]
        if len(tracer_dataset) != len(parent_dataset):
            raise ValueError(f"PartType3 lengths differ in {path}")

        for start in range(0, len(tracer_dataset), block_size):
            stop = min(start + block_size, len(tracer_dataset))
            tracer_ids = np.asarray(
                tracer_dataset[start:stop],
                dtype=np.uint64,
            )
            valid, positions = _membership_positions(
                tracer_ids,
                tracer_lookup,
            )
            if not np.any(valid):
                continue
            parent_ids = np.asarray(
                parent_dataset[start:stop],
                dtype=np.uint64,
            )[valid]
            unique_indices = positions[valid].astype(
                np.int64,
                copy=False,
            )
            primary_labels = catalog["primary_labels"][unique_indices]
            _append_grouped(selected, primary_labels, parent_ids)

            duplicate_indices = catalog["duplicate_unique_indices"]
            if len(duplicate_indices) == 0:
                continue
            duplicate_position = np.searchsorted(
                duplicate_indices,
                unique_indices,
            )
            is_duplicate = duplicate_position < len(duplicate_indices)
            if np.any(is_duplicate):
                candidate = np.where(is_duplicate)[0]
                is_duplicate[candidate] = (
                    duplicate_indices[duplicate_position[candidate]]
                    == unique_indices[candidate]
                )
            if not np.any(is_duplicate):
                continue
            duplicate_position = duplicate_position[is_duplicate]
            duplicate_parents = parent_ids[is_duplicate]
            offsets = catalog["duplicate_offsets"]
            counts = (
                offsets[duplicate_position + 1]
                - offsets[duplicate_position]
            )
            repeated_parents = np.repeat(duplicate_parents, counts)
            total = int(counts.sum())
            repeated_group_starts = np.repeat(
                np.cumsum(counts) - counts,
                counts,
            )
            label_positions = (
                np.repeat(offsets[duplicate_position], counts)
                + np.arange(total)
                - repeated_group_starts
            )
            extra_labels = catalog["duplicate_extra_labels"][
                label_positions
            ]
            _append_grouped(
                selected,
                extra_labels,
                repeated_parents,
            )
    return {
        label: np.concatenate(parts)
        for label, parts in selected.items()
    }


def _initialize_worker(
    catalog: dict[str, np.ndarray],
    block_size: int,
    scan_kind: str = "parent_to_tracer",
) -> None:
    """Install immutable lookup data once in each process-pool worker."""

    global _WORKER_CATALOG, _WORKER_BLOCK_SIZE, _WORKER_SCAN_KIND
    _WORKER_CATALOG = catalog
    _WORKER_BLOCK_SIZE = block_size
    _WORKER_SCAN_KIND = scan_kind


def _process_worker(path: str) -> dict[int, np.ndarray]:
    """Dispatch one chunk to the scan kind configured by the initializer."""

    if _WORKER_CATALOG is None or _WORKER_BLOCK_SIZE is None:
        raise RuntimeError("Tracer worker was not initialized")
    if _WORKER_SCAN_KIND == "tracer_to_parent":
        return _scan_tracer_parents_one_chunk(
            path,
            _WORKER_CATALOG,
            _WORKER_BLOCK_SIZE,
        )
    return _scan_one_chunk(path, _WORKER_CATALOG, _WORKER_BLOCK_SIZE)


def _flatten_result(
    result: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten label-keyed arrays into a compact offsets representation."""

    labels = np.asarray(sorted(result), dtype=np.int32)
    counts = np.asarray(
        [len(result[int(label)]) for label in labels],
        dtype=np.int64,
    )
    offsets = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(counts)]
    )
    tracer_ids = (
        np.concatenate([result[int(label)] for label in labels])
        if len(labels)
        else np.empty(0, dtype=np.uint64)
    )
    return labels, offsets, tracer_ids


def _unflatten_result(
    labels: np.ndarray,
    offsets: np.ndarray,
    tracer_ids: np.ndarray,
) -> dict[int, np.ndarray]:
    """Reconstruct label-keyed arrays from the compact cache representation."""

    return {
        int(label): np.asarray(
            tracer_ids[offsets[index]:offsets[index + 1]],
            dtype=np.uint64,
        )
        for index, label in enumerate(labels)
    }


def _save_chunk_cache(
    path: Path,
    result: dict[int, np.ndarray],
    fingerprint: str,
) -> None:
    """Atomically save one flattened chunk result with its fingerprint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    labels, offsets, tracer_ids = _flatten_result(result)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(
        temporary,
        fingerprint=np.asarray(fingerprint),
        labels=labels,
        offsets=offsets,
        tracer_ids=tracer_ids,
    )
    temporary.replace(path)


def _load_chunk_cache(
    path: Path,
    fingerprint: str,
) -> dict[int, np.ndarray] | None:
    """Load a chunk only when its fingerprint and array schema are valid."""

    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            stored = str(np.asarray(data["fingerprint"]).item())
            if stored != fingerprint:
                return None
            labels = np.asarray(data["labels"], dtype=np.int32)
            offsets = np.asarray(data["offsets"], dtype=np.int64)
            tracer_ids = np.asarray(data["tracer_ids"], dtype=np.uint64)
        if len(offsets) != len(labels) + 1:
            return None
        if offsets[0] != 0 or offsets[-1] != len(tracer_ids):
            return None
        return _unflatten_result(labels, offsets, tracer_ids)
    except Exception:
        return None


def _merge_chunk_results(
    results: list[dict[int, np.ndarray]],
) -> dict[int, np.ndarray]:
    """Concatenate per-chunk matches without changing label or ID order."""

    merged: dict[int, list[np.ndarray]] = {}
    for result in results:
        for label, tracer_ids in result.items():
            if len(tracer_ids):
                merged.setdefault(label, []).append(tracer_ids)
    return {
        label: np.unique(np.concatenate(parts))
        for label, parts in merged.items()
    }


def scan_snapshot_for_catalog(
    base_path: str | Path,
    snap_num: int,
    catalog: dict[str, np.ndarray],
    *,
    cache_dir: str | Path | None = None,
    block_size: int = 2_000_000,
    max_workers: int = 8,
    parallel_backend: str = "process",
    verbose: bool = True,
) -> dict[int, np.ndarray]:
    """Scan one full PartType3 snapshot once for every selected halo."""

    if block_size < 1 or max_workers < 1:
        raise ValueError("block_size and max_workers must be positive")
    if parallel_backend not in {"serial", "thread", "process"}:
        raise ValueError("Unknown parallel backend")
    paths = snapshot_chunk_paths(base_path, snap_num)
    fingerprint = parent_catalog_fingerprint(
        catalog,
        snap_num=snap_num,
        purpose="parent_to_tracer",
    )
    cache_root = (
        Path(cache_dir) / f"snap{snap_num:03d}_{fingerprint[:16]}"
        if cache_dir is not None
        else None
    )

    results: list[dict[int, np.ndarray]] = []
    pending: list[tuple[int, Path, Path | None]] = []
    for chunk_index, path in enumerate(paths):
        cache_path = (
            cache_root / f"chunk_{chunk_index:04d}.npz"
            if cache_root is not None
            else None
        )
        cached = (
            _load_chunk_cache(cache_path, fingerprint)
            if cache_path is not None
            else None
        )
        if cached is not None:
            results.append(cached)
            if verbose:
                print(
                    f"[snap {snap_num}] cached chunk "
                    f"{chunk_index + 1}/{len(paths)}"
                )
        else:
            pending.append((chunk_index, path, cache_path))

    if verbose:
        print(
            f"[snap {snap_num}] chunks={len(paths)}, "
            f"cached={len(paths) - len(pending)}, "
            f"pending={len(pending)}, workers={max_workers}"
        )
    if not pending:
        return _merge_chunk_results(results)

    serial = parallel_backend == "serial" or max_workers == 1
    if serial:
        for sequence, (_, path, cache_path) in enumerate(
            pending,
            start=1,
        ):
            result = _scan_one_chunk(path, catalog, block_size)
            results.append(result)
            if cache_path is not None:
                _save_chunk_cache(cache_path, result, fingerprint)
            if verbose:
                print(
                    f"[snap {snap_num}] scanned pending chunk "
                    f"{sequence}/{len(pending)}"
                )
        return _merge_chunk_results(results)

    executor_class = (
        ThreadPoolExecutor
        if parallel_backend == "thread"
        else ProcessPoolExecutor
    )
    executor_kwargs = {"max_workers": max_workers}
    if parallel_backend == "process":
        executor_kwargs.update(
            {
                "initializer": _initialize_worker,
                "initargs": (catalog, block_size, "parent_to_tracer"),
            }
        )
    with executor_class(**executor_kwargs) as executor:
        future_map = {}
        for chunk_index, path, cache_path in pending:
            if parallel_backend == "thread":
                future = executor.submit(
                    _scan_one_chunk,
                    path,
                    catalog,
                    block_size,
                )
            else:
                future = executor.submit(_process_worker, str(path))
            future_map[future] = (chunk_index, cache_path)

        completed = 0
        for future in as_completed(future_map):
            chunk_index, cache_path = future_map[future]
            result = future.result()
            results.append(result)
            if cache_path is not None:
                _save_chunk_cache(cache_path, result, fingerprint)
            completed += 1
            if verbose:
                print(
                    f"[snap {snap_num}] completed pending chunk "
                    f"{completed}/{len(pending)} "
                    f"(chunk index {chunk_index})"
                )
    return _merge_chunk_results(results)


def scan_snapshot_for_tracer_parents(
    base_path: str | Path,
    snap_num: int,
    catalog: dict[str, np.ndarray],
    *,
    cache_dir: str | Path | None = None,
    block_size: int = 2_000_000,
    max_workers: int = 8,
    parallel_backend: str = "process",
    verbose: bool = True,
) -> dict[int, np.ndarray]:
    """Return ParentIDs for selected TracerIDs, grouped by halo index."""

    if block_size < 1 or max_workers < 1:
        raise ValueError("block_size and max_workers must be positive")
    if parallel_backend not in {"serial", "thread", "process"}:
        raise ValueError("Unknown parallel backend")
    paths = snapshot_chunk_paths(base_path, snap_num)
    fingerprint = parent_catalog_fingerprint(
        catalog,
        snap_num=snap_num,
        purpose="tracer_to_parent",
    )
    cache_root = (
        Path(cache_dir)
        / f"tracer_parents_snap{snap_num:03d}_{fingerprint[:16]}"
        if cache_dir is not None
        else None
    )
    results: list[dict[int, np.ndarray]] = []
    pending: list[tuple[int, Path, Path | None]] = []
    for chunk_index, path in enumerate(paths):
        cache_path = (
            cache_root / f"chunk_{chunk_index:04d}.npz"
            if cache_root is not None
            else None
        )
        cached = (
            _load_chunk_cache(cache_path, fingerprint)
            if cache_path is not None
            else None
        )
        if cached is not None:
            results.append(cached)
        else:
            pending.append((chunk_index, path, cache_path))
    if verbose:
        print(
            f"[snap {snap_num} tracer->parent] chunks={len(paths)}, "
            f"cached={len(paths) - len(pending)}, "
            f"pending={len(pending)}, workers={max_workers}"
        )
    serial = parallel_backend == "serial" or max_workers == 1
    if serial:
        for sequence, (_, path, cache_path) in enumerate(
            pending,
            start=1,
        ):
            result = _scan_tracer_parents_one_chunk(
                path,
                catalog,
                block_size,
            )
            results.append(result)
            if cache_path is not None:
                _save_chunk_cache(cache_path, result, fingerprint)
            if verbose:
                print(
                    f"[snap {snap_num} tracer->parent] "
                    f"{sequence}/{len(pending)}"
                )
    elif pending:
        executor_class = (
            ThreadPoolExecutor
            if parallel_backend == "thread"
            else ProcessPoolExecutor
        )
        executor_kwargs = {"max_workers": max_workers}
        if parallel_backend == "process":
            executor_kwargs.update(
                {
                    "initializer": _initialize_worker,
                    "initargs": (
                        catalog,
                        block_size,
                        "tracer_to_parent",
                    ),
                }
            )
        with executor_class(**executor_kwargs) as executor:
            future_map = {}
            for chunk_index, path, cache_path in pending:
                if parallel_backend == "thread":
                    future = executor.submit(
                        _scan_tracer_parents_one_chunk,
                        path,
                        catalog,
                        block_size,
                    )
                else:
                    future = executor.submit(_process_worker, str(path))
                future_map[future] = (chunk_index, cache_path)
            completed = 0
            for future in as_completed(future_map):
                chunk_index, cache_path = future_map[future]
                result = future.result()
                results.append(result)
                if cache_path is not None:
                    _save_chunk_cache(cache_path, result, fingerprint)
                completed += 1
                if verbose:
                    print(
                        f"[snap {snap_num} tracer->parent] "
                        f"{completed}/{len(pending)} "
                        f"(chunk index {chunk_index})"
                    )
    merged: dict[int, list[np.ndarray]] = {}
    for result in results:
        for label, parent_ids in result.items():
            if len(parent_ids):
                merged.setdefault(label, []).append(parent_ids)
    return {
        label: np.concatenate(parts)
        for label, parts in merged.items()
    }


def tracer_sets_for_halo(
    scanned: dict[int, np.ndarray],
    halo_index: int,
) -> dict[str, np.ndarray]:
    """Decode one halo's state-labelled tracer arrays from merged scan labels."""

    result = {}
    for state_code, name in enumerate(STATE_NAMES):
        label = int(halo_index) * N_STATES + state_code
        result[name] = np.asarray(
            scanned.get(label, np.empty(0, dtype=np.uint64)),
            dtype=np.uint64,
        )
    return result
