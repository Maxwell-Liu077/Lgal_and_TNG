"""Chunked full-snapshot tracer scans and parent-particle lookup."""

from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import numpy as np


_WORKER_MODE: str | None = None
_WORKER_QUERY = None
_WORKER_BLOCK_SIZE: int | None = None


@lru_cache(maxsize=64)
def _snapshot_chunk_path_strings(base_path: str, snap: int) -> tuple[str, ...]:
    """Cache the immutable glob result for one snapshot."""

    base = Path(base_path)
    tag = f"{int(snap):03d}"
    directory = base / f"snapdir_{tag}"
    paths = list(directory.glob(f"snap_{tag}.*.hdf5")) if directory.is_dir() else list(base.glob(f"snap_{tag}.*.hdf5"))
    single = base / f"snap_{tag}.hdf5"
    if single.is_file():
        paths.append(single)
    paths = sorted(set(paths), key=lambda path: int(re.search(r"\.(\d+)\.hdf5$", path.name).group(1)) if re.search(r"\.(\d+)\.hdf5$", path.name) else 0)
    if not paths:
        raise FileNotFoundError(f"No snapshot chunks for snap={snap} in {base_path}")
    return tuple(str(path) for path in paths)


def snapshot_chunk_paths(base_path: str | Path, snap: int) -> list[Path]:
    """Return numerically ordered full-snapshot HDF5 chunks."""

    return [Path(path) for path in _snapshot_chunk_path_strings(str(base_path), int(snap))]


def _match(values: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return valid membership and sorted-target positions."""

    positions = np.searchsorted(targets, values)
    valid = positions < len(targets)
    if np.any(valid):
        indices = np.where(valid)[0]
        valid[indices] = targets[positions[indices]] == values[indices]
    return valid, positions


def _fingerprint(kind: str, snap: int, ids: np.ndarray, context: str = "") -> str:
    """Hash the scan kind, snapshot, and exact uint64 query IDs."""

    digest = hashlib.sha256(f"phase21-{kind}-snap{snap}-v1|{context}".encode())
    values = np.unique(np.asarray(ids, dtype=np.uint64))
    digest.update(values.tobytes())
    return digest.hexdigest()


def _load_map(path: Path) -> dict[int, np.ndarray] | None:
    """Load a cached integer-array map."""

    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            return {int(key[4:]): np.asarray(data[key], dtype=np.uint64) for key in data.files if key.startswith("key_")}
    except Exception:
        return None


def _save_map(path: Path, values: dict[int, np.ndarray]) -> None:
    """Atomically save an integer-array map."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **{f"key_{key}": np.asarray(value, dtype=np.uint64) for key, value in values.items()})
    temporary.replace(path)


def _build_labeled_catalog(parent_ids_by_label: dict[int, np.ndarray]) -> dict[str, np.ndarray]:
    """Build the vectorized duplicate-aware lookup used by chunk workers."""

    id_parts = []
    label_parts = []
    for label, ids in sorted(parent_ids_by_label.items()):
        values = np.unique(np.asarray(ids, dtype=np.uint64))
        if not len(values):
            continue
        id_parts.append(values)
        label_parts.append(np.full(len(values), int(label), dtype=np.int64))
    if not id_parts:
        return {
            "parent_ids": np.empty(0, dtype=np.uint64),
            "primary_labels": np.empty(0, dtype=np.int64),
            "duplicate_unique_indices": np.empty(0, dtype=np.int64),
            "duplicate_offsets": np.zeros(1, dtype=np.int64),
            "duplicate_extra_labels": np.empty(0, dtype=np.int64),
        }
    parents = np.concatenate(id_parts)
    labels = np.concatenate(label_parts)
    order = np.argsort(parents, kind="stable")
    sorted_parents = parents[order]
    sorted_labels = labels[order]
    unique_parents, first = np.unique(sorted_parents, return_index=True)
    primary_labels = sorted_labels[first]
    primary = np.zeros(len(sorted_parents), dtype=bool)
    primary[first] = True
    extra_parent_ids = sorted_parents[~primary]
    extra_labels = sorted_labels[~primary]
    if len(extra_parent_ids):
        extra_indices = np.searchsorted(unique_parents, extra_parent_ids).astype(np.int64)
        duplicate_unique_indices, counts = np.unique(extra_indices, return_counts=True)
        duplicate_offsets = np.concatenate(
            [np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64)]
        )
    else:
        duplicate_unique_indices = np.empty(0, dtype=np.int64)
        duplicate_offsets = np.zeros(1, dtype=np.int64)
    return {
        "parent_ids": unique_parents,
        "primary_labels": primary_labels,
        "duplicate_unique_indices": duplicate_unique_indices,
        "duplicate_offsets": duplicate_offsets,
        "duplicate_extra_labels": extra_labels,
    }


def _append_grouped(
    selected: dict[int, list[np.ndarray]],
    labels: np.ndarray,
    values: np.ndarray,
) -> None:
    """Append matched values to label-keyed output without Python per-ID loops."""

    if not len(labels):
        return
    order = np.argsort(labels, kind="stable")
    labels = labels[order]
    values = values[order]
    unique_labels, first = np.unique(labels, return_index=True)
    stops = np.concatenate((first[1:], np.asarray([len(labels)])))
    for label, start, stop in zip(unique_labels, first, stops):
        selected.setdefault(int(label), []).append(values[int(start):int(stop)])


def _scan_parent_chunk(
    path: str | Path,
    catalog: dict[str, np.ndarray],
    block_size: int,
) -> dict[int, np.ndarray]:
    """Scan one PartType3 chunk for labeled ParentID→TracerID matches."""

    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required for tracer scans") from exc
    targets = catalog["parent_ids"]
    if not len(targets):
        return {}
    selected: dict[int, list[np.ndarray]] = {}
    with h5py.File(path, "r") as handle:
        if "PartType3" not in handle:
            return {}
        group = handle["PartType3"]
        if "ParentID" not in group or "TracerID" not in group:
            raise KeyError(f"Missing PartType3 IDs in {path}")
        if len(group["ParentID"]) != len(group["TracerID"]):
            raise ValueError(f"PartType3 lengths differ in {path}")
        for start in range(0, len(group["ParentID"]), block_size):
            stop = min(start + block_size, len(group["ParentID"]))
            parents = np.asarray(group["ParentID"][start:stop], dtype=np.uint64)
            valid, positions = _match(parents, targets)
            if not np.any(valid):
                continue
            tracers = np.asarray(group["TracerID"][start:stop], dtype=np.uint64)[valid]
            unique_indices = positions[valid].astype(np.int64, copy=False)
            _append_grouped(selected, catalog["primary_labels"][unique_indices], tracers)

            duplicate_indices = catalog["duplicate_unique_indices"]
            if not len(duplicate_indices):
                continue
            duplicate_positions = np.searchsorted(duplicate_indices, unique_indices)
            duplicate = duplicate_positions < len(duplicate_indices)
            if np.any(duplicate):
                candidate = np.flatnonzero(duplicate)
                duplicate[candidate] = (
                    duplicate_indices[duplicate_positions[candidate]] == unique_indices[candidate]
                )
            if not np.any(duplicate):
                continue
            duplicate_positions = duplicate_positions[duplicate]
            duplicate_tracers = tracers[duplicate]
            offsets = catalog["duplicate_offsets"]
            counts = offsets[duplicate_positions + 1] - offsets[duplicate_positions]
            repeated_tracers = np.repeat(duplicate_tracers, counts)
            total = int(counts.sum())
            group_starts = np.repeat(np.cumsum(counts) - counts, counts)
            label_positions = (
                np.repeat(offsets[duplicate_positions], counts)
                + np.arange(total)
                - group_starts
            )
            _append_grouped(
                selected,
                catalog["duplicate_extra_labels"][label_positions],
                repeated_tracers,
            )
    return {label: np.concatenate(parts) for label, parts in selected.items()}


def _scan_tracer_map_chunk(
    path: str | Path,
    targets: np.ndarray,
    block_size: int,
) -> dict[str, np.ndarray]:
    """Return selected TracerID/ParentID pairs from one PartType3 chunk."""

    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required for tracer scans") from exc
    tracer_parts = []
    parent_parts = []
    with h5py.File(path, "r") as handle:
        if "PartType3" not in handle:
            return {"tracer_ids": np.empty(0, dtype=np.uint64), "parent_ids": np.empty(0, dtype=np.uint64)}
        group = handle["PartType3"]
        if "TracerID" not in group or "ParentID" not in group:
            raise KeyError(f"Missing PartType3 IDs in {path}")
        if len(group["TracerID"]) != len(group["ParentID"]):
            raise ValueError(f"PartType3 lengths differ in {path}")
        for start in range(0, len(group["TracerID"]), block_size):
            stop = min(start + block_size, len(group["TracerID"]))
            tracers = np.asarray(group["TracerID"][start:stop], dtype=np.uint64)
            valid, _ = _match(tracers, targets)
            if not np.any(valid):
                continue
            indices = np.flatnonzero(valid)
            tracer_parts.append(tracers[indices])
            parent_parts.append(np.asarray(group["ParentID"][start:stop], dtype=np.uint64)[indices])
    return {
        "tracer_ids": np.concatenate(tracer_parts) if tracer_parts else np.empty(0, dtype=np.uint64),
        "parent_ids": np.concatenate(parent_parts) if parent_parts else np.empty(0, dtype=np.uint64),
    }


def _empty_records() -> dict[str, np.ndarray]:
    """Return a typed empty parent-particle table."""

    return {
        "particle_ids": np.empty(0, dtype=np.uint64),
        "particle_type": np.empty(0, dtype=np.int8),
        "coordinates": np.empty((0, 3)),
        "sfr": np.empty(0),
        "internal_energy": np.empty(0),
        "electron_abundance": np.empty(0),
        "formation_time": np.empty(0),
    }


def _scan_records_chunk(
    path: str | Path,
    targets: np.ndarray,
    block_size: int,
) -> dict[str, np.ndarray]:
    """Read selected gas/star/BH parent records from one snapshot chunk."""

    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required for particle lookup") from exc
    parts = {key: [] for key in _empty_records()}
    specifications = {
        "PartType0": (0, ("ParticleIDs", "Coordinates", "StarFormationRate", "InternalEnergy", "ElectronAbundance")),
        "PartType4": (4, ("ParticleIDs", "Coordinates", "GFM_StellarFormationTime")),
        "PartType5": (5, ("ParticleIDs",)),
    }
    with h5py.File(path, "r") as handle:
        for name, (particle_type, _) in specifications.items():
            if name not in handle or "ParticleIDs" not in handle[name]:
                continue
            group = handle[name]
            for start in range(0, len(group["ParticleIDs"]), block_size):
                stop = min(start + block_size, len(group["ParticleIDs"]))
                ids = np.asarray(group["ParticleIDs"][start:stop], dtype=np.uint64)
                valid, _ = _match(ids, targets)
                if not np.any(valid):
                    continue
                count = int(np.count_nonzero(valid))
                parts["particle_ids"].append(ids[valid])
                parts["particle_type"].append(np.full(count, particle_type, dtype=np.int8))
                parts["coordinates"].append(np.asarray(group["Coordinates"][start:stop], dtype=float)[valid] if "Coordinates" in group else np.full((count, 3), np.nan))
                parts["sfr"].append(np.asarray(group["StarFormationRate"][start:stop], dtype=float)[valid] if "StarFormationRate" in group else np.full(count, np.nan))
                parts["internal_energy"].append(np.asarray(group["InternalEnergy"][start:stop], dtype=float)[valid] if "InternalEnergy" in group else np.full(count, np.nan))
                parts["electron_abundance"].append(np.asarray(group["ElectronAbundance"][start:stop], dtype=float)[valid] if "ElectronAbundance" in group else np.full(count, np.nan))
                parts["formation_time"].append(np.asarray(group["GFM_StellarFormationTime"][start:stop], dtype=float)[valid] if "GFM_StellarFormationTime" in group else np.full(count, np.nan))
    if not parts["particle_ids"]:
        return _empty_records()
    return {key: np.concatenate(values, axis=0) for key, values in parts.items()}


def _initialize_chunk_worker(mode: str, query, block_size: int) -> None:
    """Install immutable scan inputs once in each process worker."""

    global _WORKER_MODE, _WORKER_QUERY, _WORKER_BLOCK_SIZE
    _WORKER_MODE = mode
    _WORKER_QUERY = query
    _WORKER_BLOCK_SIZE = int(block_size)


def _process_chunk_worker(path: str):
    """Dispatch one HDF5 chunk using process-local immutable query data."""

    if _WORKER_MODE is None or _WORKER_QUERY is None or _WORKER_BLOCK_SIZE is None:
        raise RuntimeError("Tracer chunk worker was not initialized")
    if _WORKER_MODE == "parent_to_tracer":
        return _scan_parent_chunk(path, _WORKER_QUERY, _WORKER_BLOCK_SIZE)
    if _WORKER_MODE == "tracer_parent_map":
        return _scan_tracer_map_chunk(path, _WORKER_QUERY, _WORKER_BLOCK_SIZE)
    if _WORKER_MODE == "parent_records":
        return _scan_records_chunk(path, _WORKER_QUERY, _WORKER_BLOCK_SIZE)
    raise RuntimeError(f"Unknown tracer worker mode: {_WORKER_MODE}")


def _save_grouped_chunk(path: Path, result: dict[int, np.ndarray], fingerprint: str) -> None:
    """Atomically save one label-grouped tracer chunk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(sorted(result), dtype=np.int64)
    counts = np.asarray([len(result[int(label)]) for label in labels], dtype=np.int64)
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)))
    values = np.concatenate([result[int(label)] for label in labels]) if len(labels) else np.empty(0, dtype=np.uint64)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, fingerprint=np.asarray(fingerprint), labels=labels, offsets=offsets, values=values)
    temporary.replace(path)


def _load_grouped_chunk(path: Path, fingerprint: str) -> dict[int, np.ndarray] | None:
    """Load one validated label-grouped tracer chunk."""

    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(np.asarray(data["fingerprint"]).item()) != fingerprint:
                return None
            labels = np.asarray(data["labels"], dtype=np.int64)
            offsets = np.asarray(data["offsets"], dtype=np.int64)
            values = np.asarray(data["values"], dtype=np.uint64)
        if len(offsets) != len(labels) + 1 or offsets[0] != 0 or offsets[-1] != len(values):
            return None
        return {int(label): values[offsets[i]:offsets[i + 1]] for i, label in enumerate(labels)}
    except Exception:
        return None


def _save_array_chunk(path: Path, result: dict[str, np.ndarray], fingerprint: str) -> None:
    """Atomically save one pair/record chunk with a query fingerprint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, fingerprint=np.asarray(fingerprint), **result)
    temporary.replace(path)


def _load_array_chunk(
    path: Path,
    fingerprint: str,
    template: dict[str, np.ndarray],
) -> dict[str, np.ndarray] | None:
    """Load a typed pair/record chunk only when its schema is complete."""

    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(np.asarray(data["fingerprint"]).item()) != fingerprint:
                return None
            result = {key: np.asarray(data[key], dtype=value.dtype) for key, value in template.items()}
        lengths = {len(values) for values in result.values()}
        return result if len(lengths) == 1 else None
    except Exception:
        return None


def _merge_grouped(results: list[dict[int, np.ndarray]]) -> dict[int, np.ndarray]:
    """Merge chunk-local label groups into unique full-snapshot arrays."""

    merged: dict[int, list[np.ndarray]] = {}
    for result in results:
        for label, values in result.items():
            if len(values):
                merged.setdefault(int(label), []).append(np.asarray(values, dtype=np.uint64))
    return {label: np.unique(np.concatenate(parts)) for label, parts in merged.items()}


def scan_parent_to_tracer(
    base_path: str | Path,
    snap: int,
    parent_ids_by_label: dict[int, np.ndarray],
    *,
    cache_dir: str | Path | None = None,
    block_size: int = 2_000_000,
    max_workers: int = 1,
    parallel_backend: str = "serial",
    verbose: bool = False,
) -> dict[int, np.ndarray]:
    """Scan ParentID→TracerID with resumable chunk-level parallelism."""

    if block_size < 1 or max_workers < 1:
        raise ValueError("block_size and max_workers must be positive")
    if parallel_backend not in {"serial", "thread", "process"}:
        raise ValueError("Unknown tracer parallel backend")

    id_parts = [np.asarray(ids, dtype=np.uint64) for ids in parent_ids_by_label.values() if len(ids)] if parent_ids_by_label else []
    all_ids = np.concatenate(id_parts) if id_parts else np.empty(0, dtype=np.uint64)
    label_context = json.dumps(sorted((int(label), np.unique(np.asarray(ids, dtype=np.uint64)).tolist()) for label, ids in parent_ids_by_label.items()))
    fingerprint = _fingerprint("parent_to_tracer", snap, all_ids, f"{base_path}|{label_context}")
    path = Path(cache_dir) / f"parent_to_tracer_{fingerprint}.npz" if cache_dir else None
    if path is not None:
        cached = _load_map(path)
        if cached is not None:
            if verbose:
                print(f"[tracer parent->tracer snap={snap:03d}] final cache hit", flush=True)
            return cached
    catalog = _build_labeled_catalog(parent_ids_by_label)
    if not len(catalog["parent_ids"]):
        if path is not None:
            _save_map(path, {})
        return {}

    paths = snapshot_chunk_paths(base_path, snap)
    chunk_root = (
        Path(cache_dir) / "chunk_scans" / f"parent_to_tracer_snap{snap:03d}_{fingerprint[:16]}"
        if cache_dir is not None else None
    )
    results = []
    pending = []
    for index, chunk in enumerate(paths):
        chunk_cache = chunk_root / f"chunk_{index:04d}.npz" if chunk_root is not None else None
        cached = _load_grouped_chunk(chunk_cache, fingerprint) if chunk_cache is not None else None
        if cached is None:
            pending.append((index, chunk, chunk_cache))
        else:
            results.append(cached)
    if verbose:
        print(
            f"[tracer parent->tracer snap={snap:03d}] chunks={len(paths)} "
            f"cached={len(paths) - len(pending)} pending={len(pending)} workers={min(max_workers, max(1, len(pending)))}",
            flush=True,
        )
    serial = parallel_backend == "serial" or max_workers == 1
    if serial:
        for completed, (index, chunk, chunk_cache) in enumerate(pending, start=1):
            result = _scan_parent_chunk(chunk, catalog, block_size)
            results.append(result)
            if chunk_cache is not None:
                _save_grouped_chunk(chunk_cache, result, fingerprint)
            if verbose:
                print(f"[tracer parent->tracer snap={snap:03d}] {completed}/{len(pending)} chunk={index}", flush=True)
    elif pending:
        executor_class = ThreadPoolExecutor if parallel_backend == "thread" else ProcessPoolExecutor
        kwargs = {"max_workers": min(max_workers, len(pending))}
        if parallel_backend == "process":
            kwargs.update({"initializer": _initialize_chunk_worker, "initargs": ("parent_to_tracer", catalog, block_size)})
        with executor_class(**kwargs) as executor:
            future_map = {}
            for index, chunk, chunk_cache in pending:
                future = (
                    executor.submit(_scan_parent_chunk, chunk, catalog, block_size)
                    if parallel_backend == "thread"
                    else executor.submit(_process_chunk_worker, str(chunk))
                )
                future_map[future] = (index, chunk_cache)
            completed = 0
            for future in as_completed(future_map):
                index, chunk_cache = future_map.pop(future)
                result = future.result()
                results.append(result)
                if chunk_cache is not None:
                    _save_grouped_chunk(chunk_cache, result, fingerprint)
                completed += 1
                if verbose:
                    print(f"[tracer parent->tracer snap={snap:03d}] {completed}/{len(pending)} chunk={index}", flush=True)
    merged = _merge_grouped(results)
    if path is not None:
        _save_map(path, merged)
    return merged


def scan_tracer_to_parent(
    base_path: str | Path,
    snap: int,
    tracer_ids_by_label: dict[int, np.ndarray],
    *,
    cache_dir: str | Path | None = None,
    block_size: int = 2_000_000,
) -> dict[int, np.ndarray]:
    """Scan PartType3 TracerID→ParentID for selected event tracers."""

    id_parts = [np.asarray(ids, dtype=np.uint64) for ids in tracer_ids_by_label.values() if len(ids)] if tracer_ids_by_label else []
    all_ids = np.concatenate(id_parts) if id_parts else np.empty(0, dtype=np.uint64)
    label_context = json.dumps(sorted((int(label), np.unique(np.asarray(ids, dtype=np.uint64)).tolist()) for label, ids in tracer_ids_by_label.items()))
    fingerprint = _fingerprint("tracer_to_parent", snap, all_ids, f"{base_path}|{label_context}")
    path = Path(cache_dir) / f"tracer_to_parent_{fingerprint}.npz" if cache_dir else None
    if path is not None:
        cached = _load_map(path)
        if cached is not None:
            return cached
    labels_by_id: dict[int, int] = {}
    for label, ids in tracer_ids_by_label.items():
        for tracer_id in np.unique(np.asarray(ids, dtype=np.uint64)):
            labels_by_id[int(tracer_id)] = int(label)
    targets = np.array(sorted(labels_by_id), dtype=np.uint64)
    if not len(targets):
        if path is not None:
            _save_map(path, {})
        return {}
    selected: dict[int, list[np.ndarray]] = {}
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required for tracer scans") from exc
    for chunk in snapshot_chunk_paths(base_path, snap):
        with h5py.File(chunk, "r") as handle:
            if "PartType3" not in handle or "TracerID" not in handle["PartType3"] or "ParentID" not in handle["PartType3"]:
                continue
            group = handle["PartType3"]
            for start in range(0, len(group["TracerID"]), block_size):
                stop = min(start + block_size, len(group["TracerID"]))
                tracers = np.asarray(group["TracerID"][start:stop], dtype=np.uint64)
                valid, positions = _match(tracers, targets)
                if np.any(valid):
                    indices = np.flatnonzero(valid)
                    parent_values = np.asarray(
                        group["ParentID"][start:stop],
                        dtype=np.uint64,
                    )[indices]
                    for tracer_value, parent_value in zip(tracers[indices], parent_values):
                        label = labels_by_id[int(tracer_value)]
                        selected.setdefault(label, []).append(
                            np.asarray([parent_value], dtype=np.uint64)
                        )
    merged = {label: np.concatenate(parts) for label, parts in selected.items()}
    if path is not None:
        _save_map(path, merged)
    return merged


def scan_tracer_parent_map(
    base_path: str | Path,
    snap: int,
    tracer_ids: np.ndarray,
    *,
    cache_dir: str | Path | None = None,
    block_size: int = 2_000_000,
    max_workers: int = 1,
    parallel_backend: str = "serial",
    verbose: bool = False,
) -> dict[int, int]:
    """Return TracerID→ParentID using resumable chunk-level parallelism."""

    if block_size < 1 or max_workers < 1:
        raise ValueError("block_size and max_workers must be positive")
    if parallel_backend not in {"serial", "thread", "process"}:
        raise ValueError("Unknown tracer parallel backend")

    targets = np.unique(np.asarray(tracer_ids, dtype=np.uint64))
    if not len(targets):
        if cache_dir is not None:
            fingerprint = _fingerprint("tracer_parent_map", snap, targets, str(base_path))
            path = Path(cache_dir) / f"tracer_parent_map_{fingerprint}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp.npz")
            np.savez(temporary, tracer_ids=np.empty(0, dtype=np.uint64), parent_ids=np.empty(0, dtype=np.uint64))
            temporary.replace(path)
        return {}
    fingerprint = _fingerprint("tracer_parent_map", snap, targets, str(base_path))
    path = Path(cache_dir) / f"tracer_parent_map_{fingerprint}.npz" if cache_dir else None
    if path is not None and path.is_file():
        try:
            with np.load(path, allow_pickle=False) as data:
                tracer_values = np.asarray(data["tracer_ids"], dtype=np.uint64)
                parent_values = np.asarray(data["parent_ids"], dtype=np.uint64)
                if tracer_values.ndim != 1 or parent_values.shape != tracer_values.shape:
                    raise ValueError("invalid tracer-parent cache shape")
                if len(np.unique(tracer_values)) != len(tracer_values):
                    raise ValueError("duplicate TracerID in cache")
                if verbose:
                    print(f"[tracer tracer->parent snap={snap:03d}] final cache hit", flush=True)
                return {int(tracer): int(parent) for tracer, parent in zip(tracer_values, parent_values)}
        except Exception:
            pass

    paths = snapshot_chunk_paths(base_path, snap)
    chunk_root = (
        Path(cache_dir) / "chunk_scans" / f"tracer_parent_snap{snap:03d}_{fingerprint[:16]}"
        if cache_dir is not None else None
    )
    template = {"tracer_ids": np.empty(0, dtype=np.uint64), "parent_ids": np.empty(0, dtype=np.uint64)}
    results = []
    pending = []
    for index, chunk in enumerate(paths):
        chunk_cache = chunk_root / f"chunk_{index:04d}.npz" if chunk_root is not None else None
        cached = _load_array_chunk(chunk_cache, fingerprint, template) if chunk_cache is not None else None
        if cached is None:
            pending.append((index, chunk, chunk_cache))
        else:
            results.append(cached)
    if verbose:
        print(
            f"[tracer tracer->parent snap={snap:03d}] chunks={len(paths)} "
            f"cached={len(paths) - len(pending)} pending={len(pending)} workers={min(max_workers, max(1, len(pending)))}",
            flush=True,
        )
    serial = parallel_backend == "serial" or max_workers == 1
    if serial:
        for completed, (index, chunk, chunk_cache) in enumerate(pending, start=1):
            result = _scan_tracer_map_chunk(chunk, targets, block_size)
            results.append(result)
            if chunk_cache is not None:
                _save_array_chunk(chunk_cache, result, fingerprint)
            if verbose:
                print(f"[tracer tracer->parent snap={snap:03d}] {completed}/{len(pending)} chunk={index}", flush=True)
    elif pending:
        executor_class = ThreadPoolExecutor if parallel_backend == "thread" else ProcessPoolExecutor
        kwargs = {"max_workers": min(max_workers, len(pending))}
        if parallel_backend == "process":
            kwargs.update({"initializer": _initialize_chunk_worker, "initargs": ("tracer_parent_map", targets, block_size)})
        with executor_class(**kwargs) as executor:
            future_map = {}
            for index, chunk, chunk_cache in pending:
                future = (
                    executor.submit(_scan_tracer_map_chunk, chunk, targets, block_size)
                    if parallel_backend == "thread"
                    else executor.submit(_process_chunk_worker, str(chunk))
                )
                future_map[future] = (index, chunk_cache)
            completed = 0
            for future in as_completed(future_map):
                index, chunk_cache = future_map.pop(future)
                result = future.result()
                results.append(result)
                if chunk_cache is not None:
                    _save_array_chunk(chunk_cache, result, fingerprint)
                completed += 1
                if verbose:
                    print(f"[tracer tracer->parent snap={snap:03d}] {completed}/{len(pending)} chunk={index}", flush=True)

    found: dict[int, int] = {}
    for result in results:
        for tracer_value, parent_value in zip(result["tracer_ids"], result["parent_ids"]):
            tracer = int(tracer_value)
            parent = int(parent_value)
            if tracer in found and found[tracer] != parent:
                raise ValueError(f"TracerID {tracer} has multiple parents at snap {snap}")
            found[tracer] = parent
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npz")
        np.savez(temporary, tracer_ids=np.asarray(list(found), dtype=np.uint64), parent_ids=np.asarray(list(found.values()), dtype=np.uint64))
        temporary.replace(path)
    return found


def lookup_parent_records(
    base_path: str | Path,
    snap: int,
    parent_ids: np.ndarray,
    *,
    cache_dir: str | Path | None = None,
    block_size: int = 2_000_000,
    max_workers: int = 1,
    parallel_backend: str = "serial",
    verbose: bool = False,
) -> dict[str, np.ndarray]:
    """Read parent records with resumable chunk-level parallelism."""

    if block_size < 1 or max_workers < 1:
        raise ValueError("block_size and max_workers must be positive")
    if parallel_backend not in {"serial", "thread", "process"}:
        raise ValueError("Unknown tracer parallel backend")

    targets = np.unique(np.asarray(parent_ids, dtype=np.uint64))
    empty = _empty_records()
    fingerprint = _fingerprint("parent_records", snap, targets, str(base_path))
    path = Path(cache_dir) / f"parent_records_{fingerprint}.npz" if cache_dir else None
    if path is not None and path.is_file():
        try:
            with np.load(path, allow_pickle=False) as data:
                loaded = {key: np.asarray(data[key]) for key in empty}
                if loaded["particle_ids"].ndim == 1 and loaded["coordinates"].shape == (len(loaded["particle_ids"]), 3):
                    if verbose:
                        print(f"[particle parent lookup snap={snap:03d}] final cache hit", flush=True)
                    return loaded
        except Exception:
            pass
    if not len(targets):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp.npz")
            np.savez(temporary, **empty)
            temporary.replace(path)
        return empty
    paths = snapshot_chunk_paths(base_path, snap)
    chunk_root = (
        Path(cache_dir) / "chunk_scans" / f"parent_records_snap{snap:03d}_{fingerprint[:16]}"
        if cache_dir is not None else None
    )
    records_per_chunk = []
    pending = []
    for index, chunk in enumerate(paths):
        chunk_cache = chunk_root / f"chunk_{index:04d}.npz" if chunk_root is not None else None
        cached = _load_array_chunk(chunk_cache, fingerprint, empty) if chunk_cache is not None else None
        if cached is None:
            pending.append((index, chunk, chunk_cache))
        else:
            records_per_chunk.append(cached)
    if verbose:
        print(
            f"[particle parent lookup snap={snap:03d}] chunks={len(paths)} "
            f"cached={len(paths) - len(pending)} pending={len(pending)} workers={min(max_workers, max(1, len(pending)))}",
            flush=True,
        )
    serial = parallel_backend == "serial" or max_workers == 1
    if serial:
        for completed, (index, chunk, chunk_cache) in enumerate(pending, start=1):
            records = _scan_records_chunk(chunk, targets, block_size)
            records_per_chunk.append(records)
            if chunk_cache is not None:
                _save_array_chunk(chunk_cache, records, fingerprint)
            if verbose:
                print(f"[particle parent lookup snap={snap:03d}] {completed}/{len(pending)} chunk={index}", flush=True)
    elif pending:
        executor_class = ThreadPoolExecutor if parallel_backend == "thread" else ProcessPoolExecutor
        kwargs = {"max_workers": min(max_workers, len(pending))}
        if parallel_backend == "process":
            kwargs.update({"initializer": _initialize_chunk_worker, "initargs": ("parent_records", targets, block_size)})
        with executor_class(**kwargs) as executor:
            future_map = {}
            for index, chunk, chunk_cache in pending:
                future = (
                    executor.submit(_scan_records_chunk, chunk, targets, block_size)
                    if parallel_backend == "thread"
                    else executor.submit(_process_chunk_worker, str(chunk))
                )
                future_map[future] = (index, chunk_cache)
            completed = 0
            for future in as_completed(future_map):
                index, chunk_cache = future_map.pop(future)
                records = future.result()
                records_per_chunk.append(records)
                if chunk_cache is not None:
                    _save_array_chunk(chunk_cache, records, fingerprint)
                completed += 1
                if verbose:
                    print(f"[particle parent lookup snap={snap:03d}] {completed}/{len(pending)} chunk={index}", flush=True)

    nonempty = [records for records in records_per_chunk if len(records["particle_ids"])]
    if not nonempty:
        result = empty
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp.npz")
            np.savez(temporary, **result)
            temporary.replace(path)
        return result
    merged = {key: np.concatenate([records[key] for records in nonempty], axis=0) for key in empty}
    particle_ids = np.asarray(merged["particle_ids"], dtype=np.uint64)
    if len(np.unique(particle_ids)) != len(particle_ids):
        raise ValueError(f"Parent particle ID appears more than once at snap {snap}")
    order = np.argsort(merged["particle_ids"], kind="stable")
    result = {key: values[order] for key, values in merged.items()}
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npz")
        np.savez(temporary, **result)
        temporary.replace(path)
    return result
