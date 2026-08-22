"""Chunked full-snapshot tracer scans and parent-particle lookup."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np


def snapshot_chunk_paths(base_path: str | Path, snap: int) -> list[Path]:
    """Return numerically ordered full-snapshot HDF5 chunks."""

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
    return paths


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


def scan_parent_to_tracer(
    base_path: str | Path,
    snap: int,
    parent_ids_by_label: dict[int, np.ndarray],
    *,
    cache_dir: str | Path | None = None,
    block_size: int = 2_000_000,
) -> dict[int, np.ndarray]:
    """Scan PartType3 ParentID→TracerID while preserving labels."""

    id_parts = [np.asarray(ids, dtype=np.uint64) for ids in parent_ids_by_label.values() if len(ids)] if parent_ids_by_label else []
    all_ids = np.concatenate(id_parts) if id_parts else np.empty(0, dtype=np.uint64)
    label_context = json.dumps(sorted((int(label), np.unique(np.asarray(ids, dtype=np.uint64)).tolist()) for label, ids in parent_ids_by_label.items()))
    fingerprint = _fingerprint("parent_to_tracer", snap, all_ids, f"{base_path}|{label_context}")
    path = Path(cache_dir) / f"parent_to_tracer_{fingerprint}.npz" if cache_dir else None
    if path is not None:
        cached = _load_map(path)
        if cached is not None:
            return cached
    parent_labels = []
    labels = []
    for label, ids in parent_ids_by_label.items():
        values = np.unique(np.asarray(ids, dtype=np.uint64))
        parent_labels.append(values)
        labels.append(np.full(len(values), int(label), dtype=np.int64))
    if not parent_labels or not any(len(values) for values in parent_labels):
        if path is not None:
            _save_map(path, {})
        return {}
    parents = np.concatenate(parent_labels)
    label_values = np.concatenate(labels)
    order = np.argsort(parents, kind="stable")
    parents = parents[order]
    label_values = label_values[order]
    selected: dict[int, list[np.ndarray]] = {}
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required for tracer scans") from exc
    for chunk in snapshot_chunk_paths(base_path, snap):
        with h5py.File(chunk, "r") as handle:
            if "PartType3" not in handle or "ParentID" not in handle["PartType3"] or "TracerID" not in handle["PartType3"]:
                continue
            group = handle["PartType3"]
            for start in range(0, len(group["ParentID"]), block_size):
                stop = min(start + block_size, len(group["ParentID"]))
                values = np.asarray(group["ParentID"][start:stop], dtype=np.uint64)
                valid, positions = _match(values, parents)
                if np.any(valid):
                    tracer_ids = np.asarray(group["TracerID"][start:stop], dtype=np.uint64)[valid]
                    for label in np.unique(label_values[positions[valid]]):
                        selected.setdefault(int(label), []).append(tracer_ids[label_values[positions[valid]] == label])
    merged = {label: np.concatenate(parts) for label, parts in selected.items()}
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
                    for tracer_index in np.where(valid)[0]:
                        label = labels_by_id[int(tracers[tracer_index])]
                        selected.setdefault(label, []).append(np.asarray([group["ParentID"][start + int(tracer_index)]], dtype=np.uint64))
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
) -> dict[int, int]:
    """Return an explicit ``TracerID -> ParentID`` map for event tracers."""

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
                return {int(tracer): int(parent) for tracer, parent in zip(tracer_values, parent_values)}
        except Exception:
            pass
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required for tracer scans") from exc
    found: dict[int, int] = {}
    for chunk in snapshot_chunk_paths(base_path, snap):
        with h5py.File(chunk, "r") as handle:
            if "PartType3" not in handle or "TracerID" not in handle["PartType3"] or "ParentID" not in handle["PartType3"]:
                continue
            group = handle["PartType3"]
            for start in range(0, len(group["TracerID"]), block_size):
                stop = min(start + block_size, len(group["TracerID"]))
                values = np.asarray(group["TracerID"][start:stop], dtype=np.uint64)
                valid, _ = _match(values, targets)
                for index in np.where(valid)[0]:
                    tracer = int(values[index])
                    parent = int(group["ParentID"][start + int(index)])
                    if tracer in found and found[tracer] != parent:
                        raise ValueError(f"TracerID {tracer} has multiple parents at snap {snap}")
                    found[tracer] = parent
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npz")
        np.savez(temporary, tracer_ids=np.asarray(list(found), dtype=np.uint64), parent_ids=np.asarray(list(found.values()), dtype=np.uint64))
        temporary.replace(path)
    return found


def lookup_parent_records(base_path: str | Path, snap: int, parent_ids: np.ndarray, *, cache_dir: str | Path | None = None, block_size: int = 2_000_000) -> dict[str, np.ndarray]:
    """Read gas, star, and black-hole records for selected ParentIDs."""

    targets = np.unique(np.asarray(parent_ids, dtype=np.uint64))
    empty = {
        "particle_ids": np.empty(0, dtype=np.uint64), "particle_type": np.empty(0, dtype=np.int8),
        "coordinates": np.empty((0, 3)), "sfr": np.empty(0), "internal_energy": np.empty(0),
        "electron_abundance": np.empty(0), "formation_time": np.empty(0),
    }
    fingerprint = _fingerprint("parent_records", snap, targets, str(base_path))
    path = Path(cache_dir) / f"parent_records_{fingerprint}.npz" if cache_dir else None
    if path is not None and path.is_file():
        try:
            with np.load(path, allow_pickle=False) as data:
                loaded = {key: np.asarray(data[key]) for key in empty}
                if loaded["particle_ids"].ndim == 1 and loaded["coordinates"].shape == (len(loaded["particle_ids"]), 3):
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
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required for particle lookup") from exc
    parts = {key: [] for key in empty}
    specifications = {
        "PartType0": (0, ("ParticleIDs", "Coordinates", "StarFormationRate", "InternalEnergy", "ElectronAbundance")),
        "PartType4": (4, ("ParticleIDs", "Coordinates", "GFM_StellarFormationTime")),
        "PartType5": (5, ("ParticleIDs",)),
    }
    for chunk in snapshot_chunk_paths(base_path, snap):
        with h5py.File(chunk, "r") as handle:
            for name, (particle_type, fields) in specifications.items():
                if name not in handle:
                    continue
                group = handle[name]
                if "ParticleIDs" not in group:
                    continue
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
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp.npz")
            np.savez(temporary, **empty)
            temporary.replace(path)
        return empty
    merged = {key: np.concatenate(values, axis=0) for key, values in parts.items()}
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
