"""Snapshot-level group-catalogue caches used by the Phase 2.1 pipeline.

The original implementation queried the same global group catalogue once per
halo and snapshot.  Those arrays are immutable inputs for a run, so loading
them once per snapshot (and caching them atomically) removes thousands of
repeated catalogue reads without changing any scientific definition.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


CATALOGUE_SCHEMA_VERSION = 2
HALO_FIELDS = (
    "GroupFirstSub",
    "GroupNsubs",
    "Group_M_Crit200",
    "Group_R_Crit200",
)
SUBHALO_FIELDS = (
    "SubhaloFlag",
    "SubhaloMassInRadType",
    "SubhaloVmax",
    "SubhaloBHMass",
    "SubhaloLenType",
)


def _cache_path(
    cache_dir: str | Path,
    base_path: str | Path,
    snap: int,
) -> Path:
    """Return a deterministic per-snapshot catalogue cache path."""

    payload = {
        "schema": CATALOGUE_SCHEMA_VERSION,
        "base_path": str(Path(base_path).resolve()),
        "snap": int(snap),
        "halo_fields": HALO_FIELDS,
        "subhalo_fields": SUBHALO_FIELDS,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return Path(cache_dir) / "catalogues" / f"snap{int(snap):03d}_{digest}.npz"


def _normalise_catalogue(halos: dict, subhalos: dict) -> dict[str, np.ndarray]:
    """Validate and normalize loader output to numeric arrays."""

    if not isinstance(halos, dict) or not isinstance(subhalos, dict):
        raise ValueError("TNG groupcat loaders must return dictionaries")

    required_halo = {"GroupFirstSub", "GroupNsubs"}
    required_subhalo = {"SubhaloFlag", "SubhaloMassInRadType", "SubhaloLenType"}
    if not required_halo.issubset(halos):
        raise ValueError(f"Missing halo catalogue fields: {sorted(required_halo - set(halos))}")
    if not required_subhalo.issubset(subhalos):
        raise ValueError(
            "Missing subhalo catalogue fields: "
            f"{sorted(required_subhalo - set(subhalos))}"
        )

    first = np.asarray(halos["GroupFirstSub"], dtype=np.int64)
    nsubs = np.asarray(halos["GroupNsubs"], dtype=np.int64)
    m200 = np.asarray(
        halos.get("Group_M_Crit200", np.full(len(first), np.nan)),
        dtype=float,
    )
    r200 = np.asarray(
        halos.get("Group_R_Crit200", np.full(len(first), np.nan)),
        dtype=float,
    )
    flags = np.asarray(subhalos["SubhaloFlag"], dtype=bool)
    mass_type = np.asarray(subhalos["SubhaloMassInRadType"], dtype=float)
    lengths = np.asarray(subhalos["SubhaloLenType"], dtype=np.int64)

    if first.ndim != 1 or nsubs.shape != first.shape:
        raise ValueError("GroupFirstSub and GroupNsubs must be one-dimensional and aligned")
    if m200.shape != first.shape or r200.shape != first.shape:
        raise ValueError("Group mass/radius fields have inconsistent lengths")
    if flags.ndim != 1 or mass_type.ndim != 2 or lengths.ndim != 2:
        raise ValueError("Invalid subhalo catalogue dimensions")
    if mass_type.shape[0] != len(flags) or lengths.shape[0] != len(flags):
        raise ValueError("Subhalo catalogue fields have inconsistent lengths")
    if mass_type.shape[1] <= 4 or lengths.shape[1] <= 4:
        raise ValueError("Subhalo catalogue fields lack the stellar component")

    # These fields are available in TNG group catalogues.  Keep a NaN fallback
    # for custom/minimal test catalogues so the state code remains usable.
    vmax = np.asarray(
        subhalos.get("SubhaloVmax", np.full(len(flags), np.nan)),
        dtype=float,
    )
    bh_mass = np.asarray(
        subhalos.get("SubhaloBHMass", np.full(len(flags), np.nan)),
        dtype=float,
    )
    if vmax.shape != flags.shape or bh_mass.shape != flags.shape:
        raise ValueError("Optional subhalo fields have inconsistent lengths")

    return {
        "GroupFirstSub": first,
        "GroupNsubs": nsubs,
        "Group_M_Crit200": m200,
        "Group_R_Crit200": r200,
        "SubhaloFlag": flags,
        "SubhaloMassInRadType": mass_type,
        "SubhaloVmax": vmax,
        "SubhaloBHMass": bh_mass,
        "SubhaloLenType": lengths,
    }


def _load_cached(path: Path) -> dict[str, np.ndarray] | None:
    """Read a catalogue cache, returning ``None`` for corrupt/incomplete data."""

    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            values = {key: np.asarray(data[key]) for key in data.files}
        return _normalise_catalogue(
            {key: values[key] for key in HALO_FIELDS},
            {key: values[key] for key in SUBHALO_FIELDS},
        )
    except Exception:
        return None


def _save_cached(path: Path, values: dict[str, np.ndarray]) -> None:
    """Write a catalogue cache through an atomic sibling file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **values)
    temporary.replace(path)


def _load_one_from_tng(base_path: str | Path, snap: int) -> dict[str, np.ndarray]:
    """Load the minimal global fields needed by MPB and state construction."""

    try:
        import illustris_python as il
    except ImportError as exc:
        raise ImportError("illustris_python is required for group catalogue reads") from exc

    halos = il.groupcat.loadHalos(str(base_path), int(snap), fields=list(HALO_FIELDS))
    subhalos = il.groupcat.loadSubhalos(
        str(base_path),
        int(snap),
        fields=list(SUBHALO_FIELDS),
    )
    return _normalise_catalogue(halos, subhalos)


def load_catalogue_cache(
    base_path: str | Path,
    snapshots: tuple[int, ...] | list[int],
    *,
    cache_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict[int, dict[str, np.ndarray]]:
    """Load one immutable catalogue snapshot, reusing an atomic NPZ cache."""

    output: dict[int, dict[str, np.ndarray]] = {}
    for snap in snapshots:
        path = _cache_path(cache_dir, base_path, snap) if cache_dir is not None else None
        values = _load_cached(path) if path is not None else None
        source = "cache"
        if values is None:
            values = _load_one_from_tng(base_path, snap)
            source = "TNG"
            if path is not None:
                _save_cached(path, values)
        output[int(snap)] = values
        if verbose:
            print(
                f"[catalogue] snap={int(snap):03d} source={source} "
                f"groups={len(values['GroupFirstSub'])} "
                f"subhalos={len(values['SubhaloFlag'])}",
                flush=True,
            )
    return output


def load_subhalo_particle_offsets(
    base_path: str | Path,
    snap: int,
    *,
    particle_types: tuple[int, ...] = (0, 4, 5),
) -> dict[int, np.ndarray]:
    """Load global snapshot offsets used to identify bound subhalos.

    TNG stores the exact first global particle index of every subhalo in the
    separate offsets file.  Reading only the requested particle-type columns
    lets the event layer distinguish another bound galaxy from genuinely
    unbound FoF/outer fuzz without constructing a full ParticleID catalogue.
    """

    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required for TNG subhalo offsets") from exc

    try:
        from illustris_python.groupcat import offsetPath

        path = Path(offsetPath(str(base_path), int(snap)))
    except (ImportError, AttributeError):
        path = Path(base_path).resolve().parent / "postprocessing" / "offsets" / f"offsets_{int(snap):03d}.hdf5"
    if not path.is_file():
        raise FileNotFoundError(f"Missing TNG offsets file for snap={int(snap)}: {path}")

    output: dict[int, np.ndarray] = {}
    with h5py.File(path, "r") as handle:
        if "Subhalo/SnapByType" not in handle:
            raise KeyError(f"Subhalo/SnapByType missing from {path}")
        dataset = handle["Subhalo/SnapByType"]
        if dataset.ndim != 2 or dataset.shape[1] < 6:
            raise ValueError(f"Invalid Subhalo/SnapByType shape in {path}")
        for particle_type in particle_types:
            if particle_type < 0 or particle_type >= dataset.shape[1]:
                raise ValueError(f"Invalid particle type {particle_type}")
            output[int(particle_type)] = np.asarray(dataset[:, particle_type], dtype=np.uint64)
    return output


def subfind_counts(catalogue: dict[str, np.ndarray], group_id: int) -> dict[str, int]:
    """Return central/satellite gas/star counts from a cached catalogue."""

    first_values = np.asarray(catalogue["GroupFirstSub"], dtype=np.int64)
    nsubs_values = np.asarray(catalogue["GroupNsubs"], dtype=np.int64)
    lengths = np.asarray(catalogue["SubhaloLenType"], dtype=np.int64)
    group_id = int(group_id)
    if group_id < 0 or group_id >= len(first_values):
        raise ValueError(f"Invalid group ID {group_id}")
    first = int(first_values[group_id])
    nsubs = int(nsubs_values[group_id])
    if first < 0 or nsubs < 1 or first + nsubs > len(lengths):
        raise ValueError(f"Invalid Subfind counts for group {group_id}")
    central = lengths[first]
    satellites = lengths[first + 1:first + nsubs].sum(axis=0)
    return {
        "central_gas_count": int(central[0]),
        "satellite_gas_count": int(satellites[0]),
        "central_star_count": int(central[4]),
        "satellite_star_count": int(satellites[4]),
    }
