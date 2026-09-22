"""Per-snapshot gas, particle-membership, and geometric state construction."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .catalog import load_catalogue_cache, subfind_counts
from ..utils.config import Phase21Config


GAS_FIELDS = ["ParticleIDs", "Coordinates", "Masses", "StarFormationRate", "InternalEnergy", "ElectronAbundance", "GFM_Metallicity"]
PROTON_MASS_G = 1.67262192369e-24
BOLTZMANN_CGS = 1.380649e-16


def _safe_float(value, default: float = np.nan) -> float:
    """Convert a cached scalar while preserving missing values as NaN."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_center(value) -> np.ndarray:
    """Convert a cached centre to a three-vector, or a NaN vector."""

    try:
        center = np.asarray(value, dtype=float)
        if center.shape == (3,):
            return center
    except (TypeError, ValueError):
        pass
    return np.full(3, np.nan, dtype=float)


def internal_energy_to_temperature(internal_energy: np.ndarray, electron_abundance: np.ndarray) -> np.ndarray:
    """Convert TNG internal energy to Kelvin."""

    hydrogen_fraction = 0.76
    gamma = 5.0 / 3.0
    mu = 4.0 * PROTON_MASS_G / (1.0 + 3.0 * hydrogen_fraction + 4.0 * hydrogen_fraction * np.asarray(electron_abundance, dtype=float))
    return np.asarray(internal_energy, dtype=float) * 1.0e10 * (gamma - 1.0) * mu / BOLTZMANN_CGS


def periodic_radius(coordinates: np.ndarray, center: np.ndarray, box_size: float) -> np.ndarray:
    """Return minimum-image distances in a periodic cube."""

    delta = (np.asarray(coordinates, dtype=float) - np.asarray(center, dtype=float) + 0.5 * box_size) % box_size - 0.5 * box_size
    return np.linalg.norm(delta, axis=-1)


def _ids(data: dict, key: str = "ParticleIDs") -> np.ndarray:
    """Read a typed particle-ID array."""

    return np.asarray(data.get(key, np.empty(0)), dtype=np.uint64)


def _subfind_counts(
    base_path: str,
    snap: int,
    group_id: int,
    catalogue: dict[str, np.ndarray] | None = None,
) -> dict:
    """Read central/satellite particle counts for one FoF group.

    A supplied snapshot catalogue avoids loading global groupcat arrays for
    every halo.  The fallback preserves the original standalone behavior.
    """

    if catalogue is not None:
        return subfind_counts(catalogue, group_id)

    try:
        import illustris_python as il
    except ImportError as exc:
        raise ImportError("illustris_python is required for Subfind membership") from exc
    halos = il.groupcat.loadHalos(base_path, snap, fields=["GroupFirstSub", "GroupNsubs"])
    subhalos = il.groupcat.loadSubhalos(base_path, snap, fields=["SubhaloLenType"])
    lengths = np.asarray(subhalos["SubhaloLenType"] if isinstance(subhalos, dict) else subhalos, dtype=int)
    first = int(np.asarray(halos["GroupFirstSub"])[group_id])
    nsubs = int(np.asarray(halos["GroupNsubs"])[group_id])
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


def _tracked_subhalo_layout(
    catalogue: dict[str, np.ndarray],
    group_id: int,
    subfind_id: int,
    particle_type: int,
) -> tuple[int, int, int]:
    """Return tracked-subhalo start/count and total bound count in one FoF."""

    first = int(np.asarray(catalogue["GroupFirstSub"], dtype=np.int64)[group_id])
    nsubs = int(np.asarray(catalogue["GroupNsubs"], dtype=np.int64)[group_id])
    lengths = np.asarray(catalogue["SubhaloLenType"], dtype=np.int64)
    if first < 0 or nsubs < 1 or subfind_id < first or subfind_id >= first + nsubs:
        raise ValueError(
            f"Tracked subhalo {subfind_id} is not a member of FoF group {group_id}"
        )
    member_lengths = lengths[first:first + nsubs, particle_type]
    if np.any(member_lengths < 0):
        raise ValueError("Subfind particle counts must be non-negative")
    relative = subfind_id - first
    start = int(member_lengths[:relative].sum())
    count = int(member_lengths[relative])
    total = int(member_lengths.sum())
    return start, count, total


def _empty_like(data: dict, fields: list[str]) -> dict[str, np.ndarray]:
    """Normalize empty loader output without object arrays."""

    result = {}
    for field in fields:
        if field == "Coordinates":
            result[field] = np.empty((0, 3), dtype=float)
        else:
            result[field] = np.empty(0, dtype=float)
    if isinstance(data, dict):
        for field in fields:
            if field in data:
                result[field] = np.asarray(data[field])
        lengths = []
        for field in fields:
            values = result[field]
            if field == "Coordinates":
                if values.ndim != 2 or values.shape[1] != 3:
                    raise ValueError(f"{field} must have shape (N,3)")
            lengths.append(len(values))
        if len(set(lengths)) > 1:
            raise ValueError("Snapshot particle fields have inconsistent lengths")
    return result


def build_snapshot_state(
    base_path: str,
    branch: dict,
    header: dict,
    config: Phase21Config,
    *,
    catalogue: dict[str, np.ndarray] | None = None,
) -> dict:
    """Load one branch halo and derive its central/phase reservoirs."""

    if branch.get("unresolved_snapshot", False):
        raise ValueError(str(branch.get("unresolved_reason", "unresolved MPB snapshot")))

    try:
        import illustris_python as il
    except ImportError as exc:
        raise ImportError("illustris_python is required to load snapshots") from exc
    snap = int(branch["snap"])
    group_id = int(branch["group_id"])
    gas = _empty_like(il.snapshot.loadHalo(base_path, snap, group_id, "gas", fields=GAS_FIELDS), GAS_FIELDS)
    subfind_id = int(branch["subfind_id"])
    if catalogue is None:
        halos = il.groupcat.loadHalos(base_path, snap, fields=["GroupFirstSub", "GroupNsubs"])
        subhalos = il.groupcat.loadSubhalos(base_path, snap, fields=["SubhaloLenType"])
        catalogue = {
            "GroupFirstSub": np.asarray(halos["GroupFirstSub"], dtype=np.int64),
            "GroupNsubs": np.asarray(halos["GroupNsubs"], dtype=np.int64),
            "SubhaloLenType": np.asarray(
                subhalos["SubhaloLenType"] if isinstance(subhalos, dict) else subhalos,
                dtype=np.int64,
            ),
        }
    gas_start, gas_count, gas_total_bound = _tracked_subhalo_layout(
        catalogue, group_id, subfind_id, 0
    )
    gas_ids = _ids(gas)
    # ``loadHalo`` follows Subfind ordering: central, satellites, then FoF
    # fuzz.  Constructing this mask by position is exact and avoids an
    # extremely expensive np.isin over cluster-sized particle arrays.
    diffuse = np.ones(len(gas_ids), dtype=bool)
    # Exclude every bound subhalo except the tracked MPB object.  FoF fuzz is
    # retained as diffuse gas, matching the central-reservoir definition.
    diffuse[:gas_total_bound] = False
    diffuse[gas_start:gas_start + gas_count] = True
    mass = np.asarray(gas["Masses"], dtype=float) * 1.0e10 / config.h
    temperature = internal_energy_to_temperature(gas["InternalEnergy"], gas["ElectronAbundance"])
    with np.errstate(divide="ignore", invalid="ignore"):
        log_temperature = np.log10(temperature)
    sfr = np.asarray(gas["StarFormationRate"], dtype=float)
    cold = (sfr > 0) | ((sfr <= 0) & np.isfinite(log_temperature) & (log_temperature < config.cold_log10_temperature_max))
    hot = (sfr <= 0) & np.isfinite(log_temperature) & (log_temperature >= config.hot_log10_temperature_min)
    group_center = np.asarray(branch["group_center_ckpc_h"], dtype=float)
    subhalo_center = np.asarray(branch["subhalo_center_ckpc_h"], dtype=float)
    box = float(header["BoxSize"])
    r_group = periodic_radius(gas["Coordinates"], group_center, box)
    r_subhalo = periodic_radius(gas["Coordinates"], subhalo_center, box)
    r200 = float(branch["r200c_ckpc_h"])
    # The black-hole mass is read from the MPB/group catalogue layer and is
    # carried through unchanged for the AGN endpoint calculation.  Keep the
    # value on the state record even when TNG does not provide it (NaN).
    m_bh_msun = _safe_float(branch.get("m_bh_msun"))
    inner = r_subhalo < config.central_rfrac * r200
    in_halo = r_group < r200
    central_cold = inner & cold & diffuse
    hot_reservoir = in_halo & hot & diffuse
    m_hot = float(mass[hot_reservoir].sum())
    metallicity = np.asarray(gas["GFM_Metallicity"], dtype=float)
    z_hot = float(np.sum(mass[hot_reservoir] * metallicity[hot_reservoir]) / m_hot) if m_hot > 0 else np.nan
    r200_pkpc = r200 * float(header["Time"]) / config.h
    return {
        "snap": snap,
        "group_id": group_id,
        "subfind_id": subfind_id,
        "group_center_ckpc_h": group_center,
        "subhalo_center_ckpc_h": subhalo_center,
        "r200c_ckpc_h": r200,
        "r200c_pkpc": r200_pkpc,
        "m200c_msun": float(branch["m200c_msun"]),
        "mstar_msun": float(branch["mstar_msun"]),
        "vmax_km_s": float(branch.get("vmax_km_s", np.nan)),
        "m_bh_msun": m_bh_msun,
        "is_main_central": bool(branch.get("is_main_central", False)),
        # Full membership arrays are deliberately not cached.  Event parents
        # are classified exactly from their global particle index and the TNG
        # Subhalo/SnapByType offsets, which avoids multi-GiB per-halo caches.
        "central_gas_ids": np.empty(0, dtype=np.uint64),
        "satellite_gas_ids": np.empty(0, dtype=np.uint64),
        "other_gas_ids": np.empty(0, dtype=np.uint64),
        "central_star_ids": np.empty(0, dtype=np.uint64),
        "satellite_star_ids": np.empty(0, dtype=np.uint64),
        "other_star_ids": np.empty(0, dtype=np.uint64),
        # Only the two event anchors consume the complete cold-library IDs.
        # Historical classifications use the selected tracer parents instead.
        "central_cold_ids": (
            np.sort(gas_ids[central_cold])
            if snap in {config.snap_event_prev, config.snap_event_cur}
            else np.empty(0, dtype=np.uint64)
        ),
        "m_hot_msun": m_hot,
        "z_hot_mass_fraction": z_hot,
        "m_cgm_msun": float(mass[(~inner) & in_halo & diffuse].sum()),
    }


def _state_cache_path(cache_dir: str | Path, sample: dict, config: Phase21Config | None = None) -> Path:
    """Return one deterministic per-halo state path."""

    payload = {"schema": 1, "sample": sample["subfind_id_z99"]}
    if config is not None:
        payload["fingerprint"] = config.fingerprint
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return Path(cache_dir) / "states" / f"sub{int(sample['subfind_id_z99'])}_{digest}.npz"


def _save_state(path: Path, states: dict[int, dict]) -> None:
    """Atomically save nested snapshot state arrays."""

    path.parent.mkdir(parents=True, exist_ok=True)
    values = {}
    for snap, state in states.items():
        for key, value in state.items():
            values[f"snap{snap:03d}__{key}"] = np.asarray(value)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **values)
    temporary.replace(path)


def _load_state(path: Path, expected_snaps: tuple[int, ...] | None = None) -> dict[int, dict] | None:
    """Load a state cache, returning None on corruption."""

    if not path.is_file():
        return None
    try:
        output: dict[int, dict] = {}
        with np.load(path, allow_pickle=False) as data:
            for name in data.files:
                prefix, key = name.split("__", 1)
                snap = int(prefix[4:])
                value = np.asarray(data[name])
                output.setdefault(snap, {})[key] = value.item() if value.ndim == 0 else value
        if expected_snaps is not None and any(snap not in output for snap in expected_snaps):
            return None
        return output
    except Exception:
        return None


def state_cache_is_valid(path: str | Path, expected_snaps: tuple[int, ...] | None = None) -> bool:
    """Validate a state NPZ from its directory without loading its arrays."""

    path = Path(path)
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            names = tuple(data.files)
        if not names or any("__" not in name for name in names):
            return False
        if expected_snaps is not None:
            prefixes = {name.split("__", 1)[0] for name in names}
            if any(f"snap{int(snap):03d}" not in prefixes for snap in expected_snaps):
                return False
        return True
    except Exception:
        return False


def load_state_snapshot(
    path: str | Path,
    snap: int,
    *,
    fields: tuple[str, ...] | None = None,
) -> dict:
    """Load one snapshot, optionally selecting fields, from a halo state NPZ.

    NPZ members are loaded lazily by NumPy.  Selecting one snapshot and a
    small field set therefore avoids materializing the complete multi-halo
    state cache in memory.
    """

    path = Path(path)
    prefix = f"snap{int(snap):03d}__"
    selected = set(fields) if fields is not None else None
    output = {}
    with np.load(path, allow_pickle=False) as data:
        for name in data.files:
            if not name.startswith(prefix):
                continue
            key = name[len(prefix):]
            if selected is not None and key not in selected:
                continue
            value = np.asarray(data[name])
            output[key] = value.item() if value.ndim == 0 else value
    if not output:
        raise ValueError(f"State cache has no requested fields for snap {int(snap)}: {path}")
    return output


def _unresolved_state(branch: dict, snap: int, exc: Exception) -> dict:
    """Build the typed placeholder used for missing non-anchor snapshots."""

    return {
        "snap": snap,
        "unresolved_snapshot": True,
        "unresolved_reason": f"{type(exc).__name__}: {exc}",
        "group_center_ckpc_h": _safe_center(branch.get("group_center_ckpc_h")),
        "subhalo_center_ckpc_h": _safe_center(branch.get("subhalo_center_ckpc_h")),
        "r200c_ckpc_h": _safe_float(branch.get("r200c_ckpc_h")),
        "r200c_pkpc": np.nan,
        "m200c_msun": _safe_float(branch.get("m200c_msun")),
        "mstar_msun": _safe_float(branch.get("mstar_msun")),
        "vmax_km_s": np.nan,
        "m_bh_msun": np.nan,
        "central_cold_ids": np.empty(0, dtype=np.uint64),
        "central_gas_ids": np.empty(0, dtype=np.uint64),
        "satellite_gas_ids": np.empty(0, dtype=np.uint64),
        "other_gas_ids": np.empty(0, dtype=np.uint64),
        "central_star_ids": np.empty(0, dtype=np.uint64),
        "satellite_star_ids": np.empty(0, dtype=np.uint64),
        "other_star_ids": np.empty(0, dtype=np.uint64),
    }


def _build_halo_states(
    record: dict,
    config: Phase21Config,
    headers: dict[int, dict],
    catalogues: dict[int, dict[str, np.ndarray]] | None,
) -> dict[int, dict]:
    """Build one halo's complete state history in one worker."""

    states: dict[int, dict] = {}
    for snap in config.snapshots:
        branch = record["mpb"][str(snap)] if str(snap) in record["mpb"] else record["mpb"][snap]
        try:
            states[snap] = build_snapshot_state(
                config.base_path,
                branch,
                headers[snap],
                config,
                catalogue=catalogues.get(snap) if catalogues is not None else None,
            )
        except Exception as exc:
            if snap in {config.snap_event_prev, config.snap_event_cur}:
                raise
            states[snap] = _unresolved_state(branch, snap, exc)
    return states


_PROCESS_CATALOGUES: dict[int, dict[str, np.ndarray]] | None = None


def _init_process_state_worker(
    base_path: str,
    snapshots: tuple[int, ...],
    cache_dir: str | Path,
) -> None:
    """Load catalogue NPZ caches once in each process worker."""

    global _PROCESS_CATALOGUES
    _PROCESS_CATALOGUES = load_catalogue_cache(
        base_path,
        snapshots,
        cache_dir=cache_dir,
        verbose=False,
    )


def _process_state_worker(args):
    """Pickle-safe process worker for one halo."""

    record, config, headers = args
    return _build_halo_states(record, config, headers, _PROCESS_CATALOGUES)


def prepare_halo_states(
    sample: list[dict],
    *,
    config: Phase21Config,
    headers: dict[int, dict],
    cache_dir: str | Path,
    catalogues: dict[int, dict[str, np.ndarray]] | None = None,
    verbose: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Load or build all per-halo snap90--99 states.

    Cached halos are returned immediately.  Pending halos can be processed in
    serial mode, with shared catalogue data in a thread pool, or with a
    process pool whose workers load the on-disk catalogue caches once during
    initialization.  The default is a four-worker shared thread pool; set
    ``state_workers=1`` and ``state_parallel_backend='serial'`` on filesystems
    where concurrent HDF5 reads are undesirable.
    """

    cache_dir = Path(cache_dir)
    state_paths_by_index: list[Path | None] = [None] * len(sample)
    pending: list[tuple[int, dict, Path]] = []
    for index, record in enumerate(sample):
        path = _state_cache_path(cache_dir, record, config)
        if not state_cache_is_valid(path, config.snapshots):
            pending.append((index, record, path))
        else:
            state_paths_by_index[index] = path
            if verbose:
                print(
                    f"[states] cached {index + 1}/{len(sample)} "
                    f"sub={record['subfind_id_z99']}",
                    flush=True,
                )

    if verbose:
        print(
            f"[states] cache check complete: "
            f"cached={len(sample) - len(pending)} "
            f"pending={len(pending)} total={len(sample)}",
            flush=True,
        )

    # Do not load the potentially large catalogue arrays when every halo state
    # is already cached.  This makes repeated analysis/statistics runs cheap.
    if catalogues is None and pending:
        catalogues = load_catalogue_cache(
            config.base_path,
            config.snapshots,
            cache_dir=cache_dir,
            verbose=verbose,
        )

    rejected: list[dict] = []

    def accept_result(
        sequence: int,
        index: int,
        record: dict,
        path: Path,
        states: dict[int, dict] | None,
        error: Exception | None = None,
    ) -> None:
        if error is not None or states is None:
            rejected.append({
                "subfind_id_z99": record["subfind_id_z99"],
                "reason": f"{type(error).__name__}: {error}",
            })
        else:
            _save_state(path, states)
            state_paths_by_index[index] = path
        if verbose:
            status = "rejected" if error is not None or states is None else "computed"
            detail = ""
            if error is not None:
                detail = f" reason={type(error).__name__}: {error}"
            print(
                f"[states] {status} {sequence}/{len(pending)} "
                f"sub={record['subfind_id_z99']}{detail}",
                flush=True,
            )

    serial = config.state_parallel_backend == "serial" or config.state_workers == 1
    if serial:
        for sequence, (index, record, path) in enumerate(pending, start=1):
            try:
                states = _build_halo_states(record, config, headers, catalogues)
                accept_result(sequence, index, record, path, states)
            except Exception as exc:
                accept_result(sequence, index, record, path, None, exc)
    elif config.state_parallel_backend == "thread":
        with ThreadPoolExecutor(max_workers=config.state_workers) as executor:
            future_map = {
                executor.submit(_build_halo_states, record, config, headers, catalogues): (
                    sequence,
                    index,
                    record,
                    path,
                )
                for sequence, (index, record, path) in enumerate(pending, start=1)
            }
            completed = 0
            for future in as_completed(future_map):
                sequence, index, record, path = future_map.pop(future)
                completed += 1
                try:
                    states = future.result()
                    accept_result(completed, index, record, path, states)
                except Exception as exc:
                    accept_result(completed, index, record, path, None, exc)
    else:
        with ProcessPoolExecutor(
            max_workers=config.state_workers,
            initializer=_init_process_state_worker,
            initargs=(config.base_path, config.snapshots, cache_dir),
        ) as executor:
            future_map = {
                executor.submit(
                    _process_state_worker,
                    (record, config, headers),
                ): (sequence, index, record, path)
                for sequence, (index, record, path) in enumerate(pending, start=1)
            }
            completed = 0
            for future in as_completed(future_map):
                sequence, index, record, path = future_map.pop(future)
                completed += 1
                try:
                    states = future.result()
                    accept_result(completed, index, record, path, states)
                except Exception as exc:
                    accept_result(completed, index, record, path, None, exc)

    valid = [
        {"sample": sample[index], "state_cache_path": str(state_paths_by_index[index])}
        for index in range(len(sample))
        if state_paths_by_index[index] is not None
    ]
    return valid, rejected
