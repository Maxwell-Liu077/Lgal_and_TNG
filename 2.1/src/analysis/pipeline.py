"""End-to-end Phase 2.1 orchestration over TNG50-1 data."""

from __future__ import annotations

import ctypes
import gc
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ..io.sampling import load_or_build_sample
from ..io.state import load_state_snapshot, prepare_halo_states
from ..io.tracer import lookup_parent_records, scan_parent_to_tracer, scan_tracer_parent_map
from ..physics.feedback import compute_agn_strength
from ..physics.sam_cooling import compute_isothermal_sam_cooling
from ..utils.arrays import records_to_arrays
from ..utils.config import Phase21Config
from ..utils.cosmology import snapshot_interval_gyr
from .events import classify_halo_events
from .statistics import compute_agn_quartile_statistics, compute_all_normalized_statistics, compute_binned_statistics, compute_composition_statistics


def _headers(config: Phase21Config) -> dict[int, dict]:
    """Load all required snapshot headers through illustris_python."""

    try:
        import illustris_python as il
    except ImportError as exc:
        raise ImportError("illustris_python is required for Phase 2.1") from exc
    return {snap: il.groupcat.loadHeader(config.base_path, snap) for snap in config.snapshots}


ANCHOR_STATE_FIELDS = ("central_cold_ids",)
ANALYSIS_STATE_FIELDS = (
    "snap", "unresolved_snapshot", "unresolved_reason",
    "group_center_ckpc_h", "subhalo_center_ckpc_h", "r200c_ckpc_h",
    "r200c_pkpc", "m200c_msun", "mstar_msun", "m_bh_msun",
    "m_hot_msun", "z_hot_mass_fraction", "m_cgm_msun",
    "central_gas_ids", "satellite_gas_ids", "other_gas_ids",
    "central_star_ids", "satellite_star_ids", "other_star_ids",
)


def _state_for_sample(
    state_record: dict,
    snap: int,
    *,
    fields: tuple[str, ...] | None = None,
) -> dict:
    """Read one state branch from memory or a lazy per-halo cache reference."""

    if "states" in state_record:
        states = state_record["states"]
        state = states[snap] if snap in states else states[str(snap)]
        if fields is None:
            return state
        return {key: state[key] for key in fields if key in state}
    return load_state_snapshot(state_record["state_cache_path"], snap, fields=fields)


def _release_process_memory() -> None:
    """Release unreachable arrays and return free glibc arenas on Linux."""

    gc.collect()
    if not sys.platform.startswith("linux"):
        return
    try:
        trim = ctypes.CDLL(None).malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        trim(0)
    except (AttributeError, OSError):
        pass


def _assign_agn_quartiles(records: list[dict], config: Phase21Config) -> None:
    """Assign stable near-equal Q1--Q4 labels within each mass bin."""

    for record in records:
        record["agn_quartile"] = 0
    for bin_index in range(len(config.mass_bin_centers)):
        positions = [index for index, record in enumerate(records) if record["mass_bin_index"] == bin_index and not np.isnan(record["agn_strength"])]
        positions.sort(key=lambda index: (records[index]["agn_strength"], records[index]["subfind_id_z99"]))
        for quartile, group in enumerate(np.array_split(np.asarray(positions, dtype=int), 4), start=1):
            for position in group:
                records[int(position)]["agn_quartile"] = quartile


def _anchor_tracers(states: list[dict], config: Phase21Config, cache_dir: str | Path, verbose: bool) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Scan central-cold parent IDs at both event endpoints."""

    previous_catalog = {}
    current_catalog = {}
    for index, record in enumerate(states):
        previous_catalog[index] = np.asarray(
            _state_for_sample(record, config.snap_event_prev, fields=ANCHOR_STATE_FIELDS)["central_cold_ids"],
            dtype=np.uint64,
        )
        current_catalog[index] = np.asarray(
            _state_for_sample(record, config.snap_event_cur, fields=ANCHOR_STATE_FIELDS)["central_cold_ids"],
            dtype=np.uint64,
        )
    jobs = (
        (config.snap_event_prev, previous_catalog),
        (config.snap_event_cur, current_catalog),
    )
    if verbose:
        print("[events] scanning parent->tracer snap94 and snap95", flush=True)
    if not any(len(ids) for _, catalog in jobs for ids in catalog.values()):
        scanned_by_snap = {snap: {} for snap, _ in jobs}
    elif config.tracer_parallel_backend == "process" and config.tracer_workers > 1:
        with ProcessPoolExecutor(max_workers=min(2, config.tracer_workers)) as executor:
            futures = {
                executor.submit(
                    _scan_parent_to_tracer_worker,
                    config.base_path,
                    snap,
                    catalog,
                    Path(cache_dir) / "tracer",
                    config.tracer_read_block_size,
                ): snap
                for snap, catalog in jobs
            }
            scanned_by_snap = {}
            for future, snap in futures.items():
                scanned_by_snap[snap] = future.result()
    else:
        scanned_by_snap = {
            snap: scan_parent_to_tracer(
                config.base_path,
                snap,
                catalog,
                cache_dir=Path(cache_dir) / "tracer",
                block_size=config.tracer_read_block_size,
            )
            for snap, catalog in jobs
        }
    previous_scanned = scanned_by_snap[config.snap_event_prev]
    current_scanned = scanned_by_snap[config.snap_event_cur]
    enters = []
    exits = []
    for index in range(len(states)):
        previous = np.unique(previous_scanned.get(index, np.empty(0, dtype=np.uint64)))
        current = np.unique(current_scanned.get(index, np.empty(0, dtype=np.uint64)))
        enters.append(np.setdiff1d(current, previous, assume_unique=True))
        exits.append(np.setdiff1d(previous, current, assume_unique=True))
        if verbose:
            print(f"[events] halo={index} enter={len(enters[-1])} exit={len(exits[-1])}")
    return enters, exits


def _scan_parent_to_tracer_worker(
    base_path: str,
    snap: int,
    parent_ids_by_label: dict[int, np.ndarray],
    cache_dir: str | Path,
    block_size: int,
) -> dict[int, np.ndarray]:
    """Process-safe worker for one endpoint ParentID→TracerID scan."""

    return scan_parent_to_tracer(
        base_path,
        snap,
        parent_ids_by_label,
        cache_dir=cache_dir,
        block_size=block_size,
    )


def _parent_product_worker(
    base_path: str,
    snap: int,
    all_events: np.ndarray,
    cache_dir: str | Path,
    block_size: int,
) -> tuple[int, dict[int, int], dict[int, dict]]:
    """Process-safe worker for one history snapshot's tracer products."""

    tracer_cache = Path(cache_dir) / "tracer"
    particle_cache = Path(cache_dir) / "particles"
    mapping = scan_tracer_parent_map(
        base_path,
        snap,
        all_events,
        cache_dir=tracer_cache,
        block_size=block_size,
    )
    parent_ids = np.asarray(list(mapping.values()), dtype=np.uint64)
    table = lookup_parent_records(
        base_path,
        snap,
        parent_ids,
        cache_dir=particle_cache,
        block_size=block_size,
    )
    indexed = {}
    for index, parent_id in enumerate(np.asarray(table.get("particle_ids", []), dtype=np.uint64)):
        indexed[int(parent_id)] = {
            "particle_id": int(parent_id),
            "particle_type": int(table["particle_type"][index]),
            "coordinates": table["coordinates"][index],
            "sfr": float(table["sfr"][index]),
            "internal_energy": float(table["internal_energy"][index]),
            "electron_abundance": float(table["electron_abundance"][index]),
            "formation_time": float(table["formation_time"][index]),
        }
    return snap, mapping, indexed


def _parent_products(
    event_ids: list[np.ndarray],
    config: Phase21Config,
    cache_dir: str | Path,
    *,
    verbose: bool = True,
) -> tuple[dict[int, dict[int, int]], dict[int, dict[int, dict]]]:
    """Resolve all event tracer parents and physical records by snapshot."""

    nonempty = [np.asarray(values, dtype=np.uint64) for values in event_ids if len(values)]
    all_events = np.unique(np.concatenate(nonempty)) if nonempty else np.empty(0, dtype=np.uint64)
    parent_maps: dict[int, dict[int, int]] = {}
    records: dict[int, dict[int, dict]] = {}
    if not len(all_events):
        # Avoid spawning workers or touching full snapshots when no anchor
        # event exists.  The empty caches are still materialized by the scan
        # functions if a caller explicitly requests them.
        for snap in config.snapshots:
            parent_maps[snap] = {}
            records[snap] = {}
        return parent_maps, records

    jobs = list(config.snapshots)
    if config.tracer_parallel_backend == "process" and config.tracer_workers > 1:
        max_workers = min(config.tracer_workers, len(jobs))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _parent_product_worker,
                    config.base_path,
                    snap,
                    all_events,
                    cache_dir,
                    config.tracer_read_block_size,
                ): snap
                for snap in jobs
            }
            completed = 0
            for future in as_completed(futures):
                snap, mapping, indexed = future.result()
                parent_maps[snap] = mapping
                records[snap] = indexed
                completed += 1
                if verbose:
                    print(
                        f"[tracer] snap={snap:03d} ({completed}/{len(jobs)}) "
                        f"tracers={len(mapping)} parents={len(indexed)}",
                        flush=True,
                    )
    else:
        for number, snap in enumerate(jobs, start=1):
            snap, mapping, indexed = _parent_product_worker(
                config.base_path,
                snap,
                all_events,
                cache_dir,
                config.tracer_read_block_size,
            )
            parent_maps[snap] = mapping
            records[snap] = indexed
            if verbose:
                print(
                    f"[tracer] snap={snap:03d} ({number}/{len(jobs)}) "
                    f"tracers={len(mapping)} parents={len(indexed)}",
                    flush=True,
                )
    return parent_maps, records


def _mean_state_inputs(states: dict[int, dict], config: Phase21Config) -> dict:
    """Build endpoint-mean masses and radii for normalized results and AGN."""

    previous = states[config.snap_event_prev]
    current = states[config.snap_event_cur]
    return {
        "mstar_mean_msun": 0.5 * (float(previous["mstar_msun"]) + float(current["mstar_msun"])),
        "m_hot_mean_msun": 0.5 * (float(previous["m_hot_msun"]) + float(current["m_hot_msun"])),
        "m_cgm_mean_msun": 0.5 * (float(previous.get("m_cgm_msun", np.nan)) + float(current.get("m_cgm_msun", np.nan))),
        "m200c_mean_msun": 0.5 * (float(previous["m200c_msun"]) + float(current["m200c_msun"])),
    }


def run_phase21(
    cooling_function,
    *,
    config: Phase21Config | None = None,
    cache_dir: str | Path = "data/interim/cache",
    rebuild_sample: bool = False,
    verbose: bool = True,
) -> dict:
    """Run sample selection, state scans, event classification, and statistics."""

    config = config or Phase21Config()
    started = time.perf_counter()
    headers = _headers(config)
    dt_gyr = snapshot_interval_gyr(headers[config.snap_event_prev], headers[config.snap_event_cur], h=config.h, omega_m=config.omega_m, omega_lambda=config.omega_lambda)
    if verbose:
        print("[phase21] building/loading snap99 sample", flush=True)
    sample, sample_metadata = load_or_build_sample(
        config,
        cache_dir=cache_dir,
        rebuild=rebuild_sample,
        verbose=verbose,
    )
    if not sample:
        raise ValueError("No Phase 2.1 sample was selected")
    if verbose:
        print("[phase21] building/loading per-halo states", flush=True)
    halo_states, state_rejected = prepare_halo_states(
        sample,
        config=config,
        headers=headers,
        cache_dir=cache_dir,
        verbose=verbose,
    )
    if not halo_states:
        raise ValueError("No valid halo states remain")
    if verbose:
        print("[phase21] resolving event-anchor tracers", flush=True)
    enters, exits = _anchor_tracers(halo_states, config, cache_dir, verbose)
    if verbose:
        print("[phase21] resolving tracer histories", flush=True)
    parent_maps, parent_records = _parent_products(
        [*enters, *exits],
        config,
        cache_dir,
        verbose=verbose,
    )
    records = []
    ledgers = []
    for index, state_record in enumerate(halo_states):
        states = {
            snap: _state_for_sample(state_record, snap, fields=ANALYSIS_STATE_FIELDS)
            for snap in config.snapshots
        }
        event_result, ledger = classify_halo_events(enters[index], exits[index], parent_maps=parent_maps, records=parent_records, states=states, headers=headers, config=config, dt_gyr=dt_gyr)
        previous = states[config.snap_event_prev]
        current = states[config.snap_event_cur]
        sam_previous = compute_isothermal_sam_cooling(m_hot_msun=float(previous["m_hot_msun"]), z_hot_mass_fraction=float(previous["z_hot_mass_fraction"]), m200c_msun=float(previous["m200c_msun"]), r200c_pkpc=float(previous["r200c_pkpc"]), cooling_function=cooling_function, mean_molecular_weight=config.mean_molecular_weight, virial_temperature_factor=config.virial_temperature_factor)
        sam_current = compute_isothermal_sam_cooling(m_hot_msun=float(current["m_hot_msun"]), z_hot_mass_fraction=float(current["z_hot_mass_fraction"]), m200c_msun=float(current["m200c_msun"]), r200c_pkpc=float(current["r200c_pkpc"]), cooling_function=cooling_function, mean_molecular_weight=config.mean_molecular_weight, virial_temperature_factor=config.virial_temperature_factor)
        mean = _mean_state_inputs(states, config)
        mean.update({
            "m_bh_mean_msun": 0.5 * (float(previous.get("m_bh_msun", np.nan)) + float(current.get("m_bh_msun", np.nan))),
            "r200c_mean_pkpc": 0.5 * (float(previous["r200c_pkpc"]) + float(current["r200c_pkpc"])),
            "v200c_mean_km_s": 0.5 * (float(sam_previous["v200c_km_s"]) + float(sam_current["v200c_km_s"])),
        })
        agn = compute_agn_strength(m_hot_msun=mean["m_hot_mean_msun"], m_bh_msun=mean["m_bh_mean_msun"], r200c_pkpc=mean["r200c_mean_pkpc"], v200c_km_s=mean["v200c_mean_km_s"])
        record = {
            "sample_index": int(state_record["sample"]["sample_index"]),
            "subfind_id_z99": int(state_record["sample"]["subfind_id_z99"]),
            "mass_bin_index": int(state_record["sample"]["mass_bin_index"]),
            "mass_bin_low": float(state_record["sample"]["mass_bin_low"]),
            "mass_bin_high": float(state_record["sample"]["mass_bin_high"]),
            "mass_bin_center": float(state_record["sample"]["mass_bin_center"]),
            "log_mstar_z99": float(state_record["sample"]["log_mstar_z99"]),
            "dt_gyr": float(dt_gyr),
            "x_cool_prev": float(sam_previous["r_cool_isothermal_over_r200c"]),
            "x_cool_cur": float(sam_current["r_cool_isothermal_over_r200c"]),
            "x_cool_mean": 0.5 * (float(sam_previous["r_cool_isothermal_over_r200c"]) + float(sam_current["r_cool_isothermal_over_r200c"])),
            "rate_sam_isothermal_msun_per_yr": 0.5 * (float(sam_previous["rate_sam_isothermal_msun_per_yr"]) + float(sam_current["rate_sam_isothermal_msun_per_yr"])),
            "rate_cool_iso_msun_per_yr": 0.5 * (float(sam_previous["rate_sam_isothermal_msun_per_yr"]) + float(sam_current["rate_sam_isothermal_msun_per_yr"])),
            "agn_strength": float(agn["agn_strength"]),
            "A_SAM": float(agn["A_SAM"]),
            "mdot_bh_msun_per_yr": float(agn["mdot_bh_msun_per_yr"]),
            "mdot_heat_h15_msun_per_yr": float(agn["mdot_heat_h15_msun_per_yr"]),
            "t_dyn_gyr": float(agn["t_dyn_gyr"]),
            **mean,
            **event_result,
        }
        records.append(record)
        ledgers.append(ledger)
        # Raw membership arrays can be tens of GiB for a cluster-sized FoF.
        # They are no longer retained by ``halo_states`` and can be released
        # as soon as this halo's scalar result and compact ledger are ready.
        del states, previous, current
        _release_process_memory()
    _assign_agn_quartiles(records, config)
    halo_results = records_to_arrays(records)
    binned = compute_binned_statistics(halo_results, config)
    normalized = compute_all_normalized_statistics(halo_results, config)
    agn_statistics = compute_agn_quartile_statistics(halo_results, config)
    composition = compute_composition_statistics(halo_results, config)
    metadata = {
        "phase": "2.1", "base_path": config.base_path, "snap_start": config.snap_start,
        "snap_event_prev": config.snap_event_prev, "snap_event_cur": config.snap_event_cur,
        "snap_end": config.snap_end, "dt_gyr": dt_gyr, "random_seed": config.random_seed,
        "selected_halo_count": len(records), "tracer_workers": config.tracer_workers,
        "state_workers": config.state_workers,
        "state_parallel_backend": config.state_parallel_backend,
        "catalogue_cache": "snapshot-level immutable NPZ cache",
        "tracer_mass_msun": config.tracer_mass_msun, "sample_metadata": sample_metadata,
        "state_rejected": state_rejected,
        "other_policy": "other is removed from the valid mutually-exclusive mother set but retained in totals and composition fractions",
        "agn_endpoint_policy": "average Mhot, MBH, R200c, and V200c before computing A_SAM",
        "temperature_boundary_log10_K": config.hot_log10_temperature_min,
        "fingerprint": config.fingerprint,
        "fingerprint_payload": config.fingerprint_payload,
        "schema_version": config.cache_schema_version,
    }
    result = {
        "metadata": metadata, "sample": sample, "halo_results": halo_results,
        "binned_statistics": binned, "normalized_statistics": normalized,
        "agn_quartile_statistics": agn_statistics, "composition_statistics": composition,
        "closure_diagnostics": {
            "max_abs_count_in_error": int(np.max(np.abs(halo_results.get("n_in_closure_error", np.array([0]))))),
            "max_abs_count_out_error": int(np.max(np.abs(halo_results.get("n_out_closure_error", np.array([0]))))),
            "max_abs_mass_in_error_msun": float(np.max(np.abs(halo_results.get("mass_in_closure_error_msun", np.array([0.0]))))),
            "max_abs_mass_out_error_msun": float(np.max(np.abs(halo_results.get("mass_out_closure_error_msun", np.array([0.0]))))),
            "max_abs_rate_in_error_msun_per_yr": float(np.max(np.abs(halo_results.get("rate_in_closure_error_msun_per_yr", np.array([0.0]))))),
            "max_abs_rate_out_error_msun_per_yr": float(np.max(np.abs(halo_results.get("rate_out_closure_error_msun_per_yr", np.array([0.0]))))),
        },
        "event_ledgers": ledgers,
    }
    if verbose:
        print(
            f"[phase21] complete halos={len(records)} "
            f"elapsed_s={time.perf_counter() - started:.1f}",
            flush=True,
        )
    return result
