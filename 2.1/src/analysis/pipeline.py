"""End-to-end Phase 2.1 orchestration over TNG50-1 data."""

from __future__ import annotations

import ctypes
import gc
import sys
import time
from pathlib import Path

import numpy as np

from ..io.sampling import load_or_build_sample
from ..io.catalog import load_catalogue_cache, load_subhalo_particle_offsets
from ..io.state import load_state_snapshot, prepare_halo_states
from ..io.tracer import lookup_parent_records, reuse_parent_product_subset_caches, scan_parent_to_tracer, scan_tracer_parent_map
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
EVENT_STATE_FIELDS = (
    "snap", "unresolved_snapshot", "unresolved_reason",
    "group_id", "subfind_id", "is_main_central",
    "group_center_ckpc_h", "subhalo_center_ckpc_h", "r200c_ckpc_h",
)
PHYSICS_STATE_FIELDS = (
    "snap", "r200c_pkpc", "m200c_msun", "mstar_msun", "m_bh_msun",
    "m_hot_msun", "z_hot_mass_fraction", "m_cgm_msun",
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
    scanned_by_snap = {}
    for snap, catalog in jobs:
        scanned_by_snap[snap] = scan_parent_to_tracer(
            config.base_path,
            snap,
            catalog,
            cache_dir=Path(cache_dir) / "tracer",
            block_size=config.tracer_read_block_size,
            max_workers=config.tracer_workers,
            parallel_backend=config.tracer_parallel_backend,
            verbose=verbose,
        )
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


def _parent_product_worker(
    base_path: str,
    snap: int,
    all_events: np.ndarray,
    legacy_events: np.ndarray,
    cache_dir: str | Path,
    block_size: int,
    max_workers: int,
    parallel_backend: str,
    verbose: bool,
) -> tuple[int, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Resolve one history snapshot into compact columnar parent records."""

    tracer_cache = Path(cache_dir) / "tracer"
    particle_cache = Path(cache_dir) / "particles"
    reuse_parent_product_subset_caches(
        base_path,
        snap,
        all_events,
        legacy_events,
        cache_dir=cache_dir,
        verbose=verbose,
    )
    mapping = scan_tracer_parent_map(
        base_path,
        snap,
        all_events,
        cache_dir=tracer_cache,
        block_size=block_size,
        max_workers=max_workers,
        parallel_backend=parallel_backend,
        verbose=verbose,
        columnar=True,
    )
    parent_ids = np.asarray(mapping["parent_ids"], dtype=np.uint64)
    catalogue = None
    subhalo_offsets = None
    if len(parent_ids):
        catalogue = load_catalogue_cache(
            base_path,
            (snap,),
            cache_dir=cache_dir,
            verbose=False,
        )[snap]
        subhalo_offsets = load_subhalo_particle_offsets(base_path, snap)
    table = lookup_parent_records(
        base_path,
        snap,
        parent_ids,
        cache_dir=particle_cache,
        block_size=block_size,
        max_workers=max_workers,
        parallel_backend=parallel_backend,
        verbose=verbose,
        membership_catalogue=catalogue,
        subhalo_offsets=subhalo_offsets,
    )
    # Keep the sorted NumPy table returned by ``lookup_parent_records``.
    # Expanding tens of millions of rows into nested Python dictionaries
    # multiplies memory use and makes every later full-GC pass extremely slow.
    return snap, mapping, table


def _parent_products(
    enter_ids: list[np.ndarray],
    exit_ids: list[np.ndarray],
    config: Phase21Config,
    cache_dir: str | Path,
    *,
    verbose: bool = True,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, dict[str, np.ndarray]]]:
    """Resolve all event tracer parents and physical records by snapshot."""

    enter_parts = [np.asarray(values, dtype=np.uint64) for values in enter_ids if len(values)]
    exit_parts = [np.asarray(values, dtype=np.uint64) for values in exit_ids if len(values)]
    enter_events = np.unique(np.concatenate(enter_parts)) if enter_parts else np.empty(0, dtype=np.uint64)
    exit_events = np.unique(np.concatenate(exit_parts)) if exit_parts else np.empty(0, dtype=np.uint64)
    legacy_events = np.unique(np.concatenate((enter_events, exit_events)))
    parent_maps: dict[int, dict[str, np.ndarray]] = {}
    records: dict[int, dict[str, np.ndarray]] = {}
    if not len(enter_events) and not len(exit_events):
        # Avoid spawning workers or touching full snapshots when no anchor
        # event exists.  The empty caches are still materialized by the scan
        # functions if a caller explicitly requests them.
        for snap in config.snapshots:
            parent_maps[snap] = {}
            records[snap] = {}
        return parent_maps, records

    jobs = list(config.snapshots)
    for number, snap in enumerate(jobs, start=1):
        selected = []
        if snap <= config.snap_event_cur and len(enter_events):
            selected.append(enter_events)
        if snap >= config.snap_event_prev and len(exit_events):
            selected.append(exit_events)
        all_events = (
            np.unique(np.concatenate(selected))
            if selected
            else np.empty(0, dtype=np.uint64)
        )
        snap, mapping, indexed = _parent_product_worker(
            config.base_path,
            snap,
            all_events,
            legacy_events,
            cache_dir,
            config.tracer_read_block_size,
            config.tracer_workers,
            config.tracer_parallel_backend,
            verbose,
        )
        parent_maps[snap] = mapping
        records[snap] = indexed
        if verbose:
            print(
                f"[tracer] snap={snap:03d} ({number}/{len(jobs)}) "
                f"tracers={len(mapping.get('tracer_ids', []))} parents={len(indexed.get('particle_ids', []))}",
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
        enters,
        exits,
        config,
        cache_dir,
        verbose=verbose,
    )
    records = []
    ledgers = []
    classified_events = 0
    for index, state_record in enumerate(halo_states):
        event_snaps = set()
        if len(enters[index]):
            event_snaps.update(range(config.snap_start, config.snap_event_cur + 1))
        if len(exits[index]):
            event_snaps.update(range(config.snap_event_prev, config.snap_end + 1))
        needed_snaps = event_snaps | {config.snap_event_prev, config.snap_event_cur}
        states = {}
        for snap in sorted(needed_snaps):
            fields = set(EVENT_STATE_FIELDS) if snap in event_snaps else set()
            if snap in {config.snap_event_prev, config.snap_event_cur}:
                fields.update(PHYSICS_STATE_FIELDS)
            states[snap] = _state_for_sample(
                state_record,
                snap,
                fields=tuple(sorted(fields)),
            )
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
        classified_events += len(ledger)
        del states, previous, current
        if verbose and ((index + 1) % 10 == 0 or index + 1 == len(halo_states)):
            print(
                f"[events] classified halos={index + 1}/{len(halo_states)} "
                f"events={classified_events}",
                flush=True,
            )
    # The giant snapshot-level lookup products cannot be released while the
    # event loop is active.  CPython reference counting is sufficient for the
    # small per-halo temporaries, so run one full collection only after these
    # global products become unreachable instead of scanning them per halo.
    del parent_maps, parent_records, enters, exits, halo_states
    _release_process_memory()
    _assign_agn_quartiles(records, config)
    halo_results = records_to_arrays(records)
    binned = compute_binned_statistics(halo_results, config)
    normalized = compute_all_normalized_statistics(halo_results, config)
    agn_statistics = compute_agn_quartile_statistics(halo_results, config)
    composition = compute_composition_statistics(halo_results, config)
    closure_diagnostics = {
        "max_abs_count_in_error": int(np.max(np.abs(halo_results.get("n_in_closure_error", np.array([0]))))),
        "max_abs_count_out_error": int(np.max(np.abs(halo_results.get("n_out_closure_error", np.array([0]))))),
        "max_abs_mass_in_error_msun": float(np.max(np.abs(halo_results.get("mass_in_closure_error_msun", np.array([0.0]))))),
        "max_abs_mass_out_error_msun": float(np.max(np.abs(halo_results.get("mass_out_closure_error_msun", np.array([0.0]))))),
        "max_abs_rate_in_error_msun_per_yr": float(np.max(np.abs(halo_results.get("rate_in_closure_error_msun_per_yr", np.array([0.0]))))),
        "max_abs_rate_out_error_msun_per_yr": float(np.max(np.abs(halo_results.get("rate_out_closure_error_msun_per_yr", np.array([0.0]))))),
        "max_abs_composition_in_error_msun_per_yr": float(np.max(np.abs(composition.get("in_closure_error", np.array([0.0]))))),
        "max_abs_composition_out_error_msun_per_yr": float(np.max(np.abs(composition.get("out_closure_error", np.array([0.0]))))),
    }

    def global_other_fraction(prefix: str) -> float:
        total = float(np.sum(np.asarray(halo_results[f"n_total_{prefix}"], dtype=float)))
        other = float(np.sum(np.asarray(halo_results[f"n_other_{prefix}"], dtype=float)))
        return other / total if total > 0 else np.nan

    metadata = {
        "phase": "2.1", "base_path": config.base_path, "snap_start": config.snap_start,
        "snap_event_prev": config.snap_event_prev, "snap_event_cur": config.snap_event_cur,
        "snap_end": config.snap_end, "dt_gyr": dt_gyr, "random_seed": config.random_seed,
        "selected_halo_count": len(records), "tracer_workers": config.tracer_workers,
        "tracer_parallel_backend": config.tracer_parallel_backend,
        "tracer_parallel_unit": "HDF5 snapshot chunk",
        "state_workers": config.state_workers,
        "state_parallel_backend": config.state_parallel_backend,
        "catalogue_cache": "snapshot-level immutable NPZ cache",
        "particle_binding_policy": "official Subhalo/SnapByType offsets; FoF/subhalo fuzz remains unbound",
        "tracer_mass_msun": config.tracer_mass_msun, "sample_metadata": sample_metadata,
        "state_rejected": state_rejected,
        "other_policy": "other is removed from the valid mutually-exclusive mother set but retained in totals and composition fractions",
        "agn_endpoint_policy": "average Mhot, MBH, R200c, and V200c before computing A_SAM",
        "temperature_boundary_log10_K": config.hot_log10_temperature_min,
        "fingerprint": config.fingerprint,
        "fingerprint_payload": config.fingerprint_payload,
        "schema_version": config.cache_schema_version,
        "event_sequence_windows": {"in": [config.snap_start, config.snap_event_cur], "out": [config.snap_event_prev, config.snap_end]},
        "rate_definitions": {
            "rate_supply_msun_per_yr": "rate_total_in_msun_per_yr",
            "rate_feedback_msun_per_yr": "rate_total_out_msun_per_yr",
            "rate_pf_in_msun_per_yr": "rate_first_in_eff_msun_per_yr + rate_stay_out_msun_per_yr",
        },
        "effective_counts": {
            "agn_quartile_n_finite": agn_statistics["n_finite"],
            "agn_quartile_n_assigned": agn_statistics["n_assigned"],
            "composition_in_event_count": composition["in_event_count"],
            "composition_out_event_count": composition["out_event_count"],
        },
        "global_other_fraction": {"in": global_other_fraction("in"), "out": global_other_fraction("out")},
        "closure_diagnostics": closure_diagnostics,
    }
    result = {
        "metadata": metadata, "sample": sample, "halo_results": halo_results,
        "binned_statistics": binned, "normalized_statistics": normalized,
        "agn_quartile_statistics": agn_statistics, "composition_statistics": composition,
        "closure_diagnostics": closure_diagnostics,
        "event_ledgers": ledgers,
    }
    if verbose:
        print(
            f"[phase21] complete halos={len(records)} "
            f"elapsed_s={time.perf_counter() - started:.1f}",
            flush=True,
        )
    return result
