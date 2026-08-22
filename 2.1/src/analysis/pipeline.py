"""End-to-end Phase 2.1 orchestration over TNG50-1 data."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..io.sampling import load_or_build_sample
from ..io.state import prepare_halo_states
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


def _state_for_sample(state_record: dict, snap: int) -> dict:
    """Read a state branch regardless of JSON or in-memory integer keys."""

    states = state_record["states"]
    return states[snap] if snap in states else states[str(snap)]


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
        previous_catalog[index] = np.asarray(_state_for_sample(record, config.snap_event_prev)["central_cold_ids"], dtype=np.uint64)
        current_catalog[index] = np.asarray(_state_for_sample(record, config.snap_event_cur)["central_cold_ids"], dtype=np.uint64)
    previous_scanned = scan_parent_to_tracer(config.base_path, config.snap_event_prev, previous_catalog, cache_dir=Path(cache_dir) / "tracer", block_size=config.tracer_read_block_size)
    current_scanned = scan_parent_to_tracer(config.base_path, config.snap_event_cur, current_catalog, cache_dir=Path(cache_dir) / "tracer", block_size=config.tracer_read_block_size)
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


def _parent_products(event_ids: list[np.ndarray], config: Phase21Config, cache_dir: str | Path) -> tuple[dict[int, dict[int, int]], dict[int, dict[int, dict]]]:
    """Resolve all event tracer parents and physical records by snapshot."""

    nonempty = [np.asarray(values, dtype=np.uint64) for values in event_ids if len(values)]
    all_events = np.unique(np.concatenate(nonempty)) if nonempty else np.empty(0, dtype=np.uint64)
    parent_maps: dict[int, dict[int, int]] = {}
    records: dict[int, dict[int, dict]] = {}
    for snap in config.snapshots:
        mapping = scan_tracer_parent_map(config.base_path, snap, all_events, cache_dir=Path(cache_dir) / "tracer", block_size=config.tracer_read_block_size)
        parent_maps[snap] = mapping
        parent_ids = np.asarray(list(mapping.values()), dtype=np.uint64)
        table = lookup_parent_records(config.base_path, snap, parent_ids, cache_dir=Path(cache_dir) / "particles", block_size=config.tracer_read_block_size)
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
        records[snap] = indexed
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
    headers = _headers(config)
    dt_gyr = snapshot_interval_gyr(headers[config.snap_event_prev], headers[config.snap_event_cur], h=config.h, omega_m=config.omega_m, omega_lambda=config.omega_lambda)
    sample, sample_metadata = load_or_build_sample(config, cache_dir=cache_dir, rebuild=rebuild_sample, verbose=verbose)
    if not sample:
        raise ValueError("No Phase 2.1 sample was selected")
    halo_states, state_rejected = prepare_halo_states(sample, config=config, headers=headers, cache_dir=cache_dir, verbose=verbose)
    if not halo_states:
        raise ValueError("No valid halo states remain")
    enters, exits = _anchor_tracers(halo_states, config, cache_dir, verbose)
    parent_maps, parent_records = _parent_products([*enters, *exits], config, cache_dir)
    records = []
    ledgers = []
    for index, state_record in enumerate(halo_states):
        states = {snap: _state_for_sample(state_record, snap) for snap in config.snapshots}
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
        "tracer_mass_msun": config.tracer_mass_msun, "sample_metadata": sample_metadata,
        "state_rejected": state_rejected,
        "other_policy": "other is removed from the valid mutually-exclusive mother set but retained in totals and composition fractions",
        "agn_endpoint_policy": "average Mhot, MBH, R200c, and V200c before computing A_SAM",
        "temperature_boundary_log10_K": config.hot_log10_temperature_min,
        "fingerprint": config.fingerprint,
        "fingerprint_payload": config.fingerprint_payload,
        "schema_version": config.cache_schema_version,
    }
    return {
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
