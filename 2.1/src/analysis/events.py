"""Central-cold event classification and per-halo rate closure."""

from __future__ import annotations

from collections import Counter

import numpy as np

from ..io.state import internal_energy_to_temperature, periodic_radius
from ..utils.config import Phase21Config


ENTER_CLASSES = ("first-in", "recycled-in-a", "stay-in", "single-in", "recycled-in-b", "other")
EXIT_CLASSES = ("stay-out", "recycled-out", "other")


def _record_index(records: dict[str, np.ndarray]) -> dict[int, int]:
    """Index a sorted-or-unsorted parent record table by particle ID."""

    return {int(value): index for index, value in enumerate(np.asarray(records.get("particle_ids", []), dtype=np.uint64))}


def _contains_particle_id(state: dict, key: str, particle_id: int) -> bool:
    """Test membership with one in-place sort instead of a giant Python set.

    Cluster-sized Subfind membership arrays can contain hundreds of millions
    of IDs.  Turning them into Python sets for every tracer multiplies memory
    consumption.  A cached sorted ndarray keeps the representation at eight
    bytes per ID and makes subsequent lookups logarithmic.
    """

    cache_key = f"__sorted_{key}"
    values = state.get(cache_key)
    if values is None:
        values = np.asarray(state.get(key, []), dtype=np.uint64)
        if len(values) > 1:
            values.sort(kind="quicksort")
        state[cache_key] = values
    if not len(values):
        return False
    position = int(np.searchsorted(values, np.uint64(particle_id)))
    return position < len(values) and int(values[position]) == int(particle_id)


def classify_parent_state(record: dict[str, object], state: dict, header: dict, config: Phase21Config) -> dict[str, str | bool]:
    """Assign the four orthogonal README state fields to one parent."""

    if state.get("unresolved_snapshot", False):
        return {"host_state": "unresolved", "radial_state": "unresolved", "phase_state": "unresolved", "carrier_state": "unresolved", "central_cold": False, "valid": False, "resolved": False, "reason": str(state.get("unresolved_reason", "unresolved_snapshot"))}
    particle_type = int(record.get("particle_type", -1))
    particle_id = int(record.get("particle_id", -1))
    if particle_type == 5:
        carrier = "black_hole"
    elif particle_type == 4:
        formation_time = float(record.get("formation_time", np.nan))
        if not np.isfinite(formation_time):
            return {"host_state": "unresolved", "radial_state": "unresolved", "phase_state": "unresolved", "carrier_state": "unresolved", "central_cold": False, "valid": False, "resolved": False, "reason": "missing_formation_time"}
        carrier = "wind" if formation_time <= 0 else "star"
    elif particle_type == 0:
        carrier = "gas"
    else:
        return {"host_state": "unresolved", "radial_state": "unresolved", "phase_state": "unresolved", "carrier_state": "unresolved", "central_cold": False, "valid": False, "resolved": False, "reason": "unknown_carrier"}
    coordinates = np.asarray(record.get("coordinates", [np.nan, np.nan, np.nan]), dtype=float)
    if coordinates.shape != (3,) or not np.all(np.isfinite(coordinates)):
        return {"host_state": "unresolved", "radial_state": "unresolved", "phase_state": "unresolved", "carrier_state": carrier, "central_cold": False, "valid": False, "resolved": False, "reason": "missing_coordinates"}
    try:
        group_center = np.asarray(state["group_center_ckpc_h"], dtype=float)
        subhalo_center = np.asarray(state["subhalo_center_ckpc_h"], dtype=float)
        r200 = float(state["r200c_ckpc_h"])
        box = float(header["BoxSize"])
    except (KeyError, TypeError, ValueError):
        return {"host_state": "unresolved", "radial_state": "unresolved", "phase_state": "unresolved", "carrier_state": carrier, "central_cold": False, "valid": False, "resolved": False, "reason": "missing_geometry"}
    if group_center.shape != (3,) or subhalo_center.shape != (3,) or not np.all(np.isfinite(group_center)) or not np.all(np.isfinite(subhalo_center)) or not np.isfinite(r200) or r200 <= 0 or not np.isfinite(box) or box <= 0:
        return {"host_state": "unresolved", "radial_state": "unresolved", "phase_state": "unresolved", "carrier_state": carrier, "central_cold": False, "valid": False, "resolved": False, "reason": "missing_geometry"}
    group_radius = float(periodic_radius(coordinates[None, :], group_center, box)[0])
    central_radius = float(periodic_radius(coordinates[None, :], subhalo_center, box)[0])
    # README section 4.1 defines the central region first.  A tracked subhalo
    # close to a FoF boundary must therefore remain ``inner`` when measured
    # around SubhaloPos, even if its GroupPos distance exceeds R200c.
    radial = "inner" if central_radius < config.central_rfrac * r200 else ("outer_halo" if group_radius < r200 else "beyond_r200c")
    bound_subhalo = int(record.get("bound_subhalo_id", -2))
    bound_group = int(record.get("bound_group_id", -2))
    if bound_subhalo >= -1 and bound_group >= -1:
        if bound_subhalo < 0:
            host = "unbound"
        elif bound_subhalo == int(state.get("subfind_id", -999999)):
            host = "main_central"
        elif bound_group == int(state.get("group_id", -999999)):
            host = "satellite"
        else:
            host = "other_halo"
    else:
        # Backward-compatible fallback for synthetic records and old direct
        # callers.  Production records always carry official offset binding.
        suffix = "gas_ids" if particle_type == 0 else "star_ids"
        if _contains_particle_id(state, f"central_{suffix}", particle_id):
            host = "main_central"
        elif _contains_particle_id(state, f"satellite_{suffix}", particle_id):
            host = "satellite"
        elif _contains_particle_id(state, f"other_{suffix}", particle_id):
            host = "other_halo"
        elif particle_type == 0:
            host = "unbound"
        else:
            host = "other_halo"
    phase = "unresolved"
    if particle_type == 0:
        temperature = float(internal_energy_to_temperature(np.asarray([record.get("internal_energy", np.nan)]), np.asarray([record.get("electron_abundance", np.nan)]))[0])
        sfr = float(record.get("sfr", np.nan))
        with np.errstate(divide="ignore", invalid="ignore"):
            log_temperature = float(np.log10(temperature))
        if np.isfinite(log_temperature) and np.isfinite(sfr):
            phase = "cold" if sfr > 0 or (sfr <= 0 and log_temperature < config.cold_log10_temperature_max) else "hot"
    # The anchor reservoir excludes other bound galaxies but retains diffuse
    # unbound gas, so the history indicator must use the same definition.
    central_cold = carrier == "gas" and host not in {"satellite", "other_halo", "unresolved"} and radial == "inner" and phase == "cold"
    valid = carrier == "gas" and host not in {"satellite", "other_halo", "unresolved"} and radial != "unresolved" and phase != "unresolved"
    if valid:
        reason = ""
    elif carrier != "gas":
        reason = carrier
    elif host in {"satellite", "other_halo", "unresolved"}:
        reason = host
    else:
        reason = "unresolved_phase"
    resolved = carrier != "unresolved" and host != "unresolved" and radial != "unresolved" and (carrier != "gas" or phase != "unresolved")
    return {"host_state": host, "radial_state": radial, "phase_state": phase, "carrier_state": carrier, "central_cold": bool(central_cold), "valid": bool(valid), "resolved": bool(resolved), "reason": reason}


def _state_sequence(tracer_id: int, snapshots: tuple[int, ...], parent_maps: dict[int, dict[int, int]], records: dict[int, dict[int, dict]], states: dict[int, dict], headers: dict[int, dict], config: Phase21Config) -> dict[int, dict]:
    """Resolve one tracer into a state dictionary at every requested snap."""

    sequence = {}
    for snap in snapshots:
        snap_map = parent_maps.get(snap, parent_maps.get(str(snap), {}))
        parent_id = snap_map.get(int(tracer_id))
        snap_records = records.get(snap, records.get(str(snap), {}))
        if parent_id is None or parent_id not in snap_records:
            sequence[snap] = {"host_state": "unresolved", "radial_state": "unresolved", "phase_state": "unresolved", "carrier_state": "unresolved", "central_cold": False, "valid": False, "resolved": False, "reason": "unresolved"}
            continue
        state = states[snap] if snap in states else states[str(snap)]
        header = headers[snap] if snap in headers else headers[str(snap)]
        sequence[snap] = classify_parent_state(snap_records[parent_id], state, header, config)
    return sequence


def _invalid(sequence: dict[int, dict], snapshots: tuple[int, ...]) -> bool:
    """Return whether any state is excluded from the valid-history mother set."""

    return any(not sequence[snap]["valid"] for snap in snapshots)


def classify_entering_tracer(tracer_id: int, sequence: dict[int, dict], config: Phase21Config) -> tuple[str, str]:
    """Classify one C94=0 -> C95=1 event according to the README."""

    history_snaps = tuple(range(config.snap_start, config.snap_event_prev + 1))
    if _invalid(sequence, history_snaps):
        return "other", ";".join(sorted({str(sequence[snap]["reason"]) for snap in history_snaps if sequence[snap]["reason"]})) or "unresolved"
    # The target cold-library membership is known from the anchor catalogue,
    # but its parent particle still has to be resolvable for an auditable
    # state vector.  A missing/invalid target record is therefore ``other``.
    target = sequence.get(config.snap_event_cur)
    if target is None or not target.get("valid", False):
        return "other", str((target or {}).get("reason", "unresolved_target"))
    if not target.get("central_cold", False):
        return "other", "target_not_central_cold"
    anchor = sequence[config.snap_event_prev]
    if anchor.get("central_cold", False):
        return "other", "source_is_central_cold"
    if anchor["carrier_state"] != "gas" or anchor["host_state"] in {"satellite", "other_halo", "unresolved"}:
        return "other", "anchor_other"
    if anchor["radial_state"] == "inner" and anchor["phase_state"] == "hot":
        radial = [sequence[snap]["radial_state"] for snap in history_snaps]
        if all(value == "inner" for value in radial):
            return "stay-in", ""
        transitions = [(index - 1, index) for index in range(1, len(radial)) if radial[index - 1] != "inner" and radial[index] == "inner"]
        outward_transitions = [index for index in range(1, len(radial)) if radial[index - 1] == "inner" and radial[index] != "inner"]
        if len(transitions) == 1 and not outward_transitions:
            entry_index = transitions[0][1]
            ordered = list(history_snaps)
            if all(sequence[snap]["radial_state"] == "inner" and sequence[snap]["phase_state"] == "hot" for snap in ordered[entry_index:]):
                return "single-in", ""
        return "recycled-in-b", ""
    if any(sequence[snap]["central_cold"] for snap in history_snaps):
        return "recycled-in-a", ""
    return "first-in", ""


def classify_exiting_tracer(tracer_id: int, sequence: dict[int, dict], config: Phase21Config) -> tuple[str, str]:
    """Classify one C94=1 -> C95=0 event according to the README."""

    history_snaps = tuple(range(config.snap_event_cur, config.snap_end + 1))
    if _invalid(sequence, history_snaps):
        return "other", ";".join(sorted({str(sequence[snap]["reason"]) for snap in history_snaps if sequence[snap]["reason"]})) or "unresolved"
    source = sequence.get(config.snap_event_prev)
    if source is None or not source.get("valid", False):
        return "other", str((source or {}).get("reason", "unresolved_source"))
    if not source.get("central_cold", False):
        return "other", "source_not_central_cold"
    if sequence[config.snap_event_cur].get("central_cold", False):
        return "other", "target_is_central_cold"
    if any(sequence[snap]["central_cold"] for snap in range(config.snap_event_cur + 1, config.snap_end + 1)):
        return "recycled-out", ""
    return "stay-out", ""


def _event_summary(classes: list[str], prefix: str, tracer_mass_msun: float, dt_gyr: float) -> dict:
    """Convert event class labels into count, mass, and rate fields."""

    counts = Counter(classes)
    total = len(classes)
    output = {f"n_total_{prefix}": int(total), f"mass_total_{prefix}_msun": float(total * tracer_mass_msun), f"rate_total_{prefix}_msun_per_yr": float(total * tracer_mass_msun / (dt_gyr * 1.0e9))}
    names = ENTER_CLASSES if prefix == "in" else EXIT_CLASSES
    for name in names:
        key = name.replace("-", "_")
        if name == "other":
            key = f"other_{prefix}"
        count = int(counts.get(name, 0))
        output[f"n_{key}"] = count
        output[f"mass_{key}_msun"] = float(count * tracer_mass_msun)
        output[f"rate_{key}_msun_per_yr"] = float(count * tracer_mass_msun / (dt_gyr * 1.0e9))
    output[f"fraction_other_{prefix}"] = (counts.get("other", 0) / total) if total else np.nan
    output[f"fraction_classified_{prefix}"] = ((total - counts.get("other", 0)) / total) if total else np.nan
    return output


def classify_halo_events(
    enter_tracer_ids: np.ndarray,
    exit_tracer_ids: np.ndarray,
    *,
    parent_maps: dict[int, dict[int, int]],
    records: dict[int, dict[int, dict]],
    states: dict[int, dict],
    headers: dict[int, dict],
    config: Phase21Config,
    dt_gyr: float,
) -> tuple[dict, list[dict]]:
    """Classify all events for one halo and assert exact closure."""

    if not np.isfinite(dt_gyr) or dt_gyr <= 0:
        raise ValueError("dt_gyr must be finite and positive")
    enters = np.unique(np.asarray(enter_tracer_ids, dtype=np.uint64))
    exits = np.unique(np.asarray(exit_tracer_ids, dtype=np.uint64))
    overlap = np.intersect1d(enters, exits)
    if len(overlap):
        raise ValueError("A tracer cannot enter and exit the cold reservoir in one window")
    enter_classes = []
    exit_classes = []
    ledger: list[dict] = []

    def ledger_row(tracer_id: int, event: str, label: str, reason: str, sequence: dict[int, dict]) -> dict:
        """Serialize the complete state history needed to audit one event."""

        anchor_snap = config.snap_event_prev if event == "in" else config.snap_event_cur
        source_snap = config.snap_event_prev
        target_snap = config.snap_event_cur
        state_sequence = {str(snap): dict(value) for snap, value in sequence.items()}
        missing_mask = {str(snap): not bool(value.get("resolved", False)) for snap, value in sequence.items()}
        return {
            "TracerID": int(tracer_id),
            "event": event,
            "anchor_snapshot": int(anchor_snap),
            "anchor_state": dict(sequence.get(anchor_snap, {})),
            "source_state": dict(sequence.get(source_snap, {})),
            "target_state": dict(sequence.get(target_snap, {})),
            "state_sequence": state_sequence,
            "missing_mask": missing_mask,
            "rate_class": label,
            "other_reason": reason,
            "weight_msun": config.tracer_mass_msun,
        }

    for tracer_id in enters:
        sequence = _state_sequence(int(tracer_id), tuple(range(config.snap_start, config.snap_event_cur + 1)), parent_maps, records, states, headers, config)
        label, reason = classify_entering_tracer(int(tracer_id), sequence, config)
        if label not in ENTER_CLASSES:
            raise AssertionError(f"Unknown entering rate class: {label}")
        enter_classes.append(label)
        ledger.append(ledger_row(int(tracer_id), "in", label, reason, sequence))
    for tracer_id in exits:
        sequence = _state_sequence(int(tracer_id), tuple(range(config.snap_event_prev, config.snap_end + 1)), parent_maps, records, states, headers, config)
        label, reason = classify_exiting_tracer(int(tracer_id), sequence, config)
        if label not in EXIT_CLASSES:
            raise AssertionError(f"Unknown exiting rate class: {label}")
        exit_classes.append(label)
        ledger.append(ledger_row(int(tracer_id), "out", label, reason, sequence))
    result = {}
    result.update(_event_summary(enter_classes, "in", config.tracer_mass_msun, dt_gyr))
    result.update(_event_summary(exit_classes, "out", config.tracer_mass_msun, dt_gyr))
    result["n_first_in_eff"] = result["n_first_in"] + result["n_single_in"]
    result["n_recycled_in_eff"] = result["n_recycled_in_a"] + result["n_recycled_in_b"]
    result["mass_first_in_eff_msun"] = result["mass_first_in_msun"] + result["mass_single_in_msun"]
    result["mass_recycled_in_eff_msun"] = result["mass_recycled_in_a_msun"] + result["mass_recycled_in_b_msun"]
    result["rate_first_in_eff_msun_per_yr"] = result["rate_first_in_msun_per_yr"] + result["rate_single_in_msun_per_yr"]
    result["rate_recycled_in_eff_msun_per_yr"] = result["rate_recycled_in_a_msun_per_yr"] + result["rate_recycled_in_b_msun_per_yr"]
    result["rate_pf_in_msun_per_yr"] = result["rate_first_in_eff_msun_per_yr"] + result["rate_stay_out_msun_per_yr"]
    result["rate_supply_msun_per_yr"] = result["rate_total_in_msun_per_yr"]
    result["rate_feedback_msun_per_yr"] = result["rate_total_out_msun_per_yr"]
    result["n_in_closure_error"] = result["n_total_in"] - (
        result["n_first_in"] + result["n_recycled_in_a"] + result["n_stay_in"]
        + result["n_single_in"] + result["n_recycled_in_b"] + result["n_other_in"]
    )
    result["n_out_closure_error"] = result["n_total_out"] - (
        result["n_stay_out"] + result["n_recycled_out"] + result["n_other_out"]
    )
    # Every tracer has the same fixed weight, so deriving the residuals from
    # the integer count residual avoids reporting floating-point round-off as
    # a scientific closure failure.
    scale = config.tracer_mass_msun / (dt_gyr * 1.0e9)
    result["mass_in_closure_error_msun"] = float(result["n_in_closure_error"] * config.tracer_mass_msun)
    result["mass_out_closure_error_msun"] = float(result["n_out_closure_error"] * config.tracer_mass_msun)
    result["rate_in_closure_error_msun_per_yr"] = float(result["n_in_closure_error"] * scale)
    result["rate_out_closure_error_msun_per_yr"] = float(result["n_out_closure_error"] * scale)
    result["classification_partition_ok"] = bool(
        result["n_in_closure_error"] == 0 and result["n_out_closure_error"] == 0
    )
    result["n_event_ledger"] = len(ledger)
    return result, ledger
