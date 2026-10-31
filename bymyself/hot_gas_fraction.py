import argparse
from pathlib import Path
import h5py
import illustris_python as il
import numpy as np
import funcs
import illustrisFuncs as iF
import real_physical_properties_all as rppa


data_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")
hot_temperature_min_K = 10.0 ** 4.5
gas_fields = ("Coordinates", "Masses", "StarFormationRate", "InternalEnergy", "ElectronAbundance")


def compute_hot_gas_properties(gas, halo_position, r200c_raw, box_size, h):

    num_cells = int(gas.get("count", np.asarray(gas.get("Masses", [])).size))

    if num_cells == 0:
        position = np.empty((0, 3), dtype=np.float64)
        mass = sfr = internal_energy = electron_abundance = np.empty(0, dtype=np.float64)
    else:
        position = np.asarray(gas["Coordinates"], dtype=np.float64)
        mass = np.asarray(gas["Masses"], dtype=np.float64) * 1.0e10 / h
        sfr = np.asarray(gas["StarFormationRate"], dtype=np.float64)
        internal_energy = np.asarray(gas["InternalEnergy"], dtype=np.float64)
        electron_abundance = np.asarray(gas["ElectronAbundance"], dtype=np.float64)

    position_valid = np.all(np.isfinite(position), axis=1)
    distance = np.full(num_cells, np.nan, dtype=np.float64)
    if np.any(position_valid):
        distance[position_valid] = funcs.dist_vector_nb(halo_position, position[position_valid], box_size)
    inside = position_valid & np.isfinite(distance) & (distance <= r200c_raw)
    mass_valid = np.isfinite(mass) & (mass >= 0.0)
    non_star_forming = np.isfinite(sfr) & (sfr == 0.0)
    thermal_input_valid = (non_star_forming & np.isfinite(internal_energy) & (internal_energy >= 0.0)
        & np.isfinite(electron_abundance) & (electron_abundance >= 0.0))
    temperature = np.full(num_cells, np.nan, dtype=np.float64)
    if np.any(thermal_input_valid):
        temperature[thermal_input_valid] = iF.utherm_to_temp(internal_energy[thermal_input_valid], electron_abundance[thermal_input_valid])
    temperature_valid = np.isfinite(temperature) & (temperature > 0.0)
    state_known = (np.isfinite(sfr) & (sfr > 0.0)) | (non_star_forming & temperature_valid)
    hot = inside & mass_valid & non_star_forming & temperature_valid & (temperature > hot_temperature_min_K)

    invalid_position_count = int(np.count_nonzero(~position_valid))
    invalid_mass_count = int(np.count_nonzero(inside & ~mass_valid))
    unclassified_count = int(np.count_nonzero(inside & mass_valid & (mass > 0.0) & ~state_known))
    gas_mass_valid = invalid_position_count == 0 and invalid_mass_count == 0
    hot_mass_valid = gas_mass_valid and unclassified_count == 0
    return {
        "m_hot_msun": float(np.sum(mass[hot])) if hot_mass_valid else np.nan,
        "m_gas_msun": float(np.sum(mass[inside & mass_valid])) if gas_mass_valid else np.nan,
        "hot_mass_valid": hot_mass_valid,
        "gas_mass_valid": gas_mass_valid,
        "gas_cell_count": int(np.count_nonzero(inside)),
        "hot_cell_count": int(np.count_nonzero(hot)),
        "unclassified_gas_cell_count": unclassified_count,
        "invalid_mass_cell_count": invalid_mass_count,
        "invalid_position_cell_count": invalid_position_count,
    }


def hot_gas_fraction(basePath, snap):

    snap = int(snap)
    header = il.groupcat.loadHeader(basePath, snap)
    h = float(header["HubbleParam"])
    scale_factor = float(header["Time"])
    box_size = float(header["BoxSize"])
    omega_b, omega_m = float(header["OmegaBaryon"]), float(header["Omega0"])
    fb = omega_b / omega_m

    selected = rppa.select_objects(basePath, snap)
    num_trees = selected["start_subhalo_id"].size
    analysis_valid = np.asarray(selected["analysis_valid"], dtype=bool)
    results = {
        "tree_row": selected["tree_row"],
        "start_subhalo_id": selected["start_subhalo_id"],
        "subhalo_id": selected["subhalo_id"],
        "fof_id": selected["fof_id"],
        "analysis_valid": analysis_valid,
        "m200c_msun": selected["m200c_msun"],
        "r200c_physical_kpc": selected["r200c_raw"] * scale_factor / h,
    }
    mass_names = ("m_hot_msun", "m_gas_msun")
    validity_names = ("hot_mass_valid", "gas_mass_valid")
    count_names = ("gas_cell_count", "hot_cell_count", "unclassified_gas_cell_count", "invalid_mass_cell_count", "invalid_position_cell_count")
    for name in mass_names:
        results[name] = np.full(num_trees, np.nan, dtype=np.float64)
    for name in validity_names:
        results[name] = np.zeros(num_trees, dtype=bool)
    for name in count_names:
        results[name] = np.full(num_trees, -1, dtype=np.int64)

    valid_rows = np.flatnonzero(analysis_valid)
    unique_fof_ids, representative_indices, inverse = np.unique(selected["fof_id"][valid_rows], return_index=True, return_inverse=True)
    unique_results = {name: np.empty(unique_fof_ids.size, dtype=results[name].dtype) for name in mass_names + validity_names + count_names}
    for halo_index, fof_id in enumerate(unique_fof_ids):
        row = valid_rows[representative_indices[halo_index]]
        gas = il.snapshot.loadHalo(basePath, snap, int(fof_id), "gas", fields=list(gas_fields))
        measurement = compute_hot_gas_properties(gas, selected["group_pos"][row], selected["r200c_raw"][row], box_size, h)
        for name, value in measurement.items():
            unique_results[name][halo_index] = value
        del gas
        if (halo_index + 1) % 100 == 0:
            print(f"Snap {snap}: measured {halo_index + 1}/{unique_fof_ids.size} unique halos.", flush=True)
    for name, values in unique_results.items():
        results[name][valid_rows] = values[inverse]

    baryon_budget = fb * results["m200c_msun"]
    baryon_fraction_valid = analysis_valid & results["hot_mass_valid"] & np.isfinite(baryon_budget) & (baryon_budget > 0.0)
    gas_fraction_valid = analysis_valid & results["hot_mass_valid"] & results["gas_mass_valid"] & (results["m_gas_msun"] > 0.0)
    results["hot_mass_over_fb_m200c"] = np.full(num_trees, np.nan, dtype=np.float64)
    results["hot_mass_over_gas"] = np.full(num_trees, np.nan, dtype=np.float64)
    np.divide(results["m_hot_msun"], baryon_budget, out=results["hot_mass_over_fb_m200c"], where=baryon_fraction_valid)
    np.divide(results["m_hot_msun"], results["m_gas_msun"], out=results["hot_mass_over_gas"], where=gas_fraction_valid)
    results["hot_mass_over_fb_m200c_valid"] = baryon_fraction_valid
    results["hot_mass_over_gas_valid"] = gas_fraction_valid

    output_file = data_dir / f"hot_gas_fraction_{snap}.hdf5" if output_file is None else Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_name(f".{output_file.name}.tmp")
    try:
        with h5py.File(temporary_file, "w") as f:
            for name, values in results.items():
                f.create_dataset(name, data=values, compression="gzip", compression_opts=4)

        temporary_file.replace(output_file)
    finally:
        if temporary_file.exists():
            temporary_file.unlink()
    return