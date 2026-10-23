from pathlib import Path

import h5py
import illustris_python as il
import numpy as np

import funcs
import illustrisFuncs as iF
import real_physical_properties_all as rppa


data_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")
msun_g = 1.98847e33
kpc_cm = 3.085677581491367e21
proton_mass_g = 1.67262192369e-24
boltzmann_erg_per_K = 1.380649e-16
erg_per_keV = 1.602176634e-9
hydrogen_mass_fraction = 0.76


def calculate_bh_properties(basePath, snap=99, selected=None):
    """For each selected central subhalo, record properties of its most massive BH."""
    if selected is None:
        selected = rppa.select_objects(basePath, snap)

    num_objects = selected["start_subhalo_id"].size
    bh_particle_id = np.zeros(num_objects, dtype=np.uint64)
    bh_count = np.zeros(num_objects, dtype=np.int64)
    bh_mass_msun = np.full(num_objects, np.nan, dtype=np.float64)
    energy_rm_code = np.full(num_objects, np.nan, dtype=np.float64)
    energy_qm_code = np.full(num_objects, np.nan, dtype=np.float64)
    energy_kinetic_to_thermal = np.full(num_objects, np.nan, dtype=np.float64)
    bh_valid = np.zeros(num_objects, dtype=bool)
    energy_ratio_valid = np.zeros(num_objects, dtype=bool)

    bh_fields = ["ParticleIDs", "BH_Mass", "BH_CumEgyInjection_RM", "BH_CumEgyInjection_QM"]
    h = float(selected["hubble_param"])

    for row in np.flatnonzero(selected["analysis_valid"]):
        subhalo_id = int(selected["subhalo_id"][row])
        bh = il.snapshot.loadSubhalo(basePath, snap, subhalo_id, 5, fields=bh_fields)
        particle_ids = np.asarray(bh["ParticleIDs"], dtype=np.uint64)
        masses = np.asarray(bh["BH_Mass"], dtype=np.float64)
        rm_energy = np.asarray(bh["BH_CumEgyInjection_RM"], dtype=np.float64)
        qm_energy = np.asarray(bh["BH_CumEgyInjection_QM"], dtype=np.float64)
        bh_count[row] = masses.size

        if masses.size == 0:
            continue

        valid_mass = np.isfinite(masses) & (masses >= 0.0)
        if not np.any(valid_mass):
            continue

        valid_indices = np.flatnonzero(valid_mass)
        max_index = int(valid_indices[np.argmax(masses[valid_mass])])
        bh_particle_id[row] = particle_ids[max_index]
        bh_mass_msun[row] = masses[max_index] * 1.0e10 / h
        energy_rm_code[row] = rm_energy[max_index]
        energy_qm_code[row] = qm_energy[max_index]
        bh_valid[row] = True

        if np.isfinite(rm_energy[max_index]) and np.isfinite(qm_energy[max_index]) and qm_energy[max_index] > 0.0:
            energy_kinetic_to_thermal[row] = rm_energy[max_index] / qm_energy[max_index]
            energy_ratio_valid[row] = True

    return {
        "bh_particle_id": bh_particle_id,
        "bh_count": bh_count,
        "bh_mass_msun": bh_mass_msun,
        "bh_cum_energy_rm_code": energy_rm_code,
        "bh_cum_energy_qm_code": energy_qm_code,
        "energy_kinetic_to_thermal_ratio": energy_kinetic_to_thermal,
        "bh_valid": bh_valid,
        "energy_ratio_valid": energy_ratio_valid,
    }


def calculate_entropy(basePath, snap=99, selected=None):
    """Calculate mass-weighted entropy proxies for two non-star-forming gas samples."""
    if selected is None:
        selected = rppa.select_objects(basePath, snap)

    num_objects = selected["start_subhalo_id"].size
    entropy_hot = np.full(num_objects, np.nan, dtype=np.float64)
    entropy_non_sf_shell = np.full(num_objects, np.nan, dtype=np.float64)
    hot_selected_count = np.zeros(num_objects, dtype=np.int64)
    shell_non_sf_selected_count = np.zeros(num_objects, dtype=np.int64)
    shell_star_forming_count = np.zeros(num_objects, dtype=np.int64)
    hot_entropy_cell_count = np.zeros(num_objects, dtype=np.int64)
    shell_entropy_cell_count = np.zeros(num_objects, dtype=np.int64)
    hot_entropy_weight_mass_msun = np.zeros(num_objects, dtype=np.float64)
    shell_entropy_weight_mass_msun = np.zeros(num_objects, dtype=np.float64)
    entropy_valid_hot = np.zeros(num_objects, dtype=bool)
    entropy_valid_non_sf_shell = np.zeros(num_objects, dtype=bool)

    header = il.groupcat.loadHeader(basePath, snap)
    h = float(header["HubbleParam"])
    a = float(header["Time"])
    box_size = float(header["BoxSize"])
    gas_fields = ["Coordinates", "Masses", "StarFormationRate", "InternalEnergy", "ElectronAbundance", "Density"]

    for row in np.flatnonzero(selected["analysis_valid"]):
        fof_id = int(selected["fof_id"][row])
        gas = il.snapshot.loadHalo(basePath, snap, fof_id, "gas", fields=gas_fields)
        coordinates = np.asarray(gas["Coordinates"], dtype=np.float64)
        if coordinates.shape[0] == 0:
            continue

        distance_raw = funcs.dist_vector_nb(selected["group_pos"][row], coordinates, box_size)
        radius_ratio = distance_raw / selected["r200c_raw"][row]
        sfr = np.asarray(gas["StarFormationRate"], dtype=np.float64)
        non_star_forming = np.isfinite(sfr) & (sfr == 0.0)
        shell = np.isfinite(radius_ratio) & (radius_ratio >= 0.1) & (radius_ratio <= 1.0)
        inside_r200 = np.isfinite(radius_ratio) & (radius_ratio <= 1.0)
        shell_star_forming_count[row] = np.count_nonzero(shell & np.isfinite(sfr) & (sfr > 0.0))
        shell_non_sf_selected_count[row] = np.count_nonzero(shell & non_star_forming)

        internal_energy = np.asarray(gas["InternalEnergy"], dtype=np.float64)
        electron_abundance = np.asarray(gas["ElectronAbundance"], dtype=np.float64)
        temperature = np.full(sfr.shape, np.nan, dtype=np.float64)
        temperature[non_star_forming] = iF.utherm_to_temp(internal_energy[non_star_forming], electron_abundance[non_star_forming])

        hot = non_star_forming & inside_r200 & np.isfinite(temperature) & (temperature > 10.0 ** 4.5)
        non_sf_shell = non_star_forming & shell
        hot_selected_count[row] = np.count_nonzero(hot)

        density_code = np.asarray(gas["Density"], dtype=np.float64)
        density_g_cm3 = density_code * 1.0e10 * h ** 2 / a ** 3 * msun_g / kpc_cm ** 3
        n_h_cm3 = hydrogen_mass_fraction * density_g_cm3 / proton_mass_g
        n_e_cm3 = electron_abundance * n_h_cm3
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            entropy_keV_cm2 = boltzmann_erg_per_K * temperature / np.power(n_e_cm3, 2.0 / 3.0) / erg_per_keV

        mass_msun = np.asarray(gas["Masses"], dtype=np.float64) * 1.0e10 / h
        valid_entropy = np.isfinite(entropy_keV_cm2) & (entropy_keV_cm2 > 0.0) & np.isfinite(mass_msun) & (mass_msun > 0.0) & np.isfinite(n_e_cm3) & (n_e_cm3 > 0.0)

        hot_valid = hot & valid_entropy
        if np.any(hot_valid):
            hot_entropy_cell_count[row] = np.count_nonzero(hot_valid)
            hot_entropy_weight_mass_msun[row] = np.sum(mass_msun[hot_valid])
            entropy_hot[row] = np.average(entropy_keV_cm2[hot_valid], weights=mass_msun[hot_valid])
            entropy_valid_hot[row] = True

        shell_valid = non_sf_shell & valid_entropy
        if np.any(shell_valid):
            shell_entropy_cell_count[row] = np.count_nonzero(shell_valid)
            shell_entropy_weight_mass_msun[row] = np.sum(mass_msun[shell_valid])
            entropy_non_sf_shell[row] = np.average(entropy_keV_cm2[shell_valid], weights=mass_msun[shell_valid])
            entropy_valid_non_sf_shell[row] = True

    return {
        "entropy_hot_mass_weighted_keV_cm2": entropy_hot,
        "entropy_non_sf_0p1_to_1_r200c_mass_weighted_keV_cm2": entropy_non_sf_shell,
        "hot_gas_cell_count": hot_selected_count,
        "non_sf_shell_cell_count": shell_non_sf_selected_count,
        "star_forming_shell_cell_count_excluded": shell_star_forming_count,
        "hot_entropy_valid_cell_count": hot_entropy_cell_count,
        "non_sf_shell_entropy_valid_cell_count": shell_entropy_cell_count,
        "hot_entropy_weight_mass_msun": hot_entropy_weight_mass_msun,
        "non_sf_shell_entropy_weight_mass_msun": shell_entropy_weight_mass_msun,
        "entropy_valid_hot": entropy_valid_hot,
        "entropy_valid_non_sf_shell": entropy_valid_non_sf_shell,
    }


def calculate_agn_properties(basePath, snap=99, output_file=None):
    """Calculate BH and gas entropy properties for all selected trees at one snapshot."""
    selected = rppa.select_objects(basePath, snap)
    bh_properties = calculate_bh_properties(basePath, snap, selected)
    entropy_properties = calculate_entropy(basePath, snap, selected)

    if output_file is None:
        output_file = data_dir / f"AGN_properties_{snap}.hdf5"
    else:
        output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_file, "w") as f:
        f.attrs["snap"] = int(snap)
        f.attrs["scale_factor"] = float(selected["scale_factor"])
        f.attrs["hubble_param"] = float(selected["hubble_param"])
        f.attrs["entropy_definition"] = "K = k_B*T / n_e^(2/3); mass-weighted cell mean; keV cm^2"
        f.attrs["hot_gas_selection"] = "SFR == 0, T > 10^4.5 K, r <= R200c"
        f.attrs["shell_gas_selection"] = "SFR == 0, 0.1 <= r/R200c <= 1"

        f.create_dataset("tree_row", data=selected["tree_row"])
        f.create_dataset("start_subhalo_id", data=selected["start_subhalo_id"])
        f.create_dataset("subhalo_id", data=selected["subhalo_id"])
        f.create_dataset("fof_id", data=selected["fof_id"])
        f.create_dataset("analysis_valid", data=selected["analysis_valid"].astype(np.uint8))
        f.create_dataset("m200c_msun", data=selected["m200c_msun"])
        f.create_dataset("r200c_physical_kpc", data=selected["r200c_raw"] * selected["scale_factor"] / selected["hubble_param"])

        bh_group = f.create_group("black_hole")
        for name, values in bh_properties.items():
            bh_group.create_dataset(name, data=values, compression="gzip", compression_opts=4)

        entropy_group = f.create_group("entropy")
        for name, values in entropy_properties.items():
            entropy_group.create_dataset(name, data=values, compression="gzip", compression_opts=4)

    print(f"Saved AGN and entropy properties to {output_file}", flush=True)
    return output_file, selected, bh_properties, entropy_properties
