from pathlib import Path
import h5py
import illustris_python as il
import numpy as np
import funcs
import illustrisFuncs as iF

data_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")
num_radial_bins = 25
radial_min = 0.01
radial_max = 1.0
msun_g = 1.98847e33
kpc_cm = 3.085677581491367e21

def select_objects(basePath, snap):

    tree_file = data_dir / "all_selected_trees.hdf5"

    with h5py.File(tree_file, "r") as f:
        start_ids = np.asarray(f["start_subhalo_id"][:], dtype=np.int64)
        snap_numbers = np.asarray(f["snap_numbers"][:], dtype=np.int64)
        matches = np.flatnonzero(snap_numbers == snap)
        if matches.size != 1:
            raise ValueError(f"Expected one tree column for snap {snap}, found {matches.size}.")
        snap_column = int(matches[0])
        subhalo_ids = np.asarray(f["mpb_subhalo_ids"][:, snap_column], dtype=np.int64)
        fof_ids = np.asarray(f["mpb_fof_ids"][:, snap_column], dtype=np.int64)

    header = il.groupcat.loadHeader(basePath, snap)
    h = header["HubbleParam"]
    group_data = il.groupcat.loadHalos(basePath, snap, fields=["Group_M_Crit200", "Group_R_Crit200", "GroupPos"])
    group_m200_msun = np.asarray(group_data["Group_M_Crit200"], dtype=np.float64) * 1.0e10 / h
    group_r200_raw = np.asarray(group_data["Group_R_Crit200"], dtype=np.float64)
    group_pos = np.asarray(group_data["GroupPos"], dtype=np.float64)

    valid_fof = (fof_ids >= 0) & (fof_ids < group_m200_msun.size)
    valid_node = (subhalo_ids >= 0) & (subhalo_ids < int(header["Nsubgroups_Total"]))
    analysis_valid = valid_fof & valid_node
    candidate_rows = np.flatnonzero(analysis_valid)
    candidate_fof_ids = fof_ids[candidate_rows]
    analysis_valid[candidate_rows] &= np.isfinite(group_m200_msun[candidate_fof_ids]) & (group_m200_msun[candidate_fof_ids] > 0.0) & np.isfinite(group_r200_raw[candidate_fof_ids]) & (group_r200_raw[candidate_fof_ids] > 0.0) & np.all(np.isfinite(group_pos[candidate_fof_ids]), axis=1)

    selected_rows = np.arange(start_ids.size, dtype=np.int64)
    selected_fof_ids = fof_ids.copy()
    selected_m200 = np.full(start_ids.size, np.nan, dtype=np.float64)
    selected_r200 = np.full(start_ids.size, np.nan, dtype=np.float64)
    selected_group_pos = np.full((start_ids.size, 3), np.nan, dtype=np.float64)
    selected_m200[analysis_valid] = group_m200_msun[selected_fof_ids[analysis_valid]]
    selected_r200[analysis_valid] = group_r200_raw[selected_fof_ids[analysis_valid]]
    selected_group_pos[analysis_valid] = group_pos[selected_fof_ids[analysis_valid]]
    return {
        "tree_row": selected_rows,
        "start_subhalo_id": start_ids,
        "subhalo_id": subhalo_ids,
        "fof_id": selected_fof_ids,
        "analysis_valid": analysis_valid,
        "m200c_msun": selected_m200,
        "r200c_raw": selected_r200,
        "group_pos": selected_group_pos,
        "hubble_param": h,
        "scale_factor": header["Time"],
    }


def calculate_real_rou(mass_g, cell_volume_cm3, radial_bin, num_bins=num_radial_bins):

    valid_mass = np.isfinite(mass_g) & (mass_g > 0.0)
    valid_volume = valid_mass & np.isfinite(cell_volume_cm3) & (cell_volume_cm3 > 0.0)
    shell_mass_g = np.bincount(radial_bin[valid_mass], weights=mass_g[valid_mass], minlength=num_bins)
    cell_density_mass_g = np.bincount(radial_bin[valid_volume], weights=mass_g[valid_volume], minlength=num_bins)
    hot_cell_volume_cm3 = np.bincount(radial_bin[valid_volume], weights=cell_volume_cm3[valid_volume], minlength=num_bins)
    rho_hot_cell_mean = np.full(num_bins, np.nan, dtype=np.float64)
    np.divide(cell_density_mass_g, hot_cell_volume_cm3, out=rho_hot_cell_mean, where=hot_cell_volume_cm3 > 0.0)
    
    return rho_hot_cell_mean, shell_mass_g, cell_density_mass_g, hot_cell_volume_cm3


def calculate_real_T(mass_g, temperature, radial_bin, num_bins=num_radial_bins):

    valid = np.isfinite(mass_g) & (mass_g > 0.0) & np.isfinite(temperature) & (temperature > 0.0)
    mass_sum_g = np.bincount(radial_bin[valid], weights=mass_g[valid], minlength=num_bins)
    mass_temperature_sum = np.bincount(radial_bin[valid], weights=mass_g[valid] * temperature[valid], minlength=num_bins)
    mean_temperature = np.full(num_bins, np.nan, dtype=np.float64)
    np.divide(mass_temperature_sum, mass_sum_g, out=mean_temperature, where=mass_sum_g > 0.0)
    
    return mean_temperature, mass_sum_g


def calculate_real_Z(mass_g, metallicity, radial_bin, num_bins=num_radial_bins):

    valid = np.isfinite(mass_g) & (mass_g > 0.0) & np.isfinite(metallicity) & (metallicity >= 0.0)
    mass_sum_g = np.bincount(radial_bin[valid], weights=mass_g[valid], minlength=num_bins)
    mass_metallicity_sum = np.bincount(radial_bin[valid], weights=mass_g[valid] * metallicity[valid], minlength=num_bins)
    mean_metallicity = np.full(num_bins, np.nan, dtype=np.float64)
    np.divide(mass_metallicity_sum, mass_sum_g, out=mean_metallicity, where=mass_sum_g > 0.0)
    return mean_metallicity, mass_sum_g


def interpolate_profile(radial_centers, profile, interpolation_radii, log_y=True):

    values = np.asarray(profile, dtype=np.float64)
    valid = np.isfinite(values)
    if log_y:
        valid &= values > 0.0

    interpolated = np.full(interpolation_radii.shape, np.nan, dtype=np.float64)
    for i in range(radial_centers.size - 1):
        if not (valid[i] and valid[i + 1]):
            continue
        in_segment = (interpolation_radii >= radial_centers[i]) & (interpolation_radii <= radial_centers[i + 1])
        if not np.any(in_segment):
            continue
        x_target = np.log10(interpolation_radii[in_segment])
        x_pair = np.log10(radial_centers[i:i + 2])
        y_pair = np.log10(values[i:i + 2]) if log_y else values[i:i + 2]
        y_target = np.interp(x_target, x_pair, y_pair)
        interpolated[in_segment] = np.power(10.0, y_target) if log_y else y_target
    return interpolated


def real_physical_properties(basePath, snap):

    selected = select_objects(basePath, snap=snap)
    header = il.groupcat.loadHeader(basePath, snap)
    box_size = header["BoxSize"]
    h = selected["hubble_param"]
    a = selected["scale_factor"]
    radial_edges = np.geomspace(radial_min, radial_max, num_radial_bins + 1)
    radial_centers = np.sqrt(radial_edges[:-1] * radial_edges[1:])
    interpolation_radii = np.geomspace(radial_min, radial_max, 200)
    profile_names = ("rho_hot_cell_mean_g_cm3", "temperature_mass_weighted_K", "metallicity_mass_weighted")
    profiles = {name: np.full((selected["fof_id"].size, num_radial_bins), np.nan, dtype=np.float64) for name in profile_names}
    hot_cell_count = np.zeros((selected["fof_id"].size, num_radial_bins), dtype=np.int64)
    density_cell_count = np.zeros_like(hot_cell_count)
    temperature_cell_count = np.zeros_like(hot_cell_count)
    metallicity_cell_count = np.zeros_like(hot_cell_count)
    shell_hot_gas_mass_g = np.zeros((selected["fof_id"].size, num_radial_bins), dtype=np.float64)
    shell_density_valid_mass_g = np.zeros_like(shell_hot_gas_mass_g)
    shell_hot_cell_volume_cm3 = np.zeros_like(shell_hot_gas_mass_g)
    hot_gas_mass_msun = np.zeros(selected["fof_id"].size, dtype=np.float64)
    hot_gas_count_r200 = np.zeros(selected["fof_id"].size, dtype=np.int64)
    global_temperature_K = np.full(selected["fof_id"].size, np.nan, dtype=np.float64)
    global_metallicity = np.full(selected["fof_id"].size, np.nan, dtype=np.float64)
    interpolation_profiles = {name + "_interpolated": np.full((selected["fof_id"].size, interpolation_radii.size), np.nan, dtype=np.float64) for name in profile_names}

    gas_fields = ["Coordinates", "Masses", "StarFormationRate", "InternalEnergy", "ElectronAbundance", "GFM_Metallicity", "Density"]
    for object_index in np.flatnonzero(selected["analysis_valid"]):
        fof_id = selected["fof_id"][object_index]
        gas = il.snapshot.loadHalo(basePath, snap, int(fof_id), "gas", fields=gas_fields)
        if gas.get("count", 0) == 0:
            continue

        distance_raw = funcs.dist_vector_nb(selected["group_pos"][object_index], np.asarray(gas["Coordinates"], dtype=np.float64), box_size)
        radius_ratio = distance_raw / selected["r200c_raw"][object_index]
        sfr = np.asarray(gas["StarFormationRate"], dtype=np.float64)
        temperature = np.full(sfr.shape, np.nan, dtype=np.float64)
        non_star_forming = (sfr == 0.0) & np.isfinite(sfr)
        temperature[non_star_forming] = iF.utherm_to_temp(np.asarray(gas["InternalEnergy"])[non_star_forming], np.asarray(gas["ElectronAbundance"])[non_star_forming])
        hot = non_star_forming & np.isfinite(temperature) & (temperature > 10.0 ** 4.5) & np.isfinite(radius_ratio) & (radius_ratio <= radial_max)

        mass_g_all = np.asarray(gas["Masses"], dtype=np.float64) * 1.0e10 / h * msun_g
        hot_gas_count_r200[object_index] = np.count_nonzero(hot)
        valid_hot_mass = hot & np.isfinite(mass_g_all) & (mass_g_all > 0.0)
        hot_gas_mass_msun[object_index] = np.sum(mass_g_all[valid_hot_mass]) / msun_g
        if np.any(hot):
            valid_temperature = hot & np.isfinite(mass_g_all) & (mass_g_all > 0.0) & (temperature > 0.0)
            if np.any(valid_temperature):
                global_temperature_K[object_index] = np.average(temperature[valid_temperature], weights=mass_g_all[valid_temperature])
            metallicity_all = np.asarray(gas["GFM_Metallicity"], dtype=np.float64)
            valid_metallicity = hot & np.isfinite(mass_g_all) & (mass_g_all > 0.0) & np.isfinite(metallicity_all) & (metallicity_all >= 0.0)
            if np.any(valid_metallicity):
                global_metallicity[object_index] = np.average(metallicity_all[valid_metallicity], weights=mass_g_all[valid_metallicity])

        in_profile = hot & (radius_ratio >= radial_min)
        if not np.any(in_profile):
            continue

        radial_bin = np.searchsorted(radial_edges, radius_ratio[in_profile], side="right") - 1
        radial_bin = np.minimum(radial_bin, num_radial_bins - 1).astype(np.int64)
        mass_g = mass_g_all[in_profile]
        temperature_profile_cells = temperature[in_profile]
        metallicity_profile_cells = metallicity_all[in_profile]

        density_raw = np.asarray(gas["Density"], dtype=np.float64)[in_profile]
        density_msun_kpc3 = density_raw * 1.0e10 * h ** 2 / a ** 3
        mass_msun = mass_g / msun_g
        cell_volume_kpc3 = np.full(mass_msun.shape, np.nan, dtype=np.float64)
        valid_density = np.isfinite(density_msun_kpc3) & (density_msun_kpc3 > 0.0)
        np.divide(mass_msun, density_msun_kpc3, out=cell_volume_kpc3, where=valid_density)
        cell_volume_cm3 = cell_volume_kpc3 * kpc_cm ** 3

        rho_cell, shell_mass_g, density_mass_g, cell_volume_sum_cm3 = calculate_real_rou(mass_g, cell_volume_cm3, radial_bin)
        mean_temperature, _ = calculate_real_T(mass_g, temperature_profile_cells, radial_bin)
        mean_metallicity, _ = calculate_real_Z(mass_g, metallicity_profile_cells, radial_bin)
        shell_hot_gas_mass_g[object_index] = shell_mass_g
        shell_density_valid_mass_g[object_index] = density_mass_g
        shell_hot_cell_volume_cm3[object_index] = cell_volume_sum_cm3
        profiles["rho_hot_cell_mean_g_cm3"][object_index] = rho_cell
        profiles["temperature_mass_weighted_K"][object_index] = mean_temperature
        profiles["metallicity_mass_weighted"][object_index] = mean_metallicity

        hot_cell_count[object_index] = np.bincount(radial_bin, minlength=num_radial_bins)
        density_cell_count[object_index] = np.bincount(radial_bin[np.isfinite(cell_volume_cm3) & (cell_volume_cm3 > 0.0)], minlength=num_radial_bins)
        temperature_cell_count[object_index] = np.bincount(radial_bin[np.isfinite(mass_g) & (mass_g > 0.0) & np.isfinite(temperature_profile_cells) & (temperature_profile_cells > 0.0)], minlength=num_radial_bins)
        metallicity_cell_count[object_index] = np.bincount(radial_bin[np.isfinite(mass_g) & (mass_g > 0.0) & np.isfinite(metallicity_profile_cells) & (metallicity_profile_cells >= 0.0)], minlength=num_radial_bins)

        for name in profile_names:
            interpolation_profiles[name + "_interpolated"][object_index] = interpolate_profile(radial_centers, profiles[name][object_index], interpolation_radii, log_y=(name != "metallicity_mass_weighted"))

    output_file = data_dir / f"real_physical_properties_all_{snap}.hdf5"
    with h5py.File(output_file, "w") as f:
        f.create_dataset("tree_row", data=selected["tree_row"])
        f.create_dataset("start_subhalo_id", data=selected["start_subhalo_id"])
        f.create_dataset("subhalo_id", data=selected["subhalo_id"])
        f.create_dataset("fof_id", data=selected["fof_id"])
        f.create_dataset("analysis_valid", data=selected["analysis_valid"].astype(np.uint8))
        f.create_dataset("m200c_msun", data=selected["m200c_msun"])
        f.create_dataset("r200c_physical_kpc", data=selected["r200c_raw"] * a / h)
        f.create_dataset("radial_bin_edges_r200c", data=radial_edges)
        f.create_dataset("radial_bin_centers_r200c", data=radial_centers)
        f.create_dataset("interpolation_radii_r200c", data=interpolation_radii)
        f.create_dataset("hot_gas_mass_g_in_shell", data=shell_hot_gas_mass_g)
        f.create_dataset("density_valid_hot_gas_mass_g_in_shell", data=shell_density_valid_mass_g)
        f.create_dataset("hot_gas_cell_volume_cm3", data=shell_hot_cell_volume_cm3)
        f.create_dataset("hot_cell_count", data=hot_cell_count)
        f.create_dataset("hot_gas_count_within_r200c", data=hot_gas_count_r200)
        f.create_dataset("density_cell_count", data=density_cell_count)
        f.create_dataset("temperature_cell_count", data=temperature_cell_count)
        f.create_dataset("metallicity_cell_count", data=metallicity_cell_count)
        f.create_dataset("hot_gas_mass_msun_within_r200c", data=hot_gas_mass_msun)
        f.create_dataset("global_temperature_mass_weighted_K", data=global_temperature_K)
        f.create_dataset("global_metallicity_mass_weighted", data=global_metallicity)
        for name in profile_names:
            f.create_dataset(name, data=profiles[name], compression="gzip", compression_opts=4)
        for name, values in interpolation_profiles.items():
            f.create_dataset(name, data=values, compression="gzip", compression_opts=4)

    print(f"Saved hot-gas radial profiles to {output_file}", flush=True)
    return output_file, selected, profiles, interpolation_profiles