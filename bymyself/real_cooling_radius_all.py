from pathlib import Path
import sys

import h5py
import numpy as np

import cooling_radius as cr


data_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")
msun_g = 1.98847e33
kpc_cm = 3.085677581491367e21
G_kpc_kms2_msun = 4.30091e-6
mu = 0.59
m_H_g = 1.67262192369e-24
k_B_erg_K = 1.380649e-16

status_invalid_input = 0
status_root_found = 1
status_below_profile_range = 2
status_above_profile_range = 3
status_no_outward_crossing = 4

def extend_hot_gas_profiles(radius_r200c, temperature, density, metallicity, outer_max_r200c=5.0, num_outer_points=128):

    inner_radii = radius_r200c if radius_r200c[-1] == 1.0 else np.append(radius_r200c, 1.0)
    extended_radii = np.concatenate((inner_radii, np.geomspace(1.0, outer_max_r200c, num_outer_points)[1:]))
    extended_profiles = [np.full((temperature.shape[0], extended_radii.size), np.nan, dtype=np.float64) for _ in range(3)]
    anchor_radii = np.full(temperature.shape[0], np.nan, dtype=np.float64)

    for row in range(temperature.shape[0]):
        valid = np.isfinite(temperature[row]) & (temperature[row] > 0.0) & np.isfinite(density[row]) & (density[row] > 0.0) & np.isfinite(metallicity[row]) & (metallicity[row] >= 0.0)
        indices = np.flatnonzero(valid)
        for source, extended in zip((temperature, density, metallicity), extended_profiles):
            extended[row, :radius_r200c.size] = source[row]
        if indices.size < 2:
            continue
        anchor = int(indices[-1])
        anchor_radii[row] = radius_r200c[anchor]
        tail = extended_radii > radius_r200c[anchor]
        extended_profiles[0][row, tail] = temperature[row, anchor]
        extended_profiles[1][row, tail] = density[row, anchor] * (extended_radii[tail] / radius_r200c[anchor]) ** -2.0
        extended_profiles[2][row, tail] = metallicity[row, anchor]

    return extended_radii, *extended_profiles, anchor_radii


def real_cooling_radius(snap, properties_file=None, output_file=None, table_dir=None, outer_max_r200c=5.0):

    snap = int(snap)
    properties_file = data_dir / f"real_physical_properties_all_{snap}.hdf5" if properties_file is None else Path(properties_file)
    output_file = data_dir / f"real_cooling_radius_{snap}.hdf5" if output_file is None else Path(output_file)

    with h5py.File(properties_file, "r") as f:
        tree_row = np.asarray(f["tree_row"][:], dtype=np.int64)
        start_ids = np.asarray(f["start_subhalo_id"][:], dtype=np.int64)
        node_ids = np.asarray(f["subhalo_id"][:], dtype=np.int64)
        fof_ids = np.asarray(f["fof_id"][:], dtype=np.int64)
        analysis_valid = np.asarray(f["analysis_valid"][:], dtype=bool)
        halo_mass_msun = np.asarray(f["m200c_msun"][:], dtype=np.float64)
        r200_kpc = np.asarray(f["r200c_physical_kpc"][:], dtype=np.float64)
        radius_r200c = np.asarray(f["interpolation_radii_r200c"][:], dtype=np.float64)
        temperature = np.asarray(f["temperature_mass_weighted_K_interpolated"][:], dtype=np.float64)
        density = np.asarray(f["rho_hot_cell_mean_g_cm3_interpolated"][:], dtype=np.float64)
        metallicity = np.asarray(f["metallicity_mass_weighted_interpolated"][:], dtype=np.float64)

    n_rows = start_ids.size
    row_arrays = (tree_row, node_ids, fof_ids, analysis_valid, halo_mass_msun, r200_kpc)

    source_radius_r200c = radius_r200c.copy()
    radius_r200c, temperature, density, metallicity, anchor_radii = extend_hot_gas_profiles(radius_r200c, temperature, density, metallicity, outer_max_r200c=outer_max_r200c)
    boundary_index = int(np.flatnonzero(radius_r200c == 1.0)[0])
    inside_r200c = radius_r200c <= 1.0
    expected_profile_shape = (n_rows, radius_r200c.size)

    cooling_tables = cr.CoolingTableLoader(table_dir=table_dir)
    cooling_function = np.full(expected_profile_shape, np.nan, dtype=np.float64)
    cooling_time_s = np.full(expected_profile_shape, np.nan, dtype=np.float64)
    cooling_time_over_tdyn = np.full(expected_profile_shape, np.nan, dtype=np.float64)
    cooling_radius_r200c = np.full(n_rows, np.nan, dtype=np.float64)
    cooling_radius_kpc = np.full(n_rows, np.nan, dtype=np.float64)
    outer_cooling_radius_r200c = np.full(n_rows, np.nan, dtype=np.float64)
    v200_kms = np.full(n_rows, np.nan, dtype=np.float64)
    t_dyn_s = np.full(n_rows, np.nan, dtype=np.float64)
    root_lower_r200c = np.full(n_rows, np.nan, dtype=np.float64)
    root_upper_r200c = np.full(n_rows, np.nan, dtype=np.float64)
    root_count = np.zeros(n_rows, dtype=np.int16)
    status = np.full(n_rows, status_invalid_input, dtype=np.uint8)

    valid_halo = analysis_valid & np.isfinite(halo_mass_msun) & (halo_mass_msun > 0.0) & np.isfinite(r200_kpc) & (r200_kpc > 0.0) & np.isfinite(anchor_radii)
    v200_kms[valid_halo] = np.sqrt(G_kpc_kms2_msun * halo_mass_msun[valid_halo] / r200_kpc[valid_halo])
    t_dyn_s[valid_halo] = r200_kpc[valid_halo] * kpc_cm / (v200_kms[valid_halo] * 1.0e5)

    for row in np.flatnonzero(valid_halo):
        valid_profile = np.isfinite(temperature[row]) & (temperature[row] > 0.0) & np.isfinite(density[row]) & (density[row] > 0.0) & np.isfinite(metallicity[row]) & (metallicity[row] >= 0.0)
        if np.count_nonzero(valid_profile) < 2 or not np.isfinite(t_dyn_s[row]) or t_dyn_s[row] <= 0.0:
            continue

        lambda_row = np.asarray(cr.LgalCoolingFunction(temperature[row], metallicity[row], cooling_tables), dtype=np.float64)
        valid_cooling = valid_profile & np.isfinite(lambda_row) & (lambda_row > 0.0)
        cooling_function[row, valid_cooling] = lambda_row[valid_cooling]
        cooling_time_s[row, valid_cooling] = (1.5 * mu * m_H_g * k_B_erg_K * temperature[row, valid_cooling]) / (density[row, valid_cooling] * lambda_row[valid_cooling])
        cooling_time_over_tdyn[row, valid_cooling] = cooling_time_s[row, valid_cooling] / t_dyn_s[row]

        log_ratio = np.full(radius_r200c.size, np.nan, dtype=np.float64)
        finite_ratio = np.isfinite(cooling_time_over_tdyn[row]) & (cooling_time_over_tdyn[row] > 0.0)
        log_ratio[finite_ratio] = np.log10(cooling_time_over_tdyn[row, finite_ratio])
        finite_indices = np.flatnonzero(np.isfinite(log_ratio))
        if finite_indices.size < 2:
            continue

        inner_indices = finite_indices[inside_r200c[finite_indices]]
        exact_roots = inner_indices[log_ratio[inner_indices] == 0.0]
        roots = [(radius_r200c[index], radius_r200c[index], radius_r200c[index]) for index in exact_roots]
        brackets = []
        for left, right in zip(inner_indices[:-1], inner_indices[1:]):
            if right != left + 1:
                continue
            if log_ratio[left] < 0.0 and log_ratio[right] > 0.0:
                brackets.append((int(left), int(right)))

        for left, right in brackets:
            x_left = np.log10(radius_r200c[left])
            x_right = np.log10(radius_r200c[right])
            y_left = log_ratio[left]
            y_right = log_ratio[right]
            log_root = x_left - y_left * (x_right - x_left) / (y_right - y_left)
            roots.append((10.0 ** log_root, radius_r200c[left], radius_r200c[right]))

        # Outside R200c: T, Z, and Lambda are constant, so t_cool(x) = t_cool(1) * x**2.
        boundary_ratio = cooling_time_over_tdyn[row, boundary_index]
        if np.isfinite(boundary_ratio) and 0.0 < boundary_ratio < 1.0:
            outer_root = 1.0 / np.sqrt(boundary_ratio)
            if np.isfinite(outer_root):
                outer_cooling_radius_r200c[row] = outer_root
                roots.append((outer_root, outer_root, outer_root))

        root_count[row] = len(roots)
        if roots:
            cooling_radius_r200c[row], root_lower_r200c[row], root_upper_r200c[row] = min(roots, key=lambda result: result[0])
            cooling_radius_kpc[row] = cooling_radius_r200c[row] * r200_kpc[row]
            status[row] = status_root_found
        elif np.all(log_ratio[finite_indices] > 0.0):
            status[row] = status_below_profile_range
        elif np.all(log_ratio[finite_indices] < 0.0):
            status[row] = status_above_profile_range
        else:
            status[row] = status_no_outward_crossing

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_file, "w") as f:
        f.create_dataset("tree_row", data=tree_row)
        f.create_dataset("start_subhalo_id", data=start_ids)
        f.create_dataset("subhalo_id", data=node_ids)
        f.create_dataset("fof_id", data=fof_ids)
        f.create_dataset("analysis_valid", data=analysis_valid.astype(np.uint8))
        f.create_dataset("m200c_msun", data=halo_mass_msun)
        f.create_dataset("r200c_physical_kpc", data=r200_kpc)
        f.create_dataset("source_interpolation_radii_r200c", data=source_radius_r200c)
        f.create_dataset("interpolation_radii_r200c", data=radius_r200c)
        f.create_dataset("boundary_extrapolation_start_r200c", data=anchor_radii)
        f.create_dataset("temperature_at_r200c_K", data=temperature[:, boundary_index])
        f.create_dataset("density_at_r200c_g_cm3", data=density[:, boundary_index])
        f.create_dataset("metallicity_at_r200c_mass_fraction", data=metallicity[:, boundary_index])
        f.create_dataset("temperature_profile_K", data=temperature, compression="gzip", compression_opts=4)
        f.create_dataset("density_profile_g_cm3", data=density, compression="gzip", compression_opts=4)
        f.create_dataset("metallicity_profile_mass_fraction", data=metallicity, compression="gzip", compression_opts=4)
        f.create_dataset("v200c_kms", data=v200_kms)
        f.create_dataset("t_dyn_s", data=t_dyn_s)
        f.create_dataset("cooling_function_erg_cm3_s", data=cooling_function, compression="gzip", compression_opts=4)
        f.create_dataset("cooling_time_s", data=cooling_time_s, compression="gzip", compression_opts=4)
        f.create_dataset("cooling_time_over_tdyn", data=cooling_time_over_tdyn, compression="gzip", compression_opts=4)
        f.create_dataset("cooling_radius_r200c", data=cooling_radius_r200c)
        f.create_dataset("cooling_radius_physical_kpc", data=cooling_radius_kpc)
        f.create_dataset("outer_cooling_radius_r200c", data=outer_cooling_radius_r200c)
        f.create_dataset("cooling_radius_is_outside_r200c", data=(cooling_radius_r200c > 1.0).astype(np.uint8))
        f.create_dataset("root_bracket_r200c", data=np.column_stack((root_lower_r200c, root_upper_r200c)))
        f.create_dataset("root_count", data=root_count)
        f.create_dataset("status", data=status)

    n_root = np.count_nonzero(status == status_root_found)
    print(f"Saved real-profile cooling radii for snap {snap}: {n_root}/{n_rows} roots found -> {output_file}", flush=True)
    return output_file, cooling_radius_r200c, status
