from pathlib import Path
import sys
import h5py
import numpy as np
import cooling_radius as cr
from real_cooling_radius_all import extend_hot_gas_profiles

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

model_definitions = (
    ("000_Lgal_baseline", False, False, False),
    ("100_rho_profile", True, False, False),
    ("010_temperature_profile", False, True, False),
    ("001_metallicity_profile", False, False, True),
    ("110_rho_temperature_profiles", True, True, False),
    ("101_rho_metallicity_profiles", True, False, True),
    ("011_temperature_metallicity_profiles", False, True, True),
    ("111_TNG_profiles", True, True, True),
)


def find_cooling_radius(radius_r200c, cooling_time_over_tdyn):

    ratio = np.asarray(cooling_time_over_tdyn, dtype=np.float64)
    valid_ratio = np.isfinite(ratio) & (ratio > 0.0)
    finite_indices = np.flatnonzero(valid_ratio)
    if finite_indices.size < 2:
        return np.nan, np.nan, np.nan, 0, status_invalid_input, np.nan

    log_ratio = np.full(ratio.shape, np.nan, dtype=np.float64)
    log_ratio[valid_ratio] = np.log10(ratio[valid_ratio])
    inner_indices = finite_indices[radius_r200c[finite_indices] <= 1.0]
    roots = []

    for index in inner_indices:
        if log_ratio[index] == 0.0:
            roots.append((radius_r200c[index], radius_r200c[index], radius_r200c[index]))

    for left, right in zip(inner_indices[:-1], inner_indices[1:]):
        if right != left + 1 or not (log_ratio[left] < 0.0 and log_ratio[right] > 0.0):
            continue
        x_left = np.log10(radius_r200c[left])
        x_right = np.log10(radius_r200c[right])
        y_left = log_ratio[left]
        y_right = log_ratio[right]
        log_root = x_left - y_left * (x_right - x_left) / (y_right - y_left)
        roots.append((10.0 ** log_root, radius_r200c[left], radius_r200c[right]))

    boundary_indices = np.flatnonzero(np.isclose(radius_r200c, 1.0, rtol=0.0, atol=1.0e-12))
    outer_root = np.nan
    if boundary_indices.size and np.isfinite(ratio[boundary_indices[0]]) and 0.0 < ratio[boundary_indices[0]] < 1.0:
        outer_root = 1.0 / np.sqrt(ratio[boundary_indices[0]])
        roots.append((outer_root, outer_root, outer_root))

    if roots:
        root, lower, upper = min(roots, key=lambda result: result[0])
        return root, lower, upper, len(roots), status_root_found, outer_root
    if np.all(log_ratio[finite_indices] > 0.0):
        status = status_below_profile_range
    elif np.all(log_ratio[finite_indices] < 0.0):
        status = status_above_profile_range
    else:
        status = status_no_outward_crossing
    return np.nan, np.nan, np.nan, 0, status, outer_root


def calculate_replacement_cooling_radii(snap=99, properties_file=None, tree_file=None, output_file=None, table_dir=None, outer_max_r200c=5.0):

    properties_file = data_dir / f"real_physical_properties_all_{snap}.hdf5" if properties_file is None else Path(properties_file)
    tree_file = data_dir / "all_selected_trees.hdf5" if tree_file is None else Path(tree_file)
    output_file = data_dir / f"real_cooling_radius_replace_{snap}.hdf5" if output_file is None else Path(output_file)

    with h5py.File(properties_file, "r") as f:
        tree_row = np.asarray(f["tree_row"][:], dtype=np.int64)
        start_ids = np.asarray(f["start_subhalo_id"][:], dtype=np.int64)
        node_ids = np.asarray(f["subhalo_id"][:], dtype=np.int64)
        fof_ids = np.asarray(f["fof_id"][:], dtype=np.int64)
        analysis_valid = np.asarray(f["analysis_valid"][:], dtype=bool)
        halo_mass_msun = np.asarray(f["m200c_msun"][:], dtype=np.float64)
        r200_kpc = np.asarray(f["r200c_physical_kpc"][:], dtype=np.float64)
        hot_mass_msun = np.asarray(f["hot_gas_mass_msun_within_r200c"][:], dtype=np.float64)
        radius_r200c = np.asarray(f["interpolation_radii_r200c"][:], dtype=np.float64)
        temperature_source = np.asarray(f["temperature_mass_weighted_K_interpolated"][:], dtype=np.float64)
        density_source = np.asarray(f["rho_hot_cell_mean_g_cm3_interpolated"][:], dtype=np.float64)
        metallicity_source = np.asarray(f["metallicity_mass_weighted_interpolated"][:], dtype=np.float64)
        global_temperature = np.asarray(f["global_temperature_mass_weighted_K"][:], dtype=np.float64)
        global_metallicity = np.asarray(f["global_metallicity_mass_weighted"][:], dtype=np.float64)

    num_halos = start_ids.size
    if np.isclose(radius_r200c[-1], 1.0, rtol=0.0, atol=1.0e-12):
        radius_r200c[-1] = 1.0

    solver_radii, temperature_profile, density_profile, metallicity_profile, anchor_radii = extend_hot_gas_profiles(
        radius_r200c, temperature_source, density_source, metallicity_source, outer_max_r200c=outer_max_r200c
    )
    boundary_matches = np.flatnonzero(np.isclose(solver_radii, 1.0, rtol=0.0, atol=1.0e-12))
    boundary_index = int(boundary_matches[0])

    r200_cm = r200_kpc * kpc_cm
    v200_kms = np.full(num_halos, np.nan, dtype=np.float64)
    valid_virial = np.isfinite(halo_mass_msun) & (halo_mass_msun > 0.0) & np.isfinite(r200_kpc) & (r200_kpc > 0.0)
    v200_kms[valid_virial] = np.sqrt(G_kpc_kms2_msun * halo_mass_msun[valid_virial] / r200_kpc[valid_virial])
    t_dyn_s = np.full(num_halos, np.nan, dtype=np.float64)
    t_dyn_s[valid_virial] = r200_cm[valid_virial] / (v200_kms[valid_virial] * 1.0e5)
    t200_k = 35.9 * v200_kms ** 2
    hot_mass_g = hot_mass_msun * msun_g
    physical_valid = (analysis_valid & valid_virial & np.isfinite(hot_mass_g) & (hot_mass_g > 0.0)
        & np.isfinite(t_dyn_s) & (t_dyn_s > 0.0))

    cooling_tables = cr.CoolingTableLoader(table_dir=table_dir)
    results = {}

    for model_name, replace_density, replace_temperature, replace_metallicity in model_definitions:
        cooling_radius_r200c = np.full(num_halos, np.nan, dtype=np.float64)
        root_lower_r200c = np.full(num_halos, np.nan, dtype=np.float64)
        root_upper_r200c = np.full(num_halos, np.nan, dtype=np.float64)
        outer_cooling_radius_r200c = np.full(num_halos, np.nan, dtype=np.float64)
        lambda_at_r200c = np.full(num_halos, np.nan, dtype=np.float64)
        tcool_over_tdyn_at_r200c = np.full(num_halos, np.nan, dtype=np.float64)
        root_count = np.zeros(num_halos, dtype=np.int16)
        status = np.full(num_halos, status_invalid_input, dtype=np.uint8)
        input_valid = physical_valid.copy()

        if model_name == "000_Lgal_baseline":
            lambda_global = np.asarray(cr.LgalCoolingFunction(global_temperature, global_metallicity, cooling_tables), dtype=np.float64)
            input_valid &= (np.isfinite(global_temperature) & (global_temperature > 0.0)
                & np.isfinite(global_metallicity) & (global_metallicity >= 0.0)
                & np.isfinite(lambda_global) & (lambda_global > 0.0)
                & np.isfinite(t200_k) & (t200_k > 0.0))
            valid_rows = np.flatnonzero(input_valid)
            cooling_radius_r200c[valid_rows] = np.sqrt(
                hot_mass_g[valid_rows] * t_dyn_s[valid_rows] * lambda_global[valid_rows]
                / (6.0 * np.pi * mu * m_H_g * k_B_erg_K * t200_k[valid_rows] * r200_cm[valid_rows] ** 3)
            )
            root_lower_r200c[valid_rows] = cooling_radius_r200c[valid_rows]
            root_upper_r200c[valid_rows] = cooling_radius_r200c[valid_rows]
            root_count[valid_rows] = 1
            status[valid_rows] = status_root_found
            lambda_at_r200c[valid_rows] = lambda_global[valid_rows]
            tcool_over_tdyn_at_r200c[valid_rows] = 1.0 / cooling_radius_r200c[valid_rows] ** 2
            outer_rows = valid_rows[cooling_radius_r200c[valid_rows] > 1.0]
            outer_cooling_radius_r200c[outer_rows] = cooling_radius_r200c[outer_rows]
        else:
            for row in np.flatnonzero(physical_valid):
                if replace_density:
                    rho_eff = density_profile[row]
                    input_valid[row] &= np.any(np.isfinite(rho_eff) & (rho_eff > 0.0))
                else:
                    rho_eff = hot_mass_g[row] / (4.0 * np.pi * r200_cm[row] ** 3 * solver_radii ** 2)

                if replace_temperature:
                    temperature_energy = temperature_profile[row]
                    temperature_lambda = temperature_profile[row]
                    input_valid[row] &= np.any(np.isfinite(temperature_profile[row]) & (temperature_profile[row] > 0.0))
                else:
                    temperature_energy = np.full(solver_radii.size, t200_k[row], dtype=np.float64)
                    temperature_lambda = np.full(solver_radii.size, global_temperature[row], dtype=np.float64)
                    input_valid[row] &= (np.isfinite(global_temperature[row]) & (global_temperature[row] > 0.0)
                        & np.isfinite(t200_k[row]) & (t200_k[row] > 0.0))

                if replace_metallicity:
                    metallicity_eff = metallicity_profile[row]
                    input_valid[row] &= np.any(np.isfinite(metallicity_eff) & (metallicity_eff >= 0.0))
                else:
                    metallicity_eff = np.full(solver_radii.size, global_metallicity[row], dtype=np.float64)
                    input_valid[row] &= np.isfinite(global_metallicity[row]) & (global_metallicity[row] >= 0.0)

                if not input_valid[row]:
                    continue

                valid_profile = (np.isfinite(rho_eff) & (rho_eff > 0.0)
                    & np.isfinite(temperature_energy) & (temperature_energy > 0.0)
                    & np.isfinite(temperature_lambda) & (temperature_lambda > 0.0)
                    & np.isfinite(metallicity_eff) & (metallicity_eff >= 0.0))
                if np.count_nonzero(valid_profile) < 2:
                    input_valid[row] = False
                    continue

                lambda_row = np.asarray(cr.LgalCoolingFunction(temperature_lambda, metallicity_eff, cooling_tables), dtype=np.float64)
                valid_cooling = valid_profile & np.isfinite(lambda_row) & (lambda_row > 0.0)
                if np.count_nonzero(valid_cooling) < 2:
                    input_valid[row] = False
                    continue

                cooling_time_over_tdyn = np.full(solver_radii.size, np.nan, dtype=np.float64)
                cooling_time_over_tdyn[valid_cooling] = (
                    1.5 * mu * m_H_g * k_B_erg_K * temperature_energy[valid_cooling]
                    / (rho_eff[valid_cooling] * lambda_row[valid_cooling] * t_dyn_s[row])
                )
                root, lower, upper, number_of_roots, root_status, outer_root = find_cooling_radius(solver_radii, cooling_time_over_tdyn)
                cooling_radius_r200c[row] = root
                root_lower_r200c[row] = lower
                root_upper_r200c[row] = upper
                root_count[row] = number_of_roots
                status[row] = root_status
                outer_cooling_radius_r200c[row] = outer_root
                if valid_cooling[boundary_index]:
                    lambda_at_r200c[row] = lambda_row[boundary_index]
                    tcool_over_tdyn_at_r200c[row] = cooling_time_over_tdyn[boundary_index]

        cooling_radius_physical_kpc = cooling_radius_r200c * r200_kpc
        root_found = status == status_root_found
        results[model_name] = {
            "replace_density_profile": replace_density,
            "replace_temperature_profile": replace_temperature,
            "replace_metallicity_profile": replace_metallicity,
            "input_valid": input_valid,
            "valid": root_found,
            "cooling_radius_r200c": cooling_radius_r200c,
            "cooling_radius_physical_kpc": cooling_radius_physical_kpc,
            "cooling_radius_is_outside_r200c": (root_found & (cooling_radius_r200c > 1.0)),
            "cooling_radius_within_solver_range": (root_found & (cooling_radius_r200c >= solver_radii[0]) & (cooling_radius_r200c <= solver_radii[-1])),
            "root_lower_r200c": root_lower_r200c,
            "root_upper_r200c": root_upper_r200c,
            "root_count": root_count,
            "outer_cooling_radius_r200c": outer_cooling_radius_r200c,
            "cooling_function_at_r200c_erg_cm3_s": lambda_at_r200c,
            "cooling_time_over_tdyn_at_r200c": tcool_over_tdyn_at_r200c,
            "status": status,
        }
        print(f"{model_name}: {np.count_nonzero(root_found)}/{num_halos} roots found; {np.count_nonzero(input_valid)} rows have valid inputs.", flush=True)

    reference_radius = np.full(num_halos, np.nan, dtype=np.float64)
    reference_valid = np.zeros(num_halos, dtype=bool)
    reference_file = data_dir / f"cooling_radius_{snap}.hdf5"
    if reference_file.is_file():
        with h5py.File(reference_file, "r") as f:
            reference_radius_all = np.asarray(f["cooling_radius"][:], dtype=np.float64)
            reference_halo_flag = np.asarray(f["halo_flag"][:], dtype=bool)

        in_bounds = (fof_ids >= 0) & (fof_ids < reference_radius_all.size)
        rows = np.flatnonzero(in_bounds)
        reference_radius[rows] = reference_radius_all[fof_ids[rows]]
        reference_valid[rows] = reference_halo_flag[fof_ids[rows]] & np.isfinite(reference_radius[rows]) & (reference_radius[rows] >= 0.0)

    tng_reference_radius = np.full(num_halos, np.nan, dtype=np.float64)
    tng_reference_valid = np.zeros(num_halos, dtype=bool)
    tng_reference_file = data_dir / f"real_cooling_radius_{snap}.hdf5"
    if tng_reference_file.is_file():
        with h5py.File(tng_reference_file, "r") as f:
            tng_reference_radius = np.asarray(f["cooling_radius_r200c"][:], dtype=np.float64)
            tng_reference_analysis_valid = np.asarray(f["analysis_valid"][:], dtype=bool)
            tng_reference_status = np.asarray(f["status"][:], dtype=np.uint8)
        tng_reference_valid = (tng_reference_analysis_valid & (tng_reference_status == status_root_found)
            & np.isfinite(tng_reference_radius) & (tng_reference_radius > 0.0))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_file, "w") as f:

        inputs = f.create_group("inputs")
        common_inputs = {
            "tree_row": tree_row,
            "start_subhalo_id": start_ids,
            "subhalo_id": node_ids,
            "fof_id": fof_ids,
            "analysis_valid": analysis_valid.astype(np.uint8),
            "m200c_msun": halo_mass_msun,
            "r200c_physical_kpc": r200_kpc,
            "hot_gas_mass_msun_within_r200c": hot_mass_msun,
            "global_temperature_mass_weighted_K": global_temperature,
            "global_metallicity_mass_weighted": global_metallicity,
            "v200c_kms": v200_kms,
            "t200c_K": t200_k,
            "t_dyn_s": t_dyn_s,
            "source_interpolation_radii_r200c": radius_r200c,
            "solver_radii_r200c": solver_radii,
            "boundary_extrapolation_start_r200c": anchor_radii,
            "reference_lgal_cooling_radius_r200c": reference_radius,
            "reference_lgal_halo_valid": reference_valid.astype(np.uint8),
            "reference_tng_cooling_radius_r200c": tng_reference_radius,
            "reference_tng_halo_valid": tng_reference_valid.astype(np.uint8),
        }
        for name, values in common_inputs.items():
            inputs.create_dataset(name, data=values, compression="gzip", compression_opts=4)

        for model_name, model_result in results.items():
            model_group = f.create_group(model_name)
            model_group.attrs["replace_density_profile"] = np.uint8(model_result["replace_density_profile"])
            model_group.attrs["replace_temperature_profile"] = np.uint8(model_result["replace_temperature_profile"])
            model_group.attrs["replace_metallicity_profile"] = np.uint8(model_result["replace_metallicity_profile"])
            for quantity_name, values in model_result.items():
                if quantity_name in ("replace_density_profile", "replace_temperature_profile", "replace_metallicity_profile"):
                    continue
                model_group.create_dataset(quantity_name, data=np.asarray(values, dtype=np.uint8 if quantity_name in ("input_valid", "valid", "cooling_radius_is_outside_r200c", "status", "cooling_radius_within_solver_range") else None), compression="gzip", compression_opts=4)

            if model_name == "000_Lgal_baseline":
                comparison_valid = (model_group["status"][:] == status_root_found) & reference_valid & np.isfinite(reference_radius)
                delta = np.full(num_halos, np.nan, dtype=np.float64)
                delta[comparison_valid] = model_result["cooling_radius_r200c"][comparison_valid] - reference_radius[comparison_valid]
                model_group.create_dataset("reference_radius_difference_r200c", data=delta, compression="gzip", compression_opts=4)

            elif model_name == "111_TNG_profiles" and tng_reference_file.is_file():
                comparison_valid = (model_group["status"][:] == status_root_found) & tng_reference_valid
                delta = np.full(num_halos, np.nan, dtype=np.float64)
                delta[comparison_valid] = model_result["cooling_radius_r200c"][comparison_valid] - tng_reference_radius[comparison_valid]
                model_group.create_dataset("reference_radius_difference_r200c", data=delta, compression="gzip", compression_opts=4)

    return 