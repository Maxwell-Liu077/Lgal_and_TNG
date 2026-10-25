from pathlib import Path
import tempfile

import h5py
import illustris_python as il
import numpy as np
import cooling_radius as lgal_cooling
import funcs
import illustrisFuncs as iF
import real_cooling_radius as tng_cooling
import real_physical_properties_all as real_properties


def select_event_halo(basePath, snap_min, snap_max):

    data_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")
    tree_file = data_dir / "all_selected_trees.hdf5"
    output_file = data_dir / "event_object_halos.hdf5"

    major_merger_ratio_min = 0.25
    random_seed = 42

    with h5py.File(tree_file, "r") as f:
        start_ids = np.asarray(f["start_subhalo_id"][:], dtype=np.int64)

    header = il.groupcat.loadHeader(basePath, snap_max)
    h = header["HubbleParam"]
    tree_fields = ["SubhaloID", "SubfindID", "SnapNum", "FirstProgenitorID", "NextProgenitorID", "MainLeafProgenitorID", "SubhaloMassType"]
    tree_cache = {}
    candidate_tree_rows = []
    candidate_start_ids = []
    candidate_merger_counts = []
    candidate_unmeasurable_counts = []
    candidate_events = []
    unmeasurable_pairs_total = 0

    for tree_row, start_id in enumerate(start_ids):
        tree = il.sublink.loadTree(basePath, snap_max, start_id, fields=tree_fields, onlyMPB=False, treeName="SubLink", cache=tree_cache)

        subhalo_tree_ids = np.asarray(tree["SubhaloID"], dtype=np.int64)
        subfind_ids = np.asarray(tree["SubfindID"], dtype=np.int64)
        merger_snaps = np.asarray(tree["SnapNum"], dtype=np.int64)
        first_progenitor_ids = np.asarray(tree["FirstProgenitorID"], dtype=np.int64)
        next_progenitor_ids = np.asarray(tree["NextProgenitorID"], dtype=np.int64)
        main_leaf_ids = np.asarray(tree["MainLeafProgenitorID"], dtype=np.int64)
        stellar_mass_code = np.asarray(tree["SubhaloMassType"], dtype=np.float64)[:, 4]

        row_by_tree_id = {int(tree_id): row for row, tree_id in enumerate(subhalo_tree_ids)}
        tree_events = []
        unmeasurable_count = 0
        descendant_row = 0

        # Walk only the z=0 main branch; each first progenitor's NextProgenitor chain
        # contains the other branches that merge into the same descendant.
        while descendant_row < subhalo_tree_ids.size:
            merger_snap = int(merger_snaps[descendant_row])
            primary_id = int(first_progenitor_ids[descendant_row])
            if primary_id < 0:
                break
            primary_row = row_by_tree_id.get(primary_id)
            if primary_row is None:
                break

            if snap_min <= merger_snap <= snap_max:
                primary_at_snap = {}
                main_row = primary_row
                while main_row is not None:
                    main_snap = int(merger_snaps[main_row])
                    if main_snap not in primary_at_snap:
                        primary_at_snap[main_snap] = main_row
                    next_main_id = int(first_progenitor_ids[main_row])
                    main_row = row_by_tree_id.get(next_main_id) if next_main_id >= 0 else None

                secondary_id = int(next_progenitor_ids[primary_row])
                while secondary_id >= 0:
                    secondary_row = row_by_tree_id.get(secondary_id)
                    if secondary_row is None:
                        unmeasurable_count += 1
                        break

                    leaf_id = int(main_leaf_ids[secondary_row])
                    leaf_row = row_by_tree_id.get(leaf_id)
                    if leaf_row is None or leaf_row < secondary_row:
                        unmeasurable_count += 1
                    else:
                        branch_masses = stellar_mass_code[secondary_row:leaf_row + 1]
                        valid_branch_masses = np.isfinite(branch_masses) & (branch_masses > 0.0)
                        if not np.any(valid_branch_masses):
                            unmeasurable_count += 1
                        else:
                            valid_branch_rows = np.flatnonzero(valid_branch_masses)
                            peak_offset = valid_branch_rows[np.argmax(branch_masses[valid_branch_masses])]
                            secondary_peak_row = secondary_row + int(peak_offset)
                            secondary_peak_snap = int(merger_snaps[secondary_peak_row])
                            primary_peak_row = primary_at_snap.get(secondary_peak_snap)

                            if primary_peak_row is None or not np.isfinite(stellar_mass_code[primary_peak_row]) or stellar_mass_code[primary_peak_row] <= 0.0:
                                unmeasurable_count += 1
                            else:
                                secondary_peak_mass = stellar_mass_code[secondary_peak_row] * 1.0e10 / h
                                primary_peak_mass = stellar_mass_code[primary_peak_row] * 1.0e10 / h
                                mass_ratio = min(secondary_peak_mass, primary_peak_mass) / max(secondary_peak_mass, primary_peak_mass)

                                if mass_ratio >= major_merger_ratio_min:
                                    tree_events.append((merger_snap, int(subhalo_tree_ids[descendant_row]), primary_id, secondary_id, secondary_peak_snap, int(subhalo_tree_ids[primary_peak_row]), int(subfind_ids[secondary_peak_row]), int(subfind_ids[primary_peak_row]), mass_ratio, secondary_peak_mass, primary_peak_mass))

                    secondary_id = int(next_progenitor_ids[secondary_row])

            descendant_row = primary_row

        unmeasurable_pairs_total += unmeasurable_count
        if tree_events:
            candidate_tree_rows.append(tree_row)
            candidate_start_ids.append(int(start_id))
            candidate_merger_counts.append(len(tree_events))
            candidate_unmeasurable_counts.append(unmeasurable_count)
            candidate_events.append(tree_events)

        if (tree_row + 1) % 1000 == 0:
            print(f"Scanned {tree_row + 1}/{start_ids.size} selected trees; major-merger candidates={len(candidate_tree_rows)}.", flush=True)

    candidate_tree_rows = np.asarray(candidate_tree_rows, dtype=np.int64)
    candidate_start_ids = np.asarray(candidate_start_ids, dtype=np.int64)
    candidate_merger_counts = np.asarray(candidate_merger_counts, dtype=np.int64)
    candidate_unmeasurable_counts = np.asarray(candidate_unmeasurable_counts, dtype=np.int64)

    rng = np.random.default_rng(random_seed)
    selected_candidate_indices = rng.choice(candidate_tree_rows.size, size=3, replace=False)
    selected_tree_rows = candidate_tree_rows[selected_candidate_indices]
    selected_start_ids = candidate_start_ids[selected_candidate_indices]
    selected_merger_counts = candidate_merger_counts[selected_candidate_indices]

    selected_event_offsets = np.zeros(4, dtype=np.int64)
    selected_events = []
    for selected_index, candidate_index in enumerate(selected_candidate_indices):
        selected_events.extend(candidate_events[int(candidate_index)])
        selected_event_offsets[selected_index + 1] = len(selected_events)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_file, "w") as f:
        f.create_dataset("candidate_tree_row", data=candidate_tree_rows)
        f.create_dataset("candidate_start_subhalo_id", data=candidate_start_ids)
        f.create_dataset("candidate_merger_count", data=candidate_merger_counts)
        f.create_dataset("candidate_unmeasurable_pair_count", data=candidate_unmeasurable_counts)
        f.create_dataset("selected_candidate_index", data=selected_candidate_indices)
        f.create_dataset("selected_tree_row", data=selected_tree_rows)
        f.create_dataset("selected_start_subhalo_id", data=selected_start_ids)
        f.create_dataset("selected_merger_count", data=selected_merger_counts)
        f.create_dataset("selected_event_offsets", data=selected_event_offsets)
        event_names = ["merger_snap", "descendant_tree_subhalo_id", "primary_progenitor_tree_subhalo_id", "secondary_progenitor_tree_subhalo_id", "secondary_peak_snap", "primary_tree_subhalo_id_at_peak", "secondary_subfind_id_at_peak", "primary_subfind_id_at_peak", "stellar_mass_ratio", "secondary_peak_stellar_mass_msun", "primary_stellar_mass_at_peak_msun"]
        for event_column, event_name in enumerate(event_names):
            event_values = np.asarray([event[event_column] for event in selected_events], dtype=np.int64 if event_column < 8 else np.float64)
            f.create_dataset(event_name, data=event_values)

    print(f"Found {candidate_tree_rows.size} major-merger trees, sampled three with seed={random_seed}, saved to {output_file}.", flush=True)
    return output_file, selected_tree_rows, selected_start_ids


def object_cooling_radius(basePath):

    data_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")
    tree_file = data_dir / "all_selected_trees.hdf5"
    event_file = data_dir / "event_object_halos.hdf5"
    output_file = data_dir / "event_object_cooling_radius_40_to_99.hdf5"
    snaps = np.arange(40, 100, dtype=np.int64)

    with h5py.File(event_file, "r") as f:
        selected_tree_rows = np.asarray(f["selected_tree_row"][:], dtype=np.int64)
        selected_start_ids = np.asarray(f["selected_start_subhalo_id"][:], dtype=np.int64)
        selected_merger_counts = np.asarray(f["selected_merger_count"][:], dtype=np.int64)
        selected_event_offsets = np.asarray(f["selected_event_offsets"][:], dtype=np.int64)
        event_merger_snaps = np.asarray(f["merger_snap"][:], dtype=np.int64)
        event_mass_ratios = np.asarray(f["stellar_mass_ratio"][:], dtype=np.float64)

    with h5py.File(tree_file, "r") as f:
        tree_snaps = np.asarray(f["snap_numbers"][:], dtype=np.int64)

        snap_columns = []
        for snap in snaps:
            matches = np.flatnonzero(tree_snaps == snap)
            snap_columns.append(int(matches[0]))
        row_sort = np.argsort(selected_tree_rows)
        sorted_tree_rows = selected_tree_rows[row_sort]
        restore_sample_order = np.argsort(row_sort)
        node_ids = np.asarray(f["mpb_subhalo_ids"][sorted_tree_rows, :], dtype=np.int64)[restore_sample_order][:, snap_columns]
        fof_ids = np.asarray(f["mpb_fof_ids"][sorted_tree_rows, :], dtype=np.int64)[restore_sample_order][:, snap_columns]

    n_objects = selected_tree_rows.size
    n_snaps = snaps.size

    data_dir.mkdir(parents=True, exist_ok=True)
    msun_g = 1.98847e33
    radial_edges = np.geomspace(real_properties.radial_min, real_properties.radial_max, real_properties.num_radial_bins + 1)
    radial_centers = np.sqrt(radial_edges[:-1] * radial_edges[1:])
    interpolation_radii = np.geomspace(real_properties.radial_min, real_properties.radial_max, 200)
    cooling_tables = lgal_cooling.CoolingTableLoader()
    tng_radius = np.full((n_objects, n_snaps), np.nan, dtype=np.float64)
    lgal_radius = np.full((n_objects, n_snaps), np.nan, dtype=np.float64)
    tng_valid = np.zeros((n_objects, n_snaps), dtype=bool)
    lgal_valid = np.zeros((n_objects, n_snaps), dtype=bool)
    analysis_valid_history = np.zeros((n_objects, n_snaps), dtype=bool)
    tng_status = np.zeros((n_objects, n_snaps), dtype=np.uint8)
    halo_mass_msun = np.full((n_objects, n_snaps), np.nan, dtype=np.float64)
    scale_factor = np.full(n_snaps, np.nan, dtype=np.float64)
    redshift = np.full(n_snaps, np.nan, dtype=np.float64)

    for snap_index, snap in enumerate(snaps):
        header = il.groupcat.loadHeader(basePath, int(snap))
        hubble_param = float(header["HubbleParam"])
        scale_factor[snap_index] = float(header["Time"])
        redshift[snap_index] = float(header["Redshift"])
        group_data = il.groupcat.loadHalos(basePath, int(snap), fields=["Group_M_Crit200", "Group_R_Crit200", "GroupPos", "GroupFirstSub"])
        group_mass_raw = np.asarray(group_data["Group_M_Crit200"], dtype=np.float64)
        group_radius_raw = np.asarray(group_data["Group_R_Crit200"], dtype=np.float64)
        group_pos = np.asarray(group_data["GroupPos"], dtype=np.float64)
        group_first_sub = np.asarray(group_data["GroupFirstSub"], dtype=np.int64)
        box_size = float(header["BoxSize"])

        analysis_valid = np.zeros(n_objects, dtype=bool)
        density_profiles = np.full((n_objects, interpolation_radii.size), np.nan, dtype=np.float64)
        temperature_profiles = np.full_like(density_profiles, np.nan)
        metallicity_profiles = np.full_like(density_profiles, np.nan)
        for object_index in range(n_objects):
            node_id = int(node_ids[object_index, snap_index])
            fof_id = int(fof_ids[object_index, snap_index])
            if not (0 <= node_id < int(header["Nsubgroups_Total"]) and 0 <= fof_id < group_mass_raw.size):
                continue
            if group_first_sub[fof_id] != node_id:
                continue
            if not (np.isfinite(group_mass_raw[fof_id]) and group_mass_raw[fof_id] > 0.0 and np.isfinite(group_radius_raw[fof_id]) and group_radius_raw[fof_id] > 0.0 and np.all(np.isfinite(group_pos[fof_id]))):
                continue

            analysis_valid[object_index] = True
            analysis_valid_history[object_index, snap_index] = True
            halo_mass_msun[object_index, snap_index] = group_mass_raw[fof_id] * 1.0e10 / hubble_param
            r200_raw = group_radius_raw[fof_id]
            gas_fields = ["Coordinates", "Masses", "StarFormationRate", "InternalEnergy", "ElectronAbundance", "GFM_Metallicity", "Density"]
            gas = il.snapshot.loadHalo(basePath, int(snap), fof_id, "gas", fields=gas_fields)
            if gas.get("count", 0) == 0:
                continue

            gas_pos = np.asarray(gas["Coordinates"], dtype=np.float64)
            gas_mass_code = np.asarray(gas["Masses"], dtype=np.float64)
            gas_sfr = np.asarray(gas["StarFormationRate"], dtype=np.float64)
            gas_internal_energy = np.asarray(gas["InternalEnergy"], dtype=np.float64)
            gas_electron_abundance = np.asarray(gas["ElectronAbundance"], dtype=np.float64)
            gas_metallicity = np.asarray(gas["GFM_Metallicity"], dtype=np.float64)
            gas_density = np.asarray(gas["Density"], dtype=np.float64)

            # L-GALAXIES analytic cooling radius: reproduce cooling_radius.py's gas selection for this FoF only.
            gas_state = np.zeros(gas_sfr.size, dtype=np.uint8)
            gas_state[np.nonzero(gas_sfr)[0]] = 1
            non_star_forming = gas_state == 0
            gas_temperature = np.full(gas_sfr.shape, np.nan, dtype=np.float64)
            gas_temperature[non_star_forming] = iF.utherm_to_temp(gas_internal_energy[non_star_forming], gas_electron_abundance[non_star_forming])
            with np.errstate(divide="ignore", invalid="ignore"):
                gas_log_temperature = np.log10(gas_temperature)
            gas_state[gas_log_temperature <= 4.5] = 1

            halo_properties = lgal_cooling.compute_halo_properties(gas_mass_code, gas_pos, gas_metallicity, group_pos[fof_id:fof_id + 1], np.asarray([r200_raw]), np.asarray([group_mass_raw[fof_id]]), np.asarray([0], dtype=np.int64), np.asarray([gas_mass_code.size], dtype=np.int64), gas_state, gas_temperature, box_size, scale_factor[snap_index], hubble_param)
            t_hot, z_hot, m_hot, t_dyn, t200, r200_cm, halo_flag = halo_properties
            cooling_function = lgal_cooling.LgalCoolingFunction(t_hot, z_hot, cooling_tables)
            lgal_radius_one = lgal_cooling.compute_cooling_radius(m_hot, t_dyn, cooling_function, t200, r200_cm, np.asarray([r200_raw]), halo_flag, hubble_param, scale_factor[snap_index])
            if halo_flag[0] == 1 and np.isfinite(lgal_radius_one[0]) and lgal_radius_one[0] >= 0.0:
                lgal_radius[object_index, snap_index] = lgal_radius_one[0]
                lgal_valid[object_index, snap_index] = True

            # TNG measured cooling radius: build the same hot-gas profiles as real_physical_properties_all.py.
            distance_raw = funcs.dist_vector_nb(group_pos[fof_id], gas_pos, box_size)
            radius_ratio = distance_raw / r200_raw
            profile_temperature = np.full(gas_sfr.shape, np.nan, dtype=np.float64)
            profile_temperature[non_star_forming] = iF.utherm_to_temp(gas_internal_energy[non_star_forming], gas_electron_abundance[non_star_forming])
            hot = non_star_forming & np.isfinite(profile_temperature) & (profile_temperature > 10.0 ** 4.5) & np.isfinite(radius_ratio) & (radius_ratio <= 1.0)
            in_profile = hot & (radius_ratio >= real_properties.radial_min)
            if not np.any(in_profile):
                continue

            radial_bin = np.searchsorted(radial_edges, radius_ratio[in_profile], side="right") - 1
            radial_bin = np.minimum(radial_bin, real_properties.num_radial_bins - 1).astype(np.int64)
            mass_g = gas_mass_code[in_profile] * 1.0e10 / hubble_param * msun_g
            temperature_cells = profile_temperature[in_profile]
            metallicity_cells = gas_metallicity[in_profile]
            density_raw = gas_density[in_profile]
            density_msun_kpc3 = density_raw * 1.0e10 * hubble_param ** 2 / scale_factor[snap_index] ** 3
            mass_msun = mass_g / msun_g
            cell_volume_kpc3 = np.full(mass_msun.shape, np.nan, dtype=np.float64)
            valid_density = np.isfinite(density_msun_kpc3) & (density_msun_kpc3 > 0.0)
            np.divide(mass_msun, density_msun_kpc3, out=cell_volume_kpc3, where=valid_density)
            cell_volume_cm3 = cell_volume_kpc3 * real_properties.kpc_cm ** 3

            density_profile, _, _, _ = real_properties.calculate_real_rou(mass_g, cell_volume_cm3, radial_bin)
            temperature_profile, _ = real_properties.calculate_real_T(mass_g, temperature_cells, radial_bin)
            metallicity_profile, _ = real_properties.calculate_real_Z(mass_g, metallicity_cells, radial_bin)
            density_interpolated = real_properties.interpolate_profile(radial_centers, density_profile, interpolation_radii, log_y=True)
            temperature_interpolated = real_properties.interpolate_profile(radial_centers, temperature_profile, interpolation_radii, log_y=True)
            metallicity_interpolated = real_properties.interpolate_profile(radial_centers, metallicity_profile, interpolation_radii, log_y=False)
            density_profiles[object_index] = density_interpolated
            temperature_profiles[object_index] = temperature_interpolated
            metallicity_profiles[object_index] = metallicity_interpolated

        # The three-row temporary file lets the existing root solver retain its status and extrapolation logic.
        with tempfile.TemporaryDirectory(prefix=f"event_object_snap{snap}_", dir=data_dir) as temporary_dir:
            properties_file = Path(temporary_dir) / f"real_physical_properties_all_{snap}.hdf5"
            radius_file = Path(temporary_dir) / f"real_cooling_radius_{snap}.hdf5"
            with h5py.File(properties_file, "w") as f:
                f.create_dataset("tree_row", data=selected_tree_rows)
                f.create_dataset("start_subhalo_id", data=selected_start_ids)
                f.create_dataset("subhalo_id", data=node_ids[:, snap_index])
                f.create_dataset("fof_id", data=fof_ids[:, snap_index])
                f.create_dataset("analysis_valid", data=analysis_valid.astype(np.uint8))
                f.create_dataset("m200c_msun", data=halo_mass_msun[:, snap_index])
                f.create_dataset("r200c_physical_kpc", data=np.asarray([group_radius_raw[fof] * scale_factor[snap_index] / hubble_param if analysis_valid[index] else np.nan for index, fof in enumerate(fof_ids[:, snap_index])], dtype=np.float64))
                f.create_dataset("interpolation_radii_r200c", data=interpolation_radii)
                f.create_dataset("temperature_mass_weighted_K_interpolated", data=temperature_profiles)
                f.create_dataset("rho_hot_cell_mean_g_cm3_interpolated", data=density_profiles)
                f.create_dataset("metallicity_mass_weighted_interpolated", data=metallicity_profiles)

            _, current_tng_radius, current_tng_status = tng_cooling.real_cooling_radius(int(snap), properties_file=properties_file, output_file=radius_file)
            current_tng_radius = np.asarray(current_tng_radius, dtype=np.float64)
            current_tng_status = np.asarray(current_tng_status, dtype=np.uint8)
            current_tng_valid = analysis_valid & (current_tng_status == tng_cooling.status_root_found) & np.isfinite(current_tng_radius) & (current_tng_radius > 0.0)
            tng_radius[current_tng_valid, snap_index] = current_tng_radius[current_tng_valid]
            tng_valid[:, snap_index] = current_tng_valid
            tng_status[:, snap_index] = current_tng_status

        del density_profiles, temperature_profiles, metallicity_profiles
        print(f"Snap {snap}: TNG valid={np.count_nonzero(tng_valid[:, snap_index])}/{n_objects}, L-GALAXIES valid={np.count_nonzero(lgal_valid[:, snap_index])}/{n_objects}.", flush=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_file, "w") as f:
        f.create_dataset("tree_row", data=selected_tree_rows)
        f.create_dataset("start_subhalo_id", data=selected_start_ids)
        f.create_dataset("selected_merger_count", data=selected_merger_counts)
        f.create_dataset("selected_event_offsets", data=selected_event_offsets)
        f.create_dataset("event_merger_snap", data=event_merger_snaps)
        f.create_dataset("event_stellar_mass_ratio", data=event_mass_ratios)
        f.create_dataset("snap_numbers", data=snaps)
        f.create_dataset("scale_factor", data=scale_factor)
        f.create_dataset("redshift", data=redshift)
        f.create_dataset("subhalo_id", data=node_ids)
        f.create_dataset("fof_id", data=fof_ids)
        f.create_dataset("analysis_valid", data=analysis_valid_history.astype(np.uint8))
        f.create_dataset("halo_mass_msun", data=halo_mass_msun)
        f.create_dataset("cooling_radius_tng_r200c", data=tng_radius)
        f.create_dataset("cooling_radius_lgal_r200c", data=lgal_radius)
        f.create_dataset("tng_valid", data=tng_valid.astype(np.uint8))
        f.create_dataset("lgal_valid", data=lgal_valid.astype(np.uint8))
        f.create_dataset("tng_status", data=tng_status)

    print(f"Saved the three sampled halos' cooling-radius histories to {output_file}.", flush=True)
    return output_file, tng_radius, lgal_radius
