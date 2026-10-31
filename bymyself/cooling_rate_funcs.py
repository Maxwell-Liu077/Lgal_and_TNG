import cooling_radius as cr
import h5py
import numpy as np
from os.path import isfile
from pathlib import Path
import illustris_python as il
import illustrisFuncs as iF
import funcs

def load_m200c_msun(basePath, snap, fof_ids, h):
    group_m200c = il.groupcat.loadHalos(basePath, snap, fields=["Group_M_Crit200"])
    group_m200c = np.asarray(group_m200c, dtype=np.float64).ravel()

    m200c = np.full(fof_ids.shape, np.nan, dtype=np.float64)
    in_bounds = (fof_ids >= 0) & (fof_ids < group_m200c.size)

    m200c[in_bounds] = group_m200c[fof_ids[in_bounds]] * 1.0e10 / h
    return m200c

def Lgal_cooling_rate(basePath, snap_1, snap_2):
    header_1 = il.groupcat.loadHeader(basePath, snap_1)
    header_2 = il.groupcat.loadHeader(basePath, snap_2)

    h_1 = header_1["HubbleParam"]
    h_2 = header_2["HubbleParam"]
    
    assert isfile(f'/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/mpb_halos.hdf5'), 'Mpb halos file does not exist!'
    tree_file = f'/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/all_selected_trees.hdf5'
    f = h5py.File(tree_file,'r')
    snap_numbers = np.asarray(f["snap_numbers"][:], dtype=np.int64)
    start_subhalo_ids = np.asarray(f["start_subhalo_id"][:], dtype=np.int64)
    num_trees = start_subhalo_ids.size

    col_1 = funcs.get_snap_column(snap_1, snap_numbers)
    col_2 = funcs.get_snap_column(snap_2, snap_numbers)

    halos_1 = np.asarray(f["mpb_fof_ids"][:, col_1], dtype=np.int64)
    halos_2 = np.asarray(f["mpb_fof_ids"][:, col_2], dtype=np.int64)

    m200c_1 = load_m200c_msun(basePath, snap_1, halos_1, h_1)
    m200c_2 = load_m200c_msun(basePath, snap_2, halos_2, h_2)

    cr_1, M_hot_1, _, t_dyn_1, HaloFlag_1 = cr.cooling_radius(basePath, snap_1)
    cr_2, M_hot_2, _, t_dyn_2, HaloFlag_2 = cr.cooling_radius(basePath, snap_2)

    rates = np.full(num_trees, np.nan, dtype=np.float64)
    valid = np.zeros(num_trees, dtype=bool)
    mass = np.full(num_trees, np.nan, dtype=np.float64)

    for i, (halo_1, halo_2) in enumerate(zip(halos_1, halos_2)):
        if halo_1 < 0 or halo_2 < 0:
            continue

        if HaloFlag_1[halo_1] ==0 or HaloFlag_2[halo_2]==0:
            continue

        average_cr = (cr_1[halo_1] + cr_2[halo_2]) / 2
        average_M_hot = (M_hot_1[halo_1] + M_hot_2[halo_2]) / 2
        average_t_dyn = (t_dyn_1[halo_1] + t_dyn_2[halo_2]) / 2

        frac = np.clip(average_cr, 0.0, 1.0)
        mass[i] = 0.5 * (m200c_1[i] + m200c_2[i])
        rates[i] = average_M_hot * frac / average_t_dyn * 365.25 * 24 * 3600 / 1.98847e33 # M_sun/yr
        valid[i] = True

    result_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")
    result_dir.mkdir(parents=True, exist_ok=True)
    output_file = result_dir / f"Lgal_cooling_rate_{snap_1}_and_{snap_2}.hdf5"
    with h5py.File(output_file, "w") as f:
        f.create_dataset("halo_id_snap_1", data=halos_1, compression="gzip", compression_opts=4)
        f.create_dataset("halo_id_snap_2", data=halos_2, compression="gzip", compression_opts=4)
        f.create_dataset("halo_mass", data=mass, compression="gzip", compression_opts=4)
        f.create_dataset("cooling_rate", data=rates, compression="gzip", compression_opts=4)
        f.create_dataset("valid", data=valid.astype(np.uint8), compression="gzip", compression_opts=4)

    return

def TNG_cooling_rate(basePath, snap_1, snap_2):

    data_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")
    tree_file = data_dir / "all_selected_trees.hdf5"
    snaps = (int(snap_1), int(snap_2))
    radius_files = [data_dir / f"real_cooling_radius_{snap}.hdf5" for snap in snaps]

    with h5py.File(tree_file, "r") as f:
        start_ids = np.asarray(f["start_subhalo_id"][:], dtype=np.int64)
        snap_numbers = np.asarray(f["snap_numbers"][:], dtype=np.int64)
        columns = [funcs.get_snap_column(snap, snap_numbers) for snap in snaps]
        node_ids = np.column_stack([np.asarray(f["mpb_subhalo_ids"][:, column], dtype=np.int64) for column in columns])
        fof_ids = np.column_stack([np.asarray(f["mpb_fof_ids"][:, column], dtype=np.int64) for column in columns])

    num_trees = start_ids.size
    radius = np.full((num_trees, 2), np.nan, dtype=np.float64)
    hot_mass_g = np.full_like(radius, np.nan)
    t_dyn_s = np.full_like(radius, np.nan)
    halo_mass_msun = np.full_like(radius, np.nan)
    radius_status = np.zeros((num_trees, 2), dtype=np.uint8)
    valid_snap = np.zeros((num_trees, 2), dtype=bool)

    for snap_index, (snap, radius_file) in enumerate(zip(snaps, radius_files)):
        with h5py.File(radius_file, "r") as f:
            measured_radius = np.asarray(f["cooling_radius_r200c"][:], dtype=np.float64)
            analysis_valid = np.asarray(f["analysis_valid"][:], dtype=bool)
            status = np.asarray(f["status"][:], dtype=np.uint8)

        radius[:, snap_index] = measured_radius
        radius_status[:, snap_index] = status
        header = il.groupcat.loadHeader(basePath, snap)
        halo_mass_msun[:, snap_index] = load_m200c_msun(basePath, snap, fof_ids[:, snap_index], header["HubbleParam"])

        # Keep the same hot-gas mass and dynamical-time definitions as Lgal_cooling_rate.
        _, all_hot_mass_g, _, all_t_dyn_s, all_halo_flag = cr.cooling_radius(basePath, snap)
        all_hot_mass_g = np.asarray(all_hot_mass_g, dtype=np.float64)
        all_t_dyn_s = np.asarray(all_t_dyn_s, dtype=np.float64)
        all_halo_flag = np.asarray(all_halo_flag, dtype=bool)

        halo_ids = fof_ids[:, snap_index]
        valid_fof = (halo_ids >= 0) & (halo_ids < all_hot_mass_g.size)
        hot_mass_g[valid_fof, snap_index] = all_hot_mass_g[halo_ids[valid_fof]]
        t_dyn_s[valid_fof, snap_index] = all_t_dyn_s[halo_ids[valid_fof]]
        halo_valid = np.zeros(num_trees, dtype=bool)
        halo_valid[valid_fof] = all_halo_flag[halo_ids[valid_fof]]
        valid_snap[:, snap_index] = (analysis_valid & (status == 1) & (node_ids[:, snap_index] >= 0) & halo_valid
            & np.isfinite(measured_radius) & (measured_radius > 0.0)
            & np.isfinite(hot_mass_g[:, snap_index]) & (hot_mass_g[:, snap_index] >= 0.0)
            & np.isfinite(t_dyn_s[:, snap_index]) & (t_dyn_s[:, snap_index] > 0.0)
            & np.isfinite(halo_mass_msun[:, snap_index]) & (halo_mass_msun[:, snap_index] > 0.0))

    valid = np.all(valid_snap, axis=1)
    mass = np.full(num_trees, np.nan, dtype=np.float64)
    rates = np.full(num_trees, np.nan, dtype=np.float64)
    average_cr = np.mean(radius[valid], axis=1)
    average_M_hot = np.mean(hot_mass_g[valid], axis=1)
    average_t_dyn = np.mean(t_dyn_s[valid], axis=1)
    mass[valid] = np.mean(halo_mass_msun[valid], axis=1)
    rates[valid] = average_M_hot * np.clip(average_cr, 0.0, 1.0) / average_t_dyn * (365.25 * 24 * 3600) / 1.98847e33

    output_file = data_dir / f"TNG_cooling_rate_{snaps[0]}_and_{snaps[1]}.hdf5"
    with h5py.File(output_file, "w") as f:
        datasets = {"start_subhalo_id": start_ids, "snap_numbers": np.asarray(snaps, dtype=np.int64),
            "halo_id_snap_1": fof_ids[:, 0], "halo_id_snap_2": fof_ids[:, 1],
            "subhalo_id_snap_1": node_ids[:, 0], "subhalo_id_snap_2": node_ids[:, 1],
            "halo_mass": mass, "cooling_rate": rates, "valid": valid.astype(np.uint8),
            "cooling_radius_r200c": radius, "M_hot_g": hot_mass_g, "t_dyn_s": t_dyn_s,
            "cooling_radius_status": radius_status, "valid_snap": valid_snap.astype(np.uint8)}
        for name, values in datasets.items():
            f.create_dataset(name, data=values, compression="gzip", compression_opts=4)

    return 

def tracer_cooling_rate_direct(basePath, target_snap, start_snap):
    '''
    target_snap < start_snap
    '''
    assert isfile('/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/all_selected_trees.hdf5'), 'Mpb halos file does not exist!'
    tree_file = '/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/all_selected_trees.hdf5'
    f = h5py.File(tree_file,'r')

    start_ids = np.asarray(f["start_subhalo_id"][:], dtype=np.int64)
    snap_numbers = np.asarray(f["snap_numbers"][:], dtype=np.int64)
    num_trees = start_ids.size

    col_start = funcs.get_snap_column(start_snap, snap_numbers)
    col_target = funcs.get_snap_column(target_snap, snap_numbers)

    fof_start = np.asarray(f["mpb_fof_ids"][:, col_start], dtype=np.int64)
    fof_target = np.asarray(f["mpb_fof_ids"][:, col_target], dtype=np.int64)

    f.close()

    header_start = il.groupcat.loadHeader(basePath, start_snap)
    header_target = il.groupcat.loadHeader(basePath, target_snap)

    mass_start = load_m200c_msun(basePath, start_snap, fof_start, header_start["HubbleParam"])
    mass_target = load_m200c_msun(basePath, target_snap, fof_target, header_target["HubbleParam"])

    halo_mass = 0.5 * (mass_start + mass_target)

    source_counts = np.zeros((num_trees, 4), dtype=np.int64)

    cats_file = f'/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/distance_cats/distance_cats_start_{start_snap}_target_{target_snap}.hdf5'
    cf = h5py.File(cats_file,'r')
    offsets = np.asarray(cf["tree_tracer_offsets"][:], dtype=np.int64)
    tracer_class = np.asarray(cf["tracers_class"][:], dtype=np.int8)

    cf.close()

    for j in range(num_trees):
        a = offsets[j]
        b = offsets[j + 1]
        classes_j = tracer_class[a:b]

        for k in range(1, 5):
            source_counts[j, k - 1] = np.count_nonzero(classes_j == k)

    total_counts = source_counts.sum(axis=1)

    tracer_mass_msun =  5.73879e-6 * 1e10 / header_target["HubbleParam"]

    source_mass = source_counts * tracer_mass_msun
    total_mass = source_mass.sum(axis=1)

    source_fraction = np.full((num_trees, 4), np.nan, dtype=np.float64)
    has_sources = total_counts > 0
    source_fraction[has_sources] = source_counts[has_sources] / total_counts[has_sources, None]

    a_start = header_start["Time"]
    a_target = header_target["Time"]

    dt_Gyr = iF.get_time_difference_in_Gyr(a_target, a_start)

    tcrd = total_mass / dt_Gyr
    # source_mass columns: Outer cold, Inner hot, Outer hot, outside R200c.
    hot_to_cold_rate = (source_mass[:, 1] + source_mass[:, 2]) / dt_Gyr
    cold_to_cold_rate = source_mass[:, 0] / dt_Gyr

    filename = f'/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/distance_cats/tcrd_{target_snap}_to_{start_snap}.hdf5'
    with h5py.File(filename, 'w') as f:
        f.create_dataset('halo_mass', data=halo_mass)
        f.create_dataset('cooling_rate', data=tcrd).attrs['units'] = 'Msun/Gyr'
        f.create_dataset('hot_to_cold_rate_msun_per_gyr', data=hot_to_cold_rate).attrs['units'] = 'Msun/Gyr'
        f.create_dataset('cold_to_cold_rate_msun_per_gyr', data=cold_to_cold_rate).attrs['units'] = 'Msun/Gyr'
        f.create_dataset('source_counts', data=source_counts)
        f.create_dataset('source_fraction', data=source_fraction)

    return

def tracer_out_rate_direct(basePath, target_snap, start_snap):
    '''
    target_snap > start_snap
    '''
    assert isfile('/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/all_selected_trees.hdf5'), 'Mpb halos file does not exist!'
    tree_file = '/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/all_selected_trees.hdf5'
    f = h5py.File(tree_file,'r')

    start_ids = np.asarray(f["start_subhalo_id"][:], dtype=np.int64)
    snap_numbers = np.asarray(f["snap_numbers"][:], dtype=np.int64)
    num_trees = start_ids.size

    col_start = funcs.get_snap_column(start_snap, snap_numbers)
    col_target = funcs.get_snap_column(target_snap, snap_numbers)

    fof_start = np.asarray(f["mpb_fof_ids"][:, col_start], dtype=np.int64)
    fof_target = np.asarray(f["mpb_fof_ids"][:, col_target], dtype=np.int64)

    f.close()

    header_start = il.groupcat.loadHeader(basePath, start_snap)
    header_target = il.groupcat.loadHeader(basePath, target_snap)

    mass_start = load_m200c_msun(basePath, start_snap, fof_start, header_start["HubbleParam"])
    mass_target = load_m200c_msun(basePath, target_snap, fof_target, header_target["HubbleParam"])

    halo_mass = 0.5 * (mass_start + mass_target)

    source_counts = np.zeros((num_trees, 4), dtype=np.int64)

    cats_file = f'/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/distance_cats/distance_cats_start_{start_snap}_target_{target_snap}.hdf5'
    cf = h5py.File(cats_file,'r')
    offsets = np.asarray(cf["tree_tracer_offsets"][:], dtype=np.int64)
    tracer_class = np.asarray(cf["tracers_class"][:], dtype=np.int8)

    cf.close()

    for j in range(num_trees):
        a = offsets[j]
        b = offsets[j + 1]
        classes_j = tracer_class[a:b]

        for k in range(1, 5):
            source_counts[j, k - 1] = np.count_nonzero(classes_j == k)

    total_counts = source_counts.sum(axis=1)

    tracer_mass_msun =  5.73879e-6 * 1e10 / header_target["HubbleParam"]

    source_mass = source_counts * tracer_mass_msun
    total_mass = source_mass.sum(axis=1)

    source_fraction = np.full((num_trees, 4), np.nan, dtype=np.float64)
    has_sources = total_counts > 0
    source_fraction[has_sources] = source_counts[has_sources] / total_counts[has_sources, None]

    a_start = header_start["Time"]
    a_target = header_target["Time"]

    dt_Gyr = iF.get_time_difference_in_Gyr(a_start, a_target)

    tord = total_mass / dt_Gyr
    # The same target classes now describe the destinations of Inner cold tracers.
    cold_to_hot_rate = (source_mass[:, 1] + source_mass[:, 2]) / dt_Gyr
    cold_to_cold_rate = source_mass[:, 0] / dt_Gyr

    filename = f'/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/distance_cats/tord_{start_snap}_to_{target_snap}.hdf5'
    with h5py.File(filename, 'w') as f:
        f.create_dataset('halo_mass', data=halo_mass)
        f.create_dataset('out_rate', data=tord).attrs['units'] = 'Msun/Gyr'
        f.create_dataset('cold_to_hot_rate_msun_per_gyr', data=cold_to_hot_rate).attrs['units'] = 'Msun/Gyr'
        f.create_dataset('cold_to_cold_rate_msun_per_gyr', data=cold_to_cold_rate).attrs['units'] = 'Msun/Gyr'
        f.create_dataset('source_counts', data=source_counts)
        f.create_dataset('source_fraction', data=source_fraction)

    return

def tracer_classification(basePath, target_snaps_forward, start_snap_backward, start_snap_forward, target_snaps_backward, *, data_dir=None, output_file=None):

    data_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")
    distance_dir = data_dir / "distance_cats"
    start_snap_backward, start_snap_forward = int(start_snap_backward), int(start_snap_forward)

    forward_snaps = np.asarray(target_snaps_forward, dtype=np.int64)
    backward_snaps = np.asarray(target_snaps_backward, dtype=np.int64)
    in_snaps = np.sort(forward_snaps)[::-1]
    out_snaps = np.sort(backward_snaps)

    with h5py.File(data_dir / "all_selected_trees.hdf5", "r") as f:
        start_ids = np.asarray(f["start_subhalo_id"][:], dtype=np.int64)
        snap_numbers = np.asarray(f["snap_numbers"][:], dtype=np.int64)
        col_backward = funcs.get_snap_column(start_snap_backward, snap_numbers)
        col_forward = funcs.get_snap_column(start_snap_forward, snap_numbers)
        metadata = {
            "start_subhalo_id": start_ids,
            "subhalo_id_backward": np.asarray(f["mpb_subhalo_ids"][:, col_backward], dtype=np.int64),
            "subhalo_id_forward": np.asarray(f["mpb_subhalo_ids"][:, col_forward], dtype=np.int64),
            "fof_id_backward": np.asarray(f["mpb_fof_ids"][:, col_backward], dtype=np.int64),
            "fof_id_forward": np.asarray(f["mpb_fof_ids"][:, col_forward], dtype=np.int64),
        }
    num_trees = start_ids.size

    def classify_history(seed_snap, history_snaps):
        reference_offsets = None
        reference_parent_counts = None
        history_class = history_parent_type = history_known = tree_valid_by_snap = None
        tracer_to_tree = None
        for column, snap in enumerate(history_snaps):
            distance_file = distance_dir / f"distance_cats_start_{seed_snap}_target_{snap}.hdf5"
            with h5py.File(distance_file, "r") as f:
                file_start_ids = np.asarray(f["start_subhalo_id"][:], dtype=np.int64)
                offsets = np.asarray(f["tree_tracer_offsets"][:], dtype=np.int64)
                classes = np.asarray(f["tracers_class"][:], dtype=np.int8)
                tree_valid = np.asarray(f["tree_valid"][:], dtype=bool)

            parent_file = data_dir / f"parent_indices_start_{seed_snap}_target_{snap}.hdf5"
            with h5py.File(parent_file, "r") as f:
                group = f[f"snap_0{snap}"]
                parents = np.asarray(group["parent_indices"][:], dtype=np.int64)
                parent_counts = np.asarray(group["numTracersInParents"][:], dtype=np.int64)

            if column == 0:
                reference_offsets = offsets.copy()
                tracer_to_tree = np.repeat(np.arange(num_trees, dtype=np.int64), np.diff(offsets))
                history_class = np.full((classes.size, 5), -1, dtype=np.int8)
                history_parent_type = np.full_like(history_class, -1)
                history_known = np.zeros((classes.size, 5), dtype=bool)
                tree_valid_by_snap = np.zeros((num_trees, 5), dtype=bool)

            parent_found = parents[:, 0] >= 0
            parent_type = parents[:, 1]
            gas_known = parent_found & (parent_type == 0) & np.isin(classes, [0, 1, 2, 3, 4])
            non_gas_known = parent_found & np.isin(parent_type, [1, 2])
            history_class[:, column] = classes
            history_parent_type[:, column] = parent_type
            history_known[:, column] = (gas_known | non_gas_known) & tree_valid[tracer_to_tree]
            tree_valid_by_snap[:, column] = tree_valid

        nearest_known = history_known[:, 0]
        nearest_gas = nearest_known & (history_parent_type[:, 0] == 0)
        first_layer_class = np.full(history_class.shape[0], -1, dtype=np.int8)
        first_layer_class[nearest_known & np.isin(history_parent_type[:, 0], [1, 2])] = -2
        first_layer_class[nearest_gas & (history_class[:, 0] == 0)] = 0
        first_layer_class[nearest_gas & np.isin(history_class[:, 0], [1, 2, 3, 4])] = 1

        is_flow = first_layer_class == 1
        all_known = np.logical_and.accumulate(history_known, axis=1)
        is_inner = history_known & (history_parent_type == 0) & (history_class == 0)
        ever_inner = np.logical_or.accumulate(is_inner, axis=1)
        layer2_member = is_flow[:, None] & all_known & ~ever_inner
        # A confirmed inner state disproves membership even if another required state is unknown.
        layer2_valid = is_flow[:, None] & (all_known | ever_inner)
        return {
            "tree_tracer_offsets": reference_offsets,
            "history_snaps": history_snaps,
            "history_class": history_class,
            "history_parent_type": history_parent_type,
            "history_state_known": history_known,
            "tree_valid_by_snap": tree_valid_by_snap,
            "first_layer_class": first_layer_class,
            "layer2_member": layer2_member,
            "layer2_valid": layer2_valid,
        }

    output_file = distance_dir / f"tracer_history_classification_{start_snap_backward}_to_{start_snap_forward}.hdf5" if output_file is None else Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_name(f".{output_file.name}.tmp")
    try:
        with h5py.File(temporary_file, "w") as f:
            for name, values in metadata.items():
                f.create_dataset(name, data=values)
            f.create_dataset("history_levels", data=np.arange(1, 6, dtype=np.int8))
            for direction, seed_snap, history_snaps, label in (("in", start_snap_forward, in_snaps, "first_in"), ("out", start_snap_backward, out_snaps, "stay_out")):
                group = f.create_group(direction)
                group.attrs["seed_snap"] = seed_snap
                group.attrs["layer2_valid_definition"] = "Applies only to first_layer_class==1; membership is known for the requested prefix."
                group.create_dataset("layer2_labels", data=np.asarray([f"{label}_{level}" for level in range(1, 6)], dtype="S16"))
                history = classify_history(seed_snap, history_snaps)
                for name, values in history.items():
                    group.create_dataset(name, data=values, compression="gzip", compression_opts=4)
                del history
        temporary_file.replace(output_file)
    finally:
        if temporary_file.exists():
            temporary_file.unlink()
    return


def calculate_tracer_cooling_statistics(basePath, start_snap_backward, start_snap_forward, classification_file=None, output_file=None, *, data_dir=None):
    data_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")
    distance_dir = data_dir / "distance_cats"
    start_snap_backward, start_snap_forward = int(start_snap_backward), int(start_snap_forward)
    classification_file = distance_dir / f"tracer_history_classification_{start_snap_backward}_to_{start_snap_forward}.hdf5" if classification_file is None else Path(classification_file)
    output_file = distance_dir / f"tracer_cooling_rate_modified_{start_snap_backward}_to_{start_snap_forward}.hdf5" if output_file is None else Path(output_file)
    with h5py.File(classification_file, "r") as f:
        metadata = {name: np.asarray(f[name][:], dtype=np.int64) for name in ("start_subhalo_id", "subhalo_id_backward", "subhalo_id_forward", "fof_id_backward", "fof_id_forward")}
    
    num_trees = metadata["start_subhalo_id"].size

    header_backward = il.groupcat.loadHeader(basePath, start_snap_backward)
    header_forward = il.groupcat.loadHeader(basePath, start_snap_forward)
    dt_gyr = float(iF.get_time_difference_in_Gyr(header_backward["Time"], header_forward["Time"]))
    hubble_param = float(header_forward["HubbleParam"])
    tracer_mass_msun = 5.73879e-6 * 1.0e10 / hubble_param

    def count_by_tree(mask, offsets):
        prefix = np.empty(mask.size + 1, dtype=np.int64)
        prefix[0] = 0
        np.cumsum(mask, dtype=np.int64, out=prefix[1:])
        return prefix[offsets[1:]] - prefix[offsets[:-1]]

    statistics = {}
    with h5py.File(classification_file, "r") as f:
        for direction in ("in", "out"):
            group = f[direction]
            offsets = np.asarray(group["tree_tracer_offsets"][:], dtype=np.int64)
            first_class = np.asarray(group["first_layer_class"][:], dtype=np.int8)
            members = np.asarray(group["layer2_member"][:], dtype=bool)
            layer_valid = np.asarray(group["layer2_valid"][:], dtype=bool)
            tree_valid = np.asarray(group["tree_valid_by_snap"][:], dtype=bool)

            all_counts = np.column_stack([count_by_tree(first_class == 1, offsets), count_by_tree(first_class == 0, offsets)])
            first_unknown_counts = count_by_tree(first_class == -1, offsets)
            excluded_non_gas_counts = count_by_tree(first_class == -2, offsets)
            first_complete = tree_valid[:, 0] & (first_unknown_counts == 0)
            level_counts = np.column_stack([count_by_tree(members[:, level], offsets) for level in range(5)])
            level_unknown_counts = np.column_stack([count_by_tree((first_class == 1) & ~layer_valid[:, level], offsets) for level in range(5)])
            level_complete = first_complete[:, None] & (level_unknown_counts == 0)

            all_fraction = np.full((num_trees, 2), np.nan, dtype=np.float64)
            all_denominator = all_counts.sum(axis=1)
            np.divide(all_counts, all_denominator[:, None], out=all_fraction, where=first_complete[:, None] & (all_denominator[:, None] > 0))
            level_fraction = np.full((num_trees, 5), np.nan, dtype=np.float64)
            np.divide(level_counts, all_counts[:, :1], out=level_fraction, where=level_complete & (all_counts[:, :1] > 0))
            all_mass = all_counts.astype(np.float64) * tracer_mass_msun
            level_mass = level_counts.astype(np.float64) * tracer_mass_msun
            all_mass[~first_complete] = np.nan
            level_mass[~level_complete] = np.nan
            statistics[direction] = {
                "all_counts": all_counts,
                "counts": level_counts,
                "all_fraction": all_fraction,
                "fraction": level_fraction,
                "all_mass": all_mass,
                "mass": level_mass,
                "all_valid": first_complete,
                "valid": level_complete,
                "total_counts": np.diff(offsets),
                "first_layer_unknown_counts": first_unknown_counts,
                "excluded_non_gas_counts": excluded_non_gas_counts,
                "unknown_counts": level_unknown_counts,
                "history_snaps": np.asarray(group["history_snaps"][:], dtype=np.int64),
            }

    rate_valid = statistics["in"]["valid"][:, 4] & statistics["out"]["valid"][:, 4]
    modified_rate = (statistics["in"]["mass"][:, 4] + statistics["out"]["mass"][:, 4]) / dt_gyr
    modified_rate[~rate_valid] = np.nan
    mass_backward = load_m200c_msun(basePath, start_snap_backward, metadata["fof_id_backward"], float(header_backward["HubbleParam"]))
    mass_forward = load_m200c_msun(basePath, start_snap_forward, metadata["fof_id_forward"], hubble_param)
    halo_mass = 0.5 * (mass_backward + mass_forward)

    def read_ssfr_for_tree_nodes(snap, fof_ids, node_ids):
        with h5py.File(data_dir / f"ssfr_classification_{snap}.hdf5", "r") as f:
            codes = np.asarray(f["class_code"][:], dtype=np.int8)
            centrals = np.asarray(f["central_subhalo_id"][:], dtype=np.int64)
            delta = np.asarray(f["delta_log_ssfr"][:], dtype=np.float64)
            selected = np.asarray(f["selection_flag"][:], dtype=bool)
            halo_ids = np.asarray(f["halo_id"][:], dtype=np.int64) if "halo_id" in f else np.arange(codes.size, dtype=np.int64)

        ssfr_class = np.full(num_trees, -1, dtype=np.int8)
        ssfr_valid = np.zeros(num_trees, dtype=bool)
        order = np.argsort(halo_ids)
        matches = np.searchsorted(halo_ids[order], fof_ids)
        rows = np.flatnonzero((fof_ids >= 0) & (node_ids >= 0) & (matches < halo_ids.size))
        rows = rows[halo_ids[order[matches[rows]]] == fof_ids[rows]]
        catalogue_rows = order[matches[rows]]
        usable = (selected[catalogue_rows] & np.isin(codes[catalogue_rows], [0, 1, 2, 3]) & (centrals[catalogue_rows] == node_ids[rows])
            & (np.isfinite(delta[catalogue_rows]) | (np.isneginf(delta[catalogue_rows]) & (codes[catalogue_rows] == 3))))
        valid_rows = rows[usable]
        ssfr_class[valid_rows] = codes[catalogue_rows[usable]]
        ssfr_valid[valid_rows] = True
        return ssfr_class, ssfr_valid

    ssfr_backward, ssfr_valid_backward = read_ssfr_for_tree_nodes(start_snap_backward, metadata["fof_id_backward"], metadata["subhalo_id_backward"])
    ssfr_forward, ssfr_valid_forward = read_ssfr_for_tree_nodes(start_snap_forward, metadata["fof_id_forward"], metadata["subhalo_id_forward"])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_name(f".{output_file.name}.tmp")
    try:
        with h5py.File(temporary_file, "w") as f:
            for name, values in metadata.items():
                f.create_dataset(name, data=values)
            f.create_dataset("history_levels", data=np.arange(1, 6, dtype=np.int8))
            for direction, label in (("in", "first_in"), ("out", "stay_out")):
                result = statistics[direction]
                f.create_dataset(f"all_{direction}_labels", data=np.asarray([direction, "retained"], dtype="S16"))
                f.create_dataset(f"{direction}_labels", data=np.asarray([f"{label}_{level}" for level in range(1, 6)], dtype="S16"))
                for suffix in ("counts", "fraction", "mass", "valid"):
                    for prefix, key in ((f"all_{direction}", f"all_{suffix}"), (direction, suffix)):
                        dataset = f.create_dataset(f"{prefix}_{suffix}", data=result[key], compression="gzip", compression_opts=4)
                        if suffix == "mass":
                            dataset.attrs["units"] = "Msun"
                for name in ("total_counts", "first_layer_unknown_counts", "excluded_non_gas_counts", "unknown_counts", "history_snaps"):
                    f.create_dataset(f"{direction}_{name}", data=result[name], compression="gzip", compression_opts=4)
            f.create_dataset("halo_mass", data=halo_mass).attrs["units"] = "Msun"
            f.create_dataset("ssfr_class_backward", data=ssfr_backward)
            f.create_dataset("ssfr_class_forward", data=ssfr_forward)
            f.create_dataset("ssfr_valid_backward", data=ssfr_valid_backward)
            f.create_dataset("ssfr_valid_forward", data=ssfr_valid_forward)
            f.create_dataset("rate_valid", data=rate_valid)
            f.create_dataset("modified_rate_msun_per_gyr", data=modified_rate).attrs["units"] = "Msun/Gyr"
        temporary_file.replace(output_file)
    finally:
        if temporary_file.exists():
            temporary_file.unlink()

    return