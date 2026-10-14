import cooling_radius as cr
import h5py
import numpy as np
from os.path import isfile
from pathlib import Path
import illustris_python as il
import illustrisFuncs as iF

def get_snap_column(snap, snap_numbers):
    matches = np.flatnonzero(snap_numbers == snap)
    if matches.size != 1:
        raise ValueError("Snapshot {snap} must occur exactly once in snap_numbers; found {matches.size} matches.")
    return int(matches[0])

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

    col_1 = get_snap_column(snap_1, snap_numbers)
    col_2 = get_snap_column(snap_2, snap_numbers)

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

    col_start = get_snap_column(start_snap, snap_numbers)
    col_target = get_snap_column(target_snap, snap_numbers)

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

    filename = f'/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/distance_cats/tcrd_{target_snap}_to_{start_snap}.hdf5'
    f = h5py.File(filename,'w')

    f.create_dataset('halo_mass',data = halo_mass)
    f.create_dataset('cooling_rate',data = tcrd)
    f.create_dataset('source_counts', data = source_counts)
    f.create_dataset('source_fraction', data = source_fraction)

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

    col_start = get_snap_column(start_snap, snap_numbers)
    col_target = get_snap_column(target_snap, snap_numbers)

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

    filename = f'/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/distance_cats/tord_{start_snap}_to_{target_snap}.hdf5'
    f = h5py.File(filename,'w')

    f.create_dataset('halo_mass',data = halo_mass)
    f.create_dataset('out_rate',data = tord)
    f.create_dataset('source_counts', data = source_counts)
    f.create_dataset('source_fraction', data = source_fraction)

    return

def tracer_cooling_rate_modified(basePath, target_snaps_forward, start_snap_backward, start_snap_forward, target_snaps_backward):
    '''
    1. 已通过连续调用 distance_temp_cats 获得了 start_snap_forward 中 inner gas 的 tracer 在 target_snaps_forward 中、start_snap_backward 中 inner gas 的 tracer 在 target_snaps_backward 中的位置距离分类；
       已通过调用 ssfr_classify_galaxies 获得了 start_snap_backward 和 start_snap_forward 中的 ssfr 分类；
    2. target_snaps_forward 均在 start_snap 之前，target_snaps_backward 均在 start_snap 之后；
    3. tracer_mass_msun 和 dt_Gyr 沿用 tracer_out_rate_direct 中的定义；
    4. 对于 target_snaps_forward，考察 start_snap_forward 的 inner gas tracer 在其中的分类历史：
       0: 从未被分类为 0 → first_in
       1: 曾经被分类为 0，且 target_snaps 中最后一个 snap 的分类不为 0 → recycled_in
       2: 曾经被分类为 0，且 target_snaps 中最后一个 snap 的分类为 0 → former_in
    5. 对于 target_snaps_backward，考察 start_snap_backward 的 inner gas tracer 在其中的分类历史：
       0: 从未被分类为 0 → stay_out
       1: 曾经被分类为 0，且 target_snaps 中第一个 snap 的分类不为 0 → recycled_out
       2: 曾经被分类为 0，且 target_snaps 中第一个 snap 的分类为 0 → stay_in
    6. 分别统计 4、5 中各类 tracer 的数量与其占各自总数的比例，通过 tracer_mass_msun 转换为质量，利用 dt_Gyr 计算出速率；
    7. 定义 tracer_cooling_rate_modified 为 first_in + stay_out
    8. 仿照 tracer_out_rate_direct 给出顺序一致的 halo_mass  
    9. 对 start_snap_backward 和 start_snap_forward 分别读取对应的 ssfr 分类并按照顺序保存

    '''

    data_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")
    tree_file = data_dir / "all_selected_trees.hdf5"
    distance_dir = data_dir / "distance_cats"

    forward_snaps = np.asarray(target_snaps_forward, dtype=np.int64)
    backward_snaps = np.asarray(target_snaps_backward, dtype=np.int64)

    forward_snaps = np.sort(forward_snaps)
    backward_snaps = np.sort(backward_snaps)

    with h5py.File(tree_file, "r") as f:
        start_ids = np.asarray(f["start_subhalo_id"][:], dtype=np.int64)
        snap_numbers = np.asarray(f["snap_numbers"][:], dtype=np.int64)

        col_backward = get_snap_column(start_snap_backward, snap_numbers)
        col_forward = get_snap_column(start_snap_forward, snap_numbers)

        node_backward = np.asarray(f["mpb_subhalo_ids"][:, col_backward], dtype=np.int64)
        node_forward = np.asarray(f["mpb_subhalo_ids"][:, col_forward], dtype=np.int64)
        fof_backward = np.asarray(f["mpb_fof_ids"][:, col_backward], dtype=np.int64)
        fof_forward = np.asarray(f["mpb_fof_ids"][:, col_forward], dtype=np.int64)

    num_trees = start_ids.size

    def known_states_for_snap(seed_snap, snap, classes):
        parent_file = data_dir / f"parent_indices_start_{seed_snap}_target_{snap}.hdf5"
        group_name = f"snap_0{snap}"

        with h5py.File(parent_file, "r") as f:
            parents = np.asarray(f[f"{group_name}/parent_indices"][:], dtype=np.int64,)

        parent_index = parents[:, 0]
        parent_type = parents[:, 1]
        parent_found = parent_index >= 0

        gas_known = (parent_found & (parent_type == 0) & (classes >= 0) & (classes <= 4))
        non_gas_known = (parent_found & ((parent_type == 1) | (parent_type == 2)))

        return gas_known | non_gas_known

    def read_and_count_history(seed_snap, target_snaps, edge):
        """
        edge='last': 用历史最后一个快照的类别作为边界类别。
        edge='first': 用历史第一个快照的类别作为边界类别。
        """
        first_snap = target_snaps[0]
        first_file = distance_dir / f"distance_cats_start_{seed_snap}_target_{first_snap}.hdf5"

        with h5py.File(first_file, "r") as f:
            reference_offsets = np.asarray(f["tree_tracer_offsets"][:], dtype=np.int64)
            first_classes = np.asarray(f["tracers_class"][:], dtype=np.int8)
            first_tree_valid = np.asarray(f["tree_valid"][:], dtype=bool)

        ever_inner = first_classes == 0
        all_states_known = known_states_for_snap(seed_snap, first_snap, first_classes)
        tree_valid_all = first_tree_valid.copy()

        boundary_class = first_classes.copy()

        for snap in target_snaps[1:]:
            input_file = (distance_dir / f"distance_cats_start_{seed_snap}_target_{snap}.hdf5")

            with h5py.File(input_file, "r") as f:
                offsets = np.asarray(f["tree_tracer_offsets"][:], dtype=np.int64)
                classes = np.asarray(f["tracers_class"][:], dtype=np.int8)
                tree_valid = np.asarray(f["tree_valid"][:], dtype=bool)

            all_states_known &= known_states_for_snap(seed_snap, snap, classes)
            ever_inner |= classes == 0
            tree_valid_all &= tree_valid

            if edge == "last":
                boundary_class = classes.copy()

        offsets = reference_offsets
        tracer_to_tree = np.repeat(np.arange(num_trees, dtype=np.int64), np.diff(offsets))

        # 某个快照中整棵树无效时，该树的 tracer 历史也标记为未知。
        if tracer_to_tree.size:
            all_states_known &= tree_valid_all[tracer_to_tree]

        if edge == "last":
            is_boundary_inner = boundary_class == 0
        else:
            is_boundary_inner = boundary_class == 0

        first_category = all_states_known & ~ever_inner
        second_category = all_states_known & ever_inner & ~is_boundary_inner
        third_category = all_states_known & ever_inner & is_boundary_inner

        counts = np.zeros((num_trees, 3), dtype=np.int64)
        total_counts = np.diff(offsets).astype(np.int64)

        category_masks = (first_category, second_category, third_category)

        for tree in range(num_trees):
            a = offsets[tree]
            b = offsets[tree + 1]

            for category, mask in enumerate(category_masks):
                counts[tree, category] = np.count_nonzero(mask[a:b])

        unknown_counts = total_counts - counts.sum(axis=1)

        fractions = np.full((num_trees, 3), np.nan, dtype=np.float64)
        np.divide(counts, total_counts[:, None], out=fractions, where=total_counts[:, None] > 0)
        
        history_complete = unknown_counts == 0

        return {"offsets": offsets, "counts": counts, "fractions": fractions, "total_counts": total_counts, "unknown_counts": unknown_counts, "history_complete": history_complete}

    in_history = read_and_count_history(start_snap_forward, forward_snaps, edge="last")
    out_history = read_and_count_history(start_snap_backward, backward_snaps, edge="first")

    header_backward = il.groupcat.loadHeader(basePath, start_snap_backward)
    header_forward = il.groupcat.loadHeader(basePath, start_snap_forward)

    dt_Gyr = iF.get_time_difference_in_Gyr(header_backward["Time"], header_forward["Time"])

    tracer_mass_msun = 5.73879e-6 * 1.0e10 / header_forward["HubbleParam"]

    in_mass = in_history["counts"] * tracer_mass_msun
    out_mass = out_history["counts"] * tracer_mass_msun

    in_rate = in_mass / dt_Gyr
    out_rate = out_mass / dt_Gyr

    in_rate[~in_history["history_complete"], :] = np.nan
    out_rate[~out_history["history_complete"], :] = np.nan

    rate_valid = in_history["history_complete"] & out_history["history_complete"]

    # modified rate = first_in + stay_out
    modified_rate = in_rate[:, 0] + out_rate[:, 0]
    modified_rate[~rate_valid] = np.nan

    mass_backward = load_m200c_msun(basePath, start_snap_backward, fof_backward, header_backward["HubbleParam"])
    mass_forward = load_m200c_msun(basePath, start_snap_forward, fof_forward, header_forward["HubbleParam"],)

    halo_mass = 0.5 * (mass_backward + mass_forward)

    def read_ssfr_for_tree_nodes(snap, fof_ids, node_ids):
        ssfr_file = data_dir / f"ssfr_classification_{snap}.hdf5"

        with h5py.File(ssfr_file, "r") as f:
            class_code_all = np.asarray(f["class_code"][:], dtype=np.int8)
            central_ids_all = np.asarray(f["central_subhalo_id"][:], dtype=np.int64)
            delta_all = np.asarray(f["delta_log_ssfr"][:], dtype=np.float64)
            selection_flag_all = np.asarray(f["selection_flag"][:], dtype=bool)

        ssfr_class = np.full(num_trees, -1, dtype=np.int8)
        ssfr_delta = np.full(num_trees, np.nan, dtype=np.float64)
        ssfr_valid = np.zeros(num_trees, dtype=bool)

        fof_in_bounds = ((fof_ids >= 0) & (fof_ids < class_code_all.size))
        rows = np.flatnonzero(fof_in_bounds)
        fof = fof_ids[rows]
        codes = class_code_all[fof]
        delta = delta_all[fof]

        valid_code = np.isin(codes, np.array([0, 1, 2, 3], dtype=np.int8))
        central_matches = central_ids_all[fof] == node_ids[rows]

        valid_delta = np.isfinite(delta) | (np.isneginf(delta) & (codes == 3))

        valid_rows = rows[selection_flag_all[fof] & valid_code & central_matches & valid_delta]

        ssfr_class[valid_rows] = class_code_all[fof_ids[valid_rows]]
        ssfr_delta[valid_rows] = delta_all[fof_ids[valid_rows]]
        ssfr_valid[valid_rows] = True

        return ssfr_class

    ssfr_backward = read_ssfr_for_tree_nodes(start_snap_backward, fof_backward, node_backward)
    
    ssfr_forward = read_ssfr_for_tree_nodes(start_snap_forward, fof_forward, node_forward)

    output_file = distance_dir / f"tracer_cooling_rate_modified_{start_snap_backward}_to_{start_snap_forward}.hdf5"

    with h5py.File(output_file, "w") as f:
        f.create_dataset("halo_mass", data=halo_mass)
        f.create_dataset("ssfr_class_backward", data=ssfr_backward)
        f.create_dataset("ssfr_class_forward", data=ssfr_forward)
        f.create_dataset("in_fraction", data=in_history["fractions"])
        f.create_dataset("out_fraction", data=out_history["fractions"])
        f.create_dataset("modified_rate_msun_per_gyr", data=modified_rate)

    return 