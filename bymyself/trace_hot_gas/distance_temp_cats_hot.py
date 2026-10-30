import time
import illustris_python as il
import numpy as np
import h5py
import numba as nb
from numba import njit
import illustrisFuncs as iF
import funcs
from pathlib import Path

data_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")

@njit(parallel = True)
def distances_temp(parent_indices_data, final_offsets, all_gas_pos, all_gas_sfr, all_gas_temp, sub_pos_at_target_snap, subhaloFlag, r_vir, boxSize, sub_ids):
    """
    分类标准 (d 为 gas parent 到后继 FoF 晕 GroupPos 的距离):
      0: 冷气体, d <= 0.1 R200c
      1: 冷气体, 0.1 R200c < d <= R200c
      2: 热气体, d <= 0.1 R200c
      3: 热气体, 0.1 R200c < d <= R200c
      4: d > R200c (温度和 SFR 仍另外保存)
     -1: 非 gas parent，或空间/温度信息无效而无法完成此分类

    边界约定: d == 0.1 R200c 归入内区; d == R200c 仍归入 halo 内。
    同时返回每个 tracer 的温度、坐标、距离、r/R200c 和 SFR；晕外 gas 也保存这些值。
    SF gas 不使用有效状态方程温度，其温度输出为 NaN。
    """

    gas_class = np.full(parent_indices_data.shape[0], -1, dtype=np.int8)
    tracer_temperature = np.full(parent_indices_data.shape[0], np.nan, dtype=np.float64)
    tracer_position = np.full((parent_indices_data.shape[0], 3), np.nan, dtype=np.float64)
    tracer_distance = np.full(parent_indices_data.shape[0], np.nan, dtype=np.float64)
    tracer_radius_ratio = np.full(parent_indices_data.shape[0], np.nan, dtype=np.float64)
    tracer_sfr = np.full(parent_indices_data.shape[0], np.nan, dtype=np.float64)

    for i in nb.prange(sub_ids.shape[0]):

        sub_id = sub_ids[i]
        indices_of_sub = np.arange(final_offsets[sub_id],final_offsets[sub_id+1])
        parent_indices_of_sub = parent_indices_data[indices_of_sub,:]

        particle_pos = np.full((indices_of_sub.shape[0], 3), np.nan, dtype=all_gas_pos.dtype)

        gas_mask = np.where((parent_indices_of_sub[:, 1] == 0) & (parent_indices_of_sub[:, 0] >= 0) & (parent_indices_of_sub[:, 0] < all_gas_pos.shape[0]))[0]

        gas_parent_indices = parent_indices_of_sub[gas_mask,0]
        particle_pos[gas_mask,:] = all_gas_pos[gas_parent_indices,:]

        subhalo_position = sub_pos_at_target_snap[sub_id,:]
        radius = r_vir[i]
        geometry_valid = subhaloFlag[i] and np.isfinite(radius) and radius > 0.0 and np.all(np.isfinite(subhalo_position))
        rad_dist = np.full(indices_of_sub.size, np.nan, dtype=np.float64)
        if geometry_valid:
            rad_dist = funcs.dist_vector_nb(subhalo_position, particle_pos, boxSize)

        for j in range(gas_mask.size):
            local_row = gas_mask[j]
            tracer_row = indices_of_sub[local_row]
            gas_idx = gas_parent_indices[j]
            distance = rad_dist[local_row]
            sfr = all_gas_sfr[gas_idx]
            temp = all_gas_temp[gas_idx]
            tracer_position[tracer_row, :] = all_gas_pos[gas_idx, :]
            tracer_temperature[tracer_row] = temp
            tracer_sfr[tracer_row] = sfr
            if not geometry_valid or not np.isfinite(distance):
                continue
            tracer_distance[tracer_row] = distance
            tracer_radius_ratio[tracer_row] = distance / radius

            if distance > radius:
                gas_class[tracer_row] = 4
                continue

            if not np.isfinite(sfr):
                continue
            if sfr > 0.0:
                is_cold = True
            elif sfr == 0.0:
                if not np.isfinite(temp) or temp <= 0.0:
                    continue
                is_cold = temp <= 10.0 ** 4.5
            else:
                continue

            is_inner = distance <= 0.1 * radius

            if is_cold:
                gas_class[tracer_row] = 0 if is_inner else 1
            else:
                gas_class[tracer_row] = 2 if is_inner else 3

    return gas_class, tracer_temperature, tracer_position, tracer_distance, tracer_radius_ratio, tracer_sfr

def distance_temp_cats(basePath, start_snap, target_snap):
    if not 0 <= start_snap < target_snap <= 99:
        raise ValueError("Hot-gas forward tracking requires 0 <= start_snap < target_snap <= 99.")
    tree_file = data_dir / "all_selected_trees.hdf5"

    start_loading = time.time()

    header = il.groupcat.loadHeader(basePath, target_snap)
    box_size = header["BoxSize"]

    with h5py.File(tree_file, "r") as tf:
        snap_numbers = np.asarray(tf["snap_numbers"][:], dtype=np.int64)
        start_ids = np.asarray(tf["start_subhalo_id"][:], dtype=np.int64)

        start_col = funcs.get_snap_column(start_snap, snap_numbers)
        target_col = funcs.get_snap_column(target_snap, snap_numbers)

        start_node_ids = np.asarray(tf["mpb_subhalo_ids"][:, start_col], dtype=np.int64)
        target_node_ids = np.asarray(tf["mpb_subhalo_ids"][:, target_col], dtype=np.int64)
        target_fof_ids = np.asarray(tf["mpb_fof_ids"][:, target_col], dtype=np.int64)

    n_trees = start_ids.size
    tree_rows = np.arange(n_trees, dtype=np.int64)

    parent_file = data_dir / f"parent_indices_hot_start_{start_snap}_target_{target_snap}.hdf5"
    group_name = f"snap_{target_snap:03d}"

    with h5py.File(parent_file, "r") as pf:
        parent_indices_data = np.asarray(pf[f"{group_name}/parent_indices"][:], dtype=np.int64)
        num_tracers_in_parents = np.asarray(pf[f"{group_name}/numTracersInParents"][:], dtype=np.int64)
        tracer_ids = np.asarray(pf[f"{group_name}/tracer_id"][:], dtype=np.uint64)
        tracer_found = np.asarray(pf[f"{group_name}/tracer_found"][:], dtype=bool)
        parent_start_ids = np.asarray(pf["sample_start_subhalo_id"][:], dtype=np.int64)
        saved_tracer_offsets = np.asarray(pf["halo_tracer_offsets"][:], dtype=np.int64)
        parent_halo_ids = np.asarray(pf["halo_id"][:], dtype=np.int64)

    class_file = data_dir / f"gas_parent_classification_hot_{start_snap}.hdf5"

    with h5py.File(class_file, "r") as cf:
        selected_start_ids = np.asarray(cf["sample_start_subhalo_id"][:], dtype=np.int64)
        selected_node_ids = np.asarray(cf["start_node_subhalo_id"][:], dtype=np.int64)
        gas_node_offsets = np.asarray(cf["halo_offsets"][:], dtype=np.int64)
        source_halo_ids = np.asarray(cf["halo_id"][:], dtype=np.int64)
        tree_selected = np.asarray(cf["tree_valid"][:], dtype=bool)

    if not (np.array_equal(start_ids, selected_start_ids) and np.array_equal(start_ids, parent_start_ids) and np.array_equal(start_node_ids, selected_node_ids) and np.array_equal(source_halo_ids, parent_halo_ids)):
        raise ValueError("Hot-gas, tracer-parent, and selected-tree files have different sample rows.")
    if gas_node_offsets.shape != (n_trees + 1,) or gas_node_offsets[0] != 0 or gas_node_offsets[-1] != num_tracers_in_parents.size or np.any(np.diff(gas_node_offsets) < 0) or np.any(num_tracers_in_parents < 0):
        raise ValueError("Hot-gas offsets do not match the selected trees and per-parent tracer counts.")

    tracer_offsets_per_parent = np.empty(num_tracers_in_parents.size + 1, dtype=np.int64)
    tracer_offsets_per_parent[0] = 0
    tracer_offsets_per_parent[1:] = np.cumsum(num_tracers_in_parents, dtype=np.int64)

    tree_tracer_offsets = tracer_offsets_per_parent[gas_node_offsets]
    if not np.array_equal(tree_tracer_offsets, saved_tracer_offsets) or tree_tracer_offsets[-1] != parent_indices_data.shape[0] or tracer_ids.size != parent_indices_data.shape[0] or tracer_found.size != tracer_ids.size:
        raise ValueError("Hot-gas tracer offsets and parent rows are inconsistent.")

    target_halos = il.groupcat.loadHalos(basePath, target_snap, fields=["GroupFirstSub", "Group_R_Crit200", "GroupPos"])
    group_first_sub = np.asarray(target_halos["GroupFirstSub"], dtype=np.int64)
    group_r200c = np.asarray(target_halos["Group_R_Crit200"], dtype=np.float32)
    group_pos = np.asarray(target_halos["GroupPos"], dtype=np.float64)

    tree_valid = tree_selected & (target_node_ids >= 0) & (target_node_ids < int(header["Nsubgroups_Total"])) & (target_fof_ids >= 0) & (target_fof_ids < group_r200c.size)

    tree_r200c = np.full(n_trees, np.nan, dtype=np.float32)
    tree_pos = np.full((n_trees, 3), np.nan, dtype=np.float64)
    target_is_central = np.zeros(n_trees, dtype=bool)
    tree_r200c[tree_valid] = group_r200c[target_fof_ids[tree_valid]]
    tree_pos[tree_valid] = group_pos[target_fof_ids[tree_valid]]
    target_is_central[tree_valid] = group_first_sub[target_fof_ids[tree_valid]] == target_node_ids[tree_valid]

    valid_radius = np.isfinite(tree_r200c) & (tree_r200c > 0.0)
    tree_valid &= valid_radius & np.all(np.isfinite(tree_pos), axis=1)

    print(f"Selected start centrals: {np.count_nonzero(tree_selected):,}; valid target hosts: {np.count_nonzero(tree_valid):,}", flush=True)

    mid = time.time()
    print(f"Loading catalogs and indices took {mid - start_loading:.1f}s.", flush=True)

    gas = il.snapshot.loadSubset(basePath, target_snap, "gas", fields=["Coordinates","StarFormationRate","InternalEnergy","ElectronAbundance"])

    all_gas_pos = gas["Coordinates"]
    all_gas_sfr = gas["StarFormationRate"]
    all_gas_temp = np.full(all_gas_sfr.shape, np.nan, dtype=np.float64)
    non_sf = np.isfinite(all_gas_sfr) & (all_gas_sfr == 0.0)
    all_gas_temp[non_sf] = iF.utherm_to_temp(np.asarray(gas["InternalEnergy"][non_sf], dtype=np.float64), np.asarray(gas["ElectronAbundance"][non_sf], dtype=np.float64))

    start_compute = time.time()

    gas_class, tracer_temperature, tracer_position, tracer_distance, tracer_radius_ratio, tracer_sfr = distances_temp(parent_indices_data, tree_tracer_offsets, all_gas_pos, all_gas_sfr, all_gas_temp, tree_pos, tree_valid, tree_r200c, box_size, tree_rows)
    gas_parent_valid = tracer_found & (parent_indices_data[:, 1] == 0) & (parent_indices_data[:, 0] >= 0) & (parent_indices_data[:, 0] < all_gas_pos.shape[0])
    hot_to_cold_non_sf = gas_parent_valid & (tracer_sfr == 0.0) & np.isfinite(tracer_temperature) & (tracer_temperature > 0.0) & (tracer_temperature <= 10.0 ** 4.5)
    hot_to_sf = gas_parent_valid & np.isfinite(tracer_sfr) & (tracer_sfr > 0.0)
    hot_to_cold = hot_to_cold_non_sf | hot_to_sf

    end_compute = time.time()
    print(f"Distance-temperature classification took {end_compute - start_compute:.1f}s.", flush=True)

    output_dir = data_dir / "distance_cats"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"distance_cats_hot_start_{start_snap}_target_{target_snap}.hdf5"

    with h5py.File(output_file, "w") as f:
        f.create_dataset("start_subhalo_id", data=start_ids)
        f.create_dataset("start_node_subhalo_id", data=start_node_ids)
        f.create_dataset("target_node_subhalo_id", data=target_node_ids)
        f.create_dataset("target_fof_id", data=target_fof_ids)
        f.create_dataset("start_fof_id", data=source_halo_ids)
        f.create_dataset("target_is_central", data=target_is_central.astype(np.uint8))
        f.create_dataset("tree_valid", data=tree_valid.astype(np.uint8))
        f.create_dataset("tree_tracer_offsets", data=tree_tracer_offsets)
        f.create_dataset("tracers_class", data=gas_class, compression="gzip", compression_opts=4)
        f.create_dataset("tracer_id", data=tracer_ids)
        f.create_dataset("tracer_found", data=tracer_found.astype(np.uint8))
        f.create_dataset("parent_indices", data=parent_indices_data)
        f.create_dataset("tracer_temperature_K", data=tracer_temperature)
        f.create_dataset("tracer_position_ckpc_h", data=tracer_position)
        f.create_dataset("tracer_distance_ckpc_h", data=tracer_distance)
        f.create_dataset("tracer_distance_r200c", data=tracer_radius_ratio)
        f.create_dataset("tracer_sfr_msun_per_yr", data=tracer_sfr)
        f.create_dataset("hot_to_cold", data=hot_to_cold.astype(np.uint8))
        f.create_dataset("hot_to_cold_non_sf", data=hot_to_cold_non_sf.astype(np.uint8))
        f.create_dataset("hot_to_sf", data=hot_to_sf.astype(np.uint8))

    print(f"Saved classifications to {output_file}", flush=True)

    return gas_class, tree_valid, tree_tracer_offsets
