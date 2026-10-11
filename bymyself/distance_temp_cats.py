import time
import illustris_python as il
import numpy as np
import h5py
import numba as nb
from numba import njit
import tracerFuncs as tF
import illustrisFuncs as iF
import funcs
from os.path import isfile
from types import SimpleNamespace
from mergertree import loadMPBs
from pathlib import Path

@njit(parallel = True)
def distances_temp(parent_indices_data, final_offsets, all_gas_pos, all_gas_sfr, all_gas_temp, sub_pos_at_target_snap, subhaloFlag, r_vir, boxSize, sub_ids):
    """
    分类标准 (d 为 gas parent 到 subhalo 中心的距离):
      0: 冷气体, d <= 0.1 R200c
      1: 冷气体, 0.1 R200c < d <= R200c
      2: 热气体, d <= 0.1 R200c
      3: 热气体, 0.1 R200c < d <= R200c
      4: d > R200c (不再检查 SFR 和温度)
     -1: 非 gas parent、subhalo 未启用，或输入数据无效/无法分类

    边界约定: d == 0.1 R200c 归入内区; d == R200c 仍归入 halo 内。
    """

    gas_class = np.full(parent_indices_data.shape[0], -1, dtype=np.int8)

    for i in nb.prange(sub_ids.shape[0]):

        if not subhaloFlag[i]:
            continue

        sub_id = sub_ids[i]
        indices_of_sub = np.arange(final_offsets[sub_id],final_offsets[sub_id+1])
        parent_indices_of_sub = parent_indices_data[indices_of_sub,:]

        particle_pos = np.full((indices_of_sub.shape[0], 3), np.nan, dtype=all_gas_pos.dtype)

        gas_mask = np.where(parent_indices_of_sub[:,1] == 0)[0]

        gas_parent_indices = parent_indices_of_sub[gas_mask,0]
        particle_pos[gas_mask,:] = all_gas_pos[gas_parent_indices,:]

        subhalo_position = sub_pos_at_target_snap[sub_id,:] 
        rad_dist = funcs.dist_vector_nb(subhalo_position, particle_pos, boxSize)

        radius = r_vir[i]
        if not np.isfinite(radius) or radius <= 0.0:
            continue

        for j in range(gas_mask.size):
            local_row = gas_mask[j]
            tracer_row = indices_of_sub[local_row]
            gas_idx = gas_parent_indices[j]
            distance = rad_dist[local_row]

            if distance > radius:
                gas_class[tracer_row] = 4
                continue

            sfr = all_gas_sfr[gas_idx]
            temp = all_gas_temp[gas_idx]

            if sfr > 0.0:
                is_cold = True
            elif sfr == 0.0:
                if not np.isfinite(temp) or temp < 0.0:
                    continue
                is_cold = (np.log10(temp) <= 4.5)
            else:
                continue

            is_inner = distance <= 0.1 * radius

            if is_cold:
                gas_class[tracer_row] = 0 if is_inner else 1
            else:
                gas_class[tracer_row] = 2 if is_inner else 3

    return gas_class

def distance_temp_cats(basePath, start_snap, target_snap):
    data_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")
    tree_file = data_dir / "all_selected_trees.hdf5"
    position_file = data_dir / "SubhaloPos_new_extrapolated.hdf5"

    start_loading = time.time()

    header = il.groupcat.loadHeader(basePath, target_snap)
    box_size = header["BoxSize"]

    with h5py.File(tree_file, "r") as tf:
        snap_numbers = np.asarray(tf["snap_numbers"][:], dtype=np.int64)
        start_ids = np.asarray(tf["start_subhalo_id"][:], dtype=np.int64)

        start_cols = np.flatnonzero(snap_numbers == start_snap)
        target_cols = np.flatnonzero(snap_numbers == target_snap)

        start_node_ids = np.asarray(tf["mpb_subhalo_ids"][:, int(start_cols[0])], dtype=np.int64)
        target_node_ids = np.asarray(tf["mpb_subhalo_ids"][:, int(target_cols[0])], dtype=np.int64)
        target_fof_ids = np.asarray(tf["mpb_fof_ids"][:, int(target_cols[0])], dtype=np.int64,)

    n_trees = start_ids.size
    tree_rows = np.arange(n_trees, dtype=np.int64)

    flag_file = data_dir / "subhaloFlag.hdf5"
    with h5py.File(flag_file, "r") as ff:
        flags_all = np.asarray(ff["subhaloFlag"][:], dtype=bool)

    tree_selected = flags_all[start_ids]

    position_col = 99 - target_snap

    with h5py.File(position_file, "r") as pf:
        position_history = pf["SubhaloPos"]
        tree_pos = np.asarray(position_history[start_ids, position_col, :], dtype=np.float32)

    parent_file = data_dir / f"parent_indices_start_{start_snap}_target_{target_snap}.hdf5"
    group_name = f"snap_0{target_snap}"

    with h5py.File(parent_file, "r") as pf:
        parent_indices_data = np.asarray(pf[f"{group_name}/parent_indices"][:], dtype=np.int64)
        num_tracers_in_parents = np.asarray(pf[f"{group_name}/numTracersInParents"][:], dtype=np.int64)

    class_file = data_dir / f"gas_parent_classification_{start_snap}.hdf5"

    with h5py.File(class_file, "r") as cf:
        selected_start_ids = np.asarray(cf["sample_start_subhalo_id"][:], dtype=np.int64)
        gas_node_offsets = np.asarray(cf["subhalo_offsets"][:], dtype=np.int64)

    tracer_offsets_per_parent = np.empty(num_tracers_in_parents.size + 1, dtype=np.int64)
    tracer_offsets_per_parent[0] = 0
    tracer_offsets_per_parent[1:] = np.cumsum(num_tracers_in_parents, dtype=np.int64)

    tree_tracer_offsets = tracer_offsets_per_parent[gas_node_offsets]

    target_halos = il.groupcat.loadHalos(basePath, target_snap, fields=["GroupFirstSub", "Group_R_Crit200"])
    group_first_sub = np.asarray(target_halos["GroupFirstSub"], dtype=np.int64)
    group_r200c = np.asarray(target_halos["Group_R_Crit200"], dtype=np.float32)

    tree_valid = tree_selected & (start_node_ids >= 0) & (target_node_ids >= 0) & np.all(np.isfinite(tree_pos), axis=1) & (target_fof_ids >= 0) & (target_fof_ids < group_first_sub.size) & (target_fof_ids < group_r200c.size)
    
    candidate_rows = np.flatnonzero(tree_valid)
    is_target_central = (group_first_sub[target_fof_ids[candidate_rows]] == target_node_ids[candidate_rows])
    tree_valid[candidate_rows[~is_target_central]] = False

    tree_r200c = np.full(n_trees, np.nan, dtype=np.float32)
    tree_r200c[tree_valid] = group_r200c[target_fof_ids[tree_valid]]

    valid_radius = np.isfinite(tree_r200c) & (tree_r200c > 0.0)
    tree_valid &= valid_radius

    print(f"Selected trees: {np.count_nonzero(tree_selected):,}; valid target centrals: {np.count_nonzero(tree_valid):,}", flush=True)

    mid = time.time()
    print(f"Loading catalogs and indices took {mid - start_loading:.1f}s.", flush=True)

    gas = il.snapshot.loadSubset(basePath, target_snap, "gas", fields=["Coordinates","StarFormationRate","InternalEnergy","ElectronAbundance"])

    all_gas_pos = gas["Coordinates"]
    all_gas_sfr = gas["StarFormationRate"]
    all_gas_temp = iF.utherm_to_temp(gas["InternalEnergy"], gas["ElectronAbundance"])

    gas_parent_mask = parent_indices_data[:, 1] == 0
    gas_parent_indices = parent_indices_data[gas_parent_mask, 0]

    start_compute = time.time()

    gas_class = distances_temp(parent_indices_data, tree_tracer_offsets, all_gas_pos, all_gas_sfr, all_gas_temp, tree_pos, tree_valid, tree_r200c, box_size, tree_rows)

    end_compute = time.time()
    print(f"Distance-temperature classification took {end_compute - start_compute:.1f}s.", flush=True)

    output_dir = data_dir / "distance_cats"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"distance_cats_start_{start_snap}_target_{target_snap}.hdf5"

    with h5py.File(output_file, "w") as f:
        f.create_dataset("start_subhalo_id", data=start_ids)
        f.create_dataset("start_node_subhalo_id", data=start_node_ids)
        f.create_dataset("target_node_subhalo_id", data=target_node_ids)
        f.create_dataset("target_fof_id", data=target_fof_ids)
        f.create_dataset("tree_valid", data=tree_valid.astype(np.uint8))
        f.create_dataset("tree_tracer_offsets", data=tree_tracer_offsets)
        f.create_dataset("tracers_class", data=gas_class, compression="gzip", compression_opts=4)

    print(f"Saved classifications to {output_file}", flush=True)

    return gas_class, tree_valid, tree_tracer_offsets