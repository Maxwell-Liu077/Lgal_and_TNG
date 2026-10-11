import h5py
import illustris_python as il
import numpy as np
import funcs
import illustrisFuncs as iF
from pathlib import Path

def load_selected_mpb_subhalos_for_snap(tree_file, snap):

    with h5py.File(tree_file, "r") as f:
        snap_numbers = np.asarray(f["snap_numbers"][:], dtype=np.int64)
        columns = np.flatnonzero(snap_numbers == snap)
        start_ids = np.asarray(f["start_subhalo_id"][:], dtype=np.int64)
        node_subhalo_ids = np.asarray(f["mpb_subhalo_ids"][:, columns[0]], dtype=np.int64)
        node_valid = node_subhalo_ids >= 0

    return start_ids, node_subhalo_ids, node_valid

def collect_inner_gas_parents(basePath, snap, node_subhalo_ids, node_valid):
    node_subhalo_ids = np.asarray(node_subhalo_ids, dtype=np.int64)
    n_nodes = node_subhalo_ids.size

    header = il.groupcat.loadHeader(basePath, snap)
    box_size = float(header["BoxSize"])

    subhalos = il.groupcat.loadSubhalos(basePath, snap, fields=["SubhaloPos", "SubhaloGrNr"])
    subhalo_pos_all = np.asarray(subhalos["SubhaloPos"], dtype=np.float32)
    subhalo_grnr_all = np.asarray(subhalos["SubhaloGrNr"], dtype=np.int64)

    subhalo_pos = np.full((n_nodes, 3), np.nan, dtype=np.float32)
    host_group_ids = np.full(n_nodes, -1, dtype=np.int64)

    subhalo_pos[node_valid] = subhalo_pos_all[node_subhalo_ids[node_valid]]
    host_group_ids[node_valid] = subhalo_grnr_all[node_subhalo_ids[node_valid]]

    halos = il.groupcat.loadHalos(basePath, snap, fields=["GroupLenType", "Group_R_Crit200"])
    group_len_type = np.asarray(halos["GroupLenType"], dtype=np.int64)
    group_r200c_all = np.asarray(halos["Group_R_Crit200"], dtype=np.float32)

    host_r200c = np.full(n_nodes, np.nan, dtype=np.float32)
    host_r200c[node_valid] = group_r200c_all[host_group_ids[node_valid]]

    offset_file = il.groupcat.offsetPath(basePath, snap)
    with h5py.File(offset_file, "r") as f:
        gas_start_all = np.asarray(f["Group/SnapByType"][:, 0], dtype=np.int64)

    gas = il.snapshot.loadSubset(basePath, snap, "gas", fields=["Coordinates","StarFormationRate","InternalEnergy","ElectronAbundance","ParticleIDs"],float32=True)

    gas_pos = gas["Coordinates"]
    gas_sfr = gas["StarFormationRate"]
    gas_utherm = gas["InternalEnergy"]
    gas_nelec = gas["ElectronAbundance"]
    gas_particle_ids = gas["ParticleIDs"]
    gas_total = gas_particle_ids.shape[0]

    del gas

    inner = np.zeros(gas_total, dtype=np.uint8)

    selected_indices_by_node = []
    selected_ids_by_node = []
    selected_categories_by_node = []
    subhalo_offsets = np.zeros(n_nodes + 1, dtype=np.int64)

    for i in range(n_nodes):
        group_id = int(host_group_ids[i])
        radius = float(host_r200c[i])

        if not node_valid[i]:
            selected_indices_by_node.append(np.empty(0, dtype=np.int64))
            selected_ids_by_node.append(np.empty(0, dtype=np.uint64))
            selected_categories_by_node.append(np.empty(0, dtype=np.uint8))
            subhalo_offsets[i + 1] = subhalo_offsets[i]
            continue

        start = int(gas_start_all[group_id])
        length = int(group_len_type[group_id, 0])
        end = start + length


        distances = funcs.dist_vector_nb(subhalo_pos[i], gas_pos[start:end], box_size)
        inside_local = np.flatnonzero(distances < 0.1 * radius)

        sfr_inside = gas_sfr[start + inside_local]
        sf_inside = np.isfinite(sfr_inside) & (sfr_inside > 0)

        cold_inside = np.zeros(inside_local.size, dtype=bool)
        non_sf = ~sf_inside

        if np.any(non_sf):
            non_sf_global = start + inside_local[non_sf]
            temp = iF.utherm_to_temp(gas_utherm[non_sf_global],gas_nelec[non_sf_global])
            cold_inside[non_sf] = (np.isfinite(temp) & (temp > 0) & (temp < 10.0**4.5))

        keep = sf_inside | cold_inside
        selected_local = inside_local[keep]
        selected_global = start + selected_local
        categories = np.where(sf_inside[keep], 1, 2).astype(np.uint8)

        selected_indices_by_node.append(selected_global)
        selected_ids_by_node.append(gas_particle_ids[selected_global])
        selected_categories_by_node.append(categories)
        subhalo_offsets[i + 1] = (subhalo_offsets[i] + selected_global.size)

    del gas_pos, gas_sfr, gas_utherm, gas_nelec, gas_particle_ids

    if subhalo_offsets[-1] > 0:
        gas_parent_index = np.concatenate(selected_indices_by_node)
        gas_particle_id = np.concatenate(selected_ids_by_node)
        gas_category = np.concatenate(selected_categories_by_node)
    else:
        gas_parent_index = np.empty(0, dtype=np.int64)
        gas_particle_id = np.empty(0, dtype=np.uint64)
        gas_category = np.empty(0, dtype=np.uint8)

    if gas_parent_index.size > 0:
        inner[gas_parent_index] = 1

    return subhalo_offsets, gas_parent_index, gas_particle_id, gas_category, host_group_ids, inner

def gas_parent_classification(basePath, snap):
    data_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")
    tree_file = data_dir / "all_selected_trees.hdf5"

    sample_start_ids, node_subhalo_ids, node_valid =\
        load_selected_mpb_subhalos_for_snap(tree_file, snap)

    subhalo_offsets, gas_parent_index, gas_particle_id, gas_category, host_group_ids, inner =\
        collect_inner_gas_parents(basePath, snap, node_subhalo_ids, node_valid)

    output_file = data_dir / f"gas_parent_classification_{snap}.hdf5"

    with h5py.File(output_file, "w") as f:
        f.create_dataset("sample_start_subhalo_id", data=sample_start_ids)
        f.create_dataset("subhalo_offsets", data=subhalo_offsets)
        f.create_dataset("gas_parent_index", data=gas_parent_index, compression="gzip", compression_opts=4)
        f.create_dataset("gas_particle_id", data=gas_particle_id, compression="gzip", compression_opts=4)
        f.create_dataset("gas_category", data=gas_category, compression="gzip", compression_opts=4)
        f.create_dataset("inner", data=inner, compression="gzip", compression_opts=4)