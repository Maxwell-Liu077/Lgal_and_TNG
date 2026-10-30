import h5py
import illustris_python as il
import numpy as np
import funcs
import illustrisFuncs as iF
from pathlib import Path

data_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")

def load_valid_halo_ids(basePath, snap):
    """Return one FoF ID per selected-tree row; invalid start nodes have ID -1."""
    with h5py.File(data_dir / "all_selected_trees.hdf5", "r") as f:
        snap_column = funcs.get_snap_column(snap, np.asarray(f["snap_numbers"][:], dtype=np.int64))
        node_ids = np.asarray(f["mpb_subhalo_ids"][:, snap_column], dtype=np.int64)
        halo_ids = np.asarray(f["mpb_fof_ids"][:, snap_column], dtype=np.int64)

    header = il.groupcat.loadHeader(basePath, snap)
    halos = il.groupcat.loadHalos(basePath, snap, fields=["GroupFirstSub", "Group_R_Crit200", "GroupPos"])
    first_sub = np.asarray(halos["GroupFirstSub"], dtype=np.int64)
    r200c = np.asarray(halos["Group_R_Crit200"], dtype=np.float64)
    group_pos = np.asarray(halos["GroupPos"], dtype=np.float64)
    valid = (halo_ids >= 0) & (halo_ids < first_sub.size) & (node_ids >= 0) & (node_ids < int(header["Nsubgroups_Total"]))
    rows = np.flatnonzero(valid)
    hosts = halo_ids[rows]
    valid[rows] &= (first_sub[hosts] == node_ids[rows]) & np.isfinite(r200c[hosts]) & (r200c[hosts] > 0.0) & np.all(np.isfinite(group_pos[hosts]), axis=1)
    halo_ids[~valid] = -1
    if np.unique(halo_ids[valid]).size != np.count_nonzero(valid):
        raise ValueError("Selected trees contain duplicate start FoF halos; each hot-gas parent must be selected once.")
    return halo_ids

def collect_hot_gas_parents(basePath, snap, halo_ids):
    """Keep the existing function name, selecting non-SF hot gas within R200c."""
    header = il.groupcat.loadHeader(basePath, snap)
    box_size = float(header["BoxSize"])

    halos = il.groupcat.loadHalos(basePath, snap, fields=["GroupLenType", "GroupPos", "Group_R_Crit200"])

    valid_halo = halo_ids >= 0
    gas_count_by_halo = np.zeros(halo_ids.size, dtype=np.int64)
    halo_pos = np.full((halo_ids.size, 3), np.nan, dtype=np.float64)
    halo_r200c = np.full(halo_ids.size, np.nan, dtype=np.float64)
    gas_count_by_halo[valid_halo] = halos["GroupLenType"][halo_ids[valid_halo], 0]
    halo_pos[valid_halo] = halos["GroupPos"][halo_ids[valid_halo]]
    halo_r200c[valid_halo] = halos["Group_R_Crit200"][halo_ids[valid_halo]]

    offset_file = il.groupcat.offsetPath(basePath, snap)
    with h5py.File(offset_file, "r") as f:
        gas_start_all = np.asarray(f["Group/SnapByType"][:, 0], dtype=np.int64)

    gas_start_by_halo = np.zeros(halo_ids.size, dtype=np.int64)
    gas_start_by_halo[valid_halo] = gas_start_all[halo_ids[valid_halo]]
    del gas_start_all

    gas = il.snapshot.loadSubset(basePath, snap, "gas", fields=["Coordinates", "StarFormationRate", "InternalEnergy", "ElectronAbundance", "ParticleIDs"], float32=True)

    gas_pos = gas["Coordinates"]
    gas_sfr = gas["StarFormationRate"]
    gas_utherm = gas["InternalEnergy"]
    gas_nelec = gas["ElectronAbundance"]
    gas_particle_ids = gas["ParticleIDs"]

    selected_indices_by_halo = []
    selected_ids_by_halo = []
    selected_categories_by_halo = []
    halo_offsets = np.zeros(halo_ids.size + 1, dtype=np.int64)

    for i in range(halo_ids.size):
        if not valid_halo[i]:
            selected_indices_by_halo.append(np.empty(0, dtype=np.int64))
            selected_ids_by_halo.append(np.empty(0, dtype=np.uint64))
            selected_categories_by_halo.append(np.empty(0, dtype=np.uint8))
            halo_offsets[i + 1] = halo_offsets[i]
            continue
        start = int(gas_start_by_halo[i])
        length = int(gas_count_by_halo[i])
        end = start + length

        halo_gas_pos = gas_pos[start:end]
        distances = funcs.dist_vector_nb(halo_pos[i], halo_gas_pos, box_size)
        inside = np.isfinite(distances) & (distances <= halo_r200c[i])

        sfr = gas_sfr[start:end]
        hot = np.zeros(length, dtype=bool)
        non_sf = np.isfinite(sfr) & (sfr == 0.0)

        if np.any(non_sf):
            temp = iF.utherm_to_temp(np.asarray(gas_utherm[start:end][non_sf], dtype=np.float64), np.asarray(gas_nelec[start:end][non_sf], dtype=np.float64))
            hot[non_sf] = np.isfinite(temp) & (temp > 10.0 ** 4.5)

        selected_local = np.flatnonzero(inside & hot)
        selected_global = start + selected_local

        categories = np.full(selected_local.size, 3, dtype=np.uint8)  # 3 = non-SF hot gas

        selected_indices_by_halo.append(selected_global)
        selected_categories_by_halo.append(categories)
        selected_ids_by_halo.append(gas_particle_ids[selected_global])
        halo_offsets[i + 1] = halo_offsets[i] + selected_global.size

    del gas, gas_pos, gas_sfr, gas_utherm, gas_nelec, gas_particle_ids

    gas_parent_index = (np.concatenate(selected_indices_by_halo) if halo_offsets[-1] > 0 else np.empty(0, dtype=np.int64))
    gas_particle_id = (np.concatenate(selected_ids_by_halo) if halo_offsets[-1] > 0 else np.empty(0, dtype=np.uint64))
    gas_category = (np.concatenate(selected_categories_by_halo) if halo_offsets[-1] > 0 else np.empty(0, dtype=np.uint8))

    return halo_offsets, gas_parent_index, gas_particle_id, gas_category

def gas_parent_classification(basePath, snaps):
    data_dir.mkdir(parents=True, exist_ok=True)

    for snap in np.atleast_1d(snaps):
        snap = int(snap)
        print(f"Processing snap {snap}...", flush=True)

        output_file = data_dir / f"gas_parent_classification_hot_{snap}.hdf5"

        halo_ids = load_valid_halo_ids(basePath, snap)
        halo_offsets, gas_parent_index, gas_particle_id, gas_category = collect_hot_gas_parents(basePath, snap, halo_ids)
        with h5py.File(data_dir / "all_selected_trees.hdf5", "r") as f:
            start_ids = np.asarray(f["start_subhalo_id"][:], dtype=np.int64)
            snap_column = funcs.get_snap_column(snap, np.asarray(f["snap_numbers"][:], dtype=np.int64))
            node_ids = np.asarray(f["mpb_subhalo_ids"][:, snap_column], dtype=np.int64)

        with h5py.File(output_file, "w") as f:
            f.create_dataset("sample_start_subhalo_id", data=start_ids)
            f.create_dataset("start_node_subhalo_id", data=node_ids)
            f.create_dataset("tree_valid", data=(halo_ids >= 0).astype(np.uint8))
            f.create_dataset("halo_id", data=halo_ids)
            f.create_dataset("halo_offsets", data=halo_offsets)
            f.create_dataset("gas_parent_index", data=gas_parent_index, compression="gzip", compression_opts=4)
            f.create_dataset("gas_particle_id", data=gas_particle_id, compression="gzip", compression_opts=4)
            f.create_dataset("gas_category", data=gas_category, compression="gzip", compression_opts=4)
