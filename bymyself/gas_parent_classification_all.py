import h5py
import illustris_python as il
import numpy as np
import funcs
import illustrisFuncs as iF
from pathlib import Path

def load_valid_halo_ids(basePath, snap):
    header = il.groupcat.loadHeader(basePath, snap)
    h = float(header["HubbleParam"])

    halos = il.groupcat.loadHalos(basePath, snap, fields=["GroupFirstSub", "Group_R_Crit200", "GroupLenType"])
    first_sub = np.asarray(halos["GroupFirstSub"], dtype=np.int64)
    r200c = np.asarray(halos["Group_R_Crit200"], dtype=np.float64)

    subhalos = il.groupcat.loadSubhalos(basePath, snap, fields=["SubhaloMassType"])
    mass_type = np.asarray(subhalos, dtype=np.float64)

    has_central = (first_sub >= 0) 
    central_mstar = np.full(first_sub.size, np.nan, dtype=np.float64)

    central_ids = first_sub[has_central]
    central_mstar[has_central] = (mass_type[central_ids, 4] * 1.0e10 / h)
    
    valid = has_central & (central_mstar > 1.0e7)

    return np.flatnonzero(valid)

def collect_inner_gas_parents(basePath, snap, halo_ids):
    header = il.groupcat.loadHeader(basePath, snap)
    box_size = float(header["BoxSize"])

    halos = il.groupcat.loadHalos(basePath, snap, fields=["GroupLenType", "GroupPos", "Group_R_Crit200"])

    gas_count_by_halo = np.asarray(halos["GroupLenType"][halo_ids, 0], dtype=np.int64)
    halo_pos = np.asarray(halos["GroupPos"][halo_ids], dtype=np.float32)
    halo_r200c = np.asarray(halos["Group_R_Crit200"][halo_ids], dtype=np.float32)

    offset_file = il.groupcat.offsetPath(basePath, snap)
    with h5py.File(offset_file, "r") as f:
        gas_start_all = np.asarray(f["Group/SnapByType"][:, 0], dtype=np.int64)

    gas_start_by_halo = gas_start_all[halo_ids]
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
        start = int(gas_start_by_halo[i])
        length = int(gas_count_by_halo[i])
        end = start + length

        halo_gas_pos = gas_pos[start:end]
        distances = funcs.dist_vector_nb(halo_pos[i], halo_gas_pos, box_size)
        inside = distances < 0.1 * halo_r200c[i]

        sfr = gas_sfr[start:end]
        sf = np.isfinite(sfr) & (sfr > 0)

        cold = np.zeros(length, dtype=bool)
        non_sf = ~sf

        if np.any(non_sf):
            temp = iF.utherm_to_temp(gas_utherm[start:end][non_sf], gas_nelec[start:end][non_sf])
            log10_temp = np.log10(temp)
            cold[non_sf] = np.isfinite(temp) & (log10_temp < 4.5)

        selected_local = np.flatnonzero(inside & (sf | cold))
        selected_global = start + selected_local

        # 1 = SFR > 0, 2 = cold
        categories = np.where(sf[selected_local], 1, 2).astype(np.uint8)

        selected_indices_by_halo.append(selected_global)
        selected_categories_by_halo.append(categories)
        selected_ids_by_halo.append(gas_particle_ids[selected_global])
        halo_offsets[i + 1] = halo_offsets[i] + selected_global.size

    del gas, gas_pos, gas_sfr, gas_utherm, gas_nelec, gas_particle_ids

    gas_parent_index = (np.concatenate(selected_indices_by_halo) if halo_offsets[-1] > 0 else np.empty(0, dtype=np.int64))
    gas_particle_id = np.concatenate(selected_ids_by_halo)
    gas_category = (np.concatenate(selected_categories_by_halo) if halo_offsets[-1] > 0 else np.empty(0, dtype=np.uint8))    

    return halo_offsets, gas_parent_index, gas_particle_id, gas_category

def gas_parent_classification(basePath, snaps):
    data_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")

    for snap in snaps:
        print(f"Processing snap {snap}...", flush=True)

        output_file = data_dir / f"gas_parent_classification_{snap}_all.hdf5"
        
        halo_ids = load_valid_halo_ids(basePath, snap)
        halo_offsets, gas_parent_index, gas_particle_id, gas_category = collect_inner_gas_parents(basePath, snap, halo_ids)

        with h5py.File(output_file, "w") as f:
            f.create_dataset("halo_id", data=halo_ids)
            f.create_dataset("halo_offsets", data=halo_offsets)
            f.create_dataset("gas_parent_index", data=gas_parent_index, compression="gzip", compression_opts=4)
            f.create_dataset("gas_particle_id", data=gas_particle_id, compression="gzip", compression_opts=4)
            f.create_dataset("gas_category", data=gas_category, compression="gzip", compression_opts=4)