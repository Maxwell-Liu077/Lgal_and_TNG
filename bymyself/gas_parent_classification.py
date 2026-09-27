import h5py
import illustris_python as il
import numpy as np
import funcs
import illustrisFuncs as iF
from pathlib import Path

def load_mpb_halos_for_snap(mpb_file, snap):
    with h5py.File(mpb_file, "r") as f:
        snap_numbers = np.asarray(f["snap_numbers"][:], dtype=np.int64)
        columns = np.flatnonzero(snap_numbers == snap)
        halo_ids_by_track = np.asarray(f["mpb_halos"][:, columns[0]], dtype=np.int64)

    mpb_rows = np.flatnonzero(halo_ids_by_track >= 0)
    track_halo_ids = halo_ids_by_track[mpb_rows]
    halo_ids, track_to_halo = np.unique(track_halo_ids, return_inverse=True)

    return halo_ids, mpb_rows, track_to_halo

def collect_inner_gas_parents(basePath, snap, halo_ids):
    header = il.groupcat.loadHeader(basePath, snap)
    box_size = float(header["BoxSize"])

    halos = il.groupcat.loadHalos(basePath, snap, fields=["GroupLenType", "GroupPos", "Group_R_Crit200"])

    gas_count_by_halo = np.asarray(halos["GroupLenType"][halo_ids, 0], dtype=np.int64)
    halo_pos = np.asarray(halos["GroupPos"][halo_ids], dtype=np.float32)
    halo_r200c = np.asarray(halos["Group_R_Crit200"][halo_ids], dtype=np.float32)

    offset_file = il.groupcat.offsetPath(basePath, snap)
    with h5py.File(offset_file, "r") as f:
        gas_start_by_halo = np.asarray(f["Group/SnapByType"][halo_ids, 0], dtype=np.int64)

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
    mpb_file = "/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/mpb_halos.hdf5"

    for snap in snaps:
        print(f"Processing snap {snap}...", flush=True)

        output_file = data_dir / f"gas_parent_classification_{snap}.hdf5"
        
        halo_ids, mpb_rows, track_to_halo = load_mpb_halos_for_snap(mpb_file, snap)
        halo_offsets, gas_parent_index, gas_particle_id, gas_category = collect_inner_gas_parents(basePath, snap, halo_ids)

        with h5py.File(output_file, "w") as f:
            f.create_dataset("halo_id", data=halo_ids)
            f.create_dataset("halo_offsets", data=halo_offsets)
            f.create_dataset("mpb_rows", data=mpb_rows)
            f.create_dataset("track_to_halo", data=track_to_halo)
            f.create_dataset("gas_parent_index", data=gas_parent_index, compression="gzip", compression_opts=4)
            f.create_dataset("gas_particle_id", data=gas_particle_id, compression="gzip", compression_opts=4)
            f.create_dataset("gas_category", data=gas_category, compression="gzip", compression_opts=4)