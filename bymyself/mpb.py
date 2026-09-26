import illustris_python as il
import numpy as np
import sys
from pathlib import Path
import h5py
import time

def load_mpb_branches(basePath, start_snap, target_snap, first, valid, mpb_subhalos):
    row_offsets = il.sublink.subLinkOffsets(basePath, "SubLink")
    queries_by_file = {}
    valid_halos = np.flatnonzero(valid)
    central_subhalos = first[valid_halos].astype(np.int64)

    unique_subhalos, inverse = np.unique(central_subhalos, return_inverse=True)
    offset_path = il.groupcat.offsetPath(basePath, start_snap)
    prefix = f"Subhalo/SubLink/"

    with h5py.File(offset_path, "r") as f:
        row_unique = f[prefix + "RowNum"][unique_subhalos]
        subhalo_id_unique = f[prefix + "SubhaloID"][unique_subhalos]

    row_nums = row_unique[inverse]
    subhalo_ids = subhalo_id_unique[inverse]

    for halo, row_num, subhalo_id in zip(valid_halos, row_nums, subhalo_ids):
        row_num = int(row_num)
        subhalo_id = int(subhalo_id)
        if row_num < 0 or subhalo_id < 0:
            continue

        file_num = int(np.searchsorted(row_offsets, row_num, side="right") - 1)
        if file_num < 0 or file_num >= row_offsets.size:
            continue

        file_offset = int(row_num - row_offsets[file_num])
        queries_by_file.setdefault(file_num, []).append((int(halo), file_offset, subhalo_id))

    for file_num, queries in sorted(queries_by_file.items()):
        tree_file = il.sublink.treePath(basePath, "SubLink", file_num)
        with h5py.File(tree_file, "r") as f:
            snap_data = f["SnapNum"]
            subfind_data = f["SubfindID"]
            main_leaf_data = f["MainLeafProgenitorID"]

            for halo, file_offset, subhalo_id in queries:
                main_leaf_id = int(main_leaf_data[file_offset])
                branch_length = main_leaf_id - subhalo_id + 1

                if branch_length <= 0 or file_offset + branch_length > subfind_data.shape[0]:
                    continue

                branch_end = file_offset + branch_length
                branch_snaps = snap_data[file_offset:branch_end]
                branch_subs = subfind_data[file_offset:branch_end]

                for branch_snap, branch_sub in zip(branch_snaps, branch_subs):
                    branch_snap = int(branch_snap)
                    if target_snap <= branch_snap <= start_snap:
                        column = start_snap - branch_snap
                        mpb_subhalos[halo, column] = int(branch_sub)


def load_mpb(basePath, start_snap, target_snap):
    start = time.time()
    print(f"Running MPB calculation: snap {start_snap} -> {target_snap}")

    halos = il.groupcat.loadHalos(basePath, start_snap, fields=["GroupFirstSub"])
    subhalos = il.groupcat.loadSubhalos(basePath, start_snap, fields=["SubhaloFlag"])

    first = np.asarray(halos, dtype=np.int64)
    flags = np.asarray(subhalos, dtype=bool)
    snaps = np.arange(start_snap, target_snap - 1, -1, dtype = np.int64)

    numHalo = first.shape[0]
    numSnap = snaps.shape[0]
    mpb_halos = np.full((numHalo, numSnap), -1, dtype=np.int64)
    mpb_subhalos = np.full((numHalo, numSnap), -1, dtype=np.int64)

    mpb_halos[:, 0] = np.arange(numHalo, dtype=np.int64)
    mpb_subhalos[:, 0] = first

    valid = np.zeros(numHalo, dtype = bool)
    valid_halo = ((first >= 0) & (first < flags.size))
    valid[valid_halo] = flags[first[valid_halo]]
    subhalo_to_halo = {}

    for snap in snaps[1:]:
        subs = il.groupcat.loadSubhalos(basePath, snap, fields=["SubhaloGrNr"])
        subhalo_to_halo[snap] = np.asarray(subs, dtype=np.int64)

    load_mpb_branches(basePath, start_snap, target_snap, first, valid, mpb_subhalos)

    for column, snap in enumerate(snaps[1:], start=1):
        group_ids = subhalo_to_halo[snap]
        sub_ids = mpb_subhalos[:, column]
        valid_ids = (sub_ids >= 0) & (sub_ids < group_ids.size)
        mpb_halos[valid_ids, column] = group_ids[sub_ids[valid_ids]]

    end = time.time()
    print('Loading took ',np.round(end - start,3),' seconds.')

    result_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")
    result_dir.mkdir(parents=True, exist_ok=True)
    output_file = result_dir / f"mpb_halos.hdf5"
    with h5py.File(output_file, "w") as f:
        f.create_dataset("mpb_halos", data=mpb_halos, compression="gzip", compression_opts=4)
        f.create_dataset("mpb_subhalos", data=mpb_subhalos, compression="gzip", compression_opts=4)

    return {"mpb_halos": mpb_halos, "mpb_subhalos": mpb_subhalos, "snap_numbers": snaps}

if __name__ == "__main__":
    start_snap = int(sys.argv[1])
    target_snap = int(sys.argv[2])
    basePath = "/public/share/chenhouzun/TNG50-1/output/"
    load_mpb(basePath, start_snap, target_snap)
