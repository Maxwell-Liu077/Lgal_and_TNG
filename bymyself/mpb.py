import illustris_python as il
import numpy as np
import sys
from pathlib import Path
import h5py
import time

def load_mpb_branches(basePath, start_snap, target_snap, first, valid, mpb_subhalos):
    branch_start = time.time()
    print("[MPB 2/4] Preparing SubLink tree offsets...", flush=True)
    row_offsets = il.sublink.subLinkOffsets(basePath, "SubLink")
    queries_by_file = {}
    valid_halos = np.flatnonzero(valid)
    if valid_halos.size == 0:
        print("[MPB 2/4] No valid central subhalos; skipping tree reads.", flush=True)
        return

    central_subhalos = first[valid_halos].astype(np.int64)

    offset_path = il.groupcat.offsetPath(basePath, start_snap)
    prefix = f"Subhalo/SubLink/"

    print(f"[MPB 2/4] Sequentially reading offsets for {valid_halos.size:,} valid halos...", flush=True)
    offset_read_start = time.time()
    with h5py.File(offset_path, "r") as f:
        row_dataset = f[prefix + "RowNum"]
        subhalo_id_dataset = f[prefix + "SubhaloID"]

        if central_subhalos.size and (
            central_subhalos.min() < 0
            or central_subhalos.max() >= row_dataset.shape[0]
            or central_subhalos.max() >= subhalo_id_dataset.shape[0]
        ):
            raise IndexError("Central subhalo ID is outside the offsets datasets.")

        row_all = row_dataset[:]
        row_nums = row_all[central_subhalos]
        del row_all

        subhalo_id_all = subhalo_id_dataset[:]
        subhalo_ids = subhalo_id_all[central_subhalos]
        del subhalo_id_all

    print(f"[MPB 2/4] Sequential offset read finished in {time.time() - offset_read_start:.1f}s.", flush=True)

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

    files = sorted(queries_by_file.items())
    print(f"[MPB 2/4] Found {sum(map(len, queries_by_file.values())):,} trees across {len(files):,} tree files. Reading branches...", flush=True)

    report_every = max(1, len(files) // 20)
    for file_index, (file_num, queries) in enumerate(files, start=1):
        if file_index == 1 or file_index % report_every == 0 or file_index == len(files):
            elapsed = time.time() - branch_start
            print(f"[MPB 2/4] Tree files {file_index:,}/{len(files):,} (current file {file_num}, {len(queries):,} trees), elapsed {elapsed:.1f}s", flush=True)
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

    print(f"[MPB 2/4] Tree branches loaded in {time.time() - branch_start:.1f}s.",flush=True,)


def load_mpb(basePath, start_snap, target_snap):
    start = time.time()
    print(f"Running MPB calculation: snap {start_snap} -> {target_snap}", flush=True)
    print("[MPB 1/4] Loading starting-snapshot halo and subhalo catalogs...", flush=True)

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
    n_valid = np.count_nonzero(valid)
    print(f"[MPB 1/4] Loaded {numHalo:,} FoF halos; {n_valid:,} have valid central subhalos.",flush=True)
    print("[MPB 2/4] Loading SubhaloGrNr arrays for earlier snapshots...", flush=True)

    subhalo_to_halo = {}

    earlier_snaps = snaps[1:]
    for snap_index, snap in enumerate(earlier_snaps, start=1):
        print(f"[MPB 2/4] Loading snapshot {int(snap)} ({snap_index}/{earlier_snaps.size})...", flush=True)
        subs = il.groupcat.loadSubhalos(basePath, snap, fields=["SubhaloGrNr"])
        subhalo_to_halo[snap] = np.asarray(subs, dtype=np.int64)

    load_mpb_branches(basePath, start_snap, target_snap, first, valid, mpb_subhalos)

    print("[MPB 3/4] Mapping MPB subhalo IDs to FoF halo IDs...", flush=True)
    for column, snap in enumerate(earlier_snaps, start=1):
        group_ids = subhalo_to_halo[snap]
        sub_ids = mpb_subhalos[:, column]
        valid_ids = (sub_ids >= 0) & (sub_ids < group_ids.size)
        mpb_halos[valid_ids, column] = group_ids[sub_ids[valid_ids]]

    end = time.time()
    print(f"[MPB 3/4] Calculation completed in {end - start:.1f}s.", flush=True)

    result_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")
    result_dir.mkdir(parents=True, exist_ok=True)
    output_file = result_dir / f"mpb_halos.hdf5"
    print(f"[MPB 4/4] Saving results to {output_file}...", flush=True)
    save_start = time.time()
    with h5py.File(output_file, "w") as f:
        f.create_dataset("mpb_halos", data=mpb_halos, compression="gzip", compression_opts=4)
        f.create_dataset("mpb_subhalos", data=mpb_subhalos, compression="gzip", compression_opts=4)
        f.create_dataset("snap_numbers", data=snaps)
    print(f"[MPB 4/4] Saved in {time.time() - save_start:.1f}s.", flush=True)

    return {"mpb_halos": mpb_halos, "mpb_subhalos": mpb_subhalos, "snap_numbers": snaps}

if __name__ == "__main__":
    start_snap = int(sys.argv[1])
    target_snap = int(sys.argv[2])
    basePath = "/public/share/chenhouzun/TNG50-1/output/"
    load_mpb(basePath, start_snap, target_snap)
