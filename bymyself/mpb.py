import illustris_python as il
import numpy as np
import sys
from pathlib import Path
import h5py
import time

def load_mpb(basePath, start_snap, target_snap):
    start = time.time()

    halos = il.groupcat.loadHalos(basePath, start_snap, fields=["GroupFirstSub"])
    subhalos = il.groupcat.loadSubhalos(basePath, start_snap, fields=["SubhaloFlag"])

    first = np.asarray(halos["GroupFirstSub"], dtype=np.int64)
    flags = np.asarray(subhalos["SubhaloFlag"], dtype=bool)
    snaps = np.arange(start_snap, target_snap - 1, -1, dtype = np.int64)

    numHalo = first.shape[0]
    numSnap = snaps.shape[0]
    mpb_halos = np.full((numHalo, numSnap), -1, dtype=np.int64)

    mpb_halos[:, 0] = np.arange(numHalo, dtype=np.int64)
    valid = np.zeros(numHalo, dtype=bool)
    valid_halo = np.flatnonzero(
        (first >= 0) & (first < flags.size)
    )
    valid[valid_halo] = flags[first[valid_halo]]

    subhalo_to_halo = {}

    for snap in snaps:
        subs = il.groupcat.loadSubhalos(basePath, snap, fields=["SubhaloGrNr"])
        subhalo_to_halo[snap] = np.asarray(subs["SubhaloGrNr"], dtype=np.int64)

    for halo in range(numHalo):
        if not valid[halo]:
            continue

        central_sub = int(first[halo])
        tree = il.sublink.loadTree(
            basePath,
            start_snap,
            central_sub,
            onlyMPB=True,
            fields=["SnapNum", "SubfindID"],
        )

        if tree is None:
            continue

        branch_snaps = np.asarray(tree["SnapNum"], dtype=np.int64)
        branch_subs = np.asarray(tree["SubfindID"], dtype=np.int64)
        for branch_snap, branch_sub in zip(branch_snaps, branch_subs):
            if not (target_snap <= branch_snap <= start_snap):
                continue

            column = start_snap - branch_snap
            group_ids = subhalo_to_halo[int(branch_snap)]

            if 0 <= branch_sub < group_ids.size:
                mpb_halos[halo, column] = group_ids[branch_sub]

    end = time.time()
    print('Loading took ',np.round(end - start,3),' seconds.')

    result_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")
    result_dir.mkdir(parents=True, exist_ok=True)
    output_file = result_dir / f"mpb_halos.hdf5"
    with h5py.File(output_file, "w") as f:
        f.create_dataset("mpb_halos", data=mpb_halos, compression="gzip", compression_opts=4)

    return mpb_halos

if __name__ == "__main__":
    start_snap = int(sys.argv[1])
    target_snap = int(sys.argv[2])
    basePath = "/public/share/chenhouzun/TNG50-1/output/"
    load_mpb(basePath, start_snap, target_snap)
