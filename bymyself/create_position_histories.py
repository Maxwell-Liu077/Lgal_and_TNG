import illustris_python as il
import numpy as np
import h5py
import numba as nb
from numba import njit
from numpy.linalg import inv
from os.path import isfile
import illustrisFuncs as iF
from mergertree import loadMPBs
from types import SimpleNamespace

def save_subhalo_pos(basePath):
    snap_min = 40
    snap_max = 99
    snaps = np.arange(snap_max, snap_min - 1, -1, dtype=np.int64)
    num_snaps = snaps.size

    sub_flag = il.groupcat.loadSubhalos(basePath,snap_max,fields=["SubhaloFlag"])
    num_subs = sub_flag.size
    subhalo_ids = np.arange(num_subs, dtype=np.int64)

    sP = SimpleNamespace(simPath=basePath, snap=snap_max)
    trees = loadMPBs(sP,ids=subhalo_ids,fields=["SnapNum", "SubfindID", "SubhaloPos"],treeName="SubLink")

    subhalo_pos = np.full((num_subs, num_snaps, 3),np.nan,dtype=np.float32)

    for start_subhalo_id, tree in trees.items():
        start_subhalo_id = int(start_subhalo_id)

        if not (0 <= start_subhalo_id < num_subs):
            continue

        tree_snaps = np.asarray(tree["SnapNum"], dtype=np.int64)
        tree_subfind_ids = np.asarray(tree["SubfindID"], dtype=np.int64)
        tree_pos = np.asarray(tree["SubhaloPos"], dtype=np.float32)

        in_range = (tree_snaps >= snap_min) & (tree_snaps <= snap_max)
        columns = snap_max - tree_snaps[in_range]

        subhalo_pos[start_subhalo_id, columns, :] = tree_pos[in_range]

    output_file = "/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/SubhaloPos_new.hdf5"

    with h5py.File(output_file, "w") as f:
        f.create_dataset("SubhaloPos",data=subhalo_pos,compression="gzip",compression_opts=4)
        f.create_dataset("SubhaloFlag", data=sub_flag)

    return

@njit
def linear(x,a,b):
    return a*x+b

@njit
def linear_fit(x, y):
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    A = np.column_stack((np.ones(x.shape[0]), x))
    opt_param = np.ascontiguousarray(inv(A.T @ A)) @ A.T @ y
    return opt_param

@njit
def extrapolate_new(arr, a, boxSize, bound):
    is_extrapolated = np.full(arr.shape[0], False)
    for i in nb.prange(arr.shape[0]): #extrapolate for every subhalo...
        for j in range(3): #... and every coordinate
         
            #if first or second coordinate could not be extrapolated:
            if j > 0 and is_extrapolated[i] == False:
                continue #skip the other ones
            coord = arr[i,:,j]
            
            good_indices = np.arange(coord.size)
            gaps = np.where(np.isnan(coord))[0]
            
            if gaps.size > 40:
                is_extrapolated[i] = False
                continue
            
            top = np.where(coord > (1 - bound) * boxSize)[0]
            bottom = np.where(coord < bound * boxSize)[0]
            
            if top.size > 0 and bottom.size > 0:
                if top.size >= bottom.size:
                    gaps = np.concatenate((gaps, bottom))
                else:
                    gaps = np.concatenate((gaps, top))
            
            good_indices = np.delete(good_indices, gaps)

            good_reversed = good_indices[::-1]
            gap_coord = np.interp(a[gaps],a[good_reversed],coord[good_reversed])
            coord[gaps] = gap_coord
            new_coord = coord
            
            k = max(good_indices)
            if k < coord.size - 1:
                #only extrapolate with four or more (good nonzero-) entries
                if(good_indices.size > 5): 
                    popt = linear_fit(a[good_indices[-5:]],coord[good_indices[-5:]])
                    new_coord[k:] = linear(a[k:],popt[1], popt[0])

            new_coord[np.where(new_coord > boxSize)] -= boxSize
            new_coord[np.where(new_coord < 0)] += boxSize
            is_extrapolated[i] = True

            arr[i,:,j] = new_coord

    return arr, is_extrapolated

#extrapolate positions
def extrapolatePos(basePath, boxSize, a, bound):
    assert isfile('/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/SubhaloPos_new.hdf5'), 'File does not exist!'
    f = h5py.File('/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/SubhaloPos_new.hdf5', "r")
    arr = f['SubhaloPos'][:,:,:]

    boxSize = il.groupcat.loadHeader(basePath,99)['BoxSize']
    snap_min = 40
    snap_max = 99
    snaps = np.arange(snap_max, snap_min - 1, -1, dtype=np.int64)
    z = np.array([il.groupcat.loadHeader(basePath,snap)["Redshift"] for snap in snaps])
    a = 1/(1+z)
    bound = 0.1

    #extrapolate positions and save
    res, is_extrapolated = extrapolate_new(arr, a, boxSize, bound) 

    #specify path to your directory
    result = h5py.File('/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/SubhaloPos_new_extrapolated.hdf5', 'w')
    result.create_dataset('SubhaloPos',data = res)
    result.create_dataset('is_extrapolated', data = is_extrapolated)
    f.close()
    result.close()
    return

def save_always_central_trees(basePath, start_snap, subhaloFlag, snap_min=40, snap_max=99):

    snaps = np.arange(snap_max, snap_min - 1, -1, dtype=np.int64)
    n_snaps = snaps.size
    subhaloFlag = np.asarray(subhaloFlag, dtype=bool)
    selected_start_ids = np.flatnonzero(subhaloFlag).astype(np.int64)

    central_fof_by_snap = {}
    for snap in snaps:
        group_first_sub = il.groupcat.loadHalos(basePath, snap, fields=["GroupFirstSub"])
        raw = np.asarray(group_first_sub, dtype=np.int64).ravel()
        valid = raw >= 0

        central_fof_by_snap[snap] = {subhalo_id: fof_id for fof_id, subhalo_id in enumerate(raw) if valid[fof_id]}
    
    sP = SimpleNamespace(simPath=basePath, snap=start_snap)
    trees = loadMPBs(sP, ids=selected_start_ids, fields=["SnapNum", "SubfindID"], treeName="SubLink")

    accepted_start_ids = []
    accepted_node_ids = []
    accepted_fof_ids = []

    for start_id in selected_start_ids:
        tree = trees.get(start_id)
        if tree is None:
            continue

        tree_snaps = np.asarray(tree["SnapNum"], dtype=np.int64)
        tree_subfind_ids = np.asarray(tree["SubfindID"], dtype=np.int64)

        node_ids = np.full(n_snaps, -1, dtype=np.int64)
        node_fof_ids = np.full(n_snaps, -1, dtype=np.int64)

        for node_snap, node_id in zip(tree_snaps, tree_subfind_ids):

            if not (snap_min <= node_snap <= snap_max):
                continue

            column = snap_max - node_snap
            node_ids[column] = node_id

        node_fof_ids = np.full(n_snaps, -1, dtype=np.int64)

        for column, (snap, node_id) in enumerate(zip(snaps, node_ids)):

            fof_id = central_fof_by_snap[snap].get(int(node_id), -1)
            if fof_id < 0:
                break

            node_fof_ids[column] = fof_id

        if np.any(node_fof_ids < 0):
            continue

        accepted_start_ids.append(start_id)
        accepted_node_ids.append(node_ids)
        accepted_fof_ids.append(node_fof_ids)

    if accepted_start_ids:
        accepted_start_ids = np.asarray(accepted_start_ids, dtype=np.int64)
        accepted_node_ids = np.stack(accepted_node_ids)
        accepted_fof_ids = np.stack(accepted_fof_ids)
    else:
        accepted_start_ids = np.empty(0, dtype=np.int64)
        accepted_node_ids = np.empty((0, n_snaps), dtype=np.int64)
        accepted_fof_ids = np.empty((0, n_snaps), dtype=np.int64)

    output_file = "/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/all_selected_trees.hdf5"

    with h5py.File(output_file, "w") as f:
        f.create_dataset("start_subhalo_id", data=accepted_start_ids)
        f.create_dataset("snap_numbers", data=snaps)
        f.create_dataset("mpb_subhalo_ids", data=accepted_node_ids)
        f.create_dataset("mpb_fof_ids", data=accepted_fof_ids)

    return accepted_start_ids, accepted_node_ids, accepted_fof_ids
