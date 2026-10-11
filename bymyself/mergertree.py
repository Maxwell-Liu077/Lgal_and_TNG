import h5py
import illustris_python as il
import numpy as np
from numba import jit

treeName_default = "SubLink"

def loadMPBs(sP, ids, fields=None, treeName=treeName_default):
    """Load multiple MPBs at once (e.g. all of them), optimized for speed, with a full tree load (high mem).

    Basically a rewrite of illustris_python/sublink.py under specific conditions (hopefully temporary).

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      ids (list[int]): list of subhalo IDs to load.
      fields (list[str]): list of field names to load, or None for all (not recommended).
      treeName (str): which merger tree to use? 'SubLink' or 'SubLink_gal'.

    Returns:
      dict: Dictionary of MPBs where keys are subhalo IDs, and the contents of each dict value is another
      dictionary of identical stucture to the return of loadMPB().
    """
    from glob import glob

    assert treeName in ["SubLink", "SubLink_gal"]  # otherwise need to generalize tree loading

    # make sure fields is not a single element
    if isinstance(fields, str):
        fields = [fields]

    fieldsLoad = fields + ["MainLeafProgenitorID"]

    # find full tree data sizes and attributes
    numTreeFiles = len(glob(il.sublink.treePath(sP.simPath, treeName, "*")))

    lengths = {}
    dtypes = {}
    seconddims = {}

    for field in fieldsLoad:
        lengths[field] = 0
        seconddims[field] = 0

    for i in range(numTreeFiles):
        with h5py.File(il.sublink.treePath(sP.simPath, treeName, i), "r") as f:
            for field in fieldsLoad:
                dtypes[field] = f[field].dtype
                lengths[field] += f[field].shape[0]
                if len(f[field].shape) > 1:
                    seconddims[field] = f[field].shape[1]

    # allocate for a full load
    fulltree = {}

    for field in fieldsLoad:
        if seconddims[field] == 0:
            fulltree[field] = np.zeros(lengths[field], dtype=dtypes[field])
        else:
            fulltree[field] = np.zeros((lengths[field], seconddims[field]), dtype=dtypes[field])

    # load full tree
    offset = 0

    for i in range(numTreeFiles):
        with h5py.File(il.sublink.treePath(sP.simPath, treeName, i), "r") as f:
            for field in fieldsLoad:
                if seconddims[field] == 0:
                    fulltree[field][offset : offset + f[field].shape[0]] = f[field][()]
                else:
                    fulltree[field][offset : offset + f[field].shape[0], :] = f[field][()]
            offset += f[field].shape[0]

    result = {}

    # (Step 1) treeOffsets()
    offsetFile = il.groupcat.offsetPath(sP.simPath, sP.snap)
    prefix = "Subhalo/" + treeName + "/"

    with h5py.File(offsetFile, "r") as f:
        # load all merger tree offsets
        if prefix + "RowNum" not in f:
            return result  # early snapshots, no tree offset

        RowNums = f[prefix + "RowNum"][()]
        SubhaloIDs = f[prefix + "SubhaloID"][()]

    # now subhalos one at a time (memory operations only)
    for id in ids:
        if id == -1:
            continue  # skip requests for e.g. fof halos which had no central subhalo

        # (Step 2) loadTree()
        RowNum = RowNums[id]
        SubhaloID = SubhaloIDs[id]
        MainLeafProgenitorID = fulltree["MainLeafProgenitorID"][RowNum]

        if RowNum == -1:
            continue

        # load only main progenitor branch
        rowStart = RowNum
        rowEnd = RowNum + (MainLeafProgenitorID - SubhaloID)
        nRows = rowEnd - rowStart + 1

        # init dict
        result[id] = {"count": nRows}

        # loop over each requested field and copy, no error checking
        for field in fields:
            result[id][field] = fulltree[field][RowNum : RowNum + nRows]

    return result


def loadDescendants(sP, ids, fields=None, treeName=treeName_default):
    """Load the single direct descandant (at the next snapshot), for many subhalos at once, optimized for speed.

    Note: uses a full tree load (high mem).

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      ids (list[int]): list of subhalo IDs to load.
      fields (list[str]): list of field names to load, or None for all (not recommended).
      treeName (str): which merger tree to use? 'SubLink' or 'SubLink_gal'.

    Returns:
      dict: Dictionary of descendants where keys are subhalo IDs, and the contents of each dict value is another
      dictionary of identical stucture to the return of loadTree(onlyMDB=True).
    """
    from glob import glob

    assert treeName in ["SubLink", "SubLink_gal"]  # otherwise need to generalize tree loading

    # make sure fields is not a single element
    if isinstance(fields, str):
        fields = [fields]

    fieldsLoad = fields + ["DescendantID"]

    # find full tree data sizes and attributes
    numTreeFiles = len(glob(il.sublink.treePath(sP.simPath, treeName, "*")))

    lengths = {}
    dtypes = {}
    seconddims = {}

    for field in fieldsLoad:
        lengths[field] = 0
        seconddims[field] = 0

    for i in range(numTreeFiles):
        with h5py.File(il.sublink.treePath(sP.simPath, treeName, i), "r") as f:
            for field in fieldsLoad:
                dtypes[field] = f[field].dtype
                lengths[field] += f[field].shape[0]
                if len(f[field].shape) > 1:
                    seconddims[field] = f[field].shape[1]

    # allocate for a full load
    fulltree = {}

    for field in fieldsLoad:
        if seconddims[field] == 0:
            fulltree[field] = np.zeros(lengths[field], dtype=dtypes[field])
        else:
            fulltree[field] = np.zeros((lengths[field], seconddims[field]), dtype=dtypes[field])

    # load full tree
    offset = 0

    for i in range(numTreeFiles):
        with h5py.File(il.sublink.treePath(sP.simPath, treeName, i), "r") as f:
            for field in fieldsLoad:
                if seconddims[field] == 0:
                    fulltree[field][offset : offset + f[field].shape[0]] = f[field][()]
                else:
                    fulltree[field][offset : offset + f[field].shape[0], :] = f[field][()]
            offset += f[field].shape[0]

    result = {}

    # (Step 1) treeOffsets()
    offsetFile = il.groupcat.offsetPath(sP.simPath, sP.snap)
    prefix = "Subhalo/" + treeName + "/"

    with h5py.File(offsetFile, "r") as f:
        # load all merger tree offsets
        if prefix + "RowNum" not in f:
            return result  # early snapshots, no tree offset

        RowNums = f[prefix + "RowNum"][()]
        SubhaloIDs = f[prefix + "SubhaloID"][()]

    # now subhalos one at a time (memory operations only)
    for id in ids:
        if id == -1:
            continue  # skip requests for e.g. fof halos which had no central subhalo

        # (Step 2) loadTree()
        RowNum = RowNums[id]
        SubhaloID = SubhaloIDs[id]
        DescendantID = fulltree["DescendantID"][RowNum]

        if RowNum == -1:
            continue

        # load only main progenitor branch
        index_descendant = RowNum - (SubhaloID - DescendantID)

        if DescendantID == -1:
            continue

        # init dict
        result[id] = {"count": 1}

        # loop over each requested field and copy, no error checking
        for field in fields:
            result[id][field] = fulltree[field][index_descendant]

    return result