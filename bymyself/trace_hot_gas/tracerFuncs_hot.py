import numpy as np
from numba import jit, njit
import numba as nb
import illustris_python as il
import funcs
import h5py
import time
from pathlib import Path

data_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")

# 找出那些父对象 ID 属于 gas_ids 的示踪粒子，并返回它们的 ID 和位置索引
@njit
def findTracerIDs(gas_ids, tracer_ids, parent_ids):
    indices = np.nonzero(funcs.isin(parent_ids,gas_ids))[0]
    return tracer_ids[indices], indices

# 找出 ids 中属于 search_ids 的元素位置，并返回这些元素的索引数组
@njit
def getIndices(search_ids,ids):
    indices = np.nonzero(funcs.isin(ids,search_ids))[0]
    return indices

# 按照 parentIDs 排序，并用同样的排序顺序重新排列 tracer_indices，从而让属于同一个父粒子的 tracer 排在一起。
@njit
def sortIDs(parentIDs, tracer_indices):
    parent_sorted_indices = np.argsort(parentIDs) # 返回排序后应该使用的原始索引
    tracer_sorted_indices = np.ravel(tracer_indices)[parent_sorted_indices] # 用相同顺序排列 tracer 索引
    return parentIDs[parent_sorted_indices], tracer_sorted_indices

# 在 start_snap 中找到属于指定气体粒子的 tracer，然后追踪这些 tracer 到 target_snap，最后判断它们在目标快照中属于哪些粒子
@jit(forceobj=True) # 由于函数内部调用了 il.snapshot.loadSubset
def tracersOfSubhalo(basePath, gas_ids, start_snap, target_snap):
    #load tracers at z=0
    tracers = il.snapshot.loadSubset(basePath,start_snap,3)
    tracer_ids = tracers['TracerID'].copy()
    parent_ids = tracers['ParentID'].copy()
    print('tracers in total: ',tracer_ids.size)

    #load tracers at target snapshot
    target_tracers = il.snapshot.loadSubset(basePath,target_snap,3)
    parent_target_ids = target_tracers['ParentID']
    tracer_target_ids = target_tracers['TracerID']

    #find IDs of Tracers belonging to relevant gases
    tracer_search_ids, tracer_indices = findTracerIDs(gas_ids,tracer_ids,parent_ids)

    #load all relevant parent particles
    all_gas_ids = il.snapshot.loadSubset(basePath,target_snap,0,['ParticleIDs'])
    all_star_ids = il.snapshot.loadSubset(basePath,target_snap,4,['ParticleIDs'])
    all_bh_ids = il.snapshot.loadSubset(basePath,target_snap,5,['ParticleIDs'])

    #get indices of relevant tracers in array of tracers in target snapshot
    tracer_target_indices = getIndices(tracer_search_ids,tracer_target_ids)

    #sort parents as well as tracers according to sorting of parents
    parent_target_ids, tracer_target_indices = sortIDs(parent_target_ids[tracer_target_indices],tracer_target_indices)

    #identify parent types
    gas_indices = getIndices(parent_target_ids,all_gas_ids)
    star_indices = getIndices(parent_target_ids,all_star_ids)
    bh_indices = getIndices(parent_target_ids,all_bh_ids)

    return all_gas_ids[gas_indices], all_star_ids[star_indices], all_bh_ids[bh_indices]

# 根据每个子晕包含的气体范围，统计每个子晕拥有的 tracer 数量，并计算其累计数量
@njit
def tracersInSubhalo(GasInSubOffset, TracersInGasOffset):
    n = GasInSubOffset.shape[0]
    TracersInSubOffset = np.zeros(n)
    for i in nb.prange(n-1):
        TracersInSubOffset[i] = np.sum(TracersInGasOffset[GasInSubOffset[i]:GasInSubOffset[i+1]])
    TracersInSubOffset = np.cumsum(TracersInSubOffset) # 计算累计和
    return TracersInSubOffset

# 按照 gas_ids 的顺序，把属于同一气体粒子的 tracer 排到一起，并同步重新排列 tracer_ids 和 parent_ids
def createSortedArrays(gas_ids, parent_ids, tracer_ids):
    parent_ids_copy = np.copy(parent_ids)
    parent_ids_copy = np.sort(parent_ids_copy)

    num_TracersPerParent = np.searchsorted(parent_ids_copy,gas_ids,'right') - np.searchsorted(parent_ids_copy,gas_ids,'left')
    parent_ids_offsets = np.cumsum(num_TracersPerParent)
    parent_ids_offsets = np.insert(parent_ids_offsets, 0, 0)

    search_parents = np.zeros(parent_ids_copy.size, dtype=np.uint64)

    for i in range(gas_ids.size):
        search_parents[parent_ids_offsets[i]:parent_ids_offsets[i+1]] = gas_ids[i]

    order = np.argsort(parent_ids)[np.argsort(np.argsort(search_parents))] # 双重 argsort 可以得到每个元素在排序后的排名

    del parent_ids_copy

    return tracer_ids[order], parent_ids[order], num_TracersPerParent

# 用二分法在 ar1 中查找 ar2 的元素，并返回对应索引；找不到的元素用 -1 表示
def match(ar1, ar2, is_sorted = False):

    if isinstance(ar1, dict) or len(ar1) == 0:
        return np.full(len(ar2), -1, dtype=np.int64), np.empty(0, dtype=np.int64)

    if not is_sorted:
        # need a sorted copy of ar1 to run bisection against
        index = np.argsort(ar1)
        ar1_sorted = ar1[index]
    else:
        ar1_sorted = ar1
        index = np.arange(ar1.shape[0])
    # NlogN search of ar1_sorted for each element in ar2
    ar1_sorted_index = np.searchsorted(ar1_sorted, ar2)

    # 对 ar2 中每个元素做二分查找
    ar1_inds = np.take(index, ar1_sorted_index, mode="clip") # mode="clip" 会把越界索引限制到最后一个有效位置，避免 ar2 中较大的值导致数组越界

    # filter out non-matches
    mask = (ar1[ar1_inds] == ar2)
    ar2_inds = np.where(mask)[0]

    #fill non-matches with -1 for later usage
    ar1_inds[np.where(np.logical_not(mask))[0]] = -1 #shape of gas_inds doen't match

    return ar1_inds, ar2_inds

# 用二分法在 ar1 中查找 ar2 的元素，并返回对应索引；删除找不到的元素
def match_general(ar1, ar2, is_sorted = False):

    if type(ar1) == dict:
        return -1, np.array([])

    if not is_sorted:
        # need a sorted copy of ar1 to run bisection against
        index = np.argsort(ar1)
        ar1_sorted = ar1[index]
    else:
        ar1_sorted = ar1
        index = np.arange(ar1.shape[0])
    # NlogN search of ar1_sorted for each element in ar2
    ar1_sorted_index = np.searchsorted(ar1_sorted, ar2)

    # undo sort
    ar1_inds = np.take(index, ar1_sorted_index, mode="clip")
#     ar1_inds = funcs.take_nb_clip(index, ar1_sorted_index)

    # filter out non-matches
    mask = (ar1[ar1_inds] == ar2)
    ar2_inds = np.where(mask)[0]
    ar1_inds = ar1_inds[ar2_inds]

    return ar1_inds, ar2_inds

# 对每一个 parent_id，判断它对应的是什么种类的粒子，并记录它在相应粒子数组中的索引
def parentIndicesOfAll(parent_ids, all_gas_ids, all_star_ids, all_bh_ids): #NlogN
    #save index into gas('0') / star('1') / bh('2') subset array
    target_parent_indices = np.full((len(parent_ids),2), -1, dtype=np.int64)
    gas_inds, _ = match(all_gas_ids, parent_ids)
    print('gas indices done')

    n_parent = len(parent_ids)

    if all_star_ids.shape[0] > 0:
        star_inds, _ = match(all_star_ids, parent_ids)
    else:
        star_inds = np.full(n_parent, -1, dtype=np.int64)
    print('star indices done')

    if all_bh_ids.shape[0] > 0:
        bh_inds, _ = match(all_bh_ids, parent_ids)
    else:
        bh_inds = np.full(n_parent, -1, dtype=np.int64)
    print('bh indices done')

    gas_match = gas_inds >= 0
    star_match = star_inds >= 0
    bh_match = bh_inds >= 0

    #gas_inds now contains either the indices of gas parents into the gas subset or -1; same for star_inds
    target_parent_indices[gas_match,0] = gas_inds[gas_match]
    target_parent_indices[gas_match,1] = 0
    target_parent_indices[star_match,0] = star_inds[star_match]
    target_parent_indices[star_match, 1] = 1
    target_parent_indices[bh_match,0] = bh_inds[bh_match]
    target_parent_indices[bh_match, 1] = 2

    del gas_inds, star_inds, bh_inds
    print('filling done')

    return target_parent_indices

def TraceAllHotGas(basePath, gas_ids, start_snap, target_snap, return_tracer_data=False):
    """Trace supplied hot-gas IDs; preserve the existing two-result API by default."""
    start = time.time()
    gas_ids = np.asarray(gas_ids, dtype=np.uint64)
    if gas_ids.size == 0:
        empty_parents = np.empty((0, 2), dtype=np.int64)
        empty_counts = np.empty(0, dtype=np.int64)
        if return_tracer_data:
            return empty_parents, empty_counts, np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.uint64), np.empty(0, dtype=bool)
        return empty_parents, empty_counts
    tracers = il.snapshot.loadSubset(basePath, start_snap, 3, fields=["TracerID", "ParentID"])
    tracer_ids = np.asarray(tracers.get("TracerID", []), dtype=np.uint64)
    parent_ids = np.asarray(tracers.get("ParentID", []), dtype=np.uint64)
    del tracers

    load = time.time()
    print('time for tracer loading: ',load-start)

    #find IDs of Tracers belonging to relevant gas
    _, tracer_indices = findTracerIDs(gas_ids,tracer_ids,parent_ids)

    find_tracers = time.time()
    print('time for finding tracers: ',find_tracers-load, flush = True)

    #rearange all parent and tracer ids
    tracer_search_ids, _, numTracersInParents =\
    createSortedArrays(gas_ids, parent_ids[tracer_indices], tracer_ids[tracer_indices])

    create_arrays = time.time()
    print('time for creating sorted arrays: ',create_arrays-find_tracers, flush = True)
    del parent_ids, tracer_ids, gas_ids

    #load tracers at target snapshot
    target_tracers = il.snapshot.loadSubset(basePath, target_snap, 3, fields=["TracerID", "ParentID"])
    tracer_target_ids = np.asarray(target_tracers.get("TracerID", []), dtype=np.uint64)
    parent_target_ids = np.asarray(target_tracers.get("ParentID", []), dtype=np.uint64)
    del target_tracers

    after_load2 = time.time()

    #get indices of relevant tracers in array of tracers in target snapshot
    tracer_target_indices, _ = match(tracer_target_ids, tracer_search_ids)
    tracer_found = tracer_target_indices >= 0
    target_parent_search_ids = np.zeros(tracer_search_ids.size, dtype=np.uint64)
    target_parent_search_ids[tracer_found] = parent_target_ids[tracer_target_indices[tracer_found]]

    del tracer_target_ids, parent_target_ids

    sort = time.time()
    print('time for getting indices and sorting: ',sort - after_load2, flush = True)

    #load all relevant parent particles at target snapshot
    all_target_gas_ids = il.snapshot.loadSubset(basePath,target_snap,0,['ParticleIDs'])
    all_target_star_ids = il.snapshot.loadSubset(basePath,target_snap,4,['ParticleIDs'])
    all_target_bh_ids = il.snapshot.loadSubset(basePath,target_snap,5,['ParticleIDs'])

    if isinstance(all_target_gas_ids, dict):
        all_target_gas_ids = np.empty(0, dtype=np.uint64)
    if isinstance(all_target_star_ids, dict):
        all_target_star_ids = np.empty(0, dtype=np.uint64)
    if isinstance(all_target_bh_ids, dict):
        all_target_bh_ids = np.empty(0, dtype=np.uint64)

    after_load3 = time.time()

    target_parent_indices = np.full((tracer_search_ids.size, 2), -1, dtype=np.int64)
    target_parent_indices[tracer_found] = parentIndicesOfAll(parent_ids=target_parent_search_ids[tracer_found], all_gas_ids=all_target_gas_ids, all_star_ids=all_target_star_ids, all_bh_ids=all_target_bh_ids)

    end = time.time()
    print('time for finding indices into subset arrays: ',end-after_load3, flush = True)
    print('total time: ',end-start, flush = True)
    if return_tracer_data:
        return target_parent_indices, numTracersInParents, tracer_search_ids, target_parent_search_ids, tracer_found
    return target_parent_indices, numTracersInParents

# 读取 start_snap 的样本晕热气体，追踪到后继快照并保存 parent 索引。
def TraceBackAllHotGas(basePath,start_snap,target_snap):
    """Keep the existing entry point, tracing start-snapshot hot gas forward."""
    if not 0 <= start_snap < target_snap <= 99:
        raise ValueError("Hot-gas forward tracking requires 0 <= start_snap < target_snap <= 99.")
    class_file = data_dir / f"gas_parent_classification_hot_{start_snap}.hdf5"
    output_file = data_dir / f"parent_indices_hot_start_{start_snap}_target_{target_snap}.hdf5"

    with h5py.File(class_file, "r") as f:
        gas_ids = np.asarray(f["gas_particle_id"][:], dtype=np.uint64)
        start_ids = np.asarray(f["sample_start_subhalo_id"][:], dtype=np.int64)
        halo_ids = np.asarray(f["halo_id"][:], dtype=np.int64)
        halo_offsets = np.asarray(f["halo_offsets"][:], dtype=np.int64)

    if halo_offsets.shape != (start_ids.size + 1,) or halo_offsets[0] != 0 or halo_offsets[-1] != gas_ids.size or np.any(np.diff(halo_offsets) < 0):
        raise ValueError("Hot-gas offsets do not match the selected-tree rows and gas parents.")
    parent_indices, numTracersInParents, tracer_ids, target_parent_ids, tracer_found = TraceAllHotGas(basePath, gas_ids, start_snap, target_snap, return_tracer_data=True)
    tracer_offsets_per_parent = np.concatenate(([0], np.cumsum(numTracersInParents, dtype=np.int64)))
    if tracer_offsets_per_parent[-1] != parent_indices.shape[0]:
        raise ValueError("Per-parent tracer counts do not match the number of tracked tracers.")

    with h5py.File(output_file, "w") as result:
        result.attrs["start_snap"] = start_snap
        result.attrs["target_snap"] = target_snap
        result.attrs["parent_type_definition"] = "0 = gas; 1 = PartType4 (star/wind); 2 = BH; -1 = unknown"
        result.create_dataset("sample_start_subhalo_id", data=start_ids)
        result.create_dataset("halo_id", data=halo_ids)
        result.create_dataset("halo_tracer_offsets", data=tracer_offsets_per_parent[halo_offsets])
        grp = result.create_group(f"snap_{target_snap:03d}")
        grp.create_dataset("parent_indices", data=parent_indices)
        grp.create_dataset("numTracersInParents", data=numTracersInParents)
        grp.create_dataset("tracer_id", data=tracer_ids)
        grp.create_dataset("target_parent_id", data=target_parent_ids)
        grp.create_dataset("tracer_found", data=tracer_found.astype(np.uint8))
    print(target_snap, 'done',end = '; ', flush=True) # 打印
    return

# 保留旧函数名，读取第一步已经保存的样本晕热气体 offsets。
def hotGasInSubOffset(basePath, snap):
    """Offsets follow selected-tree rows, including empty or invalid start rows."""
    with h5py.File(data_dir / f"gas_parent_classification_hot_{snap}.hdf5", "r") as f:
        return np.asarray(f["halo_offsets"][:], dtype=np.int64)

# 统计每个 subhalo 中的 Inner Gas 数量，并将其转换为可用于数组分组的累计 offset
@njit(parallel = True)
def compute_offsets(gasInSubOffset, numGasInSubs, hot):
    hotGasInSubOffset = np.zeros(gasInSubOffset.shape[0], dtype = np.int32)
    for i in nb.prange(1,gasInSubOffset.shape[0]): # gasInSubOffset[i] 存储前面 subhalo 的累计数量
        gas_indices = np.arange(gasInSubOffset[i-1],gasInSubOffset[i-1] + numGasInSubs[i-1])
        hot_indices = hot[gas_indices]
        hotGasInSubOffset[i] = len(np.nonzero(hot_indices)[0])

    hotGasInSubOffset = np.cumsum(hotGasInSubOffset)
    return hotGasInSubOffset
