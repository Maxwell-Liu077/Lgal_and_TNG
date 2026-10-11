import numpy as np
from numba import jit, njit
import numba as nb
import illustris_python as il
import tracerFuncs as tF
import funcs
import h5py
import time
from pathlib import Path

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

    search_parents = np.zeros(parent_ids_copy.size)

    for i in range(gas_ids.size):
        search_parents[parent_ids_offsets[i]:parent_ids_offsets[i+1]] = gas_ids[i]
        
    order = np.argsort(parent_ids)[np.argsort(np.argsort(search_parents))] # 双重 argsort 可以得到每个元素在排序后的排名

    del parent_ids_copy
    
    return tracer_ids[order], parent_ids[order], num_TracersPerParent

# 用二分法在 ar1 中查找 ar2 的元素，并返回对应索引；找不到的元素用 -1 表示
def match(ar1, ar2, is_sorted = False):
  
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
    gas_inds, _ = tF.match(all_gas_ids, parent_ids)
    print('gas indices done')

    n_parent = len(parent_ids)
    
    if all_star_ids.shape[0] > 0:
        star_inds, _ = tF.match(all_star_ids, parent_ids)
    else:
        star_inds = np.full(n_parent, -1, dtype=np.int64)
    print('star indices done')

    if all_bh_ids.shape[0] > 0:
        bh_inds, _ = tF.match(all_bh_ids, parent_ids)
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

def TraceAllInnerGas(basePath, gas_ids, start_snap, target_snap):    
    start = time.time()
    tracer_ids = il.snapshot.loadSubset(basePath,start_snap,3, fields = ['TracerID'])
    parent_ids = il.snapshot.loadSubset(basePath,start_snap,3, fields = ['ParentID'])
    
    load = time.time()
    print('time for tracer loading: ',load-start)
    
    #find IDs of Tracers belonging to relevant gas
    _, tracer_indices = tF.findTracerIDs(gas_ids,tracer_ids,parent_ids)
    
    find_tracers = time.time()
    print('time for finding tracers: ',find_tracers-load, flush = True)
    
    #rearange all parent and tracer ids
    tracer_search_ids, _, numTracersInParents =\
    createSortedArrays(gas_ids, parent_ids[tracer_indices], tracer_ids[tracer_indices])
    
    create_arrays = time.time()
    print('time for creating sorted arrays: ',create_arrays-find_tracers, flush = True)
    del parent_ids, tracer_ids, gas_ids
      
    #load tracers at target snapshot
    tracer_target_ids = il.snapshot.loadSubset(basePath,target_snap,3, fields = ['TracerID'])
    parent_target_ids = il.snapshot.loadSubset(basePath,target_snap,3, fields = ['ParentID'])
    
    after_load2 = time.time()
    
    #get indices of relevant tracers in array of tracers in target snapshot
    tracer_target_indices = tF.getIndices(tracer_search_ids,tracer_target_ids)
    
    target_tracer_search_ids = tracer_target_ids[tracer_target_indices].copy()
    target_parent_search_ids = parent_target_ids[tracer_target_indices].copy()
    
    del tracer_target_ids, parent_target_ids
    
    original_order = np.argsort(target_tracer_search_ids)[np.argsort(np.argsort(tracer_search_ids))]
    
    del tracer_search_ids, target_tracer_search_ids
    
    sort = time.time()
    print('time for getting indices and sorting: ',sort - after_load2, flush = True)
    
    #sort parents as well as tracers according to sorting of parents at start snap
    target_parent_search_ids = target_parent_search_ids[original_order]
    
    del original_order
    
    #load all relevant parent particles at target snapshot
    all_target_gas_ids = il.snapshot.loadSubset(basePath,target_snap,0,['ParticleIDs'])
    all_target_star_ids = il.snapshot.loadSubset(basePath,target_snap,4,['ParticleIDs'])
    all_target_bh_ids = il.snapshot.loadSubset(basePath,target_snap,5,['ParticleIDs'])
    
    if type(all_target_star_ids) == dict:
        all_target_star_ids = np.array([])
    
    after_load3 = time.time()
    
    target_parent_indices = parentIndicesOfAll(parent_ids = target_parent_search_ids, all_gas_ids = all_target_gas_ids,\
                                               all_star_ids = all_target_star_ids, all_bh_ids = all_target_bh_ids)
    
    end = time.time()
    print('time for finding indices into subset arrays: ',end-after_load3, flush = True)
    print('total time: ',end-start, flush = True)
    return target_parent_indices, numTracersInParents

# 选择 start_snap 中的所有 inner gas，然后对目标 snapshot 逐个调用 TraceInnerGas()，把每个 snapshot 中对应的 parent 索引保存到一个 HDF5 文件中。
def TraceBackAllInnerGas(basePath,start_snap,target_snap):

    data_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")
    class_file = data_dir / f"gas_parent_classification_{start_snap}.hdf5"
    output_file = data_dir / f"parent_indices_start_{start_snap}_target_{target_snap}.hdf5"

    with h5py.File(class_file, "r") as f:
        gas_ids = np.asarray(f["gas_particle_id"][:], dtype=np.uint64)

    result = h5py.File(output_file,'w')
    
    parent_indices, numTracersInParents = tF.TraceAllInnerGas(basePath,gas_ids,start_snap,target_snap)  

    #save results in hdf5 file
    grp = result.create_group(f'snap_0{target_snap}') # group 相当于文件系统中的一个文件夹，用来存放当前 snapshot 的多个数据集
    dset = grp.create_dataset("parent_indices", parent_indices.shape, dtype=np.int64) # 创建 parent_indices 数据集
    dset[:] = parent_indices # 把数组写入数据集
    dset2 = grp.create_dataset('numTracersInParents',numTracersInParents.shape, dtype=np.int64) # 创建 tracer 数量数据集
    dset2[:] = numTracersInParents # 写入 tracer 数量
    print(target_snap, 'done',end = '; ', flush=True) # 打印

    result.close() # 关闭 HDF5 文件
    return

# 根据每个 subhalo 中气体的排列位置和 Inner Gas 分类，计算“压缩后的 Inner Gas 数组”中每个 subhalo 的起止 offset
def innerGasInSubOffset(basePath, snap):

    g = h5py.File('/public/share/chenhouzun/TNG50-1/postprocessing/offsets/offsets_' + str(snap).zfill(3) + '.hdf5','r')
    gasInSubOffset = g['Subhalo/SnapByType'][:,0]
    g.close()
    numGasInSubs = il.groupcat.loadSubhalos(basePath, snap, fields = ['SubhaloLenType'])[:,0]

    check = h5py.File(f'/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/gas_parent_classification_{snap}.hdf5','r')
    inner = check['inner'][:]
    inner = np.asarray(inner == 1)
    check.close()
    
    innerGasInSubOffset = compute_offsets(gasInSubOffset, numGasInSubs, inner)

    return innerGasInSubOffset

# 统计每个 subhalo 中的 Inner Gas 数量，并将其转换为可用于数组分组的累计 offset
@njit(parallel = True)
def compute_offsets(gasInSubOffset, numGasInSubs, inner):
    innerGasInSubOffset = np.zeros(gasInSubOffset.shape[0], dtype = np.int32)
    for i in nb.prange(1,gasInSubOffset.shape[0]): # gasInSubOffset[i] 存储前面 subhalo 的累计数量
        gas_indices = np.arange(gasInSubOffset[i-1],gasInSubOffset[i-1] + numGasInSubs[i-1])
        inner_indices = inner[gas_indices]
        innerGasInSubOffset[i] = len(np.nonzero(inner_indices)[0])
    
    innerGasInSubOffset = np.cumsum(innerGasInSubOffset)
    return innerGasInSubOffset


