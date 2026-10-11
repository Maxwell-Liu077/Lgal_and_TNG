import illustris_python as il
import numpy as np
import h5py
from os.path import isfile, isdir
from os import mkdir
from mergertree import loadMPBs
from types import SimpleNamespace

def create_subhalo_sample(basePath, start_snap):

    header = il.groupcat.loadHeader(basePath,start_snap)
    num_subs = header['Nsubgroups_Total']
    h_const = header['HubbleParam']

    #introduce mass bins (just for analysis, not for computation):
    central_ids = il.groupcat.loadHalos(basePath, start_snap, fields = ['GroupFirstSub'])

    sub_ids = np.arange(num_subs)

    #Filter out halos without any subhalo and satellites
    subhaloFlag = np.zeros(num_subs, dtype = np.ubyte)
    subhaloFlag[central_ids[np.where(central_ids != -1)]] = 1

    sP = SimpleNamespace(simPath=basePath, snap=start_snap)
    
    #load MPB trees
    trees = loadMPBs(sP, ids = sub_ids, fields = ['SubfindID'], treeName="SubLink")
    
    #load data from files ---------------------------------------------------------------------------------
    assert isfile('/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/SubhaloPos_new_extrapolated.hdf5')
    sub_positions = h5py.File('/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/SubhaloPos_new_extrapolated.hdf5','r') 
    is_extrapolated = sub_positions['is_extrapolated'][:]
    sub_positions.close()
       
    #which galaxies? ----------------------------------------------------------------------------------------
    not_extrapolated = np.nonzero(np.logical_not(is_extrapolated))[0]
    subhaloFlag[not_extrapolated] = 0
    
    del is_extrapolated, not_extrapolated
 
    #tree check: find missing trees and delete according galaxies from the flag
    tree_ids = np.asarray(list(trees.keys()), dtype=np.int64)
    tree_ids = tree_ids[(tree_ids >= 0) & (tree_ids < num_subs)]
    has_tree = np.zeros(num_subs, dtype=bool)
    has_tree[tree_ids] = True
    subhaloFlag[~has_tree] = 0
            
    print('# galaxies with trees: ', np.count_nonzero(subhaloFlag))

    # introduce mass cut: consider only galaxies with stellar mass > 10^8.5 Msun at z=0
    stellar_masses = il.groupcat.loadSubhalos(basePath, start_snap, fields = ['SubhaloMassInRadType'])[:,4]
    mass_mask = np.where(stellar_masses * 1e10 / h_const < 10**(8.5))[0]
    subhaloFlag[mass_mask] = 0

    print('# galaxies above logM_star = 8.5: ', np.where(subhaloFlag == 1)[0].shape[0])

    dir_name = '/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data'
    if not isdir(dir_name):
        print('Directory does not exist, creating it...')
        mkdir(dir_name)

    filename = dir_name + '/subhaloFlag.hdf5'

    f = h5py.File(filename,'w')
    f.create_dataset('subhaloFlag', data = subhaloFlag)
    f.close()
        
    return subhaloFlag
