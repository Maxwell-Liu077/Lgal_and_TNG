import cooling_radius
import mpb
import h5py
import numpy as np
from os.path import isfile

def compute_Lgal_cooling_rate(basePath, snap_1, snap_2):

    assert isfile(f'/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/mpb_halos.hdf5'), 'Mpb halos file does not exist!'
    file = f'/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/mpb_halos.hdf5'
    f = h5py.File(file,'r')
    halos_1 = f['mpb_halos'][:99 - snap_1]
    halos_2 = f['mpb_halos'][:99 - snap_2]
    numHalos = halos_1.shape[0]

    cr_1, M_hot_1, halo_R_200c_1, t_dyn_1, HaloFlag_1 = cooling_radius(basePath, snap_1)
    cr_2, M_hot_2, halo_R_200c_2, t_dyn_2, HaloFlag_2 = cooling_radius(basePath, snap_1)

    rates = np.full(numHalos, -1, dtype=np.int64)

    for i, (halo_1, halo_2) in enumerate(zip(halos_1, halos_2)):
        if HaloFlag_1[halo_1] ==0 or HaloFlag_2[halo_2]==0:
            continue

        avarage_cr = (cr_1[halo_1] + cr_2[halo_2]) / 2
        avarage_R_200c = (halo_R_200c_1[halo_1] + halo_R_200c_2[halo_2]) / 2
        avarage_M_hot = (M_hot_1[halo_1] + M_hot_2[halo_2]) / 2
        avarage_t_dyn = (t_dyn_1[halo_1] + t_dyn_2[halo_2]) / 2

        frac = np.minimum(avarage_cr / avarage_R_200c, 1)
        rates[i] = avarage_M_hot * frac / avarage_t_dyn

    return {"halos_1": halos_1, "halos_2": halos_2, "rate": rates}