import cooling_radius as cr
import mpb
import h5py
import numpy as np
from os.path import isfile

def compute_Lgal_cooling_rate(basePath, snap_1, snap_2):

    assert isfile(f'/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/mpb_halos.hdf5'), 'Mpb halos file does not exist!'
    file = f'/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/mpb_halos.hdf5'
    f = h5py.File(file,'r')
    halos_1 = f['mpb_halos'][:, 99 - snap_1]
    halos_2 = f['mpb_halos'][:, 99 - snap_2]
    numHalos = halos_1.shape[0]

    cr_1, M_hot_1, _, t_dyn_1, HaloFlag_1 = cr.cooling_radius(basePath, snap_1)
    cr_2, M_hot_2, _, t_dyn_2, HaloFlag_2 = cr.cooling_radius(basePath, snap_2)

    rates = np.full(numHalos, np.nan, dtype=np.float64)
    valid = np.zeros(numHalos, dtype=bool)

    for i, (halo_1, halo_2) in enumerate(zip(halos_1, halos_2)):
        if halo_1 < 0 or halo_2 < 0:
            continue

        if HaloFlag_1[halo_1] ==0 or HaloFlag_2[halo_2]==0:
            continue

        average_cr = (cr_1[halo_1] + cr_2[halo_2]) / 2
        average_M_hot = (M_hot_1[halo_1] + M_hot_2[halo_2]) / 2
        average_t_dyn = (t_dyn_1[halo_1] + t_dyn_2[halo_2]) / 2

        frac = np.clip(average_cr, 0.0, 1.0)
        rates[i] = average_M_hot * frac / average_t_dyn * 365.25 * 24 * 3600 / 1.98847e33 # M_sun/yr
        valid[i] = True

    return {"halos_1": halos_1, "halos_2": halos_2, "rate": rates, "valid": valid}