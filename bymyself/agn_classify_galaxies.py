import time
import sys
import illustris_python as il
import numpy as np
import h5py
from pathlib import Path
import numba as nb
from numba import njit

def load_agn_data(basePath, snap):
    header = il.groupcat.loadHeader(basePath, snap)
    h = header["HubbleParam"]

    subhalos = il.groupcat.loadSubhalos(basePath, snap, fields=["SubhaloGrNr", "SubhaloFlag", "SubhaloBHMass", "SubhaloBHMdot"])
    halos = il.groupcat.loadHalos(basePath, snap, fields=["GroupFirstSub"])

    halo_id = np.asarray(subhalos["SubhaloGrNr"], dtype=np.int64)
    subhalo_flag = np.asarray(subhalos["SubhaloFlag"], dtype=bool)
    bh_mass = np.asarray(subhalos["SubhaloBHMass"], dtype=np.float64)
    bh_mdot = np.asarray(subhalos["SubhaloBHMdot"], dtype=np.float64)
    group_first_sub = np.asarray(halos["GroupFirstSub"], dtype=np.int64)

    return {"halo_id": halo_id, "subhalo_flag": subhalo_flag, "bh_mass": bh_mass, "bh_mdot": bh_mdot, "group_first_sub": group_first_sub, "h": h}

@njit(parallel=True)
def calculate_fedd_quantities(bh_mass, bh_mdot, halo_id, subhalo_flag, h):
    n = bh_mass.shape[0]
    fedd = np.full(n, np.nan)
    log10_fedd = np.full(n, np.nan)
    valid = np.zeros(n, dtype=np.bool_)

    msun_g = 1.98847e33
    year_s = 365.25 * 24.0 * 3600.0
    c_cgs = 2.99792458e10
    re=0.1

    for i in nb.prange(n):
        if not subhalo_flag[i]:
            continue

        if halo_id[i] < 0:
            continue

        if not np.isfinite(bh_mass[i]) or bh_mass[i] <= 0:
            continue

        if not np.isfinite(bh_mdot[i]) or bh_mdot[i] < 0:
            continue

        # SubhaloBHMass: 1e10 Msun/h
        bh_mass_msun = bh_mass[i] * 1.0e10 / h
        bh_mdot_msun_per_year = bh_mdot[i] * 1.0e10 / 0.978e9
        bh_mdot_g_s = bh_mdot_msun_per_year * msun_g / year_s
        l_bol = re * bh_mdot_g_s * c_cgs**2
        l_edd = 1.26e38 * bh_mass_msun

        fedd[i] = l_bol / l_edd
        valid[i] = True

        if fedd[i] > 0:
            log10_fedd[i] = np.log10(fedd[i])
        else:
            log10_fedd[i] = -np.inf

    return fedd, log10_fedd, valid

@njit(parallel=True)
def classify_central_halos_fedd(group_first_sub, fedd, log10_fedd, valid_subhalo):
    num_halos = group_first_sub.shape[0]
    class_code = np.full(num_halos, -1, dtype=np.int8)
    central_subhalo_id = np.full(num_halos, -1, dtype=np.int64)
    central_fedd = np.full(num_halos, np.nan)
    central_log10_fedd = np.full(num_halos, np.nan)

    for halo in nb.prange(num_halos):
        subhalo = group_first_sub[halo]

        if subhalo < 0 or subhalo >= valid_subhalo.shape[0]:
            continue

        if not valid_subhalo[subhalo]:
            continue

        central_subhalo_id[halo] = subhalo
        central_fedd[halo] = fedd[subhalo]
        central_log10_fedd[halo] = log10_fedd[subhalo]
        log_lambda = log10_fedd[subhalo]

        if log_lambda < -4.0:
            class_code[halo] = 0
        elif log_lambda < -3.0:
            class_code[halo] = 1
        elif log_lambda < -2.0:
            class_code[halo] = 2
        elif log_lambda < -1.0:
            class_code[halo] = 3
        else:
            class_code[halo] = 4

    return central_subhalo_id, central_fedd, central_log10_fedd, class_code

def fedd_classification(basePath, snap):
    start = time.time()
    data = load_agn_data(basePath, snap)
    end_loading = time.time()
    print('Loading took ',np.round(end_loading - start,3),' seconds.')

    fedd, log10_fedd, valid_subhalo = calculate_fedd_quantities(data["bh_mass_code"], data["bh_mdot_code"], data["halo_id"], data["subhalo_flag"], data["h"])
    central_subhalo_id, central_fedd, central_log10_fedd, class_code = classify_central_halos_fedd(data["group_first_sub"], fedd, log10_fedd, valid_subhalo)
    end_calc = time.time()
    print('Computing took ',np.round(end_calc - end_loading,3),' seconds.')

    result_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")
    result_dir.mkdir(parents=True, exist_ok=True)

    output_file = result_dir / f"fedd_classification_{snap}.hdf5"

    with h5py.File(output_file, "w") as f:
        f.create_dataset("halo_id", data=np.arange(class_code.size))
        f.create_dataset("central_subhalo_id", data=central_subhalo_id)
        f.create_dataset("central_fedd", data=central_fedd)
        f.create_dataset("central_log10_fedd", data=central_log10_fedd)
        f.create_dataset("class_code", data=class_code)
        f.create_dataset("valid_classification", data=(class_code >= 0))
        f.attrs["class_0"] = "fEdd < 1e-4"
        f.attrs["class_1"] = "1e-4 <= fEdd < 1e-3"
        f.attrs["class_2"] = "1e-3 <= fEdd < 1e-2"
        f.attrs["class_3"] = "1e-2 <= fEdd < 1e-1"
        f.attrs["class_4"] = "fEdd >= 1e-1"

    return class_code

if __name__ == "__main__":
    snap = int(sys.argv[1])
    basePath = "/public/share/chenhouzun/TNG50-1/output/"
    fedd_classification(basePath, snap)
