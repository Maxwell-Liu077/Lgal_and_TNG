import time
import sys
import illustris_python as il
import numpy as np
import h5py
from pathlib import Path
import numba as nb
from numba import njit

def load_sfr_data(basePath, snap):
    header = il.groupcat.loadHeader(basePath, snap)
    h = header["HubbleParam"]

    subhalos = il.groupcat.loadSubhalos(basePath, snap, fields=["SubhaloGrNr", "SubhaloFlag", "SubhaloSFR", "SubhaloMassType"])
    halos = il.groupcat.loadHalos(basePath,snap,fields=["GroupFirstSub"])

    halo_id = np.asarray(subhalos["SubhaloGrNr"], dtype=np.int64)
    subhalo_flag = np.asarray(subhalos["SubhaloFlag"], dtype=bool)

    sfr = np.asarray(subhalos["SubhaloSFR"], dtype=np.float64)
    stellar_mass = np.asarray(subhalos["SubhaloMassType"][:, 4], dtype=np.float64) * 1.0e10 / h

    group_first_sub = np.asarray(halos, dtype=np.int64)

    return {"halo_id": halo_id, "subhalo_flag": subhalo_flag, "sfr": sfr, "stellar_mass": stellar_mass, "group_first_sub": group_first_sub, "h": h}

@njit(parallel=True)
def calculate_sfr_quantities(sfr, stellar_mass, halo_id, subhalo_flag):
    n = sfr.shape[0]

    log_mstar = np.full(n, np.nan)
    log_sfr = np.full(n, np.nan)

    valid = np.zeros(n, dtype=np.bool_)
    fit_valid = np.zeros(n, dtype=np.bool_)

    for i in nb.prange(n):
        if not subhalo_flag[i]:
            continue

        if halo_id[i] < 0:
            continue

        if not np.isfinite(stellar_mass[i]) or stellar_mass[i] <= 0:
            continue

        if not np.isfinite(sfr[i]) or sfr[i] < 0:
            continue

        log_mstar[i] = np.log10(stellar_mass[i])
        valid[i] = True

        if sfr[i] > 0:
            log_sfr[i] = np.log10(sfr[i])
            fit_valid[i] = True
        else:
            # 真实的 SFR=0 星系可视为低于 SFMS 无限远
            log_sfr[i] = -np.inf

    return log_mstar, log_sfr, valid, fit_valid

def calculate_bin_medians(log_mstar, log_sfr, active, mass_min=9.0, mass_max=10.2, bin_width=0.2, min_count=5):
    num_bins = int(np.ceil((mass_max - mass_min) / bin_width))
    bin_mass = np.full(num_bins, np.nan)
    bin_sfr = np.full(num_bins, np.nan)
    bin_count = np.zeros(num_bins, dtype=np.int64)

    bin_index = np.floor((log_mstar - mass_min) / bin_width).astype(np.int64)

    for i in range(num_bins):
        mask = active & (bin_index == i) & np.isfinite(log_sfr)

        if np.count_nonzero(mask) < min_count:
            continue

        bin_mass[i] = np.median(log_mstar[mask])
        bin_sfr[i] = np.median(log_sfr[mask])
        bin_count[i] = np.count_nonzero(mask)

    return bin_mass, bin_sfr, bin_count, bin_index

@njit(parallel=True)
def remove_low_sfr(log_sfr, active, bin_sfr, bin_index, lower_offset=1.0):
    new_active = active.copy()

    for i in nb.prange(log_sfr.shape[0]):
        if not active[i]:
            continue

        b = bin_index[i]

        if b < 0 or b >= bin_sfr.shape[0]:
            continue

        if log_sfr[i] < bin_sfr[b] - lower_offset:
            new_active[i] = False

    return new_active

def fit_sfms(log_mstar, log_sfr, max_iter=20):
    active = np.isfinite(log_mstar) & np.isfinite(log_sfr) & (log_mstar >= 9.0) & (log_mstar < 10.2)

    for _ in range(max_iter):
        bin_mass, bin_sfr, bin_count, bin_index = calculate_bin_medians(log_mstar, log_sfr, active)
        new_active = remove_low_sfr(log_sfr, active, bin_sfr, bin_index, lower_offset=1.0)

        if np.array_equal(new_active, active):
            break

        active = new_active

    bin_mass, bin_sfr, bin_count, bin_index = calculate_bin_medians(log_mstar, log_sfr, active)

    valid_bins = np.isfinite(bin_mass) & np.isfinite(bin_sfr) & (bin_count > 0)

    if np.count_nonzero(valid_bins) < 2:
        raise ValueError("Not enough valid bins for SFMS fitting.")

    alpha, beta = np.polyfit(bin_mass[valid_bins], bin_sfr[valid_bins], 1)
    log_sfr_sfms = alpha * log_mstar + beta
    delta_log_sfr = log_sfr - log_sfr_sfms

    return {"alpha": alpha, "beta": beta, "log_mstar_bin": bin_mass, "log_sfr_bin": bin_sfr, "bin_count": bin_count, "active_fit": active, "log_sfr_sfms": log_sfr_sfms, "delta_log_sfr": delta_log_sfr}

@njit(parallel=True)
def classify_central_halos(group_first_sub, log_mstar, log_sfr, valid_subhalo, alpha, beta):
    num_halos = group_first_sub.shape[0]

    class_code = np.full(num_halos, -1, dtype=np.int8)
    central_subhalo_id = np.full(num_halos, -1, dtype=np.int64)
    delta_log_sfr = np.full(num_halos, np.nan)

    for halo in nb.prange(num_halos):
        subhalo = group_first_sub[halo]

        if subhalo < 0 or subhalo >= valid_subhalo.shape[0]:
            continue
        if not valid_subhalo[subhalo]:
            continue

        central_subhalo_id[halo] = subhalo

        if log_sfr[subhalo] == -np.inf:
            delta_log_sfr[halo] = -np.inf
            class_code[halo] = 3
            continue

        log_sfr_ms = alpha * log_mstar[subhalo] + beta
        delta = log_sfr[subhalo] - log_sfr_ms
        delta_log_sfr[halo] = delta

        if delta > 0.3:
            class_code[halo] = 0 # Starburst
        elif delta >= -0.3:
            class_code[halo] = 1 # Main Sequence
        elif delta >= -1.0:
            class_code[halo] = 2 # Transition
        else:
            class_code[halo] = 3 # Quenched

    return central_subhalo_id, delta_log_sfr, class_code

def ssfr_classification(basePath, snap):
    start = time.time()
    data = load_sfr_data(basePath, snap)
    end_loading = time.time()
    print('Loading took ',np.round(end_loading - start,3),' seconds.')

    log_mstar, log_sfr, valid_subhalo, fit_valid = calculate_sfr_quantities(data["sfr"], data["stellar_mass"], data["halo_id"], data["subhalo_flag"])
    sfms = fit_sfms(log_mstar[fit_valid], log_sfr[fit_valid])
    central_subhalo_id, delta_log_sfr, class_code = classify_central_halos(data["group_first_sub"], log_mstar, log_sfr, valid_subhalo, sfms["alpha"], sfms["beta"])
    end_calc = time.time()
    print('Computing took ',np.round(end_calc - end_loading,3),' seconds.')

    result_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")
    result_dir.mkdir(parents=True, exist_ok=True)
    output_file = result_dir / f"ssfr_classification_{snap}.hdf5"
    with h5py.File(output_file, "w") as f:
        f.create_dataset("halo_id", data=np.arange(class_code.size))
        f.create_dataset("central_subhalo_id", data=central_subhalo_id)
        f.create_dataset("delta_log_sfr", data=delta_log_sfr)
        f.create_dataset("class_code", data=class_code)

    return class_code

if __name__ == "__main__":
    snap = int(sys.argv[1])
    basePath = "/public/share/chenhouzun/TNG50-1/output/"
    ssfr_classification(basePath, snap)
