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
def calculate_ssfr_quantities(sfr, stellar_mass, halo_id, subhalo_flag):
    n = sfr.shape[0]

    log_mstar = np.full(n, np.nan)
    log_ssfr = np.full(n, np.nan)

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
            log_ssfr[i] = np.log10(sfr[i]) - np.log10(stellar_mass[i])
            fit_valid[i] = True
        else:
            # 真实的 SFR=0 星系可视为低于 SFMS 无限远
            log_ssfr[i] = -np.inf

    return log_mstar, log_ssfr, valid, fit_valid

def calculate_bin_medians(log_mstar, log_ssfr, active, mass_min=7.0, mass_max=10.2, bin_width=0.2, min_count=5):
    num_bins = int(np.ceil((mass_max - mass_min) / bin_width))
    bin_mass = np.full(num_bins, np.nan)
    bin_ssfr = np.full(num_bins, np.nan)
    bin_count = np.zeros(num_bins, dtype=np.int64)

    bin_index = np.floor((log_mstar - mass_min) / bin_width).astype(np.int64)

    for i in range(num_bins):
        mask = active & (bin_index == i) & np.isfinite(log_ssfr)

        if np.count_nonzero(mask) < min_count:
            continue

        bin_mass[i] = np.median(log_mstar[mask])
        bin_ssfr[i] = np.median(log_ssfr[mask])
        bin_count[i] = np.count_nonzero(mask)

    return bin_mass, bin_ssfr, bin_count, bin_index

@njit(parallel=True)
def remove_low_ssfr(log_ssfr, active, bin_ssfr, bin_index, lower_offset=0.5):
    new_active = active.copy()

    for i in nb.prange(log_ssfr.shape[0]):
        if not active[i]:
            continue

        b = bin_index[i]

        if b < 0 or b >= bin_ssfr.shape[0]:
            continue

        if log_ssfr[i] < bin_ssfr[b] - lower_offset:
            new_active[i] = False

    return new_active

def fit_sfms(log_mstar, log_ssfr, max_iter=20):
    active = np.isfinite(log_mstar) & np.isfinite(log_ssfr) & (log_mstar >= 7.0) & (log_mstar < 10.2)

    for _ in range(max_iter):
        bin_mass, bin_ssfr, bin_count, bin_index = calculate_bin_medians(log_mstar, log_ssfr, active)
        new_active = remove_low_ssfr(log_ssfr, active, bin_ssfr, bin_index, lower_offset=0.5)

        if np.array_equal(new_active, active):
            break

        active = new_active

    bin_mass, bin_ssfr, bin_count, bin_index = calculate_bin_medians(log_mstar, log_ssfr, active)

    valid_bins = np.isfinite(bin_mass) & np.isfinite(bin_ssfr) & (bin_count > 0)

    if np.count_nonzero(valid_bins) < 2:
        raise ValueError("Not enough valid bins for SFMS fitting.")

    alpha, beta = np.polyfit(bin_mass[valid_bins], bin_ssfr[valid_bins], 1)

    return {"alpha": alpha, "beta": beta, "log_mstar_bin": bin_mass, "log_ssfr_bin": bin_ssfr, "bin_count": bin_count, "active_fit": active}

def evaluate_sfms_reference(log_mstar, bin_ssfr, alpha, beta, mass_min=7.0, mass_max=10.2, bin_width=0.2):
    log_ssfr_sfms = np.full(log_mstar.shape, np.nan, dtype=np.float64)

    bin_index = np.floor((log_mstar - mass_min) / bin_width).astype(np.int64)
    in_range = np.isfinite(log_mstar) & (log_mstar >= mass_min) & (log_mstar <= mass_max) & (bin_index >= 0) & (bin_index < bin_ssfr.size)
    valid = np.zeros(log_mstar.shape, dtype=bool)
    valid[in_range] = np.isfinite(bin_ssfr[bin_index[in_range]])
    log_ssfr_sfms[valid] = bin_ssfr[bin_index[valid]]

    high_mass = np.isfinite(log_mstar) & (log_mstar > mass_max)
    log_ssfr_sfms[high_mass] = alpha * log_mstar[high_mass] + beta

    return log_ssfr_sfms

@njit(parallel=True)
def classify_central_halos(group_first_sub, log_ssfr, log_ssfr_sfms, valid_subhalo):
    num_halos = group_first_sub.shape[0]

    class_code = np.full(num_halos, -1, dtype=np.int8)
    central_subhalo_id = np.full(num_halos, -1, dtype=np.int64)
    delta_log_ssfr = np.full(num_halos, np.nan)

    for halo in nb.prange(num_halos):
        subhalo = group_first_sub[halo]

        if subhalo < 0 or subhalo >= valid_subhalo.shape[0]:
            continue
        if not valid_subhalo[subhalo]:
            continue

        central_subhalo_id[halo] = subhalo

        if not np.isfinite(log_ssfr_sfms[subhalo]):
            continue

        if log_ssfr[subhalo] == -np.inf:
            delta_log_ssfr[halo] = -np.inf
            class_code[halo] = 3
            continue

        if not np.isfinite(log_ssfr_sfms[subhalo]):
            continue

        delta = log_ssfr[subhalo] - log_ssfr_sfms[subhalo]
        delta_log_ssfr[halo] = delta

        if delta > -0.5:
            class_code[halo] = 1 # Star Forming
        elif delta > -1.0:
            class_code[halo] = 2 # Green Valley
        else:
            class_code[halo] = 3 # Quenched

    return central_subhalo_id, delta_log_ssfr, class_code

def ssfr_classification(basePath, snap):
    start = time.time()
    data = load_sfr_data(basePath, snap)
    end_loading = time.time()
    print('Loading took ',np.round(end_loading - start,3),' seconds.')

    log_mstar, log_ssfr, valid_subhalo, fit_valid = calculate_ssfr_quantities(data["sfr"], data["stellar_mass"], data["halo_id"], data["subhalo_flag"])
    sfms = fit_sfms(log_mstar[fit_valid], log_ssfr[fit_valid])
    log_ssfr_sfms = evaluate_sfms_reference(log_mstar, sfms["log_ssfr_bin"], sfms["alpha"], sfms["beta"])
    central_subhalo_id, delta_log_ssfr, class_code = classify_central_halos(data["group_first_sub"], log_ssfr, log_ssfr_sfms, valid_subhalo)
    end_calc = time.time()
    print('Computing took ',np.round(end_calc - end_loading,3),' seconds.')

    result_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")
    result_dir.mkdir(parents=True, exist_ok=True)
    output_file = result_dir / f"ssfr_classification_{snap}.hdf5"
    with h5py.File(output_file, "w") as f:
        f.create_dataset("halo_id", data=np.arange(class_code.size))
        f.create_dataset("central_subhalo_id", data=central_subhalo_id)
        f.create_dataset("delta_log_ssfr", data=delta_log_ssfr)
        f.create_dataset("class_code", data=class_code)

    return class_code

if __name__ == "__main__":
    snap = int(sys.argv[1])
    basePath = "/public/share/chenhouzun/TNG50-1/output/"
    ssfr_classification(basePath, snap)
