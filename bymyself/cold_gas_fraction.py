import sys
import time
from pathlib import Path
import h5py
import illustris_python as il
import numba as nb
import numpy as np
from numba import njit
import illustrisFuncs as iF
import funcs

@njit(parallel = True)
def compute_cold_gas_fraction(gas_masses, gas_pos, halo_pos, halo_R_200c, gasInHaloOffset, numGasInHalo, gas_state, boxSize):
    numHalo = halo_pos.shape[0]

    HaloFlag = np.ones(numHalo, dtype=np.ubyte)
    cold_gas_fraction = np.ones(numHalo, dtype=np.float32)

    for i in nb.prange(numHalo):
        indices_of_halo = np.arange(gasInHaloOffset[i], gasInHaloOffset[i] + numGasInHalo[i])

        if indices_of_halo.shape[0] == 0:
            HaloFlag[i] = 0
            continue

        halo_gas_pos = gas_pos[indices_of_halo]
        halo_gas_dist = funcs.dist_vector_nb(halo_pos[i], halo_gas_pos, boxSize)

        #only choose gas cells within R200c and without ISM
        radial_mask = (halo_gas_dist <= halo_R_200c[i]) & (halo_gas_dist >= 0.1 * halo_R_200c[i])
        radial_indices = indices_of_halo[radial_mask]

        if radial_indices.size == 0:
            HaloFlag[i] = 0
            continue

        #only choose cold gas cells
        cold_indices = radial_indices[gas_state[radial_indices] == 1]

        cold_gas_fraction[i] = np.sum(gas_masses[cold_indices]) / np.sum(gas_masses[radial_indices])

    return cold_gas_fraction, HaloFlag

def cold_gas_fraction(basePath, snap):
    start = time.time()

    header = il.groupcat.loadHeader(basePath, snap)
    boxSize = header['BoxSize']

    gas = il.snapshot.loadSubset(basePath, snap, 'gas', fields = ["Coordinates", "Masses", "StarFormationRate", "InternalEnergy", "ElectronAbundance"], float32=True)
    gas_pos = gas['Coordinates'][:,:]
    gas_masses = gas['Masses'][:]
    gas_utherm = gas["InternalEnergy"][:]
    gas_nelec = gas["ElectronAbundance"][:]
    gas_sfr = gas['StarFormationRate']

    del gas

    num_gas = gas_masses.shape[0]

    sf = np.nonzero(gas_sfr)[0]
    gas_state = np.zeros(num_gas, dtype = np.ubyte)
    gas_state[sf] = 1

    del sf, gas_sfr

    non_sf = gas_state == 0
    gas_temp = np.full(num_gas, np.nan, dtype = np.float32)
    gas_temp[non_sf] = iF.utherm_to_temp(gas_utherm[non_sf], gas_nelec[non_sf])

    del gas_utherm, gas_nelec, non_sf

    gas_log10_temp = np.log10(gas_temp)
    cold = np.where(gas_log10_temp <= 4.5)[0]
    gas_state[cold] = 1

    # These arrays are no longer needed after the gas-state classification.
    del cold, gas_log10_temp

    halos = il.groupcat.loadHalos(basePath, snap, fields = ["GroupLenType", "GroupPos", "Group_R_Crit200"])
    numGasInHalo = halos["GroupLenType"][:, 0]
    halo_pos = halos["GroupPos"]
    halo_R_200c = halos["Group_R_Crit200"]

    gasInHaloOffset = np.concatenate(([0], np.cumsum(numGasInHalo[:-1], dtype=np.int64)))

    end_loading = time.time()
    print('Loading took ',np.round(end_loading - start,3),' seconds.')

    cold_gas_fraction, HaloFlag = compute_cold_gas_fraction(gas_masses, gas_pos, halo_pos, halo_R_200c, gasInHaloOffset, numGasInHalo, gas_state, boxSize)

    end_calc = time.time()
    print('Computing took ', np.round(end_calc - end_loading,3),' seconds.')

    result_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")
    result_dir.mkdir(parents=True, exist_ok=True)
    output_file = result_dir / f"cold_gas_fraction_{snap}.hdf5"
    with h5py.File(output_file, "w") as f:
        f.create_dataset("cold_gas_fraction", data=np.asarray(cold_gas_fraction, dtype=np.float32), compression="gzip", compression_opts=4)
        f.create_dataset("halo_flag",data=np.asarray(HaloFlag, dtype=np.ubyte))

    del halos, gas_pos, gas_masses, gas_state, gas_temp, halo_pos, numGasInHalo, gasInHaloOffset

    return 0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python cooling_radius.py SNAP")

    snap = int(sys.argv[1])
    basePath = "/public/share/chenhouzun/TNG50-1/output/"
    cold_gas_fraction(basePath, snap)