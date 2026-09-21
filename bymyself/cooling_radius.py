import time
import illustris_python as il
import numba as nb
from numba import jit, njit
import numpy as np
import h5py
import sys
import funcs
import illustrisFuncs as iF
from pathlib import Path

def CoolingTableLoader(table_dir = None):
    if table_dir is None:
        table_dir = Path(__file__).resolve().parent / "cooling_tables"
    else:
        table_dir = Path(table_dir)

    filenames = [
        "stripped_mzero.cie",
        "stripped_m-30.cie",
        "stripped_m-20.cie",
        "stripped_m-15.cie",
        "stripped_m-10.cie",
        "stripped_m-05.cie",
        "stripped_m-00.cie",
        "stripped_m+05.cie",
    ]

    log10_z_over_zsun = np.array([-5.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5], dtype = np.float32)
    log10_temperature = None
    log10_lambda_rows = []

    for i in filenames:
        path = table_dir / i
        table = np.loadtxt(path, dtype = np.float32)

        log10_temperature = table[:, 0]
        log10_lambda_rows.append(table[:, 5])

    z_sun = 0.02

    return {"log10_temperature": log10_temperature,
            "log10_metallicity": (log10_z_over_zsun + np.log10(z_sun)),
            "log10_lambda": np.asarray(log10_lambda_rows)}

def LgalCoolingFunction(T_hot, Z_hot, cooling_tables):
    log10_temperature_grid = cooling_tables["log10_temperature"]
    log10_metallicity_grid = cooling_tables["log10_metallicity"]
    log10_lambda_grid = cooling_tables["log10_lambda"]

    temperature, metallicity = np.broadcast_arrays(np.asarray(T_hot, dtype=np.float32), np.asarray(Z_hot, dtype=np.float32))

    with np.errstate(divide="ignore", invalid="ignore"):
        log10_temperature = np.where(np.isfinite(temperature) & (temperature > 0), np.log10(temperature), log10_temperature_grid[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        log10_metallicity = np.where(np.isfinite(metallicity) & (metallicity > 0), np.log10(metallicity), log10_metallicity_grid[0])

    log10_temperature = np.clip(log10_temperature, log10_temperature_grid[0], log10_temperature_grid[-1])
    log10_metallicity = np.clip(log10_metallicity, log10_metallicity_grid[0], log10_metallicity_grid[-1])

    flat_log10_temperature = log10_temperature.ravel()
    flat_log10_metallicity = log10_metallicity.ravel()

    # interpolation
    lambda_at_temperature = np.vstack([np.interp(flat_log10_temperature, log10_temperature_grid, row) for row in log10_lambda_grid])
    upper = np.searchsorted(log10_metallicity_grid, flat_log10_metallicity, side="right")
    upper = np.clip(upper, 1, len(log10_metallicity_grid) - 1)
    lower = upper - 1

    column = np.arange(flat_log10_metallicity.size)

    fraction = (flat_log10_metallicity - log10_metallicity_grid[lower]) / (log10_metallicity_grid[upper] - log10_metallicity_grid[lower])

    log10_lambda = (lambda_at_temperature[lower, column] + fraction * (lambda_at_temperature[upper, column] - lambda_at_temperature[lower, column]))

    result = np.power(10.0, log10_lambda)
    result = result.reshape(temperature.shape)

    return float(result) if result.ndim == 0 else result

@njit(parallel = True)
def compute_halo_properties(gas_masses, gas_pos, gas_mental, halo_pos, halo_R_200c, halo_M_200c, gasInHaloOffset, numGasInHalo, gas_state, gas_temp, boxSize, a, h):
    numHalo = halo_pos.shape[0]

    G = 4.30091e-6 # G: kpc (km/s)^2 / Msun

    M_hot = np.zeros(numHalo)
    T_hot = np.full(numHalo, np.nan)
    Z_hot = np.full(numHalo, np.nan)
    t_dyn = np.full(numHalo, np.nan)
    T_200c = np.full(numHalo, np.nan)
    V_200c = np.full(numHalo, np.nan)
    R200c_cm = np.full(numHalo, np.nan)
    HaloFlag = np.ones(numHalo, dtype=np.ubyte)

    kpc_cm = 3.085677581491367e21
    km_cm = 1.0e5
    msun_g = 1.98847e33

    for i in nb.prange(numHalo):
        indices_of_halo = np.arange(gasInHaloOffset[i], gasInHaloOffset[i] + numGasInHalo[i])

        #only choose hot gas cells
        indices_of_halo = indices_of_halo[gas_state[indices_of_halo] == 0]

        if indices_of_halo.shape[0] == 0:
            HaloFlag[i] = 0
            continue

        halo_gas_pos = gas_pos[indices_of_halo]
        halo_gas_dist = funcs.dist_vector_nb(halo_pos[i], halo_gas_pos, boxSize)

        #only choose gas cells within R200c
        indices_of_halo = indices_of_halo[np.where(halo_gas_dist <= halo_R_200c[i])[0]]

        if indices_of_halo.size == 0:
            HaloFlag[i] = 0
            continue

        R200c_cm[i] = halo_R_200c[i] * a / h * kpc_cm
        M_hot[i] = np.sum(gas_masses[indices_of_halo]) * 1.0e10 * msun_g / h
        T_hot[i] = np.sum(gas_temp[indices_of_halo] * gas_masses[indices_of_halo]) / np.sum(gas_masses[indices_of_halo])
        Z_hot[i] = np.sum(gas_mental[indices_of_halo] * gas_masses[indices_of_halo]) / np.sum(gas_masses[indices_of_halo])
        V_200c[i] = np.sqrt((G * 1.0e10 * halo_M_200c[i]) / (a * halo_R_200c[i]))
        t_dyn[i] = R200c_cm[i] / (V_200c[i] * km_cm)
        T_200c[i] = 35.9 * V_200c[i] ** 2

    return T_hot, Z_hot, M_hot, t_dyn, T_200c, R200c_cm, HaloFlag

def compute_lamda(gas_masses, gas_pos, gas_mental, halo_pos, halo_R_200c, halo_M_200c, gasInHaloOffset, numGasInHalo, gas_state, gas_temp, boxSize, a, h):
    cooling_tables = CoolingTableLoader()

    T_hot, Z_hot, _,  _, _, _, _ = compute_halo_properties(gas_masses, gas_pos, gas_mental, halo_pos, halo_R_200c, halo_M_200c, gasInHaloOffset, numGasInHalo, gas_state, gas_temp, boxSize, a, h)
    lamda = LgalCoolingFunction(T_hot, Z_hot, cooling_tables)

    return lamda

@njit(parallel=True)
def compute_cooling_radius(M_hot, t_dyn, lamda, T_200c, R200c_cm, halo_R_200c, HaloFlag, h, a):
    n_halo = M_hot.shape[0]
    cr = np.full(n_halo, np.nan)

    mu = 0.59
    m_H = 1.67262192369e-24
    k_B = 1.380649e-16
    kpc_cm = 3.085677581491367e21

    for i in nb.prange(n_halo):
        if HaloFlag[i] == 0:
            continue

        numerator = (t_dyn[i] * M_hot[i] * lamda[i])
        denominator = (6.0 * np.pi * mu * m_H * k_B * T_200c[i] * R200c_cm[i])

        r_cool_cm = np.sqrt(numerator / denominator)
        cr[i] = r_cool_cm / kpc_cm * h / a / halo_R_200c[i]

    return cr

def cooling_radius(basePath, snap):
    start = time.time()

    header = il.groupcat.loadHeader(basePath, snap)
    boxSize = header['BoxSize']
    a = header["Time"]
    h = header["HubbleParam"]

    gas = il.snapshot.loadSubset(basePath, snap, 'gas', fields = ["Coordinates", "Masses", "StarFormationRate", "InternalEnergy", "ElectronAbundance", "GFM_Metallicity"], float32=True)
    gas_pos = gas['Coordinates'][:,:]
    gas_masses = gas['Masses'][:]
    gas_utherm = gas["InternalEnergy"][:]
    gas_nelec = gas["ElectronAbundance"][:]
    gas_mental = gas["GFM_Metallicity"]
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

    halos = il.groupcat.loadHalos(basePath, snap, fields = ["GroupLenType", "GroupPos", "Group_R_Crit200", "Group_M_Crit200"])
    numGasInHalo = halos["GroupLenType"][:, 0]
    halo_pos = halos["GroupPos"]
    halo_R_200c = halos["Group_R_Crit200"]
    halo_M_200c = halos["Group_M_Crit200"]

    gasInHaloOffset = np.concatenate(([0], np.cumsum(numGasInHalo[:-1], dtype=np.int64)))

    end_loading = time.time()
    print('Loading took ',np.round(end_loading - start,3),' seconds.')

    lamda = compute_lamda(gas_masses, gas_pos, gas_mental, halo_pos, halo_R_200c, halo_M_200c, gasInHaloOffset, numGasInHalo, gas_state, gas_temp, boxSize, a, h)
    _, _, M_hot, t_dyn, T_200c, R200c_cm, HaloFlag = compute_halo_properties(gas_masses, gas_pos, gas_mental, halo_pos, halo_R_200c, halo_M_200c, gasInHaloOffset, numGasInHalo, gas_state, gas_temp, boxSize, a, h)
    cr = compute_cooling_radius(M_hot, t_dyn, lamda, T_200c, R200c_cm, halo_R_200c, HaloFlag, h, a)

    end_calc = time.time()
    print('Computing took ',np.round(end_calc - end_loading,3),' seconds.')

    result_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")
    result_dir.mkdir(parents=True, exist_ok=True)
    output_file = result_dir / f"cooling_radius_{snap}.hdf5"
    with h5py.File(output_file, "w") as f:
        f.create_dataset("cooling_radius", data=np.asarray(cr, dtype=np.float32), compression="gzip", compression_opts=4)
        f.create_dataset("halo_flag",data=np.asarray(HaloFlag, dtype=np.ubyte))

    result = (cr, M_hot, halo_R_200c, t_dyn, HaloFlag)

    # Release only large arrays that are no longer needed.  Arrays included
    # in ``result`` are intentionally kept alive by the returned tuple.
    del halos
    del gas_pos, gas_masses, gas_mental
    del gas_state, gas_temp
    del halo_pos, halo_M_200c
    del numGasInHalo, gasInHaloOffset
    del T_hot, Z_hot, lamda, T_200c, R200c_cm

    return result

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python cooling_radius.py SNAP")

    snap = int(sys.argv[1])
    basePath = "/public/share/chenhouzun/TNG50-1/output/"
    cooling_radius(basePath, snap)
