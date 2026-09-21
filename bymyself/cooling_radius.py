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

    log10_z_over_zsun = np.array([-5.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5], dtype = np.float64)
    log10_temperature = None
    log10_lambda_rows = []

    for i in filenames:
        path = table_dir / i
        table = np.loadtxt(path, dtype = np.float64)

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

    temperature, metallicity = np.broadcast_arrays(np.asarray(T_hot, dtype=np.float64), np.asarray(Z_hot, dtype=np.float64))

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

def iter_gas_chunks(basePath, snap):
    snap_dir = Path(basePath) / f"snapdir_{snap:03d}"
    files = sorted(snap_dir.glob(f"snap_{snap:03d}.*.hdf5"), key=lambda p: int(p.stem.split(".")[-1]))

    global_start = 0

    fields = ["Coordinates", "Masses", "StarFormationRate", "InternalEnergy", "ElectronAbundance", "GFM_Metallicity"]

    for filename in files:
        with h5py.File(filename, "r") as f:
            n_gas = int(f["Header"].attrs["NumPart_ThisFile"][0])
            part = f["PartType0"]

            chunk = {field: part[field].astype("f4")[:] for field in fields}

        yield global_start, n_gas, chunk
        global_start += n_gas

@njit
def accumulate_gas_chunk(chunk_start, gas_pos, gas_mass, gas_sfr, gas_temp, gas_metal, gas_offset, gas_end, halo_pos, halo_R_200c, boxSize, mass_factor, M_hot, MT_hot, MZ_hot):
    chunk_end = chunk_start + gas_mass.shape[0]
    num_halo = gas_offset.shape[0]
    temp_limit = 10.0 ** 4.5

    halo = np.searchsorted(gas_end, chunk_start, side="right")

    while halo < num_halo and gas_offset[halo] < chunk_end:
        start = max(gas_offset[halo], chunk_start)
        end = min(gas_end[halo], chunk_end)

        if end > start and halo_R_200c[halo] > 0:
            local_start = start - chunk_start
            local_end = end - chunk_start

            halo_gas_pos = gas_pos[local_start:local_end]
            halo_gas_dist = funcs.dist_vector_nb(halo_pos[halo], halo_gas_pos, boxSize)

            for local_index in range(halo_gas_dist.shape[0]):
                j = local_start + local_index

                if gas_sfr[j] != 0.0:
                    continue

                if not np.isfinite(gas_temp[j]):
                    continue

                if not np.isfinite(gas_metal[j]):
                    continue

                if gas_temp[j] <= temp_limit:
                    continue

                mass_g = gas_mass[j] * mass_factor

                M_hot[halo] += mass_g
                MT_hot[halo] += mass_g * gas_temp[j]
                MZ_hot[halo] += mass_g * gas_metal[j]

        halo += 1

def cooling_radius(basePath, snap):
    start = time.time()

    header = il.groupcat.loadHeader(basePath, snap)
    boxSize = header['BoxSize']
    a = header["Time"]
    h = header["HubbleParam"]

    halos = il.groupcat.loadHalos(basePath, snap, fields=["GroupLenType", "GroupPos", "Group_R_Crit200", "Group_M_Crit200"])
    numGasInHalo = np.asarray(halos["GroupLenType"][:, 0], dtype=np.int64)
    halo_pos = np.asarray(halos["GroupPos"], dtype=np.float32)
    halo_R_200c = np.asarray(halos["Group_R_Crit200"], dtype=np.float32)
    halo_M_200c = np.asarray(halos["Group_M_Crit200"], dtype=np.float32)

    gas_offset = np.concatenate(([0], np.cumsum(numGasInHalo[:-1], dtype=np.int64)))
    gas_end = gas_offset + numGasInHalo
    num_halo = numGasInHalo.shape[0]

    M_hot = np.zeros(num_halo, dtype=np.float64)
    MT_hot = np.zeros(num_halo, dtype=np.float64)
    MZ_hot = np.zeros(num_halo, dtype=np.float64)

    msun_g = 1.98847e33
    mass_factor = 1.0e10 * msun_g / h

    global_start = 0    

    end_loading_1 = time.time()
    print('Loading halos took ',np.round(end_loading_1 - start,3),' seconds.')
    print("Start reading gas chunks")
    for chunk_start, n_gas, chunk in iter_gas_chunks(basePath, snap):
        gas_pos = chunk["Coordinates"]
        gas_mass = chunk["Masses"]
        gas_sfr = chunk["StarFormationRate"]
        gas_utherm = chunk["InternalEnergy"]
        gas_nelec = chunk["ElectronAbundance"]
        gas_metal = chunk["GFM_Metallicity"]

        non_sf = gas_sfr == 0.0

        gas_temp = np.full(n_gas, np.nan, dtype=np.float32)        
        gas_temp[non_sf] = iF.utherm_to_temp(gas_utherm[non_sf], gas_nelec[non_sf])

        accumulate_gas_chunk(chunk_start, gas_pos, gas_mass, gas_sfr, gas_temp, gas_metal, gas_offset, gas_end, halo_pos, halo_R_200c, boxSize, mass_factor, M_hot, MT_hot, MZ_hot)

        global_start += n_gas

        del chunk, gas_pos, gas_mass, gas_sfr, gas_utherm, gas_nelec, gas_metal, gas_temp, non_sf

    end_loading_2 = time.time()
    print('Reading gas chunks took ',np.round(end_loading_2 - end_loading_1,3),' seconds.')

    expected_gas = int(numGasInHalo.sum())
    if global_start != expected_gas:
        raise ValueError(f"Gas count mismatch: chunks={global_start}, "f"halo catalog={expected_gas}")

    valid = (M_hot > 0) & np.isfinite(halo_R_200c) & (halo_R_200c > 0) & np.isfinite(halo_M_200c) & (halo_M_200c > 0)

    T_hot = np.full(num_halo, np.nan)
    Z_hot = np.full(num_halo, np.nan)

    T_hot[valid] = MT_hot[valid] / M_hot[valid]
    Z_hot[valid] = MZ_hot[valid] / M_hot[valid]      

    HaloFlag = valid.astype(np.ubyte)

    G = 4.30091e-6
    kpc_cm = 3.085677581491367e21
    km_cm = 1.0e5

    R200c_cm = np.full(num_halo, np.nan)
    V_200c = np.full(num_halo, np.nan)
    T_200c = np.full(num_halo, np.nan)
    t_dyn = np.full(num_halo, np.nan)

    R200c_cm[valid] = halo_R_200c[valid] * a / h * kpc_cm
    V_200c[valid] = np.sqrt(G * 1.0e10 * halo_M_200c[valid] / (a * halo_R_200c[valid]))
    T_200c[valid] = 35.9 * V_200c[valid] ** 2
    t_dyn[valid] = R200c_cm[valid] / (V_200c[valid] * km_cm)

    cooling_tables = CoolingTableLoader()

    end_loading_3 = time.time()
    print('Loading took ',np.round(end_loading_3 - start,3),' seconds.')

    lamda = LgalCoolingFunction(T_hot, Z_hot, cooling_tables)
    cr = compute_cooling_radius(M_hot, t_dyn, lamda, T_200c, R200c_cm, halo_R_200c, HaloFlag, h, a)

    end_calc = time.time()
    print('Computing took ',np.round(end_calc - end_loading_3,3),' seconds.')

    result_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")
    result_dir.mkdir(parents=True, exist_ok=True)
    output_file = result_dir / f"cooling_radius_{snap}.hdf5"
    with h5py.File(output_file, "w") as f:
        f.create_dataset("cooling_radius", data=np.asarray(cr, dtype=np.float32), compression="gzip", compression_opts=4)
        f.create_dataset("halo_flag",data=np.asarray(HaloFlag, dtype=np.ubyte))

    result = (cr, M_hot, halo_R_200c, t_dyn, HaloFlag)
    return result

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python cooling_radius.py SNAP")

    snap = int(sys.argv[1])
    basePath = "/public/share/chenhouzun/TNG50-1/output/"
    cooling_radius(basePath, snap)
