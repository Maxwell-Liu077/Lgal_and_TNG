import numpy as np
import illustris_python as il
from numba import jit

def utherm_to_temp(utherm, nelec):
    hydrogen_massfrac = 0.76 # approximate
    mass_proton = 1.672622e-24 # cgs
    gamma = 5/3
    boltzmann = 1.380650e-16 # cgs (erg/K)

    # unit system
    UnitLength_in_cm = 3.085678e21   # 1.0 kpc
    UnitMass_in_g = 1.989e43 # 1.0e10 solar masses
    UnitVelocity_in_cm_per_s = 1.0e5 # 1 km/sec

    UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s
    UnitEnergy_in_cgs = UnitMass_in_g * UnitLength_in_cm**2.0 / UnitTime_in_s**2.0

    # calculate mean molecular weight
    meanmolwt = 4.0/(1.0 + 3.0 * hydrogen_massfrac + 4.0* hydrogen_massfrac * nelec)
    meanmolwt *= mass_proton

    # calculate temperature (K)
    temp = utherm * (gamma-1.0) / boltzmann * UnitEnergy_in_cgs / UnitMass_in_g * meanmolwt

    return temp.astype('float32')

# 读取 100 个 snapshot 的红移，并按 snapshot 99 → 0 的顺序保存到数组中
def give_z_array(basePath):
    z = np.zeros(100)
    for i in range(99,-1,-1):
        z[99-i] = il.groupcat.loadHeader(basePath,i)['Redshift']
    return z

# 这个函数用于计算两个宇宙尺度因子 a0 和 a1 对应的宇宙时间差，结果单位是 Gyr
@jit(nopython = True)
def get_time_difference_in_Gyr(a0, a1):
    omega_lambda = 0.6911
    omega_m = 0.3089
    h = 0.6774
    H0 = 3.2407789e-18 #in s^-1
    
    SEC_PER_MEGAYEAR = 1000000 * 365 * 86400 
    
    factor1 = 2.0 / (3.0 * np.sqrt(omega_lambda))

    term1   = np.sqrt(omega_lambda / omega_m) * a0**1.5
    term2   = np.sqrt(1 + omega_lambda / omega_m * a0**3)
    factor2 = np.log(term1 + term2)

    t0 = factor1 * factor2

    term1   = np.sqrt(omega_lambda / omega_m) * a1**1.5
    term2   = np.sqrt(1 + omega_lambda / omega_m * a1**3)
    factor2 = np.log(term1 + term2)

    t1 = factor1 * factor2

    result = t1 - t0

    time_diff = result / (H0 * h) #now in seconds
    time_diff /= SEC_PER_MEGAYEAR * 1000 #now in gigayears

    return time_diff