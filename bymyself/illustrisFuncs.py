import numpy as np

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

def V200c(M_200c, R_200c, a):
    G = 4.30091e-6 # G: kpc (km/s)^2 / Msun
    V_200c = np.sqrt((G * 1.0e10 * M_200c) / (a * R_200c))

    return V_200c.astype('float32')