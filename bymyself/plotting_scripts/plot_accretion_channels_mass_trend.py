import illustris_python as il
import matplotlib.pyplot as plt
import numpy as np
import h5py

import funcs
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

import sys
from os.path import isfile, isdir

plt.style.use('fancy_plots2.mplstyle')

run = int(sys.argv[1])
numBins = 25

# specify output directory
dirname = 'files/accretion_channels'
assert isdir(dirname), 'output directory does not exist!'

basePath = f'/virgotng/universe/IllustrisTNG/TNG50-{run}/output'
h_const = il.groupcat.loadHeader(basePath, 99)['HubbleParam']
num_subs = il.groupcat.loadHeader(basePath, 99)['Nsubgroups_Total']
raw_stellar_masses = il.groupcat.loadSubhalos(basePath, 99, fields = ['SubhaloMassType'])[:,4] * 1e10 / h_const

assert isfile('/vera/ptmp/gc/olwitt/auxCats/' + basePath[32:39] + '/accretion_channels_insitu_test2.hdf5'), 'Accretion channels file (in-situ) does not exist!'
f = h5py.File('/vera/ptmp/gc/olwitt/auxCats/' + basePath[32:39] + '/accretion_channels_insitu_test2.hdf5','r')
all_ids = np.arange(num_subs)

gal_accretion_channels = f['accretion_channels/for_all_galaxies'][:,:,:]
subhaloFlag = f['subhaloFlag'][:]

f.close()

mask = np.nonzero(subhaloFlag)[0]
stellar_masses = np.log10(raw_stellar_masses[mask])#[all_ids]
gal_accretion_channels = gal_accretion_channels[mask,:,:] #choose all stars, not insitu (1) or medsitu (2)

mass_bins, from_igm_bins, _, _ = funcs.binData_med(stellar_masses, gal_accretion_channels[:,0,0], numBins)
_, mergers_bins, _, _ = funcs.binData_med(stellar_masses, gal_accretion_channels[:,0,2], numBins)
_, stripped_from_halos_bins, _, _ = funcs.binData_med(stellar_masses, gal_accretion_channels[:,0,3], numBins)
_, nep_wind_recycle_bins, _, _ = funcs.binData_med(stellar_masses, gal_accretion_channels[:,0,5], numBins)
_, stripped_from_sat_bins, _, _ = funcs.binData_med(stellar_masses, gal_accretion_channels[:,0,6], numBins)
_, recycled2_bins, _, _ = funcs.binData_med(stellar_masses, gal_accretion_channels[:,0,7], numBins)

# mass_bins, from_igm_bins = funcs.binData_mean(stellar_masses, gal_accretion_channels[:,0,0], numBins)
# _, mergers_bins = funcs.binData_mean(stellar_masses, gal_accretion_channels[:,0,2], numBins)
# _, stripped_from_halos_bins = funcs.binData_mean(stellar_masses, gal_accretion_channels[:,0,3], numBins)
# _, nep_wind_recycle_bins = funcs.binData_mean(stellar_masses, gal_accretion_channels[:,0,5], numBins)
# _, stripped_from_sat_bins = funcs.binData_mean(stellar_masses, gal_accretion_channels[:,0,6], numBins)
# _, recycled2_bins = funcs.binData_mean(stellar_masses, gal_accretion_channels[:,0,7], numBins)

#### dm ####

assert isfile('/vera/ptmp/gc/olwitt/auxCats/' + basePath[32:39] + '/accretion_channels_dm_test.hdf5'), 'Accretion channels file (DM) does not exist!'
f = h5py.File('/vera/ptmp/gc/olwitt/auxCats/' + basePath[32:39] + '/accretion_channels_dm_test.hdf5','r')
overall_fracs = f['accretion_channels/overall_fractions'][:]
dm_gal_accretion_channels = f['accretion_channels/for_all_galaxies'][:,:]
subhaloFlag = f['subhaloFlag'][:]
f.close()

mask1 = np.nonzero(subhaloFlag)[0]

stellar_masses = raw_stellar_masses[mask1]
dm_mass_mask = np.nonzero(stellar_masses)[0]
stellar_masses = np.log10(stellar_masses[dm_mass_mask])
dm_gal_accretion_channels = dm_gal_accretion_channels[mask1[dm_mass_mask],:]

dm_non_zero_mask = np.where(np.any(dm_gal_accretion_channels[:,:] > 0, axis = 1))[0]

dm_mass_bins, dm_mergers_bins,_ ,_ = funcs.binData_med(stellar_masses[dm_non_zero_mask],\
                                                       dm_gal_accretion_channels[dm_non_zero_mask,2], numBins)
_, dm_stripped_from_halos_bins,_ ,_ = funcs.binData_med(stellar_masses[dm_non_zero_mask],\
                                                       dm_gal_accretion_channels[dm_non_zero_mask,3], numBins)
_, dm_stripped_from_sat_bins,_ ,_ = funcs.binData_med(stellar_masses[dm_non_zero_mask],\
                                                            dm_gal_accretion_channels[dm_non_zero_mask,6], numBins)
_, dm_from_igm_bins,_ ,_ = funcs.binData_med(stellar_masses[dm_non_zero_mask],\
                                                         dm_gal_accretion_channels[dm_non_zero_mask,0], numBins)

#### exsitu ####

assert isfile('/vera/ptmp/gc/olwitt/auxCats/' + basePath[32:39] + '/accretion_channels_exsitu_test.hdf5'), 'Accretion channels file (ex-situ) does not exist!'
f = h5py.File('/vera/ptmp/gc/olwitt/auxCats/' + basePath[32:39] + '/accretion_channels_exsitu_test.hdf5','r')
subhaloFlag = f['subhaloFlag'][:]
ex_gal_accretion_channels = f['accretion_channels/for_all_galaxies'][:,:]
overall_fracs = f['accretion_channels/overall_fractions'][:]
f.close()

mask1 = np.nonzero(subhaloFlag)[0]
stellar_masses = raw_stellar_masses[mask1]
ex_mass_mask = np.nonzero(stellar_masses)[0]
stellar_masses = np.log10(stellar_masses[ex_mass_mask])
ex_gal_accretion_channels = ex_gal_accretion_channels[mask1[ex_mass_mask],:]

ex_non_zero_mask = np.where(np.any(ex_gal_accretion_channels[:,:] > 0, axis = 1))[0]

ex_mass_bins, ex_mergers_bins,_ ,_ = funcs.binData_med(stellar_masses[ex_non_zero_mask],\
                                                       ex_gal_accretion_channels[ex_non_zero_mask,2], numBins)
_, ex_stripped_from_halos_bins,_ ,_ = funcs.binData_med(stellar_masses[ex_non_zero_mask],\
                                                       ex_gal_accretion_channels[ex_non_zero_mask,3], numBins)
_, ex_stripped_from_sat_bins,_ ,_ = funcs.binData_med(stellar_masses[ex_non_zero_mask],\
                                                         ex_gal_accretion_channels[ex_non_zero_mask,6], numBins)

angalc_m_star = np.array([6.27728841, 9.3708718, 10.36557844, 11.42453622])
angalc_smooth = np.array([0.6, 0.18, 0.15, 0.38])
angalc_clumpy = np.array([0, 0.05, 0.11, 0.19])
angalc_inter = np.array([0, 0.21, 0.47, 0.19])
angalc_wind = np.array([0.4, 0.56, 0.27, 0.24])

fig, ax = plt.subplots(1,1, figsize=(16,9))

### in-situ ###

plt.plot(mass_bins, recycled2_bins, color = 'tab:green', label = 'new recycled')
plt.plot(mass_bins, from_igm_bins - recycled2_bins, color = 'tab:orange', label = 'new fresh accretion')
plt.plot(mass_bins, mergers_bins, color = 'tab:red', label = 'mergers: gas (clumpy)')
plt.plot(mass_bins, stripped_from_sat_bins, color = 'tab:brown', label = 'mergers: gas (clumpy)')
plt.plot(mass_bins, stripped_from_halos_bins, color = 'tab:purple', label = 'stripped/ejected from halos')

### ex-situ ###

plt.plot(ex_mass_bins, ex_stripped_from_sat_bins, color = 'tab:brown', linestyle = 'dotted', label = 'stripped from satellites')
plt.plot(ex_mass_bins, ex_stripped_from_halos_bins, color = 'tab:purple', linestyle = 'dotted')
plt.plot(ex_mass_bins, ex_mergers_bins, color = 'tab:red', linestyle = 'dotted')

### dm ###

plt.plot(dm_mass_bins, dm_mergers_bins, color = 'tab:red', linestyle = 'dashed')
# plt.plot(dm_mass_bins, dm_stripped_from_halos_bins, color = 'tab:purple', linestyle = 'dashed')
plt.plot(dm_mass_bins, dm_stripped_from_sat_bins, color = 'tab:brown', linestyle = 'dashed')
# plt.plot(dm_mass_bins, dm_from_igm_bins, color = 'tab:orange', linestyle = 'dashed')


fresh = mpatches.Patch(color='tab:orange', label='fresh accretion')
wind = mpatches.Patch(color='tab:green', label='wind recycled')
merger = mpatches.Patch(color='tab:red', label='mergers: gas')
stripped_from_sat = mpatches.Patch(color='tab:brown', label='stripped from satellites')
strip = mpatches.Patch(color='tab:purple', label='stripped/ejected from halos')

insitu = mlines.Line2D([], [], color='tab:gray', linestyle = 'solid', label='in-situ')
dm = mlines.Line2D([], [], color='tab:gray', linestyle = 'dashed', label='DM')
exsitu = mlines.Line2D([], [], color='tab:gray', linestyle = 'dotted', label='ex-situ')

plt.scatter(angalc_m_star[1:], angalc_smooth[1:], marker = 'D', color = 'tab:orange', s = 120)
plt.scatter(angalc_m_star[1:], angalc_clumpy[1:], marker = 'D', color = 'tab:red', s = 120)
plt.scatter(angalc_m_star[1:], angalc_inter[1:], marker = 'D', color = 'tab:purple', s = 120)
plt.scatter(angalc_m_star[1:], angalc_wind[1:], marker = 'D', color = 'tab:green', s = 120)

D = plt.scatter(0,0, marker = 'D', color = 'gray', label = 'Anglés-Alcázar (2017)', s = 120)

legend1 = plt.legend(handles = [fresh, wind, merger, stripped_from_sat, strip, D], loc = 'upper right')

legend2 = plt.legend(handles = [insitu, exsitu, dm], loc = 'upper left')

plt.gca().add_artist(legend1)

plt.xlabel(r'stellar mass [$\log\,\rm{M}_\odot$]')
plt.ylabel(r'$M_{\ast,\,\rm insitu}\,(\rm mode) \,/\, M_{\ast,\,\rm insitu}$')
plt.xlim(9,12.0)

ax.set_xticks([9,10,11,12])
ax.set_xticklabels([9,10,11,12])

plt.ylim(-0.03,1.03)
plt.tight_layout()
plt.savefig(dirname + f'/accretion_50-{run}_alltypes.pdf', format = 'pdf')