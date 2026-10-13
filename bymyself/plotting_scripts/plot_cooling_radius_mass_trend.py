import illustris_python as il
import matplotlib.pyplot as plt
import numpy as np
import h5py
import illustrisFuncs as iF
import funcs
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import matplotlib as mpl
from scipy import interpolate
import os
from os.path import isfile

plt.style.use('fancy_plots2.mplstyle')

basePath = "/public/share/chenhouzun/TNG50-1/output/"
header = il.groupcat.loadHeader(basePath, 99)
h = header['HubbleParam']

# set snapshots to plot
snaps = np.array([99, 67, 40], dtype=np.int64)
z_snaps = np.array([il.groupcat.loadHeader(basePath, int(snap))["Redshift"] for snap in snaps], dtype=np.float32)


# specify directory to save plots
dirname = '/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/pics/cooling_radius'
os.makedirs(dirname, exist_ok=True)

style = 'solid'
what_to_plot = 'r_vir'
with mpl.rc_context({'xtick.top' : False}):
    fig,ax = plt.subplots(1, 1, figsize = (16,9))

for i, snap in enumerate(snaps):
    header = il.groupcat.loadHeader(basePath, snap)
    h = header["HubbleParam"]

    halos = il.groupcat.loadHalos(basePath, snap, fields=["Group_M_Crit200", "GroupFirstSub"])
    group_masses = np.asarray(halos["Group_M_Crit200"], dtype=np.float64) * 1.0e10 / h
    group_first_sub = np.asarray(halos["GroupFirstSub"], dtype=np.int64)
    subhalo_mass_type = np.asarray(il.groupcat.loadSubhalos(basePath, int(snap), fields=["SubhaloMassType"]), dtype=np.float64)

    has_central = (group_first_sub >= 0) & (group_first_sub < subhalo_mass_type.shape[0])
    central_stellar_mass = np.full(group_first_sub.size, np.nan, dtype=np.float64)
    central_stellar_mass[has_central] = subhalo_mass_type[group_first_sub[has_central], 4] * 1.0e10 / h

    central_mass_ok = has_central & np.isfinite(central_stellar_mass) & (central_stellar_mass > 1.0e7)

    assert isfile(f"/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/cooling_radius_{snap}.hdf5"), "Cooling radius file does not exist!"
    with h5py.File(f"/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/cooling_radius_{snap}.hdf5", "r") as f:
        halo_flag = np.asarray(f["halo_flag"][:], dtype=bool)
        cr = np.asarray(f["cooling_radius"][:], dtype=np.float64)

    plot_mask = (halo_flag & central_mass_ok & np.isfinite(group_masses) & (group_masses > 0.0) & np.isfinite(cr))

    masses = np.log10(group_masses[plot_mask])
    cr_selected = cr[plot_mask]
    xmed, ymed, y16, y84 = funcs.binData_med(masses, cr_selected, 25)

    ax.plot(xmed, ymed, color="C" + str(i), linestyle=style)
    ax.fill_between(xmed, y16, y84, alpha=0.2)

rec_dwarf = Rectangle((10.8, -0.2), 0.4, 1900, color = 'lightgray', alpha = 0.3)
ax.add_patch(rec_dwarf)
rec_mw = Rectangle((11.8, -0.2), 0.4, 1900, color = 'lightgray', alpha = 0.3)
ax.add_patch(rec_mw)
rec_group = Rectangle((12.6, -0.2), 0.8, 1900, color = 'lightgray', alpha = 0.3)
ax.add_patch(rec_group)

ax.set_xlim(10,14)
ax.set_ylim(0.01,10)
ax.set_yscale('log')

group_m = np.array([6.86093282699585, 7.398449420928955, 7.901801109313965, 8.393900871276855, 8.886045455932617,\
                    9.339580535888672, 9.809768676757812, 10.323699951171875, 10.840004920959473, 11.344971656799316,\
                    11.868677139282227, 12.379907608032227, 12.879098892211914, 13.441740036010742, 13.891986846923828])
gal_m = np.array([4.742977619171143, 4.718081474304199, 4.761687278747559, 4.795966625213623, 4.861518859863281,\
                  5.147671699523926, 6.013262748718262, 7.66148042678833, 8.850899696350098, 9.708621978759766,\
                  10.465829849243164, 11.046808242797852, 11.511899948120117, 11.899462699890137, 12.20790958404541])

secax = ax.twiny() # 创建一个共享纵轴、但拥有独立横轴的副坐标轴
f_interp = interpolate.interp1d(gal_m, group_m) # 建立插值关系
new_tick_locations = np.array([7., 8., 8.5, 9., 9.5, 10., 10.5, 11., 11.5, 12.])
secax.set_xlim(ax.get_xlim()) # 让顶部坐标轴和底部坐标轴使用相同的横坐标范围
secax.set_xticks(f_interp(new_tick_locations)) # 将顶部恒星质量刻度转换成对应的 halo mass 位置
secax.set_xticklabels(new_tick_locations) # 将顶部刻度显示为恒星质量的数值，而不是实际的 halo mass
secax.minorticks_off() # 关闭顶部坐标轴的小刻度
secax.set_xlabel(r'stellar mass [$\log\,\rm{M}_\odot$]')

ax.minorticks_on()
ax.grid(which = 'major',axis = 'y')
ax.set_xticks([10,11,12,13,14])
ax.set_xticklabels([10,11,12,13,14])

blue = mpatches.Patch(color='C0', linestyle = 'solid', label = f'z = {z_snaps[0]:.1f}')
orange = mpatches.Patch(color='C1', linestyle = 'solid', label = f'z = {z_snaps[1]:.1f}')
green = mpatches.Patch(color='C2', linestyle = 'solid', label = f'z = {z_snaps[2]:.1f}')

legend = plt.legend(handles=[blue,orange,green], ncol=1, loc = 'upper right') # ncol=1：图例分成一列

ax.set_xlabel(r'halo mass [$\log\,\rm{M}_\odot$]')
ax.set_ylabel(r'Cooling radius $R_{\rm cr}$ [$R_{\rm 200c}$]')
fig.tight_layout() # 自动调整图中的间距
plt.savefig(dirname + '/cr_vs_mass_' + what_to_plot + f'_50-1.pdf',format='pdf')
