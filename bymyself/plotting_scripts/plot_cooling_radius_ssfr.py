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
dirname = '/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/pics/ssfr_cooling_radius'
os.makedirs(dirname, exist_ok=True)

style = 'solid'
what_to_plot = 'r_vir'

for i, snap in enumerate(snaps):
    with mpl.rc_context({'xtick.top' : False}):
        fig,ax = plt.subplots(1, 1, figsize = (16,9))

    groups = il.groupcat.loadHalos(basePath, snaps[i], fields = ['Group_M_Crit200'])
    group_masses = np.asarray(groups, dtype=np.float32) * 1e10 / h
    del groups

    assert isfile(f'/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/cooling_radius_{snaps[i]}.hdf5'), 'Cooling radius file does not exist!'
    assert isfile(f'/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/ssfr_classification_{snaps[i]}.hdf5'), 'SSFR classification file does not exist!'
    cooling_file = f'/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/cooling_radius_{snaps[i]}.hdf5'
    class_file = f'/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/ssfr_classification_{snaps[i]}.hdf5'
    f1 = h5py.File(cooling_file,'r')
    haloFlag = np.asarray(f1["halo_flag"][:], dtype=bool)
    cr = f1['cooling_radius'][:]
    f1.close()

    f2 = h5py.File(class_file,'r')
    class_halo_ids = np.asarray(f2["halo_id"][:], dtype=np.int64)
    class_code = np.asarray(f2["class_code"][:], dtype=np.int8)
    f2.close()

    class_by_halo = np.full(group_masses.size, -1, dtype=np.int8)
    valid_ids = ((class_halo_ids >= 0) & (class_halo_ids < group_masses.size))
    class_by_halo[class_halo_ids[valid_ids]] = class_code[valid_ids]

    class_info = {0: ("Star Burst", "#fd7825"), 1: ("Main Sequence", "#fde725"), 2: ("Green Valley", "#35b779"), 3: ("Quenched", "#440154")}

    for code, (label, color) in class_info.items():
        mask = haloFlag & (class_by_halo == code)
        number = np.count_nonzero(mask)
        if number < 5:
            print(f"{label}: only {number} valid halos, skip plotting.")
            continue

        x = np.log10(group_masses[mask])
        y = cr[mask]
        xmed, ymed, y16, y84 = funcs.binData_med(x, y, 25)
        ax.plot(xmed, ymed, color=color, linestyle = style)
        ax.fill_between(xmed, y16, y84, color=color, alpha = 0.2)

    rec_dwarf = Rectangle((10.8, -0.2), 0.4, 1900, color = 'lightgray', alpha = 0.3)
    ax.add_patch(rec_dwarf)
    rec_mw = Rectangle((11.8, -0.2), 0.4, 1900, color = 'lightgray', alpha = 0.3)
    ax.add_patch(rec_mw)
    rec_group = Rectangle((12.6, -0.2), 0.4, 1900, color = 'lightgray', alpha = 0.3)
    ax.add_patch(rec_group)

    ax.set_xlim(10.8,13)
    ax.set_ylim(0.1,10)
    ax.set_yscale('log')

    group_m = np.array([6.86093282699585, 7.398449420928955, 7.901801109313965, 8.393900871276855, 8.886045455932617,\
                        9.339580535888672, 9.809768676757812, 10.323699951171875, 10.840004920959473, 11.344971656799316,\
                        11.868677139282227, 12.379907608032227, 12.879098892211914, 13.441740036010742, 13.891986846923828])
    gal_m = np.array([4.742977619171143, 4.718081474304199, 4.761687278747559, 4.795966625213623, 4.861518859863281,\
                    5.147671699523926, 6.013262748718262, 7.66148042678833, 8.850899696350098, 9.708621978759766,\
                    10.465829849243164, 11.046808242797852, 11.511899948120117, 11.899462699890137, 12.20790958404541])

    secax = ax.twiny() # 创建一个共享纵轴、但拥有独立横轴的副坐标轴
    f_interp = interpolate.interp1d(gal_m, group_m) # 建立插值关系
    new_tick_locations = np.array([9., 9.5, 10., 10.5, 11., 11.5, 12.])
    secax.set_xlim(ax.get_xlim()) # 让顶部坐标轴和底部坐标轴使用相同的横坐标范围
    secax.set_xticks(f_interp(new_tick_locations)) # 将顶部恒星质量刻度转换成对应的 halo mass 位置
    secax.set_xticklabels(new_tick_locations) # 将顶部刻度显示为恒星质量的数值，而不是实际的 halo mass
    secax.minorticks_off() # 关闭顶部坐标轴的小刻度
    secax.set_xlabel(r'stellar mass [$\log\,\rm{M}_\odot$]')

    ax.minorticks_on()
    ax.grid(which = 'major',axis = 'y')
    ax.set_xticks([11,12,13])
    ax.set_xticklabels([11,12,13])

    orange = mpatches.Patch(color="#fd7825", linestyle = 'solid', label = "Star Burst")
    yellow = mpatches.Patch(color="#fde725", linestyle = 'solid', label = "Main Sequence")
    green = mpatches.Patch(color="#35b779", linestyle = 'solid', label = "Green Valley")
    purple = mpatches.Patch(color="#440154", linestyle = 'solid', label = "Quenched")

    legend = plt.legend(handles=[orange,yellow,green,purple], ncol=1, loc = 'upper right') # ncol=1：图例分成一列

    ax.text(0.05, 0.95, rf"$z={z_snaps[i]:.2f}$", transform=ax.transAxes, ha="left", va="top", fontsize=18, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    ax.set_xlabel(r'halo mass [$\log\,\rm{M}_\odot$]')
    ax.set_ylabel(r'Cooling radius $R_{\rm cr}$ [$R_{\rm 200c}$]')
    fig.tight_layout() # 自动调整图中的间距
    plt.savefig(dirname + '/ssfr_cr_vs_mass_' + what_to_plot + f'snap{snaps[i]}_50-1.pdf',format='pdf')
