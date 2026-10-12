import illustris_python as il
import matplotlib.pyplot as plt
import numpy as np
import h5py
from matplotlib.lines import Line2D
import numpy as np
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
dirname = '/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/pics/ssfr_m200c_classification'
os.makedirs(dirname, exist_ok=True)

for i, snap in enumerate(snaps):
    with mpl.rc_context({'xtick.top' : False}):
        fig,ax = plt.subplots(1, 1, figsize = (16,9))

    groups = il.groupcat.loadHalos(basePath, snaps[i], fields = ['Group_M_Crit200'])
    group_masses = np.asarray(groups, dtype=np.float32) * 1e10 / h
    del groups

    assert isfile(f"/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/ssfr_classification_{snap}.hdf5")
    class_file = f"/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/ssfr_classification_{snap}.hdf5"
    f = h5py.File(class_file, "r")
    halo_ids = np.asarray(f["halo_id"][:], dtype=np.int64)
    central_subhalo_ids = np.asarray(f["central_subhalo_id"][:], dtype=np.int64)
    class_codes = np.asarray(f["class_code"][:], dtype=np.int8)
    selection_flag = np.asarray(f["selection_flag"][:], dtype=bool)
    f.close()

    subhalos = il.groupcat.loadSubhalos(basePath, snap, fields=["SubhaloSFR", "SubhaloMassType"])
    subhalo_sfr = np.asarray(subhalos["SubhaloSFR"], dtype=np.float64)
    subhalo_mstar_msun = np.asarray(subhalos["SubhaloMassType"][:, 4], dtype=np.float64) * 1.0e10 / h
    del subhalos

    rows = np.flatnonzero(selection_flag)

    selected_halo_ids = halo_ids[rows]
    selected_central_ids = central_subhalo_ids[rows]
    selected_codes = class_codes[rows]
    m200c = group_masses[selected_halo_ids]
    mstar = subhalo_mstar_msun[selected_central_ids]
    sfr = subhalo_sfr[selected_central_ids]

    valid = (np.isfinite(m200c) & (m200c > 0.0) & np.isfinite(mstar) & (mstar > 0.0) & np.isfinite(sfr) & (sfr >= 0.0))

    log_m200c = np.log10(m200c[valid])
    log_ssfr = np.full(np.count_nonzero(valid), np.nan, dtype=np.float64)
    valid_sfr = sfr[valid] > 0.0
    valid_mstar = mstar[valid]
    log_ssfr[valid_sfr] = np.log10(sfr[valid][valid_sfr]) - np.log10(valid_mstar[valid_sfr])
    selected_codes = selected_codes[valid]
    zero_sfr = ~valid_sfr

    legend_handles = []

    class_info = {0: ("Star Burst", "#fd7825"), 1: ("Main Sequence", "#fde725"), 2: ("Green Valley", "#35b779"), 3: ("Quenched", "#440154")}

    for code, (label, color) in class_info.items():
        class_mask = selected_codes == code
        positive_sfr = class_mask & ~zero_sfr
        zero_sfr_class = class_mask & zero_sfr
        number = np.count_nonzero(class_mask)

        ax.scatter(log_m200c[positive_sfr], log_ssfr[positive_sfr], s=18, color=color, alpha=0.65, linewidths=0, rasterized=True, zorder=2)
        if np.any(zero_sfr_class):
            ax.scatter(log_m200c[zero_sfr_class], np.full(np.count_nonzero(zero_sfr_class), -14.18), s=28, marker="v", color=color, alpha=0.85, linewidths=0, rasterized=True, zorder=3)

        legend_handles.append(mpatches.Patch(color=color, label=f"{label} (N={number:,})"))

    rec_dwarf = Rectangle((10.8, -14), 0.4, 6, color="lightgray", alpha=0.3)
    ax.add_patch(rec_dwarf)
    rec_mw = Rectangle((11.8, -14), 0.4, 6, color="lightgray", alpha=0.3)
    ax.add_patch(rec_mw)
    rec_group = Rectangle((12.6, -14), 0.4, 6, color="lightgray", alpha=0.3)
    ax.add_patch(rec_group)

    ax.set_xlim(9, 13)
    ax.set_ylim(-14, -8)

    group_m = np.array([6.86093282699585, 7.398449420928955, 7.901801109313965, 8.393900871276855, 8.886045455932617, 9.339580535888672, 9.809768676757812, 10.323699951171875, 10.840004920959473, 11.344971656799316, 11.868677139282227, 12.379907608032227, 12.879098892211914, 13.441740036010742, 13.891986846923828])
    gal_m = np.array([4.742977619171143, 4.718081474304199, 4.761687278747559, 4.795966625213623, 4.861518859863281, 5.147671699523926, 6.013262748718262, 7.66148042678833, 8.850899696350098, 9.708621978759766, 10.465829849243164, 11.046808242797852, 11.511899948120117, 11.899462699890137, 12.20790958404541])

    secax = ax.twiny()
    f_interp = interpolate.interp1d(gal_m, group_m)
    new_tick_locations = np.array([9., 9.5, 10., 10.5, 11., 11.5, 12.])
    secax.set_xlim(ax.get_xlim())
    secax.set_xticks(f_interp(new_tick_locations))
    secax.set_xticklabels(new_tick_locations)
    secax.minorticks_off()
    secax.set_xlabel(r'stellar mass [$\log\,\rm{M}_\odot$]')

    ax.minorticks_on()
    ax.grid(which = 'major',axis = 'y')
    ax.set_xticks([9,10,11,12,13])
    ax.set_xticklabels([9,10,11,12,13])

    if legend_handles:
        legend_handles.append(Line2D([], [], color="black", marker="v", linestyle="None", label="SFR=0"))
        ax.legend(handles=legend_handles, ncol=1, loc="upper right", fontsize=15)

    ax.text(0.05, 0.95, rf"$z={z_snaps[i]:.2f}$", transform=ax.transAxes, ha="left", va="top", fontsize=18, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    ax.set_xlabel(r"halo mass [$\log_{10}(M_{200c}/M_\odot)$]")
    ax.set_ylabel(r"$\log_{10}(\mathrm{sSFR}/\mathrm{yr}^{-1})$")
    fig.tight_layout()
    plt.savefig(dirname + f'snap{snaps[i]}_50-1.pdf',format='pdf')
