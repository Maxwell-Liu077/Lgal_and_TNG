import illustris_python as il
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import h5py
import matplotlib as mpl
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
dirname = '/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/pics/cf_cr_diagram'
os.makedirs(dirname, exist_ok=True)

for i, snap in enumerate(snaps):

    assert isfile(f'/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/cooling_radius_{snaps[i]}.hdf5'), 'Cooling radius file does not exist!'
    assert isfile(f'/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/cold_gas_fraction_{snaps[i]}.hdf5'), 'Cold gas fraction file does not exist!'
    assert isfile(f'/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/ssfr_classification_{snaps[i]}.hdf5'), 'Classigy galaxies file does not exist!'

    cooling_file = f"/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/cooling_radius_{snaps[i]}.hdf5"
    cold_file = f"/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/cold_gas_fraction_{snaps[i]}.hdf5"
    classify_file = f"/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/ssfr_classification_{snaps[i]}.hdf5"

    with h5py.File(cooling_file, "r") as f:
        cr = np.asarray(f["cooling_radius"][:], dtype=np.float32)
        cooling_flag = np.asarray(f["halo_flag"][:], dtype=bool)

    with h5py.File(cold_file, "r") as f:
        cold_fraction = np.asarray(f["cold_gas_fraction"][:], dtype=np.float32)
        cold_flag = np.asarray(f["halo_flag"][:], dtype=bool)

    with h5py.File(classify_file, "r") as f:
        halo_id = np.asarray(f["halo_id"][:], dtype=np.int64)
        class_code = np.asarray(f["class_code"][:], dtype=np.int8)

    valid = cooling_flag & cold_flag & (cr >= 0) & (cr <= 1) & (cold_fraction >= 0) & (cold_fraction <= 1)
    class_info = {0: ("Starburst", "#d73027"), 1: ("Main Sequence", "#4575b4"), 2: ("Green Valley", "#fdae61"), 3: ("Quenched", "#542788")}

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex=True, sharey=True)
    axes = axes.ravel()
    hexbin_list = []

    for ax, (code, (label, color)) in zip(axes, class_info.items()):
        mask = valid & (class_code == code)
        number = np.count_nonzero(mask)

        if number == 0:
            ax.text(0.5, 0.5, "No valid halos", ha="center", va="center", transform=ax.transAxes)
        else:
            hb = ax.hexbin(cr[mask], cold_fraction[mask], gridsize=35, mincnt=1, cmap="plasma")
            hexbin_list.append(hb)

        ax.set_title(f"{label}  (N={number})", color=color, fontsize=16)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")

    vmax = max(np.max(hb.get_array()) for hb in hexbin_list)
    shared_norm = LogNorm(vmin=1, vmax=vmax)

    for hb in hexbin_list:
        hb.set_norm(shared_norm)

    fig.colorbar(hexbin_list[0], ax=axes.tolist(), label="Number of halos", pad=0.03)

    fig.supxlabel(r"$R_{\rm cool}/R_{\rm 200c}$")
    fig.supylabel(r"$M_{\rm cold}/M_{\rm gas}$")
    fig.suptitle(rf"$z={z_snaps[i]:.2f}$", fontsize=18)

    fig.savefig(dirname + f'/cf_cr_diagram_{snaps[i]}.pdf', format = 'pdf', dpi=300)
    plt.close(fig)