import illustris_python as il
import matplotlib.pyplot as plt
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
    with mpl.rc_context({'xtick.top' : False}):
        fig,ax = plt.subplots(1, 1, figsize = (8, 8))

    assert isfile(f'/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/cooling_radius_{snaps[i]}.hdf5'), 'Cooling radius file does not exist!'
    assert isfile(f'/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/cold_gas_fraction_{snaps[i]}.hdf5'), 'Cold gas fraction file does not exist!'

    cooling_file = f"/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/cooling_radius_{snaps[i]}.hdf5"
    cold_file = f"/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/cold_gas_fraction_{snaps[i]}.hdf5"

    with h5py.File(cooling_file, "r") as f:
        cr = np.asarray(f["cooling_radius"][:], dtype=np.float32)
        cooling_flag = np.asarray(f["halo_flag"][:], dtype=bool)

    with h5py.File(cold_file, "r") as f:
        cold_fraction = np.asarray(f["cold_gas_fraction"][:], dtype=np.float32)
        cold_flag = np.asarray(f["halo_flag"][:], dtype=bool)

    valid = cooling_flag & cold_flag & (cr >= 0) & (cr <= 1) & (cold_fraction >= 0) & (cold_fraction <= 1)

    x = cr
    y = cold_fraction

    hb = ax.hexbin(x[valid], y[valid], gridsize=35, mincnt=1, cmap='plasma', bins='log')
    fig.colorbar(hb, ax=ax, label="Number of halos")

    ax.set_xlabel(r"$R_{\rm cool}/R_{\rm 200c}$")
    ax.set_ylabel(r"$M_{\rm cold}/M_{\rm gas}$")
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.9, 0.1,f'z={z_snaps[i]:.2f}' + str(), bbox=dict(boxstyle="round", ec='lightgray', fc='white', alpha = 0.8), transform=ax.transAxes, size = 20)

    plt.tight_layout()
    plt.savefig(dirname + f'/cf_cr_diagram_snap{snaps[i]}.pdf', format = 'pdf', dpi=300)