import h5py
import illustris_python as il
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import interpolate
from matplotlib.ticker import LogLocator

import funcs


plt.style.use("fancy_plots2.mplstyle")

snap = 99
num_bins = 25
basePath = "/public/share/chenhouzun/TNG50-1/output/"
data_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")
figure_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/pics/cooling_radius")
figure_dir.mkdir(parents=True, exist_ok=True)

header = il.groupcat.loadHeader(basePath, snap)
redshift = float(header["Redshift"])

tree_file = data_dir / "all_selected_trees.hdf5"
with h5py.File(tree_file, "r") as f:
    start_ids = np.asarray(f["start_subhalo_id"][:], dtype=np.int64)
    snap_numbers = np.asarray(f["snap_numbers"][:], dtype=np.int64)
    snap_columns = np.flatnonzero(snap_numbers == snap)
    fof_ids = np.asarray(f["mpb_fof_ids"][:, int(snap_columns[0])], dtype=np.int64)

analytic_file = data_dir / f"cooling_radius_{snap}.hdf5"

with h5py.File(analytic_file, "r") as f:
    analytic_radius_all = np.asarray(f["cooling_radius"][:], dtype=np.float64)
    analytic_halo_flag = np.asarray(f["halo_flag"][:], dtype=bool)

real_file = data_dir / f"real_cooling_radius_{snap}.hdf5"

with h5py.File(real_file, "r") as f:
    real_start_ids = np.asarray(f["start_subhalo_id"][:], dtype=np.int64)
    halo_mass_msun = np.asarray(f["m200c_msun"][:], dtype=np.float64)
    real_radius = np.asarray(f["cooling_radius_r200c"][:], dtype=np.float64)
    real_status = np.asarray(f["status"][:], dtype=np.uint8)
    real_analysis_valid = np.asarray(f["analysis_valid"][:], dtype=bool)

analytic_radius = np.full(start_ids.size, np.nan, dtype=np.float64)
valid_fof = (fof_ids >= 0) & (fof_ids < analytic_radius_all.size)
valid_rows = np.flatnonzero(valid_fof)
analytic_radius[valid_rows] = analytic_radius_all[fof_ids[valid_rows]]
analytic_valid = np.zeros(start_ids.size, dtype=bool)
analytic_valid[valid_rows] = analytic_halo_flag[fof_ids[valid_rows]] & np.isfinite(analytic_radius[valid_rows]) & (analytic_radius[valid_rows] > 0.0)

valid_mass = np.isfinite(halo_mass_msun) & (halo_mass_msun > 0.0)
real_valid = real_analysis_valid & (real_status == 1) & np.isfinite(real_radius) & (real_radius > 0.0)
common_sample = valid_mass & analytic_valid & real_valid

log_halo_mass = np.log10(halo_mass_msun[common_sample])
analytic_radius_selected = analytic_radius[common_sample]
real_radius_selected = real_radius[common_sample]
x_analytic, y_analytic, y16_analytic, y84_analytic = funcs.binData_med(log_halo_mass, analytic_radius_selected, num_bins)
x_real, y_real, y16_real, y84_real = funcs.binData_med(log_halo_mass, real_radius_selected, num_bins)

fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax.plot(x_analytic, y_analytic, color="C0", linewidth=4.0, label="Lgal analytic")
ax.fill_between(x_analytic, y16_analytic, y84_analytic, color="C0", alpha=0.28)
ax.plot(x_real, y_real, color="C1", linewidth=4.0, label="TNG measured")
ax.fill_between(x_real, y16_real, y84_real, color="C1", alpha=0.28)

ax.axvspan(10.8, 11.2, color="lightgray", alpha=0.25)
ax.axvspan(11.8, 12.2, color="lightgray", alpha=0.25)
ax.axvspan(12.6, 13.4, color="lightgray", alpha=0.25)

ax.set_xlim(10.8, 13.4)
max_radius = np.nanmax(np.concatenate((analytic_radius_selected, real_radius_selected)))
ax.set_ylim(0.01, max(10.0, 1.1 * max_radius))
ax.set_yscale("log")
ax.set_xlabel(r"halo mass [$\log_{10}(M_{200c}/M_\odot)$]")
ax.set_ylabel(r"$R_{\rm cool}/R_{\rm 200c}$")
ax.grid(which="major", axis="y", alpha=0.3)
ax.minorticks_on()
ax.set_xticks([11, 12, 13])
ax.legend(loc="lower left", frameon=False, fontsize=30)

group_m = np.array([6.86093282699585, 7.398449420928955, 7.901801109313965, 8.393900871276855, 8.886045455932617, 9.339580535888672, 9.809768676757812, 10.323699951171875, 10.840004920959473, 11.344971656799316, 11.868677139282227, 12.379907608032227, 12.879098892211914, 13.441740036010742, 13.891986846923828])
gal_m = np.array([4.742977619171143, 4.718081474304199, 4.761687278747559, 4.795966625213623, 4.861518859863281, 5.147671699523926, 6.013262748718262, 7.66148042678833, 8.850899696350098, 9.708621978759766, 10.465829849243164, 11.046808242797852, 11.511899948120117, 11.899462699890137, 12.20790958404541])
secax = ax.twiny()
f_interp = interpolate.interp1d(gal_m, group_m)
stellar_mass_ticks = np.array([8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5])
secax.set_xlim(ax.get_xlim())
secax.set_xticks(f_interp(stellar_mass_ticks))
secax.set_xticklabels(stellar_mass_ticks)
secax.minorticks_off()
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
ax.tick_params(axis="y", which="minor", left=True, right=True, length=4, width=0.8)
secax.set_xlabel(r"stellar mass [$\log_{10}(M_\odot)$]")

fig.tight_layout()
output_file = figure_dir / f"cooling_radius_Lgal_vs_TNG_real_snap{snap}.pdf"
fig.savefig(output_file, format="pdf")
plt.close(fig)
