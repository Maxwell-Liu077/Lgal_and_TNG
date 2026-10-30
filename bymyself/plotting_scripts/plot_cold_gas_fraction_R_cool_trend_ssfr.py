"""Compare Lgal and TNG cooling radii against CGM cold-gas fractions and PDFs."""

import os

import h5py
import illustris_python as il
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


plt.style.use("fancy_plots2.mplstyle")

basePath = "/public/share/chenhouzun/TNG50-1/output/"
snap = 99
mw_log_m200c_min = 11.8
mw_log_m200c_max = 12.2
ssfr_color_min = -14.0  # SFR=0 objects are shown at this lower plotting limit.
ssfr_color_max = -9.0
num_pdf_bins = 25
pdf_linewidth = 2.5
starforming_color = "tab:blue"
quenched_color = "tab:red"

data_dir = "/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data"
figure_dir = "/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/pics/cf_cr_diagram"
os.makedirs(figure_dir, exist_ok=True)

header = il.groupcat.loadHeader(basePath, snap)
h = float(header["HubbleParam"])
redshift = float(header["Redshift"])

# These arrays are indexed by FoF halo ID, so they align with the group catalogue.
cooling_file = os.path.join(data_dir, f"cooling_radius_{snap}.hdf5")
cold_file = os.path.join(data_dir, f"cold_gas_fraction_{snap}_1.hdf5")
with h5py.File(cooling_file, "r") as f:
    lgal_radius = np.asarray(f["cooling_radius"][:], dtype=np.float64)
    lgal_flag = np.asarray(f["halo_flag"][:], dtype=bool)

with h5py.File(cold_file, "r") as f:
    cold_fraction = np.asarray(f["cold_gas_fraction"][:], dtype=np.float64)
    cold_flag = np.asarray(f["halo_flag"][:], dtype=bool)

halos = il.groupcat.loadHalos(basePath, snap, fields=["Group_M_Crit200", "GroupFirstSub"])
halo_mass_msun = np.asarray(halos["Group_M_Crit200"], dtype=np.float64) * 1.0e10 / h
central_subhalo_ids = np.asarray(halos["GroupFirstSub"], dtype=np.int64)
num_halos = halo_mass_msun.size

# Apply the MW-like halo-mass selection to the snap-specific nodes in the
# selected-tree cooling-radius file.
valid_halo_mass = np.isfinite(halo_mass_msun) & (halo_mass_msun > 0.0)
log_halo_mass = np.full(num_halos, np.nan, dtype=np.float64)
log_halo_mass[valid_halo_mass] = np.log10(halo_mass_msun[valid_halo_mass])
mw_like = valid_halo_mass & (log_halo_mass >= mw_log_m200c_min) & (log_halo_mass < mw_log_m200c_max)

tree_file = os.path.join(data_dir, "all_selected_trees.hdf5")
with h5py.File(tree_file, "r") as f:
    start_ids = np.asarray(f["start_subhalo_id"][:], dtype=np.int64)
    tree_snap_numbers = np.asarray(f["snap_numbers"][:], dtype=np.int64)
    snap_columns = np.flatnonzero(tree_snap_numbers == snap)
    if snap_columns.size != 1:
        raise ValueError(f"Expected one all_selected_trees column for snap {snap}, found {snap_columns.size}.")
    tree_subhalo_ids = np.asarray(f["mpb_subhalo_ids"][:, snap_columns[0]], dtype=np.int64)
    tree_fof_ids = np.asarray(f["mpb_fof_ids"][:, snap_columns[0]], dtype=np.int64)

if tree_subhalo_ids.size != start_ids.size or tree_fof_ids.size != start_ids.size:
    raise ValueError("Snapshot-specific tree node arrays must have one row per start_subhalo_id.")
valid_tree_fof = (tree_fof_ids >= 0) & (tree_fof_ids < num_halos)
tree_rows = np.flatnonzero(valid_tree_fof & (tree_subhalo_ids >= 0))
tree_rows = tree_rows[tree_subhalo_ids[tree_rows] == central_subhalo_ids[tree_fof_ids[tree_rows]]]
tree_rows = tree_rows[mw_like[tree_fof_ids[tree_rows]]]
tree_mw_fof_ids = tree_fof_ids[tree_rows]
if np.unique(tree_mw_fof_ids).size != tree_mw_fof_ids.size:
    raise ValueError("Multiple selected start_ids map to the same MW-like central FoF halo at this snapshot.")
tree_mw_like = np.zeros(num_halos, dtype=bool)
tree_mw_like[tree_mw_fof_ids] = True

real_file = os.path.join(data_dir, f"real_cooling_radius_{snap}.hdf5")
if not os.path.isfile(real_file):
    raise FileNotFoundError(f"Generate real-profile cooling radii for the selected-tree sample first: {real_file}")
with h5py.File(real_file, "r") as f:
    real_fof_ids = np.asarray(f["fof_id"][:], dtype=np.int64)
    real_subhalo_ids = np.asarray(f["subhalo_id"][:], dtype=np.int64)
    real_radius = np.asarray(f["cooling_radius_r200c"][:], dtype=np.float64)
    real_analysis_valid = np.asarray(f["analysis_valid"][:], dtype=bool)
    real_status = np.asarray(f["status"][:], dtype=np.uint8)

real_rows = np.flatnonzero((real_fof_ids >= 0) & (real_fof_ids < num_halos) & (real_subhalo_ids >= 0))
real_rows = real_rows[real_subhalo_ids[real_rows] == central_subhalo_ids[real_fof_ids[real_rows]]]
real_rows = real_rows[tree_mw_like[real_fof_ids[real_rows]]]
if np.unique(real_fof_ids[real_rows]).size != real_rows.size:
    raise ValueError("Multiple selected-tree rows describe the same MW-like central FoF halo.")
tng_file_covered = np.zeros(num_halos, dtype=bool)
tng_file_covered[real_fof_ids[real_rows]] = True
real_rows = real_rows[real_analysis_valid[real_rows] & (real_status[real_rows] == 1) & np.isfinite(real_radius[real_rows]) & (real_radius[real_rows] > 0.0)]
if np.unique(real_fof_ids[real_rows]).size != real_rows.size:
    raise ValueError("Multiple successful TNG cooling-radius rows describe the same central FoF halo.")
tng_radius = np.full(num_halos, np.nan, dtype=np.float64)
tng_flag = np.zeros(num_halos, dtype=bool)
tng_radius[real_fof_ids[real_rows]] = real_radius[real_rows]
tng_flag[real_fof_ids[real_rows]] = True

subhalos = il.groupcat.loadSubhalos(basePath, snap, fields=["SubhaloSFR", "SubhaloMassType"])
subhalo_sfr = np.asarray(subhalos["SubhaloSFR"], dtype=np.float64)
subhalo_stellar_mass_msun = np.asarray(subhalos["SubhaloMassType"][:, 4], dtype=np.float64) * 1.0e10 / h

has_central = (central_subhalo_ids >= 0) & (central_subhalo_ids < subhalo_sfr.size)
central_sfr = np.full(num_halos, np.nan, dtype=np.float64)
central_stellar_mass_msun = np.full(num_halos, np.nan, dtype=np.float64)
central_sfr[has_central] = subhalo_sfr[central_subhalo_ids[has_central]]
central_stellar_mass_msun[has_central] = subhalo_stellar_mass_msun[central_subhalo_ids[has_central]]

valid_central_sfr = has_central & np.isfinite(central_sfr) & (central_sfr >= 0.0) & np.isfinite(central_stellar_mass_msun) & (central_stellar_mass_msun > 0.0)
base_sample = (
    tree_mw_like
    & valid_central_sfr
    & cold_flag
    & np.isfinite(cold_fraction)
    & (cold_fraction >= 0.0)
    & (cold_fraction <= 1.0)
)

tree_sample_absent_count = int(np.count_nonzero(base_sample & ~tng_file_covered))
tng_radius_unavailable_count = int(np.count_nonzero(base_sample & tng_file_covered & ~tng_flag))

lgal_sample = base_sample & lgal_flag & np.isfinite(lgal_radius) & (lgal_radius >= 0.0) & (lgal_radius <= 5.0)
tng_sample = base_sample & tng_flag & np.isfinite(tng_radius) & (tng_radius > 0.0) & (tng_radius <= 5.0)

# Map the classification catalogue onto FoF rows, independently of file row order.
classification_file = os.path.join(data_dir, f"ssfr_classification_{snap}.hdf5")
with h5py.File(classification_file, "r") as f:
    class_halo_ids = np.asarray(f["halo_id"][:], dtype=np.int64)
    class_central_ids = np.asarray(f["central_subhalo_id"][:], dtype=np.int64)
    class_codes = np.asarray(f["class_code"][:], dtype=np.int8)
    class_selection = np.asarray(f["selection_flag"][:], dtype=bool)

class_valid = class_selection & np.isin(class_codes, [0, 1, 2, 3]) & (class_central_ids == central_subhalo_ids[class_halo_ids])
ssfr_class = np.full(num_halos, -1, dtype=np.int8)
ssfr_class[class_halo_ids[class_valid]] = class_codes[class_valid]
starforming = np.isin(ssfr_class, [0, 1, 2])
quenched = ssfr_class == 3

# Compute log10(sSFR / yr^-1) from the central subhalo SFR and stellar mass.
log_ssfr = np.full(num_halos, np.nan, dtype=np.float64)
positive_sfr = base_sample & (central_sfr > 0.0)
zero_sfr = base_sample & (central_sfr == 0.0)
log_ssfr[positive_sfr] = np.log10(central_sfr[positive_sfr]) - np.log10(central_stellar_mass_msun[positive_sfr])
log_ssfr[zero_sfr] = ssfr_color_min
color_values = np.clip(log_ssfr, ssfr_color_min, ssfr_color_max)

fig = plt.figure(figsize=(20, 11), layout="constrained")
grid = fig.add_gridspec(3, 3, width_ratios=[1.0, 1.0, 0.035], height_ratios=[4.5, 1.5, 0.35], wspace=0.04, hspace=0.04)
colorbar_ax = fig.add_subplot(grid[0, 2])
legend_ax = fig.add_subplot(grid[2, :2])
legend_ax.set_axis_off()
main_axes = []
pdf_axes = []
method_panels = [("Lgal", lgal_radius, lgal_sample), ("TNG", tng_radius, tng_sample)]
cooling_radius_bins = np.linspace(0.0, 5.0, num_pdf_bins + 1)
color_norm = Normalize(vmin=ssfr_color_min, vmax=ssfr_color_max)

for column, (method, radius, sample) in enumerate(method_panels):
    ax = fig.add_subplot(grid[0, column], sharex=main_axes[0] if main_axes else None, sharey=main_axes[0] if main_axes else None)
    ax_cooling_pdf = fig.add_subplot(grid[1, column], sharex=ax, sharey=pdf_axes[0] if pdf_axes else None)
    main_axes.append(ax)
    pdf_axes.append(ax_cooling_pdf)
    scatter = ax.scatter(radius[sample], cold_fraction[sample], c=color_values[sample], cmap="turbo", norm=color_norm, marker="s", s=28, alpha=0.75, edgecolors="none", rasterized=True)

    starforming_sample = sample & starforming
    quenched_sample = sample & quenched
    number = int(np.count_nonzero(sample))
    starforming_count = int(np.count_nonzero(starforming_sample))
    quenched_count = int(np.count_nonzero(quenched_sample))
    unclassified_count = number - starforming_count - quenched_count

    # Normalize each group independently over the displayed 0-5 R200c range.
    for group_sample, color in ((starforming_sample, starforming_color), (quenched_sample, quenched_color)):
        if np.any(group_sample):
            ax_cooling_pdf.hist(radius[group_sample], bins=cooling_radius_bins, density=True, histtype="step", color=color, linewidth=pdf_linewidth)

    ax.set_title(method, fontsize=22)
    ax.set_xlim(0.0, 5.0)
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(which="both", labelbottom=False, labelleft=column == 0)
    ax_cooling_pdf.set_xlabel(r"$R_{\rm cool}/R_{\rm 200c}$", fontsize=19)
    ax_cooling_pdf.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax_cooling_pdf.tick_params(which="both", labelleft=column == 0)
    if column == 0:
        ax.set_ylabel(r"$M_{\rm cold}/M_{\rm gas}$", fontsize=19)
        ax_cooling_pdf.set_ylabel("PDF", fontsize=18)
    ax.text(0.04, 0.96, f"$N={number:,}$\nStarforming: {starforming_count:,}\nQuenched: {quenched_count:,}", transform=ax.transAxes, ha="left", va="top", fontsize=15, bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
    for panel in (ax, ax_cooling_pdf):
        panel.tick_params(which="both", labelsize=16)
        panel.grid(which="major", alpha=0.25)
        panel.set_axisbelow(True)
    print(f"{method}: all={number:,}, starforming={starforming_count:,}, quenched={quenched_count:,}, unclassified={unclassified_count:,} (scatter only).", flush=True)

# Fix the shared density range after both methods have contributed their peaks.
pdf_axes[0].set_ylim(bottom=0.0)
colorbar = fig.colorbar(scatter, cax=colorbar_ax)
colorbar.set_label(r"$\log_{10}(\mathrm{sSFR}/\mathrm{yr}^{-1})$", fontsize=18)
colorbar.ax.tick_params(labelsize=16)

pdf_handles = [Line2D([], [], color=starforming_color, linewidth=pdf_linewidth, label="Starforming (starburst + main sequence + green valley)"), Line2D([], [], color=quenched_color, linewidth=pdf_linewidth, label="Quenched")]
legend_ax.legend(handles=pdf_handles, loc="center", fontsize=16, frameon=False, ncol=2)

output_file = os.path.join(figure_dir, f"cf_cr_diagram_MW_centrals_ssfr_snap{snap}.pdf")
fig.savefig(output_file, format="pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Selected-tree MW-like parent sample with valid cold-gas fractions: {np.count_nonzero(base_sample):,}; Lgal plotted: {np.count_nonzero(lgal_sample):,}; TNG plotted: {np.count_nonzero(tng_sample):,}; absent from the TNG-radius file: {tree_sample_absent_count:,}; rows without a valid TNG radius: {tng_radius_unavailable_count:,}. Saved comparison to {output_file}", flush=True)
