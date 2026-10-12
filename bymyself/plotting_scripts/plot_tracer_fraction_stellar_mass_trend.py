import illustris_python as il
import matplotlib.pyplot as plt
import numpy as np
import h5py
from pathlib import Path
import sys
from os.path import isfile, isdir

plt.style.use('fancy_plots2.mplstyle')

dirname = "/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/pics"
assert isdir(dirname), 'output directory does not exist!'

start_snap = int(sys.argv[2])
target_snaps = np.asarray([snap for snap in sys.argv[3:]], dtype=np.int64)

basePath = "/public/share/chenhouzun/TNG50-1/output/"
header = il.groupcat.loadHeader(basePath, start_snap)
h = header["HubbleParam"]
all_stellar_masses = il.groupcat.loadSubhalos(basePath, start_snap, fields=["SubhaloMassInRadType"])[:, 4] * 1.0e10 / h

assert isfile(f"/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/gas_parent_classification_{start_snap}.hdf5"), 'Gas parent classification file does not exist!'
f = h5py.File(f"/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/gas_parent_classification_{start_snap}.hdf5")
sample_start_ids = np.asarray(f["sample_start_subhalo_id"][:], dtype=np.int64)
gas_parent_offsets = np.asarray(f["subhalo_offsets"][:], dtype=np.int64)

f.close()

assert isfile("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/all_selected_trees.hdf5"), 'Selected-tree file does not exist!'
f = h5py.File("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/all_selected_trees.hdf5", "r")
tree_snap_numbers = np.asarray(f["snap_numbers"][:], dtype=np.int64)
start_columns = np.flatnonzero(tree_snap_numbers == start_snap)
start_subhalo_ids = np.asarray(f["mpb_subhalo_ids"][:, start_columns[0]], dtype=np.int64)

type_history = None
tracers_per_parent = None
expected_n_tracers = None

for snap_col, target_snap in enumerate(target_snaps):
    assert isfile(f"/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/parent_indices_start_{start_snap}_target_{target_snap}.hdf5"), 'Parent indices file does not exist!'
    f = h5py.File(f"/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data/parent_indices_start_{start_snap}_target_{int(target_snap)}.hdf5")
    parent_indices = np.asarray(f[f"snap_0{target_snap}/parent_indices"][:], dtype=np.int64)
    current_tracers_per_parent = np.asarray(f[f"snap_0{target_snap}/numTracersInParents"][:], dtype=np.int64)
    current_n_tracers = current_tracers_per_parent.sum()

    f.close()

    if type_history is None:
        expected_n_tracers = current_n_tracers
        tracers_per_parent = current_tracers_per_parent
        type_history = np.full((expected_n_tracers, target_snaps.size), -1, dtype=np.int8)

    type_history[:, snap_col] = parent_indices[:, 1].astype(np.int8)

known = np.all((type_history >= 0) & (type_history <= 2), axis=1)
always_gas = known & np.all(type_history == 0, axis=1)
gas_star_only = known & np.all((type_history == 0) | (type_history == 1), axis=1) & np.any(type_history == 1, axis=1)
gas_bh_only = known & np.all((type_history == 0) | (type_history == 2), axis=1) & np.any(type_history == 2, axis=1)
other = known & ~(always_gas | gas_star_only | gas_bh_only)

tracer_class = np.full(expected_n_tracers, -1, dtype=np.int8)
tracer_class[always_gas] = 0
tracer_class[gas_star_only] = 1
tracer_class[gas_bh_only] = 2
tracer_class[other] = 3

# Convert gas-parent boundaries to tracer-row boundaries, then count categories per start subhalo.
parent_to_tracer_offsets = np.empty(tracers_per_parent.size + 1, dtype=np.int64)
parent_to_tracer_offsets[0] = 0
parent_to_tracer_offsets[1:] = np.cumsum(tracers_per_parent, dtype=np.int64)
tracer_offsets_by_subhalo = parent_to_tracer_offsets[gas_parent_offsets]

class_counts_by_subhalo = np.zeros((start_subhalo_ids.size, 4), dtype=np.int64)
known_tracers_by_subhalo = np.zeros(start_subhalo_ids.size, dtype=np.int64)
for tree_row in range(start_subhalo_ids.size):
    first = int(tracer_offsets_by_subhalo[tree_row])
    last = int(tracer_offsets_by_subhalo[tree_row + 1])
    tree_classes = tracer_class[first:last]
    known_tracers_by_subhalo[tree_row] = np.count_nonzero(tree_classes >= 0)
    for class_id in range(4):
        class_counts_by_subhalo[tree_row, class_id] = np.count_nonzero(tree_classes == class_id)

stellar_mass_msun = all_stellar_masses[start_subhalo_ids]
valid_mass = np.isfinite(stellar_mass_msun) & (stellar_mass_msun > 0)
log_stellar_mass = np.full(start_subhalo_ids.size, np.nan, dtype=np.float64)
log_stellar_mass[valid_mass] = np.log10(stellar_mass_msun[valid_mass])
valid_for_plot = valid_mass & (known_tracers_by_subhalo > 0)

# Use fixed-width bins in log stellar mass and pool tracer counts within each bin.
bin_width = 0.25
mass_min = np.floor(np.min(log_stellar_mass[valid_for_plot]) / bin_width) * bin_width
mass_max = np.ceil(np.max(log_stellar_mass[valid_for_plot]) / bin_width) * bin_width

mass_bin_edges = np.arange(mass_min, mass_max + 0.5 * bin_width, bin_width)

if mass_bin_edges[-1] < np.max(log_stellar_mass[valid_for_plot]):
    mass_bin_edges = np.append(mass_bin_edges, mass_bin_edges[-1] + bin_width)

mass_bin_centers = 0.5 * (mass_bin_edges[:-1] + mass_bin_edges[1:])

category_counts_by_bin = np.zeros((mass_bin_centers.size, 4), dtype=np.int64)
subhalo_counts_by_bin = np.histogram(log_stellar_mass[valid_for_plot], bins=mass_bin_edges)[0]
for class_id in range(4):
    category_counts_by_bin[:, class_id] = np.histogram(log_stellar_mass[valid_for_plot], bins=mass_bin_edges, weights=class_counts_by_subhalo[valid_for_plot, class_id])[0].astype(np.int64)

classified_counts_by_bin = category_counts_by_bin.sum(axis=1)
category_fractions_by_bin = np.zeros(category_counts_by_bin.shape, dtype=np.float64)
nonempty_bins = classified_counts_by_bin > 0
category_fractions_by_bin[nonempty_bins] = category_counts_by_bin[nonempty_bins] / classified_counts_by_bin[nonempty_bins, None]

fig, ax = plt.subplots(1, 1, figsize=(14, 8))
bar_width = 0.85 * bin_width
bar_bottom = np.zeros(mass_bin_centers.size, dtype=np.float64)
category_labels = ["Always gas", "Gas/star only", "Gas/BH only", "Other"]
category_colors = ["tab:blue", "tab:orange", "tab:green", "tab:gray"]

for class_id, (label, color) in enumerate(zip(category_labels, category_colors)):
    heights = category_fractions_by_bin[:, class_id]
    ax.bar(mass_bin_centers, heights, width=bar_width, bottom=bar_bottom, color=color, edgecolor="white", linewidth=0.5, label=label)
    bar_bottom += heights

plt.xlabel(r"stellar mass [$\log_{10}(M_\odot)$]")
plt.ylabel(r"$M_{\mathrm{tracers,inner}} (\mathrm{mode}) / M_{\mathrm{tracers,inner}}$")
plt.xlim(mass_bin_edges[0], mass_bin_edges[-1])
plt.ylim(0.0, 1.05)
plt.xticks(mass_bin_edges)

plt.tick_params(axis="x", rotation=45)
plt.legend(loc="best", frameon=False)
plt.tight_layout()
plt.savefig(dirname + f'/tracer_fraction_stellar_mass_trend_snap{start_snap}_for_snap{target_snaps[0]}_to_snap{target_snaps[-1]}.pdf', format = 'pdf')