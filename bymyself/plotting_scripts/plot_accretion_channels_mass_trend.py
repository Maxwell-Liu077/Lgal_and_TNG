import sys
from pathlib import Path
from os.path import isfile, isdir
import h5py
import illustris_python as il
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import funcs

plt.style.use("fancy_plots2.mplstyle")

start_snap = int(sys.argv[1])
target_snaps = np.asarray([snap for snap in sys.argv[2:]], dtype=np.int64)

numBins = 25
basePath = "/public/share/chenhouzun/TNG50-1/output/"
data_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/pics")
tree_file = data_dir / "all_selected_trees.hdf5"
distance_dir = data_dir / "distance_cats"
output_dir = data_dir / "accretion_channels"
output_dir.mkdir(parents=True, exist_ok=True)

# Read the selected-tree row IDs and the central-subhalo IDs at start_snap.
with h5py.File(tree_file, "r") as f:
    tree_start_ids = np.asarray(f["start_subhalo_id"][:], dtype=np.int64)
    snap_numbers = np.asarray(f["snap_numbers"][:], dtype=np.int64)
    start_columns = np.flatnonzero(snap_numbers == start_snap)
    start_node_subhalo_ids = np.asarray(f["mpb_subhalo_ids"][:, start_columns[0]], dtype=np.int64)

num_trees = tree_start_ids.size

# Match the stellar-mass definition used by plot_accretion_channels.py.
header = il.groupcat.loadHeader(basePath, start_snap)
h_const = float(header["HubbleParam"])
all_stellar_masses = il.groupcat.loadSubhalos(basePath, start_snap, fields=["SubhaloMassType"])[:, 4] * 1.0e10 / h_const

stellar_mass_msun = np.full(num_trees, np.nan, dtype=np.float64)
stellar_mass_msun = all_stellar_masses[start_node_subhalo_ids]
valid_mass = np.isfinite(stellar_mass_msun) & (stellar_mass_msun > 0.0)
log_stellar_mass = np.full(num_trees, np.nan, dtype=np.float64)
log_stellar_mass[valid_mass] = np.log10(stellar_mass_msun[valid_mass])

# Use classes 1--4, matching tracer_cooling_rate_direct. Class 0 and -1
# are excluded from the denominator.
class_ids = np.asarray([0, 1, 2, 3, 4], dtype=np.int8)
category_labels = ["Inner Cold", "Outer Cold", "Inner Hot", "Outer Hot", "IGM"]
category_colors = ["tab:blue", "tab:purple", "tab:orange", "tab:red", "gold"]
linestyle_cycle = ["solid", "dashed"]

fractions_by_target = np.full((target_snaps.size, num_trees, class_ids.size), np.nan, dtype=np.float64)
valid_tree_by_target = np.zeros((target_snaps.size, num_trees), dtype=bool)

for target_index, target_snap in enumerate(target_snaps):
    distance_file = distance_dir / f"distance_cats_start_{start_snap}_target_{target_snap}.hdf5"

    with h5py.File(distance_file, "r") as f:
        file_start_ids = np.asarray(f["start_subhalo_id"][:], dtype=np.int64)
        tree_tracer_offsets = np.asarray(f["tree_tracer_offsets"][:], dtype=np.int64)
        tracer_classes = np.asarray(f["tracers_class"][:], dtype=np.int8)
        tree_valid = np.asarray(f["tree_valid"][:], dtype=bool)

    class_counts = np.zeros((num_trees, class_ids.size), dtype=np.int64)
    tracer_counts_by_tree = np.diff(tree_tracer_offsets)

    for tree_row in range(num_trees):
        first = tree_tracer_offsets[tree_row]
        last = tree_tracer_offsets[tree_row + 1]
        classes_for_tree = tracer_classes[first:last]

        for class_index, class_id in enumerate(class_ids):
            class_counts[tree_row, class_index] = np.count_nonzero(classes_for_tree == class_id)

    has_tracers = tracer_counts_by_tree > 0
    fractions_by_target[target_index, has_tracers] = class_counts[has_tracers] / tracer_counts_by_tree[has_tracers, None]
    valid_tree_by_target[target_index] = tree_valid & has_tracers & valid_mass

mass_values = log_stellar_mass[valid_mass]
mass_bins = funcs.binData_med(mass_values, np.zeros(mass_values.size), numBins)[0]
fraction_bins = np.full((target_snaps.size, class_ids.size, numBins), np.nan, dtype=np.float64)

for target_index in range(target_snaps.size):
    valid_trees = valid_tree_by_target[target_index, valid_mass]

    for class_index in range(class_ids.size):
        fractions = fractions_by_target[target_index, valid_mass, class_index].copy()
        fractions[~valid_trees] = np.nan
        _, y_median, _, _ = funcs.binData_med(mass_values, fractions, numBins)
        fraction_bins[target_index, class_index] = y_median

fig, ax = plt.subplots(1, 1, figsize=(16, 9))

for target_index, target_snap in enumerate(target_snaps):
    linestyle = linestyle_cycle[target_index]

    for class_index, (color, label) in enumerate(zip(category_colors, category_labels)):
        plt.plot(mass_bins, fraction_bins[target_index, class_index], color=color, linestyle=linestyle, label = label)

category_handles = [mpatches.Patch(color=color, label=label) for label, color in zip(category_labels, category_colors)]
snapshot_handles = [mlines.Line2D([],[],color="tab:gray",linestyle=linestyle_cycle[i],label=f"snap {snap}")for i, snap in enumerate(target_snaps)]

category_legend = ax.legend(handles=category_handles, loc="upper right")
ax.legend(handles=snapshot_handles, loc="upper left")
ax.add_artist(category_legend)

ax.set_xlabel(r"stellar mass [$\log_{10}(M_\odot)$]")
ax.set_ylabel(r"Mass fraction $f_{bar}$")
ax.set_xlim(np.min(mass_values), np.max(mass_values))
ax.set_ylim(-0.03, 1.03)

integer_ticks = np.arange(int(np.ceil(np.min(mass_values))), int(np.floor(np.max(mass_values))) + 1, dtype=int)
if integer_ticks.size:
    ax.set_xticks(integer_ticks)

fig.tight_layout()
output_file = output_dir / f"accretion_stellar_mass_trend_snap{start_snap}_for_snap{target_snaps[0]}_to_snap{target_snaps[-1]}.pdf"
plt.savefig(output_file, format = 'pdf')