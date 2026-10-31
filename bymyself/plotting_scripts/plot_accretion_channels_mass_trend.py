from pathlib import Path

import h5py
import illustris_python as il
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

import funcs


plt.style.use("fancy_plots2.mplstyle")

num_bins = 25
base_path = "/public/share/chenhouzun/TNG50-1/output/"
project_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself")
data_dir = project_dir / "pics"
tree_file = data_dir / "all_selected_trees.hdf5"
distance_dir = data_dir / "distance_cats"
output_dir = data_dir / "accretion_channels"
output_dir.mkdir(parents=True, exist_ok=True)

# Each pair is (start snapshot, target snapshot). The two panels compare
# forward-time and backward-time tracer classifications over three epochs.
panel_pairs = (
    ("start snap < target snap", ((94, 99), (64, 69), (44, 49))),
    ("start snap > target snap", ((95, 90), (65, 60), (45, 40))),
)
line_styles = ("solid", "dashed", "dotted")
class_ids = np.asarray([0, 1, 2, 3, 4], dtype=np.int8)
category_labels = ("Inner Cold", "Outer Cold", "Inner Hot", "Outer Hot", "IGM")
category_colors = ("tab:blue", "tab:purple", "tab:orange", "tab:red", "gold")
galaxy_mass_ranges = ((10.8, 11.2), (11.8, 12.2), (12.6, 13.4))
halo_mass_xlim = (10.6, 13.6)

all_pairs = tuple(pair for _, pairs in panel_pairs for pair in pairs)
required_snaps = sorted({snap for pair in all_pairs for snap in pair})

with h5py.File(tree_file, "r") as f:
    tree_start_ids = np.asarray(f["start_subhalo_id"][:], dtype=np.int64)
    snap_numbers = np.asarray(f["snap_numbers"][:], dtype=np.int64)
    tree_fof_ids_by_snap = {}
    for snap in required_snaps:
        columns = np.flatnonzero(snap_numbers == snap)
        tree_fof_ids_by_snap[snap] = np.asarray(f["mpb_fof_ids"][:, int(columns[0])], dtype=np.int64)

num_trees = tree_start_ids.size
class_trends = {}
redshift_labels = {}

for panel_index, (panel_title, pairs) in enumerate(panel_pairs):
    for epoch_index, (start_snap, target_snap) in enumerate(pairs):
        header_start = il.groupcat.loadHeader(base_path, start_snap)
        hubble_param = float(header_start["HubbleParam"])
        group_mass_raw = np.asarray(il.groupcat.loadHalos(base_path, start_snap, fields=["Group_M_Crit200"]), dtype=np.float64)
        group_mass_msun = group_mass_raw * 1.0e10 / hubble_param

        fof_ids = tree_fof_ids_by_snap[start_snap]
        halo_mass_msun = np.full(num_trees, np.nan, dtype=np.float64)
        valid_fof = (fof_ids >= 0) & (fof_ids < group_mass_msun.size)
        valid_rows = np.flatnonzero(valid_fof)
        halo_mass_msun[valid_rows] = group_mass_msun[fof_ids[valid_rows]]
        valid_mass = np.isfinite(halo_mass_msun) & (halo_mass_msun > 0.0)
        log_halo_mass = np.full(num_trees, np.nan, dtype=np.float64)
        log_halo_mass[valid_mass] = np.log10(halo_mass_msun[valid_mass])

        distance_file = distance_dir / f"distance_cats_start_{start_snap}_target_{target_snap}.hdf5"
        with h5py.File(distance_file, "r") as f:
            file_start_ids = np.asarray(f["start_subhalo_id"][:], dtype=np.int64)
            offsets = np.asarray(f["tree_tracer_offsets"][:], dtype=np.int64)
            tracer_classes = np.asarray(f["tracers_class"][:], dtype=np.int8)
            tree_valid = np.asarray(f["tree_valid"][:], dtype=bool)

        class_counts = np.zeros((num_trees, class_ids.size), dtype=np.int64)
        tracer_counts_by_tree = np.diff(offsets)
        for tree_row in range(num_trees):
            first, last = offsets[tree_row], offsets[tree_row + 1]
            classes_for_tree = tracer_classes[first:last]
            for class_index, class_id in enumerate(class_ids):
                class_counts[tree_row, class_index] = np.count_nonzero(classes_for_tree == class_id)

        has_tracers = tracer_counts_by_tree > 0
        fractions = np.full((num_trees, class_ids.size), np.nan, dtype=np.float64)
        fractions[has_tracers] = class_counts[has_tracers] / tracer_counts_by_tree[has_tracers, None]
        valid_trees = tree_valid & has_tracers & valid_mass
        mass_values = log_halo_mass[valid_mass]
        mass_bins = np.full(num_bins, np.nan, dtype=np.float64)
        median_fractions = np.full((class_ids.size, num_bins), np.nan, dtype=np.float64)

        for class_index in range(class_ids.size):
            class_fraction = fractions[valid_mass, class_index].copy()
            class_fraction[~valid_trees[valid_mass]] = np.nan
            x_median, y_median, _, _ = funcs.binData_med(mass_values, class_fraction, num_bins)
            if class_index == 0:
                mass_bins = x_median
            median_fractions[class_index] = y_median

        target_redshift = float(il.groupcat.loadHeader(base_path, target_snap)["Redshift"])
        class_trends[(start_snap, target_snap)] = {
            "mass_bins": mass_bins,
            "median_fractions": median_fractions,
            "valid_tree_count": int(np.count_nonzero(valid_trees)),
        }
        redshift_labels[(start_snap, target_snap)] = f"$z={target_redshift:.2f}$"
        print(f"{panel_title}, {start_snap}->{target_snap}: {np.count_nonzero(valid_trees):,}/{num_trees:,} valid trees.", flush=True)

fig, axes = plt.subplots(1, 2, figsize=(20, 9), sharex=True, sharey=True, layout="constrained")
category_handles = [mpatches.Patch(color=color, label=label) for label, color in zip(category_labels, category_colors)]

for ax, (panel_title, pairs) in zip(axes, panel_pairs):
    for left, right in galaxy_mass_ranges:
        ax.axvspan(left, right, color="lightgray", alpha=0.25, zorder=0)

    redshift_handles = []
    for epoch_index, (start_snap, target_snap) in enumerate(pairs):
        linestyle = line_styles[epoch_index]
        trend = class_trends[(start_snap, target_snap)]
        for class_index, (color, label) in enumerate(zip(category_colors, category_labels)):
            ax.plot(trend["mass_bins"], trend["median_fractions"][class_index], color=color, linestyle=linestyle, linewidth=2.2, zorder=3)
        redshift_handles.append(mlines.Line2D([], [], color="black", linestyle=linestyle, linewidth=2.2,
            label=f"{redshift_labels[(start_snap, target_snap)]} ({start_snap}→{target_snap})"))

    category_legend = ax.legend(handles=category_handles, loc="upper right", fontsize=11, frameon=True, facecolor="white", framealpha=0.8, edgecolor="none")
    redshift_legend = ax.legend(handles=redshift_handles, loc="upper left", fontsize=11, frameon=True, facecolor="white", framealpha=0.8, edgecolor="none")
    ax.add_artist(category_legend)
    ax.set_title(panel_title, fontsize=19)
    ax.set_xlabel(r"halo mass [$\log_{10}(M_{200c}/M_\odot)$]", fontsize=17)
    ax.set_xlim(*halo_mass_xlim)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([11, 12, 13])
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=15)
    ax.grid(which="major", axis="y", alpha=0.25)
    ax.set_axisbelow(True)

axes[0].set_ylabel(r"Mass fraction $f_{bar}$", fontsize=17)
output_file = output_dir / "accretion_channels_halo_mass_trend_forward_backward.pdf"
fig.savefig(output_file, format="pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved accretion-channel halo-mass trends to {output_file}", flush=True)
