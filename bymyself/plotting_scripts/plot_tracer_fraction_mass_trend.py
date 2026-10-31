from pathlib import Path
import h5py
import illustris_python as il
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np


project_root = Path(__file__).resolve().parents[1]
base_path = "/public/share/chenhouzun/TNG50-1/output/"
data_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")
figure_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/pics/tracer_fraction_halo_mass_trend")
bin_width = 0.25
line_width = 2.5
category_labels = ("Always gas", "Gas/star only", "Gas/BH only", "Other")
category_colors = ("#4477AA", "#EE7733", "#228833", "#BBBBBB")
panel_pairs = (
    ("Forward", ((44, 49, "dotted"), (64, 69, "dashed"), (94, 99, "solid"))),
    ("Backward", ((45, 40, "dotted"), (65, 60, "dashed"), (95, 90, "solid"))),
)

tree_file = data_dir / "all_selected_trees.hdf5"
with h5py.File(tree_file, "r") as f:
    start_ids = np.asarray(f["start_subhalo_id"][:], dtype=np.int64)
    snap_numbers = np.asarray(f["snap_numbers"][:], dtype=np.int64)
    tree_nodes = {}
    for _, pairs in panel_pairs:
        for start_snap, _, _ in pairs:
            columns = np.flatnonzero(snap_numbers == start_snap)
            column = int(columns[0])
            tree_nodes[start_snap] = (
                np.asarray(f["mpb_subhalo_ids"][:, column], dtype=np.int64),
                np.asarray(f["mpb_fof_ids"][:, column], dtype=np.int64),
            )
num_trees = start_ids.size

pair_results = {}
for _, pairs in panel_pairs:
    for start_snap, end_snap, _ in pairs:
        step = 1 if end_snap > start_snap else -1
        history_snaps = np.arange(start_snap + step, end_snap + step, step, dtype=np.int64)
        class_file = data_dir / f"gas_parent_classification_{start_snap}.hdf5"
        with h5py.File(class_file, "r") as f:
            gas_parent_offsets = np.asarray(f["subhalo_offsets"][:], dtype=np.int64)

        tracers_per_parent = None
        known = ever_star = ever_bh = None
        for target_snap in history_snaps:
            parent_file = data_dir / f"parent_indices_start_{start_snap}_target_{target_snap}.hdf5"
            with h5py.File(parent_file, "r") as f:
                group = f[f"snap_0{target_snap}"]
                parents = np.asarray(group["parent_indices"][:], dtype=np.int64)
                current_counts = np.asarray(group["numTracersInParents"][:], dtype=np.int64)
            num_tracers = int(current_counts.sum())
            if tracers_per_parent is None:
                tracers_per_parent = current_counts
                known = np.ones(num_tracers, dtype=bool)
                ever_star = np.zeros(num_tracers, dtype=bool)
                ever_bh = np.zeros(num_tracers, dtype=bool)
            parent_type = parents[:, 1]
            known &= (parents[:, 0] >= 0) & np.isin(parent_type, [0, 1, 2])
            ever_star |= parent_type == 1
            ever_bh |= parent_type == 2

        # Preserve the original four history classes and the known-tracer denominator.
        category_masks = (
            known & ~ever_star & ~ever_bh,
            known & ever_star & ~ever_bh,
            known & ~ever_star & ever_bh,
            known & ever_star & ever_bh,
        )
        parent_to_tracer_offsets = np.concatenate(([0], np.cumsum(tracers_per_parent, dtype=np.int64)))
        tracer_offsets = parent_to_tracer_offsets[gas_parent_offsets]
        category_counts = np.zeros((num_trees, 4), dtype=np.int64)
        for category, mask in enumerate(category_masks):
            prefix = np.concatenate(([0], np.cumsum(mask, dtype=np.int64)))
            category_counts[:, category] = prefix[tracer_offsets[1:]] - prefix[tracer_offsets[:-1]]
        known_counts = category_counts.sum(axis=1)
        unknown_counts = np.diff(tracer_offsets) - known_counts

        header = il.groupcat.loadHeader(base_path, start_snap)
        h = float(header["HubbleParam"])
        all_halo_mass = np.asarray(il.groupcat.loadHalos(base_path, start_snap, fields=["Group_M_Crit200"]), dtype=np.float64) * 1.0e10 / h
        node_ids, fof_ids = tree_nodes[start_snap]
        valid_node = node_ids >= 0
        if "Nsubgroups_Total" in header:
            valid_node &= node_ids < int(header["Nsubgroups_Total"])
        valid_fof = (fof_ids >= 0) & (fof_ids < all_halo_mass.size) & valid_node
        halo_mass = np.full(num_trees, np.nan, dtype=np.float64)
        halo_mass[valid_fof] = all_halo_mass[fof_ids[valid_fof]]
        valid_mass = np.isfinite(halo_mass) & (halo_mass > 0.0)
        log_halo_mass = np.full(num_trees, np.nan, dtype=np.float64)
        log_halo_mass[valid_mass] = np.log10(halo_mass[valid_mass])
        valid_for_plot = valid_mass & (known_counts > 0)
        result = {
            "history_snaps": history_snaps,
            "log_halo_mass": log_halo_mass,
            "category_counts_by_tree": category_counts,
            "known_tracers_by_tree": known_counts,
            "unknown_tracers_by_tree": unknown_counts,
            "valid_for_plot": valid_for_plot,
            "start_redshift": float(header["Redshift"]),
        }
        pair_results[(start_snap, end_snap)] = result

selected_masses = np.concatenate([result["log_halo_mass"][result["valid_for_plot"]] for result in pair_results.values()])

mass_min = np.floor(selected_masses.min() / bin_width) * bin_width
mass_max = np.ceil(selected_masses.max() / bin_width) * bin_width
if mass_max <= mass_min:
    mass_max = mass_min + bin_width
mass_bin_edges = np.arange(mass_min, mass_max + 0.5 * bin_width, bin_width)
mass_bin_centers = 0.5 * (mass_bin_edges[:-1] + mass_bin_edges[1:])
for result in pair_results.values():
    sample = result["valid_for_plot"]
    counts_by_bin = np.column_stack([
        np.histogram(result["log_halo_mass"][sample], bins=mass_bin_edges, weights=result["category_counts_by_tree"][sample, category])[0]
        for category in range(4)
    ])
    total_by_bin = counts_by_bin.sum(axis=1)
    fractions_by_bin = np.full(counts_by_bin.shape, np.nan, dtype=np.float64)
    np.divide(counts_by_bin, total_by_bin[:, None], out=fractions_by_bin, where=total_by_bin[:, None] > 0)
    result["mass_bin_centers"] = mass_bin_centers
    result["category_counts_by_bin"] = counts_by_bin
    result["category_fractions_by_bin"] = fractions_by_bin

plt.style.use(project_root / "fancy_plots2.mplstyle")
fig, axes = plt.subplots(1, 2, figsize=(20, 9), sharex=True, sharey=True)
for ax, (title, pairs) in zip(axes, panel_pairs):
    ax.axvspan(10.8, 11.2, color="lightgray", alpha=0.3, zorder=0)
    ax.axvspan(11.8, 12.2, color="lightgray", alpha=0.3, zorder=0)
    ax.axvspan(12.6, 13.0, color="lightgray", alpha=0.3, zorder=0)
    redshift_handles = []
    for start_snap, end_snap, linestyle in pairs:
        result = pair_results[(start_snap, end_snap)]
        for category, color in enumerate(category_colors):
            ax.plot(mass_bin_centers, result["category_fractions_by_bin"][:, category], color=color, linestyle=linestyle, linewidth=line_width)
        redshift_handles.append(mlines.Line2D([], [], color="black", linestyle=linestyle, linewidth=line_width,
            label=rf"$z_{{\rm start}}={result['start_redshift']:.2f}$ ({start_snap}→{end_snap})"))
    ax.set_title(title, fontsize=20)
    ax.set_xlabel(r"halo mass [$\log_{10}(M_{200c}/M_\odot)$]", fontsize=19)
    ax.set_xlim(10.6, 13.6)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks([11, 12, 13])
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=16)
    ax.grid(which="major", axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(handles=redshift_handles, loc="upper right", fontsize=13, frameon=True, facecolor="white", framealpha=0.8, edgecolor="none")
axes[0].set_ylabel("Classified tracer fraction", fontsize=19)
category_handles = [mlines.Line2D([], [], color=color, linewidth=line_width, label=label) for label, color in zip(category_labels, category_colors)]
fig.legend(handles=category_handles, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=4, frameon=False, fontsize=16)
fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
figure_dir.mkdir(parents=True, exist_ok=True)
output_file = figure_dir / "tracer_fraction_halo_mass_trend_forward_backward.pdf"
fig.savefig(output_file, format="pdf", bbox_inches="tight")
plt.close(fig)