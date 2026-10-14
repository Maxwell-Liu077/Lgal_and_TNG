import illustris_python as il
import matplotlib.pyplot as plt
import numpy as np
import h5py
from pathlib import Path

plt.style.use("fancy_plots2.mplstyle")

basePath = "/public/share/chenhouzun/TNG50-1/output/"
data_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")
figure_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/pics/tracer_fraction_stellar_mass_trend")
figure_dir.mkdir(parents=True, exist_ok=True)

snap_backward = 94
snap_forward = 95
bin_width = 0.25

fraction_file = data_dir / "distance_cats" / f"tracer_cooling_rate_modified_{snap_backward}_to_{snap_forward}.hdf5"
tree_file = data_dir / "all_selected_trees.hdf5"

with h5py.File(fraction_file, "r") as f:
    in_fraction = np.asarray(f["in_fraction"][:], dtype=np.float64)
    out_fraction = np.asarray(f["out_fraction"][:], dtype=np.float64)

with h5py.File(tree_file, "r") as f:
    start_subhalo_ids = np.asarray(f["start_subhalo_id"][:], dtype=np.int64)
    snap_numbers = np.asarray(f["snap_numbers"][:], dtype=np.int64)
    backward_columns = np.flatnonzero(snap_numbers == snap_backward)
    forward_columns = np.flatnonzero(snap_numbers == snap_forward)
    subhalo_ids_backward = np.asarray(f["mpb_subhalo_ids"][:, backward_columns[0]], dtype=np.int64)
    subhalo_ids_forward = np.asarray(f["mpb_subhalo_ids"][:, forward_columns[0]], dtype=np.int64)

num_trees = start_subhalo_ids.size

header_backward = il.groupcat.loadHeader(basePath, snap_backward)
header_forward = il.groupcat.loadHeader(basePath, snap_forward)
h_backward = float(header_backward["HubbleParam"])
h_forward = float(header_forward["HubbleParam"])

stellar_mass_type_backward = il.groupcat.loadSubhalos(basePath, snap_backward, fields=["SubhaloMassInRadType"])
stellar_mass_type_forward = il.groupcat.loadSubhalos(basePath, snap_forward, fields=["SubhaloMassInRadType"])

stellar_mass_type_backward = np.asarray(stellar_mass_type_backward, dtype=np.float64)
stellar_mass_type_forward = np.asarray(stellar_mass_type_forward, dtype=np.float64)

valid_backward_nodes = ((subhalo_ids_backward >= 0) & (subhalo_ids_backward < stellar_mass_type_backward.shape[0]))
valid_forward_nodes = ((subhalo_ids_forward >= 0) & (subhalo_ids_forward < stellar_mass_type_forward.shape[0]))

# Use each snapshot's central-subhalo stellar mass for its corresponding panel.
stellar_mass_msun_backward = np.full(num_trees, np.nan, dtype=np.float64)
stellar_mass_msun_forward = np.full(num_trees, np.nan, dtype=np.float64)
stellar_mass_msun_backward[valid_backward_nodes] = (stellar_mass_type_backward[subhalo_ids_backward[valid_backward_nodes], 4] * 1.0e10 / h_backward)
stellar_mass_msun_forward[valid_forward_nodes] = (stellar_mass_type_forward[subhalo_ids_forward[valid_forward_nodes], 4] * 1.0e10 / h_forward)

valid_mass_backward = np.isfinite(stellar_mass_msun_backward) & (stellar_mass_msun_backward > 0.0)
valid_mass_forward = np.isfinite(stellar_mass_msun_forward) & (stellar_mass_msun_forward > 0.0)
log_stellar_mass_backward = np.full(num_trees, np.nan, dtype=np.float64)
log_stellar_mass_forward = np.full(num_trees, np.nan, dtype=np.float64)
log_stellar_mass_backward[valid_mass_backward] = np.log10(stellar_mass_msun_backward[valid_mass_backward])
log_stellar_mass_forward[valid_mass_forward] = np.log10(stellar_mass_msun_forward[valid_mass_forward])

# Rows with incomplete histories have fractions summing to less than one; exclude them from 100% stacks.
in_complete = np.all(np.isfinite(in_fraction), axis=1) & np.isclose(in_fraction.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-6)
out_complete = np.all(np.isfinite(out_fraction), axis=1) & np.isclose(out_fraction.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-6)
valid_in = valid_mass_forward & in_complete
valid_out = valid_mass_backward & out_complete

def make_mass_bin_edges(log_mass, valid_rows):
    selected_mass = log_mass[valid_rows]
    mass_min = np.floor(np.min(selected_mass) / bin_width) * bin_width
    mass_max = np.ceil(np.max(selected_mass) / bin_width) * bin_width
    if mass_max <= mass_min:
        mass_max = mass_min + bin_width
    edges = np.arange(mass_min, mass_max + 0.5 * bin_width, bin_width)
    if edges.size < 2:
        edges = np.array([mass_min, mass_min + bin_width], dtype=np.float64)
    if edges[-1] < np.max(selected_mass):
        edges = np.append(edges, edges[-1] + bin_width)
    return edges

in_mass_bin_edges = make_mass_bin_edges(log_stellar_mass_forward, valid_in)
out_mass_bin_edges = make_mass_bin_edges(log_stellar_mass_backward, valid_out)
in_mass_bin_centers = 0.5 * (in_mass_bin_edges[:-1] + in_mass_bin_edges[1:])
out_mass_bin_centers = 0.5 * (out_mass_bin_edges[:-1] + out_mass_bin_edges[1:])

def mean_fraction_in_mass_bins(log_mass, fractions, valid_rows, mass_bin_edges):
    mass_bin_centers = 0.5 * (mass_bin_edges[:-1] + mass_bin_edges[1:])
    number_per_bin = np.histogram(log_mass[valid_rows], bins=mass_bin_edges)[0]
    mean_fraction = np.full((mass_bin_centers.size, fractions.shape[1]), np.nan, dtype=np.float64)
    for category in range(fractions.shape[1]):
        fraction_sum = np.histogram(log_mass[valid_rows], bins=mass_bin_edges, weights=fractions[valid_rows, category])[0]
        np.divide(fraction_sum, number_per_bin, out=mean_fraction[:, category], where=number_per_bin > 0)
    return mean_fraction

in_fraction_by_bin = mean_fraction_in_mass_bins(log_stellar_mass_forward, in_fraction, valid_in, in_mass_bin_edges)
out_fraction_by_bin = mean_fraction_in_mass_bins(log_stellar_mass_backward, out_fraction, valid_out, out_mass_bin_edges)

fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharey=True)
category_colors = ["#4477AA", "#EE7733", "#228833"]
panel_data = [
    (axes[0], in_mass_bin_centers, in_mass_bin_edges, in_fraction_by_bin, ["First in", "Recycled in", "Former in"], f"In fraction (snap {snap_forward})"),
    (axes[1], out_mass_bin_centers, out_mass_bin_edges, out_fraction_by_bin, ["Stay out", "Recycled out", "Stay in"], f"Out fraction (snap {snap_backward})"),
]

for ax, mass_bin_centers, mass_bin_edges, fractions_by_bin, category_labels, title in panel_data:
    ax.stackplot(mass_bin_centers, fractions_by_bin.T, labels=category_labels, colors=category_colors, edgecolor="black", linewidth=0.6)
    ax.set_title(title)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Mean tracer fraction per tree")
    ax.legend(loc="upper right", frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.25)
    integer_ticks = np.arange(int(np.ceil(mass_bin_edges[0])), int(np.floor(mass_bin_edges[-1])) + 1, dtype=int)
    ax.set_xlim(mass_bin_edges[0], mass_bin_edges[-1])
    if integer_ticks.size:
        ax.set_xticks(integer_ticks)

axes[0].set_xlabel(rf"stellar mass [$\log_{10}(M_\odot)$]")
axes[1].set_xlabel(rf"stellar mass [$\log_{10}(M_\odot)$]")
for ax in axes:
    ax.tick_params(axis="x", rotation=0)
fig.tight_layout()
output_file = figure_dir / f"modified_tracer_fractions_stellar_mass_{snap_backward}_to_{snap_forward}.pdf"
fig.savefig(output_file, format="pdf")
plt.close(fig)
