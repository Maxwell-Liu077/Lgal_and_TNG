from pathlib import Path
import warnings

import funcs
import h5py
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("fancy_plots2.mplstyle")

data_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")
distance_dir = data_dir / "distance_cats"
output_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/pics/cooling_rates")
output_dir.mkdir(parents=True, exist_ok=True)

start_snap_backward = 94
start_snap_forward = 95
num_bins = 25

lgal_file = data_dir / f"Lgal_cooling_rate_{start_snap_backward}_and_{start_snap_forward}.hdf5"
in_file = distance_dir / f"tcrd_{start_snap_backward}_to_{start_snap_forward}.hdf5"
out_file = distance_dir / f"tord_{start_snap_backward}_to_{start_snap_forward}.hdf5"
modified_file = distance_dir / f"tracer_cooling_rate_modified_{start_snap_backward}_to_{start_snap_forward}.hdf5"
tree_file = data_dir / "all_selected_trees.hdf5"

required_files = [lgal_file, in_file, out_file, modified_file, tree_file]
missing_files = [path for path in required_files if not path.is_file()]
if missing_files:
    raise FileNotFoundError("Missing cooling-rate inputs:\n" + "\n".join(map(str, missing_files)))

with h5py.File(tree_file, "r") as f:
    start_subhalo_ids = np.asarray(f["start_subhalo_id"][:], dtype=np.int64)
    num_trees = start_subhalo_ids.size

with h5py.File(lgal_file, "r") as f:
    lgal_halo_mass = np.asarray(f["halo_mass"][:], dtype=np.float64)
    lgal_rate_msun_per_yr = np.asarray(f["cooling_rate"][:], dtype=np.float64)
    lgal_valid = np.asarray(f["valid"][:], dtype=bool)

with h5py.File(in_file, "r") as f:
    in_halo_mass = np.asarray(f["halo_mass"][:], dtype=np.float64)
    in_rate_msun_per_gyr = np.asarray(f["cooling_rate"][:], dtype=np.float64)

with h5py.File(out_file, "r") as f:
    out_halo_mass = np.asarray(f["halo_mass"][:], dtype=np.float64)
    out_rate_msun_per_gyr = np.asarray(f["out_rate"][:], dtype=np.float64)

with h5py.File(modified_file, "r") as f:
    modified_halo_mass = np.asarray(f["halo_mass"][:], dtype=np.float64)
    modified_rate_msun_per_gyr = np.asarray(f["modified_rate_msun_per_gyr"][:], dtype=np.float64)
    ssfr_class_backward = np.asarray(f["ssfr_class_backward"][:], dtype=np.int8)
    ssfr_class_forward = np.asarray(f["ssfr_class_forward"][:], dtype=np.int8)

tree_ordered_arrays = {
    "Lgal halo_mass": lgal_halo_mass,
    "Lgal cooling_rate": lgal_rate_msun_per_yr,
    "Lgal valid": lgal_valid,
    "direct halo_mass": in_halo_mass,
    "direct cooling_rate": in_rate_msun_per_gyr,
    "out halo_mass": out_halo_mass,
    "out_rate": out_rate_msun_per_gyr,
    "modified halo_mass": modified_halo_mass,
    "modified_rate": modified_rate_msun_per_gyr,
    "ssfr_class_backward": ssfr_class_backward,
    "ssfr_class_forward": ssfr_class_forward,
}


# The three tracer rates are stored in Msun/Gyr; Lgal is already in Msun/yr.
in_rate_msun_per_yr = in_rate_msun_per_gyr / 1.0e9
out_rate_msun_per_yr = out_rate_msun_per_gyr / 1.0e9
direct_rate_msun_per_yr = in_rate_msun_per_yr + out_rate_msun_per_yr
modified_rate_msun_per_yr = modified_rate_msun_per_gyr / 1.0e9

# Use the direct-rate halo masses as the common tree-ordered x array for all curves.
valid_mass_rows = np.isfinite(in_halo_mass) & (in_halo_mass > 0.0)
if not np.any(valid_mass_rows):
    raise ValueError("No finite positive halo masses are available for plotting.")

tree_rows_for_binning = np.flatnonzero(valid_mass_rows)
log_mass_for_binning = np.log10(in_halo_mass[valid_mass_rows])
plot_mass_min = float(np.min(log_mass_for_binning))
plot_mass_max = float(np.max(log_mass_for_binning))
if plot_mass_min == plot_mass_max:
    plot_mass_min -= 0.5
    plot_mass_max += 0.5


def binned_median_log_rate(halo_mass, rate_msun_per_yr, sample_mask):
    """Return median log10 rate per mass bin using funcs.binData_med."""
    valid_rate = (
        sample_mask[tree_rows_for_binning]
        & np.isfinite(halo_mass[tree_rows_for_binning])
        & (halo_mass[tree_rows_for_binning] > 0.0)
        & np.isfinite(rate_msun_per_yr[tree_rows_for_binning])
        & (rate_msun_per_yr[tree_rows_for_binning] > 0.0)
    )
    log_rate = np.full(log_mass_for_binning.size + 1, np.nan, dtype=np.float64)
    log_rate[:-1][valid_rate] = np.log10(rate_msun_per_yr[tree_rows_for_binning][valid_rate])

    # binData_med uses x < max(x), so append an NaN-y sentinel just above the maximum
    # to include the largest-mass tree without changing its measured rate.
    x_for_binning = np.append(log_mass_for_binning, np.nextafter(np.max(log_mass_for_binning), np.inf))
    if np.min(log_mass_for_binning) == np.max(log_mass_for_binning):
        if np.any(valid_rate):
            return np.asarray([log_mass_for_binning[0]]), np.asarray([np.median(log_rate[:-1][valid_rate])])
        return np.asarray([log_mass_for_binning[0]]), np.asarray([np.nan])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        x_median, y_median, _, _ = funcs.binData_med(x_for_binning, log_rate, num_bins)

    return x_median, y_median

def finish_plot(ax, output_file):
    ax.set_xlabel(r"$\log_{10}(M_{200c}/M_\odot)$")
    ax.set_ylabel(r"$\log_{10}[\dot{M}_{\rm cool}/(M_\odot\,\mathrm{yr}^{-1})]$")
    ax.set_xlim(plot_mass_min, plot_mass_max)
    integer_ticks = np.arange(int(np.ceil(plot_mass_min)), int(np.floor(plot_mass_max)) + 1, dtype=int)
    if integer_ticks.size:
        ax.set_xticks(integer_ticks)
    ax.grid(which="major", axis="both", alpha=0.25)
    fig = ax.figure
    fig.tight_layout()
    fig.savefig(output_file, format="pdf")
    plt.close(fig)


# Figure 1: compare the galaxy cooling estimate with the two tracer estimates.
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
comparison_series = [
    ("Lgal cooling", lgal_halo_mass, lgal_rate_msun_per_yr, lgal_valid, "C0", "solid"),
    ("Tracer cooling (direct)", in_halo_mass, direct_rate_msun_per_yr, np.ones(num_trees, dtype=bool), "C1", "solid"),
    ("Tracer cooling (modified)", modified_halo_mass, modified_rate_msun_per_yr, np.ones(num_trees, dtype=bool), "C2", "solid"),
]
for label, halo_mass, rate, sample_mask, color, linestyle in comparison_series:
    x, y = binned_median_log_rate(halo_mass, rate, sample_mask)
    ax.plot(x, y, color=color, linestyle=linestyle, label=label)
comparison_handles = [mlines.Line2D([], [], color=color, linestyle=linestyle, label=label) for label, _, _, _, color, linestyle in comparison_series]
ax.legend(handles=comparison_handles, loc="lower right", frameon=False, fontsize=30, labelspacing=0.6, borderpad=0.5)
finish_plot(ax, output_dir / f"cooling_rate_comparison_{start_snap_backward}_{start_snap_forward}.pdf")

# Figure 2: compare modified, inflow, and outflow rates by sSFR group.
starforming_backward = np.isin(ssfr_class_backward, [0, 1])
quenched_backward = np.isin(ssfr_class_backward, [2, 3])
starforming_forward = np.isin(ssfr_class_forward, [0, 1])
quenched_forward = np.isin(ssfr_class_forward, [2, 3])
starforming_color = "#fd7825"
quenched_color = "#440154"

fig, ax = plt.subplots(1, 1, figsize=(16, 9))
combined_series = [
    (modified_halo_mass, modified_rate_msun_per_yr, starforming_backward, starforming_color, "solid"),
    (modified_halo_mass, modified_rate_msun_per_yr, quenched_backward, quenched_color, "solid"),
    (in_halo_mass, in_rate_msun_per_yr, starforming_backward, starforming_color, "dashed"),
    (in_halo_mass, in_rate_msun_per_yr, quenched_backward, quenched_color, "dashed"),
    (out_halo_mass, out_rate_msun_per_yr, starforming_forward, starforming_color, "dotted"),
    (out_halo_mass, out_rate_msun_per_yr, quenched_forward, quenched_color, "dotted"),
]
for halo_mass, rate, sample_mask, color, linestyle in combined_series:
    x, y = binned_median_log_rate(halo_mass, rate, sample_mask)
    ax.plot(x, y, color=color, linestyle=linestyle)

ssfr_handles = [
    mpatches.Patch(color=starforming_color, label="Star forming"),
    mpatches.Patch(color=quenched_color, label="Quenched"),
]
rate_type_handles = [
    mlines.Line2D([], [], color="tab:gray", linestyle="solid", label="Modified"),
    mlines.Line2D([], [], color="tab:gray", linestyle="dashed", label="In"),
    mlines.Line2D([], [], color="tab:gray", linestyle="dotted", label="Out"),
]
ssfr_legend = ax.legend(handles=ssfr_handles, loc="upper left", frameon=False)
ax.add_artist(ssfr_legend)
ax.legend(handles=rate_type_handles, loc="lower right", frameon=False, fontsize=30, labelspacing=0.6, borderpad=0.5)
finish_plot(ax, output_dir / f"tracer_rates_modified_in_out_ssfr_{start_snap_backward}_{start_snap_forward}.pdf")
