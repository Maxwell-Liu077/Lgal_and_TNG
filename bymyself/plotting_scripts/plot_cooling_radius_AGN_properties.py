"""Cooling radius versus halo mass with all/high/low median trends for three AGN properties."""

from pathlib import Path
import matplotlib.lines as mlines
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator
import matplotlib.patheffects as pe

root = Path(__file__).resolve().parents[1]
plt.style.use(root / "fancy_plots2.mplstyle")

snap = 99
radius_source = "TNG"  # "TNG": measured profiles; "Lgal": analytic cooling radius.
point_size = 60
point_alpha = 0.75
mass_bin_width = 0.25
min_count_per_bin = 10
median_linewidth = 5.5
median_styles = [("High", "#FFB000", "solid"), ("Low", "#FF25E9", "solid")]

data_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")
figure_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/pics/agn_cooling_radius")
agn_file = data_dir / f"AGN_properties_{snap}.hdf5"

with h5py.File(agn_file, "r") as f:
    start_ids = np.asarray(f["start_subhalo_id"][:], dtype=np.int64)
    subhalo_ids = np.asarray(f["subhalo_id"][:], dtype=np.int64)
    fof_ids = np.asarray(f["fof_id"][:], dtype=np.int64)
    halo_mass_msun = np.asarray(f["m200c_msun"][:], dtype=np.float64)
    analysis_valid = np.asarray(f["analysis_valid"][:], dtype=bool)
    scale_factor = float(f.attrs["scale_factor"])
    bh_mass_msun = np.asarray(f["black_hole/bh_mass_msun"][:], dtype=np.float64)
    energy_ratio = np.asarray(f["black_hole/energy_kinetic_to_thermal_ratio"][:], dtype=np.float64)
    bh_valid = np.asarray(f["black_hole/bh_valid"][:], dtype=bool)
    energy_ratio_valid = np.asarray(f["black_hole/energy_ratio_valid"][:], dtype=bool)
    entropy_shell = np.asarray(f["entropy/entropy_non_sf_0p1_to_1_r200c_mass_weighted_keV_cm2"][:], dtype=np.float64)
    entropy_shell_valid = np.asarray(f["entropy/entropy_valid_non_sf_shell"][:], dtype=bool)

if radius_source == "TNG":
    cooling_file = data_dir / f"real_cooling_radius_{snap}.hdf5"
    with h5py.File(cooling_file, "r") as f:
        cooling_start_ids = np.asarray(f["start_subhalo_id"][:], dtype=np.int64)
        cooling_subhalo_ids = np.asarray(f["subhalo_id"][:], dtype=np.int64)
        cooling_fof_ids = np.asarray(f["fof_id"][:], dtype=np.int64)
        cooling_radius = np.asarray(f["cooling_radius_r200c"][:], dtype=np.float64)
        cooling_analysis_valid = np.asarray(f["analysis_valid"][:], dtype=bool)
        cooling_status = np.asarray(f["status"][:], dtype=np.uint8)

    if not (np.array_equal(start_ids, cooling_start_ids) and np.array_equal(subhalo_ids, cooling_subhalo_ids) and np.array_equal(fof_ids, cooling_fof_ids)):
        raise ValueError("AGN and real cooling-radius files must describe the same tree rows and snapshot nodes.")

    cooling_valid = cooling_analysis_valid & (cooling_status == 1)
    radius_label = "TNG measured"
elif radius_source == "Lgal":
    cooling_file = data_dir / f"cooling_radius_{snap}.hdf5"
    with h5py.File(cooling_file, "r") as f:
        cooling_radius_all = np.asarray(f["cooling_radius"][:], dtype=np.float64)
        cooling_halo_flag = np.asarray(f["halo_flag"][:], dtype=bool)

    cooling_radius = np.full(start_ids.size, np.nan, dtype=np.float64)
    cooling_valid = np.zeros(start_ids.size, dtype=bool)
    valid_fof = (fof_ids >= 0) & (fof_ids < cooling_radius_all.size)
    cooling_radius[valid_fof] = cooling_radius_all[fof_ids[valid_fof]]
    cooling_valid[valid_fof] = cooling_halo_flag[fof_ids[valid_fof]]
    radius_label = "Lgal analytic"
else:
    raise ValueError("radius_source must be 'TNG' or 'Lgal'.")

valid_mass = analysis_valid & np.isfinite(halo_mass_msun) & (halo_mass_msun > 0.0)
base_sample = valid_mass & cooling_valid & np.isfinite(cooling_radius) & (cooling_radius > 0.0)

log_halo_mass = np.full(start_ids.size, np.nan, dtype=np.float64)
log_halo_mass[valid_mass] = np.log10(halo_mass_msun[valid_mass])

bin_mass_min = np.floor(np.min(log_halo_mass[valid_mass]) / mass_bin_width) * mass_bin_width
bin_mass_max = np.ceil(np.max(log_halo_mass[valid_mass]) / mass_bin_width) * mass_bin_width
if bin_mass_max <= bin_mass_min:
    bin_mass_max = bin_mass_min + mass_bin_width
mass_bin_edges = np.arange(bin_mass_min, bin_mass_max + 0.5 * mass_bin_width, mass_bin_width)
mass_bin_centers = 0.5 * (mass_bin_edges[:-1] + mass_bin_edges[1:])
num_mass_bins = mass_bin_centers.size
mass_bin_indices = np.full(start_ids.size, -1, dtype=np.int64)
mass_bin_indices[valid_mass] = np.searchsorted(mass_bin_edges, log_halo_mass[valid_mass], side="right") - 1
mass_bin_indices[valid_mass] = np.minimum(mass_bin_indices[valid_mass], num_mass_bins - 1)

panel_data = [
    (bh_mass_msun, bh_valid, "BH mass", r"$\log_{10}(M_{\rm BH}/M_\odot)$", False, None, "viridis"),
    (energy_ratio, energy_ratio_valid, r"$R_E=E_{\rm Kinetic}/E_{\rm Thermal}$", r"$\log_{10}(1+R_E)$", True, None, "plasma"),
    (entropy_shell, entropy_shell_valid, "CGM entropy", r"$\log_{10}[K_{\rm CGM}/(\mathrm{keV\,cm^2})]$", False, None, "magma"),
]

fig, axes = plt.subplots(1, 3, figsize=(30, 8), sharex=True, sharey=True, layout="constrained")

for ax, (values, property_valid, title, colorbar_label, include_zero, fixed_threshold, panel_cmap) in zip(axes.flat, panel_data):
    positive_values = values >= 0.0 if include_zero else values > 0.0
    property_sample = valid_mass & property_valid & np.isfinite(values) & positive_values
    plot_mask = base_sample & property_sample
    number = int(np.count_nonzero(plot_mask))

    high_sample = np.zeros(start_ids.size, dtype=bool)
    low_sample = np.zeros(start_ids.size, dtype=bool)
    bin_thresholds = np.full(num_mass_bins, np.nan, dtype=np.float64)
    for bin_index in range(num_mass_bins):
        reference_sample = property_sample & (mass_bin_indices == bin_index)
        if not np.any(reference_sample):
            continue
        threshold = np.median(values[reference_sample]) if fixed_threshold is None else fixed_threshold
        bin_thresholds[bin_index] = threshold
        high_sample[reference_sample] = values[reference_sample] > threshold
        low_sample[reference_sample] = values[reference_sample] <= threshold

    group_samples = (plot_mask, plot_mask & high_sample, plot_mask & low_sample)
    median_radius = np.full((3, num_mass_bins), np.nan, dtype=np.float64)
    group_counts = np.zeros((3, num_mass_bins), dtype=np.int64)
    for group_index, sample in enumerate(group_samples):
        for bin_index in range(num_mass_bins):
            in_bin = sample & (mass_bin_indices == bin_index)
            group_counts[group_index, bin_index] = np.count_nonzero(in_bin)
            if group_counts[group_index, bin_index] >= min_count_per_bin:
                median_radius[group_index, bin_index] = np.median(cooling_radius[in_bin])

    ax.axvspan(10.8, 11.2, color="lightgray", alpha=0.25)
    ax.axvspan(11.8, 12.2, color="lightgray", alpha=0.25)
    ax.axvspan(12.6, 13.4, color="lightgray", alpha=0.25)

    color_values_raw = values[plot_mask].copy()
    if include_zero:
        positive_color_values = color_values_raw[color_values_raw > 0.0]
        if positive_color_values.size == 0:
            raise ValueError(f"Cannot map log10 values for {title}: the plotted sample has no positive values.")
        color_values_raw[color_values_raw == 0.0] = np.min(positive_color_values)
    color_values = np.log10(color_values_raw)
    scatter = ax.scatter(log_halo_mass[plot_mask], cooling_radius[plot_mask], c=color_values, cmap=panel_cmap, marker="h", s=point_size, alpha=point_alpha, edgecolors="none", rasterized=True, zorder=2)
    colorbar = fig.colorbar(scatter, ax=ax, location="right", pad=0.025, fraction=0.035, aspect=28.5)
    colorbar.set_label(colorbar_label, fontsize=21)
    colorbar.ax.tick_params(labelsize=19)

    for group_index, (label, color, linestyle) in zip((1, 2), median_styles):
        line, = ax.plot(mass_bin_centers, median_radius[group_index], color=color, linestyle=linestyle, linewidth=median_linewidth, label=label, zorder=20)
        line.set_path_effects([pe.Stroke(linewidth=median_linewidth + 1.0, foreground="black"), pe.Normal()])

    ax.set_title(title, fontsize=24, pad=12)
    ax.set_xlabel(r"halo mass [$\log_{10}(M_{200c}/M_\odot)$]", fontsize=23)
    ax.set_ylabel(r"$R_{\rm cool}/R_{\rm 200c}$", fontsize=24)
    ax.set_xlim(10.6, 13.6)
    ax.set_yscale("log")
    ax.set_ylim(0.01, 10.0)
    ax.set_xticks([11, 12, 13])
    ax.minorticks_on()
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=21, labelbottom=True, labelleft=True)
    ax.grid(which="major", axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    high_count = int(np.count_nonzero(group_samples[1]))
    low_count = int(np.count_nonzero(group_samples[2]))
    plotted_bin_counts = np.count_nonzero(group_counts >= min_count_per_bin, axis=1)

median_handles = []
for label, color, linestyle in median_styles:
    handle = mlines.Line2D([], [], color=color, linestyle=linestyle, linewidth=median_linewidth, label=label)
    handle.set_path_effects([pe.Stroke(linewidth=median_linewidth + 1.0, foreground="black"), pe.Normal()])
    median_handles.append(handle)
fig.legend(handles=median_handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=30, ncol=len(median_handles))
figure_dir.mkdir(parents=True, exist_ok=True)
output_file = figure_dir / f"cooling_radius_AGN_properties_{radius_source}_snap{snap}.pdf"
fig.savefig(output_file, format="pdf", bbox_inches="tight")
plt.close(fig)
