"""Cooling radius versus halo mass, colored by BH properties and gas entropy."""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator


root = Path(__file__).resolve().parents[1]
plt.style.use(root / "fancy_plots2.mplstyle")

snap = 99
radius_source = "TNG"  # "TNG": measured profiles; "Lgal": analytic cooling radius.
point_size = 22
point_alpha = 0.75
color_map = "viridis"

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
    entropy_hot = np.asarray(f["entropy/entropy_hot_mass_weighted_keV_cm2"][:], dtype=np.float64)
    entropy_shell = np.asarray(f["entropy/entropy_non_sf_0p1_to_1_r200c_mass_weighted_keV_cm2"][:], dtype=np.float64)
    entropy_hot_valid = np.asarray(f["entropy/entropy_valid_hot"][:], dtype=bool)
    entropy_shell_valid = np.asarray(f["entropy/entropy_valid_non_sf_shell"][:], dtype=bool)

# Both upstream calculations preserve all selected-tree rows, including invalid rows.
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

base_sample = analysis_valid & cooling_valid & np.isfinite(halo_mass_msun) & (halo_mass_msun > 0.0) & np.isfinite(cooling_radius) & (cooling_radius > 0.0)
if not np.any(base_sample):
    raise ValueError(f"No selected tree halos have valid masses and {radius_source} cooling radii at snap {snap}.")

log_halo_mass = np.full(start_ids.size, np.nan, dtype=np.float64)
log_halo_mass[base_sample] = np.log10(halo_mass_msun[base_sample])
mass_min = float(np.min(log_halo_mass[base_sample]))
mass_max = float(np.max(log_halo_mass[base_sample]))
mass_padding = max(0.05, 0.03 * (mass_max - mass_min))
radius_min = 10.0 ** np.floor(np.log10(np.min(cooling_radius[base_sample])))
radius_max = 10.0 ** np.ceil(np.log10(np.max(cooling_radius[base_sample])))
if radius_max <= radius_min:
    radius_min /= 10.0
    radius_max *= 10.0
integer_ticks = np.arange(int(np.ceil(mass_min - mass_padding)), int(np.floor(mass_max + mass_padding)) + 1, dtype=int)
redshift = 1.0 / scale_factor - 1.0

# Each panel uses its own property validity; a missing BH does not remove a halo from entropy panels.
panel_data = [
    (bh_mass_msun, bh_valid, "BH mass", r"$\log_{10}(M_{\rm BH}/M_\odot)$", False),
    (energy_ratio, energy_ratio_valid, "Cumulative feedback energy ratio", r"$\log_{10}(1+R_E)$", True),
    (entropy_hot, entropy_hot_valid, "Hot gas entropy", r"$\log_{10}[K_{\rm hot}/(\mathrm{keV\,cm^2})]$", False),
    (entropy_shell, entropy_shell_valid, "Non-star-forming CGM entropy", r"$\log_{10}[K_{\rm CGM}/(\mathrm{keV\,cm^2})]$", False),
]

fig, axes = plt.subplots(2, 2, figsize=(22, 16), sharex=True, sharey=True, layout="constrained")
fig.suptitle(rf"{radius_label}: $z={redshift:.1f}$", fontsize=26)

for ax, (values, property_valid, title, colorbar_label, include_zero) in zip(axes.flat, panel_data):
    positive_values = values >= 0.0 if include_zero else values > 0.0
    plot_mask = base_sample & property_valid & np.isfinite(values) & positive_values
    number = int(np.count_nonzero(plot_mask))

    if number:
        # log10(1 + R_E) retains pure thermal cases with R_E == 0.
        color_values = np.log1p(values[plot_mask]) / np.log(10.0) if include_zero else np.log10(values[plot_mask])
        scatter = ax.scatter(log_halo_mass[plot_mask], cooling_radius[plot_mask], c=color_values, cmap=color_map, marker="s", s=point_size, alpha=point_alpha, edgecolors="none", rasterized=True)
        colorbar = fig.colorbar(scatter, ax=ax, location="right", pad=0.025, fraction=0.05)
        colorbar.set_label(colorbar_label, fontsize=21)
        colorbar.ax.tick_params(labelsize=19)
    else:
        ax.text(0.5, 0.5, "No valid samples", transform=ax.transAxes, ha="center", va="center", fontsize=22)

    ax.set_title(title, fontsize=24, pad=12)
    ax.set_xlabel(r"halo mass [$\log_{10}(M_{200c}/M_\odot)$]", fontsize=23)
    ax.set_ylabel(r"$R_{\rm cool}/R_{\rm 200c}$", fontsize=24)
    ax.set_xlim(mass_min - mass_padding, mass_max + mass_padding)
    ax.set_yscale("log")
    ax.set_ylim(radius_min, radius_max)
    if integer_ticks.size:
        ax.set_xticks(integer_ticks)
    ax.minorticks_on()
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=21, labelbottom=True, labelleft=True)
    ax.grid(which="major", axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.text(0.04, 0.96, f"$N={number:,}$", transform=ax.transAxes, ha="left", va="top", fontsize=20, bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
    print(f"Snap {snap}, {radius_label}, {title}: {number}/{start_ids.size} selected tree halos.", flush=True)

figure_dir.mkdir(parents=True, exist_ok=True)
output_file = figure_dir / f"cooling_radius_AGN_properties_{radius_source}_snap{snap}.pdf"
fig.savefig(output_file, format="pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved AGN cooling-radius scatter plots to {output_file}", flush=True)
