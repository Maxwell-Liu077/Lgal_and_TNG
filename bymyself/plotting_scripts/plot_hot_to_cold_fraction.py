"""Initial hot-to-cold fraction profiles grouped by target-snapshot cooling radius."""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, LogLocator

project_root = Path(__file__).resolve().parents[1]
plt.style.use(project_root / "fancy_plots2.mplstyle")

start_snap = 98
target_snap = 99
mw_log_m200c_min = 11.8
mw_log_m200c_max = 12.2
min_count_per_bin = 1
median_linewidth = 3.5
shade_alpha = 0.22
high_color = "tab:red"
low_color = "tab:blue"

data_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")
figure_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/pics/hot_to_cold_fraction")


def profile_percentiles(fractions, sample):
    """Return equal-halo-weight median, 16th/84th percentiles, and counts per bin."""
    selected = fractions[sample]
    finite = np.isfinite(selected)
    number_per_bin = finite.sum(axis=0)
    percentiles = np.full((3, fractions.shape[1]), np.nan, dtype=np.float64)
    valid_bins = number_per_bin >= min_count_per_bin
    if np.any(valid_bins):
        values = np.where(finite[:, valid_bins], selected[:, valid_bins], np.nan)
        percentiles[:, valid_bins] = np.nanpercentile(values, [16.0, 50.0, 84.0], axis=0)
    return percentiles[1], percentiles[0], percentiles[2], number_per_bin


def plot_hot_to_cold_fraction(fraction_file=None, cooling_file=None, output_file=None):
    """Match by start_subhalo_id and plot three comparisons, returning the PDF path."""
    fraction_file = data_dir / "distance_cats" / f"hot_to_cold_fraction_MW_start_{start_snap}_target_{target_snap}.hdf5" if fraction_file is None else Path(fraction_file)
    cooling_file = data_dir / f"real_cooling_radius_{target_snap}.hdf5" if cooling_file is None else Path(cooling_file)
    output_file = figure_dir / f"hot_to_cold_fraction_MW_start_{start_snap}_target_{target_snap}_by_Rcool.pdf" if output_file is None else Path(output_file)
    for input_file in (fraction_file, cooling_file):
        if not input_file.is_file():
            raise FileNotFoundError(f"Missing hot-to-cold plotting input: {input_file}")
        if output_file.resolve() == input_file.resolve():
            raise ValueError("The plot output must not overwrite an input file.")

    with h5py.File(fraction_file, "r") as f:
        start_ids = np.asarray(f["start_subhalo_id"][:], dtype=np.int64)
        halo_mass_msun = np.asarray(f["m200c_msun"][:], dtype=np.float64)
        fractions = np.asarray(f["hot_to_cold_fraction"][:], dtype=np.float64)
        hot_counts = np.asarray(f["hot_tracer_count"][:], dtype=np.int64)
        radial_edges = np.asarray(f["radial_bin_edges_r200c"][:], dtype=np.float64)
        radial_centers = np.asarray(f["radial_bin_centers_r200c"][:], dtype=np.float64)
        for name, expected in (("start_snap", start_snap), ("target_snap", target_snap)):
            if name in f.attrs and int(f.attrs[name]) != expected:
                raise ValueError(f"The fraction file has a different {name}.")

    with h5py.File(cooling_file, "r") as f:
        cooling_start_ids = np.asarray(f["start_subhalo_id"][:], dtype=np.int64)
        cooling_radius_all = np.asarray(f["cooling_radius_r200c"][:], dtype=np.float64)
        cooling_analysis_valid = np.asarray(f["analysis_valid"][:], dtype=bool)
        cooling_status = np.asarray(f["status"][:], dtype=np.uint8)

    fractions[(hot_counts == 0) | ~np.isfinite(fractions)] = np.nan

    # The fraction file is a compact MW subset; the real-radius file can use any tree-row order.
    order = np.argsort(cooling_start_ids)
    positions = np.searchsorted(cooling_start_ids[order], start_ids)
    matched = np.zeros(start_ids.size, dtype=bool)
    candidate_rows = np.flatnonzero(positions < cooling_start_ids.size)
    matched[candidate_rows] = cooling_start_ids[order[positions[candidate_rows]]] == start_ids[candidate_rows]
    fraction_rows = np.flatnonzero(matched)
    cooling_rows = order[positions[fraction_rows]]
    cooling_radius = np.full(start_ids.size, np.nan, dtype=np.float64)
    cooling_valid = np.zeros(start_ids.size, dtype=bool)
    cooling_radius[fraction_rows] = cooling_radius_all[cooling_rows]
    cooling_valid[fraction_rows] = cooling_analysis_valid[cooling_rows] & (cooling_status[cooling_rows] == 1)
    valid_mass = np.isfinite(halo_mass_msun) & (halo_mass_msun > 0.0)
    log_mass = np.full(start_ids.size, np.nan, dtype=np.float64)
    log_mass[valid_mass] = np.log10(halo_mass_msun[valid_mass])
    mw_sample = valid_mass & (log_mass >= mw_log_m200c_min) & (log_mass < mw_log_m200c_max)
    base_sample = mw_sample & matched & cooling_valid & np.isfinite(cooling_radius) & (cooling_radius > 0.0)
    if not np.any(base_sample & np.any(np.isfinite(fractions), axis=1)):
        raise ValueError("No matched MW-like halos have both valid real cooling radii and usable fraction bins.")

    # Rcool == R200c is excluded to preserve the requested strict inequalities.
    below = base_sample & (cooling_radius < 1.0)
    above = base_sample & (cooling_radius > 1.0)
    median_below = float(np.median(cooling_radius[below])) if np.any(below) else np.nan
    median_above = float(np.median(cooling_radius[above])) if np.any(above) else np.nan
    panel_data = [
        ("Cooling Regimes", [(r"$R_{\rm cool}/R_{\rm 200c}<1$", below, "tab:blue"), (r"$R_{\rm cool}/R_{\rm 200c}>1$", above, "tab:orange")], None),
        (r"$R_{\rm cool}/R_{\rm 200c}<1$", [("High", below & (cooling_radius > median_below), high_color), ("Low", below & (cooling_radius <= median_below), low_color)], median_below),
        (r"$R_{\rm cool}/R_{\rm 200c}>1$", [("High", above & (cooling_radius > median_above), high_color), ("Low", above & (cooling_radius <= median_above), low_color)], median_above),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(27, 8), sharex=True, sharey=True, layout="constrained")
    for ax, (title, groups, _) in zip(axes, panel_data):
        has_profile = False
        for label, sample, color in groups:
            median, lower, upper, number_per_bin = profile_percentiles(fractions, sample)
            number = int(np.count_nonzero(sample))
            has_profile |= bool(np.any(np.isfinite(median)))
            ax.plot(radial_centers, median, color=color, linewidth=median_linewidth, label=label + f" ($N={number:,}$)", zorder=3)
            ax.fill_between(radial_centers, lower, upper, color=color, alpha=shade_alpha, linewidth=0.0, zorder=2)
            print(f"{title}, {label}: N={number:,}; contributing halos per radial bin={number_per_bin.tolist()}.", flush=True)
        if not has_profile:
            ax.text(0.5, 0.5, "No valid profiles", transform=ax.transAxes, ha="center", va="center", fontsize=20)
        ax.set_title(title, fontsize=23, pad=12)
        ax.set_xlabel(r"$r/R_{\rm 200c}$", fontsize=24)
        ax.set_xscale("log")
        ax.set_xlim(radial_centers[0], radial_edges[-1])
        ax.set_ylim(0.0, 1.0)
        ax.xaxis.set_major_locator(LogLocator(base=10.0))
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=21)
        ax.grid(which="major", axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", fontsize=18, frameon=True, facecolor="white", framealpha=0.8, edgecolor="none")
    axes[0].set_ylabel(r"mass fraction $M_{hot to cold}/M_{hot}$", fontsize=24)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, format="pdf", bbox_inches="tight")
    plt.close(fig)
    equal_count = int(np.count_nonzero(base_sample & (cooling_radius == 1.0)))
    print(f"MW-like fraction rows={np.count_nonzero(mw_sample):,}; missing cooling-radius IDs={np.count_nonzero(mw_sample & ~matched):,}; valid cooling radii={np.count_nonzero(base_sample):,}; Rcool/R200c == 1 excluded={equal_count:,}.", flush=True)
    print(f"Saved median and 16-84th percentile profiles to {output_file}", flush=True)
    return output_file
