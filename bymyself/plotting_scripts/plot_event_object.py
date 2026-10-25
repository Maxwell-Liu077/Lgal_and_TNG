"""Plot cooling-radius histories for sampled major-merger halos."""

from pathlib import Path

import h5py
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np


root = Path(__file__).resolve().parents[1]
plt.style.use(root / "fancy_plots2.mplstyle")

data_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")
figure_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/pics/event_object")
input_file = data_dir / "event_object_cooling_radius_40_to_99.hdf5"
output_file = figure_dir / "event_object_cooling_radius_40_to_99.pdf"

if not input_file.is_file():
    raise FileNotFoundError(f"Event-object cooling-radius file does not exist: {input_file}")

with h5py.File(input_file, "r") as f:
    tree_rows = np.asarray(f["tree_row"][:], dtype=np.int64)
    start_subhalo_ids = np.asarray(f["start_subhalo_id"][:], dtype=np.int64)
    selected_merger_counts = np.asarray(f["selected_merger_count"][:], dtype=np.int64)
    event_offsets = np.asarray(f["selected_event_offsets"][:], dtype=np.int64)
    event_merger_snaps = np.asarray(f["event_merger_snap"][:], dtype=np.int64)
    snap_numbers = np.asarray(f["snap_numbers"][:], dtype=np.int64)
    redshift = np.asarray(f["redshift"][:], dtype=np.float64)
    halo_mass_msun = np.asarray(f["halo_mass_msun"][:], dtype=np.float64)
    analysis_valid = np.asarray(f["analysis_valid"][:], dtype=bool)
    tng_radius = np.asarray(f["cooling_radius_tng_r200c"][:], dtype=np.float64)
    lgal_radius = np.asarray(f["cooling_radius_lgal_r200c"][:], dtype=np.float64)
    tng_valid = np.asarray(f["tng_valid"][:], dtype=bool)
    lgal_valid = np.asarray(f["lgal_valid"][:], dtype=bool)

n_objects = start_subhalo_ids.size
n_snaps = snap_numbers.size
expected_shape = (n_objects, n_snaps)
if n_objects == 0 or redshift.shape != (n_snaps,):
    raise ValueError("The input file has no selected halos or its redshift array is misaligned with snap_numbers.")
for name, values in (
    ("halo_mass_msun", halo_mass_msun),
    ("analysis_valid", analysis_valid),
    ("cooling_radius_tng_r200c", tng_radius),
    ("cooling_radius_lgal_r200c", lgal_radius),
    ("tng_valid", tng_valid),
    ("lgal_valid", lgal_valid),
):
    if values.shape != expected_shape:
        raise ValueError(f"{name} has shape {values.shape}; expected {expected_shape}.")
if tree_rows.shape != (n_objects,):
    raise ValueError("tree_row and start_subhalo_id have inconsistent lengths.")
if event_offsets.shape != (n_objects + 1,) or event_offsets[0] != 0 or event_offsets[-1] != event_merger_snaps.size or np.any(np.diff(event_offsets) < 0):
    raise ValueError("selected_event_offsets does not correctly delimit the per-halo merger-event arrays.")
if selected_merger_counts.shape != (n_objects,):
    raise ValueError("Merger counts, event offsets, snapshots, and mass ratios have inconsistent lengths.")
if not np.array_equal(np.diff(event_offsets), selected_merger_counts):
    raise ValueError("selected_merger_count does not agree with selected_event_offsets.")
if not np.all(np.isfinite(redshift)):
    raise ValueError("The redshift array contains non-finite values.")

# Plot redshift in increasing order and apply the stored validity flags to each model.
redshift_order = np.argsort(redshift)
redshift_plot = redshift[redshift_order]
tng_plot = tng_radius[:, redshift_order].copy()
lgal_plot = lgal_radius[:, redshift_order].copy()
tng_plot[~(analysis_valid[:, redshift_order] & tng_valid[:, redshift_order])] = np.nan
lgal_plot[~(analysis_valid[:, redshift_order] & lgal_valid[:, redshift_order])] = np.nan
tng_plot[~np.isfinite(tng_plot) | (tng_plot <= 0.0)] = np.nan
lgal_plot[~np.isfinite(lgal_plot) | (lgal_plot <= 0.0)] = np.nan

all_positive_radii = np.concatenate((tng_plot[np.isfinite(tng_plot)], lgal_plot[np.isfinite(lgal_plot)]))
if all_positive_radii.size == 0:
    raise ValueError("No valid positive TNG or L-GALAXIES cooling radii are available to plot.")
radius_log_min = float(np.log10(np.min(all_positive_radii)))
radius_log_max = float(np.log10(np.max(all_positive_radii)))
if radius_log_min == radius_log_max:
    radius_log_min -= 0.5
    radius_log_max += 0.5
radius_limits = (10.0 ** (radius_log_min - 0.08), 10.0 ** (radius_log_max + 0.08))

fig, axes = plt.subplots(1, n_objects, figsize=(9 * n_objects, 9), sharex=True, sharey=True, layout="constrained")
axes = np.atleast_1d(axes)

for object_index, ax in enumerate(axes):
    ax.plot(redshift_plot, tng_plot[object_index], color="tab:blue", linestyle="solid", linewidth=3.0, label="TNG measured")
    ax.plot(redshift_plot, lgal_plot[object_index], color="tab:orange", linestyle="dashed", linewidth=3.0, label="L-GALAXIES analytic")

    event_first = int(event_offsets[object_index])
    event_last = int(event_offsets[object_index + 1])
    object_event_snaps = event_merger_snaps[event_first:event_last]
    for event_snap in np.unique(object_event_snaps):
        snap_match = np.flatnonzero(snap_numbers == event_snap)
        if snap_match.size != 1:
            raise ValueError(f"Merger snap {event_snap} is not present exactly once in snap_numbers.")
        event_redshift = float(redshift[snap_match[0]])
        event_multiplicity = int(np.count_nonzero(object_event_snaps == event_snap))
        ax.axvline(event_redshift, color="black", linestyle="dotted", linewidth=2.0, alpha=0.75, zorder=1)
        if event_multiplicity > 1:
            ax.text(event_redshift, 0.97, f"×{event_multiplicity}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=14, color="black")

    valid_halo_mass = halo_mass_msun[object_index]
    valid_halo_mass = valid_halo_mass[np.isfinite(valid_halo_mass) & (valid_halo_mass > 0.0)]
    mass_label = f"log M200c={np.log10(np.median(valid_halo_mass)):.2f}" if valid_halo_mass.size else "log M200c unavailable"
    ax.set_title(f"Halo {object_index + 1}: tree row {tree_rows[object_index]}, start ID {start_subhalo_ids[object_index]}\n{mass_label}; {event_last - event_first} major merger(s)")
    ax.set_xlabel("Redshift, z")
    ax.set_yscale("log")
    ax.set_ylim(*radius_limits)
    ax.set_xlim(float(np.min(redshift_plot)), float(np.max(redshift_plot)))
    ax.set_axisbelow(True)
    ax.grid(which="major", axis="both", alpha=0.25)
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)

    finite_tng = np.count_nonzero(np.isfinite(tng_plot[object_index]))
    finite_lgal = np.count_nonzero(np.isfinite(lgal_plot[object_index]))
    print(f"Halo {object_index + 1} (start ID {start_subhalo_ids[object_index]}): {event_last - event_first} major mergers; valid radii TNG={finite_tng}/{n_snaps}, L-GALAXIES={finite_lgal}/{n_snaps}.", flush=True)

axes[0].set_ylabel(r"Cooling radius [$R_{\rm cool}/R_{\rm 200c}$]")
legend_handles = [
    mlines.Line2D([], [], color="tab:blue", linestyle="solid", linewidth=3.0, label="TNG measured"),
    mlines.Line2D([], [], color="tab:orange", linestyle="dashed", linewidth=3.0, label="L-GALAXIES analytic"),
    mlines.Line2D([], [], color="black", linestyle="dotted", linewidth=2.0, label="Major merger"),
]
fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False, fontsize=18)
fig.suptitle("Cooling-radius histories of halos with major mergers", fontsize=25)

figure_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(output_file, format="pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved event-object cooling-radius plot to {output_file}", flush=True)
