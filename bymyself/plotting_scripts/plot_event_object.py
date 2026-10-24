"""Plot one major-merger host's cooling-radius history and mark its merger snapshots."""

from pathlib import Path
import argparse

import h5py
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator


root = Path(__file__).resolve().parents[1]
data_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")
default_history_file = data_dir / "event_objects" / "event_cooling_radius_history_40_99.hdf5"
default_figure_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/pics/event_objects")
line_width = 3.5


def plot_event_object(history_file=default_history_file, figure_dir=default_figure_dir, start_id=None, random_seed=None):
    """Return (PDF path, chosen start_subhalo_id); random selection requires both model curves to be drawable."""
    history_file, figure_dir = Path(history_file), Path(figure_dir)
    if not history_file.is_file():
        raise FileNotFoundError(f"Run event_object.object_cooling_radius first: {history_file}")

    with h5py.File(history_file, "r") as f:
        start_ids = np.asarray(f["start_subhalo_id"][:], dtype=np.int64)
        snaps = np.asarray(f["snap_numbers"][:], dtype=np.int64)
        redshift = np.asarray(f["redshift"][:], dtype=np.float64)
        fof_ids = np.asarray(f["mpb_fof_ids"][:], dtype=np.int64)
        analysis_valid = np.asarray(f["analysis_valid"][:], dtype=bool)
        lgal_radius = np.asarray(f["lgal_cooling_radius_r200c"][:], dtype=np.float64)
        tng_radius = np.asarray(f["tng_cooling_radius_r200c"][:], dtype=np.float64)
        lgal_valid = np.asarray(f["lgal_valid"][:], dtype=bool)
        tng_valid = np.asarray(f["tng_valid"][:], dtype=bool)
        tng_status = np.asarray(f["tng_status"][:], dtype=np.uint8)
        merger_counts = np.asarray(f["major_merger_count"][:], dtype=np.int64)
        event_offsets = np.asarray(f["event_offsets"][:], dtype=np.int64)
        event_start_ids = np.asarray(f["events/start_subhalo_id"][:], dtype=np.int64)
        event_snaps = np.asarray(f["events/merger_snap"][:], dtype=np.int64)
        event_mass_ratios = np.asarray(f["events/mass_ratio"][:], dtype=np.float64)
        selection_complete = np.asarray(f["selection_complete"][:], dtype=bool)

    if start_ids.ndim != 1 or np.any(start_ids < 0) or np.unique(start_ids).size != start_ids.size:
        raise ValueError("start_subhalo_id must contain unique nonnegative IDs.")
    if snaps.ndim != 1 or snaps.size < 2 or np.unique(snaps).size != snaps.size:
        raise ValueError("snap_numbers must contain at least two unique snapshots.")
    if redshift.shape != snaps.shape or not np.all(np.isfinite(redshift)) or np.unique(redshift).size != redshift.size:
        raise ValueError("Each snapshot must have one finite, distinct redshift.")
    expected_shape = (start_ids.size, snaps.size)
    for name, values in (("mpb_fof_ids", fof_ids), ("analysis_valid", analysis_valid), ("Lgal radius", lgal_radius), ("TNG radius", tng_radius), ("lgal_valid", lgal_valid), ("tng_valid", tng_valid), ("tng_status", tng_status)):
        if values.shape != expected_shape:
            raise ValueError(f"{name} has shape {values.shape}; expected (object, snapshot)={expected_shape}.")
    if merger_counts.shape != start_ids.shape or selection_complete.shape != start_ids.shape:
        raise ValueError("Merger counts and selection_complete must have one entry per object.")
    if event_snaps.ndim != 1 or event_start_ids.shape != event_snaps.shape or event_mass_ratios.shape != event_snaps.shape:
        raise ValueError("Merger event arrays must have matching one-dimensional shapes.")
    if event_offsets.shape != (start_ids.size + 1,) or event_offsets[0] != 0 or event_offsets[-1] != event_snaps.size or np.any(np.diff(event_offsets) < 0):
        raise ValueError("event_offsets must span the event table with one interval per object.")
    if not np.array_equal(np.diff(event_offsets), merger_counts):
        raise ValueError("major_merger_count disagrees with event_offsets.")
    if not np.array_equal(event_start_ids, np.repeat(start_ids, merger_counts)):
        raise ValueError("Merger events are not grouped in the history file's object order.")
    if not np.all(np.isin(event_snaps, snaps)):
        raise ValueError("A merger snapshot is absent from the cooling-radius history.")

    # Apply each model's mask independently and preserve missing epochs as NaNs.
    lgal_valid &= analysis_valid & np.isfinite(lgal_radius) & (lgal_radius > 0.0)
    tng_valid &= analysis_valid & (tng_status == 1) & np.isfinite(tng_radius) & (tng_radius > 0.0)
    drawable = (merger_counts > 0) & (lgal_valid.sum(axis=1) >= 2) & (tng_valid.sum(axis=1) >= 2)
    eligible_rows = np.flatnonzero(drawable)
    if start_id is None:
        if eligible_rows.size == 0:
            raise ValueError("No major-merger host has at least two valid measurements for each cooling-radius model.")
        row = int(np.random.default_rng(random_seed).choice(eligible_rows))
    else:
        matches = np.flatnonzero(start_ids == start_id)
        if matches.size == 0:
            raise ValueError(f"start_subhalo_id={start_id} is absent from {history_file}.")
        row = int(matches[0])
        if not drawable[row]:
            raise ValueError(f"start_subhalo_id={start_id} needs a major merger and at least two valid measurements per model.")
    chosen_start_id = int(start_ids[row])

    # Earlier times (higher z) appear on the left; missing measurements break the lines.
    order = np.argsort(redshift)[::-1]
    z = redshift[order]
    tng = np.where(tng_valid[row], tng_radius[row], np.nan)[order]
    lgal = np.where(lgal_valid[row], lgal_radius[row], np.nan)[order]
    first, last = event_offsets[row:row + 2]
    merger_snaps = event_snaps[first:last]
    merger_ratios = event_mass_ratios[first:last]
    snap_to_redshift = {int(snap): float(value) for snap, value in zip(snaps, redshift)}

    with plt.style.context(root / "fancy_plots2.mplstyle"):
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        ax.plot(z, tng, color="C0", linestyle="solid", linewidth=line_width, marker="o", markersize=4, label="TNG measured", zorder=3)
        ax.plot(z, lgal, color="C1", linestyle="dashed", linewidth=line_width, marker="o", markersize=4, label="Lgal analytic", zorder=3)

        # Use the common descendant's merger_snap, not the earlier mass_ratio_snap.
        unique_snaps, multiplicities = np.unique(merger_snaps, return_counts=True)
        for merger_snap, count in zip(unique_snaps, multiplicities):
            merger_z = snap_to_redshift[int(merger_snap)]
            ax.axvline(merger_z, color="dimgray", linestyle="dotted", linewidth=1.8, alpha=0.85, zorder=1)
            label = f"snap {merger_snap}" if count == 1 else f"snap {merger_snap} ({count} mergers)"
            ax.text(merger_z, 0.97, label, transform=ax.get_xaxis_transform(), rotation=90, ha="right", va="top", fontsize=14, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"), zorder=4)

        all_radii = np.concatenate((tng[np.isfinite(tng)], lgal[np.isfinite(lgal)]))
        log_min, log_max = np.log10(all_radii.min()), np.log10(all_radii.max())
        log_padding = max(0.12, 0.08 * (log_max - log_min))
        ax.set_yscale("log")
        ax.set_ylim(10.0 ** (log_min - log_padding), 10.0 ** (log_max + log_padding))
        z_padding = 0.025 * (z.max() - z.min())
        ax.set_xlim(z.max() + z_padding, z.min() - z_padding)
        ax.set_xlabel(r"redshift $z$")
        ax.set_ylabel(r"$R_{\rm cool}/R_{\rm 200c}$")
        title = f"Start subhalo ID {chosen_start_id}"
        z0_column = np.flatnonzero(snaps == 99)
        if z0_column.size and fof_ids[row, z0_column[0]] >= 0:
            title = f"Halo {fof_ids[row, z0_column[0]]} at z=0 (start subhalo {chosen_start_id})"
        ax.set_title(title, fontsize=23, pad=15)
        ax.minorticks_on()
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.tick_params(axis="y", which="minor", left=True, right=True, length=5)
        ax.grid(which="major", axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        handles, _ = ax.get_legend_handles_labels()
        handles.append(mlines.Line2D([], [], color="dimgray", linestyle="dotted", linewidth=1.8, label="Major merger"))
        ax.legend(handles=handles, loc="best", frameon=True, facecolor="white", framealpha=0.8, edgecolor="none", fontsize=21)
        fig.tight_layout()
        figure_dir.mkdir(parents=True, exist_ok=True)
        output_file = figure_dir / f"cooling_radius_evolution_start{chosen_start_id}_snap{snaps.min()}_{snaps.max()}.pdf"
        fig.savefig(output_file, format="pdf", bbox_inches="tight")
        plt.close(fig)

    print(f"Chosen start_subhalo_id={chosen_start_id}; drawable hosts={eligible_rows.size}/{start_ids.size}; valid epochs TNG/Lgal={tng_valid[row].sum()}/{lgal_valid[row].sum()}; selection_complete={bool(selection_complete[row])}.", flush=True)
    for merger_snap, ratio in zip(merger_snaps, merger_ratios):
        print(f"Major merger: snap={merger_snap}, z={snap_to_redshift[int(merger_snap)]:.3f}, stellar mass ratio={ratio:.3f}", flush=True)
    print(f"Saved cooling-radius evolution plot to {output_file}", flush=True)
    return output_file, chosen_start_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-file", type=Path, default=default_history_file)
    parser.add_argument("--figure-dir", type=Path, default=default_figure_dir)
    parser.add_argument("--start-id", type=int, default=None, help="Plot this start_subhalo_id instead of choosing randomly.")
    parser.add_argument("--seed", type=int, default=None, help="Fix the random seed to reproduce the selected host.")
    args = parser.parse_args()
    plot_event_object(args.history_file, args.figure_dir, start_id=args.start_id, random_seed=args.seed)
