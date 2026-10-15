import h5py
import illustris_python as il
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from pathlib import Path

plt.style.use("fancy_plots2.mplstyle")

basePath = "/public/share/chenhouzun/TNG50-1/output/"
data_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")
figure_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/pics/cold_gas")
figure_dir.mkdir(parents=True, exist_ok=True)

snaps = [40, 67, 99]
cooling_radius_intervals = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
interval_labels = ["[0, 1)", "[1, 2)", "[2, 3)", "[3, 4)", "[4, 5]"]
interval_colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(cooling_radius_intervals)))
cold_fraction_bins = np.linspace(0.0, 1.0, 26)

fig, axes = plt.subplots(len(snaps), 1, figsize=(14, 16), sharex=True, sharey=True)

for ax, snap in zip(axes, snaps):
    header = il.groupcat.loadHeader(basePath, snap)
    h = float(header["HubbleParam"])

    halos = il.groupcat.loadHalos(basePath, snap, fields=["Group_M_Crit200", "GroupFirstSub"])
    group_masses = np.asarray(halos["Group_M_Crit200"], dtype=np.float64) * 1.0e10 / h
    group_first_sub = np.asarray(halos["GroupFirstSub"], dtype=np.int64)

    subhalo_mass_type = il.groupcat.loadSubhalos(basePath, snap, fields=["SubhaloMassType"])
    if isinstance(subhalo_mass_type, dict):
        subhalo_mass_type = subhalo_mass_type["SubhaloMassType"]
    subhalo_mass_type = np.asarray(subhalo_mass_type, dtype=np.float64)

    has_central = (group_first_sub >= 0) & (group_first_sub < subhalo_mass_type.shape[0])
    central_stellar_mass = np.full(group_first_sub.size, np.nan, dtype=np.float64)
    central_stellar_mass[has_central] = subhalo_mass_type[group_first_sub[has_central], 4] * 1.0e10 / h
    central_mass_ok = has_central & np.isfinite(central_stellar_mass) & (central_stellar_mass > 1.0e7)

    cooling_file = data_dir / f"cooling_radius_{snap}.hdf5"
    cold_gas_file = data_dir / f"cold_gas_fraction_{snap}_1.hdf5"
    for input_file in (cooling_file, cold_gas_file):
        if not input_file.is_file():
            raise FileNotFoundError(f"Required input file does not exist: {input_file}")

    with h5py.File(cooling_file, "r") as f:
        cooling_flag = np.asarray(f["halo_flag"][:], dtype=bool)
        cooling_radius = np.asarray(f["cooling_radius"][:], dtype=np.float64)

    with h5py.File(cold_gas_file, "r") as f:
        cold_gas_fraction = np.asarray(f["cold_gas_fraction"][:], dtype=np.float64)
        cold_gas_flag = np.asarray(f["halo_flag"][:], dtype=bool)
        cold_gas_selection_flag = np.asarray(f["selection_flag"][:], dtype=bool)

    array_sizes = {
        group_masses.size,
        group_first_sub.size,
        cooling_flag.size,
        cooling_radius.size,
        cold_gas_fraction.size,
        cold_gas_flag.size,
        cold_gas_selection_flag.size,
    }
    if len(array_sizes) != 1:
        raise ValueError(f"Halo-catalogue and gas/cooling arrays are not aligned at snap {snap}.")

    valid_sample = (
        cooling_flag
        & cold_gas_flag
        & cold_gas_selection_flag
        & central_mass_ok
        & np.isfinite(group_masses)
        & (group_masses > 0.0)
        & np.isfinite(cooling_radius)
        & (cooling_radius >= 0.0)
        & np.isfinite(cold_gas_fraction)
        & (cold_gas_fraction >= 0.0)
        & (cold_gas_fraction <= 1.0)
    )

    interval_counts = []
    for interval_index, ((lower, upper), label, color) in enumerate(
        zip(cooling_radius_intervals, interval_labels, interval_colors)
    ):
        in_interval = valid_sample & (cooling_radius >= lower)
        if interval_index == len(cooling_radius_intervals) - 1:
            in_interval &= cooling_radius <= upper
        else:
            in_interval &= cooling_radius < upper

        fraction_values = cold_gas_fraction[in_interval]
        interval_counts.append(fraction_values.size)
        if fraction_values.size == 0:
            continue

        ax.hist(
            fraction_values,
            bins=cold_fraction_bins,
            density=True,
            histtype="step",
            linewidth=3.2,
            color=color,
        )

    count_text = "\n".join(
        f"{label}: N={count:,}" for label, count in zip(interval_labels, interval_counts)
    )
    ax.text(
        0.98,
        0.96,
        count_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize="small",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    ax.set_title(f"snap {snap}")
    ax.set_ylabel("PDF")
    ax.set_xlim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    print(f"snap {snap}: {np.count_nonzero(valid_sample):,} halos in the common valid sample")

legend_handles = [
    Line2D([], [], color=color, linewidth=2.2, label=label)
    for color, label in zip(interval_colors, interval_labels)
]
fig.legend(
    handles=legend_handles,
    title=r"$R_{\rm cool}/R_{\rm 200c}$ interval",
    loc="upper center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=len(interval_labels),
    frameon=False,
)
axes[-1].set_xlabel("Cold gas fraction")
fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

output_file = figure_dir / "cold_gas_fraction_pdf_by_cooling_radius_snap40_67_99.pdf"
fig.savefig(output_file, format="pdf")
plt.close(fig)
print(f"Saved cold-gas-fraction PDFs to {output_file}")
