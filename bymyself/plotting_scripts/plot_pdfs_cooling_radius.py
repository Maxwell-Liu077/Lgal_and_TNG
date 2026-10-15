import h5py
import illustris_python as il
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.style.use("fancy_plots2.mplstyle")

basePath = "/public/share/chenhouzun/TNG50-1/output/"
data_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")
figure_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/pics/cooling_radius")
figure_dir.mkdir(parents=True, exist_ok=True)

snaps = [40, 67, 99]
colors = {40: "tab:blue", 67: "tab:orange", 99: "tab:green"}
bin_edges = np.linspace(0.0, 5.0, 41)  # 40 bins, each 0.125 wide

fig, ax = plt.subplots(1, 1, figsize=(14, 8))
plotted_snaps = []

for snap in snaps:
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
    if not cooling_file.is_file():
        raise FileNotFoundError(f"Cooling-radius file does not exist: {cooling_file}")

    with h5py.File(cooling_file, "r") as f:
        halo_flag = np.asarray(f["halo_flag"][:], dtype=bool)
        cooling_radius = np.asarray(f["cooling_radius"][:], dtype=np.float64)

    if not (halo_flag.size == group_masses.size == central_mass_ok.size == cooling_radius.size):
        raise ValueError(f"Halo-catalogue and cooling-radius array lengths differ at snap {snap}.")

    plot_mask = (
        halo_flag
        & central_mass_ok
        & np.isfinite(group_masses)
        & (group_masses > 0.0)
        & np.isfinite(cooling_radius)
    )
    selected = cooling_radius[plot_mask]
    selected_in_range = selected[(selected >= 0.0) & (selected <= 5.0)]

    if selected_in_range.size == 0:
        print(f"No selected halos have 0 <= R_cool/R_200c <= 5 at snap {snap}.")
        continue

    ax.hist(
        selected_in_range,
        bins=bin_edges,
        density=True,
        histtype="step",
        linewidth=3.2,
        color=colors[snap],
        label=f"snap {snap}",
    )
    plotted_snaps.append(snap)
    print(
        f"snap {snap}: {selected.size:,} halos pass the sample selection; "
        f"{selected_in_range.size:,} lie in the plotted range."
    )

ax.set_xlim(0.0, 5.0)
ax.set_xlabel(r"Cooling radius $R_{\rm cool}/R_{\rm 200c}$")
ax.set_ylabel("PDF")
ax.grid(axis="y", alpha=0.25)
if plotted_snaps:
    ax.legend(loc="best", frameon=False)

fig.tight_layout()
output_file = figure_dir / "cooling_radius_pdf_snap40_67_99.pdf"
fig.savefig(output_file, format="pdf")
plt.close(fig)
print(f"Saved cooling-radius PDF plot to {output_file}")
