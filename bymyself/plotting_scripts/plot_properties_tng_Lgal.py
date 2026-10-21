"""每个样本生成密度、温度、金属丰度三联图；可用命令行参数指定快照，默认 snap 99。"""

from pathlib import Path
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).resolve().parents[1]
plt.style.use(project_root / "fancy_plots2.mplstyle")

snap = int(sys.argv[1]) if len(sys.argv) > 1 else 99
data_dir = project_root / "data"
figure_dir = project_root / "pics" / "properties_tng_Lgal"
input_file = data_dir / f"real_physical_properties_{snap}.hdf5"
if not input_file.is_file():
    raise FileNotFoundError(f"Radial-profile file does not exist: {input_file}. Run real_physical_properties first.")
figure_dir.mkdir(parents=True, exist_ok=True)

with h5py.File(input_file, "r") as f:
    fof_ids = np.asarray(f["fof_id"][:], dtype=np.int64)
    mass_bins = f["mass_bin"].asstr()[:]
    r200c_kpc = np.asarray(f["r200c_physical_kpc"][:], dtype=np.float64)
    hot_mass_msun = np.asarray(f["hot_gas_mass_msun_within_r200c"][:], dtype=np.float64)
    radial_edges = np.asarray(f["radial_bin_edges_r200c"][:], dtype=np.float64)
    radial_centers = np.asarray(f["radial_bin_centers_r200c"][:], dtype=np.float64)
    interpolation_radii = np.asarray(f["interpolation_radii_r200c"][:], dtype=np.float64)
    rho_cell = np.asarray(f["rho_hot_cell_mean_g_cm3"][:], dtype=np.float64)
    rho_shell = np.asarray(f["rho_shell_mean_g_cm3"][:], dtype=np.float64)
    temperature = np.asarray(f["temperature_mass_weighted_K"][:], dtype=np.float64)
    metallicity = np.asarray(f["metallicity_mass_weighted"][:], dtype=np.float64)
    rho_cell_interpolated = np.asarray(f["rho_hot_cell_mean_g_cm3_interpolated"][:], dtype=np.float64)
    rho_shell_interpolated = np.asarray(f["rho_shell_mean_g_cm3_interpolated"][:], dtype=np.float64)
    temperature_interpolated = np.asarray(f["temperature_mass_weighted_K_interpolated"][:], dtype=np.float64)
    metallicity_interpolated = np.asarray(f["metallicity_mass_weighted_interpolated"][:], dtype=np.float64)
    global_temperature = np.asarray(f["global_temperature_mass_weighted_K"][:], dtype=np.float64)
    global_metallicity = np.asarray(f["global_metallicity_mass_weighted"][:], dtype=np.float64)

msun_g = 1.98847e33
kpc_cm = 3.085677581491367e21
log_radial_centers = np.log10(radial_centers)
log_interpolation_radii = np.log10(interpolation_radii)
log_radial_limits = np.log10(radial_edges[[0, -1]])
sample_names = {"Dwarfs": "Dwarf", "Milky_Ways": "MW", "Groups": "Group"}

for object_index, fof_id in enumerate(fof_ids):
    sample_name = sample_names.get(mass_bins[object_index], mass_bins[object_index])
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharex=True)
    fig.suptitle(f"{sample_name}, halo {fof_id} (snap {snap})", fontsize=26)

    # 将 rho(r) = M_hot / (4*pi*R200c*r^2) 在 0 < r <= R200c 内归一化到全部热气体质量。
    rho_isothermal = np.full(interpolation_radii.shape, np.nan, dtype=np.float64)
    if np.isfinite(hot_mass_msun[object_index]) and hot_mass_msun[object_index] > 0.0 and np.isfinite(r200c_kpc[object_index]) and r200c_kpc[object_index] > 0.0:
        r200c_cm = r200c_kpc[object_index] * kpc_cm
        rho_isothermal = hot_mass_msun[object_index] * msun_g / (4.0 * np.pi * r200c_cm ** 3 * interpolation_radii ** 2)
    axes[0].plot(log_interpolation_radii, rho_isothermal, color="black", linestyle="dashed", label=r"Isothermal ($r^{-2}$)")

    density_series = [(rho_cell[object_index], rho_cell_interpolated[object_index], "tab:orange", "Hot-cell volume"), (rho_shell[object_index], rho_shell_interpolated[object_index], "tab:blue", "Full-shell volume")]
    for raw_profile, interpolated_profile, color, label in density_series:
        raw_values = np.where(np.isfinite(raw_profile) & (raw_profile > 0.0), raw_profile, np.nan)
        interpolated_values = np.where(np.isfinite(interpolated_profile) & (interpolated_profile > 0.0), interpolated_profile, np.nan)
        axes[0].plot(log_interpolation_radii, interpolated_values, color=color, linestyle="solid", label=label)
        axes[0].plot(log_radial_centers, raw_values, color=color, marker="o", markersize=4, linestyle="none")
    axes[0].set_title("Density", fontsize=22)
    axes[0].set_ylabel(r"$\rho_{\rm hot}$ [$\mathrm{g\,cm^{-3}}$]")
    axes[0].set_yscale("log")

    if np.isfinite(global_temperature[object_index]) and global_temperature[object_index] > 0.0:
        axes[1].axhline(global_temperature[object_index], color="black", linestyle="dashed", label="Global mass-weighted")
    temperature_values = np.where(np.isfinite(temperature[object_index]) & (temperature[object_index] > 0.0), temperature[object_index], np.nan)
    temperature_line = np.where(np.isfinite(temperature_interpolated[object_index]) & (temperature_interpolated[object_index] > 0.0), temperature_interpolated[object_index], np.nan)
    axes[1].plot(log_interpolation_radii, temperature_line, color="tab:blue", linestyle="solid", label="Shell mass-weighted")
    axes[1].plot(log_radial_centers, temperature_values, color="tab:blue", marker="o", markersize=4, linestyle="none")
    axes[1].set_title("Temperature", fontsize=22)
    axes[1].set_ylabel(r"$T_{\rm hot}$ [$\mathrm{K}$]")
    axes[1].set_yscale("log")

    if np.isfinite(global_metallicity[object_index]) and global_metallicity[object_index] >= 0.0:
        axes[2].axhline(global_metallicity[object_index], color="black", linestyle="dashed", label="Global mass-weighted")
    metallicity_values = np.where(np.isfinite(metallicity[object_index]) & (metallicity[object_index] >= 0.0), metallicity[object_index], np.nan)
    metallicity_line = np.where(np.isfinite(metallicity_interpolated[object_index]) & (metallicity_interpolated[object_index] >= 0.0), metallicity_interpolated[object_index], np.nan)
    axes[2].plot(log_interpolation_radii, metallicity_line, color="tab:blue", linestyle="solid", label="Shell mass-weighted")
    axes[2].plot(log_radial_centers, metallicity_values, color="tab:blue", marker="o", markersize=4, linestyle="none")
    axes[2].set_title("Metallicity", fontsize=22)
    axes[2].set_ylabel(r"$Z_{\rm hot}$ (metal mass fraction)")
    axes[2].set_ylim(bottom=0.0)
    axes[2].ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)

    for ax in axes:
        ax.set_xlabel(r"$\log_{10}(r/R_{\rm 200c})$")
        ax.set_xlim(log_radial_limits)
        ax.set_xticks(np.arange(log_radial_limits[0], log_radial_limits[1] + 0.25, 0.5))
        ax.minorticks_on()
        ax.grid(which="major", axis="y", alpha=0.25)
        ax.legend(loc="best", fontsize=15, frameon=True, facecolor="white", framealpha=0.75, edgecolor="none")

    raw_panels = [np.concatenate((rho_cell[object_index], rho_shell[object_index])), temperature[object_index], metallicity[object_index]]
    for ax, raw_values in zip(axes, raw_panels):
        if not np.any(np.isfinite(raw_values)):
            ax.text(0.5, 0.5, "No valid shell data", transform=ax.transAxes, ha="center", va="center", fontsize=16)
    if not any(np.any(np.isfinite(values)) for values in raw_panels):
        print(f"halo {fof_id}: no valid shell profiles; check and regenerate {input_file} if necessary.", flush=True)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    output_file = figure_dir / f"properties_tng_Lgal_snap{snap}_{sample_name}_halo{fof_id}.pdf"
    fig.savefig(output_file, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved halo-property comparison to {output_file}", flush=True)
