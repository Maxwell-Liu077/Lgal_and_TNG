"""按质量组绘制热气体密度、温度和金属丰度的中位径向剖面。"""

from pathlib import Path
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

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
    mass_bins = f["mass_bin"].asstr()[:]
    r200c_kpc = np.asarray(f["r200c_physical_kpc"][:], dtype=np.float64)
    hot_mass_msun = np.asarray(f["hot_gas_mass_msun_within_r200c"][:], dtype=np.float64)
    interpolation_radii = np.asarray(f["interpolation_radii_r200c"][:], dtype=np.float64)
    rho_hot_cell_interpolated = np.asarray(f["rho_hot_cell_mean_g_cm3_interpolated"][:], dtype=np.float64)
    temperature_interpolated = np.asarray(f["temperature_mass_weighted_K_interpolated"][:], dtype=np.float64)
    metallicity_interpolated = np.asarray(f["metallicity_mass_weighted_interpolated"][:], dtype=np.float64)
    global_temperature = np.asarray(f["global_temperature_mass_weighted_K"][:], dtype=np.float64)
    global_metallicity = np.asarray(f["global_metallicity_mass_weighted"][:], dtype=np.float64)

if not (mass_bins.size == r200c_kpc.size == hot_mass_msun.size == rho_hot_cell_interpolated.shape[0] == temperature_interpolated.shape[0] == metallicity_interpolated.shape[0] == global_temperature.size == global_metallicity.size):
    raise ValueError(f"Halo rows are not aligned in {input_file}.")

msun_g = 1.98847e33
kpc_cm = 3.085677581491367e21
log_interpolation_radii = np.log10(interpolation_radii)
log_radial_limits = np.log10(interpolation_radii[[0, -1]])
mass_groups = [("Dwarfs", "Dwarf", "tab:blue"), ("Milky_Ways", "MW", "tab:orange"), ("Groups", "Group", "tab:green")]

# Global isothermal density uses each halo's total hot-gas mass within R200c.
rho_global_isothermal = np.full((mass_bins.size, interpolation_radii.size), np.nan, dtype=np.float64)
valid_global_density = np.isfinite(hot_mass_msun) & (hot_mass_msun > 0.0) & np.isfinite(r200c_kpc) & (r200c_kpc > 0.0)
r200c_cm = r200c_kpc[valid_global_density] * kpc_cm
rho_global_isothermal[valid_global_density] = hot_mass_msun[valid_global_density, None] * msun_g / (4.0 * np.pi * r200c_cm[:, None] ** 3 * interpolation_radii[None, :] ** 2)

fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharex=True)
fig.suptitle(f"Hot-gas radial profiles (snap {snap})", fontsize=26)

for mass_bin_name, sample_name, color in mass_groups:
    sample_mask = mass_bins == mass_bin_name
    sample_count = np.count_nonzero(sample_mask)
    if sample_count == 0:
        print(f"No {sample_name} halos found in {input_file}.", flush=True)
        continue

    rho_global_sample = rho_global_isothermal[sample_mask]
    rho_global_valid = np.isfinite(rho_global_sample) & (rho_global_sample > 0.0)
    rho_global_median = np.full(interpolation_radii.size, np.nan, dtype=np.float64)
    rho_global_columns = np.any(rho_global_valid, axis=0)
    if np.any(rho_global_columns):
        rho_global_median[rho_global_columns] = np.nanmedian(np.where(rho_global_valid[:, rho_global_columns], rho_global_sample[:, rho_global_columns], np.nan), axis=0)
    if np.any(np.isfinite(rho_global_median)):
        axes[0].plot(log_interpolation_radii, rho_global_median, color=color, linestyle="dashed", linewidth=2.2)

    rho_shell_sample = rho_hot_cell_interpolated[sample_mask]
    rho_shell_valid = np.isfinite(rho_shell_sample) & (rho_shell_sample > 0.0)
    rho_shell_median = np.full(interpolation_radii.size, np.nan, dtype=np.float64)
    rho_shell_columns = np.any(rho_shell_valid, axis=0)
    if np.any(rho_shell_columns):
        rho_shell_median[rho_shell_columns] = np.nanmedian(np.where(rho_shell_valid[:, rho_shell_columns], rho_shell_sample[:, rho_shell_columns], np.nan), axis=0)
    if np.any(np.isfinite(rho_shell_median)):
        axes[0].plot(log_interpolation_radii, rho_shell_median, color=color, linestyle="solid", linewidth=2.2)

    temperature_sample = temperature_interpolated[sample_mask]
    temperature_valid = np.isfinite(temperature_sample) & (temperature_sample > 0.0)
    temperature_median = np.full(interpolation_radii.size, np.nan, dtype=np.float64)
    temperature_columns = np.any(temperature_valid, axis=0)
    if np.any(temperature_columns):
        temperature_median[temperature_columns] = np.nanmedian(np.where(temperature_valid[:, temperature_columns], temperature_sample[:, temperature_columns], np.nan), axis=0)
    if np.any(np.isfinite(temperature_median)):
        axes[1].plot(log_interpolation_radii, temperature_median, color=color, linestyle="solid", linewidth=2.2)
    global_temperature_values = global_temperature[sample_mask]
    global_temperature_values = global_temperature_values[np.isfinite(global_temperature_values) & (global_temperature_values > 0.0)]
    if global_temperature_values.size:
        axes[1].axhline(np.median(global_temperature_values), color=color, linestyle="dashed", linewidth=2.2)

    metallicity_sample = metallicity_interpolated[sample_mask]
    metallicity_valid = np.isfinite(metallicity_sample) & (metallicity_sample >= 0.0)
    metallicity_median = np.full(interpolation_radii.size, np.nan, dtype=np.float64)
    metallicity_columns = np.any(metallicity_valid, axis=0)
    if np.any(metallicity_columns):
        metallicity_median[metallicity_columns] = np.nanmedian(np.where(metallicity_valid[:, metallicity_columns], metallicity_sample[:, metallicity_columns], np.nan), axis=0)
    if np.any(np.isfinite(metallicity_median)):
        axes[2].plot(log_interpolation_radii, metallicity_median, color=color, linestyle="solid", linewidth=2.2)
    global_metallicity_values = global_metallicity[sample_mask]
    global_metallicity_values = global_metallicity_values[np.isfinite(global_metallicity_values) & (global_metallicity_values >= 0.0)]
    if global_metallicity_values.size:
        axes[2].axhline(np.median(global_metallicity_values), color=color, linestyle="dashed", linewidth=2.2)

axes[0].set_title("Density", fontsize=22)
axes[0].set_ylabel(r"$\rho_{\rm hot}$ [$\mathrm{g\,cm^{-3}}$]")
axes[0].set_yscale("log")
axes[1].set_title("Temperature", fontsize=22)
axes[1].set_ylabel(r"$T_{\rm hot}$ [$\mathrm{K}$]")
axes[1].set_yscale("log")
axes[2].set_title("Metallicity", fontsize=22)
axes[2].set_ylabel(r"$Z_{\rm hot}$ (metal mass fraction)")
axes[2].set_ylim(bottom=0.0)
axes[2].ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)

for ax in axes:
    ax.set_xlabel(r"$\log_{10}(r/R_{\rm 200c})$")
    ax.set_xlim(log_radial_limits)
    ax.set_xticks(np.arange(-2.0, 0.01, 0.5))
    ax.minorticks_on()
    ax.grid(which="major", axis="y", alpha=0.25)

color_handles = [Line2D([], [], color=color, linestyle="solid", linewidth=2.5, label=sample_name) for _, sample_name, color in mass_groups]
style_handles = [Line2D([], [], color="black", linestyle="dashed", linewidth=2.2, label="Global"), Line2D([], [], color="black", linestyle="solid", linewidth=2.2, label="Shell")]
fig.legend(handles=color_handles + style_handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=5, frameon=True, facecolor="white", framealpha=0.75, edgecolor="none", fontsize=15)
fig.tight_layout(rect=(0.0, 0.11, 1.0, 0.93))
output_file = figure_dir / f"properties_tng_Lgal_snap{snap}.pdf"
fig.savefig(output_file, format="pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved median halo-property profiles to {output_file}", flush=True)
