from pathlib import Path
import h5py
import illustris_python as il
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

project_root = Path(__file__).resolve().parents[1]
plt.style.use(project_root / "fancy_plots2.mplstyle")

basePath = "/public/share/chenhouzun/TNG50-1/output/"
snaps = [40, 67, 99]
data_dir = project_root / "data"
figure_dir = project_root / "pics" / "properties_tng_Lgal"
figure_dir.mkdir(parents=True, exist_ok=True)
mass_groups = [("Dwarfs", "dwarfs", "tab:blue"), ("Milky_Ways", "MW-like", "tab:orange"), ("Groups", "groups", "tab:green")]
percentile_band_alpha = 0.16


def profile_statistics(values, valid):
    median = np.full(values.shape[1], np.nan, dtype=np.float64)
    lower = np.full(values.shape[1], np.nan, dtype=np.float64)
    upper = np.full(values.shape[1], np.nan, dtype=np.float64)
    valid_columns = np.any(valid, axis=0)
    if np.any(valid_columns):
        profile_values = np.where(valid[:, valid_columns], values[:, valid_columns], np.nan)
        median[valid_columns] = np.nanmedian(profile_values, axis=0)
        lower[valid_columns], upper[valid_columns] = np.nanpercentile(profile_values, [16, 84], axis=0)
    return median, lower, upper


snapshot_data = {}
redshifts = {}
for snap in snaps:
    input_file = data_dir / f"real_physical_properties_{snap}.hdf5"
    if not input_file.is_file():
        raise FileNotFoundError(f"Radial-profile file does not exist: {input_file}. Run real_physical_properties first.")

    with h5py.File(input_file, "r") as f:
        data = {
            "mass_bins": f["mass_bin"].asstr()[:],
            "r200c_kpc": np.asarray(f["r200c_physical_kpc"][:], dtype=np.float64),
            "hot_mass_msun": np.asarray(f["hot_gas_mass_msun_within_r200c"][:], dtype=np.float64),
            "interpolation_radii": np.asarray(f["interpolation_radii_r200c"][:], dtype=np.float64),
            "rho_hot": np.asarray(f["rho_hot_cell_mean_g_cm3_interpolated"][:], dtype=np.float64),
            "temperature": np.asarray(f["temperature_mass_weighted_K_interpolated"][:], dtype=np.float64),
            "metallicity": np.asarray(f["metallicity_mass_weighted_interpolated"][:], dtype=np.float64),
            "global_temperature": np.asarray(f["global_temperature_mass_weighted_K"][:], dtype=np.float64),
            "global_metallicity": np.asarray(f["global_metallicity_mass_weighted"][:], dtype=np.float64),
        }

    row_count = data["mass_bins"].size
    snapshot_data[snap] = data
    redshifts[snap] = float(il.groupcat.loadHeader(basePath, snap)["Redshift"])

interpolation_radii = snapshot_data[snaps[0]]["interpolation_radii"]

msun_g = 1.98847e33
kpc_cm = 3.085677581491367e21
log_interpolation_radii = np.log10(interpolation_radii)
log_radial_limits = np.log10(interpolation_radii[[0, -1]])

fig, axes = plt.subplots(3, 3, figsize=(20, 13), sharex=True, sharey="row")

for column, snap in enumerate(snaps):
    data = snapshot_data[snap]
    ax_density, ax_temperature, ax_metallicity = axes[:, column]
    rho_global_isothermal = np.full((data["mass_bins"].size, interpolation_radii.size), np.nan, dtype=np.float64)
    valid_global_density = np.isfinite(data["hot_mass_msun"]) & (data["hot_mass_msun"] > 0.0) & np.isfinite(data["r200c_kpc"]) & (data["r200c_kpc"] > 0.0)
    r200c_cm = data["r200c_kpc"][valid_global_density] * kpc_cm
    rho_global_isothermal[valid_global_density] = data["hot_mass_msun"][valid_global_density, None] * msun_g / (4.0 * np.pi * r200c_cm[:, None] ** 3 * interpolation_radii[None, :] ** 2)

    for mass_bin_name, sample_name, color in mass_groups:
        sample_mask = data["mass_bins"] == mass_bin_name
        sample_count = np.count_nonzero(sample_mask)
        if sample_count == 0:
            print(f"No {sample_name} halos found at snap {snap}.", flush=True)
            continue

        rho_global_sample = rho_global_isothermal[sample_mask]
        rho_global_valid = np.isfinite(rho_global_sample) & (rho_global_sample > 0.0)
        rho_global_median, _, _ = profile_statistics(rho_global_sample, rho_global_valid)
        if np.any(np.isfinite(rho_global_median)):
            ax_density.plot(log_interpolation_radii, rho_global_median, color=color, linestyle="dashed", linewidth=2.0)

        rho_shell_sample = data["rho_hot"][sample_mask]
        rho_shell_valid = np.isfinite(rho_shell_sample) & (rho_shell_sample > 0.0)
        rho_shell_median, rho_shell_p16, rho_shell_p84 = profile_statistics(rho_shell_sample, rho_shell_valid)
        if np.any(np.isfinite(rho_shell_median)):
            ax_density.fill_between(log_interpolation_radii, rho_shell_p16, rho_shell_p84, color=color, alpha=percentile_band_alpha, linewidth=0)
            ax_density.plot(log_interpolation_radii, rho_shell_median, color=color, linestyle="solid", linewidth=2.0)

        temperature_sample = data["temperature"][sample_mask]
        temperature_valid = np.isfinite(temperature_sample) & (temperature_sample > 0.0)
        temperature_median, temperature_p16, temperature_p84 = profile_statistics(temperature_sample, temperature_valid)
        if np.any(np.isfinite(temperature_median)):
            ax_temperature.fill_between(log_interpolation_radii, temperature_p16, temperature_p84, color=color, alpha=percentile_band_alpha, linewidth=0)
            ax_temperature.plot(log_interpolation_radii, temperature_median, color=color, linestyle="solid", linewidth=2.0)
        global_temperature = data["global_temperature"][sample_mask]
        global_temperature = global_temperature[np.isfinite(global_temperature) & (global_temperature > 0.0)]
        if global_temperature.size:
            ax_temperature.axhline(np.median(global_temperature), color=color, linestyle="dashed", linewidth=2.0)

        metallicity_sample = data["metallicity"][sample_mask]
        metallicity_valid = np.isfinite(metallicity_sample) & (metallicity_sample >= 0.0)
        metallicity_median, metallicity_p16, metallicity_p84 = profile_statistics(metallicity_sample, metallicity_valid)
        if np.any(np.isfinite(metallicity_median)):
            ax_metallicity.fill_between(log_interpolation_radii, metallicity_p16, metallicity_p84, color=color, alpha=percentile_band_alpha, linewidth=0)
            ax_metallicity.plot(log_interpolation_radii, metallicity_median, color=color, linestyle="solid", linewidth=2.0)
        global_metallicity = data["global_metallicity"][sample_mask]
        global_metallicity = global_metallicity[np.isfinite(global_metallicity) & (global_metallicity >= 0.0)]
        if global_metallicity.size:
            ax_metallicity.axhline(np.median(global_metallicity), color=color, linestyle="dashed", linewidth=2.0)

    axes[0, column].set_title(rf"$z={redshifts[snap]:.2f}$", fontsize=21)

axes[0, 0].set_ylabel(r"$\rho_{\rm hot}$ [$\mathrm{g\,cm^{-3}}$]", fontsize=17)
axes[1, 0].set_ylabel(r"$T_{\rm hot}$ [$\mathrm{K}$]", fontsize=17)
axes[2, 0].set_ylabel(r"$Z_{\rm hot}$", fontsize=17)
axes[0, 0].set_yscale("log")
axes[1, 0].set_yscale("log")
axes[2, 0].set_ylim(bottom=0.0)

for ax in axes[2, :]:
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)

for row in range(3):
    for column in range(3):
        ax = axes[row, column]
        ax.set_xlim(log_radial_limits)
        ax.set_xticks(np.arange(-2.0, 0.01, 0.5))
        ax.minorticks_on()
        ax.grid(which="major", axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelbottom=(row == 2))
        if column > 0:
            ax.tick_params(axis="y", labelleft=False)

fig.supxlabel(r"$\log_{10}(r/R_{\rm 200c})$", fontsize=19, y=0.075)
color_handles = [Line2D([], [], color=color, linestyle="solid", linewidth=2.5, label=sample_name) for _, sample_name, color in mass_groups]
style_handles = [Line2D([], [], color="black", linestyle="dashed", linewidth=2.0, label="Global"), Line2D([], [], color="black", linestyle="solid", linewidth=2.0, label="Shell")]
fig.legend(handles=color_handles + style_handles, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=5, frameon=True, facecolor="white", framealpha=0.75, edgecolor="none", fontsize=14)
fig.tight_layout(rect=(0.0, 0.13, 1.0, 0.96))
output_file = figure_dir / "properties_tng_Lgal_snaps_40_67_99.pdf"
fig.savefig(output_file, format="pdf", bbox_inches="tight")
plt.close(fig)