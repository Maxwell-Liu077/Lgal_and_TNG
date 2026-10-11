from pathlib import Path
import h5py
import illustris_python as il
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from matplotlib.patches import Rectangle
from scipy import interpolate

BASE_PATH = "/public/share/chenhouzun/TNG50-1/output/"
DATA_DIR = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")
PLOT_DIR = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/pics/ssfr_m200c_classification")
SNAPS = (99, 67, 40)
CLASS_INFO = {
    0: ("Star Burst", "#fd7825"),
    1: ("Main Sequence", "#fde725"),
    2: ("Green Valley", "#35b779"),
    3: ("Quenched", "#440154"),
}


def load_classified_halo_ssfr(base_path, snap, data_dir=DATA_DIR):
    """Load classified FoF halos and reconstruct central-galaxy log sSFR."""
    classification_file = data_dir / f"ssfr_classification_{snap}.hdf5"
    if not classification_file.is_file():
        raise FileNotFoundError(f"sSFR classification file does not exist: {classification_file}")

    with h5py.File(classification_file, "r") as f:
        halo_ids = np.asarray(f["halo_id"][:], dtype=np.int64)
        central_subhalo_ids = np.asarray(f["central_subhalo_id"][:], dtype=np.int64)
        class_codes = np.asarray(f["class_code"][:], dtype=np.int8)
        selection_flag = np.asarray(f["selection_flag"][:], dtype=bool)

    if not (halo_ids.size == central_subhalo_ids.size == class_codes.size == selection_flag.size):
        raise ValueError(f"Classification arrays have inconsistent lengths in {classification_file}")

    header = il.groupcat.loadHeader(base_path, snap)
    hubble_param = float(header["HubbleParam"])

    group_m200c_raw = il.groupcat.loadHalos(base_path, snap, fields=["Group_M_Crit200"])
    group_m200c_msun = np.asarray(group_m200c_raw, dtype=np.float64) * 1.0e10 / hubble_param

    subhalos = il.groupcat.loadSubhalos(base_path, snap, fields=["SubhaloSFR", "SubhaloMassType"])
    subhalo_sfr = np.asarray(subhalos["SubhaloSFR"], dtype=np.float64)
    subhalo_mstar_msun = np.asarray(subhalos["SubhaloMassType"][:, 4], dtype=np.float64) * 1.0e10 / hubble_param
    del subhalos

    classed = selection_flag & np.isin(class_codes, np.array([0, 1, 2, 3], dtype=np.int8))
    ids_in_range = (
        (halo_ids >= 0)
        & (halo_ids < group_m200c_msun.size)
        & (central_subhalo_ids >= 0)
        & (central_subhalo_ids < subhalo_sfr.size)
    )
    rows = np.flatnonzero(classed & ids_in_range)

    selected_halo_ids = halo_ids[rows]
    selected_central_ids = central_subhalo_ids[rows]
    selected_codes = class_codes[rows]
    m200c = group_m200c_msun[selected_halo_ids]
    mstar = subhalo_mstar_msun[selected_central_ids]
    sfr = subhalo_sfr[selected_central_ids]

    valid = (
        np.isfinite(m200c)
        & (m200c > 0.0)
        & np.isfinite(mstar)
        & (mstar > 0.0)
        & np.isfinite(sfr)
        & (sfr >= 0.0)
    )

    log_m200c = np.log10(m200c[valid])
    log_ssfr = np.full(np.count_nonzero(valid), np.nan, dtype=np.float64)
    valid_sfr = sfr[valid] > 0.0
    valid_mstar = mstar[valid]
    log_ssfr[valid_sfr] = np.log10(sfr[valid][valid_sfr]) - np.log10(valid_mstar[valid_sfr])
    selected_codes = selected_codes[valid]
    zero_sfr = ~valid_sfr

    return log_m200c, log_ssfr, selected_codes, zero_sfr


def plot_snapshot(base_path, snap, data_dir=DATA_DIR, plot_dir=PLOT_DIR):
    header = il.groupcat.loadHeader(base_path, snap)
    redshift = float(header["Redshift"])
    log_m200c, log_ssfr, class_codes, zero_sfr = load_classified_halo_ssfr(base_path, snap, data_dir)

    x_min = min(10.8, np.floor(np.min(log_m200c) * 2.0) / 2.0) if log_m200c.size else 10.8
    x_max = max(13.0, np.ceil(np.max(log_m200c) * 2.0) / 2.0) if log_m200c.size else 13.0
    finite_log_ssfr = log_ssfr[np.isfinite(log_ssfr)]
    y_min = min(-16.5, np.floor(np.min(finite_log_ssfr) * 2.0) / 2.0 - 0.5) if finite_log_ssfr.size else -16.5
    y_max = max(-7.0, np.ceil(np.max(finite_log_ssfr) * 2.0) / 2.0 + 0.5) if finite_log_ssfr.size else -7.0
    zero_sfr_y = y_min + 0.18

    with mpl.rc_context({"xtick.top": False}):
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))

    for x0, width in ((10.8, 0.4), (11.8, 0.4), (12.6, 0.4)):
        ax.add_patch(Rectangle((x0, y_min), width, y_max - y_min, color="lightgray", alpha=0.3, zorder=0))

    class_handles = []
    for code, (label, color) in CLASS_INFO.items():
        class_mask = class_codes == code
        positive_sfr = class_mask & ~zero_sfr
        zero_sfr_class = class_mask & zero_sfr
        count = np.count_nonzero(class_mask)

        if count == 0:
            print(f"snap {snap}: {label}: no valid classified halos.")
            continue

        ax.scatter(log_m200c[positive_sfr], log_ssfr[positive_sfr], s=18, color=color, alpha=0.65, linewidths=0, rasterized=True, zorder=2)
        if np.any(zero_sfr_class):
            ax.scatter(log_m200c[zero_sfr_class], np.full(np.count_nonzero(zero_sfr_class), zero_sfr_y), s=28, marker="v", color=color, alpha=0.85, linewidths=0, rasterized=True, zorder=3)

        class_handles.append(mpatches.Patch(color=color, label=f"{label} (N={count:,})"))
        print(f"snap {snap}: {label}: {count:,} halos; {np.count_nonzero(zero_sfr_class):,} have SFR=0.")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([11, 12, 13])
    ax.set_xticklabels([11, 12, 13])
    ax.minorticks_on()
    ax.grid(which="major", axis="y", zorder=1)

    group_m = np.array([6.86093282699585, 7.398449420928955, 7.901801109313965, 8.393900871276855, 8.886045455932617, 9.339580535888672, 9.809768676757812, 10.323699951171875, 10.840004920959473, 11.344971656799316, 11.868677139282227, 12.379907608032227, 12.879098892211914, 13.441740036010742, 13.891986846923828])
    gal_m = np.array([4.742977619171143, 4.718081474304199, 4.761687278747559, 4.795966625213623, 4.861518859863281, 5.147671699523926, 6.013262748718262, 7.66148042678833, 8.850899696350098, 9.708621978759766, 10.465829849243164, 11.046808242797852, 11.511899948120117, 11.899462699890137, 12.20790958404541])

    secondary_ax = ax.twiny()
    halo_mass_from_stellar_mass = interpolate.interp1d(gal_m, group_m)
    stellar_mass_ticks = np.array([9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0])
    secondary_ax.set_xlim(ax.get_xlim())
    secondary_ax.set_xticks(halo_mass_from_stellar_mass(stellar_mass_ticks))
    secondary_ax.set_xticklabels(stellar_mass_ticks)
    secondary_ax.minorticks_off()
    secondary_ax.set_xlabel(r"stellar mass [$\log_{10}(M_\odot)$]")

    if class_handles:
        class_handles.append(Line2D([], [], color="black", marker="v", linestyle="None", label="SFR=0 (shown at lower plotting limit)"))
        ax.legend(handles=class_handles, ncol=1, loc="upper right", fontsize=15)

    ax.text(0.05, 0.95, rf"$z={redshift:.2f}$", transform=ax.transAxes, ha="left", va="top", fontsize=18, bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"})
    ax.set_xlabel(r"halo mass [$\log_{10}(M_{200c}/M_\odot)$]")
    ax.set_ylabel(r"$\log_{10}(\mathrm{sSFR}/\mathrm{yr}^{-1})$")
    fig.tight_layout()

    plot_dir.mkdir(parents=True, exist_ok=True)
    output_file = plot_dir / f"ssfr_vs_m200c_snap{snap}_50-1.pdf"
    fig.savefig(output_file, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_file}")
    return output_file


def main():
    style_file = Path(__file__).resolve().parents[1] / "fancy_plots2.mplstyle"
    plt.style.use(style_file)
    for snap in SNAPS:
        plot_snapshot(BASE_PATH, snap)


if __name__ == "__main__":
    main()
