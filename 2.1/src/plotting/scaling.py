"""AGN cooling-radius and feedback-before-accretion figures."""

from __future__ import annotations

import numpy as np

from .style import ACADEMIC_STYLE, save_figure


def _plt():
    """Import Matplotlib lazily for data-only workflows."""

    import matplotlib.pyplot as plt
    plt.rcParams.update(ACADEMIC_STYLE)
    return plt


def plot_agn_cooling(statistics: dict[str, np.ndarray], *, halo_results: dict[str, np.ndarray] | None = None, save_path=None):
    """Plot four AGN-quartile xcool curves with optional halo scatter."""

    plt = _plt()
    figure, axis = plt.subplots(figsize=(6.2, 4.4))
    x = np.asarray(statistics["mass_bin_center"])
    medians = np.asarray(statistics["median"])
    p16 = np.asarray(statistics.get("p16", np.full_like(medians, np.nan)))
    p84 = np.asarray(statistics.get("p84", np.full_like(medians, np.nan)))
    if halo_results is not None:
        halo_values = np.asarray(halo_results.get("x_cool_mean", []), dtype=float)
        halo_bins = np.asarray(halo_results.get("mass_bin_index", []), dtype=int)
        halo_quartiles = np.asarray(halo_results.get("agn_quartile", []), dtype=int)
        for quartile in range(1, 5):
            mask = (halo_quartiles == quartile) & np.isfinite(halo_values)
            valid_bins = mask & (halo_bins >= 0) & (halo_bins < len(x))
            axis.scatter(x[halo_bins[valid_bins]], halo_values[valid_bins], s=8, alpha=0.25, color=f"C{quartile - 1}")
    for quartile in range(4):
        axis.plot(x, medians[quartile], marker="o", label=f"Q{quartile + 1}")
        valid = np.isfinite(p16[quartile]) & np.isfinite(p84[quartile])
        if np.any(valid):
            axis.fill_between(x[valid], p16[quartile, valid], p84[quartile, valid], alpha=0.12)
    axis.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$")
    axis.set_ylabel(r"$\langle r_{\rm cool}/R_{200c}\rangle$")
    axis.legend(frameon=False)
    save_figure(figure, save_path)
    return figure


def plot_feedback_before_accretion(
    normalized_statistics: dict[str, dict[str, np.ndarray]],
    *,
    save_path=None,
):
    """Plot pf-in and SAM cooling for Mhot and Mstar normalizations."""

    plt = _plt()
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=False)
    for axis, normalization in zip(axes, ("mhot", "mstar")):
        stats = normalized_statistics[normalization]
        x = stats["mass_bin_center"]
        axis.plot(x, stats["pf_in_median"], label=r"$\dot M_{\rm pf,in}$")
        for key, color in (("pf_in", "C0"), ("cool_iso", "C1")):
            p16_key = f"{key}_p16"
            p84_key = f"{key}_p84"
            if p16_key in stats and p84_key in stats:
                p16 = np.asarray(stats[p16_key], dtype=float)
                p84 = np.asarray(stats[p84_key], dtype=float)
                valid = np.isfinite(p16) & np.isfinite(p84)
                if np.any(valid):
                    axis.fill_between(x[valid], p16[valid], p84[valid], color=color, alpha=0.12)
        if "cool_iso_median" in stats:
            axis.plot(x, stats["cool_iso_median"], label=r"$\dot M_{\rm cool,iso}$")
        axis.set_title(normalization)
        axis.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$")
        axis.legend(frameon=False)
    save_figure(figure, save_path)
    return figure
