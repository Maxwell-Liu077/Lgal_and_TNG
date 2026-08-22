"""AGN cooling-radius and feedback-before-accretion figures."""

from __future__ import annotations

import numpy as np

from .style import ACADEMIC_STYLE, save_figure


def _plt():
    """Import Matplotlib lazily for data-only workflows."""

    import matplotlib.pyplot as plt
    plt.rcParams.update(ACADEMIC_STYLE)
    return plt


def _plot_scatter_errorbars(axis, x, median, p16, p84, *, color, offset=0.0, label=None):
    """Draw asymmetric P16/P84 scatter around a zero median baseline."""

    x = np.asarray(x, dtype=float)
    median = np.asarray(median, dtype=float)
    p16 = np.asarray(p16, dtype=float)
    p84 = np.asarray(p84, dtype=float)
    lower = median - p16
    upper = p84 - median
    valid = (
        np.isfinite(x) & np.isfinite(lower) & np.isfinite(upper)
        & (lower >= 0) & (upper >= 0)
    )
    if np.any(valid):
        axis.errorbar(
            x[valid] + offset,
            np.zeros(np.count_nonzero(valid)),
            yerr=np.vstack((lower[valid], upper[valid])),
            fmt="_",
            markersize=9,
            color=color,
            ecolor=color,
            elinewidth=1.8,
            capsize=3.5,
            capthick=1.8,
            label=label,
        )


def _style_scatter_axis(axis, ylabel):
    """Match the README reference: boxed panel and dashed zero baseline."""

    axis.axhline(0, color="0.45", linewidth=1.0, linestyle=(0, (6, 5)), zorder=0)
    axis.set_ylabel(ylabel)
    for spine in axis.spines.values():
        spine.set_visible(True)
    axis.tick_params(which="both", direction="in", top=True, right=True)


def plot_agn_cooling(statistics: dict[str, np.ndarray], *, halo_results: dict[str, np.ndarray] | None = None, save_path=None):
    """Plot AGN-quartile medians and a separate 16--84% scatter panel."""

    plt = _plt()
    figure, (axis, scatter_axis) = plt.subplots(
        2, 1, figsize=(8.8, 6.5), sharex=True,
        gridspec_kw={"height_ratios": (3.0, 1.35), "hspace": 0.08},
    )
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
            offset = (quartile - 2.5) * 0.018
            axis.scatter(x[halo_bins[valid_bins]] + offset, halo_values[valid_bins], s=16, alpha=0.22, color=f"C{quartile - 1}", linewidths=0)
    for quartile in range(4):
        axis.plot(x, medians[quartile], label=f"Q{quartile + 1}")
        _plot_scatter_errorbars(
            scatter_axis,
            x,
            medians[quartile],
            p16[quartile],
            p84[quartile],
            color=f"C{quartile}",
            offset=(quartile - 1.5) * 0.012,
            label=f"Q{quartile + 1}",
        )
    axis.set_ylabel(r"$\langle r_{\rm cool}/R_{200c}\rangle$")
    axis.set_title(r"AGN-strength quartiles", loc="left")
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    scatter_axis.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$")
    _style_scatter_axis(scatter_axis, r"$1\sigma$ scatter")
    figure.subplots_adjust(left=0.12, right=0.80, bottom=0.13, top=0.90, hspace=0.08)
    save_figure(figure, save_path)
    return figure


def plot_feedback_before_accretion(
    normalized_statistics: dict[str, dict[str, np.ndarray]],
    *,
    halo_results: dict[str, np.ndarray] | None = None,
    save_path=None,
):
    """Plot both normalizations with medians and separate scatter panels."""

    plt = _plt()
    figure, axes = plt.subplots(
        2, 2, figsize=(12.6, 7.2), sharex="col",
        gridspec_kw={"height_ratios": (3.0, 1.35), "hspace": 0.08},
    )
    mass_fields = {"mhot": "m_hot_mean_msun", "mstar": "mstar_mean_msun"}
    for column, normalization in enumerate(("mhot", "mstar")):
        axis = axes[0, column]
        scatter_axis = axes[1, column]
        stats = normalized_statistics[normalization]
        x = np.asarray(stats["mass_bin_center"], dtype=float)
        if halo_results is not None:
            bins = np.asarray(halo_results.get("mass_bin_index", []), dtype=int)
            masses = np.asarray(halo_results.get(mass_fields[normalization], []), dtype=float)
            valid_mass = np.isfinite(masses) & (masses > 0) & (bins >= 0) & (bins < len(x))
            for field, color in (("rate_pf_in_msun_per_yr", "C0"), ("rate_cool_iso_msun_per_yr", "C1")):
                rates = np.asarray(halo_results.get(field, []), dtype=float)
                valid = valid_mass & np.isfinite(rates)
                values = rates[valid] / masses[valid] * 1.0e9
                axis.scatter(x[bins[valid]], values, color=color, s=14, alpha=0.18, linewidths=0)
        axis.plot(x, stats["pf_in_median"], color="C0", label=r"$\dot M_{\rm pf,in}$")
        for series_index, (key, color) in enumerate((("pf_in", "C0"), ("cool_iso", "C1"))):
            p16_key = f"{key}_p16"
            p84_key = f"{key}_p84"
            if p16_key in stats and p84_key in stats:
                p16 = np.asarray(stats[p16_key], dtype=float)
                p84 = np.asarray(stats[p84_key], dtype=float)
                median = np.asarray(stats[f"{key}_median"], dtype=float)
                _plot_scatter_errorbars(
                    scatter_axis,
                    x,
                    median,
                    p16,
                    p84,
                    color=color,
                    offset=(series_index - 0.5) * 0.018,
                    label=key,
                )
        if "cool_iso_median" in stats:
            axis.plot(x, stats["cool_iso_median"], color="C1", label=r"$\dot M_{\rm cool,iso}$")
        title = r"$M_{\rm hot}$ normalization" if normalization == "mhot" else r"$M_\star$ normalization"
        axis.set_title(title, loc="left")
        axis.legend(loc="upper right")
        axis.set_ylabel(r"$\dot M/M_{\rm norm}\ ({\rm Gyr}^{-1})$")
        scatter_axis.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$")
        _style_scatter_axis(scatter_axis, r"$1\sigma$ scatter $({\rm Gyr}^{-1})$")
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.13, top=0.90, wspace=0.30, hspace=0.08)
    save_figure(figure, save_path)
    return figure
