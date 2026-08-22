"""Comparison plots for reheating and inward-delivery components."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .scaling_relation import (
    NORMALIZATION_STYLES,
    _contains_series,
    _plot_selected_mass_dependence,
)
from .style import ACADEMIC_STYLE, save_figure


REHEATING_SERIES = (
    (
        "sam_reheating_mass",
        "L-GALAXIES reheating",
        "#D55E00",
    ),
    (
        "tng_cold_to_hot_mass",
        "TNG cold-to-hot transition",
        "#0072B2",
    ),
)

INNER_HOT_DELIVERY_SERIES = (
    (
        "outer_hot_to_inner_hot",
        "Outer hot to inner hot",
        "#0072B2",
    ),
    (
        "outer_cold_to_inner_hot",
        "Outer cold to inner hot",
        "#009E73",
    ),
    (
        "inner_hot_delivery",
        "Inner-hot delivery",
        "#D55E00",
    ),
)

SUPPLY_COMPARISON_SERIES = (
    ("lgal", "L-GALAXIES cooling", "#000000", "-"),
    ("retained", "TNG retained cold supply", "#0072B2", "-"),
    (
        "with_cold_to_hot",
        r"TNG supply incl. cold$\rightarrow$hot",
        "#CC79A7",
        "--",
    ),
    (
        "with_all_outflow",
        r"TNG supply incl. cold$\rightarrow$hot/cold",
        "#D55E00",
        "-.",
    ),
)


def plot_reheating_comparison(
    statistics: dict[str, np.ndarray],
    *,
    log_y: bool = True,
    save_path: str | Path | None = None,
):
    """Plot every historical reheating series present in the result."""

    available = tuple(
        series
        for series in REHEATING_SERIES
        if _contains_series(statistics, series[0])
    )
    return _plot_selected_mass_dependence(
        statistics,
        mode="slow",
        series=available,
        y_label=r"reheated mass ($M_\odot$)",
        log_y=log_y,
        save_path=save_path,
    )


def plot_inner_hot_delivery_components(
    statistics: dict[str, np.ndarray],
    *,
    log_y: bool = True,
    save_path: str | Path | None = None,
):
    """Plot the two outer-to-inner-hot channels and their per-halo sum."""

    return _plot_selected_mass_dependence(
        statistics,
        mode="slow",
        series=INNER_HOT_DELIVERY_SERIES,
        y_label=r"rate ($M_\odot\,{\rm yr}^{-1}$)",
        log_y=log_y,
        save_path=save_path,
    )


def plot_supply_comparison(
    curve_summary: dict[str, np.ndarray],
    base_statistics: dict[str, np.ndarray],
    *,
    normalization: str,
    log_y: bool = True,
    save_path: str | Path | None = None,
):
    """Plot four paired-bootstrap supply curves for one normalization."""

    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    normalization_labels = {
        name: label for name, label, _ in NORMALIZATION_STYLES
    }
    if normalization not in normalization_labels:
        raise ValueError("Unknown normalization")
    centers = np.asarray(base_statistics["mass_bin_center"], dtype=float)
    mass_low = np.asarray(base_statistics["mass_bin_low"], dtype=float)
    mass_high = np.asarray(base_statistics["mass_bin_high"], dtype=float)
    plt.rcParams.update(ACADEMIC_STYLE)
    fig, ax = plt.subplots(figsize=(9.0, 6.0), constrained_layout=True)
    for series_index, (_, label, color, linestyle) in enumerate(
        SUPPLY_COMPARISON_SERIES
    ):
        median = curve_summary["median"][series_index]
        lower = curve_summary["ci16"][series_index]
        upper = curve_summary["ci84"][series_index]
        line_valid = np.isfinite(median)
        band_valid = line_valid & np.isfinite(lower) & np.isfinite(upper)
        if log_y:
            line_valid &= median > 0
            band_valid &= (lower > 0) & (upper > 0)
        ax.fill_between(
            centers,
            np.where(band_valid, lower, np.nan),
            np.where(band_valid, upper, np.nan),
            color=color,
            alpha=0.10,
            linewidth=0,
        )
        ax.plot(
            centers,
            np.where(line_valid, median, np.nan),
            color=color,
            linestyle=linestyle,
            linewidth=2.1,
            label=label,
        )
    if log_y:
        ax.set_yscale("log")
    ax.set_ylabel(r"$\dot{M}/M_{\rm norm}\ ({\rm Gyr}^{-1})$")
    ax.set_xlabel(
        r"$\log_{10}(M_{\star,99}(<2R_{\star,1/2})/M_\odot)$"
    )
    ax.text(
        0.035,
        0.95,
        normalization_labels[normalization],
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=16,
    )
    ax.legend(loc="upper right", ncol=2, frameon=False)
    mass_min = float(np.nanmin(mass_low))
    mass_max = float(np.nanmax(mass_high))
    ax.set_xlim(mass_min, mass_max)
    ax.set_xticks(
        np.arange(np.ceil(mass_min), np.floor(mass_max) + 1.0, 1.0)
    )
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax.grid(False)
    ax.tick_params(
        which="major",
        direction="in",
        top=True,
        right=True,
        length=6,
        width=1.0,
    )
    ax.tick_params(
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=3,
        width=0.8,
    )
    save_figure(fig, save_path)
    return fig
