"""Stellar-mass scaling relations for both Phase 1.8 result variants.

The plotting functions detect the available processed fields and select the
corresponding historical series.  Baseline results therefore retain the five
original curves, while inward-delivery results retain their three-curve view.
No scientific quantity is recomputed in this presentation layer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .style import ACADEMIC_STYLE, save_figure


BASELINE_SERIES = (
    ("sam", "L-GALAXIES", "#000000"),
    ("sam_sn", "L-GALAXIES (SN)", "#E69F00"),
    ("hot_halo_cooling", "Hot to Cold", "#0072B2"),
    ("outer_cold_inflow", "Cold to Cold", "#009E73"),
    ("cooling_plus_inflow", "Hot + Cold", "#CC79A7"),
)

INWARD_DELIVERY_SERIES = (
    ("sam", "L-GALAXIES", "#000000"),
    ("cooling_plus_inflow", "Hot + Cold", "#0072B2"),
    (
        "tng_total_inward_delivery",
        "TNG total inward delivery",
        "#D55E00",
    ),
)

BASELINE_NON_SAM_SERIES = BASELINE_SERIES[2:]
INWARD_DELIVERY_NON_SAM_SERIES = INWARD_DELIVERY_SERIES[1:]

# Historical public constants remain available through ``src.plotting``.
SERIES = BASELINE_SERIES
NON_SAM_SERIES = BASELINE_NON_SAM_SERIES

NORMALIZATION_STYLES = (
    ("m200c", r"$M_{200c}$", "-"),
    ("mhot", r"$M_{\rm hot}$", "--"),
    ("mcgm", r"$M_{\rm CGM}$", ":"),
    ("mstar", r"$M_\star$", "-."),
)


def _contains_series(
    statistics: dict[str, np.ndarray],
    short_name: str,
) -> bool:
    """Return whether all percentile arrays for one series are available."""

    return all(
        f"{short_name}_{suffix}" in statistics
        for suffix in ("p16", "median", "p84")
    )


def _series_for_statistics(
    statistics: dict[str, np.ndarray],
    *,
    include_sam: bool,
):
    """Select the unchanged baseline or inward-delivery plot registry."""

    is_inward_delivery = _contains_series(
        statistics,
        "tng_total_inward_delivery",
    )
    if is_inward_delivery:
        return (
            INWARD_DELIVERY_SERIES
            if include_sam
            else INWARD_DELIVERY_NON_SAM_SERIES
        )
    return BASELINE_SERIES if include_sam else BASELINE_NON_SAM_SERIES


def _plot_selected_mass_dependence(
    statistics: dict[str, np.ndarray],
    *,
    mode: str,
    series,
    annotation_label: str | None = None,
    y_label: str = r"rate ($M_\odot\,{\rm yr}^{-1}$)",
    log_y: bool = True,
    save_path: str | Path | None = None,
):
    """Render selected median and percentile-band series.

    ``mode`` is retained in the signature for compatibility with historical
    callers.  Slow mode is the only scientifically supported pipeline mode,
    and the plotting result has never depended on this label.
    """

    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    plt.rcParams.update(ACADEMIC_STYLE)
    centers = np.asarray(statistics["mass_bin_center"], dtype=float)
    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    for short_name, label, color in series:
        p16 = np.asarray(statistics[f"{short_name}_p16"], dtype=float)
        median = np.asarray(
            statistics[f"{short_name}_median"],
            dtype=float,
        )
        p84 = np.asarray(statistics[f"{short_name}_p84"], dtype=float)
        line_valid = np.isfinite(median)
        band_valid = (
            np.isfinite(p16)
            & np.isfinite(p84)
            & np.isfinite(median)
        )
        if log_y:
            line_valid &= median > 0
            band_valid &= (p16 > 0) & (p84 > 0)
        if np.any(band_valid):
            ax.fill_between(
                centers,
                np.where(band_valid, p16, np.nan),
                np.where(band_valid, p84, np.nan),
                color=color,
                alpha=0.16,
                linewidth=0,
            )
        if np.any(line_valid):
            ax.plot(
                centers,
                np.where(line_valid, median, np.nan),
                color=color,
                linewidth=2.1,
                label=label,
            )

    if log_y:
        ax.set_yscale("log")
    mass_min = float(np.nanmin(statistics["mass_bin_low"]))
    mass_max = float(np.nanmax(statistics["mass_bin_high"]))
    ax.set_xlim(mass_min, mass_max)
    ax.set_xticks(
        np.arange(
            np.ceil(mass_min),
            np.floor(mass_max) + 1.0,
            1.0,
        )
    )
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))
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
    ax.set_xlabel(
        r"$\log_{10}(M_{\star,99}(<2R_{\star,1/2})/M_\odot)$"
    )
    ax.set_ylabel(y_label)
    if annotation_label is not None:
        ax.text(
            0.04,
            0.95,
            annotation_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=16,
        )
    ax.grid(False)
    ax.legend(ncol=2, frameon=False)
    save_figure(fig, save_path)
    return fig


def plot_mass_dependence(
    statistics: dict[str, np.ndarray],
    *,
    mode: str,
    log_y: bool = True,
    save_path: str | Path | None = None,
):
    """Plot the historical series appropriate to the supplied statistics."""

    return _plot_selected_mass_dependence(
        statistics,
        mode=mode,
        series=_series_for_statistics(statistics, include_sam=True),
        log_y=log_y,
        save_path=save_path,
    )


def plot_mass_dependence_without_sam(
    statistics: dict[str, np.ndarray],
    *,
    mode: str,
    log_y: bool = True,
    save_path: str | Path | None = None,
):
    """Plot the historical TNG-only series for the supplied result variant."""

    return _plot_selected_mass_dependence(
        statistics,
        mode=mode,
        series=_series_for_statistics(statistics, include_sam=False),
        log_y=log_y,
        save_path=save_path,
    )


def plot_normalized_mass_dependence(
    statistics: dict[str, np.ndarray],
    *,
    normalization: str,
    log_y: bool = True,
    save_path: str | Path | None = None,
):
    """Plot one per-halo mass-normalized result with percentile bands."""

    labels = {name: label for name, label, _ in NORMALIZATION_STYLES}
    if normalization not in labels:
        raise ValueError("Unknown normalization")
    return _plot_selected_mass_dependence(
        statistics,
        mode="slow",
        series=_series_for_statistics(statistics, include_sam=True),
        annotation_label=labels[normalization],
        y_label=r"$\dot{M}/M_{\rm norm}\ ({\rm Gyr}^{-1})$",
        log_y=log_y,
        save_path=save_path,
    )
