"""Heatmaps for complete tracer origins and fates.

The default annotation settings reproduce the baseline Phase 1.8 figure.
The optional contrast-aware settings reproduce the later inward-delivery
figure without requiring a second plotting implementation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .style import ACADEMIC_STYLE, save_figure


HISTORY_LABELS = {
    "origin": {
        "central_cold": "Central cold",
        "outer_hot": "Outer hot",
        "central_hot": "Central hot",
        "outer_cold": "Outer cold",
        "wind": "Wind",
        "star": "Stars",
        "satellite": "Satellites",
        "outside_r200c": r"Outside $R_{200c}$",
        "black_hole": "Black holes",
        "other": "Other",
    },
    "fate": {
        "central_cold": "Central cold",
        "central_star": "Central stars",
        "halo_hot": "Halo hot",
        "outer_cold": "Outer cold",
        "wind": "Wind",
        "outside_r200c": r"Outside $R_{200c}$",
        "satellite": "Satellites",
        "black_hole": "Black holes",
        "other": "Other",
    },
}

HISTORY_BAR_COLORS = (
    "#2381F4C9",
    "#299729D5",
    "#EC7214D4",
    "#8C564B",
    "#9467BD",
)


def _automatic_text_color(
    value: float,
    *,
    color_map,
    color_norm,
) -> str:
    """Choose black or white using the larger WCAG contrast ratio."""

    red, green, blue, _ = color_map(color_norm(value))
    rgb = np.asarray([red, green, blue])
    linear_rgb = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )
    luminance = float(np.dot(linear_rgb, [0.2126, 0.7152, 0.0722]))
    black_contrast = (luminance + 0.05) / 0.05
    white_contrast = 1.05 / (luminance + 0.05)
    return "black" if black_contrast >= white_contrast else "white"


def plot_tracer_history_heatmaps(
    statistics: dict[str, np.ndarray],
    *,
    cell_fontsize: float = 7.0,
    cell_text_color: str = "threshold",
    cell_text_outline: bool = False,
    save_path: str | Path | None = None,
):
    """Plot complete origins and fates of the central-cold tracer pool.

    ``cell_text_color="threshold"`` retains the original Phase 1.8 color
    rule.  ``cell_text_color="auto"`` selects the higher-contrast text color
    for every cell and, together with the other optional arguments, reproduces
    the Phase 1.8.1 presentation.
    """

    import matplotlib.pyplot as plt
    from matplotlib import patheffects
    from matplotlib.colors import Normalize
    from matplotlib.ticker import MultipleLocator

    if cell_fontsize <= 0:
        raise ValueError("cell_fontsize must be positive")
    plt.rcParams.update(ACADEMIC_STYLE)
    origin_labels = {
        "central_cold": "Central cold",
        "outer_hot": "Outer hot",
        "central_hot": "Central hot",
        "outer_cold": "Outer cold",
        "wind": "Wind",
        "star": "Stars",
        "satellite": "Satellites",
        "outside_r200c": r"Outside $R_{200c}$",
        "black_hole": "Black holes",
        "other": "Other",
    }
    fate_labels = {
        "central_cold": "Central cold",
        "central_star": "Central stars",
        "halo_hot": "Halo hot",
        "outer_cold": "Outer cold",
        "wind": "Wind",
        "outside_r200c": r"Outside $R_{200c}$",
        "satellite": "Satellites",
        "black_hole": "Black holes",
        "other": "Other",
    }
    panels = []
    for prefix, label_map, annotation in (
        (
            "origin",
            origin_labels,
            "snap 98 origins of snap 99 central cold gas",
        ),
        (
            "fate",
            fate_labels,
            "snap 99 fates of snap 98 central cold gas",
        ),
    ):
        names = [
            str(value)
            for value in np.asarray(statistics[f"{prefix}_state_names"])
        ]
        fractions = np.asarray(
            statistics[f"{prefix}_fractions"],
            dtype=float,
        )
        panels.append(
            (
                names,
                fractions,
                [label_map[name] for name in names],
                annotation,
            )
        )

    finite_parts = [
        fractions[np.isfinite(fractions)]
        for _, fractions, _, _ in panels
        if np.any(np.isfinite(fractions))
    ]
    finite_values = (
        np.concatenate(finite_parts)
        if finite_parts
        else np.empty(0, dtype=float)
    )
    vmax = (
        min(1.0, max(0.05, float(np.nanmax(finite_values))))
        if len(finite_values)
        else 1.0
    )
    edges = np.concatenate(
        [
            np.asarray(statistics["mass_bin_low"], dtype=float),
            [float(np.asarray(statistics["mass_bin_high"])[-1])],
        ]
    )

    enhanced_layout = (
        cell_text_color == "auto"
        or cell_text_outline
        or cell_fontsize != 7.0
    )
    figure_size = (18.0, 7.2) if enhanced_layout else (16.0, 6.4)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=figure_size,
        constrained_layout=True,
    )
    color_map = plt.get_cmap("viridis")
    color_norm = Normalize(vmin=0.0, vmax=vmax, clip=True)
    image = None
    for ax, (_, fractions, labels, annotation) in zip(axes, panels):
        y_edges = np.arange(len(labels) + 1, dtype=float)
        image = ax.pcolormesh(
            edges,
            y_edges,
            fractions,
            cmap=color_map,
            norm=color_norm,
            shading="flat",
        )
        ax.set_ylim(len(labels), 0)
        ax.set_yticks(np.arange(len(labels)) + 0.5)
        ax.set_yticklabels(labels)
        ax.set_xlim(edges[0], edges[-1])
        ax.set_xticks(
            np.arange(
                np.ceil(edges[0]),
                np.floor(edges[-1]) + 1.0,
                1.0,
            )
        )
        ax.xaxis.set_minor_locator(MultipleLocator(0.25))
        ax.tick_params(
            which="major",
            direction="in",
            top=True,
            right=True,
            length=5,
        )
        ax.tick_params(
            which="minor",
            direction="in",
            top=True,
            bottom=True,
            length=2.5,
        )
        ax.set_xlabel(
            r"$\log_{10}(M_{\star,99}(<2R_{\star,1/2})/M_\odot)$"
        )
        ax.text(
            0.02,
            1.02,
            annotation,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=13,
        )
        for row in range(fractions.shape[0]):
            for column in range(fractions.shape[1]):
                value = fractions[row, column]
                if not np.isfinite(value) or value < 0.005:
                    continue
                x = 0.5 * (edges[column] + edges[column + 1])
                if cell_text_color == "auto":
                    color = _automatic_text_color(
                        value,
                        color_map=color_map,
                        color_norm=color_norm,
                    )
                elif cell_text_color == "threshold":
                    color = "white" if value > 0.55 * vmax else "black"
                else:
                    color = cell_text_color
                text = ax.text(
                    x,
                    row + 0.5,
                    f"{100.0 * value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=cell_fontsize,
                    fontweight=("semibold" if enhanced_layout else "normal"),
                    color=color,
                )
                if cell_text_outline:
                    outline_color = "white" if color == "black" else "black"
                    text.set_path_effects(
                        [
                            patheffects.withStroke(
                                linewidth=1.2,
                                foreground=outline_color,
                            ),
                            patheffects.Normal(),
                        ]
                    )
        ax.grid(False)
    colorbar = fig.colorbar(image, ax=axes, pad=0.02)
    colorbar.set_label("stacked tracer fraction")
    save_figure(fig, save_path)
    return fig


def plot_tracer_history_top4_stacked_bars(
    history: dict[str, np.ndarray],
    grouped_statistics: dict[str, dict],
    *,
    save_path: str | Path | None = None,
):
    """Plot global top-four origin and fate groups plus a complete remainder."""

    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter, MultipleLocator

    centers = np.asarray(history["mass_bin_center"], dtype=float)
    mass_low = np.asarray(history["mass_bin_low"], dtype=float)
    mass_high = np.asarray(history["mass_bin_high"], dtype=float)
    bar_width = 0.68 * float(np.nanmedian(mass_high - mass_low))
    panel_titles = {
        "origin": "Origins of snap 99 central cold gas",
        "fate": "Fates of snap 98 central cold gas",
    }
    plt.rcParams.update(ACADEMIC_STYLE)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.0, 5.5),
        sharey=True,
        constrained_layout=False,
    )
    for ax, prefix in zip(axes, ("origin", "fate")):
        statistics = grouped_statistics[prefix]
        fractions = np.asarray(statistics["fractions"], dtype=float)
        valid_bins = np.asarray(statistics["valid_bins"], dtype=bool)
        names = list(statistics["names"])
        labels = [
            (
                "Other"
                if name == "other_combined"
                else HISTORY_LABELS[prefix][name]
            )
            for name in names
        ]
        bottom = np.zeros(len(centers), dtype=float)
        for index, (label, color) in enumerate(
            zip(labels, HISTORY_BAR_COLORS)
        ):
            height = np.where(valid_bins, fractions[index], 0.0)
            ax.bar(
                centers,
                height,
                width=bar_width,
                bottom=bottom,
                color=color,
                edgecolor="white",
                linewidth=0.45,
                label=label,
            )
            bottom += height
        ax.set_xlim(float(mass_low[0]), float(mass_high[-1]))
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(
            np.arange(
                np.ceil(mass_low[0]),
                np.floor(mass_high[-1]) + 1.0,
                1.0,
            )
        )
        ax.xaxis.set_minor_locator(MultipleLocator(0.25))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.05))
        ax.set_xlabel(
            r"$\log_{10}(M_{\star,99}(<2R_{\star,1/2})/M_\odot)$"
        )
        ax.set_title(panel_titles[prefix], fontsize=16, pad=4)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=5,
            frameon=False,
            fontsize=9.5,
            handlelength=1.8,
            columnspacing=1.0,
        )
        ax.grid(False)
        ax.tick_params(
            which="major",
            direction="in",
            top=True,
            right=True,
            length=5,
        )
        ax.tick_params(
            which="minor",
            direction="in",
            top=True,
            bottom=True,
            length=2.5,
        )
    axes[0].set_ylabel("stacked tracer fraction")
    fig.subplots_adjust(
        left=0.075,
        right=0.99,
        bottom=0.27,
        top=0.91,
        wspace=0.08,
    )
    save_figure(fig, save_path)
    return fig
