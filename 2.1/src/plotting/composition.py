"""100-percent stacked composition charts."""

from __future__ import annotations

import numpy as np

from .scaling import _plt
from .style import save_figure


def _stack(axis, x, fractions, names):
    """Draw one normalized stack and return the bottom coordinate."""

    bottom = np.zeros(len(x), dtype=float)
    for values, name in zip(fractions, names):
        axis.bar(x, values, bottom=bottom, width=0.20, label=str(name), edgecolor="none")
        bottom += np.nan_to_num(values)
    return bottom


def _decorate(axis, x, statistics, prefix, *, legend_below=False):
    """Apply labels and event-count annotations to one composition axis."""

    title = "Central cold-gas supply" if prefix == "in" else "Central cold-gas removal"
    axis.set_title(title, loc="left")
    axis.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$")
    axis.set_ylim(0, 1.12)
    axis.set_yticks(np.linspace(0, 1, 6))
    if legend_below:
        columns = 2 if prefix == "in" else 3
        axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.20),
            ncol=columns,
            fontsize=10.5,
            columnspacing=1.1,
        )
    else:
        axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=11)


def plot_composition(statistics: dict[str, np.ndarray], *, direction: str | None = None, save_path=None):
    """Plot one or both README 100-percent composition products."""

    plt = _plt()
    x = np.asarray(statistics["mass_bin_center"])
    if direction is not None and direction not in {"in", "out"}:
        raise ValueError("direction must be 'in', 'out', or None")
    prefixes = (direction,) if direction is not None else ("in", "out")
    figure, axes = plt.subplots(1, len(prefixes), figsize=(8.4 if len(prefixes) == 1 else 13.0, 5.1), sharey=True, squeeze=False)
    for axis, prefix in zip(axes[0], prefixes):
        _stack(axis, x, statistics[f"{prefix}_component_fractions"], statistics[f"{prefix}_component_names"])
        _decorate(axis, x, statistics, prefix, legend_below=len(prefixes) > 1)
    axes[0, 0].set_ylabel("Mass / rate fraction")
    figure.subplots_adjust(
        left=0.11 if len(prefixes) == 1 else 0.08,
        right=0.72 if len(prefixes) == 1 else 0.98,
        bottom=0.15 if len(prefixes) == 1 else 0.27,
        top=0.88,
        wspace=0.28,
    )
    save_figure(figure, save_path)
    return figure
