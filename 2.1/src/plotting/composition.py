"""100-percent stacked composition charts."""

from __future__ import annotations

import numpy as np

from .scaling import _plt
from .style import save_figure


def _stack(axis, x, fractions, names):
    """Draw one normalized stack and return the bottom coordinate."""

    bottom = np.zeros(len(x), dtype=float)
    for values, name in zip(fractions, names):
        axis.bar(x, values, bottom=bottom, width=0.19, label=str(name))
        bottom += np.nan_to_num(values)
    return bottom


def plot_composition(statistics: dict[str, np.ndarray], *, save_path=None):
    """Plot entry and exit 100-percent stacks side by side."""

    plt = _plt()
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=True)
    x = np.asarray(statistics["mass_bin_center"])
    _stack(axes[0], x, statistics["in_component_fractions"], statistics["in_component_names"])
    _stack(axes[1], x, statistics["out_component_fractions"], statistics["out_component_names"])
    axes[0].set_title("Central cold-gas supply")
    axes[1].set_title("Central cold-gas removal")
    for axis in axes:
        axis.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$")
        axis.set_ylim(0, 1)
        axis.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Mass fraction")
    save_figure(figure, save_path)
    return figure
