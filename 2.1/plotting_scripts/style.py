"""Shared plotting style and atomic figure output."""

from __future__ import annotations

from pathlib import Path

from cycler import cycler


REFERENCE_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")

ACADEMIC_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 12,
    "axes.labelsize": 15,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.titlepad": 10,
    "axes.linewidth": 1.25,
    "axes.prop_cycle": cycler(color=REFERENCE_COLORS),
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.grid": False,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 7,
    "ytick.major.size": 7,
    "xtick.minor.size": 3.5,
    "ytick.minor.size": 3.5,
    "xtick.major.width": 1.15,
    "ytick.major.width": 1.15,
    "xtick.minor.width": 0.9,
    "ytick.minor.width": 0.9,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "lines.linewidth": 2.5,
    "lines.marker": "None",
    "legend.fontsize": 12,
    "legend.frameon": False,
    "legend.handlelength": 2.4,
    "legend.handletextpad": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
}


def save_figure(figure, path: str | Path | None) -> None:
    """Save both publication formats when a path is supplied."""

    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix not in {".png", ".pdf"}:
        path = path.with_suffix(".png")
    figure.savefig(path, dpi=300, bbox_inches="tight")
    sibling = path.with_suffix(".pdf" if path.suffix.lower() == ".png" else ".png")
    figure.savefig(sibling, dpi=300 if sibling.suffix.lower() == ".png" else None, bbox_inches="tight")
