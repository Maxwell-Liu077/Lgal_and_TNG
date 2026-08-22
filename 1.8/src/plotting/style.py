"""Shared publication style and figure serialization helpers.

This module contains presentation settings only.  It never imports raw-data
readers or analysis routines, which keeps plotting reproducible from processed
arrays alone.
"""

from __future__ import annotations

from pathlib import Path


ACADEMIC_STYLE = {
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 13,
    "axes.labelsize": 15,
    "legend.fontsize": 12,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.linewidth": 1.1,
}


def save_figure(fig, save_path: str | Path | None) -> None:
    """Save identical figure content as high-resolution PNG and vector PDF."""

    if save_path is None:
        return
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


# Preserve the historical private helper name for existing notebook cells.
_save = save_figure
