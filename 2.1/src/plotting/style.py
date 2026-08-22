"""Shared plotting style and atomic figure output."""

from __future__ import annotations

from pathlib import Path


ACADEMIC_STYLE = {
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
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
    figure.savefig(path, dpi=220, bbox_inches="tight")
    sibling = path.with_suffix(".pdf" if path.suffix.lower() == ".png" else ".png")
    figure.savefig(sibling, dpi=220 if sibling.suffix.lower() == ".png" else None, bbox_inches="tight")
