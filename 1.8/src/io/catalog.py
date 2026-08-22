"""Focused group-catalog field readers used by diagnostic analyses."""

from __future__ import annotations

import numpy as np


def load_selected_subhalo_bh_mass(
    base_path: str,
    snap_num: int,
    subfind_ids: np.ndarray,
    *,
    h: float,
) -> np.ndarray:
    """Load black-hole masses for selected subhalos in physical solar masses."""

    try:
        import illustris_python as il
    except ImportError as exc:
        raise ImportError(
            "illustris_python is required to read SubhaloBHMass"
        ) from exc
    values = il.groupcat.loadSubhalos(
        base_path,
        int(snap_num),
        fields=["SubhaloBHMass"],
    )
    if isinstance(values, dict):
        values = values["SubhaloBHMass"]
    values = np.asarray(values, dtype=float)
    subfind_ids = np.asarray(subfind_ids, dtype=np.int64)
    if np.any(subfind_ids < 0) or np.any(subfind_ids >= len(values)):
        raise IndexError(f"Invalid SubfindID at snapshot {snap_num}")
    return values[subfind_ids] * 1.0e10 / h
