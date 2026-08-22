"""Reusable grouped tracer-history statistics for presentation."""

from __future__ import annotations

import numpy as np


def top4_plus_other(
    history: dict[str, np.ndarray],
    halo_results: dict[str, np.ndarray],
    prefix: str,
) -> dict[str, np.ndarray | list[str]]:
    """Return complete stacked fractions using a fixed global top four."""

    names = [str(value) for value in history[f"{prefix}_state_names"]]
    fractions = np.asarray(history[f"{prefix}_fractions"], dtype=float)
    denominator = np.asarray(
        history[f"{prefix}_denominator_sum"],
        dtype=float,
    )
    physical_indices = [
        index for index, name in enumerate(names) if name != "other"
    ]
    totals = np.asarray(
        [
            np.nansum(
                np.asarray(halo_results[f"n_{prefix}_{name}"], dtype=float)
            )
            for name in names
        ],
        dtype=float,
    )
    ranked = sorted(
        physical_indices,
        key=lambda index: (-totals[index], index),
    )
    top_indices = ranked[:4]
    other_indices = [
        index for index in range(len(names)) if index not in top_indices
    ]
    grouped = np.vstack(
        [
            fractions[top_indices],
            np.nansum(fractions[other_indices], axis=0, keepdims=True),
        ]
    )
    valid_bins = denominator > 0
    grouped[:, ~valid_bins] = np.nan
    closure = np.nansum(grouped, axis=0)
    if not np.allclose(
        closure[valid_bins],
        1.0,
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise ValueError(f"{prefix} Top 4 + Other does not close to unity")
    return {
        "names": [names[index] for index in top_indices] + ["other_combined"],
        "fractions": grouped,
        "global_tracer_counts": totals[top_indices],
        "other_members": [names[index] for index in other_indices],
        "valid_bins": valid_bins,
        "closure": closure,
    }


def build_top4_history_statistics(
    history: dict[str, np.ndarray],
    halo_results: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray | list[str]]]:
    """Build grouped origin and fate statistics for the stacked-bar figure."""

    return {
        prefix: top4_plus_other(history, halo_results, prefix)
        for prefix in ("origin", "fate")
    }
