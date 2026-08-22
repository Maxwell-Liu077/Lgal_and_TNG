"""Stable on-disk serialization for Phase 1.8 results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..utils.arrays import _json_ready


def save_phase18_result(
    result: dict,
    output_prefix: str | Path,
) -> dict[str, Path]:
    """Save metadata, per-halo values, and all binned statistics."""

    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    halo_path = prefix.with_name(prefix.name + "_halos").with_suffix(".npz")
    csv_path = prefix.with_name(prefix.name + "_halos").with_suffix(".csv")
    stats_path = prefix.with_name(prefix.name + "_binned").with_suffix(".npz")
    history_path = prefix.with_name(
        prefix.name + "_tracer_history"
    ).with_suffix(".npz")
    normalized_paths = {
        normalization: prefix.with_name(
            prefix.name + f"_normalized_{normalization}"
        ).with_suffix(".npz")
        for normalization in result["normalized_statistics"]
    }

    json_temporary = json_path.with_suffix(".tmp.json")
    json_temporary.write_text(
        json.dumps(
            _json_ready(
                {
                    "metadata": result["metadata"],
                    "binned_statistics": result["binned_statistics"],
                    "normalized_statistics": result[
                        "normalized_statistics"
                    ],
                    "history_statistics": result["history_statistics"],
                }
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    json_temporary.replace(json_path)
    halo_temporary = halo_path.with_suffix(".tmp.npz")
    np.savez(halo_temporary, **result["halo_results"])
    halo_temporary.replace(halo_path)
    stats_temporary = stats_path.with_suffix(".tmp.npz")
    np.savez(stats_temporary, **result["binned_statistics"])
    stats_temporary.replace(stats_path)
    history_temporary = history_path.with_suffix(".tmp.npz")
    np.savez(history_temporary, **result["history_statistics"])
    history_temporary.replace(history_path)
    for normalization, path in normalized_paths.items():
        temporary = path.with_suffix(".tmp.npz")
        np.savez(
            temporary,
            **result["normalized_statistics"][normalization],
        )
        temporary.replace(path)

    fields = list(result["halo_results"])
    csv_temporary = csv_path.with_suffix(".tmp.csv")
    with csv_temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        n_rows = len(result["halo_results"][fields[0]]) if fields else 0
        for index in range(n_rows):
            writer.writerow(
                [
                    _json_ready(result["halo_results"][field][index])
                    for field in fields
                ]
            )
    csv_temporary.replace(csv_path)
    return {
        "json": json_path,
        "halos_npz": halo_path,
        "halos_csv": csv_path,
        "binned_npz": stats_path,
        "tracer_history_npz": history_path,
        **{
            f"normalized_{normalization}_npz": path
            for normalization, path in normalized_paths.items()
        },
    }

