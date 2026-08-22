"""Atomic Phase 2.1 result serialization."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..utils.arrays import json_ready


def _atomic_json(path: Path, value) -> None:
    """Write JSON through a temporary sibling."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_ready(value), indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _save_npz(path: Path, values: dict[str, np.ndarray]) -> None:
    """Write an NPZ file atomically."""

    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **values)
    temporary.replace(path)


def _save_events(path: Path, ledgers: list[list[dict]]) -> None:
    """Write variable-length event ledgers to HDF5 without losing uint64 IDs."""

    try:
        import h5py
    except ImportError:
        return
    temporary = path.with_suffix(".tmp.h5")
    string_type = h5py.string_dtype(encoding="utf-8")
    with h5py.File(temporary, "w") as handle:
        handle.attrs["schema"] = "phase21-event-ledger-v1"
        for index, ledger in enumerate(ledgers):
            group = handle.create_group(f"halo_{index:06d}")
            group.create_dataset("TracerID", data=np.asarray([item["TracerID"] for item in ledger], dtype=np.uint64))
            group.create_dataset("event", data=np.asarray([item["event"] for item in ledger], dtype=string_type))
            group.create_dataset("anchor_snapshot", data=np.asarray([item.get("anchor_snapshot", -1) for item in ledger], dtype=np.int16))
            group.create_dataset("rate_class", data=np.asarray([item["rate_class"] for item in ledger], dtype=string_type))
            group.create_dataset("other_reason", data=np.asarray([item.get("other_reason", "") for item in ledger], dtype=string_type))
            group.create_dataset("weight_msun", data=np.asarray([item["weight_msun"] for item in ledger], dtype=float))
            group.create_dataset(
                "anchor_state",
                data=np.asarray([json.dumps(json_ready(item.get("anchor_state", {})), ensure_ascii=False) for item in ledger], dtype=string_type),
            )
            group.create_dataset(
                "source_state",
                data=np.asarray([json.dumps(json_ready(item.get("source_state", {})), ensure_ascii=False) for item in ledger], dtype=string_type),
            )
            group.create_dataset(
                "target_state",
                data=np.asarray([json.dumps(json_ready(item.get("target_state", {})), ensure_ascii=False) for item in ledger], dtype=string_type),
            )
            group.create_dataset(
                "state_sequence",
                data=np.asarray([json.dumps(json_ready(item.get("state_sequence", {})), ensure_ascii=False) for item in ledger], dtype=string_type),
            )
            group.create_dataset(
                "missing_mask",
                data=np.asarray([json.dumps(json_ready(item.get("missing_mask", {})), ensure_ascii=False) for item in ledger], dtype=string_type),
            )
    temporary.replace(path)


def save_phase21_result(result: dict, output_prefix: str | Path) -> dict[str, Path]:
    """Save metadata, per-halo products, statistics, events, and CSV."""

    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    json_path = prefix.with_suffix(".json")
    _atomic_json(
        json_path,
        {
            key: result[key]
            for key in (
                "metadata", "sample", "binned_statistics", "normalized_statistics",
                "agn_quartile_statistics", "composition_statistics", "closure_diagnostics",
            )
            if key in result
        },
    )
    paths["json"] = json_path
    for key, suffix in (("metadata", "metadata"), ("sample", "sample"), ("closure_diagnostics", "closure_diagnostics")):
        if key in result:
            path = prefix.with_name(prefix.name + f"_{suffix}").with_suffix(".json")
            _atomic_json(path, result[key])
            paths[f"{suffix}_json"] = path
    halo_path = prefix.with_name(prefix.name + "_halos").with_suffix(".npz")
    _save_npz(halo_path, result["halo_results"])
    paths["halos_npz"] = halo_path
    csv_path = prefix.with_name(prefix.name + "_halos").with_suffix(".csv")
    fields = list(result["halo_results"])
    temporary = csv_path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        n_rows = len(result["halo_results"][fields[0]]) if fields else 0
        for index in range(n_rows):
            writer.writerow([json_ready(result["halo_results"][field][index]) for field in fields])
    temporary.replace(csv_path)
    paths["halos_csv"] = csv_path
    for key in ("binned_statistics", "agn_quartile_statistics", "composition_statistics"):
        path = prefix.with_name(prefix.name + f"_{key}").with_suffix(".npz")
        _save_npz(path, result[key])
        paths[key] = path
    for normalization, values in result["normalized_statistics"].items():
        path = prefix.with_name(prefix.name + f"_normalized_{normalization}").with_suffix(".npz")
        _save_npz(path, values)
        paths[f"normalized_{normalization}"] = path
    event_path = prefix.with_name(prefix.name + "_events").with_suffix(".h5")
    _save_events(event_path, result.get("event_ledgers", []))
    if event_path.is_file():
        paths["events_h5"] = event_path
    return paths
