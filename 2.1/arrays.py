"""Conversion helpers that keep NumPy result arrays serializable."""

from __future__ import annotations

import json

import numpy as np


def row(table: dict[str, np.ndarray], index: int) -> dict:
    """Return one row from a dictionary-of-arrays table."""

    return {
        key: (value[index].item() if np.asarray(value[index]).ndim == 0 else value[index])
        for key, value in table.items()
    }


def records_to_arrays(records: list[dict]) -> dict[str, np.ndarray]:
    """Convert schema-identical records into a dictionary of arrays."""

    if not records:
        return {}
    keys = tuple(records[0])
    if any(tuple(record) != keys for record in records[1:]):
        raise ValueError("Records have inconsistent field order")
    return {key: np.asarray([record[key] for record in records]) for key in keys}


def json_ready(value):
    """Recursively convert NumPy values to JSON-compatible values."""

    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None if np.isnan(value) else ("-inf" if value < 0 else "inf")
    return value


def stable_json(value) -> str:
    """Serialize a value deterministically for cache fingerprints."""

    return json.dumps(json_ready(value), sort_keys=True, separators=(",", ":"))
