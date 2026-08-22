"""Small data-shaping helpers shared by the Phase 1.8 workflow.

These functions deliberately preserve dictionary insertion order because the
per-halo CSV header follows the order in which result fields are assembled.
"""

from __future__ import annotations

import numpy as np


def _row(table: dict[str, np.ndarray], index: int) -> dict:
    """Return one row from a dictionary-of-arrays table."""

    return {
        key: np.asarray(values)[index]
        for key, values in table.items()
    }


def _records_to_arrays(records: list[dict]) -> dict[str, np.ndarray]:
    """Convert ordered, schema-identical records to arrays."""

    if not records:
        return {}
    keys = tuple(records[0])
    if any(tuple(record) != keys for record in records[1:]):
        raise ValueError("Result records have inconsistent fields")
    return {
        key: np.asarray([record[key] for record in records])
        for key in keys
    }


def _json_ready(value):
    """Convert NumPy containers/scalars to JSON-safe Python values."""

    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


