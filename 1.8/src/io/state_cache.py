"""Per-halo particle-state cache for the Phase 1.8 workflow."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from pathlib import Path

import numpy as np

from ..utils.config import Phase18Config
from .region_state import load_halo_pair_state
from ..utils.arrays import _row


def _state_fingerprint(
    sample_row: dict,
    config: Phase18Config,
) -> str:
    """Return the unchanged v4 state-cache fingerprint."""

    payload = {
        "version": 4,
        "base_path": str(Path(config.base_path)),
        "subhalo_id_z0": int(sample_row["subhalo_id_z0"]),
        "group_id_prev": int(sample_row["group_id_prev"]),
        "group_id_cur": int(sample_row["group_id_cur"]),
        "snap_prev": config.snap_prev,
        "snap_cur": config.snap_cur,
        "central_rfrac": config.central_rfrac,
        "outer_rmax_rfrac": config.outer_rmax_rfrac,
        "hot_logt": config.hot_log10_temperature_min,
        "cold_logt": config.cold_log10_temperature_max,
        "satellite_gas_exclusion": "Subfind-bound ParticleIDs",
        "history_membership": "central/satellite gas and PartType4 IDs",
        "hot_tracer_source_min_rfrac": config.central_rfrac,
        "h": config.h,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _state_cache_path(
    cache_dir: Path,
    sample_row: dict,
    config: Phase18Config,
) -> Path:
    """Build the unchanged per-halo cache path from the v4 fingerprint."""

    fingerprint = _state_fingerprint(sample_row, config)
    return (
        cache_dir
        / "halo_states"
        / (
            f"sub{int(sample_row['subhalo_id_z0'])}"
            f"_{fingerprint[:16]}.npz"
        )
    )


def _save_state(path: Path, state: dict, fingerprint: str) -> None:
    """Atomically flatten and save source/current/target state branches."""

    path.parent.mkdir(parents=True, exist_ok=True)
    values = {"fingerprint": np.asarray(fingerprint)}
    for branch in ("source", "current", "target"):
        for key, value in state[branch].items():
            values[f"{branch}__{key}"] = np.asarray(value)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **values)
    temporary.replace(path)


def _load_state(path: Path, fingerprint: str) -> dict | None:
    """Reconstruct a cached state only when its fingerprint matches."""

    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(np.asarray(data["fingerprint"]).item()) != fingerprint:
                return None
            state = {"source": {}, "current": {}, "target": {}}
            for key in data.files:
                if key == "fingerprint":
                    continue
                branch, field = key.split("__", 1)
                value = np.asarray(data[key])
                state[branch][field] = (
                    value.item() if value.ndim == 0 else value
                )
        return state
    except Exception:
        return None


def _load_state_worker(args):
    """Process-safe top-level worker for one uncached halo state."""

    (
        base_path,
        sample_row,
        header_prev,
        header_cur,
        config,
    ) = args
    return load_halo_pair_state(
        base_path,
        sample_row,
        header_prev=header_prev,
        header_cur=header_cur,
        config=config,
    )


def prepare_halo_states(
    sample_table: dict[str, np.ndarray],
    *,
    config: Phase18Config,
    header_prev: dict,
    header_cur: dict,
    cache_dir: str | Path,
    verbose: bool = True,
) -> tuple[dict[str, np.ndarray], list[dict], list[dict]]:
    """Load/cache halo states, optionally parallel across halos."""

    cache_dir = Path(cache_dir)
    states: list[dict | None] = [None] * len(sample_table["sample_index"])
    rejected: list[dict] = []
    pending = []
    for index in range(len(states)):
        sample_row = _row(sample_table, index)
        fingerprint = _state_fingerprint(sample_row, config)
        path = _state_cache_path(cache_dir, sample_row, config)
        state = _load_state(path, fingerprint)
        if state is not None:
            states[index] = state
            if verbose:
                print(
                    f"[state] cached {index + 1}/{len(states)} "
                    f"sub={int(sample_row['subhalo_id_z0'])}"
                )
        else:
            pending.append((index, sample_row, path, fingerprint))

    if verbose:
        print(
            f"[state] cache check complete: "
            f"cached={len(states) - len(pending)}, "
            f"pending={len(pending)}, total={len(states)}",
            flush=True,
        )
    serial = (
        config.state_parallel_backend == "serial"
        or config.state_workers == 1
    )
    if serial:
        for sequence, (index, sample_row, path, fingerprint) in enumerate(
            pending,
            start=1,
        ):
            try:
                state = load_halo_pair_state(
                    config.base_path,
                    sample_row,
                    header_prev=header_prev,
                    header_cur=header_cur,
                    config=config,
                )
                states[index] = state
                _save_state(path, state, fingerprint)
            except Exception as exc:
                rejected.append(
                    {
                        "subhalo_id_z0": int(
                            sample_row["subhalo_id_z0"]
                        ),
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
            if verbose:
                print(
                    f"[state] computed {sequence}/{len(pending)}"
                )
    elif pending:
        executor_class = (
            ThreadPoolExecutor
            if config.state_parallel_backend == "thread"
            else ProcessPoolExecutor
        )
        with executor_class(max_workers=config.state_workers) as executor:
            future_map = {}
            for index, sample_row, path, fingerprint in pending:
                args = (
                    config.base_path,
                    sample_row,
                    header_prev,
                    header_cur,
                    config,
                )
                future = executor.submit(_load_state_worker, args)
                future_map[future] = (
                    index,
                    sample_row,
                    path,
                    fingerprint,
                )
            completed = 0
            for future in as_completed(future_map):
                index, sample_row, path, fingerprint = future_map[future]
                try:
                    state = future.result()
                    states[index] = state
                    _save_state(path, state, fingerprint)
                except Exception as exc:
                    rejected.append(
                        {
                            "subhalo_id_z0": int(
                                sample_row["subhalo_id_z0"]
                            ),
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
                completed += 1
                if verbose:
                    print(
                        f"[state] completed {completed}/{len(pending)}"
                    )

    valid_indices = np.asarray(
        [index for index, state in enumerate(states) if state is not None],
        dtype=int,
    )
    valid_table = {
        key: np.asarray(values)[valid_indices]
        for key, values in sample_table.items()
    }
    valid_states = [states[index] for index in valid_indices]
    return valid_table, valid_states, rejected
