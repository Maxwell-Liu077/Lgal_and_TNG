"""Snap99 sample selection and deterministic final-bin sampling."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Callable

import numpy as np

from ..utils.config import Phase21Config
from ..utils.arrays import json_ready
from .mpb import load_mpb


def load_z99_candidates(base_path: str, config: Phase21Config) -> list[dict]:
    """Read valid snap99 central candidates from the TNG group catalogue."""

    try:
        import illustris_python as il
    except ImportError as exc:
        raise ImportError("illustris_python is required for TNG selection") from exc
    halos = il.groupcat.loadHalos(base_path, config.snap_selection, fields=["GroupFirstSub", "Group_M_Crit200", "Group_R_Crit200"])
    subhalos = il.groupcat.loadSubhalos(base_path, config.snap_selection, fields=["SubhaloFlag", "SubhaloMassInRadType"])
    if not isinstance(halos, dict) or not isinstance(subhalos, dict):
        raise ValueError("TNG catalogue loaders must return dictionaries")
    first = np.asarray(halos["GroupFirstSub"], dtype=np.int64)
    flags = np.asarray(subhalos["SubhaloFlag"], dtype=bool)
    mstar_type = np.asarray(subhalos["SubhaloMassInRadType"], dtype=float)
    if mstar_type.ndim != 2 or mstar_type.shape[1] <= 4:
        raise ValueError("SubhaloMassInRadType must contain the stellar component")
    m200 = np.asarray(halos["Group_M_Crit200"], dtype=float) * 1.0e10 / config.h
    r200 = np.asarray(halos["Group_R_Crit200"], dtype=float)
    if not (len(first) == len(m200) == len(r200)):
        raise ValueError("Halo catalogue fields have inconsistent lengths")
    valid_first = (first >= 0) & (first < len(flags)) & (first < mstar_type.shape[0])
    mstar = np.full(len(first), np.nan)
    mstar[valid_first] = mstar_type[first[valid_first], 4] * 1.0e10 / config.h
    flagged = np.zeros(len(first), dtype=bool)
    if len(flags):
        flagged[valid_first] = flags[first[valid_first]]
    valid = valid_first & flagged & (mstar > 0) & (m200 > 0) & (r200 > 0)
    log_mstar = np.full(len(first), np.nan)
    log_mstar[valid] = np.log10(mstar[valid])
    boundary_tolerance = 1.0e-10
    log_mstar[np.isclose(log_mstar, config.log_mstar_min, rtol=0, atol=boundary_tolerance)] = config.log_mstar_min
    log_mstar[np.isclose(log_mstar, config.log_mstar_max, rtol=0, atol=boundary_tolerance)] = config.log_mstar_max
    valid &= (log_mstar >= config.log_mstar_min) & (log_mstar <= config.log_mstar_max)
    edges = config.mass_bin_edges
    bins = np.searchsorted(edges, log_mstar, side="right") - 1
    bins[log_mstar == edges[-1]] = len(edges) - 2
    valid &= (bins >= 0) & (bins < len(edges) - 1)
    candidates = []
    for group_id in np.where(valid)[0]:
        subfind_id = int(first[group_id])
        candidates.append({
            "group_id_z99": int(group_id),
            "subfind_id_z99": subfind_id,
            "log_mstar_z99": float(log_mstar[group_id]),
            "mstar_z99_msun": float(mstar[group_id]),
            "m200c_z99_msun": float(m200[group_id]),
            "r200c_z99_ckpc_h": float(r200[group_id]),
            "mass_bin_index": int(bins[group_id]),
        })
    return candidates


def _select_candidates(candidates: list[dict], config: Phase21Config) -> tuple[list[dict], dict]:
    """Select up to 50 candidates per final bin after deterministic sorting."""

    selected: list[dict] = []
    candidate_counts = []
    selected_counts = []
    edges = config.mass_bin_edges
    for bin_index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        group = sorted(
            (candidate for candidate in candidates if candidate["mass_bin_index"] == bin_index),
            key=lambda item: item["subfind_id_z99"],
        )
        candidate_counts.append(len(group))
        # Empty bins are a valid outcome of a catalogue query and do not
        # need the "few candidates" warning; warn only for a non-empty bin
        # that is smaller than the configured diagnostic threshold.
        if 0 < len(group) < config.minimum_bin_warning_count:
            warnings.warn(f"Mass bin [{low:.2f}, {high:.2f}) has {len(group)} candidates", RuntimeWarning)
        rng = np.random.default_rng(np.random.SeedSequence([config.random_seed, bin_index]))
        order = rng.permutation(len(group))
        take = order[: min(config.sample_per_bin, len(group))]
        chosen = [dict(group[int(position)]) for position in take]
        for rank, record in enumerate(chosen):
            record.update({
                "sample_index": len(selected),
                "sample_rank_in_bin": rank,
                "mass_bin_low": float(low),
                "mass_bin_high": float(high),
                "mass_bin_center": float(0.5 * (low + high)),
            })
            selected.append(record)
        selected_counts.append(len(chosen))
    return selected, {
        "fingerprint": config.fingerprint,
        "fingerprint_payload": config.fingerprint_payload,
        "random_seed": config.random_seed,
        "mass_bin_edges": edges.tolist(),
        "final_bin_policy": "merge [11.00,11.25) and [11.25,11.50] before sampling as [11.00,11.50]",
        "candidate_counts": candidate_counts,
        "selected_counts": selected_counts,
        "candidate_sort": "SubfindID ascending before per-bin SeedSequence permutation",
        "selection_axis": "snap99 SubhaloMassInRadType[:,4] within twice the stellar half-mass radius",
    }


def build_sample(
    base_path: str,
    config: Phase21Config,
    *,
    mpb_loader: Callable[[str, int, Phase21Config], dict] = load_mpb,
    verbose: bool = True,
) -> tuple[list[dict], dict]:
    """Build the selected sample and attach a valid snap90--99 MPB."""

    candidates = load_z99_candidates(base_path, config)
    accepted: list[dict] = []
    rejected: list[dict] = []
    candidate_counts = []
    selected_counts = []
    edges = config.mass_bin_edges
    for bin_index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        group = sorted((candidate for candidate in candidates if candidate["mass_bin_index"] == bin_index), key=lambda item: item["subfind_id_z99"])
        candidate_counts.append(len(group))
        rng = np.random.default_rng(np.random.SeedSequence([config.random_seed, bin_index]))
        accepted_in_bin = 0
        for position in rng.permutation(len(group)):
            if accepted_in_bin >= config.sample_per_bin:
                break
            candidate = group[int(position)]
            try:
                record = dict(candidate)
                record["mpb"] = mpb_loader(base_path, int(candidate["subfind_id_z99"]), config)
                record.update({
                    "sample_index": len(accepted),
                    "sample_rank_in_bin": accepted_in_bin,
                    "mass_bin_low": float(low),
                    "mass_bin_high": float(high),
                    "mass_bin_center": float(0.5 * (low + high)),
                })
                accepted.append(record)
                accepted_in_bin += 1
            except Exception as exc:
                rejected.append({"subfind_id_z99": candidate["subfind_id_z99"], "reason": f"{type(exc).__name__}: {exc}"})
            if verbose:
                print(f"[sample] bin={bin_index} {accepted_in_bin}/{min(config.sample_per_bin, len(group))} sub={candidate['subfind_id_z99']}")
        selected_counts.append(accepted_in_bin)
    metadata = {
        "fingerprint": config.fingerprint,
        "fingerprint_payload": config.fingerprint_payload,
        "random_seed": config.random_seed,
        "mass_bin_edges": edges.tolist(),
        "final_bin_policy": "merge [11.00,11.25) and [11.25,11.50] before sampling as [11.00,11.50]",
        "candidate_counts": candidate_counts,
        "selected_counts": selected_counts,
        "candidate_sort": "SubfindID ascending before per-bin SeedSequence permutation",
        "selection_axis": "snap99 SubhaloMassInRadType[:,4] within twice the stellar half-mass radius",
    }
    metadata.update({"base_path": str(base_path), "rejected_mpb": rejected, "selected_halo_count": len(accepted)})
    return accepted, metadata


def _sample_paths(cache_dir: str | Path, config: Phase21Config | None = None) -> tuple[Path, Path]:
    """Return the JSON sample and metadata paths."""

    directory = Path(cache_dir) / "samples"
    seed = int(config.random_seed) if config is not None else 202608
    tag = f"phase21_sample_snap99_seed{seed}"
    return directory / f"{tag}.json", directory / f"{tag}_metadata.json"


def save_sample(sample: list[dict], metadata: dict, cache_dir: str | Path, config: Phase21Config | None = None) -> tuple[Path, Path]:
    """Atomically write the nested sample/MPB cache."""

    sample_path, metadata_path = _sample_paths(cache_dir, config)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    for path, payload in ((sample_path, sample), (metadata_path, metadata)):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
    return sample_path, metadata_path


def load_sample(cache_dir: str | Path, config: Phase21Config | None = None) -> tuple[list[dict], dict]:
    """Load a nested sample cache."""

    sample_path, metadata_path = _sample_paths(cache_dir, config)
    return json.loads(sample_path.read_text(encoding="utf-8")), json.loads(metadata_path.read_text(encoding="utf-8"))


def load_or_build_sample(config: Phase21Config, *, cache_dir: str | Path, rebuild: bool = False, verbose: bool = True) -> tuple[list[dict], dict]:
    """Load the sample cache or build it from the external TNG catalogues."""

    paths = _sample_paths(cache_dir, config)
    if not rebuild and all(path.is_file() for path in paths):
        try:
            sample, metadata = load_sample(cache_dir, config)
            if metadata.get("fingerprint") == config.fingerprint:
                return sample, metadata
        except Exception:
            pass
    sample, metadata = build_sample(config.base_path, config, verbose=verbose)
    save_sample(sample, metadata, cache_dir, config)
    return sample, metadata
