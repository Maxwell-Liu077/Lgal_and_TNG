"""Tabular tracer-history and transition summaries for notebook reporting.

The functions return plain dictionaries and row records.  Rendering with
Pandas or IPython remains an interactive concern and is intentionally kept out
of the reusable analysis layer.
"""

from __future__ import annotations

import numpy as np


def clean_name(value) -> str:
    """Convert stored byte or string state names to regular Python strings."""

    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _state_meanings(
    snap_prev: int,
    snap_cur: int,
) -> dict[str, dict[str, str]]:
    """Return the historical Chinese display labels for physical states."""

    return {
        "origin": {
            "central_cold": f"snap {snap_prev} 中心冷气体",
            "outer_hot": f"snap {snap_prev} 晕内外部热气体",
            "central_hot": f"snap {snap_prev} 中心热气体",
            "outer_cold": f"snap {snap_prev} 晕内外部冷气体",
            "wind": f"snap {snap_prev} 风相粒子",
            "star": f"snap {snap_prev} 恒星",
            "satellite": f"snap {snap_prev} 卫星星系",
            "outside_r200c": f"snap {snap_prev} 的 R200c 以外",
            "black_hole": f"snap {snap_prev} 黑洞",
            "other": "未匹配或其他状态",
        },
        "fate": {
            "central_cold": f"snap {snap_cur} 中心冷气体",
            "central_star": f"snap {snap_cur} 中心恒星",
            "halo_hot": f"snap {snap_cur} 晕内热气体",
            "outer_cold": f"snap {snap_cur} 晕内外部冷气体",
            "wind": f"snap {snap_cur} 风相粒子",
            "outside_r200c": f"snap {snap_cur} 的 R200c 以外",
            "satellite": f"snap {snap_cur} 卫星星系",
            "black_hole": f"snap {snap_cur} 黑洞",
            "other": "未匹配或其他状态",
        },
    }


def discover_state_names(
    history: dict[str, np.ndarray],
    halo_results: dict[str, np.ndarray],
    direction: str,
) -> list[str]:
    """Read state names from history metadata or infer them from field names."""

    history_key = f"{direction}_state_names"
    if history_key in history:
        return [clean_name(value) for value in history[history_key]]
    prefix = f"n_{direction}_"
    names = [
        key[len(prefix):]
        for key in halo_results
        if key.startswith(prefix)
        and key != f"n_{direction}_denominator"
    ]
    return sorted(names)


def summarize_history(
    history: dict[str, np.ndarray],
    halo_results: dict[str, np.ndarray],
    direction: str,
    *,
    snap_prev: int,
    snap_cur: int,
) -> dict:
    """Summarize one complete tracer origin or fate classification."""

    names = discover_state_names(history, halo_results, direction)
    denominator_key = f"n_{direction}_denominator"
    if denominator_key not in halo_results:
        return {
            "missing_denominator": denominator_key,
            "rows": [],
        }
    denominator = np.asarray(halo_results[denominator_key], dtype=float)
    meanings = _state_meanings(snap_prev, snap_cur)
    rows = []
    count_arrays = []
    all_count_fields_present = True
    for state in names:
        count_key = f"n_{direction}_{state}"
        fraction_key = f"fraction_{direction}_{state}"
        count_exists = count_key in halo_results
        fraction_exists = fraction_key in halo_results
        all_count_fields_present &= count_exists
        if count_exists:
            counts = np.asarray(halo_results[count_key], dtype=float)
            count_arrays.append(counts)
            tracer_total = (
                int(np.nansum(counts))
                if np.any(np.isfinite(counts))
                else np.nan
            )
            nonzero_halos = int(np.count_nonzero(counts > 0))
        else:
            counts = np.full(denominator.shape, np.nan)
            tracer_total = np.nan
            nonzero_halos = 0
        if fraction_exists:
            fractions = np.asarray(halo_results[fraction_key], dtype=float)
        else:
            fractions = np.full(denominator.shape, np.nan)
            valid_fraction = (
                np.isfinite(counts)
                & np.isfinite(denominator)
                & (denominator > 0)
            )
            fractions[valid_fraction] = (
                counts[valid_fraction] / denominator[valid_fraction]
            )
        finite_fraction = fractions[np.isfinite(fractions)]
        median_fraction = (
            np.median(finite_fraction) if len(finite_fraction) else np.nan
        )
        denominator_sum = np.nansum(denominator)
        global_fraction = (
            np.nansum(counts) / denominator_sum
            if count_exists and denominator_sum > 0
            else np.nan
        )
        rows.append(
            {
                "state": state,
                "物理含义": meanings[direction].get(
                    state,
                    "当前代码未登记的状态",
                ),
                "count field": count_key if count_exists else "MISSING",
                "fraction field": (
                    fraction_key
                    if fraction_exists
                    else "由 count/denominator 计算"
                ),
                "Σ Ntracer": tracer_total,
                "非零星系数": nonzero_halos,
                "逐星系比例中位数": median_fraction,
                "全样本加权比例": global_fraction,
            }
        )
    closure = None
    if all_count_fields_present and count_arrays:
        classified = np.sum(np.vstack(count_arrays), axis=0)
        valid = np.isfinite(denominator)
        difference = classified - denominator
        bad = valid & (difference != 0)
        denominator_total = np.sum(denominator[valid])
        closure = {
            "closed_halos": int(np.count_nonzero(valid & ~bad)),
            "unclosed_halos": int(np.count_nonzero(bad)),
            "max_absolute_difference": (
                float(np.max(np.abs(difference[valid])))
                if np.any(valid)
                else np.nan
            ),
            "weighted_fraction": (
                float(np.sum(classified[valid]) / denominator_total)
                if denominator_total > 0
                else np.nan
            ),
        }
    return {
        "missing_denominator": None,
        "denominator_key": denominator_key,
        "denominator_sum": int(np.nansum(denominator)),
        "rows": rows,
        "closure": closure,
    }


def summarize_tracer_channels(
    halo_results: dict[str, np.ndarray],
    *,
    snap_prev: int,
    snap_cur: int,
) -> list[dict]:
    """Summarize every available tracer-intersection transition channel."""

    channels = {
        "hot_halo_cooling": (
            f"snap {snap_prev} 外晕热气体"
            f" -> snap {snap_cur} 中心冷气体或中心恒星"
        ),
        "outer_cold_inflow": (
            f"snap {snap_prev} 外部冷气体"
            f" -> snap {snap_cur} 中心冷气体或中心恒星"
        ),
        "cooling_plus_inflow": "hot_halo_cooling + outer_cold_inflow",
        "tng_cold_to_hot_transition": (
            f"snap {snap_prev} 中心冷气体"
            f" -> snap {snap_cur} 晕内热气体"
        ),
        "outer_hot_to_inner_hot": (
            f"snap {snap_prev} 外晕热气体"
            f" -> snap {snap_cur} 中心区域热气体"
        ),
        "outer_cold_to_inner_hot": (
            f"snap {snap_prev} 外部冷气体"
            f" -> snap {snap_cur} 中心区域热气体"
        ),
        "inner_hot_delivery": (
            "outer_hot_to_inner_hot + outer_cold_to_inner_hot"
        ),
        "tng_total_inward_delivery": (
            "cooling_plus_inflow + inner_hot_delivery"
        ),
    }
    rows = []
    for channel, meaning in channels.items():
        candidate_fields = [
            f"n_{channel}",
            f"mass_{channel}_msun",
            f"rate_{channel}_msun_per_yr",
            f"fraction_{channel}",
            f"poisson_fraction_{channel}",
        ]
        present_fields = [
            field for field in candidate_fields if field in halo_results
        ]
        if not present_fields:
            continue
        n_field = f"n_{channel}"
        mass_field = f"mass_{channel}_msun"
        rate_field = f"rate_{channel}_msun_per_yr"
        rows.append(
            {
                "channel": channel,
                "物理含义": meaning,
                "内存中的字段": ", ".join(present_fields),
                "Σ Ntracer": (
                    int(
                        np.nansum(
                            np.asarray(halo_results[n_field], dtype=float)
                        )
                    )
                    if n_field in halo_results
                    else np.nan
                ),
                "质量中位数 [Msun]": (
                    np.nanmedian(
                        np.asarray(halo_results[mass_field], dtype=float)
                    )
                    if mass_field in halo_results
                    else np.nan
                ),
                "速率中位数 [Msun/yr]": (
                    np.nanmedian(
                        np.asarray(halo_results[rate_field], dtype=float)
                    )
                    if rate_field in halo_results
                    else np.nan
                ),
            }
        )
    return rows
