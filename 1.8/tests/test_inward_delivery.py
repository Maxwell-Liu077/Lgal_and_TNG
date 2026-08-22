"""Synthetic regression tests for the Phase 1.8.1 branch extensions."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

# Import the unified implementation from the project package.  The
# inward-delivery extension now shares all common modules with Phase 1.8.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import Phase18Config
from src.analysis.mass_statistics import (
    compute_all_normalized_statistics,
    compute_binned_statistics,
)
from src.plotting import (
    plot_inner_hot_delivery_components,
    plot_normalized_mass_dependence,
    plot_tracer_history_heatmaps,
)
from src.analysis.rates import compute_total_inward_delivery
from src.io.tracer_history import inner_hot_parent_mask


def test_total_inward_delivery_adds_per_halo_channels() -> None:
    result = compute_total_inward_delivery(
        {
            "n_cooling_plus_inflow": 7,
            "mass_cooling_plus_inflow_msun": 14.0,
            "rate_cooling_plus_inflow_msun_per_yr": 2.0,
        },
        n_outer_hot_to_inner_hot=2,
        n_outer_cold_to_inner_hot=1,
        tracer_mass_msun=2.0,
        dt_gyr=4.0e-9,
    )
    assert result == {
        "n_outer_hot_to_inner_hot": 2,
        "mass_outer_hot_to_inner_hot_msun": 4.0,
        "rate_outer_hot_to_inner_hot_msun_per_yr": 1.0,
        "n_outer_cold_to_inner_hot": 1,
        "mass_outer_cold_to_inner_hot_msun": 2.0,
        "rate_outer_cold_to_inner_hot_msun_per_yr": 0.5,
        "n_inner_hot_delivery": 3,
        "mass_inner_hot_delivery_msun": 6.0,
        "rate_inner_hot_delivery_msun_per_yr": 1.5,
        "n_tng_total_inward_delivery": 10,
        "mass_tng_total_inward_delivery_msun": 20.0,
        "rate_tng_total_inward_delivery_msun_per_yr": 3.5,
    }


def _synthetic_halo_results() -> dict[str, np.ndarray]:
    inward = np.array([0.0, 0.0, 100.0])
    outer_hot_inner_hot = np.array([100.0, 0.0, 0.0])
    outer_cold_inner_hot = np.array([0.0, 50.0, 0.0])
    inner_hot = outer_hot_inner_hot + outer_cold_inner_hot
    total_delivery = inward + inner_hot
    return {
        "mass_bin_index": np.array([0, 0, 0], dtype=np.int16),
        "rate_sam_isothermal_msun_per_yr": np.array([3.0, 4.0, 5.0]),
        "rate_tng_cold_to_hot_transition_msun_per_yr": np.full(3, 999.0),
        "rate_outer_cold_inflow_msun_per_yr": np.array([0.0, 0.0, 40.0]),
        "rate_hot_halo_cooling_msun_per_yr": np.array([0.0, 0.0, 60.0]),
        "rate_cooling_plus_inflow_msun_per_yr": inward,
        "rate_outer_hot_to_inner_hot_msun_per_yr": outer_hot_inner_hot,
        "rate_outer_cold_to_inner_hot_msun_per_yr": outer_cold_inner_hot,
        "rate_inner_hot_delivery_msun_per_yr": inner_hot,
        "rate_tng_total_inward_delivery_msun_per_yr": total_delivery,
        "mass_tng_cold_to_hot_transition_msun": np.full(3, 999.0e8),
        "mass_outer_hot_to_inner_hot_msun": outer_hot_inner_hot * 1.0e8,
        "mass_outer_cold_to_inner_hot_msun": (
            outer_cold_inner_hot * 1.0e8
        ),
        "mass_inner_hot_delivery_msun": inner_hot * 1.0e8,
        "mass_tng_total_inward_delivery_msun": total_delivery * 1.0e8,
        "m200c_mean_msun": np.full(3, 1.0e12),
        "m_hot_mean_msun": np.full(3, 1.0e10),
        "m_cgm_mean_msun": np.full(3, 2.0e10),
        "mstar_mean_msun": np.full(3, 1.0e9),
    }


def test_statistics_use_per_halo_sum_before_median() -> None:
    config = Phase18Config()
    results = _synthetic_halo_results()
    statistics = compute_binned_statistics(results, config)
    assert statistics["tng_total_inward_delivery_median"][0] == 100.0
    separate_median_sum = (
        statistics["cooling_plus_inflow_median"][0]
        + statistics["outer_hot_to_inner_hot_median"][0]
        + statistics["outer_cold_to_inner_hot_median"][0]
    )
    assert separate_median_sum == 0.0
    assert statistics["tng_total_inward_delivery_median"][0] != (
        separate_median_sum
    )


def test_normalized_plot_contains_only_branch_series() -> None:
    config = Phase18Config()
    normalized = compute_all_normalized_statistics(
        _synthetic_halo_results(),
        config,
    )
    figure = plot_normalized_mass_dependence(
        normalized["mstar"],
        normalization="mstar",
        log_y=False,
    )
    labels = {line.get_label() for line in figure.axes[0].lines}
    assert labels == {
        "L-GALAXIES",
        "Hot + Cold",
        "TNG total inward delivery",
    }
    import matplotlib.pyplot as plt

    plt.close(figure)


def test_inner_hot_mask_applies_phase_radius_and_membership() -> None:
    records = {
        "particle_ids": np.array([1, 2, 3, 4, 5], dtype=np.uint64),
        "particle_type": np.array([0, 0, 0, 0, 4], dtype=np.int8),
        "coordinates": np.array(
            [[5, 0, 0], [20, 0, 0], [5, 0, 0], [5, 0, 0], [5, 0, 0]],
            dtype=float,
        ),
        "sfr": np.array([0, 0, 0, 0, np.nan], dtype=float),
        "internal_energy": np.array(
            [1.0e4, 1.0e4, 1.0e2, 1.0e4, np.nan],
            dtype=float,
        ),
        "electron_abundance": np.array(
            [1, 1, 1, 1, np.nan],
            dtype=float,
        ),
        "formation_time": np.array(
            [np.nan, np.nan, np.nan, np.nan, 0.9],
            dtype=float,
        ),
    }
    mask = inner_hot_parent_mask(
        np.array([1, 2, 3, 4, 5], dtype=np.uint64),
        records,
        branch_state={"satellite_gas_ids": np.array([4], dtype=np.uint64)},
        group_center_ckpc_h=np.zeros(3),
        subhalo_center_ckpc_h=np.zeros(3),
        r200c_ckpc_h=100.0,
        box_size_ckpc_h=1000.0,
        config=Phase18Config(),
    )
    assert np.array_equal(mask, [True, False, False, False, False])


def test_inner_hot_component_plot_keeps_both_sources() -> None:
    statistics = compute_binned_statistics(
        _synthetic_halo_results(),
        Phase18Config(),
    )
    figure = plot_inner_hot_delivery_components(
        statistics,
        log_y=False,
    )
    labels = {line.get_label() for line in figure.axes[0].lines}
    assert labels == {
        "Outer hot to inner hot",
        "Outer cold to inner hot",
        "Inner-hot delivery",
    }
    import matplotlib.pyplot as plt

    plt.close(figure)


def test_heatmap_annotations_are_large_and_contrast_aware() -> None:
    statistics = {
        "mass_bin_low": np.array([8.50, 8.75]),
        "mass_bin_high": np.array([8.75, 9.00]),
        "origin_state_names": np.array(["central_cold", "outer_hot"]),
        "origin_fractions": np.array([[0.01, 0.90], [0.99, 0.10]]),
        "fate_state_names": np.array(["central_cold", "halo_hot"]),
        "fate_fractions": np.array([[0.90, 0.01], [0.10, 0.99]]),
    }
    figure = plot_tracer_history_heatmaps(
        statistics,
        cell_fontsize=11.5,
        cell_text_color="auto",
        cell_text_outline=True,
    )
    numeric_text = [
        text
        for axis in figure.axes[:2]
        for text in axis.texts
        if text.get_text().isdigit()
    ]
    assert numeric_text
    assert all(text.get_fontsize() == 11.5 for text in numeric_text)
    assert {text.get_color() for text in numeric_text} == {
        "black",
        "white",
    }
    assert all(text.get_path_effects() for text in numeric_text)
    import matplotlib.pyplot as plt

    plt.close(figure)


if __name__ == "__main__":
    test_total_inward_delivery_adds_per_halo_channels()
    test_statistics_use_per_halo_sum_before_median()
    test_normalized_plot_contains_only_branch_series()
    test_inner_hot_mask_applies_phase_radius_and_membership()
    test_inner_hot_component_plot_keeps_both_sources()
    test_heatmap_annotations_are_large_and_contrast_aware()
    print("1.8.1 branch tests passed")
