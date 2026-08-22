"""Synthetic checks for diagnostics extracted from the merged notebook."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.history_diagnostics import build_top4_history_statistics
from src.analysis.supply_diagnostics import (
    add_cold_to_hot_supply_percentiles,
    build_supply_comparison_statistics,
    compute_central_cold_to_outer_cold_rate,
)
from src.physics.feedback import apply_henriques_radio_mode
from src.plotting import (
    plot_supply_comparison,
    plot_tracer_history_top4_stacked_bars,
)


def _base_statistics() -> dict[str, np.ndarray]:
    """Return two-bin coordinates used by the synthetic figures."""

    return {
        "mass_bin_center": np.array([8.625, 8.875]),
        "mass_bin_low": np.array([8.50, 8.75]),
        "mass_bin_high": np.array([8.75, 9.00]),
    }


def test_radio_mode_with_zero_black_hole_keeps_cooling() -> None:
    """A zero black-hole mass must produce zero radio-mode suppression."""

    effective, suppression = apply_henriques_radio_mode(
        np.array([10.0]),
        np.array([1.0e11]),
        np.array([0.0]),
        np.array([200.0]),
    )
    np.testing.assert_allclose(effective, [10.0])
    np.testing.assert_allclose(suppression, [0.0])


def test_supply_diagnostics_preserve_per_halo_sums() -> None:
    """Combined supply must be formed per halo before percentile statistics."""

    halos = {
        "mass_bin_index": np.array([0, 0, 0]),
        "rate_cooling_plus_inflow_msun_per_yr": np.array([0.0, 0.0, 100.0]),
        "rate_tng_cold_to_hot_transition_msun_per_yr": np.array(
            [100.0, 0.0, 0.0]
        ),
        "rate_sam_isothermal_msun_per_yr": np.array([3.0, 4.0, 5.0]),
        "n_fate_outer_cold": np.array([4.0, 0.0, 0.0]),
        "dt_gyr": np.array([2.0e-9, 2.0e-9, 2.0e-9]),
        "mstar_mean_msun": np.full(3, 1.0e9),
        "m200c_mean_msun": np.full(3, 1.0e9),
        "m_hot_mean_msun": np.full(3, 1.0e9),
        "m_cgm_mean_msun": np.full(3, 1.0e9),
    }
    rates, valid = compute_central_cold_to_outer_cold_rate(
        halos,
        tracer_mass_msun=2.0,
    )
    np.testing.assert_allclose(rates, [4.0, 0.0, 0.0])
    assert np.all(valid)
    statistics = add_cold_to_hot_supply_percentiles(
        _base_statistics(),
        halos,
        "mstar",
    )
    assert statistics["cooling_plus_cold_to_hot_median"][0] == 100.0

    summary = build_supply_comparison_statistics(
        halos,
        _base_statistics(),
        normalization="mstar",
        tracer_mass_msun=2.0,
        random_seed=42,
        seed_offset=1000,
        bootstrap_samples=64,
    )
    figure = plot_supply_comparison(
        summary,
        _base_statistics(),
        normalization="mstar",
        log_y=False,
    )
    assert [line.get_label() for line in figure.axes[0].lines] == [
        "L-GALAXIES cooling",
        "TNG retained cold supply",
        r"TNG supply incl. cold$\rightarrow$hot",
        r"TNG supply incl. cold$\rightarrow$hot/cold",
    ]
    import matplotlib.pyplot as plt

    plt.close(figure)


def test_top4_history_statistics_close_and_plot() -> None:
    """Grouped histories must retain complete fractions in every valid bin."""

    history = {
        **_base_statistics(),
        "origin_state_names": np.array(
            ["central_cold", "outer_hot", "central_hot", "wind", "other"]
        ),
        "origin_fractions": np.array(
            [
                [0.4, 0.3],
                [0.3, 0.3],
                [0.2, 0.2],
                [0.05, 0.1],
                [0.05, 0.1],
            ]
        ),
        "origin_denominator_sum": np.array([10.0, 10.0]),
        "fate_state_names": np.array(
            ["central_cold", "central_star", "halo_hot", "wind", "other"]
        ),
        "fate_fractions": np.array(
            [
                [0.4, 0.3],
                [0.3, 0.3],
                [0.2, 0.2],
                [0.05, 0.1],
                [0.05, 0.1],
            ]
        ),
        "fate_denominator_sum": np.array([10.0, 10.0]),
    }
    halos = {
        "n_origin_central_cold": np.array([4, 3]),
        "n_origin_outer_hot": np.array([3, 3]),
        "n_origin_central_hot": np.array([2, 2]),
        "n_origin_wind": np.array([0.5, 1.0]),
        "n_origin_other": np.array([0.5, 1.0]),
        "n_fate_central_cold": np.array([4, 3]),
        "n_fate_central_star": np.array([3, 3]),
        "n_fate_halo_hot": np.array([2, 2]),
        "n_fate_wind": np.array([0.5, 1.0]),
        "n_fate_other": np.array([0.5, 1.0]),
    }
    grouped = build_top4_history_statistics(history, halos)
    for prefix in ("origin", "fate"):
        np.testing.assert_allclose(
            np.nansum(grouped[prefix]["fractions"], axis=0),
            [1.0, 1.0],
        )
    figure = plot_tracer_history_top4_stacked_bars(history, grouped)
    assert len(figure.axes) == 2
    import matplotlib.pyplot as plt

    plt.close(figure)


if __name__ == "__main__":
    test_radio_mode_with_zero_black_hole_keeps_cooling()
    test_supply_diagnostics_preserve_per_halo_sums()
    test_top4_history_statistics_close_and_plot()
    print("Extracted diagnostic tests passed")
