"""Synthetic tests for Phase 2.1 scientific and structural contracts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.events import classify_halo_events
from src.analysis.statistics import compute_agn_quartile_statistics, compute_composition_statistics
from src.io.catalog import _normalise_catalogue, subfind_counts
from src.io.sampling import _select_candidates
from src.physics.cooling_function import ConstantCoolingFunction
from src.physics.feedback import compute_agn_strength
from src.physics.sam_cooling import compute_isothermal_sam_cooling
from src.utils.config import Phase21Config


def test_config_uses_merged_final_bin_and_seed() -> None:
    """The final 11--11.5 dex bin is one sampling population."""

    config = Phase21Config()
    np.testing.assert_allclose(config.mass_bin_edges[-2:], [11.0, 11.5])
    assert config.random_seed == 202608
    assert config.tracer_workers == 64
    assert config.state_workers == 4
    assert config.state_parallel_backend == "thread"


def test_snapshot_catalogue_cache_reproduces_subfind_counts() -> None:
    """Bulk catalogue arrays provide the same central/satellite split."""

    halos = {
        "GroupFirstSub": np.array([0, 2], dtype=np.int64),
        "GroupNsubs": np.array([2, 1], dtype=np.int64),
    }
    subhalos = {
        "SubhaloFlag": np.array([True, True, True]),
        "SubhaloMassInRadType": np.ones((3, 6), dtype=float),
        "SubhaloLenType": np.array(
            [
                [10, 0, 0, 0, 4, 0],
                [3, 0, 0, 0, 2, 0],
                [8, 0, 0, 0, 5, 0],
            ],
            dtype=np.int64,
        ),
        "SubhaloVmax": np.ones(3),
        "SubhaloBHMass": np.zeros(3),
    }
    catalogue = _normalise_catalogue(halos, subhalos)
    assert subfind_counts(catalogue, 0) == {
        "central_gas_count": 10,
        "satellite_gas_count": 3,
        "central_star_count": 4,
        "satellite_star_count": 2,
    }


def test_sampling_is_stable_after_subfind_sort() -> None:
    """Candidate order changes cannot change a bin's selected IDs."""

    config = Phase21Config(sample_per_bin=2)
    candidates = [
        {"subfind_id_z99": 20, "mass_bin_index": 0, "log_mstar_z99": 8.6},
        {"subfind_id_z99": 10, "mass_bin_index": 0, "log_mstar_z99": 8.6},
        {"subfind_id_z99": 30, "mass_bin_index": 0, "log_mstar_z99": 8.6},
    ]
    first, _ = _select_candidates(candidates, config)
    second, _ = _select_candidates(list(reversed(candidates)), config)
    assert [row["subfind_id_z99"] for row in first] == [row["subfind_id_z99"] for row in second]


def _synthetic_event_inputs():
    """Create gas histories covering every resolved event class."""

    config = Phase21Config()
    headers = {snap: {"BoxSize": 1000.0} for snap in config.snapshots}
    states = {
        snap: {
            "group_center_ckpc_h": np.zeros(3),
            "subhalo_center_ckpc_h": np.zeros(3),
            "r200c_ckpc_h": 100.0,
            "central_gas_ids": np.empty(0, dtype=np.uint64),
            "satellite_gas_ids": np.empty(0, dtype=np.uint64),
            "central_star_ids": np.empty(0, dtype=np.uint64),
            "satellite_star_ids": np.empty(0, dtype=np.uint64),
        }
        for snap in config.snapshots
    }
    parent_maps = {snap: {} for snap in config.snapshots}
    records = {snap: {} for snap in config.snapshots}

    def add(tracer, snap, radius, *, cold=False, hot=False, central=False, bad=False):
        parent = tracer * 1000 + snap
        parent_maps[snap][tracer] = parent
        records[snap][parent] = {
            "particle_id": parent,
            "particle_type": 4 if bad else 0,
            "coordinates": np.array([radius, 0.0, 0.0]),
            "sfr": 0.0,
            "internal_energy": 100.0 if cold else 20_000.0,
            "electron_abundance": 0.1,
            "formation_time": 1.0,
        }
        if central:
            states[snap]["central_gas_ids"] = np.append(states[snap]["central_gas_ids"], parent)

    # first-in
    for snap in range(90, 96):
        add(1, snap, 20, cold=True)
    # recycled-in-a
    for snap in range(90, 96):
        add(2, snap, 5 if snap == 92 else 20, cold=True, central=snap == 92)
    # stay-in
    for snap in range(90, 96):
        add(3, snap, 5, hot=True, central=True)
    # single-in
    for snap in range(90, 93):
        add(4, snap, 20, cold=True)
    for snap in range(93, 96):
        add(4, snap, 5, hot=True, central=True)
    # recycled-in-b
    add(5, 90, 5, hot=True, central=True)
    add(5, 91, 20, hot=True)
    for snap in range(92, 96):
        add(5, snap, 5, hot=True, central=True)
    # other history
    for snap in range(90, 96):
        add(6, snap, 20, cold=True, bad=snap == 92)
    # stay-out and recycled-out
    for snap in range(94, 100):
        add(7, snap, 5 if snap == 94 else 20, cold=snap == 94, hot=snap > 94, central=snap == 94)
        add(8, snap, 5 if snap in (94, 96) else 20, cold=snap in (94, 96), hot=snap not in (94, 96), central=snap in (94, 96))
    return config, headers, states, parent_maps, records


def test_event_classes_are_complete_and_close() -> None:
    """All valid and other events appear exactly once in the ledger."""

    config, headers, states, parent_maps, records = _synthetic_event_inputs()
    result, ledger = classify_halo_events(
        np.array([1, 2, 3, 4, 5, 6], dtype=np.uint64),
        np.array([7, 8], dtype=np.uint64),
        parent_maps=parent_maps,
        records=records,
        states=states,
        headers=headers,
        config=config,
        dt_gyr=0.134,
    )
    assert {entry["rate_class"] for entry in ledger} >= {"first-in", "recycled-in-a", "stay-in", "single-in", "recycled-in-b", "other", "stay-out", "recycled-out"}
    assert result["n_total_in"] == sum(result[key] for key in ("n_first_in", "n_recycled_in_a", "n_stay_in", "n_single_in", "n_recycled_in_b", "n_other_in"))
    assert result["n_total_out"] == result["n_stay_out"] + result["n_recycled_out"] + result["n_other_out"]
    assert np.isclose(result["mass_in_closure_error_msun"], 0.0)
    assert np.isclose(result["mass_out_closure_error_msun"], 0.0)


def test_agn_average_physical_inputs_and_cooling() -> None:
    """The pure physics functions produce finite positive diagnostics."""

    cooling = compute_isothermal_sam_cooling(
        m_hot_msun=1.0e10,
        z_hot_mass_fraction=0.01,
        m200c_msun=1.0e12,
        r200c_pkpc=100.0,
        cooling_function=ConstantCoolingFunction(),
    )
    assert cooling["rate_sam_isothermal_msun_per_yr"] >= 0
    strength = compute_agn_strength(m_hot_msun=1.0e10, m_bh_msun=1.0e8, r200c_pkpc=100.0, v200c_km_s=200.0)
    assert np.isfinite(strength["agn_strength"])


def test_composition_and_quartile_statistics() -> None:
    """Mass-weighted compositions and four AGN groups retain closure."""

    config = Phase21Config()
    results = {
        "mass_bin_index": np.array([0, 0, 1, 1]),
        "x_cool_mean": np.array([0.1, 0.2, 0.3, 0.4]),
        "agn_quartile": np.array([1, 2, 3, 4]),
        "rate_first_in_eff_msun_per_yr": np.ones(4),
        "rate_recycled_in_eff_msun_per_yr": np.ones(4),
        "rate_stay_in_msun_per_yr": np.ones(4),
        "rate_other_in_msun_per_yr": np.ones(4),
        "rate_stay_out_msun_per_yr": np.ones(4),
        "rate_recycled_out_msun_per_yr": np.ones(4),
        "rate_other_out_msun_per_yr": np.ones(4),
    }
    composition = compute_composition_statistics(results, config)
    np.testing.assert_allclose(np.nansum(composition["in_component_fractions"][:, :2], axis=0), 1.0)
    np.testing.assert_allclose(np.nansum(composition["out_component_fractions"][:, :2], axis=0), 1.0)
    agn = compute_agn_quartile_statistics(results, config)
    assert agn["n_finite"][0, 0] == 1


if __name__ == "__main__":
    test_config_uses_merged_final_bin_and_seed()
    test_sampling_is_stable_after_subfind_sort()
    test_event_classes_are_complete_and_close()
    test_agn_average_physical_inputs_and_cooling()
    test_composition_and_quartile_statistics()
    print("Phase 2.1 synthetic tests passed")
