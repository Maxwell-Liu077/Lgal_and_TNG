"""Synthetic regression tests for the Phase 1.8 scientific contracts.

These tests require no real TNG files.  They protect selection, aperture,
tracer, cache-resume, cooling, normalization, and serialization behavior while
the production modules are reorganized.
"""

from __future__ import annotations

import sys
import types
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import Phase18Config
from src.physics.cooling_function import ConstantCoolingFunction
from src.analysis.rates import compute_particleid_rates, compute_tracer_rates
from src.io.region_state import (
    build_source_state,
    build_target_state,
    load_halo_pair_state,
    load_subfind_membership_catalog,
    subfind_member_ids,
    subfind_membership_counts,
)
from src.physics.sam_cooling import (
    compute_interval_sam_reheating,
    compute_isothermal_sam_cooling,
    compute_sam_reheating_loading,
)
from src.analysis.records import _combine_sam_endpoints
from src.analysis.mass_statistics import (
    compute_all_normalized_statistics,
    compute_binned_statistics,
)
from src.io.mpb import load_two_snapshot_mpb
from src.plotting import (
    plot_normalized_mass_dependence,
    plot_reheating_comparison,
    plot_tracer_history_heatmaps,
)
from src.io import sampling
from src.io.tracer_batch import (
    HOT,
    N_STATES,
    build_parent_catalog,
    build_tracer_id_catalog,
    parent_catalog_fingerprint,
    scan_snapshot_for_catalog,
    scan_snapshot_for_tracer_parents,
)
from src.io.tracer_history import (
    FATE_NAMES,
    ORIGIN_NAMES,
    classify_parent_states,
    compute_stacked_history_statistics,
    history_record,
    lookup_parent_particle_records,
)


def _synthetic_pair():
    config = Phase18Config()
    source_gas = {
        "ParticleIDs": np.array([1, 2, 3, 4], dtype=np.uint64),
        "Coordinates": np.array(
            [
                [20.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ]
        ),
        "Masses": np.ones(4),
        "StarFormationRate": np.zeros(4),
        "InternalEnergy": np.array([1e4, 1e2, 1e4, 1e2]),
        "ElectronAbundance": np.ones(4),
        "GFM_Metallicity": np.full(4, 0.01),
    }
    target_gas = {
        "ParticleIDs": np.array([2, 9], dtype=np.uint64),
        "Coordinates": np.array([[5.0, 0.0, 0.0], [6.0, 0.0, 0.0]]),
        "Masses": np.array([2.0, 3.0]),
        "StarFormationRate": np.zeros(2),
        "InternalEnergy": np.array([1e2, 1e2]),
        "ElectronAbundance": np.ones(2),
        "GFM_Metallicity": np.full(2, 0.01),
    }
    target_stars = {
        "ParticleIDs": np.array([1, 10], dtype=np.uint64),
        "Coordinates": np.array([[4.0, 0.0, 0.0], [30.0, 0.0, 0.0]]),
        "Masses": np.array([4.0, 5.0]),
        "GFM_InitialMass": np.array([4.5, 5.5]),
        "GFM_StellarFormationTime": np.array([0.9, 0.9]),
    }
    source = build_source_state(
        source_gas,
        group_center_ckpc_h=np.zeros(3),
        subhalo_center_ckpc_h=np.zeros(3),
        r200c_ckpc_h=100.0,
        box_size_ckpc_h=1000.0,
        scale_factor=0.9,
        config=config,
    )
    target = build_target_state(
        target_gas,
        target_stars,
        subhalo_center_ckpc_h=np.zeros(3),
        r200c_ckpc_h=100.0,
        box_size_ckpc_h=1000.0,
        config=config,
    )
    return config, source, target


def test_config_uses_mstar_8p5_to_11p5():
    config = Phase18Config()
    assert len(config.mass_bin_centers) == 12
    np.testing.assert_allclose(
        config.mass_bin_edges[[0, -1]],
        [8.5, 11.5],
    )
    assert config.mass_bin_width == 0.25
    assert config.sample_per_bin == 50
    assert config.minimum_bin_warning_count == 15
    assert config.tracer_workers == 64
    assert config.tracer_parallel_backend == "process"


def test_sampling_uses_fifty_and_warns_below_fifteen():
    config = Phase18Config(
        mpb_workers=1,
        mpb_parallel_backend="serial",
    )
    bin_index = np.concatenate(
        [
            np.zeros(60, dtype=np.int16),
            np.ones(20, dtype=np.int16),
            np.full(10, 2, dtype=np.int16),
        ]
    )
    candidates = {
        "group_id_z0": np.arange(90, dtype=np.int64),
        "subhalo_id_z0": np.arange(1000, 1090, dtype=np.int64),
        "mstar_z0_msun": np.concatenate(
            [
                np.full(60, 10**8.625),
                np.full(20, 10**8.875),
                np.full(10, 10**9.125),
            ]
        ),
        "log_mstar_z0": np.concatenate(
            [
                np.full(60, 8.625),
                np.full(20, 8.875),
                np.full(10, 9.125),
            ]
        ),
        "m200c_z0_msun": np.concatenate(
            [
                np.full(60, 10**10.1),
                np.full(20, 10**10.3),
                np.full(10, 10**10.5),
            ]
        ),
        "log_m200c_z0": np.concatenate(
            [
                np.full(60, 10.1),
                np.full(20, 10.3),
                np.full(10, 10.5),
            ]
        ),
        "r200c_z0_ckpc_h": np.full(90, 100.0),
        "mass_bin_index": bin_index,
    }

    def fake_mpb(base_path, subhalo_id, **kwargs):
        return {
            "subhalo_id_z0": subhalo_id,
            "snap_prev": 98,
            "subfind_id_prev": subhalo_id - 1,
            "group_id_prev": subhalo_id - 900,
            "group_first_sub_prev": subhalo_id - 1,
            "group_center_prev_ckpc_h": np.zeros(3),
            "subhalo_center_prev_ckpc_h": np.zeros(3),
            "mstar_prev_msun": 0.9e9,
            "m200c_prev_msun": 1e11,
            "r200c_prev_ckpc_h": 100.0,
            "vmax_prev_km_s": 100.0,
            "snap_cur": 99,
            "subfind_id_cur": subhalo_id,
            "group_id_cur": subhalo_id - 1000,
            "group_first_sub_cur": subhalo_id,
            "group_center_cur_ckpc_h": np.zeros(3),
            "subhalo_center_cur_ckpc_h": np.zeros(3),
            "mstar_cur_msun": 1e9,
            "m200c_cur_msun": 1.1e11,
            "r200c_cur_ckpc_h": 105.0,
            "vmax_cur_km_s": 105.0,
        }

    original_candidates = sampling.load_z0_central_candidates
    original_mpb = sampling.load_two_snapshot_mpb
    sampling.load_z0_central_candidates = lambda *args: candidates
    sampling.load_two_snapshot_mpb = fake_mpb
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            table, metadata = sampling.select_two_snapshot_sample(
                "unused",
                config,
                verbose=False,
            )
    finally:
        sampling.load_z0_central_candidates = original_candidates
        sampling.load_two_snapshot_mpb = original_mpb

    assert metadata["selected_counts"][:3] == [50, 20, 10]
    assert len(table["sample_index"]) == 80
    assert any("has only 10 candidate galaxies" in str(item.message)
               for item in caught)


def test_candidate_selection_uses_mstar_in_rad_and_subhalo_flag():
    config = Phase18Config()
    stellar_mass_type = np.zeros((3, 6), dtype=float)
    stellar_mass_type[:, 4] = (
        np.array([1.0e9, 2.0e9, 1.0e8])
        * config.h
        / 1.0e10
    )

    class FakeGroupcat:
        @staticmethod
        def loadHalos(*args, **kwargs):
            return {
                "GroupFirstSub": np.array([0, 1, -1]),
                "Group_M_Crit200": np.full(3, 10.0),
                "Group_R_Crit200": np.full(3, 100.0),
            }

        @staticmethod
        def loadSubhalos(*args, **kwargs):
            return {
                "SubhaloFlag": np.array([True, False, True]),
                "SubhaloMassInRadType": stellar_mass_type,
            }

    previous = sys.modules.get("illustris_python")
    sys.modules["illustris_python"] = types.SimpleNamespace(
        groupcat=FakeGroupcat,
    )
    try:
        candidates = sampling.load_z0_central_candidates(
            "unused",
            config,
        )
    finally:
        if previous is None:
            sys.modules.pop("illustris_python", None)
        else:
            sys.modules["illustris_python"] = previous
    np.testing.assert_array_equal(candidates["subhalo_id_z0"], [0])
    np.testing.assert_allclose(candidates["mstar_z0_msun"], [1.0e9])
    np.testing.assert_allclose(candidates["log_mstar_z0"], [9.0])


def test_mpb_reads_endpoint_mstar_from_group_catalog():
    class FakeSublink:
        @staticmethod
        def loadTree(*args, **kwargs):
            return {
                "SnapNum": np.array([99, 98]),
                "SubfindID": np.array([10, 9]),
                "SubhaloGrNr": np.array([2, 1]),
                "SubhaloPos": np.zeros((2, 3)),
                "Group_M_Crit200": np.array([100.0, 90.0]),
                "Group_R_Crit200": np.array([200.0, 190.0]),
                "GroupPos": np.zeros((2, 3)),
            }

    class FakeGroupcat:
        @staticmethod
        def loadSingle(*args, **kwargs):
            if "haloID" in kwargs:
                return {
                    "GroupFirstSub": (
                        10 if int(kwargs["haloID"]) == 2 else 9
                    )
                }
            values = np.zeros(6)
            values[4] = (
                0.2 if int(kwargs["subhaloID"]) == 10 else 0.1
            )
            return {
                "SubhaloMassInRadType": values,
                "SubhaloVmax": 150.0,
            }

    previous = sys.modules.get("illustris_python")
    sys.modules["illustris_python"] = types.SimpleNamespace(
        sublink=FakeSublink,
        groupcat=FakeGroupcat,
    )
    try:
        result = load_two_snapshot_mpb(
            "unused",
            10,
            snap_prev=98,
            snap_cur=99,
            h=0.5,
        )
    finally:
        if previous is None:
            sys.modules.pop("illustris_python", None)
        else:
            sys.modules["illustris_python"] = previous
    assert result["mstar_prev_msun"] == 0.1 * 1.0e10 / 0.5
    assert result["mstar_cur_msun"] == 0.2 * 1.0e10 / 0.5


def test_region_states_and_particleid_channels():
    config, source, target = _synthetic_pair()
    np.testing.assert_array_equal(source["hot_ids"], [1])
    assert source["n_hot_gas"] == 2
    assert source["n_hot_source_gas"] == 1
    np.testing.assert_array_equal(source["outer_cold_ids"], [2])
    np.testing.assert_array_equal(target["central_cold_ids"], [2, 9])
    np.testing.assert_array_equal(target["central_star_ids"], [1])
    np.testing.assert_allclose(
        source["m_cgm_msun"],
        2.0e10 / config.h,
    )
    assert source["n_cgm_gas"] == 2
    result = compute_particleid_rates(source, target, dt_gyr=0.5)
    assert result["n_hot_halo_cooling"] == 1
    assert result["n_outer_cold_inflow"] == 1
    np.testing.assert_allclose(
        result["rate_cooling_plus_inflow_msun_per_yr"],
        result["rate_hot_halo_cooling_msun_per_yr"]
        + result["rate_outer_cold_inflow_msun_per_yr"],
    )


def test_subfind_membership_and_satellite_exclusion():
    catalog = {
        "group_first_sub": np.array([0]),
        "group_nsubs": np.array([3]),
        "subhalo_len_type": np.array(
            [
                [2, 0, 0, 0, 3, 0],
                [1, 0, 0, 0, 1, 0],
                [2, 0, 0, 0, 4, 0],
            ]
        ),
    }
    counts = subfind_membership_counts(
        catalog,
        group_id=0,
        central_subfind_id=0,
    )
    assert counts["central_gas_count"] == 2
    assert counts["satellite_gas_count"] == 3
    central, satellite = subfind_member_ids(
        np.array([10, 11, 20, 21, 22, 99], dtype=np.uint64),
        central_count=2,
        satellite_count=3,
        component="synthetic gas",
    )
    np.testing.assert_array_equal(central, [10, 11])
    np.testing.assert_array_equal(satellite, [20, 21, 22])

    config = Phase18Config()
    gas = {
        "ParticleIDs": np.array([10, 20, 99], dtype=np.uint64),
        "Coordinates": np.array(
            [[20.0, 0.0, 0.0], [20.0, 0.0, 0.0], [5.0, 0.0, 0.0]]
        ),
        "Masses": np.ones(3),
        "StarFormationRate": np.zeros(3),
        "InternalEnergy": np.full(3, 1e4),
        "ElectronAbundance": np.ones(3),
        "GFM_Metallicity": np.full(3, 0.01),
    }
    state = build_source_state(
        gas,
        group_center_ckpc_h=np.zeros(3),
        subhalo_center_ckpc_h=np.zeros(3),
        r200c_ckpc_h=100.0,
        box_size_ckpc_h=1000.0,
        scale_factor=1.0,
        config=config,
        excluded_gas_particle_ids=np.array([20], dtype=np.uint64),
    )
    np.testing.assert_array_equal(state["hot_ids"], [10])
    assert state["n_hot_gas"] == 2
    assert state["n_satellite_gas_excluded"] == 1


def test_membership_loader_accepts_single_field_array():
    class FakeGroupcat:
        @staticmethod
        def loadHalos(*args, **kwargs):
            return {
                "GroupFirstSub": np.array([0]),
                "GroupNsubs": np.array([1]),
            }

        @staticmethod
        def loadSubhalos(*args, **kwargs):
            return np.array([[2, 0, 0, 0, 3, 0]], dtype=np.int64)

    previous = sys.modules.get("illustris_python")
    sys.modules["illustris_python"] = types.SimpleNamespace(
        groupcat=FakeGroupcat,
    )
    try:
        catalog = load_subfind_membership_catalog("unused", 98)
    finally:
        if previous is None:
            sys.modules.pop("illustris_python", None)
        else:
            sys.modules["illustris_python"] = previous
    assert catalog["subhalo_len_type"].shape == (1, 6)
    assert catalog["subhalo_len_type"][0, 0] == 2


def test_halo_pair_state_applies_membership_to_both_endpoints():
    config = Phase18Config()

    def gas(ids, coordinates, internal_energy):
        ids = np.asarray(ids, dtype=np.uint64)
        return {
            "ParticleIDs": ids,
            "Coordinates": np.asarray(coordinates, dtype=float),
            "Masses": np.ones(len(ids)),
            "StarFormationRate": np.zeros(len(ids)),
            "InternalEnergy": np.full(len(ids), internal_energy),
            "ElectronAbundance": np.ones(len(ids)),
            "GFM_Metallicity": np.full(len(ids), 0.01),
        }

    previous_gas = gas(
        [10, 20, 99],
        [[20, 0, 0], [20, 0, 0], [20, 0, 0]],
        1.0e4,
    )
    current_gas = gas(
        [11, 21, 98],
        [[5, 0, 0], [5, 0, 0], [20, 0, 0]],
        1.0e2,
    )
    current_stars = {
        "ParticleIDs": np.array([30, 40], dtype=np.uint64),
        "Coordinates": np.array([[5, 0, 0], [5, 0, 0]], dtype=float),
        "Masses": np.ones(2),
        "GFM_InitialMass": np.ones(2),
        "GFM_StellarFormationTime": np.ones(2),
    }

    class FakeSnapshot:
        @staticmethod
        def loadHalo(base_path, snap_num, group_id, component, fields):
            if component == "stars":
                return current_stars
            return previous_gas if snap_num == 98 else current_gas

    previous = sys.modules.get("illustris_python")
    sys.modules["illustris_python"] = types.SimpleNamespace(
        snapshot=FakeSnapshot,
    )
    sample_row = {
        "group_id_prev": 0,
        "group_id_cur": 0,
        "group_center_prev_ckpc_h": np.zeros(3),
        "group_center_cur_ckpc_h": np.zeros(3),
        "subhalo_center_prev_ckpc_h": np.zeros(3),
        "subhalo_center_cur_ckpc_h": np.zeros(3),
        "r200c_prev_ckpc_h": 100.0,
        "r200c_cur_ckpc_h": 100.0,
        "central_gas_count_prev": 1,
        "satellite_gas_count_prev": 1,
        "central_gas_count_cur": 1,
        "satellite_gas_count_cur": 1,
        "central_star_count_prev": 1,
        "satellite_star_count_prev": 1,
        "central_star_count_cur": 1,
        "satellite_star_count_cur": 1,
    }
    try:
        state = load_halo_pair_state(
            "unused",
            sample_row,
            header_prev={"BoxSize": 1000.0, "Time": 0.9},
            header_cur={"BoxSize": 1000.0, "Time": 1.0},
            config=config,
        )
    finally:
        if previous is None:
            sys.modules.pop("illustris_python", None)
        else:
            sys.modules["illustris_python"] = previous
    np.testing.assert_array_equal(state["source"]["hot_ids"], [10, 99])
    np.testing.assert_array_equal(
        state["target"]["central_cold_ids"],
        [11],
    )
    np.testing.assert_array_equal(
        state["target"]["central_star_ids"],
        [30],
    )
    assert state["source"]["n_satellite_gas_excluded"] == 1
    assert state["current"]["n_satellite_gas_excluded"] == 1


def test_tracer_channels_add_exactly():
    result = compute_tracer_rates(
        tracer_hot_prev=np.array([1, 2, 3], dtype=np.uint64),
        tracer_outer_cold_prev=np.array([4, 5], dtype=np.uint64),
        tracer_central_cold_cur=np.array([2, 4, 8], dtype=np.uint64),
        tracer_central_star_cur=np.array([3, 5, 9], dtype=np.uint64),
        tracer_mass_msun=10.0,
        dt_gyr=0.5,
    )
    assert result["n_hot_halo_cooling"] == 2
    assert result["n_outer_cold_inflow"] == 2
    assert result["n_cooling_plus_inflow"] == 4
    np.testing.assert_allclose(
        result["rate_cooling_plus_inflow_msun_per_yr"],
        result["rate_hot_halo_cooling_msun_per_yr"]
        + result["rate_outer_cold_inflow_msun_per_yr"],
    )


def test_sam_reheating_is_energy_capped_and_nonnegative():
    previous = compute_sam_reheating_loading(
        vmax_km_s=80.0,
        v200c_km_s=100.0,
    )
    current = compute_sam_reheating_loading(
        vmax_km_s=90.0,
        v200c_km_s=110.0,
    )
    assert previous["sn_loading_effective"] == min(
        previous["sn_loading_disk"],
        previous["sn_loading_energy_cap"],
    )
    result = compute_interval_sam_reheating(
        formed_stellar_mass_msun=1.0e9,
        dt_gyr=0.1,
        previous_loading=previous,
        current_loading=current,
        rate_sam_isothermal_msun_per_yr=0.01,
    )
    assert result["rate_sam_sn_effective_msun_per_yr"] == 0.0
    assert result["sam_sn_cooling_clipped"]
    assert result["mass_sam_reheating_msun"] > 0


def test_complete_history_classification_and_closure():
    config = Phase18Config()
    particle_ids = np.arange(1, 11, dtype=np.uint64)
    records = {
        "particle_ids": particle_ids,
        "particle_type": np.array(
            [0, 0, 0, 0, 4, 4, 0, 0, 5, 0],
            dtype=np.int8,
        ),
        "coordinates": np.array(
            [
                [5, 0, 0],
                [20, 0, 0],
                [5, 0, 0],
                [20, 0, 0],
                [5, 0, 0],
                [5, 0, 0],
                [20, 0, 0],
                [120, 0, 0],
                [0, 0, 0],
                [20, 0, 0],
            ],
            dtype=float,
        ),
        "sfr": np.zeros(10),
        "internal_energy": np.array(
            [1e2, 1e4, 1e4, 1e2, np.nan, np.nan, 1e4, 1e2, np.nan, np.nan]
        ),
        "electron_abundance": np.array(
            [1, 1, 1, 1, np.nan, np.nan, 1, 1, np.nan, 1],
            dtype=float,
        ),
        "formation_time": np.array(
            [np.nan, np.nan, np.nan, np.nan, -0.5, 0.9,
             np.nan, np.nan, np.nan, np.nan]
        ),
    }
    branch_state = {
        "satellite_gas_ids": np.array([7], dtype=np.uint64),
        "central_star_ids_all": np.array([6], dtype=np.uint64),
        "satellite_star_ids": np.empty(0, dtype=np.uint64),
    }
    kwargs = {
        "branch_state": branch_state,
        "group_center_ckpc_h": np.zeros(3),
        "subhalo_center_ckpc_h": np.zeros(3),
        "r200c_ckpc_h": 100.0,
        "box_size_ckpc_h": 1000.0,
        "config": config,
    }
    origin = classify_parent_states(
        particle_ids,
        records,
        direction="origin",
        **kwargs,
    )
    fate = classify_parent_states(
        particle_ids,
        records,
        direction="fate",
        **kwargs,
    )
    assert set(origin) == set(range(len(ORIGIN_NAMES)))
    assert fate[FATE_NAMES.index("halo_hot")] == FATE_NAMES.index(
        "halo_hot"
    )
    result = history_record(
        origin_codes=origin,
        fate_codes=fate,
        n_origin_expected=10,
        n_fate_expected=10,
        tracer_mass_msun=10.0,
        dt_gyr=0.5,
    )
    assert result["fraction_origin_closure"] == 1.0
    assert result["fraction_fate_closure"] == 1.0
    assert result["n_tng_cold_to_hot_transition"] == 2

    halo_results = {
        "mass_bin_index": np.array([0, 0]),
        "n_origin_denominator": np.array([10, 10]),
        "n_fate_denominator": np.array([10, 10]),
    }
    for prefix, names in (("origin", ORIGIN_NAMES), ("fate", FATE_NAMES)):
        for name in names:
            halo_results[f"n_{prefix}_{name}"] = np.array(
                [result[f"n_{prefix}_{name}"]] * 2
            )
    statistics = compute_stacked_history_statistics(
        halo_results,
        config,
    )
    np.testing.assert_allclose(
        np.nansum(statistics["origin_fractions"][:, 0]),
        1.0,
    )
    np.testing.assert_allclose(
        np.nansum(statistics["fate_fractions"][:, 0]),
        1.0,
    )
    figure = plot_tracer_history_heatmaps(statistics)
    assert len(figure.axes) == 3
    import matplotlib.pyplot as plt

    plt.close(figure)


def test_particle_record_lookup_reads_gas_wind_star_and_bh():
    class FakeFile:
        def __init__(self, path, mode):
            self.values = {
                "PartType0": {
                    "ParticleIDs": np.array([10, 11], dtype=np.uint64),
                    "Coordinates": np.array([[5, 0, 0], [6, 0, 0]]),
                    "StarFormationRate": np.zeros(2),
                    "InternalEnergy": np.full(2, 1e2),
                    "ElectronAbundance": np.ones(2),
                },
                "PartType4": {
                    "ParticleIDs": np.array([20, 30], dtype=np.uint64),
                    "Coordinates": np.array([[5, 0, 0], [5, 0, 0]]),
                    "GFM_StellarFormationTime": np.array([-0.5, 0.9]),
                },
                "PartType5": {
                    "ParticleIDs": np.array([40], dtype=np.uint64),
                },
            }

        def __enter__(self):
            return self.values

        def __exit__(self, *args):
            return False

    previous_h5py = sys.modules.get("h5py")
    sys.modules["h5py"] = types.SimpleNamespace(File=FakeFile)
    try:
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            snapdir = base / "snapdir_099"
            snapdir.mkdir()
            (snapdir / "snap_099.0.hdf5").touch()
            records = lookup_parent_particle_records(
                base,
                99,
                np.array([10, 20, 30, 40], dtype=np.uint64),
                cache_dir=None,
                block_size=1,
                max_workers=1,
                verbose=False,
            )
    finally:
        if previous_h5py is None:
            sys.modules.pop("h5py", None)
        else:
            sys.modules["h5py"] = previous_h5py
    np.testing.assert_array_equal(
        records["particle_ids"],
        [10, 20, 30, 40],
    )
    np.testing.assert_array_equal(
        records["particle_type"],
        [0, 4, 4, 5],
    )


def test_parent_catalog_supports_shared_parents():
    states = [
        {
            "source": {
                "hot_ids": np.array([10], dtype=np.uint64),
                "outer_cold_ids": np.array([20], dtype=np.uint64),
            },
            "target": {
                "central_cold_ids": np.array([30], dtype=np.uint64),
                "central_star_ids": np.array([40], dtype=np.uint64),
            },
        },
        {
            "source": {
                "hot_ids": np.array([10], dtype=np.uint64),
                "outer_cold_ids": np.array([21], dtype=np.uint64),
            },
            "target": {
                "central_cold_ids": np.array([31], dtype=np.uint64),
                "central_star_ids": np.array([41], dtype=np.uint64),
            },
        },
    ]
    catalog = build_parent_catalog(states, role="source")
    np.testing.assert_array_equal(catalog["parent_ids"], [10, 20, 21])
    assert len(catalog["duplicate_unique_indices"]) == 1
    fingerprint = parent_catalog_fingerprint(catalog, snap_num=98)
    assert len(fingerprint) == 64


def test_batch_scanner_and_chunk_resume():
    states = [
        {
            "source": {
                "hot_ids": np.array([10], dtype=np.uint64),
                "outer_cold_ids": np.array([20], dtype=np.uint64),
            },
            "target": {
                "central_cold_ids": np.empty(0, dtype=np.uint64),
                "central_star_ids": np.empty(0, dtype=np.uint64),
            },
        },
        {
            "source": {
                "hot_ids": np.array([10], dtype=np.uint64),
                "outer_cold_ids": np.array([21], dtype=np.uint64),
            },
            "target": {
                "central_cold_ids": np.empty(0, dtype=np.uint64),
                "central_star_ids": np.empty(0, dtype=np.uint64),
            },
        },
    ]
    catalog = build_parent_catalog(states, role="source")

    class FakeFile:
        def __init__(self, path, mode):
            self.values = {
                "PartType3": {
                    "ParentID": np.array(
                        [10, 20, 21, 99],
                        dtype=np.uint64,
                    ),
                    "TracerID": np.array(
                        [100, 200, 210, 990],
                        dtype=np.uint64,
                    ),
                }
            }

        def __enter__(self):
            return self.values

        def __exit__(self, *args):
            return False

    previous_h5py = sys.modules.get("h5py")
    sys.modules["h5py"] = types.SimpleNamespace(File=FakeFile)
    try:
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            snapdir = base / "snapdir_098"
            snapdir.mkdir()
            (snapdir / "snap_098.0.hdf5").touch()
            cache = base / "cache"
            first = scan_snapshot_for_catalog(
                base,
                98,
                catalog,
                cache_dir=cache,
                block_size=2,
                max_workers=1,
                parallel_backend="serial",
                verbose=False,
            )
            np.testing.assert_array_equal(first[HOT], [100])
            np.testing.assert_array_equal(
                first[N_STATES + HOT],
                [100],
            )
            sys.modules["h5py"] = types.SimpleNamespace(
                File=lambda *args, **kwargs: (_ for _ in ()).throw(
                    RuntimeError("cache was not used")
                )
            )
            second = scan_snapshot_for_catalog(
                base,
                98,
                catalog,
                cache_dir=cache,
                block_size=2,
                max_workers=1,
                parallel_backend="serial",
                verbose=False,
            )
            np.testing.assert_array_equal(second[HOT], [100])
    finally:
        if previous_h5py is None:
            sys.modules.pop("h5py", None)
        else:
            sys.modules["h5py"] = previous_h5py


def test_tracer_to_parent_scanner_preserves_parent_multiplicity():
    catalog = build_tracer_id_catalog(
        [np.array([100, 101], dtype=np.uint64)]
    )

    class FakeFile:
        def __init__(self, path, mode):
            self.values = {
                "PartType3": {
                    "ParentID": np.array([10, 10, 99], dtype=np.uint64),
                    "TracerID": np.array([100, 101, 999], dtype=np.uint64),
                }
            }

        def __enter__(self):
            return self.values

        def __exit__(self, *args):
            return False

    previous_h5py = sys.modules.get("h5py")
    sys.modules["h5py"] = types.SimpleNamespace(File=FakeFile)
    try:
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            snapdir = base / "snapdir_099"
            snapdir.mkdir()
            (snapdir / "snap_099.0.hdf5").touch()
            result = scan_snapshot_for_tracer_parents(
                base,
                99,
                catalog,
                block_size=2,
                max_workers=1,
                parallel_backend="serial",
                verbose=False,
            )
    finally:
        if previous_h5py is None:
            sys.modules.pop("h5py", None)
        else:
            sys.modules["h5py"] = previous_h5py
    np.testing.assert_array_equal(result[0], [10, 10])


def test_sam_and_binned_statistics():
    sam_previous = compute_isothermal_sam_cooling(
        m_hot_msun=1e10,
        z_hot_mass_fraction=0.01,
        m200c_msun=1e12,
        r200c_pkpc=200.0,
        cooling_function=ConstantCoolingFunction(1e-23),
    )
    sam_current = compute_isothermal_sam_cooling(
        m_hot_msun=2e10,
        z_hot_mass_fraction=0.01,
        m200c_msun=1.1e12,
        r200c_pkpc=205.0,
        cooling_function=ConstantCoolingFunction(1e-23),
    )
    assert sam_previous["r_cool_isothermal_pkpc"] > 0
    assert sam_previous["rate_sam_isothermal_msun_per_yr"] > 0
    sam = _combine_sam_endpoints(sam_previous, sam_current)
    np.testing.assert_allclose(
        sam["rate_sam_isothermal_msun_per_yr"],
        0.5
        * (
            sam_previous["rate_sam_isothermal_msun_per_yr"]
            + sam_current["rate_sam_isothermal_msun_per_yr"]
        ),
    )

    config = Phase18Config()
    results = {
        "mass_bin_index": np.array([0, 0, 1]),
        "m200c_mean_msun": np.array([1e10, 2e10, 4e10]),
        "m_hot_mean_msun": np.array([1e8, 2e8, 4e8]),
        "m_cgm_mean_msun": np.array([1e9, 2e9, 4e9]),
        "mstar_mean_msun": np.array([1e7, 2e7, 4e7]),
        "rate_sam_isothermal_msun_per_yr": np.array([1.0, 3.0, 4.0]),
        "rate_sam_sn_effective_msun_per_yr": np.array([0.5, 2.0, 3.0]),
        "rate_sam_reheating_msun_per_yr": np.array([0.5, 1.0, 1.0]),
        "rate_tng_cold_to_hot_transition_msun_per_yr": np.array(
            [0.4, 0.8, 0.9]
        ),
        "mass_sam_reheating_msun": np.array([5e7, 1e8, 1e8]),
        "mass_tng_cold_to_hot_transition_msun": np.array(
            [4e7, 8e7, 9e7]
        ),
        "rate_outer_cold_inflow_msun_per_yr": np.array([2.0, 4.0, 5.0]),
        "rate_hot_halo_cooling_msun_per_yr": np.array([3.0, 5.0, 6.0]),
        "rate_cooling_plus_inflow_msun_per_yr": np.array([5.0, 9.0, 11.0]),
    }
    stats = compute_binned_statistics(results, config)
    assert stats["n_halos"][0] == 2
    assert stats["sam_median"][0] == 2.0
    assert np.isnan(stats["sam_median"][2])
    normalized = compute_all_normalized_statistics(results, config)
    np.testing.assert_allclose(
        normalized["m200c"]["sam_median"][0],
        np.median([0.1, 0.15]),
    )
    figure = plot_normalized_mass_dependence(
        normalized["m200c"],
        normalization="m200c",
        log_y=False,
    )
    axis = figure.axes[0]
    assert len(axis.lines) == 5
    assert [line.get_label() for line in axis.lines] == [
        "L-GALAXIES",
        "L-GALAXIES (SN)",
        "Hot to Cold",
        "Cold to Cold",
        "Hot + Cold",
    ]
    assert [line.get_color() for line in axis.lines] == [
        "#000000",
        "#E69F00",
        "#0072B2",
        "#009E73",
        "#CC79A7",
    ]
    assert all(line.get_marker() == "None" for line in axis.lines)
    np.testing.assert_array_equal(axis.get_xticks(), [9.0, 10.0, 11.0])
    assert axis.get_title() == ""
    assert [text.get_text() for text in axis.texts] == [r"$M_{200c}$"]
    assert not any(
        line.get_visible()
        for line in axis.get_xgridlines() + axis.get_ygridlines()
    )
    assert axis.xaxis.label.get_fontfamily() == ["STIXGeneral"]
    assert set(normalized) == {"m200c", "mhot", "mcgm", "mstar"}
    reheating_figure = plot_reheating_comparison(
        stats,
        log_y=False,
    )
    assert [
        line.get_label() for line in reheating_figure.axes[0].lines
    ] == [
        "L-GALAXIES reheating",
        "TNG cold-to-hot transition",
    ]
    import matplotlib.pyplot as plt

    plt.close(figure)
    plt.close(reheating_figure)


if __name__ == "__main__":
    test_config_uses_mstar_8p5_to_11p5()
    test_sampling_uses_fifty_and_warns_below_fifteen()
    test_candidate_selection_uses_mstar_in_rad_and_subhalo_flag()
    test_mpb_reads_endpoint_mstar_from_group_catalog()
    test_region_states_and_particleid_channels()
    test_subfind_membership_and_satellite_exclusion()
    test_membership_loader_accepts_single_field_array()
    test_halo_pair_state_applies_membership_to_both_endpoints()
    test_tracer_channels_add_exactly()
    test_sam_reheating_is_energy_capped_and_nonnegative()
    test_complete_history_classification_and_closure()
    test_particle_record_lookup_reads_gas_wind_star_and_bh()
    test_parent_catalog_supports_shared_parents()
    test_batch_scanner_and_chunk_resume()
    test_tracer_to_parent_scanner_preserves_parent_multiplicity()
    test_sam_and_binned_statistics()
    print("1.8 synthetic algorithm tests passed")
