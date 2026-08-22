"""Synthetic tests for Phase 2.1 scientific and structural contracts."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.events import classify_halo_events, classify_parent_state
import src.analysis.pipeline as pipeline_module
from src.analysis.pipeline import _assign_agn_quartiles, _mean_state_inputs
from src.analysis.statistics import compute_agn_quartile_statistics, compute_composition_statistics, normalize_halo_rates
from src.io.catalog import _normalise_catalogue, load_subhalo_particle_offsets, subfind_counts
from src.io.sampling import _select_candidates
from src.io.results import _save_events
from src.io.state import _save_state, build_snapshot_state, load_state_snapshot, state_cache_is_valid
from src.io.tracer import lookup_parent_records, scan_parent_to_tracer, scan_tracer_parent_map
from src.physics.cooling_function import ConstantCoolingFunction
from src.physics.feedback import compute_agn_strength
from src.physics.sam_cooling import compute_isothermal_sam_cooling
from src.plotting import plot_agn_cooling, plot_composition, plot_feedback_before_accretion
from src.utils.config import Phase21Config


def test_config_uses_merged_final_bin_and_seed() -> None:
    """The final 11--11.5 dex bin is one sampling population."""

    config = Phase21Config()
    np.testing.assert_allclose(config.mass_bin_edges[-2:], [11.0, 11.5])
    assert config.random_seed == 202608
    assert config.tracer_workers == 64
    assert config.state_workers == 1
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


def test_snapshot_state_keeps_mbh_from_branch(monkeypatch) -> None:
    """A valid particle read must not be rejected by the BH state field."""

    gas = {
        "ParticleIDs": np.array([1, 2, 3], dtype=np.uint64),
        "Coordinates": np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [20.0, 0.0, 0.0]]),
        "Masses": np.ones(3),
        "StarFormationRate": np.array([1.0, 0.0, 0.0]),
        "InternalEnergy": np.array([100.0, 100.0, 20_000.0]),
        "ElectronAbundance": np.full(3, 0.1),
        "GFM_Metallicity": np.full(3, 0.01),
    }
    stars = {
        "ParticleIDs": np.array([10], dtype=np.uint64),
        "Coordinates": np.array([[1.0, 0.0, 0.0]]),
        "GFM_StellarFormationTime": np.array([1.0]),
    }

    class Snapshot:
        @staticmethod
        def loadHalo(base_path, snap, group_id, particle_type, fields):
            return gas if particle_type == "gas" else stars

    monkeypatch.setitem(sys.modules, "illustris_python", types.SimpleNamespace(snapshot=Snapshot))
    catalogue = _normalise_catalogue(
        {"GroupFirstSub": np.array([0]), "GroupNsubs": np.array([2])},
        {
            "SubhaloFlag": np.array([True, True]),
            "SubhaloMassInRadType": np.ones((2, 6)),
            "SubhaloLenType": np.array([[2, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0]]),
            "SubhaloVmax": np.array([200.0, 100.0]),
            "SubhaloBHMass": np.array([1.0, 0.0]),
        },
    )
    branch = {
        "snap": 94,
        "group_id": 0,
        "subfind_id": 0,
        "group_center_ckpc_h": [0.0, 0.0, 0.0],
        "subhalo_center_ckpc_h": [0.0, 0.0, 0.0],
        "r200c_ckpc_h": 100.0,
        "m200c_msun": 1.0e12,
        "mstar_msun": 1.0e10,
        "m_bh_msun": 1.0e10,
    }
    state = build_snapshot_state(
        "/unused", branch, {"BoxSize": 1000.0, "Time": 0.5}, Phase21Config(), catalogue=catalogue
    )
    assert state["m_bh_msun"] == branch["m_bh_msun"]
    # If the MPB object is not GroupFirstSub, its own Subfind interval is the
    # tracked central membership and the FoF central is excluded as satellite.
    gas["Coordinates"][2] = [1.0, 0.0, 0.0]
    gas["StarFormationRate"][2] = 1.0
    satellite_branch = {**branch, "subfind_id": 1, "is_main_central": False}
    satellite_state = build_snapshot_state(
        "/unused", satellite_branch, {"BoxSize": 1000.0, "Time": 0.5}, Phase21Config(), catalogue=catalogue
    )
    np.testing.assert_array_equal(satellite_state["central_cold_ids"], np.array([3], dtype=np.uint64))


def test_state_cache_supports_selective_snapshot_loading(tmp_path) -> None:
    """Cached halo states can be validated and read without full materialization."""

    path = tmp_path / "state.npz"
    _save_state(
        path,
        {
            94: {"snap": 94, "central_cold_ids": np.array([1, 2], dtype=np.uint64), "unused": np.arange(100)},
            95: {"snap": 95, "central_cold_ids": np.array([3], dtype=np.uint64), "unused": np.arange(200)},
        },
    )
    assert state_cache_is_valid(path, (94, 95))
    state = load_state_snapshot(path, 95, fields=("central_cold_ids",))
    assert set(state) == {"central_cold_ids"}
    np.testing.assert_array_equal(state["central_cold_ids"], np.array([3], dtype=np.uint64))


def test_official_subhalo_offsets_are_loaded_by_particle_type(tmp_path) -> None:
    """The binding layer reads only requested columns from the TNG offsets file."""

    import h5py

    base_path = tmp_path / "output"
    base_path.mkdir()
    offset_dir = tmp_path / "postprocessing" / "offsets"
    offset_dir.mkdir(parents=True)
    with h5py.File(offset_dir / "offsets_094.hdf5", "w") as handle:
        group = handle.create_group("Subhalo")
        group["SnapByType"] = np.arange(18, dtype=np.int64).reshape(3, 6)
    offsets = load_subhalo_particle_offsets(base_path, 94, particle_types=(0, 4))
    np.testing.assert_array_equal(offsets[0], [0, 6, 12])
    np.testing.assert_array_equal(offsets[4], [4, 10, 16])


def test_tracer_scans_parallelize_and_cache_by_snapshot_chunk(tmp_path) -> None:
    """Chunk workers preserve duplicate labels and exact parent records."""

    import h5py

    base_path = tmp_path / "output"
    snap_dir = base_path / "snapdir_094"
    snap_dir.mkdir(parents=True)
    chunks = (
        (
            np.array([10, 20, 10], dtype=np.uint64),
            np.array([100, 200, 101], dtype=np.uint64),
            np.array([10, 20], dtype=np.uint64),
        ),
        (
            np.array([30, 99], dtype=np.uint64),
            np.array([300, 999], dtype=np.uint64),
            np.array([30], dtype=np.uint64),
        ),
    )
    for index, (parents, tracers, gas_ids) in enumerate(chunks):
        with h5py.File(snap_dir / f"snap_094.{index}.hdf5", "w") as handle:
            tracer_group = handle.create_group("PartType3")
            tracer_group["ParentID"] = parents
            tracer_group["TracerID"] = tracers
            gas = handle.create_group("PartType0")
            gas["ParticleIDs"] = gas_ids
            gas["Coordinates"] = np.column_stack((gas_ids, np.zeros((len(gas_ids), 2))))
            gas["StarFormationRate"] = np.zeros(len(gas_ids))
            gas["InternalEnergy"] = np.full(len(gas_ids), 100.0)
            gas["ElectronAbundance"] = np.full(len(gas_ids), 0.1)

    cache_dir = tmp_path / "cache"
    grouped = scan_parent_to_tracer(
        base_path,
        94,
        {0: np.array([10, 30], dtype=np.uint64), 1: np.array([10], dtype=np.uint64)},
        cache_dir=cache_dir,
        block_size=2,
        max_workers=2,
        parallel_backend="process",
    )
    np.testing.assert_array_equal(grouped[0], np.array([100, 101, 300], dtype=np.uint64))
    np.testing.assert_array_equal(grouped[1], np.array([100, 101], dtype=np.uint64))

    parent_map = scan_tracer_parent_map(
        base_path,
        94,
        np.array([100, 300], dtype=np.uint64),
        cache_dir=cache_dir,
        block_size=2,
        max_workers=2,
        parallel_backend="process",
    )
    assert parent_map == {100: 10, 300: 30}
    columnar_parent_map = scan_tracer_parent_map(
        base_path,
        94,
        np.array([100, 300], dtype=np.uint64),
        cache_dir=cache_dir,
        block_size=2,
        max_workers=2,
        parallel_backend="process",
        columnar=True,
    )
    np.testing.assert_array_equal(columnar_parent_map["tracer_ids"], np.array([100, 300], dtype=np.uint64))
    np.testing.assert_array_equal(columnar_parent_map["parent_ids"], np.array([10, 30], dtype=np.uint64))
    records = lookup_parent_records(
        base_path,
        94,
        np.array([10, 30], dtype=np.uint64),
        cache_dir=cache_dir,
        block_size=2,
        max_workers=2,
        parallel_backend="process",
        membership_catalogue={
            "GroupFirstSub": np.array([0, 1], dtype=np.int64),
            "GroupNsubs": np.array([1, 1], dtype=np.int64),
            "SubhaloLenType": np.array([[2, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]], dtype=np.int64),
        },
        subhalo_offsets={0: np.array([0, 2], dtype=np.uint64)},
    )
    np.testing.assert_array_equal(records["particle_ids"], np.array([10, 30], dtype=np.uint64))
    np.testing.assert_array_equal(records["particle_index"], np.array([0, 2], dtype=np.uint64))
    np.testing.assert_array_equal(records["bound_subhalo_id"], np.array([0, 1]))
    np.testing.assert_array_equal(records["bound_group_id"], np.array([0, 1]))
    assert len(list((cache_dir / "chunk_scans").glob("**/chunk_*.npz"))) == 6


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
    # Every entering event is independently known from the anchor catalogue
    # to be central cold at snap95.  Keep that endpoint consistent with C95=1.
    for tracer in range(1, 7):
        add(tracer, 95, 5, cold=True, central=True)
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
    # Resolved stars/winds/satellites are other, not falsely marked as missing.
    other_entry = next(entry for entry in ledger if entry["TracerID"] == 6)
    assert other_entry["missing_mask"]["92"] is False
    entering_entry = next(entry for entry in ledger if entry["event"] == "in")
    exiting_entry = next(entry for entry in ledger if entry["event"] == "out")
    assert tuple(map(int, entering_entry["state_sequence"])) == tuple(range(90, 96))
    assert tuple(map(int, exiting_entry["state_sequence"])) == tuple(range(94, 100))

    # The production pipeline keeps both lookup products in compact sorted
    # arrays.  It must produce exactly the same physical classification as the
    # legacy dictionary representation retained for direct callers.
    columnar_maps = {}
    columnar_records = {}
    for snap in config.snapshots:
        tracer_ids = np.asarray(sorted(parent_maps[snap]), dtype=np.uint64)
        columnar_maps[snap] = {
            "tracer_ids": tracer_ids,
            "parent_ids": np.asarray([parent_maps[snap][int(value)] for value in tracer_ids], dtype=np.uint64),
        }
        particle_ids = np.asarray(sorted(records[snap]), dtype=np.uint64)
        rows = [records[snap][int(value)] for value in particle_ids]
        columnar_records[snap] = {
            "particle_ids": particle_ids,
            "particle_type": np.asarray([row["particle_type"] for row in rows], dtype=np.int8),
            "particle_index": np.arange(len(rows), dtype=np.uint64),
            "bound_subhalo_id": np.full(len(rows), -1, dtype=np.int64),
            "bound_group_id": np.full(len(rows), -1, dtype=np.int64),
            "coordinates": np.asarray([row["coordinates"] for row in rows], dtype=float),
            "sfr": np.asarray([row["sfr"] for row in rows], dtype=float),
            "internal_energy": np.asarray([row["internal_energy"] for row in rows], dtype=float),
            "electron_abundance": np.asarray([row["electron_abundance"] for row in rows], dtype=float),
            "formation_time": np.asarray([row["formation_time"] for row in rows], dtype=float),
        }
    columnar_result, columnar_ledger = classify_halo_events(
        np.array([1, 2, 3, 4, 5, 6], dtype=np.uint64),
        np.array([7, 8], dtype=np.uint64),
        parent_maps=columnar_maps,
        records=columnar_records,
        states=states,
        headers=headers,
        config=config,
        dt_gyr=0.134,
    )
    assert columnar_result["mass_in_closure_error_msun"] == result["mass_in_closure_error_msun"]
    assert columnar_result["mass_out_closure_error_msun"] == result["mass_out_closure_error_msun"]
    assert [
        (entry["TracerID"], entry["rate_class"], entry["other_reason"])
        for entry in columnar_ledger
    ] == [
        (entry["TracerID"], entry["rate_class"], entry["other_reason"])
        for entry in ledger
    ]


def test_parent_products_use_directional_history_windows(monkeypatch, tmp_path: Path) -> None:
    """Only tracers needed by each scientific history window are resolved."""

    calls = {}

    def fake_worker(base_path, snap, all_events, cache_dir, block_size, max_workers, parallel_backend, verbose):
        calls[snap] = np.asarray(all_events, dtype=np.uint64)
        return snap, {"tracer_ids": calls[snap], "parent_ids": calls[snap]}, {"particle_ids": calls[snap]}

    monkeypatch.setattr(pipeline_module, "_parent_product_worker", fake_worker)
    config = Phase21Config()
    pipeline_module._parent_products(
        [np.array([3, 1], dtype=np.uint64)],
        [np.array([4, 3], dtype=np.uint64)],
        config,
        tmp_path,
        verbose=False,
    )
    for snap in range(90, 94):
        np.testing.assert_array_equal(calls[snap], np.array([1, 3], dtype=np.uint64))
    for snap in (94, 95):
        np.testing.assert_array_equal(calls[snap], np.array([1, 3, 4], dtype=np.uint64))
    for snap in range(96, 100):
        np.testing.assert_array_equal(calls[snap], np.array([3, 4], dtype=np.uint64))


def test_event_hdf5_derives_endpoint_views_from_sequence(tmp_path: Path) -> None:
    """Removing duplicate in-memory states must not change the HDF5 contract."""

    import h5py

    path = tmp_path / "events.h5"
    sequence = {
        "94": {"phase_state": "hot", "resolved": True},
        "95": {"phase_state": "cold", "resolved": True},
    }
    _save_events(
        path,
        [[{
            "TracerID": np.uint64(2**63 + 7),
            "event": "in",
            "anchor_snapshot": 94,
            "rate_class": "single-in",
            "other_reason": "",
            "weight_msun": 1.0,
            "state_sequence": sequence,
            "missing_mask": {"94": False, "95": False},
        }]],
    )
    with h5py.File(path, "r") as handle:
        group = handle["halo_000000"]
        assert int(group["TracerID"][0]) == 2**63 + 7
        assert json.loads(group["anchor_state"][0])["phase_state"] == "hot"
        assert json.loads(group["source_state"][0])["phase_state"] == "hot"
        assert json.loads(group["target_state"][0])["phase_state"] == "cold"


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
    assert np.isneginf(compute_agn_strength(m_hot_msun=1.0e10, m_bh_msun=0.0, r200c_pkpc=100.0, v200c_km_s=200.0)["agn_strength"])
    assert np.isnan(compute_agn_strength(m_hot_msun=0.0, m_bh_msun=1.0e8, r200c_pkpc=100.0, v200c_km_s=200.0)["agn_strength"])

    config = Phase21Config()
    means = _mean_state_inputs(
        {
            94: {"mstar_msun": 1.0e10, "m_hot_msun": 2.0e10, "m_cgm_msun": 3.0e10, "m200c_msun": 4.0e11},
            95: {"mstar_msun": 3.0e10, "m_hot_msun": 4.0e10, "m_cgm_msun": 5.0e10, "m200c_msun": 8.0e11},
        },
        config,
    )
    assert means == {
        "mstar_mean_msun": 2.0e10,
        "m_hot_mean_msun": 3.0e10,
        "m_cgm_mean_msun": 4.0e10,
        "m200c_mean_msun": 6.0e11,
    }


def test_agn_quartiles_keep_zero_bh_and_stable_ties() -> None:
    """-inf remains Q1; NaN is excluded; SubfindID resolves equal values."""

    config = Phase21Config()
    records = [
        {"mass_bin_index": 0, "agn_strength": -np.inf, "subfind_id_z99": 40},
        {"mass_bin_index": 0, "agn_strength": 1.0, "subfind_id_z99": 20},
        {"mass_bin_index": 0, "agn_strength": 1.0, "subfind_id_z99": 10},
        {"mass_bin_index": 0, "agn_strength": np.nan, "subfind_id_z99": 5},
        {"mass_bin_index": 0, "agn_strength": 2.0, "subfind_id_z99": 30},
    ]
    _assign_agn_quartiles(records, config)
    by_subfind = {record["subfind_id_z99"]: record["agn_quartile"] for record in records}
    assert by_subfind == {40: 1, 10: 2, 20: 3, 30: 4, 5: 0}


def test_radial_priority_and_exact_other_halo_binding() -> None:
    """Subhalo-centered inner wins, while offset binding finds other halos."""

    config = Phase21Config()
    state = {
        "group_id": 4,
        "subfind_id": 10,
        "group_center_ckpc_h": np.array([500.0, 0.0, 0.0]),
        "subhalo_center_ckpc_h": np.zeros(3),
        "r200c_ckpc_h": 100.0,
    }
    base_record = {
        "particle_id": 1,
        "particle_type": 0,
        "coordinates": np.array([5.0, 0.0, 0.0]),
        "sfr": 1.0,
        "internal_energy": 100.0,
        "electron_abundance": 0.1,
        "formation_time": np.nan,
    }
    tracked = classify_parent_state(
        {**base_record, "bound_subhalo_id": 10, "bound_group_id": 4},
        state,
        {"BoxSize": 1000.0},
        config,
    )
    assert tracked["radial_state"] == "inner"
    assert tracked["host_state"] == "main_central"
    assert tracked["central_cold"]
    other = classify_parent_state(
        {**base_record, "bound_subhalo_id": 99, "bound_group_id": 8},
        state,
        {"BoxSize": 1000.0},
        config,
    )
    assert other["host_state"] == "other_halo"
    assert not other["valid"]
    assert other["reason"] == "other_halo"


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
        "rate_total_in_msun_per_yr": np.full(4, 4.0),
        "rate_total_out_msun_per_yr": np.full(4, 3.0),
        "n_total_in": np.full(4, 4),
        "n_total_out": np.full(4, 3),
    }
    composition = compute_composition_statistics(results, config)
    np.testing.assert_allclose(np.nansum(composition["in_component_fractions"][:, :2], axis=0), 1.0)
    np.testing.assert_allclose(np.nansum(composition["out_component_fractions"][:, :2], axis=0), 1.0)
    agn = compute_agn_quartile_statistics(results, config)
    assert agn["n_finite"][0, 0] == 1
    assert agn["n_assigned"][0, 0] == 1

    normalization_input = {
        **results,
        "m_hot_mean_msun": np.array([1.0, 2.0, 4.0, 8.0]) * 1.0e9,
        "mstar_mean_msun": np.array([2.0, 4.0, 8.0, 16.0]) * 1.0e9,
        "rate_pf_in_msun_per_yr": np.ones(4),
        "rate_cool_iso_msun_per_yr": np.full(4, 2.0),
    }
    normalized = normalize_halo_rates(normalization_input, "mhot")
    np.testing.assert_allclose(normalized["rate_pf_in_msun_per_yr"], [1.0, 0.5, 0.25, 0.125])


def test_readme_plot_layouts() -> None:
    """The four README products include separate dispersion subpanels."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = Phase21Config()
    n_bins = len(config.mass_bin_centers)
    agn = {
        "mass_bin_center": config.mass_bin_centers,
        "median": np.ones((4, n_bins)),
        "p16": np.full((4, n_bins), 0.8),
        "p84": np.full((4, n_bins), 1.2),
        "n_finite": np.ones((4, n_bins), dtype=int),
    }
    halo = {
        "mass_bin_index": np.array([0]),
        "agn_quartile": np.array([1]),
        "x_cool_mean": np.array([1.0]),
        "m_hot_mean_msun": np.array([1.0e10]),
        "mstar_mean_msun": np.array([2.0e10]),
        "rate_pf_in_msun_per_yr": np.array([1.0]),
        "rate_cool_iso_msun_per_yr": np.array([2.0]),
    }
    normalized = {}
    for name in ("mhot", "mstar"):
        normalized[name] = {
            "mass_bin_center": config.mass_bin_centers,
            "pf_in_median": np.ones(n_bins),
            "pf_in_p16": np.full(n_bins, 0.8),
            "pf_in_p84": np.full(n_bins, 1.2),
            "cool_iso_median": np.full(n_bins, 2.0),
            "cool_iso_p16": np.full(n_bins, 1.8),
            "cool_iso_p84": np.full(n_bins, 2.2),
        }
    composition = {
        "mass_bin_center": config.mass_bin_centers,
        "in_component_names": np.array(["a", "b", "c", "other"]),
        "out_component_names": np.array(["a", "b", "other"]),
        "in_component_fractions": np.full((4, n_bins), 0.25),
        "out_component_fractions": np.full((3, n_bins), 1.0 / 3.0),
        "in_event_count": np.ones(n_bins, dtype=int),
        "out_event_count": np.ones(n_bins, dtype=int),
    }
    figures = [
        plot_agn_cooling(agn, halo_results=halo),
        plot_composition(composition, direction="in"),
        plot_composition(composition, direction="out"),
        plot_feedback_before_accretion(normalized, halo_results=halo),
    ]
    assert [len(figure.axes) for figure in figures] == [2, 1, 1, 4]
    assert len(figures[0].axes[1].collections) == 4
    assert len(figures[3].axes[2].collections) == 2
    assert len(figures[3].axes[3].collections) == 2
    for figure in figures:
        plt.close(figure)


if __name__ == "__main__":
    test_config_uses_merged_final_bin_and_seed()
    test_sampling_is_stable_after_subfind_sort()
    test_event_classes_are_complete_and_close()
    test_agn_average_physical_inputs_and_cooling()
    test_composition_and_quartile_statistics()
    print("Phase 2.1 synthetic tests passed")
