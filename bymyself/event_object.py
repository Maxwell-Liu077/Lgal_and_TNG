"""Select main-branch stellar major mergers and follow their cooling-radius histories."""

from pathlib import Path
import argparse

import h5py
import illustris_python as il
import numpy as np

import cooling_radius as cr
import real_cooling_radius as rcr
import real_physical_properties_all as rppa


data_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")
default_base_path = "/public/share/chenhouzun/TNG50-1/output/"
seconds_per_gyr = 1.0e9 * 365.25 * 86400.0


def _write_arrays(group, arrays):
    for name, values in arrays.items():
        values = np.asarray(values)
        options = {"compression": "gzip", "compression_opts": 4} if values.ndim and values.size else {}
        group.create_dataset(name, data=values, **options)


def _read_sample(tree_file, snap_min, snap_max):
    with h5py.File(tree_file, "r") as f:
        start_ids = np.asarray(f["start_subhalo_id"][:], dtype=np.int64)
        available_snaps = np.asarray(f["snap_numbers"][:], dtype=np.int64)
        node_ids = np.asarray(f["mpb_subhalo_ids"][:], dtype=np.int64)
        fof_ids = np.asarray(f["mpb_fof_ids"][:], dtype=np.int64)

    snaps = np.arange(snap_min, snap_max + 1, dtype=np.int64)
    columns = []
    for snap in snaps:
        matches = np.flatnonzero(available_snaps == snap)
        columns.append(int(matches[0]))
    return start_ids, snaps, node_ids[:, columns], fof_ids[:, columns]


def _cosmic_age_gyr(header):
    """Cosmic age in a flat matter+Lambda cosmology using the snapshot header."""
    a, h = float(header["Time"]), float(header["HubbleParam"])
    omega_m, omega_l = float(header["Omega0"]), float(header["OmegaLambda"])
    if a <= 0.0 or h <= 0.0 or omega_m <= 0.0 or omega_l <= 0.0 or not np.isclose(omega_m + omega_l, 1.0, atol=1.0e-3):
        raise ValueError("Cosmic ages require a positive scale factor and a flat matter+Lambda cosmology.")
    h0_per_s = 100.0 * h / (rppa.kpc_cm / 1.0e5 * 1000.0)
    return 2.0 * np.arcsinh(np.sqrt(omega_l / omega_m) * a ** 1.5) / (3.0 * h0_per_s * np.sqrt(omega_l) * seconds_per_gyr)


def _branch_rows(tree, first_row, id_to_row):
    rows = []
    row = int(first_row)
    while True:
        rows.append(row)
        progenitor = int(tree["FirstProgenitorID"][row])
        if progenitor == -1:
            return rows
        if progenitor not in id_to_row:
            raise ValueError(f"Incomplete SubLink tree: progenitor {progenitor} is missing.")
        next_row = id_to_row[progenitor]
        if tree["SnapNum"][next_row] >= tree["SnapNum"][row]:
            raise ValueError("FirstProgenitorID must point to an earlier snapshot.")
        row = next_row


def _scan_mergers(tree, tree_row, start_id, snap_min, snap_max, mass_factor, min_mass_ratio, expected_nodes):
    ids = np.asarray(tree["SubhaloID"], dtype=np.int64)
    if ids.size == 0 or np.unique(ids).size != ids.size:
        raise ValueError("SubLink tree has empty or duplicate internal node IDs.")
    id_to_row = {int(node): row for row, node in enumerate(ids)}
    main_rows = _branch_rows(tree, 0, id_to_row)
    main_by_snap = {int(tree["SnapNum"][row]): row for row in main_rows}
    for snap, node_id in expected_nodes.items():
        row = main_by_snap.get(snap)
        if row is None or int(tree["SubfindID"][row]) != node_id:
            raise ValueError(f"SubLink MPB and saved tree disagree for start_id={start_id}, snap={snap}.")
    records = []
    for descendant_row in main_rows:
        snap = int(tree["SnapNum"][descendant_row])
        if not snap_min <= snap <= snap_max:
            continue
        primary_id = int(tree["FirstProgenitorID"][descendant_row])
        if primary_id == -1:
            continue
        primary_row = id_to_row[primary_id]
        secondary_id = int(tree["NextProgenitorID"][primary_row])
        seen = set()
        while secondary_id != -1:
            if secondary_id in seen or secondary_id not in id_to_row:
                raise ValueError("NextProgenitorID chain is cyclic or incomplete.")
            seen.add(secondary_id)
            secondary_row = id_to_row[secondary_id]
            for row in (primary_row, secondary_row):
                if int(tree["DescendantID"][row]) != int(ids[descendant_row]):
                    raise ValueError("Merger progenitors do not share the expected descendant.")
            branch = np.asarray(_branch_rows(tree, secondary_row, id_to_row), dtype=np.int64)
            masses = np.asarray(tree["SubhaloMassType"][branch, 4], dtype=np.float64)
            positive = np.flatnonzero(np.isfinite(masses) & (masses > 0.0))
            record = {"tree_row": tree_row, "start_subhalo_id": start_id, "merger_snap": snap,
                      "descendant_subhalo_id": int(tree["SubfindID"][descendant_row]), "descendant_tree_id": int(ids[descendant_row]),
                      "primary_snap": int(tree["SnapNum"][primary_row]), "primary_subhalo_id": int(tree["SubfindID"][primary_row]),
                      "secondary_snap": int(tree["SnapNum"][secondary_row]), "secondary_subhalo_id": int(tree["SubfindID"][secondary_row]),
                      "primary_tree_id": primary_id, "secondary_tree_id": secondary_id, "mass_ratio_snap": -1,
                      "primary_mass_subhalo_id": -1, "secondary_mass_subhalo_id": -1,
                      "primary_stellar_mass_msun": np.nan, "secondary_stellar_mass_msun": np.nan,
                      "mass_ratio": np.nan, "mass_ratio_status": 2, "is_major": False}
            if positive.size:
                peak_row = int(branch[positive[np.argmax(masses[positive])]])
                peak_snap = int(tree["SnapNum"][peak_row])
                record["mass_ratio_snap"] = peak_snap
                record["secondary_mass_subhalo_id"] = int(tree["SubfindID"][peak_row])
                record["secondary_stellar_mass_msun"] = float(tree["SubhaloMassType"][peak_row, 4]) * mass_factor
                primary_peak_row = main_by_snap.get(peak_snap)
                record["mass_ratio_status"] = 3
                if primary_peak_row is not None:
                    primary_mass = float(tree["SubhaloMassType"][primary_peak_row, 4])
                    record["primary_mass_subhalo_id"] = int(tree["SubfindID"][primary_peak_row])
                    record["primary_stellar_mass_msun"] = primary_mass * mass_factor
                    record["mass_ratio_status"] = 4
                    if np.isfinite(primary_mass) and primary_mass > 0.0:
                        secondary_mass = float(tree["SubhaloMassType"][peak_row, 4])
                        ratio = min(primary_mass, secondary_mass) / max(primary_mass, secondary_mass)
                        record.update(mass_ratio=ratio, mass_ratio_status=1, is_major=ratio >= min_mass_ratio)
            records.append(record)
            secondary_id = int(tree["NextProgenitorID"][secondary_row])
    return sorted(records, key=lambda record: (record["merger_snap"], record["secondary_tree_id"]))


def select_event_halo(basePath=default_base_path, snap_min=40, snap_max=99, min_mass_ratio=0.25, tree_file=None, output_file=None, data_path=None, root_snap=99, tree_name="SubLink"):
    """Select MPB stellar major mergers, using both masses at the secondary stellar-mass maximum.

    Missing primary nodes at that exact snapshot are flagged without mass interpolation.
    Returns (catalogue_path, selected_start_ids, selected_major_merger_counts).
    """
    if not 0 <= snap_min <= snap_max <= root_snap or not 0.0 < min_mass_ratio <= 1.0:
        raise ValueError("Require 0 <= snap_min <= snap_max <= root_snap and 0 < min_mass_ratio <= 1.")
    directory = data_dir if data_path is None else Path(data_path)
    tree_file = directory / "all_selected_trees.hdf5" if tree_file is None else Path(tree_file)
    output_file = directory / "event_objects" / f"major_merger_events_{snap_min}_{snap_max}.hdf5" if output_file is None else Path(output_file)
    if output_file.resolve() == tree_file.resolve():
        raise ValueError("The event catalogue output must differ from the selected-tree input.")
    start_ids, snaps, node_ids, fof_ids = _read_sample(tree_file, snap_min, snap_max)
    _, _, root_nodes, _ = _read_sample(tree_file, root_snap, root_snap)
    if not np.array_equal(start_ids, root_nodes[:, 0]):
        raise ValueError("start_subhalo_id must identify the selected nodes at root_snap.")
    headers = {int(snap): il.groupcat.loadHeader(basePath, int(snap)) for snap in snaps}
    root_header = headers[root_snap] if root_snap in headers else il.groupcat.loadHeader(basePath, root_snap)
    mass_factor = 1.0e10 / float(root_header["HubbleParam"])
    fields = ["SubhaloID", "SubfindID", "SnapNum", "FirstProgenitorID", "NextProgenitorID", "DescendantID", "SubhaloMassType"]
    tree_available = np.zeros(start_ids.size, dtype=bool)
    merger_counts = np.zeros(start_ids.size, dtype=np.int64)
    unknown_counts = np.zeros_like(merger_counts)
    candidate_counts = np.zeros_like(merger_counts)
    records = []
    for tree_row, start_id in enumerate(start_ids):
        tree = il.sublink.loadTree(basePath, int(root_snap), int(start_id), fields=fields, onlyMPB=False, treeName=tree_name)
        if tree is None or tree.get("count", 0) == 0:
            print(f"No SubLink tree for start_id={start_id}; selection remains unknown.", flush=True)
            continue
        if int(tree["SnapNum"][0]) != root_snap or int(tree["SubfindID"][0]) != start_id:
            raise ValueError("Loaded SubLink root does not match the selected z=0 object.")
        expected = {int(snap): int(node) for snap, node in zip(snaps, node_ids[tree_row]) if node >= 0}
        local_records = _scan_mergers(tree, tree_row, int(start_id), snap_min, snap_max, mass_factor, min_mass_ratio, expected)
        tree_available[tree_row] = True
        candidate_counts[tree_row] = len(local_records)
        merger_counts[tree_row] = sum(record["is_major"] for record in local_records)
        unknown_counts[tree_row] = sum(record["mass_ratio_status"] != 1 for record in local_records)
        for record in local_records:
            record["merger_time_Gyr"] = _cosmic_age_gyr(headers[record["merger_snap"]])
            record["merger_redshift"] = float(headers[record["merger_snap"]]["Redshift"])
        records.extend(local_records)
        if (tree_row + 1) % 100 == 0:
            print(f"Scanned merger trees: {tree_row + 1}/{start_ids.size}", flush=True)
    selected_rows = np.flatnonzero(merger_counts > 0)
    object_index = {int(row): index for index, row in enumerate(selected_rows)}
    integer_fields = ("tree_row", "start_subhalo_id", "merger_snap", "descendant_subhalo_id", "descendant_tree_id", "primary_snap", "primary_subhalo_id", "secondary_snap", "secondary_subhalo_id", "primary_tree_id", "secondary_tree_id", "mass_ratio_snap", "primary_mass_subhalo_id", "secondary_mass_subhalo_id")
    float_fields = ("primary_stellar_mass_msun", "secondary_stellar_mass_msun", "mass_ratio", "merger_time_Gyr", "merger_redshift")
    candidates = {key: np.asarray([record[key] for record in records], dtype=np.int64) for key in integer_fields}
    candidates.update({key: np.asarray([record[key] for record in records], dtype=np.float64) for key in float_fields})
    candidates["mass_ratio_status"] = np.asarray([record["mass_ratio_status"] for record in records], dtype=np.uint8)
    candidates["is_major"] = np.asarray([record["is_major"] for record in records], dtype=bool)
    candidates["object_index"] = np.asarray([object_index.get(record["tree_row"], -1) for record in records], dtype=np.int64)
    major = candidates["is_major"]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_file, "w") as f:
        f.attrs.update(snap_min=int(snap_min), snap_max=int(snap_max), root_snap=int(root_snap), min_mass_ratio=float(min_mass_ratio), tree_name=tree_name, tree_file=str(tree_file.resolve()), base_path=str(basePath))
        f.attrs["mass_ratio_definition"] = "min/max of total stellar masses at the secondary MPB stellar-mass maximum; primary matched at the same exact SnapNum"
        f.attrs["merger_time_definition"] = "snapshot of the common descendant on the selected object's main branch"
        f.attrs["mass_ratio_status_codes"] = "1=measured; 2=no positive secondary stellar mass; 3=primary absent at peak snapshot; 4=invalid primary stellar mass"
        _write_arrays(f, {"tree_row": selected_rows, "start_subhalo_id": start_ids[selected_rows], "snap_numbers": snaps, "mpb_subhalo_ids": node_ids[selected_rows], "mpb_fof_ids": fof_ids[selected_rows], "major_merger_count": merger_counts[selected_rows], "selection_complete": (tree_available & (unknown_counts == 0))[selected_rows], "event_offsets": np.r_[0, np.cumsum(merger_counts[selected_rows])]})
        _write_arrays(f.create_group("sample"), {"start_subhalo_id": start_ids, "tree_available": tree_available, "candidate_count": candidate_counts, "major_merger_count": merger_counts, "unresolved_count": unknown_counts, "selection_complete": tree_available & (unknown_counts == 0)})
        _write_arrays(f.create_group("candidates"), candidates)
        _write_arrays(f.create_group("events"), {key: values[major] for key, values in candidates.items()})
    print(f"Selected {selected_rows.size}/{start_ids.size} hosts with {np.count_nonzero(major)} major mergers -> {output_file}", flush=True)
    return output_file, start_ids[selected_rows], merger_counts[selected_rows]


def _take_rows(dataset, rows):
    if rows.size == 0:
        return np.empty((0,) + dataset.shape[1:], dtype=dataset.dtype)
    order = np.argsort(rows)
    return np.asarray(dataset[rows[order]])[np.argsort(order)]


def _cache_rows(f, selected):
    if not all(name in f for name in ("start_subhalo_id", "subhalo_id", "fof_id")):
        return None
    source_ids = np.asarray(f["start_subhalo_id"][:], dtype=np.int64)
    if source_ids.ndim != 1 or np.unique(source_ids).size != source_ids.size:
        raise ValueError(f"Duplicate or invalid start_subhalo_id in {f.filename}")
    lookup = {int(start_id): row for row, start_id in enumerate(source_ids)}
    rows = np.asarray([lookup.get(int(start_id), -1) for start_id in selected["start_subhalo_id"]], dtype=np.int64)
    if np.any(rows < 0):
        return None
    for name in ("subhalo_id", "fof_id"):
        if not np.array_equal(_take_rows(f[name], rows), selected[name]):
            raise ValueError(f"Cached snapshot nodes disagree with selected trees: {f.filename}, {name}")
    return rows


def _read_cache(path, selected, names):
    if not path.is_file():
        return None
    with h5py.File(path, "r") as f:
        rows = _cache_rows(f, selected)
        if rows is None or not all(name in f for name in names):
            return None
        return {name: _take_rows(f[name], rows) for name in names}


def _subset_properties(source, target, selected, snap):
    if not source.is_file():
        return False
    required = ("analysis_valid", "m200c_msun", "r200c_physical_kpc", "interpolation_radii_r200c", "temperature_mass_weighted_K_interpolated", "rho_hot_cell_mean_g_cm3_interpolated", "metallicity_mass_weighted_interpolated", "hot_gas_mass_msun_within_r200c", "global_temperature_mass_weighted_K", "global_metallicity_mass_weighted")
    shared = {"radial_bin_edges_r200c", "radial_bin_centers_r200c", "interpolation_radii_r200c"}
    temporary = target.with_suffix(".partial.hdf5")
    with h5py.File(source, "r") as original:
        rows = _cache_rows(original, selected)
        if rows is None or not all(name in original for name in required):
            return False
        if source == target and np.array_equal(rows, np.arange(original["start_subhalo_id"].size)):
            return True
        target.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(temporary, "w") as result:
            result.attrs.update(dict(original.attrs))
            result.attrs.update(snap=int(snap), scale_factor=float(selected["scale_factor"]), hubble_param=float(selected["hubble_param"]))
            for name, dataset in original.items():
                values = dataset[:] if name in shared else _take_rows(dataset, rows)
                if name == "tree_row":
                    values = selected["tree_row"]
                _write_arrays(result, {name: values})
    temporary.replace(target)
    return True


def _lgal_from_properties(properties_file, selected, header, cooling_tables):
    names = ("analysis_valid", "m200c_msun", "r200c_physical_kpc", "hot_gas_mass_msun_within_r200c", "global_temperature_mass_weighted_K", "global_metallicity_mass_weighted")
    values = _read_cache(properties_file, selected, names)
    if values is None:
        raise ValueError(f"Properties file does not cover the selected objects: {properties_file}")
    mass, radius = values["m200c_msun"], values["r200c_physical_kpc"]
    hot_mass = values["hot_gas_mass_msun_within_r200c"] * rppa.msun_g
    temperature, metallicity = values["global_temperature_mass_weighted_K"], values["global_metallicity_mass_weighted"]
    valid = selected["analysis_valid"] & values["analysis_valid"].astype(bool) & np.isfinite(mass) & (mass > 0.0) & np.isfinite(radius) & (radius > 0.0) & np.isfinite(hot_mass) & (hot_mass > 0.0) & np.isfinite(temperature) & (temperature > 0.0) & np.isfinite(metallicity) & (metallicity >= 0.0)
    v200 = np.full(mass.shape, np.nan)
    v200[valid] = np.sqrt(rcr.G_kpc_kms2_msun * mass[valid] / radius[valid])
    radius_cm = radius * rppa.kpc_cm
    t_dyn = radius_cm / (v200 * 1.0e5)
    t200 = 35.9 * v200 ** 2
    cooling_function = np.asarray(cr.LgalCoolingFunction(temperature, metallicity, cooling_tables), dtype=np.float64)
    valid &= np.isfinite(cooling_function) & (cooling_function > 0.0)
    h, a = float(header["HubbleParam"]), float(header["Time"])
    radius_raw = radius * h / a
    cooling_radius = cr.compute_cooling_radius(hot_mass, t_dyn, cooling_function, t200, radius_cm, radius_raw, valid.astype(np.uint8), h, a)
    valid &= np.isfinite(cooling_radius) & (cooling_radius > 0.0)
    cooling_radius[~valid] = np.nan
    return {"cooling_radius": cooling_radius, "halo_flag": valid.astype(np.uint8)}


def object_cooling_radius(basePath=default_base_path, event_file=None, output_file=None, data_path=None, tree_file=None, reuse_existing=True, table_dir=None, outer_max_r200c=5.0):
    """Build (event-host, snapshot) TNG/Lgal histories using aligned caches or only the selected hosts.

    Set reuse_existing=False after physical-model changes to regenerate event-specific files.
    Returns (history_path, lgal_radius_r200c, tng_radius_r200c).
    """
    directory = data_dir if data_path is None else Path(data_path)
    event_dir = directory / "event_objects"
    event_file = event_dir / "major_merger_events_40_99.hdf5" if event_file is None else Path(event_file)
    if not event_file.is_file():
        if event_file != event_dir / "major_merger_events_40_99.hdf5":
            raise FileNotFoundError(event_file)
        select_event_halo(basePath, data_path=directory, tree_file=tree_file, output_file=event_file)
    with h5py.File(event_file, "r") as f:
        start_ids = np.asarray(f["start_subhalo_id"][:], dtype=np.int64)
        tree_rows = np.asarray(f["tree_row"][:], dtype=np.int64)
        snaps = np.asarray(f["snap_numbers"][:], dtype=np.int64)
        node_ids = np.asarray(f["mpb_subhalo_ids"][:], dtype=np.int64)
        fof_ids = np.asarray(f["mpb_fof_ids"][:], dtype=np.int64)
        saved_tree_file = Path(f.attrs["tree_file"])
    tree_file = saved_tree_file if tree_file is None else Path(tree_file)
    all_ids, current_snaps, current_nodes, current_fofs = _read_sample(tree_file, int(snaps[0]), int(snaps[-1]))
    if np.any((tree_rows < 0) | (tree_rows >= all_ids.size)) or not np.array_equal(all_ids[tree_rows], start_ids) or not np.array_equal(current_snaps, snaps) or not np.array_equal(current_nodes[tree_rows], node_ids) or not np.array_equal(current_fofs[tree_rows], fof_ids):
        raise ValueError("Event catalogue and selected-tree file describe different rows or snapshot nodes.")
    if start_ids.size == 0:
        raise ValueError("No major-merger hosts were selected; inspect the event catalogue's sample flags.")
    output_file = event_dir / f"event_cooling_radius_history_{snaps[0]}_{snaps[-1]}.hdf5" if output_file is None else Path(output_file)
    if output_file.resolve() in (event_file.resolve(), tree_file.resolve()):
        raise ValueError("The history output must differ from its event and selected-tree inputs.")
    shape = (start_ids.size, snaps.size)
    lgal_radius, tng_radius = np.full(shape, np.nan), np.full(shape, np.nan)
    halo_mass, r200_kpc = np.full(shape, np.nan), np.full(shape, np.nan)
    analysis_valid, lgal_valid, tng_valid = np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)
    tng_status = np.zeros(shape, dtype=np.uint8)
    redshifts, cosmic_times = np.full(snaps.size, np.nan), np.full(snaps.size, np.nan)
    lgal_sources, tng_sources = [], []
    cooling_tables = None
    for column, snap in enumerate(snaps):
        snap = int(snap)
        header = il.groupcat.loadHeader(basePath, snap)
        redshifts[column], cosmic_times[column] = float(header["Redshift"]), _cosmic_age_gyr(header)
        selected = rppa.select_objects(basePath, snap, tree_rows=tree_rows, tree_file=tree_file)
        halo_mass[:, column] = selected["m200c_msun"]
        r200_kpc[:, column] = selected["r200c_raw"] * float(header["Time"]) / float(header["HubbleParam"])
        analysis_valid[:, column] = selected["analysis_valid"]
        event_properties = event_dir / f"real_physical_properties_events_{snap}.hdf5"
        event_real = event_dir / f"real_cooling_radius_events_{snap}.hdf5"
        event_lgal = event_dir / f"lgal_cooling_radius_events_{snap}.hdf5"
        real_values, lgal_values = None, None
        real_source, lgal_source = event_real, event_lgal
        if reuse_existing:
            for path in (directory / f"real_cooling_radius_{snap}.hdf5", event_real):
                real_values = _read_cache(path, selected, ("cooling_radius_r200c", "analysis_valid", "status"))
                if real_values is not None:
                    real_source = path
                    break
            global_lgal = directory / f"cooling_radius_{snap}.hdf5"
            if global_lgal.is_file():
                with h5py.File(global_lgal, "r") as f:
                    radii = np.asarray(f["cooling_radius"][:], dtype=np.float64)
                    flags = np.asarray(f["halo_flag"][:], dtype=bool)
                ids = selected["fof_id"]
                usable = (ids >= 0) & (ids < radii.size)
                if flags.shape != radii.shape:
                    raise ValueError(f"Analytic cooling arrays have different shapes: {global_lgal}")
                if np.any(selected["analysis_valid"] & ~usable):
                    raise ValueError(f"Analytic cooling file does not cover the selected FoF IDs: {global_lgal}")
                mapped = np.full(start_ids.size, np.nan)
                mapped_flags = np.zeros(start_ids.size, dtype=bool)
                mapped[usable], mapped_flags[usable] = radii[ids[usable]], flags[ids[usable]]
                lgal_values = {"cooling_radius": mapped, "halo_flag": mapped_flags}
                lgal_source = global_lgal
            else:
                lgal_values = _read_cache(event_lgal, selected, ("cooling_radius", "halo_flag"))
        if real_values is None or lgal_values is None:
            profiles_ready = False
            if reuse_existing:
                for source in (event_properties, directory / f"real_physical_properties_all_{snap}.hdf5"):
                    if _subset_properties(source, event_properties, selected, snap):
                        profiles_ready = True
                        break
            if not profiles_ready:
                rppa.real_physical_properties(basePath, snap, tree_rows=tree_rows, tree_file=tree_file, output_file=event_properties)
            if real_values is None:
                rcr.real_cooling_radius(snap, properties_file=event_properties, output_file=event_real, table_dir=table_dir, outer_max_r200c=outer_max_r200c)
                real_values = _read_cache(event_real, selected, ("cooling_radius_r200c", "analysis_valid", "status"))
            if lgal_values is None:
                if cooling_tables is None:
                    cooling_tables = cr.CoolingTableLoader(table_dir=table_dir)
                lgal_values = _lgal_from_properties(event_properties, selected, header, cooling_tables)
                with h5py.File(event_lgal, "w") as f:
                    f.attrs.update(snap=snap, scale_factor=float(header["Time"]), hubble_param=float(header["HubbleParam"]), radius_unit="R200c", properties_file=str(event_properties))
                    _write_arrays(f, {**{name: selected[name] for name in ("tree_row", "start_subhalo_id", "subhalo_id", "fof_id")}, **lgal_values})
        lgal_valid[:, column] = selected["analysis_valid"] & np.asarray(lgal_values["halo_flag"], dtype=bool) & np.isfinite(lgal_values["cooling_radius"]) & (lgal_values["cooling_radius"] > 0.0)
        tng_status[:, column] = real_values["status"]
        tng_valid[:, column] = selected["analysis_valid"] & np.asarray(real_values["analysis_valid"], dtype=bool) & (tng_status[:, column] == rcr.status_root_found) & np.isfinite(real_values["cooling_radius_r200c"]) & (real_values["cooling_radius_r200c"] > 0.0)
        lgal_radius[lgal_valid[:, column], column] = lgal_values["cooling_radius"][lgal_valid[:, column]]
        tng_radius[tng_valid[:, column], column] = real_values["cooling_radius_r200c"][tng_valid[:, column]]
        lgal_sources.append(str(lgal_source))
        tng_sources.append(str(real_source))
        print(f"Snap {snap}: Lgal {np.count_nonzero(lgal_valid[:, column])}/{start_ids.size}, TNG roots {np.count_nonzero(tng_valid[:, column])}/{start_ids.size}", flush=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_file, "w") as f:
        f.attrs.update(event_file=str(event_file.resolve()), tree_file=str(tree_file.resolve()), radius_unit="R200c", axis_order="object, snapshot", cosmic_time_definition="flat matter+Lambda age, snapshot-header cosmology")
        _write_arrays(f, {"tree_row": tree_rows, "start_subhalo_id": start_ids, "snap_numbers": snaps, "redshift": redshifts, "cosmic_time_Gyr": cosmic_times, "mpb_subhalo_ids": node_ids, "mpb_fof_ids": fof_ids, "m200c_msun": halo_mass, "r200c_physical_kpc": r200_kpc, "analysis_valid": analysis_valid, "lgal_cooling_radius_r200c": lgal_radius, "tng_cooling_radius_r200c": tng_radius, "lgal_valid": lgal_valid, "tng_valid": tng_valid, "tng_status": tng_status})
        f.create_dataset("lgal_source_files", data=np.asarray(lgal_sources, dtype=h5py.string_dtype("utf-8")))
        f.create_dataset("tng_source_files", data=np.asarray(tng_sources, dtype=h5py.string_dtype("utf-8")))
        with h5py.File(event_file, "r") as catalog:
            for name in ("major_merger_count", "selection_complete", "event_offsets", "events"):
                catalog.copy(name, f)
    print(f"Saved event-host cooling-radius histories -> {output_file}", flush=True)
    return output_file, lgal_radius, tng_radius


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", default="all", choices=("select", "cooling", "all"))
    parser.add_argument("--base-path", default=default_base_path)
    parser.add_argument("--data-dir", type=Path, default=data_dir)
    parser.add_argument("--snap-min", type=int, default=40)
    parser.add_argument("--snap-max", type=int, default=99)
    parser.add_argument("--min-mass-ratio", type=float, default=0.25)
    parser.add_argument("--tree-name", default="SubLink", choices=("SubLink", "SubLink_gal"))
    parser.add_argument("--recompute", action="store_true", help="Regenerate event-host cooling files instead of reusing snapshot caches.")
    args = parser.parse_args()
    catalog_file = args.data_dir / "event_objects" / f"major_merger_events_{args.snap_min}_{args.snap_max}.hdf5"
    if args.mode in ("select", "all"):
        select_event_halo(args.base_path, snap_min=args.snap_min, snap_max=args.snap_max, min_mass_ratio=args.min_mass_ratio, tree_name=args.tree_name, output_file=catalog_file, data_path=args.data_dir)
    if args.mode in ("cooling", "all"):
        object_cooling_radius(args.base_path, event_file=catalog_file, data_path=args.data_dir, reuse_existing=not args.recompute)
