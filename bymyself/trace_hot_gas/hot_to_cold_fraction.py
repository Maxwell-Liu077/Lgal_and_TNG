"""Hot-to-cold tracer fractions in initial radial shells of selected MW-like halos."""

from pathlib import Path

import h5py
import illustris_python as il
import numpy as np
import funcs
from real_physical_properties import num_radial_bins, radial_min, radial_max

data_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")
mw_log_m200c_min = 11.8
mw_log_m200c_max = 12.2


def calculate_fraction_in_bins(radius_r200c, num_tracers_in_parents, hot_to_cold, unknown, radial_edges):
    """Count tracers using their initial parent radius; keep all initial tracers in the denominator."""
    radius_r200c = np.asarray(radius_r200c, dtype=np.float64)
    counts = np.asarray(num_tracers_in_parents, dtype=np.int64)
    hot_to_cold = np.asarray(hot_to_cold, dtype=bool)
    unknown = np.asarray(unknown, dtype=bool)
    radial_edges = np.asarray(radial_edges, dtype=np.float64)
    offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))

    # Prefix sums also handle parents containing zero tracers.
    cooled_prefix = np.concatenate(([0], np.cumsum(hot_to_cold, dtype=np.int64)))
    unknown_prefix = np.concatenate(([0], np.cumsum(unknown, dtype=np.int64)))
    cooled_per_parent = np.diff(cooled_prefix[offsets])
    unknown_per_parent = np.diff(unknown_prefix[offsets])
    radial_bin = np.searchsorted(radial_edges, radius_r200c, side="right") - 1
    radial_bin = np.minimum(radial_bin, radial_edges.size - 2)
    results = {name: np.zeros(radial_edges.size - 1, dtype=np.int64) for name in ("hot_tracer_count", "hot_to_cold_tracer_count", "unknown_tracer_count")}
    for name, weights in zip(results, (counts, cooled_per_parent, unknown_per_parent)):
        np.add.at(results[name], radial_bin, weights)
    for name, numerator in (("hot_to_cold_fraction", "hot_to_cold_tracer_count"), ("unknown_fraction", "unknown_tracer_count")):
        results[name] = np.full(radial_edges.size - 1, np.nan, dtype=np.float64)
        np.divide(results[numerator], results["hot_tracer_count"], out=results[name], where=results["hot_tracer_count"] > 0)
    results["bin_complete"] = (results["hot_tracer_count"] > 0) & (results["unknown_tracer_count"] == 0)
    return results


def hot_to_cold_fraction(basePath, start_snap, target_snap, output_file=None):
    """Save per-halo and pooled fractions; return (output_file, per_halo_fraction).

    Select MW-like centrals from the selected trees at start_snap. Radius is
    measured at start_snap; no target-radius or target-host-validity cut is used.
    The first original logarithmic bin is extended to zero, keeping 25 bins.
    """
    start_snap, target_snap = int(start_snap), int(target_snap)

    tree_file = data_dir / "all_selected_trees.hdf5"
    class_file = data_dir / f"gas_parent_classification_hot_{start_snap}.hdf5"
    parent_file = data_dir / f"parent_indices_hot_start_{start_snap}_target_{target_snap}.hdf5"
    distance_file = data_dir / "distance_cats" / f"distance_cats_hot_start_{start_snap}_target_{target_snap}.hdf5"
    output_file = data_dir / "distance_cats" / f"hot_to_cold_fraction_MW_start_{start_snap}_target_{target_snap}.hdf5" if output_file is None else Path(output_file)
    for input_file in (tree_file, class_file, parent_file, distance_file):
        if output_file.resolve() == input_file.resolve():
            raise ValueError("The output file must not overwrite an input file.")

    radial_edges = np.geomspace(radial_min, radial_max, num_radial_bins + 1)
    radial_edges[0] = 0.0
    radial_centers = np.sqrt(radial_edges[:-1] * radial_edges[1:])
    radial_centers[0] = 0.5 * radial_edges[1]
    with h5py.File(tree_file, "r") as f:
        start_ids = np.asarray(f["start_subhalo_id"][:], dtype=np.int64)
        snap_column = funcs.get_snap_column(start_snap, np.asarray(f["snap_numbers"][:], dtype=np.int64))
        node_ids = np.asarray(f["mpb_subhalo_ids"][:, snap_column], dtype=np.int64)
        tree_fof_ids = np.asarray(f["mpb_fof_ids"][:, snap_column], dtype=np.int64)

    header = il.groupcat.loadHeader(basePath, start_snap)
    h, scale_factor, box_size = float(header["HubbleParam"]), float(header["Time"]), float(header["BoxSize"])
    halos = il.groupcat.loadHalos(basePath, start_snap, fields=["Group_M_Crit200", "Group_R_Crit200", "GroupPos", "GroupFirstSub"])
    halo_mass_msun = np.asarray(halos["Group_M_Crit200"], dtype=np.float64) * 1.0e10 / h
    halo_r200c = np.asarray(halos["Group_R_Crit200"], dtype=np.float64)
    halo_pos = np.asarray(halos["GroupPos"], dtype=np.float64)
    first_sub = np.asarray(halos["GroupFirstSub"], dtype=np.int64)
    group_name = f"snap_{target_snap:03d}"

    with h5py.File(class_file, "r") as cf, h5py.File(parent_file, "r") as pf, h5py.File(distance_file, "r") as df:
        source_halo_ids = np.asarray(cf["halo_id"][:], dtype=np.int64)
        source_valid = np.asarray(cf["tree_valid"][:], dtype=bool)
        halo_offsets = np.asarray(cf["halo_offsets"][:], dtype=np.int64)
        tracer_offsets = np.asarray(pf["halo_tracer_offsets"][:], dtype=np.int64)
        parent_group = pf[group_name]

        valid = source_valid & (source_halo_ids >= 0) & (source_halo_ids < halo_mass_msun.size) & (node_ids >= 0) & (node_ids < int(header["Nsubgroups_Total"]))
        rows = np.flatnonzero(valid)
        hosts = source_halo_ids[rows]
        valid[rows] &= (first_sub[hosts] == node_ids[rows]) & np.isfinite(halo_mass_msun[hosts]) & (halo_mass_msun[hosts] > 0.0) & np.isfinite(halo_r200c[hosts]) & (halo_r200c[hosts] > 0.0) & np.all(np.isfinite(halo_pos[hosts]), axis=1)
        log_mass = np.full(start_ids.size, np.nan, dtype=np.float64)
        log_mass[valid] = np.log10(halo_mass_msun[source_halo_ids[valid]])
        selected_rows = np.flatnonzero(valid & (log_mass >= mw_log_m200c_min) & (log_mass < mw_log_m200c_max))
        selected_fof_ids = source_halo_ids[selected_rows]

        with h5py.File(il.groupcat.offsetPath(basePath, start_snap), "r") as f:
            gas_start_all = np.asarray(f["Group/SnapByType"][:, 0], dtype=np.int64)

        result_names = ("hot_tracer_count", "hot_to_cold_tracer_count", "unknown_tracer_count", "hot_to_cold_fraction", "unknown_fraction", "bin_complete")
        results = {name: np.zeros((selected_rows.size, num_radial_bins), dtype=np.int64 if name.endswith("count") else bool if name == "bin_complete" else np.float64) for name in result_names}
        for object_index, tree_row in enumerate(selected_rows):
            fof_id = int(source_halo_ids[tree_row])
            gas_slice = slice(halo_offsets[tree_row], halo_offsets[tree_row + 1])
            tracer_slice = slice(tracer_offsets[tree_row], tracer_offsets[tree_row + 1])
            counts = np.asarray(parent_group["numTracersInParents"][gas_slice], dtype=np.int64)

            radius_r200c = np.empty(counts.size, dtype=np.float64)
            if counts.size:
                # Read only this halo, then verify saved global gas indices with ParticleIDs.
                gas = il.snapshot.loadHalo(basePath, start_snap, fof_id, "gas", fields=["Coordinates", "ParticleIDs"])
                gas_ids = np.asarray(gas.get("ParticleIDs", []), dtype=np.uint64)
                local_indices = np.asarray(cf["gas_parent_index"][gas_slice], dtype=np.int64) - gas_start_all[fof_id]

                # The original hot-parent selection loaded Coordinates as float32.
                positions = np.asarray(gas["Coordinates"], dtype=np.float32)[local_indices]
                radius_r200c = funcs.dist_vector_nb(halo_pos[fof_id], positions, box_size) / halo_r200c[fof_id]
                del gas

            cooled = np.asarray(df["hot_to_cold"][tracer_slice], dtype=bool)
            found = np.asarray(df["tracer_found"][tracer_slice], dtype=bool)
            target_parents = np.asarray(df["parent_indices"][tracer_slice], dtype=np.int64)
            sfr = np.asarray(df["tracer_sfr_msun_per_yr"][tracer_slice], dtype=np.float64)
            temperature = np.asarray(df["tracer_temperature_K"][tracer_slice], dtype=np.float64)
            gas_parent = found & (target_parents[:, 0] >= 0) & (target_parents[:, 1] == 0)
            sf = gas_parent & np.isfinite(sfr) & (sfr > 0.0)
            non_sf = gas_parent & (sfr == 0.0) & np.isfinite(temperature) & (temperature > 0.0)
            other_parent = found & (target_parents[:, 0] >= 0) & np.isin(target_parents[:, 1], [1, 2])
            unknown = ~(sf | non_sf | other_parent)
            halo_result = calculate_fraction_in_bins(radius_r200c, counts, cooled, unknown, radial_edges)
            for name in result_names:
                results[name][object_index] = halo_result[name]
            if (object_index + 1) % 100 == 0 or object_index + 1 == selected_rows.size:
                print(f"Binned {object_index + 1}/{selected_rows.size} MW-like halos.", flush=True)
        target_tree_valid = np.asarray(df["tree_valid"][:], dtype=bool)[selected_rows]

    pooled = {name: results[name].sum(axis=0) for name in result_names[:3]}
    for name, numerator in (("hot_to_cold_fraction", "hot_to_cold_tracer_count"), ("unknown_fraction", "unknown_tracer_count")):
        pooled[name] = np.full(num_radial_bins, np.nan, dtype=np.float64)
        np.divide(pooled[numerator], pooled["hot_tracer_count"], out=pooled[name], where=pooled["hot_tracer_count"] > 0)
    pooled["bin_complete"] = (pooled["hot_tracer_count"] > 0) & (pooled["unknown_tracer_count"] == 0)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_file, "w") as f:
        f.create_dataset("tree_row", data=selected_rows)
        f.create_dataset("start_subhalo_id", data=start_ids[selected_rows])
        f.create_dataset("start_node_subhalo_id", data=node_ids[selected_rows])
        f.create_dataset("start_fof_id", data=selected_fof_ids)
        f.create_dataset("m200c_msun", data=halo_mass_msun[selected_fof_ids])
        f.create_dataset("r200c_ckpc_h", data=halo_r200c[selected_fof_ids])
        f.create_dataset("target_tree_valid", data=target_tree_valid.astype(np.uint8))
        f.create_dataset("radial_bin_edges_r200c", data=radial_edges)
        f.create_dataset("radial_bin_centers_r200c", data=radial_centers)
        for name, values in results.items():
            f.create_dataset(name, data=values, compression="gzip", compression_opts=4)
        pooled_group = f.create_group("pooled")
        for name, values in pooled.items():
            pooled_group.create_dataset(name, data=values)

    return output_file, results["hot_to_cold_fraction"]