from pathlib import Path
import sys
import funcs
import h5py
import illustris_python as il
import numpy as np
import cooling_radius as cr

def Lgal15_SN_rate(sfr_msun_per_yr, vmax_kms, v200c_kms):

    sfr = np.asarray(sfr_msun_per_yr, dtype=np.float64)
    vmax = np.asarray(vmax_kms, dtype=np.float64)
    v200 = np.asarray(v200c_kms, dtype=np.float64)

    valid = np.isfinite(sfr) & (sfr >= 0.0) & np.isfinite(vmax) & (vmax > 0.0) & np.isfinite(v200) & (v200 > 0.0)

    epsilon_disk = np.full(sfr.shape, np.nan, dtype=np.float64)
    epsilon_halo = np.full(sfr.shape, np.nan, dtype=np.float64)
    reheat_nominal = np.full(sfr.shape, np.nan, dtype=np.float64)
    reheat = np.full(sfr.shape, np.nan, dtype=np.float64)
    eject = np.full(sfr.shape, np.nan, dtype=np.float64)
    energy_rate = np.full(sfr.shape, np.nan, dtype=np.float64)

    epsilon_disk[valid] = 2.6 * (0.5 + (vmax[valid] / 480.0) ** -0.72)
    epsilon_halo[valid] = 0.62 * (0.5 + (vmax[valid] / 100.0) ** -0.80)
    energy_limited_loading = epsilon_halo[valid] * (630.0 / v200[valid]) ** 2

    reheat_nominal[valid] = epsilon_disk[valid] * sfr[valid]
    reheat[valid] = np.minimum(epsilon_disk[valid], energy_limited_loading) * sfr[valid]
    eject[valid] = np.maximum(energy_limited_loading - epsilon_disk[valid], 0.0) * sfr[valid]

    msun_g = 1.98847e33
    year_s = 365.25 * 24.0 * 3600.0
    energy_rate[valid] = (0.5 * epsilon_halo[valid] * sfr[valid] * msun_g / year_s * (630.0e5) ** 2)

    return {"epsilon_disk": epsilon_disk, "epsilon_halo": epsilon_halo, "reheat_nominal_msun_per_yr": reheat_nominal, "reheat_msun_per_yr": reheat, "eject_msun_per_yr": eject, "sn_energy_erg_per_s": energy_rate, "valid": valid}


def lgal_radio_agn_rate(k_agn, hot_mass_msun, bh_mass_msun, v200c_kms, cooling_rate_msun_per_yr):

    hot_mass = np.asarray(hot_mass_msun, dtype=np.float64)
    bh_mass = np.asarray(bh_mass_msun, dtype=np.float64)
    v200 = np.asarray(v200c_kms, dtype=np.float64)
    cooling_rate = np.asarray(cooling_rate_msun_per_yr, dtype=np.float64)

    valid = np.isfinite(hot_mass) & (hot_mass >= 0.0) & np.isfinite(bh_mass) & (bh_mass >= 0.0) & np.isfinite(v200) & (v200 > 0.0) & np.isfinite(cooling_rate) & (cooling_rate >= 0.0)

    bh_accretion_rate = np.full(hot_mass.shape, np.nan, dtype=np.float64)
    heating_power = np.full(hot_mass.shape, np.nan, dtype=np.float64)
    heat_rate = np.full(hot_mass.shape, np.nan, dtype=np.float64)
    cooling_suppressed = np.full(hot_mass.shape, np.nan, dtype=np.float64)
    cooling_effective = np.full(hot_mass.shape, np.nan, dtype=np.float64)

    bh_accretion_rate[valid] = k_agn * (hot_mass[valid] / 1.0e11) * (bh_mass[valid] / 1.0e8)
    msun_g = 1.98847e33
    year_s = 365.25 * 24.0 * 3600.0
    c_kms = 299792.458
    heating_power[valid] = 0.1 * bh_accretion_rate[valid] * msun_g / year_s * (c_kms * 1.0e5) ** 2
    heat_rate[valid] = 2.0 * 0.1 * bh_accretion_rate[valid] * (c_kms / v200[valid]) ** 2
    
    cooling_suppressed[valid] = np.minimum(cooling_rate[valid], heat_rate[valid])
    cooling_effective[valid] = np.maximum(cooling_rate[valid] - heat_rate[valid], 0.0)

    return {"bh_accretion_msun_per_yr": bh_accretion_rate, "heating_power_erg_per_s": heating_power, "heat_rate_msun_per_yr": heat_rate, "cooling_rate_unheated_msun_per_yr": cooling_rate, "cooling_suppressed_msun_per_yr": cooling_suppressed, "cooling_effective_msun_per_yr": cooling_effective, "valid": valid}


def Lgal15_AGN_rate(hot_mass_msun, bh_mass_msun, v200c_kms, cooling_rate_msun_per_yr):

    return lgal_radio_agn_rate(5.3e-3, hot_mass_msun, bh_mass_msun, v200c_kms, cooling_rate_msun_per_yr)


def Lgal20_AGN_rate(hot_mass_msun, bh_mass_msun, v200c_kms, cooling_rate_msun_per_yr):

    return lgal_radio_agn_rate(2.5e-3, hot_mass_msun, bh_mass_msun, v200c_kms, cooling_rate_msun_per_yr)

def calculate_feedback_rates(basePath, snap_1, snap_2):

    data_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")
    tree_file = data_dir / "all_selected_trees.hdf5"

    snaps = (snap_1, snap_2)
    tree_file = Path(tree_file)
    result_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")
    result_dir.mkdir(parents=True, exist_ok=True)
    output_file = result_dir / f"feedback_rates_{snap_1}_and_{snap_2}.hdf5"

    with h5py.File(tree_file, "r") as f:
        start_ids = np.asarray(f["start_subhalo_id"][:], dtype=np.int64)
        snap_numbers = np.asarray(f["snap_numbers"][:], dtype=np.int64)
        node_ids = np.column_stack([np.asarray(f["mpb_subhalo_ids"][:, funcs.get_snap_column(snap, snap_numbers)], dtype=np.int64) for snap in snaps])
        fof_ids = np.column_stack([np.asarray(f["mpb_fof_ids"][:, funcs.get_snap_column(snap, snap_numbers)], dtype=np.int64) for snap in snaps])

    num_trees = start_ids.size
    states = []

    for snap_index, snap in enumerate(snaps):
        header = il.groupcat.loadHeader(basePath, snap)
        h = header["HubbleParam"]
        scale_factor = float(header["Time"])

        subhalos = il.groupcat.loadSubhalos(basePath, snap, fields=["SubhaloSFR", "SubhaloVmax", "SubhaloBHMass"])
        cooling_radius_ratio, m_hot_g, r200_raw, t_dyn_s, halo_flag = cr.cooling_radius(basePath, snap)

        group_data = il.groupcat.loadHalos(basePath, snap, fields=["Group_M_Crit200"])
        group_m200_raw = np.asarray(group_data["Group_M_Crit200"], dtype=np.float64)
        group_m200_msun = group_m200_raw * 1.0e10 / h
        r200_kpc = np.asarray(r200_raw, dtype=np.float64) * scale_factor / h
        v200_all = np.full(group_m200_msun.shape, np.nan, dtype=np.float64)
        
        valid_v200 = np.isfinite(group_m200_msun) & (group_m200_msun > 0.0) & np.isfinite(r200_kpc) & (r200_kpc > 0.0)
        
        g_kpc_kms2_per_msun = 4.30091e-6
        v200_all[valid_v200] = np.sqrt(g_kpc_kms2_per_msun * group_m200_msun[valid_v200] / r200_kpc[valid_v200])

        node_id = node_ids[:, snap_index]
        fof_id = fof_ids[:, snap_index]

        valid_node = (node_id >= 0) & (node_id < len(subhalos["SubhaloSFR"]))

        sfr = np.full(num_trees, np.nan, dtype=np.float64)
        vmax = np.full(num_trees, np.nan, dtype=np.float64)
        bh_mass = np.full(num_trees, np.nan, dtype=np.float64)

        sfr_all = np.asarray(subhalos["SubhaloSFR"], dtype=np.float64)
        vmax_all = np.asarray(subhalos["SubhaloVmax"], dtype=np.float64)
        bh_mass_all = np.asarray(subhalos["SubhaloBHMass"], dtype=np.float64)

        sfr[valid_node] = sfr_all[node_id[valid_node]]
        vmax[valid_node] = vmax_all[node_id[valid_node]]
        bh_mass[valid_node] = (bh_mass_all[node_id[valid_node]] * 1.0e10 / h)

        valid_fof = (fof_id >= 0) & (fof_id < len(m_hot_g))

        m_hot = np.full(num_trees, np.nan, dtype=np.float64)
        r_cool = np.full(num_trees, np.nan, dtype=np.float64)
        t_dyn = np.full(num_trees, np.nan, dtype=np.float64)
        v200 = np.full(num_trees, np.nan, dtype=np.float64)
        halo_flag_tree = np.full(num_trees, np.nan, dtype=np.float64)

        m_hot_all = np.asarray(m_hot_g, dtype=np.float64)
        r_cool_all = np.asarray(cooling_radius_ratio, dtype=np.float64)
        t_dyn_all = np.asarray(t_dyn_s, dtype=np.float64)
        halo_flag_all = np.asarray(halo_flag, dtype=np.float64)

        # cooling_radius 返回的 M_hot 单位为克，转换为 Msun。
        msun_g = 1.98847e33
        m_hot[valid_fof] = m_hot_all[fof_id[valid_fof]] / msun_g
        r_cool[valid_fof] = r_cool_all[fof_id[valid_fof]]
        t_dyn[valid_fof] = t_dyn_all[fof_id[valid_fof]]
        v200[valid_fof] = v200_all[fof_id[valid_fof]]
        halo_flag_tree[valid_fof] = halo_flag_all[fof_id[valid_fof]]

        states.append({"sfr_msun_per_yr": sfr, "vmax_kms": vmax, "bh_mass_msun": bh_mass, "m_hot_msun": m_hot, "cooling_radius_ratio": r_cool, "t_dyn_s": t_dyn, "v200c_kms": v200, "halo_flag": halo_flag_tree})

    cooling_file = data_dir / f"Lgal_cooling_rate_{snaps[0]}_and_{snaps[1]}.hdf5"
    with h5py.File(cooling_file, "r") as f:
        cooling_rate = np.asarray(f["cooling_rate"][:], dtype=np.float64)
        cooling_valid = np.asarray(f["valid"][:], dtype=bool)
        halo_mass = np.asarray(f["halo_mass"][:], dtype=np.float64)

    sn_1 = Lgal15_SN_rate(states[0]["sfr_msun_per_yr"], states[0]["vmax_kms"], states[0]["v200c_kms"])
    sn_2 = Lgal15_SN_rate(states[1]["sfr_msun_per_yr"], states[1]["vmax_kms"], states[1]["v200c_kms"])

    agn15_1 = Lgal15_AGN_rate(states[0]["m_hot_msun"], states[0]["bh_mass_msun"], states[0]["v200c_kms"], cooling_rate)
    agn15_2 = Lgal15_AGN_rate(states[1]["m_hot_msun"], states[1]["bh_mass_msun"], states[1]["v200c_kms"], cooling_rate)

    agn20_1 = Lgal20_AGN_rate(states[0]["m_hot_msun"], states[0]["bh_mass_msun"], states[0]["v200c_kms"], cooling_rate)
    agn20_2 = Lgal20_AGN_rate(states[1]["m_hot_msun"], states[1]["bh_mass_msun"], states[1]["v200c_kms"], cooling_rate)

    valid_sfr_pair = np.isfinite(states[0]["sfr_msun_per_yr"]) & np.isfinite(states[1]["sfr_msun_per_yr"])
    sfr_mean = np.full(num_trees, np.nan, dtype=np.float64)
    sfr_mean[valid_sfr_pair] = 0.5 * (states[0]["sfr_msun_per_yr"][valid_sfr_pair] + states[1]["sfr_msun_per_yr"][valid_sfr_pair])

    valid_vmax_pair = np.isfinite(states[0]["vmax_kms"]) & np.isfinite(states[1]["vmax_kms"])
    vmax_mean = np.full(num_trees, np.nan, dtype=np.float64)
    vmax_mean[valid_vmax_pair] = 0.5 * (states[0]["vmax_kms"][valid_vmax_pair] + states[1]["vmax_kms"][valid_vmax_pair])

    valid_v200_pair = np.isfinite(states[0]["v200c_kms"]) & np.isfinite(states[1]["v200c_kms"])
    v200_mean = np.full(num_trees, np.nan, dtype=np.float64)
    v200_mean[valid_v200_pair] = 0.5 * (states[0]["v200c_kms"][valid_v200_pair] + states[1]["v200c_kms"][valid_v200_pair])

    valid_m_hot_pair = np.isfinite(states[0]["m_hot_msun"]) & np.isfinite(states[1]["m_hot_msun"])

    m_hot_mean = np.full(num_trees, np.nan, dtype=np.float64)
    m_hot_mean[valid_m_hot_pair] = 0.5 * (states[0]["m_hot_msun"][valid_m_hot_pair] + states[1]["m_hot_msun"][valid_m_hot_pair])

    valid_bh_pair = np.isfinite(states[0]["bh_mass_msun"]) & np.isfinite(states[1]["bh_mass_msun"])
    bh_mass_mean = np.full(num_trees, np.nan, dtype=np.float64)
    bh_mass_mean[valid_bh_pair] = 0.5 * (states[0]["bh_mass_msun"][valid_bh_pair] + states[1]["bh_mass_msun"][valid_bh_pair])

    sn_mean = Lgal15_SN_rate(sfr_mean, vmax_mean, v200_mean)

    cooling_rate_mean_input = cooling_rate.copy()
    cooling_rate_mean_input[~cooling_valid] = np.nan
    agn15_mean = Lgal15_AGN_rate(m_hot_mean, bh_mass_mean, v200_mean, cooling_rate_mean_input)
    agn20_mean = Lgal20_AGN_rate(m_hot_mean, bh_mass_mean, v200_mean, cooling_rate_mean_input)

    model_results = {
        "Lgal15_SN": (sn_1, sn_2, sn_mean),
        "Lgal15_AGN": (agn15_1, agn15_2, agn15_mean),
        "Lgal20_AGN": (agn20_1, agn20_2, agn20_mean),
    }

    evaluation_names = np.asarray([f"snap_{snaps[0]}", f"snap_{snaps[1]}", "mean_input_properties"], dtype="S40")

    with h5py.File(output_file, "w") as f:
        f.create_dataset("start_ids", data=start_ids)
        f.create_dataset("node_ids", data=node_ids)
        f.create_dataset("fof_ids", data=fof_ids)
        f.create_dataset("halo_mass", data=halo_mass)
        f.create_dataset("snap_numbers", data=np.asarray(snaps, dtype=np.int64))
        
        for model_name, results in model_results.items():
            result_snap_1, result_snap_2, result_mean = results
            model_group = f.create_group(model_name)
            model_group.create_dataset("evaluation_names", data=evaluation_names)

            model_group.create_dataset("valid", data=np.column_stack([result_snap_1["valid"], result_snap_2["valid"], result_mean["valid"]]).astype(np.uint8))

            for quantity_name in result_snap_1:
                if quantity_name == "valid":
                    continue

                values = np.column_stack([result_snap_1[quantity_name], result_snap_2[quantity_name], result_mean[quantity_name]])
                model_group.create_dataset(quantity_name, data=values, compression="gzip", compression_opts=4)
