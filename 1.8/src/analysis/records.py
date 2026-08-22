"""Ordered per-halo record assembly for Phase 1.8."""

from __future__ import annotations


def _base_record(
    sample_row: dict,
    state: dict,
    sam: dict,
    *,
    dt_gyr: float,
) -> dict:
    """Assemble the fixed leading block of the per-halo output schema."""

    source = state["source"]
    current = state["current"]
    target = state["target"]
    m200c_prev = float(sample_row["m200c_prev_msun"])
    m200c_cur = float(sample_row["m200c_cur_msun"])
    mstar_prev = float(sample_row["mstar_prev_msun"])
    mstar_cur = float(sample_row["mstar_cur_msun"])
    m_hot_prev = float(source["m_hot_msun"])
    m_hot_cur = float(current["m_hot_msun"])
    m_cgm_prev = float(source["m_cgm_msun"])
    m_cgm_cur = float(current["m_cgm_msun"])
    return {
        "mode": "slow",
        "sample_index": int(sample_row["sample_index"]),
        "subhalo_id_z0": int(sample_row["subhalo_id_z0"]),
        "group_id_prev": int(sample_row["group_id_prev"]),
        "group_id_cur": int(sample_row["group_id_cur"]),
        "subfind_id_prev": int(sample_row["subfind_id_prev"]),
        "subfind_id_cur": int(sample_row["subfind_id_cur"]),
        "mass_bin_index": int(sample_row["mass_bin_index"]),
        "mass_bin_low": float(sample_row["mass_bin_low"]),
        "mass_bin_high": float(sample_row["mass_bin_high"]),
        "mass_bin_center": float(sample_row["mass_bin_center"]),
        "log_mstar_z0": float(sample_row["log_mstar_z0"]),
        "mstar_z0_msun": mstar_cur,
        "mstar_prev_msun": mstar_prev,
        "mstar_cur_msun": mstar_cur,
        "mstar_mean_msun": 0.5 * (mstar_prev + mstar_cur),
        "log_m200c_z0": float(sample_row["log_m200c_z0"]),
        "m200c_z0_msun": m200c_cur,
        "m200c_prev_msun": m200c_prev,
        "m200c_cur_msun": m200c_cur,
        "m200c_mean_msun": 0.5 * (m200c_prev + m200c_cur),
        "r200c_prev_pkpc": float(source["r200c_pkpc"]),
        "r200c_cur_pkpc": float(current["r200c_pkpc"]),
        "dt_gyr": float(dt_gyr),
        "m_hot_prev_msun": m_hot_prev,
        "m_hot_cur_msun": m_hot_cur,
        "m_hot_mean_msun": 0.5 * (m_hot_prev + m_hot_cur),
        "m_hot_source_prev_msun": float(
            source["m_hot_source_msun"]
        ),
        "m_cold_prev_msun": float(source["m_cold_msun"]),
        "m_cold_cur_msun": float(current["m_cold_msun"]),
        "m_cgm_prev_msun": m_cgm_prev,
        "m_cgm_cur_msun": m_cgm_cur,
        "m_cgm_mean_msun": 0.5 * (m_cgm_prev + m_cgm_cur),
        "z_hot_prev_mass_fraction": float(
            source["z_hot_mass_fraction"]
        ),
        "z_hot_cur_mass_fraction": float(
            current["z_hot_mass_fraction"]
        ),
        "n_hot_gas_prev": int(source["n_hot_gas"]),
        "n_hot_gas_cur": int(current["n_hot_gas"]),
        "n_hot_source_gas_prev": int(source["n_hot_source_gas"]),
        "n_cgm_gas_prev": int(source["n_cgm_gas"]),
        "n_cgm_gas_cur": int(current["n_cgm_gas"]),
        "n_outer_cold_gas_prev": int(
            source["n_outer_cold_gas"]
        ),
        "n_central_cold_gas_prev": int(
            source["n_central_cold_gas"]
        ),
        "n_satellite_gas_excluded_prev": int(
            source["n_satellite_gas_excluded"]
        ),
        "n_satellite_gas_excluded_cur": int(
            current["n_satellite_gas_excluded"]
        ),
        "n_central_cold_gas_cur": int(
            target["n_central_cold_gas"]
        ),
        "n_central_stars_cur": int(target["n_central_stars"]),
        "mass_target_cur_msun": float(target["mass_target_msun"]),
        "mstar_formed_interval_msun_measured": float(
            target["mstar_formed_interval_msun"]
        ),
        "n_stars_formed_interval": int(
            target["n_stars_formed_interval"]
        ),
        **sam,
    }


def _combine_sam_endpoints(previous: dict, current: dict) -> dict:
    """Combine endpoint cooling outputs without changing field order."""

    rate_key = "rate_sam_isothermal_msun_per_yr"
    rate_previous = float(previous[rate_key])
    rate_current = float(current[rate_key])
    output = {
        rate_key: 0.5 * (rate_previous + rate_current),
        "rate_sam_snap98_msun_per_yr": rate_previous,
        "rate_sam_snap99_msun_per_yr": rate_current,
        "sam_cooling_mode_prev": previous["sam_cooling_mode"],
        "sam_cooling_mode_cur": current["sam_cooling_mode"],
    }
    for key in previous:
        if key in {rate_key, "sam_cooling_mode"}:
            continue
        output[f"{key}_prev"] = previous[key]
        output[f"{key}_cur"] = current[key]
    return output


