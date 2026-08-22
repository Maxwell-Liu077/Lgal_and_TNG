"""Load and classify gas/star particle states at both interval endpoints.

The functions in this module own aperture geometry, phase classification, and
Subfind membership exclusion.  They do not compute population statistics or
render figures, keeping scientific measurement separate from presentation.
"""

from __future__ import annotations

import numpy as np

from ..utils.config import Phase18Config


SOURCE_GAS_FIELDS = [
    "ParticleIDs",
    "Coordinates",
    "Masses",
    "StarFormationRate",
    "InternalEnergy",
    "ElectronAbundance",
    "GFM_Metallicity",
]

TARGET_GAS_FIELDS = SOURCE_GAS_FIELDS

TARGET_STAR_FIELDS = [
    "ParticleIDs",
    "Coordinates",
    "Masses",
    "GFM_InitialMass",
    "GFM_StellarFormationTime",
]

PROTON_MASS_G = 1.67262192369e-24
BOLTZMANN_CGS = 1.380649e-16

GROUP_MEMBERSHIP_FIELDS = [
    "GroupFirstSub",
    "GroupNsubs",
]

SUBHALO_MEMBERSHIP_FIELDS = [
    "SubhaloLenType",
]


def internal_energy_to_temperature(
    internal_energy: np.ndarray,
    electron_abundance: np.ndarray,
) -> np.ndarray:
    """Convert TNG internal energy to gas temperature in Kelvin."""

    hydrogen_mass_fraction = 0.76
    gamma = 5.0 / 3.0
    mean_molecular_weight_g = (
        4.0
        * PROTON_MASS_G
        / (
            1.0
            + 3.0 * hydrogen_mass_fraction
            + 4.0 * hydrogen_mass_fraction * electron_abundance
        )
    )
    return (
        np.asarray(internal_energy, dtype=float)
        * 1.0e10
        * (gamma - 1.0)
        * mean_molecular_weight_g
        / BOLTZMANN_CGS
    )


def periodic_radius(
    coordinates: np.ndarray,
    center: np.ndarray,
    box_size: float,
) -> np.ndarray:
    """Return minimum-image distances in a periodic simulation cube."""

    displacement = (
        np.asarray(coordinates, dtype=float)
        - np.asarray(center, dtype=float)
        + 0.5 * box_size
    ) % box_size - 0.5 * box_size
    return np.linalg.norm(displacement, axis=-1)


def _arrays_or_empty(
    data: dict,
    fields: list[str],
    *,
    component: str,
) -> dict[str, np.ndarray]:
    """Normalize loader output and provide typed empty component arrays."""

    if not isinstance(data, dict):
        raise ValueError(f"Expected a dictionary for {component}")
    missing = [field for field in fields if field not in data]
    if missing:
        if int(data.get("count", -1)) != 0:
            raise KeyError(f"Missing {component} fields: {missing}")
        result = {}
        for field in fields:
            if field == "Coordinates":
                result[field] = np.empty((0, 3), dtype=float)
            else:
                result[field] = np.empty(0, dtype=float)
        return result
    result = {field: np.asarray(data[field]) for field in fields}
    lengths = {len(values) for values in result.values()}
    if len(lengths) != 1:
        raise ValueError(f"{component} fields have inconsistent lengths")
    return result


def _sort_ids_and_masses(
    particle_ids: np.ndarray,
    masses_msun: np.ndarray,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate one-to-one IDs/masses and return a stable ID-sorted pair."""

    particle_ids = np.asarray(particle_ids, dtype=np.uint64)
    masses_msun = np.asarray(masses_msun, dtype=float)
    if len(particle_ids) != len(masses_msun):
        raise ValueError(f"{label} IDs and masses differ in length")
    order = np.argsort(particle_ids)
    ids = particle_ids[order]
    masses = masses_msun[order]
    if len(ids) > 1 and np.any(ids[1:] == ids[:-1]):
        raise ValueError(f"Duplicate ParticleIDs in {label}")
    return ids, masses


def load_subfind_membership_catalog(
    base_path: str,
    snap_num: int,
) -> dict[str, np.ndarray]:
    """Load the group/subhalo particle counts needed for membership masks."""

    try:
        import illustris_python as il
    except ImportError as exc:
        raise ImportError(
            "illustris_python is required to load Subfind membership"
        ) from exc
    halos = il.groupcat.loadHalos(
        base_path,
        snap_num,
        fields=GROUP_MEMBERSHIP_FIELDS,
    )
    subhalos = il.groupcat.loadSubhalos(
        base_path,
        snap_num,
        fields=SUBHALO_MEMBERSHIP_FIELDS,
    )
    if not isinstance(halos, dict):
        raise ValueError("Expected a group membership dictionary")
    if isinstance(subhalos, dict):
        subhalo_len_type = subhalos["SubhaloLenType"]
    else:
        # illustris_python returns the array directly when only one
        # group-catalogue field is requested.
        subhalo_len_type = subhalos
    return {
        "group_first_sub": np.asarray(
            halos["GroupFirstSub"],
            dtype=np.int64,
        ),
        "group_nsubs": np.asarray(halos["GroupNsubs"], dtype=np.int64),
        "subhalo_len_type": np.asarray(
            subhalo_len_type,
            dtype=np.int64,
        ),
    }


def subfind_membership_counts(
    catalog: dict[str, np.ndarray],
    *,
    group_id: int,
    central_subfind_id: int,
) -> dict[str, int]:
    """Return central/satellite particle counts for one central FoF halo."""

    first_sub = np.asarray(catalog["group_first_sub"], dtype=np.int64)
    nsubs = np.asarray(catalog["group_nsubs"], dtype=np.int64)
    len_type = np.asarray(catalog["subhalo_len_type"], dtype=np.int64)
    if group_id < 0 or group_id >= len(first_sub):
        raise ValueError(f"Group ID {group_id} is outside the catalogue")
    first = int(first_sub[group_id])
    count = int(nsubs[group_id])
    if first < 0 or count < 1 or first + count > len(len_type):
        raise ValueError(f"Invalid Subfind membership for group {group_id}")
    if int(central_subfind_id) != first:
        raise ValueError(
            f"MPB subhalo {central_subfind_id} is not GroupFirstSub "
            f"{first} in group {group_id}"
        )
    if len_type.ndim != 2 or len_type.shape[1] <= 4:
        raise ValueError("SubhaloLenType must contain gas and star counts")
    central = len_type[first]
    satellite = len_type[first + 1:first + count].sum(axis=0)
    return {
        "group_first_sub": first,
        "group_nsubs": count,
        "central_gas_count": int(central[0]),
        "satellite_gas_count": int(satellite[0]),
        "central_star_count": int(central[4]),
        "satellite_star_count": int(satellite[4]),
    }


def subfind_member_ids(
    particle_ids: np.ndarray,
    *,
    central_count: int,
    satellite_count: int,
    component: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Use Subfind ordering to return central and satellite-bound IDs."""

    ids = np.asarray(particle_ids, dtype=np.uint64)
    central_count = int(central_count)
    satellite_count = int(satellite_count)
    if central_count < 0 or satellite_count < 0:
        raise ValueError(f"Negative Subfind counts for {component}")
    stop = central_count + satellite_count
    if stop > len(ids):
        raise ValueError(
            f"Subfind {component} counts ({stop}) exceed FoF load "
            f"length ({len(ids)})"
        )
    return (
        np.asarray(ids[:central_count], dtype=np.uint64),
        np.asarray(ids[central_count:stop], dtype=np.uint64),
    )


def _membership_mask(
    particle_ids: np.ndarray,
    excluded_particle_ids: np.ndarray | None,
) -> np.ndarray:
    """Mark particles not present in the explicitly excluded Subfind IDs."""

    ids = np.asarray(particle_ids, dtype=np.uint64)
    if excluded_particle_ids is None:
        return np.ones(len(ids), dtype=bool)
    excluded = np.asarray(excluded_particle_ids, dtype=np.uint64)
    if len(excluded) == 0:
        return np.ones(len(ids), dtype=bool)
    return ~np.isin(ids, excluded)


def build_source_state(
    gas: dict[str, np.ndarray],
    *,
    group_center_ckpc_h: np.ndarray,
    subhalo_center_ckpc_h: np.ndarray,
    r200c_ckpc_h: float,
    box_size_ckpc_h: float,
    scale_factor: float,
    config: Phase18Config,
    excluded_gas_particle_ids: np.ndarray | None = None,
) -> dict:
    """Build one endpoint's diffuse reservoirs and tracer source sets."""

    arrays = _arrays_or_empty(
        gas,
        SOURCE_GAS_FIELDS,
        component="source gas",
    )
    particle_ids = np.asarray(arrays["ParticleIDs"], dtype=np.uint64)
    masses_msun = (
        np.asarray(arrays["Masses"], dtype=float) * 1.0e10 / config.h
    )
    temperature = internal_energy_to_temperature(
        np.asarray(arrays["InternalEnergy"], dtype=float),
        np.asarray(arrays["ElectronAbundance"], dtype=float),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        log_temperature = np.log10(temperature)
    sfr = np.asarray(arrays["StarFormationRate"], dtype=float)
    metallicity = np.asarray(arrays["GFM_Metallicity"], dtype=float)
    diffuse = _membership_mask(
        particle_ids,
        excluded_gas_particle_ids,
    )

    radius_halo = periodic_radius(
        arrays["Coordinates"],
        group_center_ckpc_h,
        box_size_ckpc_h,
    )
    radius_galaxy = periodic_radius(
        arrays["Coordinates"],
        subhalo_center_ckpc_h,
        box_size_ckpc_h,
    )
    hot_reservoir = (
        (radius_halo < r200c_ckpc_h)
        & (sfr <= 0)
        & np.isfinite(log_temperature)
        & (log_temperature >= config.hot_log10_temperature_min)
        & diffuse
    )
    hot_source = (
        hot_reservoir
        & (radius_halo >= config.central_rfrac * r200c_ckpc_h)
    )
    cgm = (
        (radius_halo >= config.central_rfrac * r200c_ckpc_h)
        & (radius_halo < config.outer_rmax_rfrac * r200c_ckpc_h)
        & diffuse
    )
    cold = (sfr > 0) | (
        (sfr <= 0)
        & np.isfinite(log_temperature)
        & (log_temperature < config.cold_log10_temperature_max)
    )
    central_cold = (
        (radius_galaxy < config.central_rfrac * r200c_ckpc_h)
        & cold
        & diffuse
    )
    outer_cold = (
        (radius_galaxy >= config.central_rfrac * r200c_ckpc_h)
        & (radius_galaxy < config.outer_rmax_rfrac * r200c_ckpc_h)
        & cold
        & diffuse
    )
    if np.any(hot_source & outer_cold):
        raise ValueError("Hot and outer-cold source masks overlap")

    hot_ids, hot_masses = _sort_ids_and_masses(
        particle_ids[hot_source],
        masses_msun[hot_source],
        "hot source",
    )
    outer_ids, outer_masses = _sort_ids_and_masses(
        particle_ids[outer_cold],
        masses_msun[outer_cold],
        "outer-cold source",
    )
    central_cold_ids, central_cold_masses = _sort_ids_and_masses(
        particle_ids[central_cold],
        masses_msun[central_cold],
        "central-cold endpoint",
    )
    m_hot = float(masses_msun[hot_reservoir].sum())
    z_hot = (
        float(
            np.sum(
                masses_msun[hot_reservoir]
                * metallicity[hot_reservoir]
            )
            / m_hot
        )
        if m_hot > 0
        else np.nan
    )
    return {
        "hot_ids": hot_ids,
        "hot_masses_msun": hot_masses,
        "outer_cold_ids": outer_ids,
        "outer_cold_masses_msun": outer_masses,
        "central_cold_ids": central_cold_ids,
        "central_cold_masses_msun": central_cold_masses,
        "m_hot_msun": m_hot,
        "m_hot_source_msun": float(hot_masses.sum()),
        "m_cgm_msun": float(masses_msun[cgm].sum()),
        "z_hot_mass_fraction": z_hot,
        "r200c_pkpc": float(r200c_ckpc_h * scale_factor / config.h),
        "n_hot_gas": int(np.count_nonzero(hot_reservoir)),
        "n_hot_source_gas": int(len(hot_ids)),
        "n_cgm_gas": int(np.count_nonzero(cgm)),
        "n_outer_cold_gas": int(len(outer_ids)),
        "mass_outer_cold_msun": float(outer_masses.sum()),
        "n_central_cold_gas": int(len(central_cold_ids)),
        "m_cold_msun": float(central_cold_masses.sum()),
        "n_satellite_gas_excluded": int(
            len(particle_ids) - np.count_nonzero(diffuse)
        ),
    }


def build_target_state(
    gas: dict[str, np.ndarray],
    stars: dict[str, np.ndarray],
    *,
    subhalo_center_ckpc_h: np.ndarray,
    r200c_ckpc_h: float,
    box_size_ckpc_h: float,
    config: Phase18Config,
    excluded_gas_particle_ids: np.ndarray | None = None,
    central_star_particle_ids: np.ndarray | None = None,
) -> dict:
    """Build snap-99 central-cold gas and real-star target sets."""

    gas_arrays = _arrays_or_empty(
        gas,
        TARGET_GAS_FIELDS,
        component="target gas",
    )
    star_arrays = _arrays_or_empty(
        stars,
        TARGET_STAR_FIELDS,
        component="target stars",
    )
    gas_ids = np.asarray(gas_arrays["ParticleIDs"], dtype=np.uint64)
    diffuse_gas = _membership_mask(
        gas_ids,
        excluded_gas_particle_ids,
    )
    gas_masses = (
        np.asarray(gas_arrays["Masses"], dtype=float)
        * 1.0e10
        / config.h
    )
    temperature = internal_energy_to_temperature(
        np.asarray(gas_arrays["InternalEnergy"], dtype=float),
        np.asarray(gas_arrays["ElectronAbundance"], dtype=float),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        log_temperature = np.log10(temperature)
    sfr = np.asarray(gas_arrays["StarFormationRate"], dtype=float)
    cold = (sfr > 0) | (
        (sfr <= 0)
        & np.isfinite(log_temperature)
        & (log_temperature < config.cold_log10_temperature_max)
    )
    gas_radius = periodic_radius(
        gas_arrays["Coordinates"],
        subhalo_center_ckpc_h,
        box_size_ckpc_h,
    )
    central_cold = (
        gas_radius < config.central_rfrac * r200c_ckpc_h
    ) & cold & diffuse_gas

    star_ids = np.asarray(star_arrays["ParticleIDs"], dtype=np.uint64)
    star_masses = (
        np.asarray(star_arrays["Masses"], dtype=float)
        * 1.0e10
        / config.h
    )
    star_radius = periodic_radius(
        star_arrays["Coordinates"],
        subhalo_center_ckpc_h,
        box_size_ckpc_h,
    )
    formation_time = np.asarray(
        star_arrays["GFM_StellarFormationTime"],
        dtype=float,
    )
    if central_star_particle_ids is None:
        central_membership = np.ones(len(star_ids), dtype=bool)
    else:
        central_membership = np.isin(
            star_ids,
            np.asarray(central_star_particle_ids, dtype=np.uint64),
        )
    central_star = (
        (star_radius < config.central_rfrac * r200c_ckpc_h)
        & np.isfinite(formation_time)
        & (formation_time > 0)
        & central_membership
    )

    cold_ids, cold_masses = _sort_ids_and_masses(
        gas_ids[central_cold],
        gas_masses[central_cold],
        "central-cold target",
    )
    central_star_ids, central_star_masses = _sort_ids_and_masses(
        star_ids[central_star],
        star_masses[central_star],
        "central-star target",
    )
    if len(np.intersect1d(cold_ids, central_star_ids)):
        raise ValueError("Gas and star target ParticleIDs overlap")
    target_ids, target_masses = _sort_ids_and_masses(
        np.concatenate([cold_ids, central_star_ids]),
        np.concatenate([cold_masses, central_star_masses]),
        "combined target",
    )
    return {
        "central_cold_ids": cold_ids,
        "central_cold_masses_msun": cold_masses,
        "central_star_ids": central_star_ids,
        "central_star_masses_msun": central_star_masses,
        "target_ids": target_ids,
        "target_masses_msun": target_masses,
        "n_central_cold_gas": int(len(cold_ids)),
        "n_central_stars": int(len(central_star_ids)),
        "mass_target_msun": float(target_masses.sum()),
        "n_satellite_gas_excluded": int(
            len(gas_ids) - np.count_nonzero(diffuse_gas)
        ),
    }


def load_halo_pair_state(
    base_path: str,
    sample_row: dict,
    *,
    header_prev: dict,
    header_cur: dict,
    config: Phase18Config,
) -> dict:
    """Load all particle fields needed for one halo at snapshots 98 and 99."""

    try:
        import illustris_python as il
    except ImportError as exc:
        raise ImportError(
            "illustris_python is required to load TNG snapshots"
        ) from exc

    source_gas = il.snapshot.loadHalo(
        base_path,
        config.snap_prev,
        int(sample_row["group_id_prev"]),
        "gas",
        fields=SOURCE_GAS_FIELDS,
    )
    target_gas = il.snapshot.loadHalo(
        base_path,
        config.snap_cur,
        int(sample_row["group_id_cur"]),
        "gas",
        fields=TARGET_GAS_FIELDS,
    )
    target_stars = il.snapshot.loadHalo(
        base_path,
        config.snap_cur,
        int(sample_row["group_id_cur"]),
        "stars",
        fields=TARGET_STAR_FIELDS,
    )
    source_stars = il.snapshot.loadHalo(
        base_path,
        config.snap_prev,
        int(sample_row["group_id_prev"]),
        "stars",
        fields=TARGET_STAR_FIELDS,
    )
    _, satellite_gas_prev = subfind_member_ids(
        np.asarray(source_gas["ParticleIDs"], dtype=np.uint64),
        central_count=int(sample_row["central_gas_count_prev"]),
        satellite_count=int(sample_row["satellite_gas_count_prev"]),
        component="gas at snap_prev",
    )
    _, satellite_gas_cur = subfind_member_ids(
        np.asarray(target_gas["ParticleIDs"], dtype=np.uint64),
        central_count=int(sample_row["central_gas_count_cur"]),
        satellite_count=int(sample_row["satellite_gas_count_cur"]),
        component="gas at snap_cur",
    )
    central_stars_prev, satellite_stars_prev = subfind_member_ids(
        np.asarray(source_stars["ParticleIDs"], dtype=np.uint64),
        central_count=int(sample_row["central_star_count_prev"]),
        satellite_count=int(sample_row["satellite_star_count_prev"]),
        component="stars at snap_prev",
    )
    central_stars_cur, satellite_stars_cur = subfind_member_ids(
        np.asarray(target_stars["ParticleIDs"], dtype=np.uint64),
        central_count=int(sample_row["central_star_count_cur"]),
        satellite_count=int(sample_row["satellite_star_count_cur"]),
        component="stars at snap_cur",
    )
    source = build_source_state(
        source_gas,
        group_center_ckpc_h=sample_row[
            "group_center_prev_ckpc_h"
        ],
        subhalo_center_ckpc_h=sample_row[
            "subhalo_center_prev_ckpc_h"
        ],
        r200c_ckpc_h=float(sample_row["r200c_prev_ckpc_h"]),
        box_size_ckpc_h=float(header_prev["BoxSize"]),
        scale_factor=float(header_prev["Time"]),
        config=config,
        excluded_gas_particle_ids=satellite_gas_prev,
    )
    current = build_source_state(
        target_gas,
        group_center_ckpc_h=sample_row[
            "group_center_cur_ckpc_h"
        ],
        subhalo_center_ckpc_h=sample_row[
            "subhalo_center_cur_ckpc_h"
        ],
        r200c_ckpc_h=float(sample_row["r200c_cur_ckpc_h"]),
        box_size_ckpc_h=float(header_cur["BoxSize"]),
        scale_factor=float(header_cur["Time"]),
        config=config,
        excluded_gas_particle_ids=satellite_gas_cur,
    )
    source.update(
        {
            "satellite_gas_ids": satellite_gas_prev,
            "central_star_ids_all": central_stars_prev,
            "satellite_star_ids": satellite_stars_prev,
        }
    )
    current.update(
        {
            "satellite_gas_ids": satellite_gas_cur,
            "central_star_ids_all": central_stars_cur,
            "satellite_star_ids": satellite_stars_cur,
        }
    )
    target = build_target_state(
        target_gas,
        target_stars,
        subhalo_center_ckpc_h=sample_row[
            "subhalo_center_cur_ckpc_h"
        ],
        r200c_ckpc_h=float(sample_row["r200c_cur_ckpc_h"]),
        box_size_ckpc_h=float(header_cur["BoxSize"]),
        config=config,
        excluded_gas_particle_ids=satellite_gas_cur,
        central_star_particle_ids=central_stars_cur,
    )
    formation_time = np.asarray(
        target_stars["GFM_StellarFormationTime"],
        dtype=float,
    )
    initial_mass = np.asarray(
        target_stars["GFM_InitialMass"],
        dtype=float,
    )
    target_star_ids = np.asarray(
        target_stars["ParticleIDs"],
        dtype=np.uint64,
    )
    central_membership = np.isin(target_star_ids, central_stars_cur)
    formed_interval = (
        central_membership
        & np.isfinite(formation_time)
        & (formation_time > float(header_prev["Time"]))
        & (formation_time <= float(header_cur["Time"]))
    )
    target["mstar_formed_interval_msun"] = float(
        initial_mass[formed_interval].sum() * 1.0e10 / config.h
    )
    target["n_stars_formed_interval"] = int(
        np.count_nonzero(formed_interval)
    )
    return {"source": source, "current": current, "target": target}
