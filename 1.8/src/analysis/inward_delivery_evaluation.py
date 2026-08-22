"""Per-halo evaluation for the optional inward-delivery extension."""

from __future__ import annotations

import numpy as np

from ..utils.config import Phase18Config
from ..io.tracer_batch import tracer_sets_for_halo
from ..physics.sam_cooling import compute_isothermal_sam_cooling
from .records import _base_record, _combine_sam_endpoints
from ..utils.arrays import _row
from .rates import (
    compute_total_inward_delivery,
    compute_tracer_rates,
)
from ..io.tracer_history import (
    classify_parent_states,
    history_record,
    inner_hot_parent_mask,
)


def evaluate_halos(
    sample: dict[str, np.ndarray],
    states: list[dict],
    tracer_products: dict,
    cooling_function,
    *,
    config: Phase18Config,
    header_prev: dict,
    header_cur: dict,
    dt_gyr: float,
) -> list[dict]:
    """Evaluate inward-delivery rates while preserving historical field order.

    Every record is assembled in the original insertion order because that
    order defines the CSV schema written by :func:`save_phase18_result`.
    """

    source_tracers = tracer_products["source_tracers"]
    target_tracers = tracer_products["target_tracers"]
    source_sets_by_halo = tracer_products["source_sets_by_halo"]
    central_cur_tracers = tracer_products["central_cur_tracers"]
    central_prev_tracer_list = tracer_products[
        "central_prev_tracer_list"
    ]
    origin_parents = tracer_products["origin_parents"]
    fate_parents = tracer_products["fate_parents"]
    supply_fate_parents = tracer_products["supply_fate_parents"]
    origin_particle_records = tracer_products["origin_particle_records"]
    fate_particle_records = tracer_products["fate_particle_records"]
    supply_fate_particle_records = tracer_products[
        "supply_fate_particle_records"
    ]

    records = []
    for halo_index, state in enumerate(states):
        sample_row = _row(sample, halo_index)
        sam_previous = compute_isothermal_sam_cooling(
            m_hot_msun=float(state["source"]["m_hot_msun"]),
            z_hot_mass_fraction=float(
                state["source"]["z_hot_mass_fraction"]
            ),
            m200c_msun=float(sample_row["m200c_prev_msun"]),
            r200c_pkpc=float(state["source"]["r200c_pkpc"]),
            cooling_function=cooling_function,
            mean_molecular_weight=config.mean_molecular_weight,
            virial_temperature_factor=config.virial_temperature_factor,
        )
        sam_current = compute_isothermal_sam_cooling(
            m_hot_msun=float(state["current"]["m_hot_msun"]),
            z_hot_mass_fraction=float(
                state["current"]["z_hot_mass_fraction"]
            ),
            m200c_msun=float(sample_row["m200c_cur_msun"]),
            r200c_pkpc=float(state["current"]["r200c_pkpc"]),
            cooling_function=cooling_function,
            mean_molecular_weight=config.mean_molecular_weight,
            virial_temperature_factor=config.virial_temperature_factor,
        )
        sam = _combine_sam_endpoints(sam_previous, sam_current)
        source_sets = source_sets_by_halo[halo_index]
        target_sets = tracer_sets_for_halo(
            target_tracers,
            halo_index,
        )
        transition = compute_tracer_rates(
            tracer_hot_prev=source_sets["hot"],
            tracer_outer_cold_prev=source_sets["outer_cold"],
            tracer_central_cold_cur=target_sets["central_cold"],
            tracer_central_star_cur=target_sets["central_star"],
            tracer_mass_msun=config.tracer_mass_msun,
            dt_gyr=dt_gyr,
        )
        origin_parent_ids = np.asarray(
            origin_parents.get(
                halo_index,
                np.empty(0, dtype=np.uint64),
            ),
            dtype=np.uint64,
        )
        fate_parent_ids = np.asarray(
            fate_parents.get(
                halo_index,
                np.empty(0, dtype=np.uint64),
            ),
            dtype=np.uint64,
        )
        origin_codes = classify_parent_states(
            origin_parent_ids,
            origin_particle_records,
            branch_state=state["source"],
            group_center_ckpc_h=sample_row[
                "group_center_prev_ckpc_h"
            ],
            subhalo_center_ckpc_h=sample_row[
                "subhalo_center_prev_ckpc_h"
            ],
            r200c_ckpc_h=float(sample_row["r200c_prev_ckpc_h"]),
            box_size_ckpc_h=float(header_prev["BoxSize"]),
            config=config,
            direction="origin",
        )
        fate_codes = classify_parent_states(
            fate_parent_ids,
            fate_particle_records,
            branch_state=state["current"],
            group_center_ckpc_h=sample_row[
                "group_center_cur_ckpc_h"
            ],
            subhalo_center_ckpc_h=sample_row[
                "subhalo_center_cur_ckpc_h"
            ],
            r200c_ckpc_h=float(sample_row["r200c_cur_ckpc_h"]),
            box_size_ckpc_h=float(header_cur["BoxSize"]),
            config=config,
            direction="fate",
        )
        history = history_record(
            origin_codes=origin_codes,
            fate_codes=fate_codes,
            n_origin_expected=len(central_cur_tracers[halo_index]),
            n_fate_expected=len(central_prev_tracer_list[halo_index]),
            tracer_mass_msun=config.tracer_mass_msun,
            dt_gyr=dt_gyr,
        )
        outer_hot_fate_parents = np.asarray(
            supply_fate_parents.get(
                2 * halo_index,
                np.empty(0, dtype=np.uint64),
            ),
            dtype=np.uint64,
        )
        outer_cold_fate_parents = np.asarray(
            supply_fate_parents.get(
                2 * halo_index + 1,
                np.empty(0, dtype=np.uint64),
            ),
            dtype=np.uint64,
        )
        inner_hot_common = {
            "records": supply_fate_particle_records,
            "branch_state": state["current"],
            "group_center_ckpc_h": sample_row[
                "group_center_cur_ckpc_h"
            ],
            "subhalo_center_ckpc_h": sample_row[
                "subhalo_center_cur_ckpc_h"
            ],
            "r200c_ckpc_h": float(sample_row["r200c_cur_ckpc_h"]),
            "box_size_ckpc_h": float(header_cur["BoxSize"]),
            "config": config,
        }
        outer_hot_to_inner_hot = inner_hot_parent_mask(
            outer_hot_fate_parents,
            **inner_hot_common,
        )
        outer_cold_to_inner_hot = inner_hot_parent_mask(
            outer_cold_fate_parents,
            **inner_hot_common,
        )
        inward_delivery = compute_total_inward_delivery(
            transition,
            n_outer_hot_to_inner_hot=int(
                np.count_nonzero(outer_hot_to_inner_hot)
            ),
            n_outer_cold_to_inner_hot=int(
                np.count_nonzero(outer_cold_to_inner_hot)
            ),
            tracer_mass_msun=config.tracer_mass_msun,
            dt_gyr=dt_gyr,
        )
        records.append(
            {
                **_base_record(
                    sample_row,
                    state,
                    sam,
                    dt_gyr=dt_gyr,
                ),
                **transition,
                **history,
                **inward_delivery,
            }
        )

    return records
