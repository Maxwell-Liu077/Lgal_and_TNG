"""Two-snapshot main-progenitor-branch access for the selected sample.

This module isolates SubLink tree reads from sample selection and downstream
physics, allowing merger-tree loading to be tested or replaced independently.
"""

from __future__ import annotations

import numpy as np


TREE_FIELDS = [
    "SnapNum",
    "SubfindID",
    "SubhaloGrNr",
    "SubhaloPos",
    "Group_M_Crit200",
    "Group_R_Crit200",
    "GroupPos",
]


def load_two_snapshot_mpb(
    base_path: str,
    subhalo_id_z0: int,
    *,
    snap_prev: int = 98,
    snap_cur: int = 99,
    h: float = 0.6774,
) -> dict:
    """Load the two required nodes of one SubLink main progenitor branch."""

    try:
        import illustris_python as il
    except ImportError as exc:
        raise ImportError("illustris_python is required for SubLink") from exc

    tree = il.sublink.loadTree(
        base_path,
        int(snap_cur),
        int(subhalo_id_z0),
        onlyMPB=True,
        fields=TREE_FIELDS,
    )
    if tree is None or len(tree.get("SnapNum", [])) == 0:
        raise ValueError(f"No MPB for SubhaloID(z=0)={subhalo_id_z0}")

    snap_nums = np.asarray(tree["SnapNum"], dtype=int)
    mass_conversion = 1.0e10 / h
    result = {"subhalo_id_z0": int(subhalo_id_z0)}
    for suffix, snap_num in (("prev", snap_prev), ("cur", snap_cur)):
        positions = np.where(snap_nums == int(snap_num))[0]
        if len(positions) != 1:
            raise ValueError(
                f"MPB of subhalo {subhalo_id_z0} has "
                f"{len(positions)} nodes at snap {snap_num}"
            )
        index = int(positions[0])
        group_id = int(tree["SubhaloGrNr"][index])
        subfind_id = int(tree["SubfindID"][index])
        group = il.groupcat.loadSingle(
            base_path,
            int(snap_num),
            haloID=group_id,
        )
        group_first_sub = int(group["GroupFirstSub"])
        if subfind_id != group_first_sub:
            raise ValueError(
                f"MPB node {subfind_id} is not central "
                f"GroupFirstSub={group_first_sub} at snap {snap_num}"
            )
        subhalo = il.groupcat.loadSingle(
            base_path,
            int(snap_num),
            subhaloID=subfind_id,
        )
        vmax_km_s = float(subhalo["SubhaloVmax"])
        m200c_msun = (
            float(tree["Group_M_Crit200"][index]) * mass_conversion
        )
        r200c_ckpc_h = float(tree["Group_R_Crit200"][index])
        mass_in_rad_type = np.asarray(
            subhalo["SubhaloMassInRadType"],
            dtype=float,
        )
        if mass_in_rad_type.ndim != 1 or len(mass_in_rad_type) <= 4:
            raise ValueError(
                f"Invalid SubhaloMassInRadType for subhalo "
                f"{subhalo_id_z0} at snap {snap_num}"
            )
        mstar_msun = float(mass_in_rad_type[4] * mass_conversion)
        if (
            group_id < 0
            or m200c_msun <= 0
            or r200c_ckpc_h <= 0
            or mstar_msun <= 0
            or vmax_km_s <= 0
        ):
            raise ValueError(
                f"Invalid halo properties for subhalo {subhalo_id_z0} "
                f"at snap {snap_num}"
            )
        result.update(
            {
                f"snap_{suffix}": int(snap_num),
                f"subfind_id_{suffix}": subfind_id,
                f"group_id_{suffix}": group_id,
                f"group_first_sub_{suffix}": group_first_sub,
                f"group_center_{suffix}_ckpc_h": np.asarray(
                    tree["GroupPos"][index],
                    dtype=float,
                ),
                f"subhalo_center_{suffix}_ckpc_h": np.asarray(
                    tree["SubhaloPos"][index],
                    dtype=float,
                ),
                f"mstar_{suffix}_msun": mstar_msun,
                f"m200c_{suffix}_msun": m200c_msun,
                f"r200c_{suffix}_ckpc_h": r200c_ckpc_h,
                f"vmax_{suffix}_km_s": vmax_km_s,
            }
        )
    return result


def build_two_snapshot_mpb_table(
    base_path: str,
    subhalo_ids_z0: np.ndarray,
    *,
    snap_prev: int = 98,
    snap_cur: int = 99,
    h: float = 0.6774,
    skip_invalid: bool = True,
    verbose: bool = True,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Self-contained multi-halo replacement for the earlier MPB helper."""

    records = []
    rejected = []
    subhalo_ids = np.asarray(subhalo_ids_z0, dtype=np.int64)
    for sequence, subhalo_id in enumerate(subhalo_ids, start=1):
        if verbose:
            print(
                f"[MPB] {sequence}/{len(subhalo_ids)} "
                f"SubhaloID(z=0)={int(subhalo_id)}"
            )
        try:
            records.append(
                load_two_snapshot_mpb(
                    base_path,
                    int(subhalo_id),
                    snap_prev=snap_prev,
                    snap_cur=snap_cur,
                    h=h,
                )
            )
        except Exception as exc:
            if not skip_invalid:
                raise
            rejected.append(
                {
                    "subhalo_id_z0": int(subhalo_id),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    if not records:
        return {}, rejected
    keys = tuple(records[0])
    table = {
        key: np.asarray([record[key] for record in records])
        for key in keys
    }
    return table, rejected
