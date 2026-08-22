"""Multi-snapshot SubLink main-progenitor branch loading."""

from __future__ import annotations

import numpy as np

from ..utils.config import Phase21Config


TREE_FIELDS = [
    "SnapNum", "SubfindID", "SubhaloGrNr", "SubhaloPos", "GroupPos",
    "Group_M_Crit200", "Group_R_Crit200",
]


def _unresolved_branch(snap: int, reason: str) -> dict:
    """Return a typed placeholder for a missing non-anchor MPB node.

    The event anchors are validated before this helper is used.  A missing
    historical node is intentionally retained so the later state/event layer
    can mark only the affected tracer as ``unresolved``/``other``.
    """

    nan3 = [float("nan")] * 3
    return {
        "snap": int(snap),
        "group_id": -1,
        "subfind_id": -1,
        "group_first_sub": -1,
        "is_main_central": False,
        "subhalo_flag": False,
        "group_center_ckpc_h": nan3,
        "subhalo_center_ckpc_h": nan3.copy(),
        "m200c_msun": float("nan"),
        "r200c_ckpc_h": float("nan"),
        "mstar_msun": float("nan"),
        "vmax_km_s": float("nan"),
        "m_bh_msun": float("nan"),
        "unresolved_snapshot": True,
        "unresolved_reason": str(reason),
    }


def load_mpb(
    base_path: str,
    subhalo_id_z99: int,
    config: Phase21Config,
    *,
    catalogues: dict[int, dict[str, np.ndarray]] | None = None,
) -> dict[int, dict]:
    """Load one snap90--99 branch and endpoint subhalo properties.

    ``catalogues`` is an optional snapshot-level cache.  When supplied, it
    replaces one ``loadSingle`` group/subhalo query per MPB node with array
    indexing into data loaded once per snapshot.  The fallback keeps the
    public three-argument API usable for small callers and tests.
    """

    try:
        import illustris_python as il
    except ImportError as exc:
        raise ImportError("illustris_python is required for SubLink") from exc
    tree = il.sublink.loadTree(base_path, config.snap_selection, int(subhalo_id_z99), onlyMPB=True, fields=TREE_FIELDS)
    if tree is None or len(tree.get("SnapNum", [])) == 0:
        raise ValueError(f"No MPB for subhalo {subhalo_id_z99}")
    snaps = np.asarray(tree["SnapNum"], dtype=int)
    output: dict[int, dict] = {}
    for snap in config.snapshots:
        positions = np.where(snaps == snap)[0]
        if len(positions) != 1:
            reason = f"MPB has {len(positions)} nodes at snap {snap}"
            if snap in {config.snap_event_prev, config.snap_event_cur}:
                raise ValueError(reason)
            output[snap] = _unresolved_branch(snap, reason)
            continue
        index = int(positions[0])
        try:
            group_id = int(tree["SubhaloGrNr"][index])
            subfind_id = int(tree["SubfindID"][index])
            m200 = float(tree["Group_M_Crit200"][index]) * 1.0e10 / config.h
            r200 = float(tree["Group_R_Crit200"][index])
            group_center = np.asarray(tree["GroupPos"][index], dtype=float)
            subhalo_center = np.asarray(tree["SubhaloPos"][index], dtype=float)
            catalogue = catalogues.get(snap) if catalogues is not None else None
            if catalogue is not None:
                group_first = int(catalogue["GroupFirstSub"][group_id])
                mass_type = np.asarray(catalogue["SubhaloMassInRadType"][subfind_id], dtype=float)
                subhalo_flag = bool(catalogue["SubhaloFlag"][subfind_id])
                vmax = float(catalogue["SubhaloVmax"][subfind_id])
                bh_mass = float(catalogue["SubhaloBHMass"][subfind_id])
            else:
                group = il.groupcat.loadSingle(base_path, snap, haloID=group_id)
                subhalo = il.groupcat.loadSingle(base_path, snap, subhaloID=subfind_id)
                group_first = int(group["GroupFirstSub"])
                mass_type = np.asarray(subhalo["SubhaloMassInRadType"], dtype=float)
                subhalo_flag = bool(subhalo.get("SubhaloFlag", True))
                vmax = float(subhalo.get("SubhaloVmax", np.nan))
                bh_mass = float(subhalo["SubhaloBHMass"]) if "SubhaloBHMass" in subhalo else float("nan")
            if (
                not np.isfinite(m200) or not np.isfinite(r200) or m200 <= 0 or r200 <= 0
                or mass_type.ndim != 1 or len(mass_type) <= 4
                or group_center.shape != (3,) or subhalo_center.shape != (3,)
                or not np.all(np.isfinite(group_center)) or not np.all(np.isfinite(subhalo_center))
            ):
                raise ValueError(f"Invalid MPB properties at snap {snap}")
            output[snap] = {
                "snap": snap,
                "group_id": group_id,
                "subfind_id": subfind_id,
                "group_first_sub": group_first,
                "is_main_central": subfind_id == group_first,
                "subhalo_flag": subhalo_flag,
                "group_center_ckpc_h": group_center.tolist(),
                "subhalo_center_ckpc_h": subhalo_center.tolist(),
                "m200c_msun": m200,
                "r200c_ckpc_h": r200,
                "mstar_msun": float(mass_type[4] * 1.0e10 / config.h),
                "vmax_km_s": vmax,
                "m_bh_msun": bh_mass * 1.0e10 / config.h if np.isfinite(bh_mass) else float("nan"),
            }
        except Exception as exc:
            if snap in {config.snap_event_prev, config.snap_event_cur}:
                raise
            output[snap] = _unresolved_branch(snap, f"{type(exc).__name__}: {exc}")
    return output
