"""Validated scientific configuration for the Phase 2.1 workflow."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np


@dataclass(frozen=True)
class Phase21Config:
    """Single source of defaults for the snap90--99 event experiment."""

    base_path: str = "/public/share/chenhouzun/TNG50-1/output/"
    snap_start: int = 90
    snap_event_prev: int = 94
    snap_event_cur: int = 95
    snap_selection: int = 99
    snap_end: int = 99

    log_mstar_min: float = 8.5
    log_mstar_max: float = 11.5
    mass_bin_width: float = 0.25
    sample_per_bin: int = 50
    random_seed: int = 202608
    minimum_bin_warning_count: int = 15

    h: float = 0.6774
    omega_m: float = 0.3089
    omega_lambda: float = 0.6911
    central_rfrac: float = 0.1
    hot_log10_temperature_min: float = 4.5
    cold_log10_temperature_max: float = 4.5
    mean_molecular_weight: float = 0.59
    virial_temperature_factor: float = 35.9
    tracer_mass_1e10_msun_h: float = 5.73879e-6

    tracer_read_block_size: int = 2_000_000
    tracer_workers: int = 64
    tracer_parallel_backend: str = "process"
    # Four shared-memory workers overlap independent halo reads while using a
    # single snapshot-level catalogue cache.  Set workers=1/backend=serial on
    # filesystems where concurrent HDF5 reads are undesirable.
    state_workers: int = 1
    state_parallel_backend: str = "thread"
    cache_schema_version: int = 1

    # These strings are part of the cache contract.  Keeping them in the
    # configuration makes a cache invalid when a scientific definition is
    # changed, rather than silently reusing a result made with another rule.
    state_definition_version: str = "host-radial-phase-carrier-v1"
    other_rule_version: str = "exclude-from-valid-mother-retain-total-v1"
    classification_rule_version: str = "literal-readme-6.2-v1"

    def __post_init__(self) -> None:
        """Reject configurations that violate the experiment contract."""

        if (self.snap_start, self.snap_event_prev, self.snap_event_cur, self.snap_end, self.snap_selection) != (90, 94, 95, 99, 99):
            raise ValueError("Phase 2.1 requires the fixed snap90--99 history and snap94 -> snap95 anchors")
        if self.snap_event_cur - self.snap_event_prev != 1:
            raise ValueError("snap94 -> snap95 must be adjacent")
        if self.log_mstar_max <= self.log_mstar_min:
            raise ValueError("Invalid stellar-mass range")
        if self.mass_bin_width <= 0:
            raise ValueError("mass_bin_width must be positive")
        if not (np.isclose(self.log_mstar_min, 8.5) and np.isclose(self.log_mstar_max, 11.5) and np.isclose(self.mass_bin_width, 0.25)):
            raise ValueError("Phase 2.1 uses fixed 8.5--11.5 dex bins of width 0.25")
        if self.sample_per_bin < 1:
            raise ValueError("sample_per_bin must be positive")
        if self.minimum_bin_warning_count < 1:
            raise ValueError("minimum_bin_warning_count must be positive")
        if self.h <= 0 or self.tracer_mass_1e10_msun_h <= 0:
            raise ValueError("h and tracer mass must be positive")
        if not 0 < self.central_rfrac < 1:
            raise ValueError("central_rfrac must be between zero and one")
        if self.hot_log10_temperature_min != self.cold_log10_temperature_max:
            raise ValueError("Hot and cold boundaries must match")
        if self.tracer_workers != 64:
            raise ValueError("Phase 2.1 tracer_workers is fixed at 64")
        if self.tracer_read_block_size < 1:
            raise ValueError("tracer_read_block_size must be positive")
        if self.state_workers < 1:
            raise ValueError("state_workers must be positive")
        if self.state_parallel_backend not in {"serial", "thread", "process"}:
            raise ValueError("Invalid state_parallel_backend")

    @property
    def mass_bin_edges(self) -> np.ndarray:
        """Return the ten 0.25-dex bins plus the final 0.5-dex bin."""

        low_edges = np.arange(8.5, 11.000001, 0.25, dtype=float)
        return np.concatenate((low_edges, np.array([11.5], dtype=float)))

    @property
    def mass_bin_centers(self) -> np.ndarray:
        """Return arithmetic centers of the final bins."""

        edges = self.mass_bin_edges
        return 0.5 * (edges[:-1] + edges[1:])

    @property
    def tracer_mass_msun(self) -> float:
        """Convert the TNG tracer mass to solar masses."""

        return self.tracer_mass_1e10_msun_h * 1.0e10 / self.h

    @property
    def snapshots(self) -> tuple[int, ...]:
        """Return the complete state-history snapshot sequence."""

        return tuple(range(self.snap_start, self.snap_end + 1))

    @property
    def fingerprint_payload(self) -> dict:
        """Return the scientific inputs that determine reusable caches.

        The payload deliberately includes definitions as well as numerical
        settings.  A cache made before an ``other`` or state-rule change must
        therefore never be mistaken for a result from the current analysis.
        """

        return {
            "schema_version": int(self.cache_schema_version),
            "simulation": "TNG50-1",
            "base_path": str(self.base_path),
            "snapshots": list(self.snapshots),
            "event_anchor": [int(self.snap_event_prev), int(self.snap_event_cur)],
            "mass_range": [float(self.log_mstar_min), float(self.log_mstar_max)],
            "mass_bin_width": float(self.mass_bin_width),
            "mass_bin_edges": self.mass_bin_edges.tolist(),
            "sample_per_bin": int(self.sample_per_bin),
            "random_seed": int(self.random_seed),
            "state_definition": self.state_definition_version,
            "state_definition_parameters": {
                "central_rfrac": float(self.central_rfrac),
                "cold_log10_temperature_max": float(self.cold_log10_temperature_max),
                "hot_log10_temperature_min": float(self.hot_log10_temperature_min),
                "carrier_states": ["gas", "star", "wind", "black_hole", "unresolved"],
                "host_states": ["main_central", "satellite", "other_halo", "unbound", "unresolved"],
                "radial_states": ["inner", "outer_halo", "beyond_r200c", "unresolved"],
                "phase_states": ["cold", "hot", "unresolved"],
            },
            "other_rule": self.other_rule_version,
            "classification_rule": self.classification_rule_version,
            "tracer_mass_1e10_msun_h": float(self.tracer_mass_1e10_msun_h),
            "tracer_weight_msun": float(self.tracer_mass_msun),
            "temperature_boundary_log10_K": float(self.hot_log10_temperature_min),
            "central_rfrac": float(self.central_rfrac),
        }

    @property
    def fingerprint(self) -> str:
        """Return a stable SHA-256 fingerprint for scientific caches."""

        encoded = json.dumps(self.fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
