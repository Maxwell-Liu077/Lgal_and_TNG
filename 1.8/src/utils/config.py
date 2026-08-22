"""Validated configuration for the Phase 1.8 scientific workflow.

The dataclass intentionally retains every historical default, including the
fixed 64-process tracer scan.  Keeping validation here makes the same contract
available to notebooks, batch jobs, and synthetic tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Phase18Config:
    """Configuration for the z=0 mass-dependence experiment."""

    base_path: str = "/public/share/chenhouzun/TNG50-1/output/"
    snap_prev: int = 98
    snap_cur: int = 99

    log_mstar_min: float = 8.5
    log_mstar_max: float = 11.5
    mass_bin_width: float = 0.25
    sample_per_bin: int = 50
    minimum_bin_warning_count: int = 15
    random_seed: int = 42

    h: float = 0.6774
    omega_m: float = 0.3089
    omega_lambda: float = 0.6911

    central_rfrac: float = 0.1
    outer_rmax_rfrac: float = 1.0
    hot_log10_temperature_min: float = 4.5
    cold_log10_temperature_max: float = 4.5

    mean_molecular_weight: float = 0.59
    virial_temperature_factor: float = 35.9
    tracer_mass_1e10_msun_h: float = 5.73879e-6

    sn_epsilon_disk: float = 2.6
    sn_v_reheat_km_s: float = 480.0
    sn_beta_disk: float = 0.72
    sn_eta_halo: float = 0.62
    sn_v_eject_km_s: float = 100.0
    sn_beta_halo: float = 0.80
    sn_v_sn_km_s: float = 630.0

    tracer_read_block_size: int = 2_000_000
    tracer_workers: int = 64
    tracer_parallel_backend: str = "process"
    mpb_workers: int = 8
    mpb_parallel_backend: str = "process"
    state_workers: int = 1
    state_parallel_backend: str = "serial"

    def __post_init__(self) -> None:
        """Reject configurations that would change the Phase 1.8 design.

        Validation is intentionally strict: adjacent snapshots, fixed arrival
        at snapshot 99, matched hot/cold thresholds, and the 64-process tracer
        backend are part of the reproducibility contract rather than optional
        performance settings.
        """

        if self.snap_cur != self.snap_prev + 1:
            raise ValueError("Phase 1.8 requires adjacent snapshots")
        if self.snap_cur != 99:
            raise ValueError("Phase 1.8 is defined for arrival at snapshot 99")
        if self.log_mstar_max <= self.log_mstar_min:
            raise ValueError("Invalid stellar-mass range")
        if self.mass_bin_width <= 0:
            raise ValueError("mass_bin_width must be positive")
        n_bins = (
            self.log_mstar_max - self.log_mstar_min
        ) / self.mass_bin_width
        if not np.isclose(n_bins, round(n_bins), rtol=0, atol=1e-10):
            raise ValueError("Mass range must be divisible by mass_bin_width")
        if self.sample_per_bin < 1:
            raise ValueError("sample_per_bin must be positive")
        if self.minimum_bin_warning_count < 1:
            raise ValueError("minimum_bin_warning_count must be positive")
        if self.h <= 0:
            raise ValueError("h must be positive")
        if not 0 < self.central_rfrac < self.outer_rmax_rfrac <= 1:
            raise ValueError(
                "Require 0 < central_rfrac < outer_rmax_rfrac <= 1"
            )
        if self.hot_log10_temperature_min != (
            self.cold_log10_temperature_max
        ):
            raise ValueError(
                "Hot and cold temperature boundaries must be identical"
            )
        if self.tracer_mass_1e10_msun_h <= 0:
            raise ValueError("tracer mass must be positive")
        sn_positive = (
            self.sn_epsilon_disk,
            self.sn_v_reheat_km_s,
            self.sn_beta_disk,
            self.sn_eta_halo,
            self.sn_v_eject_km_s,
            self.sn_beta_halo,
            self.sn_v_sn_km_s,
        )
        if not all(value > 0 for value in sn_positive):
            raise ValueError("SN feedback parameters must be positive")
        if self.tracer_read_block_size < 1:
            raise ValueError("tracer_read_block_size must be positive")
        if self.tracer_workers != 64:
            raise ValueError("Phase 1.8 tracer_workers is fixed at 64")
        if self.mpb_workers < 1 or self.state_workers < 1:
            raise ValueError("Worker counts must be positive")
        valid_backends = {"serial", "thread", "process"}
        if self.tracer_parallel_backend != "process":
            raise ValueError(
                "Phase 1.8 tracer_parallel_backend is fixed at process"
            )
        if self.mpb_parallel_backend not in valid_backends:
            raise ValueError("Invalid mpb_parallel_backend")
        if self.state_parallel_backend not in valid_backends:
            raise ValueError("Invalid state_parallel_backend")

    @property
    def mass_bin_edges(self) -> np.ndarray:
        """Return inclusive stellar-mass bin edges in log10 solar masses."""

        n_bins = int(
            round(
                (self.log_mstar_max - self.log_mstar_min)
                / self.mass_bin_width
            )
        )
        return np.linspace(
            self.log_mstar_min,
            self.log_mstar_max,
            n_bins + 1,
            dtype=float,
        )

    @property
    def mass_bin_centers(self) -> np.ndarray:
        """Return arithmetic centers of the configured logarithmic bins."""

        edges = self.mass_bin_edges
        return 0.5 * (edges[:-1] + edges[1:])

    @property
    def tracer_mass_msun(self) -> float:
        """Convert the stored TNG tracer mass from 1e10 Msun/h to Msun."""

        return self.tracer_mass_1e10_msun_h * 1.0e10 / self.h
