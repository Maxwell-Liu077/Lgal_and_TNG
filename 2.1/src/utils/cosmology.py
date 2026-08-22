"""Small cosmology helpers used by the Phase 2.1 rate definitions."""

from __future__ import annotations

import numpy as np


def cosmic_age_gyr_from_scale_factor(
    scale_factor: float,
    *,
    h: float = 0.6774,
    omega_m: float = 0.3089,
    omega_lambda: float = 0.6911,
) -> float:
    """Return the flat matter+Lambda cosmic age in Gyr."""

    if scale_factor <= 0 or h <= 0 or omega_m <= 0 or omega_lambda <= 0:
        raise ValueError("Invalid cosmological parameters")
    hubble_time_gyr = 9.778 / h
    argument = np.sqrt(omega_lambda / omega_m) * scale_factor ** 1.5
    return float(
        2.0
        / (3.0 * np.sqrt(omega_lambda))
        * np.arcsinh(argument)
        * hubble_time_gyr
    )


def snapshot_interval_gyr(
    header_prev: dict,
    header_cur: dict,
    *,
    h: float = 0.6774,
    omega_m: float = 0.3089,
    omega_lambda: float = 0.6911,
) -> float:
    """Compute a positive interval from two TNG snapshot headers."""

    previous = cosmic_age_gyr_from_scale_factor(
        float(header_prev["Time"]), h=h, omega_m=omega_m, omega_lambda=omega_lambda
    )
    current = cosmic_age_gyr_from_scale_factor(
        float(header_cur["Time"]), h=h, omega_m=omega_m, omega_lambda=omega_lambda
    )
    interval = current - previous
    if interval <= 0:
        raise ValueError("Snapshot interval must be positive")
    return float(interval)
