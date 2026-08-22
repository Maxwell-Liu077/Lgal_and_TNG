"""Cosmological time conversions used by the Phase 1.8 workflow.

This module contains only deterministic numerical transformations.  It does
not read simulation data or write analysis products, which keeps the physics
calculation reusable from notebooks, tests, and batch workflows.
"""

from __future__ import annotations

import numpy as np


def cosmic_age_gyr_from_scale_factor(
    scale_factor: float,
    *,
    h: float,
    omega_m: float,
    omega_lambda: float,
) -> float:
    """Age of a flat matter+Lambda universe at scale factor ``a``."""

    if scale_factor <= 0:
        raise ValueError("scale_factor must be positive")
    if h <= 0 or omega_m <= 0 or omega_lambda <= 0:
        raise ValueError("Cosmology parameters must be positive")
    h0_gyr = 100.0 * h * 1.022712e-3
    prefactor = 2.0 / (3.0 * h0_gyr * np.sqrt(omega_lambda))
    argument = np.sqrt(omega_lambda / omega_m) * scale_factor**1.5
    return float(prefactor * np.arcsinh(argument))


def snapshot_interval_gyr(
    header_prev: dict,
    header_cur: dict,
    *,
    h: float,
    omega_m: float,
    omega_lambda: float,
) -> float:
    """Return the positive cosmic-time interval between two snapshot headers.

    The headers supply scale factors through ``Time``. Cosmological parameters
    remain explicit so the function is deterministic and independently
    testable.
    """

    age_prev = cosmic_age_gyr_from_scale_factor(
        float(header_prev["Time"]),
        h=h,
        omega_m=omega_m,
        omega_lambda=omega_lambda,
    )
    age_cur = cosmic_age_gyr_from_scale_factor(
        float(header_cur["Time"]),
        h=h,
        omega_m=omega_m,
        omega_lambda=omega_lambda,
    )
    dt_gyr = age_cur - age_prev
    if dt_gyr <= 0:
        raise ValueError("Snapshot interval is not positive")
    return float(dt_gyr)
