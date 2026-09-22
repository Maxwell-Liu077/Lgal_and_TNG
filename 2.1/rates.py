"""Rate-level helpers shared by event evaluation and diagnostics."""

from __future__ import annotations

import numpy as np


def rate_from_mass(mass_msun: float, dt_gyr: float) -> float:
    """Convert an event mass to Msun/yr using the anchor interval."""

    if mass_msun < 0 or dt_gyr <= 0:
        raise ValueError("Mass must be non-negative and dt positive")
    return float(mass_msun / (dt_gyr * 1.0e9))


def closure_error(total: float, components: list[float]) -> float:
    """Return a numerically stable additive closure residual."""

    return float(total - np.sum(np.asarray(components, dtype=float)))
