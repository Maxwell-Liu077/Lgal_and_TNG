"""Henriques radio-mode feedback diagnostic used by the baseline notebook."""

from __future__ import annotations

import numpy as np


# Henriques et al. (2015), equations S24-S26 and Table S1.
K_AGN_MSUN_PER_YR = 5.3e-3
RADIO_EFFICIENCY = 0.1
LIGHT_SPEED_CM_S = 2.99792458e10
MSUN_G = 1.98847e33
SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0


def apply_henriques_radio_mode(
    cooling_rate: np.ndarray,
    m_hot: np.ndarray,
    m_bh: np.ndarray,
    v200c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the vectorized Henriques S24-S26 radio-mode correction."""

    cooling_rate = np.asarray(cooling_rate, dtype=float)
    m_hot = np.asarray(m_hot, dtype=float)
    m_bh = np.asarray(m_bh, dtype=float)
    v200c = np.asarray(v200c, dtype=float)
    effective = np.full(cooling_rate.shape, np.nan, dtype=float)
    suppression = np.full(cooling_rate.shape, np.nan, dtype=float)
    valid = (
        np.isfinite(cooling_rate)
        & np.isfinite(m_hot)
        & np.isfinite(m_bh)
        & np.isfinite(v200c)
        & (cooling_rate >= 0)
        & (m_hot >= 0)
        & (m_bh >= 0)
        & (v200c > 0)
    )
    radio_mdot = (
        K_AGN_MSUN_PER_YR
        * (m_hot[valid] / 1.0e11)
        * (m_bh[valid] / 1.0e8)
    )
    radio_power = (
        RADIO_EFFICIENCY
        * radio_mdot
        * MSUN_G
        / SECONDS_PER_YEAR
        * LIGHT_SPEED_CM_S**2
    )
    suppression_valid = (
        2.0
        * radio_power
        / (v200c[valid] * 1.0e5) ** 2
        / MSUN_G
        * SECONDS_PER_YEAR
    )
    suppression[valid] = suppression_valid
    effective[valid] = np.maximum(
        cooling_rate[valid] - suppression_valid,
        0.0,
    )
    return effective, suppression
