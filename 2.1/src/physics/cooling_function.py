"""Henriques-2015 cooling table loading and interpolation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


LGAL_TABLE_FILENAMES = (
    "stripped_mzero.cie", "stripped_m-30.cie", "stripped_m-20.cie",
    "stripped_m-15.cie", "stripped_m-10.cie", "stripped_m-05.cie",
    "stripped_m-00.cie", "stripped_m+05.cie",
)
LGAL_LOG10_Z_OVER_ZSUN = np.array([-5.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5])


@dataclass(frozen=True)
class ConstantCoolingFunction:
    """Positive constant cooling law for synthetic runs."""

    lambda_erg_cm3_s: float = 1.0e-23

    def __post_init__(self) -> None:
        """Validate the coefficient."""

        if self.lambda_erg_cm3_s <= 0:
            raise ValueError("lambda_erg_cm3_s must be positive")

    def __call__(self, temperature_k, metallicity_mass_fraction):
        """Return the constant coefficient with broadcast shape."""

        temperature, _ = np.broadcast_arrays(temperature_k, metallicity_mass_fraction)
        result = np.full(temperature.shape, self.lambda_erg_cm3_s, dtype=float)
        return float(result) if result.ndim == 0 else result


@dataclass(frozen=True)
class LgalCoolingFunction:
    """Bilinear log-space interpolation over Henriques cooling tables."""

    log10_temperature_grid: np.ndarray
    log10_metallicity_grid: np.ndarray
    log10_lambda_grid: np.ndarray
    source_directory: str = ""

    def __post_init__(self) -> None:
        """Validate interpolation grids."""

        t = np.asarray(self.log10_temperature_grid, dtype=float)
        z = np.asarray(self.log10_metallicity_grid, dtype=float)
        values = np.asarray(self.log10_lambda_grid, dtype=float)
        if t.ndim != 1 or z.ndim != 1 or values.shape != (len(z), len(t)):
            raise ValueError("Invalid cooling table dimensions")
        if np.any(np.diff(t) <= 0) or np.any(np.diff(z) <= 0) or not np.all(np.isfinite(values)):
            raise ValueError("Cooling grids must be increasing and finite")
        object.__setattr__(self, "log10_temperature_grid", t)
        object.__setattr__(self, "log10_metallicity_grid", z)
        object.__setattr__(self, "log10_lambda_grid", values)

    @classmethod
    def from_directory(cls, directory: str | Path) -> "LgalCoolingFunction":
        """Load and cross-check the eight published tables."""

        directory = Path(directory)
        missing = [name for name in LGAL_TABLE_FILENAMES if not (directory / name).is_file()]
        if missing:
            raise FileNotFoundError("Missing cooling tables: " + ", ".join(missing))
        temperature = None
        rows = []
        for name in LGAL_TABLE_FILENAMES:
            values = np.loadtxt(directory / name, dtype=float)
            if values.ndim != 2 or values.shape[1] < 6 or len(values) != 91:
                raise ValueError(f"Unexpected cooling table: {name}")
            current = values[:, 0]
            if temperature is None:
                temperature = current
            elif not np.allclose(current, temperature, rtol=0, atol=1e-8):
                raise ValueError("Cooling tables use different temperature grids")
            rows.append(values[:, 5])
        log_z = LGAL_LOG10_Z_OVER_ZSUN + np.log10(0.02)
        return cls(np.asarray(temperature), log_z, np.asarray(rows), str(directory.resolve()))

    def __call__(self, temperature_k, metallicity_mass_fraction):
        """Evaluate cooling efficiency in erg cm^-3 s^-1."""

        temperature, metallicity = np.broadcast_arrays(
            np.asarray(temperature_k, dtype=float),
            np.asarray(metallicity_mass_fraction, dtype=float),
        )
        if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0):
            raise ValueError("temperature must be finite and positive")
        log_t = np.clip(np.log10(temperature).ravel(), self.log10_temperature_grid[0], self.log10_temperature_grid[-1])
        with np.errstate(divide="ignore", invalid="ignore"):
            log_z = np.where(np.isfinite(metallicity) & (metallicity > 0), np.log10(metallicity), self.log10_metallicity_grid[0]).ravel()
        log_z = np.clip(log_z, self.log10_metallicity_grid[0], self.log10_metallicity_grid[-1])
        at_t = np.vstack([np.interp(log_t, self.log10_temperature_grid, row) for row in self.log10_lambda_grid])
        upper = np.clip(np.searchsorted(self.log10_metallicity_grid, log_z, side="right"), 1, len(self.log10_metallicity_grid) - 1)
        lower = upper - 1
        columns = np.arange(len(log_z))
        fraction = (log_z - self.log10_metallicity_grid[lower]) / (self.log10_metallicity_grid[upper] - self.log10_metallicity_grid[lower])
        log_lambda = at_t[lower, columns] + fraction * (at_t[upper, columns] - at_t[lower, columns])
        result = np.power(10.0, log_lambda).reshape(temperature.shape)
        return float(result) if result.ndim == 0 else result
