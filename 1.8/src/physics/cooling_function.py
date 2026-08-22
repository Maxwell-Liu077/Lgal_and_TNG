"""L-Galaxies cooling-table download and interpolation utilities.

Cooling tables are immutable reference inputs.  This module only downloads or
evaluates those tables; workflow code decides where reference data live.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np


LGAL_TABLE_FILENAMES = (
    "stripped_mzero.cie",
    "stripped_m-30.cie",
    "stripped_m-20.cie",
    "stripped_m-15.cie",
    "stripped_m-10.cie",
    "stripped_m-05.cie",
    "stripped_m-00.cie",
    "stripped_m+05.cie",
)

LGAL_LOG10_Z_OVER_ZSUN = np.array(
    [-5.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5],
    dtype=float,
)
LGAL_ZSUN_MASS_FRACTION = 0.02
LGAL_RAW_BASE_URL = (
    "https://raw.githubusercontent.com/"
    "LGalaxiesPublicRelease/LGalaxies_PublicRepository/"
    "refs/tags/Henriques2015/CoolFunctions"
)


def download_lgal_cooling_tables(
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Download cooling tables pinned to the Henriques-2015 release."""

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename in LGAL_TABLE_FILENAMES:
        path = destination / filename
        paths.append(path)
        if path.is_file() and not overwrite:
            continue
        url = f"{LGAL_RAW_BASE_URL}/{quote(filename, safe='')}"
        temporary = path.with_suffix(path.suffix + ".part")
        try:
            with urlopen(url, timeout=60) as response:
                payload = response.read()
            temporary.write_bytes(payload)
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
    return tuple(paths)


@dataclass(frozen=True)
class LgalCoolingFunction:
    """Bilinear interpolator over the eight Henriques cooling tables.

    Temperature and metallicity coordinates are interpolated in log space;
    finite out-of-grid values are clipped to the published table boundaries,
    matching the historical implementation.
    """

    log10_temperature_grid: np.ndarray
    log10_metallicity_grid: np.ndarray
    log10_lambda_grid: np.ndarray
    source_directory: str = ""

    def __post_init__(self) -> None:
        """Validate grid dimensionality, ordering, shape, and finiteness."""

        log_t = np.asarray(self.log10_temperature_grid, dtype=float)
        log_z = np.asarray(self.log10_metallicity_grid, dtype=float)
        log_lambda = np.asarray(self.log10_lambda_grid, dtype=float)
        if log_t.ndim != 1 or log_z.ndim != 1:
            raise ValueError("Cooling grids must be one-dimensional")
        if log_lambda.shape != (len(log_z), len(log_t)):
            raise ValueError("Cooling table has an invalid shape")
        if np.any(np.diff(log_t) <= 0) or np.any(np.diff(log_z) <= 0):
            raise ValueError("Cooling grids must be increasing")
        if not np.all(np.isfinite(log_lambda)):
            raise ValueError("Cooling table contains non-finite values")
        object.__setattr__(self, "log10_temperature_grid", log_t)
        object.__setattr__(self, "log10_metallicity_grid", log_z)
        object.__setattr__(self, "log10_lambda_grid", log_lambda)

    @classmethod
    def from_directory(cls, directory: str | Path) -> "LgalCoolingFunction":
        """Load and cross-check all required cooling tables from a directory."""

        directory = Path(directory)
        missing = [
            filename
            for filename in LGAL_TABLE_FILENAMES
            if not (directory / filename).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Cooling tables missing from {directory}: "
                + ", ".join(missing)
            )
        temperature_grid = None
        rows = []
        for filename in LGAL_TABLE_FILENAMES:
            values = np.loadtxt(directory / filename, dtype=float)
            if values.ndim != 2 or values.shape[1] < 6 or len(values) != 91:
                raise ValueError(f"Unexpected table: {directory / filename}")
            current_temperature = values[:, 0]
            if temperature_grid is None:
                temperature_grid = current_temperature
            elif not np.allclose(
                current_temperature,
                temperature_grid,
                rtol=0,
                atol=1e-8,
            ):
                raise ValueError("Cooling tables use different T grids")
            rows.append(values[:, 5])
        log_z = (
            LGAL_LOG10_Z_OVER_ZSUN
            + np.log10(LGAL_ZSUN_MASS_FRACTION)
        )
        return cls(
            log10_temperature_grid=np.asarray(temperature_grid),
            log10_metallicity_grid=log_z,
            log10_lambda_grid=np.asarray(rows),
            source_directory=str(directory.resolve()),
        )

    def __call__(
        self,
        temperature_k: float | np.ndarray,
        metallicity_mass_fraction: float | np.ndarray,
    ) -> float | np.ndarray:
        """Evaluate cooling efficiency for broadcast temperature/metallicity.

        Scalar inputs return a Python ``float``; array inputs retain their
        broadcast shape. Non-positive temperatures remain explicit errors.
        """

        temperature = np.asarray(temperature_k, dtype=float)
        metallicity = np.asarray(metallicity_mass_fraction, dtype=float)
        temperature, metallicity = np.broadcast_arrays(
            temperature,
            metallicity,
        )
        if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0):
            raise ValueError("temperature_k must be finite and positive")

        log_t = np.clip(
            np.log10(temperature).ravel(),
            self.log10_temperature_grid[0],
            self.log10_temperature_grid[-1],
        )
        min_log_z = self.log10_metallicity_grid[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            log_z = np.where(
                np.isfinite(metallicity) & (metallicity > 0),
                np.log10(metallicity),
                min_log_z,
            ).ravel()
        log_z = np.clip(
            log_z,
            self.log10_metallicity_grid[0],
            self.log10_metallicity_grid[-1],
        )
        rates_at_temperature = np.vstack(
            [
                np.interp(log_t, self.log10_temperature_grid, row)
                for row in self.log10_lambda_grid
            ]
        )
        upper = np.searchsorted(
            self.log10_metallicity_grid,
            log_z,
            side="right",
        )
        upper = np.clip(upper, 1, len(self.log10_metallicity_grid) - 1)
        lower = upper - 1
        column = np.arange(len(log_z))
        fraction = (
            (log_z - self.log10_metallicity_grid[lower])
            / (
                self.log10_metallicity_grid[upper]
                - self.log10_metallicity_grid[lower]
            )
        )
        log_lambda = (
            rates_at_temperature[lower, column]
            + fraction
            * (
                rates_at_temperature[upper, column]
                - rates_at_temperature[lower, column]
            )
        )
        result = np.power(10.0, log_lambda).reshape(temperature.shape)
        return float(result) if result.ndim == 0 else result


@dataclass(frozen=True)
class ConstantCoolingFunction:
    """Positive constant cooling law used by deterministic synthetic tests."""

    lambda_erg_cm3_s: float

    def __post_init__(self) -> None:
        """Require a physically positive cooling coefficient."""

        if self.lambda_erg_cm3_s <= 0:
            raise ValueError("lambda_erg_cm3_s must be positive")

    def __call__(self, temperature_k, metallicity_mass_fraction):
        """Return the constant coefficient with the broadcast input shape."""

        temperature, metallicity = np.broadcast_arrays(
            np.asarray(temperature_k, dtype=float),
            np.asarray(metallicity_mass_fraction, dtype=float),
        )
        result = np.full(temperature.shape, self.lambda_erg_cm3_s)
        return float(result) if result.ndim == 0 else result
