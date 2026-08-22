# Galaxy Gas Accretion — Phase 1.8

This project contains one integrated codebase for the historical Phase 1.8
baseline analysis and its inward-delivery extension (formerly stored as
Phase 1.8.1). Shared physics, TNG access, caches, statistics, serialization,
and plotting code now have a single implementation.

The refactor changes organization only. Historical scientific definitions,
function names, cache fingerprints, result keys, field order, file formats,
plot series, and the fixed 64-process tracer scan are retained.

## Project structure

```text
1.8/
├── README.md
├── environment.yml
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                    # External TNG location documentation
│   ├── external/               # Immutable L-Galaxies cooling tables
│   ├── interim/                # Resumable sample/state/tracer caches
│   └── processed/              # JSON, CSV, and NPZ scientific products
│
├── notebooks/
│   └── 01_phase18_analysis.ipynb
│
├── src/
│   ├── io/                     # TNG readers, caches, tracer scans, writers
│   ├── physics/                # Cooling and SAM equations
│   ├── analysis/               # Rates, statistics, and workflow assembly
│   ├── plotting/               # Plot-only modules; no raw-data access
│   └── utils/                  # Configuration, cosmology, array helpers
│
├── scripts/
│   └── run_analysis.py         # Unified baseline/extension CLI
├── configs/                    # Configuration policy documentation
├── results/
│   ├── figures/                # PNG and PDF figures only
│   ├── tables/                 # Publication-ready tables only
│   └── logs/                   # Batch logs only
├── paper/                      # Methodology and manuscript material
└── tests/                      # Flat synthetic regression suite
```

No Python compatibility facades remain at the project root, and there are no
separate `phase18` or `phase181` source packages.

## Integrated workflows

The baseline entry point retains its historical name and signature:

```python
from src.analysis.pipeline import run_phase18
```

The inward-delivery extension is part of the same package and also retains its
historical function name inside its focused workflow module:

```python
from src.analysis.inward_delivery_pipeline import run_phase18
```

The extension reuses the baseline configuration, cooling physics, sample,
endpoint state cache, low-level tracer scanner, result writer, statistical
implementation, and plotting package. Its dedicated modules contain only the
additional snapshot-99 parent query and inner-hot delivery calculation needed
to preserve its historical outputs.

## Plotting organization

Plotting is deliberately separated by figure responsibility:

- `src/plotting/style.py`: academic style and PNG/PDF serialization.
- `src/plotting/scaling_relation.py`: mass-dependent and normalized curves.
- `src/plotting/comparison.py`: reheating and inner-hot comparisons.
- `src/plotting/tracer_history.py`: complete origin/fate heatmaps.

Notebook-only AGN, supply-variant, bootstrap, and grouped-history calculations
have also been moved into focused `src/io`, `src/physics`, and `src/analysis`
modules. The notebook now configures workflows and displays returned results;
it no longer defines reusable analysis or plotting functions.

All public plotting functions are re-exported from `src.plotting`. They accept
processed arrays only and never read snapshots, catalogs, or caches.

## Running the analysis

The merged notebook contains the baseline section followed by the integrated
inward-delivery section:

```text
notebooks/01_phase18_analysis.ipynb
```

For batch use:

```powershell
python scripts\run_analysis.py --variant baseline
python scripts\run_analysis.py --variant inward-delivery
```

Use `--help` for TNG, cache, cooling-table, and output path options. Raw TNG50-1
snapshots remain external and read-only; `illustris_python` must be supplied by
the execution environment.

## Artifact boundaries

- `data/raw`: documentation for raw simulation inputs; no copied snapshots.
- `data/external`: immutable third-party scientific tables.
- `data/interim`: restartable caches that may be regenerated.
- `data/processed`: final numerical products only (`.json`, `.csv`, `.npz`).
- `results/figures`: generated figures only (`.png`, `.pdf`).
- `paper`: methodology and manuscript material; no runtime caches.

Historical `phase18_` and `phase181_` filename prefixes remain unchanged so
existing products are identifiable after their directories are merged.

## Regression checks

Run from this directory:

```powershell
python -m pytest -q
python -m compileall -q src scripts tests
python scripts\run_analysis.py --help
```

These checks use synthetic inputs. A real scientific run still requires the
external TNG50-1 data path, `illustris_python`, and `h5py`.
