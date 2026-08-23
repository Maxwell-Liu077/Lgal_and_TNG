"""Command-line entry point for the independent Phase 2.1 workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.pipeline import run_phase21
from src.io.results import save_phase21_result
from src.physics.cooling_function import LgalCoolingFunction
from src.plotting import plot_agn_cooling, plot_composition, plot_feedback_before_accretion
from src.utils.config import Phase21Config


def _parser() -> argparse.ArgumentParser:
    """Build the non-I/O CLI parser."""

    parser = argparse.ArgumentParser(description="Run the Phase 2.1 AGN/cold-reservoir analysis.")
    parser.add_argument("--base-path", default=Phase21Config.base_path, help="External TNG50-1 output directory")
    parser.add_argument("--cooling-table-dir", type=Path, default=PROJECT_ROOT / "data" / "external" / "cooling_tables")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "data" / "interim" / "cache")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--figure-dir", type=Path, default=PROJECT_ROOT / "results" / "figures")
    parser.add_argument("--state-workers", type=int, default=Phase21Config.state_workers, help="Concurrent halo-state workers")
    parser.add_argument("--state-backend", choices=("serial", "thread", "process"), default=Phase21Config.state_parallel_backend, help="Halo-state concurrency backend")
    parser.add_argument("--rebuild-sample", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run Phase 2.1, and serialize its result."""

    args = _parser().parse_args(argv)
    cooling = LgalCoolingFunction.from_directory(args.cooling_table_dir)
    config = Phase21Config(
        base_path=args.base_path,
        state_workers=args.state_workers,
        state_parallel_backend=args.state_backend,
    )
    result = run_phase21(cooling, config=config, cache_dir=args.cache_dir, rebuild_sample=args.rebuild_sample, verbose=not args.quiet)
    prefix = args.output_dir / "phase21_snap090_099_seed202608"
    for label, path in save_phase21_result(result, prefix).items():
        if not args.quiet:
            print(f"{label}: {path}")
    try:
        figure_dir = args.figure_dir
        figure_dir.mkdir(parents=True, exist_ok=True)
        plot_agn_cooling(result["agn_quartile_statistics"], halo_results=result["halo_results"], save_path=figure_dir / "phase21_mdot_heat_h15_quartile_cooling.png")
        plot_composition(result["composition_statistics"], direction="in", save_path=figure_dir / "phase21_supply_composition.png")
        plot_composition(result["composition_statistics"], direction="out", save_path=figure_dir / "phase21_feedback_composition.png")
        plot_feedback_before_accretion(result["normalized_statistics"], halo_results=result["halo_results"], save_path=figure_dir / "phase21_feedback_before_accretion.png")
    except ImportError as exc:
        if not args.quiet:
            print(f"[plotting] skipped: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
