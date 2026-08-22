"""Command-line entry point for the unified Phase 1.8 workflows.

The script owns runtime paths and variant selection.  Scientific calculations
remain in ``src`` so notebooks, batch jobs, and tests reuse the same code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.inward_delivery_pipeline import (
    run_phase18 as run_inward_delivery,
)
from src.analysis.pipeline import run_phase18
from src.io.results import save_phase18_result
from src.physics.cooling_function import (
    LGAL_TABLE_FILENAMES,
    LgalCoolingFunction,
    download_lgal_cooling_tables,
)
from src.utils.config import Phase18Config


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser without performing any I/O."""

    parser = argparse.ArgumentParser(
        description="Run the baseline or inward-delivery Phase 1.8 analysis."
    )
    parser.add_argument(
        "--variant",
        choices=("baseline", "inward-delivery"),
        default="baseline",
        help="Select the historical 1.8 result or the integrated 1.8.1 extension.",
    )
    parser.add_argument(
        "--base-path",
        default=Phase18Config.base_path,
        help="Path to the external TNG50-1 output directory.",
    )
    parser.add_argument(
        "--cooling-table-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "external" / "cooling_tables",
        help="Directory containing the immutable L-Galaxies cooling tables.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "cache",
        help="Directory for resumable sample, halo-state, and chunk caches.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Directory for JSON, CSV, and NPZ scientific products.",
    )
    parser.add_argument(
        "--download-cooling-tables",
        action="store_true",
        help="Download missing Henriques-2015 cooling tables.",
    )
    parser.add_argument(
        "--rebuild-sample",
        action="store_true",
        help="Ignore the cached sample catalog and rebuild it.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress pipeline progress messages.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the selected workflow and save its unchanged product set."""

    args = _parser().parse_args(argv)
    missing = [
        name
        for name in LGAL_TABLE_FILENAMES
        if not (args.cooling_table_dir / name).is_file()
    ]
    if missing and args.download_cooling_tables:
        download_lgal_cooling_tables(args.cooling_table_dir)
    elif missing:
        raise FileNotFoundError(
            "Missing cooling tables: "
            + ", ".join(missing)
            + ". Use --download-cooling-tables to fetch them."
        )

    config = Phase18Config(base_path=args.base_path)
    cooling_function = LgalCoolingFunction.from_directory(
        args.cooling_table_dir
    )
    if args.variant == "baseline":
        runner = run_phase18
        prefix_name = "phase18_slow_mstar8p5-11p5_snap098_099"
    else:
        runner = run_inward_delivery
        prefix_name = "phase181_slow_mstar8p5-11p5_snap098_099"

    result = runner(
        cooling_function,
        config=config,
        cache_dir=args.cache_dir,
        rebuild_sample=args.rebuild_sample,
        verbose=not args.quiet,
    )
    saved = save_phase18_result(result, args.output_dir / prefix_name)
    for label, path in saved.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
