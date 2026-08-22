# Phase 2.1 notebooks

`phase21_end_to_end.ipynb` is the top-level notebook caller. It imports the
public `Phase21Config`, `run_phase21`, and `save_phase21_result` interfaces,
performs a dependency/pure-physics smoke check, and conditionally runs the
complete snap90--99 pipeline, serialization, closure audit, and figures.

It defaults to `RUN_ANALYSIS = False` so it can be opened without local TNG
snapshots. Set `RUN_ANALYSIS = True`, update `TNG50_BASE_PATH` or `BASE_PATH`,
and set `USE_CONSTANT_COOLING = False` when the eight Henriques cooling tables
are available. The reproducible non-notebook entry point remains
`scripts/run_analysis.py`.
