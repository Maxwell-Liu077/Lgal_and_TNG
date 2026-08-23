# Phase 2.1 notebooks

`phase21_end_to_end.ipynb` is the top-level notebook caller. It imports the
public `Phase21Config`, `run_phase21`, and `save_phase21_result` interfaces,
loads the published Henriques cooling tables and runs the complete snap90--99
pipeline, serialization, closure audit, and figures.

The notebook runs only the production cooling calculation: update
`TNG50_BASE_PATH` or `BASE_PATH` and provide all eight Henriques tables under
`data/external/cooling_tables`. `STATE_WORKERS` and `STATE_BACKEND` expose the
conservative parallelism controls used by the optimized state builder. The
reproducible non-notebook entry point remains `scripts/run_analysis.py`.
