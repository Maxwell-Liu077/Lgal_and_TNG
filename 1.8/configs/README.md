# Configuration

The validated `Phase18Config` dataclass in `src/utils/config.py` is the single
source of scientific defaults. Runtime paths and the workflow variant are set
by `scripts/run_analysis.py` or by the merged notebook; they are deliberately
not duplicated in a second configuration format.
