"""Event, rate, statistics, and workflow assembly for Phase 2.1."""

from .events import classify_halo_events, classify_entering_tracer, classify_exiting_tracer, classify_parent_state
from .pipeline import run_phase21
from .rates import closure_error, rate_from_mass

__all__ = [
    "classify_halo_events", "classify_entering_tracer", "classify_exiting_tracer",
    "classify_parent_state", "closure_error", "rate_from_mass", "run_phase21",
]
