"""Structural and public-API regression checks for the unified project.

The JSON contract records every historical function and class from the two
pre-merge implementations.  Module locations may change during a structural
refactor, so this test resolves each symbol in the new responsibility-based
layout and verifies that all historical call parameters remain accepted.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))


MODULE_LOCATIONS = {
    "phase18.config": ("src.utils.config",),
    "phase18.physics.cooling_function": (
        "src.physics.cooling_function",
    ),
    "phase18.physics.cosmology": ("src.utils.cosmology",),
    "phase18.physics.sam_cooling": ("src.physics.sam_cooling",),
    "phase18.analysis.mass_statistics": (
        "src.analysis.mass_statistics",
    ),
    "phase181.analysis.mass_statistics": (
        "src.analysis.mass_statistics",
    ),
    "phase18.analysis.rates": ("src.analysis.rates",),
    "phase181.analysis.rates": ("src.analysis.rates",),
    "phase18.data_access.mpb": ("src.io.mpb",),
    "phase18.data_access.region_state": ("src.io.region_state",),
    "phase181.data_access.region_state": ("src.io.region_state",),
    "phase18.data_access.sampling": ("src.io.sampling",),
    "phase18.data_access.tracer_batch": ("src.io.tracer_batch",),
    "phase18.data_access.tracer_history": ("src.io.tracer_history",),
    "phase181.data_access.tracer_history": ("src.io.tracer_history",),
    "phase18.visualization.plotting": ("src.plotting",),
    "phase181.visualization.plotting": ("src.plotting",),
    "phase18.workflow.pipeline": (
        "src.utils.arrays",
        "src.io.sample",
        "src.io.state_cache",
        "src.analysis.records",
        "src.analysis.pipeline",
        "src.io.results",
    ),
    "phase181.workflow.pipeline": (
        "src.utils.arrays",
        "src.io.sample",
        "src.io.state_cache",
        "src.analysis.records",
        "src.analysis.inward_delivery_pipeline",
        "src.io.results",
    ),
}


def _resolve_symbol(historical_module: str, symbol_name: str):
    """Resolve one historical symbol in the responsibility-based modules."""

    for module_name in MODULE_LOCATIONS[historical_module]:
        module = importlib.import_module(module_name)
        if hasattr(module, symbol_name):
            return getattr(module, symbol_name)
    raise AssertionError(f"missing historical symbol: {symbol_name}")


def _assert_signature_accepts_historical_parameters(
    function,
    expected_parameters: list[list[str | None]],
) -> None:
    """Check old keyword names while allowing optional merged capabilities."""

    current = inspect.signature(function).parameters
    expected_names = {item[0] for item in expected_parameters}
    for name, kind, _default in expected_parameters:
        assert name in current, f"missing parameter: {function.__name__}.{name}"
        assert current[name].kind.name == kind
    for name, parameter in current.items():
        if name not in expected_names:
            assert parameter.default is not inspect.Parameter.empty, (
                f"new required parameter: {function.__name__}.{name}"
            )


def test_frozen_api_contract() -> None:
    """Confirm that all historical names and call parameters remain usable."""

    contract_path = Path(__file__).with_name("api_contract.json")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    for experiment in ("phase18", "phase181"):
        for module_name, symbols in contract[experiment].items():
            for symbol_name, expected_parameters in symbols.items():
                value = _resolve_symbol(module_name, symbol_name)
                if expected_parameters is None:
                    assert inspect.isclass(value)
                else:
                    assert inspect.isfunction(value)
                    _assert_signature_accepts_historical_parameters(
                        value,
                        expected_parameters,
                    )


def test_implementation_is_fully_documented() -> None:
    """Require English docstrings on every production module and definition."""

    for path in SRC_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert ast.get_docstring(tree), f"missing module docstring: {path}"
        for node in ast.walk(tree):
            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            ):
                assert ast.get_docstring(node), (
                    f"missing definition docstring: {path}:{node.lineno}"
                )


def test_flat_responsibility_based_source_layout() -> None:
    """Check that phase-specific package trees and root facades are gone."""

    for name in ("io", "physics", "analysis", "plotting", "utils"):
        directory = SRC_DIR / name
        assert directory.is_dir(), name
        nested = [
            path
            for path in directory.iterdir()
            if path.is_dir() and path.name != "__pycache__"
        ]
        assert not nested, f"nested source directories under {name}: {nested}"

    assert not (SRC_DIR / "phase18").exists()
    assert not (SRC_DIR / "phase181").exists()
    assert not (PROJECT_ROOT / "1.8.1").exists()
    assert not list(PROJECT_ROOT.glob("*.py"))
    assert {
        "style.py",
        "scaling_relation.py",
        "comparison.py",
        "tracer_history.py",
    }.issubset({path.name for path in (SRC_DIR / "plotting").glob("*.py")})


def test_artifact_layers_are_separate() -> None:
    """Check code, data, notebook, result, and paper boundaries on disk."""

    required = (
        "data/raw",
        "data/external/cooling_tables",
        "data/interim/cache",
        "data/processed",
        "notebooks",
        "scripts",
        "configs",
        "results/figures",
        "results/tables",
        "results/logs",
        "paper",
        "tests",
    )
    for relative in required:
        assert (PROJECT_ROOT / relative).is_dir(), relative

    assert list((PROJECT_ROOT / "data/external/cooling_tables").glob("*.cie"))
    assert list((PROJECT_ROOT / "data/interim/cache").iterdir())
    assert not list((PROJECT_ROOT / "data/processed").rglob("*.png"))
    assert not list((PROJECT_ROOT / "data/processed").rglob("*.pdf"))
    assert not list((PROJECT_ROOT / "results/figures").rglob("*.npz"))
    assert not list((PROJECT_ROOT / "results/figures").rglob("*.csv"))
    assert not list((PROJECT_ROOT / "results/figures").rglob("*.json"))


def test_notebook_is_valid_json() -> None:
    """Parse the merged interactive entry point as notebook JSON."""

    notebook = json.loads(
        (PROJECT_ROOT / "notebooks" / "01_phase18_analysis.ipynb").read_text(
            encoding="utf-8"
        )
    )
    assert notebook["nbformat"] == 4
    assert notebook["cells"]


if __name__ == "__main__":
    test_frozen_api_contract()
    test_implementation_is_fully_documented()
    test_flat_responsibility_based_source_layout()
    test_artifact_layers_are_separate()
    test_notebook_is_valid_json()
    print("Unified Phase 1.8 structure and API contract tests passed")
