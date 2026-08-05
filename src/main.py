"""Member 5: unified project entry point.

This module provides one command-line interface for the baseline,
improved model, evaluation pipeline, and project tests.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path


MODULE_COMMANDS = {
    "baseline": "src.baseline",
    "improvement": "src.improvement",
    "evaluate": "src.evaluate",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PREDICTIONS = PROJECT_ROOT / "results" / "baseline_predictions.csv"
IMPROVED_PREDICTIONS = (
    PROJECT_ROOT / "results" / "member3" / "v3_full_run" / "improved_predictions.csv"
)


def run_python_module(module_name: str, module_args: Sequence[str]) -> int:
    """Run another project module using the current Python environment."""
    command = [
        sys.executable,
        "-m",
        module_name,
        *module_args,
    ]

    print(f"Running: {' '.join(command)}")

    completed_process = subprocess.run(
        command,
        check=False,
    )

    return completed_process.returncode


def run_tests(test_args: Sequence[str]) -> int:
    """Run the project test suite."""
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *test_args,
    ]

    print(f"Running: {' '.join(command)}")

    completed_process = subprocess.run(
        command,
        check=False,
    )

    return completed_process.returncode


def _normalize_sql(sql: str) -> str:
    """Normalize SQL formatting for the project's diagnostic exact match."""
    normalized = re.sub(r"\s+", " ", sql.strip().lower())
    normalized = re.sub(r"\s*([(),=<>])\s*", r"\1", normalized)
    return normalized[:-1].rstrip() if normalized.endswith(";") else normalized


def _score_predictions(path: Path) -> tuple[int, int, float]:
    """Return correct count, total count, and normalized exact match."""
    if not path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"target_sql", "predicted_sql"}.issubset(rows[0]):
        raise ValueError(
            f"{path} must contain target_sql and predicted_sql columns"
        )

    correct = sum(
        _normalize_sql(row["target_sql"]) == _normalize_sql(row["predicted_sql"])
        for row in rows
    )
    return correct, len(rows), correct / len(rows)


def run_demo() -> int:
    """Reproduce the committed baseline/improved comparison without downloads."""
    baseline = _score_predictions(BASELINE_PREDICTIONS)
    improved = _score_predictions(IMPROVED_PREDICTIONS)
    gain = 100 * (improved[2] - baseline[2])

    print("\nSQL Prompt Generation - offline reproducible demo")
    print("Random seed: 42")
    print(f"{'Model':<20}{'Correct':>12}{'Total':>10}{'Normalized EM':>18}")
    print("-" * 60)
    for name, result in (("T5-small baseline", baseline), ("Improved model", improved)):
        print(f"{name:<20}{result[0]:>12}{result[1]:>10}{result[2]:>17.2%}")
    print(f"Absolute improvement: {gain:.2f} percentage points")
    print("Limitation: exact match is diagnostic; execution accuracy needs databases.")
    return 0


def run_doctor() -> int:
    """Report whether the environment can run the full project pipeline."""
    missing = []
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    for package in ("torch", "transformers", "datasets", "sqlglot", "pytest"):
        available = importlib.util.find_spec(package) is not None
        print(f"{package:<14} {'OK' if available else 'MISSING'}")
        if not available:
            missing.append(package)
    for path in (BASELINE_PREDICTIONS, IMPROVED_PREDICTIONS):
        print(f"{path.relative_to(PROJECT_ROOT)}: {'OK' if path.is_file() else 'MISSING'}")
    if missing:
        print("Install dependencies: python -m pip install -r requirements.txt")
        return 1
    return 0


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the unified command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Unified entry point for the Natural Language to SQL project."
        ),
        epilog=(
            "Examples:\n"
            "  python -m src.main baseline --help\n"
            "  python -m src.main improvement --help\n"
            "  python -m src.main evaluate --help\n"
            "  python -m src.main demo\n"
            "  python -m src.main doctor\n"
            "  python -m src.main test"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "command",
        nargs="?",
        choices=[
            "baseline",
            "improvement",
            "evaluate",
            "demo",
            "doctor",
            "test",
        ],
        help=(
            "Select baseline training, improved-model training, "
            "unified evaluation, or project tests."
        ),
    )

    parser.add_argument(
        "module_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed to the selected module.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected project component."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    forwarded_args = list(args.module_args)

    # Allow commands written with an optional separator:
    # python -m src.main baseline -- --epochs 1
    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]

    if args.command == "test":
        return run_tests(forwarded_args)

    if args.command == "demo":
        return run_demo()

    if args.command == "doctor":
        return run_doctor()

    module_name = MODULE_COMMANDS[args.command]
    return run_python_module(module_name, forwarded_args)


if __name__ == "__main__":
    raise SystemExit(main())
