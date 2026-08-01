"""Member 5: unified project entry point.

This module provides one command-line interface for the baseline,
improved model, evaluation pipeline, and project tests.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence


MODULE_COMMANDS = {
    "baseline": "src.baseline",
    "improvement": "src.improvement",
    "evaluate": "src.evaluate",
}


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

    module_name = MODULE_COMMANDS[args.command]
    return run_python_module(module_name, forwarded_args)


if __name__ == "__main__":
    raise SystemExit(main())