"""
Error presentation example: how the inspector renders pipeline failures.

Three scenarios are demonstrated:

  mid_pipeline   — error in the middle of a flat pipeline; steps that ran
                   before the failure show their outputs normally, the failing
                   step is highlighted in red, and any steps after it are absent.

  first_step     — error on the very first operator; no previous output to
                   carry forward, so only the error card is visible.

  nested         — error inside a Scatter region; Scatter re-raises immediately,
                   so the Scatter span itself is marked as failed. Demonstrates
                   that a region operator error surfaces at the region boundary.

Usage:
    # Open result in the default web browser (default behaviour)
    python run_inspect_errors.py

    # Choose which scenario to run
    python run_inspect_errors.py --scenario first_step
    python run_inspect_errors.py --scenario nested

    # Save HTML report instead of opening the browser
    python run_inspect_errors.py --save report.html

    # Print step labels to stdout only
    python run_inspect_errors.py --print-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ml_pipes import Gather, Inline, Pipeline, PipelineInspector, Scatter


# ---------------------------------------------------------------------------
# Tiny operators — no model, no file IO, fully self-contained
# ---------------------------------------------------------------------------

class MakeArray:
    def __call__(self, value: int) -> np.ndarray:
        return np.full((4, 4, 3), value, dtype=np.uint8)


class _ScaleArray:
    def __init__(self, factor: float) -> None:
        self.factor = factor

    def __call__(self, array: np.ndarray) -> np.ndarray:
        return (array.astype(np.float32) * self.factor).astype(np.uint8)


class Fail:
    """Always raises — stands in for a real operator that can go wrong."""
    def __init__(self, message: str = "intentional failure") -> None:
        self.message = message

    def __call__(self, value: Any) -> Any:
        raise RuntimeError(self.message)


class AddOne:
    def __call__(self, array: np.ndarray) -> np.ndarray:
        return np.clip(array.astype(np.int32) + 1, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def run_mid_pipeline() -> Any:
    pipeline = Pipeline([MakeArray(), _ScaleArray(2.0), Fail(), AddOne()])
    result = pipeline.inspect(100)
    print(result)
    return result


def run_first_step() -> Any:
    pipeline = Pipeline([Fail("bad input"), _ScaleArray(1.5), AddOne()])
    result = pipeline.inspect(42)
    print(result)
    return result


def run_nested() -> Any:
    inner = Pipeline([_ScaleArray(3.0), Fail("inner failure"), AddOne()])
    pipeline = Pipeline([
        Scatter(max_concurrency=2),
        Inline(inner),
        Gather(),
    ])
    arrays = [np.full((4, 4, 3), v, dtype=np.uint8) for v in (50, 60)]
    result = pipeline.inspect(arrays)
    print(result)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_SCENARIOS = {
    "mid_pipeline": run_mid_pipeline,
    "first_step": run_first_step,
    "nested": run_nested,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scenario",
        choices=list(_SCENARIOS),
        default="mid_pipeline",
        help="Which error scenario to run (default: mid_pipeline).",
    )
    parser.add_argument(
        "--orientation",
        choices=["horizontal", "vertical"],
        default="horizontal",
        help="HTML inspection layout orientation (default: horizontal). Use vertical for text-heavy or tabular outputs.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--save", metavar="PATH", type=Path, default=None,
                       help="Save HTML report to PATH instead of opening a browser.")
    group.add_argument("--print-only", action="store_true",
                       help="Print step labels to stdout; do not open any window.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    result = _SCENARIOS[args.scenario]()
    inspector = PipelineInspector()
    if args.save:
        saved = inspector.save_to_html(result, args.save, orientation=args.orientation)
        print(f"Saved to: {saved}", file=sys.stderr)
    elif not args.print_only:
        inspector.show_in_browser(result, orientation=args.orientation)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
