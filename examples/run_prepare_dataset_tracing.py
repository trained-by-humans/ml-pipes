"""
Tracing helper for the prepare-dataset example.

Runs the prepare-dataset pipeline against a dataset path and prints the
InvocationTrace so the execution order and timing are visible. Optionally,
it can also run a second inspection pass and save an HTML report with the
captured intermediate outputs.

Usage:
    python examples/run_prepare_dataset_tracing.py /path/to/dataset/collected

    python examples/run_prepare_dataset_tracing.py \
        /path/to/dataset/collected \
        --pipeline collection \
        --mode both \
        --inspection-path dataset/generated/prepared_trace.html

    python examples/run_prepare_dataset_tracing.py \
        /path/to/dataset/collected \
        --output-path dataset/generated/prepared_traced.json \
        --where 'content.gw == "rakuten"' \
        --label-limits ham=2000,spam=200 \
        --dedupe-key normalized \
        --message-format normalized \
        --shuffle 42 \
        --sort-labels
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
_REEXEC_ENV = "ML_PIPES_PREPARE_DATASET_REEXEC"


def _maybe_reexec_with_repo_venv(exc: ModuleNotFoundError) -> None:
    if os.environ.get(_REEXEC_ENV) == "1":
        return
    if not VENV_PYTHON.exists():
        return
    if Path(sys.executable).resolve() == VENV_PYTHON.resolve():
        return

    # This helper is expected to run inside the repo virtualenv, which has the
    # example dependencies installed. If the caller uses a different Python and
    # misses a dependency such as numpy/cv2, transparently retry with .venv.
    if exc.name not in {"ml_pipes", "numpy", "cv2"}:
        return

    env = dict(os.environ)
    env[_REEXEC_ENV] = "1"
    os.execvpe(
        str(VENV_PYTHON),
        [str(VENV_PYTHON), str(SCRIPT_PATH), *sys.argv[1:]],
        env,
    )


try:
    from ml_pipes import PipelineInspector, PrintCollector
except ModuleNotFoundError as exc:
    _maybe_reexec_with_repo_venv(exc)
    raise SystemExit(
        "Failed to import ml_pipes dependencies. "
        f"Run this script with {VENV_PYTHON} or install the missing module: {exc.name}."
    ) from exc

try:
    from . import run_prepare_dataset as prepare_dataset
except ImportError:
    try:
        import run_prepare_dataset as prepare_dataset
    except ModuleNotFoundError as exc:
        _maybe_reexec_with_repo_venv(exc)
        raise SystemExit(
            "Failed to import the prepare-dataset example dependencies. "
            f"Run this script with {VENV_PYTHON} or install the missing module: {exc.name}."
        ) from exc


DEFAULT_OUTPUT_PATH = Path("dataset/generated/prepared_traced.json")
DEFAULT_INSPECTION_PATH = Path("dataset/generated/prepared_trace.html")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pipeline",
        choices=["stream", "collection"],
        default="stream",
        help="Which prepare-dataset pipeline to run.",
    )
    parser.add_argument(
        "--mode",
        choices=["trace", "inspect", "both"],
        default="trace",
        help="Run only tracing, only inspection, or both.",
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Dataset file or directory to process.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Prepared dataset output path.",
    )
    parser.add_argument(
        "--inspection-path",
        type=Path,
        default=DEFAULT_INSPECTION_PATH,
        help="Where to save the HTML inspection report when inspection runs.",
    )
    parser.add_argument(
        "--orientation",
        choices=["horizontal", "vertical"],
        default="vertical",
        help="Inspection HTML layout orientation.",
    )
    parser.add_argument("--where", default="", help="Optional filter expression.")
    parser.add_argument("--label-limits", default="", help="Optional label limits, e.g. ham=100,spam=100.")
    parser.add_argument(
        "--dedupe-key",
        choices=["cleaned", "normalized"],
        default="normalized",
        help="Text representation used for deduplication.",
    )
    parser.add_argument(
        "--message-format",
        choices=["raw", "cleaned", "normalized"],
        default="raw",
        help="Output message representation written to the prepared dataset.",
    )
    parser.add_argument("--min-length", type=int, default=2, help="Minimum cleaned/dedupe text length.")
    parser.add_argument(
        "--shuffle",
        default="disabled",
        help="Final ordering mode: disabled, auto, or a non-negative integer seed.",
    )
    parser.add_argument(
        "--sort-labels",
        action="store_true",
        help="Sort by label before seeded shuffle. Requires shuffle to be enabled.",
    )
    parser.add_argument(
        "--is-jp",
        dest="is_jp",
        action="store_true",
        default=True,
        help="Use Japanese-aware normalization rules.",
    )
    parser.add_argument(
        "--no-is-jp",
        dest="is_jp",
        action="store_false",
        help="Disable Japanese-aware normalization rules.",
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        default=True,
        help="Overwrite the prepared dataset output if it already exists.",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Fail when the prepared dataset output already exists.",
    )
    parser.add_argument(
        "--capture-shapes",
        dest="capture_shapes",
        action="store_true",
        default=True,
        help="Include output shapes in the printed trace.",
    )
    parser.add_argument(
        "--no-capture-shapes",
        dest="capture_shapes",
        action="store_false",
        help="Disable output shape capture in the printed trace.",
    )
    parser.add_argument(
        "--capture-config",
        dest="capture_config",
        action="store_true",
        default=True,
        help="Include operator configuration in the printed trace.",
    )
    parser.add_argument(
        "--no-capture-config",
        dest="capture_config",
        action="store_false",
        help="Disable operator configuration capture in the printed trace.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    config = {
        "output_path": args.output_path,
        "where": args.where,
        "label_limits": args.label_limits,
        "dedupe_key": args.dedupe_key,
        "message_format": args.message_format,
        "min_length": args.min_length,
        "is_jp": args.is_jp,
        "shuffle": args.shuffle,
        "sort_labels": args.sort_labels,
        "overwrite": args.overwrite,
    }
    if args.pipeline == "collection":
        pipeline = prepare_dataset.build_prepare_dataset_collection_pipeline(**config)
    else:
        pipeline = prepare_dataset.build_prepare_dataset_pipeline(config)

    print(
        f"Running {args.pipeline} prepare-dataset pipeline on {args.input_path}",
        file=sys.stderr,
    )
    print(f"Prepared output will be written to: {args.output_path}", file=sys.stderr)

    if args.mode in {"trace", "both"}:
        print("\n=== Invocation trace ===\n", file=sys.stderr)
        collector = PrintCollector()
        pipeline.set_tracing(
            collector,
            capture_shapes=args.capture_shapes,
            capture_config=args.capture_config,
        )
        try:
            result = pipeline(str(args.input_path))
        finally:
            pipeline.set_tracing(None)
        print(f"Prepared records: {len(result.records)}", file=sys.stderr)

    if args.mode in {"inspect", "both"}:
        print("\n=== Inspection summary ===\n", file=sys.stderr)
        inspection = pipeline.inspect(str(args.input_path))
        print(inspection)
        saved_path = PipelineInspector().save_to_html(
            inspection,
            args.inspection_path,
            orientation=args.orientation,
        )
        print(f"Inspection report saved to: {saved_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
