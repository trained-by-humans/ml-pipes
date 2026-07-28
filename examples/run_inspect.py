"""
Inspect intermediate data for the YOLOv8 example pipelines.

Choose `simple` for the plain image pipeline, `tiled` for sliced inference, or
`error` for a synthetic mid-pipeline failure.

Run from the repo root:
    python examples/run_inspect.py
    python examples/run_inspect.py --pipeline tiled
    python examples/run_inspect.py --pipeline error
    python examples/run_inspect.py --save-html examples/.example_assets/inspect.html
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TypeVar

import numpy as np

from common import (
    ASSETS_DIR,
    COCO_CLASSES,
    COCO_IMAGE_NAME,
    COCO_IMAGE_URL,
    build_output_path,
    decode,
    resolve_input_path,
    resolve_model_path,
    visualize_detections_and_store,
)
from run_yolo8_onnx import BUNDLED_MODEL_PATH, yolo8_inference_pipeline
from run_yolo8_tile import yolo8_tiled_pipeline
from ml_pipes.core import Pipeline
from ml_pipes.inspection import (
    InspectionResult,
    InspectionSerializer,
    PipelineInspector,
)

ValueT = TypeVar("ValueT")


class MakeArray:
    def __call__(self, value: int) -> np.ndarray:
        return np.full((4, 4, 3), value, dtype=np.uint8)


class ScaleArray:
    def __init__(self, factor: float) -> None:
        self.factor = factor

    def __call__(self, array: np.ndarray) -> np.ndarray:
        return (array.astype(np.float32) * self.factor).astype(np.uint8)


class Fail:
    def __init__(self, message: str = "intentional failure") -> None:
        self.message = message

    def __call__(self, value: ValueT) -> ValueT:
        raise RuntimeError(self.message)


class AddOne:
    def __call__(self, array: np.ndarray) -> np.ndarray:
        return np.clip(array.astype(np.int32) + 1, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Inspection logic
# ---------------------------------------------------------------------------

def run_inspection_simple(model_path: Path, image_path: Path, output_path: Path) -> InspectionResult:
    pipeline = (
        decode()
        + yolo8_inference_pipeline(model_path)
        + visualize_detections_and_store(output_path, COCO_CLASSES)
    )
    pipeline.validate()
    pipeline.describe()
    print("Running simple inspection...", file=sys.stderr)
    result = pipeline.inspect(image_path)
    print(result)
    return result


def run_inspection_tiled(model_path: Path, image_path: Path, output_path: Path) -> InspectionResult:
    pipeline = (
        decode()
        + yolo8_tiled_pipeline(model_path)
        + visualize_detections_and_store(output_path, COCO_CLASSES)
    )
    pipeline.validate()
    pipeline.describe()
    print("Running tiled inspection...", file=sys.stderr)
    result = pipeline.inspect(image_path)
    print(result)
    return result


def run_inspection_error() -> InspectionResult:
    pipeline = Pipeline([MakeArray(), ScaleArray(2.0), Fail(), AddOne()])
    pipeline.describe()
    print("Running error inspection...", file=sys.stderr)
    result = pipeline.inspect(100)
    print(result)
    return result


def show_result(result: InspectionResult, args: argparse.Namespace) -> None:
    inspector = PipelineInspector()
    if args.save_html:
        saved = inspector.save_to_html(result, args.save_html)
        print(f"Inspection report saved to: {saved}", file=sys.stderr)
    elif args.plot:
        saved = inspector.save_to_plot(result, args.plot)
        print(f"Plot saved to: {saved}", file=sys.stderr)
    elif args.dump:
        saved = InspectionSerializer().dump(result, args.dump)
        print(f"Inspection result serialized to: {saved}", file=sys.stderr)
    elif args.print_only:
        pass
    else:
        inspector.show_in_browser(result)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_or_run_result(args: argparse.Namespace) -> InspectionResult:
    if args.load:
        result = InspectionResult.load(args.load)
        print(result)
        return result

    if args.pipeline == "error":
        return run_inspection_error()

    image_path = resolve_input_path(args.input, ASSETS_DIR / COCO_IMAGE_NAME, COCO_IMAGE_URL)
    model_path = resolve_model_path(args.model_path, BUNDLED_MODEL_PATH)
    output_path = args.output or build_output_path(ASSETS_DIR, image_path.name, model_path.name)
    if args.pipeline == "tiled":
        return run_inspection_tiled(model_path, image_path, output_path)
    return run_inspection_simple(model_path, image_path, output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", type=Path, default=None, help="Path to a local ONNX model for the simple and tiled pipelines. Defaults to the bundled YOLOv8n model in the assets directory.")
    parser.add_argument("--input", type=Path, default=None, help="Input image path. Defaults to the sample COCO image.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output image path for the simple and tiled pipelines. Defaults to a file under the assets directory.")
    parser.add_argument("--pipeline", choices=["simple", "tiled", "error"], default="simple",
                        help="Which pipeline to inspect (default: simple).")
    parser.add_argument("--load", metavar="PATH", type=Path, default=None,
                        help="Load a previously serialized result instead of running the pipeline.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--save-html", metavar="PATH", type=Path, default=None,
                       help="Save HTML report to PATH instead of opening a browser.")
    group.add_argument("--plot", metavar="PATH", type=Path, default=None,
                       help="Save a matplotlib figure to PATH (e.g. inspect.png).")
    group.add_argument("--dump", metavar="PATH", type=Path, default=None,
                       help="Serialize the InspectionResult to PATH for later analysis.")
    group.add_argument("--print-only", action="store_true",
                       help="Print step labels and shapes to stdout; do not open any window.")
    args = parser.parse_args()

    result = load_or_run_result(args)
    show_result(result, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
