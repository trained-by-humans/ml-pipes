"""
Inspection example: visualise the data at every pipeline step.

Three pipelines are demonstrated:

  Simple   — single image through the full YOLOv8 detection pipeline.

  Batched  — 8 image paths in, Scatter decodes them concurrently,
             Batch groups them into batches of 4 for inference,
             UnBatch/Gather collect per-image detections.

  Tiled    — single image tiled into overlapping 200×200 patches,
             each tile inferred independently via Scatter, results
             stitched back and deduplicated with NMM.

Usage:
    # Open result in the default web browser (default behavior)
    python run_inspect.py

    # Use the tiled pipeline
    python run_inspect.py --pipeline tiled

    # Use the batched pipeline
    python run_inspect.py --pipeline batched

    # Save HTML report to a file (does not open browser)
    python run_inspect.py --save report.html

    # Save a matplotlib figure to a PNG
    python run_inspect.py --plot inspect.png

    # Save the annotated image for the simple or tiled pipeline
    python run_inspect.py --output annotated.jpg

    # Serialize result to disk for later analysis
    python run_inspect.py --dump result.pkl

    # Load a previously serialized result and open it in the browser
    python run_inspect.py --load result.pkl

    # Print step labels and shapes to stdout only
    python run_inspect.py --print-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
from run_yolo8_onnx import BUNDLED_MODEL_NAME, yolo8_inference_pipeline
from run_yolo8_batch import build_pipeline as build_batch_pipeline
from run_yolo8_tile import yolo8_tiled_pipeline
from ml_pipes.core import (
    Inline,
    Pipeline,
)
from ml_pipes.inspection import (
    InspectionResult,
    InspectionSerializer,
    PipelineInspector,
)
from ml_pipes.standard import (
    Gather,
    Scatter,
)


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


def run_inspection_batched(assets_dir: Path, image_path: Path) -> InspectionResult:
    from run_yolo8_batch import _ensure_yolov8n_dynamic_model

    model_path = _ensure_yolov8n_dynamic_model(assets_dir)
    inference_pipeline = build_batch_pipeline(model_path, batch_size=4, timeout=1.0)
    pipeline = Pipeline([
        Scatter(max_concurrency=4),
        Inline(inference_pipeline),
        Gather(),
    ])
    pipeline.validate()
    pipeline.describe()
    print("Running batched inspection...", file=sys.stderr)
    result = pipeline.inspect([image_path] * 8)
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


def show_result(result: InspectionResult, args: argparse.Namespace) -> None:
    inspector = PipelineInspector()
    if args.save:
        saved = inspector.save_to_html(result, args.save)
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

    assets_dir: Path = args.assets_dir
    image_path = resolve_input_path(args.input, assets_dir, COCO_IMAGE_NAME, COCO_IMAGE_URL)

    if args.pipeline == "batched":
        return run_inspection_batched(assets_dir, image_path)

    model_path = resolve_model_path(args.model_path, assets_dir, BUNDLED_MODEL_NAME)
    output_path = args.output or build_output_path(assets_dir, image_path.name, model_path.name)
    if args.pipeline == "tiled":
        return run_inspection_tiled(model_path, image_path, output_path)
    return run_inspection_simple(model_path, image_path, output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=ASSETS_DIR,
        help="Directory used to cache downloaded models and sample assets.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to a local ONNX model for the simple and tiled pipelines. Defaults to the bundled yolov8n model.",
    )
    parser.add_argument("--input", type=Path, default=None, help="Input image path. Defaults to the sample COCO image.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output image path for the simple and tiled pipelines. Defaults to a file under the assets directory.")
    parser.add_argument("--pipeline", choices=["simple", "batched", "tiled"], default="simple",
                        help="Which pipeline to inspect (default: simple).")
    parser.add_argument("--load", metavar="PATH", type=Path, default=None,
                        help="Load a previously serialized result instead of running the pipeline.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--save", metavar="PATH", type=Path, default=None,
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
