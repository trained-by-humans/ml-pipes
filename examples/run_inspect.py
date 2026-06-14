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
    COCO_CLASSES,
    COCO_IMAGE_NAME,
    COCO_IMAGE_URL,
    add_assets_dir_arg,
    add_model_arg,
    build_output_path,
    decode,
    download_if_missing,
    resolve_model_path,
    visualize_detections_and_store,
)
from run_yolo8_onnx import YOLO8_MODELS, yolo8_inference_pipeline
from run_yolo8_batch import MODEL_NAME as BATCH_MODEL_NAME, build_pipeline as build_batch_pipeline
from run_yolo8_tile import yolo8_tiled_pipeline
from ml_pipes import (
    Gather,
    Inline,
    InspectionResult,
    InspectionSerializer,
    Pipeline,
    PipelineInspector,
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
    from run_yolo8_batch import _export_dynamic_model

    model_path = assets_dir / BATCH_MODEL_NAME
    if not model_path.exists():
        print(f"Exporting dynamic-batch model → {model_path}", file=sys.stderr)
        _export_dynamic_model(model_path)

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

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_assets_dir_arg(parser)
    add_model_arg(parser, list(YOLO8_MODELS))
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
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    if args.load:
        result = InspectionResult.load(args.load)
        print(result)
        show_result(result, args)
        return 0

    assets_dir: Path = args.assets_dir
    model_name, model_url = YOLO8_MODELS[args.model]
    output_path = build_output_path(assets_dir, COCO_IMAGE_NAME, model_name)
    model_path = resolve_model_path(assets_dir, model_name, model_url, args.model)
    if model_path is None:
        return 1

    image_path = assets_dir / COCO_IMAGE_NAME
    print(f"Downloading image to {image_path} if needed...", file=sys.stderr)
    download_if_missing(COCO_IMAGE_URL, image_path)

    if args.pipeline == "batched":
        result = run_inspection_batched(assets_dir, image_path)
    elif args.pipeline == "tiled":
        result = run_inspection_tiled(model_path, image_path, output_path)
    else:
        result = run_inspection_simple(model_path, image_path, output_path)
    show_result(result, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
