"""
Inspection example: visualise the data at every pipeline step.

Runs a full YOLOv8 detection pipeline on a sample image and shows
the actual output at each step — decoded image, letterboxed resize,
normalised tensor heatmap, raw inference outputs, post-processing
state, and final detections.

Usage:
    # Open result in the default web browser (default behaviour)
    python run_inspect.py

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
    COCO_IMAGE_NAME,
    COCO_IMAGE_URL,
    add_assets_dir_arg,
    add_model_arg,
    download_if_missing,
    resolve_model_path,
    decode,
)
from examples.common import COCO_CLASSES, build_output_path, visualize_detections_and_store
from run_yolo8_onnx import YOLO8_MODELS, yolo8_inference_pipeline
from ml_pipes import HtmlRenderer, InspectionResult, InspectionSerializer, Pipeline


# ---------------------------------------------------------------------------
# Inspection logic
# ---------------------------------------------------------------------------

def run_inspection(model_path: Path, image_path: Path, output_path: Path) -> InspectionResult:
    pipeline: Pipeline = (
        decode()
        + yolo8_inference_pipeline(model_path)
        + visualize_detections_and_store(output_path, COCO_CLASSES)
    )
    pipeline.validate()
    print("Running inspection...", file=sys.stderr)
    result = pipeline.inspect(image_path)
    print(result)
    return result


def show_result(result: InspectionResult, args: argparse.Namespace) -> None:
    if args.save:
        saved = HtmlRenderer().save(result, args.save)
        print(f"Inspection report saved to: {saved}", file=sys.stderr)
    elif args.plot:
        fig = result.plot()
        fig.savefig(args.plot, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {args.plot}", file=sys.stderr)
    elif args.dump:
        saved = InspectionSerializer().dump(result, args.dump)
        print(f"Inspection result serialized to: {saved}", file=sys.stderr)
    elif args.print_only:
        pass
    else:
        result.show_in_browser()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_assets_dir_arg(parser)
    add_model_arg(parser, list(YOLO8_MODELS))
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

    result = run_inspection(model_path, image_path, output_path)
    show_result(result, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
