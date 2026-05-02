"""
Inspection example: visualise the data at every pipeline step.

Runs a full YOLOv8 detection pipeline on a sample image and saves an HTML
report showing the actual output at each step — decoded image, letterboxed
resize, normalised tensor heatmap, raw inference outputs, post-processing
state, and final detections.

Usage:
    python run_inspect.py
    python run_inspect.py --output /tmp/my_report.html --no-browser
    python run_inspect.py --model s --assets-dir /tmp/assets
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
    decode,
    download_if_missing,
    resolve_model_path,
)
from examples.common import visualize_and_store, COCO_CLASSES, build_output_path, visualize_detections_and_store
from run_yolo8_onnx import YOLO8_MODELS, yolo8_inference_pipeline
from ml_pipes import Pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_assets_dir_arg(parser)
    add_model_arg(parser, list(YOLO8_MODELS))
    parser.add_argument("--output", type=Path, default=None, help="Path for the HTML report (default: temp file).")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the report in a browser.")
    args = parser.parse_args()

    assets_dir: Path = args.assets_dir
    model_name, model_url = YOLO8_MODELS[args.model]
    output_path = args.output or build_output_path(assets_dir, COCO_IMAGE_NAME, model_name)
    model_path = resolve_model_path(assets_dir, model_name, model_url, args.model)
    if model_path is None:
        return 1

    image_path = assets_dir / COCO_IMAGE_NAME
    print(f"Downloading image to {image_path} if needed...", file=sys.stderr)
    download_if_missing(COCO_IMAGE_URL, image_path)

    pipeline: Pipeline = decode() + yolo8_inference_pipeline(model_path) + visualize_detections_and_store(output_path, COCO_CLASSES)
    pipeline.validate()

    print("Running inspection...", file=sys.stderr)
    result = pipeline.inspect(image_path)

    print(result)
    result.save(path=args.output, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
